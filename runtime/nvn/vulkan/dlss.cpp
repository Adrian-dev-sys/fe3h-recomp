// NVIDIA DLAA and DLSS Frame Generation through Streamline (third_party/streamline, MIT; DLLs beside the .exe:
// sl.interposer / sl.common / sl.dlss / nvngx_dlss, and for frame generation sl.dlss_g / nvngx_dlssg / sl.reflex /
// sl.pcl). Loaded at run time (no import library), manual hooking: our own Vulkan instance and device, told to
// Streamline with slSetVulkanInfo; the swapchain goes through the interposer (frame generation presents from it).
// exec.c feeds it the game image, depth, zero motion vectors (camera motion from depth + clipToPrevClip) and the
// jitter it applied; device.c marks each frame's present (Reflex / PCL markers, one frame token per present).
#ifdef _WIN32
#include <windows.h>
#endif
#include <vulkan/vulkan.h>
#ifdef _WIN32
#include <vulkan/vulkan_win32.h>
#endif
#include <stdio.h>
#include <string.h>

#include "../../../third_party/streamline/sl.h"
#include "../../../third_party/streamline/sl_dlss.h"
#include "../../../third_party/streamline/sl_dlss_g.h"
#include "../../../third_party/streamline/sl_reflex.h"
#include "../../../third_party/streamline/sl_pcl.h"
#include "../../../third_party/streamline/sl_helpers_vk.h"

#include "dlss.h"

#define DLOG(...) fprintf(stderr, __VA_ARGS__)

static struct {
    PFun_slInit *init;
    PFun_slShutdown *shutdown;
    PFun_slIsFeatureSupported *supported;
    PFun_slGetFeatureRequirements *reqs;
    PFun_slSetVulkanInfo *vkinfo;
    PFun_slGetNewFrameToken *token;
    PFun_slSetConstants *consts;
    PFun_slSetTagForFrame *tag;
    PFun_slEvaluateFeature *eval;
    PFun_slGetFeatureFunction *featfn;
    PFun_slDLSSSetOptions *dlss_opts;
    PFun_slDLSSGSetOptions *fg_opts;
    PFun_slDLSSGGetState *fg_state;
    PFun_slReflexSetOptions *reflex_opts;
    PFun_slPCLSetMarker *marker;
} sl_;
static void *g_mod;  // (sl.interposer)
static bool g_inited, g_ready, g_fg_loaded, g_fg_ready, g_dev_done;
static PFN_vkQueuePresentKHR g_present;
static sl::FeatureRequirements g_req, g_req_fg;
static uint32_t g_fg_gfx, g_fg_comp;  // (Streamline's own queues, after ours: dlss_extra_queues)
static sl::FrameToken *g_tok;  // this frame's token (one per present)
static int g_fg_cur = -1, g_fg_idle;  // frames generated as configured (-1: never); presents without a scene
static uint32_t g_frame;

static void sl_log(sl::LogType type, const char *msg) {
    if (type != sl::LogType::eInfo) DLOG("[dlss] sl: %s", msg);
}

// The union of two features' Vulkan extension / feature name lists.
static const char *g_inst[32], *g_dev[64], *g_f12[64], *g_f13[32];
static uint32_t g_ninst, g_ndev, g_nf12, g_nf13;
static void merge(const char **dst, uint32_t *n, uint32_t cap, const char *const *src, uint32_t ns) {
    for (uint32_t i = 0; i < ns; i++) {
        bool dup = false;
        for (uint32_t j = 0; j < *n && !dup; j++) dup = !strcmp(dst[j], src[i]);
        if (!dup && *n < cap) dst[(*n)++] = src[i];
    }
}

extern "C" bool dlss_preinit(void) {
#ifdef _WIN32
    if (getenv("LARECOMP_NO_DLSS")) return false;
    wchar_t dir[MAX_PATH];
    DWORD n = GetModuleFileNameW(NULL, dir, MAX_PATH);
    while (n && dir[n - 1] != L'\\' && dir[n - 1] != L'/') n--;
    dir[n] = 0;
    wchar_t path[MAX_PATH + 32];
    swprintf(path, MAX_PATH + 32, L"%lssl.interposer.dll", dir);
    HMODULE m = LoadLibraryW(path);
    if (!m) { DLOG("[dlss] sl.interposer.dll not found: DLAA unavailable\n"); return false; }
    g_mod = m;
#define GET(f, name) sl_.f = (decltype(sl_.f))GetProcAddress(m, name)
    GET(init, "slInit"); GET(shutdown, "slShutdown"); GET(supported, "slIsFeatureSupported");
    GET(reqs, "slGetFeatureRequirements"); GET(vkinfo, "slSetVulkanInfo"); GET(token, "slGetNewFrameToken");
    GET(consts, "slSetConstants"); GET(tag, "slSetTagForFrame"); GET(eval, "slEvaluateFeature");
    GET(featfn, "slGetFeatureFunction");
#undef GET
    if (!getenv("LARECOMP_DLSS_NO_PROXY_PRESENT")) g_present = (PFN_vkQueuePresentKHR)GetProcAddress(m, "vkQueuePresentKHR");
    if (!sl_.init || !sl_.vkinfo || !sl_.eval || !sl_.tag || !sl_.featfn) { DLOG("[dlss] Streamline entry points missing\n"); return false; }
    swprintf(path, MAX_PATH + 32, L"%lssl.dlss_g.dll", dir);
    g_fg_loaded = !getenv("LARECOMP_NO_DLSS_G") && GetFileAttributesW(path) != INVALID_FILE_ATTRIBUTES;
    static const sl::Feature feats[] = {sl::kFeatureDLSS, sl::kFeatureReflex, sl::kFeaturePCL, sl::kFeatureDLSS_G};
    static wchar_t plugdir[MAX_PATH];
    wcscpy(plugdir, dir);
    static const wchar_t *paths[] = {plugdir};
    sl::Preferences p;
    p.showConsole = false;
    p.logLevel = getenv("LARECOMP_DLSS_LOG") ? sl::LogLevel::eVerbose : sl::LogLevel::eDefault;
    p.logMessageCallback = sl_log;
    p.pathsToPlugins = paths;
    p.numPathsToPlugins = 1;
    p.flags = sl::PreferenceFlags::eDisableCLStateTracking | sl::PreferenceFlags::eUseManualHooking |
              sl::PreferenceFlags::eUseFrameBasedResourceTagging | sl::PreferenceFlags::eDisableDebugText;
    p.featuresToLoad = feats;
    p.numFeaturesToLoad = g_fg_loaded ? 4 : 1;
    p.engine = sl::EngineType::eCustom;
    p.engineVersion = "1.0.0";
    p.projectId = "6f3e2a1c-9b7d-4e85-a4c2-3d1f0b8e7a65";  // (NGX: a custom engine identifies itself by project GUID)
    p.renderAPI = sl::RenderAPI::eVulkan;
    sl::Result r = sl_.init(p, sl::kSDKVersion);
    if (r != sl::Result::eOk) { DLOG("[dlss] slInit failed: %d\n", (int)r); return false; }
    r = sl_.reqs(sl::kFeatureDLSS, g_req);
    if (r != sl::Result::eOk) { DLOG("[dlss] no DLSS requirements: %d\n", (int)r); return false; }
    merge(g_inst, &g_ninst, 32, g_req.vkInstanceExtensions, g_req.vkNumInstanceExtensions);
    merge(g_dev, &g_ndev, 64, g_req.vkDeviceExtensions, g_req.vkNumDeviceExtensions);
    merge(g_f12, &g_nf12, 64, g_req.vkFeatures12, g_req.vkNumFeatures12);
    merge(g_f13, &g_nf13, 32, g_req.vkFeatures13, g_req.vkNumFeatures13);
    if (g_fg_loaded && sl_.reqs(sl::kFeatureDLSS_G, g_req_fg) == sl::Result::eOk) {
        merge(g_inst, &g_ninst, 32, g_req_fg.vkInstanceExtensions, g_req_fg.vkNumInstanceExtensions);
        merge(g_dev, &g_ndev, 64, g_req_fg.vkDeviceExtensions, g_req_fg.vkNumDeviceExtensions);
        merge(g_f12, &g_nf12, 64, g_req_fg.vkFeatures12, g_req_fg.vkNumFeatures12);
        merge(g_f13, &g_nf13, 32, g_req_fg.vkFeatures13, g_req_fg.vkNumFeatures13);
        g_fg_gfx = g_req_fg.vkNumGraphicsQueuesRequired, g_fg_comp = g_req_fg.vkNumComputeQueuesRequired;
        DLOG("[dlss] frame generation wants %u graphics / %u compute / %u optical flow queues\n", g_fg_gfx, g_fg_comp,
             g_req_fg.vkNumOpticalFlowQueuesRequired);
    } else if (g_fg_loaded) {
        DLOG("[dlss] no frame generation requirements\n");
        g_fg_loaded = false;
    }
    DLOG("[dlss] Streamline up: %u instance / %u device extensions, %u+%u features\n", g_ninst, g_ndev, g_nf12, g_nf13);
    for (uint32_t i = 0; i < g_ndev; i++) DLOG("[dlss]   device ext %s\n", g_dev[i]);
    for (uint32_t i = 0; i < g_ninst; i++) DLOG("[dlss]   instance ext %s\n", g_inst[i]);
    g_inited = true;
    return true;
#else
    return false;
#endif
}

extern "C" uint32_t dlss_instance_exts(const char *const **names) {
    if (!g_inited) return 0;
    *names = g_inst;
    return g_ninst;
}

extern "C" uint32_t dlss_device_exts(const char *const **names) {
    if (!g_inited) return UINT32_MAX;
    *names = g_dev;
    return g_ndev;
}

extern "C" uint32_t dlss_extra_queues(void) { return g_inited && g_fg_loaded ? g_fg_gfx + g_fg_comp : 0; }
extern "C" uint32_t dlss_optical_flow_queues(void) { return g_inited && g_fg_loaded ? g_req_fg.vkNumOpticalFlowQueuesRequired : 0; }

// Turns on the 1.2 / 1.3 features Streamline asks for in the device's feature structs.
extern "C" void dlss_merge_features(VkPhysicalDeviceVulkan12Features *f12, VkPhysicalDeviceVulkan13Features *f13) {
    if (!g_inited) return;
    VkPhysicalDeviceVulkan12Features w12 = sl::getVkPhysicalDeviceVulkan12Features(g_nf12, g_f12);
    VkPhysicalDeviceVulkan13Features w13 = sl::getVkPhysicalDeviceVulkan13Features(g_nf13, g_f13);
    VkBool32 *d = &f12->samplerMirrorClampToEdge, *s = &w12.samplerMirrorClampToEdge;
    for (; d <= &f12->subgroupBroadcastDynamicId; d++, s++) *d |= *s;
    d = &f13->robustImageAccess; s = &w13.robustImageAccess;
    for (; d <= &f13->maintenance4; d++, s++) *d |= *s;
}

template <class T> static bool feat_fn(sl::Feature f, const char *name, T *&out) {
    void *fn = nullptr;
    if (sl_.featfn(f, name, fn) != sl::Result::eOk || !fn) { DLOG("[dlss] no %s\n", name); return false; }
    out = (T *)fn;
    return true;
}

extern "C" bool dlss_device_created(VkInstance inst, VkPhysicalDevice phys, VkDevice dev, uint32_t qfam, uint32_t nqueues,
                                   uint32_t offam) {
    if (!g_inited) return false;
    g_dev_done = true;
    sl::VulkanInfo vi;
    vi.instance = inst;
    vi.physicalDevice = phys;
    vi.device = dev;
    vi.graphicsQueueFamily = qfam;
    vi.graphicsQueueIndex = 1;  // (Streamline's queues follow ours: graphics, then compute)
    vi.computeQueueFamily = qfam;
    vi.computeQueueIndex = 1 + g_fg_gfx;
    if (offam != UINT32_MAX) vi.opticalFlowQueueFamily = offam, vi.opticalFlowQueueIndex = 0, vi.useNativeOpticalFlowMode = true;
    if (nqueues < 1 + dlss_extra_queues()) g_fg_loaded = false;  // (the family has too few queues)
    sl::Result r = sl_.vkinfo(vi);
    if (r != sl::Result::eOk) { DLOG("[dlss] slSetVulkanInfo failed: %d\n", (int)r); return false; }
    sl::AdapterInfo ai;
    ai.vkPhysicalDevice = phys;
    r = sl_.supported(sl::kFeatureDLSS, ai);
    if (r != sl::Result::eOk) { DLOG("[dlss] DLSS not supported on this GPU / driver: %d\n", (int)r); return false; }
    if (!feat_fn(sl::kFeatureDLSS, "slDLSSSetOptions", sl_.dlss_opts)) return false;
    g_ready = true;
    DLOG("[dlss] DLAA available\n");
    if (g_fg_loaded) {
        r = sl_.supported(sl::kFeatureDLSS_G, ai);
        if (r != sl::Result::eOk) DLOG("[dlss] frame generation not supported on this GPU / driver: %d\n", (int)r);
        else if (feat_fn(sl::kFeatureDLSS_G, "slDLSSGSetOptions", sl_.fg_opts) &&
                 feat_fn(sl::kFeatureDLSS_G, "slDLSSGGetState", sl_.fg_state) &&
                 feat_fn(sl::kFeatureReflex, "slReflexSetOptions", sl_.reflex_opts) &&
                 feat_fn(sl::kFeaturePCL, "slPCLSetMarker", sl_.marker)) {
            sl::ReflexOptions ro;
            ro.mode = sl::ReflexMode::eLowLatency;  // (frame generation needs Reflex on)
            r = sl_.reflex_opts(ro);
            if (r != sl::Result::eOk) DLOG("[dlss] slReflexSetOptions failed: %d\n", (int)r);
            else g_fg_ready = true, DLOG("[dlss] frame generation available\n");
        }
    }
    return true;
}

extern "C" bool dlss_available(void) { return g_ready; }
extern "C" bool dlss_fg_available(void) { return g_fg_ready; }

extern "C" PFN_vkQueuePresentKHR dlss_present_proxy(void) { return g_ready ? g_present : nullptr; }

// The interposer's surface / swapchain functions with frame generation (it presents the generated frames).
extern "C" PFN_vkVoidFunction dlss_vk_proc(const char *name) {
#ifdef _WIN32
    return g_fg_ready ? (PFN_vkVoidFunction)GetProcAddress((HMODULE)g_mod, name) : nullptr;
#else
    return nullptr;
#endif
}

// The interposer's instance-level functions (instance, surface) whenever frame generation was loaded.
extern "C" PFN_vkVoidFunction dlss_instance_proc(const char *name) {
#ifdef _WIN32
    return g_inited && g_fg_loaded ? (PFN_vkVoidFunction)GetProcAddress((HMODULE)g_mod, name) : nullptr;
#else
    return nullptr;
#endif
}

// The window's surface made through the interposer (frame generation maps swapchains to windows by it).
extern "C" VkSurfaceKHR dlss_create_surface(VkInstance inst, void *hwnd) {
#ifdef _WIN32
    PFN_vkCreateWin32SurfaceKHR cs = (PFN_vkCreateWin32SurfaceKHR)dlss_instance_proc("vkCreateWin32SurfaceKHR");
    if (!cs || !hwnd) return VK_NULL_HANDLE;
    VkWin32SurfaceCreateInfoKHR ci = {VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    ci.hinstance = GetModuleHandleW(NULL);
    ci.hwnd = (HWND)hwnd;
    VkSurfaceKHR s = VK_NULL_HANDLE;
    if (cs(inst, &ci, nullptr, &s) != VK_SUCCESS) { DLOG("[dlss] surface through Streamline failed\n"); return VK_NULL_HANDLE; }
    return s;
#else
    return VK_NULL_HANDLE;
#endif
}

static sl::FrameToken *frame_token(void) {
    if (!g_tok) {
        uint32_t fi = g_frame++;
        if (sl_.token(g_tok, &fi) != sl::Result::eOk) g_tok = nullptr;
    }
    return g_tok;
}

static void mark(sl::PCLMarker m) {
    sl::FrameToken *t = frame_token();
    if (t && sl_.marker) sl_.marker(m, *t);
}

// device.c around each present: the frame's latency markers, then a new token for the next frame.
extern "C" void dlss_present_begin(void) {
    if (!g_fg_ready) return;
    // no 3D scene for a while (title, loading, full-screen menus): off, nothing to generate from
    if (g_fg_cur > 0 && ++g_fg_idle > 10) {
        sl::DLSSGOptions o;
        o.mode = sl::DLSSGMode::eOff;
        o.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
        if (sl_.fg_opts(sl::ViewportHandle(0), o) == sl::Result::eOk) g_fg_cur = 0, DLOG("[dlss] frame generation paused (no scene)\n");
    }
    mark(sl::PCLMarker::eSimulationStart);
    mark(sl::PCLMarker::eSimulationEnd);
    mark(sl::PCLMarker::eRenderSubmitStart);
    mark(sl::PCLMarker::eRenderSubmitEnd);
    mark(sl::PCLMarker::ePresentStart);
}
extern "C" void dlss_present_end(void) {
    if (g_fg_ready) mark(sl::PCLMarker::ePresentEnd);
    g_tok = nullptr;
}

// Frame generation on (frames: generated per real frame, 0 = off) for a window of sw x sh.
static void fg_configure(int frames, uint32_t sw, uint32_t sh, uint32_t mw, uint32_t mh, VkFormat mf, VkFormat df) {
    int &cur = g_fg_cur;
    static uint32_t csw, csh, cmw, cmh;
    if (frames == cur && sw == csw && sh == csh && mw == cmw && mh == cmh) return;
    sl::DLSSGOptions o;
    o.mode = frames > 0 ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
    o.numFramesToGenerate = frames > 0 ? (uint32_t)frames : 1;
    o.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
    o.colorWidth = sw, o.colorHeight = sh;
    o.mvecDepthWidth = mw, o.mvecDepthHeight = mh;
    o.mvecBufferFormat = (uint32_t)mf;
    o.depthBufferFormat = (uint32_t)df;
    sl::Result r = sl_.fg_opts(sl::ViewportHandle(0), o);
    if (r != sl::Result::eOk) { DLOG("[dlss] slDLSSGSetOptions failed: %d\n", (int)r); return; }
    cur = frames, csw = sw, csh = sh, cmw = mw, cmh = mh;
    DLOG("[dlss] frame generation %s (%d generated a frame, window %ux%u, depth %ux%u)\n", frames > 0 ? "on" : "off", frames,
         sw, sh, mw, mh);
}

extern "C" void dlss_fg_state_log(void) {
    if (!g_fg_ready) return;
    sl::DLSSGState s;
    if (sl_.fg_state(sl::ViewportHandle(0), s, nullptr) == sl::Result::eOk)
        DLOG("[dlss] frame generation: status %u, presented %u a frame, max %u generated\n", (unsigned)s.status,
             s.numFramesActuallyPresented, s.numFramesToGenerateMax);
}

static sl::Resource res(const DlssImage *im) {
    sl::Resource r(sl::ResourceType::eTex2d, (void *)im->image, nullptr, (void *)im->view, VK_IMAGE_LAYOUT_GENERAL);
    r.width = im->width;
    r.height = im->height;
    r.nativeFormat = (uint32_t)im->format;
    r.mipLevels = 1;
    r.arrayLayers = 1;
    r.flags = 0;
    r.usage = im->usage;
    return r;
}

static void mat(sl::float4x4 &d, const float *m) { memcpy(&d, m, 64); }

extern "C" bool dlss_evaluate(VkCommandBuffer cb, const DlssImage *color, const DlssImage *depth, const DlssImage *mvec, const DlssImage *bias,
                              const DlssImage *out, const DlssConstants *c, int fg_frames, uint32_t win_w, uint32_t win_h) {
    if (!g_ready) return false;
    sl::FrameToken *tok = frame_token();
    if (!tok) return false;
    sl::ViewportHandle vp(0);
    bool fg = g_fg_ready && fg_frames > 0;
    g_fg_idle = 0;
    if (g_fg_ready) fg_configure(fg ? fg_frames : 0, win_w, win_h, depth->width, depth->height, mvec->format, depth->format);

    static uint32_t ow, oh;
    if (ow != out->width || oh != out->height) {
        sl::DLSSOptions o;
        o.mode = sl::DLSSMode::eDLAA;
        o.outputWidth = out->width;
        o.outputHeight = out->height;
        o.colorBuffersHDR = sl::Boolean::eFalse;
        sl::Result r = sl_.dlss_opts(vp, o);
        if (r != sl::Result::eOk) { DLOG("[dlss] slDLSSSetOptions failed: %d\n", (int)r); return false; }
        ow = out->width, oh = out->height;
        DLOG("[dlss] DLAA at %ux%u\n", ow, oh);
    }

    sl::Constants k;
    mat(k.cameraViewToClip, c->view_to_clip);
    mat(k.clipToCameraView, c->clip_to_view);
    mat(k.clipToLensClip, c->identity);
    mat(k.clipToPrevClip, c->clip_to_prev_clip);
    mat(k.prevClipToClip, c->prev_clip_to_clip);
    k.jitterOffset = {c->jitter[0], c->jitter[1]};
    k.mvecScale = {1.0f, 1.0f};
    k.cameraPinholeOffset = {0, 0};
    k.cameraPos = {c->cam_pos[0], c->cam_pos[1], c->cam_pos[2]};
    k.cameraUp = {c->cam_up[0], c->cam_up[1], c->cam_up[2]};
    k.cameraRight = {c->cam_right[0], c->cam_right[1], c->cam_right[2]};
    k.cameraFwd = {c->cam_fwd[0], c->cam_fwd[1], c->cam_fwd[2]};
    k.cameraNear = c->znear;
    k.cameraFar = c->zfar;
    k.cameraFOV = c->fov;
    k.cameraAspectRatio = c->aspect;
    k.depthInverted = c->depth_inverted ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    k.cameraMotionIncluded = sl::Boolean::eFalse;  // (zero vectors: DLSS adds the camera's motion from depth)
    k.motionVectors3D = sl::Boolean::eFalse;
    k.reset = c->reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    k.orthographicProjection = sl::Boolean::eFalse;
    k.motionVectorsDilated = sl::Boolean::eFalse;
    k.motionVectorsJittered = sl::Boolean::eFalse;
    k.motionVectorsInvalidValue = -1e30f;  // (ours are all valid: zero)
    if (sl_.consts(k, *tok, vp) != sl::Result::eOk) { DLOG("[dlss] slSetConstants failed\n"); return false; }

    // frame generation reads depth and motion at present: copied now (the depth is drawn on before then)
    sl::ResourceLifecycle dm = fg ? sl::ResourceLifecycle::eOnlyValidNow : sl::ResourceLifecycle::eValidUntilEvaluate;
    sl::Resource rc = res(color), rd = res(depth), rm = res(mvec), ro = res(out), rb = bias ? res(bias) : sl::Resource{};
    sl::ResourceTag tags[] = {
        sl::ResourceTag(&rc, sl::kBufferTypeScalingInputColor, sl::ResourceLifecycle::eValidUntilEvaluate),
        sl::ResourceTag(&rd, sl::kBufferTypeDepth, dm),
        sl::ResourceTag(&rm, sl::kBufferTypeMotionVectors, dm),
        sl::ResourceTag(&ro, sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eValidUntilEvaluate),
        sl::ResourceTag(&rb, sl::kBufferTypeBiasCurrentColorHint, sl::ResourceLifecycle::eValidUntilEvaluate),
    };
    static int nfmt;
    if (!nfmt++) DLOG("[dlss] tag formats: color %d depth %d mvec %d out %d bias %d\n", (int)color->format, (int)depth->format,
                      (int)mvec->format, (int)out->format, bias ? (int)bias->format : -1);
    if (sl_.tag(*tok, vp, tags, bias ? 5 : 4, (sl::CommandBuffer *)cb) != sl::Result::eOk) { DLOG("[dlss] slSetTagForFrame failed\n"); return false; }
    const sl::BaseStructure *inputs[] = {&vp};
    sl::Result r = sl_.eval(sl::kFeatureDLSS, *tok, inputs, 1, (sl::CommandBuffer *)cb);
    if (r != sl::Result::eOk) {
        static int n;
        if (n++ < 5) DLOG("[dlss] slEvaluateFeature failed: %d\n", (int)r);
        return false;
    }
    return true;
}
