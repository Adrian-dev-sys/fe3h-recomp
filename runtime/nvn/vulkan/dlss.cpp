// NVIDIA DLAA through Streamline (third_party/streamline, MIT; DLLs beside the .exe: sl.interposer / sl.common /
// sl.dlss / nvngx_dlss). Loaded at run time (no import library), manual hooking: our own Vulkan instance and device,
// told to Streamline with slSetVulkanInfo. exec.c feeds it the game image, depth, zero motion vectors (camera motion
// from depth + clipToPrevClip) and the jitter it applied.
#ifdef _WIN32
#include <windows.h>
#endif
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <string.h>

#include "../../../third_party/streamline/sl.h"
#include "../../../third_party/streamline/sl_dlss.h"
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
} sl_;
static bool g_inited, g_ready;
static PFN_vkQueuePresentKHR g_present;
static sl::FeatureRequirements g_req;

static void sl_log(sl::LogType type, const char *msg) {
    if (type != sl::LogType::eInfo) DLOG("[dlss] sl: %s", msg);
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
#define GET(f, name) sl_.f = (decltype(sl_.f))GetProcAddress(m, name)
    GET(init, "slInit"); GET(shutdown, "slShutdown"); GET(supported, "slIsFeatureSupported");
    GET(reqs, "slGetFeatureRequirements"); GET(vkinfo, "slSetVulkanInfo"); GET(token, "slGetNewFrameToken");
    GET(consts, "slSetConstants"); GET(tag, "slSetTagForFrame"); GET(eval, "slEvaluateFeature");
    GET(featfn, "slGetFeatureFunction");
#undef GET
    if (!getenv("LARECOMP_DLSS_NO_PROXY_PRESENT")) g_present = (PFN_vkQueuePresentKHR)GetProcAddress(m, "vkQueuePresentKHR");
    if (!sl_.init || !sl_.vkinfo || !sl_.eval || !sl_.tag || !sl_.featfn) { DLOG("[dlss] Streamline entry points missing\n"); return false; }
    static const sl::Feature feats[] = {sl::kFeatureDLSS};
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
    p.numFeaturesToLoad = 1;
    p.engine = sl::EngineType::eCustom;
    p.engineVersion = "1.0.0";
    p.projectId = "6f3e2a1c-9b7d-4e85-a4c2-3d1f0b8e7a65";  // (NGX: a custom engine identifies itself by project GUID)
    p.renderAPI = sl::RenderAPI::eVulkan;
    sl::Result r = sl_.init(p, sl::kSDKVersion);
    if (r != sl::Result::eOk) { DLOG("[dlss] slInit failed: %d\n", (int)r); return false; }
    r = sl_.reqs(sl::kFeatureDLSS, g_req);
    if (r != sl::Result::eOk) { DLOG("[dlss] no DLSS requirements: %d\n", (int)r); return false; }
    DLOG("[dlss] Streamline up: %u instance / %u device extensions, %u+%u features, %u compute / %u graphics queues\n",
         g_req.vkNumInstanceExtensions, g_req.vkNumDeviceExtensions, g_req.vkNumFeatures12, g_req.vkNumFeatures13,
         g_req.vkNumComputeQueuesRequired, g_req.vkNumGraphicsQueuesRequired);
    for (uint32_t i = 0; i < g_req.vkNumDeviceExtensions; i++) DLOG("[dlss]   device ext %s\n", g_req.vkDeviceExtensions[i]);
    for (uint32_t i = 0; i < g_req.vkNumInstanceExtensions; i++) DLOG("[dlss]   instance ext %s\n", g_req.vkInstanceExtensions[i]);
    for (uint32_t i = 0; i < g_req.vkNumFeatures12; i++) DLOG("[dlss]   1.2 feature %s\n", g_req.vkFeatures12[i]);
    for (uint32_t i = 0; i < g_req.vkNumFeatures13; i++) DLOG("[dlss]   1.3 feature %s\n", g_req.vkFeatures13[i]);
    g_inited = true;
    return true;
#else
    return false;
#endif
}

extern "C" uint32_t dlss_instance_exts(const char *const **names) {
    if (!g_inited) return 0;
    *names = g_req.vkInstanceExtensions;
    return g_req.vkNumInstanceExtensions;
}

extern "C" uint32_t dlss_device_exts(const char *const **names) {
    if (!g_inited) return UINT32_MAX;
    *names = g_req.vkDeviceExtensions;
    return g_req.vkNumDeviceExtensions;
}

// Turns on the 1.2 / 1.3 features Streamline asks for in the device's feature structs.
extern "C" void dlss_merge_features(VkPhysicalDeviceVulkan12Features *f12, VkPhysicalDeviceVulkan13Features *f13) {
    if (!g_inited) return;
    VkPhysicalDeviceVulkan12Features w12 = sl::getVkPhysicalDeviceVulkan12Features(g_req.vkNumFeatures12, g_req.vkFeatures12);
    VkPhysicalDeviceVulkan13Features w13 = sl::getVkPhysicalDeviceVulkan13Features(g_req.vkNumFeatures13, g_req.vkFeatures13);
    VkBool32 *d = &f12->samplerMirrorClampToEdge, *s = &w12.samplerMirrorClampToEdge;
    for (; d <= &f12->subgroupBroadcastDynamicId; d++, s++) *d |= *s;
    d = &f13->robustImageAccess; s = &w13.robustImageAccess;
    for (; d <= &f13->maintenance4; d++, s++) *d |= *s;
}

extern "C" bool dlss_device_created(VkInstance inst, VkPhysicalDevice phys, VkDevice dev, uint32_t qfam) {
    if (!g_inited) return false;
    sl::VulkanInfo vi;
    vi.instance = inst;
    vi.physicalDevice = phys;
    vi.device = dev;
    vi.graphicsQueueFamily = qfam;
    vi.graphicsQueueIndex = 1;  // (any SL queues would follow ours)
    vi.computeQueueFamily = qfam;
    vi.computeQueueIndex = 1;
    sl::Result r = sl_.vkinfo(vi);
    if (r != sl::Result::eOk) { DLOG("[dlss] slSetVulkanInfo failed: %d\n", (int)r); return false; }
    sl::AdapterInfo ai;
    ai.vkPhysicalDevice = phys;
    r = sl_.supported(sl::kFeatureDLSS, ai);
    if (r != sl::Result::eOk) { DLOG("[dlss] DLSS not supported on this GPU / driver: %d\n", (int)r); return false; }
    void *fn = nullptr;
    if (sl_.featfn(sl::kFeatureDLSS, "slDLSSSetOptions", fn) != sl::Result::eOk || !fn) { DLOG("[dlss] no slDLSSSetOptions\n"); return false; }
    sl_.dlss_opts = (PFun_slDLSSSetOptions *)fn;
    g_ready = true;
    DLOG("[dlss] DLAA available\n");
    return true;
}

extern "C" bool dlss_available(void) { return g_ready; }

extern "C" PFN_vkQueuePresentKHR dlss_present_proxy(void) { return g_ready ? g_present : nullptr; }

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

extern "C" bool dlss_evaluate(VkCommandBuffer cb, const DlssImage *color, const DlssImage *depth, const DlssImage *mvec,
                              const DlssImage *out, const DlssConstants *c) {
    if (!g_ready) return false;
    static uint32_t frame;
    sl::FrameToken *tok = nullptr;
    uint32_t fi = frame++;
    if (sl_.token(tok, &fi) != sl::Result::eOk || !tok) return false;
    sl::ViewportHandle vp(0);

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

    sl::Resource rc = res(color), rd = res(depth), rm = res(mvec), ro = res(out);
    sl::ResourceTag tags[] = {
        sl::ResourceTag(&rc, sl::kBufferTypeScalingInputColor, sl::ResourceLifecycle::eValidUntilEvaluate),
        sl::ResourceTag(&rd, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilEvaluate),
        sl::ResourceTag(&rm, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilEvaluate),
        sl::ResourceTag(&ro, sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eValidUntilEvaluate),
    };
    if (sl_.tag(*tok, vp, tags, 4, (sl::CommandBuffer *)cb) != sl::Result::eOk) { DLOG("[dlss] slSetTagForFrame failed\n"); return false; }
    const sl::BaseStructure *inputs[] = {&vp};
    sl::Result r = sl_.eval(sl::kFeatureDLSS, *tok, inputs, 1, (sl::CommandBuffer *)cb);
    if (r != sl::Result::eOk) {
        static int n;
        if (n++ < 5) DLOG("[dlss] slEvaluateFeature failed: %d\n", (int)r);
        return false;
    }
    return true;
}
