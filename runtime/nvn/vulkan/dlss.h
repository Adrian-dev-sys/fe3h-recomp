// NVIDIA DLAA through Streamline (dlss.cpp).
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    VkImage image;
    VkImageView view;
    VkFormat format;
    uint32_t width, height;
    VkImageUsageFlags usage;
} DlssImage;

// Matrices column-major (GLSL / guest layout), which is Streamline's row-major row-vector layout.
typedef struct {
    float view_to_clip[16], clip_to_view[16], clip_to_prev_clip[16], prev_clip_to_clip[16], identity[16];
    float jitter[2];  // pixels, the sample's offset from the pixel centre
    float cam_pos[3], cam_up[3], cam_right[3], cam_fwd[3];
    float znear, zfar, fov, aspect;
    bool depth_inverted, reset;
    bool camera_motion;  // the motion vectors already hold the camera's motion
} DlssConstants;

bool dlss_preinit(void);  // before vkCreateInstance
uint32_t dlss_instance_exts(const char *const **names);
uint32_t dlss_device_exts(const char *const **names);
void dlss_merge_features(VkPhysicalDeviceVulkan12Features *f12, VkPhysicalDeviceVulkan13Features *f13);
uint32_t dlss_extra_queues(void);  // queues to create after ours in its family (frame generation's own)
uint32_t dlss_optical_flow_queues(void);
// offam: the optical flow queue's family (UINT32_MAX: none)
bool dlss_device_created(VkInstance inst, VkPhysicalDevice phys, VkDevice dev, uint32_t qfam, uint32_t nqueues, uint32_t offam);
PFN_vkVoidFunction dlss_instance_proc(const char *name);  // interposer instance functions (frame generation)
VkSurfaceKHR dlss_create_surface(VkInstance inst, void *hwnd);  // VK_NULL_HANDLE: make it the usual way
bool dlss_available(void);
bool dlss_fg_available(void);  // DLSS Frame Generation (RTX 40 / 50)
PFN_vkQueuePresentKHR dlss_present_proxy(void);  // sl.interposer's present (its per-frame upkeep), NULL without DLSS
PFN_vkVoidFunction dlss_vk_proc(const char *name);  // the interposer's swapchain functions with frame generation
void dlss_present_begin(void);  // around each present (Reflex markers; a new frame token after)
void dlss_present_end(void);
void dlss_fg_state_log(void);
// fg_frames: frames generated per real frame (0 off) into a window of win_w x win_h
// bias: optional mask (1 = take the current frame over history: FE3H's characters, animated without motion vectors)
bool dlss_evaluate(VkCommandBuffer cb, const DlssImage *color, const DlssImage *depth, const DlssImage *mvec, const DlssImage *bias,
                   const DlssImage *out, const DlssConstants *c, int fg_frames, uint32_t win_w, uint32_t win_h);

#ifdef __cplusplus
}
#endif
