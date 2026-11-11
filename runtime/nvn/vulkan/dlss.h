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
} DlssConstants;

bool dlss_preinit(void);  // before vkCreateInstance
uint32_t dlss_instance_exts(const char *const **names);
uint32_t dlss_device_exts(const char *const **names);
void dlss_merge_features(VkPhysicalDeviceVulkan12Features *f12, VkPhysicalDeviceVulkan13Features *f13);
bool dlss_device_created(VkInstance inst, VkPhysicalDevice phys, VkDevice dev, uint32_t qfam);
bool dlss_available(void);
PFN_vkQueuePresentKHR dlss_present_proxy(void);  // sl.interposer's present (its per-frame upkeep), NULL without DLSS
bool dlss_evaluate(VkCommandBuffer cb, const DlssImage *color, const DlssImage *depth, const DlssImage *mvec,
                   const DlssImage *out, const DlssConstants *c);

#ifdef __cplusplus
}
#endif
