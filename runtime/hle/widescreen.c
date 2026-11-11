// Widescreen: the camera code's 16:9 aspect literal loads are patched (tools/patches.py) to call
// rt_aspect_site(); the renderer sets the wide factor (output aspect / 16:9) each frame.
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#include "../hle.h"
#include "../settings.h"

#define BACKDROP_HASHES 0x71fe31bbf03c3515ull

static float g_wide = 1.0f;  // output aspect divided by 16:9 (1 = normal)
static int g_lit_keep;       // (RE: widescreen_lit_toggle)

// FE3H: every camera reads its aspect from one 16:9 literal (main+0xcb801c, 19 loads); it becomes the window's aspect
// (as Fl4sh9174's 21:9 emulator mod writes 21:9 there)
#define FE3H_ASPECT_LITERAL (MAIN_BASE + 0xcb801cull)
static void find_proj_tick(void);
static void icon_scan_tick(void);
static void vp_watch_tick(uint64_t f);
void widescreen_set_factor(float f) {
    g_wide = f;
    find_proj_tick();
    icon_scan_tick();
    extern uint64_t renderer_present_count(void);
    vp_watch_tick(renderer_present_count());
    static bool rw;
    if (!rw) { extern int plat_protect(uint64_t, uint64_t, int); plat_protect(FE3H_ASPECT_LITERAL & ~0x3FFFull, 0x4000, 3); rw = true; }
    float a = 16.0f / 9.0f * (f > 1.0f && !g_lit_keep ? f : 1.0f);
    memcpy((void *)(uintptr_t)FE3H_ASPECT_LITERAL, &a, 4);
}
// (RE) LARECOMP_LIT_TOGGLE=1: F10 keeps the literal 16:9 (the patched camera reads still widen) or not
int widescreen_lit_toggle(void) {
    g_lit_keep = !g_lit_keep;
    LOG("[aspect] literal kept 16:9: %d\n", g_lit_keep);
    return g_lit_keep;
}

// (RE) LARECOMP_FIND_PROJ=<frame>: find the main camera's projection in writable memory (its x / y scales, logged
// by exec.c LARECOMP_LOG_PROJ) and the 16:9 floats; LARECOMP_FIND_PROJ_WATCH=1 watches the first projection hit
float g_proj_lx, g_proj_ly;
static void find_proj_tick(void) {
    {   // LARECOMP_WATCH_EARLY=<hex lo>,<hex hi>: watch from the first frame (writers of a field set once)
        static int armed;
        const char *e = armed ? NULL : getenv("LARECOMP_WATCH_EARLY");
        armed = 1;
        if (e) {
            extern uint64_t g_watch_lo, g_watch_hi;
            extern void watch_arm(void);
            char *p;
            g_watch_lo = strtoull(e, &p, 16);
            g_watch_hi = *p == ',' ? strtoull(p + 1, NULL, 16) : g_watch_lo + 4;
            watch_arm();
        }
    }
    static long at = -2;
    if (at == -2) at = getenv("LARECOMP_FIND_PROJ") ? atol(getenv("LARECOMP_FIND_PROJ")) : -1;
    extern uint64_t renderer_present_count(void);
    if (at < 0 || (long)renderer_present_count() != at) return;
    at = -1;
    extern int kernel_data_regions(uint64_t *addr, uint64_t *size, int max);
    uint64_t ra[64], rs[64];
    int nr = kernel_data_regions(ra, rs, 64), np = 0, na = 0;
    float lx = g_proj_lx, ly = g_proj_ly, asp = 16.0f / 9.0f;
    static bool na_heap_watched;
    LOG("[findproj] x %.5f y %.5f\n", lx, ly);
    for (int i = 0; i < nr; i++) {
        const float *p = (const float *)(uintptr_t)ra[i];
        uint64_t nw = rs[i] / 4;
        for (uint64_t k = 0; k < nw; k++) {
            float v = p[k];
            if (na < 60 && v == asp) {
                uint64_t a = ra[i] + k * 4;
                static int nth = -1;  // LARECOMP_FIND_ASPECT_WATCH=<n>: the n-th heap hit's block (position .. aspect)
                if (nth < 0) nth = ENV("LARECOMP_FIND_ASPECT_WATCH") ? atoi(getenv("LARECOMP_FIND_ASPECT_WATCH")) : 999;
                if (!na_heap_watched && a >= HEAP_BASE && nth-- == 0) {
                    extern uint64_t g_watch_lo, g_watch_hi;
                    extern void watch_arm(void);
                    g_watch_lo = a - 0x18, g_watch_hi = a + 0x400;  // (and the blocks after it on that page)
                    watch_arm();
                    na_heap_watched = true;
                }
                LOG("[findproj] 16:9 at %llx:", (unsigned long long)a);
                for (int d = -6; d < 10 && k + d < nw; d++) if ((int64_t)k + d >= 0) LOG(" %g", p[k + d]);
                LOG("\n");
                na++;
            }
            if (np < 40 && lx > 0 && fabsf(v - lx) < 2e-4f * lx) {
                bool y = false;  // y scale within the next 4x4 rows (column / row major)
                for (int d = 1; d <= 6 && k + d < nw; d++) y |= fabsf(p[k + d] - ly) < 2e-3f * ly;
                if (!y) continue;
                uint64_t a = ra[i] + k * 4;
                LOG("[findproj] proj at %llx:", (unsigned long long)a);
                for (int d = 0; d < 16 && k + d < nw; d++) LOG(" %g", p[k + d]);
                LOG("\n");
                if (!np && ENV("LARECOMP_FIND_PROJ_WATCH")) {
                    extern uint64_t g_watch_lo, g_watch_hi;
                    extern void watch_arm(void);
                    g_watch_lo = a, g_watch_hi = a + 4;
                    watch_arm();
                }
                np++;
            }
        }
    }
    LOG("[findproj] %d projection hits, %d 16:9 floats\n", np, na);
}

// (RE) LARECOMP_FIND_ICONS=1 with the F9 capture: the capture's small 2D quads (map unit icons among them) are looked
// for in writable memory as float pairs (corner / centre, screen or centre-relative y-up); the best hit is watched
#define ICON_MAX 128
static float icon_q[ICON_MAX][4];
static int icon_n, icon_nc, icon_phase, icon_watch_i;
static uint64_t icon_frame;
static uint64_t icon_vb[16][2];  // (the quads' vertex buffers: not searched)
static int icon_nvb;
void icon_scan_add(float x0, float y0, float x1, float y1, uint64_t vb, uint64_t vbsize) {
    if (!ENV("LARECOMP_FIND_ICONS") || x1 - x0 > 400) return;
    if (icon_phase == 3) icon_phase = 0, icon_nc = 0, icon_nvb = 0;  // (another F9: a new search)
    if (icon_phase || icon_n >= ICON_MAX) return;
    extern uint64_t renderer_present_count(void);
    float *q = icon_q[icon_n++];
    q[0] = x0, q[1] = y0, q[2] = x1, q[3] = y1;
    icon_frame = renderer_present_count();
    bool known = false;
    for (int i = 0; i < icon_nvb; i++) known |= icon_vb[i][0] == vb;
    if (!known && icon_nvb < 16) icon_vb[icon_nvb][0] = vb, icon_vb[icon_nvb++][1] = vbsize;
}
static bool icon_val(float v) { return isnormal(v) && fabsf(v) >= 1.0f && fabsf(v) < 5000.0f; }
// convention matched (-1: none): form * 9 + x anchor * 3 + y anchor; forms screen, centred y-up, centred y-down;
// anchors left / centre / right, top / centre / bottom
static int icon_match(const float *q, float a, float b) {
    float ax[3] = {q[0], (q[0] + q[2]) / 2, q[2]}, ay[3] = {q[1], (q[1] + q[3]) / 2, q[3]};
    for (int fm = 0; fm < 3; fm++)
        for (int xi = 0; xi < 3; xi++) {
            float x = fm ? ax[xi] - 960 : ax[xi];
            if (fabsf(a - x) >= 0.6f) continue;
            for (int yi = 0; yi < 3; yi++) {
                float y = fm == 0 ? ay[yi] : fm == 1 ? 540 - ay[yi] : ay[yi] - 540;
                if (fabsf(b - y) < 0.6f) return fm * 9 + xi * 3 + yi;
            }
        }
    return -1;
}
#define ICON_CAND 256
static struct { uint64_t a; int q, conv; } icon_c[ICON_CAND];
static void icon_scan_tick(void) {
    extern uint64_t renderer_present_count(void);
    uint64_t f = renderer_present_count();
    static long at = -2;  // (test) LARECOMP_FIND_ICONS_AT=<frame>: start without F9
    if (at == -2) at = ENV("LARECOMP_FIND_ICONS_AT") ? atol(getenv("LARECOMP_FIND_ICONS_AT")) : -1;
    if (at >= 0 && (long)f == at) icon_n = 1, icon_frame = f - 6, icon_phase = 0;
    static long dat = -2;  // (test) LARECOMP_DIFF_AT=<frame>: F10 at that frame
    if (dat == -2) dat = ENV("LARECOMP_DIFF_AT") ? atol(getenv("LARECOMP_DIFF_AT")) : -1;
    if (dat >= 0 && (long)f == dat) { void widescreen_diff_key(void); widescreen_diff_key(); }
    if (!icon_n || f < icon_frame + 6) return;  // (after the capture's frames)
    if (icon_phase == 0 && ENV("LARECOMP_FIND_ICONS")[0] == 'p') {  // =proj: the map camera's matrices; readers watched
        extern int kernel_data_regions(uint64_t *addr, uint64_t *size, int max);
        uint64_t ra[64], rs[64];
        int nr = kernel_data_regions(ra, rs, 64);
        float lx = g_proj_lx, ly = g_proj_ly;
        LOG("[icons] projection x %.5f y %.5f\n", lx, ly);
        for (int i = 0; i < nr && lx > 0; i++) {
            if (ra[i] < HEAP_BASE) continue;
            const float *p = (const float *)(uintptr_t)ra[i];
            uint64_t nw = rs[i] / 4;
            for (uint64_t k = 0; k + 8 < nw && icon_nc < ICON_CAND; k++) {
                if (fabsf(p[k] - lx) > 2e-4f * lx || p[k + 1] != 0 || p[k + 2] != 0) continue;  // (a projection: zeros after x)
                bool y = false;
                for (int d = 1; d <= 6; d++) y |= fabsf(p[k + d] - ly) < 2e-3f * ly;
                if (!y) continue;
                LOG("[icons] cand %d: matrix at %llx:", icon_nc, (unsigned long long)(ra[i] + k * 4));
                for (int d = 0; d < 16 && k + d < nw; d++) LOG(" %g", p[k + d]);
                LOG("\n");
                icon_c[icon_nc].a = ra[i] + k * 4, icon_c[icon_nc].q = 0, icon_c[icon_nc].conv = -1;
                icon_nc++;
            }
        }
        icon_phase = 2, icon_watch_i = 0, icon_frame = f;
        LOG("[icons] %d matrices; watching each for 20 frames\n", icon_nc);
    }
    if (icon_phase == 0) {  // scan: map-area, icon-sized quads only; real-valued pairs
        extern int kernel_data_regions(uint64_t *addr, uint64_t *size, int max);
        uint64_t ra[64], rs[64];
        int nr = kernel_data_regions(ra, rs, 64);
        for (int i = 0; i < nr && icon_nc < ICON_CAND; i++) {
            if (ra[i] < HEAP_BASE) continue;
            const float *p = (const float *)(uintptr_t)ra[i];
            uint64_t nw = rs[i] / 4;
            for (uint64_t k = 0; k + 4 < nw && icon_nc < ICON_CAND; k++) {
                if (!icon_val(p[k]) || !icon_val(p[k + 1]) || isnan(p[k + 3])) continue;  // (NaN: the UI vertex layout)
                uint64_t ad = ra[i] + k * 4;
                bool in_vb = false;
                for (int v = 0; v < icon_nvb; v++) in_vb |= ad >= icon_vb[v][0] && ad < icon_vb[v][0] + icon_vb[v][1];
                if (in_vb) continue;
                for (int j = 0; j < icon_n; j++) {
                    const float *q = icon_q[j];
                    float cx = (q[0] + q[2]) / 2, cy = (q[1] + q[3]) / 2, w = q[2] - q[0];
                    if (!(cx > 450 && cx < 1500 && cy > 120 && cy < 820 && w > 15 && w < 220)) continue;
                    int m = icon_match(q, p[k], p[k + 1]);
                    if (m < 0) continue;
                    icon_c[icon_nc].a = ra[i] + k * 4, icon_c[icon_nc].q = j, icon_c[icon_nc].conv = m;
                    icon_nc++;
                    break;
                }
            }
        }
        LOG("[icons] %d candidates from %d quads\n", icon_nc, icon_n);
        icon_phase = 1;
        icon_frame = f;
        return;
    }
    if (icon_phase == 1) {  // still there a few frames later (camera still): log them, keep one per 0x40 block
        int n = 0;
        for (int i = 0; i < icon_nc; i++) {
            const float *p = (const float *)(uintptr_t)icon_c[i].a;
            if (icon_match(icon_q[icon_c[i].q], p[0], p[1]) != icon_c[i].conv) continue;
            bool dup = false;
            for (int j = 0; j < n; j++) dup |= (icon_c[j].a >> 6) == (icon_c[i].a >> 6);
            if (dup) continue;
            const float *q = icon_q[icon_c[i].q];
            LOG("[icons] cand %d: quad (%.0f,%.0f)-(%.0f,%.0f) conv %d at %llx:", n, q[0], q[1], q[2], q[3], icon_c[i].conv,
                (unsigned long long)icon_c[i].a);
            for (int d = -8; d < 12; d++) LOG(" %g", p[d]);
            LOG("\n");
            icon_c[n++] = icon_c[i];
        }
        icon_nc = n;
        icon_phase = 2;
        icon_watch_i = 0;
        icon_frame = f;
        LOG("[icons] %d stable candidates; watching each for 20 frames\n", n);
    }
    if (icon_phase == 2 && f >= icon_frame) {  // each stable candidate watched in turn: its writers' call chains
        if (icon_watch_i >= icon_nc || icon_watch_i >= 48) { LOG("[icons] done\n"); icon_phase = 3; icon_n = 0; return; }
        extern uint64_t g_watch_lo, g_watch_hi;
        extern void watch_arm(void);
        g_watch_lo = icon_c[icon_watch_i].a, g_watch_hi = g_watch_lo + (icon_c[icon_watch_i].conv < 0 ? 64 : 8);
        LOG("[icons] watch cand %d at %llx\n", icon_watch_i, (unsigned long long)g_watch_lo);
        watch_arm();
        icon_watch_i++;
        icon_frame = f + 60 - 6;  // (60 frames each: the tick waits icon_frame + 6)
    }
}

// (RE) LARECOMP_DIFF_ICONS=1: F10 in the wide view snapshots the heap; switch to 16:9 (F1 menu, camera still), F10
// again compares: floats whose x (centred, or screen-relative to 960) grew by the wide factor with y unchanged are
// screen positions placed by the widened camera (the map icons). Each is then watched for its writer.
#define DIFF_MAX 128
static struct { uint64_t a, n; float *s; } diff_r[DIFF_MAX];
static int diff_nr;
static float diff_g;
static void diff_free(void) {
    for (int i = 0; i < diff_nr; i++) free(diff_r[i].s);
    diff_nr = 0;
}
float g_proj_vp[16];  // (the main camera's last view-projection: exec.c LARECOMP_LOG_PROJ)
static uint64_t g_vp_watch_stop;
static void vp_watch_tick(uint64_t f) {
    if (!g_vp_watch_stop || f < g_vp_watch_stop) return;
    g_vp_watch_stop = 0;
    extern void watch_multi(int n, const uint64_t *lo, const uint64_t *hi);
    watch_multi(0, NULL, NULL);
    LOG("[vp] done\n");
}
void widescreen_diff_key(void) {
    extern int kernel_data_regions(uint64_t *addr, uint64_t *size, int max);
    uint64_t ra[64], rs[64];
    int nr = kernel_data_regions(ra, rs, 64);
    if (ENV("LARECOMP_DIFF_ICONS")[0] == 'v') {  // =vp: copies of the main view-projection; each watched for readers
        float vp[16];
        memcpy(vp, g_proj_vp, 64);
        icon_nc = 0;
        for (int i = 0; i < nr; i++) {
            if (ra[i] < HEAP_BASE || vp[0] == 0) continue;
            const float *p = (const float *)(uintptr_t)ra[i];
            uint64_t nw = rs[i] / 4;
            for (uint64_t k = 0; k + 16 < nw && icon_nc < ICON_CAND; k++) {
                if (p[k] != vp[0] || p[k + 1] != vp[1] || p[k + 5] != vp[5] || p[k + 6] != vp[6] || p[k + 10] != vp[10]) continue;
                uint64_t ad = ra[i] + k * 4;
                if ((ad & 0xff) == 0 || (ad & 0xff) == 0x10 || (ad & 0xff) == 0x20) continue;  // (GPU constant-buffer copies)
                LOG("[vp] cand %d at %llx\n", icon_nc, (unsigned long long)ad);
                icon_c[icon_nc].a = ad, icon_c[icon_nc].q = 0, icon_c[icon_nc++].conv = -1;
            }
        }
        LOG("[vp] %d copies of %g %g %g %g; all watched for 900 frames (pan the camera)\n", icon_nc, vp[0], vp[1], vp[5], vp[6]);
        uint64_t lo[96], hi[96];
        int n = icon_nc < 96 ? icon_nc : 96;
        for (int j = 0; j < n; j++) lo[j] = icon_c[j].a, hi[j] = icon_c[j].a + 64;
        extern void watch_multi(int n, const uint64_t *lo, const uint64_t *hi);
        extern uint64_t renderer_present_count(void);
        watch_multi(n, lo, hi);
        g_vp_watch_stop = renderer_present_count() + 900;
        return;
    }
    if (!diff_nr) {  // snapshot
        uint64_t tot = 0;
        int fails = 0;
        for (int i = 0; i < nr; i++) {  // (64 MB pieces: one huge allocation can fail)
            if (ra[i] < HEAP_BASE) continue;
            for (uint64_t o = 0; o < rs[i] && diff_nr < DIFF_MAX && tot < (6ull << 30); o += 64ull << 20) {
                uint64_t n = rs[i] - o < (64ull << 20) ? rs[i] - o : 64ull << 20;
                float *s = malloc(n);
                if (!s) { fails++; continue; }
                memcpy(s, (const void *)(uintptr_t)(ra[i] + o), n);
                diff_r[diff_nr].a = ra[i] + o, diff_r[diff_nr].n = n / 4, diff_r[diff_nr++].s = s;
                tot += n;
            }
        }
        if (fails) LOG("[diff] %d pieces not allocated\n", fails);
        diff_g = g_wide;
        LOG("[diff] snapshot: %d regions, %llu MB, wide factor %.4f\n", diff_nr, (unsigned long long)(tot >> 20), diff_g);
        return;
    }
    float r = diff_g / fmaxf(g_wide, 1.0f);  // (expected growth of x)
    LOG("[diff] compare: wide factor %.4f -> %.4f, x growth %.4f\n", diff_g, g_wide, r);
    int nm = 0;
    icon_nc = 0;
    for (int i = 0; i < diff_nr; i++) {
        const float *s = diff_r[i].s, *p = (const float *)(uintptr_t)diff_r[i].a;
        for (uint64_t k = 0; k + 1 < diff_r[i].n; k++) {
            float a = s[k], b = p[k], ya = s[k + 1], yb = p[k + 1];
            if (a == b || !isnormal(a) || !isnormal(b) || !isnormal(ya) || fabsf(ya) < 1 || fabsf(ya - yb) > 0.03f * fabsf(ya) + 0.05f) continue;
            int conv = -1;
            if (fabsf(a) >= 20 && fabsf(a) < 4000 && fabsf(b / a - r) < 0.01f * r) conv = 0;            // centred
            else if (fabsf(a - 960) >= 20 && fabsf(a) < 4000 && fabsf((b - 960) / (a - 960) - r) < 0.01f * r) conv = 1;  // screen
            if (conv < 0) continue;
            uint64_t ad = diff_r[i].a + k * 4;
            if (nm++ < 200) {
                LOG("[diff] %llx conv %d: x %.2f -> %.2f y %.2f |", (unsigned long long)ad, conv, a, b, yb);
                for (int d = -4; d < 8; d++) if ((int64_t)k + d >= 0 && k + d < diff_r[i].n) LOG(" %g", p[k + d]);
                LOG("\n");
            }
            bool vtx = (k >= 1 && isnan(p[k - 1])) || (k + 3 < diff_r[i].n && isnan(p[k + 3]));  // (UI vertex buffers)
            bool dup = vtx;
            for (int j = 0; j < icon_nc; j++) dup |= (icon_c[j].a >> 8) == (ad >> 8);
            if (!dup && icon_nc < ICON_CAND) icon_c[icon_nc].a = ad, icon_c[icon_nc].q = 0, icon_c[icon_nc++].conv = conv;
        }
    }
    LOG("[diff] %d matches, %d blocks; all watched for 600 frames (pan the camera)\n", nm, icon_nc);
    for (int j = 0; j < icon_nc; j++) LOG("[diff] watch list %d: %llx\n", j, (unsigned long long)icon_c[j].a);
    diff_free();
    uint64_t lo[96], hi[96];
    int n = icon_nc < 96 ? icon_nc : 96;
    for (int j = 0; j < n; j++) lo[j] = icon_c[j].a, hi[j] = icon_c[j].a + 8;
    extern void watch_multi(int n, const uint64_t *lo, const uint64_t *hi);
    extern uint64_t renderer_present_count(void);
    watch_multi(n, lo, hi);
    g_vp_watch_stop = renderer_present_count() + 600;
}

// A camera's aspect as its projection / frustum code reads it (tools/patches.py FE3H_CAM_ASPECT_SITES): 16:9 cameras
// take the window's shape; other shapes (shadow maps, square portrait renders) stay
float rt_cam_aspect(float a) {
    if (g_wide > 1.005f && fabsf(a - 16.0f / 9.0f) < 0.01f) return a * g_wide;
    return a;
}

// Per site (tools/patches.py FE3H_ASPECT_SITES). (RE) LARECOMP_ASPECT_BISECT=1: F10 un-widens one site at a time
// (shown in the window title) - which camera places the map's unit icons?
#define FE3H_ASPECT_NSITES 21
static int g_aspect_skip = -1;
float rt_cam_aspect_site(float a, int id) {
    if (id == g_aspect_skip) return fabsf(a - 16.0f / 9.0f * g_wide) < 0.01f ? 16.0f / 9.0f : a;
    return rt_cam_aspect(a);
}
int widescreen_aspect_site_next(void) {
    g_aspect_skip = g_aspect_skip + 1 >= FE3H_ASPECT_NSITES ? -1 : g_aspect_skip + 1;
    LOG("[aspect] un-widened site %d\n", g_aspect_skip);
    return g_aspect_skip;
}

// The two calls to main+0xa32db0 the same mod removes for wide views (tools/patches.py FE3H_WIDE_SKIP_SITES)
int rt_fe3h_wide_skip(void) { return g_wide > 1.005f && !ENV("LARECOMP_KEEP_A32DB0"); }
float widescreen_factor(void) { return g_wide; }

// Effects drawn as a mesh around the camera, sized to cover a 16:9 view: with the widened camera their outer edge
// shows (Mysterious Woods fog ring: a hard-edged oval at 32:9, a stray cloud piece at the far left). Their draws
// get a viewport widened by the widescreen factor (centred), so they cover the window as they cover 16:9.
// Identified by fragment shader hash (both backends hash the shader code the same way).
float widescreen_effect_stretch(uint64_t fs_hash) {
    static const uint64_t k[] = {
        0xffad3b2d795231a7ull,  // Mysterious Woods fog ring (Field 06C...; found 2026-10-09 by skipping shaders)
        0xd66b33b13b2eac29ull,  // Mysterious Woods haze layer
        0x1372d4fe34dd4bb4ull,  // Mysterious Woods light-shaft boxes (their sides: vertical streaks at the far left)
    };
    if (g_wide <= 1.0f) return 1.0f;
    for (size_t i = 0; i < sizeof k / sizeof k[0]; i++)
        if (k[i] == fs_hash) return g_wide;
    // dungeon darkness around the camera: a hard-edged dark oval once a room shows wider than 16:9 (boxed rooms
    // keep the original vignette)
    extern bool g_room_wide;
    if (fs_hash == 0xc42e882c08c5e5ebull && g_room_wide) return g_wide;
    return 1.0f;
}

// Off in rooms that should keep 16:9 framing (interiors, dungeons) -- see widescreen_update.
static bool g_allowed = true;
static bool g_indoor;
bool widescreen_allowed(void) { return g_allowed && g_settings.aspect == ASPECT_UNLOCKED && !ENV("LARECOMP_NO_WIDESCREEN"); }
void widescreen_set_allowed(bool on) { g_allowed = on; }

// Interiors and dungeons use cameras that never load the patched overworld aspect (site 0). Count
// site-0 loads per presented frame; frames without any are framed 16:9 (pillar/letterboxed).
static volatile uint64_t g_site0_frame = ~0ull;
static int g_indoor_streak, g_outdoor_streak;

RT_ON_RESET(widescreen_reset) { g_indoor = false; g_indoor_streak = g_outdoor_streak = 0; }

void widescreen_frame_end(uint64_t frame) {
    bool saw = g_site0_frame == frame;
    if (saw) { g_outdoor_streak++; g_indoor_streak = 0; }
    else { g_indoor_streak++; g_outdoor_streak = 0; }
    bool was = g_indoor;
    if (!g_indoor && g_indoor_streak >= 3) g_indoor = true;
    if (g_indoor && g_outdoor_streak >= 1) g_indoor = false;
    if (was != g_indoor && ENV("LARECOMP_LOG_INDOOR"))
        LOG("[widescreen] %s at frame %llu\n", g_indoor ? "interior (16:9 framing)" : "overworld (wide)", (unsigned long long)frame);
}

float rt_aspect_site(int site) {
    if (site == 0) { extern uint64_t renderer_present_count(void); g_site0_frame = renderer_present_count(); }
    static long mask = -2;
    if (mask == -2) mask = getenv("LARECOMP_ASPECT_SITES") ? strtol(getenv("LARECOMP_ASPECT_SITES"), NULL, 0) : -1;
    float base = 16.0f / 9.0f;
    if (mask != -1 && !(mask & (1L << site))) return base;
    return base * g_wide;  // the output's aspect (wider or narrower than 16:9)
}

// Culling aspect: the game's own value (camera field), widened like the camera.
float rt_cull_aspect(float orig) {
    static float last = -1;
    if (orig != last && ENV("LARECOMP_LOG_CULL")) { LOG("[widescreen] culling aspect field %f\n", orig); last = orig; }
    return rt_aspect_site(6) * (orig > 0.1f && orig < 10.0f ? orig / (16.0f / 9.0f) : 1.0f);
}

// Vertical field of view: unchanged for wide outputs; for tall ones (factor < 1) widened so the
// horizontal view stays what the game shows at 16:9.
float rt_fovy(float orig) {
    static float last = -1;
    if (orig != last && ENV("LARECOMP_LOG_CULL")) { LOG("[widescreen] fovy field %f\n", orig); last = orig; }
    if (g_wide >= 1.0f || !(orig > 0)) return orig;
    bool degrees = orig > 3.2f;  // radians otherwise
    float half = (degrees ? orig * (float)M_PI / 180.0f : orig) * 0.5f;
    float wide = 2.0f * atanf(tanf(half) / g_wide);
    return degrees ? wide * 180.0f / (float)M_PI : wide;
}

// Interiors: the indoor shadow setup runs while an interior is rendered (tools/patches.py).
static uint64_t g_indoor_frame;
float rt_indoor_aspect(void) {
    extern uint64_t renderer_present_count(void);
    uint64_t f = renderer_present_count();
    if (ENV("LARECOMP_LOG_INDOOR") && f / 60 != g_indoor_frame / 60) LOG("[widescreen] indoor at frame %llu\n", (unsigned long long)f);
    g_indoor_frame = f;
    return 16.0f / 9.0f;
}

void romfs_file_read_hook(const char *path) { (void)path; }

// Interior backdrop (16:9 quad covering the overworld behind rooms), identified by shader code hash.
static uint64_t g_backdrop_hash = 0;  // set below once known
static uint64_t g_backdrop_frame = ~0ull;
void widescreen_backdrop_seen(uint64_t hash) {
    static const uint64_t known[] = {BACKDROP_HASHES};
    for (unsigned i = 0; i < sizeof known / sizeof known[0]; i++)
        if (known[i] && hash == known[i]) { extern uint64_t renderer_present_count(void); g_backdrop_frame = renderer_present_count(); }
    (void)g_backdrop_hash;
}
// Interiors and dungeons only render their 16:9 centre. Besides the backdrop detection, the game's current level
// decides: in play and not on the overworld ("Field") -> boxed. That also covers fades and loading frames, where
// no backdrop is drawn and the sides would show stale pixels.
static bool level_is_interior(void) {
    extern int game_in_play(void);
    if (!game_in_play()) return false;
    uint64_t ps = LD64(MAIN_BASE + 0x1CC1368);  // PlayerStatus*: current level name at +0
    if (!ps) return false;
    const char *lv = (const char *)(uintptr_t)ps;
    return lv[0] && strcmp(lv, "Field") != 0;
}

bool widescreen_frame_boxed(uint64_t frame) {
    (void)level_is_interior;
    if (1) return false;  // FE3H: the whole frame is wide (the room logic below reads Link's Awakening memory)
    static bool last;
    static uint64_t interior_until;  // keep boxing briefly after leaving: the first (dark) frames still show stale sides
    bool in = level_is_interior();
    if (in) interior_until = frame + 8;
    // title / file select (not in play): 16:9 screens; the title's logo glow is drawn too large in the wide image
    // (a blurred copy in the left bar, user 2026-10-09)
    extern int game_in_play(void);
    bool b = g_backdrop_frame == frame || in || frame < interior_until || !game_in_play();
    if (b != last && ENV("LARECOMP_LOG_INDOOR")) LOG("[widescreen] boxed %d at frame %llu\n", b, (unsigned long long)frame);
    last = b;
    return b;
}

// Rooms wider than one screen (multi-cell dungeon rooms, side-view passages) show what the wide camera sees of the
// room itself: the image is cut at the room's outer walls (the camera still scrolls and stops as at 16:9, so near
// an end wall that side is black) instead of at the 16:9 box. Single-screen rooms stay boxed.
// The current camera zone (the level's zone record: id, AABB min at +0x10, max at +0x20, camera params at +0x30)
// is reached through [[sdk+0xcf9e38]+0x680] (found 2026-10-09 by pointer scans, LARECOMP_FIND_FLOATS_PATH);
// the camera's view-projection comes from the scene UBO (vertex slot 5: view, projection, view-projection).
static float g_vp[16];  // rows
bool g_room_wide;  // the last presented frame showed a room wider than 16:9
static uint64_t g_vp_frame = ~0ull;
void widescreen_scene_ubo(const float *u, uint64_t size) {
    extern uint64_t renderer_present_count(void);
    uint64_t f = renderer_present_count();
    if (size != 320 || g_vp_frame == f) return;  // (the frame's first scene draw)
    if (u[12] != 0 || u[13] != 0 || u[14] != 0 || u[15] != 1 || u[30] != -1 || u[31] != 0) return;  // perspective
    memcpy(g_vp, u + 32, sizeof g_vp);
    g_vp_frame = f;
}

static bool heap_ptr(uint64_t p) { return p >= HEAP_BASE && p + 0x40 <= HEAP_BASE + HEAP_MAX && !(p & 7); }

// Current camera zone's box; false when there is none (title, overworld, loading).
static bool current_zone(float lo[3], float hi[3], uint32_t *id) {
    uint64_t o = LD64(SDK_BASE + 0xcf9e38);
    if (!heap_ptr(o)) return false;
    uint64_t z = LD64(o + 0x680);
    if (!heap_ptr(z)) return false;
    const float *f = (const float *)(uintptr_t)(z + 0x10);
    for (int i = 0; i < 3; i++) {
        lo[i] = f[i], hi[i] = f[4 + i];
        if (!(hi[i] > lo[i]) || hi[i] - lo[i] > 2000) return false;
    }
    *id = *(const uint32_t *)(uintptr_t)z;
    return true;
}

// Horizontal crop (uv of the wide image): ROOM_CROP for a room wider than one screen (u0..u1), ROOM_BOX for a
// single-screen room (16:9 box), ROOM_FULL when the zone doesn't match the camera (no box: the game's camera clamps
// for the wide view, so a 16:9 box around a stale zone's centre cut Link out - user 2026-10-09), ROOM_KEEP when
// there's no recent scene camera (keep the last choice; after half a second without one, e.g. menus: box).
enum { ROOM_KEEP, ROOM_BOX, ROOM_CROP, ROOM_FULL };
static int room_crop(uint64_t frame, double shape, float *u0, float *u1) {
    float lo[3], hi[3];
    uint32_t id;
    if (ENV("LARECOMP_NO_ROOM_WIDE")) return ROOM_BOX;
    if (g_vp_frame > frame || g_vp_frame + 4 < frame) return g_vp_frame + 30 < frame ? ROOM_BOX : ROOM_KEEP;  // (menus: no scene)
    if (!current_zone(lo, hi, &id)) return ROOM_FULL;
    const float *r0 = g_vp, *r1 = g_vp + 4, *r3 = g_vp + 12;
    bool side = hi[2] - lo[2] < 6.0f;  // side-view zones are thin in z (the camera looks along z)
    // the point under the screen centre (ndc 0, 0): on the floor (y 0) for top-down rooms, on the zone's centre
    // plane for side views
    float c[3];
    if (side) {
        c[2] = (lo[2] + hi[2]) / 2;
        float a = r0[0], b = r0[1], e = -(r0[2] * c[2] + r0[3]), d = r1[0], f = r1[1], g = -(r1[2] * c[2] + r1[3]);
        float det = a * f - b * d;
        if (fabsf(det) < 1e-6f) return ROOM_KEEP;
        c[0] = (e * f - b * g) / det, c[1] = (a * g - e * d) / det;
    } else {
        c[1] = 0;
        float a = r0[0], b = r0[2], e = -r0[3], d = r1[0], f = r1[2], g = -r1[3];
        float det = a * f - b * d;
        if (fabsf(det) < 1e-6f) return ROOM_KEEP;
        c[0] = (e * f - b * g) / det, c[2] = (a * g - e * d) / det;
    }
    int r;
    bool wide = hi[0] - lo[0] >= 16.0f;  // (cells are 15)
    if (c[0] < lo[0] - 1 || c[0] > hi[0] + 1) r = ROOM_FULL;  // camera outside the zone: stale zone
    else if (!wide) r = fabsf(c[0] - (lo[0] + hi[0]) / 2) < 1.0f ? ROOM_BOX : ROOM_FULL;  // centred or stale
    else {
        float ndc[2];
        r = ROOM_CROP;
        for (int e = 0; e < 2; e++) {
            float x = e ? hi[0] : lo[0];
            float w = r3[0] * x + r3[1] * c[1] + r3[2] * c[2] + r3[3];
            if (!(w > 0.01f)) return ROOM_KEEP;
            ndc[e] = (r0[0] * x + r0[1] * c[1] + r0[2] * c[2] + r0[3]) / w;
        }
        // never less than the 16:9 view the game frames
        float b0 = (float)((1.0 - 1.0 / shape) / 2), b1 = 1.0f - b0;
        *u0 = fminf(b0, fmaxf(0.0f, (ndc[0] + 1) / 2));
        *u1 = fmaxf(b1, fminf(1.0f, (ndc[1] + 1) / 2));
    }
    static uint32_t last_id = ~0u;
    static int last_r = -1;
    if ((id != last_id || r != last_r) && ENV("LARECOMP_LOG_INDOOR"))
        LOG("[widescreen] zone %u x %.1f..%.1f z %.1f..%.1f%s, camera x %.1f: %s at frame %llu\n", id, lo[0], hi[0], lo[2],
            hi[2], side ? " (side view)" : "", c[0], r == ROOM_CROP ? "crop" : r == ROOM_BOX ? "box" : "full (stale zone)",
            (unsigned long long)frame);
    last_id = id, last_r = r;
    return r;
}

// Presented part of the image (crop: uv origin and size) and its place in the window: interiors and dungeons show
// their 16:9 centre (taller outputs: the 16:9 band), rooms wider than one screen the part inside the room.
void widescreen_present_crop(uint64_t frame, double shape, float crop[4], double *vx, double *vy, double *vw, double *vh) {
    static uint64_t last;  // (last wide boxed frame: room changes slide the edges instead of jumping)
    static float cu0, cu1;
    crop[0] = crop[1] = 0, crop[2] = crop[3] = 1;
    g_room_wide = false;
    if (shape == 1.0 || !widescreen_frame_boxed(frame)) return;
    if (shape < 1.0) {
        crop[1] = (float)((1.0 - shape) / 2); crop[3] = (float)shape;
        double nh = *vh * shape; *vy += (*vh - nh) / 2; *vh = nh;
        return;
    }
    static int mode = ROOM_BOX;
    float u0 = 0, u1 = 1;
    static float k0, k1;  // (ROOM_KEEP: the last target)
    extern int game_in_play(void);
    int m = game_in_play() ? room_crop(frame, shape, &u0, &u1) : ROOM_BOX;
    if (m == ROOM_KEEP) m = mode, u0 = k0, u1 = k1;
    if (m == ROOM_BOX) u0 = (float)((1.0 - 1.0 / shape) / 2), u1 = 1.0f - u0;
    if (m == ROOM_FULL) u0 = 0, u1 = 1;
    mode = m, k0 = u0, k1 = u1;
    g_room_wide = m != ROOM_BOX;
    if (frame <= last + 2 && frame > last) {
        const float step = 0.03f;  // of the image width per frame
        cu0 += fmaxf(-step, fminf(step, u0 - cu0));
        cu1 += fmaxf(-step, fminf(step, u1 - cu1));
    } else {
        cu0 = u0, cu1 = u1;
    }
    last = frame;
    crop[0] = cu0, crop[2] = cu1 - cu0;
    *vx += *vw * cu0, *vw *= cu1 - cu0;
}

// Current level name for logs ("-" when not in play).
const char *game_level_name(void) {
    extern int game_in_play(void);
    uint64_t ps = LD64(MAIN_BASE + 0x1CC1368);
    return game_in_play() && ps && *(const char *)(uintptr_t)ps ? (const char *)(uintptr_t)ps : "-";
}
