// Executes recorded NVN command lists with Vulkan (port of renderer.m; same structure, same behaviour).
#include <SDL3/SDL_cpuinfo.h>
#include <SDL3/SDL_timer.h>
#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "../../kernel.h"
#include "shaders.h"
#include "vk.h"
#include "dlss.h"
#include "../renderer.h"
#include "../../settings.h"

// Why draws / dispatches are skipped (the stats line breaks stat_skipped down; the first few of each are logged).
enum { SK_CS_PROG, SK_CS_DEBUG, SK_NO_PROG, SK_PROG, SK_PROG_MIRROR, SK_FS_LIST, SK_POT, SK_TOPO, SK_PIPE, SK_CURSOR, SK_N };
static const char *const skip_names[SK_N] = {"compute shader", "LARECOMP_SKIP_CS", "no program", "program", "program (mirror)",
                                             "LARECOMP_SKIP_FSHASH", "pot glow", "topology", "pipeline", "item cursor"};
static uint64_t stat_skip_why[SK_N];
static int skip_logged[SK_N];
#define SKIP(why, ...) do { stat_skipped++; stat_skip_why[why]++; \
        if (skip_logged[why] < 5) { skip_logged[why]++; LOG("[skip] %s: ", skip_names[why]); LOG(__VA_ARGS__); } return; } while (0)
static uint64_t stat_draws, stat_skipped, stat_passes, stat_dispatches, stat_wait_ns, stat_flushes, stat_feedback_copies;
static uint64_t stat_upload_bytes;
// LARECOMP_VK_PROF=1: render-thread time per category in the fps log (exec = whole command lists, draw = draw(),
// sets = descriptor sets incl. cbuf copies, vtx = index scan + vertex copies, submit = vkQueueSubmit2)
static int prof = -1;
static uint64_t prof_exec, prof_draw, prof_sets, prof_vtx, prof_submit, prof_pre, prof_key, prof_dyn, prof_tex, prof_up, prof_gpu;
static VkQueryPool prof_qp;
static uint64_t cum_exec, cum_wait, cum_draws, cum_submits;  // (profiling) never reset: hitch log
static uint32_t prof_end_line[4096];  // passes ended, by the exec.c line that ended them (0: submit / present)
uint64_t prof_views_created;  // textures.c vk_image_view cache misses
uint64_t prof_up_ns = ~0ull;  // textures.c upload checks (~0: profiling off)
#define PROF_T0() uint64_t prof_t0_ = prof ? SDL_GetTicksNS() : 0
#define PROF_ADD(v) do { if (prof) v += SDL_GetTicksNS() - prof_t0_; } while (0)
static _Atomic int stat_compiles;
static uint64_t op_counts[CMD_COUNT];
bool g_debug_frame;
static char stats_text[2048];
static _Atomic uint64_t xlat_hits, xlat_misses, xlat_ns;  // (translations: cache/xlat, below)
void vk_collect_dead(void);

// ================================================================== submissions
// A submission = optional upload command buffer + the main one, signalling the timeline with its number.
#define SUBS 8
typedef struct {
    VkCommandBuffer main, upload;
    bool main_begun, upload_begun, ts;  // ts: GPU timestamps written (profiling)
    uint64_t seq;
    VkDescriptorPool pools[16];
    int npools, cur_pool;
} Sub;
static Sub subs[SUBS];
static int cur_sub;
static void ss_forget(void);
static uint64_t submit_seq;  // last submitted
static pthread_mutex_t seq_lock = PTHREAD_MUTEX_INITIALIZER;
static bool in_pass;

uint64_t vk_completed_seq(void) {
    uint64_t v = 0;
    vkGetSemaphoreCounterValue(g_vk.dev, g_vk.timeline, &v);
    return v;
}

// (LARECOMP_VK_PROF) waits on the GPU by cause: the caller names it in wait_why just before waiting
static __thread const char *wait_why;
static struct { const char *why; uint64_t ns, n; } wait_tab[12];
static pthread_mutex_t wait_tab_lock = PTHREAD_MUTEX_INITIALIZER;
static void wait_account(uint64_t ns) {
    const char *why = wait_why ? wait_why : "other";
    wait_why = NULL;
    pthread_mutex_lock(&wait_tab_lock);
    for (int i = 0; i < 12; i++)
        if (!wait_tab[i].why || wait_tab[i].why == why) { wait_tab[i].why = why; wait_tab[i].ns += ns; wait_tab[i].n++; break; }
    pthread_mutex_unlock(&wait_tab_lock);
}

static void cq_resolve_done(uint64_t done);

void vk_wait_seq(uint64_t seq) {
    if (!seq) { wait_why = NULL; return; }
    VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1, .pSemaphores = &g_vk.timeline,
                              .pValues = &seq};
    uint64_t t0 = SDL_GetTicksNS();
    VkResult r = vkWaitSemaphores(g_vk.dev, &wi, 10ull * 1000000000ull);
    stat_wait_ns += SDL_GetTicksNS() - t0;
    cum_wait += SDL_GetTicksNS() - t0;
    wait_account(SDL_GetTicksNS() - t0);
    if (r == VK_TIMEOUT) LOG("[vk] GPU did not finish submission %llu within 10 s (device lost?)\n", (unsigned long long)seq);
    else if (r == VK_SUCCESS) cq_resolve_done(seq);  // (the game reads its timing counters after waiting)
    else {
        extern void vk_devlost_dump(void);
        vk_devlost_dump();
        vk_device_fault_report();
        FATAL("vkWaitSemaphores: %d (device lost)\n", r);
    }
}

uint64_t vk_pending_seq(void) { return submit_seq + 1; }

// ---- the game's GPU timing counters (nvnCommandBufferReportCounter, two per frame)
// The game times its GPU work with these and drops to 30 fps when frames look slow. They used to get the host clock
// at translation time, so our command translation counted as GPU time: busy scenes (outdoors, ~400 draws) looked slow
// and the game locked itself at 30 even on fast GPUs, at any resolution. Now each counter also records a GPU
// timestamp at its place in the command stream; once that submission is done (the game waits on its fence before
// reading - on the Switch the GPU writes them too) the GPU time replaces the provisional host value, in the host
// clock's terms (offset calibrated at each resolve: a timestamp can't be later than the moment its submission is seen
// done). LARECOMP_VK_COUNTER=cpu: host clock only; =frozen: one value per presented frame (test); =legacy: the host
// clock in ns instead of GPU ticks, as before (test: reproduces the locked 30).
#define CQ_MAX 64
typedef struct { VkQueryPool pool; uint32_t n; uint64_t seq; uint64_t *out[CQ_MAX]; } CounterQ;
static CounterQ cq[SUBS];
static pthread_mutex_t cq_lock = PTHREAD_MUTEX_INITIALIZER;
static int64_t cq_offset;  // host ns - GPU ns (the smallest seen)
// Counter timestamps are in the Switch GPU's ticks (614.4 MHz): the game converts the difference to ns itself
// (x 625 / 384). Writing ns made every GPU time look 1.63x as long.
static uint64_t ns_to_gpu_ticks(uint64_t ns) { return (uint64_t)((unsigned __int128)ns * 384 / 625); }
static bool cq_calibrated;
static int counter_mode = -1;  // 0 cpu, 1 frozen, 2 gpu
static uint64_t stat_counters_gpu;

static int vk_counter_mode(void) {
    if (counter_mode < 0) {
        const char *e = getenv("LARECOMP_VK_COUNTER");
        counter_mode = e && !strcmp(e, "cpu") ? 0 : e && !strcmp(e, "frozen") ? 1 : e && !strcmp(e, "legacy") ? 3 : 2;
        if (counter_mode == 2 && !g_vk.props.limits.timestampComputeAndGraphics) counter_mode = 0;
    }
    return counter_mode;
}

// ---- frame pacing: the game's own 30 fps mode when we can't hold 60
// The game advances its world one fixed step per frame in its 60 fps mode, and two steps per frame in its 30 fps mode
// (measured: Link walks twice as far per frame there) - so a 60-mode frame that takes 20 ms plays in slow motion, while
// the 30 mode plays at full speed. The game picks its mode from these counters (GPU time per frame), as on the Switch;
// but our frames can be slow for reasons its GPU time doesn't show (command translation, a slower CPU). So each
// presented frame's busy time is measured (present to present, minus the time the presenting thread was blocked in
// kernel waits and sleeps: idle - the game waits for vsync through its own threads); when
// frames don't fit 60 (12 of the last 60 over 16.2 ms), the counters report a frame of at least PACE_SLOW_NS: the game
// takes its 30 fps mode, at full speed. Back to the real times after 2 s with headroom (busy under 13 ms on average,
// at most 2 of the last 60 frames over). The game itself also tries 60 again now and then.
// LARECOMP_PACE=0: off.
#define PACE_SLOW_NS 20000000ull
extern __thread uint64_t g_blocked_ns;  // kernel.c: this thread's time blocked in waits / sleeps
static bool pace_slow;
static uint64_t stat_pace_over;

void vk_pace_frame(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("LARECOMP_PACE"); on = !e || atoi(e) != 0; }
    static uint64_t last, last_wait;
    static uint8_t over[60];
    static int idx, nover, calm;
    static double avg;  // busy ms, moving average (~20 frames)
    uint64_t now = plat_now_ns(), w = g_blocked_ns;
    uint64_t p = now - last, wt = w - last_wait;
    bool first = !last;
    last = now, last_wait = w;
    if (first || p > 100000000ull) return;  // (loading, a hitch: not a pace)
    double busy = (p > wt ? p - wt : 0) / 1e6;
    avg = avg ? avg + (busy - avg) * 0.05 : busy;
    static int log = -1;  // LARECOMP_LOG_PACE=1: each frame's period, vsync wait and busy time
    if (log < 0) log = getenv("LARECOMP_LOG_PACE") != NULL;
    if (log) LOG("[pace] period %.2f blocked %.2f busy %.2f ms\n", p / 1e6, wt / 1e6, busy);
    int o = busy > 16.2;
    stat_pace_over += o;
    nover += o - over[idx];
    over[idx] = (uint8_t)o;
    idx = (idx + 1) % 60;
    if (!on) return;  // (measured and counted all the same: LARECOMP_LOG_PACE, the stats)
    if (!pace_slow && nover >= 12) {
        pace_slow = true, calm = 0;
        LOG("[pace] frames don't fit 60 fps (%d of the last 60 over 16.2 ms busy): the game's 30 fps mode\n", nover);
    } else if (pace_slow) {
        // headroom: busy well under a 60 fps frame on average, hardly a frame over (single spikes are normal play)
        calm = avg < 13.0 && nover <= 2 ? calm + 1 : 0;
        if (calm >= 120) {
            pace_slow = false, nover = 0;
            memset(over, 0, sizeof over);
            LOG("[pace] 2 s with headroom: back to the game's own choice\n");
        }
    }
}

// a pair's end counter (the 16 bytes after its start) is at least PACE_SLOW_NS after it while pace_slow
static uint64_t pace_end_counter(const uint64_t *start_entry, const uint64_t *entry, uint64_t value) {
    if (!pace_slow || !start_entry || (const uint8_t *)entry - (const uint8_t *)start_entry != 16) return value;
    uint64_t min = start_entry[1] + ns_to_gpu_ticks(PACE_SLOW_NS);
    return value < min ? min : value;
}

static void cq_resolve(int i) {  // (cq_lock held) submission slot i is done: its counters get their GPU times
    CounterQ *q = &cq[i];
    if (!q->n) return;
    uint64_t t[CQ_MAX];
    if (vkGetQueryPoolResults(g_vk.dev, q->pool, 0, q->n, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
        double per = g_vk.props.limits.timestampPeriod;
        int64_t cand = (int64_t)plat_now_ns() - (int64_t)((double)t[q->n - 1] * per);
        if (!cq_calibrated || cand < cq_offset) cq_offset = cand, cq_calibrated = true;
        static int log = -1;  // LARECOMP_LOG_COUNTER=1: the host (provisional) and GPU times of each counter
        if (log < 0) log = getenv("LARECOMP_LOG_COUNTER") != NULL;
        for (uint32_t k = 0; k < q->n; k++) {
            uint64_t gpu = ns_to_gpu_ticks((uint64_t)((int64_t)((double)t[k] * per) + cq_offset));
            if (log) LOG("[counter] seq %llu #%u at %llx host %.3f gpu %.3f ms\n", (unsigned long long)q->seq, k,
                         (unsigned long long)(uintptr_t)q->out[k], q->out[k][1] * 625.0 / 384 / 1e6, gpu * 625.0 / 384 / 1e6);
            q->out[k][1] = pace_end_counter(k ? q->out[k - 1] : NULL, q->out[k], gpu);
        }
        stat_counters_gpu += q->n;
    }
    q->n = 0;
}

static void cq_resolve_done(uint64_t done) {  // every slot whose submission is done
    pthread_mutex_lock(&cq_lock);
    for (int i = 0; i < SUBS; i++) if (cq[i].n && cq[i].seq <= done) cq_resolve(i);
    pthread_mutex_unlock(&cq_lock);
}

static Sub *sub(void) {
    Sub *s = &subs[cur_sub];
    if (s->seq != submit_seq + 1) {  // slot reused: its previous submission must be done
        wait_why = "ring slot";
        vk_wait_seq(s->seq);
        cq_resolve_done(s->seq);
        if (s->ts) {  // (profiling) GPU time of that submission
            uint64_t t[2];
            if (vkGetQueryPoolResults(g_vk.dev, prof_qp, (uint32_t)cur_sub * 2, 2, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && t[1] > t[0])
                prof_gpu += (uint64_t)((double)(t[1] - t[0]) * g_vk.props.limits.timestampPeriod);
            s->ts = false;
        }
        for (int i = 0; i < s->npools; i++) vkResetDescriptorPool(g_vk.dev, s->pools[i], 0);
        s->cur_pool = 0;
        s->seq = submit_seq + 1;
        s->main_begun = s->upload_begun = false;
    }
    return s;
}

static void begin_cb(VkCommandBuffer *cb, bool *begun) {
    if (*begun) return;
    if (!*cb) {
        VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = g_vk.pool,
                                          .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
        VK_CHECK(vkAllocateCommandBuffers(g_vk.dev, &ai, cb));
    }
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK_CHECK(vkBeginCommandBuffer(*cb, &bi));
    *begun = true;
}

static void gp_mark(VkCommandBuffer cb, const char *fmt, ...);
VkCommandBuffer vk_cmd(void) {
    Sub *s = sub();
    if (!s->main_begun) {
        begin_cb(&s->main, &s->main_begun);
        ss_forget();  // (a new command buffer starts with no state)
        if (vk_counter_mode() == 2) {  // this slot's timing counter queries (reset outside any render pass)
            CounterQ *q = &cq[cur_sub];
            if (!q->pool) {
                VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                            .queryCount = CQ_MAX};
                if (vkCreateQueryPool(g_vk.dev, &qi, NULL, &q->pool) != VK_SUCCESS) q->pool = VK_NULL_HANDLE;
            }
            pthread_mutex_lock(&cq_lock);
            q->n = 0;  // (resolved when the slot was freed: sub())
            q->seq = s->seq;
            pthread_mutex_unlock(&cq_lock);
            if (q->pool) vkCmdResetQueryPool(s->main, q->pool, 0, CQ_MAX);
        }
        gp_mark(s->main, "(submission start; includes GPU idle before it)");
        if (prof && prof_qp) {
            vkCmdResetQueryPool(s->main, prof_qp, (uint32_t)cur_sub * 2, 2);
            vkCmdWriteTimestamp2(s->main, VK_PIPELINE_STAGE_2_NONE, prof_qp, (uint32_t)cur_sub * 2);
            s->ts = true;
        }
    }
    return s->main;
}

VkCommandBuffer vk_upload_cmd(void) {
    Sub *s = sub();
    if (!s->upload_begun) {
        begin_cb(&s->upload, &s->upload_begun);
        // after everything submitted before it: a submission's commands may start while the previous one's still run,
        // and these copies overwrite what those may read (textures being replaced, vertex cache pages)
        vk_full_barrier(s->upload);
    }
    return s->upload;
}

void vk_full_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

static void ring_retire(void);
void vc_flush(void);

// ---- LARECOMP_GPUPROF_AT=<frame>: GPU time per pass / dispatch / copy of that frame, logged as "[gpuprof]".
// A timestamp (ALL_COMMANDS: after everything before it) at each mark; a row = the work since the previous mark.
bool g_gpuprof_frame;
static VkQueryPool gp_qp;
static int gp_n;
static char gp_label[512][112];
static uint32_t gp_draws[512];
// LARECOMP_VK_DEVLOST_TRACE=1: the last 64 GPU operations, dumped on device loss (with LARECOMP_SYNC_SUBMIT=1 the
// dump ends with the command list that faulted).
static char dl_ring[64][128];
static unsigned dl_n;
static int dl_on = -1;
static void dl_note(const char *fmt, ...) {
    if (dl_on < 0) dl_on = getenv("LARECOMP_VK_DEVLOST_TRACE") ? atoi(getenv("LARECOMP_VK_DEVLOST_TRACE")) : 0;
    if (!dl_on) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(dl_ring[dl_n++ & 63], sizeof dl_ring[0], fmt, ap);
    va_end(ap);
}
void vk_dl_note_upload(void *img, uint32_t w, uint32_t h, uint32_t lv) { dl_note("  upload %p %ux%u lv%u", img, w, h, lv); }
void vk_devlost_dump(void) {
    if (!dl_on) return;
    LOG("[vk] last GPU operations before the device loss (oldest first):\n");
    for (unsigned i = dl_n > 64 ? dl_n - 64 : 0; i < dl_n; i++) LOG("[vk]   %s\n", dl_ring[i & 63]);
}
static void gp_mark(VkCommandBuffer cb, const char *fmt, ...) {
    if (!g_gpuprof_frame || gp_n >= 511) return;
    if (!gp_qp) {
        VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 512};
        if (vkCreateQueryPool(g_vk.dev, &qi, NULL, &gp_qp) != VK_SUCCESS) { g_gpuprof_frame = false; return; }
    }
    if (!gp_n) vkCmdResetQueryPool(cb, gp_qp, 0, 512);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, gp_qp, (uint32_t)gp_n);
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(gp_label[gp_n], sizeof gp_label[0], fmt, ap);
    va_end(ap);
    gp_draws[gp_n] = 0;
    gp_n++;
}
void renderer_flush(void);
void vk_gpuprof_report(void) {
    if (!gp_n) return;
    renderer_flush();
    uint64_t t[512];
    if (vkGetQueryPoolResults(g_vk.dev, gp_qp, 0, (uint32_t)gp_n, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
        double per = g_vk.props.limits.timestampPeriod / 1e6, total = (t[gp_n - 1] - t[0]) * per;
        LOG("[gpuprof] %d marks, %.2f ms first to last (rows: the work after each mark, until the next)\n", gp_n, total);
        for (int i = 0; i + 1 < gp_n; i++) {
            double ms = (t[i + 1] - t[i]) * per;
            LOG("[gpuprof] %6.3f ms %5.1f%%  %s%s", ms, 100 * ms / (total > 0 ? total : 1), gp_label[i], gp_draws[i] ? "" : "\n");
            if (gp_draws[i]) LOG(", %u draws\n", gp_draws[i]);
        }
    }
    gp_n = 0;
}

// Submit the pending work (if any). wait/signal: binary semaphores for presentation (may be null).
void vk_submit_extra(VkSemaphore wait, VkSemaphore signal, VkCommandBuffer extra) {
    (void)extra;
    vk_end_pass();
    Sub *s = sub();
    if (!s->main_begun && !s->upload_begun && !wait && !signal) return;
    VkCommandBufferSubmitInfo cbs[2];
    int n = 0;
    vc_flush();
    if (s->upload_begun) {
        vk_full_barrier(s->upload);  // uploads complete before anything in the main buffer reads them
        VK_CHECK(vkEndCommandBuffer(s->upload));
        cbs[n++] = (VkCommandBufferSubmitInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = s->upload};
    }
    vk_cmd();  // (an empty main buffer still carries the semaphores)
    if (s->ts) vkCmdWriteTimestamp2(s->main, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, prof_qp, (uint32_t)cur_sub * 2 + 1);
    VK_CHECK(vkEndCommandBuffer(s->main));
    cbs[n++] = (VkCommandBufferSubmitInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = s->main};
    VkSemaphoreSubmitInfo sig[2] = {{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = g_vk.timeline,
                                     .value = submit_seq + 1, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
                                    {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = signal,
                                     .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT}};
    VkSemaphoreSubmitInfo wt = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = wait,
                                .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .commandBufferInfoCount = (uint32_t)n, .pCommandBufferInfos = cbs,
                        .signalSemaphoreInfoCount = signal ? 2 : 1, .pSignalSemaphoreInfos = sig,
                        .waitSemaphoreInfoCount = wait ? 1 : 0, .pWaitSemaphoreInfos = &wt};
    {
        PROF_T0();
        dl_note("submit %llu", (unsigned long long)submit_seq + 1);
        VkResult sr = vkQueueSubmit2(g_vk.queue, 1, &si, VK_NULL_HANDLE);
        if (sr == VK_ERROR_DEVICE_LOST) vk_devlost_dump();
        VK_CHECK(sr);
        PROF_ADD(prof_submit);
    }
    pthread_mutex_lock(&seq_lock);
    submit_seq++;
    pthread_mutex_unlock(&seq_lock);
    cur_sub = (cur_sub + 1) % SUBS;
    stat_flushes++;
    cum_submits++;
    ring_retire();
    vk_collect_dead();
}

// One guest thread at a time in the backend. Every NVN queue submits on the guest thread that owns it, and Three Houses
// uses more than one, but the backend keeps a single set of execution state (subs[] and their open command buffers,
// the bound state `st`, the render pass): two threads in it at once recorded into a command buffer the other had just
// ended (vkCmdCopyBufferToImage outside the recording state; vkEndCommandBuffer -13). Recursive: execution submits,
// presenting draws. Only GPU waits happen inside, never a wait on another guest thread.
static pthread_mutex_t backend_mu;
static pthread_once_t backend_mu_once = PTHREAD_ONCE_INIT;
static void backend_mu_init(void) {
    pthread_mutexattr_t a;
    pthread_mutexattr_init(&a);
    pthread_mutexattr_settype(&a, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&backend_mu, &a);
    pthread_mutexattr_destroy(&a);
}
void vk_backend_lock(void) {
    pthread_once(&backend_mu_once, backend_mu_init);
    pthread_mutex_lock(&backend_mu);
}
void vk_backend_unlock(void) { pthread_mutex_unlock(&backend_mu); }

void renderer_submit(void) {
    vk_backend_lock();
    vk_submit_extra(VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE);
    vk_backend_unlock();
}

void renderer_flush(void) {
    renderer_submit();
    if (!wait_why) wait_why = "flush";
    vk_wait_seq(submit_seq);
}

void renderer_sync(void) {
    pthread_mutex_lock(&seq_lock);
    uint64_t s = submit_seq;
    pthread_mutex_unlock(&seq_lock);
    wait_why = "QueueFinish / sync all";
    vk_wait_seq(s);
}

uint64_t renderer_fence_seq(void) {
    pthread_mutex_lock(&seq_lock);
    uint64_t q = submit_seq + (subs[cur_sub].main_begun || subs[cur_sub].upload_begun ? 1 : 0);
    pthread_mutex_unlock(&seq_lock);
    return q;
}

void renderer_wait_seq(uint64_t seq) {
    if (!seq) return;
    pthread_mutex_lock(&seq_lock);
    uint64_t s = submit_seq;
    pthread_mutex_unlock(&seq_lock);
    wait_why = seq > s ? "SyncWait (fence not submitted yet)" : "SyncWait";
    vk_wait_seq(seq > s ? s : seq);  // not submitted yet (other thread): everything submitted so far
}

VkDescriptorSet vk_alloc_set(VkDescriptorSetLayout layout) {
    Sub *s = sub();
    for (;;) {
        if (s->cur_pool == s->npools) {
            if (s->npools == 16) FATAL("descriptor pools exhausted\n");
            VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 32768}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096},
                                            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 32768},
                                            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_SAMPLER, 1024},
                                            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024}};
            VkDescriptorPoolCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 8192,
                                             .poolSizeCount = 5, .pPoolSizes = sizes};
            VK_CHECK(vkCreateDescriptorPool(g_vk.dev, &ci, NULL, &s->pools[s->npools++]));
        }
        VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                          .descriptorPool = s->pools[s->cur_pool], .descriptorSetCount = 1, .pSetLayouts = &layout};
        VkDescriptorSet set;
        VkResult r = vkAllocateDescriptorSets(g_vk.dev, &ai, &set);
        if (r == VK_SUCCESS) return set;
        s->cur_pool++;
    }
}

// ================================================================== transient memory ring
// 64 MB host-visible chunks (device-local when the GPU exposes resizable BAR); a chunk is reused once every
// submission that allocated from it has completed.
#define CHUNK_SIZE (64ull << 20)
typedef struct { VkBuffer buf; VmaAllocation mem; uint8_t *ptr; uint64_t size, used, seq; } Chunk;
static Chunk *chunks;
static int nchunks, active = -1;

static int new_chunk(uint64_t size) {
    chunks = realloc(chunks, (nchunks + 1) * sizeof *chunks);
    Chunk *c = &chunks[nchunks];
    memset(c, 0, sizeof *c);
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                             .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    if (getenv("LARECOMP_VK_RING_SYSMEM")) ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;  // (test: a GPU without resizable BAR)
    VmaAllocationInfo info;
    VK_CHECK(vmaCreateBuffer(g_vk.vma, &bi, &ai, &c->buf, &c->mem, &info));
    c->ptr = info.pMappedData;
    c->size = size;
    if (!nchunks) {  // where the ring lives: device-local (resizable BAR) or system memory read across PCIe
        VkMemoryPropertyFlags f;
        vmaGetMemoryTypeProperties(g_vk.vma, info.memoryType, &f);
        LOG("[vk] transient ring: memory type %u (%s%s%s)\n", info.memoryType, f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT ? "device-local " : "system memory ",
            f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ? "host-visible " : "", f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT ? "cached" : "");
    }
    return nchunks++;
}

static void ring_retire(void) {}  // chunks are recycled lazily in vk_transient

VkSpan vk_transient(uint64_t size, uint64_t align) {
    if (align < 16) align = 16;
    uint64_t pending = vk_pending_seq();
    if (active >= 0) {
        Chunk *c = &chunks[active];
        uint64_t off = (c->used + align - 1) & ~(align - 1);
        if (off + size <= c->size) {
            c->used = off + size;
            c->seq = pending;
            stat_upload_bytes += size;
            return (VkSpan){c->buf, off, c->ptr + off};
        }
    }
    uint64_t done = vk_completed_seq();
    int pick = -1;
    for (int i = 0; i < nchunks; i++)
        if (i != active && chunks[i].seq <= done && chunks[i].size >= size) { pick = i; break; }
    if (pick < 0) pick = new_chunk(size > CHUNK_SIZE ? size : CHUNK_SIZE);
    active = pick;
    chunks[pick].used = size;
    chunks[pick].seq = pending;
    stat_upload_bytes += size;
    return (VkSpan){chunks[pick].buf, 0, chunks[pick].ptr};
}

// GPU -> CPU copies: the ring is write-combined (often device-local over the BAR), where CPU reads crawl (a
// 1280x720 screenshot took 7 s). Readbacks use one host-cached buffer instead. Every user waits for the GPU and
// reads the data before the next readback (render thread), so a single grow-only buffer suffices.
VkSpan vk_readback(uint64_t size) {
    static VkBuffer buf;
    static VmaAllocation mem;
    static uint8_t *ptr;
    static uint64_t cap;
    if (size > cap) {
        if (buf) { vk_wait_seq(submit_seq); vmaDestroyBuffer(g_vk.vma, buf, mem); }
        cap = (size + (16 << 20) - 1) & ~((16ull << 20) - 1);
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = cap, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        VK_CHECK(vmaCreateBuffer(g_vk.vma, &bi, &ai, &buf, &mem, &info));
        ptr = info.pMappedData;
    }
    return (VkSpan){buf, 0, ptr};
}

VkSpan vk_transient_copy(const void *data, uint64_t size, uint64_t align) {
    VkSpan s = vk_transient(size ? size : 16, align);
    if (size) memcpy(s.ptr, data, size);
    return s;
}

// Per-draw constants that usually repeat the previous draw's (scale table, tint, driver constants): reuse that copy
// while it's identical, within one submission. slot: one per kind of data.
enum { DEDUPE_RSCALE_V, DEDUPE_RSCALE_F, DEDUPE_TINT, DEDUPE_DRV_V, DEDUPE_DRV_F, DEDUPE_DRV_C, DEDUPE_SLOTS };
static VkSpan copy_dedupe(int slot, const void *data, uint64_t size, uint64_t align) {
    static struct { uint64_t seq, size; uint8_t data[0x800]; VkSpan span; } last[DEDUPE_SLOTS];
    uint64_t seq = vk_pending_seq();
    if (size > sizeof last[0].data) return vk_transient_copy(data, size, align);
    if (last[slot].seq == seq && last[slot].size == size && !memcmp(last[slot].data, data, size)) return last[slot].span;
    VkSpan s = vk_transient_copy(data, size, align);
    last[slot].seq = seq; last[slot].size = size; memcpy(last[slot].data, data, size); last[slot].span = s;
    return s;
}

// Guest memory copied once per submission: draws binding the same range share the copy.
#define GCOPY_TAB 4096
static struct { uint64_t addr, size, seq; VkSpan span; } gcopy[GCOPY_TAB];
static uint64_t gcopy_gen;  // bumped to drop every shared copy (guest memory written mid-list)
static void guest_copy_flush(void) { gcopy_gen++; }
static VkSpan guest_copy(uint64_t addr, uint64_t size) {
    uint64_t seq = vk_pending_seq() + (gcopy_gen << 40), h = (addr * 0x9E3779B97F4A7C15ull) ^ (size * 0xC2B2AE3D27D4EB4Full);
    uint32_t slot = (uint32_t)(h ^ (h >> 32)) & (GCOPY_TAB - 1);
    if (gcopy[slot].seq == seq && gcopy[slot].addr == addr && gcopy[slot].size == size) return gcopy[slot].span;
    VkSpan s = vk_transient_copy((const void *)(uintptr_t)addr, size, 256);
    gcopy[slot].addr = addr; gcopy[slot].size = size; gcopy[slot].seq = seq; gcopy[slot].span = s;
    return s;
}

// ================================================================== worker threads (shader / pipeline builds)
// First-use compiles run off the render thread; draws needing them are skipped until ready (as on Metal).
// LARECOMP_SYNC_SHADERS=1 builds on first use, blocking (deterministic tests).
typedef struct Job { void (*fn)(void *); void *arg; struct Job *next; } Job;
static Job *jobs_head, *jobs_tail;
static Job *ljobs_head, *ljobs_tail;  // low priority (prewarm, below): taken when nothing else is waiting
static _Atomic int jobs_queued, jobs_running, nworkers;  // (high-priority jobs; the stats line and wait_built)
static Job *heavy_head, *heavy_tail;  // builds of very large shaders: their own thread (heavy_worker), so a driver
static pthread_cond_t heavy_cond = PTHREAD_COND_INITIALIZER;  // compile taking minutes can't hold up the others
#define NHEAVY 3                      // (one driver compile that never ends doesn't hold up the other large ones)
static _Atomic uint64_t heavy_since[NHEAVY];  // when each heavy thread's current build started (0: idle)
static pthread_mutex_t jobs_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t jobs_cond = PTHREAD_COND_INITIALIZER;

// Builds finishing (a program's or pipeline's state leaving 0) wake the draws waiting for them (wait_built)
static pthread_mutex_t built_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t built_cond = PTHREAD_COND_INITIALIZER;
static void set_state(_Atomic int *state, int v) {
    pthread_mutex_lock(&built_lock);
    atomic_store(state, v);
    pthread_cond_broadcast(&built_cond);
    pthread_mutex_unlock(&built_lock);
}

static void *worker(void *a) {
    (void)a;
    for (;;) {
        pthread_mutex_lock(&jobs_lock);
        while (!jobs_head && !ljobs_head) pthread_cond_wait(&jobs_cond, &jobs_lock);
        Job *j;
        if (jobs_head) {
            j = jobs_head;
            jobs_head = j->next;
            if (!jobs_head) jobs_tail = NULL;
            atomic_fetch_sub(&jobs_queued, 1);
        } else {
            j = ljobs_head;
            ljobs_head = j->next;
            if (!ljobs_head) ljobs_tail = NULL;
        }
        pthread_mutex_unlock(&jobs_lock);
        atomic_fetch_add(&jobs_running, 1);
        j->fn(j->arg);
        atomic_fetch_sub(&jobs_running, 1);
        free(j);
    }
    return NULL;
}

static void *heavy_worker(void *a) {
    int slot = (int)(intptr_t)a;
    for (;;) {
        pthread_mutex_lock(&jobs_lock);
        while (!heavy_head) pthread_cond_wait(&heavy_cond, &jobs_lock);
        Job *j = heavy_head;
        heavy_head = j->next;
        if (!heavy_head) heavy_tail = NULL;
        pthread_mutex_unlock(&jobs_lock);
        atomic_store(&heavy_since[slot], SDL_GetTicksNS());
        j->fn(j->arg);
        atomic_store(&heavy_since[slot], 0);
        free(j);
    }
    return NULL;
}

static bool sync_shaders(void) {
    static int v = -1;
    if (v < 0) v = getenv("LARECOMP_SYNC_SHADERS") != NULL;
    return v;
}

// Shader translation: one dedicated thread (the translator isn't re-entrant - translate.c's RSC2 string ring - and
// translating on the game thread cost up to ~90 ms per frame on entering an area). Compiles go on to the workers.
static pthread_mutex_t translate_lock = PTHREAD_MUTEX_INITIALIZER;  // (compute shaders translate on the game thread)
static Job *tjobs_head, *tjobs_tail;
static pthread_mutex_t tjobs_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t tjobs_cond = PTHREAD_COND_INITIALIZER;
static void *translator(void *a) {
    (void)a;
    for (;;) {
        pthread_mutex_lock(&tjobs_lock);
        while (!tjobs_head) pthread_cond_wait(&tjobs_cond, &tjobs_lock);
        Job *j = tjobs_head;
        tjobs_head = j->next;
        if (!tjobs_head) tjobs_tail = NULL;
        pthread_mutex_unlock(&tjobs_lock);
        j->fn(j->arg);
        free(j);
    }
    return NULL;
}
static bool sync_shaders(void);
static void run_translate(void (*fn)(void *), void *arg) {
    if (sync_shaders()) { fn(arg); return; }
    static bool started;
    if (!started) { started = true; pthread_t t; pthread_create(&t, NULL, translator, NULL); }
    Job *j = malloc(sizeof *j);
    j->fn = fn; j->arg = arg; j->next = NULL;
    pthread_mutex_lock(&tjobs_lock);
    if (tjobs_tail) tjobs_tail->next = j; else tjobs_head = j;
    tjobs_tail = j;
    pthread_cond_signal(&tjobs_cond);
    pthread_mutex_unlock(&tjobs_lock);
}

static void start_workers(void) {
    // first entry into an area builds dozens of pipelines at once: use most of the CPU (draws wait for them)
    int n = SDL_GetNumLogicalCPUCores() - 4;
    if (n < 4) n = 4;
    if (n > 16) n = 16;
    for (int i = 0; i < n; i++) { pthread_t t; pthread_create(&t, NULL, worker, NULL); }
    atomic_store(&nworkers, n);
    for (int i = 0; i < NHEAVY; i++) { pthread_t t; pthread_create(&t, NULL, heavy_worker, (void *)(intptr_t)i); }
}
static pthread_once_t workers_once = PTHREAD_ONCE_INIT;

static void run_job_at(void (*fn)(void *), void *arg, bool low) {
    if (sync_shaders()) { fn(arg); return; }
    pthread_once(&workers_once, start_workers);
    Job *j = malloc(sizeof *j);
    j->fn = fn; j->arg = arg; j->next = NULL;
    pthread_mutex_lock(&jobs_lock);
    Job **head = low ? &ljobs_head : &jobs_head, **tail = low ? &ljobs_tail : &jobs_tail;
    if (*tail) (*tail)->next = j; else *head = j;
    *tail = j;
    if (!low) atomic_fetch_add(&jobs_queued, 1);
    pthread_cond_signal(&jobs_cond);
    pthread_mutex_unlock(&jobs_lock);
}
static void run_job(void (*fn)(void *), void *arg) { run_job_at(fn, arg, false); }
static void run_heavy_job(void (*fn)(void *), void *arg) {
    if (sync_shaders()) { fn(arg); return; }
    pthread_once(&workers_once, start_workers);
    Job *j = malloc(sizeof *j);
    j->fn = fn; j->arg = arg; j->next = NULL;
    pthread_mutex_lock(&jobs_lock);
    if (heavy_tail) heavy_tail->next = j; else heavy_head = j;
    heavy_tail = j;
    pthread_cond_signal(&heavy_cond);
    pthread_mutex_unlock(&jobs_lock);
}

// Huge compute shaders (over 4000 instructions, battle effects): a thread of their own each, so a compile of minutes
// holds nothing else up. NVIDIA's driver takes ~160 s for the largest the first time, then has it in its own cache;
// AMD's never finished, so elsewhere they're skipped (LARECOMP_VK_BIG_CS=1 / 0 to choose).
static bool big_cs_ok(void) {
    static int ok = -1;
    if (ok < 0) {
        const char *e = getenv("LARECOMP_VK_BIG_CS");
        ok = e ? atoi(e) != 0 : g_vk.props.vendorID == 0x10DE;
    }
    return ok;
}
typedef struct { void (*fn)(void *); void *arg; } HugeJob;
static void *huge_thread(void *a) {  // (two at a time: eight at once ran the driver's compiler out of memory)
    static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
    static pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
    static int running;
    HugeJob j = *(HugeJob *)a;
    free(a);
    pthread_mutex_lock(&m);
    while (running >= 2) pthread_cond_wait(&cv, &m);
    running++;
    pthread_mutex_unlock(&m);
    j.fn(j.arg);
    pthread_mutex_lock(&m);
    running--;
    pthread_cond_signal(&cv);
    pthread_mutex_unlock(&m);
    return NULL;
}
static void run_huge_job(void (*fn)(void *), void *arg) {
    if (sync_shaders()) { fn(arg); return; }
    HugeJob *j = malloc(sizeof *j);
    j->fn = fn, j->arg = arg;
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    pthread_t t;
    if (pthread_create(&t, &at, huge_thread, j)) { free(j); run_heavy_job(fn, arg); }
    pthread_attr_destroy(&at);
}

// Builds slower than a quarter second are logged (what, how long, where the time went)
static void slow_build_log(const char *what, uint64_t code, uint32_t insts, uint64_t t0, uint64_t t_spirv) {
    uint64_t t1 = SDL_GetTicksNS();
    if (t1 - t0 < 250000000ull) return;
    static _Atomic int nlog;
    if (atomic_fetch_add(&nlog, 1) < 40)
        LOG("[shader] slow build: %s %llx (%u instructions) %.2f s (GLSL to SPIR-V %.2f s, driver %.2f s)\n", what,
            (unsigned long long)code, insts, (t1 - t0) / 1e9, (t_spirv - t0) / 1e9, (t1 - t_spirv) / 1e9);
}

// ================================================================== state
typedef struct {
    NvnTexture *color[8];
    NvnTextureView cview[8];
    uint8_t has_cview[8];
    int ncolor;
    NvnTexture *depth;
    NvnTextureView dview;
    uint8_t has_dview;
} RtState;

typedef struct { uint64_t addr, size; } Range;

static struct {
    RtState rt;
    bool rt_dirty;
    NvnProgram *program;           // what draws / dispatches use: stage_prog combined (bind_program)
    NvnProgram *stage_prog[6];     // bound per stage: vertex, fragment, geometry, tess control, tess eval, compute
    uint8_t blend[8][8];
    uint32_t channel_mask;
    uint8_t color_state[4], ds[8], ms[0x18], poly[4];
    float poly_offset[3];
    uint32_t attrib[16];
    int nattrib;
    uint32_t stream[16][2];
    int nstream;
    Range vbuf[16], ubuf[6][18], sbuf[6][16];
    uint64_t tex[6][32], img[6][8];
    NvnDescPool *tpool, *spool;
    float viewport[16][4];
    int32_t scissor[16][4];
    float depth_range[16][2];
    float blend_color[4];
    uint32_t stencil_ref[2], stencil_mask[2], stencil_vmask[2];
    bool prim_restart;
    uint32_t restart_index;
    uint64_t copy_row_stride, copy_image_stride;
} st;

uint32_t nvn_window_origin(void);
uint32_t nvn_depth_mode(void);
bool renderer_origin_lower_left(void) { return nvn_window_origin() == 0; }

// ================================================================== waiting for shaders being built
// A draw (or dispatch) whose program or pipeline is still building used to be skipped: things popped in a frame or
// more later, every time something was first seen. With the translations, SPIR-V and pipelines kept on disk most
// builds take a millisecond or two, so the draw waits for them instead, up to a budget a frame (the menu's "Wait for
// shaders": Full 50 ms, Balanced 4 ms, Off none) - past it the frame goes on and the rest are skipped as before.
// A build still not done after a whole frame's budget of waiting is a long one (a driver compile never seen before):
// it isn't waited for again - its draws are skipped until it's ready, instead of each frame losing its budget to it.
static _Atomic uint64_t wait_budget_ns, stat_build_wait_ns;
static uint64_t wait_frame_ns;
static void wait_budget_reset(void) {
    static const uint64_t budget_ms[3] = {0, 4, 50};  // (Off, Balanced, Full)
    wait_frame_ns = budget_ms[g_settings.shader_wait % 3] * 1000000ull;
    atomic_store(&wait_budget_ns, wait_frame_ns);
}
// true when ready (state 1), after waiting for it to finish building within the frame's budget. waited_ns: the
// object's own total so far.
static bool wait_built(_Atomic int *state, uint64_t *waited_ns) {
    int s = atomic_load(state);
    if (s != 0) return s == 1;
    uint64_t left = atomic_load(&wait_budget_ns);
    if (!left || *waited_ns >= wait_frame_ns) return false;
    if (atomic_load(&jobs_queued) > atomic_load(&nworkers)) return false;  // (a backlog: it won't be done in time)
    if (left > wait_frame_ns - *waited_ns) left = wait_frame_ns - *waited_ns;
    uint64_t t0 = SDL_GetTicksNS(), el = 0;
    struct timespec dl;
    clock_gettime(CLOCK_REALTIME, &dl);
    uint64_t end = (uint64_t)dl.tv_sec * 1000000000ull + (uint64_t)dl.tv_nsec + left;
    dl.tv_sec = (time_t)(end / 1000000000ull);
    dl.tv_nsec = (long)(end % 1000000000ull);
    pthread_mutex_lock(&built_lock);
    while ((s = atomic_load(state)) == 0)
        if (pthread_cond_timedwait(&built_cond, &built_lock, &dl)) { s = atomic_load(state); break; }
    pthread_mutex_unlock(&built_lock);
    el = SDL_GetTicksNS() - t0;
    *waited_ns += el;
    uint64_t all = atomic_load(&wait_budget_ns);
    atomic_store(&wait_budget_ns, el < all ? all - el : 0);
    atomic_fetch_add(&stat_build_wait_ns, el);
    return s == 1;
}

// (LARECOMP_VK_PROF) called once per presented frame: frames over 50 ms are logged with where their time went.
// present_ns: time in backend_present's acquire/blit/present.
void vk_prof_frame(uint64_t frame, uint64_t present_ns) {
    wait_budget_reset();  // (each frame: its own budget for waiting on shaders)
    if (!prof) return;
    static uint64_t last_t, l_exec, l_wait, l_draws, l_subs;
    uint64_t now = SDL_GetTicksNS();
    if (last_t && now - last_t > 50000000) {
        uint64_t dt = now - last_t, ex = cum_exec - l_exec, wt = cum_wait - l_wait;
        LOG("[hitch] frame %llu: %.0f ms = backend exec %.1f (of which GPU waits %.1f) + present %.1f + other %.1f; %llu draws, "
            "%llu submits, compiles %d pending\n", (unsigned long long)frame, dt / 1e6, ex / 1e6, wt / 1e6, present_ns / 1e6,
            ((double)dt - ex - present_ns) / 1e6, (unsigned long long)(cum_draws - l_draws), (unsigned long long)(cum_submits - l_subs),
            atomic_load(&stat_compiles));
    }
    last_t = now; l_exec = cum_exec; l_wait = cum_wait; l_draws = cum_draws; l_subs = cum_submits;
}

const char *renderer_stats(void) {
    static int dumps;
    if (dumps++ < 3) {
        LOG("[vk] command counts:");
        for (int i = 0; i < CMD_COUNT; i++) if (op_counts[i]) LOG(" %d:%llu", i, (unsigned long long)op_counts[i]);
        LOG("\n");
    }
    snprintf(stats_text, sizeof stats_text,
             "%llu draws, %llu skipped, %llu passes, %llu dispatches, %llu flushes, wait %.0fms, compile %d, fbcopy %llu, upload %.1fMB",
             (unsigned long long)stat_draws, (unsigned long long)stat_skipped, (unsigned long long)stat_passes,
             (unsigned long long)stat_dispatches, (unsigned long long)stat_flushes, (double)stat_wait_ns / 1e6,
             atomic_exchange(&stat_compiles, 0), (unsigned long long)stat_feedback_copies, (double)stat_upload_bytes / 1048576.0);
    {  // translations this period: from the disk cache / translated (and the translator's time)
        uint64_t hit = atomic_exchange(&xlat_hits, 0), miss = atomic_exchange(&xlat_misses, 0), ns = atomic_exchange(&xlat_ns, 0);
        if (hit || miss) {
            size_t n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, ", xlat %llu cached / %llu new %.0fms", (unsigned long long)hit,
                     (unsigned long long)miss, ns / 1e6);
        }
        for (int i = 0; i < NHEAVY; i++) {
            uint64_t hs = atomic_load(&heavy_since[i]);
            static uint64_t hs_logged[NHEAVY];
            if (hs && hs != hs_logged[i] && SDL_GetTicksNS() - hs > 10000000000ull) {
                hs_logged[i] = hs;
                LOG("[shader] a large compute shader has been building for over 10 s\n");
            }
        }
        int q = atomic_load(&jobs_queued);
        if (q) {
            size_t n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, ", %d builds queued", q);
        }
        uint64_t w = atomic_exchange(&stat_build_wait_ns, 0);
        if (w) {
            size_t n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, ", waited %.0fms for shaders", w / 1e6);
        }
    }
    stat_feedback_copies = stat_upload_bytes = 0;
    if (stat_skipped) {
        size_t n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | skipped:");
        for (int i = 0; i < SK_N; i++)
            if (stat_skip_why[i]) {
                n = strlen(stats_text);
                snprintf(stats_text + n, sizeof stats_text - n, " %s %llu", skip_names[i], (unsigned long long)stat_skip_why[i]);
            }
    }
    memset(stat_skip_why, 0, sizeof stat_skip_why);
    {
        extern void vk_alias_stats(void);
        vk_alias_stats();
    }
    if (prof) {
        prof_up = prof_up_ns;
        prof_up_ns = 0;
        size_t n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | gpu %.0f cpu exec %.0f draw %.0f (pre %.0f key %.0f dyn %.0f sets %.0f tex %.0f (upload %.0f) vtx %.0f) submit %.0f ms",
                 prof_gpu / 1e6, prof_exec / 1e6, prof_draw / 1e6, prof_pre / 1e6, prof_key / 1e6, prof_dyn / 1e6, prof_sets / 1e6,
                 prof_tex / 1e6, prof_up / 1e6, prof_vtx / 1e6, prof_submit / 1e6);
        extern uint64_t prof_tex_checks, prof_tex_samples, prof_tex_full, prof_tex_full_bytes, prof_tex_dyn;
        n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | tex checks %llu samples %llu full %llu (%.1f MB, dynamic %llu)",
                 (unsigned long long)prof_tex_checks, (unsigned long long)prof_tex_samples, (unsigned long long)prof_tex_full,
                 prof_tex_full_bytes / 1048576.0, (unsigned long long)prof_tex_dyn);
        n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | pass ends:");
        for (int l = 0; l < 4096; l++)
            if (prof_end_line[l] > 20) { n = strlen(stats_text); snprintf(stats_text + n, sizeof stats_text - n, " %d:%u", l, prof_end_line[l]); }
        memset(prof_end_line, 0, sizeof prof_end_line);
        n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | views created %llu | counters gpu %llu | pace %s, %llu frames over", (unsigned long long)prof_views_created,
                 (unsigned long long)stat_counters_gpu, pace_slow ? "30" : "60", (unsigned long long)stat_pace_over);
        stat_counters_gpu = stat_pace_over = 0;
        {
            extern uint64_t prof_vc_hits, prof_vc_bypass, prof_vc_copied, prof_vc_queries, prof_vc_scan_pages, prof_vc_scan_ns;
            n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n,  " | vcache %llu hit %llu in place %.1f MB copied, %llu queries %.2f ms, %llu pages written",
                     (unsigned long long)prof_vc_hits, (unsigned long long)prof_vc_bypass, prof_vc_copied / 1048576.0,
                     (unsigned long long)prof_vc_queries, prof_vc_scan_ns / 1e6, (unsigned long long)prof_vc_scan_pages);
            prof_vc_hits = prof_vc_bypass = prof_vc_copied = prof_vc_queries = prof_vc_scan_pages = prof_vc_scan_ns = 0;
            extern uint64_t prof_tex_watch_skips;
            n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, " | tex unwritten %llu", (unsigned long long)prof_tex_watch_skips);
            prof_tex_watch_skips = 0;
        }
        pthread_mutex_lock(&wait_tab_lock);
        for (int i = 0; i < 12 && wait_tab[i].why; i++) {
            n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, "%s%s %llux %.0f ms", i ? "," : " | waits:", wait_tab[i].why,
                     (unsigned long long)wait_tab[i].n, wait_tab[i].ns / 1e6);
        }
        memset(wait_tab, 0, sizeof wait_tab);
        pthread_mutex_unlock(&wait_tab_lock);
        prof_views_created = 0;
        prof_tex_checks = prof_tex_samples = prof_tex_full = prof_tex_full_bytes = prof_tex_dyn = 0;
        prof_exec = prof_draw = prof_sets = prof_vtx = prof_submit = prof_pre = prof_key = prof_dyn = prof_tex = prof_up = prof_gpu = 0;
    }
    stat_draws = stat_skipped = stat_passes = stat_dispatches = stat_flushes = stat_wait_ns = 0;
    return stats_text;
}

// nvnCommandBufferBindProgram(cmd, program, stages) binds a program to the given stages only (NVNshaderStageBits:
// vertex 1, fragment 2, geometry 4, tess control 8, tess eval 16, compute 32 - the stages[] index order). Link's
// Awakening binds one program for everything; Three Houses binds vertex and fragment programs separately, so the last
// one bound (a fragment-only program) was used for every stage and its draws were skipped. When the bound stages come
// from more than one program, draws use a combined program, made once per set of shader code addresses.
#define MERGED_TAB 4096
static struct { uint64_t code[6]; NvnProgram *prog; } merged_tab[MERGED_TAB];
static NvnProgram *combined_program(void) {
    NvnProgram *one = NULL;
    bool same = true;
    for (int i = 0; i < 6; i++) {
        NvnProgram *p = st.stage_prog[i];
        if (!p) continue;
        if (!one) one = p;
        else if (p != one) same = false;
    }
    if (same) return one;  // (the usual case: the program as bound, its backend caches intact)
    uint64_t code[6], h = 1469598103934665603ull;
    for (int i = 0; i < 6; i++) {
        code[i] = st.stage_prog[i] ? st.stage_prog[i]->stages[i].code_addr : 0;
        h = (h ^ code[i]) * 1099511628211ull;
    }
    uint32_t slot = (uint32_t)(h ^ (h >> 32)) & (MERGED_TAB - 1);
    for (int n = 0; n < MERGED_TAB; n++, slot = (slot + 1) & (MERGED_TAB - 1)) {
        if (merged_tab[slot].prog && !memcmp(merged_tab[slot].code, code, sizeof code)) return merged_tab[slot].prog;
        if (!merged_tab[slot].prog) {
            NvnProgram *m = calloc(1, sizeof *m);
            if (!m) return one;
            for (int i = 0; i < 6; i++) {
                if (!st.stage_prog[i]) continue;
                m->stages[i] = st.stage_prog[i]->stages[i];
                m->stages[i].backend = NULL;
                m->stage_mask |= 1u << i;
            }
            memcpy(merged_tab[slot].code, code, sizeof code);
            merged_tab[slot].prog = m;
            return m;
        }
    }
    static bool warned;
    if (!warned) { warned = true; LOG("[vk] combined program table full\n"); }
    return one;
}
static void bind_program(NvnProgram *p, uint32_t stages) {
    if (!stages) stages = p ? p->stage_mask : 0;  // (no stage bits given: the program's own stages)
    for (int i = 0; i < 6; i++)
        if (stages & (1u << i)) st.stage_prog[i] = p && (p->stage_mask & (1u << i)) ? p : NULL;
    st.program = combined_program();
}

static void st_defaults(void) {
    memset(&st, 0, sizeof st);
    for (int i = 0; i < 16; i++) {
        st.viewport[i][2] = 1280; st.viewport[i][3] = 720;
        st.scissor[i][2] = 16384; st.scissor[i][3] = 16384;
        st.depth_range[i][1] = 1.0f;
    }
    st.channel_mask = 0xFFFFFFFF;
    st.stencil_mask[0] = st.stencil_mask[1] = 0xFF;
    st.stencil_vmask[0] = st.stencil_vmask[1] = 0xFF;
    st.ds[1] = 1; st.ds[2] = 2;
    st.poly[1] = 1; st.poly[2] = 2;
    for (int i = 0; i < 8; i++) { st.blend[i][1] = 2; st.blend[i][2] = 1; st.blend[i][3] = 2; st.blend[i][4] = 1; st.blend[i][5] = 1; st.blend[i][6] = 1; }
}

void vk_exec_init(void) {
    prof = getenv("LARECOMP_VK_PROF") != NULL;
    if (prof) {
        prof_up_ns = 0;
        VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = SUBS * 2};
        if (vkCreateQueryPool(g_vk.dev, &qi, NULL, &prof_qp) != VK_SUCCESS) prof_qp = VK_NULL_HANDLE;
    }
    st_defaults();
}

void renderer_texture_scale(NvnTexture *t, float *sx, float *sy);
bool renderer_texture_presented(NvnTexture *t);
bool renderer_texture_first_write(NvnTexture *t);
bool renderer_texture_level_fresh(NvnTexture *t);
bool renderer_texture_frame_fresh(NvnTexture *t);
void renderer_texture_mark_presented(NvnTexture *t);
uint64_t renderer_present_count(void);
void renderer_save_texture_tex(TexBackend *b, const char *path);
static float pass_sx = 1, pass_sy = 1;
static float vp_sx = 1, vp_sy = 1, vp_ox = 0, vp_oy = 0;
static float sc_box;  // (this draw's UI box factor, for its scissor; 0 = none)
static float g_label_dx;  // (this draw's world-label shift in pass pixels; ui_world_label)
static float tex_sc[2][64][2];
static uint32_t tex_wh[2][64][2];

// ================================================================== render passes
static VkFormat pass_color[8], pass_depth, pass_stencil;
#define MV_FMT VK_FORMAT_R16G16_SFLOAT
static const float *g_post_pc80;  // (the next post_pass's push constants, when 80 bytes: mvfull.frag)
static float g_scene_flip[2];     // (the scene's clip-space flips x / y, from its motion draws)
static TexBackend g_mv_img;      // (DLAA) object motion target, the G-buffer pass's color 4: see mv_prog
static bool g_mv_pass;           // the current pass has it attached
static uint64_t g_mv_clear_frame = ~0ull;
static bool mv_enabled(void);
static TexBackend *dlaa_image(TexBackend *d, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage, bool zero);
static VkImageView post_view(TexBackend *d);
static uint32_t pass_ncolor, pass_samples = 1, pass_w, pass_h;
static int debug_pass_no;

// Redundant-state filter: what draw() last set in the current command buffer, so unchanged pipeline binds, dynamic
// state and vertex / index buffer binds are not recorded again (a draw used to issue ~20 vkCmd* calls; most repeat the
// previous draw's). Forgotten (ss_forget) whenever something else may have recorded state: a new command buffer, a new
// render pass (present, ImGui, post-processing and copies all run between game passes) and the clear pipeline.
// Descriptor sets are not filtered: every draw writes new ones (sharing identical sets within a submission was tried:
// a third matched, but comparing them cost more than allocating and writing them on NVIDIA).
// LARECOMP_VK_STATE_FILTER=0: every draw sets everything, as before.
enum { SS_PIPE = 1 << 0, SS_DT = 1 << 1, SS_DW = 1 << 2, SS_DOP = 1 << 3, SS_STEN = 1 << 4, SS_SFACE0 = 1 << 5, SS_SFACE1 = 1 << 6,
       SS_BC = 1 << 7, SS_CULL = 1 << 8, SS_FF = 1 << 9, SS_TOPO = 1 << 10, SS_PRE = 1 << 11, SS_DBE = 1 << 12, SS_DB = 1 << 13,
       SS_VP = 1 << 14, SS_SC = 1 << 15, SS_IB = 1 << 16 };
static struct {
    uint32_t known;  // SS_* bits: the value below is what the command buffer has
    uint32_t vb_known;  // per vertex binding
    VkPipeline pipe;
    VkBool32 dt, dw, sten, pre, dbe;
    VkCompareOp dop;
    uint32_t sface[2][7];  // fail, pass, depth fail, compare, compare mask, write mask, reference
    float bc[4], db[4];  // db: slope, constant, clamp, 24-bit-as-float
    VkCullModeFlags cull;
    VkFrontFace ff;
    VkPrimitiveTopology topo;
    VkViewport vp;
    VkRect2D sc;
    VkBuffer ib; VkDeviceSize io; VkIndexType it;
    VkBuffer vb[17]; VkDeviceSize vo[17];
} ss;
static int ss_on = -1;
static void ss_forget(void) { ss.known = 0; ss.vb_known = 0; }
static bool ss_enabled(void) {
    if (ss_on < 0) { const char *e = getenv("LARECOMP_VK_STATE_FILTER"); ss_on = !e || atoi(e) != 0; }
    return ss_on;
}
// true: the state must be recorded (and is remembered as known)
static inline bool ss_need(uint32_t bit, bool same) {
    if (ss_on && (ss.known & bit) && same) return false;
    ss.known |= bit;
    return true;
}

static void vk_end_pass_at(int line);
void vk_end_pass(void) { vk_end_pass_at(0); }
#define vk_end_pass() vk_end_pass_at(__LINE__)
static void vk_end_pass_at(int line) {
    if (!in_pass) return;
    if (prof && line < 4096) prof_end_line[line]++;
    vkCmdEndRendering(vk_cmd());
    in_pass = false;
    if (g_debug_frame && !getenv("LARECOMP_NAN_SCAN") && st.rt.ncolor && st.rt.color[0] && st.rt.color[0]->backend) {
        char path[256];
        static uint64_t debug_pass_frame = ~0ull;
        static int multi = -1;
        if (multi < 0) multi = getenv("LARECOMP_DEBUG_FRAMES") != NULL;
        if (multi && renderer_present_count() != debug_pass_frame) { debug_pass_frame = renderer_present_count(); debug_pass_no = 0; }
        if (multi) snprintf(path, sizeof path, "build/pass_%llu_%03d.png", (unsigned long long)debug_pass_frame, debug_pass_no++);
        else snprintf(path, sizeof path, "build/pass_%03d.png", debug_pass_no++);
        renderer_save_texture_tex(st.rt.color[0]->backend, path);
        path[strlen(path) - 4] = 0;
        for (int i = 1; i < st.rt.ncolor && i < 8; i++) {  // the other color targets: build/pass_NNN_c<i>.png
            if (!st.rt.color[i] || !st.rt.color[i]->backend) continue;
            char p2[280];
            snprintf(p2, sizeof p2, "%s_c%d.png", path, i);
            renderer_save_texture_tex(st.rt.color[i]->backend, p2);
        }
        if (st.rt.depth && st.rt.depth->backend) {  // its depth / stencil too: build/pass_NNN_d.png, _s.png
            void renderer_save_depth_tex(TexBackend *b, const char *prefix);
            renderer_save_depth_tex(st.rt.depth->backend, path);
        }
    }
}

static uint32_t level_extent(uint32_t v, uint32_t lv) { v >>= lv; return v ? v : 1; }

static void begin_pass(void) {
    if (in_pass && !st.rt_dirty) return;
    vk_end_pass();
    if (dl_on == 2) renderer_flush();  // (LARECOMP_VK_DEVLOST_TRACE=2: each pass its own submission)
    st.rt_dirty = false;
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);  // passes, copies and dispatches see each other's results
    {
        NvnTexture *t0n = st.rt.ncolor ? st.rt.color[0] : NULL;
        dl_note("pass %d color %ux%u fmt %#x depth %p", st.rt.ncolor, t0n ? t0n->width : 0, t0n ? t0n->height : 0, t0n ? t0n->format : 0,
                (void *)st.rt.depth);
    }
    if (g_gpuprof_frame) {
        NvnTexture *t0 = st.rt.ncolor ? st.rt.color[0] : NULL;
        gp_mark(cb, "pass: %d color %ux%u fmt %#x, depth %s", st.rt.ncolor, t0 ? t0->width : 0, t0 ? t0->height : 0,
                t0 ? t0->format : 0, st.rt.depth ? "yes" : "no");
    }
    VkRenderingAttachmentInfo ca[8], da = {0}, sa = {0};
    uint32_t w = 0, minw = UINT32_MAX, minh = UINT32_MAX;  // w: first attachment seen
    pass_ncolor = (uint32_t)st.rt.ncolor;
    pass_samples = 1;
    for (int i = 0; i < st.rt.ncolor; i++) {
        NvnTexture *t = st.rt.color[i];
        ca[i] = (VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        pass_color[i] = VK_FORMAT_UNDEFINED;
        if (!t) continue;
        renderer_mark_gpu_owned(t);
        VkImageView v = vk_view(t, &st.rt.cview[i], st.rt.has_cview[i], true, TEXTYPE_2D);
        TexBackend *b = t->backend;
        if (!v) continue;
        ca[i].imageView = v;
        ca[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        ca[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        bool fresh = (int)renderer_texture_level_fresh(t) | (int)renderer_texture_frame_fresh(t);  // both: each marks its state
        if (renderer_texture_first_write(t) || fresh) {
            ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            ca[i].clearValue.color.float32[3] = fresh ? 0.0f : 1.0f;
        }
        pass_color[i] = st.rt.has_cview[i] && (st.rt.cview[i].setmask & 4) && vk_format(t->format) == b->format
                      ? vk_format(st.rt.cview[i].format) : b->format;
        uint32_t lv = st.rt.has_cview[i] && (st.rt.cview[i].setmask & 1) ? st.rt.cview[i].base_level : 0;
        uint32_t ew = level_extent(b->width, lv), eh = level_extent(b->height, lv);
        if (!w) { renderer_texture_scale(t, &pass_sx, &pass_sy); w = ew; }
        if (ew < minw) minw = ew;
        if (eh < minh) minh = eh;
        pass_samples = b->samples;
    }
    g_mv_pass = false;
    if (st.rt.ncolor == 4 && w && st.rt.depth && pass_samples == 1 && mv_enabled()) {  // (DLAA: the G-buffer)
        dlaa_image(&g_mv_img, minw, minh, MV_FMT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                   VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, true);
        uint64_t f = renderer_present_count();
        ca[4] = (VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = post_view(&g_mv_img),
                                            .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                            .loadOp = g_mv_clear_frame != f ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD};
        g_mv_clear_frame = f;
        pass_color[4] = MV_FMT;
        pass_ncolor = 5;
        g_mv_pass = true;
    }
    pass_depth = pass_stencil = VK_FORMAT_UNDEFINED;
    bool has_d = false, has_s = false;
    if (st.rt.depth) {
        renderer_mark_gpu_owned(st.rt.depth);
        VkImageView v = vk_view(st.rt.depth, &st.rt.dview, st.rt.has_dview, true, TEXTYPE_2D);
        TexBackend *b = st.rt.depth->backend;
        if (v && renderer_format_is_depth(st.rt.depth->format)) {
            VkRenderingAttachmentInfo a = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = v,
                                           .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                                           .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
            if (b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) { da = a; has_d = true; pass_depth = b->format; }
            if (b->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) { sa = a; has_s = true; pass_stencil = b->format; }
            uint32_t lv = st.rt.has_dview && (st.rt.dview.setmask & 1) ? st.rt.dview.base_level : 0;
            uint32_t ew = level_extent(b->width, lv), eh = level_extent(b->height, lv);
            if (!w) { renderer_texture_scale(st.rt.depth, &pass_sx, &pass_sy); w = ew; }
            if (ew < minw) minw = ew;
            if (eh < minh) minh = eh;
            if (!pass_ncolor) pass_samples = b->samples;
        }
    }
    if (!w) { pass_sx = pass_sy = 1; minw = minh = 16; }
    pass_w = minw; pass_h = minh;
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {minw, minh}}, .layerCount = 1,
                          .colorAttachmentCount = pass_ncolor, .pColorAttachments = ca,
                          .pDepthAttachment = has_d ? &da : NULL, .pStencilAttachment = has_s ? &sa : NULL};
    vkCmdBeginRendering(cb, &ri);
    ss_forget();  // (whatever ran between the passes may have set its own state)
    in_pass = true;
    vp_sx = pass_sx; vp_sy = pass_sy; vp_ox = 0; vp_oy = 0; sc_box = 0;
    stat_passes++;
}

static void pass_size(int *w, int *h) {
    *w = *h = 0;
    for (int i = 0; i < st.rt.ncolor && !*w; i++)
        if (st.rt.color[i]) { *w = (int)st.rt.color[i]->width; *h = (int)st.rt.color[i]->height; }
    if (!*w && st.rt.depth) { *w = (int)st.rt.depth->width; *h = (int)st.rt.depth->height; }
    *w = (int)lroundf(*w * pass_sx);
    *h = (int)lroundf(*h * pass_sy);
}

static VkRect2D scissor_rect(void) {
    int w, h;
    pass_size(&w, &h);
    if (!w) return (VkRect2D){{0, 0}, {pass_w, pass_h}};
    int x = st.scissor[0][0], y = st.scissor[0][1], sw = st.scissor[0][2], sh = st.scissor[0][3];
    if (vp_sx != 1 || vp_sy != 1 || vp_ox != 0 || vp_oy != 0) {
        double x0 = floor(x * (double)vp_sx + vp_ox), y0 = floor(y * (double)vp_sy + vp_oy);
        double x1 = ceil(((double)x + sw) * vp_sx + vp_ox), y1 = ceil(((double)y + sh) * vp_sy + vp_oy);
        if (sw >= 16384) x1 = 1e9;
        if (sh >= 16384) y1 = 1e9;
        if (x1 > 1e9) x1 = 1e9;
        if (y1 > 1e9) y1 = 1e9;
        x = (int)x0; y = (int)y0; sw = (int)(x1 - x0); sh = (int)(y1 - y0);
    }
    if (sc_box > 0 && (x > 0 || x + sw < w)) {  // ultrawide UI box: a partial scissor follows its boxed UI (minimap)
        double c = w * 0.5, x0 = floor(c + (x - c) * sc_box), x1 = ceil(c + (x + sw - c) * sc_box);
        x = (int)x0; sw = (int)(x1 - x0);
    }
    if (x < 0) { sw += x; x = 0; }
    if (y < 0) { sh += y; y = 0; }
    if (w > (int)pass_w) w = (int)pass_w;
    if (h > (int)pass_h) h = (int)pass_h;
    if (x >= w || y >= h) { x = y = 0; sw = sh = 0; }
    if (x + sw > w) sw = w - x;
    if (y + sh > h) sh = h - y;
    if (sw < 0) sw = 0;
    if (sh < 0) sh = 0;
    return (VkRect2D){{x, y}, {(uint32_t)sw, (uint32_t)sh}};
}

static void apply_scissor(void) {
    VkRect2D r = scissor_rect();
    if (!ss_need(SS_SC, !memcmp(&r, &ss.sc, sizeof r))) return;
    ss.sc = r;
    vkCmdSetScissor(vk_cmd(), 0, 1, &r);
}

// Metal convention (y up in NDC, origin top-left) through a negative-height viewport.
static void set_viewport(float x, float y, float w, float h, float zn, float zf) {
    if (w <= 0) w = 1;
    if (h == 0) h = 1;
    VkViewport vp = {x, y + h, w, -h, fminf(fmaxf(zn, 0), 1), fminf(fmaxf(zf, 0), 1)};
    if (!ss_need(SS_VP, !memcmp(&vp, &ss.vp, sizeof vp))) return;
    ss.vp = vp;
    vkCmdSetViewport(vk_cmd(), 0, 1, &vp);
}

// ================================================================== clears
static VkPipelineLayout pc_layout(void) {  // push constants only (clears)
    static VkPipelineLayout l;
    if (!l) {
        VkPushConstantRange r = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32};
        VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1,
                                         .pPushConstantRanges = &r};
        VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &l));
    }
    return l;
}

static const VkDynamicState DYN[] = {
    VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_BLEND_CONSTANTS,
    VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
    VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE, VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
    VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
    VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_OP, VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE,
    VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE};
#define NDYN (sizeof DYN / sizeof DYN[0])

// Fixed-function state that isn't part of a pipeline is set explicitly on every bind (it persists across pipelines).
static void dyn_defaults(VkCommandBuffer cb) {
    vkCmdSetDepthBias(cb, 0, 0, 0);
    vkCmdSetDepthBiasEnable(cb, VK_FALSE);
    float bc[4] = {0, 0, 0, 0};
    vkCmdSetBlendConstants(cb, bc);
    vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0xFF);
    vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0xFF);
    vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0);
    vkCmdSetCullMode(cb, VK_CULL_MODE_NONE);
    vkCmdSetFrontFace(cb, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    vkCmdSetPrimitiveTopology(cb, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    vkCmdSetDepthTestEnable(cb, VK_FALSE);
    vkCmdSetDepthWriteEnable(cb, VK_FALSE);
    vkCmdSetDepthCompareOp(cb, VK_COMPARE_OP_ALWAYS);
    vkCmdSetStencilTestEnable(cb, VK_FALSE);
    vkCmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_AND_BACK, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP,
                      VK_COMPARE_OP_ALWAYS);
    vkCmdSetPrimitiveRestartEnable(cb, VK_FALSE);
}

typedef struct { uint64_t key; VkPipeline pipe; } PipeEnt;
#define CLEAR_TAB 256
static PipeEnt clear_tab[CLEAR_TAB];

static VkPipeline clear_pipeline(int index, int kind, uint32_t mask, bool depth, bool stencil) {
    uint64_t key = 0xcbf29ce484222325ull;
#define MIX(v) do { uint64_t v_ = (uint64_t)(v); for (int b_ = 0; b_ < 8; b_++) { key ^= (v_ >> (b_ * 8)) & 0xFF; key *= 0x100000001B3ull; } } while (0)
    MIX(index); MIX(kind); MIX(mask); MIX(depth); MIX(stencil); MIX(pass_ncolor); MIX(pass_depth); MIX(pass_stencil); MIX(pass_samples);
    for (uint32_t i = 0; i < pass_ncolor; i++) MIX(pass_color[i]);
    uint32_t slot = (uint32_t)key & (CLEAR_TAB - 1);
    for (int p = 0; p < CLEAR_TAB; p++, slot = (slot + 1) & (CLEAR_TAB - 1)) {
        if (clear_tab[slot].key == key) return clear_tab[slot].pipe;
        if (!clear_tab[slot].key) break;
    }
    char fsname[32];
    if (index >= 0) snprintf(fsname, sizeof fsname, "clear.frag.%c%d", kind == 1 ? 'i' : kind == 2 ? 'u' : 'f', index);
    else snprintf(fsname, sizeof fsname, "clear.frag.none");
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vk_builtin_shader("clear.vert"), .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = vk_builtin_shader(fsname), .pName = "main"}};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = (VkSampleCountFlagBits)pass_samples};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cba[8] = {0};
    for (uint32_t i = 0; i < pass_ncolor; i++)
        if ((int)i == index) cba[i].colorWriteMask = mask & 0xF;
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                              .attachmentCount = pass_ncolor, .pAttachments = cba};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                           .dynamicStateCount = NDYN, .pDynamicStates = DYN};
    VkPipelineRenderingCreateInfo rinfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = pass_ncolor,
                                           .pColorAttachmentFormats = pass_color, .depthAttachmentFormat = pass_depth,
                                           .stencilAttachmentFormat = pass_stencil};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rinfo, .stageCount = 2,
                                       .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vs,
                                       .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds,
                                       .pColorBlendState = &cb, .pDynamicState = &dy, .layout = pc_layout()};
    VkPipeline p = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &pi, NULL, &p) != VK_SUCCESS) LOG("[vk] clear pipeline failed\n");
    clear_tab[slot].key = key;
    clear_tab[slot].pipe = p;
    return p;
}

static void clear(int index, int kind, const void *color, uint32_t mask, bool depth, float dval, bool stencil, uint32_t sval,
                  uint32_t smask) {
    begin_pass();
    VkCommandBuffer cb = vk_cmd();
    vp_sx = pass_sx; vp_sy = pass_sy; vp_ox = 0; vp_oy = 0; sc_box = 0;
    VkRect2D rect = scissor_rect();
    if (!rect.extent.width || !rect.extent.height) return;
    if (index >= 0 && (index >= (int)pass_ncolor || pass_color[index] == VK_FORMAT_UNDEFINED)) return;
    bool simple = (index < 0 || (mask & 0xF) == 0xF) && (!stencil || (smask & 0xFF) == 0xFF);
    if (simple) {
        VkClearAttachment a[2];
        uint32_t n = 0;
        if (index >= 0) {
            a[n] = (VkClearAttachment){.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = (uint32_t)index};
            memcpy(a[n].clearValue.color.float32, color, 16);
            n++;
        }
        VkImageAspectFlags dsa = (depth && pass_depth ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (stencil && pass_stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
        if (dsa) a[n++] = (VkClearAttachment){.aspectMask = dsa, .clearValue.depthStencil = {dval, sval}};
        VkClearRect cr = {rect, 0, 1};
        if (n) vkCmdClearAttachments(cb, n, a, 1, &cr);
        return;
    }
    VkPipeline p = clear_pipeline(index, kind, mask, depth, stencil);
    if (!p) return;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    ss_forget();  // (its own pipeline and state: the next draw sets everything again)
    dyn_defaults(cb);
    if (depth) { vkCmdSetDepthTestEnable(cb, VK_TRUE); vkCmdSetDepthWriteEnable(cb, VK_TRUE); }
    if (stencil) {
        vkCmdSetStencilTestEnable(cb, VK_TRUE);
        vkCmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_AND_BACK, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_REPLACE,
                          VK_COMPARE_OP_ALWAYS);
        vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, smask);
        vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, sval);
    }
    int w, h;
    pass_size(&w, &h);
    VkViewport vp = {0, 0, (float)(w ? w : (int)pass_w), (float)(h ? h : (int)pass_h), 0, 1};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &rect);
    float pc[8] = {0};
    if (color) memcpy(pc, color, 16);
    pc[4] = dval;
    vkCmdPushConstants(cb, pc_layout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32, pc);
    vkCmdDraw(cb, 3, 1, 0, 0);
}

// ================================================================== programs
#define MAX_BIND 96
typedef struct {
    ShaderInfo *vi, *fi;
    uint64_t code[2], cmax[2];  // translation inputs (copied: translated on the translator thread)
    ShaderKey vkey, fkey;
    VkShaderModule vs, fs;
    VkBindingInfo binds[MAX_BIND];
    int nbinds;
    VkDescriptorSetLayout set_layout[2];
    VkPipelineLayout layout;
    bool vs_uses_vid;
    uint64_t xh[2];     // the translations' cache hashes (cache/xlat): the program by content, 0 when not cached
    bool warm;          // prebuilt at launch (prewarm below)
    uint64_t waited_ns; // draws' time waiting for it to build (wait_built)
    _Atomic int state;  // 0 building, 1 ready, 2 failed
    void *motion;       // (DLAA) its object-motion variant (a Prog), once asked for: mv_prog
    signed char bf_only[2];  // vertex / fragment: guest heap read by buffer textures only (0 unknown, 1 yes, 2 no)
    bool is_motion;
} Prog;

static VkPipelineLayout make_layout(Prog *p, int nsets, VkShaderStageFlags s0, VkShaderStageFlags s1) {
    p->set_layout[0] = vk_set_layout(p->binds, p->nbinds, 0, s0);
    if (nsets > 1) p->set_layout[1] = vk_set_layout(p->binds, p->nbinds, 1, s1);
    VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = (uint32_t)nsets,
                                     .pSetLayouts = p->set_layout};
    VkPipelineLayout l;
    VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &l));
    return l;
}

static void save_glsl(const char *src, uint64_t addr, const char *suffix) {
    if (!getenv("LARECOMP_SAVE_MSL")) return;
    mkdir("build/glsl", 0755);
    char path[256];
    snprintf(path, sizeof path, "build/glsl/%llx_%s.glsl", (unsigned long long)addr, suffix);
    FILE *f = fopen(path, "w");
    if (f) { fputs(src, f); fclose(f); }
}

// ================================================================== translations kept on disk
// A program's translation (ShaderInfo and its GLSL) depends only on the bytes the translator reads and the key, so it
// is kept in cache/xlat/<hash>.bin and reused by later sessions: the shaders met before skip the translator (one
// thread, its queue the main cause of the draws skipped on entering an area) and are ready a frame or two after first
// use instead of many. The SPIR-V (cache/spirv) and the driver's pipelines (cache/vk_pipelines.bin) were already
// kept. TRANSLATOR_ID (Makefile: a hash of translate.c and shader.h) keys a new translator to new entries.
// LARECOMP_VK_NO_XLAT_CACHE=1 turns it off.
#ifndef TRANSLATOR_ID
#define TRANSLATOR_ID "dev"
#endif
#define XLAT_MAGIC 0x54414C58u  // "XLAT"

static uint64_t xlat_hash(const uint8_t *blob, size_t n, int stage, const ShaderKey *key) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (const char *p = TRANSLATOR_ID; *p; p++) { h ^= (uint8_t)*p; h *= 0x100000001B3ull; }
    // (and the translator's settings: denormals, the guest heap's place)
    uint64_t g[4] = {(uint64_t)stage, maxwell_glsl_denorm_preserve, maxwell_gmem_base, maxwell_gmem_size};
    for (size_t i = 0; i < sizeof g; i++) { h ^= ((const uint8_t *)g)[i]; h *= 0x100000001B3ull; }
    for (size_t i = 0; i < sizeof *key; i++) { h ^= ((const uint8_t *)key)[i]; h *= 0x100000001B3ull; }
    for (size_t i = 0; i < n; i++) { h ^= blob[i]; h *= 0x100000001B3ull; }
    return h;
}

static ShaderInfo *xlat_load(uint64_t h) {
    char path[64];
    snprintf(path, sizeof path, "cache/xlat/%016llx.bin", (unsigned long long)h);
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    uint32_t magic = 0, len = 0, size = 0;
    ShaderInfo *si = calloc(1, sizeof *si);
    bool ok = fread(&magic, 4, 1, f) == 1 && magic == XLAT_MAGIC && fread(&size, 4, 1, f) == 1 && size == sizeof *si &&
              fread(si, sizeof *si, 1, f) == 1 && fread(&len, 4, 1, f) == 1 && len < (16u << 20);
    if (ok) {
        si->msl = malloc(len + 1);
        ok = fread(si->msl, 1, len, f) == len;
        if (ok) si->msl[len] = 0;
    } else si->msl = NULL;
    fclose(f);
    if (!ok) { free(si->msl); free(si); return NULL; }
    return si;
}

static void xlat_store(uint64_t h, const ShaderInfo *si) {
    if (!si->msl) return;  // (failures aren't kept: a fixed translator retries them)
    static bool made;
    if (!made) { made = true; mkdir("cache", 0755); mkdir("cache/xlat", 0755); }
    char path[64], tmp[72];
    snprintf(path, sizeof path, "cache/xlat/%016llx.bin", (unsigned long long)h);
    snprintf(tmp, sizeof tmp, "%s.%llx", path, (unsigned long long)(uintptr_t)si);  // (two threads, one entry)
    FILE *f = fopen(tmp, "wb");
    if (!f) return;
    ShaderInfo copy = *si;
    copy.msl = NULL;
    uint32_t magic = XLAT_MAGIC, size = sizeof copy, len = (uint32_t)strlen(si->msl);
    bool ok = fwrite(&magic, 4, 1, f) == 1 && fwrite(&size, 4, 1, f) == 1 && fwrite(&copy, sizeof copy, 1, f) == 1 &&
              fwrite(&len, 4, 1, f) == 1 && fwrite(si->msl, 1, len, f) == len;
    if (fclose(f) || !ok) { remove(tmp); return; }
    remove(path);
    if (rename(tmp, path)) remove(tmp);
}

// The disk cache's key for a translation, without translating (0: not cached)
static uint64_t xlat_key(const uint8_t *blob, size_t max_size, int stage, const ShaderKey *key) {
    static int off = -1;
    if (off < 0) off = getenv("LARECOMP_VK_NO_XLAT_CACHE") != NULL;
    size_t n = off || maxwell_dbg_code ? 0 : maxwell_input_size(blob, max_size, stage);
    return n ? xlat_hash(blob, n, stage, key) : 0;
}

// maxwell_translate through the disk cache (translate_lock taken here, around the translation only)
static ShaderInfo *translate_cached(const uint8_t *blob, size_t max_size, int stage, const ShaderKey *key, uint64_t *hash) {
    static int off = -1;
    if (off < 0) off = getenv("LARECOMP_VK_NO_XLAT_CACHE") != NULL;
    size_t n = off || maxwell_dbg_code ? 0 : maxwell_input_size(blob, max_size, stage);  // (debugging: not cached)
    uint64_t h = n ? xlat_hash(blob, n, stage, key) : 0;
    if (hash) *hash = h;
    ShaderInfo *si = h ? xlat_load(h) : NULL;
    if (si) { atomic_fetch_add(&xlat_hits, 1); return si; }
    uint64_t t0 = SDL_GetTicksNS();
    pthread_mutex_lock(&translate_lock);
    si = maxwell_translate(blob, max_size, stage, key);
    pthread_mutex_unlock(&translate_lock);
    atomic_fetch_add(&xlat_ns, SDL_GetTicksNS() - t0);
    atomic_fetch_add(&xlat_misses, 1);
    if (h) xlat_store(h, si);
    return si;
}

// FE3H's character shadow mask (fs hash 991d1be5df7b0f7f, 304 instructions): replaced by the same maths in full
// precision - the game reads depth on a 960 x 540 grid and weighs its PCF in half floats (blocky shading on faces at
// high resolutions). c3: [0..3] light matrix, [4..7] second matrix (picked past plane [10]), [8] / [9] atlas
// scale / offset, [11] texel size / map size, [12] PCF offsets, [13] fade, [14..17] inverse view-projection.
// LARECOMP_NO_HQ_SHADOW_MASK=1 keeps the game's.
static const char *k_hq_shadow_mask =
    "#version 460\n"
    "layout(std140, set = 1, binding = 3) uniform CB3 { vec4 C[32]; };\n"
    "layout(std140, set = 1, binding = 18) uniform RScale { vec4 rscale[65]; };\n"
    "layout(std140, set = 1, binding = 19) uniform Tint { vec4 ltint[2]; };\n"
    "layout(set = 1, binding = 32) uniform sampler2D t0;\n"
    "layout(set = 1, binding = 33) uniform sampler2D t1;\n"
    "layout(location = 0) in vec4 v0;\n"
    "layout(location = 0) out vec4 c0;\n"
    "float tap(vec2 q, vec4 at, float z) { return textureLod(t1, at.xy * clamp(q, 0.0, 1.0) + at.zw, 0.0).x < z ? 1.0 : 0.0; }\n"
    "float bil(vec2 q, vec4 at, float z) {\n"
    "  vec2 f = q * C[11].zw - 0.5, i = floor(f), w = f - i, b = (i + 0.5) * C[11].xy, t = C[11].xy;\n"
    "  return mix(mix(tap(b, at, z), tap(b + vec2(t.x, 0), at, z), w.x),\n"
    "             mix(tap(b + vec2(0, t.y), at, z), tap(b + t, at, z), w.x), w.y);\n"
    "}\n"
    "void main() {\n"
    "  vec2 g = (gl_FragCoord.xy - rscale[0].zw) / rscale[0].xy;\n"
    "  ivec2 dp = clamp(ivec2(g * 2.0 * rscale[1].xy), ivec2(0), textureSize(t0, 0) - 1);\n"
    "  float d = texelFetch(t0, dp, 0).x;\n"
    "  vec4 P = v0.x * C[14] + v0.y * C[15] + (2.0 * d - 1.0) * C[16] + C[17];\n"
    "  P /= P.w;\n"
    "  vec4 L = P.x * C[0] + P.y * C[1] + P.z * C[2] + C[3];\n"
    "  bool second = dot(L, C[10]) > 0.0;\n"
    "  vec4 S = second ? P.x * C[4] + P.y * C[5] + P.z * C[6] + C[7] : L;\n"
    "  vec4 at = second ? C[9] : C[8];\n"
    "  vec2 uv = S.xy / S.w;\n"
    "  float z = min(S.z / S.w, 1.0);\n"
    "  vec2 o = uv + C[12].xy;\n"
    "  float s = bil(o, at, z) + bil(o + vec2(C[12].z, 0), at, z) + bil(o + vec2(0, C[12].w), at, z) + bil(o + C[12].zw, at, z);\n"
    "  float r = 1.0 - 0.25 * s * clamp(dot(L, C[13]), 0.0, 1.0);\n"
    "  c0 = vec4(r);\n"
    "  if (ltint[0].w > 0.0) { vec3 q = c0.rgb; float l = dot(q, vec3(0.299, 0.587, 0.114));\n"
    "    c0.rgb = mix(q, ltint[0].rgb * pow(l, ltint[1].w) + ltint[1].rgb * l, ltint[0].w); }\n"
    "}\n";
static const char *hq_fs_override(const ShaderInfo *fi) {
    static int off = -1;
    if (off < 0) off = getenv("LARECOMP_NO_HQ_SHADOW_MASK") != NULL;
    if (off || !fi || !fi->msl || fi->num_insts != 304 || fi->ntex != 2 || fi->cbuf_mask != (1u << 3)) return NULL;
    if (!strstr(fi->msl, "cb3[16][3]") || !strstr(fi->msl, "cb3[11][3]") || !strstr(fi->msl, "cb3[10][3]")) return NULL;
    return k_hq_shadow_mask;
}

static void compile_program(void *arg) {
    Prog *p = arg;
    size_t nv = 0, nf = 0;
    uint64_t t0 = SDL_GetTicksNS();
    const char *fsrc = hq_fs_override(p->fi);
    if (fsrc) LOG("[shader] character shadow mask: full-precision version\n");
    uint32_t *sv = vk_glsl_to_spirv(p->vi->msl, STAGE_VERTEX, "vs", &nv);
    uint32_t *sf = sv ? vk_glsl_to_spirv(fsrc ? fsrc : p->fi->msl, STAGE_FRAGMENT, "fs", &nf) : NULL;
    if (!sv || !sf) { free(sv); free(sf); set_state(&p->state, 2); return; }
    p->nbinds = vk_reflect(sv, nv, p->binds, MAX_BIND);
    p->nbinds += vk_reflect(sf, nf, p->binds + p->nbinds, MAX_BIND - p->nbinds);
    p->vs = vk_module(sv, nv);
    p->fs = vk_module(sf, nf);
    free(sv);
    free(sf);
    p->layout = make_layout(p, 2, VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT);
    slow_build_log("program vs/fs", p->code[1] ? p->code[1] : p->xh[1], p->fi->num_insts, t0, SDL_GetTicksNS());
    atomic_fetch_add(&stat_compiles, 1);
    set_state(&p->state, p->vs && p->fs ? 1 : 2);
}

static void translate_program(void *arg) {
    Prog *p = arg;
    maxwell_gmem_base = HEAP_BASE;
    maxwell_gmem_size = HEAP_MAX;
    p->vi = translate_cached((const uint8_t *)(uintptr_t)p->code[0], p->cmax[0], STAGE_VERTEX, &p->vkey, &p->xh[0]);
    if (p->vi->msl) {
        p->fkey.vs_out_mask = p->vi->out_mask;
        p->fi = translate_cached((const uint8_t *)(uintptr_t)p->code[1], p->cmax[1], STAGE_FRAGMENT, &p->fkey, &p->xh[1]);
    }
    if (!p->vi->msl) {
        LOG("[shader] VS translate failed: %s (code %llx)\n", p->vi->error, (unsigned long long)p->code[0]);
        set_state(&p->state, 2);
        return;
    }
    if (!p->fi->msl) {
        LOG("[shader] FS translate failed: %s\n", p->fi->error);
        set_state(&p->state, 2);
        return;
    }
    p->vs_uses_vid = strstr(p->vi->msl, "gl_VertexIndex") != NULL;
    save_glsl(p->vi->msl, p->code[0], "vs");
    save_glsl(p->fi->msl, p->code[1], "fs");
    run_job(compile_program, p);
}

#define PROG_TAB 4096
static struct { uint64_t key; Prog *prog; } prog_tab[PROG_TAB];

static NvnTexture *shader_texture_object(int stage, ShaderTexture *st_);
// Texture slots 0-31 of a stage's shader that are 1D in the shader and bound to buffer textures (NVN target 10).
static uint32_t buffer_texture_mask(int stage, const ShaderInfo *si) {
    uint32_t m = 0;
    for (int i = 0; i < si->ntex && i < 32; i++) {
        if (si->tex[i].type != TEXTYPE_1D || si->tex[i].is_image) continue;
        NvnTexture *t = shader_texture_object(stage, (ShaderTexture *)&si->tex[i]);
        if (t && t->target == 10) m |= 1u << i;
    }
    return m;
}

static Prog *warm_find(NvnProgram *np, const ShaderKey *vkey, const ShaderKey *fkey);
static void warm_start(void);
static Prog *mv_prog(Prog *p);
static void mv_prepare(const Prog *p, const CmdRec *r, const ShaderKey *vk);
static Prog *get_program(NvnProgram *np, ShaderKey *vkey, ShaderKey *fkey) {
    if (!(np->stage_mask & 1) || !(np->stage_mask & 2)) return NULL;
    warm_start();
    uint64_t h = 0xcbf29ce484222325ull;
    for (int i = 0; i < 2; i++) { uint64_t a = np->stages[i].code_addr; for (int k = 0; k < 8; k++) { h ^= (a >> (k * 8)) & 0xFF; h *= 0x100000001B3ull; } }
    const uint8_t *vb = (const uint8_t *)vkey;
    for (size_t i = 0; i < sizeof *vkey; i++) { h ^= vb[i]; h *= 0x100000001B3ull; }
    for (int i = 0; i < 8; i++) { h ^= fkey->rt_kind[i]; h *= 0x100000001B3ull; }
    if (fkey->attr_kind[0]) { h ^= fkey->attr_kind[0]; h *= 0x100000001B3ull; h ^= fkey->attr_kind[1]; h *= 0x100000001B3ull; }
    if (key_buf_mask(fkey)) {  // (buffer textures: only keys that have them hash differently)
        uint32_t m = key_buf_mask(fkey);
        for (int i = 0; i < 4; i++) { h ^= (m >> (i * 8)) & 0xFF; h *= 0x100000001B3ull; }
    }
    if (!h) h = 1;
    uint32_t slot = (uint32_t)(h ^ (h >> 31)) & (PROG_TAB - 1);
    while (prog_tab[slot].key && prog_tab[slot].key != h) slot = (slot + 1) & (PROG_TAB - 1);
    if (prog_tab[slot].key) {
        Prog *q = prog_tab[slot].prog;
        if (!q->warm || atomic_load(&q->state) != 2) return q;
        // (a prebuilt one that failed - its cached translation gone: the game's own build instead)
    }
    Prog *w = prog_tab[slot].key ? NULL : warm_find(np, vkey, fkey);  // (built at launch from an earlier session's)
    if (w && atomic_load(&w->state) == 2) w = NULL;
    if (w) {
        prog_tab[slot].key = h;
        prog_tab[slot].prog = w;
        return w;
    }
    Prog *p = calloc(1, sizeof *p);
    prog_tab[slot].key = h;
    prog_tab[slot].prog = p;
    for (int i = 0; i < 2; i++) {
        p->code[i] = np->stages[i].code_addr;
        p->cmax[i] = np->stages[i].const_offset ? np->stages[i].const_offset : 0x10000;
    }
    p->vkey = *vkey;
    p->fkey = *fkey;
    run_translate(translate_program, p);
    return p;
}

// ================================================================== vertex formats / fixed function tables
static int attr_kind(uint32_t fmt) {
    switch (fmt) {
    case 0x03: case 0x08: case 0x0B: case 0x0F: case 0x14: case 0x17: case 0x1B: case 0x20: case 0x23:
    case 0x27: case 0x2C: case 0x2F: case 0x3E:
        return 2;
    case 0x04: case 0x09: case 0x0C: case 0x10: case 0x15: case 0x18: case 0x1C: case 0x21: case 0x24:
    case 0x28: case 0x2D: case 0x30:
        return 1;
    default: return 0;
    }
}

static VkFormat vertex_format(uint32_t f) {
    switch (f) {
    case 0x01: return VK_FORMAT_R8_UNORM;
    case 0x02: return VK_FORMAT_R8_SNORM;
    case 0x03: return VK_FORMAT_R8_UINT;
    case 0x04: return VK_FORMAT_R8_SINT;
    case 0x05: return VK_FORMAT_R16_SFLOAT;
    case 0x06: return VK_FORMAT_R16_UNORM;
    case 0x07: return VK_FORMAT_R16_SNORM;
    case 0x08: return VK_FORMAT_R16_UINT;
    case 0x09: return VK_FORMAT_R16_SINT;
    case 0x0A: return VK_FORMAT_R32_SFLOAT;
    case 0x0B: return VK_FORMAT_R32_UINT;
    case 0x0C: return VK_FORMAT_R32_SINT;
    case 0x0D: return VK_FORMAT_R8G8_UNORM;
    case 0x0E: return VK_FORMAT_R8G8_SNORM;
    case 0x0F: return VK_FORMAT_R8G8_UINT;
    case 0x10: return VK_FORMAT_R8G8_SINT;
    case 0x11: return VK_FORMAT_R16G16_SFLOAT;
    case 0x12: return VK_FORMAT_R16G16_UNORM;
    case 0x13: return VK_FORMAT_R16G16_SNORM;
    case 0x14: return VK_FORMAT_R16G16_UINT;
    case 0x15: return VK_FORMAT_R16G16_SINT;
    case 0x16: return VK_FORMAT_R32G32_SFLOAT;
    case 0x17: return VK_FORMAT_R32G32_UINT;
    case 0x18: return VK_FORMAT_R32G32_SINT;
    case 0x19: return VK_FORMAT_R8G8B8_UNORM;
    case 0x1A: return VK_FORMAT_R8G8B8_SNORM;
    case 0x1B: return VK_FORMAT_R8G8B8_UINT;
    case 0x1C: return VK_FORMAT_R8G8B8_SINT;
    case 0x1D: return VK_FORMAT_R16G16B16_SFLOAT;
    case 0x1E: return VK_FORMAT_R16G16B16_UNORM;
    case 0x1F: return VK_FORMAT_R16G16B16_SNORM;
    case 0x20: return VK_FORMAT_R16G16B16_UINT;
    case 0x21: return VK_FORMAT_R16G16B16_SINT;
    case 0x22: return VK_FORMAT_R32G32B32_SFLOAT;
    case 0x23: return VK_FORMAT_R32G32B32_UINT;
    case 0x24: return VK_FORMAT_R32G32B32_SINT;
    case 0x25: return VK_FORMAT_R8G8B8A8_UNORM;
    case 0x26: return VK_FORMAT_R8G8B8A8_SNORM;
    case 0x27: return VK_FORMAT_R8G8B8A8_UINT;
    case 0x28: return VK_FORMAT_R8G8B8A8_SINT;
    case 0x29: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case 0x2A: return VK_FORMAT_R16G16B16A16_UNORM;
    case 0x2B: return VK_FORMAT_R16G16B16A16_SNORM;
    case 0x2C: return VK_FORMAT_R16G16B16A16_UINT;
    case 0x2D: return VK_FORMAT_R16G16B16A16_SINT;
    case 0x2E: return VK_FORMAT_R32G32B32A32_SFLOAT;
    case 0x2F: return VK_FORMAT_R32G32B32A32_UINT;
    case 0x30: return VK_FORMAT_R32G32B32A32_SINT;
    case 0x3D: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case 0x69: return VK_FORMAT_A2B10G10R10_SNORM_PACK32;
    case 0x3F: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case 0x40: return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
    case 0x51: return VK_FORMAT_R8_UINT;
    default: return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
}

// 3-component 8/16-bit vertex formats aren't universally supported: read 4 components instead.
static VkFormat vertex_format_supported(VkFormat f) {
    static uint8_t known[256];  // 1 ok, 2 not
    uint32_t i = f < 256 ? f : 0;
    if (!known[i]) {
        VkFormatProperties p;
        vkGetPhysicalDeviceFormatProperties(g_vk.phys, f, &p);
        known[i] = (p.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT) ? 1 : 2;
    }
    if (known[i] == 1) return f;
    { static uint8_t told[256]; if (!told[i]++) LOG("[vk] vertex format %d unsupported: read as 4 components\n", f); }
    switch (f) {
    case VK_FORMAT_R8G8B8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_R8G8B8_SNORM: return VK_FORMAT_R8G8B8A8_SNORM;
    case VK_FORMAT_R8G8B8_UINT: return VK_FORMAT_R8G8B8A8_UINT;
    case VK_FORMAT_R8G8B8_SINT: return VK_FORMAT_R8G8B8A8_SINT;
    case VK_FORMAT_R16G16B16_SFLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case VK_FORMAT_R16G16B16_UNORM: return VK_FORMAT_R16G16B16A16_UNORM;
    case VK_FORMAT_R16G16B16_SNORM: return VK_FORMAT_R16G16B16A16_SNORM;
    case VK_FORMAT_R16G16B16_UINT: return VK_FORMAT_R16G16B16A16_UINT;
    case VK_FORMAT_R16G16B16_SINT: return VK_FORMAT_R16G16B16A16_SINT;
    default: return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
}

static VkBlendFactor blend_factor(uint8_t f) {
    switch (f) {
    case 1: return VK_BLEND_FACTOR_ZERO;
    case 2: return VK_BLEND_FACTOR_ONE;
    case 3: return VK_BLEND_FACTOR_SRC_COLOR;
    case 4: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case 5: return VK_BLEND_FACTOR_SRC_ALPHA;
    case 6: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case 7: return VK_BLEND_FACTOR_DST_ALPHA;
    case 8: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case 9: return VK_BLEND_FACTOR_DST_COLOR;
    case 10: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case 11: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case 0x10: return VK_BLEND_FACTOR_SRC1_COLOR;
    case 0x11: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
    case 0x12: return VK_BLEND_FACTOR_SRC1_ALPHA;
    case 0x13: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
    case 0x61: return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case 0x62: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case 0x63: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
    case 0x64: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    default: {
        static uint8_t warned[256];
        if (!warned[f]++) LOG("[vk] unknown blend factor %#x\n", f);
        return VK_BLEND_FACTOR_ONE;
    }
    }
}

static VkBlendOp blend_op(uint8_t e) {
    switch (e) {
    case 2: return VK_BLEND_OP_SUBTRACT;
    case 3: return VK_BLEND_OP_REVERSE_SUBTRACT;
    case 4: return VK_BLEND_OP_MIN;
    case 5: return VK_BLEND_OP_MAX;
    default: return VK_BLEND_OP_ADD;
    }
}

static VkCompareOp compare_fn(uint8_t f) {
    switch (f) {
    case 1: return VK_COMPARE_OP_NEVER;
    case 2: return VK_COMPARE_OP_LESS;
    case 3: return VK_COMPARE_OP_EQUAL;
    case 4: return VK_COMPARE_OP_LESS_OR_EQUAL;
    case 5: return VK_COMPARE_OP_GREATER;
    case 6: return VK_COMPARE_OP_NOT_EQUAL;
    case 7: return VK_COMPARE_OP_GREATER_OR_EQUAL;
    default: return VK_COMPARE_OP_ALWAYS;
    }
}

static VkStencilOp stencil_op(uint8_t o) {
    switch (o) {
    case 2: return VK_STENCIL_OP_ZERO;
    case 3: return VK_STENCIL_OP_REPLACE;
    case 4: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
    case 5: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
    case 6: return VK_STENCIL_OP_INVERT;
    case 7: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
    case 8: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
    default: return VK_STENCIL_OP_KEEP;
    }
}

static int rt_kind(uint32_t f) {
    switch (f) {
    case 0x03: case 0x08: case 0x0B: case 0x0F: case 0x14: case 0x17: case 0x27: case 0x2C: case 0x2F: case 0x3E: return 2;
    case 0x04: case 0x09: case 0x0C: case 0x10: case 0x15: case 0x18: case 0x28: case 0x2D: case 0x30: return 1;
    default: return 0;
    }
}

// ================================================================== graphics pipelines
// Everything a pipeline bakes in, captured by value so it can be built on a worker thread.
#define ZERO_BINDING 16
typedef struct {
    Prog *prog;
    VkFormat color[8], depth, stencil;
    uint32_t ncolor, samples;
    uint32_t attr_fmt[32], attr_off[32], attr_bind[32];
    uint32_t attr_mask;
    uint32_t stride[17], divisor[17], used_streams;
    uint8_t blend[8][8];
    uint8_t blend_on, a2c, topo_class, poly_fill;
    uint32_t write_mask[8];
} PipeKey;

typedef struct { PipeKey k; uint64_t hash; VkPipeline pipe; _Atomic int state; uint64_t waited_ns; } Pipe;  // state: 0 building, 1 ready, 2 failed
#define PIPE_TAB 32768
static struct { uint64_t key; Pipe *pipe; } pipe_tab[PIPE_TAB];

// 8 bytes a step (PipeKey is ~700 bytes, hashed per draw); in-memory tables only - hmix values are compared with
// stored constants (fragment-shader hashes) and must not change.
static uint64_t hash_words(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = p;
    for (; n >= 8; n -= 8, b += 8) { uint64_t w; memcpy(&w, b, 8); h = (h ^ w) * 0x9E3779B97F4A7C15ull; h ^= h >> 29; }
    for (; n; n--, b++) { h = (h ^ *b) * 0x100000001B3ull; }
    return h ^ (h >> 32);
}

static uint64_t hmix(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001B3ull; }
    return h;
}

static VkPrimitiveTopology topo_class_base(uint8_t c) {
    return c == 0 ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST : c == 1 ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
}

static void warm_record(const Pipe *pp);
static void build_pipeline(void *arg) {
    Pipe *pp = arg;
    PipeKey *k = &pp->k;
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = k->prog->vs, .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = k->prog->fs, .pName = "main"}};
    VkVertexInputAttributeDescription va[32];
    VkVertexInputBindingDescription vb[17];
    VkVertexInputBindingDivisorDescriptionEXT vd[17];
    uint32_t na = 0, nb = 0, nd = 0;
    for (int i = 0; i < 32; i++)
        if (k->attr_mask & (1u << i)) va[na++] = (VkVertexInputAttributeDescription){(uint32_t)i, k->attr_bind[i], k->attr_fmt[i], k->attr_off[i]};
    for (uint32_t s = 0; s < 17; s++) {
        if (!(k->used_streams & (1u << s))) continue;
        bool inst = k->divisor[s] != 0;
        vb[nb++] = (VkVertexInputBindingDescription){s, k->stride[s], inst ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX};
        if (inst && k->divisor[s] > 1 && g_vk.divisor) vd[nd++] = (VkVertexInputBindingDivisorDescriptionEXT){s, k->divisor[s]};
    }
    VkPipelineVertexInputDivisorStateCreateInfoEXT dvi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT,
                                                          .vertexBindingDivisorCount = nd, .pVertexBindingDivisors = vd};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, .pNext = nd ? &dvi : NULL,
                                               .vertexBindingDescriptionCount = nb, .pVertexBindingDescriptions = vb,
                                               .vertexAttributeDescriptionCount = na, .pVertexAttributeDescriptions = va};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = topo_class_base(k->topo_class)};
    VkPipelineViewportStateCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = k->poly_fill ? VK_POLYGON_MODE_FILL : VK_POLYGON_MODE_LINE, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = (VkSampleCountFlagBits)k->samples, .alphaToCoverageEnable = k->a2c};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cba[8] = {0};
    for (uint32_t i = 0; i < k->ncolor; i++) {
        cba[i].colorWriteMask = k->write_mask[i];
        if ((k->blend_on >> i) & 1) {
            const uint8_t *b = k->blend[i];
            cba[i].blendEnable = VK_TRUE;
            cba[i].srcColorBlendFactor = blend_factor(b[1]);
            cba[i].dstColorBlendFactor = blend_factor(b[2]);
            cba[i].srcAlphaBlendFactor = blend_factor(b[3]);
            cba[i].dstAlphaBlendFactor = blend_factor(b[4]);
            cba[i].colorBlendOp = blend_op(b[5]);
            cba[i].alphaBlendOp = blend_op(b[6]);
        }
    }
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                              .attachmentCount = k->ncolor, .pAttachments = cba};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                           .dynamicStateCount = NDYN, .pDynamicStates = DYN};
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = k->ncolor,
                                        .pColorAttachmentFormats = k->color, .depthAttachmentFormat = k->depth,
                                        .stencilAttachmentFormat = k->stencil};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri, .stageCount = 2,
                                       .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vs,
                                       .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds,
                                       .pColorBlendState = &cb, .pDynamicState = &dy, .layout = k->prog->layout};
    uint64_t t0 = SDL_GetTicksNS();
    VkResult r = vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &pi, NULL, &pp->pipe);
    slow_build_log("pipeline of vs/fs", k->prog->code[1] ? k->prog->code[1] : k->prog->xh[1], k->prog->fi ? k->prog->fi->num_insts : 0, t0, t0);
    if (r != VK_SUCCESS) {
        static int n;
        if (n++ < 10) LOG("[vk] pipeline creation failed: %d\n", r);
    }
    if (r == VK_SUCCESS) warm_record(pp);
    atomic_fetch_add(&stat_compiles, 1);
    set_state(&pp->state, r == VK_SUCCESS ? 1 : 2);
}

// ================================================================== prewarm: last sessions' pipelines, built at launch
// Every pipeline built is recorded (cache/warm.bin) by content: its two translations' cache hashes (cache/xlat) and
// its PipeKey. At the first draw of a session (the game's first loading), each recorded program and its pipelines
// are built again on the workers, at low priority, from the caches on disk. A program the game then binds is
// recognised by its translations' hashes (worked out on the spot: no translation) and the prebuilt one is used, with
// its pipelines: an area entered before draws everything at once, with no pop-in and no wait. New ones are recorded
// as they come. LARECOMP_VK_NO_PREWARM=1 turns it off (LARECOMP_VK_NO_XLAT_CACHE=1 too).
#define WARM_MAGIC 0x4D524157u  // "WARM"
#define WARM_VERSION 1
#define WARM_PROGS 8192         // (table sizes; at most half of each used)
#define WARM_PIPES 16384
typedef struct { uint64_t xh[2]; uint32_t vs_out_mask, pad; PipeKey k; } WarmRec;  // (k.prog NULL on disk)
typedef struct { uint64_t vxh, fxh; Prog *p; uint32_t vs_out_mask; Pipe **pipes; int npipes, cap; _Atomic int taken; } WarmProg;
static WarmProg warm_progs[WARM_PROGS];
static int nwarm_progs;
static Pipe **warm_pipes;   // every prebuilt pipeline (put back in pipe_tab after a reset)
static int nwarm_pipes;
static uint64_t warm_seen[WARM_PIPES * 2];  // records written or loaded, by hash (no duplicates in the file)
static int nwarm_seen;
static pthread_mutex_t warm_lock = PTHREAD_MUTEX_INITIALIZER;
static FILE *warm_file;
static _Atomic int warm_left;
static uint64_t warm_t0;

static bool warm_off(void) {
    static int off = -1;
    if (off < 0) off = getenv("LARECOMP_VK_NO_PREWARM") || getenv("LARECOMP_VK_NO_XLAT_CACHE") || sync_shaders();
    return off;
}

static uint64_t pair_hash(uint64_t a, uint64_t b) { uint64_t h = a * 0x9E3779B97F4A7C15ull ^ b; return h ^ (h >> 31); }

static WarmProg *warm_prog_slot(uint64_t vxh, uint64_t fxh, bool add) {
    uint32_t i = (uint32_t)pair_hash(vxh, fxh) & (WARM_PROGS - 1);
    while (warm_progs[i].vxh && (warm_progs[i].vxh != vxh || warm_progs[i].fxh != fxh)) i = (i + 1) & (WARM_PROGS - 1);
    if (warm_progs[i].vxh) return &warm_progs[i];
    if (!add || nwarm_progs >= WARM_PROGS / 2) return NULL;
    nwarm_progs++;
    warm_progs[i].vxh = vxh;
    warm_progs[i].fxh = fxh;
    return &warm_progs[i];
}

// a vertex shader's output mask (the fragment shader's key depends on it), from the recorded programs that use it
static bool warm_vs_mask(uint64_t vxh, uint32_t *mask) {
    for (int i = 0; i < WARM_PROGS; i++)  // (only while the game meets a program for the first time this session)
        if (warm_progs[i].vxh == vxh) { *mask = warm_progs[i].vs_out_mask; return true; }
    return false;
}

static bool warm_seen_add(uint64_t h) {  // (warm_lock held) false: already there
    if (!h) h = 1;
    uint32_t i = (uint32_t)(h ^ (h >> 29)) & (WARM_PIPES * 2 - 1);
    while (warm_seen[i] && warm_seen[i] != h) i = (i + 1) & (WARM_PIPES * 2 - 1);
    if (warm_seen[i] || nwarm_seen >= WARM_PIPES) return false;
    warm_seen[i] = h;
    nwarm_seen++;
    return true;
}

static void warm_record(const Pipe *pp) {
    const Prog *pr = pp->k.prog;
    if (warm_off() || !pr || !pr->xh[0] || !pr->xh[1] || !pr->vi) return;
    WarmRec r;
    memset(&r, 0, sizeof r);
    r.xh[0] = pr->xh[0];
    r.xh[1] = pr->xh[1];
    r.vs_out_mask = pr->vi->out_mask;
    r.k = pp->k;
    r.k.prog = NULL;
    uint64_t h = hash_words(0xcbf29ce484222325ull, &r, sizeof r);
    pthread_mutex_lock(&warm_lock);
    if (warm_seen_add(h)) {
        if (!warm_file) {
            mkdir("cache", 0755);
            warm_file = fopen("cache/warm.bin", "ab");
            if (warm_file && (fseek(warm_file, 0, SEEK_END), ftell(warm_file)) == 0) {
                uint32_t hdr[4] = {WARM_MAGIC, WARM_VERSION, (uint32_t)sizeof r, 0};
                fwrite(hdr, sizeof hdr, 1, warm_file);
            }
        }
        if (warm_file) { fwrite(&r, sizeof r, 1, warm_file); fflush(warm_file); }
    }
    pthread_mutex_unlock(&warm_lock);
}

static void warm_insert(Pipe *pp) {  // into pipe_tab, as get_pipeline would
    uint64_t ph = hash_words(0xcbf29ce484222325ull, &pp->k, sizeof pp->k);
    if (!ph) ph = 1;
    uint32_t s = (uint32_t)(ph ^ (ph >> 29)) & (PIPE_TAB - 1);
    while (pipe_tab[s].key && pipe_tab[s].key != ph) s = (s + 1) & (PIPE_TAB - 1);
    if (pipe_tab[s].key) return;
    pipe_tab[s].key = ph;
    pipe_tab[s].pipe = pp;
}

static void warm_done(int n) {
    if (atomic_fetch_sub(&warm_left, n) == n)
        LOG("[shader] prewarm done: %d programs, %d pipelines in %.1f s\n", nwarm_progs, nwarm_pipes,
            (SDL_GetTicksNS() - warm_t0) / 1e9);
}

static void warm_build(void *arg) {  // one recorded program and its pipelines (a worker, low priority)
    WarmProg *w = arg;
    if (atomic_exchange(&w->taken, 1)) return;  // (already built: the game needed it first, warm_find)
    Prog *p = w->p;
    p->vi = xlat_load(w->vxh);
    p->fi = p->vi ? xlat_load(w->fxh) : NULL;
    if (p->vi && p->fi && p->vi->msl && p->fi->msl) {
        p->vs_uses_vid = strstr(p->vi->msl, "gl_VertexIndex") != NULL;
        compile_program(p);
    } else {
        if (!p->vi) p->vi = calloc(1, sizeof *p->vi);  // (gone from cache/xlat: never ready, the game's own builds instead)
        if (!p->fi) p->fi = calloc(1, sizeof *p->fi);
        set_state(&p->state, 2);
    }
    for (int i = 0; i < w->npipes; i++) {
        if (atomic_load(&p->state) == 1) build_pipeline(w->pipes[i]);
        else set_state(&w->pipes[i]->state, 2);
    }
    warm_done(1);
}

// The program the game binds, if it was prebuilt: its translations' hashes as translate_program would compute them
static Prog *warm_find(NvnProgram *np, const ShaderKey *vkey, const ShaderKey *fkey) {
    if (!nwarm_progs || maxwell_dbg_code) return NULL;
    maxwell_gmem_base = HEAP_BASE;  // (as translate_program: part of the hash)
    maxwell_gmem_size = HEAP_MAX;
    const uint8_t *vc = (const uint8_t *)(uintptr_t)np->stages[0].code_addr, *fc = (const uint8_t *)(uintptr_t)np->stages[1].code_addr;
    size_t vmax = np->stages[0].const_offset ? np->stages[0].const_offset : 0x10000;
    size_t fmax = np->stages[1].const_offset ? np->stages[1].const_offset : 0x10000;
    size_t vn = maxwell_input_size(vc, vmax, STAGE_VERTEX);
    if (!vn) return NULL;
    uint64_t vxh = xlat_hash(vc, vn, STAGE_VERTEX, vkey);
    ShaderKey fk = *fkey;
    if (!warm_vs_mask(vxh, &fk.vs_out_mask)) return NULL;
    size_t fn = maxwell_input_size(fc, fmax, STAGE_FRAGMENT);
    if (!fn) return NULL;
    WarmProg *w = warm_prog_slot(vxh, xlat_hash(fc, fn, STAGE_FRAGMENT, &fk), false);
    if (!w) return NULL;
    if (!atomic_load(&w->taken)) run_job(warm_build, w);  // (needed now: ahead of the rest of the prewarm)
    return w->p;
}

// At the first draw: the recorded programs and pipelines, queued to be built
static void warm_cs_start(void);
static void warm_start(void) {
    static bool started;
    if (started || warm_off()) return;
    started = true;
    warm_cs_start();
    FILE *f = fopen("cache/warm.bin", "rb");
    if (!f) return;
    uint32_t hdr[4] = {0};
    if (fread(hdr, sizeof hdr, 1, f) != 1 || hdr[0] != WARM_MAGIC || hdr[1] != WARM_VERSION || hdr[2] != sizeof(WarmRec)) {
        fclose(f);
        LOG("[shader] cache/warm.bin is from another version: starting a new one\n");
        remove("cache/warm.bin");
        return;
    }
    warm_t0 = SDL_GetTicksNS();
    WarmRec r;
    warm_pipes = calloc(WARM_PIPES, sizeof *warm_pipes);
    WarmRec *live = malloc(WARM_PIPES * sizeof *live);  // (records whose translations are still cached)
    int nlive = 0, ndead = 0;
    pthread_mutex_lock(&warm_lock);
    while (fread(&r, sizeof r, 1, f) == 1 && nwarm_pipes < WARM_PIPES) {
        // (a translator update retires every cached translation: their records go too, rewritten below)
        bool have = warm_prog_slot(r.xh[0], r.xh[1], false) != NULL;
        if (!have) {
            char p0[64], p1[64];
            struct stat sb;
            snprintf(p0, sizeof p0, "cache/xlat/%016llx.bin", (unsigned long long)r.xh[0]);
            snprintf(p1, sizeof p1, "cache/xlat/%016llx.bin", (unsigned long long)r.xh[1]);
            if (stat(p0, &sb) || stat(p1, &sb)) { ndead++; continue; }
        }
        if (!warm_seen_add(hash_words(0xcbf29ce484222325ull, &r, sizeof r))) continue;
        WarmProg *w = warm_prog_slot(r.xh[0], r.xh[1], true);
        if (!w) continue;
        live[nlive++] = r;
        if (!w->p) {
            w->p = calloc(1, sizeof *w->p);
            w->p->warm = true;
            w->p->xh[0] = r.xh[0];
            w->p->xh[1] = r.xh[1];
            w->vs_out_mask = r.vs_out_mask;
        }
        Pipe *pp = calloc(1, sizeof *pp);
        pp->k = r.k;
        pp->k.prog = w->p;
        if (w->npipes == w->cap) { w->cap = w->cap ? w->cap * 2 : 4; w->pipes = realloc(w->pipes, w->cap * sizeof *w->pipes); }
        w->pipes[w->npipes++] = pp;
        warm_pipes[nwarm_pipes++] = pp;
        warm_insert(pp);
    }
    fclose(f);
    if (ndead) {  // (the file without them)
        FILE *o = fopen("cache/warm.bin.tmp", "wb");
        bool ok = o && fwrite(hdr, sizeof hdr, 1, o) == 1 && (int)fwrite(live, sizeof *live, nlive, o) == nlive;
        if (o && fclose(o)) ok = false;
        if (ok) { remove("cache/warm.bin"); ok = !rename("cache/warm.bin.tmp", "cache/warm.bin"); }
        if (!ok) remove("cache/warm.bin.tmp");
        LOG("[shader] prewarm: %d records dropped (their translations are from an older translator)\n", ndead);
    }
    pthread_mutex_unlock(&warm_lock);
    free(live);
    if (!nwarm_progs) return;
    LOG("[shader] prewarm: building %d programs, %d pipelines from earlier sessions\n", nwarm_progs, nwarm_pipes);
    atomic_store(&warm_left, nwarm_progs);
    for (int i = 0; i < WARM_PROGS; i++)
        if (warm_progs[i].p) run_job_at(warm_build, &warm_progs[i], true);
}

// ================================================================== feedback snapshots (renderer.m resolve_feedback)
static NvnTexture *feedback_tex[4];
static VkImageView feedback_view[4];
static int nfeedback;
static uint64_t g_cur_fs_hash;
static NvnTexture *snap_tex[8];
static bool snap_ok[8];
static const char *snap_why[8];
static const char *snap_reason = "?";
static void snapshots_invalidate(NvnTexture *t) {
    for (int i = 0; i < 8; i++) if ((!t || snap_tex[i] == t) && snap_ok[i]) { snap_ok[i] = false; snap_why[i] = snap_reason; }
}

typedef struct { NvnTexture *key; TexBackend copy; } Snap;
static Snap snaps[16];
static int nsnaps;

static NvnTexture *shader_texture_object(int stage, ShaderTexture *st_) {
    int slot = ((int)st_->handle_offset - 0x20) / 8;
    uint64_t h = (slot >= 0 && slot < 32) ? st.tex[stage][slot] : 0;
    uint32_t tic = h & 0xFFFFF;
    if (!h || !st.tpool || tic >= (uint32_t)st.tpool->count) return NULL;
    return ((NvnTexEntry *)st.tpool->entries)[tic].tex;
}

static bool is_attachment(NvnTexture *t) {
    if (!t) return false;
    if (t == st.rt.depth) return true;
    for (int i = 0; i < st.rt.ncolor; i++) if (t == st.rt.color[i]) return true;
    return false;
}

static TexBackend *snapshot_image(NvnTexture *t, TexBackend *src) {
    Snap *s = NULL;
    for (int i = 0; i < nsnaps; i++) if (snaps[i].key == t) s = &snaps[i];
    if (!s) {
        if (nsnaps == 16) { static int rr; s = &snaps[rr++ & 15]; } else s = &snaps[nsnaps++];
        s->key = t;
    }
    TexBackend *d = &s->copy;
    if (d->image && (d->width != src->width || d->height != src->height || d->format != src->format)) {
        vk_views_forget(d);  // (resolution change)
        vk_defer_image(d->image, d->mem);
        d->image = VK_NULL_HANDLE;
        d->mem = VK_NULL_HANDLE;
    }
    if (!d->image) {
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = src->format,
                                .extent = {src->width, src->height, 1}, .mipLevels = 1, .arrayLayers = 1,
                                .samples = (VkSampleCountFlagBits)src->samples, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &d->image, &d->mem, NULL));
        d->format = src->format; d->width = src->width; d->height = src->height; d->depth = 1; d->layers = 1; d->levels = 1;
        d->samples = src->samples; d->aspect = src->aspect; d->base_type = VK_IMAGE_VIEW_TYPE_2D;
        vk_image_init_layout(vk_cmd(), d->image, d->aspect);
    }
    return d;
}

// (debug frames) a depth image's depth values as a greyscale PGM, stretched over its own min..max
static void debug_save_depth(TexBackend *b, const char *path) {
    if (!b || !b->image || !(b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) || b->format != VK_FORMAT_D32_SFLOAT_S8_UINT) return;
    uint64_t n = (uint64_t)b->width * b->height;
    VkSpan out = vk_readback(n * 4);
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkBufferImageCopy c = {.bufferOffset = out.off, .imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1},
                           .imageExtent = {b->width, b->height, 1}};
    vkCmdCopyImageToBuffer(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, out.buf, 1, &c);
    renderer_flush();
    const float *d = out.ptr;
    float lo = 1e30f, hi = -1e30f;
    for (uint64_t i = 0; i < n; i++) { if (d[i] < lo) lo = d[i]; if (d[i] > hi) hi = d[i]; }
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P5 %u %u 255\n", b->width, b->height);
    for (uint64_t i = 0; i < n; i++) fputc(hi > lo ? (int)((d[i] - lo) / (hi - lo) * 255.0f) : 0, f);
    fclose(f);
    LOG("[vk] saved %s (%ux%u depth %.6f..%.6f)\n", path, b->width, b->height, lo, hi);
}

// (debug) LARECOMP_COPY_TRACE=<frame>: draws' attachments / sampled textures, copies and snapshots of that frame ("[ct]")
static bool ct_on(void) {
    static const char *e = (const char *)1;
    if (e == (const char *)1) e = getenv("LARECOMP_COPY_TRACE");
    if (!e) return false;
    static uint64_t memo_f = ~0ull;
    static bool memo;
    uint64_t f = renderer_present_count();
    if (f == memo_f) return memo;
    memo_f = f;
    memo = false;
    for (const char *p = e; *p;) {
        if ((uint64_t)strtoll(p, (char **)&p, 10) == f) { memo = true; break; }
        if (*p == ',') p++; else break;
    }
    if (memo) LOG("[ct] ===== frame %llu\n", (unsigned long long)f);
    return memo;
}
static void ct_draw(Prog *pr, const char *what) {
    LOG("[ct] %s fs %016llx rt", what, (unsigned long long)g_cur_fs_hash);
    for (int i = 0; i < st.rt.ncolor; i++) if (st.rt.color[i]) LOG(" c%d=%p/%#x", i, (void *)st.rt.color[i], st.rt.color[i]->format);
    if (st.rt.depth) LOG(" d=%p/%#x%s", (void *)st.rt.depth, st.rt.depth->format, st.ds[0] && st.ds[1] ? "W" : "");
    LOG(" cm %08x vp %.0f %.0f %.0f %.0f sc %d %d %d %d | tex", st.channel_mask, st.viewport[0][0], st.viewport[0][1], st.viewport[0][2],
        st.viewport[0][3], st.scissor[0][0], st.scissor[0][1], st.scissor[0][2], st.scissor[0][3]);
    ShaderInfo *infos[2] = {pr->vi, pr->fi};
    for (int sg = 0; sg < 2; sg++)
        for (int i = 0; i < (infos[sg] ? infos[sg]->ntex : 0); i++) {
            NvnTexture *t = shader_texture_object(sg, &infos[sg]->tex[i]);
            if (t) LOG(" %s%p/%#x", sg ? "" : "v:", (void *)t, t->format);
        }
    LOG("\n");
}

// ================================================================== copy elision (depth copies)
// The game copies its main depth buffer whole, every frame, into textures that are then only sampled while the
// source keeps its depth (the water / transparency depth input), or that get their depth cleared right away
// (stencil kept). At 3x internal resolution each such copy is ~0.15 ms of pure bandwidth. Instead of copying, the
// destination becomes an alias: sampling it (and copying from it) reads the source's image. The alias ends before
// anything changes either side:
//   - the source is about to be overwritten whole (both aspects cleared): the two images swap - the destination
//     keeps the contents without a copy, the source is cleared in the destination's old image;
//   - the destination is about to be overwritten in some aspects (its depth cleared): only the other aspects are
//     copied (stencil, a quarter of the bytes);
//   - anything else (a draw writing depth / stencil, a partial clear, a copy or upload into either, the
//     destination bound as an attachment, a texture freed or reallocated): the real copy is made first.
// Only same-size, same-format, single-level, single-sample depth/stencil images, whole-image copies.
// LARECOMP_VK_COPY_OPT=0: always copy (the old behaviour).
typedef struct { NvnTexture *src, *dst; VkImage simg, dimg; } CopyAlias;
extern uint32_t vk_view_gen;
static CopyAlias aliases[8];
static int naliases;
static uint64_t stat_alias_made, stat_alias_renamed, stat_alias_partial, stat_alias_copied;

static int copy_opt(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("LARECOMP_VK_COPY_OPT"); on = e && *e ? atoi(e) : 1; }
    return on;
}

// still valid: both textures keep the images the alias was made with (a resolution change reallocates them)
static bool alias_valid(const CopyAlias *a) {
    const TexBackend *s = a->src->backend, *d = a->dst->backend;
    return s && d && s->image == a->simg && d->image == a->dimg;
}

static void alias_drop(int i) {
    aliases[i] = aliases[--naliases];
    vk_view_gen++;  // (exec.c's per-frame view memo: the destination's views change)
}

// the real copy of the aspects asked for, into the destination's own image
static void alias_copy(CopyAlias *a, VkImageAspectFlags aspects, const char *why) {
    TexBackend *s = a->src->backend, *d = a->dst->backend;
    aspects &= d->aspect;
    if (!aspects) return;
    vk_end_pass();
    st.rt_dirty = true;
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    gp_mark(cb, "copy texture (deferred, %s%s) %ux%u", why, aspects == d->aspect ? "" : ", stencil only", a->dst->width, a->dst->height);
    VkImageCopy c = {.srcSubresource = {aspects, 0, 0, 1}, .dstSubresource = {aspects, 0, 0, 1}, .extent = {d->width, d->height, 1}};
    vkCmdCopyImage(cb, s->image, VK_IMAGE_LAYOUT_GENERAL, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    if (aspects == d->aspect) stat_alias_copied++; else stat_alias_partial++;
    if (ct_on()) LOG("[ct] alias copy %p -> %p aspects %#x (%s)\n", (void *)a->src, (void *)a->dst, aspects, why);
}

// t is about to be written; full: the aspects of it the write overwrites whole (0: unknown / partial)
static void alias_before_write(NvnTexture *t, VkImageAspectFlags full, const char *why) {
    if (!naliases || !t) return;
    for (int i = naliases - 1; i >= 0; i--) {
        CopyAlias *a = &aliases[i];
        if (!alias_valid(a)) { alias_drop(i); continue; }
        if (a->dst != t) continue;
        // (LARECOMP_VK_COPY_OPT=2: all aspects even then - for comparing the stencil-only copy's cost)
        alias_copy(a, ((TexBackend *)t->backend)->aspect & (copy_opt() == 2 ? ~0u : ~full), why);
        alias_drop(i);
    }
    NvnTexture *heir = NULL;  // t overwritten whole: one destination can take its image instead of a copy
    if (t->backend && (((TexBackend *)t->backend)->aspect & ~full) == 0)
        for (int i = 0; i < naliases && !heir; i++) if (aliases[i].src == t) heir = aliases[i].dst;
    for (int i = naliases - 1; i >= 0; i--) {
        CopyAlias *a = &aliases[i];
        if (a->src != t || a->dst == heir) continue;
        alias_copy(a, ((TexBackend *)a->dst->backend)->aspect, why);
        alias_drop(i);
    }
    int rename = -1;
    for (int i = 0; i < naliases; i++) if (heir && aliases[i].src == t && aliases[i].dst == heir) rename = i;
    if (rename >= 0) {
        CopyAlias *a = &aliases[rename];
        TexBackend *s = a->src->backend, *d = a->dst->backend;
        vk_end_pass();
        st.rt_dirty = true;
        VkImage im = s->image; s->image = d->image; d->image = im;
        VmaAllocation m = s->mem; s->mem = d->mem; d->mem = m;
        vk_views_forget(s);
        vk_views_forget(d);
        stat_alias_renamed++;
        if (ct_on()) LOG("[ct] alias rename %p -> %p (%s)\n", (void *)a->src, (void *)a->dst, why);
        alias_drop(rename);
    }
}

// the texture whose image holds t's contents (t itself unless it is an alias)
static NvnTexture *alias_read(NvnTexture *t) {
    for (int i = 0; i < naliases; i++)
        if (aliases[i].dst == t) {
            if (alias_valid(&aliases[i])) return aliases[i].src;
            alias_drop(i);
            return t;
        }
    return t;
}

// copy_texture: a whole-image copy that can be an alias instead (true: done)
static bool alias_make(NvnTexture *src, NvnTexture *dst, TexBackend *s, TexBackend *d) {
    if (!copy_opt() || src == dst) return false;
    if (!(s->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))) return false;
    if (s->format != d->format || s->width != d->width || s->height != d->height || s->aspect != d->aspect || s->levels != 1 ||
        d->levels != 1 || s->layers != 1 || d->layers != 1 || s->depth != 1 || d->depth != 1 || s->samples != 1 || d->samples != 1 ||
        s->base_type != d->base_type || s->presented || d->presented || !s->image || !d->image || !s->gpu_owned || !d->gpu_owned)
        return false;
    // the destination's previous contents (or alias) are replaced whole; a destination that is itself the source of
    // aliases gets those resolved first
    alias_before_write(dst, d->aspect, "copy dst");
    src = alias_read(src);  // (a copy of an alias: of its source)
    s = src->backend;
    if (src == dst) return true;  // (copied back onto itself)
    if (naliases == (int)(sizeof aliases / sizeof aliases[0])) return false;
    aliases[naliases++] = (CopyAlias){src, dst, s->image, d->image};
    vk_view_gen++;
    stat_alias_made++;
    return true;
}

// textures.c: t is being freed - an alias source hands its contents over first
void vk_alias_texture_destroyed(NvnTexture *t) {
    if (naliases) alias_before_write(t, 0, "freed");
}
void vk_alias_reset(void) { naliases = 0; }  // (reboot: every texture goes)

// (renderer_stats, once per stats interval) copies made aliases / renamed / partly or fully copied later
void vk_alias_stats(void) {
    static int n;
    if (!stat_alias_made || (n++ > 5 && n % 30)) return;
    LOG("[vk] copy alias: %llu made, %llu renamed, %llu stencil-only, %llu copied later (live %d)\n",
        (unsigned long long)stat_alias_made, (unsigned long long)stat_alias_renamed, (unsigned long long)stat_alias_partial,
        (unsigned long long)stat_alias_copied, naliases);
    stat_alias_made = stat_alias_renamed = stat_alias_partial = stat_alias_copied = 0;
}

// draws: attachments written (or a destination bound at all) end the aliases involved
static void alias_draw_check(void) {
    if (!naliases) return;
    for (int i = 0; i < st.rt.ncolor; i++)
        if (st.rt.color[i]) alias_before_write(st.rt.color[i], 0, "color target");
    NvnTexture *d = st.rt.depth;
    if (!d) return;
    bool writes = (st.ds[0] && st.ds[1]) || (st.ds[3] && ((st.stencil_mask[0] | st.stencil_mask[1]) & 0xFF));
    bool is_dst = false;
    for (int i = 0; i < naliases; i++) is_dst |= aliases[i].dst == d;
    if (writes || is_dst) alias_before_write(d, 0, writes ? "depth write" : "dst bound");
}

static void resolve_feedback(Prog *pr) {
    nfeedback = 0;
    static int off = -1;
    if (off < 0) off = getenv("LARECOMP_NO_FEEDBACK") != NULL;
    if (off) return;
    ShaderInfo *infos[2] = {pr->vi, pr->fi};
    for (int sg = 0; sg < 2; sg++)
        for (int i = 0; i < infos[sg]->ntex; i++) {
            NvnTexture *t = shader_texture_object(sg, &infos[sg]->tex[i]);
            if (!is_attachment(t)) continue;
            bool have = false;
            for (int k = 0; k < nfeedback; k++) have |= feedback_tex[k] == t;
            if (have || nfeedback == 4) continue;
            static int nosnap = -1;
            if (nosnap < 0) nosnap = getenv("LARECOMP_NO_SNAPSHOT_REUSE") != NULL;
            int si = -1;
            for (int k = 0; k < 8; k++) if (snap_tex[k] == t) si = k;
            bool self_write = t == st.rt.depth && st.ds[0] && st.ds[1];
            TexBackend *src = vk_tex(t, true);
            Snap *existing = NULL;
            for (int k = 0; k < nsnaps; k++) if (snaps[k].key == t && snaps[k].copy.image) existing = &snaps[k];
            VkImageAspectFlags sa = src->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : src->aspect;
            static const VkComponentMapping id = {0};
            if (!nosnap && si >= 0 && (snap_ok[si] || self_write) && existing) {
                feedback_tex[nfeedback] = t;
                feedback_view[nfeedback++] = vk_image_view(&existing->copy, VK_IMAGE_VIEW_TYPE_2D, src->format, 0, 1, 0, 1, id, sa);
                continue;
            }
            stat_feedback_copies++;
            if (ct_on()) LOG("[ct] snapshot %p %ux%u (was %s, si %d)\n", (void *)t, src->width, src->height, si >= 0 ? snap_why[si] : "new", si);
            vk_end_pass();
            TexBackend *dst = snapshot_image(t, src);
            VkCommandBuffer cb = vk_cmd();
            vk_full_barrier(cb);
            gp_mark(cb, "feedback snapshot %ux%u fmt %d", src->width, src->height, src->format);
            VkImageCopy rg = {.srcSubresource = {src->aspect, 0, 0, 1}, .dstSubresource = {src->aspect, 0, 0, 1},
                              .extent = {src->width, src->height, 1}};
            vkCmdCopyImage(cb, src->image, VK_IMAGE_LAYOUT_GENERAL, dst->image, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
            if (g_debug_frame && (src->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) {
                static int nd;
                char path[128];
                snprintf(path, sizeof path, "build/fbdepth_%03d_src.pgm", nd);
                debug_save_depth(src, path);
                snprintf(path, sizeof path, "build/fbdepth_%03d_snap.pgm", nd++);
                debug_save_depth(dst, path);
                cb = vk_cmd();
            }
            feedback_tex[nfeedback] = t;
            feedback_view[nfeedback++] = vk_image_view(dst, VK_IMAGE_VIEW_TYPE_2D, src->format, 0, 1, 0, 1, id, sa);
            if (si < 0) { static int rr; si = rr++ & 7; snap_tex[si] = t; }
            snap_ok[si] = true;
            st.rt_dirty = true;
        }
}

// ================================================================== resource binding
static VkDescriptorBufferInfo ubo_span(VkSpan s, uint64_t size) {
    uint64_t max = g_vk.props.limits.maxUniformBufferRange;
    return (VkDescriptorBufferInfo){s.buf, s.off, size > max ? max : size};
}

// FE3H's character shadow map drawn at a fixed scale (textures.c vk_shadow_scale): its fragment shaders take its texel
// size from a constant block - (1/1024, 1/1024, 1024, 1024) then the PCF offsets - scaled to match in the copy.
static void shadow_cbuf_patch(ShaderInfo *si, void *p, uint64_t size) {
    extern float vk_shadow_scale(void);
    float S = vk_shadow_scale();
    if (S <= 1 || !p) return;
    bool any = false;
    for (int i = 0; i < si->ntex && !any; i++) {
        NvnTexture *t = shader_texture_object(1, &si->tex[i]);
        any = t && t->backend && ((TexBackend *)t->backend)->fixed_scale > 1;
    }
    if (!any) return;
    float *f = p;
    for (uint64_t k = 0; k + 8 <= size / 4; k += 4) {
        if (f[k] != 1.0f / 1024 || f[k + 1] != 1.0f / 1024 || f[k + 2] != 1024 || f[k + 3] != 1024) continue;
        f[k] /= S, f[k + 1] /= S, f[k + 2] *= S, f[k + 3] *= S;
        for (int j = 4; j < 8; j++) f[k + j] /= S;  // (PCF offsets, in texels of 1/1024)
    }
}

// Constant buffer k of a stage (renderer.m bind_cbufs), copied into transient memory.
static VkDescriptorBufferInfo cbuf_info(int stage, ShaderInfo *si, int k, uint32_t gx, uint32_t gy, uint32_t gz) {
    static const uint8_t zeros[0x10000];
    if (k == 0) {
        uint32_t drv[0x800 / 4] = {0};
        if (stage == 5) {
            drv[0x200 / 4] = gx; drv[0x204 / 4] = gy; drv[0x208 / 4] = gz;
            for (int i = 0; i < 16; i++) {
                uint32_t *u = &drv[(0x210 + 16 * i) / 4];
                u[0] = (uint32_t)st.ubuf[5][i].addr; u[1] = (uint32_t)(st.ubuf[5][i].addr >> 32); u[2] = (uint32_t)st.ubuf[5][i].size;
                uint32_t *e = &drv[(0x310 + 16 * i) / 4];
                e[0] = (uint32_t)st.sbuf[5][i].addr; e[1] = (uint32_t)(st.sbuf[5][i].addr >> 32); e[2] = (uint32_t)st.sbuf[5][i].size;
            }
        } else {
            uint32_t base = 0x110 + 0x100 * (stage == 1 ? 4 : 0);
            for (int i = 0; i < 16; i++) {
                uint32_t *e = &drv[(base + 16 * i) / 4];
                e[0] = (uint32_t)st.sbuf[stage][i].addr;
                e[1] = (uint32_t)(st.sbuf[stage][i].addr >> 32);
                e[2] = (uint32_t)st.sbuf[stage][i].size;
            }
        }
        uint64_t size = si->cbuf_size[0] > sizeof drv ? sizeof drv : (si->cbuf_size[0] ? si->cbuf_size[0] : 256);
        if (size < 0x500 && stage == 5) size = 0x500;
        return ubo_span(copy_dedupe(stage == 5 ? DEDUPE_DRV_C : stage == 1 ? DEDUPE_DRV_F : DEDUPE_DRV_V, drv, size, 256), size);
    }
    if (k == 1) {  // compiler constants stored after the shader code
        NvnShaderStage *ss = &st.program->stages[stage];
        uint64_t size = ss->const_size ? ss->const_size : 16;
        if (size < si->cbuf_size[1]) size = si->cbuf_size[1];
        return ubo_span(guest_copy(ss->code_addr + ss->const_offset, size), size);  // (constant: shared per submission)
    }
    int idx = k - 3;  // cbuf_base
    Range r = (idx >= 0 && idx < 18) ? st.ubuf[stage][idx] : (Range){0, 0};
    uint64_t size = si->cbuf_size[k] ? si->cbuf_size[k] : 256;
    if (!r.addr) return ubo_span(vk_transient_copy(zeros, size, 256), size);
    if ((si->cbuf_dynamic & (1u << k)) && r.size) size = r.size > 0x10000 ? 0x10000 : r.size;
    static int full = -1;  // (test) LARECOMP_CBUF_FULL=1: dynamically indexed cbufs bound as a full 64 KB of guest memory
    if (full < 0) full = getenv("LARECOMP_CBUF_FULL") != NULL;
    if (full && (si->cbuf_dynamic & (1u << k))) size = 0x10000;
    else if (size > r.size && r.size) size = r.size;
    if (size < 16) size = 16;
    // snapshot: later CPU writes must not leak in. Shared by the draws of one submission binding the same range:
    // within a command list nothing writes guest memory (no inline UpdateUniformBuffer: that writes through below
    // and drops the shared copies).
    VkSpan s = guest_copy(r.addr, size);
    if (stage == 1) shadow_cbuf_patch(si, s.ptr, size);
    // FE3H's final composite (fs 94ed752582273966): c3 [7].x and [8].xyz weigh its two blurred bloom layers
    extern float g_bloom_scale;
    if (stage == 1 && k == 3 && g_cur_fs_hash == 0x94ed752582273966ull && g_bloom_scale != 1 && size >= 0x90) {
        s = vk_transient_copy((const void *)(uintptr_t)r.addr, size, 256);  // (own copy: scaled once)
        float *f = s.ptr;
        f[28] *= g_bloom_scale, f[32] *= g_bloom_scale, f[33] *= g_bloom_scale, f[34] *= g_bloom_scale;
    }
    return ubo_span(s, size);
}

// Global memory (shader LDG/STG, maxwell_gmem_base = HEAP_BASE): the guest heap imported as a GPU buffer
// (VK_EXT_external_memory_host), zero-copy like metal.m's g_guest_buffer, so compute results land in guest
// memory. The import pins its pages, so it covers only the heap the game set up (SetHeapSize), made on first use.
// LARECOMP_VK_NO_GMEM=1: no import, reads return zero.
static VkBuffer gmem_buf;
static VkDeviceMemory gmem_mem;
static uint64_t gmem_size;

static bool gmem_import(uint64_t size) {
    PFN_vkGetMemoryHostPointerPropertiesEXT props =
        (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(g_vk.dev, "vkGetMemoryHostPointerPropertiesEXT");
    VkMemoryHostPointerPropertiesEXT mp = {.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (!props || props(g_vk.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, (void *)(uintptr_t)HEAP_BASE, &mp) != VK_SUCCESS ||
        !mp.memoryTypeBits)
        return false;
    VkImportMemoryHostPointerInfoEXT imp = {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
                                            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                            .pHostPointer = (void *)(uintptr_t)HEAP_BASE};
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &imp, .allocationSize = size,
                               .memoryTypeIndex = (uint32_t)__builtin_ctz(mp.memoryTypeBits)};
    VkDeviceMemory mem;
    if (vkAllocateMemory(g_vk.dev, &ai, NULL, &mem) != VK_SUCCESS) return false;
    VkExternalMemoryBufferCreateInfo eb = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                                           .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &eb, .size = size,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT};
    VkBuffer buf;
    if (vkCreateBuffer(g_vk.dev, &bi, NULL, &buf) != VK_SUCCESS) { vkFreeMemory(g_vk.dev, mem, NULL); return false; }
    vkBindBufferMemory(g_vk.dev, buf, mem, 0);
    if (gmem_buf) {  // the heap grew: the old import may still be in use by submitted work
        vk_wait_seq(submit_seq);
        vkDestroyBuffer(g_vk.dev, gmem_buf, NULL);
        vkFreeMemory(g_vk.dev, gmem_mem, NULL);
    }
    gmem_buf = buf;
    gmem_mem = mem;
    gmem_size = size;
    LOG("[vk] guest heap imported for shader global memory: %llu MB\n", (unsigned long long)(size >> 20));
    return true;
}

static uint64_t gmem_tried;  // heap size the import was last attempted for (gmem_info imports / re-imports)

static VkDescriptorBufferInfo gmem_info(void) {
    static int off = -1;
    if (off < 0) off = getenv("LARECOMP_VK_NO_GMEM") != NULL || !g_vk.host_import;
    uint64_t want = (kernel_heap_size() + 0xFFFF) & ~0xFFFFull;
    if (want > HEAP_MAX) want = HEAP_MAX;
    if (!off && want > gmem_size && !gmem_import(want)) {
        LOG("[vk] guest heap import failed (%llu MB): shader global memory reads return zero\n", (unsigned long long)(want >> 20));
        off = 1;
    }
    if (gmem_buf) return (VkDescriptorBufferInfo){gmem_buf, 0, gmem_size};
    static uint32_t zero[1024];
    VkSpan s = vk_transient_copy(zero, sizeof zero, 256);
    return (VkDescriptorBufferInfo){s.buf, s.off, sizeof zero};
}

// Compute dispatches that use global memory: the imported heap is host memory across PCIe on a discrete GPU, and a
// shader storing to it in a loop crawls (the little chain chomps in Madam MeowMeow's house: ~8 ms a frame on a GTX 1660
// Super and on an RTX 3080 Ti alike, which the game then waits for on its fence - 49-53 fps there on the 1660). Before
// such a dispatch its bound storage buffers (sorted, merged) are copied into a device-local buffer, the address table
// (binding GMAP_BUFFER_INDEX) tells the shader (translate.c, GLSL compute) where each range sits, loads and stores
// inside them use that copy, and a shader that stores gets the ranges copied back to the heap right after it (the
// game alternates its buffer sets per frame: the CPU isn't writing them meanwhile). Accesses outside the table use
// the heap as before. LARECOMP_VK_GMEM_FAST=0: no copy (empty table: everything goes to the heap, the old behaviour).
#define GFAST_SIZE (32ull << 20)
static VkBuffer gfast_buf;
static VmaAllocation gfast_mem;
static uint32_t gmap_tab[(GMAP_MAX + 1) * 4];  // this dispatch's table (stage_set copies it into binding 20)
static VkBufferCopy gfast_cp[GMAP_MAX];       // its copies heap -> gfast_buf (reversed after a storing dispatch)
static int gfast_n;
static uint64_t g_heap_gpu_gen;  // bumped after a dispatch stores to the heap (bf_gpu_copy: copies stale)

static VkDescriptorBufferInfo gfast_info(void) {
    if (!gfast_buf) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = GFAST_SIZE,
                                 .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateBuffer(g_vk.vma, &bi, &ai, &gfast_buf, &gfast_mem, NULL));
    }
    return (VkDescriptorBufferInfo){gfast_buf, 0, GFAST_SIZE};
}

static void gfast_prepare(VkCommandBuffer cb) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("LARECOMP_VK_GMEM_FAST"); on = !e || atoi(e) != 0; }
    memset(gmap_tab, 0, sizeof gmap_tab);
    gfast_n = 0;
    gmem_info();  // (the heap import, made on first use)
    gfast_info();
    if (!on || !gmem_buf) return;
    struct { uint64_t lo, hi; } r[16];
    int n = 0;
    for (int i = 0; i < 16; i++) {  // heap offsets of the bound storage buffers, 256-aligned
        uint64_t a = st.sbuf[5][i].addr, sz = st.sbuf[5][i].size;
        if (!a || !sz || a < HEAP_BASE || a - HEAP_BASE + sz > gmem_size) continue;
        uint64_t lo = (a - HEAP_BASE) & ~255ull, hi = (a - HEAP_BASE + sz + 255) & ~255ull;
        if (hi > gmem_size) hi = gmem_size;
        int k = n++;
        while (k > 0 && r[k - 1].lo > lo) { r[k] = r[k - 1]; k--; }  // (insertion sort by start)
        r[k].lo = lo; r[k].hi = hi;
    }
    int m = 0;
    for (int i = 0; i < n; i++) {  // merge overlapping / nearby ranges (a small gap costs less than another table entry)
        if (m && r[i].lo <= r[m - 1].hi + 4096) { if (r[i].hi > r[m - 1].hi) r[m - 1].hi = r[i].hi; }
        else r[m++] = r[i];
    }
    VkBufferCopy cp[16];
    uint64_t off = 0;
    for (int i = 0; i < m; i++) {
        uint64_t len = r[i].hi - r[i].lo;
        if (m > GMAP_MAX || off + len > GFAST_SIZE) return;  // (too much: the heap as before)
        cp[i] = (VkBufferCopy){r[i].lo, off, len};
        gmap_tab[i * 4 + 0] = (uint32_t)r[i].lo;
        gmap_tab[i * 4 + 1] = (uint32_t)len;
        gmap_tab[i * 4 + 2] = (uint32_t)(off >> 2);
        off += (len + 255) & ~255ull;
    }
    if (!m) return;
    if (getenv("LARECOMP_LOG_GFAST")) {  // (debug) =<shader code address hex>: that shader's first tables
        static int nlog;
        if (st.program->stages[5].code_addr == strtoull(getenv("LARECOMP_LOG_GFAST"), NULL, 16) && nlog++ < 40) {
            LOG("[gfast] cs %llx: %d ranges, %llu KB:", (unsigned long long)st.program->stages[5].code_addr, m, (unsigned long long)(off >> 10));
            for (int i = 0; i < m; i++) LOG(" %llx+%llx", (unsigned long long)(r[i].lo + HEAP_BASE), (unsigned long long)(r[i].hi - r[i].lo));
            LOG("\n");
        }
    }
    vkCmdCopyBuffer(cb, gmem_buf, gfast_buf, (uint32_t)m, cp);
    vk_full_barrier(cb);
    gmap_tab[GMAP_MAX * 4] = (uint32_t)m;
    memcpy(gfast_cp, cp, (size_t)m * sizeof cp[0]);
    gfast_n = m;
}

// After a storing dispatch: its ranges back to the heap (before anything else reads them: draws, the CPU after the fence).
static void gfast_finish(VkCommandBuffer cb) {
    if (!gfast_n) return;
    VkBufferCopy back[GMAP_MAX];
    for (int i = 0; i < gfast_n; i++) back[i] = (VkBufferCopy){gfast_cp[i].dstOffset, gfast_cp[i].srcOffset, gfast_cp[i].size};
    vk_full_barrier(cb);
    vkCmdCopyBuffer(cb, gfast_buf, gmem_buf, (uint32_t)gfast_n, back);
    vk_full_barrier(cb);
    gfast_n = 0;
    g_heap_gpu_gen++;  // (buffer-texture copies of the heap are stale now: bf_gpu_copy)
}

// ---- vertex / index cache
// Reading vertex / index data in place from the imported heap crosses PCIe for every vertex fetched on a discrete GPU
// (outdoors at 3x internal resolution on an RTX 3080 Ti: ~0.7-1.9 ms of GPU a frame; more on PCIe 3.0), and copying it
// per draw instead moves about as much onto the CPU. So each range bound for drawing gets a device-local copy, refreshed
// only where the CPU wrote since (the OS's page write tracking, plat_written_pages, asked at submission: vc_flush): one
// sequential GPU copy of the written pages in the submission's upload buffer (which completes before its draws) instead
// of fetches across the bus. Overlapping ranges are separate entries; the per-page generation keeps them all current.
// Pages the GPU writes itself (a compute dispatch's copy-back, gfast_finish) are never cached: draws over them read the
// heap in place as before, and so does everything when the OS gives no write tracking. LARECOMP_VK_VCACHE=0: off.
#define VC_PAGE 4096ull
#define VC_TAB 8192                   // entries (open addressing)
#define VC_MAX_RANGE (64ull << 20)    // larger bound ranges: read in place
#define VC_BUDGET (768ull << 20)      // device-local bytes kept; the least recently used go first
typedef struct { uint64_t lo, hi, used_seq; uint32_t gen, epoch; VkBuffer buf; VmaAllocation mem; bool gpu; } VcEnt;
static VcEnt vc_tab[VC_TAB];
static uint32_t vc_pend[VC_TAB];  // entries the pending submission draws from (vc_flush brings them up to date)
static int vc_npend;
static uint32_t *vc_page_gen;  // per heap page: the generation of its last CPU write seen (vc_gen when observed)
static uint8_t *vc_page_gpu;   // per heap page: written by the GPU
static uint32_t vc_gen = 1, vc_epoch = 1;  // (epoch: bumped when pages turn GPU-written: entries check again)
static uint64_t vc_bytes;
#define VC_DEAD 1024
static struct { VkBuffer b; VmaAllocation m; uint64_t seq; } vc_dead[VC_DEAD];
static int vc_ndead;
static int vc_on = -1;
// (profiler: draws served / read in place, bytes refreshed, write tracking queries, written pages seen, query time)
uint64_t prof_vc_hits, prof_vc_bypass, prof_vc_copied, prof_vc_queries, prof_vc_scan_pages, prof_vc_scan_ns;

static void vc_free_dead(bool all) {
    uint64_t done = vk_completed_seq();
    int k = 0;
    for (int i = 0; i < vc_ndead; i++) {
        if (!all && vc_dead[i].seq > done) { vc_dead[k++] = vc_dead[i]; continue; }
        vmaDestroyBuffer(g_vk.vma, vc_dead[i].b, vc_dead[i].m);
    }
    vc_ndead = k;
}

static bool vc_drop(VcEnt *e) {  // (freed once the submissions that may read it are done)
    if (vc_ndead == VC_DEAD) { vk_wait_seq(vk_pending_seq() - 1); vc_free_dead(false); }
    if (vc_ndead == VC_DEAD) return false;  // (all read by the pending submission: keep it)
    vc_dead[vc_ndead].b = e->buf; vc_dead[vc_ndead].m = e->mem; vc_dead[vc_ndead++].seq = e->used_seq;
    vc_bytes -= e->hi - e->lo;
    memset(e, 0, sizeof *e);
    return true;
}

// The page tables (vc_page_gen / vc_page_gpu) exist whenever the OS gives write tracking: the vertex cache and the
// texture upload checks (vk_pages_changed) share them. Each write tracking query resets the OS bits, so every query
// stamps the pages it saw written with a new generation, and each user compares against the generation it last saw.
static int pt_state = -1;  // 1: tracking, 0: none
static bool pt_init(void) {
    if (pt_state >= 0) return pt_state;
    uint64_t pg = VC_PAGE, one;
    int n = plat_written_pages(HEAP_BASE, VC_PAGE, &one, 1, &pg);  // (is there write tracking?)
    pt_state = n >= 0 && pg == VC_PAGE;
    if (!pt_state) { LOG("[vk] no page write tracking (%d, page %llu)\n", n, (unsigned long long)pg); return false; }
    vc_page_gen = calloc(HEAP_MAX / VC_PAGE, sizeof *vc_page_gen);
    vc_page_gpu = calloc(HEAP_MAX / VC_PAGE, 1);
    vc_page_gen[0] = ++vc_gen;  // (the probe took page 0's tracking)
    return true;
}

// Texture upload checks (textures.c): may guest memory [addr, addr + size) have changed since generation `since`?
// 0: no CPU write since, 1: written, -1: unknown (no tracking, outside the heap, or pages the GPU writes).
// *now: the generation to pass next time.
int vk_pages_changed(uint64_t addr, uint64_t size, uint32_t since, uint32_t *now) {
    if (!size || addr < HEAP_BASE || addr - HEAP_BASE + size > HEAP_MAX || !pt_init()) return -1;
    uint64_t lo = (addr - HEAP_BASE) & ~(VC_PAGE - 1), hi = (addr - HEAP_BASE + size + VC_PAGE - 1) & ~(VC_PAGE - 1);
    static uint64_t pages[4096];
    uint64_t pg = VC_PAGE;
    int n = plat_written_pages(HEAP_BASE + lo, hi - lo, pages, 4096, &pg);
    if (n < 0) return -1;
    uint32_t g = ++vc_gen;
    if (n >= 4096) for (uint64_t p = lo / VC_PAGE; p < hi / VC_PAGE; p++) vc_page_gen[p] = g;
    else for (int k = 0; k < n; k++) vc_page_gen[(pages[k] - HEAP_BASE) / VC_PAGE] = g;
    *now = g;
    int changed = 0;
    for (uint64_t p = lo / VC_PAGE; p < hi / VC_PAGE; p++) {
        if (vc_page_gpu[p]) return -1;
        if (vc_page_gen[p] > since) changed = 1;
    }
    return changed;
}

// Heap pages [lo, hi) (offsets) written by the GPU: never cached from now on.
static void vc_mark_gpu(uint64_t lo, uint64_t hi) {
    if (!pt_init()) return;
    for (uint64_t p = lo / VC_PAGE; p < (hi + VC_PAGE - 1) / VC_PAGE && p < HEAP_MAX / VC_PAGE; p++) vc_page_gpu[p] = 1;
    vc_epoch++;
}

// A storing shader (stage 0 / 1 / 5) may write any of its bound storage buffers.
static void vc_mark_sbufs(int stage) {
    for (int i = 0; i < 16; i++) {
        uint64_t a = st.sbuf[stage][i].addr, sz = st.sbuf[stage][i].size;
        if (a >= HEAP_BASE && sz && a - HEAP_BASE < HEAP_MAX) vc_mark_gpu(a - HEAP_BASE, a - HEAP_BASE + sz);
    }
}

static bool vcache_lookup(uint64_t addr, uint64_t size, VkBuffer *buf, VkDeviceSize *off) {
    if (vc_on < 0) {
        const char *e = getenv("LARECOMP_VK_VCACHE");
        vc_on = !e || atoi(e) != 0;
        if (vc_on && !pt_init()) {
            LOG("[vk] vertex cache off: no page write tracking\n");
            vc_on = 0;
        }
    }
    if (!vc_on || !gmem_buf || size > VC_MAX_RANGE) return false;
    uint64_t lo = (addr - HEAP_BASE) & ~(VC_PAGE - 1), hi = (addr - HEAP_BASE + size + VC_PAGE - 1) & ~(VC_PAGE - 1);
    if (hi > gmem_size) return false;
    uint64_t pend = vk_pending_seq();
    uint64_t h = (lo * 0x9E3779B97F4A7C15ull) ^ (hi * 0xC2B2AE3D27D4EB4Full);
    uint32_t slot = (uint32_t)(h ^ (h >> 32)) & (VC_TAB - 1), free_slot = ~0u, old_slot = ~0u;
    VcEnt *e = NULL;
    for (int probe = 0; probe < 16; probe++, slot = (slot + 1) & (VC_TAB - 1)) {
        VcEnt *c = &vc_tab[slot];
        if (c->buf && c->lo == lo && c->hi == hi) { e = c; break; }
        if (!c->buf && free_slot == ~0u) free_slot = slot;
        if (c->buf && c->used_seq < pend && (old_slot == ~0u || c->used_seq < vc_tab[old_slot].used_seq)) old_slot = slot;
    }
    if (!e) {  // a new range: its own device-local buffer (gen 0: copied whole at the submission)
        if (vc_npend == VC_TAB) return false;
        if (vc_ndead) vc_free_dead(false);
        if (free_slot == ~0u) {
            if (old_slot == ~0u || !vc_drop(&vc_tab[old_slot])) return false;
            free_slot = old_slot;
        }
        while (vc_bytes + (hi - lo) > VC_BUDGET) {  // over budget: the least recently used entry goes
            int lru = -1;
            for (int i = 0; i < VC_TAB; i++)
                if (vc_tab[i].buf && vc_tab[i].used_seq < pend && (lru < 0 || vc_tab[i].used_seq < vc_tab[lru].used_seq)) lru = i;
            if (lru < 0 || !vc_drop(&vc_tab[lru])) return false;
        }
        e = &vc_tab[free_slot];
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = hi - lo,
                                 .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateBuffer(g_vk.vma, &bi, &ai, &e->buf, &e->mem, NULL) != VK_SUCCESS) { memset(e, 0, sizeof *e); return false; }
        e->lo = lo; e->hi = hi;
        vc_bytes += hi - lo;
    }
    if (e->epoch != vc_epoch) {  // (pages turned GPU-written since: is this range one of them?)
        e->gpu = false;
        for (uint64_t p = lo / VC_PAGE; p < hi / VC_PAGE && !e->gpu; p++) e->gpu = vc_page_gpu[p];
        e->epoch = vc_epoch;
    }
    if (e->gpu) { prof_vc_bypass++; return false; }
    if (e->used_seq != pend) {  // first use in this submission: brought up to date by vc_flush before it runs
        e->used_seq = pend;
        vc_pend[vc_npend++] = (uint32_t)(e - vc_tab);
    }
    prof_vc_hits++;
    *buf = e->buf;
    *off = addr - HEAP_BASE - lo;
    return true;
}

static int vc_pend_cmp(const void *a, const void *b) {
    uint64_t x = vc_tab[*(const uint32_t *)a].lo, y = vc_tab[*(const uint32_t *)b].lo;
    return x < y ? -1 : x > y;
}

// At submission (vk_submit_extra, before the upload buffer closes): the pages the CPU wrote in the ranges this
// submission draws from - one write tracking query per run of nearby ranges (a query costs about as much as scanning
// ~64 pages: ranges closer than that share one) - and their copies into the cache, ahead of the draws.
void vc_flush(void) {
    if (!vc_npend) return;
    qsort(vc_pend, (size_t)vc_npend, sizeof vc_pend[0], vc_pend_cmp);
    uint32_t g = ++vc_gen;
    static uint64_t pages[4096];
    uint64_t t0 = SDL_GetTicksNS();
    for (int i = 0; i < vc_npend;) {
        uint64_t lo = vc_tab[vc_pend[i]].lo, hi = vc_tab[vc_pend[i]].hi;
        int j = i + 1;
        for (; j < vc_npend && vc_tab[vc_pend[j]].lo <= hi + 64 * VC_PAGE; j++)
            if (vc_tab[vc_pend[j]].hi > hi) hi = vc_tab[vc_pend[j]].hi;
        uint64_t pg = VC_PAGE;
        int n = plat_written_pages(HEAP_BASE + lo, hi - lo, pages, 4096, &pg);
        prof_vc_queries++;
        if (n < 0 || n >= 4096) for (uint64_t p = lo / VC_PAGE; p < hi / VC_PAGE; p++) vc_page_gen[p] = g;  // (all)
        else for (int k = 0; k < n; k++) vc_page_gen[(pages[k] - HEAP_BASE) / VC_PAGE] = g;
        if (n > 0) prof_vc_scan_pages += (uint64_t)n;
        i = j;
    }
    prof_vc_scan_ns += SDL_GetTicksNS() - t0;
    VkCommandBuffer cb = vk_upload_cmd();
    for (int i = 0; i < vc_npend; i++) {
        VcEnt *e = &vc_tab[vc_pend[i]];
        VkBufferCopy cp[64];
        int nc = 0;
        for (uint64_t p = e->lo / VC_PAGE; p < e->hi / VC_PAGE; p++) {
            if (e->gen && vc_page_gen[p] <= e->gen) continue;
            uint64_t o = p * VC_PAGE;
            if (nc && cp[nc - 1].srcOffset + cp[nc - 1].size == o) { cp[nc - 1].size += VC_PAGE; continue; }
            if (nc == 64) { nc = 1; cp[0] = (VkBufferCopy){e->lo, 0, e->hi - e->lo}; break; }  // (scattered: the whole range)
            cp[nc++] = (VkBufferCopy){o, o - e->lo, VC_PAGE};
        }
        if (nc) {
            vkCmdCopyBuffer(cb, gmem_buf, e->buf, (uint32_t)nc, cp);
            for (int k = 0; k < nc; k++) prof_vc_copied += cp[k].size;
        }
        e->gen = g;
    }
    vc_npend = 0;
}

// Vertex / index data from the imported guest heap (renderer.m bind_guest reads g_guest_buffer the same way): the
// device-local cache above, else read in place - no per-draw copy, no index scan for the copy size. The game fences its
// CPU writes as on the Switch. Falls back to copies outside the heap or without the import; LARECOMP_VK_COPY_VERTICES=1
// forces copies.
static bool guest_direct(uint64_t addr, uint64_t size, VkBuffer *buf, VkDeviceSize *off) {
    static int copy = -1;
    if (copy < 0) copy = getenv("LARECOMP_VK_COPY_VERTICES") != NULL;
    if (copy || addr < HEAP_BASE) return false;
    if (addr + size > HEAP_BASE + gmem_size && kernel_heap_size() != gmem_tried) { gmem_tried = kernel_heap_size(); gmem_info(); }
    if (!gmem_buf || addr + size > HEAP_BASE + gmem_size) return false;
    if (vcache_lookup(addr, size, buf, off)) return true;
    *buf = gmem_buf;
    *off = addr - HEAP_BASE;
    return true;
}

static int g_cs_n;  // (compute program table below)

// In-place reboot (device.c backend_reset; GPU idle): drop what is keyed by guest addresses - programs by code
// address (the old Prog / Pipe objects are leaked, so pipeline keys holding their pointers can't collide), compute
// programs, shared copies - and the heap import: its pages are replaced (re-imported on first use).
void vk_exec_reset(void) {
    memset(prog_tab, 0, sizeof prog_tab);
    memset(pipe_tab, 0, sizeof pipe_tab);
    for (int i = 0; i < nwarm_pipes; i++) warm_insert(warm_pipes[i]);  // (by content: still valid)
    g_cs_n = 0;
    gcopy_gen++;
    if (gmem_buf) {
        vkDestroyBuffer(g_vk.dev, gmem_buf, NULL);
        vkFreeMemory(g_vk.dev, gmem_mem, NULL);
        gmem_buf = VK_NULL_HANDLE;
        gmem_mem = VK_NULL_HANDLE;
    }
    gmem_size = gmem_tried = 0;
    for (int i = 0; i < VC_TAB; i++)  // (the vertex cache: copies of the old pages)
        if (vc_tab[i].buf) vmaDestroyBuffer(g_vk.vma, vc_tab[i].buf, vc_tab[i].mem);
    memset(vc_tab, 0, sizeof vc_tab);
    vc_npend = 0;
    vc_free_dead(true);
    vc_bytes = 0;
    if (vc_page_gpu) memset(vc_page_gpu, 0, HEAP_MAX / VC_PAGE);
    vc_epoch++;
    st_defaults();
}

// Images (SUST in compute shaders): their 64-bit handles follow the 32 textures' in the bound handles (0x120 on),
// each the index of a texture pool entry registered as an image (nvnTexturePoolRegisterImage).
static NvnTexEntry *image_entry(int stage, const ShaderTexture *st_) {
    int slot = ((int)st_->handle_offset - 0x120) / 8;
    uint64_t h = (slot >= 0 && slot < 8) ? st.img[stage][slot] : 0;
    uint32_t tic = h & 0xFFFFF;
    if (!h || !st.tpool || tic >= (uint32_t)st.tpool->count) return NULL;
    NvnTexEntry *e = &((NvnTexEntry *)st.tpool->entries)[tic];
    return e->tex ? e : NULL;
}

// A storage view of the image (one level), or VK_NULL_HANDLE when it can't be stored to: its writes are kept as the
// GPU's (never re-uploaded from guest memory over them).
static VkImageView image_view(int stage, const ShaderTexture *st_) {
    NvnTexEntry *e = image_entry(stage, st_);
    if (!e) return VK_NULL_HANDLE;
    TexBackend *b = vk_tex(e->tex, false);  // (its data uploaded first if the CPU wrote it)
    if (b->scalable && (b->sx != 1 || b->sy != 1)) {  // (stores use guest texel coordinates: would cover a corner only)
        static int nlog;
        if (nlog++ < 10) LOG("[vk] image store to a scaled target %ux%u fmt %#x (x%.3f)\n", e->tex->width, e->tex->height, e->tex->format, b->sx);
    }
    if (!b->storage_fmt || rt_kind(e->tex->format)) {  // (integer formats: the translator stores floats)
        static int nlog;
        if (nlog++ < 10) LOG("[vk] image %p (nvn fmt %#x flags %#x) isn't a storage image: compute store skipped\n", (void *)e->tex,
                             e->tex->format, e->tex->flags);
        return VK_NULL_HANDLE;
    }
    b->gpu_owned = true;
    uint32_t lv = e->has_view && (e->view.setmask & 1) ? e->view.base_level : 0;
    if (lv >= b->levels) lv = 0;
    VkImageViewType vt = st_->type == TEXTYPE_1D ? VK_IMAGE_VIEW_TYPE_1D : st_->type == TEXTYPE_3D ? VK_IMAGE_VIEW_TYPE_3D
                       : st_->type == TEXTYPE_2D_ARRAY ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    bool want3d = vt == VK_IMAGE_VIEW_TYPE_3D, have3d = b->base_type == VK_IMAGE_VIEW_TYPE_3D;
    if (want3d != have3d) return VK_NULL_HANDLE;
    static const VkComponentMapping id = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                          VK_COMPONENT_SWIZZLE_IDENTITY};
    uint32_t nly = vt == VK_IMAGE_VIEW_TYPE_2D_ARRAY ? b->layers : 1;
    return vk_image_view(b, vt, b->storage_fmt, lv, 1, 0, have3d ? 1 : nly, id, VK_IMAGE_ASPECT_COLOR_BIT);
}

static bool dlaa_jittered(void);
static float scene_tex_bias(void);
static bool tex_video[64];  // (this draw's fragment textures with uploaded content, movie frames: not scene buffers)
static void texture_info(int stage, ShaderInfo *si, int i, VkDescriptorImageInfo *out, float sc[2], uint32_t wh[2]) {
    if (stage == 1 && i < 64) tex_video[i] = false;
    ShaderTexture *st_ = &si->tex[i];
    int slot = ((int)st_->handle_offset - 0x20) / 8;  // 64-bit handles from offset 0x20
    uint64_t h = (slot >= 0 && slot < 32) ? st.tex[stage][slot] : 0;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler smp = VK_NULL_HANDLE;
    uint32_t tic = h & 0xFFFFF, tsc = (uint32_t)(h >> 20) & 0xFFF;
    sc[0] = sc[1] = 1;
    wh[0] = wh[1] = 0;
    if (h && st.tpool && tic < (uint32_t)st.tpool->count) {
        NvnTexEntry *e = &((NvnTexEntry *)st.tpool->entries)[tic];
        if (e->tex) {
            // Per-frame memo: ~2400 lookups per overworld frame recomputed the view (format / range math, cache
            // probe). The first lookup of a frame still runs vk_view (upload check); a view that may have been
            // destroyed (eviction, image re-created) bumps vk_view_gen. LARECOMP_VK_NO_TEX_MEMO=1 disables.
            extern uint32_t vk_view_gen;
            extern uint64_t vk_frame_counter(void);
            static int nomemo = -1;
            if (nomemo < 0) nomemo = getenv("LARECOMP_VK_NO_TEX_MEMO") != NULL;
            static struct { NvnTexture *tex; uint64_t vh, frame; uint32_t gen; uint8_t type, has_view; VkImageView view; } memo[1024];
            uint64_t vh = e->has_view ? hash_words(0, &e->view, sizeof e->view) : 0, frame = vk_frame_counter();
            uint32_t ms = (uint32_t)((((uintptr_t)e->tex >> 4) ^ vh ^ (uint64_t)st_->type * 0x9E3779B97F4A7C15ull) & 1023);
            if (!nomemo && memo[ms].tex == e->tex && memo[ms].vh == vh && memo[ms].frame == frame && memo[ms].gen == vk_view_gen &&
                memo[ms].type == st_->type && memo[ms].has_view == e->has_view) {
                view = memo[ms].view;
            } else {
                view = vk_view(naliases ? alias_read(e->tex) : e->tex, &e->view, e->has_view, false, st_->type);
                memo[ms].tex = e->tex; memo[ms].vh = vh; memo[ms].frame = frame; memo[ms].gen = vk_view_gen;
                memo[ms].type = (uint8_t)st_->type; memo[ms].has_view = e->has_view; memo[ms].view = view;
            }
            renderer_texture_scale(e->tex, &sc[0], &sc[1]);
            wh[0] = e->tex->width; wh[1] = e->tex->height;
            if (stage == 1 && i < 64 && e->tex->backend) tex_video[i] = ((TexBackend *)e->tex->backend)->video || !((TexBackend *)e->tex->backend)->gpu_owned;
            for (int k = 0; k < nfeedback; k++)
                if (e->tex == feedback_tex[k] && (st_->type == TEXTYPE_2D || st_->type == TEXTYPE_2D_ARRAY)) {
                    view = feedback_view[k];
                    if (st_->type == TEXTYPE_2D_ARRAY) view = vk_dummy_view(TEXTYPE_2D_ARRAY);  // (2D snapshot as array: rare)
                }
        }
    }
    if (h && st.spool && tsc < (uint32_t)st.spool->count)  // (DLAA: the jittered scene's textures one mip sharper, as NVIDIA advises)
        smp = vk_sampler_bias(&((NvnSamplerDesc *)st.spool->entries)[tsc], stage == 1 ? scene_tex_bias() : 0);
    if (!view) view = vk_dummy_view(st_->type);
    if (!smp) smp = vk_default_sampler(true);
    *out = (VkDescriptorImageInfo){smp, view, VK_IMAGE_LAYOUT_GENERAL};
}

// Writes the descriptor set of one stage. gx..gz: compute group counts (driver constants).
// pre_ii: texture descriptors already resolved by the caller (draw: fragment textures, needed earlier for the
// viewport decision), or NULL.
static float g_mirror_w;  // mirror mode's composite: its render target width (rscale[65].x; 0 otherwise)
static float g_vs_rows[64][4];  // vertex stage rscale[1..]: the UI box's per-quad centres (ui_world_icons)
static int g_vs_nrows;
static float g_vs_px[2];  // vertex rscale[65].zw: the boxed UI's pass size (2D vertices snap to its pixels)

static bool mv_binding(const VkBindingInfo *b, int stage, ShaderInfo *si, VkDescriptorBufferInfo *out);
// Buffer textures (bfetch) read the guest heap, which is host memory across PCIe: FE3H's deferred lighting reads its
// light lists there for every pixel and light (16 of 20 ms a frame at 2x). A stage whose only heap reads are
// buffer textures gets their ranges copied into the device-local ring instead (once per submission) and its heap
// binding pointed there. LARECOMP_VK_BFETCH_HEAP=1: the old way.
// The copy is a GPU copy recorded before the draw's pass (bf_gpu_copy): the lists are GPU-written (a CPU memcpy at
// record time read last frame's - lights cut off in tiles while the camera moved, DLC map). CPU copy: fallback.
static struct { bool valid; VkSpan span; uint64_t total, key; uint32_t off[64]; bool has[64]; } g_bf;
typedef struct { uint64_t key, total, addr[64], len[64]; uint32_t off[64]; bool has[64]; } BfRanges;
static bool bf_ranges(Prog *p, int stage, ShaderInfo *si, BfRanges *b) {
    static int off = -1;
    if (off < 0) off = getenv("LARECOMP_VK_BFETCH_HEAP") != NULL;
    int sg = stage == 1 ? 1 : 0;
    if (off || (stage != 0 && stage != 1) || !si || !si->uses_gmem || !si->msl) return false;
    if (!p->bf_only[sg]) {  // (bfetch_at holds the shader's only 4 heap reads)
        int n = 0;
        for (const char *q = strstr(si->msl, "gmem_.w["); q; q = strstr(q + 8, "gmem_.w[")) n++;
        p->bf_only[sg] = n == 4 && strstr(si->msl, "vec4 bfetch_at(") ? 1 : 2;
    }
    if (p->bf_only[sg] != 1) return false;
    memset(b->has, 0, sizeof b->has);
    b->key = vk_pending_seq() * 0x9E3779B97F4A7C15ull ^ g_heap_gpu_gen, b->total = 0;
    for (int t = 0; t < si->ntex && t < 32; t++) {
        if (si->tex[t].type != TEXTYPE_1D) continue;
        NvnTexture *bt = shader_texture_object(stage, &si->tex[t]);
        if (!bt || bt->target != 10 || bt->addr < HEAP_BASE) continue;
        const FormatInfo *fi = nvn_format_info(bt->format);
        uint64_t len = (uint64_t)bt->width * (fi && fi->bpb ? fi->bpb : 4);
        if (!len || len > (64ull << 20)) continue;
        b->addr[t] = bt->addr, b->len[t] = len;
        b->off[t] = (uint32_t)b->total, b->has[t] = true;
        b->total += (len + 255) & ~255ull;
        b->key = (b->key ^ bt->addr) * 0x100000001B3ull ^ len;
    }
    return b->total != 0;
}
// Before the draw's pass: a new set of ranges is copied heap -> ring on the GPU, in order after earlier GPU writes
static void bf_gpu_copy(Prog *p, int stage, ShaderInfo *si) {
    BfRanges b;
    if (!bf_ranges(p, stage, si, &b) || (g_bf.span.buf && g_bf.key == b.key && g_bf.total == b.total)) return;
    VkBufferCopy cp[64];
    int n = 0;
    for (int t = 0; t < 64; t++) {
        if (!b.has[t]) continue;
        if (!gmem_buf || b.addr[t] + b.len[t] > HEAP_BASE + gmem_size) return;  // (stage_set copies on the CPU)
        cp[n++] = (VkBufferCopy){b.addr[t] - HEAP_BASE, 0, b.len[t]};
    }
    VkSpan s = vk_transient(b.total, 256);
    if (!s.buf) return;
    n = 0;
    for (int t = 0; t < 64; t++) if (b.has[t]) cp[n++].dstOffset = s.off + b.off[t];
    vk_end_pass();
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    vkCmdCopyBuffer(cb, gmem_buf, s.buf, (uint32_t)n, cp);
    vk_full_barrier(cb);
    g_bf.span = s, g_bf.total = b.total, g_bf.key = b.key;
}
static void bf_prepare(Prog *p, int stage, ShaderInfo *si) {
    g_bf.valid = false;
    BfRanges b;
    if (!bf_ranges(p, stage, si, &b)) return;
    uint64_t total = b.total;
    uint32_t *offs = b.off;
    bool *has = b.has;
    if (!g_bf.span.buf || g_bf.key != b.key || g_bf.total != total) {  // (not copied before the pass: on the CPU)
        VkSpan s = vk_transient(total, 256);
        if (!s.ptr) return;
        for (int t = 0; t < 64; t++) if (has[t]) memcpy((uint8_t *)s.ptr + offs[t], (const void *)(uintptr_t)b.addr[t], b.len[t]);
        g_bf.span = s, g_bf.total = total, g_bf.key = b.key;
    }
    memcpy(g_bf.off, offs, sizeof g_bf.off);
    memcpy(g_bf.has, has, sizeof g_bf.has);
    g_bf.valid = true;
}
static VkDescriptorSet stage_set(Prog *p, VkDescriptorSetLayout layout, uint32_t set, int stage, ShaderInfo *si, const float *rscale_row0,
                                 const float tint[8], uint32_t gx, uint32_t gy, uint32_t gz, const VkDescriptorImageInfo *pre_ii) {
    VkDescriptorSet ds = vk_alloc_set(layout);
    bf_prepare(p, stage, si);
    VkWriteDescriptorSet w[MAX_BIND];
    VkDescriptorBufferInfo bi[MAX_BIND];
    VkDescriptorImageInfo ii[MAX_BIND];
    int n = 0;
    int sg = stage == 1 ? 1 : 0;
    float rs[66][4] = {{1, 1, 0, 0}};  // ([65]: mirror mode, renderer.m)
    if (rscale_row0) memcpy(rs[0], rscale_row0, 16);
    if (sg == 0 && g_vs_nrows) memcpy(rs[1], g_vs_rows, (size_t)g_vs_nrows * 16);
    rs[65][0] = g_mirror_w;
    if (sg == 0) rs[65][2] = g_vs_px[0], rs[65][3] = g_vs_px[1];
    for (int i = 0; i < p->nbinds; i++) {
        VkBindingInfo *b = &p->binds[i];
        if (b->set != set) continue;
        w[n] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = b->binding,
                                      .descriptorCount = 1, .descriptorType = b->type};
        if (b->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER && b->binding >= 32) {
            int ti = (int)b->binding - 32;
            float sc[2]; uint32_t wh[2];
            if (ti < si->ntex && pre_ii && ti < 64) { ii[n] = pre_ii[ti]; sc[0] = tex_sc[sg][ti][0]; sc[1] = tex_sc[sg][ti][1]; wh[0] = tex_wh[sg][ti][0]; wh[1] = tex_wh[sg][ti][1]; }
            else if (ti < si->ntex) texture_info(stage, si, ti, &ii[n], sc, wh);
            else { ii[n] = (VkDescriptorImageInfo){vk_default_sampler(true), vk_dummy_view(TEXTYPE_2D), VK_IMAGE_LAYOUT_GENERAL}; sc[0] = sc[1] = 1; wh[0] = wh[1] = 0; }
            if (ti < 64) {
                tex_sc[sg][ti][0] = sc[0]; tex_sc[sg][ti][1] = sc[1];
                tex_wh[sg][ti][0] = wh[0]; tex_wh[sg][ti][1] = wh[1];
            }
            w[n].pImageInfo = &ii[n];
        } else if (b->type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE && b->binding >= 32) {
            int ti = (int)b->binding - 32;
            VkImageView v = ti < si->ntex ? image_view(stage, &si->tex[ti]) : VK_NULL_HANDLE;  // (checked before: images_ready)
            ii[n] = (VkDescriptorImageInfo){VK_NULL_HANDLE, v, VK_IMAGE_LAYOUT_GENERAL};
            w[n].pImageInfo = &ii[n];
        } else if (p->is_motion && b->binding >= 40 && mv_binding(b, stage, si, &bi[n])) {
            w[n].pBufferInfo = &bi[n];
        } else if (b->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER && b->binding < 17) {
            bi[n] = cbuf_info(stage, si, (int)b->binding, gx, gy, gz);
            w[n].pBufferInfo = &bi[n];
        } else if (b->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
            bi[n] = b->binding == GFAST_BUFFER_INDEX ? gfast_info()
                  : g_bf.valid ? (VkDescriptorBufferInfo){g_bf.span.buf, g_bf.span.off, g_bf.total} : gmem_info();
            w[n].pBufferInfo = &bi[n];
        } else {
            continue;  // rscale / tint: written below, once the textures (and their scales) are known
        }
        n++;
    }
    for (int i = 0; i < p->nbinds; i++) {
        VkBindingInfo *b = &p->binds[i];
        if (b->set != set || b->type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
            (b->binding != 18 && b->binding != 19 && b->binding != GMAP_BUFFER_INDEX)) continue;
        w[n] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = b->binding,
                                      .descriptorCount = 1, .descriptorType = b->type};
        if (b->binding == GMAP_BUFFER_INDEX) {  // (compute) this dispatch's fast-copy table: gfast_prepare
            bi[n] = ubo_span(vk_transient_copy(gmap_tab, sizeof gmap_tab, 256), sizeof gmap_tab);
        } else if (b->binding == 18) {
            for (int t = 0; t < si->ntex && t < 64; t++) { rs[1 + t][0] = tex_sc[sg][t][0]; rs[1 + t][1] = tex_sc[sg][t][1]; }
            for (int t = 0; t < si->ntex && t < 32; t++) {  // buffer textures (bfetch): heap byte offset, count | format << 24
                if (si->tex[t].type != TEXTYPE_1D) continue;
                NvnTexture *bt = shader_texture_object(stage, &si->tex[t]);
                if (!bt || bt->target != 10) continue;
                const FormatInfo *fi = nvn_format_info(bt->format);
                uint64_t bytes = (uint64_t)bt->width * (fi && fi->bpb ? fi->bpb : 4);
                uint32_t off = 0xFFFFFFFFu, cnt = (bt->width < 0xFFFFFFu ? bt->width : 0xFFFFFFu) | (bt->format & 0xFFu) << 24;
                gmem_info();  // (imports the heap on first use)
                if (bt->addr >= HEAP_BASE && bt->addr + bytes <= HEAP_BASE + gmem_size && gmem_buf) off = (uint32_t)(bt->addr - HEAP_BASE);
                if (g_bf.valid) off = g_bf.has[t] ? g_bf.off[t] : 0xFFFFFFFFu;  // (the device-local copy: bf_prepare)
                {   // LARECOMP_WATCH_BUFTEX=1 (debug): watch the first compute-bound buffer texture's page - who touches it
                    static bool armed;
                    // LARECOMP_WATCH_ADDR=<hex addr>,<hex size> instead: that range (armed at the same moment)
                    if (!armed && stage == 5 && (getenv("LARECOMP_WATCH_BUFTEX") || getenv("LARECOMP_WATCH_ADDR")) &&
                        renderer_present_count() >= 60) {
                        extern uint64_t g_watch_lo, g_watch_hi;
                        extern void watch_arm(void);
                        armed = true;
                        g_watch_lo = bt->addr;
                        g_watch_hi = bt->addr + bytes;
                        const char *wa = getenv("LARECOMP_WATCH_ADDR");
                        if (wa) {
                            char *e;
                            g_watch_lo = strtoull(wa, &e, 16);
                            g_watch_hi = g_watch_lo + (*e == ',' ? strtoull(e + 1, NULL, 16) : 0x10);
                        }
                        LOG("[buftex] watching %llx..%llx\n", (unsigned long long)g_watch_lo, (unsigned long long)g_watch_hi);
                        watch_arm();
                    }
                }
                if (getenv("LARECOMP_LOG_BUFTEX")) {  // (debug) each buffer texture when bound, and 3 frames later
                    static struct { uint64_t addr, bytes, frame; int stage, slot; bool rescanned; } bl[64];
                    static int nbl;
                    uint64_t now = renderer_present_count();
                    for (int k = 0; k < nbl; k++) {
                        if (bl[k].rescanned || now < bl[k].frame + 3) continue;
                        bl[k].rescanned = true;
                        const uint32_t *wd = (const uint32_t *)(uintptr_t)bl[k].addr;
                        uint64_t nw = bl[k].bytes / 4, nz = 0, first = ~0ull;
                        for (uint64_t q = 0; q < nw; q++) if (wd[q]) { nz++; if (first == ~0ull) first = q; }
                        LOG("[buftex] f%llu: f%llu's stage %d slot %d buffer %llx now %llu of %llu words non-zero",
                            (unsigned long long)now, (unsigned long long)bl[k].frame, bl[k].stage, bl[k].slot,
                            (unsigned long long)bl[k].addr, (unsigned long long)nz, (unsigned long long)nw);
                        if (first != ~0ull) {
                            LOG(", from word %llu:", (unsigned long long)first);
                            for (uint64_t q = first; q < first + 8 && q < nw; q++) LOG(" %08x", wd[q]);
                        }
                        LOG("\n");
                    }
                    static uint64_t dbg_frame = ~0ull;  // (F9: this frame's bindings; else the first after frame 60)
                    if (g_debug_frame && dbg_frame != now) dbg_frame = now, nbl = 0;
                    if (nbl < 64 && (g_debug_frame || (dbg_frame == ~0ull && now >= 60))) {
                        const uint32_t *wd = (const uint32_t *)(uintptr_t)bt->addr;
                        uint64_t nw = bytes / 4, nz = 0;
                        for (uint64_t q = 0; q < nw; q++) if (wd[q]) nz++;
                        LOG("[buftex] f%llu stage %d slot %d buffer %llx %u texels fmt %#x off %#x: %llu of %llu words non-zero when bound\n",
                            (unsigned long long)now, stage, t, (unsigned long long)bt->addr, bt->width, bt->format, off,
                            (unsigned long long)nz, (unsigned long long)nw);
                        bl[nbl].addr = bt->addr; bl[nbl].bytes = bytes; bl[nbl].frame = now;
                        bl[nbl].stage = stage; bl[nbl].slot = t; bl[nbl].rescanned = false;
                        nbl++;
                    }
                }
                memcpy(&rs[1 + t][2], &off, 4);
                memcpy(&rs[1 + t][3], &cnt, 4);
            }
            bi[n] = ubo_span(copy_dedupe(sg ? DEDUPE_RSCALE_F : DEDUPE_RSCALE_V, rs, sizeof rs, 256), sizeof rs);
        } else {
            float zero[8] = {0};
            bi[n] = ubo_span(copy_dedupe(DEDUPE_TINT, tint ? tint : zero, 32, 256), 32);
        }
        w[n].pBufferInfo = &bi[n];
        n++;
    }
    if (n) vkUpdateDescriptorSets(g_vk.dev, (uint32_t)n, w, 0, NULL);
    return ds;
}

// ================================================================== compute
typedef struct { Prog p; ShaderInfo *si; VkPipeline pipe; uint32_t local[3]; const uint8_t *blob; size_t cmax; ShaderKey key; } CsProg;

static void warm_cs_record(const CsProg *cp);
static void build_compute(void *arg) {
    CsProg *cp = arg;
    size_t n = 0;
    uint64_t t0 = SDL_GetTicksNS();
    uint32_t *spv = vk_glsl_to_spirv(cp->si->msl, STAGE_COMPUTE, "cs", &n);
    uint64_t t_spirv = SDL_GetTicksNS();
    if (!spv) { set_state(&cp->p.state, 2); return; }
    cp->p.nbinds = vk_reflect(spv, n, cp->p.binds, MAX_BIND);
    VkShaderModule m = vk_module(spv, n);
    free(spv);
    cp->p.layout = make_layout(&cp->p, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0);
    VkSpecializationMapEntry me[3] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}};
    VkSpecializationInfo spec = {3, me, 12, cp->local};
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo sg32 = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, .requiredSubgroupSize = 32};
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .pNext = g_vk.cs_subgroup32 ? &sg32 : NULL,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main",
                                                .pSpecializationInfo = &spec},
                                      .layout = cp->p.layout};
    VkResult r = vkCreateComputePipelines(g_vk.dev, g_vk.pcache, 1, &ci, NULL, &cp->pipe);
    slow_build_log("compute shader", cp->p.code[0], cp->si->num_insts, t0, t_spirv);
    atomic_fetch_add(&stat_compiles, 1);
    if (r == VK_SUCCESS) warm_cs_record(cp);
    set_state(&cp->p.state, r == VK_SUCCESS ? 1 : 2);
    if (r != VK_SUCCESS) LOG("[compute] pipeline failed: %d\n", r);
}

// Compute shaders in the prewarm too (cache/warm_cs.bin): each one built is recorded by its translation's hash and
// workgroup size, and built again at launch like the graphics pipelines above. Without it a battle effect's compute
// shaders (1-3 s each in AMD's driver the first time) were built only when the effect first played, and it was drawn
// without them that once (a combat art cut-in as outlines). Large ones on the large-shader threads, as in play.
#define WARM_CS_MAGIC 0x43524157u  // "WARC"
#define WARM_CS_VERSION 1
#define WARM_CS 2048               // (table size; at most half used)
typedef struct { uint64_t xh; uint32_t local[3], ninsts; } WarmCsRec;
typedef struct { WarmCsRec r; CsProg *cp; _Atomic int taken; } WarmCs;
static WarmCs warm_cs[WARM_CS];
static int nwarm_cs;
static FILE *warm_cs_file;

static WarmCs *warm_cs_slot(uint64_t xh, const uint32_t local[3], bool add) {
    uint32_t i = (uint32_t)pair_hash(xh, local[0] | (uint64_t)local[1] << 16 | (uint64_t)local[2] << 32) & (WARM_CS - 1);
    while (warm_cs[i].r.xh && (warm_cs[i].r.xh != xh || memcmp(warm_cs[i].r.local, local, 12)))
        i = (i + 1) & (WARM_CS - 1);
    if (warm_cs[i].r.xh) return &warm_cs[i];
    if (!add || nwarm_cs >= WARM_CS / 2) return NULL;
    nwarm_cs++;
    warm_cs[i].r.xh = xh;
    memcpy(warm_cs[i].r.local, local, 12);
    return &warm_cs[i];
}

static void warm_cs_record(const CsProg *cp) {
    if (warm_off() || !cp->p.xh[0]) return;
    pthread_mutex_lock(&warm_lock);
    WarmCs *w = warm_cs_slot(cp->p.xh[0], cp->local, false);
    if (!w && (w = warm_cs_slot(cp->p.xh[0], cp->local, true))) {  // (new: not in the file yet)
        w->r.ninsts = cp->si->num_insts;
        if (!warm_cs_file) {
            mkdir("cache", 0755);
            warm_cs_file = fopen("cache/warm_cs.bin", "ab");
            if (warm_cs_file && (fseek(warm_cs_file, 0, SEEK_END), ftell(warm_cs_file)) == 0) {
                uint32_t hdr[4] = {WARM_CS_MAGIC, WARM_CS_VERSION, (uint32_t)sizeof(WarmCsRec), 0};
                fwrite(hdr, sizeof hdr, 1, warm_cs_file);
            }
        }
        if (warm_cs_file) { fwrite(&w->r, sizeof w->r, 1, warm_cs_file); fflush(warm_cs_file); }
    }
    pthread_mutex_unlock(&warm_lock);
}

static void warm_cs_build(void *arg) {
    WarmCs *w = arg;
    if (atomic_exchange(&w->taken, 1)) return;  // (already built: the game needed it first, warm_cs_find)
    CsProg *cp = w->cp;
    cp->si = xlat_load(w->r.xh);
    if (!cp->si || !cp->si->msl) {
        if (!cp->si) cp->si = calloc(1, sizeof *cp->si);
        set_state(&cp->p.state, 2);
        return;
    }
    cp->p.vi = cp->si;
    build_compute(cp);
}

static void warm_cs_queue(WarmCs *w, bool low) {
    if (w->r.ninsts > 4000 && big_cs_ok()) run_huge_job(warm_cs_build, w);
    else if (w->r.ninsts > 2000) run_heavy_job(warm_cs_build, w);
    else run_job_at(warm_cs_build, w, low);
}

static void warm_cs_start(void) {
    FILE *f = fopen("cache/warm_cs.bin", "rb");
    if (!f) return;
    uint32_t hdr[4] = {0};
    if (fread(hdr, sizeof hdr, 1, f) != 1 || hdr[0] != WARM_CS_MAGIC || hdr[1] != WARM_CS_VERSION || hdr[2] != sizeof(WarmCsRec)) {
        fclose(f);
        remove("cache/warm_cs.bin");
        return;
    }
    WarmCsRec r;
    int ndead = 0;
    pthread_mutex_lock(&warm_lock);
    while (fread(&r, sizeof r, 1, f) == 1) {
        char path[64];
        struct stat sb;
        snprintf(path, sizeof path, "cache/xlat/%016llx.bin", (unsigned long long)r.xh);
        if (stat(path, &sb)) { ndead++; continue; }  // (an older translator's: built again when met)
        if (warm_cs_slot(r.xh, r.local, false)) continue;
        WarmCs *w = warm_cs_slot(r.xh, r.local, true);
        if (!w) break;
        w->r = r;
        w->cp = calloc(1, sizeof *w->cp);
        w->cp->p.warm = true;
        w->cp->p.xh[0] = r.xh;
        w->cp->p.code[0] = r.xh;  // (for the slow build log)
        memcpy(w->cp->local, r.local, 12);
    }
    fclose(f);
    if (ndead) {  // (the file without them: the live table, rewritten)
        FILE *o = fopen("cache/warm_cs.bin.tmp", "wb");
        bool ok = o && fwrite(hdr, sizeof hdr, 1, o) == 1;
        for (int i = 0; ok && i < WARM_CS; i++)
            if (warm_cs[i].cp) ok = fwrite(&warm_cs[i].r, sizeof warm_cs[i].r, 1, o) == 1;
        if (o && fclose(o)) ok = false;
        if (ok) { remove("cache/warm_cs.bin"); ok = !rename("cache/warm_cs.bin.tmp", "cache/warm_cs.bin"); }
        if (!ok) remove("cache/warm_cs.bin.tmp");
    }
    pthread_mutex_unlock(&warm_lock);
    if (!nwarm_cs) return;
    LOG("[shader] prewarm: building %d compute shaders from earlier sessions\n", nwarm_cs);
    for (int i = 0; i < WARM_CS; i++)
        if (warm_cs[i].cp) warm_cs_queue(&warm_cs[i], true);
}

// The compute program the game binds, if it was prebuilt (by its translation's hash and workgroup size)
static CsProg *warm_cs_find(uint64_t xh, const uint32_t local[3]) {
    if (!nwarm_cs || !xh) return NULL;
    pthread_mutex_lock(&warm_lock);
    WarmCs *w = warm_cs_slot(xh, local, false);
    pthread_mutex_unlock(&warm_lock);
    if (!w || !w->cp) return NULL;
    if (!atomic_load(&w->taken)) warm_cs_queue(w, false);  // (needed now: ahead of the rest of the prewarm)
    return w->cp;
}

// A new compute program's translation (worker), then its build: very large ones (battle shaders of ~9,000 instructions
// with subroutine calls, as a branch state machine) on threads of their own where the driver finishes them (big_cs_ok),
// else skipped; large ones on the large-shader thread, the rest right here.
static void translate_compute(void *arg) {
    CsProg *cp = arg;
    maxwell_gmem_base = HEAP_BASE;
    maxwell_gmem_size = HEAP_MAX;
    uint64_t xh = 0;
    cp->si = translate_cached(cp->blob, cp->cmax, STAGE_COMPUTE, &cp->key, &xh);
    cp->p.vi = cp->si;
    if (!cp->si->msl) {
        LOG("[compute] translate failed: %s (code %llx)\n", cp->si->error, (unsigned long long)cp->p.code[0]);
        set_state(&cp->p.state, 2);
        return;
    }
    save_glsl(cp->si->msl, cp->p.code[0], "cs");
    uint32_t ni = cp->si->num_insts;
    if (ni > 4000 && !big_cs_ok()) {
        static _Atomic int nlog;
        if (atomic_fetch_add(&nlog, 1) < 20)
            LOG("[compute] %llx: %u instructions, too large to build: skipped\n", (unsigned long long)cp->p.code[0], ni);
        set_state(&cp->p.state, 2);
    } else if (ni > 2000) {
        LOG("[compute] %llx: %u instructions, building on a thread for %s shaders\n", (unsigned long long)cp->p.code[0], ni,
            ni > 4000 ? "very large" : "large");
        if (ni > 4000) run_huge_job(build_compute, cp);
        else run_heavy_job(build_compute, cp);
    } else {
        build_compute(cp);
    }
}

static struct CsEnt { uint64_t addr; uint32_t bufmask; CsProg *cp; } g_cs_tab[256];  // compute programs by code address

// bufmask: texture slots bound to buffer textures (a variant reading them from the heap, see buffer_texture_mask)
static CsProg *compute_program(NvnProgram *np, uint32_t bufmask) {
    NvnShaderStage *ss = &np->stages[5];
    if (!ss->code_addr) return NULL;
    struct CsEnt *tab = g_cs_tab;
    int n = g_cs_n;
    for (int i = 0; i < n; i++) if (tab[i].addr == ss->code_addr && tab[i].bufmask == bufmask) return tab[i].cp;
    CsProg *cp = calloc(1, sizeof *cp);
    if (n < 256) { tab[n].addr = ss->code_addr; tab[n].bufmask = bufmask; tab[n].cp = cp; g_cs_n = n + 1; }
    maxwell_gmem_base = HEAP_BASE;
    maxwell_gmem_size = HEAP_MAX;
    cp->key.glsl = 1;
    key_set_buf_mask(&cp->key, bufmask);
    cp->blob = (const uint8_t *)(uintptr_t)ss->code_addr;
    cp->cmax = ss->const_offset ? ss->const_offset : 0x10000;
    cp->p.code[0] = ss->code_addr;
    cp->p.xh[0] = xlat_key(cp->blob, cp->cmax, STAGE_COMPUTE, &cp->key);
    for (int i = 0; i < 3; i++) {
        cp->local[i] = ss->control ? *(const uint32_t *)(ss->control + 0x748 + 4 * i) : 1;
        if (!cp->local[i]) cp->local[i] = 1;
    }
    CsProg *w = warm_cs_find(cp->p.xh[0], cp->local);
    if (w) {  // (prebuilt at launch, or building)
        if (n < 256) tab[n].cp = w;
        free(cp);
        return w;
    }
    // translated on a worker too (the battle's large ones took ~20 ms each on this thread: a hitch at the first attack)
    run_job(translate_compute, cp);
    return cp;
}

static void dispatch_compute(uint32_t gx, uint32_t gy, uint32_t gz) {
    snap_reason = "compute"; snapshots_invalidate(NULL);
    if (!st.program || !gx || !gy || !gz) return;
    // (the compute program as bound: not a combination with whatever graphics programs are bound, which would build
    // the same compute shader again for every graphics combination)
    NvnProgram *csp = st.stage_prog[5] ? st.stage_prog[5] : st.program;
    CsProg *cp = compute_program(csp, 0);
    if (!cp || !wait_built(&cp->p.state, &cp->p.waited_ns))
        SKIP(SK_CS_PROG, "compute code %llx state %d\n", (unsigned long long)st.program->stages[5].code_addr, cp ? atomic_load(&cp->p.state) : -1);
    {
        uint32_t m = buffer_texture_mask(5, cp->si);
        if (m) {
            cp = compute_program(csp, m);
            if (!cp || !wait_built(&cp->p.state, &cp->p.waited_ns)) SKIP(SK_CS_PROG, "buffer-texture compute variant building\n");
        }
    }
    {   // LARECOMP_SKIP_CS=<code address hex> (debug): skip that compute shader's dispatches, to see what it does
        static uint64_t skip = ~0ull;
        if (skip == ~0ull) { const char *e = getenv("LARECOMP_SKIP_CS"); skip = e ? strtoull(e, NULL, 16) : 0; }
        if (skip && st.program->stages[5].code_addr == skip) SKIP(SK_CS_DEBUG, "%llx\n", (unsigned long long)skip);
    }
    if (cp->si->uses_gmem && getenv("LARECOMP_LOG_CS_SBUF")) {  // (debug) the storage buffers bound to gmem dispatches
        static uint64_t seen_cs[16];
        static int seen_n[16];
        int k = 0;
        while (k < 15 && seen_cs[k] && seen_cs[k] != st.program->stages[5].code_addr) k++;
        seen_cs[k] = st.program->stages[5].code_addr;
        if (seen_n[k]++ < 24) {
            LOG("[cs] %llx %ux%ux%u:", (unsigned long long)st.program->stages[5].code_addr, gx, gy, gz);
            for (int i = 0; i < 16; i++)
                if (st.sbuf[5][i].addr) LOG(" sb%d %llx+%llx", i, (unsigned long long)st.sbuf[5][i].addr, (unsigned long long)st.sbuf[5][i].size);
            LOG("\n");
        }
    }
    bool images = false;  // (SUST) every image the shader stores to: storable, else the dispatch is skipped
    for (int i = 0; i < cp->si->ntex; i++)
        if (cp->si->tex[i].is_image) {
            images = true;
            if (!image_view(5, &cp->si->tex[i])) SKIP(SK_CS_PROG, "image %d not storable\n", i);
        }
    vk_end_pass();
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    gp_mark(cb, "dispatch %ux%ux%u (local %ux%ux%u) cs %llx gmem %d", gx, gy, gz, cp->local[0], cp->local[1], cp->local[2],
            (unsigned long long)st.program->stages[5].code_addr, cp->si->uses_gmem);
    dl_note("dispatch %ux%ux%u cs %llx gmem %d", gx, gy, gz, (unsigned long long)st.program->stages[5].code_addr, cp->si->uses_gmem);
    nfeedback = 0;
    if (ct_on()) {
        LOG("[ct] dispatch cs %llx %ux%ux%u | tex", (unsigned long long)st.program->stages[5].code_addr, gx, gy, gz);
        for (int i = 0; i < cp->si->ntex; i++) {
            NvnTexture *t = shader_texture_object(5, &cp->si->tex[i]);
            if (t) LOG(" %s%p/%#x", cp->si->tex[i].is_image ? "I:" : "", (void *)t, t->format);
        }
        LOG("\n");
    }
    if (cp->si->uses_gmem_map) gfast_prepare(cb);
    VkDescriptorSet ds = stage_set(&cp->p, cp->p.set_layout[0], 0, 5, cp->si, NULL, NULL, gx, gy, gz, NULL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cp->pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cp->p.layout, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cb, gx, gy, gz);
    if (images) vk_full_barrier(cb);  // (its image stores, before they're sampled)
    if (cp->si->uses_gmem_map && cp->si->uses_gmem_store) gfast_finish(cb);
    if (cp->si->uses_gmem_store) vc_mark_sbufs(5);
    stat_dispatches++;
}

// ================================================================== draws
static void debug_draw(Prog *pr, CmdRec *r, bool indexed);

// Quads / fans / polygons as a 32-bit triangle index list.
static uint32_t *expand_indices(uint32_t mode, uint32_t count, const void *indices, int itype, uint32_t first, uint32_t *out_count) {
    uint32_t tris = mode == 7 ? count / 4 * 2 : count >= 3 ? count - 2 : 0;
    uint32_t *out = malloc(tris * 3 * 4 + 4);
    uint32_t n = 0;
#define IDX(i) (indices ? (itype == 0 ? ((const uint8_t *)indices)[i] : itype == 1 ? ((const uint16_t *)indices)[i] : ((const uint32_t *)indices)[i]) : first + (i))
    if (mode == 7) {
        for (uint32_t q = 0; q + 3 < count; q += 4) {
            out[n++] = IDX(q); out[n++] = IDX(q + 1); out[n++] = IDX(q + 2);
            out[n++] = IDX(q); out[n++] = IDX(q + 2); out[n++] = IDX(q + 3);
        }
    } else {
        for (uint32_t i = 1; i + 1 < count; i++) { out[n++] = IDX(0); out[n++] = IDX(i); out[n++] = IDX(i + 1); }
    }
#undef IDX
    *out_count = n;
    return out;
}

static bool topology(uint32_t mode, VkPrimitiveTopology *t, uint8_t *cls) {
    switch (mode) {
    case 0: *t = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; *cls = 0; return true;
    case 1: *t = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; *cls = 1; return true;
    case 3: *t = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; *cls = 1; return true;
    case 4: *t = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; *cls = 2; return true;
    case 5: *t = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; *cls = 2; return true;
    default: *t = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; *cls = 2; return false;
    }
}

// LARECOMP_NAN_SCAN=1 (with F9): after every draw of the debug frames, the float colour targets are read back and a
// change in their NaN / Inf texel counts is logged (which draw first writes them). No pass PNGs in this mode.
static int nan_scan_on = -1;
static void nan_scan(void) {
    static struct { VkImage img; uint32_t nan, inf; } seen[64];
    static int nseen;
    for (int i = 0; i < st.rt.ncolor && i < 8; i++) {
        TexBackend *b = st.rt.color[i] ? st.rt.color[i]->backend : NULL;
        if (!b || !b->image || b->samples > 1) continue;
        VkFormat f = b->format;
        uint32_t bpp = f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || f == VK_FORMAT_R32_SFLOAT ? 4 : f == VK_FORMAT_R16G16B16A16_SFLOAT ? 8 : 0;
        if (!bpp) continue;
        vk_end_pass();
        VkCommandBuffer cb = vk_cmd();
        vk_full_barrier(cb);
        uint32_t w = b->width, h = b->height;
        VkSpan out = vk_readback((uint64_t)w * h * bpp);
        VkBufferImageCopy c = {.bufferOffset = out.off, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {w, h, 1}};
        vkCmdCopyImageToBuffer(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, out.buf, 1, &c);
        renderer_flush();
        uint32_t nan = 0, inf = 0, x0 = w, y0 = h, x1 = 0, y1 = 0;
        for (uint32_t y = 0; y < h; y++)
            for (uint32_t x = 0; x < w; x++) {
                const uint8_t *p = (const uint8_t *)out.ptr + ((uint64_t)y * w + x) * bpp;
                int bad = 0;  // 1 inf, 2 nan
                if (f == VK_FORMAT_R32_SFLOAT) {
                    uint32_t v; memcpy(&v, p, 4);
                    if ((v & 0x7F800000u) == 0x7F800000u) bad = (v & 0x7FFFFFu) ? 2 : 1;
                } else if (bpp == 8) {
                    for (int k = 0; k < 4; k++) {
                        uint16_t v; memcpy(&v, p + k * 2, 2);
                        if ((v & 0x7C00) == 0x7C00) { int t = (v & 0x3FF) ? 2 : 1; if (t > bad) bad = t; }
                    }
                } else {
                    uint32_t v; memcpy(&v, p, 4);
                    uint32_t ch[3] = {v & 0x7FF, (v >> 11) & 0x7FF, v >> 22 << 1};  // (blue: 5-bit mantissa, aligned to 6)
                    for (int k = 0; k < 3; k++)
                        if ((ch[k] & 0x7C0) == 0x7C0) { int t = (ch[k] & 0x3F) ? 2 : 1; if (t > bad) bad = t; }
                }
                if (!bad) continue;
                if (bad == 2) nan++; else inf++;
                if (x < x0) x0 = x;
                if (x > x1) x1 = x;
                if (y < y0) y0 = y;
                if (y > y1) y1 = y;
            }
        int k = 0;
        while (k < nseen && seen[k].img != b->image) k++;
        if (k == nseen) { if (nseen == 64) continue; seen[nseen++].img = b->image; }
        if (seen[k].nan == nan && seen[k].inf == inf) continue;
        LOG("[nanscan] f%llu after fs=%llx: rt%d %ux%u vkfmt %d: nan %u (was %u) inf %u (was %u) bbox %u,%u..%u,%u\n",
            (unsigned long long)renderer_present_count(), st.program ? (unsigned long long)st.program->stages[1].code_addr : 0ull, i, w, h,
            (int)f, nan, seen[k].nan, inf, seen[k].inf, x0, y0, x1, y1);
        seen[k].nan = nan; seen[k].inf = inf;
    }
}

static Pipe *pipe_get(const PipeKey *k) {  // the pipeline for k: found, or queued to build
    uint64_t ph = hash_words(0xcbf29ce484222325ull, k, sizeof *k);
    if (!ph) ph = 1;
    uint32_t pslot = (uint32_t)(ph ^ (ph >> 29)) & (PIPE_TAB - 1);
    while (pipe_tab[pslot].key && pipe_tab[pslot].key != ph) pslot = (pslot + 1) & (PIPE_TAB - 1);
    Pipe *pp = pipe_tab[pslot].pipe;
    if (!pp) {
        pp = calloc(1, sizeof *pp);
        pp->k = *k;
        pipe_tab[pslot].key = ph;
        pipe_tab[pslot].pipe = pp;
        run_job(build_pipeline, pp);
    }
    return pp;
}

// FE3H ultrawide: the map's unit icons are placed by the game with the widened camera; boxed like the HUD they'd sit
// short of their units. Each UI layout draws with its own pixel -> clip cbuf; the icon layout is the one that masks
// its portrait diamonds with depth writes (seen the frame before, by its order among the frame's layouts). Its quads
// keep their place and only shrink around their own centres (g_vs_rows, the box variant's vertex shader).
// FE3H's UI perspective camera: vertex cbuf c3 starts with a pure 16:9 projection (the 3D cameras are widened or view-projection)
static bool ui_camera_16x9(void) {
    const float *u = (const float *)(uintptr_t)st.ubuf[0][0].addr;
    if (!u || st.ubuf[0][0].size < 64 || u[5] <= 0) return false;
    for (int i = 1; i < 16; i++)
        if (i != 5 && i != 10 && i != 11 && i != 14 && u[i] != 0) return false;
    return u[14] == -1 && u[15] == 0 && fabsf(u[0] / u[5] - 0.5625f) < 0.002f;
}

// A small 2D draw touching a screen edge without spanning the width (status L/R arrows): its edge vertices get boxed too
static bool ui_edge_sprite(CmdRec *r, bool indexed) {
    const float *u = (const float *)(uintptr_t)st.ubuf[0][0].addr;
    float w = (float)st.rt.color[0]->width, h = (float)st.rt.color[0]->height;
    if (indexed || !u || st.ubuf[0][0].size < 16 || fabsf(u[0] * w - 2) > 0.02f || fabsf(fabsf(u[1]) * h - 2) > 0.02f) return false;
    if (r->x[2] == 0 || r->x[2] > 384 || !st.vbuf[0].addr || st.stream[0][0] < 8) return false;
    float x0 = 1e9f, x1 = -1e9f;
    for (uint64_t v = 0; v < r->x[2]; v++) {
        uint64_t o = (r->x[1] + v) * st.stream[0][0];
        if (o + 8 > st.vbuf[0].size) return false;
        const float *p = (const float *)(uintptr_t)(st.vbuf[0].addr + o);
        x0 = fminf(x0, p[0]), x1 = fmaxf(x1, p[0]);
    }
    return (x0 <= 0 || x1 >= w) && (x0 > 0 || x1 < w);
}

static bool ui_world_icons(CmdRec *r, bool indexed) {
    static int off = -1;
    if (off < 0) off = getenv("LARECOMP_NO_ICON_FIX") != NULL;
    const float *u = (const float *)(uintptr_t)st.ubuf[0][0].addr;
    float w = (float)st.rt.color[0]->width, h = (float)st.rt.color[0]->height;
    if (off || !u || st.ubuf[0][0].size < 16 || fabsf(u[0] * w - 2) > 0.02f || fabsf(fabsf(u[1]) * h - 2) > 0.02f) return false;
    static uint64_t frame = ~0ull, cbuf;
    static int ord;
    static uint32_t prev, cur, prev_big, big;  // (per layout: depth-masking / has something no icon is: menus)
    uint64_t f = renderer_present_count();
    if (f != frame) frame = f, prev = cur, prev_big = big, cur = big = 0, ord = -1, cbuf = 0;
    if (st.ubuf[0][0].addr != cbuf) cbuf = st.ubuf[0][0].addr, ord++;
    if (ord < 0 || ord >= 32) return false;
    if (st.ds[0] && st.ds[1]) cur |= 1u << ord;  // (a depth-writing UI draw: the diamonds' masks)
    // icons are separate small quads (6 vertices each, under 300 px); indexed batches (text) or big panels: a menu
    int nq = (int)(r->x[2] / 6);
    if (indexed || r->x[2] % 6 || nq > 64 || !st.vbuf[0].addr || st.stream[0][0] < 12) { big |= 1u << ord; return false; }
    for (int q = 0; q < nq; q++) {
        float sx = 0, x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f;
        for (int v = 0; v < 6; v++) {
            uint64_t o = (r->x[1] + (uint64_t)q * 6 + v) * st.stream[0][0];
            if (o + 8 > st.vbuf[0].size) { big |= 1u << ord; return false; }
            const float *p = (const float *)(uintptr_t)(st.vbuf[0].addr + o);
            sx += p[0];
            x0 = fminf(x0, p[0]), x1 = fmaxf(x1, p[0]), y0 = fminf(y0, p[1]), y1 = fmaxf(y1, p[1]);
        }
        if (x1 - x0 > 300 || y1 - y0 > 300) big |= 1u << ord;
        g_vs_rows[q][0] = sx / 6 * u[0] + u[2];  // (the quad's centre, clip x)
    }
    if (!(prev >> ord & 1) || (prev_big >> ord & 1) || (big >> ord & 1)) return false;
    g_vs_nrows = nq;
    if (g_debug_frame) LOG("[draw]   world icon: %d quads, layout %d\n", nq, ord);
    return true;
}

// FE3H ultrawide: world-anchored 2D labels (monastery speech bubbles, name tags, quest marks) are depth-writing UI
// draws placed with the widened camera. Boxed whole they drift to the centre; instead each keeps its anchor on the
// wide screen (a viewport shift) and only its size is boxed. A draw inside the previous one's bbox (a bubble's text
// in its frame) shares that anchor. Returns the anchor's clip x.
static bool ui_world_label(CmdRec *r, bool indexed, float *ax) {
    static int off = -1;
    if (off < 0) off = getenv("LARECOMP_NO_LABEL_FIX") != NULL;
    const float *u = (const float *)(uintptr_t)st.ubuf[0][0].addr;
    float w = (float)st.rt.color[0]->width, h = (float)st.rt.color[0]->height;
    if (off || !u || st.ubuf[0][0].size < 16 || fabsf(u[0] * w - 2) > 0.02f || fabsf(fabsf(u[1]) * h - 2) > 0.02f) return false;
    // labels: depth test GEQUAL + write, one small z per draw (their world depth); menus' masks / minimap differ
    if (!st.ds[0] || !st.ds[1] || st.ds[2] != 7 || indexed || !st.vbuf[0].addr || st.stream[0][0] < 12 || !r->x[2] || r->x[2] > 8192)
        return false;
    float b[4] = {1e9f, 1e9f, -1e9f, -1e9f}, z0 = 0;
    for (uint64_t v = 0; v < r->x[2]; v++) {
        uint64_t o = (r->x[1] + v) * st.stream[0][0];
        if (o + 12 > st.vbuf[0].size) return false;
        const float *p = (const float *)(uintptr_t)(st.vbuf[0].addr + o);
        if (!v) z0 = p[2];
        if (p[2] != z0 || !(z0 > 0 && z0 < 0.2f)) return false;
        b[0] = fminf(b[0], p[0]), b[1] = fminf(b[1], p[1]), b[2] = fmaxf(b[2], p[0]), b[3] = fmaxf(b[3], p[1]);
    }
    if (b[2] - b[0] > w * 0.6f) return false;  // (not a label: spans the screen)
    static uint64_t frame = ~0ull, cbuf;
    static float a[4];
    static bool have;
    uint64_t f = renderer_present_count();
    if (f != frame || st.ubuf[0][0].addr != cbuf) frame = f, cbuf = st.ubuf[0][0].addr, have = false;
    if (!(have && b[0] >= a[0] - 2 && b[1] >= a[1] - 2 && b[2] <= a[2] + 2 && b[3] <= a[3] + 2)) memcpy(a, b, sizeof a), have = true;
    *ax = (a[0] + a[2]) * 0.5f * u[0] + u[2];
    if (g_debug_frame) LOG("[draw]   world label: bbox %.0f %.0f %.0f %.0f, anchor x %.1f\n", b[0], b[1], b[2], b[3], (a[0] + a[2]) * 0.5f);
    return true;
}

// ================================================================== DLAA (dlss.cpp)
// The 3D scene (draws testing the G-buffer pass's depth) is jittered by a sub-pixel viewport shift; once the game has
// composited it into the window image, the next draw there (HUD, menus) first runs DLAA over it with that depth, zero
// motion vectors (DLSS takes the camera's motion from depth + clipToPrevClip) and the camera captured from the
// G-buffer draws (their c3 starts with the view-projection). Characters' own motion is not in the vectors yet.
static NvnTexture *g_main_depth;  // the G-buffer pass's depth
static NvnTexture *g_gbuf_mat;    // its material target (green: characters)
static uint64_t g_dl_frame = ~0ull, g_scene_frame = ~0ull;
static float g_cam[16], g_cam_prev[16], g_cam_cand[16], g_jit[2];
static bool g_cam_ok, g_cam_prev_ok, g_dlaa_pending, g_dlaa_ran, g_dlaa_tried, g_dlaa_seen;  // (tried: once a frame)
static bool g_resolve_down, g_dlaa_sharpen_late;  // the last present averaged down to the window; DLAA left its RCAS to that step
// Sharpening level -> RCAS stops (exp2(-stops); -1 off) and, for levels 4 / 5, the extra low-contrast detail boost.
static float sharpen_level(float *boost) {
    static const float stops[6] = {-1, 1.0f, 0.5f, 0.2f, 0, 0}, extra[6] = {0, 0, 0, 0, 0.7f, 1.4f};
    int l = g_settings.dlaa_sharpen < 0 ? 0 : g_settings.dlaa_sharpen > 5 ? 5 : g_settings.dlaa_sharpen;
    const char *e = getenv("LARECOMP_SHARPEN_BOOST");  // (test: the boost amount)
    if (boost) *boost = e && l ? (float)atof(e) : extra[l];
    return stops[l];
}
static int g_cam_hits;
// (test, LARECOMP_JITTER_TEST + F10) the jitter's sign as passed to DLSS: 0 (+x +y), 1 (-x +y), 2 (+x -y), 3 (-x -y),
// 4 no jitter
static int g_jit_mode;
static bool g_jit_mode_changed;
float g_bloom_scale = 1;  // (the composite's bloom weight: LARECOMP_DLAA_TUNE + F10 cycles 1 / 0.5 / 0)
float bloom_scale_cycle(void) {
    g_bloom_scale = g_bloom_scale == 1 ? 0.5f : g_bloom_scale == 0.5f ? 0 : 1;
    LOG("[post] bloom %.1f\n", g_bloom_scale);
    return g_bloom_scale;
}
static float g_char_bias = -1;  // (DLAA's character mask strength; LARECOMP_DLAA_TUNE + F10 cycles it)
float dlaa_char_bias_cycle(void) {
    static const float v[4] = {0, 0.5f, 0.75f, 1.0f};
    int i = 0;
    while (i < 4 && v[i] != g_char_bias) i++;
    g_char_bias = v[(i + 1) % 4];
    LOG("[dlss] character bias %.2f\n", g_char_bias);
    return g_char_bias;
}
int dlaa_jitter_cycle(void) {
    g_jit_mode = (g_jit_mode + 1) % 5;
    g_jit_mode_changed = true;
    LOG("[dlss] jitter test mode %d\n", g_jit_mode);
    return g_jit_mode;
}

static bool dlaa_on(void) {
    static int off = -1;
    if (off < 0) off = getenv("LARECOMP_NO_DLAA") != NULL;
    return !off && g_settings.antialias == AA_DLAA && dlss_available();
}

static float halton(int i, int b) {
    float f = 1, r = 0;
    for (; i > 0; i /= b) f /= b, r += f * (i % b);
    return r;
}

static void m4_mul(float *o, const float *a, const float *b) {  // column-major: o = a * b
    float t[16];
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++)
            t[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    memcpy(o, t, 64);
}

static bool m4_inv(float *o, const float *m) {
    float v[16];
    v[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    v[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    v[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    v[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    v[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    v[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    v[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    v[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    v[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    v[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    v[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    v[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    v[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    v[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    v[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    v[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    float det = m[0] * v[0] + m[1] * v[4] + m[2] * v[8] + m[3] * v[12];
    if (fabsf(det) < 1e-30f) return false;
    for (int i = 0; i < 16; i++) o[i] = v[i] / det;
    return true;
}

// Per draw, first: a new frame's jitter, and the G-buffer pass's depth and camera.
static void dlaa_track(void) {
    uint64_t f = renderer_present_count();
    if (f != g_dl_frame) {
        static int nwarn;  // (why a 3D frame got no DLAA: no camera found, or no composite seen)
        if (dlaa_on() && g_scene_frame == g_dl_frame && !g_dlaa_ran && (nwarn++ < 5 || nwarn % 600 == 0))
            LOG("[dlss] frame %llu: 3D scene without DLAA (camera %s, composite %s)\n", (unsigned long long)g_dl_frame,
                g_cam_ok ? "found" : "not found", g_dlaa_seen ? "seen" : "not seen");
        g_dl_frame = f;
        g_cam_prev_ok = g_cam_ok && g_dlaa_ran;
        if (g_cam_ok) memcpy(g_cam_prev, g_cam, 64);
        g_cam_ok = false, g_cam_hits = 0, g_dlaa_pending = false, g_dlaa_ran = false, g_dlaa_tried = false, g_dlaa_seen = false;
        g_dlaa_sharpen_late = false;
        int i = (int)(f % 16) + 1;
        g_jit[0] = halton(i, 2) - 0.5f, g_jit[1] = halton(i, 3) - 0.5f;
        if (g_jit_mode == 4) g_jit[0] = g_jit[1] = 0;  // (test: no jitter)
    }
    if (st.rt.ncolor != 4 || !st.rt.depth) return;  // (the G-buffer: albedo, normals, material, extra)
    g_main_depth = st.rt.depth;
    g_gbuf_mat = st.rt.color[2];
    g_scene_frame = f;
    const float *base = (const float *)(uintptr_t)st.ubuf[0][0].addr, *m = NULL;
    if (g_cam_ok || !base) return;
    extern float widescreen_factor(void);
    // the view-projection heads c3 (battles, maps), or follows a vector or two (the monastery's shaders: +16 / +32 bytes)
    for (int off = 0; off <= 8 && !m && st.ubuf[0][0].size >= (off + 16) * 4u; off += 4) {
        const float *c = base + off;
        float lx = sqrtf(c[0] * c[0] + c[4] * c[4] + c[8] * c[8]), ly = sqrtf(c[1] * c[1] + c[5] * c[5] + c[9] * c[9]);
        float lw = sqrtf(c[3] * c[3] + c[7] * c[7] + c[11] * c[11]);
        if (ly >= 1e-3f && fabsf(lw - 1) <= 0.02f && fabsf(lx / ly - 0.5625f / fmaxf(widescreen_factor(), 1.0f)) <= 0.005f) m = c;
    }
    if (!m) return;
    if (g_cam_hits && !memcmp(m, g_cam_cand, 64)) {  // (two draws agree: the camera, not one object's matrix)
        memcpy(g_cam, m, 64);
        g_cam_ok = true;
    } else memcpy(g_cam_cand, m, 64), g_cam_hits = 1;
}

// A depth-tested draw on the G-buffer pass's depth (scene geometry, its effects): shifted by this frame's jitter.
static bool dlaa_jittered(void) {
    return dlaa_on() && st.rt.depth && st.rt.depth == g_main_depth && st.ds[0] &&
           !(st.rt.ncolor && st.rt.color[0] && renderer_texture_presented(st.rt.color[0]));
}

// NVIDIA's texture mip bias for DLSS: log2(render / output width) - 1 (DLAA: -1). LARECOMP_DLAA_MIP_BIAS overrides.
static float dlaa_mip_bias(void) {
    static float b = 1;
    if (b == 1) b = getenv("LARECOMP_DLAA_MIP_BIAS") ? (float)atof(getenv("LARECOMP_DLAA_MIP_BIAS")) : -1.0f;
    return b;
}
// The 3D scene's draws (on the G-buffer pass's depth, depth-tested): Video tab "Texture detail" and DLAA's bias
static float scene_tex_bias(void) {
    if (!st.rt.depth || st.rt.depth != g_main_depth || !st.ds[0]) return 0;
    if (st.rt.ncolor && st.rt.color[0] && renderer_texture_presented(st.rt.color[0])) return 0;
    static const float td[4] = {0, -1, -2, -15};
    float b = td[g_settings.tex_detail & 3];
    return dlaa_jittered() ? fminf(b, dlaa_mip_bias()) : b;
}

static TexBackend *dlaa_image(TexBackend *d, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage, bool zero) {
    if (d->image && (d->width != w || d->height != h || d->format != fmt)) {
        vk_views_forget(d);
        vk_defer_image(d->image, d->mem);
        d->image = VK_NULL_HANDLE, d->mem = VK_NULL_HANDLE;
    }
    if (!d->image) {
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
                                .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &d->image, &d->mem, NULL));
        d->format = fmt; d->width = w; d->height = h; d->depth = 1; d->layers = 1; d->levels = 1;
        d->samples = 1; d->aspect = VK_IMAGE_ASPECT_COLOR_BIT; d->base_type = VK_IMAGE_VIEW_TYPE_2D;
        VkCommandBuffer cb = vk_cmd();
        vk_image_init_layout(cb, d->image, d->aspect);
        if (zero) {
            VkClearColorValue z = {{0}};
            VkImageSubresourceRange rg = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cb, d->image, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rg);
        }
    }
    return d;
}

// ================================================================== object motion vectors (DLAA)
// Characters animate without motion vectors: DLSS blurred them or left them jagged. Their skinned G-buffer draws
// (vertex shaders indexing their bones' cbuf) get a variant: the vertex shader runs twice - with last frame's cbufs
// (bones, camera; kept per draw by its mesh) and this frame's - and the fragment shader writes the object's own motion
// (minus the camera's, which DLSS adds from depth) into a 5th G-buffer attachment, DLSS's motion vectors.
// LARECOMP_NO_OBJECT_MV=1 turns it off.
static struct {                  // this draw's motion context (stage_set)
    bool on, valid;
    const void *prev[18];
    uint32_t prev_len[18];
    float u[20];                 // mat4 (last frame's view-projection * this frame's inverse) + flags
} g_mvd;

static int g_mv_off = -1;
int mv_toggle(void) {  // (test: LARECOMP_DLAA_TUNE + F10)
    g_mv_off = g_mv_off == 1 ? 0 : 1;
    LOG("[dlss] object motion vectors %s\n", g_mv_off ? "off" : "on");
    return !g_mv_off;
}
static bool mv_enabled(void) {
    if (g_mv_off < 0) g_mv_off = getenv("LARECOMP_NO_OBJECT_MV") != NULL;
    return !g_mv_off && dlaa_on();
}

static char *str_replace_tok(const char *s, const char *from, const char *to) {  // whole-identifier replace
    size_t fl = strlen(from), tl = strlen(to), n = 0;
    for (const char *p = strstr(s, from); p; p = strstr(p + fl, from)) n++;
    char *o = malloc(strlen(s) + n * (tl + 1) + 1), *d = o;
    const char *p = s;
    for (const char *q; (q = strstr(p, from));) {
        bool lb = q == s || !(isalnum((unsigned char)q[-1]) || q[-1] == '_');
        bool rb = !(isalnum((unsigned char)q[fl]) || q[fl] == '_');
        memcpy(d, p, (size_t)(q - p)); d += q - p;
        if (lb && rb) { memcpy(d, to, tl); d += tl; } else { memcpy(d, from, fl); d += fl; }
        p = q + fl;
    }
    strcpy(d, p);
    return o;
}

// The vertex shader: main() as vs_prev() (cbufs 3+ from pcb*, outputs to dummies) and vs_cur(), both clip positions out
static char *mv_vs_src(const char *src) {
    const char *m = strstr(src, "void main() {");
    if (!m) return NULL;
    size_t hl = (size_t)(m - src);
    char *hdr = malloc(hl + 4096), *h = hdr;
    memcpy(h, src, hl); h += hl;
    h += sprintf(h, "layout(location = 30) out vec4 o_mvc;\nlayout(location = 31) out vec4 o_mvp;\nvec4 g_pp; float g_pps;\n");
    for (int k = 3; k < 18; k++) {  // last frame's copies of the cbufs, at binding 40 + k
        char pat[64];
        snprintf(pat, sizeof pat, "uniform CB%d { uvec4 cb%d[", k, k);
        const char *q = strstr(src, pat);
        if (!q || q > m) continue;
        int nrow = atoi(q + strlen(pat));
        h += sprintf(h, "layout(std140, set = 0, binding = %d) uniform PCB%d { uvec4 pcb%d[%d]; };\n", 40 + k, k, k, nrow);
    }
    for (int i = 0; i < 32; i++) {
        char pat[48];
        snprintf(pat, sizeof pat, "out vec4 v%d;", i);
        const char *q = strstr(src, pat);
        if (q && q < m) h += sprintf(h, "vec4 pv%d;\n", i);
    }
    *h = 0;
    char *prev = strdup(m), *t;
    for (int k = 3; k < 18; k++) {
        char a[16], b[16];
        snprintf(a, sizeof a, "cb%d", k), snprintf(b, sizeof b, "pcb%d", k);
        t = str_replace_tok(prev, a, b); free(prev); prev = t;
    }
    for (int i = 0; i < 32; i++) {
        char a[16], b[16];
        snprintf(a, sizeof a, "v%d", i), snprintf(b, sizeof b, "pv%d", i);
        t = str_replace_tok(prev, a, b); free(prev); prev = t;
    }
    t = str_replace_tok(prev, "gl_Position", "g_pp"); free(prev); prev = t;
    t = str_replace_tok(prev, "gl_PointSize", "g_pps"); free(prev); prev = t;
    t = str_replace_tok(prev, "main", "vs_prev"); free(prev); prev = t;
    char *cur = str_replace_tok(m, "main", "vs_cur");
    const char *tail = "void main() { vs_prev(); vec4 pp_ = g_pp; vs_cur(); o_mvc = gl_Position; o_mvp = pp_; }\n";
    char *o = malloc(strlen(hdr) + strlen(prev) + strlen(cur) + strlen(tail) + 4);
    sprintf(o, "%s%s\n%s\n%s", hdr, prev, cur, tail);
    free(hdr); free(prev); free(cur);
    return o;
}

// The fragment shader: main() as fs_main(), then the object's motion (UV units, to last frame) into color 4
static char *mv_fs_src(const char *src) {
    const char *m = strstr(src, "void main() {");
    if (!m) return NULL;
    static const char *decl =
        "layout(location = 30) in vec4 i_mvc;\nlayout(location = 31) in vec4 i_mvp;\nlayout(location = 4) out vec4 o_mv;\n"
        "layout(std140, set = 1, binding = 60) uniform MVU { mat4 mvM; vec4 mvP; };\n";
    static const char *tail =
        "void main() {\n  fs_main();\n  o_mv = vec4(0.0);\n"
        "  if (mvP.x > 0.5 && i_mvp.w > 0.0 && i_mvc.w > 0.0) {\n"
        "    vec4 g = i_mvc;\n"  // (undo the translator's epilogue: the game's own clip space)
        "    if (mvP.y > 0.5) g.x = -g.x;\n    if (mvP.z > 0.5) g.y = -g.y;\n    if (mvP.w > 0.5) g.z = 2.0 * g.z - g.w;\n"
        "    vec4 c = mvM * g;\n    if (mvP.y > 0.5) c.x = -c.x;\n    if (mvP.z > 0.5) c.y = -c.y;\n"
        "    if (c.w > 0.0) { vec2 d = i_mvp.xy / i_mvp.w - c.xy / c.w; o_mv = vec4(0.5 * d.x, -0.5 * d.y, 0.0, 0.0); }\n"
        "    if (dot(o_mv.xy, o_mv.xy) > 4e-4) o_mv = vec4(0.0);\n"  // (> 2% of the screen a frame: a copy matched to another's last frame)
        "  }\n}\n";
    char *body = str_replace_tok(m, "main", "fs_main");
    size_t hl = (size_t)(m - src);
    char *o = malloc(hl + strlen(decl) + strlen(body) + strlen(tail) + 4);
    memcpy(o, src, hl);
    sprintf(o + hl, "%s%s\n%s", decl, body, tail);
    free(body);
    return o;
}

static void mv_build(void *arg) {
    Prog *pm = arg;
    char *vs = mv_vs_src(pm->vi->msl), *fs = vs ? mv_fs_src(pm->fi->msl) : NULL;
    size_t nv = 0, nf = 0;
    uint32_t *sv = fs ? vk_glsl_to_spirv(vs, STAGE_VERTEX, "vs", &nv) : NULL;
    uint32_t *sf = sv ? vk_glsl_to_spirv(fs, STAGE_FRAGMENT, "fs", &nf) : NULL;
    if (getenv("LARECOMP_SAVE_MSL") && vs && fs) { save_glsl(vs, pm->code[0], "mv_vs"); save_glsl(fs, pm->code[1], "mv_fs"); }
    free(vs); free(fs);
    if (!sv || !sf) {
        static int n;
        if (n++ < 5) LOG("[dlss] motion variant of vs %llx failed to build\n", (unsigned long long)pm->code[0]);
        free(sv); free(sf);
        set_state(&pm->state, 2);
        return;
    }
    pm->nbinds = vk_reflect(sv, nv, pm->binds, MAX_BIND);
    pm->nbinds += vk_reflect(sf, nf, pm->binds + pm->nbinds, MAX_BIND - pm->nbinds);
    pm->vs = vk_module(sv, nv);
    pm->fs = vk_module(sf, nf);
    free(sv); free(sf);
    pm->layout = make_layout(pm, 2, VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT);
    static int nok;
    if (!nok++) LOG("[dlss] object motion vectors: first variant built (vs %llx)\n", (unsigned long long)pm->code[0]);
    set_state(&pm->state, pm->vs && pm->fs ? 1 : 2);
}

// The motion variant of a skinned G-buffer program (queued on first use; NULL until built)
static Prog *mv_prog(Prog *p) {
    if (!p->motion) {
        Prog *pm = calloc(1, sizeof *pm);
        *pm = *p;
        pm->xh[0] = pm->xh[1] = 0;  // (not recorded for the prewarm)
        pm->motion = NULL;
        pm->is_motion = true;
        pm->waited_ns = 0;
        atomic_store(&pm->state, 0);
        p->motion = pm;
        run_job(mv_build, pm);
    }
    Prog *pm = p->motion;
    return atomic_load(&pm->state) == 1 ? pm : NULL;
}

// Last frame's vertex cbufs per mesh (keyed by program and mesh), one per instance: copies of a mesh (a battalion's
// soldiers) are told apart by their cbufs' heads (root bone, world matrix) - each takes last frame's nearest copy.
#define MV_TAB 2048
#define MV_SIG 48
#define MV_MAXINST 512
typedef struct { float sig[MV_SIG]; void *cb[18]; uint32_t len[18]; int prev; } MvInst;
typedef struct { uint64_t key, frame; MvInst *cur, *old; int ncur, nold, capcur; uint8_t *claimed; } MvEnt;
static MvEnt *mv_tab;

static int g_mv_stat[6];  // draws, no slot, no last frame, no camera, valid, cbuf size changed
static uint64_t mv_key(const Prog *p, const CmdRec *r) {
    uint64_t h = hmix(1469598103934665603ull, &st.program->stages[0].code_addr, 8);  // (warm programs: code[0] 0)
    (void)p;
    h = hmix(h, &st.vbuf[0].addr, 8);
    h = hmix(h, &r->x[1], 8 * 4);  // (first index / vertex, count, instances, base vertex)
    return h ? h : 1;
}

static void mv_free_insts(MvInst *v, int n) {
    for (int i = 0; i < n; i++)
        for (int k = 0; k < 18; k++) free(v[i].cb[k]);
    free(v);
}

// Before a motion draw's descriptor sets: last frame's cbufs (if this object was drawn then), this frame's kept for next
static void mv_prepare(const Prog *p, const CmdRec *r, const ShaderKey *vk) {  // (vk: the draw's key; warm programs' vkey is unset)
    if (!mv_tab) mv_tab = calloc(MV_TAB, sizeof *mv_tab);
    uint64_t f = renderer_present_count(), key = mv_key(p, r);
    static uint64_t swept;
    if (swept != f) {  // (meshes not drawn for a while: their copies freed, 64 slots a frame)
        static uint32_t sw;
        swept = f;
        for (int i = 0; i < 64; i++, sw = (sw + 1) & (MV_TAB - 1)) {
            MvEnt *x = &mv_tab[sw];
            if (!x->key || x->frame + 30 >= f) continue;
            mv_free_insts(x->cur, x->ncur), mv_free_insts(x->old, x->nold), free(x->claimed);
            memset(x, 0, sizeof *x);
        }
    }
    uint32_t s = (uint32_t)(key ^ (key >> 29)) & (MV_TAB - 1);
    MvEnt *e = NULL;
    for (int i = 0; i < 16 && !e; i++, s = (s + 1) & (MV_TAB - 1))
        if (mv_tab[s].key == key || !mv_tab[s].key || mv_tab[s].frame + 120 < f) e = &mv_tab[s];
    memset(&g_mvd, 0, sizeof g_mvd);
    g_mvd.on = true;
    g_mv_stat[0]++;
    if (!e) { g_mv_stat[1]++; return; }
    if (e->key != key) {  // (new or stale slot)
        mv_free_insts(e->cur, e->ncur), mv_free_insts(e->old, e->nold), free(e->claimed);
        memset(e, 0, sizeof *e);
        e->key = key;
    }
    if (e->frame != f) {  // first draw of this mesh this frame: last frame's instances become old
        bool recent = e->frame && e->frame + 3 >= f;
        mv_free_insts(e->old, e->nold);
        e->old = recent ? e->cur : NULL, e->nold = recent ? e->ncur : 0;
        if (!recent) mv_free_insts(e->cur, e->ncur);
        e->cur = NULL, e->ncur = e->capcur = 0;
        free(e->claimed);
        e->claimed = e->nold ? calloc((size_t)e->nold, 1) : NULL;
        e->frame = f;
    }
    float sig[MV_SIG] = {0};
    int ns = 0;
    for (int k = 3; k < 18 && ns < MV_SIG; k++) {
        if (!((p->vi->cbuf_mask >> k) & 1) || !st.ubuf[0][k - 3].addr) continue;
        uint64_t len = st.ubuf[0][k - 3].size ? st.ubuf[0][k - 3].size : 256;
        const float *src = (const float *)(uintptr_t)st.ubuf[0][k - 3].addr;
        for (uint64_t i = 0; i < 16 && i * 4 < len && ns < MV_SIG; i++) sig[ns++] = isfinite(src[i]) ? src[i] : 0;
    }
    MvInst *in = NULL;
    for (int i = 0; i < e->ncur && !in; i++)  // (the same object drawn again this frame)
        if (!memcmp(e->cur[i].sig, sig, sizeof sig)) {
            bool same = true;
            for (int k = 3; k < 18 && same; k++)
                if (e->cur[i].cb[k] && (!st.ubuf[0][k - 3].addr ||
                    memcmp(e->cur[i].cb[k], (const void *)(uintptr_t)st.ubuf[0][k - 3].addr, e->cur[i].len[k]))) same = false;
            if (same) in = &e->cur[i];
        }
    if (!in) {
        if (e->ncur >= MV_MAXINST) { g_mv_stat[1]++; return; }
        if (e->ncur == e->capcur) {
            int nc = e->capcur ? e->capcur * 2 : 4;
            MvInst *nv = realloc(e->cur, (size_t)nc * sizeof *nv);
            if (!nv) { g_mv_stat[1]++; return; }
            e->cur = nv, e->capcur = nc;
        }
        in = &e->cur[e->ncur++];
        memset(in, 0, sizeof *in);
        memcpy(in->sig, sig, sizeof sig);
        for (int k = 3; k < 18; k++) {
            if (!((p->vi->cbuf_mask >> k) & 1) || !st.ubuf[0][k - 3].addr) continue;
            uint64_t len = st.ubuf[0][k - 3].size ? st.ubuf[0][k - 3].size : 256;
            if (len > 0x10000) len = 0x10000;
            if (p->vi->cbuf_size[k] && len > p->vi->cbuf_size[k]) len = p->vi->cbuf_size[k];  // (only what it reads)
            in->cb[k] = malloc(len);
            if (in->cb[k]) memcpy(in->cb[k], (const void *)(uintptr_t)st.ubuf[0][k - 3].addr, len), in->len[k] = (uint32_t)len;
        }
        in->prev = -1;  // (last frame's nearest unclaimed copy)
        float best = INFINITY;
        for (int i = 0; i < e->nold; i++) {
            if (e->claimed[i]) continue;
            float d = 0;
            for (int j = 0; j < MV_SIG; j++) d += fabsf(e->old[i].sig[j] - sig[j]);
            if (d < best) best = d, in->prev = i;
        }
        if (in->prev >= 0) e->claimed[in->prev] = 1;
    }
    const MvInst *old = in->prev >= 0 ? &e->old[in->prev] : NULL;
    bool any = false, all = true;
    for (int k = 3; k < 18; k++) {
        if (!((p->vi->cbuf_mask >> k) & 1) || !st.ubuf[0][k - 3].addr) continue;
        g_mvd.prev[k] = old ? old->cb[k] : NULL, g_mvd.prev_len[k] = old ? old->len[k] : 0;
        any = true;
        all &= g_mvd.prev[k] != NULL;
    }
    float inv[16];
    if (!any || !all) { g_mv_stat[2]++; return; }
    if (!g_cam_ok || !g_cam_prev_ok || !m4_inv(inv, g_cam)) { g_mv_stat[3]++; return; }
    g_mv_stat[4]++;
    m4_mul(g_mvd.u, g_cam_prev, inv);
    g_mvd.u[16] = 1;
    g_mvd.u[17] = vk->flip_x, g_mvd.u[18] = vk->flip_y, g_mvd.u[19] = vk->depth_minus_one;
    g_scene_flip[0] = g_mvd.u[17], g_scene_flip[1] = g_mvd.u[18];
    g_mvd.valid = true;
}

// stage_set: the motion variant's extra uniform buffers (set 0 binding 40 + k: last frame's cbuf k; set 1 binding 60)
static bool mv_binding(const VkBindingInfo *b, int stage, ShaderInfo *si, VkDescriptorBufferInfo *out) {
    if (b->type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) return false;
    if (b->binding == 60) {
        float u[20];
        memcpy(u, g_mvd.u, sizeof u);
        if (!g_mvd.valid) u[16] = 0;
        *out = ubo_span(vk_transient_copy(u, sizeof u, 256), sizeof u);
        return true;
    }
    if (b->binding < 43 || b->binding >= 58) return false;
    int k = (int)b->binding - 40;
    VkDescriptorBufferInfo cur = cbuf_info(stage, si, k, 0, 0, 0);
    if (g_mvd.valid && g_mvd.prev[k] && g_mvd.prev_len[k] >= cur.range)
        *out = ubo_span(vk_transient_copy(g_mvd.prev[k], cur.range, 256), cur.range);
    else { if (g_mvd.valid) g_mv_stat[5]++; g_mvd.valid = false, *out = cur; }
    return true;
}


// The camera for DLSS from the view-projection (column-major, VP = P * V with V rigid, P without off-centre terms).
static void dlaa_camera(DlssConstants *k, const float *m) {
    float r0[4] = {m[0], m[4], m[8], m[12]}, r1[4] = {m[1], m[5], m[9], m[13]}, r2[4] = {m[2], m[6], m[10], m[14]},
          r3[4] = {m[3], m[7], m[11], m[15]};
    float p00 = sqrtf(r0[0] * r0[0] + r0[1] * r0[1] + r0[2] * r0[2]), p11 = sqrtf(r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2]);
    float s = sqrtf(r3[0] * r3[0] + r3[1] * r3[1] + r3[2] * r3[2]);
    float v0[4], v1[4], v2[4];
    for (int i = 0; i < 4; i++) v0[i] = r0[i] / p00, v1[i] = r1[i] / p11, v2[i] = -r3[i] / s;  // (w = -z_view)
    float p22 = r2[0] * v2[0] + r2[1] * v2[1] + r2[2] * v2[2], p23 = r2[3] - p22 * v2[3];
    float P[16] = {0};
    P[0] = p00, P[5] = p11, P[10] = p22, P[14] = p23, P[11] = -s;
    memcpy(k->view_to_clip, P, 64);
    m4_inv(k->clip_to_view, P);
    for (int i = 0; i < 3; i++) {
        k->cam_right[i] = v0[i], k->cam_up[i] = v1[i], k->cam_fwd[i] = -v2[i];
        k->cam_pos[i] = -(v0[i] * v0[3] + v1[i] * v1[3] + v2[i] * v2[3]);
    }
    k->depth_inverted = true;  // (GEQUAL depth test: 1 = near)
    k->zfar = fabsf(p22) > 1e-6f ? p23 / p22 : 1e6f;
    k->znear = p23 / (p22 + 1);
    if (!(k->znear > 0)) k->znear = 0.1f;
    if (!(k->zfar > k->znear)) k->zfar = 1e6f;
    k->fov = 2 * atanf(1 / p11);
    k->aspect = p11 / p00;
}

static void post_pass(VkCommandBuffer cb, const char *fs, const VkImageView *srcs, int n, TexBackend *dst, const float pc[8],
                      bool clear);

static VkImageView post_view(TexBackend *d);

// Over the window image the scene was just composited into, before the next draw there.
static void dlaa_run(NvnTexture *tex) {
    TexBackend *c = tex ? (TexBackend *)tex->backend : NULL, *d = g_main_depth ? (TexBackend *)g_main_depth->backend : NULL;
    if (!c || !d || !c->image || !d->image || c->width != d->width || c->height != d->height || !g_cam_ok) return;
    static TexBackend mv_img, out_img;
    g_dlaa_tried = true;  // (one evaluation a frame: a second would feed DLSS the same frame twice)
    static const VkComponentMapping id = {0};
    vk_end_pass();
    VkCommandBuffer cb = vk_cmd();
    TexBackend *mv = dlaa_image(&mv_img, c->width, c->height, VK_FORMAT_R16G16_SFLOAT,
                                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, true);
    if (mv_enabled() && g_mv_img.image && g_mv_img.width == c->width && g_mv_img.height == c->height &&
        g_mv_clear_frame == renderer_present_count()) mv = &g_mv_img;  // (characters' own motion: mv_prog)
    if (g_debug_frame && mv == &g_mv_img) {  // (F9: build/dlaa_motion.png - red / green = x / y x 20, blue = non-zero)
        static TexBackend viz;
        TexBackend *vb = dlaa_image(&viz, mv->width, mv->height, VK_FORMAT_R8G8B8A8_UNORM,
                                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, false);
        VkImageView src[1] = {post_view(mv)};
        float pv[8] = {20.0f};
        post_pass(cb, "mvviz.frag", src, 1, vb, pv, false);
        vk_full_barrier(cb);
        renderer_save_texture_tex(vb, "build/dlaa_motion.png");
        cb = vk_cmd();  // (the save submitted the old command buffer)
    }
    TexBackend *o = dlaa_image(&out_img, c->width, c->height, c->format,
                               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, false);
    DlssImage ic = {c->image, vk_image_view(c, VK_IMAGE_VIEW_TYPE_2D, c->format, 0, 1, 0, 1, id, VK_IMAGE_ASPECT_COLOR_BIT),
                    c->format, c->width, c->height, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
    DlssImage idp = {d->image, vk_image_view(d, VK_IMAGE_VIEW_TYPE_2D, d->format, 0, 1, 0, 1, id, VK_IMAGE_ASPECT_DEPTH_BIT),
                     d->format, d->width, d->height, VK_IMAGE_USAGE_SAMPLED_BIT};
    DlssImage im = {mv->image, vk_image_view(mv, VK_IMAGE_VIEW_TYPE_2D, mv->format, 0, 1, 0, 1, id, VK_IMAGE_ASPECT_COLOR_BIT),
                    mv->format, mv->width, mv->height, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT};
    DlssImage io = {o->image, vk_image_view(o, VK_IMAGE_VIEW_TYPE_2D, o->format, 0, 1, 0, 1, id, VK_IMAGE_ASPECT_COLOR_BIT),
                    o->format, o->width, o->height, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
    DlssConstants k = {0};
    for (int i = 0; i < 16; i++) k.identity[i] = (i % 5) == 0;
    // the depth buffer's clip space: GL-style -1..1 games are drawn with z' = (z + w) / 2 (translate.c depth_minus_one)
    float cur[16], prev[16], D[16];
    memcpy(D, k.identity, 64);
    if (nvn_depth_mode() == 0) D[10] = 0.5f, D[14] = 0.5f;  // (column-major: z row = 0.5 z + 0.5 w)
    m4_mul(cur, D, g_cam);
    m4_mul(prev, D, g_cam_prev);
    dlaa_camera(&k, cur);
    float inv_cur[16], inv_prev[16];
    bool reset = !g_cam_prev_ok || !m4_inv(inv_cur, cur) || !m4_inv(inv_prev, prev);
    if (!reset) {
        m4_mul(k.clip_to_prev_clip, prev, inv_cur);
        m4_mul(k.prev_clip_to_clip, cur, inv_prev);
        // a cut (the view turned more than ~25 degrees): history from another shot would ghost
        float fwd_dot = g_cam[3] * g_cam_prev[3] + g_cam[7] * g_cam_prev[7] + g_cam[11] * g_cam_prev[11];
        if (fwd_dot < 0.9f) reset = true;
    }
    if (reset) memcpy(k.clip_to_prev_clip, k.identity, 64), memcpy(k.prev_clip_to_clip, k.identity, 64);
    k.reset = reset;
    static float sx = 2, sy = 2;  // (LARECOMP_DLSS_JSIGN="x y": the jitter's sign convention, for testing)
    if (sx == 2) {
        sx = sy = 1;  // (the offset the scene was shifted by, in y-down pixels: as UE's DLSS plugin passes it)
        const char *e = getenv("LARECOMP_DLSS_JSIGN");
        if (e) sscanf(e, "%f %f", &sx, &sy);
    }
    float tx = g_jit_mode == 1 || g_jit_mode == 3 ? -1.0f : 1.0f, ty = g_jit_mode == 2 || g_jit_mode == 3 ? -1.0f : 1.0f;
    k.jitter[0] = tx * sx * g_jit[0], k.jitter[1] = ty * sy * g_jit[1];
    if (g_jit_mode_changed) k.reset = true, g_jit_mode_changed = false;
    static int nev, nreset;  // (every 600 evaluations: how often the history was dropped)
    nreset += reset;
    if (++nev % 600 == 0) {
        LOG("[dlss] %d evaluations, %d resets in the last 600\n", nev, nreset), nreset = 0;
        LOG("[dlss] character motion: %d draws, %d no slot, %d no last frame, %d no camera, %d valid, %d size changed\n",
            g_mv_stat[0], g_mv_stat[1], g_mv_stat[2], g_mv_stat[3], g_mv_stat[4], g_mv_stat[5]);
        memset(g_mv_stat, 0, sizeof g_mv_stat);
        dlss_fg_state_log();
    }
    vk_full_barrier(cb);
    extern VkExtent2D vk_frame_extent(void);
    VkExtent2D win = vk_frame_extent();
    // characters animate without motion vectors: DLSS blended their history into a soft face (half the detail of the
    // game's image) - a mask from the G-buffer makes it favour their current frame. LARECOMP_DLAA_CHAR_BIAS=<0..1>.
    if (g_char_bias < 0) g_char_bias = getenv("LARECOMP_DLAA_CHAR_BIAS") ? (float)atof(getenv("LARECOMP_DLAA_CHAR_BIAS")) : 0.0f;
    float cbias = g_char_bias;
    TexBackend *gm = g_gbuf_mat ? (TexBackend *)g_gbuf_mat->backend : NULL;
    DlssImage ib, *pib = NULL;
    if (cbias > 0 && gm && gm->image && gm->width == c->width && gm->height == c->height) {
        static TexBackend bias_img;
        TexBackend *bi = dlaa_image(&bias_img, c->width, c->height, VK_FORMAT_R8G8B8A8_UNORM,
                                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, false);
        VkImageView src[1] = {vk_image_view(gm, VK_IMAGE_VIEW_TYPE_2D, gm->format, 0, 1, 0, 1, id, VK_IMAGE_ASPECT_COLOR_BIT)};
        float pb[8] = {cbias};
        post_pass(cb, "charmask.frag", src, 1, bi, pb, false);
        vk_full_barrier(cb);
        static int nlog;
        if (!nlog++) LOG("[dlss] character mask %ux%u from material target fmt %d, bias %.2f\n", bi->width, bi->height, gm->format, cbias);
        if (g_debug_frame) {  // (F9: the mask and its source)
            renderer_save_texture_tex(bi, "build/dlaa_charmask.png");
            renderer_save_texture_tex(gm, "build/dlaa_material.png");
            cb = vk_cmd();
        }
        ib = (DlssImage){bi->image, post_view(bi), bi->format, bi->width, bi->height,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
        pib = &ib;
    }
    // frame generation: the camera's motion added here (SL's own from depth, merged with ours, haloed the characters).
    // LARECOMP_MV_SL_CAMERA=1: the old way.
    static int sl_cam = -1;
    if (sl_cam < 0) sl_cam = getenv("LARECOMP_MV_SL_CAMERA") != NULL;
    if (!sl_cam) {
        static TexBackend full_img;
        TexBackend *fm = dlaa_image(&full_img, c->width, c->height, VK_FORMAT_R16G16_SFLOAT,
                                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT, false);
        float pc80[20];
        memcpy(pc80, k.clip_to_prev_clip, 64);
        pc80[16] = g_scene_flip[0], pc80[17] = g_scene_flip[1], pc80[18] = pc80[19] = 1;
        if (getenv("LARECOMP_OBJMV_SCALE")) sscanf(getenv("LARECOMP_OBJMV_SCALE"), "%f %f", &pc80[18], &pc80[19]);  // (test)
        VkImageView src[2] = {idp.view, im.view};
        g_post_pc80 = pc80;
        post_pass(cb, "mvfull.frag", src, 2, fm, pc80, false);
        vk_full_barrier(cb);
        im = (DlssImage){fm->image, post_view(fm), fm->format, fm->width, fm->height,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
        k.camera_motion = true;
        if (g_debug_frame) {  // (F9: build/dlaa_motion_full.png, x / y x 20 around grey)
            static TexBackend viz2;
            TexBackend *vb = dlaa_image(&viz2, fm->width, fm->height, VK_FORMAT_R8G8B8A8_UNORM,
                                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, false);
            VkImageView s1[1] = {im.view};
            float pv[8] = {20.0f};
            post_pass(cb, "mvviz.frag", s1, 1, vb, pv, false);
            vk_full_barrier(cb);
            renderer_save_texture_tex(vb, "build/dlaa_motion_full.png");
            cb = vk_cmd();
        }
    }
    if (dlss_evaluate(cb, &ic, &idp, &im, pib, &io, &k, g_settings.frame_gen & 3, win.width, win.height)) {
        vk_full_barrier(cb);
        float boost, sh = sharpen_level(&boost);  // (RCAS: exp2(-stops); 0.2 is AMD's default, fairly strong)
        // internal res above the window: sharpen after the resolve instead (the box average undoes RCAS done here)
        g_dlaa_sharpen_late = sh >= 0 && g_resolve_down && !getenv("LARECOMP_DLAA_SHARPEN_EARLY");
        if (sh >= 0 && !g_dlaa_sharpen_late && c->format == VK_FORMAT_R8G8B8A8_UNORM) {  // (the scene only: the UI is drawn after this)
            float pr[8] = {exp2f(-sh), boost > 0, boost, 0, (float)o->width, (float)o->height};
            VkImageView in[3] = {io.view};
            post_pass(cb, "rcas.frag", in, 1, c, pr, false);
        } else {
            VkImageCopy cp = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                              .extent = {c->width, c->height, 1}};
            vkCmdCopyImage(cb, o->image, VK_IMAGE_LAYOUT_GENERAL, c->image, VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
        }
        g_dlaa_ran = true;
        static int n;
        if (!n++) LOG("[dlss] first DLAA frame %ux%u (fov %.1f, near %.2f far %.0f)\n", c->width, c->height, k.fov * 57.2958f,
                      k.znear, k.zfar);
    }
    vk_full_barrier(cb);
    ss_forget();  // (Streamline bound its own pipelines and sets)
    st.rt_dirty = true;  // (the next draw begins its pass again)
}

void vk_dlaa_present(NvnTexture *t) {  // (device.c, at present: the scene was the frame's last draw)
    if (!g_dlaa_pending) return;
    g_dlaa_pending = false;
    dlaa_run(t);
}

static void draw_(CmdRec *r, bool indexed);
static void draw(CmdRec *r, bool indexed) {
    PROF_T0();
    draw_(r, indexed);
    if (g_debug_frame) {
        if (nan_scan_on < 0) nan_scan_on = getenv("LARECOMP_NAN_SCAN") != NULL;
        if (nan_scan_on) nan_scan();
    }
    PROF_ADD(prof_draw);
}
static uint64_t prof_tp_;
static void draw_(CmdRec *r, bool indexed) {
    if (prof) prof_tp_ = SDL_GetTicksNS();
    if (!st.program) SKIP(SK_NO_PROG, "draw mode %llu with no program bound\n", (unsigned long long)r->x[0]);
    dlaa_track();
    if (g_dlaa_pending && st.rt.ncolor && st.rt.color[0] && renderer_texture_presented(st.rt.color[0])) {
        g_dlaa_pending = false;  // (the scene is in the window image: DLAA before the HUD goes on it)
        dlaa_run(st.rt.color[0]);
    }
    ShaderKey vkey = {0}, fkey = {0};
    vkey.glsl = fkey.glsl = 1;
    vkey.flip_y = fkey.flip_y = nvn_window_origin() == 0;
    vkey.depth_minus_one = nvn_depth_mode() == 0;
    for (int i = 0; i < st.nattrib; i++) vkey.attr_kind[i] = (uint8_t)attr_kind(st.attrib[i] & 0xFF);
    for (int i = 0; i < st.rt.ncolor; i++) if (st.rt.color[i]) fkey.rt_kind[i] = (uint8_t)rt_kind(st.rt.color[i]->format);
    {   // blended, no depth write, into a scaled target (UI): sharp-bilinear 2D lookups, so atlas pieces magnified past
        // 1:1 don't bleed in their neighbours (a line in the dialogue box at 2x; translate.c glsl_sample)
        static int off = -1;
        if (off < 0) off = getenv("LARECOMP_NO_UI_SHARP") != NULL;
        float usx = 1, usy = 1;
        if (!off && st.rt.ncolor && st.rt.color[0]) renderer_texture_scale(st.rt.color[0], &usx, &usy);
        if ((usx != 1 || usy != 1) && (st.color_state[0] & 1) && !(st.ds[0] && st.ds[1])) {
            fkey.attr_kind[0] = 1;
            fkey.attr_kind[1] = (uint8_t)fminf(255.0f, lroundf(fminf(usx, usy) * 16));  // (scale x16: 1:1-ish lookups only)
        }
    }
    if (getenv("LARECOMP_LOG_PROJ") && st.ubuf[0][0].addr && st.ubuf[0][0].size >= 64) {
        // (debug) a 3D draw's view-projection (c3, 4x4): x scale / y scale = the camera's 1 / aspect
        static uint64_t lastf = ~0ull;
        uint64_t f = renderer_present_count();
        const float *m = (const float *)(uintptr_t)st.ubuf[0][0].addr;
        float lx = sqrtf(m[0] * m[0] + m[4] * m[4] + m[8] * m[8]), ly = sqrtf(m[1] * m[1] + m[5] * m[5] + m[9] * m[9]);
        bool persp = fabsf(m[3]) + fabsf(m[7]) + fabsf(m[11]) > 0.01f && lx > 0.01f && ly > 0.01f;
        extern float g_proj_lx, g_proj_ly;
        extern float widescreen_factor(void);  // (the main camera: the widened shape)
        bool main_cam = fabsf(lx / ly - 0.5625f / fmaxf(widescreen_factor(), 1.0f)) < 0.005f &&
                        fabsf(m[3]) < 1e-4f;  // (a 4x4 projection, not the 2D UI's scale + offset)
        extern float g_proj_vp[16];
        if (persp && main_cam && st.rt.ncolor && st.rt.color[0] && st.rt.color[0]->width == 1920) {
            g_proj_lx = lx, g_proj_ly = ly;
            memcpy(g_proj_vp, m, 64);
        }
        if (persp && f / 60 != lastf / 60) {
            lastf = f;
            LOG("[proj] f%llu vs %llx c3 %llx: x/y %.4f (16:9 = 0.5625) rt %ux%u\n", (unsigned long long)f,
                (unsigned long long)st.program->stages[0].code_addr, (unsigned long long)st.ubuf[0][0].addr, lx / ly,
                st.rt.ncolor && st.rt.color[0] ? st.rt.color[0]->width : 0, st.rt.ncolor && st.rt.color[0] ? st.rt.color[0]->height : 0);
        }
    }
    Prog *pr = get_program(st.program, &vkey, &fkey);
    if (!pr || !wait_built(&pr->state, &pr->waited_ns))
        SKIP(SK_PROG, "vs %llx fs %llx state %d\n", (unsigned long long)st.program->stages[0].code_addr,
             (unsigned long long)st.program->stages[1].code_addr, pr ? atomic_load(&pr->state) : -1);
    {   // buffer textures bound to the fragment shader's 1D slots: the variant that reads them from the guest heap
        uint32_t m = buffer_texture_mask(1, pr->fi);
        if (m) {
            key_set_buf_mask(&fkey, m);
            pr = get_program(st.program, &vkey, &fkey);
            if (!pr || !wait_built(&pr->state, &pr->waited_ns)) SKIP(SK_PROG, "buffer-texture variant building\n");
        }
    }
    if (g_mv_pass && pr->vi->cbuf_dynamic && !pr->vi->uses_gmem && !pr->fi->uses_gmem && st.ds[0]) {  // (object motion)
        Prog *pm = mv_prog(pr);
        if (pm) { mv_prepare(pr, r, &vkey); pr = pm; }
    }
    g_mirror_w = 0;
    {   // mirror mode (runtime/hle/mirror.c; as renderer.m): the scene's composite into the window image is mirrored
        extern bool mirror_draw(bool composite);
        bool composite = false;
        if (st.rt.ncolor && st.rt.color[0] && renderer_texture_presented(st.rt.color[0]) && !(st.color_state[0] & 1) &&
            r->x[2] <= 6 && st.viewport[0][0] == 0 && st.viewport[0][1] == 0 &&
            st.viewport[0][2] >= st.rt.color[0]->width && st.viewport[0][3] >= st.rt.color[0]->height && pr->fi->ntex) {
            NvnTexture *t0 = shader_texture_object(1, &pr->fi->tex[0]);
            composite = t0 && t0->width == st.rt.color[0]->width && t0->height == st.rt.color[0]->height;
        }
        if (mirror_draw(composite)) {
            vkey.flip_x = fkey.flip_x = true;
            pr = get_program(st.program, &vkey, &fkey);
            if (!pr || !wait_built(&pr->state, &pr->waited_ns)) SKIP(SK_PROG_MIRROR, "\n");
            g_mirror_w = (float)st.rt.color[0]->width;
        }
    }
    {   // fragment-shader hash: interior backdrop detection (widescreen), skip lists, item twinkle (renderer.m)
        static uint64_t last_fs, last_hash;
        uint64_t fs = st.program->stages[1].code_addr;
        if (fs != last_fs) {
            last_fs = fs;
            last_hash = fs ? hmix(1469598103934665603ull, (const void *)(uintptr_t)(fs + 0x30), 0x200) : 0;
        }
        static uint64_t skip[16];
        static int nskip = -1;
        if (nskip < 0) {
            nskip = 0;
            for (const char *p = getenv("LARECOMP_SKIP_FSHASH"); p && *p && nskip < 16;) {
                skip[nskip++] = strtoull(p, (char **)&p, 16);
                while (*p == ',') p++;
            }
        }
        for (int i = 0; i < nskip; i++) if (last_hash == skip[i]) SKIP(SK_FS_LIST, "%016llx\n", (unsigned long long)last_hash);
        extern bool pot_level_active(void);
        if (last_hash == 0xa2177dbb72197bd9ull && pot_level_active()) SKIP(SK_POT, "\n");
        if (g_debug_frame && getenv("LARECOMP_LOG_FSHASH")) LOG("[fshash] fs=%llx hash %016llx\n", (unsigned long long)fs, (unsigned long long)last_hash);
        g_cur_fs_hash = last_hash;
        extern void widescreen_backdrop_seen(uint64_t hash);
        if (last_hash) widescreen_backdrop_seen(last_hash);
        extern void widescreen_scene_ubo(const float *u, uint64_t size);  // (the camera's view-projection)
        if (st.ubuf[0][5].addr) widescreen_scene_ubo((const float *)(uintptr_t)st.ubuf[0][5].addr, st.ubuf[0][5].size);
    }
    if (g_debug_frame) debug_draw(pr, r, indexed);
    if (ct_on()) ct_draw(pr, "draw");
    alias_draw_check();
    resolve_feedback(pr);
    snap_reason = "draw-depthwrite";
    if (st.rt.depth && st.ds[0] && st.ds[1]) snapshots_invalidate(st.rt.depth);
    snap_reason = "draw-colorwrite";
    for (int i = 0; i < st.rt.ncolor; i++)
        if (st.rt.color[i] && ((st.channel_mask >> (i * 4)) & 0xF)) snapshots_invalidate(st.rt.color[i]);
    bf_gpu_copy(pr, 0, pr->vi);  // (light lists: copied outside the pass)
    bf_gpu_copy(pr, 1, pr->fi);
    begin_pass();
    VkCommandBuffer cb = vk_cmd();

    uint32_t mode = (uint32_t)r->x[0];
    VkPrimitiveTopology topo;
    uint8_t cls;
    bool native = topology(mode, &topo, &cls);
    if (!native && mode != 6 && mode != 7 && mode != 9) SKIP(SK_TOPO, "draw mode %u\n", mode);

    uint64_t prof_tk_ = prof ? SDL_GetTicksNS() : 0;
    if (prof) prof_pre += prof_tk_ - prof_tp_;
    // ---- pipeline
    PipeKey k;
    memset(&k, 0, sizeof k);
    k.prog = pr;
    k.ncolor = pass_ncolor;
    memcpy(k.color, pass_color, sizeof k.color);
    k.depth = pass_depth;
    k.stencil = pass_stencil;
    k.samples = pass_samples;
    for (int i = 0; i < 32; i++) {
        if (!(pr->vi->attr_mask & (1u << i))) continue;
        k.attr_mask |= 1u << i;
        uint32_t a = i < st.nattrib ? st.attrib[i] : 0x2E;
        uint32_t fmt = a & 0xFF, stream = (a >> 8) & 0x1F, off = a >> 13;
        if (i >= st.nattrib || fmt == 0 || stream >= 16) {  // unbound attribute: (0, 0, 0, 1) from a constant stream
            k.attr_fmt[i] = VK_FORMAT_R32G32B32A32_SFLOAT;
            k.attr_bind[i] = ZERO_BINDING;
            k.used_streams |= 1u << ZERO_BINDING;
            continue;
        }
        k.attr_fmt[i] = vertex_format_supported(vertex_format(fmt));
        k.attr_off[i] = off;
        k.attr_bind[i] = stream;
        k.used_streams |= 1u << stream;
    }
    for (int s = 0; s < 16; s++) {
        if (!(k.used_streams & (1u << s))) continue;
        k.stride[s] = s < st.nstream ? st.stream[s][0] : 0;
        k.divisor[s] = s < st.nstream && k.stride[s] ? st.stream[s][1] : 0;
        if (k.divisor[s] > 1 && !g_vk.divisor) k.divisor[s] = 1;
    }
    for (uint32_t i = 0; i < k.ncolor && i < 8; i++) {
        if (!st.rt.color[i] || k.color[i] == VK_FORMAT_UNDEFINED) continue;
        // Targets the fragment shader doesn't output keep their contents on Metal, but are undefined in Vulkan
        // (NVIDIA writes black): mask them. Foliage depth pre-passes left black silhouettes (user 2026-10-08).
        k.write_mask[i] = ((pr->fi->out_mask >> (i * 4)) & 0xF) ? (st.channel_mask >> (i * 4)) & 0xF : 0;
        if (((st.color_state[0] >> i) & 1) && !rt_kind(st.rt.color[i]->format)) {
            k.blend_on |= (uint8_t)(1u << i);
            memcpy(k.blend[i], st.blend[i], 8);
        }
    }
    if (k.ncolor > 4) {  // (the motion target: written by motion variants only)
        k.write_mask[4] = pr->is_motion ? 0x3 : 0;
        k.blend_on &= 0xF;
    }
    k.a2c = st.ms[0xC] != 0;
    k.topo_class = cls;
    k.poly_fill = st.poly[2] != 1;
    Pipe *pp = pipe_get(&k);
    if (!wait_built(&pp->state, &pp->waited_ns)) SKIP(SK_PIPE, "state %d\n", atomic_load(&pp->state));
    ss_enabled();
    if (ss_need(SS_PIPE, ss.pipe == pp->pipe)) { ss.pipe = pp->pipe; vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pp->pipe); }
    uint64_t prof_td_ = prof ? SDL_GetTicksNS() : 0;
    if (prof) prof_key += prof_td_ - prof_tk_;

    // ---- depth / stencil (dynamic)
    static int nodt = -1;
    if (nodt < 0) nodt = getenv("LARECOMP_NO_DEPTH_TEST") != NULL;
    // (test) LARECOMP_NO_DEPTH_TEST_FS=<fragment shader hash>[,...]: those draws only, without depth test: their whole
    // shape on top of everything (is stray geometry misplaced, or just not hidden?)
    static uint64_t nodt_fs[8];
    static int nnodt_fs = -1;
    if (nnodt_fs < 0) {
        nnodt_fs = 0;
        for (const char *p = getenv("LARECOMP_NO_DEPTH_TEST_FS"); p && *p && nnodt_fs < 8;) {
            nodt_fs[nnodt_fs++] = strtoull(p, (char **)&p, 16);
            while (*p == ',') p++;
        }
    }
    bool nodt_here = false;
    for (int i = 0; i < nnodt_fs; i++) nodt_here |= g_cur_fs_hash == nodt_fs[i];
    bool dt = st.ds[0] && !nodt && !nodt_here && pass_depth;
    if (ss_need(SS_DT, ss.dt == (VkBool32)dt)) { ss.dt = dt; vkCmdSetDepthTestEnable(cb, dt); }
    VkBool32 dw = dt && st.ds[1];
    if (ss_need(SS_DW, ss.dw == dw)) { ss.dw = dw; vkCmdSetDepthWriteEnable(cb, dw); }
    VkCompareOp dop = dt ? compare_fn(st.ds[2]) : VK_COMPARE_OP_ALWAYS;
    // EQUAL against a depth pre-pass written by a different vertex shader (foliage, trees) needs both shaders to land
    // on identical depths. They do since the translator stopped using unpackHalf2x16 (NVIDIA ran the half math after
    // it in fp16, differently per shader: black slashes in foliage). LARECOMP_VK_EQUAL_TOL=1 brings back the old
    // workaround: LESS_OR_EQUAL with a few ulps of bias towards the camera, which also let walls paint over the
    // plants stuck flat on them (beach) - so it is off.
    static int tol = -1;
    if (tol < 0) tol = getenv("LARECOMP_VK_EQUAL_TOL") != NULL;
    bool equal_tol = dop == VK_COMPARE_OP_EQUAL && tol;
    if (equal_tol) dop = VK_COMPARE_OP_LESS_OR_EQUAL;
    if (ss_need(SS_DOP, ss.dop == dop)) { ss.dop = dop; vkCmdSetDepthCompareOp(cb, dop); }
    bool sten = st.ds[3] && pass_stencil;
    if (ss_need(SS_STEN, ss.sten == (VkBool32)sten)) { ss.sten = sten; vkCmdSetStencilTestEnable(cb, sten); }
    if (sten) {
        uint32_t ops = st.ds[5] | st.ds[6] << 8 | st.ds[7] << 16;
        for (int f = 0; f < 2; f++) {
            uint32_t o = f ? ops >> 12 : ops;
            VkStencilFaceFlags face = f ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT;
            uint32_t v[7] = {stencil_op(o & 0xF), stencil_op((o >> 8) & 0xF), stencil_op((o >> 4) & 0xF),
                             compare_fn(f ? st.ds[4] >> 4 : st.ds[4] & 0xF), st.stencil_vmask[f] & 0xFF, st.stencil_mask[f] & 0xFF,
                             st.stencil_ref[f]};
            if (!ss_need(f ? SS_SFACE1 : SS_SFACE0, !memcmp(v, ss.sface[f], sizeof v))) continue;
            memcpy(ss.sface[f], v, sizeof v);
            vkCmdSetStencilOp(cb, face, (VkStencilOp)v[0], (VkStencilOp)v[1], (VkStencilOp)v[2], (VkCompareOp)v[3]);
            vkCmdSetStencilCompareMask(cb, face, v[4]);
            vkCmdSetStencilWriteMask(cb, face, v[5]);
            vkCmdSetStencilReference(cb, face, v[6]);
        }
    }
    if (ss_need(SS_BC, !memcmp(ss.bc, st.blend_color, sizeof ss.bc))) { memcpy(ss.bc, st.blend_color, sizeof ss.bc); vkCmdSetBlendConstants(cb, st.blend_color); }

    // ---- rasterizer
    uint8_t cull = st.poly[0];
    if (vkey.flip_x) cull = 0;  // (mirror mode's composite: one full-screen quad, either winding)
    VkCullModeFlags cm = cull == 1 ? VK_CULL_MODE_FRONT_BIT : cull == 2 ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
    if (ss_need(SS_CULL, ss.cull == cm)) { ss.cull = cm; vkCmdSetCullMode(cb, cm); }
    bool ccw = st.poly[1] == 1;
    static int flipw = -1;  // LARECOMP_VK_FLIP_WINDING: invert the front-face mapping (A/B against Metal)
    if (flipw < 0) flipw = getenv("LARECOMP_VK_FLIP_WINDING") != NULL;
    bool front_ccw = (ccw != vkey.flip_y) != (flipw != 0);
    VkFrontFace ff = front_ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
    if (ss_need(SS_FF, ss.ff == ff)) { ss.ff = ff; vkCmdSetFrontFace(cb, ff); }
    VkPrimitiveTopology tp = native ? topo : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    if (ss_need(SS_TOPO, ss.topo == tp)) { ss.topo = tp; vkCmdSetPrimitiveTopology(cb, tp); }
    VkBool32 pre = st.prim_restart && native && (mode == 3 || mode == 5);
    if (ss_need(SS_PRE, ss.pre == pre)) { ss.pre = pre; vkCmdSetPrimitiveRestartEnable(cb, pre); }
    {
        float f[3] = {0, 0, 0};
        static int off = -1;
        if (off < 0) off = getenv("LARECOMP_NO_DEPTH_BIAS") != NULL;
        bool on = (st.poly[3] & 4) && !off;
        if (on) memcpy(f, st.poly_offset, sizeof f);
        if (equal_tol) { f[1] -= 8; on = true; }  // (see the EQUAL comment above: a few depth ulps nearer)
        if (ss_need(SS_DBE, ss.dbe == (VkBool32)on)) { ss.dbe = on; vkCmdSetDepthBiasEnable(cb, on); }
        // a DEPTH24 target standing in as float depth (0x33 always, 0x35 where the GPU has no D24S8): the bias's units are
        // 2^-24 steps as on the Switch, not the float format's own (device.c: VK_EXT_depth_bias_control)
        bool d24_as_float = st.rt.depth && (st.rt.depth->format == 0x33 ||
                            (st.rt.depth->format == 0x35 && pass_depth == VK_FORMAT_D32_SFLOAT_S8_UINT));
        float key[4] = {f[0], f[1], f[2], d24_as_float ? 1.0f : 0.0f};
        if (ss_need(SS_DB, !memcmp(ss.db, key, sizeof key))) {
            memcpy(ss.db, key, sizeof key);
            if (d24_as_float && g_vk.bias_unorm) {
                VkDepthBiasRepresentationInfoEXT rep = {.sType = VK_STRUCTURE_TYPE_DEPTH_BIAS_REPRESENTATION_INFO_EXT,
                    .depthBiasRepresentation = VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT,
                    .depthBiasExact = g_vk.bias_exact};
                VkDepthBiasInfoEXT bi = {.sType = VK_STRUCTURE_TYPE_DEPTH_BIAS_INFO_EXT, .pNext = &rep, .depthBiasConstantFactor = f[1],
                                         .depthBiasClamp = f[2], .depthBiasSlopeFactor = f[0]};
                g_vk.set_bias2(cb, &bi);
            } else {
                vkCmdSetDepthBias(cb, f[1], f[2], f[0]);
            }
        }
    }

    // ---- textures first (their scales feed the widescreen decision and the rscale table), then the viewport
    float *vp = st.viewport[0];
    float zn = st.depth_range[0][0], zf = st.depth_range[0][1];
    vp_sx = pass_sx; vp_sy = pass_sy; vp_ox = 0; vp_oy = 0; sc_box = 0; g_label_dx = 0;
    {   // the Items page's cursor brackets while the cursor is on the companion card (runtime/hle/pets.c)
        extern bool pets_hide_item_cursor(void);
        extern bool texpatch_is_item_cursor(int id);
        if (pets_hide_item_cursor())
            for (int i = 0; i < pr->fi->ntex; i++) {
                NvnTexture *t = shader_texture_object(1, &pr->fi->tex[i]);
                const TexBackend *b = t ? (const TexBackend *)t->backend : NULL;
                if (b && b->tp > 0 && texpatch_is_item_cursor(b->tp - 1)) SKIP(SK_CURSOR, "\n");
            }
    }
    float tint[8] = {0};
    if (pr->fi->uses_tint) {
        extern bool pot_tint(const float *m, float tint[8], uint64_t mesh);
        extern bool pot_glow_active(void);
        if (st.ubuf[0][2].addr && st.ubuf[0][2].size >= 48 && pot_glow_active())
            pot_tint((const float *)(uintptr_t)st.ubuf[0][2].addr, tint, st.vbuf[0].addr);
        // the gold Escape Bottle (runtime/hle/escape.c): draws sampling the bottle's item icon / model map
        // and the HUD hearts (heart colour, Cosmetics tab)
        extern int texpatch_bottle_kind(int id);
        extern int texpatch_heart_kind(int id);
        extern void texpatch_heart_tint(int kind, float tint[8]);
        int bottle = 0, heart = 0;
        for (int i = 0; i < pr->fi->ntex; i++) {
            NvnTexture *t = shader_texture_object(1, &pr->fi->tex[i]);
            const TexBackend *b = t ? (const TexBackend *)t->backend : NULL;
            if (b && b->tp > 0 && !bottle) bottle = texpatch_bottle_kind(b->tp - 1);
            if (b && b->tp > 0 && !heart) heart = texpatch_heart_kind(b->tp - 1);
        }
        if (heart) texpatch_heart_tint(heart, tint);
        if (bottle) {
            extern void escape_bottle_tint(int kind, const float *ui, uint32_t ui_size, float tint[8]);
            int u = bottle == 3 ? 3 : 0;  // (the fairy model: its bone matrices, for where it stands - souls.c)
            escape_bottle_tint(bottle, (const float *)(uintptr_t)st.ubuf[0][u].addr, (uint32_t)st.ubuf[0][u].size, tint);
        }
    }
    // fragment textures are bound before deciding the viewport (tex_sc / tex_wh), rscale depends on the decision:
    // build the vertex set now and the fragment set after the decision (stage_set fills tex_sc for stage 1).
    if (prof) prof_dyn += SDL_GetTicksNS() - prof_td_;
    PROF_T0();
    VkDescriptorSet vset = stage_set(pr, pr->set_layout[0], 0, 0, pr->vi, NULL, NULL, 0, 0, 0, NULL);
    PROF_ADD(prof_sets);
    VkDescriptorImageInfo fii[64];
    uint64_t prof_tx_ = prof ? SDL_GetTicksNS() : 0;
    for (int i = 0; i < pr->fi->ntex && i < 64; i++) {
        float sc[2]; uint32_t wh[2];
        texture_info(1, pr->fi, i, &fii[i], sc, wh);
        tex_sc[1][i][0] = sc[0]; tex_sc[1][i][1] = sc[1]; tex_wh[1][i][0] = wh[0]; tex_wh[1][i][1] = wh[1];
    }
    if (prof) prof_tex += SDL_GetTicksNS() - prof_tx_;
    if (st.rt.ncolor && st.rt.color[0] && st.rt.color[0]->backend && !renderer_texture_presented(st.rt.color[0])) {
        // a movie frame converted into a (widened) target stays 2D; any other draw into it makes it a scene buffer
        // (a plain copy: not blended or depth-tested - a monastery overlay quad over the scene reads only uploaded
        // textures, and marked the scene buffer a movie: the scene boxed as 2D, no DLAA)
        bool conv = pr->fi->ntex > 0 && r->x[2] <= 6 && !(st.color_state[0] & 1) && !st.ds[0];
        for (int i = 0; i < pr->fi->ntex && i < 64; i++) conv &= tex_video[i];
        ((TexBackend *)st.rt.color[0]->backend)->video = conv;
    }
    if (fabsf(pass_sx - pass_sy) > 0.01f * pass_sy && st.rt.ncolor && renderer_texture_presented(st.rt.color[0])) {
        // Draws into the swapchain image (see renderer.m): composites fill the window, UI goes in a centred 16:9 box.
        bool reads_scene = false, video = false;  // (movie frames: 2D, boxed with their screen edges)
        for (int i = 0; i < pr->fi->ntex && i < 64; i++) video |= tex_video[i];
        for (int i = 0; i < pr->fi->ntex && i < 64; i++)
            if (!tex_video[i] && (tex_sc[1][i][0] != 1 || tex_sc[1][i][1] != 1) && tex_wh[1][i][0] == st.rt.color[0]->width &&
                tex_wh[1][i][1] == st.rt.color[0]->height)
                reads_scene = true;
        bool fullscreen_quad = r->x[2] <= 6 && st.viewport[0][0] == 0 && st.viewport[0][1] == 0 &&
                               st.viewport[0][2] >= st.rt.color[0]->width && st.viewport[0][3] >= st.rt.color[0]->height;
        bool fullscreen_fill = pr->fi->ntex == 0 && fullscreen_quad;
        if (fullscreen_quad && !reads_scene && !video && pr->fi->ntex == 1 && (tex_sc[1][0][0] != 1 || tex_sc[1][0][1] != 1)) reads_scene = true;
        {   // after a 3D scene, the frame's first textured fullscreen quad here is its composite, whatever it reads (the
            // monastery at some angles composites from smaller buffers only: boxed as UI, no DLAA)
            static uint64_t first_pres = ~0ull;
            uint64_t now_f = renderer_present_count();
            bool first = first_pres != now_f;
            first_pres = now_f;
            if (first && fullscreen_quad && !video && pr->fi->ntex > 0 && g_scene_frame == now_f) reads_scene = true;
        }
        static uint64_t wide_frame = ~0ull;
        if (reads_scene) wide_frame = renderer_present_count();
        if (reads_scene && fullscreen_quad && !g_dlaa_tried && g_scene_frame == renderer_present_count() && dlaa_on())
            g_dlaa_pending = g_dlaa_seen = true;  // (the first scene composite only: overlays reading depth come after it)
        bool fade_fill = false;
        if (fullscreen_fill && st.ubuf[1][0].addr && st.ubuf[1][0].size >= 12) {
            const float *c = (const float *)(uintptr_t)st.ubuf[1][0].addr;
            bool black = c[0] < 0.01f && c[1] < 0.01f && c[2] < 0.01f, white = c[0] > 0.99f && c[1] > 0.99f && c[2] > 0.99f;
            fade_fill = black || white;
        }
        bool composite = reads_scene || fade_fill || (fullscreen_fill && wide_frame == renderer_present_count());
        if (!composite) wide_frame = ~0ull;
        bool full_vp = st.viewport[0][0] == 0 && st.viewport[0][2] >= st.rt.color[0]->width;
        static int vs_box = -1;
        if (vs_box < 0) vs_box = getenv("LARECOMP_NO_VS_BOX") == NULL;
        if (!reads_scene && full_vp && vs_box && pass_sx > pass_sy) {  // (fills: their screen-edge vertices stay put)
            // FE3H: boxed per vertex - 2D (w == 1) in the 16:9 box, 3D overlays (unit rings, tiles) keep the wide camera
            uint16_t q = (uint16_t)lroundf(pass_sy / pass_sx * 65535);
            vkey.rt_kind[0] = 1; vkey.rt_kind[1] = (uint8_t)q; vkey.rt_kind[2] = (uint8_t)(q >> 8);
            pr = get_program(st.program, &vkey, &fkey);
            if (!pr || !wait_built(&pr->state, &pr->waited_ns)) SKIP(SK_PROG, "vs box variant building\n");
            k.prog = pr;  // (its pipeline: the one bound above has the plain vertex shader)
            Pipe *bp = pipe_get(&k);
            if (!wait_built(&bp->state, &bp->waited_ns)) SKIP(SK_PIPE, "vs box pipeline building\n");
            if (ss_need(SS_PIPE, ss.pipe == bp->pipe)) { ss.pipe = bp->pipe; vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, bp->pipe); }
            float rs0[4] = {1, 1, 0, 0}, lax;
            // (labels first: with a menu open the bubble layout passed as the icon layout and its quads split apart)
            bool icon = ui_world_icons(r, indexed), label = ui_world_label(r, indexed, &lax);
            if (label) {  // (whole pixels: the 2D snap stays aligned)
                int pw_, ph_;
                pass_size(&pw_, &ph_);
                g_label_dx = roundf((1 - q / 65535.0f) * lax * pw_ * 0.5f);
            }
            else if (icon) rs0[2] = 1, rs0[3] = (float)r->x[1];  // (centres: g_vs_rows)
            else if (ui_camera_16x9()) rs0[1] = 2;  // (tilted menu plates, forecast, battle HUD: w != 1)
            else if (video || ui_edge_sprite(r, indexed)) rs0[0] = 2;  // (its screen-edge vertices are not a fill's)
            int pw_, ph_;
            pass_size(&pw_, &ph_);
            g_vs_px[0] = (float)pw_, g_vs_px[1] = (float)ph_;
            vset = stage_set(pr, pr->set_layout[0], 0, 0, pr->vi, rs0, NULL, 0, 0, 0, NULL);
            g_vs_nrows = 0, g_vs_px[0] = g_vs_px[1] = 0;
            if (rs0[2] == 0 && !label) sc_box = q / 65535.0f;  // (world icons / labels keep the wide scissor)
            composite = true;  // (full-width viewport: nothing clipped at the box edges)
        }
        if (!composite) {
            int pw, ph;
            pass_size(&pw, &ph);
            float u = fminf(pass_sx, pass_sy);
            vp_sx = vp_sy = u;
            vp_ox = (pw - st.rt.color[0]->width * u) * 0.5f;
            vp_oy = (ph - st.rt.color[0]->height * u) * 0.5f;
        }
    }
    {   // camera-attached effects sized for 16:9 are widened with the view (runtime/hle/widescreen.c)
        extern float widescreen_effect_stretch(uint64_t fs_hash);
        float k = widescreen_effect_stretch(g_cur_fs_hash);
        float vx = vp[0] * vp_sx + vp_ox + g_label_dx, vw = vp[2] * vp_sx;
        g_label_dx = 0;
        if (k > 1.0f) { vx -= vw * (k - 1) / 2; vw *= k; }
        float jx = 0, jy = 0;
        if (dlaa_jittered()) jx = g_jit[0], jy = g_jit[1];  // (DLAA: the scene a sub-pixel off each frame)
        set_viewport(vx + jx, vp[1] * vp_sy + vp_oy + jy, vw, vp[3] * vp_sy, zn, zf);
        if (g_debug_frame) {
            VkRect2D sr = scissor_rect();
            if (vkey.rt_kind[0]) LOG("[draw]   vs box x%.4f\n", (vkey.rt_kind[1] | vkey.rt_kind[2] << 8) / 65535.0);
            LOG("[draw]   vp %.2f %.2f %.2f %.2f (stretch %.3f, fs hash %016llx) scissor %d %d %u %u | guest sc %d %d %d %d | pass %ux%u s %.4f %.4f\n",
                vx, vp[1] * vp_sy + vp_oy, vw, vp[3] * vp_sy, k, (unsigned long long)g_cur_fs_hash, sr.offset.x, sr.offset.y,
                sr.extent.width, sr.extent.height, st.scissor[0][0], st.scissor[0][1], st.scissor[0][2], st.scissor[0][3], pass_w, pass_h,
                vp_sx, vp_sy);
        }
    }
    apply_scissor();
    float rs0[4] = {vp_sx, vp_sy, vp_ox, vp_oy};
    uint64_t prof_t1_ = prof ? SDL_GetTicksNS() : 0;
    VkDescriptorSet fset = stage_set(pr, pr->set_layout[1], 1, 1, pr->fi, rs0, tint, 0, 0, 0, fii);
    if (prof) prof_sets += SDL_GetTicksNS() - prof_t1_;
    VkDescriptorSet sets[2] = {vset, fset};
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pr->layout, 0, 2, sets, 0, NULL);  // (always new sets)
    if (pr->vi->uses_gmem_store) vc_mark_sbufs(0);
    if (pr->fi->uses_gmem_store) vc_mark_sbufs(1);

    uint64_t prof_t2_ = prof ? SDL_GetTicksNS() : 0;
    // ---- indices (copied; the largest index bounds the vertex data copied below)
    uint32_t first = 0, count = (uint32_t)r->x[2], inst, base_inst, max_index = 0;
    int32_t base_vertex = 0;
    VkSpan ib = {0};
    VkIndexType itype_vk = VK_INDEX_TYPE_UINT32;
    uint32_t icount = 0;
    bool use_index = indexed || !native;
    const void *scan_src = NULL;  // deferred index scan (direct index buffer)
    uint32_t scan_isz = 0, scan_restart = 0;
    if (!indexed) {
        first = (uint32_t)r->x[1]; base_inst = (uint32_t)r->x[3]; inst = (uint32_t)r->x[4] ? (uint32_t)r->x[4] : 1;
        max_index = first + count;
        if (!native) {
            uint32_t *e = expand_indices(mode, count, NULL, 0, first, &icount);
            ib = vk_transient_copy(e, (uint64_t)icount * 4 + 4, 4);
            free(e);
        }
    } else {
        uint32_t itype = (uint32_t)r->x[1];
        uint64_t iaddr = r->x[3];
        base_vertex = (int32_t)r->x[4];
        base_inst = (uint32_t)r->x[5];
        inst = (uint32_t)r->x[6] ? (uint32_t)r->x[6] : 1;
        const void *src = (const void *)(uintptr_t)iaddr;
        uint32_t restart = st.prim_restart ? st.restart_index : 0xFFFFFFFFu;
        if (native && itype != 0) {
            uint32_t isz = itype == 1 ? 2 : 4;
            itype_vk = itype == 1 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
            icount = count;
            if (iaddr % isz == 0 && guest_direct(iaddr, (uint64_t)count * isz, &ib.buf, &ib.off)) {
                scan_src = src; scan_isz = isz; scan_restart = restart;  // max_index only if a stream gets copied
            } else {
                ib = vk_transient_copy(src, (uint64_t)count * isz + 4, 4);
                if (isz == 2) { const uint16_t *s = src; for (uint32_t i = 0; i < count; i++) if (s[i] > max_index && s[i] != (uint16_t)restart) max_index = s[i]; }
                else { const uint32_t *s = src; for (uint32_t i = 0; i < count; i++) if (s[i] > max_index && s[i] != restart) max_index = s[i]; }
            }
        } else if (native) {  // 8-bit indices: widen to 16
            ib = vk_transient((uint64_t)count * 2 + 4, 4);
            uint16_t *w = ib.ptr;
            const uint8_t *s = src;
            for (uint32_t i = 0; i < count; i++) { w[i] = s[i] == 0xFF && st.prim_restart ? 0xFFFF : s[i]; if (s[i] > max_index && w[i] != 0xFFFF) max_index = s[i]; }
            itype_vk = VK_INDEX_TYPE_UINT16;
            icount = count;
        } else {
            uint32_t *e = expand_indices(mode, count, src, (int)itype, 0, &icount);
            for (uint32_t i = 0; i < icount; i++) if (e[i] > max_index) max_index = e[i];
            ib = vk_transient_copy(e, (uint64_t)icount * 4 + 4, 4);
            free(e);
        }
        if (!scan_src) {
            max_index += 1;
            if (base_vertex > 0) max_index += (uint32_t)base_vertex;
        }
    }
    if (use_index && ss_need(SS_IB, ss.ib == ib.buf && ss.io == ib.off && ss.it == itype_vk)) {
        ss.ib = ib.buf; ss.io = ib.off; ss.it = itype_vk;
        vkCmdBindIndexBuffer(cb, ib.buf, ib.off, itype_vk);
    }

    // ---- vertex streams: the range this draw can read, copied once per submission
    for (int s = 0; s < 17; s++) {
        if (!(k.used_streams & (1u << s))) continue;
        VkSpan sp;
        if (s == ZERO_BINDING) {
            static const float zero[4] = {0, 0, 0, 1};
            sp = vk_transient_copy(zero, 16, 16);
        } else {
            Range v = st.vbuf[s];
            VkBuffer dbuf;
            VkDeviceSize doff;
            if (v.addr && v.size && guest_direct(v.addr, v.size, &dbuf, &doff)) {
                if (ss_on && (ss.vb_known >> s & 1) && ss.vb[s] == dbuf && ss.vo[s] == doff) continue;
                ss.vb[s] = dbuf; ss.vo[s] = doff; ss.vb_known |= 1u << s;
                vkCmdBindVertexBuffers(cb, (uint32_t)s, 1, &dbuf, &doff);
                continue;
            }
            if (scan_src) {  // copying after all: the index scan bounds the copy
                if (scan_isz == 2) { const uint16_t *q = scan_src; for (uint32_t i = 0; i < icount; i++) if (q[i] > max_index && q[i] != (uint16_t)scan_restart) max_index = q[i]; }
                else { const uint32_t *q = scan_src; for (uint32_t i = 0; i < icount; i++) if (q[i] > max_index && q[i] != scan_restart) max_index = q[i]; }
                max_index += 1;
                if (base_vertex > 0) max_index += (uint32_t)base_vertex;
                scan_src = NULL;
            }
            uint64_t stride = k.stride[s], need;
            if (!stride) need = 256;
            else if (k.divisor[s]) need = ((uint64_t)base_inst + inst + k.divisor[s] - 1) / k.divisor[s] * stride + 256;
            else need = (uint64_t)max_index * stride + 256;
            if (v.size && need > v.size) need = v.size;
            if (!v.addr) { static const uint8_t zero[256]; sp = vk_transient_copy(zero, 256, 16); }
            else sp = guest_copy(v.addr, need ? need : 16);
        }
        VkDeviceSize off = sp.off;
        if (ss_on && (ss.vb_known >> s & 1) && ss.vb[s] == sp.buf && ss.vo[s] == off) continue;
        ss.vb[s] = sp.buf; ss.vo[s] = off; ss.vb_known |= 1u << s;
        vkCmdBindVertexBuffers(cb, (uint32_t)s, 1, &sp.buf, &off);
    }

    if (prof) prof_vtx += SDL_GetTicksNS() - prof_t2_;
    // ---- draw
    // Instanced draws with item spots among their instances (bushsanity: bushes are drawn in batches, each instance's
    // model matrix in vertex storage buffer 0, indexed by instance id): drawn as runs of instances, gold runs tinted
    // (as renderer.m; gl_InstanceIndex includes the base instance, so a run [b + k, b + k + n) draws exactly those).
    enum { MAX_RUNS = 64 };
    uint32_t run_at[MAX_RUNS], run_n[MAX_RUNS];
    bool run_gold[MAX_RUNS];
    int nruns = 0;
    if (pr->fi->uses_tint && inst > 1 && st.sbuf[0][0].addr) {
        extern bool pot_glow_active(void);
        extern bool pot_instance_gold(const float *m);
        uint64_t recs = st.sbuf[0][0].size / 64;
        if (pot_glow_active() && (uint64_t)base_inst + inst <= recs) {
            const float *m = (const float *)(uintptr_t)st.sbuf[0][0].addr;
            for (uint32_t i = 0; i < inst; i++) {
                bool g = pot_instance_gold(m + (size_t)(base_inst + i) * 16);
                if (nruns && run_gold[nruns - 1] == g) { run_n[nruns - 1]++; continue; }
                if (nruns == MAX_RUNS) { nruns = 0; break; }  // (too fragmented: draw as is)
                run_at[nruns] = i; run_n[nruns] = 1; run_gold[nruns] = g; nruns++;
            }
            if (nruns == 1 && !run_gold[0]) nruns = 0;
        }
    }
    if (!nruns) { run_at[0] = 0; run_n[0] = inst; nruns = 1; run_gold[0] = false; }
    if (g_gpuprof_frame && st.rt.ncolor == 1 && st.rt.color[0] && st.rt.color[0]->format == 0x3f && st.rt.depth) {
        static int per = -1;  // (LARECOMP_GPUPROF_DRAWS=1: each draw into the lit scene timed on its own)
        if (per < 0) per = getenv("LARECOMP_GPUPROF_DRAWS") != NULL;
        if (per) gp_mark(cb, "  draw fs %016llx %s %u x%u, ds %02x%02x%02x, stencil ref %u mask %#x, vp %.0fx%.0f", (unsigned long long)g_cur_fs_hash,
                         use_index ? "idx" : "vtx", use_index ? icount : count, inst, st.ds[0], st.ds[1], st.ds[2], st.stencil_ref[0],
                         st.stencil_vmask[0], st.viewport[0][2], st.viewport[0][3]);
    }
    for (int ri = 0; ri < nruns; ri++) {
        if (nruns > 1 || run_gold[0]) {  // this run's tint: a fragment set of its own
            float rtint[8] = {0};
            extern void pot_gold_tint(float tint[8]);
            if (run_gold[ri]) pot_gold_tint(rtint);
            VkDescriptorSet rset = stage_set(pr, pr->set_layout[1], 1, 1, pr->fi, rs0, rtint, 0, 0, 0, fii);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pr->layout, 1, 1, &rset, 0, NULL);
        }
        if (!use_index) vkCmdDraw(cb, count, run_n[ri], first, base_inst + run_at[ri]);
        else vkCmdDrawIndexed(cb, icount, run_n[ri], 0, base_vertex, base_inst + run_at[ri]);
    }
    stat_draws++;
    cum_draws++;
    if (g_gpuprof_frame && gp_n) {
        if (!gp_draws[gp_n - 1]) {  // first draw of the row: its fragment shader hash
            size_t l = strlen(gp_label[gp_n - 1]);
            snprintf(gp_label[gp_n - 1] + l, sizeof gp_label[0] - l, ", first fs %016llx", (unsigned long long)g_cur_fs_hash);
        }
        gp_draws[gp_n - 1]++;
    }
}

// (frame capture) LARECOMP_DUMP_FS=<fragment shader hash>[,...]: the data of those draws into build/dump/: the vertex
// streams, index buffer, vertex-stage uniform buffers and compiler constants as raw files, and what they are in
// d<NNN>_info.txt - to run a translated shader on the CPU with the game's own inputs.
static void dump_file(const char *path, uint64_t addr, uint64_t size) {
    if (!addr || !size) return;
    if (size > (64ull << 20)) size = 64ull << 20;
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite((const void *)(uintptr_t)addr, 1, size, f);
    fclose(f);
}
static void dump_draw(Prog *pr, CmdRec *r, bool indexed) {
    static uint64_t want[16];
    static int nwant = -1, ndump;
    if (nwant < 0) {
        nwant = 0;
        for (const char *p = getenv("LARECOMP_DUMP_FS"); p && *p && nwant < 16;) {
            want[nwant++] = strtoull(p, (char **)&p, 16);
            while (*p == ',') p++;
        }
    }
    bool hit = false;
    for (int i = 0; i < nwant; i++) hit |= want[i] == g_cur_fs_hash;
    if (!hit) return;
    mkdir("build/dump", 0755);
    int n = ndump++;
    char path[256];
    snprintf(path, sizeof path, "build/dump/d%03d_info.txt", n);
    FILE *f = fopen(path, "w");
    if (!f) return;
    NvnShaderStage *vs = &st.program->stages[0];
    fprintf(f, "fs_hash %016llx\nvs %llx\nfs %llx\nindexed %d\nmode %llu\ncount %llu\n", (unsigned long long)g_cur_fs_hash,
            (unsigned long long)vs->code_addr, (unsigned long long)st.program->stages[1].code_addr, indexed,
            (unsigned long long)r->x[0], (unsigned long long)r->x[2]);
    if (indexed) {
        uint32_t itype = (uint32_t)r->x[1], isz = itype == 0 ? 1 : itype == 1 ? 2 : 4;
        fprintf(f, "index_type %u\nbase_vertex %d\n", itype, (int32_t)r->x[4]);
        snprintf(path, sizeof path, "build/dump/d%03d_index.bin", n);
        dump_file(path, r->x[3], (uint64_t)r->x[2] * isz);
    } else {
        fprintf(f, "first %llu\n", (unsigned long long)r->x[1]);
    }
    uint32_t streams = 0;
    for (int i = 0; i < 32; i++) {
        if (!(pr->vi->attr_mask & (1u << i))) continue;
        uint32_t a = i < st.nattrib ? st.attrib[i] : 0;
        fprintf(f, "attr %d fmt 0x%x stream %u offset %u\n", i, a & 0xFF, (a >> 8) & 0x1F, a >> 13);
        if ((a & 0xFF) && ((a >> 8) & 0x1F) < 16) streams |= 1u << ((a >> 8) & 0x1F);
    }
    for (int s = 0; s < 16; s++) {
        if (!(streams & (1u << s))) continue;
        fprintf(f, "stream %d stride %u divisor %u size %llu\n", s, s < st.nstream ? st.stream[s][0] : 0, s < st.nstream ? st.stream[s][1] : 0,
                (unsigned long long)st.vbuf[s].size);
        snprintf(path, sizeof path, "build/dump/d%03d_stream%d.bin", n, s);
        dump_file(path, st.vbuf[s].addr, st.vbuf[s].size);
    }
    for (int i = 0; i < 18; i++) {
        if (!st.ubuf[0][i].addr) continue;
        fprintf(f, "cbuf %d size %llu\n", i + 3, (unsigned long long)st.ubuf[0][i].size);
        snprintf(path, sizeof path, "build/dump/d%03d_cb%d.bin", n, i + 3);
        dump_file(path, st.ubuf[0][i].addr, st.ubuf[0][i].size);
    }
    fprintf(f, "cb1 size %u\n", vs->const_size);
    snprintf(path, sizeof path, "build/dump/d%03d_cb1.bin", n);
    dump_file(path, vs->code_addr + vs->const_offset, vs->const_size);
    fclose(f);
    LOG("[dump] d%03d: fs %016llx vs %llx\n", n, (unsigned long long)g_cur_fs_hash, (unsigned long long)vs->code_addr);
}

static void debug_draw(Prog *pr, CmdRec *r, bool indexed) {
    dump_draw(pr, r, indexed);
    LOG("[draw] %s inst %llu mode %llu count %llu prog vs=%llx fs=%llx | vs cb %#x fs cb %#x | rt %d", indexed ? "e" : "a",
        (unsigned long long)(indexed ? r->x[6] : r->x[4]), (unsigned long long)r->x[0], (unsigned long long)r->x[2],
        (unsigned long long)st.program->stages[0].code_addr, (unsigned long long)st.program->stages[1].code_addr,
        pr->vi->cbuf_mask, pr->fi->cbuf_mask, st.rt.ncolor);
    for (int i = 0; i < st.rt.ncolor; i++) if (st.rt.color[i]) LOG(" %ux%u/f%#x", st.rt.color[i]->width, st.rt.color[i]->height, st.rt.color[i]->format);
    if (st.rt.depth) LOG(" d%#x %p", st.rt.depth->format, (void *)st.rt.depth);
    LOG(" | vp %.0f %.0f %.0f %.0f | attrs %#x blend %#x ds %02x%02x%02x poly %02x%02x chmask %08x\n", st.viewport[0][0], st.viewport[0][1],
        st.viewport[0][2], st.viewport[0][3], pr->vi->attr_mask, st.color_state[0], st.ds[0], st.ds[1], st.ds[2], st.poly[0], st.poly[1],
        st.channel_mask);
    if (st.poly[3] & 4)
        LOG("[draw]   depth bias: slope %g units %g clamp %g\n", st.poly_offset[0], st.poly_offset[1], st.poly_offset[2]);
    LOG("[draw]   ds %02x %02x %02x %02x %02x %02x %02x %02x | stencil ref %u/%u mask %#x/%#x vmask %#x/%#x | depth range %g %g\n",
        st.ds[0], st.ds[1], st.ds[2], st.ds[3], st.ds[4], st.ds[5], st.ds[6], st.ds[7], st.stencil_ref[0], st.stencil_ref[1],
        st.stencil_mask[0], st.stencil_mask[1], st.stencil_vmask[0], st.stencil_vmask[1], st.depth_range[0][0], st.depth_range[0][1]);
    for (int sg = 0; sg < 2; sg++) {  // the bound uniform buffers (cbuf index - 3) and their sizes
        LOG("[draw]   %s ubufs:", sg ? "fs" : "vs");
        for (int i = 0; i < 18; i++)
            if (st.ubuf[sg][i].addr) LOG(" c%d=%llx+%#llx", i + 3, (unsigned long long)st.ubuf[sg][i].addr, (unsigned long long)st.ubuf[sg][i].size);
        LOG(" | dynamic %#x\n", sg ? pr->fi->cbuf_dynamic : pr->vi->cbuf_dynamic);
    }
    if (pr->vi->attr_mask) {  // the vertex inputs: NVN format, stream, offset; the streams' strides
        LOG("[draw]   attribs:");
        for (int i = 0; i < 32; i++) {
            if (!(pr->vi->attr_mask & (1u << i))) continue;
            uint32_t a = i < st.nattrib ? st.attrib[i] : 0;
            LOG(" a%d=f%#x/s%u+%u", i, a & 0xFF, (a >> 8) & 0x1F, a >> 13);
        }
        LOG(" | strides:");
        for (int s = 0; s < st.nstream && s < 16; s++) if (st.stream[s][0]) LOG(" s%d=%u", s, st.stream[s][0]);
        LOG("\n");
    }
    if (getenv("LARECOMP_DEBUG_DATA")) {  // vertex streams 0-1 and fragment cbufs: the first floats
        for (int s = 0; s < 2; s++) {
            if (!st.vbuf[s].addr) continue;
            const float *v = (const float *)(uintptr_t)st.vbuf[s].addr;
            LOG("[draw]   vbuf%d %llx size %llu:", s, (unsigned long long)st.vbuf[s].addr, (unsigned long long)st.vbuf[s].size);
            for (int k = 0; k < 16; k++) LOG(" %.4g", v[k]);
            LOG("\n");
        }
        uint32_t sd = st.stream[0][0];
        if (!indexed && st.vbuf[0].addr && sd >= 8 && r->x[2] > 6) {  // (bigger 2D batches: bbox + the first 8 quads')
            float b[4] = {1e9f, 1e9f, -1e9f, -1e9f};
            LOG("[draw]   quads:");
            for (uint64_t k = 0; k < r->x[2] && k < 8192; k++) {
                uint64_t off = (r->x[1] + k) * sd;
                if (off + 8 > st.vbuf[0].size) break;
                const float *q = (const float *)(uintptr_t)(st.vbuf[0].addr + off);
                b[0] = fminf(b[0], q[0]), b[1] = fminf(b[1], q[1]), b[2] = fmaxf(b[2], q[0]), b[3] = fmaxf(b[3], q[1]);
                static float qb[4];
                if (k % 6 == 0) qb[0] = qb[1] = 1e9f, qb[2] = qb[3] = -1e9f;
                qb[0] = fminf(qb[0], q[0]), qb[1] = fminf(qb[1], q[1]), qb[2] = fmaxf(qb[2], q[0]), qb[3] = fmaxf(qb[3], q[1]);
                if (k % 6 == 5 && k < 48) LOG(" [%.0f %.0f %.0f %.0f]", qb[0], qb[1], qb[2], qb[3]);
            }
            LOG(" | bbox %.1f %.1f %.1f %.1f\n", b[0], b[1], b[2], b[3]);
        }
        if (!indexed && st.vbuf[0].addr && sd >= 24 && r->x[2] <= 6) {  // (small non-indexed draws: each vertex's first 6 floats)
            float bx0 = 1e9f, by0 = 1e9f, bx1 = -1e9f, by1 = -1e9f;
            for (uint64_t k = 0; k < r->x[2]; k++) {
                uint64_t off = (r->x[1] + k) * sd;
                if (off + 24 > st.vbuf[0].size) break;
                const float *q = (const float *)(uintptr_t)(st.vbuf[0].addr + off);
                LOG("[draw]   v%llu: %.4f %.4f %.4f | %.6f %.6f\n", (unsigned long long)k, q[0], q[1], q[2], q[4], q[5]);
                bx0 = fminf(bx0, q[0]); by0 = fminf(by0, q[1]); bx1 = fmaxf(bx1, q[0]); by1 = fmaxf(by1, q[1]);
            }
            const float *u = (const float *)(uintptr_t)st.ubuf[0][0].addr;  // (2D pixel quads: LARECOMP_FIND_ICONS)
            if (u && st.ubuf[0][0].size >= 16 && fabsf(u[0] * 1920 - 2) < 0.02f && bx1 > bx0) {
                extern void icon_scan_add(float x0, float y0, float x1, float y1, uint64_t vb, uint64_t vbsize);
                icon_scan_add(bx0, by0, bx1, by1, st.vbuf[0].addr, st.vbuf[0].size);
            }
        }
        if (st.ubuf[0][0].addr) {  // vertex cbuf c3 (UI sprites: their rows 33-34 too)
            const float *u = (const float *)(uintptr_t)st.ubuf[0][0].addr;
            LOG("[draw]   vcb0:");
            for (int k = 0; k < 144 && k * 4 < (int)st.ubuf[0][0].size; k++) LOG(" %.6g", u[k]);
            LOG("\n");
        }
        for (int c = 3; c < 21; c++) {  // (cbuf_mask bits are cbuf indices; ubuf[] starts at c3)
            if (!((pr->fi->cbuf_mask >> c) & 1) || !st.ubuf[1][c - 3].addr) continue;
            const float *u = (const float *)(uintptr_t)st.ubuf[1][c - 3].addr;
            LOG("[draw]   fcb%d %llx size %u:", c, (unsigned long long)st.ubuf[1][c - 3].addr, (unsigned)st.ubuf[1][c - 3].size);
            for (int k = 0; k < 128 && k * 4 < (int)st.ubuf[1][c - 3].size; k++) LOG(" %.6g", u[k]);
            LOG("\n");
        }
    }
    for (int i = 0; i < pr->fi->ntex && i < 64; i++) {  // fragment textures: guest size / format, allocation, scale
        NvnTexture *t = shader_texture_object(1, &pr->fi->tex[i]);
        const TexBackend *b = t ? (const TexBackend *)t->backend : NULL;
        if (!t) { LOG("[draw]   tex%d: none\n", i); continue; }
        LOG("[draw]   tex%d: %p %ux%u f%#x lv %u -> %ux%u scale %.4f %.4f%s\n", i, (void *)t, t->width, t->height, t->format,
            b ? b->levels : 0, b ? b->width : 0, b ? b->height : 0, b ? b->sx : 0, b ? b->sy : 0, b && b->scalable ? " scalable" : "");
        int slot = ((int)pr->fi->tex[i].handle_offset - 0x20) / 8;  // its sampler: wrap / filter / compare / border
        uint64_t h = slot >= 0 && slot < 32 ? st.tex[1][slot] : 0;
        uint32_t tsc = (uint32_t)(h >> 20) & 0xFFF;
        if (h && st.spool && tsc < (uint32_t)st.spool->count) {
            const NvnSamplerDesc *d = &((const NvnSamplerDesc *)st.spool->entries)[tsc];
            LOG("[draw]     sampler wrap %u %u %u filt %u/%u cmp %u/%u border %g %g %g %g lod %g..%g bias %g aniso %g\n", d->wrap[0],
                d->wrap[1], d->wrap[2], d->min_filter, d->mag_filter, d->compare_mode, d->compare_func, d->border[0], d->border[1],
                d->border[2], d->border[3], d->lod_min, d->lod_max, d->lod_bias, d->max_aniso);
        }
    }
    // LARECOMP_DUMP_TEX=<fs hash>[,...] and/or min:<n> (any fragment shader sampling at least n textures): with F9, save
    // the textures the draw samples as build/tex_<hash>_<i>.png (depth as _d / _s; level 0, first layer, clamped to 0..1)
    const char *dt = getenv("LARECOMP_DUMP_TEX");
    if (dt && *dt) {
        bool want = false;
        for (const char *p = dt; *p;) {
            if (!strncmp(p, "min:", 4)) {
                if (pr->fi->ntex >= atoi(p + 4)) want = true;
            } else if (strtoull(p, NULL, 16) == g_cur_fs_hash) {
                want = true;
            }
            while (*p && *p != ',') p++;
            while (*p == ',') p++;
        }
        static uint64_t done[160];
        static int ndone;
        static uint64_t done_frame = ~0ull;
        if (done_frame != renderer_present_count()) done_frame = renderer_present_count(), ndone = 0;
        for (int k = 0; k < ndone; k++) if (done[k] == g_cur_fs_hash) want = false;
        if (want && ndone < 160) {
            extern uint32_t g_save_layer;
            done[ndone++] = g_cur_fs_hash;
            void renderer_save_depth_tex(TexBackend *b, const char *prefix);
            for (int i = 0; i < pr->fi->ntex && i < 64; i++) {
                NvnTexture *t = shader_texture_object(1, &pr->fi->tex[i]);
                TexBackend *b = t ? (TexBackend *)t->backend : NULL;
                if (!b || !b->image || b->width > 8192 || b->height > 8192 || b->height < 2) continue;  // (not texel buffers)
                char path[96];
                snprintf(path, sizeof path, "build/tex_%016llx_%d", (unsigned long long)g_cur_fs_hash, i);
                if (b->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT) {
                    renderer_save_depth_tex(b, path);
                } else if (b->layers > 1 && b->depth == 1) {  // every layer (up to 64): _L<k>.png
                    for (uint32_t ly = 0; ly < b->layers && ly < 64; ly++) {
                        char p2[128];
                        snprintf(p2, sizeof p2, "%s_L%u.png", path, ly);
                        g_save_layer = ly;
                        renderer_save_texture_tex(b, p2);
                    }
                    g_save_layer = 0;
                } else {
                    strcat(path, ".png");
                    renderer_save_texture_tex(b, path);
                }
                LOG("[dumptex] %s: %ux%u f%#x layers %u depth %u target %u\n", path, t->width, t->height, t->format, b->layers,
                    b->depth, t->target);
            }
        }
    }
}

// ================================================================== texture copies
typedef struct { uint64_t tex; NvnTextureView view; uint8_t has_view; int32_t region[6]; } TexRefRec;

static void copy_region_dims(NvnTexture *t, const TexRefRec *ref, const FormatInfo **fi, int32_t rg[6], int *level,
                             uint64_t *row, uint64_t *img) {
    *fi = nvn_format_info(t->format);
    memcpy(rg, ref->region, sizeof ref->region);
    if (!rg[3]) rg[3] = (int32_t)t->width;
    if (!rg[4]) rg[4] = (int32_t)t->height;
    if (!rg[5]) rg[5] = 1;
    *level = ref->has_view && (ref->view.setmask & 1) ? ref->view.base_level : 0;
    uint32_t wb = ((uint32_t)rg[3] + (*fi)->bw - 1) / (*fi)->bw, hb = ((uint32_t)rg[4] + (*fi)->bh - 1) / (*fi)->bh;
    *row = st.copy_row_stride ? st.copy_row_stride : (uint64_t)wb * (*fi)->bpb;
    *img = st.copy_image_stride ? st.copy_image_stride : *row * hb;
}

static void copy_buffer_to_texture(uint64_t src, const TexRefRec *dst) {
    snap_reason = "buf2tex"; snapshots_invalidate(NULL);
    NvnTexture *t = (NvnTexture *)(uintptr_t)dst->tex;
    if (!t || !src) return;
    if (t->target == 10) {
        // A buffer texture is its memory: shaders read it from the heap (bfetch), not from an image. FE3H uploads its
        // per-frame light tables this way.
        const FormatInfo *bf = nvn_format_info(t->format);
        uint32_t bpb = bf && bf->bpb ? bf->bpb : 4;
        uint64_t x = dst->region[0] > 0 ? (uint64_t)dst->region[0] : 0;
        uint64_t w = dst->region[3] > 0 ? (uint64_t)dst->region[3] : t->width;
        if (x + w > t->width) w = x < t->width ? t->width - x : 0;
        static int nlog;
        if (nlog++ < 8)
            LOG("[vk] buffer -> buffer texture %llx (%u texels fmt %#x): texels %llu+%llu from %llx\n", (unsigned long long)t->addr,
                t->width, t->format, (unsigned long long)x, (unsigned long long)w, (unsigned long long)src);
        if (t->addr && w) memmove((void *)(uintptr_t)(t->addr + x * bpb), (const void *)(uintptr_t)src, w * bpb);
        return;
    }
    alias_before_write(t, 0, "upload");
    vk_end_pass();
    const FormatInfo *fi;
    int32_t rg[6];
    int level;
    uint64_t row, img;
    copy_region_dims(t, dst, &fi, rg, &level, &row, &img);
    if (!fi || !fi->bpb) return;
    TexBackend *b = vk_tex(t, false);  // prior CPU contents uploaded first
    renderer_mark_gpu_owned(t);
    b->video = true;  // (uploaded content, e.g. movie frames: not a scene buffer)
    if (b->aspect == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) return;
    if (b->astc_bw) {  // (blocks would need the CPU decode: not seen in the game yet)
        static int nlog;
        if (nlog++ < 5) LOG("[vk] buffer -> ASTC texture copy skipped (%ux%u)\n", t->width, t->height);
        return;
    }
    bool array = b->layers > 1 && b->depth == 1;
    uint32_t hb = ((uint32_t)rg[4] + fi->bh - 1) / fi->bh, wb = ((uint32_t)rg[3] + fi->bw - 1) / fi->bw;
    uint32_t out_bpb = fi->bpb == 3 ? 4 : fi->bpb;
    uint32_t nz = (uint32_t)rg[5];
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    uint64_t out_row = (uint64_t)wb * out_bpb;
    VkSpan s = vk_transient(out_row * hb * nz, 16);
    for (uint32_t z = 0; z < nz; z++)
        for (uint32_t y = 0; y < hb; y++) {
            const uint8_t *sr = (const uint8_t *)(uintptr_t)(src + (uint64_t)z * img + (uint64_t)y * row);
            uint8_t *dr = (uint8_t *)s.ptr + ((uint64_t)z * hb + y) * out_row;
            if (out_bpb == fi->bpb) memcpy(dr, sr, out_row);
            else for (uint32_t x = 0; x < wb; x++) { memcpy(dr + x * 4, sr + x * 3, 3); dr[x * 4 + 3] = 255; }
        }
    VkBufferImageCopy c = {.bufferOffset = s.off, .bufferRowLength = wb * fi->bw, .bufferImageHeight = hb * fi->bh,
                           .imageSubresource = {b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : b->aspect,
                                                (uint32_t)level, array ? (uint32_t)rg[2] : 0, array ? nz : 1},
                           .imageOffset = {rg[0], rg[1], array ? 0 : rg[2]},
                           .imageExtent = {(uint32_t)rg[3], (uint32_t)rg[4], array ? 1 : nz}};
    vkCmdCopyBufferToImage(cb, s.buf, b->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
}

static void copy_texture_to_buffer(const TexRefRec *src, uint64_t dst) {
    NvnTexture *t = (NvnTexture *)(uintptr_t)src->tex;
    if (!t || !dst || dst < HEAP_BASE) return;
    t = alias_read(t);
    vk_end_pass();
    const FormatInfo *fi;
    int32_t rg[6];
    int level;
    uint64_t row, img;
    copy_region_dims(t, src, &fi, rg, &level, &row, &img);
    if (!fi || !fi->bpb || fi->bpb == 3 || row % fi->bpb) return;
    TexBackend *b = vk_tex(t, false);
    float tsx, tsy;
    renderer_texture_scale(t, &tsx, &tsy);
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkImage srcimg = b->image;
    VmaAllocation small_mem = NULL;
    if ((tsx != 1 || tsy != 1) && level == 0 && !renderer_format_is_depth(t->format)) {
        // the CPU expects guest-sized texels: filter the scaled allocation down first
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = b->format,
                                .extent = {t->width, t->height, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &srcimg, &small_mem, NULL));
        vk_image_init_layout(cb, srcimg, VK_IMAGE_ASPECT_COLOR_BIT);
        VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffsets = {{0, 0, 0}, {(int32_t)b->width, (int32_t)b->height, 1}},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {(int32_t)t->width, (int32_t)t->height, 1}}};
        vkCmdBlitImage(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, srcimg, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_LINEAR);
        vk_full_barrier(cb);
    }
    bool array = b->layers > 1 && b->depth == 1;
    uint32_t nz = (uint32_t)rg[5];
    uint64_t total = img * nz;
    VkSpan out = vk_readback(total);
    VkBufferImageCopy c = {.bufferOffset = out.off, .bufferRowLength = (uint32_t)(row / fi->bpb) * fi->bw,
                           .bufferImageHeight = (uint32_t)(img / row) * fi->bh,
                           .imageSubresource = {b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : b->aspect,
                                                (uint32_t)level, array ? (uint32_t)rg[2] : 0, array ? nz : 1},
                           .imageOffset = {rg[0], rg[1], array ? 0 : rg[2]},
                           .imageExtent = {(uint32_t)rg[3], (uint32_t)rg[4], array ? 1 : nz}};
    vkCmdCopyImageToBuffer(cb, srcimg, VK_IMAGE_LAYOUT_GENERAL, out.buf, 1, &c);
    wait_why = "texture -> buffer readback";
    renderer_flush();  // the CPU may read the result right after the fence
    memcpy((void *)(uintptr_t)dst, out.ptr, total);
    if (small_mem) vmaDestroyImage(g_vk.vma, srcimg, small_mem);
}

static void copy_texture(const TexRefRec *src, const TexRefRec *dst, uint64_t flags) {
    snap_reason = "tex2tex"; snapshots_invalidate(NULL);
    NvnTexture *st_ = (NvnTexture *)(uintptr_t)src->tex, *dt = (NvnTexture *)(uintptr_t)dst->tex;
    if (!st_ || !dt) return;
    vk_end_pass();
    if (dl_on == 2) renderer_flush();  // (LARECOMP_VK_DEVLOST_TRACE=2: each copy its own submission, waited on)
    gp_mark(vk_cmd(), "copy texture %ux%u -> %ux%u", st_->width, st_->height, dt->width, dt->height);
    dl_note("copy texture %ux%u f%#x -> %ux%u f%#x (%p -> %p)", st_->width, st_->height, st_->format, dt->width, dt->height,
            dt->format, (void *)st_, (void *)dt);
    TexBackend *s = vk_tex(st_, false), *d = vk_tex(dt, true);
    renderer_mark_gpu_owned(dt);
    if (renderer_texture_presented(dt) && st_->width == dt->width && st_->height == dt->height) renderer_texture_mark_presented(st_);
    renderer_texture_first_write(dt);
    renderer_texture_level_fresh(dt);
    const int32_t *sr = src->region, *dr = dst->region;
    float ssx, ssy, dsx, dsy;
    renderer_texture_scale(st_, &ssx, &ssy);
    renderer_texture_scale(dt, &dsx, &dsy);
    int sx0 = (int)lroundf(sr[0] * ssx), sy0 = (int)lroundf(sr[1] * ssy);
    int dx0 = (int)lroundf(dr[0] * dsx), dy0 = (int)lroundf(dr[1] * dsy);
    int sw = sr[3] ? (int)lroundf(sr[3] * ssx) : (int)s->width, sh = sr[4] ? (int)lroundf(sr[4] * ssy) : (int)s->height;
    int dw = dr[3] ? (int)lroundf(dr[3] * dsx) : (int)d->width, dh = dr[4] ? (int)lroundf(dr[4] * dsy) : (int)d->height;
    uint32_t slv = src->has_view && (src->view.setmask & 1) ? src->view.base_level : 0;
    uint32_t dlv = dst->has_view && (dst->view.setmask & 1) ? dst->view.base_level : 0;
    uint32_t sly = src->has_view && (src->view.setmask & 2) ? src->view.min_layer : 0;
    uint32_t dly = dst->has_view && (dst->view.setmask & 2) ? dst->view.min_layer : 0;
    if (slv >= s->levels || dlv >= d->levels || sly >= s->layers || dly >= d->layers) return;
    {   // whole-image regions are the level's extent, and every region stays inside its level: scaled allocations
        // (internal resolution below 1x) round guest regions past the edge, and out-of-bounds copies / blits are
        // undefined in Vulkan (garbage that changed every frame on NVIDIA: flicker in the depth pyramid's users)
        int slw = (int)level_extent(s->width, slv), slh = (int)level_extent(s->height, slv);
        int dlw = (int)level_extent(d->width, dlv), dlh = (int)level_extent(d->height, dlv);
        if (!sr[3]) sw = slw - sx0;
        if (!sr[4]) sh = slh - sy0;
        if (!dr[3]) dw = dlw - dx0;
        if (!dr[4]) dh = dlh - dy0;
        if (sx0 + sw > slw) sw = slw - sx0;
        if (sy0 + sh > slh) sh = slh - sy0;
        if (dx0 + dw > dlw) dw = dlw - dx0;
        if (dy0 + dh > dlh) dh = dlh - dy0;
        if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return;
    }
    if (g_debug_frame || ct_on())
        LOG("[copytex] %p %ux%u f%#x lv%u (%d,%d %dx%d) [%ux%u] -> %p %ux%u f%#x lv%u (%d,%d %dx%d) [%ux%u] phys (%d,%d %dx%d) -> (%d,%d %dx%d) flags %llx\n",
            (void *)st_, st_->width, st_->height, st_->format, slv, sr[0], sr[1], sr[3], sr[4], s->width, s->height, (void *)dt, dt->width,
            dt->height, dt->format, dlv, dr[0], dr[1], dr[3], dr[4], d->width, d->height, sx0, sy0, sw, sh, dx0, dy0, dw, dh,
            (unsigned long long)flags);
    if (copy_opt() && sx0 == 0 && sy0 == 0 && dx0 == 0 && dy0 == 0 && sw == (int)s->width && sh == (int)s->height && dw == (int)d->width &&
        dh == (int)d->height && slv == 0 && dlv == 0 && sly == 0 && dly == 0 && alias_make(st_, dt, s, d)) {
        if (ct_on()) LOG("[ct] alias made %p -> %p\n", (void *)st_, (void *)dt);
        return;
    }
    alias_before_write(dt, 0, "copy dst");
    if (naliases) s = alias_read(st_)->backend;
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkImageAspectFlags sa = s->aspect, da = d->aspect;
    if (s->samples > 1 && d->samples == 1 && sw == dw && sh == dh && s->format == d->format && !(sa & VK_IMAGE_ASPECT_DEPTH_BIT)) {
        VkImageResolve rv = {.srcSubresource = {sa, slv, sly, 1}, .srcOffset = {sx0, sy0, 0}, .dstSubresource = {da, dlv, dly, 1},
                             .dstOffset = {dx0, dy0, 0}, .extent = {(uint32_t)sw, (uint32_t)sh, 1}};
        vkCmdResolveImage(cb, s->image, VK_IMAGE_LAYOUT_GENERAL, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &rv);
        return;
    }
    if (s->samples != d->samples) return;
    if (sw == dw && sh == dh && s->format == d->format) {
        VkImageCopy c = {.srcSubresource = {sa, slv, sly, 1}, .srcOffset = {sx0, sy0, 0}, .dstSubresource = {da, dlv, dly, 1},
                         .dstOffset = {dx0, dy0, 0}, .extent = {(uint32_t)sw, (uint32_t)sh, 1}};
        vkCmdCopyImage(cb, s->image, VK_IMAGE_LAYOUT_GENERAL, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
        return;
    }
    // scaled / converting copy: the game builds its depth pyramid this way (nearest), colour copies filter if asked
    bool depth = (sa | da) & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
    if (depth && s->format != d->format) return;
    VkImageBlit bl = {.srcSubresource = {depth ? (sa & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : sa) : sa, slv, sly, 1},
                      .srcOffsets = {{sx0, sy0, 0}, {sx0 + sw, sy0 + sh, 1}},
                      .dstSubresource = {depth ? (da & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : da) : da, dlv, dly, 1},
                      .dstOffsets = {{dx0, dy0, 0}, {dx0 + dw, dy0 + dh, 1}}};
    bool linear = (flags & 1) && !depth && !rt_kind(st_->format);
    dl_note("  blit %p lv%u/%u %ux%u (%d,%d %dx%d) -> %p lv%u/%u %ux%u (%d,%d %dx%d) %s", (void *)s->image, slv, s->levels, s->width,
            s->height, sx0, sy0, sw, sh, (void *)d->image, dlv, d->levels, d->width, d->height, dx0, dy0, dw, dh, linear ? "linear" : "nearest");
    vkCmdBlitImage(cb, s->image, VK_IMAGE_LAYOUT_GENERAL, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
}

// ================================================================== present helpers (device.c)
static VkPipelineLayout present_layout;
static VkDescriptorSetLayout present_set_layout;

static VkPipeline present_pipeline(VkFormat fmt) {
    static struct { VkFormat f; VkPipeline p; } cache[4];
    for (int i = 0; i < 4; i++) if (cache[i].p && cache[i].f == fmt) return cache[i].p;
    if (!present_layout) {
        VkDescriptorSetLayoutBinding b = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
        VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1, .pBindings = &b};
        VK_CHECK(vkCreateDescriptorSetLayout(g_vk.dev, &li, NULL, &present_set_layout));
        VkPushConstantRange r = {VK_SHADER_STAGE_VERTEX_BIT, 0, 32};
        VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                         .pSetLayouts = &present_set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &r};
        VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &present_layout));
    }
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vk_builtin_shader("present.vert"), .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = vk_builtin_shader("present.frag"), .pName = "main"}};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xF};
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1,
                                              .pAttachments = &cba};
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dyn};
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1,
                                        .pColorAttachmentFormats = &fmt};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri, .stageCount = 2,
                                       .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vs,
                                       .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds,
                                       .pColorBlendState = &cb, .pDynamicState = &dy, .layout = present_layout};
    VkPipeline p;
    VK_CHECK(vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &pi, NULL, &p));
    for (int i = 0; i < 4; i++) if (!cache[i].p) { cache[i].f = fmt; cache[i].p = p; break; }
    return p;
}

// Scanout shows the bytes the game wrote: sRGB images are read through a UNORM view (no decode), like metal.m.
static VkImageView raw_view(TexBackend *b) {
    VkFormat f = b->format == VK_FORMAT_R8G8B8A8_SRGB ? VK_FORMAT_R8G8B8A8_UNORM
               : b->format == VK_FORMAT_B8G8R8A8_SRGB ? VK_FORMAT_B8G8R8A8_UNORM : b->format;
    static const VkComponentMapping id = {0};
    extern uint32_t g_save_layer;
    uint32_t ly = g_save_layer < b->layers ? g_save_layer : 0;
    return vk_image_view(b, VK_IMAGE_VIEW_TYPE_2D, f, 0, 1, ly, 1, id, VK_IMAGE_ASPECT_COLOR_BIT);
}
uint32_t g_save_layer;  // (frame capture) the array layer renderer_save_texture_tex saves

// ---- post passes (Video tab "Anti-aliasing" / "Upscaler"): SMAA 1x at the game image's size, then AMD FSR 1 (EASU
// upscale to the image's size in the window, RCAS sharpen) when it is smaller than its place in the window, as
// metal.m does with MetalFX. Each pass: a fullscreen triangle into an RGBA8 image (gamma-encoded, like the input);
// up to three inputs (bindings 0-2), sampled linear / clamp.
static VkPipelineLayout post_layout;
static VkDescriptorSetLayout post_set_layout;

static VkPipeline post_pipeline(const char *fs) {
    static struct { const char *fs; VkPipeline p; } cache[16];
    for (int i = 0; i < 16; i++) if (cache[i].p && cache[i].fs == fs) return cache[i].p;
    if (!post_layout) {
        VkDescriptorSetLayoutBinding b[3];
        for (uint32_t i = 0; i < 3; i++)
            b[i] = (VkDescriptorSetLayoutBinding){i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
        VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 3, .pBindings = b};
        VK_CHECK(vkCreateDescriptorSetLayout(g_vk.dev, &li, NULL, &post_set_layout));
        VkPushConstantRange r = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 80};
        VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                         .pSetLayouts = &post_set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &r};
        VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &post_layout));
    }
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vk_builtin_shader("post.vert"), .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = vk_builtin_shader(fs), .pName = "main"}};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xF};
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1,
                                              .pAttachments = &cba};
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dyn};
    VkFormat fmt = !strcmp(fs, "mvfull.frag") ? VK_FORMAT_R16G16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1,
                                        .pColorAttachmentFormats = &fmt};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri, .stageCount = 2,
                                       .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vs,
                                       .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds,
                                       .pColorBlendState = &cb, .pDynamicState = &dy, .layout = post_layout};
    VkPipeline p;
    VK_CHECK(vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &pi, NULL, &p));
    for (int i = 0; i < 16; i++) if (!cache[i].p) { cache[i].fs = fs; cache[i].p = p; break; }
    return p;
}

// Post image `i` at w x h (RGBA8, kept until the size changes). 0, 1: SMAA edges / weights, 2: SMAA output,
// 3, 4: FSR EASU / RCAS, 5: supersampling resolve.
static TexBackend *post_image(int i, uint32_t w, uint32_t h) {
    static TexBackend img[8];  // (6: RCAS over the final image)
    TexBackend *d = &img[i];
    if (d->image && (d->width != w || d->height != h)) {
        vk_views_forget(d);
        vk_defer_image(d->image, d->mem);
        d->image = VK_NULL_HANDLE;
        d->mem = VK_NULL_HANDLE;
    }
    if (!d->image) {
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                                .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
                                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &d->image, &d->mem, NULL));
        d->format = VK_FORMAT_R8G8B8A8_UNORM; d->width = w; d->height = h; d->depth = 1; d->layers = 1; d->levels = 1;
        d->samples = 1; d->aspect = VK_IMAGE_ASPECT_COLOR_BIT; d->base_type = VK_IMAGE_VIEW_TYPE_2D;
        vk_image_init_layout(vk_cmd(), d->image, d->aspect);
    }
    return d;
}

static VkImageView post_view(TexBackend *d) {
    static const VkComponentMapping id = {0};
    return vk_image_view(d, VK_IMAGE_VIEW_TYPE_2D, d->format, 0, 1, 0, 1, id, VK_IMAGE_ASPECT_COLOR_BIT);
}

// SMAA's lookup textures (runtime/nvn/smaa_data.c): 0 = area (160 x 560 R8G8), 1 = search (64 x 16 R8), uploaded once.
static VkImageView smaa_lut(int i) {
    static TexBackend t[2];
    TexBackend *d = &t[i];
    if (!d->image) {
        extern const unsigned char *smaa_area_tex(void);
        extern const unsigned char *smaa_search_tex(void);
        uint32_t w = i ? 64 : 160, h = i ? 16 : 560, bpp = i ? 1 : 2;
        VkFormat f = i ? VK_FORMAT_R8_UNORM : VK_FORMAT_R8G8_UNORM;
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f,
                                .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &d->image, &d->mem, NULL));
        d->format = f; d->width = w; d->height = h; d->depth = 1; d->layers = 1; d->levels = 1;
        d->samples = 1; d->aspect = VK_IMAGE_ASPECT_COLOR_BIT; d->base_type = VK_IMAGE_VIEW_TYPE_2D;
        VkCommandBuffer cb = vk_cmd();
        vk_image_init_layout(cb, d->image, d->aspect);
        VkSpan s = vk_transient_copy(i ? smaa_search_tex() : smaa_area_tex(), (uint64_t)w * h * bpp, 16);
        VkBufferImageCopy c = {.bufferOffset = s.off, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {w, h, 1}};
        vkCmdCopyBufferToImage(cb, s.buf, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    }
    return post_view(d);
}

// One post pass (outside the window image's rendering): srcs (bindings 0..n-1) -> dst, fragment shader `fs`.
// clear: start from zero (SMAA's edge pass only writes edge pixels; its weights / output read them).
static void post_pass(VkCommandBuffer cb, const char *fs, const VkImageView *srcs, int n, TexBackend *dst, const float pc[8],
                      bool clear) {
    VkPipeline pipe = post_pipeline(fs);  // (before the rendering begins: created inside it, the NVIDIA driver crashed)
    vk_full_barrier(cb);
    VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = post_view(dst),
                                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {dst->width, dst->height}},
                          .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ca};
    vkCmdBeginRendering(cb, &ri);
    VkDescriptorSet ds = vk_alloc_set(post_set_layout);
    VkDescriptorImageInfo ii[3];
    VkWriteDescriptorSet w[3];
    for (int i = 0; i < n; i++) {
        ii[i] = (VkDescriptorImageInfo){vk_default_sampler(true), srcs[i], VK_IMAGE_LAYOUT_GENERAL};
        w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = (uint32_t)i,
                                      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                      .pImageInfo = &ii[i]};
    }
    vkUpdateDescriptorSets(g_vk.dev, (uint32_t)n, w, 0, NULL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, post_layout, 0, 1, &ds, 0, NULL);
    VkViewport vp = {0, 0, (float)dst->width, (float)dst->height, 0, 1};
    VkRect2D sc = {{0, 0}, {dst->width, dst->height}};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdPushConstants(cb, post_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, g_post_pc80 ? 80 : 32, g_post_pc80 ? g_post_pc80 : pc);
    g_post_pc80 = NULL;
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
}

// The view the present samples: the game image, or its anti-aliased / upscaled version. ow x oh: the whole image's
// size in the window (its crop is drawn into the viewport).
static VkImageView post_process(VkCommandBuffer cb, TexBackend *b, double ow_d, double oh_d) {
    VkImageView v = raw_view(b);
    bool smaa = g_settings.antialias == AA_SMAA || (g_settings.antialias == AA_DLAA && !dlss_available());
    uint32_t ow = (uint32_t)lround(ow_d), oh = (uint32_t)lround(oh_d);
    if (ow > 16384) ow = 16384;
    if (oh > 16384) oh = 16384;
    bool fsr = g_settings.upscaler == UPSCALE_FSR1 && (ow > b->width + 1 || oh > b->height + 1);  // (an upscale only)
    // larger than the window: average every source pixel (bilinear would skip most of them and keep the shimmer)
    bool down = !fsr && ow && oh && (ow + 1 < b->width || oh + 1 < b->height) && !getenv("LARECOMP_NO_RESOLVE");
    float boost, stops = sharpen_level(&boost);
    bool sharpen = !fsr && (g_settings.antialias != AA_DLAA || (down && g_dlaa_sharpen_late)) && stops >= 0;
    g_resolve_down = down;
    if (!smaa && !fsr && !down && !sharpen) return v;
    extern void vk_frame_suspend(void);
    extern void vk_frame_resume(void);
    vk_frame_suspend();
    if (smaa) {  // SMAA 1x: luma edges -> blending weights -> neighbourhood blending
        gp_mark(cb, "post: smaa %ux%u", b->width, b->height);
        float w = (float)b->width, h = (float)b->height;
        float m[8] = {1.0f / w, 1.0f / h, w, h};
        TexBackend *edges = post_image(0, b->width, b->height), *weights = post_image(1, b->width, b->height);
        TexBackend *out = post_image(2, b->width, b->height);
        VkImageView in[3] = {v};
        post_pass(cb, "smaa_edge.frag", in, 1, edges, m, true);
        VkImageView wi[3] = {post_view(edges), smaa_lut(0), smaa_lut(1)};
        post_pass(cb, "smaa_weight.frag", wi, 3, weights, m, true);
        VkImageView bi[3] = {v, post_view(weights)};
        post_pass(cb, "smaa_blend.frag", bi, 2, out, m, false);
        v = post_view(out);
    }
    if (fsr) {
        gp_mark(cb, "post: fsr1");
        float iw = (float)b->width, ih = (float)b->height;
        TexBackend *e = post_image(3, ow, oh), *r = post_image(4, ow, oh);
        float pe[8] = {iw / ow, ih / oh, 0.5f * iw / ow - 0.5f, 0.5f * ih / oh - 0.5f, iw, ih};
        VkImageView ei[3] = {v};
        post_pass(cb, "easu.frag", ei, 1, e, pe, false);
        float pr[8] = {exp2f(-0.2f), 0, 0, 0, (float)ow, (float)oh};  // (sharpness 0.2 stops: AMD's default)
        VkImageView ri_[3] = {post_view(e)};
        post_pass(cb, "rcas.frag", ri_, 1, r, pr, false);
        v = post_view(r);
    }
    if (down) {
        gp_mark(cb, "post: resolve -> %ux%u", ow, oh);
        float iw = (float)b->width, ih = (float)b->height;
        TexBackend *d = post_image(5, ow, oh);
        float pd[8] = {iw / ow, ih / oh, 0, 0, iw, ih};
        VkImageView di[3] = {v};
        post_pass(cb, "downscale.frag", di, 1, d, pd, false);
        v = post_view(d);
    }
    if (sharpen) {  // (SMAA / no AA: RCAS over the final image, at the window's size - as Ryujinx's FSR filter)
        uint32_t sw = down ? ow : b->width, sh = down ? oh : b->height;
        TexBackend *r = post_image(6, sw, sh);
        float pr[8] = {exp2f(-stops), 1, boost, 0, (float)sw, (float)sh};  // (no noise damping)
        static int late = -1;
        static float lb = -1;
        if (late != (int)g_dlaa_sharpen_late || lb != boost)
            LOG("[post] rcas at %ux%u (after dlaa: %d, boost %.2f)\n", sw, sh, late = g_dlaa_sharpen_late, lb = boost);
        VkImageView ri_[3] = {v};
        post_pass(cb, "rcas.frag", ri_, 1, r, pr, false);
        v = post_view(r);
    }
    gp_mark(cb, "post: end");
    vk_full_barrier(cb);
    vk_frame_resume();
    {
        static uint32_t piw, pih, pow_, poh;
        static int psm = -1, pfs = -1;
        if (b->width != piw || b->height != pih || ow != pow_ || oh != poh || smaa != psm || fsr != pfs)
            LOG("[post] %ux%u%s%s%s -> %ux%u\n", b->width, b->height, smaa ? " smaa" : "", fsr ? " fsr1" : "",
                down ? " resolve" : "", ow, oh);
        piw = b->width; pih = b->height; pow_ = ow; poh = oh; psm = smaa; pfs = fsr;
    }
    return v;
}

// Inside the window image's rendering (device.c vk_frame_begin): the game image into (vx, vy, vw, vh).
void vk_present_draw(VkCommandBuffer cb, TexBackend *b, float vx, float vy, float vw, float vh, const float pv[8]) {
    if (!b || !b->image || (b->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT)) return;
    VkImageView src = post_process(cb, b, pv[6] > 0 ? vw / pv[6] : vw, pv[7] > 0 ? vh / pv[7] : vh);
    VkPipeline p = present_pipeline(g_vk.sc_format);
    VkDescriptorSet ds = vk_alloc_set(present_set_layout);
    VkDescriptorImageInfo ii = {vk_default_sampler(true), src, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0, .descriptorCount = 1,
                              .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii};
    vkUpdateDescriptorSets(g_vk.dev, 1, &w, 0, NULL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, present_layout, 0, 1, &ds, 0, NULL);
    VkViewport vp = {vx, vy, vw, vh, 0, 1};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdPushConstants(cb, present_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 32, pv);
    vkCmdDraw(cb, 3, 1, 0, 0);
    VkViewport full = {0, 0, (float)g_vk.sc_extent.width, (float)g_vk.sc_extent.height, 0, 1};
    vkCmdSetViewport(cb, 0, 1, &full);
}

// Screenshot of a game texture (presented crop applied to build/frame_* shots), via the present shader into RGBA8.
// (frame capture) A depth / stencil target as two PNGs: <prefix>_d.png, the depth stretched over the range it holds
// (white = largest), and <prefix>_s.png, each stencil value its own grey (value * 37 mod 256; 0 black).
void renderer_save_depth_tex(TexBackend *b, const char *prefix) {
    if (!b || !b->image || !(b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) return;
    uint32_t w = b->width, h = b->height;
    bool f32 = b->format == VK_FORMAT_D32_SFLOAT || b->format == VK_FORMAT_D32_SFLOAT_S8_UINT;
    bool d16 = b->format == VK_FORMAT_D16_UNORM;
    bool has_s = (b->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) != 0;
    vk_end_pass();
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    uint64_t dbytes = (uint64_t)w * h * (d16 ? 2 : 4);
    // (one readback buffer: depth at 0, stencil after it)
    VkSpan dout = vk_readback(dbytes + (has_s ? (uint64_t)w * h : 0)), sout = dout;
    sout.off = dout.off + dbytes;
    sout.ptr = (uint8_t *)dout.ptr + dbytes;
    VkBufferImageCopy c = {.bufferOffset = dout.off, .imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1}, .imageExtent = {w, h, 1}};
    vkCmdCopyImageToBuffer(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, dout.buf, 1, &c);
    if (has_s) {
        VkBufferImageCopy cs = {.bufferOffset = sout.off, .imageSubresource = {VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0, 1}, .imageExtent = {w, h, 1}};
        vkCmdCopyImageToBuffer(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, sout.buf, 1, &cs);
    }
    renderer_flush();
    uint8_t *px = malloc((size_t)w * h * 4);
    if (!px) return;
    double lo = 1e30, hi = -1e30;
    for (uint64_t i = 0; i < (uint64_t)w * h; i++) {
        double v = f32 ? ((const float *)dout.ptr)[i] : d16 ? ((const uint16_t *)dout.ptr)[i] / 65535.0
                                                            : (((const uint32_t *)dout.ptr)[i] & 0xFFFFFF) / 16777215.0;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }
    for (uint64_t i = 0; i < (uint64_t)w * h; i++) {
        double v = f32 ? ((const float *)dout.ptr)[i] : d16 ? ((const uint16_t *)dout.ptr)[i] / 65535.0
                                                            : (((const uint32_t *)dout.ptr)[i] & 0xFFFFFF) / 16777215.0;
        uint8_t g = hi > lo ? (uint8_t)((v - lo) / (hi - lo) * 255.0 + 0.5) : 0;
        px[i * 4] = px[i * 4 + 1] = px[i * 4 + 2] = g;
        px[i * 4 + 3] = 255;
    }
    char path[300];
    snprintf(path, sizeof path, "%s_d.png", prefix);
    vk_save_png(path, px, w, h);
    LOG("[capture] %s: depth %.9g .. %.9g\n", path, lo, hi);
    if (has_s) {
        for (uint64_t i = 0; i < (uint64_t)w * h; i++) {
            uint8_t g = (uint8_t)(((const uint8_t *)sout.ptr)[i] * 37u);
            px[i * 4] = px[i * 4 + 1] = px[i * 4 + 2] = g;
        }
        snprintf(path, sizeof path, "%s_s.png", prefix);
        vk_save_png(path, px, w, h);
    }
    free(px);
}

void renderer_save_texture_tex(TexBackend *b, const char *path) {
    if (!b || !b->image || (b->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT)) return;
    extern const float *vk_shot_crop(void);
    float crop[4] = {0, 0, 1, 1};
    if (strstr(path, "frame_")) memcpy(crop, vk_shot_crop(), sizeof crop);
    uint32_t w = (uint32_t)(b->width * crop[2]), h = (uint32_t)(b->height * crop[3]);
    if (!w || !h) return;
    vk_end_pass();
    VkImage img;
    VmaAllocation mem;
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
                            .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &img, &mem, NULL));
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    vk_image_init_layout(cb, img, VK_IMAGE_ASPECT_COLOR_BIT);
    VkImageViewCreateInfo vci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                 .format = VK_FORMAT_R8G8B8A8_UNORM, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkImageView tv;
    VK_CHECK(vkCreateImageView(g_vk.dev, &vci, NULL, &tv));
    VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = tv, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {w, h}}, .layerCount = 1,
                          .colorAttachmentCount = 1, .pColorAttachments = &ca};
    vkCmdBeginRendering(cb, &ri);
    VkPipeline p = present_pipeline(VK_FORMAT_R8G8B8A8_UNORM);
    VkDescriptorSet ds = vk_alloc_set(present_set_layout);
    VkDescriptorImageInfo ii = {vk_default_sampler(true), raw_view(b), VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wr = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0, .descriptorCount = 1,
                               .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii};
    vkUpdateDescriptorSets(g_vk.dev, 1, &wr, 0, NULL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, present_layout, 0, 1, &ds, 0, NULL);
    VkViewport vp = {0, 0, (float)w, (float)h, 0, 1};
    VkRect2D sc = {{0, 0}, {w, h}};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &sc);
    float pv[8] = {renderer_origin_lower_left() && !strstr(path, "window") ? 1.0f : 0.0f, 0, 0, 0, crop[0], crop[1], crop[2], crop[3]};
    vkCmdPushConstants(cb, present_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 32, pv);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
    vk_full_barrier(cb);
    VkSpan out = vk_readback((uint64_t)w * h * 4);
    VkBufferImageCopy c = {.bufferOffset = out.off, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {w, h, 1}};
    vkCmdCopyImageToBuffer(cb, img, VK_IMAGE_LAYOUT_GENERAL, out.buf, 1, &c);
    renderer_flush();
    vk_save_png(path, out.ptr, w, h);
    vkDestroyImageView(g_vk.dev, tv, NULL);
    vmaDestroyImage(g_vk.vma, img, mem);
}

// ================================================================== executor (renderer.m exec_list)
static void exec_list(uint64_t handle, int depth) {
    uint64_t p = handle;
    for (int guard = 0; guard < 10000000; guard++) {
        CmdRec *r = (CmdRec *)(uintptr_t)p;
        void *x = r->extra;
        if (r->op < CMD_COUNT) op_counts[r->op]++;
        switch (r->op) {
        case CMD_END: return;
        case CMD_JUMP: p = r->x[0]; continue;
        case CMD_CALL: {
            const uint64_t *h = x;
            for (uint64_t i = 0; i < r->x[0] && depth < 8; i++) exec_list(h[i], depth + 1);
            break;
        }
        case CMD_BIND_PROGRAM: bind_program((NvnProgram *)(uintptr_t)r->x[0], (uint32_t)r->x[1]); break;
        case CMD_BIND_BLEND: { uint8_t *b = x; memcpy(st.blend[b[0] & 7], b, 8); break; }
        case CMD_BIND_CHANNEL_MASK: memcpy(&st.channel_mask, x, 4); break;
        case CMD_BIND_COLOR: memcpy(st.color_state, x, 4); break;
        case CMD_BIND_DEPTH_STENCIL: memcpy(st.ds, x, 8); break;
        case CMD_BIND_MULTISAMPLE: memcpy(st.ms, x, 0x18); break;
        case CMD_BIND_POLYGON: memcpy(st.poly, x, 4); break;
        case CMD_BIND_VERTEX_ATTRIB:
            st.nattrib = (int)r->x[0] > 16 ? 16 : (int)r->x[0];
            memcpy(st.attrib, x, st.nattrib * 4);
            break;
        case CMD_BIND_VERTEX_STREAM:
            st.nstream = (int)r->x[0] > 16 ? 16 : (int)r->x[0];
            memcpy(st.stream, x, st.nstream * 8);
            break;
        case CMD_BIND_VERTEX_BUFFERS:
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 16; i++) memcpy(&st.vbuf[r->x[0] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_BIND_UNIFORM_BUFFERS:
            for (uint64_t i = 0; i < r->x[2] && r->x[1] + i < 18 && r->x[0] < 6; i++)
                memcpy(&st.ubuf[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_BIND_STORAGE_BUFFERS:
            for (uint64_t i = 0; i < r->x[2] && r->x[1] + i < 16 && r->x[0] < 6; i++)
                memcpy(&st.sbuf[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_BIND_TEXTURES:
            for (uint64_t i = 0; i < r->x[2] && r->x[1] + i < 32 && r->x[0] < 6; i++)
                memcpy(&st.tex[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 8, 8);
            break;
        case CMD_BIND_IMAGES:
            for (uint64_t i = 0; i < r->x[2] && r->x[1] + i < 8 && r->x[0] < 6; i++)
                memcpy(&st.img[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 8, 8);
            break;
        case CMD_SET_TEXTURE_POOL: st.tpool = (NvnDescPool *)(uintptr_t)r->x[0]; break;
        case CMD_SET_SAMPLER_POOL: st.spool = (NvnDescPool *)(uintptr_t)r->x[0]; break;
        case CMD_SET_RENDER_TARGETS: {
            const struct { uint64_t tex[8]; NvnTextureView views[8]; uint8_t has_view[8]; uint64_t depth;
                           NvnTextureView dview; uint8_t has_dview; } *rt = x;
            RtState prev = st.rt;  // the same targets again: keep the pass (a restart = full barrier + reload)
            memset(&st.rt, 0, sizeof st.rt);
            st.rt.ncolor = (int)r->x[0] > 8 ? 8 : (int)r->x[0];
            for (int i = 0; i < st.rt.ncolor; i++) {
                st.rt.color[i] = (NvnTexture *)(uintptr_t)rt->tex[i];
                st.rt.cview[i] = rt->views[i];
                st.rt.has_cview[i] = rt->has_view[i];
            }
            st.rt.depth = (NvnTexture *)(uintptr_t)rt->depth;
            st.rt.dview = rt->dview;
            st.rt.has_dview = rt->has_dview;
            static int always = -1;
            if (always < 0) always = getenv("LARECOMP_VK_PASS_RESTART") != NULL;
            if (always || memcmp(&prev, &st.rt, sizeof prev)) st.rt_dirty = true;
            break;
        }
        case CMD_SET_VIEWPORTS:
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 16; i++) memcpy(st.viewport[r->x[0] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_SET_SCISSORS:
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 16; i++) memcpy(st.scissor[r->x[0] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_SET_DEPTH_RANGES:
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 16; i++) memcpy(st.depth_range[r->x[0] + i], (uint8_t *)x + i * 8, 8);
            break;
        case CMD_SET_BLEND_COLOR: memcpy(st.blend_color, x, 16); break;
        case CMD_SET_STENCIL: {
            int face = (int)r->x[0];
            for (int f = 0; f < 2; f++) {
                if (!(face & (1 << f))) continue;
                if (r->x[6] == 0) st.stencil_ref[f] = (uint32_t)r->x[1];
                else if (r->x[6] == 1) st.stencil_mask[f] = (uint32_t)r->x[1];
                else st.stencil_vmask[f] = (uint32_t)r->x[1];
            }
            break;
        }
        case CMD_SET_POLYGON_OFFSET: memcpy(st.poly_offset, r->f, sizeof st.poly_offset); break;
        case CMD_SET_PRIMITIVE_RESTART: st.prim_restart = r->x[0] != 0; st.restart_index = (uint32_t)r->x[1]; break;
        case CMD_SET_COPY_STRIDES: st.copy_row_stride = r->x[0]; st.copy_image_stride = r->x[1]; break;
        case CMD_CLEAR_COLOR: {
            snap_reason = "clear-color"; snapshots_invalidate(NULL);
            int index = (int)r->x[0];
            if (getenv("LARECOMP_LOG_CLEARS") && index < st.rt.ncolor && st.rt.color[index])
                LOG("[cclear] frame %llu idx %d %ux%u f%#x mask %#x\n", (unsigned long long)renderer_present_count(), index,
                    st.rt.color[index]->width, st.rt.color[index]->height, st.rt.color[index]->format, (uint32_t)r->x[2]);
            if (ct_on() && index < st.rt.ncolor && st.rt.color[index])
                LOG("[ct] clear-color %p mask %#x sc %d %d %d %d\n", (void *)st.rt.color[index], (uint32_t)r->x[2], st.scissor[0][0], st.scissor[0][1],
                    st.scissor[0][2], st.scissor[0][3]);
            if (index < st.rt.ncolor && st.rt.color[index]) alias_before_write(st.rt.color[index], 0, "clear");
            if (index < st.rt.ncolor && st.rt.color[index]) clear(index, (int)r->x[5], x, (uint32_t)r->x[2], false, 0, false, 0, 0);
            break;
        }
        case CMD_CLEAR_TEXTURE: {
            // ClearTexture(cmd, texture, view, region, color, mask): a render-target clear of the texture alone (renderer.m)
            const struct { uint64_t tex; NvnTextureView view; uint8_t has_view; int32_t region[6]; float color[4]; } *ct = x;
            NvnTexture *t = (NvnTexture *)(uintptr_t)ct->tex;
            if (!t) break;
            snap_reason = "clear-texture"; snapshots_invalidate(NULL);
            alias_before_write(t, 0, "clear");
            RtState saved = st.rt;
            int32_t sc[4];
            memcpy(sc, st.scissor[0], sizeof sc);
            vk_end_pass();
            memset(&st.rt, 0, sizeof st.rt);
            st.rt.ncolor = 1;
            st.rt.color[0] = t;
            if (ct->has_view) { st.rt.cview[0] = ct->view; st.rt.has_cview[0] = 1; }
            st.rt_dirty = true;
            const int32_t *rg = ct->region;
            if (rg[3] > 0 && rg[4] > 0) { st.scissor[0][0] = rg[0]; st.scissor[0][1] = rg[1]; st.scissor[0][2] = rg[3]; st.scissor[0][3] = rg[4]; }
            else { st.scissor[0][0] = st.scissor[0][1] = 0; st.scissor[0][2] = st.scissor[0][3] = 16384; }
            clear(0, (int)r->x[6], ct->color, (uint32_t)r->x[4] ? (uint32_t)r->x[4] : 0xF, false, 0, false, 0, 0);
            vk_end_pass();
            st.rt = saved;
            memcpy(st.scissor[0], sc, sizeof sc);
            st.rt_dirty = true;
            break;
        }
        case CMD_CLEAR_DEPTH_STENCIL:
            snap_reason = "clear-depth"; snapshots_invalidate(NULL);
            if (ct_on() && st.rt.depth)
                LOG("[ct] clear-depth %p d%d s%d sc %d %d %d %d\n", (void *)st.rt.depth, r->x[0] != 0, r->x[2] != 0, st.scissor[0][0],
                    st.scissor[0][1], st.scissor[0][2], st.scissor[0][3]);
            if (st.rt.depth && naliases) {  // (copy aliases: which aspects of the depth target this clear replaces whole)
                NvnTexture *d = st.rt.depth;
                bool whole = st.scissor[0][0] <= 0 && st.scissor[0][1] <= 0 && (int64_t)st.scissor[0][0] + st.scissor[0][2] >= (int64_t)d->width &&
                             (int64_t)st.scissor[0][1] + st.scissor[0][3] >= (int64_t)d->height &&
                             !(st.rt.has_dview && (((st.rt.dview.setmask & 1) && st.rt.dview.base_level) ||
                                                   ((st.rt.dview.setmask & 2) && st.rt.dview.min_layer)));
                for (int i = 0; i < st.rt.ncolor; i++)  // (the clear covers the smallest attachment only)
                    if (st.rt.color[i] && (st.rt.color[i]->width < d->width || st.rt.color[i]->height < d->height)) whole = false;
                VkImageAspectFlags full = 0;
                if (whole && r->x[0] != 0) full |= VK_IMAGE_ASPECT_DEPTH_BIT;
                if (whole && renderer_format_has_stencil(d->format) && (r->x[2] & 0xFF) == 0xFF) full |= VK_IMAGE_ASPECT_STENCIL_BIT;
                if (!renderer_format_has_stencil(d->format)) full |= VK_IMAGE_ASPECT_STENCIL_BIT;
                if (ct_on())
                    LOG("[ct] alias clear %p full %#x (whole %d, x0 %llu x1 %#llx x2 %#llx, dview %d/%#x, ncolor %d)\n", (void *)d, full, whole,
                        (unsigned long long)r->x[0], (unsigned long long)r->x[1], (unsigned long long)r->x[2], st.rt.has_dview,
                        st.rt.dview.setmask, st.rt.ncolor);
                alias_before_write(d, full, "clear");
            }
            if (st.rt.depth) {
                bool has_stencil = renderer_format_has_stencil(st.rt.depth->format);
                clear(-1, 0, NULL, 0, r->x[0] != 0, r->f[0], has_stencil && r->x[2] != 0, (uint32_t)r->x[1], (uint32_t)r->x[2]);
            }
            break;
        case CMD_DISPATCH: dispatch_compute((uint32_t)r->x[0], (uint32_t)r->x[1], (uint32_t)r->x[2]); break;
        case CMD_DRAW_ARRAYS: draw(r, false); break;
        case CMD_DRAW_ELEMENTS: draw(r, true); break;
        case CMD_UPDATE_UNIFORM_BUFFER:
            memcpy((void *)(uintptr_t)(r->x[0] + r->x[2]), x, r->x[3]);
            guest_copy_flush();  // (cbuf copies shared within the submission are stale now)
            break;
        case CMD_FENCE_SYNC: {
            static int fence_flush = -1;
            if (fence_flush < 0) fence_flush = getenv("LARECOMP_FENCE_FLUSH") ? 1 : 0;
            if (fence_flush) renderer_flush();
            extern void nvn_sync_set_seq(uint64_t guest, uint64_t seq);
            if (r->x[1]) nvn_sync_set_seq(r->x[1], renderer_fence_seq());
            break;
        }
        case CMD_REPORT_COUNTER: {
            // The game times its GPU work with these (two per frame) and drops to 30 fps when that looks slow: the
            // host clock now, replaced by the GPU's time at this point once the submission is done (cq_resolve).
            uint64_t *out = (uint64_t *)(uintptr_t)r->x[1];
            uint32_t type = (uint32_t)r->x[0];
            static uint32_t seen;
            if (type < 32 && !(seen & (1u << type))) { seen |= 1u << type; LOG("[nvn] ReportCounter type %u\n", type); }
            int mode = vk_counter_mode();
            static uint64_t frame_seen = ~0ull, frame_ts;
            extern uint64_t vk_frame_counter(void);
            if (mode == 1 && vk_frame_counter() != frame_seen) { frame_seen = vk_frame_counter(); frame_ts = ns_to_gpu_ticks(plat_now_ns()); }
            if (!out) break;
            out[0] = 0;
            out[1] = mode == 1 ? frame_ts : mode == 3 ? SDL_GetTicksNS() : ns_to_gpu_ticks(plat_now_ns());  // (3 legacy: ns, the old bug)
            static uint64_t *prev_out;  // (frame pacing: the pair's start)
            if (mode == 0 || mode == 2) out[1] = pace_end_counter(prev_out, out, out[1]);
            prev_out = out;
            {   // (test) LARECOMP_TEST_COUNTER_DELAY_MS=<ms>: stall the host after a frame's first counter - a slow CPU
                static double delay = -1;
                static uint64_t delayed_frame = ~0ull;
                if (delay < 0) delay = getenv("LARECOMP_TEST_COUNTER_DELAY_MS") ? atof(getenv("LARECOMP_TEST_COUNTER_DELAY_MS")) : 0;
                if (delay > 0 && vk_frame_counter() != delayed_frame) {
                    delayed_frame = vk_frame_counter();
                    SDL_DelayPrecise((Uint64)(delay * 1e6));
                }
            }
            if (mode == 2 && type == 0) {  // (type 0: timestamp)
                VkCommandBuffer cb = vk_cmd();  // (first: may begin the buffer, which resets the slot's queries)
                CounterQ *q = &cq[cur_sub];
                pthread_mutex_lock(&cq_lock);
                if (q->pool && q->n < CQ_MAX) {
                    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, q->pool, q->n);
                    q->out[q->n++] = out;
                }
                pthread_mutex_unlock(&cq_lock);
            }
            break;
        }
        case CMD_COPY_TEXTURE_TO_TEXTURE: { const TexRefRec *t = x; copy_texture(&t[0], &t[1], r->x[6]); break; }
        case CMD_DOWNSAMPLE: { const TexRefRec *t = x; copy_texture(&t[0], &t[1], 1); break; }
        case CMD_COPY_BUFFER_TO_TEXTURE: copy_buffer_to_texture(r->x[0], x); break;
        case CMD_COPY_TEXTURE_TO_BUFFER: copy_texture_to_buffer(x, r->x[3]); break;
        case CMD_CLEAR_BUFFER: {  // (buffer address, size, 32-bit value): as the copies, on the CPU once the GPU is idle
            static int nlog;
            if (nlog++ < 8) LOG("[nvn] ClearBuffer %llx size %llu value %#x\n", (unsigned long long)r->x[0], (unsigned long long)r->x[1], (uint32_t)r->x[2]);
            if (!r->x[0] || r->x[1] > (256ull << 20)) break;
            wait_why = "clear buffer";
            renderer_flush();
            uint32_t *d = (uint32_t *)(uintptr_t)r->x[0], v = (uint32_t)r->x[2];
            for (uint64_t i = 0; i < r->x[1] / 4; i++) d[i] = v;
            break;
        }
        case CMD_COPY_BUFFER_TO_BUFFER:
            wait_why = "buffer -> buffer copy";
            renderer_flush();
            memmove((void *)(uintptr_t)r->x[1], (void *)(uintptr_t)r->x[0], r->x[2]);
            break;
        default: break;
        }
        p += r->size;
    }
}

void backend_execute(uint64_t handle) {
    vk_backend_lock();
    PROF_T0();
    exec_list(handle, 0);
    PROF_ADD(prof_exec);
    if (prof) cum_exec += SDL_GetTicksNS() - prof_t0_;
    static int sync_submit = -1;
    if (sync_submit < 0) sync_submit = getenv("LARECOMP_SYNC_SUBMIT") ? 1 : 0;
    if (sync_submit) renderer_flush();
    else renderer_submit();
    vk_backend_unlock();
}
