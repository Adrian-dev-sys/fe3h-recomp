// Six-axis sensor (gyro) and HD rumble for the SDL gamepad.
// Fusion adapted from MacNativeGames games/totk/runtime/motion_fusion.c @ b75be8cf (MIT, (c) 2026 MacNativeGames contributors).
#include <SDL3/SDL.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <string.h>

#include "../hle.h"
#include "../settings.h"

// ------------------------------------------------------------------ six-axis fusion
// nn::hid::SixAxisSensorState: G, revolutions/s, revolutions, controller axes in a z-up reference frame.
typedef struct {
    int64_t delta_ns, sampling;
    float acc[3], gyro[3], angle[3], dir[9];
    uint32_t attributes, reserved;  // bit 0 connected
} MotionState;
_Static_assert(sizeof(MotionState) == 0x60, "SixAxisSensorState ABI");

enum { MOTION_RING = 64 };
#define STALE_NS 200000000ull  // no sample for 200 ms: at rest

static pthread_mutex_t m_lock = PTHREAD_MUTEX_INITIALIZER;
static MotionState ring[MOTION_RING];
static uint64_t produced, first, last_ns;
static double R[3][3], angle[3];
static int restart = 1;
static _Atomic int six_started;

static double norm3(const double v[3]) { return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
static void cross3(const double a[3], const double b[3], double o[3]) {
    o[0] = a[1] * b[2] - a[2] * b[1];
    o[1] = a[2] * b[0] - a[0] * b[2];
    o[2] = a[0] * b[1] - a[1] * b[0];
}
static void orthonormalize(double M[3][3]) {
    double n = norm3(M[2]);
    for (int j = 0; j < 3; j++) M[2][j] /= n;
    double d = M[0][0] * M[2][0] + M[0][1] * M[2][1] + M[0][2] * M[2][2];
    for (int j = 0; j < 3; j++) M[0][j] -= d * M[2][j];
    n = norm3(M[0]);
    for (int j = 0; j < 3; j++) M[0][j] /= n;
    cross3(M[2], M[0], M[1]);
}
// Fresh orientation: z up against measured gravity.
static void align_to(const double acc[3]) {
    double n = norm3(acc), up[3] = {0, 0, 1}, x[3] = {1, 0, 0};
    if (n > 0.5)
        for (int j = 0; j < 3; j++) up[j] = -acc[j] / n;
    if (fabs(up[0]) > 0.9) { x[0] = 0; x[1] = 1; }
    memcpy(R[2], up, sizeof up);
    memcpy(R[0], x, sizeof x);
    orthonormalize(R);
}
// R = R * exp([w] dt), with a Mahony pull toward gravity while not being accelerated.
static void integrate(const double gyro[3], const double acc[3], double dt) {
    double w[3] = {gyro[0], gyro[1], gyro[2]}, n = norm3(acc);
    if (fabs(n - 1) < 0.15) {
        double meas[3] = {acc[0] / n, acc[1] / n, acc[2] / n}, est[3] = {-R[2][0], -R[2][1], -R[2][2]}, e[3];
        cross3(meas, est, e);
        for (int j = 0; j < 3; j++) w[j] += 0.5 * e[j];
    }
    double rate = norm3(w), th = rate * dt;
    if (th < 1e-12) return;
    double k[3] = {w[0] / rate, w[1] / rate, w[2] / rate}, s = sin(th), cc = 1 - cos(th);
    double K[3][3] = {{0, -k[2], k[1]}, {k[2], 0, -k[0]}, {-k[1], k[0], 0}}, E[3][3], out[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double kk = 0;
            for (int q = 0; q < 3; q++) kk += K[i][q] * K[q][j];
            E[i][j] = (i == j) + s * K[i][j] + cc * kk;
        }
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double v = 0;
            for (int q = 0; q < 3; q++) v += R[i][q] * E[q][j];
            out[i][j] = v;
        }
    memcpy(R, out, sizeof out);
    orthonormalize(R);
}

// SDL axes (x right, y out of the face, z toward the player; accel = push vs gravity, m/s^2) -> Switch (x right,
// y toward the top edge, z out of the face; accel as gravity reads, G). A rotation, so the gyro keeps its sense.
static void sdl_to_switch(const float a[3], const float g[3], double acc[3], double gyro[3]) {
    acc[0] = -a[0] / SDL_STANDARD_GRAVITY;
    acc[1] = a[2] / SDL_STANDARD_GRAVITY;
    acc[2] = -a[1] / SDL_STANDARD_GRAVITY;
    gyro[0] = g[0];
    gyro[1] = -g[2];
    gyro[2] = g[1];
}

// (UI thread) one SDL gyro sample with the newest accelerometer sample; dt from the sensor timestamps.
void motion_feed_sdl(const float a[3], const float g[3], double dt) {
    if (!g_settings.gyro) return;
    double acc[3], gyro[3];
    sdl_to_switch(a, g, acc, gyro);
    uint64_t now = SDL_GetTicksNS();
    pthread_mutex_lock(&m_lock);
    if (!(dt > 0) || dt > 0.05) dt = 0.004;  // a gap: no integration across it
    if (restart) {
        align_to(acc);
        memset(angle, 0, sizeof angle);
        restart = 0;
    } else integrate(gyro, acc, dt);
    last_ns = now;
    MotionState *s = &ring[produced % MOTION_RING];
    *s = (MotionState){.delta_ns = (int64_t)(dt * 1e9), .sampling = (int64_t)produced + 1, .attributes = 1};
    for (int i = 0; i < 3; i++) {
        double rps = gyro[i] / (2 * M_PI);
        angle[i] += rps * dt;
        s->acc[i] = (float)acc[i];
        s->gyro[i] = (float)rps;
        s->angle[i] = (float)angle[i];
        for (int j = 0; j < 3; j++) s->dir[i * 3 + j] = (float)R[j][i];
    }
    produced++;
    pthread_mutex_unlock(&m_lock);
    static _Atomic int logged;
    if (!atomic_exchange(&logged, 1)) LOG("[motion] first gyro sample fed (acc %.2f %.2f %.2f G)\n", acc[0], acc[1], acc[2]);
}

static void motion_restart(void) {
    pthread_mutex_lock(&m_lock);
    restart = 1;
    first = produced;
    last_ns = 0;
    memset(angle, 0, sizeof angle);
    pthread_mutex_unlock(&m_lock);
}

// Newest first, all max entries (a short history repeats its oldest); a controller at rest when there is no motion.
static int six_fill(MotionState *out, int max) {
    if (max <= 0 || !out) return 0;
    int n = 0;
    if (g_settings.gyro) {
        pthread_mutex_lock(&m_lock);
        uint64_t now = SDL_GetTicksNS();
        if (produced > first && (now < last_ns || now - last_ns < STALE_NS)) {
            uint64_t have = produced - first < MOTION_RING ? produced - first : MOTION_RING;
            n = (int)(have < (uint64_t)max ? have : (uint64_t)max);
            for (int i = 0; i < n; i++) out[i] = ring[(produced - 1 - (uint64_t)i) % MOTION_RING];
        }
        pthread_mutex_unlock(&m_lock);
    }
    if (!n) {
        static const MotionState rest = {.acc = {0, 0, -1}, .dir = {1, 0, 0, 0, 1, 0, 0, 0, 1}};
        for (int i = 0; i < max; i++) out[i] = rest;
    } else
        for (int i = n; i < max; i++) out[i] = out[n - 1];
    static _Atomic int live_logged;
    if (n && !atomic_exchange(&live_logged, 1)) LOG("[motion] first six-axis read with live gyro data\n");
    return max;
}

// int GetSixAxisSensorHandles(SixAxisSensorHandle *out, int count, const NpadIdType &id, NpadStyleSet style)
HLE_FUNC("_ZN2nn3hid23GetSixAxisSensorHandlesEPNS0_19SixAxisSensorHandleEiRKjNS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE", hid_six_handles) {
    uint32_t *out = (uint32_t *)(uintptr_t)c->x[0];
    int max = (int)c->x[1];
    uint32_t id = c->x[2] ? *(uint32_t *)(uintptr_t)c->x[2] : 0, style = (uint32_t)c->x[3];
    int n = out && max > 0 ? 1 : 0;
    if (n) out[0] = 3u | (id & 0xFF) << 8 | 2u << 16;  // FullKey style index, player, six-axis device 2
    static _Atomic int logged;
    if (!atomic_exchange(&logged, 1)) LOG("[motion] game asked for six-axis handles (id %#x style %#x lr %llx)\n", id, style, c->x[30]);
    c->x[0] = (uint64_t)n;
}

HLE_FUNC("_ZN2nn3hid18StartSixAxisSensorERKNS0_19SixAxisSensorHandleE", hid_six_start) {
    // (main+0x45e260 starts it before every read: only the first start resets, as on hardware)
    if (!atomic_exchange(&six_started, 1)) { LOG("[motion] game started the six-axis sensor (lr %llx)\n", c->x[30]); motion_restart(); }
}

HLE_FUNC("_ZN2nn3hid21GetSixAxisSensorStateEPNS0_18SixAxisSensorStateERKNS0_19SixAxisSensorHandleE", hid_six_state) {
    six_fill((MotionState *)(uintptr_t)c->x[0], 1);
}

HLE_FUNC("_ZN2nn3hid22GetSixAxisSensorStatesEPNS0_18SixAxisSensorStateEiRKNS0_19SixAxisSensorHandleE", hid_six_states) {
    c->x[0] = (uint64_t)six_fill((MotionState *)(uintptr_t)c->x[0], (int)c->x[1]);
}

// ------------------------------------------------------------------ HD rumble
// Switch HD rumble (two bands per side: amplitude + frequency) -> SDL's heavy (low) and light (high) motors.
typedef struct { float amp_low, freq_low, amp_high, freq_high; } VibrationValue;
static const VibrationValue vib_neutral = {0, 160, 0, 320};
#define RUMBLE_MS 250
#define RUMBLE_STALE_NS 250000000ull   // a side not updated for 250 ms is silent
#define RUMBLE_REFRESH_NS 50000000ull  // an unchanged value is resent every 50 ms
#define RUMBLE_MIN_GAP_NS 10000000ull  // a changed value at most every 10 ms

static pthread_mutex_t vib_lock = PTHREAD_MUTEX_INITIALIZER;
static VibrationValue vib_last[2] = {{0, 160, 0, 320}, {0, 160, 0, 320}};
static uint64_t vib_at[2], sent_at;
static uint16_t sent_low, sent_high;

extern bool input_rumble(uint16_t low, uint16_t high, uint32_t ms);  // runtime/hle/input.c
extern int ds5_haptics_ready(void);                                  // runtime/hle/ds5_haptics.c
extern void ds5_haptics_start(void);
extern void ds5_haptics_set(int side, float al, float fl, float ah, float fh, uint64_t now);

static float clamp01(float v) { return v > 0 ? (v < 1 ? v : 1) : 0; }  // NaN -> 0
// Share of a band at f Hz for the light motor: none at <= 160 Hz, all at >= 320 Hz, log scale between.
static float light_share(float f) { return f > 160 ? (f < 320 ? log2f(f / 160) : 1) : 0; }
// ERMs barely turn at low drive: lift weak values (curve 0.6), scale by 1.3 x the strength setting.
static uint16_t motor(float a) {
    float st = 1.3f * g_settings.rumble_strength;
    if (!(a > 0) || !(st > 0)) return 0;
    float v = powf(a > 1 ? 1 : a, 0.6f) * st * 65535.0f + 0.5f;
    return v >= 65535.0f ? 65535 : (uint16_t)v;
}

// Under vib_lock. Bands split by frequency and summed as energy per motor; the stronger side drives each motor.
static void rumble_update(uint64_t now) {
    float heavy = 0, light = 0;
    for (int s = 0; s < 2 && g_settings.rumble; s++) {
        if (!vib_at[s] || now - vib_at[s] > RUMBLE_STALE_NS) continue;
        const VibrationValue *v = &vib_last[s];
        float al = clamp01(v->amp_low), ah = clamp01(v->amp_high), sl = light_share(v->freq_low), sh = light_share(v->freq_high);
        float h = sqrtf(al * (1 - sl) * al * (1 - sl) + ah * (1 - sh) * ah * (1 - sh));
        float l = sqrtf(al * sl * al * sl + ah * sh * ah * sh);
        if (h > heavy) heavy = h;
        if (l > light) light = l;
    }
    uint16_t lo = motor(clamp01(heavy)), hi = motor(clamp01(light));
    if (ds5_haptics_ready()) lo = hi = 0;  // the DualSense's actuators play it instead
    int changed = lo != sent_low || hi != sent_high, on = lo || hi;
    if (changed ? on && now - sent_at < RUMBLE_MIN_GAP_NS : !on || now - sent_at < RUMBLE_REFRESH_NS) return;
    input_rumble(lo, hi, on ? RUMBLE_MS : 0);
    sent_low = lo, sent_high = hi, sent_at = now;
}

// Handle layout: style index | player << 8 | device << 16 (device 0 left, 1 right).
int vibration_side(uint32_t handle) { return (int)(handle >> 16 & 0xFF) & 1; }

// (any guest thread) a value the game sent: hid IPC SendVibrationValue(s), or main's own import
void vibration_send(uint32_t handle, const VibrationValue *in, const char *via) {
    VibrationValue v = *in;
    if (!isfinite(v.amp_low) || !isfinite(v.amp_high) || !isfinite(v.freq_low) || !isfinite(v.freq_high)) v = vib_neutral;
    int side = vibration_side(handle);
    static _Atomic int first, first_on;
    if (!atomic_exchange(&first, 1)) LOG("[rumble] first vibration value via %s (handle %#x)\n", via, handle);
    if ((v.amp_low > 0.01f || v.amp_high > 0.01f) && !atomic_exchange(&first_on, 1))
        LOG("[rumble] first non-zero vibration via %s: %s %.2f@%.0f Hz %.2f@%.0f Hz\n", via, side ? "right" : "left",
            (double)v.amp_low, (double)v.freq_low, (double)v.amp_high, (double)v.freq_high);
    uint64_t now = SDL_GetTicksNS();
    pthread_mutex_lock(&vib_lock);
    vib_last[side] = v;
    vib_at[side] = now;
    rumble_update(now);
    pthread_mutex_unlock(&vib_lock);
    ds5_haptics_start();
    float on = g_settings.rumble ? 1 : 0;
    ds5_haptics_set(side, v.amp_low * on, v.freq_low, v.amp_high * on, v.freq_high, now);
}

void vibration_actual(uint32_t handle, VibrationValue *out) {
    pthread_mutex_lock(&vib_lock);
    *out = vib_last[vibration_side(handle)];
    pthread_mutex_unlock(&vib_lock);
}

HLE_FUNC("_ZN2nn3hid18SendVibrationValueERKNS0_21VibrationDeviceHandleERKNS0_14VibrationValueE", hid_send_vibration) {
    const uint32_t *h = (const uint32_t *)(uintptr_t)c->x[0];
    const VibrationValue *v = (const VibrationValue *)(uintptr_t)c->x[1];
    if (h && v) vibration_send(*h, v, "import");
}

RT_ON_RESET(motion_reset) {
    pthread_mutex_lock(&vib_lock);
    memset(vib_at, 0, sizeof vib_at);
    vib_last[0] = vib_last[1] = vib_neutral;
    sent_low = sent_high = 0;
    sent_at = 0;
    pthread_mutex_unlock(&vib_lock);
    input_rumble(0, 0, 0);
    six_started = 0;
    motion_restart();
}
