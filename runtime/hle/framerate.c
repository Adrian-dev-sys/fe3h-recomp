// FE3H's game logic runs a fixed 60 Hz tick: its limiter (main+0x5be430) waits until 1 / 59.94 s has passed since
// the last tick (the NVN present interval is ignored here). Higher rates come from DLSS Frame Generation instead.
#include <sched.h>

#include "../hle.h"

// One wait of the limiter's loop (tools/patches.py), instead of its YieldThread spin: a host sleep to 1 ms before the
// target (timers wake late), then yields - the loop re-checks the time after each. elapsed / target: 100 ns units.
void rt_frame_spin(uint64_t elapsed, uint64_t target) {
    uint64_t left = target > elapsed ? (target - elapsed) * 100 : 0;
    if (left > 1500000) plat_sleep_until_ns(plat_now_ns() + left - 1000000);
    else sched_yield();
}
