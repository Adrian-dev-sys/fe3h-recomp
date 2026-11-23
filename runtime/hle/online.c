// Fake online (Settings > Game > Online, off by default): local stand-in data for FE3H's online features, no server.
// For now only Lost Spirits (glowing tiles on battle maps where "other players" lost units). Sites: tools/patches.py.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../hle.h"
#include "../settings.h"

#define ONLINE_OBJ_GOT 0x7001aa82e0ull  // -> the game's online request object (main+0x526000.. module)
#define REQ_TYPE 0x1c0                  // request type (-1 none, 7 = Lost Spirits grid for a map)
#define REQ_STATE 0x1c4                 // 0xf1 idle, 0xf3 error, others = steps
#define REQ_PREV 0x1c8
#define REQ_DONE 0x1cc                  // u8 finished, u8 +0x1cd failed
#define REQ_MAP 0x1da                   // u16 map id of the type-7 request
#define SPIRIT_GRIDS 0x1dc              // two int32[32][32] per-tile counts (sorted; top tiles get a spot)

static uint8_t *online_obj(void) { return (uint8_t *)(uintptr_t)*(uint64_t *)(uintptr_t)ONLINE_OBJ_GOT; }

// Online-connected bit (global +0x231ec bit 6) as the battle-map code reads it. Offline requests that never
// finished would keep battle setup pumping forever, so a stale non-idle request is reset first.
uint32_t rt_fake_online_bit(int site) {
    if (!g_settings.fake_online) return 0;
    uint8_t *o = online_obj();
    if (site == 0 && o && *(int32_t *)(o + REQ_TYPE) != 7 && *(uint32_t *)(o + REQ_STATE) != 0xf1) {
        LOG("[online] reset stale request type %d state %#x\n", *(int32_t *)(o + REQ_TYPE), *(uint32_t *)(o + REQ_STATE));
        *(int32_t *)(o + REQ_TYPE) = -1;
        *(uint32_t *)(o + REQ_STATE) = *(uint32_t *)(o + REQ_PREV) = 0xf1;
        o[REQ_DONE] = o[REQ_DONE + 1] = 0;
    }
    return 0x40;
}

// main+0x52b6f0 (start the Lost Spirits request for map x1): fill random counts and report it done at once.
int rt_fake_lost_spirits(Cpu *c) {
    if (!g_settings.fake_online) return 0;
    uint8_t *o = (uint8_t *)(uintptr_t)c->x[0];
    uint16_t map = (uint16_t)c->x[1];
    static unsigned seed;
    if (!seed) seed = (unsigned)time(NULL);
    seed = seed * 1103515245u + 12345u + map;
    unsigned r = seed;
    int32_t *g = (int32_t *)(o + SPIRIT_GRIDS);
    memset(g, 0, 2 * 32 * 32 * 4);
    for (int k = 0; k < 2; k++)
        for (int n = 0; n < 160; n++) {  // many tiles: the game drops out-of-map / blocked ones, keeps its own top N
            r = r * 1103515245u + 12345u;
            g[k * 1024 + ((r >> 8) & 1023)] += 1 + (int)((r >> 20) % 30);
        }
    *(int32_t *)(o + REQ_TYPE) = 7;
    *(uint32_t *)(o + REQ_PREV) = *(uint32_t *)(o + REQ_STATE);
    *(uint32_t *)(o + REQ_STATE) = 0x30;  // busy: battle setup then sees done + ok and places the spots
    o[REQ_DONE] = 1;
    o[REQ_DONE + 1] = 0;
    o[0x1d6] = 0;
    *(uint16_t *)(o + REQ_MAP) = map;
    LOG("[online] fake Lost Spirits for map %u\n", map);
    return 1;
}
