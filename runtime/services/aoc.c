// aoc:u (add-on content, DLC) and the DLC data the game mounts (fsp-srv OpenDataStorageByDataId).
//
// Each DLC is a folder named by its content id under <game files>/dlc, holding the DLC's extracted romfs:
//   extracted/dlc/010055D009F79001/romfs/...
// The first time a DLC is used its romfs/ folder becomes one RomFS image, romfs.bin next to it (as the start screen
// does for the base game's). The game sees each one as installed and owned: an add-on content id is the DLC base id
// (the application id + 0x1000) plus the DLC's index, and the game asks for the indices, then mounts each by id.
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../hle.h"
#include "../ipc.h"
#include "../romfs_build.h"
#include "../mods.h"

#define MAX_DLC 64

typedef struct { uint64_t id; char dir[1024]; int fd; uint64_t size; Vfs *vfs; } Dlc;  // (vfs: with mods' files)
static Dlc dlcs[MAX_DLC];
static int ndlc = -1;  // -1: not scanned yet

static int by_id(const void *a, const void *b) {
    const Dlc *x = a, *y = b;
    return x->id < y->id ? -1 : x->id > y->id;
}

// <game files>/dlc: g_romfs_dir is <game files>/romfs.
static void dlc_root(char *out, size_t n) {
    const char *r = g_romfs_dir ? g_romfs_dir : "extracted/romfs";
    size_t l = strlen(r);
    while (l && (r[l - 1] == '/' || r[l - 1] == '\\')) l--;
    size_t cut = l;
    while (cut && r[cut - 1] != '/' && r[cut - 1] != '\\') cut--;  // (drop the last component, "romfs")
    snprintf(out, n, "%.*sdlc", (int)cut, r);
}

static void dlc_scan(void) {
    if (ndlc >= 0) return;
    ndlc = 0;
    char root[1024];
    dlc_root(root, sizeof root);
    DIR *d = opendir(root);
    if (!d) { LOG("[aoc] no DLC folder (%s): no add-on content\n", root); return; }
    struct dirent *e;
    while ((e = readdir(d)) && ndlc < MAX_DLC) {
        const char *nm = e->d_name;
        if (nm[0] == '.') continue;
        char *end;
        uint64_t id = strtoull(nm, &end, 16);
        if (strlen(nm) != 16 || *end || !id) {
            LOG("[aoc] %s/%s: not a 16-digit content id, skipped\n", root, nm);
            continue;
        }
        Dlc *x = &dlcs[ndlc];
        snprintf(x->dir, sizeof x->dir, "%s/%s", root, nm);
        x->id = id;
        x->fd = -1;
        ndlc++;
    }
    closedir(d);
    qsort(dlcs, ndlc, sizeof *dlcs, by_id);
    for (int i = 0; i < ndlc; i++) {
        if (i && (dlcs[i].id & ~0xFFFull) != (dlcs[0].id & ~0xFFFull))
            LOG("[aoc] warning: %016llx belongs to another game than %016llx\n", (unsigned long long)dlcs[i].id,
                (unsigned long long)dlcs[0].id);
        LOG("[aoc] DLC %016llx (index %u)\n", (unsigned long long)dlcs[i].id, (unsigned)(dlcs[i].id & 0xFFF));
    }
}

static uint64_t base_id(void) { return ndlc > 0 ? dlcs[0].id & ~0xFFFull : 0; }

static void build_progress(uint64_t done, uint64_t total) {
    static uint64_t last;
    if (done < last || done - last >= (256ull << 20) || done == total) {
        LOG("[aoc]   %llu / %llu MB\n", (unsigned long long)(done >> 20), (unsigned long long)(total >> 20));
        last = done;
    }
}

// The DLC's RomFS image, built once from its romfs/ folder. false: not available.
static bool dlc_open(Dlc *x) {
    if (x->fd >= 0) return true;
    char bin[1100], src[1100], tmp[1110];
    snprintf(bin, sizeof bin, "%s/romfs.bin", x->dir);
    snprintf(src, sizeof src, "%s/romfs", x->dir);
    struct stat sb;
    if (stat(bin, &sb) != 0) {
        if (stat(src, &sb) != 0) { LOG("[aoc] %s: no romfs folder\n", x->dir); return false; }
        LOG("[aoc] building %s (one time)\n", bin);
        snprintf(tmp, sizeof tmp, "%s.part", bin);
        char why[256] = "";
        if (!romfs_build(src, tmp, build_progress, why, sizeof why)) {
            LOG("[aoc] %s: %s\n", bin, why);
            unlink(tmp);
            return false;
        }
        if (rename(tmp, bin) != 0) { LOG("[aoc] rename %s: %s\n", tmp, strerror(errno)); return false; }
    }
    x->fd = open(bin, O_RDONLY);
    if (x->fd < 0) { LOG("[aoc] cannot open %s: %s\n", bin, strerror(errno)); return false; }
    fstat(x->fd, &sb);
    x->size = (uint64_t)sb.st_size;
    char sub[64];
    snprintf(sub, sizeof sub, "dlc/%016llX/romfs", (unsigned long long)x->id);
    x->vfs = romfs_vfs_open(x->fd, sub);
    if (x->vfs) x->size = romfs_vfs_size(x->vfs);
    return true;
}

RT_ON_RESET(aoc_reset) {
    for (int i = 0; i < ndlc; i++)
        if (dlcs[i].fd >= 0) close(dlcs[i].fd), dlcs[i].fd = -1, romfs_vfs_free(dlcs[i].vfs), dlcs[i].vfs = NULL;
}

// ------------------------------------------------------------------ IStorage of one DLC's RomFS
static void dlc_storage(Service *s, IpcReq *req, IpcResp *resp) {
    Dlc *x = s->state;
    switch (req->cmd) {
    case 0: {  // Read(offset, size) -> B buffer
        uint64_t off = IN(uint64_t, 0), size = IN(uint64_t, 8);
        IpcBuf *b = out_buf(req, 0);
        if (!b) { resp->result = MAKE_RESULT(2, 6001); break; }
        if (size > b->size) size = b->size;
        uint8_t *dst = (uint8_t *)(uintptr_t)b->addr;
        ssize_t n = off >= x->size ? 0 : x->vfs ? romfs_vfs_read(x->vfs, dst, size, off) : pread(x->fd, dst, size, (off_t)off);
        if (n < 0) { LOG("[aoc] read %016llx at %llx: %s\n", (unsigned long long)x->id, (unsigned long long)off, strerror(errno)); n = 0; }
        if ((uint64_t)n < size) memset(dst + n, 0, size - n);
        break;
    }
    case 4: OUT(uint64_t, 0) = x->size; resp_size(resp, 8); break;  // GetSize
    default: LOG("[aoc] IStorage cmd %u (ok)\n", req->cmd);
    }
}

// fsp-srv OpenDataStorageByDataId (services/fs.c): a DLC's data, or NULL when the id is no DLC here.
Service *aoc_open_storage(uint64_t id) {
    dlc_scan();
    for (int i = 0; i < ndlc; i++)
        if (dlcs[i].id == id) {
            if (!dlc_open(&dlcs[i])) return NULL;
            LOG("[aoc] mounted %016llx (%llu MB)\n", (unsigned long long)id, (unsigned long long)(dlcs[i].size >> 20));
            return service_new("IStorage(dlc)", dlc_storage, &dlcs[i]);
        }
    return NULL;
}

// ------------------------------------------------------------------ aoc:u
static uint32_t new_event(void) {
    uint32_t w, r;
    kernel_create_event_pair(&w, &r);
    return r;
}

// IPurchaseEventManager: no purchases happen here.
static void purchase_events(Service *s, IpcReq *req, IpcResp *resp) {
    (void)s;
    switch (req->cmd) {
    case 2: resp->copy_handles[resp->ncopy++] = new_event(); break;  // GetPurchasedEventReadableHandle
    case 3: case 4: resp->result = MAKE_RESULT(164, 400); break;     // PopPurchasedProductInfo: none available
    default: break;                                                    // SetDefaultDeliveryTarget, SetDeliveryTarget
    }
}

// The indices of the DLC, from `offset`, at most `count` and what fits in the out buffer.
static void list_indices(IpcReq *req, IpcResp *resp, uint32_t offset, uint32_t count) {
    IpcBuf *b = out_buf(req, 0);
    uint32_t cap = b ? (uint32_t)(b->size / 4) : 0, n = 0;
    for (int i = (int)offset; i < ndlc && n < count && n < cap; i++)
        ((uint32_t *)(uintptr_t)b->addr)[n++] = (uint32_t)(dlcs[i].id & 0xFFF);
    OUT(uint32_t, 0) = n;
    resp_size(resp, 4);
    LOG("[aoc] ListAddOnContent(%u, %u): %u\n", offset, count, n);
}

SERVICE("aoc:u", aoc_u) {
    (void)s;
    dlc_scan();
    switch (req->cmd) {
    case 0: case 2:  // CountAddOnContent(ByApplicationId)
        OUT(uint32_t, 0) = (uint32_t)ndlc;
        resp_size(resp, 4);
        LOG("[aoc] CountAddOnContent: %d\n", ndlc);
        break;
    case 1: case 3: list_indices(req, resp, IN(uint32_t, 0), IN(uint32_t, 4)); break;  // ListAddOnContent(ByApplicationId)
    case 4: case 5:  // GetAddOnContentBaseId(ByApplicationId)
        OUT(uint64_t, 0) = base_id();
        resp_size(resp, 8);
        break;
    case 6: case 7:  // PrepareAddOnContent(ByApplicationId)(index): everything is local
        LOG("[aoc] PrepareAddOnContent %u\n", IN(uint32_t, 0));
        break;
    case 8: case 10: resp->copy_handles[resp->ncopy++] = new_event(); break;  // GetAddOnContentListChangedEvent
    case 9: break;                                                           // GetAddOnContentLostErrorCode: none
    case 50: break;                                                          // CheckAddOnContentMountStatus: ok
    case 100: case 101:  // Create(Permanent)EcPurchasedEventManager
        ipc_return_object(resp, service_new("IPurchaseEventManager", purchase_events, NULL));
        break;
    default:
        LOG("[aoc] aoc:u cmd %u (ok)\n", req->cmd);
        resp_size(resp, 0x10);
    }
}
