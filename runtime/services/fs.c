// fsp-srv: RomFS as a raw storage image, save data and SD card backed by host directories.
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef _WIN32
#include <windows.h>
#undef IN   // (windows.h's empty parameter markers; ipc.h defines its own)
#undef OUT
#endif

#include "../hle.h"
#include "../rando.h"
#include "../mods.h"
#include "../ipc.h"

#define FS_PATH_NOT_FOUND MAKE_RESULT(2, 1)
#define FS_PATH_EXISTS MAKE_RESULT(2, 2)
#define FS_TARGET_LOCKED MAKE_RESULT(2, 7)
#define FS_NOT_EMPTY MAKE_RESULT(2, 8)

const char *g_save_dir = "save";
// ------------------------------------------------------------------ RomFS images with mod files (runtime/mods.h)
// The game reads RomFS images raw and parses their tables itself. Replaced and new files are appended past the end
// of the image; copies of the tables (and the header) are patched to point at them, and reads there go to host files.
typedef struct { uint64_t start, size; char *path; int fd; const uint8_t *mem; } Overlay;
typedef struct { uint64_t off, size; uint8_t *data; } Patch;  // in-image patches: header, tables
typedef struct { uint64_t off, size; char *path; uint32_t meta; } RomFile;
struct Vfs {
    int fd;
    Patch patches[8];
    int npatches;
    Overlay *ov;
    int ov_n, ov_cap;
    uint8_t *ov_meta;  // the file table with replacements
    uint64_t ov_end, size, data_off;
    RomFile *files;  // (sorted by offset: which game file is being streamed)
    int nfiles;
};
static Vfs *g_rom;
static int romfs_fd = -1;

void romfs_vfs_free(Vfs *v) {
    if (!v) return;
    for (int i = 0; i < v->ov_n; i++) {
        if (v->ov[i].fd >= 0) close(v->ov[i].fd);
        free(v->ov[i].path), free((void *)v->ov[i].mem);  // (mem: the relocated file / directory tables)
    }
    for (int i = 0; i < v->npatches; i++) free(v->patches[i].data);
    for (int i = 0; i < v->nfiles; i++) free(v->files[i].path);
    free(v->ov), free(v->files), free(v);
}

// Reboot: the next boot builds its own index / overlay (the mod choice may have changed).
RT_ON_RESET(fs_reset) {
    romfs_vfs_free(g_rom);
    g_rom = NULL;
    if (romfs_fd >= 0) close(romfs_fd);
    romfs_fd = -1;
    g_save_dir = "save";
}

static int rom_cmp(const void *a, const void *b) {
    const RomFile *x = a, *y = b;
    return x->off < y->off ? -1 : x->off > y->off;
}

static void rom_dir_path(const uint8_t *dmeta, uint32_t dsize, uint32_t dir, char *out, size_t cap) {
    char parts[16][256];
    int n = 0;
    while (dir != 0 && dir < dsize && n < 16) {  // root (offset 0) has an empty name
        uint32_t parent, namelen;
        memcpy(&parent, dmeta + dir, 4);
        memcpy(&namelen, dmeta + dir + 0x14, 4);
        if (namelen > 255) namelen = 255;
        memcpy(parts[n], dmeta + dir + 0x18, namelen);
        parts[n][namelen] = 0;
        n++;
        dir = parent;
    }
    out[0] = 0;
    for (int i = n - 1; i >= 0; i--) { strlcat(out, parts[i], cap); strlcat(out, "/", cap); }
}

static uint32_t romfs_hash(uint32_t parent, const char *name, uint32_t len) {
    uint32_t hash = parent ^ 123456789;
    for (uint32_t i = 0; i < len; i++) { hash = (hash >> 5) | (hash << 27); hash ^= (uint8_t)name[i]; }
    return hash;
}

typedef struct { char rel[512]; char host[1024]; uint64_t size; int order; } ModFile;

static int mod_cmp(const void *a, const void *b) {
    const ModFile *x = a, *y = b;
    int c = strcmp(x->rel, y->rel);
    return c ? c : x->order - y->order;
}

static void walk_mod(const char *root, const char *rel, int order, ModFile **out, int *n, int *cap) {
    char dirp[1200];
    snprintf(dirp, sizeof dirp, "%s/%s", root, rel);
    DIR *d = opendir(dirp);
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        char r2[512], h2[1200];
        snprintf(r2, sizeof r2, "%s%s%s", rel, rel[0] ? "/" : "", e->d_name);
        snprintf(h2, sizeof h2, "%s/%s", root, r2);
        struct stat sb;
        if (stat(h2, &sb)) continue;
        if (S_ISDIR(sb.st_mode)) { walk_mod(root, r2, order, out, n, cap); continue; }
        if (*n == *cap) { *cap = *cap ? *cap * 2 : 64; *out = realloc(*out, sizeof **out * *cap); }
        ModFile *m = &(*out)[(*n)++];
        strlcpy(m->rel, r2, sizeof m->rel);
        strlcpy(m->host, h2, sizeof m->host);
        m->size = (uint64_t)sb.st_size, m->order = order;
    }
    closedir(d);
}

static int str_cmp(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

// All mod files for one image (sub: "romfs" or "dlc/<id>/romfs"), one per path: the highest-priority copy.
static ModFile *mod_files(const char *sub, int *count) {
    ModFile *mf = NULL;
    int n = 0, cap = 0, order = 0;
    if (!strcmp(sub, "romfs") && rando_seed_dir()) {
        char root[1100];
        snprintf(root, sizeof root, "%s/romfs", rando_seed_dir());
        walk_mod(root, "", order++, &mf, &n, &cap);
    }
    char roots[128][600];
    int nr = mods_roots(sub, roots, 128);
    for (int i = 0; i < nr; i++) {
        int before = n;
        walk_mod(roots[i], "", order++, &mf, &n, &cap);
        LOG("[mods] %s: %d files\n", roots[i], n - before);
    }
    if (n) qsort(mf, n, sizeof *mf, mod_cmp);
    int k = 0;
    for (int i = 0; i < n; i++) {  // keep the last (highest order) copy of each path
        if (k && !strcmp(mf[k - 1].rel, mf[i].rel)) {
            LOG("[mods] %s: %s wins over %s\n", mf[i].rel, mf[i].host, mf[k - 1].host);
            mf[k - 1] = mf[i];
        } else mf[k++] = mf[i];
    }
    *count = k;
    return mf;
}

// directory entry offset for "a/b/c" (walk from the root, offset 0); missing ones are created when dh != NULL
static int64_t dir_find(uint8_t **dm, uint64_t *dsz, uint8_t *dh, uint32_t nb, const char *path) {
    uint32_t cur = 0;
    char buf[512];
    strlcpy(buf, path, sizeof buf);
    for (char *tok = strtok(buf, "/"); tok; tok = strtok(NULL, "/")) {
        uint32_t child, nl = (uint32_t)strlen(tok);
        memcpy(&child, *dm + cur + 8, 4);
        while (child != 0xFFFFFFFFu && child + 0x18 <= *dsz) {
            uint32_t l;
            memcpy(&l, *dm + child + 0x14, 4);
            if (l == nl && !memcmp(*dm + child + 0x18, tok, nl)) break;
            memcpy(&child, *dm + child + 4, 4);
        }
        if (child == 0xFFFFFFFFu || child + 0x18 > *dsz) {
            if (!dh) return -1;
            // new entry: parent, sibling (the old first child), no children / files, hash chain, name
            uint32_t at = (uint32_t)*dsz, none = 0xFFFFFFFFu, bucket = romfs_hash(cur, tok, nl) % nb, prev;
            *dsz += 0x18 + ((nl + 3) & ~3u);
            *dm = realloc(*dm, *dsz);
            memset(*dm + at, 0, *dsz - at);
            memcpy(&prev, dh + 4 * bucket, 4);
            memcpy(*dm + at, &cur, 4);
            memcpy(*dm + at + 4, *dm + cur + 8, 4);
            memcpy(*dm + at + 8, &none, 4);
            memcpy(*dm + at + 0xC, &none, 4);
            memcpy(*dm + at + 0x10, &prev, 4);
            memcpy(*dm + at + 0x14, &nl, 4);
            memcpy(*dm + at + 0x18, tok, nl);
            memcpy(*dm + cur + 8, &at, 4);
            memcpy(dh + 4 * bucket, &at, 4);
            child = at;
        }
        cur = child;
    }
    return cur;
}

static void ov_push(Vfs *v, Overlay o) {
    if (v->ov_n == v->ov_cap) { v->ov_cap = v->ov_cap ? v->ov_cap * 2 : 256; v->ov = realloc(v->ov, sizeof *v->ov * v->ov_cap); }
    v->ov[v->ov_n++] = o;
}

// Files the image doesn't have: copies of the file and directory tables with the new entries (linked into their
// directory's lists and the hash buckets) go past the end; the header and hash tables are patched to match.
static void add_new_files(Vfs *v, const uint64_t *h, const uint8_t *dmeta, const uint8_t *fmeta, const ModFile *nf, int n) {
    if (!n) return;
    int fd = v->fd;
    uint64_t dsz0 = h[4], dsz = dsz0, fsz = h[8], dhsz = h[2], fhsz = h[6];
    uint8_t *dcopy = malloc(dsz), *dh = malloc(dhsz), *fh = malloc(fhsz), *head = malloc(0x50);
    memcpy(dcopy, dmeta, dsz);
    if (pread(fd, dh, dhsz, (off_t)h[1]) != (ssize_t)dhsz || pread(fd, fh, fhsz, (off_t)h[5]) != (ssize_t)fhsz ||
        pread(fd, head, 0x50, 0) != 0x50) {
        LOG("[mods] cannot read the RomFS tables\n");
        free(dcopy), free(dh), free(fh), free(head);
        return;
    }
    uint64_t extra = 0;  // pass 1: directories, and the file table's new size
    for (int i = 0; i < n; i++) {
        char dirp[512];
        strlcpy(dirp, nf[i].rel, sizeof dirp);
        char *slash = strrchr(dirp, '/');
        const char *name = slash ? slash + 1 : nf[i].rel;
        extra += 0x20 + ((strlen(name) + 3) & ~3u);
        if (slash) *slash = 0; else dirp[0] = 0;
        dir_find(&dcopy, &dsz, dh, (uint32_t)(dhsz / 4), dirp);
    }
    uint64_t nsz = fsz + extra;
    uint8_t *meta = calloc(1, nsz);
    memcpy(meta, v->ov_meta ? v->ov_meta : fmeta, fsz);  // (the replacement-patched table if there is one)
    // past the end: file table, directory table (if it grew), then the new files' data
    uint64_t meta_at = v->ov_end, dmeta_at = (meta_at + nsz + 0xF) & ~0xFull;
    uint64_t data_at = dsz > dsz0 ? (dmeta_at + dsz + 0xF) & ~0xFull : dmeta_at;
    ov_push(v, (Overlay){meta_at, nsz, NULL, -1, meta});
    if (dsz > dsz0) ov_push(v, (Overlay){dmeta_at, dsz, NULL, -1, dcopy});
    v->files = realloc(v->files, sizeof(RomFile) * (v->nfiles + n));
    uint32_t nb = (uint32_t)(fhsz / 4);
    uint64_t o = fsz;
    for (int i = 0; i < n; i++) {
        char dirp[512];
        strlcpy(dirp, nf[i].rel, sizeof dirp);
        char *slash = strrchr(dirp, '/');
        const char *name = slash ? slash + 1 : nf[i].rel;
        if (slash) *slash = 0; else dirp[0] = 0;
        int64_t dir = dir_find(&dcopy, &dsz, NULL, 0, dirp);
        if (dir < 0) continue;
        uint32_t nl = (uint32_t)strlen(name), d32 = (uint32_t)dir, entry = (uint32_t)o, child, prev;
        uint32_t bucket = romfs_hash(d32, name, nl) % nb;
        uint64_t rel = data_at - v->data_off, size = nf[i].size;
        memcpy(&child, dcopy + dir + 0xC, 4);  // dir's first file -> becomes our sibling
        memcpy(&prev, fh + 4 * bucket, 4);
        memcpy(meta + o, &d32, 4);
        memcpy(meta + o + 4, &child, 4);
        memcpy(meta + o + 8, &rel, 8);
        memcpy(meta + o + 0x10, &size, 8);
        memcpy(meta + o + 0x18, &prev, 4);
        memcpy(meta + o + 0x1C, &nl, 4);
        memcpy(meta + o + 0x20, name, nl);
        memcpy(dcopy + dir + 0xC, &entry, 4);
        memcpy(fh + 4 * bucket, &entry, 4);
        o += 0x20 + ((nl + 3) & ~3u);
        ov_push(v, (Overlay){data_at, size, strdup(nf[i].host), -1, NULL});
        v->files[v->nfiles++] = (RomFile){data_at, size, strdup(nf[i].rel), entry};
        data_at = (data_at + size + 0xF) & ~0xFull;
    }
    v->ov_end = data_at;
    uint64_t fmeta_size = o;
    memcpy(head + 7 * 8, &meta_at, 8);
    memcpy(head + 8 * 8, &fmeta_size, 8);
    if (dsz > dsz0) {
        memcpy(head + 3 * 8, &dmeta_at, 8);
        memcpy(head + 4 * 8, &dsz, 8);
    } else v->patches[v->npatches++] = (Patch){h[3], dsz, dcopy};
    v->patches[v->npatches++] = (Patch){0, 0x50, head};
    v->patches[v->npatches++] = (Patch){h[1], dhsz, dh};
    v->patches[v->npatches++] = (Patch){h[5], fhsz, fh};
    LOG("[mods] %d new files (%llu new directory bytes)\n", n, (unsigned long long)(dsz - dsz0));
}

// The image in fd with the mods' files for `sub` ("romfs", "dlc/<id>/romfs"). NULL: the image's header is unreadable.
Vfs *romfs_vfs_open(int fd, const char *sub) {
    uint64_t h[10];
    if (pread(fd, h, sizeof h, 0) != (ssize_t)sizeof h) return NULL;
    // header: size, dir hash (off,size), dir meta (off,size), file hash (off,size), file meta (off,size), data off
    uint64_t doff = h[3], dsz = h[4], foff = h[7], fsz = h[8];
    Vfs *v = calloc(1, sizeof *v);
    v->fd = fd;
    v->data_off = h[9];
    struct stat rs;
    fstat(fd, &rs);
    v->size = (uint64_t)rs.st_size;
    v->ov_end = (v->size + 0xF) & ~0xFull;
    uint8_t *dmeta = malloc(dsz), *fmeta = malloc(fsz);
    if (pread(fd, dmeta, dsz, (off_t)doff) != (ssize_t)dsz || pread(fd, fmeta, fsz, (off_t)foff) != (ssize_t)fsz) {
        free(dmeta), free(fmeta);
        return v;  // (plain reads)
    }
    int cap = 4096;
    v->files = malloc(sizeof(RomFile) * cap);
    for (uint64_t o = 0; o + 0x20 <= fsz;) {
        uint32_t parent, namelen;
        uint64_t data, size;
        memcpy(&parent, fmeta + o, 4);
        memcpy(&data, fmeta + o + 8, 8);
        memcpy(&size, fmeta + o + 0x10, 8);
        memcpy(&namelen, fmeta + o + 0x1C, 4);
        char dir[1024], name[256];
        rom_dir_path(dmeta, (uint32_t)dsz, parent, dir, sizeof dir);
        uint32_t nl = namelen > 255 ? 255 : namelen;
        memcpy(name, fmeta + o + 0x20, nl);
        name[nl] = 0;
        if (v->nfiles == cap) { cap *= 2; v->files = realloc(v->files, sizeof(RomFile) * cap); }
        char *path = malloc(strlen(dir) + nl + 1);
        strcpy(path, dir);
        strcat(path, name);
        v->files[v->nfiles++] = (RomFile){v->data_off + data, size, path, (uint32_t)o};
        o += 0x20 + ((namelen + 3) & ~3u);
    }
    int nmod = 0, nnew = 0, nrep = 0;
    ModFile *mf = mod_files(sub, &nmod), *nf = malloc(sizeof *nf * (nmod ? nmod : 1));
    char **known = malloc(sizeof *known * (v->nfiles ? v->nfiles : 1));  // (sorted path pointers)
    for (int i = 0; i < v->nfiles; i++) known[i] = v->files[i].path;
    qsort(known, v->nfiles, sizeof *known, str_cmp);
    for (int i = 0; i < nmod; i++) {
        const char *key = mf[i].rel;
        char **k = bsearch(&key, known, v->nfiles, sizeof *known, str_cmp);
        int hit = -1;
        if (k) for (int j = 0; j < v->nfiles; j++) if (v->files[j].path == *k) { hit = j; break; }
        if (hit < 0) { nf[nnew++] = mf[i]; continue; }
        if (!v->ov_meta) {
            v->ov_meta = malloc(fsz);
            memcpy(v->ov_meta, fmeta, fsz);
        }
        uint64_t rel = v->ov_end - v->data_off, size = mf[i].size;
        memcpy(v->ov_meta + v->files[hit].meta + 8, &rel, 8);
        memcpy(v->ov_meta + v->files[hit].meta + 0x10, &size, 8);
        ov_push(v, (Overlay){v->ov_end, size, strdup(mf[i].host), -1, NULL});
        v->ov_end = (v->ov_end + size + 0xF) & ~0xFull;
        nrep++;
    }
    free(known);
    if (nrep) LOG("[mods] %s: %d files replaced\n", sub, nrep);
    if (v->ov_meta) v->patches[v->npatches++] = (Patch){foff, fsz, v->ov_meta};
    add_new_files(v, h, dmeta, fmeta, nf, nnew);
    free(nf), free(mf);
    qsort(v->files, v->nfiles, sizeof *v->files, rom_cmp);
    free(dmeta), free(fmeta);
    return v;
}

uint64_t romfs_vfs_size(const Vfs *v) { return v->ov_n ? v->ov_end : v->size; }

// Image read: original bytes, with the patched tables and the appended files.
ssize_t romfs_vfs_read(Vfs *v, uint8_t *dst, uint64_t size, uint64_t off) {
    if (!v->ov_n) return pread(v->fd, dst, size, (off_t)off);
    Overlay *ov = v->ov;
    uint64_t done = 0;
    while (done < size) {
        uint64_t o = off + done, n = size - done;
        if (o >= ov[0].start) {  // appended region
            int i = 0;
            while (i < v->ov_n && !(o >= ov[i].start && o < ov[i].start + ((ov[i].size + 0xF) & ~0xFull))) i++;
            if (i == v->ov_n) { memset(dst + done, 0, n); done += n; break; }
            uint64_t in = o - ov[i].start, avail = ((ov[i].size + 0xF) & ~0xFull) - in;
            if (n > avail) n = avail;
            ssize_t r;
            if (ov[i].mem) {  // in-memory blob (a relocated table)
                r = in < ov[i].size ? (ssize_t)(n < ov[i].size - in ? n : ov[i].size - in) : 0;
                memcpy(dst + done, ov[i].mem + in, (size_t)r);
            } else {
                if (ov[i].fd < 0) ov[i].fd = open(ov[i].path, O_RDONLY);
                r = in < ov[i].size ? pread(ov[i].fd, dst + done, n < ov[i].size - in ? n : ov[i].size - in, (off_t)in) : 0;
            }
            if (r < 0) r = 0;
            if ((uint64_t)r < n) memset(dst + done + r, 0, n - r);
        } else {
            if (o + n > ov[0].start) n = ov[0].start - o;
            ssize_t r = pread(v->fd, dst + done, n, (off_t)o);
            if (r < 0) r = 0;
            if ((uint64_t)r < n) memset(dst + done + r, 0, n - r);
            for (int k = 0; k < v->npatches; k++) {  // the patched tables
                const Patch *p = &v->patches[k];
                uint64_t a = o > p->off ? o : p->off, b = o + n < p->off + p->size ? o + n : p->off + p->size;
                if (a < b) memcpy(dst + done + (a - o), p->data + (a - p->off), b - a);
            }
        }
        done += n;
    }
    return (ssize_t)done;
}

// Path of the game RomFS file containing image offset `off` (NULL if none).
const char *romfs_file_at(uint64_t off) {
    if (!g_rom) return NULL;
    int lo = 0, hi = g_rom->nfiles - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (off < g_rom->files[mid].off) hi = mid - 1;
        else if (off >= g_rom->files[mid].off + g_rom->files[mid].size) lo = mid + 1;
        else return g_rom->files[mid].path;
    }
    return NULL;
}

void romfs_file_read_hook(const char *path);

// ------------------------------------------------------------------ IStorage (raw RomFS image)
static void storage(Service *s, IpcReq *req, IpcResp *resp) {
    Vfs *v = s->state;
    switch (req->cmd) {
    case 0: {  // Read(offset, size) -> B buffer
        uint64_t off = IN(uint64_t, 0), size = IN(uint64_t, 8);
        IpcBuf *b = out_buf(req, 0);
        if (!b) { resp->result = MAKE_RESULT(2, 6001); break; }
        if (size > b->size) size = b->size;
        ssize_t n = romfs_vfs_read(v, (uint8_t *)(uintptr_t)b->addr, size, off);
        if (n < 0) LOG("[fs] storage read failed at %llx: %s\n", off, strerror(errno));
        {
            const char *path = romfs_file_at(off);
            static const char *last;
            if (path && path != last) {
                last = path;
                if (getenv("LARECOMP_LOG_FILES")) LOG("[fs] read %s\n", path);
                romfs_file_read_hook(path);
            }
        }
        static uint64_t reads, bytes;
        reads++;
        bytes += size;
        if ((reads & (reads - 1)) == 0) LOG("[fs] romfs reads: %llu (%llu KB)\n", reads, bytes >> 10);
        break;
    }
    case 4: OUT(uint64_t, 0) = romfs_vfs_size(v); resp_size(resp, 8); break;  // GetSize
    default: LOG("[fs] IStorage cmd %u (ok)\n", req->cmd);
    }
}

static Service *open_romfs_storage(void) {
    if (!g_rom) {
        char path[1024];
        snprintf(path, sizeof path, "%s.bin", g_romfs_dir);  // extracted/romfs.bin
        romfs_fd = open(path, O_RDONLY);
        if (romfs_fd < 0) FATAL("cannot open %s (set up the game files on the start screen)\n", path);
        g_rom = romfs_vfs_open(romfs_fd, "romfs");
        if (!g_rom) FATAL("cannot read %s\n", path);
        LOG("[fs] romfs index: %d files\n", g_rom->nfiles);
    }
    return service_new("IStorage(romfs)", storage, g_rom);
}


// ------------------------------------------------------------------ host-directory filesystem
// Save data is all-or-nothing, as on the Switch (its save storage commits a mount's changes at once): a file opened
// for writing on a save mount works on a copy, <file>.pending, which the mount's Commit syncs and renames over the
// file - so a crash or power loss mid-save leaves the last committed save, never a half-written one (the save
// backups PR #55 was closed in favour of this). Reads see the pending copy, as the Switch's do before a commit. A
// copy left by a crash is dropped at the next mount (uncommitted, as on the Switch). The SD card mount writes in place.
#define PENDING_MAX 16
typedef struct { char root[1024]; bool journaled; int npending; char pending[PENDING_MAX][1100]; } HostFs;
typedef struct { int fd; char path[1100]; } HostFile;

static void sync_dir_of(const char *path) {
    char d[1100];
    snprintf(d, sizeof d, "%s", path);
    char *slash = strrchr(d, '/');
    if (!slash) return;
    *slash = 0;
#ifndef _WIN32
    int fd = open(d, O_RDONLY);
    if (fd >= 0) { fsync(fd); close(fd); }
#endif
}

// the working copy over the file in one step (Windows' rename doesn't replace an existing file; with the UTF-8
// code page the A function takes our paths as they are)
static int replace_file(const char *from, const char *to) {
#ifdef _WIN32
    return MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) ? 0 : -1;
#else
    return rename(from, to);
#endif
}

static bool copy_file(const char *from, const char *to) {
    int in = open(from, O_RDONLY);
    int out = open(to, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    bool ok = out >= 0;
    char buf[65536];
    ssize_t n = 0;
    while (ok && in >= 0 && (n = read(in, buf, sizeof buf)) > 0) ok = write(out, buf, (size_t)n) == n;
    if (in >= 0) close(in);
    if (out >= 0) close(out);
    return ok && n >= 0;
}

// the working copy of a save file (made on its first open for writing; "" for the SD card mount)
static bool pending_of(HostFs *fs, const char *p, bool create, char *out, size_t n) {
    if (!fs->journaled) return false;
    snprintf(out, n, "%s.pending", p);
    for (int i = 0; i < fs->npending; i++) if (!strcmp(fs->pending[i], p)) return true;
    if (!create || fs->npending == PENDING_MAX) return false;
    if (!copy_file(p, out)) { unlink(out); return false; }
    snprintf(fs->pending[fs->npending++], sizeof fs->pending[0], "%s", p);
    return true;
}

static void commit(HostFs *fs) {
    for (int i = 0; i < fs->npending; i++) {
        char tmp[1100];
        snprintf(tmp, sizeof tmp, "%s.pending", fs->pending[i]);
        int fd = open(tmp, O_RDWR);
        if (fd >= 0) { fsync(fd); close(fd); }
        if (replace_file(tmp, fs->pending[i])) {
#ifdef _WIN32
            LOG("[save] commit %s failed (Windows error %lu)\n", fs->pending[i], (unsigned long)GetLastError());
#else
            LOG("[save] commit %s failed (%d)\n", fs->pending[i], errno);
#endif
        }
        else { LOG("[save] committed %s\n", fs->pending[i]); sync_dir_of(fs->pending[i]); }
    }
    fs->npending = 0;
}
typedef struct { DIR *d; char path[1024]; int mode; } HostDir;

// The game closed the file / directory: close the host one too. They used to stay open for the whole run - a leak
// everywhere, and on Windows the open working copy blocked the Commit's replace, so saves never landed (a file can't
// be replaced while it's open there; the game closes its files before it commits, as nn::fs requires).
static void file_close(Service *s) {
    HostFile *f = s->state;
    if (f) { close(f->fd); free(f); s->state = NULL; }
}
static void dir_close(Service *s) {
    HostDir *d = s->state;
    if (d) { if (d->d) closedir(d->d); free(d); s->state = NULL; }
}

static const char *in_path(IpcReq *req, int i) {
    if (i < req->nx) return (const char *)(uintptr_t)req->x[i].addr;
    if (i < req->na) return (const char *)(uintptr_t)req->a[i].addr;
    return "";
}

static void host_path(HostFs *fs, const char *p, char *out, size_t n) {
    snprintf(out, n, "%s%s%s", fs->root, p[0] == '/' ? "" : "/", p);
}

static uint32_t errno_result(void) {
    if (errno == ENOENT || errno == ENOTDIR) return FS_PATH_NOT_FOUND;
    if (errno == EEXIST) return FS_PATH_EXISTS;
    if (errno == ENOTEMPTY) return FS_NOT_EMPTY;
    return MAKE_RESULT(2, 4000);
}

static void file(Service *s, IpcReq *req, IpcResp *resp) {
    HostFile *f = s->state;
    switch (req->cmd) {
    case 0: {  // Read(option, offset, size) -> B
        int64_t off = IN(int64_t, 8), size = IN(int64_t, 16);
        IpcBuf *b = out_buf(req, 0);
        if ((uint64_t)size > b->size) size = (int64_t)b->size;
        ssize_t n = pread(f->fd, (void *)(uintptr_t)b->addr, (size_t)size, off);
        if (strstr(f->path, "savedata")) {
            LOG("[save] read %s off %lld size %lld -> guest %llx\n", f->path, (long long)off, (long long)size, (unsigned long long)b->addr);
            extern void save_editor_file_read(const char *path, uint64_t addr, int64_t off, int64_t size);
            save_editor_file_read(f->path, b->addr, off, n);
            extern void flags_ext_save_read(const char *path, uint64_t addr, int64_t off, int64_t size);
            flags_ext_save_read(f->path, b->addr, off, n);
        }
        OUT(int64_t, 0) = n < 0 ? 0 : n;
        resp_size(resp, 8);
        break;
    }
    case 1: {  // Write(option, offset, size) <- A
        int64_t off = IN(int64_t, 8), size = IN(int64_t, 16);
        IpcBuf *a = in_buf(req, 0);
        if (strstr(f->path, "savedata"))
            LOG("[save] write %s off %lld size %lld <- guest %llx\n", f->path, (long long)off, (long long)size, (unsigned long long)a->addr);
        if (pwrite(f->fd, (void *)(uintptr_t)a->addr, (size_t)size, off) < 0) resp->result = errno_result();
        else if (strstr(f->path, "savedata")) {
            extern void flags_ext_save_written(const char *path, uint64_t addr, int64_t off, int64_t size);
            flags_ext_save_written(f->path, a->addr, off, size);
        }
        break;
    }
    case 2: fsync(f->fd); break;
    case 3: if (ftruncate(f->fd, IN(int64_t, 0))) resp->result = errno_result(); break;
    case 4: {
        struct stat sb;
        fstat(f->fd, &sb);
        OUT(int64_t, 0) = sb.st_size;
        resp_size(resp, 8);
        break;
    }
    default: LOG("[fs] IFile cmd %u (ok)\n", req->cmd);
    }
}

static void directory(Service *s, IpcReq *req, IpcResp *resp) {
    HostDir *d = s->state;
    switch (req->cmd) {
    case 0: {  // Read -> B buffer of 0x310-byte entries
        IpcBuf *b = out_buf(req, 0);
        uint8_t *out = (uint8_t *)(uintptr_t)b->addr;
        int64_t max = (int64_t)(b->size / 0x310), n = 0;
        struct dirent *e;
        while (n < max && (e = readdir(d->d))) {
            if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
#ifdef _WIN32  // no d_type
            char full[2048];
            struct stat es;
            snprintf(full, sizeof full, "%s/%s", d->path, e->d_name);
            bool isdir = !stat(full, &es) && S_ISDIR(es.st_mode);
#else
            bool isdir = e->d_type == DT_DIR;
#endif
            if ((isdir && !(d->mode & 1)) || (!isdir && !(d->mode & 2))) continue;
            uint8_t *ent = out + n * 0x310;
            memset(ent, 0, 0x310);
            strncpy((char *)ent, e->d_name, 0x300);
            ent[0x304] = isdir ? 0 : 1;
            if (!isdir) {
                char p[2048];
                struct stat sb;
                snprintf(p, sizeof p, "%s/%s", d->path, e->d_name);
                if (!stat(p, &sb)) *(int64_t *)(ent + 0x308) = sb.st_size;
            }
            n++;
        }
        OUT(int64_t, 0) = n;
        resp_size(resp, 8);
        break;
    }
    case 1: {  // GetEntryCount
        long pos = telldir(d->d);
        rewinddir(d->d);
        int64_t n = 0;
        struct dirent *e;
        while ((e = readdir(d->d)))
            if (strcmp(e->d_name, ".") && strcmp(e->d_name, "..")) n++;
        seekdir(d->d, pos);
        OUT(int64_t, 0) = n;
        resp_size(resp, 8);
        break;
    }
    }
}

static int rm_rf(const char *p, bool keep_root) {
    DIR *d = opendir(p);
    if (!d) return unlink(p);
    struct dirent *e;
    while ((e = readdir(d))) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        char c[2048];
        snprintf(c, sizeof c, "%s/%s", p, e->d_name);
        rm_rf(c, false);
    }
    closedir(d);
    return keep_root ? 0 : rmdir(p);
}

static void filesystem(Service *s, IpcReq *req, IpcResp *resp) {
    HostFs *fs = s->state;
    char p[2048], p2[2048];
    host_path(fs, in_path(req, 0), p, sizeof p);
    switch (req->cmd) {
    case 0: {  // CreateFile(option, size)
        int fd = open(p, O_CREAT | O_EXCL | O_RDWR, 0644);
        if (fd < 0) { resp->result = errno_result(); break; }
        if (ftruncate(fd, IN(int64_t, 8))) resp->result = errno_result();
        close(fd);
        break;
    }
    case 1: if (unlink(p)) resp->result = errno_result(); break;
    case 2: if (mkdir(p, 0755)) resp->result = errno_result(); break;
    case 3: if (rmdir(p)) resp->result = errno_result(); break;
    case 4: if (rm_rf(p, false)) resp->result = errno_result(); break;
    case 5: case 6:
        host_path(fs, in_path(req, 1), p2, sizeof p2);
        if (rename(p, p2)) resp->result = errno_result();
        break;
    case 7: {  // GetEntryType
        struct stat sb;
        if (stat(p, &sb)) { resp->result = FS_PATH_NOT_FOUND; break; }
        OUT(uint32_t, 0) = S_ISDIR(sb.st_mode) ? 0 : 1;
        resp_size(resp, 4);
        break;
    }
    case 8: {  // OpenFile(mode)
        uint32_t mode = IN(uint32_t, 0);
        char work[2100];
        struct stat sb;
        if (!stat(p, &sb) && pending_of(fs, p, (mode & 2) != 0, work, sizeof work)) snprintf(p, sizeof p, "%s", work);
        int fd = open(p, (mode & 2) ? O_RDWR : O_RDONLY);
        if (fd < 0) { resp->result = errno_result(); break; }
        HostFile *f = calloc(1, sizeof *f);
        f->fd = fd;
        strlcpy(f->path, p, sizeof f->path);
        Service *fsvc = service_new("IFile", file, f);
        fsvc->on_close = file_close;
        ipc_return_object(resp, fsvc);
        break;
    }
    case 9: {  // OpenDirectory(mode)
        DIR *d = opendir(p);
        if (!d) { resp->result = errno_result(); break; }
        HostDir *hd = calloc(1, sizeof *hd);
        hd->d = d;
        hd->mode = (int)IN(uint32_t, 0);
        snprintf(hd->path, sizeof hd->path, "%s", p);
        Service *dsvc = service_new("IDirectory", directory, hd);
        dsvc->on_close = dir_close;
        ipc_return_object(resp, dsvc);
        break;
    }
    case 10: commit(fs); break;  // Commit
    case 11: case 12: OUT(int64_t, 0) = 0x40000000; resp_size(resp, 8); break;
    case 13: rm_rf(p, true); break;
    case 14: memset(resp->data, 0, 0x20); resp_size(resp, 0x20); break;
    default: LOG("[fs] IFileSystem cmd %u (ok)\n", req->cmd);
    }
}

static Service *open_host_fs(const char *sub) {
    HostFs *fs = calloc(1, sizeof *fs);
    if (getenv("LARECOMP_SAVE_DIR")) g_save_dir = getenv("LARECOMP_SAVE_DIR");  // test runs use a scratch copy
    else if (rando_seed_dir()) {  // each randomizer seed has its own save slots
        static char seed_save[600];
        snprintf(seed_save, sizeof seed_save, "%s/save", rando_seed_dir());
        g_save_dir = seed_save;
    }
    snprintf(fs->root, sizeof fs->root, "%s/%s", g_save_dir, sub);
    char cmd[1200];
    snprintf(cmd, sizeof cmd, "%s", fs->root);
    for (char *q = cmd + 1; *q; q++)
        if (*q == '/') { *q = 0; mkdir(cmd, 0755); *q = '/'; }
    mkdir(cmd, 0755);
    fs->journaled = strcmp(sub, "sdcard") != 0;
    if (fs->journaled) {  // (a working copy left by a crash: never committed - dropped, as on the Switch)
        DIR *d = opendir(fs->root);
        struct dirent *e;
        while (d && (e = readdir(d))) {
            size_t l = strlen(e->d_name);
            if (l > 8 && !strcmp(e->d_name + l - 8, ".pending")) {
                char stale[1300];
                snprintf(stale, sizeof stale, "%s/%s", fs->root, e->d_name);
                LOG("[save] dropping uncommitted %s\n", stale);
                unlink(stale);
            }
        }
        if (d) closedir(d);
    }
    LOG("[fs] mounted host directory %s\n", fs->root);
    return service_new("IFileSystem", filesystem, fs);
}

// ------------------------------------------------------------------ fsp-srv
SERVICE("fsp-srv", fsp_srv) {
    (void)s;
    switch (req->cmd) {
    case 1: break;  // SetCurrentProcess
    case 18: ipc_return_object(resp, open_host_fs("sdcard")); break;
    case 22: break;  // CreateSaveDataFileSystem
    case 51: case 53: {  // OpenSaveDataFileSystem(space, attribute)
        uint8_t type = IN(uint8_t, 0x8 + 0x18);  // attribute.type
        char sub[64];
        snprintf(sub, sizeof sub, "save_%u", type);
        ipc_return_object(resp, open_host_fs(sub));
        break;
    }
    case 200: ipc_return_object(resp, open_romfs_storage()); break;  // OpenDataStorageByCurrentProcess
    case 202: {  // OpenDataStorageByDataId: DLC (services/aoc.c), else system data archives (not available)
        extern Service *aoc_open_storage(uint64_t id);
        Service *dlc = aoc_open_storage(IN(uint64_t, 8));
        if (dlc) { ipc_return_object(resp, dlc); break; }
        LOG("[fs] OpenDataStorageByDataId %llx (not available)\n", IN(uint64_t, 8));
        resp->result = MAKE_RESULT(2, 1001);
        break;
    }
    case 1005: OUT(uint32_t, 0) = 0; resp_size(resp, 4); break;  // GetGlobalAccessLogMode
    case 1003: case 1004: case 1006: break;
    case 1011: memset(resp->data, 0, 0x10); resp_size(resp, 0x10); break;
    default: LOG("[fs] fsp-srv cmd %u (ok)\n", req->cmd);
    }
}
