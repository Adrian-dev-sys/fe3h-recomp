// Mods: one folder each in <data root>/mods (docs/mods.md), Switch (LayeredFS) layout:
//   romfs/            game files, replaced or added (services/fs.c)
//   dlc/<id>/romfs/   a DLC pack's files (services/aoc.c); <id>/romfs/ works too
//   exefs/            .pchtxt / .ips patches of the main module, applied each boot
// mods/mods.txt keeps the load order and on/off ("+name" / "-name"); a later mod wins. New mods go last, on (a name
// starting with '_' off). Code patches can't change recompiled code by writing memory: gen.py compiles them in as
// variants (tools/cheats.py) and cheat_word() asks mods_code_word() which one runs.
#include "mods.h"

#include <ctype.h>
#include <dirent.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>

#include "hle.h"
#include "hle/cheatvm.h"

#define MAX_MODS 128
#define MAX_WORDS 4096
#define LIST_FILE "mods/mods.txt"
#define OLD_DISABLED_FILE "mods/disabled.txt"

typedef struct {
    char name[128];
    bool on, romfs, dlc, exefs, needs_gen;
    int ncode, ndata;
    char **files;  // what it changes, sorted: "romfs/<path>", "dlc/<ID>/romfs/<path>" (conflicts)
    int nfiles;
} Mod;

static Mod g_mods[MAX_MODS];
static int g_nmods = -1;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static struct { uint64_t addr; uint32_t word; } g_words[MAX_WORDS];
static int g_nwords;
static char g_boot[MAX_MODS][128];  // the enabled mods this boot runs with, in order (the list may change meanwhile)
static int g_nboot = -1;

static bool is_dir(const char *p) {
    struct stat sb;
    return !stat(p, &sb) && S_ISDIR(sb.st_mode);
}

static int name_cmp(const void *a, const void *b) { return strcmp(((const Mod *)a)->name, ((const Mod *)b)->name); }
static int str_cmp(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

static bool is_id(const char *s) {
    size_t n = strlen(s), k = 0;
    while (k < n && isxdigit((unsigned char)s[k])) k++;
    return n == 16 && k == 16;
}

static void add_files(Mod *m, const char *host, const char *rel, int *cap) {
    DIR *d = opendir(host);
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        char h2[1200], r2[700];
        snprintf(h2, sizeof h2, "%s/%s", host, e->d_name);
        snprintf(r2, sizeof r2, "%s/%s", rel, e->d_name);
        if (is_dir(h2)) { add_files(m, h2, r2, cap); continue; }
        if (m->nfiles == *cap) { *cap = *cap ? *cap * 2 : 64; m->files = realloc(m->files, sizeof *m->files * *cap); }
        for (char *c = r2; *c; c++) *c = (char)tolower((unsigned char)*c);  // (Windows paths: case doesn't matter)
        m->files[m->nfiles++] = strdup(r2);
    }
    closedir(d);
}

static void index_mod(Mod *m) {
    char p[700];
    int cap = 0;
    snprintf(p, sizeof p, "mods/%s/romfs", m->name);
    if ((m->romfs = is_dir(p))) add_files(m, p, "romfs", &cap);
    snprintf(p, sizeof p, "mods/%s/exefs", m->name);
    m->exefs = is_dir(p);
    for (int pass = 0; pass < 2; pass++) {  // dlc/<id>/romfs, and <id>/romfs at the top
        snprintf(p, sizeof p, pass ? "mods/%s" : "mods/%s/dlc", m->name);
        DIR *d = opendir(p);
        struct dirent *e;
        while (d && (e = readdir(d))) {
            if (!is_id(e->d_name)) continue;
            char h[800], r[64];
            snprintf(h, sizeof h, "%s/%s/romfs", p, e->d_name);
            snprintf(r, sizeof r, "dlc/%s/romfs", e->d_name);
            if (is_dir(h)) m->dlc = true, add_files(m, h, r, &cap);
        }
        if (d) closedir(d);
    }
    if (m->nfiles) qsort(m->files, m->nfiles, sizeof *m->files, str_cmp);
}

static void save_locked(void) {
    FILE *f = fopen(LIST_FILE, "w");
    if (!f) return;
    for (int i = 0; i < g_nmods; i++) fprintf(f, "%c%s\n", g_mods[i].on ? '+' : '-', g_mods[i].name);
    fclose(f);
}

static void scan_locked(void) {
    if (g_nmods >= 0) return;
    g_nmods = 0;
    DIR *d = opendir("mods");
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d)) && g_nmods < MAX_MODS) {
        char p[600];
        snprintf(p, sizeof p, "mods/%s", e->d_name);
        if (e->d_name[0] == '.' || !is_dir(p)) continue;
        Mod *m = &g_mods[g_nmods++];
        memset(m, 0, sizeof *m);
        strlcpy(m->name, e->d_name, sizeof m->name);
        m->on = e->d_name[0] != '_';
        index_mod(m);
    }
    closedir(d);
    qsort(g_mods, g_nmods, sizeof *g_mods, name_cmp);
    // the saved order first (the rest keep name order, after it)
    FILE *f = fopen(LIST_FILE, "r");
    bool old = !f;
    if (!f) f = fopen(OLD_DISABLED_FILE, "r");
    if (f) {
        char line[256];
        int at = 0;
        while (fgets(line, sizeof line, f)) {
            line[strcspn(line, "\r\n")] = 0;
            const char *nm = old ? line : line + 1;
            if (!old && line[0] != '+' && line[0] != '-') continue;
            for (int i = at; i < g_nmods; i++)
                if (!strcmp(g_mods[i].name, nm)) {
                    g_mods[i].on = !old && line[0] == '+';
                    if (!old) { Mod t = g_mods[i]; memmove(g_mods + at + 1, g_mods + at, sizeof *g_mods * (i - at)); g_mods[at++] = t; }
                    break;
                }
        }
        fclose(f);
    }
}

static void free_locked(void) {
    for (int i = 0; i < g_nmods; i++) {
        for (int k = 0; k < g_mods[i].nfiles; k++) free(g_mods[i].files[k]);
        free(g_mods[i].files);
    }
    g_nmods = -1;
}

static void scan(void) {
    pthread_mutex_lock(&g_lock);
    scan_locked();
    pthread_mutex_unlock(&g_lock);
}

int mods_roots(const char *sub, char roots[][600], int max) {
    pthread_mutex_lock(&g_lock);
    scan_locked();
    if (g_nboot < 0) {  // (before mods_boot: the list as it is)
        g_nboot = 0;
        for (int i = 0; i < g_nmods; i++)
            if (g_mods[i].on) strlcpy(g_boot[g_nboot++], g_mods[i].name, sizeof g_boot[0]);
    }
    int n = 0;
    for (int i = 0; i < g_nboot && n < max; i++) {
        snprintf(roots[n], 600, "mods/%s/%s", g_boot[i], sub);
        if (is_dir(roots[n])) { n++; continue; }
        if (!strncmp(sub, "dlc/", 4)) {  // (<id>/romfs at the mod's top)
            snprintf(roots[n], 600, "mods/%s/%s", g_boot[i], sub + 4);
            if (is_dir(roots[n])) n++;
        }
    }
    pthread_mutex_unlock(&g_lock);
    return n;
}

// ------------------------------------------------------------------ exefs patches
static const Module *main_module(void) {
    for (int i = 0; i < g_nmodules; i++)
        if (g_modules[i].name && !strcmp(g_modules[i].name, "main")) return &g_modules[i];
    return g_nmodules ? &g_modules[0] : NULL;
}

// patch files name the module by its build id, as hex (trailing zero bytes may be left out)
static bool bid_match(const char *hex, size_t n, const char *full) {
    if (n < 16 || n > 64) return false;
    for (size_t i = 0; i < 64; i++) {
        char c = i < n ? (char)toupper((unsigned char)hex[i]) : '0';
        if (i < n && !isxdigit((unsigned char)c)) return false;
        if (c != full[i]) return false;
    }
    return true;
}

static void put_word(uint64_t addr, uint32_t w) {
    for (int i = 0; i < g_nwords; i++)
        if (g_words[i].addr == addr) { g_words[i].word = w; return; }
    if (g_nwords < MAX_WORDS) g_words[g_nwords].addr = addr, g_words[g_nwords++].word = w;
}

// bytes at an offset of the main module's image: code goes to the variant table, the rest into memory
static void apply(Mod *m, const Module *mm, uint64_t off, const uint8_t *b, size_t n) {
    uint64_t a = mm->base + off, t0 = mm->base + mm->text_off, t1 = t0 + mm->text_size;
    if (a + n <= t0 || a >= t1) {
        uint8_t *p = (uint8_t *)(uintptr_t)a;
        uintptr_t page = (uintptr_t)p & ~0x3FFFull;
        plat_protect(page, ((uintptr_t)p + n - page + 0x3FFF) & ~0x3FFFull, PLAT_PROT_RW);
        memcpy(p, b, n);
        m->ndata++;
        return;
    }
    for (uint64_t w = a & ~3ull; w < a + n; w += 4) {
        uint32_t word, orig;
        memcpy(&orig, (const void *)(uintptr_t)w, 4);  // (the original, for partly patched words)
        word = orig;
        for (int k = 0; k < 4; k++)
            if (w + k >= a && w + k < a + n) ((uint8_t *)&word)[k] = b[w + k - a];
        if (word == orig) continue;
        put_word(w, word);
        m->ncode++;
        if (!cheatvm_compiled(w, word)) {
            if (!m->needs_gen) LOG("[mods] %s: code patch at main+%llx isn't built in (rebuild the game code)\n", m->name,
                                   (unsigned long long)(w - mm->base));
            m->needs_gen = true;
        }
    }
}

static void load_pchtxt(Mod *m, const Module *mm, const char *path, const char *bid) {
    FILE *f = fopen(path, "r");
    if (!f) return;
    char line[4096];
    bool enabled = false, ours = false;
    long shift = 0;
    while (fgets(line, sizeof line, f)) {
        char *p = line;
        while (isspace((unsigned char)*p)) p++;
        if (!strncmp(p, "@nsobid-", 8)) {
            size_t n = strspn(p + 8, "0123456789abcdefABCDEF");
            ours = bid_match(p + 8, n, bid);
            continue;
        }
        if (!strncmp(p, "@flag offset_shift", 18)) { shift = strtol(p + 18, NULL, 0); continue; }
        if (!strncmp(p, "@enabled", 8)) { enabled = true; continue; }
        if (!strncmp(p, "@disabled", 9) || !strncmp(p, "@stop", 5)) { enabled = false; continue; }
        if (!enabled || !ours || !isxdigit((unsigned char)*p)) continue;
        char *e;
        unsigned long off = strtoul(p, &e, 16);
        while (isspace((unsigned char)*e)) e++;
        uint8_t bytes[2048];
        size_t n = 0;
        if (*e == '"') {  // "text" (a string, with its terminator)
            char *q = strrchr(e + 1, '"');
            if (!q) continue;
            n = (size_t)(q - e - 1) < sizeof bytes - 1 ? (size_t)(q - e - 1) : sizeof bytes - 1;
            memcpy(bytes, e + 1, n);
            bytes[n++] = 0;
        } else {
            while (isxdigit((unsigned char)e[0]) && isxdigit((unsigned char)e[1]) && n < sizeof bytes) {
                unsigned v;
                sscanf(e, "%2x", &v);
                bytes[n++] = (uint8_t)v;
                e += 2;
            }
        }
        long img = (long)off + shift - 0x100;  // (offsets count the 0x100-byte NSO header unless shifted)
        if (n && img >= 0) apply(m, mm, (uint64_t)img, bytes, n);
    }
    fclose(f);
}

static void load_ips(Mod *m, const Module *mm, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return;
    char magic[5];
    bool ips32 = false;
    if (fread(magic, 1, 5, f) != 5) { fclose(f); return; }
    if (!memcmp(magic, "IPS32", 5)) ips32 = true;
    else if (memcmp(magic, "PATCH", 5)) { fclose(f); LOG("[mods] %s: not an IPS file\n", path); return; }
    for (;;) {
        uint8_t h[4];
        int ol = ips32 ? 4 : 3;
        if (fread(h, 1, ol, f) != (size_t)ol) break;
        if ((!ips32 && !memcmp(h, "EOF", 3)) || (ips32 && !memcmp(h, "EEOF", 4))) break;
        uint64_t off = 0;
        for (int i = 0; i < ol; i++) off = off << 8 | h[i];
        uint8_t s[2];
        if (fread(s, 1, 2, f) != 2) break;
        size_t n = (size_t)s[0] << 8 | s[1];
        uint8_t *buf;
        if (n == 0) {  // run: count, value
            uint8_t r[3];
            if (fread(r, 1, 3, f) != 3) break;
            n = (size_t)r[0] << 8 | r[1];
            buf = malloc(n ? n : 1);
            memset(buf, r[2], n);
        } else {
            buf = malloc(n);
            if (fread(buf, 1, n, f) != n) { free(buf); break; }
        }
        if (off >= 0x100 && n) apply(m, mm, off - 0x100, buf, n);  // (the NSO header comes first)
        free(buf);
    }
    fclose(f);
}

void mods_boot(void) {
    pthread_mutex_lock(&g_lock);
    free_locked();  // (rescan: mods may have been added or switched since the last boot)
    g_nwords = 0;
    scan_locked();
    g_nboot = 0;
    for (int i = 0; i < g_nmods; i++)
        if (g_mods[i].on) strlcpy(g_boot[g_nboot++], g_mods[i].name, sizeof g_boot[0]);
    const Module *mm = main_module();
    char bid[65] = "";
    if (mm && g_romfs_dir) {  // main's build id: its NSO header, <game files>/exefs/main
        char p[1100];
        snprintf(p, sizeof p, "%.*sexefs/main", (int)(strlen(g_romfs_dir) - 5), g_romfs_dir);
        FILE *f = fopen(p, "rb");
        uint8_t h[0x60];
        if (f && fread(h, 1, sizeof h, f) == sizeof h && !memcmp(h, "NSO0", 4))
            for (int i = 0; i < 32; i++) snprintf(bid + 2 * i, 3, "%02X", h[0x40 + i]);
        if (f) fclose(f);
    }
    for (int i = 0; i < g_nmods; i++) {
        Mod *m = &g_mods[i];
        if (!m->on) { LOG("[mods] %s: off\n", m->name); continue; }
        if (m->exefs && mm && bid[0]) {
            char dir[600];
            snprintf(dir, sizeof dir, "mods/%s/exefs", m->name);
            DIR *d = opendir(dir);
            struct dirent *e;
            while (d && (e = readdir(d))) {
                char p[1200];
                snprintf(p, sizeof p, "%s/%s", dir, e->d_name);
                const char *dot = strrchr(e->d_name, '.');
                if (!dot) continue;
                if (!strcasecmp(dot, ".pchtxt")) load_pchtxt(m, mm, p, bid);
                else if (!strcasecmp(dot, ".ips") && bid_match(e->d_name, (size_t)(dot - e->d_name), bid)) load_ips(m, mm, p);
                else if (!strcasecmp(dot, ".ips")) LOG("[mods] %s: %s is for another module / game version\n", m->name, e->d_name);
            }
            if (d) closedir(d);
            if (!m->ncode && !m->ndata) LOG("[mods] %s: no exefs patches for this game version (main %.16s...)\n", m->name, bid);
        }
        LOG("[mods] %s: on%s%s%s, %d code words, %d data patches\n", m->name, m->romfs ? ", romfs" : "",
            m->dlc ? ", dlc" : "", m->exefs ? ", exefs" : "", m->ncode, m->ndata);
    }
    pthread_mutex_unlock(&g_lock);
}

uint32_t mods_code_word(uint64_t addr) {
    for (int i = 0; i < g_nwords; i++)
        if (g_words[i].addr == addr) return g_words[i].word;
    return 0;
}

// ------------------------------------------------------------------ the menu's side (runtime/ui/mods_tab.mm)
int mods_count(void) {
    scan();
    return g_nmods;
}

void mods_rescan(void) {  // (folders added or removed while the menu is open)
    pthread_mutex_lock(&g_lock);
    bool gen[MAX_MODS];
    char names[MAX_MODS][128];
    int n = g_nmods > 0 ? g_nmods : 0;
    for (int i = 0; i < n; i++) gen[i] = g_mods[i].needs_gen, strlcpy(names[i], g_mods[i].name, sizeof names[0]);
    free_locked();
    scan_locked();
    for (int i = 0; i < g_nmods; i++)  // (what the last boot found about code patches stays)
        for (int k = 0; k < n; k++)
            if (!strcmp(names[k], g_mods[i].name)) g_mods[i].needs_gen = gen[k];
    pthread_mutex_unlock(&g_lock);
}

bool mods_info(int i, ModInfo *out) {
    if (i < 0 || i >= g_nmods) return false;
    const Mod *m = &g_mods[i];
    *out = (ModInfo){m->name, m->on, m->romfs, m->dlc, m->exefs, m->needs_gen, m->ncode, m->ndata};
    return true;
}

void mods_set(int i, bool on) {
    pthread_mutex_lock(&g_lock);
    if (i >= 0 && i < g_nmods) {
        g_mods[i].on = on;
        save_locked();
        LOG("[mods] %s %s (next boot)\n", g_mods[i].name, on ? "on" : "off");
    }
    pthread_mutex_unlock(&g_lock);
}

void mods_move(int from, int to) {
    pthread_mutex_lock(&g_lock);
    if (from >= 0 && from < g_nmods && to >= 0 && to < g_nmods && from != to) {
        Mod t = g_mods[from];
        if (from < to) memmove(g_mods + from, g_mods + from + 1, sizeof *g_mods * (to - from));
        else memmove(g_mods + to + 1, g_mods + to, sizeof *g_mods * (from - to));
        g_mods[to] = t;
        save_locked();
    }
    pthread_mutex_unlock(&g_lock);
}

int mods_conflict(int a, int b, const char **example) {
    if (a < 0 || b < 0 || a >= g_nmods || b >= g_nmods || a == b) return 0;
    const Mod *x = &g_mods[a], *y = &g_mods[b];
    int n = 0;
    for (int i = 0, j = 0; i < x->nfiles && j < y->nfiles;) {  // (both sorted)
        int c = strcmp(x->files[i], y->files[j]);
        if (c == 0) {
            if (!n && example) *example = x->files[i];
            n++, i++, j++;
        } else if (c < 0) i++;
        else j++;
    }
    return n;
}