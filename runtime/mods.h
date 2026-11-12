// Mods: folders in <data root>/mods (runtime/mods.c, docs/mods.md).
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

// RomFS images with the mods' files (runtime/services/fs.c); sub: "romfs" or "dlc/<id>/romfs"
typedef struct Vfs Vfs;
Vfs *romfs_vfs_open(int fd, const char *sub);
ssize_t romfs_vfs_read(Vfs *v, uint8_t *dst, uint64_t size, uint64_t off);
uint64_t romfs_vfs_size(const Vfs *v);
void romfs_vfs_free(Vfs *v);

// enabled mods' folders holding `sub`, lowest priority first
int mods_roots(const char *sub, char roots[][600], int max);
void mods_boot(void);                    // each boot, after the modules are loaded: exefs patches
uint32_t mods_code_word(uint64_t addr);  // patched instruction at addr (0: none); asked by cheat_word()

typedef struct {
    const char *name;
    bool on, romfs, dlc, exefs;
    bool needs_gen;  // has code patches the game code wasn't built with (seen at the last boot)
    int ncode, ndata;
} ModInfo;
int mods_count(void);
void mods_rescan(void);
bool mods_info(int i, ModInfo *out);
// these apply at the next boot (mods/mods.txt: order and on/off)
void mods_set(int i, bool on);
void mods_move(int from, int to);
int mods_conflict(int a, int b, const char **example);  // files both change (example: one of them)

#ifdef __cplusplus
}
#endif
