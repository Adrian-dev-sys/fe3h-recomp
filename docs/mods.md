# Mods

Each mod goes in its own folder under `mods/` in the data folder (next to `extracted/`), in the usual Switch mod
layout (GameBanana / Ryujinx / yuzu):

```
mods/
  PlayableSeiros/romfs/patch4/nx/data/fixed_persondata.bin   game files (replaced or added)
  SomeCodeMod/exefs/<name>.pchtxt  or  <build id>.ips        code / data patches of the main module
  SomeDlcMod/dlc/010055D009F79001/romfs/...                  a DLC pack's files (<id>/romfs/ also works)
  _Old/...                                                   a leading "_": off until ticked
```

- Settings > "Mods" tab (start screen and in game): a checkbox per mod, drag rows (or the arrows) to set the load
  order, "Open mods folder", "Refresh". Saved in `mods/mods.txt` (`+name` / `-name`, in order); new mods go last.
  Changes apply the next time the game starts (each boot takes a snapshot of the list).
- Lower in the list wins when two mods ship the same file. Enabled mods that change the same files get a warning in
  the tab (and a popup when ticking one on); `run.log` shows `[mods] ... wins over ...`.
- `run.log` lists each mod at boot: `[mods] <name>: on, romfs, dlc, exefs, N code words, M data patches`.

## Code patches (exefs)

- `.pchtxt`: only `@enabled` sections whose `@nsobid-` matches the game's main build id; `@flag offset_shift` is
  honoured. `.ips` / IPS32: the file name must be main's build id.
- Data patches (outside the code) are written into memory at boot.
- Code patches can't change recompiled code at run time. The game code generator compiles each patched instruction
  in as a variant (`tools/cheats.py mod_variants`, the same path as code-patching cheats), and `cheat_word()` asks
  `mods_code_word()` which one runs. A code mod added later needs a rebuild: Settings > Game Files > "Rebuild game
  code" (source checkout: `make gen-main` then `make`). The log line `isn't built in` and the Mods list say so.
  Code variants of every mod are compiled in, so switching mods on/off later needs no rebuild.

## How it works

- `runtime/mods.c`: mod list, order, on/off, conflicts (file lists), exefs patches; `runtime/ui/mods_tab.mm`: the tab. `runtime/services/fs.c` (`romfs_vfs_open`): the game reads RomFS
  images raw and parses their tables itself. Replaced and new files are appended past the end of the image, and
  copies of the file / directory tables, the hash tables and the header are patched to point at them. New folders
  are created too (mods whose `INFO0.bin` points at their own `romfs/mods/...` files). DLC images
  (`services/aoc.c`) use the same overlay.
