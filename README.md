# FE3H Recomp

Native Windows recompilation of *Fire Emblem: Three Houses* (Nintendo Switch, version 1.2.0), a fork of
[FE3H Recomp](https://github.com/Bakanda07/fe3h-recomp) built on [LAS Recomp](https://github.com/peiikai/LASRecomp).

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## About this fork

This is a personal fork of [FE3H Recomp](https://github.com/Bakanda07/fe3h-recomp), made for my own use. It has no
releases and no ready-made download: build it from source as described below. Its changes were written with an AI
assistant (Claude) and tested by hand on one Windows PC with an NVIDIA GPU and a 3440x1440 ultrawide screen. Its
additions include:

- **Ultrawide** (21:9 and wider): the 3D view fills the screen while the HUD, menus and movies stay in 16:9; map unit
  icons, monastery speech bubbles and name tags stay over their characters (also when zoomed in).
- **Resolution and sharpness:** internal resolution with a supersampling resolve (1.5x and up) and a sharp UI, forced
  16x anisotropic filtering, a texture detail option, and RCAS sharpening levels up to Extreme.
- **NVIDIA DLAA and DLSS Frame Generation** (2x-4x, with Reflex) through Streamline. The game logic stays at its
  fixed 60 Hz; frame generation adds the extra frames.
- **Rendering fixes:** full-resolution lighting (faces and walls no longer blurred above 1x), character motion
  vectors, shadow-mask artefacts, black boxes from NaN pixels, DLC map floor lights cutting off.
- **Performance:** 2x internal resolution went from about 44 to a steady 60 fps (shader local memory sizing, static
  branch targets, light lists in GPU memory), and a host-side frame limiter.
- **Mods** in the usual Switch layout from a `mods/` folder: game files, DLC files and exefs code patches (`.pchtxt` /
  `.ips`). A Mods tab in Settings switches them on and off, sets their load order by drag and drop and warns when two
  enabled mods change the same files. See [docs/mods.md](docs/mods.md).
- **Debug launchers** used while making these changes (F9 frame capture, GPU profiling and others) are kept in
  [tools/debug](tools/debug).

The NVIDIA libraries are not included. To use DLAA and frame generation, put the
[Streamline SDK](https://github.com/NVIDIA-RTX/Streamline)'s `sl.interposer.dll`, `sl.common.dll`, `sl.dlss.dll`,
`nvngx_dlss.dll` and, for frame generation, `sl.dlss_g.dll`, `nvngx_dlssg.dll`, `sl.reflex.dll`, `sl.pcl.dll` and
`NvLowLatencyVk.dll` next to `fe3hrecomp.exe`. Without them the game runs as before, with SMAA.

## Game files stay on your computer

This repository contains the port's source and tools, not the game. Supply your own legally obtained copy of the
game, version **1.2.0**, extracted to `exefs` and `romfs` folders (and optionally your DLC), and build locally. No
instructions for obtaining game files are provided.

Never commit or upload game files, extracted or generated game code (`extracted/`, `generated/`), a built game
library, logs or F9 captures (they contain game code and data), mods, saves or the NVIDIA libraries. `.gitignore`
covers the usual folders.

## Current status

On the tested PC (Windows 11, NVIDIA GPU, 3440x1440):

- The monastery, battles and DLC maps play at a steady 60 fps at 2x internal resolution with DLAA. The upstream
  author also tested cutscenes, the calendar, combat arts and gambits with their cut-ins, and the start of Cindered
  Shadows.
- Audio, keyboard and controllers work. Press **F1** (or Minus + Plus on a controller) for the settings menu.
- Ultrawide, DLAA, frame generation and game-file mods work.

Known limits:

- The first time an area or a battle effect is seen, its shaders are built while it plays: some pop-in or stutter,
  only once. The largest battle-effect compute shaders take a few minutes to build on NVIDIA the first time; on other
  GPUs they're skipped (`LARECOMP_VK_BIG_CS=1` builds them anyway), so some battle effects may be missing a layer.
- Objects in the monastery pop in at a distance.
- DLAA and frame generation need an NVIDIA RTX GPU and the libraries above; other GPUs use SMAA.
- Cheats and mods that patch the game's code need the game code rebuilt after they're added (see below). DLC-file and
  code mods were only checked with a test mod; game-file mods were tested with real ones.
- One texture format (BC3, written by a compute shader) can't be written to on PC GPUs; the log mentions it. No
  visible effect found so far.
- Experience multiplier cheats can overflow the EXP bar on the results screen for some classes (archers); the game
  itself is fine.
- Long playthroughs, other GPUs, macOS and Linux are untested with this fork. The upstream author tested its base on
  AMD GPUs.

## Building from source

Requirements: Windows 10 or 11 (64-bit), a CPU with AVX2, a GPU with Vulkan 1.3 drivers, and about 20 GB free for the
game files and the build.

1. **One-time setup**, in PowerShell from the repo root:
   `powershell -ExecutionPolicy Bypass -File tools\setup_windows.ps1`
   (installs MSYS2 with clang, SDL3, FFmpeg, Vulkan and shaderc, plus Python for the tools). More detail in
   [docs/BUILD-windows.md](docs/BUILD-windows.md), written for Link's Awakening; the steps are the same.
2. **Game files:** put your extracted `exefs/` and `romfs/` in `extracted/`.
3. **Build**, from an MSYS2 UCRT64 shell:
   ```
   make gen          # once: recompile the game code from extracted/exefs into generated/
   make -j$(nproc)   # after make gen, hundreds of files compile
   ```
4. **First run:** start `./build/fe3hrecomp.exe` and, on the start screen, set up the game files (Settings > Game
   Files). That builds `extracted/romfs.bin` once. Then click **Play**.
5. **Later runs:** `./build/fe3hrecomp.exe extracted --play > run.log 2>&1` skips the start screen.

### DLC

Extract each DLC's romfs into a folder named by its content id:
```
extracted/dlc/010055D009F79001/romfs/...
extracted/dlc/010055D009F79002/romfs/...
```
The game sees every pack there as installed. The first launch converts each folder into a single image next to it
(once; it takes extra disk space about the size of the DLC).

### Mods

Put each mod in its own folder in `mods/` (`mods/<name>/romfs/...`, `exefs/`, `dlc/<content id>/romfs/...`) and
manage them in Settings > Mods. Changes apply the next time the game starts. Details: [docs/mods.md](docs/mods.md).

### Cheats

Cheats use the Atmosphere format: put them in `cheats.txt` in the project folder (a `[Name]` line, then its codes; a
`{Name}` block is a master code, always on; `#` lines under the name describe it), and they show up in the menu's
**Game** tab to switch on and off. The file stays out of git.

Cheats that only change memory work straight away. Cheats that patch the game's code (most multipliers and "max"
cheats) can't be applied to an already-recompiled game at run time, so their patched instructions get compiled in
beside the originals: after adding or changing those (or a code-patching mod), run `make gen-main` and then `make`.
The menu marks any cheat still waiting on that. Switching a cheat off puts back any code it changed; values it already
wrote stay. Cheats that patch the same instruction (Exp x2 / x4 / x8, say) exclude each other.

### Settings and logs

- **F1** opens the settings menu: Video, Audio, Controls, Game and Mods. For the sharpest image use Internal resolution
  1.5x or 2x with DLAA (SMAA without an NVIDIA GPU).
- Saves are in `save/`, settings in `fe3hrecomp.ini`.
- `run.log` is the first place to look when something goes wrong. `LARECOMP_*` environment variables turn on extra
  logging for specific systems (audio, textures, shaders); they're documented where they're read in `runtime/`.
- The start screen shows your own picture if you put one in the game files folder as `extracted/background.png` (or
  `.jpg`); `.gitignore` keeps it out.

The upstream author's notes on what it took to get *Three Houses* running (lighting, audio, stutter, battle effects,
cheats, setup time) are in [docs/upstream-notes.md](docs/upstream-notes.md).

## Credits and license

- **[FE3H Recomp](https://github.com/Bakanda07/fe3h-recomp)** by Bakanda07: the *Three Houses* port this fork builds on
  (audio, movies, lighting, DLC, shaders, cheats, the start screen and menus).
- **[LAS Recomp](https://github.com/peiikai/LASRecomp)** by Peiikai and its contributors: the recompiler, runtime and
  Vulkan renderer (MIT). Its README lists its full credits and third-party components, including the Link's Awakening
  randomizer still in `third_party/`.
- **[Ryujinx](https://ryujinx.app)** (MIT): the audio renderer's reverb, I3DL2 reverb and auxiliary send / return are
  ports of its implementation.
- **[NVIDIA Streamline](https://github.com/NVIDIA-RTX/Streamline)** (MIT): the SDK headers for DLAA and frame
  generation (`third_party/streamline`). The DLSS runtime libraries are NVIDIA's and are not included.
- **[AMD FidelityFX Super Resolution 1](https://github.com/GPUOpen-Effects/FidelityFX-FSR)** (MIT, including RCAS) and
  **[SMAA](https://github.com/iryoku/smaa)** (MIT).
- The Switch modding community (Atmosphere, yuzu, Ryujinx) for the LayeredFS mod layout and the IPSwitch `.pchtxt` /
  IPS patch formats, which the mod loader reads (written anew, no code copied), and Fl4sh9174's "Level of Detail" mod
  for FE3H, which inspired the texture detail option.
- **[EB Garamond](https://github.com/georgd/EB-Garamond)** by Georg Duffner: the start screen's title (SIL Open Font
  License 1.1, `third_party/fonts/EBGaramond/OFL.txt`).
- **Libraries:** [SDL 3](https://libsdl.org) (zlib), [FFmpeg](https://ffmpeg.org) (LGPL),
  [Dear ImGui](https://github.com/ocornut/imgui) (MIT),
  [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) (MIT),
  [shaderc](https://github.com/google/shaderc) (Apache 2.0); the code generator uses
  [Capstone](https://www.capstone-engine.org), [Unicorn](https://www.unicorn-engine.org) and
  [python-lz4](https://github.com/python-lz4/python-lz4).
- The Switch homebrew community's documentation ([Switchbrew](https://switchbrew.org)).

The code is under the [MIT License](LICENSE), as LAS Recomp's is; third-party components keep their own licenses. The
license covers this project's code only and grants nothing for Nintendo's game. This is an independent fan project,
not affiliated with, endorsed by or sponsored by Nintendo, Intelligent Systems or Koei Tecmo. *Fire Emblem* and
*Fire Emblem: Three Houses* are trademarks of Nintendo.
