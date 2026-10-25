# FE3H Recomp

A static recompilation of *Fire Emblem: Three Houses* (Nintendo Switch, 2019) into a native Windows application,
built on [LAS Recomp](https://github.com/peiikai/LASRecomp) by Peiikai.

> [!IMPORTANT]
> **This is a personal project, and I made it mostly for myself.** I wanted to play *Three Houses* on my PC, and so
> far emulators haven't given me the experience I'm after — at least not without my CPU and GPU fans sounding like a
> jet engine. So I built this instead, with a lot of help from an AI (Anthropic's Claude wrote most of the
> FE3H-specific code; I directed and tested).
>
> I'm doing this casually, in my spare time, and I'm not planning to pour a ton of time, energy or resources into it.
> Honestly, I doubt I'll finish a full playthrough before GTA VI comes out, and once that happens this will probably
> fall by the wayside for a while. It's shared as-is: no support, no releases, no promises. Run it at your own
> discretion.

> [!NOTE]
> **No game files are included**, and no instructions for obtaining them are provided. You need your own legally
> obtained copy of the game, version **1.2.0**, extracted to `exefs` and `romfs` folders (and optionally your DLC).

## What this is

Static recompilation translates the game's original Switch (AArch64) code into C ahead of time, then compiles it into
a native program. It is not emulation: no CPU emulator runs the game. The Switch system software the game talks to
(graphics, audio, files, controllers and so on) is reimplemented in the runtime, and graphics go through Vulkan.

LAS Recomp did the heavy lifting: its recompiler, runtime and Vulkan renderer were built for *The Legend of Zelda:
Link's Awakening*. This fork adds what *Three Houses* needs on top.

## Status

Playable. Tested on Windows with an AMD Radeon RX 9070: the opening chapters, battles, the monastery, cutscenes, the
calendar, and the start of Cindered Shadows. Longer playthroughs are ongoing. Other platforms LAS supports (macOS,
Linux) may build but haven't been tried with this game.

Added for *Three Houses*:
- **Audio:** splitter routing (music, voices and effects reach the right mixes), 8 / 24 / 32-bit and float PCM,
  I3DL2 room reverb, and the game's own effect send / return.
- **Movies:** the nnMedia player the game uses for its cutscenes, decoded with FFmpeg.
- **Lighting:** clustered lights work: buffer textures in the shader translator, lane IDs and masks, compute
  barriers that are safe inside branches, 32-lane compute subgroups, and buffer clears.
- **System services:** an always-offline network service (the calendar crashed without it).
- **DLC:** the Expansion Pass, Cindered Shadows and the bonus waves, from your own DLC files.
- **Cheats:** Atmosphere-format cheat codes from a `cheats.txt` of your own, switched on and off in the menu's Game
  tab, including the ones that patch the game's code (see [Cheats](#cheats) below).

The start screen and the in-game menu (F1) are FE3H's own: Video, Audio, Controls and Game settings in slate
and gold. LAS's Link's Awakening features (its randomizer, cheats, cosmetics, warps) are still in the code,
switched off and out of the menus.

The start screen shows your own picture if you put one in the game files folder as `extracted/background.png` (or
`.jpg`); it isn't part of this repository, and `.gitignore` keeps it out.

Known gaps: no packaged release.

## Building and running (Windows)

Requirements: Windows 10+ x86-64, a CPU with AVX2 and a GPU with Vulkan 1.3.

1. **One-time setup**, in PowerShell from the repo root:
   `powershell -ExecutionPolicy Bypass -File tools\setup_windows.ps1`
   (installs MSYS2 with clang, SDL3, FFmpeg, Vulkan and shaderc, plus Python for the tools). More detail in
   [docs/BUILD-windows.md](docs/BUILD-windows.md), which LAS wrote for Link's Awakening; the steps are the same.
2. **Game files:** put your extracted `exefs/` and `romfs/` in `extracted/` and set them up on the start screen as
   in LAS (its game-file setup builds `extracted/romfs.bin`).
3. **Build**, from an MSYS2 UCRT64 shell:
   ```
   make gen    # once: recompile the game code from extracted/exefs into generated/
   make
   ```
4. **Run:** `./build/fe3hrecomp.exe extracted --play > run.log 2>&1`

### DLC

Extract each DLC's romfs into a folder named by its content id:
```
extracted/dlc/010055D009F79001/romfs/...
extracted/dlc/010055D009F79002/romfs/...
```
The game sees every pack there as installed. The first launch converts each folder into a single image next to it
(once; it takes extra disk space about the size of the DLC).

### Cheats

Cheats use the Atmosphere format: put them in `cheats.txt` in the project folder (a `[Name]` line, then its codes; a
`{Name}` block is a master code, always on; `#` lines under the name describe it), and they show up in the menu's **Game** tab to switch on and off. The
file stays out of git.

Cheats that only change memory work straight away. Cheats that patch the game's code (most multipliers and "max"
cheats) can't be applied to an already-recompiled game at run time, so their patched instructions get compiled in
beside the originals: after adding or changing those, run `make gen-main` and then `make`. The menu marks any cheat
still waiting on that. Switching a cheat off puts back any code it changed; values it already wrote stay. Cheats
that patch the same instruction (Exp x2 / x4 / x8, say) exclude each other: switching one on turns the others off.

### Settings and logs

- **F1** (or Minus + Plus on a controller) opens the settings overlay. For a sharp image use Internal resolution
  "Match display" or higher, with SMAA.
- Saves are in `save/`, settings in `fe3hrecomp.ini` (an older `larecomp.ini` is renamed to it once).
- `run.log` is the first place to look when something goes wrong. `LARECOMP_*` environment variables turn on extra
  logging for specific systems (audio, textures, shaders); they're documented where they're read in `runtime/`.

## Rules for this repository

Nothing owned by Nintendo is ever committed: no game files, no extracted or generated code (`extracted/`,
`generated/`), no logs or dumps that contain game code, no assets. `.gitignore` covers the usual folders.

## Credits

- **[LAS Recomp](https://github.com/peiikai/LASRecomp)** by Peiikai and its contributors: the recompiler, runtime,
  renderer and almost everything else here (MIT). Its README lists its full credits and third-party components.
- **[Ryujinx](https://ryujinx.app)** (MIT): the audio renderer's reverb, I3DL2 reverb and auxiliary send / return
  are ports of its implementation.
- **[EB Garamond](https://github.com/georgd/EB-Garamond)** by Georg Duffner: the start screen's title (SIL Open Font License
  1.1, `third_party/fonts/EBGaramond/OFL.txt`).
- **[AMD FidelityFX Super Resolution 1](https://github.com/GPUOpen-Effects/FidelityFX-FSR)** (MIT) and
  **[SMAA](https://github.com/iryoku/smaa)** (MIT), as used by LAS.
- **Libraries:** [SDL 3](https://libsdl.org) (zlib), [FFmpeg](https://ffmpeg.org) (LGPL),
  [Dear ImGui](https://github.com/ocornut/imgui) (MIT),
  [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) (MIT),
  [shaderc](https://github.com/google/shaderc) (Apache 2.0); the code generator uses
  [Capstone](https://www.capstone-engine.org), [Unicorn](https://www.unicorn-engine.org) and
  [python-lz4](https://github.com/python-lz4/python-lz4).
- The Switch homebrew community's documentation ([Switchbrew](https://switchbrew.org)).

## Legal

This is a fan project, not affiliated with, endorsed by or sponsored by Nintendo, Intelligent Systems or Koei Tecmo.
*Fire Emblem* and *Fire Emblem: Three Houses* are trademarks of Nintendo. No game files, game code or assets are
included in this repository. You must provide your own legally obtained copy of the game.

## License

The code is open source under the [MIT License](LICENSE), as LAS Recomp's is. Third-party components keep their own
licenses (see LAS Recomp's README for the full list, including the Link's Awakening randomizer still in
`third_party/`). The license covers this project's code only; it grants nothing for Nintendo's game.
