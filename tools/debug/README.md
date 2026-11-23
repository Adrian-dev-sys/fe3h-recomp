# Debug launchers

Windows launchers used while working on this fork. Each one sets a few `LARECOMP_*` environment variables (see the
`rem` lines inside), starts `build\fe3hrecomp.exe` from the repo root and writes `run.log` there.

- `Debug-FE3H.cmd`: press F9 in game to dump the next 4 frames: every render pass as `build\pass_*.png` and every
  draw (state, constants, vertices) in `run.log`. Start any rendering or UI investigation with one of these.
- `Debug-Perf.cmd`: GPU time per pass / draw. `Debug-Light`, `-Motion`, `-Shadow`, `-Textures`, `-Walls`, `DLAA-Tune`,
  `FG-*`, `Jitter-Test`, `Light-Test`, `Aspect-*`, `*-Icons`, `Cutin-Capture` (auto F9 on a shader hash: `LARECOMP_DEBUG_ON_FSHASH`): narrower captures and on/off switches used for one bug each.

Captures and logs can contain game code and data: never commit them.