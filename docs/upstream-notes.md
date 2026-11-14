# FE3H Recomp: upstream notes

Written by the author of [FE3H Recomp](https://github.com/Bakanda07/fe3h-recomp), the project this fork builds on.
Kept for history and attribution; some details (the largest battle shaders being skipped) have changed since.

Some of what it took to get from Link's Awakening's recompiler to a playable *Three Houses*, in case it helps anyone
doing the same for another game:

- **Lighting was missing.** FE3H culls its lights in compute shaders that count lights per screen tile with subgroup
  operations. The translator returned 0 for lane IDs and masks, so no tile had any lights; and a barrier inside the
  shader's branch state machine ran before every wave had added its own lights. Fixed in the shader translator.
- **Audio routing.** Music, voices and effects go through the audio renderer's splitters and an I3DL2 reverb that
  LA never used; until those were implemented, sounds didn't reach the right mixes.
- **Stutter.** Each new shader was translated, compiled to SPIR-V and built by the driver in the middle of play.
  Three layers fixed it: translations cached on disk, draws waiting a few milliseconds a frame for a shader still
  building instead of skipping it, and a prewarm that rebuilds everything seen in earlier sessions at launch.
- **Battle effects.** Combat art and gambit cut-ins are drawn by compute shaders writing into images, with
  subroutine calls and memory reductions the translator didn't know yet. Some are enormous (9,000-11,000
  instructions); building them stalled the driver for minutes and held up every other shader behind them, so the
  effects never appeared. Large ones now build on threads of their own, and the largest are skipped.
- **Cheats.** Many Switch cheats patch the game's code, which a recompiled game no longer runs. Their patched
  instructions are compiled in beside the originals and picked at run time. One popular "always critical" cheat
  turned out to zero unit stats (stranding units unable to move); it was replaced with a one-instruction code patch
  on the crit roll.
- **Setup time.** A few of the game's functions are huge (one is 4 MB of C). A compiler can't split one function
  across cores, so the first setup spent 15 minutes compiling a single file. The generator now writes very long
  functions as parts that compile in parallel: setup went from about 20 minutes to 6.

