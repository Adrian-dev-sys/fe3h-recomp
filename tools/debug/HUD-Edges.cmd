@echo off
rem Ultrawide test: side HUD panels move to the window edges (unscaled). F9 dumps 4 frames to run.log (hud layout lines).
rem Optional: set LARECOMP_HUD_EDGE_SPLIT=0.3 (outer fraction of the 16:9 box that counts as a side; default 0.45).
cd /d "%~dp0..\.."
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
set LARECOMP_HUD_EDGES=1
set LARECOMP_DEBUG_FRAMES=4
set LARECOMP_DEBUG_DATA=1
del /q build\pass_*.png 2>nul
build\fe3hrecomp.exe %* > run.log 2>&1
