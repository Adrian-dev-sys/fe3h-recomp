@echo off
rem Ultrawide gambit / crit cut-in: dumps 6 frames (build\pass_*.png, run.log) the first time the portrait panel draws.
cd /d "%~dp0..\.."
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
set LARECOMP_DEBUG_ON_FSHASH=dedd5e1fdb9cb083
set LARECOMP_DEBUG_FRAMES=6
set LARECOMP_DEBUG_DATA=1
del /q build\pass_*.png 2>nul
build\fe3hrecomp.exe %* > run.log 2>&1
