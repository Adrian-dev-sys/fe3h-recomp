@echo off
rem RE run: on a battle map (Fill screen, windowed is easiest) press F10 to keep the game's 16:9 aspect value
rem unchanged (title shows ON/off); press again to undo. Watch the unit icons and the 3D view each way.
cd /d "%~dp0..\.."
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
set LARECOMP_LIT_TOGGLE=1
build\fe3hrecomp.exe %* > run.log 2>&1
