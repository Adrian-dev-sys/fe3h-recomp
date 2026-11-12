@echo off
rem RE run: on a battle map with unit icons showing, press F9 once, keep the camera still ~15 s, close the game. Writes run.log.
cd /d "%~dp0..\.."
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
set LARECOMP_DEBUG_FRAMES=1
set LARECOMP_DEBUG_DATA=1
set LARECOMP_FIND_ICONS=1
del /q build\pass_*.png 2>nul
build\fe3hrecomp.exe %* > run.log 2>&1
