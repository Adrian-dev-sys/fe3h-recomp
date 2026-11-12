@echo off
rem RE run (map icons): on a battle map in Fill screen press F10 once, then PAN THE CAMERA (right stick, or push the
rem cursor against the screen edges) for ~60 s so the icons move. The game is slow while it watches. Close the game
rem from its window. Writes run.log.
cd /d "%~dp0..\.."
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
set LARECOMP_DIFF_ICONS=vp
set LARECOMP_LOG_PROJ=1
build\fe3hrecomp.exe %* > run.log 2>&1
