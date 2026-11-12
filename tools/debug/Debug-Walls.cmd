@echo off
rem Wall/texture detail run: F9 dumps one frame's passes, its draws, shader sources and every scene texture.
rem Log goes to run-walls.log (a normal launch doesn't overwrite it).
cd /d "%~dp0..\.."
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
set LARECOMP_DEBUG_FRAMES=1
set LARECOMP_DEBUG_DATA=1
set LARECOMP_SAVE_MSL=1
set LARECOMP_VK_NO_PREWARM=1
set LARECOMP_DUMP_TEX=min:2
del /q build\pass_*.png build\tex_*.png 2>nul
rmdir /s /q build\glsl 2>nul
build\fe3hrecomp.exe %* > run-walls.log 2>&1
