@echo off
rem Shadow debug run: F9 dumps one frame with its uniform data (run.log); shader sources go to build\glsl\.
cd /d "%~dp0..\.."
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
set LARECOMP_DEBUG_FRAMES=1
set LARECOMP_DEBUG_DATA=1
set LARECOMP_SAVE_MSL=1
set LARECOMP_VK_NO_PREWARM=1
del /q build\pass_*.png 2>nul
rmdir /s /q build\glsl 2>nul
build\fe3hrecomp.exe %* > run.log 2>&1
