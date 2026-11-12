@echo off
rem Motion vector capture: keep walking (camera following) and press F9; saves build\dlaa_motion*.png.
cd /d "%~dp0..\.."
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
set LARECOMP_DEBUG_FRAMES=1
del /q build\dlaa_motion*.png 2>nul
build\fe3hrecomp.exe %* > run.log 2>&1
