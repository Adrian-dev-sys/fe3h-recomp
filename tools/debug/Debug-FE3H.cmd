@echo off
rem Debug run: press F9 to dump the next 4 frames' passes (build\pass_*.png) and every draw (run.log).
rem set LARECOMP_NAN_SCAN=1 below to log which draw first writes NaN/Inf instead of dumping PNGs.
cd /d "%~dp0..\.."
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
set LARECOMP_DEBUG_FRAMES=4
set LARECOMP_DEBUG_DATA=1
del /q build\pass_*.png 2>nul
build\fe3hrecomp.exe %* > run.log 2>&1
