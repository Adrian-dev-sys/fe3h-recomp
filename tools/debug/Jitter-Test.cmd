@echo off
rem DLAA jitter test: F10 cycles the jitter convention (0..3 signs, 4 no jitter); each press is logged in run.log.
cd /d "%~dp0..\.."
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
set LARECOMP_JITTER_TEST=1
build\fe3hrecomp.exe %* > run.log 2>&1
