@echo off
rem Performance profile: GPU time per pass every 10 s ("[gpuprof]" in run.log). Play normally for a minute, then close.
cd /d "%~dp0..\.."
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
set LARECOMP_GPUPROF_AT=every:600
set LARECOMP_GPUPROF_DRAWS=1
build\fe3hrecomp.exe %* > run.log 2>&1
