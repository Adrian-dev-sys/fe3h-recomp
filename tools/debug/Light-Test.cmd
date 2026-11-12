@echo off
rem Light flicker test (bisect step): only the light lists read from the guest heap again (old bfetch path).
cd /d "%~dp0..\.."
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
set LARECOMP_VK_BFETCH_HEAP=1
build\fe3hrecomp.exe %* > run.log 2>&1
