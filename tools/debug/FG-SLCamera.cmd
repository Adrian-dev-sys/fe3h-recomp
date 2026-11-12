@echo off
rem Frame gen test: character motion vectors on, camera motion by Streamline instead of our depth pass.
cd /d "%~dp0..\.."
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
set LARECOMP_MV_SL_CAMERA=1
build\fe3hrecomp.exe %* > run.log 2>&1
