@echo off
rem Frame gen test: no character motion vectors, camera motion by Streamline (as before they were added).
cd /d "%~dp0..\.."
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
set LARECOMP_NO_OBJECT_MV=1
set LARECOMP_MV_SL_CAMERA=1
build\fe3hrecomp.exe %* > run.log 2>&1
