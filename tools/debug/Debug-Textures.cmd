@echo off
rem Texture debug: F9 logs every draw's samplers and saves the main environment material's textures (build\tex_*.png).
cd /d "%~dp0..\.."
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
set LARECOMP_DUMP_TEX=bb0d7676c3a9cd19
del /q build\pass_*.png build\tex_*.png 2>nul
build\fe3hrecomp.exe %* > run.log 2>&1
