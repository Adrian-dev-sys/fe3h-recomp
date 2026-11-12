@echo off
rem Lighting debug: F9 saves the lighting pass's inputs (build\tex_*.png); F10 cycles bloom 1 / 0.5 / 0 (logged).
cd /d "%~dp0..\.."
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
set LARECOMP_DUMP_TEX=min:10
set LARECOMP_BLOOM_TUNE=1
del /q build\pass_*.png build\tex_*.png 2>nul
build\fe3hrecomp.exe %* > run.log 2>&1
