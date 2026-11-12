@echo off
rem DLAA test: F10 turns the characters' motion vectors off / on (on at start); each press is logged in run.log.
cd /d "%~dp0..\.."
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
set LARECOMP_DLAA_TUNE=1
build\fe3hrecomp.exe %* > run.log 2>&1
