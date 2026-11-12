@echo off
rem RE run: on a battle map, press F10 repeatedly; each press un-widens one camera aspect read (21 sites, then off).
rem The window title shows the site. Note which number makes the unit icons sit on their units. Writes run.log.
cd /d "%~dp0..\.."
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
set LARECOMP_ASPECT_BISECT=1
build\fe3hrecomp.exe %* > run.log 2>&1
