@echo off

rem     *****************
rem     *    HEGTool.bat    *
rem     *****************
rem Change directory to HEG bin directory.HEGTOOL.bat now can be run from anywhere.
cd c:\heg\HEG_Win\bin

rem Set the MRTDATADIR environmental var to the HEG data directory.

set CYGWIN=nodosfilewarning

set LD_LIBRARY_PATH=c:\heg\HEG_Win\bin

set MRTDATADIR=c:\heg\HEG_Win\data

set MRTBINDIR=c:\heg\HEG_Win\bin

set PGSHOME=c:\heg\HEG_Win\TOOLKIT_MTD

set HEGUSER=heg

"C:\heg\HEG_Win\bin\swtif.exe" -P C:\Users\kater\Documents\test\parametros_swath