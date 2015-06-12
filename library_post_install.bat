@ECHO OFF

REM creates directory hierarchy

IF EXIST C:\install GOTO CREATE_LIB
MKDIR C:\install
:CREATE_LIB
IF EXIST C:\install\lib GOTO CREATE_INCLUDE
MKDIR C:\install\lib
:CREATE_INCLUDE
IF EXIST C:\install\include GOTO COPY_LIBRARY
MKDIR C:\install\include

:COPY_LIBRARY

REM copy headers and library file to install directories

IF EXIST C:\install\include\dinrhiw GOTO CREATE_DINRHIW_DONE
MKDIR C:\install\include\dinrhiw

:CREATE_DINRHIW_DONE

COPY libdinrhiw.a C:\install\lib\

copy ..\config.h c:\install\include\dinrhiw\
FOR /R ..\src %%f in (*.h) do copy %%f c:\install\include\dinrhiw\
FOR /R ..\src %%f in (*.cpp) do copy %%f c:\install\include\dinrhiw\



