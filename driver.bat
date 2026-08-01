@echo off
setlocal

set "ROOT_DIR=%~dp0"
set "BUILD_DIR=%ROOT_DIR%build"
set "CONFIG=Release"

if not "%~1"=="" set "CONFIG=%~1"

if not exist "%BUILD_DIR%" (
    mkdir "%BUILD_DIR%"
)

cmake -S "%ROOT_DIR%" -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=%CONFIG%
if errorlevel 1 goto :error

cmake --build "%BUILD_DIR%" --config %CONFIG%
if errorlevel 1 goto :error

goto :eof

:error
echo Build failed.
exit /b 1
