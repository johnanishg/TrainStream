@echo off
REM Builds trainstream._C in place on Windows.
REM Loads the MSVC x64 toolchain and points the build at the CUDA toolkit.
setlocal

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
  for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
)
call "%VCVARS%" >nul || exit /b 1

if "%CUDA_HOME%"=="" (
  for /d %%d in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do set "CUDA_HOME=%%d"
)
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set DISTUTILS_USE_SDK=1

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

cd /d "%~dp0"
"%PY%" setup.py build_ext --inplace %*
