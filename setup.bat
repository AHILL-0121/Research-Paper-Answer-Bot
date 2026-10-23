@echo off
rem Set up the Research Paper Answer Bot. All logic is in backend\scripts\setup.py;
rem run "setup.bat --help" for the options.
setlocal
cd /d "%~dp0"

set "PY="
py -3.12 -c "import sys" >nul 2>nul && set "PY=py -3.12"
if not defined PY (
    python -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo Python 3.12 was not found. Install it from https://www.python.org/downloads/ and rerun setup.bat.
    exit /b 1
)

%PY% backend\scripts\setup.py %*
exit /b %errorlevel%
