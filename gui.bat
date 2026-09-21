@echo off
setlocal
cd /d "%~dp0"
set "VIRTUAL_ENV=%~dp0venv"
set "PYTHONHOME="
set "PYTHONPATH="
set "PATH=%~dp0venv\Scripts;%~dp0venv\Lib\site-packages\torch\lib;%PATH%"
"%~dp0venv\Scripts\python.exe" "%~dp0kohya_gui.py" --noverify %*
exit /b %errorlevel%
