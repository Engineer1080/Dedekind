@echo off
REM Dedekind Studio .exe bauen (PyInstaller)
REM Vorher: pip install -r requirements.txt, pip install -r DedekindStudio\requirements-dedekind-studio.txt, pip install pyinstaller, pip install -e DedekindStudio

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%~dp0.."
cd /d "%SCRIPT_DIR%"

python -c "import spyder" 2>nul
if errorlevel 1 (
    echo Spyder nicht gefunden. Bitte aus Projektroot ausfuehren:
    echo   pip install -e DedekindStudio
    pause
    exit /b 1
)

python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo PyInstaller nicht installiert. Bitte: pip install pyinstaller
    pause
    exit /b 1
)

echo Baue Dedekind Studio .exe ...
pyinstaller build_exe.spec
if errorlevel 1 (
    echo Build fehlgeschlagen.
    pause
    exit /b 1
)

echo.
echo Fertig. Ausgabe: dist\Dedekind Studio\Dedekind Studio.exe
echo Den gesamten Ordner "dist\Dedekind Studio" weitergeben.
pause
