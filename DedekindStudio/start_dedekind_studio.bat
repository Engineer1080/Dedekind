@echo off
REM Dedekind Studio – Startskript (Doppelklick oder von Konsole aus)
REM Fokus: Dedekind-IDE; PyTorch ist fuer die Dedekind-Laufzeit zwingend.
REM Liegt im DedekindStudio-Verzeichnis, startet python bootstrap.py
REM Bei Doppelklick: Fenster bleibt mit cmd /k offen

if "%~1" neq "keepopen" (
    cmd /k "%~f0" keepopen
    exit /b 0
)

set "SPYDER_DIR=%~dp0"
set "PROJECT_ROOT=%~dp0.."
cd /d "%SPYDER_DIR%"

if not exist "bootstrap.py" (
    echo Fehler: bootstrap.py nicht gefunden.
    pause
    exit /b 1
)

REM Installation ins Benutzerverzeichnis, damit c:\Python310\ nicht beschrieben werden muss
set PIP_USER=1

REM Spyder-Abhängigkeiten (PyQt5, qtpy, chardet, ipython, …) – bei Bedarf alle auf einmal installieren
python -c "import PyQt5, chardet" 2>nul
if errorlevel 1 (
    if not exist "requirements-dedekind-studio.txt" (
        echo Fehler: requirements-dedekind-studio.txt nicht gefunden im Verzeichnis: %CD%
        pause
        exit /b 1
    )
    echo Spyder-Abhaengigkeiten fehlen. Installiere alle - kann beim ersten Mal dauern.
    python -m pip install -r "requirements-dedekind-studio.txt"
    if errorlevel 1 (
        echo.
        echo Installation fehlgeschlagen.
        pause
        exit /b 1
    )
    echo Abhaengigkeiten installiert.
)

REM Dedekind-Backend (PyTorch etc.) – essentiell fuer .ddk-Ausfuehrung und Dedekind-Kernel
python -c "import torch" 2>nul
if errorlevel 1 (
    if not exist "%PROJECT_ROOT%\requirements.txt" (
        echo Fehler: requirements.txt nicht gefunden im Projektroot: %PROJECT_ROOT%
        pause
        exit /b 1
    )
    echo PyTorch/Dedekind-Backend fehlt. Installiere aus Projektroot - kann beim ersten Mal dauern.
    python -m pip install -r "%PROJECT_ROOT%\requirements.txt"
    if errorlevel 1 (
        echo.
        echo Installation Dedekind-Backend fehlgeschlagen.
        pause
        exit /b 1
    )
    echo Dedekind-Backend ^(PyTorch, matplotlib, ipykernel^) installiert.
)

echo Starte Dedekind Studio.
python bootstrap.py
if errorlevel 1 (
    echo.
    echo Dedekind Studio ist mit einem Fehler beendet.
)
echo.
pause
