# Dedekind Studio als .exe bauen (Windows)

**Ziel:** Ein **fertiges Programm** bauen, das du weitergeben kannst – **ohne Quellcode** zu versenden. Der Empfänger bekommt nur die .exe (bzw. den Programmordner), keine .py-, .ddk- oder Repo-Dateien.

---

## Voraussetzungen

- **Windows** (getestet unter Windows 10/11)
- **Python 3.10+** (nur zum Bauen, nicht für den Freund)
- Aus dem **Projektroot** (Dedekind-Repo): alle Abhängigkeiten installiert

---

## Schritte

### 1. Virtuelle Umgebung (empfohlen)

```batch
cd C:\Pfad\zu\Dedekind
python -m venv venv_build
venv_build\Scripts\activate
```

### 2. Abhängigkeiten installieren

```batch
pip install -r requirements.txt
pip install -r DedekindStudio\requirements-dedekind-studio.txt
pip install pyinstaller>=6.0
```

Dedekind Studio (Spyder) **editable** installieren, damit PyInstaller Spyder findet:

```batch
pip install -e DedekindStudio
```

Falls `external-deps` (spyder-kernels, qtconsole, python-lsp-server) fehlen, aus dem Repo installieren:

```batch
cd DedekindStudio\external-deps\spyder-kernels
pip install -e .
cd ..\qtconsole
pip install -e .
cd ..\python-lsp-server
pip install -e .
cd ..\..
```

### 3. .exe bauen

**Wichtig:** PyInstaller muss die Spec-Datei finden. Entweder:

```batch
cd DedekindStudio
pyinstaller --noconfirm build_exe.spec
```

oder aus dem Projektroot:

```batch
pyinstaller --noconfirm DedekindStudio\build_exe.spec
```

`--noconfirm` überschreibt den Ausgabeordner ohne Nachfrage.

Alternativ aus dem Projektroot: `DedekindStudio\build_exe.bat` starten (wechselt selbst in DedekindStudio).

Das erzeugt den Ordner **`dist/Dedekind Studio/`** mit:

- **`Dedekind Studio.exe`** – Doppelklick startet die IDE
- allen benötigten DLLs und Python-Bibliotheken

### 4. Weitergeben – nur das Programm, kein Code

Du versendest **nur** den Ordner **`dist/Dedekind Studio`** (z. B. als ZIP). Darin sind nur die .exe und die Laufzeit-Dateien – **kein Quellcode** (kein Repo, keine .py-, keine .ddk-Quelldateien).

Dein Freund:

1. ZIP entpacken
2. **`Dedekind Studio.exe`** starten
3. Neue .ddk-Dateien anlegen oder öffnen und mit **Run / F5** ausführen

**Hinweis:** Die **Dedekind-Konsole** (Jupyter-Kernel) startet im gebündelten Build ggf. nicht, weil der Kernel normalerweise als separates Python-Modul gestartet wird. Editor, Syntax-Highlighting und **„Run .ddk“** (Kompilieren + Ausführen der aktuellen Datei) funktionieren.

---

## Optionen in `build_exe.spec`

- **`console=False`** – kein schwarzes Konsolenfenster (reine GUI). Zum Debuggen auf `True` setzen.
- **`icon=...`** – Pfad zu einer `.ico`-Datei für das Fenster-Icon (z. B. `DedekindStudio\spyder\images\...` falls vorhanden).

---

## Fehlerbehebung

- **„ModuleNotFoundError: spyder“** – Spyder nicht gefunden. Sicherstellen, dass `pip install -e DedekindStudio` ausgeführt wurde.
- **„No module named 'src.compiler'“** beim Ausführen einer .ddk-Datei – Die Daten `src` und `dedekind_jupyter_kernel` werden per `datas` ins Bundle übernommen; bei abweichender Ordnerstruktur `build_exe.spec` anpassen.
- **PyInstaller warnt vor fehlenden Modulen** – Diese ggf. in `hiddenimports` in der Spec-Datei ergänzen.

---

*Stand: v1.7.0*
