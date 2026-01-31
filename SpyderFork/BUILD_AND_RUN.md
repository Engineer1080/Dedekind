# Spyder (Dedekind Studio) bauen und ausführen

Spyder ist eine **Python-Anwendung** – es gibt keine klassische „Kompilation“ wie bei C++. Ablauf: Umgebung einrichten, Abhängigkeiten installieren, Spyder im Entwicklungsmodus starten.

## Voraussetzungen

- **Python 3.9+**
- **Conda** (empfohlen) oder **pip** + **venv**

---

## Option A: Mit Conda (empfohlen, besonders unter Windows)

Aus dem **DedekindLanguage**-Repo-Root (oder direkt in `SpyderFork/`):

```bash
# 1. Conda-Umgebung anlegen und aktivieren
conda create -n spyder-dev -c conda-forge python=3.9
conda activate spyder-dev

# 2. Abhängigkeiten aus Spyder-requirements installieren
cd SpyderFork
conda env update --file requirements/main.yml

# Unter Windows ggf. zusätzlich:
conda env update --file requirements/windows.yml

# 3. Spyder im Entwicklungsmodus installieren (editierbar)
pip install -e .

# 4. Spyder starten
python bootstrap.py
```

**Debug-Modus:** `python bootstrap.py --debug`  
**Spyder-Optionen:** `python bootstrap.py -- --help`

---

## Option B: Mit pip und venv

```bash
# 1. Virtuelle Umgebung anlegen und aktivieren
cd SpyderFork
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# 2. Spyder inkl. Abhängigkeiten im Entwicklungsmodus installieren
pip install -e .

# 3. Spyder starten
python bootstrap.py
```

**Hinweis:** Bei reinem pip können je nach System noch Qt/PyQt fehlen; ggf. vorher z. B. `pip install PyQt5` (oder PyQt6) installieren. Conda bringt Qt in der Regel mit.

---

## Kurzfassung

| Schritt | Befehl (in SpyderFork/) |
|--------|--------------------------|
| Umgebung (Conda) | `conda create -n spyder-dev -c conda-forge python=3.9` → `conda activate spyder-dev` |
| Dependencies (Conda) | `conda env update --file requirements/main.yml` |
| Spyder installieren | `pip install -e .` |
| **Starten** | **`python bootstrap.py`** |

Nach dem ersten `pip install -e .` reicht zum Starten jeweils `python bootstrap.py` (aus `SpyderFork/`). Für Dedekind Studio: Nach euren Änderungen (Branding, Dedekind-Kernel) ebenfalls `python bootstrap.py` aus diesem Ordner ausführen.
