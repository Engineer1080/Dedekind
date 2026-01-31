# Fourier Studio – Spyder-Fork

Dieser Ordner enthält einen **Fork von [Spyder](https://github.com/spyder-ide/spyder)** für **Fourier Studio**. Fourier Studio bietet nativ **Python** und **Fourier** in einer IDE (siehe [Fourier_Studio_Spyder_Fork.md](../Documentation/Fourier_Studio_Spyder_Fork.md) im Hauptrepo).

## Upstream

- **Original:** https://github.com/spyder-ide/spyder  
- **Lizenz:** MIT (unverändert)

## Nächste Schritte (Fourier Studio)

1. **Branding:** Name, Icons und Fenstertitel auf „Fourier Studio“ umstellen.
2. **Fourier-Kernel:** Sicherstellen, dass der Fourier Jupyter Kernel (`fourier_jupyter_kernel/` im Hauptrepo) beim Start einer neuen Konsole wählbar ist.
3. **Optional:** Standard-Kernel für neue Konsolen auf „Fourier“ setzbar machen; `.fourier`-Editor mit Run und Syntax-Highlighting.

## Bauen und Ausführen

Spyder ist eine Python-Anwendung (keine klassische Kompilierung). Aus dem Ordner **SpyderFork/**:

1. **Conda-Umgebung** (empfohlen):  
   `conda create -n spyder-dev -c conda-forge python=3.9` → `conda activate spyder-dev`  
   `conda env update --file requirements/main.yml`  
   (Windows ggf. zusätzlich: `conda env update --file requirements/windows.yml`)

2. **Spyder im Entwicklungsmodus installieren:**  
   `pip install -e .`

3. **Starten:**  
   `python bootstrap.py`

Details und Alternative mit pip/venv: siehe **[BUILD_AND_RUN.md](BUILD_AND_RUN.md)** in diesem Ordner.

## Upstream aktualisieren

Falls ihr wieder ein separates Spyder-Clone als Remote nutzt:

```bash
cd SpyderFork
git fetch origin
git merge origin/master
```
