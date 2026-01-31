# Fourier Jupyter Kernel

Jupyter-Kernel für die Sprache **Fourier**. Erlaubt die Ausführung von Fourier-Code in Jupyter Notebooks, JupyterLab und **Spyder** (IPython Console → „Connect to an existing kernel“ oder nach Integration als Standard-Kernel).

## Abhängigkeiten

- Python 3.10+
- Fourier-Compiler (dieses Repo, `src.compiler`)
- `ipykernel` (z. B. `pip install ipykernel` oder aus dem Repo-Root: `pip install -r requirements.txt`)

## Installation (aus FourierLanguage-Repo-Root)

```bash
# Repo-Root: FourierLanguage/
pip install ipykernel
cd fourier_jupyter_kernel
jupyter kernelspec install kernelspec --user --name fourier
```

Damit wird der Kernel unter dem Namen „Fourier“ registriert. In Spyder: **Consoles** → **Connect to an existing kernel** → Kernel „Fourier“ wählen (falls Spyder Jupyter-Kernel auflistet) oder Verbindungsdatei angeben.

## Nutzung in Spyder / Fourier Studio

- **Spyder:** Neue Konsole starten und als Kernel „Fourier“ auswählen (sofern der Fourier-Kernel in Spyder sichtbar ist), oder „Connect to an existing kernel“ und einen laufenden Fourier-Kernel verbinden.
- **Jupyter:** Beim Erstellen eines neuen Notebooks „Fourier“ als Kernel wählen.

Der Kernel kompiliert jede Zelle mit dem Fourier-Compiler und führt den generierten Python-Code in einem **persistenten Kontext** aus (Variablen bleiben über Zellen hinweg erhalten).
