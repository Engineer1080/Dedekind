# Dedekind Jupyter Kernel

Jupyter-Kernel für die Sprache **Dedekind**. Erlaubt die Ausführung von Dedekind-Code in Jupyter Notebooks, JupyterLab und **Spyder** (IPython Console → „Connect to an existing kernel“ oder nach Integration als Standard-Kernel).

## Abhängigkeiten

- Python 3.10+
- Dedekind-Compiler (dieses Repo, `src.compiler`)
- `ipykernel` (z. B. `pip install ipykernel` oder aus dem Repo-Root: `pip install -r requirements.txt`)

## Installation (aus DedekindLanguage-Repo-Root)

```bash
# Repo-Root: DedekindLanguage/ (oder DedekindLanguage/)
pip install ipykernel
cd dedekind_jupyter_kernel
jupyter kernelspec install kernelspec --user --name dedekind
```

Damit wird der Kernel unter dem Namen „Dedekind“ registriert. In Spyder: **Consoles** → **Connect to an existing kernel** → Kernel „Dedekind“ wählen (falls Spyder Jupyter-Kernel auflistet) oder Verbindungsdatei angeben.

## Nutzung in Spyder / Dedekind Studio

- **Spyder:** Neue Konsole starten und als Kernel „Dedekind“ auswählen (sofern der Dedekind-Kernel in Spyder sichtbar ist), oder „Connect to an existing kernel“ und einen laufenden Dedekind-Kernel verbinden.
- **Jupyter:** Beim Erstellen eines neuen Notebooks „Dedekind“ als Kernel wählen.

Der Kernel kompiliert jede Zelle mit dem Dedekind-Compiler und führt den generierten Python-Code in einem **persistenten Kontext** aus (Variablen bleiben über Zellen hinweg erhalten).
