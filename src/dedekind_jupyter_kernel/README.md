# Dedekind Jupyter Kernel

Jupyter kernel for the **Dedekind** language. Lets you run `.ddk` code in
Jupyter Notebooks, JupyterLab, and Spyder.

## Install

```bash
pip install "dedekind[jupyter]"
python -m dedekind.install_kernel
```

That's it. Start `jupyter notebook` / `jupyter lab` / Spyder and pick
**Dedekind** from the kernel list.

## What this gives you

- Each cell is compiled by the Dedekind compiler and executed in a
  **persistent** Python interpreter — variables carry across cells.
- LaTeX output via `print_latex(...)` renders in the rich console.
- Errors are mapped back to `.ddk` line numbers.

## Editable / dev install

If you're hacking on the language itself:

```bash
git clone <repo>
cd Dedekind
pip install -e ".[jupyter]"
python -m dedekind.install_kernel
```

The kernel will pick up `src/dedekind/` automatically.
