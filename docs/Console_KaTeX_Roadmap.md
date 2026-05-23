# Console: Scientific Characters and KaTeX/Web

**Status:** Brief overview of how the console displays scientific formulas and what a native KaTeX integration could look like.

---

## Current (v2.9.0)

- **Console (stdout):** `print_latex(s)` sends a **Unicode version** of the formula to the console. Greek letters (α, Δ, π), operators (∫, ∑, √, ∞), fractions (½, ⅓) and superscripts/subscripts (⁰¹², ₀₁₂) are converted from LaTeX and shown as **UTF-8 text**. The console must use a font with mathematical characters (e.g. DejaVu Sans, Segoe UI). **No images** — `print_latex` does not open plots and does not emit PNG output.

---

## Future: KaTeX/Web for native formula rendering

**Goal:** The console should render LaTeX formulas **directly** (as on websites using KaTeX), not just as Unicode text or as a separate image.

**Options:**

1. **Web-based console (e.g. Qt WebEngine):**  
   The console output runs in an HTML/JavaScript frontend (e.g. QWebEngineView). The kernel sends `text/latex`; the frontend renders with **KaTeX** (npm: `katex`) and displays the formula inline. Advantage: full LaTeX support as in the browser. Disadvantage: any IDE integration would need to switch the console to web technology or use a web view for output.

2. **JupyterLab / Jupyter Notebook:**  
   There `text/latex` is already rendered with MathJax/KaTeX in the browser. Anyone using Dedekind in JupyterLab already has native formula rendering in the "console" (cell output).

3. **Qt-based console with embedded renderer:**  
   Instead of web: a Qt widget that renders LaTeX to pixmap/SVG (e.g. via a small LaTeX render library or an embedded KaTeX-like module). Higher effort; platform-independent without a browser engine.

**Reference:** KaTeX (https://katex.org/) — fast LaTeX renderer for the web, npm package `katex`. For IDE integration, this would only make sense if the console or part of it is switched to HTML/JS (e.g. WebEngine).

---

*This document serves as an idea collection for later extensions; the current Unicode console and the plot image cover everyday needs.*
