# Konsole: Wissenschaftliche Zeichen und KaTeX/Web

**Stand:** Kurzüberblick, wie die Konsole wissenschaftliche Formeln anzeigt und wie eine native KaTeX-Integration aussehen könnte.

---

## Aktuell (v1.0)

- **Konsole (stdout):** `print_latex(s)` sendet eine **Unicode-Version** der Formel in die Konsole. Griechische Buchstaben (α, Δ, π), Operatoren (∫, ∑, √, ∞), Brüche (½, ⅓) und Hoch-/Tiefstellen (⁰¹², ₀₁₂) werden aus LaTeX konvertiert und als **UTF-8-Text** angezeigt. Die Konsole muss eine Schriftart mit mathematischen Zeichen nutzen (z. B. DejaVu Sans, Segoe UI). **Keine Bilder** – `print_latex` öffnet keine Plots und sendet keine PNG-Ausgabe.

---

## Zukünftig: KaTeX/Web für native Formel-Darstellung

**Ziel:** Die Konsole soll LaTeX-Formeln **direkt** rendern (wie auf Websites mit KaTeX), nicht nur als Unicode-Text oder als separates Bild.

**Optionen:**

1. **Web-basierte Konsole (z. B. Qt WebEngine):**  
   Die Konsole-Ausgabe läuft in einem HTML/JavaScript-Frontend (z. B. QWebEngineView). Kernel sendet `text/latex`; das Frontend rendert mit **KaTeX** (npm: `katex`) und zeigt die Formel inline. Vorteil: volle LaTeX-Unterstützung wie im Browser. Nachteil: Dedekind Studio müsste die Konsole auf Web-Technik umstellen oder eine Web-View für Ausgabe nutzen.

2. **JupyterLab / Jupyter Notebook:**  
   Dort wird `text/latex` bereits mit MathJax/KaTeX im Browser gerendert. Wer Dedekind in JupyterLab nutzt, hat damit schon native Formel-Darstellung in der „Konsole“ (Zellen-Ausgabe).

3. **Qt-basierte Konsole mit eingebettetem Renderer:**  
   Statt Web: Ein Qt-Widget, das LaTeX zu Pixmap/SVG rendert (z. B. über eine kleine LaTeX-Render-Bibliothek oder ein eingebettetes KaTeX-ähnliches Modul). Aufwand höher; plattformunabhängig ohne Browser-Engine.

**Referenz:** KaTeX (https://katex.org/) – schneller LaTeX-Renderer für das Web, npm-Paket `katex`. Für Dedekind Studio wäre eine Integration nur sinnvoll, wenn die Konsole oder ein Teil davon auf HTML/JS (z. B. WebEngine) umgestellt wird.

---

*Dieses Dokument dient als Ideensammlung für spätere Erweiterungen; die aktuelle Unicode-Konsole und das Plot-Bild decken den Alltagsbedarf ab.*
