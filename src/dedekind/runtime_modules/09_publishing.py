def _notebook_in_progress_set():
    """Process-weite Re-Entry-Tracking-Menge. Liegt auf sys, damit sie über exec-Kontexte
    persistent ist (jeder exec(compile_source(...)) bekommt sonst eine frische ml_runtime-Kopie)."""
    import sys as _sys
    key = "_dedekind_notebook_export_in_progress"
    s = getattr(_sys, key, None)
    if s is None:
        s = set()
        setattr(_sys, key, s)
    return s


def export_notebook(source_path, output_path=None, format="html", title=None,
                    include_hash=True, capture_plots=True):
    """Führt eine .ddk-Datei aus und schreibt eine Standalone-Datei (HTML oder Markdown),
    die Quellcode, Stdout-Ausgabe, Plots (Base64-PNG) und einen SHA-256-Hash des Quelltexts bündelt.

    Parameter:
      source_path: Pfad zur .ddk-Datei.
      output_path: Zieldatei (default: `<source>.html` bzw. `.md`).
      format: "html" oder "md".
      title: Optionaler Titel; default = Dateiname ohne Endung.
      include_hash: Fügt SHA-256-Hash des Quelltexts in den Output ein (Reproduzierbarkeit).
      capture_plots: Sammelt `_dedekind_plots` und bettet sie als Base64-PNG ein.

    Rückgabe: Pfad zur erzeugten Datei.
    """
    import os
    import sys
    import io
    import hashlib
    src_path = os.path.abspath(str(source_path))
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"export_notebook: Datei nicht gefunden: {src_path}")
    # Re-Entry-Schutz: wenn die Quelldatei sich selbst exportiert, würde sie sich beim Ausführen
    # endlos wiederaufrufen. Wir markieren den Pfad und liefern beim re-entry einen Stub zurück.
    in_progress = _notebook_in_progress_set()
    if src_path in in_progress:
        return output_path or (os.path.splitext(src_path)[0] +
                               (".html" if str(format).lower() == "html" else ".md"))
    in_progress.add(src_path)
    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()
    fmt = str(format).lower()
    if fmt not in ("html", "md", "markdown"):
        raise ValueError("export_notebook: format muss 'html' oder 'md' sein.")
    if output_path is None:
        ext = "html" if fmt == "html" else "md"
        output_path = os.path.splitext(src_path)[0] + f".{ext}"
    title_str = str(title) if title else os.path.basename(os.path.splitext(src_path)[0])
    src_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()

    # Quelltext kompilieren und in isoliertem Globals-Dict ausführen; Stdout abfangen.
    # Wir importieren lokal, um Zirkular-Imports beim Inlinen zu vermeiden.
    try:
        from dedekind import compile_source  # type: ignore[import-not-found]
        py_code = compile_source(source, filepath=src_path)
        old_stdout = sys.stdout
        captured = io.StringIO()
        exec_globals = {}
        try:
            sys.stdout = captured
            exec(py_code, exec_globals)
        finally:
            sys.stdout = old_stdout
        stdout_text = captured.getvalue()
        plots_b64 = []
        if capture_plots:
            plots_b64 = list(exec_globals.get("_dedekind_plots", []))

        if fmt == "html":
            content = _render_notebook_html(title_str, source, stdout_text, plots_b64,
                                            src_hash if include_hash else None,
                                            os.path.basename(src_path))
        else:
            content = _render_notebook_markdown(title_str, source, stdout_text, plots_b64,
                                                src_hash if include_hash else None,
                                                os.path.basename(src_path))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        return output_path
    finally:
        in_progress.discard(src_path)


def _render_notebook_html(title, source, stdout_text, plots_b64, src_hash, src_name):
    import html as _html
    import datetime
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{_html.escape(title)}</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "max-width:900px;margin:2em auto;padding:0 1em;color:#222;}",
        "h1,h2{border-bottom:1px solid #ddd;padding-bottom:.2em;}",
        "pre{background:#f6f8fa;padding:1em;border-radius:6px;overflow-x:auto;"
        "font-family:'SF Mono',Consolas,monospace;font-size:.9em;}",
        ".meta{color:#666;font-size:.85em;margin:.5em 0;}",
        ".plot{margin:1em 0;text-align:center;}",
        ".plot img{max-width:100%;border:1px solid #ddd;border-radius:4px;}",
        ".hash{font-family:monospace;font-size:.8em;color:#888;"
        "word-break:break-all;background:#f6f8fa;padding:.3em .5em;border-radius:4px;}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{_html.escape(title)}</h1>",
        f'<div class="meta">Source: <code>{_html.escape(src_name)}</code> · '
        f'Generated: {datetime.datetime.now().isoformat(timespec="seconds")}</div>',
    ]
    if src_hash:
        parts.append(f'<div class="meta">SHA-256: <span class="hash">{src_hash}</span></div>')
    parts.append("<h2>Source</h2>")
    parts.append(f"<pre><code>{_html.escape(source)}</code></pre>")
    parts.append("<h2>Output</h2>")
    parts.append(f"<pre>{_html.escape(stdout_text) if stdout_text else '(no output)'}</pre>")
    if plots_b64:
        parts.append(f"<h2>Plots ({len(plots_b64)})</h2>")
        for i, b64 in enumerate(plots_b64, 1):
            parts.append(f'<div class="plot"><img alt="Plot {i}" '
                         f'src="data:image/png;base64,{b64}"></div>')
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


def _render_notebook_markdown(title, source, stdout_text, plots_b64, src_hash, src_name):
    import datetime
    lines = [
        f"# {title}",
        "",
        f"*Source:* `{src_name}`  ·  *Generated:* {datetime.datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    if src_hash:
        lines.append(f"*SHA-256:* `{src_hash}`")
        lines.append("")
    lines.append("## Source")
    lines.append("")
    lines.append("```")
    lines.append(source.rstrip())
    lines.append("```")
    lines.append("")
    lines.append("## Output")
    lines.append("")
    lines.append("```")
    lines.append((stdout_text or "(no output)").rstrip())
    lines.append("```")
    lines.append("")
    if plots_b64:
        lines.append(f"## Plots ({len(plots_b64)})")
        lines.append("")
        for i, b64 in enumerate(plots_b64, 1):
            lines.append(f"![Plot {i}](data:image/png;base64,{b64})")
            lines.append("")
    return "\n".join(lines) + "\n"


# ============================================================================
# Paper-Mode-Output: print_table mit LaTeX-Booktabs / Markdown / CSV + ± für UncertainQuantity
# ============================================================================

def _format_cell_value(v, precision=4):
    """Formatiert eine Zelle: UncertainQuantity → 'val ± std [unit]', Quantity → 'val [unit]'."""
    if isinstance(v, UncertainQuantity):
        unit = f" [{v.unit}]" if getattr(v, "unit", "") else ""
        return f"{v.value:.{precision}g} ± {v.std:.{precision}g}{unit}"
    if isinstance(v, Quantity):
        unit = f" [{v.unit}]" if v.unit else ""
        return f"{v.value:.{precision}g}{unit}"
    if isinstance(v, float):
        return f"{v:.{precision}g}"
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            iv = v.item()
            return _format_cell_value(iv, precision)
        except Exception:
            pass
    return str(v)


def _format_cell_latex(v, precision=4):
    """Wie _format_cell_value, aber LaTeX-tauglich (`\\pm`, `\\,[\\mathrm{...}]`)."""
    if isinstance(v, UncertainQuantity):
        unit = f"\\,[\\mathrm{{{v.unit}}}]" if getattr(v, "unit", "") else ""
        return f"${v.value:.{precision}g} \\pm {v.std:.{precision}g}{unit}$"
    if isinstance(v, Quantity):
        unit = f"\\,[\\mathrm{{{v.unit}}}]" if v.unit else ""
        return f"${v.value:.{precision}g}{unit}$"
    if isinstance(v, float):
        return f"${v:.{precision}g}$"
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            return _format_cell_latex(v.item(), precision)
        except Exception:
            pass
    return str(v)


def print_table(rows, headers=None, format="markdown", precision=4, caption=None, label=None):
    """Erzeugt eine Tabelle in einem von vier Formaten und gibt sie via `print()` aus.

    - `rows`: Liste von Listen/Tupeln *oder* eine `DataFrame`.
    - `headers`: optional Liste von Spaltennamen (wenn rows kein DataFrame ist).
    - `format`: `"markdown"` (default), `"latex"` (booktabs), `"csv"`, `"plain"` (ASCII-Tabelle).
    - `precision`: signifikante Stellen für float/Quantity (default 4).
    - `caption`, `label`: nur für LaTeX (Tabellen-Caption und `\\label{...}`).

    UncertainQuantity wird automatisch als `val ± std [unit]` formatiert,
    Quantity als `val [unit]`. Für LaTeX werden ± und Einheiten mathmodisch gesetzt.
    """
    if isinstance(rows, DataFrame):
        df = rows
        headers = list(df.columns)
        # Einheiten in Header für Märkdown/Plain/CSV — LaTeX bleibt einheitslos im Header.
        units_map = dict(df._units)
        rows = [[df._cols[j][i] for j in range(len(headers))] for i in range(df.n_rows)]
    else:
        units_map = {}
        if headers is None:
            headers = [f"col{i+1}" for i in range(len(rows[0]) if rows else 0)]
        else:
            headers = list(headers)
    fmt = str(format).lower()
    if fmt == "latex":
        out = _table_latex(rows, headers, precision, caption, label)
    elif fmt == "csv":
        out = _table_csv(rows, headers, precision, units_map)
    elif fmt == "plain":
        out = _table_plain(rows, headers, precision, units_map)
    else:
        out = _table_markdown(rows, headers, precision, units_map)
    print(out)
    return out


def _decorate_header_with_unit(name, units_map):
    u = units_map.get(name) if units_map else None
    return f"{name} [{u}]" if u else name


def _table_markdown(rows, headers, precision, units_map):
    hdr = [_decorate_header_with_unit(h, units_map) for h in headers]
    body = [[_format_cell_value(v, precision) for v in row] for row in rows]
    widths = [_builtin_max(len(hdr[j]),
                           *(len(b[j]) for b in body) if body else (len(hdr[j]),))
              for j in range(len(headers))]
    lines = []
    lines.append("| " + " | ".join(hdr[j].ljust(widths[j]) for j in range(len(headers))) + " |")
    lines.append("|" + "|".join("-" * (widths[j] + 2) for j in range(len(headers))) + "|")
    for b in body:
        lines.append("| " + " | ".join(b[j].ljust(widths[j]) for j in range(len(headers))) + " |")
    return "\n".join(lines)


def _table_latex(rows, headers, precision, caption, label):
    cols = "l" * len(headers)
    lines = [r"\begin{table}[h]", r"\centering"]
    if caption:
        lines.append(r"\caption{" + str(caption) + "}")
    if label:
        lines.append(r"\label{" + str(label) + "}")
    lines.append(r"\begin{tabular}{" + cols + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(str(h) for h in headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        cells = [_format_cell_latex(v, precision) for v in row]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _table_csv(rows, headers, precision, units_map):
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([_decorate_header_with_unit(h, units_map) for h in headers])
    for row in rows:
        w.writerow([_format_cell_value(v, precision) for v in row])
    return buf.getvalue().rstrip("\n")


def _table_plain(rows, headers, precision, units_map):
    hdr = [_decorate_header_with_unit(h, units_map) for h in headers]
    body = [[_format_cell_value(v, precision) for v in row] for row in rows]
    widths = [_builtin_max(len(hdr[j]),
                           *(len(b[j]) for b in body) if body else (len(hdr[j]),))
              for j in range(len(headers))]
    sep = "  "
    lines = [sep.join(hdr[j].ljust(widths[j]) for j in range(len(headers)))]
    lines.append(sep.join("-" * widths[j] for j in range(len(headers))))
    for b in body:
        lines.append(sep.join(b[j].ljust(widths[j]) for j in range(len(headers))))
    return "\n".join(lines)
