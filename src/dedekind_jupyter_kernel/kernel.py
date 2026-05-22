"""
Dedekind Jupyter Kernel: executes Dedekind source in a persistent context.
Requires the `dedekind` package to be importable (installed via pip, or src/
on PYTHONPATH for editable / dev runs).
"""
import re
import sys
import io
import os

# Dev/editable fallback: if `dedekind` is not installed and we're sitting next
# to a src/dedekind tree, prepend that src to sys.path so imports resolve.
try:
    import dedekind as _ddk  # noqa: F401
except ImportError:
    _src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    if os.path.isdir(os.path.join(_src, "dedekind")) and _src not in sys.path:
        sys.path.insert(0, _src)

from ipykernel.kernelbase import Kernel

try:
    from dedekind.ast_nodes import CompileError
except ImportError:
    CompileError = Exception  # fallback if run outside repo

# LaTeX → Unicode für native Darstellung in der Konsole (UTF-8; Font mit mathematischen Zeichen)
_LATEX_TO_UNICODE = [
    # Griechisch (lange zuerst)
    (r"\Alpha", "Α"), (r"\Beta", "Β"), (r"\Gamma", "Γ"), (r"\Delta", "Δ"), (r"\Epsilon", "Ε"),
    (r"\Zeta", "Ζ"), (r"\Eta", "Η"), (r"\Theta", "Θ"), (r"\Iota", "Ι"), (r"\Kappa", "Κ"),
    (r"\Lambda", "Λ"), (r"\Mu", "Μ"), (r"\Nu", "Ν"), (r"\Xi", "Ξ"), (r"\Omicron", "Ο"),
    (r"\Pi", "Π"), (r"\Rho", "Ρ"), (r"\Sigma", "Σ"), (r"\Tau", "Τ"), (r"\Upsilon", "Υ"),
    (r"\Phi", "Φ"), (r"\Chi", "Χ"), (r"\Psi", "Ψ"), (r"\Omega", "Ω"),
    (r"\alpha", "α"), (r"\beta", "β"), (r"\gamma", "γ"), (r"\delta", "δ"), (r"\epsilon", "ε"),
    (r"\zeta", "ζ"), (r"\eta", "η"), (r"\theta", "θ"), (r"\iota", "ι"), (r"\kappa", "κ"),
    (r"\lambda", "λ"), (r"\mu", "μ"), (r"\nu", "ν"), (r"\xi", "ξ"), (r"\omicron", "ο"),
    (r"\pi", "π"), (r"\rho", "ρ"), (r"\sigma", "σ"), (r"\tau", "τ"), (r"\upsilon", "υ"),
    (r"\phi", "φ"), (r"\chi", "χ"), (r"\psi", "ψ"), (r"\omega", "ω"), (r"\hbar", "ℏ"),
    # Operatoren und Symbole
    (r"\int", "∫"), (r"\sum", "∑"), (r"\prod", "∏"), (r"\sqrt", "√"), (r"\infty", "∞"),
    (r"\partial", "∂"), (r"\nabla", "∇"), (r"\cdot", "·"), (r"\times", "×"), (r"\div", "÷"),
    (r"\pm", "±"), (r"\mp", "∓"), (r"\leq", "≤"), (r"\geq", "≥"), (r"\neq", "≠"),
    (r"\approx", "≈"), (r"\equiv", "≡"), (r"\in", "∈"), (r"\forall", "∀"), (r"\exists", "∃"),
    (r"\rightarrow", "→"), (r"\leftarrow", "←"), (r"\Rightarrow", "⇒"), (r"\Leftarrow", "⇐"),
    (r"\left(", "("), (r"\right)", ")"), (r"\quad", " "), (r"\,", " "), (r"\;", " "),
]
# Hoch-/Tiefstellen (Ziffern)
_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _latex_to_unicode(s):
    """Konvertiert einfaches LaTeX in Unicode-Text für die Konsole (UTF-8)."""
    t = s.strip()
    if t.startswith("$") and t.endswith("$"):
        t = t[1:-1].strip()
    for latex, unicode_char in _LATEX_TO_UNICODE:
        t = t.replace(latex, unicode_char)
    # \mathrm{...} und \texttt{...} -> Inhalt als Text (z. B. .sparse(), grad, Code)
    t = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\texttt\s*\{([^{}]*)\}", r"\1", t)
    # \frac{1}{2} -> ½ bzw. a/b; häufige Brüche als Unicode
    frac_map = {(1, 2): "½", (1, 3): "⅓", (2, 3): "⅔", (1, 4): "¼", (3, 4): "¾", (1, 5): "⅕"}
    def frac_repl(m):
        a, b = m.group(1).strip(), m.group(2).strip()
        try:
            na, nb = int(a), int(b)
            return frac_map.get((na, nb), f"{a}/{b}")
        except ValueError:
            return f"{a}/{b}"
    t = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", frac_repl, t)
    # Einfache Hochstellen: ^1 -> ¹ (nur einzelne Ziffer)
    t = re.sub(r"\^(\d)", lambda m: m.group(1).translate(_SUP), t)
    t = re.sub(r"_(\d)", lambda m: m.group(1).translate(_SUB), t)
    # Leerzeichen nach ∂ entfernen: ∂ T -> ∂T (saubere Bruchdarstellung)
    t = re.sub(r"∂\s+", "∂", t)
    return t.strip()


class DedekindKernel(Kernel):
    implementation = "Dedekind"
    implementation_version = "2.0.0"
    language = "dedekind"
    language_version = "2.0.0"
    language_info = {
        "name": "dedekind",
        "mimetype": "text/x-dedekind",
        "file_extension": ".ddk",
    }
    banner = "Dedekind Kernel – compile and run Dedekind code (Dedekind Language v2.0.0)"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._globals = {}
        self._execution_count = 0
        # comm_open auf Shell-Channel abfangen (ipykernel hat es nicht in msg_types)
        self.shell_handlers['comm_open'] = self._comm_open_noop

    def _comm_open_noop(self, stream, ident, msg):
        """No-op damit comm_open vom Frontend keine 'Unknown message type'-Warnung auslöst."""
        pass

    def do_execute(self, code, silent, store_history=True, user_expressions=None, allow_stdin=False):
        if not code.strip():
            return {"status": "ok", "payload": [], "user_expressions": {}, "execution_count": self._execution_count}

        if store_history:
            self._execution_count += 1
        reply = {"execution_count": self._execution_count}

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_capture, stderr_capture

        try:
            from dedekind import compile_source, dedekind_exec
        except ImportError as e:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            msg = (
                "Dedekind-Kernel: das `dedekind`-Paket ist nicht importierbar. "
                "Installiere via `pip install dedekind`, oder setze PYTHONPATH auf das "
                "src/-Verzeichnis dieses Repos fuer einen Editable-Run.\n"
                "ImportError: %s"
            ) % (e,)
            if not silent:
                self.send_response(
                    self.iopub_socket,
                    "stream",
                    {"name": "stderr", "text": msg + "\n"},
                )
            reply.update(status="error", ename="ImportError", evalue=str(e), traceback=[msg])
            return reply
        try:
            python_code = compile_source(code, filepath=None, check_units=True)
            # Injizieren: plot() kann Abbildungen an die Plots-Pane senden
            import base64
            _kernel_self = self
            def _dedekind_display_image(data_bytes, mime_type='image/png'):
                _kernel_self.send_response(
                    _kernel_self.iopub_socket,
                    'display_data',
                    {
                        'data': {mime_type: base64.b64encode(data_bytes).decode('ascii')},
                        'metadata': {},
                        'transient': {},
                    },
                )
            def _dedekind_display_latex(latex_str):
                """Formel nur in der Konsole als Unicode (α, Δ, ∫ etc.) – keine Bilder in Plots.
                Schreibt in stdout_capture, damit die Reihenfolge mit print() stimmt."""
                s = str(latex_str).strip()
                if not s:
                    return
                console_text = _latex_to_unicode(s)
                stdout_capture.write((console_text if console_text else s) + '\n')
            self._globals['_dedekind_display_image'] = _dedekind_display_image
            self._globals['_dedekind_display_latex'] = _dedekind_display_latex
            dedekind_exec(python_code, ddk_file=None, exec_globals=self._globals, ddk_source=code)
        except CompileError as e:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            err_msg = str(e)
            if not silent:
                self.send_response(
                    self.iopub_socket,
                    "stream",
                    {"name": "stderr", "text": err_msg + "\n"},
                )
            reply.update(status="error", ename="CompileError", evalue=err_msg, traceback=[err_msg])
            return reply
        except Exception as e:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            import traceback
            tb = traceback.format_exc()
            if not silent:
                self.send_response(
                    self.iopub_socket,
                    "stream",
                    {"name": "stderr", "text": tb + "\n"},
                )
            reply.update(status="error", ename=type(e).__name__, evalue=str(e), traceback=tb.splitlines())
            return reply

        sys.stdout, sys.stderr = old_stdout, old_stderr
        out = stdout_capture.getvalue()
        err = stderr_capture.getvalue()
        if not silent:
            if out:
                self.send_response(
                    self.iopub_socket,
                    "stream",
                    {"name": "stdout", "text": out},
                )
            if err:
                self.send_response(
                    self.iopub_socket,
                    "stream",
                    {"name": "stderr", "text": err},
                )
        reply.update(status="ok", payload=[], user_expressions={})
        return reply
