def read_file(path):
    """
    Liest eine Datei als Text (UTF-8).
    path: String (Dateipfad).
    Rückgabe: String (Inhalt).
    """
    p = str(path)
    with open(p, 'r', encoding='utf-8') as f:
        return f.read()


def write_file(path, content):
    """
    Schreibt Text in eine Datei (UTF-8); überschreibt bei Existenz.
    path: String (Dateipfad). content: String (Inhalt).
    """
    p = str(path)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(str(content))


def file_exists(path):
    """
    Prüft, ob eine Datei existiert.
    path: String (Dateipfad). Rückgabe: bool.
    """
    import os
    return os.path.isfile(str(path))


def http_get(url):
    """
    HTTP GET-Anfrage; gibt Antworttext (UTF-8) zurück.
    url: String (z. B. \"https://example.com\").
    """
    import urllib.request
    req = urllib.request.Request(str(url), method='GET')
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode('utf-8')


def http_post(url, data):
    """
    HTTP POST-Anfrage. data: String (Body) oder Dict/List (wird als JSON gesendet).
    url: String. Rückgabe: Antworttext (UTF-8).
    """
    import urllib.request
    import json
    u = str(url)
    if isinstance(data, dict) or isinstance(data, list):
        body = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(u, data=body, method='POST', headers={'Content-Type': 'application/json'})
    else:
        body = str(data).encode('utf-8')
        req = urllib.request.Request(u, data=body, method='POST')
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode('utf-8')


def json_parse(s):
    """
    Parst einen JSON-String zu einem Objekt (Dict/List); für Zugriff z. B. obj[\"key\"].
    s: String (gültiger JSON).
    """
    import json
    return json.loads(str(s))


def json_stringify(obj):
    """
    Wandelt ein Objekt (Dict, List, Zahl, String) in einen JSON-String um.
    """
    import json
    if hasattr(obj, 'tolist'):
        obj = obj.tolist()
    elif hasattr(obj, 'item'):
        obj = obj.item()
    return json.dumps(obj, ensure_ascii=False)

# --- Standard Library: Assert (Testing) ---

def _dedekind_assert(condition, message=None):
    """
    Prüft eine Bedingung; löst AssertionError aus, wenn condition falsch ist.
    assert(condition) oder assert(condition, "Fehlermeldung").
    """
    val = condition
    if hasattr(val, "item") and (hasattr(val, "numel") or hasattr(val, "size")):
        size = val.numel() if hasattr(val, "numel") else val.size
        val = val.item() if size == 1 else bool(val.all().item())
    elif hasattr(val, "__len__") and len(val) == 1:
        val = val[0]
    if not bool(val):
        raise AssertionError(message if message is not None else "Assertion failed")

# --- Standard Library: Symbolische Ableitung (eingebettet, kein Import) ---
# Einfacher AST für Ausdrücke
class _SymExpr:
    pass

class _SymVar(_SymExpr):
    def __init__(self, name: str):
        self.name = name

class _SymConst(_SymExpr):
    def __init__(self, value: float):
        self.value = value

class _SymAdd(_SymExpr):
    def __init__(self, left: _SymExpr, right: _SymExpr):
        self.left = left
        self.right = right

class _SymSub(_SymExpr):
    def __init__(self, left: _SymExpr, right: _SymExpr):
        self.left = left
        self.right = right

class _SymMul(_SymExpr):
    def __init__(self, left: _SymExpr, right: _SymExpr):
        self.left = left
        self.right = right

class _SymDiv(_SymExpr):
    def __init__(self, left: _SymExpr, right: _SymExpr):
        self.left = left
        self.right = right

class _SymPow(_SymExpr):
    def __init__(self, base: _SymExpr, exp: _SymExpr):
        self.base = base
        self.exp = exp

class _SymNeg(_SymExpr):
    def __init__(self, arg: _SymExpr):
        self.arg = arg

class _SymSin(_SymExpr):
    def __init__(self, arg: _SymExpr):
        self.arg = arg

class _SymCos(_SymExpr):
    def __init__(self, arg: _SymExpr):
        self.arg = arg

class _SymTan(_SymExpr):
    def __init__(self, arg: _SymExpr):
        self.arg = arg

class _SymExp(_SymExpr):
    def __init__(self, arg: _SymExpr):
        self.arg = arg

class _SymLog(_SymExpr):
    def __init__(self, arg: _SymExpr):
        self.arg = arg

class _SymSqrt(_SymExpr):
    def __init__(self, arg: _SymExpr):
        self.arg = arg

def _sym_tokenize(s):
    """Tokenizer ohne re: Zeichen für Zeichen."""
    tokens = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c in " \t":
            i += 1
            continue
        if c in "+":
            tokens.append(("PLUS", "+"))
            i += 1
            continue
        if c in "-":
            tokens.append(("MINUS", "-"))
            i += 1
            continue
        if c in "*":
            tokens.append(("MUL", "*"))
            i += 1
            continue
        if c in "/":
            tokens.append(("DIV", "/"))
            i += 1
            continue
        if c in "^":
            tokens.append(("POW", "^"))
            i += 1
            continue
        if c in "(":
            tokens.append(("LPAREN", "("))
            i += 1
            continue
        if c in ")":
            tokens.append(("RPAREN", ")"))
            i += 1
            continue
        if c.isdigit() or (c == "." and i + 1 < n and s[i + 1].isdigit()):
            start = i
            if s[i] == ".":
                i += 1
            while i < n and s[i].isdigit():
                i += 1
            if i < n and s[i] == ".":
                i += 1
                while i < n and s[i].isdigit():
                    i += 1
            if i < n and s[i] in "eE":
                i += 1
                if i < n and s[i] in "+-":
                    i += 1
                while i < n and s[i].isdigit():
                    i += 1
            try:
                value = float(s[start:i])
            except ValueError:
                raise ValueError("Ungültige Zahl: " + s[start:i])
            tokens.append(("NUMBER", value))
            continue
        if c.isalpha() or c == "_":
            start = i
            while i < n and (s[i].isalnum() or s[i] == "_"):
                i += 1
            tokens.append(("ID", s[start:i]))
            continue
        raise ValueError("Unerwartetes Zeichen: " + repr(c))
    return tokens

class _SymParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def _peek(self):
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def _advance(self):
        if self.pos >= len(self.tokens):
            return None
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def _parse_expr(self) -> _SymExpr:
        left = self._parse_term()
        while True:
            p = self._peek()
            if p is None:
                break
            if p[0] == "PLUS":
                self._advance()
                left = _SymAdd(left, self._parse_term())
            elif p[0] == "MINUS":
                self._advance()
                left = _SymSub(left, self._parse_term())
            else:
                break
        return left

    def _parse_term(self) -> _SymExpr:
        left = self._parse_factor()
        while True:
            p = self._peek()
            if p is None:
                break
            if p[0] == "MUL":
                self._advance()
                left = _SymMul(left, self._parse_factor())
            elif p[0] == "DIV":
                self._advance()
                left = _SymDiv(left, self._parse_factor())
            else:
                break
        return left

    def _parse_factor(self) -> _SymExpr:
        base = self._parse_unary()
        p = self._peek()
        if p is not None and p[0] == "POW":
            self._advance()
            exp = self._parse_factor()
            return _SymPow(base, exp)
        return base

    def _parse_unary(self) -> _SymExpr:
        p = self._peek()
        if p is None:
            raise ValueError("Unerwartetes Ende des Ausdrucks")
        if p[0] == "PLUS":
            self._advance()
            return self._parse_unary()
        if p[0] == "MINUS":
            self._advance()
            return _SymNeg(self._parse_unary())
        return self._parse_atom()

    def _parse_atom(self) -> _SymExpr:
        p = self._advance()
        if p is None:
            raise ValueError("Unerwartetes Ende des Ausdrucks")
        kind, value = p
        if kind == "NUMBER":
            return _SymConst(float(value))
        if kind == "ID":
            name = str(value)
            if name in ("sin", "cos", "tan", "exp", "log", "sqrt"):
                lp = self._peek()
                if lp is not None and lp[0] == "LPAREN":
                    self._advance()
                    arg = self._parse_expr()
                    rp = self._advance()
                    if rp is None or rp[0] != "RPAREN":
                        raise ValueError("Fehlende ')' nach " + name)
                    if name == "sin":
                        return _SymSin(arg)
                    if name == "cos":
                        return _SymCos(arg)
                    if name == "tan":
                        return _SymTan(arg)
                    if name == "exp":
                        return _SymExp(arg)
                    if name == "log":
                        return _SymLog(arg)
                    if name == "sqrt":
                        return _SymSqrt(arg)
            return _SymVar(name)
        if kind == "LPAREN":
            e = self._parse_expr()
            rp = self._advance()
            if rp is None or rp[0] != "RPAREN":
                raise ValueError("Fehlende ')'")
            return e
        raise ValueError(f"Unerwartetes Token: {p}")

def _sym_parse(expr_str):
    s = expr_str.strip().replace(" ", "")
    if not s:
        raise ValueError("Leerer Ausdruck")
    tokens = _sym_tokenize(s)
    if not tokens:
        raise ValueError("Keine gültigen Tokens")
    p = _SymParser(tokens)
    e = p._parse_expr()
    if p.pos < len(tokens):
        raise ValueError("Rest nach Ausdruck")
    return e

def _sym_diff(e, var):
    if isinstance(e, _SymVar):
        return _SymConst(1.0) if e.name == var else _SymConst(0.0)
    if isinstance(e, _SymConst):
        return _SymConst(0.0)
    if isinstance(e, _SymAdd):
        return _SymAdd(_sym_diff(e.left, var), _sym_diff(e.right, var))
    if isinstance(e, _SymSub):
        return _SymSub(_sym_diff(e.left, var), _sym_diff(e.right, var))
    if isinstance(e, _SymMul):
        return _SymAdd(
            _SymMul(_sym_diff(e.left, var), e.right),
            _SymMul(e.left, _sym_diff(e.right, var))
        )
    if isinstance(e, _SymDiv):
        return _SymDiv(
            _SymSub(_SymMul(_sym_diff(e.left, var), e.right), _SymMul(e.left, _sym_diff(e.right, var))),
            _SymPow(e.right, _SymConst(2.0))
        )
    if isinstance(e, _SymPow):
        if isinstance(e.exp, _SymConst):
            c = e.exp.value
            if c == 0:
                return _SymConst(0.0)
            if c == 1:
                return _sym_diff(e.base, var)
            return _SymMul(_SymConst(c), _SymMul(_SymPow(e.base, _SymConst(c - 1)), _sym_diff(e.base, var)))
        return _SymMul(
            _SymPow(e.base, e.exp),
            _SymAdd(
                _SymMul(_sym_diff(e.exp, var), _SymLog(e.base)),
                _SymMul(e.exp, _SymDiv(_sym_diff(e.base, var), e.base))
            )
        )
    if isinstance(e, _SymNeg):
        return _SymNeg(_sym_diff(e.arg, var))
    if isinstance(e, _SymSin):
        return _SymMul(_SymCos(e.arg), _sym_diff(e.arg, var))
    if isinstance(e, _SymCos):
        return _SymMul(_SymNeg(_SymSin(e.arg)), _sym_diff(e.arg, var))
    if isinstance(e, _SymTan):
        return _SymMul(_SymAdd(_SymConst(1.0), _SymPow(_SymTan(e.arg), _SymConst(2.0))), _sym_diff(e.arg, var))
    if isinstance(e, _SymExp):
        return _SymMul(_SymExp(e.arg), _sym_diff(e.arg, var))
    if isinstance(e, _SymLog):
        return _SymMul(_SymDiv(_SymConst(1.0), e.arg), _sym_diff(e.arg, var))
    if isinstance(e, _SymSqrt):
        return _SymMul(_SymDiv(_SymConst(1.0), _SymMul(_SymConst(2.0), _SymSqrt(e.arg))), _sym_diff(e.arg, var))
    raise TypeError(f"Unbekannter Ausdruckstyp: {type(e)}")

def _sym_to_string(e, parent_prec=0):
    if isinstance(e, _SymVar):
        return e.name
    if isinstance(e, _SymConst):
        v = e.value
        if v == int(v):
            return str(int(v))
        return str(v)
    if isinstance(e, _SymAdd):
        s = _sym_to_string(e.left, 1) + " + " + _sym_to_string(e.right, 1)
        return f"({s})" if parent_prec > 0 else s
    if isinstance(e, _SymSub):
        s = _sym_to_string(e.left, 1) + " - " + _sym_to_string(e.right, 2)
        return f"({s})" if parent_prec > 0 else s
    if isinstance(e, _SymMul):
        left = _sym_to_string(e.left, 2)
        right = _sym_to_string(e.right, 2)
        if isinstance(e.left, (_SymAdd, _SymSub)):
            left = f"({left})"
        if isinstance(e.right, (_SymAdd, _SymSub)):
            right = f"({right})"
        s = f"{left}*{right}"
        return f"({s})" if parent_prec > 1 else s
    if isinstance(e, _SymDiv):
        num = _sym_to_string(e.left, 2)
        den = _sym_to_string(e.right, 2)
        if isinstance(e.left, (_SymAdd, _SymSub)):
            num = f"({num})"
        if isinstance(e.right, (_SymAdd, _SymSub, _SymMul)):
            den = f"({den})"
        s = f"{num}/{den}"
        return f"({s})" if parent_prec > 1 else s
    if isinstance(e, _SymPow):
        base = _sym_to_string(e.base, 3)
        exp = _sym_to_string(e.exp, 3)
        if isinstance(e.base, (_SymAdd, _SymSub, _SymMul, _SymDiv)):
            base = f"({base})"
        s = f"{base}^{exp}"
        return f"({s})" if parent_prec > 2 else s
    if isinstance(e, _SymNeg):
        return "-" + _sym_to_string(e.arg, 3)
    if isinstance(e, _SymSin):
        return "sin(" + _sym_to_string(e.arg, 0) + ")"
    if isinstance(e, _SymCos):
        return "cos(" + _sym_to_string(e.arg, 0) + ")"
    if isinstance(e, _SymTan):
        return "tan(" + _sym_to_string(e.arg, 0) + ")"
    if isinstance(e, _SymExp):
        return "exp(" + _sym_to_string(e.arg, 0) + ")"
    if isinstance(e, _SymLog):
        return "log(" + _sym_to_string(e.arg, 0) + ")"
    if isinstance(e, _SymSqrt):
        return "sqrt(" + _sym_to_string(e.arg, 0) + ")"
    return "?"

def _sym_simplify(e):
    if isinstance(e, _SymConst):
        return e
    if isinstance(e, _SymVar):
        return e
    if isinstance(e, _SymAdd):
        l, r = _sym_simplify(e.left), _sym_simplify(e.right)
        if isinstance(l, _SymConst) and l.value == 0:
            return r
        if isinstance(r, _SymConst) and r.value == 0:
            return l
        return _SymAdd(l, r)
    if isinstance(e, _SymSub):
        l, r = _sym_simplify(e.left), _sym_simplify(e.right)
        if isinstance(r, _SymConst) and r.value == 0:
            return l
        return _SymSub(l, r)
    if isinstance(e, _SymMul):
        l, r = _sym_simplify(e.left), _sym_simplify(e.right)
        if isinstance(l, _SymConst):
            if l.value == 0:
                return _SymConst(0.0)
            if l.value == 1:
                return r
        if isinstance(r, _SymConst):
            if r.value == 0:
                return _SymConst(0.0)
            if r.value == 1:
                return l
        return _SymMul(l, r)
    if isinstance(e, _SymDiv):
        l, r = _sym_simplify(e.left), _sym_simplify(e.right)
        if isinstance(l, _SymConst) and l.value == 0:
            return _SymConst(0.0)
        return _SymDiv(l, r)
    if isinstance(e, _SymPow):
        b, x = _sym_simplify(e.base), _sym_simplify(e.exp)
        if isinstance(x, _SymConst):
            if x.value == 0:
                return _SymConst(1.0)
            if x.value == 1:
                return b
        return _SymPow(b, x)
    if isinstance(e, _SymNeg):
        a = _sym_simplify(e.arg)
        if isinstance(a, _SymConst):
            return _SymConst(-a.value)
        return _SymNeg(a)
    if isinstance(e, (_SymSin, _SymCos, _SymTan, _SymExp, _SymLog, _SymSqrt)):
        a = _sym_simplify(e.arg)
        return type(e)(a)
    return e

def diff_sym(expr, var):
    """
    Symbolische Ableitung: leitet den Ausdruck expr nach der Variable var ab.
    expr: String, z.B. 'x^2 + sin(x)', 'exp(x)*log(x)'.
    var: Name der Variable, z.B. 'x'.
    Rückgabe: Ableitung als String.
    """
    expr = str(expr).strip()
    var = str(var).strip()
    if not var:
        raise ValueError("Variable darf nicht leer sein")
    ast = _sym_parse(expr)
    d = _sym_diff(ast, var)
    d = _sym_simplify(d)
    return _sym_to_string(d)


def integrate_sym(expr, var):
    """
    Unbestimmtes (symbolisches) Integral: ∫ expr d(var).
    expr: String, z.B. 'x^2', 'sin(x)', 'exp(x)*x'.
    var: Name der Integrationsvariable, z.B. 'x'.
    Rückgabe: Stammfunktion als String (ohne Integrationskonstante C).
    Nutzt SymPy für robuste symbolische Integration.
    """
    try:
        import sympy  # type: ignore[reportMissingImports]
    except ImportError:
        raise ImportError("integrate_sym erfordert sympy. Bitte installieren: pip install sympy")
    expr_str = str(expr).strip().replace(" ", "")
    var_str = str(var).strip()
    if not var_str:
        raise ValueError("Integrationsvariable darf nicht leer sein")
    # Dedekind-Syntax (^) für SymPy (**) konvertieren
    expr_sympy = expr_str.replace("^", "**")
    try:
        x = sympy.Symbol(var_str)
        sym_expr = sympy.sympify(expr_sympy)
        result = sympy.integrate(sym_expr, x)
        out = str(result)
        # Optional: ** zurück zu ^ für Dedekind-Konsistenz
        return out.replace("**", "^")
    except Exception as e:
        raise ValueError(f"integrate_sym: Konnte '{expr}' nicht nach {var_str} integrieren: {e}") from e

# --- Standard Library: Jacobian / Hessian (Autograd) ---

def jacobian(f, x):
    """
    Jacobi-Matrix von f an der Stelle x. f: R^n -> R^m (Callable, Tensor -> Tensor).
    x: 1D-Tensor der Länge n. Rückgabe: Tensor (m, n); Zeile i = Gradient von f_i.
    """
    x_t = _to_tensor(x).float().clone().detach().requires_grad_(True)
    out = f(x_t)
    out = _to_tensor(out).float()
    if out.dim() == 0:
        out = out.unsqueeze(0)
    out_flat = out.flatten()
    m, n = out_flat.numel(), x_t.numel()
    J = torch.zeros(m, n, device=x_t.device, dtype=x_t.dtype)
    for i in range(m):
        grad_out = torch.zeros_like(out_flat)
        grad_out[i] = 1.0
        g, = torch.autograd.grad(out_flat, x_t, grad_outputs=grad_out, retain_graph=True)
        J[i] = g.flatten()
    return J

def hessian(f, x):
    """
    Hesse-Matrix von f an der Stelle x. f: R^n -> R (skalar).
    x: 1D-Tensor der Länge n. Rückgabe: Tensor (n, n).
    """
    x_t = _to_tensor(x).float().clone().detach().requires_grad_(True)
    out = f(x_t)
    out = _to_tensor(out).float()
    if out.dim() != 0:
        out = out.sum()
    grad_x, = torch.autograd.grad(out, x_t, create_graph=True)
    n = x_t.numel()
    H = torch.zeros(n, n, device=x_t.device, dtype=x_t.dtype)
    for i in range(n):
        g, = torch.autograd.grad(grad_x[i], x_t, retain_graph=True)
        H[i] = g.flatten()
    return H

# --- Standard Library: Sorting ---

def sort(data, descending=False):
    data = _to_tensor(data)
    sorted_values, _ = torch.sort(data, descending=descending)
    return sorted_values

def quicksort(data):
    return sort(data)

# --- Standard Library: Visualization ---
# Plots werden in _dedekind_plots gesammelt und vom Server an die IDE zurückgegeben.

_dedekind_plots = []

def _plot_ndarray(x, y=None, title=None, xlabel=None, ylabel=None, kind="line", xscale="linear", yscale="linear"):
    """Intern: Erzeugt einen Plot und hängt ihn als Base64-PNG an _dedekind_plots. kind: line, scatter."""
    try:
        import base64
        import io
        import matplotlib  # type: ignore[import-untyped]
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]
    except ImportError:
        print("plot(): matplotlib nicht installiert. pip install matplotlib")
        return
    if y is None:
        y = x
        x = list(range(len(y)))
    if x is None:
        x = list(range(len(y) if hasattr(y, '__len__') else 0))
    if hasattr(x, 'cpu'): x = x.detach().cpu().numpy()
    elif not isinstance(x, (list, tuple)): x = list(x)
    if hasattr(y, 'cpu'): y = y.detach().cpu().numpy()
    elif not isinstance(y, (list, tuple)): y = list(y)
    import numpy as np  # type: ignore[reportMissingImports]
    if getattr(x, 'dtype', None) is not None and np.issubdtype(x.dtype, np.complexfloating):
        x = np.real(x)
    if getattr(y, 'dtype', None) is not None and np.issubdtype(y.dtype, np.complexfloating):
        y = np.real(y)
    fig, ax = plt.subplots()
    if kind == "scatter":
        ax.scatter(x, y)
    else:
        ax.plot(x, y)
    if xscale == "log": ax.set_xscale("log")
    if yscale == "log": ax.set_yscale("log")
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    png_bytes = buf.getvalue()
    _dedekind_plots.append(base64.b64encode(png_bytes).decode('ascii'))
    try:
        import sys
        for i in range(1, 8):
            try:
                f = sys._getframe(i)
            except ValueError:
                break
            g = f.f_globals
            if '_dedekind_display_image' in g:
                g['_dedekind_display_image'](png_bytes, 'image/png')
                break
        else:
            from IPython import get_ipython  # type: ignore[import-untyped]
            ip = get_ipython()
            if ip is not None:
                from IPython.display import display, Image  # type: ignore[import-untyped]
                display(Image(data=png_bytes))
    except Exception:
        pass

def _plot_contour_inner(X, Y, Z, title=None, xlabel=None, ylabel=None, levels=10):
    """Intern: Contour-Plot; X, Y, Z als NumPy-Arrays (Z 2D)."""
    try:
        import base64
        import io
        import matplotlib  # type: ignore[import-untyped]
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]
        import numpy as np  # type: ignore[reportMissingImports]
    except ImportError:
        print("contour(): matplotlib nicht installiert. pip install matplotlib")
        return
    fig, ax = plt.subplots()
    nlev = int(levels) if levels is not None else 10
    ax.contour(X, Y, Z, levels=nlev)
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    png_bytes = buf.getvalue()
    _dedekind_plots.append(base64.b64encode(png_bytes).decode('ascii'))
    try:
        import sys
        for i in range(1, 8):
            try:
                f = sys._getframe(i)
            except ValueError:
                break
            g = f.f_globals
            if '_dedekind_display_image' in g:
                g['_dedekind_display_image'](png_bytes, 'image/png')
                break
        else:
            from IPython import get_ipython  # type: ignore[import-untyped]
            ip = get_ipython()
            if ip is not None:
                from IPython.display import display, Image  # type: ignore[import-untyped]
                display(Image(data=png_bytes))
    except Exception:
        pass

def print_latex(s):
    """
    Gibt LaTeX-String in der Konsole als Unicode-Formel aus (α, Δ, ∫, ½ etc.); keine Bilder in Plots.
    Bsp.: print_latex(r"\\frac{1}{2}"), print_latex(r"\\alpha + \\Delta x").
    Wenn kein Kernel vorhanden, Fallback: print(s).
    """
    s = str(s)
    try:
        import sys
        for i in range(1, 8):
            try:
                f = sys._getframe(i)
            except ValueError:
                break
            g = f.f_globals
            if '_dedekind_display_latex' in g:
                g['_dedekind_display_latex'](s)
                return
        print(s)
    except Exception:
        print(s)


def plot(x=None, y=None, title=None, xlabel=None, ylabel=None, xscale="linear", yscale="linear"):
    """
    Zeichnet Daten (Linie) und zeigt sie in Dedekind Studio an.
    plot(y) – y über Index; plot(x, y) – y über x.
    xscale, yscale: "linear" (default) oder "log".
    """
    if x is None and y is None:
        return
    if y is None:
        y = _to_tensor(x) if isinstance(x, (list, tuple)) else x
        x = None
    elif not isinstance(x, (list, tuple)) and not hasattr(x, 'shape'):
        x = _to_tensor(x) if hasattr(x, '__iter__') else x
    if not isinstance(y, (list, tuple)) and not hasattr(y, 'shape'):
        y = _to_tensor(y) if hasattr(y, '__iter__') else y
    _plot_ndarray(x, y, title=title, xlabel=xlabel, ylabel=ylabel, kind="line", xscale=xscale, yscale=yscale)

def scatter(x=None, y=None, title=None, xlabel=None, ylabel=None):
    """
    Streudiagramm: Punkte (x, y). scatter(y) – y über Index; scatter(x, y) – Punkte.
    Zeigt in Dedekind Studio an.
    """
    if x is None and y is None:
        return
    if y is None:
        y = _to_tensor(x) if isinstance(x, (list, tuple)) else x
        x = None
    elif not isinstance(x, (list, tuple)) and not hasattr(x, 'shape'):
        x = _to_tensor(x) if hasattr(x, '__iter__') else x
    if not isinstance(y, (list, tuple)) and not hasattr(y, 'shape'):
        y = _to_tensor(y) if hasattr(y, '__iter__') else y
    _plot_ndarray(x, y, title=title, xlabel=xlabel, ylabel=ylabel, kind="scatter")

def contour(X, Y, Z, title=None, xlabel=None, ylabel=None, levels=10):
    """
    Höhenlinien-Plot. X, Y: 1D- oder 2D-Koordinaten (z. B. linspace); Z: 2D-Matrix (Werte).
    levels: Anzahl Konturlinien (default 10). Zeigt in Dedekind Studio an.
    """
    import numpy as np  # type: ignore[reportMissingImports]
    X_t = _to_tensor(X).float().detach().cpu().numpy()
    Y_t = _to_tensor(Y).float().detach().cpu().numpy()
    Z_t = _to_tensor(Z).float().detach().cpu().numpy()
    if Z_t.ndim != 2:
        raise ValueError("contour: Z muss 2D-Matrix sein.")
    if X_t.ndim == 1 and Y_t.ndim == 1:
        X_t, Y_t = np.meshgrid(X_t, Y_t)
    _plot_contour_inner(X_t, Y_t, Z_t, title=title, xlabel=xlabel, ylabel=ylabel, levels=levels)


# ============================================================================
# Reproduzierbarkeit: Seed-Manager + Daten-Hash
# ============================================================================

def seed(n):
    """Setzt deterministischen Seed in Python (`random`), NumPy und PyTorch.
    Verwendung: `seed(42)` ganz am Anfang eines Skripts; Zufallsoperationen werden reproduzierbar."""
    try:
        import random as _random
        _random.seed(int(n))
    except Exception:
        pass
    try:
        import numpy as _np  # type: ignore[reportMissingImports]
        _np.random.seed(int(n) & 0xFFFFFFFF)
    except Exception:
        pass
    try:
        torch.manual_seed(int(n))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(n))
    except Exception:
        pass
    return int(n)


def data_hash(x):
    """Liefert SHA256-Hex-Digest einer Eingabe (Tensor, Liste, Dict, Zahl, String, DataFrame).
    Nützlich für Reproduzierbarkeits-Protokolle: `print("Input hash:", data_hash(data))`."""
    import hashlib
    import json
    if isinstance(x, DataFrame):
        payload = json.dumps({"cols": x.columns, "units": x._units, "data": [list(c) for c in x._cols]},
                             sort_keys=True, default=_json_default).encode("utf-8")
    elif hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        arr = x.detach().cpu().numpy()
        payload = arr.tobytes() + str(arr.shape).encode("utf-8") + str(arr.dtype).encode("utf-8")
    elif isinstance(x, (list, tuple, dict)):
        payload = json.dumps(x, sort_keys=True, default=_json_default).encode("utf-8")
    elif isinstance(x, (bytes, bytearray)):
        payload = bytes(x)
    else:
        payload = str(x).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_default(o):
    if isinstance(o, Quantity):
        return {"__quantity__": True, "value": o.value, "unit": o.unit}
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


# ============================================================================
# DataFrame + tabular I/O (CSV nativ; Parquet/HDF5/NetCDF optional)
# ============================================================================

