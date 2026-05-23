"""
Symbolic differentiation for Dedekind: parses mathematical expressions as strings
and returns the derivative with respect to a variable as a string.

Supports: +, -, *, /, ^, parentheses, sin, cos, tan, exp, log, sqrt; constants and variables.
"""

import re
from typing import Optional, Union, List, Tuple

# --- Simple AST for expressions ---
class Expr:
    pass

class Var(Expr):
    def __init__(self, name: str):
        self.name = name

class Const(Expr):
    def __init__(self, value: float):
        self.value = value

class Add(Expr):
    def __init__(self, left: Expr, right: Expr):
        self.left = left
        self.right = right

class Sub(Expr):
    def __init__(self, left: Expr, right: Expr):
        self.left = left
        self.right = right

class Mul(Expr):
    def __init__(self, left: Expr, right: Expr):
        self.left = left
        self.right = right

class Div(Expr):
    def __init__(self, left: Expr, right: Expr):
        self.left = left
        self.right = right

class Pow(Expr):
    def __init__(self, base: Expr, exp: Expr):
        self.base = base
        self.exp = exp

class Neg(Expr):
    def __init__(self, arg: Expr):
        self.arg = arg

class Sin(Expr):
    def __init__(self, arg: Expr):
        self.arg = arg

class Cos(Expr):
    def __init__(self, arg: Expr):
        self.arg = arg

class Tan(Expr):
    def __init__(self, arg: Expr):
        self.arg = arg

class Exp(Expr):
    def __init__(self, arg: Expr):
        self.arg = arg

class Log(Expr):
    def __init__(self, arg: Expr):
        self.arg = arg

class Sqrt(Expr):
    def __init__(self, arg: Expr):
        self.arg = arg


# --- Tokenizer ---
_TOKEN_SPEC = [
    ("NUMBER", r"\d+(\.\d+)?([eE][+-]?\d+)?"),
    ("ID", r"[a-zA-Z_][a-zA-Z0-9_]*"),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("MUL", r"\*"),
    ("DIV", r"/"),
    ("POW", r"\^"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("SKIP", r"[ \t]+"),
]
_TOKEN_RE = re.compile("|".join(f"(?P<{n}>{p})" for n, p in _TOKEN_SPEC))

def _tokenize(s: str) -> List[Tuple[str, str]]:
    tokens = []
    for m in _TOKEN_RE.finditer(s):
        kind = m.lastgroup
        value = m.group()
        if kind == "SKIP":
            continue
        if kind == "NUMBER":
            value = float(value)
        tokens.append((kind, value))
    return tokens


# --- Parser (recursive descent) ---
class _Parser:
    def __init__(self, tokens: List[Tuple[str, Union[str, float]]]):
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> Optional[Tuple[str, Union[str, float]]]:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def _advance(self) -> Optional[Tuple[str, Union[str, float]]]:
        if self.pos >= len(self.tokens):
            return None
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def _parse_expr(self) -> Expr:
        left = self._parse_term()
        while True:
            p = self._peek()
            if p is None:
                break
            if p[0] == "PLUS":
                self._advance()
                left = Add(left, self._parse_term())
            elif p[0] == "MINUS":
                self._advance()
                left = Sub(left, self._parse_term())
            else:
                break
        return left

    def _parse_term(self) -> Expr:
        left = self._parse_factor()
        while True:
            p = self._peek()
            if p is None:
                break
            if p[0] == "MUL":
                self._advance()
                left = Mul(left, self._parse_factor())
            elif p[0] == "DIV":
                self._advance()
                left = Div(left, self._parse_factor())
            else:
                break
        return left

    def _parse_factor(self) -> Expr:
        base = self._parse_unary()
        p = self._peek()
        if p is not None and p[0] == "POW":
            self._advance()
            exp = self._parse_factor()
            return Pow(base, exp)
        return base

    def _parse_unary(self) -> Expr:
        p = self._peek()
        if p is None:
            raise ValueError("Unerwartetes Ende des Ausdrucks")
        if p[0] == "PLUS":
            self._advance()
            return self._parse_unary()
        if p[0] == "MINUS":
            self._advance()
            return Neg(self._parse_unary())
        return self._parse_atom()

    def _parse_atom(self) -> Expr:
        p = self._advance()
        if p is None:
            raise ValueError("Unerwartetes Ende des Ausdrucks")
        kind, value = p
        if kind == "NUMBER":
            return Const(float(value))
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
                        return Sin(arg)
                    if name == "cos":
                        return Cos(arg)
                    if name == "tan":
                        return Tan(arg)
                    if name == "exp":
                        return Exp(arg)
                    if name == "log":
                        return Log(arg)
                    if name == "sqrt":
                        return Sqrt(arg)
            return Var(name)
        if kind == "LPAREN":
            e = self._parse_expr()
            rp = self._advance()
            if rp is None or rp[0] != "RPAREN":
                raise ValueError("Fehlende ')'")
            return e
        raise ValueError(f"Unerwartetes Token: {p}")


def _parse(expr_str: str) -> Expr:
    s = expr_str.strip().replace(" ", "")
    if not s:
        raise ValueError("Empty expression")
    tokens = _tokenize(s)
    if not tokens:
        raise ValueError("No valid tokens")
    p = _Parser(tokens)
    e = p._parse_expr()
    if p.pos < len(tokens):
        raise ValueError("Trailing content after expression")
    return e


# --- Differentiation (d/dvar) ---
def _diff(e: Expr, var: str) -> Expr:
    if isinstance(e, Var):
        return Const(1.0) if e.name == var else Const(0.0)
    if isinstance(e, Const):
        return Const(0.0)
    if isinstance(e, Add):
        return Add(_diff(e.left, var), _diff(e.right, var))
    if isinstance(e, Sub):
        return Sub(_diff(e.left, var), _diff(e.right, var))
    if isinstance(e, Mul):
        return Add(
            Mul(_diff(e.left, var), e.right),
            Mul(e.left, _diff(e.right, var))
        )
    if isinstance(e, Div):
        # (u/v)' = (u'v - uv')/v^2
        return Div(
            Sub(Mul(_diff(e.left, var), e.right), Mul(e.left, _diff(e.right, var))),
            Pow(e.right, Const(2.0))
        )
    if isinstance(e, Pow):
        # d/dx(f^g): f^g * (g'*log(f) + g*f'/f)  when f,g depend on x
        # Simple case: base^const -> const*base^(const-1)*base'
        if isinstance(e.exp, Const):
            c = e.exp.value
            if c == 0:
                return Const(0.0)
            if c == 1:
                return _diff(e.base, var)
            return Mul(Const(c), Mul(Pow(e.base, Const(c - 1)), _diff(e.base, var)))
        # exp^g: (f^g)' = f^g * (g'*log(f) + g*f'/f)
        return Mul(
            Pow(e.base, e.exp),
            Add(
                Mul(_diff(e.exp, var), Log(e.base)),
                Mul(e.exp, Div(_diff(e.base, var), e.base))
            )
        )
    if isinstance(e, Neg):
        return Neg(_diff(e.arg, var))
    if isinstance(e, Sin):
        return Mul(Cos(e.arg), _diff(e.arg, var))
    if isinstance(e, Cos):
        return Mul(Neg(Sin(e.arg)), _diff(e.arg, var))
    if isinstance(e, Tan):
        # tan' = 1/cos^2 * arg' = (1+tan^2)*arg'
        return Mul(Add(Const(1.0), Pow(Tan(e.arg), Const(2.0))), _diff(e.arg, var))
    if isinstance(e, Exp):
        return Mul(Exp(e.arg), _diff(e.arg, var))
    if isinstance(e, Log):
        return Mul(Div(Const(1.0), e.arg), _diff(e.arg, var))
    if isinstance(e, Sqrt):
        # sqrt(x) = x^(1/2), (sqrt(x))' = 1/(2*sqrt(x)) * x'
        return Mul(Div(Const(1.0), Mul(Const(2.0), Sqrt(e.arg))), _diff(e.arg, var))
    raise TypeError(f"Unknown expression type: {type(e)}")


# --- AST to string (readable, with parentheses where needed) ---
def _to_string(e: Expr, parent_prec: int = 0) -> str:
    """parent_prec: 0=expr, 1=term, 2=factor, 3=unary, 4=atom."""
    if isinstance(e, Var):
        return e.name
    if isinstance(e, Const):
        v = e.value
        if v == int(v):
            return str(int(v))
        return str(v)
    if isinstance(e, Add):
        s = _to_string(e.left, 1) + " + " + _to_string(e.right, 1)
        return f"({s})" if parent_prec > 0 else s
    if isinstance(e, Sub):
        s = _to_string(e.left, 1) + " - " + _to_string(e.right, 2)
        return f"({s})" if parent_prec > 0 else s
    if isinstance(e, Mul):
        left = _to_string(e.left, 2)
        right = _to_string(e.right, 2)
        if isinstance(e.left, (Add, Sub)):
            left = f"({left})"
        if isinstance(e.right, (Add, Sub)):
            right = f"({right})"
        s = f"{left}*{right}"
        return f"({s})" if parent_prec > 1 else s
    if isinstance(e, Div):
        num = _to_string(e.left, 2)
        den = _to_string(e.right, 2)
        if isinstance(e.left, (Add, Sub)):
            num = f"({num})"
        if isinstance(e.right, (Add, Sub, Mul)):
            den = f"({den})"
        s = f"{num}/{den}"
        return f"({s})" if parent_prec > 1 else s
    if isinstance(e, Pow):
        base = _to_string(e.base, 3)
        exp = _to_string(e.exp, 3)
        if isinstance(e.base, (Add, Sub, Mul, Div)):
            base = f"({base})"
        s = f"{base}^{exp}"
        return f"({s})" if parent_prec > 2 else s
    if isinstance(e, Neg):
        return "-" + _to_string(e.arg, 3)
    if isinstance(e, Sin):
        return "sin(" + _to_string(e.arg, 0) + ")"
    if isinstance(e, Cos):
        return "cos(" + _to_string(e.arg, 0) + ")"
    if isinstance(e, Tan):
        return "tan(" + _to_string(e.arg, 0) + ")"
    if isinstance(e, Exp):
        return "exp(" + _to_string(e.arg, 0) + ")"
    if isinstance(e, Log):
        return "log(" + _to_string(e.arg, 0) + ")"
    if isinstance(e, Sqrt):
        return "sqrt(" + _to_string(e.arg, 0) + ")"
    return "?"


def _simplify(e: Expr) -> Expr:
    """Einfache Vereinfachung: 1*x -> x, 0+x -> x, 0*x -> 0, etc."""
    if isinstance(e, Const):
        return e
    if isinstance(e, Var):
        return e
    if isinstance(e, Add):
        l, r = _simplify(e.left), _simplify(e.right)
        if isinstance(l, Const) and l.value == 0:
            return r
        if isinstance(r, Const) and r.value == 0:
            return l
        return Add(l, r)
    if isinstance(e, Sub):
        l, r = _simplify(e.left), _simplify(e.right)
        if isinstance(r, Const) and r.value == 0:
            return l
        return Sub(l, r)
    if isinstance(e, Mul):
        l, r = _simplify(e.left), _simplify(e.right)
        if isinstance(l, Const):
            if l.value == 0:
                return Const(0.0)
            if l.value == 1:
                return r
        if isinstance(r, Const):
            if r.value == 0:
                return Const(0.0)
            if r.value == 1:
                return l
        return Mul(l, r)
    if isinstance(e, Div):
        l, r = _simplify(e.left), _simplify(e.right)
        if isinstance(l, Const) and l.value == 0:
            return Const(0.0)
        return Div(l, r)
    if isinstance(e, Pow):
        b, x = _simplify(e.base), _simplify(e.exp)
        if isinstance(x, Const):
            if x.value == 0:
                return Const(1.0)
            if x.value == 1:
                return b
        return Pow(b, x)
    if isinstance(e, Neg):
        a = _simplify(e.arg)
        if isinstance(a, Const):
            return Const(-a.value)
        return Neg(a)
    if isinstance(e, (Sin, Cos, Tan, Exp, Log, Sqrt)):
        a = _simplify(e.arg)
        return type(e)(a)
    return e


def diff_sym(expr: str, var: str) -> str:
    """
    Symbolic differentiation: differentiates the expression expr with respect to var.
    expr: string, e.g. "x^2 + sin(x)", "exp(x)*log(x)".
    var: name of the variable, e.g. "x".
    Returns: derivative as string.
    Supports: +, -, *, /, ^, sin, cos, tan, exp, log, sqrt; constants and variables.
    """
    expr = str(expr).strip()
    var = str(var).strip()
    if not var:
        raise ValueError("Variable must not be empty")
    ast = _parse(expr)
    d = _diff(ast, var)
    d = _simplify(d)
    return _to_string(d)
