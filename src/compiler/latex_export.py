"""
LaTeX-Export von Fourier-Ausdrücken (AST → LaTeX).
Für Papers und Notizen: Formeln aus Fourier-Code als LaTeX erzeugen.
"""
from .ast_nodes import (
    Node, Program, BinaryOp, Call, Identifier, Literal, Quantity,
    MemberAccess, IndexedVariable, Subscript, VectorLiteral,
    Assignment, ReturnStmt, FunctionDef, Lambda, QuaternionLiteral,
)


# Bekannte Funktionen → LaTeX-Befehl (ohne Backslash)
_LATEX_FUNCS = {
    "sin": "sin", "cos": "cos", "tan": "tan",
    "exp": "exp", "log": "log", "log10": "log_{10}", "sqrt": "sqrt", "abs": "abs",
    "asin": "arcsin", "acos": "arccos", "atan": "arctan", "atan2": "atan2",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "diff": "frac", "grad": "nabla",
}


def _escape_latex(s: str) -> str:
    """Sonderzeichen für LaTeX escapen."""
    return (
        str(s)
        .replace("\\", "\\backslash ")
        .replace("_", "\\_")
        .replace("^", "\\^{}")
        .replace("#", "\\#")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


class LatexExporter:
    """Visitor: wandelt AST-Knoten in LaTeX-Strings um (für Formeln)."""

    def visit(self, node: Node) -> str:
        name = type(node).__name__
        method = getattr(self, f"visit_{name}", None)
        if method is not None:
            return method(node)
        return "?"

    def visit_Program(self, node: Program) -> str:
        lines = []
        for stmt in node.statements:
            line = self.visit(stmt)
            if line:
                lines.append(line)
        return "\n".join(lines)

    def visit_Assignment(self, node: Assignment) -> str:
        right = self.visit(node.value)
        name = _escape_latex(node.target)
        return f"{name} = {right}"

    def visit_ReturnStmt(self, node: ReturnStmt) -> str:
        return self.visit(node.value)

    def visit_FunctionDef(self, node: FunctionDef) -> str:
        return ""  # Nur Körper ggf. später; hier nur Ausdrücke

    def visit_BinaryOp(self, node: BinaryOp) -> str:
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = node.op
        if op == "^":
            return f"{{{left}}}^{{{right}}}"
        if op == "**":
            return f"{{{left}}}^{{{right}}}"
        if op == "*":
            if left == "1" or right == "1":
                return right if left == "1" else left
            return f"{left} \\cdot {right}"
        if op == "/":
            return f"\\frac{{{left}}}{{{right}}}"
        if op in ("+", "-"):
            return f"{left} {op} {right}"
        cmp_map = {"<": "\\lt", ">": "\\gt", "<=": "\\leq", ">=": "\\geq", "==": "=", "!=": "\\neq"}
        if op in cmp_map:
            return f"{left} \\mathrel{{{cmp_map[op]}}} {right}"
        return f"{left} \\mathbin{{{_escape_latex(op)}}} {right}"

    def visit_Literal(self, node: Literal) -> str:
        v = node.value
        if isinstance(v, str):
            return f"\\text{{{_escape_latex(v)}}}"
        if isinstance(v, bool):
            return "\\mathrm{true}" if v else "\\mathrm{false}"
        return str(v)

    def visit_Quantity(self, node: Quantity) -> str:
        unit = _escape_latex(node.unit).strip()
        if unit:
            return f"{node.value} \\, \\mathrm{{{unit}}}"
        return str(node.value)

    def visit_Identifier(self, node: Identifier) -> str:
        n = node.name
        if len(n) <= 2 and n.isalpha():
            return n
        return f"\\mathit{{{_escape_latex(n)}}}"

    def visit_IndexedVariable(self, node: IndexedVariable) -> str:
        base = _escape_latex(node.name)
        idx = node.indices
        if len(idx) == 1:
            return f"{base}_{{{idx}}}"
        return f"{base}_{{{idx}}}"

    def visit_Subscript(self, node: Subscript) -> str:
        val = self.visit(node.value)
        idx = self.visit(node.index)
        return f"{{{val}}}_{{{idx}}}"

    def visit_Call(self, node: Call) -> str:
        func = node.func_name
        if isinstance(func, Identifier):
            fname = func.name
        elif isinstance(func, MemberAccess):
            fname = func.member
        else:
            fname = self.visit(func)
        args = [self.visit(a) for a in node.args]
        fname_lower = fname.lower() if isinstance(fname, str) else ""
        if fname_lower in _LATEX_FUNCS:
            cmd = _LATEX_FUNCS[fname_lower]
            if cmd == "sqrt":
                return f"\\sqrt{{{args[0]}}}" if args else "\\sqrt{}"
            if cmd == "abs":
                return f"\\left| {args[0]} \\right|" if args else "|"
            if cmd == "atan2" and len(args) >= 2:
                return f"\\operatorname{{atan2}}\\left( {args[0]}, {args[1]} \\right)"
            if cmd == "frac" and len(args) >= 2:
                return f"\\frac{{{args[0]}}}{{{args[1]}}}"
            if cmd == "log_{10}" and args:
                return f"\\log_{{10}}\\left( {args[0]} \\right)"
            if args:
                return f"\\{cmd}\\left( {', '.join(args)} \\right)"
            return f"\\{cmd}"
        if args:
            return f"\\mathrm{{{_escape_latex(str(fname))}}}\\left( {', '.join(args)} \\right)"
        return f"\\mathrm{{{_escape_latex(str(fname))}}}"

    def visit_MemberAccess(self, node: MemberAccess) -> str:
        obj = self.visit(node.obj)
        return f"{obj}.\\mathrm{{{_escape_latex(node.member)}}}"

    def visit_VectorLiteral(self, node: VectorLiteral) -> str:
        els = [self.visit(e) for e in node.elements]
        return "\\left( " + ", ".join(els) + " \\right)"

    def visit_Lambda(self, node: Lambda) -> str:
        body = self.visit(node.body)
        return f"\\lambda {_escape_latex(node.arg)}. \\, {body}"

    def visit_QuaternionLiteral(self, node: QuaternionLiteral) -> str:
        c = node.component
        v = node.value
        if c in ("i", "j", "k"):
            return f"{v} \\mathbf{{{c}}}"
        return str(v)


def expression_to_latex(node: Node) -> str:
    """Einen einzelnen AST-Knoten (typisch Expression) als LaTeX zurückgeben."""
    return LatexExporter().visit(node)


def program_to_latex(node: Program) -> str:
    """Gesamtes Program als LaTeX (Gleichungen/Statements)."""
    return LatexExporter().visit(node)
