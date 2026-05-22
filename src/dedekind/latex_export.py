"""
LaTeX-Export von Dedekind-Ausdrücken (AST → LaTeX).
Für Papers und Notizen: Formeln aus Dedekind-Code als LaTeX erzeugen.
"""
from typing import List
from .ast_nodes import (
    Node, Program, BinaryOp, Call, Identifier, Literal, Quantity,
    MemberAccess, IndexedVariable, Subscript, VectorLiteral,
    Assignment, ReturnStmt, FunctionDef, Lambda, QuaternionLiteral,
    IfStmt, WhileStmt, ForStmt,
)


# Bekannte Funktionen → LaTeX-Befehl (ohne Backslash)
_LATEX_FUNCS = {
    "sin": "sin", "cos": "cos", "tan": "tan",
    "exp": "exp", "log": "log", "log10": "log_{10}", "sqrt": "sqrt", "abs": "abs",
    "asin": "arcsin", "acos": "arccos", "atan": "arctan", "atan2": "atan2",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "min": "min", "max": "max", "argmin": "argmin", "argmax": "argmax",
    "round": "round", "floor": "floor", "ceil": "ceil",
    "norm": "norm", "det": "det", "trace": "operatorname{tr}",
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
        return self._visit_body(node.body)

    def _visit_body(self, stmts: List[Node]) -> str:
        lines = []
        for s in stmts:
            line = self.visit(s)
            if line:
                lines.append(line)
        return "\n".join(lines)

    def visit_IfStmt(self, node: IfStmt) -> str:
        parts = [self._visit_body(node.then_branch)]
        if node.else_branch:
            parts.append(self._visit_body(node.else_branch))
        return "\n".join(p for p in parts if p)

    def visit_WhileStmt(self, node: WhileStmt) -> str:
        return self._visit_body(node.body)

    def visit_ForStmt(self, node: ForStmt) -> str:
        return self._visit_body(node.body)

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
            if isinstance(node.left, IndexedVariable) and isinstance(node.right, IndexedVariable):
                return f"{left}\\, {right}"
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
        return f"{base}^{{{idx}}}"

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

        # Domain-specific scientific rendering ---------------------------------
        if fname == "partial" and len(args) >= 2:
            u, x = args[0], args[1]
            order = 1
            for k, v in (node.kwargs or []):
                if k == "order" and isinstance(v, Literal) and isinstance(v.value, (int, float)):
                    order = int(v.value)
            if order == 1:
                return f"\\frac{{\\partial {u}}}{{\\partial {x}}}"
            return f"\\frac{{\\partial^{{{order}}} {u}}}{{\\partial {x}^{{{order}}}}}"

        # PDE solver calls -> canonical PDE in LaTeX. Operand list intentionally
        # ignored: the equation is fixed by the solver name; arguments are u0,
        # boundary data, parameters, grid -- all numerical, not symbolic.
        _PDE_FORMS = {
            "pde_heat_1d":  "\\partial_t u = \\alpha\\, \\partial_x^2 u",
            "pde_heat_2d":  "\\partial_t u = \\alpha\\, \\nabla^2 u",
            "pde_wave_1d":  "\\partial_t^2 u = c^2\\, \\partial_x^2 u",
            "pde_wave_2d":  "\\partial_t^2 u = c^2\\, \\nabla^2 u",
            "pde_advection_1d": "\\partial_t u + v\\, \\partial_x u = 0",
            "pde_advection_2d": "\\partial_t u + \\mathbf{v} \\cdot \\nabla u = 0",
            "pde_burgers_1d":   "\\partial_t u + u\\, \\partial_x u = \\nu\\, \\partial_x^2 u",
            "pde_burgers_2d":   "\\partial_t \\mathbf{u} + (\\mathbf{u} \\cdot \\nabla)\\mathbf{u} = \\nu\\, \\nabla^2 \\mathbf{u}",
            "pde_reaction_diffusion_1d": "\\partial_t u = D\\, \\partial_x^2 u + f(u)",
            "pde_reaction_diffusion_2d": "\\partial_t u = D\\, \\nabla^2 u + f(u)",
            "pde_advection_diffusion_1d": "\\partial_t u + v\\, \\partial_x u = D\\, \\partial_x^2 u",
            "pde_advection_diffusion_2d": "\\partial_t u + \\mathbf{v} \\cdot \\nabla u = D\\, \\nabla^2 u",
            "pde_maxwell_1d": "\\partial_t \\mathbf{E} = c^2\\, \\nabla \\times \\mathbf{B},\\quad \\partial_t \\mathbf{B} = -\\nabla \\times \\mathbf{E}",
            "pde_maxwell_2d": "\\partial_t \\mathbf{E} = c^2\\, \\nabla \\times \\mathbf{B},\\quad \\partial_t \\mathbf{B} = -\\nabla \\times \\mathbf{E}",
            "pde_navier_stokes_2d": "\\partial_t \\mathbf{u} + (\\mathbf{u} \\cdot \\nabla)\\mathbf{u} = -\\nabla p + \\nu\\, \\nabla^2 \\mathbf{u},\\quad \\nabla \\cdot \\mathbf{u} = 0",
        }
        if fname in _PDE_FORMS:
            return _PDE_FORMS[fname]

        # ode_solve(f, y0, t)  ->  dy/dt = f(t, y)
        if fname == "ode_solve":
            return "\\frac{d\\mathbf{y}}{dt} = f(t, \\mathbf{y})"

        # Lagrangian / Hamiltonian wrappers expand to canonical equations.
        if fname == "lagrange_ode_rhs":
            L = args[0] if args else "L"
            return (f"\\frac{{d}}{{dt}} \\frac{{\\partial {L}}}{{\\partial \\dot q}} "
                    f"- \\frac{{\\partial {L}}}{{\\partial q}} = 0")
        if fname == "hamilton_ode_rhs":
            H = args[0] if args else "H"
            return (f"\\dot q = \\frac{{\\partial {H}}}{{\\partial p}},\\quad "
                    f"\\dot p = -\\frac{{\\partial {H}}}{{\\partial q}}")

        # grad(f, x)  ->  \nabla_x f
        if fname == "grad" and len(args) >= 2:
            return f"\\nabla_{{{args[1]}}} {args[0]}"
        # integrate(f, a, b, n)  ->  \int_a^b f(x)\, dx
        if fname == "integrate" and len(args) >= 3:
            return f"\\int_{{{args[1]}}}^{{{args[2]}}} {args[0]}\\, dx"
        # ----------------------------------------------------------------------

        if fname_lower in _LATEX_FUNCS:
            cmd = _LATEX_FUNCS[fname_lower]
            if cmd == "sqrt":
                return f"\\sqrt{{{args[0]}}}" if args else "\\sqrt{}"
            if cmd == "abs":
                return f"\\left| {args[0]} \\right|" if args else "|"
            if cmd == "norm" and args:
                return f"\\left\\| {args[0]} \\right\\|"
            if cmd == "floor" and args:
                return f"\\lfloor {args[0]} \\rfloor"
            if cmd == "ceil" and args:
                return f"\\lceil {args[0]} \\rceil"
            if cmd == "operatorname{tr}" and args:
                return f"\\operatorname{{tr}}\\left( {args[0]} \\right)"
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
