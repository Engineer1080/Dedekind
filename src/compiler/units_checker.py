"""
Einheiten-Check zur Compile-Zeit (Phase 3b).
1[m] + 1[s] → CompileError mit Zeile. Verhindert Unit-Bugs vor dem Lauf.
"""
from typing import Optional, Dict, Any
from .ast_nodes import (
    Node, Program, BinaryOp, Literal, Quantity, Identifier,
    Call, MemberAccess, ReturnStmt, Assignment, ItemAssignment, FunctionDef,
    IfStmt, WhileStmt, ForStmt, VectorLiteral, Subscript,
    CompileError,
)

# Bekannte Konstanten und ihre Einheiten (wie in ml_runtime)
KNOWN_UNITS: Dict[str, str] = {
    "c": "m/s", "G": "m^3/(kg*s^2)", "h": "J*s", "hbar": "J*s",
    "k_B": "J/K", "k_e": "N*m^2/C^2", "e_charge": "C", "epsilon_0": "F/m",
    "mu_0": "N/A^2", "m_e": "kg", "m_p": "kg", "N_A": "1/mol", "R_gas": "J/(K*mol)",
    "alpha": "", "sigma_SB": "W/(m^2*K^4)", "F_faraday": "C/mol",
    "pi": "", "e": "",
}


def _unit_mul(u1: str, u2: str) -> str:
    if not u1:
        return u2
    if not u2:
        return u1
    return f"({u1})*({u2})"


def _unit_div(u1: str, u2: str) -> str:
    if not u2:
        return u1
    if not u1:
        return f"1/({u2})"
    return f"({u1})/({u2})"


def _unit_pow(u: str, exp: Any) -> str:
    try:
        e = float(exp)
    except (TypeError, ValueError):
        return u or ""
    if not u:
        return ""
    if abs(e + 1.0) < 1e-12:
        return f"1/({u})"
    base = f"({u})" if "*" in u or "/" in u else u
    if abs(e - round(e)) < 1e-12:
        return "*".join([base] * int(round(e)))
    return f"{base}^{e}"


def _infer_unit(node: Node) -> Optional[str]:
    """Inferiert die Einheit eines Ausdrucks; None = unbekannt."""
    if node is None:
        return None
    if isinstance(node, Literal):
        return ""
    if isinstance(node, Quantity):
        return (node.unit or "").strip()
    if isinstance(node, Identifier):
        return KNOWN_UNITS.get(node.name)
    if isinstance(node, BinaryOp):
        left_u = _infer_unit(node.left)
        right_u = _infer_unit(node.right)
        if node.op in ("+", "-"):
            if left_u is not None and right_u is not None and left_u != right_u:
                return None  # Signal: mismatch, handled in check
            return left_u if left_u is not None else right_u
        if node.op in ("*", "**"):
            lu = left_u or ""
            ru = right_u or ""
            if node.op == "**":
                return _unit_pow(lu, getattr(node.right, "value", 0))
            return _unit_mul(lu, ru)
        if node.op == "/":
            lu = left_u or ""
            ru = right_u or ""
            return _unit_div(lu, ru)
        if node.op == "^":
            lu = left_u or ""
            ru = getattr(node.right, "value", None)
            if ru is None and isinstance(node.right, Literal):
                ru = node.right.value
            return _unit_pow(lu, ru)
    return None


def _check_expr(node: Node, filepath: Optional[str]) -> None:
    """Prüft einen Ausdruck rekursiv; wirft CompileError bei Einheiten-Mismatch."""
    if node is None:
        return
    if isinstance(node, BinaryOp):
        _check_expr(node.left, filepath)
        _check_expr(node.right, filepath)
        if node.op in ("+", "-"):
            left_u = _infer_unit(node.left)
            right_u = _infer_unit(node.right)
            # Unäres Minus: 0 - x → erlaubt (Ergebnis hat Einheit von x)
            if node.op == "-" and isinstance(node.left, Literal):
                try:
                    if float(getattr(node.left, "value", 1)) == 0:
                        return
                except (TypeError, ValueError):
                    pass
            if left_u is not None and right_u is not None and left_u != right_u:
                line = getattr(node, "line", None)
                raise CompileError(
                    f"Einheiten passen nicht für {node.op}: [{left_u}] vs [{right_u}]. "
                    "Für Addition/Subtraktion muss die gleiche Einheit verwendet werden.",
                    line=line,
                    filepath=filepath,
                )
        return
    if isinstance(node, (Call, MemberAccess, Subscript)):
        for child in [getattr(node, "left", None), getattr(node, "right", None),
                      getattr(node, "value", None), getattr(node, "index", None),
                      getattr(node, "obj", None), getattr(node, "func_name", None)]:
            if child is not None:
                _check_expr(child, filepath)
        for arg in getattr(node, "args", []) or []:
            _check_expr(arg, filepath)
        return
    if isinstance(node, ReturnStmt):
        _check_expr(node.value, filepath)
        return
    if isinstance(node, Assignment):
        _check_expr(node.value, filepath)
        return
    if isinstance(node, (IfStmt, WhileStmt)):
        _check_expr(node.condition, filepath)
        for stmt in node.then_branch if hasattr(node, "then_branch") else (node.body if hasattr(node, "body") else []):
            _check_stmt(stmt, filepath)
        if getattr(node, "else_branch", None):
            for stmt in node.else_branch:
                _check_stmt(stmt, filepath)
        return
    if isinstance(node, ForStmt):
        _check_expr(node.collection, filepath)
        for stmt in node.body:
            _check_stmt(stmt, filepath)
        return
    if isinstance(node, VectorLiteral):
        for e in node.elements:
            _check_expr(e, filepath)
        return
    if isinstance(node, FunctionDef):
        for stmt in node.body:
            _check_stmt(stmt, filepath)
        return


def _check_stmt(stmt: Node, filepath: Optional[str]) -> None:
    if stmt is None:
        return
    if isinstance(stmt, (ReturnStmt, Assignment, ItemAssignment)):
        _check_expr(getattr(stmt, "value", None), filepath)
        return
    if isinstance(stmt, (IfStmt, WhileStmt, ForStmt, FunctionDef)):
        _check_expr(stmt, filepath)
        return
    _check_expr(stmt, filepath)  # expression statement (Call, BinaryOp, etc.)


def check_units(ast: Program, filepath: Optional[str] = None) -> None:
    """
    Durchläuft das AST und wirft CompileError, wenn bei + oder -
    zwei Größen mit unterschiedlichen Einheiten verwendet werden (soweit zur
    Compile-Zeit bekannt).
    """
    for stmt in ast.statements:
        _check_stmt(stmt, filepath)
