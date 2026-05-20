"""
Einheiten-Check zur Compile-Zeit (Phase 3b).
1[m] + 1[s] → CompileError mit Zeile. Verhindert Unit-Bugs vor dem Lauf.
"""
from typing import Optional, Dict, Any, List
from .ast_nodes import (
    Node, Program, BinaryOp, Literal, Quantity, Identifier,
    Call, MemberAccess, ReturnStmt, Assignment, ItemAssignment, FunctionDef,
    IfStmt, WhileStmt, ForStmt, VectorLiteral, Subscript, UnitDef,
    CompileError,
)
from .ml_runtime import ADDITIVE_DIMENSION_UNIT_SETS, _parse_unit_to_base_dims, _register_user_unit

# Bekannte Konstanten und ihre Einheiten (wie in ml_runtime)
KNOWN_UNITS: Dict[str, str] = {
    "c": "m/s", "G": "m^3/(kg*s^2)", "h": "J*s", "hbar": "J*s",
    "k_B": "J/K", "k_e": "N*m^2/C^2", "e_charge": "C", "epsilon_0": "F/m",
    "mu_0": "N/A^2", "m_e": "kg", "m_p": "kg", "N_A": "1/mol", "R_gas": "J/(K*mol)",
    "alpha": "", "sigma_SB": "W/(m^2*K^4)", "F_faraday": "C/mol",
    "pi": "", "e": "",
}

class Environment:
    """Trackt lokale und globale Variablen und ihre abgeleiteten Einheiten."""
    def __init__(self):
        self.scopes: List[Dict[str, Optional[str]]] = [{}]

    def get(self, name: str) -> Optional[str]:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return KNOWN_UNITS.get(name)

    def set(self, name: str, unit: Optional[str]):
        self.scopes[-1][name] = unit

    def push(self):
        self.scopes.append({})

    def pop(self):
        self.scopes.pop()


def _normalize_chemical_unit(u: Optional[str]) -> Optional[str]:
    """Normalisiert chemische Einheiten für Vergleich: mol/L und M gelten als gleich."""
    if u is None or u == "":
        return u
    u = u.strip()
    if u in ("M", "mol/L", "mol*L^-1", "mol*L^(-1)"):
        return "M"
    return u


def _same_dimension(u1: Optional[str], u2: Optional[str]) -> bool:
    """True, wenn beide Einheiten addierbar/subtrahierbar sind."""
    if u1 is None or u2 is None:
        return False
    u1 = (u1 or "").strip()
    u2 = (u2 or "").strip()
    if u1 == u2:
        return True
        
    n1 = _normalize_chemical_unit(u1)
    n2 = _normalize_chemical_unit(u2)
    if n1 is not None and n2 is not None and n1 == n2:
        return True

    # Kanonische Dimensionen (SI Basis) vergleichen
    d1 = _parse_unit_to_base_dims(u1)
    d2 = _parse_unit_to_base_dims(u2)
    # _parse_unit_to_base_dims gibt None zurück, wenn Parse-Error
    if d1 is not None and d2 is not None and d1 == d2:
        return True

    # Fallback
    for unit_set in ADDITIVE_DIMENSION_UNIT_SETS:
        if u1 in unit_set and u2 in unit_set:
            return True
    return False


def _unit_mul(u1: str, u2: str) -> str:
    if not u1: return u2
    if not u2: return u1
    return f"({u1})*({u2})"


def _unit_div(u1: str, u2: str) -> str:
    if not u2: return u1
    if not u1: return f"1/({u2})"
    return f"({u1})/({u2})"


def _unit_pow(u: str, exp: Any) -> str:
    try:
        e = float(exp)
    except (TypeError, ValueError):
        return u or ""
    if not u: return ""
    if abs(e + 1.0) < 1e-12:
        return f"1/({u})"
    base = f"({u})" if "*" in u or "/" in u else u
    if abs(e - round(e)) < 1e-12:
        return "*".join([base] * int(round(e)))
    return f"{base}^{e}"


def _infer_unit(node: Node, env: Environment) -> Optional[str]:
    """Inferiert die Einheit eines Ausdrucks; None = unbekannt."""
    if node is None:
        return None
    if isinstance(node, Literal):
        return ""
    if isinstance(node, Quantity):
        return (node.unit or "").strip()
    if isinstance(node, Identifier):
        return env.get(node.name)
    if isinstance(node, BinaryOp):
        left_u = _infer_unit(node.left, env)
        right_u = _infer_unit(node.right, env)
        if node.op in ("+", "-"):
            if left_u is not None and right_u is not None and not _same_dimension(left_u, right_u):
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


def _check_expr(node: Node, env: Environment, filepath: Optional[str]) -> None:
    """Prüft einen Ausdruck rekursiv; wirft CompileError bei Einheiten-Mismatch."""
    if node is None:
        return
    if isinstance(node, BinaryOp):
        _check_expr(node.left, env, filepath)
        _check_expr(node.right, env, filepath)
        if node.op in ("+", "-"):
            left_u = _infer_unit(node.left, env)
            right_u = _infer_unit(node.right, env)
            # Unäres Minus: 0 - x → erlaubt (Ergebnis hat Einheit von x)
            if node.op == "-" and isinstance(node.left, Literal):
                try:
                    if float(getattr(node.left, "value", 1)) == 0:
                        return
                except (TypeError, ValueError):
                    pass
            # Check Dimension
            if left_u is not None and right_u is not None and not _same_dimension(left_u, right_u):
                line = getattr(node, "line", None)
                raise CompileError(
                    f"Einheiten passen nicht für {node.op}: [{left_u}] vs [{right_u}]. "
                    "Gleiche Einheit oder kompatible Dimension (Länge/Masse/Zeit/Druck/Winkel rad/deg) erforderlich.",
                    line=line,
                    filepath=filepath,
                )
        return
    if isinstance(node, (Call, MemberAccess, Subscript)):
        for child in [getattr(node, "left", None), getattr(node, "right", None),
                      getattr(node, "value", None), getattr(node, "index", None),
                      getattr(node, "obj", None), getattr(node, "func_name", None)]:
            if child is not None:
                _check_expr(child, env, filepath)
        for arg in getattr(node, "args", []) or []:
            _check_expr(arg, env, filepath)
        return
    if isinstance(node, ReturnStmt):
        _check_expr(node.value, env, filepath)
        return
    if isinstance(node, Assignment):
        _check_expr(node.value, env, filepath)
        return
    if isinstance(node, (IfStmt, WhileStmt)):
        _check_expr(node.condition, env, filepath)
        env.push()
        for stmt in node.then_branch if hasattr(node, "then_branch") else (node.body if hasattr(node, "body") else []):
            _check_stmt(stmt, env, filepath)
        env.pop()
        if getattr(node, "else_branch", None):
            env.push()
            for stmt in node.else_branch:
                _check_stmt(stmt, env, filepath)
            env.pop()
        return
    if isinstance(node, ForStmt):
        _check_expr(node.collection, env, filepath)
        env.push()
        # In a real checker we would infer unit of iteration var
        for stmt in node.body:
            _check_stmt(stmt, env, filepath)
        env.pop()
        return
    if isinstance(node, VectorLiteral):
        for e in node.elements:
            _check_expr(e, env, filepath)
        return
    if isinstance(node, FunctionDef):
        env.push()
        arg_units = getattr(node, "arg_units", None)
        type_params = set(getattr(node, "type_params", []) or [])
        if arg_units:
            for arg_name, u in zip(node.args, arg_units):
                if u is not None and u not in type_params:
                    env.set(arg_name, u)
        for stmt in node.body:
            _check_stmt(stmt, env, filepath)
        env.pop()
        return


def _check_stmt(stmt: Node, env: Environment, filepath: Optional[str]) -> None:
    if stmt is None:
        return
    if isinstance(stmt, UnitDef):
        return  # bereits in check_units() vorab registriert
    if isinstance(stmt, Assignment):
        _check_expr(stmt.value, env, filepath)
        # Type Inference für Variablenzuweisung
        if isinstance(stmt.target, Identifier):
            inferred_unit = _infer_unit(stmt.value, env)
            env.set(stmt.target.name, inferred_unit)
        return
    if isinstance(stmt, ItemAssignment):
        _check_expr(stmt.value, env, filepath)
        return
    if isinstance(stmt, ReturnStmt):
        _check_expr(stmt.value, env, filepath)
        return
    if isinstance(stmt, (IfStmt, WhileStmt, ForStmt, FunctionDef)):
        _check_expr(stmt, env, filepath)
        return
    _check_expr(stmt, env, filepath)  # expression statement (Call, BinaryOp, etc.)


def check_units(ast: Program, filepath: Optional[str] = None) -> None:
    """
    Durchläuft das AST und wirft CompileError, wenn bei + oder -
    zwei Größen mit inkompatiblen Einheiten verwendet werden.
    """
    # Pre-Pass: alle UnitDef-Statements registrieren, damit Checker neue Units kennt
    for stmt in ast.statements:
        if isinstance(stmt, UnitDef):
            try:
                _register_user_unit(stmt.name, stmt.factor, stmt.base_unit)
            except Exception as e:
                raise CompileError(
                    f"Einheit registrieren fehlgeschlagen: {e}",
                    line=getattr(stmt, "line", None),
                    filepath=filepath,
                )
    global_env = Environment()
    for stmt in ast.statements:
        _check_stmt(stmt, global_env, filepath)

