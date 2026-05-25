"""
Compile-time unit check (Phase 3b).
1[m] + 1[s] -> CompileError with line. Prevents unit bugs before runtime.
"""
from typing import Optional, Dict, Any, List
from .ast_nodes import (
    Node, Program, BinaryOp, Literal, Quantity, Identifier,
    Call, MemberAccess, ReturnStmt, Assignment, ItemAssignment, FunctionDef,
    IfStmt, WhileStmt, ForStmt, VectorLiteral, Subscript, UnitDef,
    CompileError,
)
from .ml_runtime import ADDITIVE_DIMENSION_UNIT_SETS, _parse_unit_to_base_dims, _register_user_unit

# Known constants and their units (matches ml_runtime)
KNOWN_UNITS: Dict[str, str] = {
    "c": "m/s", "G": "m^3/(kg*s^2)", "h": "J*s", "hbar": "J*s",
    "k_B": "J/K", "k_e": "N*m^2/C^2", "e_charge": "C", "epsilon_0": "F/m",
    "mu_0": "N/A^2", "m_e": "kg", "m_p": "kg", "N_A": "1/mol", "R_gas": "J/(K*mol)",
    "alpha": "", "sigma_SB": "W/(m^2*K^4)", "F_faraday": "C/mol",
    "g_0": "m/s^2", "m_n": "kg", "mu_B": "J/T", "R_inf": "1/m",
    "pi": "", "e": "",
}

class Environment:
    """Tracks local and global variables and their inferred units."""
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
    """Normalizes chemical units for comparison: mol/L and M are treated as equal."""
    if u is None or u == "":
        return u
    u = u.strip()
    if u in ("M", "mol/L", "mol*L^-1", "mol*L^(-1)"):
        return "M"
    return u


def _same_dimension(u1: Optional[str], u2: Optional[str]) -> bool:
    """True if both units are addable/subtractable."""
    if u1 is None or u2 is None:
        return False
    u1 = (u1 or "").strip()
    u2 = (u2 or "").strip()
    if u1 == u2:
        return True
        
    # Logarithmic units special case: absolute level +/- ratio (dB)
    log_units = {"dB", "dB_power", "dB_amp", "dBW", "dBm", "dBV", "dBuV", "dBSPL"}
    if u1 in log_units or u2 in log_units:
        if u1 in log_units and u2 in ("dB", "dB_power", "dB_amp") and u1 not in ("dB", "dB_power", "dB_amp"):
            return True
        if u2 in log_units and u1 in ("dB", "dB_power", "dB_amp") and u2 not in ("dB", "dB_power", "dB_amp"):
            return True

    n1 = _normalize_chemical_unit(u1)
    n2 = _normalize_chemical_unit(u2)
    if n1 is not None and n2 is not None and n1 == n2:
        return True

    # Compare canonical dimensions (SI base)
    d1 = _parse_unit_to_base_dims(u1)
    d2 = _parse_unit_to_base_dims(u2)
    # _parse_unit_to_base_dims returns None on parse error
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
    """Infers the unit of an expression; None = unknown."""
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
            if left_u is not None and right_u is not None:
                if not _same_dimension(left_u, right_u):
                    return None  # Signal: mismatch, handled in check
                
                # Temperature units (affine offset and difference)
                affine_units = {"degC", "°C", "degF", "°F"}
                d1 = _parse_unit_to_base_dims(left_u)
                d2 = _parse_unit_to_base_dims(right_u)
                if d1 == {"K": 1} and d2 == {"K": 1}:
                    is_aff1 = left_u in affine_units
                    is_aff2 = right_u in affine_units
                    if is_aff1 and is_aff2:
                        if node.op == "+":
                            return None  # Compile error: P + P is invalid
                        # Subtraction: P - P -> Difference
                        if left_u in ("degF", "°F"):
                            return "delta_F"
                        return "K"
                    elif is_aff1:
                        # P +/- V -> P
                        return left_u
                    elif is_aff2:
                        # V + P -> P, but V - P is invalid
                        if node.op == "-":
                            return None
                        return right_u
                
                log_units = {"dB", "dB_power", "dB_amp", "dBW", "dBm", "dBV", "dBuV", "dBSPL"}
                if left_u in log_units or right_u in log_units:
                    if left_u in ("dB", "dB_power", "dB_amp") and right_u in ("dB", "dB_power", "dB_amp"):
                        return left_u
                    if node.op == "-" and left_u in log_units and right_u in log_units and left_u not in ("dB", "dB_power", "dB_amp") and right_u not in ("dB", "dB_power", "dB_amp"):
                        return "dB"
                    if left_u in log_units and left_u not in ("dB", "dB_power", "dB_amp") and right_u in ("dB", "dB_power", "dB_amp"):
                        return left_u
                    if right_u in log_units and right_u not in ("dB", "dB_power", "dB_amp") and left_u in ("dB", "dB_power", "dB_amp"):
                        return right_u
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
    """Checks an expression recursively; raises CompileError on unit mismatch."""
    if node is None:
        return
    if isinstance(node, BinaryOp):
        _check_expr(node.left, env, filepath)
        _check_expr(node.right, env, filepath)
        if node.op in ("+", "-"):
            left_u = _infer_unit(node.left, env)
            right_u = _infer_unit(node.right, env)
            # Unary minus: 0 - x -> allowed (result has unit of x)
            if node.op == "-" and isinstance(node.left, Literal):
                try:
                    if float(getattr(node.left, "value", 1)) == 0:
                        return
                except (TypeError, ValueError):
                    pass
            # Check dimension
            if left_u is not None and right_u is not None and not _same_dimension(left_u, right_u):
                line = getattr(node, "line", None)
                raise CompileError(
                    f"Units do not match for {node.op}: [{left_u}] vs [{right_u}]. "
                    "Same unit or compatible dimension (length/mass/time/pressure/angle rad/deg) required.",
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
        return  # already registered in check_units() pre-pass
    if isinstance(stmt, Assignment):
        _check_expr(stmt.value, env, filepath)
        # Type inference for variable assignment
        if isinstance(stmt.target, str):
            inferred_unit = _infer_unit(stmt.value, env)
            env.set(stmt.target, inferred_unit)
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
    Traverses the AST and raises CompileError if + or - is applied
    to two quantities with incompatible units.
    """
    # Pre-pass: register all UnitDef statements so the checker knows new units
    for stmt in ast.statements:
        if isinstance(stmt, UnitDef):
            try:
                _register_user_unit(stmt.name, stmt.factor, stmt.base_unit)
            except Exception as e:
                raise CompileError(
                    f"Failed to register unit: {e}",
                    line=getattr(stmt, "line", None),
                    filepath=filepath,
                )
    global_env = Environment()
    for stmt in ast.statements:
        _check_stmt(stmt, global_env, filepath)

