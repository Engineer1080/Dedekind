from dataclasses import dataclass, field
from typing import List, Optional, Union, Any


class CompileError(Exception):
    """Compiler-Fehler mit Zeile und optionalem Kontext."""
    def __init__(self, message: str, line: Optional[int] = None, filepath: Optional[str] = None):
        self.message = message
        self.line = line
        self.filepath = filepath
        super().__init__(self._format())

    def _format(self) -> str:
        parts = []
        if self.filepath:
            parts.append(self.filepath)
        if self.line is not None:
            parts.append(f"Zeile {self.line}")
        parts.append(self.message)
        return ": ".join(str(p) for p in parts)


@dataclass
class Node:
    """Basis für alle AST-Knoten; line für Fehlermeldungen."""
    line: Optional[int] = field(default=None, kw_only=True)

@dataclass
class Program(Node):
    statements: List[Node]

@dataclass
class FunctionDef(Node):
    name: str
    args: List[str]
    body: List[Node]
    arg_units: Optional[List[Optional[str]]] = None  # Per-arg unit annotations ([m], [kg], …); None per slot = not annotated
    return_unit: Optional[str] = None  # Return unit annotation
    arg_shapes: Optional[List[Optional[List]]] = None  # Per-arg shape annotations (Vector[2], Tensor[batch,N]); None per slot = not annotated
    return_shape: Optional[List] = None  # Return shape annotation


@dataclass
class UseStmt(Node):
    """Modul-/Import-Anweisung: `use mathlib` lädt mathlib.ddk in den aktuellen Kompilier-Pass."""
    module: str


@dataclass
class UnitDef(Node):
    """Benutzerdefinierte Einheit: `unit Foot = 0.3048[m]` registriert Foot als Längeneinheit
    mit Umrechnungsfaktor 0.3048 zur Basis. base_unit muss bereits einer bekannten Dimension angehören."""
    name: str
    factor: float
    base_unit: str


@dataclass
class PyImport(Node):
    """Importiert ein Python-Modul aus dem PyPI-Ökosystem: `pyimport scipy.special as ss`.
    Erzeugt im generierten Code ein `import MODULE as ALIAS`."""
    module: str
    alias: str

@dataclass
class ReturnStmt(Node):
    value: Node

@dataclass
class Assignment(Node):
    target: str
    value: Node

@dataclass
class BinaryOp(Node):
    left: Node
    op: str
    right: Node

@dataclass
class UnaryOp(Node):
    """Unärer Operator, z.B. not x."""
    op: str
    operand: Node

@dataclass
class Call(Node):
    func_name: Node  # Can be an identifier or another call (chaining)
    args: List[Node]
    kwargs: List[Any]  # List of (name, value) tuples
    modifiers: List[str]  # e.g., ['gpu', 'cpu']

@dataclass
class Identifier(Node):
    name: str

@dataclass
class Literal(Node):
    value: Any
    raw: bool = False  # True = Raw-String (r"...") in Dedekind

@dataclass
class Quantity(Node):
    """Physikalische Größe: Zahl mit Einheit, z. B. 10[m], 5[m/s]."""
    value: Union[int, float]
    unit: str

@dataclass
class QuaternionLiteral(Node):
    value: float
    component: str # 'i', 'j', or 'k'

@dataclass
class VectorLiteral(Node):
    elements: List[Node]

@dataclass
class DictLiteral(Node):
    """Dict-Literal: {"key": value, "k2": v2} — wird zur Python-dict transpiliert."""
    keys: List[Node]
    values: List[Node]

@dataclass
class Lambda(Node):
    arg: str
    body: Node

@dataclass
class IfStmt(Node):
    condition: Node
    then_branch: List[Node]
    else_branch: Optional[List[Node]]

@dataclass
class WhileStmt(Node):
    condition: Node
    body: List[Node]

@dataclass
class ForStmt(Node):
    variable: str
    collection: Node
    body: List[Node]

@dataclass
class MemberAccess(Node):
    obj: Node
    member: str

@dataclass
class IndexedVariable(Node):
    name: str
    indices: str # String of indices, e.g., "ij"

@dataclass
class Subscript(Node):
    value: Node
    index: Node

@dataclass
class ItemAssignment(Node):
    target: Subscript
    value: Node

@dataclass
class PostfixFactorial(Node):
    """Postfix-Fakultät: n!"""
    operand: Node