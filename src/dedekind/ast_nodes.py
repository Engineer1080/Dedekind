from dataclasses import dataclass, field
from typing import List, Optional, Union, Any


class CompileError(Exception):
    """Compiler error with line and optional context."""
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
            parts.append(f"Line {self.line}")
        parts.append(self.message)
        return ": ".join(str(p) for p in parts)


@dataclass
class Node:
    """Base for all AST nodes; line for error messages."""
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
    is_pub: bool = False  # `pub fn` -> True; only relevant for module visibility
    type_params: List[str] = field(default_factory=list)  # `fn name<T, U>(...)`: polymorphic type parameters


@dataclass
class UseStmt(Node):
    """Module/Import statement: `use math` loads math.ddk into the current compilation pass."""
    module: str


@dataclass
class UnitDef(Node):
    """User-defined unit: `unit Foot = 0.3048[m]` registers Foot as a length unit
    with conversion factor 0.3048 to the base unit. base_unit must already belong to a known dimension."""
    name: str
    factor: float
    base_unit: str


@dataclass
class PyImport(Node):
    """Imports a Python module from the PyPI ecosystem: `pyimport scipy.special as ss`.
    Generates an `import MODULE as ALIAS` in the generated code."""
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
    """Unary operator, e.g., not x."""
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
    raw: bool = False  # True = Raw string (r"...") in Dedekind

@dataclass
class Quantity(Node):
    """Physical quantity: number with unit, e.g., 10[m], 5[m/s]."""
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
    """Dict literal: {"key": value, "k2": v2} — gets transpiled to a Python dict."""
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
class Slice(Node):
    """Python-style slice for subscript indices: x[start:stop:step].
    Each component can be None (open bound)."""
    start: Optional[Node] = None
    stop: Optional[Node] = None
    step: Optional[Node] = None


@dataclass
class TryCatch(Node):
    """try { body } catch var { handler } — catches every exception and binds it to var."""
    body: List[Node]
    catch_var: str
    handler: List[Node]

@dataclass
class ItemAssignment(Node):
    target: Subscript
    value: Node

@dataclass
class PostfixFactorial(Node):
    """Postfix factorial: n!"""
    operand: Node