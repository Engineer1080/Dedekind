from dataclasses import dataclass
from typing import List, Optional, Union, Any

@dataclass
class Node:
    pass

@dataclass
class Program(Node):
    statements: List[Node]

@dataclass
class FunctionDef(Node):
    name: str
    args: List[str]
    body: List[Node]

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

@dataclass
class QuaternionLiteral(Node):
    value: float
    component: str # 'i', 'j', or 'k'

@dataclass
class VectorLiteral(Node):
    elements: List[Node]

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
