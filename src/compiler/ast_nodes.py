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
    modifiers: List[str]  # e.g., ['gpu', 'cpu']

@dataclass
class Identifier(Node):
    name: str

@dataclass
class Literal(Node):
    value: Any

@dataclass
class VectorLiteral(Node):
    elements: List[Node]

@dataclass
class Lambda(Node):
    arg: str
    body: Node
