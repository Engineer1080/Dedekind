from .ast_nodes import *
from typing import List

class MLIRCodeGenerator:
    """
    Prototype MLIR Generator for Fourier.
    Emits a structured IR (Fourier Dialect) that can be lowered to LLVM/MLIR.
    """
    def __init__(self):
        self.output = []
        self.indent = 0
        self.temp_count = 0

    def generate(self, node: Node) -> str:
        self.output = []
        self.emit("// Fourier MLIR Dialect Prototype v0.2")
        self.emit("module {")
        self.indent += 1
        self.visit(node)
        self.indent -= 1
        self.emit("}")
        return "\n".join(self.output)

    def emit(self, line: str):
        self.output.append("  " * self.indent + line)

    def next_temp(self):
        self.temp_count += 1
        return f"%{self.temp_count}"

    def visit(self, node: Node):
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: Node):
        for child in getattr(node, 'statements', []):
            self.visit(child)

    def visit_Program(self, node: Program):
        for stmt in node.statements:
            self.visit(stmt)

    def visit_FunctionDef(self, node: FunctionDef):
        args_str = ", ".join([f"%{a}: tensor<*xf32>" for a in node.args])
        self.emit(f"func.func @{node.name}({args_str}) -> tensor<*xf32> {{")
        self.indent += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent -= 1
        self.emit("}")

    def visit_Assignment(self, node: Assignment):
        val_name = self.visit_expression(node.value)
        self.emit(f"// assignment to {node.target}")
        self.emit(f"fourier.store {val_name}, @{node.target}")

    def visit_Call(self, node: Call):
        func_name = node.func_name.name if isinstance(node.func_name, Identifier) else "closure"
        args = [self.visit_expression(a) for a in node.args]
        res = self.next_temp()
        
        # Special mapping for built-ins
        if func_name == "matmul":
            self.emit(f"{res} = linalg.matmul {args[0]}, {args[1]}")
        elif func_name == "relu":
            self.emit(f"{res} = fourier.relu {args[0]}")
        else:
            args_str = ", ".join(args)
            self.emit(f"{res} = call @{func_name}({args_str})")
        return res

    def visit_Literal(self, node: Literal):
        res = self.next_temp()
        self.emit(f"{res} = arith.constant {node.value} : f32")
        return res

    def visit_VectorLiteral(self, node: VectorLiteral):
        res = self.next_temp()
        elements = ", ".join([str(e.value) for e in node.elements if isinstance(e, Literal)])
        self.emit(f"{res} = fourier.constant_tensor [{elements}] : tensor<{len(node.elements)}xf32>")
        return res

    def visit_Identifier(self, node: Identifier):
        res = self.next_temp()
        self.emit(f"{res} = fourier.load @{node.name}")
        return res

    def visit_BinaryOp(self, node: BinaryOp):
        left = self.visit_expression(node.left)
        right = self.visit_expression(node.right)
        res = self.next_temp()
        op_map = {'+': 'addf', '-': 'subf', '*': 'mulf', '/': 'divf'}
        mlir_op = op_map.get(node.op, 'unknown')
        self.emit(f"{res} = arith.{mlir_op} {left}, {right} : f32")
        return res

    def visit_expression(self, node: Node):
        return self.visit(node)
