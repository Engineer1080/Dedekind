from .ast_nodes import *

class CodeGenerator:
    def __init__(self):
        self.code = []
        self.indent_level = 0

    def generate(self, node: Node) -> str:
        self.code = []
        self.add_line("import numpy as np")
        self.add_line("import sys")
        self.add_line("")
        
        # Helper for GPU simulation
        self.add_line("def _to_gpu(data):")
        self.add_line("    print(f'Transferring {type(data)} to GPU...', file=sys.stderr)")
        self.add_line("    return data # In real implementation, return torch.tensor(data).cuda()")
        self.add_line("")
        
        self.visit(node)
        return "\n".join(self.code)

    def visit(self, node: Node):
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: Node):
        raise Exception(f"No visit method for {type(node).__name__}")

    def add_line(self, line: str):
        self.code.append("    " * self.indent_level + line)

    def visit_Program(self, node: Program):
        for stmt in node.statements:
            res = self.visit(stmt)
            if isinstance(res, str):
                self.add_line(res)

    def visit_IfStmt(self, node: IfStmt):
        cond = self.visit_expression(node.condition)
        self.add_line(f"if {cond}:")
        self.indent_level += 1
        for stmt in node.then_branch:
            res = self.visit(stmt)
            if isinstance(res, str):
                self.add_line(res)
        self.indent_level -= 1
        
        if node.else_branch:
            self.add_line("else:")
            self.indent_level += 1
            for stmt in node.else_branch:
                res = self.visit(stmt)
                if isinstance(res, str):
                    self.add_line(res)
            self.indent_level -= 1

    def visit_WhileStmt(self, node: WhileStmt):
        cond = self.visit_expression(node.condition)
        self.add_line(f"while {cond}:")
        self.indent_level += 1
        for stmt in node.body:
            res = self.visit(stmt)
            if isinstance(res, str):
                self.add_line(res)
        self.indent_level -= 1

    def visit_ForStmt(self, node: ForStmt):
        collection = self.visit_expression(node.collection)
        self.add_line(f"for {node.variable} in {collection}:")
        self.indent_level += 1
        for stmt in node.body:
            res = self.visit(stmt)
            if isinstance(res, str):
                self.add_line(res)
        self.indent_level -= 1

    def visit_FunctionDef(self, node: FunctionDef):
        args_str = ", ".join(node.args)
        self.add_line(f"def {node.name}({args_str}):")
        self.indent_level += 1
        for stmt in node.body:
            res = self.visit(stmt)
            if isinstance(res, str):
                self.add_line(res)
        self.indent_level -= 1
        self.add_line("")

    def visit_ReturnStmt(self, node: ReturnStmt):
        val = self.visit_expression(node.value)
        self.add_line(f"return {val}")

    def visit_Assignment(self, node: Assignment):
        val = self.visit_expression(node.value)
        self.add_line(f"{node.target} = {val}")

    def visit_BinaryOp(self, node: BinaryOp):
        left = self.visit_expression(node.left)
        right = self.visit_expression(node.right)
        return f"({left} {node.op} {right})"

    def visit_Literal(self, node: Literal):
        if isinstance(node.value, str):
            return f'"{node.value}"'
        return str(node.value)

    def visit_VectorLiteral(self, node: VectorLiteral):
        elements = [self.visit_expression(e) for e in node.elements]
        return f"np.array([{', '.join(elements)}])"

    def visit_Identifier(self, node: Identifier):
        return node.name

    def visit_Call(self, node: Call):
        func_name = ""
        if isinstance(node.func_name, Identifier):
            func_name = node.func_name.name
        else:
            func_name = self.visit_expression(node.func_name)
            
        args = [self.visit_expression(a) for a in node.args]
        
        # Special handling for matrix operations
        if func_name == "matmul" or func_name == "dot":
             # Assuming method call syntax: matrix.matmul(other) -> matmul(matrix, other) 
             # In our parser logic for method chaining: node = Call(Identifier(method), [node] + args, [])
             # So args[0] is the matrix found, args[1] is the argument.
             if len(args) >= 2:
                 return f"({args[0]} @ {args[1]})"
        
        args_str = ", ".join(args)
        
        call_str = f"{func_name}({args_str})"
        
        # Handle modifiers
        if 'gpu' in node.modifiers:
            call_str = f"_to_gpu({call_str})"
            
        return call_str

    def visit_expression(self, node: Node):
        # Helper to visit expression nodes and return string instead of adding line
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
