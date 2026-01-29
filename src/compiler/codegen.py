from .ast_nodes import *

class CodeGenerator:
    def __init__(self):
        self.code = []
        self.indent_level = 0

    def generate(self, node: Node) -> str:
        self.code = []
        self.add_line("import torch")
        self.add_line("import sys")
        self.add_line("")
        
        # Helper for Device Management
        self.add_line("def _get_device():")
        self.add_line("    if torch.cuda.is_available(): return 'cuda'")
        self.add_line("    # logic for mps (mac) could go here")
        self.add_line("    return 'cpu'")
        self.add_line("")
        
        self.add_line("def _to_gpu(data):")
        self.add_line("    if torch.cuda.is_available():")
        self.add_line("        return data.to('cuda')")
        self.add_line("    print('Warning: GPU not available, running on CPU', file=sys.stderr)")
        self.add_line("    return data")
        self.add_line("")

        self.add_line("def _to_cpu(data):")
        self.add_line("    return data.to('cpu')")
        self.add_line("")
        
        self.visit(node)
        return "\n".join(self.code)

    def visit(self, node: Node):
        method_name = f'visit_{self.type_name(node)}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
    
    def type_name(self, node):
        return type(node).__name__

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
        # PyTorch Tensor creation
        return f"torch.tensor([{', '.join(elements)}])"

    def visit_Identifier(self, node: Identifier):
        return node.name

    def visit_Call(self, node: Call):
        func_name = ""
        if isinstance(node.func_name, Identifier):
            func_name = node.func_name.name
        else:
            func_name = self.visit_expression(node.func_name)
            
        args = [self.visit_expression(a) for a in node.args]
        
        # Matrix Math Overrides
        if func_name == "matmul":
             if len(args) >= 2:
                 return f"torch.matmul({args[0]}, {args[1]})" # Better for Torch than @ sometimes
        
        args_str = ", ".join(args)
        call_str = f"{func_name}({args_str})"
        
        # Modifiers -> PyTorch device moves
        if 'gpu' in node.modifiers:
            call_str = f"_to_gpu({call_str})"
        if 'cpu' in node.modifiers:
            call_str = f"_to_cpu({call_str})"
            
        return call_str

    def visit_expression(self, node: Node):
        method_name = f'visit_{self.type_name(node)}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
