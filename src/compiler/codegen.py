from .ast_nodes import *

class CodeGenerator:
    def __init__(self):
        self.code = []
        self.indent_level = 0

    def generate(self, node: Node) -> str:
        self.code = []
        self.code.append("import torch")
        self.code.append("import torch.nn as nn")
        self.code.append("import sys")
        self.code.append("")
        
        self.add_line("def _to_tensor(data):")
        self.add_line("    if isinstance(data, torch.Tensor): return data")
        self.add_line("    if isinstance(data, (list, tuple)):")
        self.add_line("        try:")
        self.add_line("            converted = [_to_tensor(x) for x in data]")
        self.add_line("            if any(isinstance(x, torch.Tensor) for x in converted):")
        self.add_line("                return torch.stack(converted)")
        self.add_line("        except: pass")
        self.add_line("    try: return torch.as_tensor(data, dtype=torch.float32)")
        self.add_line("    except: return data")
        self.add_line("")
        self.add_line("def _to_gpu(data):")
        self.add_line("    data = _to_tensor(data)")
        self.add_line("    if torch.cuda.is_available(): return data.to('cuda')")
        self.add_line("    return data")
        self.add_line("")
        self.add_line("def _to_cpu(data):")
        self.add_line("    data = _to_tensor(data)")
        self.add_line("    return data.to('cpu')")
        self.add_line("")
        
        self.code.append("# Fourier ML Runtime")
        ml_runtime_path = __file__.replace('codegen.py', 'ml_runtime.py')
        try:
            with open(ml_runtime_path, 'r') as f:
                ml_code = f.read()
                reached_code = False
                for line in ml_code.split('\n'):
                    if not reached_code:
                        if line.startswith('class ') or line.startswith('def '):
                            reached_code = True
                        else:
                            continue
                    if reached_code:
                        self.code.append(line.rstrip())
        except FileNotFoundError:
            pass
        
        self.code.append("")
        
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
            if isinstance(res, str): self.add_line(res)
        self.indent_level -= 1
        if node.else_branch:
            self.add_line("else:")
            self.indent_level += 1
            for stmt in node.else_branch:
                res = self.visit(stmt)
                if isinstance(res, str): self.add_line(res)
            self.indent_level -= 1

    def visit_WhileStmt(self, node: WhileStmt):
        cond = self.visit_expression(node.condition)
        self.add_line(f"while {cond}:")
        self.indent_level += 1
        for stmt in node.body:
            res = self.visit(stmt)
            if isinstance(res, str): self.add_line(res)
        self.indent_level -= 1

    def visit_ForStmt(self, node: ForStmt):
        collection = self.visit_expression(node.collection)
        self.add_line(f"for {node.variable} in {collection}:")
        self.indent_level += 1
        for stmt in node.body:
            res = self.visit(stmt)
            if isinstance(res, str): self.add_line(res)
        self.indent_level -= 1

    def visit_FunctionDef(self, node: FunctionDef):
        args_str = ", ".join(node.args)
        self.add_line(f"def {node.name}({args_str}):")
        self.indent_level += 1
        for stmt in node.body:
            res = self.visit(stmt)
            if isinstance(res, str): self.add_line(res)
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
        if isinstance(node.value, str): return f'"{node.value}"'
        return str(node.value)

    def visit_VectorLiteral(self, node: VectorLiteral):
        elements = [self.visit_expression(e) for e in node.elements]
        return f"_to_tensor([{', '.join(elements)}])"

    def visit_Identifier(self, node: Identifier):
        return node.name

    def visit_Call(self, node: Call):
        if isinstance(node.func_name, Identifier):
            func_name = node.func_name.name
        else:
            func_name = self.visit_expression(node.func_name)
            
        args = [self.visit_expression(a) for a in node.args]
        kwargs = [f"{k}={self.visit_expression(v)}" for k, v in node.kwargs]
        
        if func_name == "gpu": return f"_to_gpu({args[0]})"
        if func_name == "cpu": return f"_to_cpu({args[0]})"
        if func_name == "fast": return f"compile_model({args[0]})"
        if func_name == "with_grad": return f"_to_grad({args[0]})"
        if func_name == "grad":
            # Native Autograd implementation: grad(f, x)
            # We map this to torch.autograd.grad(f(x), x)[0]
            # args[0] is the function, args[1] is the parameter
            return f"torch.autograd.grad({args[0]}({args[1]}), {args[1]}, create_graph=True)[0]"
        if func_name == "einsum":
            # Map Fourier einsum("equation", ...) to torch.einsum("equation", ...)
            # The first argument is the string equation
            return f"torch.einsum({args[0]}, {', '.join(args[1:])})"
        if func_name == "matmul": return f"torch.matmul({args[0]}, {args[1]})"
        if func_name == "forward":
             all_args = args[1:] + kwargs
             return f"{args[0]}({', '.join(all_args)})"
        
        all_args_str = ", ".join(args + kwargs)
        call_str = f"{func_name}({all_args_str})"
        
        if 'gpu' in node.modifiers: call_str = f"_to_gpu({call_str})"
        if 'cpu' in node.modifiers: call_str = f"_to_cpu({call_str})"
        if 'fast' in node.modifiers: call_str = f"compile_model({call_str})"
        return call_str

    def visit_MemberAccess(self, node: MemberAccess):
        obj = self.visit_expression(node.obj)
        return f"{obj}.{node.member}"

    def visit_expression(self, node: Node):
        method_name = f'visit_{self.type_name(node)}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
