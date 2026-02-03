from .ast_nodes import *

# Names that require torch / ML runtime when used as function or member
_TORCH_FUNC_NAMES = frozenset({
    'gpu', 'cpu', 'fast', 'with_grad', 'sparse', 'grad', 'einsum', 'matmul',
    'randn', 'relu', 'softmax', 'conv2d', 'max_pool2d', 'fft', 'ifft',
    'linspace', 'stack', 'to_tensor', 'to_gpu', 'to_cpu',
})
_TORCH_MEMBER_NAMES = frozenset({'gpu', 'cpu', 'single'})

# All built-ins provided by ml_runtime (functions, classes, constants).
# If the program uses any of these, we must inject the ML runtime.
_RUNTIME_BUILTIN_NAMES = frozenset({
    'Quantity', 'Quaternion', 'Dense', 'Sequential', 'compile_model',
    'random_vector', 'random_matrix', 'transpose', 'inverse', 'dot_product', 'cross',
    'relu', 'softmax', 'convolution', 'pooling', 'fft', 'ifft', 'fftfreq', 'diff', 'cumsum', 'clip', 'shuffle', 'linspace',
    'ode_solve', 'pde_heat_1d', 'pde_heat_2d', 'pde_advection_1d', 'pde_advection_2d', 'pde_wave_1d', 'pde_wave_2d', 'pde_burgers_1d', 'pde_burgers_2d',
    'sparse_laplacian_2d', 'sparse_diffusion_step', 'sparse_diffusion_simulate',
    'Normal', 'Uniform', 'Bernoulli', 'Exponential', 'Gamma', 'Beta', 'Poisson',
    'sample', 'log_prob', 'metropolis', 'hmc',
    'sin', 'cos', 'tan', 'exp', 'log', 'log10', 'sqrt', 'abs',
    'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh', 'erf', 'erfc', 'gamma', 'lgamma',
    'bessel_j0', 'bessel_j1', 'bessel_j', 'legendre', 'hypergeom',
    'min', 'max', 'argmin', 'argmax', 'round', 'floor', 'ceil',
    'mean', 'std', 'var', 'median', 'quantile', 'percentile', 'cov', 'corrcoef', 'skew', 'kurtosis', 'histogram',
    'norm', 'det', 'trace', 'solve', 'eigh', 'eig', 'svd', 'lstsq',
    'cond', 'rank', 'pinv', 'expm', 'logm', 'interp', 'trapz', 'root_bisect',
    'qr', 'cholesky', 'lu', 'matrix_power', 'kron', 'outer', 'vander', 'matrix_sqrt', 'matrix_norm',
    'cdist', 'polyfit', 'polyval', 'unique', 'argsort', 'convolve1d',
    'minimize_scalar', 'newton', 'minimize', 'fsolve', 'integrate', 'simpson',
    'permutation', 'choice', 'autocorr', 'moving_mean',
    'UncertainQuantity', 'uncertain', 'fit',
    'michaelis_menten', 'logistic', 'logistic_growth_dt', 'arrhenius', 'linear_regression',
    'atomic_mass', 'atomic_number',
    'gcd', 'is_prime', 'mod', 'mod_inv', 'mod_pow', 'factorial',
    'concentration_to_pH', 'pH_to_concentration',
    'christoffel_symbols', 'riemann_tensor', 'covariant_derivative',
    'balance_equation',
    'read_file', 'write_file', 'file_exists', 'http_get', 'http_post',
    'json_parse', 'json_stringify', 'sort', 'quicksort', 'plot', 'scatter', 'contour', 'print_latex',
    'assert', 'diff_sym', 'jacobian', 'hessian',
    # Constants (from ml_runtime)
    'pi', 'e', 'c', 'G', 'h', 'k_B', 'k_e', 'hbar', 'e_charge', 'epsilon_0', 'mu_0',
    'm_e', 'm_p', 'N_A', 'R_gas', 'alpha', 'sigma_SB', 'F_faraday',
})


def _program_uses_torch(node: Node) -> bool:
    """Return True if the AST uses tensor/ML features or any runtime built-in."""
    if node is None:
        return False
    if type(node).__name__ == 'VectorLiteral':
        return True
    if type(node).__name__ == 'QuaternionLiteral':
        return True
    if type(node).__name__ == 'Quantity':
        return True
    if type(node).__name__ == 'IndexedVariable':
        return True
    if type(node).__name__ == 'PostfixFactorial':
        return True  # factorial() requires runtime
    if type(node).__name__ == 'Identifier':
        if getattr(node, 'name', None) in _RUNTIME_BUILTIN_NAMES:
            return True
    if type(node).__name__ == 'Call':
        if getattr(node, 'modifiers', None):
            return True
        name = None
        if isinstance(getattr(node, 'func_name', None), Identifier):
            name = node.func_name.name
        elif type(getattr(node, 'func_name', None)).__name__ == 'MemberAccess':
            name = node.func_name.member
        if name and (name in _TORCH_FUNC_NAMES or name in _RUNTIME_BUILTIN_NAMES):
            return True
    if type(node).__name__ == 'MemberAccess':
        if getattr(node, 'member', None) in _TORCH_MEMBER_NAMES:
            return True
    # Recurse
    for child in _ast_children(node):
        if _program_uses_torch(child):
            return True
    return False


def _ast_children(node: Node):
    """Yield direct AST children of node for traversal."""
    if node is None:
        return
    if type(node).__name__ == 'Program':
        for s in getattr(node, 'statements', []):
            yield s
        return
    if type(node).__name__ in ('FunctionDef', 'IfStmt', 'WhileStmt', 'ForStmt'):
        for attr in ('body', 'then_branch', 'else_branch', 'condition', 'collection'):
            val = getattr(node, attr, None)
            if val is None:
                continue
            if isinstance(val, list):
                for item in val:
                    yield item
            else:
                yield val
        if type(node).__name__ == 'FunctionDef':
            pass  # args are names, not nodes
        return
    if type(node).__name__ == 'Call':
        yield getattr(node, 'func_name', None)
        for a in getattr(node, 'args', []):
            yield a
        for _, v in getattr(node, 'kwargs', []):
            yield v
        return
    if type(node).__name__ == 'Assignment':
        yield getattr(node, 'value', None)
        return
    if type(node).__name__ == 'ReturnStmt':
        yield getattr(node, 'value', None)
        return
    if type(node).__name__ == 'BinaryOp':
        yield getattr(node, 'left', None)
        yield getattr(node, 'right', None)
        return
    if type(node).__name__ == 'MemberAccess':
        yield getattr(node, 'obj', None)
        return
    if type(node).__name__ == 'Subscript':
        yield getattr(node, 'value', None)
        yield getattr(node, 'index', None)
        return
    if type(node).__name__ == 'ItemAssignment':
        yield getattr(node, 'target', None)
        yield getattr(node, 'value', None)
        return
    if type(node).__name__ == 'VectorLiteral':
        for e in getattr(node, 'elements', []):
            yield e
        return
    if type(node).__name__ == 'Lambda':
        yield getattr(node, 'body', None)
        return


class CodeGenerator:
    def __init__(self):
        self.code = []
        self.indent_level = 0

    def generate(self, node: Node) -> str:
        self.code = []
        self.code.append("import sys")
        self.code.append("import builtins")
        self.code.append("")

        if not _program_uses_torch(node):
            self.visit(node)
            return "\n".join(self.code)

        self.code.append("import torch")
        self.code.append("import torch.nn as nn")
        self.code.append("")

        self.add_line("def _to_tensor(data):")
        self.add_line("    if isinstance(data, torch.Tensor): return data")
        self.add_line("    if isinstance(data, (list, tuple)):")
        self.add_line("        if not data: return torch.tensor([], dtype=torch.float32)")
        self.add_line("        try: return torch.as_tensor(data)")
        self.add_line("        except: pass")
        self.add_line("        converted = []")
        self.add_line("        for x in data:")
        self.add_line("            try:")
        self.add_line("                t = _to_tensor(x)")
        self.add_line("                if isinstance(t, torch.Tensor): converted.append(t)")
        self.add_line("                else: return data")
        self.add_line("            except: return data")
        self.add_line("        if converted and len(converted) == len(data): return torch.stack(converted)")
        self.add_line("        return data")
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
        
        self.code.append("# Dedekind ML Runtime")
        ml_runtime_path = __file__.replace('codegen.py', 'ml_runtime.py')
        try:
            with open(ml_runtime_path, 'r') as f:
                ml_code = f.read()
                reached_code = False
                for line in ml_code.split('\n'):
                    if not reached_code:
                        if line.startswith('class ') or line.startswith('def ') or (line.strip() and not line.startswith('import ') and '=' in line and not line.startswith(' ')):
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
        # Unäres Minus: 0 - expr -> -expr (für Quantity und andere Typen)
        if node.op == '-' and isinstance(node.left, Literal) and node.left.value == 0:
            right = self.visit_expression(node.right)
            return f"(-{right})"
        if node.op == '*' and (isinstance(node.left, (IndexedVariable, BinaryOp)) or isinstance(node.right, (IndexedVariable, BinaryOp))):
            # Try to build a Ricci contraction
            def collect_indexed(n):
                if isinstance(n, IndexedVariable): return [(n.name, n.indices)]
                if isinstance(n, BinaryOp) and n.op == '*':
                    l = collect_indexed(n.left)
                    r = collect_indexed(n.right)
                    if l is not None and r is not None: return l + r
                return None
            
            items = collect_indexed(node)
            if items and len(items) >= 2:
                tensor_names = [it[0] for it in items]
                index_groups = [it[1] for it in items]
                
                all_indices = "".join(index_groups)
                from collections import Counter
                counts = Counter(all_indices)
                # Keep indices that appear exactly once
                result_indices = ""
                seen = set()
                for c in all_indices:
                    if counts[c] == 1 and c not in seen:
                        result_indices += c
                        seen.add(c)
                
                equation = f"{','.join(index_groups)}->{result_indices}"
                return f"torch.einsum('{equation}', {', '.join(tensor_names)})"

        left = self.visit_expression(node.left)
        right = self.visit_expression(node.right)
        op = node.op
        if op == '@':
            return f"torch.matmul({left}, {right})"
        if op == '^': op = '**'
        return f"({left} {op} {right})"

    def visit_Literal(self, node: Literal):
        if isinstance(node.value, str):
            if getattr(node, 'raw', False):
                return 'r"' + node.value.replace('"', '\\"') + '"'
            return '"' + node.value.replace('\\', '\\\\').replace('"', '\\"') + '"'
        return str(node.value)

    def visit_Quantity(self, node):
        unit = (node.unit or "").replace('\\', '\\\\').replace('"', '\\"')
        return f'Quantity({node.value}, "{unit}")'

    def visit_QuaternionLiteral(self, node: QuaternionLiteral):
        if node.component == 'i': return f"Quaternion(0, {node.value}, 0, 0)"
        if node.component == 'j': return f"Quaternion(0, 0, {node.value}, 0)"
        if node.component == 'k': return f"Quaternion(0, 0, 0, {node.value})"
        return "Quaternion()"

    def visit_VectorLiteral(self, node: VectorLiteral):
        elements = [self.visit_expression(e) for e in node.elements]
        return f"_to_tensor([{', '.join(elements)}])"

    def visit_Identifier(self, node: Identifier):
        if node.name == "true":
            return "True"
        if node.name == "false":
            return "False"
        return node.name

    def visit_IndexedVariable(self, node: IndexedVariable):
        # If used standalone, it's the tensor name plus indices (e.g., F_gravity)
        return f"{node.name}_{node.indices}"

    def visit_Call(self, node: Call):
        obj_expr = None
        if isinstance(node.func_name, Identifier):
            func_name = node.func_name.name
        elif isinstance(node.func_name, MemberAccess):
            func_name = node.func_name.member
            obj_expr = self.visit_expression(node.func_name.obj)
        else:
            func_name = self.visit_expression(node.func_name)
            
        args = [self.visit_expression(a) for a in node.args]
        kwargs = [f"{k}={self.visit_expression(v)}" for k, v in node.kwargs]
        
        # Unified Target Selection (obj.func() or func(obj))
        target = obj_expr if obj_expr else (args[0] if args else None)
        other_args = args if obj_expr else args[1:]
        
        if func_name == "gpu": return f"_to_gpu({target})"
        if func_name == "cpu": return f"_to_cpu({target})"
        if func_name == "fast": return f"compile_model({target})"
        if func_name == "with_grad": return f"_to_grad({target})"
        if func_name == "sparse": return f"_to_sparse({target})"
        
        if func_name == "grad":
            # Native Autograd implementation: grad(f, x)
            return f"torch.autograd.grad({args[0]}({args[1]}), {args[1]}, create_graph=True)[0]"
        if func_name == "einsum":
            return f"torch.einsum({args[0]}, {', '.join(args[1:])})"
        if func_name == "matmul": return f"torch.matmul({args[0]}, {args[1]})"
        if func_name == "forward":
             all_args = other_args + kwargs
             return f"{target}({', '.join(all_args)})"
        
        all_args_str = ", ".join(args + kwargs)
        if func_name == "assert":
            return f"_dedekind_assert({all_args_str})"
        if obj_expr:
            return f"{obj_expr}.{func_name}({all_args_str})"
        else:
            return f"{func_name}({all_args_str})"
        
        if 'gpu' in node.modifiers: call_str = f"_to_gpu({call_str})"
        if 'cpu' in node.modifiers: call_str = f"_to_cpu({call_str})"
        if 'fast' in node.modifiers: call_str = f"compile_model({call_str})"
        return call_str

    def visit_MemberAccess(self, node: MemberAccess):
        obj = self.visit_expression(node.obj)
        return f"{obj}.{node.member}"

    def visit_Subscript(self, node: Subscript):
        val = self.visit_expression(node.value)
        idx = self.visit_expression(node.index)
        return f"{val}[{idx}]"

    def visit_ItemAssignment(self, node: ItemAssignment):
        target = self.visit_Subscript(node.target)
        value = self.visit_expression(node.value)
        return f"{target} = {value}"

    def visit_PostfixFactorial(self, node: PostfixFactorial):
        operand = self.visit_expression(node.operand)
        return f"factorial({operand})"

    def visit_expression(self, node: Node):
        method_name = f'visit_{self.type_name(node)}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
