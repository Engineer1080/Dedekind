from typing import List
from .ast_nodes import *
from .lexer import Token, Lexer

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def _line(self) -> int:
        """Current line (for error messages)."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos].line
        return 1

    def consume(self, type: str = None):
        if self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            if type is None or token.type == type:
                self.pos += 1
                return token
            raise CompileError(
                f"Erwartet {type}, gefunden '{token.value}' ({token.type}).",
                line=token.line,
            )
        tok = self.tokens[-1] if self.tokens else None
        raise CompileError(
            f"Erwartet {type}, Dateiende erreicht.",
            line=tok.line if tok else 1,
        )

    def peek(self, offset=0):
        if self.pos + offset < len(self.tokens):
            return self.tokens[self.pos + offset]
        return None

    @staticmethod
    def _looks_like_ricci_suffix(id_value: str) -> bool:
        """True wenn id_value die Form name_indices hat und indices wie Tensor-Indizes (i,j,k) aussehen."""
        if '_' not in id_value:
            return False
        _, _, suffix = id_value.rpartition('_')
        return 1 <= len(suffix) <= 4 and all(c in 'ijk' for c in suffix)

    def parse(self) -> Program:
        statements = []
        while self.pos < len(self.tokens):
            if self.peek() and self.peek().type == 'NEWLINE':
                self.consume()
                continue
            statements.append(self.parse_statement())
        return Program(statements)

    def parse_statement(self):
        token = self.peek()
        
        # Check for assignment: ID (=) or ID (^/_) ID (=) or ID[...] (=)
        if token and token.type == 'ID':
            p = 1
            # Skip indices to see if an ASSIGN follows
            while self.peek(p) and self.peek(p).type in ['CARET', 'UNDERSCORE']:
                p += 1
                if self.peek(p) and self.peek(p).type == 'ID':
                    p += 1
                else: break
            
            # Skip subscripts to see if an ASSIGN follows
            while self.peek(p) and self.peek(p).type == 'LBRACKET':
                p += 1
                # We need a more robust way to skip the whole subscript [expr].
                # Simple count of brackets for now.
                depth = 1
                while depth > 0 and self.peek(p):
                    if self.peek(p).type == 'LBRACKET': depth += 1
                    elif self.peek(p).type == 'RBRACKET': depth -= 1
                    p += 1
            
            if self.peek(p) and self.peek(p).type == 'ASSIGN':
                return self.parse_assignment()

        if token.type == 'FN':
            return self.parse_function_def(is_pub=False)
        elif token.type == 'PUB':
            self.consume('PUB')
            if not self.peek() or self.peek().type != 'FN':
                raise CompileError(
                    "`pub` muss vor einer Funktionsdefinition stehen (`pub fn name(...) { ... }`).",
                    line=token.line,
                )
            return self.parse_function_def(is_pub=True)
        elif token.type == 'USE':
            return self.parse_use_stmt()
        elif token.type == 'ID' and token.value == 'unit' \
                and self.peek(1) and self.peek(1).type == 'ID' \
                and self.peek(2) and self.peek(2).type == 'ASSIGN':
            # Soft-Keyword `unit`: nur als Statement-Anfang `unit NAME = ...` interpretiert.
            return self.parse_unit_def()
        elif token.type == 'ID' and token.value == 'pyimport' \
                and self.peek(1) and self.peek(1).type == 'ID':
            # Soft-Keyword `pyimport`: nur am Statement-Anfang `pyimport mod[.sub] [as alias]`.
            return self.parse_pyimport_stmt()
        elif token.type == 'RETURN':
            start_line = token.line
            self.consume('RETURN')
            expr = self.parse_expression()
            stmt = ReturnStmt(expr)
            stmt.line = start_line
            return stmt
        elif token.type == 'IF':
            return self.parse_if_stmt()
        elif token.type == 'WHILE':
            return self.parse_while_stmt()
        elif token.type == 'FOR':
            return self.parse_for_stmt()
        elif token.type == 'TRY':
            return self.parse_try_catch()
        else:
            return self.parse_expression()

    def parse_try_catch(self):
        start_line = self.peek().line
        self.consume('TRY')
        self.consume('LBRACE')
        body = []
        while self.peek().type != 'RBRACE':
            body.append(self.parse_statement())
        self.consume('RBRACE')
        if not self.peek() or self.peek().type != 'CATCH':
            raise CompileError(
                "`try { ... }` braucht `catch <name> { ... }`.",
                line=start_line,
            )
        self.consume('CATCH')
        var_tok = self.consume('ID')
        self.consume('LBRACE')
        handler = []
        while self.peek().type != 'RBRACE':
            handler.append(self.parse_statement())
        self.consume('RBRACE')
        node = TryCatch(body=body, catch_var=var_tok.value, handler=handler)
        node.line = start_line
        return node

    def parse_if_stmt(self):
        start_line = self.peek().line
        self.consume('IF')
        condition = self.parse_expression()
        self.consume('LBRACE')
        then_branch = []
        while self.peek().type != 'RBRACE':
            then_branch.append(self.parse_statement())
        self.consume('RBRACE')
        else_branch = None
        if self.peek() and self.peek().type == 'ELSE':
            self.consume('ELSE')
            self.consume('LBRACE')
            else_branch = []
            while self.peek().type != 'RBRACE':
                else_branch.append(self.parse_statement())
            self.consume('RBRACE')
        node = IfStmt(condition, then_branch, else_branch)
        node.line = start_line
        return node

    def parse_while_stmt(self):
        start_line = self.peek().line
        self.consume('WHILE')
        condition = self.parse_expression()
        self.consume('LBRACE')
        body = []
        while self.peek().type != 'RBRACE':
            body.append(self.parse_statement())
        self.consume('RBRACE')
        node = WhileStmt(condition, body)
        node.line = start_line
        return node
        
    def parse_for_stmt(self):
        start_line = self.peek().line
        self.consume('FOR')
        var_name = self.consume('ID').value
        if self.peek() and self.peek().type == 'IN':
            self.consume('IN')
        collection = self.parse_expression()
        self.consume('LBRACE')
        body = []
        while self.peek().type != 'RBRACE':
            body.append(self.parse_statement())
        self.consume('RBRACE')
        node = ForStmt(var_name, collection, body)
        node.line = start_line
        return node

    def _parse_unit_bracket(self):
        """Reads `[m/s]` and returns 'm/s'. Precondition: current token = LBRACKET."""
        self.consume('LBRACKET')
        parts = []
        while self.peek() and self.peek().type != 'RBRACKET':
            t = self.consume()
            if t.type == 'ID':
                parts.append(t.value)
            elif t.type == 'NUMBER':
                parts.append(str(t.value))
            elif t.type == 'MUL':
                parts.append('*')
            elif t.type == 'DIV':
                parts.append('/')
            elif t.type == 'CARET' and self.peek() and self.peek().type == 'NUMBER':
                parts.append('^')
                parts.append(self.consume().value)
            elif t.type == 'MINUS':
                parts.append('-')
            else:
                break
        self.consume('RBRACKET')
        return ''.join(parts)

    def parse_use_stmt(self):
        """Accepts `use foo`, `use foo.bar`, `use foo.bar.baz` -- dotted
        paths resolve to modules/foo/bar/baz.ddk."""
        start_line = self.peek().line
        self.consume('USE')
        parts = [self.consume('ID').value]
        while self.peek() and self.peek().type == 'DOT' \
                and self.peek(1) and self.peek(1).type == 'ID':
            self.consume('DOT')
            parts.append(self.consume('ID').value)
        node = UseStmt(".".join(parts))
        node.line = start_line
        return node

    def parse_pyimport_stmt(self):
        """`pyimport scipy.special as ss` or `pyimport numpy` (alias = last segment)."""
        start_line = self.peek().line
        self.consume('ID')  # 'pyimport' (soft keyword)
        parts = [self.consume('ID').value]
        while self.peek() and self.peek().type == 'DOT' \
                and self.peek(1) and self.peek(1).type == 'ID':
            self.consume('DOT')
            parts.append(self.consume('ID').value)
        module = ".".join(parts)
        alias = parts[-1]
        if self.peek() and self.peek().type == 'ID' and self.peek().value == 'as':
            self.consume('ID')
            alias_tok = self.consume('ID')
            alias = alias_tok.value
        node = PyImport(module=module, alias=alias)
        node.line = start_line
        return node

    def parse_unit_def(self):
        """`unit NAME = NUMBER[base_unit]` -- registers a new unit (length, mass, pressure symbol, etc.)."""
        start_line = self.peek().line
        self.consume('ID')  # 'unit' (soft keyword)
        name_tok = self.consume('ID')
        self.consume('ASSIGN')
        # Optional sign
        sign = 1.0
        if self.peek() and self.peek().type == 'MINUS':
            self.consume('MINUS')
            sign = -1.0
        elif self.peek() and self.peek().type == 'PLUS':
            self.consume('PLUS')
        num_tok = self.consume('NUMBER')
        try:
            factor = sign * float(num_tok.value)
        except ValueError:
            raise CompileError(
                f"Invalid number for `unit {name_tok.value}`: {num_tok.value!r}.",
                line=start_line,
            )
        if not self.peek() or self.peek().type != 'LBRACKET':
            raise CompileError(
                f"`unit {name_tok.value}` requires a base unit in brackets, e.g. `unit Foot = 0.3048[m]`.",
                line=start_line,
            )
        base_unit = self._parse_unit_bracket()
        if not base_unit:
            raise CompileError(
                f"`unit {name_tok.value}`: base unit must not be empty.",
                line=start_line,
            )
        node = UnitDef(name=name_tok.value, factor=factor, base_unit=base_unit)
        node.line = start_line
        return node

    _SHAPE_TYPES = {'scalar', 'vector', 'matrix', 'tensor', 'graph', 'labeledtensor', 'sequence',
                    'qubit', 'circuit', 'statevec'}

    def _parse_unit_bracket_inline(self):
        """Reads the inside of an already-consumed LBRACKET (for unit annotation)."""
        parts = []
        while self.peek() and self.peek().type != 'RBRACKET':
            t = self.consume()
            if t.type == 'ID': parts.append(t.value)
            elif t.type == 'NUMBER': parts.append(str(t.value))
            elif t.type == 'MUL': parts.append('*')
            elif t.type == 'DIV': parts.append('/')
            elif t.type == 'CARET' and self.peek() and self.peek().type == 'NUMBER':
                parts.append('^'); parts.append(self.consume().value)
            elif t.type == 'MINUS': parts.append('-')
            else: break
        self.consume('RBRACKET')
        return ''.join(parts)

    def _parse_shape_annotation(self):
        """Parst `Scalar`, `Vector[N]`, `Matrix[M,N]`, `Tensor[d1,...]`, `Graph[N,E]`.
        Dimensionen sind Integer-Literale oder Identifier (symbolisch, Caller-bound).
        Liefert ein Tuple (kind, dims) — kind ∈ {scalar,vector,matrix,tensor,graph}."""
        name_tok = self.consume('ID')
        name = name_tok.value.lower()
        if name == 'scalar':
            return ('scalar', [])
        if not self.peek() or self.peek().type != 'LBRACKET':
            raise CompileError(
                f"Shape-Annotation '{name_tok.value}' braucht Dimensionen in [...] "
                f"(z. B. {name_tok.value}[N]).",
                line=name_tok.line,
            )
        self.consume('LBRACKET')
        dims = []
        while self.peek() and self.peek().type != 'RBRACKET':
            t = self.consume()
            if t.type == 'NUMBER':
                try:
                    dims.append(int(t.value))
                except ValueError:
                    raise CompileError(
                        f"Shape-Dimension muss ganzzahlig oder symbolisch sein, bekam {t.value!r}.",
                        line=t.line,
                    )
            elif t.type == 'ID':
                dims.append(t.value)
            elif t.type == 'COMMA':
                continue
            else:
                break
        self.consume('RBRACKET')
        if name == 'vector' and len(dims) != 1:
            raise CompileError(
                f"Vector[...] erwartet genau 1 Dimension, bekam {len(dims)}.",
                line=name_tok.line,
            )
        if name == 'matrix' and len(dims) != 2:
            raise CompileError(
                f"Matrix[...] erwartet genau 2 Dimensionen, bekam {len(dims)}.",
                line=name_tok.line,
            )
        if name == 'graph' and len(dims) != 2:
            raise CompileError(
                f"Graph[...] erwartet genau 2 Dimensionen (N_nodes, E_edges), bekam {len(dims)}.",
                line=name_tok.line,
            )
        if name == 'sequence':
            if len(dims) != 1 or not isinstance(dims[0], str):
                raise CompileError(
                    f"Sequence[...] erwartet genau einen Typ-Identifier (DNA, RNA, Protein), "
                    f"bekam {dims!r}.",
                    line=name_tok.line,
                )
            if dims[0].upper() not in ('DNA', 'RNA', 'PROTEIN'):
                raise CompileError(
                    f"Sequence[{dims[0]}]: erwartet DNA, RNA oder Protein, bekam {dims[0]!r}.",
                    line=name_tok.line,
                )
            dims = [dims[0].upper()]  # normalisiert
        if name == 'qubit' and len(dims) != 1:
            raise CompileError(
                f"Qubit[...] erwartet genau 1 Dimension (Anzahl Qubits), bekam {len(dims)}.",
                line=name_tok.line,
            )
        if name == 'circuit' and len(dims) != 2:
            raise CompileError(
                f"Circuit[...] erwartet genau 2 Dimensionen (N_qubits, N_gates), bekam {len(dims)}.",
                line=name_tok.line,
            )
        if name == 'statevec' and len(dims) != 1:
            raise CompileError(
                f"StateVec[...] erwartet genau 1 Dimension (N Qubits, Laenge=2^N), bekam {len(dims)}.",
                line=name_tok.line,
            )
        return (name, dims)

    def _parse_signature_annotation(self):
        """Nach `:` oder `->`: parst Unit ([m]) ODER Shape (Vector[N]).
        Liefert (unit_str_or_None, shape_list_or_None) — genau eine Komponente ist non-None."""
        tok = self.peek()
        if tok and tok.type == 'LBRACKET':
            self.consume('LBRACKET')
            return self._parse_unit_bracket_inline(), None
        if tok and tok.type == 'ID' and tok.value.lower() in self._SHAPE_TYPES:
            return None, self._parse_shape_annotation()
        raise CompileError(
            f"Ungueltige Signatur-Annotation: erwartete `[unit]` oder `Scalar`/`Vector[N]`/`Matrix[M,N]`/`Tensor[...]`.",
            line=tok.line if tok else None,
        )

    def parse_function_def(self, is_pub=False):
        start_line = self.peek().line
        self.consume('FN')
        name = self.consume('ID').value
        # Optional: Typ-Parameter `<T, U, ...>` (polymorphe Einheiten/Shapes).
        type_params = []
        if self.peek() and self.peek().type == 'LT':
            self.consume('LT')
            if self.peek() and self.peek().type != 'GT':
                type_params.append(self.consume('ID').value)
                while self.peek() and self.peek().type == 'COMMA':
                    self.consume('COMMA')
                    type_params.append(self.consume('ID').value)
            self.consume('GT')
        self.consume('LPAREN')
        args = []
        arg_units = []
        arg_shapes = []
        has_annotations = False
        if self.peek().type != 'RPAREN':
            args.append(self.consume('ID').value)
            u, sh = None, None
            if self.peek() and self.peek().type == 'COLON':
                self.consume('COLON')
                u, sh = self._parse_signature_annotation()
                has_annotations = True
            arg_units.append(u)
            arg_shapes.append(sh)
            while self.peek().type == 'COMMA':
                self.consume('COMMA')
                args.append(self.consume('ID').value)
                u, sh = None, None
                if self.peek() and self.peek().type == 'COLON':
                    self.consume('COLON')
                    u, sh = self._parse_signature_annotation()
                    has_annotations = True
                arg_units.append(u)
                arg_shapes.append(sh)
        self.consume('RPAREN')
        return_unit = None
        return_shape = None
        if self.peek() and self.peek().type == 'RETURNS':
            self.consume('RETURNS')
            return_unit, return_shape = self._parse_signature_annotation()
            has_annotations = True
        self.consume('LBRACE')
        body = []
        while self.peek().type != 'RBRACE':
            body.append(self.parse_statement())
        self.consume('RBRACE')
        node = FunctionDef(
            name, args, body,
            arg_units=arg_units if has_annotations and any(u is not None for u in arg_units) else None,
            return_unit=return_unit,
            arg_shapes=arg_shapes if has_annotations and any(s is not None for s in arg_shapes) else None,
            return_shape=return_shape,
            is_pub=is_pub,
            type_params=type_params,
        )
        node.line = start_line
        return node

    def parse_assignment(self):
        start_line = self.peek().line
        target_node = self.parse_atom()
        self.consume('ASSIGN')
        value = self.parse_expression()
        if isinstance(target_node, Subscript):
            node = ItemAssignment(target_node, value)
            node.line = start_line
            return node
        target_name = ""
        if isinstance(target_node, IndexedVariable):
            target_name = f"{target_node.name}_{target_node.indices}"
        elif isinstance(target_node, Identifier):
            target_name = target_node.name
        else:
            raise CompileError(
                f"Invalid assignment target: {type(target_node).__name__}. Expected: identifier or index.",
                line=start_line,
            )
        node = Assignment(target_name, value)
        node.line = start_line
        return node

    def parse_expression(self):
        # Lowest precedence: logical operators (or, xor, and, nand, nor, xnor, not)
        return self.parse_logical_or()

    def parse_logical_or(self):
        left = self.parse_logical_xor()
        while self.peek() and self.peek().type == 'OR':
            op_tok = self.consume()
            right = self.parse_logical_xor()
            left = BinaryOp(left, 'or', right)
            left.line = op_tok.line
        return left

    def parse_logical_xor(self):
        left = self.parse_logical_and()
        while self.peek() and self.peek().type == 'XOR':
            op_tok = self.consume()
            right = self.parse_logical_and()
            left = BinaryOp(left, 'xor', right)
            left.line = op_tok.line
        return left

    def parse_logical_and(self):
        left = self.parse_logical_not()
        while self.peek() and self.peek().type in ['AND', 'NAND', 'NOR', 'XNOR']:
            op_tok = self.consume()
            op_val = op_tok.value  # 'and', 'nand', 'nor', 'xnor'
            right = self.parse_logical_not()
            left = BinaryOp(left, op_val, right)
            left.line = op_tok.line
        return left

    def parse_logical_not(self):
        if self.peek() and self.peek().type == 'NOT':
            op_tok = self.consume()
            operand = self.parse_logical_not()
            node = UnaryOp('not', operand)
            node.line = op_tok.line
            return node
        return self.parse_comparison()

    def parse_comparison(self):
        left = self.parse_term_expr()
        while self.peek() and self.peek().type in ['EQ', 'NEQ', 'LT', 'GT', 'LE', 'GE']:
            op_tok = self.consume()
            right = self.parse_term_expr()
            left = BinaryOp(left, op_tok.value, right)
            left.line = op_tok.line
        return left

    def parse_term_expr(self):
        left = self.parse_factor_expr()
        while self.peek() and self.peek().type in ['PLUS', 'MINUS']:
            op_tok = self.consume()
            right = self.parse_factor_expr()
            left = BinaryOp(left, op_tok.value, right)
            left.line = op_tok.line
        return left

    def parse_factor_expr(self):
        left = self.parse_power_expr()
        while self.peek() and self.peek().type in ['MUL', 'DIV', 'AT']:
            op_tok = self.consume()
            right = self.parse_power_expr()
            left = BinaryOp(left, op_tok.value, right)
            left.line = op_tok.line
        return left

    def parse_power_expr(self):
        left = self.parse_atom()
        while self.peek() and self.peek().type == 'CARET':
            op_tok = self.consume()
            right = self.parse_atom()
            left = BinaryOp(left, op_tok.value, right)
            left.line = op_tok.line
        return left

    def parse_atom(self):
        token = self.peek()
        line_at = token.line if token else 1
        node = None
        if token.type == 'NUMBER':
            num_tok = self.consume()
            value = float(num_tok.value) if ('.' in num_tok.value or 'e' in num_tok.value.lower()) else int(num_tok.value)
            if self.peek() and self.peek().type == 'LBRACKET':
                self.consume('LBRACKET')
                unit_parts = []
                while self.peek() and self.peek().type != 'RBRACKET':
                    t = self.consume()
                    if t.type == 'ID':
                        unit_parts.append(t.value)
                    elif t.type == 'NUMBER':
                        unit_parts.append(str(t.value))
                    elif t.type == 'MUL':
                        unit_parts.append('*')
                    elif t.type == 'DIV':
                        unit_parts.append('/')
                    elif t.type == 'CARET' and self.peek() and self.peek().type == 'NUMBER':
                        unit_parts.append('^')
                        unit_parts.append(self.consume().value)
                    else:
                        break
                self.consume('RBRACKET')
                unit_str = ''.join(unit_parts) if unit_parts else ''
                node = Quantity(value, unit_str)
            else:
                node = Literal(value)
            node.line = line_at
        elif token.type == 'QUATERNION':
            self.consume()
            suffix = token.value[-1]
            val = float(token.value[:-1])
            node = QuaternionLiteral(val, suffix)
            node.line = line_at
        elif token.type == 'COMPLEX':
            self.consume()
            val = float(token.value[:-1])
            node = QuaternionLiteral(val, 'i')
            node.line = line_at
        elif token.type == 'RAWSTRING':
            self.consume()
            node = Literal(token.value, raw=True)
            node.line = line_at
        elif token.type == 'STRING':
            self.consume()
            node = Literal(token.value.strip('"'))
            node.line = line_at
        elif token.type == 'LBRACKET':
            self.consume('LBRACKET')
            elements = []
            if self.peek().type != 'RBRACKET':
                elements.append(self.parse_expression())
                while self.peek().type == 'COMMA':
                    self.consume('COMMA')
                    elements.append(self.parse_expression())
            self.consume('RBRACKET')
            node = VectorLiteral(elements)
            node.line = line_at
        elif token.type == 'LBRACE':
            # Dict-Literal: { "k": v, "k2": v2 } oder { id: v, ... }
            self.consume('LBRACE')
            keys = []
            values = []
            if self.peek() and self.peek().type != 'RBRACE':
                while True:
                    k = self.parse_expression()
                    self.consume('COLON')
                    v = self.parse_expression()
                    keys.append(k)
                    values.append(v)
                    if self.peek() and self.peek().type == 'COMMA':
                        self.consume('COMMA')
                    else:
                        break
            self.consume('RBRACE')
            node = DictLiteral(keys, values)
            node.line = line_at
        elif token.type == 'LPAREN':
            self.consume('LPAREN')
            node = self.parse_expression()
            self.consume('RPAREN')
            if hasattr(node, 'line') and node.line is None:
                node.line = line_at
        elif token.type == 'PIPE':
            self.consume('PIPE')
            inner = self.parse_expression()
            self.consume('PIPE')
            node = Call(Identifier('abs'), [inner], [], [])
            node.line = line_at
        elif token.type == 'ID':
            name_token = self.consume()
            value = name_token.value
            node = Identifier(value)
            node.line = line_at
            if self.peek() and self.peek().type in ['CARET', 'UNDERSCORE']:
                if self.peek(1) and self.peek(1).type == 'ID':
                    indices = ""
                    while self.peek() and self.peek().type in ['CARET', 'UNDERSCORE'] and self.peek(1) and self.peek(1).type == 'ID':
                        self.consume()
                        idx_token = self.consume('ID')
                        indices += idx_token.value
                    node = IndexedVariable(value, indices)
                    node.line = line_at
            elif '_' in value and self._looks_like_ricci_suffix(value):
                name, _, indices = value.rpartition('_')
                if name and indices:
                    node = IndexedVariable(name, indices)
                    node.line = line_at
        elif token.type == 'GRAD':
            self.consume('GRAD')
            node = Identifier('grad')
            node.line = line_at
        elif token.type == 'EINSUM':
            self.consume('EINSUM')
            node = Identifier('einsum')
            node.line = line_at
        elif token.type == 'MINUS':
            self.consume('MINUS')
            operand = self.parse_atom()
            if isinstance(operand, Literal) and isinstance(operand.value, (int, float)):
                node = Literal(-operand.value)
                node.line = line_at
                return node
            node = BinaryOp(Literal(0), '-', operand)
            node.line = line_at
            return node
        else:
            raise CompileError(
                f"Unerwartetes Token im Ausdruck: {token.type} '{getattr(token, 'value', '')}'.",
                line=line_at,
            )

        # Handle Postfix Factorial: n!
        if self.peek() and self.peek().type == 'FACTORIAL':
            fact_tok = self.consume('FACTORIAL')
            node = PostfixFactorial(node)
            node.line = fact_tok.line

        # Handle Method Chaining, Modifiers & Subscripts
        while self.peek() and self.peek().type in ['LPAREN', 'DOT', 'MODIFIER', 'LBRACKET']:
            
            if self.peek().type == 'LPAREN':
                self.consume('LPAREN')
                args, kwargs = self.parse_call_args()
                self.consume('RPAREN')
                node = Call(node, args, kwargs, [])
                
            elif self.peek().type == 'DOT':
                self.consume('DOT')
                member_name = self.consume('ID').value
                if self.peek() and self.peek().type == 'LPAREN':
                    self.consume('LPAREN')
                    args, kwargs = self.parse_call_args()
                    self.consume('RPAREN')
                    # Keep it as a method call: obj.method(args)
                    node = Call(MemberAccess(node, member_name), args, kwargs, [])
                else:
                    node = MemberAccess(node, member_name)
                
            elif self.peek().type == 'MODIFIER':
                modifier = self.consume('MODIFIER').value.strip('.') 
                self.consume('LPAREN')
                self.consume('RPAREN')
                if isinstance(node, Call):
                    node.modifiers.append(modifier)
                else:
                    node = Call(Identifier(modifier), [node], [], [modifier])

            elif self.peek().type == 'LBRACKET':
                self.consume('LBRACKET')
                # Subscript ODER Slice (Python-Style x[a:b], x[:b], x[a:], x[::s], ...).
                # Wir sammeln bis zu 3 durch ':' getrennte Komponenten; jede darf leer sein.
                parts = []
                # erste Komponente
                if self.peek().type not in ('COLON', 'RBRACKET'):
                    parts.append(self.parse_expression())
                else:
                    parts.append(None)
                while self.peek() and self.peek().type == 'COLON':
                    self.consume('COLON')
                    if self.peek().type not in ('COLON', 'RBRACKET'):
                        parts.append(self.parse_expression())
                    else:
                        parts.append(None)
                self.consume('RBRACKET')
                if len(parts) == 1:
                    # Klassisches Subscript
                    index = parts[0]
                else:
                    # Slice: [start, stop] oder [start, stop, step]
                    start = parts[0]
                    stop = parts[1] if len(parts) > 1 else None
                    step = parts[2] if len(parts) > 2 else None
                    index = Slice(start=start, stop=stop, step=step)
                    index.line = line_at
                node = Subscript(node, index)
                if getattr(node, 'line', None) is None:
                    node.line = line_at
        if getattr(node, 'line', None) is None:
            node.line = line_at
        return node

    def parse_call_args(self):
        args = []
        kwargs = []
        if self.peek().type != 'RPAREN':
            while True:
                # Named argument: ID = expression
                if self.peek().type == 'ID' and self.peek(1) and self.peek(1).type == 'ASSIGN':
                    name = self.consume('ID').value
                    self.consume('ASSIGN')
                    value = self.parse_expression()
                    kwargs.append((name, value))
                else:
                    args.append(self.parse_expression())
                
                if self.peek().type == 'COMMA':
                    self.consume('COMMA')
                else:
                    break
        return args, kwargs


