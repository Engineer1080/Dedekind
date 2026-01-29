from typing import List
from .ast_nodes import *
from .lexer import Token, Lexer

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def consume(self, type: str = None):
        if self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            if type is None or token.type == type:
                self.pos += 1
                return token
        raise Exception(f"Expected token type {type}, but found {self.tokens[self.pos] if self.pos < len(self.tokens) else 'EOF'}")

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
            return self.parse_function_def()
        elif token.type == 'RETURN':
            self.consume('RETURN')
            expr = self.parse_expression()
            return ReturnStmt(expr)
        elif token.type == 'IF':
            return self.parse_if_stmt()
        elif token.type == 'WHILE':
            return self.parse_while_stmt()
        elif token.type == 'FOR':
            return self.parse_for_stmt()
        else:
            return self.parse_expression()

    def parse_if_stmt(self):
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
            
        return IfStmt(condition, then_branch, else_branch)

    def parse_while_stmt(self):
        self.consume('WHILE')
        condition = self.parse_expression()
        self.consume('LBRACE')
        body = []
        while self.peek().type != 'RBRACE':
            body.append(self.parse_statement())
        self.consume('RBRACE')
        return WhileStmt(condition, body)
        
    def parse_for_stmt(self):
        self.consume('FOR')
        var_name = self.consume('ID').value
        if self.peek().type == 'ID' and self.peek().value == 'in': 
             self.consume() # consume 'in' (lexer sees it as ID)
        elif self.tokens[self.pos].value == 'in': # Fallback if 'in' keyword handling varies
             self.consume()

        collection = self.parse_expression()
        self.consume('LBRACE')
        body = []
        while self.peek().type != 'RBRACE':
            body.append(self.parse_statement())
        self.consume('RBRACE')
        return ForStmt(var_name, collection, body)

    def parse_function_def(self):
        self.consume('FN')
        name = self.consume('ID').value
        self.consume('LPAREN')
        args = []
        if self.peek().type != 'RPAREN':
            args.append(self.consume('ID').value)
            while self.peek().type == 'COMMA':
                self.consume('COMMA')
                args.append(self.consume('ID').value)
        self.consume('RPAREN')
        self.consume('LBRACE')
        body = []
        while self.peek().type != 'RBRACE':
            body.append(self.parse_statement())
        self.consume('RBRACE')
        return FunctionDef(name, args, body)

    def parse_assignment(self):
        # The target can be an IndexedVariable, a plain ID, or a Subscript
        target_node = self.parse_atom()
        
        self.consume('ASSIGN')
        value = self.parse_expression()

        if isinstance(target_node, Subscript):
            return ItemAssignment(target_node, value)
        
        target_name = ""
        if isinstance(target_node, IndexedVariable):
            target_name = f"{target_node.name}_{target_node.indices}"
        elif isinstance(target_node, Identifier):
            target_name = target_node.name
        else:
            raise Exception(f"Invalid assignment target: {target_node}")
            
        return Assignment(target_name, value)

    def parse_expression(self):
        # Lowest precedence: Comparisons
        return self.parse_comparison()
        
    def parse_comparison(self):
        left = self.parse_term_expr()
        
        while self.peek() and self.peek().type in ['EQ', 'NEQ', 'LT', 'GT', 'LE', 'GE']:
            op = self.consume().value
            right = self.parse_term_expr()
            left = BinaryOp(left, op, right)
            
        return left

    def parse_term_expr(self):
        # Addition/Subtraction
        left = self.parse_factor_expr()
        
        while self.peek() and self.peek().type in ['PLUS', 'MINUS']:
            op = self.consume().value
            right = self.parse_factor_expr()
            left = BinaryOp(left, op, right)
            
        return left

    def parse_factor_expr(self):
        # Multiplication/Division
        left = self.parse_power_expr()
        
        while self.peek() and self.peek().type in ['MUL', 'DIV']:
            op = self.consume().value
            right = self.parse_power_expr()
            left = BinaryOp(left, op, right)
            
        return left

    def parse_power_expr(self):
        # Exponentiation (higher precedence than MUL)
        left = self.parse_atom()
        
        while self.peek() and self.peek().type == 'CARET':
            # Check if this is a Ricci index (followed by ID) or a Power (followed by anything else)
            # Actually, to be safe, if parse_atom didn't consume it as an IndexedVariable,
            # we should treat it as a power here if possible.
            op = self.consume().value
            right = self.parse_atom()
            left = BinaryOp(left, op, right)
            
        return left

    def parse_atom(self):
        token = self.peek()
        
        node = None
        if token.type == 'NUMBER':
            self.consume()
            node = Literal(float(token.value) if '.' in token.value else int(token.value))
        elif token.type == 'QUATERNION':
            self.consume()
            # Convert "5.0j" or "3k" to QuaternionLiteral
            suffix = token.value[-1]
            val = float(token.value[:-1])
            node = QuaternionLiteral(val, suffix)
        elif token.type == 'COMPLEX':
            self.consume()
            # In Fourier, 'i' is the first quaternion component
            val = float(token.value[:-1])
            node = QuaternionLiteral(val, 'i')
        elif token.type == 'STRING':
            self.consume()
            node = Literal(token.value.strip('"'))
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
        elif token.type == 'LPAREN':
            self.consume('LPAREN')
            node = self.parse_expression()
            self.consume('RPAREN')
        elif token.type == 'ID':
            name_token = self.consume()
            value = name_token.value
            node = Identifier(value)
            
            # Ricci: A^ij / A_ij (entweder getrennte Tokens oder ein ID "A_ij")
            if self.peek() and self.peek().type in ['CARET', 'UNDERSCORE']:
                if self.peek(1) and self.peek(1).type == 'ID':
                    indices = ""
                    while self.peek() and self.peek().type in ['CARET', 'UNDERSCORE'] and self.peek(1) and self.peek(1).type == 'ID':
                        self.consume()
                        idx_token = self.consume('ID')
                        indices += idx_token.value
                    node = IndexedVariable(value, indices)
            elif '_' in value and self._looks_like_ricci_suffix(value):
                # Ein Token "A_ij" -> IndexedVariable
                name, _, indices = value.rpartition('_')
                if name and indices:
                    node = IndexedVariable(name, indices)
        elif token.type == 'GRAD':
            name_token = self.consume('GRAD')
            node = Identifier('grad')
        elif token.type == 'EINSUM':
            name_token = self.consume('EINSUM')
            node = Identifier('einsum')
        elif token.type == 'MINUS':
            self.consume('MINUS')
            operand = self.parse_atom()
            # We can simplify this by returning a BinaryOp(0, '-', operand) 
            # or better, just wrap it in a special way. For Python gen, -operand works.
            # But since our BinaryOp expects left and right, let's just make it a literal with negative value if it's a number, 
            # or a custom representation.
            if isinstance(operand, Literal) and isinstance(operand.value, (int, float)):
                return Literal(-operand.value)
            return BinaryOp(Literal(0), '-', operand)
        else:
            raise Exception(f"Unexpected token in expression: {token}")

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
                index = self.parse_expression()
                self.consume('RBRACKET')
                node = Subscript(node, index)
                    
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


