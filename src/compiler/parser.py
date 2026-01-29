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

    def parse(self) -> Program:
        statements = []
        while self.pos < len(self.tokens):
            statements.append(self.parse_statement())
        return Program(statements)

    def parse_statement(self):
        token = self.peek()
        if token.type == 'FN':
            return self.parse_function_def()
        elif token.type == 'RETURN':
            self.consume('RETURN')
            expr = self.parse_expression()
            return ReturnStmt(expr)
        elif token.type == 'ID' and self.peek(1) and self.peek(1).type == 'ASSIGN':
            return self.parse_assignment()
        else:
            return self.parse_expression()

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
        target = self.consume('ID').value
        self.consume('ASSIGN')
        value = self.parse_expression()
        return Assignment(target, value)

    def parse_expression(self):
        left = self.parse_term()
        
        while self.peek() and self.peek().type in ['PLUS', 'MINUS']:
            op = self.consume().value
            right = self.parse_term()
            left = BinaryOp(left, op, right)
            
        return left

    def parse_term(self):
        left = self.parse_factor()
        
        while self.peek() and self.peek().type in ['MUL', 'DIV']:
            op = self.consume().value
            right = self.parse_factor()
            left = BinaryOp(left, op, right)
            
        return left

    def parse_factor(self):
        token = self.peek()
        
        if token.type == 'NUMBER':
            self.consume()
            return Literal(float(token.value) if '.' in token.value else int(token.value))
        elif token.type == 'STRING':
            self.consume()
            return Literal(token.value.strip('"'))
        elif token.type == 'LBRACKET':
            return self.parse_vector()
        elif token.type == 'LPAREN':
            # Check if this is a lambda: (a, b) => ...
            # Simple heuristic: if we see LPAREN -> ID -> COMMA or LPAREN -> ID -> CPAREN -> ARROW
            # For now, let's keep it simple: Parenthesized expression
            self.consume('LPAREN')
            expr = self.parse_expression()
            self.consume('RPAREN')
            return expr
        elif token.type == 'ID':
            return self.parse_identifier_or_call()
        else:
            raise Exception(f"Unexpected token in expression: {token}")

    def parse_vector(self):
        self.consume('LBRACKET')
        elements = []
        if self.peek().type != 'RBRACKET':
            elements.append(self.parse_expression())
            while self.peek().type == 'COMMA':
                self.consume('COMMA')
                elements.append(self.parse_expression())
        self.consume('RBRACKET')
        return VectorLiteral(elements)

    def parse_identifier_or_call(self):
        name_token = self.consume('ID')
        node = Identifier(name_token.value)
        
        # Handle method calls / chaining: name.method() or name()
        while self.peek() and (self.peek().type == 'LPAREN' or self.peek().type == 'DOT' or self.peek().type == 'MODIFIER'):
            
            if self.peek().type == 'LPAREN':
                self.consume('LPAREN')
                args = []
                if self.peek().type != 'RPAREN':
                    args.append(self.parse_expression())
                    while self.peek().type == 'COMMA':
                        self.consume('COMMA')
                        args.append(self.parse_expression())
                self.consume('RPAREN')
                node = Call(node, args, [])
                
            elif self.peek().type == 'DOT':
                self.consume('DOT')
                method_name = self.consume('ID').value
                # Usually followed by call parens
                self.consume('LPAREN')
                args = []
                if self.peek().type != 'RPAREN':
                    args.append(self.parse_expression())
                    while self.peek().type == 'COMMA':
                        self.consume('COMMA')
                        args.append(self.parse_expression())
                self.consume('RPAREN')
                # Wrap the previous node as the 'self' argument or special method call structure
                # For simplicity, we'll chain it: Call(method_name, [previous_node, *args])
                # Or keep it as a dotted access in AST. Let's make a specialized Call structure handling.
                # Actually, spec has `data.map(...)`.
                # Let's treat it as: Call(Identifier(method_name), args, [implicit_first_arg=node])
                # But to keep AST simple for codegen:
                # We can construct proper Python calls: method_name(node, *args)
                node = Call(Identifier(method_name), [node] + args, [])
                
            elif self.peek().type == 'MODIFIER':
                modifier = self.consume('MODIFIER').value.strip('.') # e.g. "gpu"
                self.consume('LPAREN')
                self.consume('RPAREN')
                if isinstance(node, Call):
                    node.modifiers.append(modifier)
                else:
                    # Modifier on a variable? e.g. matrix.gpu()
                    # Treat as a call to `.cpu()`/`.gpu()` which is identity function with side effect/cast
                    node = Call(Identifier(modifier), [node], [modifier])
                    
        return node
