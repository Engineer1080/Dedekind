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
        elif token.type == 'IF':
            return self.parse_if_stmt()
        elif token.type == 'WHILE':
            return self.parse_while_stmt()
        elif token.type == 'FOR':
            return self.parse_for_stmt()
        elif token.type == 'ID' and self.peek(1) and self.peek(1).type == 'ASSIGN':
            return self.parse_assignment()
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
        target = self.consume('ID').value
        self.consume('ASSIGN')
        value = self.parse_expression()
        return Assignment(target, value)

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
        left = self.parse_atom()
        
        while self.peek() and self.peek().type in ['MUL', 'DIV']:
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
            name_token = self.consume('ID')
            node = Identifier(name_token.value)
        else:
            raise Exception(f"Unexpected token in expression: {token}")

        # Handle Method Chaining & Modifiers
        while self.peek() and (self.peek().type == 'LPAREN' or self.peek().type == 'DOT' or self.peek().type == 'MODIFIER'):
            
            if self.peek().type == 'LPAREN':
                self.consume('LPAREN')
                args, kwargs = self.parse_call_args()
                self.consume('RPAREN')
                node = Call(node, args, kwargs, [])
                
            elif self.peek().type == 'DOT':
                self.consume('DOT')
                method_name = self.consume('ID').value
                self.consume('LPAREN')
                args, kwargs = self.parse_call_args()
                self.consume('RPAREN')
                node = Call(Identifier(method_name), [node] + args, kwargs, [])
                
            elif self.peek().type == 'MODIFIER':
                modifier = self.consume('MODIFIER').value.strip('.') 
                self.consume('LPAREN')
                self.consume('RPAREN')
                if isinstance(node, Call):
                    node.modifiers.append(modifier)
                else:
                    node = Call(Identifier(modifier), [node], [], [modifier])
                    
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


