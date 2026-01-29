import re
from typing import List, Tuple

class Token:
    def __init__(self, type: str, value: str, line: int):
        self.type = type
        self.value = value
        self.line = line

    def __repr__(self):
        return f"Token({self.type}, '{self.value}', {self.line})"

class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.tokens: List[Token] = []
        self.current_line = 1
        self.pos = 0

    def tokenize(self) -> List[Token]:
        # Normalize line endings to prevent Windows \r issues
        self.source = self.source.replace('\r\n', '\n').replace('\r', '\n')
        
        token_specs = [
            ('QUATERNION', r'\d+(\.\d+)?[jk]\b'),   # Quaternion component (j or k)
            ('COMPLEX',  r'\d+(\.\d+)?i\b'),      # Complex number 5.0i
            ('NUMBER',   r'\d+(\.\d+)?([eE][+-]?\d+)?'), # scientific notation 지원
            ('STRING',   r'"[^"]*"'),           # String literal
            ('MODIFIER', r'\.(gpu|cpu|single)'), # Modifiers .gpu, .cpu, .single
            ('ARROW',    r'=>'),                # Arrow operator
            ('EQ',       r'=='),                # Equal
            ('NEQ',      r'!='),                # Not equal
            ('LE',       r'<='),                # Less or equal
            ('GE',       r'>='),                # Greater or equal
            ('ASSIGN',   r'='),                 # Assignment operator (CHECK ORDER: EQ before ASSIGN)
            ('LT',       r'<'),                 # Less than
            ('GT',       r'>'),                 # Greater than
            ('PLUS',     r'\+'),                # Add
            ('MINUS',    r'-'),                 # Subtract
            ('MUL',      r'\*'),                # Multiply
            ('COMMENT',  r'//.*'),              # Comments
            ('DIV',      r'/'),                 # Divide
            ('LPAREN',   r'\('),                # (
            ('RPAREN',   r'\)'),                # )
            ('LBRACE',   r'\{'),                # {
            ('RBRACE',   r'\}'),                # }
            ('LBRACKET', r'\['),                # [
            ('RBRACKET', r'\]'),                # ]
            ('COMMA',    r','),                 # ,
            ('DOT',      r'\.'),                # .
            ('CARET',    r'\^'),                # ^ (contravariant index)
            ('UNDERSCORE', r'_'),               # _ (covariant index)
            ('ID',       r'[A-Za-z][A-Za-z0-9]*'), # Identifiers (No underscores to allow _ as index op)
            ('NEWLINE',  r'\n'),                # Line endings
            ('SKIP',     r'[ \t\r]+'),           # Skip over spaces, tabs, and carriage returns
            ('MISMATCH', r'.'),                 # Any other character
        ]

        # Join all patterns
        token_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specs)
        get_token = re.compile(token_regex).match

        tokens = []
        line = 1
        pos = 0
        mo = get_token(self.source)
        
        while mo is not None:
            kind = mo.lastgroup
            value = mo.group(kind)
            
            if kind == 'NEWLINE':
                line += 1
            elif kind == 'SKIP' or kind == 'COMMENT':
                pass
            elif kind == 'MISMATCH':
                raise RuntimeError(f'{value!r} unexpected on line {line}')
            else:
                keywords = {'fn', 'return', 'if', 'else', 'while', 'for', 'in', 'grad', 'einsum'}
                if kind == 'ID' and value in keywords:
                    kind = value.upper()
                tokens.append(Token(kind, value, line))
                
            pos = mo.end()
            mo = get_token(self.source, pos)
            
        self.tokens = tokens
        return tokens
