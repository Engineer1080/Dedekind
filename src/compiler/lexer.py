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
        token_specs = [
            ('NUMBER',   r'\d+(\.\d+)?'),       # Integer or decimal number
            ('STRING',   r'"[^"]*"'),           # String literal
            ('MODIFIER', r'\.(gpu|cpu|single)'), # Modifiers .gpu, .cpu, .single
            ('ARROW',    r'=>'),                # Arrow operator
            ('ASSIGN',   r'='),                 # Assignment operator
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
            ('ID',       r'[A-Za-z_][A-Za-z0-9_]*'), # Identifiers
            ('NEWLINE',  r'\n'),                # Line endings
            ('SKIP',     r'[ \t]+'),            # Skip over spaces and tabs
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
                if kind == 'ID' and value in ['fn', 'return', 'if', 'else', 'for', 'while']:
                    kind = value.upper()
                tokens.append(Token(kind, value, line))
                
            pos = mo.end()
            mo = get_token(self.source, pos)
            
        self.tokens = tokens
        return tokens
