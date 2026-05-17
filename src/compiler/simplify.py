"""
Symbolische Vereinfachung für Dedekind: AST-zu-AST-Pass.
Phase 1 MVP: Konstantenfaltung, x+0, x*1, 0*x, x-0, x^0, x^1.
Keine externen Abhängigkeiten.
"""
from .ast_nodes import (
    Node, Program, BinaryOp, Call, Identifier, Literal, Quantity,
    Assignment, ReturnStmt, FunctionDef, IfStmt, WhileStmt, ForStmt,
    MemberAccess, Subscript, VectorLiteral, Lambda, ItemAssignment,
)


def _is_numeric(node):
    """True wenn node ein Literal mit numerischem Wert ist."""
    if not isinstance(node, Literal):
        return False
    v = node.value
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _literal_value(node):
    """Wert eines Literal-Knotens; None wenn nicht Literal."""
    if isinstance(node, Literal):
        return node.value
    return None


def _make_literal(value, line=None):
    """Erzeugt Literal-Knoten mit optionaler Zeile."""
    return Literal(value, line=line)


def _constant_fold(op, a, b):
    """Konstantenfaltung für zwei numerische Werte. None bei Division durch 0."""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    if isinstance(a, bool) or isinstance(b, bool):
        return None
    try:
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            if b == 0:
                return None
            return a / b
        if op == "^" or op == "**":
            return a ** b
    except (OverflowError, ValueError):
        return None
    return None


class SimplifyVisitor:
    """Visitor: vereinfacht Ausdrücke im Dedekind-AST."""

    def visit(self, node):
        if node is None:
            return None
        name = type(node).__name__
        method = getattr(self, f"visit_{name}", self._generic_visit)
        return method(node)

    def _generic_visit(self, node):
        return node

    def visit_Program(self, node):
        return Program(statements=[self.visit(s) for s in node.statements], line=node.line)

    def visit_FunctionDef(self, node):
        return FunctionDef(
            name=node.name,
            args=node.args,
            body=[self.visit(s) for s in node.body],
            arg_units=getattr(node, "arg_units", None),
            return_unit=getattr(node, "return_unit", None),
            arg_shapes=getattr(node, "arg_shapes", None),
            return_shape=getattr(node, "return_shape", None),
            line=node.line,
        )

    def visit_Assignment(self, node):
        return Assignment(target=node.target, value=self.visit(node.value), line=node.line)

    def visit_ReturnStmt(self, node):
        return ReturnStmt(value=self.visit(node.value), line=node.line)

    def visit_IfStmt(self, node):
        return IfStmt(
            condition=self.visit(node.condition),
            then_branch=[self.visit(s) for s in node.then_branch],
            else_branch=[self.visit(s) for s in node.else_branch] if node.else_branch else None,
            line=node.line,
        )

    def visit_WhileStmt(self, node):
        return WhileStmt(
            condition=self.visit(node.condition),
            body=[self.visit(s) for s in node.body],
            line=node.line,
        )

    def visit_ForStmt(self, node):
        return ForStmt(
            variable=node.variable,
            collection=self.visit(node.collection),
            body=[self.visit(s) for s in node.body],
            line=node.line,
        )

    def visit_ItemAssignment(self, node):
        return ItemAssignment(
            target=node.target,
            value=self.visit(node.value),
            line=node.line,
        )

    def visit_BinaryOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = node.op
        line = node.line

        # Konstantenfaltung: Literal op Literal
        la, lb = _literal_value(left), _literal_value(right)
        if _is_numeric(left) and _is_numeric(right):
            folded = _constant_fold(op, la, lb)
            if folded is not None:
                return _make_literal(folded, line)

        # Arithmetische Vereinfachungen
        if op == "+":
            if _is_numeric(left) and la == 0:
                return right
            if _is_numeric(right) and lb == 0:
                return left
        if op == "-":
            if _is_numeric(right) and lb == 0:
                return left
        if op == "*":
            # Nur falten wenn beide Literale (x*0 mit Tensor x darf nicht zu 0 werden)
            if _is_numeric(left):
                if la == 0 and _is_numeric(right):
                    return _make_literal(0, line)
                if la == 1:
                    return right
            if _is_numeric(right):
                if lb == 0 and _is_numeric(left):
                    return _make_literal(0, line)
                if lb == 1:
                    return left
        if op == "/":
            if _is_numeric(left) and la == 0:
                return _make_literal(0, line)
            if _is_numeric(right) and lb == 1:
                return left
        if op in ("^", "**"):
            if _is_numeric(right):
                if lb == 0:
                    return _make_literal(1, line)
                if lb == 1:
                    return left
            if _is_numeric(left) and la == 0 and _is_numeric(right) and lb > 0:
                return _make_literal(0, line)

        return BinaryOp(left=left, op=op, right=right, line=line)

    def visit_Call(self, node):
        func = self.visit(node.func_name)
        args = [self.visit(a) for a in node.args]
        kwargs = [(k, self.visit(v)) for k, v in node.kwargs]
        return Call(func_name=func, args=args, kwargs=kwargs, modifiers=node.modifiers, line=node.line)

    def visit_Literal(self, node):
        return node

    def visit_Identifier(self, node):
        return node

    def visit_Quantity(self, node):
        return node

    def visit_QuaternionLiteral(self, node):
        return node

    def visit_VectorLiteral(self, node):
        return VectorLiteral(elements=[self.visit(e) for e in node.elements], line=node.line)

    def visit_DictLiteral(self, node):
        from .ast_nodes import DictLiteral
        return DictLiteral(
            keys=[self.visit(k) for k in node.keys],
            values=[self.visit(v) for v in node.values],
            line=node.line,
        )

    def visit_Lambda(self, node):
        return Lambda(arg=node.arg, body=self.visit(node.body), line=node.line)

    def visit_MemberAccess(self, node):
        return MemberAccess(obj=self.visit(node.obj), member=node.member, line=node.line)

    def visit_Subscript(self, node):
        return Subscript(value=self.visit(node.value), index=self.visit(node.index), line=node.line)

    def visit_IndexedVariable(self, node):
        return node


def simplify_program(ast):
    """Haupt-API: vereinfacht das gesamte Programm."""
    return SimplifyVisitor().visit(ast)
