# Symbolic Simplification — Implementation Roadmap

**Dedekind Language**  
Draft: January 2026 · last status update: v1.17.0 (March 2027)

---

## Status Update (v1.17.0)

| Phase | Topic | Status | Delivered in |
|---|---|---|---|
| 1 MVP | `src/compiler/simplify.py` with constant folding, `x+0`, `0+x`, `x*1`, `1*x`, `x*0` → `0`, `x-0`, literal BinOp folding | ✅ done | v1.3 |
| 1 MVP | Hook after parse, before codegen | ✅ done | v1.3 |
| 2 | `x^0`→1, `x^1`→x, recursive sub-expressions, Ricci/Einsum rules | ✅ partial (power rules; deeper tensor algebra not yet) |
| 3 | `--no-units-check` flag (analogous to `--simplify`) | ✅ done | v0.9.5 |
| 4 | SymPy integration as optional dep — `solve_sym`, `simplify_sym`, `series`, `diff_sym`, `integrate_sym` as built-ins | ✅ done | v1.3 – v1.6 |
| 5 | Compile-time unit-dimension simplification | ⚠️ partial (units checker validates; full dimensional reduction not yet) |

**Status:** The essential phases are **covered** through a combination of the internal `simplify.py` (Phase 1) and SymPy bridge built-ins (Phase 4). Remaining fine-grain work (full power/polynomial simplification on the AST, automatic dimension reduction `m*s/s → m`) is conceptually open, but low priority since the SymPy bridge solves this in practice.

For current symbolic features see `Dedekind_Language_Specification.md` (sections on `diff_sym`, `integrate_sym`, `solve_sym`, `simplify_sym`, `series`).

---

## 1. Goal and Benefits

**Symbolic Simplification** refers to the algebraic simplification of expressions **before** code generation (Python/MLIR). Goals:

- **Readability**: Generated code contains shorter, clearer formulas instead of bloated intermediate expressions.
- **Numerical Stability**: Redundant operations (e.g. `(x+1)-1` → `x`) avoid unnecessary rounding errors.
- **Performance**: Fewer operations, better chances for fused ops and compiler optimizations; noticeable for Ricci expressions and long differentiation chains.
- **Physics/Units**: Combine constants, simplify units in formulas (optional, when units are represented in the AST).

This roadmap outlines phases, technical options, and integration into the Dedekind compiler.

---

## 2. Scope: What Should Be Simplified?

| Category | Examples | Priority |
|----------|----------|----------|
| **Constant Folding** | `2*3` → `6`, `0*x` → `0`, `1*y` → `y` | High |
| **Arithmetic** | `(a+b)-b` → `a`, `x*1`, `x+0` | High |
| **Power/Log** | `x^0` → `1`, `x^1` → `x`, `exp(log(x))` → `x` (with care) | Medium |
| **Ricci / Einsum** | Simplify contractions, redundant indices | Medium |
| **Units** | `(kg*m/s^2)` simplify as type (when compile-time units) | Later |
| **Trigonometry** | `sin(0)` → `0`, sum formulas (optional) | Low |

**MVP (Phase 1)** should cover: constant folding, `+0`/`-0`, `*1`/`/1`, `*0` → `0`.

---

## 3. Technical Options

### Option A: Integrate SymPy

- **Advantage**: Powerful simplification, differentiation, LaTeX export possible.
- **Disadvantage**: External dependency, Dedekind AST → SymPy → back: AST ↔ SymPy-Expr translation needed; overhead for small expressions.
- **Use**: Optional module or CLI "simplify-only"; or only in a later phase for heavy cases.

### Option B: Custom Simplifier Module (AST-to-AST)

- **Advantage**: No new dependency, full control, tailored to Dedekind AST (BinaryOp, Literal, Identifier, etc.).
- **Disadvantage**: Only the rules that are explicitly implemented; no "complete" CAS.
- **Use**: Recommended for **Phase 1–2**; clearly defined rewrite rules on the existing AST.

### Option C: Hybrid

- Lightweight simplification (constants, ±0, ·1, ·0) in the **custom simplifier**.
- Heavy or user-driven simplification (e.g. `simplify(expr)`) optionally via **SymPy**, when available.

---

## 4. Implementation Phases

### Phase 1: MVP — Constants and Trivial Arithmetic (estimated: 1–2 weeks)

**Goal**: Insert a **simplify pass** into the compiler pipeline after the parser and before codegen, applying only safe, local rules.

**Steps**:

1. **New module** `src/compiler/simplify.py` (or `symbolic_simplify.py`).
2. **Visitor over AST**: A pass that traverses all expressions (e.g. `BinaryOp`, `Literal`) and replaces according to fixed rules:
   - `Literal(0) + x` → `x` (and analogously `x + 0`)
   - `Literal(1) * x` → `x`, `x * 1` → `x`
   - `Literal(0) * x` → `Literal(0)` (caution: if `x` had side effects, currently not the case in Dedekind)
   - `x - Literal(0)` → `x`
   - `BinaryOp(Literal(a), '+', Literal(b))` → `Literal(a+b)`; analogous for `-`, `*`, `/` (division only if b≠0).
3. **Integration**: In `compiler.py` after `parser.parse()` and before `codegen.generate(ast)` call:  
   `ast = simplify.simplify_program(ast)` (or per statement/expression).
4. **Tests**: Unit tests with small Dedekind snippets; expected generated code without `x+0`, `1*y`, etc.
5. **Documentation**: Brief section in Language Spec and README ("Optional: Symbolic Simplification Pass").

**Success criterion**: Simple expressions like `a + 0`, `b * 1`, `2 * 3` appear simplified in the generated code, without regression in existing examples.

---

### Phase 2: Extended Arithmetic and Power Rules (estimated: 1–2 weeks)

**Goal**: Additional safe rules; no change in semantics (no assumptions about domains as with `log(exp(x))` without care).

**Steps**:

1. **Additional rules** in `simplify.py`:
   - `x^0` → `1` (for numeric literals/constant exponents),
   - `x^1` → `x`,
   - `0^x` → `0` (for x > 0; in Dedekind real-valued, optionally only for literal exponents).
2. **Sub-expressions**: Simplification recursively on all sub-expressions (already in Phase 1 via visitor; more cases here).
3. **Ricci/Einsum**: If present, identify "trivial" contractions (e.g. tensor times scalar 1) and pass to codegen; optional.
4. **Tests**: Extended tests; regression suite (all existing Dedekind examples run unchanged).

**Success criterion**: Expressions with powers and constants are generated more readably and shorter; no incorrect simplifications.

---

### Phase 3: Configuration and Visibility (estimated: approx. 1 week)

**Goal**: Make simplification configurable and transparent for users.

**Steps**:

1. **Compiler flag/option**: e.g. `--simplify` / `--no-simplify` (default: `--simplify` from Phase 1, or initially opt-in).
2. **IDE/Server**: Option in the IDE (e.g. "Symbolic Simplification: On/Off"); passed to backend.
3. **Debug output**: Optionally "before/after" AST or generated expression in verbose mode.
4. **Documentation**: Add a short section "Symbolic Simplification" to the Language Spec (goal, option, examples).

**Success criterion**: Users can enable and disable simplification; behavior is documented.

---

### Phase 4: SymPy Integration (optional, estimated: 2–3 weeks)

**Goal**: Offer strong simplification for complex expressions or explicit user invocation.

**Steps**:

1. **Dependency**: `sympy` as **optional** dependency (e.g. `extras` or separate `requirements-simplify.txt`).
2. **AST → SymPy**: Translator from Dedekind AST (only supported parts: literals, identifiers, +, -, *, /, ^) to `sympy.Expr`.
3. **SymPy → AST**: Back-translation from `sympy.Expr` to Dedekind AST (restricted to allowed constructs).
4. **Usage**: Either as a second pass after the custom simplifier (only when SymPy is available) or via an explicit function/built-in `simplify(expr)` in the language concept (later).
5. **Fallback**: If SymPy is not installed or translation fails: leave the original AST unchanged.

**Success criterion**: With the SymPy option enabled, heavy expressions (e.g. with many brackets and constants) are simplified more aggressively; without SymPy the compiler runs as in Phase 2.

---

### Phase 5: Units and Special Domains (long-term)

**Goal**: When **compile-time units** (dimensions as types) are introduced, extend simplification to include unit arithmetic; possibly simplification of Ricci/Einsum structures.

**Steps**: Depends on the design of compile-time units; separate roadmap. Here only as a placeholder: "Symbolic simplification takes dimension types into account and simplifies unit terms."

---

## 5. Integration into the Compiler

**Current flow** (simplified):

```
Source code → Lexer → Parser → AST → Codegen → Python code
```

**With Symbolic Simplification**:

```
Source code → Lexer → Parser → AST → [ Simplify Pass ] → AST' → Codegen → Python code
```

- **Input**: Complete AST (Program with Statements, containing Expressions).
- **Output**: AST of the same structure, only with simplified expressions (replaced nodes).
- **Location**: In `compiler.py` (or central pipeline), e.g.:

  ```python
  from .parser import Parser
  from .simplify import simplify_program   # Phase 1
  from .codegen import CodeGenerator

  ast = Parser(tokens).parse()
  if options.get("simplify", True):
      ast = simplify_program(ast)
  code = CodeGenerator().generate(ast)
  ```

- **No changes** to lexer, parser, or codegen syntax needed; codegen continues to work on the same AST format.

---

## 6. Risks and Fallbacks

| Risk | Mitigation |
|------|------------|
| Simplification changes semantics (e.g. floats, NaN/Inf) | Only safe rules; when in doubt (e.g. `0*expr`), optionally only for literals or explicitly documented. |
| Compiler performance | Simplifier runs only once per program; rules are local (no full CAS in the MVP). |
| SymPy dependency | Optional; without SymPy the custom simplifier continues to work. |
| Ricci/Einsum incorrectly simplified | Phase 2: only very conservative Ricci rules, or omit until Phase 5. |

---

## 7. Success Metrics (across all phases)

- **Correctness**: All existing Dedekind examples (hello, universal_constants, differentiable_ode, probabilistic, …) produce the same result with simplification enabled as without (or a documented, intended simplification).
- **Readability**: Generated code for typical scientific expressions contains fewer redundant terms (subjective or via metric "number of binary ops").
- **Performance**: Runtime of generated programs no worse; ideally slightly better for expression-heavy programs due to fewer operations.

---

## 8. References and Next Steps

- **Language Specification**: §12 Implementation Roadmap; "Beyond v1.0" — Symbolic Simplification.
- **Codebase**: `src/compiler/ast_nodes.py` (AST definitions), `src/compiler/parser.py` (AST generation), `src/compiler/compiler.py` (pipeline), `src/compiler/codegen.py` (AST → code).
- **Next concrete step**: Phase 1 — Create module `simplify.py`, implement visitor for `BinaryOp`/`Literal`, hook into `compiler.py`, write tests.

---

*This document is the implementation roadmap for Symbolic Simplification. When the compiler changes (e.g. new AST, new options), the roadmap should be updated accordingly.*
