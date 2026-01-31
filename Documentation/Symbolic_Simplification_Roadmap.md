# Symbolic Simplification — Implementierungs-Roadmap

**Dedekind Language**  
Draft: January 2026

---

## 1. Ziel und Nutzen

**Symbolic Simplification** bezeichnet die algebraische Vereinfachung von Ausdrücken **vor** der Code-Generierung (Python/MLIR). Ziele:

- **Lesbarkeit**: Generierter Code enthält kürzere, klare Formeln statt aufgeblähter Zwischenausdrücke.
- **Numerische Stabilität**: Redundante Operationen (z. B. `(x+1)-1` → `x`) vermeiden unnötige Rundungsfehler.
- **Performance**: Weniger Operationen, bessere Chancen für Fused Ops und Compiler-Optimierungen; bei Ricci-Ausdrücken und langen Ableitungsketten spürbar.
- **Physik/Units**: Konstanten zusammenfassen, Einheiten in Formeln vereinfachen (optional, wenn Units im AST repräsentiert sind).

Diese Roadmap skizziert Phasen, technische Optionen und Integration in den Dedekind-Compiler.

---

## 2. Scope: Was soll vereinfacht werden?

| Kategorie | Beispiele | Priorität |
|-----------|-----------|-----------|
| **Konstantenfaltung** | `2*3` → `6`, `0*x` → `0`, `1*y` → `y` | Hoch |
| **Arithmetik** | `(a+b)-b` → `a`, `x*1`, `x+0` | Hoch |
| **Potenz/Log** | `x^0` → `1`, `x^1` → `x`, `exp(log(x))` → `x` (vorsichtig) | Mittel |
| **Ricci / Einsum** | Kontraktionen vereinfachen, redundante Indizes | Mittel |
| **Units** | `(kg*m/s^2)` als Typ vereinfachen (wenn Compile-Time-Units) | Später |
| **Trigonometrie** | `sin(0)` → `0`, Summenformeln (optional) | Niedrig |

**MVP (Phase 1)** sollte: Konstantenfaltung, `+0`/`-0`, `*1`/`/1`, `*0` → `0` abdecken.

---

## 3. Technische Optionen

### Option A: SymPy integrieren

- **Vorteil**: Mächtige Vereinfachung, Differenziation, LaTeX-Export möglich.
- **Nachteil**: Externe Abhängigkeit, Dedekind-AST → SymPy → zurück: Übersetzung AST ↔ SymPy-Expr nötig; Overhead für kleine Ausdrücke.
- **Einsatz**: Optionales Modul oder CLI „simplify-only“; oder nur in einer späteren Phase für schwere Fälle.

### Option B: Eigenes Simplifier-Modul (AST-zu-AST)

- **Vorteil**: Keine neue Abhängigkeit, volle Kontrolle, an Dedekind-AST (BinaryOp, Literal, Identifier, etc.) angepasst.
- **Nachteil**: Nur die Regeln, die man explizit implementiert; kein „vollständiges“ CAS.
- **Einsatz**: Empfohlen für **Phase 1–2**; klar definierte Rewrite-Regeln auf dem bestehenden AST.

### Option C: Hybrid

- Leichte Vereinfachung (Konstanten, ±0, ·1, ·0) im **eigenen Simplifier**.
- Schwere oder nutzergetriebene Vereinfachung (z. B. `simplify(expr)`) optional über **SymPy**, wenn verfügbar.

---

## 4. Implementierungs-Phasen

### Phase 1: MVP — Konstanten und triviale Arithmetik (geschätzt: 1–2 Wochen)

**Ziel**: Im Compiler-Pipeline nach Parser, vor Codegen, einen **Simplify-Pass** einbauen, der nur sichere, lokale Regeln anwendet.

**Schritte**:

1. **Neues Modul** `src/compiler/simplify.py` (oder `symbolic_simplify.py`).
2. **Visitor über AST**: Ein Pass, der alle Ausdrücke (z. B. `BinaryOp`, `Literal`) traversiert und nach festen Regeln ersetzt:
   - `Literal(0) + x` → `x` (und analog `x + 0`)
   - `Literal(1) * x` → `x`, `x * 1` → `x`
   - `Literal(0) * x` → `Literal(0)` (Vorsicht: wenn `x` Side-Effects hätte, in Dedekind aktuell nicht)
   - `x - Literal(0)` → `x`
   - `BinaryOp(Literal(a), '+', Literal(b))` → `Literal(a+b)`; analog für `-`, `*`, `/` (Division nur wenn b≠0).
3. **Integration**: In `compiler.py` nach `parser.parse()` und vor `codegen.generate(ast)` aufrufen:  
   `ast = simplify.simplify_program(ast)` (oder pro Statement/Expression).
4. **Tests**: Unit-Tests mit kleinen Dedekind-Snippets; erwarteter generierter Code ohne `x+0`, `1*y`, etc.
5. **Dokumentation**: Kurz in Language Spec und README („Optional: Symbolic Simplification Pass“).

**Erfolgskriterium**: Einfache Ausdrücke wie `a + 0`, `b * 1`, `2 * 3` erscheinen im generierten Code vereinfacht, ohne Regression bei bestehenden Beispielen.

---

### Phase 2: Erweiterte arithmetische und Potenz-Regeln (geschätzt: 1–2 Wochen)

**Ziel**: Weitere sichere Regeln; keine Änderung der Semantik (keine Annahmen über Definitionsbereiche wie bei `log(exp(x))` ohne Vorsicht).

**Schritte**:

1. **Weitere Regeln** in `simplify.py`:
   - `x^0` → `1` (für numerische Literale/konstante Exponenten),
   - `x^1` → `x`,
   - `0^x` → `0` (für x > 0; bei Dedekind reell, optional nur für Literal-Exponenten).
2. **Subausdrücke**: Simplification rekursiv auf alle Teilausdrücke (bereits in Phase 1 durch Visitor; hier mehr Fälle).
3. **Ricci/Einsum**: Falls vorhanden, Identifikation von „trivialen“ Kontraktionen (z. B. Tensor mal Skalar 1) und Weitergabe an Codegen; optional.
4. **Tests**: Erweiterte Tests; Regression-Suite (alle bestehenden Dedekind-Beispiele laufen unverändert).

**Erfolgskriterium**: Ausdrücke mit Potenzen und Konstanten werden lesbarer und kürzer generiert; keine Fehlvereinfachungen.

---

### Phase 3: Konfiguration und Sichtbarkeit (geschätzt: ca. 1 Woche)

**Ziel**: Simplification steuerbar machen und für Nutzer nachvollziehbar.

**Schritte**:

1. **Compiler-Flag/Option**: z. B. `--simplify` / `--no-simplify` (Default: `--simplify` ab Phase 1, oder zunächst opt-in).
2. **IDE/Server**: Option in Dedekind Studio (z. B. „Symbolic Simplification: Ein/Aus“); Übergabe an Backend.
3. **Debug-Output**: Optional „vorher/nachher“ AST oder generierter Ausdruck bei aktiviertem Verbose-Modus.
4. **Dokumentation**: Language Spec um einen kurzen Abschnitt „Symbolic Simplification“ ergänzen (Ziel, Option, Beispiele).

**Erfolgskriterium**: Nutzer können Simplification ein- und ausschalten; Verhalten ist dokumentiert.

---

### Phase 4: SymPy-Integration (optional, geschätzt: 2–3 Wochen)

**Ziel**: Für komplexe Ausdrücke oder expliziten Nutzeraufruf eine starke Vereinfachung anbieten.

**Schritte**:

1. **Abhängigkeit**: `sympy` als **optionale** Abhängigkeit (z. B. `extras` oder separates `requirements-simplify.txt`).
2. **AST → SymPy**: Übersetzer von Dedekind-AST (nur unterstützte Teile: Literale, Identifier, +, -, *, /, ^) nach `sympy.Expr`.
3. **SymPy → AST**: Rückübersetzung von `sympy.Expr` in Dedekind-AST (eingeschränkt auf erlaubte Konstrukte).
4. **Nutzung**: Entweder als zweiter Pass nach dem eigenen Simplifier (nur wenn SymPy verfügbar) oder über eine explizite Funktion/Built-in `simplify(expr)` im Sprachkonzept (später).
5. **Fallback**: Wenn SymPy nicht installiert oder Übersetzung fehlschlägt: Original-AST unverändert lassen.

**Erfolgskriterium**: Bei aktivierter SymPy-Option werden schwere Ausdrücke (z. B. mit vielen Klammern und Konstanten) stärker vereinfacht; ohne SymPy läuft der Compiler wie in Phase 2.

---

### Phase 5: Einheiten und spezielle Domänen (langfristig)

**Ziel**: Wenn **Compile-Time-Units** (Dimensionen als Typen) eingeführt werden, Simplification um Einheiten-Arithmetik erweitern; ggf. Vereinfachung von Ricci/Einsum-Strukturen.

**Schritte**: Abhängig vom Design von Compile-Time-Units; eigene Roadmap. Hier nur als Platzhalter: „Symbolic Simplification berücksichtigt Dimensions-Typen und vereinfacht Einheiten-Terme.“

---

## 5. Integration in den Compiler

**Aktueller Ablauf** (vereinfacht):

```
Quelltext → Lexer → Parser → AST → Codegen → Python-Code
```

**Mit Symbolic Simplification**:

```
Quelltext → Lexer → Parser → AST → [ Simplify-Pass ] → AST' → Codegen → Python-Code
```

- **Eingabe**: Vollständiger AST (Program mit Statements, darin Expressions).
- **Ausgabe**: AST gleicher Struktur, nur mit vereinfachten Ausdrücken (ersetzte Knoten).
- **Ort**: In `compiler.py` (oder zentraler Pipeline), z. B.:

  ```python
  from .parser import Parser
  from .simplify import simplify_program   # Phase 1
  from .codegen import CodeGenerator

  ast = Parser(tokens).parse()
  if options.get("simplify", True):
      ast = simplify_program(ast)
  code = CodeGenerator().generate(ast)
  ```

- **Keine Änderung** an Lexer, Parser oder Codegen-Syntax nötig; Codegen arbeitet weiter auf dem gleichen AST-Format.

---

## 6. Risiken und Fallbacks

| Risiko | Mitigation |
|--------|------------|
| Vereinfachung ändert Semantik (z. B. Floats, NaN/Inf) | Nur sichere Regeln; bei Zweifel (z. B. `0*expr`) optional nur für Literale oder explizit dokumentieren. |
| Performance des Compilers | Simplifier nur einmal pro Programm; Regeln lokal (kein vollständiges CAS im MVP). |
| SymPy-Abhängigkeit | Optional; ohne SymPy läuft der eigene Simplifier weiter. |
| Ricci/Einsum falsch vereinfacht | Phase 2 nur sehr konservative Ricci-Regeln oder weglassen bis Phase 5. |

---

## 7. Erfolgsmetriken (über alle Phasen)

- **Korrektheit**: Alle bestehenden Dedekind-Beispiele (hello, universal_constants, differentiable_ode, probabilistic, …) liefern mit aktivierter Simplification dasselbe Ergebnis wie ohne (oder dokumentierte, gewollte Vereinfachung).
- **Lesbarkeit**: Generierter Code für typische wissenschaftliche Ausdrücke enthält weniger redundante Terme (subjektiv oder per Metrik „Anzahl Binär-Ops“).
- **Performance**: Laufzeit der generierten Programme nicht schlechter; idealerweise bei ausdrucksstarken Programmen leicht besser durch weniger Operationen.

---

## 8. Referenzen und nächste Schritte

- **Language Specification**: §12 Implementation Roadmap; „Beyond v1.0“ — Symbolic Simplification.
- **Codebasis**: `src/compiler/ast_nodes.py` (AST-Definitionen), `src/compiler/parser.py` (AST-Erzeugung), `src/compiler/compiler.py` (Pipeline), `src/compiler/codegen.py` (AST → Code).
- **Nächster konkreter Schritt**: Phase 1 — Modul `simplify.py` anlegen, Visitor für `BinaryOp`/`Literal` implementieren, in `compiler.py` einhängen, Tests schreiben.

---

*Dieses Dokument ist die Implementierungs-Roadmap für Symbolic Simplification. Bei Änderungen am Compiler (z. B. neues AST, neue Optionen) sollte die Roadmap angepasst werden.*
