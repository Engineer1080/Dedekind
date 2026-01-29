# Fourier Language — Research Foundation & Architecture Principles

**Key Papers for Implementation v0.1**  
Mario Michael Heinrich · github.com/Engineer1080  
Januar 2026 · Aktualisiert mit Sprachfeatures v0.6

---

## 1. Überblick

Dieses Dokument sammelt die wichtigsten wissenschaftlichen Papers und Forschungsergebnisse, die als Grundlage für die Architektur der Fourier-Programmiersprache dienen. Fourier ist darauf ausgelegt, GPU/TPU-beschleunigte, parallelisierte Berechnungen für Machine Learning und Grafikrendering zu ermöglichen.

Kernbereiche: MLIR, Triton, Work-Stealing Scheduler, Automatische Differenzierung, Lineare und Affine Typsysteme.

## 2. MLIR: Compiler-Infrastruktur

MLIR (Multi-Level Intermediate Representation) ist die empfohlene Compiler-Infrastruktur für Fourier. Kernpaper: *The MLIR Transform Dialect* (Lücke et al., CGO 2025), *MLIR: Scaling Compiler Infrastructure* (Google Research, 2020). Architektur-Prinzipien: Multi-Level IR, Dialect-System, Progressive Lowering, Hardware-Abstraction.

## 3. Triton: GPU Kernel Programming

Triton als Python-ähnliche Sprache für GPU-Programmierung. Kernpaper: *Triton: An Intermediate Language and Compiler* (Tillet et al., MAPL 2019), *OpenAI Triton on NVIDIA Blackwell* (NVIDIA, 2025). Prinzipien: Block-Level Parallelism, automatische Optimierungen, Python-ähnliche Syntax, Hardware-Agnostic IR.

## 4. Work-Stealing Scheduler

Work-Stealing für dynamisches Load-Balancing. Kernpaper: *Scheduling Multithreaded Computations by Work Stealing* (Blumofe & Leiserson, JACM 1999), *Proactive Work Stealing for Futures* (PPoPP 2019), *Work Assisting* (de Wolff & Keller, ARRAY 2024). Prinzipien: Provably Good Bounds, Adaptive Thread Count, Nested Parallelism, Low Overhead Deques.

## 5. Automatische Differenzierung

AD als First-Class Feature. Kernpaper: *MimIrADe* (Ullrich, Hack & Leißa, CC 2025), *A Tensor Algebra Compiler for Sparse Differentiation* (CGO 2024). Prinzipien: Source-to-Source AD, Mixed Mode, Sparse Tensor Support, Integration mit Type System.

## 6. Lineare und Affine Typen

Speichersicherheit ohne GC. Kernpaper: *Affect: An Affine Type and Effect System* (van Rooij & Krebbers, POPL 2025), *Functional Ownership through Fractional Uniqueness* (Marshall & Orchard, OOPSLA 2024), *Linear Haskell* (Bernardy et al., POPL 2018). Prinzipien: Affine Types, Ownership & Borrowing, `.address` read-only, Zero-Cost Abstractions.

## 7. Implementierungs-Roadmap

Komponenten und zugehörige Papers (Compiler Backend LLVM+MLIR, GPU Code Gen Triton, CPU/GPU Scheduler Work-Stealing, Auto-Diff, Type System, Memory Mgmt). Phase-1-Prioritäten: MLIR Integration, Basic Type System, Simple Scheduler, GPU Backend.

## 8. Nächste Schritte

Paper Deep-Dive, MLIR/Triton Tutorials, Proof-of-Concept, LLVM Basics, Community Contact. Hilfreiche Links: MLIR Docs, Triton GitHub, LLVM Getting Started, Rust Book, Linear Haskell Proposal.

## 9. Fazit

Fourier's Architektur steht auf solider wissenschaftlicher Grundlage. Die Kombination aus MLIR, Triton-ähnlichem GPU-Programming, Work-Stealing, integrierter Auto-Diff und Affine Types ist ambitioniert aber machbar.

---

## 10. Sprachfeatures v0.6 (Aktualisierung)

Die aktuelle Prototype-Implementation (v0.6) erweitert die Sprache um **physikalische Einheiten** und **universelle Konstanten**, die für wissenschaftliches Rechnen und Physik-Simulationen relevant sind:

- **Physical Units (Option B)**: Einheiten-Literale (`10[m]`, `5[m/s]`, `1.0[kg]`), Rechenregeln (gleiche Einheit für +/-; Multiplikation/Division kombinieren Einheiten; Potenz `^` für z.B. `r^2`). Laufzeitprüfungen verhindern Einheiten-Fehler.
- **Universal Constants as Quantity**: Die Konstanten `c`, `G`, `h`, `k_B`, `k_e` sind als **Quantity** mit korrekten SI-Einheiten definiert (z.B. `c` in m/s). Ausdrücke wie `E = m * c^2` und `F = G * m1 * m2 / r^2` liefern dimensionell konsistente Ergebnisse; die Anzeige wird auf J (Joule) und N (Newton) vereinfacht.
- **Quaternion & Unary Minus**: Komplexe Zahlen als Quaternion (Komponente `i`); unäres Minus (`-z`, `-1.0 + 0i`) ist für Quantity und Quaternion unterstützt, sodass Signalverarbeitung (z.B. FFT mit negativen Imaginärteilen) korrekt funktioniert.

Diese Features sind in der **Language Specification v0.2** (§15 Physical Units and Universal Constants) detailliert beschrieben. Sie ergänzen die compiler- und runtime-seitige Forschung (MLIR, Triton, Scheduler) um sprachliche Abstraktionen für numerische und physikalische Anwendungen.

---

*PDF kann aus dieser Markdown-Quelle erzeugt werden, siehe Documentation/README.md.*
