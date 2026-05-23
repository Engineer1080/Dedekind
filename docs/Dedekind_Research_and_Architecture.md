# Dedekind Language — Research Foundation & Architecture Principles

**Key Papers for Implementation v0.1**  
Mario Michael Heinrich · github.com/Engineer1080  
January 2026 · Updated with language features v0.6

---

## 1. Overview

This document collects the key scientific papers and research results that serve as the foundation for the architecture of the Dedekind programming language. Dedekind is designed to enable GPU/TPU-accelerated, parallelized computations for machine learning and scientific computing.

Core areas: MLIR, Triton, work-stealing scheduler, automatic differentiation, linear and affine type systems.

## 2. MLIR: Compiler Infrastructure

MLIR (Multi-Level Intermediate Representation) is the recommended compiler infrastructure for Dedekind. Key papers: *The MLIR Transform Dialect* (Lücke et al., CGO 2025), *MLIR: Scaling Compiler Infrastructure* (Google Research, 2020). Architectural principles: multi-level IR, dialect system, progressive lowering, hardware abstraction.

## 3. Triton: GPU Kernel Programming

Triton as a Python-like language for GPU programming. Key papers: *Triton: An Intermediate Language and Compiler* (Tillet et al., MAPL 2019), *OpenAI Triton on NVIDIA Blackwell* (NVIDIA, 2025). Principles: block-level parallelism, automatic optimizations, Python-like syntax, hardware-agnostic IR.

## 4. Work-Stealing Scheduler

Work-stealing for dynamic load balancing. Key papers: *Scheduling Multithreaded Computations by Work Stealing* (Blumofe & Leiserson, JACM 1999), *Proactive Work Stealing for Futures* (PPoPP 2019), *Work Assisting* (de Wolff & Keller, ARRAY 2024). Principles: provably good bounds, adaptive thread count, nested parallelism, low-overhead deques.

## 5. Automatic Differentiation

AD as a first-class feature. Key papers: *MimIrADe* (Ullrich, Hack & Leißa, CC 2025), *A Tensor Algebra Compiler for Sparse Differentiation* (CGO 2024). Principles: source-to-source AD, mixed mode, sparse tensor support, integration with type system.

## 6. Linear and Affine Types

Memory safety without GC. Key papers: *Affect: An Affine Type and Effect System* (van Rooij & Krebbers, POPL 2025), *Functional Ownership through Fractional Uniqueness* (Marshall & Orchard, OOPSLA 2024), *Linear Haskell* (Bernardy et al., POPL 2018). Principles: affine types, ownership & borrowing, `.address` read-only, zero-cost abstractions.

## 7. Implementation Roadmap

Components and associated papers (compiler backend LLVM+MLIR, GPU code gen Triton, CPU/GPU scheduler work-stealing, auto-diff, type system, memory management). Phase 1 priorities: MLIR integration, basic type system, simple scheduler, GPU backend.

## 8. Next Steps

Paper deep dive, MLIR/Triton tutorials, proof of concept, LLVM basics, community contact. Useful links: MLIR docs, Triton GitHub, LLVM Getting Started, Rust Book, Linear Haskell Proposal.

## 9. Conclusion

Dedekind's architecture stands on a solid scientific foundation. The combination of MLIR, Triton-like GPU programming, work-stealing, integrated auto-diff and affine types is ambitious but feasible.

---

## 10. Language Features v0.6 (Update)

The current prototype implementation (v0.6) extends the language with **physical units** and **universal constants** that are relevant for scientific computing and physics simulations:

- **Physical Units (Option B)**: Unit literals (`10[m]`, `5[m/s]`, `1.0[kg]`), arithmetic rules (same unit for +/-; multiplication/division combine units; power `^` for e.g. `r^2`). Runtime checks prevent unit errors.
- **Universal Constants as Quantity**: The constants `c`, `G`, `h`, `k_B`, `k_e` are defined as **Quantity** with correct SI units (e.g. `c` in m/s). Expressions like `E = m * c^2` and `F = G * m1 * m2 / r^2` yield dimensionally consistent results; the display is simplified to J (Joule) and N (Newton).
- **Quaternion & Unary Minus**: Complex numbers as Quaternion (component `i`); unary minus (`-z`, `-1.0 + 0i`) is supported for Quantity and Quaternion, so signal processing (e.g. FFT with negative imaginary parts) works correctly.

These features are described in detail in the **Language Specification v0.2** (§15 Physical Units and Universal Constants). They complement the compiler- and runtime-side research (MLIR, Triton, scheduler) with linguistic abstractions for numerical and physical applications.

---

*A PDF can be generated from this Markdown source.*
