# Security Policy

## Supported Versions

Dedekind is currently distributed as a single supported line: the most
recent `3.x` release on PyPI. Older `2.x` releases are not maintained.

| Version | Supported |
|---------|-----------|
| 3.x     | Yes       |
| <= 2.x  | No        |

## Reporting a Vulnerability

If you believe you have found a security vulnerability in Dedekind,
please report it privately so we can address it before public
disclosure.

**Preferred channel:** email `heinrich.mario.m@gmail.com` with a
description of the issue and, if possible, a minimal reproduction.

We will acknowledge receipt within 7 days and aim to issue a fix or
mitigation within 30 days for high-severity issues. Coordinated
disclosure is appreciated.

Please do **not** open a public GitHub issue for security
vulnerabilities.

## Scope

In scope:

- The Dedekind compiler, runtime, and CLI shipped from this
  repository.
- The `dedekind` package as published on PyPI.

Out of scope:

- Vulnerabilities in third-party dependencies (PyTorch, NumPy, SciPy,
  etc.) — please report those upstream.
- Issues in user-authored `.ddk` programs (the language is
  Turing-complete; misuse by a program author is not a Dedekind
  vulnerability).
