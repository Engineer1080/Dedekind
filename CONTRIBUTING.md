# Contributing to Dedekind

Thank you for considering a contribution. Dedekind is a young project
and most workflows are intentionally lightweight.

## Reporting Issues

Open a GitHub issue at
<https://github.com/Engineer1080/Dedekind/issues>. For security
vulnerabilities, follow `SECURITY.md` instead.

A good bug report includes:

1. The Dedekind version (`dedekind --version`) and Python version.
2. A minimal `.ddk` snippet that reproduces the problem.
3. The full traceback, including the mapped `.ddk` line numbers.

## Development Setup

```bash
git clone https://github.com/Engineer1080/Dedekind
cd Dedekind
python -m venv .venv && source .venv/bin/activate
pip install -e ".[sci,plot,jupyter]"
python src/dedekind/build_runtime.py   # regenerate ml_runtime.py
python run_tests.py                    # full test suite
python run_examples.py --compile -q    # compile every example
```

## Pull Requests

- Keep PRs focused. Small, single-purpose patches are easiest to
  review.
- Run `python run_tests.py` and `python run_examples.py --compile -q`
  locally before opening the PR; CI runs the same checks on Python
  3.10, 3.11, 3.12, and 3.13.
- If you add or rename a runtime built-in in `src/dedekind/runtime_modules/`,
  remember to regenerate `ml_runtime.py` via
  `python src/dedekind/build_runtime.py`.
- New language features should come with a `.ddk` test under
  `tests/dedekind/` and, where useful, an example under
  `examples/dedekind/`.

## Coding Style

- Python: standard PEP 8, no enforced formatter. Type hints are
  encouraged but not required outside of public APIs.
- Dedekind (`.ddk`): match the style of existing standard-library
  modules in `src/dedekind/stdlib/`.

## License

By contributing you agree that your contribution will be licensed
under the project's Apache License 2.0.
