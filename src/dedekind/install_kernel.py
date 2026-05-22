"""Register the Dedekind Jupyter kernel into the active Python environment.

After `pip install dedekind`, run::

    python -m dedekind.install_kernel

which copies the kernelspec next to ipykernel's so that `jupyter notebook`
and JupyterLab show "Dedekind" under "New". Idempotent; safe to re-run.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile


KERNEL_NAME = "dedekind"
DISPLAY_NAME = "Dedekind"


def _kernelspec_dict() -> dict:
    return {
        "argv": [sys.executable, "-m", "dedekind_jupyter_kernel", "-f", "{connection_file}"],
        "display_name": DISPLAY_NAME,
        "language": "dedekind",
    }


def install(user: bool = True, prefix: str | None = None) -> str:
    """Install the kernelspec; returns the install directory."""
    try:
        from jupyter_client.kernelspec import KernelSpecManager
    except ImportError as exc:
        raise SystemExit(
            "jupyter_client is not installed. Install it (it ships with ipykernel) and retry."
        ) from exc

    with tempfile.TemporaryDirectory() as td:
        spec_dir = os.path.join(td, KERNEL_NAME)
        os.makedirs(spec_dir, exist_ok=True)
        with open(os.path.join(spec_dir, "kernel.json"), "w", encoding="utf-8") as f:
            json.dump(_kernelspec_dict(), f, indent=2)

        # Copy bundled assets (logos) if present in the package.
        pkg_specdir = os.path.join(os.path.dirname(__file__), "kernelspec")
        if os.path.isdir(pkg_specdir):
            for fname in os.listdir(pkg_specdir):
                if fname == "kernel.json":
                    continue
                shutil.copy2(os.path.join(pkg_specdir, fname), spec_dir)

        ksm = KernelSpecManager()
        dest = ksm.install_kernel_spec(
            spec_dir, kernel_name=KERNEL_NAME, user=user, prefix=prefix, replace=True
        )
    return dest


def main():
    p = argparse.ArgumentParser(description="Install the Dedekind Jupyter kernel.")
    p.add_argument("--system", action="store_true",
                   help="Install for all users (requires sufficient privileges).")
    p.add_argument("--prefix", default=None,
                   help="Install into a specific prefix (e.g. a virtualenv root).")
    args = p.parse_args()

    dest = install(user=not args.system and args.prefix is None, prefix=args.prefix)
    print(f"Dedekind kernel installed: {dest}")
    print('Start JupyterLab/Notebook and pick "Dedekind" from the kernel list.')


if __name__ == "__main__":
    main()
