#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Einstiegspunkt für die mit PyInstaller gebaute Dedekind Studio .exe.
Setzt sys.path und Umgebung, dann startet Spyder.
"""
import os
import sys

def _setup_paths():
    """Wenn gefroren (PyInstaller): Dedekind-Compiler und Kernel im Bundle finden."""
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS  # type: ignore[union-attr]
        if base not in sys.path:
            sys.path.insert(0, base)
        # Dedekind-Repo-Struktur: base enthält src/ und dedekind_jupyter_kernel/
        os.environ["DEDEKIND_FROZEN_BASE"] = base

def _setup_qt():
    """Qt-Backend setzen (PyQt5 wie beim normalen Start)."""
    if "QT_API" not in os.environ:
        os.environ["QT_API"] = "pyqt5"
    # Kein Dev-Modus in der gebündelten App
    os.environ.pop("SPYDER_DEV", None)

def main():
    _setup_paths()
    _setup_qt()
    from spyder.app import start
    start.main()

if __name__ == "__main__":
    main()
