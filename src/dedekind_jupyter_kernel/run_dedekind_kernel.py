#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wrapper to start the Dedekind kernel with the repo root on sys.path and PYTHONPATH.
Called as: python run_dedekind_kernel.py <connection_file>
Use this when the kernel is started by Dedekind Studio/Spyder so the compiler is found
even if PYTHONPATH is not passed through.
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_here)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_path = os.environ.get('PYTHONPATH', '')
os.environ['PYTHONPATH'] = _repo_root + os.pathsep + _path if _path else _repo_root

if len(sys.argv) >= 2:
    conn_file = sys.argv[1]
    sys.argv = [sys.argv[0], '-f', conn_file]

import runpy
runpy.run_module('dedekind_jupyter_kernel', run_name='__main__', alter_sys=True)
