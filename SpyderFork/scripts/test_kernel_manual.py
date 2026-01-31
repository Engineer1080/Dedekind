#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kernel manuell starten, um Fehlermeldungen im Terminal zu sehen.
Ausführung: Aus Dedekind-Repo-Root oder aus SpyderFork:
  python SpyderFork/scripts/test_kernel_manual.py
  oder (von SpyderFork aus): python scripts/test_kernel_manual.py

Erstellt eine temporäre connection-Datei und startet den Spyder-Kernel.
So siehst du sofort Tracebacks (z. B. ModuleNotFoundError), wenn der Kernel stirbt.
"""
import os
import sys
import tempfile
import json

# Repo-Root: von SpyderFork/scripts/ zwei Ebenen hoch = SpyderFork, drei = Repo
_script_dir = os.path.dirname(os.path.abspath(__file__))
_spyder_fork = os.path.dirname(_script_dir)
_repo_root = os.path.dirname(_spyder_fork)

# PYTHONPATH so setzen, dass spyder_kernels aus external-deps gefunden wird
_external = os.path.join(_spyder_fork, 'external-deps', 'spyder-kernels')
if os.path.isdir(_external):
    sys.path.insert(0, _external)
    os.environ['PYTHONPATH'] = _external + os.pathsep + os.environ.get('PYTHONPATH', '')
    print("PYTHONPATH (spyder-kernels):", _external, file=sys.stderr)
else:
    print("Warnung: external-deps/spyder-kernels nicht gefunden:", _external, file=sys.stderr)

# Connection-Datei anlegen (minimal für ipykernel)
try:
    from jupyter_core.paths import jupyter_runtime_dir
    runtime = jupyter_runtime_dir()
except Exception:
    runtime = tempfile.gettempdir()
os.makedirs(runtime, exist_ok=True)
conn_file = os.path.join(runtime, 'test_kernel_manual_connection.json')
conn = {
    'shell_port': 0, 'iopub_port': 0, 'stdin_port': 0, 'control_port': 0, 'hb_port': 0,
    'key': '', 'transport': 'tcp', 'signature_scheme': 'hmac-sha256', 'kernel_name': 'python3'
}
with open(conn_file, 'w') as f:
    json.dump(conn, f)
print("Connection-Datei:", conn_file, file=sys.stderr)
print("Starte Spyder-Kernel (Strg+C zum Beenden)...", file=sys.stderr)

# Kernel starten (wie Spyder: python -m spyder_kernels.console -f connection_file)
sys.argv = [sys.argv[0], '-f', conn_file]
try:
    import runpy
    runpy.run_module('spyder_kernels.console', run_name='__main__', alter_sys=True)
except Exception as e:
    print("Fehler beim Start:", e, file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
