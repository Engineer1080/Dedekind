# -*- mode: python ; coding: utf-8 -*-
# PyInstaller-Spec für Dedekind Studio (Windows .exe)
# Build: Siehe Documentation/Build_Dedekind_Studio_Exe.md

import os

SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))
REPO_ROOT = os.path.normpath(os.path.join(SPEC_DIR, '..'))

# Dedekind-Compiler und Kernel als Daten ins Bundle (unter _MEIPASS)
dedekind_datas = [
    (os.path.join(REPO_ROOT, 'src'), 'src'),
    (os.path.join(REPO_ROOT, 'dedekind_jupyter_kernel'), 'dedekind_jupyter_kernel'),
]

a = Analysis(
    [os.path.join(SPEC_DIR, 'launch_frozen.py')],
    pathex=[REPO_ROOT, SPEC_DIR],
    datas=dedekind_datas,
    hiddenimports=[
        'spyder',
        'spyder.app',
        'spyder.app.start',
        'spyder.app.mainwindow',
        'spyder.config',
        'spyder.config.base',
        'spyder.config.manager',
        'spyder.utils',
        'spyder.plugins',
        'spyder.plugins.editor',
        'spyder.plugins.ipythonconsole',
        'spyder.plugins.mainmenu',
        'spyder_kernels',
        'qtconsole',
        'ipykernel',
        'jupyter_core',
        'zmq',
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'torch',
        'matplotlib',
        'scipy',
        'sympy',
        'src.compiler.compiler',
        'src.compiler.ml_runtime',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Dedekind Studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Kein Konsolenfenster (GUI-App)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Optional: Pfad zu .ico für Fenster-Icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Dedekind Studio',
)
