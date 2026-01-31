# -*- coding: utf-8 -*-
#
# Copyright © Spyder Project Contributors / Dedekind Studio
# Licensed under the terms of the MIT License
# (see spyder/__init__.py for details)

"""
Kernel spec for the Dedekind language (first-class kernel in Dedekind Studio).
"""

import os
import sys
import os.path as osp

# Local imports
from jupyter_client.kernelspec import KernelSpec
from spyder.api.translations import _

# SpyderFork is inside the Dedekind repo. Repo root = parent of SpyderFork.
HERE = osp.abspath(os.path.dirname(__file__))
# HERE = .../SpyderFork/spyder/plugins/ipythonconsole/utils
# 4 dirnames -> SpyderFork root, 5 -> repo root (Dedekind)
for _ in range(4):
    HERE = osp.dirname(HERE)
_SPYDERFORK_ROOT = HERE
_REPO_ROOT = osp.dirname(_SPYDERFORK_ROOT)
HERE = osp.abspath(os.path.dirname(__file__))  # restore for other uses


def get_dedekind_repo_root():
    """Return the Dedekind repo root (parent of SpyderFork) if detectable."""
    repo_root = osp.normpath(osp.abspath(_REPO_ROOT))
    if osp.isdir(osp.join(repo_root, 'dedekind_jupyter_kernel')) or osp.isdir(osp.join(repo_root, 'src', 'compiler')):
        return repo_root
    return None


class DedekindKernelSpec(KernelSpec):
    """Kernel spec for the Dedekind language."""

    is_spyder_kernel = False  # So KernelHandler uses IpykernelReady path

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.display_name = 'Dedekind'
        self.language = 'dedekind'
        self.resource_dir = ''
        self._env_vars = None

    @property
    def argv(self):
        """Command to start the Dedekind kernel (via wrapper so PYTHONPATH is set)."""
        repo_root = get_dedekind_repo_root()
        if repo_root:
            wrapper = osp.join(repo_root, 'dedekind_jupyter_kernel', 'run_dedekind_kernel.py')
            if osp.isfile(wrapper):
                return [sys.executable, wrapper, '{connection_file}']
        return [
            sys.executable,
            '-m', 'dedekind_jupyter_kernel',
            '-f', '{connection_file}'
        ]

    def _base_env(self):
        """Build base env with PYTHONPATH so the kernel finds the compiler and itself."""
        env = dict(os.environ)
        repo_root = get_dedekind_repo_root()
        if repo_root:
            repo_root = osp.normpath(osp.abspath(repo_root))
            path = env.get('PYTHONPATH', '') or ''
            if path:
                env['PYTHONPATH'] = repo_root + os.pathsep + path
            else:
                env['PYTHONPATH'] = repo_root
        return env

    @property
    def env(self):
        """Environment for the Dedekind kernel."""
        if self._env_vars is not None:
            return self._env_vars
        return self._base_env()

    @env.setter
    def env(self, value):
        """Allow overwriting env (e.g. from get_user_environment_variables)."""
        base = self._base_env()
        if value:
            base.update(value)
        # Always keep repo root on PYTHONPATH so the kernel finds the compiler
        # and dedekind_jupyter_kernel (caller may have removed PYTHONPATH).
        repo_root = get_dedekind_repo_root()
        if repo_root:
            path = base.get('PYTHONPATH', '')
            if path:
                base['PYTHONPATH'] = repo_root + os.pathsep + path
            else:
                base['PYTHONPATH'] = repo_root
        self._env_vars = base
