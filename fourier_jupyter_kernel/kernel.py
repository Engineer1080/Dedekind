"""
Fourier Jupyter Kernel: executes Fourier source in a persistent context.
Requires the Fourier compiler (src.compiler) on PYTHONPATH or repo root.
"""
import sys
import io
import os

# Ensure Fourier compiler is importable (repo root or installed)
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from ipykernel.kernelbase import Kernel

try:
    from src.compiler.ast_nodes import CompileError
except ImportError:
    CompileError = Exception  # fallback if run outside repo


class FourierKernel(Kernel):
    implementation = "Fourier"
    implementation_version = "0.9.8"
    language = "fourier"
    language_version = "0.9.8"
    language_info = {
        "name": "fourier",
        "mimetype": "text/x-fourier",
        "file_extension": ".fourier",
    }
    banner = "Fourier Kernel – compile and run Fourier code (Fourier Language v0.9.8)"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._globals = {}

    def do_execute(self, code, silent, store_history=True, user_expressions=None, allow_stdin=False):
        if not code.strip():
            return {"status": "ok", "payload": [], "user_expressions": {}}

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_capture, stderr_capture

        try:
            from src.compiler.compiler import compile_source
            python_code = compile_source(code, filepath=None, check_units=True)
            exec(python_code, self._globals)
        except CompileError as e:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            err_msg = str(e)
            if not silent:
                self.send_response(
                    self.iopub_socket,
                    "stream",
                    {"name": "stderr", "text": err_msg + "\n"},
                )
            return {
                "status": "error",
                "ename": "CompileError",
                "evalue": err_msg,
                "traceback": [err_msg],
            }
        except Exception as e:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            import traceback
            tb = traceback.format_exc()
            if not silent:
                self.send_response(
                    self.iopub_socket,
                    "stream",
                    {"name": "stderr", "text": tb + "\n"},
                )
            return {
                "status": "error",
                "ename": type(e).__name__,
                "evalue": str(e),
                "traceback": tb.splitlines(),
            }

        sys.stdout, sys.stderr = old_stdout, old_stderr
        out = stdout_capture.getvalue()
        err = stderr_capture.getvalue()
        if not silent:
            if out:
                self.send_response(
                    self.iopub_socket,
                    "stream",
                    {"name": "stdout", "text": out},
                )
            if err:
                self.send_response(
                    self.iopub_socket,
                    "stream",
                    {"name": "stderr", "text": err},
                )
        return {"status": "ok", "payload": [], "user_expressions": {}}
