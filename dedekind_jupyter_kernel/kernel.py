"""
Dedekind Jupyter Kernel: executes Dedekind source in a persistent context.
Requires the Dedekind compiler (src.compiler) on PYTHONPATH or repo root.
"""
import sys
import io
import os

# Ensure Dedekind compiler is importable: repo root from this file or from PYTHONPATH
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
# Fallback: if started with PYTHONPATH (e.g. by Dedekind Studio), prepend so src.compiler is found
_pypath = os.environ.get("PYTHONPATH", "")
if _pypath:
    for _p in _pypath.split(os.pathsep):
        _p = os.path.abspath(os.path.normpath(_p.strip()))
        if _p and _p not in sys.path and os.path.isdir(os.path.join(_p, "src", "compiler")):
            sys.path.insert(0, _p)
            break

from ipykernel.kernelbase import Kernel

try:
    from src.compiler.ast_nodes import CompileError
except ImportError:
    CompileError = Exception  # fallback if run outside repo


class DedekindKernel(Kernel):
    implementation = "Dedekind"
    implementation_version = "1.0.0"
    language = "dedekind"
    language_version = "1.0.0"
    language_info = {
        "name": "dedekind",
        "mimetype": "text/x-dedekind",
        "file_extension": ".ddk",
    }
    banner = "Dedekind Kernel – compile and run Dedekind code (Dedekind Language v1.0.0)"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._globals = {}
        self._execution_count = 0

    def do_comm_open(self, stream, ident, msg):
        """No-op so frontend comm_open messages do not trigger 'Unknown message type'."""
        pass

    def do_execute(self, code, silent, store_history=True, user_expressions=None, allow_stdin=False):
        if not code.strip():
            return {"status": "ok", "payload": [], "user_expressions": {}, "execution_count": self._execution_count}

        if store_history:
            self._execution_count += 1
        reply = {"execution_count": self._execution_count}

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_capture, stderr_capture

        try:
            from src.compiler.compiler import compile_source
        except ImportError as e:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            msg = (
                "Dedekind-Kernel: Compiler nicht gefunden. Bitte PYTHONPATH auf das "
                "Dedekind-Repo-Root setzen (z. B. wo src/compiler und dedekind_jupyter_kernel liegen).\n"
                "ImportError: %s"
            ) % (e,)
            if not silent:
                self.send_response(
                    self.iopub_socket,
                    "stream",
                    {"name": "stderr", "text": msg + "\n"},
                )
            reply.update(status="error", ename="ImportError", evalue=str(e), traceback=[msg])
            return reply
        try:
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
            reply.update(status="error", ename="CompileError", evalue=err_msg, traceback=[err_msg])
            return reply
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
            reply.update(status="error", ename=type(e).__name__, evalue=str(e), traceback=tb.splitlines())
            return reply

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
        reply.update(status="ok", payload=[], user_expressions={})
        return reply
