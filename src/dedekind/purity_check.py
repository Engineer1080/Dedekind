"""
Purity-Check fuer Dedekind (v1.8+).

Verhindert syntaktisch, dass I/O- und Side-Effect-Aufrufe (print, plot,
write_file, http_*, ...) innerhalb von Funktionen stehen, die an pure
Kontexte uebergeben werden:

    jit(fn)                  -> torch.compile-Graph
    grad(fn, x)              -> Autograd-Tape
    fit(loss_fn, ...)        -> GD/MCMC/HMC-Loop
    metropolis(log_prior, log_likelihood, ...)
    hmc(log_prior, log_likelihood, ...)
    sde_solve(drift, diffusion, ...)

Der Check ist transitiv: ruft `loss` eine Funktion `helper` auf und `helper`
ruft `print`, wird der gesamte Pfad als unrein gemeldet.

Opt-Out: `compile_source(..., check_purity=False)` oder CLI `--no-purity-check`.
"""

from typing import Optional, Dict, List, Tuple, Set
import dataclasses

from .ast_nodes import (
    Node, Program, FunctionDef, Call, Identifier, CompileError,
)

# I/O-Built-ins mit klaren Seiteneffekten (Konsolen-Output, Datei-/Netzwerk-I/O).
# Benchmark/Profile/time_block sind bewusst NICHT enthalten — sie werden gelegentlich
# fuer punktuelle Messungen in (eigentlich reinen) Funktionen verwendet.
IMPURE_BUILTINS = frozenset({
    "print", "plot", "scatter", "contour",
    "print_latex", "print_table",
    "write_file", "read_file", "file_exists",
    "http_get", "http_post",
    "write_csv", "read_csv",
    "write_parquet", "read_parquet",
    "write_hdf5", "read_hdf5", "read_netcdf",
    "export_notebook",
})

# Aufruf -> Liste der Argumentindizes, die als pure Funktion erwartet werden.
PURE_CONTEXT_CALLS: Dict[str, List[int]] = {
    "jit":         [0],
    "grad":        [0],
    "fit":         [0],
    "metropolis":  [0, 1],
    "hmc":         [0, 1],
    "nuts":        [0, 1],
    "vi":          [0, 1],
    "sde_solve":   [0, 1],
}


def _iter_field_values(node: Node):
    """Yieldet alle Feld-Werte eines AST-Knotens (dataclass-basiert)."""
    if not dataclasses.is_dataclass(node):
        return
    for f in dataclasses.fields(node):
        yield getattr(node, f.name)


def _walk(node):
    """Pre-order Walk durch alle Node-Nachfolger."""
    if not isinstance(node, Node):
        return
    yield node
    for value in _iter_field_values(node):
        if isinstance(value, Node):
            yield from _walk(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, Node):
                    yield from _walk(item)
                elif isinstance(item, tuple):
                    for sub in item:
                        if isinstance(sub, Node):
                            yield from _walk(sub)
        elif isinstance(value, tuple):
            for sub in value:
                if isinstance(sub, Node):
                    yield from _walk(sub)


def _collect_functions(ast: Program) -> Dict[str, FunctionDef]:
    """Sammelt alle FunctionDef-Knoten im Programm (auch verschachtelte)."""
    table: Dict[str, FunctionDef] = {}
    for node in _walk(ast):
        if isinstance(node, FunctionDef):
            table[node.name] = node
    return table


def _find_pure_context_calls(ast: Program):
    """Yieldet (Call, [arg_indices]) fuer jeden pure-context-Aufruf."""
    for node in _walk(ast):
        if isinstance(node, Call) and isinstance(node.func_name, Identifier):
            name = node.func_name.name
            if name in PURE_CONTEXT_CALLS:
                yield node, PURE_CONTEXT_CALLS[name]


def _find_impure_calls(fn_def: FunctionDef, fn_table: Dict[str, FunctionDef],
                       visited: Set[str]) -> List[Tuple[str, Optional[int], str]]:
    """Liefert (impure_name, line, containing_fn) fuer alle unreinen Aufrufe im Body,
    transitiv durch User-Funktionen."""
    if fn_def.name in visited:
        return []
    visited = visited | {fn_def.name}
    findings: List[Tuple[str, Optional[int], str]] = []
    for node in _walk(fn_def):
        if isinstance(node, Call) and isinstance(node.func_name, Identifier):
            called = node.func_name.name
            if called in IMPURE_BUILTINS:
                findings.append((called, getattr(node, "line", None), fn_def.name))
            elif called in fn_table and called != fn_def.name:
                findings.extend(_find_impure_calls(fn_table[called], fn_table, visited))
    return findings


def check_purity(ast: Program, filepath: Optional[str] = None) -> None:
    """Wirft CompileError, sobald eine an einen pure-context uebergebene Funktion
    transitiv eine I/O-/Side-Effect-Funktion aufruft."""
    fn_table = _collect_functions(ast)
    for call_node, arg_indices in _find_pure_context_calls(ast):
        ctx_name = call_node.func_name.name
        for idx in arg_indices:
            if idx >= len(call_node.args):
                continue
            arg = call_node.args[idx]
            if not isinstance(arg, Identifier):
                continue  # Lambda oder Ausdruck — nicht statisch aufloesbar, skip
            fn_name = arg.name
            if fn_name not in fn_table:
                continue
            impure = _find_impure_calls(fn_table[fn_name], fn_table, set())
            if not impure:
                continue
            first_name, first_line, first_fn = impure[0]
            location = f"in Zeile {first_line}" if first_line is not None else ""
            raise CompileError(
                f"Purity-Check: Funktion '{fn_name}' wird an "
                f"'{ctx_name}(...)' uebergeben, ruft aber '{first_name}()' "
                f"(unreine Built-in mit I/O- oder Konsolen-Seiteneffekt) {location} "
                f"in '{first_fn}'. Solche Aufrufe sind in jit/grad/fit/metropolis/"
                f"hmc/sde_solve-Kontexten nicht erlaubt — sie brechen Autograd-Tapes, "
                f"torch.compile-Graphen und sind in MCMC-Loops vermutlich unbeabsichtigt. "
                f"Entferne den Aufruf oder kompiliere mit `--no-purity-check`.",
                line=getattr(call_node, "line", None),
                filepath=filepath,
            )
