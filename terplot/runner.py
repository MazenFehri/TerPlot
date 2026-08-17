"""Execute LLM-generated plotting snippets in an isolated subprocess.

Generated statements are wrapped in a fixed PREAMBLE/POSTAMBLE, written to a temp
file, and run with `sys.executable` — never `exec`'d in-process, so a stray
infinite loop, crash, or segfault can't take down the REPL. A timeout bounds
runaway scripts.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

# `load_expr` and `out_path` are interpolated with plain str.format() below —
# both are trusted, wrapper-controlled strings (load_expr comes from data.py,
# out_path from cli.py), not model output. The model's `snippet` is inserted
# as opaque plotting statements between the two comment markers.
PREAMBLE = '''import pandas as pd, seaborn as sns, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
df = {load_expr}
sns.set_theme()
plt.figure(figsize=(10, 6))
# --- model snippet ---
'''

POSTAMBLE = '''
# --- end ---
_terplot_fig = plt.gcf()
if any(_terplot_ax.has_data() for _terplot_ax in _terplot_fig.axes):
    plt.savefig({out_path_literal}, dpi=120, bbox_inches="tight")
'''

_STDERR_TAIL_CHARS = 2000


def run(snippet: str, load_expr: str, out_path: str, timeout: int = 30) -> tuple[bool, str]:
    """Assemble PREAMBLE + snippet + POSTAMBLE, write to a temp .py, execute with
    sys.executable in a subprocess. Returns (ok, stderr).
    ok is False on nonzero exit, timeout, or if out_path was not created.
    stderr is the captured stderr (or a timeout message), trimmed to last ~2000 chars.
    """
    # repr() so Windows backslashes and quotes round-trip through the script.
    out_path_literal = repr(out_path)

    script = (
        PREAMBLE.format(load_expr=load_expr)
        + snippet
        + POSTAMBLE.format(out_path_literal=out_path_literal)
    )

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(script)

        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return (
                False,
                f"timed out after {timeout}s — the plot took too long, "
                "try a simpler plot or aggregate the data first",
            )

        produced = os.path.isfile(out_path) and os.path.getsize(out_path) > 0

        if proc.returncode != 0:
            return False, _trim(proc.stderr)

        if not produced:
            return False, (
                "script exited cleanly but produced no plot — the snippet "
                "must create a plot on the current figure"
            )

        return True, _trim(proc.stderr)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _trim(text: str) -> str:
    """Keep only the last ~2000 chars — the traceback's tail is the useful part."""
    if text is None:
        return ""
    return text[-_STDERR_TAIL_CHARS:]
