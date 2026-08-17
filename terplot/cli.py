"""TerPlot REPL: load a data file, describe a plot in English, get a PNG."""

import argparse
import os
import sys
import textwrap
import webbrowser
from pathlib import Path

from terplot.data import load, profile
from terplot.llm import DEFAULT_MODEL, LLMError, generate
from terplot.runner import run

HISTORY_TURNS = 5

ENV_FILE = ".env"

# ponytail: baked in rather than a pyfiglet dependency — regenerate with
# `pyfiglet -f big TerPlot`.
BANNER = r"""
 _______        _____  _       _
|__   __|      |  __ \| |     | |
   | | ___ _ __| |__) | | ___ | |_
   | |/ _ \ '__|  ___/| |/ _ \| __|
   | |  __/ |  | |    | | (_) | |_
   |_|\___|_|  |_|    |_|\___/ \__|
"""

HELP = """commands:
  /schema    show what the model knows about your data
  /code      show the code behind the last plot
  /help      this
  exit       quit (or Ctrl-D)
anything else is treated as a plot request."""


def colors() -> dict:
    """ANSI codes, or empty strings when colour would be noise (piped output,
    NO_COLOR set, dumb terminal)."""
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return {"c": "", "d": "", "b": "", "g": "", "r": ""}
    if os.name == "nt":
        os.system("")  # ponytail: enables ANSI on legacy Windows consoles
    return {
        "c": "\033[36m",
        "d": "\033[2m",
        "b": "\033[1m",
        "g": "\033[32m",
        "r": "\033[0m",
    }


def banner(name: str, df, model: str, col: dict) -> None:
    c, d, b, r = col["c"], col["d"], col["b"], col["r"]
    print(f"{c}{BANNER}{r}")
    # ponytail: ASCII only — non-UTF-8 consoles mangle box/bullet characters.
    print(f"  {d}natural language -> seaborn  |  {model}{r}")
    print()
    print(f"  {b}{name}{r}  {d}{len(df):,} rows x {len(df.columns)} cols{r}")
    cols = ", ".join(str(x) for x in df.columns)
    print(f"  {d}{textwrap.shorten(cols, width=72, placeholder=' ...')}{r}")
    print()
    print(f"  {d}Describe a plot. /help for commands, Ctrl-D to quit.{r}")
    print()


def load_env(*candidates: Path) -> None:
    """Read KEY=value lines from the first .env found and put them in os.environ.

    Handles comments, blank lines, an `export ` prefix, surrounding quotes, and
    `=` inside values. A variable already set in the real environment wins, so
    the shell can always override the file.

    ponytail: ~15 lines instead of a python-dotenv dependency. If .env handling
    ever needs interpolation or multiline values, swap in python-dotenv.
    """
    for path in candidates:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            key, sep, value = line.partition("=")
            if not sep:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key.strip(), value)
        return


def show(path: Path) -> None:
    """Open a PNG in the OS default image viewer."""
    try:
        # ponytail: os.startfile is Windows-only, webbrowser covers the rest
        if hasattr(os, "startfile"):
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            webbrowser.open(path.as_uri())
    except Exception as e:
        print(f"  (couldn't open viewer: {e})")


def turn(request, schema, history, load_expr, out_path, show_code):
    """One request -> plot. Generates, runs, retries once on failure.

    Returns the working snippet, or None if both attempts failed.
    """
    try:
        snippet = generate(schema, history[-HISTORY_TURNS:], request)
    except LLMError as e:
        print(f"  model error: {e}")
        return None

    if show_code:
        print_code(snippet)

    ok, err = run(snippet, load_expr, str(out_path))
    if ok:
        return snippet

    print("  that failed, retrying once...")
    retry_history = history[-(HISTORY_TURNS - 1):] + [(request, snippet)]
    try:
        snippet = generate(schema, retry_history, request, error=err)
    except LLMError as e:
        print(f"  model error: {e}")
        return None

    if show_code:
        print_code(snippet)

    ok, retry_err = run(snippet, load_expr, str(out_path))
    if ok:
        return snippet

    print("  still failing. the code it tried:")
    print_code(snippet)
    print(f"  error: {retry_err.strip().splitlines()[-1] if retry_err.strip() else 'unknown'}")
    return None


def print_code(snippet: str) -> None:
    for line in snippet.strip().splitlines():
        print(f"    | {line}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="terplot", description="Plot your data by describing it in English."
    )
    ap.add_argument("path", help="a .csv, .xlsx or .xls file")
    ap.add_argument("--outdir", default="plots", help="where PNGs go (default: plots/)")
    ap.add_argument("--no-open", action="store_true", help="don't open the image viewer")
    ap.add_argument("--show-code", action="store_true", help="print generated code every turn")
    args = ap.parse_args(argv)

    # cwd first, then the project root, so it works from any directory.
    load_env(Path(ENV_FILE), Path(__file__).resolve().parent.parent / ENV_FILE)

    try:
        df, load_expr = load(args.path)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    name = Path(args.path).name
    schema = profile(df, name)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    col = colors()
    banner(name, df, os.environ.get("TERPLOT_MODEL") or DEFAULT_MODEL, col)

    history: list[tuple[str, str]] = []
    n = 0

    while True:
        try:
            # ﻿: some shells prepend a BOM when piping commands in.
            request = input(f"{col['c']}>{col['r']} ").replace("﻿", "").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not request:
            continue
        if request in ("exit", "quit"):
            return 0
        if request == "/help":
            print(HELP)
            continue
        if request == "/schema":
            print(schema)
            continue
        if request == "/code":
            if history:
                print_code(history[-1][1])
            else:
                print("  nothing plotted yet")
            continue

        n += 1
        out_path = outdir / f"plot_{n:03d}.png"
        snippet = turn(request, schema, history, load_expr, out_path, args.show_code)

        if snippet is None:
            n -= 1  # don't burn a plot number on a failure
            continue

        history.append((request, snippet))
        print(f"  {col['g']}->{col['r']} {out_path}")
        if not args.no_open:
            show(out_path)


if __name__ == "__main__":
    sys.exit(main())
