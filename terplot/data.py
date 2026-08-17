"""Load tabular data and produce the plain-text schema profile that is the
model's ONLY view of the DataFrame.

`load()` also hands back a reload expression, interpolated verbatim into the
generated script (see runner.py's PREAMBLE) as `df = {load_expr}`, so it must be
valid Python for any path. `profile()` has to stay readable for the LLM without
crashing on the awkward inputs real data throws at it (empty frames, all-null
columns, very wide frames, embedded "|" or newlines, long binary-float tails).
"""

from __future__ import annotations

import os

import pandas as pd

_MAX_COLS_SHOWN = 40
_MAX_SAMPLE_COLS = 15
_UNIQ_THRESHOLD = 20
_LEVEL_LIST_MAX_CHARS = 60
_EXAMPLE_VALUE_MAX_CHARS = 40
_MAX_SAMPLE_ROWS = 3
_FLOAT_ROUND_NDIGITS = 6
_TOTAL_BUDGET_CHARS = 2000


def load(path: str) -> tuple[pd.DataFrame, str]:
    """Read a .csv/.xlsx/.xls into a DataFrame.

    Returns (df, load_expr) where load_expr is a literal Python expression string
    that the generated script uses to reload the same data, e.g.
        'pd.read_csv(r"C:\\data\\sales.csv")'
        'pd.read_excel(r"C:\\data\\sales.xlsx", sheet_name=0)'
    Raises FileNotFoundError if missing, ValueError on unsupported extension.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"file not found: {path}")

    abs_path = os.path.abspath(path)
    ext = os.path.splitext(abs_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(abs_path)
        # repr() escapes backslashes/quotes correctly no matter the platform.
        load_expr = f"pd.read_csv({abs_path!r})"
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(abs_path, sheet_name=0)
        load_expr = f"pd.read_excel({abs_path!r}, sheet_name=0)"
    else:
        raise ValueError(
            f"unsupported file extension {ext!r} — expected .csv, .xlsx, or .xls"
        )

    return df, load_expr


def profile(df: pd.DataFrame, name: str) -> str:
    """Plain-text schema block sent to the model. Exact format below.

    sales.csv — 4,301 rows × 3 cols

    date     object   0 null    e.g. "2024-01-15"
    revenue  float64  12 null   min 0.0  max 98421.3
    region   object   0 null    4 uniq: EU, US, APAC, LATAM

    sample rows:
      2024-01-15 | 1203.5 | EU
      2024-01-16 | 980.2  | US
      2024-01-17 | 1544.0 | APAC
    """
    n_rows, n_cols = df.shape
    header = f"{name} — {n_rows:,} rows × {n_cols:,} cols"

    columns = list(df.columns)
    shown_columns = columns[:_MAX_COLS_SHOWN]
    col_rows = [_describe_column(df, col) for col in shown_columns]

    # Sample rows are capped to a smaller column subset than the detail lines —
    # a pipe-separated row across 40+ columns would blow the char budget on its
    # own long before the detail section does.
    sample_columns = shown_columns[:_MAX_SAMPLE_COLS]
    sample_lines = _sample_row_lines(df, sample_columns)
    if len(sample_columns) < len(shown_columns):
        sample_header = (
            f"sample rows (first {len(sample_columns)} of "
            f"{len(shown_columns)} columns):"
        )
    else:
        sample_header = "sample rows:"

    def render(n_shown: int) -> str:
        lines = [header, ""]
        rows = col_rows[:n_shown]
        if rows:
            name_w = max(len(r[0]) for r in rows) + 2
            dtype_w = max(len(r[1]) for r in rows) + 2
            null_w = max(len(r[2]) for r in rows) + 3
            for cname, dtype, null_str, hint in rows:
                lines.append(
                    f"{cname.ljust(name_w)}{dtype.ljust(dtype_w)}{null_str.ljust(null_w)}{hint}"
                )
        omitted = n_cols - n_shown
        if omitted > 0:
            noun = "column" if omitted == 1 else "columns"
            lines.append(f"... and {omitted} more {noun} not shown")
        lines.append("")
        lines.append(sample_header)
        lines.extend(sample_lines)
        return "\n".join(lines)

    # Normally all of shown_columns (up to 40) render fine. As a hard guarantee
    # against pathological data (e.g. many near-20-unique text columns, each
    # maxing out the 60-char level-list allowance), shrink the detail section
    # until the whole block fits the budget.
    n_shown = len(shown_columns)
    text = render(n_shown)
    while len(text) > _TOTAL_BUDGET_CHARS and n_shown > 1:
        n_shown -= 1
        text = render(n_shown)

    return text


def _describe_column(df: pd.DataFrame, col) -> tuple[str, str, str, str]:
    s = df[col]
    dtype = s.dtype
    null_count = int(s.isna().sum())
    null_str = f"{null_count} null"
    cname = str(col)

    if pd.api.types.is_datetime64_any_dtype(dtype):
        hint = _datetime_hint(s)
    elif (
        isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
    ):
        hint = _object_hint(s)
    elif pd.api.types.is_numeric_dtype(dtype):
        hint = _numeric_hint(s)
    else:
        hint = _object_hint(s)

    return cname, str(dtype), null_str, hint


def _empty_reason(s: pd.Series) -> str:
    return "no data" if len(s) == 0 else "all null"


def _numeric_hint(s: pd.Series) -> str:
    valid = s.dropna()
    if valid.empty:
        return _empty_reason(s)
    return f"min {_fmt_number(valid.min())}  max {_fmt_number(valid.max())}"


def _datetime_hint(s: pd.Series) -> str:
    valid = s.dropna()
    if valid.empty:
        return _empty_reason(s)
    return f"min {_fmt_datetime(valid.min())}  max {_fmt_datetime(valid.max())}"


def _object_hint(s: pd.Series) -> str:
    valid = s.dropna()
    if valid.empty:
        return _empty_reason(s)

    nuniq = valid.nunique()
    if nuniq <= _UNIQ_THRESHOLD:
        levels = [_sanitize(str(v)) for v in pd.unique(valid)]
        joined = ", ".join(levels)
        if len(joined) > _LEVEL_LIST_MAX_CHARS:
            joined = joined[:_LEVEL_LIST_MAX_CHARS].rstrip() + "…"
        return f"{nuniq} uniq: {joined}"

    first = _truncate(_sanitize(str(valid.iloc[0])), _EXAMPLE_VALUE_MAX_CHARS)
    return f'e.g. "{first}"'


def _fmt_number(v) -> str:
    """Round floats so binary-float noise (0.1 + 0.2 -> 15 digits) never leaks
    into the profile, while keeping a plain int/float looking natural."""
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float):
        v = round(v, _FLOAT_ROUND_NDIGITS)
    return str(v)


def _fmt_datetime(ts) -> str:
    try:
        if ts == ts.normalize():
            return ts.strftime("%Y-%m-%d")
    except AttributeError:
        pass
    return str(ts)


def _sanitize(text: str) -> str:
    """Keep a single value on a single line and out of the pipe-delimited
    sample-row grid: collapse newlines to spaces, escape literal pipes."""
    return text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").replace("|", "/")


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def _fmt_sample_value(v) -> str:
    if pd.isna(v):
        return "NaN"
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float):
        v = round(v, _FLOAT_ROUND_NDIGITS)
    if hasattr(v, "strftime"):
        try:
            if v == v.normalize():
                return v.strftime("%Y-%m-%d")
        except AttributeError:
            pass
        return str(v)
    return _truncate(_sanitize(str(v)), _EXAMPLE_VALUE_MAX_CHARS)


def _sample_row_lines(df: pd.DataFrame, columns: list) -> list[str]:
    if df.empty or not columns:
        return ["  (no rows)"]

    sample = df[columns].head(_MAX_SAMPLE_ROWS)
    n = len(sample)

    formatted = [
        [_fmt_sample_value(v) for v in sample[col].tolist()] for col in columns
    ]
    widths = [max((len(v) for v in col_vals), default=0) for col_vals in formatted]

    lines = []
    for i in range(n):
        parts = [formatted[c][i].ljust(widths[c]) for c in range(len(columns))]
        lines.append(("  " + " | ".join(parts)).rstrip())
    return lines
