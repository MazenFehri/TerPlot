"""No network, no LLM — small hardcoded snippets stand in for model output."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terplot.runner import run


def _write_csv(path):
    path.write_text("a,b\n1,2\n2,4\n3,6\n", encoding="utf-8")


def test_success_end_to_end(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)
    out_path = tmp_path / "plot.png"

    load_expr = f'pd.read_csv(r"{csv_path}")'
    snippet = 'sns.scatterplot(data=df, x="a", y="b")'

    ok, stderr = run(snippet, load_expr, str(out_path), timeout=30)

    assert ok is True, f"expected success, got stderr: {stderr}"
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_snippet_raises_missing_column(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)
    out_path = tmp_path / "plot.png"

    load_expr = f'pd.read_csv(r"{csv_path}")'
    snippet = 'sns.scatterplot(data=df, x="does_not_exist", y="b")'

    ok, stderr = run(snippet, load_expr, str(out_path), timeout=30)

    assert ok is False
    assert not out_path.exists()
    assert stderr != ""
    assert "Error" in stderr or "error" in stderr


def test_snippet_runs_but_draws_nothing(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)
    out_path = tmp_path / "plot.png"

    load_expr = f'pd.read_csv(r"{csv_path}")'
    snippet = "x = 1 + 1"  # exits cleanly, never touches the figure

    ok, stderr = run(snippet, load_expr, str(out_path), timeout=30)

    assert ok is False
    assert not out_path.exists()
    assert "no plot" in stderr


def test_timeout(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)
    out_path = tmp_path / "plot.png"

    load_expr = f'pd.read_csv(r"{csv_path}")'
    snippet = "import time; time.sleep(5)"

    ok, stderr = run(snippet, load_expr, str(out_path), timeout=1)

    assert ok is False
    assert "timed out" in stderr
    assert not out_path.exists()


def test_out_dir_created_if_missing(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)
    out_path = tmp_path / "nested" / "subdir" / "plot.png"

    assert not out_path.parent.exists()

    load_expr = f'pd.read_csv(r"{csv_path}")'
    snippet = 'sns.scatterplot(data=df, x="a", y="b")'

    ok, stderr = run(snippet, load_expr, str(out_path), timeout=30)

    assert ok is True, f"expected success, got stderr: {stderr}"
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_windows_backslash_path_escaping(tmp_path):
    """Regression test: out_path with backslashes (Windows-style) must not
    break the generated script, whether or not it happens to end in a
    backslash (which would break a naive r"..." literal)."""
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)
    out_dir = tmp_path / "windows style dir"
    out_dir.mkdir()
    out_path = out_dir / "plot with spaces.png"

    load_expr = f'pd.read_csv(r"{csv_path}")'
    snippet = 'sns.scatterplot(data=df, x="a", y="b")'

    ok, stderr = run(snippet, load_expr, str(out_path), timeout=30)

    assert ok is True, f"expected success, got stderr: {stderr}"
    assert out_path.exists()
    assert out_path.stat().st_size > 0
