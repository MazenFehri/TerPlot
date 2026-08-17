"""Cross-module seam: data.load()'s load_expr must survive being interpolated into
runner.py's generated script. Each module's own tests pass without this holding.

No network, no LLM — a hand-written snippet stands in for the model.
"""

import pandas as pd

from terplot.data import load, profile
from terplot.runner import run

CSV = "date,region,revenue\n2024-01-15,EU,1203.5\n2024-01-16,US,980.2\n2024-01-17,EU,1544.0\n"

PLOT = 'sns.scatterplot(data=df, x="date", y="revenue", hue="region")'


def _sample(tmp_path):
    # A subdirectory with a space, so the path has real backslashes on Windows.
    d = tmp_path / "my data"
    d.mkdir()
    csv = d / "sales.csv"
    csv.write_text(CSV, encoding="utf-8")
    return csv


def test_load_expr_survives_into_the_generated_script(tmp_path):
    csv = _sample(tmp_path)
    df, load_expr = load(str(csv))
    out = tmp_path / "plots" / "plot_001.png"

    ok, err = run(PLOT, load_expr, str(out))

    assert ok, err
    assert out.stat().st_size > 0


def test_load_expr_is_evaluable_python(tmp_path):
    csv = _sample(tmp_path)
    df, load_expr = load(str(csv))
    assert eval(load_expr, {"pd": pd}).equals(df)


def test_profile_names_every_column(tmp_path):
    csv = _sample(tmp_path)
    df, _ = load(str(csv))
    schema = profile(df, "sales.csv")
    for col in df.columns:
        assert col in schema
    assert len(schema) < 2000


def test_bad_column_fails_with_an_error_the_model_can_act_on(tmp_path):
    csv = _sample(tmp_path)
    _, load_expr = load(str(csv))
    out = tmp_path / "plot.png"

    ok, err = run('sns.scatterplot(data=df, x="nope", y="revenue")', load_expr, str(out))

    assert not ok
    assert "nope" in err


def test_snippet_that_draws_nothing_is_a_failure(tmp_path):
    csv = _sample(tmp_path)
    _, load_expr = load(str(csv))
    out = tmp_path / "plot.png"

    ok, err = run('x = df["revenue"].sum()', load_expr, str(out))

    assert not ok
    assert not out.exists()
