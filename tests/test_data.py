"""No network. Uses tmp_path so every path is real and, on Windows, contains real
backslashes — that's what exercises the load_expr escaping.
"""

from __future__ import annotations

import pandas as pd
import pytest

from terplot.data import load, profile


def test_load_csv_roundtrip_and_load_expr_is_valid_python(tmp_path):
    subdir = tmp_path / "sub dir" / "nested"
    subdir.mkdir(parents=True)
    csv_path = subdir / "sales.csv"

    original = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    original.to_csv(csv_path, index=False)

    df, load_expr = load(str(csv_path))

    assert isinstance(load_expr, str)
    assert "read_csv" in load_expr
    assert df.equals(original)

    # The escaping test: valid Python despite backslashes and a space in the path.
    reloaded = eval(load_expr, {"pd": pd})
    assert reloaded.equals(df)


def test_load_xlsx_roundtrip_and_load_expr_is_valid_python(tmp_path):
    subdir = tmp_path / "sub dir" / "nested"
    subdir.mkdir(parents=True)
    xlsx_path = subdir / "sales.xlsx"

    original = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    original.to_excel(xlsx_path, index=False)

    df, load_expr = load(str(xlsx_path))

    assert isinstance(load_expr, str)
    assert "read_excel" in load_expr
    assert "sheet_name=0" in load_expr
    assert df.equals(original)

    reloaded = eval(load_expr, {"pd": pd})
    assert reloaded.equals(df)


def test_load_xls_extension_is_routed_to_read_excel(tmp_path):
    # .xls dispatches to the same read_excel path, so no real legacy file needed.
    xlsx_path = tmp_path / "legacy.xls"
    pd.DataFrame({"a": [1]}).to_excel(xlsx_path, index=False, engine="openpyxl")

    df, load_expr = load(str(xlsx_path))
    assert "read_excel" in load_expr
    assert "sheet_name=0" in load_expr


def test_load_missing_file_raises_file_not_found_error(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load(str(missing))


def test_load_unsupported_extension_raises_value_error(tmp_path):
    bad_path = tmp_path / "notes.txt"
    bad_path.write_text("just some text", encoding="utf-8")
    with pytest.raises(ValueError):
        load(str(bad_path))


def test_load_returns_absolute_path_in_expr(tmp_path, monkeypatch):
    csv_path = tmp_path / "rel.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv_path, index=False)

    monkeypatch.chdir(tmp_path)
    df, load_expr = load("rel.csv")

    # Must embed the absolute path so the reload works from any cwd. load_expr
    # holds repr()'d text, so compare against repr(str(...)), not the raw path.
    assert repr(str(tmp_path.resolve() / "rel.csv"))[1:-1] in load_expr

    reloaded = eval(load_expr, {"pd": pd})
    assert reloaded.equals(df)


def test_profile_matches_spec_example_layout():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-15", "2024-01-16", "2024-01-17"]),
            "revenue": [1203.5, 980.2, 1544.0],
            "region": ["EU", "US", "APAC"],
        }
    )
    text = profile(df, "sales.csv")

    assert "sales.csv" in text
    assert "3 rows" in text
    assert "3 cols" in text
    assert "date" in text and "revenue" in text and "region" in text
    assert "min" in text and "max" in text
    assert "uniq:" in text
    assert "sample rows:" in text
    assert "1203.5" in text
    assert "EU" in text


def test_profile_names_every_column_and_shows_row_count_for_mixed_frame():
    df = pd.DataFrame(
        {
            "id": range(30),
            "amount": [float(i) * 1.5 for i in range(30)],
            "status": (["active", "inactive", "pending"] * 10),
            "free_text": [f"note number {i} with detail" for i in range(30)],
        }
    )
    text = profile(df, "mixed.csv")

    assert "30 rows" in text
    assert "4 cols" in text
    for col in df.columns:
        assert col in text

    assert "min" in text and "max" in text
    assert "uniq:" in text
    assert "active" in text
    # 30 distinct free-text values is above the uniq threshold
    assert 'e.g. "' in text


def test_profile_all_null_column_does_not_crash():
    df = pd.DataFrame({"x": [None, None, None], "y": [1, 2, 3]})
    text = profile(df, "n.csv")

    assert "x" in text
    assert "all null" in text
    assert "3 null" in text


def test_profile_empty_dataframe_does_not_crash():
    df = pd.DataFrame()
    text = profile(df, "empty.csv")

    assert "0 rows" in text
    assert "0 cols" in text
    assert "sample rows:" in text


def test_profile_empty_dataframe_with_columns_but_no_rows():
    df = pd.DataFrame({"a": pd.Series([], dtype="float64"), "b": pd.Series([], dtype="object")})
    text = profile(df, "empty2.csv")

    assert "0 rows" in text
    assert "a" in text and "b" in text
    assert "no data" in text


def test_profile_float_with_long_binary_tail_is_rounded():
    df = pd.DataFrame({"v": [0.1 + 0.2, 1.0 / 3.0, 100.0]})
    text = profile(df, "f.csv")

    assert "0.30000000000000004" not in text
    for line in text.splitlines():
        assert len(line) < 200


def test_profile_value_with_newline_and_pipe_stays_on_one_line():
    df = pd.DataFrame({"note": ["line1\nline2", "has|pipe", "normal"]})
    text = profile(df, "np.csv")

    lines = text.splitlines()
    # an embedded newline must not have split the value across two lines
    assert any("line1" in l and "line2" in l for l in lines)
    # a raw pipe must not read as a column separator in the sample rows
    sample_section = text.split("sample rows:")[-1]
    assert "has|pipe" not in sample_section


def test_profile_very_long_column_name_does_not_crash():
    df = pd.DataFrame({"a" * 80: [1, 2, 3]})
    text = profile(df, "l.csv")
    assert "a" * 80 in text


def test_profile_wide_frame_under_2000_chars():
    data = {}
    for i in range(60):
        if i % 2 == 0:
            data[f"col_{i}"] = list(range(50))
        else:
            data[f"col_{i}"] = (["low", "med", "high"] * 17)[:50]
    df = pd.DataFrame(data)

    text = profile(df, "wide.csv")

    assert len(text) < 2000
    assert "60 cols" in text
    assert "not shown" in text


def test_profile_adversarial_wide_frame_still_under_2000_chars():
    # Worst case for the char budget: every column near the 20-unique ceiling.
    import random

    random.seed(0)
    data = {}
    for i in range(60):
        if i % 3 == 0:
            data[f"col_{i}"] = [random.random() * 1000 for _ in range(5)]
        elif i % 3 == 1:
            data[f"col_{i}"] = [random.choice(["alpha", "beta", "gamma"]) for _ in range(5)]
        else:
            data[f"col_{i}"] = [f"value_{i}_{j}" for j in range(5)]
    df = pd.DataFrame(data)

    text = profile(df, "wide_adversarial.csv")
    assert len(text) < 2000
