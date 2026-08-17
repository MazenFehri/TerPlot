"""The .env reader and the retry loop.

A key silently mangled by the .env reader looks exactly like a bad API key, so the
awkward formats are worth pinning down. The retry loop only fires when generated
code fails, which is exactly when nobody is watching — so it gets stubbed tests
rather than waiting for the model to make a mistake on its own.
"""

import os
from pathlib import Path

import terplot.cli as cli
from terplot.cli import load_env, turn
from terplot.llm import LLMError


def _write(tmp_path, text):
    p = tmp_path / ".env"
    p.write_text(text, encoding="utf-8")
    return p


def test_reads_a_plain_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    load_env(_write(tmp_path, "GROQ_API_KEY=gsk_abc123\n"))
    assert os.environ["GROQ_API_KEY"] == "gsk_abc123"


def test_strips_quotes_comments_blanks_and_export(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TERPLOT_MODEL", raising=False)
    _write(
        tmp_path,
        '# a comment\n'
        '\n'
        'export GROQ_API_KEY="gsk_quoted"\n'
        "TERPLOT_MODEL='some-model'\n",
    )
    load_env(tmp_path / ".env")
    assert os.environ["GROQ_API_KEY"] == "gsk_quoted"
    assert os.environ["TERPLOT_MODEL"] == "some-model"


def test_value_containing_equals_is_not_truncated(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_TOKEN", raising=False)
    load_env(_write(tmp_path, "SOME_TOKEN=abc=def==\n"))
    assert os.environ["SOME_TOKEN"] == "abc=def=="


def test_real_environment_wins_over_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "from_shell")
    load_env(_write(tmp_path, "GROQ_API_KEY=from_file\n"))
    assert os.environ["GROQ_API_KEY"] == "from_shell"


def test_missing_file_is_not_an_error(tmp_path):
    load_env(tmp_path / "nope.env")  # must not raise


def _stub(monkeypatch, snippets, results):
    """Stub generate/run. Returns the list of generate() kwargs actually seen."""
    calls = []

    def fake_generate(schema, history, request, error=None, model=None):
        calls.append({"history": list(history), "request": request, "error": error})
        return snippets[len(calls) - 1]

    runs = iter(results)
    monkeypatch.setattr(cli, "generate", fake_generate)
    monkeypatch.setattr(cli, "run", lambda *a, **k: next(runs))
    return calls


def test_first_attempt_succeeds_without_retrying(tmp_path, monkeypatch):
    calls = _stub(monkeypatch, ["good()"], [(True, "")])
    got = turn("plot it", "schema", [], "pd.read_csv('x')", tmp_path / "p.png", False)
    assert got == "good()"
    assert len(calls) == 1


def test_failure_retries_once_with_the_error_and_the_broken_snippet(tmp_path, monkeypatch):
    calls = _stub(monkeypatch, ["bad()", "fixed()"], [(False, "NameError: nope"), (True, "")])

    got = turn("plot it", "schema", [], "pd.read_csv('x')", tmp_path / "p.png", False)

    assert got == "fixed()"
    assert len(calls) == 2
    # The retry must carry both the stderr and the snippet that produced it.
    assert calls[1]["error"] == "NameError: nope"
    assert calls[1]["history"][-1] == ("plot it", "bad()")


def test_two_failures_give_up_and_return_none(tmp_path, monkeypatch):
    calls = _stub(monkeypatch, ["bad()", "still_bad()"], [(False, "boom"), (False, "boom again")])
    got = turn("plot it", "schema", [], "pd.read_csv('x')", tmp_path / "p.png", False)
    assert got is None
    assert len(calls) == 2  # exactly one retry, never two


def test_model_error_returns_none_instead_of_propagating(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise LLMError("no key")

    monkeypatch.setattr(cli, "generate", boom)
    assert turn("plot it", "schema", [], "pd.read_csv('x')", tmp_path / "p.png", False) is None


def test_first_existing_candidate_wins(tmp_path, monkeypatch):
    monkeypatch.delenv("PICKED", raising=False)
    second = tmp_path / "second.env"
    second.write_text("PICKED=second\n", encoding="utf-8")
    first = tmp_path / "first.env"
    first.write_text("PICKED=first\n", encoding="utf-8")
    load_env(first, second)
    assert os.environ["PICKED"] == "first"
