"""No network, no real Groq calls. Message construction is tested through the pure
helper _build_messages(); the only path exercised against generate() itself is the
missing-API-key error, which raises before any network call.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terplot.llm import LLMError, _build_messages, generate, strip_fences


def test_strip_fences_python_tagged_block():
    text = '```python\nsns.scatterplot(data=df, x="a", y="b")\n```'
    assert strip_fences(text) == 'sns.scatterplot(data=df, x="a", y="b")'


def test_strip_fences_bare_fenced_block():
    text = '```\nsns.scatterplot(data=df, x="a", y="b")\n```'
    assert strip_fences(text) == 'sns.scatterplot(data=df, x="a", y="b")'


def test_strip_fences_already_bare_code():
    text = 'sns.scatterplot(data=df, x="a", y="b")'
    assert strip_fences(text) == 'sns.scatterplot(data=df, x="a", y="b")'


def test_strip_fences_prose_before_block():
    text = (
        "Sure, here's a plot for that:\n\n"
        '```python\nsns.lineplot(data=df, x="date", y="revenue")\n```'
    )
    assert strip_fences(text) == 'sns.lineplot(data=df, x="date", y="revenue")'


def test_strip_fences_prose_after_block():
    text = (
        '```python\nsns.lineplot(data=df, x="date", y="revenue")\n```\n\n'
        "Let me know if you'd like any changes!"
    )
    assert strip_fences(text) == 'sns.lineplot(data=df, x="date", y="revenue")'


def test_strip_fences_prose_before_and_after_block():
    text = (
        "Here you go:\n\n"
        '```python\nsns.barplot(data=df, x="region", y="revenue")\n```\n\n'
        "Hope that helps."
    )
    assert strip_fences(text) == 'sns.barplot(data=df, x="region", y="revenue")'


def test_strip_fences_multiple_blocks_keeps_first():
    text = (
        '```python\nsns.barplot(data=df, x="region", y="revenue")\n```\n\n'
        "Alternatively:\n\n"
        '```python\nsns.boxplot(data=df, x="region", y="revenue")\n```'
    )
    assert strip_fences(text) == 'sns.barplot(data=df, x="region", y="revenue")'


def test_strip_fences_multiline_code_preserved():
    code = 'df["date"] = pd.to_datetime(df["date"])\nsns.lineplot(data=df, x="date", y="revenue")'
    text = f"```python\n{code}\n```"
    assert strip_fences(text) == code


def test_strip_fences_empty_string():
    assert strip_fences("") == ""


def test_strip_fences_none_never_raises():
    assert strip_fences(None) == ""


def test_strip_fences_whitespace_only():
    assert strip_fences("   \n\n  ") == ""


def test_strip_fences_bare_code_with_surrounding_whitespace():
    assert strip_fences("  \n  x = 1\n  ") == "x = 1"


def test_strip_fences_fenced_block_with_trailing_whitespace_inside():
    text = '```python\nsns.histplot(data=df, x="revenue")\n\n```'
    assert strip_fences(text) == 'sns.histplot(data=df, x="revenue")'


def test_strip_fences_unclosed_fence_falls_back_to_stripped_input():
    text = '```python\nsns.histplot(data=df, x="revenue")'
    result = strip_fences(text)
    assert result == text.strip()


def test_strip_fences_other_language_tag():
    text = '```py\nsns.histplot(data=df, x="revenue")\n```'
    assert strip_fences(text) == 'sns.histplot(data=df, x="revenue")'


_SCHEMA = (
    "sales.csv — 4,301 rows x 3 cols\n\n"
    'date     object   0 null    e.g. "2024-01-15"\n'
    "revenue  float64  12 null   min 0.0  max 98421.3\n"
)


def test_build_messages_first_turn_shape():
    messages = _build_messages(_SCHEMA, [], "revenue over time", error=None)

    assert messages[0]["role"] == "system"
    assert "df" in messages[0]["content"]

    roles = [m["role"] for m in messages]
    assert roles[0] == "system"
    for i in range(1, len(roles)):
        if roles[i] == roles[i - 1]:
            raise AssertionError(f"roles did not alternate: {roles}")

    assert any(_SCHEMA in m["content"] for m in messages)

    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "revenue over time"


def test_build_messages_history_appears_in_order():
    history = [
        ("plot revenue over time", 'sns.lineplot(data=df, x="date", y="revenue")'),
        ("now split by region", 'sns.lineplot(data=df, x="date", y="revenue", hue="region")'),
    ]
    messages = _build_messages(_SCHEMA, history, "make it a bar chart", error=None)

    contents = [m["content"] for m in messages]

    idx_req1 = next(i for i, c in enumerate(contents) if c == "plot revenue over time")
    idx_snip1 = next(
        i for i, c in enumerate(contents) if 'sns.lineplot(data=df, x="date", y="revenue")' in c
    )
    idx_req2 = next(i for i, c in enumerate(contents) if c == "now split by region")
    idx_snip2 = next(
        i
        for i, c in enumerate(contents)
        if 'hue="region"' in c
    )
    idx_final = len(messages) - 1

    assert idx_req1 < idx_snip1 < idx_req2 < idx_snip2 < idx_final

    assert messages[idx_req1]["role"] == "user"
    assert messages[idx_snip1]["role"] == "assistant"
    assert messages[idx_req2]["role"] == "user"
    assert messages[idx_snip2]["role"] == "assistant"

    assert messages[idx_final]["content"] == "make it a bar chart"
    assert messages[idx_final]["role"] == "user"


def test_build_messages_roles_alternate_with_history():
    history = [
        ("a", "snippet_a"),
        ("b", "snippet_b"),
        ("c", "snippet_c"),
    ]
    messages = _build_messages(_SCHEMA, history, "d", error=None)
    roles = [m["role"] for m in messages]
    assert roles[0] == "system"
    for i in range(1, len(roles)):
        assert roles[i] != roles[i - 1], f"roles did not alternate at index {i}: {roles}"


def test_build_messages_error_text_in_final_message():
    history = [
        ("plot revenue over time", 'sns.lineplot(data=df, x="dat", y="revenue")'),
    ]
    error_text = 'KeyError: "dat"'
    messages = _build_messages(_SCHEMA, history, "plot revenue over time", error=error_text)

    assert messages[-1]["role"] == "user"
    assert error_text in messages[-1]["content"]


def test_build_messages_error_none_uses_plain_request():
    messages = _build_messages(_SCHEMA, [], "plot revenue over time", error=None)
    assert messages[-1]["content"] == "plot revenue over time"


def test_build_messages_all_content_values_are_strings():
    history = [("a", "snip_a")]
    messages = _build_messages(_SCHEMA, history, "b", error="boom")
    for m in messages:
        assert isinstance(m["content"], str)
        assert isinstance(m["role"], str)


def test_generate_raises_llmerror_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    try:
        generate(_SCHEMA, [], "plot revenue over time")
        raise AssertionError("expected LLMError to be raised")
    except LLMError as exc:
        assert "GROQ_API_KEY" in str(exc)


def test_generate_error_message_mentions_how_to_set_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    try:
        generate(_SCHEMA, [], "plot revenue over time")
        raise AssertionError("expected LLMError to be raised")
    except LLMError as exc:
        msg = str(exc)
        assert "GROQ_API_KEY" in msg
        # must say how to set it, not just that it's missing
        assert len(msg) > len("GROQ_API_KEY is not set.")


if __name__ == "__main__":
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__, "-q"]))
