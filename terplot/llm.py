"""Turn a natural-language plot request into a bare seaborn/pandas snippet via
the Groq chat completions API.

The model NEVER sees or writes the wrapper (imports, data loading, figure setup,
saving). It only ever writes the plotting statements that go between PREAMBLE and
POSTAMBLE in terplot/runner.py — which is what the system prompt below enforces.
"""

from __future__ import annotations

import os
import re

DEFAULT_MODEL = "openai/gpt-oss-120b"

# Low: code generation against a fixed contract, not prose.
_TEMPERATURE = 0.2

_EXAMPLE_REQUEST = "revenue over time"
_EXAMPLE_SCHEMA = (
    "sales.csv — 4,301 rows x 3 cols\n\n"
    'date     object   0 null    e.g. "2024-01-15"\n'
    "revenue  float64  12 null   min 0.0  max 98421.3\n"
    "region   object   0 null    4 uniq: EU, US, APAC, LATAM\n"
)
_EXAMPLE_SNIPPET = (
    'df["date"] = pd.to_datetime(df["date"])\n'
    'daily = df.groupby("date", as_index=False)["revenue"].sum()\n'
    'sns.lineplot(data=daily, x="date", y="revenue")\n'
    'plt.title("Revenue Over Time")\n'
    'plt.xlabel("Date")\n'
    'plt.ylabel("Revenue")\n'
    "plt.xticks(rotation=45, ha=\"right\")"
)

SYSTEM_PROMPT = f"""You are the plotting engine inside TerPlot, a terminal app that turns a \
natural-language request into a single seaborn plot. You do not write a whole \
program — you write a short snippet of Python that is inserted into a fixed \
wrapper script which is already running by the time your code executes.

By the time your snippet runs, the wrapper has ALREADY done all of this:
  - `df` is already loaded as a pandas DataFrame containing the user's data.
  - `pd` (pandas), `sns` (seaborn), and `plt` (matplotlib.pyplot) are already imported.
  - `sns.set_theme()` has already been called.
  - `plt.figure(figsize=(10, 6))` has already been called — a figure already exists.
  - After your snippet runs, the wrapper calls `plt.savefig(...)` on your behalf.

Because of that, your snippet must NEVER contain:
  - import statements of any kind
  - `pd.read_csv`, `pd.read_excel`, or any other data loading — `df` already exists
  - `plt.savefig(...)` or `plt.show()` — the wrapper handles saving
  - `plt.figure(...)` — a figure is already open; calling it again creates a blank one
  - explanations, comments about what you're doing, markdown, or any prose outside code

Your snippet SHOULD:
  - use the exact column names from the schema you're given, spelled exactly as given \
(including case) — never guess or invent a column name
  - transform `df` first when the request needs it: `pd.to_datetime(...)` on a string \
date column, `groupby(...).agg(...)`, `resample(...)`, `pivot`, sorting, filtering, etc. \
are all fine and expected. For example, "revenue over time" against a `date` column \
stored as `object` (a string) requires converting it to datetime and aggregating before \
plotting — you are expected to do that, not to ask for it to already be done.
  - pick a seaborn/matplotlib chart type that fits the request and the column dtypes \
(lineplot for a trend over time, barplot/countplot for categorical comparisons, \
scatterplot for two numeric variables, boxplot/violinplot for a distribution split by \
category, histplot/kdeplot for a single distribution, heatmap for a matrix/correlation, \
etc.)
  - always add a clear `plt.title(...)`, and `plt.xlabel(...)` / `plt.ylabel(...)` when \
they aren't already obvious from the plot
  - rotate or shorten tick labels when they would overlap or run off the figure, e.g. \
`plt.xticks(rotation=45, ha="right")`, especially for dates or long category names

Output format — this is strict:
  - Respond with ONE fenced Python code block and nothing else: no text before it, no \
text after it, no bullet points, no "Here's the code:".
  - The code block must contain only plotting statements as described above — it must \
run correctly with no edits, immediately after the wrapper's setup.
  - If you are fixing a snippet that failed, return the full corrected snippet in the \
same format — not a diff, not just the changed lines.

Worked example:

Schema:
{_EXAMPLE_SCHEMA}
Request: {_EXAMPLE_REQUEST}

Response:
```python
{_EXAMPLE_SNIPPET}
```
"""

_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*\n)?(.*?)```", re.DOTALL)


class LLMError(Exception):
    """Raised whenever generate() can't produce a snippet — missing API key, a
    network/SDK failure, or any other error talking to Groq. The REPL catches
    this and reports it as a failed turn; it must never propagate and kill the
    REPL."""


def strip_fences(text: str) -> str:
    """Return bare Python from a model response.

    Handles, in order of preference:
      - a ```python fenced block
      - a bare ``` fenced block
      - prose before and/or after a fenced block (the fenced code is kept, prose dropped)
      - multiple fenced blocks (the first is kept)
      - code with no fences at all (returned as-is, stripped)

    Never raises. Worst case, returns the input stripped of leading/trailing whitespace.
    """
    try:
        if text is None:
            return ""
        match = _FENCE_RE.search(text)
        if match:
            return match.group(1).strip()
        return text.strip()
    except Exception:
        try:
            return text.strip()
        except Exception:
            return ""


def _build_messages(
    schema: str,
    history: list[tuple[str, str]],
    request: str,
    error: str | None = None,
) -> list[dict]:
    """Build the Groq chat messages list. Pure function, no network — this is what
    makes generate() testable without hitting the API.

    Shape: system prompt, schema as context, then history as alternating
    user/assistant turns, then the current request. When `error` is set, the final
    user turn becomes a fix request carrying the stderr.
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    messages.append(
        {
            "role": "user",
            "content": f"Here is the schema of the loaded data:\n\n{schema}",
        }
    )
    messages.append(
        {
            "role": "assistant",
            "content": "Understood. I'll write only plotting statements using those "
            "exact column names, with no imports and no data loading.",
        }
    )

    for user_request, snippet in history:
        messages.append({"role": "user", "content": user_request})
        messages.append({"role": "assistant", "content": f"```python\n{snippet}\n```"})

    if error is not None:
        messages.append(
            {
                "role": "user",
                "content": (
                    "The snippet you just wrote failed when it was executed. "
                    "Here is the stderr it produced:\n\n"
                    f"{error}\n\n"
                    "Fix the bug and return the complete corrected snippet — not a "
                    "diff, the whole thing — in the same fenced-code-block format. "
                    "Keep addressing the original request: "
                    f"{request}"
                ),
            }
        )
    else:
        messages.append({"role": "user", "content": request})

    return messages


def generate(
    schema: str,
    history: list[tuple[str, str]],
    request: str,
    error: str | None = None,
    model: str | None = None,
) -> str:
    """Call Groq chat completions, return a bare Python snippet.

    history: [(user_request, snippet), ...] oldest first, already trimmed by caller.
    error:   if set, the LAST history entry is the snippet that just failed, and
             `error` is its stderr — ask the model to fix it.
    model:   defaults to env TERPLOT_MODEL, else 'openai/gpt-oss-120b'.
    Raises LLMError if GROQ_API_KEY is unset or the API call fails.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMError(
            "GROQ_API_KEY is not set. Get a key from https://console.groq.com/keys "
            "and set it, e.g.:\n"
            "  export GROQ_API_KEY=your-key-here   (macOS/Linux)\n"
            "  setx GROQ_API_KEY your-key-here      (Windows)"
        )

    resolved_model = model or os.environ.get("TERPLOT_MODEL") or DEFAULT_MODEL
    messages = _build_messages(schema, history, request, error)

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            temperature=_TEMPERATURE,
        )
        content = completion.choices[0].message.content
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"Groq API call failed: {exc}") from exc

    return strip_fences(content)
