"""Optional Claude layer: names themes in plain English and drafts PRD sections.

Everything here is optional. Without the `anthropic` package or an ANTHROPIC_API_KEY
the pipeline runs fully offline and produces keyword-labelled themes and template PRDs.
Every call is grounded in the feedback it is given and fails soft: if a call errors,
the deterministic output is kept and a warning is printed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re

import pandas as pd

from .themes import Theme

DEFAULT_MODEL = os.environ.get("S2S_MODEL", "claude-sonnet-5-5")


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY")) and importlib.util.find_spec("anthropic") is not None


def _ask(prompt: str, max_tokens: int = 1500) -> str:
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")


def _json(text: str):
    """Pull the first JSON object or array out of a model reply."""
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON in reply")
    return json.loads(match.group(1))


def name_themes(themes: list[Theme], df: pd.DataFrame, quotes_per_theme: int = 6) -> None:
    """Give each theme a short human name and a one-line problem statement (in place)."""
    payload = []
    for t in themes:
        if t.is_mixed:
            continue
        sample = df.iloc[t.member_index]["clean_text"].drop_duplicates().head(quotes_per_theme).tolist()
        payload.append({"id": t.theme_id, "keywords": t.keywords, "feedback": sample})
    prompt = (
        "You are a product manager synthesising customer feedback.\n"
        "For each theme below, write:\n"
        '- "name": 3-6 words naming the customer problem (not a solution), sentence case\n'
        '- "problem": one sentence stating who is affected and what goes wrong, using only the feedback given\n'
        'Return only a JSON array of {"id", "name", "problem"}.\n\n'
        + json.dumps(payload, indent=1)
    )
    try:
        named = {item["id"]: item for item in _json(_ask(prompt))}
    except Exception as exc:  # noqa: BLE001 - optional layer must never break the run
        print(f"  ! Claude theme naming skipped ({exc.__class__.__name__}: {exc})")
        return
    for t in themes:
        if t.theme_id in named:
            t.name = str(named[t.theme_id].get("name", "")).strip()
            t.problem = str(named[t.theme_id].get("problem", "")).strip()


def draft_prd_sections(theme: Theme, stats: dict, quotes: list[str]) -> dict | None:
    """User stories with acceptance criteria, risks and open questions for one theme."""
    prompt = (
        "You are a product manager writing the first draft of a PRD for engineering and design.\n"
        "Use ONLY the evidence below. Do not invent numbers, customers or features they did not ask for.\n\n"
        f"Theme: {theme.display_name}\nProblem: {theme.problem or '(not yet written)'}\n"
        f"Evidence: {json.dumps(stats)}\nCustomer feedback:\n"
        + "\n".join(f"- {q}" for q in quotes)
        + "\n\nReturn only JSON with keys:\n"
        '"user_stories": 3-5 items of {"story": "As a ..., I want ..., so that ...", '
        '"acceptance_criteria": [2-3 testable statements]},\n'
        '"risks": 2-4 short strings,\n'
        '"open_questions": 3-5 questions to validate with customers or engineering.'
    )
    try:
        data = _json(_ask(prompt, max_tokens=2000))
        if not isinstance(data, dict) or "user_stories" not in data:
            raise ValueError("unexpected shape")
        return data
    except Exception as exc:  # noqa: BLE001
        print(f"  ! Claude PRD draft skipped for '{theme.display_name}' ({exc.__class__.__name__}: {exc})")
        return None
