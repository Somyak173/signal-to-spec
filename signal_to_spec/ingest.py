"""Load, validate and clean raw customer feedback."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REQUIRED_COLUMNS = ["id", "date", "source", "segment", "account", "seats", "text"]

# Prefixes that note-takers add ("Customer said: ...") carry no signal about the problem.
NOTE_PREFIXES = r"^(customer said:|interview note:|note:)\s*"

# Sentences that tell us HOW MUCH something hurts, not WHAT the problem is.
# They are kept for urgency scoring but removed from the text used to find themes,
# otherwise "please prioritize this" would pull unrelated feedback together.
URGENCY_SENTENCE = re.compile(
    r"(please prioriti[sz]e|happens (daily|every day)|evaluating other tools|switching to|"
    r"renewal|several people|not a dealbreaker|otherwise we (like|love)|blocking our rollout|"
    r"save us hours|upgrade our plan|considering alternatives)",
    re.IGNORECASE,
)


class FeedbackError(ValueError):
    """Raised when the input file can't be used."""


def _topic_text(text: str) -> str:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    kept = [s for s in sentences if not URGENCY_SENTENCE.search(s)]
    return " ".join(kept) if kept else text


def load_feedback(path: str | Path) -> pd.DataFrame:
    """Read a feedback CSV and return a clean frame.

    Adds `clean_text` (note prefixes removed) and `topic_text` (urgency sentences removed).
    """
    path = Path(path)
    if not path.exists():
        raise FeedbackError(f"File not found: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise FeedbackError(
            f"Missing column(s): {', '.join(missing)}. Expected: {', '.join(REQUIRED_COLUMNS)}"
        )

    df = df[REQUIRED_COLUMNS].copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = int(df["date"].isna().sum())
    if bad_dates:
        print(f"  ! dropped {bad_dates} row(s) with unreadable dates")
    df = df.dropna(subset=["date"])

    df["seats"] = pd.to_numeric(df["seats"], errors="coerce").fillna(1).clip(lower=1).astype(int)
    for col in ("source", "segment", "account"):
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    df["clean_text"] = df["text"].str.replace(NOTE_PREFIXES, "", regex=True, case=False).str.strip()
    df["topic_text"] = df["clean_text"].map(_topic_text)
    if df.empty:
        raise FeedbackError("No usable feedback rows after cleaning.")
    return df.reset_index(drop=True)


def drop_near_duplicates(df: pd.DataFrame, threshold: float = 0.95) -> tuple[pd.DataFrame, int]:
    """Remove the same customer saying the same thing twice (e.g. a ticket and a forum post).

    Two rows count as duplicates only when they come from the same account AND their
    text is nearly identical, so separate customers raising the same issue are kept:
    repeated asks from different accounts are signal, not noise.
    """
    keep = np.ones(len(df), dtype=bool)
    for _, group in df.groupby("account"):
        if len(group) < 2:
            continue
        idx = group.index.to_numpy()
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(group["clean_text"])
        sim = cosine_similarity(vec)
        for i in range(len(idx)):
            if not keep[idx[i]]:
                continue
            for j in range(i + 1, len(idx)):
                if sim[i, j] >= threshold:
                    keep[idx[j]] = False
    removed = int((~keep).sum())
    return df[keep].reset_index(drop=True), removed
