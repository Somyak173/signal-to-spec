"""Sentiment, urgency and opportunity scoring (a RICE-style model, explained in the README)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .themes import Theme

# How much one account counts toward Reach, by plan. Paid and larger plans count more
# because more seats and more revenue sit behind each request. Tune these per business.
SEGMENT_WEIGHTS = {"Free": 1.0, "Standard": 2.0, "Premium": 3.0, "Enterprise": 5.0}
DEFAULT_SEGMENT_WEIGHT = 1.0

# Language that signals urgency: churn or revenue risk, a blocked workflow, or willingness to pay.
URGENCY = re.compile(
    r"(?:can't|cannot|can not|blocker|blocking|blocked|dealbreaker|deal breaker|unusable|useless|"
    r"broken|crash|fail|evaluating other tools|switching to|alternatives|renewal|churn|cancel|"
    r"every (?:day|morning)|daily|compliance|security team|gave up|impossible|times out|"
    r"we'd pay|pay extra|upgrade|top request|leadership|execs|competitor|killer feature)",
    re.IGNORECASE,
)
# Complaint or request language. Positive-sounding feedback that contains any of this is
# NOT praise ("would love a summary feature" is a request, not a compliment).
NOT_PRAISE = re.compile(
    r"(?:can't|cannot|no way|missing|broken|fail|slow|crash|useless|confus|impossible|lag|hang|"
    r"time[sd]? out|wish|should|lack|weak|limited|problem|complaint|annoying|too many|too much|"
    r"skipped|request|would (?:love|help|be)|can relay|could|please|need|want|\?)",
    re.IGNORECASE,
)

_analyzer = SentimentIntensityAnalyzer()


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sentiment"] = df["clean_text"].map(lambda t: _analyzer.polarity_scores(t)["compound"])
    df["urgent"] = df["clean_text"].str.contains(URGENCY)
    df["is_praise"] = (df["sentiment"] >= 0.4) & ~df["clean_text"].str.contains(NOT_PRAISE)
    return df


def _impact(pain: float, urgency: float) -> float:
    """Map pain and urgency onto the standard RICE impact scale (0.25 to 3)."""
    raw = 0.6 * pain + 0.4 * urgency
    if raw >= 0.60:
        return 3.0
    if raw >= 0.48:
        return 2.0
    if raw >= 0.36:
        return 1.0
    if raw >= 0.24:
        return 0.5
    return 0.25


def _confidence(n_sources: int, n_segments: int, mentions: int, total_sources: int, total_segments: int) -> float:
    """Evidence quality: is the theme corroborated across channels, plans and enough customers?"""
    breadth = 0.5 * n_sources / max(total_sources, 1) + 0.3 * n_segments / max(total_segments, 1)
    volume = 0.2 * min(mentions / 25, 1.0)
    score = breadth + volume
    if score >= 0.9:
        return 1.0
    if score >= 0.7:
        return 0.8
    return 0.5


def _momentum(dates: pd.Series, end: pd.Timestamp, window_days: int = 28) -> tuple[float, int, int]:
    recent = int((dates > end - pd.Timedelta(days=window_days)).sum())
    prior = int(((dates <= end - pd.Timedelta(days=window_days)) & (dates > end - pd.Timedelta(days=2 * window_days))).sum())
    change = (recent - prior) / prior if prior else (1.0 if recent else 0.0)
    return change, recent, prior


def weekly_counts(dates: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> list[int]:
    weeks = pd.date_range(start.normalize(), end.normalize() + pd.Timedelta(days=6), freq="7D")
    bins = np.searchsorted(weeks.values, dates.values, side="right") - 1
    counts = np.bincount(bins.clip(0), minlength=len(weeks))
    return counts[: len(weeks)].tolist()


def load_review(path: str | Path | None) -> pd.DataFrame | None:
    """Optional team review file: `match` (a keyword) plus `name` and/or `effort_weeks`.

    This is the human step: a PM names the themes they recognise and engineering adds
    rough estimates. Each row applies to the first theme whose keywords contain `match`.
    """
    if not path:
        return None
    review = pd.read_csv(path)
    review.columns = [c.strip().lower() for c in review.columns]
    if "match" not in review.columns or not ({"name", "effort_weeks"} & set(review.columns)):
        raise ValueError("Review file needs a `match` column plus `name` and/or `effort_weeks`")
    review["match"] = review["match"].astype(str).str.lower().str.strip()
    for col in ("name", "effort_weeks"):
        if col not in review.columns:
            review[col] = np.nan
    review["effort_weeks"] = pd.to_numeric(review["effort_weeks"], errors="coerce")
    return review


def _review_row(theme: Theme, review: pd.DataFrame | None) -> pd.Series | None:
    if review is None:
        return None
    haystack = " ".join(theme.keywords).lower() + " " + theme.label.lower()
    for _, r in review.iterrows():
        if r["match"] and r["match"] in haystack:
            return r
    return None


def apply_review_names(themes: list[Theme], review: pd.DataFrame | None) -> int:
    """Give themes the names the team chose. A human name beats a Claude-written one."""
    named = 0
    for t in themes:
        r = _review_row(t, review)
        if r is not None and isinstance(r["name"], str) and r["name"].strip():
            t.name = r["name"].strip()
            named += 1
    return named


def score_themes(df: pd.DataFrame, themes: list[Theme], review: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per theme with Reach, Impact, Confidence, Effort and the resulting scores."""
    end = df["date"].max()
    start = df["date"].min()
    total_sources = df["source"].nunique()
    total_segments = df["segment"].nunique()
    rows = []
    for t in themes:
        items = df.iloc[t.member_index]
        accounts = items.drop_duplicates("account")
        reach = float(sum(SEGMENT_WEIGHTS.get(s, DEFAULT_SEGMENT_WEIGHT) for s in accounts["segment"]))
        pain = float((1 - items["sentiment"].mean()) / 2)  # 0 = delighted, 1 = furious
        urgency = float(items["urgent"].mean())
        impact = _impact(pain, urgency)
        confidence = _confidence(
            items["source"].nunique(), items["segment"].nunique(), len(items), total_sources, total_segments
        )
        change, recent, prior = _momentum(items["date"], end)

        r = _review_row(t, review)
        effort_weeks = float(r["effort_weeks"]) if r is not None and pd.notna(r["effort_weeks"]) else np.nan

        opportunity = reach * impact * confidence
        rows.append(
            {
                "theme_id": t.theme_id,
                "theme": t.display_name,
                "keywords": ", ".join(t.keywords),
                "mentions": len(items),
                "accounts": len(accounts),
                "seats_affected": int(accounts["seats"].sum()),
                "reach": round(reach, 1),
                "avg_sentiment": round(float(items["sentiment"].mean()), 3),
                "pain": round(pain, 3),
                "urgency": round(urgency, 3),
                "impact": impact,
                "confidence": confidence,
                "momentum_pct": round(change * 100),
                "last_4w": recent,
                "prior_4w": prior,
                "opportunity_score": round(opportunity, 1),
                "effort_weeks": effort_weeks,
                "rice": round(opportunity / effort_weeks, 1) if effort_weeks > 0 else np.nan,
                "coherence": t.coherence,
                "needs_review": t.is_mixed,
                "segment_mix": items["segment"].value_counts().to_dict(),
                "source_mix": items["source"].value_counts().to_dict(),
                "weekly": weekly_counts(items["date"], start, end),
            }
        )
    out = pd.DataFrame(rows)
    sort_col = "rice" if out["rice"].notna().any() else "opportunity_score"
    # Mixed (low-coherence) groups go last and get no rank: they need a human to split them.
    out = out.sort_values(
        ["needs_review", sort_col, "opportunity_score"], ascending=[True, False, False], na_position="last"
    ).reset_index(drop=True)
    ranked = (~out["needs_review"]).cumsum()
    out.insert(0, "rank", ranked.where(~out["needs_review"]).astype("Int64"))
    return out
