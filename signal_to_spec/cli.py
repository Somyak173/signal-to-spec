"""Command line: feedback CSV in, ranked opportunities + PRD + Jira backlog + report out."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from . import llm
from .artifacts import build_jira_rows, build_prd, write_jira_csv, write_opportunities_csv
from .ingest import FeedbackError, drop_near_duplicates, load_feedback
from .report import build_report
from .scoring import add_sentiment, apply_review_names, load_review, score_themes
from .themes import discover_themes


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "theme"


def run(
    input_path: str | Path,
    out_dir: str | Path = "output",
    review_path: str | Path | None = None,
    k: int | None = None,
    top_n: int = 3,
    use_llm: bool = False,
    quiet: bool = False,
) -> dict:
    """Run the full pipeline. Returns paths and the scored table (handy for tests and notebooks)."""
    say = (lambda *a: None) if quiet else print
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    say(f"1/5  Loading {input_path}")
    df = load_feedback(input_path)
    df, dupes = drop_near_duplicates(df)
    df = add_sentiment(df)
    praise = df[df["is_praise"]].reset_index(drop=True)
    work = df[~df["is_praise"]].reset_index(drop=True)
    say(f"     {len(df)} items after removing {dupes} duplicate(s); {len(praise)} routed to “what's working”")

    say("2/5  Finding themes")
    themes, diag = discover_themes(work, k=k)
    say(f"     {len(themes)} themes (k={diag['k']}, {diag['merged_fragments']} split theme(s) merged)")

    want_llm = use_llm and llm.available()
    if use_llm and not want_llm:
        say("     ! --llm ignored: set ANTHROPIC_API_KEY and `pip install anthropic` to enable it")
    if want_llm:
        say("     Naming themes with Claude")
        llm.name_themes(themes, work)

    say("3/5  Scoring opportunities")
    review = load_review(review_path)
    if review is not None:
        say(f"     Applied team review: {apply_review_names(themes, review)} theme name(s), effort estimates where given")
    scored = score_themes(work, themes, review)
    ranked = scored[~scored["needs_review"]]
    by_id = {t.theme_id: t for t in themes}

    say("4/5  Writing PRD and backlog")
    llm_sections: dict[int, dict] = {}
    top = [(by_id[r["theme_id"]], r) for _, r in ranked.head(top_n).iterrows()]
    prd_paths = []
    for i, (theme, row) in enumerate(top):
        if want_llm:
            quotes = work.iloc[theme.member_index]["clean_text"].drop_duplicates().head(12).tolist()
            stats = {k_: row[k_] for k_ in ("mentions", "accounts", "seats_affected", "urgency", "momentum_pct")}
            stats = {k_: (v.item() if hasattr(v, "item") else v) for k_, v in stats.items()}
            sections = llm.draft_prd_sections(theme, stats, quotes)
            if sections:
                llm_sections[theme.theme_id] = sections
        # Full PRD for the #1 opportunity; the next ones get the same template for comparison.
        prd = build_prd(theme, row, work, len(ranked), llm_sections.get(theme.theme_id))
        path = out / f"prd_{i + 1}_{_slug(theme.display_name)}.md"
        path.write_text(prd, encoding="utf-8")
        prd_paths.append(path)

    jira_path = out / "jira_backlog.csv"
    write_jira_csv(build_jira_rows(top, work, llm_sections), jira_path)
    opp_path = out / "opportunities.csv"
    write_opportunities_csv(scored, opp_path)

    say("5/5  Building report")
    report_path = out / "report.html"
    report_path.write_text(
        build_report(work, praise, themes, scored, diag, Path(input_path).name, top_n, want_llm), encoding="utf-8"
    )

    if not quiet:
        cols = ["rank", "theme", "mentions", "accounts", "urgency", "momentum_pct", "opportunity_score"]
        if scored["rice"].notna().any():
            cols += ["effort_weeks", "rice"]
        with pd.option_context("display.width", 160, "display.max_colwidth", 42):
            print("\n" + ranked[cols].to_string(index=False))
        if len(scored) > len(ranked):
            print(f"\n({len(scored) - len(ranked)} mixed group(s) flagged for manual review in the report)")
        print(f"\nWrote: {report_path}\n       {opp_path}\n       {jira_path}")
        for p in prd_paths:
            print(f"       {p}")

    return {"report": report_path, "opportunities": opp_path, "jira": jira_path, "prds": prd_paths, "scored": scored}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="signal-to-spec",
        description="Turn raw customer feedback into ranked opportunities, a PRD draft and a Jira-ready backlog.",
    )
    p.add_argument("input", help="feedback CSV with columns: id, date, source, segment, account, seats, text")
    p.add_argument("-o", "--out", default="output", help="output folder (default: output)")
    p.add_argument("--review", help="optional team review CSV: match, name, effort_weeks (see data/team_review.csv)")
    p.add_argument("--k", type=int, help="number of themes (default: chosen automatically)")
    p.add_argument("--top", type=int, default=3, help="how many opportunities get a PRD and epics (default: 3)")
    p.add_argument("--llm", action="store_true", help="use Claude to name themes and draft user stories")
    args = p.parse_args(argv)
    try:
        run(args.input, args.out, args.review, args.k, args.top, args.llm)
    except FeedbackError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0
