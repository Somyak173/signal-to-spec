import csv
from pathlib import Path

import pandas as pd
import pytest

from signal_to_spec import llm
from signal_to_spec.cli import run
from signal_to_spec.ingest import FeedbackError, drop_near_duplicates, load_feedback
from signal_to_spec.scoring import add_sentiment

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample_feedback.csv"
REVIEW = ROOT / "data" / "team_review.csv"


def _write(tmp_path, rows, name="fb.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_missing_columns_are_reported(tmp_path):
    path = _write(tmp_path, [{"id": 1, "text": "hello"}])
    with pytest.raises(FeedbackError, match="Missing column"):
        load_feedback(path)


def test_duplicates_removed_only_within_the_same_account(tmp_path):
    base = {"date": "2026-09-01", "source": "Support ticket", "segment": "Free", "seats": 3}
    rows = [
        {**base, "id": 1, "account": "A", "text": "Search can't find tasks by title."},
        {**base, "id": 2, "account": "A", "text": "Search can't find tasks by title."},  # same customer, twice
        {**base, "id": 3, "account": "B", "text": "Search can't find tasks by title."},  # different customer
    ]
    df, removed = drop_near_duplicates(load_feedback(_write(tmp_path, rows)))
    assert removed == 1
    assert sorted(df["account"]) == ["A", "B"]


def test_praise_is_separated_from_requests(tmp_path):
    base = {"date": "2026-09-01", "source": "NPS comment", "segment": "Free", "account": "A", "seats": 3}
    rows = [
        {**base, "id": 1, "text": "Great product, the team loves the timeline view."},
        {**base, "id": 2, "text": "Would love an AI summary of long threads, it would be great."},
        {**base, "id": 3, "text": "The board is painfully slow and keeps crashing."},
    ]
    df = add_sentiment(load_feedback(_write(tmp_path, rows)))
    assert df["is_praise"].tolist() == [True, False, False]


def test_urgency_sentences_do_not_drive_themes(tmp_path):
    rows = [{"id": 1, "date": "2026-09-01", "source": "x", "segment": "Free", "account": "A", "seats": 1,
             "text": "Customer said: search misses old docs. Please prioritize this."}]
    df = load_feedback(_write(tmp_path, rows))
    assert df.loc[0, "clean_text"].startswith("search")
    assert "prioritize" not in df.loc[0, "topic_text"]


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    return run(SAMPLE, out, review_path=REVIEW, top_n=3, quiet=True)


def test_end_to_end_outputs_exist(result):
    for key in ("report", "opportunities", "jira"):
        assert Path(result[key]).exists()
    assert len(result["prds"]) == 3
    html = Path(result["report"]).read_text()
    assert "Priority stack" in html and "How the ranking works" in html


def test_known_themes_are_recovered(result):
    ranked = result["scored"][~result["scored"]["needs_review"]]
    assert len(ranked) >= 6
    keywords = " ".join(ranked["keywords"]).lower()
    for expected in ("notification", "permission", "search", "summary", "mobile"):
        assert expected in keywords, f"expected a theme about {expected}"


def test_review_file_sets_names_and_rice(result):
    ranked = result["scored"][~result["scored"]["needs_review"]]
    assert "Notification overload" in ranked["theme"].tolist()
    assert ranked["rice"].notna().all()
    # ranking follows RICE when effort is known
    assert ranked["rice"].is_monotonic_decreasing


def test_jira_csv_has_epics_with_children(result):
    with open(result["jira"], newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    types = [r[header.index("Issue Type")] for r in body]
    assert types.count("Epic") == 3
    epic_ids = {r[0] for r in body if r[header.index("Issue Type")] == "Epic"}
    children = [r for r in body if r[header.index("Issue Type")] != "Epic"]
    assert children and all(r[header.index("Parent Id")] in epic_ids for r in children)


def test_llm_layer_is_optional_and_fails_soft(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert llm.available() is False

    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(llm, "_ask", boom)
    from signal_to_spec.themes import Theme

    theme = Theme(theme_id=0, keywords=["search"], label="search", member_index=[0], representative_index=[0])
    df = pd.DataFrame({"clean_text": ["search is broken"]})
    llm.name_themes([theme], df)
    assert theme.name == ""  # unchanged; keyword label still used
    assert llm.draft_prd_sections(theme, {}, ["search is broken"]) is None
