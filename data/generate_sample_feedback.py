"""Generate a synthetic customer-feedback dataset for a fictional product, "Relay".

Relay is an invented team work-tracking tool. Every account, quote and number in
the output is synthetic; it exists so the pipeline can be run and demoed without
real customer data. The generator is seeded, so the CSV is identical on every run.

Usage:
    python data/generate_sample_feedback.py            # writes data/sample_feedback.csv
    python data/generate_sample_feedback.py out.csv    # custom path
"""

from __future__ import annotations

import csv
import random
import sys
from datetime import date, timedelta
from pathlib import Path

SEED = 42
START = date(2026, 8, 10)  # Monday; 12 weeks of feedback
WEEKS = 12

SEGMENT_SEATS = {
    "Free": (2, 10),
    "Standard": (10, 120),
    "Premium": (60, 600),
    "Enterprise": (800, 5000),
}

SOURCES = ["Support ticket", "App review", "Sales call", "NPS comment", "Community forum", "Customer interview"]

# Each theme: phrase pool, segment weights, source weights, total volume, and a
# time profile (how volume shifts across the 12 weeks).
THEMES = {
    "performance": {
        "count": 42,
        "trend": "rising",
        "segments": {"Free": 1, "Standard": 3, "Premium": 4, "Enterprise": 4},
        "sources": {"Support ticket": 5, "App review": 2, "Sales call": 2, "NPS comment": 3, "Community forum": 2, "Customer interview": 1},
        "phrases": [
            "Boards with more than 500 issues take forever to load, sometimes 20+ seconds.",
            "The board view is painfully slow once a project gets big.",
            "Loading our main project board freezes the browser tab for several seconds.",
            "Drag and drop on the board lags badly on large projects.",
            "Our roadmap board times out when we filter by assignee.",
            "Every morning standup starts with waiting for the board to load.",
            "Board performance on big projects has gotten worse since the last update.",
            "Switching between boards is slow, the spinner just keeps spinning.",
            "Large backlogs are basically unusable, scrolling stutters and cards load late.",
            "We split one project into three just because the board was too slow to load.",
            "The board loads slowly for our India team, they wait ages every morning.",
            "Filtering a board with a few thousand cards hangs for 10 seconds.",
            "Backlog load time is the number one complaint from my engineers.",
            "Slow board performance is pushing the team back to spreadsheets.",
        ],
    },
    "notifications": {
        "count": 36,
        "trend": "flat",
        "segments": {"Free": 3, "Standard": 3, "Premium": 2, "Enterprise": 2},
        "sources": {"Support ticket": 3, "App review": 3, "Sales call": 1, "NPS comment": 4, "Community forum": 3, "Customer interview": 1},
        "phrases": [
            "I get an email notification for every single comment, it's way too noisy.",
            "There's no way to mute notifications for a single project.",
            "Notification overload: I'm auto-watching 40 tasks and can't turn it off.",
            "Please let me batch notifications into a daily digest instead of instant emails.",
            "My inbox is flooded with Relay notifications, I've started ignoring all of them.",
            "Notification settings are buried and too coarse, it's all or nothing.",
            "I missed an important mention because it was buried in 200 other notifications.",
            "Being auto-watched on every task I touch creates endless notification noise.",
            "Mobile push notifications fire for trivial updates like label changes.",
            "We need per-channel notification controls for email, in-app and Slack.",
            "Teammates turned notifications off entirely, so now they miss real mentions.",
            "Notification spam is the most common complaint in our internal survey.",
        ],
    },
    "ai_summaries": {
        "count": 40,
        "trend": "surging",
        "segments": {"Free": 1, "Standard": 3, "Premium": 3, "Enterprise": 3},
        "sources": {"Support ticket": 1, "App review": 2, "Sales call": 4, "NPS comment": 2, "Community forum": 3, "Customer interview": 3},
        "phrases": [
            "Coming back from vacation, I need a summary of what changed on my tasks.",
            "Long comment threads are impossible to catch up on, an AI summary would save hours.",
            "Would love an automatic summary of a ticket's history before standup.",
            "Can Relay summarize a project's weekly progress so I don't have to write status updates?",
            "Our managers want a quick AI summary of blockers across all projects.",
            "Other tools now offer AI summaries of threads, we keep asking when Relay will.",
            "Writing the weekly status update takes me an hour, please generate a draft summary.",
            "Catching up on a 150-comment thread is painful, give me an AI summary of the key decisions.",
            "We'd pay extra for AI that summarizes what happened in a sprint and flags risks.",
            "New joiners can't get context fast, an AI summary of each epic's history would help.",
            "I want to ask 'what's blocking the launch' and get an AI summary back.",
            "An AI catch-up summary is the top request from our leadership team.",
            "Our execs saw AI summary features in a competitor demo and asked why Relay has none.",
            "A one-click AI summary of the decisions made in a thread would be a killer feature.",
        ],
    },
    "permissions": {
        "count": 26,
        "trend": "flat",
        "segments": {"Free": 0, "Standard": 1, "Premium": 3, "Enterprise": 7},
        "sources": {"Support ticket": 2, "App review": 0, "Sales call": 5, "NPS comment": 1, "Community forum": 1, "Customer interview": 3},
        "phrases": [
            "We need guest access for contractors limited to one project.",
            "Permissions are too coarse, we can't hide a confidential project from the whole company.",
            "Our security team blocked the rollout because we can't restrict access to HR projects.",
            "We need role-based permissions, not just admin and member.",
            "We can't give clients read-only access to a board without exposing every project.",
            "Audit requirements mean we must control and log who can access each project.",
            "Granular project permissions are a dealbreaker for our enterprise renewal.",
            "External agency users can see our internal projects, that's a compliance problem.",
            "We want custom permission roles, for example can comment but not edit.",
            "Restricting project access by team instead of by individual user would save admins hours.",
            "Managing permissions for 3,000 users is impossible without groups.",
        ],
    },
    "search": {
        "count": 26,
        "trend": "flat",
        "segments": {"Free": 2, "Standard": 3, "Premium": 3, "Enterprise": 2},
        "sources": {"Support ticket": 3, "App review": 2, "Sales call": 1, "NPS comment": 3, "Community forum": 3, "Customer interview": 1},
        "phrases": [
            "Search can't find tasks I know exist unless I type the exact title.",
            "Searching old documents is useless, the search results are not relevant.",
            "I can never find last quarter's decision docs through search.",
            "Search doesn't look inside comments or attachments.",
            "We need search filters by date, project and owner.",
            "Search results are ordered randomly, what I need is always on page 3.",
            "A typo in search returns nothing, search should be more forgiving.",
            "We recreate documents because finding the original through search takes too long.",
            "Searching across projects is slow and the search misses archived items.",
            "Search should understand questions, not just exact keywords.",
        ],
    },
    "integrations": {
        "count": 24,
        "trend": "flat",
        "segments": {"Free": 1, "Standard": 4, "Premium": 3, "Enterprise": 1},
        "sources": {"Support ticket": 5, "App review": 1, "Sales call": 1, "NPS comment": 1, "Community forum": 3, "Customer interview": 1},
        "phrases": [
            "The Slack integration stopped posting updates to our channel again.",
            "GitHub pull requests don't link to tasks reliably, the GitHub sync keeps breaking.",
            "Slack messages from the Relay integration arrive hours late.",
            "When a PR merges in GitHub the task should move to done, but the integration fails.",
            "We had to reconnect the GitHub integration three times this month.",
            "Slack link previews for Relay tasks are broken in our workspace.",
            "There is no GitLab integration, we can't switch our engineers until it exists.",
            "The Google Calendar sync for due dates keeps duplicating events.",
            "The webhook integration drops events when we push a lot of updates.",
            "Our developers live in GitHub, if the sync isn't reliable they won't update tasks.",
        ],
    },
    "mobile": {
        "count": 20,
        "trend": "flat",
        "segments": {"Free": 4, "Standard": 3, "Premium": 1, "Enterprise": 1},
        "sources": {"Support ticket": 2, "App review": 6, "Sales call": 0, "NPS comment": 2, "Community forum": 2, "Customer interview": 1},
        "phrases": [
            "The mobile app can't edit task descriptions, only comment.",
            "The mobile app crashes when I open a large project.",
            "No offline mode in the mobile app makes it useless on flights.",
            "The mobile app is read-only for most things, I need to update status on the go.",
            "Attachments fail to upload from the mobile app.",
            "The Android app is much slower than the iPhone app.",
            "Our field teams need to tick off checklists on the mobile app without signal.",
            "The mobile app logs me out every few days.",
        ],
    },
    "reporting": {
        "count": 20,
        "trend": "flat",
        "segments": {"Free": 0, "Standard": 2, "Premium": 4, "Enterprise": 4},
        "sources": {"Support ticket": 2, "App review": 0, "Sales call": 3, "NPS comment": 2, "Community forum": 2, "Customer interview": 3},
        "phrases": [
            "We need sprint reports we can export to PDF for leadership.",
            "The burndown report doesn't account for scope changes.",
            "Custom dashboards are too limited, we can't chart by custom fields.",
            "Exporting reports to CSV loses all the custom fields.",
            "Velocity reporting across multiple teams is missing.",
            "I spend hours building reports in spreadsheets because Relay reporting is weak.",
            "We need cycle time and lead time reports out of the box.",
            "Leadership wants a portfolio dashboard that reports across all projects.",
        ],
    },
    "onboarding": {
        "count": 18,
        "trend": "falling",
        "segments": {"Free": 6, "Standard": 3, "Premium": 1, "Enterprise": 0},
        "sources": {"Support ticket": 2, "App review": 4, "Sales call": 1, "NPS comment": 2, "Community forum": 3, "Customer interview": 2},
        "phrases": [
            "Onboarding was confusing, the setup has too many options for a new team.",
            "The onboarding templates don't match how small teams actually work.",
            "The onboarding tour skipped what we needed, like inviting teammates during setup.",
            "It took us a week of setup to understand workflows vs boards during onboarding.",
            "I wish onboarding offered setup templates for marketing teams, not just software teams.",
            "First-time setup asked too many onboarding questions before showing any value.",
            "Our non-technical teammates gave up halfway through onboarding setup.",
            "Importing from spreadsheets during onboarding setup failed silently.",
        ],
    },
    "praise": {
        "count": 22,
        "trend": "flat",
        "segments": {"Free": 3, "Standard": 3, "Premium": 2, "Enterprise": 2},
        "sources": {"Support ticket": 0, "App review": 5, "Sales call": 1, "NPS comment": 5, "Community forum": 2, "Customer interview": 2},
        "phrases": [
            "Love how easy it is to drag tasks between columns.",
            "The new timeline view is great, the team uses it every day.",
            "Relay keeps our whole team aligned, a big improvement over email.",
            "Support was super helpful when we migrated.",
            "The keyboard shortcuts are fantastic, a huge time saver.",
            "Clean interface, our designers actually enjoy using it.",
            "Automation rules saved us a lot of manual work.",
            "Great product overall, and the pricing is fair for small teams.",
        ],
    },
}

PAIN_SUFFIXES = [
    " This happens daily.",
    " It's blocking our rollout to the rest of the company.",
    " We're evaluating other tools because of this.",
    " Several people on my team have raised this.",
    " Not a dealbreaker, but annoying.",
    " Please prioritize this.",
    " Our renewal is coming up and this keeps coming up.",
    " Otherwise we like the product.",
]
REQUEST_SUFFIXES = [
    " Please prioritize this.",
    " Several people on my team have asked for this.",
    " We'd upgrade our plan for this.",
    " It would save us hours every week.",
]

COMPANY_A = ["Blue", "North", "Bright", "Silver", "Harbor", "Cedar", "Summit", "Copper", "Lumen", "Pine",
             "Atlas", "Coral", "Iron", "Maple", "Orbit", "Quartz", "River", "Stone", "Tidal", "Vector"]
COMPANY_B = ["fin Labs", "gate Health", "path Logistics", "line Studio", "wave Retail", "point Bank",
             "field Media", "works", "stack Software", "ridge Energy", "light Education", "craft Agency"]


def _weighted(rng: random.Random, weights: dict[str, float]) -> str:
    keys = [k for k, w in weights.items() if w > 0]
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def _week_weights(trend: str) -> list[float]:
    if trend == "rising":
        return [0.6 + 0.08 * w for w in range(WEEKS)]
    if trend == "surging":
        return [0.15 + 0.03 * w * w for w in range(WEEKS)]
    if trend == "falling":
        return [1.6 - 0.11 * w for w in range(WEEKS)]
    return [1.0] * WEEKS


def _build_accounts(rng: random.Random) -> dict[str, list[dict]]:
    names = rng.sample([a + b for a in COMPANY_A for b in COMPANY_B], 120)
    plan = {"Free": 40, "Standard": 38, "Premium": 26, "Enterprise": 16}
    accounts: dict[str, list[dict]] = {}
    i = 0
    for segment, n in plan.items():
        lo, hi = SEGMENT_SEATS[segment]
        accounts[segment] = [
            {"account": names[i + j], "segment": segment, "seats": rng.randint(lo, hi)} for j in range(n)
        ]
        i += n
    return accounts


def _voice(rng: random.Random, source: str, text: str) -> str:
    """Make sales notes and interview notes read like notes rather than direct quotes."""
    if source == "Sales call" and rng.random() < 0.6:
        return "Customer said: " + text[0].lower() + text[1:]
    if source == "Customer interview" and rng.random() < 0.5:
        return "Interview note: " + text
    return text


def generate() -> list[dict]:
    rng = random.Random(SEED)
    accounts = _build_accounts(rng)
    rows: list[dict] = []
    for theme, spec in THEMES.items():
        weeks = rng.choices(range(WEEKS), weights=_week_weights(spec["trend"]), k=spec["count"])
        for week in weeks:
            segment = _weighted(rng, spec["segments"])
            acct = rng.choice(accounts[segment])
            source = _weighted(rng, spec["sources"])
            text = rng.choice(spec["phrases"])
            if theme != "praise" and rng.random() < 0.45:
                pool = REQUEST_SUFFIXES if theme == "ai_summaries" else PAIN_SUFFIXES
                text += rng.choice(pool)
            day = START + timedelta(days=7 * week + rng.randint(0, 6))
            rows.append(
                {
                    "date": day.isoformat(),
                    "source": source,
                    "segment": segment,
                    "account": acct["account"],
                    "seats": acct["seats"],
                    "text": _voice(rng, source, text),
                    "_theme": theme,
                }
            )
    # A few tickets logged twice (same customer, two channels): the pipeline should dedupe these.
    for row in rng.sample(rows, 6):
        dup = dict(row)
        dup["source"] = "Support ticket" if row["source"] != "Support ticket" else "Community forum"
        rows.append(dup)

    rows.sort(key=lambda r: (r["date"], r["account"]))
    for n, row in enumerate(rows, start=1):
        row["id"] = f"FB-{n:04d}"
    return rows


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("sample_feedback.csv")
    rows = generate()
    fields = ["id", "date", "source", "segment", "account", "seats", "text"]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} synthetic feedback items to {out}")


if __name__ == "__main__":
    main()
