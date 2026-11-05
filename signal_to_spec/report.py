"""Self-contained HTML report: ranked opportunities, evidence and method, in one file."""

from __future__ import annotations

from datetime import date
from html import escape

import pandas as pd

from .themes import Theme

SEGMENT_ORDER = ["Free", "Standard", "Premium", "Enterprise"]

CSS = """
:root{color-scheme:light;--bg:#f6f5f2;--surface:#fcfcfb;--line:#e4e2dc;--text:#0b0b0b;--text-2:#52514e;
--muted:#7b7a75;--accent:#2a78d6;--accent-soft:#cde2fb;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;
--up:#0ca30c;--down:#d03b3b;--chip:#efeee9}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#111110;--surface:#1a1a19;
--line:#2e2e2b;--text:#f0efec;--text-2:#c3c2b7;--muted:#8f8e86;--accent:#3987e5;--accent-soft:#184f95;--s1:#3987e5;
--s2:#d95926;--s3:#199e70;--s4:#c98500;--chip:#262624}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#111110;--surface:#1a1a19;--line:#2e2e2b;--text:#f0efec;--text-2:#c3c2b7;
--muted:#8f8e86;--accent:#3987e5;--accent-soft:#184f95;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--chip:#262624}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif}
main{max-width:1080px;margin:0 auto;padding:40px 16px 64px}
h1{font-size:28px;margin:0 0 4px;letter-spacing:-.01em}h2{font-size:19px;margin:44px 0 12px}
h3{font-size:16px;margin:0}.sub{color:var(--text-2);margin:0}.muted{color:var(--muted)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:24px}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi b{display:block;font-size:24px;font-variant-numeric:tabular-nums}.kpi b.text{font-size:17px;line-height:1.35}.kpi span{color:var(--text-2);font-size:13px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin-bottom:12px}
.tablewrap{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 10px;text-align:left;border-bottom:1px solid var(--line);
vertical-align:middle}th{font-size:12px;color:var(--text-2);font-weight:600;white-space:nowrap}tr:last-child td{border-bottom:0}
td.num{font-variant-numeric:tabular-nums;text-align:right}th.num{text-align:right}
td.theme{min-width:210px}.kw{display:block;color:var(--muted);font-size:12px}.rank{font-weight:700;color:var(--text-2)}
.bar{position:relative;height:10px;min-width:72px;width:100%;background:var(--chip);border-radius:4px}
.bar i{position:absolute;inset:0 auto 0 0;background:var(--accent);border-radius:0 4px 4px 0}
.score{display:flex;align-items:center;gap:8px}.score b{font-variant-numeric:tabular-nums;min-width:36px;text-align:right}
.trend{display:flex;align-items:center;gap:6px;white-space:nowrap}.up{color:var(--up)}.down{color:var(--down)}
.spark polyline{fill:none;stroke:var(--accent);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.spark circle{fill:var(--accent)}
.chip{display:inline-block;background:var(--chip);border-radius:999px;padding:1px 9px;font-size:12px;color:var(--text-2);margin-right:4px}
blockquote{margin:12px 0 0;padding:2px 0 2px 14px;border-left:3px solid var(--accent-soft);color:var(--text)}
blockquote footer{color:var(--muted);font-size:12px;margin-top:2px}
.mix{display:flex;gap:2px;height:12px;margin:14px 0 6px}.mix span{border-radius:3px;min-width:3px}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:var(--text-2)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.method li{margin-bottom:6px}code{background:var(--chip);border-radius:4px;padding:1px 5px;font-size:13px}
.flag{border-left:3px solid var(--s4)}
"""


def _sparkline(counts: list[int], width: int = 96, height: int = 26) -> str:
    if not counts:
        return ""
    top = max(max(counts), 1)
    step = width / max(len(counts) - 1, 1)
    pts = [(round(i * step, 1), round(height - 3 - (c / top) * (height - 6), 1)) for i, c in enumerate(counts)]
    poly = " ".join(f"{x},{y}" for x, y in pts)
    lx, ly = pts[-1]
    label = escape("Weekly mentions: " + ", ".join(map(str, counts)))
    return (
        f'<svg class="spark" width="{width + 4}" height="{height}" viewBox="-2 0 {width + 4} {height}" role="img" '
        f'aria-label="{label}"><title>{label}</title><polyline points="{poly}"/><circle cx="{lx}" cy="{ly}" r="2.5"/></svg>'
    )


def _trend(row: pd.Series) -> str:
    pct = row["momentum_pct"]
    if not row["prior_4w"]:
        txt, cls = "new", "up"
    else:
        txt, cls = f"{pct:+d}%", ("up" if pct > 10 else "down" if pct < -10 else "muted")
    arrow = "↑" if cls == "up" else "↓" if cls == "down" else "→"
    return f'<div class="trend">{_sparkline(row["weekly"])}<span class="{cls}">{arrow} {txt}</span></div>'


def _mix_bar(mix: dict) -> str:
    total = sum(mix.values()) or 1
    segs, legend = [], []
    for i, seg in enumerate(SEGMENT_ORDER + sorted(set(mix) - set(SEGMENT_ORDER))):
        n = mix.get(seg, 0)
        color = f"var(--s{i + 1})" if i < 4 else "var(--muted)"
        if n:
            segs.append(f'<span style="flex:{n};background:{color}" title="{escape(seg)}: {n} mentions"></span>')
            legend.append(f'<span><i style="background:{color}"></i>{escape(seg)} {n / total:.0%}</span>')
    return f'<div class="mix">{"".join(segs)}</div><div class="legend">{"".join(legend)}</div>'


def _quotes(df: pd.DataFrame, idx: list[int]) -> str:
    out = []
    for _, r in df.iloc[idx].iterrows():
        out.append(
            f"<blockquote>“{escape(r['clean_text'])}”<footer>{escape(r['segment'])} customer · "
            f"{escape(r['source'])} · {r['date']:%b %d}</footer></blockquote>"
        )
    return "".join(out)


def build_report(
    df: pd.DataFrame,
    praise: pd.DataFrame,
    themes: list[Theme],
    scored: pd.DataFrame,
    diagnostics: dict,
    dataset_name: str,
    top_n: int,
    used_llm: bool,
) -> str:
    by_id = {t.theme_id: t for t in themes}
    ranked = scored[~scored["needs_review"]]
    mixed = scored[scored["needs_review"]]
    has_rice = ranked["rice"].notna().any()
    max_score = float(ranked["rice" if has_rice else "opportunity_score"].max() or 1)
    all_rows = pd.concat([df, praise])
    top = ranked.iloc[0] if len(ranked) else None

    kpis = [
        (f"{len(all_rows):,}", "feedback items analyzed"),
        (f"{all_rows['account'].nunique():,}", "customer accounts"),
        (f"{len(ranked)}", "themes ranked"),
        (escape(top["theme"]) if top is not None else "—", "top opportunity"),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><b{" class=text" if i == 3 else ""}>{v}</b><span>{label}</span></div>'
        for i, (v, label) in enumerate(kpis)
    )

    head = (
        "<tr><th>#</th><th>Theme</th><th class='num'>Mentions</th><th class='num'>Accounts</th>"
        "<th class='num'>Seats</th><th>Trend (12 wks)</th><th class='num'>Urgency</th>"
        "<th class='num'>Confidence</th>"
        + ("<th class='num'>Effort (wks)</th><th>RICE score</th>" if has_rice else "<th>Opportunity score</th>")
        + "</tr>"
    )
    body = []
    for _, r in ranked.iterrows():
        val = r["rice"] if has_rice else r["opportunity_score"]
        width = 0 if pd.isna(val) else max(2, 100 * float(val) / max_score)
        val_txt = "—" if pd.isna(val) else f"{val:g}"
        t = by_id[r["theme_id"]]
        sub = escape(r["keywords"]) if t.name else ""
        body.append(
            f"<tr><td class='rank'>{r['rank']}</td><td class='theme'><b>{escape(r['theme'])}</b>"
            f"{f'<span class=kw>{sub}</span>' if sub else ''}</td>"
            f"<td class='num'>{r['mentions']}</td><td class='num'>{r['accounts']}</td>"
            f"<td class='num'>{r['seats_affected']:,}</td><td>{_trend(r)}</td>"
            f"<td class='num'>{r['urgency']:.0%}</td><td class='num'>{r['confidence']:.0%}</td>"
            + (f"<td class='num'>{'—' if pd.isna(r['effort_weeks']) else f'{r.effort_weeks:g}'}</td>" if has_rice else "")
            + f"<td><div class='score'><b>{val_txt}</b><div class='bar' title='{val_txt}'><i style='width:{width:.0f}%'></i></div></div></td></tr>"
        )

    cards = []
    for _, r in ranked.head(top_n).iterrows():
        t = by_id[r["theme_id"]]
        summary = escape(t.problem) if t.problem else (
            f"{r['accounts']} accounts ({r['seats_affected']:,} seats) raised this {r['mentions']} times; "
            f"{r['urgency']:.0%} of mentions signal urgency."
        )
        cards.append(
            f"<div class='card'><h3>#{r['rank']} · {escape(r['theme'])}</h3><p class='sub'>{summary}</p>"
            f"{_mix_bar(r['segment_mix'])}{_quotes(df, t.representative_index)}"
            f"<p class='muted' style='margin:12px 0 0;font-size:13px'>Reach {r['reach']:g} · Impact {r['impact']:g} "
            f"· Confidence {r['confidence']:.0%} · Channels: {escape(', '.join(r['source_mix']))}</p></div>"
        )

    review = ""
    if len(mixed):
        items = []
        for _, r in mixed.iterrows():
            t = by_id[r["theme_id"]]
            items.append(
                f"<div class='card flag'><h3>{escape(r['theme'])} <span class='chip'>{r['mentions']} items</span></h3>"
                f"<p class='sub'>Only {t.coherence:.0%} of these items share the group's keywords, so it is probably "
                f"several smaller issues. Not ranked; worth a manual read.</p>{_quotes(df, t.representative_index)}</div>"
            )
        review = "<h2>Needs a human read</h2>" + "".join(items)

    praise_html = ""
    if len(praise):
        sample = praise.sort_values("sentiment", ascending=False).drop_duplicates("clean_text").head(4)
        praise_html = (
            f"<h2>What's working</h2><div class='card'><p class='sub'>{len(praise)} items were clear praise and are "
            f"kept out of the problem ranking.</p>"
            + "".join(
                f"<blockquote>“{escape(r['clean_text'])}”<footer>{escape(r['segment'])} customer · "
                f"{escape(r['source'])}</footer></blockquote>"
                for _, r in sample.iterrows()
            )
            + "</div>"
        )

    k_note = (
        f"k = {diagnostics['k']} chosen by silhouette score" if diagnostics.get("silhouette_by_k") else f"k = {diagnostics['k']} (set manually)"
    )
    method = f"""
<h2>How the ranking works</h2>
<div class="card method"><ol>
<li><b>Clean.</b> Note prefixes are stripped, and the same customer posting the same thing twice is counted once.
Sentences that only express urgency (“please prioritize this”) are kept for scoring but removed from theme detection.</li>
<li><b>Find themes.</b> TF-IDF on stemmed words and phrases, compressed with LSA, clustered with k-means ({k_note});
split themes that share defining words are merged back together. Clear praise is routed to “What's working”.</li>
<li><b>Reach</b> = accounts raising the theme, weighted by plan (Free 1, Standard 2, Premium 3, Enterprise 5).</li>
<li><b>Impact</b> (RICE scale 0.25–3) from pain (VADER sentiment) and urgency language: blocked work, churn risk, willingness to pay.</li>
<li><b>Confidence</b> (50/80/100%) from how many channels and plans corroborate it, and volume.</li>
<li><b>Opportunity score</b> = Reach × Impact × Confidence. With engineering estimates
(<code>--review</code>), the <b>RICE score</b> divides by effort in person-weeks, which is where the engineering view comes in.</li>
</ol><p class="muted" style="margin:0">Scores rank themes against each other within this dataset; they are a starting point for
discovery, not a substitute for talking to customers.</p></div>"""

    date_range = f"{all_rows['date'].min():%b %d} – {all_rows['date'].max():%b %d, %Y}"
    llm_note = " · theme names drafted with Claude" if used_llm else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Opportunity Report</title><style>{CSS}</style></head>
<body><main>
<h1>Customer opportunity report</h1>
<p class="sub">{escape(dataset_name)} · {date_range} · generated {date.today():%b %d, %Y} by signal-to-spec{llm_note}</p>
<div class="kpis">{kpi_html}</div>
<h2>Priority stack</h2>
<div class="tablewrap"><table><thead>{head}</thead><tbody>{''.join(body)}</tbody></table></div>
<h2>Top opportunities, with evidence</h2>
<div class="grid2">{''.join(cards)}</div>
{review}
{praise_html}
{method}
</main></body></html>
"""
