"""Discover themes in feedback (TF-IDF -> LSA -> k-means) and label them in plain words."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import snowballstemmer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# Words that show up across all kinds of feedback and say nothing about the problem.
DOMAIN_STOP_WORDS = {
    "relay", "app", "please", "really", "just", "like", "want", "need", "team", "teams",
    "would", "love", "make", "use", "using", "every", "time", "way", "lot", "thing", "things",
    "people", "product", "actually", "keeps", "able", "get", "got", "ve", "don", "doesn", "isn",
    "won", "can", "t", "s", "much", "many", "us", "our", "we", "one", "also", "still",
}
STOP_WORDS = ENGLISH_STOP_WORDS.union(DOMAIN_STOP_WORDS)
RANDOM_STATE = 7
_STEMMER = snowballstemmer.stemmer("english")


class _StemAnalyzer:
    """Tokenise, drop stop words, stem, add bigrams; remember a readable form for each stem."""

    def __init__(self) -> None:
        self.surface: dict[str, Counter] = defaultdict(Counter)

    def __call__(self, text: str) -> list[str]:
        words = [w for w in re.findall(r"[a-z][a-z]+", text.lower()) if w not in STOP_WORDS]
        stems = _STEMMER.stemWords(words)
        for w, s in zip(words, stems):
            self.surface[s][w] += 1
        bigrams = [f"{a} {b}" for a, b in zip(stems, stems[1:])]
        for (w1, w2), bg in zip(zip(words, words[1:]), bigrams):
            self.surface[bg][f"{w1} {w2}"] += 1
        return stems + bigrams

    def readable(self, term: str) -> str:
        forms = self.surface.get(term)
        return forms.most_common(1)[0][0] if forms else term


@dataclass
class Theme:
    theme_id: int
    keywords: list[str]
    label: str
    member_index: list[int]
    representative_index: list[int]
    coherence: float = 1.0  # share of items that use one of the theme's keywords
    name: str = ""          # human-readable name (LLM-written when --llm is on)
    problem: str = ""       # one-line problem statement (LLM-written when --llm is on)
    extras: dict = field(default_factory=dict)

    MIN_COHERENCE = 0.75

    @property
    def display_name(self) -> str:
        return self.name or self.label

    @property
    def is_mixed(self) -> bool:
        return self.coherence < self.MIN_COHERENCE


def _top_terms(tfidf, members, terms, global_mean, top_n: int = 3) -> list[str]:
    """Top (stemmed) terms that are weighted heavily in this theme relative to the whole dataset."""
    centroid = np.asarray(tfidf[members].mean(axis=0)).ravel()
    ranked = terms[np.argsort(-(centroid - global_mean))]
    chosen: list[str] = []
    for term in ranked:
        # skip a term that repeats a word already covered by a chosen term
        if any(set(term.split()) & set(c.split()) for c in chosen):
            continue
        chosen.append(term)
        if len(chosen) == top_n:
            break
    return chosen


def _refine(tfidf, labels: np.ndarray, rounds: int = 5) -> np.ndarray:
    """Reassign each item to the theme whose actual vocabulary it matches best.

    LSA is good at finding the groups but blurs words that appear everywhere
    ("project", "board"), which leaves a catch-all cluster. A few passes of
    nearest-centroid in the full TF-IDF space move those items to their real theme.
    """
    labels = labels.copy()
    for _ in range(rounds):
        ids = np.unique(labels)
        centroids = normalize(np.vstack([np.asarray(tfidf[labels == c].mean(axis=0)) for c in ids]))
        sims = np.asarray(tfidf @ centroids.T)
        new = ids[sims.argmax(axis=1)]
        # items with no shared vocabulary at all keep their LSA assignment
        no_overlap = sims.max(axis=1) == 0
        new[no_overlap] = labels[no_overlap]
        if np.array_equal(new, labels):
            break
        labels = new
    return labels


def _merge_shared_keywords(
    tfidf, labels: np.ndarray, terms, global_mean, min_concentration: float = 0.7
) -> tuple[np.ndarray, int]:
    """Merge clusters that are two halves of one theme (e.g. "board · load" and "backlog · load").

    k-means is told how many groups to make, so it sometimes splits one real theme in two.
    Two clusters are merged when they share a defining word AND most uses of that word in
    the whole dataset fall inside those two clusters. A word used everywhere ("project"
    in a project-management tool) is shared by many clusters and proves nothing.
    """
    labels = labels.copy()
    present = (tfidf > 0).tocsc()
    column = {t: i for i, t in enumerate(terms)}
    merged = 0
    changed = True
    while changed:
        changed = False
        ids = sorted(set(labels))
        words = {c: set(" ".join(_top_terms(tfidf, np.where(labels == c)[0], terms, global_mean)).split()) for c in ids}
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                for word in words[a] & words[b]:
                    if word not in column:
                        continue
                    docs = present[:, column[word]].nonzero()[0]
                    share = np.isin(labels[docs], [a, b]).mean() if len(docs) else 0.0
                    if share >= min_concentration:
                        labels[labels == b] = a
                        merged += 1
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
    return labels, merged


def discover_themes(
    df: pd.DataFrame,
    k: int | None = None,
    k_range: tuple[int, int] = (6, 12),
    lsa_dims: int = 10,
    n_quotes: int = 3,
) -> tuple[list[Theme], dict]:
    """Cluster feedback into themes.

    Short feedback is sparse, so TF-IDF is compressed to a small number of latent
    "topics" (LSA) before k-means; this groups "board is slow" with "backlog takes
    forever to load". When k is not given, it is picked by silhouette score.
    """
    texts = df["topic_text"] if "topic_text" in df else df["clean_text"]
    analyzer = _StemAnalyzer()
    vectorizer = TfidfVectorizer(analyzer=analyzer, min_df=2, max_df=0.5, sublinear_tf=True)
    tfidf = vectorizer.fit_transform(texts)
    dims = max(2, min(lsa_dims, tfidf.shape[1] - 1, tfidf.shape[0] - 1))
    reduced = normalize(TruncatedSVD(n_components=dims, random_state=RANDOM_STATE).fit_transform(tfidf))

    scores: dict[int, float] = {}
    if k is None:
        lo, hi = k_range[0], min(k_range[1], len(df) - 1)
        for cand in range(lo, hi + 1):
            labels = KMeans(n_clusters=cand, n_init=20, random_state=RANDOM_STATE).fit_predict(reduced)
            scores[cand] = float(silhouette_score(reduced, labels, metric="cosine"))
        k = max(scores, key=scores.get)

    labels = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(reduced)
    labels = _refine(tfidf, labels)
    terms = vectorizer.get_feature_names_out()
    global_mean = np.asarray(tfidf.mean(axis=0)).ravel()
    labels, merged_a = _merge_shared_keywords(tfidf, labels, terms, global_mean)
    labels, merged_b = _merge_fragments(tfidf, labels, min_size=max(8, int(0.03 * len(df))))
    merged = merged_a + merged_b

    themes: list[Theme] = []
    for cid in sorted(set(labels)):
        members = np.where(labels == cid)[0]
        top = _top_terms(tfidf, members, terms, global_mean)
        keywords = [analyzer.readable(t) for t in top]
        # Coherence: share of items that actually use one of the theme's defining words.
        # Low coherence means k-means made a catch-all group; flag it for a human instead
        # of pretending it is one problem.
        stems = set(" ".join(top).split())
        coherence = float(np.mean([bool(stems & set(_STEMMER.stemWords(re.findall(r"[a-z]+", t.lower()))))
                                   for t in texts.iloc[members]]))
        # representative quotes: closest to the theme centre, one per account, no repeats
        center = normalize(reduced[members].mean(axis=0, keepdims=True))[0]
        sims = reduced[members] @ center
        reps: list[int] = []
        seen_accounts: set[str] = set()
        seen_text: set[str] = set()
        for pos in members[np.argsort(-sims)]:
            row = df.iloc[pos]
            key = row["topic_text"].lower() if "topic_text" in df else row["clean_text"].lower()
            if row["account"] in seen_accounts or key in seen_text:
                continue
            reps.append(int(pos))
            seen_accounts.add(row["account"])
            seen_text.add(key)
            if len(reps) == n_quotes:
                break
        themes.append(
            Theme(
                theme_id=len(themes),
                keywords=keywords,
                label=" · ".join(keywords),
                member_index=[int(m) for m in members],
                representative_index=reps,
                coherence=round(coherence, 2),
            )
        )
    return themes, {"k": k, "merged_fragments": merged, "silhouette_by_k": scores, "lsa_dims": dims}


def _merge_fragments(tfidf, labels: np.ndarray, min_size: int, min_overlap: float = 0.08) -> tuple[np.ndarray, int]:
    """Fold tiny clusters into the theme they share the most vocabulary with.

    k-means on short text sometimes splits one theme into a big cluster plus a small
    fragment (e.g. "notifications" and "inbox flooded with notifications"). A fragment
    is merged only if it actually overlaps with a larger theme; genuinely new, small
    themes are kept, because an emerging issue often starts small.
    """
    labels = labels.copy()
    ids, sizes = np.unique(labels, return_counts=True)
    big = [c for c, s in zip(ids, sizes) if s >= min_size]
    small = [c for c, s in zip(ids, sizes) if s < min_size]
    if not big:
        return labels, 0
    centroids = {c: normalize(np.asarray(tfidf[labels == c].mean(axis=0)))[0] for c in ids}
    merged = 0
    for frag in small:
        best = max(big, key=lambda c: float(centroids[frag] @ centroids[c]))
        if float(centroids[frag] @ centroids[best]) >= min_overlap:
            labels[labels == frag] = best
            merged += 1
    return labels, merged
