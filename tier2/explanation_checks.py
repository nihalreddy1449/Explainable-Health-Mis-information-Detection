"""Does Tier 2's sentence-level SHAP explanation point at the right sentences?

Two independent checks over the held-out test articles:

1. Agreement with the human reviewers. FakeHealth keeps each HealthNewsReview
   reviewer's written reason for every criterion (`why_<criterion>`), and those
   reasons often QUOTE the article. A sentence the reviewer quoted while judging a
   criterion is a human rationale for that criterion. We ask whether SHAP ranks
   those sentences above the others, compared with chance. This is the Tier 2
   counterpart of Tier 1's SciFact rationale test.

   A sentence counts as quoted when it shares a run of 6 consecutive words with the
   reviewer's text (after lower-casing and dropping punctuation, which also undoes
   the dataset's broken curly quotes: "don't" and "don�t" both become "don t").

2. Deletion test. Remove the 3 sentences SHAP says pushed the article most toward
   "falls short" and re-score it: the credibility score should rise, and by more
   than when 3 random sentences are removed. Symmetrically for the 3 sentences
   that pushed toward "meets standards".

SHAP is expensive (a few hundred ensemble forward passes per article), so the
per-article results are cached, one JSON line per article, and can be re-analysed
without recomputing:

    python -m tier2.explanation_checks              # compute (resumes if interrupted)
"""
from __future__ import annotations

import json
import zlib
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "outputs" / "tier2_explanations.jsonl"
K = 3            # sentences removed / inspected per article
N_RANDOM = 10    # random draws for the deletion baseline
NGRAM = 6


def norm_words(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).split()


def _grams(words, n=NGRAM):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def quoted_sentences(sentences: list[str], why: str) -> list[int]:
    """Indices of sentences sharing a 6-word run with the reviewer's text."""
    if not isinstance(why, str) or not why.strip():
        return []
    g = _grams(norm_words(why))
    return [i for i, s in enumerate(sentences) if _grams(norm_words(s)) & g]


def explain_article(eng, text: str, rng: np.random.Generator) -> dict:
    sents = eng.sentences(text)
    ex = eng.explain(text, sents)
    read = ex["sentences"]
    n = len(read)
    full = float(eng.score([" ".join(read)])["credibility"][0])
    out = {"n_read": n, "cred_window": full, **{k: ex[k] for k in
           ("sentences", "credibility", "criteria", "base_credibility", "max_evals")}}
    if n >= 2 * K:
        cred = np.asarray(ex["credibility"])
        order = np.argsort(cred)
        def drop(idx):
            keep = [s for i, s in enumerate(read) if i not in set(idx)]
            return " ".join(keep)
        rand = [rng.choice(n, K, replace=False) for _ in range(N_RANDOM)]
        texts = [drop(order[:K]), drop(order[::-1][:K])] + [drop(r) for r in rand]
        sc = eng.score(texts)["credibility"]
        out["deletion"] = {"drop_neg": float(sc[0] - full), "drop_pos": float(sc[1] - full),
                           "drop_random": [float(x - full) for x in sc[2:]]}
    return out


def compute(limit: int | None = None):
    from tier2.engine import Tier2Engine, article_text
    te = pd.read_csv(ROOT / "datasets" / "processed_tier2" / "test.csv")
    done = set()
    if CACHE.exists():
        done = {json.loads(l)["article_id"] for l in open(CACHE, encoding="utf-8")}
    eng = Tier2Engine()
    rows = te[~te.article_id.isin(done)]
    if limit:
        rows = rows.head(limit)
    print(f"{len(done)} cached, {len(rows)} to compute", flush=True)
    with open(CACHE, "a", encoding="utf-8") as f:
        for i, r in enumerate(rows.itertuples(), 1):
            t0 = time.time()
            rng = np.random.default_rng(zlib.crc32(r.article_id.encode()))
            res = explain_article(eng, article_text(r.title, r.text), rng)
            res.update({"article_id": r.article_id, "binary_label": r.binary_label,
                        "seconds": round(time.time() - t0, 1)})
            f.write(json.dumps(res) + "\n"); f.flush()
            print(f"  {i}/{len(rows)} {r.article_id} n_read={res['n_read']} "
                  f"{res['seconds']}s", flush=True)


def load() -> list[dict]:
    return [json.loads(l) for l in open(CACHE, encoding="utf-8")]


# ------------------------------------------------------------------ analysis
CRITERIA = ["costs", "benefits", "harms", "evidence_quality", "disease_mongering",
            "independent_srcs", "alternatives", "availability", "novelty", "press_release"]


def _rank_auc(scores, pos):
    """P(a quoted sentence outranks an unquoted one), ties count half."""
    s = np.asarray(scores); p = np.zeros(len(s), bool); p[pos] = True
    a, b = s[p], s[~p]
    if len(a) == 0 or len(b) == 0:
        return None
    return float(((a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum())
                 / (len(a) * len(b)))


def reviewer_agreement(results: list[dict], test: pd.DataFrame, min_read: int = 2 * K) -> pd.DataFrame:
    """One row per (article, criterion) where the reviewer quoted at least one read
    sentence (and not all of them)."""
    why = test.set_index("article_id")
    rows = []
    for r in results:
        if r["n_read"] < min_read:
            continue
        for k in CRITERIA:
            if k == "disease_mongering":
                continue
            q = quoted_sentences(r["sentences"], why.at[r["article_id"], f"why_{k}"])
            if not q or len(q) == r["n_read"]:
                continue
            mag = np.abs(np.asarray(r["criteria"][k]))
            top = np.argsort(-mag)[:K]
            rows.append({"article_id": r["article_id"], "criterion": k, "n_read": r["n_read"],
                         "n_quoted": len(q),
                         "precision_at_k": float(np.isin(top, q).mean()),
                         "chance_at_k": len(q) / r["n_read"],
                         "rank_auc": _rank_auc(mag, q)})
    return pd.DataFrame(rows)


def verdict_agreement(results: list[dict], test: pd.DataFrame, min_read: int = 2 * K) -> pd.DataFrame:
    """Verdict level: rationale = every read sentence the reviewer quoted under ANY
    criterion; ranking = |SHAP| of the credibility score."""
    why = test.set_index("article_id")
    rows = []
    for r in results:
        if r["n_read"] < min_read:
            continue
        q = sorted(set().union(*[quoted_sentences(r["sentences"], why.at[r["article_id"], f"why_{k}"])
                                 for k in CRITERIA]))
        if not q or len(q) == r["n_read"]:
            continue
        mag = np.abs(np.asarray(r["credibility"]))
        top = np.argsort(-mag)[:K]
        rows.append({"article_id": r["article_id"], "n_read": r["n_read"], "n_quoted": len(q),
                     "precision_at_k": float(np.isin(top, q).mean()), "chance_at_k": len(q) / r["n_read"],
                     "rank_auc": _rank_auc(mag, q)})
    return pd.DataFrame(rows)


def deletion_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        d = r.get("deletion")
        if not d:
            continue
        rnd = np.asarray(d["drop_random"])
        rows.append({"article_id": r["article_id"], "binary_label": r["binary_label"],
                     "drop_neg": d["drop_neg"], "drop_pos": d["drop_pos"],
                     "random_mean": float(rnd.mean()), "random_abs_mean": float(np.abs(rnd).mean())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    compute(int(sys.argv[1]) if len(sys.argv) > 1 else None)
