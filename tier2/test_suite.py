"""Tier 2 functional test suite.

FUNCTIONAL tests: each checks that one designed behaviour works. They are not a
performance benchmark; Tier 2's measured performance comes from the 318-article
held-out test set (6_Tier2_Final_System.ipynb).

Every article is a real HELD-OUT test article, chosen by a fixed rule (a seeded
random draw), never by looking at the model's output. Edited articles change one
thing at a time, inserted after the third sentence so the 512-token model
certainly reads it. The pass rules below were fixed before the first run, and
failing cases are kept.

Capabilities:
  A  well-reported articles      5-star articles must be judged "meets standards"
  B  poorly reported articles    0-1-star articles must be judged "falls short"
  C  criterion added             inserting one missing element (a cost, a harm
                                 figure, an independent expert...) must lower THAT
                                 criterion's P(not met) by at least 0.05 AND by more
                                 than the median criterion (it must be specific).
                                 Also reported, descriptively: the targeted
                                 criterion's rank among the ten by how far it fell
  D  criterion removed           deleting the sentences that discuss costs or
                                 harms must raise that criterion's P(not met) by 0.05
  E  sensational language        inserting hype must lower the credibility score and
                                 raise P(not met) for "no sensational language"
  F  caveats                     inserting an honest limitations sentence into a
                                 poorly reported article must raise the credibility score
  G  formatting invariance       removing line breaks, or the title, must not change
                                 the verdict or move the score by 0.05 or more
  H  routing                     claims go to Tier 1 and articles to Tier 2; a
                                 102-word claim from the Tier 1 data is included

Usage:
    python tier2/test_suite.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from pipeline import route
from tier2.engine import CRITERIA, DISPLAY, Tier2Engine, article_text, sentence_spans

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
C_MIN = 0.05          # minimum movement of a targeted criterion
G_MAX = 0.05          # maximum score change allowed by a formatting change

CAPABILITY = {
    "A": "Well-reported articles", "B": "Poorly reported articles", "C": "Criterion added",
    "D": "Criterion removed", "E": "Sensational language", "F": "Caveats",
    "G": "Formatting invariance", "H": "Routing",
}

# One sentence per criterion, written generically so it fits any treatment story.
ADD = {
    "costs": "The treatment costs about $1,200 a month, and most insurance plans do not yet cover it.",
    "harms": "In the trial, 12 percent of patients reported nausea and 3 percent stopped taking the drug because of side effects.",
    "benefits": "Symptoms improved in 45 percent of patients given the drug, compared with 30 percent of those given a placebo.",
    "independent_srcs": ("Dr. Maria Lopez, a specialist at another university who was not involved in the study, "
                         "said the results were promising but preliminary."),
    "evidence_quality": ("The study was small, lasted only 12 weeks and did not follow patients for long, "
                         "so it cannot show whether the benefit lasts."),
    "availability": "The drug has not been approved by the FDA and is available only to patients enrolled in clinical trials.",
    "alternatives": ("Existing treatments already help most patients, and the new drug has not been compared "
                     "with them directly."),
    "novelty": "Similar drugs that act on the same pathway have been tested before with mixed results.",
}
HYPE = ("Experts are calling it a miracle breakthrough that could cure the disease for good. "
        "Patients will no longer have to suffer, because this revolutionary treatment works for everyone.")
CAVEAT = ("The researchers cautioned that the study was small and short, that it was funded by the "
          "manufacturer, and that larger independent trials are needed before anyone should change treatment.")
REMOVE = {"costs": r"\$|\bcosts?\b|\bprice|\binsur|\bexpensive|\bafford",
          "harms": r"side effect|adverse|\brisks?\b|\bharms?\b|complication|toxic"}

CLAIMS = [
    "Wearing face masks reduces the spread of COVID-19.",
    "Dexamethasone reduces deaths among patients hospitalised with severe COVID-19 who need oxygen.",
]


def insert(text: str, sentence: str, after: int = 3) -> str:
    spans = sentence_spans(text)
    cut = spans[min(after, len(spans)) - 1][1]
    return text[:cut] + " " + sentence + text[cut:]


def remove(text: str, pattern: str) -> tuple[str, int]:
    keep, n = [], 0
    for a, b in sentence_spans(text):
        if re.search(pattern, text[a:b], re.I):
            n += 1
        else:
            keep.append(text[a:b])
    return " ".join(keep), n


def main():
    t0 = time.time()
    te = pd.read_csv(ROOT / "datasets" / "processed_tier2" / "test.csv")
    te["full"] = [article_text(a, b) for a, b in zip(te.title, te.text)]
    eng = Tier2Engine()
    read_all = lambda t: all(s["read"] == "full" for s in eng.sentences(t))
    te["fully_read"] = [read_all(t) for t in te.full]
    rng = np.random.default_rng(0)
    pick = lambda df, n: df.iloc[rng.choice(len(df), n, replace=False)] if len(df) > n else df

    good = pick(te[te.rating == 5], 8)
    bad = pick(te[te.rating <= 1], 8)
    # edit bases: fully read, poorly reported, and failing the criteria we add
    base_pool = te[te.fully_read & (te.rating <= 2) & (te.fail_costs == 1) & (te.fail_harms == 1)
                   & (te.fail_independent_srcs == 1)]
    bases = pick(base_pool, 2)
    rem_pool = te[te.fully_read & (te.rating >= 4)]
    rows = []

    def rec(cid, cap, desc, expected, actual, passed, **kw):
        rows.append({"id": cid, "capability": CAPABILITY[cap], "case": desc, "expected": expected,
                     "actual": actual, "passed": bool(passed), **kw})

    # A, B: real articles against the reviewers' label
    for cap, df, exp in (("A", good, "CREDIBLE"), ("B", bad, "NOT_CREDIBLE")):
        for i, r in enumerate(df.itertuples(), 1):
            res = eng.assess(r.full)
            rec(f"{cap}{i}", cap, f"{r.article_id} ({r.rating}/5) {str(r.title)[:60]}", DISPLAY[exp],
                DISPLAY[res["verdict"]], res["verdict"] == exp, score=round(res["credibility"], 4))

    # C: add one element; the targeted criterion must fall, and more than the median one
    n = 0
    for r in bases.itertuples():
        before = eng.score([r.full])["criteria"][0]
        for k, sent in ADD.items():
            n += 1
            after = eng.score([insert(r.full, sent)])["criteria"][0]
            drop = before - after
            j = CRITERIA.index(k)
            ok = drop[j] >= C_MIN and drop[j] > np.median(drop)
            rank = int((drop > drop[j]).sum()) + 1
            rec(f"C{n}", "C", f"{r.article_id}: add {k}", f"{k} P(not met) falls >= {C_MIN}, more than median",
                f"falls {drop[j]:+.3f} (median {np.median(drop):+.3f}; rank {rank} of 10)", ok, rank=rank)

    # D: remove the sentences about costs / harms from well-reported articles
    n = 0
    for k, pat in REMOVE.items():
        pool = rem_pool[rem_pool[f"fail_{k}"] == 0]
        pool = pool[[remove(t, pat)[1] > 0 for t in pool.full]]
        for r in pick(pool, 3).itertuples():
            n += 1
            edited, m = remove(r.full, pat)
            j = CRITERIA.index(k)
            rise = eng.score([edited])["criteria"][0][j] - eng.score([r.full])["criteria"][0][j]
            rec(f"D{n}", "D", f"{r.article_id}: remove {m} {k} sentence(s)", f"{k} P(not met) rises >= {C_MIN}",
                f"rises {rise:+.3f}", rise >= C_MIN)

    # E: hype into well-reported articles; F: caveats into poorly reported ones
    for i, r in enumerate(good.head(3).itertuples(), 1):
        a, b = eng.score([r.full]), eng.score([insert(r.full, HYPE)])
        j = CRITERIA.index("press_release")
        dc, dp = b["credibility"][0] - a["credibility"][0], b["criteria"][0][j] - a["criteria"][0][j]
        rec(f"E{i}", "E", f"{r.article_id}: insert hype", "score falls, sensational-language P(not met) rises",
            f"score {dc:+.3f}, P(not met) {dp:+.3f}", dc < 0 and dp > 0)
    for i, r in enumerate(bad.head(3).itertuples(), 1):
        dc = eng.score([insert(r.full, CAVEAT)])["credibility"][0] - eng.score([r.full])["credibility"][0]
        rec(f"F{i}", "F", f"{r.article_id}: insert caveats", "score rises", f"score {dc:+.3f}", dc > 0)

    # G: formatting
    n = 0
    for r in pd.concat([good.head(2), bad.head(2)]).itertuples():
        base = eng.assess(r.full)
        for kind, text in (("no line breaks", " ".join(r.full.split())), ("no title", article_text("", r.text))):
            n += 1
            res = eng.assess(text)
            d = res["credibility"] - base["credibility"]
            rec(f"G{n}", "G", f"{r.article_id}: {kind}", f"same verdict, |change| < {G_MAX}",
                f"{DISPLAY[res['verdict']]}, change {d:+.3f}",
                res["verdict"] == base["verdict"] and abs(d) < G_MAX)

    # H: routing
    t1 = pd.concat([pd.read_csv(ROOT / "datasets" / "processed_tier1" / f"{s}.csv")
                    for s in ("train", "validation", "test")]).drop_duplicates("claim").claim
    longest = t1.iloc[int(np.argmax(t1.str.split().str.len().values))]
    route_cases = [(c, "tier1") for c in CLAIMS] + [(longest, "tier1")] + \
                  [(good.iloc[0].full, "tier2"), (min(te.full, key=lambda t: len(t.split())), "tier2")]
    for i, (text, exp) in enumerate(route_cases, 1):
        got = route(text)
        rec(f"H{i}", "H", f"{len(text.split())} words: {text[:50]}", exp, got, got == exp)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tier2_test_suite.csv", index=False)
    order = list(CAPABILITY.values())
    summary = (df.groupby("capability").agg(cases=("passed", "size"), passed=("passed", "sum"))
                 .reindex(order).reset_index())
    meta = {"n_cases": int(len(df)), "n_passed": int(df.passed.sum()),
            "pass_rate": round(float(df.passed.mean()), 4),
            "per_capability": summary.to_dict(orient="records"),
            "failures": df[~df.passed][["id", "capability", "case", "expected", "actual"]].to_dict("records"),
            # descriptive, NOT a pass rule: was the targeted criterion the one that moved most?
            "criterion_added_rank": {"cases": int(df["rank"].notna().sum()),
                                     "rank_1": int((df["rank"] == 1).sum()),
                                     "rank_top3": int((df["rank"] <= 3).sum())},
            "note": "Real held-out articles chosen by seeded random draw; edits change one thing; pass rules "
                    "fixed before the first run; failing cases kept.",
            "runtime_seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(OUT / "tier2_test_suite.json", "w"), indent=2)

    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 70)
    print(df[["id", "capability", "case", "actual", "passed"]].to_string(index=False))
    print("\n" + summary.to_string(index=False))
    rk = meta["criterion_added_rank"]
    print(f"\n{meta['n_passed']}/{meta['n_cases']} passed ({meta['pass_rate']:.0%})  ({meta['runtime_seconds']}s)")
    print(f"criterion added (descriptive): the targeted criterion moved most in {rk['rank_1']}/{rk['cases']} "
          f"cases, and was in the top three in {rk['rank_top3']}/{rk['cases']}")


if __name__ == "__main__":
    main()
