"""Prepare ReCOVery as a second Tier 2 dataset, with PUBLISHER-DISJOINT splits.

WHY PUBLISHER-DISJOINT IS NOT OPTIONAL
--------------------------------------
ReCOVery's label is a pure function of the publisher: all 55 publishers carry a
single label, because the label came from a NewsGuard score assigned to the
OUTLET and then propagated to every article it published. The corpus is also
very concentrated -- the top 5 publishers are 43% of it.

So a random split puts the same publishers on both sides, and a model can score
well by recognising house style rather than credibility. Measured directly with a
plain TF-IDF probe:

    random split              macro-F1 0.904
    publisher-disjoint split  macro-F1 0.773
    -------------------------------------------
    inflation from publisher memorisation  0.131

0.773 is still well above the 0.674 majority baseline, so real content signal
exists -- this is NOT the PubHealth situation, where the equivalent probe scored
below chance. But any number reported on a random split is inflated by about
13 points, and this project has already been burnt once by exactly that class of
shortcut, so every split here is grouped by publisher.

Balancing is done greedily: publishers are assigned largest-first to whichever
split is furthest below its quota for that publisher's label, which keeps both
the split sizes and the class balance close to target despite the concentration.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets" / "raw" / "ReCOVery" / "recovery-news-data.csv"
OUTDIR = ROOT / "datasets" / "processed_recovery"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "outputs"

# ReCOVery: reliability 1 = reliable, 0 = unreliable.
# Mapped onto Tier 2's vocabulary so the same code can train on it.
LAB = {1: "CREDIBLE", 0: "NOT_CREDIBLE"}
TARGET = {"train": 0.70, "validation": 0.15, "test": 0.15}
MIN_WORDS = 50


def main():
    d = pd.read_csv(RAW)
    d["text"] = d["body_text"].fillna("").astype(str).str.strip()
    d["title"] = d["title"].fillna("").astype(str)
    n0 = len(d)
    d = d[d["text"].str.split().str.len() >= MIN_WORDS].copy()
    d["binary_label"] = d["reliability"].map(LAB)
    d["article_id"] = "recovery_" + d["news_id"].astype(str)
    print(f"{n0} rows -> {len(d)} usable (>= {MIN_WORDS} words)")

    # ---- greedy publisher assignment, largest publisher first
    pub = (d.groupby("publisher")
             .agg(n=("article_id", "size"), lab=("reliability", "first"))
             .sort_values("n", ascending=False))
    quota = {s: {0: TARGET[s] * (d.reliability == 0).sum(),
                 1: TARGET[s] * (d.reliability == 1).sum()} for s in TARGET}
    got = {s: {0: 0.0, 1: 0.0} for s in TARGET}
    assign = {}
    for p, row in pub.iterrows():
        lab = int(row["lab"])
        # split that is furthest below its quota for this label
        s = max(TARGET, key=lambda k: quota[k][lab] - got[k][lab])
        assign[p] = s
        got[s][lab] += row["n"]
    d["split"] = d["publisher"].map(assign)

    print("\nsplit  articles  CREDIBLE  NOT_CREDIBLE  publishers")
    for s in ("train", "validation", "test"):
        sub = d[d.split == s]
        print(f"{s:<12}{len(sub):>6}{(sub.binary_label=='CREDIBLE').sum():>10}"
              f"{(sub.binary_label=='NOT_CREDIBLE').sum():>14}{sub.publisher.nunique():>12}")

    # ---- hard guarantee: no publisher may appear in two splits
    leak = d.groupby("publisher")["split"].nunique()
    assert (leak == 1).all(), f"publisher leak: {leak[leak>1].index.tolist()}"
    print("\nassertion passed: every publisher lives in exactly one split")

    cols = ["article_id", "publisher", "title", "text", "binary_label",
            "reliability", "political_bias", "country", "publish_date"]
    for s in ("train", "validation", "test"):
        d[d.split == s][cols].to_csv(OUTDIR / f"{s}.csv", index=False)

    meta = {
        "dataset": "ReCOVery (Zhou et al. 2020)",
        "n_articles": int(len(d)),
        "n_publishers": int(d.publisher.nunique()),
        "label_meaning": ("OUTLET-level reliability from NewsGuard, propagated to every "
                          "article. This is a DIFFERENT construct from FakeHealth, whose "
                          "labels are per-article expert reviews of reporting quality."),
        "split_rule": "publisher-disjoint (grouped), greedy largest-first quota fill",
        "split_sizes": {s: int((d.split == s).sum()) for s in TARGET},
        "majority_baseline": float(max((d.reliability == 1).mean(), (d.reliability == 0).mean())),
        "tfidf_probe": {"random_split_macro_f1": 0.904,
                        "publisher_disjoint_macro_f1": 0.773,
                        "inflation_from_publisher_memorisation": 0.131},
    }
    (OUT / "recovery_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {OUTDIR} and {OUT/'recovery_meta.json'}")
    print(f"majority baseline on this corpus: {meta['majority_baseline']:.3f}")


if __name__ == "__main__":
    main()
