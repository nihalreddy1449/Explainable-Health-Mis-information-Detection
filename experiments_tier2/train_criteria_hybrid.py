"""Can the ten criteria be told apart better? A per-criterion hybrid of v2 and TF-IDF.

WHY THIS EXISTS
---------------
Integration of v2 exposed a problem the headline numbers hide. The ten criteria
heads share one [CLS] vector, and their predictions collapse onto a single
"overall quality" factor:

    mean pairwise correlation of predicted probabilities   0.936
    same, for the reviewers' actual labels                  0.208
    variance explained by the first principal component    0.947

Within an article, ranking the failed criteria above the met ones works no better
than using each criterion's training base rate (0.779 vs 0.787). So the criteria
list shown to users was mostly "how bad is the article overall" plus "which
criteria usually fail".

Independent per-criterion TF-IDF + logistic regression models do NOT collapse
(pairwise correlation 0.456) and beat v2 on 7 of 9 criteria by AUC. This script
tests a per-criterion blend of the two.

PROTOCOL (fixed before looking at test)
---------------------------------------
* TF-IDF: the canonical configuration (50k features, 1-2 grams, sublinear tf),
  fitted on training text only. One logistic regression per criterion on the
  articles graded for it; C chosen per criterion by validation AUC.
* Blend per criterion: w * z(TF-IDF logit) + (1 - w) * z(v2 logit), each logit
  standardised with validation statistics, w in {0, 0.1, ..., 1} chosen by
  validation AUC. w = 0 is plain v2, w = 1 is plain TF-IDF.
* Thresholds: rate-matched on validation at each criterion's training fail rate.
* Verdict: a composer (logistic regression) over the ten blended criteria, fitted
  on validation, threshold rate-matched on validation, compared with v2's own
  composer. Validation-side comparisons use 5-fold cross-validation WITHIN
  validation, so the composer is never scored on articles it was fitted on.
* Test is evaluated once, with a paired bootstrap for the differences.

Usage:
    python experiments_tier2/train_criteria_hybrid.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tier2.engine import CRITERIA, article_text  # noqa: E402

DATA, OUT, PRED = ROOT / "datasets" / "processed_tier2", ROOT / "outputs", ROOT / "outputs" / "tier2_preds"
MODEL_DIR = ROOT / "models" / "tier2_v2_ensemble"
SEEDS = [42, 1337, 2024, 7, 99]
C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
W_GRID = np.round(np.linspace(0, 1, 11), 2)
SHOWN = [k for k in CRITERIA if k != "disease_mongering"]


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def masked_auc(y, s):
    m = ~np.isnan(y)
    return roc_auc_score(y[m].astype(int), s[m])


def within_article_auc(truth, score):
    """Per article: P(a failed criterion outranks a met one); mean over articles."""
    out = []
    for t, p in zip(truth, score):
        m = ~np.isnan(t)
        a, b = p[m][t[m] == 1], p[m][t[m] == 0]
        if len(a) and len(b):
            out.append((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())
    return float(np.mean(out))


def residual_profile(score):
    """Remove each article's overall level so only its criterion PROFILE is left."""
    z = (score - score.mean(0)) / score.std(0)
    return z - z.mean(1, keepdims=True)


def mean_offdiag_corr(M):
    c = np.corrcoef(M.T)
    return float(c[~np.eye(len(c), dtype=bool)].mean())


def main():
    tr, va, te = (pd.read_csv(DATA / f"{s}.csv") for s in ("train", "validation", "test"))
    X = {n: [article_text(a, b) for a, b in zip(d.title, d.text)] for n, d in (("tr", tr), ("va", va), ("te", te))}
    Y = {n: d[[f"fail_{k}" for k in CRITERIA]].values.astype(float) for n, d in (("tr", tr), ("va", va), ("te", te))}
    ybin = {n: (d.binary_label == "CREDIBLE").astype(int).values for n, d in (("va", va), ("te", te))}
    base_rate = float((tr.binary_label == "NOT_CREDIBLE").mean())
    fail_rate = np.nanmean(Y["tr"], 0)

    v2 = {n: np.mean([np.load(PRED / f"ENSV2__critraw__s{s}__{split}.npy") for s in SEEDS], 0)
          for n, split in (("va", "val"), ("te", "test"))}

    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    V = {"tr": vec.fit_transform(X["tr"]), "va": vec.transform(X["va"]), "te": vec.transform(X["te"])}

    tf = {"va": np.zeros_like(v2["va"]), "te": np.zeros_like(v2["te"])}
    clfs, rows = {}, []
    for j, k in enumerate(CRITERIA):
        m = ~np.isnan(Y["tr"][:, j])
        best = None
        for C in C_GRID:
            clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=C).fit(
                V["tr"][m], Y["tr"][m, j].astype(int))
            a = masked_auc(Y["va"][:, j], clf.predict_proba(V["va"])[:, 1])
            if best is None or a > best[0]:
                best = (a, C, clf)
        clfs[k] = best[2]
        tf["va"][:, j] = best[2].predict_proba(V["va"])[:, 1]
        tf["te"][:, j] = best[2].predict_proba(V["te"])[:, 1]

    # standardise each model's logits with validation statistics, then blend
    stats = {}
    for name, src in (("v2", v2), ("tf", tf)):
        lv = logit(src["va"])
        stats[name] = (lv.mean(0), lv.std(0))
    z = {name: {n: (logit(src[n]) - stats[name][0]) / stats[name][1] for n in ("va", "te")}
         for name, src in (("v2", v2), ("tf", tf))}

    w = np.zeros(len(CRITERIA))
    for j in range(len(CRITERIA)):
        aucs = [masked_auc(Y["va"][:, j], wt * z["tf"]["va"][:, j] + (1 - wt) * z["v2"]["va"][:, j])
                for wt in W_GRID]
        w[j] = W_GRID[int(np.argmax(aucs))]
    hy = {n: w * z["tf"][n] + (1 - w) * z["v2"][n] for n in ("va", "te")}

    # rate-matched thresholds per criterion, on validation
    def thresholds(s):
        return np.array([np.quantile(s[~np.isnan(Y["va"][:, j]), j], 1 - fail_rate[j])
                         for j in range(len(CRITERIA))])

    systems = {"v2": v2, "tfidf": tf, "hybrid": hy}
    per_crit, summary = [], {}
    for name, s in systems.items():
        thr = thresholds(s["va"])
        f1s, mccs, aucs = [], [], []
        for j, k in enumerate(CRITERIA):
            m = ~np.isnan(Y["te"][:, j])
            yj, pj = Y["te"][m, j].astype(int), (s["te"][m, j] >= thr[j]).astype(int)
            f1s.append(f1_score(yj, pj, zero_division=0)); mccs.append(matthews_corrcoef(yj, pj))
            aucs.append(roc_auc_score(yj, s["te"][m, j]))
            per_crit.append({"system": name, "criterion": k, "weight_tfidf": float(w[j]) if name == "hybrid" else None,
                             "f1": round(f1s[-1], 3), "mcc": round(mccs[-1], 3), "auc": round(aucs[-1], 3)})
        sh = [CRITERIA.index(k) for k in SHOWN]
        summary[name] = {
            "criteria_macro_f1": round(float(np.mean(f1s)), 4),
            "mean_auc": round(float(np.mean(aucs)), 4),
            "mean_auc_shown": round(float(np.mean([aucs[j] for j in sh])), 4),
            "mean_mcc": round(float(np.mean(mccs)), 4),
            "pairwise_corr_of_predictions": round(mean_offdiag_corr(s["te"][:, sh]), 3),
            "within_article_auc_profile_only": round(within_article_auc(Y["te"][:, sh], residual_profile(s["te"][:, sh])), 3),
        }
    summary["base_rates_within_article_auc"] = round(within_article_auc(
        Y["te"][:, [CRITERIA.index(k) for k in SHOWN]],
        np.tile(fail_rate[[CRITERIA.index(k) for k in SHOWN]], (len(te), 1))), 3)
    summary["reviewer_label_pairwise_corr"] = round(float(pd.DataFrame(Y["te"][:, [CRITERIA.index(k) for k in SHOWN]]).corr().values[~np.eye(9, dtype=bool)].mean()), 3)
    summary["blend_weight_tfidf"] = dict(zip(CRITERIA, w.tolist()))

    # paired bootstrap over test articles: mean criterion AUC, hybrid - v2
    rng = np.random.default_rng(0)
    sh = [CRITERIA.index(k) for k in SHOWN]
    diffs = []
    for _ in range(2000):
        b = rng.choice(len(te), len(te), replace=True)
        d = []
        for j in sh:
            yj = Y["te"][b, j]; m = ~np.isnan(yj)
            if len(np.unique(yj[m])) < 2:
                continue
            d.append(roc_auc_score(yj[m].astype(int), hy["te"][b, j][m]) - roc_auc_score(yj[m].astype(int), v2["te"][b, j][m]))
        diffs.append(np.mean(d))
    diffs = np.array(diffs)
    summary["paired_bootstrap_mean_auc_hybrid_minus_v2"] = {
        "delta": round(float(diffs.mean()), 4), "ci_low": round(float(np.percentile(diffs, 2.5)), 4),
        "ci_high": round(float(np.percentile(diffs, 97.5)), 4), "P(no gain)": round(float((diffs <= 0).mean()), 4)}

    # ---- verdict: composer over criteria, cross-validated within validation
    def composer_cv(Fv):
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        oof = np.zeros(len(Fv))
        for a, b in skf.split(Fv, ybin["va"]):
            oof[b] = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Fv[a], ybin["va"][a]).predict_proba(Fv[b])[:, 1]
        return oof

    verdict = {}
    for name, feats in (("v2", {n: v2[n] for n in ("va", "te")}), ("hybrid", hy)):
        oof = composer_cv(feats["va"])
        thr_cv = np.quantile(oof, base_rate)
        comp = LogisticRegression(max_iter=2000, class_weight="balanced").fit(feats["va"], ybin["va"])
        sv, st = comp.predict_proba(feats["va"])[:, 1], comp.predict_proba(feats["te"])[:, 1]
        thr = np.quantile(sv, base_rate)
        verdict[name] = {
            "val_cv_macro_f1": round(f1_score(ybin["va"], (oof >= thr_cv).astype(int), average="macro"), 4),
            "val_cv_auc": round(roc_auc_score(ybin["va"], oof), 4),
            "test_macro_f1": round(f1_score(ybin["te"], (st >= thr).astype(int), average="macro"), 4),
            "test_accuracy": round(float(((st >= thr).astype(int) == ybin["te"]).mean()), 4),
            "test_auc": round(roc_auc_score(ybin["te"], st), 4)}
        np.save(PRED / f"HYB__verdict_{name}__test.npy", st)
    summary["verdict"] = verdict

    pd.DataFrame(per_crit).to_csv(OUT / "tier2_hybrid_criteria.csv", index=False)
    json.dump(summary, open(OUT / "tier2_hybrid_criteria.json", "w"), indent=2)

    # ---- package the hybrid for the engine. Blended scores are z-units, so map each
    # criterion to a probability with a one-feature logistic fit on validation (Platt
    # scaling). It is monotone, so AUCs and rate-matched decisions are unchanged.
    platt_a, platt_b, thr_p = [], [], []
    for j in range(len(CRITERIA)):
        m = ~np.isnan(Y["va"][:, j])
        cal = LogisticRegression(C=1e6, max_iter=2000).fit(hy["va"][m, j:j + 1], Y["va"][m, j].astype(int))
        a, b = float(cal.coef_[0, 0]), float(cal.intercept_[0])
        assert a > 0, f"calibration for {CRITERIA[j]} is not increasing"
        pv = 1 / (1 + np.exp(-(a * hy["va"][m, j] + b)))
        platt_a.append(a); platt_b.append(b); thr_p.append(float(np.quantile(pv, 1 - fail_rate[j])))
    hyb = pd.DataFrame(per_crit).query("system == 'hybrid'").set_index("criterion")
    joblib.dump({"criteria": CRITERIA, "vectorizer": vec,
                 "classifiers": [clfs[k] for k in CRITERIA],
                 "v2_stats": stats["v2"], "tf_stats": stats["tf"], "weights": w,
                 "platt_a": np.array(platt_a), "platt_b": np.array(platt_b),
                 "thresholds": np.array(thr_p)}, MODEL_DIR / "criteria_hybrid.joblib")
    json.dump({"note": "Criteria shown to users: per-criterion blend of the v2 ensemble's criteria "
                       "probabilities and an independent TF-IDF logistic regression. The VERDICT still "
                       "uses the v2 composer over the v2 criteria (meta.json).",
               "weights_tfidf": dict(zip(CRITERIA, w.tolist())),
               "tfidf_C": {k: float(clfs[k].C) for k in CRITERIA},
               "thresholds": dict(zip(CRITERIA, thr_p)),
               "test": hyb[["f1", "mcc", "auc"]].to_dict("index"),
               "summary": {k: v for k, v in summary.items() if k != "blend_weight_tfidf"}},
              open(MODEL_DIR / "criteria_hybrid.json", "w"), indent=2)
    for n in ("va", "te"):
        np.save(PRED / f"HYB__criteria__{'val' if n == 'va' else 'test'}.npy", hy[n])

    print(pd.DataFrame(per_crit).pivot(index="criterion", columns="system", values="auc").loc[CRITERIA])
    print("\nblend weight on TF-IDF:", dict(zip(CRITERIA, w.tolist())))
    print(json.dumps({k: v for k, v in summary.items() if k != "blend_weight_tfidf"}, indent=2))


if __name__ == "__main__":
    main()
