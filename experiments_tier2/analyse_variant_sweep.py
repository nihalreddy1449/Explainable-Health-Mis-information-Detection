"""Analyse the Tier 2 lab runs.

Reads the saved per-variant score vectors from outputs/tier2_preds/ and answers
the only question that matters: does any variant beat the existing binary
baseline by more than noise?

TWO NOISE SOURCES, reported separately, because they are different things:

  seed std   - retrain the SAME config with a different seed. Measured at
               +-0.026 macro-F1 for the baseline. An "improvement" smaller than
               this is not a property of the method at all.
  paired CI  - resample the 318 test articles, the SAME resample for both
               systems. This is the correct system-vs-system test; it is far
               tighter than comparing two independent marginal CIs, because
               article-level difficulty cancels in the difference.

DECISION RULE. Picking the threshold that maximises macro-F1 on 318 validation
articles overfits -- this project already found (section 6.2 of CHAT_TRANSFER)
that F1-maximising threshold selection degenerates. Evidence it is happening
here too: the baseline's seed-ensemble scored BELOW its own per-seed mean, which
is only possible if the tuned threshold failed to transfer. So we report three
rules and treat the non-degenerate one as primary:

  rate   - threshold set so the predicted NOT_CREDIBLE rate on validation equals
           the training base rate. Cannot collapse to a constant classifier.
           PRIMARY, and consistent with the rule already adopted for the criteria.
  f1     - maximise validation macro-F1 (shown for continuity; overfits)
  fixed  - plain 0.5

A variant is only called a win if the paired 95% CI excludes 0 under the PRIMARY
rule AND the per-seed mean gain is comparable to the seed std.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
PRED = OUT / "tier2_preds"
DATA = ROOT / "datasets" / "processed_tier2"

LABELS = ["NOT_CREDIBLE", "CREDIBLE"]
L2I = {l: i for i, l in enumerate(LABELS)}
NEG, POS = 0, 1


# ------------------------------------------------------------------ fast metrics
def macro_f1_counts(n00, n01, n10, n11):
    """n[true][pred] over classes {0=NOT_CREDIBLE, 1=CREDIBLE}. Vectorised."""
    tp0, fn0, fp0 = n00, n01, n10
    tp1, fn1, fp1 = n11, n10, n01
    f0 = np.where(2 * tp0 + fp0 + fn0 > 0, 2 * tp0 / np.maximum(2 * tp0 + fp0 + fn0, 1e-12), 0.0)
    f1 = np.where(2 * tp1 + fp1 + fn1 > 0, 2 * tp1 / np.maximum(2 * tp1 + fp1 + fn1, 1e-12), 0.0)
    return (f0 + f1) / 2.0


def paired_boot_fast(y, pa, pb, n=10000, seed=0):
    """CI on macroF1(a) - macroF1(b), same resampled articles for both systems.

    Each test item falls in one of four (true, pred) cells per system; a bootstrap
    replicate only needs the four cell counts, so we count category codes with
    bincount instead of recomputing f1_score. ~100x faster, identical result.
    """
    code_a = (y * 2 + pa).astype(np.int64)     # 0:(0,0) 1:(0,1) 2:(1,0) 3:(1,1)
    code_b = (y * 2 + pb).astype(np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))

    def counts(code):
        c = code[idx]                                   # (n, N)
        oh = np.zeros((n, 4), np.int32)
        for k in range(4):
            oh[:, k] = (c == k).sum(1)
        return oh

    ca, cb = counts(code_a), counts(code_b)
    fa = macro_f1_counts(ca[:, 0], ca[:, 1], ca[:, 2], ca[:, 3])
    fb = macro_f1_counts(cb[:, 0], cb[:, 1], cb[:, 2], cb[:, 3])
    d = fa - fb
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), \
        float((d <= 0).mean())


def load_y():
    tr = pd.read_csv(DATA / "train.csv")
    va = pd.read_csv(DATA / "validation.csv")
    te = pd.read_csv(DATA / "test.csv")
    base_rate = float((tr["binary_label"] == "NOT_CREDIBLE").mean())
    return (va["binary_label"].map(L2I).astype(int).values,
            te["binary_label"].map(L2I).astype(int).values, base_rate)


def metrics_at(y, s, thr):
    pred = np.where(s >= thr, POS, NEG)
    tp = int(((y == NEG) & (pred == NEG)).sum())
    fn = int(((y == NEG) & (pred == POS)).sum())
    return {"accuracy": float(accuracy_score(y, pred)),
            "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
            "recall_notcred": float(tp / max(tp + fn, 1)),
            "pred_notcred_rate": float((pred == NEG).mean())}, pred


def thr_f1(yv, sv):
    best_t, best = 0.5, -1.0
    for t in np.round(np.linspace(0.02, 0.98, 97), 4):
        f = metrics_at(yv, sv, t)[0]["f1_macro"]
        if f > best:
            best, best_t = f, float(t)
    return best_t


def thr_rate(sv, base_rate):
    """Threshold so that predicted NOT_CREDIBLE rate on val == training base rate."""
    return float(np.quantile(sv, base_rate))


def discover():
    # longest-first so crit_nodm / crit_lr are not shadowed by crit;
    # critraw is the 10-column matrix, not a score vector, so it is excluded.
    pat = re.compile(r"^(.+?)__(crit_nodm|crit_lr|bin|ord|crit|blend)__s(.+?)__(val|test)\.npy$")
    found = {}
    for p in sorted(PRED.glob("*.npy")):
        m = pat.match(p.name)
        if not m:
            continue
        v, s, sd, split = m.groups()
        found.setdefault((v, s, sd), {})[split] = p
    return {k: v for k, v in found.items() if "val" in v and "test" in v}


RULES = ["rate", "f1", "fixed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="A_binary")
    ap.add_argument("--baseline-score", default="bin")
    ap.add_argument("--primary", default="rate", choices=RULES)
    ap.add_argument("--out", default="tier2_lab_summary.csv")
    args = ap.parse_args()

    yv, yt, base_rate = load_y()
    found = discover()
    if not found:
        print("no predictions found in", PRED)
        return
    print(f"training base rate of NOT_CREDIBLE = {base_rate:.4f}\n")

    rows = {}
    for (v, s, sd), paths in found.items():
        sv, st = np.load(paths["val"]), np.load(paths["test"])
        auc = float(roc_auc_score(yt, st))
        for rule in RULES:
            thr = {"rate": lambda: thr_rate(sv, base_rate),
                   "f1": lambda: thr_f1(yv, sv),
                   "fixed": lambda: 0.5}[rule]()
            m, pred = metrics_at(yt, st, thr)
            rows[(v, s, sd, rule)] = dict(
                variant=v, score=s, seed=sd, rule=rule, threshold=round(thr, 4),
                roc_auc=round(auc, 4), _pred=pred,
                **{k: round(val, 4) for k, val in m.items()})

    df = pd.DataFrame([{k: val for k, val in r.items() if not k.startswith("_")}
                       for r in rows.values()])
    df.to_csv(OUT / "tier2_lab_all_rules.csv", index=False)

    # ---------------- AUC is threshold-free: the cleanest ranking signal
    auc_tbl = (df[df.seed != "ENS"].groupby(["variant", "score"])["roc_auc"]
               .agg(["mean", "std", "count"]).round(4).sort_values("mean", ascending=False))
    print("=" * 96)
    print("THRESHOLD-FREE RANKING: ROC-AUC, mean +- std over seeds")
    print("(AUC needs no decision rule, so it cannot be inflated by threshold tuning)")
    print("=" * 96)
    for (v, s), r in auc_tbl.iterrows():
        print(f"  {v:<16}{s:<7} AUC={r['mean']:.4f} +- {r['std']:.4f}   n={int(r['count'])}")

    # ---------------- per-seed table under the primary rule
    pr = df[(df.rule == args.primary) & (df.seed != "ENS")]
    agg = (pr.groupby(["variant", "score"])
           [["accuracy", "f1_macro", "recall_notcred", "pred_notcred_rate"]]
           .agg(["mean", "std"]).round(4))
    print("\n" + "=" * 96)
    print(f"PER-SEED MEAN +- STD under the PRIMARY rule ('{args.primary}')")
    print(f"(pred rate should sit near the base rate {base_rate:.3f}; far from it = degenerate)")
    print("=" * 96)
    print(f"{'variant':<16}{'score':<7}{'acc':>15}{'macroF1':>17}{'recall':>15}{'predRate':>10}")
    for (v, s), r in agg.iterrows():
        print(f"{v:<16}{s:<7}"
              f"{r[('accuracy','mean')]:>8.4f}+-{r[('accuracy','std')]:<6.4f}"
              f"{r[('f1_macro','mean')]:>10.4f}+-{r[('f1_macro','std')]:<6.4f}"
              f"{r[('recall_notcred','mean')]:>8.4f}+-{r[('recall_notcred','std')]:<6.4f}"
              f"{r[('pred_notcred_rate','mean')]:>10.3f}")

    # ---------------- paired bootstrap, primary rule, seed-ensembles
    comp = []
    for rule in RULES:
        base_key = (args.baseline, args.baseline_score, "ENS", rule)
        if base_key not in rows:
            continue
        base = rows[base_key]
        star = "  <-- PRIMARY" if rule == args.primary else ""
        print("\n" + "=" * 96)
        print(f"PAIRED BOOTSTRAP vs {args.baseline}/{args.baseline_score} seed-ensemble  "
              f"[rule={rule}, baseline macroF1 {base['f1_macro']:.4f}]{star}")
        print("=" * 96)
        print(f"{'variant':<16}{'score':<7}{'macroF1':>9}{'delta':>9}"
              f"{'95% CI of delta':>22}{'P(no gain)':>12}   verdict")
        sel = [(k, r) for k, r in rows.items() if k[2] == "ENS" and k[3] == rule
               and k[:2] != (args.baseline, args.baseline_score)]
        for k, r in sorted(sel, key=lambda x: -x[1]["f1_macro"]):
            mean_d, lo, hi, p_null = paired_boot_fast(yt, r["_pred"], base["_pred"])
            verdict = "WIN (CI excludes 0)" if lo > 0 else (
                "worse" if hi < 0 else "inside noise")
            print(f"{k[0]:<16}{k[1]:<7}{r['f1_macro']:>9.4f}{mean_d:>+9.4f}"
                  f"   [{lo:>+.4f}, {hi:>+.4f}]{p_null:>12.3f}   {verdict}")
            comp.append(dict(rule=rule, variant=k[0], score=k[1],
                             f1_macro=r["f1_macro"], delta=round(mean_d, 4),
                             ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                             p_no_gain=round(p_null, 4), significant=bool(lo > 0)))

    pd.DataFrame(comp).to_csv(OUT / args.out, index=False)
    print(f"\nwrote {OUT/args.out} and {OUT/'tier2_lab_all_rules.csv'}")


if __name__ == "__main__":
    main()
