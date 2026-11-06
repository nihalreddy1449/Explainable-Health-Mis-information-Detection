"""Does training on train+validation help? (the +21% data lever)

The v2 work established that SUPERVISION VOLUME is Tier 2's bottleneck: the gain
came from using ~15 signals per article instead of 1. The obvious next move is
more articles. The validation split is 318 articles sitting idle at training
time -- folding it in takes training data from 1,483 to 1,801, a 21% increase.

The catch: validation is what selects the checkpoint, sets the rate-matched
threshold, and fits the criteria composer. Train on it naively and all three
become self-fitted, which would inflate the result rather than improve it.

So this uses proper K-fold cross-validation over the pooled train+validation
articles:

  * each fold trains on K-1 parts and uses the held-out part for checkpoint
    selection, exactly as the single-split protocol does;
  * every article gets an OUT-OF-FOLD prediction from a model that never saw it,
    and the composer and threshold are fitted on those OOF predictions only;
  * the K fold-models are ensembled for the test prediction.

The test set is never touched by any of it, and is the same 318 articles used
everywhere else, so the number is directly comparable to the v2 baseline.

Usage:
    python experiments_tier2/more_data_cross_validation.py --folds 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer

import model_and_variant_sweep as L

OUT = L.OUT
PRED = L.PRED


def binmetrics(y, s, thr):
    pred = np.where(s >= thr, L.POS, L.NEG)
    tp = int(((y == L.NEG) & (pred == L.NEG)).sum())
    fn = int(((y == L.NEG) & (pred == L.POS)).sum())
    return {"accuracy": round(float(accuracy_score(y, pred)), 4),
            "f1_macro": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
            "recall_notcred": round(float(tp / max(tp + fn, 1)), 4),
            "roc_auc": round(float(roc_auc_score(y, s)), 4)}, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--backbone", default="pubmedbert")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    bb, max_len, bs, accum, lr = L.BACKBONES[args.backbone]
    tr, va, te = L.load_splits()
    pool = pd.concat([tr, va], ignore_index=True)
    y_pool = pool["binary_label"].map(L.L2I).astype(int).values
    yt = te["binary_label"].map(L.L2I).astype(int).values
    base_rate = float((pool["binary_label"] == "NOT_CREDIBLE").mean())

    L.log(f"\n{'#'*80}\nT2 CV  {datetime.now():%Y-%m-%d %H:%M:%S}\n{'#'*80}")
    L.log(f"pooled train+validation = {len(pool)} articles "
          f"(was {len(tr)} train only, +{100*(len(pool)-len(tr))/len(tr):.0f}%)")
    L.log(f"test = {len(te)} articles, untouched. base rate {base_rate:.4f}")

    tok = AutoTokenizer.from_pretrained(bb)
    te_ds = L.ArtDS(te, tok, max_len)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_raw = np.zeros((len(pool), L.N_CRIT), dtype=np.float32)
    oof = {k: np.zeros(len(pool), dtype=np.float32) for k in ("ord", "crit", "crit_nodm")}
    test_scores = {k: [] for k in ("ord", "crit", "crit_nodm")}
    test_raw = []

    for fold, (itr, iva) in enumerate(skf.split(pool, y_pool), 1):
        L.log(f"\n  fold {fold}/{args.folds}   train {len(itr)}  held-out {len(iva)}")
        f_tr, f_va = pool.iloc[itr].reset_index(drop=True), pool.iloc[iva].reset_index(drop=True)
        torch.manual_seed(args.seed + fold); np.random.seed(args.seed + fold)
        if L.DEVICE == "cuda":
            torch.cuda.manual_seed_all(args.seed + fold)

        cw = L.class_weights(f_tr).to(L.DEVICE)
        cpw = L.crit_pos_weight(f_tr).to(L.DEVICE)
        opw = L.ord_pos_weight(f_tr).to(L.DEVICE)
        model = L.T2Model(bb, {"ordinal", "criteria"}, cw, cpw, opw).to(L.DEVICE)
        t0 = time.time()
        model = L.train_one(model, L.ArtDS(f_tr, tok, max_len), L.ArtDS(f_va, tok, max_len),
                            y_pool[iva], args.epochs, lr, bs, accum, f"cv{fold}")
        sv = L.predict(model, L.ArtDS(f_va, tok, max_len))
        st = L.predict(model, te_ds)
        for k in oof:
            oof[k][iva] = sv[k]
        oof_raw[iva] = sv["_crit_raw"]
        for k in test_scores:
            test_scores[k].append(st[k])
        test_raw.append(st["_crit_raw"])
        L.log(f"    fold done in {(time.time()-t0)/60:.1f} min")
        del model
        if L.DEVICE == "cuda":
            torch.cuda.empty_cache()

    # composer fitted purely on out-of-fold predictions
    comp = LogisticRegression(max_iter=2000, class_weight="balanced").fit(oof_raw, y_pool)
    oof["crit_lr"] = comp.predict_proba(oof_raw)[:, 1]
    ens_raw = np.mean(test_raw, 0)
    ens = {k: np.mean(v, 0) for k, v in test_scores.items()}
    ens["crit_lr"] = comp.predict_proba(ens_raw)[:, 1]

    L.log("\n" + "=" * 84)
    L.log(f"CV ENSEMBLE ON TEST  ({args.folds} folds, trained on train+validation)")
    L.log("=" * 84)
    rows = []
    for k in ("ord", "crit", "crit_nodm", "crit_lr"):
        thr = float(np.quantile(oof[k], base_rate))   # threshold from OOF, not test
        m, _ = binmetrics(yt, ens[k], thr)
        L.log(f"  {k:<10} acc={m['accuracy']:.4f}  macroF1={m['f1_macro']:.4f}  "
              f"recall={m['recall_notcred']:.4f}  auc={m['roc_auc']:.4f}")
        rows.append({"exp": "CV_trainval", "score": k, "folds": args.folds,
                     "threshold": round(thr, 4), **m})
        np.save(PRED / f"CV5__{k}__sENS__test.npy", ens[k])

    L.log("\n  reference, v2 trained on train only (8-seed mean, same rule):")
    L.log("    crit_lr    acc=0.7229  macroF1=0.6851  recall=0.5596  auc=0.7179")
    L.log("  reference, v2 5-seed ensemble (train only):")
    L.log("    ord        acc=0.7358  macroF1=0.6999  recall=0.5794  auc=0.7274")

    pd.DataFrame(rows).to_csv(OUT / "tier2_cv_results.csv", index=False)
    json.dump({"folds": args.folds, "pool_size": int(len(pool)),
               "train_only_size": int(len(tr)), "base_rate": base_rate,
               "results": rows}, open(OUT / "tier2_cv_metrics.json", "w"), indent=2)
    L.log(f"\nwrote {OUT/'tier2_cv_results.csv'}")


if __name__ == "__main__":
    main()
