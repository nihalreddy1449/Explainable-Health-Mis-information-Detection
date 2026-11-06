"""Train and SAVE every member of the Tier 2 v2 seed ensemble.

WHY THIS EXISTS
---------------
`confirm_over_8_seeds.py --mode final` trained five seeds and reported the ensemble, but saved
only the FIRST seed's weights (models/tier2_final_v2/model.pt). So the reported
number came from an ensemble that nothing could load, and the saved composer had
been fitted on ensemble outputs but was applied to a single model.

This script re-runs the identical configuration (PubMedBERT, ordinal + criteria
heads, 5 epochs, the same five seeds, the same checkpoint rule) and saves all five
members, so the notebook and the demo run exactly the system whose numbers are
quoted. It writes to a NEW directory; models/tier2_final_v2/ is left untouched.

Two evaluation rules, fixed before looking at test:
  * The verdict is `crit_lr`: a logistic regression over the ten predicted
    criteria, fitted on validation. This is the rule the 8-seed confirmation
    run tested (p = 0.022). The other rules are reported for transparency only.
  * All thresholds come from VALIDATION at the TRAINING base rate (rate
    matching). Per-criterion thresholds previously used the test set's own
    label rate; the old protocol is reported alongside for comparison.

Usage:
    python experiments_tier2/train_final_ensemble.py
"""
from __future__ import annotations

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
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from transformers import AutoConfig, AutoTokenizer

import model_and_variant_sweep as L
from confirm_over_8_seeds import bin_metrics, criteria_metrics, rate_threshold

SEEDS = [42, 1337, 2024, 7, 99]
BACKBONE_KEY = "pubmedbert"
HEADS = {"ordinal", "criteria"}
EPOCHS = 5
DEST = L.ROOT / "models" / "tier2_v2_ensemble"
TAG = "ENSV2"


def criteria_val_thresholds(va, raw_v, tr):
    """Per-criterion fail threshold: the validation quantile that flags the same
    fraction of articles as fail that criterion in TRAINING. No test labels used."""
    thr = {}
    for j, k in enumerate(L.CRITERIA):
        ytr = tr[f"fail_{k}"].dropna()
        mv = ~pd.isna(va[f"fail_{k}"].values)
        thr[k] = float(np.quantile(raw_v[mv, j], 1 - float(ytr.mean())))
    return thr


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    bb, max_len, bs, accum, lr = L.BACKBONES[BACKBONE_KEY]
    tr, va, te = L.load_splits()
    yv = va["binary_label"].map(L.L2I).astype(int).values
    yt = te["binary_label"].map(L.L2I).astype(int).values
    base_rate = float((tr["binary_label"] == "NOT_CREDIBLE").mean())

    tok = AutoTokenizer.from_pretrained(bb)
    tok.save_pretrained(DEST)
    AutoConfig.from_pretrained(bb).save_pretrained(DEST)
    tr_ds, va_ds, te_ds = (L.ArtDS(d, tok, max_len) for d in (tr, va, te))
    cw = L.class_weights(tr).to(L.DEVICE)
    cpw = L.crit_pos_weight(tr).to(L.DEVICE)
    opw = L.ord_pos_weight(tr).to(L.DEVICE)

    L.log(f"\n{'#'*78}\nT2 ENSEMBLE  {datetime.now():%Y-%m-%d %H:%M:%S}  seeds={SEEDS}  -> {DEST}\n{'#'*78}")

    # capture the per-epoch lines train_one prints, so the notebook can show them
    history, _orig_log = {}, L.log

    members = []
    for sd in SEEDS:
        torch.manual_seed(sd); np.random.seed(sd)
        if L.DEVICE == "cuda":
            torch.cuda.manual_seed_all(sd)
        L.log(f"  [{TAG}] seed {sd}")
        lines = []
        L.log = lambda *a, _l=lines: (_l.append(" ".join(str(x) for x in a).strip()), _orig_log(*a))
        t0 = time.time()
        model = L.T2Model(bb, HEADS, cw, cpw, opw).to(L.DEVICE)
        model = L.train_one(model, tr_ds, va_ds, yv, EPOCHS, lr, bs, accum, TAG)
        L.log = _orig_log
        mins = round((time.time() - t0) / 60, 2)
        sv, st = L.predict(model, va_ds), L.predict(model, te_ds)
        if any(not np.isfinite(v).all() for k, v in st.items()):
            raise RuntimeError(f"non-finite scores for seed {sd}")
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()
                 if k not in ("cw", "cpw", "opw")}
        torch.save(state, DEST / f"seed_{sd}.pt")
        history[sd] = {"minutes": mins, "epochs": [l for l in lines if l.startswith("ep")]}
        members.append({"seed": sd, "val": sv, "test": st})
        for split, s in (("val", sv), ("test", st)):
            np.save(L.PRED / f"{TAG}__critraw__s{sd}__{split}.npy", s["_crit_raw"])
            np.save(L.PRED / f"{TAG}__ord__s{sd}__{split}.npy", s["ord"])
        L.log(f"    saved seed_{sd}.pt  ({mins} min)")
        del model
        if L.DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------- ensemble
    raw_v = np.mean([m["val"]["_crit_raw"] for m in members], 0)
    raw_t = np.mean([m["test"]["_crit_raw"] for m in members], 0)
    comp = LogisticRegression(max_iter=2000, class_weight="balanced").fit(raw_v, yv)

    keep = [j for j, k in enumerate(L.CRITERIA) if k != "disease_mongering"]
    ens = {"val": {}, "test": {}}
    for split, raw in (("val", raw_v), ("test", raw_t)):
        ens[split]["ord"] = np.mean([m[split]["ord"] for m in members], 0)
        ens[split]["crit"] = 1.0 - raw.mean(1)
        ens[split]["crit_nodm"] = 1.0 - raw[:, keep].mean(1)
        ens[split]["crit_lr"] = comp.predict_proba(raw)[:, 1]

    report = {}
    for k in ("crit_lr", "ord", "crit", "crit_nodm"):
        thr = rate_threshold(ens["val"][k], base_rate)
        m, _ = bin_metrics(yt, ens["test"][k], thr)
        report[k] = {**m, "threshold": round(thr, 4)}
        L.log(f"  {k:<10} acc={m['accuracy']:.4f} macroF1={m['f1_macro']:.4f} "
              f"recall={m['recall_notcred']:.4f} auc={m['roc_auc']:.4f}"
              + ("   <- VERDICT RULE" if k == "crit_lr" else ""))
        for split in ("val", "test"):
            np.save(L.PRED / f"{TAG}__{k}__sENS__{split}.npy", ens[split][k])

    # per-seed crit_lr (each member with its own composer), for the spread
    per_seed = []
    for m in members:
        c = LogisticRegression(max_iter=2000, class_weight="balanced").fit(m["val"]["_crit_raw"], yv)
        sv_, st_ = c.predict_proba(m["val"]["_crit_raw"])[:, 1], c.predict_proba(m["test"]["_crit_raw"])[:, 1]
        mm, _ = bin_metrics(yt, st_, rate_threshold(sv_, base_rate))
        per_seed.append({"seed": m["seed"], **mm})
    ps = pd.DataFrame(per_seed)
    L.log(f"  per-seed crit_lr macroF1 {ps.f1_macro.mean():.4f} +- {ps.f1_macro.std():.4f}")

    # ---------------------------------------------------------------- criteria
    crit_thr = criteria_val_thresholds(va, raw_v, tr)
    tbl_val = criteria_metrics(te, raw_t, crit_thr)          # honest: val thresholds
    tbl_old = criteria_metrics(te, raw_t)                     # old: test-rate thresholds
    tbl_val.to_csv(L.OUT / "tier2_v2_ensemble_criteria.csv", index=False)
    L.log("\nper-criterion on test (validation thresholds):")
    L.log(tbl_val.to_string(index=False))
    L.log(f"criteria macro-F1: val-threshold {tbl_val.f1.mean():.4f}   "
          f"test-rate (old protocol) {tbl_old.f1.mean():.4f}")

    meta = {
        "backbone": bb, "max_len": max_len, "heads": sorted(HEADS), "epochs": EPOCHS,
        "seeds": SEEDS, "criteria": L.CRITERIA, "ord_bin_k": L.ORD_BIN_K,
        "train_base_rate_not_credible": base_rate,
        "verdict_rule": "crit_lr",
        "verdict_threshold": report["crit_lr"]["threshold"],
        "composer": {"coef": comp.coef_[0].tolist(), "intercept": float(comp.intercept_[0])},
        "criteria_thresholds": crit_thr,
        "criteria_test": tbl_val.set_index("criterion")[["f1", "mcc", "auc"]].to_dict("index"),
        "report": report,
        "per_seed_crit_lr": per_seed,
        "criteria_macro_f1_val_thresholds": round(float(tbl_val.f1.mean()), 4),
        "criteria_macro_f1_test_rate_thresholds": round(float(tbl_old.f1.mean()), 4),
        "criteria_mean_mcc": round(float(tbl_val.mcc.mean()), 4),
        "history": history,
        "built": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "note": "Every member is saved (seed_<s>.pt). Scores are averaged across members; "
                "the verdict is the composer over the averaged criteria probabilities, "
                "thresholded at the validation quantile equal to the training base rate.",
    }
    json.dump(meta, open(DEST / "meta.json", "w"), indent=2)
    json.dump({k: v for k, v in meta.items() if k not in ("composer", "criteria_thresholds")},
              open(L.OUT / "tier2_v2_ensemble_metrics.json", "w"), indent=2)
    L.log(f"\nsaved -> {DEST}")


if __name__ == "__main__":
    main()
