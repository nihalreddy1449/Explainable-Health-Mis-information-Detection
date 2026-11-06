"""Confirmation run + final Tier 2 model.

Two jobs:

  --mode confirm   Retrain the baseline and the best variant with MANY seeds.
                   With 3 seeds the standard error of the mean is seed_std/sqrt(3)
                   ~= 0.015 macro-F1, which is the same size as the effect we are
                   trying to measure. More seeds is the only honest way to decide
                   whether the objective change is real.

  --mode final     Train the shipped model with the chosen config, evaluate it
                   fully (binary verdict AND per-criterion explanations), and
                   save it next to the existing tier2_final so nothing is lost.

The per-criterion evaluation matters independently of the binary verdict: Tier 2's
product is "verdict PLUS the reason", so if the joint model explains better it is
an improvement even when the binary number is flat.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             roc_auc_score)

import model_and_variant_sweep as L

OUT = L.OUT
MODELS = L.ROOT / "models"
CRITERIA = L.CRITERIA


def rate_threshold(sv, base_rate):
    return float(np.quantile(sv, base_rate))


def bin_metrics(y, s, thr):
    pred = np.where(s >= thr, L.POS, L.NEG)
    tp = int(((y == L.NEG) & (pred == L.NEG)).sum())
    fn = int(((y == L.NEG) & (pred == L.POS)).sum())
    return {"accuracy": round(float(accuracy_score(y, pred)), 4),
            "f1_macro": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
            "recall_notcred": round(float(tp / max(tp + fn, 1)), 4),
            "roc_auc": round(float(roc_auc_score(y, s)), 4),
            "pred_notcred_rate": round(float((pred == L.NEG).mean()), 4)}, pred


def criteria_metrics(df_split, crit_probs, thresholds=None):
    """Per-criterion F1 and MCC, with the rate-matching rule the project adopted.

    Also reports predicted-vs-base rate side by side, because that diagnostic is
    what caught the earlier F1-tuning degeneracy (CHAT_TRANSFER section 6.2).
    """
    rows = []
    thresholds = thresholds or {}
    for j, k in enumerate(CRITERIA):
        y = df_split[f"fail_{k}"].values
        m = ~pd.isna(y)
        if m.sum() < 10 or len(np.unique(y[m])) < 2:
            continue
        yj = y[m].astype(int)
        pj = crit_probs[m, j]
        base = float(yj.mean())
        thr = thresholds.get(k, float(np.quantile(pj, 1 - base)))
        pred = (pj >= thr).astype(int)
        rows.append({"criterion": k, "n": int(m.sum()), "base_rate": round(base, 3),
                     "pred_rate": round(float(pred.mean()), 3),
                     "threshold": round(thr, 3),
                     "f1": round(float(f1_score(yj, pred, zero_division=0)), 3),
                     "mcc": round(float(matthews_corrcoef(yj, pred)), 3),
                     "auc": round(float(roc_auc_score(yj, pj)), 3)})
    return pd.DataFrame(rows)


def train_seeds(name, backbone_key, heads, seeds, epochs, train_df=None):
    """Returns per-seed dicts of val/test scores plus the trained models."""
    bb, max_len, bs, accum, lr = L.BACKBONES[backbone_key]
    L.FORCE_FP32 = backbone_key in L.FP32_BACKBONES
    tr, va, te = L.load_splits()
    if train_df is not None:
        tr = train_df
    yv = va["binary_label"].map(L.L2I).astype(int).values

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(bb)
    tr_ds = L.ArtDS(tr, tok, max_len)
    va_ds = L.ArtDS(va, tok, max_len)
    te_ds = L.ArtDS(te, tok, max_len)
    cw = L.class_weights(tr).to(L.DEVICE)
    cpw = L.crit_pos_weight(tr).to(L.DEVICE)
    opw = L.ord_pos_weight(tr).to(L.DEVICE)

    runs = []
    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd)
        if L.DEVICE == "cuda":
            torch.cuda.manual_seed_all(sd)
        L.log(f"  [{name}] seed {sd}")
        t0 = time.time()
        model = L.T2Model(bb, heads, cw, cpw, opw).to(L.DEVICE)
        model = L.train_one(model, tr_ds, va_ds, yv, epochs, lr, bs, accum, name)
        sv, st = L.predict(model, va_ds), L.predict(model, te_ds)
        bad = [k for k, v in st.items() if not k.startswith("_") and not np.isfinite(v).all()]
        if bad:
            L.log(f"    !! non-finite {bad}, dropping seed {sd}")
            del model; torch.cuda.empty_cache(); continue
        runs.append({"seed": sd, "val": sv, "test": st, "model": model, "tok": tok,
                     "minutes": round((time.time() - t0) / 60, 2)})
    return runs, (tr, va, te)


def confirm(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    tr, va, te = L.load_splits()
    yv = va["binary_label"].map(L.L2I).astype(int).values
    yt = te["binary_label"].map(L.L2I).astype(int).values
    base_rate = float((tr["binary_label"] == "NOT_CREDIBLE").mean())

    configs = [
        ("BASE_binary", "pubmedbert", {"binary"}, 4, "bin"),
        ("BEST_ord_crit", "pubmedbert", {"ordinal", "criteria"}, 5, "ord"),
    ]
    all_rows, store = [], {}
    for name, bbk, heads, epochs, main_score in configs:
        runs, _ = train_seeds(name, bbk, heads, seeds, epochs)
        store[name] = runs
        for r in runs:
            for key in [k for k in ("bin", "ord", "crit", "crit_nodm") if k in r["test"]]:
                thr = rate_threshold(r["val"][key], base_rate)
                m, _ = bin_metrics(yt, r["test"][key], thr)
                all_rows.append({"config": name, "seed": r["seed"], "score": key,
                                 "minutes": r["minutes"], **m})
            if "_crit_raw" in r["test"]:
                comp = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
                    r["val"]["_crit_raw"], yv)
                sv = comp.predict_proba(r["val"]["_crit_raw"])[:, 1]
                st = comp.predict_proba(r["test"]["_crit_raw"])[:, 1]
                r["val"]["crit_lr"], r["test"]["crit_lr"] = sv, st
                thr = rate_threshold(sv, base_rate)
                m, _ = bin_metrics(yt, st, thr)
                all_rows.append({"config": name, "seed": r["seed"], "score": "crit_lr",
                                 "minutes": r["minutes"], **m})
        for r in runs:
            del r["model"]
        torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "tier2_confirm_seeds.csv", index=False)

    L.log("\n" + "=" * 92)
    L.log(f"CONFIRMATION RUN  n_seeds={len(seeds)}  (rate-matched threshold, base rate {base_rate:.3f})")
    L.log("=" * 92)
    L.log(f"{'config':<16}{'score':<10}{'macroF1 mean+-std':>24}{'SE':>8}{'acc':>9}{'recall':>9}{'AUC':>9}")
    for (c, s), g in df.groupby(["config", "score"]):
        n = len(g)
        mu, sd_ = g.f1_macro.mean(), g.f1_macro.std()
        L.log(f"{c:<16}{s:<10}{mu:>15.4f} +- {sd_:<6.4f}{sd_/max(np.sqrt(n),1):>8.4f}"
              f"{g.accuracy.mean():>9.4f}{g.recall_notcred.mean():>9.4f}{g.roc_auc.mean():>9.4f}  n={n}")

    # Welch t-test on the per-seed macro-F1 populations: does the objective change
    # move the MODEL, independently of which test articles we happened to draw?
    from scipy import stats as st_
    a = df[(df.config == "BASE_binary") & (df.score == "bin")].f1_macro.values
    for s in sorted(df[df.config == "BEST_ord_crit"].score.unique()):
        b = df[(df.config == "BEST_ord_crit") & (df.score == s)].f1_macro.values
        if len(a) > 1 and len(b) > 1:
            t, p = st_.ttest_ind(b, a, equal_var=False)
            L.log(f"  BEST/{s:<10} vs BASE/bin : delta={b.mean()-a.mean():+.4f}  "
                  f"Welch t={t:+.2f}  p={p:.4f}")
    L.log(f"\nwrote {OUT/'tier2_confirm_seeds.csv'}")


def final(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    heads = set(args.heads.split(","))
    runs, (tr, va, te) = train_seeds("FINAL", args.backbone, heads, seeds, args.epochs)
    if not runs:
        L.log("no successful runs"); return
    yv = va["binary_label"].map(L.L2I).astype(int).values
    yt = te["binary_label"].map(L.L2I).astype(int).values
    base_rate = float((tr["binary_label"] == "NOT_CREDIBLE").mean())

    # ensemble the score vectors across seeds
    ens_v = {k: np.mean([r["val"][k] for r in runs], 0)
             for k in runs[0]["val"] if not k.startswith("_")}
    ens_t = {k: np.mean([r["test"][k] for r in runs], 0)
             for k in runs[0]["test"] if not k.startswith("_")}
    raw_v = np.mean([r["val"]["_crit_raw"] for r in runs], 0)
    raw_t = np.mean([r["test"]["_crit_raw"] for r in runs], 0)

    comp = LogisticRegression(max_iter=2000, class_weight="balanced").fit(raw_v, yv)
    ens_v["crit_lr"] = comp.predict_proba(raw_v)[:, 1]
    ens_t["crit_lr"] = comp.predict_proba(raw_t)[:, 1]

    report = {}
    for k, s in ens_t.items():
        thr = rate_threshold(ens_v[k], base_rate)
        m, pred = bin_metrics(yt, s, thr)
        report[k] = {**m, "threshold": round(thr, 4)}
        L.log(f"  {k:<10} acc={m['accuracy']:.4f} macroF1={m['f1_macro']:.4f} "
              f"recall={m['recall_notcred']:.4f} auc={m['roc_auc']:.4f}")

    crit_tbl = criteria_metrics(te, raw_t)
    L.log("\nper-criterion (rate-matched):")
    L.log(crit_tbl.to_string(index=False))
    crit_tbl.to_csv(OUT / "tier2_v2_criteria_performance.csv", index=False)

    dest = MODELS / "tier2_final_v2"
    dest.mkdir(parents=True, exist_ok=True)
    best = runs[0]
    torch.save({"state_dict": best["model"].state_dict(),
                "backbone": L.BACKBONES[args.backbone][0],
                "heads": sorted(heads), "criteria": CRITERIA,
                "ord_bin_k": L.ORD_BIN_K}, dest / "model.pt")
    best["tok"].save_pretrained(dest)
    np.save(dest / "composer_coef.npy", comp.coef_)
    np.save(dest / "composer_intercept.npy", comp.intercept_)
    with open(dest / "meta.json", "w") as f:
        json.dump({"backbone": L.BACKBONES[args.backbone][0], "heads": sorted(heads),
                   "seeds": [r["seed"] for r in runs], "criteria": CRITERIA,
                   "base_rate": base_rate, "report": report,
                   "criteria_macro_f1": float(crit_tbl["f1"].mean()),
                   "criteria_mean_mcc": float(crit_tbl["mcc"].mean()),
                   "built": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}, f, indent=2)
    for k in ens_t:
        np.save(OUT / "tier2_preds" / f"FINALV2__{k}__sENS__test.npy", ens_t[k])
        np.save(OUT / "tier2_preds" / f"FINALV2__{k}__sENS__val.npy", ens_v[k])
    L.log(f"\nsaved -> {dest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="confirm", choices=["confirm", "final"])
    ap.add_argument("--seeds", default="42,1337,2024,7,99,555,31415,2718")
    ap.add_argument("--backbone", default="pubmedbert")
    ap.add_argument("--heads", default="ordinal,criteria")
    ap.add_argument("--epochs", type=int, default=5)
    a = ap.parse_args()
    L.log(f"\n{'#'*78}\nT2 FINAL mode={a.mode}  {datetime.now():%Y-%m-%d %H:%M:%S}\n{'#'*78}")
    (confirm if a.mode == "confirm" else final)(a)
