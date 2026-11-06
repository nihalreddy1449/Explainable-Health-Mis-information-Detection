"""ReCOVery experiments: a second dataset, and whether it can help FakeHealth.

ReCOVery carries ONLY a binary outlet-reliability label -- no 0-5 rating and no
ten criteria. So the v2 architecture's whole advantage (richer supervision) does
not exist on this corpus. What ReCOVery can offer instead is (a) a second,
different Tier 2 capability, and (b) 1,409 extra health-news articles to attack
the data bottleneck that the v2 work identified.

Four experiments:

  R1  recovery      binary PubMedBERT on ReCOVery, publisher-disjoint splits.
                    Reference point: a plain TF-IDF scores 0.773 here, so the
                    transformer has to beat that to justify itself.

  R2  zeroshot      Does a FakeHealth model transfer to ReCOVery, and vice versa?
                    If the two label sets measured the same thing, transfer would
                    work. This is the empirical test of whether they may be merged.

  R3  intermediate  ReCOVery -> FakeHealth sequential transfer (STILT). Train the
                    encoder on ReCOVery's binary task, keep the encoder, throw the
                    head away, then fine-tune with the v2 ordinal+criteria heads.
                    Labels are never merged, so the PubHealth failure (two
                    incompatible label schemes in one target) cannot recur.

  R4  combined      Same idea but multi-corpus in one run: not implemented as a
                    separate path -- R3 is the cheaper test of the same hypothesis.

Every FakeHealth number uses the rate-matched threshold and multi-seed reporting
established in model_and_variant_sweep.py.
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
from transformers import AutoTokenizer

import model_and_variant_sweep as L

ROOT = L.ROOT
RECDIR = ROOT / "datasets" / "processed_recovery"
OUT = L.OUT
PRED = L.PRED


def load_recovery():
    """ReCOVery frames padded with the columns ArtDS expects.

    rating/criteria are filled with sentinels that the ordinal and criteria heads
    would ignore; only the binary head is ever trained on this corpus.
    """
    out = []
    for s in ("train", "validation", "test"):
        d = pd.read_csv(RECDIR / f"{s}.csv")
        d["rating"] = np.where(d["binary_label"] == "CREDIBLE", 4, 1)  # unused
        for k in L.CRITERIA:
            d[f"fail_{k}"] = np.nan                                     # fully masked
        out.append(d)
    return out


def rate_thr(sv, base_rate):
    return float(np.quantile(sv, base_rate))


def binmetrics(y, s, thr):
    pred = np.where(s >= thr, L.POS, L.NEG)
    tp = int(((y == L.NEG) & (pred == L.NEG)).sum())
    fn = int(((y == L.NEG) & (pred == L.POS)).sum())
    return {"accuracy": round(float(accuracy_score(y, pred)), 4),
            "f1_macro": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
            "recall_notcred": round(float(tp / max(tp + fn, 1)), 4),
            "roc_auc": round(float(roc_auc_score(y, s)), 4)}


def train_on(frames, heads, seed, epochs, backbone="pubmedbert", init_encoder=None):
    bb, max_len, bs, accum, lr = L.BACKBONES[backbone]
    tr, va, te = frames
    tok = AutoTokenizer.from_pretrained(bb)
    yv = va["binary_label"].map(L.L2I).astype(int).values

    torch.manual_seed(seed); np.random.seed(seed)
    if L.DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)

    cw = L.class_weights(tr).to(L.DEVICE)
    cpw = L.crit_pos_weight(tr).to(L.DEVICE) if "criteria" in heads else None
    opw = L.ord_pos_weight(tr).to(L.DEVICE) if "ordinal" in heads else None
    model = L.T2Model(bb, heads, cw, cpw, opw).to(L.DEVICE)
    if init_encoder is not None:
        model.enc.load_state_dict(init_encoder)
        L.log("      (encoder initialised from the ReCOVery run)")

    model = L.train_one(model, L.ArtDS(tr, tok, max_len), L.ArtDS(va, tok, max_len),
                        yv, epochs, lr, bs, accum, "x")
    sv = L.predict(model, L.ArtDS(va, tok, max_len))
    st = L.predict(model, L.ArtDS(te, tok, max_len))
    return model, tok, sv, st


# ------------------------------------------------------------------ R1
def exp_recovery(seeds, epochs=4):
    L.log("\n" + "=" * 80)
    L.log("R1 - binary PubMedBERT on ReCOVery (publisher-disjoint splits)")
    L.log("=" * 80)
    frames = load_recovery()
    tr, va, te = frames
    yt = te["binary_label"].map(L.L2I).astype(int).values
    br = float((tr["binary_label"] == "NOT_CREDIBLE").mean())
    rows = []
    encs, vv, tt = [], [], []
    for sd in seeds:
        L.log(f"  seed {sd}")
        t0 = time.time()
        model, tok, sv, st = train_on(frames, {"binary"}, sd, epochs)
        m = binmetrics(yt, st["bin"], rate_thr(sv["bin"], br))
        L.log(f"    acc={m['accuracy']} macroF1={m['f1_macro']} recall={m['recall_notcred']} "
              f"auc={m['roc_auc']}  ({(time.time()-t0)/60:.1f} min)")
        rows.append({"exp": "R1_recovery", "seed": sd, **m})
        encs.append({k: v.detach().cpu().clone() for k, v in model.enc.state_dict().items()})
        vv.append(sv["bin"]); tt.append(st["bin"])
        np.save(PRED / f"REC_bin__s{sd}__test.npy", st["bin"])
        np.save(PRED / f"REC_bin__s{sd}__val.npy", sv["bin"])
        del model; torch.cuda.empty_cache()

    ev, et = np.mean(vv, 0), np.mean(tt, 0)
    m = binmetrics(yt, et, rate_thr(ev, br))
    L.log(f"  seed-ensemble  acc={m['accuracy']} macroF1={m['f1_macro']} auc={m['roc_auc']}")
    rows.append({"exp": "R1_recovery", "seed": "ENS", **m})
    df = pd.DataFrame(rows)
    L.log(f"\n  per-seed macroF1 {df[df.seed!='ENS'].f1_macro.mean():.4f} "
          f"+- {df[df.seed!='ENS'].f1_macro.std():.4f}   (TF-IDF reference 0.773, "
          f"majority baseline {max(1-br,br):.3f})")
    torch.save(encs[0], OUT / "recovery_encoder_seed0.pt")
    return rows, encs


# ------------------------------------------------------------------ R2
def exp_zeroshot():
    """Cross-domain transfer with NO target-domain training."""
    L.log("\n" + "=" * 80)
    L.log("R2 - zero-shot cross-domain transfer")
    L.log("=" * 80)
    rows = []
    fh_tr, fh_va, fh_te = L.load_splits()
    rec_tr, rec_va, rec_te = load_recovery()

    # --- FakeHealth v2 model -> ReCOVery
    d = ROOT / "models" / "tier2_final_v2"
    if d.exists():
        meta = json.load(open(d / "meta.json"))
        ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
        model = L.T2Model(ck["backbone"], set(ck["heads"])).to(L.DEVICE)
        model.load_state_dict(ck["state_dict"], strict=False)
        tok = AutoTokenizer.from_pretrained(d)
        yv = rec_va["binary_label"].map(L.L2I).astype(int).values
        yt = rec_te["binary_label"].map(L.L2I).astype(int).values
        br = float((rec_tr["binary_label"] == "NOT_CREDIBLE").mean())
        sv = L.predict(model, L.ArtDS(rec_va, tok, 512))
        st = L.predict(model, L.ArtDS(rec_te, tok, 512))
        for key in ("ord", "crit"):
            m = binmetrics(yt, st[key], rate_thr(sv[key], br))
            L.log(f"  FakeHealth-v2 ({key}) -> ReCOVery : acc={m['accuracy']} "
                  f"macroF1={m['f1_macro']} auc={m['roc_auc']}")
            rows.append({"exp": "R2_fh_to_rec", "score": key, **m})
        del model; torch.cuda.empty_cache()
    else:
        L.log("  models/tier2_final_v2 missing, skipping FakeHealth->ReCOVery")

    # --- ReCOVery model -> FakeHealth
    p = OUT / "recovery_encoder_seed0.pt"
    L.log("  (ReCOVery -> FakeHealth needs the R1 head, handled inside R1/R3)")
    return rows


# ------------------------------------------------------------------ R3
def exp_intermediate(seeds, encs, epochs=5):
    """ReCOVery-pretrained encoder -> FakeHealth v2 heads (STILT).

    Labels are never merged: ReCOVery only shapes the encoder, and the FakeHealth
    target task keeps its own ordinal+criteria heads.
    """
    L.log("\n" + "=" * 80)
    L.log("R3 - ReCOVery -> FakeHealth intermediate-task transfer")
    L.log("=" * 80)
    frames = L.load_splits()
    tr, va, te = frames
    yt = te["binary_label"].map(L.L2I).astype(int).values
    yv = va["binary_label"].map(L.L2I).astype(int).values
    br = float((tr["binary_label"] == "NOT_CREDIBLE").mean())
    rows = []
    for i, sd in enumerate(seeds):
        L.log(f"  seed {sd}")
        enc = encs[i % len(encs)]
        model, tok, sv, st = train_on(frames, {"ordinal", "criteria"}, sd, epochs,
                                      init_encoder=enc)
        comp = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
            sv["_crit_raw"], yv)
        sv["crit_lr"] = comp.predict_proba(sv["_crit_raw"])[:, 1]
        st["crit_lr"] = comp.predict_proba(st["_crit_raw"])[:, 1]
        for key in ("ord", "crit", "crit_nodm", "crit_lr"):
            m = binmetrics(yt, st[key], rate_thr(sv[key], br))
            rows.append({"exp": "R3_intermediate", "seed": sd, "score": key, **m})
            np.save(PRED / f"R3int_{key}__s{sd}__test.npy", st[key])
            np.save(PRED / f"R3int_{key}__s{sd}__val.npy", sv[key])
        L.log("    " + "  ".join(
            f"{r['score']}={r['f1_macro']:.4f}" for r in rows[-4:]))
        del model; torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    L.log("\n  per-seed mean +- std (FakeHealth test, rate-matched):")
    for k, g in df.groupby("score"):
        L.log(f"    {k:<10} macroF1={g.f1_macro.mean():.4f} +- {g.f1_macro.std():.4f}  "
              f"acc={g.accuracy.mean():.4f}  recall={g.recall_notcred.mean():.4f}  "
              f"auc={g.roc_auc.mean():.4f}")
    L.log("\n  compare with the direct FakeHealth v2 confirmation run "
          "(outputs/tier2_confirm_seeds.csv): crit_lr 0.6851 +- 0.0160")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=["recovery", "zeroshot", "intermediate", "all"])
    ap.add_argument("--seeds", default="42,1337,2024")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    L.log(f"\n{'#'*80}\nT2 TRANSFER  {datetime.now():%Y-%m-%d %H:%M:%S}  device={L.DEVICE}\n{'#'*80}")
    rows, encs = [], None
    if args.exp in ("recovery", "all", "intermediate"):
        r, encs = exp_recovery(seeds)
        rows += r
    if args.exp in ("zeroshot", "all"):
        rows += exp_zeroshot()
    if args.exp in ("intermediate", "all"):
        rows += exp_intermediate(seeds, encs)
    if rows:
        p = OUT / "tier2_transfer_results.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        L.log(f"\nwrote {p}")


if __name__ == "__main__":
    main()
