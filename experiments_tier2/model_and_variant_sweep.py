"""Tier 2 improvement lab.

WHY THIS EXISTS
---------------
Everything previously tried on Tier 2 changed the INPUT REPRESENTATION (longer
context via chunking, hand-built lexical features). Neither helped. The three
axes never tested were: decision rule, encoder, and OBJECTIVE.

The key structural observation driving this file:

    binary_label = (rating <= 2)            # rating is an ordinal 0..5
    rating       ~= (#criteria satisfied)/2 # corr(rating, frac_failed) = -0.944

So the binary label is a *coarsening* of an ordinal rating, which is itself a
sum over ~8.9 human criteria judgements per article. Training on the binary
label alone throws away ~15 supervised signals per article. With only 1,483
training articles that is the dominant bottleneck -- not the encoder.

Three heads, sharing one encoder:

  binary    2-way CE, class-weighted                    (1 signal/article)
  ordinal   CORAL: P(rating > k) for k=0..4             (5 signals/article)
            note k=2 IS the binary task, so the ordinal head strictly
            *contains* the target task and adds 4 aligned auxiliaries
  criteria  10 masked BCE heads, pos-weighted           (~8.9 signals/article)

EVALUATION
----------
The test set is 318 articles. Marginal bootstrap CIs are ~+-0.05 wide, so
comparing two overlapping marginal CIs is the wrong test and is far too
conservative. We use a PAIRED bootstrap on the difference (same resampled
articles for both systems), which cancels article-level difficulty and can
resolve differences an order of magnitude smaller.

Every variant runs on N seeds; we report mean +- std across seeds so that a
"win" cannot be an artefact of one lucky initialisation.

Usage:
    python experiments_tier2/model_and_variant_sweep.py --phase objective
    python experiments_tier2/model_and_variant_sweep.py --phase encoder
    python experiments_tier2/model_and_variant_sweep.py --phase input
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "processed_tier2"
OUT = ROOT / "outputs"
PRED = OUT / "tier2_preds"
PRED.mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(parents=True, exist_ok=True)
_fh = open(OUT / "logs" / "model_and_variant_sweep.log", "a", encoding="utf-8")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# bf16 on Blackwell/Ada. Using bf16 autocast WITHOUT a GradScaler is also the
# fix for the DeBERTa-v3 "Attempting to unscale FP16 gradients" crash that
# killed the previous run (see CHAT_TRANSFER section 12.7).
AMP_DTYPE = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
USE_SCALER = AMP_DTYPE == torch.float16
# DeBERTa-v3's disentangled attention overflows in reduced precision (it produced
# "unscale FP16 gradients" under fp16+GradScaler, and NaN logits under bf16), so
# some backbones must run in true fp32. Set per-backbone below.
FORCE_FP32 = False


def amp_on():
    return DEVICE == "cuda" and not FORCE_FP32

LABELS = ["NOT_CREDIBLE", "CREDIBLE"]
L2I = {l: i for i, l in enumerate(LABELS)}
NEG, POS = L2I["NOT_CREDIBLE"], L2I["CREDIBLE"]

CRITERIA = ["costs", "benefits", "harms", "evidence_quality", "disease_mongering",
            "independent_srcs", "alternatives", "availability", "novelty", "press_release"]
N_CRIT = len(CRITERIA)
ORD_K = 5           # cumulative cut-points for rating 0..5 -> k = 0..4
ORD_BIN_K = 2       # P(rating > 2) == P(CREDIBLE); the binary task itself

BACKBONES = {
    #  key            hf name                                                          max_len bs accum lr
    "pubmedbert":  ("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",    512, 8, 1, 2e-5),
    "deberta":     ("microsoft/deberta-v3-base",                                        512, 4, 2, 1e-5),
    "biomedlarge": ("microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",            512, 2, 4, 1e-5),
    "modernbert":  ("answerdotai/ModernBERT-base",                                     2048, 2, 4, 2e-5),
    "scibert":     ("allenai/scibert_scivocab_uncased",                                 512, 8, 1, 2e-5),
    "roberta":     ("roberta-base",                                                     512, 8, 1, 2e-5),
}

SEEDS = [42, 1337, 2024]


def log(*a):
    m = " ".join(str(x) for x in a)
    print(m, flush=True)
    _fh.write(m + "\n")
    _fh.flush()


# ------------------------------------------------------------------ data
def load_splits():
    return tuple(pd.read_csv(DATA / f"{s}.csv") for s in ("train", "validation", "test"))


def article_text(df):
    return (df["title"].fillna("").astype(str) + ". " +
            df["text"].fillna("").astype(str)).tolist()


class ArtDS(Dataset):
    """input_mode:
         head     - first max_len tokens (the existing protocol)
         headtail - first 3/4 + last 1/4. News is inverted-pyramid: the claim is
                    up top but the caveats (harms, costs, limitations, funding)
                    are usually at the bottom, which plain truncation never sees.
    """
    def __init__(self, df, tok, max_len, input_mode="head"):
        self.texts = article_text(df)
        self.tok, self.max_len, self.mode = tok, max_len, input_mode
        self.y = df["binary_label"].map(L2I).astype(int).values
        self.rating = df["rating"].astype(int).values
        self.crit = df[[f"fail_{k}" for k in CRITERIA]].fillna(-1.0).values.astype(np.float32)

    def __len__(self):
        return len(self.texts)

    def _encode(self, text):
        if self.mode == "head":
            return self.tok(text, truncation=True, max_length=self.max_len,
                            padding="max_length")
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        budget = self.max_len - 2
        if len(ids) > budget:
            n_head = int(budget * 0.75)
            ids = ids[:n_head] + ids[-(budget - n_head):]
        # transformers 5.x removed build_inputs_with_special_tokens from the
        # tokenizer base, so wrap manually (BERT-style cls/sep, RoBERTa bos/eos).
        cls = self.tok.cls_token_id if self.tok.cls_token_id is not None else self.tok.bos_token_id
        sep = self.tok.sep_token_id if self.tok.sep_token_id is not None else self.tok.eos_token_id
        ids = ([cls] if cls is not None else []) + ids + ([sep] if sep is not None else [])
        pad = self.max_len - len(ids)
        am = [1] * len(ids) + [0] * pad
        ids = ids + [self.tok.pad_token_id] * pad
        return {"input_ids": ids, "attention_mask": am}

    def __getitem__(self, i):
        e = self._encode(self.texts[i])
        r = int(self.rating[i])
        return {
            "input_ids": torch.tensor(e["input_ids"]),
            "attention_mask": torch.tensor(e["attention_mask"]),
            "labels": torch.tensor(int(self.y[i])),
            "criteria": torch.tensor(self.crit[i]),
            # CORAL targets: 1[rating > k] for k = 0..4
            "ordinal": torch.tensor([(1.0 if r > k else 0.0) for k in range(ORD_K)],
                                    dtype=torch.float32),
        }


# ------------------------------------------------------------------ model
class T2Model(nn.Module):
    """Shared encoder, up to three heads.

    ordinal head is CORAL: one shared direction w plus K-1 ordered biases, so the
    cumulative probabilities P(rating>k) are monotone in k by construction and the
    five tasks cannot contradict each other.
    """

    def __init__(self, backbone, heads, class_w=None, crit_pw=None, ord_pw=None,
                 w_bin=1.0, w_crit=0.5, w_ord=0.5, grad_ckpt=False):
        super().__init__()
        # transformers 5.x loads weights in the checkpoint's own dtype. deberta-v3-base
        # ships as fp16, which ran the encoder in pure fp16 (-> NaN logits) while the
        # heads stayed fp32 (-> "Half != float"). Always materialise in fp32 and let
        # autocast handle mixed precision.
        self.enc = AutoModel.from_pretrained(backbone, dtype=torch.float32)
        if grad_ckpt:
            self.enc.gradient_checkpointing_enable()
        h = self.enc.config.hidden_size
        self.heads = set(heads)
        self.drop = nn.Dropout(0.1)
        if "binary" in self.heads:
            self.bin_head = nn.Linear(h, 2)
        if "criteria" in self.heads:
            self.crit_head = nn.Linear(h, N_CRIT)
        if "ordinal" in self.heads:
            self.ord_dir = nn.Linear(h, 1, bias=False)
            self.ord_bias = nn.Parameter(torch.zeros(ORD_K))
        self.w_bin, self.w_crit, self.w_ord = w_bin, w_crit, w_ord
        self.register_buffer("cw", class_w if class_w is not None else torch.ones(2))
        self.register_buffer("cpw", crit_pw if crit_pw is not None else torch.ones(N_CRIT))
        self.register_buffer("opw", ord_pw if ord_pw is not None else torch.ones(ORD_K))

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                criteria=None, ordinal=None, **kw):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.drop(out.last_hidden_state[:, 0])

        res, loss = {}, None

        def add(l):
            nonlocal loss
            loss = l if loss is None else loss + l

        if "binary" in self.heads:
            bl = self.bin_head(pooled)
            res["bin_logits"] = bl
            if labels is not None:
                add(self.w_bin * nn.CrossEntropyLoss(weight=self.cw)(bl, labels.view(-1)))

        if "ordinal" in self.heads:
            ol = self.ord_dir(pooled) + self.ord_bias          # (B, K)
            res["ord_logits"] = ol
            if ordinal is not None:
                l = nn.BCEWithLogitsLoss(reduction="none", pos_weight=self.opw)(ol, ordinal)
                add(self.w_ord * l.mean())

        if "criteria" in self.heads:
            cl = self.crit_head(pooled)
            res["crit_logits"] = cl
            if criteria is not None:
                mask = (criteria >= 0).float()
                l = nn.BCEWithLogitsLoss(reduction="none", pos_weight=self.cpw)(
                    cl, criteria.clamp(min=0))
                add(self.w_crit * (l * mask).sum() / mask.sum().clamp(min=1))

        res["loss"] = loss
        return res


@torch.no_grad()
def predict(model, ds, bs=16):
    """Returns dict of credibility scores (higher = MORE credible) per head."""
    model.eval()
    bins, ords, crits = [], [], []
    for i in range(0, len(ds), bs):
        batch = [ds[j] for j in range(i, min(i + bs, len(ds)))]
        bt = {k: torch.stack([b[k] for b in batch]).to(DEVICE)
              for k in ("input_ids", "attention_mask")}
        with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=amp_on()):
            r = model(**bt)
        if "bin_logits" in r:
            bins.append(torch.softmax(r["bin_logits"].float(), -1)[:, POS].cpu().numpy())
        if "ord_logits" in r:
            ords.append(torch.sigmoid(r["ord_logits"].float()).cpu().numpy())
        if "crit_logits" in r:
            crits.append(torch.sigmoid(r["crit_logits"].float()).cpu().numpy())

    s = {}
    if bins:
        s["bin"] = np.concatenate(bins)
    if ords:
        o = np.concatenate(ords)
        # CORAL shares one direction across cut-points, so every P(rating>k) is a
        # monotone transform of the same scalar: an "expected rating" score would be
        # rank-identical to this one and is therefore not computed.
        s["ord"] = o[:, ORD_BIN_K]                       # P(rating > 2) = P(CREDIBLE)
    if crits:
        c = np.concatenate(crits)
        s["crit"] = 1.0 - c.mean(1)                      # 1 - expected failure fraction
        # disease_mongering has validation MCC 0.000 (zero discrimination), so
        # averaging it in only injects noise. Drop it from the pooled score.
        keep = [j for j, k in enumerate(CRITERIA) if k != "disease_mongering"]
        s["crit_nodm"] = 1.0 - c[:, keep].mean(1)
        s["_crit_raw"] = c
    return s


def train_one(model, tr_ds, va_ds, y_val, epochs, lr, bs, accum, tag):
    loader = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=0)
    steps = max(1, (len(loader) // accum) * epochs)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_SCALER and amp_on())

    best, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        run, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=amp_on()):
                loss = model(**batch)["loss"]
            if USE_SCALER and amp_on():
                scaler.scale(loss / accum).backward()
            else:
                (loss / accum).backward()
            if (step + 1) % accum == 0 or step == len(loader) - 1:
                if USE_SCALER and amp_on():
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if USE_SCALER and amp_on():
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
            run += float(loss); nb += 1

        sv = predict(model, va_ds)
        key = "bin" if "bin" in sv else ("ord" if "ord" in sv else "crit")
        f = f1_score(y_val, (sv[key] >= 0.5).astype(int), average="macro", zero_division=0)
        star = ""
        if f > best:
            best, star = f, "  <- best"
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log(f"      ep{ep} loss={run/max(nb,1):.4f} val_macroF1({key})={f:.4f}{star}")
    if best_state:
        model.load_state_dict(best_state)
    return model


# ------------------------------------------------------------------ metrics
def metrics_at(y, score_cred, thr):
    """score_cred: higher = more credible. Predict NOT_CREDIBLE when below thr."""
    pred = np.where(score_cred >= thr, POS, NEG)
    rec = precision_recall_fscore_support(y, pred, average="binary", pos_label=NEG,
                                          zero_division=0)[1]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "recall_notcred": float(rec),
    }, pred


def tune_thr(y, s):
    best_t, best = 0.5, -1.0
    for t in np.unique(np.round(np.linspace(0.02, 0.98, 97), 4)):
        f = metrics_at(y, s, t)[0]["f1_macro"]
        if f > best:
            best, best_t = f, float(t)
    return best_t, best


def boot_ci(y, pred, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); v = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        v.append(f1_score(y[b], pred[b], average="macro", zero_division=0))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def paired_boot(y, pred_a, pred_b, n=5000, seed=0):
    """CI on macroF1(a) - macroF1(b) using the SAME resampled articles for both.

    This is the correct test for comparing two systems on one test set. It is far
    more sensitive than checking whether two marginal CIs overlap, because
    article-level difficulty cancels in the difference.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); d = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        d.append(f1_score(y[b], pred_a[b], average="macro", zero_division=0) -
                 f1_score(y[b], pred_b[b], average="macro", zero_division=0))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), \
        float((d <= 0).mean())


# ------------------------------------------------------------------ runner
def class_weights(tr):
    return torch.tensor(
        compute_class_weight("balanced", classes=np.array([0, 1]),
                             y=tr["binary_label"].map(L2I).values),
        dtype=torch.float32)


def crit_pos_weight(tr):
    ct = tr[[f"fail_{k}" for k in CRITERIA]].fillna(-1.0).values.astype(np.float32)
    valid = ct >= 0
    rate = np.array([ct[:, j][valid[:, j]].mean() for j in range(N_CRIT)])
    return torch.tensor(((1 - rate) / np.clip(rate, 1e-3, None)).astype(np.float32))


def ord_pos_weight(tr):
    r = tr["rating"].astype(int).values
    w = []
    for k in range(ORD_K):
        pos = (r > k).sum(); neg = len(r) - pos
        w.append(neg / max(pos, 1))
    return torch.tensor(np.array(w, dtype=np.float32))


FP32_BACKBONES = {"deberta"}


def run_variant(name, backbone_key, heads, input_mode="head", epochs=4,
                w_bin=1.0, w_crit=0.5, w_ord=0.5, seeds=None, grad_ckpt=False):
    global FORCE_FP32
    FORCE_FP32 = backbone_key in FP32_BACKBONES
    bb, max_len, bs, accum, lr = BACKBONES[backbone_key]
    seeds = seeds or SEEDS
    tr, va, te = load_splits()
    yv = va["binary_label"].map(L2I).astype(int).values
    yt = te["binary_label"].map(L2I).astype(int).values

    tok = AutoTokenizer.from_pretrained(bb)
    tr_ds = ArtDS(tr, tok, max_len, input_mode)
    va_ds = ArtDS(va, tok, max_len, input_mode)
    te_ds = ArtDS(te, tok, max_len, input_mode)

    cw, cpw, opw = class_weights(tr).to(DEVICE), crit_pos_weight(tr).to(DEVICE), ord_pos_weight(tr).to(DEVICE)

    log("\n" + "=" * 78)
    log(f"VARIANT {name}   backbone={backbone_key} heads={sorted(heads)} "
        f"input={input_mode} max_len={max_len} bs={bs}x{accum} lr={lr} epochs={epochs}")
    log("=" * 78)

    rows, per_seed_scores = [], []
    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd)
        if DEVICE == "cuda":
            torch.cuda.manual_seed_all(sd)
        log(f"  seed {sd}")
        t0 = time.time()
        model = T2Model(bb, heads, cw, cpw, opw, w_bin, w_crit, w_ord, grad_ckpt).to(DEVICE)
        model = train_one(model, tr_ds, va_ds, yv, epochs, lr, bs, accum, name)
        mins = (time.time() - t0) / 60

        sv, st = predict(model, va_ds), predict(model, te_ds)
        bad = [k for k, v in st.items() if not k.startswith("_") and not np.isfinite(v).all()]
        if bad:
            log(f"    !! non-finite scores in {bad}; skipping seed {sd}")
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            continue
        per_seed_scores.append({"seed": sd, "val": sv, "test": st})

        for key in [k for k in ("bin", "ord", "crit", "crit_nodm") if k in st]:
            thr, _ = tune_thr(yv, sv[key])
            m, pred = metrics_at(yt, st[key], thr)
            auc = roc_auc_score(yt, st[key])
            rows.append({"variant": name, "backbone": backbone_key, "heads": "+".join(sorted(heads)),
                         "input": input_mode, "seed": sd, "score": key, "threshold": round(thr, 3),
                         "train_minutes": round(mins, 2), "roc_auc": round(auc, 4), **
                         {k: round(v, 4) for k, v in m.items()}})
            np.save(PRED / f"{name}__{key}__s{sd}__test.npy", st[key])
            np.save(PRED / f"{name}__{key}__s{sd}__val.npy", sv[key])

        # the 10 individual criteria probabilities are far more informative than
        # their mean: the criteria differ enormously in reliability (costs F1
        # 0.811 vs disease_mongering 0.108), so a learned weighting should beat
        # an unweighted average. Composer is fitted on validation only.
        if "_crit_raw" in st:
            np.save(PRED / f"{name}__critraw__s{sd}__test.npy", st["_crit_raw"])
            np.save(PRED / f"{name}__critraw__s{sd}__val.npy", sv["_crit_raw"])
            comp = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
                sv["_crit_raw"], yv)
            sv["crit_lr"] = comp.predict_proba(sv["_crit_raw"])[:, 1]
            st["crit_lr"] = comp.predict_proba(st["_crit_raw"])[:, 1]
            thr, _ = tune_thr(yv, sv["crit_lr"])
            m, _ = metrics_at(yt, st["crit_lr"], thr)
            rows.append({"variant": name, "backbone": backbone_key,
                         "heads": "+".join(sorted(heads)), "input": input_mode,
                         "seed": sd, "score": "crit_lr", "threshold": round(thr, 3),
                         "train_minutes": round(mins, 2),
                         "roc_auc": round(roc_auc_score(yt, st["crit_lr"]), 4),
                         **{k: round(v, 4) for k, v in m.items()}})
            np.save(PRED / f"{name}__crit_lr__s{sd}__test.npy", st["crit_lr"])
            np.save(PRED / f"{name}__crit_lr__s{sd}__val.npy", sv["crit_lr"])

        # blended decision variable, fitted on validation only
        keys = [k for k in ("bin", "ord", "crit", "crit_nodm", "crit_lr") if k in st]
        if len(keys) > 1:
            Xv = np.column_stack([sv[k] for k in keys])
            Xt = np.column_stack([st[k] for k in keys])
            lrm = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xv, yv)
            s_blend_v = lrm.predict_proba(Xv)[:, 1]
            s_blend_t = lrm.predict_proba(Xt)[:, 1]
            thr, _ = tune_thr(yv, s_blend_v)
            m, pred = metrics_at(yt, s_blend_t, thr)
            rows.append({"variant": name, "backbone": backbone_key, "heads": "+".join(sorted(heads)),
                         "input": input_mode, "seed": sd, "score": "blend", "threshold": round(thr, 3),
                         "train_minutes": round(mins, 2),
                         "roc_auc": round(roc_auc_score(yt, s_blend_t), 4),
                         **{k: round(v, 4) for k, v in m.items()}})
            np.save(PRED / f"{name}__blend__s{sd}__test.npy", s_blend_t)
            np.save(PRED / f"{name}__blend__s{sd}__val.npy", s_blend_v)

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # seed ensemble: average the score vectors across seeds
    for key in [k for k in ("bin", "ord", "crit", "crit_nodm", "crit_lr") if k in per_seed_scores[0]["test"]]:
        ev = np.mean([p["val"][key] for p in per_seed_scores], 0)
        et = np.mean([p["test"][key] for p in per_seed_scores], 0)
        thr, _ = tune_thr(yv, ev)
        m, pred = metrics_at(yt, et, thr)
        rows.append({"variant": name, "backbone": backbone_key, "heads": "+".join(sorted(heads)),
                     "input": input_mode, "seed": "ENS", "score": key, "threshold": round(thr, 3),
                     "train_minutes": None, "roc_auc": round(roc_auc_score(yt, et), 4),
                     **{k: round(v, 4) for k, v in m.items()}})
        np.save(PRED / f"{name}__{key}__sENS__test.npy", et)
        np.save(PRED / f"{name}__{key}__sENS__val.npy", ev)

    df = pd.DataFrame(rows)
    agg = (df[df.seed != "ENS"].groupby("score")[["accuracy", "f1_macro", "recall_notcred", "roc_auc"]]
           .agg(["mean", "std"]).round(4))
    log("\n  per-seed mean +- std:")
    for key in agg.index:
        r = agg.loc[key]
        log(f"    {key:<9} acc={r[('accuracy','mean')]:.4f}+-{r[('accuracy','std')]:.4f}  "
            f"macroF1={r[('f1_macro','mean')]:.4f}+-{r[('f1_macro','std')]:.4f}  "
            f"recall={r[('recall_notcred','mean')]:.4f}+-{r[('recall_notcred','std')]:.4f}  "
            f"auc={r[('roc_auc','mean')]:.4f}+-{r[('roc_auc','std')]:.4f}")
    ens = df[df.seed == "ENS"]
    log("  seed-ensemble:")
    for _, r in ens.iterrows():
        log(f"    {r['score']:<9} acc={r['accuracy']:.4f}  macroF1={r['f1_macro']:.4f}  "
            f"recall={r['recall_notcred']:.4f}  auc={r['roc_auc']:.4f}")
    return rows


PHASES = {
    # objective axis, all on pubmedbert / 512 / head -- protocol matched to the notebook
    "objective": [
        dict(name="A_binary",        backbone_key="pubmedbert", heads={"binary"}),
        dict(name="B_ordinal",       backbone_key="pubmedbert", heads={"ordinal"}),
        dict(name="C_criteria",      backbone_key="pubmedbert", heads={"criteria"}, epochs=5),
        dict(name="D_bin_crit",      backbone_key="pubmedbert", heads={"binary", "criteria"}, epochs=5),
        dict(name="E_ord_crit",      backbone_key="pubmedbert", heads={"ordinal", "criteria"}, epochs=5),
        dict(name="F_all_heads",     backbone_key="pubmedbert", heads={"binary", "ordinal", "criteria"}, epochs=5),
    ],
    "input": [
        dict(name="G_headtail",      backbone_key="pubmedbert", heads={"ordinal", "criteria"},
             input_mode="headtail", epochs=5),
    ],
    "encoder": [
        dict(name="H_deberta",       backbone_key="deberta",     heads={"ordinal", "criteria"}, epochs=5),
        dict(name="I_modernbert",    backbone_key="modernbert",  heads={"ordinal", "criteria"}, epochs=5),
        dict(name="J_biomedlarge",   backbone_key="biomedlarge", heads={"ordinal", "criteria"},
             epochs=5, grad_ckpt=True),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="objective")
    ap.add_argument("--only", default=None, help="comma-separated variant names")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS
    specs = PHASES[args.phase]
    if args.only:
        keep = set(args.only.split(","))
        specs = [s for s in specs if s["name"] in keep]

    log(f"\n{'#'*78}\nT2 LAB phase={args.phase}  {datetime.now():%Y-%m-%d %H:%M:%S}  "
        f"device={DEVICE} amp={AMP_DTYPE}\n{'#'*78}")

    all_rows = []
    for spec in specs:
        try:
            all_rows += run_variant(seeds=seeds, **spec)
        except Exception as e:
            import traceback
            log(f"  !! {spec['name']} FAILED: {type(e).__name__}: {str(e)[:300]}")
            log(traceback.format_exc()[-1500:])
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    if all_rows:
        df = pd.DataFrame(all_rows)
        p = OUT / f"tier2_lab_{args.phase}.csv"
        if p.exists():
            df = pd.concat([pd.read_csv(p), df], ignore_index=True)
        df.to_csv(p, index=False)
        log(f"\nwrote {p}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
