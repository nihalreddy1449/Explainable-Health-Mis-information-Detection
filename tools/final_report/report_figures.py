"""Every data figure in the final report, drawn from saved results in one style.

Black and grey only, Computer Modern text to match the LaTeX body, vector PDF
output (plus a PNG copy for quick checking). Diagrams are drawn in TikZ inside
the report, not here.

    python tools/final_report/report_figures.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tier2 import explanation_checks as FA
from tier2.engine import CRITERIA, CRITERION_NAME, article_text

OUT = ROOT / "Presentation" / "Final Review" / "Report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DATA1, DATA2 = ROOT / "datasets" / "processed_tier1", ROOT / "datasets" / "processed_tier2"
OUTP = ROOT / "outputs"

K, G1, G2, G3 = "#111111", "#555555", "#999999", "#d0d0d0"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5, "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#444",
    "axes.linewidth": 0.7, "xtick.color": "#333", "ytick.color": "#333",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
SHORT = {"costs": "Costs", "benefits": "Benefits", "harms": "Harms", "evidence_quality": "Evidence quality",
         "disease_mongering": "Disease mongering", "independent_srcs": "Independent sources",
         "alternatives": "Alternatives", "availability": "Availability", "novelty": "Novelty",
         "press_release": "Beyond the press release"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=170)
    plt.close(fig)
    print("wrote", name)


def bar_labels(ax, bars, fmt="{:.3f}", dy=0.012, size=8):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt.format(b.get_height()),
                ha="center", va="bottom", fontsize=size, color="#222")


# ------------------------------------------------------------------ data
t1 = {s: pd.read_csv(DATA1 / f"{s}.csv") for s in ("train", "validation", "test")}
t2 = {s: pd.read_csv(DATA2 / f"{s}.csv") for s in ("train", "validation", "test")}
m1 = json.load(open(OUTP / "tier1_metrics.json"))
extra = json.load(open(OUTP / "final_report_extra.json"))

# 1. Tier 1 data overview
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.4), gridspec_kw={"width_ratios": [1.1, 1]})
labs, names = ["SUPPORTED", "REFUTED", "NEI"], ["Supported", "Refuted", "Not enough info"]
x = np.arange(3); w = 0.26
for k, (s, c) in enumerate(zip(("train", "validation", "test"), (K, G2, G3))):
    v = t1[s].label.value_counts().reindex(labs).values
    axes[0].bar(x + (k - 1) * w, v, w, color=c, edgecolor="white", linewidth=0.5,
                label={"train": "Train", "validation": "Validation", "test": "Test"}[s])
axes[0].set_xticks(x, names); axes[0].set_ylabel("claim and evidence pairs")
axes[0].set_ylim(0, 5200)
axes[0].legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=0.8)
axes[0].set_title("(a) Labels in each split", loc="left")
cl = pd.concat(t1.values()).drop_duplicates("claim").claim.str.split().str.len()
axes[1].hist(cl.clip(upper=60), bins=30, color=G1, edgecolor="white", linewidth=0.4)
axes[1].set_xlabel("words per claim"); axes[1].set_ylabel("claims")
axes[1].set_title(f"(b) Claim length, median {int(cl.median())} words", loc="left")
save(fig, "fig_tier1_data")

# 2. Tier 2 data overview
d2 = pd.concat(t2.values(), ignore_index=True)
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.75})
rc = d2.rating.value_counts().reindex(range(6), fill_value=0)
bars = axes[0].bar(rc.index, rc.values, 0.7, color=[G1 if r <= 2 else G3 for r in rc.index],
                   edgecolor="#333", linewidth=0.5)
bar_labels(axes[0], bars, "{:.0f}", dy=8)
axes[0].axvline(2.5, color=K, ls="--", lw=0.8)
axes[0].text(1.0, rc.max() * 1.18, "not credible", ha="center", fontsize=8)
axes[0].text(4.0, rc.max() * 1.18, "credible", ha="center", fontsize=8)
axes[0].set_xlabel("HealthNewsReview star rating"); axes[0].set_ylabel("articles")
axes[0].set_ylim(0, rc.max() * 1.3); axes[0].set_title("(a) Ratings of 2,119 articles", loc="left")
fr = pd.Series({SHORT[k]: t2["train"][f"fail_{k}"].mean() for k in CRITERIA}).sort_values()
axes[1].barh(fr.index, fr.values, 0.62, color=G1)
for yv, v in enumerate(fr.values):
    axes[1].text(v + 0.01, yv, f"{v:.2f}", va="center", fontsize=7.5)
axes[1].set_xlim(0, 0.9); axes[1].set_xlabel("share of graded articles that fail")
axes[1].set_title("(b) Criterion failure rates (training set)", loc="left")
save(fig, "fig_tier2_data")

# 3. Word counts: claims against articles (router)
aw = pd.Series([len(article_text(a, b).split()) for a, b in zip(d2.title, d2.text)])
fig, ax = plt.subplots(figsize=(6.0, 2.3))
bins = np.logspace(np.log10(2), np.log10(5000), 60)
ax.hist(cl, bins=bins, color=G1, alpha=0.95, label=f"Tier 1 claims (n = {len(cl):,})")
ax.hist(aw, bins=bins, color=G3, edgecolor="#777", linewidth=0.3, label=f"Tier 2 articles (n = {len(aw):,})")
ax.axvline(60, color=K, ls="--", lw=0.9)
ax.text(64, ax.get_ylim()[1] * 0.9, "60 words", fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("words (log scale)"); ax.set_ylabel("texts"); ax.legend(loc="upper right")
save(fig, "fig_lengths")

# 4. Confidence gate on all held-out test claims
s = np.array(extra["gate_similarities"])
fig, ax = plt.subplots(figsize=(5.6, 2.3))
ax.hist(s, bins=np.linspace(0.2, 1.0, 41), color=G1, edgecolor="white", linewidth=0.4)
ax.axvspan(0.2, 0.60, color=G3, alpha=0.35, lw=0)
ax.axvline(0.60, color=K, ls="--", lw=0.9)
ax.text(0.585, ax.get_ylim()[1] * 0.88, f"declined {1 - extra['gate_answered_share']:.1%}", ha="right", fontsize=8)
ax.text(0.615, ax.get_ylim()[1] * 0.88, f"answered {extra['gate_answered_share']:.1%}", ha="left", fontsize=8)
ax.set_xlabel("best cosine similarity between the claim and any corpus passage"); ax.set_ylabel("test claims")
save(fig, "fig_gate")

# 5. Tier 1 model comparison
r = m1["results"]
rows = [("PubMedBERT\n+ evidence", r["tier1_pubmedbert_evidence"]), ("BERT base\n+ evidence", r["tier1_bertbase_evidence"]),
        ("PubMedBERT\nclaim only", r["tier1_pubmedbert_claim_only"])]
fig, ax = plt.subplots(figsize=(5.4, 2.6))
x = np.arange(3); w = 0.34
b1 = ax.bar(x - w / 2, [v["accuracy"] for _, v in rows], w, color=G3, edgecolor="#555", linewidth=0.5, label="Accuracy")
b2 = ax.bar(x + w / 2, [v["f1_macro"] for _, v in rows], w, color=K, label="Macro F1")
bar_labels(ax, b1); bar_labels(ax, b2)
ax.axhline(m1["majority_baseline"], color=G1, ls="--", lw=0.8, label=f"Majority class ({m1['majority_baseline']:.3f})")
ax.set_xticks(x, [n for n, _ in rows]); ax.set_ylim(0, 1.0); ax.set_ylabel("score on 2,165 test pairs")
ax.legend(loc="upper right", ncol=3, fontsize=7.8, handlelength=1.4, columnspacing=1.0)
save(fig, "fig_tier1_models")

# 6. Retrieval
ks = [1, 3, 5, 10]
fig, ax = plt.subplots(figsize=(5.0, 2.6))
for name, lab, sty in (("bm25", "BM25", dict(color=G2, ls=":", marker="s")),
                       ("dense", "Dense (PubMedBERT embeddings)", dict(color=G1, ls="--", marker="^")),
                       ("hybrid", "Hybrid (reciprocal rank fusion)", dict(color=K, ls="-", marker="o"))):
    v = [m1["retrieval"][name][f"recall@{k}"] for k in ks]
    ax.plot(ks, v, lw=1.3, ms=4.5, label=lab, **sty)
    ax.text(10.25, v[-1], f"{v[-1]:.3f}", va="center", fontsize=7.5)
ax.set_xticks(ks); ax.set_xlabel("passages retrieved (k)"); ax.set_ylabel("recall@k (150 test claims)")
ax.set_ylim(0, 0.9); ax.set_xlim(0.5, 11.4); ax.legend(loc="lower right")
save(fig, "fig_retrieval")


def cm_plot(cm, labels, name, title, size=(3.2, 2.8)):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=size)
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap="Greys", vmin=0, vmax=1.25)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.0%})", ha="center", va="center", fontsize=8.5,
                    color="white" if norm[i, j] > 0.62 else "#111")
    ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true label"); ax.set_title(title, fontsize=9.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    save(fig, name)


# 7. Tier 1 confusion
cm_plot(extra["tier1_confusion"], ["Supported", "Refuted", "NEI"], "fig_tier1_cm",
        "PubMedBERT + evidence, 2,165 test pairs", size=(3.6, 3.0))

# 8. Faithfulness lift per claim
f = pd.read_csv(OUTP / "tier1_faithfulness.csv")
fig, ax = plt.subplots(figsize=(5.0, 2.3))
ax.hist(f.lift, bins=np.linspace(-0.45, 0.65, 23), color=G1, edgecolor="white", linewidth=0.4)
ax.axvline(0, color=K, lw=0.9)
ax.axvline(f.lift.mean(), color=K, ls="--", lw=0.9)
ax.text(f.lift.mean() + 0.01, ax.get_ylim()[1] * 0.92, f"mean +{f.lift.mean():.3f}", fontsize=8)
ax.set_xlabel("share of SHAP words in the rationale, minus chance"); ax.set_ylabel("claims")
save(fig, "fig_faithfulness")

# ------------------------------------------------------------------ Tier 2
meta = json.load(open(ROOT / "models" / "tier2_v2_ensemble" / "meta.json"))
te = t2["test"]
y = (te.binary_label == "CREDIBLE").astype(int).values
st = np.load(OUTP / "tier2_preds" / "ENSV2__crit_lr__sENS__test.npy")
pred = (st >= meta["verdict_threshold"]).astype(int)

# 9. Tier 2 confusion (rows: not credible, credible)
from sklearn.metrics import confusion_matrix
cm_plot(confusion_matrix(y, pred, labels=[0, 1]), ["Not credible", "Credible"], "fig_tier2_cm",
        "Tier 2 verdict, 318 test articles")

# 10. criteria AUC, v2 heads against hybrid (and v1)
hyb = pd.read_csv(OUTP / "tier2_hybrid_criteria.csv").pivot(index="criterion", columns="system", values="auc")
t = hyb.loc[CRITERIA, ["v2", "hybrid"]].copy(); t.index = [SHORT[k] for k in CRITERIA]
t = t.sort_values("hybrid")
fig, ax = plt.subplots(figsize=(5.6, 3.0))
yy = np.arange(len(t))
ax.hlines(yy, t.v2, t.hybrid, color=G3, lw=2.2)
ax.scatter(t.v2, yy, s=26, facecolor="white", edgecolor=G1, zorder=3, label="Shared criteria heads (v2)")
ax.scatter(t.hybrid, yy, s=26, color=K, zorder=3, label="Hybrid with TF-IDF (shown)")
ax.axvline(0.5, color=G2, lw=0.8, ls=":")
ax.set_yticks(yy, t.index); ax.set_xlim(0.48, 0.8); ax.set_xlabel("test AUC (0.5 = chance)")
ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=2)
save(fig, "fig_criteria_auc")

# 11. criteria correlation: reviewers, v2 heads, hybrid
raw_v2 = np.mean([np.load(OUTP / "tier2_preds" / f"ENSV2__critraw__s{s_}__test.npy") for s_ in meta["seeds"]], 0)
import joblib
_h = joblib.load(ROOT / "models" / "tier2_v2_ensemble" / "criteria_hybrid.joblib")
hz = 1 / (1 + np.exp(-(_h["platt_a"] * np.load(OUTP / "tier2_preds" / "HYB__criteria__test.npy") + _h["platt_b"])))
truth = te[[f"fail_{k}" for k in CRITERIA]].astype(float)
mats = [("(a) Reviewers' labels", truth.corr().values), ("(b) Shared heads (v2)", np.corrcoef(raw_v2.T)),
        ("(c) Hybrid (shown)", np.corrcoef(hz.T))]
fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.4))
for ax, (title, M) in zip(axes, mats):
    im = ax.imshow(M, cmap="Greys", vmin=0, vmax=1)
    off = M[~np.eye(10, dtype=bool)].mean()
    ax.set_title(f"{title}\nmean off-diagonal {off:.2f}", fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02); cb.ax.tick_params(labelsize=7)
cb.set_label("correlation", fontsize=8)
save(fig, "fig_criteria_corr")

# 12. accuracy by star rating
acc = pd.DataFrame({"rating": te.rating, "ok": pred == y}).groupby("rating").ok.agg(["mean", "size"])
fig, ax = plt.subplots(figsize=(4.8, 2.3))
bars = ax.bar(acc.index, acc["mean"], 0.65, color=[G1 if r <= 2 else G3 for r in acc.index], edgecolor="#333", linewidth=0.5)
for b, n in zip(bars, acc["size"]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.2f}\nn={n}", ha="center", fontsize=7.5)
ax.axvline(2.5, color=K, ls="--", lw=0.8)
ax.set_ylim(0, 1.2); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("reviewers' star rating"); ax.set_ylabel("verdict accuracy")
save(fig, "fig_tier2_by_rating")

# 13. deletion test and 14. reviewer agreement
res = FA.load()
dl = FA.deletion_table(res)
fig, ax = plt.subplots(figsize=(5.6, 2.2))
ax.boxplot([dl.drop_pos, dl.random_mean, dl.drop_neg], vert=False, widths=0.5, showfliers=False,
           medianprops={"color": K, "lw": 1.2}, boxprops={"color": G1}, whiskerprops={"color": G1},
           capprops={"color": G1})
ax.set_yticks([1, 2, 3], ["remove the 3 sentences\nSHAP credits", "remove 3 random\nsentences",
                          "remove the 3 sentences\nSHAP blames"])
ax.axvline(0, color=G2, lw=0.8)
ax.set_xlabel(f"change in credibility score ({len(dl)} test articles)")
save(fig, "fig_deletion")

ag = FA.reviewer_agreement(res, te)
per = ag.groupby("criterion").agg(n=("rank_auc", "size"), auc=("rank_auc", "mean")).sort_values("auc")
fig, ax = plt.subplots(figsize=(5.4, 2.6))
ax.barh([SHORT[k] for k in per.index], per.auc - 0.5, left=0.5, height=0.6,
        color=[K if a > 0.5 else G3 for a in per.auc], edgecolor="#555", linewidth=0.4)
for yv, (a, n) in enumerate(zip(per.auc, per.n)):
    ax.text(max(a, 0.5) + 0.006, yv, f"{a:.2f}  (n={n})", va="center", fontsize=7.5)
ax.axvline(0.5, color=K, lw=0.8)
ax.set_xlim(0.3, 0.78); ax.set_xlabel("rank AUC of reviewer-quoted sentences (0.5 = chance)")
save(fig, "fig_reviewer_agreement")
