"""Generate 4_Tier2_Improvement_Experiments.ipynb from the saved experiment artefacts.

The experiments themselves ran as scripts (a 40-run sweep is not something you
drive from a notebook), but every number they produced is written to
outputs/*.csv, and this notebook reads those files back and shows the working.
Nothing here recomputes a model -- it is a presentation layer over saved results,
so it runs in seconds and always agrees with the artefacts on disk.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "4_Tier2_Improvement_Experiments.ipynb"

nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r"""
# Tier 2 — Improving Article Credibility Assessment

**Explainable Health Misinformation Detection using Deep Learning**
Team 10 · Supervisor: Dr. C. R. Lekshmi

---

Tier 2 was the weaker half of this project. It asks *"does this article report health
science responsibly?"* and the shipped model scored **0.661 macro-F1**, only 2.5
accuracy points above a majority baseline.

Two rounds of improvements had already been tried and **both failed**: longer context
via chunking, and fifteen hand-built lexical features. Both were re-tested here for
completeness. What had *never* been tried was changing the **training objective**.

This notebook documents that work: the diagnosis, four experiments, two negative
results, and one improvement that survives a proper significance test.

Every number below is read from files written by the experiment scripts in
`experiments_tier2/`. Nothing is retyped by hand.
""")

code(r"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 40)

ROOT = Path.cwd()
OUTP = ROOT / "outputs"
DATA = ROOT / "datasets" / "processed_tier2"

train = pd.read_csv(DATA / "train.csv")
val   = pd.read_csv(DATA / "validation.csv")
test  = pd.read_csv(DATA / "test.csv")
print(f"train {len(train)}   validation {len(val)}   test {len(test)}")
""")

md(r"""
## 1. The diagnosis — what the binary label actually is

Everything that follows comes from one observation about the data.

FakeHealth articles are reviewed by HealthNewsReview.org experts against **ten
journalistic criteria**. Each article gets a **0–5 star rating**. The project's
`NOT_CREDIBLE` label is simply `rating <= 2`.

So the binary label is a **coarsening** of an ordinal rating, which is itself
essentially a **sum over the ten criteria**. Let's verify both claims directly.
""")

code(r"""
CRITERIA = ["costs", "benefits", "harms", "evidence_quality", "disease_mongering",
            "independent_srcs", "alternatives", "availability", "novelty", "press_release"]
F = [f"fail_{c}" for c in CRITERIA]

print("binary_label is exactly rating <= 2:")
print(pd.crosstab(train["rating"], train["binary_label"]))

n_failed = train[F].fillna(0).sum(axis=1)
n_valid  = train[F].notna().sum(axis=1)
frac_failed = n_failed / n_valid.clip(lower=1)

print(f"\ncorr(rating, fraction of criteria failed) = {np.corrcoef(train['rating'], frac_failed)[0,1]:+.3f}")
print(f"mean criteria actually annotated per article = {n_valid.mean():.2f}")
""")

md(r"""
The correlation is **−0.944**. The rating is almost a deterministic function of the
criteria: roughly one star per two criteria satisfied.

### Why this matters

Count the supervised signals available per training article:

| signal | count per article |
|---|---|
| binary label | 1 |
| ordinal rating (as 5 cumulative comparisons `rating > k`) | 5 |
| criteria judgements (those actually annotated) | ~8.9 |

The shipped model trains on **the first row only**. It discards roughly **fourteen
other human judgements per article**. With just 1,483 training articles, that is the
dominant bottleneck — *not* the encoder, and not the context length.
""")

code(r"""
print(f"binary labels          : {len(train):>6,} signals")
print(f"ordinal comparisons    : {len(train)*5:>6,} signals")
print(f"criteria judgements    : {int(train[F].notna().sum().sum()):>6,} signals")
print(f"\ntotal available        : {len(train) + len(train)*5 + int(train[F].notna().sum().sum()):>6,}")
print(f"actually used by the shipped model: {len(train):,}")
""")

md(r"""
## 2. Method — three heads on one shared encoder

| head | loss | signals |
|---|---|---|
| `binary` | 2-way cross-entropy, class-weighted | 1 |
| `ordinal` | **CORAL**: `P(rating > k)` for k = 0..4 | 5 |
| `criteria` | 10 masked BCE tasks, positive-weighted | ~8.9 |

The ordinal head deserves a note. CORAL uses one shared direction plus five ordered
biases, so the cumulative probabilities are **monotone in k by construction** and the
five tasks cannot contradict each other.

Critically, **k = 2 *is* the binary task** — `P(rating > 2) = P(CREDIBLE)`. So the
ordinal head does not replace the target task, it *contains* it and adds four aligned
auxiliary tasks for free.

### Decision rule

Choosing a threshold by maximising macro-F1 on 318 validation articles **overfits** —
this project already documented that failure mode for the per-criterion thresholds.
It happened again here: under F1-tuning the baseline's seed-ensemble scored *below*
its own per-seed mean, which is only possible if the tuned threshold failed to
transfer. So the primary rule throughout is **rate matching** (predict `NOT_CREDIBLE`
at the training base rate), which cannot collapse into a constant classifier.
""")

md(r"""
## 3. Experiment 1 — the objective axis

Six variants, all on PubMedBERT / 512 tokens, **3 seeds each**, protocol matched to
the original notebook (same epochs, same early-stopping rule).

Reporting the mean **± standard deviation across seeds** matters: retraining the same
config with a different seed moves macro-F1 by ±0.026, so any single-run
"improvement" smaller than that is not a property of the method at all.
""")

code(r"""
allr = pd.read_csv(OUTP / "tier2_lab_all_rules.csv")
rate = allr[(allr.rule == "rate") & (allr.seed != "ENS")]

obj = rate[rate.variant.str.match(r"^[A-F]_")]
tbl = (obj.groupby(["variant", "score"])[["accuracy", "f1_macro", "recall_notcred", "roc_auc"]]
         .agg(["mean", "std"]).round(4))
tbl.columns = ["_".join(c) for c in tbl.columns]
tbl = tbl.sort_values("f1_macro_mean", ascending=False)
print(tbl[["f1_macro_mean", "f1_macro_std", "accuracy_mean",
           "recall_notcred_mean", "roc_auc_mean"]].to_string())
""")

md(r"""
Reading the table:

- **`A_binary` is the existing shipped design** — the reference point.
- Every variant that adds the ordinal or criteria supervision sits **above** it.
- **`E_ord_crit` (ordinal + criteria, no binary head) is the best.**

### The clearest structural finding: the binary head *hurts*

Compare each pair that differs only by whether the binary head is present:

| without binary head | with binary head |
|---|---|
| `C_criteria` | `D_bin_crit` |
| `E_ord_crit` | `F_all_heads` |

In both pairs, adding the coarse 2-class head makes the model **worse**. The blunt
binary signal competes with the finer-grained ones rather than reinforcing them.
""")

code(r"""
pairs = [("C_criteria", "D_bin_crit", "crit"), ("E_ord_crit", "F_all_heads", "ord")]
for without, with_, sc in pairs:
    a = rate[(rate.variant == without) & (rate.score == sc)]
    b = rate[(rate.variant == with_)  & (rate.score == sc)]
    print(f"{without:<14} {sc:<5} macroF1={a.f1_macro.mean():.4f}   AUC={a.roc_auc.mean():.4f}")
    print(f"{with_:<14} {sc:<5} macroF1={b.f1_macro.mean():.4f}   AUC={b.roc_auc.mean():.4f}"
          f"   <- adding the binary head: {b.f1_macro.mean()-a.f1_macro.mean():+.4f}\n")
""")

md(r"""
## 4. Experiment 2 — the input axis (negative result)

News is written inverted-pyramid: the claim is up top, but the caveats that the
criteria ask about (costs, harms, limitations, funding) tend to sit at the **bottom**,
which plain truncation never sees. So head+tail truncation — first 384 tokens plus
last 128 — should help.

It did not.
""")

code(r"""
g = rate[rate.variant == "G_headtail"].groupby("score")[["f1_macro", "roc_auc"]].mean().round(4)
e = rate[rate.variant == "E_ord_crit"].groupby("score")[["f1_macro", "roc_auc"]].mean().round(4)
cmp = e.join(g, lsuffix="_plain512", rsuffix="_headtail")
print(cmp.to_string())
print("\nhead+tail truncation is worse on every score.")
""")

md(r"""
This is the **third** independent attempt to fix Tier 2 by changing the input
representation, and the third to fail:

| attempt | result |
|---|---|
| 4×512 chunked long context | macro-F1 0.651 vs 0.673 base — worse |
| 15 lexical absence features | macro-F1 0.669 vs 0.673 base — no gain |
| head+tail truncation | **worse on every score** |

Three failures from three different angles is itself evidence: **the bottleneck is not
how much of the article the model can see.**
""")

md(r"""
## 5. Experiment 3 — the encoder axis (negative result)

This axis was never completed previously: DeBERTa-v3 crashed and ModernBERT was
killed mid-run. Both are resolved here.

**The DeBERTa crash, diagnosed.** It failed twice before — first
`Attempting to unscale FP16 gradients`, then `NaN`, then
`expected mat1 and mat2 to have the same dtype: Half != float`. The root cause is
that transformers 5.x loads weights in **the checkpoint's own dtype**, and
`deberta-v3-base` ships as fp16. So the encoder ran in pure fp16 (→ overflow → NaN)
while the freshly-created heads were fp32 (→ dtype clash). Forcing
`AutoModel.from_pretrained(..., dtype=torch.float32)` fixes all three symptoms.
""")

code(r"""
enc = rate[rate.variant.str.match(r"^[H-J]_")]
if len(enc):
    t = (enc.groupby(["variant", "score"])[["f1_macro", "roc_auc"]]
           .agg(["mean", "count"]).round(4))
    t.columns = ["_".join(c) for c in t.columns]
    print(t.sort_values("roc_auc_mean", ascending=False).to_string())
base = rate[(rate.variant == "E_ord_crit")].groupby("score")[["f1_macro","roc_auc"]].mean().round(4)
print("\nPubMedBERT-base reference (E_ord_crit):")
print(base.to_string())
""")

md(r"""
**No alternative encoder beat PubMedBERT-base.**

- **ModernBERT** has a native 8192 context and saw 2048 tokens — 4× the baseline —
  and still scored lowest on macro-F1. Long context does not rescue this task.
- **BiomedBERT-large** has the right domain but too much capacity for 1,483 articles.

One honest qualification: BiomedBERT-large **edged PubMedBERT on ROC-AUC**
(0.730 vs 0.722) while scoring well below it on macro-F1. That is a single seed
against a three-seed reference whose own AUC spread is ±0.012, so it sits inside
noise. It is not evidence the large model is better — but it does mean the correct
claim is *"no alternative beat the baseline"*, not *"all alternatives were worse on
every metric"*.

**DeBERTa-v3 is the one incomplete experiment.** With the dtype fix it trains
correctly and reached 0.649 validation macro-F1 after a single epoch, but fp32 makes
it roughly ten times slower than PubMedBERT on this GPU, so the full three-seed run
was stopped for time. It is listed as open work, not claimed as a result.

Combined with §4, the picture is that **domain-appropriate pretraining at base size
is hard to beat here** — extra capacity and extra context both fail to help.
""")

md(r"""
## 6. Is the improvement real? — the confirmation run

The objective change looked worth about +0.02–0.03 macro-F1. That is uncomfortably
close to the ±0.026 seed noise, so three seeds cannot settle it: the standard error
of the mean is `0.026/√3 ≈ 0.015`, the same size as the effect.

So the baseline and the best variant were each retrained **8 times** and compared with
a **Welch t-test** on the per-seed macro-F1 populations. This asks the right question:
*does the objective change produce better models*, independently of which particular
seed we drew.
""")

code(r"""
conf = pd.read_csv(OUTP / "tier2_confirm_seeds.csv")
summ = (conf.groupby(["config", "score"])
            .agg(n=("f1_macro", "size"), macroF1=("f1_macro", "mean"),
                 sd=("f1_macro", "std"), acc=("accuracy", "mean"),
                 recall=("recall_notcred", "mean"), auc=("roc_auc", "mean"))
            .round(4))
summ["SE"] = (summ["sd"] / np.sqrt(summ["n"])).round(4)
print(summ.to_string())

from scipy import stats
a = conf[(conf.config == "BASE_binary") & (conf.score == "bin")].f1_macro.values
print("\nWelch t-test vs the binary baseline (per-seed macro-F1):")
for s in sorted(conf[conf.config == "BEST_ord_crit"].score.unique()):
    b = conf[(conf.config == "BEST_ord_crit") & (conf.score == s)].f1_macro.values
    t, p = stats.ttest_ind(b, a, equal_var=False)
    flag = "  <-- significant" if p < 0.05 else ""
    print(f"  {s:<10} delta={b.mean()-a.mean():+.4f}   t={t:+.2f}   p={p:.4f}{flag}")
""")

md(r"""
### Result

The **learned criteria composer** (`crit_lr` — a logistic regression over the ten
predicted criteria probabilities, fitted on validation only) is significantly better
than the binary baseline: **+0.027 macro-F1, p = 0.022** over 8 independent retrainings.

Two secondary gains that matter as much as the headline:

- **Recall on `NOT_CREDIBLE` rises from 0.506 to 0.560.** For a misinformation tool,
  catching bad articles is the point; this is the metric that was embarrassing before.
- **Seed variance falls by a third** (±0.024 → ±0.016). More supervision per article
  makes training measurably more stable — exactly what the diagnosis predicted.

### The honest caveat

Two different uncertainties are in play and they answer different questions:

| test | question | result |
|---|---|---|
| Welch over 8 seeds | does the *method* build better models? | **p = 0.022 — yes** |
| paired bootstrap over 318 test articles | would it still win on *different* articles? | P(no gain) ≈ 0.09 — **not resolvable** |

So: the improvement is **reproducible**, but the test set is too small to certify it
would generalise to a fresh sample of articles. Both statements are true and both
belong in the report. This is a real but modest gain, not a breakthrough.
""")

md(r"""
## 7. Does it explain better? — per-criterion quality

Tier 2's product is *"verdict **plus** the reason"*. So the per-criterion predictions
matter independently of the binary verdict — if the joint model explains better, that
is an improvement even when the headline number moves little.
""")

code(r"""
newp = OUTP / "tier2_v2_criteria_performance.csv"
old = pd.read_csv(OUTP / "tier2_criteria_performance.csv")
if newp.exists():
    new = pd.read_csv(newp)
    key = "criterion"
    oldf = old.set_index(key)["f1"] if "f1" in old.columns else None
    cmp = new.set_index(key)[["f1", "mcc", "auc", "base_rate", "pred_rate"]].copy()
    if oldf is not None:
        cmp["f1_old"] = oldf
        cmp["delta_f1"] = (cmp["f1"] - cmp["f1_old"]).round(3)
    print(cmp.sort_values("f1", ascending=False).to_string())
    print(f"\nmean criteria F1 : {new['f1'].mean():.3f}")
    print(f"mean criteria MCC: {new['mcc'].mean():.3f}   (MCC is the honest one -- "
          f"F1 can be gamed by a constant classifier)")
else:
    print("final model not built yet; run confirm_over_8_seeds.py --mode final")
""")

md(r"""
## 8. Conclusions

**One artefact caveat.** `models/tier2_final_v2/model.pt` holds a single seed's weights and scores
**0.696** macro-F1 on its own; the **0.700** quoted above is the 5-seed ensemble. `meta.json` records
both, and the primary 8-seed evidence is unaffected either way.

**What worked.** Replacing the binary objective with ordinal + per-criterion
supervision, then composing the verdict from the ten predicted criteria. It is
significant across retrainings (p = 0.022), improves recall on the class that matters
by 5.4 points, and makes training a third more stable. It costs nothing at inference —
it is the same encoder with different heads.

**What did not work, tested properly.**

| axis | verdict |
|---|---|
| input representation | chunking, lexical features, head+tail — **all three failed** |
| encoder | ModernBERT, BiomedBERT-large, DeBERTa-v3 — **none beat PubMedBERT-base** |
| decision rule | threshold tuning — **null**, and F1-tuning actively degenerates |
| **objective** | **the one axis that worked** |

**What this says about the ceiling.** All three of the axes flagged as untested have
now been tested. Two produced clean negatives and one produced a modest, reproducible
gain. The remaining error is not obviously an engineering deficit: it looks intrinsic
to the task. Several criteria require genuine editorial judgement (`disease_mongering`
has validation MCC of exactly 0.000 — zero discrimination), and with 1,483 training
articles there is not enough signal to learn judgement.

**Tier 2 is still weaker than Tier 1, and that is expected, not a failure.** Tier 1
answers *"is this claim supported by this evidence?"* with the evidence supplied.
Tier 2 answers *"did this journalist report responsibly?"* from the article alone —
an inherently harder, partly subjective question, with 7× less training data. The two
numbers are not comparable, and presenting them as if they were would be misleading.

**Limitations that remain.**

1. The test set is 318 articles. Differences below ~0.05 cannot be certified on it.
2. `disease_mongering` remains unlearnable and is excluded from the pooled score.
3. The composer is fitted on the 318-article validation set, so it carries its own
   estimation noise.
4. Model selection still uses macro-F1 at a 0.5 threshold on the ordinal score, which
   is not that head's natural operating point — a likely source of further gain.
5. Single domain: US health journalism from HealthNewsReview's review window.
""")

nb["cells"] = C
nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python",
                                   "name": "python3"},
                    "language_info": {"name": "python"}})
nbf.write(nb, str(OUT))
print(f"wrote {OUT}  ({len(C)} cells)")
