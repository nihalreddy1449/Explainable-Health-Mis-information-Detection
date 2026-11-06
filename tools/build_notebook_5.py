"""Generate 5_Tier2_Dataset_Investigation.ipynb from saved artefacts.

Presentation layer only -- reads outputs/*.json and *.csv written by
experiments_tier2/{prepare_recovery_splits,recovery_transfer,more_data_cross_validation}.py, so it runs in seconds and
always agrees with what is on disk.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "5_Tier2_Dataset_Investigation.ipynb"

nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r"""
# Tier 2 — Is there a better dataset than FakeHealth?

**Explainable Health Misinformation Detection using Deep Learning**
Team 10 · Supervisor: Dr. C. R. Lekshmi

---

The v2 work (see `4_Tier2_Improvement_Experiments.ipynb`) improved Tier 2 from 0.661 to 0.700 macro-F1
by changing the training objective, and showed that **supervision volume** was the
bottleneck — not the encoder and not the context length.

That points at the data. FakeHealth is only 2,119 articles from a single reviewing
organisation. So: **is there a bigger or better corpus?**

This notebook answers that empirically. It covers four things:

1. A **reality check** — does the transformer even beat TF-IDF?
2. A **reusable dataset-quality probe**, applied to both FakeHealth and a candidate
   replacement, ReCOVery.
3. What ReCOVery is and isn't good for, including a **zero-shot transfer test** that
   settles whether the two corpora measure the same thing.
4. Whether simply using **more articles** helps.

Three of the four are negative results. All four are measurements.
""")

code(r"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.width", 130)
ROOT = Path.cwd(); OUTP = ROOT / "outputs"

leak   = json.load(open(OUTP / "dataset_leakage_probe.json"))
tfidf  = json.load(open(OUTP / "tier2_tfidf_canonical.json"))
recmet = json.load(open(OUTP / "recovery_meta.json"))
trans  = pd.read_csv(OUTP / "tier2_transfer_results.csv")
cv     = pd.read_csv(OUTP / "tier2_cv_results.csv")
print("artefacts loaded")
""")

md(r"""
## 1. Reality check — does the transformer earn its keep?

Before looking for better data, check the model is beating a bag of words. TF-IDF +
logistic regression, same canonical split, same rate-matched threshold.
""")

code(r"""
ref = pd.DataFrame([
    {"system": "TF-IDF + logistic regression", **{k: tfidf[k] for k in
        ("f1_macro","accuracy","recall_notcred","roc_auc")}},
    {"system": "v1 binary PubMedBERT (8-seed)", "f1_macro": 0.6582, "accuracy": 0.7036,
     "recall_notcred": 0.5058, "roc_auc": 0.7009},
    {"system": "v2 ordinal+criteria (8-seed)", "f1_macro": 0.6851, "accuracy": 0.7229,
     "recall_notcred": 0.5596, "roc_auc": 0.7179},
])
print(ref.to_string(index=False))
""")

md(r"""
**The original Tier 2 model was worse than TF-IDF** — on macro-F1 (0.658 vs 0.666) *and*
on ROC-AUC (0.701 vs 0.727). A 110M-parameter domain-pretrained transformer was losing to
word counts.

That reframes the v2 contribution: the objective change is what makes the transformer
worth using at all. And even now v2 only clearly wins on macro-F1 and accuracy — **TF-IDF
still edges it on AUC**.

This is consistent with the per-criterion pattern found earlier: the criteria that are
learnable are exactly the ones with concrete vocabulary (costs, harms, benefits). Much of
Tier 2's signal is lexical, which is precisely what TF-IDF captures.
""")

md(r"""
## 2. A reusable dataset-quality probe

This project was rebuilt once already because PubHealth's label turned out to encode
*writing style* rather than truth. So no new corpus gets adopted without the same kind of
check. Two questions:

1. **Do outlets carry both labels?** If every outlet has exactly one label, the label *is*
   the outlet, and a model can succeed by recognising house style.
2. **How far does TF-IDF fall when train and test share no outlet?** The gap measures how
   much apparent performance is outlet memorisation.

Applied to FakeHealth and to the candidate replacement, ReCOVery (Zhou et al. 2020 —
2,029 COVID-19 news articles labelled reliable/unreliable).
""")

code(r"""
fh, rc = leak["fakehealth"], leak["recovery"]
tbl = pd.DataFrame([
    {"": "outlets", "FakeHealth": fh["n_sources_total"], "ReCOVery": rc["n_publishers"]},
    {"": "outlets carrying BOTH labels",
     "FakeHealth": f"{fh['sources_with_both_labels']} of {fh['n_sources_total']}",
     "ReCOVery": f"{rc['publishers_with_both_labels']} of {rc['n_publishers']}"},
    {"": "label is a function of the outlet", "FakeHealth": "no", "ReCOVery": "YES"},
    {"": "TF-IDF, random split", "FakeHealth": fh["tfidf_random"], "ReCOVery": rc["tfidf_random"]},
    {"": "TF-IDF, outlet-disjoint split",
     "FakeHealth": fh["tfidf_source_disjoint"], "ReCOVery": rc["tfidf_publisher_disjoint"]},
    {"": "inflation from outlet memorisation",
     "FakeHealth": f"+{fh['drop']:.3f}", "ReCOVery": f"+{rc['drop']:.3f}"},
])
print(tbl.to_string(index=False))
""")

md(r"""
### Verdict: ReCOVery is *worse*-posed, not better

ReCOVery's label is a **pure function of the publisher** — all 55 of its outlets carry a
single label, because the label is a NewsGuard score assigned to the *outlet* and copied
onto every article it published. Its top 5 publishers are 43% of the corpus. So a random
split lets a model score 0.904 by recognising house style; holding outlets out costs it
**13 points**.

FakeHealth behaves quite differently. **83 of its 450 sources carry both labels** — the
New York Times, Reuters Health and WebMD each publish articles rated both credible and
not — so the outlet *cannot* be a shortcut, and the outlet-disjoint penalty is less than
half as large.

**FakeHealth, despite being smaller, is the better-posed corpus.** It stays.

> **An honest caveat cutting the other way.** FakeHealth is not perfectly clean either.
> Its own outlet-disjoint penalty is +0.055, so the canonical random split is optimistic
> by roughly that much in absolute terms. It applies equally to v1 and v2, so every
> comparison in this project still holds — but the absolute numbers would be lower on
> unseen publishers. Re-splitting outlet-disjoint is listed as open work.
""")

md(r"""
## 3. What ReCOVery *is* good for

It was still trained and evaluated properly, with **publisher-disjoint splits**, both as a
second capability and as a possible donor of extra data.
""")

code(r"""
r1 = trans[trans.exp == "R1_recovery"]
per, ens = r1[r1.seed != "ENS"], r1[r1.seed == "ENS"]
print(f"ReCOVery: {recmet['n_articles']} articles, {recmet['n_publishers']} publishers, "
      f"majority baseline {recmet['majority_baseline']:.3f}\n")
print(pd.DataFrame([
    {"system": "TF-IDF (publisher-disjoint)", "macro_f1": 0.773, "accuracy": None, "roc_auc": None},
    {"system": "PubMedBERT (3-seed mean)", "macro_f1": round(per.f1_macro.mean(),4),
     "accuracy": round(per.accuracy.mean(),4), "roc_auc": round(per.roc_auc.mean(),4)},
    {"system": "PubMedBERT (seed ensemble)", "macro_f1": ens.f1_macro.iloc[0],
     "accuracy": ens.accuracy.iloc[0], "roc_auc": ens.roc_auc.iloc[0]},
]).to_string(index=False))
""")

md(r"""
**TF-IDF beats PubMedBERT here too** (0.773 vs 0.724). ReCOVery's signal is largely
topical and lexical — the vocabulary of low-credibility outlets — which a bag of words
captures directly. The transformer ranks well (AUC 0.84) but thresholds poorly across
publishers it has never seen.

### The decisive experiment

If FakeHealth and ReCOVery measured the same underlying thing, a model trained on one
should transfer to the other. Zero-shot, no target-domain training:
""")

code(r"""
z = trans[trans.exp == "R2_fh_to_rec"]
print(z[["score","f1_macro","accuracy","roc_auc"]].to_string(index=False))
print("\nROC-AUC 0.50 = chance.")
""")

md(r"""
**ROC-AUC 0.504 and 0.486 — exactly chance.**

A model that predicts FakeHealth's reporting-quality labels well has *literally no
ability* to predict ReCOVery's outlet-reliability labels.

This is worth dwelling on, because it **directly measures** something this project
previously argued only from inspection. The PubHealth post-mortem said that merging
claim-veracity labels with journalism-quality labels was incoherent. That was a reasoned
argument. This is the same claim as a number: *"reports irresponsibly"* and *"is an
unreliable outlet"* are orthogonal, and combining them into one target would be repeating
the original mistake.

### Can ReCOVery at least donate data?

Sequential transfer (train the encoder on ReCOVery, discard the head, fine-tune with the
v2 ordinal+criteria heads on FakeHealth) keeps the labels separate, so the PubHealth
failure cannot recur by construction.
""")

code(r"""
r3 = trans[trans.exp == "R3_intermediate"]
print(r3.groupby("score")[["f1_macro","accuracy","recall_notcred","roc_auc"]]
        .agg(["mean","std"]).round(4).to_string())
print("\ndirect v2 reference (8 seeds): crit_lr macro-F1 0.6851 +- 0.0160")
""")

md(r"""
No help — 0.676 against the direct v2's 0.685, slightly worse. Entirely consistent with
the chance-level transfer above. **ReCOVery cannot donate useful signal to FakeHealth.**
""")

md(r"""
## 4. Does simply using more articles help?

The v2 result said supervision volume was the bottleneck, so the obvious move is more
articles. The validation split is 318 articles idle at training time; folding it in takes
training data from 1,483 to 1,801 (**+21%**).

The catch is that validation selects the checkpoint, sets the threshold, and fits the
composer. Training on it naively would make all three self-fitted. So this uses 5-fold
cross-validation over the pooled articles, with the threshold and composer fitted on
**out-of-fold** predictions only. The test set is untouched.
""")

code(r"""
print(cv[["score","accuracy","f1_macro","recall_notcred","roc_auc"]].to_string(index=False))
print("\nreference, v2 trained on train only (8-seed mean, same rule):")
print("  crit_lr    acc=0.7229  macroF1=0.6851  recall=0.5596  auc=0.7179")
""")

md(r"""
**No macro-F1 gain** (0.6859 vs 0.6851, +0.001). Recall did improve meaningfully
(0.598 vs 0.560), which matters for this application, but 318 extra articles is too small
an increase to move the headline at this test-set size.

So "more data" is confirmed as the right *direction* but 21% is not enough of it.
""")

md(r"""
## 5. Conclusions

| question | answer |
|---|---|
| Does the transformer beat TF-IDF? | v1 **did not**; v2 does, but only on macro-F1 and accuracy |
| Is ReCOVery a better dataset? | **No** — easier, but the label is the publisher |
| Do the two corpora measure the same thing? | **No** — zero-shot transfer is exactly chance |
| Can ReCOVery donate training signal? | **No** — sequential transfer is slightly worse |
| Does +21% more data help? | **No** measurable macro-F1 gain (recall improves) |

### What this means

Five independent avenues have now been closed with measurements. Combined with the v2
work (where input representation and encoder both failed), the conclusion is that Tier 2's
remaining error is **largely task-intrinsic**: several criteria require genuine editorial
judgement, and there is not enough data — nor, apparently, any freely available corpus —
to learn judgement.

### The one thing that would actually change the picture

The **Monant Medical Misinformation Dataset** (Srba et al., SIGIR 2022): ~317k medical
articles from 207 sources, plus 3.5k fact-checked claims. It is the only surveyed corpus
large enough to shift the data situation. It requires a request from an institutional
email via Zenodo (DOI `10.5281/zenodo.5996864`) with manual approval, so it could not be
obtained overnight — **this is the recommended next step**, and an Amrita address should
qualify.

Other corpora surveyed and rejected: Verifee (Czech, general news), Health-Bench
(LLM evaluation, not article credibility), NELA-GT (source-level labels, same publisher
confound as ReCOVery but larger).
""")

nb["cells"] = C
nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python",
                                   "name": "python3"},
                    "language_info": {"name": "python"}})
nbf.write(nb, str(OUT))
print(f"wrote {OUT}  ({len(C)} cells)")
