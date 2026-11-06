"""Result-dependent markdown for 6_Tier2_Final_System.ipynb.

Every number here was copied from the notebook's printed output after the full run
and re-checked. If anything is retrained, re-run the notebook and re-check.
"""

TEXT = {}

TEXT["s4_verdict"] = r"""
## 4. The verdict

The threshold is not tuned to maximise F1 on 318 articles; that was shown to manufacture
gains that disappear. It is set on validation so that the model flags the same share of
articles as the training data contains (33.5%). On the test set it flags 32.1%, against a
true rate of 33.6%.
"""

TEXT["s4b_compare"] = r"""
### Against every baseline, on the same 318 articles

Every baseline below is re-run here, not copied from earlier notebooks. v1 and BERT-base
use their shipped rule. TF-IDF + logistic regression appears twice: the configuration used
in the dataset investigation, and the best of 24 configurations **chosen on validation**.
Its test macro-F1 ranges from 0.624 to 0.689 depending on the configuration, so quoting a
configuration picked after seeing test results would be unfair in either direction.
"""

TEXT["s4_read"] = r"""
**Reading the table.** v2 has the best macro-F1 (0.697) and accuracy (0.733) of every
system, and its predicted not-credible rate (32.1%) is the closest to the true rate
(33.6%). It does **not** rank articles better: v1 (AUC 0.7275), the canonical TF-IDF model
(0.7268) and v2 (0.7256) are level. v2's advantage lies in turning that ranking into balanced
decisions, not in the ranking itself. BERT-base, trained identically to v1, is clearly
worse than PubMedBERT (macro-F1 0.596), which is why PubMedBERT is the backbone.
"""

TEXT["s4c_rules"] = r"""
### Four ways to turn the same ensemble into a verdict

The ensemble produces several scores that could each serve as the verdict. The rule was
fixed in advance: `crit_lr`, the composer over the ten criteria, is the rule tested in the
eight-seed confirmation run below. All four are shown for transparency. Picking whichever
scores best on **test** (here `crit_nodm`, 0.701) would be choosing on the test set. On
validation, `crit_lr` and `crit` tie for best and the ordinal head's own rule (`ord`) is
worst. The improvement report of 16 September quoted 0.700 from the `ord` rule of an
unsaved ensemble; that was a test-set choice, and this notebook supersedes it.
"""

TEXT["s5_signif"] = r"""
## 5. Is the improvement real?

Two separate questions, two tests.

1. **On these 318 articles, is v2 better than each baseline?** A paired bootstrap
   resamples the test articles 5,000 times and recomputes both systems' macro-F1 on the
   *same* resample, so article difficulty cancels out. `P(no gain)` is the share of
   resamples where v2 is not ahead.
2. **Would the gain survive retraining?** Retraining the same design with a different
   random seed moves macro-F1 by about ±0.02, as much as the effects being measured. So
   each design was trained eight times, and the two sets of scores were compared with a
   Welch t-test.
"""

TEXT["s5_read"] = r"""
**Reading the results.** Against BERT-base the gain is clear even on one test set (+0.101,
confidence interval above zero). Against v1 (+0.036) and TF-IDF (+0.031 to +0.039), the
paired bootstrap cannot rule out *no* gain (P = 0.07 to 0.15): 318 articles are too few to
resolve differences this size. The retraining test answers what one test set cannot: over
eight fresh trainings each, the v2 design beats the v1 design by +0.027 macro-F1
(p = 0.022), and varies less between trainings. The shipped ensemble (0.697) scores above
its own members (0.686 ± 0.011), as averaging should.

So the objective change is a real improvement over v1, but a modest one, and on a single
test set it cannot be told apart from a well-configured bag-of-words model.
"""

TEXT["s6_criteria"] = r"""
## 6. The criteria: the reasons behind the verdict

Tier 2's output is a verdict **and** the criteria the article fails. Building the demo
app exposed a problem the headline numbers hide, so this section asks directly whether the
ten criteria predictions carry ten different signals.

- **pairwise correlation**: how closely the ten predictions move together across
  articles. The reviewers' own judgements are the reference.
- **article-specific signal**: remove each article's overall level, then check whether
  what is left still ranks the criteria the article failed above the ones it met.
  0.5 means the model knows nothing about *which* criteria an article fails beyond how
  bad it is overall.
- **decision accuracy**: how often the met / not-met decisions match the reviewers,
  over every graded (article, criterion) pair, compared with always predicting each
  criterion's usual outcome.
"""

TEXT["s6_read"] = r"""
**The ten criteria heads collapse onto one factor.** The reviewers' ten judgements are
only loosely related (pairwise correlation 0.18), but the heads' predictions move almost
in lockstep (0.86; a single component explains 93% of their variation). All ten heads read
the same [CLS] vector, and with 1,483 training articles that vector mostly encodes overall
quality. The decisions show the cost. The heads match the reviewers on 68.2% of graded
criteria, slightly **worse** than always predicting each criterion's usual outcome
(69.2%). And what they know about which criterion an article fails, beyond its overall
quality, is barely above nothing (0.56).
"""

TEXT["s6b_hybrid"] = r"""
### A per-criterion fix

Separate per-criterion models do not share a representation, so they cannot collapse. A
TF-IDF logistic regression per criterion, trained only on the articles graded for that
criterion, predicts 7 of the 10 criteria better than the v2 heads. The criteria that are
**shown** blend the two, per criterion:

    score = w × TF-IDF + (1 − w) × v2        (each a standardised logit)

`w` is chosen per criterion on validation, and the blend is then calibrated on validation
into a probability. Everything is fixed on validation and test is scored once
(`experiments_tier2/train_criteria_hybrid.py`). Every system below, including v1's criteria
model, is scored with the same thresholds rule.
"""

TEXT["s6b_read"] = r"""
**Result.** The hybrid is the best criteria predictor on every summary measure. Criteria
macro-F1 rises 0.527 (v1) → 0.540 (v2 heads) → 0.581 (hybrid), and mean AUC
0.641 → 0.655 → 0.688. The gain over the v2 heads holds on the test articles (mean AUC
+0.026, 95% CI +0.013 to +0.039). Its predictions are much less collapsed (pairwise
correlation 0.56), they carry more article-specific signal (0.62), and its decisions now
beat the constant baseline, if only just (71.0% against 69.2%). Disease-mongering, which
the v2 head could not learn at all (AUC 0.54), becomes usable (AUC 0.64, MCC 0.26), so all
ten criteria are now shown.

This is still the weakest part of Tier 2. The criteria are predicted with AUC 0.62 to 0.75,
and articles still tend to be marked as failing most criteria or few: 16.0% are marked as
failing 9 or 10 criteria, against 1.9% by the reviewers (who grade 8.9 criteria per article
on average, which flatters them slightly). The app shows each criterion's AUC beside it for
exactly this reason.
"""

TEXT["s6c_verdict"] = r"""
**The verdict stays on the v2 composer.** A composer over the hybrid criteria looks better
on validation (cross-validated macro-F1 0.658 against 0.637) but worse on test (0.668
against 0.697). Both gaps are within retraining noise. The v2 verdict has the eight-seed
significance test behind it, and the rule is that a change must be demonstrated, not merely
not ruled out. So the verdict is computed from the v2 criteria, while the criteria *shown*
come from the hybrid. They usually agree; where they do not, the verdict is the
better-evidenced output.
"""

TEXT["s7_errors"] = r"""
## 7. Where the verdict goes wrong
"""

TEXT["s7_read"] = r"""
Errors concentrate near the label boundary: 2- and 3-star articles are judged correctly
57.5% and 70.5% of the time, 4- and 5-star articles 87.9% and 97.5%. The 1-star articles
(61.3%) are a weak spot; some badly rated articles read like careful journalism. News
releases are much harder than news stories (63.1% against 76.9%). They are rated on slightly different criteria
and make up only 28% of the data.

**Length is a mild bias, not a shortcut.** Word count alone barely predicts the label
(AUC 0.575; the model 0.726), and within each length quarter the model still ranks
articles well (AUC 0.63 to 0.78) while word count does not. But the model does call the
longest quarter credible more often than the reviewers did (81.2% against 71.2%).
"""

TEXT["s8_router"] = r"""
## 8. The router

The system routes text by length: 60 words or fewer goes to Tier 1 (claim verification),
anything longer to Tier 2. The cut-off is a heuristic, not learned. On the project's own
data:
"""

TEXT["s8_read"] = r"""
5,067 of 5,075 texts go to the right tier. The eight misrouted Tier 1 items are long
advice passages from HealthVer, like the one printed, rather than single claims; neither
tier is designed for them. The router has no notion of topic: a non-health article would
still be sent to Tier 2 and judged against health-reporting criteria.
"""

TEXT["s9_explain"] = r"""
## 9. Explanations

Tier 1 explains with word-level SHAP. On a 600-word article, word-level SHAP scatters
highlights over dozens of isolated words, so Tier 2 explains at the level of
**sentences**. SHAP's Partition explainer treats each sentence as one feature. It removes a
sentence by deleting it from the text and measures how the output changes. Only the text
the model actually reads (the first 512 tokens) is explained, since later sentences cannot
affect the output.

One explanation covers eleven outputs at once: the credibility score behind the verdict,
and the ten criteria. The budget is max(300, 12 × sentences) evaluations. Raising it to
1,000 left the sentence ranking unchanged (Spearman 1.0 on two articles checked). Each
explanation takes several seconds on a laptop GPU and a few minutes on a CPU.
"""

TEXT["s9_read"] = r"""
In this example, the sentence that pushes the article hardest toward *falls short* is the
researcher's claim that the pill could be a "game changer". That is the same sentence the
reviewers singled out: "leaving unchallenged a researcher's assertion that fevipiprant may
be a game changer". For *independent sources*, the three most influential sentences are all
quotes from the study's own researchers. For *benefits*, the sentences pulling toward "met"
describe the trial size and the eosinophil result, which the reviewer also credited before
failing the criterion on other grounds (and, less sensibly, an image credit). One example
proves nothing, so the next section measures this over the whole test set.
"""

TEXT["s9b_faith"] = r"""
### Are the explanations faithful, and do they match the reviewers?

These are two different properties, and both are tested over the whole test set.

- **Faithful to the model**: do the sentences SHAP points at actually drive the model's
  output? *Deletion test*: remove the 3 sentences SHAP says pushed hardest toward "falls
  short" and re-score. The credibility score should rise, and by more than when 3 random
  sentences are removed (10 random draws per article). The same is done for the 3
  sentences pushing toward "meets standards".
- **Plausible to humans**: are those the sentences a human expert would point at?
  FakeHealth keeps each reviewer's written reason for every criterion, and the reasons
  often quote the article. A read sentence that shares a run of six consecutive words
  with the reviewer's reason for a criterion counts as a human rationale for that
  criterion. The question is whether SHAP ranks those sentences above the others more
  often than chance. This is the Tier 2 counterpart of Tier 1's test against SciFact's
  expert rationales.
"""


TEXT["s9b_read"] = r"""
**Faithful to the model: yes.** Removing the three sentences SHAP blames raises the
credibility score by +0.031 on average, while removing three random sentences changes it by
−0.0015. SHAP beats random removal on 76.8% of the 311 articles long enough to test
(p = 3 × 10⁻²⁴), and the reverse also holds (73.0%, p = 3 × 10⁻²⁰). The explanations
point at sentences the model genuinely relies on.

**Plausible to humans: weakly, and only for some criteria.** Over 355 (article, criterion)
pairs, SHAP ranks the sentences the reviewer quoted above the others slightly more often
than chance (rank AUC 0.548, p = 0.001). Its top three sentences contain a quoted one 13.0%
of the time, against 9.5% by chance, which is not significant on its own (p = 0.13). The
agreement comes from the concrete criteria, where a single sentence either states a number
or does not: costs (0.67, 16 pairs), harms (0.63, 29 pairs) and benefits (0.60, the largest
group with 105 pairs). For abstract criteria it is at chance (evidence quality 0.49) or
below it (independent sources 0.34, only 19 pairs). At the verdict level there is no
agreement at all (0.502).

Two caveats pull in opposite directions. Matching quotes is a noisy proxy (reviewers
also quote sentences for context), so it understates agreement. But the model also truly
relies on cues a reviewer would not cite. The deletion test shows that the sentences
matter *to the model*; this test shows that they are often not the ones a person would
pick.

The hybrid criteria also made the per-criterion explanations more specific. On the 140
articles explained under both criteria models, the ten explanations correlate 0.44 with
each other instead of 0.77, and agreement with the reviewers rises from 0.487 to 0.520.

**Compared with Tier 1.** Tier 1's word-level explanations land in expert rationales
45.6% of the time against 33.7% by chance. Tier 2's explanations are just as faithful to
their model but much less aligned with human reasoning. That fits everything above: Tier 2
learns overall reporting quality far better than it learns the specific reasons.
"""

TEXT["s10_examples"] = r"""
## 10. The demonstration articles

These are the four articles offered in the demo app (`app.py`), all from the held-out test
set. The last one is included on purpose: a 4-star article the model judges as falling
short. A demo that only shows successes would misrepresent the system.
"""

TEXT["s11_summary"] = r"""
## 11. Summary

| | Tier 2 v2 |
|---|---|
| Verdict, 318 held-out articles | macro-F1 **0.697**, accuracy 0.733, AUC 0.726 |
| vs v1 (binary objective) | +0.036 on this test set (P(no gain) = 0.07); over 8 retrainings each +0.027, **p = 0.022** |
| vs TF-IDF tuned on validation | +0.039 (P(no gain) = 0.10); equal AUC |
| Criteria shown (hybrid) | macro-F1 0.581, mean AUC 0.688 (v1: 0.527 / 0.641) |
| Router | 5,067 of 5,075 texts sent to the right tier |
| Explanations | faithful (beats random deletion on 76.8% of articles); weakly aligned with reviewers (rank AUC 0.548) |

**What changed in v2.** The objective, not the encoder. Training on the ordinal rating and
the ten criteria instead of one binary label gave a gain that holds up across retrainings.
Integration then showed that the criteria heads collapse onto one factor, and a
per-criterion blend with TF-IDF models fixed most of that.

**Limitations.**

1. The verdict is only modestly better than a bag-of-words model and indistinguishable
   from it on one test set. 318 test articles cannot resolve differences smaller than
   about 0.05 macro-F1.
2. The model reads only the first 512 tokens. Only 44 of the 318 test articles fit
   entirely; for the rest it reads a median of the first 379 words, and caveats often sit
   further down.
3. The criteria are the weakest output: AUC 0.62 to 0.75, and still partly
   all-or-nothing.
4. The explanations are faithful to the model, but only weakly match what human reviewers
   point to.
5. News releases are harder than news stories (63.1% against 76.9% accuracy).
6. FakeHealth's canonical split is random, not by outlet, so absolute numbers are
   optimistic by about 0.055 on unseen publishers (`5_Tier2_Dataset_Investigation.ipynb`).
7. The label measures reporting quality as judged by HealthNewsReview.org reviewers of US
   health journalism. It does not measure factual accuracy; that is Tier 1's job.
"""
