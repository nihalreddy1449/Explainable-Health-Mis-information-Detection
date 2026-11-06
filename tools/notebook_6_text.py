"""Markdown for 6_Tier2_Final_System.ipynb (used by build_notebook_6.py).

Every number quoted here was checked against the notebook's printed output after
it ran. If the models are retrained, re-run the notebook and re-check this file.
"""

TEXT = {}

TEXT["title"] = r"""
# Tier 2 — Article Credibility Assessment (v2)

**Explainable Health Misinformation Detection using Deep Learning**
Team 10 · Amrita School of Artificial Intelligence, Coimbatore · Supervisor: Dr. C. R. Lekshmi

---

Tier 1 checks a short claim against biomedical evidence. **Tier 2 reads a full health news
article and asks a different question: does it report health science responsibly?**

The labels come from HealthNewsReview.org, where experts graded each article on ten
journalistic criteria (does it discuss costs, quantify benefits and harms, explain the
quality of the evidence, use independent sources...) and gave it a 0–5 star rating.
None of the criteria asks whether the article is factually true: an accurate article
can score one star for hyping a preliminary study and leaving out the harms. So Tier 2
flags *irresponsible reporting*, not *false information*. Factual checking is Tier 1's job.

This notebook documents the complete Tier 2 system:

1. the data, and why the training objective matters more than the encoder;
2. the model — one PubMedBERT encoder with an ordinal rating head and ten criteria
   heads, trained five times and ensembled;
3. the verdict, compared with every baseline on the same 318 held-out articles, with
   significance tests;
4. the ten criteria, which are the reasons it gives, including a problem with them that
   integration exposed, and the fix;
5. the router that sends short claims to Tier 1 and articles to Tier 2;
6. sentence-level SHAP explanations, checked against the human reviewers' own reasons.

The model runs through `tier2/engine.py`, the same code the demo app (`app.py`) uses.
The earlier binary model is documented unchanged in `3_Tier2_v1_Binary_Model.ipynb`;
the experiments that led here are in `4_Tier2_Improvement_Experiments.ipynb` and
`5_Tier2_Dataset_Investigation.ipynb`.
"""

TEXT["s1_data"] = r"""
## 1. Data

**FakeHealth** (Dai et al., 2020): 2,119 health news articles and news releases reviewed
by HealthNewsReview.org. The canonical split is 1,483 / 318 / 318.

The binary label is derived from the star rating: **2 stars or fewer is `NOT_CREDIBLE`**.
The cell below checks this for every article.
"""

TEXT["s2_model"] = r"""
## 2. Model

### Why the objective, not the encoder

The star rating is essentially a count of criteria met, so it and the fraction of
criteria failed move together almost perfectly (correlation printed above). The binary
label is a coarsening of that rating. A model trained only on the binary label sees
**one** bit per article and throws away the ten criteria judgements the reviewers
made. With 1,483 training articles, that was the bottleneck.

Changing the encoder (ModernBERT with 2,048 tokens of context, BiomedBERT-large; a
DeBERTa-v3 comparison was started but not completed), the input (chunking, head+tail truncation, fifteen lexical features), the
decision threshold, or adding data did not help; those negative results are recorded in
`4_Tier2_Improvement_Experiments.ipynb`. Changing the **objective** did.

### Architecture

One PubMedBERT encoder reads the first 512 tokens of *title. text*. Its `[CLS]` vector
feeds two heads:

- **Ordinal head (CORAL).** Predicts P(rating > k) for k = 0…4, sharing one direction
  with five ordered biases, so the five probabilities can never contradict each other.
  P(rating > 2) *is* the binary task, so this head contains the target and adds four
  aligned auxiliary tasks.
- **Criteria head.** Ten sigmoid outputs, P(article fails criterion). Articles are not
  graded on every criterion, so missing grades are masked out of the loss.

There is **no binary head**: adding one consistently made results worse.

**The verdict** is a logistic regression over the ten predicted criteria probabilities,
fitted on the validation set. The article is judged to fall short when this score is
below a threshold set on validation so that the predicted rate matches the training
base rate (rate matching; F1-tuned thresholds were shown to be unreliable on 318
articles). Section 6 explains why the criteria *shown* to the user come from a
per-criterion blend with a word-based model rather than from these heads alone.

**Five models.** Retraining the same model with a different random seed moves test
macro-F1 by about ±0.02, which is as large as the effects being measured. The shipped
system trains five seeds and averages their predictions. The training code is below.
"""

TEXT["s3_load"] = r"""
## 3. Loading the saved system

`Tier2Engine` loads the five saved models, the verdict composer and threshold, and the
criteria hybrid (section 6) from `models/tier2_v2_ensemble/`. The check at the end confirms that the engine reproduces
the scores the training script recorded, so everything below is the deployed system.
"""

