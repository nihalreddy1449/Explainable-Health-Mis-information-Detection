# Explainable Health Misinformation Detection using Deep Learning

B.Tech final year project, Department of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore.
Team 10, supervised by Dr. C. R. Lekshmi.

The system checks health information in two ways and explains every result.

* **Tier 1, short claims.** It searches a corpus of biomedical research for evidence about the claim, then
  decides whether that evidence supports the claim, contradicts it, or is not enough to decide. If nothing
  relevant is found, it says so instead of guessing.
* **Tier 2, full articles.** It judges whether a health news article reports the science responsibly, using
  the ten reporting criteria of HealthNewsReview.org (costs, benefits, harms, quality of evidence, independent
  sources and so on), and shows which criteria the article falls short on.

A router sends text of 60 words or fewer to Tier 1 and anything longer to Tier 2. SHAP highlights the words
(Tier 1) or sentences (Tier 2) that drove each result.

Tier 1 reports what the retrieved evidence says, not what is true in the world. Tier 2 judges reporting
quality, not factual accuracy.

![The web application](Presentation/Final%20Review/screenshots/app_entry.png)

## Results

| | Test data | Result |
|---|---|---|
| Tier 1 verifier | 444 claims never seen in training (2,165 claim and evidence pairs) | macro F1 **0.740** |
| Same model without the evidence | same | macro F1 0.508 |
| Same setup with BERT base | same | macro F1 0.712 |
| Evidence retrieval (hybrid) | 150 test claims | recall@10 **0.780** |
| SHAP words inside expert rationales | SciFact test claims | 45.6%, against 33.7% by chance |
| Tier 2 verdict | 318 held out articles | macro F1 **0.697**, accuracy 0.733, ROC AUC 0.726 |
| Tier 2 first version (binary label) | same | macro F1 0.661 |
| TF-IDF, tuned on validation | same | macro F1 0.658 |
| Tier 2 with BERT base | same | macro F1 0.596 |
| New design against the first, 8 retrainings each | | +0.027 macro F1, p = 0.022 (Welch t test) |
| Tier 2 criteria (the reasons shown) | 318 held out articles | criteria macro F1 0.581, mean AUC 0.688 |
| Router | 5,075 texts from both tiers | 5,067 sent to the right tier |

Every number above comes from the saved outputs of notebooks 1 and 6. The experiments that did not work
out are recorded in notebooks 3 to 5.

## How it works

**Tier 1**

1. Retrieval: BM25 and dense search with `NeuML/pubmedbert-base-embeddings` over 6,559 research passages,
   combined with reciprocal rank fusion. The top 3 passages become the evidence.
2. Confidence gate: if the best passage has a cosine similarity below 0.60, the system declines to answer.
3. Verification: a fine tuned PubMedBERT reads `[CLS] claim [SEP] evidence [SEP]` and outputs supported,
   refuted or not enough information.
4. Explanation: SHAP over whole words.

**Tier 2**

1. Five PubMedBERT models (different random seeds) read the title and the first 512 tokens of the article.
2. Each has an ordinal head (CORAL) that learns the 0 to 5 star rating and a head that predicts the ten
   criteria. Learning the rating and the criteria, instead of one credible or not label, is what improved
   the model.
3. The verdict is a logistic regression over the ten criteria, with a threshold set so the share of articles
   flagged matches the training data.
4. The criteria shown to the user blend each head with a separate TF-IDF model per criterion, because the
   shared heads alone collapsed into one overall quality score.
5. Explanation: SHAP over sentences.

## Repository layout

```
app.py                  web application (Streamlit)
pipeline.py             router and the combined two tier pipeline
tier1/                  Tier 1 engine and its functional test suite
tier2/                  Tier 2 engine, explanation checks, test suite, demo articles
experiments_tier2/      scripts that ran the Tier 2 experiments and train the final models
tools/                  notebook builders and runner, screenshot tool, report figures
outputs/                saved metrics, predictions, figures and training logs
1_ ... 6_*.ipynb        the notebooks, with outputs saved
Presentation/           screenshots of the web application
```

| Notebook | Contents |
|---|---|
| `1_Tier1_Claim_Verification.ipynb` | Tier 1 data, training, evaluation, retrieval and explanation checks |
| `2_Tier1_Demo_Examples.ipynb` | Five Tier 1 examples with their evidence and explanations |
| `3_Tier2_v1_Binary_Model.ipynb` | The first, binary version of Tier 2 |
| `4_Tier2_Improvement_Experiments.ipynb` | The experiments that led to the final Tier 2 design |
| `5_Tier2_Dataset_Investigation.ipynb` | TF-IDF baselines, a dataset quality check and the ReCOVery experiments |
| `6_Tier2_Final_System.ipynb` | The final Tier 2 system: evaluation, significance tests, criteria, router, explanations |

## Running it

The trained models (about 5 GB) and the datasets are not in this repository. They have to be downloaded and
trained as described below.

**1. Environment** (Python 3.13)

```
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

**2. Data**

* HealthVer and SciFact are downloaded automatically by notebook 1.
* FakeHealth: download [EnyanDai/FakeHealth](https://github.com/EnyanDai/FakeHealth) and place it at
  `datasets/raw/FakeHealth/FakeHealth-master/`.
* ReCOVery (only needed for notebook 5): place `recovery-news-data.csv` from
  [apurvamulay/ReCOVery](https://github.com/apurvamulay/ReCOVery) at `datasets/raw/ReCOVery/`.

**3. Models**

```
# Tier 1: run notebook 1. It prepares the data and saves models/tier1_final/
# Tier 2 data and first version: run notebook 3. It saves the splits and the v1 models
python experiments_tier2/train_final_ensemble.py     # five Tier 2 models, about 19 minutes on a laptop GPU
python experiments_tier2/train_criteria_hybrid.py    # criteria hybrid, about 2 minutes on the CPU
```

**4. Application and tests**

```
streamlit run app.py               # then open http://localhost:8501
python tier1/test_suite.py
python tier2/test_suite.py
```

The results were produced on an NVIDIA RTX 5050 laptop GPU with 8 GB of memory.

## Data sources

* HealthVer: Sarrouti et al., 2021. COVID-19 claims paired with evidence from research papers.
* SciFact: Wadden et al., 2020. Expert written scientific claims paired with research abstracts.
* FakeHealth: Dai, Sun and Wang, 2020. Health news reviewed by HealthNewsReview.org experts.
* ReCOVery: Zhou et al., 2020. COVID-19 news with outlet reliability labels (investigated, not used in the final system).

## Limitations

* Tier 1 can only use the evidence in its corpus, which is mostly COVID-19 and biomedical research. Outside
  it the system declines, and it answers 64.6% of the held out test claims.
* The correct passage is ranked first for only 24.7% of claims. A reranker would help.
* Tier 2 reads the first 512 tokens of an article, and FakeHealth has only 2,119 articles. On a single test
  set the gain over TF-IDF is not statistically certain.

## Team

K. V. Nihal Reddy, V. Sai Ved, U. Sandeep and V. Bhuvanesh, under the guidance of Dr. C. R. Lekshmi,
Assistant Professor, Department of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore.

## Licence

The code is released under the MIT licence, see [LICENSE](LICENSE). The datasets keep their own licences.
