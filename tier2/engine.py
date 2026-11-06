"""Tier 2 inference engine: judge how responsibly a health news article reports.

This loads the saved five-member ensemble (models/tier2_v2_ensemble/) and applies
exactly the decision rule that was evaluated, so the notebook, the demo app and
the test suite all run the same system whose numbers are reported:

  1. Each member (PubMedBERT, ordinal + criteria heads) reads the first 512
     tokens of "title. text" and predicts P(fails criterion) for ten
     HealthNewsReview criteria. The five members' probabilities are averaged.
  2. The verdict is a logistic regression over the ten averaged criteria
     probabilities (the `crit_lr` rule, fitted on validation), thresholded at the
     validation quantile equal to the training base rate.
  3. The criteria SHOWN to the user come from a per-criterion hybrid: the ensemble's
     probability blended with an independent TF-IDF logistic regression
     (experiments_tier2/train_criteria_hybrid.py). The shared-encoder heads alone
     collapse onto one "overall quality" factor (predictions correlate 0.94 with
     each other, against 0.21 for the reviewers' labels); the hybrid tells the
     criteria apart better (mean test AUC 0.668 -> 0.694, paired bootstrap CI
     excludes zero). Each is marked "not met" above a threshold rate-matched on
     validation.
  4. Explanation: SHAP over SENTENCES (not words). A masked sentence is deleted
     from the text. Only the text the model actually reads is explained.

What the verdict means: whether the article REPORTS health science responsibly
(costs, harms, quality of evidence, independent sources...). It is not a check of
whether the article's claims are true; that is Tier 1's job.

Usage:
    from tier2.engine import Tier2Engine
    eng = Tier2Engine()
    res = eng.assess(article_text, explain=True)
"""
from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "tier2_v2_ensemble"

LABELS = ["NOT_CREDIBLE", "CREDIBLE"]

# Internal names stay CREDIBLE / NOT_CREDIBLE (datasets, metrics, CSVs). The
# displayed wording says what the label actually measures: reporting quality,
# not factual truth.
DISPLAY = {
    "CREDIBLE": "Meets reporting standards",
    "NOT_CREDIBLE": "Falls short of reporting standards",
}

CRITERIA = ["costs", "benefits", "harms", "evidence_quality", "disease_mongering",
            "independent_srcs", "alternatives", "availability", "novelty", "press_release"]

# Criteria never shown as findings. disease_mongering used to be here: the
# ensemble's own head cannot learn it (test MCC 0.009), but the hybrid can
# (test MCC 0.26, AUC 0.64, in line with the other criteria), so all ten are shown.
HIDDEN: set[str] = set()

# Short names and the HealthNewsReview question behind each criterion. Criteria 6
# and 10 are worded differently for news releases; both wordings are covered.
CRITERION_NAME = {
    "costs": "Discusses costs",
    "benefits": "Quantifies the benefits",
    "harms": "Quantifies the harms",
    "evidence_quality": "Explains the quality of the evidence",
    "disease_mongering": "Avoids disease-mongering",
    "independent_srcs": "Independent sources, conflicts of interest",
    "alternatives": "Compares with existing alternatives",
    "availability": "Establishes availability",
    "novelty": "Establishes true novelty",
    "press_release": "Goes beyond the press release, no sensational language",
}
CRITERION_QUESTION = {
    "costs": "Does it adequately discuss the costs of the intervention?",
    "benefits": "Does it adequately quantify the benefits?",
    "harms": "Does it adequately explain and quantify the harms?",
    "evidence_quality": "Does it seem to grasp the quality of the evidence?",
    "disease_mongering": "Does it avoid disease-mongering?",
    "independent_srcs": "Does it use independent sources and identify conflicts of interest?",
    "alternatives": "Does it compare the new approach with existing alternatives?",
    "availability": "Does it establish the availability of the treatment, test or procedure?",
    "novelty": "Does it establish the true novelty of the approach?",
    "press_release": "Does it go beyond a news release, without unjustified sensational language?",
}

def article_text(title, text) -> str:
    """The exact input format used in training: "title. text" (an empty title
    still leaves the ". " prefix, as it did for the dataset's untitled articles)."""
    title = "" if title is None or (isinstance(title, float) and np.isnan(title)) else str(title)
    text = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    return title + ". " + text


# ------------------------------------------------------------------ sentences
_SPLIT = re.compile(
    r"\s*\n\s*"                                           # line / paragraph breaks
    r"|(?<=[.!?])\s+(?=[\"'“‘(\[]?[A-Z0-9])"              # . ! ? then a capital
    r"|(?<=[.!?][\"'”’)\]])\s+(?=[\"'“‘(\[]?[A-Z0-9])")   # ." ?' .) then a capital
_ABBREV = re.compile(r"(?:\b(?:Dr|Mr|Mrs|Ms|Prof|St|Jr|Sr|vs|No|Inc|Co|Corp|Ltd|Gov|Sen|Rep|"
                     r"Jan|Feb|Mar|Apr|Aug|Sept?|Oct|Nov|Dec|Fig|al|approx)\."
                     r"|\b(?:[A-Z]\.){1,3}|\be\.g\.|\bi\.e\.)$")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of the sentences in `text`. Splits on line breaks and on
    sentence-final punctuation followed by a capital, then re-joins pieces that
    were cut after an abbreviation such as "Dr." or "U.S."."""
    spans, pos = [], 0
    for m in _SPLIT.finditer(text):
        if m.start() > pos:
            spans.append([pos, m.start()])
        pos = m.end()
    if pos < len(text):
        spans.append([pos, len(text)])
    merged = []
    for s in spans:
        if merged and text[merged[-1][0]:merged[-1][1]].count("\n") == 0 and \
                _ABBREV.search(text[merged[-1][0]:merged[-1][1]]) and \
                "\n" not in text[merged[-1][1]:s[0]]:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [(a, b) for a, b in merged if text[a:b].strip()]


class SentenceTokenizer:
    """Tokenizer for shap.maskers.Text whose "tokens" are whole sentences.

    Same return format as shap's SimpleTokenizer (strings + character offsets), so
    SHAP masks and attributes whole sentences.
    """

    def __call__(self, s, return_offsets_mapping=True):
        # "" must yield no tokens (shap probes it to detect special tokens); the
        # whitespace mask token must yield exactly one.
        spans = sentence_spans(s) if s.strip() else ([(0, len(s))] if s else [])
        out = {"input_ids": [s[a:b] for a, b in spans]}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out


# ------------------------------------------------------------------ model
class _Member(nn.Module):
    """Inference-only copy of experiments_tier2/model_and_variant_sweep.py::T2Model with the ordinal
    and criteria heads. Parameter names match, so the saved state dicts load
    strictly."""

    def __init__(self, config):
        super().__init__()
        self.enc = AutoModel.from_config(config)
        h = config.hidden_size
        self.crit_head = nn.Linear(h, len(CRITERIA))
        self.ord_dir = nn.Linear(h, 1, bias=False)
        self.ord_bias = nn.Parameter(torch.zeros(5))

    def forward(self, input_ids, attention_mask):
        pooled = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.crit_head(pooled), self.ord_dir(pooled) + self.ord_bias


class Tier2Engine:
    def __init__(self, model_dir: Path = MODEL_DIR, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.meta = json.load(open(model_dir / "meta.json"))
        self.max_len = int(self.meta["max_len"])
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        config = AutoConfig.from_pretrained(model_dir)
        self.members = []
        for sd in self.meta["seeds"]:
            m = _Member(config)
            m.load_state_dict(torch.load(model_dir / f"seed_{sd}.pt", map_location="cpu"), strict=True)
            self.members.append(m.to(self.device).eval())
        comp = self.meta["composer"]
        self.coef = np.asarray(comp["coef"], dtype=np.float64)
        self.intercept = float(comp["intercept"])
        self.threshold = float(self.meta["verdict_threshold"])
        import joblib
        self.hyb = joblib.load(model_dir / "criteria_hybrid.joblib")
        assert self.hyb["criteria"] == CRITERIA
        hyb_meta = json.load(open(model_dir / "criteria_hybrid.json"))
        self.crit_thr = np.asarray(self.hyb["thresholds"])
        self.crit_auc = {k: v.get("auc") for k, v in hyb_meta["test"].items()}
        # evaluation ran under bf16 autocast on the GPU; do the same so the engine
        # reproduces the reported scores. On CPU everything runs in fp32.
        self.amp = self.device == "cuda" and torch.cuda.is_bf16_supported()
        self._explainer = None

    # ---------------------------------------------------------------- scoring
    @torch.no_grad()
    def score(self, texts: list[str], batch_size: int = 16) -> dict:
        """Ensemble scores for a list of article texts ("title. body")."""
        crit, ordn = [], []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(list(texts[i:i + batch_size]), truncation=True,
                           max_length=self.max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
            c_sum, o_sum = 0.0, 0.0
            for m in self.members:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                    cl, ol = m(**enc)
                c_sum = c_sum + torch.sigmoid(cl.float())
                o_sum = o_sum + torch.sigmoid(ol.float())
            crit.append((c_sum / len(self.members)).cpu().numpy())
            ordn.append((o_sum / len(self.members)).cpu().numpy())
        crit, ordn = np.concatenate(crit), np.concatenate(ordn)
        return {"criteria": self._hybrid(list(texts), crit), "criteria_v2": crit,
                "ordinal": ordn, "credibility": self._compose(crit)}

    def _hybrid(self, texts, crit_v2):
        """Calibrated P(fails criterion) from the v2 + TF-IDF blend."""
        h = self.hyb
        X = h["vectorizer"].transform(texts)
        tf = np.column_stack([c.predict_proba(X)[:, 1] for c in h["classifiers"]])
        lg = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))
        z_v2 = (lg(crit_v2) - h["v2_stats"][0]) / h["v2_stats"][1]
        z_tf = (lg(tf) - h["tf_stats"][0]) / h["tf_stats"][1]
        z = h["weights"] * z_tf + (1 - h["weights"]) * z_v2
        return 1.0 / (1.0 + np.exp(-(h["platt_a"] * z + h["platt_b"])))

    def _compose(self, crit):
        z = crit.astype(np.float64) @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    # ---------------------------------------------------------------- reading window
    def sentences(self, text: str) -> list[dict]:
        """Split into sentences and mark which ones fit in the model's 512-token
        window. BERT's tokenizer splits on whitespace first, so per-sentence token
        counts add up exactly to the count for the joined text."""
        budget = self.max_len - 2                           # [CLS] ... [SEP]
        used, out = 0, []
        for a, b in sentence_spans(text):
            n = len(self.tok(text[a:b], add_special_tokens=False)["input_ids"])
            read = "full" if used + n <= budget else ("partial" if used < budget else "none")
            out.append({"text": text[a:b], "start": a, "end": b, "tokens": n, "read": read})
            used += n
        return out

    # ---------------------------------------------------------------- verdict
    def assess(self, text: str, explain: bool = False, max_evals: int | None = None) -> dict:
        text = text.strip()
        s = self.score([text])
        cred, crit = float(s["credibility"][0]), s["criteria"][0]
        sents = self.sentences(text)
        criteria = []
        for j, k in enumerate(CRITERIA):
            if k in HIDDEN:
                continue
            criteria.append({"key": k, "name": CRITERION_NAME[k], "question": CRITERION_QUESTION[k],
                             "p_fail": float(crit[j]), "threshold": float(self.crit_thr[j]),
                             "met": bool(crit[j] < self.crit_thr[j]), "auc": self.crit_auc.get(k)})
        res = {"text": text, "n_words": len(text.split()),
               "verdict": "CREDIBLE" if cred >= self.threshold else "NOT_CREDIBLE",
               "credibility": cred, "threshold": self.threshold,
               "criteria": criteria, "sentences": sents,
               "words_read": len(" ".join(x["text"] for x in sents if x["read"] != "none").split()),
               "explanation": None}
        if explain:
            res["explanation"] = self.explain(text, sents, max_evals=max_evals)
        return res

    # ---------------------------------------------------------------- explanation
    def _outputs(self, texts):
        """SHAP target: column 0 = credibility score (the verdict), columns 1..10 =
        the displayed P(fail criterion) from the hybrid."""
        s = self.score([str(t) for t in texts], batch_size=16)
        return np.column_stack([s["credibility"], s["criteria"]])

    def _get_explainer(self):
        if self._explainer is None:
            import shap
            # mask_token " " -> a masked sentence is simply deleted (runs of
            # whitespace are collapsed by the masker)
            masker = shap.maskers.Text(SentenceTokenizer(), mask_token=" ", collapse_mask_token=True)
            self._explainer = shap.Explainer(self._outputs, masker,
                                             output_names=["credibility"] + CRITERIA)
        return self._explainer

    def explain(self, text: str, sents: list[dict] | None = None, max_evals: int | None = None) -> dict:
        """Sentence-level SHAP over the part of the article the model reads.

        Returns one row per read sentence: its SHAP value for the credibility score
        (negative = pushed toward "falls short") and for each criterion's
        P(not met) (positive = pushed toward "not met").
        """
        sents = sents if sents is not None else self.sentences(text)
        read = [x for x in sents if x["read"] != "none"]
        window = " ".join(x["text"] for x in read)
        n = len(read)
        max_evals = max_evals or max(300, 12 * n)
        sv = self._get_explainer()([window], max_evals=max_evals, batch_size=32, silent=True)
        vals = np.asarray(sv[0].values)                       # (n_tokens, 11)
        toks = [str(t) for t in sv[0].data]
        if len(toks) != n:
            # the masker re-tokenised the joined window; realign by position
            n = min(n, len(toks))
        return {"sentences": [x["text"] for x in read][:n],
                "credibility": vals[:n, 0].tolist(),
                "criteria": {k: vals[:n, 1 + j].tolist() for j, k in enumerate(CRITERIA)},
                "base_credibility": float(np.asarray(sv[0].base_values)[0]),
                "max_evals": max_evals}
