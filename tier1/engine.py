"""Tier 1 inference engine: retrieve evidence, verify a claim, explain the verdict.

This is the pipeline from 2_Tier1_Demo_Examples.ipynb lifted into one importable
module, so the demo app and the test suite run exactly the same code as the
system whose numbers are reported. The verification logic is deliberately
UNCHANGED: top-k passages concatenated as evidence, 256-token truncation, the
0.60 abstention gate, and the whole-word SHAP masker.

Usage:
    from tier1.engine import Tier1Engine
    eng = Tier1Engine()
    res = eng.verify("Masks reduce the spread of COVID-19.", explain=True)
"""
from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from transformers import (AutoModel, AutoModelForSequenceClassification, AutoTokenizer,
                          TextClassificationPipeline)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "tier1_final"

LABELS = ["SUPPORTED", "REFUTED", "NEI"]

# Calibrated in the Tier 1 notebook: in-corpus claims average a max cosine
# similarity of 0.678 (5th percentile 0.556); out-of-corpus claims average 0.419.
# A cut at 0.60 answers ~82% of in-corpus claims and abstains on every
# out-of-corpus probe.
ABSTAIN_BELOW = 0.60

# Internal names stay SUPPORTED / REFUTED / NEI (model config, metrics, CSVs).
# Only the displayed wording differs. "True"/"False" is avoided on purpose: the
# system reports what the retrieved EVIDENCE says, not what is true in the world.
DISPLAY = {
    "SUPPORTED": "Supported by evidence",
    "REFUTED": "Contradicted by evidence",
    "NEI": "Not enough evidence",
    "ABSTAINED": "No relevant evidence found",
}

_STOPWORDS = set("""
a an the is are was were be been being of to in on at for with and or but not no this that
these those it its as by from has have had do does did will would can could should may might
must if then than so such very just only also more most we our you your they their he she
""".split())


class BM25:
    """Okapi BM25 over the evidence corpus (lexical half of the hybrid retriever)."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.vec = CountVectorizer(stop_words="english", min_df=2)
        self.tf = self.vec.fit_transform(docs).astype(np.float32)
        dl = np.asarray(self.tf.sum(axis=1)).ravel()
        self.dl, self.avgdl = dl, dl.mean()
        n_dt = np.asarray((self.tf > 0).sum(axis=0)).ravel()
        self.idf = np.log(1 + (self.tf.shape[0] - n_dt + .5) / (n_dt + .5)).astype(np.float32)

    def scores(self, q):
        qq = self.vec.transform([q])
        idx = qq.indices
        if len(idx) == 0:
            return np.zeros(self.tf.shape[0], np.float32)
        sub = self.tf[:, idx].toarray()
        den = sub + self.k1 * (1 - self.b + self.b * (self.dl / self.avgdl))[:, None]
        return ((sub * (self.k1 + 1) / den) * self.idf[idx]).sum(1)


def _rrf(scores, k=60):
    order = np.argsort(-scores)
    r = np.zeros(len(scores), np.float32)
    r[order] = 1.0 / (k + np.arange(1, len(scores) + 1))
    return r


class Tier1Engine:
    def __init__(self, model_dir: Path = MODEL_DIR, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = (AutoModelForSequenceClassification.from_pretrained(model_dir)
                      .to(self.device).eval())
        self.corpus = pd.read_parquet(model_dir / "retrieval_corpus.parquet")
        meta = json.load(open(model_dir / "retriever_meta.json"))
        self.bm25 = BM25(self.corpus["text"].tolist())
        self.corpus_emb = np.load(model_dir / "retrieval_corpus_emb.npy")
        self.dense_tok = AutoTokenizer.from_pretrained(meta["dense_model_id"])
        self.dense_model = AutoModel.from_pretrained(meta["dense_model_id"]).to(self.device).eval()
        self.dense_model_id = meta["dense_model_id"]
        self._explainer = None

    # ---------------------------------------------------------------- retrieval
    @torch.no_grad()
    def _embed(self, texts):
        e = self.dense_tok(texts, padding=True, truncation=True, max_length=256,
                           return_tensors="pt")
        e = {k: v.to(self.device) for k, v in e.items()}
        h = self.dense_model(**e).last_hidden_state
        m = e["attention_mask"].unsqueeze(-1).float()
        return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).float().cpu().numpy()

    def retrieve(self, claim: str, k: int = 3):
        """Hybrid BM25 + dense retrieval fused by Reciprocal Rank Fusion.

        Returns the top-k passages by fused rank, the best dense similarity anywhere in
        the corpus (which drives the abstention gate), and the passage that achieved it.
        The fused top passage is not always the most similar one, so the abstention
        display uses the latter to stay consistent with the similarity it reports.
        """
        sim = self.corpus_emb @ self._embed([claim])[0]
        fused = _rrf(self.bm25.scores(claim)) + _rrf(sim)
        top = np.argsort(-fused)[:k]
        passages = [self._passage(i, sim) for i in top]
        return passages, float(sim.max()), self._passage(int(sim.argmax()), sim)

    def _passage(self, i, sim):
        return {"text": self.corpus["text"].iloc[i], "source": self.corpus["source"].iloc[i],
                "similarity": float(sim[i])}

    # ---------------------------------------------------------------- verification
    @torch.no_grad()
    def verify(self, claim: str, evidence: str | None = None, k: int = 3,
               explain: bool = False) -> dict:
        passages, best_sim, best_passage = [], None, None
        if evidence is None:
            passages, best_sim, best_passage = self.retrieve(claim, k)
            if best_sim < ABSTAIN_BELOW:
                return {"claim": claim, "verdict": "ABSTAINED", "confidence": None,
                        "scores": None, "evidence": None, "passages": passages,
                        "best_passage": best_passage,
                        "best_similarity": best_sim, "top_words": []}
            evidence = " ".join(p["text"] for p in passages)

        enc = self.tok(claim, evidence, truncation="only_second", padding=True,
                       max_length=256, return_tensors="pt")
        enc = {kk: v.to(self.device) for kk, v in enc.items()}
        probs = torch.softmax(self.model(**enc).logits, -1).float().cpu().numpy()[0]
        verdict = LABELS[int(probs.argmax())]
        words = self.explain(claim, evidence, verdict) if explain else []
        return {"claim": claim, "verdict": verdict, "confidence": float(probs.max()),
                "scores": {l: float(probs[i]) for i, l in enumerate(LABELS)},
                "evidence": evidence, "passages": passages, "best_passage": best_passage,
                "best_similarity": best_sim, "top_words": words}

    # ---------------------------------------------------------------- explanation
    def _get_explainer(self):
        if self._explainer is None:
            import shap
            pipe = TextClassificationPipeline(
                model=self.model, tokenizer=self.tok,
                device=0 if self.device == "cuda" else -1,
                top_k=None, truncation=True, max_length=256)
            # Whole-word masker. Without it SHAP splits on subword boundaries and
            # reports fragments like "ump" and "moz" (see the Tier 1 notebook).
            self._explainer = shap.Explainer(pipe, shap.maskers.Text(r"\W+"))
        return self._explainer

    def explain(self, claim: str, evidence: str, target: str, k: int = 8):
        text = claim + " " + self.tok.sep_token + " " + evidence
        sv = self._get_explainer()([text[:1200]])
        toks = [str(t).strip() for t in sv[0].data]
        vals = sv[0].values
        sc = vals[:, LABELS.index(target)] if getattr(vals, "ndim", 1) > 1 else vals
        seen, out = set(), []
        for w, s in sorted(zip(toks, sc), key=lambda x: -x[1]):
            # SHAP's whole-word masker leaves separators attached (e.g. "males. [" where
            # the claim meets [SEP]), so trim every non-word character from both ends.
            # That also merges "masks." and "masks,". Display-only: the verdict is
            # already decided before explanation runs.
            wl = re.sub(r"^\W+|\W+$", "", w.lower())
            if (not wl or wl == "sep" or wl in _STOPWORDS or len(wl) <= 2
                    or not re.search(r"[a-z]", wl) or wl in seen):
                continue
            seen.add(wl)
            out.append((wl, float(s)))
            if len(out) == k:
                break
        return out
