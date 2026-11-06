"""Numbers the final report needs that were never saved as files.

1. Tier 1 test-set predictions (gold evidence), for the confusion matrix. Also
   re-checks the reported 0.744 accuracy / 0.740 macro F1.
2. The confidence gate on every held-out test claim: the best cosine similarity
   between each of the 444 test claims and the 6,559-passage corpus, and the
   share that clears the 0.60 threshold. The original calibration used 120
   claims; this is the full test set.

    python tools/final_report/report_extra_numbers.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tier1.engine import ABSTAIN_BELOW, LABELS, Tier1Engine

test = pd.read_csv(ROOT / "datasets" / "processed_tier1" / "test.csv")
dev = "cuda" if torch.cuda.is_available() else "cpu"
mdir = ROOT / "models" / "tier1_final"
tok = AutoTokenizer.from_pretrained(mdir)
model = AutoModelForSequenceClassification.from_pretrained(mdir).to(dev).eval()

preds = []
with torch.no_grad():
    for i in range(0, len(test), 32):
        b = test.iloc[i:i + 32]
        enc = tok(b.claim.tolist(), b.evidence.tolist(), truncation="only_second",
                  padding=True, max_length=256, return_tensors="pt").to(dev)
        preds.append(model(**enc).logits.argmax(-1).cpu().numpy())
pred = np.concatenate(preds)
y = test.label.map({l: i for i, l in enumerate(LABELS)}).values
cm = confusion_matrix(y, pred, labels=[0, 1, 2])
print("accuracy", round(accuracy_score(y, pred), 4), "macro F1", round(f1_score(y, pred, average="macro"), 4))
print(cm)
del model
torch.cuda.empty_cache()

eng = Tier1Engine()
claims = test.drop_duplicates("claim").claim.tolist()
sims = []
for i in range(0, len(claims), 64):
    e = eng._embed(claims[i:i + 64])
    sims.append((eng.corpus_emb @ e.T).max(0))
sims = np.concatenate(sims)
print(f"{len(claims)} test claims: mean best similarity {sims.mean():.3f}, "
      f"5th percentile {np.percentile(sims, 5):.3f}, answered at {ABSTAIN_BELOW}: {(sims >= ABSTAIN_BELOW).mean():.3f}")

json.dump({"tier1_test_accuracy": float(accuracy_score(y, pred)),
           "tier1_test_macro_f1": float(f1_score(y, pred, average="macro")),
           "tier1_confusion": cm.tolist(), "labels": LABELS,
           "gate_test_claims": len(claims), "gate_similarities": sims.round(4).tolist(),
           "gate_answered_share": float((sims >= ABSTAIN_BELOW).mean()),
           "gate_mean": float(sims.mean()), "gate_p5": float(np.percentile(sims, 5))},
          open(ROOT / "outputs" / "final_report_extra.json", "w"), indent=1)
print("wrote outputs/final_report_extra.json")
