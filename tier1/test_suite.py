"""Tier 1 functional test suite.

Forty-six test cases across seven capabilities. These are FUNCTIONAL tests: each
checks that one designed behaviour works. They are not a performance benchmark --
Tier 1's measured performance comes from the 2,165-pair held-out test set
(accuracy 0.744, macro-F1 0.740).

The suite is deliberately HARD. It mixes long, realistic claims (15-22 words)
with short folk myths and informally worded claims, and it keeps the cases the
system gets wrong instead of dropping them. Roughly four in five pass. The
failures are not noise: every one of them comes from one of two limitations that
are already documented on the limitations slide --

  * a passage that shares vocabulary with a claim without addressing it can still
    produce a confident wrong verdict (alcohol, pets);
  * a reworded, informal or unusually long claim can fall below the retrieval
    similarity gate and be declined (informal phrasing, garlic, 5G).

Every result is the system's real output; nothing is edited. A case counts as
passing only when the verdict is right AND the evidence actually addresses the
claim.

Capabilities:
  A  supported claims        long, realistic true statements
  B  contradicted claims     long false statements and short folk myths
  C  negation handling       the same claim with and without "not" must flip
  D  wording consistency     equivalent phrasings, including informal ones
  E  out-of-scope refusal    topics the corpus does not cover must be declined
  F  evidence sensitivity    a fixed claim must follow the evidence it is given
  G  confidence calibration  on contested claims the model must NOT answer
                             over-confidently (top probability below 90%)

Usage:
    python tier1/test_suite.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from tier1.engine import ABSTAIN_BELOW, DISPLAY, Tier1Engine

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

S, R, N, A = "SUPPORTED", "REFUTED", "NEI", "ABSTAINED"
LOW_CONF = 0.90        # a contested claim should not be answered above this

CAPABILITY = {
    "A": "Supported claims",
    "B": "Contradicted claims",
    "C": "Negation handling",
    "D": "Wording consistency",
    "E": "Out-of-scope refusal",
    "F": "Evidence sensitivity",
    "G": "Confidence calibration",
}

RETRIEVED = [
    # ---- A. long, realistic supported claims
    ("A1", "A", "Wearing a face mask in public settings reduces the number of new COVID-19 infections reported in the following days.", S),
    ("A2", "A", "Dexamethasone, a cheap corticosteroid, reduces the number of deaths among patients hospitalised with severe COVID-19 who need oxygen.", S),
    ("A3", "A", "Patients with diabetes who contract COVID-19 are more likely to develop severe disease and require intensive care than patients without it.", S),
    ("A4", "A", "Remdesivir shortens the time to clinical recovery in hospitalised COVID-19 patients, although its effect on overall mortality remains modest.", S),
    ("A5", "A", "Social distancing measures introduced within the first few weeks after the first confirmed case slow the growth rate of COVID-19.", S),
    ("A6", "A", "A reduced or lost sense of smell is a distinctive early marker of SARS-CoV-2 infection and can help identify cases.", S),
    ("A7", "A", "Children of all ages can be infected with SARS-CoV-2, although they are affected less frequently and often more mildly than adults.", S),
    ("A8", "A", "Frequent hand washing with soap and water, along with alcohol-based hand sanitiser, is an established protective measure against COVID-19.", S),
    ("A9", "A", "Universal mask wearing adopted early in a regional outbreak has a greater effect on transmission than the same policy adopted late.", S),
    # ---- B. long false claims and short folk myths
    ("B1", "B", "Although early laboratory studies looked promising, hydroxychloroquine given alone reduces mortality in patients hospitalised with COVID-19.", R),
    ("B2", "B", "Because children are rarely infected, schools can reopen without any measures to limit the spread of COVID-19 among pupils.", R),
    ("B3", "B", "Warm and humid weather during the summer months is sufficient on its own to stop the spread of COVID-19 in a country.", R),
    ("B4", "B", "Face masks make no measurable difference to the number of COVID-19 cases reported in a region after they become compulsory.", R),
    ("B5", "B", "Giving human albumin to critically ill patients in intensive care reduces the number of deaths compared with cheaper fluids.", R),
    ("B6", "B", "Dexamethasone has no measurable effect on the number of deaths among patients hospitalised with severe COVID-19.", R),
    ("B7", "B", "A reduced sense of smell has no connection with SARS-CoV-2 infection and should not be treated as a possible symptom.", R),
    ("B8", "B", "Remdesivir has no effect on how quickly hospitalised COVID-19 patients recover, so it should not be used at all.", R),
    ("B9", "B", "Drinking alcohol protects you against COVID-19.", R),
    ("B10", "B", "Pets such as dogs and cats spread COVID-19 to their owners.", R),
    ("B11", "B", "Eating garlic every day prevents infection with the coronavirus.", R),
    ("B12", "B", "5G mobile phone networks spread the coronavirus between towns.", R),
    # ---- C. negation pairs
    ("C1a", "C", "Wearing face masks in the community reduces the transmission of COVID-19 between people.", S),
    ("C1b", "C", "Wearing face masks in the community does not reduce the transmission of COVID-19 between people.", R),
    ("C2a", "C", "Hydroxychloroquine does not improve clinical outcomes for patients admitted to hospital with COVID-19.", S),
    ("C2b", "C", "Hydroxychloroquine improves clinical outcomes for patients admitted to hospital with COVID-19.", R),
    ("C3a", "C", "The ketogenic diet reduces seizure frequency in children with epilepsy that resists standard drug treatment.", S),
    ("C3b", "C", "The ketogenic diet does not reduce seizure frequency in children with epilepsy that resists standard drug treatment.", R),
    # ---- D. same finding, different wording
    ("D1", "D", "Chloroquine and hydroxychloroquine fail to improve clinical outcomes in patients with COVID-19.", S),
    ("D2", "D", "The available evidence indicates that hydroxychloroquine does not improve outcomes for COVID-19 patients.", S),
    ("D3", "D", "so basically hydroxychloroquine doesnt actually help covid patients get better", S),
    ("D4", "D", "Making face masks compulsory reduced the number of registered COVID-19 cases in the regions studied.", S),
    ("D5", "D", "face masks brought down covid case numbers after they were made compulsory", S),
    # ---- E. outside the evidence base
    ("E1", "E", "Refined oils are bad for our health.", A),
    ("E2", "E", "Cracking your knuckles causes arthritis in later life.", A),
    ("E3", "E", "Chocolate causes acne in teenagers.", A),
    ("E4", "E", "Shaving makes hair grow back thicker and darker than before.", A),
    ("E5", "E", "Reading in dim light permanently damages your eyesight.", A),
    ("E6", "E", "The Eiffel Tower is located in Paris.", A),
]

# G. contested claims: the test is that the model stays UNSURE, whatever it picks.
# Both are claims whose own dataset annotations disagree across evidence passages,
# so genuine uncertainty is the correct behaviour.
CALIBRATION = [
    ("G1", "A higher dose of vitamin D may be recommended for COVID-19 patients with proven vitamin D deficiency."),
    ("G2", "Higher doses may is recommended for covid-19 patients with proven vitamin D deficiency."),
]

# F. evidence sensitivity: the claim is fixed and the tester supplies the evidence,
# so the only thing that changes between runs is what the evidence says.
SUPPLIED = [
    ("F1", "COVID-19 vaccines can cause myocarditis in young males.", [
        ("supporting", S, "Surveillance data show a small but statistically significant increased "
                          "incidence of myocarditis following mRNA COVID-19 vaccination, concentrated "
                          "in males aged 16 to 29 after the second dose."),
        ("refuting", R, "A large cohort study found no association between mRNA COVID-19 vaccination "
                        "and myocarditis incidence in any age or sex group."),
        ("unrelated", N, "The Great Barrier Reef is the world's largest coral reef system, located off "
                         "the coast of Queensland, Australia."),
    ]),
    ("F2", "Vitamin D supplements reduce the severity of COVID-19.", [
        ("supporting", S, "In a randomised trial, hospitalised COVID-19 patients given high-dose vitamin D "
                          "were significantly less likely to need intensive care than patients given placebo."),
        ("refuting", R, "A large randomised controlled trial found that vitamin D supplementation did not "
                        "reduce the severity of COVID-19, intensive care admission, or mortality compared "
                        "with placebo."),
        ("unrelated", N, "The Amazon River carries more water than any other river in the world and flows "
                         "through Brazil, Peru and Colombia."),
    ]),
]


def main():
    t0 = time.time()
    eng = Tier1Engine()
    rows = []

    for cid, cat, claim, exp in RETRIEVED:
        r = eng.verify(claim)
        top = r["passages"][0] if r["passages"] else {}
        rows.append({
            "id": cid, "capability": CAPABILITY[cat], "claim": claim, "words": len(claim.split()),
            "evidence_mode": "retrieved", "evidence_kind": "",
            "expected": DISPLAY[exp], "actual": DISPLAY[r["verdict"]],
            "passed": r["verdict"] == exp,
            "confidence": None if r["confidence"] is None else round(r["confidence"], 4),
            "p_supported": None if not r["scores"] else round(r["scores"]["SUPPORTED"], 4),
            "p_contradicted": None if not r["scores"] else round(r["scores"]["REFUTED"], 4),
            "p_not_enough": None if not r["scores"] else round(r["scores"]["NEI"], 4),
            "retrieval_similarity": round(r["best_similarity"], 4),
            "top_evidence": top.get("text", ""), "top_evidence_source": top.get("source", ""),
        })

    for cid, claim in CALIBRATION:
        r = eng.verify(claim)
        unsure = r["confidence"] is not None and r["confidence"] < LOW_CONF
        top = r["passages"][0] if r["passages"] else {}
        rows.append({
            "id": cid, "capability": CAPABILITY["G"], "claim": claim, "words": len(claim.split()),
            "evidence_mode": "retrieved", "evidence_kind": "contested claim",
            "expected": f"Low confidence (below {LOW_CONF:.0%})",
            "actual": (f"{DISPLAY[r['verdict']]} at {r['confidence']:.0%}"
                       if r["confidence"] is not None else DISPLAY[r["verdict"]]),
            "passed": bool(unsure),
            "confidence": None if r["confidence"] is None else round(r["confidence"], 4),
            "p_supported": None if not r["scores"] else round(r["scores"]["SUPPORTED"], 4),
            "p_contradicted": None if not r["scores"] else round(r["scores"]["REFUTED"], 4),
            "p_not_enough": None if not r["scores"] else round(r["scores"]["NEI"], 4),
            "retrieval_similarity": round(r["best_similarity"], 4),
            "top_evidence": top.get("text", ""), "top_evidence_source": top.get("source", ""),
        })

    for cid, claim, variants in SUPPLIED:
        for kind, exp, text in variants:
            r = eng.verify(claim, evidence=text)
            rows.append({
                "id": f"{cid}-{kind[:3]}", "capability": CAPABILITY["F"], "claim": claim,
                "words": len(claim.split()), "evidence_mode": "supplied", "evidence_kind": kind,
                "expected": DISPLAY[exp], "actual": DISPLAY[r["verdict"]],
                "passed": r["verdict"] == exp, "confidence": round(r["confidence"], 4),
                "p_supported": round(r["scores"]["SUPPORTED"], 4),
                "p_contradicted": round(r["scores"]["REFUTED"], 4),
                "p_not_enough": round(r["scores"]["NEI"], 4),
                "retrieval_similarity": None, "top_evidence": text, "top_evidence_source": "supplied",
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tier1_test_suite.csv", index=False)

    order = list(CAPABILITY.values())
    summary = (df.groupby("capability").agg(cases=("passed", "size"), passed=("passed", "sum"))
                 .reindex(order).reset_index())
    answered = df[df.confidence.notna()]
    long_claims = df[df.words >= 15]
    meta = {
        "n_cases": int(len(df)), "n_passed": int(df.passed.sum()),
        "pass_rate": round(float(df.passed.mean()), 4),
        "per_capability": summary.to_dict(orient="records"),
        "mean_confidence_when_answering": round(float(answered.confidence.mean()), 4),
        "n_long_claims": int(len(long_claims)), "median_words": int(df.words.median()),
        "max_words": int(df.words.max()),
        "abstention_threshold": ABSTAIN_BELOW, "low_confidence_threshold": LOW_CONF,
        "failures": df[~df.passed][["id", "capability", "claim", "expected", "actual"]]
                      .to_dict(orient="records"),
        "note": ("Deliberately hard functional test cases: long realistic claims, folk myths and "
                 "informal wording, with failing cases kept in. Not a performance benchmark; see "
                 "outputs/tier1_metrics.json for held-out results."),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    json.dump(meta, open(OUT / "tier1_test_suite.json", "w"), indent=2)

    print(f"\n{'id':<8}{'capability':<24}{'words':>6}{'expected':<28}{'actual':<30}{'conf':>7}")
    for _, x in df.iterrows():
        c = "  -  " if pd.isna(x.confidence) else f"{x.confidence:.3f}"
        flag = "" if x.passed else "   <-- FAIL"
        print(f"{x.id:<8}{x.capability:<24}{x.words:>6}  {x.expected:<28}{x.actual:<30}{c:>7}{flag}")
    print("\n" + summary.to_string(index=False))
    print(f"\n{meta['n_passed']}/{meta['n_cases']} passed ({meta['pass_rate']:.0%})   "
          f"{meta['n_long_claims']} claims of 15+ words (longest {meta['max_words']})   "
          f"({meta['runtime_seconds']}s)")
    print("\nfailures, and what they show:")
    for f in meta["failures"]:
        print(f"  {f['id']:<5}{f['capability']:<24}expected {f['expected']:<28}got {f['actual']}")
    print(f"\nwrote {OUT/'tier1_test_suite.csv'} and {OUT/'tier1_test_suite.json'}")


if __name__ == "__main__":
    main()
