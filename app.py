"""Explainable health misinformation detection - the two-tier demonstration.

Run from the project root:
    streamlit run app.py

Paste a short health claim or a full health news article. The router sends text of
60 words or fewer to Tier 1 (is the claim supported by biomedical evidence?) and
anything longer to Tier 2 (does the article report health science responsibly?).

URL parameters pre-fill and run the app, which is how screenshots are produced:
    http://localhost:8501/?text=Masks+reduce+transmission+of+COVID-19.
    http://localhost:8501/?example=<article_id>        (a Tier 2 test-set article)
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from pipeline import ROUTER_MAX_WORDS, HealthMisinfoPipeline, route, word_count
from tier1.engine import ABSTAIN_BELOW, DISPLAY as T1_DISPLAY
from tier2.engine import DISPLAY as T2_DISPLAY, article_text

st.set_page_config(page_title="Health Misinformation Detection", layout="centered",
                   initial_sidebar_state="collapsed")

CLAIM_EXAMPLES = {
    "Supported": [
        "Wearing a face mask in public settings reduces the number of new COVID-19 infections reported in the following days.",
        "Dexamethasone, a cheap corticosteroid, reduces the number of deaths among patients hospitalised with severe COVID-19 who need oxygen.",
    ],
    "Contradicted": [
        "Although early laboratory studies looked promising, hydroxychloroquine given alone reduces mortality in patients hospitalised with COVID-19.",
        "Face masks make no measurable difference to the number of COVID-19 cases reported in a region after they become compulsory.",
    ],
    "Model is unsure": [
        "A higher dose of vitamin D may be recommended for COVID-19 patients with proven vitamin D deficiency.",
    ],
    "Outside the evidence base": [
        "Cracking your knuckles causes arthritis in later life.",
    ],
}

CSS = """
<style>
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
[data-testid="stAppDeployButton"] { display: none !important; visibility: hidden; }

.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 780px; padding-top: 2.6rem; padding-bottom: 3rem;
}
html, body, [class*="css"], .stMarkdown, p, li, label, textarea, input, button {
    font-family: "Segoe UI", -apple-system, "Helvetica Neue", Arial, sans-serif;
}
h1, h2, h3 { font-family: Georgia, "Times New Roman", serif; font-weight: 400; color: #111; }

.eyebrow { font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: #6b6b6b; margin: 0; }
.title { font-family: Georgia, "Times New Roman", serif; font-size: 2.05rem; line-height: 1.15;
         margin: 0.25rem 0 0.4rem 0; color: #111; }
.byline { color: #5a5a5a; font-size: 0.92rem; margin: 0 0 1.4rem 0; }
.lede { color: #2b2b2b; font-size: 0.98rem; line-height: 1.6; margin-bottom: 1.4rem; }
.rule { border: 0; border-top: 1px solid #dcdcdc; margin: 1.8rem 0 1.4rem 0; }
.section { font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: #6b6b6b;
           margin: 1.6rem 0 0.7rem 0; }
.route { font-size: 0.86rem; color: #444; margin: -0.3rem 0 1rem 0; }

.verdict { font-family: Georgia, "Times New Roman", serif; font-size: 1.9rem; color: #111; margin: 0; }
.verdict-sub { color: #555; font-size: 0.9rem; margin: 0.3rem 0 1.1rem 0; line-height: 1.5; }

.bar-row { display: grid; grid-template-columns: 170px 1fr 56px; align-items: center;
           gap: 12px; margin: 5px 0; font-size: 0.88rem; color: #333; }
.bar-track { background: #ededed; height: 8px; }
.bar-fill { background: #111; height: 8px; }
.bar-val { text-align: right; font-variant-numeric: tabular-nums; color: #111; }

.crit { display: grid; grid-template-columns: 1fr 64px 120px 44px; gap: 7px 12px; align-items: center;
        font-size: 0.88rem; color: #222; }
.crit .h { font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: #777; }
.crit .st { font-weight: 600; }
.crit .st.met { font-weight: 400; color: #666; }
.crit .n { text-align: right; font-variant-numeric: tabular-nums; color: #666; font-size: 0.8rem; }
.crit .bar-track { position: relative; }
.crit .tick { position: absolute; top: -3px; width: 1px; height: 14px; background: #111; }

.passage { border-left: 2px solid #111; padding: 0.15rem 0 0.15rem 0.9rem; margin: 0 0 1.1rem 0; }
.passage.dim { border-left-color: #c4c4c4; }
.passage-meta { font-size: 0.76rem; color: #6b6b6b; letter-spacing: 0.02em; margin-bottom: 0.25rem; }
.passage-text { font-size: 0.93rem; line-height: 1.55; color: #222; }

.art { border-left: 2px solid #111; padding: 0.1rem 0 0.1rem 0.9rem; font-size: 0.92rem;
       line-height: 1.7; color: #222; max-height: 560px; overflow-y: auto; }
.s-lo { border-bottom: 1.5px solid #111; }
.s-hi { border-bottom: 1.5px dotted #666; }
.s-unread { color: #a0a0a0; }
.mark { font-size: 0.68rem; vertical-align: super; color: #111; margin-left: 1px; }

.hl { padding: 0 2px; border-bottom: 1px solid #111; }
.words { display: grid; grid-template-columns: 150px 1fr 70px; gap: 6px 12px; align-items: center;
         font-size: 0.88rem; }
.words .w { color: #111; }
.words .v { text-align: right; font-variant-numeric: tabular-nums; color: #444; }
.sents { display: grid; grid-template-columns: 22px 1fr 64px; gap: 8px 10px; font-size: 0.88rem;
         line-height: 1.5; color: #222; margin-bottom: 0.4rem; }
.sents .i { color: #777; font-variant-numeric: tabular-nums; }
.sents .v { text-align: right; font-variant-numeric: tabular-nums; color: #444; }

.notice { background: #f5f5f5; padding: 0.9rem 1rem; font-size: 0.9rem; color: #333; line-height: 1.55; }
.review { background: #f5f5f5; padding: 0.8rem 1rem; font-size: 0.88rem; color: #333; line-height: 1.55; }
.steps { font-size: 0.86rem; color: #444; line-height: 1.7; margin: 0; padding-left: 1.1rem; }
.foot { color: #8a8a8a; font-size: 0.78rem; margin-top: 2.6rem; border-top: 1px solid #e6e6e6;
        padding-top: 0.9rem; }

div[data-testid="stTextArea"] textarea { border-radius: 2px; font-size: 0.95rem; }
.stButton > button, [data-testid="stBaseButton-primary"] {
    border-radius: 2px; background: #111; color: #fff; border: 1px solid #111;
    padding: 0.45rem 1.4rem; font-weight: 500;
}
.stButton > button:hover { background: #333; border-color: #333; color: #fff; }
div[data-baseweb="select"] > div { border-radius: 2px; }
div[role="radiogroup"] label p { font-size: 0.92rem; }
</style>
"""


@st.cache_resource(show_spinner=False)
def load_pipeline():
    return HealthMisinfoPipeline()


@st.cache_data(show_spinner=False)
def load_article_examples():
    """Tier 2 examples are real TEST-set articles, never seen in training, with the
    human reviewers' rating kept for comparison."""
    spec = json.load(open(ROOT / "tier2" / "demo_examples.json", encoding="utf-8"))
    te = pd.read_csv(ROOT / "datasets" / "processed_tier2" / "test.csv").set_index("article_id")
    out = []
    for e in spec:
        r = te.loc[e["article_id"]]
        out.append({**e, "text": article_text(r["title"], r["text"]), "rating": int(r["rating"]),
                    "binary_label": r["binary_label"],
                    "summary": str(r["review_summary"]) if pd.notna(r["review_summary"]) else ""})
    return out


def esc(s: str) -> str:
    return html.escape(s or "")


def clip_sentences(s: str, limit: int) -> str:
    """Cut at the last full sentence within `limit` characters."""
    s = " ".join((s or "").split())
    if len(s) <= limit:
        return s
    cut = max(s.rfind(". ", 0, limit), s.rfind("? ", 0, limit), s.rfind("! ", 0, limit))
    return s[:cut + 1] if cut > 0 else s[:limit].rsplit(" ", 1)[0] + "..."


# ---------------------------------------------------------------- tier 1 rendering
def highlight_words(text: str, words) -> str:
    out = esc(text)
    for w, s in words:
        if s <= 0:
            continue
        out = re.sub(rf"(?i)\b({re.escape(esc(w))})\b", r'<span class="hl">\1</span>', out)
    return out


def render_tier1(res, explain, eng):
    if res["verdict"] == "ABSTAINED":
        st.markdown(f'<div class="verdict">{T1_DISPLAY["ABSTAINED"]}</div>'
                    f'<div class="verdict-sub">No verdict given</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="notice">The closest passage in the evidence base has a similarity of '
            f'<b>{res["best_similarity"]:.3f}</b>, below the <b>{ABSTAIN_BELOW:.2f}</b> threshold. '
            f'Rather than return a confident verdict based on unrelated text, the system declines. '
            f'The evidence base covers COVID-19 and biomedical research literature.</div>',
            unsafe_allow_html=True)
        p = res["best_passage"]
        st.markdown('<div class="section">Closest passage (rejected as unrelated)</div>'
                    f'<div class="passage dim"><div class="passage-meta">{esc(p["source"])} &middot; '
                    f'similarity {p["similarity"]:.3f}</div><div class="passage-text">'
                    f'{esc(p["text"][:420])}{"..." if len(p["text"]) > 420 else ""}</div></div>',
                    unsafe_allow_html=True)
        return

    src = "evidence retrieved automatically" if res["passages"] else "evidence you provided"
    st.markdown(f'<div class="verdict">{T1_DISPLAY[res["verdict"]]}</div>'
                f'<div class="verdict-sub">Confidence {res["confidence"]:.1%} &middot; {src}</div>',
                unsafe_allow_html=True)
    rows = []
    for key, label in (("SUPPORTED", "Supported"), ("REFUTED", "Contradicted"), ("NEI", "Not enough evidence")):
        v = res["scores"][key]
        rows.append(f'<div class="bar-row"><div>{label}</div><div class="bar-track">'
                    f'<div class="bar-fill" style="width:{v*100:.1f}%"></div></div>'
                    f'<div class="bar-val">{v:.3f}</div></div>')
    st.markdown("".join(rows), unsafe_allow_html=True)

    words = []
    if explain:
        with st.spinner("Computing word attributions (SHAP)..."):
            words = eng.explain(res["claim"], res["evidence"], res["verdict"])
    if res["passages"]:
        st.markdown(f'<div class="section">Evidence retrieved &middot; best similarity '
                    f'{res["best_similarity"]:.3f}</div>', unsafe_allow_html=True)
        for i, p in enumerate(res["passages"], 1):
            cls = "passage" if i == 1 else "passage dim"
            st.markdown(f'<div class="{cls}"><div class="passage-meta">Passage {i} &middot; '
                        f'{esc(p["source"])} &middot; similarity {p["similarity"]:.3f}</div>'
                        f'<div class="passage-text">{highlight_words(p["text"], words)}</div></div>',
                        unsafe_allow_html=True)
    else:
        st.markdown('<div class="section">Evidence provided</div>'
                    f'<div class="passage"><div class="passage-text">'
                    f'{highlight_words(res["evidence"], words)}</div></div>', unsafe_allow_html=True)
    if explain and words:
        mx = max(abs(s) for _, s in words) or 1.0
        rows = "".join(
            f'<div class="w">{esc(w)}</div><div class="bar-track">'
            f'<div class="bar-fill" style="width:{max(abs(s)/mx*100, 2):.0f}%"></div></div>'
            f'<div class="v">{s:+.4f}</div>' for w, s in words)
        st.markdown('<div class="section">Words that drove the verdict &middot; SHAP</div>'
                    f'<div class="words">{rows}</div>'
                    '<div class="verdict-sub" style="margin-top:0.8rem">Underlined words in the evidence '
                    'are the ones ranked here. A higher score means the word pushed the model more '
                    'strongly toward this verdict.</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- tier 2 rendering
TOP_K = 3
MIN_SHARE = 0.2   # list a sentence only if its effect is at least this share of the largest


def render_article(text, sents, marks):
    """The article with SHAP-ranked sentences underlined. marks: {sentence_index: (css, label)}."""
    parts, pos = [], 0
    for i, s in enumerate(sents):
        gap = text[pos:s["start"]]
        parts.append("<br><br>" if "\n" in gap else (" " if gap else ""))
        body = esc(s["text"])
        if s["read"] == "none":
            parts.append(f'<span class="s-unread">{body}</span>')
        elif i in marks:
            css, label = marks[i]
            parts.append(f'<span class="{css}">{body}</span><span class="mark">{label}</span>')
        else:
            parts.append(body)
        pos = s["end"]
    return "".join(parts)


def sentence_rows(items):
    return "".join(f'<div class="i">{lab}</div><div>{esc(t)}</div><div class="v">{v:+.3f}</div>'
                   for lab, t, v in items)


def render_tier2(res, example):
    unread = res["n_words"] - res["words_read"]
    read_note = (f'the model read all {res["n_words"]} words' if unread <= 0 else
                 f'the model read the first {res["words_read"]} of {res["n_words"]} words '
                 f'(its limit is 512 tokens); the rest is shown in grey')
    st.markdown(f'<div class="verdict">{T2_DISPLAY[res["verdict"]]}</div>'
                f'<div class="verdict-sub">Credibility score {res["credibility"]:.3f} against a threshold '
                f'of {res["threshold"]:.3f} &middot; {read_note}</div>', unsafe_allow_html=True)

    # criteria, not-met first, then by how far past the threshold
    crits = sorted(res["criteria"], key=lambda c: (c["met"], -(c["p_fail"] - c["threshold"])))
    rows = ['<div class="h">Criterion</div><div class="h">Model</div>'
            '<div class="h">P(not met)</div><div class="h" style="text-align:right">AUC</div>']
    for c in crits:
        status = "Met" if c["met"] else "Not met"
        rows.append(f'<div title="{esc(c["question"])}">{esc(c["name"])}</div>'
                    f'<div class="st {"met" if c["met"] else ""}">{status}</div>'
                    f'<div class="bar-track"><div class="bar-fill" style="width:{c["p_fail"]*100:.1f}%"></div>'
                    f'<div class="tick" style="left:{c["threshold"]*100:.1f}%"></div></div>'
                    f'<div class="n">{(c["auc"] or 0):.2f}</div>')
    st.markdown('<div class="section">Reporting criteria (HealthNewsReview.org)</div>'
                f'<div class="crit">{"".join(rows)}</div>'
                '<div class="verdict-sub" style="margin-top:0.8rem">The bar is the model\'s probability '
                'that the criterion is not met; the tick is the point above which it counts as not met. '
                'AUC is how well the model predicts that criterion on held-out articles (0.5 is chance), '
                'so criteria with a low AUC deserve less trust.</div>', unsafe_allow_html=True)

    ex = res.get("explanation")
    sents = res["sentences"]
    read_idx = [i for i, s in enumerate(sents) if s["read"] != "none"]
    marks, lo, hi = {}, [], []
    if ex:
        cred = ex["credibility"]
        floor = MIN_SHARE * max((abs(v) for v in cred), default=0.0)
        order = sorted(range(len(cred)), key=lambda j: cred[j])
        lo = [j for j in order[:TOP_K] if cred[j] < 0 and abs(cred[j]) >= floor]
        hi = [j for j in order[::-1][:TOP_K] if cred[j] > 0 and abs(cred[j]) >= floor]
        for r, j in enumerate(lo, 1):
            marks[read_idx[j]] = ("s-lo", f"&minus;{r}")
        for r, j in enumerate(hi, 1):
            marks[read_idx[j]] = ("s-hi", f"+{r}")

    st.markdown('<div class="section">Article</div>'
                f'<div class="art">{render_article(res["text"], sents, marks)}</div>',
                unsafe_allow_html=True)

    if ex:
        items = [(f"&minus;{r}", ex["sentences"][j], ex["credibility"][j]) for r, j in enumerate(lo, 1)]
        items += [(f"+{r}", ex["sentences"][j], ex["credibility"][j]) for r, j in enumerate(hi, 1)]
        st.markdown('<div class="section">Sentences that moved the verdict &middot; SHAP</div>'
                    f'<div class="sents">{sentence_rows(items)}</div>'
                    '<div class="verdict-sub">Solid underline (&minus;): pushed the article toward '
                    '"falls short". Dotted underline (+): pushed it toward "meets standards". The number '
                    'is the change in credibility score attributed to that sentence. Sentences with less '
                    'than a fifth of the largest effect are not listed.</div>',
                    unsafe_allow_html=True)
        failed = [c for c in crits if not c["met"]]
        if failed:
            st.markdown('<div class="section">Why each criterion is not met &middot; SHAP</div>',
                        unsafe_allow_html=True)
        for c in failed:
            v = ex["criteria"][c["key"]]
            floor = max(0.005, MIN_SHARE * max((abs(x) for x in v), default=0.0))
            top = [j for j in sorted(range(len(v)), key=lambda j: -v[j])[:TOP_K] if v[j] >= floor]
            with st.expander(c["name"]):
                if top:
                    st.markdown(f'<div class="sents">{sentence_rows([(str(r), ex["sentences"][j], v[j]) for r, j in enumerate(top, 1)])}</div>'
                                '<div class="verdict-sub">Sentences that most raised the model\'s '
                                'probability that this criterion is not met.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="verdict-sub">No single sentence has a noticeable effect. '
                                'The judgement rests on the article as a whole, usually on what it '
                                'leaves out.</div>', unsafe_allow_html=True)

    if example:
        lab = "not credible" if example["binary_label"] == "NOT_CREDIBLE" else "credible"
        st.markdown('<div class="section">What the human reviewers said</div>'
                    f'<div class="review">HealthNewsReview.org rated this article <b>{example["rating"]} '
                    f'of 5 stars</b> ({lab}; 2 stars or fewer counts as not credible). '
                    f'{esc(clip_sentences(example["summary"], 600))}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- page
def set_example():
    choice = st.session_state.get("example")
    st.session_state["_example"] = None
    if not choice or choice == "-":
        return
    kind, rest = choice.split(" | ", 1)
    if kind == "Article":
        ex = next(e for e in load_article_examples() if e["label"] == rest)
        st.session_state["text"] = ex["text"]
        st.session_state["_example"] = ex
    else:
        st.session_state["text"] = rest.split(" - ", 1)[1]


st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="eyebrow">Explainable health misinformation detection</div>'
    '<div class="title">Check a health claim or a health news article</div>'
    '<div class="byline">Team 10 &middot; Amrita School of Artificial Intelligence, Coimbatore</div>'
    f'<div class="lede">Paste a short health claim or a full news article. Text of {ROUTER_MAX_WORDS} '
    'words or fewer is treated as a <b>claim</b>: the system searches 6,559 peer-reviewed biomedical '
    'passages and decides whether the evidence supports or contradicts it. Longer text is treated as an '
    '<b>article</b>: the system judges whether it reports health science responsibly, against the ten '
    'criteria used by HealthNewsReview.org. Both tiers show what their decision rested on.</div>',
    unsafe_allow_html=True)

qp = st.query_params
if not st.session_state.get("_qp_done") and ("text" in qp or "example" in qp):
    if "example" in qp:
        ex = next((e for e in load_article_examples() if e["article_id"] == qp["example"]), None)
        if ex:
            st.session_state["text"], st.session_state["_example"] = ex["text"], ex
    else:
        st.session_state["text"] = qp["text"]
    if "explain" in qp:
        st.session_state["explain"] = qp["explain"] != "0"
    st.session_state["_qp_done"] = True
    st.session_state["_auto"] = True

options = (["-"] + [f"Claim | {g} - {c}" for g, cs in CLAIM_EXAMPLES.items() for c in cs]
           + [f"Article | {e['label']}" for e in load_article_examples()])
st.selectbox("Load an example", options, key="example", on_change=set_example,
             format_func=lambda o: "Choose an example claim or article..." if o == "-" else o)

text = st.text_area("Claim or article", key="text", height=200,
                    placeholder="e.g. Wearing face masks reduces the spread of COVID-19.")
n = word_count(text)
auto = route(text) if n else None
choice = st.radio("Tier", ["Automatic", "Tier 1 - claim", "Tier 2 - article"], key="tier",
                  horizontal=True)
tier = {"Automatic": auto, "Tier 1 - claim": "tier1", "Tier 2 - article": "tier2"}[choice]
if n:
    why = (f"{n} words, so it is treated as {'a claim (Tier 1)' if auto == 'tier1' else 'an article (Tier 2)'}"
           if choice == "Automatic" else f"{n} words; tier chosen manually")
    st.markdown(f'<div class="route">{why}</div>', unsafe_allow_html=True)

evidence = None
if tier == "tier1":
    if st.checkbox("Use evidence I provide instead of searching the corpus", key="own_ev"):
        evidence = st.text_area("Evidence", key="evidence", height=110,
                                placeholder="Paste a passage, e.g. the abstract of a study.")
st.session_state.setdefault("explain", True)
explain = st.checkbox("Explain the result (SHAP)", key="explain")

run = st.button("Analyse", type="primary") or st.session_state.pop("_auto", False)

if run:
    if not (text or "").strip():
        st.markdown('<div class="notice">Enter a claim or an article first.</div>', unsafe_allow_html=True)
        st.stop()
    if evidence is not None and not evidence.strip():
        st.markdown('<div class="notice">Paste some evidence, or untick the evidence box.</div>',
                    unsafe_allow_html=True)
        st.stop()

    with st.spinner("Loading the models (first run only, up to a minute)..."):
        pipe = load_pipeline()
        eng = pipe.tier1 if tier == "tier1" else pipe.tier2

    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    if tier == "tier1":
        st.markdown('<div class="section">Result &middot; Tier 1 &middot; claim verification</div>',
                    unsafe_allow_html=True)
        with st.spinner("Verifying..."):
            res = eng.verify(text.strip(), evidence=(evidence.strip() if evidence else None))
        render_tier1(res, explain, eng)
    else:
        st.markdown('<div class="section">Result &middot; Tier 2 &middot; article reporting quality</div>',
                    unsafe_allow_html=True)
        with st.spinner("Assessing the article" + (" and computing sentence attributions (SHAP)..."
                                                   if explain else "...")):
            res = eng.assess(text.strip(), explain=explain)
        ex = st.session_state.get("_example")
        render_tier2(res, ex if ex and ex["text"].strip() == text.strip() else None)

with st.expander("How it works"):
    st.markdown(
        '<ol class="steps">'
        f'<li><b>Route.</b> Text of {ROUTER_MAX_WORDS} words or fewer goes to Tier 1, longer text to Tier 2. '
        'The two tiers answer different questions.</li>'
        '<li><b>Tier 1, claims.</b> BM25 and PubMedBERT embeddings each rank 6,559 biomedical passages, '
        'merged by Reciprocal Rank Fusion. If the best passage has cosine similarity below '
        f'{ABSTAIN_BELOW:.2f} the system declines. Otherwise a PubMedBERT verifier fine-tuned on HealthVer '
        'and SciFact reads the claim with the top three passages. SHAP scores each word.</li>'
        '<li><b>Tier 2, articles.</b> Five PubMedBERT models, trained on FakeHealth with an ordinal rating '
        'head and ten criteria heads, read the first 512 tokens. Their averaged criteria predictions are '
        'combined into the verdict. The criteria shown blend these predictions with a separate word-based '
        'model for each criterion, which tells the criteria apart better. SHAP scores each sentence by '
        'deleting it and measuring the change.</li>'
        '</ol>', unsafe_allow_html=True)

st.markdown('<div class="foot">Tier 1 describes what the retrieved evidence says about a claim. Tier 2 judges '
            'how an article reports, not whether its claims are true. Neither is medical advice.<br>'
            'B.Tech project &middot; Team 10 &middot; Supervisor: Dr. C. R. Lekshmi</div>',
            unsafe_allow_html=True)
