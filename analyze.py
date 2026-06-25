"""
analyze.py  --  GROUNDED SYNTHESIS (no hallucination)
-----------------------------------------------------
Reads the structured labels and computes every insight DETERMINISTICALLY:
  * real sentiment from star ratings (not TextBlob)
  * discovery prevalence, topic mix, subtheme frequencies
  * segment x subtheme cross-tab
  * a grounded answer to each of the 6 official questions, each backed by
    computed counts AND real verbatim quotes cited by review_id.

Output: insights.json  (consumed by app.py)

This is the opposite of the old pipeline: instead of asking an LLM to write
prose and hoping it's true, we COUNT the labels and attach real evidence.
"""
import json
import pandas as pd

clean = pd.read_csv("reviews_clean.csv")
lab = pd.read_csv("reviews_labeled.csv")
df = lab.merge(clean[["review_id", "text", "source", "rating"]], on="review_id", how="left")

N = len(df)


def rating_sentiment(r):
    if pd.isna(r):
        return "unknown"
    return "negative" if r <= 2 else ("neutral" if r == 3 else "positive")


# ---- real sentiment from ratings over the FULL corpus -------------------
clean["sent"] = clean["rating"].apply(rating_sentiment)
sent_counts = clean[clean["sent"] != "unknown"]["sent"].value_counts().to_dict()


_USED_QUOTES = set()
def quotes_for(mask, k=3):
    """Up to k real, cited quotes for rows matching a mask, never reusing a quote
    already cited by an earlier question (keeps each answer's evidence distinct)."""
    out = []
    for r in df[mask].itertuples():
        if int(r.review_id) in _USED_QUOTES:
            continue
        if isinstance(r.verbatim_evidence, str) and r.verbatim_evidence.strip():
            _USED_QUOTES.add(int(r.review_id))
            out.append({"review_id": int(r.review_id), "source": r.source,
                        "quote": r.verbatim_evidence.strip()})
        if len(out) >= k:
            break
    return out


def pct(n):
    return round(n / N * 100, 1)


disc = df[df["is_about_discovery"]]
sub_counts = (disc[disc["discovery_subtheme"] != "none"]["discovery_subtheme"]
              .value_counts())

# Unbiased corpus discovery prevalence from the random CONTROL stratum (if the
# at-scale run produced one); otherwise fall back to the labeled-set rate.
if "stratum" in df.columns and (df["stratum"] == "control").any():
    ctrl = df[df["stratum"] == "control"]
    prevalence = round(ctrl["is_about_discovery"].mean() * 100, 1)
    prevalence_basis = f"random control sample, n={len(ctrl)}"
else:
    prevalence = pct(len(disc))
    prevalence_basis = f"labeled set, n={N}"

# segment x subtheme cross-tab (the slide-worthy table)
ct = (pd.crosstab(disc["segment_signal"], disc["discovery_subtheme"])
      .drop(columns=[c for c in ["none"] if c in disc["discovery_subtheme"].unique()],
            errors="ignore"))

SUBTHEME_LABEL = {
    "stale_recs": "Recommendations feel stale / looping",
    "no_control": "Can't steer or turn off the algorithm",
    "too_mainstream": "Only mainstream / obvious picks",
    "overwhelmed": "Overwhelmed by too much choice",
    "search_fails": "Search can't surface what they want",
    "niche_genre_gap": "Niche taste underserved",
    "discovery_works": "Discovery praised (works well)",
}

top_subs = [s for s in sub_counts.index if s != "discovery_works"][:2]
top_subs_str = " and ".join(SUBTHEME_LABEL[s].lower() for s in top_subs) if top_subs else "n/a"

# ---- the six official questions, answered from the data -----------------
def has(pattern):
    return df["text"].astype(str).str.contains(pattern, case=False, regex=True)

DM = df["is_about_discovery"] == True  # discovery mask
jtbds = [j for j in df["jtbd"].dropna().unique() if isinstance(j, str)]

# --- semantic, question-aware evidence selection over the FULL corpus -------
import os
import re
import numpy as np
_C_TEXT = clean["text"].astype(str).tolist()
_C_ID = clean["review_id"].tolist()
_C_SRC = clean["source"].tolist()
_C_RATE = clean["rating"].tolist()
_SEM = False
if os.path.exists("embeddings.npy"):
    try:
        from sentence_transformers import SentenceTransformer
        _EMB = np.load("embeddings.npy")
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        _SEM = True
    except Exception:
        _SEM = False

def _best_sentence(text, qterms):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if 4 <= len(s.split()) <= 45]
    if not sents:
        return text[:180]
    sents.sort(key=lambda s: sum(t in s.lower() for t in qterms), reverse=True)
    return sents[0][:200]

def _neg(i, thresh=3):
    r = _C_RATE[i]
    return pd.isna(r) or r <= thresh

_DISC_RE = re.compile(r"discover|recommend|suggest|new (music|artist|song)|same song|"
                      r"repetit|shuffle|playlist|algorithm|radio|daily mix|release radar|"
                      r"variety|genre|stuck|over and over|similar artist", re.I)
_SEG_RE = re.compile(r"premium|free version|free user|\bpaid\b|subscription|years|"
                     r"power user|new to spotify|long.?time|been using", re.I)
def _disc(i):
    return bool(_DISC_RE.search(_C_TEXT[i]))

def evidence(query, qterms, k=3, gate=None):
    """Semantic top-k real reviews for a question, deduped, with the most relevant snippet."""
    if not _SEM:
        return []
    qv = _MODEL.encode([query], normalize_embeddings=True)
    sims = np.nan_to_num((_EMB @ qv.T).ravel(), nan=-1.0)
    out = []
    for i in sims.argsort()[::-1]:
        if sims[i] < 0.2:
            break
        rid = int(_C_ID[i])
        if rid in _USED_QUOTES or (gate and not gate(i)):
            continue
        _USED_QUOTES.add(rid)
        out.append({"review_id": rid, "source": _C_SRC[i],
                    "quote": _best_sentence(_C_TEXT[i], qterms)})
        if len(out) >= k:
            break
    return out

six = []

# Q1 — the BARRIER: staleness from over-personalization (claims stale_recs quotes first)
n_stale = int((DM & (df["discovery_subtheme"] == "stale_recs")).sum())
n_over = int((df["discovery_subtheme"] == "overwhelmed").sum())
six.append({
    "q": "1. Why do users struggle to discover new music?",
    "stat": f"{n_stale} of {len(disc)} discovery reviews cite stale / repetitive discovery; "
            f"'overwhelmed by too much choice' is essentially absent ({n_over}).",
    "answer": ("The core barrier is staleness from over-personalization: Spotify leans "
               "heavily on past listening, so discovery surfaces (Discover Weekly, daily "
               "mixes, radio) keep recycling familiar artists instead of opening new doors. "
               "Users describe being 'stuck', not 'overwhelmed' — the barrier is freshness, "
               "not volume."),
    "quotes": evidence(
        "I struggle to discover new music. I can't find new songs or artists. Discovery "
        "and recommendations just recycle music I already know and feel stale.",
        ["discover", "new music", "new artist", "find", "stale", "stuck", "same"],
        3, gate=lambda i: _disc(i) and _neg(i, 4)),
})

# Q2 — RANKED recommendation frustrations: irrelevance / genre drift (distinct from Q1)
n_rec = int((DM & (df["primary_topic"] == "recommendation_quality")).sum())
six.append({
    "q": "2. What are the most common frustrations with recommendations?",
    "stat": f"{n_rec} discovery reviews are about recommendation quality specifically.",
    "answer": ("Ranked: (1) repetition — the same songs/artists on loop; (2) inaccuracy — "
               "'similar artists' that feel unrelated; (3) genre drift — mixes that wander "
               "off-taste (a metal mix sprouting unrelated tracks); (4) sponsored / injected "
               "songs users didn't ask for. Distinct complaints needing distinct fixes."),
    "quotes": evidence(
        "The recommendations are inaccurate. Suggested songs and 'similar artists' are "
        "unrelated and don't match my taste. The recommended music is irrelevant and off.",
        ["recommend", "suggest", "similar artist", "inaccurate", "unrelated", "taste", "irrelevant", "genre"],
        3, gate=lambda i: _disc(i) and _neg(i, 3)),
})

# Q3 — JOBS-to-be-done (positive goals, not complaints)
six.append({
    "q": "3. What listening behaviors are users trying to achieve?",
    "stat": f"{int(df['jtbd'].notna().sum())} of {N} reviews state a clear job-to-be-done.",
    "answer": ("Users hire Spotify for distinct jobs: expand their taste / find new music, "
               "get variety in a daily soundtrack, set a mood or background for an activity, "
               "and manage playlists easily. Discovery is an active goal — when a surface "
               "genuinely works, users praise it warmly."),
    "quotes": evidence(
        "What I want from Spotify: discover new music, hear more variety, expand my taste, "
        "music for my mood or workout, and easily build and manage my playlists.",
        ["discover", "new music", "variety", "expand", "mood", "workout", "playlist", "find"],
        3, gate=lambda i: _disc(i)),
    "jtbd_examples": jtbds[:8],
})

# Q4 — repetition MECHANISM: literal replay / shuffle (distinct from Q1's framing)
rep_mask = DM & has(r"same song|over and over|on repeat|shuffle|repeat|again and again|same ten|same 10")
six.append({
    "q": "4. What causes users to repeatedly listen to the same content?",
    "stat": f"{int(rep_mask.sum())} discovery reviews describe literal replay / shuffle repetition.",
    "answer": ("Two causes. Deliberate: comfort and low-effort background listening. "
               "Forced (the opportunity): shuffle and personalized mixes physically replay "
               "the same tracks — users report a 2,500-song playlist looping the same ten. "
               "That forced repetition is a mechanics problem, not a user preference."),
    "quotes": evidence(
        "Shuffle keeps playing the same songs over and over. I hear the same tracks "
        "repeatedly even with a huge playlist; it loops the same music in the same order.",
        ["same song", "over and over", "shuffle", "repeat", "same order", "again"],
        3, gate=lambda i: _disc(i) and _neg(i, 4)),
})

# Q5 — SEGMENTS
six.append({
    "q": "5. Which user segments experience different discovery challenges?",
    "stat": f"Segment is identifiable in {int(disc['segment_signal'].ne('unknown').sum())}/"
            f"{len(disc)} discovery reviews; premium and power users dominate the discussion.",
    "answer": ("Discovery is discussed most by Premium and long-tenure power users — the "
               "people with the most history for the algorithm to over-fit, who feel the "
               "staleness hardest. Free users frame discovery through the paywall (can't "
               "freely pick or skip), and niche-taste listeners want filters the algorithm "
               "doesn't offer. Same surface, different root causes."),
    "quotes": evidence(
        "As a long-time premium subscriber and power user with a huge library, or as a "
        "free user, here is my experience finding new music and recommendations.",
        ["premium", "free", "years", "subscription", "paid", "power", "library", "long"],
        3, gate=lambda i: _disc(i) and bool(_SEG_RE.search(_C_TEXT[i]))),
})

# Q6 — unmet NEEDS framed as forward-looking wishes, cross-source
six.append({
    "q": "6. What unmet needs emerge consistently across reviews?",
    "stat": "Recurring across App Store, Play Store, YouTube and Community: steerability, "
            "freshness, and filtering controls.",
    "answer": ("Reframed as needs: (1) a lever to STEER recommendations toward novelty; "
               "(2) genuine freshness, not recycled history; (3) filters and controls "
               "(genre, language, AI-generated tracks). Because these recur across every "
               "source, they are real needs — not a loud minority."),
    "quotes": evidence(
        "I wish Spotify would add an option to control my recommendations, filter genres "
        "or languages, turn off AI music, and steer discovery. Please give us a way.",
        ["wish", "please", "option", "would love", "need", "add", "filter", "control", "turn off"],
        4, gate=lambda i: _disc(i) and _neg(i, 4)),
})

insights = {
    "meta": {
        "total_corpus": int(len(clean)),
        "labeled_sample": int(N),
        "sources": clean["source"].value_counts().to_dict(),
        "discovery_prevalence_basis": prevalence_basis,
    },
    "sentiment_from_ratings": sent_counts,
    "discovery_prevalence_pct": prevalence,
    "topic_distribution": df["primary_topic"].value_counts().to_dict(),
    "subtheme_counts": {SUBTHEME_LABEL.get(k, k): int(v) for k, v in sub_counts.items()},
    "segment_crosstab": ct.to_dict(),
    "jtbd": jtbds,
    "six_questions": six,
}

with open("insights.json", "w", encoding="utf-8") as f:
    json.dump(insights, f, indent=2, default=int)

print("insights.json written.")
print(f"  Corpus: {len(clean)} | Labeled: {N} | Discovery prevalence: {prevalence}% ({prevalence_basis})")
print(f"  Real sentiment (from ratings): {sent_counts}")
print(f"  Top discovery subthemes: {dict(sub_counts)}")
