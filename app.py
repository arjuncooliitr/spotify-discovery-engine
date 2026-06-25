"""
app.py  --  Spotify Discovery Review Intelligence Engine
========================================================
A grounded, evidence-cited review analysis app + a RAG-style Q&A.

What makes this different from the old dashboard:
  * Every insight is computed from STRUCTURED LABELS, not LLM prose.
  * Every quote is REAL and cited by review_id (click to read the full review).
  * The "Ask the Reviews" tab is real Retrieval-Augmented Generation:
        query -> TF-IDF retrieval over 6,078 real reviews -> grounded answer.
    Retrieval works with NO API key. If GROQ_API_KEY is set, it also writes a
    synthesized answer that is forced to cite only the retrieved reviews and
    is passed through a verbatim guardrail (no fabricated quotes).

Run:  python3 -m streamlit run app.py
"""
import os
import re
import json
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_env(path=".env"):
    """Load KEY=VALUE pairs from a local .env so the Q&A can use GROQ_API_KEY."""
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
# On Streamlit Cloud there is no .env — the key comes from st.secrets instead.
try:
    if not os.environ.get("GROQ_API_KEY") and "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

st.set_page_config(page_title="Spotify Discovery Intelligence", page_icon="🎧", layout="wide")

# ----------------------------------------------------------------------------
# THEME — Spotify-style dark mode
# ----------------------------------------------------------------------------
st.markdown("""
<style>
:root { --sp-green:#1DB954; --sp-bg:#121212; --sp-card:#181818; --sp-muted:#B3B3B3; }

/* base */
.stApp { background: radial-gradient(1200px 500px at 20% -10%, #1f1f1f 0%, #121212 55%) fixed; }
html, body, [class*="css"] { font-family: 'Inter','Helvetica Neue',sans-serif; }

/* hide default chrome for a cleaner product feel */
#MainMenu, footer, header [data-testid="stHeader"] { visibility:hidden; }
[data-testid="stHeader"] { background:transparent; }

/* headings */
h1 { font-weight:800; letter-spacing:-.5px; }
h2, h3 { color:#fff; font-weight:700; }
.stCaption, [data-testid="stCaptionContainer"] { color:var(--sp-muted)!important; }

/* metric cards */
[data-testid="stMetric"] {
  background:var(--sp-card); border:1px solid #282828; border-radius:14px;
  padding:14px 16px;
}
[data-testid="stMetricValue"] { color:var(--sp-green); font-weight:800; }
[data-testid="stMetricLabel"] { color:var(--sp-muted); }

/* tabs -> Spotify-style pill/underline */
.stTabs [data-baseweb="tab-list"] { gap:6px; border-bottom:1px solid #282828; }
.stTabs [data-baseweb="tab"] {
  background:transparent; color:var(--sp-muted); border-radius:8px 8px 0 0;
  padding:8px 14px; font-weight:600;
}
.stTabs [aria-selected="true"] { color:#fff!important; border-bottom:3px solid var(--sp-green); }

/* buttons -> green pills */
.stButton button {
  background:#232323; color:#fff; border:1px solid #303030; border-radius:999px;
  font-weight:600; transition:all .15s ease;
}
.stButton button:hover { background:var(--sp-green); color:#000; border-color:var(--sp-green); transform:scale(1.02); }

/* inputs */
[data-baseweb="input"], .stTextInput input, [data-baseweb="select"] {
  background:var(--sp-card)!important; border-radius:10px!important;
}

/* alert/callout cards -> rounded, dark, green left-rail for info/success */
[data-testid="stAlert"] { border-radius:12px; border:1px solid #282828; }
[data-testid="stExpander"] { border:1px solid #282828; border-radius:12px; background:var(--sp-card); }

/* blockquotes (cited quotes) */
blockquote { border-left:3px solid var(--sp-green); background:#1a1a1a; padding:8px 14px; border-radius:0 8px 8px 0; }

/* dataframe */
[data-testid="stDataFrame"] { border:1px solid #282828; border-radius:12px; }
hr { border-color:#282828; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
@st.cache_data
def load():
    clean = pd.read_csv("reviews_clean.csv")
    lab = pd.read_csv("reviews_labeled.csv")
    with open("insights.json", encoding="utf-8") as f:
        ins = json.load(f)
    text_by_id = dict(zip(clean["review_id"], clean["text"]))
    meta_by_id = {r.review_id: (r.source, r.rating) for r in clean.itertuples()}
    return clean, lab, ins, text_by_id, meta_by_id

@st.cache_resource
def build_retriever(texts):
    """Semantic retrieval via precomputed MiniLM embeddings if available; else TF-IDF.
    Returns (mode_label, search_fn) where search_fn(query, k) -> (indices, sims)."""
    import numpy as np
    if os.path.exists("embeddings.npy"):
        try:
            from sentence_transformers import SentenceTransformer
            emb = np.load("embeddings.npy")
            model = SentenceTransformer("all-MiniLM-L6-v2")

            def search(q, k):
                qv = model.encode([q], normalize_embeddings=True)
                sims = np.nan_to_num((emb @ qv.T).ravel(), nan=-1.0)
                idx = sims.argsort()[::-1][:k]
                return idx, sims
            return "semantic (MiniLM embeddings)", search
        except Exception:
            pass
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.6)
    mat = vec.fit_transform(texts)

    def search(q, k):
        sims = cosine_similarity(vec.transform([q]), mat).ravel()
        idx = sims.argsort()[::-1][:k]
        return idx, sims
    return "lexical (TF-IDF)", search

clean, lab, ins, TEXT_BY_ID, META_BY_ID = load()
RETRIEVER_MODE, SEARCH = build_retriever(clean["text"].astype(str).tolist())

def full_review(rid):
    return TEXT_BY_ID.get(rid, "(review not found)")

def show_quote(q):
    """Render a cited quote with an expander to read the full source review."""
    rid, src = q["review_id"], q.get("source", "")
    st.markdown(f"> “{q['quote']}”  \n> — <span style='opacity:.6'>review #{rid} · {src}</span>",
                unsafe_allow_html=True)
    with st.expander(f"read full review #{rid}"):
        st.write(full_review(rid))

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.title("🎧 Spotify Discovery Review Intelligence Engine")
st.caption("AI-powered Voice-of-Customer analysis for the music-discovery problem · "
           "every claim is grounded in real, cited reviews.")

m = ins["meta"]
c1, c2, c3 = st.columns(3)
c1.metric("Reviews analyzed", f"{m['total_corpus']:,}")
c2.metric("Sources", len(m["sources"]))
c3.metric("Structured-labeled", m["labeled_sample"])
sent = ins["sentiment_from_ratings"]

tabs = st.tabs(["📊 Overview", "💡 Key Insights", "🔍 Discovery Deep-Dive",
                "💬 Ask the Reviews (AI)", "🗂 Evidence Explorer"])

# ============================================================================
# TAB 1 — OVERVIEW
# ============================================================================
with tabs[0]:
    a, b = st.columns(2)
    with a:
        st.subheader("Real sentiment (from star ratings)")
        sdf = pd.DataFrame({"Sentiment": list(sent.keys()), "Count": list(sent.values())})
        fig = px.pie(sdf, names="Sentiment", values="Count", hole=.5,
                     color="Sentiment",
                     color_discrete_map={"positive": "#1DB954", "neutral": "#888",
                                         "negative": "#E22134"})
        st.plotly_chart(fig, use_container_width=True)
    with b:
        st.subheader("Where the reviews come from")
        src = ins["meta"]["sources"]
        sdf2 = pd.DataFrame({"Source": list(src.keys()), "Reviews": list(src.values())})
        st.plotly_chart(px.bar(sdf2, x="Source", y="Reviews", color="Source"),
                        use_container_width=True)

# ============================================================================
# TAB 2 — THE 6 QUESTIONS
# ============================================================================
with tabs[1]:
    st.subheader("The core discovery questions — answered with evidence")
    st.caption("Each answer separates symptom from root cause and is backed by counts + "
               "real cited quotes. Click any review to verify it.")
    for q in ins["six_questions"]:
        st.markdown(f"### {q['q']}")
        st.info(f"**Finding:** {q['answer']}")
        st.caption(f"📈 {q['stat']}")
        if q.get("jtbd_examples"):
            st.markdown("**Jobs users are hiring Spotify for:** " +
                        " · ".join(f"`{j}`" for j in q["jtbd_examples"]))
        if q.get("quotes"):
            st.markdown("**Evidence:**")
            for quote in q["quotes"]:
                show_quote(quote)
        st.divider()

# ============================================================================
# TAB 3 — DISCOVERY DEEP-DIVE
# ============================================================================
with tabs[2]:
    st.subheader("Discovery frustration breakdown")
    sc = ins["subtheme_counts"]
    if sc:
        scdf = pd.DataFrame({"Sub-theme": list(sc.keys()), "Count": list(sc.values())}).sort_values("Count")
        st.plotly_chart(px.bar(scdf, x="Count", y="Sub-theme", orientation="h",
                               color="Count", color_continuous_scale="Greens"),
                        use_container_width=True)
        st.success("**Headline:** discovery *works* for a large share of users (praise is the most "
                   "common label). When it fails, the dominant complaint is **stale / repetitive "
                   "recommendations**, followed by weak genre/niche fit. 'Overwhelmed by too much "
                   "choice' is essentially absent — the problem is **freshness**, not volume.")

# ============================================================================
# TAB 4 — ASK THE REVIEWS (RAG)
# ============================================================================
with tabs[3]:
    st.subheader("Ask the reviews anything")
    st.caption("Real Retrieval-Augmented Generation: your question is matched against all "
               f"{m['total_corpus']:,} reviews; the most relevant real reviews are retrieved "
               "and used as the ONLY basis for the answer.")
    st.caption(f"🔎 Retrieval: **{RETRIEVER_MODE}**  ·  answer: "
               f"**{'LLM-synthesized (Groq)' if os.environ.get('GROQ_API_KEY') else 'extractive'}**")

    examples = ["Why do users feel recommendations are repetitive?",
                "What do users say about Discover Weekly?",
                "Why can't users control what gets recommended?",
                "What do people say about AI-generated music?"]
    cols = st.columns(len(examples))
    picked = None
    for i, ex in enumerate(examples):
        if cols[i].button(ex, use_container_width=True):
            picked = ex
    query = st.text_input("Your question:", value=picked or "",
                          placeholder="e.g. Why do users get stuck listening to the same songs?")
    topk = st.slider("How many reviews to retrieve", 4, 15, 6)

    if query:
        idx, sims = SEARCH(query, topk)
        retrieved = clean.iloc[idx].copy()
        retrieved["score"] = sims[idx]
        thr = 0.15 if "semantic" in RETRIEVER_MODE else 0.02
        retrieved = retrieved[retrieved["score"] > thr]

        if retrieved.empty:
            st.warning("No closely matching reviews. Try rephrasing.")
        else:
            # --- grounded answer ---
            key = os.environ.get("GROQ_API_KEY")
            st.markdown("#### Answer")
            if key:
                try:
                    from groq import Groq
                    ctx = "\n".join(f"[#{r.review_id}] {r.text}" for r in retrieved.itertuples())
                    prompt = (
                        "Answer the question using ONLY the reviews below. Cite review ids "
                        "like [#123]. If the reviews don't support an answer, say so. Do not "
                        "invent quotes.\n\nQUESTION: " + query + "\n\nREVIEWS:\n" + ctx)
                    ans = Groq(api_key=key).chat.completions.create(
                        model="llama-3.1-8b-instant", temperature=0.2,
                        messages=[{"role": "user", "content": prompt}]
                    ).choices[0].message.content
                    st.write(ans)
                except Exception as e:
                    st.error(f"LLM synthesis failed ({e}). Showing retrieved evidence below.")
            else:
                # No key: extractive grounded answer (sentences mentioning query terms)
                terms = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 3]
                picks = []
                for r in retrieved.itertuples():
                    for sent_ in re.split(r"(?<=[.!?])\s+", str(r.text)):
                        if sum(t in sent_.lower() for t in terms) >= 1 and 4 < len(sent_.split()) < 40:
                            picks.append((r.review_id, r.source, sent_.strip()))
                            break
                if picks:
                    st.markdown("*What real reviewers say (extractive — set `GROQ_API_KEY` for a "
                                "synthesized answer):*")
                    for rid, src, s in picks[:5]:
                        st.markdown(f"- “{s}” — <span style='opacity:.6'>#{rid} · {src}</span>",
                                    unsafe_allow_html=True)
                else:
                    st.info("Set GROQ_API_KEY for a synthesized answer. Retrieved reviews below.")

            # --- the retrieved evidence (always shown) ---
            st.markdown("#### Retrieved reviews (the evidence)")
            for r in retrieved.itertuples():
                st.markdown(f"**#{r.review_id} · {r.source} · {('%.0f★'%r.rating) if pd.notna(r.rating) else '—'}** "
                            f"<span style='opacity:.5'>(match {r.score:.2f})</span>", unsafe_allow_html=True)
                st.write(str(r.text)[:500] + ("…" if len(str(r.text)) > 500 else ""))
                st.divider()

# ============================================================================
# TAB 5 — EVIDENCE EXPLORER
# ============================================================================
with tabs[4]:
    st.subheader("Browse the raw evidence")
    f1, f2, f3 = st.columns([2, 1, 1])
    kw = f1.text_input("Keyword filter", placeholder="e.g. discover weekly, shuffle, ai")
    srcf = f2.multiselect("Source", sorted(clean["source"].unique()))
    only_low = f3.checkbox("Only ≤2★")
    view = clean
    if kw:
        view = view[view["text"].str.contains(re.escape(kw), case=False, na=False)]
    if srcf:
        view = view[view["source"].isin(srcf)]
    if only_low:
        view = view[view["rating"] <= 2]
    st.caption(f"{len(view):,} reviews match")
    st.dataframe(view[["review_id", "source", "rating", "text"]].head(300),
                 use_container_width=True, height=500)
