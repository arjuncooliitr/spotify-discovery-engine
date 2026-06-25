# 🎧 Spotify Discovery Review Intelligence Engine — Phase 1

An AI-powered Voice-of-Customer engine that analyzes thousands of real Spotify
reviews to answer **why users struggle to discover new music** — with every
insight grounded in real, cited reviews (no fabrication).

Built for the Growth-PM fellowship problem: *increase meaningful music discovery
and reduce repetitive listening.*

---

## Why this engine is different
The first version dumped reviews into an LLM and asked for prose → generic,
unverifiable answers, and even **fabricated user quotes**. This version fixes the
three root causes:

1. **The data now contains the answer.** Recovered App Store reviews (0 → 4,540)
   and merged everything into one clean schema. Discovery-relevant content went
   from **0.7% → ~14%** of the corpus.
2. **Insights are counted, not narrated.** Each review is labeled into structured
   JSON; the 6 questions are answered by **aggregating labels**, with real counts.
3. **Fabrication is impossible.** Every quote is **verbatim-validated** against the
   source review or dropped. Sentiment comes from **star ratings**, not a guessing
   lexicon.

---

## Pipeline
```
collect_appstore.py ─┐
  (Apple RSS, 4,540) │
spotify_reviews_*.csv ├─► build_dataset.py ─► reviews_clean.csv  (7,296 reviews, 1 schema)
  (Play Store, 1,469) │         │
youtube / community ─┘         ▼
  (1,120 from 10 videos)  label_reviews.py  ─► reviews_labeled.csv  (structured JSON + stratum)
                        (LLM, per-review, guardrail)        │
                                                            ▼
                                                  analyze.py ─► insights.json  (grounded answers)
                                                            │
                                                            ▼
                                                       app.py  (dashboard + RAG Q&A)
```

## Files
| File | Role |
|---|---|
| `collect_appstore.py` | Pull App Store reviews via Apple's RSS feed (no key) |
| `collect_youtube.py` | Pull comments from discovery-focused Spotify videos (no key) |
| `collect_community.py` | Scrape Spotify Community discovery threads, full post text (no key) |
| `collect_reddit.py` | Reddit collector via PRAW (optional; needs free API keys) |
| `build_dataset.py` | Merge all sources into one clean `text` schema → `reviews_clean.csv` |
| `label_reviews.py` | **Intelligence core** — label each review into structured JSON (topic, discovery sub-theme, segment, JTBD, verbatim quote). Verbatim guardrail. |
| `build_labeled_sample.py` | A 61-review hand-labeled demo sample (used until the full run) |
| `analyze.py` | Grounded synthesis → `insights.json` (counts + cited quotes; unbiased prevalence from control sample) |
| `app.py` | Streamlit app: evidence dashboard + RAG "Ask the Reviews" Q&A |
| `run_pipeline.sh` | One command to run the at-scale pipeline |

## The 6 questions it answers
1. Why do users struggle to discover new music?
2. What are the most common frustrations with recommendations?
3. What listening behaviors are users trying to achieve?
4. What causes users to repeatedly listen to the same content?
5. Which user segments experience different discovery challenges?
6. What unmet needs emerge consistently across reviews?

**Headline finding:** the discovery problem is about **control + freshness**
(users can't steer the algorithm, and it recycles the familiar) — *not* "too much
music to choose from."

---

## How to run

**View the app (works now, on the 61-review demo sample):**
```bash
python3 -m streamlit run app.py        # → http://localhost:8502
```

**Finish at scale (labels ~1,002 reviews — needs a Groq key):**
```bash
cp .env.example .env        # then paste your ROTATED GROQ_API_KEY
bash run_pipeline.sh        # label_reviews → analyze
python3 -m streamlit run app.py
```
Optional AI-synthesized answers in the Q&A: ensure `GROQ_API_KEY` is set before launching.

## Data sources
| Source | Reviews | Method |
|---|---|---|
| App Store | 4,540 | Apple RSS (7 English storefronts) |
| Play Store | 1,469 | google-play-scraper |
| YouTube | 1,120 | comments from 10 discovery-focused videos (`collect_youtube.py`) |
| Community | 167 | scraped discovery threads (`collect_community.py`) + original export |
| Reddit | — | OAuth-blocked; collector included, runs with API keys |

## Status
- ✅ Engine built, verified, app running on the demo sample
- ⏳ At-scale labeling pending a (rotated) Groq key → run `bash run_pipeline.sh`
