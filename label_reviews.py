"""
label_reviews.py  --  THE INTELLIGENCE CORE (at-scale)
-------------------------------------------------------
Labels each review into structured JSON so insights can be COUNTED and CITED,
not hallucinated.

To keep the run cheap/fast we don't label all 6,078 reviews. We label:
  * EVERY discovery-relevant review  (high value -> deep subtheme/segment/quote analysis)
  * PLUS a random CONTROL sample      (unbiased -> lets us estimate true corpus
                                        discovery prevalence & topic mix)
Each row is tagged with `stratum` = relevant | control so analyze.py can report
an unbiased prevalence from the control stratum while using ALL discovery
reviews for evidence depth.

Guardrail: verbatim_evidence must be a real substring of the review or it is
dropped -> fabricated quotes are impossible.

Setup (no keys in code or chat):
  1. Rotate your old leaked Groq key, then create a .env file (see .env.example):
        GROQ_API_KEY=gsk_your_new_key
        CONTROL_SAMPLE=500          # optional, default 500
  2. python label_reviews.py
Resumable: re-run after any interruption; it skips already-labeled reviews.
"""
import os
import re
import json
import time
import pandas as pd
from groq import Groq

MODEL = "llama-3.1-8b-instant"   # 70b hit its 100k tokens/day cap; 8b has a separate, larger budget
BATCH = 12
IN_FILE = "reviews_clean.csv"
OUT_FILE = "reviews_labeled.csv"

# High-precision "is this about the discovery / recommendation experience or
# repetition?" pre-filter (same definition that yielded ~502 relevant reviews).
DISCOVERY_RE = re.compile(
    r"(discover weekly|release radar|daily mix|smart shuffle|made for you|\benhance\b|"
    r"autoplay|repetit|same song|same artist|over and over|on repeat|new music|"
    r"new artist|new song|filter bubble|algorithm|discover\b|discovery|suggested song|"
    r"recommend(?:s|ed|ation|ing) (?:me|the|same|songs|music|playlist|artist))", re.I)

TOPICS = ["discovery", "recommendation_quality", "repetition", "ads", "pricing",
          "performance_bugs", "ui_ux", "catalog", "podcasts", "social", "other"]
SUBTHEMES = ["stale_recs", "too_mainstream", "no_control", "overwhelmed",
             "search_fails", "niche_genre_gap", "discovery_works", "none"]
SEGMENTS = ["new_user", "power_user", "free_user", "premium_user", "niche_taste", "unknown"]
SENTIMENT = ["positive", "negative", "neutral"]


def load_env(path=".env"):
    """Load KEY=VALUE lines from a local .env into the environment (no dependency)."""
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


SYSTEM = (
    "You are a meticulous user-research analyst doing qualitative coding of app "
    "reviews. You label one review at a time. You NEVER infer beyond what the text "
    "says. If a review does not mention something, you use the 'none'/'unknown'/null/"
    "false value. You copy evidence VERBATIM and never paraphrase. You output ONLY a "
    "valid JSON array, nothing else."
)

PROMPT_TMPL = """Label each review below. Return a JSON array with one object per review:
{{
  "review_id": <int, copy exactly>,
  "is_about_discovery": <true|false: discusses finding/recommending NEW music, or repetition of music?>,
  "primary_topic": <one of {topics}>,
  "discovery_subtheme": <one of {subthemes}; "none" if not about discovery. stale_recs=repetitive/looping; too_mainstream=only popular picks; no_control=can't steer/turn off; overwhelmed=too much choice; search_fails=can't find; niche_genre_gap=taste underserved; discovery_works=praises discovery>,
  "segment_signal": <one of {segments}; who the user is, else "unknown">,
  "jtbd": <short phrase: the job they're trying to get done, or null>,
  "verbatim_evidence": <the EXACT span copied char-for-char (<=180 chars) supporting your labels, or null>,
  "topic_sentiment": <one of {sentiment}>
}}
Rules: copy verbatim_evidence character-for-character (do NOT paraphrase or fix typos). If unsure use none/unknown/null/false. Output ONLY the JSON array.

REVIEWS:
{block}"""


def build_prompt(batch_df):
    block = "\n".join(f'[review_id={r.review_id}] {r.text}' for r in batch_df.itertuples())
    return PROMPT_TMPL.format(topics=TOPICS, subthemes=SUBTHEMES, segments=SEGMENTS,
                             sentiment=SENTIMENT, block=block)


def parse_json_array(content):
    content = content.replace("```json", "").replace("```", "").strip()
    s, e = content.find("["), content.rfind("]")
    if s == -1 or e == -1:
        return []
    try:
        return json.loads(content[s:e + 1])
    except json.JSONDecodeError:
        return []


def norm(s):
    s = str(s).lower()
    for a, b in [("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"')]:
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def coerce(obj, text_by_id, stratum_by_id):
    rid = obj.get("review_id")
    if rid not in text_by_id:
        return None, False
    src = norm(text_by_id[rid])
    ev = obj.get("verbatim_evidence")
    fabricated = False
    if ev and isinstance(ev, str):
        if norm(ev) not in src:   # GUARDRAIL
            ev, fabricated = None, True
    pick = lambda v, allowed, d: v if v in allowed else d
    row = {
        "review_id": rid,
        "is_about_discovery": bool(obj.get("is_about_discovery", False)),
        "primary_topic": pick(obj.get("primary_topic"), TOPICS, "other"),
        "discovery_subtheme": pick(obj.get("discovery_subtheme"), SUBTHEMES, "none"),
        "segment_signal": pick(obj.get("segment_signal"), SEGMENTS, "unknown"),
        "jtbd": obj.get("jtbd") or None,
        "verbatim_evidence": ev,
        "topic_sentiment": pick(obj.get("topic_sentiment"), SENTIMENT, "neutral"),
        "stratum": stratum_by_id.get(rid, "relevant"),
    }
    return row, fabricated


def build_target(df):
    """Relevant reviews + a random control sample, each tagged with a stratum."""
    control_n = int(os.environ.get("CONTROL_SAMPLE", "500"))
    mask = df["text"].astype(str).str.contains(DISCOVERY_RE)
    relevant = df[mask].copy()
    relevant["stratum"] = "relevant"
    rest = df[~mask]
    control = rest.sample(min(control_n, len(rest)), random_state=42).copy()
    control["stratum"] = "control"
    target = pd.concat([relevant, control]).drop_duplicates("review_id").reset_index(drop=True)
    print(f"Target set: {len(relevant)} relevant + {len(control)} control = {len(target)} reviews")
    return target


def main():
    load_env()
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY not found. Add it to a .env file (see .env.example).")
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    df = pd.read_csv(IN_FILE)
    target = build_target(df)
    stratum_by_id = dict(zip(target["review_id"], target["stratum"]))

    done = set()
    if os.path.exists(OUT_FILE):
        prev = pd.read_csv(OUT_FILE)
        if "stratum" in prev.columns:           # only resume a real at-scale run
            done = set(prev["review_id"].tolist())
            print(f"Resuming: {len(done)} already labeled.")
    todo = target[~target["review_id"].isin(done)].reset_index(drop=True)
    n_batches = (len(todo) + BATCH - 1) // BATCH
    print(f"To label: {len(todo)} reviews in {n_batches} batches (~{n_batches*0.7:.0f}s + API time)")

    fabricated = 0
    for i in range(0, len(todo), BATCH):
        batch = todo.iloc[i:i + BATCH]
        text_by_id = dict(zip(batch["review_id"], batch["text"].astype(str)))
        try:
            resp = client.chat.completions.create(
                model=MODEL, temperature=0.1,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": build_prompt(batch)}])
            rows = []
            for obj in parse_json_array(resp.choices[0].message.content):
                row, fab = coerce(obj, text_by_id, stratum_by_id)
                fabricated += int(fab)
                if row:
                    rows.append(row)
            if rows:
                pd.DataFrame(rows).to_csv(OUT_FILE, mode="a", index=False,
                                          header=not os.path.exists(OUT_FILE))
            print(f"  batch {i//BATCH + 1}/{n_batches}: +{len(rows)} (fabricated blocked: {fabricated})")
            time.sleep(0.4)
        except Exception as ex:
            print(f"  ! batch {i//BATCH + 1} failed: {ex} -- rerun to resume")
            time.sleep(3)

    print(f"\nDone. Fabricated quotes blocked by guardrail: {fabricated}")
    print(f"Now run:  python analyze.py")


if __name__ == "__main__":
    main()
