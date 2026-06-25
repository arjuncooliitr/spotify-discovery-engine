"""
relabel_70b.py
--------------
Sharpen the labels: re-label ONLY the discovery reviews on llama-3.3-70b-versatile
(more nuanced than 8B, which missed 'no_control' entirely). The discovery set
(~363) fits inside the 100k tokens/day cap; the control sample + non-discovery
rows keep their 8B labels (fine — they don't feed the subtheme chart).

Reuses label_reviews.py's prompt, schema, and verbatim guardrail. Saves the
merged file after every batch so an interruption never loses progress.
"""
import os
import time
import pandas as pd
from groq import Groq
import label_reviews as L

L.load_env()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"
BATCH = 12

lab = pd.read_csv("reviews_labeled.csv").set_index("review_id")
clean = pd.read_csv("reviews_clean.csv")
stratum_by_id = lab["stratum"].to_dict()
if "label_model" not in lab.columns:
    lab["label_model"] = "8b"

# discovery rows not yet upgraded to 70B (resumable, won't re-spend budget)
target_ids = lab[(lab["is_about_discovery"] == True) &
                 (lab["label_model"] != "70b")].index.tolist()
target_df = clean[clean["review_id"].isin(target_ids)].reset_index(drop=True)
print(f"Re-labeling {len(target_df)} discovery reviews on {MODEL} ...")

LABEL_COLS = ["is_about_discovery", "primary_topic", "discovery_subtheme",
              "segment_signal", "jtbd", "verbatim_evidence", "topic_sentiment"]
fabricated = 0
for i in range(0, len(target_df), BATCH):
    batch = target_df.iloc[i:i + BATCH]
    tbi = dict(zip(batch["review_id"], batch["text"].astype(str)))
    try:
        resp = client.chat.completions.create(
            model=MODEL, temperature=0.1,
            messages=[{"role": "system", "content": L.SYSTEM},
                      {"role": "user", "content": L.build_prompt(batch)}])
        n = 0
        for obj in L.parse_json_array(resp.choices[0].message.content):
            row, fab = L.coerce(obj, tbi, stratum_by_id)
            fabricated += int(fab)
            if row and row["review_id"] in lab.index:
                for c in LABEL_COLS:
                    lab.at[row["review_id"], c] = row[c]
                lab.at[row["review_id"], "label_model"] = "70b"
                n += 1
        lab.reset_index().to_csv("reviews_labeled.csv", index=False)  # save each batch
        print(f"  batch {i//BATCH + 1}/{(len(target_df)+BATCH-1)//BATCH}: +{n} (fab blocked {fabricated})")
        time.sleep(0.4)
    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e):
            print("  hit daily cap — stopping; re-labeled rows kept, rest stay 8B")
            break
        print(f"  ! batch {i//BATCH + 1} failed: {str(e)[:110]}")
        time.sleep(2)

lab = lab.reset_index()
lab.to_csv("reviews_labeled.csv", index=False)
disc = lab[lab.is_about_discovery == True]
print(f"\nDone. Fabricated blocked: {fabricated}")
print("Discovery subtheme mix now:", disc["discovery_subtheme"].value_counts().to_dict())
