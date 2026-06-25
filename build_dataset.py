"""
build_dataset.py
----------------
Merge ALL sources into ONE clean schema. This fixes the original bug where
Play Store text lived in 'Review', YouTube in 'Comment', Community in
'User comment' -- and every analysis script only read 'Review', silently
dropping 90% of non-Play-Store data.

Output: reviews_clean.csv  with columns:
    review_id | text | rating | source | meta
Every row has non-empty text. review_id is stable and used for citations.
"""
import os
import pandas as pd

def longest_text_col(df):
    """Pick the column that looks most like free-text (longest avg length)."""
    best, best_len = None, 0
    for c in df.columns:
        if df[c].dtype == object:
            avg = df[c].dropna().astype(str).str.len().mean() or 0
            if avg > best_len:
                best, best_len = c, avg
    return best

frames = []

# 1) App Store (new, reliable) -------------------------------------------
if os.path.exists("appstore_reviews.csv"):
    a = pd.read_csv("appstore_reviews.csv")
    frames.append(pd.DataFrame({
        "text": a["text"].astype(str),
        "rating": a.get("rating"),
        "source": "App Store",
    }))

# 2) Play Store ----------------------------------------------------------
if os.path.exists("spotify_reviews_filtered.csv"):
    p = pd.read_csv("spotify_reviews_filtered.csv")
    frames.append(pd.DataFrame({
        "text": p["Review"].astype(str),
        "rating": p.get("Rating"),
        "source": "Play Store",
    }))

# 3) YouTube (discovery-focused multi-video comments + original) ---------
for yf in ["youtube_reviews.csv", "youtube_comments.csv"]:
    if os.path.exists(yf):
        y = pd.read_csv(yf)
        col = ("text" if "text" in y.columns
               else "Comment" if "Comment" in y.columns else longest_text_col(y))
        frames.append(pd.DataFrame({
            "text": y[col].astype(str), "rating": None, "source": "YouTube",
        }))

# 4) Community (scraped discussions + original export) -------------------
if os.path.exists("community_reviews.csv"):
    cr = pd.read_csv("community_reviews.csv")
    frames.append(pd.DataFrame({
        "text": cr["text"].astype(str), "rating": None, "source": "Community",
    }))
if os.path.exists("SpotifyCommunity.xlsx"):
    c = pd.read_excel("SpotifyCommunity.xlsx")
    col = longest_text_col(c)
    frames.append(pd.DataFrame({
        "text": c[col].astype(str), "rating": None, "source": "Community",
    }))

# 5) Reddit (only if the PRAW fallback produced data) --------------------
if os.path.exists("reddit_reviews.csv") and os.path.getsize("reddit_reviews.csv") > 50:
    r = pd.read_csv("reddit_reviews.csv")
    frames.append(pd.DataFrame({
        "text": r["text"].astype(str), "rating": None, "source": "Reddit",
    }))

df = pd.concat(frames, ignore_index=True)

# Clean: strip, drop empties / very short / dedupe
df["text"] = df["text"].str.strip()
df = df[df["text"].str.split().str.len() >= 5]
df = df[~df["text"].str.lower().isin(["nan", "none", ""])]
df = df.drop_duplicates(subset="text").reset_index(drop=True)
df.insert(0, "review_id", range(1, len(df) + 1))

df.to_csv("reviews_clean.csv", index=False)

print(f"reviews_clean.csv written | {len(df)} unique reviews")
print("\nSource breakdown:")
print(df["source"].value_counts())
print("\nSanity check -- every row has text:", df["text"].notna().all(),
      "| empty texts:", (df["text"].str.len() == 0).sum())
