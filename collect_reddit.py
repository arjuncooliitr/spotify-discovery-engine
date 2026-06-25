"""
collect_reddit.py  (FALLBACK - requires free Reddit API credentials)
---------------------------------------------------------------------
Reddit blocks unauthenticated scraping (HTTP 403), so the only reliable way
to collect Reddit discussions is the official API via PRAW.

ONE-TIME SETUP (2 minutes, free):
  1. Go to https://www.reddit.com/prefs/apps  ->  "create another app"
  2. Choose type "script". Note the client_id (under the app name) and secret.
  3. pip install praw
  4. Set env vars (or paste below):
       export REDDIT_CLIENT_ID=...
       export REDDIT_CLIENT_SECRET=...
       export REDDIT_USER_AGENT="spotify-research by u/yourname"

Then: python collect_reddit.py
Output: reddit_reviews.csv  with columns [text, rating, date, source]
"""
import os
import pandas as pd

try:
    import praw
except ImportError:
    raise SystemExit("Run: pip install praw")


def load_env(path=".env"):
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
if not os.environ.get("REDDIT_CLIENT_ID"):
    raise SystemExit("Reddit credentials not found. Add them to .env (see .env.example).")

reddit = praw.Reddit(
    client_id=os.environ["REDDIT_CLIENT_ID"],
    client_secret=os.environ["REDDIT_CLIENT_SECRET"],
    user_agent=os.environ.get("REDDIT_USER_AGENT", "spotify-research"),
)

SUBREDDITS = ["spotify", "truespotify"]
QUERIES = ["discover", "recommendations", "repetitive", "same songs",
           "discover weekly", "algorithm", "new music"]

rows, seen = [], set()
for sub in SUBREDDITS:
    for q in QUERIES:
        for post in reddit.subreddit(sub).search(q, limit=40, sort="relevance"):
            if post.id in seen:
                continue
            seen.add(post.id)
            body = (post.title or "") + ". " + (post.selftext or "")
            rows.append({"text": body.strip(), "rating": None,
                         "date": post.created_utc, "source": "Reddit"})
            # also grab a few top comments (often the richest complaints)
            post.comments.replace_more(limit=0)
            for c in post.comments[:5]:
                rows.append({"text": c.body, "rating": None,
                             "date": c.created_utc, "source": "Reddit"})

df = pd.DataFrame(rows)
df = df[df["text"].str.split().str.len() >= 5].drop_duplicates("text")
df.to_csv("reddit_reviews.csv", index=False)
print(f"Saved reddit_reviews.csv | {len(df)} Reddit items")
