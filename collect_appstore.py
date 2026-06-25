"""
collect_appstore.py
--------------------
Collect Spotify App Store (iOS) reviews via Apple's official customer-reviews
RSS feed. This is the reliable, no-API-key way to pull App Store reviews.

Endpoint:
  https://itunes.apple.com/{country}/rss/customerreviews/id={app_id}/sortBy={sort}/page={n}/json

Apple caps each feed at 10 pages x 50 reviews = 500 per (country, sort).
We pull several English-speaking storefronts and both sort orders, then
de-duplicate by review id, so we end up with a few thousand unique reviews.

Output: appstore_reviews.csv  with columns [text, rating, date, author, country, Source]
"""

import time
import requests
import pandas as pd

APP_ID = "324684580"  # Spotify - Music and Podcasts (iOS)
COUNTRIES = ["us", "gb", "ca", "au", "ie", "nz", "in"]  # English storefronts
SORTS = ["mostRecent", "mostHelpful"]
MAX_PAGES = 10  # Apple hard limit
HEADERS = {"User-Agent": "Mozilla/5.0 (review-research)"}

rows = []
seen_ids = set()

for country in COUNTRIES:
    for sort in SORTS:
        for page in range(1, MAX_PAGES + 1):
            url = (
                f"https://itunes.apple.com/{country}/rss/customerreviews/"
                f"id={APP_ID}/sortBy={sort}/page={page}/json"
            )
            try:
                r = requests.get(url, headers=HEADERS, timeout=15)
                if r.status_code != 200:
                    break
                feed = r.json().get("feed", {})
                entries = feed.get("entry", [])
                # The first entry is app metadata, not a review -> skip if no rating
                page_reviews = 0
                for e in entries:
                    if "im:rating" not in e:
                        continue  # app summary entry
                    rid = e.get("id", {}).get("label", "")
                    if rid in seen_ids:
                        continue
                    seen_ids.add(rid)
                    title = e.get("title", {}).get("label", "")
                    content = e.get("content", {}).get("label", "")
                    text = (title + ". " + content).strip(". ").strip()
                    rows.append({
                        "text": text,
                        "rating": int(e.get("im:rating", {}).get("label", 0) or 0),
                        "date": e.get("updated", {}).get("label", ""),
                        "author": e.get("author", {}).get("name", {}).get("label", ""),
                        "country": country,
                        "Source": "App Store",
                    })
                    page_reviews += 1
                print(f"{country}/{sort}/page{page}: +{page_reviews} (total {len(rows)})")
                if page_reviews == 0:
                    break
                time.sleep(0.3)  # be polite
            except Exception as ex:
                print(f"  ! {country}/{sort}/page{page} failed: {ex}")
                break

df = pd.DataFrame(rows)
# keep reviews with at least a few words of signal
df = df[df["text"].str.split().str.len() >= 5].reset_index(drop=True)
df.to_csv("appstore_reviews.csv", index=False)
print(f"\nSaved appstore_reviews.csv  |  {len(df)} unique reviews")
print(df["rating"].value_counts().sort_index())
