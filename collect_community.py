"""
collect_community.py
--------------------
Collect Spotify Community (community.spotify.com) discussions about music
discovery / recommendations / repetition. The community runs on Khoros, whose
pages are server-rendered HTML — so we can fetch search results, follow the
thread links, and extract the FULL post text (not the truncated search snippet).

Strategy:
  1. Run discovery-related search queries -> collect unique thread URLs.
  2. Fetch each thread -> extract the opening post + a couple of replies
     (replies are where users pile on with the same complaint).
  3. Filter for substance, dedupe.

Writes incrementally. Output: community_reviews.csv  [text, rating, source, url]
"""
import os
import re
import time
import urllib.parse
import requests
import pandas as pd
from bs4 import BeautifulSoup

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")
HEADERS = {"User-Agent": UA}
BASE = "https://community.spotify.com"
OUT = "community_reviews.csv"

QUERIES = [
    "discover weekly", "music discovery", "recommendations repetitive",
    "release radar", "same songs over and over", "algorithm stuck",
    "daily mix repetitive", "smart shuffle", "made for you playlist",
    "recommend new music", "discovery bubble", "autoplay recommendations",
]
PAGES_PER_QUERY = 2
MAX_THREADS = 140
MSGS_PER_THREAD = 3       # opening post + up to 2 replies
MIN_WORDS = 10

LINK_RE = re.compile(r'/t5/[^"\']*?/(?:m-p|td-p)/\d+')


def get(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        return r.text if r.status_code == 200 else ""
    except Exception:
        return ""


# 1) gather unique thread URLs from search ----------------------------------
thread_urls = []
seen_urls = set()
for q in QUERIES:
    for page in range(1, PAGES_PER_QUERY + 1):
        u = (f"{BASE}/t5/forums/searchpage/tab/message?q="
             f"{urllib.parse.quote(q)}&page={page}")
        html = get(u)
        found = 0
        for m in LINK_RE.findall(html):
            full = BASE + m.split("?")[0]
            if full not in seen_urls:
                seen_urls.add(full)
                thread_urls.append(full)
                found += 1
        print(f"q='{q}' p{page}: +{found} threads (total {len(thread_urls)})")
        time.sleep(0.4)
        if len(thread_urls) >= MAX_THREADS:
            break
    if len(thread_urls) >= MAX_THREADS:
        break

thread_urls = thread_urls[:MAX_THREADS]
print(f"\nFetching {len(thread_urls)} threads for full post text...\n")

# 2) fetch each thread, extract full message bodies -------------------------
if os.path.exists(OUT):
    os.remove(OUT)
seen_text = set()
total = 0
for i, url in enumerate(thread_urls, 1):
    html = get(url)
    if not html:
        continue
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for body in soup.select(".lia-message-body-content")[:MSGS_PER_THREAD]:
        t = body.get_text(" ", strip=True)
        t = re.sub(r"\s+", " ", t).strip()
        if len(t.split()) < MIN_WORDS:
            continue
        key = t[:120].lower()
        if key in seen_text:
            continue
        seen_text.add(key)
        rows.append({"text": t, "rating": None, "source": "Community", "url": url})
    if rows:
        pd.DataFrame(rows).to_csv(OUT, mode="a", index=False,
                                  header=not os.path.exists(OUT))
        total += len(rows)
    if i % 20 == 0:
        print(f"  {i}/{len(thread_urls)} threads -> {total} posts")
    time.sleep(0.3)

print(f"\nSaved {OUT} | {total} community posts")
