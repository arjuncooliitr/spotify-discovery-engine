"""
collect_youtube.py
------------------
Pull comments from YouTube videos ABOUT Spotify's algorithm / recommendations /
music discovery (listener-perspective, not artist-marketing videos). Comments on
these videos are rich in real discovery opinions, unlike comments on a music video.

Writes incrementally (appends after each video) so a slow/interrupted run still
keeps progress. Output: youtube_reviews.csv  [text, rating, source, video_id]
"""
import os
import time
import pandas as pd
from youtube_comment_downloader import YoutubeCommentDownloader

# Listener-perspective videos about Spotify discovery / recommendations / algorithm.
VIDEOS = [
    ("yEYbgVZ-HRU", "How Spotify's Algorithm is Killing Culture"),
    ("cW9hwyxwK6k", "How the Spotify algorithm is destroying music"),
    ("bhXFHuwkOd8", "Is Spotify Actually Good For Music Discovery?"),
    ("trXHTH2C1mk", "Algorithms are ruining the music industry + your taste"),
    ("H-AVBZFjIZU", "Spotify's Algorithm Sucks"),
    ("gvLs-uU9QN8", "How Algorithms Changed Music Discovery"),
    ("JPXgCGaqreA", "I Switched From Spotify To Apple Music"),
    ("OlzP5ZLTH9U", "I Switched From Apple Music To Spotify"),
    ("egTTpfcHlV0", "Spotify vs Apple Music 2026"),
    ("setq8C_kKDk", "Spotify vs Apple Music - Which is Better?"),
]
PER_VIDEO = 180
MIN_WORDS = 8          # drop "love it 😍" style low-signal comments
OUT = "youtube_reviews.csv"

downloader = YoutubeCommentDownloader()
seen = set()
if os.path.exists(OUT):
    os.remove(OUT)

total = 0
for vid, title in VIDEOS:
    rows = []
    try:
        n = 0
        for c in downloader.get_comments_from_url(f"https://www.youtube.com/watch?v={vid}"):
            t = (c.get("text") or "").strip().replace("\n", " ")
            if len(t.split()) < MIN_WORDS or t in seen:
                continue
            seen.add(t)
            rows.append({"text": t, "rating": None, "source": "YouTube", "video_id": vid})
            n += 1
            if n >= PER_VIDEO:
                break
        if rows:
            pd.DataFrame(rows).to_csv(OUT, mode="a", index=False,
                                      header=not os.path.exists(OUT))
        total += len(rows)
        print(f"{vid}  {title[:38]:40} +{len(rows)}  (total {total})")
        time.sleep(1)
    except Exception as e:
        print(f"! {vid} failed: {e}")

print(f"\nSaved {OUT} | {total} substantive comments from {len(VIDEOS)} videos")
