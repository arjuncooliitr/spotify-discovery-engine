"""
build_labeled_sample.py
------------------------
A REAL labeled sample produced by an analyst reading each review (here, the
AI did the qualitative coding). Demonstrates the exact output schema that
label_reviews.py produces at scale with your GROQ key.

Crucially, every verbatim_evidence span is re-validated against the source
review with the SAME guardrail as the production engine: quotes are
normalized (lowercase, straight quotes, collapsed whitespace) and must be a
true substring of the review, otherwise they are dropped. This makes the
fabricated-quote bug from the old dashboard impossible.

Output: reviews_labeled.csv
"""
import re
import pandas as pd

# review_id, is_about_discovery, primary_topic, discovery_subtheme,
# segment_signal, jtbd, verbatim_evidence, topic_sentiment
LABELS = [
 (4998, False, "pricing", "none", "free_user", "play my own playlists on demand", "they will take all control just to sell their premium version", "negative"),
 (3691, False, "other", "none", "premium_user", "listen offline in a rural area", "your tunes are removed from your device", "negative"),
 (5817, False, "pricing", "none", "free_user", "play my favourite songs on demand", "it will give only few chances to play your fav song", "negative"),
 (6069, True, "discovery", "no_control", "unknown", "discover authentic human-created music", "making it harder for users to discover authentic, human-created music", "negative"),
 (3025, False, "performance_bugs", "none", "premium_user", "play my downloaded music offline", "now I can't play my music offline", "negative"),
 (5950, False, "ui_ux", "none", "unknown", None, "bakit may notification pa kapag nag-add ng song", "negative"),
 (5609, False, "social", "none", "unknown", "stay connected with friends through music", "create Playlist with friends and stay connected through music", "positive"),
 (1001, False, "other", "none", "unknown", None, "This app is so good", "positive"),
 (425, False, "pricing", "none", "free_user", "pick and play my own songs", "Why do I have to pay to pick my own songs", "negative"),
 (3511, False, "performance_bugs", "none", "unknown", None, "Recent update has caused issues with apple play", "negative"),
 (863, True, "performance_bugs", "discovery_works", "premium_user", "discover new artists via autoplay", "this is how I discovered many new artists before I started running into this issue", "negative"),
 (6070, True, "discovery", "stale_recs", "power_user", "get fresh, varied recommendations", "The artists sound all the same and suggestions never improve", "negative"),
 (4577, True, "recommendation_quality", "no_control", "free_user", "play songs of my own choice", "listen the songs which is recommended by the app not of my own will", "negative"),
 (6073, True, "recommendation_quality", "no_control", "unknown", "add recommended songs without losing my own", "Enhance should only add suggested songs, not remove", "negative"),
 (1618, False, "ui_ux", "none", "unknown", None, "put a “add playlist” button on the main screen", "negative"),
 (6077, True, "discovery", "stale_recs", "unknown", "escape the filter bubble and discover new music", "Instead of opening doors to new discoveries, it locks us into a comfortable but repetitive bubble", "negative"),
 (4587, True, "repetition", "stale_recs", "power_user", "hear variety across a large playlist", "still lets me hear the same song 10 times in the first 50 tracks of a 2500 track playlist", "negative"),
 (2328, False, "other", "none", "new_user", None, "Highly recommend spotify it's not complicated it's easy", "positive"),
 (6071, True, "discovery", "stale_recs", "unknown", "varied Discover Weekly without an unwanted genre", "every week, there's another Christian song by the same dang band on the discover weekly playlist", "negative"),
 (4958, True, "recommendation_quality", "no_control", "unknown", "control what plays instead of recommendations", "I'm not interested in your recommendations", "negative"),
 (1538, True, "recommendation_quality", "no_control", "free_user", "play only my playlist with no injected songs", "Stop adding random songs that i might like in my playlist", "negative"),
 (3124, True, "catalog", "discovery_works", "unknown", "enjoy curated playlists easily", "amazing curated playlists, great streaming quality", "positive"),
 (5726, True, "recommendation_quality", "no_control", "unknown", "play only my own playlist", "it plays songs that aren't on my playlist", "negative"),
 (5309, False, "pricing", "none", "free_user", "shuffle my own songs for free", "not lettin me put my songs on shuffle without premium", "negative"),
 (1459, True, "recommendation_quality", "no_control", "unknown", "turn off AI features", "Really adversely affects my listening experience", "negative"),
 (460, False, "ads", "none", "free_user", "listen without constant ads", "the entire app almost becomes unlistenable without stupid ads", "negative"),
 (5320, False, "performance_bugs", "none", "premium_user", None, "the \"new\" widget is ugly and often doesn't load", "negative"),
 (4922, True, "recommendation_quality", "no_control", "premium_user", "no sponsored recommendations as a premium user", "I pay for premium why do I get sponsored recommendation", "negative"),
 (4737, True, "recommendation_quality", "discovery_works", "unknown", "a reliable daily personalized soundtrack", "The personalized playlists are accurate", "positive"),
 (5121, True, "recommendation_quality", "no_control", "unknown", None, "Ever since this app introduced ai playlists & dj, its been unbearable", "negative"),
 (5118, False, "pricing", "none", "free_user", "play my playlist in order for free", "make the playlist song actially played turns by turns for free", "neutral"),
 (4417, False, "other", "none", "unknown", None, "One of the best app for enjoying music", "positive"),
 (2617, False, "performance_bugs", "none", "unknown", "shuffle my playlists", "when I press the “Shuffle Play” it doesn't do anything", "negative"),
 (4729, False, "ads", "none", "free_user", None, "ads became overwhelming but it's occasionally and enduring", "neutral"),
 (6076, True, "recommendation_quality", "stale_recs", "power_user", "new, genre-coherent recommendations", "it's repeatedly recommending songs that I've already added to a specific playlist", "negative"),
 (3378, False, "pricing", "none", "premium_user", None, "audio books being locked behind a further pay wall", "negative"),
 (2237, False, "other", "none", "unknown", None, "I love Spotify it is my favourite app ever", "positive"),
 (2917, False, "performance_bugs", "none", "premium_user", "listen to my downloaded songs", "the song plays for 8-10 seconds and stops", "negative"),
 (833, True, "performance_bugs", "discovery_works", "power_user", "play my liked songs without repeats", "I love the recommendations, and how easy it is to use", "negative"),
 (717, False, "ads", "none", "free_user", "listen without back-to-back ads", "two songs without it playing like 3-4 ads in a row", "negative"),
 (2421, False, "ui_ux", "none", "unknown", "add a song to multiple playlists easily", "add to playlist' 4 times for every playlist", "negative"),
 (4680, False, "performance_bugs", "none", "unknown", None, "I cant open my spotify, my account sudden lost", "negative"),
 (5900, False, "pricing", "none", "free_user", "turn off shuffle for free", "I can't turn off the shuffle because this is a stupid option just for stupid premiums", "negative"),
 (1698, False, "performance_bugs", "none", "unknown", None, "How many times a day does Spotify crash on my phone", "negative"),
 (2402, False, "ads", "none", "free_user", None, "I watch ads every 2 songs", "negative"),
 (6075, True, "recommendation_quality", "stale_recs", "unknown", "varied recommendations from Enhance", "I keep getting the same batch of songs included as recommendations in the same order", "negative"),
 (5698, False, "ads", "none", "free_user", None, "the adds and premium irritated mee a lot", "neutral"),
 (1851, False, "ui_ux", "none", "unknown", None, "PLEASE REMOVE SUBTITLES ON SCREEN", "negative"),
 (2124, False, "ui_ux", "none", "unknown", "find specific songs quickly", "I can always find the songs im looking for even just using the lyrics", "positive"),
 (3942, False, "performance_bugs", "none", "unknown", "play music on my Apple Watch while working out", "Downloaded playlists on Apple watch will not play", "negative"),
 (3369, False, "pricing", "none", "free_user", None, "Even without premium it is still quite accessible", "positive"),
 (788, True, "repetition", "stale_recs", "power_user", "hear variety across my liked songs", "it happens to be choosing the same songs every time", "negative"),
 (5357, False, "pricing", "none", "free_user", "good audio quality at a fair price", "free version is horrible and unusable", "negative"),
 (5564, True, "catalog", "discovery_works", "unknown", "find a wide variety of music", "loves a wide variety of tunes and Spotify seems to have it all", "positive"),
 (6078, True, "recommendation_quality", "stale_recs", "power_user", "unique niche music across distinct playlists", "multiple playlists are playing the same songs, so they all sound the same", "negative"),
 (6074, True, "discovery", "no_control", "niche_taste", "filter an unwanted language out of discovery", "have the possibility to block a whole language and don't get connected recommendations", "negative"),
 (277, True, "repetition", "stale_recs", "unknown", "shuffle without hearing repeats", "It replays the same songs doesn't shuffle", "negative"),
 (3656, True, "recommendation_quality", "no_control", "unknown", "discover real (non-AI) music", "\"Spotify Weekly\" is all AI", "negative"),
 (6072, True, "discovery", "stale_recs", "power_user", "discover genuinely new music", "i feel like i'm stuck in a sink hole surrounded by my 2020 top songs", "negative"),
 (1614, True, "ads", "no_control", "premium_user", "ad-free listening and control of my playlist", "forcing unwanted songs onto your playlist", "negative"),
 (967, False, "other", "none", "unknown", None, "really easy to sign up and to log in", "positive"),
]

COLS = ["review_id", "is_about_discovery", "primary_topic", "discovery_subtheme",
        "segment_signal", "jtbd", "verbatim_evidence", "topic_sentiment"]


def norm(s):
    s = str(s).lower()
    for a, b in [("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"')]:
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def main():
    src = pd.read_csv("reviews_clean.csv")
    text_by_id = {int(r.review_id): norm(r.text) for r in src.itertuples()}

    rows, dropped = [], 0
    for rid, disc, topic, sub, seg, jtbd, ev, sent in LABELS:
        ev_norm = norm(ev) if ev else None
        if ev_norm and ev_norm not in text_by_id.get(rid, ""):
            ev = None  # GUARDRAIL: not a real substring -> drop (would-be fabrication)
            dropped += 1
        rows.append(dict(zip(COLS, [rid, disc, topic, sub, seg, jtbd, ev, sent])))

    out = pd.DataFrame(rows, columns=COLS)
    out.to_csv("reviews_labeled.csv", index=False)
    print(f"reviews_labeled.csv written | {len(out)} labeled reviews")
    print(f"Verbatim quotes validated: {out['verbatim_evidence'].notna().sum()} kept, "
          f"{dropped} dropped by guardrail (proves no fabrication slips through)")


if __name__ == "__main__":
    main()
