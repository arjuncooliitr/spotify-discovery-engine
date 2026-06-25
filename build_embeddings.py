"""
build_embeddings.py
-------------------
Precompute semantic sentence-embeddings for every review so the app's
"Ask the Reviews" Q&A retrieves by MEANING, not just keyword overlap.
Run once after the corpus changes. Output: embeddings.npy
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_csv("reviews_clean.csv")
texts = df["text"].astype(str).tolist()
print(f"Encoding {len(texts)} reviews with all-MiniLM-L6-v2 ...")
model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                   normalize_embeddings=True).astype("float32")
np.save("embeddings.npy", emb)
print("Saved embeddings.npy", emb.shape)
