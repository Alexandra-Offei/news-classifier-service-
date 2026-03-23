"""
save_model.py  —  Run this once in your Module 3 notebook/environment
to export your trained pipeline so the microservice can load it.

Paste this into your existing notebook after training, then copy
the output file (model_pipeline.pkl) into the news-classifier/ folder.
"""

import pickle
from sklearn.pipeline import Pipeline

# ── OPTION A: You have a separate vectorizer + classifier ─────────────────────
# Replace `vectorizer` and `clf` with your actual variable names.
#
# from sklearn.pipeline import Pipeline
# pipeline = Pipeline([("tfidf", vectorizer), ("clf", clf)])
# with open("model_pipeline.pkl", "wb") as f:
#     pickle.dump(pipeline, f)
# print("Saved! File size:", os.path.getsize("model_pipeline.pkl") / 1024, "KB")

# ── OPTION B: You already have a Pipeline object ──────────────────────────────
# Replace `my_pipeline` with your variable name.
#
# with open("model_pipeline.pkl", "wb") as f:
#     pickle.dump(my_pipeline, f)
# print("Saved!")

# ── OPTION C: Verify a saved file loads correctly ─────────────────────────────
import os
if os.path.exists("model_pipeline.pkl"):
    with open("model_pipeline.pkl", "rb") as f:
        loaded = pickle.load(f)
    test = loaded.predict(["NASA launches new Mars rover"])
    label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Science/Technology"}
    print("Model loaded successfully. Test prediction:", label_map[test[0]])
else:
    print("model_pipeline.pkl not found — run Option A or B first.")
