"""
Compute semantic similarity between each original story and its retellings
using BERT sentence embeddings (all-MiniLM-L6-v2 via sentence-transformers).

This connects directly to L3/L5 course content (transformer-based text
representations) and complements the graph-structural metrics in compare_cultures.py
by measuring meaning-level preservation, not just graph shape.

Run from project root:
    python src/semantic_similarity.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

STORIES_CSV = "stories.csv"
RETELLINGS_DIR = os.path.join("results", "retellings")
OUT_DIR = "results/analysis"

MODEL_NAME = "all-MiniLM-L6-v2"
ALL_CULTURES = ["European", "Asian", "African_Diaspora"]
CULTURE_COLORS = {"European": "#4C72B0", "Asian": "#DD8452", "African_Diaspora": "#55A868"}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(STORIES_CSV)

    print(f"Loading sentence-transformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    rows = []
    for _, row in df.iterrows():
        story_id = str(row["story_id"])
        orig_culture = str(row["culture"])
        orig_text = str(row["text"])

        orig_emb = model.encode([orig_text])

        for target in [c for c in ALL_CULTURES if c != orig_culture]:
            path = os.path.join(RETELLINGS_DIR, f"{story_id}__to_{target}.txt")
            if not os.path.exists(path):
                print(f"[WARN] Missing retelling: {path}")
                continue

            with open(path, "r", encoding="utf-8") as f:
                retelling = f.read().strip()
            if not retelling:
                continue

            ret_emb = model.encode([retelling])
            sim = float(cosine_similarity(orig_emb, ret_emb)[0][0])

            rows.append({
                "story_id": story_id,
                "orig_culture": orig_culture,
                "target_culture": target,
                "semantic_similarity": sim,
            })
            print(f"[OK] {story_id} -> {target}: {sim:.3f}")

    details = pd.DataFrame(rows)
    details.to_csv(os.path.join(OUT_DIR, "semantic_similarity.csv"), index=False)
    print(f"\nSaved: {OUT_DIR}/semantic_similarity.csv")

    summary = details.groupby("target_culture")["semantic_similarity"].mean().reset_index()
    print("\nMean cosine similarity (original vs. retelling) by target culture:")
    print(summary.to_string(index=False))

    cultures = summary["target_culture"].tolist()
    x_pos = {c: i for i, c in enumerate(cultures)}

    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in summary.iterrows():
        c = r["target_culture"]
        ax.bar(x_pos[c], r["semantic_similarity"],
               color=CULTURE_COLORS.get(c, "steelblue"), alpha=0.75, width=0.5, zorder=2)

    for c in cultures:
        vals = details[details["target_culture"] == c]["semantic_similarity"].dropna()
        ax.scatter([x_pos[c]] * len(vals), vals, color="black", s=18, alpha=0.45, zorder=3)

    ax.set_xticks(range(len(cultures)))
    ax.set_xticklabels([c.replace("_", "\n") for c in cultures], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Cosine similarity (BERT embeddings)", fontsize=10)
    ax.set_title(
        "Semantic similarity: original vs. retelling\n"
        "(bars = mean, dots = individual stories)",
        fontsize=11,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "semantic_similarity.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved chart: {out_png}")


if __name__ == "__main__":
    main()
