import json
import os
from collections import defaultdict

import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRAPHML_DIR = "results/graphs/graphs"
ROLES_PATH = "results/graphs/role_sequences.jsonl"
OUT_DIR = "results/analysis"

STOPWORDS = {"the", "a", "an", "to", "of", "his", "her", "their", "its", "and", "for", "with", "on", "in"}


def relation_words(seq):
    words = set()
    for ev in seq:
        for w in ev["relation"].split():
            w = w.strip(".,!?;:").lower()
            if w and w not in STOPWORDS:
                words.add(w)
    return words


def normalized_ged(g1: nx.DiGraph, g2: nx.DiGraph, timeout=10) -> float:
    size = max(g1.number_of_nodes() + g1.number_of_edges(),
               g2.number_of_nodes() + g2.number_of_edges())
    if size == 0:
        return 0.0
    ged = nx.graph_edit_distance(g1, g2, timeout=timeout)
    if ged is None:
        return None
    return ged / size


def role_alignment(seq1, seq2):
    n = min(len(seq1), len(seq2))
    if n == 0:
        return None, None
    subj_matches = sum(1 for i in range(n) if seq1[i]["subject_role"] == seq2[i]["subject_role"])
    obj_pairs = [(seq1[i]["object_role"], seq2[i]["object_role"]) for i in range(n)
                  if seq1[i]["object_role"] is not None and seq2[i]["object_role"] is not None]
    obj_matches = sum(1 for a, b in obj_pairs if a == b)
    subj_score = subj_matches / n
    obj_score = (obj_matches / len(obj_pairs)) if obj_pairs else None
    return subj_score, obj_score


def load_graph(sid, ver):
    path = os.path.join(GRAPHML_DIR, f"{sid}__{ver}.graphml")
    g = nx.read_graphml(path)
    return nx.DiGraph(g)  # collapse multi-edges, drop labels for structural GED


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    roles = {}
    with open(ROLES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            roles[(r["story_id"], r["version"])] = r

    story_ids = sorted({sid for sid, _ in roles.keys()}, key=int)

    rows = []
    for sid in story_ids:
        orig = roles.get((sid, "orig"))
        if orig is None:
            continue
        for (s2, ver2), rec2 in roles.items():
            if s2 != sid or ver2 == "orig":
                continue
            target_culture = rec2["culture"]

            g1 = load_graph(sid, "orig")
            g2 = load_graph(sid, ver2)
            ged = normalized_ged(g1, g2)

            jac_num = relation_words(orig["sequence"]) & relation_words(rec2["sequence"])
            jac_den = relation_words(orig["sequence"]) | relation_words(rec2["sequence"])
            relation_overlap = len(jac_num) / len(jac_den) if jac_den else None

            subj_score, obj_score = role_alignment(orig["sequence"], rec2["sequence"])

            rows.append({
                "story_id": sid,
                "orig_culture": orig["culture"],
                "target_culture": target_culture,
                "norm_ged": ged,
                "relation_overlap": relation_overlap,
                "subject_role_alignment": subj_score,
                "object_role_alignment": obj_score,
            })

    details = pd.DataFrame(rows)
    details.to_csv(os.path.join(OUT_DIR, "comparison_details.csv"), index=False)

    summary = details.groupby("target_culture")[
        ["norm_ged", "relation_overlap", "subject_role_alignment", "object_role_alignment"]
    ].mean().reset_index()
    summary.to_csv(os.path.join(OUT_DIR, "comparison_summary.csv"), index=False)

    print(details)
    print("\nSummary by target culture:")
    print(summary)

    # Bar chart with individual story points overlaid
    metrics = ["norm_ged", "relation_overlap", "subject_role_alignment", "object_role_alignment"]
    metric_labels = {
        "norm_ged": "Normalized GED",
        "relation_overlap": "Relation overlap (Jaccard)",
        "subject_role_alignment": "Subject-role alignment",
        "object_role_alignment": "Object-role alignment",
    }
    cultures = summary["target_culture"].tolist()
    culture_colors = {"African_Diaspora": "#55A868", "Asian": "#DD8452", "European": "#4C72B0"}
    x_pos = {c: i for i, c in enumerate(cultures)}

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))
    for ax, m in zip(axes, metrics):
        for _, row in summary.iterrows():
            c = row["target_culture"]
            ax.bar(x_pos[c], row[m], color=culture_colors.get(c, "steelblue"),
                   alpha=0.75, width=0.5, zorder=2)
        # Overlay individual story dots
        for c in cultures:
            vals = details[details["target_culture"] == c][m].dropna()
            jitter = [x_pos[c] + 0.0] * len(vals)
            ax.scatter(jitter, vals, color="black", s=18, alpha=0.45, zorder=3)
        ax.set_xticks(range(len(cultures)))
        ax.set_xticklabels([c.replace("_", "\n") for c in cultures], fontsize=9)
        ax.set_title(metric_labels[m], fontsize=10, fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.suptitle("Cross-cultural comparison metrics (bars = mean, dots = individual stories)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "comparison_by_culture.png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved chart to {OUT_DIR}/comparison_by_culture.png")


if __name__ == "__main__":
    main()
