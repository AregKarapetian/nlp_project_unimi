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

    # Bar chart
    metrics = ["norm_ged", "relation_overlap", "subject_role_alignment", "object_role_alignment"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4))
    for ax, m in zip(axes, metrics):
        ax.bar(summary["target_culture"], summary[m])
        ax.set_title(m)
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "comparison_by_culture.png"), dpi=150)
    print(f"\nSaved chart to {OUT_DIR}/comparison_by_culture.png")


if __name__ == "__main__":
    main()
