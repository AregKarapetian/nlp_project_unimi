"""
Generates two additional figures:
  results/analysis/pca_stories.png   - PCA of original stories by culture
  results/analysis/example_graphs.png - narrative graph visualizations for 2 stories
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

OUT_DIR = "results/analysis"
STATS_CSV = "results/graphs/graph_stats.csv"
GRAPHML_DIR = "results/graphs/graphs"

CULTURE_COLORS = {
    "European": "#4C72B0",
    "Asian": "#DD8452",
    "African_Diaspora": "#55A868",
}

FEATURES = [
    "n_nodes", "n_edges", "n_self_loops", "n_distinct_relations",
    "avg_out_degree", "max_out_degree", "density", "n_weakly_connected_components",
]


def plot_pca():
    df = pd.read_csv(STATS_CSV)
    orig = df[df["version"] == "orig"].copy().reset_index(drop=True)

    X = StandardScaler().fit_transform(orig[FEATURES])
    coords = PCA(n_components=2, random_state=42).fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 5))
    for culture, color in CULTURE_COLORS.items():
        mask = orig["culture"] == culture
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, label=culture.replace("_", " "),
            s=80, edgecolors="white", linewidths=0.5, alpha=0.9,
        )

    ax.set_xlabel("PC 1", fontsize=11)
    ax.set_ylabel("PC 2", fontsize=11)
    ax.set_title("PCA of original stories by narrative-graph structure", fontsize=12)
    ax.legend(title="Culture", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "pca_stories.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_example_graphs():
    # Pick two contrasting stories: one dense (story 9 - African Diaspora, 13 nodes)
    # and one sparse (story 1 - Asian, 8 nodes)
    examples = [
        ("9", "orig", "Story 9 (African Diaspora original)"),
        ("1", "orig", "Story 1 (Asian original)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (sid, ver, title) in zip(axes, examples):
        path = os.path.join(GRAPHML_DIR, f"{sid}__{ver}.graphml")
        g = nx.read_graphml(path)

        # Collapse multi-edges for a readable diagram
        simple = nx.DiGraph()
        for u, v, data in g.edges(data=True):
            if simple.has_edge(u, v):
                simple[u][v]["label"] += "\n" + data.get("relation", "")
            else:
                simple.add_edge(u, v, label=data.get("relation", ""))

        # Degree-based node size
        degrees = dict(simple.degree())
        sizes = [300 + degrees[n] * 150 for n in simple.nodes()]

        # Color nodes by role (most-active subject = red)
        subject_counts = {}
        for u, v in simple.edges():
            subject_counts[u] = subject_counts.get(u, 0) + 1
        ranked = sorted(subject_counts, key=subject_counts.get, reverse=True)
        colors = []
        for n in simple.nodes():
            if ranked and n == ranked[0]:
                colors.append("#E74C3C")   # protagonist
            elif len(ranked) > 1 and n == ranked[1]:
                colors.append("#F39C12")   # deuteragonist
            else:
                colors.append("#3498DB")   # other

        pos = nx.spring_layout(simple, seed=42, k=1.5)
        nx.draw_networkx_nodes(simple, pos, ax=ax, node_size=sizes,
                               node_color=colors, alpha=0.9)
        nx.draw_networkx_labels(simple, pos, ax=ax, font_size=7, font_color="white",
                                font_weight="bold")
        nx.draw_networkx_edges(simple, pos, ax=ax, arrows=True,
                               arrowstyle="-|>", arrowsize=15,
                               edge_color="#555", width=1.2,
                               connectionstyle="arc3,rad=0.1")

        # Edge labels (truncate long relations)
        edge_labels = {(u, v): d["label"].split("\n")[0][:20]
                       for u, v, d in simple.edges(data=True)}
        nx.draw_networkx_edge_labels(simple, pos, edge_labels=edge_labels,
                                     ax=ax, font_size=6, label_pos=0.35)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(color="#E74C3C", label="Protagonist"),
        mpatches.Patch(color="#F39C12", label="Deuteragonist"),
        mpatches.Patch(color="#3498DB", label="Other"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Example narrative graphs", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "example_graphs.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_pca()
    plot_example_graphs()
