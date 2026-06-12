import os

import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

STATS_CSV = "results/graphs/graph_stats.csv"
OUT_DIR = "results/analysis"

FEATURES = [
    "n_nodes", "n_edges", "n_self_loops", "n_distinct_relations",
    "avg_out_degree", "max_out_degree", "density", "n_weakly_connected_components",
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(STATS_CSV)
    orig = df[df["version"] == "orig"].copy()
    orig = orig.sort_values("story_id", key=lambda s: s.astype(int)).reset_index(drop=True)

    X = orig[FEATURES].values
    X_scaled = StandardScaler().fit_transform(X)

    k = 3  # one cluster per culture, as a starting point
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    orig["kmeans_cluster"] = km.fit_predict(X_scaled)

    Z = linkage(X_scaled, method="ward")
    orig["hier_cluster"] = fcluster(Z, t=k, criterion="maxclust")

    out_cols = ["story_id", "culture", "kmeans_cluster", "hier_cluster"] + FEATURES
    orig[out_cols].to_csv(os.path.join(OUT_DIR, "clusters.csv"), index=False)

    for col in ["kmeans_cluster", "hier_cluster"]:
        ct = pd.crosstab(orig[col], orig["culture"])
        ct.to_csv(os.path.join(OUT_DIR, f"{col}_vs_culture.csv"))
        print(f"\n{col} vs culture:\n{ct}")

    # Dendrogram
    labels = [f"{sid}-{cul}" for sid, cul in zip(orig["story_id"], orig["culture"])]
    plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=labels, leaf_rotation=90)
    plt.title("Hierarchical clustering of original-story narrative graphs")
    plt.ylabel("Ward distance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "dendrogram.png"), dpi=150)
    print(f"\nSaved dendrogram to {OUT_DIR}/dendrogram.png")


if __name__ == "__main__":
    main()
