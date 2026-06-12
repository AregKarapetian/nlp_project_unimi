import json
import os
import re
from collections import Counter
from typing import Dict, Optional

import networkx as nx
import pandas as pd

TRIPLES_PATH = "results/graphs/triples.jsonl"
OUT_DIR = "results/graphs"
GRAPHML_DIR = os.path.join(OUT_DIR, "graphs")
STATS_CSV = os.path.join(OUT_DIR, "graph_stats.csv")
ROLES_PATH = os.path.join(OUT_DIR, "role_sequences.jsonl")

ARTICLES_RE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)


def normalize_entity(name: str) -> str:
    name = name.strip().lower()
    name = ARTICLES_RE.sub("", name)
    name = re.sub(r"\s+", " ", name)
    return name


def assign_roles(triples) -> Dict[str, str]:
    subj_counts = Counter()
    for t in triples:
        subj_counts[normalize_entity(t["subject"])] += 1

    ranked = [ent for ent, _ in subj_counts.most_common()]
    roles = {}
    for idx, ent in enumerate(ranked):
        if idx == 0:
            roles[ent] = "PROTAGONIST"
        elif idx == 1:
            roles[ent] = "DEUTERAGONIST"
        else:
            roles[ent] = "OTHER"
    return roles


def build_graph(triples) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for t in triples:
        s = normalize_entity(t["subject"])
        o = normalize_entity(t["object"]) if t["object"] else s  # self-loop if no object
        g.add_node(s)
        g.add_node(o)
        g.add_edge(s, o, relation=t["relation"], order=t["i"])
    return g


def graph_stats(g: nx.MultiDiGraph, triples) -> Dict:
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    self_loops = sum(1 for u, v in g.edges() if u == v)
    distinct_relations = len(set(t["relation"] for t in triples))
    out_degrees = [d for _, d in g.out_degree()]
    underlying = nx.DiGraph(g)  # collapse parallel edges for density/components
    density = nx.density(underlying) if n_nodes > 1 else 0.0
    n_wcc = nx.number_weakly_connected_components(underlying)
    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_self_loops": self_loops,
        "n_distinct_relations": distinct_relations,
        "avg_out_degree": sum(out_degrees) / len(out_degrees) if out_degrees else 0.0,
        "max_out_degree": max(out_degrees) if out_degrees else 0,
        "density": density,
        "n_weakly_connected_components": n_wcc,
    }


def main():
    os.makedirs(GRAPHML_DIR, exist_ok=True)

    rows = []
    role_rows = []

    with open(TRIPLES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid, ver, culture = rec["story_id"], rec["version"], rec["culture"]
            triples = rec["triples"]

            g = build_graph(triples)
            stats = graph_stats(g, triples)
            stats.update({"story_id": sid, "version": ver, "culture": culture})
            rows.append(stats)

            roles = assign_roles(triples)
            seq = []
            for t in triples:
                s = normalize_entity(t["subject"])
                o = normalize_entity(t["object"]) if t["object"] else None
                seq.append({
                    "i": t["i"],
                    "relation": t["relation"],
                    "subject_role": roles.get(s, "OTHER"),
                    "object_role": roles.get(o, "OTHER") if o else None,
                })
            role_rows.append({"story_id": sid, "version": ver, "culture": culture, "sequence": seq})

            out_path = os.path.join(GRAPHML_DIR, f"{sid}__{ver}.graphml")
            nx.write_graphml(g, out_path)

    pd.DataFrame(rows).to_csv(STATS_CSV, index=False)
    with open(ROLES_PATH, "w", encoding="utf-8") as f:
        for r in role_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {STATS_CSV}")
    print(f"Wrote {len(role_rows)} rows to {ROLES_PATH}")
    print(f"Wrote graphml files to {GRAPHML_DIR}")


if __name__ == "__main__":
    main()
