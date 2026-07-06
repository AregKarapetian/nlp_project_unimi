# Stories We Tell (and the Machines Retell)

**How Large Language Models represent and reproduce narrative structure across cultures**

NLP final project (P12).

This project looks at whether a Large Language Model (Llama 3.1 8B, run
locally via Ollama) preserves the *structure* of a folktale when it's
asked to retell the same story "in the style of" a different cultural
tradition. Starting from a small multicultural corpus of folktales, each
story is retold by the model in two other cultural styles, and both the
originals and the retellings are converted into narrative graphs
(characters/objects as nodes, plot events as edges) so that their
structures can be compared directly.

## Key findings

- At the level of simple narrative-graph statistics, the 30 original
  stories do not separate cleanly by culture: both K-Means and
  hierarchical clustering produce one dominant mixed-culture group.
  This is mild evidence for shared "universal" narrative shapes,
  though the sample is small.
- Retellings into a **European** style change the narrative graph the
  most: they have the highest graph-edit distance from the original
  (norm. GED ≈ 0.32) and the weakest object-role alignment (≈ 0.51).
- Retellings into an **African_Diaspora** style stay closest to the
  original: lowest graph-edit distance (≈ 0.23) and the highest
  object-role alignment (≈ 0.71).
- Across all target cultures, the model is fairly consistent about *who
  the protagonist is* (subject-role alignment stays around 0.46–0.50) —
  the differences show up mainly in secondary relationships and word
  choice, not in who drives the story.

## Pipeline overview

```
Phase 1 - Corpus & retellings
  stories.csv              -> 30 folktales (10 European, 10 Asian, 10 African_Diaspora)
  retell_all.py            -> retell each story into the 2 other cultural styles

Phase 2 - Event extraction
  extract_events_all.py    -> ordered plot-event list per story/retelling
                              -> results/events/events_final.jsonl (90 records)
  retry_failed_events.py   -> recover any records that failed on first pass

Phase 3 - Narrative graph construction
  extract_triples.py       -> (subject, relation, object) triple per event
  build_graphs.py          -> narrative graphs + structural stats + role sequences

Phase 4 - Clustering & cross-cultural comparison
  cluster_structures.py    -> cluster originals by narrative-graph structure
  compare_cultures.py      -> orig vs. retelling: GED, motif overlap, role alignment
  visualize_results.py     -> PCA plot + example graph figures
  semantic_similarity.py   -> BERT cosine similarity between original and retelling
```

## Repository structure

```
.
├── README.md
├── requirements.txt
├── stories.csv
├── src/
│   ├── retell_all.py            phase 1: generate retellings
│   ├── extract_events_all.py    phase 2: extract plot events
│   ├── retry_failed_events.py   phase 2: recover any extraction failures
│   ├── extract_triples.py       phase 3: extract (subject, relation, object) triples
│   ├── build_graphs.py          phase 3: build narrative graphs
│   ├── cluster_structures.py    phase 4: cluster original stories
│   ├── compare_cultures.py      phase 4: cross-cultural comparison metrics
│   ├── visualize_results.py     phase 4: PCA + example graph figures
│   └── semantic_similarity.py   phase 4: BERT semantic similarity
└── results/
    ├── retellings/              60 LLM retellings (.txt)
    ├── events/
    │   └── events_final.jsonl   plot events for all 90 story-versions
    ├── graphs/
    │   ├── triples.jsonl        (subject, relation, object) triples per event
    │   ├── graph_stats.csv      structural features per narrative graph
    │   ├── role_sequences.jsonl per-event role sequence per story-version
    │   └── graphs/*.graphml     narrative graphs (open in Gephi/Cytoscape/networkx)
    └── analysis/
        ├── clusters.csv, *_vs_culture.csv
        ├── dendrogram.png, pca_stories.png, example_graphs.png
        ├── comparison_details.csv, comparison_summary.csv
        ├── comparison_by_culture.png
        ├── semantic_similarity.csv
        └── semantic_similarity.png
```

## Setup

1. Python 3.12, then:
   ```
   pip install -r requirements.txt
   ```
2. [Ollama](https://ollama.com/) installed and running locally, with the
   model pulled:
   ```
   ollama pull llama3.1:8b
   ollama serve
   ```

## Reproducing the pipeline

```bash
python src/retell_all.py          # -> results/retellings/ (60 files)
python src/extract_events_all.py  # -> results/events/events_final.jsonl

# only needed if extract_events_all.py left anything in results/events/raw_failed/:
python src/retry_failed_events.py

python src/extract_triples.py     # -> results/graphs/triples.jsonl
python src/build_graphs.py        # -> results/graphs/graph_stats.csv, role_sequences.jsonl, graphs/
python src/cluster_structures.py      # -> results/analysis/clusters.csv, dendrogram.png
python src/compare_cultures.py        # -> results/analysis/comparison_*.csv, comparison_by_culture.png
python src/visualize_results.py       # -> results/analysis/pca_stories.png, example_graphs.png
python src/semantic_similarity.py     # -> results/analysis/semantic_similarity.csv, semantic_similarity.png
```

All intermediate outputs are already included under `results/`, so the
analysis steps (`extract_triples.py` onward) can be re-run without
repeating the LLM generation steps.

## Methodology notes / metrics

- **Narrative graph**: nodes are entities (characters/objects), edges are
  plot events labeled with a short relation phrase and their order; an
  event with no clear object becomes a self-loop.
- **Roles**: within each story-version, the entity that is the subject of
  the most events is labeled `PROTAGONIST`, the runner-up
  `DEUTERAGONIST`, everything else `OTHER`.
- **Normalized graph edit distance**: edit distance between the
  (unlabeled) original and retelling graphs, divided by
  `max(|V|+|E|)` so it's comparable across stories of different lengths.
- **Relation/motif overlap**: Jaccard similarity between the sets of
  action-words used in the original's and the retelling's relations.
- **Role alignment**: fraction of event positions where the
  subject's (resp. object's) role matches between the original and the
  retelling.
- **Semantic similarity**: cosine similarity between BERT sentence embeddings
  (all-MiniLM-L6-v2 via `sentence-transformers`) of the full original and
  retelling texts. Measures meaning-level preservation independently of the
  graph structure.

## Limitations

- The corpus is English-only. Working in a single language with a single
  local model kept the event/triple extraction and the comparison metrics
  tractable, at the cost of not testing any cross-lingual effects.
- All 60 retellings use the same prompt (`retell_all.py`), but the
  prompt gives the model latitude in how it interprets each cultural
  style, so the retelling quality is uneven across stories.
- Event lists, entity labels, and relations all come from single-pass
  Llama 3.1 8B outputs with no human verification, so some noise is
  expected (a few stories produced sparse graphs with very few distinct
  entities).
- 30 stories / 60 orig-vs-retelling comparisons is a small sample - the
  differences reported above are descriptive trends, not statistically
  tested.


