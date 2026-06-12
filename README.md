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
  stories don't separate cleanly by culture: clustering puts most of them
  (24/30) into a single group with roughly the same culture mix as the
  whole corpus. This is mild evidence for shared/"universal" narrative
  shapes, though the sample is small.
- Retellings into a **European** style change the narrative graph the
  most: they have the highest graph-edit distance from the original and
  by far the weakest preservation of "who does what to whom" (object-role
  alignment ~0.25, vs. ~0.47-0.67 for the other targets).
- Retellings into an **African_Diaspora** style stay closest to the
  original: highest overlap in the action vocabulary used, and the
  highest object-role alignment (~0.67).
- Across all target cultures, the model is fairly consistent about *who
  the protagonist is* (subject-role alignment stays around 0.44-0.46) -
  the differences show up mainly in secondary relationships and word
  choice, not in who drives the story.

## Pipeline overview

```
Phase 1 - Corpus & retellings
  stories.csv              -> 30 folktales (10 European, 10 Asian, 10 African_Diaspora)
  retell_all.py            -> retell each story into the 2 other cultural styles
  retell_one.py            -> alternate/refined retelling prompt (used for story 1)

Phase 2 - Event extraction
  extract_events_all.py    -> ordered plot-event list per story/retelling
  extract_events_one.py, repair_raw_failed.py,
  merge_repaired_into_events.py, fix_events_final.py
                            -> clean up and merge into events_final.jsonl

Phase 3 - Narrative graph construction
  extract_triples.py       -> (subject, relation, object) triple per event
  build_graphs.py           -> narrative graphs + structural stats + role sequences

Phase 4 - Clustering & cross-cultural comparison
  cluster_structures.py    -> cluster originals by narrative-graph structure
  compare_cultures.py      -> orig vs. retelling: graph edit distance,
                               motif/relation overlap, role alignment
```

## Repository structure

```
.
├── README.md
├── requirements.txt
├── stories.csv
├── src/
│   ├── retell_all.py
│   ├── retell_one.py
│   ├── extract_events_all.py
│   ├── extract_events_one.py
│   ├── repair_raw_failed.py
│   ├── merge_repaired_into_events.py
│   ├── fix_events_final.py
│   ├── check_events_coverage.py
│   ├── extract_triples.py
│   ├── build_graphs.py
│   ├── cluster_structures.py
│   └── compare_cultures.py
└── results/
    ├── retellings/                60 LLM retellings (.txt)
    ├── events/
    │   └── events_final.jsonl     plot events for all 90 story-versions
    ├── graphs/
    │   ├── triples.jsonl          (subject, relation, object) triples per event
    │   ├── graph_stats.csv        structural features per narrative graph
    │   ├── role_sequences.jsonl   per-event role sequence per story-version
    │   └── graphs/*.graphml       narrative graphs (open in Gephi/Cytoscape/networkx)
    └── analysis/
        ├── clusters.csv, *_vs_culture.csv
        ├── dendrogram.png
        ├── comparison_details.csv, comparison_summary.csv
        └── comparison_by_culture.png
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
python src/retell_all.py            # results/retellings/
python src/extract_events_all.py    # results/events/events.jsonl (+ raw_failed/ on errors)

# only needed if extract_events_all.py left anything in results/events/raw_failed/:
python src/repair_raw_failed.py
python src/merge_repaired_into_events.py

python src/fix_events_final.py      # -> results/events/events_final.jsonl
python src/check_events_coverage.py # sanity check (90 = 30 stories x 3 versions)

python src/extract_triples.py       # -> results/graphs/triples.jsonl
python src/build_graphs.py          # -> results/graphs/graph_stats.csv, role_sequences.jsonl, graphs/*.graphml
python src/cluster_structures.py    # -> results/analysis/clusters.csv, dendrogram.png
python src/compare_cultures.py      # -> results/analysis/comparison_*.csv, comparison_by_culture.png
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

## Limitations

- The corpus is English-only. Working in a single language with a single
  local model kept the event/triple extraction and the comparison metrics
  tractable, at the cost of not testing any cross-lingual effects.
- Story 1's retellings were generated with a slightly different, more
  refined prompt (`retell_one.py`) than stories 2-30
  (`retell_all.py`), so the 60 retellings aren't fully uniform.
- Event lists, entity labels, and relations all come from single-pass
  Llama 3.1 8B outputs with no human verification, so some noise is
  expected (a few stories produced sparse graphs with very few distinct
  entities).
- 30 stories / 60 orig-vs-retelling comparisons is a small sample - the
  differences reported above are descriptive trends, not statistically
  tested.

## AI usage disclaimer

Parts of this project (cleaning up the event-extraction pipeline, building
the narrative-graph construction, clustering, and cross-cultural comparison
scripts) were developed with the assistance of
an AI coding assistant (Claude, Anthropic). All AI-assisted code and text
were reviewed and adjusted by the author, who takes full responsibility for
the final content.
