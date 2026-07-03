"""
Retries event extraction for any records missing from events_final.jsonl.
First tries to recover from existing raw_failed files using the regex fallback,
then re-calls Ollama for anything still missing.
Run from project root after extract_events_all.py.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from extract_events_all import (
    EVENTS_JSONL, FAILED_DIR, ALL_CULTURES,
    extract_events_for_text, extract_events_fallback,
    read_retelling_text,
)

STORIES_CSV = "stories.csv"


def load_done():
    done = set()
    if os.path.exists(EVENTS_JSONL):
        with open(EVENTS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done.add((str(r["story_id"]), r["version"]))
    return done


def try_recover_from_raw(story_id, version, culture):
    """Try to salvage events from the existing raw_failed file."""
    import re
    fname = f"{story_id}__{version}__{culture}.txt"
    path = os.path.join(FAILED_DIR, fname)
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    events = extract_events_fallback(raw)
    if not events:
        return False
    record = {"story_id": story_id, "version": version, "culture": culture, "events": events}
    with open(EVENTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Recovered from raw_failed: {story_id} {version} ({len(events)} events)")
    return True


def main():
    df = pd.read_csv(STORIES_CSV)
    done = load_done()
    print(f"Already done: {len(done)} records")

    for _, row in df.iterrows():
        story_id = str(row["story_id"])
        culture = str(row["culture"])
        text = str(row["text"])

        if (story_id, "orig") not in done:
            if not try_recover_from_raw(story_id, "orig", culture):
                print(f"Re-extracting: {story_id} orig")
                extract_events_for_text(story_id, "orig", culture, text)

        for target in [c for c in ALL_CULTURES if c != culture]:
            version = f"to_{target}"
            if (story_id, version) not in done:
                if not try_recover_from_raw(story_id, version, target):
                    retelling = read_retelling_text(story_id, target)
                    if retelling:
                        print(f"Re-extracting: {story_id} {version}")
                        extract_events_for_text(story_id, version, target, retelling)

    total = sum(1 for _ in open(EVENTS_JSONL))
    print(f"Done. events_final.jsonl now has {total} records.")


if __name__ == "__main__":
    main()
