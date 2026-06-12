import os
import json
import re
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import requests

# ----------------------------
# CONFIG
# ----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"

ALL_CULTURES = ["European", "Asian", "African_Diaspora"]

# Input
STORIES_CSV = "stories.csv"  # you run script from project root (C:\Users\aregk\data)
RETELLINGS_DIR = os.path.join("results", "retellings")

# Output
EVENTS_DIR = os.path.join("results", "events")
EVENTS_JSONL = os.path.join(EVENTS_DIR, "events.jsonl")
FAILED_DIR = os.path.join(EVENTS_DIR, "raw_failed")


# ----------------------------
# HELPERS
# ----------------------------
def word_count(s: str) -> int:
    return len(str(s).split())


def ollama_generate(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,   # keep stable for structured output
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["response"].strip()


def safe_filename(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
    return s[:180]


def find_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Tries to find a JSON object in text.
    Accepts:
      - pure JSON output
      - JSON wrapped in text
      - JSON inside ``` fences
    """
    if not text:
        return None

    # Remove code fences if present
    text2 = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()

    # Fast path: try parse as-is
    try:
        obj = json.loads(text2)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try to extract first {...} block (greedy)
    # This is a basic approach but works well for model outputs.
    m = re.search(r"\{.*\}", text2, flags=re.DOTALL)
    if not m:
        return None

    candidate = m.group(0).strip()
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None


def normalize_events(obj: Dict[str, Any]) -> Optional[List[str]]:
    """
    We expect something like:
    { "events": [ {"i": 1, "event": "..."}, ... ] }
    or:
    { "events": ["...", "..."] }
    """
    if "events" not in obj:
        return None

    ev = obj["events"]
    if isinstance(ev, list) and all(isinstance(x, str) for x in ev):
        # already list of strings
        return [x.strip() for x in ev if x.strip()]

    if isinstance(ev, list) and all(isinstance(x, dict) for x in ev):
        out = []
        for x in ev:
            e = x.get("event")
            if isinstance(e, str) and e.strip():
                out.append(e.strip())
        return out if out else None

    return None


def make_event_prompt(story_text: str) -> str:
    """
    IMPORTANT:
    - Output must be JSON only
    - Ask for Standard English (no dialect spelling)
    - Keep it concise and factual
    """
    wc = word_count(story_text)
    # Event count guidance based on length
    # 8–16 is usually fine. We'll suggest it.
    return f"""
You are extracting plot events from a folktale.

Rules:
- Write in Standard English only.
- Do NOT add new events not supported by the story.
- Keep events chronological.
- Keep each event as a single short sentence.
- Return JSON ONLY, with this exact schema:

{{
  "events": [
    {{"i": 1, "event": "..." }},
    {{"i": 2, "event": "..." }}
  ]
}}

Story length is about {wc} words, so return 10–14 events if possible.

STORY:
{story_text}
""".strip()


def read_retelling_text(story_id: str, target: str) -> Optional[str]:
    path = os.path.join(RETELLINGS_DIR, f"{story_id}__to_{target}.txt")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip() or None


def write_failed(story_id: str, version: str, culture: str, raw: str) -> None:
    os.makedirs(FAILED_DIR, exist_ok=True)
    fn = safe_filename(f"{story_id}__{version}__{culture}.txt")
    with open(os.path.join(FAILED_DIR, fn), "w", encoding="utf-8") as f:
        f.write(raw)


def append_jsonl(record: Dict[str, Any]) -> None:
    os.makedirs(EVENTS_DIR, exist_ok=True)
    with open(EVENTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ----------------------------
# MAIN LOGIC
# ----------------------------
def extract_events_for_text(
    story_id: str,
    version: str,
    culture: str,
    story_text: str,
) -> bool:
    prompt = make_event_prompt(story_text)
    raw = ollama_generate(prompt)

    obj = find_json_object(raw)
    if obj is None:
        print(f"[ERROR] Failed events: {story_id} {version} (no JSON found)")
        write_failed(story_id, version, culture, raw)
        return False

    events = normalize_events(obj)
    if events is None:
        print(f"[ERROR] Failed events: {story_id} {version} (bad schema)")
        write_failed(story_id, version, culture, raw)
        return False

    record = {
        "story_id": str(story_id),
        "version": version,     # "orig" or "to_European" etc
        "culture": culture,     # culture label for this version
        "events": events,
    }

    append_jsonl(record)
    print("Saved:", story_id, version)
    return True


def main():
    # Ensure folders exist
    os.makedirs(EVENTS_DIR, exist_ok=True)

    # Reset output each run (so you don't duplicate lines)
    if os.path.exists(EVENTS_JSONL):
        os.remove(EVENTS_JSONL)

    # Load stories.csv
    if not os.path.exists(STORIES_CSV):
        raise FileNotFoundError(
            f"Could not find {STORIES_CSV} in current folder. "
            "Run this script from your project root (C:\\Users\\aregk\\data)."
        )

    df = pd.read_csv(STORIES_CSV)

    required_cols = {"story_id", "culture", "text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"stories.csv missing columns: {missing}")

    # 1) Extract events for ORIGINAL stories
    for _, row in df.iterrows():
        story_id = str(row["story_id"])
        culture = str(row["culture"])
        text = str(row["text"])

        ok = extract_events_for_text(
            story_id=story_id,
            version="orig",
            culture=culture,
            story_text=text,
        )
        if not ok:
            # keep going, don't crash the whole run
            continue

    # 2) Extract events for RETELLINGS we have on disk
    # For each original story, we expect 2 target cultures (not equal to original)
    for _, row in df.iterrows():
        story_id = str(row["story_id"])
        orig_culture = str(row["culture"])

        targets = [c for c in ALL_CULTURES if c != orig_culture]
        for target in targets:
            retelling = read_retelling_text(story_id, target)
            if not retelling:
                # retelling missing file
                print(f"[WARN] Missing retelling file for story_id={story_id}, target={target}")
                continue

            version = f"to_{target}"
            ok = extract_events_for_text(
                story_id=story_id,
                version=version,
                culture=target,
                story_text=retelling,
            )
            if not ok:
                continue

    print("Done! Output:", EVENTS_JSONL)


if __name__ == "__main__":
    main()
