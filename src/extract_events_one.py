import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

# -----------------------------
# Config
# -----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"

STORIES_CSV = "stories.csv"
RETELLINGS_DIR = "results/retellings"

OUT_REPAIRED_DIR = "results/events/repaired"
OUT_RAW_FAILED_DIR = "results/events/raw_failed"

# -----------------------------
# Helpers
# -----------------------------
def word_count(s: str) -> int:
    return len(str(s).split())

def ollama_generate(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2}  # lower temp = more consistent JSON
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["response"]

def load_story_text(story_id: str, version: str) -> str:
    """
    version:
      - 'orig' -> load from stories.csv (column: story_id, text)
      - 'to_European'/'to_Asian'/'to_African_Diaspora' -> load from results/retellings
    """
    if version == "orig":
        df = pd.read_csv(STORIES_CSV)
        row = df.loc[df["story_id"].astype(str) == str(story_id)]
        if row.empty:
            raise ValueError(f"story_id={story_id} not found in {STORIES_CSV}")
        return str(row.iloc[0]["text"])

    # retelling
    path = os.path.join(RETELLINGS_DIR, f"{story_id}__{version}.txt")
    if not os.path.exists(path):
        # common alternate naming you used earlier:
        alt = os.path.join(RETELLINGS_DIR, f"{story_id}__to_{version.replace('to_', '')}.txt")
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(f"Retelling file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def make_events_prompt(story_text: str) -> str:
    """
    Force strict JSON with a stable schema.
    """
    return f"""
Extract a concise ordered list of the MAIN plot events from the story below.

Rules:
- Output ONLY valid JSON. No markdown, no comments, no extra text.
- JSON must be a single object with exactly one key: "events"
- "events" must be a list of objects like: {{"i": 1, "event": "..."}} starting from i=1
- Each item MUST contain both keys: "i" and "event"
- Keep events high-level (no tiny details). Usually 8-15 events.

STORY:
{story_text}
""".strip()

def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Try direct JSON parse first.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        return None

def extract_first_json_object(text: str) -> Optional[str]:
    """
    If the model returns extra stuff, try to extract the first {...} JSON object.
    """
    s = text.strip()
    # Find first '{' and last '}' and try progressively.
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = s[start:end+1].strip()
    return candidate

def repair_common_json_issues(raw: str) -> str:
    """
    Repairs common Ollama JSON mistakes:
    - Trailing commas
    - Single quotes (rare)
    - Missing "event" key: {"i":2, "A crow ..."} -> {"i":2, "event":"A crow ..."}
    - Multiple JSON objects concatenated -> keep first
    """
    s = raw.strip()

    # If multiple JSON objects are concatenated, keep only the first valid-looking object block
    first_obj = extract_first_json_object(s)
    if first_obj:
        s = first_obj

    # remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # replace fancy quotes just in case
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")

    # convert single-quoted JSON to double quotes (best-effort, not perfect)
    # only do this if it looks like it mainly uses single quotes
    if s.count('"') < 2 and s.count("'") > 5:
        s = s.replace("'", '"')

    # Fix pattern: {"i": 2, "Some text"}  (missing key)
    # We replace: {"i": N, "<something>"} with {"i": N, "event": "<something>"}
    def fix_missing_event_key(match: re.Match) -> str:
        i_part = match.group(1)
        txt_part = match.group(2)
        return f'{{"i": {i_part}, "event": {txt_part}}}'

    s = re.sub(
        r'\{\s*"i"\s*:\s*([0-9]+)\s*,\s*(".*?")\s*\}',
        fix_missing_event_key,
        s,
        flags=re.DOTALL
    )

    return s

def validate_events_schema(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(obj, dict) or "events" not in obj:
        raise ValueError("JSON must be an object with key 'events'.")

    events = obj["events"]
    if not isinstance(events, list) or len(events) == 0:
        raise ValueError("'events' must be a non-empty list.")

    cleaned: List[Dict[str, Any]] = []
    for idx, e in enumerate(events, start=1):
        if not isinstance(e, dict):
            raise ValueError(f"Event #{idx} is not an object.")
        if "i" not in e or "event" not in e:
            raise ValueError(f"Event #{idx} must contain keys 'i' and 'event'. Got: {e}")
        # normalize
        try:
            i_val = int(e["i"])
        except Exception:
            i_val = idx
        ev_text = str(e["event"]).strip()
        if not ev_text:
            raise ValueError(f"Event #{idx} has empty 'event' text.")
        cleaned.append({"i": i_val, "event": ev_text})

    # renumber sequentially (optional but keeps consistency)
    for j, e in enumerate(cleaned, start=1):
        e["i"] = j

    return cleaned

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--story_id", required=True, help="Story id, e.g. 5")
    parser.add_argument("--version", required=True, help="orig, to_European, to_Asian, to_African_Diaspora")
    args = parser.parse_args()

    story_id = str(args.story_id)
    version = str(args.version)

    os.makedirs(OUT_REPAIRED_DIR, exist_ok=True)
    os.makedirs(OUT_RAW_FAILED_DIR, exist_ok=True)

    story_text = load_story_text(story_id, version)

    prompt = make_events_prompt(story_text)
    raw = ollama_generate(prompt).strip()

    # try parse raw
    obj = try_parse_json(raw)

    # if fail, try extract+repair+parse
    if obj is None:
        repaired_str = repair_common_json_issues(raw)
        obj = try_parse_json(repaired_str)

    if obj is None:
        # save raw failed
        fail_path = os.path.join(OUT_RAW_FAILED_DIR, f"{story_id}__{version}.txt")
        with open(fail_path, "w", encoding="utf-8") as f:
            f.write(raw)
        raise ValueError(f"Could not parse JSON. Saved raw output to: {fail_path}")

    # validate schema
    try:
        events_clean = validate_events_schema(obj)
    except Exception as e:
        fail_path = os.path.join(OUT_RAW_FAILED_DIR, f"{story_id}__{version}.txt")
        with open(fail_path, "w", encoding="utf-8") as f:
            f.write(raw)
        raise ValueError(f"Schema invalid: {e}. Saved raw output to: {fail_path}")

    out_obj = {"events": events_clean}

    out_path = os.path.join(OUT_REPAIRED_DIR, f"{story_id}__{version}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print("Saved:", out_path)

if __name__ == "__main__":
    main()
