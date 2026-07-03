import json
import os
import re
from typing import Any, Dict, List, Optional

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"

EVENTS_PATH = "results/events/events_final.jsonl"
OUT_DIR = "results/graphs"
OUT_PATH = os.path.join(OUT_DIR, "triples.jsonl")
FAILED_DIR = os.path.join(OUT_DIR, "raw_failed")


def ollama_generate(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["response"]


def find_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text2 = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    # Fix invalid \' escape that LLMs sometimes produce
    text2 = text2.replace("\\'", "'")
    try:
        obj = json.loads(text2)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", text2, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0).strip())
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def make_prompt(events: List[str]) -> str:
    numbered = "\n".join(f"{i+1}. {e}" for i, e in enumerate(events))
    return f"""
You convert a list of folktale plot events into a simple narrative graph.

For EACH numbered event below, output ONE triple:
  - "subject": the entity (character, animal, or object) that DOES the action
  - "relation": a short verb phrase describing the action (2-4 words, lowercase, no names inside it)
  - "object": the entity that the action is done TO (or null if there is none)

CRITICAL RULES:
- Output exactly one triple per event, same order, same count as the input ({len(events)} events).
- Use SHORT, CONSISTENT entity labels (1-3 words, e.g. "the crow", "the king", "Akua").
  The SAME character/object must get the EXACT SAME label every time it appears.
- Do not invent entities that are not in the event text.
- "object" must be null (not a string) if the event has no clear target entity.
- Return JSON ONLY, with this exact schema:

{{
  "triples": [
    {{"i": 1, "subject": "...", "relation": "...", "object": "..." }},
    {{"i": 2, "subject": "...", "relation": "...", "object": null }}
  ]
}}

EVENTS:
{numbered}
""".strip()


def normalize_triples(obj: Dict[str, Any], n_events: int) -> Optional[List[Dict[str, Any]]]:
    if "triples" not in obj or not isinstance(obj["triples"], list):
        return None
    out = []
    for idx, t in enumerate(obj["triples"], start=1):
        if not isinstance(t, dict):
            return None
        subj = t.get("subject")
        rel = t.get("relation")
        objv = t.get("object")
        if not isinstance(subj, str) or not subj.strip():
            subj = "the situation"  # e.g. "there was a drought" has no clear subject
        if not isinstance(rel, str) or not rel.strip():
            return None
        if objv is not None and not isinstance(objv, str):
            objv = str(objv)
        if isinstance(objv, str) and not objv.strip():
            objv = None
        out.append({
            "i": idx,
            "subject": subj.strip(),
            "relation": rel.strip().lower(),
            "object": objv.strip() if isinstance(objv, str) else None,
        })
    if len(out) != n_events:
        return None
    return out


def write_failed(story_id: str, version: str, raw: str) -> None:
    os.makedirs(FAILED_DIR, exist_ok=True)
    fn = f"{story_id}__{version}.txt"
    with open(os.path.join(FAILED_DIR, fn), "w", encoding="utf-8") as f:
        f.write(raw)


def process_record(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    events = rec["events"]
    prompt = make_prompt(events)

    raw = ollama_generate(prompt)
    obj = find_json_object(raw)
    triples = normalize_triples(obj, len(events)) if obj else None

    if triples is None:
        # one retry with a stricter reminder
        retry_prompt = prompt + f"\n\nREMINDER: you MUST output exactly {len(events)} triples, one per event, valid JSON only."
        raw = ollama_generate(retry_prompt)
        obj = find_json_object(raw)
        triples = normalize_triples(obj, len(events)) if obj else None

    if triples is None:
        write_failed(rec["story_id"], rec["version"], raw)
        return None

    return {
        "story_id": rec["story_id"],
        "version": rec["version"],
        "culture": rec["culture"],
        "triples": triples,
    }


def main(limit: Optional[int] = None):
    os.makedirs(OUT_DIR, exist_ok=True)

    records = []
    with open(EVENTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if limit:
        records = records[:limit]

    done = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    o = json.loads(line)
                    done.add((o["story_id"], o["version"]))

    with open(OUT_PATH, "a", encoding="utf-8") as f:
        for rec in records:
            key = (rec["story_id"], rec["version"])
            if key in done:
                continue
            result = process_record(rec)
            if result is None:
                print(f"[FAIL] {key}")
                continue
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[OK] {key} -> {len(result['triples'])} triples")


if __name__ == "__main__":
    import sys
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=lim)
