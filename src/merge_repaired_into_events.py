import os
import json
import re
from collections import OrderedDict

EVENTS_PATH = r"results\events\events.jsonl"
REPAIRED_DIR = r"results\events\repaired"   # change if your repaired jsons are elsewhere
OUT_PATH = r"results\events\events_merged.jsonl"

# Accept:
#  12__to_Asian.json
#  12__to_Asian__Asian.json
#  22__orig.json
FILENAME_RE = re.compile(r"^(?P<sid>\d+)__(?P<ver>orig|to_[A-Za-z_]+)(?:__(?P<culture>[A-Za-z_]+))?\.json$")

def parse_filename(name: str):
    m = FILENAME_RE.match(name)
    if not m:
        return None
    sid = m.group("sid")
    ver = m.group("ver")
    culture = m.group("culture")
    return sid, ver, culture

def normalize_events(raw_events):
    """
    Accepts:
    - ["...", "..."]
    - [{"i":1,"event":"..."}, ...]
    Also tries to recover if someone forgot the "event" key (your example has that).
    Returns list[str].
    """
    if raw_events is None:
        return []

    # already list of strings
    if isinstance(raw_events, list) and all(isinstance(x, str) for x in raw_events):
        return [x.strip() for x in raw_events if x and str(x).strip()]

    out = []
    if isinstance(raw_events, list):
        for item in raw_events:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
                continue

            if isinstance(item, dict):
                # normal case
                if "event" in item and isinstance(item["event"], str):
                    s = item["event"].strip()
                    if s:
                        out.append(s)
                    continue

                # recovery case: dict has exactly one string field besides "i"
                # e.g. {"i":2, "A crow swooped...": ""} or {"i":2, "text":"..."}
                # We'll grab the first string value, else first string key.
                string_vals = [v for v in item.values() if isinstance(v, str) and v.strip()]
                if string_vals:
                    out.append(string_vals[0].strip())
                    continue

                string_keys = [k for k in item.keys() if isinstance(k, str) and k.strip() and k != "i"]
                if string_keys:
                    out.append(string_keys[0].strip())
                    continue

    return out

def load_events_jsonl(path: str):
    data = {}  # key: (story_id, version) -> obj
    if not os.path.exists(path):
        return data

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = str(obj.get("story_id"))
            ver = str(obj.get("version"))
            data[(sid, ver)] = obj
    return data

def write_events_jsonl(path: str, data: dict):
    # stable-ish sort: numeric story_id then version
    def sid_key(x):
        sid, ver = x
        return (int(sid) if sid.isdigit() else sid, ver)

    items = sorted(data.items(), key=lambda kv: sid_key(kv[0]))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for (_, _), obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    base = load_events_jsonl(EVENTS_PATH)
    merged = 0
    skipped = 0

    if not os.path.isdir(REPAIRED_DIR):
        print(f"[ERROR] Repaired dir not found: {REPAIRED_DIR}")
        return

    for fn in os.listdir(REPAIRED_DIR):
        if not fn.lower().endswith(".json"):
            continue

        parsed = parse_filename(fn)
        if not parsed:
            print(f"[SKIP] Unknown filename format: {fn}")
            skipped += 1
            continue

        sid, ver, culture_from_name = parsed
        full_path = os.path.join(REPAIRED_DIR, fn)

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"[SKIP] Failed to read JSON {fn}: {e}")
            skipped += 1
            continue

        # Expect {"events": [...]}, but be flexible
        raw_events = obj.get("events") if isinstance(obj, dict) else None
        events = normalize_events(raw_events)

        # culture priority: content > filename suffix > None
        culture = None
        if isinstance(obj, dict) and isinstance(obj.get("culture"), str):
            culture = obj["culture"]
        elif culture_from_name:
            culture = culture_from_name

        out_obj = {
            "story_id": sid,
            "version": ver,
            "culture": culture,
            "events": events,
        }

        base[(sid, ver)] = out_obj
        merged += 1

    write_events_jsonl(OUT_PATH, base)

    print(f"Done. Wrote merged file: {OUT_PATH}")
    print(f"Merged repaired entries: {merged}")
    print(f"Skipped repaired files: {skipped}")

if __name__ == "__main__":
    main()
