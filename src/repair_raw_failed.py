import json
import re
from pathlib import Path

RAW_DIR = Path("results/events/raw_failed")
OUT_DIR = Path("results/events/repaired")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fix pattern: {"i": 2, "Some sentence." }
BROKEN_EVENT_RE = re.compile(
    r'(\{\s*"i"\s*:\s*\d+\s*,)\s*"([^"]+)"\s*(\})'
)

def repair_text(s: str) -> str:
    # 1) Fix the broken event objects missing "event":
    s = BROKEN_EVENT_RE.sub(r'\1 "event": "\2"\3', s)

    # 2) Remove trailing commas before ] or }
    s = re.sub(r",\s*([}\]])", r"\1", s)

    return s

def parse_multiple_json_objects(s: str):
    """
    Some raw_failed files may contain multiple JSON objects concatenated.
    This yields each parsed object in order.
    """
    dec = json.JSONDecoder()
    idx = 0
    n = len(s)
    objs = []

    while True:
        # skip whitespace
        while idx < n and s[idx].isspace():
            idx += 1
        if idx >= n:
            break

        obj, end = dec.raw_decode(s, idx)
        objs.append(obj)
        idx = end

    return objs

def normalize_events_obj(obj: dict) -> dict:
    """
    Ensures schema:
      {"events": [{"i": int, "event": str}, ...]}
    """
    if not isinstance(obj, dict) or "events" not in obj or not isinstance(obj["events"], list):
        raise ValueError("Missing or invalid 'events' list")

    cleaned = []
    for item in obj["events"]:
        if not isinstance(item, dict) or "i" not in item:
            continue

        i_val = item["i"]
        event_val = item.get("event", None)

        # If somehow event is still missing but there is exactly one other key, use it
        if event_val is None:
            other_keys = [k for k in item.keys() if k != "i"]
            if len(other_keys) == 1:
                event_val = other_keys[0]

        if event_val is None:
            continue

        cleaned.append({"i": int(i_val), "event": str(event_val).strip()})

    return {"events": cleaned}

def main():
    if not RAW_DIR.exists():
        print(f"[ERROR] Folder not found: {RAW_DIR}")
        return

    files = list(RAW_DIR.glob("*.txt"))
    if not files:
        print(f"[INFO] No .txt files found in {RAW_DIR}")
        return

    repaired = 0
    still_failed = 0

    for fp in files:
        raw = fp.read_text(encoding="utf-8", errors="replace").strip()
        fixed = repair_text(raw)

        try:
            # Parse possibly multiple JSON objects in the same file
            objs = parse_multiple_json_objects(fixed)

            for k, obj in enumerate(objs, start=1):
                norm = normalize_events_obj(obj)

                # If multiple objects, suffix filename
                suffix = f"__part{k}" if len(objs) > 1 else ""
                out_path = OUT_DIR / (fp.stem + suffix + ".json")

                out_path.write_text(
                    json.dumps(norm, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )

                print("[OK] Repaired:", fp.name, "->", out_path.as_posix())
                repaired += 1

        except Exception as e:
            print("[FAIL]", fp.name, "|", str(e))
            still_failed += 1

    print(f"\nDone. Saved repaired JSON files: {repaired}. Still failed: {still_failed}")

if __name__ == "__main__":
    main()
