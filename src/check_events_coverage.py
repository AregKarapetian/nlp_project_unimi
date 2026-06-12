import json
import pandas as pd
from collections import defaultdict

EVENTS_PATH = "results/events/events_final.jsonl"
STORIES_CSV = "stories.csv"

ALL_CULTURES = ["European", "Asian", "African_Diaspora"]

def main():
    # Load original culture per story_id from stories.csv
    df = pd.read_csv(STORIES_CSV)
    id_to_culture = {str(r["story_id"]): str(r["culture"]) for _, r in df.iterrows()}

    # Read which versions we actually have in events_merged.jsonl
    seen = defaultdict(set)
    total_lines = 0

    with open(EVENTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            obj = json.loads(line)
            sid = str(obj.get("story_id"))
            ver = str(obj.get("version"))
            seen[sid].add(ver)

    print("Total lines:", total_lines)
    print("Unique story_id count in events file:", len(seen))
    print("Unique story_id count in stories.csv:", len(id_to_culture))
    print()

    missing_any = False

    # Check each story in stories.csv (source of truth)
    for sid in sorted(id_to_culture.keys(), key=lambda x: int(x) if x.isdigit() else x):
        orig_culture = id_to_culture[sid]

        required = {"orig"} | {f"to_{c}" for c in ALL_CULTURES if c != orig_culture}
        present = seen.get(sid, set())

        missing = required - present
        extra = present - required

        if missing:
            missing_any = True
            print(f"Story {sid} (orig={orig_culture}) missing: {sorted(missing)}")
        if extra:
            print(f"Story {sid} (orig={orig_culture}) extra/unexpected: {sorted(extra)}")

    print()
    if not missing_any:
        print("OK: Coverage looks good (all required versions exist).")

if __name__ == "__main__":
    main()
