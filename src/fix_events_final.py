import json
import re
import pandas as pd

IN_PATH = "results/events/events_merged.jsonl"
OUT_PATH = "results/events/events_final.jsonl"
STORIES_CSV = "stories.csv"


def normalize_version(ver: str) -> str:
    ver = ver.strip()
    # "to_X__to_X" -> "to_X"
    ver = re.sub(r"^(to_[A-Za-z_]+)__\1$", r"\1", ver)
    # "orig__European" -> "orig"
    ver = re.sub(r"^(orig)__(European|Asian|African_Diaspora)$", r"\1", ver)
    # "to_X__X" -> "to_X"  (the bug fixed here)
    m = re.match(r"^(to_([A-Za-z_]+))__\2$", ver)
    if m:
        ver = m.group(1)
    return ver


def main():
    df = pd.read_csv(STORIES_CSV)
    orig_culture = {str(r["story_id"]): str(r["culture"]) for _, r in df.iterrows()}

    best = {}  # (story_id, version) -> obj
    with open(IN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = str(obj["story_id"])
            ver = normalize_version(str(obj["version"]))

            # fill in culture
            if ver == "orig":
                culture = orig_culture.get(sid)
            elif ver.startswith("to_"):
                culture = ver[len("to_"):]
            else:
                culture = obj.get("culture")

            obj["story_id"] = sid
            obj["version"] = ver
            obj["culture"] = culture

            key = (sid, ver)
            if key not in best or len(obj["events"]) > len(best[key]["events"]):
                best[key] = obj

    def sid_key(k):
        sid, ver = k
        return (int(sid) if sid.isdigit() else sid, ver)

    items = sorted(best.items(), key=lambda kv: sid_key(kv[0]))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for _, obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(items)} records to {OUT_PATH}")


if __name__ == "__main__":
    main()
