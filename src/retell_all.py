# src/retell_all.py
import os
import pandas as pd
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"
ALL_CULTURES = ["European", "Asian", "African_Diaspora"]

def word_count(s: str) -> int:
    return len(str(s).split())

def clean_commentary(text: str) -> str:
    """
    Best-effort cleanup: remove common meta-comment lines if they appear.
    (Your prompt should already prevent most of these.)
    """
    bad_prefixes = ("this is", "i will", "i'll", "here is", "retelling", "sure")
    cleaned_lines = []
    for line in str(text).splitlines():
        ls = line.strip().lower()
        if any(ls.startswith(p) for p in bad_prefixes):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()

def ollama_generate(prompt: str, temperature: float = 0.4) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["response"].strip()

def make_prompt(text: str, target: str, wc: int) -> str:
    lo = int(wc * 0.85)
    hi = int(wc * 1.15)

    # Force "African_Diaspora" to be about narrative style, not dialect imitation.
    extra = ""
    if target == "African_Diaspora":
        extra = (
            "Use STANDARD English only. Do NOT imitate dialects, accents, or phonetic spellings.\n"
            "Reflect cultural style through themes, imagery, proverbs, community framing, and oral storytelling cadence.\n"
        )

    return f"""
Retell the following folktale in the style of {target} folktales.
{extra}
Keep the SAME main events and the SAME ending.
Do NOT add new major plot points.
Do not change who wins each challenge (coat, cap, wall picture): Chonguita must be the best in all three.
Keep the length similar to the original: target between {lo} and {hi} words.
Output ONLY the retold story text. No title. No explanations.

STORY:
{text}
""".strip()

def main():
    # Your folder layout: DATA/stories.csv, DATA/src/, DATA/results/
    df = pd.read_csv("stories.csv")

    os.makedirs("results/retellings", exist_ok=True)

    for _, row in df.iterrows():
        story_id = str(row["story_id"])
        original_culture = str(row["culture"])
        text = str(row["text"])

        wc = word_count(text)
        targets = [c for c in ALL_CULTURES if c != original_culture]

        for target in targets:
            out_path = f"results/retellings/{story_id}__to_{target}.txt"

            # Skip if already exists (lets you resume)
            if os.path.exists(out_path):
                print("Skipping (exists):", out_path)
                continue

            prompt = make_prompt(text, target, wc)

            # First attempt (more rule-following)
            raw = ollama_generate(prompt, temperature=0.35)
            retelling = clean_commentary(raw)

            # If too short, retry once (slightly more creative)
            if len(retelling.split()) < 200:
                print(f"[WARN] Too short, retrying once: story_id={story_id}, target={target}")
                raw2 = ollama_generate(prompt, temperature=0.55)
                retelling = clean_commentary(raw2)

            # Final safety: if still too short, do one rescue prompt (no tags, just full story)
            if len(retelling.split()) < 200:
                print(f"[WARN] Still short, rescue attempt: story_id={story_id}, target={target}")
                rescue_prompt = f"""
Retell the story fully in the style of {target} folktales.
Use STANDARD English only. Do NOT imitate dialects, accents, or phonetic spellings.
Keep the SAME main events and the SAME ending.
Do NOT add new major plot points.
Do not change who wins each challenge (coat, cap, wall picture): Chonguita must be the best in all three.
Output ONLY the full story text.

STORY:
{text}
""".strip()
                retelling = clean_commentary(ollama_generate(rescue_prompt, temperature=0.6))

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(retelling)

            print("Saved:", out_path)

    print("All done! Check results/retellings for output files.")

if __name__ == "__main__":
    main()
