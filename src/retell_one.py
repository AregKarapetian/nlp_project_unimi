# src/retell_two_for_one.py
import os
import pandas as pd
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"
ALL_CULTURES = ["European", "Asian", "African_Diaspora"]

def word_count(s: str) -> int:
    return len(str(s).split())

def clean_commentary(text: str) -> str:
    bad_prefixes = ("this is", "i will", "i'll", "here is", "retelling", "sure")
    cleaned_lines = []
    for line in str(text).splitlines():
        ls = line.strip().lower()
        if any(ls.startswith(p) for p in bad_prefixes):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()

def ollama_generate(prompt: str, temperature: float = 0.35) -> str:
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

    style = ""
    if target == "African_Diaspora":
        style = (
            "Use STANDARD English only. Do NOT imitate dialects, accents, or phonetic spellings.\n"
            "Reflect African Diaspora folktale style through: oral storytelling cadence, vivid imagery, "
            "community framing, and a short proverb-like closing line.\n"
        )
    elif target == "Asian":
        style = (
            "Use STANDARD English only.\n"
            "Reflect Asian folktale style through: calm/simple narration, nature imagery, themes of harmony, "
            "humility, duty, and consequence, and a brief proverb-like closing line.\n"
        )
    elif target == "European":
        style = (
            "Use STANDARD English only.\n"
            "Reflect European folktale style through: fairy-tale tone, court/kingdom atmosphere (if relevant), "
            "and a clear moral at the end (one sentence).\n"
        )

    return f"""
Retell the following folktale in the style of {target} folktales.

STYLE RULES:
{style}
- Include 2–3 small culture-style details (setting, imagery, values), but do NOT add new major plot events.
- Keep ALL original character names exactly as in the story (do NOT rename anyone).
- Keep the SAME main events, the SAME winners of each challenge (if any), and the SAME ending.
- Do NOT change who wins any challenge: the same person must win as in the original.
- Keep the length similar to the original: target between {lo} and {hi} words.
- Output ONLY the retold story text. No title. No explanations.
- Do NOT introduce new proper nouns (new names of people/places). If a character/place is unnamed in the original, keep it unnamed.

STORY:
{text}
""".strip()

def main():
    df = pd.read_csv("stories.csv")

    # --- Choose ONE story to test ---
    # Option A: first row
    row = df.iloc[0]

    # Option B: choose by ID (recommended)
    # TEST_ID = 1          # if your story_id is numeric
    # TEST_ID = "E01"      # if your story_id is like E01
    # row = df.loc[df["story_id"] == TEST_ID].iloc[0]

    story_id = str(row["story_id"])
    original_culture = str(row["culture"])
    text = str(row["text"])

    wc = word_count(text)
    targets = [c for c in ALL_CULTURES if c != original_culture]

    os.makedirs("results/retellings", exist_ok=True)

    for target in targets:
        out_path = f"results/retellings/{story_id}__to_{target}.txt"

        prompt = make_prompt(text, target, wc)
        raw = ollama_generate(prompt, temperature=0.35)
        retelling = clean_commentary(raw)

        # If too short, retry once
        if len(retelling.split()) < 200:
            print(f"[WARN] Too short, retrying once: story_id={story_id}, target={target}")
            raw2 = ollama_generate(prompt, temperature=0.55)
            retelling = clean_commentary(raw2)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(retelling)

        print("Saved:", out_path)

    print("Done. Original:", original_culture, "Targets:", targets)

if __name__ == "__main__":
    main()
