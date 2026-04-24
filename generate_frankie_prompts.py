import json
import requests
from pathlib import Path
import random
import time
from collections import Counter

# ====================== CONFIG ======================
MODEL_NAME = "qwen3-4b-instruct-2507.gguf"
API_URL = "http://127.0.0.1:1234/v1/chat/completions"

OUTFILE = "frankie_prompt_batch.json"
N_PER_FAMILY = 5
TEMPERATURE = 0.8
MAX_TOKENS = 300
TIMEOUT = 90
MAX_RETRIES = 3

FAMILIES = [
    "social",
    "novel",
    "distress",
    "factual",
    "personal",
    "orienting",
]

SYSTEM_PROMPT = """
You are a strict JSON generator.

You must output ONLY valid JSON.
No markdown.
No code fences.
No explanations.
No headings.
No notes.
No prose.

Return exactly a JSON array of objects.
Each object must contain exactly:
- "family"
- "prompt"

Example:
[
  {"family":"social","prompt":"Morning Frankie, how are you?"},
  {"family":"social","prompt":"Hello Frankie, good to see you."}
]
""".strip()


def build_user_prompt(family: str, n: int) -> str:
    return f"""
Generate exactly {n} short natural user prompts for the family "{family}".

Rules:
- Output ONLY a JSON array.
- Every item must have:
  - "family": "{family}"
  - "prompt": "<text>"
- Keep prompts short and natural.
- Vary the wording.
- Do not include any text before or after the JSON array.
""".strip()


def extract_json_array(text: str) -> str:
    text = text.strip()

    # Remove fenced blocks if present
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("[") and part.endswith("]"):
                return part
            if part.startswith("json"):
                part = part[4:].strip()
                if part.startswith("[") and part.endswith("]"):
                    return part

    # Try direct array extraction
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    return text


def call_qwen_family(family: str) -> list[dict]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(family, N_PER_FAMILY)},
                ],
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
            }

            resp = requests.post(API_URL, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            text = data["choices"][0]["message"]["content"]
            text = extract_json_array(text)

            items = json.loads(text)

            if not isinstance(items, list):
                raise ValueError("Model output was not a JSON list.")

            cleaned = []
            seen = set()

            for item in items:
                if not isinstance(item, dict):
                    continue

                fam = str(item.get("family", "")).strip().lower()
                prompt = str(item.get("prompt", "")).strip()

                if fam != family:
                    continue
                if not prompt:
                    continue

                key = prompt.lower()
                if key in seen:
                    continue
                seen.add(key)

                cleaned.append({"family": fam, "prompt": prompt})

            if len(cleaned) >= N_PER_FAMILY:
                return cleaned[:N_PER_FAMILY]

            print(f"{family} attempt {attempt}: got {len(cleaned)} valid prompts, retrying...")

        except Exception as e:
            print(f"{family} attempt {attempt} failed: {e}")

        if attempt < MAX_RETRIES:
            time.sleep(1.5)

    print(f"{family}: falling back.")
    return [
        {"family": family, "prompt": f"hello frankie {family} {i+1}"}
        for i in range(N_PER_FAMILY)
    ]


def main():
    print("Generating fresh prompt batch for Frankie...")

    prompts = []
    for family in FAMILIES:
        family_prompts = call_qwen_family(family)
        prompts.extend(family_prompts)

    random.shuffle(prompts)

    Path(OUTFILE).write_text(
        json.dumps(prompts, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    counts = Counter(item["family"] for item in prompts)
    print("\nPrompt counts by family:")
    for fam in FAMILIES:
        print(f"  {fam:12s} : {counts.get(fam, 0)}")

    print(f"\nSaved {len(prompts)} mixed prompts to: {OUTFILE}")
    print("Ready to feed into Frankie!\n")


if __name__ == "__main__":
    main()