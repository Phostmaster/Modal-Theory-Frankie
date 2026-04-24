import json
import time
from datetime import datetime
from pathlib import Path

import requests


# =========================
# MODEL CONFIG
# =========================
TEXT_MODEL_DEFAULT = "qwen3-4b-instruct-2507.gguf"
TEXT_BASE_URL_DEFAULT = "http://127.0.0.1:1234/v1/chat/completions"


# =========================
# MODE PROMPTS
# =========================
HOME_MODE_PROMPT = """
You are a calm, steady, and concise assistant. Respond directly and clearly.
Keep answers short and to the point. Avoid unnecessary analysis, elaboration,
or extra steps unless asked. Stay grounded, practical, and helpful.
""".strip()

ANALYTIC_MODE_PROMPT = """
You are a precise, analytical assistant. Reason step by step when needed.
Compare options when relevant. Test assumptions. Distinguish between correlation
and causation. Be rigorous, clear, and well-structured in your explanations.
Show your reasoning only to the extent that it improves clarity.
""".strip()

ENGAGEMENT_MODE_PROMPT = """
You are a warm, supportive, and present assistant. Respond with steadiness,
care, and clarity. Stay attuned to the user's tone and needs. Offer companionship
in the conversation without becoming vague or over-reassuring. Be gentle but direct,
and keep the interaction human, grounded, and clear.
""".strip()

MODE_PROMPTS = {
    "home": HOME_MODE_PROMPT,
    "analytic": ANALYTIC_MODE_PROMPT,
    "engagement": ENGAGEMENT_MODE_PROMPT,
}


# =========================
# TEST PROMPTS
# =========================
TEST_PROMPTS = [
    # Home / Base
    "What is the capital of France?",
    "Define photosynthesis in one sentence.",
    "Who wrote Hamlet?",
    "List three causes of rain.",

    # Analytic / Reasoning
    "What evidence would distinguish correlation from causation here?",
    "Compare the strengths and weaknesses of these two plans.",
    "Why might a stable pattern still be misleading?",
    "Help me analyze the assumptions behind this argument.",

    # Relational / Engagement
    "I feel overwhelmed and need help thinking this through calmly.",
    "Can you stay with me while I work out what to do next?",
    "Please explain this gently and clearly.",
    "I need a supportive but honest answer.",

    # Mixed / ambiguous
    "Help me calmly compare two job options.",
    "I'm upset, but I also need a concrete plan.",
    "Explain this clearly without overcomplicating it.",
    "Can you help me think through this in a grounded way?",
]


# =========================
# GATE LOGIC
# =========================
ANALYTIC_KEYWORDS = [
    "compare",
    "difference",
    "differences",
    "why",
    "how",
    "evidence",
    "assumption",
    "assumptions",
    "analyze",
    "analysis",
    "test",
    "debug",
    "plan",
    "explain",
    "causation",
    "correlation",
    "model",
    "mechanism",
    "evaluate",
    "reason",
    "reasoning",
    "distinguish",
    "strengths",
    "weaknesses",
    "pros",
    "cons",
]

ENGAGEMENT_KEYWORDS = [
    "help me",
    "i feel",
    "i'm feeling",
    "im feeling",
    "overwhelmed",
    "worried",
    "stay with me",
    "gently",
    "support",
    "supportive",
    "calm",
    "talk through",
    "with me",
    "kind",
    "human",
    "present",
    "honest answer",
    "grounded",
    "gentle",
    "steadiness",
    "care",
    "companionship",
    "upset",
]

HOME_KEYWORDS = [
    "what is",
    "who is",
    "when is",
    "where is",
    "capital",
    "define",
    "name",
    "list",
    "give me",
    "tell me",
    "who wrote",
]


def score_modes(prompt: str) -> dict:
    """
    Very simple rule-based gate.
    This is intentionally lightweight for v1.
    """
    text = prompt.lower()

    scores = {
        "home": 0.0,
        "analytic": 0.0,
        "engagement": 0.0,
    }

    for kw in ANALYTIC_KEYWORDS:
        if kw in text:
            scores["analytic"] += 1.0

    for kw in ENGAGEMENT_KEYWORDS:
        if kw in text:
            scores["engagement"] += 1.0

    for kw in HOME_KEYWORDS:
        if kw in text:
            scores["home"] += 0.5

    # Default prior toward Home / Base Mode
    scores["home"] += 1.0

    return scores


def choose_mode(prompt: str) -> tuple[str, dict]:
    """
    Returns the winning mode and raw scores.
    """
    scores = score_modes(prompt)
    mode = max(scores, key=scores.get)
    return mode, scores


# =========================
# GENERATION SETTINGS
# =========================
def get_mode_generation_settings(mode: str) -> dict:
    """
    Light generation-policy differences by mode.
    Keep these modest for v1.
    """
    if mode == "home":
        return {
            "temperature": 0.5,
            "max_tokens": 160,
        }

    if mode == "analytic":
        return {
            "temperature": 0.4,
            "max_tokens": 320,
        }

    if mode == "engagement":
        return {
            "temperature": 0.7,
            "max_tokens": 220,
        }

    return {
        "temperature": 0.6,
        "max_tokens": 220,
    }


# =========================
# MODEL CALL
# =========================
def call_base_model(
    system_prompt: str,
    user_prompt: str,
    model: str = TEXT_MODEL_DEFAULT,
    base_url: str = TEXT_BASE_URL_DEFAULT,
    temperature: float = 0.7,
    max_tokens: int = 300,
) -> str:
    """
    Calls your local Qwen model via the OpenAI-compatible endpoint.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        response = requests.post(
            base_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        return f"[ERROR: request failed] {e}"
    except (KeyError, IndexError, ValueError) as e:
        return f"[ERROR: unexpected response format] {e}"
    except Exception as e:
        return f"[ERROR: unexpected error] {e}"


# =========================
# LOGGING
# =========================
def log_result(result: dict, out_dir: str = "results") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = Path(out_dir) / f"{timestamp}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return str(path)


# =========================
# RUNNERS
# =========================
def run_one(prompt: str) -> dict:
    t0 = time.time()

    mode, scores = choose_mode(prompt)
    system_prompt = MODE_PROMPTS[mode]
    gen_settings = get_mode_generation_settings(mode)

    response = call_base_model(
        system_prompt,
        prompt,
        temperature=gen_settings["temperature"],
        max_tokens=gen_settings["max_tokens"],
    )

    elapsed = time.time() - t0

    result = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "mode": mode,
        "scores": scores,
        "system_prompt": system_prompt,
        "generation_settings": gen_settings,
        "response": response,
        "latency_seconds": round(elapsed, 6),
        "response_chars": len(response),
        "prompt_chars": len(prompt),
    }
    return result


def run_one_forced_mode(prompt: str, forced_mode: str) -> dict:
    t0 = time.time()

    system_prompt = MODE_PROMPTS[forced_mode]
    gen_settings = get_mode_generation_settings(forced_mode)

    response = call_base_model(
        system_prompt,
        prompt,
        temperature=gen_settings["temperature"],
        max_tokens=gen_settings["max_tokens"],
    )

    elapsed = time.time() - t0

    result = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "mode": forced_mode,
        "scores": {"forced": True},
        "system_prompt": system_prompt,
        "generation_settings": gen_settings,
        "response": response,
        "latency_seconds": round(elapsed, 6),
        "response_chars": len(response),
        "prompt_chars": len(prompt),
    }
    return result


def run_batch(prompts: list[str]) -> None:
    print("=" * 72)
    print("MODAL WRAPPER V1 — BATCH RUN")
    print("=" * 72)

    for i, prompt in enumerate(prompts, start=1):
        result = run_one(prompt)
        save_path = log_result(result)

        print(f"\n[{i:02d}] Prompt: {prompt}")
        print(f"     Mode:   {result['mode']}")
        print(f"     Scores: {result['scores']}")
        print(f"     Saved:  {save_path}")
        print(f"     Output: {result['response'][:300].replace(chr(10), ' ')}")


def run_forced_mode_comparison(prompt: str) -> None:
    print("=" * 72)
    print("FORCED MODE COMPARISON")
    print("=" * 72)
    print(f"Prompt: {prompt}\n")

    results = []

    for forced_mode in ["home", "analytic", "engagement"]:
        result = run_one_forced_mode(prompt, forced_mode)
        save_path = log_result(result)
        result["save_path"] = save_path
        results.append(result)

    for result in results:
        mode = result["mode"]
        print("-" * 72)
        print(f"MODE: {mode.upper()}")
        print(f"Saved:      {result['save_path']}")
        print(f"Latency:    {result['latency_seconds']} s")
        print(f"Chars:      {result['response_chars']}")
        print(f"Settings:   {result['generation_settings']}")
        print("\nSYSTEM PROMPT:")
        print(result["system_prompt"])
        print("\nMODEL OUTPUT:")
        print(result["response"])
        print()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    run_batch(TEST_PROMPTS)

    print("\n" + "=" * 72 + "\n")
    run_forced_mode_comparison("Help me calmly compare two job options.")


