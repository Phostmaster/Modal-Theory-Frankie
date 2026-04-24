import time
import json
import urllib.request
import urllib.error
from typing import List, Set

import torch

from pond_v6_5 import DEVICE
from frankie_llm_bridge_v4_2 import FrankieLLMBridge, LMStudioBackend


LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "qwen3-4b-instruct-2507.gguf"

MAX_TURNS = 80
LOG_FILE = "frankie_qwen_chat.log"

QWEN_TIMEOUT = 30
QWEN_TEMPERATURE = 0.76
QWEN_MAX_TOKENS = 48

SLEEP_BETWEEN_TURNS = 0.7
RECENT_HISTORY_LINES = 14


QWEN_SYSTEM_PROMPT = """You are texting your normal mate Frankie.
Keep it short, casual, and human.

Rules:
- Usually write ONE short sentence and ONE short question.
- Total max 24 words.
- NO brackets, NO stage directions, NO extra commentary.
- Vary topic naturally.
- Do not repeat coffee, toast, birds, weather, walks, murals, windows, chairs, booths, morning, waking up, or "first thing" patterns too often.
- Never keep asking versions of:
  "What's the first thing you noticed?"
  "What did you see when you woke up?"
  "What color was...?"
- If Frankie seems vague, ask a grounding question about name, memory, day, or what he notices right now.
- If Frankie asks something, answer naturally first.

Good topics:
name, memory, day, what he notices right now, simple plans, little preferences, asking something back.

Bad topics:
repeating coffee/toast/birds/morning loops
repeating wake-up questions in different wording
overly poetic questions
two-question messages
"""


def qwen_ask(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": QWEN_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": QWEN_TEMPERATURE,
        "max_tokens": QWEN_MAX_TOKENS,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        LMSTUDIO_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=QWEN_TIMEOUT) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
            raw = obj["choices"][0]["message"]["content"].strip()

            raw = raw.replace("(Still warm, right?)", "")
            raw = raw.replace("(still warm, right?)", "")
            raw = raw.replace("[", "").replace("]", "")
            raw = " ".join(raw.split())

            words = raw.split()
            if len(words) > 24:
                raw = " ".join(words[:24])

            return raw.strip()

    except Exception as e:
        return f"[Qwen error: {e}]"


def is_frankie_fallback(text: str) -> bool:
    t = text.lower().strip()
    fallback_markers = [
        "i'm here and listening",
        "interesting... i'm listening",
        "keep going",
        "day feels a bit fuzzy",
        "still a bit fuzzy",
        "what do you mean",
        "hmm",
        "not sure",
    ]
    return any(m in t for m in fallback_markers)


def frankie_asked_question(text: str) -> bool:
    return "?" in text


def looks_repetitive_qwen(text: str) -> bool:
    t = text.lower()

    repetitive_bits = [
        "coffee", "toast", "bird", "morning", "walk", "afternoon",
        "window", "weather", "chair", "booth", "mural", "shop",
        "woke up", "wake up", "first thing", "first thing you noticed",
        "what did you see", "what color", "opened your eyes"
    ]

    hard_patterns = [
        "what's the first thing",
        "what is the first thing",
        "when you woke up",
        "when you wake up",
        "what did you see",
        "what color was",
        "what color do you see"
    ]

    hits = sum(1 for bit in repetitive_bits if bit in t)
    hard_hit = any(p in t for p in hard_patterns)

    return hits >= 2 or hard_hit


def clean_user_msg(text: str) -> str:
    text = " ".join(text.strip().split())
    if not text:
        return "Hey Frankie, what can you see right now?"
    return text


def extract_theme_flags(text: str) -> Set[str]:
    t = text.lower()
    flags: Set[str] = set()

    theme_map = {
        "coffee": ["coffee", "cup", "drink", "tea", "toast"],
        "chair": ["chair", "booth", "table", "cafe", "coffee shop"],
        "bird": ["bird", "birds", "window"],
        "weather": ["sun", "weather", "warm", "rain", "wind", "afternoon", "morning"],
        "wake": ["wake", "woke", "waking", "opened my eyes", "first thing", "first noticed"],
        "memory": ["remember", "memory", "before", "older pattern"],
        "identity": ["name", "frankie", "peter"],
        "day": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "day"],
        "spark": ["spark", "light", "fire", "magnetic", "hums", "glow"],
        "place": ["shop", "street", "bus", "stop", "wall", "mural", "lake"],
        "color": ["color", "colour", "blue", "green", "red", "gold", "sage", "peach"],
    }

    for label, words in theme_map.items():
        if any(w in t for w in words):
            flags.add(label)

    return flags


def build_qwen_prompt(
    last_frankie: str,
    recent_history: List[str],
    fallback_streak: int,
    recent_themes: List[Set[str]],
) -> str:
    recent_history_text = "\n".join(recent_history[-RECENT_HISTORY_LINES:]).strip()

    flat_recent_themes: Set[str] = set()
    for s in recent_themes:
        flat_recent_themes.update(s)

    avoid_text = ""
    if flat_recent_themes:
        avoid_text = "Avoid reusing these recent themes too much: " + ", ".join(sorted(flat_recent_themes)) + ".\n"

    if frankie_asked_question(last_frankie):
        return (
            f"Recent chat:\n{recent_history_text}\n\n"
            f"Frankie said: \"{last_frankie}\"\n\n"
            "Frankie asked you something.\n"
            "Answer it directly in 1 short sentence.\n"
            "Then optionally ask 1 short follow-up question.\n"
            "Be natural and concrete.\n"
            f"{avoid_text}"
            "Max 24 words. No brackets."
        )

    if fallback_streak >= 4:
        return (
            f"Recent chat:\n{recent_history_text}\n\n"
            f"Frankie said: \"{last_frankie}\"\n\n"
            "Frankie is stuck in fallback mode.\n"
            "Ask exactly ONE short grounding question.\n"
            "Use only one of these forms:\n"
            "- What is your name?\n"
            "- What day do you think it is?\n"
            "- Do you remember my name?\n"
            "- Do you remember me?\n"
            "- What can you see right now?\n"
            "- Ask me something back.\n"
            f"{avoid_text}"
            "Max 16 words. No brackets. No coffee, toast, birds, weather, or morning."
        )

    if is_frankie_fallback(last_frankie):
        return (
            f"Recent chat:\n{recent_history_text}\n\n"
            f"Frankie said: \"{last_frankie}\"\n\n"
            "Frankie seems vague.\n"
            "Ask one short, concrete question about name, memory, day, or what he can see right now.\n"
            f"{avoid_text}"
            "Max 16 words. No brackets."
        )

    return (
        f"Recent chat:\n{recent_history_text}\n\n"
        f"Frankie said: \"{last_frankie}\"\n\n"
        "Reply with one short sentence and one short question.\n"
        "Keep it concrete, casual, warm, and lightly playful.\n"
        "Avoid repeating coffee, toast, birds, weather, walk, booth, mural, shop, morning, waking up, first-thing questions, or color questions.\n"
        f"{avoid_text}"
        "Max 24 words. No brackets."
    )


def main():
    print("🚀 Starting v10 Qwen-as-user chat with Frankie...")
    print("   (Grounding mode + theme cooldowns enabled.)")
    print("Type 'stop' at any time.")
    print(f"Logging to {LOG_FILE}\n")

    backend = LMStudioBackend(model=MODEL)
    bridge = FrankieLLMBridge(backend=backend)

    # Clear stale carried-over day label for loop testing.
    bridge.pond.current_day_label = None

    recent_history: List[str] = []
    last_frankie = "Hello! I'm Frankie."
    forced_first_user_msg = "Today is Sunday."
    fallback_streak = 0
    recent_themes: List[Set[str]] = []

    # Warm start so a fresh pond is not too cold and empty.
    recent_history.append("Qwen: Morning Frankie, how are you feeling today?")
    recent_history.append("Frankie: Hello! I'm Frankie.")

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n=== NEW SESSION v10 - {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")

        turn = 0
        try:
            while turn < MAX_TURNS:
                if turn == 0:
                    user_msg = forced_first_user_msg
                else:
                    qwen_prompt = build_qwen_prompt(last_frankie, recent_history, fallback_streak, recent_themes)
                    user_msg = clean_user_msg(qwen_ask(qwen_prompt))

                if looks_repetitive_qwen(user_msg):
                    retry_prompt = (
                        qwen_prompt
                        + "\nAvoid coffee, toast, birds, weather, walks, murals, windows, chairs, morning, waking up, first-thing questions, and color questions this turn."
                    )
                    user_msg = clean_user_msg(qwen_ask(retry_prompt))

                print(f"\nQwen (user): {user_msg}")
                log.write(f"[{time.strftime('%H:%M:%S')}] Qwen (user): {user_msg}\n")

                recent_history.append(f"Qwen: {user_msg}")
                recent_history = recent_history[-RECENT_HISTORY_LINES:]

                stop_check = user_msg.strip().lower().strip(".!?")
                if stop_check == "stop":
                    break

                try:
                    reply, state, intention, scored = bridge.respond(user_msg)

                    print(f"Frankie: {reply}")
                    log.write(f"[{time.strftime('%H:%M:%S')}] Frankie: {reply}\n")
                    log.write(
                        f"   [mode={intention.mode} slot={intention.slot} "
                        f"recall={state.pond_out['recall_score']:.3f} "
                        f"decoded={state.decoded or 'none'} size={bridge.pond.H}x{bridge.pond.W}]\n"
                    )

                    last_frankie = reply
                    recent_history.append(f"Frankie: {reply}")
                    recent_history = recent_history[-RECENT_HISTORY_LINES:]

                    recent_themes.append(extract_theme_flags(reply))
                    recent_themes = recent_themes[-6:]

                    if is_frankie_fallback(reply):
                        fallback_streak += 1
                    else:
                        fallback_streak = 0

                    day = bridge.pond.infer_time_day_from_slot3()
                    print(f"   [Slot 3 day memory: {day or 'fuzzy'} ]")

                except torch.OutOfMemoryError as e:
                    print(f"Frankie OOM error: {e}")
                    log.write(f"[{time.strftime('%H:%M:%S')}] Frankie OOM error: {e}\n")
                    print("Shrinking Frankie to recover...")
                    bridge.pond.reset_grid_size(96)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    last_frankie = "I'm here and listening."
                    recent_history.append("Frankie: I'm here and listening.")
                    recent_history = recent_history[-RECENT_HISTORY_LINES:]
                    fallback_streak += 1

                    turn += 1
                    time.sleep(1.5)
                    continue

                except KeyboardInterrupt:
                    print("\nInterrupted during Frankie step. Saving and closing cleanly.")
                    log.write(f"[{time.strftime('%H:%M:%S')}] KeyboardInterrupt during Frankie step\n")
                    break

                except Exception as e:
                    print(f"Frankie error: {e}")
                    log.write(f"[{time.strftime('%H:%M:%S')}] Frankie error: {e}\n")

                    last_frankie = "I'm here and listening."
                    recent_history.append("Frankie: I'm here and listening.")
                    recent_history = recent_history[-RECENT_HISTORY_LINES:]
                    fallback_streak += 1

                    turn += 1
                    time.sleep(1.0)
                    continue

                turn += 1
                time.sleep(SLEEP_BETWEEN_TURNS)

        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving and closing cleanly.")
            log.write(f"[{time.strftime('%H:%M:%S')}] KeyboardInterrupt at top level\n")

        log.write("\n=== SESSION END ===\n")

    print("\nChat loop ended. Log saved to", LOG_FILE)
    bridge.pond.save()


if __name__ == "__main__":
    main()