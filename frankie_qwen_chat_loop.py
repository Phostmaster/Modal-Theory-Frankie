import time
import json
import urllib.request
import urllib.error
from typing import List

import torch

from pond_v6_5 import DEVICE
from frankie_llm_bridge_v4_2 import FrankieLLMBridge, LMStudioBackend


LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "qwen3-4b-instruct-2507.gguf"

MAX_TURNS = 160
LOG_FILE = "frankie_qwen_chat.log"
QWEN_TIMEOUT = 30
QWEN_TEMPERATURE = 0.70
QWEN_MAX_TOKENS = 80

SLEEP_BETWEEN_TURNS = 0.8
RECENT_HISTORY_LINES = 12


QWEN_SYSTEM_PROMPT = """You are a warm, natural human chatting with Frankie.
Be simple, concrete, and varied.
Do not be overly poetic, philosophical, or repetitive.

Rules:
- If Frankie asks you a direct question, answer it first in a normal human way.
- After answering, you may ask one short follow-up question.
- If Frankie seems vague or passive, ask a concrete grounding question.
- Prefer short everyday questions over dreamy abstract ones.
- Do not keep asking about mornings, waking up, colours, birds, coffee, feelings, or memories in slightly different words.
- Do not repeat the same question pattern.
- Keep replies short: 1 or 2 sentences max.
- Sound like a real friendly person, not a therapist, interviewer, or poet.

Good question types:
- name
- what day it is
- whether he remembers something
- what he notices right now
- what he would say to someone
- simple preferences
- recent conversation continuity

Good examples:
- "Fair enough. What day do you think it is?"
- "Alright. Do you remember my name?"
- "What do you notice right now?"
- "Would you ask me something back?"
- "What sort of thing feels familiar today?"

Bad examples:
- repeated wake-up questions
- repeated colour questions
- repeated poetic metaphor questions
- long emotional monologues
- ignoring Frankie’s question
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
            return obj["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Qwen error: {e}]"


def is_frankie_fallback(text: str) -> bool:
    t = text.lower()
    fallback_markers = [
        "i'm here and listening",
        "interesting... i'm listening",
        "keep going",
        "day feels a bit fuzzy",
        "still a bit fuzzy",
        "what do you mean",
        "pulled me off centre",
    ]
    return any(m in t for m in fallback_markers)


def build_qwen_prompt(last_frankie: str, recent_history: List[str], fallback_streak: int) -> str:
    recent_history_text = "\n".join(recent_history[-RECENT_HISTORY_LINES:]).strip()

    if "?" in last_frankie:
        return (
            f"Recent chat:\n{recent_history_text}\n\n"
            f"Frankie just said: \"{last_frankie}\"\n\n"
            "Frankie asked you a question.\n"
            "Answer it directly in 1 short sentence.\n"
            "Then optionally ask 1 short follow-up question.\n"
            "Be natural and concrete."
        )

    if fallback_streak >= 2:
        return (
            f"Recent chat:\n{recent_history_text}\n\n"
            f"Frankie just said: \"{last_frankie}\"\n\n"
            "Frankie is stuck in a vague loop.\n"
            "Ask exactly one short grounding question.\n"
            "Use only one of these forms:\n"
            "- What day do you think it is?\n"
            "- Do you remember my name?\n"
            "- What is your name?\n"
            "- Do you remember me?\n"
            "- What are you noticing right now?\n"
            "- Would you ask me something?\n"
            "Do not ask about mornings, colours, birds, coffee, memories from childhood, or feelings in poetic language."
        )

    if is_frankie_fallback(last_frankie):
        return (
            f"Recent chat:\n{recent_history_text}\n\n"
            f"Frankie just said: \"{last_frankie}\"\n\n"
            "Frankie seems vague.\n"
            "Ask one short, concrete, everyday question.\n"
            "Avoid poetic or dreamy wording."
        )

    return (
        f"Recent chat:\n{recent_history_text}\n\n"
        f"Frankie just said: \"{last_frankie}\"\n\n"
        "Reply naturally like a friendly human.\n"
        "Then ask at most one short follow-up question.\n"
        "Keep it concrete and varied.\n"
        "Avoid repeating recent themes."
    )


def clean_user_msg(text: str) -> str:
    text = " ".join(text.strip().split())
    if not text:
        return "Hello Frankie, how are you feeling today?"
    return text


def main():
    print("Starting improved Qwen-as-user chat with Frankie...")
    print("Type 'stop' at any time.")
    print("Logging to", LOG_FILE)

    backend = LMStudioBackend(model=MODEL)
    bridge = FrankieLLMBridge(backend=backend)
    bridge.pond.current_day_label = None

    recent_history: List[str] = []
    last_frankie = "Hello! I'm Frankie."
    fallback_streak = 0

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n=== NEW SESSION {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")

        turn = 0
        while turn < MAX_TURNS:
            qwen_prompt = build_qwen_prompt(last_frankie, recent_history, fallback_streak)
            user_msg = clean_user_msg(qwen_ask(qwen_prompt))

            print(f"\nQwen (user): {user_msg}")
            log.write(f"Qwen (user): {user_msg}\n")
            recent_history.append(f"Qwen: {user_msg}")

            if "stop" in user_msg.lower():
                break

            try:
                reply, state, intention, scored = bridge.respond(user_msg)

                print(f"Frankie: {reply}")
                log.write(f"Frankie: {reply}\n")
                log.write(
                    f"   [mode={intention.mode} slot={intention.slot} "
                    f"recall={state.pond_out['recall_score']:.3f} "
                    f"decoded={state.decoded or 'none'} size={bridge.pond.H}x{bridge.pond.W}]\n"
                )

                last_frankie = reply
                recent_history.append(f"Frankie: {reply}")
                recent_history = recent_history[-RECENT_HISTORY_LINES:]

                if is_frankie_fallback(reply):
                    fallback_streak += 1
                else:
                    fallback_streak = 0

                day = bridge.pond.infer_time_day_from_slot3()
                print(f"   [Slot 3 day memory: {day or 'fuzzy'} ]")

            except torch.OutOfMemoryError as e:
                print(f"Frankie OOM error: {e}")
                log.write(f"Frankie OOM error: {e}\n")
                print("Shrinking Frankie to recover...")
                bridge.pond.reset_grid_size(96)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                last_frankie = "I'm here and listening."
                recent_history.append("Frankie: I'm here and listening.")
                recent_history = recent_history[-RECENT_HISTORY_LINES:]
                turn += 1
                time.sleep(1.5)
                continue

            except Exception as e:
                print(f"Frankie error: {e}")
                log.write(f"Frankie error: {e}\n")
                last_frankie = "I'm here and listening."
                recent_history.append("Frankie: I'm here and listening.")
                recent_history = recent_history[-RECENT_HISTORY_LINES:]
                turn += 1
                time.sleep(1.5)
                continue

            turn += 1
            time.sleep(SLEEP_BETWEEN_TURNS)

        log.write("\n=== SESSION END ===\n")

    print("Chat loop ended. Log saved to", LOG_FILE)
    bridge.pond.save()


if __name__ == "__main__":
    main()