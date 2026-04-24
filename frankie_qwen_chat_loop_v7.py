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
MAX_TURNS = 80
LOG_FILE = "frankie_qwen_chat.log"
QWEN_TIMEOUT = 30
QWEN_TEMPERATURE = 0.85
QWEN_MAX_TOKENS = 38              # super short & fresh
SLEEP_BETWEEN_TURNS = 0.7
RECENT_HISTORY_LINES = 14

# ====================== v7 — NO MORE POETRY LOOPS ======================
QWEN_SYSTEM_PROMPT = """You are texting your cheeky penguin mate Frankie at breakfast.
Rules you MUST follow:
- ONE short casual sentence + exactly ONE question
- TOTAL max 28 words
- NO emojis, NO poetry, NO haikus, NO repeating snow/socks/flakes
- Change topic gently every 3-4 turns (sun, wind, snacks, dreams, music, whatever)
- If Frankie says something fun, chase it once then switch

Examples:
Hey Frankie, how’s the morning frost feeling?
Did the wind try any tricks today?
What’s the silliest dream you had last night?

Keep it warm, playful and fresh every time."""

def qwen_ask(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": QWEN_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "temperature": QWEN_TEMPERATURE,
        "max_tokens": QWEN_MAX_TOKENS,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(LMSTUDIO_URL, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=QWEN_TIMEOUT) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
            raw = obj["choices"][0]["message"]["content"].strip()
            # nuclear trim
            words = raw.split()
            if len(words) > 28:
                raw = " ".join(words[:14]) + " " + " ".join(words[-10:])
            return raw
    except Exception as e:
        return f"[Qwen error: {e}]"

def is_frankie_fallback(text: str) -> bool:
    t = text.lower().strip()
    fallback_markers = [
        "i'm here and listening", "interesting... i'm listening", "keep going",
        "day feels a bit fuzzy", "still a bit fuzzy", "what do you mean",
        "hmm", "not sure", "i think today is saturday"
    ]
    return any(m in t for m in fallback_markers)

def build_qwen_prompt(last_frankie: str, recent_history: List[str]) -> str:
    recent_history_text = "\n".join(recent_history[-RECENT_HISTORY_LINES:]).strip()
    
    if is_frankie_fallback(last_frankie):
        return (
            f"Recent chat:\n{recent_history_text}\n\n"
            f"Frankie said: \"{last_frankie}\"\n\n"
            "He's safe again. Wake him with one short warm sentence + one fresh question (max 28 words total).\n"
            "Change topic — no more snow/socks/haikus. Make it fun and new."
        )
    
    return (
        f"Recent chat:\n{recent_history_text}\n\n"
        f"Frankie said: \"{last_frankie}\"\n\n"
        "Reply with one short casual sentence + one question (max 28 words).\n"
        "Chase the fun once, then gently switch topic. Keep it fresh and playful."
    )

def main():
    print("🚀 Starting v7 Qwen-as-user chat with Frankie...")
    print("   (No more poetry loops — fresh topics every few turns!)")
    print("Type 'stop' at any time.")
    print(f"Logging to {LOG_FILE}\n")
    
    backend = LMStudioBackend(model=MODEL)
    bridge = FrankieLLMBridge(backend=backend)
    
    recent_history: List[str] = []
    last_frankie = "Hello! I'm Frankie."
    
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n=== NEW SESSION v7 - {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        
        turn = 0
        while turn < MAX_TURNS:
            qwen_prompt = build_qwen_prompt(last_frankie, recent_history)
            user_msg = qwen_ask(qwen_prompt)
            
            print(f"\nQwen (user): {user_msg}")
            log.write(f"[{time.strftime('%H:%M:%S')}] Qwen (user): {user_msg}\n")
            
            recent_history.append(f"Qwen: {user_msg}")
            
            if "stop" in user_msg.lower():
                break
                
            try:
                reply, state, intention, scored = bridge.respond(user_msg)
                print(f"Frankie: {reply}")
                log.write(f"[{time.strftime('%H:%M:%S')}] Frankie: {reply}\n")
                log.write(
                    f"   [mode={intention.mode} slot={intention.slot} "
                    f"recall={state.pond_out['recall_score']:.3f}]\n"
                )
                
                last_frankie = reply
                recent_history.append(f"Frankie: {reply}")
                recent_history = recent_history[-RECENT_HISTORY_LINES:]
                
                day = bridge.pond.infer_time_day_from_slot3()
                print(f"   [Slot 3 day memory: {day or 'fuzzy'} ]")
                
            except Exception as e:
                print(f"Frankie error: {e}")
                log.write(f"[{time.strftime('%H:%M:%S')}] Frankie error: {e}\n")
                last_frankie = "I'm here and listening."
                recent_history.append("Frankie: I'm here and listening.")
                recent_history = recent_history[-RECENT_HISTORY_LINES:]
            
            turn += 1
            time.sleep(SLEEP_BETWEEN_TURNS)
        
        log.write("\n=== SESSION END ===\n")
    
    print("\nChat loop ended. Log saved to", LOG_FILE)
    bridge.pond.save()

if __name__ == "__main__":
    main()