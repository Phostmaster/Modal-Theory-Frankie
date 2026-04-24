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
QWEN_TEMPERATURE = 0.82
QWEN_MAX_TOKENS = 42              # short but warm
SLEEP_BETWEEN_TURNS = 0.7
RECENT_HISTORY_LINES = 14

# ====================== v6 — best of v5 + smarter nudge ======================
QWEN_SYSTEM_PROMPT = """You are texting your cheeky penguin best friend Frankie.
Keep every reply natural, warm and SHORT: one casual sentence + exactly one clear question.
Max ~30 words total. No emoji spam, no long stories.
When Frankie says something fun or poetic, gently chase it and try to get him to say something even more creative or silly.
Vary topics lightly so we don't loop on socks/snacks.
Goal: make him smile and talk like a real little penguin with stories."""

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
            # safety trim
            if len(raw.split()) > 32:
                raw = " ".join(raw.split()[:18]) + " " + " ".join(raw.split()[-8:])
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
            f"Frankie just said: \"{last_frankie}\"\n\n"
            "He's being safe again. Wake him up with one short warm sentence + one playful question.\n"
            "Try to spark something silly or poetic from him."
        )
    
    return (
        f"Recent chat:\n{recent_history_text}\n\n"
        f"Frankie just said: \"{last_frankie}\"\n\n"
        "Reply with one short casual sentence + one question (~30 words total).\n"
        "Chase anything fun he said. Vary topics gently. Make him smile."
    )

def main():
    print("🚀 Starting v6 Qwen-as-user chat with Frankie...")
    print("   (Balanced short + creative nudge — best of v5!)")
    print("Type 'stop' at any time.")
    print(f"Logging to {LOG_FILE}\n")
    
    backend = LMStudioBackend(model=MODEL)
    bridge = FrankieLLMBridge(backend=backend)
    
    recent_history: List[str] = []
    last_frankie = "Hello! I'm Frankie."
    
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n=== NEW SESSION v6 - {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        
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