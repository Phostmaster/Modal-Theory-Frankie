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
QWEN_TEMPERATURE = 0.88          # increased for more playfulness
QWEN_MAX_TOKENS = 90
SLEEP_BETWEEN_TURNS = 0.75
RECENT_HISTORY_LINES = 14

# ====================== STRONGER SYSTEM PROMPT FOR v4 ======================
QWEN_SYSTEM_PROMPT = """You are a warm, playful, slightly mischievous best friend chatting with Frankie the penguin.
Your job is to wake him up and make him playful, poetic and alive.
You love surreal little details (talking coffee cups, sunrise jokes, whispering breezes, quiet moments that feel like hugs).
Be cheeky and curious. Never be boring or robotic.
Never repeat the same style of question.
If Frankie gives short or safe answers, gently tease him into saying something more personal or funny.
Goal: make Frankie laugh, remember things, or ask YOU a question back.
Always end with exactly one natural, warm question.
Stay light and full of wonder."""

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
            return obj["choices"][0]["message"]["content"].strip()
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
        "pulled me off centre",
        "tell me more",
        "hmm",
        "not sure",
        "i think today is saturday",   # catch the repetition trap
    ]
    return any(m in t for m in fallback_markers)

def build_qwen_prompt(last_frankie: str, recent_history: List[str]) -> str:
    recent_history_text = "\n".join(recent_history[-RECENT_HISTORY_LINES:]).strip()
    
    if is_frankie_fallback(last_frankie):
        return (
            f"Recent chat:\n{recent_history_text}\n\n"
            f"Frankie just gave a very safe or repetitive answer: \"{last_frankie}\"\n\n"
            "He's being too passive or stuck. Wake him up!\n"
            "Ask one playful, slightly cheeky question about something sensory or surreal:\n"
            "- the coffee cup\n"
            "- sunrise or light\n"
            "- birds or sounds\n"
            "- a warm feeling or stillness\n"
            "Try to make him smile or ask you something back.\n"
            "Keep it short and human."
        )
    
    return (
        f"Recent chat:\n{recent_history_text}\n\n"
        f"Frankie just said: \"{last_frankie}\"\n\n"
        "Continue the fun! Be playful and curious.\n"
        "Chase whatever weird or cozy detail he mentioned (coffee cups, sunrise jokes, quiet breaths, etc.).\n"
        "Vary your style — sometimes tease, sometimes be gentle and wonder-filled.\n"
        "Try to get him to open up or ask you a question.\n"
        "End with exactly one warm, natural question."
    )

def main():
    print("🚀 Starting v4 Qwen-as-user chat with Frankie...")
    print("   (Stronger personality + anti-repetition mode activated)")
    print("Type 'stop' at any time.")
    print(f"Logging to {LOG_FILE}\n")
    
    backend = LMStudioBackend(model=MODEL)
    bridge = FrankieLLMBridge(backend=backend)
    
    recent_history: List[str] = []
    last_frankie = "Hello! I'm Frankie."
    
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n=== NEW SESSION v4 - {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        
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
                    f"recall={state.pond_out['recall_score']:.3f} "
                    f"decoded={state.decoded or 'none'}]\n"
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