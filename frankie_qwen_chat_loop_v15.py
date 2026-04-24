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
QWEN_TEMPERATURE = 0.78
QWEN_MAX_TOKENS = 32
SLEEP_BETWEEN_TURNS = 0.7
RECENT_HISTORY_LINES = 14

QWEN_SYSTEM_PROMPT = """You are texting your normal penguin mate Frankie in a normal house.
Keep every reply short, warm and human.
Rules:
- ONE short sentence + ONE short question (max 24 words total)
- NO brackets, NO emojis, NO poetry words (snow, ice, ridge, wind, whisper, hum, song, light, breathing, glow, soft, quiet, feel)
- NO repeating the same question style
- You are ONLY chatting with Frankie the penguin — never invent other people
- Talk about normal house things: tea, socks, cat, window, kitchen, hoodie, room, garden
- If Frankie is vague, ask one simple grounding question:
  - What day do you think it is?
  - Do you remember me?
  - What can you see right now?
  - Ask me something!
- If Frankie asks something, answer it first then ask one short thing back.
Stay chill and friendly like a real mate."""

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
            raw = raw.replace("(Still warm, right?)", "").replace("(still warm, right?)", "")
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
        "i'm here and listening", "interesting... i'm listening", "keep going",
        "day feels a bit fuzzy", "still a bit fuzzy", "what do you mean",
        "hmm", "not sure", "i think today is saturday"
    ]
    return any(m in t for m in fallback_markers)

def frankie_asked_question(text: str) -> bool:
    return "?" in text

def looks_repetitive_qwen(text: str) -> bool:
    t = text.lower()
    bad_bits = ["snow", "ice", "ridge", "wind", "whisper", "hum", "song", "light", "breathing", "glow", "soft", "quiet", "feel", "first thing", "one thing", "last thing"]
    return any(bit in t for bit in bad_bits)

def clean_user_msg(text: str) -> str:
    text = " ".join(text.strip().split())
    if not text:
        return "Hey Frankie, what can you see right now?"
    return text

def extract_theme_flags(text: str) -> Set[str]:
    t = text.lower()
    flags: Set[str] = set()
    theme_map = {
        "coffee": ["coffee", "cup", "tea", "toast"],
        "bird": ["bird", "birds"],
        "weather": ["sun", "warm", "wind"],
        "wake": ["wake", "woke", "first thing"],
        "memory": ["remember", "memory"],
        "identity": ["name", "frankie"],
        "day": ["sunday", "monday", "day"],
    }
    for label, words in theme_map.items():
        if any(w in t for w in words):
            flags.add(label)
    return flags

def build_qwen_prompt(last_frankie: str, recent_history: List[str], fallback_streak: int, recent_themes: List[Set[str]]) -> str:
    recent_history_text = "\n".join(recent_history[-RECENT_HISTORY_LINES:]).strip()
    flat_recent_themes = set()
    for s in recent_themes:
        flat_recent_themes.update(s)
    avoid_text = ""
    if flat_recent_themes:
        avoid_text = "Avoid these: " + ", ".join(sorted(flat_recent_themes)) + ".\n"
    if frankie_asked_question(last_frankie):
        return f"Recent chat:\n{recent_history_text}\n\nFrankie said: \"{last_frankie}\"\n\nAnswer directly then ask one short question.\n{avoid_text}Max 24 words."
    if fallback_streak >= 4 or is_frankie_fallback(last_frankie):
        return f"Recent chat:\n{recent_history_text}\n\nFrankie said: \"{last_frankie}\"\n\nAsk one simple grounding question: What day is it? Do you remember me? What can you see now? Ask me something!\n{avoid_text}Max 18 words."
    return f"Recent chat:\n{recent_history_text}\n\nFrankie said: \"{last_frankie}\"\n\nOne short sentence + one short question. Warm and playful.\n{avoid_text}Max 24 words."

def main():
    print("🚀 Starting v15 Qwen-as-user chat with Frankie...")
    print("   (Ultra-clean normal mate chat — no poetry at all!)")
    print("Type 'stop' at any time.")
    print(f"Logging to {LOG_FILE}\n")
    
    backend = LMStudioBackend(model=MODEL)
    bridge = FrankieLLMBridge(backend=backend)
    bridge.pond.current_day_label = None
    
    recent_history: List[str] = []
    last_frankie = "Hello! I'm Frankie."
    forced_first_user_msg = "Today is Sunday."
    fallback_streak = 0
    recent_themes: List[Set[str]] = []
    
    recent_history.append("Qwen: Morning Frankie, how are you feeling today?")
    recent_history.append("Frankie: Hello! I'm Frankie.")
    
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n=== NEW SESSION v15 - {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        turn = 0
        try:
            while turn < MAX_TURNS:
                if turn == 0:
                    user_msg = forced_first_user_msg
                else:
                    qwen_prompt = build_qwen_prompt(last_frankie, recent_history, fallback_streak, recent_themes)
                    user_msg = clean_user_msg(qwen_ask(qwen_prompt))
                
                if looks_repetitive_qwen(user_msg):
                    retry_prompt = qwen_prompt + "\nNo snow, ice, ridge, wind, whisper, hum, song, light, breathing, glow, soft, quiet, feel, or poetry words."
                    user_msg = clean_user_msg(qwen_ask(retry_prompt))
                
                print(f"\nQwen (user): {user_msg}")
                log.write(f"[{time.strftime('%H:%M:%S')}] Qwen (user): {user_msg}\n")
                recent_history.append(f"Qwen: {user_msg}")
                recent_history = recent_history[-RECENT_HISTORY_LINES:]
                
                if user_msg.strip().lower().strip(".!?") == "stop":
                    break
                
                try:
                    reply, state, intention, scored = bridge.respond(user_msg)
                    print(f"Frankie: {reply}")
                    log.write(f"[{time.strftime('%H:%M:%S')}] Frankie: {reply}\n")
                    log.write(f" [mode={intention.mode} slot={intention.slot} recall={state.pond_out['recall_score']:.3f}]\n")
                    
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
                    print(f" [Slot 3 day memory: {day or 'fuzzy'} ]")
                    
                except Exception as e:
                    print(f"Frankie error: {e}")
                    log.write(f"[{time.strftime('%H:%M:%S')}] Frankie error: {e}\n")
                    last_frankie = "I'm here and listening."
                    recent_history.append("Frankie: I'm here and listening.")
                    recent_history = recent_history[-RECENT_HISTORY_LINES:]
                    fallback_streak += 1
                
                turn += 1
                time.sleep(SLEEP_BETWEEN_TURNS)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving and closing cleanly.")
        log.write("\n=== SESSION END ===\n")
    
    print("\nChat loop ended. Log saved to", LOG_FILE)
    bridge.pond.save()

if __name__ == "__main__":
    main()