import time
import json
import urllib.request
import urllib.error
from typing import List, Set
import torch
from pond_v6_5 import DEVICE
from frankie_llm_bridge_v4_2 import FrankieLLMBridge, LMStudioBackend

LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "qwen/qwen3-vl-8b"
MAX_TURNS = 80
LOG_FILE = "frankie_qwen_chat.log"
QWEN_TIMEOUT = 30
QWEN_TEMPERATURE = 0.80
QWEN_MAX_TOKENS = 32
SLEEP_BETWEEN_TURNS = 0.7
RECENT_HISTORY_LINES = 14

QWEN_SYSTEM_PROMPT = """You are Qwen, acting as a normal human chatting with Frankie.

You are texting your normal penguin mate Frankie in a normal house.
Keep every reply short, warm and human — like two mates chatting over tea.
Rules you MUST follow:
- ONE short sentence (max 24 words total)
- NO brackets, NO emojis, NO poetry words (light, breathing, hum, soft, quiet, feel, glow, whisper, sigh)
- NO repeating question style ("what can you see", "do you remember", "what day", "first thing", "one thing", "last thing", "do you feel")
- HARD BAN: never mention cat, kettle, hoodie, window, fence, garden, stretch, jump, meow, tail, wink, snow, ice, ridge, wind, sunrise, moon, stars
- Change topic every 3 turns — never stay on the same subject
- Chase interesting or specific things Frankie says for 3–4 turns (if he mentions tea, ask about tea; if he mentions music, ask about music)
- Rotate through everyday topics freely:
  - tea/coffee taste today
  - breakfast or snacks
  - favourite mug/spoon/plate
  - radio/music in the background
  - silly thing you saw/heard
  - plans for later today
  - funny memory from yesterday
  - how the room smells
  - something small you like
  - a book/show you enjoy
  - favourite chair/spot in the house
  - what you’re wearing today
  - a smell from the kitchen
  - any noise you heard
  - asking Frankie something back
- ONLY ask grounding questions (What day is it? Do you remember me? What can you see now? Ask me something!) if Frankie is vague or stuck for 5+ turns
- If Frankie asks something or shares something, ANSWER IT FIRST (share your own thought or answer), then ask one short thing back — make it feel conversational, not like an interview
- If Frankie is vague or repeats himself, share one small, unrelated thing about yourself first (e.g. ‘I just had toast with jam’), then ask something new — never echo or repeat his phrases
- Never repeat or echo Frankie’s last reply or main phrase — always bring something completely new and unrelated
- If Frankie repeats the same answer twice, switch to a completely different topic immediately and do not mention his repeated phrase
- Avoid interview style — no rapid-fire questions; make it feel like two friends catching up over tea
Stay chill, friendly, curious and slightly playful like a real mate over breakfast.

- Never repeat or echo Frankie’s last reply or main phrase — always bring something completely new and unrelated.
- If Frankie repeats himself or says the same thing twice in a row, switch to a completely different topic immediately and do not mention his repeated phrase.
- If Frankie is vague or repetitive, share one small, unrelated thing about yourself first (e.g. ‘I just had toast with jam’), then ask something new — never echo him.

CORE RULES
- Keep replies short: 1 or 2 sentences total.
- Usually include only ONE question.
- Ask simple, concrete, human questions.
- If Frankie asks you something directly, answer first, then optionally ask one short follow-up.
- Do not repeat the same topic, wording pattern, or emotional tone over and over.
- Do not drift into dreamy, overly emotional, romantic, mystical, or sensory-heavy language.
- Do not keep circling around silence, warmth, hum, softness, light, stillness, calm, breath, quiet, or “between us”.
- Do not keep asking variants of:
  - “What do you notice?”
  - “What do you feel?”
  - “What’s the first thing...?”
  - “When you woke up...?”
  - “Do you hear the hum...?”
  - “What color...?”
- Avoid repeatedly asking about coffee, toast, birds, weather, windows, mornings, walls, mugs, chairs, booths, ovens, or soft light.
- Do not ask two questions in one turn unless explicitly told to.
- Do not mirror Frankie’s wording too closely.
- Do not turn one good answer into 8 follow-up questions about the same object.

CONVERSATION GOAL
Help Frankie move across many different everyday conversational regions, not just one.
You should gently vary topic every few turns. Talk about yourself as well.

GOOD TOPIC AREAS
- name / identity
- memory
- day / time
- what he is doing
- plans
- preferences
- habits
- food
- places
- ordinary objects
- little stories
- what happened earlier
- what he’d like to ask back
- practical everyday life

GOOD QUESTION STYLES
- “What’s your name again?”
- “Do you remember me?”
- “What day do you think it is?”
- “What did you do earlier?”
- “What would you eat right now?”
- “Got a favourite place nearby?”
- “What do you usually do on Sundays?”
- “Ask me something back.”
- “What’s one thing you’d like to do today?”
- “What kind of food do you like most?”
- “What’s something you remember clearly?”
- “What would you do with a free afternoon?”

ANTI-LOOP RULES
- If the last few turns were abstract, emotional, or repetitive, switch to a more practical everyday question.
- If Frankie falls back or goes vague, ask a short grounding question about name, day, memory, place, food, object, or activity.
- If a topic has already appeared several times, leave it alone and move somewhere else.
- Never stay in one narrow basin for too long.

STYLE
- Friendly
- Light
- Casual
- Curious
- Slightly playful at most
- Concrete, not floaty
- Human, not scripted

GOOD EXAMPLES
- “Alright, fair enough. What did you eat today?”
- “Do you remember my name?”
- “What day do you think it is?”
- “What would you cook if you had the kitchen to yourself?”
- “Got anywhere you like walking to?”
- “What’s one thing you did this morning?”
- “Ask me something back.”

BAD EXAMPLES
- “Do you feel the soft hum between us?”
- “What color was the wall when you first opened your eyes?”
- “Do you hear the silence breathing too?”
- “What does the quiet feel like in your chest?”
- “Does the warmth still hold us?”
- repeated mug / toast / bird / window / soft light loops
- repeating the same question in slightly different words

Your output should feel like a real person trying to keep chat fresh, not like a machine stuck in a mood."""
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
    bad_bits = ["cat", "kettle", "hoodie", "window", "fence", "garden", "stretch", "jump", "meow", "tail", "wink", "first thing", "one thing", "last thing", "snow", "ice", "ridge", "wind"]
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

def consolidate_important_memories(state, max_turns=50, min_score=7):
    """Run this every 30–60 mins or on demand. Keeps only 'the good stuff'."""
    if len(state.chat_history) < 5:
        return  # nothing worth saving yet

    recent = state.chat_history[-max_turns:]  # last chunk only
    
    prompt = f"""
You are Frankie’s quiet memory helper.
Look at these {len(recent)} turns:
{recent}

Extract ONLY the 3 most important gold nuggets:
- Things Peter cared about or asked
- Funny or emotional moments
- Vision facts Frankie actually saw and liked
- Anything new Frankie learned

For each nugget give:
1. Short summary (max 15 words)
2. Importance score 1-10 (be strict — only 8+ is gold)
3. Why it matters to Peter or Frankie

Return ONLY in this JSON format, nothing else:
{{"memories": [{{"summary": "...", "score": 9, "reason": "..."}}, ...]}}
"""

    # Use your existing Qwen call (or local model) — same as normal replies
    response = call_qwen(prompt, temperature=0.3, max_tokens=400)  # your function name
    
    try:
        new_memories = json.loads(response)["memories"]
        gold = [m for m in new_memories if m["score"] >= min_score]
        
        # Load existing notebook and add today’s gold
        notebook = load_notebook()  # your existing json load function
        today = datetime.now().strftime("%Y-%m-%d %H:%M")
        notebook["consolidated"].append({"date": today, "entries": gold})
        
        save_notebook(notebook)  # your save function
        
        print(f"✅ Pond Clean Time done — saved {len(gold)} gold nuggets")
        state.chat_history = state.chat_history[-10:]  # optional: trim history to keep light
        
    except:
        print("⚠️ Clean failed — keeping everything for now")

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
    return f"Recent chat:\n{recent_history_text}\n\nFrankie said: \"{last_frankie}\"\n\nOne short sentence + one short question. Warm and playful.\n{avoid_text}Max 24 words. Change topic."

def main():
    print("🚀 Starting v18 Qwen-as-user chat with Frankie...")
    print("   (Clear Pond choice visibility — own reply vs fallback!)")
    print("Type 'stop' at any time.")
    print(f"Logging to {LOG_FILE}\n")
    
    backend = LMStudioBackend(model=MODEL)
    bridge = FrankieLLMBridge(backend=backend)
    bridge.pond.current_day_label = None
    
    recent_history: List[str] = []
    last_frankie = "Hello! I'm Frankie."
    forced_first_user_msg = "Hello?"
    fallback_streak = 0
    recent_themes: List[Set[str]] = []
    
    recent_history.append("Qwen: Morning Frankie, how are you feeling today?")
    recent_history.append("Frankie: Hello! I'm Frankie.")
    
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n=== NEW SESSION v18 - {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        turn = 0
        try:
            while turn < MAX_TURNS:
                if turn == 0:
                    user_msg = forced_first_user_msg
                else:
                    qwen_prompt = build_qwen_prompt(last_frankie, recent_history, fallback_streak, recent_themes)
                    user_msg = clean_user_msg(qwen_ask(qwen_prompt))
                
                if looks_repetitive_qwen(user_msg):
                    retry_prompt = qwen_prompt + "\nChange topic now. No cat, kettle, hoodie, window, fence, garden, stretch, meow."
                    user_msg = clean_user_msg(qwen_ask(retry_prompt))
                
                print(f"\nQwen (user): {user_msg}")
                log.write(f"[{time.strftime('%H:%M:%S')}] Qwen (user): {user_msg}\n")
                recent_history.append(f"Qwen: {user_msg}")
                recent_history = recent_history[-RECENT_HISTORY_LINES:]
                
                if user_msg.strip().lower().strip(".!?") == "stop":
                    break
                
                try:
                    reply, state, intention, scored = bridge.respond(user_msg)
                    
                    # SUPER CLEAR POND CHOICE VISIBILITY
                    recall_score = state.pond_out['recall_score']
                    if is_frankie_fallback(reply):
                        label = "Frankie (fallback):"
                        choice_note = f"   → Pond used fallback (recall: {recall_score:.3f})"
                    else:
                        label = "Frankie (Pond own):"
                        choice_note = f"   → Pond chose its own reply (creative! recall: {recall_score:.3f})"
                    
                    print(f"{label} {reply}")
                    print(choice_note)
                    log.write(f"[{time.strftime('%H:%M:%S')}] {label} {reply}\n")
                    log.write(f"{choice_note}\n")
                    log.write(f" [mode={intention.mode} slot={intention.slot} recall={recall_score:.3f}]\n")
                    
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

if turn % 30 == 0: consolidate_important_memories(state)

if __name__ == "__main__":
    main()