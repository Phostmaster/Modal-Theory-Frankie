import os
import time
import json
from datetime import datetime
from typing import Optional, Tuple
from frankie_llm_bridge_v4_2 import (
    FrankieLLMBridge,
    LMStudioBackend,
    TEXT_MODEL_DEFAULT,
    TEXT_BASE_URL_DEFAULT,
    VISION_MODEL_DEFAULT,
    VISION_BASE_URL_DEFAULT,
)

MAX_TURNS = 200
LOG_FILE = "frankie_qwen_vision_chat.log"
SLEEP_BETWEEN_TURNS = 0.2
IMAGE_FOLDER = r"C:\Users\Peter\Desktop\frankie_images"

def parse_user_input(user_input: str) -> Tuple[str, Optional[str]]:
    user_input = user_input.strip()
    if "[" in user_input and user_input.endswith("]"):
        text_part, image_part = user_input.rsplit("[", 1)
        user_text = text_part.strip()
        image_filename = image_part[:-1].strip()
        image_path = os.path.join(IMAGE_FOLDER, image_filename)
        return user_text, image_path
    return user_input, None

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

def load_notebook():
    try:
        with open("frankie_notebook.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"consolidated": []}

def save_notebook(notebook):
    with open("frankie_notebook.json", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

def consolidate_important_memories(chat_history, bridge, max_turns=50, min_score=7):
    """Frankie’s quiet memory helper — keeps only the gold nuggets."""
    if len(chat_history) < 5:
        return

    recent = chat_history[-max_turns:]
    
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

Return ONLY this JSON, nothing else:
{{"memories": [{{"summary": "...", "score": 9, "reason": "..."}}, ...]}}
"""

    response = bridge.backend.generate(prompt, temperature=0.3, max_tokens=400)
    
    try:
        new_memories = json.loads(response)["memories"]
        gold = [m for m in new_memories if m["score"] >= min_score]
        
        notebook = load_notebook()
        today = datetime.now().strftime("%Y-%m-%d %H:%M")
        notebook["consolidated"].append({"date": today, "entries": gold})
        save_notebook(notebook)
        
        print(f"✅ Pond Clean Time done — saved {len(gold)} gold nuggets")
        
    except Exception as e:
        print(f"⚠️ Clean failed (keeping everything): {e}")

def main():
    print("🚀 Starting Frankie vision chat loop...")
    print("Type 'exit' to quit.")
    print(f"Images folder: {IMAGE_FOLDER}")
    print("Use: your message [image.jpg]")
    print(f"Logging to {LOG_FILE}\n")

    os.makedirs(IMAGE_FOLDER, exist_ok=True)

    backend = LMStudioBackend(
        model=TEXT_MODEL_DEFAULT,
        base_url=TEXT_BASE_URL_DEFAULT,
    )
    bridge = FrankieLLMBridge(
        backend=backend,
        vision_model=VISION_MODEL_DEFAULT,
        vision_base_url=VISION_BASE_URL_DEFAULT,
    )

    bridge.pond.current_day_label = None
    chat_history = []  # ← our new safe memory list

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n=== NEW SESSION VISION - {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        turn = 0
        try:
            while turn < MAX_TURNS:
                raw = input("\nYou: ").strip()
                if not raw:
                    continue
                if raw.lower() == "exit":
                    break

                user_text, image_path = parse_user_input(raw)

                # === SPECIAL COMMAND: clean pond ===
                if user_text.lower().strip() in ["clean pond", "pond clean", "clean"]:
                    consolidate_important_memories(chat_history, bridge)
                    print("🧼 Pond Clean Time triggered manually")
                    continue

                if image_path:
                    print(f"[Debug] Resolved image path: {image_path}")
                if image_path and not os.path.exists(image_path):
                    print(f"[Warning] Image not found: {image_path}")
                    image_path = None

                print(f"\nUser: {user_text}")
                if image_path:
                    print(f" (with image: {os.path.basename(image_path)})")

                log.write(f"[{time.strftime('%H:%M:%S')}] User: {user_text}")
                if image_path:
                    log.write(f" [image: {os.path.basename(image_path)}]\n")
                else:
                    log.write("\n")

                try:
                    vision_desc = None
                    if image_path:
                        vision_desc = bridge.get_vision_description(image_path)
                        print(f"Vision desc: {vision_desc}")
                        log.write(f"[{time.strftime('%H:%M:%S')}] Vision desc: {vision_desc}\n")

                    reply, state, intention, scored = bridge.respond(
                        user_text,
                        image_path=image_path,
                        vision_desc=vision_desc,
                    )

                    if is_frankie_fallback(reply):
                        label = "Frankie (fallback):"
                        choice_note = f" → Pond used fallback (recall: {state.pond_out['recall_score']:.3f})"
                    else:
                        label = "Frankie (Pond own):"
                        choice_note = f" → Pond chose its own reply (creative! recall: {state.pond_out['recall_score']:.3f})"

                    print(f"{label} {reply}")
                    print(choice_note)
                    if getattr(state, 'vision_desc', None):
                        print(f" [Vision used: {state.vision_desc}]")
                    day = bridge.pond.infer_time_day_from_slot3()
                    print(f" [Slot 3 day memory: {day or 'fuzzy'} ]")

                    log.write(f"[{time.strftime('%H:%M:%S')}] {label} {reply}\n")
                    log.write(f"{choice_note}\n")
                    if getattr(state, 'vision_desc', None):
                        log.write(f"[{time.strftime('%H:%M:%S')}] Vision used: {state.vision_desc}\n")
                    log.write(
                        f"[{time.strftime('%H:%M:%S')}] mode={intention.mode} "
                        f"slot={intention.slot} recall={state.pond_out['recall_score']:.3f}\n"
                    )

                    # Add to our safe history list
                    chat_history.append(f"User: {user_text}\nFrankie: {reply}")

                    # === Auto Pond Clean every 30 turns ===
                    if turn % 30 == 0:
                        consolidate_important_memories(chat_history, bridge)

                except Exception as e:
                    print(f"Error: {e}")
                    log.write(f"[{time.strftime('%H:%M:%S')}] Error: {e}\n")

                turn += 1
                time.sleep(SLEEP_BETWEEN_TURNS)

        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving and closing cleanly.")

        log.write("\n=== SESSION END ===\n")

    print("\nChat loop ended. Log saved to", LOG_FILE)
    bridge.pond.save()

if __name__ == "__main__":
    main()