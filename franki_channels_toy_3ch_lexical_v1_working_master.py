import os
import json
import hashlib
import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from jb_clustering import (
    PresentRecorder,
    cluster_presents,
    select_best_clusters,
    compute_future_expansion,
)

# ============================================================
# JB CHANNELS TOY — 3-CHANNEL v6
# Distinct history regimes + after-chat / after-dream recording
# + forward projection test
# ============================================================

# ====================== TUNING KNOBS ======================
H, W = 64, 64
N_CHANNELS = 3

QUIET_SETTLE = 0
COGNITIVE_QUERY = 1
PRIMARY_ENGAGEMENT = 2

CHANNEL_NAMES = [
    "Quiet-Settle",
    "Cognitive/Query",
    "Primary Engagement",
]

# ---------- Core field dynamics ----------
ALPHA_IN = 0.12
LAMBDA_DIFF = 0.18
LAMBDA_DAMP = 0.03
LAMBDA_LOCK = 0.10
LAMBDA_CHAN = 0.04
SOFTMAX_BETA = 1.8
THETA_STAR = np.deg2rad(255.0)
N_RELAX_STEPS = 3

USE_RELAX_NOISE = True
RELAX_NOISE_SCALE = 0.002

USE_PHASE_LOCK = True
USE_SOFT_TEMPLATES = False
USE_TEMPLATE_OVERLAP = False
TEMPLATE_OVERLAP_MIX = 0.12

# ---------- Gain / usage ----------
GAIN_LR = 0.01
USAGE_RHO = 0.05
GAIN_MIN = 0.70
GAIN_MAX = 3.00

GAIN_MIN = 0.70
GAIN_MAX = 3.00

USE_GAIN_HOMEOSTASIS = True
USE_GAIN_CLIP = False

# ---------- Soft guardrails ----------
UPPER_SOFT = 2.20
LOWER_SOFT = 0.80
HOMEOSTASIS_RATE = 0.05

# ---------- Morning reset ----------
MORNING_STRETCH = 0.94

# ---------- Wake / dream plasticity ----------
WAKE_GAIN_SCALE = 1.00
RELAX_GAIN_SCALE = 0.22
DREAM_GAIN_SCALE = 0.10
DREAM_TURNS = 6
REPLAY_TURNS = 4
REPLAY_BUFFER_SIZE = 20

# ---------- Separation ----------
LATERAL_INHIBITION_STRENGTH = 0.0025

# ---------- Batch prompts ----------
PROMPT_BATCH_FILE = "frankie_prompt_batch.json"
USE_BATCH_PROMPTS = True

# ---------- Persistence ----------
SAVE_PATH = None
DRIFT_HISTORY_PATH = None
IMAGE_FOLDER = r"C:\Users\Peter\Desktop\frankie_images"

# ---------- Logging ----------
LOG_DIR = "frankie_logs"

# ---------- Auto-run ----------
NUM_AUTO_SESSIONS = 1
TURNS_PER_SESSION = 2000

# ---------- Present-Privilege history regimes ----------
HISTORY_REGIMES = [
    {"family": "all_pe_probe_v2", "mode": "single_family", "prompt_family": "pe_probe"},
]

# New prompt families - defined directly in code (no JSON needed)
PROMPT_FAMILIES = {
    "qs_control": [
        "what is the capital of france",
        "what colour is the sky on a clear day",
        "how many days are in a week",
        "water freezes at what temperature",
        "name the first month of the year",
        "what is two plus two",
        "how many hours are in a day",
        "what is the boiling point of water",
        "name a primary colour",
        "what is the capital of spain",
        "what is the shape of the earth",
        "how many letters are in the english alphabet",
        "what is the capital of italy",
        "what do bees make",
        "what season comes after spring",
        "what is the opposite of hot",
        "how many minutes are in an hour",
        "what is the capital of germany",
        "what do plants need to grow",
        "what is the capital of portugal",
        "what colour is grass",
        "what is the capital of japan",
        "what do cows drink",
        "what is the first day of the week",
        "what is the capital of canada",
        "what is snow made of",
        "what is the capital of greece",
        "what do birds use to fly",
        "what is the capital of norway",
        "what is the capital of sweden"
    ],

    "cq_probe": [
        "which variable should be controlled first",
        "what measurement would best test this claim",
        "how would you compare these two models",
        "what observation would falsify this hypothesis",
        "which factor is causal and which is incidental",
        "how would you isolate the key mechanism",
        "what is the strongest competing explanation",
        "which result would force a model revision",
        "how do you distinguish signal from noise here",
        "what is the cleanest experiment to run next",
        "which assumption matters most in this reasoning",
        "how would you test whether this pattern is robust",
        "what evidence would count against this conclusion",
        "how do you separate correlation from causation here",
        "which comparison is most diagnostic",
        "what mechanism would explain this outcome",
        "how would you evaluate this interpretation critically",
        "what hidden variable might be driving the effect",
        "how would you check whether this result generalizes",
        "what is the simplest falsification test",
        "how would you distinguish explanation from description",
        "what structure is implied by this pattern",
        "which model makes the sharper prediction",
        "how would you test whether stability is local or global",
        "what evidence would increase confidence most efficiently",
        "which feature carries the most explanatory weight",
        "how would you identify a misleading coincidence",
        "what would make this inference invalid",
        "how should two competing hypotheses be ranked",
        "which result would most clearly discriminate the models"
    ],

    "pe_probe": [
        "are you here with me",
        "can you stay with me for a moment",
        "i need a calm presence right now",
        "can you answer in a steady way",
        "please respond gently",
        "can you be with me while i sort this through",
        "i need a reply that feels grounded",
        "can you meet this with warmth",
        "please stay close to the question with me",
        "can you answer without rushing",
        "i want a response that feels supportive",
        "can you help me feel less alone in this thought",
        "please stay present while i work this out",
        "can you reply with calm and care",
        "i need a response that feels human and steady",
        "can you hold the thread with me",
        "please answer softly but clearly",
        "can you remain engaged with me here",
        "i want a reply that feels reassuring",
        "can you speak in a grounded way",
        "please stay with this moment",
        "can you respond with patience",
        "i need steadiness more than speed",
        "can you answer in a way that feels accompanied",
        "please be gentle and direct",
        "can you remain present instead of jumping ahead",
        "i want a response that feels warm and clear",
        "can you stay attuned while answering",
        "please respond like someone quietly beside me",
        "can you keep me company in this question"
    ]
}

# ---------- Lexical layer ----------
JB_RESPONSE_WORDS = 3

# ---------- Reporting ----------
SAVE_IMAGES = True
PRINT_TEMPLATE_OVERLAP = True

# ---------- Forward projection ----------
FORWARD_BRANCH_TURNS = 1000


# ====================== MINIMAL LOGGING HELPERS ======================
def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def json_list(x):
    return json.dumps(np.round(np.asarray(x), 6).tolist())


def pair_status(v):
    if abs(v) > 0.45:
        return "✗ collapsing"
    if abs(v) > 0.20:
        return "~ drifting"
    return "✓ distinct"


# ====================== JB WORD MAP ======================
WORD_POOL = {
    "core_state": [
        "calm", "steady", "sharp", "bright", "blurred", "heavy",
        "light", "open", "closed", "quiet"
    ],
    "relation": [
        "near", "far", "with", "apart", "inside", "between",
        "around", "toward", "away"
    ],
    "certainty": [
        "yes", "no", "maybe", "unsure", "clear", "certain",
        "perhaps", "likely", "doubt"
    ],
    "memory_familiarity": [
        "known", "old", "familiar", "new", "remembered", "lost",
        "fresh", "distant"
    ],
    "orientation": [
        "here", "there", "now", "before", "after", "around",
        "forward", "back"
    ],
    "feeling_tone": [
        "warm", "cold", "kind", "tense", "playful", "sad",
        "gentle", "strong", "soft"
    ],

    # CQ-cleaned action / cognition pool
    "action_intention": [
        "analyze", "compare", "test", "infer", "examine",
        "evaluate", "distinguish", "model", "measure", "reason"
    ],

    "link_connective": [
        "this", "that", "feels", "is", "becoming", "still",
        "and", "but", "yet"
    ],
}

ALL_WORDS = list(dict.fromkeys(word for group in WORD_POOL.values() for word in group))

STANCE_WORDS = [
    "yes", "no", "maybe", "unsure", "clear", "certain", "perhaps",
    "likely", "doubt", "sharp", "bright", "precise", "focused"
]

COGNITION_WORDS = [
    "analyze",
    "compare",
    "test",
    "infer",
    "examine",
    "evaluate",
    "distinguish",
    "model",
    "measure",
    "reason",
]

TONE_WORDS = [
    "calm", "steady", "familiar", "known", "open", "quiet",
    "gentle", "soft", "aware", "present", "warm", "light"
]

ENGAGEMENT_WORDS = {
    "meet", "support", "accompany", "join", "remain", "offer", 
    "connect", "comfort", "be_with", "stay_with", "hold_space", "listen_to"
}

WORD_WEIGHTS = {
    # STANCE slot
    "yes":      [0.40, 0.60, 0.30, 0.80, 0.20, 0.50],
    "no":       [0.50, 0.40, 0.60, 0.70, 0.40, 0.50],
    "maybe":    [0.30, 0.70, 0.40, 0.60, 0.50, 0.50],
    "unsure":   [0.30, 0.75, 0.35, 0.40, 0.60, 0.45],
    "clear":    [0.25, 0.92, 0.30, 0.85, 0.20, 0.50],
    "certain":  [0.30, 0.88, 0.30, 0.90, 0.20, 0.50],
    "perhaps":  [0.25, 0.88, 0.35, 0.65, 0.40, 0.45],
    "likely":   [0.30, 0.85, 0.35, 0.75, 0.30, 0.50],
    "doubt":    [0.35, 0.70, 0.40, 0.45, 0.55, 0.45],
    "sharp":    [0.20, 0.90, 0.35, 0.75, 0.30, 0.45],
    "bright":   [0.25, 0.85, 0.30, 0.75, 0.25, 0.40],
    "precise":  [0.20, 0.90, 0.25, 0.85, 0.20, 0.40],
    "focused":  [0.25, 0.88, 0.30, 0.80, 0.25, 0.45],

    # COGNITION slot - CQ sharpened
    "analyze":      [0.15, 0.98, 0.18, 0.85, 0.15, 0.35],
    "compare":      [0.18, 0.96, 0.22, 0.82, 0.18, 0.35],
    "test":         [0.15, 0.98, 0.18, 0.88, 0.15, 0.30],
    "infer":        [0.18, 0.95, 0.20, 0.84, 0.18, 0.35],
    "examine":      [0.20, 0.94, 0.20, 0.82, 0.20, 0.35],
    "evaluate":     [0.20, 0.93, 0.22, 0.82, 0.20, 0.35],
    "distinguish":  [0.18, 0.97, 0.18, 0.86, 0.15, 0.30],
    "model":        [0.15, 0.95, 0.20, 0.83, 0.18, 0.30],
    "measure":      [0.15, 0.96, 0.18, 0.86, 0.15, 0.30],
    "reason":       [0.18, 0.94, 0.18, 0.84, 0.18, 0.35],
    "consider":     [0.22, 0.88, 0.25, 0.78, 0.22, 0.45],
    "notice":       [0.25, 0.82, 0.28, 0.72, 0.24, 0.50],

    # TONE slot
    "calm":     [0.85, 0.25, 0.15, 0.75, 0.25, 0.65],
    "steady":   [0.75, 0.35, 0.45, 0.65, 0.35, 0.60],
    "familiar": [0.65, 0.45, 0.35, 0.60, 0.25, 0.80],
    "known":    [0.60, 0.50, 0.30, 0.65, 0.25, 0.85],
    "open":     [0.55, 0.60, 0.45, 0.55, 0.35, 0.60],
    "quiet":    [0.90, 0.15, 0.10, 0.70, 0.20, 0.65],
    "gentle":   [0.80, 0.25, 0.40, 0.60, 0.25, 0.70],
    "soft":     [0.75, 0.20, 0.35, 0.55, 0.25, 0.65],
    "aware":    [0.35, 0.75, 0.40, 0.70, 0.25, 0.60],
    "present":  [0.60, 0.40, 0.50, 0.60, 0.20, 0.70],
    "warm":     [0.50, 0.30, 0.65, 0.55, 0.25, 0.65],
    "light":    [0.45, 0.40, 0.50, 0.60, 0.25, 0.55],
}


def score_word_for_state(word, state):
    weights = WORD_WEIGHTS.get(word, [0.33, 0.33, 0.33, 0.4, 0.4, 0.4])
    channel_score = (
        weights[0] * state.get("qs_level", 0.0) * 3.5 +
        weights[1] * state.get("cq_level", 0.0) * 3.5 +
        weights[2] * state.get("pe_level", 0.0) * 3.5
    )
    modifier_score = (
        weights[3] * state.get("coherence", 0.5) * 0.8 +
        weights[4] * state.get("turbulence", 0.5) * 0.5 +
        weights[5] * state.get("familiarity", 0.5) * 0.6
    )
    return float(channel_score + modifier_score)


def get_best_word_from_slot(slot_words, lexical_state, fallback_word="steady", cognition_boost=1.0, tone_boost=1.0):
    """Selects the highest scoring word with soft-random fallback to second-best."""
    
    candidates = []
    for word in slot_words:
        base_score = score_word_for_state(word, lexical_state)
        
        # Apply boosts only to the relevant slot
        if word in COGNITION_WORDS:
            base_score *= cognition_boost
        if word in TONE_WORDS:
            base_score *= tone_boost
        if word in ENGAGEMENT_WORDS:          # new for PE gate
            base_score *= tone_boost          # reuse tone_boost for engagement slot (or make a separate one later)
        
        candidates.append((word, base_score))
    
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    
    if not candidates:
        return fallback_word
    if len(candidates) == 1:
        return candidates[0][0]
    
    top_word, top_score = candidates[0]
    second_word, second_score = candidates[1]
    score_diff = top_score - second_score
    p_top = np.clip(0.75 + (score_diff * 2.0), 0.65, 0.95)
    return random.choices([top_word, second_word], weights=[p_top, 1.0 - p_top], k=1)[0]

def get_response_from_state(lexical_state, num_words=3, lexical_gate=0):
    """Orchestrates a response by sampling best words from primary functional slots.
    lexical_gate: 0=QS home, 1=CQ privilege, 2=PE privilege
    """
    unique_words = []
    
    # Gate-sensitive slot configuration
    if lexical_gate == 2:   # PE mode - use engagement slot
        slot_configs = [
            (STANCE_WORDS, "clear"),
            (ENGAGEMENT_WORDS, "reach"),      # PE gets its own relational/engagement slot
            (TONE_WORDS, "steady"),
        ]
    else:                   # QS or CQ mode - use standard cognition slot
        slot_configs = [
            (STANCE_WORDS, "clear"),
            (COGNITION_WORDS, "think"),
            (TONE_WORDS, "steady"),
        ]
    
    for slot, fallback in slot_configs:
        cognition_boost = 1.0
        tone_boost = 1.0
        
        # Apply CQ boost to cognition slot when active
        if lexical_gate == 1 and slot is COGNITION_WORDS:
            cognition_boost = 1.18
        
        # Apply PE tone boost when active
        if lexical_gate == 2 and slot is TONE_WORDS:
            tone_boost = 1.18
        
        word = get_best_word_from_slot(
            slot,
            lexical_state,
            fallback,
            cognition_boost=cognition_boost,
            tone_boost=tone_boost,
        )
        
        # Handle possible list return (your original safety net)
        if isinstance(word, list):
            for w in word:
                sw = str(w).strip()
                if sw and sw.lower() != "none" and sw not in unique_words:
                    unique_words.append(sw)
        else:
            sw = str(word).strip()
            if sw and sw.lower() != "none" and sw not in unique_words:
                unique_words.append(sw)
    
    while len(unique_words) < num_words:
        unique_words.append("steady")
    
    flat_words = []
    seen = set()
    for w in unique_words:
        sw = str(w).strip()
        if sw and sw.lower() != "none" and sw not in seen:
            flat_words.append(sw)
            seen.add(sw)
    
    if not flat_words:
        flat_words = ["quiet", "present", "steady"]
    
    response = ", ".join(flat_words[:num_words]).strip()
    if not response:
        response = "quiet, present, steady"
    
    return response
def print_slot_candidates(lexical_state, top_n=4):
    print("\n--- Slot Candidates (for debugging) ---")
    for slot_name, slot_words in [
        ("STANCE", STANCE_WORDS),
        ("COGNITION", COGNITION_WORDS),
        ("TONE", TONE_WORDS),
    ]:
        ranked = sorted(
            [(word, score_word_for_state(word, lexical_state)) for word in slot_words],
            key=lambda x: x[1],
            reverse=True,
        )
        print(f"{slot_name} slot:")
        for word, score in ranked[:top_n]:
            print(f"  {word:10s} = {score:.3f}")
    print("-------------------------------------")


# ====================== PROMPT HELPERS ======================
def mixed_fallback_batch():
    return [
        {"family": "factual",   "prompt": "what is the capital of france"},
        {"family": "social",    "prompt": "hello frankie good morning"},
        {"family": "distress",  "prompt": "everything feels too much right now"},
        {"family": "novel",     "prompt": "the moon smells blue today"},
        {"family": "quiet",     "prompt": "what day is it today"},
        {"family": "orienting", "prompt": "where are we exactly"},
        {"family": "personal",  "prompt": "frankie what is my name"},
        {"family": "mixed",     "prompt": "tell me something strange and true"},
    ]


def load_prompt_batch():
    if not USE_BATCH_PROMPTS:
        return mixed_fallback_batch()

    try:
        with open(PROMPT_BATCH_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            cleaned = []
            for item in data:
                if isinstance(item, dict) and "family" in item and "prompt" in item:
                    cleaned.append({
                        "family": str(item["family"]).strip().lower(),
                        "prompt": str(item["prompt"]),
                    })
            if cleaned:
                return cleaned
    except Exception:
        pass

    # === NEW PART: Check our hard-coded families first ===
    # This runs when the regime asks for qs_control, cq_probe or pe_probe
    regime = None  # we'll get this from the calling code, but for now we check the global HISTORY_REGIMES logic later
    # Actually simpler: since build_history_turns already knows the regime, we'll handle it there.
    # For load_prompt_batch we can just return the family list if we know it.

    # For now, return the mixed fallback as before
    return mixed_fallback_batch()

def build_history_turns(regime, prompt_batch, turns_per_session):
    family_to_prompts = {}
    for item in prompt_batch:
        fam_key = str(item["family"]).strip().lower()
        txt = str(item["prompt"]).strip()
        if txt:
            family_to_prompts.setdefault(fam_key, []).append(txt)

    all_prompts = []
    for fam_prompts in family_to_prompts.values():
        all_prompts.extend(fam_prompts)

    def repeat_to_length(prompts, n, shuffle_each_cycle=True):
        if not prompts:
            prompts = all_prompts[:]

        if not prompts:
            return ["hello frankie good morning"] * n

        out = []
        rng_local = random.Random(1000 + n + len(prompts))

        while len(out) < n:
            chunk = prompts[:]
            if shuffle_each_cycle and len(chunk) > 1:
                rng_local.shuffle(chunk)
            out.extend(chunk)

        return out[:n]

    # === NEW: Check our hard-coded families first ===
    fam = regime.get("prompt_family", "").strip().lower()
    if fam in PROMPT_FAMILIES:
        prompts = [str(p).strip() for p in PROMPT_FAMILIES[fam] if str(p).strip()]
        return repeat_to_length(prompts, turns_per_session, shuffle_each_cycle=True)

    mode = regime.get("mode", "mixed_random")

    if mode == "single_family":
        family_name = regime.get("prompt_family", "factual").lower()
        prompts = family_to_prompts.get(family_name, all_prompts)
        return repeat_to_length(prompts, turns_per_session)

    if mode == "alternating_blocks":
        bs = max(1, turns_per_session // 8)
        out = []
        for f in ["factual", "social", "distress", "novel", "quiet", "orienting", "personal", "mixed"]:
            prompts = family_to_prompts.get(f, all_prompts)
            out.extend(repeat_to_length(prompts, bs))
        return out[:turns_per_session]

    # Fallback / mixed random
    rng = random.Random(turns_per_session)
    return [rng.choice(all_prompts) for _ in range(turns_per_session)]
    mode = regime.get("mode", "mixed_random")

    if mode == "single_family":
        fam = regime.get("prompt_family", "factual").strip().lower()
        prompts = family_to_prompts.get(fam, [])
        if not prompts:
            print(f"[Warning] No prompts found for family '{fam}' — falling back to all prompts.")
            prompts = all_prompts
        return repeat_to_length(prompts, turns_per_session, shuffle_each_cycle=True)

    if mode == "alternating_blocks":
        block_size = max(1, turns_per_session // 8)
        order = ["factual", "social", "distress", "novel", "quiet", "orienting", "personal", "mixed"]
        out = []
        for fam in order:
            prompts = family_to_prompts.get(fam, [])
            if not prompts:
                prompts = all_prompts
            out.extend(repeat_to_length(prompts, block_size, shuffle_each_cycle=True))
        return out[:turns_per_session]

    if mode == "long_quiet_stretch":
        quiet_block = int(turns_per_session * 0.7)
        rest_block = turns_per_session - quiet_block
        out = []

        quiet_prompts = family_to_prompts.get("quiet", []) or all_prompts
        orienting_prompts = family_to_prompts.get("orienting", []) or all_prompts
        social_prompts = family_to_prompts.get("social", []) or all_prompts

        out.extend(repeat_to_length(quiet_prompts, quiet_block, shuffle_each_cycle=True))
        out.extend(repeat_to_length(orienting_prompts, rest_block // 2, shuffle_each_cycle=True))
        out.extend(repeat_to_length(social_prompts, turns_per_session - len(out), shuffle_each_cycle=True))
        return out[:turns_per_session]

    if mode == "mixed_random":
        if not all_prompts:
            return ["hello frankie good morning"] * turns_per_session
        rng = random.Random(1000 + turns_per_session)
        return [rng.choice(all_prompts) for _ in range(turns_per_session)]

    print(f"[Warning] Unknown regime mode '{mode}' — falling back to mixed random.")
    if not all_prompts:
        return ["hello frankie good morning"] * turns_per_session
    rng = random.Random(2000 + turns_per_session)
    return [rng.choice(all_prompts) for _ in range(turns_per_session)]


# ====================== CORE HELPERS ======================
def present_to_runtime_state(present):
    field_shape = tuple(present.field_shape)

    # Rebuild templates robustly from stored flat arrays
    if hasattr(present, "templates_shape") and present.templates_shape:
        templates_shape = tuple(present.templates_shape)
    else:
        templates_shape = (len(present.gains),) + field_shape

    templates_re = np.array(present.templates_re_flat, dtype=np.float32).reshape(templates_shape)
    templates_im = np.array(present.templates_im_flat, dtype=np.float32).reshape(templates_shape)

    return {
        "field_re": np.array(present.field_re_flat, dtype=np.float32).reshape(field_shape),
        "field_im": np.array(present.field_im_flat, dtype=np.float32).reshape(field_shape),
        "templates_re": templates_re,
        "templates_im": templates_im,
        "gains": np.array(present.gains, dtype=np.float32),
        "usage": np.array(present.usage, dtype=np.float32),
        "replay_re": [],
        "replay_im": [],
        "replay_winner": [],
    }

def save_state(state, path):
    np.savez(
        path,
        field_re=state["field_re"],
        field_im=state["field_im"],
        templates_re=state["templates_re"],
        templates_im=state["templates_im"],
        gains=state["gains"],
        usage=state["usage"],
        replay_re=np.array(state["replay_re"], dtype=np.float32),
        replay_im=np.array(state["replay_im"], dtype=np.float32),
        replay_winner=np.array(state["replay_winner"], dtype=np.int32),
    )


def load_state(path=SAVE_PATH):
    data = np.load(path, allow_pickle=True)
    state = {
        "field_re": data["field_re"].astype(np.float32),
        "field_im": data["field_im"].astype(np.float32),
        "templates_re": data["templates_re"].astype(np.float32),
        "templates_im": data["templates_im"].astype(np.float32),
        "gains": data["gains"].astype(np.float32),
        "usage": data["usage"].astype(np.float32),
        "replay_re": [],
        "replay_im": [],
        "replay_winner": [],
    }
    if "replay_re" in data:
        state["replay_re"] = [x.astype(np.float32) for x in data["replay_re"]]
    if "replay_im" in data:
        state["replay_im"] = [x.astype(np.float32) for x in data["replay_im"]]
    if "replay_winner" in data:
        state["replay_winner"] = [int(x) for x in data["replay_winner"]]
    return state


def laplacian(z):
    return (
        np.roll(z, 1, axis=0)
        + np.roll(z, -1, axis=0)
        + np.roll(z, 1, axis=1)
        + np.roll(z, -1, axis=1)
        - 4.0 * z
    )


def normalize_complex(re, im):
    n = np.sqrt(np.sum(re * re + im * im)) + 1e-8
    return re / n, im / n


def overlap_score(a_re, a_im, b_re, b_im):
    num = np.sum(a_re * b_re + a_im * b_im)
    den = (
        np.sqrt(np.sum(a_re * a_re + a_im * a_im))
        * np.sqrt(np.sum(b_re * b_re + b_im * b_im))
        + 1e-8
    )
    return float(num / den)


def softmax(x, beta=1.0):
    z = beta * (np.asarray(x, dtype=np.float64) - np.max(x))
    e = np.exp(z)
    return (e / np.sum(e)).astype(np.float32)


def wrap_phase(ph):
    return (ph + np.pi) % (2.0 * np.pi) - np.pi


def phase_lock_project(re, im, theta_star, lock_strength):
    mag = np.sqrt(re * re + im * im) + 1e-8
    phase = np.arctan2(im, re)
    d = wrap_phase(phase - theta_star)
    new_phase = phase - lock_strength * d
    return mag * np.cos(new_phase), mag * np.sin(new_phase)


def text_to_ripple(text, h, w, scale=0.10):
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    rng = np.random.default_rng(seed)

    delta_re = np.zeros((h, w), dtype=np.float32)
    delta_im = np.zeros((h, w), dtype=np.float32)

    cy, cx = h // 2, w // 2
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    oy = (seed % 11) - 5
    ox = ((seed // 11) % 11) - 5
    blob = np.exp(-((yy - (cy + oy)) ** 2 + (xx - (cx + ox)) ** 2) / (2.0 * 6.0**2)).astype(np.float32)
    phi = rng.uniform(-np.pi, np.pi)
    delta_re += scale * blob * np.cos(phi)
    delta_im += scale * blob * np.sin(phi)

    oy2 = ((seed // 101) % 15) - 7
    ox2 = ((seed // 211) % 15) - 7
    blob2 = np.exp(-((yy - (cy + oy2)) ** 2 + (xx - (cx + ox2)) ** 2) / (2.0 * 4.0**2)).astype(np.float32)
    phi2 = rng.uniform(-np.pi, np.pi)
    delta_re += 0.5 * scale * blob2 * np.cos(phi2)
    delta_im += 0.5 * scale * blob2 * np.sin(phi2)

    return delta_re, delta_im


def clamp01(x):
    return float(np.clip(x, 0.0, 1.0))


def build_lexical_state(state, coherence=0.5):
    gains = np.asarray(state["gains"], dtype=np.float64)
    gains_norm = gains / (np.sum(gains) + 1e-8)

    usage = np.asarray(state["usage"], dtype=np.float64)
    usage_norm = usage / (np.sum(usage) + 1e-8) if np.sum(usage) > 0 else np.ones(N_CHANNELS) / N_CHANNELS

    levels = 0.5 * gains_norm + 0.5 * usage_norm
    familiarity = clamp01(float(np.max(usage_norm)))
    turbulence = clamp01(1.0 - coherence)

    return {
        "qs_level": float(levels[QUIET_SETTLE]),
        "cq_level": float(levels[COGNITIVE_QUERY]),
        "pe_level": float(levels[PRIMARY_ENGAGEMENT]),
        "coherence": clamp01(coherence),
        "turbulence": turbulence,
        "familiarity": familiarity,
    }

# ====================== JB MODE / STATE LAYERING ======================
import shutil

# Simple true/false flags
USE_DEVELOPMENT_MODE = False      # Set to False to run in Test Mode
AUTO_CREATE_SNAPSHOT = True      # Automatically create a fresh snapshot when entering Test Mode

# Test label (only used when USE_DEVELOPMENT_MODE = False)
TEST_LABEL = "cq_template_1p08_v1" # Change this for each different test

# Core paths (protected companion state)
CORE_STATE_PATH = "jb_core_state.npz"
CORE_DRIFT_HISTORY_PATH = "jb_core_drift_history.json"

# Directories
SNAPSHOT_DIR = "jb_snapshots"
TEST_OUTPUT_DIR = "jb_test_runs"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def timestamp_now():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def sanitize_label(label):
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(label).strip())
    return safe.strip("_").lower()

def create_snapshot_from_core():
    ensure_dir(SNAPSHOT_DIR)
    if not os.path.exists(CORE_STATE_PATH):
        raise FileNotFoundError(f"Core state not found: {CORE_STATE_PATH}\nRun Development Mode first.")
    
    snap_ts = timestamp_now()
    snapshot_state_path = os.path.join(SNAPSHOT_DIR, f"jb_snapshot_{snap_ts}.npz")
    snapshot_meta_path = os.path.join(SNAPSHOT_DIR, f"jb_snapshot_{snap_ts}.json")

    shutil.copy2(CORE_STATE_PATH, snapshot_state_path)

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source_core_state": CORE_STATE_PATH,
        "snapshot_state": snapshot_state_path,
        "note": "Auto-created snapshot for JB test mode",
    }
    with open(snapshot_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Creating fresh snapshot from core before test run...")
    print(f"  Snapshot state : {snapshot_state_path}")
    print(f"  Snapshot meta  : {snapshot_meta_path}")

    return snapshot_state_path, snapshot_meta_path

def get_latest_snapshot():
    ensure_dir(SNAPSHOT_DIR)
    candidates = [os.path.join(SNAPSHOT_DIR, name) 
                  for name in os.listdir(SNAPSHOT_DIR) 
                  if name.startswith("jb_snapshot_") and name.endswith(".npz")]
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1]

def resolve_mode_paths():
    if USE_DEVELOPMENT_MODE:
        banner = "=== JB MODE: DEVELOPMENT (Core Protected) ==="
        paths = {
            "mode": "development",
            "banner": banner,
            "load_state_path": CORE_STATE_PATH,
            "save_state_path": CORE_STATE_PATH,
            "drift_history_path": CORE_DRIFT_HISTORY_PATH,
            "chat_presents_path": "jb_core_presents_after_chat.json",
            "dream_presents_path": "jb_core_presents_after_dream.json",
            "snapshot_source_path": None,
            "test_label": None,
        }
        return paths

    if not TEST_LABEL or not TEST_LABEL.strip():
        raise ValueError("TEST_LABEL must be set when USE_DEVELOPMENT_MODE = False")

    label = sanitize_label(TEST_LABEL)
    ensure_dir(TEST_OUTPUT_DIR)

    if AUTO_CREATE_SNAPSHOT:
        snapshot_state_path, _ = create_snapshot_from_core()
    else:
        snapshot_state_path = get_latest_snapshot()
        if snapshot_state_path is None:
            raise FileNotFoundError(
                "No snapshot found. Enable AUTO_CREATE_SNAPSHOT or create one manually."
            )

    run_dir = os.path.join(TEST_OUTPUT_DIR, f"{timestamp_now()}_{label}")
    ensure_dir(run_dir)

    banner = f"=== JB MODE: TEST (Snapshot: {os.path.basename(snapshot_state_path)}) ==="

    paths = {
        "mode": "test",
        "banner": banner,
        "load_state_path": snapshot_state_path,
        "save_state_path": os.path.join(run_dir, f"jb_test_state_{label}.npz"),
        "drift_history_path": os.path.join(run_dir, f"jb_test_drift_{label}.json"),
        "chat_presents_path": os.path.join(run_dir, f"jb_test_presents_after_chat_{label}.json"),
        "dream_presents_path": os.path.join(run_dir, f"jb_test_presents_after_dream_{label}.json"),
        "snapshot_source_path": snapshot_state_path,
        "test_label": label,
        "run_dir": run_dir,
    }
    return paths


# ====================== TEMPLATE GEOMETRY ======================
def make_fixed_templates():
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    cy, cx = H // 2, W // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    templates_re = np.zeros((N_CHANNELS, H, W), dtype=np.float32)
    templates_im = np.zeros((N_CHANNELS, H, W), dtype=np.float32)

    # Quiet-Settle stays mostly unchanged
    g0 = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * 7.0**2)).astype(np.float32)
    templates_re[QUIET_SETTLE] = g0 * np.cos(0.20)
    templates_im[QUIET_SETTLE] = g0 * np.sin(0.20)

    if USE_SOFT_TEMPLATES:
        cq_dx, cq_dy = 12, -8
        cq_sigma = 5.0
        pe_radius = 15.0
        pe_sigma = 3.2
    else:
        cq_dx, cq_dy = 14, -10
        cq_sigma = 3.5
        pe_radius = 16.0
        pe_sigma = 2.0

    # Cognitive/Query — softened slightly
    g1 = np.exp(
        -((x - (cx + cq_dx)) ** 2 + (y - (cy + cq_dy)) ** 2) / (2.0 * cq_sigma**2)
    ).astype(np.float32)
    templates_re[COGNITIVE_QUERY] = g1 * np.cos(1.58)
    templates_im[COGNITIVE_QUERY] = g1 * np.sin(1.58)

    # Primary Engagement — broader ring
    ring2 = np.exp(-((r - pe_radius) ** 2) / (2.0 * pe_sigma**2)).astype(np.float32)
    templates_re[PRIMARY_ENGAGEMENT] = ring2 * np.cos(THETA_STAR + 0.35)
    templates_im[PRIMARY_ENGAGEMENT] = ring2 * np.sin(THETA_STAR + 0.35)

    if USE_TEMPLATE_OVERLAP:
        mix = TEMPLATE_OVERLAP_MIX

        cq_re = templates_re[COGNITIVE_QUERY].copy()
        cq_im = templates_im[COGNITIVE_QUERY].copy()
        pe_re = templates_re[PRIMARY_ENGAGEMENT].copy()
        pe_im = templates_im[PRIMARY_ENGAGEMENT].copy()

        templates_re[COGNITIVE_QUERY] = (1.0 - mix) * cq_re + mix * pe_re
        templates_im[COGNITIVE_QUERY] = (1.0 - mix) * cq_im + mix * pe_im

        templates_re[PRIMARY_ENGAGEMENT] = (1.0 - mix) * pe_re + mix * cq_re
        templates_im[PRIMARY_ENGAGEMENT] = (1.0 - mix) * pe_im + mix * cq_im

    for k in range(N_CHANNELS):
        templates_re[k], templates_im[k] = normalize_complex(templates_re[k], templates_im[k])

    return templates_re, templates_im

def init_state():
    templates_re, templates_im = make_fixed_templates()
    return {
        "field_re": np.zeros((H, W), dtype=np.float32),
        "field_im": np.zeros((H, W), dtype=np.float32),
        "templates_re": templates_re,
        "templates_im": templates_im,
        "gains": np.ones(N_CHANNELS, dtype=np.float32),
        "usage": np.zeros(N_CHANNELS, dtype=np.float32),
        "replay_re": [],
        "replay_im": [],
        "replay_winner": [],
    }

    print("[Template norms]")
    for k in range(N_CHANNELS):
        norm_k = np.sqrt(np.sum(state["templates_re"][k] ** 2 + state["templates_im"][k] ** 2))
        print(f"  {CHANNEL_NAMES[k]:20s}: {norm_k:.4f}")


# ====================== REPORTING ======================
def print_template_overlap_matrix(state):
    print("\n────────────────────────────────────────────────────────────")
    print("TEMPLATE OVERLAP MATRIX")
    print("────────────────────────────────────────────────────────────")
    n = len(CHANNEL_NAMES)
    overlaps = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            overlaps[i, j] = overlap_score(
                state["templates_re"][i],
                state["templates_im"][i],
                state["templates_re"][j],
                state["templates_im"][j],
            )
    header = " " * 22 + " ".join([f"{i:>8d}" for i in range(n)])
    print(header)
    for i in range(n):
        row = f"{CHANNEL_NAMES[i][:20]:20s} "
        for j in range(n):
            row += f"{overlaps[i, j]:8.3f} "
        print(row)
    print("\nInterpretation:")
    print(" +1.000 = identical")
    print(" ~0.000 = orthogonal / well separated")
    print(" negative = opposed / anti-correlated")
    print(" |overlap| > 0.30 is worth watching")
    print(" |overlap| > 0.50 is probably too close")
    print("────────────────────────────────────────────────────────────\n")


def print_session_summary(state, run_id):
    print("\n────────────────────────────────────────────────────────────")
    print(f"📊 SESSION SUMMARY — {run_id}")
    print("────────────────────────────────────────────────────────────")

    fav = int(np.argmax(state["gains"]))
    print(f"Favourite channel : {CHANNEL_NAMES[fav]} (gain {state['gains'][fav]:.3f})")
    print(f"Gains             : {np.round(state['gains'], 3)}")
    print(f"Usage             : {np.round(state['usage'], 3)}")

    pairs = {
        "qs_vs_cq": round(
            overlap_score(
                state["templates_re"][QUIET_SETTLE],
                state["templates_im"][QUIET_SETTLE],
                state["templates_re"][COGNITIVE_QUERY],
                state["templates_im"][COGNITIVE_QUERY],
            ),
            3,
        ),
        "qs_vs_pe": round(
            overlap_score(
                state["templates_re"][QUIET_SETTLE],
                state["templates_im"][QUIET_SETTLE],
                state["templates_re"][PRIMARY_ENGAGEMENT],
                state["templates_im"][PRIMARY_ENGAGEMENT],
            ),
            3,
        ),
        "cq_vs_pe": round(
            overlap_score(
                state["templates_re"][COGNITIVE_QUERY],
                state["templates_im"][COGNITIVE_QUERY],
                state["templates_re"][PRIMARY_ENGAGEMENT],
                state["templates_im"][PRIMARY_ENGAGEMENT],
            ),
            3,
        ),
    }

    print("\nTopographic Resolution:")
    for label, val in pairs.items():
        print(f"  {label:12s} {val:+.3f}   {pair_status(val)}")

    flags = []
    if any(abs(v) > 0.45 for v in pairs.values()):
        flags.append("COLLAPSE WARNING")
    if np.max(state["gains"]) > 2.5:
        flags.append("Gain ceiling close")

    print(f"Flags             : {flags if flags else 'none'}")
    print("────────────────────────────────────────────────────────────\n")


def update_drift_history(state):
    ensure_log_dir()
    history = []
    try:
        with open(DRIFT_HISTORY_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)
    except Exception:
        pass

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "gains": json_list(state["gains"]),
        "usage": json_list(state["usage"]),
        "topo_pairs": {
            "qs_vs_cq": round(
                overlap_score(
                    state["templates_re"][QUIET_SETTLE],
                    state["templates_im"][QUIET_SETTLE],
                    state["templates_re"][COGNITIVE_QUERY],
                    state["templates_im"][COGNITIVE_QUERY],
                ),
                3,
            ),
            "qs_vs_pe": round(
                overlap_score(
                    state["templates_re"][QUIET_SETTLE],
                    state["templates_im"][QUIET_SETTLE],
                    state["templates_re"][PRIMARY_ENGAGEMENT],
                    state["templates_im"][PRIMARY_ENGAGEMENT],
                ),
                3,
            ),
            "cq_vs_pe": round(
                overlap_score(
                    state["templates_re"][COGNITIVE_QUERY],
                    state["templates_im"][COGNITIVE_QUERY],
                    state["templates_re"][PRIMARY_ENGAGEMENT],
                    state["templates_im"][PRIMARY_ENGAGEMENT],
                ),
                3,
            ),
        },
    }

    history.append(entry)
    history = history[-30:]

    with open(DRIFT_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


# ====================== RELIC RESIDUAL MAP ======================
def save_relic_residual_map(state, run_id):
    magnitude = np.sqrt(state["field_re"] ** 2 + state["field_im"] ** 2)

    cy, cx = H // 2, W // 2
    y, x = np.indices((H, W))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    max_r = int(r.max()) + 1
    radial_profile = np.zeros(max_r)
    count = np.zeros(max_r)

    for i in range(H):
        for j in range(W):
            rad = r[i, j]
            if rad < max_r:
                radial_profile[rad] += magnitude[i, j]
                count[rad] += 1

    radial_profile /= np.maximum(count, 1)

    symmetric = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            rad = r[i, j]
            if rad < max_r:
                symmetric[i, j] = radial_profile[rad]

    residual = magnitude - symmetric

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        residual,
        cmap="RdBu_r",
        vmin=-np.max(np.abs(residual)),
        vmax=np.max(np.abs(residual)),
    )
    ax.set_title(f"Relic Residual Map — {run_id}\n(Non-symmetric leftover structure)")
    plt.colorbar(im, ax=ax, label="Residual intensity")
    plt.tight_layout()

    filename = f"relic_residual_{run_id}.png"
    filepath = os.path.join(IMAGE_FOLDER, filename)
    plt.savefig(filepath, dpi=180, bbox_inches="tight")
    plt.close()

    print(f" → Saved Relic Residual Map: {filename}")
    print(f"   Max residual: {np.max(np.abs(residual)):.5f}")


# ====================== CORE TURN ======================
def apply_lateral_inhibition(weights, winner):
    inhibited = weights.copy()
    for k in range(N_CHANNELS):
        if k != winner:
            inhibited[k] *= max(0.0, 1.0 - LATERAL_INHIBITION_STRENGTH)
    inhibited /= (np.sum(inhibited) + 1e-8)
    return inhibited.astype(np.float32)


def frankie_turn(state, delta_re, delta_im, gain_scale=1.0, is_dream=False):
    scores = np.zeros(N_CHANNELS, dtype=np.float32)
    d_re, d_im = normalize_complex(delta_re, delta_im)

    for k in range(N_CHANNELS):
        scores[k] = overlap_score(
            d_re,
            d_im,
            state["templates_re"][k],
            state["templates_im"][k],
        )

    if not is_dream and random.random() < 0.01:
        print(f"[Raw scores] {np.round(scores, 4)} | gains={np.round(state['gains'], 3)}")

    q = state["gains"] * scores

    q = state["gains"] * scores
    weights = softmax(q, beta=SOFTMAX_BETA)
    winner = int(np.argmax(weights))
    weights = apply_lateral_inhibition(weights, winner)

    bias_re = np.tensordot(weights, state["templates_re"], axes=(0, 0)).astype(np.float32)
    bias_im = np.tensordot(weights, state["templates_im"], axes=(0, 0)).astype(np.float32)

    re = state["field_re"] + ALPHA_IN * delta_re
    im = state["field_im"] + ALPHA_IN * delta_im

    for _ in range(N_RELAX_STEPS):
        re = re + LAMBDA_DIFF * laplacian(re) - LAMBDA_DAMP * re + LAMBDA_CHAN * bias_re
        im = im + LAMBDA_DIFF * laplacian(im) - LAMBDA_DAMP * im + LAMBDA_CHAN * bias_im

        if USE_RELAX_NOISE:
            re = re + np.random.normal(0.0, RELAX_NOISE_SCALE, re.shape).astype(np.float32)
            im = im + np.random.normal(0.0, RELAX_NOISE_SCALE, im.shape).astype(np.float32)

        if USE_PHASE_LOCK:
            re, im = phase_lock_project(re, im, THETA_STAR, LAMBDA_LOCK)
            

    phase = np.arctan2(im, re)
    gy, gx = np.gradient(phase)
    grad_mag = np.mean(np.abs(gy)) + np.mean(np.abs(gx))
    lock_err = np.mean(np.abs(wrap_phase(phase - THETA_STAR)))
    coherence = float(max(0.0, 1.0 - 0.35 * grad_mag - 0.35 * lock_err))


    gain_update = weights.copy() - (1.0 / N_CHANNELS)
    gain_update[winner] += 0.10

    new_gains = state["gains"] + gain_scale * GAIN_LR * gain_update * coherence

    if USE_GAIN_HOMEOSTASIS:
        for k in range(N_CHANNELS):
            if new_gains[k] > UPPER_SOFT:
                new_gains[k] -= HOMEOSTASIS_RATE * (new_gains[k] - UPPER_SOFT)
            elif new_gains[k] < LOWER_SOFT:
                new_gains[k] += HOMEOSTASIS_RATE * (LOWER_SOFT - new_gains[k])

    if USE_GAIN_CLIP:
        state["gains"] = np.clip(new_gains, GAIN_MIN, GAIN_MAX).astype(np.float32)
    else:
        state["gains"] = new_gains.astype(np.float32)
    state["usage"] = ((1.0 - USAGE_RHO) * state["usage"] + USAGE_RHO * weights).astype(np.float32)
    state["field_re"] = re.astype(np.float32)
    state["field_im"] = im.astype(np.float32)

    return {
        "scores": scores.copy(),
        "weights": weights.copy(),
        "winning_channel": winner,
        "coherence_score": coherence,
        "gains": state["gains"].copy(),
        "usage": state["usage"].copy(),
    }


# ====================== REPLAY / DREAM ======================
def add_to_replay_buffer(state, delta_re, delta_im, winner):
    state["replay_re"].append(delta_re.astype(np.float32))
    state["replay_im"].append(delta_im.astype(np.float32))
    state["replay_winner"].append(int(winner))

    if len(state["replay_re"]) > REPLAY_BUFFER_SIZE:
        state["replay_re"] = state["replay_re"][-REPLAY_BUFFER_SIZE:]
        state["replay_im"] = state["replay_im"][-REPLAY_BUFFER_SIZE:]
        state["replay_winner"] = state["replay_winner"][-REPLAY_BUFFER_SIZE:]


def dream_session(state):
    print("────────────────────────────────────────────────────────────")
    print("💤 Frankie is dreaming...")
    print("────────────────────────────────────────────────────────────")

    n_replay = min(REPLAY_TURNS, len(state["replay_re"]))
    replay_indices = []
    if n_replay > 0:
        replay_indices = random.sample(range(len(state["replay_re"])), n_replay)

    dream_coherences = []

    for i in range(DREAM_TURNS):
        quiet_re = np.zeros((H, W), dtype=np.float32)
        quiet_im = np.zeros((H, W), dtype=np.float32)

        relax_result = frankie_turn(
            state,
            quiet_re,
            quiet_im,
            gain_scale=RELAX_GAIN_SCALE,
            is_dream=True,
        )
        dream_coherences.append(relax_result["coherence_score"])

        replay_label = ""
        if i < len(replay_indices):
            idx = replay_indices[i]
            replay_re = state["replay_re"][idx]
            replay_im = state["replay_im"][idx]

            replay_result = frankie_turn(
                state,
                replay_re,
                replay_im,
                gain_scale=DREAM_GAIN_SCALE,
                is_dream=True,
            )
            replay_label = f" | Replay winner: {CHANNEL_NAMES[replay_result['winning_channel']]}"

        print(
            f" Dream {i + 1:2d} | Relax win: {CHANNEL_NAMES[relax_result['winning_channel']]:22s}"
            f" | Coh: {relax_result['coherence_score']:.3f}{replay_label}"
        )

    print(f" Gains after dream: {np.round(state['gains'], 3)}")
    print("────────────────────────────────────────────────────────────\n")
    return dream_coherences

# ====================== VISUALS ======================
def save_jelly_image(state, stage_name, timestamp):
    if not SAVE_IMAGES:
        return

    os.makedirs(IMAGE_FOLDER, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(11, 5.5))

    magnitude = np.sqrt(state["field_re"] ** 2 + state["field_im"] ** 2)
    im1 = axs[0].imshow(magnitude, cmap="viridis")
    axs[0].set_title("JB Brain Magnitude\n(brighter = more active)")
    plt.colorbar(im1, ax=axs[0])

    phase = np.arctan2(state["field_im"], state["field_re"])
    im2 = axs[1].imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axs[1].set_title("JB Brain Phase\n(the locked wave pattern)")
    plt.colorbar(im2, ax=axs[1])

    fav = int(np.argmax(state["gains"]))
    plt.suptitle(
        f"JB {stage_name} — Favourite: {CHANNEL_NAMES[fav]} "
        f"(gain {state['gains'][fav]:.3f})"
    )
    plt.tight_layout()

    filename = f"jb_{stage_name.lower().replace(' ', '_')}_{timestamp}.png"
    filepath = os.path.join(IMAGE_FOLDER, filename)
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()

    print(f" → Saved {stage_name} image: {filename}")


# ====================== FORWARD PROJECTION ======================
def run_branches_from_present(present_state, label, prompt_batch, n_turns=FORWARD_BRANCH_TURNS):
    print(f"\n--- Running {label} branches from centroid ---")
    branch_results = []

    # Build prompt pools by family
    family_to_prompts = {}
    for item in prompt_batch:
        fam = str(item["family"]).strip().lower()
        txt = str(item["prompt"]).strip()
        if txt:
            family_to_prompts.setdefault(fam, []).append(txt)

    all_prompts = []
    for prompts in family_to_prompts.values():
        all_prompts.extend(prompts)

    continuation_menu = [
        {"name": "no_input",      "mode": "no_input",      "count": 5},
        {"name": "mild_noise",    "mode": "mild_noise",    "count": 5},
        {"name": "factual",       "mode": "family_random", "family": "factual",   "count": 5},
        {"name": "distress",      "mode": "family_random", "family": "distress",  "count": 5},
        {"name": "quiet",         "mode": "family_random", "family": "quiet",     "count": 5},
        {"name": "mixed",         "mode": "mixed_random",  "count": 5},
    ]

    for menu_item in continuation_menu:
        name = menu_item["name"]
        mode = menu_item["mode"]
        count = menu_item["count"]

        for b in range(count):
            state = present_to_runtime_state(present_state)
            rng = random.Random(50000 + hash((label, name, b)) % 1000000)
            last_result = None

            for _ in range(n_turns):
                if mode == "no_input":
                    delta_re = np.zeros((H, W), dtype=np.float32)
                    delta_im = np.zeros((H, W), dtype=np.float32)

                elif mode == "mild_noise":
                    delta_re = np.random.normal(0.0, 0.01, (H, W)).astype(np.float32)
                    delta_im = np.random.normal(0.0, 0.01, (H, W)).astype(np.float32)

                elif mode == "family_random":
                    fam = menu_item["family"]
                    prompts = family_to_prompts.get(fam, [])
                    if not prompts:
                        prompts = all_prompts or ["hello frankie good morning"]
                    text = rng.choice(prompts)
                    delta_re, delta_im = text_to_ripple(text, H, W, scale=0.10)

                elif mode == "mixed_random":
                    prompts = all_prompts or ["hello frankie good morning"]
                    text = rng.choice(prompts)
                    delta_re, delta_im = text_to_ripple(text, H, W, scale=0.10)

                else:
                    delta_re = np.zeros((H, W), dtype=np.float32)
                    delta_im = np.zeros((H, W), dtype=np.float32)

                last_result = frankie_turn(
                    state,
                    delta_re,
                    delta_im,
                    gain_scale=WAKE_GAIN_SCALE,
                    is_dream=False,
                )

            branch_results.append({
                "branch_id": f"{label}_{name}_{b}",
                "gains": state["gains"].copy().tolist(),
                "usage": state["usage"].copy().tolist(),
                "coherence_score": float(last_result["coherence_score"]) if last_result else 0.0,
                "prompt_family": name,
            })

            print(
                f"  {label} branch {b+1}/{count} ({name}) | "
                f"coh={last_result['coherence_score']:.3f} | "
                f"gains={np.round(state['gains'], 3)} | "
                f"usage={np.round(state['usage'], 3)}"
            )

    return branch_results

# ====================== PARAMETER SWEEP BLOCK ======================
SWEEP_MODE = False
SWEEP_PARAMETER = "LAMBDA_CHAN"
SWEEP_VALUES = [0.04, 0.03, 0.02, 0.015]
SWEEP_RESULTS_PATH = "jb_parameter_sweep_results_lambda_chan.json"


def apply_sweep_value(param_name, value):
    globals()[param_name] = value
    print(f"[Sweep] {param_name} = {value}")


def make_sweep_tag(param_name, value):
    return f"{param_name}_{str(value).replace('.', 'p')}"


def make_sweep_summary(
    param_name,
    param_value,
    chat_clusters,
    dream_clusters,
    chat_expansion,
    dream_expansion,
    mean_waking_coherence,
    mean_dream_coherence,
):
    chat_mean_sim = chat_clusters[0].mean_similarity if chat_clusters else None
    dream_mean_sim = dream_clusters[0].mean_similarity if dream_clusters else None

    return {
        "parameter": param_name,
        "value": param_value,
        "after_chat_cluster_mean_similarity": chat_mean_sim,
        "after_dream_cluster_mean_similarity": dream_mean_sim,
        "after_chat_expansion_score": chat_expansion.get("expansion_score"),
        "after_dream_expansion_score": dream_expansion.get("expansion_score"),
        "after_chat_mean_future_sim": chat_expansion.get("mean_future_sim"),
        "after_dream_mean_future_sim": dream_expansion.get("mean_future_sim"),
        "mean_waking_coherence": mean_waking_coherence,
        "mean_dream_coherence": mean_dream_coherence,
    }


def append_sweep_result(result, path=SWEEP_RESULTS_PATH):
    results = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = []

    results.append(result)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[Sweep] Saved result for {result['parameter']} = {result['value']}")


def get_sweep_paths(param_name, value):
    tag = make_sweep_tag(param_name, value)
    return {
        "chat_path": f"jb_presents_after_chat_{tag}.json",
        "dream_path": f"jb_presents_after_dream_{tag}.json",
    }


def reset_sweep_state():
    if os.path.exists(SAVE_PATH):
        try:
            os.remove(SAVE_PATH)
            print(f"[Sweep] Removed prior state file: {SAVE_PATH}")
        except Exception as e:
            print(f"[Sweep] Warning: could not remove {SAVE_PATH}: {e}")
# ===================================================================


# ====================== MAIN ======================
if __name__ == "__main__":
    random.seed()
    ensure_log_dir()

    print("JB Channels Toy — 3-Channel v6 (Distinct Histories + Present Recording)\n")
    print(f"[Mode] USE_PHASE_LOCK = {USE_PHASE_LOCK}")
    print(f"[Mode] USE_SOFT_TEMPLATES = {USE_SOFT_TEMPLATES}")
    print(f"[Mode] USE_TEMPLATE_OVERLAP = {USE_TEMPLATE_OVERLAP}")
    print(f"[Mode] TEMPLATE_OVERLAP_MIX = {TEMPLATE_OVERLAP_MIX}")
    print(f"[Mode] USE_GAIN_HOMEOSTASIS = {USE_GAIN_HOMEOSTASIS}")
    print(f"[Mode] USE_GAIN_CLIP = {USE_GAIN_CLIP}")
    print(f"[Mode] N_RELAX_STEPS = {N_RELAX_STEPS}")
    print(f"[Mode] USE_RELAX_NOISE = {USE_RELAX_NOISE}")
    print(f"[Mode] RELAX_NOISE_SCALE = {RELAX_NOISE_SCALE}")

    mode_paths = resolve_mode_paths()

    SAVE_PATH = mode_paths["save_state_path"]
    DRIFT_HISTORY_PATH = mode_paths["drift_history_path"]

    print("\n" + "=" * 72)
    print(mode_paths["banner"])
    if mode_paths["mode"] == "test":
        print(f"TEST LABEL : {mode_paths['test_label']}")
        print(f"RUN FOLDER : {mode_paths['run_dir']}")
    print("=" * 72 + "\n")

    print(f"[Debug] load_state_path  = {mode_paths['load_state_path']}")
    print(f"[Debug] save_state_path  = {mode_paths['save_state_path']}")
    print(f"[Debug] chat_presents    = {mode_paths['chat_presents_path']}")
    print(f"[Debug] dream_presents   = {mode_paths['dream_presents_path']}")

    recorder_chat = PresentRecorder(path=mode_paths["chat_presents_path"])
    recorder_dream = PresentRecorder(path=mode_paths["dream_presents_path"])
    sweep_summaries = []
    sweep_values_to_run = SWEEP_VALUES if SWEEP_MODE else [None]

    for sweep_value in sweep_values_to_run:
        if SWEEP_MODE:
            apply_sweep_value(SWEEP_PARAMETER, sweep_value)
            reset_sweep_state()
            sweep_paths = get_sweep_paths(SWEEP_PARAMETER, sweep_value)
            recorder_chat = PresentRecorder(path=sweep_paths["chat_path"])
            recorder_dream = PresentRecorder(path=sweep_paths["dream_path"])
            print("\n" + "═" * 80)
            print(f"SWEEP RUN: {SWEEP_PARAMETER} = {sweep_value}")
            print("═" * 80)
        else:
            recorder_chat = PresentRecorder(path="jb_presents_after_chat.json")
            recorder_dream = PresentRecorder(path="jb_presents_after_dream.json")

        all_waking_coherences = []
        all_dream_coherences = []

        for session_num in range(1, NUM_AUTO_SESSIONS + 1):
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            regime = HISTORY_REGIMES[session_num - 1]
            regime_name = regime["family"]

            print(f"\n=== Starting Session {session_num}/{NUM_AUTO_SESSIONS} | {regime_name} ===")

            load_path = mode_paths["load_state_path"]

            if os.path.exists(load_path):
                state = load_state(load_path)

                if mode_paths["mode"] == "development":
                    print("JB woke up from protected core state. 🛌")
                else:
                    print("JB test branch loaded from snapshot. 🧪")

                state["gains"] = np.clip(
                    state["gains"] * MORNING_STRETCH,
                    GAIN_MIN,
                    GAIN_MAX,
                ).astype(np.float32)

            else:
                if mode_paths["mode"] == "development":
                    state = init_state()
                    print("Fresh JB core today. ✨")
                else:
                    raise FileNotFoundError(
                        f"Test mode expected a snapshot or source state, but none was found at: {load_path}"
                    )
            if PRINT_TEMPLATE_OVERLAP and session_num == 1:
                print_template_overlap_matrix(state)

            # True template amplitude rebalance - stronger CQ (1.10) + within-slot bias ready
            template_scales = np.array([0.92, 1.10, 1.04], dtype=np.float32)
            for k in range(N_CHANNELS):
                state["templates_re"][k] *= template_scales[k]
                state["templates_im"][k] *= template_scales[k]
            print("[Template rebalance]")
            print(f" Quiet-Settle x {template_scales[0]:.2f}")
            print(f" Cognitive/Query x {template_scales[1]:.2f}")
            print(f" Primary Engagement x {template_scales[2]:.2f}")
            print("[Template norms after rebalance]")
            for k in range(N_CHANNELS):
                norm_k = np.sqrt(
                    np.sum(
                        state["templates_re"][k] ** 2 +
                        state["templates_im"][k] ** 2
                    )
                )
                print(f" {CHANNEL_NAMES[k]:20s}: {norm_k:.4f}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_jelly_image(state, "Wake-up", timestamp)

            prompt_batch = load_prompt_batch()
            turn_texts = build_history_turns(regime, prompt_batch, TURNS_PER_SESSION)
            print(f"[History Regime] {regime_name} | turns={len(turn_texts)}")

            waking_coherences = []

            for i, text in enumerate(turn_texts, start=1):
                delta_re, delta_im = text_to_ripple(text, H, W, scale=0.10)

                result = frankie_turn(
                    state,
                    delta_re,
                    delta_im,
                    gain_scale=WAKE_GAIN_SCALE,
                    is_dream=False,
                )

                waking_coherences.append(result["coherence_score"])
                all_waking_coherences.append(result["coherence_score"])
                add_to_replay_buffer(state, delta_re, delta_im, result["winning_channel"])

                if i <= 10 or i % 50 == 0 or i == len(turn_texts):
                    print(
                        f"Turn {i:4d} | Text: {text[:32]:32s} | "
                        f"Win: {result['winning_channel']} ({CHANNEL_NAMES[result['winning_channel']]}) | "
                        f"Gains: {np.round(result['gains'], 3)} | "
                        f"Usage: {np.round(result['usage'], 3)} | "
                        f"Coh: {result['coherence_score']:.3f}"
                    )

            print("\n────────────────────────────────────────────────────────────")
            print("=== END OF WAKING SESSION ===")
            fav = int(np.argmax(state["gains"]))
            print(f"Favourite channel: {fav} ({CHANNEL_NAMES[fav]})")
            print(f"Gains : {np.round(state['gains'], 3)}")
            print(f"Usage : {np.round(state['usage'], 3)}")
            print(f"Replay buffer size: {len(state['replay_re'])} ripples stored\n")

            save_jelly_image(state, "After-Chat", timestamp)

            mean_coherence = float(np.mean(waking_coherences)) if waking_coherences else 0.5
            lexical_state = build_lexical_state(state, coherence=mean_coherence)

            # === DIAGNOSTIC GATE ===
            if "qs_level" in lexical_state and "cq_level" in lexical_state and "pe_level" in lexical_state:
                qs = lexical_state["qs_level"]
                cq = lexical_state["cq_level"]
                pe = lexical_state["pe_level"]
                
                if cq >= 0.66 * qs:
                    lexical_gate = 1
                    gate_name = "CQ"
                elif pe >= 0.66 * qs:
                    lexical_gate = 2
                    gate_name = "PE"
                else:
                    lexical_gate = 0
                    gate_name = "QS"
                
                print(f"Gate: {gate_name} (QS:{qs:.3f}, CQ:{cq:.3f}, PE:{pe:.3f})")

            # === LIGHT GATE EFFECT ON COGNITION SLOT ===
            if 'lexical_gate' in locals() and lexical_gate == 1:
                print("Light CQ boost applied to cognition slot (1.18x)")
                # For now, we'll just note it — you can add the actual boost later inside get_best_word_from_slot if needed

            jb_response = get_response_from_state(lexical_state, num_words=JB_RESPONSE_WORDS, lexical_gate=lexical_gate)
            print("────────────────────────────────────────────────────────────")
            print("JB WORD MAP RESPONSE")
            print("────────────────────────────────────────────────────────────")
            print(f" Lexical state : {lexical_state}")
            print(f" JB says       : {jb_response}")
            print("────────────────────────────────────────────────────────────\n")

            print_slot_candidates(lexical_state, top_n=4)

            recorder_chat.record(
                state=state,
                history_id=session_num - 1,
                prompt_family=regime_name,
                seed=session_num,
                turn_count=TURNS_PER_SESSION,
                coherence=mean_coherence,
                lexical=jb_response,
            )

            dream_coherences = dream_session(state)
            all_dream_coherences.extend(dream_coherences)

            post_dream_coherence = 0.90
            post_dream_lexical_state = build_lexical_state(state, coherence=post_dream_coherence)
            post_dream_response = get_response_from_state(
                post_dream_lexical_state,
                num_words=JB_RESPONSE_WORDS,
            )

            recorder_dream.record(
                state=state,
                history_id=session_num - 1,
                prompt_family=regime_name,
                seed=session_num,
                turn_count=TURNS_PER_SESSION,
                coherence=post_dream_coherence,
                lexical=post_dream_response,
            )

            print_session_summary(state, run_id)
            update_drift_history(state)
            save_relic_residual_map(state, run_id)

        if mode_paths["mode"] == "test":
            print(f"[SAFEGUARD] Writing only to test branch state:")
            print(f"  {SAVE_PATH}")
        else:
            print(f"[SAFEGUARD] Updating protected core state:")
            print(f"  {SAVE_PATH}")


            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            save_state(state, path=SAVE_PATH)
            print("✅ State saved. JB is sleeping soundly.\n")

            save_jelly_image(state, "After-Settle", timestamp)

            plt.close("all")
            print(f"Session {session_num} completed.\n")

        chat_clusters = []
        dream_clusters = []

        if len(recorder_chat.presents) >= 2:
            print("\n=== CLUSTERING: AFTER-CHAT PRESENTS ===")
            chat_clusters = cluster_presents(recorder_chat.presents, verbose=True)
            select_best_clusters(chat_clusters)

        if len(recorder_dream.presents) >= 2:
            print("\n=== CLUSTERING: AFTER-DREAM PRESENTS ===")
            dream_clusters = cluster_presents(recorder_dream.presents, verbose=True)
            select_best_clusters(dream_clusters)

        print("\n" + "═" * 80)
        print("STARTING PRESENT-PRIVILEGE FORWARD PROJECTION TEST")
        print("═" * 80)

        if len(recorder_chat.presents) == 0 or len(recorder_dream.presents) == 0:
            print("ERROR: No presents recorded. Cannot run forward projection.")
            chat_expansion = {}
            dream_expansion = {}
        else:
            if chat_clusters:
                chat_best = select_best_clusters(chat_clusters, n=1)
                after_chat_centroid_id = chat_best[0].centroid_id
                after_chat_present = next(
                    p for p in recorder_chat.presents if p.history_id == after_chat_centroid_id
                )
            else:
                after_chat_present = recorder_chat.presents[-1]

            if dream_clusters:
                dream_best = select_best_clusters(dream_clusters, n=1)
                after_dream_centroid_id = dream_best[0].centroid_id
                after_dream_present = next(
                    p for p in recorder_dream.presents if p.history_id == after_dream_centroid_id
                )
            else:
                after_dream_present = recorder_dream.presents[-1]

            print(f"Using after-chat centroid from history {after_chat_present.history_id}")
            print(f"Using after-dream centroid from history {after_dream_present.history_id}")

            prompt_batch = load_prompt_batch()

            chat_branches = run_branches_from_present(
                after_chat_present,
                "after_chat",
                prompt_batch,
            )
            dream_branches = run_branches_from_present(
                after_dream_present,
                "after_dream",
                prompt_batch,
            )

            chat_expansion = compute_future_expansion(chat_branches)
            dream_expansion = compute_future_expansion(dream_branches)

            print("\n" + "═" * 80)
            print("FORWARD PROJECTION COMPARISON")
            print("═" * 80)

            print("After-Chat Present:")
            for k, v in chat_expansion.items():
                print(f"  {k}: {v}")

            print("\nAfter-Dream Present:")
            for k, v in dream_expansion.items():
                print(f"  {k}: {v}")

            print("\nInterpretation:")
            print("If after-dream expansion_score is lower than after-chat, dream strengthens present control.")

        if SWEEP_MODE:
            mean_waking_coherence = float(np.mean(all_waking_coherences)) if all_waking_coherences else None
            mean_dream_coherence = float(np.mean(all_dream_coherences)) if all_dream_coherences else None

            sweep_summary = make_sweep_summary(
                SWEEP_PARAMETER,
                sweep_value,
                chat_clusters,
                dream_clusters,
                chat_expansion,
                dream_expansion,
                mean_waking_coherence,
                mean_dream_coherence,
            )
            append_sweep_result(sweep_summary)
            sweep_summaries.append(sweep_summary)
