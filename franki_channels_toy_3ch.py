import os
import json
import hashlib
import random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# FRANKIE CHANNELS TOY — 3-CHANNEL v4 + RELIC RESIDUAL MAP
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
    "Primary Engagement"
]

# ---------- Core field dynamics ----------
ALPHA_IN = 0.12
LAMBDA_DIFF = 0.18
LAMBDA_DAMP = 0.03
LAMBDA_LOCK = 0.10
LAMBDA_CHAN = 0.04
SOFTMAX_BETA = 3.0
THETA_STAR = np.deg2rad(255.0)
N_RELAX_STEPS = 6

# ---------- Gain / usage ----------
GAIN_LR = 0.018
USAGE_RHO = 0.05
GAIN_MIN = 0.70
GAIN_MAX = 3.00

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
BATCH_TURNS = 24

# ---------- Persistence ----------
SAVE_PATH = "frankie_state_3ch.npz"
DRIFT_HISTORY_PATH = "frankie_drift_history_3ch.json"
IMAGE_FOLDER = r"C:\Users\Peter\Desktop\frankie_images"

# ---------- Logging ----------
LOG_DIR = "frankie_logs"

# ---------- Auto-run ----------
NUM_AUTO_SESSIONS = 5
TURNS_PER_SESSION = 50

# ---------- Lexical layer ----------
JB_RESPONSE_WORDS = 3

# ---------- Reporting ----------
SAVE_IMAGES = True
PRINT_TEMPLATE_OVERLAP = True

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

def print_session_summary(state, run_id):
    print("\n────────────────────────────────────────────────────────────")
    print(f"📊 SESSION SUMMARY — {run_id}")
    print("────────────────────────────────────────────────────────────")
    
    fav = int(np.argmax(state["gains"]))
    print(f"Favourite channel : {CHANNEL_NAMES[fav]} (gain {state['gains'][fav]:.3f})")
    print(f"Gains             : {np.round(state['gains'], 3)}")
    print(f"Usage             : {np.round(state['usage'], 3)}")
    
    pairs = {
        "qs_vs_cq": round(overlap_score(state["templates_re"][QUIET_SETTLE], state["templates_im"][QUIET_SETTLE],
                                        state["templates_re"][COGNITIVE_QUERY], state["templates_im"][COGNITIVE_QUERY]), 3),
        "qs_vs_pe": round(overlap_score(state["templates_re"][QUIET_SETTLE], state["templates_im"][QUIET_SETTLE],
                                        state["templates_re"][PRIMARY_ENGAGEMENT], state["templates_im"][PRIMARY_ENGAGEMENT]), 3),
        "cq_vs_pe": round(overlap_score(state["templates_re"][COGNITIVE_QUERY], state["templates_im"][COGNITIVE_QUERY],
                                        state["templates_re"][PRIMARY_ENGAGEMENT], state["templates_im"][PRIMARY_ENGAGEMENT]), 3),
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
    except:
        pass
    
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "gains": json_list(state["gains"]),
        "usage": json_list(state["usage"]),
        "topo_pairs": {
            "qs_vs_cq": round(overlap_score(state["templates_re"][QUIET_SETTLE], state["templates_im"][QUIET_SETTLE],
                                            state["templates_re"][COGNITIVE_QUERY], state["templates_im"][COGNITIVE_QUERY]), 3),
            "qs_vs_pe": round(overlap_score(state["templates_re"][QUIET_SETTLE], state["templates_im"][QUIET_SETTLE],
                                            state["templates_re"][PRIMARY_ENGAGEMENT], state["templates_im"][PRIMARY_ENGAGEMENT]), 3),
            "cq_vs_pe": round(overlap_score(state["templates_re"][COGNITIVE_QUERY], state["templates_im"][COGNITIVE_QUERY],
                                            state["templates_re"][PRIMARY_ENGAGEMENT], state["templates_im"][PRIMARY_ENGAGEMENT]), 3),
        }
    }
    history.append(entry)
    history = history[-30:]
    
    with open(DRIFT_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

# ====================== JB WORD MAP (Starter Version) ======================

# Core pool of ~100 everyday words, grouped into overlapping layers
WORD_POOL = {
    "core_state": ["calm", "steady", "sharp", "bright", "blurred", "heavy", "light", "open", "closed", "quiet"],
    "relation": ["near", "far", "with", "apart", "inside", "between", "around", "toward", "away"],
    "certainty": ["yes", "no", "maybe", "unsure", "clear", "certain", "perhaps", "likely", "doubt"],
    "memory_familiarity": ["known", "old", "familiar", "new", "remembered", "lost", "fresh", "distant"],
    "orientation": ["here", "there", "now", "before", "after", "around", "forward", "back"],
    "feeling_tone": ["warm", "cold", "kind", "tense", "playful", "sad", "gentle", "strong", "soft"],
    "action_intention": ["reach", "listen", "hold", "answer", "ask", "stay", "move", "rest", "think"],
    "link_connective": ["this", "that", "feels", "is", "becoming", "still", "and", "but", "yet"]
}

# Flatten into a single list for scoring
ALL_WORDS = [word for group in WORD_POOL.values() for word in group]

# Weight vector for each word: [qs_affinity, cq_affinity, pe_affinity, coherence_pref, turbulence_pref, familiarity_pref]
WORD_WEIGHTS = {
    "calm": [0.85, 0.25, 0.15, 0.75, 0.25, 0.65],   # slightly reduced QS bias
    "steady": [0.75, 0.35, 0.45, 0.65, 0.35, 0.60],
    "sharp": [0.20, 0.85, 0.35, 0.70, 0.40, 0.45],   # boosted CQ affinity
    "bright": [0.25, 0.80, 0.30, 0.75, 0.25, 0.40],
    "heavy": [0.40, 0.35, 0.75, 0.45, 0.55, 0.50],
    "open": [0.55, 0.60, 0.45, 0.55, 0.35, 0.60],    # more balanced
    "unsure": [0.25, 0.70, 0.35, 0.35, 0.50, 0.40],
    "familiar": [0.65, 0.45, 0.35, 0.60, 0.25, 0.85], # reduced familiarity pull a bit
    "here": [0.50, 0.45, 0.60, 0.50, 0.35, 0.65],
    "now": [0.40, 0.55, 0.60, 0.60, 0.40, 0.50],
    "clear": [0.30, 0.85, 0.25, 0.90, 0.20, 0.50],    # stronger CQ boost
    "certain": [0.30, 0.80, 0.30, 0.95, 0.20, 0.50],
    "perhaps": [0.25, 0.65, 0.35, 0.45, 0.40, 0.45],
    "think": [0.20, 0.90, 0.35, 0.70, 0.30, 0.50],    # strong CQ affinity
    # ... add more as we test
}

def score_word_for_state(word, state):
    """Score how well a word matches the current field state"""
    weights = WORD_WEIGHTS.get(word, [0.33, 0.33, 0.33, 0.4, 0.4, 0.4])  # softened fallback
    score = (
        weights[0] * state.get("qs_level", 0.0) +
        weights[1] * state.get("cq_level", 0.0) +
        weights[2] * state.get("pe_level", 0.0) +
        weights[3] * state.get("coherence", 0.5) +
        weights[4] * state.get("turbulence", 0.5) +        # fixed: now correctly uses turbulence_pref
        weights[5] * state.get("familiarity", 0.5)
    )
    return score

def get_response_from_state(state, num_words=3):
    """Pull a small semantic cluster based on current state"""
    scored = [(word, score_word_for_state(word, state)) for word in ALL_WORDS]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_words = [w[0] for w in scored[:num_words]]
    return ", ".join(top_words) if top_words else "steady"

# ====================== RELIC RESIDUAL MAP ======================
def save_relic_residual_map(state, run_id):
    magnitude = np.sqrt(state["field_re"]**2 + state["field_im"]**2)
    
    # Attractor centre (approximate centre of Quiet-Settle)
    cy, cx = H // 2, W // 2
    
    # Compute radial distance
    y, x = np.indices((H, W))
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r = r.astype(int)
    
    # Radial average
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
    
    # Reconstruct symmetric field
    symmetric = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            rad = r[i, j]
            if rad < max_r:
                symmetric[i, j] = radial_profile[rad]
    
    # Residual
    residual = magnitude - symmetric
    
    # Save image
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(residual, cmap="RdBu_r", vmin=-np.max(np.abs(residual)), vmax=np.max(np.abs(residual)))
    ax.set_title(f"Relic Residual Map — {run_id}\n(Non-symmetric leftover structure)")
    plt.colorbar(im, ax=ax, label="Residual intensity")
    plt.tight_layout()
    
    filename = f"relic_residual_{run_id}.png"
    filepath = os.path.join(IMAGE_FOLDER, filename)
    plt.savefig(filepath, dpi=180, bbox_inches="tight")
    plt.close()
    
    print(f" → Saved Relic Residual Map: {filename}")
    print(f"   Max residual: {np.max(np.abs(residual)):.5f}")

# ====================== HELPERS ======================
def save_state(state, path=SAVE_PATH):
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

def text_to_ripple(text, H, W, scale=0.10):
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    rng = np.random.default_rng(seed)
    delta_re = np.zeros((H, W), dtype=np.float32)
    delta_im = np.zeros((H, W), dtype=np.float32)
    cy, cx = H // 2, W // 2
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    oy = (seed % 11) - 5
    ox = ((seed // 11) % 11) - 5
    blob = np.exp(
        -((yy - (cy + oy)) ** 2 + (xx - (cx + ox)) ** 2) / (2.0 * 6.0**2)
    ).astype(np.float32)
    phi = rng.uniform(-np.pi, np.pi)
    delta_re += scale * blob * np.cos(phi)
    delta_im += scale * blob * np.sin(phi)
    oy2 = ((seed // 101) % 15) - 7
    ox2 = ((seed // 211) % 15) - 7
    blob2 = np.exp(
        -((yy - (cy + oy2)) ** 2 + (xx - (cx + ox2)) ** 2) / (2.0 * 4.0**2)
    ).astype(np.float32)
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

# ====================== JB WORD MAP (Starter Version) ======================
WORD_POOL = {
    "core_state": ["calm", "steady", "sharp", "bright", "blurred", "heavy", "light", "open", "closed", "quiet"],
    "relation": ["near", "far", "with", "apart", "inside", "between", "around", "toward", "away"],
    "certainty": ["yes", "no", "maybe", "unsure", "clear", "certain", "perhaps", "likely", "doubt"],
    "memory_familiarity": ["known", "old", "familiar", "new", "remembered", "lost", "fresh", "distant"],
    "orientation": ["here", "there", "now", "before", "after", "around", "forward", "back"],
    "feeling_tone": ["warm", "cold", "kind", "tense", "playful", "sad", "gentle", "strong", "soft"],
    "action_intention": ["reach", "listen", "hold", "answer", "ask", "stay", "move", "rest", "think"],
    "link_connective": ["this", "that", "feels", "is", "becoming", "still", "and", "but", "yet"],
}

ALL_WORDS = [word for group in WORD_POOL.values() for word in group]

WORD_WEIGHTS = {
    "calm": [0.85, 0.25, 0.15, 0.75, 0.25, 0.65],
    "steady": [0.75, 0.35, 0.45, 0.65, 0.35, 0.60],
    "sharp": [0.20, 0.90, 0.35, 0.70, 0.40, 0.45],   # boosted CQ
    "bright": [0.25, 0.85, 0.30, 0.75, 0.25, 0.40],   # boosted CQ
    "heavy": [0.40, 0.35, 0.75, 0.45, 0.55, 0.50],
    "open": [0.55, 0.60, 0.45, 0.55, 0.35, 0.60],
    "unsure": [0.25, 0.70, 0.35, 0.35, 0.50, 0.40],
    "familiar": [0.65, 0.45, 0.35, 0.60, 0.25, 0.80], # slightly reduced
    "clear": [0.25, 0.92, 0.30, 0.85, 0.20, 0.50],    # strong CQ boost
    "certain": [0.30, 0.88, 0.30, 0.90, 0.20, 0.50],  # strong CQ boost
    "think": [0.20, 0.95, 0.35, 0.75, 0.30, 0.50],    # strong CQ boost
    "answer": [0.25, 0.90, 0.40, 0.70, 0.35, 0.45],   # strong CQ boost
    "here": [0.50, 0.45, 0.60, 0.50, 0.35, 0.65],
    "now": [0.40, 0.55, 0.60, 0.60, 0.40, 0.50],
    # ... add more as we test
}

def score_word_for_state(word, state):
    weights = WORD_WEIGHTS.get(word, [0.33, 0.33, 0.33, 0.4, 0.4, 0.4])
    
    # Channel identity is the primary driver
    channel_score = (
        weights[0] * state.get("qs_level", 0.0) * 3.5 +
        weights[1] * state.get("cq_level", 0.0) * 3.5 +
        weights[2] * state.get("pe_level", 0.0) * 3.5
    )
    
    # Modifiers are secondary tone/colour
    modifier_score = (
        weights[3] * state.get("coherence", 0.5) * 0.8 +
        weights[4] * state.get("turbulence", 0.5) * 0.5 +
        weights[5] * state.get("familiarity", 0.5) * 0.6
    )
    
    return float(channel_score + modifier_score)

def get_response_from_state(state, num_words=3):
    scored = [(word, score_word_for_state(word, state)) for word in ALL_WORDS]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_words = [w[0] for w in scored[:num_words]]
    return ", ".join(top_words) if top_words else "steady"

# ====================== TEMPLATE GEOMETRY (3 channels) ======================
def make_fixed_templates():
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    cy, cx = H // 2, W // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    templates_re = np.zeros((N_CHANNELS, H, W), dtype=np.float32)
    templates_im = np.zeros((N_CHANNELS, H, W), dtype=np.float32)

    # 0) Quiet-Settle — central calm default
    g0 = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * 7.0**2)).astype(np.float32)
    templates_re[QUIET_SETTLE] = g0 * np.cos(0.20)
    templates_im[QUIET_SETTLE] = g0 * np.sin(0.20)

    # 1) Cognitive/Query — tighter, more focused off-centre blob
    g1 = np.exp(
        -((x - (cx + 14)) ** 2 + (y - (cy - 10)) ** 2) / (2.0 * 3.5**2)
    ).astype(np.float32)
    templates_re[COGNITIVE_QUERY] = g1 * np.cos(1.58)
    templates_im[COGNITIVE_QUERY] = g1 * np.sin(1.58)

    # 2) Primary Engagement — narrower ring, slightly shifted phase
    ring2 = np.exp(-((r - 16.0) ** 2) / (2.0 * 2.0**2)).astype(np.float32)
    templates_re[PRIMARY_ENGAGEMENT] = ring2 * np.cos(THETA_STAR + 0.35)
    templates_im[PRIMARY_ENGAGEMENT] = ring2 * np.sin(THETA_STAR + 0.35)

    for k in range(N_CHANNELS):
        templates_re[k], templates_im[k] = normalize_complex(
            templates_re[k], templates_im[k]
        )
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
        "qs_vs_cq": round(overlap_score(state["templates_re"][QUIET_SETTLE], state["templates_im"][QUIET_SETTLE],
                                        state["templates_re"][COGNITIVE_QUERY], state["templates_im"][COGNITIVE_QUERY]), 3),
        "qs_vs_pe": round(overlap_score(state["templates_re"][QUIET_SETTLE], state["templates_im"][QUIET_SETTLE],
                                        state["templates_re"][PRIMARY_ENGAGEMENT], state["templates_im"][PRIMARY_ENGAGEMENT]), 3),
        "cq_vs_pe": round(overlap_score(state["templates_re"][COGNITIVE_QUERY], state["templates_im"][COGNITIVE_QUERY],
                                        state["templates_re"][PRIMARY_ENGAGEMENT], state["templates_im"][PRIMARY_ENGAGEMENT]), 3),
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
    except:
        pass
    
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "gains": json_list(state["gains"]),
        "usage": json_list(state["usage"]),
        "topo_pairs": {
            "qs_vs_cq": round(overlap_score(state["templates_re"][QUIET_SETTLE], state["templates_im"][QUIET_SETTLE],
                                            state["templates_re"][COGNITIVE_QUERY], state["templates_im"][COGNITIVE_QUERY]), 3),
            "qs_vs_pe": round(overlap_score(state["templates_re"][QUIET_SETTLE], state["templates_im"][QUIET_SETTLE],
                                            state["templates_re"][PRIMARY_ENGAGEMENT], state["templates_im"][PRIMARY_ENGAGEMENT]), 3),
            "cq_vs_pe": round(overlap_score(state["templates_re"][COGNITIVE_QUERY], state["templates_im"][COGNITIVE_QUERY],
                                            state["templates_re"][PRIMARY_ENGAGEMENT], state["templates_im"][PRIMARY_ENGAGEMENT]), 3),
        }
    }
    history.append(entry)
    history = history[-30:]
    
    with open(DRIFT_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

# ====================== RELIC RESIDUAL MAP ======================
def save_relic_residual_map(state, run_id):
    magnitude = np.sqrt(state["field_re"]**2 + state["field_im"]**2)
    
    # Attractor centre (approximate centre of Quiet-Settle)
    cy, cx = H // 2, W // 2
    
    # Compute radial distance
    y, x = np.indices((H, W))
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r = r.astype(int)
    
    # Radial average
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
    
    # Reconstruct symmetric field
    symmetric = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            rad = r[i, j]
            if rad < max_r:
                symmetric[i, j] = radial_profile[rad]
    
    # Residual
    residual = magnitude - symmetric
    
    # Save image
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(residual, cmap="RdBu_r", vmin=-np.max(np.abs(residual)), vmax=np.max(np.abs(residual)))
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
            d_re, d_im,
            state["templates_re"][k],
            state["templates_im"][k],
        )
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
        re, im = phase_lock_project(re, im, THETA_STAR, LAMBDA_LOCK)
    phase = np.arctan2(im, re)
    gy, gx = np.gradient(phase)
    grad_mag = np.mean(np.abs(gy)) + np.mean(np.abs(gx))
    lock_err = np.mean(np.abs(wrap_phase(phase - THETA_STAR)))
    coherence = float(max(0.0, 1.0 - 0.35 * grad_mag - 0.35 * lock_err))
    gain_update = 0.20 * weights.copy()
    gain_update[winner] += 0.50
    new_gains = state["gains"] + gain_scale * GAIN_LR * gain_update * coherence
    for k in range(N_CHANNELS):
        if new_gains[k] > UPPER_SOFT:
            new_gains[k] -= HOMEOSTASIS_RATE * (new_gains[k] - UPPER_SOFT)
        elif new_gains[k] < LOWER_SOFT:
            new_gains[k] += HOMEOSTASIS_RATE * (LOWER_SOFT - new_gains[k])
    state["gains"] = np.clip(new_gains, GAIN_MIN, GAIN_MAX).astype(np.float32)
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
            f" Dream {i+1:2d} | Relax win: {CHANNEL_NAMES[relax_result['winning_channel']]:22s}"
            f" | Coh: {relax_result['coherence_score']:.3f}{replay_label}"
        )
    print(f" Gains after dream: {np.round(state['gains'], 3)}")
    print("────────────────────────────────────────────────────────────\n")

# ====================== VISUALS ======================
def save_jelly_image(state, stage_name, timestamp):
    if not SAVE_IMAGES:
        return
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.5))
    magnitude = np.sqrt(state["field_re"]**2 + state["field_im"]**2)
    im1 = axs[0].imshow(magnitude, cmap="viridis")
    axs[0].set_title("Frankie's Brain Magnitude\n(brighter = more active)")
    plt.colorbar(im1, ax=axs[0])
    phase = np.arctan2(state["field_im"], state["field_re"])
    im2 = axs[1].imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axs[1].set_title("Frankie's Brain Phase\n(the locked wave pattern)")
    plt.colorbar(im2, ax=axs[1])
    fav = int(np.argmax(state["gains"]))
    plt.suptitle(
        f"Frankie {stage_name} — Favourite: {CHANNEL_NAMES[fav]} "
        f"(gain {state['gains'][fav]:.3f})"
    )
    plt.tight_layout()
    filename = f"frankie_{stage_name.lower().replace(' ', '_')}_{timestamp}.png"
    filepath = os.path.join(IMAGE_FOLDER, filename)
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f" → Saved {stage_name} image: {filename}")

def draw_jelly_brain(state):
    print("Drawing Frankie's jelly brain... (close the window to exit)")
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.5))
    magnitude = np.sqrt(state["field_re"]**2 + state["field_im"]**2)
    im1 = axs[0].imshow(magnitude, cmap="viridis")
    axs[0].set_title("Frankie's Brain Magnitude")
    plt.colorbar(im1, ax=axs[0])
    phase = np.arctan2(state["field_im"], state["field_re"])
    im2 = axs[1].imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axs[1].set_title("Frankie's Brain Phase")
    plt.colorbar(im2, ax=axs[1])
    plt.tight_layout()
    plt.show()

# ====================== MAIN ======================
if __name__ == "__main__":
    random.seed()
    ensure_log_dir()
    
    print("Frankie Channels Toy — 3-Channel v4 with Relic Map (Auto Mode)\n")
    
    for session_num in range(1, NUM_AUTO_SESSIONS + 1):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n=== Starting Session {session_num}/{NUM_AUTO_SESSIONS} ===")
        
        if os.path.exists(SAVE_PATH):
            state = load_state(SAVE_PATH)
            print("Frankie woke up with yesterday's blanket. 🛌")
            state["gains"] = np.clip(state["gains"] * MORNING_STRETCH, GAIN_MIN, GAIN_MAX).astype(np.float32)
        else:
            state = init_state()
            print("Fresh Frankie today. ✨")
        
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_jelly_image(state, "Wake-up", timestamp)
        
        # ---------------- Waking session ----------------
        waking_coherences = []   # ← ADD THIS LINE HERE

        waking_inputs = [
            "hello frankie good morning",
            "the moon smells blue today",
            "everything feels too much right now",
            "frankie what is the capital of france",
            "frankie what is my name",
            "what day is it today",
        ]
        turn_texts = (waking_inputs * (TURNS_PER_SESSION // len(waking_inputs) + 1))[:TURNS_PER_SESSION]
        for i, text in enumerate(turn_texts, start=1):
            delta_re, delta_im = text_to_ripple(text, H, W, scale=0.10)
            result = frankie_turn(state, delta_re, delta_im, gain_scale=WAKE_GAIN_SCALE, is_dream=False)
            
            waking_coherences.append(result["coherence_score"])   # ← ADD THIS LINE HERE

            add_to_replay_buffer(state, delta_re, delta_im, result["winning_channel"])
            print(
                f"Turn {i:2d} | Text: {text[:32]:32s} | "
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

        # ---------------- JB lexical response ----------------
        mean_coherence = float(np.mean(waking_coherences)) if waking_coherences else 0.5
        lexical_state = build_lexical_state(state, coherence=mean_coherence)
        jb_response = get_response_from_state(lexical_state, num_words=JB_RESPONSE_WORDS)
        print("────────────────────────────────────────────────────────────")
        print("JB WORD MAP RESPONSE")
        print("────────────────────────────────────────────────────────────")
        print(f" Lexical state : {lexical_state}")
        print(f" JB says       : {jb_response}")
        print("────────────────────────────────────────────────────────────\n")
        







        # ---------------- Dream ----------------
        dream_session(state)
        
        # ---------------- Summary, history & relic map ----------------
        print_session_summary(state, run_id)
        update_drift_history(state)
        save_relic_residual_map(state, run_id)   # ← new relic map
        
        save_state(state)
        print("✅ State saved. Frankie is sleeping soundly.\n")
        
        save_jelly_image(state, "After-Settle", timestamp)
        
        plt.close('all')
        print(f"Session {session_num} completed.\n")
    
    print("All auto sessions completed. Frankie is resting.")