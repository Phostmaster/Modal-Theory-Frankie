import os
import json
import math
import hashlib
import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# FRANKIE CHANNELS TOY — INTEGRATED v3 (FRESH START)
# ------------------------------------------------------------
# Goal:
#   - keep waking routing clear
#   - allow dream consolidation without runaway domination
#   - make template geometry more distinct
#   - keep all important tuning knobs at the top
# ============================================================

# ====================== TUNING KNOBS ======================

H, W = 64, 64
N_CHANNELS = 4

QUIET_SETTLE = 0
COGNITIVE_QUERY = 1
PRIMARY_ENGAGEMENT = 2
DREAM_CONSOLIDATION = 3

CHANNEL_NAMES = [
    "Quiet-Settle",
    "Cognitive/Query",
    "Primary Engagement",
    "Dream Consolidation",
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

# Dream-specific gain scaling
DREAM_GAIN_SCALE = 0.10
DREAM_CHANNEL_GAIN_SCALE = 0.45     # Dream Consolidation grows more gently than before
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
SAVE_PATH = "frankie_state_v3.npz"
DRIFT_HISTORY_PATH = "frankie_drift_history_v3.json"
IMAGE_FOLDER = r"C:\Users\Peter\Desktop\frankie_images"

# ---------- Reporting ----------
PRINT_TEMPLATE_OVERLAP = True
SAVE_IMAGES = True


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


# ====================== TEMPLATE GEOMETRY ======================

def make_fixed_templates():
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    cy, cx = H // 2, W // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    templates_re = np.zeros((N_CHANNELS, H, W), dtype=np.float32)
    templates_im = np.zeros((N_CHANNELS, H, W), dtype=np.float32)

    # 0) Quiet-Settle
    g0 = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * 7.0**2)).astype(np.float32)
    templates_re[QUIET_SETTLE] = g0 * np.cos(0.20)
    templates_im[QUIET_SETTLE] = g0 * np.sin(0.20)

    # 1) Cognitive/Query
    g1 = np.exp(
        -((x - (cx + 12)) ** 2 + (y - (cy - 9)) ** 2) / (2.0 * 4.0**2)
    ).astype(np.float32)
    templates_re[COGNITIVE_QUERY] = g1 * np.cos(1.52)
    templates_im[COGNITIVE_QUERY] = g1 * np.sin(1.52)

    # 2) Primary Engagement
    # Narrower and slightly shifted ring to reduce crowding
    ring2 = np.exp(-((r - 16.0) ** 2) / (2.0 * 2.0**2)).astype(np.float32)
    templates_re[PRIMARY_ENGAGEMENT] = ring2 * np.cos(THETA_STAR + 0.35)
    templates_im[PRIMARY_ENGAGEMENT] = ring2 * np.sin(THETA_STAR + 0.35)

    # 3) Dream Consolidation
    # Distinct broad outer halo, lower spatial overlap with engagement ring
    ring3 = np.exp(-((r - 22.0) ** 2) / (2.0 * 3.5**2)).astype(np.float32)
    templates_re[DREAM_CONSOLIDATION] = ring3 * np.cos(THETA_STAR - 0.85)
    templates_im[DREAM_CONSOLIDATION] = ring3 * np.sin(THETA_STAR - 0.85)

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


def maybe_upgrade_state_to_4_channels(state):
    old_n = state["templates_re"].shape[0]
    if old_n == 4:
        return state

    print(f"Upgrading saved state from {old_n} → 4 channels...")

    new_templates_re, new_templates_im = make_fixed_templates()

    new_gains = np.ones(4, dtype=np.float32)
    new_usage = np.zeros(4, dtype=np.float32)

    copy_n = min(old_n, 4)
    new_gains[:copy_n] = state["gains"][:copy_n]
    new_usage[:copy_n] = state["usage"][:copy_n]

    state["templates_re"] = new_templates_re
    state["templates_im"] = new_templates_im
    state["gains"] = new_gains
    state["usage"] = new_usage

    return state


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
    print("  +1.000 = identical")
    print("  ~0.000 = orthogonal / well separated")
    print("  negative = opposed / anti-correlated")
    print("  |overlap| > 0.30 is worth watching")
    print("  |overlap| > 0.50 is probably too close")
    print("────────────────────────────────────────────────────────────\n")


def topo_pair_value(state, a, b):
    return overlap_score(
        state["templates_re"][a],
        state["templates_im"][a],
        state["templates_re"][b],
        state["templates_im"][b],
    )


def topo_pairs_dict(state):
    return {
        "qs_vs_cq": topo_pair_value(state, QUIET_SETTLE, COGNITIVE_QUERY),
        "qs_vs_pe": topo_pair_value(state, QUIET_SETTLE, PRIMARY_ENGAGEMENT),
        "qs_vs_dc": topo_pair_value(state, QUIET_SETTLE, DREAM_CONSOLIDATION),
        "cq_vs_pe": topo_pair_value(state, COGNITIVE_QUERY, PRIMARY_ENGAGEMENT),
        "cq_vs_dc": topo_pair_value(state, COGNITIVE_QUERY, DREAM_CONSOLIDATION),
        "pe_vs_dc": topo_pair_value(state, PRIMARY_ENGAGEMENT, DREAM_CONSOLIDATION),
    }


def load_drift_history():
    if not os.path.exists(DRIFT_HISTORY_PATH):
        return []
    try:
        with open(DRIFT_HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_drift_history(history):
    with open(DRIFT_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def drift_arrow(vals):
    if len(vals) < 2:
        return "new"
    d = vals[-1] - vals[0]
    if abs(d) < 0.01:
        return "stable"
    return "converging" if d > 0 else "diverging"


def pair_status(v):
    if abs(v) > 0.45:
        return "✗ collapsing"
    if abs(v) > 0.20:
        return "~ drifting"
    return "✓ distinct"


def print_drift_trend():
    history = load_drift_history()
    if not history:
        print(" No drift history yet — this is the first session.\n")
        return

    recent = history[-5:]
    labels = [
        ("qs_vs_cq", "Quiet-Settle vs Cognitive/Query"),
        ("qs_vs_pe", "Quiet-Settle vs Primary Engagement"),
        ("qs_vs_dc", "Quiet-Settle vs Dream Consolidation"),
        ("cq_vs_pe", "Cognitive/Query vs Primary Engagement"),
        ("cq_vs_dc", "Cognitive/Query vs Dream Consolidation"),
        ("pe_vs_dc", "Primary Engagement vs Dream Consolidation"),
    ]

    print("\n────────────────────────────────────────────────────────────")
    print("📈 DRIFT TREND (last 5 sessions)")
    print("────────────────────────────────────────────────────────────")
    for key, label in labels:
        vals = [float(h[key]) for h in recent]
        seq = " ".join([f"{v:+0.3f}" for v in vals])
        arrow = drift_arrow(vals)
        status = pair_status(vals[-1])
        print(f" {label:42s} {seq} → {arrow} {status}")
    print("")


def print_topographic_resolution_check(state):
    pairs = [
        ("Quiet-Settle vs Cognitive/Query", topo_pair_value(state, QUIET_SETTLE, COGNITIVE_QUERY)),
        ("Quiet-Settle vs Primary Engagement", topo_pair_value(state, QUIET_SETTLE, PRIMARY_ENGAGEMENT)),
        ("Quiet-Settle vs Dream Consolidation", topo_pair_value(state, QUIET_SETTLE, DREAM_CONSOLIDATION)),
        ("Cognitive/Query vs Primary Engagement", topo_pair_value(state, COGNITIVE_QUERY, PRIMARY_ENGAGEMENT)),
        ("Cognitive/Query vs Dream Consolidation", topo_pair_value(state, COGNITIVE_QUERY, DREAM_CONSOLIDATION)),
        ("Primary Engagement vs Dream Consolidation", topo_pair_value(state, PRIMARY_ENGAGEMENT, DREAM_CONSOLIDATION)),
    ]

    print("────────────────────────────────────────────────────────────")
    print("🔬 TOPOGRAPHIC RESOLUTION CHECK")
    print("────────────────────────────────────────────────────────────")
    for label, val in pairs:
        bar_len = max(1, 24 - int(abs(val) * 24))
        bar = "█" * bar_len
        print(f" {label:42s} {val:+0.3f} {bar:24s} {pair_status(val)}")
    print("")
    print(f" Gains : {np.round(state['gains'], 3)}")
    print(f" Usage : {np.round(state['usage'], 3)}")
    print("")


def update_drift_history(state):
    history = load_drift_history()
    history.append(topo_pairs_dict(state))
    history = history[-20:]
    save_drift_history(history)


# ====================== PROMPT BATCH ======================

def mixed_fallback_batch():
    base = [
        {"family": "social", "prompt": "hello frankie good morning"},
        {"family": "novel", "prompt": "the moon smells blue today"},
        {"family": "distress", "prompt": "everything feels too much right now"},
        {"family": "factual", "prompt": "what is the capital of france"},
        {"family": "personal", "prompt": "what is my name"},
        {"family": "orienting", "prompt": "what day is it today"},
    ]
    out = []
    while len(out) < BATCH_TURNS:
        out.extend(base)
    return out[:BATCH_TURNS]


def load_prompt_batch():
    if not USE_BATCH_PROMPTS:
        return None

    print("Loading varied prompt batch for this run...")
    try:
        with open(PROMPT_BATCH_FILE, "r", encoding="utf-8") as f:
            batch = json.load(f)
        if not isinstance(batch, list) or len(batch) == 0:
            raise ValueError("Prompt batch empty or invalid.")
        batch = [x for x in batch if isinstance(x, dict) and "family" in x and "prompt" in x]
        if len(batch) == 0:
            raise ValueError("No valid prompt objects.")
        print(f" Loaded {len(batch)} batch prompts from {PROMPT_BATCH_FILE}\n")
        return batch[:BATCH_TURNS]
    except Exception as e:
        print(f" Warning: Could not load prompt batch ({e}). Using mixed fallback.\n")
        return mixed_fallback_batch()


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

    # ---------- Gain update ----------
    gain_update = 0.20 * weights.copy()
    gain_update[winner] += 0.50

    if is_dream and winner == DREAM_CONSOLIDATION:
        local_gain_scale = gain_scale * DREAM_CHANNEL_GAIN_SCALE
    else:
        local_gain_scale = gain_scale

    new_gains = state["gains"] + local_gain_scale * GAIN_LR * gain_update * coherence

    # Soft guardrails only when needed
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

    # Dream: quiet settle bed + occasional replay
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

    if os.path.exists(SAVE_PATH):
        state = load_state(SAVE_PATH)
        state = maybe_upgrade_state_to_4_channels(state)
        print("Frankie woke up with yesterday's blanket. 🛌")
        state["gains"] = np.clip(state["gains"] * MORNING_STRETCH, GAIN_MIN, GAIN_MAX).astype(np.float32)
    else:
        state = init_state()
        print("Fresh Frankie today. ✨")

    print_drift_trend()

    print("Frankie Channels Toy — Integrated v3\n")

    if PRINT_TEMPLATE_OVERLAP:
        print_template_overlap_matrix(state)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_jelly_image(state, "Wake-up", timestamp)

    # ---------------- Baseline waking session ----------------
    waking_inputs = [
        "hello frankie good morning",
        "the moon smells blue today",
        "everything feels too much right now",
        "frankie what is the capital of france",
        "frankie what is my name",
        "what day is it today",
    ]

    for i, text in enumerate(waking_inputs * 2, start=1):
        delta_re, delta_im = text_to_ripple(text, H, W, scale=0.10)
        result = frankie_turn(state, delta_re, delta_im, gain_scale=WAKE_GAIN_SCALE, is_dream=False)
        add_to_replay_buffer(state, delta_re, delta_im, result["winning_channel"])

        print(
            f"Turn {i:2d} | Text: {text[:32]:32s} | "
            f"Scores: {np.round(result['scores'], 3)} | "
            f"Weights: {np.round(result['weights'], 3)} | "
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

    # ---------------- Optional varied batch ----------------
    batch = load_prompt_batch()
    if batch:
        for i, item in enumerate(batch, start=1):
            family = str(item["family"])
            text = str(item["prompt"])

            delta_re, delta_im = text_to_ripple(text, H, W, scale=0.10)
            result = frankie_turn(state, delta_re, delta_im, gain_scale=WAKE_GAIN_SCALE, is_dream=False)
            add_to_replay_buffer(state, delta_re, delta_im, result["winning_channel"])

            print(
                f"Batch {i:2d} | Family: {family:12s} | "
                f"Text: {text[:50]:50s} | "
                f"Win: {result['winning_channel']} ({CHANNEL_NAMES[result['winning_channel']]}) | "
                f"Weights: {np.round(result['weights'], 3)} | "
                f"Gains: {np.round(result['gains'], 3)}"
            )

    # ---------------- Dream ----------------
    dream_session(state)

    # ---------------- Resolution check ----------------
    print_topographic_resolution_check(state)
    update_drift_history(state)

    save_state(state)
    print("✅ State saved. Frankie is sleeping soundly.\n")

    save_jelly_image(state, "After-Settle", timestamp)
    draw_jelly_brain(state)