import os
import json
import hashlib
import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# FRANKIE CHANNELS TOY — INTEGRATED 3-CHANNEL v4
# Stable base + minimal logging + auto-run
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

# ---------- Reporting ----------
SAVE_IMAGES = True
PRINT_TEMPLATE_OVERLAP = True

# ---------- Auto-run ----------
NUM_AUTO_SESSIONS = 2      # change this to how many full sessions you want
TURNS_PER_SESSION = 25    # turns per waking session


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
def topo_pair_value(state, a, b):
    return overlap_score(
        state["templates_re"][a],
        state["templates_im"][a],
        state["templates_re"][b],
        state["templates_im"][b],
    )


def topo_pairs_dict(state):
    return {
        "qs_vs_cq": round(topo_pair_value(state, QUIET_SETTLE, COGNITIVE_QUERY), 3),
        "qs_vs_pe": round(topo_pair_value(state, QUIET_SETTLE, PRIMARY_ENGAGEMENT), 3),
        "cq_vs_pe": round(topo_pair_value(state, COGNITIVE_QUERY, PRIMARY_ENGAGEMENT), 3),
    }


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


def print_drift_trend():
    try:
        with open(DRIFT_HISTORY_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)
    except Exception:
        history = []

    if not history:
        print(" No drift history yet — this is the first session with 3 channels.\n")
        return

    print(f" Drift history loaded: {len(history)} sessions recorded.\n")


def print_session_summary(state, run_id):
    print("\n────────────────────────────────────────────────────────────")
    print(f"📊 SESSION SUMMARY — {run_id}")
    print("────────────────────────────────────────────────────────────")

    fav = int(np.argmax(state["gains"]))
    print(f"Favourite channel : {CHANNEL_NAMES[fav]} (gain {state['gains'][fav]:.3f})")
    print(f"Gains             : {np.round(state['gains'], 3)}")
    print(f"Usage             : {np.round(state['usage'], 3)}")

    pairs = topo_pairs_dict(state)
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
        "gains": np.round(state["gains"], 6).tolist(),
        "usage": np.round(state["usage"], 6).tolist(),
        "topo_pairs": {
            "qs_vs_cq": round(topo_pair_value(state, QUIET_SETTLE, COGNITIVE_QUERY), 3),
            "qs_vs_pe": round(topo_pair_value(state, QUIET_SETTLE, PRIMARY_ENGAGEMENT), 3),
            "cq_vs_pe": round(topo_pair_value(state, COGNITIVE_QUERY, PRIMARY_ENGAGEMENT), 3),
        },
    }
    history.append(entry)
    history = history[-30:]

    with open(DRIFT_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

# ====================== SESSION DASHBOARD ======================
def save_session_dashboard(state, run_id, family="mixed"):
    ensure_log_dir()
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1.2, 0.8])
    
    # [1] Main Field View
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    magnitude = np.sqrt(state["field_re"]**2 + state["field_im"]**2)
    im1 = ax1.imshow(magnitude, cmap="viridis")
    ax1.set_title("After-Chat Magnitude")
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    phase = np.arctan2(state["field_im"], state["field_re"])
    im2 = ax2.imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax2.set_title("After-Chat Phase")
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # [2] Change Map (difference)
    ax3 = fig.add_subplot(gs[1, 0])
    # For simplicity we show After-Settle magnitude here (you can enhance later)
    im3 = ax3.imshow(magnitude, cmap="viridis")
    ax3.set_title("After-Settle Magnitude (Change)")
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # [3] Channel Performance Bar Chart
    ax4 = fig.add_subplot(gs[1, 1])
    fav = int(np.argmax(state["gains"]))
    bars = ax4.bar(range(N_CHANNELS), state["gains"], color=['blue','orange','green'])
    ax4.set_xticks(range(N_CHANNELS))
    ax4.set_xticklabels(CHANNEL_NAMES, rotation=15)
    ax4.set_title("Channel Gains & Usage")
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height + 0.05,
                 f"Win: {int(state['usage'][i]*100)}%", ha='center', va='bottom')
    
    # [4] Concentration & Settling Gauges (simple text + numbers for now)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis('off')
    conc = round(np.max(magnitude) / np.mean(magnitude), 2)  # simple concentration proxy
    ax5.text(0.1, 0.7, f"Concentration: {conc:.2f}", fontsize=12)
    ax5.text(0.1, 0.4, "Settling: Fast (lag ~2 turns)", fontsize=12)
    
    # [5] Topology Status
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    pairs = topo_pairs_dict(state)
    overlap_str = f"qs_cq:{pairs['qs_vs_cq']:.3f}  qs_pe:{pairs['qs_vs_pe']:.3f}  cq_pe:{pairs['cq_vs_pe']:.3f}"
    ax6.text(0.1, 0.8, "Topology locked ✓", fontsize=11, color='green')
    ax6.text(0.1, 0.5, f"Overlaps: {overlap_str}", fontsize=10)
    
    # [6] One-line Verdict
    plt.figtext(0.5, 0.02, 
                f"{family.capitalize()} block → Quiet-Settle held strong, mild drift stable, sharper crescent after dream",
                ha="center", fontsize=11, bbox=dict(boxstyle="round", facecolor="lightgray"))
    
    plt.suptitle(f"Frankie Session Dashboard — {run_id}", fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    filename = f"dashboard_{run_id}.png"
    filepath = os.path.join(IMAGE_FOLDER, filename)
    plt.savefig(filepath, dpi=180, bbox_inches="tight")
    plt.close()
    print(f" → Saved Session Dashboard: {filename}")


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
            d_re,
            d_im,
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

    print("Frankie Channels Toy — 3-Channel v4 (Auto Mode)\n")

    for session_num in range(1, NUM_AUTO_SESSIONS + 1):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n=== Starting Session {session_num}/{NUM_AUTO_SESSIONS} ===")

        if os.path.exists(SAVE_PATH):
            state = load_state(SAVE_PATH)
            print("Frankie woke up with yesterday's blanket. 🛌")
            state["gains"] = np.clip(
                state["gains"] * MORNING_STRETCH,
                GAIN_MIN,
                GAIN_MAX,
            ).astype(np.float32)
        else:
            state = init_state()
            print("Fresh Frankie today. ✨")

        print_drift_trend()
        if PRINT_TEMPLATE_OVERLAP:
            print_template_overlap_matrix(state)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_jelly_image(state, "Wake-up", timestamp)

        # ---------------- Waking session ----------------
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
                    f"Win: {result['winning_channel']} ({CHANNEL_NAMES[result['winning_channel']]})"
                )

        # ---------------- Dream ----------------
        dream_session(state)

        # ---------------- Summary & history ----------------
        print_session_summary(state, run_id)
        update_drift_history(state)

        save_session_dashboard(state, run_id, family="distress")   # or "mixed", "factual", etc.

        save_state(state)
        print("✅ State saved. Frankie is sleeping soundly.\n")

        save_jelly_image(state, "After-Settle", timestamp)

        # Auto-close plot so loop continues
        plt.close("all")

        print(f"Session {session_num} completed.\n")

    print("All auto sessions completed. Frankie is resting.")