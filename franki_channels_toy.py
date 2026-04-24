import hashlib
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import json
import random

# ====================== TUNING KNOBS (all in one place) ======================
H, W = 64, 64
N_CHANNELS = 3

# Channel names
QUIET_SETTLE, COGNITIVE_QUERY, PRIMARY_ENGAGEMENT = 0, 1, 2
CHANNEL_NAMES = ["Quiet-Settle", "Cognitive/Query", "Primary Engagement"]

# Core dynamics
ALPHA_IN = 0.12          # how strongly input ripples affect the field
LAMBDA_DIFF = 0.18       # diffusion (spreading) strength
LAMBDA_DAMP = 0.03       # natural damping
LAMBDA_LOCK = 0.10       # how strongly the field locks to target phase
LAMBDA_CHAN = 0.04       # how strongly channel bias pulls the field
SOFTMAX_BETA = 3.0       # how sharp the winner selection is

# Gain & usage regulation
GAIN_LR = 0.02           # learning rate for gains during active turns
USAGE_RHO = 0.05         # how quickly usage memory updates
GAIN_MIN = 0.7           # floor for gains (you set this)
GAIN_MAX = 2.0           # ceiling for gains

# Soft self-regulation guardrails
UPPER_SOFT = 1.95        # only nudge gains down if they rise above this
LOWER_SOFT = 0.75        # only nudge gains up if they fall below this
HOMEOSTASIS_RATE = 0.06  # strength of those nudges (lower = gentler)

# Morning stretch - gives fresh breathing room each day
MORNING_STRETCH = 0.92   # scales gains down slightly on wake-up
                         # 1.0 = no stretch, lower = stronger morning reset (try 0.85-0.95)

# Relaxation settings
RELAX_GAIN_SCALE = 0.35  # how strongly gains update during quiet time (lower = gentler learning)
N_RELAX_STEPS = 6        # internal relaxation steps per turn

# Phase target
THETA_STAR = np.deg2rad(255.0)

# Persistent state
SAVE_PATH = "frankie_state.npz"
IMAGE_FOLDER = r"C:\Users\Peter\Desktop\frankie_images"

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
    )

def load_state(path=SAVE_PATH):
    data = np.load(path)
    return {
        "field_re": data["field_re"].astype(np.float32),
        "field_im": data["field_im"].astype(np.float32),
        "templates_re": data["templates_re"].astype(np.float32),
        "templates_im": data["templates_im"].astype(np.float32),
        "gains": data["gains"].astype(np.float32),
        "usage": data["usage"].astype(np.float32),
    }

def laplacian(z):
    return (
        np.roll(z, 1, axis=0) + np.roll(z, -1, axis=0) +
        np.roll(z, 1, axis=1) + np.roll(z, -1, axis=1) - 4.0 * z
    )

def phase_lock_project(re, im, theta_star, lock_strength):
    mag = np.sqrt(re * re + im * im) + 1e-8
    phase = np.arctan2(im, re)
    d = (phase - theta_star + np.pi) % (2 * np.pi) - np.pi
    new_phase = phase - lock_strength * d
    return mag * np.cos(new_phase), mag * np.sin(new_phase)

def normalize_complex(re, im):
    n = np.sqrt(np.sum(re * re + im * im)) + 1e-8
    return re / n, im / n

def overlap_score(a_re, a_im, b_re, b_im):
    num = np.sum(a_re * b_re + a_im * b_im)
    den = (
        np.sqrt(np.sum(a_re * a_re + a_im * a_im)) *
        np.sqrt(np.sum(b_re * b_re + b_im * b_im)) + 1e-8
    )
    return float(num / den)

def softmax(x, beta=1.0):
    z = beta * (np.asarray(x, dtype=np.float64) - np.max(x))
    e = np.exp(z)
    return (e / np.sum(e)).astype(np.float32)

# ====================== INPUT RIPPLE ======================
def text_to_ripple(text, H, W, scale=0.10):
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    rng = np.random.default_rng(seed)
    delta_re = np.zeros((H, W), dtype=np.float32)
    delta_im = np.zeros((H, W), dtype=np.float32)
    cy, cx = H // 2, W // 2
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # Main blob
    oy = (seed % 11) - 5
    ox = ((seed // 11) % 11) - 5
    blob = np.exp(-((yy - (cy + oy))**2 + (xx - (cx + ox))**2) / (2.0 * 6.0**2)).astype(np.float32)
    phi = rng.uniform(-np.pi, np.pi)
    delta_re += scale * blob * np.cos(phi)
    delta_im += scale * blob * np.sin(phi)

    # Secondary smaller blob
    oy2 = ((seed // 101) % 15) - 7
    ox2 = ((seed // 211) % 15) - 7
    blob2 = np.exp(-((yy - (cy + oy2))**2 + (xx - (cx + ox2))**2) / (2.0 * 4.0**2)).astype(np.float32)
    phi2 = rng.uniform(-np.pi, np.pi)
    delta_re += 0.5 * scale * blob2 * np.cos(phi2)
    delta_im += 0.5 * scale * blob2 * np.sin(phi2)

    return delta_re, delta_im

# ====================== INIT ======================
def make_fixed_templates():
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    cy, cx = H // 2, W // 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    templates_re = np.zeros((N_CHANNELS, H, W), dtype=np.float32)
    templates_im = np.zeros((N_CHANNELS, H, W), dtype=np.float32)

    # Quiet-Settle: centred smooth blob
    g = np.exp(-((x - cx)**2 + (y - cy)**2) / (2.0 * 7.0**2)).astype(np.float32)
    templates_re[QUIET_SETTLE] = g * np.cos(0.2)
    templates_im[QUIET_SETTLE] = g * np.sin(0.2)

    # Cognitive/Query: off-centre skewed
    g = np.exp(-((x - (cx + 10))**2 + (y - (cy - 8))**2) / (2.0 * 6.0**2)).astype(np.float32)
    templates_re[COGNITIVE_QUERY] = g * np.cos(1.4)
    templates_im[COGNITIVE_QUERY] = g * np.sin(1.4)

    # Primary Engagement: soft ring
    ring = np.exp(-((r - 14.0)**2) / (2.0 * 3.0**2)).astype(np.float32)
    templates_re[PRIMARY_ENGAGEMENT] = ring * np.cos(THETA_STAR)
    templates_im[PRIMARY_ENGAGEMENT] = ring * np.sin(THETA_STAR)

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
    }

# ====================== CORE TURN ======================
def frankie_turn(state, delta_re, delta_im, gain_scale=1.0):
    scores = np.zeros(N_CHANNELS, dtype=np.float32)
    d_re, d_im = normalize_complex(delta_re, delta_im)

    for k in range(N_CHANNELS):
        scores[k] = overlap_score(d_re, d_im, state["templates_re"][k], state["templates_im"][k])

    q = state["gains"] * scores
    weights = softmax(q, beta=SOFTMAX_BETA)

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
    lock_err = np.mean(np.abs((phase - THETA_STAR + np.pi) % (2 * np.pi) - np.pi))

    score = float(max(0.0, 1.0 - 0.35 * grad_mag - 0.35 * lock_err))

    # Gain update with winner sharpening
    winner = int(np.argmax(weights))
    gain_update = 0.25 * weights.copy()
    gain_update[winner] += 0.55

    new_gains = state["gains"] + gain_scale * GAIN_LR * gain_update * score

    # Gentle self-regulation only when needed
    for k in range(N_CHANNELS):
        if new_gains[k] > UPPER_SOFT:
            new_gains[k] -= HOMEOSTASIS_RATE * (new_gains[k] - UPPER_SOFT)
        elif new_gains[k] < LOWER_SOFT:
            new_gains[k] += HOMEOSTASIS_RATE * (LOWER_SOFT - new_gains[k])

    state["gains"] = np.clip(new_gains, GAIN_MIN, GAIN_MAX).astype(np.float32)

    state["usage"] = (
        (1.0 - USAGE_RHO) * state["usage"] + USAGE_RHO * weights
    ).astype(np.float32)

    state["field_re"] = re.astype(np.float32)
    state["field_im"] = im.astype(np.float32)

    return {
        "scores": scores.copy(),
        "weights": weights.copy(),
        "winning_channel": winner,
        "coherence_score": score,
        "gains": state["gains"].copy(),
        "usage": state["usage"].copy(),
    }

# ====================== IMAGE SAVE ======================
def save_jelly_image(state, stage_name, timestamp):
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

    fav_gain = int(np.argmax(state["gains"]))
    plt.suptitle(
        f"Frankie {stage_name} — Favourite by gain: "
        f"{CHANNEL_NAMES[fav_gain]} (gain {state['gains'][fav_gain]:.3f})"
    )
    plt.tight_layout()

    filename = f"frankie_{stage_name.lower().replace(' ', '_')}_{timestamp}.png"
    filepath = os.path.join(IMAGE_FOLDER, filename)
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f" → Saved {stage_name} image: {filename}")

# ====================== MAIN ======================
if __name__ == "__main__":
    if os.path.exists(SAVE_PATH):
        state = load_state(SAVE_PATH)
        print("Frankie woke up with yesterday's blanket. 🛌")
        
        # Gentle morning stretch - gives fresh breathing room each day
        state["gains"] = state["gains"] * MORNING_STRETCH
        state["gains"] = np.clip(state["gains"], GAIN_MIN, GAIN_MAX).astype(np.float32)
        
    else:
        state = init_state()
        print("Fresh Frankie today. ✨")

    print("Frankie Channels Toy - Cleaned & Tunable Test Started\n")

    test_inputs = [
        "hello frankie good morning",
        "the moon smells blue today",
        "everything feels too much right now",
        "frankie what is the capital of france",
        "frankie what is my name",
        "what day is it today",
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Wake-up snapshot
    save_jelly_image(state, "Wake-up", timestamp)

    # Active chat turns
    for i, text in enumerate(test_inputs * 4, start=1):
        delta_re, delta_im = text_to_ripple(text, H, W, scale=0.10)
        result = frankie_turn(state, delta_re, delta_im, gain_scale=1.0)
        win_name = CHANNEL_NAMES[result["winning_channel"]]
        print(
            f"Turn {i:2d} | "
            f"Text: {text[:60]:60} | "
            f"Scores: {np.round(result['scores'], 3)} | "
            f"Weights: {np.round(result['weights'], 3)} | "
            f"Win: {result['winning_channel']} ({win_name}) | "
            f"Gains: {np.round(result['gains'], 3)} | "
            f"Usage: {np.round(result['usage'], 3)} | "
            f"Score: {result['coherence_score']:.3f}"
        )

    print("\n=== FINAL STATE ===")
    print("Gains: ", np.round(state["gains"], 3))
    print("Usage: ", np.round(state["usage"], 3))
    fav_gain = int(np.argmax(state["gains"]))
    fav_usage = int(np.argmax(state["usage"]))
    print("Favourite by gain :", fav_gain, f"({CHANNEL_NAMES[fav_gain]})")
    print("Favourite by usage:", fav_usage, f"({CHANNEL_NAMES[fav_usage]})")

    save_jelly_image(state, "After-Chat", timestamp)

    # ====================== LOAD VARIED PROMPT BATCH ======================
    # This replaces the old fixed list with fresh, mixed prompts from Qwen
    print("\nLoading varied prompt batch for this run...")
    try:
        with open("frankie_prompt_batch.json", "r", encoding="utf-8") as f:
            prompt_batch = json.load(f)
        
        # Shuffle so families are nicely mixed (your suggestion)
        random.shuffle(prompt_batch)
        
        print(f"Loaded and shuffled {len(prompt_batch)} natural prompts.\n")
    except Exception as e:
        print(f"Warning: Could not load prompt batch ({e}). Using fallback.")
        prompt_batch = [{"family": "social", "prompt": "hello frankie good morning"}] * 24

    # Run the varied prompts
    for i, item in enumerate(prompt_batch, start=1):
        family = item.get("family", "unknown")
        text = item.get("prompt", "hello frankie")

        delta_re, delta_im = text_to_ripple(text, H, W, scale=0.10)
        result = frankie_turn(state, delta_re, delta_im, gain_scale=1.0)

        win_name = CHANNEL_NAMES[result["winning_channel"]]
        print(
            f"Turn {i:2d} | "
            f"Family: {family:12s} | "
            f"Text: {text[:60]:60} | "
            f"Win: {result['winning_channel']} ({win_name}) | "
            f"Gains: {np.round(result['gains'], 3)} | "
            f"Usage: {np.round(result['usage'], 3)}"
        )

    # ====================== RELAXATION ======================
    print("\nLetting Frankie relax for 10 extra turns with no new text...")
    for _ in range(10):
        delta_re = np.zeros((H, W), dtype=np.float32)
        delta_im = np.zeros((H, W), dtype=np.float32)
        result = frankie_turn(state, delta_re, delta_im, gain_scale=RELAX_GAIN_SCALE)
        win_name = CHANNEL_NAMES[result["winning_channel"]]
        print(
            f"Relax turn | Win: {result['winning_channel']} ({win_name}) | "
            f"Score: {result['coherence_score']:.3f} | "
            f"Gains: {np.round(result['gains'], 3)}"
        )

    # ====================== RELAXATION ======================
    print("\nLetting Frankie relax for 10 extra turns with no new text...")
    for _ in range(10):
        delta_re = np.zeros((H, W), dtype=np.float32)
        delta_im = np.zeros((H, W), dtype=np.float32)
        result = frankie_turn(state, delta_re, delta_im, gain_scale=RELAX_GAIN_SCALE)
        win_name = CHANNEL_NAMES[result["winning_channel"]]
        print(
            f"Relax turn | Win: {result['winning_channel']} ({win_name}) | "
            f"Score: {result['coherence_score']:.3f} | "
            f"Gains: {np.round(result['gains'], 3)}"
        )



    # Relaxation (quiet cuddle time)
    print("\nLetting Frankie relax for 10 extra turns with no new text...")
    for _ in range(10):
        delta_re = np.zeros((H, W), dtype=np.float32)
        delta_im = np.zeros((H, W), dtype=np.float32)
        result = frankie_turn(state, delta_re, delta_im, gain_scale=RELAX_GAIN_SCALE)
        win_name = CHANNEL_NAMES[result["winning_channel"]]
        print(
            f"Relax turn | Win: {result['winning_channel']} ({win_name}) | "
            f"Score: {result['coherence_score']:.3f} | "
            f"Gains: {np.round(result['gains'], 3)}"
        )

    save_jelly_image(state, "After-Settle", timestamp)

    save_state(state)
    print("\nFrankie tucked himself in. Goodnight little mate. 🌙")