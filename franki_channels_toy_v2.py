import hashlib
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ====================== CONFIG ======================
H, W = 64, 64
N_CHANNELS = 4

# Channel indices — 4th channel is the new Dream Consolidation channel
QUIET_SETTLE, ALTERNATE_TURBULENT, PRIMARY_ENGAGEMENT, DREAM_CONSOLIDATION = 0, 1, 2, 3

CHANNEL_NAMES = [
    "Quiet-Settle",
    "Alternate/Turbulent",
    "Primary Engagement",
    "Dream Consolidation",
]

# ── Field dynamics ──────────────────────────────────
ALPHA_IN = 0.12
LAMBDA_DIFF = 0.18
LAMBDA_DAMP = 0.03
LAMBDA_LOCK = 0.10
LAMBDA_CHAN = 0.04

# ── Routing ────────────────────────────────────────
SOFTMAX_BETA = 3.0

# ── Gain adaptation ────────────────────────────────
GAIN_LR = 0.02
USAGE_RHO = 0.05

# ── Soft renormalisation ───────────────────────────
GAIN_SOFT_TARGET = 1.0
GAIN_SOFT_RATE = 0.01
GAIN_FLOOR = 0.3
GAIN_CEIL = 3.0

# ── Template learning ──────────────────────────────
WAKE_PLASTICITY = 0.0025
DREAM_PLASTICITY = 0.001
LATERAL_INHIBITION_STRENGTH = 0.000

# ── Dream loop ─────────────────────────────────────
DREAM_TURNS = 4
REPLAY_BUFFER_SIZE = 20
REPLAY_TURNS = 6

# ── Attractor ──────────────────────────────────────
THETA_STAR = np.deg2rad(255.0)

# ── Relaxation ─────────────────────────────────────
N_RELAX_STEPS = 6

# ── Persistence ────────────────────────────────────
SAVE_PATH = "frankie_state.npz"
DRIFT_LOG_PATH = "frankie_drift_log.json"

# ====================== HELPERS ======================
def laplacian(z):
    return (
        np.roll(z, 1, axis=0)
        + np.roll(z, -1, axis=0)
        + np.roll(z, 1, axis=1)
        + np.roll(z, -1, axis=1)
        - 4.0 * z
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
        np.sqrt(np.sum(a_re * a_re + a_im * a_im))
        * np.sqrt(np.sum(b_re * b_re + b_im * b_im))
        + 1e-8
    )
    return float(num / den)


def softmax(x, beta=1.0):
    z = beta * (np.asarray(x, dtype=np.float64) - np.max(x))
    e = np.exp(z)
    return (e / np.sum(e)).astype(np.float32)


def coherence_score(re, im):
    phase = np.arctan2(im, re)
    gy, gx = np.gradient(phase)
    grad_mag = np.mean(np.abs(gy)) + np.mean(np.abs(gx))
    lock_err = np.mean(np.abs((phase - THETA_STAR + np.pi) % (2 * np.pi) - np.pi))
    return float(max(0.0, 1.0 - 0.35 * grad_mag - 0.35 * lock_err))


def soft_renormalise_gains(gains):
    nudge = GAIN_SOFT_RATE * (GAIN_SOFT_TARGET - gains)
    gains = gains + nudge
    return np.clip(gains, GAIN_FLOOR, GAIN_CEIL).astype(np.float32)


# ====================== TEMPLATE LEARNING ======================
def update_templates_with_inhibition(state, winner, d_re, d_im, is_dream=False):
    plasticity = DREAM_PLASTICITY if is_dream else WAKE_PLASTICITY
    inhib = LATERAL_INHIBITION_STRENGTH * (0.5 if is_dream else 1.0)
    for k in range(N_CHANNELS):
        if k == winner:
            state["templates_re"][k] += plasticity * d_re
            state["templates_im"][k] += plasticity * d_im
        else:
            state["templates_re"][k] -= inhib * state["templates_re"][winner]
            state["templates_im"][k] -= inhib * state["templates_im"][winner]
        state["templates_re"][k], state["templates_im"][k] = normalize_complex(
            state["templates_re"][k], state["templates_im"][k]
        )


# ====================== INPUT RIPPLE ======================
def text_to_ripple(text, H, W, scale=0.10):
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    rng = np.random.default_rng(seed)
    delta_re = np.zeros((H, W), dtype=np.float32)
    delta_im = np.zeros((H, W), dtype=np.float32)
    cy, cx = H // 2, W // 2
    oy = (seed % 11) - 5
    ox = ((seed // 11) % 11) - 5
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    blob = np.exp(
        -((yy - (cy + oy)) ** 2 + (xx - (cx + ox)) ** 2) / (2.0 * 6.0 ** 2)
    ).astype(np.float32)
    phi = rng.uniform(-np.pi, np.pi)
    delta_re += scale * blob * np.cos(phi)
    delta_im += scale * blob * np.sin(phi)
    oy2 = ((seed // 101) % 15) - 7
    ox2 = ((seed // 211) % 15) - 7
    blob2 = np.exp(
        -((yy - (cy + oy2)) ** 2 + (xx - (cx + ox2)) ** 2) / (2.0 * 4.0 ** 2)
    ).astype(np.float32)
    phi2 = rng.uniform(-np.pi, np.pi)
    delta_re += 0.5 * scale * blob2 * np.cos(phi2)
    delta_im += 0.5 * scale * blob2 * np.sin(phi2)
    return delta_re, delta_im


# ====================== INIT ======================
def make_fixed_templates():
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    cy, cx = H // 2, W // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    templates_re = np.zeros((N_CHANNELS, H, W), dtype=np.float32)
    templates_im = np.zeros((N_CHANNELS, H, W), dtype=np.float32)

    # Quiet-Settle (0)
    g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * 7.0 ** 2)).astype(np.float32)
    templates_re[QUIET_SETTLE] = g * np.cos(0.2)
    templates_im[QUIET_SETTLE] = g * np.sin(0.2)

    # Alternate/Turbulent (1)
    g = np.exp(
        -((x - (cx + 10)) ** 2 + (y - (cy - 8)) ** 2) / (2.0 * 6.0 ** 2)
    ).astype(np.float32)
    templates_re[ALTERNATE_TURBULENT] = g * np.cos(1.4)
    templates_im[ALTERNATE_TURBULENT] = g * np.sin(1.4)

    # Primary Engagement (2)
    ring = np.exp(-((r - 14.0) ** 2) / (2.0 * 3.0 ** 2)).astype(np.float32)
    templates_re[PRIMARY_ENGAGEMENT] = ring * np.cos(THETA_STAR)
    templates_im[PRIMARY_ENGAGEMENT] = ring * np.sin(THETA_STAR)

    # Dream Consolidation (3)
    halo = np.exp(-((r - 22.0) ** 2) / (2.0 * 5.0 ** 2)).astype(np.float32)
    templates_re[DREAM_CONSOLIDATION] = halo * np.cos(THETA_STAR + 0.3)
    templates_im[DREAM_CONSOLIDATION] = halo * np.sin(THETA_STAR + 0.3)

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
        "replay_buffer": [],
    }


# ====================== PERSISTENCE ======================
def save_state(state, path=SAVE_PATH):
    buf_re = np.array([r[0] for r in state["replay_buffer"]], dtype=np.float32) \
        if state["replay_buffer"] else np.zeros((0, H, W), dtype=np.float32)
    buf_im = np.array([r[1] for r in state["replay_buffer"]], dtype=np.float32) \
        if state["replay_buffer"] else np.zeros((0, H, W), dtype=np.float32)

    np.savez(
        path,
        field_re=state["field_re"],
        field_im=state["field_im"],
        templates_re=state["templates_re"],
        templates_im=state["templates_im"],
        gains=state["gains"],
        usage=state["usage"],
        buf_re=buf_re,
        buf_im=buf_im,
    )


def load_state(path=SAVE_PATH):
    data = np.load(path)
    buf_re = data.get("buf_re", np.zeros((0, H, W), dtype=np.float32))
    buf_im = data.get("buf_im", np.zeros((0, H, W), dtype=np.float32))
    replay_buffer = list(zip(buf_re, buf_im))

    # Graceful upgrade if old 3-channel state is loaded
    saved_re = data["templates_re"].astype(np.float32)
    saved_im = data["templates_im"].astype(np.float32)
    if saved_re.shape[0] < N_CHANNELS:
        print("Upgrading saved state from 3 → 4 channels...")
        fresh_re, fresh_im = make_fixed_templates()
        pad_re = fresh_re[saved_re.shape[0]:]
        pad_im = fresh_im[saved_im.shape[0]:]
        saved_re = np.concatenate([saved_re, pad_re], axis=0)
        saved_im = np.concatenate([saved_im, pad_im], axis=0)

    saved_gains = data["gains"].astype(np.float32)
    if saved_gains.shape[0] < N_CHANNELS:
        saved_gains = np.concatenate([saved_gains, np.ones(N_CHANNELS - saved_gains.shape[0])])

    saved_usage = data["usage"].astype(np.float32)
    if saved_usage.shape[0] < N_CHANNELS:
        saved_usage = np.concatenate([saved_usage, np.zeros(N_CHANNELS - saved_usage.shape[0])])

    return {
        "field_re": data["field_re"].astype(np.float32),
        "field_im": data["field_im"].astype(np.float32),
        "templates_re": saved_re,
        "templates_im": saved_im,
        "gains": saved_gains,
        "usage": saved_usage,
        "replay_buffer": replay_buffer,
    }


# ====================== CORE TURN ======================
def frankie_turn(state, delta_re, delta_im, learn=True):
    d_re, d_im = normalize_complex(delta_re, delta_im)

    scores = np.array([
        overlap_score(d_re, d_im, state["templates_re"][k], state["templates_im"][k])
        for k in range(N_CHANNELS)
    ], dtype=np.float32)

    q = state["gains"] * scores
    weights = softmax(q, beta=SOFTMAX_BETA)
    winner = int(np.argmax(weights))

    bias_re = np.tensordot(weights, state["templates_re"], axes=(0, 0)).astype(np.float32)
    bias_im = np.tensordot(weights, state["templates_im"], axes=(0, 0)).astype(np.float32)

    re = state["field_re"] + ALPHA_IN * delta_re
    im = state["field_im"] + ALPHA_IN * delta_im

    for _ in range(N_RELAX_STEPS):
        re = re + LAMBDA_DIFF * laplacian(re) - LAMBDA_DAMP * re + LAMBDA_CHAN * bias_re
        im = im + LAMBDA_DIFF * laplacian(im) - LAMBDA_DAMP * im + LAMBDA_CHAN * bias_im
        re, im = phase_lock_project(re, im, THETA_STAR, LAMBDA_LOCK)

    state["field_re"] = re.astype(np.float32)
    state["field_im"] = im.astype(np.float32)

    coh = coherence_score(re, im)

    if learn:
        gain_update = 0.25 * weights.copy()
        gain_update[winner] += 0.75
        state["gains"] = state["gains"] + GAIN_LR * gain_update * coh
        state["gains"] = soft_renormalise_gains(state["gains"])

        state["usage"] = (
            (1.0 - USAGE_RHO) * state["usage"] + USAGE_RHO * weights
        ).astype(np.float32)

        update_templates_with_inhibition(state, winner, d_re, d_im, is_dream=False)

        energy = float(np.mean(delta_re ** 2 + delta_im ** 2))
        if energy > 1e-8:
            state["replay_buffer"].append((delta_re.copy(), delta_im.copy()))
            if len(state["replay_buffer"]) > REPLAY_BUFFER_SIZE:
                state["replay_buffer"].pop(0)

    return {
        "scores": scores.copy(),
        "weights": weights.copy(),
        "winning_channel": winner,
        "coherence_score": coh,
        "gains": state["gains"].copy(),
        "usage": state["usage"].copy(),
    }


# ====================== DREAM CONSOLIDATION LOOP ======================
def dream_loop(state, verbose=True):
    if verbose:
        print(f"\n{'─'*60}")
        print("💤 Frankie is dreaming...")
        print(f"{'─'*60}")

    state["gains"][DREAM_CONSOLIDATION] = min(
        state["gains"][DREAM_CONSOLIDATION] * 1.15, GAIN_CEIL
    )

    buf = state["replay_buffer"]
    replay_indices = list(reversed(range(min(REPLAY_TURNS, len(buf)))))

    for turn_idx in range(DREAM_TURNS):
        # Phase A: spontaneous relaxation
        zero_re = np.zeros((H, W), dtype=np.float32)
        zero_im = np.zeros((H, W), dtype=np.float32)
        result_a = frankie_turn(state, zero_re, zero_im, learn=False)

        # Phase B: replay a stored ripple
        replay_result = None
        if replay_indices and turn_idx < len(replay_indices):
            buf_idx = replay_indices[turn_idx % len(replay_indices)]
            r_re, r_im = buf[buf_idx]
            replay_re = (r_re * 0.4).astype(np.float32)
            replay_im = (r_im * 0.4).astype(np.float32)
            d_re, d_im = normalize_complex(replay_re, replay_im)

            scores = np.array([
                overlap_score(d_re, d_im, state["templates_re"][k], state["templates_im"][k])
                for k in range(N_CHANNELS)
            ], dtype=np.float32)

            q = state["gains"] * scores
            weights = softmax(q, beta=SOFTMAX_BETA)
            winner = int(np.argmax(weights))

            update_templates_with_inhibition(state, winner, d_re, d_im, is_dream=True)

            replay_result = {"winner": winner}

        if verbose:
            win_name = CHANNEL_NAMES[result_a["winning_channel"]]
            replay_str = f" | Replay winner: {CHANNEL_NAMES[replay_result['winner']]}" if replay_result else ""
            print(
                f" Dream {turn_idx + 1:2d} | "
                f"Relax win: {win_name:<22} | "
                f"Coh: {result_a['coherence_score']:.3f}{replay_str}"
            )

    state["gains"] = soft_renormalise_gains(state["gains"])

    for k in range(N_CHANNELS):
        state["templates_re"][k], state["templates_im"][k] = normalize_complex(
            state["templates_re"][k], state["templates_im"][k]
        )

    if verbose:
        print(f" Gains after dream: {np.round(state['gains'], 3)}")
        print(f"{'─'*60}")


# ====================== DRIFT MONITOR ======================
def check_topographic_resolution(state, save=True, verbose=True):
    results = {
        "timestamp": datetime.now().isoformat(),
        "pairs": {},
        "gains": [round(float(g), 4) for g in state["gains"]],
        "usage": [round(float(u), 4) for u in state["usage"]],
    }

    if verbose:
        print(f"\n{'─'*60}")
        print("🔬 TOPOGRAPHIC RESOLUTION CHECK")
        print(f"{'─'*60}")

    for i in range(N_CHANNELS):
        for j in range(i + 1, N_CHANNELS):
            sim = overlap_score(
                state["templates_re"][i], state["templates_im"][i],
                state["templates_re"][j], state["templates_im"][j]
            )
            label = f"{CHANNEL_NAMES[i]} vs {CHANNEL_NAMES[j]}"
            results["pairs"][label] = round(float(sim), 4)

            if verbose:
                bar_len = int((1.0 - abs(sim)) * 24)
                bar = "█" * bar_len
                status = (
                    "✓ distinct" if abs(sim) < 0.15 else
                    "~ drifting" if abs(sim) < 0.35 else
                    "✗ collapsing"
                )
                print(f" {label:<45} {sim:+.3f} {bar:<24} {status}")

    if verbose:
        print(f"\n Gains : {np.round(state['gains'], 3)}")
        print(f" Usage : {np.round(state['usage'], 3)}")

    if save:
        log = []
        try:
            with open(DRIFT_LOG_PATH, "r") as f:
                log = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        log.append(results)
        with open(DRIFT_LOG_PATH, "w") as f:
            json.dump(log, f, indent=2)

    return results


def print_drift_trend(n_sessions=5):
    try:
        with open(DRIFT_LOG_PATH, "r") as f:
            log = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(" No drift history yet — this is the first session.")
        return

    recent = log[-n_sessions:]
    if not recent:
        return

    print(f"\n{'─'*60}")
    print(f"📈 DRIFT TREND (last {len(recent)} sessions)")
    print(f"{'─'*60}")

    pair_labels = list(recent[0]["pairs"].keys())
    for label in pair_labels:
        values = [entry["pairs"].get(label) for entry in recent if label in entry["pairs"]]
        if len(values) < 2:
            continue
        trend = values[-1] - values[0]
        arrow = (
            "↓ diverging ✓" if trend < -0.02 else
            "↑ converging ✗" if trend > 0.02 else
            "→ stable"
        )
        sparkline = " ".join([f"{v:+.3f}" for v in values])
        print(f" {label:<45} {sparkline} {arrow}")


# ====================== MAIN ======================
if __name__ == "__main__":
    if os.path.exists(SAVE_PATH):
        state = load_state(SAVE_PATH)
        print("Frankie woke up with yesterday's blanket. 🛌")
    else:
        state = init_state()
        print("Fresh Frankie today. ✨")

    print_drift_trend(n_sessions=5)
    print("\nFrankie Channels Toy — Integrated v2\n")

def print_template_overlap_matrix(state, channel_names):
    """
    Full overlap matrix between all templates.
    +1.000 = identical
    ~0.000 = orthogonal / well separated
    negative = opposed / anti-correlated
    """
    print("\n────────────────────────────────────────────────────────────")
    print("TEMPLATE OVERLAP MATRIX")
    print("────────────────────────────────────────────────────────────")
    
    n = len(channel_names)
    overlaps = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        for j in range(n):
            overlaps[i, j] = overlap_score(
                state["templates_re"][i],
                state["templates_im"][i],
                state["templates_re"][j],
                state["templates_im"][j],
            )
    
    # Header
    header = " " * 20 + " ".join([f"{i:>8d}" for i in range(n)])
    print(header)
    
    # Rows
    for i in range(n):
        row = f"{channel_names[i][:18]:18s} "
        for j in range(n):
            row += f"{overlaps[i, j]:8.3f} "
        print(row)
    
    print("\nInterpretation:")
    print(" +1.000 = identical")
    print(" ~0.000 = orthogonal / well separated")
    print(" negative = opposed / anti-correlated")
    print(" |overlap| > 0.30 is worth watching")
    print(" |overlap| > 0.50 is probably too close for comfort")
    print("────────────────────────────────────────────────────────────\n")

    test_inputs = [
        "hello frankie good morning",
        "the moon smells blue today",
        "everything feels too much right now",
    ]

    for i, text in enumerate(test_inputs * 4, start=1):
        delta_re, delta_im = text_to_ripple(text, H, W, scale=0.10)
        result = frankie_turn(state, delta_re, delta_im, learn=True)
        win_name = CHANNEL_NAMES[result["winning_channel"]]
        print(
            f"Turn {i:2d} | "
            f"Text: {text[:32]:32} | "
            f"Scores: {np.round(result['scores'], 3)} | "
            f"Weights: {np.round(result['weights'], 3)} | "
            f"Win: {result['winning_channel']} ({win_name}) | "
            f"Gains: {np.round(result['gains'], 3)} | "
            f"Usage: {np.round(result['usage'], 3)} | "
            f"Coh: {result['coherence_score']:.3f}"
        )

    print(f"\n{'─'*60}")
    print("=== END OF WAKING SESSION ===")
    fav = int(np.argmax(state["gains"]))
    print(f"Favourite channel: {fav} ({CHANNEL_NAMES[fav]})")
    print(f"Gains : {np.round(state['gains'], 3)}")
    print(f"Usage : {np.round(state['usage'], 3)}")
    print(f"Replay buffer size: {len(state['replay_buffer'])} ripples stored")

    dream_loop(state, verbose=True)

    check_topographic_resolution(state, save=True, verbose=True)

    save_state(state)
    print("\n✅ State saved. Frankie is sleeping soundly.")

    # Optional visualisation
    try:
        print("\nDrawing Frankie's jelly brain... (close the window to exit)")
        fav = int(np.argmax(state["gains"]))
        fig = plt.figure(figsize=(16, 8))
        gs = plt.GridSpec(2, 4, figure=fig)

        ax_mag = fig.add_subplot(gs[0, 0])
        ax_pha = fig.add_subplot(gs[0, 1])
        magnitude = np.sqrt(state["field_re"] ** 2 + state["field_im"] ** 2)
        ax_mag.imshow(magnitude, cmap="viridis")
        ax_mag.set_title("Field Magnitude")
        phase = np.arctan2(state["field_im"], state["field_re"])
        ax_pha.imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax_pha.set_title("Field Phase")

        ax_gain = fig.add_subplot(gs[0, 2])
        ax_use = fig.add_subplot(gs[0, 3])
        colours = ["#4fc3f7", "#ff8a65", "#81c784", "#ce93d8"]
        short_names = ["QS", "AT", "PE", "DC"]
        ax_gain.bar(short_names, state["gains"], color=colours)
        ax_gain.set_title("Channel Gains")
        ax_use.bar(short_names, state["usage"], color=colours)
        ax_use.set_title("Channel Usage")

        for k in range(N_CHANNELS):
            ax_t = fig.add_subplot(gs[1, k])
            tmag = np.sqrt(state["templates_re"][k] ** 2 + state["templates_im"][k] ** 2)
            ax_t.imshow(tmag, cmap="magma")
            ax_t.set_title(f"Template: {CHANNEL_NAMES[k]}")

        plt.suptitle(
            f"Frankie — Favourite: {CHANNEL_NAMES[fav]} (gain {state['gains'][fav]:.3f})",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("matplotlib not available — skipping visualisation.")