import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =============================================================================
# MODE
# =============================================================================
# "animate" = normal visual toy + printouts + CSV logging
# "sweep"   = no animation, run parameter sweeps and save summary CSV
MODE = "animate"

# =============================================================================
# Configuration
# =============================================================================
GRID_SIZE_H = 150
GRID_SIZE_W = 160

DT = 0.18
INTERVAL_MS = 40
N_STEPS_HISTORY = 220
MAX_FRAMES_ANIMATE = 1200

# Field evolution
DIFFUSION = 0.22
DAMPING = 0.030
SELF_RESTORE = 0.010

# Source driving
DRIVE_AMPLITUDE = 0.95
DRIVE_OMEGA = 0.42
SOURCE_SIGMA = 5.0

# Coherence shell / attractor
LOCK_STRENGTH = 0.0025
THETA_STAR_DEG = 255.0
THETA_STAR = np.deg2rad(THETA_STAR_DEG)

# Shell geometry
CY = GRID_SIZE_H // 2
CX = GRID_SIZE_W // 2
INNER_RADIUS = 28.0
OUTER_RADIUS = 58.0
SHELL_EDGE_SOFTNESS = 4.0

# Visual scaling
VMIN = -1.1
VMAX = 1.1

# Logging / diagnostics
PRINT_EVERY = 50
LOG_DIR = Path("coherence_shell_logs")
ANIMATE_CSV = LOG_DIR / "coherence_shell_timeseries.csv"
SWEEP_CSV = LOG_DIR / "coherence_shell_sweep_summary.csv"

# Threshold-style diagnostics
SHELL_CAPTURE_THRESHOLD_DEG = 10.0
INTERIOR_CAPTURE_THRESHOLD_DEG = 10.0
ROLLING_WINDOW = 100  # for rolling averages / capture fraction

# Sweep settings
SWEEP_LOCK_STRENGTHS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
SWEEP_INNER_RADII = [20.0, 28.0, 36.0]
SWEEP_OUTER_RADII = [46.0, 58.0, 70.0]
SWEEP_DRIVE_AMPLITUDES = [0.70, 0.95, 1.20]
SWEEP_FRAMES = 700

# =============================================================================
# Grid and geometry
# =============================================================================
y, x = np.meshgrid(np.arange(GRID_SIZE_H), np.arange(GRID_SIZE_W), indexing="ij")
r_center = np.sqrt((x - CX) ** 2 + (y - CY) ** 2)

source1 = (CY - 18, CX - 22)
source2 = (CY + 18, CX + 22)

r1 = np.sqrt((x - source1[1]) ** 2 + (y - source1[0]) ** 2)
r2 = np.sqrt((x - source2[1]) ** 2 + (y - source2[0]) ** 2)

source_mask1 = np.exp(-(r1 ** 2) / (2.0 * SOURCE_SIGMA ** 2))
source_mask2 = np.exp(-(r2 ** 2) / (2.0 * SOURCE_SIGMA ** 2))

# =============================================================================
# Helpers
# =============================================================================
def ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_shell_masks(inner_radius, outer_radius, softness):
    inner_gate = sigmoid((r_center - inner_radius) / softness)
    outer_gate = sigmoid((outer_radius - r_center) / softness)
    shell_mask = (inner_gate * outer_gate).astype(np.float32)

    interior_region = r_center < inner_radius
    shell_region = (r_center >= inner_radius) & (r_center <= outer_radius)
    exterior_region = r_center > outer_radius
    return shell_mask, interior_region, shell_region, exterior_region


def laplacian(z):
    return (
        np.roll(z, 1, axis=0)
        + np.roll(z, -1, axis=0)
        + np.roll(z, 1, axis=1)
        + np.roll(z, -1, axis=1)
        - 4.0 * z
    )


def wrap_phase(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def project_phase_with_lock(phi, chi, lock_mask, theta_star, lock_strength):
    mag = np.sqrt(phi * phi + chi * chi) + 1e-8
    phase = np.arctan2(chi, phi)

    dtheta = wrap_phase(phase - theta_star)
    new_phase = phase - lock_strength * lock_mask * dtheta

    phi_new = mag * np.cos(new_phase)
    chi_new = mag * np.sin(new_phase)
    return phi_new, chi_new


def phase_error_field(phi, chi):
    phase = np.arctan2(chi, phi)
    return np.abs(wrap_phase(phase - THETA_STAR))


def rolling_mean(values, window):
    if not values:
        return float("nan")
    arr = np.asarray(values[-window:], dtype=np.float64)
    return float(arr.mean())


def capture_fraction(values_deg, threshold_deg, window):
    if not values_deg:
        return float("nan")
    arr = np.asarray(values_deg[-window:], dtype=np.float64)
    return float(np.mean(arr < threshold_deg))


def classify_run(stats):
    shell_frac = stats["rolling_capture_fraction_shell"]
    interior_t = stats["time_to_interior_below_deg"]
    capture_eps = stats["shell_capture_episodes"]
    final_ratio = stats["containment_ratio"]

    if capture_eps == 0 and shell_frac < 0.05:
        return "No capture"

    if capture_eps >= 1 and 0.05 <= shell_frac < 0.80 and interior_t is None:
        return "Metastable shell"

    if shell_frac >= 0.80 and interior_t is None:
        return "Stable shell"

    if shell_frac >= 0.80 and interior_t is not None:
        if final_ratio >= 3.0:
            return "Strong protected zone"
        return "Shell + interior protection"

    if capture_eps >= 1 and interior_t is not None:
        if final_ratio >= 3.0:
            return "Strong protected zone"
        return "Shell + interior protection"

    return "Transitional / mixed"


# =============================================================================
# Simulator
# =============================================================================
class CoherenceShellSim:
    def __init__(
        self,
        lock_strength=LOCK_STRENGTH,
        inner_radius=INNER_RADIUS,
        outer_radius=OUTER_RADIUS,
        drive_amplitude=DRIVE_AMPLITUDE,
        softness=SHELL_EDGE_SOFTNESS,
    ):
        self.lock_strength = float(lock_strength)
        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)
        self.drive_amplitude = float(drive_amplitude)
        self.softness = float(softness)

        self.shell_mask, self.interior_region, self.shell_region, self.exterior_region = make_shell_masks(
            self.inner_radius, self.outer_radius, self.softness
        )

        self.phi_free = np.zeros((GRID_SIZE_H, GRID_SIZE_W), dtype=np.float32)
        self.chi_free = np.zeros((GRID_SIZE_H, GRID_SIZE_W), dtype=np.float32)
        self.phi_lock = np.zeros((GRID_SIZE_H, GRID_SIZE_W), dtype=np.float32)
        self.chi_lock = np.zeros((GRID_SIZE_H, GRID_SIZE_W), dtype=np.float32)

        self.phase_err_history_rad = []

        self.history_interior_deg = []
        self.history_shell_deg = []
        self.history_shell_std_deg = []
        self.history_exterior_deg = []
        self.history_containment_ratio = []

        self.t = 0.0
        self.frame_counter = 0

        self.peak_containment_ratio = -np.inf
        self.peak_containment_frame = None

        self.time_to_shell_below_deg = None
        self.time_to_interior_below_deg = None

        self.shell_capture_episodes = 0
        self._shell_was_captured_prev = False

    def advance_one_step(self):
        drive_phi = self.drive_amplitude * (
            np.cos(DRIVE_OMEGA * self.t) * source_mask1
            + np.cos(DRIVE_OMEGA * self.t + 0.8) * source_mask2
        )
        drive_chi = self.drive_amplitude * (
            np.sin(DRIVE_OMEGA * self.t) * source_mask1
            + np.sin(DRIVE_OMEGA * self.t + 0.8) * source_mask2
        )

        self.phi_free[:] = self.phi_free + DT * (
            DIFFUSION * laplacian(self.phi_free)
            - DAMPING * self.phi_free
            - SELF_RESTORE * (self.phi_free ** 3)
            + drive_phi
        )
        self.chi_free[:] = self.chi_free + DT * (
            DIFFUSION * laplacian(self.chi_free)
            - DAMPING * self.chi_free
            - SELF_RESTORE * (self.chi_free ** 3)
            + drive_chi
        )

        self.phi_lock[:] = self.phi_lock + DT * (
            DIFFUSION * laplacian(self.phi_lock)
            - DAMPING * self.phi_lock
            - SELF_RESTORE * (self.phi_lock ** 3)
            + drive_phi
        )
        self.chi_lock[:] = self.chi_lock + DT * (
            DIFFUSION * laplacian(self.chi_lock)
            - DAMPING * self.chi_lock
            - SELF_RESTORE * (self.chi_lock ** 3)
            + drive_chi
        )

        if self.lock_strength > 0.0:
            phi_new, chi_new = project_phase_with_lock(
                self.phi_lock, self.chi_lock, self.shell_mask, THETA_STAR, self.lock_strength
            )
            self.phi_lock[:] = phi_new
            self.chi_lock[:] = chi_new

        self.t += DT
        self.frame_counter += 1

        shell_err_rad = self.shell_phase_error_rad()
        self.phase_err_history_rad.append(shell_err_rad)

        stats = self.current_stats()

        self.history_interior_deg.append(stats["interior_mean_abs_err_deg"])
        self.history_shell_deg.append(stats["shell_mean_abs_err_deg"])
        self.history_shell_std_deg.append(stats["shell_std_deg"])
        self.history_exterior_deg.append(stats["exterior_mean_abs_err_deg"])
        self.history_containment_ratio.append(stats["containment_ratio"])

        if stats["containment_ratio"] > self.peak_containment_ratio:
            self.peak_containment_ratio = stats["containment_ratio"]
            self.peak_containment_frame = self.frame_counter

        if self.time_to_shell_below_deg is None and stats["shell_mean_abs_err_deg"] < SHELL_CAPTURE_THRESHOLD_DEG:
            self.time_to_shell_below_deg = self.frame_counter

        if self.time_to_interior_below_deg is None and stats["interior_mean_abs_err_deg"] < INTERIOR_CAPTURE_THRESHOLD_DEG:
            self.time_to_interior_below_deg = self.frame_counter

        shell_is_captured_now = stats["shell_mean_abs_err_deg"] < SHELL_CAPTURE_THRESHOLD_DEG
        if shell_is_captured_now and not self._shell_was_captured_prev:
            self.shell_capture_episodes += 1
        self._shell_was_captured_prev = shell_is_captured_now

    def shell_phase_error_rad(self):
        err = phase_error_field(self.phi_lock, self.chi_lock)
        return float(err[self.shell_region].mean())

    def current_stats(self):
        err = phase_error_field(self.phi_lock, self.chi_lock)

        interior = err[self.interior_region]
        shell = err[self.shell_region]
        exterior = err[self.exterior_region]

        mean_interior_deg = float(np.rad2deg(interior.mean()))
        mean_shell_deg = float(np.rad2deg(shell.mean()))
        std_shell_deg = float(np.rad2deg(shell.std()))
        mean_exterior_deg = float(np.rad2deg(exterior.mean()))

        containment_ratio = mean_exterior_deg / max(mean_interior_deg, 1e-9)

        stats = {
            "interior_mean_abs_err_deg": mean_interior_deg,
            "shell_mean_abs_err_deg": mean_shell_deg,
            "shell_std_deg": std_shell_deg,
            "exterior_mean_abs_err_deg": mean_exterior_deg,
            "containment_ratio": containment_ratio,
        }

        stats["rolling_mean_containment_ratio"] = rolling_mean(self.history_containment_ratio, ROLLING_WINDOW)
        stats["rolling_capture_fraction_shell"] = capture_fraction(
            self.history_shell_deg, SHELL_CAPTURE_THRESHOLD_DEG, ROLLING_WINDOW
        )
        stats["rolling_capture_fraction_interior"] = capture_fraction(
            self.history_interior_deg, INTERIOR_CAPTURE_THRESHOLD_DEG, ROLLING_WINDOW
        )

        stats["peak_containment_ratio"] = (
            self.peak_containment_ratio if np.isfinite(self.peak_containment_ratio) else float("nan")
        )
        stats["peak_containment_frame"] = self.peak_containment_frame
        stats["time_to_shell_below_deg"] = self.time_to_shell_below_deg
        stats["time_to_interior_below_deg"] = self.time_to_interior_below_deg
        stats["shell_capture_episodes"] = self.shell_capture_episodes
        stats["run_class"] = classify_run(stats)

        return stats


# =============================================================================
# Printing
# =============================================================================
def print_stats(frame_number, stats, lock_strength):
    lock_msg = "LOCK ACTIVE" if lock_strength > 0.0 else "LOCK OFF"
    print(
        f"[frame {frame_number:4d}] "
        f"interior_mean_abs_err = {stats['interior_mean_abs_err_deg']:8.3f} deg | "
        f"shell_mean_abs_err = {stats['shell_mean_abs_err_deg']:8.3f} deg | "
        f"shell_std = {stats['shell_std_deg']:8.3f} deg | "
        f"exterior_mean_abs_err = {stats['exterior_mean_abs_err_deg']:8.3f} deg | "
        f"containment_ratio = {stats['containment_ratio']:7.3f} | "
        f"roll_mean_ratio = {stats['rolling_mean_containment_ratio']:7.3f} | "
        f"shell_capture_frac = {stats['rolling_capture_fraction_shell']:5.3f} | "
        f"interior_capture_frac = {stats['rolling_capture_fraction_interior']:5.3f} | "
        f"capture_eps = {stats['shell_capture_episodes']:3d} | "
        f"{lock_msg}"
    )


def print_final_summary(sim):
    stats = sim.current_stats()
    print("\nFinal statistics:")
    print_stats(sim.frame_counter, stats, sim.lock_strength)

    shell_t = stats["time_to_shell_below_deg"]
    interior_t = stats["time_to_interior_below_deg"]

    print("\nDerived diagnostics:")
    print(f"  peak_containment_ratio: {stats['peak_containment_ratio']:.3f}")
    print(f"  peak_containment_frame: {stats['peak_containment_frame']}")
    print(f"  mean_containment_ratio_last_{ROLLING_WINDOW}: {stats['rolling_mean_containment_ratio']:.3f}")
    print(f"  shell_capture_fraction_last_{ROLLING_WINDOW}: {stats['rolling_capture_fraction_shell']:.3f}")
    print(f"  interior_capture_fraction_last_{ROLLING_WINDOW}: {stats['rolling_capture_fraction_interior']:.3f}")
    print(f"  shell_capture_episodes: {stats['shell_capture_episodes']}")
    print(f"  time_to_shell_below_{SHELL_CAPTURE_THRESHOLD_DEG:.1f}deg: {shell_t if shell_t is not None else 'never'}")
    print(f"  time_to_interior_below_{INTERIOR_CAPTURE_THRESHOLD_DEG:.1f}deg: {interior_t if interior_t is not None else 'never'}")
    print(f"  run_class: {stats['run_class']}")


# =============================================================================
# CSV logging
# =============================================================================
def write_timeseries_header(path):
    ensure_log_dir()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame",
            "time",
            "lock_strength",
            "inner_radius",
            "outer_radius",
            "drive_amplitude",
            "interior_mean_abs_err_deg",
            "shell_mean_abs_err_deg",
            "shell_std_deg",
            "exterior_mean_abs_err_deg",
            "containment_ratio",
            f"mean_containment_ratio_last_{ROLLING_WINDOW}",
            f"shell_capture_fraction_last_{ROLLING_WINDOW}",
            f"interior_capture_fraction_last_{ROLLING_WINDOW}",
            f"time_to_shell_below_{SHELL_CAPTURE_THRESHOLD_DEG:.1f}deg",
            f"time_to_interior_below_{INTERIOR_CAPTURE_THRESHOLD_DEG:.1f}deg",
            "shell_capture_episodes",
            "peak_containment_ratio",
            "peak_containment_frame",
            "run_class",
            "lock_status",
        ])


def append_timeseries_row(path, sim):
    stats = sim.current_stats()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            sim.frame_counter,
            sim.t,
            sim.lock_strength,
            sim.inner_radius,
            sim.outer_radius,
            sim.drive_amplitude,
            stats["interior_mean_abs_err_deg"],
            stats["shell_mean_abs_err_deg"],
            stats["shell_std_deg"],
            stats["exterior_mean_abs_err_deg"],
            stats["containment_ratio"],
            stats["rolling_mean_containment_ratio"],
            stats["rolling_capture_fraction_shell"],
            stats["rolling_capture_fraction_interior"],
            stats["time_to_shell_below_deg"],
            stats["time_to_interior_below_deg"],
            stats["shell_capture_episodes"],
            stats["peak_containment_ratio"],
            stats["peak_containment_frame"],
            stats["run_class"],
            "LOCK ACTIVE" if sim.lock_strength > 0.0 else "LOCK OFF",
        ])


def write_sweep_header(path):
    ensure_log_dir()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "lock_strength",
            "inner_radius",
            "outer_radius",
            "drive_amplitude",
            "frames",
            "final_interior_mean_abs_err_deg",
            "final_shell_mean_abs_err_deg",
            "final_shell_std_deg",
            "final_exterior_mean_abs_err_deg",
            "final_containment_ratio",
            f"mean_containment_ratio_last_{ROLLING_WINDOW}",
            f"shell_capture_fraction_last_{ROLLING_WINDOW}",
            f"interior_capture_fraction_last_{ROLLING_WINDOW}",
            f"time_to_shell_below_{SHELL_CAPTURE_THRESHOLD_DEG:.1f}deg",
            f"time_to_interior_below_{INTERIOR_CAPTURE_THRESHOLD_DEG:.1f}deg",
            "shell_capture_episodes",
            "peak_containment_ratio",
            "peak_containment_frame",
            "best_interior_mean_abs_err_deg",
            "best_shell_mean_abs_err_deg",
            "best_exterior_mean_abs_err_deg",
            "run_class",
        ])


def append_sweep_row(path, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# =============================================================================
# Animation mode
# =============================================================================
def run_animation_mode():
    sim = CoherenceShellSim(
        lock_strength=LOCK_STRENGTH,
        inner_radius=INNER_RADIUS,
        outer_radius=OUTER_RADIUS,
        drive_amplitude=DRIVE_AMPLITUDE,
        softness=SHELL_EDGE_SOFTNESS,
    )

    write_timeseries_header(ANIMATE_CSV)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 5.0))
    plt.tight_layout(pad=2.0)

    im_free = ax1.imshow(sim.phi_free, cmap="RdBu_r", vmin=VMIN, vmax=VMAX, animated=True)
    ax1.set_title("Reference: no coherence shell")
    ax1.axis("off")

    im_lock = ax2.imshow(sim.phi_lock, cmap="RdBu_r", vmin=VMIN, vmax=VMAX, animated=True)
    ax2.set_title(f"Coherence shell: θ*={int(THETA_STAR_DEG)}°, k={sim.lock_strength:.2f}")
    ax2.axis("off")

    for ax in (ax1, ax2):
        c1 = plt.Circle((CX, CY), sim.inner_radius, color="k", fill=False, linewidth=0.8, alpha=0.55)
        c2 = plt.Circle((CX, CY), sim.outer_radius, color="k", fill=False, linewidth=0.8, alpha=0.55)
        ax.add_patch(c1)
        ax.add_patch(c2)

    phase_line, = ax3.plot([], [], color="tab:blue", linewidth=1.2, label="Locked shell mean |Δθ|")
    ax3.axhline(0.0, color="k", linewidth=0.6, alpha=0.4)
    ax3.set_xlim(0, N_STEPS_HISTORY)
    ax3.set_ylim(0, np.pi)
    ax3.set_xlabel("Time step")
    ax3.set_ylabel("Mean |phase error| in shell (rad)")
    ax3.set_title("Shell phase error vs 255° attractor")
    ax3.legend(loc="upper right", frameon=False)

    status_text = ax3.text(
        0.02, 0.95, "", transform=ax3.transAxes,
        ha="left", va="top", fontsize=9
    )

    def animate(_frame):
        sim.advance_one_step()

        im_free.set_array(sim.phi_free)
        im_lock.set_array(sim.phi_lock)

        recent = sim.phase_err_history_rad[-N_STEPS_HISTORY:]
        xs = np.arange(len(recent))
        phase_line.set_data(xs, recent)

        if len(recent) > 0:
            if sim.lock_strength > 0.0:
                ymax = max(0.35, 1.15 * max(recent))
                ax3.set_ylim(0, ymax)
            else:
                ax3.set_ylim(0, np.pi)

        err = sim.shell_phase_error_rad()
        status_text.set_text(
            f"t = {sim.t:6.2f}\n"
            f"shell mean |Δθ| = {err:0.4f} rad\n"
            f"shell lock = {sim.lock_strength:0.2f}"
        )

        append_timeseries_row(ANIMATE_CSV, sim)

        if sim.frame_counter % PRINT_EVERY == 0:
            print_stats(sim.frame_counter, sim.current_stats(), sim.lock_strength)

        return im_free, im_lock, phase_line, status_text

    def on_close(_event):
        if sim.frame_counter > 0:
            print_final_summary(sim)
            print(f"\nCSV written to: {ANIMATE_CSV}")

    fig.canvas.mpl_connect("close_event", on_close)

    _ani = FuncAnimation(
        fig,
        animate,
        frames=MAX_FRAMES_ANIMATE,
        interval=INTERVAL_MS,
        blit=False,
        cache_frame_data=False
    )

    plt.show()


# =============================================================================
# Sweep mode
# =============================================================================
def run_sweep_mode():
    write_sweep_header(SWEEP_CSV)

    run_count = 0

    for lock_strength in SWEEP_LOCK_STRENGTHS:
        for inner_radius in SWEEP_INNER_RADII:
            for outer_radius in SWEEP_OUTER_RADII:
                if outer_radius <= inner_radius:
                    continue
                for drive_amplitude in SWEEP_DRIVE_AMPLITUDES:
                    run_count += 1

                    sim = CoherenceShellSim(
                        lock_strength=lock_strength,
                        inner_radius=inner_radius,
                        outer_radius=outer_radius,
                        drive_amplitude=drive_amplitude,
                        softness=SHELL_EDGE_SOFTNESS,
                    )

                    best_interior = float("inf")
                    best_shell = float("inf")
                    best_exterior = float("inf")

                    for _ in range(SWEEP_FRAMES):
                        sim.advance_one_step()
                        stats = sim.current_stats()

                        best_interior = min(best_interior, stats["interior_mean_abs_err_deg"])
                        best_shell = min(best_shell, stats["shell_mean_abs_err_deg"])
                        best_exterior = min(best_exterior, stats["exterior_mean_abs_err_deg"])

                    final_stats = sim.current_stats()

                    append_sweep_row(SWEEP_CSV, [
                        lock_strength,
                        inner_radius,
                        outer_radius,
                        drive_amplitude,
                        SWEEP_FRAMES,
                        final_stats["interior_mean_abs_err_deg"],
                        final_stats["shell_mean_abs_err_deg"],
                        final_stats["shell_std_deg"],
                        final_stats["exterior_mean_abs_err_deg"],
                        final_stats["containment_ratio"],
                        final_stats["rolling_mean_containment_ratio"],
                        final_stats["rolling_capture_fraction_shell"],
                        final_stats["rolling_capture_fraction_interior"],
                        final_stats["time_to_shell_below_deg"],
                        final_stats["time_to_interior_below_deg"],
                        final_stats["shell_capture_episodes"],
                        final_stats["peak_containment_ratio"],
                        final_stats["peak_containment_frame"],
                        best_interior,
                        best_shell,
                        best_exterior,
                        final_stats["run_class"],
                    ])

                    print(
                        f"[run {run_count:03d}] "
                        f"k={lock_strength:.5f}, rin={inner_radius:.1f}, rout={outer_radius:.1f}, drive={drive_amplitude:.2f} | "
                        f"final_ratio={final_stats['containment_ratio']:.3f} | "
                        f"peak_ratio={final_stats['peak_containment_ratio']:.3f} | "
                        f"capture_eps={final_stats['shell_capture_episodes']} | "
                        f"class={final_stats['run_class']}"
                    )

    print(f"\nSweep complete. Summary CSV written to: {SWEEP_CSV}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    if MODE == "animate":
        run_animation_mode()
    elif MODE == "sweep":
        run_sweep_mode()
    else:
        raise ValueError("MODE must be 'animate' or 'sweep'")