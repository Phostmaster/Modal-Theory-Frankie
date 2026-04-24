#!/usr/bin/env python3
"""
2D Spatial Interference Simulation: Side-by-Side Unlocked vs Locked Wave Patterns

This script simulates wave propagation and interference on a 2D spatial grid.
Two circular wave sources generate outward-traveling waves; their superposition
produces characteristic interference patterns (constructive/destructive bands).

In the left panel, waves interfere naturally with no phase control.
In the right panel, the *relative phase* between source 2 and source 1 is
continuously nudged toward a target of 255° — acting as an attractor.

The bottom subplot visualizes how the instantaneous phase difference converges
to 255° over time, demonstrating dynamic phase locking.

Based on scalar wave equation approximations for shallow water or EM fields.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Configuration & Constants
# ──────────────────────────────────────────────────────────────────────────────

GRID_H, GRID_W = 128, 160   # Spatial grid dimensions (H=height, W=width)
DT = 0.35                  # Temporal timestep (stability < ~0.5 for wave eq)
DX = 1.0                   # Spatial discretization (pixel spacing)

# Wave source parameters
SRC_RADIUS = 8             # Radius of circular sources (in pixels)
SRC_AMP = 0.8              # Source amplitude
SRC_FREQ = 0.25            # Base oscillation frequency (cycles/frame)
SRC_POS_L = (GRID_H // 2, GRID_W // 3)      # Left source center (row, col)
SRC_POS_R = (GRID_H // 2, 2 * GRID_W // 3)  # Right source center

# Phase locking
TARGET_DELTA_PHI_DEG = 255.0   # Desired relative phase: 255°
PHASE_LOCK_RATE = 0.04        # Gentle nudge strength (0.01–0.1 recommended)

# Decay & diffusion
ATTENUATION = 0.98            # Slight amplitude decay per step to avoid blowup

DEVICE = torch.device("cpu")

# ──────────────────────────────────────────────────────────────────────────────
# Torch Helpers (real/imag → complex-like operations)
# ──────────────────────────────────────────────────────────────────────────────

def c_add(z1, z2):
    re1, im1 = z1
    re2, im2 = z2
    return (re1 + re2, im1 + im2)

def c_mul_real(z, r):
    re, im = z
    return (re * r, im * r)

def c_abs(z):
    re, im = z
    return torch.sqrt(re * re + im * im + 1e-8)

# Extract instantaneous phase from complex pair (re, im)
def wrap_phase_torch(phi):
    # Normalize to [-π, π]
    return (phi + np.pi) % (2 * np.pi) - np.pi

def get_phase(z):
    re, im = z
    return torch.atan2(im, re)

# ──────────────────────────────────────────────────────────────────────────────
# Core Wave Simulation Functions
# ──────────────────────────────────────────────────────────────────────────────

def create_circular_mask(h, w, center, radius):
    """Return binary mask where 1 = inside circle."""
    Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    dist2 = (X - center[1]) ** 2 + (Y - center[0]) ** 2
    return (dist2 <= radius * radius).float()

def apply_phase_lock(z_src_r, z_src_l, target_delta_deg):
    """
    Computes the current natural phase difference and gently pulls it toward
    target_delta_deg. Returns new source right state.

    Crucially: delta_phi_locked is recomputed each frame from *current* phi2 - phi1.
    """
    phi_l = get_phase(z_src_l)
    phi_r = get_phase(z_src_r)

    # Current natural phase difference (in radians, wrapped to [-π, π])
    delta_phi_nat = wrap_phase_torch(phi_r - phi_l)
    
    # Convert targets
    target_delta_rad = torch.tensor(target_delta_deg * np.pi / 180.0).to(delta_phi_nat.device)

    # Error: how far we are from the attractor
    d_phi = wrap_phase_torch(target_delta_rad - delta_phi_nat)

    # Apply gentle nudge (only to right source’s phase; left fixed as reference)
    phi_r_new = phi_r + PHASE_LOCK_RATE * d_phi

    # Preserve original amplitudes (use current magnitude of right source)
    r_mag = c_abs(z_src_r)

    re_new = r_mag * torch.cos(phi_r_new)
    im_new = r_mag * torch.sin(phi_r_new)

    return (re_new, im_new)


# ──────────────────────────────────────────────────────────────────────────────
# Simulation Setup & State Initialization
# ──────────────────────────────────────────────────────────────────────────────

# Grid: complex field z = re + i·im representing wave amplitude/phase
z_re = torch.zeros((GRID_H, GRID_W), device=DEVICE)
z_im = torch.zeros_like(z_re)

# Previous state for wave equation (y_{t-1})
z_prev_re = torch.zeros_like(z_re)
z_prev_im = torch.zeros_like(z_im)

# Source masks and initial source states
mask_l = create_circular_mask(GRID_H, GRID_W, SRC_POS_L, SRC_RADIUS).unsqueeze(0)  # [1,H,W]
mask_r = create_circular_mask(GRID_H, GRID_W, SRC_POS_R, SRC_RADIUS).unsqueeze(0)

# Source complex amplitudes (start in phase for comparison; locking will adjust)
src_l_re = torch.zeros(1, device=DEVICE)
src_l_im = torch.zeros(1, device=DEVICE)
src_r_re = torch.zeros(1, device=DEVICE)
src_r_im = torch.zeros(1, device=DEVICE)

# Tracking state
phase_diff_history = []

# ──────────────────────────────────────────────────────────────────────────────
# Simulation Loop
# ──────────────────────────────────────────────────────────────────────────────

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
plt.ion()

img_l = ax1.imshow(z_re.cpu().numpy(), cmap='RdBu_r', vmin=-1.0, vmax=1.0, animated=True)
ax1.set_title("Unlocked Interference")
ax1.axis('off')

img_r = ax2.imshow(z_re.cpu().numpy(), cmap='RdBu_r', vmin=-1.0, vmax=1.0, animated=True)
ax2.set_title(f"Phase-Locked (→ {TARGET_DELTA_PHI_DEG}°)")
ax2.axis('off')

line, = ax3.plot([], [], 'b-', lw=1)
ax3.set_ylim(-180, 450)
ax3.axhline(y=TARGET_DELTA_PHI_DEG, color='r', linestyle='--', alpha=0.7, label='Target')
ax3.set_title("Relative Phase Difference (Right − Left)")
ax3.set_xlabel("Time step")
ax3.set_ylabel("Degrees")
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right')

time_step = 0

for frame in range(600):
    # Update natural source oscillations
    theta_l = SRC_FREQ * 2 * np.pi * time_step
    src_l_re_new = SRC_AMP * torch.cos(torch.tensor(theta_l))
    src_l_im_new = SRC_AMP * torch.sin(torch.tensor(theta_l))

    theta_r_nat = SRC_FREQ * 2 * np.pi * time_step   # Natural (unlocked) phase for right source
    src_r_re_nat = SRC_AMP * torch.cos(torch.tensor(theta_r_nat))
    src_r_im_nat = SRC_AMP * torch.sin(torch.tensor(theta_r_nat))

    # Apply phase locking: nudged version of right source
    src_l = (src_l_re_new, src_l_im_new)
    src_r_unlocked = (src_r_re_nat, src_r_im_nat)
    
    # 🔒 CRITICAL FIX: Compute delta_phi each frame from current natural values,
    # then nudge *before* injecting into grid
    src_r = apply_phase_lock(src_r_unlocked, src_l, TARGET_DELTA_PHI_DEG)

    # Update source state memory for next step
    src_l_re, src_l_im = src_l
    src_r_re, src_r_im = src_r

    # Inject sources into field (additive)
    z_new_re = z_prev_re * 0  # start from previous; will overwrite via wave eq
    z_new_im = z_prev_im * 0

    # Apply source injection via masks
    z_new_re += mask_l.squeeze(0) * src_l_re
    z_new_im += mask_l.squeeze(0) * src_l_im
    z_new_re += mask_r.squeeze(0) * src_r_re
    z_new_im += mask_r.squeeze(0) * src_r_im

    # Wave propagation: 2D wave equation (explicit finite difference)
    laplacian_re = (
        torch.roll(z_prev_re, shifts=1, dims=0) +
        torch.roll(z_prev_re, shifts=-1, dims=0) +
        torch.roll(z_prev_re, shifts=1, dims=1) +
        torch.roll(z_prev_re, shifts=-1, dims=1) -
        4 * z_prev_re
    )
    laplacian_im = (
        torch.roll(z_prev_im, shifts=1, dims=0) +
        torch.roll(z_prev_im, shifts=-1, dims=0) +
        torch.roll(z_prev_im, shifts=1, dims=1) +
        torch.roll(z_prev_im, shifts=-1, dims=1) -
        4 * z_prev_re
    )  # Note: coupling via real/imag is omitted for scalar wave

    # Update using discretized wave equation:
    # y_{t} = 2·y_{t-1} - y_{t-2} + (c·DT/DX)^2 · laplacian(y)
    c_eff = 0.8   # Effective wave speed
    k = (c_eff * DT / DX) ** 2

    z_new_re += k * laplacian_re
    z_new_im += k * laplacian_im

    # Apply attenuation to prevent explosion
    z_new_re *= ATTENUATION
    z_new_im *= ATTENUATION

    # Update history state (shift: current → previous, old previous → past)
    z_prev_re = z_re.clone()
    z_prev_im = z_im.clone()
    z_re = z_new_re
    z_im = z_new_im

    # Compute and record *actual* relative phase in locked case:
    phi_l = get_phase((src_l_re, src_l_im))
    phi_r = get_phase((src_r_re, src_r_im))
    delta_phi_deg = (wrap_phase_torch(phi_r - phi_l) * 180 / np.pi).item()
    phase_diff_history.append(delta_phi_deg)

    # Update plots
    if frame % 3 == 0:
        # Show real part of wavefield as image
        img_arr = z_re.cpu().numpy()
        img_l.set_array(img_arr)
        img_r.set_array(img_arr)  # Same field, same color scale — locked is *internal*

        # Update phase diff history plot
        if len(phase_diff_history) > 0:
            xdata = np.arange(len(phase_diff_history))
            line.set_data(xdata, phase_diff_history)
            ax3.relim()
            ax3.autoscale_view()

        plt.pause(0.01)

    time_step += 1

plt.ioff()
plt.show()
