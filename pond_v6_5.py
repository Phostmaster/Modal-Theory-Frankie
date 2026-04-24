import math
import os
import re
import hashlib
from typing import Optional

import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "frankie_pond_v6_5.pt"

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


# ─────────────────────────────────────────────────────────────────────────────
# Complex helpers
# ─────────────────────────────────────────────────────────────────────────────

def c_abs(z):
    re, im = z
    return torch.sqrt(re * re + im * im + 1e-8)


def c_add(a, b):
    ar, ai = a
    br, bi = b
    return (ar + br, ai + bi)


def c_scale(a, s):
    ar, ai = a
    return (ar * s, ai * s)


def c_mul(a, b):
    ar, ai = a
    br, bi = b
    return (ar * br - ai * bi, ar * bi + ai * br)


def c_dot(a, b):
    ar, ai = a
    br, bi = b
    return (ar * br + ai * bi).sum()


def c_normalize(z, eps=1e-8):
    mag = c_abs(z).mean().clamp_min(eps)
    return c_scale(z, 1.0 / mag)


def wrap_phase_torch(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


# ─────────────────────────────────────────────────────────────────────────────
# Field helpers
# ─────────────────────────────────────────────────────────────────────────────

def spatial_coherence(theta):
    gy, gx = torch.gradient(theta, dim=(2, 3))
    return torch.mean(torch.sqrt(gx * gx + gy * gy + 1e-8))


def phase_lock_error(z, theta_star):
    re, im = z
    theta = torch.atan2(im, re)
    d = wrap_phase_torch(theta_star - theta)
    return d.abs().mean()


def admissibility_mask(z, theta_star, coherence_cut=1.40, lock_cut=2.20):
    re, im = z
    theta = torch.atan2(im, re)
    gy, gx = torch.gradient(theta, dim=(2, 3))
    gradmag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    lock = wrap_phase_torch(theta_star - theta).abs()

    m1 = torch.exp(-(gradmag / coherence_cut) ** 2)
    m2 = torch.exp(-(lock / lock_cut) ** 2)
    return (m1 * m2).clamp(0.0, 1.0)


def admissibility_project(z, theta_star):
    return c_scale(z, admissibility_mask(z, theta_star))


def overlap_score(a, b):
    num = c_dot(a, b)
    den = torch.sqrt(c_dot(a, a) * c_dot(b, b) + 1e-8)
    return num / den


def nudge_toward_phase(z, theta_star, rate=0.01):
    re, im = z
    r = torch.sqrt(re * re + im * im + 1e-8)
    theta = torch.atan2(im, re)
    d = wrap_phase_torch(theta_star - theta)
    theta2 = theta + rate * d
    return (r * torch.cos(theta2), r * torch.sin(theta2))


# ─────────────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────────────

TOKEN_GROUPS = {
    "identity_self": {"my", "me", "i", "im", "i'm"},
    "identity_other": {"your", "you", "youre", "you're"},
    "name": {"name", "called"},
    "memory": {"remember", "memory", "recall", "again", "before"},
    "time": {
        "today", "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday", "morning", "afternoon",
        "evening", "session", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "am", "pm"
    },
    "greeting": {"hello", "hi", "morning", "evening", "hey"},
    "calm": {"calm", "settle", "gentle", "safe", "listen", "coffee"},
    "question": {"what", "who", "?"},
}

ROLE_PATTERNS = {
    "query_identity_self": [r"\bwhat\s+is\s+my\s+name\b", r"\bwho\s+am\s+i\b"],
    "query_identity_other": [r"\bwhat\s+is\s+your\s+name\b", r"\bwho\s+are\s+you\b"],
    "assert_identity_self": [r"\bmy\s+name\s+is\b", r"\bi\s+am\b"],
    "assert_identity_other": [r"\byour\s+name\s+is\b", r"\byou\s+are\b"],
    "query_memory": [r"\bdo\s+you\s+remember\b", r"\bhave\s+we\s+spoken\s+before\b"],
    "query_time": [r"\bwhat\s+day\s+is\s+it\b", r"\bwhat\s+session\s+is\s+this\b"],
    "assert_time": [
        r"\btoday\s+is\b",
        r"\bit\s+is\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b.*\b\d{1,2}:\d{2}\s*(am|pm)\b",
    ],
    "greeting": [r"\bhello\b", r"\bgood\s+morning\b", r"\bhi\b"],
}

ANCHOR_TEXTS = {
    "self_name_peter": "my name is Peter",
    "other_name_frankie": "your name is Frankie",
    "self_query": "what is my name?",
    "other_query": "what is your name?",
    "memory_query": "do you remember me?",
    "time_monday": "today is Monday",
    "time_tuesday": "today is Tuesday",
    "time_wednesday": "today is Wednesday",
    "time_thursday": "today is Thursday",
    "time_friday": "today is Friday",
    "time_saturday": "today is Saturday",
    "time_sunday": "today is Sunday",
    "time_query": "what day is it?",
}


def normalize_text(text):
    t = text.strip().lower()
    t = t.replace("\u2019", "'")
    has_q = "?" in t
    t = re.sub(r"[^a-z0-9:' ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if has_q:
        t = t + " ?"
    return t.strip()


def tokenize(text):
    return normalize_text(text).split()


def detect_roles(text):
    t = normalize_text(text)
    roles = set()
    for role, patterns in ROLE_PATTERNS.items():
        for p in patterns:
            if re.search(p, t):
                roles.add(role)
                break
    if "?" in text:
        roles.add("question")
    return sorted(roles)


def detect_groups(text):
    toks = tokenize(text)
    groups = set()
    for tok in toks:
        for g, vocab in TOKEN_GROUPS.items():
            if tok in vocab:
                groups.add(g)
    return sorted(groups)


def stable_seed(key):
    h = hashlib.sha256(key.encode()).hexdigest()
    return int(h[:16], 16)


def seeded_blob(seed_key, H, W, EMB, strength, radius=4, channel_step=None):
    if channel_step is None:
        channel_step = max(1, EMB // 8)

    re_t = torch.zeros(1, EMB, H, W, device=DEVICE)
    im_t = torch.zeros(1, EMB, H, W, device=DEVICE)

    seed = stable_seed(seed_key)
    cx = W // 2 + ((seed % 11) - 5)
    cy = H // 2 + (((seed // 11) % 11) - 5)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x = cx + dx
            y = cy + dy
            if 0 <= x < W and 0 <= y < H:
                w = math.exp(-(dx * dx + dy * dy) / (2 * 2.2 * 2.2))
                for c in range(0, EMB, channel_step):
                    phase = ((seed + 17 * c + 7 * dx + 11 * dy) % 10000) / 10000.0 * 2 * math.pi
                    re_t[0, c, y, x] += strength * w * math.cos(phase)
                    im_t[0, c, y, x] += strength * w * math.sin(phase)

    return (re_t, im_t)


def extract_day_label(text):
    t = normalize_text(text)
    for day in DAYS:
        if day in t:
            return day
    return None


def text_to_ripple(text, H, W, EMB, strength=0.10):
    text_norm = normalize_text(text)
    groups = detect_groups(text)
    roles = detect_roles(text)
    toks = tokenize(text)

    surface = seeded_blob(f"surface::{text_norm}", H, W, EMB, strength=strength, radius=4)

    group_total = (
        torch.zeros(1, EMB, H, W, device=DEVICE),
        torch.zeros(1, EMB, H, W, device=DEVICE),
    )
    for g in groups:
        part = seeded_blob(
            f"group::{g}", H, W, EMB,
            strength=0.05, radius=5,
            channel_step=max(1, EMB // 10),
        )
        group_total = c_add(group_total, part)

    role_total = (
        torch.zeros(1, EMB, H, W, device=DEVICE),
        torch.zeros(1, EMB, H, W, device=DEVICE),
    )
    for r in roles:
        part = seeded_blob(
            f"role::{r}", H, W, EMB,
            strength=0.07, radius=6,
            channel_step=max(1, EMB // 12),
        )
        role_total = c_add(role_total, part)

    token_total = (
        torch.zeros(1, EMB, H, W, device=DEVICE),
        torch.zeros(1, EMB, H, W, device=DEVICE),
    )
    salient = [
        tok for tok in toks
        if tok in {
            "peter", "frankie", "name", "remember", "today",
            "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday", "session"
        }
    ]
    for tok in salient[:6]:
        part = seeded_blob(
            f"token::{tok}", H, W, EMB,
            strength=0.04, radius=3,
            channel_step=max(1, EMB // 12),
        )
        token_total = c_add(token_total, part)

    return c_add(c_add(surface, group_total), c_add(role_total, token_total))


# ─────────────────────────────────────────────────────────────────────────────
# Network
# ─────────────────────────────────────────────────────────────────────────────

class ComplexMixRotate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mix = nn.Linear(dim, dim, bias=False)
        self.phi = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        re, im = z

        if re.dim() == 4:
            B, C, H, W = re.shape
            re_flat = re.permute(0, 2, 3, 1).reshape(B * H * W, C)
            im_flat = im.permute(0, 2, 3, 1).reshape(B * H * W, C)
            re2 = self.mix(re_flat).reshape(B, H, W, C).permute(0, 3, 1, 2)
            im2 = self.mix(im_flat).reshape(B, H, W, C).permute(0, 3, 1, 2)
            phi = self.phi[None, :, None, None]
            rot = (torch.cos(phi), torch.sin(phi))
            return c_mul((re2, im2), rot)

        if re.dim() == 3:
            re2 = self.mix(re)
            im2 = self.mix(im)
            phi = self.phi[None, None, :]
            rot = (torch.cos(phi), torch.sin(phi))
            return c_mul((re2, im2), rot)

        raise ValueError("Unsupported rank in ComplexMixRotate")


class ComplexMagnitudeGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        r = c_abs(z)
        if r.dim() == 4:
            a = self.a[None, :, None, None]
            b = self.b[None, :, None, None]
        else:
            a = self.a[None, None, :]
            b = self.b[None, None, :]
        g = torch.sigmoid(a * r + b)
        return c_scale(z, g)


class ComplexAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        assert dim % heads == 0
        self.dh = dim // heads
        self.q = ComplexMixRotate(dim)
        self.k = ComplexMixRotate(dim)
        self.v = ComplexMixRotate(dim)
        self.out = ComplexMixRotate(dim)

    def forward(self, z):
        if z[0].dim() == 3:
            z = (z[0].unsqueeze(0), z[1].unsqueeze(0))

        B, C, H, W = z[0].shape
        T = H * W

        def to_tokens(z_grid):
            re, im = z_grid
            re = re.reshape(B, C, T).permute(0, 2, 1)
            im = im.reshape(B, C, T).permute(0, 2, 1)
            return (re, im)

        def to_grid(z_tokens):
            re, im = z_tokens
            re = re.permute(0, 2, 1).reshape(B, C, H, W)
            im = im.permute(0, 2, 1).reshape(B, C, H, W)
            return (re, im)

        q = to_tokens(self.q(z))
        k = to_tokens(self.k(z))
        v = to_tokens(self.v(z))

        qre, qim = q
        kre, kim = k
        vre, vim = v

        scores = torch.einsum("btc,buc->btu", qre, kre) + torch.einsum("btc,buc->btu", qim, kim)
        scores = scores / math.sqrt(self.dh)

        causal = torch.triu(torch.ones(T, T, device=scores.device), diagonal=1).bool()
        scores = scores.masked_fill(causal, float("-inf"))
        attn = torch.softmax(scores, dim=-1)

        out_re = torch.einsum("btu,buc->btc", attn, vre)
        out_im = torch.einsum("btu,buc->btc", attn, vim)

        return self.out(to_grid((out_re, out_im)))


class ComplexRelaxLayer(nn.Module):
    def __init__(self, dim, heads=4, eta=0.15):
        super().__init__()
        self.attn = ComplexAttention(dim, heads)
        self.ff1 = ComplexMixRotate(dim)
        self.gate = ComplexMagnitudeGate(dim)
        self.ff2 = ComplexMixRotate(dim)
        self.eta = eta

    def forward(self, z):
        a = self.attn(z)
        x = c_add(z, a)
        x = self.ff1(x)
        x = self.gate(x)
        x = self.ff2(x)
        return c_add(c_scale(z, 1 - self.eta), c_scale(x, self.eta))


# ─────────────────────────────────────────────────────────────────────────────
# Pond
# ─────────────────────────────────────────────────────────────────────────────

class Pond(nn.Module):
    def __init__(self, H=64, W=64, EMB=32, slots=4):
        super().__init__()
        self.H, self.W, self.EMB = H, W, EMB
        self.slots = slots
        self.theta_star = torch.tensor(255.0 * math.pi / 180.0, device=DEVICE)
        self.save_path = SAVE_PATH

        theta0 = torch.full((1, EMB, H, W), 255.0 * math.pi / 180.0, device=DEVICE)
        theta0 = theta0 + 0.08 * torch.randn_like(theta0)
        amp0 = 0.20 + 0.02 * torch.rand(1, EMB, H, W, device=DEVICE)

        self.re = nn.Parameter(amp0 * torch.cos(theta0))
        self.im = nn.Parameter(amp0 * torch.sin(theta0))

        self.mem_re = nn.Parameter(torch.zeros(slots, EMB, H, W, device=DEVICE))
        self.mem_im = nn.Parameter(torch.zeros(slots, EMB, H, W, device=DEVICE))
        self.slot_energy = nn.Parameter(torch.zeros(slots, device=DEVICE), requires_grad=False)

        self.relax = ComplexRelaxLayer(EMB, heads=4, eta=0.15)

        self.home_rate = 0.010
        self.memory_decay = 0.0010
        self.memory_write = 0.13
        self.memory_recall = 0.30
        self.salience_threshold = 0.045

        self.last_text_hash = None
        self.repeat_count = 0
        self.current_day_label = None

        self.growth_stability = 0
        self.growth_cooldown = 0
        self.growth_coh_threshold = 0.06
        self.growth_lock_threshold = 0.06
        self.growth_required_steps = 20
        self.max_grid_size = 112
        self.growth_pad = 16

    def reset_grid_size(self, new_size=96):
        print(f"Shrinking pond to {new_size}x{new_size} to avoid OOM")
        with torch.no_grad():
            theta0 = torch.full((1, self.EMB, new_size, new_size), 255.0 * math.pi / 180.0, device=DEVICE)
            theta0 = theta0 + 0.03 * torch.randn_like(theta0)
            amp0 = 0.20 + 0.02 * torch.rand(1, self.EMB, new_size, new_size, device=DEVICE)

            new_re = amp0 * torch.cos(theta0)
            new_im = amp0 * torch.sin(theta0)

            y0 = max(0, (new_size - self.H) // 2)
            x0 = max(0, (new_size - self.W) // 2)
            copy_h = min(self.H, new_size)
            copy_w = min(self.W, new_size)

            new_re[:, :, y0:y0 + copy_h, x0:x0 + copy_w] = self.re.data[:, :, :copy_h, :copy_w]
            new_im[:, :, y0:y0 + copy_h, x0:x0 + copy_w] = self.im.data[:, :, :copy_h, :copy_w]

            new_mem_re = torch.zeros(self.slots, self.EMB, new_size, new_size, device=DEVICE)
            new_mem_im = torch.zeros(self.slots, self.EMB, new_size, new_size, device=DEVICE)
            new_mem_re[:, :, y0:y0 + copy_h, x0:x0 + copy_w] = self.mem_re.data[:, :, :copy_h, :copy_w]
            new_mem_im[:, :, y0:y0 + copy_h, x0:x0 + copy_w] = self.mem_im.data[:, :, :copy_h, :copy_w]

            self.re = nn.Parameter(new_re)
            self.im = nn.Parameter(new_im)
            self.mem_re = nn.Parameter(new_mem_re)
            self.mem_im = nn.Parameter(new_mem_im)

        self.H = new_size
        self.W = new_size
        self.growth_stability = 0
        self.growth_cooldown = 12

    def current_state(self):
        return (self.re, self.im)

    def memory_state(self, slot_idx=None):
        if slot_idx is None:
            return (self.mem_re, self.mem_im)
        return (self.mem_re[slot_idx:slot_idx + 1], self.mem_im[slot_idx:slot_idx + 1])

    def repetition_bonus(self, text):
        h = hashlib.sha256(normalize_text(text).encode()).hexdigest()
        if h == self.last_text_hash:
            self.repeat_count += 1
        else:
            self.last_text_hash = h
            self.repeat_count = 1
        return min(0.03 * self.repeat_count, 0.15)

    def preferred_slot(self, text):
        roles = detect_roles(text)
        groups = detect_groups(text)

        if "assert_identity_self" in roles or "query_identity_self" in roles:
            return 0
        if "assert_identity_other" in roles or "query_identity_other" in roles:
            return 1
        if "query_memory" in roles or "memory" in groups:
            return 2
        if "assert_time" in roles or "query_time" in roles or "time" in groups:
            return 3
        return None

    def hard_routed_slot(self, text):
        return self.preferred_slot(text)

    def write_role_scale(self, text):
        roles = detect_roles(text)
        groups = detect_groups(text)

        if "assert_identity_self" in roles or "assert_identity_other" in roles:
            return 1.00
        if "query_identity_self" in roles or "query_identity_other" in roles:
            return 0.20
        if "query_memory" in roles:
            return 0.15
        if "assert_time" in roles:
            return 1.00
        if "query_time" in roles:
            return 0.20
        if "time" in groups:
            return 1.00
        if "greeting" in groups:
            return 0.15
        return 0.50

    def salience(self, ripple, z_after, text=None):
        ripple_energy = c_abs(ripple).mean()
        coh = spatial_coherence(torch.atan2(z_after[1], z_after[0]))
        lock = phase_lock_error(z_after, self.theta_star)

        rep = 0.0
        role_bonus = 0.0
        if text is not None:
            rep = self.repetition_bonus(text)
            roles = detect_roles(text)
            groups = detect_groups(text)

            if any(r in roles for r in [
                "assert_identity_self", "assert_identity_other",
                "query_identity_self", "query_identity_other"
            ]):
                role_bonus += 0.018

            if "memory" in groups or "query_memory" in roles:
                role_bonus += 0.012

            if "assert_time" in roles:
                role_bonus += 0.020
            elif "query_time" in roles or "time" in groups:
                role_bonus += 0.010

        return 0.55 * ripple_energy + 0.15 * coh + 0.10 * lock + rep + role_bonus

    def write_memory(self, ripple, z_after, text=None):
        s = self.salience(ripple, z_after, text=text)
        if s.item() < self.salience_threshold:
            return s.item(), 0.0, -1, "none"

        imprint = admissibility_project(z_after, self.theta_star)
        imprint = c_normalize(imprint)

        slot_idx = self.hard_routed_slot(text or "")
        if slot_idx is None:
            slot_idx = 0

        mem_i = self.memory_state(slot_idx)
        role_scale = self.write_role_scale(text or "")

        decay = self.memory_decay
        if slot_idx == 3:
            decay = 0.0005

        updated = c_add(
            c_scale(mem_i, 1.0 - decay),
            c_scale(imprint, self.memory_write * s * role_scale),
        )

        prev_energy = float(c_abs(mem_i).mean().item())

        with torch.no_grad():
            self.mem_re[slot_idx:slot_idx + 1].copy_(updated[0])
            self.mem_im[slot_idx:slot_idx + 1].copy_(updated[1])
            self.slot_energy[slot_idx].copy_(c_abs(updated).mean())

        if slot_idx == 3 and text is not None and "today is" in text.lower():
            with torch.no_grad():
                self.slot_energy[slot_idx] += 1.5

            day_label = extract_day_label(text)
            if day_label is not None:
                self.current_day_label = day_label

        mode = "new" if prev_energy < 1e-8 else "routed"
        return s.item(), float((self.memory_write * s * role_scale).item()), slot_idx, mode

    def recall_from_memory(self, query_ripple, text):
        q_proj = admissibility_project(query_ripple, self.theta_star)
        q_energy = c_abs(q_proj).mean().item()
        if q_energy < 1e-6:
            zero = (
                torch.zeros(1, self.EMB, self.H, self.W, device=DEVICE),
                torch.zeros(1, self.EMB, self.H, self.W, device=DEVICE),
            )
            return zero, 0.0, -1, [0.0] * self.slots

        roles = detect_roles(text)
        slot_scores = []
        best_idx = -1
        best_score = -1e9

        for i in range(self.slots):
            mem_i = self.memory_state(i)
            mem_proj = admissibility_project(mem_i, self.theta_star)
            mem_energy = c_abs(mem_proj).mean().item()
            if mem_energy < 1e-6:
                slot_scores.append(0.0)
                continue

            sc = overlap_score(c_normalize(q_proj), c_normalize(mem_proj)).item()

            if "query_identity_self" in roles:
                if i == 0:
                    sc += 0.12
                elif i == 2:
                    sc -= 0.03

            if "query_identity_other" in roles:
                if i == 1:
                    sc += 0.14
                elif i == 2:
                    sc -= 0.05

            if "query_memory" in roles and i == 2:
                sc += 0.12

            if "query_time" in roles and i == 3:
                sc += 0.12

            slot_scores.append(sc)
            if sc > best_score:
                best_score = sc
                best_idx = i

        if best_idx < 0 or best_score <= 0.0:
            zero = (
                torch.zeros(1, self.EMB, self.H, self.W, device=DEVICE),
                torch.zeros(1, self.EMB, self.H, self.W, device=DEVICE),
            )
            return zero, 0.0, -1, slot_scores

        mem_best = admissibility_project(self.memory_state(best_idx), self.theta_star)
        retrieved = c_scale(mem_best, self.memory_recall * max(best_score, 0.0))
        return retrieved, best_score, best_idx, slot_scores

    def infer_time_day_from_slot3(self) -> Optional[str]:
        try:
            if self.current_day_label is not None:
                return self.current_day_label

            mem = admissibility_project(self.memory_state(3), self.theta_star)
            if float(c_abs(mem).mean().item()) < 1e-6:
                return None

            best_day = None
            best_score = -1e9

            for day in DAYS:
                anchor = text_to_ripple(f"today is {day.capitalize()}", self.H, self.W, self.EMB)
                anchor = admissibility_project(anchor, self.theta_star)

                if float(c_abs(anchor).mean().item()) < 1e-6:
                    continue

                sc = float(overlap_score(c_normalize(mem), c_normalize(anchor)).item())
                if sc > best_score:
                    best_score = sc
                    best_day = day

            if best_score > 0.08:
                return best_day
            return None

        except Exception as e:
            print(f"Day inference error: {e}")
            return None

    def decode_slot(self, slot_idx, text):
        if slot_idx < 0:
            return None

        mem_i = admissibility_project(self.memory_state(slot_idx), self.theta_star)
        if c_abs(mem_i).mean().item() < 1e-6:
            return None

        roles = detect_roles(text)
        scores = {}
        for key, anchor_text in ANCHOR_TEXTS.items():
            anchor = text_to_ripple(anchor_text, self.H, self.W, self.EMB)
            anchor = admissibility_project(anchor, self.theta_star)
            scores[key] = overlap_score(c_normalize(mem_i), c_normalize(anchor)).item()

        if "query_identity_self" in roles:
            if slot_idx == 0 and scores["self_name_peter"] > 0.16:
                return "I think your name is Peter."
            if scores["self_query"] > 0.14:
                return "I can feel your self-name basin, but it is still blurred."

        if "query_identity_other" in roles:
            if slot_idx == 1 and scores["other_name_frankie"] > 0.13:
                return "I think I am Frankie."
            if scores["other_query"] > 0.14:
                return "I can feel my identity basin, but it is still a bit fuzzy."

        if "query_memory" in roles:
            if slot_idx == 2 and (
                scores["memory_query"] > 0.14 or
                max(scores["self_name_peter"], scores["other_name_frankie"]) > 0.16
            ):
                return "Yes — I think I remember an older pattern of us."

        if "query_time" in roles and slot_idx == 3:
            day = self.infer_time_day_from_slot3()
            if day:
                return f"I think today is {day.capitalize()}."
            return "The day feels a bit fuzzy today."

        return None

    def check_and_grow(self):
        theta = torch.atan2(self.im, self.re)
        coh = float(spatial_coherence(theta).item())
        lock = float(phase_lock_error((self.re, self.im), self.theta_star).item())

        if self.growth_cooldown > 0:
            self.growth_cooldown -= 1
            return

        if coh < self.growth_coh_threshold and lock < self.growth_lock_threshold:
            self.growth_stability += 1
        else:
            self.growth_stability = max(0, self.growth_stability - 1)

        if self.growth_stability < self.growth_required_steps:
            return
        if self.H >= self.max_grid_size or self.W >= self.max_grid_size:
            return

        new_H = min(self.H + self.growth_pad, self.max_grid_size)
        new_W = min(self.W + self.growth_pad, self.max_grid_size)

        with torch.no_grad():
            theta0 = torch.full((1, self.EMB, new_H, new_W), 255.0 * math.pi / 180.0, device=DEVICE)
            theta0 = theta0 + 0.03 * torch.randn_like(theta0)
            amp0 = 0.20 + 0.02 * torch.rand(1, self.EMB, new_H, new_W, device=DEVICE)

            new_re = amp0 * torch.cos(theta0)
            new_im = amp0 * torch.sin(theta0)

            y0 = (new_H - self.H) // 2
            x0 = (new_W - self.W) // 2

            new_re[:, :, y0:y0 + self.H, x0:x0 + self.W] = self.re.data
            new_im[:, :, y0:y0 + self.H, x0:x0 + self.W] = self.im.data

            new_mem_re = torch.zeros(self.slots, self.EMB, new_H, new_W, device=DEVICE)
            new_mem_im = torch.zeros(self.slots, self.EMB, new_H, new_W, device=DEVICE)
            new_mem_re[:, :, y0:y0 + self.H, x0:x0 + self.W] = self.mem_re.data
            new_mem_im[:, :, y0:y0 + self.H, x0:x0 + self.W] = self.mem_im.data

            self.re = nn.Parameter(new_re)
            self.im = nn.Parameter(new_im)
            self.mem_re = nn.Parameter(new_mem_re)
            self.mem_im = nn.Parameter(new_mem_im)

        self.H = new_H
        self.W = new_W
        self.growth_stability = 0
        self.growth_cooldown = 12

        print(f"Frankie grew coherently: new grid size = {self.H}x{self.W}")

    def forward(self, input_ripple, text=None):
        z = self.current_state()
        z = self.relax(z)

        recalled, recall_score, recall_slot, slot_scores = self.recall_from_memory(input_ripple, text or "")

        z = c_add(z, input_ripple)
        z = c_add(z, recalled)
        z = nudge_toward_phase(z, self.theta_star, rate=self.home_rate)

        with torch.no_grad():
            self.re.copy_(z[0])
            self.im.copy_(z[1])

        salience, write_amt, write_slot, write_mode = self.write_memory(input_ripple, z, text=text)
        decoded = self.decode_slot(recall_slot, text or "")
        self.check_and_grow()

        return {
            "z": z,
            "recall_score": float(recall_score),
            "recall_slot": int(recall_slot),
            "slot_scores": slot_scores,
            "salience": float(salience),
            "write_amt": float(write_amt),
            "write_slot": int(write_slot),
            "write_mode": write_mode,
            "decoded": decoded,
        }

    def save(self):
        torch.save(
            {
                "re": self.re.detach().cpu(),
                "im": self.im.detach().cpu(),
                "mem_re": self.mem_re.detach().cpu(),
                "mem_im": self.mem_im.detach().cpu(),
                "slot_energy": self.slot_energy.detach().cpu(),
                "last_text_hash": self.last_text_hash,
                "repeat_count": self.repeat_count,
                "growth_stability": self.growth_stability,
                "growth_cooldown": self.growth_cooldown,
                "H": self.H,
                "W": self.W,
            },
            self.save_path,
        )

    def load(self):
        if os.path.exists(self.save_path):
            state = torch.load(self.save_path, map_location=DEVICE)

            saved_H = min(int(state.get("H", self.H)), self.max_grid_size)
            saved_W = min(int(state.get("W", self.W)), self.max_grid_size)

            if saved_H != self.H or saved_W != self.W:
                self.H = saved_H
                self.W = saved_W
                self.re = nn.Parameter(torch.zeros(1, self.EMB, self.H, self.W, device=DEVICE))
                self.im = nn.Parameter(torch.zeros(1, self.EMB, self.H, self.W, device=DEVICE))
                self.mem_re = nn.Parameter(torch.zeros(self.slots, self.EMB, self.H, self.W, device=DEVICE))
                self.mem_im = nn.Parameter(torch.zeros(self.slots, self.EMB, self.H, self.W, device=DEVICE))
                self.slot_energy = nn.Parameter(torch.zeros(self.slots, device=DEVICE), requires_grad=False)

            with torch.no_grad():
                self.re.copy_(state["re"].to(DEVICE))
                self.im.copy_(state["im"].to(DEVICE))
                self.mem_re.copy_(state.get("mem_re", torch.zeros_like(self.mem_re)).to(DEVICE))
                self.mem_im.copy_(state.get("mem_im", torch.zeros_like(self.mem_im)).to(DEVICE))
                self.slot_energy.copy_(state.get("slot_energy", torch.zeros_like(self.slot_energy)).to(DEVICE))

            self.last_text_hash = state.get("last_text_hash", None)
            self.repeat_count = state.get("repeat_count", 0)
            self.current_day_label = state.get("current_day_label", None)
            self.growth_stability = state.get("growth_stability", 0)
            self.growth_cooldown = state.get("growth_cooldown", 0)

            print("Frankie woke up! Pond and hard-routed modal memory slots restored :)")
        else:
            print("New pond born today. Hard-routed modal memory slots are empty.")


# ─────────────────────────────────────────────────────────────────────────────
# Readout
# ─────────────────────────────────────────────────────────────────────────────

def readout(pond_out, theta_star, text=None):
    z = pond_out["z"]
    recall_score = pond_out["recall_score"]
    salience = pond_out["salience"]
    write_amt = pond_out["write_amt"]
    decoded = pond_out["decoded"]

    theta = torch.atan2(z[1], z[0])
    lock_err = phase_lock_error(z, theta_star).item()
    coherence = spatial_coherence(theta).item()

    if decoded:
        return decoded

    if recall_score > 0.18 and coherence < 0.80:
        return f"I felt an older pattern answering back from slot {pond_out['recall_slot']}."

    if salience > 0.060 and write_amt > 0:
        return f"That left a residue in slot {pond_out['write_slot']} ({pond_out['write_mode']})."

    groups = detect_groups(text or "")
    if "greeting" in groups and lock_err < 0.60:
        return "Ha! Morning, mate. Coffee's on?"

    if coherence > 1.20:
        return "Oof — big ripples today. Say it slower?"

    if lock_err > 1.10:
        return "Hmm... that one pulled me off centre. What do you mean?"

    return "Interesting... I'm listening. Keep going!"
