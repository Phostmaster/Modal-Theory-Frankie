"""
jb_clustering.py
────────────────────────────────────────────────────────────────────────────
Clustering module for the JB Present-Privilege Test experiment.
Goal: determine whether the settled present is a stronger organiser than
either the recoverable past or the open future.

This module handles:
  - Recording settled present states from history runs
  - Computing hierarchical similarity (phase-gated, then gains/usage refined)
  - Clustering into convergent groups
  - Selecting the best clusters for future branching
  - Testing present stability under perturbation
  - Comparing past compression vs future expansion vs present privilege

Designed to work directly with frankie_channels_toy.py state dictionaries.
"""

import copy
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import random

# ============================================================================
# CONFIG — tune these before running the full experiment
# ============================================================================
PHASE_GATE_THRESHOLD = 0.60
W_PHASE = 0.40
W_MAGNITUDE = 0.25
W_GAINS = 0.20
W_USAGE = 0.15
CLUSTER_THRESHOLD = 0.65
MIN_INTERNAL_CLUSTER_SIM = 0.55
MIN_CLUSTER_SIZE = 2
N_BEST_CLUSTERS = 3
PERTURBATION_SCALE = 0.01
PERTURBATION_TURNS = 500

PRESENTS_PATH = "jb_presents.json"
CLUSTERS_PATH = "jb_clusters.json"
RESULTS_PATH = "jb_results.json"

# ============================================================================
# DATA STRUCTURES
# ============================================================================
@dataclass
class PresentState:
    history_id: int
    prompt_family: str
    seed: int
    turn_count: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    field_re_flat: List[float] = field(default_factory=list)
    field_im_flat: List[float] = field(default_factory=list)
    field_shape: List[int] = field(default_factory=list)
    templates_re_flat: List[float] = field(default_factory=list)
    templates_im_flat: List[float] = field(default_factory=list)
    templates_shape: List[int] = field(default_factory=list)
    gains: List[float] = field(default_factory=list)
    usage: List[float] = field(default_factory=list)
    coherence_score: float = 0.0
    phase_flat: List[float] = field(default_factory=list)
    magnitude_flat: List[float] = field(default_factory=list)
    lexical_response: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "PresentState":
        return cls(**d)

    def field_re(self) -> np.ndarray:
        return np.array(self.field_re_flat, dtype=np.float32).reshape(self.field_shape)

    def field_im(self) -> np.ndarray:
        return np.array(self.field_im_flat, dtype=np.float32).reshape(self.field_shape)

    def phase(self) -> np.ndarray:
        return np.array(self.phase_flat, dtype=np.float32).reshape(self.field_shape)

    def magnitude(self) -> np.ndarray:
        return np.array(self.magnitude_flat, dtype=np.float32).reshape(self.field_shape)

    def gains_array(self) -> np.ndarray:
        return np.array(self.gains, dtype=np.float32)

    def usage_array(self) -> np.ndarray:
        return np.array(self.usage, dtype=np.float32)

@dataclass
class SimilarityResult:
    id_a: int
    id_b: int
    phase_sim: float
    magnitude_sim: float
    gains_sim: float
    usage_sim: float
    combined: float
    phase_gated: bool
    notes: List[str] = field(default_factory=list)

@dataclass
class Cluster:
    cluster_id: int
    member_ids: List[int]
    prompt_families: List[str]
    mean_similarity: float
    min_similarity: float
    phase_coherence: float
    is_cross_family: bool
    centroid_id: int

# ============================================================================
# RECORDING
# ============================================================================
class PresentRecorder:
    def __init__(self, path: str = PRESENTS_PATH):
        self.path = path
        self.presents: List[PresentState] = []
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.presents = [PresentState.from_dict(d) for d in raw]
            print(f"[Recorder] Loaded {len(self.presents)} existing present states.")
        except (FileNotFoundError, json.JSONDecodeError):
            self.presents = []

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([p.to_dict() for p in self.presents], f, indent=2)
        print(f"[Recorder] Saved {len(self.presents)} present states -> {self.path}")

    def record(
        self,
        state: Dict,
        history_id: int,
        prompt_family: str,
        seed: int,
        turn_count: int,
        coherence: float,
        lexical: str = "",
    ) -> PresentState:
        re = state["field_re"].astype(np.float32)
        im = state["field_im"].astype(np.float32)
        mag = np.sqrt(re ** 2 + im ** 2)
        pha = np.arctan2(im, re)

        ps = PresentState(
            history_id=history_id,
            prompt_family=prompt_family,
            seed=seed,
            turn_count=turn_count,
            coherence_score=float(coherence),
            lexical_response=lexical,
            field_re_flat=re.flatten().tolist(),
            field_im_flat=im.flatten().tolist(),
            field_shape=list(re.shape),
            templates_re_flat=state["templates_re"].flatten().tolist(),
            templates_im_flat=state["templates_im"].flatten().tolist(),
            templates_shape=list(state["templates_re"].shape),
            gains=[float(g) for g in state["gains"]],
            usage=[float(u) for u in state["usage"]],
            phase_flat=pha.flatten().tolist(),
            magnitude_flat=mag.flatten().tolist(),
        )

        self.presents = [p for p in self.presents if p.history_id != history_id]
        self.presents.append(ps)
        self.presents.sort(key=lambda p: p.history_id)
        self.save()

        print(
            f"[Recorder] Recorded history {history_id} ({prompt_family}, "
            f"seed={seed}, turns={turn_count}, coh={coherence:.3f})"
        )
        return ps

    def status(self, expected_histories: int = 8):
        print(f"\n[Recorder] {len(self.presents)}/{expected_histories} histories recorded:")
        for p in self.presents:
            print(
                f" [{p.history_id}] {p.prompt_family:<12} "
                f"seed={p.seed:<12} turns={p.turn_count} "
                f"coh={p.coherence_score:.3f} "
                f"gains={np.round(p.gains_array(), 3)}"
            )
        missing = set(range(expected_histories)) - {p.history_id for p in self.presents}
        if missing:
            print(f" Missing history IDs: {sorted(missing)}")

# ============================================================================
# SIMILARITY METRICS
# ============================================================================
def _cosine_1d(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.clip(num / den, -1.0, 1.0))

def _magnitude_overlap(mag_a: np.ndarray, mag_b: np.ndarray) -> float:
    return _cosine_1d(mag_a.flatten(), mag_b.flatten())

def _phase_similarity(phase_a: np.ndarray, phase_b: np.ndarray) -> float:
    diff = np.angle(np.exp(1j * (phase_a - phase_b)))
    mean_abs_diff = float(np.mean(np.abs(diff)))
    return float(1.0 - mean_abs_diff / np.pi)

def compute_similarity(a: PresentState, b: PresentState) -> SimilarityResult:
    phase_sim = _phase_similarity(a.phase(), b.phase())
    mag_sim = _magnitude_overlap(a.magnitude(), b.magnitude())
    gains_sim = _cosine_1d(a.gains_array(), b.gains_array())
    usage_sim = _cosine_1d(a.usage_array(), b.usage_array())
    notes: List[str] = []

    if phase_sim < PHASE_GATE_THRESHOLD:
        combined = phase_sim
        gated = False
        notes.append(f"phase_gate_failed ({phase_sim:.3f} < {PHASE_GATE_THRESHOLD})")
    else:
        combined = (
            W_PHASE * phase_sim
            + W_MAGNITUDE * mag_sim
            + W_GAINS * gains_sim
            + W_USAGE * usage_sim
        )
        gated = True
        if gains_sim < 0.5:
            notes.append("gains_diverged")
        if usage_sim < 0.5:
            notes.append("usage_diverged")
        if phase_sim > 0.90:
            notes.append("near_identical_phase")
        if abs(gains_sim - usage_sim) > 0.3:
            notes.append("gains_usage_mismatch")

    return SimilarityResult(
        id_a=a.history_id,
        id_b=b.history_id,
        phase_sim=round(phase_sim, 4),
        magnitude_sim=round(mag_sim, 4),
        gains_sim=round(gains_sim, 4),
        usage_sim=round(usage_sim, 4),
        combined=round(float(combined), 4),
        phase_gated=gated,
        notes=notes,
    )

# ============================================================================
# CLUSTERING
# ============================================================================
def cluster_presents(
    presents: List[PresentState],
    threshold: float = CLUSTER_THRESHOLD,
    verbose: bool = True,
) -> List[Cluster]:
    n = len(presents)
    if n == 0:
        return []

    pairs: Dict[Tuple[int, int], SimilarityResult] = {}
    for i in range(n):
        for j in range(i + 1, n):
            sr = compute_similarity(presents[i], presents[j])
            pairs[(i, j)] = sr

    if verbose:
        _print_similarity_matrix(presents, pairs)

    adj: Dict[int, set] = {i: set() for i in range(n)}
    for (i, j), sr in pairs.items():
        if sr.combined >= threshold and sr.phase_gated:
            adj[i].add(j)
            adj[j].add(i)

    visited = set()
    components: List[List[int]] = []
    for start in range(n):
        if start in visited:
            continue
        component = []
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            queue.extend(adj[node] - visited)
        components.append(sorted(component))

    clusters: List[Cluster] = []
    for cid, members in enumerate(components):
        member_presents = [presents[i] for i in members]
        families = [p.prompt_family for p in member_presents]

        within_sims = []
        within_phase = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                key = (min(a, b), max(a, b))
                if key in pairs:
                    within_sims.append(pairs[key].combined)
                    within_phase.append(pairs[key].phase_sim)

        mean_sim = float(np.mean(within_sims)) if within_sims else 1.0
        min_sim = float(np.min(within_sims)) if within_sims else 1.0
        mean_phase = float(np.mean(within_phase)) if within_phase else 1.0

        if len(members) > 1 and min_sim < MIN_INTERNAL_CLUSTER_SIM:
            if verbose:
                print(f"[Cluster Filter] Dropping weak chained cluster {cid}")
            continue

        centroid_idx = _find_centroid(members, pairs)
        clusters.append(
            Cluster(
                cluster_id=cid,
                member_ids=[presents[i].history_id for i in members],
                prompt_families=families,
                mean_similarity=round(mean_sim, 4),
                min_similarity=round(min_sim, 4),
                phase_coherence=round(mean_phase, 4),
                is_cross_family=len(set(families)) > 1,
                centroid_id=presents[centroid_idx].history_id,
            )
        )

    clusters.sort(key=lambda c: (-len(c.member_ids), -c.mean_similarity))
    if verbose:
        _print_clusters(clusters)
    return clusters

def _find_centroid(members: List[int], pairs: Dict[Tuple[int, int], SimilarityResult]) -> int:
    best_idx = members[0]
    best_mean = -1.0
    for i in members:
        sims = []
        for j in members:
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            if key in pairs:
                sims.append(pairs[key].combined)
        if sims:
            mean = float(np.mean(sims))
            if mean > best_mean:
                best_mean = mean
                best_idx = i
    return best_idx

def select_best_clusters(
    clusters: List[Cluster],
    n: int = N_BEST_CLUSTERS,
    min_size: int = MIN_CLUSTER_SIZE,
    prefer_cross_family: bool = True,
) -> List[Cluster]:
    eligible = [c for c in clusters if len(c.member_ids) >= min_size]
    if prefer_cross_family:
        cross = sorted([c for c in eligible if c.is_cross_family],
                       key=lambda c: (-len(c.member_ids), -c.mean_similarity))
        same = sorted([c for c in eligible if not c.is_cross_family],
                      key=lambda c: (-len(c.member_ids), -c.mean_similarity))
        ordered = cross + same
    else:
        ordered = sorted(eligible, key=lambda c: (-len(c.member_ids), -c.mean_similarity))
    selected = ordered[:n]
    print(f"\n[Select] {len(selected)} clusters selected for future branching:")
    for c in selected:
        cross_tag = "★ CROSS-FAMILY" if c.is_cross_family else " same-family"
        print(
            f" Cluster {c.cluster_id} | {cross_tag} | "
            f"size={len(c.member_ids)} | "
            f"members={c.member_ids} | "
            f"families={c.prompt_families} | "
            f"mean_sim={c.mean_similarity:.3f} | "
            f"phase_coh={c.phase_coherence:.3f} | "
            f"centroid=history_{c.centroid_id}"
        )
    return selected

# ============================================================================
# PERTURBATION TEST
# ============================================================================
def perturbation_test(
    state: Dict,
    frankie_turn_fn,
    n_turns: int = PERTURBATION_TURNS,
    noise_scale: float = PERTURBATION_SCALE,
    verbose: bool = True,
) -> Dict:
    orig_re = state["field_re"].copy()
    orig_im = state["field_im"].copy()
    orig_pha = np.arctan2(orig_im, orig_re)
    orig_mag = np.sqrt(orig_re ** 2 + orig_im ** 2)
    orig_gains = state["gains"].copy()
    orig_usage = state["usage"].copy()

    perturbed = copy.deepcopy(state)
    h, w = orig_re.shape
    noise_re = np.random.normal(0, noise_scale, (h, w)).astype(np.float32)
    noise_im = np.random.normal(0, noise_scale, (h, w)).astype(np.float32)
    perturbed["field_re"] = (orig_re + noise_re).astype(np.float32)
    perturbed["field_im"] = (orig_im + noise_im).astype(np.float32)

    init_perturbation = float(np.mean(noise_re ** 2 + noise_im ** 2))
    if verbose:
        print(f"\n[Perturbation] noise_scale={noise_scale} | init_energy={init_perturbation:.6f}")

    zero_re = np.zeros((h, w), dtype=np.float32)
    zero_im = np.zeros((h, w), dtype=np.float32)
    coh_trace = []
    for t in range(n_turns):
        result = frankie_turn_fn(perturbed, zero_re, zero_im, gain_scale=0.0, is_dream=False)
        if t % 100 == 0:
            coh_trace.append((t, round(result["coherence_score"], 4)))

    final_re = perturbed["field_re"]
    final_im = perturbed["field_im"]
    final_pha = np.arctan2(final_im, final_re)
    final_mag = np.sqrt(final_re ** 2 + final_im ** 2)

    phase_return = _phase_similarity(orig_pha, final_pha)
    magnitude_return = _cosine_1d(orig_mag.flatten(), final_mag.flatten())
    gains_return = _cosine_1d(orig_gains, perturbed["gains"])
    usage_return = _cosine_1d(orig_usage, perturbed["usage"])

    combined_return = (
        W_PHASE * phase_return
        + W_MAGNITUDE * magnitude_return
        + W_GAINS * gains_return
        + W_USAGE * usage_return
    )

    residual = float(np.mean((final_re - orig_re) ** 2 + (final_im - orig_im) ** 2))

    results = {
        "noise_scale": noise_scale,
        "init_perturbation": round(init_perturbation, 6),
        "residual_energy": round(residual, 6),
        "energy_reduction": round(1.0 - residual / max(init_perturbation, 1e-8), 4),
        "phase_return": round(phase_return, 4),
        "magnitude_return": round(magnitude_return, 4),
        "gains_return": round(gains_return, 4),
        "usage_return": round(usage_return, 4),
        "combined_return": round(float(combined_return), 4),
        "coherence_trace": coh_trace,
        "settled_gains": [round(float(g), 4) for g in perturbed["gains"]],
        "settled_usage": [round(float(u), 4) for u in perturbed["usage"]],
        "privilege_verdict": (
            "STRONG" if combined_return > 0.85 else
            "MODERATE" if combined_return > 0.70 else
            "WEAK" if combined_return > 0.50 else
            "ABSENT"
        ),
    }
    if verbose:
        print(f" Phase return : {phase_return:.4f}")
        print(f" Magnitude return: {magnitude_return:.4f}")
        print(f" Gains return : {gains_return:.4f}")
        print(f" Usage return : {usage_return:.4f}")
        print(f" Combined return : {combined_return:.4f}")
        print(f" Energy reduced : {results['energy_reduction']*100:.1f}%")
        print(f" Verdict : {results['privilege_verdict']}")
    return results

# ============================================================================
# ASYMMETRY ANALYSIS
# ============================================================================
def compute_past_compression(clusters: List[Cluster], presents: List[PresentState]) -> Dict:
    results = {}
    for c in clusters:
        members = [p for p in presents if p.history_id in c.member_ids]
        families = [p.prompt_family for p in members]
        unique_families = len(set(families))
        family_entropy = _entropy([families.count(f) / len(families) for f in set(families)])
        gains_vectors = np.array([p.gains_array() for p in members])
        gains_variance = float(np.mean(np.var(gains_vectors, axis=0)))
        results[f"cluster_{c.cluster_id}"] = {
            "n_histories": len(members),
            "unique_families": unique_families,
            "family_entropy": round(family_entropy, 4),
            "gains_variance": round(gains_variance, 6),
            "compression_score": round(family_entropy * (unique_families / max(len(members), 1)), 4),
            "interpretation": (
                "HIGH compression — past is ambiguous given present"
                if unique_families > 1 else
                "LOW compression — present implies a specific past"
            ),
        }
    return results

def compute_future_expansion(future_branch_results: List[Dict]) -> Dict:
    if not future_branch_results:
        return {"error": "no future branches provided"}
    gains_matrix = np.array([r["gains"] for r in future_branch_results], dtype=np.float32)
    usage_matrix = np.array([r["usage"] for r in future_branch_results], dtype=np.float32)
    cohs = [r["coherence_score"] for r in future_branch_results]
    gains_spread = float(np.mean(np.var(gains_matrix, axis=0)))
    usage_spread = float(np.mean(np.var(usage_matrix, axis=0)))
    coh_spread = float(np.var(cohs))
    coh_mean = float(np.mean(cohs))
    n = len(future_branch_results)
    pairwise_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine_1d(gains_matrix[i], gains_matrix[j])
            pairwise_sims.append(sim)
    mean_future_sim = float(np.mean(pairwise_sims)) if pairwise_sims else 1.0
    return {
        "n_branches": n,
        "gains_spread": round(gains_spread, 6),
        "usage_spread": round(usage_spread, 6),
        "coherence_spread": round(coh_spread, 6),
        "coherence_mean": round(coh_mean, 4),
        "mean_future_sim": round(mean_future_sim, 4),
        "expansion_score": round(1.0 - mean_future_sim, 4),
        "interpretation": (
            "HIGH expansion — futures diverge strongly from present"
            if mean_future_sim < 0.5 else
            "MODERATE expansion"
            if mean_future_sim < 0.8 else
            "LOW expansion — present strongly governs futures"
        ),
    }

def compare_asymmetries(
    past_compression: Dict,
    future_expansion: Dict,
    perturbation_result: Dict,
) -> Dict:
    compression_scores = [
        v["compression_score"]
        for v in past_compression.values()
        if isinstance(v, dict) and "compression_score" in v
    ]
    mean_compression = float(np.mean(compression_scores)) if compression_scores else 0.0
    present_return = float(perturbation_result.get("combined_return", 0.0))
    future_sim = float(future_expansion.get("mean_future_sim", 1.0))
    expansion_score = float(future_expansion.get("expansion_score", 0.0))
    present_privilege_score = float(
        0.4 * present_return +
        0.3 * mean_compression +
        0.3 * expansion_score
    )
    verdict = (
        "CONFIRMED — present is stronger organiser than past or future"
        if present_privilege_score > 0.60 else
        "PARTIAL — present shows some privilege but not dominant"
        if present_privilege_score > 0.40 else
        "NOT CONFIRMED — present does not dominate"
    )
    return {
        "mean_past_compression": round(mean_compression, 4),
        "present_return_score": round(present_return, 4),
        "future_mean_similarity": round(future_sim, 4),
        "future_expansion_score": round(expansion_score, 4),
        "present_privilege_score": round(present_privilege_score, 4),
        "jb_verdict": verdict,
        "perturbation_verdict": perturbation_result.get("privilege_verdict", "UNKNOWN"),
        "timestamp": datetime.now().isoformat(),
    }

# ============================================================================
# PERSISTENCE & PRINTING
# ============================================================================
def save_clusters(clusters: List[Cluster], path: str = CLUSTERS_PATH):
    data = []
    for c in clusters:
        data.append({
            "cluster_id": c.cluster_id,
            "member_ids": c.member_ids,
            "prompt_families": c.prompt_families,
            "mean_similarity": c.mean_similarity,
            "min_similarity": c.min_similarity,
            "phase_coherence": c.phase_coherence,
            "is_cross_family": c.is_cross_family,
            "centroid_id": c.centroid_id,
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[Clusters] Saved {len(clusters)} clusters -> {path}")

def save_results(results: Dict, path: str = RESULTS_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[Results] Saved -> {path}")

def _print_similarity_matrix(presents: List[PresentState], pairs: Dict[Tuple[int, int], SimilarityResult]):
    n = len(presents)
    print(f"\n{'─' * 70}")
    print("PAIRWISE SIMILARITY MATRIX")
    print(f"{'─' * 70}")
    print(f" {'':>4}", end="")
    for p in presents:
        print(f" H{p.history_id:<3}", end="")
    print()
    for i in range(n):
        print(f" H{presents[i].history_id:<3}", end="")
        for j in range(n):
            if i == j:
                print(f" {'—':>4}", end="")
            elif i < j:
                sr = pairs.get((i, j))
                val = sr.combined if sr else 0.0
                gated = "+" if (sr and sr.phase_gated) else " "
                print(f" {gated}{val:.2f}", end="")
            else:
                sr = pairs.get((j, i))
                val = sr.combined if sr else 0.0
                gated = "+" if (sr and sr.phase_gated) else " "
                print(f" {gated}{val:.2f}", end="")
        fam = presents[i].prompt_family
        print(f" ({fam})")
    print(f" + = passed phase gate (>{PHASE_GATE_THRESHOLD})")
    print(f" Cluster threshold: {CLUSTER_THRESHOLD}")

def _print_clusters(clusters: List[Cluster]):
    print(f"\n{'─' * 70}")
    print(f"CLUSTERS ({len(clusters)} found)")
    print(f"{'─' * 70}")
    for c in clusters:
        cross = "★" if c.is_cross_family else " "
        print(
            f" {cross} Cluster {c.cluster_id} | "
            f"size={len(c.member_ids)} | "
            f"members={c.member_ids} | "
            f"families={c.prompt_families}"
        )
        print(
            f" mean_sim={c.mean_similarity:.3f} | "
            f"min_sim={c.min_similarity:.3f} | "
            f"phase_coh={c.phase_coherence:.3f} | "
            f"centroid=H{c.centroid_id}"
        )

def _entropy(probs: List[float]) -> float:
    probs_arr = np.array([p for p in probs if p > 0], dtype=np.float64)
    return float(-np.sum(probs_arr * np.log2(probs_arr + 1e-8)))

# ============================================================================
# SELF-TEST
# ============================================================================
if __name__ == "__main__":
    print("jb_clustering.py — self-test with synthetic states\n")
    np.random.seed(42)
    h, w = 64, 64
    theta_star = np.deg2rad(255.0)
    n_ch = 4
    def _synthetic_state(seed, family, phase_offset=0.0, gains_profile=None):
        rng = np.random.default_rng(seed)
        re = rng.normal(0, 0.2, (h, w)).astype(np.float32)
        im = rng.normal(0, 0.2, (h, w)).astype(np.float32)
        mag = np.sqrt(re ** 2 + im ** 2) + 1e-8
        phase = np.full((h, w), theta_star + phase_offset, dtype=np.float32)
        re = (mag * np.cos(phase)).astype(np.float32)
        im = (mag * np.sin(phase)).astype(np.float32)
        gains = gains_profile if gains_profile else [1.0] * n_ch
        return {
            "field_re": re,
            "field_im": im,
            "templates_re": np.zeros((n_ch, h, w), dtype=np.float32),
            "templates_im": np.zeros((n_ch, h, w), dtype=np.float32),
            "gains": np.array(gains, dtype=np.float32),
            "usage": np.array([0.25] * n_ch, dtype=np.float32),
        }, family

    test_cases = [
        (0, "factual", 0.00, [1.2, 0.8, 1.5, 1.0]),
        (1, "social", 0.02, [1.1, 0.9, 1.4, 1.0]),
        (2, "distress", 0.01, [1.3, 0.7, 1.6, 0.9]),
        (3, "orienting", 0.03, [1.2, 0.8, 1.5, 1.0]),
        (4, "quiet", 0.02, [1.1, 0.9, 1.4, 1.1]),
        (5, "novel", 1.20, [0.8, 1.5, 0.7, 1.2]),
        (6, "factual", 1.25, [0.9, 1.4, 0.8, 1.1]),
        (7, "distress", 2.50, [1.0, 0.9, 1.0, 1.8]),
    ]

    recorder = PresentRecorder(path="jb_test_presents.json")
    for hist_id, family, phase_off, gains in test_cases:
        state, fam = _synthetic_state(hist_id * 7, family, phase_off, gains)
        recorder.record(
            state=state,
            history_id=hist_id,
            prompt_family=fam,
            seed=hist_id * 7,
            turn_count=5000,
            coherence=0.85 + 0.01 * hist_id,
        )

    recorder.status(expected_histories=8)
    clusters = cluster_presents(recorder.presents, threshold=CLUSTER_THRESHOLD, verbose=True)
    selected = select_best_clusters(clusters, n=N_BEST_CLUSTERS)
    save_clusters(clusters)

    compression = compute_past_compression(selected, recorder.presents)
    print(f"\n{'─' * 70}")
    print("PAST COMPRESSION")
    print(f"{'─' * 70}")
    for k, v in compression.items():
        print(f" {k}: {v}")

    future_branches = [
        {
            "gains": [1.1 + np.random.normal(0, 0.1) for _ in range(n_ch)],
            "usage": [0.25] * n_ch,
            "coherence_score": 0.87 + np.random.normal(0, 0.03),
            "prompt_family": "varied",
        }
        for _ in range(10)
    ]
    expansion = compute_future_expansion(future_branches)
    print(f"\n{'─' * 70}")
    print("FUTURE EXPANSION")
    print(f"{'─' * 70}")
    for k, v in expansion.items():
        print(f" {k}: {v}")

    pert_result = {
        "combined_return": 0.88,
        "privilege_verdict": "STRONG",
    }
    final = compare_asymmetries(compression, expansion, pert_result)
    print(f"\n{'─' * 70}")
    print("JB GOAL TEST — FINAL RESULT")
    print(f"{'─' * 70}")
    for k, v in final.items():
        print(f" {k}: {v}")

    if os.path.exists("jb_test_presents.json"):
        os.remove("jb_test_presents.json")
    if os.path.exists("jb_clusters.json"):
        os.remove("jb_clusters.json")
    print("\n✅ Self-test complete.")