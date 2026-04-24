

"""
memory_gate.py — Frankie selective memory system

Architecture:
    Prompt arrives
        ↓
    [MemoryGate.retrieve()]     → injects relevant context into system prompt
        ↓
    [ModalRouter + Adapter]     → generates response
        ↓
    [MemoryGate.ingest_turn()]  → scores turn, detects shift, accumulates thread
        ↓
    [MemoryGate.consolidate()]  → triggered on subject shift or session end

Design principles:
    - Nothing is deleted. Threads get a composite weight tag and stay.
    - Gate scores (home/analytic/engagement) drive memory weighting.
    - Retrieval blends semantic similarity with composite weight.
    - Summarisation is extractive v1 — marked for LLM upgrade later.
    - Defensive throughout: embedder failures, corrupt JSON, edge cases handled.
"""

import json
import uuid
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# =========================
# CONFIG
# =========================
MEMORY_DIR_DEFAULT        = "memory"
MEMORY_CONTEXT_BUDGET     = 400      # max tokens injected into system prompt
SUBJECT_SHIFT_THRESHOLD   = 0.25     # cosine similarity drop → new thread
CONSOLIDATION_MIN_TURNS   = 3        # minimum turns before consolidating
HIGH_WEIGHT_THRESHOLD     = 0.55
MEDIUM_WEIGHT_THRESHOLD   = 0.25
RETRIEVAL_FLOOR           = 0.25     # minimum retrieval score to inject

# Composite weight coefficients
W_ANALYTIC   = 0.35
W_ENGAGEMENT = 0.6
W_HOME       = 0.05

# Retrieval blend
W_SEMANTIC   = 0.55
W_COMPOSITE  = 0.45

# Explicit consolidation trigger phrases
CONSOLIDATION_TRIGGERS = [
    "okay, moving on",
    "thanks for that",
    "let's move on",
    "that's sorted",
    "got it, next",
    "moving on",
    "that's enough on that",
    "let's talk about something else",
    "different topic",
    "new topic",
]

SHARED_EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# DATA STRUCTURES
# =========================
@dataclass
class Turn:
    turn_id:       str
    timestamp:     str
    prompt:        str
    response:      str
    modal_scores:  Dict[str, float]
    dominant_mode: str
    route_name:    str


@dataclass
class ThreadStrength:
    emotional_weight: float   # engagement-derived
    topic_importance: float   # analytic-derived
    composite:        float
    tag:              str     # "high" / "medium" / "low"


@dataclass
class TopicThread:
    thread_id:     str
    topic_summary: str
    first_seen:    str
    last_seen:     str
    turn_count:    int
    strength:      ThreadStrength
    modal_context: str
    embedding:     List[float]
    turns:         List[Turn] = field(default_factory=list)
    consolidated:  bool = False


# =========================
# MEMORY GATE
# =========================
class MemoryGate:
    def __init__(
        self,
        memory_dir:     str = MEMORY_DIR_DEFAULT,
        session_id:     Optional[str] = None,
        embedder:       Optional[SentenceTransformer] = None,
        context_budget: int = MEMORY_CONTEXT_BUDGET,
    ) -> None:
        self.memory_dir     = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.session_id     = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.embedder       = embedder or SHARED_EMBEDDER
        self.context_budget = context_budget

        self.threads:        List[TopicThread]    = []
        self.active_thread:  Optional[TopicThread] = None
        self.last_embedding: Optional[torch.Tensor] = None

        self._load()

    # =========================
    # PUBLIC API
    # =========================
    def retrieve(self, prompt: str) -> str:
        """
        Called BEFORE modal routing.
        Returns formatted memory context for system prompt injection.
        Returns empty string if nothing relevant exists.
        """
        if not self.threads:
            return ""

        try:
            prompt_embedding = self._embed(prompt)
        except Exception as e:
            logger.warning(f"[Memory] Embed failed during retrieve: {e}")
            return ""

        ranked = self._rank_threads(prompt_embedding)
        if not ranked:
            return ""

        return self._format_context(ranked)

    def ingest_turn(
        self,
        prompt:       str,
        response:     str,
        modal_scores: Dict[str, float],
        privileges:   Dict[str, float],
    ) -> None:
        """
        Called AFTER generation.
        Scores the turn, detects subject shift, accumulates into active thread.
        Triggers consolidation on shift or explicit trigger.
        """
        turn = Turn(
            turn_id       = str(uuid.uuid4())[:8],
            timestamp     = datetime.now().isoformat(),
            prompt        = prompt,
            response      = response,
            modal_scores  = {
                "home":       modal_scores.get("home",       0.0),
                "analytic":   modal_scores.get("analytic",   0.0),
                "engagement": modal_scores.get("engagement", 0.0),
            },
            dominant_mode = privileges.get("dominant_mode", "home"),
            route_name    = privileges.get("route_name",    "home"),
        )

        try:
            prompt_embedding = self._embed(prompt)
        except Exception as e:
            logger.warning(f"[Memory] Embed failed during ingest: {e}")
            return

        explicit_trigger = self._is_explicit_trigger(prompt)
        subject_shift    = self._detect_subject_shift(prompt_embedding)

        if explicit_trigger or subject_shift:
            self._try_consolidate_active()
            # active_thread is now None — new thread will open below

        # Open new thread or accumulate into existing
        if self.active_thread is None:
            self.active_thread = self._open_thread(turn, prompt_embedding)
        else:
            self._accumulate_turn(self.active_thread, turn)

        self.last_embedding = prompt_embedding

        try:
            self._save()
        except Exception as e:
            logger.warning(f"[Memory] Save failed: {e}")

    def consolidate(self, thread: Optional[TopicThread] = None) -> None:
        """
        Consolidate a specific thread, or the active thread if none given.
        Safe to call even if active_thread is None or too short.
        """
        target = thread if thread is not None else self.active_thread

        if target is None:
            return

        # Guard: don't consolidate threads that are too short
        if target.turn_count < 1:
 
            self._discard_thread(target)
            return

        try:
            target.strength      = self._compute_strength(target.turns)
            target.topic_summary = self._summarise_thread(target.turns)
            target.embedding     = self._embed(target.topic_summary).tolist()
        except Exception as e:
            logger.warning(f"[Memory] Consolidation failed for {target.thread_id}: {e}")
            self._discard_thread(target)
            return

        target.consolidated = True
        target.last_seen    = datetime.now().isoformat()

        if target not in self.threads:
            self.threads.append(target)

        if target is self.active_thread:
            self.active_thread = None

        logger.info(
            f"[Memory] Consolidated '{target.topic_summary[:60]}' "
            f"[{target.strength.tag}, composite={target.strength.composite:.2f}]"
        )
    
    def end_session(self) -> None:
        """Consolidate any open thread at session end."""
        if self.active_thread is not None:
            self.consolidate(self.active_thread)

    # =========================
    # INTERNAL — THREAD FLOW
    # =========================
    def _try_consolidate_active(self) -> None:
        """
        Attempt to consolidate the active thread.
        Handles the None case and the too-short case cleanly.
        Sets active_thread to None regardless of outcome.
        """
        if self.active_thread is None:
            return
        self.consolidate(self.active_thread)
        # consolidate() sets active_thread to None when target is active_thread
        # but if it was discarded, we need to ensure cleanup
        self.active_thread = None

    def _open_thread(
        self,
        turn:      Turn,
        embedding: torch.Tensor,
    ) -> TopicThread:
        """Create a fresh thread from the first turn."""
        return TopicThread(
            thread_id     = str(uuid.uuid4())[:12],
            topic_summary = turn.prompt[:120],
            first_seen    = datetime.now().isoformat(),
            last_seen     = datetime.now().isoformat(),
            turn_count    = 1,
            strength      = ThreadStrength(0.0, 0.0, 0.0, "low"),
            modal_context = turn.dominant_mode,
            embedding     = embedding.tolist(),
            turns         = [turn],
            consolidated  = False,
        )

    def _accumulate_turn(self, thread: TopicThread, turn: Turn) -> None:
        """Add a turn to an existing thread and update metadata."""
        thread.turns.append(turn)
        thread.turn_count += 1
        thread.last_seen   = datetime.now().isoformat()

        # Dominant modal context by majority vote across all turns
        mode_counts: Dict[str, int] = {}
        for t in thread.turns:
            mode_counts[t.dominant_mode] = mode_counts.get(t.dominant_mode, 0) + 1
        thread.modal_context = max(mode_counts, key=mode_counts.get)

    def _discard_thread(self, thread: TopicThread) -> None:
        """Remove a thread from all state cleanly."""
        if thread in self.threads:
            self.threads.remove(thread)
        if thread is self.active_thread:
            self.active_thread = None

    # =========================
    # INTERNAL — SUBJECT SHIFT
    # =========================
    def _detect_subject_shift(self, prompt_embedding: torch.Tensor) -> bool:
        """
        Returns True if cosine similarity to last prompt drops below threshold.
        Returns False if no previous embedding exists (first turn).
        """
        if self.last_embedding is None:
            return False
        try:
            sim = torch.nn.functional.cosine_similarity(
                prompt_embedding.unsqueeze(0),
                self.last_embedding.unsqueeze(0),
            ).item()
            return sim < SUBJECT_SHIFT_THRESHOLD
        except Exception as e:
            logger.warning(f"[Memory] Subject shift detection failed: {e}")
            return False

    def _is_explicit_trigger(self, prompt: str) -> bool:
        p = prompt.lower().strip()
        return any(trigger in p for trigger in CONSOLIDATION_TRIGGERS)

    # =========================
    # INTERNAL — COMPOSITE WEIGHT
    # =========================
    def _compute_strength(self, turns: List[Turn]) -> ThreadStrength:
        if not turns:
            return ThreadStrength(0.0, 0.0, 0.0, "low")

        n = len(turns)
        analytic_sum   = sum(t.modal_scores.get("analytic",   0.0) for t in turns)
        engagement_sum = sum(t.modal_scores.get("engagement", 0.0) for t in turns)
        home_sum       = sum(t.modal_scores.get("home",       0.0) for t in turns)

        topic_importance  = analytic_sum   / n
        emotional_weight  = engagement_sum / n
        home_contribution = home_sum       / n

        composite = (
            topic_importance  * W_ANALYTIC   +
            emotional_weight  * W_ENGAGEMENT +
            home_contribution * W_HOME
        )

        tag = (
            "high"   if composite >= HIGH_WEIGHT_THRESHOLD   else
            "medium" if composite >= MEDIUM_WEIGHT_THRESHOLD else
            "low"
        )

        return ThreadStrength(
            emotional_weight = round(emotional_weight, 4),
            topic_importance = round(topic_importance, 4),
            composite        = round(composite, 4),
            tag              = tag,
        )

    # =========================
    # INTERNAL — SUMMARISATION
    # =========================
    def _summarise_thread(self, turns: List[Turn]) -> str:
        """
        V1: Lightweight extractive summary.
        Takes first prompt as anchor; appends last prompt if topic has drifted.

        TODO (v2): Replace with a local model call once LoRA stack is stable.
        Suggested call signature:
            summary = self.local_model_summarise(
                turns=turns,
                max_tokens=80,
                instruction="Summarise the topic of this conversation thread
                             in one or two sentences. Be specific and concise."
            )
        """
        if not turns:
            return ""

        first = turns[0].prompt.strip()

        if len(turns) == 1:
            return first[:200]

        last = turns[-1].prompt.strip()

        if first == last:
            return first[:200]

        # Check semantic drift between first and last prompt
        try:
            first_emb = self._embed(first)
            last_emb  = self._embed(last)
            sim = torch.nn.functional.cosine_similarity(
                first_emb.unsqueeze(0),
                last_emb.unsqueeze(0),
            ).item()
            if sim < 0.7:
                return f"{first[:120]} … {last[:80]}"
        except Exception as e:
            logger.warning(f"[Memory] Summary drift check failed: {e}")

        return first[:200]

    # =========================
    # INTERNAL — RETRIEVAL
    # =========================
    def _rank_threads(
        self,
        prompt_embedding: torch.Tensor,
    ) -> List[Tuple[float, TopicThread]]:
        """
        Rank consolidated threads by retrieval score.
        retrieval_score = semantic_sim * W_SEMANTIC + composite * W_COMPOSITE
        """
        ranked = []
        for thread in self.threads:
            if not thread.consolidated:
                continue
            try:
                thread_emb   = torch.tensor(thread.embedding)
                semantic_sim = torch.nn.functional.cosine_similarity(
                    prompt_embedding.unsqueeze(0),
                    thread_emb.unsqueeze(0),
                ).item()
                retrieval_score = (
                    semantic_sim              * W_SEMANTIC +
                    thread.strength.composite * W_COMPOSITE
                )
                ranked.append((retrieval_score, thread))
            except Exception as e:
                logger.warning(f"[Memory] Ranking failed for thread {thread.thread_id}: {e}")
                continue

        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked

    def _format_context(
        self,
        ranked: List[Tuple[float, TopicThread]],
    ) -> str:
        """
        Fill context budget greedily from top of ranked list.
        Stops at RETRIEVAL_FLOOR or when budget is exhausted.
        """
        lines       = ["[Memory context — relevant prior topics:]"]
        char_budget = self.context_budget * 4  # rough chars-per-token

        for score, thread in ranked:
            if score < RETRIEVAL_FLOOR:
                break

            entry = (
                f"- [{thread.strength.tag.upper()}] {thread.topic_summary}"
            )

            if len("\n".join(lines)) + len(entry) > char_budget:
                break

            lines.append(entry)   

        return "\n".join(lines) if len(lines) > 1 else ""

    # =========================
    # INTERNAL — EMBED
    # =========================
    def _embed(self, text: str) -> torch.Tensor:
        """Embed text. Raises on failure — callers handle the exception."""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        return self.embedder.encode(text.strip(), convert_to_tensor=True)

    # =========================
    # INTERNAL — PERSISTENCE
    # =========================
    def _memory_path(self) -> Path:
        return self.memory_dir / f"session_{self.session_id}.json"

    def _save(self) -> None:
        payload = {
            "session_id": self.session_id,
            "saved_at":   datetime.now().isoformat(),
            "threads":    [self._serialise_thread(t) for t in self.threads],
        }
        with open(self._memory_path(), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _load(self) -> None:
        path = self._memory_path()
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            loaded = 0
            for t in payload.get("threads", []):
                try:
                    self.threads.append(self._deserialise_thread(t))
                    loaded += 1
                except Exception as e:
                    logger.warning(f"[Memory] Skipping corrupt thread record: {e}")
            logger.info(f"[Memory] Loaded {loaded} threads from {path}")
        except json.JSONDecodeError as e:
            logger.error(f"[Memory] JSON corrupt, starting fresh: {e}")
        except Exception as e:
            logger.error(f"[Memory] Load failed, starting fresh: {e}")

    def _serialise_thread(self, t: TopicThread) -> Dict:
        return asdict(t)

    def _deserialise_thread(self, d: Dict) -> TopicThread:
        d           = dict(d)
        d["strength"] = ThreadStrength(**d["strength"])
        d["turns"]    = [Turn(**turn) for turn in d["turns"]]
        return TopicThread(**d)


# =========================
# INTEGRATION NOTES
# =========================
"""
── __init__ in ModalOrchestrator ──────────────────────────────────────────

    from memory_gate import MemoryGate

    self.memory = MemoryGate(
        memory_dir  = "memory",
        session_id  = self._timestamp_slug(),
        embedder    = self.embedder,   # reuse shared embedder — no extra load
    )

── run_prompt — three insertion points ────────────────────────────────────

    # 1. BEFORE build_system_prompt_and_settings:
    memory_context = self.memory.retrieve(prompt)

    # 2. INSIDE build_system_prompt_and_settings (or just before the call):
    #    pass memory_context in and prepend to base_prompt:
    if memory_context:
        base_prompt = memory_context + "\n\n" + base_prompt

    # 3. AFTER response is generated, before return:
    self.memory.ingest_turn(
        prompt       = prompt,
        response     = response,
        modal_scores = scores,
        privileges   = privileges,
    )



── Session end ─────────────────────────────────────────────────────────────

    # Add to a shutdown hook or call explicitly after run_batch:
    self.memory.end_session()

── Cross-session memory (future) ───────────────────────────────────────────

    # Pass a fixed session_id to load a prior session's threads:
    self.memory = MemoryGate(session_id="persistent", ...)
    # All threads from that file will be loaded and available for retrieval.
    # Consolidation will append new threads to the same file.
"""
