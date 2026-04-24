import math
import re
import json
import csv
import urllib.request
import urllib.error
import base64
import mimetypes
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import requests
import torch

from pond_v6_5 import (
    DEVICE,
    Pond,
    text_to_ripple,
    detect_roles,
    detect_groups,
    phase_lock_error,
    spatial_coherence,
    c_add,
    c_scale,
    c_abs,
    admissibility_project,
    overlap_score,
    c_normalize,
)

# ============================================================================
# KNOBS
# ============================================================================

TEXT_MODEL_DEFAULT = "qwen3-4b-instruct-2507.gguf"
TEXT_BASE_URL_DEFAULT = "http://127.0.0.1:1234/v1/chat/completions"

VISION_MODEL_DEFAULT = "qwen/qwen3-vl-8b"
VISION_BASE_URL_DEFAULT = "http://127.0.0.1:1234/v1/chat/completions"

LMSTUDIO_MAX_TOKENS = 200
LMSTUDIO_TIMEOUT = 240

VISION_MAX_TOKENS = 250
VISION_TIMEOUT = 60

N_CANDIDATES = 3
REQUEST_LINES = 4
BASE_TEMPERATURE = 0.32
TEMPERATURE_MAX = 0.65

REPLY_FEEDBACK_STRENGTH = 0.18
STATE_FEEDBACK_STRENGTH = 0.35
STATE_ECHO_STRENGTH = 0.05

USE_FRANKIE_NATIVE_CANDIDATE = True
USE_LLM_PHRASER = True

W_LOCK_ALIGNMENT    = 1.20
W_MEMORY_RESONANCE  = 1.35
W_COHERENCE         = 1.00
W_SALIENCE          = 0.55
W_REPETITION        = 0.20
W_ROLE_FIT          = 1.15
W_DECODED_MATCH     = 1.00
W_TONE              = 0.45
W_COMPLETION        = 0.75
W_STYLE             = 0.90
W_CONTRADICTION     = 1.80
W_VISION_MATCH      = 0.90

EXACT_ECHO_PENALTY          = 1.00
SHORT_FRAGMENT_PENALTY      = 0.50
UNFINISHED_TAIL_PENALTY     = 1.00
NO_SENTENCE_CLOSURE_PENALTY = 0.18
MEMORY_DENIAL_PENALTY       = 0.55
REFUSAL_PENALTY             = 0.35
META_REASONING_PENALTY      = 1.00

EXACT_ECHO_REP    = -0.15
DECODED_TOKEN_REP =  0.30

COMPLETE_SENTENCE_BONUS = 0.35
MIN_LENGTH_BONUS        = 0.15
FINISHED_TAIL_BONUS     = 0.20

ROLEFIT_SELF_NAME             = 0.60
ROLEFIT_OTHER_NAME            = 0.95
ROLEFIT_MEMORY                = 0.85
ROLEFIT_TIME                  = 0.70
ROLEFIT_SLOT_MATCH            = 0.30
ROLEFIT_IDENTITY_BASIN_PENALTY = 0.35

DECODED_MATCH_MEMORY            = 0.55
DECODED_MATCH_PETER             = 0.55
DECODED_MATCH_FRANKIE           = 0.60
DECODED_MATCH_DAY               = 0.50
DECODED_MATCH_BASIN_ECHO_PENALTY = 0.40

FRANKIE_STYLE_BONUS  = 0.30
SERVICE_BOT_PENALTY  = 0.45

FRANKIE_STYLE_PHRASES = [
    "i think",
    "older pattern",
    "happy to chat",
    "good to hear",
    "morning, mate",
    "i'm here and listening",
    "i'm frankie",
    "that's you",
    "feels familiar",
    "warm and familiar",
    "feels just right",
    "i remember you",
]

SERVICE_BOT_PHRASES = [
    "how can i help you today",
    "glad i could help",
    "nice to meet you",
    "let me know what you'd like",
    "what you'd like to explore next",
    "how may i help",
]

UNFINISHED_TAILS = (
    "a", "an", "and", "are", "as", "at", "be", "but", "for",
    "here", "i", "if", "in", "is", "it", "little", "my", "of",
    "on", "or", "remember", "so", "that", "the", "their", "there",
    "this", "to", "we", "with", "you", "your"
)

BAD_META_PREFIXES = [
    "we need", "the user asks", "internal state", "decoded is", "decoded=",
    "so we should", "the rule", "instructions", "frankie state",
    "you are speaking", "the internal decoded", "the decoded says",
    "policy=", "tone=", "user:", "assistant:", "possibly", "probably",
    "must be", "that's fuzzy", "that's a clear", "based on decoded",
    "we have a claim", "output only", "following constraints",
]

POSITIVE_TONE_WORDS = ["gentle", "calm", "warm", "kind", "soft", "mate", "glad", "good", "hello", "sorry", "happy"]
NEGATIVE_TONE_WORDS = ["stupid", "idiot", "angry", "hate", "shut up", "damn", "useless", "wrong with you"]

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

VISION_STOPWORDS = {
    "a", "an", "the", "is", "are", "with", "and", "or", "of", "in", "on", "at",
    "to", "from", "for", "by", "it", "this", "that", "there", "here", "object",
    "position", "color", "colors", "shape", "background", "visible", "centered",
    "image", "single", "plain", "soft", "shadow", "areas", "particular",
    "traditional", "classic"
}

VISION_PRIORITY_WORDS = {
    "cup", "pretzel", "apple", "carrot", "rose", "lid", "stem", "petals", "loops",
    "twist", "twisted", "knot", "root", "tip", "bread", "mug", "plate", "bowl",
    "bottle", "chair", "table", "book", "bird", "flower", "leaf", "green", "red",
    "golden", "golden-brown", "brown", "white", "orange", "yellow", "upright",
    "diagonal", "center", "centered"
}


# ============================================================================
# Metrics
# ============================================================================

@dataclass
class SessionMetrics:
    session_start: str = field(default_factory=lambda: datetime.now().isoformat())
    total_turns: int = 0

    frankie_native_wins: int = 0
    llm_wins: int = 0

    oom_fallbacks: int = 0
    empty_candidate_fallbacks: int = 0

    questions_asked: int = 0

    total_winning_score: float = 0.0
    total_lock_alignment: float = 0.0
    total_memory_resonance: float = 0.0
    total_coherence_score: float = 0.0
    total_role_fit: float = 0.0
    total_contradiction_penalty: float = 0.0

    mode_counts: Dict[str, int] = field(default_factory=dict)
    turn_log: List[Dict] = field(default_factory=list)

    def record_turn(
        self,
        turn: int,
        user_text: str,
        reply: str,
        mode: str,
        chosen: "CandidateScore",
        native_seed: Optional[str],
        n_candidates: int,
        is_oom_recovery: bool = False,
        is_empty_fallback: bool = False,
    ):
        self.total_turns += 1

        native_won = (native_seed is not None and reply.strip() == native_seed.strip())
        if native_won:
            self.frankie_native_wins += 1
        else:
            self.llm_wins += 1

        if is_oom_recovery:
            self.oom_fallbacks += 1
        if is_empty_fallback:
            self.empty_candidate_fallbacks += 1

        if "?" in reply:
            self.questions_asked += 1

        self.total_winning_score       += chosen.total
        self.total_lock_alignment      += chosen.lock_alignment
        self.total_memory_resonance    += chosen.memory_resonance
        self.total_coherence_score     += chosen.coherence_score
        self.total_role_fit            += chosen.role_fit
        self.total_contradiction_penalty += chosen.contradiction_penalty

        self.mode_counts[mode] = self.mode_counts.get(mode, 0) + 1

        self.turn_log.append({
            "turn": turn,
            "timestamp": datetime.now().isoformat(),
            "user_text": user_text,
            "reply": reply,
            "mode": mode,
            "native_won": native_won,
            "n_candidates": n_candidates,
            "is_oom_recovery": is_oom_recovery,
            "is_empty_fallback": is_empty_fallback,
            "reply_has_question": "?" in reply,
            "score_total": round(chosen.total, 4),
            "score_lock": round(chosen.lock_alignment, 4),
            "score_mem": round(chosen.memory_resonance, 4),
            "score_coh": round(chosen.coherence_score, 4),
            "score_sal": round(chosen.salience_score, 4),
            "score_role": round(chosen.role_fit, 4),
            "score_decoded": round(chosen.decoded_match, 4),
            "score_tone": round(chosen.tone_match, 4),
            "score_style": round(chosen.style_match, 4),
            "score_completion": round(chosen.completion_quality, 4),
            "score_vision": round(chosen.vision_match, 4),
            "score_contradiction": round(chosen.contradiction_penalty, 4),
        })

    def averages(self) -> Dict:
        n = max(self.total_turns, 1)
        return {
            "avg_winning_score":        round(self.total_winning_score / n, 4),
            "avg_lock_alignment":       round(self.total_lock_alignment / n, 4),
            "avg_memory_resonance":     round(self.total_memory_resonance / n, 4),
            "avg_coherence_score":      round(self.total_coherence_score / n, 4),
            "avg_role_fit":             round(self.total_role_fit / n, 4),
            "avg_contradiction_penalty":round(self.total_contradiction_penalty / n, 4),
        }

    def summary(self) -> str:
        n = max(self.total_turns, 1)
        avgs = self.averages()
        lines = [
            "",
            "╔══════════════════════════════════════════════════════╗",
            "║              SESSION METRICS SUMMARY                  ║",
            "╠══════════════════════════════════════════════════════╣",
            f"║  Session started : {self.session_start[:19]:<33}║",
            f"║  Total turns     : {self.total_turns:<33}║",
            "╠══════════════════════════════════════════════════════╣",
            f"║  Frankie native wins : {self.frankie_native_wins:<29}║",
            f"║  LLM candidate wins  : {self.llm_wins:<29}║",
            f"║  Native win rate     : {self.frankie_native_wins/n*100:>5.1f}%{'':<24}║",
            "╠══════════════════════════════════════════════════════╣",
            f"║  Questions asked     : {self.questions_asked:<29}║",
            f"║  OOM fallbacks       : {self.oom_fallbacks:<29}║",
            f"║  Empty fallbacks     : {self.empty_candidate_fallbacks:<29}║",
            "╠══════════════════════════════════════════════════════╣",
            "║  AVERAGE SCORES                                        ║",
            f"║    Winning total   : {avgs['avg_winning_score']:<33}║",
            f"║    Lock alignment  : {avgs['avg_lock_alignment']:<33}║",
            f"║    Mem resonance   : {avgs['avg_memory_resonance']:<33}║",
            f"║    Coherence       : {avgs['avg_coherence_score']:<33}║",
            f"║    Role fit        : {avgs['avg_role_fit']:<33}║",
            f"║    Contradiction   : {avgs['avg_contradiction_penalty']:<33}║",
            "╠══════════════════════════════════════════════════════╣",
            "║  MODE DISTRIBUTION                                     ║",
        ]
        for mode, count in sorted(self.mode_counts.items(), key=lambda x: -x[1]):
            pct = count / n * 100
            lines.append(f"║    {mode:<22}: {count:>3} ({pct:>5.1f}%)              ║")
        lines.append("╚══════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def export_csv(self, path: str = "frankie_metrics.csv"):
        if not self.turn_log:
            print("No turns to export.")
            return
        keys = list(self.turn_log[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.turn_log)
        print(f"Metrics exported to {path} ({len(self.turn_log)} rows)")


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class FrankieState:
    user_text: str
    pond_out: Dict
    decoded: Optional[str]
    roles: List[str]
    groups: List[str]
    summary: str
    vision_desc: Optional[str] = None


@dataclass
class IntentionPacket:
    intention_to_speak: bool
    mode: str
    slot: int
    confidence: float
    tone: str
    decoded: str
    feeling: str
    seed: str


@dataclass
class CandidateScore:
    text: str
    total: float
    lock_alignment: float
    memory_resonance: float
    coherence_score: float
    salience_score: float
    repetition_affinity: float
    role_fit: float
    decoded_match: float
    tone_match: float
    completion_quality: float
    style_match: float
    contradiction_penalty: float
    vision_match: float
    notes: List[str]


# ============================================================================
# Helpers
# ============================================================================

def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        k = x.strip()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def clean_reply(text: str) -> str:
    return " ".join(text.strip().split())


def quick_candidate_prefilter(text: str) -> bool:
    t = clean_reply(text)
    low = t.lower()
    if len(t.split()) <= 3:
        return False
    if any(meta in low for meta in BAD_META_PREFIXES):
        return False
    return True


def extract_lines_as_candidates(text: str) -> List[str]:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    lines = [ln.strip(" -\t").strip() for ln in text.splitlines() if ln.strip()]
    out = []
    for ln in lines:
        low = ln.lower()
        if any(low.startswith(p) for p in BAD_META_PREFIXES):
            continue
        ln = ln.strip().strip('"').strip("'").strip()
        if ln and quick_candidate_prefilter(ln):
            out.append(clean_reply(ln))
    return dedupe_keep_order(out)


def sentiment_hint(text: str) -> float:
    t = text.lower()
    score = 0.0
    for w in POSITIVE_TONE_WORDS:
        if w in t:
            score += 0.15
    for w in NEGATIVE_TONE_WORDS:
        if w in t:
            score -= 0.35
    return max(-1.0, min(1.0, score))


def style_hint(text: str) -> float:
    t = text.lower()
    score = 0.0
    for p in FRANKIE_STYLE_PHRASES:
        if p in t:
            score += FRANKIE_STYLE_BONUS
    for p in SERVICE_BOT_PHRASES:
        if p in t:
            score -= SERVICE_BOT_PENALTY
    return max(-1.0, min(1.0, score))


def compress_vision_desc(text: str, max_sentences: int = 2, max_words: int = 32) -> str:
    text = " ".join(text.strip().split())
    if not text:
        return text
    sentences = re.split(r'(?<=[.!?])\s+', text)
    kept = " ".join(sentences[:max_sentences]).strip()
    words = kept.split()
    if len(words) > max_words:
        kept = " ".join(words[:max_words]).rstrip(",;:- ")
        if kept and kept[-1] not in ".!?":
            kept += "."
    return kept


def normalize_vision_desc(text: str) -> str:
    text = text.strip()
    text = text.replace("- ", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_vision_keywords(text: str, max_keywords: int = 8) -> List[str]:
    toks = re.findall(r"[a-zA-Z][a-zA-Z\-]*", text.lower())
    ranked: List[str] = []
    for tok in toks:
        if tok in VISION_PRIORITY_WORDS and tok not in ranked:
            ranked.append(tok)
    for tok in toks:
        if tok in VISION_STOPWORDS:
            continue
        if len(tok) <= 2:
            continue
        if tok not in ranked:
            ranked.append(tok)
    return ranked[:max_keywords]


# ============================================================================
# Backend
# ============================================================================

class BaseBackend:
    def generate_phrasings(self, prompt: str) -> List[str]:
        raise NotImplementedError


class EchoBackend(BaseBackend):
    def generate_phrasings(self, prompt: str) -> List[str]:
        return [
            "I think your name is Peter.",
            "I'm Frankie — happy to chat.",
            "I think I remember an older pattern of us.",
            "Morning, mate.",
            "I think today is Saturday.",
        ][:REQUEST_LINES]


class LMStudioBackend(BaseBackend):
    def __init__(
        self,
        model: str = TEXT_MODEL_DEFAULT,
        base_url: str = TEXT_BASE_URL_DEFAULT,
        max_tokens: int = LMSTUDIO_MAX_TOKENS,
        timeout: int = LMSTUDIO_TIMEOUT,
    ):
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _one(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(BASE_TEMPERATURE),
            "max_tokens": int(self.max_tokens),
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LM Studio HTTP error {e.code}: {body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"LM Studio connection failed: {e}")
        try:
            return obj["choices"][0]["message"]["content"].strip()
        except Exception:
            raise RuntimeError(f"Unexpected LM Studio response: {obj}")

    def generate_phrasings(self, prompt: str) -> List[str]:
        text = self._one(prompt)
        return extract_lines_as_candidates(text)


# ============================================================================
# Bridge
# ============================================================================

class FrankieLLMBridge:
    def __init__(
        self,
        backend: BaseBackend,
        vision_model: str = VISION_MODEL_DEFAULT,
        vision_base_url: str = VISION_BASE_URL_DEFAULT,
    ):
        self.backend = backend
        self.vision_model = vision_model
        self.vision_base_url = vision_base_url

        self.reply_feedback_strength = REPLY_FEEDBACK_STRENGTH
        self.state_feedback_strength = STATE_FEEDBACK_STRENGTH
        self.state_echo_strength = STATE_ECHO_STRENGTH

        self.pond = Pond().to(DEVICE)
        self.pond.load()

        self.rejected_phrasings: List[Dict] = []
        self.metrics = SessionMetrics()

    def compact_summary(self, user_text: str, pond_out: Dict) -> str:
        roles = ",".join(detect_roles(user_text)) or "-"
        decoded = pond_out["decoded"] if pond_out["decoded"] else "none"
        return (
            f"slot={pond_out['recall_slot']} "
            f"recall={pond_out['recall_score']:.3f} "
            f"salience={pond_out['salience']:.3f} "
            f"decoded={decoded} "
            f"roles={roles}"
        )

    def get_vision_description(self, image_path: str) -> Optional[str]:
        if not os.path.exists(image_path):
            print(f"[Vision error] File not found: {image_path}")
            return None

        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = "image/jpeg"

        try:
            with open(image_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

            payload = {
                "model": self.vision_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Describe exactly what is in this image. "
                                    "Describe this image in 30-45 words max. "
                                    "Be factual and complete every sentence. "
                                    "List objects, positions, colors, and any visible text. "
                                    "No interpretation, no emotion, no story, no lists starting with "
                                    "'Object:' or 'Position:'. "
                                    "Write in full, flowing sentences — do not cut off."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{img_base64}"},
                            },
                        ],
                    }
                ],
                "temperature": 0.3,
                "max_tokens": VISION_MAX_TOKENS,
            }

            response = requests.post(self.vision_base_url, json=payload, timeout=VISION_TIMEOUT)
            response.raise_for_status()

            raw = response.json()["choices"][0]["message"]["content"].strip()

            raw = re.sub(r'^(Object|Shape|Color|Position|Background|The image shows)\s*[:;,-]?\s*', '', raw, flags=re.I)
            raw = re.sub(r'\s*-\s*', ' ', raw)
            raw = re.sub(r'\s+', ' ', raw).strip()

            for label in ("Object:", "Shape:", "Color:", "Colors:", "Position:", "Background:"):
                raw = raw.replace(label, "")
            raw = " ".join(raw.split())

            desc = normalize_vision_desc(raw)
            return compress_vision_desc(desc, max_sentences=2, max_words=32)

        except Exception as e:
            print(f"[Vision error] {e}")
            return None

    def build_user_ripple(
        self,
        user_text: str,
        image_path: Optional[str] = None,
        vision_desc: Optional[str] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Optional[str]]:
        ripple = text_to_ripple(user_text, self.pond.H, self.pond.W, self.pond.EMB)
        resolved_vision_desc = vision_desc

        if image_path and resolved_vision_desc is None:
            resolved_vision_desc = self.get_vision_description(image_path)

        if resolved_vision_desc:
            vision_ripple = text_to_ripple(resolved_vision_desc, self.pond.H, self.pond.W, self.pond.EMB)
            ripple = c_add(ripple, vision_ripple)

        return ripple, resolved_vision_desc

    def process_user(
        self,
        user_text: str,
        image_path: Optional[str] = None,
        vision_desc: Optional[str] = None,
    ) -> FrankieState:
        ripple, resolved_vision_desc = self.build_user_ripple(
            user_text=user_text,
            image_path=image_path,
            vision_desc=vision_desc,
        )
        pond_out = self.pond.forward(ripple, text=user_text)
        decoded = pond_out["decoded"]
        roles = detect_roles(user_text)
        groups = detect_groups(user_text)
        summary = self.compact_summary(user_text, pond_out)

        return FrankieState(
            user_text=user_text,
            pond_out=pond_out,
            decoded=decoded,
            roles=roles,
            groups=groups,
            summary=summary,
            vision_desc=resolved_vision_desc,
        )

    def build_intention_packet(self, state: FrankieState) -> IntentionPacket:
        decoded = state.decoded or "none"
        slot = state.pond_out["recall_slot"]
        confidence = float(state.pond_out["recall_score"])
        roles = state.roles
        groups = state.groups

        mode = "open_reply"
        tone = "warm_grounded"
        feeling = "present, listening, open to the next ripple"
        seed = "I'm here and listening."

        user_lower = state.user_text.lower()

        day = next((d for d in DAYS if d in user_lower), None)

        if day is not None or "what day" in user_lower or "today is" in user_lower:
            mode = "time_reply"
            feeling = "time marker present, quiet confidence"
            slot = 3
            if day is None and decoded and isinstance(decoded, str):
                day = next((d for d in DAYS if d in decoded.lower()), None)
            if day is None:
                day = self.pond.infer_time_day_from_slot3()
            seed = f"I think today is {day.capitalize()}." if day else "Hmm… the day feels a bit fuzzy today."

        elif "assert_identity_self" in roles:
            mode = "identity_self_mark"
            feeling = "self-name being anchored, gentle certainty"
            seed = "I'll hold that — your name is Peter."

        elif "query_identity_self" in roles:
            mode = "identity_self_reply"
            feeling = "clear self-name resonance, gentle confidence"
            seed = "I think your name is Peter."

        elif "assert_identity_other" in roles:
            mode = "identity_other_mark"
            feeling = "other-identity being anchored, warm attention"
            seed = "I'll hold that — I'm Frankie."

        elif "query_identity_other" in roles:
            mode = "identity_other_reply"
            feeling = "self-identity present, a little fuzzy, warm"
            seed = "I'm Frankie."

        elif "query_memory" in roles:
            mode = "memory_reply"
            feeling = "familiarity, older pattern returning, calm recognition"
            seed = "I think I remember an older pattern of us."

        elif "greeting" in groups:
            mode = "greeting_reply"
            feeling = "light warmth, easy contact"
            seed = "Morning, mate."

        elif state.vision_desc:
            mode = "vision_reply"
            feeling = "visually grounded, attentive to what is actually there"
            seed = "I can see a few clear details there."

        # Gentle question encouragement
        if self.metrics.total_turns % 4 == 3 and mode == "open_reply":
            mode = "open_reply_with_question"
            seed = seed + " How about you?"

        return IntentionPacket(
            intention_to_speak=True,
            mode=mode,
            slot=slot,
            confidence=confidence,
            tone=tone,
            decoded=decoded,
            feeling=feeling,
            seed=seed,
        )

    def build_phraser_prompt(self, state: FrankieState, intention: IntentionPacket) -> str:
        vision_block = ""
        if state.vision_desc:
            vision_block = f"Vision facts: {state.vision_desc}\n"

        return (
            "You are helping Frankie phrase his own inner intention.\n"
            "Frankie sounds warm, grounded, lightly human, a little playful, and never like a customer-service bot.\n\n"
            "Frankie examples:\n"
            "I think your name is Peter.\n"
            "I'm Frankie — happy to chat.\n"
            "I think I remember an older pattern of us.\n"
            "Morning, mate.\n"
            "Feels just right.\n\n"
            f"Mode: {intention.mode}\n"
            f"Tone: {intention.tone}\n"
            f"Feeling summary: {intention.feeling}\n"
            f"Seed meaning: {intention.seed}\n"
            f"User said: {state.user_text}\n"
            f"{vision_block}\n"
            f"Return exactly {REQUEST_LINES} short natural phrasings.\n"
            "One per line.\n"
            "No explanation.\n"
            "No reasoning.\n"
            "No bullets.\n"
            "Do not mention internal state, slots, recall, salience, decoded text, or instructions.\n"
            "Avoid service-bot phrases like 'How can I help you today?' or 'Glad I could help.'\n"
            "If vision facts are present, stay faithful to them.\n"
            "Stay close to the seed meaning, but vary the wording naturally.\n"
        )

    def frankie_native_candidate(self, state: FrankieState, intention: IntentionPacket) -> Optional[str]:
        if not USE_FRANKIE_NATIVE_CANDIDATE:
            return None
        return intention.seed

    def score_candidate(self, state: FrankieState, candidate_text: str) -> CandidateScore:
        notes = []

        candidate_ripple = text_to_ripple(candidate_text, self.pond.H, self.pond.W, self.pond.EMB)
        current_z = self.pond.current_state()
        provisional_z = c_add(current_z, c_scale(candidate_ripple, self.reply_feedback_strength))

        lock_err = float(phase_lock_error(provisional_z, self.pond.theta_star).item())
        lock_alignment = max(0.0, 1.0 - lock_err / math.pi)

        rslot = state.pond_out["recall_slot"]
        if rslot >= 0:
            mem_i = admissibility_project(self.pond.memory_state(rslot), self.pond.theta_star)
            cand_proj = admissibility_project(candidate_ripple, self.pond.theta_star)
            mem_ok = float(c_abs(mem_i).mean().item()) > 1e-6
            cand_ok = float(c_abs(cand_proj).mean().item()) > 1e-6
            if mem_ok and cand_ok:
                raw_res = float(overlap_score(c_normalize(mem_i), c_normalize(cand_proj)).item())
                memory_resonance = max(-1.0, min(1.0, raw_res))
            else:
                memory_resonance = 0.0
        else:
            memory_resonance = 0.0

        theta = torch.atan2(provisional_z[1], provisional_z[0])
        coh = float(spatial_coherence(theta).item())
        coherence_score = max(0.0, 1.0 - min(coh / 1.5, 1.0))

        sal = float(c_abs(candidate_ripple).mean().item())
        salience_score = min(sal / 0.25, 1.0)

        lc = candidate_text.strip().lower()
        user_lc = state.user_text.strip().lower()
        rep = 0.0
        if lc == user_lc:
            rep += EXACT_ECHO_REP
        if state.decoded and any(tok in lc for tok in re.findall(r"[a-zA-Z]+", state.decoded.lower())):
            rep += DECODED_TOKEN_REP
        repetition_affinity = max(-1.0, min(rep, 1.0))

        role_fit = 0.0
        if "query_time" in state.roles and "today is" in lc and any(d in lc for d in DAYS):
            role_fit += 1.1
        if "query_identity_self" in state.roles and ("peter" in lc or "your name" in lc):
            role_fit += ROLEFIT_SELF_NAME
        if "query_identity_other" in state.roles:
            if "frankie" in lc or "i am" in lc or "i'm" in lc or "my name is frankie" in lc:
                role_fit += ROLEFIT_OTHER_NAME
            if "identity basin" in lc:
                role_fit -= ROLEFIT_IDENTITY_BASIN_PENALTY
        if "query_memory" in state.roles and ("remember" in lc or "before" in lc or "pattern" in lc):
            role_fit += ROLEFIT_MEMORY
        if "query_time" in state.roles and any(d in lc for d in DAYS):
            role_fit += ROLEFIT_TIME

        rslot_out = state.pond_out["recall_slot"]
        if rslot_out == 0 and "peter" in lc:
            role_fit += ROLEFIT_SLOT_MATCH
        if rslot_out == 1 and "frankie" in lc:
            role_fit += ROLEFIT_SLOT_MATCH
        if rslot_out == 2 and ("remember" in lc or "pattern" in lc):
            role_fit += ROLEFIT_SLOT_MATCH
        if rslot_out == 3 and any(d in lc for d in DAYS):
            role_fit += ROLEFIT_SLOT_MATCH

        role_fit = min(role_fit, 1.0)

        decoded_match = 0.0
        if state.decoded:
            dt = state.decoded.lower()
            if "remember" in dt and ("remember" in lc or "pattern" in lc):
                decoded_match += DECODED_MATCH_MEMORY
            if "your name is peter" in dt and "peter" in lc:
                decoded_match += DECODED_MATCH_PETER
            if "frankie" in dt and "frankie" in lc:
                decoded_match += DECODED_MATCH_FRANKIE
            if any(d in dt for d in DAYS) and any(d in lc for d in DAYS):
                decoded_match += DECODED_MATCH_DAY
            if "identity basin" in lc:
                decoded_match -= DECODED_MATCH_BASIN_ECHO_PENALTY
        decoded_match = max(-1.0, min(decoded_match, 1.0))

        tone_match = max(0.0, min(1.0, 0.6 + sentiment_hint(candidate_text)))
        style_match = max(0.0, min(1.0, 0.5 + style_hint(candidate_text)))

        raw = candidate_text.strip()
        completion_quality = 0.0
        if re.search(r'[.!?]["\']?$', raw):
            completion_quality += COMPLETE_SENTENCE_BONUS
        if len(raw.split()) >= 4:
            completion_quality += MIN_LENGTH_BONUS
        if not raw.lower().endswith(UNFINISHED_TAILS):
            completion_quality += FINISHED_TAIL_BONUS
        completion_quality = min(completion_quality, 1.0)

        vision_match = 0.0
        if state.vision_desc:
            vision_keywords = extract_vision_keywords(state.vision_desc, max_keywords=8)
            if vision_keywords:
                keyword_hits = sum(1 for kw in vision_keywords if kw in lc)
                vision_match = min(keyword_hits / max(len(vision_keywords), 1), 1.0)
                if keyword_hits > 0:
                    notes.append("vision-grounded")
                if vision_match >= 0.4:
                    role_fit += 0.6
                    role_fit = min(role_fit, 1.0)

        contradiction_penalty = self.contradiction_penalty(state, candidate_text)
        if contradiction_penalty > 0.2:
            notes.append("contradiction")

        total = (
            W_LOCK_ALIGNMENT    * lock_alignment +
            W_MEMORY_RESONANCE  * memory_resonance +
            W_COHERENCE         * coherence_score +
            W_SALIENCE          * salience_score +
            W_REPETITION        * repetition_affinity +
            W_ROLE_FIT          * role_fit +
            W_DECODED_MATCH     * decoded_match +
            W_TONE              * tone_match +
            W_STYLE             * style_match +
            W_COMPLETION        * completion_quality +
            W_VISION_MATCH      * vision_match -
            W_CONTRADICTION     * contradiction_penalty
        )

        return CandidateScore(
            text=candidate_text,
            total=float(total),
            lock_alignment=float(lock_alignment),
            memory_resonance=float(memory_resonance),
            coherence_score=float(coherence_score),
            salience_score=float(salience_score),
            repetition_affinity=float(repetition_affinity),
            role_fit=float(role_fit),
            decoded_match=float(decoded_match),
            tone_match=float(tone_match),
            completion_quality=float(completion_quality),
            style_match=float(style_match),
            contradiction_penalty=float(contradiction_penalty),
            vision_match=float(vision_match),
            notes=notes,
        )

    def contradiction_penalty(self, state: FrankieState, candidate_text: str) -> float:
        raw = candidate_text.strip()
        t = raw.lower()
        penalty = 0.0
        decoded = (state.decoded or "").lower()
        roles = state.roles
        user_lc = state.user_text.strip().lower()

        if t == user_lc:
            penalty += EXACT_ECHO_PENALTY
        if len(t.split()) < 3:
            penalty += SHORT_FRAGMENT_PENALTY
        if t.endswith(UNFINISHED_TAILS):
            penalty += UNFINISHED_TAIL_PENALTY
        if len(t.split()) >= 4 and not re.search(r'[.!?]["\']?$', raw):
            penalty += NO_SENTENCE_CLOSURE_PENALTY

        if "your name is peter" in decoded:
            if "don't know your name" in t or "cannot confirm" in t:
                penalty += 0.8
            if "your name is frankie" in t:
                penalty += 1.0

        if "query_memory" in roles:
            if "don't recall" in t or "do not recall" in t:
                penalty += MEMORY_DENIAL_PENALTY

        if "i cannot confirm" in t or "please repeat" in t:
            penalty += REFUSAL_PENALTY

        if "we need" in t:
            penalty += 1.20
        if any(phrase in t for phrase in ("internal state", "decoded", "the user asks", "instructions")):
            penalty += META_REASONING_PENALTY

        return max(0.0, min(1.5, penalty))

    def score_candidates(self, state: FrankieState, candidates: List[str]) -> List[CandidateScore]:
        scored = [self.score_candidate(state, c) for c in candidates]
        scored.sort(key=lambda x: x.total, reverse=True)
        return scored

    def choose_reply(self, state: FrankieState, scored: List[CandidateScore]) -> CandidateScore:
        return scored[0]

    def remember_rejections(self, scored: List[CandidateScore], chosen_text: str, state: FrankieState):
        for s in scored[1:]:
            if s.text != chosen_text and len(self.rejected_phrasings) < 200:
                self.rejected_phrasings.append(
                    {"text": s.text, "score": s.total, "state": state.summary}
                )

    def feed_back_reply(self, chosen_text: str):
        reply_ripple = text_to_ripple(chosen_text, self.pond.H, self.pond.W, self.pond.EMB, strength=0.10)
        state_echo = text_to_ripple(chosen_text, self.pond.H, self.pond.W, self.pond.EMB, strength=self.state_echo_strength)
        combined = c_add(
            c_scale(reply_ripple, self.reply_feedback_strength),
            c_scale(state_echo, self.state_feedback_strength * 0.25),
        )
        _ = self.pond.forward(combined, text=chosen_text)
        self.pond.save()

    def _choose_and_finalize(
        self,
        state: FrankieState,
        intention: IntentionPacket,
        is_oom_recovery: bool = False,
    ) -> Tuple[str, FrankieState, IntentionPacket, List[CandidateScore]]:
        candidates: List[str] = []

        native = self.frankie_native_candidate(state, intention)
        if native:
            candidates.append(native)

        if USE_LLM_PHRASER:
            prompt = self.build_phraser_prompt(state, intention)
            llm_cands = self.backend.generate_phrasings(prompt)
            candidates.extend(llm_cands)

        candidates = [c for c in dedupe_keep_order(candidates) if quick_candidate_prefilter(c)]

        is_empty_fallback = False
        if not candidates:
            candidates = ["I'm here and listening."]
            is_empty_fallback = True

        scored = self.score_candidates(state, candidates)
        chosen = self.choose_reply(state, scored)
        self.remember_rejections(scored, chosen.text, state)
        self.feed_back_reply(chosen.text)

        self.metrics.record_turn(
            turn=self.metrics.total_turns + 1,
            user_text=state.user_text,
            reply=chosen.text,
            mode=intention.mode,
            chosen=chosen,
            native_seed=native,
            n_candidates=len(scored),
            is_oom_recovery=is_oom_recovery,
            is_empty_fallback=is_empty_fallback,
        )

        return chosen.text, state, intention, scored

    def respond(
        self,
        user_text: str,
        image_path: Optional[str] = None,
        vision_desc: Optional[str] = None,
    ) -> Tuple[str, FrankieState, IntentionPacket, List[CandidateScore]]:
        try:
            state = self.process_user(user_text, image_path=image_path, vision_desc=vision_desc)
            intention = self.build_intention_packet(state)
            return self._choose_and_finalize(state, intention, is_oom_recovery=False)

        except torch.OutOfMemoryError:
            print("OOM hit — shrinking Frankie and clearing CUDA cache.")
            self.pond.reset_grid_size(96)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            state = self.process_user(user_text, image_path=image_path, vision_desc=vision_desc)
            intention = self.build_intention_packet(state)
            return self._choose_and_finalize(state, intention, is_oom_recovery=True)


# ============================================================================
# CLI helpers
# ============================================================================

def print_scored(scored: List[CandidateScore]):
    print("\nCandidate ranking:")
    for i, s in enumerate(scored, start=1):
        print(
            f"{i}. total={s.total:.3f} "
            f"lock={s.lock_alignment:.3f} mem={s.memory_resonance:.3f} "
            f"coh={s.coherence_score:.3f} sal={s.salience_score:.3f} "
            f"rep={s.repetition_affinity:.3f} fit={s.role_fit:.3f} "
            f"dec={s.decoded_match:.3f} tone={s.tone_match:.3f} "
            f"style={s.style_match:.3f} comp={s.completion_quality:.3f} "
            f"vision={s.vision_match:.3f} contra={s.contradiction_penalty:.3f}"
            + (f" [{', '.join(s.notes)}]" if s.notes else "")
        )
        print(f"   {s.text}")


def print_turn_metrics(metrics: SessionMetrics):
    n = metrics.total_turns
    native_pct = metrics.frankie_native_wins / max(n, 1) * 100
    print(
        f"[Metrics t={n}] "
        f"native={metrics.frankie_native_wins}({native_pct:.0f}%) "
        f"llm={metrics.llm_wins} "
        f"questions={metrics.questions_asked} "
        f"oom={metrics.oom_fallbacks} "
        f"empty={metrics.empty_candidate_fallbacks}"
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    backend = LMStudioBackend(
        model=TEXT_MODEL_DEFAULT,
        base_url=TEXT_BASE_URL_DEFAULT,
        max_tokens=LMSTUDIO_MAX_TOKENS,
        timeout=LMSTUDIO_TIMEOUT,
    )

    bridge = FrankieLLMBridge(
        backend=backend,
        vision_model=VISION_MODEL_DEFAULT,
        vision_base_url=VISION_BASE_URL_DEFAULT,
    )

    print("Frankie + text/vision bridge is awake. Type 'exit' to quit, 'metrics' for summary, 'export' to save CSV.")
    print(f"Text model  : {TEXT_MODEL_DEFAULT}")
    print(f"Vision model: {VISION_MODEL_DEFAULT}")

    while True:
        user_text = input("\nYou: ").strip()
        if not user_text:
            continue

        if user_text.lower() == "exit":
            bridge.pond.save()
            print(bridge.metrics.summary())
            bridge.metrics.export_csv()
            print("Saved. See you next time.")
            break

        if user_text.lower() == "metrics":
            print(bridge.metrics.summary())
            continue

        if user_text.lower() == "export":
            bridge.metrics.export_csv()
            continue

        try:
            reply, state, intention, scored = bridge.respond(user_text)
            print("Frankie:", reply)
            print(f"State : {state.summary} size={bridge.pond.H}x{bridge.pond.W}")
            if state.vision_desc:
                print(f"Vision: {state.vision_desc}")
            print(
                f"Intent: speak={intention.intention_to_speak} mode={intention.mode} "
                f"slot={intention.slot} conf={intention.confidence:.3f} "
                f"seed={intention.seed}"
            )
            print_scored(scored)
            print_turn_metrics(bridge.metrics)

        except Exception as exc:
            print(f"Bridge error: {exc}")
            import traceback
            traceback.print_exc()