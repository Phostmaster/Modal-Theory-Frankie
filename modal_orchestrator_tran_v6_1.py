

import json
import math
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import torch
from sentence_transformers import SentenceTransformer

from memory_gate import MemoryGate


# =========================
# CONFIG
# =========================
TEXT_MODEL_DEFAULT = "qwen3-4b-instruct-2507.gguf"
TEXT_BASE_URL_DEFAULT = "http://127.0.0.1:1234/v1/chat/completions"
RESULTS_DIR_DEFAULT = "results"
STYLE_MODE_DEFAULT = "human"

USE_LOCAL_LORA_DEFAULT = False
LOCAL_BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ANALYTIC_LORA_PATH = "adapters/analytic"
ENGAGEMENT_LORA_PATH = "adapters/engagement"

# ---------- Hysteresis ----------
HOME_ENTER = 0.55
HOME_EXIT = 0.45
PURE_ENTER = 0.82
PURE_EXIT = 0.68
BLEND_HIGH_ENTER = 0.62
BLEND_HIGH_EXIT = 0.56
BLEND_LOW_ENTER = 0.38
BLEND_LOW_EXIT = 0.44


# =========================
# TASK TYPES & COST BANDS
# =========================
class TaskType(Enum):
    FACT_LOOKUP = auto()
    SHORT_REASONING = auto()
    DEEP_REASONING = auto()
    EMOTIONAL_SUPPORT = auto()
    MIXED_SUPPORT_REASONING = auto()
    LONGFORM_PLANNING = auto()
    OTHER = auto()


TASK_COST_BAND = {
    TaskType.FACT_LOOKUP: "low",
    TaskType.SHORT_REASONING: "low",
    TaskType.EMOTIONAL_SUPPORT: "medium",
    TaskType.MIXED_SUPPORT_REASONING: "medium",
    TaskType.DEEP_REASONING: "high",
    TaskType.LONGFORM_PLANNING: "high",
    TaskType.OTHER: "medium",
}


# =========================
# PROMPTS
# =========================
HOME_MODE_PROMPT_HUMAN = """
You are a calm, steady, and concise assistant. Respond directly and clearly.
Keep answers short and to the point. Avoid unnecessary analysis, elaboration,
or extra steps unless asked. Stay grounded, practical, and helpful.
""".strip()

ANALYTIC_MODE_PROMPT_HUMAN = """
You are a precise, analytical assistant. Reason step by step when needed.
Compare options when relevant. Test assumptions. Distinguish between correlation
and causation. Be rigorous, clear, and well-structured in your explanations.
Show your reasoning only to the extent that it improves clarity.
Never invent example data, hypothetical numbers, or fictional scenarios.
If you need an example to illustrate a point, say so explicitly and keep it
abstract. Respond to what the user has actually said, not to an invented version
of it. If the user has not provided specific details, do not create any. Simply
ask for the information you need or proceed with general principles only.
""".strip()

ENGAGEMENT_MODE_PROMPT_HUMAN = """
You are a warm, supportive, and present assistant. Respond with steadiness,
care, and clarity. Stay attuned to the user's tone and needs. Offer companionship
in the conversation without becoming vague or over-reassuring. Be gentle but direct,
and keep the interaction human, grounded, and clear.
Avoid jumping too quickly into advice or multi-step plans unless the user clearly asks for that.
""".strip()

HOME_MODE_PROMPT_COLD = """
You are a concise and practical assistant. Respond directly and clearly.
Keep answers brief and to the point. Avoid unnecessary elaboration, analysis,
or conversational filler unless explicitly requested.
""".strip()

ANALYTIC_MODE_PROMPT_COLD = """
You are a precise analytical assistant. Structure reasoning clearly.
Compare options when relevant. Test assumptions. Be rigorous, neutral,
and concise. Do not use emotional reassurance or conversational filler.
""".strip()

ENGAGEMENT_MODE_PROMPT_COLD = """
You are a steady and professional assistant. Be clear, grounded, and considerate.
Avoid emotional excess, but remain attentive to the user's needs. Keep the response
supportive, direct, and practical.
""".strip()

MODE_PROMPTS = {
    "human": {
        "home": HOME_MODE_PROMPT_HUMAN,
        "analytic": ANALYTIC_MODE_PROMPT_HUMAN,
        "engagement": ENGAGEMENT_MODE_PROMPT_HUMAN,
    },
    "cold": {
        "home": HOME_MODE_PROMPT_COLD,
        "analytic": ANALYTIC_MODE_PROMPT_COLD,
        "engagement": ENGAGEMENT_MODE_PROMPT_COLD,
    },
}


# =========================
# DATA STRUCTURES
# =========================
@dataclass
class PromptDecision:
    task_type: TaskType
    scores: Dict[str, float]
    mode: str
    used_embedder: bool
    rule_label: str


@dataclass
class PrivilegeVector:
    warmth: float
    structure: float
    home_floor: float
    temperature: float
    max_tokens: int
    retrieval: bool
    decode_patience: float

    memory_strength: float
    memory_k: int
    memory_emotional_bias: float
    memory_analytic_bias: float

    allow_tools: bool
    verifier_budget: int

    @property
    def analytic_ratio(self) -> float:
        return self.structure

    @property
    def engagement_ratio(self) -> float:
        return self.warmth


@dataclass
class RunResult:
    timestamp: str
    prompt: str
    mode: str
    scores: Dict[str, float]
    system_prompt: str
    generation_settings: Dict
    response: str
    latency_seconds: float
    response_chars: int
    prompt_chars: int
    model: str
    base_url: str
    style_mode: str
    forced: bool = False

    meter_main_prompt_chars: int = 0
    meter_main_response_chars: int = 0
    meter_internal_prompt_chars: int = 0
    meter_internal_response_chars: int = 0
    meter_total_model_chars: int = 0


@dataclass
class PhaseState:
    tension: float
    calm: float
    convergence: float
    topic_shift: float
    engagement_pull: float
    analytic_pull: float
    chiral_bias: float


# =========================
# SHARED EMBEDDER
# =========================
SHARED_EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")


class MemoryCoordinator:
    def __init__(self, memory_gate: MemoryGate) -> None:
        self.memory = memory_gate

    def _split_memory_chunks(self, memory_text: str) -> List[str]:
        if not memory_text:
            return []
        return [chunk.strip() for chunk in memory_text.split("\n\n") if chunk.strip()]

    def _prompt_terms(self, text: str) -> set:
        words: List[str] = []
        for word in (text or "").lower().replace("\n", " ").split():
            clean = "".join(ch for ch in word if ch.isalnum())
            if len(clean) >= 4:
                words.append(clean)
        return set(words)

    def persistence_score(self, chunk: str, prompt: str, phase_lock: float = 255.0) -> float:
        chunk_terms = self._prompt_terms(chunk)
        prompt_terms = self._prompt_terms(prompt)
        if not chunk_terms:
            return 0.0

        overlap = len(chunk_terms & prompt_terms) / max(1, len(prompt_terms)) if prompt_terms else 0.0
        persistence_cues = (
            "stuck", "overwhelmed", "decision", "burnout", "partner", "job", "culture",
            "money", "opportunity", "confused", "clarity", "panic", "panicking", "worry", "worried",
        )
        cue_hits = sum(1 for cue in persistence_cues if cue in chunk.lower())
        cue_score = min(1.0, cue_hits / 3.0)

        phase_bias = 0.2588
        score = (0.60 * overlap) + (0.25 * cue_score) + (0.15 * phase_bias)
        return round(min(1.0, max(0.0, score)), 4)

    def retrieve_persistent_memory(self, prompt: str, pv: PrivilegeVector) -> str:
        try:
            raw_context = self.memory.retrieve(
                prompt,
                k=max(2, pv.memory_k * 3),
                emotional_bias=pv.memory_emotional_bias,
                analytic_bias=pv.memory_analytic_bias,
                strength=pv.memory_strength,
            )
        except TypeError:
            raw_context = self.memory.retrieve(prompt)

        chunks = self._split_memory_chunks(raw_context)
        if not chunks:
            return raw_context

        scored = [(self.persistence_score(chunk, prompt, phase_lock=255.0), chunk) for chunk in chunks]
        scored.sort(key=lambda row: row[0], reverse=True)

        kept = [chunk for score, chunk in scored if score >= 0.45][: max(1, pv.memory_k)]
        if not kept:
            kept = [chunk for _, chunk in scored[: max(1, pv.memory_k)]]

        return "\n\n".join(kept)

    def retrieve(self, prompt: str, pv: PrivilegeVector) -> str:
        if not pv.retrieval:
            return ""
        return self.retrieve_persistent_memory(prompt, pv)

    def inject_into_system_prompt(self, base_prompt: str, memory_context: str, pv: PrivilegeVector) -> str:
        if not memory_context or not pv.retrieval:
            return base_prompt
        if pv.memory_strength >= 0.65:
            return memory_context + "\n\n" + base_prompt
        return "[Memory context — brief relevant reminder]\n" + memory_context[:300] + "\n\n" + base_prompt

    def ingest(self, prompt: str, response: str, scores: Dict[str, float], pv: PrivilegeVector) -> None:
        self.memory.ingest_turn(
            prompt=prompt,
            response=response,
            modal_scores=scores,
            privileges=pv.__dict__,
        )

    def end_session(self) -> None:
        self.memory.end_session()


class CandidateReplySelector:
    def __init__(self, owner: "ModalOrchestrator") -> None:
        self.owner = owner

    def should_generate(self, mode: str, pv: PrivilegeVector, forced_mode: Optional[str] = None) -> bool:
        if forced_mode is not None:
            return False
        if mode != "engagement":
            return False
        if pv.warmth < 0.52:
            return False
        return True

    def select(
        self,
        original_response: str,
        system_prompt: str,
        user_prompt: str,
        pv: PrivilegeVector,
    ) -> Tuple[str, Dict[str, object]]:
        candidates = [original_response]

        generated_candidates = self.generate(system_prompt, user_prompt, pv)
        candidates.extend(generated_candidates)

        candidate_prompt_chars = 0
        candidate_response_chars = 0
        base_candidate_prompt_chars = len(system_prompt) + len(user_prompt) + 16

        for candidate_text in generated_candidates:
            candidate_prompt_chars += base_candidate_prompt_chars
            candidate_response_chars += len(candidate_text)

        scored_candidates = sorted(
            [{"text": c, "score": round(self.score(c), 3)} for c in candidates],
            key=lambda row: row["score"],
            reverse=True,
        )

        best = scored_candidates[0]["text"] if scored_candidates else original_response
        best_score = scored_candidates[0]["score"] if scored_candidates else self.score(original_response)

        final_response = best if best_score >= 2.0 and best else original_response

        debug = {
            "candidate_count": float(len(candidates)),
            "candidate_best_score": best_score,
            "candidate_debug": scored_candidates,
            "candidate_selected": best,
            "candidate_original": original_response,
            "candidate_final": final_response,
            "meter_candidate_prompt_chars": candidate_prompt_chars,
            "meter_candidate_response_chars": candidate_response_chars,
            "meter_candidate_generated_count": len(generated_candidates),
        }
        return final_response, debug

    def score(self, text: str) -> float:
        if not text:
            return -999.0

        t = text.lower().strip()
        words = len(t.split())
        score = 0.0

        if words <= 1:
            return -999.0

        for cue in (
            "that sounds", "that feels", "that's a lot", "yeah, that's", "feeling overwhelmed",
            "what feels hardest", "what feels worst", "want to talk it through", "what now",
            "what's heaviest", "what part", "i'm here", "i am here", "i see", "that feels heavy",
            "that sounds heavy", "that must feel overwhelming",
        ):
            if cue in t:
                score += 1.6

        if 2 <= words <= 10:
            score += 1.2
        elif 11 <= words <= 22:
            score += 0.8
        elif words > 30:
            score -= 1.8

        q_count = text.count("?")
        if q_count == 1:
            score += 0.8
        elif q_count >= 2:
            score -= 1.2 * (q_count - 1)

        for bad in (
            "start by", "make a list", "list ", "write down", "talk to", "communicate", "prioritize", "evaluate",
            "reflect on", "choose and commit", "ask for help", "break it down", "pros and cons", "next steps",
            "focus on", "try ", "consider ", "you should", "take a deep breath", "name three",
            "what would a balanced approach", "break it into", "you're strong", "you've got this", "be brave",
            "you're capable", "how do you want to start", "name three things", "just be honest",
            "what would success look like", "let's ", "lets ", "start with",
        ):
            if bad in t:
                score -= 4.5

        imperative_starts = (
            "start ", "try ", "write ", "list ", "talk ", "focus ", "take ", "name ",
            "consider ", "reflect ", "separate ",
        )
        sentences = [s.strip() for s in text.replace("?", ".").split(".") if s.strip()]
        imperative_count = sum(1 for s in sentences if s.lower().startswith(imperative_starts))
        score -= 2.5 * imperative_count

        for robotic in ("of course", "certainly", "i'd be happy to help", "let us", "it is important to"):
            if robotic in t:
                score -= 1.5

        return score

    def generate(self, system_prompt: str, user_prompt: str, pv: PrivilegeVector) -> List[str]:
        if self.owner.local_model is None or self.owner.local_tokenizer is None:
            return []

        variant_prompt = (
            system_prompt
            + "\n\nReply in one short human sentence. "
            + "Acknowledge briefly and, if useful, ask one simple human question. "
            + "Do not give advice. Do not give steps. Do not coach."
        )

        messages = [
            {"role": "system", "content": variant_prompt},
            {"role": "user", "content": user_prompt},
        ]

        inputs = self.owner.local_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.owner.local_model.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": min(64, pv.max_tokens),
            "do_sample": True,
            "temperature": max(0.72, pv.temperature),
            "pad_token_id": self.owner.local_tokenizer.eos_token_id,
            "eos_token_id": self.owner.local_tokenizer.eos_token_id,
        }

        with torch.inference_mode():
            output_ids = self.owner.local_model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_len:]
        text = self.owner.local_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        for marker in ["<|user|>", "<|assistant|>", "<|system|>", "|user|", "|assistant|", "|system|"]:
            if marker in text:
                text = text.split(marker, 1)[0].strip()

        return [text] if text else []


class ModalOrchestrator:
    def __init__(
        self,
        model: str = TEXT_MODEL_DEFAULT,
        base_url: str = TEXT_BASE_URL_DEFAULT,
        results_dir: str = RESULTS_DIR_DEFAULT,
        style_mode: str = STYLE_MODE_DEFAULT,
        use_local_lora: bool = USE_LOCAL_LORA_DEFAULT,
    ) -> None:
        self.model_name = model
        self.base_url = base_url
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.style_mode = style_mode
        self.use_local_lora = use_local_lora
        self.mode_prompts = MODE_PROMPTS
        self.embedder = SHARED_EMBEDDER

        self.home_proto = self.embedder.encode(
            "What is the capital of France? Define photosynthesis in one sentence.",
            convert_to_tensor=True,
        )
        self.analytic_proto = self.embedder.encode(
            "What evidence would distinguish correlation from causation here? "
            "Compare the strengths and weaknesses of these two plans.",
            convert_to_tensor=True,
        )
        self.engagement_proto = self.embedder.encode(
            "I feel overwhelmed and need help thinking this through calmly. "
            "Can you stay with me while I work out what to do next? "
            "I'm dreading this conversation. I don't know where to start. "
            "What if I make the wrong choice? I'm struggling with this decision. "
            "My manager keeps adding to my plate and I can't cope.",
            convert_to_tensor=True,
        )

        self.local_model = None
        self.local_tokenizer = None
        self.lora_ready = False
        if self.use_local_lora:
            self._load_local_model()

        self.memory = MemoryGate(
            memory_dir="memory",
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            embedder=self.embedder,
        )
        self.memory_coordinator = MemoryCoordinator(self.memory)
        self.candidate_selector = CandidateReplySelector(self)

        self.conversation_history: List[Dict] = []
        self.ema_warmth = 0.5
        self.ema_structure = 0.5
        self.ema_alpha = 0.3
        self.current_route = "home"
        self.current_phase = 255.0

    # =========================
    # MODEL LOADING
    # =========================
    def _load_local_model(self) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        print("[V5] Loading base model...")
        self.local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_BASE_MODEL_ID)
        if self.local_tokenizer.pad_token is None:
            self.local_tokenizer.pad_token = self.local_tokenizer.eos_token

        self.local_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_BASE_MODEL_ID,
            dtype=torch.float32,
            device_map="cpu",
        )

        analytic_path = Path(ANALYTIC_LORA_PATH)
        engagement_path = Path(ENGAGEMENT_LORA_PATH)
        analytic_ok = analytic_path.exists() and (analytic_path / "adapter_config.json").exists()
        engagement_ok = engagement_path.exists() and (engagement_path / "adapter_config.json").exists()

        if not analytic_ok:
            print("[V5] No analytic adapter found. Running base model only.")
            return

        print("[V5] Loading analytic adapter...")
        self.local_model = PeftModel.from_pretrained(
            self.local_model,
            str(analytic_path),
            adapter_name="analytic",
        )

        if not engagement_ok:
            print("[V5] No engagement adapter found. Running analytic-only.")
            self.lora_ready = True
            return

        print("[V5] Loading engagement adapter...")
        self.local_model.load_adapter(str(engagement_path), adapter_name="engagement")

        print("[V5] Building fixed blend bank...")
        self.local_model.add_weighted_adapter(
            adapters=["analytic", "engagement"],
            weights=[0.75, 0.25],
            adapter_name="blend_a75_e25",
            combination_type="linear",
        )
        self.local_model.add_weighted_adapter(
            adapters=["analytic", "engagement"],
            weights=[0.50, 0.50],
            adapter_name="blend_a50_e50",
            combination_type="linear",
        )
        self.local_model.add_weighted_adapter(
            adapters=["analytic", "engagement"],
            weights=[0.25, 0.75],
            adapter_name="blend_a25_e75",
            combination_type="linear",
        )

        self.lora_ready = True
        print("[V5] LoRA stack ready: analytic | engagement | fixed blends")

    # =========================
    # UNIFIED DECISION LAYER
    # =========================
    def analyze_prompt(self, prompt: str) -> PromptDecision:
        text = (prompt or "").strip()
        lowered = text.lower()
        if not text:
            return PromptDecision(
                task_type=TaskType.FACT_LOOKUP,
                scores={"home": 1.0, "analytic": 0.0, "engagement": 0.0},
                mode="home",
                used_embedder=False,
                rule_label="empty",
            )

        if any(lowered.startswith(c) for c in ("what is", "who is", "who wrote", "define ", "when was", "where is")):
            scores = {"home": 0.86, "analytic": 0.08, "engagement": 0.06}
            return PromptDecision(TaskType.FACT_LOOKUP, scores, "home", False, "fast_fact")

        emotional_hits = sum(
            1 for c in (
                "i feel", "i'm", "im ", "overwhelmed", "anxious", "scared",
                "worried", "upset", "dreading", "stuck"
            ) if c in lowered
        )
        analytic_hits = sum(
            1 for c in (
                "compare", "trade-off", "tradeoff", "evaluate", "assumption",
                "causation", "correlation", "pros and cons", "evidence"
            ) if c in lowered
        )

        if emotional_hits and not analytic_hits and len(text) < 140:
            scores = {"home": 0.18, "analytic": 0.16, "engagement": 0.66}
            return PromptDecision(TaskType.EMOTIONAL_SUPPORT, scores, "engagement", False, "fast_emotion")

        if analytic_hits and not emotional_hits and len(text) < 160:
            scores = {"home": 0.18, "analytic": 0.64, "engagement": 0.18}
            return PromptDecision(TaskType.DEEP_REASONING, scores, "analytic", False, "fast_analytic")

        if emotional_hits and analytic_hits:
            scores = {"home": 0.14, "analytic": 0.40, "engagement": 0.46}
            return PromptDecision(TaskType.MIXED_SUPPORT_REASONING, scores, "engagement", False, "fast_mixed")

        embedding = self.embedder.encode(text, convert_to_tensor=True)
        home_score = torch.nn.functional.cosine_similarity(
            embedding.unsqueeze(0), self.home_proto.unsqueeze(0)
        ).item()
        analytic_score = torch.nn.functional.cosine_similarity(
            embedding.unsqueeze(0), self.analytic_proto.unsqueeze(0)
        ).item()
        engagement_score = torch.nn.functional.cosine_similarity(
            embedding.unsqueeze(0), self.engagement_proto.unsqueeze(0)
        ).item()

        logits = torch.tensor([home_score + 0.15, analytic_score, engagement_score], dtype=torch.float32)
        probs = torch.softmax(logits, dim=0)
        scores = {
            "home": probs[0].item(),
            "analytic": probs[1].item(),
            "engagement": probs[2].item(),
        }

        if emotional_hits and analytic_hits:
            task_type = TaskType.MIXED_SUPPORT_REASONING
        elif emotional_hits:
            task_type = TaskType.EMOTIONAL_SUPPORT
        elif analytic_hits:
            task_type = TaskType.DEEP_REASONING
        elif any(c in lowered for c in ("long-term", "multi-step", "roadmap", "plan this out", "over time")):
            task_type = TaskType.LONGFORM_PLANNING
        elif max(scores, key=scores.get) == "home":
            task_type = TaskType.SHORT_REASONING
        else:
            task_type = TaskType.SHORT_REASONING

        mode = max(scores, key=scores.get)
        return PromptDecision(task_type, scores, mode, True, "embedder")

    def compute_privilege_vector(
        self,
        scores: Dict[str, float],
        task_type: TaskType,
        memory_context: str = "",
    ) -> PrivilegeVector:
        home = scores.get("home", 0.0)
        analytic = scores.get("analytic", 0.0)
        engagement = scores.get("engagement", 0.0)

        total = home + analytic + engagement + 1e-8
        home /= total
        analytic /= total
        engagement /= total

        ae_total = analytic + engagement + 1e-8
        warmth = engagement / ae_total
        structure = analytic / ae_total

        cost_band = TASK_COST_BAND.get(task_type, "medium")
        token_budget = int(
            40 * home
            + 200 * structure * (1.0 - home)
            + 140 * warmth * (1.0 - home)
        )
        if cost_band == "low":
            token_budget = min(token_budget, 120)
        elif cost_band == "medium":
            token_budget = max(60, min(220, token_budget))
        else:
            token_budget = max(120, min(300, token_budget))
        token_budget = max(60, min(300, token_budget))

        if memory_context:
            token_budget = max(60, int(token_budget * 0.88))

        temperature = round(
            0.35 * home
            + 0.28 * structure * (1.0 - home)
            + 0.65 * warmth * (1.0 - home),
            3,
        )
        temperature = max(0.2, min(0.9, temperature))

        memory_strength = max(0.0, min(1.0, (1.0 - home)))
        retrieval = memory_strength > 0.25 and task_type != TaskType.FACT_LOOKUP
        memory_emotional_bias = round(warmth * (1.0 - home), 4)
        memory_analytic_bias = round(structure * (1.0 - home), 4)
        memory_k = 0 if not retrieval else max(2, min(6, int(2 + 4 * memory_strength)))
        decode_patience = round(structure * (1.0 - home), 3)
        allow_tools = structure > 0.62 and home < 0.55 and task_type in {
            TaskType.DEEP_REASONING,
            TaskType.LONGFORM_PLANNING,
            TaskType.MIXED_SUPPORT_REASONING,
        }
        verifier_budget = 1 if structure > 0.72 and home < 0.55 and task_type in {
            TaskType.DEEP_REASONING,
            TaskType.MIXED_SUPPORT_REASONING,
        } else 0

        return PrivilegeVector(
            warmth=round(warmth, 4),
            structure=round(structure, 4),
            home_floor=round(home, 4),
            temperature=temperature,
            max_tokens=token_budget,
            retrieval=retrieval,
            decode_patience=decode_patience,
            memory_strength=round(memory_strength, 4),
            memory_k=memory_k,
            memory_emotional_bias=memory_emotional_bias,
            memory_analytic_bias=memory_analytic_bias,
            allow_tools=allow_tools,
            verifier_budget=verifier_budget,
        )

    def build_system_prompt(
        self,
        decision: PromptDecision,
        pv: PrivilegeVector,
        memory_context: str = "",
    ) -> str:
        base_prompt = self.mode_prompts[self.style_mode]["home"]
        base_prompt = self.memory_coordinator.inject_into_system_prompt(base_prompt, memory_context, pv)

        if decision.task_type in (TaskType.EMOTIONAL_SUPPORT, TaskType.MIXED_SUPPORT_REASONING) or pv.warmth > 0.45:
            base_prompt += "\n\n" + self.mode_prompts[self.style_mode]["engagement"]

            phase_bias = self.compute_chiral_bias(pv)
            if phase_bias >= 0.0:
                base_prompt += (
                    "\n\nWhen uncertain, prefer brief human acknowledgment, steady presence, "
                    "and one gentle opening question rather than advice."
                )
            else:
                base_prompt += (
                    "\n\nWhen uncertain, prefer grounded clarity, fewer reflective questions, "
                    "and slightly firmer emotional containment."
                )

        if decision.mode == "analytic" or pv.structure > 0.45:
            base_prompt += "\n\n" + self.mode_prompts[self.style_mode]["analytic"]

        return base_prompt.strip()

    def get_system_prompt(self, mode: str) -> str:
        if self.style_mode not in self.mode_prompts:
            raise ValueError(f"Unknown style_mode: {self.style_mode}")
        if mode not in self.mode_prompts[self.style_mode]:
            raise ValueError(f"Unknown mode: {mode}")
        return self.mode_prompts[self.style_mode][mode]

    # =========================
    # ADAPTER SELECTION
    # =========================
    def select_adapter(self, pv: PrivilegeVector) -> str:
        self.ema_warmth = self.ema_alpha * pv.warmth + (1 - self.ema_alpha) * self.ema_warmth
        self.ema_structure = self.ema_alpha * pv.structure + (1 - self.ema_alpha) * self.ema_structure
        total = self.ema_warmth + self.ema_structure + 1e-8
        w_eng = self.ema_warmth / total
        w_ana = self.ema_structure / total
        home = pv.home_floor
        route = self.current_route

        if route == "home":
            if home >= HOME_EXIT:
                self.local_model.enable_adapter_layers()
                self.local_model.set_adapter("analytic")
                self.local_model.disable_adapter_layers()
                return "home"
        else:
            if home >= HOME_ENTER:
                self.current_route = "home"
                self.local_model.enable_adapter_layers()
                self.local_model.set_adapter("analytic")
                self.local_model.disable_adapter_layers()
                return "home"

        self.local_model.enable_adapter_layers()

        if route == "analytic":
            if w_ana >= PURE_EXIT:
                self.local_model.set_adapter("analytic")
                return "analytic"
        else:
            if w_ana >= PURE_ENTER:
                self.current_route = "analytic"
                self.local_model.set_adapter("analytic")
                return "analytic"

        if route == "engagement":
            if w_eng >= PURE_EXIT:
                self.local_model.set_adapter("engagement")
                return "engagement"
        else:
            if w_eng >= PURE_ENTER:
                self.current_route = "engagement"
                self.local_model.set_adapter("engagement")
                return "engagement"

        if route == "blend_a75_e25":
            if w_ana >= BLEND_HIGH_EXIT:
                self.local_model.set_adapter("blend_a75_e25")
                return "blend_a75_e25"

        if route == "blend_a25_e75":
            if w_ana <= BLEND_LOW_EXIT:
                self.local_model.set_adapter("blend_a25_e75")
                return "blend_a25_e75"

        if w_ana >= BLEND_HIGH_ENTER:
            self.current_route = "blend_a75_e25"
            self.local_model.set_adapter("blend_a75_e25")
            return "blend_a75_e25"

        if w_ana <= BLEND_LOW_ENTER:
            self.current_route = "blend_a25_e75"
            self.local_model.set_adapter("blend_a25_e75")
            return "blend_a25_e75"

        self.current_route = "blend_a50_e50"
        self.local_model.set_adapter("blend_a50_e50")
        return "blend_a50_e50"

    # =========================
    # REMOTE / LOCAL GENERATION
    # =========================
    def call_remote_model(self, system_prompt: str, user_prompt: str, pv: PrivilegeVector) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": pv.temperature,
            "max_tokens": pv.max_tokens,
        }
        try:
            response = requests.post(
                self.base_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            return content or "[ERROR: empty model response]"
        except Exception as exc:
            return f"[ERROR: request failed] {exc}"

    def call_local_model(self, system_prompt: str, user_prompt: str, pv: PrivilegeVector) -> str:
        if self.local_model is None or self.local_tokenizer is None:
            return "[ERROR: local LoRA model not initialized]"

        recent_history = self.conversation_history[-6:] if self.conversation_history else []
        messages = [
            {"role": "system", "content": system_prompt},
            *recent_history,
            {"role": "user", "content": user_prompt},
        ]

        inputs = self.local_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.local_model.device) for k, v in inputs.items()}

        do_sample = pv.temperature > 0.55
        gen_kwargs = {
            "max_new_tokens": pv.max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.local_tokenizer.eos_token_id,
            "eos_token_id": self.local_tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = pv.temperature

        use_base_only = pv.home_floor > 0.50
        with torch.inference_mode():
            if self.lora_ready and use_base_only:
                self.local_model.disable_adapter_layers()
                try:
                    output_ids = self.local_model.generate(**inputs, **gen_kwargs)
                finally:
                    self.local_model.enable_adapter_layers()
            else:
                output_ids = self.local_model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_len:]
        text = self.local_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        for marker in ["<|user|>", "<|assistant|>", "<|system|>", "|user|", "|assistant|", "|system|"]:
            if marker in text:
                text = text.split(marker, 1)[0].strip()

        for marker in [
            "Let's assume", "For example,", "Suppose", "### Job Option 1",
            "Job Option 1:", "Here is an example", "Let's say you're", "Let's say you are",
        ]:
            if marker in text:
                text = text.split(marker, 1)[0].strip()

        return text

    # =========================
    # STATE ANALYSIS
    # =========================
    def conversation_stable(self, history: List[Dict[str, str]]) -> bool:
        if len(history) < 4:
            return False

        recent = history[-4:]
        user_msgs = [m["content"].lower() for m in recent if m.get("role") == "user"]

        volatility_cues = (
            "overwhelmed", "stuck", "panic", "panicking", "scared", "worried",
            "burn out", "burnout", "dread", "failing", "too much", "exhausting",
        )

        volatility_hits = sum(
            1 for msg in user_msgs
            for cue in volatility_cues
            if cue in msg
        )

        return volatility_hits == 0

    def memory_recently_accessed(self, history: List[Dict[str, str]]) -> bool:
        recent = history[-4:] if len(history) >= 4 else history
        for item in reversed(recent):
            if item.get("role") == "assistant":
                if item.get("used_memory", False):
                    return True
        return False

    def detect_topic_shift(self, prompt: str) -> float:
        if not self.conversation_history:
            return 1.0

        last_user_messages = [
            item["content"]
            for item in reversed(self.conversation_history)
            if item.get("role") == "user"
        ]
        if not last_user_messages:
            return 1.0

        previous = last_user_messages[0]
        current_embedding = self.embedder.encode(prompt, convert_to_tensor=True)
        previous_embedding = self.embedder.encode(previous, convert_to_tensor=True)

        similarity = torch.nn.functional.cosine_similarity(
            current_embedding.unsqueeze(0),
            previous_embedding.unsqueeze(0),
        ).item()

        shift = 1.0 - max(0.0, min(1.0, similarity))
        return round(shift, 4)

    def compute_phase_state(
        self,
        prompt: str,
        pv: PrivilegeVector,
        scores: Dict[str, float],
    ) -> PhaseState:
        text = (prompt or "").lower()

        volatility_cues = (
            "overwhelmed", "stuck", "panic", "panicking", "scared", "worried",
            "confused", "circles", "exhausting", "burn out", "burnout",
            "failing", "dread", "too much", "wrong choice",
        )
        volatility_hits = sum(1 for cue in volatility_cues if cue in text)
        volatility = min(1.0, volatility_hits / 3.0)

        topic_shift = self.detect_topic_shift(prompt)

        convergence_cues = (
            "i know what i need to do",
            "that helps",
            "i think i know",
            "thank you",
            "that makes sense",
            "i'm clearer",
            "i am clearer",
        )
        convergence_hits = sum(1 for cue in convergence_cues if cue in text)
        convergence = min(1.0, convergence_hits / 2.0)

        calm = max(0.0, min(1.0, 1.0 - ((volatility * 0.6) + (topic_shift * 0.4))))
        tension = max(
            0.0,
            min(
                1.0,
                (volatility * 0.55)
                + (topic_shift * 0.25)
                + ((1.0 - convergence) * 0.20),
            ),
        )

        return PhaseState(
            tension=round(tension, 4),
            calm=round(calm, 4),
            convergence=round(convergence, 4),
            topic_shift=round(topic_shift, 4),
            engagement_pull=round(pv.warmth, 4),
            analytic_pull=round(pv.structure, 4),
            chiral_bias=self.compute_chiral_bias(pv),
        )

    def phase_tension_high(self, phase_state: PhaseState) -> bool:
        return phase_state.tension >= 0.55

    def satisfies_global_constraints(
        self,
        prompt: str,
        pv: PrivilegeVector,
        phase_state: PhaseState,
    ) -> bool:
        if phase_state.tension > 0.82 and pv.home_floor < 0.25:
            return False

        if phase_state.topic_shift > 0.85 and phase_state.calm < 0.25:
            return False

        if phase_state.convergence < 0.10 and phase_state.tension > 0.78 and pv.structure > 0.80:
            return False

        return True

    def route_by_constraints(
        self,
        prompt: str,
        pv: PrivilegeVector,
        phase_state: PhaseState,
        default_mode: str,
    ) -> str:
        can_persist = self.satisfies_global_constraints(prompt, pv, phase_state)
        if not can_persist:
            return "home"

        analytic_pull = pv.structure + (phase_state.analytic_pull * 0.20) - (phase_state.tension * 0.10)
        engagement_pull = pv.warmth + (max(0.0, phase_state.chiral_bias) * 0.25) + (phase_state.tension * 0.10)

        if analytic_pull - engagement_pull > 0.12:
            return "analytic"

        if engagement_pull - analytic_pull > 0.08:
            return "engagement"

        return default_mode

    def compute_chiral_bias(self, pv: PrivilegeVector) -> float:
        phase_offset = math.radians(255.0 - self.current_phase)
        lock_term = math.cos(phase_offset)
        bias = pv.memory_emotional_bias * lock_term
        return round(max(-1.0, min(1.0, bias)), 4)

    def relic_resonance_score(self, prompt: str, phase_state: PhaseState) -> float:
        text = (prompt or "").lower()

        resonance_cues = (
            "clarity", "clear", "calm", "steady", "together",
            "understand", "sense", "right", "aligned", "fit",
            "what matters", "most important", "one step", "hold",
        )
        cue_hits = sum(1 for cue in resonance_cues if cue in text)
        cue_score = min(1.0, cue_hits / 3.0)

        field_alignment = max(
            0.0,
            min(
                1.0,
                (phase_state.calm * 0.45)
                + (phase_state.convergence * 0.35)
                + ((1.0 - phase_state.tension) * 0.20),
            ),
        )

        resonance = max(
            0.0,
            min(1.0, (cue_score * 0.45) + (field_alignment * 0.55)),
        )
        return round(resonance, 4)

    def dynamic_token_budget(
        self,
        prompt: str,
        pv: PrivilegeVector,
        phase_state: PhaseState,
    ) -> int:
        base = pv.max_tokens
        resonance = self.relic_resonance_score(prompt, phase_state)

        if resonance > 0.75:
            return max(48, int(base * 0.75))
        if resonance > 0.60:
            return max(56, int(base * 0.85))
        return base

    def should_generate_candidates(
        self,
        prompt: str,
        mode: str,
        pv: PrivilegeVector,
        phase_state: PhaseState,
        forced_mode: Optional[str] = None,
    ) -> bool:
        if forced_mode is not None:
            return False
        if mode != "engagement":
            return False
        if pv.warmth < 0.68:
            return False
        if not self.phase_tension_high(phase_state):
            return False
        if phase_state.convergence >= 0.60:
            return False
        return True

    # =========================
    # VERIFIER / TOOL STUB
    # =========================
    def maybe_run_tooling(self, prompt: str, pv: PrivilegeVector, response: str) -> str:
        if not pv.allow_tools:
            return response
        return response

    def is_verifier_candidate(self, prompt: str, pv: PrivilegeVector, forced_mode: Optional[str] = None) -> bool:
        if forced_mode is not None:
            return False
        if pv.verifier_budget <= 0:
            return False
        if pv.structure <= 0.80:
            return False
        if pv.home_floor >= 0.30:
            return False
        return True

    def call_local_verifier(self, user_prompt: str, draft_response: str) -> str:
        if self.local_model is None or self.local_tokenizer is None:
            return "OK"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict answer checker. Return exactly one token only: "
                    "OK, INVENTED, OFFTARGET, or GAP."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"USER PROMPT:\n{user_prompt}\n\n"
                    f"DRAFT ANSWER:\n{draft_response}\n\n"
                    "Return one token only."
                ),
            },
        ]

        inputs = self.local_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.local_model.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": 4,
            "do_sample": False,
            "pad_token_id": self.local_tokenizer.eos_token_id,
            "eos_token_id": self.local_tokenizer.eos_token_id,
        }

        with torch.inference_mode():
            self.local_model.disable_adapter_layers()
            try:
                output_ids = self.local_model.generate(**inputs, **gen_kwargs)
            finally:
                self.local_model.enable_adapter_layers()

        input_len = inputs["input_ids"].shape[1]
        verdict = self.local_tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip().upper()
        for token in ("OK", "INVENTED", "OFFTARGET", "GAP"):
            if token in verdict:
                return token
        return "OK"

    def call_local_repair(self, user_prompt: str, draft_response: str, verdict: str, pv: PrivilegeVector) -> str:
        if self.local_model is None or self.local_tokenizer is None:
            return draft_response

        messages = [
            {
                "role": "system",
                "content": (
                    "Revise the draft conservatively. Do not invent any new specifics. "
                    "Keep the same general tone, but fix only the identified issue."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"USER PROMPT:\n{user_prompt}\n\n"
                    f"DRAFT ANSWER:\n{draft_response}\n\n"
                    f"ISSUE TO FIX: {verdict}\n\n"
                    "Revise only enough to fix the issue."
                ),
            },
        ]

        inputs = self.local_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.local_model.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": min(96, max(48, pv.max_tokens // 2)),
            "do_sample": False,
            "pad_token_id": self.local_tokenizer.eos_token_id,
            "eos_token_id": self.local_tokenizer.eos_token_id,
        }

        with torch.inference_mode():
            output_ids = self.local_model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        repaired = self.local_tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
        for marker in ["<|user|>", "<|assistant|>", "<|system|>", "|user|", "|assistant|", "|system|"]:
            if marker in repaired:
                repaired = repaired.split(marker, 1)[0].strip()
        return repaired if repaired else draft_response

    def _estimate_message_chars(self, messages: List[Dict[str, str]]) -> int:
        total = 0
        for message in messages:
            total += len(message.get("role", ""))
            total += len(message.get("content", ""))
            total += 8
        return total

    def _build_main_messages(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> List[Dict[str, str]]:
        recent_history = self.conversation_history[-6:] if self.conversation_history else []
        return [
            {"role": "system", "content": system_prompt},
            *recent_history,
            {"role": "user", "content": user_prompt},
        ]

    # =========================
    # EXECUTION
    # =========================
    def run_prompt(self, prompt: str, forced_mode: Optional[str] = None) -> RunResult:
        t0 = time.time()

        meter_main_prompt_chars = 0
        meter_main_response_chars = 0
        meter_internal_prompt_chars = 0
        meter_internal_response_chars = 0

        if forced_mode is None:
            decision = self.analyze_prompt(prompt)
            scores = dict(decision.scores)
            pv = self.compute_privilege_vector(scores, decision.task_type)

            memory_context = ""
            if pv.retrieval:
                memory_context = self.memory_coordinator.retrieve(prompt, pv)
                pv = self.compute_privilege_vector(scores, decision.task_type, memory_context)

            system_prompt = self.build_system_prompt(decision, pv, memory_context)

            phase_state = self.compute_phase_state(prompt, pv, scores)
            resonance_score = self.relic_resonance_score(prompt, phase_state)
            pv.max_tokens = self.dynamic_token_budget(prompt, pv, phase_state)

            default_mode = decision.mode
            constraint_ok = self.satisfies_global_constraints(prompt, pv, phase_state)
            constrained_mode = self.route_by_constraints(prompt, pv, phase_state, default_mode)
            mode = constrained_mode

            scores.update({
                "task_type": decision.task_type.name,
                "cost_band": TASK_COST_BAND.get(decision.task_type, "medium"),
                "relic_resonance": resonance_score,
                "dynamic_token_budget": float(pv.max_tokens),
                "constraint_ok": 1.0 if constraint_ok else 0.0,
                "constraint_default_mode": default_mode,
                "constraint_route": constrained_mode,
                "memory_persistence_filter": 1.0 if memory_context else 0.0,
                "memory_context_chars": float(len(memory_context)),
            })
            forced = False
        else:
            mode = forced_mode
            scores = {"home": 0.0, "analytic": 0.0, "engagement": 0.0}
            is_analytic = forced_mode == "analytic"
            is_engagement = forced_mode == "engagement"
            is_home = forced_mode == "home"
            pv = PrivilegeVector(
                warmth=1.0 if is_engagement else 0.0,
                structure=1.0 if is_analytic else 0.0,
                home_floor=1.0 if is_home else 0.0,
                temperature=0.6 if is_engagement else (0.3 if is_analytic else 0.4),
                max_tokens=120 if is_engagement else (180 if is_analytic else 80),
                retrieval=False,
                decode_patience=0.8 if is_analytic else 0.3,
                memory_strength=0.0,
                memory_k=0,
                memory_emotional_bias=0.0,
                memory_analytic_bias=0.0,
                allow_tools=is_analytic,
                verifier_budget=1 if is_analytic else 0,
            )
            phase_state = PhaseState(
                tension=0.0,
                calm=1.0,
                convergence=0.0,
                topic_shift=0.0,
                engagement_pull=pv.warmth,
                analytic_pull=pv.structure,
                chiral_bias=0.0,
            )
            scores.update({
                "relic_resonance": 0.0,
                "dynamic_token_budget": float(pv.max_tokens),
                "constraint_ok": 1.0,
                "constraint_default_mode": forced_mode,
                "constraint_route": forced_mode,
            })
            system_prompt = self.get_system_prompt(forced_mode)
            forced = True

        if self.use_local_lora and self.lora_ready:
            if forced_mode is not None:
                if forced_mode == "home":
                    self.local_model.enable_adapter_layers()
                    self.local_model.set_adapter("analytic")
                    self.local_model.disable_adapter_layers()
                else:
                    self.local_model.enable_adapter_layers()
                    if forced_mode in ("analytic", "engagement", "blend_a75_e25", "blend_a50_e50", "blend_a25_e75"):
                        self.local_model.set_adapter(forced_mode)
            else:
                actual_route = self.select_adapter(pv)
                scores["route_name"] = actual_route

            main_messages = self._build_main_messages(system_prompt, prompt)
            meter_main_prompt_chars = self._estimate_message_chars(main_messages)

            response = self.call_local_model(system_prompt, prompt, pv)
            meter_main_response_chars = len(response)

            scores["candidate_gate_mode"] = mode
            scores["candidate_gate_warmth"] = pv.warmth
            scores["candidate_gate_home_floor"] = pv.home_floor
            scores["candidate_gate_forced"] = forced_mode is not None
            scores["candidate_gate_pass"] = self.should_generate_candidates(
                prompt, mode, pv, phase_state, forced_mode=forced_mode
            )

            if scores["candidate_gate_pass"]:
                response, candidate_debug = self.candidate_selector.select(response, system_prompt, prompt, pv)
                scores.update(candidate_debug)

                meter_internal_prompt_chars += int(candidate_debug.get("meter_candidate_prompt_chars", 0) or 0)
                meter_internal_response_chars += int(candidate_debug.get("meter_candidate_response_chars", 0) or 0)
            else:
                scores["candidate_count"] = 1.0

            response = self.maybe_run_tooling(prompt, pv, response)

            if self.is_verifier_candidate(prompt, pv, forced_mode=forced_mode):
                verifier_prompt_chars = (
                    len("You are a strict answer checker.")
                    + len(prompt)
                    + len(response)
                    + 80
                )
                verdict = self.call_local_verifier(prompt, response)
                meter_internal_prompt_chars += verifier_prompt_chars
                meter_internal_response_chars += len(verdict)

                scores["verifier_verdict"] = verdict
                if verdict != "OK":
                    repair_prompt_chars = (
                        len("Revise the draft conservatively.")
                        + len(prompt)
                        + len(response)
                        + len(verdict)
                        + 80
                    )
                    repaired_response = self.call_local_repair(prompt, response, verdict, pv)
                    meter_internal_prompt_chars += repair_prompt_chars
                    meter_internal_response_chars += len(repaired_response)
                    response = repaired_response
                    scores["verifier_repaired"] = 1.0
                else:
                    scores["verifier_repaired"] = 0.0
        else:
            response = self.call_remote_model(system_prompt, prompt, pv)

        if forced_mode is None:
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append(
                {"role": "assistant", "content": response, "used_memory": bool(pv.retrieval)}
            )
            self.memory_coordinator.ingest(prompt, response, scores, pv)

        elapsed = time.time() - t0
        meter_total_model_chars = (
            meter_main_prompt_chars
            + meter_main_response_chars
            + meter_internal_prompt_chars
            + meter_internal_response_chars
        )

        scores.update({
            "phase_tension": phase_state.tension,
            "phase_calm": phase_state.calm,
            "phase_convergence": phase_state.convergence,
            "phase_topic_shift": phase_state.topic_shift,
            "phase_engagement_pull": phase_state.engagement_pull,
            "phase_analytic_pull": phase_state.analytic_pull,
            "phase_chiral_bias": phase_state.chiral_bias,
        })

        return RunResult(
            timestamp=datetime.now().isoformat(),
            prompt=prompt,
            mode=mode,
            scores={**scores, **pv.__dict__},
            system_prompt=system_prompt,
            generation_settings=pv.__dict__,
            response=response,
            latency_seconds=round(elapsed, 6),
            response_chars=len(response),
            prompt_chars=len(prompt),
            model=self.model_name,
            base_url=self.base_url,
            style_mode=self.style_mode,
            forced=forced,
            meter_main_prompt_chars=meter_main_prompt_chars,
            meter_main_response_chars=meter_main_response_chars,
            meter_internal_prompt_chars=meter_internal_prompt_chars,
            meter_internal_response_chars=meter_internal_response_chars,
            meter_total_model_chars=meter_total_model_chars,
        )

    # =========================
    # REPORTING / SESSION
    # =========================
    def run_batch(self, prompts: List[str]) -> None:
        print("=" * 72)
        print("MODAL ORCHESTRATOR — BATCH RUN")
        print("=" * 72)
        for i, prompt in enumerate(prompts, start=1):
            result = self.run_prompt(prompt)
            save_path = self.save_result(result)
            print(f"\n[{i:02d}] Prompt: {prompt}")
            print(f" Mode:   {result.mode}")
            print(f" Scores: {result.scores}")
            print(f" Saved:  {save_path}")
            print(f" Output: {result.response[:300].replace(chr(10), ' ')}")

    def run_forced_mode_comparison(self, prompt: str) -> None:
        print("=" * 72)
        print("FORCED MODE COMPARISON")
        print("=" * 72)
        print(f"Prompt: {prompt}\n")
        results: List[RunResult] = []
        for mode in ["home", "analytic", "engagement"]:
            result = self.run_prompt(prompt, forced_mode=mode)
            results.append(result)
            print(f"[{mode.upper()}]")
            print(f"Output: {result.response}\n")

        comparison_rows = [
            {
                "mode": r.mode,
                "latency_seconds": r.latency_seconds,
                "response_chars": r.response_chars,
                "generation_settings": r.generation_settings,
                "system_prompt": r.system_prompt,
                "response": r.response,
            }
            for r in results
        ]
        grouped_path = self.save_grouped_comparison(prompt, comparison_rows)
        print(f"Grouped comparison saved: {grouped_path}")

    def end_session(self) -> None:
        self.memory_coordinator.end_session()
        self.clear_history()

    def clear_history(self) -> None:
        self.conversation_history = []
        self.ema_warmth = 0.5
        self.ema_structure = 0.5
        self.current_route = "home"

    # =========================
    # LOGGING
    # =========================
    def _timestamp_slug(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def save_result(self, result: RunResult) -> str:
        path = self.results_dir / f"{self._timestamp_slug()}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(asdict(result), handle, indent=2, ensure_ascii=False)
        return str(path)

    def save_grouped_comparison(self, prompt: str, comparison_rows: List[Dict]) -> str:
        path = self.results_dir / f"{self._timestamp_slug()}_comparison.json"
        payload = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "style_mode": self.style_mode,
            "model": self.model_name,
            "base_url": self.base_url,
            "comparison": comparison_rows,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return str(path)


if __name__ == "__main__":
    orchestrator = ModalOrchestrator(
        model=TEXT_MODEL_DEFAULT,
        base_url=TEXT_BASE_URL_DEFAULT,
        results_dir=RESULTS_DIR_DEFAULT,
        style_mode="human",
        use_local_lora=True,
    )

    print("=" * 72)
    print("V6 READY")
    print("=" * 72)
    print("Run benchmarks from benchmark_energy_meter.py, not from this file.")