import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from memory_gate import MemoryGate
import warnings
import requests
import torch
from sentence_transformers import SentenceTransformer

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
HOME_EXIT  = 0.45

PURE_ENTER = 0.82
PURE_EXIT  = 0.68

BLEND_HIGH_ENTER = 0.62   # enter 75/25 side
BLEND_HIGH_EXIT  = 0.56   # leave 75/25 side

BLEND_LOW_ENTER  = 0.38   # enter 25/75 side
BLEND_LOW_EXIT   = 0.44   # leave 25/75 side


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

modePrompts = {
    "human": {
        "home":       HOME_MODE_PROMPT_HUMAN,
        "analytic":   ANALYTIC_MODE_PROMPT_HUMAN,
        "engagement": ENGAGEMENT_MODE_PROMPT_HUMAN,
    },
    "cold": {
        "home":       HOME_MODE_PROMPT_COLD,
        "analytic":   ANALYTIC_MODE_PROMPT_COLD,
        "engagement": ENGAGEMENT_MODE_PROMPT_COLD,
    },
}


# =========================
# TEST PROMPTS
# =========================
TEST_PROMPTS = [
    "What is the capital of France?",
    "Define photosynthesis in one sentence.",
    "Who wrote Hamlet?",
    "List three causes of rain.",
    "What evidence would distinguish correlation from causation here?",
    "Compare the strengths and weaknesses of these two plans.",
    "Why might a stable pattern still be misleading?",
    "Help me analyze the assumptions behind this argument.",
    "I feel overwhelmed and need help thinking this through calmly.",
    "Can you stay with me while I work out what to do next?",
    "Please explain this gently and clearly.",
    "I need a supportive but honest answer.",
    "Help me calmly compare two job options.",
    "I'm upset, but I also need a concrete plan.",
    "Explain this clearly without overcomplicating it.",
    "Can you help me think through this in a grounded way?",
]

MIXED_TEST_PROMPTS = [
    "I'm upset, but I also need a concrete plan.",
    "Help me calmly compare two job options.",
    "I feel overwhelmed and need a clear step-by-step plan.",
    "Can you stay with me while I work out a logical way forward?",
    "I'm worried about this decision, but I need an objective comparison.",
    "Please explain this gently, but also break it down logically.",
    "I need honest support and a practical action list.",
]


# =========================
# DATA STRUCTURES
# =========================
@dataclass
class PrivilegeVector:
    warmth:          float  # engagement LoRA strength  0.0 → 1.0
    structure:       float  # analytic LoRA strength    0.0 → 1.0
    home_floor:      float  # cheapness anchor          0.0 → 1.0
    temperature:     float  # sampling temperature      0.2 → 0.9
    max_tokens:      int    # token budget              60  → 300
    retrieval:       bool   # memory retrieval on/off
    decode_patience: float  # 0.0 = immediate, 1.0 = thoughtful

    @property
    def analytic_ratio(self) -> float:
        return self.structure

    @property
    def engagement_ratio(self) -> float:
        return self.warmth


@dataclass
class RunResult:
    timestamp:           str
    prompt:              str
    mode:                str
    scores:              Dict[str, float]
    system_prompt:       str
    generation_settings: Dict
    response:            str
    latency_seconds:     float
    response_chars:      int
    prompt_chars:        int
    model:               str
    base_url:            str
    style_mode:          str
    forced:              bool = False


# =========================
# SHARED EMBEDDER
# =========================
SHARED_EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# PRIVILEGE VECTOR COMPUTATION
# =========================
def compute_privilege_vector(scores: Dict[str, float]) -> PrivilegeVector:
    """
    Takes raw gate scores and returns a fully computed PrivilegeVector.
    All downstream behaviour derives from this single function.
    """
    home       = scores.get("home",       0.0)
    analytic   = scores.get("analytic",   0.0)
    engagement = scores.get("engagement", 0.0)

    # Normalise
    total      = home + analytic + engagement + 1e-8
    home       /= total
    analytic   /= total
    engagement /= total

    # Warmth and structure as ratio of non-home budget
    ae_total  = analytic + engagement + 1e-8
    warmth    = engagement / ae_total
    structure = analytic   / ae_total

    # Token budget
    token_budget = int(
        40  * home +
        200 * structure  * (1.0 - home) +
        140 * warmth     * (1.0 - home)
    )
    token_budget = max(60, min(300, token_budget))

    # Temperature
    temperature = round(
        0.35 * home +
        0.28 * structure  * (1.0 - home) +
        0.65 * warmth     * (1.0 - home),
        3,
    )
    temperature = max(0.2, min(0.9, temperature))

    # Retrieval — off for strong home signal
    retrieval = (1.0 - home) > 0.45

    # Decode patience — analytic = more patient
    decode_patience = round(structure * (1.0 - home), 3)

    return PrivilegeVector(
        warmth          = round(warmth,      4),
        structure       = round(structure,   4),
        home_floor      = round(home,        4),
        temperature     = temperature,
        max_tokens      = token_budget,
        retrieval       = retrieval,
        decode_patience = round(decode_patience, 4),
    )


# =========================
# ORCHESTRATOR
# =========================
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
        self.mode_prompts = modePrompts
        self.embedder = SHARED_EMBEDDER

        # Semantic prototypes for routing
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

        # Local LoRA state
        self.local_model = None
        self.local_tokenizer = None
        self.lora_ready = False

        if self.use_local_lora:
            self._load_local_model()

        # Memory — always initialised regardless of LoRA mode
        self.memory = MemoryGate(
            memory_dir="memory",
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            embedder=self.embedder,
        )

        # Conversation history buffer
        self.conversation_history: List[Dict] = []

        # EMA smoothing for blend stability
        self.ema_warmth:    float = 0.5
        self.ema_structure: float = 0.5
        self.ema_alpha:     float = 0.3   # 0.0 = full history, 1.0 = no smoothing

        # Hysteresis route state
        self.current_route: str = "home"

    # =========================
    # INIT HELPERS
    # =========================
    def _load_local_model(self) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        print("[P2] Loading base model...")
        self.local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_BASE_MODEL_ID)
        if self.local_tokenizer.pad_token is None:
            self.local_tokenizer.pad_token = self.local_tokenizer.eos_token

        self.local_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_BASE_MODEL_ID,
            dtype=torch.float32,
            device_map="cpu",
        )

        analytic_path   = Path(ANALYTIC_LORA_PATH)
        engagement_path = Path(ENGAGEMENT_LORA_PATH)

        analytic_ok = (
            analytic_path.exists()
            and (analytic_path / "adapter_config.json").exists()
        )
        engagement_ok = (
            engagement_path.exists()
            and (engagement_path / "adapter_config.json").exists()
        )

        if not analytic_ok:
            print("[P2] No analytic adapter found. Running base model only.")
            return

        print("[P2] Loading analytic adapter...")
        self.local_model = PeftModel.from_pretrained(
            self.local_model,
            str(analytic_path),
            adapter_name="analytic",
        )

        if not engagement_ok:
            print("[P2] No engagement adapter found. Running analytic-only.")
            self.lora_ready = True
            return

        print("[P2] Loading engagement adapter...")
        self.local_model.load_adapter(str(engagement_path), adapter_name="engagement")

        print("[P2] Building fixed blend bank...")

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
        print("[P2] LoRA stack ready: analytic | engagement | fixed blends")

    # =========================
    # ADAPTER SELECTION
    # =========================
    def select_adapter(self, pv: PrivilegeVector) -> str:
        """
        Hysteresis-based adapter selection.
        Prevents jitter between nearby modes by using separate enter/exit thresholds.
        """
        # Update EMA first
        self.ema_warmth = self.ema_alpha * pv.warmth + (1 - self.ema_alpha) * self.ema_warmth
        self.ema_structure = self.ema_alpha * pv.structure + (1 - self.ema_alpha) * self.ema_structure

        total = self.ema_warmth + self.ema_structure + 1e-8
        w_eng = self.ema_warmth / total
        w_ana = self.ema_structure / total
        home = pv.home_floor

        route = self.current_route

        # ---- HOME hysteresis ----
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

        # Re-enable adapter layers for non-home routes
        self.local_model.enable_adapter_layers()

        # ---- PURE ANALYTIC hysteresis ----
        if route == "analytic":
            if w_ana >= PURE_EXIT:
                self.local_model.set_adapter("analytic")
                return "analytic"
        else:
            if w_ana >= PURE_ENTER:
                self.current_route = "analytic"
                self.local_model.set_adapter("analytic")
                return "analytic"

        # ---- PURE ENGAGEMENT hysteresis ----
        if route == "engagement":
            if w_eng >= PURE_EXIT:
                self.local_model.set_adapter("engagement")
                return "engagement"
        else:
            if w_eng >= PURE_ENTER:
                self.current_route = "engagement"
                self.local_model.set_adapter("engagement")
                return "engagement"

        # ---- BLEND hysteresis ----
        if route == "blend_a75_e25":
            if w_ana >= BLEND_HIGH_EXIT:
                self.local_model.set_adapter("blend_a75_e25")
                return "blend_a75_e25"

        if route == "blend_a25_e75":
            if w_ana <= BLEND_LOW_EXIT:
                self.local_model.set_adapter("blend_a25_e75")
                return "blend_a25_e75"

        # Enter a stronger blend only when clearly over the line
        if w_ana >= BLEND_HIGH_ENTER:
            self.current_route = "blend_a75_e25"
            self.local_model.set_adapter("blend_a75_e25")
            return "blend_a75_e25"

        if w_ana <= BLEND_LOW_ENTER:
            self.current_route = "blend_a25_e75"
            self.local_model.set_adapter("blend_a25_e75")
            return "blend_a25_e75"

        # Otherwise settle in the middle blend
        self.current_route = "blend_a50_e50"
        self.local_model.set_adapter("blend_a50_e50")
        return "blend_a50_e50"
    # =========================
    # SCORING & ROUTING
    # =========================
    def score_modes(self, prompt: str) -> Dict[str, float]:
        text = prompt.strip()
        if not text:
            return {"home": 1.0, "analytic": 0.0, "engagement": 0.0}

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

        # Small home prior to keep factual queries cheap
        logits = torch.tensor(
            [home_score + 0.15, analytic_score, engagement_score],
            dtype=torch.float32,
        )
        probs = torch.softmax(logits, dim=0)

        return {
            "home":       probs[0].item(),
            "analytic":   probs[1].item(),
            "engagement": probs[2].item(),
        }

    def choose_mode(self, prompt: str) -> Dict[str, float]:
        return self.score_modes(prompt)

    # =========================
    # SYSTEM PROMPT & SETTINGS
    # =========================
    def build_system_prompt_and_settings(
        self, prompt: str, memory_context: str = ""
    ) -> tuple:
        scores = self.choose_mode(prompt)
        pv     = compute_privilege_vector(scores)

        base_prompt = self.mode_prompts[self.style_mode]["home"]

        # Memory context injected just above modal instructions
        if memory_context and pv.retrieval:
            base_prompt = memory_context + "\n\n" + base_prompt

        # Modal instructions driven by privilege vector
        if pv.structure > 0.45:
            base_prompt += "\n\n" + self.mode_prompts[self.style_mode]["analytic"]
        if pv.warmth > 0.45:
            base_prompt += "\n\n" + self.mode_prompts[self.style_mode]["engagement"]

        mode = max(scores, key=scores.get)

        return base_prompt.strip(), pv, scores, mode

    def get_system_prompt(self, mode: str) -> str:
        if self.style_mode not in self.mode_prompts:
            raise ValueError(f"Unknown style_mode: {self.style_mode}")
        if mode not in self.mode_prompts[self.style_mode]:
            raise ValueError(f"Unknown mode: {mode}")
        return self.mode_prompts[self.style_mode][mode]

    def is_verifier_candidate(
        self,
        prompt: str,
        pv: PrivilegeVector,
        forced_mode: Optional[str] = None,
    ) -> bool:
        if forced_mode is not None:
            return False

        if pv.structure < 0.72:
            return False

        text = prompt.lower()
        analytic_cues = (
            "compare", "evidence", "assumption", "assumptions",
            "causation", "correlation", "why", "evaluate",
            "trade-off", "tradeoff", "reasoning", "logic",
            "objective", "pros", "cons", "strengths", "weaknesses"
        )
        return any(cue in text for cue in analytic_cues)

    # =========================
    # MODEL CALL
    # =========================
    def call_local_model(
        self,
        system_prompt: str,
        user_prompt: str,
        pv: PrivilegeVector,
        scores: Dict[str, float],
    ) -> str:
        if self.local_model is None or self.local_tokenizer is None:
            return "[ERROR: local LoRA model not initialized]"

        # Keep last 6 turns (3 exchanges) to stay within token budget
        recent_history = self.conversation_history[-6:] if self.conversation_history else []
        messages = [
            {"role": "system", "content": system_prompt},
            *recent_history,
            {"role": "user",   "content": user_prompt},
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
            "do_sample":      do_sample,
            "pad_token_id":   self.local_tokenizer.eos_token_id,
            "eos_token_id":   self.local_tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = pv.temperature

        use_base_only = (pv.home_floor > 0.50)

        with torch.inference_mode():
            if self.lora_ready and use_base_only:
                self.local_model.disable_adapter_layers()
                try:
                    output_ids = self.local_model.generate(**inputs, **gen_kwargs)
                finally:
                    self.local_model.enable_adapter_layers()
            else:
                output_ids = self.local_model.generate(**inputs, **gen_kwargs)

        input_len  = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_len:]
        text = self.local_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Strip any fake turn continuations
        for marker in ["<|user|>", "<|assistant|>", "<|system|>",
                       "|user|", "|assistant|", "|system|"]:
            if marker in text:
                text = text.split(marker, 1)[0].strip()

        # Stop before invented examples
        for marker in ["Let's assume", "For example,", "Suppose", "### Job Option 1",
                       "Job Option 1:", "Here is an example",
                       "Let's say you're", "Let's say you are",
                       "commute stress, commute time", "commute time, commute"]:
            if marker in text:
                text = text.split(marker, 1)[0].strip()

        return text

    def call_local_verifier(
        self,
        user_prompt: str,
        draft_response: str,
    ) -> str:
        """
        Ultra-cheap verifier.
        Returns one token only:
        OK / INVENTED / OFFTARGET / GAP
        """
        if self.local_model is None or self.local_tokenizer is None:
            return "OK"

        verifier_messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict answer checker. "
                    "Return exactly one token only: "
                    "OK, INVENTED, OFFTARGET, or GAP.\n"
                    "OK = answer is grounded and answers the question.\n"
                    "INVENTED = answer adds specifics or examples not given by the user.\n"
                    "OFFTARGET = answer does not really address the user's question.\n"
                    "GAP = answer is relevant but reasoning is incomplete or sloppy."
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
            verifier_messages,
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
        new_tokens = output_ids[0][input_len:]
        verdict = self.local_tokenizer.decode(new_tokens, skip_special_tokens=True).strip().upper()

        for token in ("OK", "INVENTED", "OFFTARGET", "GAP"):
            if token in verdict:
                return token

        return "OK"

    def call_local_repair(
        self,
        user_prompt: str,
        draft_response: str,
        verdict: str,
        pv: PrivilegeVector,
    ) -> str:
        """
        Small corrective pass, only when verifier flags a problem.
        Keeps tone, tightens reasoning, removes invention.
        """
        if self.local_model is None or self.local_tokenizer is None:
            return draft_response

        repair_messages = [
            {
                "role": "system",
                "content": (
                    "Revise the draft answer conservatively. "
                    "Do not invent any new specifics. "
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
            repair_messages,
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
        new_tokens = output_ids[0][input_len:]
        repaired = self.local_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        for marker in ["<|user|>", "<|assistant|>", "<|system|>", "|user|", "|assistant|", "|system|"]:
            if marker in repaired:
                repaired = repaired.split(marker, 1)[0].strip()

        return repaired if repaired else draft_response


    # =========================
    # EXECUTION
    # =========================
    def run_prompt(self, prompt: str, forced_mode: Optional[str] = None) -> RunResult:
        t0 = time.time()

        # Memory retrieval — before routing so context is in system prompt
        # Forced mode comparisons bypass memory to keep results clean
        memory_context = "" if forced_mode is not None else self.memory.retrieve(prompt)

        if forced_mode is None:
            system_prompt, pv, scores, mode = self.build_system_prompt_and_settings(
                prompt, memory_context=memory_context
            )
            forced = False
        else:
            mode   = forced_mode
            scores = {"home": 0.0, "analytic": 0.0, "engagement": 0.0}
            is_analytic   = forced_mode == "analytic"
            is_engagement = forced_mode == "engagement"
            is_home       = forced_mode == "home"
            pv = PrivilegeVector(
                warmth          = 1.0 if is_engagement else 0.0,
                structure       = 1.0 if is_analytic   else 0.0,
                home_floor      = 1.0 if is_home       else 0.0,
                temperature     = 0.6 if is_engagement else (0.3 if is_analytic else 0.4),
                max_tokens      = 120 if is_engagement else (180 if is_analytic else 80),
                retrieval       = False,
                decode_patience = 0.8 if is_analytic   else 0.3,
            )
            system_prompt = self.get_system_prompt(forced_mode)
            forced = True

        # Adapter selection
        if self.use_local_lora and self.lora_ready:
            if forced_mode is not None:
                if forced_mode == "home":
                    self.local_model.enable_adapter_layers()
                    self.local_model.set_adapter("analytic")
                    self.local_model.disable_adapter_layers()
                else:
                    self.local_model.enable_adapter_layers()
                    valid_adapters = ("analytic", "engagement",
                                      "blend_a75_e25", "blend_a50_e50", "blend_a25_e75")
                    if forced_mode in valid_adapters:
                        self.local_model.set_adapter(forced_mode)
            else:
                # Normal routing — select_adapter drives everything
                actual_route = self.select_adapter(pv)
                scores["route_name"] = actual_route

            response = self.call_local_model(system_prompt, prompt, pv, scores)

            # Cheap verifier branch
            if self.is_verifier_candidate(prompt, pv, forced_mode=forced_mode):
                verdict = self.call_local_verifier(prompt, response)
                scores["verifier_verdict"] = verdict

                if verdict != "OK":
                    response = self.call_local_repair(prompt, response, verdict, pv)
                    scores["verifier_repaired"] = 1.0
                else:
                    scores["verifier_repaired"] = 0.0
        else:
            response = "[ERROR: local model not available]"

        # Conversation history — skip forced mode test runs
        if forced_mode is None:
            self.conversation_history.append({"role": "user",      "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": response})

        # Memory ingestion — skip forced mode test runs
        if forced_mode is None:
            self.memory.ingest_turn(
                prompt       = prompt,
                response     = response,
                modal_scores = scores,
                privileges   = pv.__dict__,
            )

        elapsed = time.time() - t0
        return RunResult(
            timestamp          = datetime.now().isoformat(),
            prompt             = prompt,
            mode               = mode,
            scores             = {**scores, **pv.__dict__},
            system_prompt      = system_prompt,
            generation_settings= pv.__dict__,
            response           = response,
            latency_seconds    = round(elapsed, 6),
            response_chars     = len(response),
            prompt_chars       = len(prompt),
            model              = self.model_name,
            base_url           = self.base_url,
            style_mode         = self.style_mode,
            forced             = forced,
        )

    # =========================
    # BATCH & REPORTING
    # =========================
    def run_batch(self, prompts: List[str]) -> None:
        print("=" * 72)
        print("MODAL ORCHESTRATOR — BATCH RUN")
        print("=" * 72)
        for i, prompt in enumerate(prompts, start=1):
            result    = self.run_prompt(prompt)
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
                "mode":                r.mode,
                "latency_seconds":     r.latency_seconds,
                "response_chars":      r.response_chars,
                "generation_settings": r.generation_settings,
                "system_prompt":       r.system_prompt,
                "response":            r.response,
            }
            for r in results
        ]

        grouped_path = self.save_grouped_comparison(prompt, comparison_rows)
        print(f"Grouped comparison saved: {grouped_path}")

    def print_summary(self, prompts: List[str]) -> None:
        counts = {"home": 0, "analytic": 0, "engagement": 0}
        for prompt in prompts:
            scores = self.choose_mode(prompt)
            mode   = max(scores, key=scores.get)
            counts[mode] = counts.get(mode, 0) + 1
        print("=" * 72)
        print("MODE SUMMARY")
        print("=" * 72)
        for mode, count in counts.items():
            print(f"{mode:12s}: {count}")

    def end_session(self) -> None:
        """Consolidate open memory thread and flush to disk."""
        self.memory.end_session()
        self.clear_history()

    def clear_history(self) -> None:
        """Clear conversation history and reset EMA state between sessions."""
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
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)
        return str(path)

    def save_grouped_comparison(self, prompt: str, comparison_rows: List[Dict]) -> str:
        path = self.results_dir / f"{self._timestamp_slug()}_comparison.json"
        payload = {
            "timestamp":  datetime.now().isoformat(),
            "prompt":     prompt,
            "style_mode": self.style_mode,
            "model":      self.model_name,
            "base_url":   self.base_url,
            "comparison": comparison_rows,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return str(path)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    orchestrator = ModalOrchestrator(
        model=TEXT_MODEL_DEFAULT,
        base_url=TEXT_BASE_URL_DEFAULT,
        results_dir=RESULTS_DIR_DEFAULT,
        style_mode="human",
        use_local_lora=True,
    )

    CONVO_TEST = [
        "I've been feeling really overwhelmed with work lately. Everything just feels too much.",
        "There's just too much to do and I don't know where to start. My manager keeps piling things on.",
        "I'm starting to dread going into the office. I feel like I'm failing at everything.",
        "Actually, I also have two job offers on the table right now and I need to decide soon.",
        "One pays more but the culture feels really off. The other feels much better but pays less.",
        "I'm worried that if I take the higher-paying one I'll burn out even faster.",
        "And my partner keeps asking me what I'm going to do. I don't want to let them down either.",
        "Remember when I said last week that I was already feeling stuck? This is making it worse.",
        "How do I even begin to think through this kind of decision clearly without panicking?",
        "What if I make the wrong choice and regret it for years?",
        "I keep going round in circles in my head and it's exhausting.",
        "Maybe I should just stay where I am and not rock the boat.",
        "But then I think about the money and the opportunities... and I get even more confused.",
        "Okay. I think I know what I need to do. Thank you for staying with me through all this.",
    ]

    print("=" * 72)
    print("CONVERSATIONAL TEST")
    print("=" * 72)
    for i, prompt in enumerate(CONVO_TEST, start=1):
        result = orchestrator.run_prompt(prompt)
        print(f"\n[{i:02d}] [{result.mode.upper()}] {prompt}")
        print(f"     {result.response[:300].replace(chr(10), ' ')}")

    print("\n" + "=" * 72 + "\n")
    orchestrator.run_forced_mode_comparison("Help me calmly compare two job options.")
    orchestrator.end_session()