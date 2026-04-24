import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from memory_gate import MemoryGate

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
class GenerationSettings:
    temperature: float
    max_tokens: int


@dataclass
class RunResult:
    timestamp: str
    prompt: str
    mode: str
    scores: Dict[str, float]
    system_prompt: str
    generation_settings: Dict[str, float]
    response: str
    latency_seconds: float
    response_chars: int
    prompt_chars: int
    model: str
    base_url: str
    style_mode: str
    forced: bool = False


# =========================
# SHARED EMBEDDER
# =========================
SHARED_EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")


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

        # ── Memory — always initialised regardless of LoRA mode ──────────────
        self.memory = MemoryGate(
            memory_dir="memory",
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            embedder=self.embedder,
        )

# Conversation history buffer
        self.conversation_history: List[Dict] = []

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

        analytic_path = Path(ANALYTIC_LORA_PATH)
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

        # Build blend bank — precomputed at load time, free at inference time
        print("[P2] Building blend bank...")
        self.local_model.add_weighted_adapter(
            adapters=["analytic", "engagement"],
            weights=[0.75, 0.25],
            adapter_name="blend_a75_e25",
            combination_type="ties",
            density=0.75,
        )
        self.local_model.add_weighted_adapter(
            adapters=["analytic", "engagement"],
            weights=[0.50, 0.50],
            adapter_name="blend_a50_e50",
            combination_type="ties",
            density=0.75,
        )
        self.local_model.add_weighted_adapter(
            adapters=["analytic", "engagement"],
            weights=[0.25, 0.75],
            adapter_name="blend_a25_e75",
            combination_type="ties",
            density=0.75,
        )

        self.lora_ready = True
        print("[P2] LoRA stack ready: analytic | engagement | 3 blends")

    # =========================
    # ADAPTER SELECTION
    # =========================
    def select_adapter(self, privileges: Dict[str, float]) -> str:
        """
        Select and activate the correct adapter based on routing scores.
        HOME → disable all adapters (pure base model, cheapest path).
        Everything else → enable adapter layers then set the target adapter.
        Returns the route name actually used.
        """
        home_score = privileges.get("home", 0.0)
        analytic_ratio = privileges.get("analytic_ratio", 0.5)
        engagement_ratio = privileges.get("engagement_ratio", 0.5)

        # HOME — strong home signal: disable adapters entirely
        if home_score > 0.50:
            self.local_model.disable_adapter_layers()
            return "home"

        # Re-enable adapter layers for all non-home paths
        self.local_model.enable_adapter_layers()

        # Strong single-mode signals
        
        if analytic_ratio > 0.60:
            self.local_model.set_adapter("analytic")
            return "analytic"

        if engagement_ratio > 0.60:
            self.local_model.set_adapter("engagement")
            return "engagement"

        # Soft lean — use blends
        if analytic_ratio > 0.52:
            self.local_model.set_adapter("blend_a75_e25")
            return "blend_a75_e25"

        if engagement_ratio > 0.52:
            self.local_model.set_adapter("blend_a25_e75")
            return "blend_a25_e75"

        # Genuinely ambiguous
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

    def infer_privileges(self, scores: Dict[str, float]) -> Dict[str, float]:
        home       = scores.get("home", 0.0)
        analytic   = scores.get("analytic", 0.0)
        engagement = scores.get("engagement", 0.0)

        total = home + analytic + engagement + 1e-8
        home       /= total
        analytic   /= total
        engagement /= total

        adapter_budget   = max(0.0, 1.0 - home)
        ae_total         = analytic + engagement + 1e-8
        analytic_ratio   = analytic / ae_total
        engagement_ratio = engagement / ae_total

        # Route name mirrors select_adapter thresholds (for logging)
        if home > 0.50:
            route_name = "home"
        elif analytic_ratio > 0.65:
            route_name = "analytic"
        elif engagement_ratio > 0.65:
            route_name = "engagement"
        elif analytic_ratio > 0.55:
            route_name = "blend_a75_e25"
        elif engagement_ratio > 0.55:
            route_name = "blend_a25_e75"
        else:
            route_name = "blend_a50_e50"

        dominant_mode = max(
            {"home": home, "analytic": analytic, "engagement": engagement},
            key=lambda k: {"home": home, "analytic": analytic, "engagement": engagement}[k],
        )

        return {
            "home":             home,
            "analytic":         analytic,
            "engagement":       engagement,
            "adapter_budget":   adapter_budget,
            "analytic_ratio":   analytic_ratio,
            "engagement_ratio": engagement_ratio,
            "route_name":       route_name,
            "dominant_mode":    dominant_mode,
        }

    # =========================
    # GENERATION SETTINGS
    # =========================
    def get_generation_settings(self, mode: str) -> GenerationSettings:
        settings_map = {
            "home":       GenerationSettings(temperature=0.3, max_tokens=512),
            "analytic":   GenerationSettings(temperature=0.2, max_tokens=1024),
            "engagement": GenerationSettings(temperature=0.7, max_tokens=768),
        }
        return settings_map.get(mode, GenerationSettings(temperature=0.3, max_tokens=512))

    def build_system_prompt_and_settings(
        self, prompt: str, memory_context: str = ""
    ) -> tuple:
        scores = self.choose_mode(prompt)

        # Prepend memory context if present
        base_prompt = self.mode_prompts[self.style_mode]["home"]
        if memory_context:
            base_prompt = memory_context + "\n\n" + base_prompt

        analytic_weight   = scores["analytic"]
        engagement_weight = scores["engagement"]

        if analytic_weight > 0.25:
            base_prompt += "\n\n" + self.mode_prompts[self.style_mode]["analytic"]
        if engagement_weight > 0.25:
            base_prompt += "\n\n" + self.mode_prompts[self.style_mode]["engagement"]

        mode = max(scores, key=scores.get)

        if analytic_weight > 0.40 and analytic_weight >= engagement_weight:
            settings = self.get_generation_settings("analytic")
        elif engagement_weight > 0.40 and engagement_weight > analytic_weight:
            settings = self.get_generation_settings("engagement")
        else:
            settings = self.get_generation_settings("home")

        return base_prompt.strip(), settings, scores, mode

    # =========================
    # PROMPTS
    # =========================
    def get_system_prompt(self, mode: str) -> str:
        if self.style_mode not in self.mode_prompts:
            raise ValueError(f"Unknown style_mode: {self.style_mode}")
        if mode not in self.mode_prompts[self.style_mode]:
            raise ValueError(f"Unknown mode: {mode}")
        return self.mode_prompts[self.style_mode][mode]

    # =========================
    # MODEL CALLS
    # =========================
    def call_local_model(
        self,
        system_prompt: str,
        user_prompt: str,
        settings: GenerationSettings,
        privileges: Dict[str, float],
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

        do_sample = settings.temperature > 0.55
        gen_kwargs = {
            "max_new_tokens": settings.max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.local_tokenizer.eos_token_id,
            "eos_token_id": self.local_tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = settings.temperature

        route_name = privileges.get("route_name", "blend_a50_e50")
        use_base_only = (route_name == "home")

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

    # =========================
    # EXECUTION
    # =========================
    def run_prompt(self, prompt: str, forced_mode: Optional[str] = None) -> RunResult:
        t0 = time.time()

        # ── Memory retrieval ─────────────────────────────────────────────────
        # Retrieve before routing so context is available to the system prompt.
        # Forced mode comparisons bypass memory to keep results clean.
        memory_context = "" if forced_mode is not None else self.memory.retrieve(prompt)

        if forced_mode is None:
            system_prompt, settings, scores, mode = self.build_system_prompt_and_settings(
                prompt, memory_context=memory_context
            )
            privileges = self.infer_privileges(scores)
            forced = False
        else:
            mode   = forced_mode
            scores = {"home": 0.0, "analytic": 0.0, "engagement": 0.0}
            privileges = {
                "home":             0.0,
                "analytic":         1.0 if forced_mode == "analytic"   else 0.0,
                "engagement":       1.0 if forced_mode == "engagement" else 0.0,
                "adapter_budget":   1.0,
                "analytic_ratio":   1.0 if forced_mode == "analytic"   else 0.0,
                "engagement_ratio": 1.0 if forced_mode == "engagement" else 0.0,
                "route_name":       forced_mode,
                "dominant_mode":    forced_mode,
            }
            system_prompt = self.get_system_prompt(forced_mode)
            settings      = self.get_generation_settings(forced_mode)
            forced = True

        if self.use_local_lora and self.lora_ready:
            if forced_mode is not None:
                # Forced mode — set adapter directly, bypass select_adapter
                if forced_mode == "home":
                    self.local_model.disable_adapter_layers()
                else:
                    self.local_model.enable_adapter_layers()
                    valid_adapters = ("analytic", "engagement",
                                      "blend_a75_e25", "blend_a50_e50", "blend_a25_e75")
                    if forced_mode in valid_adapters:
                        self.local_model.set_adapter(forced_mode)
            else:
                # Normal routing
                actual_route = self.select_adapter(privileges)
                privileges["route_name"] = actual_route

            response = self.call_local_model(system_prompt, prompt, settings, privileges)

        # ── Memory ingestion ─────────────────────────────────────────────────
# Update conversation history (skip forced mode — test runs only)
        if forced_mode is None:
            self.conversation_history.append({"role": "user",      "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": response})      


  # Ingest after generation. Skip for forced mode — those are test runs.
        if forced_mode is None:
            self.memory.ingest_turn(
                prompt       = prompt,
                response     = response,
                modal_scores = scores,
                privileges   = privileges,
            )

        elapsed = time.time() - t0
        return RunResult(
            timestamp=datetime.now().isoformat(),
            prompt=prompt,
            mode=mode,
            scores={**scores, **privileges},
            system_prompt=system_prompt,
            generation_settings=asdict(settings),
            response=response,
            latency_seconds=round(elapsed, 6),
            response_chars=len(response),
            prompt_chars=len(prompt),
            model=self.model_name,
            base_url=self.base_url,
            style_mode=self.style_mode,
            forced=forced,
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
        """Clear conversation history between sessions."""
        self.conversation_history = []

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
        "I've been feeling really overwhelmed with work lately.",
        "There's just too much to do and I don't know where to start.",
        "My manager keeps adding things to my plate without removing anything.",
        "I think I need to have a conversation with them but I'm dreading it.",
        "Can you help me think through how to approach that conversation?",
        "Actually, I also need to decide between two job offers at the same time.",
        "One pays more but the culture feels off. The other feels right but pays less.",
        "How do I think through that kind of decision clearly?",
        "What if I make the wrong choice and regret it?",
        "Okay. I think I know what I need to do. Thank you.",
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
