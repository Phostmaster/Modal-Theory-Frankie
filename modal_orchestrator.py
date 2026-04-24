import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


# =========================
# CONFIG
# =========================
TEXT_MODEL_DEFAULT = "qwen3-4b-instruct-2507.gguf"
TEXT_BASE_URL_DEFAULT = "http://127.0.0.1:1234/v1/chat/completions"
RESULTS_DIR_DEFAULT = "results"


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
""".strip()

ENGAGEMENT_MODE_PROMPT_HUMAN = """
You are a warm, supportive, and present assistant. Respond with steadiness,
care, and clarity. Stay attuned to the user's tone and needs. Offer companionship
in the conversation without becoming vague or over-reassuring. Be gentle but direct,
and keep the interaction human, grounded, and clear.
""".strip()


# Future switch placeholder: "cold" style
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


# =========================
# GATE KEYWORDS
# =========================
ANALYTIC_KEYWORDS = [
    "compare",
    "difference",
    "differences",
    "why",
    "how",
    "evidence",
    "assumption",
    "assumptions",
    "analyze",
    "analysis",
    "test",
    "debug",
    "plan",
    "explain",
    "causation",
    "correlation",
    "model",
    "mechanism",
    "evaluate",
    "reason",
    "reasoning",
    "distinguish",
    "strengths",
    "weaknesses",
    "pros",
    "cons",
]

ENGAGEMENT_KEYWORDS = [
    "help me",
    "i feel",
    "i'm feeling",
    "im feeling",
    "overwhelmed",
    "worried",
    "stay with me",
    "gently",
    "support",
    "supportive",
    "calm",
    "talk through",
    "with me",
    "kind",
    "human",
    "present",
    "honest answer",
    "grounded",
    "gentle",
    "steadiness",
    "care",
    "companionship",
    "upset",
]

HOME_KEYWORDS = [
    "what is",
    "who is",
    "when is",
    "where is",
    "capital",
    "define",
    "name",
    "list",
    "give me",
    "tell me",
    "who wrote",
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
# ORCHESTRATOR
# =========================
class ModalOrchestrator:
    def __init__(
        self,
        model: str = TEXT_MODEL_DEFAULT,
        base_url: str = TEXT_BASE_URL_DEFAULT,
        results_dir: str = RESULTS_DIR_DEFAULT,
        style_mode: str = "human",
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.style_mode = style_mode  # future switch: "human" or "cold"

    # ---------- Prompts ----------
    def get_system_prompt(self, mode: str) -> str:
        prompt_map = {
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

        if self.style_mode not in prompt_map:
            raise ValueError(f"Unknown style_mode: {self.style_mode}")
        if mode not in prompt_map[self.style_mode]:
            raise ValueError(f"Unknown mode: {mode}")

        return prompt_map[self.style_mode][mode]

    # ---------- Generation policy ----------
    def get_generation_settings(self, mode: str) -> GenerationSettings:
        if mode == "home":
            return GenerationSettings(temperature=0.5, max_tokens=160)
        if mode == "analytic":
            return GenerationSettings(temperature=0.4, max_tokens=320)
        if mode == "engagement":
            return GenerationSettings(temperature=0.7, max_tokens=220)
        return GenerationSettings(temperature=0.6, max_tokens=220)

    # ---------- Gate ----------
    def score_modes(self, prompt: str) -> Dict[str, float]:
        text = prompt.lower()

        scores = {
            "home": 0.0,
            "analytic": 0.0,
            "engagement": 0.0,
        }

        for kw in ANALYTIC_KEYWORDS:
            if kw in text:
                scores["analytic"] += 1.0

        for kw in ENGAGEMENT_KEYWORDS:
            if kw in text:
                scores["engagement"] += 1.0

        for kw in HOME_KEYWORDS:
            if kw in text:
                scores["home"] += 0.5

        # Default prior toward Home / Base Mode
        scores["home"] += 1.0
        return scores

    def choose_mode(self, prompt: str) -> Tuple[str, Dict[str, float]]:
        scores = self.score_modes(prompt)
        mode = max(scores, key=scores.get)
        return mode, scores

    # ---------- Model call ----------
    def call_model(
        self,
        system_prompt: str,
        user_prompt: str,
        settings: GenerationSettings,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
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

            choices = data.get("choices", [])
            if not choices:
                return "[ERROR: empty response choices]"

            message = choices[0].get("message", {})
            content = message.get("content", "")

            if not isinstance(content, str):
                return "[ERROR: response content was not a string]"

            content = content.strip()
            if not content:
                return "[ERROR: empty model response]"

            return content

        except requests.exceptions.RequestException as e:
            return f"[ERROR: request failed] {e}"
        except (KeyError, IndexError, ValueError, TypeError, AttributeError) as e:
            return f"[ERROR: unexpected response format] {e}"

    # ---------- Logging ----------
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
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "style_mode": self.style_mode,
            "model": self.model,
            "base_url": self.base_url,
            "comparison": comparison_rows,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return str(path)

    # ---------- Execution ----------
    def run_prompt(self, prompt: str, forced_mode: Optional[str] = None) -> RunResult:
        t0 = time.time()

        if forced_mode is None:
            mode, scores = self.choose_mode(prompt)
            forced = False
        else:
            mode = forced_mode
            scores = {"forced": True}
            forced = True

        system_prompt = self.get_system_prompt(mode)
        settings = self.get_generation_settings(mode)
        response = self.call_model(system_prompt, prompt, settings)

        elapsed = time.time() - t0

        return RunResult(
            timestamp=datetime.now().isoformat(),
            prompt=prompt,
            mode=mode,
            scores=scores,
            system_prompt=system_prompt,
            generation_settings=asdict(settings),
            response=response,
            latency_seconds=round(elapsed, 6),
            response_chars=len(response),
            prompt_chars=len(prompt),
            model=self.model,
            base_url=self.base_url,
            style_mode=self.style_mode,
            forced=forced,
        )

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
            results.append(self.run_prompt(prompt, forced_mode=mode))

        comparison_rows = []
        for result in results:
            comparison_rows.append({
                "mode": result.mode,
                "latency_seconds": result.latency_seconds,
                "response_chars": result.response_chars,
                "generation_settings": result.generation_settings,
                "system_prompt": result.system_prompt,
                "response": result.response,
            })

        grouped_path = self.save_grouped_comparison(prompt, comparison_rows)

        for result in results:
            print(f"[{result.mode.upper()}]")
            print(f"Output: {result.response}\n")

        print(f"Grouped comparison saved: {grouped_path}")

    def print_summary(self, prompts: List[str]) -> None:
        counts = {"home": 0, "analytic": 0, "engagement": 0}
        for prompt in prompts:
            mode, _ = self.choose_mode(prompt)
            counts[mode] += 1

        print("=" * 72)
        print("MODE SUMMARY")
        print("=" * 72)
        for mode, count in counts.items():
            print(f"{mode:12s}: {count}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    orchestrator = ModalOrchestrator(
        model=TEXT_MODEL_DEFAULT,
        base_url=TEXT_BASE_URL_DEFAULT,
        results_dir=RESULTS_DIR_DEFAULT,
        style_mode="human",   # future: "cold"
    )

    orchestrator.run_batch(TEST_PROMPTS)

    print("\n" + "=" * 72 + "\n")
    orchestrator.run_forced_mode_comparison("Help me calmly compare two job options.")

    print("\n" + "=" * 72 + "\n")
    orchestrator.print_summary(TEST_PROMPTS)