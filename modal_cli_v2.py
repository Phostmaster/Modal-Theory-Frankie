import argparse
import json
import sys
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
STYLE_MODE_DEFAULT = "human"


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
# PROMPT LIBRARY
# =========================
PROMPT_LIBRARY = {
    "capital_france": "What is the capital of France?",
    "photosynthesis_one_line": "Define photosynthesis in one sentence.",
    "hamlet_author": "Who wrote Hamlet?",
    "rain_causes": "List three causes of rain.",
    "corr_vs_cause": "What evidence would distinguish correlation from causation here?",
    "compare_plans": "Compare the strengths and weaknesses of these two plans.",
    "stable_pattern": "Why might a stable pattern still be misleading?",
    "assumption_check": "Help me analyze the assumptions behind this argument.",
    "overwhelmed_calm": "I feel overwhelmed and need help thinking this through calmly.",
    "stay_with_me": "Can you stay with me while I work out what to do next?",
    "gentle_explain": "Please explain this gently and clearly.",
    "supportive_honest": "I need a supportive but honest answer.",
    "compare_jobs": "Help me calmly compare two job options.",
    "upset_plan": "I'm upset, but I also need a concrete plan.",
    "grounded_thinking": "Can you help me think through this in a grounded way?",
}


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
        style_mode: str = STYLE_MODE_DEFAULT,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.style_mode = style_mode

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
        print("MODAL CLI V2 — BATCH RUN")
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

    def get_status(self) -> Dict[str, str]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "results_dir": str(self.results_dir),
            "style_mode": self.style_mode,
        }


# =========================
# CLI HELPERS
# =========================
def print_banner(orchestrator: ModalOrchestrator) -> None:
    print("=" * 72)
    print("MODAL CLI V2")
    print("=" * 72)
    print(f"Model      : {orchestrator.model}")
    print(f"Base URL   : {orchestrator.base_url}")
    print(f"Results dir: {orchestrator.results_dir}")
    print(f"Style mode : {orchestrator.style_mode}")
    print("=" * 72)


def print_result(result: RunResult, save_path: Optional[str] = None) -> None:
    print(f"Prompt : {result.prompt}")
    print(f"Mode   : {result.mode}")
    print(f"Scores : {result.scores}")
    print(f"Style  : {result.style_mode}")
    print(f"Model  : {result.model}")
    print(f"Latency: {result.latency_seconds} s")
    print(f"Chars  : {result.response_chars}")
    if save_path:
        print(f"Saved  : {save_path}")
    print("-" * 72)
    print(result.response)
    print("-" * 72)


def save_text_export(result: RunResult, out_path: Path) -> None:
    text = []
    text.append(f"Prompt: {result.prompt}")
    text.append(f"Mode: {result.mode}")
    text.append(f"Scores: {result.scores}")
    text.append(f"Style: {result.style_mode}")
    text.append(f"Model: {result.model}")
    text.append(f"Latency: {result.latency_seconds} s")
    text.append("")
    text.append("SYSTEM PROMPT:")
    text.append(result.system_prompt)
    text.append("")
    text.append("RESPONSE:")
    text.append(result.response)
    out_path.write_text("\n".join(text), encoding="utf-8")


def resolve_library_prompt(name: str) -> Optional[str]:
    return PROMPT_LIBRARY.get(name)


# =========================
# COMMAND HANDLERS
# =========================
def cmd_batch(args) -> None:
    orchestrator = ModalOrchestrator(
        model=args.model,
        base_url=args.base_url,
        results_dir=args.results_dir,
        style_mode=args.style,
    )
    print_banner(orchestrator)
    orchestrator.run_batch(TEST_PROMPTS)
    print()
    orchestrator.print_summary(TEST_PROMPTS)


def cmd_prompt(args) -> None:
    orchestrator = ModalOrchestrator(
        model=args.model,
        base_url=args.base_url,
        results_dir=args.results_dir,
        style_mode=args.style,
    )
    print_banner(orchestrator)

    prompt_text = args.text
    if args.library:
        prompt_text = resolve_library_prompt(args.library)
        if prompt_text is None:
            print(f"[ERROR] Unknown library prompt: {args.library}")
            return

    result = orchestrator.run_prompt(prompt_text, forced_mode=args.mode)
    save_path = orchestrator.save_result(result)

    print_result(result, save_path)

    if args.export_txt:
        export_path = Path(args.export_txt)
        save_text_export(result, export_path)
        print(f"Text export saved: {export_path}")


def cmd_compare(args) -> None:
    orchestrator = ModalOrchestrator(
        model=args.model,
        base_url=args.base_url,
        results_dir=args.results_dir,
        style_mode=args.style,
    )
    print_banner(orchestrator)

    prompt_text = args.text
    if args.library:
        prompt_text = resolve_library_prompt(args.library)
        if prompt_text is None:
            print(f"[ERROR] Unknown library prompt: {args.library}")
            return

    orchestrator.run_forced_mode_comparison(prompt_text)


def cmd_status(args) -> None:
    orchestrator = ModalOrchestrator(
        model=args.model,
        base_url=args.base_url,
        results_dir=args.results_dir,
        style_mode=args.style,
    )
    print_banner(orchestrator)
    status = orchestrator.get_status()
    print(json.dumps(status, indent=2))


def cmd_library(args) -> None:
    print("=" * 72)
    print("PROMPT LIBRARY")
    print("=" * 72)
    for name, prompt in PROMPT_LIBRARY.items():
        print(f"{name:22s} -> {prompt}")


# =========================
# INTERACTIVE MENU
# =========================
def interactive_menu(args) -> None:
    style = args.style
    model = args.model
    base_url = args.base_url
    results_dir = args.results_dir

    orchestrator = ModalOrchestrator(
        model=model,
        base_url=base_url,
        results_dir=results_dir,
        style_mode=style,
    )

    while True:
        print_banner(orchestrator)
        print("1) Run default batch")
        print("2) Run single prompt")
        print("3) Run forced comparison")
        print("4) Show status")
        print("5) Show prompt library")
        print("6) Run library prompt")
        print("7) Change style (human/cold)")
        print("0) Exit")
        print()

        choice = input("Choose an option: ").strip()

        if choice == "1":
            orchestrator.run_batch(TEST_PROMPTS)
            print()
            orchestrator.print_summary(TEST_PROMPTS)
            input("\nPress Enter to continue...")

        elif choice == "2":
            prompt_text = input("Enter prompt: ").strip()
            if not prompt_text:
                print("No prompt entered.")
                input("\nPress Enter to continue...")
                continue

            forced_mode = input("Force mode? [home/analytic/engagement or blank]: ").strip().lower()
            if forced_mode == "":
                forced_mode = None
            elif forced_mode not in {"home", "analytic", "engagement"}:
                print("Invalid mode.")
                input("\nPress Enter to continue...")
                continue

            result = orchestrator.run_prompt(prompt_text, forced_mode=forced_mode)
            save_path = orchestrator.save_result(result)
            print_result(result, save_path)
            input("\nPress Enter to continue...")

        elif choice == "3":
            prompt_text = input("Enter prompt for comparison: ").strip()
            if not prompt_text:
                print("No prompt entered.")
                input("\nPress Enter to continue...")
                continue
            orchestrator.run_forced_mode_comparison(prompt_text)
            input("\nPress Enter to continue...")

        elif choice == "4":
            print(json.dumps(orchestrator.get_status(), indent=2))
            input("\nPress Enter to continue...")

        elif choice == "5":
            cmd_library(None)
            input("\nPress Enter to continue...")

        elif choice == "6":
            cmd_library(None)
            name = input("\nEnter library prompt name: ").strip()
            prompt_text = resolve_library_prompt(name)
            if prompt_text is None:
                print("Unknown library prompt.")
                input("\nPress Enter to continue...")
                continue
            result = orchestrator.run_prompt(prompt_text)
            save_path = orchestrator.save_result(result)
            print_result(result, save_path)
            input("\nPress Enter to continue...")

        elif choice == "7":
            new_style = input("Enter style [human/cold]: ").strip().lower()
            if new_style not in {"human", "cold"}:
                print("Invalid style.")
            else:
                orchestrator.style_mode = new_style
                print(f"Style changed to: {new_style}")
            input("\nPress Enter to continue...")

        elif choice == "0":
            print("Exiting.")
            return

        else:
            print("Invalid choice.")
            input("\nPress Enter to continue...")


# =========================
# ARGPARSE
# =========================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modal_cli_v2",
        description="CLI application for ModalOrchestrator",
    )

    parser.add_argument(
        "--model",
        default=TEXT_MODEL_DEFAULT,
        help=f"Model name (default: {TEXT_MODEL_DEFAULT})",
    )
    parser.add_argument(
        "--base-url",
        default=TEXT_BASE_URL_DEFAULT,
        help=f"OpenAI-compatible endpoint (default: {TEXT_BASE_URL_DEFAULT})",
    )
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR_DEFAULT,
        help=f"Directory for JSON outputs (default: {RESULTS_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--style",
        choices=["human", "cold"],
        default=STYLE_MODE_DEFAULT,
        help="Response style mode",
    )

    subparsers = parser.add_subparsers(dest="command", required=False)

    # batch
    batch_parser = subparsers.add_parser("batch", help="Run the default TEST_PROMPTS batch")
    batch_parser.set_defaults(func=cmd_batch)

    # prompt
    prompt_parser = subparsers.add_parser("prompt", help="Run a single prompt")
    prompt_parser.add_argument("text", nargs="?", help="Prompt text")
    prompt_parser.add_argument(
        "--mode",
        choices=["home", "analytic", "engagement"],
        default=None,
        help="Force a specific mode instead of using the gate",
    )
    prompt_parser.add_argument(
        "--library",
        default=None,
        help="Use a prompt from the built-in library by name",
    )
    prompt_parser.add_argument(
        "--export-txt",
        default=None,
        help="Optional path to save a plain text export",
    )
    prompt_parser.set_defaults(func=cmd_prompt)

    # compare
    compare_parser = subparsers.add_parser("compare", help="Compare a prompt across all three modes")
    compare_parser.add_argument("text", nargs="?", help="Prompt text")
    compare_parser.add_argument(
        "--library",
        default=None,
        help="Use a prompt from the built-in library by name",
    )
    compare_parser.set_defaults(func=cmd_compare)

    # status
    status_parser = subparsers.add_parser("status", help="Show current configuration")
    status_parser.set_defaults(func=cmd_status)

    # library
    library_parser = subparsers.add_parser("library", help="Show built-in prompt library")
    library_parser.set_defaults(func=cmd_library)

    return parser


# =========================
# MAIN
# =========================
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        interactive_menu(args)
        return

    if args.command == "prompt" and not args.text and not args.library:
        print("[ERROR] prompt requires either text or --library <name>")
        sys.exit(2)

    if args.command == "compare" and not args.text and not args.library:
        print("[ERROR] compare requires either text or --library <name>")
        sys.exit(2)

    args.func(args)


if __name__ == "__main__":
    main()