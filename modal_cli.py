import argparse
import json
from pathlib import Path

from modal_orchestrator import (
    ModalOrchestrator,
    TEXT_MODEL_DEFAULT,
    TEXT_BASE_URL_DEFAULT,
    RESULTS_DIR_DEFAULT,
    TEST_PROMPTS,
)


STYLE_MODE_DEFAULT = "human"


def print_banner(orchestrator: ModalOrchestrator) -> None:
    print("=" * 72)
    print("MODAL CLI")
    print("=" * 72)
    print(f"Model      : {orchestrator.model}")
    print(f"Base URL   : {orchestrator.base_url}")
    print(f"Results dir: {orchestrator.results_dir}")
    print(f"Style mode : {orchestrator.style_mode}")
    print("=" * 72)


def print_result(result, save_path: str | None = None) -> None:
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


def save_text_export(result, out_path: Path) -> None:
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

    result = orchestrator.run_prompt(args.prompt, forced_mode=args.mode)
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
    orchestrator.run_forced_mode_comparison(args.prompt)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modal_cli",
        description="CLI wrapper for ModalOrchestrator",
    )

    parser.add_argument(
        "--model",
        default=TEXT_MODEL_DEFAULT,
        help=f"Model name (default: {TEXT_MODEL_DEFAULT})",
    )
    parser.add_argument(
        "--base-url",
        default=TEXT_BASE_URL_DEFAULT,
        help=f"OpenAI-compatible chat completions endpoint (default: {TEXT_BASE_URL_DEFAULT})",
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

    subparsers = parser.add_subparsers(dest="command", required=True)

    # batch
    batch_parser = subparsers.add_parser(
        "batch",
        help="Run the default test prompt batch",
    )
    batch_parser.set_defaults(func=cmd_batch)

    # prompt
    prompt_parser = subparsers.add_parser(
        "prompt",
        help="Run a single prompt through the gate or a forced mode",
    )
    prompt_parser.add_argument("prompt", help="Prompt text to run")
    prompt_parser.add_argument(
        "--mode",
        choices=["home", "analytic", "engagement"],
        default=None,
        help="Force a specific mode instead of using the gate",
    )
    prompt_parser.add_argument(
        "--export-txt",
        default=None,
        help="Optional path to save a plain text export",
    )
    prompt_parser.set_defaults(func=cmd_prompt)

    # compare
    compare_parser = subparsers.add_parser(
        "compare",
        help="Run a single prompt through all three modes",
    )
    compare_parser.add_argument("prompt", help="Prompt text to compare")
    compare_parser.set_defaults(func=cmd_compare)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()