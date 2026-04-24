import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from modal_orchestrator_tran_v5 import (
        ModalOrchestrator,
        TEXT_MODEL_DEFAULT,
        TEXT_BASE_URL_DEFAULT,
        RESULTS_DIR_DEFAULT,
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


@dataclass
class TurnWorkLog:
        turn_index: int
        prompt: str
        mode: str
        response: str
        latency_seconds: float
        prompt_chars: int
        response_chars: int
        estimated_prompt_tokens: int
        estimated_response_tokens: int
        estimated_total_tokens: int
        embedder_calls: int
        memory_retrieval_used: bool
        memory_items_requested: int
        verifier_used: bool
        candidate_generation_used: bool
        candidate_count: int
        route_name: str
        task_type: str
        cost_band: str
        estimated_work_score: float


@dataclass
class BenchmarkSummary:
        label: str
        timestamp: str
        total_turns: int
        total_prompt_chars: int
        total_response_chars: int
        total_estimated_tokens: int
        total_embedder_calls: int
        total_verifier_calls: int
        total_candidate_generations: int
        total_memory_retrievals: int
        average_latency_seconds: float
        total_latency_seconds: float
        total_estimated_work_score: float
        turns: List[Dict[str, Any]]


class EnergyMeter:
        def __init__(self, label: str, results_dir: str = "results") -> None:
                self.label = label
                self.results_dir = Path(results_dir)
                self.results_dir.mkdir(parents=True, exist_ok=True)
                self.turn_logs: List[TurnWorkLog] = []
                self._embedder_calls = 0
                self._verifier_calls = 0
                self._candidate_generations = 0
                self._memory_retrievals = 0

        @staticmethod
        def approx_tokens_from_chars(text: str) -> int:
                text = text or ""
                return max(1, int(round(len(text) / 4.0))) if text else 0

        @staticmethod
        def approx_tokens_from_char_count(char_count: int) -> int:
                return max(1, int(round(char_count / 4.0))) if char_count > 0 else 0

        def estimate_turn_work_score(
                self,
                prompt_tokens: int,
                response_tokens: int,
                embedder_calls: int,
                verifier_used: bool,
                candidate_generation_used: bool,
                memory_items_requested: int,
        ) -> float:
                token_work = prompt_tokens + response_tokens
                embedder_work = 40.0 * embedder_calls
                memory_work = 12.0 * memory_items_requested
                return round(token_work + embedder_work + memory_work, 3)

        def record_turn(self, turn_index: int, prompt: str, result: Any) -> None:
                scores = getattr(result, "scores", {}) or {}
                response = getattr(result, "response", "") or ""
                mode = getattr(result, "mode", "unknown")
                latency_seconds = float(getattr(result, "latency_seconds", 0.0) or 0.0)

                main_prompt_chars = int(getattr(result, "meter_main_prompt_chars", 0) or 0)
                main_response_chars = int(getattr(result, "meter_main_response_chars", 0) or 0)
                internal_prompt_chars = int(getattr(result, "meter_internal_prompt_chars", 0) or 0)
                internal_response_chars = int(getattr(result, "meter_internal_response_chars", 0) or 0)
                total_model_chars = int(getattr(result, "meter_total_model_chars", 0) or 0)

                if total_model_chars > 0:
                        prompt_tokens = self.approx_tokens_from_char_count(
                                main_prompt_chars + internal_prompt_chars
                        )
                        response_tokens = self.approx_tokens_from_char_count(
                                main_response_chars + internal_response_chars
                        )
                        total_tokens = self.approx_tokens_from_char_count(total_model_chars)
                else:
                        prompt_tokens = self.approx_tokens_from_chars(prompt)
                        response_tokens = self.approx_tokens_from_chars(response)
                        total_tokens = prompt_tokens + response_tokens

                verifier_used = "verifier_verdict" in scores
                candidate_generation_used = bool(scores.get("candidate_gate_pass", False))
                candidate_count = int(scores.get("candidate_count", 1) or 1)
                memory_items_requested = int(scores.get("memory_k", 0) or 0) if scores.get("retrieval") else 0
                per_turn_embedder_calls = 1 if scores.get("decision_used_embedder", 0.0) else 0

                self._embedder_calls += per_turn_embedder_calls
                if verifier_used:
                        self._verifier_calls += 1
                if candidate_generation_used:
                        self._candidate_generations += 1
                if scores.get("retrieval"):
                        self._memory_retrievals += 1

                work_score = self.estimate_turn_work_score(
                        prompt_tokens=prompt_tokens,
                        response_tokens=response_tokens,
                        embedder_calls=per_turn_embedder_calls,
                        verifier_used=verifier_used,
                        candidate_generation_used=candidate_generation_used,
                        memory_items_requested=memory_items_requested,
                )

                self.turn_logs.append(
                        TurnWorkLog(
                                turn_index=turn_index,
                                prompt=prompt,
                                mode=mode,
                                response=response,
                                latency_seconds=latency_seconds,
                                prompt_chars=len(prompt),
                                response_chars=len(response),
                                estimated_prompt_tokens=prompt_tokens,
                                estimated_response_tokens=response_tokens,
                                estimated_total_tokens=total_tokens,
                                embedder_calls=per_turn_embedder_calls,
                                memory_retrieval_used=bool(scores.get("retrieval", False)),
                                memory_items_requested=memory_items_requested,
                                verifier_used=verifier_used,
                                candidate_generation_used=candidate_generation_used,
                                candidate_count=candidate_count,
                                route_name=str(scores.get("route_name", "")),
                                task_type=str(scores.get("task_type", "")),
                                cost_band=str(scores.get("cost_band", "")),
                                estimated_work_score=work_score,
                        )
                )

        def summarize(self) -> BenchmarkSummary:
                total_turns = len(self.turn_logs)
                total_prompt_chars = sum(t.prompt_chars for t in self.turn_logs)
                total_response_chars = sum(t.response_chars for t in self.turn_logs)
                total_tokens = sum(t.estimated_total_tokens for t in self.turn_logs)
                total_latency = sum(t.latency_seconds for t in self.turn_logs)
                total_work = sum(t.estimated_work_score for t in self.turn_logs)
                avg_latency = total_latency / total_turns if total_turns else 0.0

                return BenchmarkSummary(
                        label=self.label,
                        timestamp=datetime.now().isoformat(),
                        total_turns=total_turns,
                        total_prompt_chars=total_prompt_chars,
                        total_response_chars=total_response_chars,
                        total_estimated_tokens=total_tokens,
                        total_embedder_calls=self._embedder_calls,
                        total_verifier_calls=self._verifier_calls,
                        total_candidate_generations=self._candidate_generations,
                        total_memory_retrievals=self._memory_retrievals,
                        average_latency_seconds=round(avg_latency, 6),
                        total_latency_seconds=round(total_latency, 6),
                        total_estimated_work_score=round(total_work, 3),
                        turns=[asdict(t) for t in self.turn_logs],
                )

        def save_summary(self) -> str:
                summary = self.summarize()
                filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{self.label}_benchmark.json"
                path = self.results_dir / filename
                with open(path, "w", encoding="utf-8") as handle:
                        json.dump(asdict(summary), handle, indent=2, ensure_ascii=False)
                return str(path)


class LocalMonolithicBaselineRunner:
        def __init__(self, orchestrator: ModalOrchestrator) -> None:
                self.orchestrator = orchestrator
                self.history: List[Dict[str, str]] = []

        def run_prompt(self, prompt: str):
                t0 = time.time()

                combined_prompt_parts: List[str] = []
                for item in self.history:
                        role = item["role"].upper()
                        combined_prompt_parts.append(f"{role}: {item['content']}")
                combined_prompt_parts.append(f"USER: {prompt}")
                combined_user_prompt = "\n\n".join(combined_prompt_parts)

                system_prompt = self.orchestrator.get_system_prompt("home")
                pv = type(
                        "BaselinePV",
                        (),
                        {
                                "temperature": 0.4,
                                "max_tokens": 180,
                                "home_floor": 1.0,
                        },
                )()

                if self.orchestrator.use_local_lora and self.orchestrator.local_model is not None:
                        response = self.orchestrator.call_local_model(
                                system_prompt=system_prompt,
                                user_prompt=combined_user_prompt,
                                pv=pv,
                        )
                else:
                        response = self.orchestrator.call_remote_model(
                                system_prompt=system_prompt,
                                user_prompt=combined_user_prompt,
                                pv=pv,
                        )

                self.history.append({"role": "user", "content": prompt})
                self.history.append({"role": "assistant", "content": response})

                class BaselineResult:
                        pass

                result = BaselineResult()
                result.mode = "monolithic"
                result.response = response
                result.latency_seconds = round(time.time() - t0, 6)
                result.scores = {
                        "task_type": "MONOLITHIC_BASELINE",
                        "cost_band": "full_context",
                        "route_name": "base_only",
                        "decision_used_embedder": 0.0,
                        "retrieval": False,
                        "memory_k": 0,
                        "candidate_gate_pass": False,
                        "candidate_count": 1.0,
                }

                result.meter_main_prompt_chars = len(system_prompt) + len(combined_user_prompt) + 16
                result.meter_main_response_chars = len(response)
                result.meter_internal_prompt_chars = 0
                result.meter_internal_response_chars = 0
                result.meter_total_model_chars = (
                        result.meter_main_prompt_chars
                        + result.meter_main_response_chars
                )
                return result


def run_benchmark_conversation(runner: Any, prompts: List[str], label: str) -> Dict[str, Any]:
        meter = EnergyMeter(label=label)
        outputs: List[Dict[str, Any]] = []

        for idx, prompt in enumerate(prompts, start=1):
                result = runner.run_prompt(prompt)
                meter.record_turn(idx, prompt, result)
                outputs.append(
                        {
                                "turn": idx,
                                "prompt": prompt,
                                "mode": result.mode,
                                "response": result.response,
                                "scores": result.scores,
                                "latency_seconds": result.latency_seconds,
                        }
                )

        save_path = meter.save_summary()
        return {
                "label": label,
                "summary": asdict(meter.summarize()),
                "outputs": outputs,
                "saved_to": save_path,
        }


def pct_reduction(old: float, new: float):
        if old == 0:
                return None
        return round(((old - new) / old) * 100.0, 3)


def compare_benchmarks(v5_run: Dict[str, Any], baseline_run: Dict[str, Any]) -> Dict[str, Any]:
        v5 = v5_run["summary"]
        base = baseline_run["summary"]

        return {
                "timestamp": datetime.now().isoformat(),
                "v5_label": v5_run["label"],
                "baseline_label": baseline_run["label"],
                "token_reduction_percent": pct_reduction(base["total_estimated_tokens"], v5["total_estimated_tokens"]),
                "latency_reduction_percent": pct_reduction(base["total_latency_seconds"], v5["total_latency_seconds"]),
                "work_score_reduction_percent": pct_reduction(base["total_estimated_work_score"], v5["total_estimated_work_score"]),
                "v5_summary": v5,
                "baseline_summary": base,
        }


def save_comparison_report(comparison: Dict[str, Any], results_dir: str = "results") -> str:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_benchmark_comparison.json"
        path = results_path / filename
        with open(path, "w", encoding="utf-8") as handle:
                json.dump(comparison, handle, indent=2, ensure_ascii=False)
        return str(path)


if __name__ == "__main__":
        frankie = ModalOrchestrator(
                model=TEXT_MODEL_DEFAULT,
                base_url=TEXT_BASE_URL_DEFAULT,
                results_dir=RESULTS_DIR_DEFAULT,
                style_mode="human",
                use_local_lora=True,
        )

        baseline_orchestrator = ModalOrchestrator(
                model=TEXT_MODEL_DEFAULT,
                base_url=TEXT_BASE_URL_DEFAULT,
                results_dir=RESULTS_DIR_DEFAULT,
                style_mode="human",
                use_local_lora=True,
        )
        baseline = LocalMonolithicBaselineRunner(baseline_orchestrator)

        print("=" * 72)
        print("RUNNING FRANKIE V5 BENCHMARK")
        print("=" * 72)
        v5_run = run_benchmark_conversation(frankie, CONVO_TEST, "frankie_v5_local")
        frankie.end_session()

        print("=" * 72)
        print("RUNNING LOCAL MONOLITHIC BASELINE BENCHMARK")
        print("=" * 72)
        baseline_run = run_benchmark_conversation(baseline, CONVO_TEST, "monolithic_local")
        baseline_orchestrator.end_session()

        comparison = compare_benchmarks(v5_run, baseline_run)
        comparison_path = save_comparison_report(comparison, results_dir=RESULTS_DIR_DEFAULT)

        print("\n" + "=" * 72)
        print("BENCHMARK SUMMARY")
        print("=" * 72)
        print(f"Frankie total estimated tokens:   {v5_run['summary']['total_estimated_tokens']}")
        print(f"Baseline total estimated tokens:  {baseline_run['summary']['total_estimated_tokens']}")
        print(f"Token reduction (%):              {comparison['token_reduction_percent']}")
        print(f"Frankie total latency (s):        {v5_run['summary']['total_latency_seconds']}")
        print(f"Baseline total latency (s):       {baseline_run['summary']['total_latency_seconds']}")
        print(f"Latency reduction (%):            {comparison['latency_reduction_percent']}")
        print(f"Frankie work score:               {v5_run['summary']['total_estimated_work_score']}")
        print(f"Baseline work score:              {baseline_run['summary']['total_estimated_work_score']}")
        print(f"Work reduction (%):               {comparison['work_score_reduction_percent']}")
        print(f"\nComparison saved: {comparison_path}")
