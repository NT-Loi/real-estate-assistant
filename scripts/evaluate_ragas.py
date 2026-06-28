"""Run Ragas metrics on saved Agentic RAG evaluation outputs.

Input is the JSONL produced by scripts/run_evaluation.py.

Recommended judge model:
  gemini-2.5-pro via Vertex AI ADC for higher-quality factual judging.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
import warnings
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass


def install_ragas_vertexai_compat() -> None:
    """Bridge old Ragas imports to the current LangChain Vertex package.

    Ragas 0.4.x still imports ChatVertexAI from:
      langchain_community.chat_models.vertexai

    Newer LangChain moved Vertex AI chat models to:
      langchain_google_vertexai
    """
    legacy_module = "langchain_community.chat_models.vertexai"
    if legacy_module in sys.modules:
        return

    try:
        from langchain_google_vertexai import ChatVertexAI
    except ImportError:
        return

    shim = types.ModuleType(legacy_module)
    shim.ChatVertexAI = ChatVertexAI
    sys.modules[legacy_module] = shim


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_ragas_samples(rows: list[dict[str, Any]], max_contexts: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in rows:
        actual = row.get("actual") or {}
        answer = str(actual.get("answer") or "").strip()
        question = str(row.get("question") or "").strip()
        contexts = [str(c) for c in actual.get("retrieved_contexts") or [] if str(c).strip()]
        reference = str(row.get("reference") or "").strip()
        if not question or not answer or not contexts:
            continue
        sample = {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts[:max_contexts],
        }
        if reference:
            sample["reference"] = reference
        samples.append(sample)
    return samples


def import_ragas_bits(metric_names: list[str]):
    install_ragas_vertexai_compat()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*langchain-community.*")
            from ragas import EvaluationDataset, evaluate
            from ragas.run_config import RunConfig
    except ImportError as exc:
        raise SystemExit(
            "Missing Ragas evaluation dependency. Install evaluation deps in the same Python environment first:\n"
            "  python -m pip install -r requirements-eval.txt\n\n"
            f"Original import error: {exc}"
        ) from exc

    metric_map = {}
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            from ragas.metrics import Faithfulness, FactualCorrectness
            from ragas.metrics import LLMContextRecall as ContextRecall
            from ragas.metrics import ResponseRelevancy

        metric_map.update({
            "faithfulness": Faithfulness,
            "context_recall": ContextRecall,
            "response_relevancy": ResponseRelevancy,
            "factual_correctness": FactualCorrectness,
        })
    except ImportError:
        # Older Ragas versions expose snake_case metric instances.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            from ragas.metrics import faithfulness, answer_relevancy, context_recall

        metric_map.update({
            "faithfulness": lambda: faithfulness,
            "context_recall": lambda: context_recall,
            "response_relevancy": lambda: answer_relevancy,
        })

    metrics = []
    unknown = []
    for name in metric_names:
        factory = metric_map.get(name)
        if not factory:
            unknown.append(name)
            continue
        metrics.append(factory())
    if unknown:
        raise SystemExit(f"Unknown/unsupported Ragas metric(s): {unknown}. Available: {sorted(metric_map)}")
    return EvaluationDataset, evaluate, RunConfig, metrics


def build_vertex_judge(model: str, project: str, location: str, temperature: float):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=Warning, message=".*ChatVertexAI.*")
            warnings.filterwarnings("ignore", category=Warning, message=".*LangchainLLMWrapper.*")
            warnings.filterwarnings("ignore", category=Warning, message=".*langchain-community.*")
            from langchain_google_vertexai import ChatVertexAI
            from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency for Gemini Vertex judge. Install:\n"
            "  python -m pip install -r requirements-eval.txt\n\n"
            f"Original import error: {exc}"
        ) from exc

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning, message=".*ChatVertexAI.*")
        warnings.filterwarnings("ignore", category=Warning, message=".*ChatGoogleGenerativeAI.*")
        warnings.filterwarnings("ignore", category=Warning, message=".*LangchainLLMWrapper.*")
        llm = ChatVertexAI(
            model_name=model,
            project=project or None,
            location=location,
            temperature=temperature,
            max_output_tokens=8192,
        )
        return LangchainLLMWrapper(llm)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved RAG outputs with Ragas.")
    parser.add_argument("--input", type=Path, required=True, help="Path to eval results.jsonl.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--max-contexts", type=int, default=8)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["faithfulness", "context_recall", "factual_correctness"],
        help="Supported: faithfulness context_recall response_relevancy factual_correctness",
    )
    parser.add_argument("--judge-provider", choices=["vertexai"], default="vertexai")
    parser.add_argument("--judge-model", default=os.getenv("RAGAS_JUDGE_MODEL", "gemini-2.5-pro"))
    parser.add_argument("--judge-project", default=os.getenv("PROJECT_ID", ""))
    parser.add_argument("--judge-location", default=os.getenv("VERTEX_LOCATION", "global"))
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=600, help="Ragas per-job timeout in seconds.")
    parser.add_argument("--max-workers", type=int, default=4, help="Ragas concurrent judge jobs.")
    parser.add_argument("--max-retries", type=int, default=3, help="Ragas retry count per failed judge job.")
    parser.add_argument("--batch-size", type=int, default=4, help="Ragas evaluate batch size.")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if args.category:
        allowed = set(args.category)
        rows = [row for row in rows if row.get("category") in allowed]
    if args.limit:
        rows = rows[: args.limit]

    samples = build_ragas_samples(rows, max_contexts=args.max_contexts)
    if not samples:
        raise SystemExit("No Ragas samples available. Need question, answer, and retrieved_contexts.")

    EvaluationDataset, evaluate, RunConfig, metrics = import_ragas_bits(args.metrics)
    dataset = EvaluationDataset.from_list(samples)

    if args.judge_provider == "vertexai":
        evaluator_llm = build_vertex_judge(
            model=args.judge_model,
            project=args.judge_project,
            location=args.judge_location,
            temperature=args.judge_temperature,
        )
    else:
        raise SystemExit(f"Unsupported judge provider: {args.judge_provider}")

    run_config = RunConfig(
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_workers=args.max_workers,
    )
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        run_config=run_config,
        batch_size=args.batch_size,
        raise_exceptions=False,
    )

    output = args.output or (args.input.parent / "ragas_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        result_df = result.to_pandas()
        records = result_df.to_dict(orient="records")
        aggregate = {
            key: float(value)
            for key, value in result.items()
            if isinstance(value, (int, float))
        } if hasattr(result, "items") else {}
    except Exception:
        records = []
        aggregate = {}

    payload = {
        "input": str(args.input),
        "sample_count": len(samples),
        "judge_provider": args.judge_provider,
        "judge_model": args.judge_model,
        "judge_location": args.judge_location,
        "metrics": args.metrics,
        "timeout": args.timeout,
        "max_workers": args.max_workers,
        "max_retries": args.max_retries,
        "batch_size": args.batch_size,
        "aggregate": aggregate,
        "records": records,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(output), "sample_count": len(samples), "aggregate": aggregate}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
