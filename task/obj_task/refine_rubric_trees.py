#!/usr/bin/env python3
import argparse
import asyncio
import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import litellm
from litellm import completion

from refine_rubric_prompt import RUBRIC_REFINE_PROMPT_TEMPLATE

litellm.drop_params = True


def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def normalize_refined_tree(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = parse_json_response(value)
        if isinstance(parsed, dict):
            return parsed
    return None


def build_litellm_kwargs(model_name: str, messages: list, max_tokens: int, temperature: float) -> Dict[str, Any]:
    call_kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "num_retries": 3,
        "reasoning_effort": os.environ.get("REFINE_REASONING_EFFORT", "high"),
    }

    api_base = os.environ.get("REFINE_API_BASE") or os.environ.get("FILTER_API_BASE")

    if model_name.startswith("azure/"):
        api_key = os.environ.get("REFINE_AZURE_API_KEY") or os.environ.get("FILTER_AZURE_API_KEY")
        api_base = os.environ.get("REFINE_AZURE_API_BASE") or os.environ.get("FILTER_AZURE_API_BASE")
        api_version = os.environ.get("REFINE_AZURE_API_VERSION") or os.environ.get("FILTER_AZURE_API_VERSION")
        if api_key:
            call_kwargs["api_key"] = api_key
        if api_base:
            call_kwargs["api_base"] = api_base
        if api_version:
            call_kwargs["api_version"] = api_version
    elif model_name.startswith("bedrock/"):
        access_key = os.environ.get("REFINE_AWS_ACCESS_KEY_ID") or os.environ.get("FILTER_AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("REFINE_AWS_SECRET_ACCESS_KEY") or os.environ.get("FILTER_AWS_SECRET_ACCESS_KEY")
        region = os.environ.get("REFINE_AWS_REGION_NAME") or os.environ.get("FILTER_AWS_REGION_NAME")
        if access_key:
            call_kwargs["aws_access_key_id"] = access_key
        if secret_key:
            call_kwargs["aws_secret_access_key"] = secret_key
        if region:
            call_kwargs["aws_region_name"] = region
    elif model_name.startswith("vllm/"):
        call_kwargs["api_key"] = os.environ.get("REFINE_OPENAI_API_KEY") or os.environ.get("FILTER_OPENAI_API_KEY", "EMPTY")
        if not api_base:
            raise ValueError("vLLM refine models require REFINE_API_BASE.")
        call_kwargs["api_base"] = api_base
    else:
        api_key = os.environ.get("REFINE_OPENAI_API_KEY") or os.environ.get("FILTER_OPENAI_API_KEY")
        if api_key:
            call_kwargs["api_key"] = api_key
        if api_base:
            call_kwargs["api_base"] = api_base

    return call_kwargs


async def call_refine_model(
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, Dict[str, Any]]:
    messages = [{"role": "user", "content": prompt}]
    call_kwargs = build_litellm_kwargs(model_name, messages, max_tokens, temperature)

    response = await asyncio.to_thread(lambda: completion(**call_kwargs))
    content = response.choices[0].message.content or ""

    usage = getattr(response, "usage", None)
    cost_info = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
        "cost": 0.0,
    }
    hidden = getattr(response, "_hidden_params", {}) or {}
    if "response_cost" in hidden:
        cost_info["cost"] = float(hidden["response_cost"])

    return content.strip(), cost_info


def build_prompt(data: Dict[str, Any], rubric_tree: Dict[str, Any]) -> str:
    return RUBRIC_REFINE_PROMPT_TEMPLATE.format(
        question=data.get("proposed_question") or data.get("metadata", {}).get("question", ""),
        constraints=json.dumps(data.get("constraints", []), ensure_ascii=False, indent=2),
        solution=json.dumps(data.get("solution", {}), ensure_ascii=False, indent=2),
        rubric_tree=json.dumps(rubric_tree, ensure_ascii=False, indent=2),
    )


def get_formatted_tree(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    analysis = data.get("rubric_tree_analysis_refined") or {}
    tree = analysis.get("formatted_tree")
    return tree if isinstance(tree, dict) else None


async def refine_file(
    path: Path,
    output_dir: Path,
    log_dir: Path,
    model_name: str,
    max_refine_iterations: int,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int,
) -> Dict[str, Any]:
    async with semaphore:
        print(f"[{index}/{total}] Refining {path.name}")

        data = json.loads(path.read_text(encoding="utf-8"))
        current_tree = get_formatted_tree(data)
        if current_tree is None:
            return {"file": path.name, "decision": "error", "error": "missing rubric_tree_analysis_refined.formatted_tree"}

        history = []
        total_cost = {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        final_decision = "unknown"
        final_reason = ""

        for iteration in range(max_refine_iterations + 1):
            prompt = build_prompt(data, current_tree)
            raw_response, cost_info = await call_refine_model(
                model_name=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            parsed = parse_json_response(raw_response) or {}
            decision = str(parsed.get("decision", "unknown")).lower()
            refined_tree = normalize_refined_tree(parsed.get("refined_rubric_tree"))

            for key in total_cost:
                total_cost[key] += cost_info.get(key, 0)

            history.append(
                {
                    "iteration": iteration,
                    "decision": decision,
                    "reason": parsed.get("reason", ""),
                    "raw_response": raw_response,
                    "parsed_response": parsed,
                    "cost_info": cost_info,
                }
            )

            final_decision = decision
            final_reason = parsed.get("reason", "")

            if decision == "fixable" and refined_tree is not None:
                current_tree = refined_tree
                if iteration < max_refine_iterations:
                    continue

            break

        output_data = copy.deepcopy(data)
        output_data["rubric_refine_result"] = {
            "model_name": model_name,
            "decision": final_decision,
            "reason": final_reason,
            "cost_info": total_cost,
            "history": history,
        }

        last_parsed = history[-1].get("parsed_response", {}) if history else {}
        last_refined_tree = normalize_refined_tree(last_parsed.get("refined_rubric_tree"))
        accepted = final_decision == "valid" or (
            final_decision == "fixable" and last_refined_tree is not None
        )
        if accepted:
            output_data["rubric_tree_analysis_refined"]["formatted_tree"] = current_tree
            output_path = output_dir / path.name
            output_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")

        log_path = log_dir / f"{path.stem}_refine.json"
        log_path.write_text(json.dumps(output_data["rubric_refine_result"], ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "file": path.name,
            "decision": final_decision,
            "accepted": accepted,
            "reason": final_reason,
            "cost_info": total_cost,
            "output_path": str(output_dir / path.name) if accepted else None,
            "log_path": str(log_path),
        }


async def run_refinement(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        path for path in input_dir.glob("*_formatted.json")
        if path.is_file() and "_filter_result" not in path.name
    )
    if args.max_tasks:
        files = files[: args.max_tasks]

    if not files:
        raise FileNotFoundError(f"No *_formatted.json files found in {input_dir}")

    semaphore = asyncio.Semaphore(args.workers)
    tasks = [
        refine_file(
            path=path,
            output_dir=output_dir,
            log_dir=log_dir,
            model_name=args.model,
            max_refine_iterations=args.max_refine_iterations,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            semaphore=semaphore,
            index=index,
            total=len(files),
        )
        for index, path in enumerate(files, 1)
    ]
    results = await asyncio.gather(*tasks)

    accepted_count = sum(1 for item in results if item.get("accepted"))
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model_name": args.model,
        "total_files": len(files),
        "accepted_count": accepted_count,
        "filtered_count": len(files) - accepted_count,
        "results": results,
    }
    summary_path = output_dir / "rubric_refine_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Accepted {accepted_count}/{len(files)} files")
    print(f"Refined files: {output_dir}")
    print(f"Summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine formatted objective rubric trees before verification.")
    parser.add_argument(
        "--input-dir",
        default="./outputs/objective_trajectories/formatted",
        help="Directory containing *_formatted.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for refined formatted files. Defaults to <input-dir>/refined.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for refine logs. Defaults to <input-dir>/refine_logs.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("REFINE_MODEL_NAME", os.environ.get("FILTER_MODEL_NAME", "openai/gpt-5.2")),
        help="LiteLLM model name, e.g. openai/gpt-5.2, azure/<deployment>, bedrock/<model-id>, or vllm/<model>.",
    )
    parser.add_argument("--workers", type=int, default=int(os.environ.get("REFINE_WORKERS", "20")))
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-refine-iterations", type=int, default=int(os.environ.get("REFINE_MAX_ITERATIONS", "3")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("REFINE_MAX_TOKENS", "10000")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("REFINE_TEMPERATURE", "0.6")))
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(Path(args.input_dir) / "refined")
    if args.log_dir is None:
        args.log_dir = str(Path(args.input_dir) / "refine_logs")

    return args


def main() -> None:
    asyncio.run(run_refinement(parse_args()))


if __name__ == "__main__":
    main()
