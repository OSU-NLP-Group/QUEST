from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional

from tqdm import tqdm

from mind2web2.eval_runner import evaluate_task, merge_all_results
from mind2web2.llm_client.base_client import LLMClient
# from mind2web2.utils.llm_trace import set_default_trace_path
from mind2web2.utils.path_config import PathConfig


# --------------------------------------------------------------------------- #
# CLI                                                                        #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Mind2Web2 task evaluation.")

    # Task specification
    p.add_argument("--task_id", help="Task folder name (if not provided, evaluates all tasks)")
    p.add_argument("--agent_name", required=True, help="Agent name for evaluation")

    # Required path
    p.add_argument("--answer_folder", type=Path,
                   help="Directory containing answer files (required)")

    # Optional path overrides
    p.add_argument("--eval_scripts_root", type=Path,
                   help="Override evaluation scripts directory")
    p.add_argument("--eval_results_root", type=Path,
                   help="Override output directory for results/logs")
    p.add_argument("--cache_root", type=Path,
                   help="Override cache directory")
    p.add_argument("--eval_version", default="2026_01_12_1000",
                   help="Version of evaluation scripts to use (default: 2025_10_23)")

    # LLM configuration
    p.add_argument("--llm_provider", choices=["openai", "azure_openai", "local_openai"],
                   default="openai", help="LLM provider to use")
    p.add_argument("--judge_model", default="gpt-5-mini", help="Judge model name passed to eval scripts")
    p.add_argument("--llm_trace_path",type=Path,default=None,
                   help="Append per-call LLM trace as JSONL (request/response/error) to this path (optional).",)
    
    # Runtime options - Concurrency control
    p.add_argument("--max_concurrent_tasks", type=int, default=300,
                   help="Maximum number of tasks to evaluate concurrently (default: 2)")
    p.add_argument("--max_concurrent_answers", type=int, default=15,
                   help="Maximum number of answers to evaluate concurrently per task (default: 10)")
    p.add_argument("--max_webpage_retrieval", type=int, default=60,
                   help="Maximum number of concurrent webpage retrieval operations (playwright) (default: 5)") 
    p.add_argument("--max_llm_requests", type=int, default=60,
                   help="Maximum number of concurrent LLM API requests (default: 30)")

    # Other runtime options
    p.add_argument("--dump_cache", action="store_true", default=True,
                   help="Persist cache to disk at the end (default: True)")
    p.add_argument("--self_debug", action="store_true",
                   help="Add *_debug suffix to logs / result files")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing results")
    p.add_argument("--best_of_k", type=int, default=15,
               help="Evaluate top-k answers per task and aggregate by max score (default: 10)")
    p.add_argument("--best_jsonl_out", type=Path, default=None,
               help="Write best-per-task answers to a jsonl file (optional)")

    return p


# --------------------------------------------------------------------------- #
# Helpers                                                                    #
# --------------------------------------------------------------------------- #


async def evaluate_single_task(
        task_id: str,
        agent_name: str,
        client: LLMClient,
        paths: PathConfig,
        args: argparse.Namespace,
        webpage_semaphore: asyncio.Semaphore,
        llm_semaphore: asyncio.Semaphore
) -> List[Dict[str, Any]]:
    """Evaluate a single task."""
    # Resolve evaluation script
    script_path = paths.default_script_for(task_id)
    if not script_path.exists():
        logging.error(f"Evaluation script not found: {script_path}")
        return []

    # Invoke evaluation with proper concurrency controls
    return await evaluate_task(
        client=client,
        task_id=task_id,
        agent_name=agent_name,
        answer_dir=paths.answers_root,
        cache_dir=paths.cache_root,
        output_dir=paths.eval_results_root,
        script_path=script_path,
        dump_cache=args.dump_cache,
        is_self_debug=args.self_debug,
        overwrite=args.overwrite,
        max_concurrent_answers=args.max_concurrent_answers,
        webpage_semaphore=webpage_semaphore,
        llm_semaphore=llm_semaphore,
        best_of_k=args.best_of_k,
        model = args.judge_model
    )


async def evaluate_all_tasks(
        agent_name: str,
        client: LLMClient,
        paths: PathConfig,
        args: argparse.Namespace,
        webpage_semaphore: asyncio.Semaphore,
        llm_semaphore: asyncio.Semaphore
) -> Dict[str, List[Dict[str, Any]]]:
    """Evaluate all tasks based on available answers for the specified agent."""
    results = {}

    # Find all task directories in the agent's answers folder
    agent_dir = paths.answers_root / agent_name
    if not agent_dir.exists():
        logging.error(f"Agent directory not found: {agent_dir}")
        return results
    
    # Get all task directories (subdirectories in agent folder)
    task_dirs = [d for d in agent_dir.iterdir() if d.is_dir()]
    if not task_dirs:
        logging.warning(f"No task directories found in {agent_dir}")
        return results

    # Verify that corresponding eval scripts exist for each task
    available_tasks = []
    for task_dir in task_dirs:
        task_id = task_dir.name
        script_path = paths.default_script_for(task_id)
        if script_path.exists():
            available_tasks.append(task_id)
        else:
            logging.warning(f"No evaluation script found for task {task_id} at {script_path}")

    if not available_tasks:
        logging.warning(f"No tasks with both answers and evaluation scripts found")
        return results
    
    logging.info(f"Found {len(available_tasks)} tasks with answers for agent '{agent_name}'")
    logging.info(
        f"Concurrency: {args.max_concurrent_tasks} tasks, {args.max_concurrent_answers} answers/task, {args.max_webpage_retrieval} webpage ops, {args.max_llm_requests} LLM requests")

    # Create a semaphore to limit concurrent task evaluations
    task_semaphore = asyncio.Semaphore(args.max_concurrent_tasks)

    async def evaluate_task_with_semaphore(current_task_id: str) -> tuple[str, List[Dict[str, Any]]]:
        """Evaluate a single task with semaphore control."""
        async with task_semaphore:
            try:
                logging.info(f"🚀 Starting evaluation for task: {current_task_id}")
                current_results = await evaluate_single_task(
                    task_id=current_task_id,
                    agent_name=agent_name,
                    client=client,
                    paths=paths,
                    args=args,
                    webpage_semaphore=webpage_semaphore,
                    llm_semaphore=llm_semaphore
                )
                if current_results:
                    logging.info(f"✅ Task {current_task_id}: {len(current_results)} results")
                else:
                    logging.warning(f"⚠️ Task {current_task_id}: No results")
                return current_task_id, current_results
            except Exception as e:
                logging.error(f"❌ Failed to evaluate task {current_task_id}: {e}")
                return current_task_id, []

    # Create tasks for all evaluations
    tasks = []
    for task_id in available_tasks:
        tasks.append(evaluate_task_with_semaphore(task_id))

    # Run all tasks concurrently with progress bar
    logging.info(f"🏃 Starting concurrent evaluation of {len(tasks)} tasks")

    # Use tqdm to show progress
    completed = 0
    with tqdm(total=len(tasks), desc="Evaluating tasks", unit="task") as pbar:
        for coro in asyncio.as_completed(tasks):
            task_id, task_results = await coro
            results[task_id] = task_results
            completed += 1
            pbar.update(1)
            pbar.set_postfix({"completed": f"{completed}/{len(tasks)}"})

    return results


def _safe_score(value: Any) -> float:
    """Best-effort float conversion for scores."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _walk_verification_trees(result: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield all verification trees present in the result payload."""
    tree = result.get("verification_tree")
    if isinstance(tree, dict):
        yield tree

    eval_breakdown = result.get("eval_breakdown")
    if isinstance(eval_breakdown, list):
        for section in eval_breakdown:
            subsection_tree = section.get("verification_tree")
            if isinstance(subsection_tree, dict):
                yield subsection_tree


def _collect_failure_reasons(result: Dict[str, Any], limit: int = 5) -> List[str]:
    """Extract human-readable failure reasons from verification trees."""
    reasons: List[str] = []

    def _enqueue_nodes(tree: Dict[str, Any]):
        queue = [tree]
        while queue and len(reasons) < limit:
            node = queue.pop(0)
            if not isinstance(node, dict):
                continue
            status = node.get("status")
            children = node.get("children") or []
            if isinstance(children, list):
                queue.extend(children)
            if status != "failed":
                continue
            desc = (node.get("desc") or "").strip()
            node_id = node.get("id")
            label = desc or (f"Node {node_id}" if node_id else "Verification failed")
            if node_id and desc:
                label = f"{node_id}: {desc}"
            elif node_id and not desc:
                label = f"Node {node_id} failed"
            reasons.append(label)

    for tree in _walk_verification_trees(result):
        _enqueue_nodes(tree)
        if len(reasons) >= limit:
            break

    if not reasons:
        fallback = result.get("failure_reason") or result.get("error_message")
        if fallback:
            reasons.append(str(fallback))

    if not reasons:
        reasons.append("Final score below success threshold")

    return reasons[:limit]


def _summarize_single_task(task_id: str, task_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Construct summary information for a single task."""
    summary: Dict[str, Any] = {
        "task_id": task_id,
        "num_results": len(task_results),
        "avg_score": 0.0,
        "status": "no_results",
        "success_answers": [],
        "failure_details": []
    }

    if not task_results:
        summary["failure_details"] = [{"reason": "No evaluation results generated"}]
        return summary

    scores = [_safe_score(res.get("final_score")) for res in task_results]
    best_score = max(scores) if scores else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    summary["avg_score"] = avg_score          # keep average score
    summary["best_score"] = best_score        # add best score
    
    summary["best_answer"] = None
    if task_results:
        best_res = max(task_results, key=lambda r: _safe_score(r.get("final_score")))
        summary["best_answer"] = best_res.get("answer_name", "unknown")

    
    summary["success_answers"] = [
        res.get("answer_name", "unknown")
        for res in task_results
        if _safe_score(res.get("final_score")) >= 1.0
    ]

    failure_entries = []
    for res in task_results:
        score = _safe_score(res.get("final_score"))
        if score >= 1.0:
            continue
        failure_entries.append({
            "answer_name": res.get("answer_name", "unknown"),
            "score": score,
            "failure_reasons": _collect_failure_reasons(res)
        })
    summary["failure_details"] = failure_entries

    if len(summary["success_answers"]) == len(task_results):
        summary["status"] = "success"
    elif summary["success_answers"]:
        summary["status"] = "partial"
    else:
        summary["status"] = "failed"

    return summary


def write_run_summary(
        agent_name: str,
        results: Dict[str, List[Dict[str, Any]]],
        eval_results_root: Path,
        *,
        best_of_k: int = 3,
        run_started_at_utc: Optional[datetime] = None,
        run_finished_at_utc: Optional[datetime] = None,
) -> Optional[Path]:
    """Persist a comprehensive summary JSON for the current evaluation run."""
    if results is None:
        return None

    tasks_summary = [
        _summarize_single_task(task_id, task_results)
        for task_id, task_results in sorted(results.items())
    ]
    tasks_with_results = [t for t in tasks_summary if t.get("num_results", 0) > 0]

    # Per-answer success rates (answer_1.md / answer_2.md / ...), computed over tasks
    # where that specific answer exists in the evaluation output.
    per_answer_stats: Dict[str, Dict[str, Any]] = {}
    if best_of_k and best_of_k > 0:
        for i in range(1, best_of_k + 1):
            answer_name = f"answer_{i}.md"
            evaluated_tasks = 0
            success_tasks = 0
            scores: List[float] = []

            for task_results in results.values():
                if not task_results:
                    continue
                res = next(
                    (r for r in task_results if r.get("answer_name") == answer_name),
                    None,
                )
                if res is None:
                    continue
                evaluated_tasks += 1
                score = _safe_score(res.get("final_score"))
                scores.append(score)
                if score >= 1.0:
                    success_tasks += 1

            per_answer_stats[answer_name] = {
                "tasks_evaluated": evaluated_tasks,
                "success_task_count": success_tasks,
                "success_rate": round(success_tasks / evaluated_tasks, 4) if evaluated_tasks else 0.0,
            }

    success_task_count = sum(
        1 for t in tasks_with_results
        if _safe_score(t.get("best_score")) >= 1.0
    )

    success_rate = (
        success_task_count / len(tasks_with_results)
        if tasks_with_results else 0.0
    )

    total_results = sum(task["num_results"] for task in tasks_summary)
    overall_avg = (
        sum(
            _safe_score(res.get("final_score"))
            for task_results in results.values()
            for res in task_results
        ) / total_results
        if total_results else 0.0
    )

    total_input_tokens = sum(
        int((res.get("token_usage") or {}).get("input_tokens", 0) or 0)
        for task_results in results.values()
        for res in task_results
        if isinstance(res, dict)
    )
    total_output_tokens = sum(
        int((res.get("token_usage") or {}).get("output_tokens", 0) or 0)
        for task_results in results.values()
        for res in task_results
        if isinstance(res, dict)
    )

    tasks_with_results_count = len(tasks_with_results)
    best_avg = (
        sum(_safe_score(task.get("best_score")) for task in tasks_with_results) / tasks_with_results_count
        if tasks_with_results_count else 0.0
    )

    summary_payload: Dict[str, Any] = {
        "agent": agent_name,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "overall": {
            "total_tasks": len(tasks_summary),
            "tasks_with_results": len(tasks_with_results),
            "success_task_count": success_task_count,
            "success_rate": round(success_rate, 4),
            "per_answer_success_rate": per_answer_stats,
            "total_results": total_results,
            "average_score": round(overall_avg, 4),  
            "best_of_k_average_score": round(best_avg, 4), 
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "successful_tasks": [
                task["task_id"] for task in tasks_summary if task["status"] == "success"
            ],
            "failed_tasks": [
                task["task_id"]
                for task in tasks_summary
                if task["status"] in ("failed", "no_results")
            ],
            "partial_tasks": [
                task["task_id"] for task in tasks_summary if task["status"] == "partial"
            ],
        },
        "tasks": tasks_summary,
    }

    if run_started_at_utc and run_finished_at_utc:
        elapsed_seconds = max(0.0, (run_finished_at_utc - run_started_at_utc).total_seconds())
        summary_payload["run"] = {
            "started_at_utc": run_started_at_utc.isoformat() + "Z",
            "finished_at_utc": run_finished_at_utc.isoformat() + "Z",
            "elapsed_seconds": round(elapsed_seconds, 3),
        }

    summary_dir = eval_results_root / agent_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary_payload, fp, ensure_ascii=False, indent=2)

    return summary_path

def write_best_answers_jsonl(*, agent_name: str, results: Dict[str, List[Dict[str, Any]]], answers_root: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for task_id, task_results in sorted(results.items()):
            if not task_results:
                continue
            best_res = max(task_results, key=lambda r: _safe_score(r.get("final_score")))
            answer_name = best_res.get("answer_name", "unknown")
            answer_path = answers_root / agent_name / task_id / answer_name
            answer_text = answer_path.read_text(encoding="utf-8") if answer_path.exists() else ""

            row = {
                "task_id": task_id,
                "agent_name": agent_name,
                "answer_name": answer_name,
                "final_score": _safe_score(best_res.get("final_score")),
                "answer": answer_text,
                "answer_path": str(answer_path),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def run_evaluation(args: argparse.Namespace, paths: PathConfig):
    """Main evaluation runner."""
    # Build async client
    client = LLMClient(provider=args.llm_provider, is_async=True)

    # Create separate semaphores for webpage retrieval and LLM requests
    webpage_semaphore = asyncio.Semaphore(args.max_webpage_retrieval)
    llm_semaphore = asyncio.Semaphore(args.max_llm_requests)

    if args.task_id:
        # Evaluate single task
        logging.info(f"Evaluating single task: {args.task_id}")
        results = await evaluate_single_task(
            task_id=args.task_id,
            agent_name=args.agent_name,
            client=client,
            paths=paths,
            args=args,
            webpage_semaphore=webpage_semaphore,
            llm_semaphore=llm_semaphore
        )
        return {args.task_id: results}
    else:
        # Evaluate all tasks
        logging.info("Evaluating all tasks")
        return await evaluate_all_tasks(
            agent_name=args.agent_name,
            client=client,
            paths=paths,
            args=args,
            webpage_semaphore=webpage_semaphore,
            llm_semaphore=llm_semaphore
        )


# --------------------------------------------------------------------------- #
# Entrypoint                                                                 #
# --------------------------------------------------------------------------- #


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Initialize paths
    project_root = Path(__file__).resolve().parent
    paths = PathConfig(project_root)

    # Parse arguments
    args = build_parser().parse_args()

    # if args.llm_trace_path is not None:
    #     set_default_trace_path(str(args.llm_trace_path))

    # Apply path overrides
    paths.apply_overrides(
        answers_root=args.answer_folder,
        eval_scripts_root=args.eval_scripts_root,
        eval_results_root=args.eval_results_root,
        cache_root=args.cache_root,
        eval_version=args.eval_version,
    )

    # Validate answer folder structure
    agent_dir = paths.answers_root / args.agent_name
    if not agent_dir.exists():
        logging.error(f"Agent directory not found: {agent_dir}")
        logging.error(f"Expected structure: {paths.answers_root}/<agent_name>/<task_id>/answer_*.md")
        return

    logging.info("=" * 60)
    logging.info("Mind2Web2 Evaluation Runner")
    logging.info("=" * 60)
    logging.info(f"Agent: {args.agent_name}")
    logging.info(f"Answer folder: {paths.answers_root}")
    logging.info(f"Eval scripts root: {paths.eval_scripts_root}")
    logging.info(f"Eval results root: {paths.eval_results_root}")
    logging.info(f"Cache root: {paths.cache_root}")
    logging.info(f"LLM Provider: {args.llm_provider}")
    logging.info(f"LLM Trace path: {args.llm_trace_path}")
    logging.info("Concurrency Settings:")
    if not args.task_id:
        logging.info(f"  • Max concurrent tasks: {args.max_concurrent_tasks}")
    logging.info(f"  • Max concurrent answers per task: {args.max_concurrent_answers}")
    logging.info(f"  • Max concurrent webpage retrieval (global): {args.max_webpage_retrieval}")
    logging.info(f"  • Max concurrent LLM requests (global): {args.max_llm_requests}")
    logging.info("=" * 60)

    # Run async evaluation
    run_started_at_utc = datetime.utcnow()
    results = asyncio.run(run_evaluation(args, paths))
    run_finished_at_utc = datetime.utcnow()

    # Log summary
    logging.info("=" * 60)
    logging.info("Evaluation Summary")
    logging.info("=" * 60)

    if args.task_id:
        task_results = results.get(args.task_id, [])
        logging.info(f"Task {args.task_id}: {len(task_results)} results")
        for res in task_results:
            score = res.get('final_score', 'N/A')
            answer = res.get('answer_name', 'unknown')
            logging.info(f"  - {answer}: score={score}")
    else:
        total_results = sum(len(r) for r in results.values())
        logging.info(f"Evaluated {len(results)} tasks with {total_results} total results")
        for task_id, task_results in sorted(results.items()):
            if task_results:
                avg_score = sum(r.get('final_score', 0) for r in task_results) / len(task_results)
                logging.info(f"  - {task_id}: {len(task_results)} results, avg_score={avg_score:.2f}")
            else:
                logging.info(f"  - {task_id}: No results")

    # Merge all results if evaluating all tasks
    if not args.task_id and results:
        logging.info("=" * 60)
        logging.info("Merging all results...")
        merge_all_results(paths.eval_results_root)
        logging.info("✅ Results merged successfully")

    summary_path = write_run_summary(
        args.agent_name,
        results,
        paths.eval_results_root,
        best_of_k=args.best_of_k,
        run_started_at_utc=run_started_at_utc,
        run_finished_at_utc=run_finished_at_utc,
    )
    if summary_path:
        logging.info(f"📄 Detailed summary saved to {summary_path}")
        
    if args.best_jsonl_out:
        write_best_answers_jsonl(
            agent_name=args.agent_name,
            results=results,
            answers_root=paths.answers_root,
            out_path=args.best_jsonl_out,
        )


    logging.info("=" * 60)
    logging.info("🎉 Evaluation completed!")


if __name__ == "__main__":
    main()
