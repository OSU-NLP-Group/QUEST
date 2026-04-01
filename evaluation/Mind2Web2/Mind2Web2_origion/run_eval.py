from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm

from mind2web2.eval_runner import evaluate_task, merge_all_results, generate_result_summary
from mind2web2.llm_client.base_client import LLMClient
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
    p.add_argument("--eval_version", default="2025_10_23",
                   help="Version of evaluation scripts to use (default: 2025_10_23)")

    # LLM configuration
    p.add_argument("--llm_provider", choices=["openai", "azure_openai"],
                   default="openai", help="LLM provider to use")

    # Runtime options - Concurrency control
    p.add_argument("--max_concurrent_tasks", type=int, default=3,
                   help="Maximum number of tasks to evaluate concurrently (default: 2)")
    p.add_argument("--max_concurrent_answers", type=int, default=3,
                   help="Maximum number of answers to evaluate concurrently per task (default: 3)")
    p.add_argument("--max_webpage_retrieval", type=int, default=10,
                   help="Maximum number of concurrent webpage retrieval operations (playwright) (default: 5)")
    p.add_argument("--max_llm_requests", type=int, default=30,
                   help="Maximum number of concurrent LLM API requests (default: 30)")

    # Other runtime options
    p.add_argument("--dump_cache", action="store_true", default=True,
                   help="Persist cache to disk at the end (default: True)")
    p.add_argument("--self_debug", action="store_true",
                   help="Add *_debug suffix to logs / result files")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing results")
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
    logging.info("Concurrency Settings:")
    if not args.task_id:
        logging.info(f"  • Max concurrent tasks: {args.max_concurrent_tasks}")
    logging.info(f"  • Max concurrent answers per task: {args.max_concurrent_answers}")
    logging.info(f"  • Max concurrent webpage retrieval (global): {args.max_webpage_retrieval}")
    logging.info(f"  • Max concurrent LLM requests (global): {args.max_llm_requests}")
    logging.info("=" * 60)

    # Run async evaluation
    results = asyncio.run(run_evaluation(args, paths))

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

    # Generate agent-level summary
    if results:
        logging.info("=" * 60)
        logging.info("Generating agent summary...")
        summary = generate_result_summary(paths.eval_results_root, args.agent_name)
        if summary:
            num_runs = summary["num_runs"]
            pass_key = f"pass_at_{num_runs}"
            logging.info(f"  Partial Completion: {summary['avg_score']:.4f} ± {summary['avg_score_std']:.4f}")
            logging.info(f"  Success Rate:       {summary['success_rate']:.4f} ± {summary['success_rate_std']:.4f}")
            logging.info(f"  Pass@{num_runs}:             {summary[pass_key]:.4f}")
            logging.info(f"  Avg Word Count:     {summary['avg_answer_word_count']:.1f} ± {summary['avg_answer_word_count_std']:.1f}")

    logging.info("=" * 60)
    logging.info("🎉 Evaluation completed!")


if __name__ == "__main__":
    main()
