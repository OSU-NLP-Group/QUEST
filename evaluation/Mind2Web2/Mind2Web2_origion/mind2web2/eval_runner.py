from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, Optional

from tqdm import tqdm

from .utils.cache_filesys import CacheFileSys
from .utils.load_eval_script import load_eval_script
from .utils.logging_setup import create_logger, cleanup_logger


class DualSemaphore:
    """Wrapper to hold both webpage and LLM semaphores."""

    def __init__(self, webpage_semaphore: asyncio.Semaphore, llm_semaphore: asyncio.Semaphore):
        self.webpage = webpage_semaphore
        self.llm = llm_semaphore
        # Default to webpage semaphore for backward compatibility
        self._default = webpage_semaphore

    async def __aenter__(self):
        """For backward compatibility with code expecting a single semaphore."""
        return await self._default.__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """For backward compatibility with code expecting a single semaphore."""
        return await self._default.__aexit__(exc_type, exc_val, exc_tb)


def _answer_base(answer_name: str) -> str:
    """Strip the trailing extension, e.g. 'answer_3.md' → 'answer_3'."""
    return answer_name.rsplit(".", 1)[0]


def _extract_ts_from_name(fname: str) -> str | None:
    """
    Extract a 14-digit timestamp (YYYYMMDDHHMMSS) from the filename, or return None if not found.
    """
    m = re.search(r"(\d{8})[_]?(\d{6})", fname)
    if not m:
        return None
    return "".join(m.groups())


def _latest_json(result_dir: Path) -> Path | None:
    """Return newest *.json file in <result_dir> (by timestamp in filename)."""
    if not result_dir.exists():
        return None
    json_files = [p for p in result_dir.iterdir() if p.suffix == ".json"]
    if not json_files:
        return None

    def _ts_key(fp: Path):
        ts_str = _extract_ts_from_name(fp.name)
        return datetime.strptime(ts_str, "%Y%m%d%H%M%S") if ts_str else datetime.min

    return max(json_files, key=_ts_key)


# --------------------------------------------------------------------------- #
# Single‑answer evaluation                                                    #
# --------------------------------------------------------------------------- #


async def _eval_one_answer(
        eval_fn,
        client,
        task_id: str,
        agent_name: str,
        answer_path: Path,
        cache: CacheFileSys,
        webpage_semaphore: asyncio.Semaphore,
        llm_semaphore: asyncio.Semaphore,
        output_dir: Path,
        is_self_debug: bool = False,
):
    """Evaluate a single answer file and write its result JSON / logs."""

    answer_name = answer_path.name
    answer_base = _answer_base(answer_name)

    # ---------- Create isolated logging ----------
    log_dir = output_dir / agent_name / task_id / answer_base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Use a more specific logger name to ensure uniqueness
    log_tag = f"{task_id}_{agent_name}_{answer_name}"
    if is_self_debug:
        log_tag += "_debug"

    # Important: Disable console output in concurrent environments to avoid log confusion
    logger, timestamp = create_logger(
        log_tag,
        str(log_dir),
        enable_console=False  # Disable console output during concurrency, only output to file
    )

    # Add structured log for task start
    logger.info(
        f"🚀 Starting evaluation for {agent_name}/{answer_name}",
        extra={
            "task_id": task_id,
            "agent_name": agent_name,
            "answer_name": answer_name,
            "answer_base": answer_base,
            "is_debug": is_self_debug,
            "operation": "eval_start"
        }
    )

    # ---------- Read answer ----------
    try:
        answer_text = answer_path.read_text(encoding="utf-8")
        logger.debug(
            f"Answer loaded: {len(answer_text)} characters",
            extra={"answer_length": len(answer_text)}
        )
    except Exception as e:
        logger.error(f"Failed to read answer file: {e}")
        return e

    result = None
    try:
        # Create a dual semaphore wrapper for the eval function
        dual_semaphore = DualSemaphore(webpage_semaphore, llm_semaphore)

        logger.info("🔄 Starting evaluation function")

        result: Dict = await eval_fn(
            client=client,
            answer=answer_text,
            agent_name=agent_name,
            answer_name=answer_name,
            cache=cache,
            semaphore=dual_semaphore,
            logger=logger,
            model="o4-mini",
        )

        logger.info(
            f"✅ Evaluation completed with score: {result.get('final_score', 'unknown')}",
            extra={
                "final_score": result.get("final_score"),
                "operation": "eval_complete"
            }
        )

    except Exception as exc:
        logger.exception(
            "❌ Evaluation raised an exception",
            extra={
                "error_type": type(exc).__name__,
                "operation": "eval_error"
            }
        )
        return exc
    finally:
        # Clean up logger resources
        try:
            cleanup_logger(logger)
        except Exception:
            pass  # Cleanup failure should not affect main flow

    # ---------- Save result ----------
    try:
        if result is not None:
            _save_result_json(result, output_dir / agent_name / task_id, timestamp, is_self_debug)
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to save result for {agent_name}/{answer_name}: {e}")
        return e

    return result


def _save_result_json(result: Dict, agent_task_out_dir: Path, ts: str, is_debug: bool):
    """Write per‑answer result JSON to disk."""

    answer = result["answer_name"]
    answer_base = _answer_base(answer)

    save_dir = agent_task_out_dir / answer_base / "results"
    save_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{ts}_{answer}{'_debug' if is_debug else ''}.json"
    with (save_dir / fname).open("w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=4)


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


async def evaluate_task(
        client,
        task_id: str,
        agent_name: str,
        answer_dir: Union[str, Path],
        cache_dir: Union[str, Path],
        output_dir: Union[str, Path],
        script_path: Union[str, Path],
        dump_cache: bool = True,
        is_self_debug: bool = False,
        overwrite: bool = False,
        max_concurrent_answers: int = 3,
        webpage_semaphore: Optional[asyncio.Semaphore] = None,
        llm_semaphore: Optional[asyncio.Semaphore] = None,
) -> List[Dict]:
    """Evaluate all answers for a specific task and agent.

    Parameters
    ----------
    client : LLMClient
        The LLM client to use for evaluation
    task_id : str
        The task identifier
    agent_name : str
        The agent name to evaluate
    answer_dir : Union[str, Path]
        Base directory containing answers (structure: <answer_dir>/<agent_name>/<task_id>/answer_*.md)
    cache_dir : Union[str, Path]
        Directory for cache files
    output_dir : Union[str, Path]
        Directory for output results
    script_path : Union[str, Path]
        Path to the evaluation script
    dump_cache : bool, default True
        Whether to persist cache to disk
    is_self_debug : bool, default False
        Whether to add debug suffix to logs/results
    overwrite : bool, default False
        Whether to overwrite existing results
    max_concurrent_answers : int, default 3
        Maximum number of concurrent answer evaluations
    webpage_semaphore : Optional[asyncio.Semaphore], default None
        Semaphore for controlling concurrent webpage retrieval operations
    llm_semaphore : Optional[asyncio.Semaphore], default None
        Semaphore for controlling concurrent LLM API requests

    Returns
    -------
    List[Dict]
        List of evaluation results for all answers
    """

    # ------------------------------------------------------------------
    # 0. Setup paths & ensure dirs exist
    # ------------------------------------------------------------------
    answer_root = Path(answer_dir) / agent_name / task_id
    output_root = Path(output_dir)
    cache_root = Path(cache_dir) / agent_name

    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    # Check if answer directory exists
    if not answer_root.exists():
        logging.getLogger(__name__).warning(f"No answers found for {agent_name}/{task_id} at {answer_root}")
        return []

    # ------------------------------------------------------------------
    # 1. Create main task logger (for overall progress tracking)
    # ------------------------------------------------------------------
    main_log_dir = output_root / agent_name / task_id / "main_logs"
    main_log_dir.mkdir(parents=True, exist_ok=True)
    main_logger, main_timestamp = create_logger(
        f"main_{task_id}_{agent_name}",
        str(main_log_dir),
        enable_console=True  # Main logger can output to console
    )

    try:
        main_logger.info(
            f"🎯 Starting task evaluation: {task_id} for agent: {agent_name}",
            extra={
                "task_id": task_id,
                "agent_name": agent_name,
                "max_concurrent_answers": max_concurrent_answers,
                "operation": "task_start"
            }
        )

        # ------------------------------------------------------------------
        # 2. Load eval script & cache
        # ------------------------------------------------------------------
        main_logger.info("📜 Loading evaluation script")
        eval_fn = load_eval_script(script_path)

        cache_path = cache_root / f"{task_id}"
        cache = CacheFileSys(task_dir=str(cache_path))
        main_logger.info(f"💾 Cache loaded from {cache_path}")

        # ------------------------------------------------------------------
        # 3. Collect answer files
        # ------------------------------------------------------------------
        answer_paths = sorted([p for p in answer_root.iterdir() if p.is_file() and p.suffix == ".md"])
        main_logger.info(
            f"📁 Found {len(answer_paths)} answer files to evaluate",
            extra={
                "answer_count": len(answer_paths),
                "answer_paths": [p.name for p in answer_paths]
            }
        )
        main_logger.info(f"-->> Answer Root: {answer_root}")
        main_logger.info(f"-->> Answers to Eval: {[p.name for p in answer_paths]}")

        if not answer_paths:
            main_logger.warning(f"No answer files found in {answer_root}")
            return []

        ok_results: List[Dict] = []

        # ------------------------------------------------------------------
        # 4. Concurrency control
        # ------------------------------------------------------------------
        # Use an outer semaphore to control concurrent answer evaluations
        outer_semaphore = asyncio.Semaphore(max_concurrent_answers)

        # Create default semaphores if not provided
        if webpage_semaphore is None:
            webpage_semaphore = asyncio.Semaphore(5)  # Default webpage limit
        if llm_semaphore is None:
            llm_semaphore = asyncio.Semaphore(30)  # Default LLM limit

        # ------------------------------------------------------------------
        # 5. Define per‑answer coroutine
        # ------------------------------------------------------------------
        async def _process_answer(ans_path: Path):
            async with outer_semaphore:  # Control concurrent answer evaluations
                answer_name = ans_path.name
                answer_base = _answer_base(answer_name)

                main_logger.info(
                    f"👉 Processing {agent_name}/{answer_name}",
                    extra={
                        "agent_name": agent_name,
                        "answer_name": answer_name,
                        "operation": "answer_start"
                    }
                )
                main_logger.info(f"👉 Starting {agent_name} {answer_name}")

                # 5‑A. Copy original md to answer folder (if not copied yet)
                answer_folder = output_root / agent_name / task_id / answer_base
                dst_md = answer_folder / answer_name
                if not dst_md.exists():
                    answer_folder.mkdir(parents=True, exist_ok=True)
                    dst_md.write_bytes(ans_path.read_bytes())
                    main_logger.debug(f"📋 Copied answer file to {dst_md}")

                # 5‑B. Result reuse check
                result_dir = answer_folder / "results"
                latest = _latest_json(result_dir)
                if latest and not overwrite:
                    main_logger.info(
                        f"⚠️ Using existing result for {agent_name}/{answer_name}",
                        extra={
                            "agent_name": agent_name,
                            "answer_name": answer_name,
                            "existing_result": str(latest),
                            "operation": "reuse_result"
                        }
                    )
                    main_logger.info(f"⚠️ Existing result -- {agent_name} {answer_name}")
                    try:
                        result = json.loads(latest.read_text(encoding="utf-8"))
                        main_logger.debug(
                            f"✅ Loaded existing result with score: {result.get('final_score')}",
                            extra={
                                "agent_name": agent_name,
                                "answer_name": answer_name,
                                "final_score": result.get('final_score'),
                                "operation": "existing_result_loaded"
                            }
                        )
                        return result
                    except Exception as exc:
                        main_logger.error(
                            f"❌ Failed to load existing result: {exc}",
                            extra={
                                "agent_name": agent_name,
                                "answer_name": answer_name,
                                "error": str(exc),
                                "operation": "existing_result_error"
                            }
                        )

                # 5‑C. Real evaluation
                try:
                    res = await _eval_one_answer(
                        eval_fn,
                        client,
                        task_id,
                        agent_name,
                        ans_path,
                        cache,
                        webpage_semaphore,
                        llm_semaphore,
                        output_root,
                        is_self_debug,
                    )

                    if isinstance(res, dict):
                        main_logger.info(
                            f"✅ Successfully evaluated {agent_name}/{answer_name}",
                            extra={
                                "agent_name": agent_name,
                                "answer_name": answer_name,
                                "final_score": res.get('final_score'),
                                "operation": "answer_complete"
                            }
                        )
                    else:
                        main_logger.error(
                            f"❌ Evaluation failed for {agent_name}/{answer_name}: {res}",
                            extra={
                                "agent_name": agent_name,
                                "answer_name": answer_name,
                                "error": str(res),
                                "operation": "answer_error"
                            }
                        )

                    return res
                except Exception as exc:
                    main_logger.exception(
                        f"💥 Unexpected error evaluating {agent_name}/{answer_name}",
                        extra={
                            "agent_name": agent_name,
                            "answer_name": answer_name,
                            "error_type": type(exc).__name__,
                            "operation": "answer_exception"
                        }
                    )
                    return exc

        # ------------------------------------------------------------------
        # 6. Kick off evaluations
        # ------------------------------------------------------------------
        main_logger.info(f"🚀 Starting concurrent evaluation of {len(answer_paths)} answers")
        tasks = [asyncio.create_task(_process_answer(p)) for p in answer_paths]

        completed_count = 0
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"[{task_id}/{agent_name}] Evaluating"):
            res = await coro
            completed_count += 1

            if isinstance(res, dict):
                ok_results.append(res)
                main_logger.debug(
                    f"✅ [{completed_count}/{len(tasks)}] Completed evaluation for {res.get('agent_name')}/{res.get('answer_name')}",
                    extra={
                        "completed_count": completed_count,
                        "total_count": len(tasks),
                        "agent_name": res.get('agent_name'),
                        "answer_name": res.get('answer_name'),
                        "final_score": res.get('final_score'),
                        "operation": "progress_update"
                    }
                )
            else:
                main_logger.error(
                    f"❌ [{completed_count}/{len(tasks)}] Evaluation failed with error: {res}",
                    extra={
                        "completed_count": completed_count,
                        "total_count": len(tasks),
                        "error": str(res),
                        "operation": "progress_error"
                    }
                )

        # ------------------------------------------------------------------
        # 7. Persist cache & save summary
        # ------------------------------------------------------------------
        cache.save()

        # Save summary for this agent/task combination
        _save_agent_task_summary(output_root / agent_name / task_id, ok_results)
        main_logger.info("📊 Summary saved successfully")

        main_logger.info(
            f"🎉 Task evaluation completed: {len(ok_results)}/{len(answer_paths)} successful results",
            extra={
                "task_id": task_id,
                "agent_name": agent_name,
                "successful_count": len(ok_results),
                "total_count": len(answer_paths),
                "success_rate": len(ok_results) / len(answer_paths) if answer_paths else 0,
                "operation": "task_complete"
            }
        )

        return ok_results

    except Exception as e:
        main_logger.exception(
            f"💥 Task evaluation failed: {e}",
            extra={
                "task_id": task_id,
                "agent_name": agent_name,
                "error_type": type(e).__name__,
                "operation": "task_error"
            }
        )
        raise
    finally:
        # Clean up main logger
        try:
            cleanup_logger(main_logger)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Summary helpers                                                             #
# --------------------------------------------------------------------------- #


def _save_agent_task_summary(agent_task_dir: Path, results: List[Dict]):
    """Save summary for a specific agent/task combination."""
    if not results:
        return

    summary = []
    for res in sorted(results, key=lambda x: x.get("answer_name", "")):
        summary.append({
            "answer_name": res["answer_name"],
            "score": float(res["final_score"]),
            "status": "success" if res["final_score"] > 0 else "failed",
            "success": res["final_score"] == 1,
        })

    with (agent_task_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=4)


def generate_result_summary(output_dir: Union[str, Path], agent_name: str) -> Optional[Dict]:
    """Generate an agent-level summary from per-task evaluation results.

    Metrics computed:
    - partial_completion: avg score per run → avg across runs (with std)
    - success_rate: avg binary success per run → avg across runs (with std)
    - pass_at_k: task passes if ANY run succeeded
    - word_count: avg word count per run → avg across runs (with std)

    Parameters
    ----------
    output_dir : Union[str, Path]
        Base eval results directory (e.g. ``eval_results/``)
    agent_name : str
        Agent folder name

    Returns
    -------
    dict or None
        Summary dict (also saved to ``<output_dir>/<agent_name>/summary.json``)
    """
    import numpy as np

    agent_dir = Path(output_dir) / agent_name
    if not agent_dir.is_dir():
        logging.getLogger(__name__).error(f"Agent directory not found: {agent_dir}")
        return None

    # ------------------------------------------------------------------ #
    # 1. Collect per-task, per-answer data
    # ------------------------------------------------------------------ #
    # tasks_data[task_id] = [{"answer_name", "score", "success", "word_count"}, ...]
    tasks_data: Dict[str, List[Dict]] = {}

    for task_dir in sorted(agent_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name

        # Skip non-task directories (e.g. main_logs)
        summary_file = task_dir / "summary.json"
        if not summary_file.exists():
            continue

        try:
            with summary_file.open("r", encoding="utf-8") as fp:
                task_summary = json.load(fp)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load {summary_file}: {e}")
            continue

        answers = []
        for entry in task_summary:
            answer_name = entry["answer_name"]
            answer_base = answer_name.rsplit(".", 1)[0]  # answer_1.md → answer_1

            # Word count from the copied .md file
            md_path = task_dir / answer_base / answer_name
            word_count = 0
            if md_path.exists():
                try:
                    word_count = len(md_path.read_text(encoding="utf-8").split())
                except Exception:
                    pass

            answers.append({
                "answer_name": answer_name,
                "score": float(entry.get("score", 0)),
                "success": bool(entry.get("success", False)),
                "word_count": word_count,
            })

        if answers:
            tasks_data[task_id] = answers

    if not tasks_data:
        logging.getLogger(__name__).warning(f"No task results found for {agent_name}")
        return None

    # ------------------------------------------------------------------ #
    # 2. Group by run (answer_k.md)
    # ------------------------------------------------------------------ #
    # run_metrics[answer_name] = {"scores": [...], "successes": [...], "word_counts": [...]}
    run_metrics: Dict[str, Dict[str, list]] = defaultdict(lambda: {"scores": [], "successes": [], "word_counts": []})

    for task_id, answers in tasks_data.items():
        for ans in answers:
            name = ans["answer_name"]
            run_metrics[name]["scores"].append(ans["score"])
            run_metrics[name]["successes"].append(float(ans["success"]))
            run_metrics[name]["word_counts"].append(ans["word_count"])

    # ------------------------------------------------------------------ #
    # 3. Per-run aggregates (avg across tasks within each run)
    # ------------------------------------------------------------------ #
    per_run = {}
    run_avg_scores = []
    run_avg_successes = []
    run_avg_word_counts = []

    for run_name in sorted(run_metrics.keys()):
        m = run_metrics[run_name]
        avg_score = float(np.mean(m["scores"]))
        avg_success = float(np.mean(m["successes"]))
        avg_wc = float(np.mean(m["word_counts"]))

        per_run[run_name] = {
            "num_tasks": len(m["scores"]),
            "avg_score": round(avg_score, 4),
            "success_rate": round(avg_success, 4),
            "avg_word_count": round(avg_wc, 1),
        }

        run_avg_scores.append(avg_score)
        run_avg_successes.append(avg_success)
        run_avg_word_counts.append(avg_wc)

    num_runs = len(per_run)

    # ------------------------------------------------------------------ #
    # 4. Across-run aggregates (avg & std of per-run values)
    # ------------------------------------------------------------------ #
    avg_score = float(np.mean(run_avg_scores))
    avg_success = float(np.mean(run_avg_successes))
    avg_word_count = float(np.mean(run_avg_word_counts))

    std_score = float(np.std(run_avg_scores, ddof=0)) if num_runs > 1 else 0.0
    std_success = float(np.std(run_avg_successes, ddof=0)) if num_runs > 1 else 0.0
    std_word_count = float(np.std(run_avg_word_counts, ddof=0)) if num_runs > 1 else 0.0

    # ------------------------------------------------------------------ #
    # 5. Pass@k — task passes if any run succeeded
    # ------------------------------------------------------------------ #
    num_pass = sum(
        1 for answers in tasks_data.values()
        if any(a["success"] for a in answers)
    )
    pass_at_k = num_pass / len(tasks_data) if tasks_data else 0.0

    # ------------------------------------------------------------------ #
    # 6. Per-task detail
    # ------------------------------------------------------------------ #
    tasks_detail: Dict[str, Dict] = {}
    for task_id in sorted(tasks_data.keys()):
        answers = tasks_data[task_id]
        scores = [a["score"] for a in answers]
        tasks_detail[task_id] = {
            "answers": answers,
            "best_score": max(scores),
            "avg_score": round(float(np.mean(scores)), 6),
            "pass": any(a["success"] for a in answers),
        }

    # ------------------------------------------------------------------ #
    # 7. Build & save summary
    # ------------------------------------------------------------------ #
    summary = {
        "agent_name": agent_name,
        "num_tasks": len(tasks_data),
        "num_runs": num_runs,
        "avg_score": round(avg_score, 4),
        "avg_score_std": round(std_score, 4),
        "success_rate": round(avg_success, 4),
        "success_rate_std": round(std_success, 4),
        f"pass_at_{num_runs}": round(pass_at_k, 4),
        "avg_answer_word_count": round(avg_word_count, 1),
        "avg_answer_word_count_std": round(std_word_count, 1),
        "per_run": per_run,
        "tasks": tasks_detail,
    }

    summary_path = agent_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=4)

    logging.getLogger(__name__).info(f"Agent summary saved to {summary_path}")
    return summary


def merge_all_results(output_dir: Union[str, Path]) -> Dict[str, Dict[str, List[Dict]]]:
    """Merge all evaluation results across tasks and agents.
    
    Returns a nested dictionary: {task_id: {agent_name: [results]}}
    """
    output_root = Path(output_dir)
    merged_results = defaultdict(lambda: defaultdict(list))

    # Iterate through all agent directories
    for agent_dir in output_root.iterdir():
        if not agent_dir.is_dir():
            continue
        agent_name = agent_dir.name

        # Iterate through all task directories within each agent
        for task_dir in agent_dir.iterdir():
            if not task_dir.is_dir():
                continue
            task_id = task_dir.name

            # Look for summary.json
            summary_file = task_dir / "summary.json"
            if summary_file.exists():
                try:
                    with summary_file.open("r", encoding="utf-8") as fp:
                        results = json.load(fp)
                        merged_results[task_id][agent_name] = results
                except Exception as e:
                    logging.getLogger(__name__).error(f"Failed to load summary from {summary_file}: {e}")

    # Save merged results
    merged_file = output_root / "all_results.json"
    with merged_file.open("w", encoding="utf-8") as fp:
        json.dump(dict(merged_results), fp, ensure_ascii=False, indent=4)

    logging.getLogger(__name__).info(f"Merged results saved to {merged_file}")
    return dict(merged_results)
