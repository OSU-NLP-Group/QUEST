#!/usr/bin/env python3
"""
Standalone script to generate an agent-level summary.json from existing eval results.

Usage:
    python generate_result_summary.py eval_results/JudyAgent
    python generate_result_summary.py eval_results/ChatGPTAgent
    python generate_result_summary.py JudyAgent --results-dir eval_results
"""

import argparse
import json
from pathlib import Path

from mind2web2.eval_runner import generate_result_summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate agent-level summary.json from eval results"
    )
    parser.add_argument(
        "agent_path",
        help="Path to agent results folder (e.g. eval_results/JudyAgent) "
             "or just the agent name when --results-dir is provided",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Base results directory (default: inferred from agent_path)",
    )
    args = parser.parse_args()

    agent_path = Path(args.agent_path)

    if args.results_dir is not None:
        # agent_path is just the agent name
        output_dir = Path(args.results_dir)
        agent_name = agent_path.name
    elif agent_path.is_dir():
        # agent_path is a full path like eval_results/JudyAgent
        output_dir = agent_path.parent
        agent_name = agent_path.name
    else:
        parser.error(f"Directory not found: {agent_path}")
        return

    summary = generate_result_summary(output_dir, agent_name)

    if summary:
        num_runs = summary['num_runs']
        pass_key = f"pass_at_{num_runs}"

        print(f"\n{'='*60}")
        print(f"Agent:              {summary['agent_name']}")
        print(f"Tasks:              {summary['num_tasks']}")
        print(f"Runs:               {num_runs}")
        print(f"-" * 60)
        print(f"Partial Completion: {summary['avg_score']:.4f} ± {summary['avg_score_std']:.4f}")
        print(f"Success Rate:       {summary['success_rate']:.4f} ± {summary['success_rate_std']:.4f}")
        print(f"Pass@{num_runs}:             {summary[pass_key]:.4f}")
        print(f"Avg Word Count:     {summary['avg_answer_word_count']:.1f} ± {summary['avg_answer_word_count_std']:.1f}")
        print(f"-" * 60)

        print(f"Per-run breakdown:")
        for run_name, run_data in summary['per_run'].items():
            print(f"  {run_name}: score={run_data['avg_score']:.4f}  "
                  f"success={run_data['success_rate']:.4f}  "
                  f"words={run_data['avg_word_count']:.1f}  "
                  f"(n={run_data['num_tasks']})")

        print(f"{'='*60}")
        print(f"Summary saved to: {Path(output_dir) / agent_name / 'summary.json'}")


if __name__ == "__main__":
    main()
