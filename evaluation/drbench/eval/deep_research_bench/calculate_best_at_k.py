#!/usr/bin/env python3
"""
Compute best@k and average overall_score.
For each question, keep the highest overall_score across the three iter runs.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent


def load_results(file_path: Path) -> Dict[int, float]:
    """
    Load a result file and return an id -> overall_score mapping.

    Args:
        file_path: path to the result file

    Returns:
        A dictionary mapping id -> overall_score.
    """
    results = {}
    if not file_path.exists():
        return results
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                question_id = data.get('id')
                overall_score = data.get('overall_score')
                if question_id is not None and overall_score is not None:
                    results[question_id] = overall_score
            except json.JSONDecodeError as e:
                print(f"Warning: failed to parse JSON in {file_path}: {e}")
                continue
    
    return results


def calculate_best_at_k(base_dir: Path, config: str, k: int = 3) -> Dict:
    """
    Compute best@k statistics.

    Args:
        base_dir: base result directory
        config: configuration name (for example vanilla or all-data)
        k: number of iter runs to consider

    Returns:
        A dictionary with the aggregated statistics.
    """
    # Load results from all iter runs.
    iter_results = {}
    iter_avgs = {}
    
    for i in range(1, k + 1):
        # Directory format: {config}-iter{i}
        iter_dir = base_dir / f"{config}-iter{i}"
        
        if iter_dir.exists():
            result_file = iter_dir / "raw_results.jsonl"
            if result_file.exists():
                iter_results[i] = load_results(result_file)
                if iter_results[i]:
                    iter_avgs[i] = sum(iter_results[i].values()) / len(iter_results[i])
                else:
                    iter_avgs[i] = 0.0
            else:
                print(f"  Warning: file does not exist {result_file}")
        else:
            print(f"  Warning: directory does not exist {iter_dir}")
    
    if not iter_results:
        return None
    
    # Collect all question ids.
    all_ids = set()
    for results in iter_results.values():
        all_ids.update(results.keys())
    
    # Compute best@k: for each question, keep the highest overall_score across the k iter runs.
    best_scores = {}
    for question_id in all_ids:
        scores = []
        for i in range(1, k + 1):
            if i in iter_results and question_id in iter_results[i]:
                scores.append(iter_results[i][question_id])
        
        if scores:
            best_scores[question_id] = max(scores)
    
    # Compute summary statistics.
    if best_scores:
        best_at_k_avg = sum(best_scores.values()) / len(best_scores)
    else:
        best_at_k_avg = 0.0
    
    return {
        "config": config,
        "best_at_k": {
            "k": k,
            "avg_overall_score": best_at_k_avg,
            "total_questions": len(best_scores)
        },
        "iter_avgs": iter_avgs,
        "all_iter_avg": sum(iter_avgs.values()) / len(iter_avgs) if iter_avgs else 0.0
    }


def main():
    # Result directory.
    base_dir = SCRIPT_DIR / "results/race"
    
    # Configuration list (adjust as needed).
    configs = [
        "a3b-results",
        "base-mid-training-20260223-post-training-20260227-2k-20k-traj-6ep",
        "base-mid-training-20260223-post-training-20260227",
        "base-vanilla-post-training-20260227-2k-20k-traj-6ep"
    ]
    
    all_stats = []
    
    print("=" * 80)
    print("Best@3 and Average Overall Score")
    print("=" * 80)
    print()
    
    for config in configs:
        # Check whether this configuration exists (at least one iter directory must exist).
        found = False
        for i in range(1, 4):
            iter_dir = base_dir / f"{config}-iter{i}"
            if iter_dir.exists():
                found = True
                break
        
        if not found:
            print(f"Skipping: configuration not found: {config}")
            continue
        
        print(f"Processing configuration: {config}")
        stats = calculate_best_at_k(base_dir, config, k=3)
        
        if stats:
            all_stats.append(stats)
            print(f"  Best@3 Average Overall Score: {stats['best_at_k']['avg_overall_score']:.4f}")
            print(f"  Total questions: {stats['best_at_k']['total_questions']}")
            print(f"  Per-iter averages:")
            for iter_num, avg in sorted(stats['iter_avgs'].items()):
                print(f"    Iter{iter_num}: {avg:.4f}")
            print(f"  All-iter average: {stats['all_iter_avg']:.4f}")
        else:
            print(f"  Warning: unable to compute statistics")
        print()
    
    # Print the summary table.
    print("=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(f"{'Config':<25} {'Best@3 Avg':<15} {'Iter1 Avg':<15} {'Iter2 Avg':<15} {'Iter3 Avg':<15} {'All Iter Avg':<15}")
    print("-" * 80)
    
    for stats in sorted(all_stats, key=lambda x: x['best_at_k']['avg_overall_score'], reverse=True):
        config = stats['config']
        best_at_3 = stats['best_at_k']['avg_overall_score']
        iter_avgs = stats['iter_avgs']
        all_iter_avg = stats['all_iter_avg']
        
        iter1_avg = iter_avgs.get(1, 0.0)
        iter2_avg = iter_avgs.get(2, 0.0)
        iter3_avg = iter_avgs.get(3, 0.0)
        
        print(f"{config:<25} {best_at_3:<15.4f} {iter1_avg:<15.4f} {iter2_avg:<15.4f} {iter3_avg:<15.4f} {all_iter_avg:<15.4f}")
    
    # Save the results to a JSON file.
    output_file = base_dir / "best_at_k_stats.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    
    print()
    print(f"Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
