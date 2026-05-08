#!/usr/bin/env python3
"""
Average results from multiple provider runs.

Usage:
    # Average two summary JSON files (recommended)
    python average_results.py \
        --input-a results/reports_graded_openai_gpt-5/summary_2025-10-31T01-38-49.json \
        --input-b results/reports_graded_gemini_gemini-2.5-pro/summary_2025-10-31T01-45-23.json \
        --output results/averaged/summary_multi_judge.json
    
    # Average two detailed results JSON files (legacy)
    python average_results.py \
        --input-a results/reports_graded_gpt5.json \
        --input-b results/reports_graded_gemini.json \
        --output results/reports_averaged.json
    
    # Average two CSV summary files
    python average_results.py \
        --input-a results/run_gpt5/summary.csv \
        --input-b results/run_gemini/summary.csv \
        --output results/averaged_summary.csv
"""

import argparse
import json
import sys
import pandas as pd
from pathlib import Path


def average_criterion_scores(results_a: dict, results_b: dict) -> dict:
    """
    Average scores from two provider results for a single criterion.
    
    Args:
        results_a: Results from provider A
        results_b: Results from provider B
        
    Returns:
        Averaged results
    """
    averaged = {
        'provider_a': results_a.get('provider', 'unknown'),
        'provider_b': results_b.get('provider', 'unknown'),
        'averaged_at': pd.Timestamp.now().isoformat()
    }
    
    # Average score
    if 'score' in results_a and 'score' in results_b:
        averaged['score'] = (results_a['score'] + results_b['score']) / 2
        averaged['score_a'] = results_a['score']
        averaged['score_b'] = results_b['score']
    
    # Average issue count if present
    if 'total_issues' in results_a and 'total_issues' in results_b:
        averaged['total_issues'] = (results_a['total_issues'] + results_b['total_issues']) / 2
        averaged['total_issues_a'] = results_a['total_issues']
        averaged['total_issues_b'] = results_b['total_issues']
    
    # Combine specific issues
    if 'specific_issues' in results_a and 'specific_issues' in results_b:
        averaged['specific_issues_a'] = results_a['specific_issues']
        averaged['specific_issues_b'] = results_b['specific_issues']
    
    return averaged


def average_summary_files(summary_a_path: str, summary_b_path: str, output_path: str):
    """
    Average two summary JSON files (new format with results_by_model and overall_results).
    
    Args:
        summary_a_path: Path to first summary file
        summary_b_path: Path to second summary file
        output_path: Path for averaged output
    """
    print(f"📄 Loading summary files...")
    with open(summary_a_path) as f:
        data_a = json.load(f)
    with open(summary_b_path) as f:
        data_b = json.load(f)
    
    metadata_a = data_a.get('metadata', {})
    metadata_b = data_b.get('metadata', {})
    
    print(f"Provider A: {metadata_a.get('provider', 'unknown')} ({metadata_a.get('model', 'unknown')})")
    print(f"Provider B: {metadata_b.get('provider', 'unknown')} ({metadata_b.get('model', 'unknown')})")
    
    # Average results by model
    results_by_model_a = data_a.get('results_by_model', {})
    results_by_model_b = data_b.get('results_by_model', {})
    
    averaged_by_model = {}
    all_models = set(results_by_model_a.keys()) | set(results_by_model_b.keys())
    
    for model_name in all_models:
        model_a = results_by_model_a.get(model_name, {})
        model_b = results_by_model_b.get(model_name, {})
        
        averaged_by_model[model_name] = {}
        all_criteria = set(model_a.keys()) | set(model_b.keys())
        
        for criterion in all_criteria:
            stats_a = model_a.get(criterion, {})
            stats_b = model_b.get(criterion, {})
            
            if stats_a and stats_b:
                if criterion == 'depth':
                    # Depth: average win rates across judges (excluding ties)
                    averaged_by_model[model_name][criterion] = {
                        'wins': (stats_a.get('wins', 0) + stats_b.get('wins', 0)) / 2,
                        'losses': (stats_a.get('losses', 0) + stats_b.get('losses', 0)) / 2,
                        'ties': (stats_a.get('ties', 0) + stats_b.get('ties', 0)) / 2,
                        'total': stats_a.get('total', 0),
                        'decisive_games': (stats_a.get('decisive_games', 0) + stats_b.get('decisive_games', 0)) / 2,
                        'win_rate': (stats_a.get('win_rate', 0) + stats_b.get('win_rate', 0)) / 2,
                        'reference_model': stats_a.get('reference_model', 'open-deep-research'),
                        # By judge breakdown
                        'by_judge': {
                            'judge_a': {
                                'provider': metadata_a.get('provider', 'unknown'),
                                'model': metadata_a.get('model', 'unknown'),
                                'win_rate': stats_a.get('win_rate', 0),
                                'wins': stats_a.get('wins', 0),
                                'losses': stats_a.get('losses', 0),
                                'ties': stats_a.get('ties', 0),
                                'decisive_games': stats_a.get('decisive_games', 0)
                            },
                            'judge_b': {
                                'provider': metadata_b.get('provider', 'unknown'),
                                'model': metadata_b.get('model', 'unknown'),
                                'win_rate': stats_b.get('win_rate', 0),
                                'wins': stats_b.get('wins', 0),
                                'losses': stats_b.get('losses', 0),
                                'ties': stats_b.get('ties', 0),
                                'decisive_games': stats_b.get('decisive_games', 0)
                            }
                        }
                    }
                else:
                    # Other criteria: standard averaging
                    averaged_by_model[model_name][criterion] = {
                        'mean': (stats_a.get('mean', 0) + stats_b.get('mean', 0)) / 2,
                        'mean_a': stats_a.get('mean', 0),
                        'mean_b': stats_b.get('mean', 0),
                        'count': stats_a.get('count', 0),  # Should be same
                        'min': min(stats_a.get('min', 0), stats_b.get('min', 0)),
                        'max': max(stats_a.get('max', 0), stats_b.get('max', 0))
                    }
            elif stats_a:
                # Only A has data
                averaged_by_model[model_name][criterion] = stats_a
            elif stats_b:
                # Only B has data
                averaged_by_model[model_name][criterion] = stats_b
    
    # Average overall results
    overall_a = data_a.get('overall_results', {})
    overall_b = data_b.get('overall_results', {})
    
    averaged_overall = {}
    all_criteria = set(overall_a.keys()) | set(overall_b.keys())
    
    for criterion in all_criteria:
        stats_a = overall_a.get(criterion, {})
        stats_b = overall_b.get(criterion, {})
        
        if stats_a and stats_b:
            if criterion == 'depth':
                # Depth: average win rates across judges (excluding ties)
                averaged_overall[criterion] = {
                    'wins': (stats_a.get('wins', 0) + stats_b.get('wins', 0)) / 2,
                    'losses': (stats_a.get('losses', 0) + stats_b.get('losses', 0)) / 2,
                    'ties': (stats_a.get('ties', 0) + stats_b.get('ties', 0)) / 2,
                    'total': stats_a.get('total', 0),  # Should be same
                    'decisive_games': (stats_a.get('decisive_games', 0) + stats_b.get('decisive_games', 0)) / 2,
                    'win_rate': (stats_a.get('win_rate', 0) + stats_b.get('win_rate', 0)) / 2,
                    # By judge breakdown
                    'by_judge': {
                        'judge_a': {
                            'provider': metadata_a.get('provider', 'unknown'),
                            'model': metadata_a.get('model', 'unknown'),
                            'win_rate': stats_a.get('win_rate', 0),
                            'wins': stats_a.get('wins', 0),
                            'losses': stats_a.get('losses', 0),
                            'ties': stats_a.get('ties', 0),
                            'decisive_games': stats_a.get('decisive_games', 0)
                        },
                        'judge_b': {
                            'provider': metadata_b.get('provider', 'unknown'),
                            'model': metadata_b.get('model', 'unknown'),
                            'win_rate': stats_b.get('win_rate', 0),
                            'wins': stats_b.get('wins', 0),
                            'losses': stats_b.get('losses', 0),
                            'ties': stats_b.get('ties', 0),
                            'decisive_games': stats_b.get('decisive_games', 0)
                        }
                    }
                }
            else:
                # Other criteria: standard averaging
                averaged_overall[criterion] = {
                    'mean': (stats_a.get('mean', 0) + stats_b.get('mean', 0)) / 2,
                    'mean_a': stats_a.get('mean', 0),
                    'mean_b': stats_b.get('mean', 0),
                    'count': stats_a.get('count', 0),
                    'min': min(stats_a.get('min', 0), stats_b.get('min', 0)),
                    'max': max(stats_a.get('max', 0), stats_b.get('max', 0))
                }
        elif stats_a:
            averaged_overall[criterion] = stats_a
        elif stats_b:
            averaged_overall[criterion] = stats_b
    
    # Build output
    output_data = {
        'metadata': {
            'averaged_from': {
                'file_a': summary_a_path,
                'provider_a': metadata_a.get('provider', 'unknown'),
                'model_a': metadata_a.get('model', 'unknown'),
                'file_b': summary_b_path,
                'provider_b': metadata_b.get('provider', 'unknown'),
                'model_b': metadata_b.get('model', 'unknown')
            },
            'averaged_at': pd.Timestamp.now().isoformat(),
            'total_reports': metadata_a.get('total_reports', 0),
            'criteria_evaluated': metadata_a.get('criteria_evaluated', [])
        },
        'results_by_model': averaged_by_model,
        'overall_results': averaged_overall
    }
    
    # Save
    import os
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Averaged summary results")
    print(f"📄 Saved to: {output_path}")
    
    # Print summary
    print(f"\n📊 Averaged Results:")
    for criterion, stats in averaged_overall.items():
        if criterion == 'depth':
            judge_a_info = stats.get('by_judge', {}).get('judge_a', {})
            judge_b_info = stats.get('by_judge', {}).get('judge_b', {})
            print(f"  {criterion}: Win Rate {stats.get('win_rate', 0):.2f}% "
                  f"({stats.get('wins', 0):.0f}W/{stats.get('losses', 0):.0f}L/{stats.get('ties', 0):.0f}T)")
            print(f"    Judge A ({judge_a_info.get('provider', 'unknown')}): {judge_a_info.get('win_rate', 0):.2f}%")
            print(f"    Judge B ({judge_b_info.get('provider', 'unknown')}): {judge_b_info.get('win_rate', 0):.2f}%")
        else:
            print(f"  {criterion}: {stats.get('mean', 0):.2f} (A: {stats.get('mean_a', 0):.2f}, B: {stats.get('mean_b', 0):.2f})")


def average_json_results(json_a_path: str, json_b_path: str, output_path: str):
    """Average two JSON result files (detailed results format)."""
    print(f"📄 Loading JSON files...")
    with open(json_a_path) as f:
        data_a = json.load(f)
    with open(json_b_path) as f:
        data_b = json.load(f)
    
    reports_a = {r['query_id']: r for r in data_a.get('reports', [])}
    reports_b = {r['query_id']: r for r in data_b.get('reports', [])}
    
    print(f"Found {len(reports_a)} reports in file A, {len(reports_b)} reports in file B")
    
    # Find common query IDs
    common_qids = set(reports_a.keys()) & set(reports_b.keys())
    print(f"Found {len(common_qids)} matching reports")
    
    # Average results
    averaged_reports = []
    for qid in common_qids:
        report = reports_a[qid].copy()
        
        # Find grading result keys (e.g., presentation_grading_results, consistency_grading_results)
        grading_keys = [k for k in report.keys() if k.endswith('_grading_results')]
        
        for key in grading_keys:
            if key in reports_b[qid]:
                report[key] = average_criterion_scores(
                    report[key],
                    reports_b[qid][key]
                )
        
        averaged_reports.append(report)
    
    # Save averaged results
    output_data = {
        'metadata': {
            'averaged_from': {
                'file_a': json_a_path,
                'file_b': json_b_path
            },
            'total_reports': len(averaged_reports),
            'averaged_at': pd.Timestamp.now().isoformat()
        },
        'reports': averaged_reports
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Averaged {len(averaged_reports)} reports")
    print(f"📄 Saved to: {output_path}")


def average_csv_summaries(csv_a_path: str, csv_b_path: str, output_path: str):
    """Average two CSV summary files."""
    print(f"📄 Loading CSV files...")
    df_a = pd.read_csv(csv_a_path)
    df_b = pd.read_csv(csv_b_path)
    
    # Merge on model column (or first column)
    merge_col = 'model' if 'model' in df_a.columns else df_a.columns[0]
    merged = df_a.merge(df_b, on=merge_col, suffixes=('_a', '_b'))
    
    print(f"Found {len(merged)} matching rows")
    
    # Average numeric columns
    numeric_cols = df_a.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if col != merge_col:
            if f'{col}_a' in merged.columns and f'{col}_b' in merged.columns:
                merged[f'{col}_avg'] = (merged[f'{col}_a'] + merged[f'{col}_b']) / 2
    
    # Save
    merged.to_csv(output_path, index=False)
    print(f"✅ Averaged CSV summary")
    print(f"📄 Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Average multi-provider results")
    
    parser.add_argument(
        "--input-a",
        required=True,
        help="First input file (JSON or CSV)"
    )
    
    parser.add_argument(
        "--input-b",
        required=True,
        help="Second input file (JSON or CSV)"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Output file"
    )
    
    args = parser.parse_args()
    
    try:
        # Determine file type and format
        if args.input_a.endswith('.csv'):
            average_csv_summaries(args.input_a, args.input_b, args.output)
        else:
            # Check if it's a summary file or detailed results file
            with open(args.input_a) as f:
                data = json.load(f)
            
            if 'results_by_model' in data and 'overall_results' in data:
                # New summary format
                average_summary_files(args.input_a, args.input_b, args.output)
            else:
                # Old detailed results format
                average_json_results(args.input_a, args.input_b, args.output)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

