#!/usr/bin/env python3
"""
Real integration test using GPT-5 to grade sample reports.
This test makes actual API calls and validates the grading workflow end-to-end.

Tests all 5 criteria:
1. Presentation & Organization (checklist-based)
2. Factual & Logical Consistency (pointwise)
3. Coverage & Comprehensiveness (checklist-based)
4. Citation Association (pointwise)
5. Analysis Depth (pairwise comparison)

Run with: python tests/test_real_grading.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from liveresearchbench.common.io_utils import (
    load_json,
    load_liveresearchbench_dataset,
    get_question_for_qid,
    get_checklists_for_qid
)
from liveresearchbench.graders.checklist_grader import ChecklistGrader
from liveresearchbench.graders.pointwise_grader import PointwiseGrader
from liveresearchbench.graders.pairwise_grader import PairwiseGrader

# Test configuration
TEST_DATA_PATH = "extracted_reports/reports_20251030_221703.json"
GRADER_MODEL = "gpt-5-mini"
PROVIDER = "openai"
MAX_REPORTS_PER_TEST = 2  # Grade 2 reports per criterion
USE_REALTIME_PLACEHOLDERS = False  # Set to True to use {{current_year}} etc.


def check_api_key() -> bool:
    """Check if OpenAI API key is configured"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-openai-key-here":
        print("❌ Error: OPENAI_API_KEY not configured in .env file")
        print("   Please set up your .env file with a valid API key")
        return False
    return True


def load_report_content(report: Dict[str, Any]) -> str:
    """Load report content from file path"""
    report_file = report.get("report_file_path", "")
    if not report_file or not os.path.exists(report_file):
        return ""
    
    try:
        with open(report_file, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"   ⚠️  Error loading report from {report_file}: {e}")
        return ""


async def test_presentation_grading():
    """Test 1: Presentation & Organization grading (checklist-based)"""
    print("\n" + "="*80)
    print("Test 1: Presentation & Organization Grading")
    print("="*80)
    
    # Load test data
    print(f"Loading test data from {TEST_DATA_PATH}...")
    reports_data = load_json(TEST_DATA_PATH)
    reports = reports_data["reports"][:MAX_REPORTS_PER_TEST]
    print(f"   ✅ Loaded {len(reports)} reports")
    
    # Initialize grader
    print(f"Initializing ChecklistGrader with {GRADER_MODEL}...")
    grader = ChecklistGrader()
    print(f"   ✅ Grader initialized")
    
    # Grade each report
    results = []
    for i, report in enumerate(reports, 1):
        query_id = report.get("query_id", f"report_{i}")
        query = report.get("query", "No query available")
        content = load_report_content(report)
        
        print(f"\nGrading report {i}/{len(reports)} (query_id: {query_id})...")
        print(f"   Query: {query}...")
        print(f"   Report length: {len(content)} chars")
        
        result = await grader.grade_presentation_async(
            query=query,
            report_content=content,
            provider=PROVIDER,
            model=GRADER_MODEL
        )
        
        results.append(result)
        
        # Display results
        summary = result.get("summary", {})
        passed = summary.get("passed_count", 0)
        total = summary.get("total_criteria", 10)
        pass_rate = summary.get("average_pass_rate", 0)
        
        print(f"   ✅ Grading complete:")
        print(f"      Passed: {passed}/{total} questions ({pass_rate:.1f}%)")
        
        # Show a few examples
        evaluations = result.get("evaluations", {})
        for q_num in list(evaluations.keys())[:3]:
            eval_data = evaluations[q_num]
            score = eval_data.get("score", 0)
            status = "✅" if score == 1 else "❌"
            print(f"      {status} {q_num}: Score={score}")
    
    print(f"\n✅ Test 1 Complete: Graded {len(results)} reports for Presentation")
    return results


async def test_consistency_grading():
    """Test 2: Factual & Logical Consistency grading (pointwise)"""
    print("\n" + "="*80)
    print("Test 2: Factual & Logical Consistency Grading")
    print("="*80)
    
    # Load test data
    print(f"Loading test data from {TEST_DATA_PATH}...")
    reports_data = load_json(TEST_DATA_PATH)
    reports = reports_data["reports"][:MAX_REPORTS_PER_TEST]
    print(f"   ✅ Loaded {len(reports)} reports")
    
    # Initialize grader
    print(f"Initializing PointwiseGrader with {GRADER_MODEL}...")
    grader = PointwiseGrader()
    print(f"   ✅ Grader initialized")
    
    # Grade each report
    results = []
    for i, report in enumerate(reports, 1):
        query_id = report.get("query_id", f"report_{i}")
        query = report.get("query", "No query available")
        content = load_report_content(report)
        
        print(f"\nGrading report {i}/{len(reports)} (query_id: {query_id})...")
        print(f"   Query: {query}...")
        print(f"   Report length: {len(content)} chars")
        
        result = await grader.grade_consistency_async(
            query=query,
            report_content=content,
            provider=PROVIDER,
            model=GRADER_MODEL
        )
        
        results.append(result)
        
        # Display results
        score = result.get("score", 0)
        total_issues = result.get("total_issues", 0)
        
        print(f"   ✅ Grading complete:")
        print(f"      Score: {score}/100")
        print(f"      Issues found: {total_issues}")
        
        # Show specific issues if any
        specific = result.get("specific_issues", [])
        if specific:
            print(f"      Specific issues:")
            for issue in specific:  
                print(f"         - {issue}...")
    
    print(f"\n✅ Test 2 Complete: Graded {len(results)} reports for Consistency")
    return results


async def test_coverage_grading(benchmark_data: dict):
    """Test 3: Coverage & Comprehensiveness grading (checklist-based)"""
    print("\n" + "="*80)
    print("Test 3: Coverage & Comprehensiveness Grading")
    print("="*80)
    
    # Load test data
    print(f"Loading test data from {TEST_DATA_PATH}...")
    reports_data = load_json(TEST_DATA_PATH)
    reports = reports_data["reports"][:MAX_REPORTS_PER_TEST]
    print(f"   ✅ Loaded {len(reports)} reports")
    
    # Initialize grader
    print(f"Initializing ChecklistGrader with {GRADER_MODEL}...")
    grader = ChecklistGrader()
    print(f"   ✅ Grader initialized")
    
    # Grade each report
    results = []
    for i, report in enumerate(reports, 1):
        query_id = report.get("query_id", f"report_{i}")
        content = load_report_content(report)
        
        # Get question and checklist from benchmark dataset
        query = get_question_for_qid(benchmark_data, query_id)
        if not query:
            print(f"\n⚠️  No question found for query_id: {query_id}, skipping...")
            continue
        
        checklists = get_checklists_for_qid(benchmark_data, query_id)
        if not checklists:
            print(f"\n⚠️  No checklist found for query_id: {query_id}, skipping...")
            continue
        
        print(f"\nGrading report {i}/{len(reports)} (query_id: {query_id})...")
        print(f"   Query: {query}...")
        print(f"   Report length: {len(content)} chars")
        print(f"   Checklist items: {len(checklists)}")
        
        result = await grader.grade_coverage_async(
            query=query,
            report_content=content,
            checklists=checklists,
            provider=PROVIDER,
            model=GRADER_MODEL
        )
        
        results.append(result)
        
        # Display results
        summary = result.get("summary", {})
        addressed = summary.get("addressed_count", 0)
        total = summary.get("total_checklists", len(checklists))
        percentage = summary.get("percentage_addressed", 0)
        
        print(f"   ✅ Grading complete:")
        print(f"      Addressed: {addressed}/{total} items ({percentage:.1f}%)")
        
        # Show a few examples
        evaluations = result.get("evaluations", {})
        for checklist_id in list(evaluations.keys())[:3]:
            eval_data = evaluations[checklist_id]
            score = eval_data.get("score", 0)
            status = "✅" if score == 1 else "❌"
            print(f"      {status} {checklist_id}: Score={score}")
    
    print(f"\n✅ Test 3 Complete: Graded {len(results)} reports for Coverage")
    return results


async def test_citation_grading():
    """Test 4: Citation Association grading (pointwise)"""
    print("\n" + "="*80)
    print("Test 4: Citation Association Grading")
    print("="*80)
    
    # Load test data
    print(f"Loading test data from {TEST_DATA_PATH}...")
    reports_data = load_json(TEST_DATA_PATH)
    reports = reports_data["reports"][:MAX_REPORTS_PER_TEST]
    print(f"   ✅ Loaded {len(reports)} reports")
    
    # Initialize grader
    print(f"Initializing PointwiseGrader with {GRADER_MODEL}...")
    grader = PointwiseGrader()
    print(f"   ✅ Grader initialized")
    
    # Grade each report
    results = []
    for i, report in enumerate(reports, 1):
        query_id = report.get("query_id", f"report_{i}")
        query = report.get("query", "No query available")
        content = load_report_content(report)
        
        print(f"\nGrading report {i}/{len(reports)} (query_id: {query_id})...")
        print(f"   Query: {query}...")
        print(f"   Report length: {len(content)} chars")
        
        result = await grader.grade_citation_async(
            query=query,
            report_content=content,
            provider=PROVIDER,
            model=GRADER_MODEL
        )
        
        results.append(result)
        
        # Display results
        score = result.get("score", 0)
        total_issues = result.get("total_issues", 0)
        
        print(f"   ✅ Grading complete:")
        print(f"      Score: {score}/100")
        print(f"      Issues found: {total_issues}")
        
        # Show specific issues if any
        specific = result.get("specific_issues", [])
        if specific:
            print(f"      Specific issues:")
            for issue in specific: 
                print(f"         - {issue}...")
    
    print(f"\n✅ Test 4 Complete: Graded {len(results)} reports for Citation")
    return results


async def test_depth_grading():
    """Test 5: Analysis Depth grading (pairwise comparison vs reference)"""
    print("\n" + "="*80)
    print("Test 5: Analysis Depth Grading (Pairwise vs Reference)")
    print("="*80)
    
    # Load test data
    print(f"Loading test data from {TEST_DATA_PATH}...")
    reports_data = load_json(TEST_DATA_PATH)
    
    if len(reports_data["reports"]) < 1:
        print(f"⚠️  Warning: Need at least 1 report for comparison")
        return []
    
    report = reports_data["reports"][0]  # Take first report
    print(f"   ✅ Loaded {len(reports_data['reports'])} reports")
    
    # Load reference report
    from liveresearchbench.common.reference_reports import load_reference_report, REFERENCE_MODEL_NAME
    
    query_id = report.get("query_id", "")
    print(f"\n📚 Loading reference report for query_id: {query_id}")
    reference_content = load_reference_report(query_id)
    
    if not reference_content:
        print(f"   ⚠️  No reference report found for {query_id}, skipping...")
        return []
    
    print(f"   ✅ Reference report loaded ({len(reference_content)} chars)")
    
    # Initialize grader
    print(f"Initializing PairwiseGrader with {GRADER_MODEL}...")
    grader = PairwiseGrader()
    print(f"   ✅ Grader initialized")
    
    # Get report details
    query = report.get("query", "No query available")
    content = load_report_content(report)
    
    print(f"\n📝 Comparing against reference...")
    print(f"   Model: {report.get('model_name')}")
    print(f"   Query ID: {query_id}")
    print(f"   Report length: {len(content)} chars")
    print(f"   Reference model: {REFERENCE_MODEL_NAME}")
    print(f"   Reference length: {len(reference_content)} chars")
    print(f"   Query: {query[:100]}...")
    
    # Position-swap averaging is enabled by default to mitigate position bias
    # Report A = model being evaluated, Report B = reference
    result = await grader.compare_depth_async(
        query=query,
        report_a=content,
        report_b=reference_content,
        provider=PROVIDER,
        model=GRADER_MODEL,
        swap_positions=True  # Position-swap averaging (default)
    )
    
    # Display results with actual model names
    winner_1 = result.get("winner_comparison_1", "unknown")
    winner_2 = result.get("winner_comparison_2", "unknown")
    justification = result.get("justification", "")
    
    # Get model names
    evaluated_model_name = report.get("model_name", "unknown")
    
    print(f"   ✅ Position-swap comparison complete:")
    
    # Translate semantic winners to actual model names for display
    def translate_winner(winner):
        if winner == 'evaluated_model':
            return evaluated_model_name
        elif winner == 'reference_model':
            return REFERENCE_MODEL_NAME
        else:
            return 'tie'
    
    winner_1_display = translate_winner(winner_1)
    winner_2_display = translate_winner(winner_2)
    
    print(f"      Comparison 1: {winner_1_display} wins")
    print(f"      Comparison 2: {winner_2_display} wins")
    print(f"")
    
    # Aggregate winner count
    wins_evaluated = sum(1 for w in [winner_1, winner_2] if w == 'evaluated_model')
    wins_reference = sum(1 for w in [winner_1, winner_2] if w == 'reference_model')
    ties = sum(1 for w in [winner_1, winner_2] if w == 'tie')
    
    print(f"      📊 Winner count: {evaluated_model_name} wins {wins_evaluated}/2, "
          f"{REFERENCE_MODEL_NAME} wins {wins_reference}/2, ties {ties}/2")
    print(f"")
    
    # Display scores from raw comparisons (for debugging/informative purposes)
    raw_1 = result.get("raw_comparison_1", {})
    raw_2 = result.get("raw_comparison_2", {})
    
    if raw_1 and raw_2:
        scores_1 = raw_1.get('scores', {})
        scores_2 = raw_2.get('scores', {})
        
        # Average scores across both positions
        # In comparison 1: A=evaluated, B=reference
        # In comparison 2: A=reference, B=evaluated (swapped)
        avg_evaluated = {}
        avg_reference = {}
        
        dims = ['granularity', 'insight', 'critique', 'evidence', 'density']
        for dim in dims:
            score_eval_pos1 = scores_1.get('A', {}).get(dim, 0)  # Evaluated in position A
            score_eval_pos2 = scores_2.get('B', {}).get(dim, 0)  # Evaluated in position B
            avg_evaluated[dim] = (score_eval_pos1 + score_eval_pos2) / 2
            
            score_ref_pos1 = scores_1.get('B', {}).get(dim, 0)  # Reference in position B
            score_ref_pos2 = scores_2.get('A', {}).get(dim, 0)  # Reference in position A
            avg_reference[dim] = (score_ref_pos1 + score_ref_pos2) / 2
        
        avg_evaluated['total'] = sum(avg_evaluated[d] for d in dims)
        avg_reference['total'] = sum(avg_reference[d] for d in dims)
        
        print(f"      📊 Scores (averaged across positions):")
        print(f"         {evaluated_model_name}: {avg_evaluated['total']:.1f}/25")
        print(f"            Granularity: {avg_evaluated['granularity']:.1f}/5 | "
              f"Insight: {avg_evaluated['insight']:.1f}/5 | "
              f"Critique: {avg_evaluated['critique']:.1f}/5")
        print(f"            Evidence: {avg_evaluated['evidence']:.1f}/5 | "
              f"Density: {avg_evaluated['density']:.1f}/5")
        print(f"")
        print(f"         {REFERENCE_MODEL_NAME}: {avg_reference['total']:.1f}/25")
        print(f"            Granularity: {avg_reference['granularity']:.1f}/5 | "
              f"Insight: {avg_reference['insight']:.1f}/5 | "
              f"Critique: {avg_reference['critique']:.1f}/5")
        print(f"            Evidence: {avg_reference['evidence']:.1f}/5 | "
              f"Density: {avg_reference['density']:.1f}/5")
        print(f"")
    
    print(f"      💬 Justification: {justification[:200]}...")
    
    print(f"\n✅ Test 5 Complete: Pairwise comparison for Depth")
    return [result]


async def run_all_tests():
    """Run all real grading tests"""
    print("\n" + "🧪"*40)
    print("Demo: Grading with GPT-5")
    print("🧪"*40)
    
    # Check API key
    if not check_api_key():
        return False
    
    # Check test data exists
    if not os.path.exists(TEST_DATA_PATH):
        print(f"❌ Error: Test data not found at {TEST_DATA_PATH}")
        print("   Run the data preparation step first")
        return False
    
    # Load LiveResearchBench dataset
    print(f"\n📥 Loading LiveResearchBench dataset from HuggingFace...")
    print(f"   Dataset: Salesforce/LiveResearchBench")
    print(f"   Mode: {'Realtime (with placeholder replacement)' if USE_REALTIME_PLACEHOLDERS else 'Static (no placeholders)'}")
    benchmark_data = load_liveresearchbench_dataset(use_realtime=USE_REALTIME_PLACEHOLDERS)
    if not benchmark_data:
        print(f"❌ Error: Failed to load benchmark dataset")
        return False
    print(f"   ✅ Loaded {len(benchmark_data)} benchmark questions")
    
    all_results = {}
    
    try:
        # Run all 5 criteria tests
        all_results["presentation"] = await test_presentation_grading()
        all_results["consistency"] = await test_consistency_grading()
        all_results["coverage"] = await test_coverage_grading(benchmark_data)
        all_results["citation"] = await test_citation_grading()
        all_results["depth"] = await test_depth_grading()
        
        # Summary
        print("\n" + "="*80)
        print("📊 Test Summary")
        print("="*80)
        print(f"✅ All tests completed successfully!")
        print(f"   1. Presentation: {len(all_results['presentation'])} reports graded")
        print(f"   2. Consistency: {len(all_results['consistency'])} reports graded")
        print(f"   3. Coverage: {len(all_results['coverage'])} reports graded")
        print(f"   4. Citation: {len(all_results['citation'])} reports graded")
        print(f"   5. Depth: {len(all_results['depth'])} pairwise comparisons")
        
        total_calls = (len(all_results['presentation']) + 
                       len(all_results['consistency']) + 
                       len(all_results['coverage']) + 
                       len(all_results['citation']) + 
                       len(all_results['depth']))
        print(f"\nTotal API calls made: ~{total_calls}")
        
        # Save results
        output_dir = Path("results/test_real_grading")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "grading_results.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
