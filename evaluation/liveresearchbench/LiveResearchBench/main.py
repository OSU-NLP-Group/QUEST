#!/usr/bin/env python3
"""
Grade research reports across multiple criteria.

Usage:
    # Single file, single criterion
    python main.py --input reports.json --criteria presentation --provider gemini --model gemini-2.5-pro
    
    # Single file, multiple criteria
    python main.py --input reports.json \
        --criteria presentation,coverage,consistency,citation,depth \
        --provider openai --model gpt-5-mini
    
    # Resume existing run by simply running the same command again
    python main.py --input reports.json \
        --criteria presentation,coverage,consistency,citation,depth \
        --provider openai --model gpt-5-mini

Note: reports.json is the output of the preprocessing script.
"""

import argparse
import yaml
import sys
import logging
from pathlib import Path
from liveresearchbench.batch_evaluator import BatchEvaluator

SUPPORTED_CRITERIA = {
    'presentation', # Presentation & Organization
    'coverage', # Coverage & Comprehensiveness
    'consistency', # Factual & Logical Consistency
    'citation', # Citation Association
    'depth' # Analysis Depth
}

def validate_criteria(criteria_list):
    """
    Validate criteria and return supported/unsupported sets.
    
    Args:
        criteria_list: List of criterion names
        
    Returns:
        Tuple of (supported_criteria, unsupported_criteria)
    """
    criteria_set = set(criteria_list)
    supported = criteria_set & SUPPORTED_CRITERIA
    unsupported = criteria_set - SUPPORTED_CRITERIA
    return supported, unsupported


def main():
    # Import TqdmLoggingHandler from batch_evaluator
    from liveresearchbench.batch_evaluator import TqdmLoggingHandler
    
    # Configure logging with custom handler that works with tqdm
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Clear any existing handlers and add our custom one
    logging.root.handlers = []
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)
    
    parser = argparse.ArgumentParser(description="Grade research reports")
    
    # Mode selection
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: grade multiple JSON files from config"
    )
    
    # Single file mode
    parser.add_argument(
        "--input",
        help="Input JSON file (for single file mode)"
    )
    
    parser.add_argument(
        "--criteria",
        help="Comma-separated criteria: presentation,coverage,consistency,citation,depth"
    )
    
    # Provider settings (required)
    parser.add_argument(
        "--provider",
        required=True,
        choices=['openai', 'gemini'],
        help="AI provider: openai or gemini"
    )
    
    parser.add_argument(
        "--model",
        help="Specific model (e.g., gpt-5-mini, gemini-2.5-pro)"
    )
    
    # Batch mode
    parser.add_argument(
        "--config",
        help="Batch config YAML file (for batch mode)"
    )
    
    parser.add_argument(
        "--run-id",
        help="Run ID for resuming/debugging (optional)"
    )
    
    # Common options
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent API calls"
    )
    
    parser.add_argument(
        "--force-regrade",
        action="store_true",
        help="Force re-grading even if already graded"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set log level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create evaluator
    evaluator = BatchEvaluator(
        provider=args.provider,
        model=args.model,
        output_dir=args.output_dir,
        max_concurrent=args.max_concurrent,
        verbose=args.verbose
    )
    
    try:
        if args.batch:
            # Batch mode
            if not args.config:
                parser.error("--config required for batch mode")
            
            print(f"📋 Loading batch config: {args.config}")
            with open(args.config) as f:
                config = yaml.safe_load(f)
            
            # Validate criteria from config
            config_criteria = config.get('criteria', [])
            supported, unsupported = validate_criteria(config_criteria)
            
            if unsupported:
                print(f"\n⚠️  Warning: Unsupported criteria detected and will be skipped:")
                for criterion in unsupported:
                    print(f"   - {criterion}")
                print(f"\n✅ Supported criteria: {', '.join(supported)}")
                print(f"   Valid options: {', '.join(sorted(SUPPORTED_CRITERIA))}\n")
                
                if not supported:
                    print("❌ Error: No valid criteria to evaluate!")
                    return 1
                
                # Update config with only supported criteria
                config['criteria'] = list(supported)
            
            evaluator.evaluate_batch(
                config=config,
                run_id=args.run_id,
                force_regrade=args.force_regrade
            )
        else:
            # Single file mode
            if not args.input or not args.criteria:
                parser.error("--input and --criteria required for single file mode")
            
            criteria = [c.strip() for c in args.criteria.split(",")]
            
            # Validate criteria
            supported, unsupported = validate_criteria(criteria)
            
            if unsupported:
                print(f"\n⚠️  Warning: Unsupported criteria detected and will be skipped:")
                for criterion in unsupported:
                    print(f"   - {criterion}")
                print(f"\n✅ Supported criteria: {', '.join(supported)}")
                print(f"   Valid options: {', '.join(sorted(SUPPORTED_CRITERIA))}\n")
                
                if not supported:
                    print("❌ Error: No valid criteria to evaluate!")
                    return 1
            
            print(f"📄 Processing single file: {args.input}")
            print(f"🎯 Criteria: {', '.join(supported)}")
            
            evaluator.evaluate_single(
                input_file=args.input,
                criteria=list(supported),
                force_regrade=args.force_regrade
            )
        
        print("\n✅ Grading completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during grading: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
