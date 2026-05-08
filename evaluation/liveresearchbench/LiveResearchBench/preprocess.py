#!/usr/bin/env python3
"""
Preprocess research reports from MD files to JSON format.

Usage:
    # Preprocess single experiment
    python preprocess.py exp_name1 exp_name2 --output-dir extracted_reports/
    
    # With custom base path
    python preprocess.py exp_name --base-path /path/to/reports/ --output-dir out/
"""

"""
Preprocess research reports from directory structure to JSON.

Expected directory structure:
    base_path/
        model_name_1/
            qid_<qid>_report.md
        model_name_2/
            qid_<qid>_report.md
        ...

Example usage:
    # Process all models in directory
    python preprocess.py /path/to/model_outputs
    
    # Process specific models only
    python preprocess.py /path/to/model_outputs -m gpt-5-search gemini-pro
    
    # With custom output directory
    python preprocess.py /path/to/model_outputs -o my_reports
"""

import argparse
import sys
from liveresearchbench.common.io_utils import preprocess_reports


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess research reports from directory structure to JSON",
        epilog="""
Expected directory structure:
  base_path/
    model_name_1/
      qid_<qid>_report.md
    model_name_2/
      qid_<qid>_report.md
    ...
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "base_path",
        help="Base directory containing model subdirectories (each with qid_*_report.md files)"
    )
    
    parser.add_argument(
        "-m", "--models",
        nargs="+",
        help="Specific model names to process (default: process all subdirectories)"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        default="extracted_reports",
        help="Output directory for JSON file (default: extracted_reports)"
    )
    
    parser.add_argument(
        "--use-realtime",
        action="store_true",
        help="Replace temporal placeholders ({{current_year}}, etc.) with current values"
    )

    parser.add_argument(
        "--questions-file",
        default=None,
        help="Optional local jsonl (schema: id/qid + question) to override the HF "
             "question text. Keeps infer/eval question identical when eval runs on a "
             "later day than infer. Checklist still comes from HF (method A)."
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    try:
        json_file = preprocess_reports(
            base_path=args.base_path,
            model_names=args.models,
            output_dir=args.output_dir,
            verbose=args.verbose,
            use_realtime=args.use_realtime,
            questions_override_file=args.questions_file
        )
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

