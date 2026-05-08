"""I/O utilities for loading, saving, and preprocessing reports."""

import os
import json
import logging
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HuggingFace dataset configuration.
# Overridable via LRB_HF_DATASET_NAME env var.
HF_DATASET_NAME = os.environ.get(
    "LRB_HF_DATASET_NAME", "Salesforce/LiveResearchBenchFull"
)
HF_SUBSET = "question_with_checklist"
HF_SPLIT = "test"


def replace_placeholders(text: str, use_realtime: bool = False) -> str:
    """
    Replace temporal placeholders in text.
    
    Args:
        text: Text with potential placeholders
        use_realtime: If True, replace with current values; if False, return as-is
        
    Returns:
        Text with placeholders replaced (if use_realtime=True)
    """
    if not use_realtime or not text:
        return text
    
    now = datetime.now()
    current_year = now.year
    last_year = current_year - 1
    current_date = now.strftime("%B %d, %Y")
    
    replacements = {
        "{{current_year}}": str(current_year),
        "{{last_year}}": str(last_year),
        "{{date}}": current_date,
    }
    
    result = text
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)
    
    return result


def load_liveresearchbench_dataset(
    dataset_name: str = HF_DATASET_NAME,
    subset: str = HF_SUBSET,
    split: str = HF_SPLIT,
    use_realtime: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Load LiveResearchBench dataset from HuggingFace.
    
    Args:
        dataset_name: HuggingFace dataset name
        subset: Dataset subset name (default: question_with_checklist)
        split: Dataset split name (default: test)
        use_realtime: If True, use questions/checklists with placeholders filled;
                     if False, use no_placeholder versions
        
    Returns:
        Dictionary mapping qid to data:
        {
            'qid': {
                'question': 'research query',
                'checklists': ['checklist item 1', 'checklist item 2', ...]
            }
        }
    """
    try:
        logger.info(f"Loading dataset {dataset_name}/{subset}/{split} from HuggingFace...")
        dataset = load_dataset(dataset_name, subset, split=split)
        
        # Group by qid (one row per checklist item, so need to aggregate)
        from collections import defaultdict
        grouped_data = defaultdict(lambda: {'question': '', 'checklists': []})
        
        for row in dataset:
            qid = row['qid']
            
            # Choose version based on use_realtime flag
            if use_realtime:
                question = replace_placeholders(row.get('question', ''), use_realtime=True)
                checklist_item = replace_placeholders(row.get('checklist', ''), use_realtime=True)
            else:
                question = row.get('question_no_placeholder', '')
                checklist_item = row.get('checklist_no_placeholder', '')
            
            # Store question (same for all rows with same qid)
            if not grouped_data[qid]['question']:
                grouped_data[qid]['question'] = question
            
            # Append this checklist item
            if checklist_item:
                grouped_data[qid]['checklists'].append(checklist_item)
        
        # Convert to final format
        benchmark_data = {}
        for qid, data in grouped_data.items():
            benchmark_data[qid] = {
                'qid': qid,
                'question': data['question'],
                'checklists': data['checklists']
            }
        
        logger.info(f"Loaded {len(benchmark_data)} unique questions from HuggingFace")
        return benchmark_data
        
    except Exception as e:
        logger.error(f"Error loading dataset from HuggingFace: {e}")
        return {}


def get_question_for_qid(benchmark_data: Dict, qid: str) -> Optional[str]:
    """Get question for a specific query ID."""
    if qid in benchmark_data:
        return benchmark_data[qid].get('question')
    
    # Try with qid_ prefix removed
    if qid.startswith('qid_'):
        clean_qid = qid[4:]
        if clean_qid in benchmark_data:
            return benchmark_data[clean_qid].get('question')
    
    # Try adding qid_ prefix
    prefixed_qid = f"qid_{qid}"
    if prefixed_qid in benchmark_data:
        return benchmark_data[prefixed_qid].get('question')
    
    return None


def get_checklists_for_qid(benchmark_data: Dict, qid: str) -> Optional[List[str]]:
    """Get checklist items for a specific query ID."""
    if qid in benchmark_data:
        return benchmark_data[qid].get('checklists', [])
    
    # Try with qid_ prefix removed
    if qid.startswith('qid_'):
        clean_qid = qid[4:]
        if clean_qid in benchmark_data:
            return benchmark_data[clean_qid].get('checklists', [])
    
    # Try adding qid_ prefix
    prefixed_qid = f"qid_{qid}"
    if prefixed_qid in benchmark_data:
        return benchmark_data[prefixed_qid].get('checklists', [])
    
    return None

# Model name mapping
MODEL_NAME_MAPPING = {
    "openai__gpt-4.1": "gpt-4.1",
    "openai__gpt-4o-2024-08-06": "gpt-4o",
    "gemini-2.5-pro": "gemini-2.5-pro"
}


def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        raise


def save_json(data: Dict[str, Any], file_path: str, indent: int = 2) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    logger.info(f"Saved JSON to: {file_path}")


def load_report_content(report_file_path: str) -> str:
    """Load content from a report markdown file."""
    try:
        if not os.path.exists(report_file_path):
            logger.warning(f"Report file not found: {report_file_path}")
            return ""
        
        with open(report_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading report file {report_file_path}: {e}")
        return ""


def load_query_mapping(use_realtime: bool = False) -> Dict[str, str]:
    """
    Load query mapping from HuggingFace.
    
    Args:
        use_realtime: If True, replace placeholders with current values
        
    Returns:
        Dictionary mapping qid to question text
    """
    # Load from HuggingFace
    benchmark_data = load_liveresearchbench_dataset(use_realtime=use_realtime)
    query_mapping = {qid: data['question'] for qid, data in benchmark_data.items()}
    logger.info(f"Loaded {len(query_mapping)} query mappings from HuggingFace")
    return query_mapping


def parse_report_metadata(content: str) -> Dict[str, Any]:
    """Parse metadata from the markdown report content."""
    metadata = {}
    
    parts = content.split('---', 2)
    if len(parts) < 3:
        logging.warning("Could not find metadata section separator")
        return metadata
    
    metadata_section = parts[1].strip()
    report_content = parts[2].strip()
    
    current_section = None
    
    for line in metadata_section.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        if line.endswith(':') and not line.startswith('-'):
            if line.startswith('**') and line.endswith(':**'):
                current_section = line[2:-3].strip()
            else:
                current_section = line[:-1].strip()
            metadata[current_section] = {}
            continue
        
        if ':' in line:
            if line.startswith('-'):
                key, value = line[1:].split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if current_section and isinstance(metadata.get(current_section), dict):
                    metadata[current_section][key] = value
                else:
                    metadata[key] = value
            elif line.startswith('**') and ':**' in line:
                key_value_split = line.split(':**', 1)
                key_part = key_value_split[0].strip('*').strip()
                if len(key_value_split) > 1:
                    value = key_value_split[1].strip()
                    metadata[key_part] = value
            else:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                metadata[key] = value
    
    metadata['report_content'] = report_content
    
    return metadata


def extract_and_save_report(file_path: str, report_content: str, query_id: str, copy_entire_file: bool = False) -> str:
    """Extract report content and save to separate file."""
    base_path = Path(file_path).parent
    new_filename = f"report_{query_id}_sd0_only_report.md"
    new_file_path = base_path / new_filename
    
    try:
        if copy_entire_file:
            with open(file_path, 'r', encoding='utf-8') as source_f:
                content = source_f.read()
            with open(new_file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            with open(new_file_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
        return str(new_file_path)
    except Exception as e:
        logger.error(f"Error saving extracted report to {new_file_path}: {e}")
        return ""


def extract_query_id(filename: str) -> Optional[str]:
    """
    Extract query ID from report filename.
    
    Expected pattern: qid_<qid>_report.md
    Returns just the ID part without 'qid_' prefix.
    """
    match = re.search(r'qid_(.+)_report\.md$', filename)
    return match.group(1) if match else None


def find_report_files(base_path: str, model_names: List[str] = None) -> List[Dict[str, Any]]:
    """
    Find all report files in the flat directory structure.
    
    Expected structure:
        base_path/
            model_name_1/
                qid_<qid>_report.md
            model_name_2/
                qid_<qid>_report.md
    """
    report_files = []
    base_path = Path(base_path)
    
    if not base_path.exists():
        logger.error(f"Base path does not exist: {base_path}")
        return report_files
    
    # Get list of model directories
    if model_names is None:
        model_dirs = [d for d in base_path.iterdir() if d.is_dir()]
        logger.info(f"Auto-discovered {len(model_dirs)} model directories")
    else:
        model_dirs = [base_path / model_name for model_name in model_names]
        logger.info(f"Processing {len(model_names)} specified models")
    
    for model_dir in model_dirs:
        if not model_dir.exists() or not model_dir.is_dir():
            logger.warning(f"Model directory does not exist: {model_dir}")
            continue
        
        model_name = model_dir.name
        logger.info(f"  📁 {model_name}")
        
        report_count = 0
        for report_file in model_dir.glob("qid_*_report.md"):
            query_id = extract_query_id(report_file.name)
            if not query_id:
                logger.warning(f"    ⚠️  Could not extract query ID from: {report_file.name}")
                continue
            
            report_files.append({
                'file_path': str(report_file),
                'model_name': model_name,
                'query_id': query_id
            })
            report_count += 1
        
        logger.info(f"    ✅ Found {report_count} reports")
    
    logger.info(f"\n📊 Total: {len(report_files)} reports across {len(model_dirs)} models")
    return report_files


def process_report_file(file_info: Dict[str, Any], query_mapping: Dict[str, str], base_path: str = "") -> Optional[Dict[str, Any]]:
    """Process a single report file and extract all data."""
    try:
        with open(file_info['file_path'], 'r', encoding='utf-8') as f:
            content = f.read()
        
        needs_metadata_extraction = ("deerflow" in base_path.lower() or "open_deep_research" in base_path.lower())
        
        original_model = file_info['model_name']
        mapped_model = MODEL_NAME_MAPPING.get(original_model, original_model)
        
        query_id = file_info['query_id']
        
        query_text = query_mapping.get(query_id, '')
        if not query_text and query_id.startswith('qid_'):
            clean_query_id = query_id[4:]
            query_text = query_mapping.get(clean_query_id, '')
        
        if not query_text:
            logger.warning(f"No query found for {query_id} in benchmark dataset")
        
        result = {
            'model_name': file_info['model_name'],
            'query_id': query_id,
            'query': query_text,
            'report_file_path': file_info['file_path']
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing {file_info['file_path']}: {e}")
        return None


def remove_duplicates(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate reports based on model_name and query_id."""
    seen = set()
    unique_reports = []
    
    for report in reports:
        identifier = (report['model_name'], report['query_id'])
        
        if identifier not in seen:
            seen.add(identifier)
            unique_reports.append(report)
    
    if len(reports) != len(unique_reports):
        logger.info(f"Removed {len(reports) - len(unique_reports)} duplicates")
    
    return unique_reports


def preprocess_reports(base_path: str, model_names: List[str] = None, output_dir: str = "extracted_reports",
                      verbose: bool = False, use_realtime: bool = False,
                      questions_override_file: str = None) -> str:
    """
    Preprocess reports from the simplified flat directory structure.
    
    Expected directory structure:
        base_path/
            model_name_1/
                qid_<qid>_report.md
            model_name_2/
                qid_<qid>_report.md
            ...
    
    Args:
        base_path: Base directory containing model subdirectories
        model_names: Optional list of specific model names to process.
                    If None, processes all subdirectories.
        output_dir: Output directory for JSON file
        verbose: Enable verbose logging
        use_realtime: If True, replace placeholders with current values
    
    Returns:
        Path to the generated JSON file
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("📦 LiveResearchBench - Report Preprocessing")
    logger.info("=" * 80)
    
    # Load query mappings from HuggingFace
    logger.info("\n🌐 Loading query mappings from HuggingFace...")
    query_mapping = load_query_mapping(use_realtime=use_realtime)

    if not query_mapping:
        raise ValueError("Failed to load query mappings from HuggingFace!")

    logger.info(f"✅ Loaded {len(query_mapping)} queries")

    # Optionally override question text from a local jsonl (keeps infer/eval
    # question text identical even when eval runs on a later day than infer).
    # jsonl schema: {"id" or "qid": str, "question": str, ...}
    if questions_override_file:
        logger.info(f"\n🔒 Overriding question text from local file: {questions_override_file}")
        if not os.path.isfile(questions_override_file):
            raise ValueError(f"questions_override_file not found: {questions_override_file}")
        overridden = 0
        new_qids = 0
        with open(questions_override_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                qid = row.get("id") or row.get("qid")
                question = row.get("question")
                if not qid or not question:
                    continue
                if qid in query_mapping:
                    overridden += 1
                else:
                    new_qids += 1
                query_mapping[qid] = question
        logger.info(f"   Overrode {overridden} existing qid question(s); "
                    f"added {new_qids} new qid(s) from file")
    
    # Find all report files
    logger.info(f"\n📂 Scanning directory: {base_path}")
    if model_names:
        logger.info(f"🎯 Target models: {', '.join(model_names)}")
    else:
        logger.info("🎯 Processing all models in directory")
    
    report_files = find_report_files(base_path, model_names)
    
    if not report_files:
        raise ValueError("No report files found!")
    
    logger.info(f"Processing {len(report_files)} report files...")
    reports = []
    
    for i, file_info in enumerate(report_files, 1):
        if i % 10 == 0:
            logger.info(f"  Processed {i}/{len(report_files)} files...")
        
        result = process_report_file(file_info, query_mapping, base_path)
        if result:
            reports.append(result)
    
    logger.info(f"Successfully processed {len(reports)}/{len(report_files)} files")
    
    reports = remove_duplicates(reports)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = os.path.join(output_dir, f"reports_{timestamp}.json")
    
    output_data = {
        'metadata': {
            'timestamp': timestamp,
            'total_reports': len(reports),
            'total_models': len(set(r['model_name'] for r in reports)),
            'base_path': base_path,
            'use_realtime': use_realtime
        },
        'reports': reports
    }
    
    save_json(output_data, json_file)
    
    logger.info(f"✅ Preprocessing completed successfully!")
    logger.info(f"📄 JSON file: {json_file}")
    
    return json_file

