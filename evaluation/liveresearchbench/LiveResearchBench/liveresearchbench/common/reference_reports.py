"""
Reference report utilities for pairwise depth comparison.

The reference reports (from open-deep-research model) serve as a baseline
for comparing other models' reports in depth analysis.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Default reference reports directory
DEFAULT_REFERENCE_DIR = "data/reference_reports"
REFERENCE_MODEL_NAME = "open-deep-research"


def get_reference_report_path(qid: str, reference_dir: str = None) -> Optional[str]:
    """
    Get the file path for a reference report by query ID.
    
    Args:
        qid: Query ID (with or without 'qid_' prefix)
        reference_dir: Directory containing reference reports (default: data/reference_reports)
        
    Returns:
        Path to reference report file, or None if not found
    """
    if reference_dir is None:
        reference_dir = DEFAULT_REFERENCE_DIR
    
    ref_dir = Path(reference_dir)
    if not ref_dir.exists():
        logger.warning(f"Reference directory does not exist: {ref_dir}")
        return None
    
    # Clean qid (ensure it has qid_ prefix)
    if not qid.startswith('qid_'):
        qid = f'qid_{qid}'
    
    # Look for the report file
    report_file = ref_dir / f"{qid}_report.md"
    
    if report_file.exists():
        return str(report_file)
    
    logger.warning(f"Reference report not found for {qid}")
    return None


def load_reference_report(qid: str, reference_dir: str = None) -> Optional[str]:
    """
    Load reference report content by query ID.
    
    Args:
        qid: Query ID
        reference_dir: Directory containing reference reports
        
    Returns:
        Report content as string, or None if not found
    """
    report_path = get_reference_report_path(qid, reference_dir)
    if not report_path:
        return None
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading reference report from {report_path}: {e}")
        return None


def list_available_reference_qids(reference_dir: str = None) -> Dict[str, str]:
    """
    List all available reference reports.
    
    Args:
        reference_dir: Directory containing reference reports
        
    Returns:
        Dictionary mapping qid to file path
    """
    if reference_dir is None:
        reference_dir = DEFAULT_REFERENCE_DIR
    
    ref_dir = Path(reference_dir)
    if not ref_dir.exists():
        return {}
    
    available = {}
    for report_file in ref_dir.glob("qid_*_report.md"):
        # Extract qid from filename: qid_<qid>_report.md
        qid = report_file.stem.replace('_report', '')
        available[qid] = str(report_file)
    
    return available
