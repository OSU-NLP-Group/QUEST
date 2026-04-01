"""Keyword detection for identifying problematic cached content."""

from __future__ import annotations
import re
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of keyword detection on text content."""
    has_issues: bool
    matched_keywords: List[str]
    matched_patterns: List[str]
    severity: str  # "definite" or "possible"
    
    @property
    def issue_count(self) -> int:
        return len(self.matched_keywords) + len(self.matched_patterns)


class KeywordDetector:
    """Keyword detection with ergonomic config and two severity levels."""
    
    # Definite problem keywords (serious issues)
    DEFAULT_DEFINITE = [
        "this url could not be loaded (navigation error)",
        "robot or human",
        "access denied",
        "403 forbidden", 
        "please complete the verification",
        "cloudflare",
        "blocked",
        "unusual activity",
        "rate limit",
        "verification required",
        "you have been blocked",
        "suspicious activity",
        "security check failed",
        "Spotify is unavailable on this browser.",
        "该网页无法正常运作",
    ]
    
    # Possible problem keywords
    DEFAULT_POSSIBLE = [
        "404 not found",
        "500 internal server error", 
        "site can't be reached",
        "connection timeout",
        "captcha",
        "security check",
        "please verify",
        "anti-bot"
    ]
    
    # Regex patterns for common issues with levels
    DEFAULT_PATTERNS = [
        (r"robot\s+or\s+human", "Robot verification check", "definite"),
        (r"verify\s+you\s+are\s+(not\s+)?a\s+(robot|bot)", "Bot verification", "definite"),
        (r"unusual\s+(traffic|activity)", "Unusual traffic detection", "definite"),
        (r"rate\s+limit|too\s+many\s+requests", "Rate limiting", "possible"),
        (r"access\s+(denied|blocked|restricted)", "Access restriction", "definite"),
        (r"cloudflare|cf-", "Cloudflare protection", "definite"),
        (r"please\s+complete\s+the\s+(verification|captcha)", "Verification required", "definite"),
        (r"this\s+site\s+can't\s+be\s+reached", "Connection failure", "possible")
    ]
    
    def __init__(self, config_path: Path = None):
        self.config_path = config_path
        self.definite_keywords: Set[str] = set()
        self.possible_keywords: Set[str] = set()
        self.patterns: List[Tuple[str, str, str]] = []  # (pattern, description, level)
        
        # Default config path if none provided
        if not self.config_path:
            self.config_path = Path(__file__).resolve().parents[1] / "resources" / "keywords.json"

        self._load_default_rules()
        if self.config_path and self.config_path.exists():
            self._load_config()
    
    def _load_default_rules(self):
        """Load default detection rules."""
        self.definite_keywords.update([k.lower() for k in self.DEFAULT_DEFINITE])
        self.possible_keywords.update([k.lower() for k in self.DEFAULT_POSSIBLE])
        self.patterns.extend(self.DEFAULT_PATTERNS)
        
        logger.debug(f"Loaded default keywords and {len(self.patterns)} patterns")
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Load custom keywords (two levels)
            self.definite_keywords.update([k.lower() for k in config.get('definite', [])])
            self.possible_keywords.update([k.lower() for k in config.get('possible', [])])

            # Load custom patterns
            custom_patterns = config.get('patterns', [])
            for pattern_config in custom_patterns:
                if isinstance(pattern_config, dict) and 'pattern' in pattern_config:
                    pattern = pattern_config['pattern']
                    description = pattern_config.get('description', pattern)
                    level = pattern_config.get('level', 'possible')
                    self.patterns.append((pattern, description, level))
            
            logger.info(f"Loaded custom configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
    
    def save_config(self):
        """Save current configuration to file."""
        if not self.config_path:
            return False
        
        try:
            default_patterns = {(p, d, l) for p, d, l in self.DEFAULT_PATTERNS}
            config = {
                'definite': sorted(list(self.definite_keywords - set([k.lower() for k in self.DEFAULT_DEFINITE]))),
                'possible': sorted(list(self.possible_keywords - set([k.lower() for k in self.DEFAULT_POSSIBLE]))),
                'patterns': [
                    {'pattern': pattern, 'description': desc, 'level': lvl}
                    for pattern, desc, lvl in self.patterns
                    if (pattern, desc, lvl) not in default_patterns
                ]
            }
            
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved configuration to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save config to {self.config_path}: {e}")
            return False
    
    def detect_issues(self, text: str) -> DetectionResult:
        """Detect issues in text content."""
        if text is None or not text.strip():
            # Empty or whitespace-only content is a definite issue
            return DetectionResult(True, ["empty content"], [], "definite")
        
        text_lower = text.lower()
        matched_keywords = []
        matched_patterns = []
        level: str | None = None
        
        # Definite keywords first
        for keyword in self.definite_keywords:
            if keyword in text_lower:
                matched_keywords.append(keyword)
                level = "definite"
        
        # Possible keywords only if no definite yet
        if level != "definite":
            for keyword in self.possible_keywords:
                if keyword in text_lower:
                    matched_keywords.append(keyword)
                    if level is None:
                        level = "possible"
        
        # Check regex patterns
        for pattern, description, pat_level in self.patterns:
            try:
                if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                    matched_patterns.append(description)
                    # escalate to definite if any pattern is definite
                    if pat_level == "definite":
                        level = "definite"
                    elif level is None:
                        level = "possible"
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
        
        has_issues = bool(matched_keywords or matched_patterns)
        return DetectionResult(has_issues, matched_keywords, matched_patterns, level or "possible")
    
    def add_keyword(self, keyword: str, priority: str = "possible") -> bool:
        """Add a new keyword with specified priority."""
        if not keyword or not keyword.strip():
            return False
        
        keyword = keyword.strip().lower()
        
        # Remove existing
        self.definite_keywords.discard(keyword)
        self.possible_keywords.discard(keyword)
        # Add
        if priority == "definite":
            self.definite_keywords.add(keyword)
        else:
            self.possible_keywords.add(keyword)
        
        logger.debug(f"Added keyword '{keyword}' with {priority} priority")
        return True
    
    def remove_keyword(self, keyword: str) -> bool:
        """Remove a keyword from all priority levels."""
        keyword = keyword.strip().lower()
        removed = False
        
        if keyword in self.definite_keywords:
            self.definite_keywords.remove(keyword)
            removed = True
        if keyword in self.possible_keywords:
            self.possible_keywords.remove(keyword)
            removed = True
        
        if removed:
            logger.debug(f"Removed keyword '{keyword}'")
        
        return removed
    
    def add_pattern(self, pattern: str, description: str = None) -> bool:
        """Add a new regex pattern."""
        try:
            # Test if pattern is valid
            re.compile(pattern)
            
            if not description:
                description = pattern
            
            # Remove existing pattern if it exists
            self.patterns = [(p, d, l) for p, d, l in self.patterns if p != pattern]
            
            # Add new pattern
            self.patterns.append((pattern, description, "possible"))
            logger.debug(f"Added pattern '{pattern}' - {description}")
            return True
        except re.error as e:
            logger.error(f"Invalid regex pattern '{pattern}': {e}")
            return False
    
    def remove_pattern(self, pattern: str) -> bool:
        """Remove a regex pattern."""
        original_count = len(self.patterns)
        self.patterns = [(p, d, l) for p, d, l in self.patterns if p != pattern]
        
        if len(self.patterns) < original_count:
            logger.debug(f"Removed pattern '{pattern}'")
            return True
        return False
    
    def get_all_keywords(self) -> Dict[str, List[str]]:
        """Get all keywords organized by priority."""
        return {
            "definite": sorted(self.definite_keywords),
            "possible": sorted(self.possible_keywords)
        }
    
    def get_all_patterns(self) -> List[Tuple[str, str]]:
        """Get all regex patterns."""
        return self.patterns.copy()
    
    def get_keyword_priority(self, keyword: str) -> str:
        """Get priority level of a keyword."""
        keyword_lower = keyword.lower()
        if keyword_lower in self.definite_keywords:
            return "definite"
        elif keyword_lower in self.possible_keywords:
            return "possible"
        return "none"
    
    def scan_all_text_content(self, cache_manager) -> Dict[str, List[Tuple[str, DetectionResult]]]:
        """Scan all text content across all tasks for issues.
        
        Returns:
            Dict mapping task_id to list of (url, DetectionResult) tuples
        """
        results = {}
        
        for task_id in cache_manager.get_task_ids():
            task_results = []
            url_infos = cache_manager.get_task_urls(task_id)
            
            for url_info in url_infos:
                if url_info.content_type == "web":
                    text, _ = cache_manager.get_url_content(task_id, url_info.url,get_screenshot=False)
                    if text:
                        detection_result = self.detect_issues(text)
                        if detection_result.has_issues:
                            task_results.append((url_info.url, detection_result))
            
            if task_results:
                results[task_id] = task_results
        
        return results
