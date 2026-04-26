import asyncio
import logging
from typing import Any, List, Optional, Dict
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "outage_stats_percentages"
TASK_DESCRIPTION = (
    "According to industry reports on network and data center outages, what percentage of major outages "
    "are attributed to IT software issues, and what percentage are attributed to cyberattacks and ransomware? "
    "Please provide the source reference for this data."
)

EXPECTED_IT_SOFTWARE_PERCENT = "18%"
EXPECTED_CYBERATTACKS_PERCENT = "11%"
ACCEPTED_SOURCE_DOMAINS = [
    "uptimeinstitute.com",  # Any Uptime Institute report page/domain
]
ACCEPTED_SPECIFIC_CIRCLEID_PATH = "/posts/20230906-the-causes-of-network-outages"
ACCEPTED_CIRCLEID_DOMAIN = "circleid.com"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class OutageStatsExtraction(BaseModel):
    """
    Structured extraction of the two required percentages and any source URLs
    explicitly provided in the answer.
    """
    it_software_pct: Optional[str] = None
    cyberattacks_pct: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_outage_stats() -> str:
    return """
    Extract from the answer:
    1) it_software_pct: The stated percentage for IT software issues (a.k.a. software-related issues, software failures/bugs) accounting for major (or publicly reported major) network/data center outages. Return it exactly as written (e.g., "18%", "18 percent", or "18").
    2) cyberattacks_pct: The stated percentage for cyberattacks and ransomware accounting for major (or publicly reported major) network/data center outages. Return it exactly as written (e.g., "11%", "11 percent", or "11").
    3) sources: A list of all source URLs explicitly provided in the answer that are cited for these percentages (e.g., Uptime Institute report pages, CircleID article links, etc.). Extract only actual URLs present in the answer text (including markdown links). If the answer mentions a source without a URL, do not fabricate a URL; just omit it.

    If a requested field is not present in the answer, set it to null (for strings) or empty list (for sources).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _url_matches_allowed_sources(urls: List[str]) -> bool:
    """
    Determine whether the provided URLs include at least one acceptable source:
    - Any Uptime Institute domain (uptimeinstitute.com or subdomains), OR
    - The specific CircleID article at /posts/20230906-the-causes-of-network-outages (with flexible scheme/hostname).
    """
    for raw in urls:
        if not raw:
            continue
        try:
            parsed = urlparse(raw.strip())
            host = (parsed.netloc or "").lower()
            path = (parsed.path or "").rstrip("/").lower()

            # Accept direct Uptime Institute host matches (including subdomains)
            if any(host.endswith(dom) for dom in ACCEPTED_SOURCE_DOMAINS):
                return True

            # Accept CircleID exact article path (allowing any subdomain under circleid.com)
            if host.endswith(ACCEPTED_CIRCLEID_DOMAIN) and path.endswith(ACCEPTED_SPECIFIC_CIRCLEID_PATH.strip("/")):
                return True

            # Also accept if the entire raw URL string contains the specific circleid article path,
            # to be lenient with archive/proxy URLs that embed the original link in the path.
            low = raw.lower()
            if ACCEPTED_CIRCLEID_DOMAIN in low and ACCEPTED_SPECIFIC_CIRCLEID_PATH in low:
                return True

        except Exception:
            # Skip malformed URLs
            continue
    return False


def _normalize_percent_str(s: Optional[str]) -> Optional[str]:
    """
    Normalize a percentage representation from the answer into a comparable string form.
    - Keeps numbers only, strips spaces and the percent symbol, e.g., "18%", "18 percent", "18" -> "18".
    - Returns None if input is None or empty after normalization.
    """
    if not s:
        return None
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits if digits else None


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify(
    evaluator: Evaluator,
    extracted: OutageStatsExtraction,
    parent_node
) -> None:
    """
    Build the verification tree according to the rubric and execute the checks.
    - One critical parent node "Answer_Completeness" (parallel).
    - Three critical leaves:
        • IT_Software_Issues_Percentage
        • Cyberattacks_Percentage
        • Source_Reference
    """
    # Create the main critical node
    completeness_node = evaluator.add_parallel(
        id="Answer_Completeness",
        desc="The answer provides both required percentage values and appropriate source reference",
        parent=parent_node,
        critical=True
    )

    # Leaf 1: IT software issues = 18%
    it_leaf = evaluator.add_leaf(
        id="IT_Software_Issues_Percentage",
        desc="The answer correctly identifies that IT software issues account for 18% of publicly reported major network outages",
        parent=completeness_node,
        critical=True
    )
    # Verify directly against the answer content (simple verify).
    # This ensures the answer text contains the correct percentage.
    it_claim = (
        "The answer states that IT software issues (i.e., software-related problems, software failures/bugs) "
        "account for 18% of publicly reported major network/data center outages."
    )
    await evaluator.verify(
        claim=it_claim,
        node=it_leaf,
        additional_instruction=(
            "Judge from the answer text whether the specific percentage for IT software issues is 18%. "
            "Allow minor wording variants such as 'software issues', 'software failures', 'software-related problems', "
            "and formats like '18%' or '18 percent'. The key is that the answer explicitly gives 18% for IT software-related outages."
        ),
    )

    # Leaf 2: Cyberattacks and ransomware = 11%
    cyber_leaf = evaluator.add_leaf(
        id="Cyberattacks_Percentage",
        desc="The answer correctly identifies that cyberattacks and ransomware account for 11% of major network outages",
        parent=completeness_node,
        critical=True
    )
    # Verify directly against the answer content (simple verify).
    cyber_claim = (
        "The answer states that cyberattacks and ransomware account for 11% of major network/data center outages."
    )
    await evaluator.verify(
        claim=cyber_claim,
        node=cyber_leaf,
        additional_instruction=(
            "Judge from the answer text whether the specific percentage for cyberattacks (including ransomware) is 11%. "
            "Accept phrasing like 'cyberattacks and ransomware', 'ransomware and other cyber attacks', etc., "
            "and formats like '11%' or '11 percent'. The key is that the answer explicitly gives 11%."
        ),
    )

    # Leaf 3: Source reference presence and correctness (Uptime Institute report or the CircleID article)
    sources_list = extracted.sources or []
    has_accepted_source = _url_matches_allowed_sources(sources_list)
    evaluator.add_custom_node(
        result=has_accepted_source,
        id="Source_Reference",
        desc="The answer references the Uptime Institute report or the CircleID article (https://circleid.com/posts/20230906-the-causes-of-network-outages) as the source of this data",
        parent=completeness_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the outage statistics task.
    """
    # Initialize evaluator with a parallel root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract the needed fields from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_outage_stats(),
        template_class=OutageStatsExtraction,
        extraction_name="outage_stats_extraction"
    )

    # Add ground truth info and extraction normalization for transparency
    evaluator.add_ground_truth({
        "expected_it_software_pct": EXPECTED_IT_SOFTWARE_PERCENT,
        "expected_cyberattacks_pct": EXPECTED_CYBERATTACKS_PERCENT,
        "accepted_sources": {
            "domains": ACCEPTED_SOURCE_DOMAINS,
            "specific_circleid_article": f"https://{ACCEPTED_CIRCLEID_DOMAIN}{ACCEPTED_SPECIFIC_CIRCLEID_PATH}"
        }
    })

    # Optionally record a normalized version of extracted percentages (non-scoring info)
    normalized_info = {
        "it_software_pct_raw": extracted.it_software_pct,
        "it_software_pct_normalized": _normalize_percent_str(extracted.it_software_pct),
        "cyberattacks_pct_raw": extracted.cyberattacks_pct,
        "cyberattacks_pct_normalized": _normalize_percent_str(extracted.cyberattacks_pct),
        "num_sources_found": len(extracted.sources or []),
    }
    evaluator.add_custom_info(normalized_info, info_type="extraction_normalization", info_name="normalized_extraction")

    # Build tree and run verifications
    await build_and_verify(evaluator, extracted, root)

    # Return summary
    return evaluator.get_summary()