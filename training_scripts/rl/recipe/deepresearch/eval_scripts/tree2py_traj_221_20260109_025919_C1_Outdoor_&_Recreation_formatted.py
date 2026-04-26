import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "america_the_beautiful_annual_pass_2026"
TASK_DESCRIPTION = (
    "As of January 2026, what is the cost of the America the Beautiful Annual Pass for U.S. residents, "
    "and how many motorcycles does a single pass cover?"
)

# Ground truth expectations (used for context in summary/debug)
GROUND_TRUTH = {
    "cost_usd": "$80",
    "motorcycles_covered": "2",
    "official_domains": ["nps.gov", "doi.gov", "usgs.gov"]
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PassExtraction(BaseModel):
    # Values explicitly stated in the answer (as text, not normalized)
    cost_value_text: Optional[str] = None
    cost_value_usd: Optional[str] = None  # Prefer a simple string like "80" or "$80"
    motorcycles_covered_text: Optional[str] = None
    motorcycles_covered_count: Optional[str] = None  # e.g., "2"

    # URLs cited in the answer
    cost_source_urls: List[str] = Field(default_factory=list)
    motorcycle_source_urls: List[str] = Field(default_factory=list)
    all_cited_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pass_info() -> str:
    return """
    Extract from the answer the following fields, strictly based on what is explicitly stated:

    1) cost_value_text: The cost of the America the Beautiful Annual Pass for U.S. residents as described in the answer (verbatim text).
    2) cost_value_usd: The same cost rendered as a simple value string if possible (e.g., "80" or "$80"). If ambiguous, keep the exact phrasing (e.g., "about $80").
    3) motorcycles_covered_text: The description of motorcycle coverage (verbatim text), focusing on how many motorcycles a single pass covers.
    4) motorcycles_covered_count: A normalized numeric string if the answer clearly states a count (e.g., "2"). Otherwise, return null.
    5) cost_source_urls: A list of URLs that the answer explicitly associates with verifying the cost.
    6) motorcycle_source_urls: A list of URLs that the answer explicitly associates with verifying the motorcycle coverage.
    7) all_cited_urls: All other URLs present in the answer that could be relevant to the pass, even if not clearly tied to a specific claim.

    Special rules for URL extraction:
    - Only include actual URLs explicitly present in the answer (plain URLs or markdown links).
    - Ignore malformed URLs.
    - If a URL is missing a protocol, prepend "http://".
    - Do not invent URLs.

    If any of the above items are missing from the answer, return null for the text fields and empty lists for the URL arrays.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_official_url(url: str) -> bool:
    """Return True if the URL appears to be an official NPS or DOI site (including USGS)."""
    if not url:
        return False
    u = url.lower()
    return ("nps.gov" in u) or ("doi.gov" in u) or ("usgs.gov" in u)


def filter_official_urls(urls: List[str]) -> List[str]:
    """Filter to official NPS/DOI/USGS URLs, deduplicate while preserving order."""
    seen = set()
    out: List[str] = []
    for url in urls:
        if not url:
            continue
        if is_official_url(url) and url not in seen:
            out.append(url)
            seen.add(url)
    return out


def coalesce_official_sources_for_cost(extracted: PassExtraction) -> List[str]:
    """
    Choose official URLs for verifying the cost claim.
    Prefer cost_source_urls; if none are official, fall back to all_cited_urls.
    """
    primary = filter_official_urls(extracted.cost_source_urls)
    if primary:
        return primary
    return filter_official_urls(extracted.all_cited_urls)


def coalesce_official_sources_for_motorcycle(extracted: PassExtraction) -> List[str]:
    """
    Choose official URLs for verifying the motorcycle coverage claim.
    Prefer motorcycle_source_urls; if none are official, fall back to all_cited_urls.
    """
    primary = filter_official_urls(extracted.motorcycle_source_urls)
    if primary:
        return primary
    return filter_official_urls(extracted.all_cited_urls)


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: PassExtraction) -> None:
    """
    Build the verification tree according to the rubric:
    - Root: sequential, critical
    - Stated Values: parallel, critical
      • Cost leaf: ensure the answer states $80 as of 2026
      • Motorcycle coverage leaf: ensure the answer states 2 motorcycles as of 2026
    - Official Source Verifiability: parallel, critical
      • Cost sources: ensure at least one official source, and that the source supports $80
      • Motorcycle sources: ensure at least one official source, and that the source supports coverage of two motorcycles
    """

    # Make root critical to enforce that all children are critical as per rubric
    evaluator.root.critical = True

    # 1) Stated Values (parallel, critical)
    stated_values_node = evaluator.add_parallel(
        id="Stated_Values",
        desc="Check that the response provides the required cost and motorcycle coverage values for 2026.",
        parent=evaluator.root,
        critical=True
    )

    # 1.a) Cost stated ($80)
    cost_leaf = evaluator.add_leaf(
        id="Cost_for_US_Residents_2026",
        desc="States that the 2026 cost for U.S. residents is $80.",
        parent=stated_values_node,
        critical=True
    )
    cost_claim = (
        "The answer explicitly states that the 2026 cost for U.S. residents for the America the Beautiful "
        "Annual Pass (Interagency Annual Pass) is $80."
    )
    await evaluator.verify(
        claim=cost_claim,
        node=cost_leaf,
        additional_instruction=(
            "Focus exclusively on the answer text: determine whether it clearly asserts that the price is $80. "
            "Minor phrasing variations like '$80 per year' or 'costs 80 dollars' should count as stating $80."
        )
    )

    # 1.b) Motorcycle coverage stated (two motorcycles)
    moto_leaf = evaluator.add_leaf(
        id="Motorcycle_Coverage_2026",
        desc="States that a single pass covers two motorcycles as of 2026.",
        parent=stated_values_node,
        critical=True
    )
    moto_claim = (
        "The answer explicitly states that a single America the Beautiful Annual Pass covers two motorcycles "
        "as of 2026."
    )
    await evaluator.verify(
        claim=moto_claim,
        node=moto_leaf,
        additional_instruction=(
            "Focus only on the answer text: determine whether it clearly asserts coverage for two motorcycles "
            "(e.g., 'two motorcycles traveling together are covered by one pass'). Minor phrasing variations are acceptable."
        )
    )

    # 2) Official Source Verifiability (parallel, critical)
    official_sources_node = evaluator.add_parallel(
        id="Official_Source_Verifiability",
        desc="Check that the response provides official NPS or DOI sources that support the stated values.",
        parent=evaluator.root,
        critical=True
    )

    # 2.a) Cost official source subtree (sequential, critical)
    cost_source_seq = evaluator.add_sequential(
        id="Official_Source_For_Cost",
        desc="Provides at least one official NPS or DOI source that verifies the stated cost.",
        parent=official_sources_node,
        critical=True
    )

    # Existence of official source(s) for cost
    official_cost_urls = coalesce_official_sources_for_cost(extracted)
    evaluator.add_custom_node(
        result=len(official_cost_urls) > 0,
        id="Official_Source_For_Cost_Exists",
        desc="At least one official NPS/DOI/USGS URL is provided for the cost claim.",
        parent=cost_source_seq,
        critical=True
    )

    # Verification of cost via official source(s)
    cost_support_leaf = evaluator.add_leaf(
        id="Official_Source_For_Cost_Supports",
        desc="The official NPS/DOI/USGS source(s) support the $80 cost.",
        parent=cost_source_seq,
        critical=True
    )
    cost_support_claim = (
        "The official page confirms that the America the Beautiful Annual Pass (Interagency Annual Pass) "
        "cost is $80."
    )
    await evaluator.verify(
        claim=cost_support_claim,
        node=cost_support_leaf,
        sources=official_cost_urls,
        additional_instruction=(
            "Use only official NPS/DOI/USGS pages (e.g., nps.gov, doi.gov, usgs.gov). "
            "The claim should be supported if the page explicitly states a price of $80 for the annual pass."
        )
    )

    # 2.b) Motorcycle coverage official source subtree (sequential, critical)
    moto_source_seq = evaluator.add_sequential(
        id="Official_Source_For_Motorcycle_Coverage",
        desc="Provides at least one official NPS or DOI source that verifies the stated motorcycle coverage.",
        parent=official_sources_node,
        critical=True
    )

    # Existence of official source(s) for motorcycle coverage
    official_moto_urls = coalesce_official_sources_for_motorcycle(extracted)
    evaluator.add_custom_node(
        result=len(official_moto_urls) > 0,
        id="Official_Source_For_Motorcycle_Coverage_Exists",
        desc="At least one official NPS/DOI/USGS URL is provided for the motorcycle coverage claim.",
        parent=moto_source_seq,
        critical=True
    )

    # Verification of motorcycle coverage via official source(s)
    moto_support_leaf = evaluator.add_leaf(
        id="Official_Source_For_Motorcycle_Coverage_Supports",
        desc="The official NPS/DOI/USGS source(s) support coverage of two motorcycles.",
        parent=moto_source_seq,
        critical=True
    )
    moto_support_claim = (
        "The official page confirms that one America the Beautiful Annual Pass covers two motorcycles "
        "travelling together."
    )
    await evaluator.verify(
        claim=moto_support_claim,
        node=moto_support_leaf,
        sources=official_moto_urls,
        additional_instruction=(
            "Use only official NPS/DOI/USGS pages. The claim should be supported if the page explicitly states "
            "that a single pass covers two motorcycles (often phrased as two motorcycles traveling together)."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
) -> Dict[str, Any]:
    """
    Evaluate the answer for the America the Beautiful Annual Pass (2026) task and return
    a structured summary including the verification tree and final score.
    """
    # Initialize evaluator with sequential root to enforce order: stated values → sources
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_pass_info(),
        template_class=PassExtraction,
        extraction_name="pass_values_and_sources",
    )

    # Ground truth info (for debugging/context)
    evaluator.add_ground_truth({
        "expected_cost": GROUND_TRUTH["cost_usd"],
        "expected_motorcycles_covered": GROUND_TRUTH["motorcycles_covered"],
        "official_domains": GROUND_TRUTH["official_domains"]
    })

    # Build and run verification tree
    await build_verification_tree(evaluator, extracted)

    # Return structured result
    return evaluator.get_summary()