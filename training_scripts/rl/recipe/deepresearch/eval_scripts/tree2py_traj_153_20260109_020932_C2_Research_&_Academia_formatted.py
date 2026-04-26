import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "usnews_2025_cs_and_grfp"
TASK_DESCRIPTION = (
    "I am researching top computer science PhD programs in the United States and graduate fellowship opportunities. "
    "According to the U.S. News 2025 Best Graduate Computer Science Schools rankings, which university's program is ranked #1? "
    "Additionally, for the NSF Graduate Research Fellowship Program (GRFP), which provides funding to graduate students in STEM fields, "
    "what is the annual stipend amount provided to fellows, and what is the Cost of Education allowance amount provided per fellowship year?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class USNewsRankingInfo(BaseModel):
    top_university: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class GRFPFinanceInfo(BaseModel):
    annual_stipend: Optional[str] = None
    cost_of_education_allowance: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ResearchExtraction(BaseModel):
    usnews: Optional[USNewsRankingInfo] = None
    grfp: Optional[GRFPFinanceInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt builders                                                  #
# --------------------------------------------------------------------------- #
def prompt_extract_research() -> str:
    return """
    Extract two sections from the answer:

    1) usnews:
       - top_university: The university named as ranked #1 in the "U.S. News 2025 Best Graduate Computer Science Schools" ranking.
       - sources: An array of URLs explicitly provided in the answer that are intended to support this ranking claim. Include only actual URLs (e.g., https://...).
    
    2) grfp:
       - annual_stipend: The annual stipend amount stated for the NSF GRFP (e.g., "$37,000 per year"). Extract exactly as written in the answer, including currency symbols and punctuation if present.
       - cost_of_education_allowance: The Cost of Education allowance amount per fellowship year stated for the NSF GRFP (e.g., "$12,000"). Extract exactly as written in the answer, including currency symbols/punctuation if present.
       - sources: An array of URLs explicitly provided in the answer for GRFP financial information. Include only actual URLs (e.g., https://...).

    Rules and notes:
    - Do not invent values or URLs. If a field is not present in the answer, set it to null (for strings) or [] (for arrays).
    - For URLs, include only valid, complete URLs mentioned in the answer. If a URL lacks a protocol, prepend http://
    - If multiple URLs are provided, include them all.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_domain(url: str, allowed_suffixes: List[str]) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
        return any(netloc.endswith(suffix) for suffix in allowed_suffixes)
    except Exception:
        return False


def _filter_urls(urls: List[str]) -> List[str]:
    # Deduplicate while preserving order, keep non-empty and plausible URLs
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        normalized = u.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def is_usnews_url(url: str) -> bool:
    return _is_domain(url, ["usnews.com"])


def is_nsf_official_url(url: str) -> bool:
    # Treat NSF/official GRFP documentation domains as official sources
    return _is_domain(url, ["nsf.gov", "research.gov", "nsfgrfp.nsf.gov", "nsfgrfp.org"])


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def build_usnews_verification(
    evaluator: Evaluator,
    parent_node,
    extracted: ResearchExtraction,
) -> None:
    """
    Build and execute verification nodes for the U.S. News 2025 CS #1 university identification.
    """
    usnews_info = extracted.usnews or USNewsRankingInfo()
    top_uni = (usnews_info.top_university or "").strip()
    raw_sources = _filter_urls(usnews_info.sources)
    usnews_sources = [u for u in raw_sources if is_usnews_url(u)]

    # Aggregator for University Identification (Critical)
    uni_node = evaluator.add_parallel(
        id="University_Identification",
        desc="State which university is ranked #1 in the U.S. News 2025 Best Graduate Computer Science Schools rankings, "
             "and ensure the claim is verifiable via an official U.S. News source (e.g., citation/link).",
        parent=parent_node,
        critical=True
    )

    # Existence check: name + at least one source present
    evaluator.add_custom_node(
        result=(top_uni != "" and len(raw_sources) > 0),
        id="usnews_top_exists",
        desc="U.S. News #1 university is identified and at least one source is provided",
        parent=uni_node,
        critical=True
    )

    # Official source check: at least one U.S. News URL provided
    evaluator.add_custom_node(
        result=(len(usnews_sources) > 0),
        id="usnews_official_source_present",
        desc="At least one official U.S. News source is provided",
        parent=uni_node,
        critical=True
    )

    # Content support: Verify that the provided U.S. News source(s) support the claim for 2025
    top_supported_node = evaluator.add_leaf(
        id="usnews_top_supported",
        desc="U.S. News source(s) support the stated #1 CS program for 2025",
        parent=uni_node,
        critical=True
    )
    claim = (
        f"According to the U.S. News 2025 Best Graduate Computer Science Schools ranking, "
        f"the #1 program is {top_uni}. If the ranking shows a tie for #1, the claim is correct if {top_uni} is among "
        f"the tied #1 schools."
    )
    await evaluator.verify(
        claim=claim,
        node=top_supported_node,
        sources=usnews_sources if usnews_sources else raw_sources,
        additional_instruction=(
            "Focus on the 2025 'Best Graduate Computer Science' rankings on usnews.com. "
            "Accept 'tie' for #1 if the named university appears among those tied at rank #1. "
            "Minor formatting differences in the university name are acceptable."
        )
    )


async def build_grfp_verification(
    evaluator: Evaluator,
    parent_node,
    extracted: ResearchExtraction,
) -> None:
    """
    Build and execute verification nodes for NSF GRFP financial information.
    """
    grfp_info = extracted.grfp or GRFPFinanceInfo()
    stipend = (grfp_info.annual_stipend or "").strip()
    coe = (grfp_info.cost_of_education_allowance or "").strip()
    raw_sources = _filter_urls(grfp_info.sources)
    official_sources = [u for u in raw_sources if is_nsf_official_url(u)]

    # Aggregator for GRFP financial info (Critical)
    grfp_node = evaluator.add_parallel(
        id="NSF_GRFP_Financial_Information",
        desc="Report NSF GRFP financial components using values that are verifiable from official NSF GRFP documentation.",
        parent=parent_node,
        critical=True
    )

    # Global official source presence check for GRFP (Critical sibling to gate detailed verifications)
    evaluator.add_custom_node(
        result=(len(official_sources) > 0),
        id="grfp_official_source_present",
        desc="At least one official NSF/GRFP source URL is provided (e.g., nsf.gov, nsfgrfp.nsf.gov, research.gov).",
        parent=grfp_node,
        critical=True
    )

    # Annual Stipend verification block (Critical)
    stipend_block = evaluator.add_parallel(
        id="Annual_Stipend",
        desc="Provide the annual stipend amount for NSF GRFP fellows, verifiable from official NSF GRFP documentation.",
        parent=grfp_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(stipend != "" and len(raw_sources) > 0),
        id="grfp_stipend_exists",
        desc="Annual stipend amount is specified and at least one source is provided",
        parent=stipend_block,
        critical=True
    )

    stipend_supported_node = evaluator.add_leaf(
        id="grfp_stipend_supported",
        desc="Official NSF/GRFP source(s) support the stated annual stipend amount",
        parent=stipend_block,
        critical=True
    )
    stipend_claim = (
        f"The NSF GRFP annual stipend amount is {stipend} per fellowship year (or per 12-month period), as stated in official NSF GRFP materials."
    )
    await evaluator.verify(
        claim=stipend_claim,
        node=stipend_supported_node,
        sources=official_sources if official_sources else raw_sources,
        additional_instruction=(
            "Verify the stipend amount from official NSF/GRFP documentation (e.g., nsf.gov pages, program solicitations, "
            "official GRFP site/handbook). Minor formatting differences (like presence/absence of '$' or commas) are acceptable "
            "as long as the numeric amount matches."
        )
    )

    # Cost of Education Allowance verification block (Critical)
    coe_block = evaluator.add_parallel(
        id="Cost_of_Education_Allowance",
        desc="Provide the Cost of Education allowance amount per fellowship year, verifiable from official NSF GRFP documentation.",
        parent=grfp_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(coe != "" and len(raw_sources) > 0),
        id="grfp_coe_exists",
        desc="Cost of Education allowance amount is specified and at least one source is provided",
        parent=coe_block,
        critical=True
    )

    coe_supported_node = evaluator.add_leaf(
        id="grfp_coe_supported",
        desc="Official NSF/GRFP source(s) support the stated Cost of Education allowance amount",
        parent=coe_block,
        critical=True
    )
    coe_claim = (
        f"The NSF GRFP Cost of Education (COE) allowance amount is {coe} per fellowship year, as stated in official NSF GRFP materials."
    )
    await evaluator.verify(
        claim=coe_claim,
        node=coe_supported_node,
        sources=official_sources if official_sources else raw_sources,
        additional_instruction=(
            "Verify the COE allowance amount from official NSF/GRFP documentation (e.g., nsf.gov pages, program solicitations, "
            "official GRFP site/handbook). Minor formatting differences are acceptable as long as the numeric amount matches."
        )
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
    Evaluate an answer for identifying the U.S. News 2025 #1 CS program and NSF GRFP stipend/COE amounts.
    """
    # Initialize evaluator (root is non-critical; we add a critical node under it)
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_research(),
        template_class=ResearchExtraction,
        extraction_name="research_extraction",
    )

    # Build main "Complete_Task" critical aggregator
    complete_task = evaluator.add_parallel(
        id="Complete_Task",
        desc="Identify the #1 U.S. News 2025 CS program and report NSF GRFP stipend and Cost of Education amounts, with official-source verification.",
        parent=root,
        critical=True
    )

    # University Identification subtree
    await build_usnews_verification(evaluator, complete_task, extracted)

    # NSF GRFP Financial Information subtree
    await build_grfp_verification(evaluator, complete_task, extracted)

    # Return summary
    return evaluator.get_summary()