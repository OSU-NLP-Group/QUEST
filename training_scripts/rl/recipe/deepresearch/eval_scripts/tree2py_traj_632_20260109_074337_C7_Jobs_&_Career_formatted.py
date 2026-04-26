import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "bls_fastest_healthcare_single_occupation_2024_2034"
TASK_DESCRIPTION = (
    "From the Bureau of Labor Statistics' list of 20 fastest-growing occupations (2024-2034), identify one healthcare occupation that meets ALL of the following criteria: "
    "(1) Appears on the official BLS fastest-growing occupations list with employment projections for 2024-2034, "
    "(2) Is classified as a healthcare occupation, "
    "(3) Has a projected employment growth rate of at least 15% from 2024 to 2034, "
    "(4) Has a 2024 median annual wage between $60,000 and $100,000, "
    "(5) Requires an associate's degree as the typical entry-level education, "
    "(6) Requires state-issued licensure or certification in most or all U.S. states, "
    "(7) Requires passing a professional qualifying exam for licensure or certification, "
    "(8) Requires no prior work experience in a related occupation, "
    "(9) Had more than 40,000 jobs in 2024, "
    "(10) Has state-level employment and wage data available through the BLS OEWS program, "
    "(11) Has published 10th and 90th percentile wage data for May 2024, "
    "(12) Has documented primary industries of employment, and "
    "(13) Has published annual average job openings projections for 2024-2034. "
    "Provide the occupation title exactly as it appears on the BLS Occupational Outlook Handbook, along with the direct URL to its official BLS OOH profile page."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SelectedOccupation(BaseModel):
    title: Optional[str] = None
    ooh_url: Optional[str] = None


class URLBuckets(BaseModel):
    fastest_growing_urls: List[str] = Field(default_factory=list)
    oews_urls: List[str] = Field(default_factory=list)
    other_urls: List[str] = Field(default_factory=list)


class OOHSelectionExtraction(BaseModel):
    selected_occupations: List[SelectedOccupation] = Field(default_factory=list)
    urls: URLBuckets = Field(default_factory=URLBuckets)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_selection() -> str:
    return """
    Extract the single occupation the answer ultimately selects to satisfy the user's request. If multiple occupations are presented as candidates or alternatives, extract all explicitly selected ones in the order they are presented, but do not invent any.

    Return:
    - selected_occupations: an array of objects, each with:
        • title: The occupation title exactly as written in the answer (do NOT rewrite it; preserve capitalization and punctuation).
        • ooh_url: The direct URL to the official BLS Occupational Outlook Handbook profile page provided in the answer for that occupation, if any. This should be an official bls.gov OOH occupation URL (e.g., https://www.bls.gov/ooh/...); if the answer provides a non-OOH URL or no URL, set to null.
    - urls:
        • fastest_growing_urls: All URLs in the answer that point to the official BLS “fastest-growing occupations” list for the 2024–2034 projections (typically on bls.gov at Employment Projections/EP pages; examples often include 'emp/tables/fastest-growing-occupations.htm' or pages clearly labeled as the BLS 2024–34 fastest-growing list). Include only URLs actually present in the answer.
        • oews_urls: All URLs in the answer that point to BLS Occupational Employment and Wage Statistics (OEWS/OES) pages for this occupation, which publish wage percentiles and state/area data (e.g., bls.gov/oews or bls.gov/oes occupation pages). Include only URLs actually present in the answer.
        • other_urls: Any other URLs explicitly provided in the answer that may support requirements such as licensure, exams, or projections. Include only URLs actually present in the answer.

    Important:
    - Do NOT infer or create any URL. Only extract URLs explicitly present in the answer.
    - Preserve all URLs as full URLs (with protocol).
    - If more than one occupation is explicitly and definitively “selected,” include them all in selected_occupations in the presented order. Otherwise, return a single item.
    - If the answer provides no URLs for a category, return an empty list for that category.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _safe_first_selected(extraction: OOHSelectionExtraction) -> SelectedOccupation:
    if extraction and extraction.selected_occupations:
        return extraction.selected_occupations[0]
    return SelectedOccupation()


def _non_empty_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


def _merge_unique(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst:
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_criteria(
    evaluator: Evaluator,
    parent_node,
    extraction: OOHSelectionExtraction,
) -> None:
    """
    Build the verification leaves according to the rubric and run verifications.
    All criteria under this parent are critical: failing any will fail the task node.
    """

    # Prepare extracted core data
    sel = _safe_first_selected(extraction)
    title = sel.title or ""
    ooh_url = sel.ooh_url or ""

    fastest_urls = _non_empty_urls(extraction.urls.fastest_growing_urls)
    oews_urls = _non_empty_urls(extraction.urls.oews_urls)
    other_urls = _non_empty_urls(extraction.urls.other_urls)

    # Convenience source groups
    sources_ooh = ooh_url if ooh_url else None
    sources_fastest = fastest_urls if fastest_urls else (_non_empty_urls([ooh_url]) if ooh_url else other_urls)
    sources_oews = oews_urls if oews_urls else other_urls

    # 1) exactly_one_occupation (custom existence/count check)
    evaluator.add_custom_node(
        result=(len(extraction.selected_occupations) == 1),
        id="exactly_one_occupation",
        desc="The answer identifies exactly one occupation (not multiple occupations) as the selected occupation",
        parent=parent_node,
        critical=True
    )

    # Create all leaf nodes (critical) and collect claims for batch verification
    claims_and_nodes: List[Dict[str, Any]] = []

    # 2) occupation_title_exact_ooH
    node_title_exact = evaluator.add_leaf(
        id="occupation_title_exact_ooH",
        desc="The provided occupation title matches exactly how it appears on the official BLS Occupational Outlook Handbook (OOH) profile page",
        parent=parent_node,
        critical=True
    )
    claim_title_exact = (
        f"The main title shown on this OOH profile page matches exactly '{title}', including capitalization and punctuation."
    )
    claims_and_nodes.append({
        "claim": claim_title_exact,
        "sources": sources_ooh,
        "node": node_title_exact,
        "add_ins": "Check the page's main heading (hero title). Treat minor whitespace differences as mismatches if the words differ; evaluate exact match string equality."
    })

    # 3) bls_fastest_growing_list
    node_fastest = evaluator.add_leaf(
        id="bls_fastest_growing_list",
        desc="The identified occupation appears on the official BLS list of 20 fastest-growing occupations for the 2024-2034 projection period",
        parent=parent_node,
        critical=True
    )
    claim_fastest = (
        f"The occupation '{title}' appears on the official BLS list of the 20 fastest-growing occupations for the 2024–2034 projections."
    )
    claims_and_nodes.append({
        "claim": claim_fastest,
        "sources": sources_fastest,
        "node": node_fastest,
        "add_ins": "Only accept if an official BLS Employment Projections page for the 2024–34 period explicitly lists the occupation among the top 20 fastest-growing. If the provided page(s) are not from bls.gov EP content or do not explicitly list the occupation, judge as not supported."
    })

    # 4) healthcare_classification
    node_healthcare = evaluator.add_leaf(
        id="healthcare_classification",
        desc="The occupation is classified as a healthcare occupation in the BLS Occupational Outlook Handbook system",
        parent=parent_node,
        critical=True
    )
    claim_healthcare = (
        "This OOH profile indicates the occupation is classified under the Healthcare occupational group (e.g., via breadcrumbs or the page's section path)."
    )
    claims_and_nodes.append({
        "claim": claim_healthcare,
        "sources": sources_ooh,
        "node": node_healthcare,
        "add_ins": "Look for OOH breadcrumbs, URL path containing /ooh/healthcare/, or explicit labeling that indicates the Healthcare group."
    })

    # 5) growth_rate_threshold (>= 15%)
    node_growth = evaluator.add_leaf(
        id="growth_rate_threshold",
        desc="The occupation has a projected employment growth rate of at least 15% from 2024 to 2034 according to BLS employment projections",
        parent=parent_node,
        critical=True
    )
    claim_growth = (
        "The projected percent change in employment from 2024 to 2034 for this occupation is at least 15%."
    )
    claims_and_nodes.append({
        "claim": claim_growth,
        "sources": _merge_unique([ooh_url] if ooh_url else [], fastest_urls),
        "node": node_growth,
        "add_ins": "Prefer the OOH Job Outlook section or official BLS projections tables. If the page shows a percent change, confirm it is >= 15%."
    })

    # 6) median_wage_range ($60k–$100k inclusive, May 2024)
    node_median = evaluator.add_leaf(
        id="median_wage_range",
        desc="The occupation's May 2024 median annual wage is between $60,000 and $100,000 inclusive according to BLS wage statistics",
        parent=parent_node,
        critical=True
    )
    claim_median = (
        "The May 2024 median annual pay for this occupation is between $60,000 and $100,000 inclusive."
    )
    claims_and_nodes.append({
        "claim": claim_median,
        "sources": sources_ooh,
        "node": node_median,
        "add_ins": "Use OOH Quick Facts or Pay section. If the page clearly states May 2024 median pay within the range, accept."
    })

    # 7) associate_degree_requirement
    node_assoc = evaluator.add_leaf(
        id="associate_degree_requirement",
        desc="The typical entry-level education is an associate's degree as specified by BLS (e.g., in OOH Quick Facts)",
        parent=parent_node,
        critical=True
    )
    claim_assoc = (
        "The typical entry-level education for this occupation is an associate's degree."
    )
    claims_and_nodes.append({
        "claim": claim_assoc,
        "sources": sources_ooh,
        "node": node_assoc,
        "add_ins": "Check OOH Quick Facts or How to Become One section. Accept if it clearly states associate's degree is the typical entry-level education."
    })

    # 8) state_licensure_requirement
    node_licensure = evaluator.add_leaf(
        id="state_licensure_requirement",
        desc="The occupation requires state-issued licensure or certification in most or all U.S. states according to the OOH profile or linked authoritative sources",
        parent=parent_node,
        critical=True
    )
    claim_licensure = (
        "This occupation requires a state-issued license or certification in most or all U.S. states."
    )
    claims_and_nodes.append({
        "claim": claim_licensure,
        "sources": _merge_unique([ooh_url] if ooh_url else [], other_urls),
        "node": node_licensure,
        "add_ins": "Look for explicit statements about licensure or certification requirements across states in the OOH 'How to become one' section or authoritative links cited there."
    })

    # 9) qualifying_exam_requirement
    node_exam = evaluator.add_leaf(
        id="qualifying_exam_requirement",
        desc="The occupation requires passing a professional qualifying exam to obtain licensure or certification",
        parent=parent_node,
        critical=True
    )
    claim_exam = (
        "To obtain licensure or certification for this occupation, passing a professional qualifying examination is required."
    )
    claims_and_nodes.append({
        "claim": claim_exam,
        "sources": _merge_unique([ooh_url] if ooh_url else [], other_urls),
        "node": node_exam,
        "add_ins": "Verify explicit mention of a required exam (e.g., a national board or standardized professional exam) as part of licensure or certification."
    })

    # 10) no_prior_experience
    node_no_exp = evaluator.add_leaf(
        id="no_prior_experience",
        desc="BLS specifies that no work experience in a related occupation is required (e.g., 'None' in OOH Quick Facts)",
        parent=parent_node,
        critical=True
    )
    claim_no_exp = (
        "The OOH indicates that work experience in a related occupation required is 'None' for this occupation."
    )
    claims_and_nodes.append({
        "claim": claim_no_exp,
        "sources": sources_ooh,
        "node": node_no_exp,
        "add_ins": "Check OOH Quick Facts for 'Work experience in a related occupation: None'."
    })

    # 11) employment_threshold (> 40,000 jobs in 2024)
    node_jobs = evaluator.add_leaf(
        id="employment_threshold",
        desc="The occupation had more than 40,000 jobs in 2024 according to BLS employment statistics",
        parent=parent_node,
        critical=True
    )
    claim_jobs = (
        "The number of jobs in 2024 for this occupation exceeds 40,000."
    )
    claims_and_nodes.append({
        "claim": claim_jobs,
        "sources": sources_ooh,
        "node": node_jobs,
        "add_ins": "Use OOH Quick Facts. Accept if 'Number of jobs, 2024' is greater than 40,000."
    })

    # 12) state_oews_data (availability via OEWS)
    node_state_oews = evaluator.add_leaf(
        id="state_oews_data",
        desc="State-level employment and wage data for the occupation are available through the BLS OEWS program (as indicated by BLS state/area data availability)",
        parent=parent_node,
        critical=True
    )
    claim_state_oews = (
        "The BLS OEWS program publishes state-level employment and wage data for this occupation (i.e., there is an OEWS page with state or area tables for the occupation)."
    )
    claims_and_nodes.append({
        "claim": claim_state_oews,
        "sources": sources_oews,
        "node": node_state_oews,
        "add_ins": "Verify on a BLS OEWS/OES occupation page that state or area data tables are provided for this occupation."
    })

    # 13) percentile_wage_data (10th and 90th for May 2024)
    node_percentiles = evaluator.add_leaf(
        id="percentile_wage_data",
        desc="Both 10th-percentile and 90th-percentile wage data for May 2024 are published by BLS for this occupation",
        parent=parent_node,
        critical=True
    )
    claim_percentiles = (
        "For May 2024, BLS publishes both the 10th-percentile and 90th-percentile wage estimates for this occupation."
    )
    claims_and_nodes.append({
        "claim": claim_percentiles,
        "sources": sources_oews,
        "node": node_percentiles,
        "add_ins": "On the BLS OEWS/OES occupation page, confirm that the 'Percentile wage estimates' include both the 10th and 90th percentiles and that the data correspond to May 2024."
    })

    # 14) industry_employment_data
    node_industries = evaluator.add_leaf(
        id="industry_employment_data",
        desc="The BLS occupational profile documents the primary industries employing this occupation (with employment figures or percentages)",
        parent=parent_node,
        critical=True
    )
    claim_industries = (
        "BLS documents the primary industries employing this occupation, with employment counts or percentages, on an occupational profile/OEWS page."
    )
    claims_and_nodes.append({
        "claim": claim_industries,
        "sources": sources_oews,
        "node": node_industries,
        "add_ins": "On the OEWS occupation page, look for 'Industries with the highest published employment and wages' or similar industry employment distribution information."
    })

    # 15) annual_openings_projection
    node_openings = evaluator.add_leaf(
        id="annual_openings_projection",
        desc="Annual average job openings projections for 2024-2034 are published by BLS for this occupation",
        parent=parent_node,
        critical=True
    )
    claim_openings = (
        "The OOH profile publishes the annual average job openings for 2024–34 for this occupation."
    )
    claims_and_nodes.append({
        "claim": claim_openings,
        "sources": sources_ooh,
        "node": node_openings,
        "add_ins": "Check OOH Quick Facts/Job Outlook for 'Job openings, 2024–34, yearly average' or equivalent phrasing."
    })

    # 16) direct_ooh_profile_url (simple verification that the answer provided such a URL)
    node_direct_ooh = evaluator.add_leaf(
        id="direct_ooh_profile_url",
        desc="The answer provides a direct URL to the official BLS OOH profile page for the identified occupation (i.e., an official bls.gov OOH occupation page corresponding to the named occupation)",
        parent=parent_node,
        critical=True
    )
    claim_direct_ooh = (
        "The answer includes a direct URL to the official BLS Occupational Outlook Handbook (OOH) profile page for the identified occupation."
    )
    # For this check, verify against the answer content (no URL fetch required)
    claims_and_nodes.append({
        "claim": claim_direct_ooh,
        "sources": None,
        "node": node_direct_ooh,
        "add_ins": "Verify, based on the provided answer text, that it contains a direct official BLS OOH occupation URL (https://www.bls.gov/ooh/...). Do not accept non-OOH or non-bls.gov URLs."
    })

    # Execute all verifications in parallel
    await evaluator.batch_verify(
        [
            (item["claim"], item["sources"], item["node"], item["add_ins"])
            for item in claims_and_nodes
        ]
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
) -> Dict:
    """
    Evaluate a single answer for the BLS fastest-growing healthcare occupation task and return a structured result dictionary.
    """
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
    extraction = await evaluator.extract(
        prompt=prompt_extract_selection(),
        template_class=OOHSelectionExtraction,
        extraction_name="selected_occupation_and_urls",
    )

    # Add a critical aggregator node under root (since framework root is non-critical)
    task_node = evaluator.add_parallel(
        id="task_verification",
        desc="Identify exactly one healthcare occupation from the BLS 2024–2034 fastest-growing list that meets all specified criteria and provide exact OOH title and direct OOH URL",
        parent=root,
        critical=True
    )

    # Add some helpful custom info for debugging/trace
    first_sel = _safe_first_selected(extraction)
    evaluator.add_custom_info(
        info={
            "selected_count": len(extraction.selected_occupations),
            "selected_title_first": first_sel.title,
            "selected_ooh_url_first": first_sel.ooh_url,
            "fastest_growing_urls": extraction.urls.fastest_growing_urls,
            "oews_urls": extraction.urls.oews_urls,
            "other_urls": extraction.urls.other_urls,
        },
        info_type="extraction_summary",
        info_name="selection_summary"
    )

    # Build and verify all rubric criteria under the critical aggregator
    await build_and_verify_criteria(evaluator, task_node, extraction)

    # Return the evaluation summary
    return evaluator.get_summary()