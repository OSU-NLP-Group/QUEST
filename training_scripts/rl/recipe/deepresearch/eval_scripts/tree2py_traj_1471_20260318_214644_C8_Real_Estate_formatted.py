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
TASK_ID = "real_estate_companies_2026"
TASK_DESCRIPTION = """
Identify four distinct real estate companies or entities operating in the United States, each meeting the following specific criteria:

Company 1: The company that ranks as the #1 largest property management company in the United States as of 2026.
- Provide the total number of units it manages in the USA
- Provide its headquarters location (city and state)
- Verify that it appears in the top 5 largest property management companies

Company 2: The company that ranked #1 on Multi-Housing News' 2025 Top Multifamily Developers list.
- Provide the approximate number of units it had under construction as of June 30, 2025
- Provide the primary type of development projects in its active pipeline (e.g., market-rate, student housing, senior housing, or affordable housing)
- Verify whether it has held the #1 developer ranking for multiple consecutive years

Company 3: The healthcare-focused REIT that appears among the top 3 largest REITs worldwide by market capitalization as of February 2026.
- Verify its specialization in healthcare real estate, particularly seniors housing or senior living communities
- Provide information about the size of its portfolio in terms of number of properties or communities
- Confirm that it is headquartered in the United States

Company 4: The industrial or logistics-focused REIT that appears among the top 3 largest REITs worldwide by market capitalization as of February 2026.
- Verify its specialization in industrial or logistics real estate
- Confirm that it invests primarily in warehouse, distribution center, or logistics facility properties
- Confirm that it is headquartered in the United States

For each company, include at least one reference URL that supports your identification and the provided information.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Company1Info(BaseModel):
    name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list, description="URLs supporting the #1 ranking in 2026 (or latest 2026 list)")
    additional_urls: List[str] = Field(default_factory=list, description="Any other URLs cited for units/HQ/top-5 confirmation")
    units_managed: Optional[str] = None
    headquarters: Optional[str] = None


class Company2Info(BaseModel):
    name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list, description="URLs supporting #1 on MHN 2025 Top Multifamily Developers")
    additional_urls: List[str] = Field(default_factory=list, description="Any other URLs cited for units/development focus/consecutive ranking")
    units_under_construction: Optional[str] = None  # as of June 30, 2025
    development_focus: Optional[str] = None
    consecutive_ranking: Optional[str] = None  # free text claim about multi-year #1 (e.g., 'Yes, 2024 & 2025')


class Company3Info(BaseModel):
    name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list, description="URLs supporting top-3 REIT by market cap as of Feb 2026")
    additional_urls: List[str] = Field(default_factory=list, description="Any other URLs cited for specialization/portfolio/HQ")
    specialization: Optional[str] = None  # e.g., healthcare, seniors housing
    portfolio_size: Optional[str] = None  # number of properties/communities
    headquarters: Optional[str] = None    # city/state or country statement


class Company4Info(BaseModel):
    name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list, description="URLs supporting top-3 REIT by market cap as of Feb 2026")
    additional_urls: List[str] = Field(default_factory=list, description="Any other URLs cited for specialization/property types/HQ")
    specialization: Optional[str] = None  # industrial/logistics
    property_type: Optional[str] = None   # warehouse/distribution/logistics facilities
    headquarters: Optional[str] = None    # city/state or country statement


class RealEstateExtraction(BaseModel):
    company1: Optional[Company1Info] = None
    company2: Optional[Company2Info] = None
    company3: Optional[Company3Info] = None
    company4: Optional[Company4Info] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_companies() -> str:
    return """
Extract the four required companies/entities exactly as presented in the answer. Return all fields strictly from the answer text without inventing.

For each company, extract the following JSON structure. If a field is missing in the answer, return null (for strings) or an empty array (for URL lists).

company1:
- name: The company that ranks #1 as the largest property management company in the US as of 2026.
- reference_urls: Array of URLs the answer cites to prove the #1 ranking in 2026 (e.g., industry ranking pages). Extract actual URLs only.
- additional_urls: Array of any other URLs the answer cites for this company (e.g., company profile, Wikipedia, news, or ranking pages that help units/HQ/top-5).
- units_managed: The number of units the answer claims this company manages in the USA (keep formatting as in the answer; e.g., '1,234,567', 'about 800k', '≈ 500,000').
- headquarters: The HQ city and state as written in the answer (e.g., 'Irvine, CA').

company2:
- name: The company ranked #1 on Multi-Housing News' 2025 Top Multifamily Developers list.
- reference_urls: URLs the answer cites to prove #1 on MHN 2025 Top Multifamily Developers.
- additional_urls: Any other URLs the answer cites for this company.
- units_under_construction: The approximate number of units under construction as of June 30, 2025 (exactly as presented).
- development_focus: The primary type of development projects in its active pipeline (e.g., 'market-rate', 'student housing', 'senior housing', 'affordable').
- consecutive_ranking: The answer's statement about whether the company has held #1 for multiple consecutive years (e.g., 'Yes, 2024 and 2025', 'No', 'Unknown').

company3:
- name: The healthcare-focused REIT that appears among the top 3 largest REITs worldwide by market capitalization as of Feb 2026.
- reference_urls: URLs the answer cites to prove top-3 global REIT by market cap as of Feb 2026.
- additional_urls: Any other URLs the answer cites for this company.
- specialization: The answer's statement about its focus in healthcare real estate (ideally mentions seniors housing / senior living).
- portfolio_size: The size of its portfolio (number of properties/communities) as claimed.
- headquarters: The HQ location as written (e.g., 'Toledo, OH, USA').

company4:
- name: The industrial/logistics-focused REIT that appears among the top 3 largest REITs worldwide by market capitalization as of Feb 2026.
- reference_urls: URLs the answer cites to prove top-3 global REIT by market cap as of Feb 2026.
- additional_urls: Any other URLs the answer cites for this company.
- specialization: The answer's statement about its focus in industrial/logistics real estate.
- property_type: The answer's statement confirming it invests primarily in warehouses, distribution centers, or logistics facilities.
- headquarters: The HQ location as written (e.g., 'San Francisco, CA, USA').

Rules:
- Extract only URLs explicitly present in the answer (including markdown links). Do not infer or fabricate.
- Keep all numbers and phrases as free text exactly as in the answer (do not normalize).
- If a field is not provided, use null (strings) or [] (URL lists).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def combine_sources(*lists: List[str]) -> List[str]:
    """Combine and de-duplicate multiple URL lists while preserving order."""
    seen = set()
    combined: List[str] = []
    for lst in lists:
        for u in lst or []:
            if not isinstance(u, str):
                continue
            url = u.strip()
            if not url:
                continue
            if url not in seen:
                seen.add(url)
                combined.append(url)
    return combined


def has_any_url(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls)


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_company_1(evaluator: Evaluator, parent_node, info: Optional[Company1Info]) -> None:
    node = evaluator.add_parallel(
        id="company_1_largest_property_manager",
        desc="Identify the company that ranks #1 as the largest property management company in the United States as of 2026.",
        parent=parent_node,
        critical=False
    )

    name = (info.name or "").strip() if info else ""
    ref_urls = info.reference_urls if info else []
    add_urls = info.additional_urls if info else []
    all_urls = combine_sources(ref_urls, add_urls)

    # Existence / gating (critical)
    evaluator.add_custom_node(
        result=bool(name),
        id="company_1_name_provided",
        desc="Company 1 name is provided in the answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_any_url(ref_urls),
        id="company_1_sources_provided",
        desc="At least one reference URL is provided for Company 1",
        parent=node,
        critical=True
    )

    # Optional value gating for subfacts (critical to avoid unguided checks)
    evaluator.add_custom_node(
        result=bool(info and info.units_managed and info.units_managed.strip()),
        id="company_1_units_value_present",
        desc="Company 1 units managed value is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info and info.headquarters and info.headquarters.strip()),
        id="company_1_headquarters_value_present",
        desc="Company 1 headquarters value is provided",
        parent=node,
        critical=True
    )

    # Leaves per rubric
    leaf_rank = evaluator.add_leaf(
        id="company_1_reference_url",
        desc="Provide a reference URL that verifies the company's #1 ranking as the largest US property management company in 2026.",
        parent=node,
        critical=True
    )
    leaf_units = evaluator.add_leaf(
        id="company_1_units_managed",
        desc="Provide the number of units managed by this company in the USA.",
        parent=node,
        critical=True
    )
    leaf_hq = evaluator.add_leaf(
        id="company_1_headquarters_location",
        desc="Provide the headquarters location (city and state) of this company.",
        parent=node,
        critical=True
    )
    leaf_top5 = evaluator.add_leaf(
        id="company_1_ranking_consistency",
        desc="Verify that this company appears in the top 5 largest property management companies.",
        parent=node,
        critical=True
    )

    claims_and_sources = [
        (
            f"As of 2026, {name} is ranked #1 as the largest property management (or apartment management) company in the United States.",
            ref_urls if has_any_url(ref_urls) else all_urls,
            leaf_rank,
            "Rely on the cited ranking page(s) for 2026. Accept equivalent phrasing such as 'apartment managers' if it clearly refers to U.S. property management scale."
        ),
        (
            f"{name} manages approximately {info.units_managed if info else ''} residential units in the United States.",
            all_urls,
            leaf_units,
            "Match the claimed units as closely as possible. Allow rounding, commas, and approximate phrasing such as 'about' or 'approximately'. If a range is provided, treat it as approximate."
        ),
        (
            f"The headquarters of {name} is located in {info.headquarters if info else ''}.",
            all_urls,
            leaf_hq,
            "Confirm the HQ city and state as claimed. Minor variations like abbreviations (e.g., 'CA' vs 'California') are acceptable."
        ),
        (
            f"{name} appears within the top 5 largest property management (or apartment management) companies in the United States.",
            ref_urls if has_any_url(ref_urls) else all_urls,
            leaf_top5,
            "Use the ranking source(s). Presence at #1 necessarily satisfies top-5, but still confirm via the provided list."
        )
    ]

    await evaluator.batch_verify(claims_and_sources)


async def verify_company_2(evaluator: Evaluator, parent_node, info: Optional[Company2Info]) -> None:
    node = evaluator.add_parallel(
        id="company_2_multifamily_developer",
        desc="Identify the company that ranked #1 on Multi-Housing News' 2025 Top Multifamily Developers list.",
        parent=parent_node,
        critical=False
    )

    name = (info.name or "").strip() if info else ""
    ref_urls = info.reference_urls if info else []
    add_urls = info.additional_urls if info else []
    all_urls = combine_sources(ref_urls, add_urls)

    # Existence / gating (critical)
    evaluator.add_custom_node(
        result=bool(name),
        id="company_2_name_provided",
        desc="Company 2 name is provided in the answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_any_url(ref_urls),
        id="company_2_sources_provided",
        desc="At least one reference URL is provided for Company 2",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info and info.units_under_construction and info.units_under_construction.strip()),
        id="company_2_units_value_present",
        desc="Company 2 'units under construction' value (as of June 30, 2025) is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info and info.development_focus and info.development_focus.strip()),
        id="company_2_dev_focus_present",
        desc="Company 2 development focus is provided",
        parent=node,
        critical=True
    )

    # Leaves
    leaf_rank = evaluator.add_leaf(
        id="company_2_reference_url",
        desc="Provide a reference URL that verifies the company's #1 ranking on the 2025 Top Multifamily Developers list.",
        parent=node,
        critical=True
    )
    leaf_units_uc = evaluator.add_leaf(
        id="company_2_units_under_construction",
        desc="Provide the approximate number of units this company had under construction as of June 30, 2025.",
        parent=node,
        critical=True
    )
    leaf_dev_focus = evaluator.add_leaf(
        id="company_2_development_focus",
        desc="Identify the primary type of development projects in this company's active pipeline.",
        parent=node,
        critical=True
    )
    leaf_consecutive = evaluator.add_leaf(
        id="company_2_consecutive_ranking",
        desc="Verify whether this company has held the #1 developer ranking for multiple consecutive years.",
        parent=node,
        critical=False
    )

    claims_and_sources = [
        (
            f"{name} is ranked #1 on Multi-Housing News' 2025 Top Multifamily Developers list.",
            ref_urls if has_any_url(ref_urls) else all_urls,
            leaf_rank,
            "Verify on MHN's official 'Top Multifamily Developers of 2025' list. Accept minor title variations."
        ),
        (
            f"As of June 30, 2025, {name} had approximately {info.units_under_construction if info else ''} units under construction.",
            all_urls,
            leaf_units_uc,
            "Match the claimed value as closely as possible. Allow rounding and approximate phrasing (e.g., 'about', 'roughly')."
        ),
        (
            f"The primary development focus in {name}'s active pipeline is {info.development_focus if info else ''}.",
            all_urls,
            leaf_dev_focus,
            "Confirm the main segment (e.g., market-rate, student, senior, affordable). Accept close synonyms."
        ),
        (
            f"{name} has held the #1 developer ranking for multiple consecutive years.",
            all_urls,
            leaf_consecutive,
            "Check MHN lists across adjacent years (e.g., 2024 and 2025). If the evidence shows #1 in successive years, pass; otherwise fail."
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)


async def verify_company_3(evaluator: Evaluator, parent_node, info: Optional[Company3Info]) -> None:
    node = evaluator.add_parallel(
        id="company_3_healthcare_reit",
        desc="Identify the healthcare-focused REIT that appears among the top 3 largest REITs worldwide by market capitalization as of February 2026.",
        parent=parent_node,
        critical=False
    )

    name = (info.name or "").strip() if info else ""
    ref_urls = info.reference_urls if info else []
    add_urls = info.additional_urls if info else []
    all_urls = combine_sources(ref_urls, add_urls)

    # Existence / gating
    evaluator.add_custom_node(
        result=bool(name),
        id="company_3_name_provided",
        desc="Company 3 name is provided in the answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_any_url(ref_urls),
        id="company_3_sources_provided",
        desc="At least one reference URL is provided for Company 3",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info and info.specialization and info.specialization.strip()),
        id="company_3_specialization_present",
        desc="Company 3 specialization is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info and info.portfolio_size and info.portfolio_size.strip()),
        id="company_3_portfolio_size_present",
        desc="Company 3 portfolio size is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info and info.headquarters and info.headquarters.strip()),
        id="company_3_headquarters_present",
        desc="Company 3 headquarters info is provided",
        parent=node,
        critical=True
    )

    # Leaves
    leaf_rank = evaluator.add_leaf(
        id="company_3_reference_url",
        desc="Provide a reference URL that verifies this REIT's position in the top 3 largest REITs by market cap as of February 2026.",
        parent=node,
        critical=True
    )
    leaf_spec = evaluator.add_leaf(
        id="company_3_specialization",
        desc="Verify that this REIT specializes in healthcare real estate, particularly seniors housing or senior living communities.",
        parent=node,
        critical=True
    )
    leaf_portfolio = evaluator.add_leaf(
        id="company_3_portfolio_size",
        desc="Provide information about the size of this REIT's portfolio in terms of number of properties or communities.",
        parent=node,
        critical=True
    )
    leaf_hq = evaluator.add_leaf(
        id="company_3_headquarters",
        desc="Verify that this REIT is headquartered in the United States.",
        parent=node,
        critical=True
    )

    claims_and_sources = [
        (
            f"As of February 2026, {name} appears among the top three largest REITs worldwide by market capitalization.",
            ref_urls if has_any_url(ref_urls) else all_urls,
            leaf_rank,
            "Use credible market-cap ranking pages that state 'as of February 2026' (or closest available 2026 timestamp)."
        ),
        (
            f"{name} specializes in healthcare real estate, particularly seniors housing or senior living communities.",
            all_urls,
            leaf_spec,
            "Look for explicit statements about healthcare focus and seniors housing/senior living in company profiles, investor materials, or reputable summaries."
        ),
        (
            f"{name}'s portfolio size is approximately {info.portfolio_size if info else ''}.",
            all_urls,
            leaf_portfolio,
            "Verify the portfolio size figure (number of properties/communities). Allow rounding and phrasing like 'over' or 'approximately'."
        ),
        (
            f"{name} is headquartered in the United States (e.g., {info.headquarters if info else ''}).",
            all_urls,
            leaf_hq,
            "Confirm that the HQ location is within the United States. City and state details can support this."
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)


async def verify_company_4(evaluator: Evaluator, parent_node, info: Optional[Company4Info]) -> None:
    node = evaluator.add_parallel(
        id="company_4_industrial_logistics_reit",
        desc="Identify the industrial/logistics-focused REIT that appears among the top 3 largest REITs worldwide by market capitalization as of February 2026.",
        parent=parent_node,
        critical=False
    )

    name = (info.name or "").strip() if info else ""
    ref_urls = info.reference_urls if info else []
    add_urls = info.additional_urls if info else []
    all_urls = combine_sources(ref_urls, add_urls)

    # Existence / gating
    evaluator.add_custom_node(
        result=bool(name),
        id="company_4_name_provided",
        desc="Company 4 name is provided in the answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_any_url(ref_urls),
        id="company_4_sources_provided",
        desc="At least one reference URL is provided for Company 4",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info and info.specialization and info.specialization.strip()),
        id="company_4_specialization_present",
        desc="Company 4 specialization is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info and info.property_type and info.property_type.strip()),
        id="company_4_property_type_present",
        desc="Company 4 property type statement is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info and info.headquarters and info.headquarters.strip()),
        id="company_4_headquarters_present",
        desc="Company 4 headquarters info is provided",
        parent=node,
        critical=True
    )

    # Leaves
    leaf_rank = evaluator.add_leaf(
        id="company_4_reference_url",
        desc="Provide a reference URL that verifies this REIT's position in the top 3 largest REITs by market cap as of February 2026.",
        parent=node,
        critical=True
    )
    leaf_spec = evaluator.add_leaf(
        id="company_4_specialization",
        desc="Verify that this REIT specializes in industrial or logistics real estate, including warehouses and distribution centers.",
        parent=node,
        critical=True
    )
    leaf_prop_type = evaluator.add_leaf(
        id="company_4_property_type",
        desc="Confirm that this REIT invests primarily in warehouse, distribution center, or logistics facility properties.",
        parent=node,
        critical=True
    )
    leaf_hq = evaluator.add_leaf(
        id="company_4_headquarters",
        desc="Verify that this REIT is headquartered in the United States.",
        parent=node,
        critical=True
    )

    claims_and_sources = [
        (
            f"As of February 2026, {name} appears among the top three largest REITs worldwide by market capitalization.",
            ref_urls if has_any_url(ref_urls) else all_urls,
            leaf_rank,
            "Use credible market-cap ranking pages that state 'as of February 2026' (or closest available 2026 timestamp)."
        ),
        (
            f"{name} specializes in industrial or logistics real estate.",
            all_urls,
            leaf_spec,
            "Look for explicit statements about industrial/logistics specialization in company profiles or investor materials."
        ),
        (
            f"{name} primarily invests in warehouses, distribution centers, and/or logistics facilities.",
            all_urls,
            leaf_prop_type,
            "Confirm emphasis on warehouses, distribution centers, or similar logistics properties."
        ),
        (
            f"{name} is headquartered in the United States (e.g., {info.headquarters if info else ''}).",
            all_urls,
            leaf_hq,
            "Confirm that the HQ location is within the United States."
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the 'real_estate_companies_2026' task and return a structured summary.
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
        default_model=model
    )

    # Extract structured info from the answer
    extracted: RealEstateExtraction = await evaluator.extract(
        prompt=prompt_extract_companies(),
        template_class=RealEstateExtraction,
        extraction_name="extracted_companies"
    )

    # Optional custom info
    evaluator.add_custom_info(
        info={"as_of_date_context": "February/March 2026 windows referenced in task; verification relies on cited URLs"},
        info_type="context",
        info_name="temporal_context"
    )

    # Build verification tree according to rubric
    await asyncio.gather(
        verify_company_1(evaluator, root, extracted.company1),
        verify_company_2(evaluator, root, extracted.company2),
        verify_company_3(evaluator, root, extracted.company3),
        verify_company_4(evaluator, root, extracted.company4),
    )

    return evaluator.get_summary()