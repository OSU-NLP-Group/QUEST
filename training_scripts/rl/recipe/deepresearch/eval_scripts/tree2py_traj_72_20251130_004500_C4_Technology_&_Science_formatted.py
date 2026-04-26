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
TASK_ID = "att_5g_midband_nov2025"
TASK_DESCRIPTION = (
    "In November 2025, AT&T announced a major nationwide 5G network enhancement through the rapid deployment of "
    "mid-band spectrum. Based on official announcements and reliable sources, provide the following specific details "
    "about this deployment: (1) What frequency (in GHz) of mid-band spectrum was deployed? (2) From which company did "
    "AT&T acquire this spectrum? (3) Approximately how many cell sites received this spectrum upgrade? (4) How many "
    "cities across the United States received enhanced 5G coverage from this deployment? (5) How many U.S. states were "
    "covered by this deployment, and which states (if any) were explicitly excluded? (6) What percentage improvement in "
    "download speeds can mobility customers expect (up to what percentage)? (7) In which month and year was this "
    "deployment completed, and approximately how long did the deployment process take? (8) Provide at least one official "
    "or authoritative reference URL that documents this announcement."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ATTMidBandDetailsExtraction(BaseModel):
    """
    Structured extraction of AT&T November 2025 mid-band 5G deployment details as stated in the agent's answer.
    All fields are extracted as strings to maximize robustness to formatting and ranges (e.g., '3.45–3.55 GHz', 'about 50,000').
    """
    frequency_ghz: Optional[str] = None  # e.g., "3.45 GHz", "3.45–3.55 GHz"
    spectrum_source_company: Optional[str] = None  # e.g., "Company X"
    cell_sites_upgraded: Optional[str] = None  # e.g., "50,000"
    cities_covered: Optional[str] = None  # e.g., "500"
    states_covered_count: Optional[str] = None  # e.g., "50"
    excluded_states_text: Optional[str] = None  # e.g., "None", "Alaska and Hawaii", "no states excluded"
    excluded_states: List[str] = Field(default_factory=list)  # explicit list of excluded state names mentioned
    speed_improvement_pct: Optional[str] = None  # e.g., "up to 75%"
    completion_month_year: Optional[str] = None  # e.g., "December 2025"
    deployment_duration: Optional[str] = None  # e.g., "60 days", "2 months"
    reference_urls: List[str] = Field(default_factory=list)  # authoritative URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_att_midband_details() -> str:
    return """
    Extract the following details exactly as stated in the answer text about AT&T's nationwide mid-band 5G deployment announced in November 2025.

    Return a JSON object with these fields:
    1. frequency_ghz: The mid-band spectrum frequency deployed, expressed in GHz (string). If a range is given (e.g., 3.45–3.55 GHz), include the full range string. If not mentioned, return null.
    2. spectrum_source_company: The company from which AT&T acquired/obtained the spectrum (string). If not mentioned, return null.
    3. cell_sites_upgraded: The approximate number of cell sites that received the spectrum upgrade (string). If not mentioned, return null.
    4. cities_covered: The number of U.S. cities that received enhanced 5G coverage (string). If not mentioned, return null.
    5. states_covered_count: The number of U.S. states covered by the deployment (string). If not mentioned, return null.
    6. excluded_states_text: If the answer mentions that certain states were excluded or explicitly states that none were excluded, capture that phrase exactly as written (e.g., "None", "no states excluded", or a phrase naming specific states). If not mentioned, return null.
    7. excluded_states: An array listing any explicitly named excluded U.S. states (e.g., ["Alaska","Hawaii"]). If none were named or the answer explicitly states none were excluded, return an empty array.
    8. speed_improvement_pct: The expected improvement in download speeds for mobility customers, including the maximum (e.g., "up to 75%"). Keep the text as-is (string). If not mentioned, return null.
    9. completion_month_year: The stated month and year when the deployment was completed (string, e.g., "December 2025"). If not mentioned, return null.
    10. deployment_duration: The approximate duration of the deployment process (string, e.g., "60 days", "about 2 months"). If not mentioned, return null.
    11. reference_urls: A list of authoritative/official reference URLs (e.g., att.com newsroom, investor.att.com, fcc.gov, press releases, reputable industry outlets) explicitly mentioned in the answer that document this announcement and support the details. Extract only actual URLs presented in the answer. If none are provided, return an empty list.

    Important:
    - Do not invent any information. If a required field is not present in the answer text, return null (or empty list for URLs).
    - Keep numbers as strings exactly as presented (e.g., "approx. 50,000", "up to 75%").
    - For URLs, extract only valid explicit URLs found in the answer (including those embedded in markdown).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_text(val: Optional[str]) -> bool:
    return bool(val) and isinstance(val, str) and val.strip() != ""

def _contains_ghz(val: Optional[str]) -> bool:
    if not _has_text(val):
        return False
    s = val.lower()
    return "ghz" in s or "gigahertz" in s

def _excluded_states_provided(excluded_text: Optional[str], excluded_list: List[str]) -> bool:
    if _has_text(excluded_text):
        return True
    return len(excluded_list) > 0


# --------------------------------------------------------------------------- #
# Verification functions (builds the tree and performs checks)                #
# --------------------------------------------------------------------------- #
async def build_and_verify(
    evaluator: Evaluator,
    parent_node,
    details: ATTMidBandDetailsExtraction
) -> None:
    """
    Build verification tree (critical parallel under the parent) and verify each requested item.
    """

    # 0) Authoritative reference URL parent (do first so we can use it as a prerequisite for other source-based checks)
    ref_parent = evaluator.add_parallel(
        id="Authoritative_Reference_URL_Provided",
        desc="Provides at least one official or authoritative/reliable reference URL that documents the announcement",
        parent=parent_node,
        critical=True
    )

    # 0.1) Existence of at least one reference URL
    ref_provided_leaf = evaluator.add_custom_node(
        result=(len(details.reference_urls) > 0),
        id="reference_urls_exist",
        desc="At least one reference URL is provided",
        parent=ref_parent,
        critical=True
    )

    # 0.2) Reference URL documents the announcement (supports the deployment details)
    ref_support_leaf = evaluator.add_leaf(
        id="reference_urls_support_announcement",
        desc="Provided reference URL(s) document AT&T's November 2025 nationwide mid-band 5G deployment announcement",
        parent=ref_parent,
        critical=True
    )
    await evaluator.verify(
        claim="This page documents AT&T's nationwide mid-band 5G deployment announced in November 2025.",
        node=ref_support_leaf,
        sources=details.reference_urls,
        additional_instruction=(
            "Accept pages that are official or authoritative (e.g., att.com newsroom/investor sites, FCC.gov, "
            "major reputable industry publications). The page should explicitly mention the nationwide mid-band "
            "5G deployment and be tied to the November 2025 announcement."
        )
    )

    # 1) Spectrum frequency in GHz
    freq_parent = evaluator.add_parallel(
        id="Spectrum_Frequency_Provided_in_GHz",
        desc="Provides the mid-band spectrum frequency deployed and expresses it in GHz units.",
        parent=parent_node,
        critical=True
    )

    # 1.1) Existence with GHz unit
    freq_exist_leaf = evaluator.add_custom_node(
        result=_contains_ghz(details.frequency_ghz),
        id="frequency_in_ghz_present",
        desc="Frequency is provided and expressed in GHz",
        parent=freq_parent,
        critical=True
    )

    # 1.2) Supported by reference URL(s)
    freq_support_leaf = evaluator.add_leaf(
        id="frequency_supported_by_sources",
        desc="The stated frequency is supported by authoritative source(s)",
        parent=freq_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"AT&T deployed mid-band spectrum at {details.frequency_ghz}.",
        node=freq_support_leaf,
        sources=details.reference_urls,
        additional_instruction=(
            "Verify that the cited page(s) explicitly state the deployed mid-band spectrum frequency (in GHz or equivalent). "
            "Allow ranges if the answer used a range. Minor formatting variations are acceptable."
        ),
        extra_prerequisites=[ref_provided_leaf]
    )

    # 2) Spectrum source company identified
    source_parent = evaluator.add_parallel(
        id="Spectrum_Source_Company_Identified",
        desc="Identifies the company from which AT&T acquired/obtained the spectrum.",
        parent=parent_node,
        critical=True
    )

    source_exist_leaf = evaluator.add_custom_node(
        result=_has_text(details.spectrum_source_company),
        id="spectrum_source_company_present",
        desc="Spectrum source company is provided",
        parent=source_parent,
        critical=True
    )

    source_support_leaf = evaluator.add_leaf(
        id="spectrum_source_company_supported",
        desc="The stated spectrum source company is supported by authoritative source(s)",
        parent=source_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"AT&T acquired/obtained this mid-band spectrum from {details.spectrum_source_company}.",
        node=source_support_leaf,
        sources=details.reference_urls,
        additional_instruction=(
            "Verify that the cited page(s) explicitly indicate the company or entity from which AT&T acquired or "
            "obtained the mid-band spectrum referenced in the November 2025 deployment announcement."
        ),
        extra_prerequisites=[ref_provided_leaf]
    )

    # 3) Cell sites upgraded count
    sites_parent = evaluator.add_parallel(
        id="Cell_Sites_Upgraded_Count_Provided",
        desc="Provides the approximate number of cell sites that received the spectrum upgrade.",
        parent=parent_node,
        critical=True
    )

    sites_exist_leaf = evaluator.add_custom_node(
        result=_has_text(details.cell_sites_upgraded),
        id="cell_sites_upgraded_present",
        desc="Approximate number of upgraded cell sites is provided",
        parent=sites_parent,
        critical=True
    )

    sites_support_leaf = evaluator.add_leaf(
        id="cell_sites_upgraded_supported",
        desc="The stated cell sites upgraded count is supported by authoritative source(s)",
        parent=sites_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"Approximately {details.cell_sites_upgraded} cell sites received the mid-band spectrum upgrade.",
        node=sites_support_leaf,
        sources=details.reference_urls,
        additional_instruction=(
            "Verify that the page(s) explicitly state the approximate number of cell sites upgraded. Allow phrasing "
            "such as 'about', 'over', or 'approximately', and minor numeric formatting variations."
        ),
        extra_prerequisites=[ref_provided_leaf]
    )

    # 4) Cities covered count
    cities_parent = evaluator.add_parallel(
        id="Cities_Covered_Count_Provided",
        desc="Provides the number of U.S. cities that received enhanced 5G coverage from the deployment.",
        parent=parent_node,
        critical=True
    )

    cities_exist_leaf = evaluator.add_custom_node(
        result=_has_text(details.cities_covered),
        id="cities_covered_present",
        desc="Number of covered cities is provided",
        parent=cities_parent,
        critical=True
    )

    cities_support_leaf = evaluator.add_leaf(
        id="cities_covered_supported",
        desc="The stated number of covered cities is supported by authoritative source(s)",
        parent=cities_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The deployment provided enhanced 5G coverage to {details.cities_covered} cities in the U.S.",
        node=cities_support_leaf,
        sources=details.reference_urls,
        additional_instruction=(
            "Verify that the page(s) explicitly state the number of cities covered by the November 2025 deployment. "
            "Accept approximate wording if used in the answer."
        ),
        extra_prerequisites=[ref_provided_leaf]
    )

    # 5) States covered and excluded
    states_parent = evaluator.add_parallel(
        id="States_Covered_and_Any_Excluded_States_Provided",
        desc="Provides how many U.S. states were covered and names any states explicitly excluded (or that none were excluded).",
        parent=parent_node,
        critical=True
    )

    states_count_exist_leaf = evaluator.add_custom_node(
        result=_has_text(details.states_covered_count),
        id="states_covered_count_present",
        desc="Number of U.S. states covered is provided",
        parent=states_parent,
        critical=True
    )

    excluded_states_exist_leaf = evaluator.add_custom_node(
        result=_excluded_states_provided(details.excluded_states_text, details.excluded_states),
        id="excluded_states_info_present",
        desc="Information about excluded states (or explicit 'none excluded') is provided",
        parent=states_parent,
        critical=True
    )

    states_count_support_leaf = evaluator.add_leaf(
        id="states_covered_count_supported",
        desc="The stated number of covered U.S. states is supported by authoritative source(s)",
        parent=states_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The deployment covered {details.states_covered_count} U.S. states.",
        node=states_count_support_leaf,
        sources=details.reference_urls,
        additional_instruction=(
            "Verify that the page(s) state the number of U.S. states covered by the deployment."
        ),
        extra_prerequisites=[ref_provided_leaf]
    )

    excluded_statement = (
        f"States explicitly excluded: {', '.join(details.excluded_states)}."
        if details.excluded_states
        else "No U.S. states were explicitly excluded from this deployment."
        if _has_text(details.excluded_states_text) and ("none" in details.excluded_states_text.lower() or "no" in details.excluded_states_text.lower())
        else details.excluded_states_text or ""
    )

    excluded_support_leaf = evaluator.add_leaf(
        id="excluded_states_supported",
        desc="The stated exclusion (which states were excluded, or none) is supported by authoritative source(s)",
        parent=states_parent,
        critical=True
    )
    await evaluator.verify(
        claim=excluded_statement,
        node=excluded_support_leaf,
        sources=details.reference_urls,
        additional_instruction=(
            "Verify whether the page(s) state that certain states were excluded, or that none were excluded. "
            "If the answer claims 'none excluded', confirm that coverage is nationwide/all states."
        ),
        extra_prerequisites=[ref_provided_leaf]
    )

    # 6) Download speed improvement percentage
    speed_parent = evaluator.add_parallel(
        id="Download_Speed_Improvement_Percentage_Provided",
        desc="Provides the expected download-speed improvement for mobility customers, including the stated maximum (up to X%).",
        parent=parent_node,
        critical=True
    )

    speed_exist_leaf = evaluator.add_custom_node(
        result=_has_text(details.speed_improvement_pct),
        id="speed_improvement_present",
        desc="Expected download-speed improvement percentage (including 'up to') is provided",
        parent=speed_parent,
        critical=True
    )

    speed_support_leaf = evaluator.add_leaf(
        id="speed_improvement_supported",
        desc="The stated download-speed improvement percentage is supported by authoritative source(s)",
        parent=speed_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"Mobility customers can expect {details.speed_improvement_pct} improvement in download speeds.",
        node=speed_support_leaf,
        sources=details.reference_urls,
        additional_instruction=(
            "Verify that the page(s) explicitly mention the expected improvement magnitude for download speeds for "
            "mobility customers, e.g., 'up to X%'. Minor phrasing variations are acceptable."
        ),
        extra_prerequisites=[ref_provided_leaf]
    )

    # 7) Completion month/year and deployment duration
    completion_parent = evaluator.add_parallel(
        id="Completion_MonthYear_and_Deployment_Duration_Provided",
        desc="States the month and year the deployment was completed and provides an approximate duration of the deployment process.",
        parent=parent_node,
        critical=True
    )

    completion_exist_leaf = evaluator.add_custom_node(
        result=_has_text(details.completion_month_year),
        id="completion_month_year_present",
        desc="Completion month and year are provided",
        parent=completion_parent,
        critical=True
    )

    duration_exist_leaf = evaluator.add_custom_node(
        result=_has_text(details.deployment_duration),
        id="deployment_duration_present",
        desc="Approximate deployment duration is provided",
        parent=completion_parent,
        critical=True
    )

    completion_support_leaf = evaluator.add_leaf(
        id="completion_month_year_supported",
        desc="The stated completion month/year is supported by authoritative source(s)",
        parent=completion_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The deployment was completed in {details.completion_month_year}.",
        node=completion_support_leaf,
        sources=details.reference_urls,
        additional_instruction=(
            "Verify that the page(s) state the month and year in which the deployment was completed."
        ),
        extra_prerequisites=[ref_provided_leaf]
    )

    duration_support_leaf = evaluator.add_leaf(
        id="deployment_duration_supported",
        desc="The stated deployment duration is supported by authoritative source(s)",
        parent=completion_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The deployment process took approximately {details.deployment_duration}.",
        node=duration_support_leaf,
        sources=details.reference_urls,
        additional_instruction=(
            "Verify that the page(s) give an approximate duration of the deployment process. "
            "Allow phrasing such as 'about', 'approximately', and ranges."
        ),
        extra_prerequisites=[ref_provided_leaf]
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
    Evaluate an agent's answer for the AT&T November 2025 mid-band 5G deployment details.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregator
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

    # Add the critical top-level rubric node under root
    top_node = evaluator.add_parallel(
        id="ATandT_5G_MidBand_Deployment_Details",
        desc="Answer provides all requested details about AT&T's November 2025 nationwide 5G mid-band deployment and cites at least one authoritative source.",
        parent=root,
        critical=True
    )

    # Extract structured information from the answer
    details = await evaluator.extract(
        prompt=prompt_extract_att_midband_details(),
        template_class=ATTMidBandDetailsExtraction,
        extraction_name="att_midband_details"
    )

    # Optional: record custom info about the timeframe
    evaluator.add_custom_info(
        info={"announcement_month": "November", "announcement_year": 2025},
        info_type="context",
        info_name="announcement_timeframe"
    )

    # Build verification tree and perform checks
    await build_and_verify(evaluator, top_node, details)

    # Return structured evaluation summary
    return evaluator.get_summary()