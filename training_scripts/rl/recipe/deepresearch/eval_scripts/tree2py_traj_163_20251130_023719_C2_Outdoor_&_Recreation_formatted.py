import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "access_pass_ca_planning"
TASK_DESCRIPTION = (
    "A U.S. permanent resident who receives Social Security Disability Income (SSDI) is planning to visit national "
    "parks in California during December 2025. They want to apply for an Access Pass and need to know: (1) whether "
    "their SSDI documentation qualifies as acceptable federal documentation for obtaining an Access Pass, (2) whether "
    "a digital Access Pass is available for use in December 2025, and (3) which national parks in California are among "
    "the 11 parks that will charge a $100 per-person surcharge to nonresidents starting January 1, 2026. Please "
    "provide this information with supporting references."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AccessPassExtraction(BaseModel):
    # Access Pass eligibility and SSDI documentation
    eligibility_statement: Optional[str] = None
    eligibility_sources: List[str] = Field(default_factory=list)

    ssdi_acceptability_statement: Optional[str] = None
    ssdi_sources: List[str] = Field(default_factory=list)

    # Digital Access Pass status for December 2025
    digital_status_dec2025: Optional[str] = None
    digital_sources: List[str] = Field(default_factory=list)

    # Physical Access Pass options as of Dec 2025
    physical_pass_options_dec2025: Optional[str] = None
    physical_pass_sources: List[str] = Field(default_factory=list)

    # Surcharge policy and CA parks
    surcharge_core_facts: Optional[str] = None
    surcharge_sources: List[str] = Field(default_factory=list)

    ca_parks_list: List[str] = Field(default_factory=list)
    ca_parks_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_access_pass_info() -> str:
    return """
    Extract the specific claims and URLs provided in the answer related to the following topics. Return exactly and only what the answer explicitly states.

    1) Access Pass eligibility and SSDI documentation:
       - eligibility_statement: The answer's statement regarding who can get an Access Pass (e.g., that it is free/available to U.S. citizens or permanent residents with permanent disabilities). Extract as written.
       - eligibility_sources: A list of URLs cited for the eligibility statement (e.g., NPS/USGS official pages). Include only valid URLs that appear in the answer.

       - ssdi_acceptability_statement: The answer's statement about whether SSDI documentation qualifies as acceptable federal documentation for obtaining an Access Pass. Extract as written (for example, if the answer says SSDI award letter or a benefits verification letter is accepted).
       - ssdi_sources: A list of URLs cited for the SSDI documentation acceptability claim. Include only valid URLs that appear in the answer.

    2) Digital Access Pass availability for December 2025:
       - digital_status_dec2025: The answer's statement about whether a digital Access Pass can be used in December 2025 (e.g., 'Not available until January 2026' or 'Available in Dec 2025'). Extract the statement as written.
       - digital_sources: A list of URLs cited that support the digital Access Pass availability status. Include only valid URLs that appear in the answer.

    3) Physical Access Pass options as of December 2025:
       - physical_pass_options_dec2025: The answer's statement about how a physical Access Pass can be obtained (e.g., 'in person at federal recreation sites' or 'ordered from the USGS Online Store'). Extract as written.
       - physical_pass_sources: A list of URLs cited for the physical Access Pass options. Include only valid URLs that appear in the answer.

    4) Surcharge policy and California parks:
       - surcharge_core_facts: The answer's statement of the core policy facts about the $100 per-person surcharge to nonresidents starting January 1, 2026, and that it applies to exactly 11 national parks. Extract as written.
       - surcharge_sources: A list of URLs cited for the surcharge policy. Include only valid URLs that appear in the answer.

       - ca_parks_list: The list of California national parks that the answer claims are among the 11 parks that will charge the $100 per-person nonresident surcharge starting January 1, 2026. Return an array of park names exactly as written in the answer.
       - ca_parks_sources: A list of URLs cited specifically to support the identification of which California parks are in the 11-park group. Include only valid URLs that appear in the answer.

    GENERAL RULES:
    - Do not invent any text or URLs. Only extract what appears in the answer.
    - If a field is not addressed in the answer, set it to null (for strings) or an empty list (for arrays of URLs).
    - Extract full URLs, including protocol. If a URL is missing protocol, prepend 'http://'.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_sources(*lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for u in lst or []:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def _is_negative_digital_status(text: Optional[str]) -> bool:
    if not text:
        return True  # Default to "not available" per given constraints context
    t = text.lower()
    negative_markers = [
        "not available", "unavailable", "no digital", "not offered", "starts in january 2026",
        "starting january 2026", "until january 2026", "after january 2026", "from january 2026",
        "not until january", "available in january 2026"
    ]
    return any(m in t for m in negative_markers)


def _parks_list_to_str(parks: List[str]) -> str:
    if not parks:
        return ""
    return ", ".join([p.strip() for p in parks if p and p.strip()])


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: AccessPassExtraction) -> None:
    root = evaluator.root

    # Access_Pass_Eligibility_and_SSDI_Documentation (critical, parallel)
    node_elig_ssdi = evaluator.add_parallel(
        id="access_pass_eligibility_and_ssdi_documentation",
        desc="Check Access Pass eligibility criteria and whether SSDI documentation is acceptable federal documentation.",
        parent=root,
        critical=True
    )

    # Leaf: Eligibility_Permanent_Disability_And_Status (critical)
    leaf_eligibility = evaluator.add_leaf(
        id="eligibility_permanent_disability_and_status",
        desc="States that Access Pass is free/available to U.S. citizens or permanent residents with permanent disabilities.",
        parent=node_elig_ssdi,
        critical=True
    )
    eligibility_claim = extracted.eligibility_statement or (
        "The Access Pass is available to U.S. citizens or permanent residents with permanent disabilities, and the pass "
        "itself is free (noting that online orders may include a processing fee)."
    )
    eligibility_sources = _safe_sources(extracted.eligibility_sources)
    elig_ins = (
        "Verify using the cited official sources (NPS/USGS) that Access Pass eligibility includes U.S. citizens or "
        "permanent residents with permanent disabilities. If the page mentions the pass is free but notes separate "
        "processing or ordering fees, still consider the claim correct."
    )

    # Leaf: SSDI_Acceptable_Federal_Documentation (critical)
    leaf_ssdi = evaluator.add_leaf(
        id="ssdi_acceptable_federal_documentation",
        desc="States that SSDI documentation qualifies as acceptable federal documentation for obtaining an Access Pass.",
        parent=node_elig_ssdi,
        critical=True
    )
    ssdi_claim = extracted.ssdi_acceptability_statement or (
        "An SSDI (Social Security Disability Insurance) award letter or benefits verification letter qualifies as "
        "acceptable federal documentation for obtaining an Access Pass."
    )
    ssdi_sources = _safe_sources(extracted.ssdi_sources, extracted.eligibility_sources)
    ssdi_ins = (
        "Verify that the official documentation page(s) for the Access Pass list SSDI-related documents "
        "as acceptable proof of permanent disability. Accept reasonable synonyms such as 'Social Security Disability "
        "Insurance benefits verification letter' or 'SSDI award letter'."
    )

    # Digital_Access_Pass_Availability_Dec_2025 (critical, parallel)
    node_digital = evaluator.add_parallel(
        id="digital_access_pass_availability_dec_2025",
        desc="Check whether a digital Access Pass can be used in December 2025 and related options.",
        parent=root,
        critical=True
    )

    # Leaf: Digital_Access_Pass_Status_Dec_2025 (critical)
    leaf_digital_status = evaluator.add_leaf(
        id="digital_access_pass_status_dec_2025",
        desc="Correctly answers whether a digital Access Pass is available for use in December 2025 (expected: not until January 2026).",
        parent=node_digital,
        critical=True
    )
    if _is_negative_digital_status(extracted.digital_status_dec2025):
        digital_claim = "A digital Access Pass is not available for use in December 2025 (it becomes available starting January 2026)."
    else:
        digital_claim = "A digital Access Pass is available for use in December 2025."
    digital_sources = _safe_sources(extracted.digital_sources)
    digital_ins = (
        "Confirm the availability status of a digital Access Pass in December 2025 per the cited official source(s). "
        "If the source states that digital Access Passes begin in January 2026 (or later), then the December 2025 "
        "availability is 'not available'."
    )

    # Leaf: Physical_Access_Pass_Options_Dec_2025 (critical in this script for consistency)
    leaf_physical = evaluator.add_leaf(
        id="physical_access_pass_options_dec_2025",
        desc="States that a physical Access Pass can be obtained in person at federal recreation sites or ordered from the USGS Online Store as of December 2025.",
        parent=node_digital,
        critical=True  # Adjusted to critical to satisfy consistent critical-child constraint and evaluation gating
    )
    physical_claim = extracted.physical_pass_options_dec2025 or (
        "As of December 2025, a physical Access Pass can be obtained in person at federal recreation sites or ordered "
        "from the USGS Online Store."
    )
    physical_sources = _safe_sources(extracted.physical_pass_sources)
    physical_ins = (
        "Verify from the cited official source(s) that as of December 2025, a physical Access Pass can be obtained "
        "in person at federal recreation sites and/or ordered from the USGS Online Store."
    )

    # CA_Parks_In_11_Park_Surcharge_Group (critical, parallel)
    node_surcharge = evaluator.add_parallel(
        id="ca_parks_in_11_park_surcharge_group",
        desc="Identify the California national parks among the 11 parks charging the $100 nonresident surcharge starting Jan 1, 2026.",
        parent=root,
        critical=True
    )

    # Leaf: Surcharge_Policy_Core_Facts (critical)
    leaf_surcharge_core = evaluator.add_leaf(
        id="surcharge_policy_core_facts",
        desc="States the surcharge policy core facts: starting Jan 1, 2026, exactly 11 national parks will charge a $100 per-person surcharge to nonresidents (in addition to standard entrance fees).",
        parent=node_surcharge,
        critical=True
    )
    surcharge_claim = extracted.surcharge_core_facts or (
        "Starting January 1, 2026, exactly 11 national parks will charge a $100 per-person surcharge to nonresidents, "
        "in addition to standard entrance fees."
    )
    surcharge_sources = _safe_sources(extracted.surcharge_sources)
    surcharge_ins = (
        "Verify that the cited source(s) explicitly state the surcharge begins Jan 1, 2026, is $100 per person, applies "
        "to nonresidents, and that exactly 11 national parks are included. If any element is missing or contradicted, mark as not supported."
    )

    # Leaf: Correct_CA_Parks_Selected_From_11_List (critical)
    leaf_ca_parks = evaluator.add_leaf(
        id="correct_ca_parks_selected_from_11_list",
        desc="Selects the correct California park(s) from the provided 11-park surcharge list using the provided park-location constraints, and does not include any parks not in that CA subset.",
        parent=node_surcharge,
        critical=True
    )
    ca_parks_str = _parks_list_to_str(extracted.ca_parks_list)
    ca_parks_claim = (
        f"The California national parks among the 11 surcharge parks are: {ca_parks_str}."
        if ca_parks_str else
        "The California national parks among the 11 surcharge parks are: (none listed)."
    )
    ca_parks_sources = _safe_sources(extracted.ca_parks_sources, extracted.surcharge_sources)
    ca_parks_ins = (
        "Using the cited source(s), verify that every listed park is (a) in California and (b) part of the 11-park "
        "surcharge group beginning Jan 1, 2026. If any listed park is not in California or is not in the 11-park group, "
        "or if required California parks are missing, mark as not supported."
    )

    # Supporting_References_Provided (critical, parallel)
    node_refs = evaluator.add_parallel(
        id="supporting_references_provided",
        desc="Answer includes supporting references for each requested factual area.",
        parent=root,
        critical=True
    )

    # Leaf-equivalent: presence checks via custom nodes (binary judgments)
    evaluator.add_custom_node(
        result=bool(extracted.ssdi_sources or extracted.eligibility_sources),
        id="references_for_ssdi_and_eligibility",
        desc="Provides at least one supporting reference/URL for the SSDI documentation acceptability / Access Pass eligibility claim(s).",
        parent=node_refs,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.digital_sources),
        id="references_for_digital_pass_status",
        desc="Provides at least one supporting reference/URL for the digital Access Pass availability status in December 2025 (and any stated rollout timing).",
        parent=node_refs,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.surcharge_sources or extracted.ca_parks_sources),
        id="references_for_surcharge_and_ca_identification",
        desc="Provides at least one supporting reference/URL for the $100 nonresident surcharge policy and the information used to identify which California parks are in the 11-park group.",
        parent=node_refs,
        critical=True
    )

    # Prepare batch verifications for all non-custom leaves
    claims_and_sources: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = [
        (eligibility_claim, eligibility_sources, leaf_eligibility, elig_ins),
        (ssdi_claim, ssdi_sources, leaf_ssdi, ssdi_ins),
        (digital_claim, digital_sources, leaf_digital_status, digital_ins),
        (physical_claim, physical_sources, leaf_physical, physical_ins),
        (surcharge_claim, surcharge_sources, leaf_surcharge_core, surcharge_ins),
        (ca_parks_claim, ca_parks_sources, leaf_ca_parks, ca_parks_ins),
    ]

    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the Access Pass + California planning task.
    Note: To satisfy framework constraints (critical-parent must have critical-children), the digital cluster's
    'Physical_Access_Pass_Options_Dec_2025' check is treated as critical in this script.
    """
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_access_pass_info(),
        template_class=AccessPassExtraction,
        extraction_name="access_pass_extraction"
    )

    # Record brief custom info about critical-structure adjustment for transparency
    evaluator.add_custom_info(
        info={
            "note": "To comply with the framework's critical-children constraint, "
                    "'Physical_Access_Pass_Options_Dec_2025' is evaluated as critical.",
        },
        info_type="implementation_note",
        info_name="criticality_adjustment"
    )

    # Build verification tree and run verifications
    await build_and_verify_tree(evaluator, extracted)

    # Return the structured summary
    return evaluator.get_summary()