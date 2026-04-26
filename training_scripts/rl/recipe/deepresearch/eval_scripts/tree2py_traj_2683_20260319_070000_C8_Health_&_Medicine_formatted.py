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
TASK_ID = "cms_5star_midwest_4hospitals"
TASK_DESCRIPTION = """
Identify four acute care hospitals that have achieved a 5-star rating in the CMS (Centers for Medicare & Medicaid Services) Overall Hospital Quality Star Rating system as of the most recent publicly available rating period. All four hospitals must be located in the Midwest region, specifically within one or more of the following states: Ohio, Indiana, Pennsylvania, or Michigan.

For each of the four hospitals, provide the following information:

1. Official Hospital Name: The complete official name of the hospital as it is registered with Medicare.
2. Physical Address: The complete street address, city, and state where the hospital is located.
3. Star Rating Confirmation: Explicit confirmation that the hospital currently holds a 5-star CMS Overall Hospital Quality Star Rating.
4. Medicare Care Compare URL: A direct link to the hospital's profile page on Medicare.gov's Care Compare tool (https://www.medicare.gov/care-compare/).

All hospitals identified must be general acute care hospitals (not Veterans Administration hospitals, Department of Defense hospitals, or nursing homes) that are eligible for and have received CMS star ratings. The information provided must be verifiable through official CMS or Medicare.gov sources.
"""

ALLOWED_STATES_FULL = {"ohio", "indiana", "pennsylvania", "michigan"}
STATE_ABBR_TO_FULL = {
    "oh": "ohio",
    "in": "indiana",
    "pa": "pennsylvania",
    "mi": "michigan",
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HospitalItem(BaseModel):
    name: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # Accepts full name or abbr in extraction; will normalize
    zip_code: Optional[str] = None
    star_rating_value: Optional[str] = None  # e.g., "5", "5 stars"
    care_compare_url: Optional[str] = None
    other_urls: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class HospitalsExtraction(BaseModel):
    hospitals: List[HospitalItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hospitals() -> str:
    return """
    Extract up to 6 hospital entries mentioned in the answer. For each entry, return:
    - name: The hospital's official name as referenced (prefer Medicare/Medicare.gov naming if present).
    - street_address: The street address (e.g., "123 Main St", include suite if given).
    - city: The city name.
    - state: The state (accept either full state name like "Ohio" or postal abbreviation like "OH").
    - zip_code: Zip/postal code if present.
    - star_rating_value: The Overall Hospital Quality Star Rating value if stated (e.g., "5", "5 stars"). If not explicitly stated, set null.
    - care_compare_url: The direct Medicare.gov Care Compare URL for the hospital profile, if provided in the answer. Prefer a URL beginning with "https://www.medicare.gov/care-compare/". If multiple Medicare/CMS URLs are provided, place the most direct profile URL here.
    - other_urls: Any additional relevant official CMS/Medicare URLs cited in the answer for this hospital (exclude non-official third-party sites).
    - notes: Any additional contextual notes explicitly stated in the answer (optional).

    Rules:
    - Do not invent URLs. Extract only URLs explicitly present in the answer. If a URL lacks protocol, prepend "http://".
    - Prefer the Medicare.gov Care Compare hospital profile URL for 'care_compare_url'. If none is given, set it to null.
    - Keep names as written in the answer (do not rewrite).
    - If any field is missing, set it to null (or [] for lists).
    - Return a JSON object with key 'hospitals' as an array of these objects.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_state_name(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip().lower()
    if s in STATE_ABBR_TO_FULL:
        return STATE_ABBR_TO_FULL[s]
    return s


def _is_allowed_state(state: Optional[str]) -> bool:
    norm = _normalize_state_name(state)
    return norm in ALLOWED_STATES_FULL if norm else False


def _is_medicare_care_compare_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip().lower()
    return ("medicare.gov" in u) and ("/care-compare" in u)


def _normalize_name(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _hospital_brief(h: HospitalItem) -> Dict[str, Optional[str]]:
    return {
        "name": h.name,
        "city": h.city,
        "state": h.state,
        "care_compare_url": h.care_compare_url,
    }


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_hospital(
    evaluator: Evaluator,
    parent_node,
    hospital: HospitalItem,
    idx_one_based: int,
) -> None:
    """
    Build verification sub-tree for one hospital according to rubric leaves.
    Order of verification ensures the Care Compare URL is checked first to serve as evidence for subsequent checks.
    """
    # Create parent node for this hospital
    hosp_node = evaluator.add_parallel(
        id=f"Hospital_{idx_one_based}",
        desc=f"Hospital #{idx_one_based} satisfies all constraints and required fields are provided.",
        parent=parent_node,
        critical=False,
    )

    # Prepare fields
    name = hospital.name or ""
    street = hospital.street_address or ""
    city = hospital.city or ""
    state = hospital.state or ""
    url = (hospital.care_compare_url or "").strip()
    star_text = hospital.star_rating_value or ""

    # 1) Care Compare URL first (critical)
    cc_url_node = evaluator.add_leaf(
        id=f"H{idx_one_based}_Care_Compare_URL",
        desc=f"Provides a direct Medicare.gov Care Compare URL to the hospital’s profile page corresponding to this hospital.",
        parent=hosp_node,
        critical=True,
    )
    if not _is_medicare_care_compare_url(url):
        # Hard fail if URL missing or obviously not a Medicare Care Compare page
        cc_url_node.score = 0.0
        cc_url_node.status = "failed"
    else:
        cc_url_claim = f"The provided URL leads directly to the Medicare.gov Care Compare hospital profile page for the hospital '{name}' located in {city}, {state} (i.e., not a general search page or unrelated site)."
        await evaluator.verify(
            claim=cc_url_claim,
            node=cc_url_node,
            sources=url,
            additional_instruction="Verify the page is a specific hospital profile on Medicare.gov Care Compare. It should display the hospital's details (name, address, rating). Confirm the hospital identity matches the provided name/location.",
        )

    # 2) Official name (critical)
    name_node = evaluator.add_leaf(
        id=f"H{idx_one_based}_Name",
        desc=f"Provides the official hospital name as registered with Medicare.",
        parent=hosp_node,
        critical=True,
    )
    name_claim = f"On the Medicare.gov Care Compare hospital profile page, the hospital's official name appears as '{name}', allowing for minor case/punctuation variants but referring to the same facility."
    await evaluator.verify(
        claim=name_claim,
        node=name_node,
        sources=url if _is_medicare_care_compare_url(url) else None,
        additional_instruction="Match names leniently (case, punctuation, common abbreviations). Ensure it's the same facility by cross-checking location on the page.",
    )

    # 3) Address complete (critical)
    addr_node = evaluator.add_leaf(
        id=f"H{idx_one_based}_Address_Complete",
        desc=f"Provides the complete physical address (street, city, state).",
        parent=hosp_node,
        critical=True,
    )
    addr_claim = f"The Medicare.gov Care Compare profile shows the hospital address including street '{street}', city '{city}', and state '{state}' (minor formatting or abbreviations allowed)."
    await evaluator.verify(
        claim=addr_claim,
        node=addr_node,
        sources=url if _is_medicare_care_compare_url(url) else None,
        additional_instruction="Accept minor formatting differences (e.g., 'St' vs 'Street', punctuation). All three components (street, city, state) must be present on the page.",
    )

    # 4) State allowed (critical)
    state_node = evaluator.add_leaf(
        id=f"H{idx_one_based}_State_Allowed",
        desc=f"Hospital is located in one of: Ohio, Indiana, Pennsylvania, Michigan.",
        parent=hosp_node,
        critical=True,
    )
    # Compose canonical check text
    state_canon = _normalize_state_name(state) or state
    allowed_list_text = "Ohio, Indiana, Pennsylvania, Michigan"
    state_claim = f"The Medicare.gov Care Compare page shows the hospital is located in the state '{state}', and this state is among the allowed set: {allowed_list_text}."
    await evaluator.verify(
        claim=state_claim,
        node=state_node,
        sources=url if _is_medicare_care_compare_url(url) else None,
        additional_instruction="Interpret common postal abbreviations (OH, IN, PA, MI) as Ohio, Indiana, Pennsylvania, Michigan respectively.",
    )

    # 5) Facility type: general acute care hospital (critical)
    type_node = evaluator.add_leaf(
        id=f"H{idx_one_based}_Facility_Type",
        desc=f"Hospital is a general acute care hospital eligible for CMS star ratings (not VA, not Department of Defense, not a nursing home/other non-hospital facility type).",
        parent=hosp_node,
        critical=True,
    )
    type_claim = (
        "On the Medicare.gov Care Compare profile, the hospital type is 'Acute Care Hospital' (or 'Short Term Acute Care Hospital'), "
        "and it is not a Veterans Administration (VA) facility, not a Department of Defense/military hospital, and not a nursing home or other non-hospital facility."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_node,
        sources=url if _is_medicare_care_compare_url(url) else None,
        additional_instruction="Look for 'Hospital type' or similar label showing 'Acute Care Hospital' (or 'Short Term Acute Care'). Reject if it's VA, DoD/military, or a nursing home.",
    )

    # 6) Minimum reporting threshold implied by presence of a rating (critical)
    min_thresh_node = evaluator.add_leaf(
        id=f"H{idx_one_based}_Min_Reporting_Threshold",
        desc=f"Hospital meets CMS minimum reporting threshold for an Overall Star Rating (≥3 measures in ≥3 measure groups, with one being Safety of Care or Mortality).",
        parent=hosp_node,
        critical=True,
    )
    min_thresh_claim = (
        "The Medicare.gov Care Compare hospital profile displays an Overall Hospital Quality Star Rating for this hospital, "
        "which implies the hospital met CMS's minimum reporting threshold for an overall rating."
    )
    await evaluator.verify(
        claim=min_thresh_claim,
        node=min_thresh_node,
        sources=url if _is_medicare_care_compare_url(url) else None,
        additional_instruction="If the profile shows an Overall Hospital Quality Star Rating (any value), conclude the minimum reporting threshold was met.",
    )

    # 7) Star rating is 5 (critical)
    rating_node = evaluator.add_leaf(
        id=f"H{idx_one_based}_Star_Rating_5",
        desc=f"Confirms the hospital has a 5-star CMS Overall Hospital Quality Star Rating as of the most recent publicly available rating period.",
        parent=hosp_node,
        critical=True,
    )
    rating_claim = "The hospital’s Overall Hospital Quality Star Rating shown on the Medicare.gov Care Compare profile is 5 stars (i.e., 5 out of 5)."
    await evaluator.verify(
        claim=rating_claim,
        node=rating_node,
        sources=url if _is_medicare_care_compare_url(url) else None,
        additional_instruction="Accept '5', '5 stars', or '5 out of 5' as equivalent. Use both page text and screenshot if needed.",
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the CMS 5-star Midwest hospitals task.
    """
    # Initialize evaluator (sequential root so we can gate with global constraints first)
    evaluator = Evaluator()
    root = evaluator.initialize(
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
        default_model=model,
    )

    # Extract hospitals from the answer
    extraction: HospitalsExtraction = await evaluator.extract(
        prompt=prompt_extract_hospitals(),
        template_class=HospitalsExtraction,
        extraction_name="hospitals_extraction",
    )

    hospitals_all = extraction.hospitals or []
    selected: List[HospitalItem] = hospitals_all[:4] if len(hospitals_all) >= 4 else hospitals_all + [HospitalItem() for _ in range(4 - len(hospitals_all))]

    # Record some custom info for debugging
    evaluator.add_custom_info(
        info={
            "selected_first_four": [_hospital_brief(h) for h in selected],
            "total_extracted": len(hospitals_all),
        },
        info_type="selection_info",
        info_name="first_four_selected",
    )

    # -------------------- Global gating: Exactly four distinct hospitals -------------------- #
    global_constraints = evaluator.add_sequential(
        id="response_constraints",
        desc="Response lists four distinct hospitals (using the first four entries found).",
        parent=root,
        critical=True,  # Critical gating node
    )

    # Check that we have 4 named hospitals in the selected list
    four_named = all(bool((h.name or "").strip()) for h in selected)
    evaluator.add_custom_node(
        result=four_named,
        id="exactly_four_named",
        desc="Exactly four hospitals identified for evaluation (first four entries) and each has a name.",
        parent=global_constraints,
        critical=True,
    )

    # Check no duplicates among the four (by normalized name or Care Compare URL if provided)
    names = [_normalize_name(h.name) for h in selected]
    urls = [(h.care_compare_url or "").strip().lower() for h in selected]
    # Detect duplicates by name
    unique_names = len([n for n in set(names) if n]) == len([n for n in names if n])
    # Detect duplicates by URL (for non-empty URLs)
    non_empty_urls = [u for u in urls if u]
    unique_urls = len(set(non_empty_urls)) == len(non_empty_urls)
    no_duplicates = unique_names and unique_urls

    evaluator.add_custom_node(
        result=no_duplicates,
        id="no_duplicates",
        desc="No duplicates/aliases among the four selected hospitals (by normalized name and Care Compare URL).",
        parent=global_constraints,
        critical=True,
    )

    # -------------------- Per-hospital verification subtrees -------------------- #
    for i in range(4):
        await verify_hospital(
            evaluator=evaluator,
            parent_node=root,
            hospital=selected[i],
            idx_one_based=i + 1,
        )

    # Return evaluation summary
    return evaluator.get_summary()