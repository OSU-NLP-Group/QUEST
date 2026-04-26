import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ma_leed_2024_foxborough_facility"
TASK_DESCRIPTION = (
    "In 2024, Massachusetts ranked first among U.S. states for LEED-certified green building space per capita. "
    "One of the notable projects certified that year was a major sports and entertainment facility located in "
    "Foxborough, Massachusetts. This facility serves as the home venue for both a National Football League team "
    "and a Major League Soccer team. The facility achieved LEED Gold certification under the LEED v4.1 Operations "
    "and Maintenance: Existing Buildings rating system in the first quarter of 2024. What is the name of this facility? "
    "Provide the exact date it received LEED certification, the total square footage of the certified space, the number "
    "of LEED points it earned, and the specific LEED rating system version under which it was certified."
)

ROOT_DESC = (
    "Identify the Foxborough, MA sports/entertainment facility and report its LEED Gold (LEED v4.1 O+M: Existing Buildings) "
    "certification details from Q1 2024, including date, certified square footage, and LEED points, consistent with official "
    "USGBC records and MA 2024 LEED project listing."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FacilityCertificationExtraction(BaseModel):
    facility_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    nfl_team: Optional[str] = None
    mls_team: Optional[str] = None

    leed_level: Optional[str] = None
    rating_system: Optional[str] = None
    certification_date: Optional[str] = None
    total_square_footage: Optional[str] = None
    leed_points: Optional[str] = None

    # Source URLs
    usgbc_urls: List[str] = Field(default_factory=list)
    ma_listing_urls: List[str] = Field(default_factory=list)
    other_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facility_certification() -> str:
    return (
        "Extract from the answer the following fields exactly as presented:\n"
        "1. facility_name: The specific name of the sports/entertainment facility.\n"
        "2. city: The city where the facility is located (e.g., Foxborough).\n"
        "3. state: The state (e.g., Massachusetts).\n"
        "4. nfl_team: The NFL team for which this facility is the home venue (if mentioned).\n"
        "5. mls_team: The MLS team for which this facility is the home venue (if mentioned).\n"
        "6. leed_level: The LEED certification level (e.g., Gold).\n"
        "7. rating_system: The LEED rating system version and type (e.g., LEED v4.1 O+M: Existing Buildings).\n"
        "8. certification_date: The exact LEED certification date as stated (include month/day/year if provided).\n"
        "9. total_square_footage: The total certified square footage.\n"
        "10. leed_points: The LEED points earned.\n"
        "11. usgbc_urls: All URLs of USGBC Project Directory entries or official USGBC pages that directly present the facility's certification details.\n"
        "12. ma_listing_urls: URLs for Massachusetts's 2024 LEED-certified projects listing (e.g., USGBC Top States 2024 articles or MA-specific listings) that mention this facility.\n"
        "13. other_sources: Any other URLs cited that relate to the facility (e.g., official stadium pages, team sites, news articles).\n\n"
        "Return null for any missing fields. For URLs, return a list of valid, complete URLs extracted from the answer."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
MONTH_PATTERN = re.compile(
    r"\b(january|february|march)\b", re.IGNORECASE
)

MM_DD_YYYY_Q1_2024 = re.compile(
    r"\b(0?[1-3])/(0?[1-9]|[12][0-9]|3[01])/2024\b"
)

MDYYYY_WITH_MONTH_NAME = re.compile(
    r"\b(january|february|march)\s+(0?[1-9]|[12][0-9]|3[01]),\s*2024\b",
    re.IGNORECASE
)


def has_day_month_year_2024(date_str: Optional[str]) -> bool:
    if not date_str:
        return False
    s = date_str.strip()
    if MM_DD_YYYY_Q1_2024.search(s):
        return True
    if MDYYYY_WITH_MONTH_NAME.search(s):
        return True
    # Also accept ISO-style 2024-01-.., 2024-02-.., 2024-03-..
    if re.search(r"\b2024-(0[1-3])-(0[1-9]|[12][0-9]|3[01])\b", s):
        return True
    return False


def is_q1_2024(date_str: Optional[str]) -> bool:
    if not date_str:
        return False
    s = date_str.strip()

    # mm/dd/2024 format
    m = MM_DD_YYYY_Q1_2024.search(s)
    if m:
        month_num = int(m.group(1))
        return 1 <= month_num <= 3

    # Month name day, 2024
    m2 = MDYYYY_WITH_MONTH_NAME.search(s)
    if m2:
        month_name = m2.group(1).lower()
        return month_name in ("january", "february", "march")

    # ISO 2024-01.. etc.
    m3 = re.search(r"\b2024-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])\b", s)
    if m3:
        month_num = int(m3.group(1))
        return 1 <= month_num <= 3

    # If only month name present without explicit day, don't count for exactness/Q1 check
    if MONTH_PATTERN.search(s) and "2024" in s:
        # Might be month/year only—insufficient for exact-date check; Q1 check could be true but we require day too
        return False
    return False


def non_empty_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if isinstance(u, str) and u.strip()]


def combine_sources(primary: List[str], secondary: List[str]) -> List[str]:
    seen = set()
    combined = []
    for u in primary + secondary:
        if not u:
            continue
        u2 = u.strip()
        if u2 and u2 not in seen:
            seen.add(u2)
            combined.append(u2)
    return combined


def safe_text(val: Optional[str]) -> str:
    return val.strip() if isinstance(val, str) else ""


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_facility_identity_constraints(
    evaluator: Evaluator,
    parent_node,
    ext: FacilityCertificationExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="facility_identity_constraints",
        desc="Facility identification satisfies all facility-related constraints.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: facility_name_present (custom existence check)
    has_name = bool(ext.facility_name and ext.facility_name.strip())
    evaluator.add_custom_node(
        result=has_name,
        id="facility_name_present",
        desc="Answer provides a specific facility name.",
        parent=node,
        critical=True,
    )

    # Leaf: located_in_foxborough_ma
    loc_leaf = evaluator.add_leaf(
        id="located_in_foxborough_ma",
        desc="The named facility is located in Foxborough, Massachusetts.",
        parent=node,
        critical=True,
    )
    facility_name = safe_text(ext.facility_name)
    claim_loc = (
        f"The facility '{facility_name}' is located in Foxborough, Massachusetts (also acceptable as 'Foxboro, MA')."
        if facility_name
        else "This facility is located in Foxborough, Massachusetts."
    )
    loc_sources = combine_sources(non_empty_urls(ext.usgbc_urls), non_empty_urls(ext.other_sources))
    await evaluator.verify(
        claim=claim_loc,
        node=loc_leaf,
        sources=loc_sources if loc_sources else None,
        additional_instruction=(
            "Verify with the provided webpage(s) that the facility's location is Foxborough, MA. "
            "Allow minor spelling variations such as 'Foxboro'."
        ),
    )

    # Leaf: home_to_nfl_and_mls
    nfl = safe_text(ext.nfl_team)
    mls = safe_text(ext.mls_team)
    home_leaf = evaluator.add_leaf(
        id="home_to_nfl_and_mls",
        desc="The facility serves as a home venue for both an NFL team and an MLS team.",
        parent=node,
        critical=True,
    )
    if nfl and mls:
        claim_home = (
            f"The facility '{facility_name}' is the home venue for both the NFL team '{nfl}' and the MLS team '{mls}'."
        )
    else:
        claim_home = (
            f"The facility '{facility_name}' serves as the home venue for both an NFL team and an MLS team."
            if facility_name
            else "This facility serves as the home venue for both an NFL team and an MLS team."
        )
    home_sources = combine_sources(non_empty_urls(ext.other_sources), non_empty_urls(ext.usgbc_urls))
    await evaluator.verify(
        claim=claim_home,
        node=home_leaf,
        sources=home_sources if home_sources else None,
        additional_instruction=(
            "Verify with the referenced webpage(s) that the facility is the home venue for one NFL team and one MLS team. "
            "Names may vary slightly or include prefixes/suffixes; allow reasonable variations."
        ),
    )


async def build_leed_certification_constraints(
    evaluator: Evaluator,
    parent_node,
    ext: FacilityCertificationExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="leed_certification_constraints",
        desc="LEED certification details satisfy all certification-related constraints.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: leed_gold_level
    gold_leaf = evaluator.add_leaf(
        id="leed_gold_level",
        desc="Facility achieved LEED Gold certification level.",
        parent=node,
        critical=True,
    )
    claim_gold = "The facility achieved LEED Gold certification."
    await evaluator.verify(
        claim=claim_gold,
        node=gold_leaf,
        sources=non_empty_urls(ext.usgbc_urls) or None,
        additional_instruction="Verify the certification level shown on the official USGBC directory page.",
    )

    # Leaf: rating_system_v41_om_eb
    rs_leaf = evaluator.add_leaf(
        id="rating_system_v41_om_eb",
        desc="Certification is under LEED v4.1 O+M: Existing Buildings (or an unambiguous equivalent phrasing).",
        parent=node,
        critical=True,
    )
    claim_rs = (
        "This facility was certified under LEED v4.1 Operations and Maintenance: Existing Buildings "
        "(also referred to as LEED v4.1 O+M: Existing Buildings)."
    )
    await evaluator.verify(
        claim=claim_rs,
        node=rs_leaf,
        sources=non_empty_urls(ext.usgbc_urls) or None,
        additional_instruction=(
            "Match the rating system wording precisely or with clear equivalent phrasing (e.g., "
            "'LEED v4.1 O+M: EB', 'LEED v4.1 Operations + Maintenance – Existing Buildings')."
        ),
    )

    # Leaf: exact_certification_date_provided (custom existence/format check)
    date_str = safe_text(ext.certification_date)
    evaluator.add_custom_node(
        result=has_day_month_year_2024(date_str),
        id="exact_certification_date_provided",
        desc="Answer provides the exact LEED certification date (month/day/year).",
        parent=node,
        critical=True,
    )

    # Leaf: certification_date_in_q1_2024 (custom quarter check)
    evaluator.add_custom_node(
        result=is_q1_2024(date_str),
        id="certification_date_in_q1_2024",
        desc="The certification date is in Q1 2024 (January–March 2024).",
        parent=node,
        critical=True,
    )


async def build_official_record_verifiability(
    evaluator: Evaluator,
    parent_node,
    ext: FacilityCertificationExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="official_record_verifiability",
        desc="Reported numerical details are consistent with official USGBC certification records/directory as required by constraints.",
        parent=parent_node,
        critical=True,
    )

    # Optional gating leaf: ensure USGBC source is provided
    usgbc_urls = non_empty_urls(ext.usgbc_urls)
    evaluator.add_custom_node(
        result=bool(usgbc_urls),
        id="usgbc_source_provided",
        desc="USGBC Project Directory/official USGBC source URL is provided in the answer.",
        parent=node,
        critical=True,
    )

    # Leaf: square_footage_matches_usgbc_directory
    sqft_leaf = evaluator.add_leaf(
        id="square_footage_matches_usgbc_directory",
        desc="Provides total certified square footage, and the value matches/verifies against the USGBC project directory record for the facility.",
        parent=node,
        critical=True,
    )
    sqft = safe_text(ext.total_square_footage)
    facility_name = safe_text(ext.facility_name)
    claim_sqft = (
        f"The total certified square footage for '{facility_name}' is {sqft}."
        if facility_name and sqft
        else f"The total certified square footage is {sqft}."
    )
    await evaluator.verify(
        claim=claim_sqft,
        node=sqft_leaf,
        sources=usgbc_urls or None,
        additional_instruction=(
            "Verify the certified area from the USGBC Project Directory page. Allow minor formatting differences "
            "in numbers (commas, spaces) and minor rounding variations."
        ),
    )

    # Leaf: leed_points_match_usgbc_record
    points_leaf = evaluator.add_leaf(
        id="leed_points_match_usgbc_record",
        desc="Provides LEED points earned, and the value matches/verifies against the USGBC certification record for the facility.",
        parent=node,
        critical=True,
    )
    points = safe_text(ext.leed_points)
    claim_points = (
        f"The number of LEED points earned by '{facility_name}' is {points}."
        if facility_name and points
        else f"The number of LEED points earned is {points}."
    )
    await evaluator.verify(
        claim=claim_points,
        node=points_leaf,
        sources=usgbc_urls or None,
        additional_instruction=(
            "Verify the LEED points from the USGBC Project Directory page. Allow reasonable rounding if needed."
        ),
    )

    # Leaf: certification_date_available_in_usgbc_records
    date_leaf = evaluator.add_leaf(
        id="certification_date_available_in_usgbc_records",
        desc="The stated exact certification date is available/confirmed in official USGBC records for the facility.",
        parent=node,
        critical=True,
    )
    date_str = safe_text(ext.certification_date)
    claim_date = (
        f"The LEED certification date for '{facility_name}' is {date_str}."
        if facility_name and date_str
        else f"The LEED certification date is {date_str}."
    )
    await evaluator.verify(
        claim=claim_date,
        node=date_leaf,
        sources=usgbc_urls or None,
        additional_instruction=(
            "Confirm that the exact certification date (including day) is shown on the official USGBC record."
        ),
    )


async def build_ma_listing_verification(
    evaluator: Evaluator,
    parent_node,
    ext: FacilityCertificationExtraction,
) -> None:
    # Optional gating leaf to ensure listing source is provided
    listing_urls = non_empty_urls(ext.ma_listing_urls)
    gating_leaf = evaluator.add_custom_node(
        result=bool(listing_urls),
        id="ma_listing_source_provided",
        desc="A Massachusetts 2024 LEED project listing URL (or USGBC 'Top States 2024' page mentioning the facility) is provided.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: ma_2024_project_listing
    leaf = evaluator.add_leaf(
        id="ma_2024_project_listing",
        desc="Facility is listed among Massachusetts's 2024 LEED-certified projects (per the relevant official listing referenced by the task).",
        parent=parent_node,
        critical=True,
    )
    facility_name = safe_text(ext.facility_name)
    claim_listing = (
        f"The facility '{facility_name}' appears on a Massachusetts 2024 LEED-certified projects listing or USGBC 'Top States for LEED in 2024' page as a notable MA project."
        if facility_name
        else "This facility appears on a Massachusetts 2024 LEED-certified projects listing or USGBC 'Top States for LEED in 2024' page as a notable MA project."
    )
    await evaluator.verify(
        claim=claim_listing,
        node=leaf,
        sources=listing_urls or None,
        additional_instruction=(
            "Check the provided listing webpage to confirm the facility is included as a Massachusetts 2024 LEED-certified project."
        ),
        extra_prerequisites=[gating_leaf],  # Ensure gating applies
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

    # Extract structured information
    extraction: FacilityCertificationExtraction = await evaluator.extract(
        prompt=prompt_extract_facility_certification(),
        template_class=FacilityCertificationExtraction,
        extraction_name="facility_certification_extraction",
    )

    # Build a critical task-level node to mirror rubric root (since initialize sets a non-critical root)
    task_root = evaluator.add_parallel(
        id="task_root",
        desc=ROOT_DESC,
        parent=root,
        critical=True,
    )

    # Subtrees
    await build_facility_identity_constraints(evaluator, task_root, extraction)
    await build_leed_certification_constraints(evaluator, task_root, extraction)
    await build_official_record_verifiability(evaluator, task_root, extraction)
    await build_ma_listing_verification(evaluator, task_root, extraction)

    # Add minimal custom info about sources used
    evaluator.add_custom_info(
        info={
            "usgbc_urls_count": len(non_empty_urls(extraction.usgbc_urls)),
            "ma_listing_urls_count": len(non_empty_urls(extraction.ma_listing_urls)),
            "other_sources_count": len(non_empty_urls(extraction.other_sources)),
        },
        info_type="source_stats",
    )

    return evaluator.get_summary()