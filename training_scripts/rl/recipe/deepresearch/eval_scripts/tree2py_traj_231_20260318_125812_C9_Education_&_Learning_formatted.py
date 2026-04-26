import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "ca_superintendent_departures_2025_2026"
TASK_DESCRIPTION = """
Identify three California school district superintendents or county superintendents of schools who announced their retirement or resignation between October 2025 and February 2026 (inclusive), have served at least 10 years in their current superintendent position, and whose effective departure date falls before November 1, 2026. For each of the three superintendents, provide the following information: (1) their full name, (2) the specific school district or county office of education they lead, (3) the date on which they announced their retirement or resignation, (4) their effective departure date (last day in the position), (5) the length of their tenure as superintendent in their current position, and (6) at least one verifiable characteristic about their district or county office (such as student enrollment size, district formation history, California state ranking by size, or geographic coverage). Ensure all information is accurate and can be verified through reliable sources.
"""

# Logical windows for this evaluation
ANNOUNCEMENT_START = "2025-10-01"
ANNOUNCEMENT_END = "2026-02-28"
DEPARTURE_DEADLINE = "2026-11-01"


# -----------------------------------------------------------------------------
# Data models for extraction
# -----------------------------------------------------------------------------
class SuperintendentItem(BaseModel):
    # Identity
    full_name: Optional[str] = None
    organization: Optional[str] = None  # School district or county office of education name
    organization_type: Optional[str] = None  # e.g., "school district" or "county office of education"
    announcement_type: Optional[str] = None  # "retirement", "resignation", or similar paraphrase

    # Dates
    announcement_date: Optional[str] = None  # Prefer ISO like 2025-10-15, but allow any textual date
    departure_date: Optional[str] = None     # Prefer ISO like 2026-06-30, but allow any textual date

    # Tenure
    tenure_years: Optional[str] = None       # e.g., "12", "10+", "more than 15", "over a decade"
    tenure_text: Optional[str] = None        # free text like "since 2014", "served 12 years"

    # District/County characteristic
    characteristic_text: Optional[str] = None  # e.g., "enrollment ~25,000", "formed in 1879", etc.

    # URLs (by category)
    identity_urls: List[str] = Field(default_factory=list)        # proves identity and role
    announcement_urls: List[str] = Field(default_factory=list)    # proves announcement and its date
    departure_urls: List[str] = Field(default_factory=list)       # proves effective departure date
    tenure_urls: List[str] = Field(default_factory=list)          # proves tenure length or start date
    characteristic_urls: List[str] = Field(default_factory=list)  # proves the district characteristic


class SuperintendentsExtraction(BaseModel):
    items: List[SuperintendentItem] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_superintendents() -> str:
    return """
    Extract from the answer a list of superintendents or county superintendents of schools in California who meet the task's constraints. Return JSON under key "items" with up to all entries mentioned in the answer (we will later take the first three). For each item, extract the following fields exactly as stated in the answer:

    REQUIRED IDENTITY:
    - full_name: The person's full name.
    - organization: The exact name of the California school district or county office of education.
    - organization_type: Either "school district" or "county office of education" (or a close variant) if stated.
    - announcement_type: "retirement", "resignation", or a close paraphrase (if clearly stated), otherwise null.

    DATES:
    - announcement_date: The announcement date text (prefer ISO like "2025-10-15" if provided; else use the exact wording from the answer).
    - departure_date: The effective last day / departure date text (prefer ISO like "2026-06-30" if provided; else use the exact wording from the answer).

    TENURE:
    - tenure_years: A short text representing "years in current superintendent role" (e.g., "12", "10+", "more than 15", "over a decade", etc.) if provided.
    - tenure_text: Any longer text about tenure (e.g., "served since 2014", "12 years as superintendent", etc.) if present.

    DISTRICT/COUNTY CHARACTERISTIC:
    - characteristic_text: At least one verifiable characteristic (e.g., approximate enrollment, formation history, state size rank, geographic coverage). If multiple are mentioned, pick one succinct statement.

    URL SOURCES (strictly extract actual URLs present in the answer text):
    - identity_urls: URL(s) that directly support identity/role (official bio, district page, credible news).
    - announcement_urls: URL(s) that directly support the announcement and its date.
    - departure_urls: URL(s) that directly support the effective departure / last day.
    - tenure_urls: URL(s) that support tenure length (or start year that implies the tenure).
    - characteristic_urls: URL(s) that support the chosen characteristic.

    RULES FOR URL FIELDS:
    - Extract only URLs explicitly present in the answer (plain or markdown links). Do not invent any URL.
    - Include full URLs with protocol; if missing, prepend http://
    - If a given category has no URLs, return an empty list for that category.

    If any field above is missing in the answer, set it to null (or empty list for URL fields).
    """


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _unique_urls(url_lists: List[List[str]]) -> List[str]:
    seen = set()
    result: List[str] = []
    for urls in url_lists:
        for u in urls or []:
            u2 = (u or "").strip()
            if not u2:
                continue
            if u2 not in seen:
                seen.add(u2)
                result.append(u2)
    return result


def _gather_identity_support_urls(item: SuperintendentItem) -> List[str]:
    # Identity can often also be supported by announcement/departure/tenure articles.
    return _unique_urls([item.identity_urls, item.announcement_urls, item.departure_urls, item.tenure_urls])


def _gather_all_urls(item: SuperintendentItem) -> List[str]:
    return _unique_urls([
        item.identity_urls,
        item.announcement_urls,
        item.departure_urls,
        item.tenure_urls,
        item.characteristic_urls
    ])


def _build_announcement_claim(item: SuperintendentItem) -> str:
    # Flexible wording to cover retirement or resignation mention
    typ = item.announcement_type or "retirement or resignation"
    name = item.full_name or "[NAME MISSING]"
    org = item.organization or "[ORG MISSING]"
    date = item.announcement_date or "[DATE MISSING]"
    return f"On {date}, {name} announced their {typ} from {org}."


def _build_departure_claim(item: SuperintendentItem) -> str:
    name = item.full_name or "[NAME MISSING]"
    org = item.organization or "[ORG MISSING]"
    date = item.departure_date or "[DATE MISSING]"
    return f"{name}'s effective last day as superintendent of {org} is {date}."


def _build_identity_claim(item: SuperintendentItem) -> str:
    name = item.full_name or "[NAME MISSING]"
    org = item.organization or "[ORG MISSING]"
    # Phrase to allow either current or outgoing superintendent during transition
    return f"{name} is or was the superintendent (or county superintendent of schools) of {org} in California."


def _build_tenure_reference_claim(item: SuperintendentItem) -> Optional[str]:
    name = item.full_name or "[NAME MISSING]"
    org = item.organization or "[ORG MISSING]"
    if _nonempty(item.tenure_years):
        # Normalize readable phrasing, e.g., "12" -> "12 years"
        yrs = item.tenure_years.strip()
        # If it's purely digits, append " years"
        if yrs.isdigit():
            yrs_phrase = f"{yrs} years"
        else:
            yrs_phrase = yrs
        return f"{name} has served as superintendent of {org} for {yrs_phrase}."
    if _nonempty(item.tenure_text):
        return f"{name} tenure statement: {item.tenure_text}"
    return None


def _build_characteristic_claim(item: SuperintendentItem) -> Optional[str]:
    if _nonempty(item.characteristic_text):
        return item.characteristic_text.strip()
    return None


def _build_range_claim(date_str: str, start: str, end: str) -> str:
    return (
        f"The date '{date_str}' falls between {start} and {end}, inclusive."
    )


def _build_before_claim(date_str: str, deadline: str) -> str:
    return f"The date '{date_str}' is before {deadline}."


def _build_tenure_requirement_claim(item: SuperintendentItem) -> Optional[str]:
    # Let LLM judge whether the stated phrase indicates 10+ years.
    if _nonempty(item.tenure_years):
        return f"The tenure stated as '{item.tenure_years}' indicates at least 10 years in the role."
    if _nonempty(item.tenure_text):
        return f"The tenure description '{item.tenure_text}' indicates at least 10 years in the role."
    return None


# -----------------------------------------------------------------------------
# Verification sub-tree for one superintendent
# -----------------------------------------------------------------------------
async def verify_superintendent(
    evaluator: Evaluator,
    parent_node,
    item: SuperintendentItem,
    index: int,
) -> None:
    """
    Build the verification sub-tree for a single superintendent (index 0..2).
    """
    idx = index + 1
    sup_node = evaluator.add_parallel(
        id=f"superintendent_{idx}",
        desc=f"{['First','Second','Third'][index]} qualifying superintendent identified with complete information",
        parent=parent_node,
        critical=False  # allow partial credit per superintendent
    )

    # 1) Identification (Critical, Sequential)
    ident_node = evaluator.add_sequential(
        id=f"sup_{idx}_identification",
        desc="Superintendent's full name and district/county office correctly provided",
        parent=sup_node,
        critical=True
    )

    # 1.1 Provided (custom existence)
    provided_ident = evaluator.add_custom_node(
        result=_nonempty(item.full_name) and _nonempty(item.organization),
        id=f"sup_{idx}_name_and_district_provided",
        desc="Both full name and district/county office name are provided",
        parent=ident_node,
        critical=True
    )

    # 1.3 Identification reference (URL presence as gating)
    ident_support_urls = _gather_identity_support_urls(item)
    ident_ref = evaluator.add_custom_node(
        result=len(ident_support_urls) > 0,
        id=f"sup_{idx}_identification_reference",
        desc="URL reference supporting the superintendent's identity and position",
        parent=ident_node,
        critical=True
    )

    # 1.2 Accuracy (verify with URLs)
    name_district_accuracy = evaluator.add_leaf(
        id=f"sup_{idx}_name_and_district_accuracy",
        desc="Provided name and district/office match the actual superintendent and their organization",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim=_build_identity_claim(item),
        node=name_district_accuracy,
        sources=ident_support_urls,
        additional_instruction=(
            "Confirm using the webpage(s) that this person is or was the superintendent (or county superintendent of schools) "
            "of the specified California district/county office. Allow for present/past tense due to announced departures."
        )
    )

    # 2) Announcement timing (Critical, Sequential)
    ann_node = evaluator.add_sequential(
        id=f"sup_{idx}_announcement_timing",
        desc="Announcement date falls within October 2025 to February 2026 (inclusive)",
        parent=sup_node,
        critical=True
    )

    # 2.1 Announcement date provided
    ann_provided = evaluator.add_custom_node(
        result=_nonempty(item.announcement_date),
        id=f"sup_{idx}_announcement_date_provided",
        desc="Announcement date is explicitly stated",
        parent=ann_node,
        critical=True
    )

    # 2.2 Announcement date verification (window logic via LLM simple check)
    ann_in_window = evaluator.add_leaf(
        id=f"sup_{idx}_announcement_date_verification",
        desc="Announcement date is verified to be between October 1, 2025 and February 28, 2026 (inclusive)",
        parent=ann_node,
        critical=True
    )
    await evaluator.verify(
        claim=_build_range_claim(item.announcement_date or "[DATE MISSING]", ANNOUNCEMENT_START, ANNOUNCEMENT_END),
        node=ann_in_window,
        additional_instruction=(
            "Judge only the logical comparison of the given date text with the stated date window. "
            "If the date text is missing or ambiguous, mark incorrect."
        )
    )

    # 2.3 Announcement reference (verify by URLs)
    ann_ref = evaluator.add_leaf(
        id=f"sup_{idx}_announcement_reference",
        desc="URL reference supporting the announcement date",
        parent=ann_node,
        critical=True
    )
    await evaluator.verify(
        claim=_build_announcement_claim(item),
        node=ann_ref,
        sources=item.announcement_urls,
        additional_instruction=(
            "Verify the page states that the person announced a retirement or resignation on that exact date (or a clearly "
            "equivalent phrasing that unambiguously indicates the date of announcement)."
        )
    )

    # 3) Departure timing (Critical, Sequential)
    dep_node = evaluator.add_sequential(
        id=f"sup_{idx}_departure_timing",
        desc="Effective departure date is before November 1, 2026",
        parent=sup_node,
        critical=True
    )

    dep_provided = evaluator.add_custom_node(
        result=_nonempty(item.departure_date),
        id=f"sup_{idx}_departure_date_provided",
        desc="Effective departure date is explicitly stated",
        parent=dep_node,
        critical=True
    )

    dep_before_deadline = evaluator.add_leaf(
        id=f"sup_{idx}_departure_date_verification",
        desc="Departure date is verified to be before November 1, 2026",
        parent=dep_node,
        critical=True
    )
    await evaluator.verify(
        claim=_build_before_claim(item.departure_date or "[DATE MISSING]", DEPARTURE_DEADLINE),
        node=dep_before_deadline,
        additional_instruction=(
            "Judge only the logical comparison of the date with the deadline. "
            "If the date text is missing or ambiguous, mark incorrect."
        )
    )

    dep_ref = evaluator.add_leaf(
        id=f"sup_{idx}_departure_reference",
        desc="URL reference supporting the effective departure date",
        parent=dep_node,
        critical=True
    )
    await evaluator.verify(
        claim=_build_departure_claim(item),
        node=dep_ref,
        sources=item.departure_urls,
        additional_instruction=(
            "Verify that the page states or clearly implies the effective last day (or effective date) of the superintendent's service."
        )
    )

    # 4) Tenure information (Critical, Sequential)
    ten_node = evaluator.add_sequential(
        id=f"sup_{idx}_tenure_information",
        desc="Superintendent has served at least 10 years in current position",
        parent=sup_node,
        critical=True
    )

    ten_provided = evaluator.add_custom_node(
        result=_nonempty(item.tenure_years) or _nonempty(item.tenure_text),
        id=f"sup_{idx}_tenure_length_provided",
        desc="Tenure length in superintendent position is stated",
        parent=ten_node,
        critical=True
    )

    ten_meets_req = evaluator.add_leaf(
        id=f"sup_{idx}_tenure_requirement_met",
        desc="Stated tenure is verified to be 10 years or more",
        parent=ten_node,
        critical=True
    )
    await evaluator.verify(
        claim=_build_tenure_requirement_claim(item) or "The tenure information indicates at least 10 years in the role.",
        node=ten_meets_req,
        additional_instruction=(
            "Judge purely based on the phrasing: if it clearly indicates 10 or more years (e.g., '12 years', 'more than a decade', "
            "'since 2014' with implicit 10+ years by 2026), mark correct; otherwise, incorrect."
        )
    )

    ten_ref_claim = _build_tenure_reference_claim(item) or _build_identity_claim(item)
    ten_ref_sources = item.tenure_urls if item.tenure_urls else _gather_identity_support_urls(item)
    ten_ref = evaluator.add_leaf(
        id=f"sup_{idx}_tenure_reference",
        desc="URL reference supporting the tenure information",
        parent=ten_node,
        critical=True
    )
    await evaluator.verify(
        claim=ten_ref_claim,
        node=ten_ref,
        sources=ten_ref_sources,
        additional_instruction=(
            "Verify that the page supports the tenure length (either explicitly states number of years or provides a start year/date "
            "from which a 10+ year tenure is clear by 2026). Minor rounding or phrasing differences are acceptable."
        )
    )

    # 5) District/County characteristic (Critical, Sequential)
    char_node = evaluator.add_sequential(
        id=f"sup_{idx}_district_characteristic",
        desc="At least one verifiable district characteristic is provided",
        parent=sup_node,
        critical=True
    )

    char_provided = evaluator.add_custom_node(
        result=_nonempty(item.characteristic_text),
        id=f"sup_{idx}_characteristic_provided",
        desc="A district characteristic is explicitly stated (enrollment, formation, ranking, etc.)",
        parent=char_node,
        critical=True
    )

    # Verify characteristic with URLs
    char_verify = evaluator.add_leaf(
        id=f"sup_{idx}_characteristic_verification",
        desc="The stated characteristic is factually accurate",
        parent=char_node,
        critical=True
    )
    await evaluator.verify(
        claim=_build_characteristic_claim(item) or "The district characteristic as stated is correct.",
        node=char_verify,
        sources=item.characteristic_urls,
        additional_instruction=(
            "Verify the specific characteristic (e.g., enrollment magnitude, formation year/history, ranking by size in CA, "
            "geographic coverage) is supported by the page. Allow reasonable rounding or recent slight enrollment fluctuation."
        )
    )

    # Reference presence (URL existence for characteristic)
    char_ref = evaluator.add_custom_node(
        result=len(item.characteristic_urls) > 0,
        id=f"sup_{idx}_characteristic_reference",
        desc="URL reference supporting the district characteristic",
        parent=char_node,
        critical=True
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the California superintendent departures task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # as rubric root: parallel aggregation across the three superintendents
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

    # Record evaluation constraints for transparency
    evaluator.add_custom_info(
        info={
            "announcement_window_inclusive": [ANNOUNCEMENT_START, ANNOUNCEMENT_END],
            "departure_deadline_before": DEPARTURE_DEADLINE,
            "required_count": 3,
            "jurisdiction": "California",
        },
        info_type="constraints",
        info_name="task_constraints",
    )

    # Extract structured items from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_superintendents(),
        template_class=SuperintendentsExtraction,
        extraction_name="superintendents_extraction",
    )

    # Take first 3 items; if fewer, pad with empty items
    items: List[SuperintendentItem] = list(extraction.items[:3])
    while len(items) < 3:
        items.append(SuperintendentItem())

    # Build verification for each required superintendent
    # Children (superintendent_1/2/3) are parallel under root per rubric
    for i in range(3):
        await verify_superintendent(evaluator, root, items[i], i)

    # Return summary
    return evaluator.get_summary()