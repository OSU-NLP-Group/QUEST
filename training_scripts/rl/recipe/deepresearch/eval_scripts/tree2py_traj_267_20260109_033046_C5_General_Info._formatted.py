import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "free_cultural_institutions"
TASK_DESCRIPTION = (
    "I am planning a multi-state educational tour and need to visit public cultural institutions "
    "(such as state capitol buildings, public libraries, museums, or botanical gardens) that offer free admission. "
    "I have early morning schedules and need to arrive before 10:00 AM on weekdays. Please identify four such institutions, "
    "each located in a different U.S. state. For each institution, provide: (1) the complete street address including city, state, and ZIP code, "
    "(2) a phone contact number, (3) confirmation that the institution opens before 10:00 AM on at least one weekday, and (4) a direct link to "
    "the institution's official webpage where this information can be verified."
)


# -----------------------------------------------------------------------------
# Data models for extraction
# -----------------------------------------------------------------------------
class InstitutionItem(BaseModel):
    """Single institution information extracted from the answer."""
    name: Optional[str] = None
    category: Optional[str] = None  # e.g., "state capitol building", "public library", "museum", "botanical garden"
    street_number: Optional[str] = None
    street_name: Optional[str] = None  # Include street name/type (e.g., "Main St", "Capitol Ave")
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    phone_number: Optional[str] = None
    weekday_open_day: Optional[str] = None  # e.g., "Monday"
    weekday_open_time: Optional[str] = None  # e.g., "9:00 AM"
    official_url: Optional[str] = None


class InstitutionsExtraction(BaseModel):
    """Container for up to four institutions."""
    institutions: List[InstitutionItem] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_institutions() -> str:
    return (
        "Extract up to four institutions from the answer that are public cultural institutions and meet the tour requirements. "
        "For each institution, extract exactly these fields:\n"
        "1) name: the institution's official name.\n"
        "2) category: one of these categories if applicable — 'state capitol building', 'public library', 'museum', or 'botanical garden'. "
        "   If a synonym is used in the answer (e.g., 'capitol', 'botanic garden'), normalize to one of the four categories.\n"
        "3) street_number: the street number (e.g., '120').\n"
        "4) street_name: the street name and type (e.g., 'Main St', 'Capitol Ave').\n"
        "5) city: the city name.\n"
        "6) state: the U.S. state name or two-letter abbreviation (e.g., 'CA' or 'California').\n"
        "7) zip_code: the 5-digit ZIP code (or ZIP+4 if provided).\n"
        "8) phone_number: a phone contact number for the institution as written in the answer (e.g., '(555) 123-4567').\n"
        "9) weekday_open_day: a weekday (Monday–Friday) on which the institution opens before 10:00 AM, as specified in the answer.\n"
        "10) weekday_open_time: the opening time for that weekday (e.g., '9:00 AM').\n"
        "11) official_url: a direct URL to the institution's official webpage where the above information can be verified. "
        "    Only extract a URL if the answer provides one explicitly. Prefer official domains (e.g., .gov for state capitols, "
        "    or the organization's own site), and avoid aggregator/review sites (Yelp, TripAdvisor, etc.). If no URL is present, return null.\n\n"
        "Rules:\n"
        "- Extract exactly what the answer provides; do not invent missing parts.\n"
        "- If the answer lists more than four institutions, extract the first four only.\n"
        "- If some fields are missing for an institution, set those fields to null.\n"
        "- If an address is provided as a single line, split it into street_number, street_name, city, state, and zip_code as best as possible. "
        "  If splitting is not possible, set unknown components to null.\n"
        "- For phone_number, extract the number exactly as written in the answer.\n"
    )


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def build_full_address(inst: InstitutionItem) -> str:
    parts = []
    line1 = " ".join([p for p in [inst.street_number or "", inst.street_name or ""] if p.strip()])
    if line1.strip():
        parts.append(line1.strip())
    city_state_zip = ", ".join([p for p in [inst.city or "", inst.state or ""] if p.strip()])
    if city_state_zip.strip():
        parts.append(city_state_zip.strip())
    if inst.zip_code and inst.zip_code.strip():
        # Append ZIP after the city/state with space
        if parts:
            parts[-1] = parts[-1] + f" {inst.zip_code.strip()}"
        else:
            parts.append(inst.zip_code.strip())
    return ", ".join(parts) if parts else ""


def normalize_institutions(extracted: InstitutionsExtraction) -> List[InstitutionItem]:
    """Return exactly four InstitutionItem entries (truncate or pad with empty)."""
    items = extracted.institutions[:4]
    while len(items) < 4:
        items.append(InstitutionItem())
    return items


# -----------------------------------------------------------------------------
# Verification logic per institution
# -----------------------------------------------------------------------------
async def verify_institution(
    evaluator: Evaluator,
    parent_node,
    inst: InstitutionItem,
    index: int,
) -> None:
    """
    Build verification nodes and run checks for a single institution.
    """
    inst_num = index + 1
    inst_parent = evaluator.add_parallel(
        id=f"Institution_{inst_num}",
        desc=f"Institution #{inst_num} meeting all per-institution criteria",
        parent=parent_node,
        critical=False
    )

    # Critical: Reference URL must be provided and be an official page
    url_leaf = evaluator.add_leaf(
        id=f"Reference_URL_{inst_num}",
        desc="Provides a direct URL to the official institution webpage where the required information can be verified",
        parent=inst_parent,
        critical=True,
    )
    institution_name = inst.name or "the institution"
    await evaluator.verify(
        claim=f"This webpage is the official webpage of {institution_name}. Aggregator/review pages are not official.",
        node=url_leaf,
        sources=inst.official_url,
        additional_instruction=(
            "Judge 'Correct' only if the provided URL is an official webpage belonging to the institution or its governing body "
            "(e.g., .gov for state capitols, the library/museum/botanical garden's own site). "
            "If the URL is missing or appears to be an aggregator/review site (Yelp, TripAdvisor, etc.), judge 'Incorrect'."
        ),
    )

    # Critical: Institution type must be one of the allowed categories and supported by the official page
    type_leaf = evaluator.add_leaf(
        id=f"Institution_Type_{inst_num}",
        desc="Institution is a public cultural institution: state capitol building, public library, museum, or botanical garden",
        parent=inst_parent,
        critical=True,
    )
    category_text = inst.category or "public cultural institution"
    await evaluator.verify(
        claim=(
            f"The institution '{institution_name}' is a {category_text} and qualifies as a public cultural institution "
            "(state capitol building, public library, museum, or botanical garden)."
        ),
        node=type_leaf,
        sources=inst.official_url,
        additional_instruction=(
            "Use the official page to confirm the institution's nature. Accept synonyms (e.g., 'state capitol' for 'state capitol building', "
            "'botanic garden' for 'botanical garden'). Do not accept venues that are not clearly one of the four categories."
        ),
    )

    # Critical: Free admission must be supported by the official page
    free_leaf = evaluator.add_leaf(
        id=f"Free_Admission_{inst_num}",
        desc="Institution offers free admission to the general public",
        parent=inst_parent,
        critical=True,
    )
    await evaluator.verify(
        claim="The institution offers free admission to the general public (no general admission fee).",
        node=free_leaf,
        sources=inst.official_url,
        additional_instruction=(
            "Judge 'Correct' only if the official page clearly indicates free general admission (e.g., 'free admission', 'no admission fee', "
            "'free to all'). If free admission is limited only to specific days/times or specific groups (e.g., residents, members), "
            "and general admission is otherwise paid, judge 'Incorrect'."
        ),
    )

    # Critical: Complete address must be present and match the official page
    address_leaf = evaluator.add_leaf(
        id=f"Complete_Address_{inst_num}",
        desc="Provides complete street address including street number, street name, city, state, and ZIP code",
        parent=inst_parent,
        critical=True,
    )
    full_address = build_full_address(inst)
    await evaluator.verify(
        claim=(
            f"The official webpage lists the full address for {institution_name} as '{full_address}', including street number, street name, "
            "city, state, and ZIP code."
        ),
        node=address_leaf,
        sources=inst.official_url,
        additional_instruction=(
            "Judge 'Correct' only if all components are present (street number, street name, city, state, ZIP) and the official page "
            "shows the same address (minor formatting differences are acceptable, e.g., 'St' vs 'Street'). If any component is missing in the provided address, judge 'Incorrect'."
        ),
    )

    # Critical: Phone contact number must be present and match the official page
    phone_leaf = evaluator.add_leaf(
        id=f"Phone_Number_{inst_num}",
        desc="Provides a valid phone contact number",
        parent=inst_parent,
        critical=True,
    )
    phone_text = inst.phone_number or ""
    await evaluator.verify(
        claim=f"The official webpage lists the phone contact number for {institution_name} as '{phone_text}'.",
        node=phone_leaf,
        sources=inst.official_url,
        additional_instruction=(
            "Accept reasonable formatting variants (e.g., '(555) 123-4567', '555-123-4567', '+1 555-123-4567'). "
            "If no phone number is provided in the answer or the number does not appear on the official page, judge 'Incorrect'."
        ),
    )

    # Critical: Early opening hours before 10:00 AM on a weekday
    hours_leaf = evaluator.add_leaf(
        id=f"Early_Opening_Hours_{inst_num}",
        desc="Confirms the institution opens before 10:00 AM on at least one weekday (and states the weekday opening time used for this confirmation)",
        parent=inst_parent,
        critical=True,
    )
    weekday = inst.weekday_open_day or "a weekday"
    open_time = inst.weekday_open_time or "an opening time"
    await evaluator.verify(
        claim=(
            f"On {weekday}, the institution opens at {open_time}, which is before 10:00 AM on a weekday."
        ),
        node=hours_leaf,
        sources=inst.official_url,
        additional_instruction=(
            "Use the official page's hours table/schedule. Only Monday–Friday counts as 'weekday'. Opening times of 10:00 AM or later do NOT satisfy. "
            "If the provided day/time is missing in the answer or the official page does not support opening before 10:00 AM on any weekday, judge 'Incorrect'."
        ),
    )


# -----------------------------------------------------------------------------
# Main evaluation entry
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
    Evaluate an answer for the free-admission public cultural institutions task.
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

    # Extract institutions from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_institutions(),
        template_class=InstitutionsExtraction,
        extraction_name="institutions_extraction",
    )

    institutions = normalize_institutions(extracted)

    # Critical global check: distinct U.S. states across the four institutions
    states = [inst.state for inst in institutions if inst.state and inst.state.strip()]
    distinct_states_result = (len(states) == 4 and len(set(s.strip().lower() for s in states)) == 4)

    evaluator.add_custom_node(
        result=distinct_states_result,
        id="Distinct_US_States",
        desc="The four institutions are located in four different U.S. states (no state is repeated across the set)",
        parent=root,
        critical=True,
    )

    # Add the group node for institutions (optional grouping to match rubric naming)
    group_node = evaluator.add_parallel(
        id="Find_Four_Institutions",
        desc="Identify four free-admission public cultural institutions that open before 10:00 AM on at least one weekday, and provide required contact/location/verification details",
        parent=root,
        critical=False,
    )

    # Verify each institution
    for i, inst in enumerate(institutions):
        await verify_institution(evaluator, group_node, inst, i)

    # Return standard summary
    return evaluator.get_summary()