import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_univ_presidents_dec2025_jan2026"
TASK_DESCRIPTION = (
    "Identify two university presidents in the United States who were appointed to their current presidential positions "
    "between December 1, 2025 and January 31, 2026, and who held dean-level or higher administrative positions (such as "
    "Dean, Chancellor, Provost, or Vice President) immediately before their presidential appointments. For each president, provide: "
    "(1) Their full name, (2) The name of the university where they are serving (or will serve) as president, "
    "(3) The date their appointment was announced, (4) The date they officially assumed or will assume office, "
    "(5) The name of the institution where they held their previous position, (6) The title of their previous administrative position, "
    "(7) The length of time they served in that previous position, and (8) Reference URLs from official university sources or credible news outlets "
    "confirming each piece of information."
)

DATE_RANGE_START = "December 1, 2025"
DATE_RANGE_END = "January 31, 2026"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PresidentRecord(BaseModel):
    # Identification
    name: Optional[str] = None
    current_institution: Optional[str] = None
    current_position_title: Optional[str] = None
    identification_urls: List[str] = Field(default_factory=list)

    # Appointment timeline
    announcement_date: Optional[str] = None
    announcement_urls: List[str] = Field(default_factory=list)
    effective_date: Optional[str] = None
    effective_date_urls: List[str] = Field(default_factory=list)

    # Prior role details
    previous_institution: Optional[str] = None
    previous_position_title: Optional[str] = None
    years_in_position: Optional[str] = None
    prior_role_urls: List[str] = Field(default_factory=list)


class PresidentsExtraction(BaseModel):
    presidents: List[PresidentRecord] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_presidents() -> str:
    return """
    Extract from the answer all university presidents that the answer claims match the following criteria:
    - Appointed to their current presidential positions between December 1, 2025 and January 31, 2026.
    - Immediately before their presidential appointment, they held a dean-level or higher administrative role (e.g., Dean, Chancellor, Provost, Vice President).

    For each president mentioned in the answer, extract the following fields exactly as provided in the answer:
    1) name: Full name of the president.
    2) current_institution: Name of the university where they are (or will be) president.
    3) current_position_title: The title of the position at the current institution (e.g., President, Chancellor, etc.).
    4) identification_urls: An array of URLs (official university pages or credible news articles) confirming the identity and appointment.
    5) announcement_date: The date the appointment was publicly announced.
    6) announcement_urls: An array of URLs that confirm the announcement and its date (official or credible).
    7) effective_date: The date on which they officially assumed or will assume office.
    8) effective_date_urls: An array of URLs that confirm the effective/assumption date (official or credible).
    9) previous_institution: Name of the institution where they held their previous role immediately before the presidential appointment.
    10) previous_position_title: Title of the previous role (e.g., Provost, Vice President, Dean, etc.).
    11) years_in_position: The amount of time they served in the previous role (e.g., "3 years", "since 2021", "July 2022–December 2025").
    12) prior_role_urls: An array of URLs that confirm the previous role details (official or credible).

    IMPORTANT:
    - Only extract information explicitly present in the answer text.
    - If a field is missing in the answer, set it to null (for strings) or an empty list (for arrays).
    - For URLs, include actual URLs if present (plain URLs or within markdown links). If no URL is provided for a given field, leave the array empty.
    - Do not invent or infer any data beyond what the answer states.

    Return a JSON object of the following structure:
    {
      "presidents": [
        {
          "name": string | null,
          "current_institution": string | null,
          "current_position_title": string | null,
          "identification_urls": string[],

          "announcement_date": string | null,
          "announcement_urls": string[],

          "effective_date": string | null,
          "effective_date_urls": string[],

          "previous_institution": string | null,
          "previous_position_title": string | null,
          "years_in_position": string | null,
          "prior_role_urls": string[]
        },
        ...
      ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper                                                                      #
# --------------------------------------------------------------------------- #
def _first_n(items: List[PresidentRecord], n: int) -> List[PresidentRecord]:
    """Return first n items, padding with empty records if needed."""
    result = items[:n]
    while len(result) < n:
        result.append(PresidentRecord())
    return result


def _unique_urls(*url_lists: List[str]) -> List[str]:
    """Combine multiple URL lists and deduplicate while preserving order."""
    seen = set()
    combined: List[str] = []
    for lst in url_lists:
        for u in lst:
            if u and u not in seen:
                seen.add(u)
                combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification for one president                                              #
# --------------------------------------------------------------------------- #
async def verify_president(
    evaluator: Evaluator,
    root_parent,
    pres: PresidentRecord,
    idx_one_based: int,
) -> None:
    # Top-level node for this president (non-critical to allow partial credit across presidents)
    pres_node = evaluator.add_parallel(
        id=f"president_{idx_one_based}",
        desc="First qualifying university president" if idx_one_based == 1 else "Second qualifying university president",
        parent=root_parent,
        critical=False,
    )

    # ---------------- Identification ---------------- #
    ident_node = evaluator.add_parallel(
        id=f"president_{idx_one_based}_identification",
        desc=f"Basic identification information for the {'first' if idx_one_based == 1 else 'second'} president",
        parent=pres_node,
        critical=True,
    )

    # Name provided (existence check)
    evaluator.add_custom_node(
        result=bool(pres.name and pres.name.strip()),
        id=f"president_{idx_one_based}_name",
        desc=f"The full name of the {'first' if idx_one_based == 1 else 'second'} president is provided",
        parent=ident_node,
        critical=True,
    )

    # Current institution provided (existence check)
    evaluator.add_custom_node(
        result=bool(pres.current_institution and pres.current_institution.strip()),
        id=f"president_{idx_one_based}_current_institution",
        desc="The name of the university where this president is serving (or will serve) as president is provided",
        parent=ident_node,
        critical=True,
    )

    # Identification URL provided (existence check)
    evaluator.add_custom_node(
        result=bool(pres.identification_urls and len(pres.identification_urls) > 0),
        id=f"president_{idx_one_based}_identification_url",
        desc="A reference URL from an official university source or credible news outlet confirming the identity and appointment is provided",
        parent=ident_node,
        critical=True,
    )

    # US institution confirmation (verification by URLs)
    us_inst_leaf = evaluator.add_leaf(
        id=f"president_{idx_one_based}_us_institution",
        desc="The university is confirmed to be located in the United States",
        parent=ident_node,
        critical=True,
    )
    claim_us = f"The institution '{pres.current_institution}' is located in the United States."
    await evaluator.verify(
        claim=claim_us,
        node=us_inst_leaf,
        sources=pres.identification_urls,
        additional_instruction=(
            "Use the provided official or credible source(s) to confirm the institution is in the U.S. "
            "Look for city/state or country indicators on the page."
        ),
    )

    # Current position confirmation (verification by URLs)
    curr_pos_leaf = evaluator.add_leaf(
        id=f"president_{idx_one_based}_current_position",
        desc="The title 'President' or equivalent top leadership position is confirmed",
        parent=ident_node,
        critical=True,
    )
    person = pres.name or "the person"
    inst = pres.current_institution or "the university"
    title = pres.current_position_title or "President"
    claim_pos = (
        f"{person} has been appointed to a top leadership role at {inst}, specifically '{title}', "
        f"which is equivalent to a university President or system head (including interim/president-elect/chancellor)."
    )
    await evaluator.verify(
        claim=claim_pos,
        node=curr_pos_leaf,
        sources=pres.identification_urls,
        additional_instruction=(
            "Confirm the page states this person is appointed as President or an equivalent top leadership role "
            "(e.g., Chancellor, President-elect, Interim President, University/System President). Allow reasonable synonyms."
        ),
    )

    # ---------------- Appointment timeline ---------------- #
    timeline_node = evaluator.add_parallel(
        id=f"president_{idx_one_based}_appointment_timeline",
        desc=f"Timeline details of the {'first' if idx_one_based == 1 else 'second'} president's appointment",
        parent=pres_node,
        critical=True,
    )

    # Announcement URL provided (existence check)
    evaluator.add_custom_node(
        result=bool(pres.announcement_urls and len(pres.announcement_urls) > 0),
        id=f"president_{idx_one_based}_announcement_url",
        desc="A reference URL confirming the announcement date is provided",
        parent=timeline_node,
        critical=True,
    )

    # Announcement date verification (must be in Dec 1, 2025 – Jan 31, 2026)
    ann_date_leaf = evaluator.add_leaf(
        id=f"president_{idx_one_based}_announcement_date",
        desc="The date when the appointment was publicly announced is provided and falls between December 1, 2025 and January 31, 2026",
        parent=timeline_node,
        critical=True,
    )
    claim_ann = (
        f"The presidential appointment of {person} at {inst} was announced on {pres.announcement_date}, "
        f"which is between {DATE_RANGE_START} and {DATE_RANGE_END}."
    )
    await evaluator.verify(
        claim=claim_ann,
        node=ann_date_leaf,
        sources=pres.announcement_urls,
        additional_instruction=(
            "Verify the announcement date shown on the source page and confirm that the date lies within the specified window: "
            "December 1, 2025 through January 31, 2026 (inclusive)."
        ),
    )

    # Effective date URL provided (existence check)
    evaluator.add_custom_node(
        result=bool(pres.effective_date_urls and len(pres.effective_date_urls) > 0),
        id=f"president_{idx_one_based}_effective_date_url",
        desc="A reference URL confirming the effective date is provided",
        parent=timeline_node,
        critical=True,
    )

    # Effective date verification
    eff_date_leaf = evaluator.add_leaf(
        id=f"president_{idx_one_based}_effective_date",
        desc="The date when the president officially assumed or will assume office is provided",
        parent=timeline_node,
        critical=True,
    )
    claim_eff = (
        f"{person} officially assumed (or will assume) the office of President at {inst} on {pres.effective_date}."
    )
    await evaluator.verify(
        claim=claim_eff,
        node=eff_date_leaf,
        sources=pres.effective_date_urls,
        additional_instruction=(
            "Confirm that the page clearly states the effective date (assumption of office) for this presidency; "
            "future scheduled dates are acceptable."
        ),
    )

    # ---------------- Prior role ---------------- #
    prior_node = evaluator.add_parallel(
        id=f"president_{idx_one_based}_prior_role",
        desc=f"Information about the {'first' if idx_one_based == 1 else 'second'} president's previous administrative position",
        parent=pres_node,
        critical=True,
    )

    # Prior role URL provided (existence check)
    evaluator.add_custom_node(
        result=bool(pres.prior_role_urls and len(pres.prior_role_urls) > 0),
        id=f"president_{idx_one_based}_prior_role_url",
        desc="A reference URL confirming the previous position details is provided",
        parent=prior_node,
        critical=True,
    )

    # Previous institution provided (existence check)
    evaluator.add_custom_node(
        result=bool(pres.previous_institution and pres.previous_institution.strip()),
        id=f"president_{idx_one_based}_previous_institution",
        desc="The name of the institution where the president held their previous position is provided",
        parent=prior_node,
        critical=True,
    )

    # Previous institution type verification (higher education institution)
    prev_inst_type_leaf = evaluator.add_leaf(
        id=f"president_{idx_one_based}_previous_institution_type",
        desc="The previous institution is confirmed to be a higher education institution",
        parent=prior_node,
        critical=True,
    )
    claim_prev_inst_type = (
        f"'{pres.previous_institution}' is a higher education institution (e.g., university or college)."
    )
    await evaluator.verify(
        claim=claim_prev_inst_type,
        node=prev_inst_type_leaf,
        sources=pres.prior_role_urls,
        additional_instruction=(
            "Confirm via the provided source(s) that the previous institution is a higher education institution. "
            "Look for descriptors such as 'university', 'college', 'campus', or similar."
        ),
    )

    # Previous position is dean-level or higher (verification by URLs with reasoning)
    prev_pos_leaf = evaluator.add_leaf(
        id=f"president_{idx_one_based}_previous_position",
        desc="The title of the previous position is provided and confirmed to be dean-level or higher (e.g., Dean, Chancellor, Provost, Vice President)",
        parent=prior_node,
        critical=True,
    )
    prev_title = pres.previous_position_title or "unknown prior role"
    claim_prev_pos = (
        f"Immediately prior to the presidency, {person} held the role '{prev_title}' at {pres.previous_institution}, "
        f"and this role is dean-level or higher (e.g., Dean, Chancellor, Provost, Vice President)."
    )
    await evaluator.verify(
        claim=claim_prev_pos,
        node=prev_pos_leaf,
        sources=pres.prior_role_urls,
        additional_instruction=(
            "First, verify the page lists the exact prior role title. Then, use generally accepted academic administrative hierarchies to conclude "
            "whether the role qualifies as dean-level or higher (Dean, Chancellor, Provost, Vice President, Executive/Senior VP, Vice Provost). "
            "Minor variations (interim, acting) still count if the role is at that seniority."
        ),
    )

    # Immediately before (no intervening positions)
    immediately_leaf = evaluator.add_leaf(
        id=f"president_{idx_one_based}_immediately_before",
        desc="The previous position was held immediately before the presidential appointment (no intervening positions)",
        parent=prior_node,
        critical=True,
    )
    claim_immediately = (
        f"{person} held the role '{prev_title}' at {pres.previous_institution} immediately before the presidential appointment, "
        f"with no intervening positions."
    )
    await evaluator.verify(
        claim=claim_immediately,
        node=immediately_leaf,
        sources=_unique_urls(pres.identification_urls, pres.prior_role_urls),
        additional_instruction=(
            "Confirm the announcement or biography indicates the prior role was the most recent role (e.g., 'most recently', 'previously served as'). "
            "If the page shows intervening roles, this should fail."
        ),
    )

    # Years in position
    years_leaf = evaluator.add_leaf(
        id=f"president_{idx_one_based}_years_in_position",
        desc="The number of years or the time period the president served in their previous position is provided",
        parent=prior_node,
        critical=True,
    )
    claim_years = (
        f"{person} served in the prior role for {pres.years_in_position}."
    )
    await evaluator.verify(
        claim=claim_years,
        node=years_leaf,
        sources=pres.prior_role_urls,
        additional_instruction=(
            "Verify that the duration or dates of service in the prior role match the page (e.g., ranges or total years). "
            "Allow reasonable interpretations (e.g., 'about three years')."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate the answer for identifying two qualifying university presidents appointed between Dec 1, 2025 and Jan 31, 2026,
    with dean-level or higher prior roles immediately before appointment.
    """
    # Initialize evaluator with parallel root
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

    # Extract presidents from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_presidents(),
        template_class=PresidentsExtraction,
        extraction_name="presidents_extraction",
    )

    # Use only the first two presidents, pad if fewer
    pres_list = _first_n(extracted.presidents, 2)

    # Build verification tree per president
    await verify_president(evaluator, root, pres_list[0], idx_one_based=1)
    await verify_president(evaluator, root, pres_list[1], idx_one_based=2)

    # Return summary
    return evaluator.get_summary()