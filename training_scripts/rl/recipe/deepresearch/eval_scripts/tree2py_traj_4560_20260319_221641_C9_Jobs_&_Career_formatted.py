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
TASK_ID = "state_hs_athletic_execs_2020_2024"
TASK_DESCRIPTION = """
As of March 2026, identify three executive-level leadership positions within state high school athletic associations from three different U.S. states. Each identified individual must meet all of the following criteria:

1. Appointment Timeframe: The individual was appointed or promoted to their current executive position (such as Executive Director or Associate Executive Director) between January 1, 2020 and December 31, 2024.

2. Experience Requirement: The individual has at least 20 years of total experience in education or educational administration.

3. Administrative Background: The individual previously held or currently holds a superintendent or principal position at a K-12 school or school district.

4. Organizational Affiliation: The state high school athletic association where the individual serves must be a member of the National Federation of State High School Associations (NFHS).

For each of the three positions, provide:

- Appointment Details: The specific appointment date (month, day, year), official start date if different, and information about their predecessor if applicable
- Career Background: Total years of education experience, details about their superintendent or principal service (including specific school districts or schools and duration), and how they began their career in education
- Current Role: Their official executive title, the full name of the state athletic association, the U.S. state it serves, and key responsibilities
- Organizational Context: Confirmation of NFHS membership and any service on the NFHS Board of Directors if applicable

Include URL references from official sources to support the information for each position.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PositionItem(BaseModel):
    # Person and role basics
    person_name: Optional[str] = None
    official_title: Optional[str] = None
    association_full_name: Optional[str] = None
    association_state: Optional[str] = None
    role_responsibilities: Optional[str] = None
    role_urls: List[str] = Field(default_factory=list)

    # Appointment details
    appointment_date: Optional[str] = None  # e.g., "June 15, 2021"
    official_start_date: Optional[str] = None  # if different
    predecessor_name: Optional[str] = None
    predecessor_tenure_end: Optional[str] = None
    appointment_urls: List[str] = Field(default_factory=list)

    # Career background
    total_experience: Optional[str] = None  # free text like "over 25 years"
    administrative_role_summary: Optional[str] = None  # free text summary including districts/schools and durations
    career_start: Optional[str] = None  # e.g., "began as a teacher in 1996"
    career_urls: List[str] = Field(default_factory=list)

    # Organizational/NFHS context
    nfhs_membership_urls: List[str] = Field(default_factory=list)  # Prefer NFHS.org membership directory page(s)
    nfhs_board_service: Optional[str] = None  # description if applicable
    organizational_urls: List[str] = Field(default_factory=list)  # other org context URLs


class PositionsExtraction(BaseModel):
    positions: List[PositionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
Extract up to three (3) executive-level leaders of U.S. state high school athletic associations mentioned in the answer.
For each position, fill the following fields as precisely as they appear in the answer. Use null for any field not provided.
Always extract explicit URLs as provided in the answer; do not invent URLs.

Return a JSON object with field "positions": an array of up to three PositionItem objects, each with:

- person_name: Full name of the individual
- official_title: Executive title (e.g., Executive Director, Associate Executive Director)
- association_full_name: Full name of the state high school athletic association
- association_state: The U.S. state served by the association (e.g., "Ohio", "California")
- role_responsibilities: Key duties/responsibilities described for the role (free text, if present)
- role_urls: All URLs the answer cites that support current role/title/association/state

Appointment details:
- appointment_date: Specific appointment or promotion announcement date in Month Day, Year format as given in the answer (e.g., "June 15, 2021"). If the answer gives a different format, keep it as-is.
- official_start_date: If different from announcement, the official start date (free text)
- predecessor_name: If provided, whom they replaced
- predecessor_tenure_end: If provided, when the predecessor’s tenure ended (free text)
- appointment_urls: All URLs cited that announce or confirm the appointment/promotion

Career background:
- total_experience: The total years of experience in education/educational administration as explicitly stated (e.g., "over 20 years", "25 years")
- administrative_role_summary: Details about superintendent or principal service, including the district/school and durations if provided (free text)
- career_start: How they began their education career (initial role and approximate year if provided; free text)
- career_urls: All URLs supporting the career background (experience and administrative roles)

Organizational context:
- nfhs_membership_urls: URLs that confirm the association is an NFHS member (prefer NFHS.org membership directory page if available)
- nfhs_board_service: If applicable, any statement about the individual serving (or named to serve) on the NFHS Board of Directors (free text)
- organizational_urls: Any URLs supporting organizational context (in addition to or separate from nfhs_membership_urls)

Rules:
- Only include URLs explicitly present in the answer. Extract them as full URLs.
- If multiple URLs are present for the same field, include all of them.
- Do not infer or fabricate any dates, names, or facts.
- Maintain the exact strings from the answer for dates and names.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not _nonempty(u):
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _merge_urls(*url_lists: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in url_lists:
        merged.extend(lst or [])
    return _dedup_urls(merged)


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def _verify_appointment_details(
    evaluator: Evaluator,
    parent_node,
    pos: PositionItem,
    idx: int
) -> None:
    # Parent node (non-critical to allow optional subchecks)
    app_parent = evaluator.add_parallel(
        id=f"position_{idx}_appointment_details",
        desc=f"Verify appointment date and transition context for Position {idx}",
        parent=parent_node,
        critical=False
    )

    # 1) Appointment within timeframe (2020-01-01 to 2024-12-31), critical
    n1 = evaluator.add_leaf(
        id=f"position_{idx}_appointment_date",
        desc="The individual was appointed or promoted to their current executive position between January 1, 2020 and December 31, 2024",
        parent=app_parent,
        critical=True
    )
    claim_timeframe = (
        f"{pos.person_name or 'The individual'} was appointed or promoted to the role "
        f"{pos.official_title or 'an executive role'} at {pos.association_full_name or 'the association'} "
        f"between January 1, 2020 and December 31, 2024 (inclusive)."
    )
    await evaluator.verify(
        claim=claim_timeframe,
        node=n1,
        sources=_dedup_urls(pos.appointment_urls),
        additional_instruction=(
            "Use the cited announcement/press release or official news page to check the appointment date. "
            "Confirm that the appointment/promotion date stated on the page lies within the given timeframe."
        )
    )

    # 2) Specific appointment date (critical)
    n2 = evaluator.add_leaf(
        id=f"position_{idx}_specific_appointment_date",
        desc="Provide the specific appointment date (month, day, year)",
        parent=app_parent,
        critical=True
    )
    if _nonempty(pos.appointment_date):
        claim_specific_date = (
            f"{pos.person_name or 'The individual'} was appointed or promoted to "
            f"{pos.official_title or 'the executive role'} at {pos.association_full_name or 'the association'} "
            f"on {pos.appointment_date}."
        )
        await evaluator.verify(
            claim=claim_specific_date,
            node=n2,
            sources=_dedup_urls(pos.appointment_urls),
            additional_instruction="Verify the exact appointment/promotion date as written. Allow minor format variants."
        )
    else:
        # Missing specific date in the answer -> fail critical node
        n2.score = 0.0
        n2.status = "failed"

    # 3) Official start date (if different) (non-critical)
    n3 = evaluator.add_leaf(
        id=f"position_{idx}_official_start",
        desc="If different from announcement date, provide the official start date of the position",
        parent=app_parent,
        critical=False
    )
    if _nonempty(pos.official_start_date):
        claim_start = (
            f"{pos.person_name or 'The individual'}'s official start date for "
            f"{pos.official_title or 'the role'} at {pos.association_full_name or 'the association'} "
            f"was {pos.official_start_date}."
        )
        await evaluator.verify(
            claim=claim_start,
            node=n3,
            sources=_dedup_urls(pos.appointment_urls + pos.role_urls),
            additional_instruction="Confirm the stated official start date if the page distinguishes start vs. announcement."
        )
    else:
        n3.score = 0.0
        n3.status = "skipped"

    # 4) Predecessor info (non-critical)
    n4 = evaluator.add_leaf(
        id=f"position_{idx}_predecessor_info",
        desc="If applicable, identify who the individual replaced and when that person's tenure ended",
        parent=app_parent,
        critical=False
    )
    if _nonempty(pos.predecessor_name) or _nonempty(pos.predecessor_tenure_end):
        part_end = f", whose tenure ended {pos.predecessor_tenure_end}" if _nonempty(pos.predecessor_tenure_end) else ""
        claim_pred = (
            f"{pos.person_name or 'The individual'} replaced {pos.predecessor_name or 'the predecessor'}"
            f"{part_end}."
        )
        await evaluator.verify(
            claim=claim_pred,
            node=n4,
            sources=_dedup_urls(_merge_urls(pos.appointment_urls, pos.role_urls)),
            additional_instruction="Verify predecessor identification and, if provided, the predecessor's end of tenure."
        )
    else:
        n4.score = 0.0
        n4.status = "skipped"

    # 5) Appointment URL exists and supports appointment (critical)
    n5 = evaluator.add_leaf(
        id=f"position_{idx}_appointment_url",
        desc="Provide URL reference supporting the appointment information",
        parent=app_parent,
        critical=True
    )
    ap_urls = _dedup_urls(pos.appointment_urls)
    if not ap_urls:
        n5.score = 0.0
        n5.status = "failed"
    else:
        claim_ap_url = (
            f"An official or authoritative source confirms that "
            f"{pos.person_name or 'the individual'} was appointed or promoted to "
            f"{pos.official_title or 'the executive role'} at {pos.association_full_name or 'the association'}."
        )
        await evaluator.verify(
            claim=claim_ap_url,
            node=n5,
            sources=ap_urls,
            additional_instruction="Prefer state association press releases, NFHS news, or reputable local/state news outlets."
        )


async def _verify_career_background(
    evaluator: Evaluator,
    parent_node,
    pos: PositionItem,
    idx: int
) -> None:
    car_parent = evaluator.add_parallel(
        id=f"position_{idx}_career_background",
        desc=f"Verify the individual's career background meets experience requirements for Position {idx}",
        parent=parent_node,
        critical=False
    )

    # Union of potential career-supporting URLs
    career_sources = _dedup_urls(_merge_urls(pos.career_urls, pos.role_urls, pos.appointment_urls))

    # 1) Total experience >= 20 years (critical)
    n1 = evaluator.add_leaf(
        id=f"position_{idx}_total_experience",
        desc="The individual has at least 20 years of total experience in education or educational administration",
        parent=car_parent,
        critical=True
    )
    claim_years = (
        f"{pos.person_name or 'The individual'} has at least 20 years of experience in education or educational administration."
    )
    await evaluator.verify(
        claim=claim_years,
        node=n1,
        sources=career_sources,
        additional_instruction="Accept reasonable phrasings like 'more than 20 years', 'over 25 years', or specific totals ≥ 20."
    )

    # 2) Administrative role: superintendent or principal (critical)
    n2 = evaluator.add_leaf(
        id=f"position_{idx}_administrative_role",
        desc="The individual previously held or currently holds a superintendent or principal position",
        parent=car_parent,
        critical=True
    )
    claim_admin = (
        f"{pos.person_name or 'The individual'} has served as a superintendent or as a principal at a K-12 district or school."
    )
    await evaluator.verify(
        claim=claim_admin,
        node=n2,
        sources=career_sources,
        additional_instruction="Look for superintendent and/or principal positions explicitly; K-12 district/school contexts qualify."
    )

    # 3) Administrative details (critical)
    n3 = evaluator.add_leaf(
        id=f"position_{idx}_administrative_details",
        desc="Provide details about their superintendent or principal service, including specific school districts or schools and duration, for whichever role(s) they held",
        parent=car_parent,
        critical=True
    )
    if _nonempty(pos.administrative_role_summary):
        claim_admin_details = (
            f"{pos.person_name or 'The individual'} held superintendent/principal role(s) as described: "
            f"{pos.administrative_role_summary}."
        )
        await evaluator.verify(
            claim=claim_admin_details,
            node=n3,
            sources=career_sources,
            additional_instruction="Confirm districts/schools and any durations mentioned in the summary."
        )
    else:
        # If details missing in answer, it fails this critical requirement
        n3.score = 0.0
        n3.status = "failed"

    # 4) Career start (non-critical)
    n4 = evaluator.add_leaf(
        id=f"position_{idx}_career_start",
        desc="Provide information about how the individual began their career in education (initial role and approximate year)",
        parent=car_parent,
        critical=False
    )
    if _nonempty(pos.career_start):
        claim_cstart = (
            f"{pos.person_name or 'The individual'} began their career in education as stated: {pos.career_start}."
        )
        await evaluator.verify(
            claim=claim_cstart,
            node=n4,
            sources=career_sources,
            additional_instruction="Match the initial role and approximate year if stated."
        )
    else:
        n4.score = 0.0
        n4.status = "skipped"

    # 5) Career URLs exist and support background (critical)
    n5 = evaluator.add_leaf(
        id=f"position_{idx}_career_url",
        desc="Provide URL reference supporting the career background information",
        parent=car_parent,
        critical=True
    )
    if not career_sources:
        n5.score = 0.0
        n5.status = "failed"
    else:
        claim_career_urls = (
            f"These sources support {pos.person_name or 'the individual'}'s total experience and superintendent/principal background."
        )
        await evaluator.verify(
            claim=claim_career_urls,
            node=n5,
            sources=career_sources,
            additional_instruction="At least one page should substantiate years of experience and admin role(s)."
        )


async def _verify_current_role(
    evaluator: Evaluator,
    parent_node,
    pos: PositionItem,
    idx: int,
    prior_states: List[str]
) -> None:
    role_parent = evaluator.add_parallel(
        id=f"position_{idx}_current_role",
        desc=f"Verify the individual's current organizational role and title for Position {idx}",
        parent=parent_node,
        critical=False
    )

    role_sources = _dedup_urls(_merge_urls(pos.role_urls, pos.appointment_urls))

    # 1) Official executive title (critical)
    n1 = evaluator.add_leaf(
        id=f"position_{idx}_official_title",
        desc="Provide the individual's official executive-level title (e.g., Executive Director, Associate Executive Director)",
        parent=role_parent,
        critical=True
    )
    if _nonempty(pos.official_title):
        claim_title = (
            f"{pos.person_name or 'The individual'} holds the title '{pos.official_title}' at "
            f"{pos.association_full_name or 'the association'}."
        )
        await evaluator.verify(
            claim=claim_title,
            node=n1,
            sources=role_sources,
            additional_instruction="Verify exact or equivalent official title."
        )
    else:
        n1.score = 0.0
        n1.status = "failed"

    # 2) Association full name (critical)
    n2 = evaluator.add_leaf(
        id=f"position_{idx}_association_name",
        desc="Provide the full name of the state high school athletic association",
        parent=role_parent,
        critical=True
    )
    if _nonempty(pos.association_full_name):
        claim_assoc = f"The full name of the association is '{pos.association_full_name}'."
        await evaluator.verify(
            claim=claim_assoc,
            node=n2,
            sources=role_sources,
            additional_instruction="Confirm the exact formal name as used by the organization."
        )
    else:
        n2.score = 0.0
        n2.status = "failed"

    # 3) Association state (critical)
    n3 = evaluator.add_leaf(
        id=f"position_{idx}_association_state",
        desc="Identify which U.S. state the association serves",
        parent=role_parent,
        critical=True
    )
    if _nonempty(pos.association_state):
        different_note = ""
        if prior_states:
            different_note = (
                f" Ensure the state '{pos.association_state}' is different from previously used states: {prior_states}."
            )
        claim_state = (
            f"{pos.association_full_name or 'The association'} serves the U.S. state of {pos.association_state}."
        )
        await evaluator.verify(
            claim=claim_state,
            node=n3,
            sources=role_sources,
            additional_instruction="Confirm the state served by the association." + different_note
        )
    else:
        n3.score = 0.0
        n3.status = "failed"

    # 4) Role responsibilities (non-critical)
    n4 = evaluator.add_leaf(
        id=f"position_{idx}_role_responsibilities",
        desc="Describe key responsibilities or duties of the position",
        parent=role_parent,
        critical=False
    )
    if _nonempty(pos.role_responsibilities):
        claim_resp = (
            f"Key duties for the role include: {pos.role_responsibilities}."
        )
        await evaluator.verify(
            claim=claim_resp,
            node=n4,
            sources=role_sources,
            additional_instruction="Verify that the described responsibilities align with the official role description."
        )
    else:
        n4.score = 0.0
        n4.status = "skipped"

    # 5) Role URLs exist and support current role (critical)
    n5 = evaluator.add_leaf(
        id=f"position_{idx}_role_url",
        desc="Provide URL reference supporting current role information",
        parent=role_parent,
        critical=True
    )
    if not role_sources:
        n5.score = 0.0
        n5.status = "failed"
    else:
        claim_role_urls = (
            f"These sources confirm {pos.person_name or 'the individual'}'s current executive role/title and association."
        )
        await evaluator.verify(
            claim=claim_role_urls,
            node=n5,
            sources=role_sources,
            additional_instruction="At least one link must substantiate current role/title at the stated association."
        )


async def _verify_organizational_context(
    evaluator: Evaluator,
    parent_node,
    pos: PositionItem,
    idx: int
) -> None:
    org_parent = evaluator.add_parallel(
        id=f"position_{idx}_organizational_context",
        desc=f"Verify organizational affiliation and structure for Position {idx}",
        parent=parent_node,
        critical=False
    )

    nfhs_sources = _dedup_urls(_merge_urls(pos.nfhs_membership_urls, pos.organizational_urls, pos.role_urls))

    # 1) NFHS membership (critical)
    n1 = evaluator.add_leaf(
        id=f"position_{idx}_nfhs_membership",
        desc="Confirm the state association is a member of the National Federation of State High School Associations (NFHS)",
        parent=org_parent,
        critical=True
    )
    claim_nfhs = (
        f"{pos.association_full_name or 'The association'} is a member of the National Federation of State High School Associations (NFHS)."
    )
    await evaluator.verify(
        claim=claim_nfhs,
        node=n1,
        sources=nfhs_sources,
        additional_instruction="Prefer NFHS.org membership directory pages; an official NFHS or association page stating NFHS membership suffices."
    )

    # 2) NFHS Board service (non-critical)
    n2 = evaluator.add_leaf(
        id=f"position_{idx}_nfhs_board",
        desc="If applicable, identify if the individual serves or has been named to serve on the NFHS Board of Directors",
        parent=org_parent,
        critical=False
    )
    if _nonempty(pos.nfhs_board_service):
        claim_board = (
            f"{pos.person_name or 'The individual'} serves or has been named to serve on the NFHS Board of Directors: "
            f"{pos.nfhs_board_service}."
        )
        await evaluator.verify(
            claim=claim_board,
            node=n2,
            sources=nfhs_sources,
            additional_instruction="Confirm board membership/appointment details; skip if none mentioned."
        )
    else:
        n2.score = 0.0
        n2.status = "skipped"

    # 3) Organizational URL exists and supports context (critical)
    n3 = evaluator.add_leaf(
        id=f"position_{idx}_organizational_url",
        desc="Provide URL reference supporting organizational context",
        parent=org_parent,
        critical=True
    )
    if not nfhs_sources:
        n3.score = 0.0
        n3.status = "failed"
    else:
        claim_org_urls = (
            f"These sources support the organizational context for "
            f"{pos.association_full_name or 'the association'} (e.g., NFHS membership or related info)."
        )
        await evaluator.verify(
            claim=claim_org_urls,
            node=n3,
            sources=nfhs_sources,
            additional_instruction="At least one URL should explicitly support NFHS membership or closely related organizational context."
        )


async def verify_position(
    evaluator: Evaluator,
    parent_node,
    pos: PositionItem,
    idx: int,
    prior_states: List[str]
) -> None:
    """
    Build and verify the subtree for a single position (idx in {1,2,3}).
    """
    # Appointment details
    await _verify_appointment_details(evaluator, parent_node, pos, idx)

    # Career background
    await _verify_career_background(evaluator, parent_node, pos, idx)

    # Current role and association information
    await _verify_current_role(evaluator, parent_node, pos, idx, prior_states)

    # Organizational context (NFHS, etc.)
    await _verify_organizational_context(evaluator, parent_node, pos, idx)


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
    Evaluate an answer for the state high school athletic associations executive leadership task.
    """
    # Initialize evaluator with a non-critical root to allow mixed criticality inside
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # three positions evaluated independently
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

    # Extract structured positions data
    extracted = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction"
    )

    # Normalize to exactly 3 positions (pad with empty if fewer; trim if more)
    positions: List[PositionItem] = list(extracted.positions or [])
    positions = positions[:3]
    while len(positions) < 3:
        positions.append(PositionItem())

    # Build top-level nodes for each position
    prior_states: List[str] = []
    for i in range(3):
        pos_node = evaluator.add_parallel(
            id=f"position_{i+1}",
            desc=f"{['First', 'Second', 'Third'][i]} executive-level position meeting all specified criteria",
            parent=root,
            critical=False
        )

        # Verify this position
        await verify_position(
            evaluator=evaluator,
            parent_node=pos_node,
            pos=positions[i],
            idx=i+1,
            prior_states=prior_states
        )

        # Track states to encourage distinctness in subsequent checks
        if _nonempty(positions[i].association_state):
            prior_states.append(positions[i].association_state.strip())

    # Return structured result
    return evaluator.get_summary()