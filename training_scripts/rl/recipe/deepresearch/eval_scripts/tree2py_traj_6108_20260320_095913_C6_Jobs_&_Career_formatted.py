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
TASK_ID = "leadership_positions_2025_2026"
TASK_DESCRIPTION = """
Identify four distinct leadership positions in education and collegiate athletics that were filled between November 1, 2025, and March 31, 2026.

Requirements:
- Two collegiate head coaching positions with announcements between Nov 1, 2025 and Mar 31, 2026; multi-year (>=3 years), >=$2M base per year, >=20 years prior coaching experience.
- Two Texas school district superintendent (including acting/interim) positions with announcements between Dec 1, 2025 and Mar 31, 2026; master's degree or higher; >=2 years as principal; >=30 years in public education; prior district-level leadership roles.

For each position, provide: name, institution/district, position title, announcement date, and verifiable reference URL(s) supporting all required criteria.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CoachPosition(BaseModel):
    name: Optional[str] = None
    institution: Optional[str] = None
    title: Optional[str] = None
    announcement_date: Optional[str] = None
    contract_length_years: Optional[str] = None  # keep as string to be robust
    annual_base_compensation: Optional[str] = None  # keep as string to be robust
    prior_coaching_years: Optional[str] = None  # keep as string to be robust
    source_urls: List[str] = Field(default_factory=list)


class SuperintendentPosition(BaseModel):
    name: Optional[str] = None
    district: Optional[str] = None
    title: Optional[str] = None
    announcement_date: Optional[str] = None
    highest_degree: Optional[str] = None
    principal_experience_years: Optional[str] = None
    total_years_public_education: Optional[str] = None
    prior_district_leadership_roles: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


class PositionsExtraction(BaseModel):
    coaches: List[CoachPosition] = Field(default_factory=list)
    superintendents: List[SuperintendentPosition] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
    Extract structured information for up to TWO collegiate head coaching positions and up to TWO Texas school district superintendent positions described in the answer.

    Create a JSON object with two arrays: "coaches" and "superintendents".
    For each "coaches" item, extract:
      - name: individual's name
      - institution: college or university name
      - title: specific position title (e.g., Head Football Coach)
      - announcement_date: the date the hiring or contract signing was publicly announced (string as written)
      - contract_length_years: the described contract length (e.g., "5 years", "through 2030–31 season"); do NOT convert to a number
      - annual_base_compensation: the described annual base pay (e.g., "$2.3 million", "at least $2 million"), keep original formatting
      - prior_coaching_years: total years of coaching experience at professional or collegiate levels, as stated (e.g., "22 years", "more than 20 years")
      - source_urls: list of all URLs in the answer that support any of these facts for this position (press releases, news, official pages)

    For each "superintendents" item, extract:
      - name: individual's name
      - district: school district name
      - title: specific position title (e.g., Superintendent, Acting Superintendent, Interim Superintendent)
      - announcement_date: the date the appointment was publicly announced (string as written)
      - highest_degree: the individual's highest degree (e.g., "M.Ed.", "Ed.D.", "Ph.D.", "master's", "doctorate")
      - principal_experience_years: stated years serving as a school principal (e.g., "3 years", "at least two years")
      - total_years_public_education: total years in public education (e.g., "30 years", "over 35 years")
      - prior_district_leadership_roles: list of prior district-level leadership roles (e.g., "Assistant Superintendent", "Deputy Superintendent", "Executive Director")
      - source_urls: list of all URLs in the answer that support any of these facts for this position

    Rules:
    - Only extract items explicitly mentioned in the answer.
    - Do not invent information. If a field is not present, set it to null (or empty list for arrays).
    - Keep numbers and dates as strings exactly as written; do not normalize formats.
    - Include all URLs that support any required criteria, including separate pages for contracts, resumes, bios, or press releases.
    - If more than two items are present for a category, include only the first two mentioned.
    """


# --------------------------------------------------------------------------- #
# Helper functions to add presence checks                                     #
# --------------------------------------------------------------------------- #
def _add_basic_info_presence_nodes_for_coach(evaluator: Evaluator, parent, idx: int, pos: CoachPosition):
    basic_node = evaluator.add_parallel(
        id=f"coach_{idx}_basic_info",
        desc="Basic identifying information for the position is provided",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(pos.name and pos.name.strip()),
        id=f"coach_{idx}_individual_name",
        desc="The individual's name is provided",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(pos.institution and pos.institution.strip()),
        id=f"coach_{idx}_institution_name",
        desc="The college or university name is provided",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(pos.title and pos.title.strip()),
        id=f"coach_{idx}_position_title",
        desc="The specific position title is provided",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(pos.announcement_date and pos.announcement_date.strip()),
        id=f"coach_{idx}_announcement_date",
        desc="The date the appointment was announced is provided",
        parent=basic_node,
        critical=True
    )
    # Additional presence: references provided (to enforce source-grounding)
    refs_node = evaluator.add_custom_node(
        result=bool(pos.source_urls and len(pos.source_urls) > 0),
        id=f"coach_{idx}_references_provided",
        desc="At least one reference URL is provided for this position",
        parent=basic_node,
        critical=True
    )
    return basic_node, refs_node


def _add_basic_info_presence_nodes_for_super(evaluator: Evaluator, parent, idx: int, pos: SuperintendentPosition):
    basic_node = evaluator.add_parallel(
        id=f"super_{idx}_basic_info",
        desc="Basic identifying information for the position is provided",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(pos.name and pos.name.strip()),
        id=f"super_{idx}_individual_name",
        desc="The individual's name is provided",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(pos.district and pos.district.strip()),
        id=f"super_{idx}_district_name",
        desc="The school district name is provided",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(pos.title and pos.title.strip()),
        id=f"super_{idx}_position_title",
        desc="The specific position title is provided",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(pos.announcement_date and pos.announcement_date.strip()),
        id=f"super_{idx}_announcement_date",
        desc="The date the appointment was announced is provided",
        parent=basic_node,
        critical=True
    )
    # Additional presence: references provided (to enforce source-grounding)
    refs_node = evaluator.add_custom_node(
        result=bool(pos.source_urls and len(pos.source_urls) > 0),
        id=f"super_{idx}_references_provided",
        desc="At least one reference URL is provided for this position",
        parent=basic_node,
        critical=True
    )
    return basic_node, refs_node


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_coach_position(
    evaluator: Evaluator,
    parent_node,
    pos: CoachPosition,
    idx: int,
) -> None:
    # Position node (non-critical aggregator for partial credit across positions)
    position_node = evaluator.add_parallel(
        id=f"position_{idx+1}_collegiate_coach",
        desc="Collegiate athletics head coaching position meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    # Basic presence checks
    _, refs_prereq = _add_basic_info_presence_nodes_for_coach(evaluator, position_node, idx, pos)

    # Position Type verification
    pos_type_node = evaluator.add_leaf(
        id=f"coach_{idx}_position_type",
        desc="The position is a head coach role at a college or university",
        parent=position_node,
        critical=True
    )
    claim_pos_type = (
        f"At least one of the cited pages announces that "
        f"{pos.name or 'the individual'} was hired/appointed as a head coach at "
        f"{pos.institution or 'a college or university'}. "
        f"It should be a collegiate (not professional) head coaching role."
    )
    await evaluator.verify(
        claim=claim_pos_type,
        node=pos_type_node,
        sources=pos.source_urls,
        additional_instruction="Accept standard variants like 'Head Coach', 'Men's/Women's Head Coach', or sport-specific (e.g., Football Head Coach). Confirm it's a college/university role.",
        extra_prerequisites=[refs_prereq],
    )

    # Appointment timing verification
    timing_node = evaluator.add_leaf(
        id=f"coach_{idx}_appointment_timing",
        desc="The hiring or contract signing was announced between November 1, 2025 and March 31, 2026",
        parent=position_node,
        critical=True
    )
    claim_timing = (
        "The cited page(s) show the public hiring or contract-signing announcement date "
        "falling between November 1, 2025 and March 31, 2026 (inclusive)."
    )
    await evaluator.verify(
        claim=claim_timing,
        node=timing_node,
        sources=pos.source_urls,
        additional_instruction="Use the press release or article publish date or an explicitly stated announcement date. Consider time zones; treat boundary dates inclusively.",
        extra_prerequisites=[refs_prereq],
    )

    # Contract Terms (critical parallel)
    contract_node = evaluator.add_parallel(
        id=f"coach_{idx}_contract_terms",
        desc="Contract details meet minimum requirements",
        parent=position_node,
        critical=True
    )
    # Contract reference presence (critical)
    contract_ref_node = evaluator.add_custom_node(
        result=bool(pos.source_urls and len(pos.source_urls) > 0),
        id=f"coach_{idx}_contract_reference",
        desc="Contract details are documented with a verifiable source URL",
        parent=contract_node,
        critical=True
    )
    # Multi-year contract (>=3 years)
    multi_year_node = evaluator.add_leaf(
        id=f"coach_{idx}_multi_year_contract",
        desc="The contract is for at least 3 years",
        parent=contract_node,
        critical=True
    )
    claim_multi_year = (
        "The cited page(s) explicitly report a multi-year contract with a term of at least 3 years "
        "(e.g., '5-year deal', 'through the 2029 season' equating to >=3 years)."
    )
    await evaluator.verify(
        claim=claim_multi_year,
        node=multi_year_node,
        sources=pos.source_urls,
        additional_instruction="Look for explicit contract length or a phrasing implying >=3 years (e.g., through a specific season).",
        extra_prerequisites=[contract_ref_node],
    )
    # Minimum compensation (>= $2M/year base)
    min_comp_node = evaluator.add_leaf(
        id=f"coach_{idx}_min_compensation",
        desc="The annual base compensation is at least $2 million per year",
        parent=contract_node,
        critical=True
    )
    claim_min_comp = (
        "The cited page(s) state that the coach's annual BASE compensation is at least $2,000,000 per year. "
        "Accept synonymous phrasing like 'base salary', 'guaranteed salary', or 'base pay'."
    )
    await evaluator.verify(
        claim=claim_min_comp,
        node=min_comp_node,
        sources=pos.source_urls,
        additional_instruction="If ranges or totals are given, ensure the base component is >= $2M per year. Ignore performance bonuses when assessing the base.",
        extra_prerequisites=[contract_ref_node],
    )

    # Prior Experience (critical parallel)
    experience_node = evaluator.add_parallel(
        id=f"coach_{idx}_prior_experience",
        desc="The individual has extensive prior coaching experience",
        parent=position_node,
        critical=True
    )
    # Experience reference presence (critical)
    exp_ref_node = evaluator.add_custom_node(
        result=bool(pos.source_urls and len(pos.source_urls) > 0),
        id=f"coach_{idx}_experience_reference",
        desc="Experience details are documented with a verifiable source URL",
        parent=experience_node,
        critical=True
    )
    # Years of experience (>=20 years)
    years_exp_node = evaluator.add_leaf(
        id=f"coach_{idx}_years_experience",
        desc="The individual has at least 20 years of coaching experience at the professional or collegiate level",
        parent=experience_node,
        critical=True
    )
    claim_years_exp = (
        "The cited page(s) indicate the individual has at least 20 years of prior coaching experience "
        "at the professional and/or collegiate level (counting assistant, position, coordinator, or head coach roles)."
    )
    await evaluator.verify(
        claim=claim_years_exp,
        node=years_exp_node,
        sources=pos.source_urls,
        additional_instruction="Allow reasonable aggregation across roles and seasons; minor rounding is acceptable (e.g., 19.5 ~ 20).",
        extra_prerequisites=[exp_ref_node],
    )


async def verify_superintendent_position(
    evaluator: Evaluator,
    parent_node,
    pos: SuperintendentPosition,
    idx: int,
) -> None:
    # Position node (non-critical aggregator for partial credit across positions)
    position_node = evaluator.add_parallel(
        id=f"position_{idx+1}_texas_superintendent",
        desc="Texas school district superintendent position meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    # Basic presence checks
    _, refs_prereq = _add_basic_info_presence_nodes_for_super(evaluator, position_node, idx, pos)

    # Position Type verification (superintendent in Texas)
    pos_type_node = evaluator.add_leaf(
        id=f"super_{idx}_position_type",
        desc="The position is a superintendent or acting/interim superintendent role in a Texas school district",
        parent=position_node,
        critical=True
    )
    claim_pos_type = (
        f"At least one cited page announces that {pos.name or 'the individual'} was appointed as "
        f"{pos.title or 'a superintendent (or acting/interim superintendent)'} for {pos.district or 'a Texas school district'}, "
        f"and confirms the district is in Texas (TX)."
    )
    await evaluator.verify(
        claim=claim_pos_type,
        node=pos_type_node,
        sources=pos.source_urls,
        additional_instruction="Confirm superintendent (including interim/acting) status and that the district operates in Texas. Accept any clear mention of TX or Texas on the page.",
        extra_prerequisites=[refs_prereq],
    )

    # Appointment timing verification (Dec 1, 2025 to Mar 31, 2026)
    timing_node = evaluator.add_leaf(
        id=f"super_{idx}_appointment_timing",
        desc="The appointment was announced between December 1, 2025 and March 31, 2026",
        parent=position_node,
        critical=True
    )
    claim_timing = (
        "The cited page(s) show the public announcement date of the superintendent appointment "
        "falling between December 1, 2025 and March 31, 2026 (inclusive)."
    )
    await evaluator.verify(
        claim=claim_timing,
        node=timing_node,
        sources=pos.source_urls,
        additional_instruction="Use press release/article dates or explicitly stated announcement dates. Treat boundary dates inclusively.",
        extra_prerequisites=[refs_prereq],
    )

    # Texas Qualifications (critical parallel)
    qual_node = evaluator.add_parallel(
        id=f"super_{idx}_texas_qualifications",
        desc="The individual meets Texas superintendent certification requirements",
        parent=position_node,
        critical=True
    )
    # Qualifications reference presence (critical)
    qual_ref_node = evaluator.add_custom_node(
        result=bool(pos.source_urls and len(pos.source_urls) > 0),
        id=f"super_{idx}_qualifications_reference",
        desc="Educational qualifications are documented with a verifiable source URL",
        parent=qual_node,
        critical=True
    )
    # Master's degree or higher
    masters_node = evaluator.add_leaf(
        id=f"super_{idx}_masters_degree",
        desc="The individual holds a master's degree or higher",
        parent=qual_node,
        critical=True
    )
    claim_masters = (
        "The cited page(s) indicate the individual holds a master's degree or higher (e.g., M.A./M.S./M.Ed., Ed.S., Ed.D., Ph.D.)."
    )
    await evaluator.verify(
        claim=claim_masters,
        node=masters_node,
        sources=pos.source_urls,
        additional_instruction="Accept explicit degree abbreviations and equivalent advanced degrees.",
        extra_prerequisites=[qual_ref_node],
    )
    # Principal experience >= 2 years
    principal_node = evaluator.add_leaf(
        id=f"super_{idx}_principal_experience",
        desc="The individual has served as a school principal for at least 2 years",
        parent=qual_node,
        critical=True
    )
    claim_principal = (
        "The cited page(s) indicate the individual has at least two years of experience serving as a school principal."
    )
    await evaluator.verify(
        claim=claim_principal,
        node=principal_node,
        sources=pos.source_urls,
        additional_instruction="Look for explicit durations or phrases like 'more than two years' as principal.",
        extra_prerequisites=[qual_ref_node],
    )

    # Professional Experience (critical parallel)
    prof_node = evaluator.add_parallel(
        id=f"super_{idx}_professional_experience",
        desc="The individual has extensive experience in public education",
        parent=position_node,
        critical=True
    )
    # Experience reference presence (critical)
    exp_ref_node = evaluator.add_custom_node(
        result=bool(pos.source_urls and len(pos.source_urls) > 0),
        id=f"super_{idx}_experience_reference",
        desc="Professional experience is documented with a verifiable source URL",
        parent=prof_node,
        critical=True
    )
    # Years in public education >= 30
    years_ed_node = evaluator.add_leaf(
        id=f"super_{idx}_years_in_education",
        desc="The individual has at least 30 years of experience in public education",
        parent=prof_node,
        critical=True
    )
    claim_years_edu = (
        "The cited page(s) indicate the individual has at least 30 years of experience in public education."
    )
    await evaluator.verify(
        claim=claim_years_edu,
        node=years_ed_node,
        sources=pos.source_urls,
        additional_instruction="Accept approximate wordings like 'over 30 years' or 'three decades'.",
        extra_prerequisites=[exp_ref_node],
    )
    # District-level leadership prior to this appointment
    district_lead_node = evaluator.add_leaf(
        id=f"super_{idx}_district_leadership",
        desc="The individual has held district-level leadership positions (such as assistant superintendent, deputy superintendent, or executive director) before this appointment",
        parent=prof_node,
        critical=True
    )
    claim_district_lead = (
        "The cited page(s) indicate the individual previously held district-level leadership roles such as "
        "assistant/associate/deputy superintendent or executive director prior to the current appointment."
    )
    await evaluator.verify(
        claim=claim_district_lead,
        node=district_lead_node,
        sources=pos.source_urls,
        additional_instruction="Recognize synonymous titles (e.g., associate superintendent). The experience must predate the current appointment.",
        extra_prerequisites=[exp_ref_node],
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
    Evaluate an answer for the leadership positions task (two collegiate head coaches and two Texas superintendents).
    """
    # Initialize evaluator (root is non-critical by framework design)
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction"
    )

    # Prepare exactly two coaches and two superintendents
    coaches: List[CoachPosition] = list(extracted.coaches[:2])
    while len(coaches) < 2:
        coaches.append(CoachPosition())

    supers: List[SuperintendentPosition] = list(extracted.superintendents[:2])
    while len(supers) < 2:
        supers.append(SuperintendentPosition())

    # Create a top-level parallel node to reflect the four positions block
    task_node = evaluator.add_parallel(
        id="positions_block",
        desc="Four leadership positions validated against specified criteria",
        parent=root,
        critical=False
    )

    # Build verification subtrees for two collegiate coaches
    coach_group = evaluator.add_parallel(
        id="coaches_group",
        desc="Two collegiate head coaching positions",
        parent=task_node,
        critical=False
    )
    await verify_coach_position(evaluator, coach_group, coaches[0], 0)
    await verify_coach_position(evaluator, coach_group, coaches[1], 1)

    # Build verification subtrees for two Texas superintendents
    super_group = evaluator.add_parallel(
        id="superintendents_group",
        desc="Two Texas school district superintendent positions",
        parent=task_node,
        critical=False
    )
    await verify_superintendent_position(evaluator, super_group, supers[0], 0)
    await verify_superintendent_position(evaluator, super_group, supers[1], 1)

    # Return standard summary
    return evaluator.get_summary()