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
TASK_ID = "leadership_appt_2024_2026"
TASK_DESCRIPTION = (
    "Identify three individuals who were appointed to serve as president of a public research university or as "
    "superintendent of a K-12 school district serving at least 100,000 students, where their appointment became effective "
    "between July 1, 2024 and January 31, 2026 (inclusive). Each individual must satisfy ALL of the following criteria:\n\n"
    "1. The individual must have held at least one prior position at the same institution before being appointed to the top leadership role (demonstrating internal promotion)\n"
    "2. The individual must hold a terminal degree appropriate for higher education leadership (Ph.D., Ed.D., J.D., M.D., or equivalent doctorate)\n"
    "3. The individual must have served in at least one senior leadership role (such as provost, dean, vice president, deputy superintendent, or associate superintendent) at a different institution prior to joining their current institution\n"
    "4. The individual must have at least 20 years of professional experience in education administration, higher education leadership, or related fields\n"
    "5. The individual either: (a) earned at least one degree from an institution located in the same state as their current position, OR (b) held a professional position at an institution in the same state prior to joining their current institution\n\n"
    "For each individual, provide their full name, current position title, institution name, appointment effective date, and supporting reference URLs that verify each of the required criteria."
)

WINDOW_START_ISO = "2024-07-01"
WINDOW_END_ISO = "2026-01-31"
WINDOW_PRETTY = "July 1, 2024 through January 31, 2026 (inclusive)"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class IndividualSources(BaseModel):
    identity: List[str] = Field(default_factory=list)  # general biography/identity/role references
    position_requirement: List[str] = Field(default_factory=list)  # president/public research OR superintendent >=100k
    timeframe: List[str] = Field(default_factory=list)  # effective date references
    internal_promotion: List[str] = Field(default_factory=list)  # prior role at same institution
    terminal_degree: List[str] = Field(default_factory=list)  # doctorate degree references
    external_leadership: List[str] = Field(default_factory=list)  # senior role at different institution (pre-joining)
    professional_experience: List[str] = Field(default_factory=list)  # >=20 years experience references
    geographic_connection: List[str] = Field(default_factory=list)  # same-state degree or prior role references


class Individual(BaseModel):
    full_name: Optional[str] = None
    current_position_title: Optional[str] = None
    institution_name: Optional[str] = None
    appointment_effective_date: Optional[str] = None  # Keep as string for flexibility
    current_state: Optional[str] = None  # If answer provides it
    sources: IndividualSources = Field(default_factory=IndividualSources)


class IndividualsExtraction(BaseModel):
    individuals: List[Individual] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_individuals() -> str:
    return (
        "Extract up to three individuals presented in the answer who are claimed to meet the task criteria. "
        "For each individual, return the following fields under an 'individuals' array:\n"
        "1. full_name: The individual's full name exactly as provided.\n"
        "2. current_position_title: The exact title of the current leadership role (e.g., 'President', 'Superintendent').\n"
        "3. institution_name: The name of the university or school district.\n"
        "4. appointment_effective_date: The effective date string claimed in the answer (e.g., 'January 3, 2025' or '2025-01-03').\n"
        "5. current_state: The U.S. state of the current institution (only if explicitly stated in the answer; otherwise null).\n"
        "6. sources: A nested object containing arrays of URLs used in the answer to support each criterion:\n"
        "   - identity: URLs that establish the individual's identity, current role and institution.\n"
        "   - position_requirement: URLs showing that the role satisfies the position requirement "
        "     (President of a public research university OR Superintendent of a K-12 district with ≥100,000 students).\n"
        "   - timeframe: URLs confirming the appointment effective date.\n"
        "   - internal_promotion: URLs confirming a prior position at the same institution (before current role).\n"
        "   - terminal_degree: URLs confirming the individual holds a doctorate (Ph.D., Ed.D., J.D., M.D., or equivalent).\n"
        "   - external_leadership: URLs confirming senior leadership role at a different institution prior to joining the current institution.\n"
        "   - professional_experience: URLs confirming at least 20 years of relevant professional experience.\n"
        "   - geographic_connection: URLs confirming same-state degree or prior role as defined.\n\n"
        "GENERAL URL RULES:\n"
        "- Extract only actual URLs explicitly present in the answer (plain or Markdown).\n"
        "- Do not invent or infer URLs.\n"
        "- If a required source category is missing for an individual, return an empty array for that category.\n\n"
        "LIMITING:\n"
        "- If the answer lists more than three individuals, include only the first three.\n"
        "- If fewer than three are listed, include whatever is provided.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _all_required_sources_present(ind: Individual) -> bool:
    s = ind.sources
    required_lists = [
        s.identity,
        s.position_requirement,
        s.timeframe,
        s.internal_promotion,
        s.terminal_degree,
        s.external_leadership,
        s.professional_experience,
        s.geographic_connection,
    ]
    return all(isinstance(lst, list) and len(lst) > 0 for lst in required_lists)


def _non_empty_str(val: Optional[str]) -> bool:
    return bool(val and isinstance(val, str) and val.strip())


# --------------------------------------------------------------------------- #
# Verification logic per individual                                           #
# --------------------------------------------------------------------------- #
async def verify_individual(
    evaluator: Evaluator,
    parent_node,
    ind: Individual,
    idx: int,
) -> None:
    # Create the individual's main sequential node
    indiv_node = evaluator.add_sequential(
        id=f"individual_{idx + 1}",
        desc=(
            "First qualified individual meeting all appointment and professional background criteria"
            if idx == 0 else (
                "Second qualified individual meeting all appointment and professional background criteria"
                if idx == 1 else
                "Third qualified individual meeting all appointment and professional background criteria"
            )
        ),
        parent=parent_node,
        critical=False,
    )

    # Identification group (critical, parallel)
    ident_node = evaluator.add_parallel(
        id=f"individual_{idx + 1}_identification",
        desc=(
            "Identification of the first individual with name, current position, and institution"
            if idx == 0 else (
                "Identification of the second individual with name, current position, and institution"
                if idx == 1 else
                "Identification of the third individual with name, current position, and institution"
            )
        ),
        parent=indiv_node,
        critical=True,
    )

    # Critical existence checks for the four required identity fields
    evaluator.add_custom_node(
        result=_non_empty_str(ind.full_name),
        id=f"individual_{idx + 1}_full_name",
        desc="Provide the individual's full name",
        parent=ident_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty_str(ind.current_position_title),
        id=f"individual_{idx + 1}_current_position",
        desc="Provide the exact title of the current leadership position",
        parent=ident_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty_str(ind.institution_name),
        id=f"individual_{idx + 1}_institution_name",
        desc="Provide the name of the institution or school district",
        parent=ident_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty_str(ind.appointment_effective_date),
        id=f"individual_{idx + 1}_appointment_effective_date",
        desc="Provide the date when the appointment became effective",
        parent=ident_node,
        critical=True,
    )

    # Criteria verification (non-critical parent, parallel). Each leaf inside is critical.
    criteria_node = evaluator.add_parallel(
        id=f"individual_{idx + 1}_criteria_verification",
        desc=(
            "Verification that Individual 1 meets all required criteria"
            if idx == 0 else (
                "Verification that Individual 2 meets all required criteria"
                if idx == 1 else
                "Verification that Individual 3 meets all required criteria"
            )
        ),
        parent=indiv_node,
        critical=False,
    )

    # 1) Position requirement
    pr_node = evaluator.add_leaf(
        id=f"individual_{idx + 1}_position_requirement",
        desc=(
            "The position is either President of a public research university OR Superintendent of a K-12 school district serving at least 100,000 students"
        ),
        parent=criteria_node,
        critical=True,
    )
    pr_sources = ind.sources.position_requirement
    pr_claim = (
        f"{ind.full_name} serves as {ind.current_position_title} at {ind.institution_name}. "
        f"This satisfies the position requirement: either President of a public research university "
        f"OR Superintendent of a K-12 school district that serves at least 100,000 students."
    )
    if pr_sources and len(pr_sources) > 0:
        await evaluator.verify(
            claim=pr_claim,
            node=pr_node,
            sources=pr_sources,
            additional_instruction=(
                "Use the provided URLs to confirm BOTH the role and the institution category. "
                "For a university case: confirm the person is the top leader (e.g., President; treat 'Chancellor' as equivalent when it is the top executive) "
                "AND the university is public AND research-oriented (e.g., clearly described as a public research university or Carnegie R1/R2). "
                "For a district case: confirm they are Superintendent (or equivalent top executive) AND the district enrollment is ≥100,000 (approximately acceptable). "
                "Minor title variants like 'Interim President' or district 'CEO' used synonymously with Superintendent are acceptable when clearly top executive."
            ),
        )
    else:
        pr_node.score = 0.0
        pr_node.status = "failed"

    # 2) Appointment timeframe
    tf_node = evaluator.add_leaf(
        id=f"individual_{idx + 1}_appointment_timeframe",
        desc="The appointment became effective between July 1, 2024 and January 31, 2026 (inclusive)",
        parent=criteria_node,
        critical=True,
    )
    tf_sources = ind.sources.timeframe
    tf_claim = (
        f"The appointment of {ind.full_name} to {ind.current_position_title} at {ind.institution_name} "
        f"became effective on {ind.appointment_effective_date}, which falls within {WINDOW_PRETTY}."
    )
    if tf_sources and len(tf_sources) > 0:
        await evaluator.verify(
            claim=tf_claim,
            node=tf_node,
            sources=tf_sources,
            additional_instruction=(
                f"Confirm that the effective date lies inclusive between {WINDOW_START_ISO} and {WINDOW_END_ISO}. "
                "Accept press releases, official announcements, board agenda/minutes, or official institution pages. "
                "If only month/year are given, judge reasonably whether the date falls within the window. "
                "Treat synonymous phrases ('effective', 'starts', 'assumes role', 'begins') equivalently."
            ),
        )
    else:
        tf_node.score = 0.0
        tf_node.status = "failed"

    # 3) Internal promotion
    ip_node = evaluator.add_leaf(
        id=f"individual_{idx + 1}_internal_promotion",
        desc="The individual held at least one prior position at the same institution before being appointed to the current leadership role",
        parent=criteria_node,
        critical=True,
    )
    ip_sources = ind.sources.internal_promotion
    ip_claim = (
        f"Before being appointed as {ind.current_position_title} at {ind.institution_name}, "
        f"{ind.full_name} held at least one prior position at {ind.institution_name}."
    )
    if ip_sources and len(ip_sources) > 0:
        await evaluator.verify(
            claim=ip_claim,
            node=ip_node,
            sources=ip_sources,
            additional_instruction=(
                "Confirm that the person had a prior paid professional role at the SAME institution "
                "(e.g., provost, dean, vice president, associate superintendent, faculty leadership, etc.) "
                "before being appointed to the top leadership role."
            ),
        )
    else:
        ip_node.score = 0.0
        ip_node.status = "failed"

    # 4) Terminal degree
    td_node = evaluator.add_leaf(
        id=f"individual_{idx + 1}_terminal_degree",
        desc="The individual holds a terminal degree appropriate for higher education leadership (Ph.D., Ed.D., J.D., M.D., or equivalent doctorate)",
        parent=criteria_node,
        critical=True,
    )
    td_sources = ind.sources.terminal_degree
    td_claim = (
        f"{ind.full_name} holds a terminal degree appropriate for higher education leadership, "
        f"such as a Ph.D., Ed.D., J.D., M.D., or an equivalent doctorate."
    )
    if td_sources and len(td_sources) > 0:
        await evaluator.verify(
            claim=td_claim,
            node=td_node,
            sources=td_sources,
            additional_instruction=(
                "Verify that the degree is a recognized terminal doctorate (Ph.D., Ed.D., J.D., M.D., or equivalent). "
                "Titles like 'Doctor of Philosophy', 'Doctor of Education', 'Juris Doctor', 'Doctor of Medicine' count."
            ),
        )
    else:
        td_node.score = 0.0
        td_node.status = "failed"

    # 5) External senior leadership experience
    el_node = evaluator.add_leaf(
        id=f"individual_{idx + 1}_external_leadership_experience",
        desc="The individual served in at least one senior leadership role (provost, dean, vice president, deputy superintendent, or associate superintendent) at a different institution prior to joining their current institution",
        parent=criteria_node,
        critical=True,
    )
    el_sources = ind.sources.external_leadership
    el_claim = (
        f"Prior to joining {ind.institution_name}, {ind.full_name} served in at least one senior leadership role "
        "at a different institution (e.g., provost, dean, vice president, deputy superintendent, associate superintendent)."
    )
    if el_sources and len(el_sources) > 0:
        await evaluator.verify(
            claim=el_claim,
            node=el_node,
            sources=el_sources,
            additional_instruction=(
                "Confirm that the senior role was at an institution DIFFERENT from the current one, and occurred BEFORE joining the current institution. "
                "Look for explicit titles indicating senior leadership responsibility."
            ),
        )
    else:
        el_node.score = 0.0
        el_node.status = "failed"

    # 6) Professional experience (≥20 years)
    pe_node = evaluator.add_leaf(
        id=f"individual_{idx + 1}_professional_experience",
        desc="The individual has at least 20 years of professional experience in education administration, higher education leadership, or related fields",
        parent=criteria_node,
        critical=True,
    )
    pe_sources = ind.sources.professional_experience
    pe_claim = (
        f"{ind.full_name} has at least 20 years of professional experience in education administration, "
        "higher education leadership, or closely related fields."
    )
    if pe_sources and len(pe_sources) > 0:
        await evaluator.verify(
            claim=pe_claim,
            node=pe_node,
            sources=pe_sources,
            additional_instruction=(
                "If an explicit statement such as 'over 20 years of experience' exists, that suffices. "
                "Otherwise, approximate by subtracting the earliest documented role start year from the appointment year, "
                "or by summing clearly documented tenure lengths; accept reasonable evidence (e.g., 'two decades')."
            ),
        )
    else:
        pe_node.score = 0.0
        pe_node.status = "failed"

    # 7) Geographic connection (same-state degree OR prior role)
    gc_node = evaluator.add_leaf(
        id=f"individual_{idx + 1}_geographic_connection",
        desc="The individual either: (a) earned at least one degree from an institution located in the same state as their current position, OR (b) held a professional position at an institution in the same state prior to joining their current institution",
        parent=criteria_node,
        critical=True,
    )
    gc_sources = ind.sources.geographic_connection
    current_state_text = ind.current_state if _non_empty_str(ind.current_state) else "the state of the current institution"
    gc_claim = (
        f"{ind.full_name} either earned at least one degree from an institution located in the same state as {ind.institution_name} "
        f"(i.e., {current_state_text}), OR previously held a professional position in that same state prior to joining {ind.institution_name}."
    )
    if gc_sources and len(gc_sources) > 0:
        await evaluator.verify(
            claim=gc_claim,
            node=gc_node,
            sources=gc_sources,
            additional_instruction=(
                "Determine the U.S. state for the current institution from the references, then confirm either a degree from a same-state institution "
                "OR a prior professional role within that same state. Accept common state abbreviations and minor naming variations."
            ),
        )
    else:
        gc_node.score = 0.0
        gc_node.status = "failed"

    # Supporting references provided (critical)
    sup_node = evaluator.add_custom_node(
        result=_all_required_sources_present(ind),
        id=f"individual_{idx + 1}_supporting_references",
        desc=(
            "Reference URLs are provided that verify the individual's identity, position, appointment details, and satisfaction of all required criteria"
        ),
        parent=indiv_node,
        critical=True,
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
) -> Dict[str, Any]:
    # Initialize evaluator with root parallel aggregation
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify three individuals who were appointed to senior leadership positions meeting all specified criteria",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Add an info block for the timeframe window
    evaluator.add_custom_info(
        info={
            "time_window_iso": {"start": WINDOW_START_ISO, "end": WINDOW_END_ISO},
            "time_window_human": WINDOW_PRETTY,
        },
        info_type="evaluation_parameters",
    )

    # Extract individuals and their sources from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_individuals(),
        template_class=IndividualsExtraction,
        extraction_name="individuals_extraction",
    )

    # Normalize to exactly 3 individuals (pad with empties if fewer; truncate if more)
    individuals: List[Individual] = list(extracted.individuals[:3])
    while len(individuals) < 3:
        individuals.append(Individual())

    # Build the verification tree for each individual
    for i in range(3):
        await verify_individual(evaluator, root, individuals[i], i)

    # Return evaluation summary
    return evaluator.get_summary()