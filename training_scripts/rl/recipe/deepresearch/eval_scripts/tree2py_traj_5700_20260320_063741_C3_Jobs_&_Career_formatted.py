import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "higher_ed_admin_identification"
TASK_DESCRIPTION = (
    "Identify the individual who meets all of the following career criteria in the field of higher education administration: "
    "(1) Served as an associate dean for academic affairs at a law school before becoming a dean, "
    "(2) Served as dean of a law school from 1997 to 2005 (8 years) - First deanship, "
    "(3) Subsequently served as dean of a different law school from 2005 to 2013 (8 years) - Second deanship, "
    "(4) After the law school deanships, became a university chancellor/president, "
    "(5) Has been appointed as the 16th president of the University of Michigan, with the appointment effective July 1, 2026. "
    "Provide the individual's name and include the names of the two law schools where they served as dean, as well as the university where they served as chancellor/president before the University of Michigan appointment."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RoleInfo(BaseModel):
    institution_name: Optional[str] = None
    role_title: Optional[str] = None
    start_year: Optional[str] = None
    end_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class UMichAppointment(BaseModel):
    ordinal: Optional[str] = None  # e.g., "16th", "sixteenth"
    effective_date: Optional[str] = None  # e.g., "July 1, 2026"
    sources: List[str] = Field(default_factory=list)


class CandidateExtraction(BaseModel):
    name: Optional[str] = None
    associate_dean: Optional[RoleInfo] = None  # Associate Dean for Academic Affairs (before first deanship)
    first_deanship: Optional[RoleInfo] = None  # 1997–2005
    second_deanship: Optional[RoleInfo] = None  # 2005–2013
    chancellor_or_president: Optional[RoleInfo] = None  # After second deanship
    umich_appointment: Optional[UMichAppointment] = None  # 16th, effective July 1, 2026


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_candidate_details() -> str:
    return """
Extract the following structured information about the single individual proposed in the answer. Return exactly the requested JSON fields. If the answer does not provide a requested field, set it to null (or [] for lists). Do not invent information.

Fields to extract:
- name: The individual's full name.

- associate_dean: Information about the "Associate Dean for Academic Affairs" role BEFORE the first deanship.
  • institution_name: The law school (or university/college unit) where the associate dean for academic affairs role was held.
  • role_title: The exact title as written (e.g., "Associate Dean for Academic Affairs"). Include qualifiers if present (e.g., "and Student Services").
  • start_year: The start year as a 4-digit string if mentioned; else null.
  • end_year: The end year as a 4-digit string if mentioned; else null.
  • sources: A list of URL(s) explicitly cited in the answer that support this associate dean role. If none are provided, return an empty list.

- first_deanship: Information about the first law school deanship (expected 1997–2005).
  • institution_name: The law school where the first deanship occurred.
  • role_title: The title as written (e.g., "Dean", "Dean of the School of Law").
  • start_year: The start year as a 4-digit string if mentioned; else null.
  • end_year: The end year as a 4-digit string if mentioned; else null.
  • sources: URL(s) cited that support this first deanship. If none, [].

- second_deanship: Information about the second, different law school deanship (expected 2005–2013).
  • institution_name
  • role_title
  • start_year
  • end_year
  • sources

- chancellor_or_president: The university-level chancellor or president role held AFTER the two deanships.
  • institution_name: University name.
  • role_title: The exact title (e.g., "Chancellor", "President").
  • start_year: 4-digit string if mentioned; else null.
  • end_year: 4-digit string if mentioned; else null.
  • sources: URL(s) supporting this role.

- umich_appointment: The University of Michigan presidency appointment.
  • ordinal: The stated ordinal (e.g., "16th", "sixteenth") for the UMich presidency.
  • effective_date: The stated effective date text (e.g., "July 1, 2026").
  • sources: URL(s) supporting the UMich appointment claim.

Rules:
- Extract only what is explicitly present in the answer text.
- For URL fields, include only valid URLs (plain or in Markdown). If a URL is missing but a site is mentioned without a link, do not invent a URL.
- Keep years as strings (e.g., "1997"). If a range like "1997–2005" appears, assign start_year="1997", end_year="2005" when possible.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _parse_year(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.findall(r"(19|20)\d{2}", text)
    if not m:
        return None
    # Return the first 4-digit year found
    return int(m[0] + "") if isinstance(m[0], str) and len(m[0]) == 4 else int(re.findall(r"(19|20)\d{2}", text)[0])


def _first_year(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    yrs = re.findall(r"(?:19|20)\d{2}", text)
    return int(yrs[0]) if yrs else None


def _last_year(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    yrs = re.findall(r"(?:19|20)\d{2}", text)
    return int(yrs[-1]) if yrs else None


def _safe_sources(s: Optional[List[str]]) -> List[str]:
    return [u for u in (s or []) if isinstance(u, str) and len(u.strip()) > 0]


def _nonempty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, parent_node, data: CandidateExtraction) -> None:
    # Add primary sequential critical node to mirror the rubric root
    main_node = evaluator.add_sequential(
        id="Individual_Identification",
        desc="Correctly identifies an individual who meets all specified career criteria",
        parent=parent_node,
        critical=True,
    )

    # --------------------- 1) Associate Dean Position --------------------- #
    assoc_node = evaluator.add_parallel(
        id="Associate_Dean_Position",
        desc="Confirms the individual held an associate dean for academic affairs position at a law school before becoming a dean",
        parent=main_node,
        critical=True,
    )

    # 1.a) Associate_Dean_Verification (held before first deanship) - Logic check
    # Use extracted years where available; default first deanship start to 1997 as per task constraints.
    ad_start = _first_year(data.associate_dean.start_year) if data.associate_dean else None
    ad_end = _last_year(data.associate_dean.end_year) if data.associate_dean else None
    first_start_extracted = _first_year(data.first_deanship.start_year) if data.first_deanship else None
    first_start_expected = 1997
    first_dean_start_year = first_start_extracted or first_start_expected

    held_before = False
    if ad_end is not None:
        held_before = ad_end <= (first_dean_start_year - 1)
    elif ad_start is not None:
        held_before = ad_start < first_dean_start_year

    evaluator.add_custom_node(
        result=held_before,
        id="Associate_Dean_Verification",
        desc="Verifies the associate dean position was held before the first deanship",
        parent=assoc_node,
        critical=True,
    )

    # 1.b) Associate_Dean_Reference_URL (must provide URL references)
    assoc_sources = _safe_sources(data.associate_dean.sources if data.associate_dean else [])
    evaluator.add_custom_node(
        result=len(assoc_sources) > 0,
        id="Associate_Dean_Reference_URL",
        desc="Provides URL reference for the associate dean position",
        parent=assoc_node,
        critical=True,
    )

    # --------------------- 2) First Law School Deanship ------------------- #
    first_node = evaluator.add_parallel(
        id="First_Law_School_Deanship",
        desc="Confirms the individual served as dean of a law school from 1997 to 2005",
        parent=main_node,
        critical=True,
    )

    first_inst = data.first_deanship.institution_name if data.first_deanship else None
    first_sources = _safe_sources(data.first_deanship.sources if data.first_deanship else [])

    # 2.a) First_Deanship_Institution_Name - Verify school and role via sources
    leaf_first_inst = evaluator.add_leaf(
        id="First_Deanship_Institution_Name",
        desc="Provides the name of the first law school where the individual served as dean",
        parent=first_node,
        critical=True,
    )
    claim_first_inst = (
        f"The individual served as dean of {first_inst} (a law school)."
        if _nonempty(first_inst) else
        "The individual served as dean of a specific law school (first deanship)."
    )
    await evaluator.verify(
        claim=claim_first_inst,
        node=leaf_first_inst,
        sources=first_sources,
        additional_instruction="Verify the page(s) explicitly state the individual served as dean at this named law school (allow title variations like 'Dean of the School of Law').",
    )

    # 2.b) First_Deanship_Duration - Verify exact years via sources
    leaf_first_duration = evaluator.add_leaf(
        id="First_Deanship_Duration",
        desc="Verifies the first deanship lasted 8 years from 1997 to 2005",
        parent=first_node,
        critical=True,
    )
    claim_first_duration = (
        f"The individual served as dean of {first_inst} from 1997 to 2005 (8 years)."
        if _nonempty(first_inst) else
        "The individual served as dean from 1997 to 2005 (8 years)."
    )
    await evaluator.verify(
        claim=claim_first_duration,
        node=leaf_first_duration,
        sources=first_sources,
        additional_instruction="Confirm both endpoints (1997 and 2005) are correct. Accept phrasing like '1997–2005', 'from 1997 through 2005', or similar.",
    )

    # 2.c) First_Deanship_Reference_URL - Presence of references
    evaluator.add_custom_node(
        result=len(first_sources) > 0,
        id="First_Deanship_Reference_URL",
        desc="Provides URL reference for the first deanship details",
        parent=first_node,
        critical=True,
    )

    # --------------------- 3) Second Law School Deanship ------------------ #
    second_node = evaluator.add_parallel(
        id="Second_Law_School_Deanship",
        desc="Confirms the individual served as dean of a different law school from 2005 to 2013",
        parent=main_node,
        critical=True,
    )

    second_inst = data.second_deanship.institution_name if data.second_deanship else None
    second_sources = _safe_sources(data.second_deanship.sources if data.second_deanship else [])

    # 3.a) Second_Deanship_Institution_Name - Verify school and role via sources
    leaf_second_inst = evaluator.add_leaf(
        id="Second_Deanship_Institution_Name",
        desc="Provides the name of the second law school where the individual served as dean",
        parent=second_node,
        critical=True,
    )
    claim_second_inst = (
        f"The individual served as dean of {second_inst} (a law school)."
        if _nonempty(second_inst) else
        "The individual served as dean of a specific (second) law school."
    )
    await evaluator.verify(
        claim=claim_second_inst,
        node=leaf_second_inst,
        sources=second_sources,
        additional_instruction="Verify the page(s) explicitly state the individual served as dean at this named law school (second deanship). Title variations are acceptable.",
    )

    # 3.b) Second_Deanship_Duration - Verify exact years via sources
    leaf_second_duration = evaluator.add_leaf(
        id="Second_Deanship_Duration",
        desc="Verifies the second deanship lasted 8 years from 2005 to 2013",
        parent=second_node,
        critical=True,
    )
    claim_second_duration = (
        f"The individual served as dean of {second_inst} from 2005 to 2013 (8 years)."
        if _nonempty(second_inst) else
        "The individual served as dean from 2005 to 2013 (8 years)."
    )
    await evaluator.verify(
        claim=claim_second_duration,
        node=leaf_second_duration,
        sources=second_sources,
        additional_instruction="Confirm both endpoints (2005 and 2013) are correct. Accept phrasing like '2005–2013' or equivalent.",
    )

    # 3.c) Second_Deanship_Reference_URL - Presence of references
    evaluator.add_custom_node(
        result=len(second_sources) > 0,
        id="Second_Deanship_Reference_URL",
        desc="Provides URL reference for the second deanship details",
        parent=second_node,
        critical=True,
    )

    # --------------------- 4) Chancellor/President after Deanships -------- #
    ch_node = evaluator.add_parallel(
        id="University_Chancellor_President_Position",
        desc="Confirms the individual became a university chancellor or president after completing the two law school deanships",
        parent=main_node,
        critical=True,
    )

    ch_inst = data.chancellor_or_president.institution_name if data.chancellor_or_president else None
    ch_role = data.chancellor_or_president.role_title if data.chancellor_or_president else None
    ch_sources = _safe_sources(data.chancellor_or_president.sources if data.chancellor_or_president else [])

    # 4.a) Chancellor_Institution_Name - Verify role and institution via sources
    leaf_ch_inst = evaluator.add_leaf(
        id="Chancellor_Institution_Name",
        desc="Provides the name of the university where the individual served as chancellor/president",
        parent=ch_node,
        critical=True,
    )
    claim_ch_inst = (
        f"The individual served as {ch_role or 'a chancellor or president'} at {ch_inst}."
        if _nonempty(ch_inst) else
        f"The individual served as {ch_role or 'a chancellor or president'} at a named university."
    )
    await evaluator.verify(
        claim=claim_ch_inst,
        node=leaf_ch_inst,
        sources=ch_sources,
        additional_instruction="Verify that the page(s) show the person as a university-level Chancellor or President at the stated institution.",
    )

    # 4.b) Chancellor_Position_Verification - Logic check (after second deanship)
    ch_start = _first_year(data.chancellor_or_president.start_year) if data.chancellor_or_president else None
    second_end_extracted = _last_year(data.second_deanship.end_year) if data.second_deanship else None
    second_end_expected = 2013
    second_end_year = second_end_extracted or second_end_expected
    ch_after = False
    if ch_start is not None:
        ch_after = ch_start >= second_end_year  # Allow same-year transition; typically 2013 or later

    evaluator.add_custom_node(
        result=ch_after,
        id="Chancellor_Position_Verification",
        desc="Verifies the individual held a chancellor or president position after the second deanship",
        parent=ch_node,
        critical=True,
    )

    # 4.c) Chancellor_Reference_URL - Presence of references
    evaluator.add_custom_node(
        result=len(ch_sources) > 0,
        id="Chancellor_Reference_URL",
        desc="Provides URL reference for the chancellor/president position",
        parent=ch_node,
        critical=True,
    )

    # --------------------- 5) University of Michigan Presidency ------------ #
    um_node = evaluator.add_parallel(
        id="University_of_Michigan_Presidency",
        desc="Verifies the individual has been appointed as the 16th president of the University of Michigan, effective July 1, 2026",
        parent=main_node,
        critical=True,
    )

    um_ordinal = (data.umich_appointment.ordinal or "").strip().lower() if data.umich_appointment else ""
    um_effective = data.umich_appointment.effective_date if data.umich_appointment else None
    um_sources = _safe_sources(data.umich_appointment.sources if data.umich_appointment else [])

    # 5.a) Claim verification via sources
    leaf_um_claim = evaluator.add_leaf(
        id="University_of_Michigan_Presidency_Claim",
        desc="UMich presidency: appointed as the 16th president, effective July 1, 2026",
        parent=um_node,
        critical=True,
    )
    claim_um = (
        "The individual has been appointed as the 16th (sixteenth) president of the University of Michigan, with the appointment effective July 1, 2026."
    )
    await evaluator.verify(
        claim=claim_um,
        node=leaf_um_claim,
        sources=um_sources,
        additional_instruction="Confirm the announcement explicitly states the ordinal '16th' (or 'sixteenth') and the effective date July 1, 2026.",
    )

    # 5.b) UMich_Presidential_Appointment_Reference_URL - Presence of references
    evaluator.add_custom_node(
        result=len(um_sources) > 0,
        id="UMich_Presidential_Appointment_Reference_URL",
        desc="Provides URL reference for the University of Michigan presidential appointment",
        parent=um_node,
        critical=True,
    )

    # Record some helpful custom info for debugging
    evaluator.add_custom_info(
        info={
            "extracted_person_name": data.name,
            "first_deanship_school": first_inst,
            "second_deanship_school": second_inst,
            "chancellor_university": ch_inst,
            "umich_ordinal_extracted": um_ordinal,
            "umich_effective_date_extracted": um_effective,
        },
        info_type="extraction_overview",
        info_name="extracted_summary"
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
    Evaluate an answer for the higher education administration identification task.
    """
    # Initialize evaluator with a sequential strategy (the rubric is inherently ordered)
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

    # Extraction
    extracted: CandidateExtraction = await evaluator.extract(
        prompt=prompt_extract_candidate_details(),
        template_class=CandidateExtraction,
        extraction_name="candidate_extraction",
    )

    # Optional informational ground truth constraints (for transparency only)
    evaluator.add_ground_truth({
        "required_first_deanship_years": "1997–2005",
        "required_second_deanship_years": "2005–2013",
        "required_umich_ordinal": "16th (sixteenth)",
        "required_umich_effective_date": "July 1, 2026"
    }, gt_type="task_constraints")

    # Build verification tree and run checks
    await build_verification_tree(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()