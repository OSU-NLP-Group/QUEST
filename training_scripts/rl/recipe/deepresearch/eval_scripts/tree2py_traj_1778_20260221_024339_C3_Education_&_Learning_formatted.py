import asyncio
import logging
from typing import Optional, Dict, Any

from pydantic import BaseModel
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_ed_secretary_2026_career"
TASK_DESCRIPTION = (
    "Who currently serves as the U.S. Secretary of Education as of February 2026? "
    "For this individual: 1. Provide their full name and an official U.S. Department of Education URL confirming "
    "their current position. 2. Verify whether they previously served as Administrator of the U.S. Small Business "
    "Administration, and provide an official URL confirming this previous role. 3. Identify the specific start month "
    "and year, and end month and year of their service as SBA Administrator, along with an official URL confirming "
    "these dates. 4. Identify which U.S. President appointed them as SBA Administrator, and provide a URL confirming "
    "which presidential administration made this appointment. 5. Provide the date they were sworn in as U.S. Secretary "
    "of Education, identify which U.S. President appointed them to this position, and provide an official URL "
    "documenting their career progression from SBA to the Department of Education."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class SecretaryCareerExtraction(BaseModel):
    # Identity and ED confirmation
    current_secretary_name: Optional[str] = None
    current_secretary_official_url: Optional[str] = None  # Prefer an ed.gov URL

    # SBA role confirmation
    sba_role_confirm_url: Optional[str] = None  # Official SBA.gov / whitehouse.gov / congress.gov etc.

    # SBA service dates
    sba_service_start: Optional[str] = None  # e.g., "March 2021"
    sba_service_end: Optional[str] = None    # e.g., "January 2025"
    sba_dates_confirm_url: Optional[str] = None

    # SBA appointing administration
    sba_appointing_president: Optional[str] = None  # e.g., "Joe Biden", "Donald Trump"
    sba_appointing_president_url: Optional[str] = None

    # Education Secretary appointment
    education_sworn_in_date: Optional[str] = None  # e.g., "February 3, 2026" or "February 2026"
    education_appointing_president: Optional[str] = None
    career_trajectory_url: Optional[str] = None  # Official URL documenting SBA -> Education path


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_secretary_career() -> str:
    return """
Extract the following information exactly as stated in the answer text. Return null for any item that is not explicitly present.

Required fields:
1) current_secretary_name: Full name of the current U.S. Secretary of Education.
2) current_secretary_official_url: A single official U.S. Department of Education URL (ed.gov) that confirms this person is the U.S. Secretary of Education. If multiple are present, pick the best single URL.

3) sba_role_confirm_url: A single official URL (prefer .gov such as sba.gov, whitehouse.gov, congress.gov) confirming that this person served as Administrator of the U.S. Small Business Administration.

4) sba_service_start: The month and year the person started as SBA Administrator (e.g., "March 2021"). Use month name and four-digit year if possible.
5) sba_service_end: The month and year the person ended as SBA Administrator (e.g., "January 2025"). Use month name and four-digit year if possible.
6) sba_dates_confirm_url: A single official URL that confirms the start and end dates as SBA Administrator.

7) sba_appointing_president: The name of the U.S. President who appointed (or nominated) the person as SBA Administrator.
8) sba_appointing_president_url: A single official URL confirming which presidential administration appointed (or nominated) this person as SBA Administrator.

9) education_sworn_in_date: The date the person was sworn in (or assumed office) as U.S. Secretary of Education. Prefer full date (e.g., "February 3, 2026"); if only month and year are given, use that.
10) education_appointing_president: The name of the U.S. President who appointed (or nominated) the person as U.S. Secretary of Education.
11) career_trajectory_url: A single official URL documenting the career progression from SBA to the Department of Education (e.g., a biography page or official announcement mentioning both roles).

Rules:
- Only extract URLs explicitly provided in the answer. Do not invent or infer URLs.
- Prefer .gov domains when available for confirmation URLs.
- If multiple candidate URLs are listed for the same item, choose one best official source and return only that single URL.
"""


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_current_secretary_identification(
    evaluator: Evaluator,
    parent,
    ex: SecretaryCareerExtraction,
):
    # Parallel parent node
    current_node = evaluator.add_parallel(
        id="current_secretary_identification",
        desc="Identify who currently serves as U.S. Secretary of Education as of February 2026, providing their full name and official confirmation",
        parent=parent,
        critical=False
    )

    # Existence check (name + ED URL)
    has_name_and_url = bool(ex.current_secretary_name and ex.current_secretary_name.strip()) and bool(
        ex.current_secretary_official_url and ex.current_secretary_official_url.strip()
    )
    evaluator.add_custom_node(
        result=has_name_and_url,
        id="secretary_name_and_source_presence",
        desc="Presence check: Secretary name and an official ED URL are provided",
        parent=current_node,
        critical=True
    )

    # Verification leaf: ED URL supports that {name} is U.S. Secretary of Education
    sec_leaf = evaluator.add_leaf(
        id="secretary_name_and_source",
        desc="Provide the full name of the current U.S. Secretary of Education with an official U.S. Department of Education URL confirming their identity",
        parent=current_node,
        critical=True
    )

    name = ex.current_secretary_name or ""
    ed_url = ex.current_secretary_official_url or None
    claim = f"The page explicitly identifies {name} as the U.S. Secretary of Education for the U.S. Department of Education."
    await evaluator.verify(
        claim=claim,
        node=sec_leaf,
        sources=ed_url,
        additional_instruction=(
            "Treat 'U.S. Secretary of Education' and 'Secretary of Education' as equivalent. "
            "The page should clearly indicate this person holds the role of Secretary of Education. "
            "Allow minor variations in name formatting (e.g., middle initials)."
        )
    )


async def build_previous_sba_role_verification(
    evaluator: Evaluator,
    parent,
    ex: SecretaryCareerExtraction,
):
    # Parallel parent node
    sba_role_node = evaluator.add_parallel(
        id="previous_sba_role_verification",
        desc="Verify that the identified individual previously served as Administrator of the U.S. Small Business Administration",
        parent=parent,
        critical=False
    )

    # Existence check (SBA role confirm URL)
    has_role_url = bool(ex.sba_role_confirm_url and ex.sba_role_confirm_url.strip())
    evaluator.add_custom_node(
        result=has_role_url,
        id="sba_role_with_source_presence",
        desc="Presence check: An official URL confirming SBA Administrator role is provided",
        parent=sba_role_node,
        critical=True
    )

    # Verification leaf
    role_leaf = evaluator.add_leaf(
        id="sba_role_with_source",
        desc="Confirm the individual held the title of Administrator of the U.S. Small Business Administration with an official SBA or government URL confirming this role",
        parent=sba_role_node,
        critical=True
    )

    name = ex.current_secretary_name or ""
    claim = f"The page explicitly states that {name} served as Administrator of the U.S. Small Business Administration (SBA)."
    await evaluator.verify(
        claim=claim,
        node=role_leaf,
        sources=ex.sba_role_confirm_url or None,
        additional_instruction=(
            "Accept wording such as 'SBA Administrator', 'Administrator, Small Business Administration', "
            "'Administrator of the SBA', or equivalents. The page must clearly indicate that this person held the top SBA Administrator role."
        )
    )


async def build_sba_service_dates(
    evaluator: Evaluator,
    parent,
    ex: SecretaryCareerExtraction,
):
    # Parallel parent node
    dates_node = evaluator.add_parallel(
        id="sba_service_dates",
        desc="Identify the specific dates of service as SBA Administrator",
        parent=parent,
        critical=False
    )

    # Presence checks
    has_start = bool(ex.sba_service_start and ex.sba_service_start.strip())
    has_end = bool(ex.sba_service_end and ex.sba_service_end.strip())
    has_dates_url = bool(ex.sba_dates_confirm_url and ex.sba_dates_confirm_url.strip())

    evaluator.add_custom_node(
        result=has_start,
        id="start_date_presence",
        desc="Presence check: Start month and year as SBA Administrator are provided",
        parent=dates_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_end,
        id="end_date_presence",
        desc="Presence check: End month and year as SBA Administrator are provided",
        parent=dates_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_dates_url,
        id="dates_source_presence",
        desc="Presence check: An official URL confirming SBA service dates is provided",
        parent=dates_node,
        critical=True
    )

    # Start date verification
    start_leaf = evaluator.add_leaf(
        id="start_date",
        desc="Provide the month and year the individual began service as SBA Administrator",
        parent=dates_node,
        critical=True
    )
    start_claim = f"The page states that {ex.current_secretary_name or ''} began service as Administrator of the U.S. Small Business Administration in {ex.sba_service_start or ''}."
    await evaluator.verify(
        claim=start_claim,
        node=start_leaf,
        sources=ex.sba_dates_confirm_url or None,
        additional_instruction=(
            "Match on month and year even if the page provides a full exact date. "
            "Accept synonyms like 'took office', 'assumed office', or 'sworn in' for the start of service."
        )
    )

    # End date verification
    end_leaf = evaluator.add_leaf(
        id="end_date",
        desc="Provide the month and year the individual ended service as SBA Administrator",
        parent=dates_node,
        critical=True
    )
    end_claim = f"The page states that {ex.current_secretary_name or ''} ended service as Administrator of the U.S. Small Business Administration in {ex.sba_service_end or ''}."
    await evaluator.verify(
        claim=end_claim,
        node=end_leaf,
        sources=ex.sba_dates_confirm_url or None,
        additional_instruction=(
            "Match on month and year even if the page provides a full exact date. "
            "Accept wording like 'tenure ended', 'resigned effective', or 'left office' as indicating the end of service."
        )
    )

    # Dates source verification (confirms both)
    dates_src_leaf = evaluator.add_leaf(
        id="dates_source",
        desc="Provide official URL confirming these service dates",
        parent=dates_node,
        critical=True
    )
    both_claim = (
        f"This page includes both the start month and year '{ex.sba_service_start or ''}' and "
        f"the end month and year '{ex.sba_service_end or ''}' for {ex.current_secretary_name or ''}'s tenure as SBA Administrator."
    )
    await evaluator.verify(
        claim=both_claim,
        node=dates_src_leaf,
        sources=ex.sba_dates_confirm_url or None,
        additional_instruction=(
            "Confirm that the page contains evidence for both the start and end timeframe (month+year acceptable). "
            "It may present exact dates; if so, ensure they correspond to the provided months and years."
        )
    )


async def build_appointing_administration(
    evaluator: Evaluator,
    parent,
    ex: SecretaryCareerExtraction,
):
    # Parallel parent node
    appoint_node = evaluator.add_parallel(
        id="appointing_administration",
        desc="Identify which U.S. presidential administration appointed the individual as SBA Administrator",
        parent=parent,
        critical=False
    )

    # Presence check
    has_pres = bool(ex.sba_appointing_president and ex.sba_appointing_president.strip())
    has_url = bool(ex.sba_appointing_president_url and ex.sba_appointing_president_url.strip())
    evaluator.add_custom_node(
        result=(has_pres and has_url),
        id="president_and_source_presence",
        desc="Presence check: SBA appointing president name and a confirming URL are provided",
        parent=appoint_node,
        critical=True
    )

    # Verification leaf
    pres_leaf = evaluator.add_leaf(
        id="president_and_source",
        desc="Provide the name of the U.S. President who appointed the individual as SBA Administrator with a URL confirming which presidential administration made this appointment",
        parent=appoint_node,
        critical=True
    )

    claim = (
        f"The page states that {ex.current_secretary_name or ''} was appointed or nominated as SBA Administrator by "
        f"President {ex.sba_appointing_president or ''}."
    )
    await evaluator.verify(
        claim=claim,
        node=pres_leaf,
        sources=ex.sba_appointing_president_url or None,
        additional_instruction=(
            "Accept phrasing like 'nominated by President X', 'appointed by President X', or 'under the X administration'. "
            "The page should clearly attribute the appointment/nomination to the named President."
        )
    )


async def build_complete_career_trajectory(
    evaluator: Evaluator,
    parent,
    ex: SecretaryCareerExtraction,
):
    # Parallel parent node
    traj_node = evaluator.add_parallel(
        id="complete_career_trajectory",
        desc="Document the complete career path from SBA Administrator to Secretary of Education",
        parent=parent,
        critical=False
    )

    # Presence checks for this section
    has_traj_url = bool(ex.career_trajectory_url and ex.career_trajectory_url.strip())
    has_sworn_date = bool(ex.education_sworn_in_date and ex.education_sworn_in_date.strip())
    has_ed_pres = bool(ex.education_appointing_president and ex.education_appointing_president.strip())

    evaluator.add_custom_node(
        result=has_traj_url,
        id="trajectory_source_presence",
        desc="Presence check: An official URL documenting SBA → Education career progression is provided",
        parent=traj_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_sworn_date,
        id="education_sworn_date_presence",
        desc="Presence check: Sworn-in (or assumed office) date for Education Secretary is provided",
        parent=traj_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_ed_pres,
        id="education_appointing_presence",
        desc="Presence check: Name of the President who appointed the Education Secretary is provided",
        parent=traj_node,
        critical=True
    )

    # Leaf: Education Secretary sworn-in date
    sworn_leaf = evaluator.add_leaf(
        id="education_secretary_appointment_date",
        desc="Provide the date the individual was sworn in as U.S. Secretary of Education",
        parent=traj_node,
        critical=True
    )
    sworn_claim = (
        f"The page states that {ex.current_secretary_name or ''} was sworn in (or assumed office) as U.S. Secretary of Education "
        f"on {ex.education_sworn_in_date or ''}."
    )
    await evaluator.verify(
        claim=sworn_claim,
        node=sworn_leaf,
        sources=ex.career_trajectory_url or None,
        additional_instruction=(
            "Accept synonyms like 'sworn in', 'assumed office', 'took office'. "
            "If the page lists a full date, it should match the provided date; if the provided value is month/year only, allow match to a full date within that month/year."
        )
    )

    # Leaf: Education appointing president
    ed_pres_leaf = evaluator.add_leaf(
        id="appointing_president_education",
        desc="Provide the name of the U.S. President who appointed the individual as Secretary of Education",
        parent=traj_node,
        critical=True
    )
    ed_pres_claim = (
        f"The page states that President {ex.education_appointing_president or ''} appointed or nominated "
        f"{ex.current_secretary_name or ''} as U.S. Secretary of Education."
    )
    await evaluator.verify(
        claim=ed_pres_claim,
        node=ed_pres_leaf,
        sources=ex.career_trajectory_url or None,
        additional_instruction=(
            "Accept wording such as 'nominated by President X', 'appointed by President X', "
            "or 'under the X administration' clearly linking the appointment to the named President."
        )
    )

    # Leaf: Trajectory source (documents SBA → Education path)
    traj_leaf = evaluator.add_leaf(
        id="trajectory_source",
        desc="Provide official URL documenting the career path from SBA to Education Department",
        parent=traj_node,
        critical=True
    )
    traj_claim = (
        f"This page documents that {ex.current_secretary_name or ''} previously served as Administrator of the U.S. Small "
        f"Business Administration before becoming U.S. Secretary of Education."
    )
    await evaluator.verify(
        claim=traj_claim,
        node=traj_leaf,
        sources=ex.career_trajectory_url or None,
        additional_instruction=(
            "The page should mention both roles (SBA Administrator and U.S. Secretary of Education) in the biographical or announcement context, "
            "indicating the career progression."
        )
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
    Evaluate an answer for the 2026 U.S. Secretary of Education career progression task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,   # Enforce ordered dependency across sections
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_secretary_career(),
        template_class=SecretaryCareerExtraction,
        extraction_name="secretary_career_extraction"
    )

    # Build and verify sections in order
    await build_current_secretary_identification(evaluator, root, extraction)
    await build_previous_sba_role_verification(evaluator, root, extraction)
    await build_sba_service_dates(evaluator, root, extraction)
    await build_appointing_administration(evaluator, root, extraction)
    await build_complete_career_trajectory(evaluator, root, extraction)

    return evaluator.get_summary()