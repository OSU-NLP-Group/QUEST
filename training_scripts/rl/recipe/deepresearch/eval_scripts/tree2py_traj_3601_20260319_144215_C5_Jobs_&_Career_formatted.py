import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pa_superintendent_early_2026"
TASK_DESCRIPTION = """
A Pennsylvania school district appointed a new superintendent in early 2026. The appointment was publicly announced between January 1, 2026 and March 19, 2026 (inclusive). The appointed individual holds a Pennsylvania Superintendent Letter of Eligibility and has a doctoral degree from a Pennsylvania university. This person has at least 20 years of experience in education and, immediately before this appointment, held a district-level administrative position. The appointment involves succeeding a superintendent who is retiring, and the official start date is after June 2026. Identify the name of the appointed superintendent and the school district where they were appointed.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SuperintendentExtraction(BaseModel):
    """Structured info extracted from the agent's answer."""
    # Identification
    name: Optional[str] = None
    district: Optional[str] = None

    # Dates and appointment context
    announcement_date: Optional[str] = None
    start_date: Optional[str] = None
    predecessor_status: Optional[str] = None  # e.g., "retiring", "retirement"

    # Qualifications
    letter_of_eligibility: Optional[str] = None  # textual mention if any
    doctoral_degree: Optional[str] = None        # e.g., "EdD", "PhD", "doctorate"
    doctoral_university: Optional[str] = None    # e.g., "Temple University"
    years_experience: Optional[str] = None       # textual (e.g., "over 20 years")
    prior_position: Optional[str] = None         # immediately prior district-level role

    # Evidence
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_superintendent_info() -> str:
    return """
    Extract the superintendent appointment details presented in the answer.

    Required fields:
    - name: The full name of the appointed superintendent.
    - district: The name of the Pennsylvania school district where they were appointed.
    - announcement_date: The public announcement date if explicitly stated (e.g., the press release date or board approval date). If not provided, return null.
    - start_date: The official start/effective date for the superintendent role. If not provided, return null.
    - predecessor_status: If the answer specifies the outgoing superintendent is "retiring", include a short phrase like "retiring" or "retirement". Otherwise, return null.
    - letter_of_eligibility: If the answer mentions a Pennsylvania Superintendent Letter of Eligibility (or equivalent phrasing), include a short phrase capturing this. Otherwise, return null.
    - doctoral_degree: The doctoral degree type if mentioned (e.g., "EdD", "PhD", or "doctorate"). Otherwise, return null.
    - doctoral_university: The university that awarded the doctorate if mentioned. Otherwise, return null.
    - years_experience: The years of experience statement if provided (e.g., "over 20 years", "more than two decades"). Otherwise, return null.
    - prior_position: The most recent district-level administrative role the individual held immediately before the appointment (e.g., "assistant superintendent", "executive director"). Otherwise, return null.

    - source_urls: Extract all URLs explicitly present in the answer that serve as evidence (press releases, board minutes, district announcements, bios, news coverage, etc.). 
      Return them as a list of URLs (do not invent any). Include markdown links' target URLs if used.

    If any required information is missing, set that field to null. Do not invent data not present in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_correct_superintendent(
    evaluator: Evaluator,
    parent_node,
    ext: SuperintendentExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and run URL-grounded checks.
    All nodes under the main rubric node are critical as specified.
    """
    # Top-level rubric node (critical, parallel)
    main_node = evaluator.add_parallel(
        id="correct_superintendent_identification",
        desc="Identifies the Pennsylvania school district superintendent appointed in early 2026 who meets all specified qualifications and appointment context requirements",
        parent=parent_node,
        critical=True,
    )

    # Helper text for names/districts in claims
    person = ext.name or "the appointed individual"
    district = ext.district or "the school district referenced in the sources"
    sources = ext.source_urls if ext.source_urls else []

    # ------------------ Geographic and Temporal Verification ------------------ #
    geo_temp_node = evaluator.add_parallel(
        id="geographic_temporal_verification",
        desc="Verifies the appointment occurred in Pennsylvania within the specified timeframe",
        parent=main_node,
        critical=True,
    )

    # Pennsylvania School District
    pa_district_leaf = evaluator.add_leaf(
        id="pennsylvania_school_district",
        desc="The appointment is for a school district located in Pennsylvania",
        parent=geo_temp_node,
        critical=True,
    )
    claim_pa_district = (
        f"The appointment of {person} as superintendent is at {district}, "
        f"and {district} is a public school district in Pennsylvania."
    )
    # Verify using any provided sources
    await evaluator.verify(
        claim=claim_pa_district,
        node=pa_district_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that the school district mentioned is located in the Commonwealth of Pennsylvania. "
            "Accept official district pages, board materials, or reputable news confirming the district is in PA."
        ),
    )

    # Announcement Date Range (Jan 1, 2026 to Mar 19, 2026 inclusive)
    ann_leaf = evaluator.add_leaf(
        id="announcement_date_range",
        desc="The appointment was publicly announced between January 1, 2026 and March 19, 2026 (inclusive)",
        parent=geo_temp_node,
        critical=True,
    )
    if ext.announcement_date:
        claim_announcement = (
            f"The appointment of {person} as superintendent of {district} was publicly announced on {ext.announcement_date}, "
            f"which falls between January 1, 2026 and March 19, 2026 inclusive."
        )
    else:
        claim_announcement = (
            f"The appointment of {person} as superintendent of {district} was publicly announced between "
            f"January 1, 2026 and March 19, 2026 inclusive."
        )
    await evaluator.verify(
        claim=claim_announcement,
        node=ann_leaf,
        sources=sources,
        additional_instruction=(
            "Use the publication date on the press release, board agenda/minutes date, or article date. "
            "If the page clearly states board approval or an announcement date within that window, consider it valid. "
            "The window is inclusive: 2026-01-01 through 2026-03-19."
        ),
    )

    # -------------------------- Individual Qualifications --------------------- #
    qual_node = evaluator.add_parallel(
        id="individual_qualifications",
        desc="Verifies the appointed superintendent possesses all required credentials and experience",
        parent=main_node,
        critical=True,
    )

    # Pennsylvania Superintendent Letter of Eligibility
    loe_leaf = evaluator.add_leaf(
        id="pa_superintendent_letter_of_eligibility",
        desc="The individual holds a Pennsylvania Superintendent Letter of Eligibility",
        parent=qual_node,
        critical=True,
    )
    claim_loe = f"{person} holds a Pennsylvania Superintendent Letter of Eligibility."
    await evaluator.verify(
        claim=claim_loe,
        node=loe_leaf,
        sources=sources,
        additional_instruction=(
            "Look for explicit mention of 'Pennsylvania Superintendent Letter of Eligibility' or standard variants "
            "like 'PA Superintendent Letter of Eligibility'. Statements in official bios, resumes, district releases, "
            "or board docs are acceptable."
        ),
    )

    # Doctoral Degree (any doctoral degree)
    doc_leaf = evaluator.add_leaf(
        id="doctoral_degree",
        desc="The individual holds a doctoral degree (PhD, EdD, or equivalent terminal degree)",
        parent=qual_node,
        critical=True,
    )
    claim_doc = f"{person} holds a doctoral degree such as an Ed.D., Ph.D., or equivalent terminal degree."
    await evaluator.verify(
        claim=claim_doc,
        node=doc_leaf,
        sources=sources,
        additional_instruction=(
            "Accept explicit mentions of 'Ph.D.', 'Ed.D.', 'doctorate', or 'doctoral degree'. "
            "Confirm it refers to an earned terminal doctoral degree."
        ),
    )

    # Pennsylvania University Doctorate
    pa_uni_leaf = evaluator.add_leaf(
        id="pa_university_doctorate",
        desc="The doctoral degree was earned from a university located in Pennsylvania",
        parent=qual_node,
        critical=True,
    )
    if ext.doctoral_university:
        claim_pa_uni = (
            f"{person}'s doctoral degree was earned from {ext.doctoral_university}, a university located in Pennsylvania."
        )
    else:
        claim_pa_uni = (
            f"{person}'s doctoral degree was earned from a university located in Pennsylvania."
        )
    await evaluator.verify(
        claim=claim_pa_uni,
        node=pa_uni_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that the awarding university is based in Pennsylvania (e.g., Temple University, "
            "Penn State, University of Pennsylvania, Duquesne, etc.). If the university name is present, "
            "verify it is a PA institution."
        ),
    )

    # Minimum Experience (>= 20 years)
    exp_leaf = evaluator.add_leaf(
        id="minimum_experience",
        desc="The individual has at least 20 years of experience in education",
        parent=qual_node,
        critical=True,
    )
    claim_exp = f"{person} has at least 20 years of experience in education."
    await evaluator.verify(
        claim=claim_exp,
        node=exp_leaf,
        sources=sources,
        additional_instruction=(
            "Accept phrasings like '20 years', 'over 20 years', 'more than two decades', 'two decades of experience'. "
            "The experience must be in the education field."
        ),
    )

    # ---------------------------- Appointment Context ------------------------- #
    context_node = evaluator.add_parallel(
        id="appointment_context",
        desc="Verifies the circumstances and timing of the superintendent appointment",
        parent=main_node,
        critical=True,
    )

    # District-Level Prior Position
    prior_leaf = evaluator.add_leaf(
        id="district_level_prior_position",
        desc="Immediately before this appointment, the individual held a district-level administrative position (such as assistant superintendent, executive director, or similar)",
        parent=context_node,
        critical=True,
    )
    claim_prior = (
        f"Immediately before this appointment, {person} held a district-level administrative position "
        f"(for example, assistant superintendent, executive director, chief academic officer, or similar)."
    )
    await evaluator.verify(
        claim=claim_prior,
        node=prior_leaf,
        sources=sources,
        additional_instruction=(
            "The evidence should indicate the immediately prior role was district-level administration (not a school-level principal role). "
            "Accept titles like assistant superintendent, deputy superintendent, executive director, chief academic officer, etc."
        ),
    )

    # Succeeding Retiring Superintendent
    retiring_leaf = evaluator.add_leaf(
        id="succeeding_retiring_superintendent",
        desc="The appointment involves succeeding a superintendent who is retiring (not replacing someone who resigned, was terminated, or filling a newly created position)",
        parent=context_node,
        critical=True,
    )
    claim_retiring = (
        "The appointment involves succeeding an outgoing superintendent who is retiring."
    )
    await evaluator.verify(
        claim=claim_retiring,
        node=retiring_leaf,
        sources=sources,
        additional_instruction=(
            "Look for explicit indications that the outgoing superintendent is retiring. "
            "Do not accept resignations, terminations, or newly-created positions."
        ),
    )

    # Start Date After June 2026
    start_leaf = evaluator.add_leaf(
        id="start_date_after_june_2026",
        desc="The official start date for the superintendent role is after June 2026",
        parent=context_node,
        critical=True,
    )
    if ext.start_date:
        claim_start = (
            f"The official start date for {person} as superintendent of {district} is {ext.start_date}, "
            f"which is after June 2026 (i.e., on or after July 1, 2026)."
        )
    else:
        claim_start = (
            f"The official start date for {person} as superintendent of {district} is after June 2026 "
            f"(i.e., on or after July 1, 2026)."
        )
    await evaluator.verify(
        claim=claim_start,
        node=start_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm the effective/start date is strictly after June 30, 2026. "
            "Dates like 'July 1, 2026' or later qualify. Ambiguous 'summer 2026' is not sufficient unless explicitly July or later."
        ),
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
    Evaluate an agent's answer for the Pennsylvania superintendent appointment task.
    """
    # Initialize evaluator
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_superintendent_info(),
        template_class=SuperintendentExtraction,
        extraction_name="superintendent_extraction",
    )

    # Build verification tree and run checks
    await verify_correct_superintendent(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()