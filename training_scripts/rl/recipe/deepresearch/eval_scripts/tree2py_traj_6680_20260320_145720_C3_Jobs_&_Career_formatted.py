import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "al_superintendent_min_years"
TASK_DESCRIPTION = """
In the state where Auburn University is located, what is the minimum total number of years of full-time educational experience (combining both teaching and administrative roles) that an individual would need to accumulate to become eligible for a superintendent position, assuming they begin their career with a bachelor's degree and follow the standard progression of first obtaining the necessary teaching experience before pursuing administrative certification, and then gaining the required administrative experience before applying for a superintendent role?
"""

ROOT_DESC = (
    "Correctly identify the minimum total years of full-time educational experience (combining teaching and administrative roles) "
    "required for someone to become eligible for a superintendent position in Alabama, assuming they start with a bachelor's degree "
    "and follow the standard career progression through principal certification"
)

STATE_NODE_DESC = "Correctly identify that Auburn University is located in Alabama"
TEACHING_NODE_DESC = "Correctly identify the minimum years of teaching experience required in Alabama before one can pursue principal/administrative certification"
ADMIN_NODE_DESC = "Correctly identify the minimum years of administrative experience (as principal or in administrative role) required in Alabama for superintendent eligibility"
TOTAL_NODE_DESC = "Correctly calculate the total minimum years as the sum of teaching experience years plus administrative experience years"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StateInfo(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ExperienceRequirement(BaseModel):
    # Text exactly as stated in the answer (e.g., "three years", "2–3 years", "36 months")
    min_years_text: Optional[str] = None
    # Minimum numeric years as an integer string if possible (e.g., "3"); if unknown, set null
    min_years_number: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    statement: Optional[str] = None  # Optional snippet or summary from the answer


class TotalInfo(BaseModel):
    # Total years as stated in the answer (e.g., "6", "5-6", "at least 5")
    total_years_text: Optional[str] = None
    # Minimum numeric total as an integer string if possible (e.g., "6"); if unknown, set null
    total_min_years_number: Optional[str] = None
    rationale: Optional[str] = None  # Optional explanation from the answer


class SuperintendentEligibilityExtraction(BaseModel):
    state: Optional[StateInfo] = None
    teaching: Optional[ExperienceRequirement] = None
    administrative: Optional[ExperienceRequirement] = None
    total: Optional[TotalInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
    Extract the following information exactly as presented in the answer. Do NOT invent anything.

    1) state:
       - name: The U.S. state explicitly identified in the answer as the state where Auburn University is located.
       - sources: All URLs cited in the answer that directly support Auburn University's location in that state.

    2) teaching:
       - min_years_text: The minimum years of full-time teaching (or equivalent "P-12 professional educator experience") required in Alabama before pursuing principal/administrative certification, as stated in the answer.
       - min_years_number: Convert the minimum to a numeric years string when possible (e.g., "three years" -> "3"; "2–3 years" -> "2"; "36 months" -> "3"). If unclear or not numeric, set to null.
       - sources: All URLs in the answer that document Alabama's requirement for teaching experience before principal/administrative certification.
       - statement: Short snippet summarizing the requirement from the answer (optional).

    3) administrative:
       - min_years_text: The minimum years of administrative experience (e.g., as a principal or in an administrative leadership role) required in Alabama to be eligible for a superintendent role, as stated in the answer.
       - min_years_number: Convert the minimum to a numeric years string when possible (e.g., ranges -> take the minimum). If unclear or not numeric, set to null.
       - sources: All URLs cited in the answer that document Alabama's administrative experience requirement for superintendent eligibility.
       - statement: Short snippet summarizing the requirement from the answer (optional).

    4) total:
       - total_years_text: The total minimum years of full-time educational experience (teaching + administrative) stated in the answer.
       - total_min_years_number: Convert the minimum total to a numeric years string when possible. If unclear or not provided, set to null.
       - rationale: The explanation or calculation from the answer (optional).

    IMPORTANT:
    - Only extract URLs explicitly present in the answer (including Markdown links). Do not infer or create URLs.
    - If any field is missing from the answer, set it to null (or an empty list for URLs).
    - When converting to numeric years, always choose the minimum value if ranges/options are presented.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_state_section(evaluator: Evaluator, parent, extracted: SuperintendentEligibilityExtraction):
    state_node = evaluator.add_sequential(
        id="State_Identification",
        desc=STATE_NODE_DESC,
        parent=parent,
        critical=True
    )

    state_sources = (extracted.state.sources if extracted and extracted.state and extracted.state.sources else [])
    # Existence check for URL(s)
    evaluator.add_custom_node(
        result=bool(state_sources),
        id="state_url_reference_provided",
        desc="At least one URL is provided to support Auburn University's location in Alabama",
        parent=state_node,
        critical=True
    )

    # Verify the claim using the provided URL(s)
    state_verify_leaf = evaluator.add_leaf(
        id="State_URL_Reference",
        desc="Provide a valid URL reference confirming Auburn University's location in Alabama",
        parent=state_node,
        critical=True
    )
    await evaluator.verify(
        claim="Auburn University is located in Alabama.",
        node=state_verify_leaf,
        sources=state_sources,
        additional_instruction="Use only the provided URL(s) to confirm the institution's state. Accept authoritative pages such as the official Auburn University site or reputable reference pages that clearly state the location."
    )


async def verify_teaching_section(evaluator: Evaluator, parent, extracted: SuperintendentEligibilityExtraction):
    teaching_node = evaluator.add_sequential(
        id="Teaching_Experience_Requirement",
        desc=TEACHING_NODE_DESC,
        parent=parent,
        critical=True
    )

    t_years_text = extracted.teaching.min_years_text if extracted and extracted.teaching else None
    t_years_num = extracted.teaching.min_years_number if extracted and extracted.teaching else None
    t_sources = extracted.teaching.sources if extracted and extracted.teaching and extracted.teaching.sources else []

    # Existence check: needs both a stated minimum and at least one URL
    evaluator.add_custom_node(
        result=bool(t_years_text) and bool(t_sources),
        id="teaching_requirement_present",
        desc="Answer provides a stated minimum teaching experience and at least one supporting URL for Alabama",
        parent=teaching_node,
        critical=True
    )

    teaching_verify_leaf = evaluator.add_leaf(
        id="Teaching_Experience_URL_Reference",
        desc="Provide a valid URL reference documenting Alabama's teaching experience requirement for administrative certification",
        parent=teaching_node,
        critical=True
    )

    # Prefer numeric years if available; otherwise fall back to text
    years_phrase = t_years_num if (t_years_num and t_years_num.strip()) else (t_years_text or "UNKNOWN")

    await evaluator.verify(
        claim=f"In Alabama, the minimum years of full-time teaching (or equivalent P-12 professional educator experience) required before pursuing principal/administrative certification is {years_phrase} year(s).",
        node=teaching_verify_leaf,
        sources=t_sources,
        additional_instruction=(
            "Verify this against Alabama's principal/educational leadership certification requirements. "
            "Allow synonymous phrases like 'P-12 professional educator experience' or 'successful teaching experience'. "
            "If multiple pathways/options exist, judge the minimum years under a standard route."
        )
    )


async def verify_admin_section(evaluator: Evaluator, parent, extracted: SuperintendentEligibilityExtraction):
    admin_node = evaluator.add_sequential(
        id="Administrative_Experience_Requirement",
        desc=ADMIN_NODE_DESC,
        parent=parent,
        critical=True
    )

    a_years_text = extracted.administrative.min_years_text if extracted and extracted.administrative else None
    a_years_num = extracted.administrative.min_years_number if extracted and extracted.administrative else None
    a_sources = extracted.administrative.sources if extracted and extracted.administrative and extracted.administrative.sources else []

    # Existence check: needs both a stated minimum and at least one URL
    evaluator.add_custom_node(
        result=bool(a_years_text) and bool(a_sources),
        id="administrative_requirement_present",
        desc="Answer provides a stated minimum administrative experience and at least one supporting URL for Alabama superintendent eligibility",
        parent=admin_node,
        critical=True
    )

    admin_verify_leaf = evaluator.add_leaf(
        id="Administrative_Experience_URL_Reference",
        desc="Provide a valid URL reference documenting Alabama's administrative experience requirement for superintendent positions",
        parent=admin_node,
        critical=True
    )

    years_phrase = a_years_num if (a_years_num and a_years_num.strip()) else (a_years_text or "UNKNOWN")

    await evaluator.verify(
        claim=f"In Alabama, the minimum years of administrative leadership experience required to be eligible for a superintendent position is {years_phrase} year(s).",
        node=admin_verify_leaf,
        sources=a_sources,
        additional_instruction=(
            "Confirm the minimum administrative experience threshold specifically for Alabama superintendent eligibility. "
            "Administrative roles may include principal or district-level leadership. "
            "If multiple standards exist, evaluate the minimum common requirement under a standard route."
        )
    )


async def verify_total_section(evaluator: Evaluator, parent, extracted: SuperintendentEligibilityExtraction):
    # The JSON places this as a single critical check; we implement as a single leaf under the critical parent.
    total_leaf = evaluator.add_leaf(
        id="Total_Calculation_Verification",
        desc=TOTAL_NODE_DESC,
        parent=parent,
        critical=True
    )

    t_years = None
    a_years = None
    z_total = None

    if extracted and extracted.teaching:
        t_years = extracted.teaching.min_years_number or extracted.teaching.min_years_text
    if extracted and extracted.administrative:
        a_years = extracted.administrative.min_years_number or extracted.administrative.min_years_text
    if extracted and extracted.total:
        z_total = extracted.total.total_min_years_number or extracted.total.total_years_text

    # Fall back to 'UNKNOWN' tokens to make the claim explicit even if something is missing;
    # If missing, the verifier is expected to mark it incorrect.
    t_phrase = t_years or "UNKNOWN"
    a_phrase = a_years or "UNKNOWN"
    z_phrase = z_total or "UNKNOWN"

    claim = (
        f"Given a minimum teaching experience of '{t_phrase}' year(s) and a minimum administrative experience of '{a_phrase}' year(s), "
        f"the stated minimum total required years is '{z_phrase}' year(s), and this total correctly equals the sum of the two minima."
    )

    await evaluator.verify(
        claim=claim,
        node=total_leaf,
        additional_instruction=(
            "Check the arithmetic only. Interpret words or ranges by taking the minimum value implied for each component. "
            "Accept reasonable normalization (e.g., '2–3 years' -> minimum 2). If any component is UNKNOWN or missing, mark as incorrect."
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
    Evaluate an answer for Alabama superintendent minimum years requirement (teaching + admin) with URL-grounded verification.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # As specified by the rubric root
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

    # Create an overall critical node under the (non-critical) framework root to mirror rubric root's criticality
    overall = evaluator.add_parallel(
        id="Overall",
        desc=ROOT_DESC,
        parent=root,
        critical=True
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=SuperintendentEligibilityExtraction,
        extraction_name="extracted_superintendent_requirements"
    )

    # Ground truth/context info (non-binding, for summary)
    evaluator.add_ground_truth({
        "target_state": "Alabama",
        "assumptions": [
            "Candidate starts with a bachelor's degree.",
            "Follows the standard progression: gain required teaching experience -> pursue administrative/principal certification -> gain required administrative experience -> apply for superintendent."
        ]
    })

    # Build and verify sections
    await verify_state_section(evaluator, overall, extracted)
    await verify_teaching_section(evaluator, overall, extracted)
    await verify_admin_section(evaluator, overall, extracted)
    await verify_total_section(evaluator, overall, extracted)

    return evaluator.get_summary()