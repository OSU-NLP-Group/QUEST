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
TASK_ID = "ad_career_profiles_2025"
TASK_DESCRIPTION = (
    "You are preparing a professional development resource document analyzing recent athletic director career transitions at NCAA institutions. "
    "Provide comprehensive career background profiles for the following three individuals who were appointed to athletic director positions in 2025:\n\n"
    "1. Jeremy L. Gibson (appointed to Lehigh University, announced January 2025)\n"
    "2. Shawn Tucker (appointed to Rowan University, announced April 2025)\n"
    "3. Damon Evans (appointed to SMU, announced March 2025)\n\n"
    "For each individual, provide:\n"
    "a) Their complete job title and institution in their immediately prior position before their current appointment\n"
    "b) The specific announcement date or effective date of their current appointment\n"
    "c) Their highest earned degree, including the field of study and the granting institution\n"
    "d) One major professional accomplishment or significant achievement from their immediately prior role\n\n"
    "Present the information in a structured format with supporting reference URLs."
)

# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class PersonProfile(BaseModel):
    name: Optional[str] = None
    prior_title: Optional[str] = None
    prior_institution: Optional[str] = None
    appointment_date: Optional[str] = None  # Either announcement or effective date presented in the answer
    degree_level: Optional[str] = None      # e.g., Bachelor's, Master's, etc.
    degree_field: Optional[str] = None      # e.g., Psychology, City & Regional Planning, Sport Management
    degree_institution: Optional[str] = None
    accomplishment: Optional[str] = None
    sources: List[str] = Field(default_factory=list)  # All URLs cited in the answer specifically for this person


class ProfilesExtraction(BaseModel):
    jeremy_gibson: Optional[PersonProfile] = None
    shawn_tucker: Optional[PersonProfile] = None
    damon_evans: Optional[PersonProfile] = None


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_profiles() -> str:
    return (
        "Extract structured career background information for the following three individuals, based strictly on the provided answer text. "
        "You must not invent information; only extract what is present.\n\n"
        "Required people and fields:\n"
        "1) Jeremy L. Gibson\n"
        "2) Shawn Tucker\n"
        "3) Damon Evans\n\n"
        "For each person, extract the following fields (use null if missing):\n"
        "- name: The full name of the individual.\n"
        "- prior_title: The complete job title in their immediately prior role before the current 2025 appointment.\n"
        "- prior_institution: The institution or organization of that prior role.\n"
        "- appointment_date: The specific announcement date OR effective date for the current appointment (choose the one explicitly stated in the answer; do not invent). "
        "Accept any reasonable date formatting if present (e.g., 'Jan. 6, 2025' or 'January 6, 2025').\n"
        "- degree_level: Highest earned degree (e.g., Bachelor's, Master's, PhD). If multiple degrees are listed, select the highest level.\n"
        "- degree_field: The field of study for the highest degree.\n"
        "- degree_institution: The granting institution for the highest degree.\n"
        "- accomplishment: One major professional accomplishment or significant achievement from their immediately prior role as described in the answer.\n"
        "- sources: An array of all URLs explicitly cited in the answer that pertain to this person's profile. "
        "Include URLs in any format (plain URL or markdown link). Do not include unrelated URLs.\n\n"
        "Return a JSON object with keys 'jeremy_gibson', 'shawn_tucker', and 'damon_evans', each containing the above fields."
    )


# --------------------------------------------------------------------------- #
# Helper Functions                                                            #
# --------------------------------------------------------------------------- #
def safe_str(x: Optional[str]) -> str:
    return x if (x is not None) else ""


async def verify_person_profile(
    evaluator: Evaluator,
    parent_node,
    person: PersonProfile,
    person_label: str,
    id_prefix: str
) -> None:
    """
    Build verification subtree for a single person's profile and run verifications.
    All four checks are critical under this person's node, matching rubric.
    """
    profile_node = evaluator.add_parallel(
        id=f"{id_prefix}_profile",
        desc=f"Complete career background information for {person_label}",
        parent=parent_node,
        critical=False
    )

    # 1) Prior position (title + institution)
    prior_leaf = evaluator.add_leaf(
        id=f"{id_prefix}_prior_position",
        desc=f"Correctly identifies {person_label}'s prior position as the full title and institution immediately before the 2025 appointment",
        parent=profile_node,
        critical=True
    )
    prior_title = safe_str(person.prior_title)
    prior_inst = safe_str(person.prior_institution)
    prior_claim = (
        f"Immediately prior to the 2025 appointment, {person_label}'s position was '{prior_title}' at {prior_inst}."
    )
    await evaluator.verify(
        claim=prior_claim,
        node=prior_leaf,
        sources=person.sources,
        additional_instruction=(
            "Use the cited URLs to confirm the immediate prior role (title and institution). "
            "Allow minor wording/casing variants (e.g., '&' vs 'and', presence/absence of department qualifiers). "
            "The role must be the position held directly before the current 2025 athletic director appointment."
        ),
    )

    # 2) Appointment date (announcement or effective date)
    appt_leaf = evaluator.add_leaf(
        id=f"{id_prefix}_appointment_date",
        desc=f"Correctly identifies {person_label}'s appointment announcement date OR effective date (as cited in the answer)",
        parent=profile_node,
        critical=True
    )
    appt_date = safe_str(person.appointment_date)
    appt_claim = (
        f"The appointment announcement or effective date for {person_label}'s 2025 athletic director role was {appt_date}."
    )
    await evaluator.verify(
        claim=appt_claim,
        node=appt_leaf,
        sources=person.sources,
        additional_instruction=(
            "Verify the specific appointment date (either announcement or effective date) from the cited official sources (e.g., press releases). "
            "Accept standard date formatting variations (e.g., 'Jan. 6, 2025' vs 'January 6, 2025')."
        ),
    )

    # 3) Highest earned degree (level + field + institution)
    edu_leaf = evaluator.add_leaf(
        id=f"{id_prefix}_education",
        desc=f"Correctly identifies {person_label}'s highest degree as degree level + field + granting institution",
        parent=profile_node,
        critical=True
    )
    degree_level = safe_str(person.degree_level)
    degree_field = safe_str(person.degree_field)
    degree_inst = safe_str(person.degree_institution)
    edu_claim = (
        f"{person_label}'s highest earned degree is {degree_level} in {degree_field} from {degree_inst}."
    )
    await evaluator.verify(
        claim=edu_claim,
        node=edu_leaf,
        sources=person.sources,
        additional_instruction=(
            "Confirm the highest earned degree using the cited URLs. "
            "Allow common variants (e.g., Bachelor's vs BA; Master's vs MS; 'Sport Management' vs 'Sports Management'). "
            "Ensure the field and institution match the source."
        ),
    )

    # 4) Major accomplishment from prior role
    acc_leaf = evaluator.add_leaf(
        id=f"{id_prefix}_accomplishment",
        desc=f"Correctly identifies a major accomplishment from {person_label}'s immediately prior role",
        parent=profile_node,
        critical=True
    )
    accomplishment = safe_str(person.accomplishment)
    acc_claim = (
        f"A major accomplishment from {person_label}'s immediately prior role was: {accomplishment}."
    )
    await evaluator.verify(
        claim=acc_claim,
        node=acc_leaf,
        sources=person.sources,
        additional_instruction=(
            "Verify that the stated accomplishment is explicitly supported by the cited sources and tied to the person's immediately prior role. "
            "Accept concise paraphrases and reasonable wording variations if the source clearly supports the substance of the achievement."
        ),
    )


# --------------------------------------------------------------------------- #
# Main Evaluation Entry                                                       #
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
    Evaluate the provided answer for completeness and accuracy of the 2025 athletic director profiles.
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

    # Add top-level node mirroring rubric
    profiles_main = evaluator.add_parallel(
        id="Athletic_Director_Profiles",
        desc="Complete and accurate career background profiles for all three athletic directors",
        parent=root,
        critical=False
    )

    # Extract structured profiles from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_profiles(),
        template_class=ProfilesExtraction,
        extraction_name="profiles_extraction"
    )

    # Build and verify each person's subtree
    # Jeremy L. Gibson
    await verify_person_profile(
        evaluator=evaluator,
        parent_node=profiles_main,
        person=extracted.jeremy_gibson or PersonProfile(name="Jeremy L. Gibson"),
        person_label="Jeremy L. Gibson",
        id_prefix="Jeremy_Gibson"
    )

    # Shawn Tucker
    await verify_person_profile(
        evaluator=evaluator,
        parent_node=profiles_main,
        person=extracted.shawn_tucker or PersonProfile(name="Shawn Tucker"),
        person_label="Shawn Tucker",
        id_prefix="Shawn_Tucker"
    )

    # Damon Evans
    await verify_person_profile(
        evaluator=evaluator,
        parent_node=profiles_main,
        person=extracted.damon_evans or PersonProfile(name="Damon Evans"),
        person_label="Damon Evans",
        id_prefix="Damon_Evans"
    )

    return evaluator.get_summary()