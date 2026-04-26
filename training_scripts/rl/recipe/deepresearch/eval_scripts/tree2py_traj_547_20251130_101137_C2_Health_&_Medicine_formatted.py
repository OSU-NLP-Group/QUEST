import asyncio
import logging
from typing import Any, List, Dict, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "musician_actor_stroke_2024"
TASK_DESCRIPTION = (
    "In 2024, a 42-year-old musician and actor experienced a significant medical emergency while on tour. "
    "During a performance in New Orleans, Louisiana, this individual suffered severe head pain and vision problems "
    "but continued to perform the show. At the next tour stop in Houston, Texas, the individual sought medical evaluation "
    "at a hospital where doctors diagnosed a stroke and discovered a hole in the individual's heart. Identify this musician/actor "
    "by providing their legal name and stage name. Additionally, provide a reference URL that confirms these specific details "
    "about the medical event."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PersonEventInfo(BaseModel):
    """Extracted identification and sources from the answer."""
    legal_name: Optional[str] = None
    stage_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_person_event_info() -> str:
    return (
        "Extract the identification details and reference URLs provided in the answer.\n"
        "Return a JSON object with the following fields:\n"
        "1. legal_name: The individual's legal/birth name as stated in the answer (return null if not provided).\n"
        "2. stage_name: The individual's stage/artist name as stated in the answer (return null if not provided).\n"
        "3. reference_urls: An array of all URLs cited or linked in the answer that are intended to corroborate the described medical event. "
        "   Extract actual URLs even if they are in markdown form. If none are provided, return an empty array.\n"
        "Do not invent or infer information; extract exactly what appears in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def subject_ref(info: PersonEventInfo) -> str:
    """Build a reference string for the subject, preferring stage name."""
    if info.stage_name and info.legal_name:
        return f"{info.stage_name} (legal name {info.legal_name})"
    if info.stage_name:
        return info.stage_name
    if info.legal_name:
        return info.legal_name
    return "the individual"


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def build_verification_tree_and_verify(
    evaluator: Evaluator,
    extracted: PersonEventInfo,
) -> None:
    """
    Build the verification tree according to the rubric and run all checks.
    """

    # Create the top-level critical node mirroring the rubric's "Individual_Identification_and_Evidence"
    top_node = evaluator.add_parallel(
        id="Individual_Identification_and_Evidence",
        desc="Answer identifies the correct individual, provides required names, satisfies all stated constraints, and includes a corroborating reference URL.",
        critical=True,
    )

    # 1) Names Provided (critical, parallel)
    names_node = evaluator.add_parallel(
        id="Names_Provided",
        desc="Answer provides the required identification names for the individual.",
        parent=top_node,
        critical=True,
    )

    legal_name_exists = bool(extracted.legal_name and extracted.legal_name.strip())
    stage_name_exists = bool(extracted.stage_name and extracted.stage_name.strip())

    evaluator.add_custom_node(
        result=legal_name_exists,
        id="Legal_Name_Provided",
        desc="Provides the individual's legal name.",
        parent=names_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=stage_name_exists,
        id="Stage_Name_Provided",
        desc="Provides the individual's stage name.",
        parent=names_node,
        critical=True,
    )

    # 2) Reference URL (critical, parallel)
    ref_node = evaluator.add_parallel(
        id="Reference_URL",
        desc="Answer includes a reference URL that corroborates the required details.",
        parent=top_node,
        critical=True,
    )

    # Existence prerequisite for all URL-based checks
    url_exists_node = evaluator.add_custom_node(
        result=(len(extracted.reference_urls) > 0),
        id="Reference_URL_Exists",
        desc="At least one reference URL is provided in the answer.",
        parent=ref_node,
        critical=True,
    )

    # Leaf that checks core corroboration by URLs (stroke + hole in heart)
    url_corroborate_leaf = evaluator.add_leaf(
        id="URL_Provided_And_Corroborates_Required_Details",
        desc="Provides at least one reference URL, and the URL(s) corroborate the required constraints listed in the task.",
        parent=ref_node,
        critical=True,
    )
    # Core corroboration claim
    subj = subject_ref(extracted)
    core_claim = (
        f"The provided reference page(s) corroborate that {subj} suffered a stroke in 2024 "
        f"and that doctors discovered a hole in the heart (e.g., a patent foramen ovale/PFO)."
    )
    await evaluator.verify(
        claim=core_claim,
        node=url_corroborate_leaf,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Confirm both parts: (1) the stroke diagnosis in 2024, and (2) that a hole in the heart "
            "was discovered (e.g., PFO). If multiple URLs are provided, it's acceptable if different URLs "
            "corroborate different parts, as long as collectively they support the claim."
        ),
        extra_prerequisites=[url_exists_node],
    )

    # 3) Event and Context Constraints (critical, parallel)
    constraints_node = evaluator.add_parallel(
        id="Event_and_Context_Constraints",
        desc="All required event/context constraints from the constraints list are satisfied.",
        parent=top_node,
        critical=True,
    )

    # Build all constraint leaves
    # Age 42 in 2024
    age_leaf = evaluator.add_leaf(
        id="Age_42_in_2024",
        desc="The individual was 42 years old at the time of the stroke in 2024.",
        parent=constraints_node,
        critical=True,
    )
    age_claim = f"In 2024, {subj} was 42 years old at the time of the stroke."
    await evaluator.verify(
        claim=age_claim,
        node=age_leaf,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Accept either (a) an explicit statement that the person was 42 years old in 2024, "
            "or (b) a birthdate that implies age 42 in 2024."
        ),
        extra_prerequisites=[url_exists_node],
    )

    # Musician & actor; on tour
    ma_leaf = evaluator.add_leaf(
        id="Musician_Actor_On_Tour",
        desc="The individual is a musician and actor who was performing on tour.",
        parent=constraints_node,
        critical=True,
    )
    ma_claim = f"{subj} is described as a musician and actor and was performing on a tour."
    await evaluator.verify(
        claim=ma_claim,
        node=ma_leaf,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Allow common synonyms for 'musician' (e.g., 'rapper', 'singer', 'recording artist') and for 'actor'. "
            "Confirm that the page indicates they were performing on a tour."
        ),
        extra_prerequisites=[url_exists_node],
    )

    # Symptoms in New Orleans
    no_leaf = evaluator.add_leaf(
        id="Symptoms_in_New_Orleans",
        desc="Severe head pain and vision problems occurred during a performance in New Orleans, Louisiana.",
        parent=constraints_node,
        critical=True,
    )
    no_claim = (
        f"During a performance in New Orleans, Louisiana, {subj} experienced severe head pain and vision problems."
    )
    await evaluator.verify(
        claim=no_claim,
        node=no_leaf,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Vision problems may include phrasing such as 'blurry vision', 'double vision', or 'vision issues'. "
            "Head pain may be described as 'severe headache' or similar."
        ),
        extra_prerequisites=[url_exists_node],
    )

    # Continued performance despite symptoms
    cont_leaf = evaluator.add_leaf(
        id="Continued_Performance",
        desc="The individual continued to perform the show despite experiencing symptoms.",
        parent=constraints_node,
        critical=True,
    )
    cont_claim = f"Despite these symptoms, {subj} continued to perform the show."
    await evaluator.verify(
        claim=cont_claim,
        node=cont_leaf,
        sources=extracted.reference_urls,
        additional_instruction="Confirm that the page states the person kept performing after the symptoms began.",
        extra_prerequisites=[url_exists_node],
    )

    # Diagnosis in Houston next stop
    hou_leaf = evaluator.add_leaf(
        id="Diagnosis_in_Houston_Next_Stop",
        desc="At the next tour stop in Houston, Texas, the individual sought medical evaluation at a hospital where doctors diagnosed a stroke.",
        parent=constraints_node,
        critical=True,
    )
    hou_claim = (
        f"At the next tour stop in Houston, Texas, {subj} went to a hospital where doctors diagnosed a stroke."
    )
    await evaluator.verify(
        claim=hou_claim,
        node=hou_leaf,
        sources=extracted.reference_urls,
        additional_instruction=(
            "The timeline must indicate that Houston was the next stop and that a hospital evaluation there resulted in a stroke diagnosis."
        ),
        extra_prerequisites=[url_exists_node],
    )

    # Hole in heart found
    hole_leaf = evaluator.add_leaf(
        id="Hole_in_Heart_Found",
        desc="Doctors discovered a hole in the individual's heart during the medical evaluation.",
        parent=constraints_node,
        critical=True,
    )
    hole_claim = (
        f"Doctors discovered a hole in {subject_ref(extracted)}'s heart, such as a patent foramen ovale (PFO)."
    )
    await evaluator.verify(
        claim=hole_claim,
        node=hole_leaf,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Accept phrasing such as 'hole in the heart', 'PFO', or 'patent foramen ovale' as equivalent."
        ),
        extra_prerequisites=[url_exists_node],
    )

    # Two surgeries
    surg_leaf = evaluator.add_leaf(
        id="Two_Surgeries",
        desc="The individual underwent two surgeries related to the medical findings.",
        parent=constraints_node,
        critical=True,
    )
    surg_claim = f"{subj} underwent two surgeries related to these medical findings."
    await evaluator.verify(
        claim=surg_claim,
        node=surg_leaf,
        sources=extracted.reference_urls,
        additional_instruction="Accept equivalents such as 'two procedures' if the context clearly indicates surgeries.",
        extra_prerequisites=[url_exists_node],
    )

    # Public reveal at Camp Flog Gnaw on Nov 22 in Los Angeles
    cfg_leaf = evaluator.add_leaf(
        id="Public_Reveal_Camp_Flog_Gnaw_Nov_22",
        desc="The information was publicly revealed on November 22 at the Camp Flog Gnaw music festival in Los Angeles.",
        parent=constraints_node,
        critical=True,
    )
    cfg_claim = (
        f"The information was publicly revealed on November 22 at the Camp Flog Gnaw music festival in Los Angeles."
    )
    await evaluator.verify(
        claim=cfg_claim,
        node=cfg_leaf,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Allow phrasing variations, e.g., 'Nov. 22', references to 'Tyler, the Creator's Camp Flog Gnaw festival', "
            "and confirm Los Angeles as the location."
        ),
        extra_prerequisites=[url_exists_node],
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
    Evaluate an answer for the musician/actor stroke-in-2024 identification task.
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

    # 1) Extract identification and sources
    extracted = await evaluator.extract(
        prompt=prompt_extract_person_event_info(),
        template_class=PersonEventInfo,
        extraction_name="person_event_info",
    )

    # 2) Build verification tree and run checks
    await build_verification_tree_and_verify(evaluator, extracted)

    # 3) Return structured summary
    return evaluator.get_summary()