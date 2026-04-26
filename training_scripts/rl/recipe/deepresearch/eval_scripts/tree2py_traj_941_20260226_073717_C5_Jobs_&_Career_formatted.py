import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "superintendent_identification"
TASK_DESCRIPTION = """
Identify the current school district superintendent who meets all of the following criteria:

1. Earned an EdD (Doctor of Education) degree in Educational Leadership from Gardner-Webb University
2. Serves as superintendent of a school district that is ranked as the 16th largest public school system in the United States and the 2nd largest in its state
3. Was appointed or hired as superintendent in the year 2023
4. Earned both a bachelor's degree and a master's degree from North Carolina A&T State University
5. Previously held a chief-level leadership position in the same school district before becoming superintendent

Provide the superintendent's full name and the name of the school district.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SuperintendentInfo(BaseModel):
    """Structured extraction of superintendent identification answer."""
    superintendent_name: Optional[str] = None
    district_name: Optional[str] = None
    # Optional helpful details if present in the answer; verification will rely on sources.
    appointment_year: Optional[str] = None
    edd_university: Optional[str] = None
    edd_program: Optional[str] = None
    bachelors_university: Optional[str] = None
    masters_university: Optional[str] = None
    prior_role_title: Optional[str] = None
    state_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_superintendent_info() -> str:
    return """
    Extract the following fields from the answer. Return null for any field not explicitly stated.

    Required identification:
    - superintendent_name: The full name of the superintendent identified in the answer.
    - district_name: The complete name of the school district (e.g., "Charlotte-Mecklenburg Schools").
    
    Helpful optional fields (extract if present verbatim):
    - appointment_year: The year when the person was appointed or hired as superintendent (just the year, e.g., "2023").
    - edd_university: The university awarding the EdD (e.g., "Gardner-Webb University").
    - edd_program: The program name for the EdD (e.g., "Educational Leadership").
    - bachelors_university: University awarding the bachelor's degree (e.g., "North Carolina A&T State University").
    - masters_university: University awarding the master's degree (e.g., "North Carolina A&T State University").
    - prior_role_title: The chief-level position title previously held in the same district (e.g., "Chief of Schools", "Chief Academic Officer").
    - state_name: The state the district belongs to if explicitly mentioned (e.g., "North Carolina").
    
    Sources:
    - sources: Extract ALL URLs explicitly mentioned in the answer that may support the identification and constraints.
      These may be presented as plain URLs or markdown links; extract the actual URL strings. If none are present, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_superintendent_verification(
    evaluator: Evaluator,
    root_node,
    info: SuperintendentInfo,
) -> None:
    """
    Build the verification tree and execute the checks for superintendent identification.
    """
    # Parent node: critical parallel aggregator to enforce "all must be true"
    super_node = evaluator.add_parallel(
        id="Superintendent_Identification",
        desc="Identify the superintendent and school district that satisfy all stated constraints, and provide both names.",
        parent=root_node,
        critical=True
    )

    # Existence check: both names must be present in the answer
    has_names = bool(info.superintendent_name and info.superintendent_name.strip()) and \
                bool(info.district_name and info.district_name.strip())
    evaluator.add_custom_node(
        result=has_names,
        id="Answer_Provides_Both_Required_Names",
        desc="Answer provides both the superintendent's full name and the complete name of the school district.",
        parent=super_node,
        critical=True
    )

    name = info.superintendent_name or ""
    district = info.district_name or ""
    sources: List[str] = info.sources or []

    # Prepare leaf nodes
    node_current = evaluator.add_leaf(
        id="Current_Superintendent_Status",
        desc="The identified person is the current (incumbent) superintendent of the named school district.",
        parent=super_node,
        critical=True
    )
    node_edd = evaluator.add_leaf(
        id="EdD_Gardner_Webb_Educational_Leadership",
        desc="Superintendent earned an EdD (Doctor of Education) in Educational Leadership from Gardner-Webb University.",
        parent=super_node,
        critical=True
    )
    node_rank16 = evaluator.add_leaf(
        id="District_Ranked_16th_US",
        desc="The superintendent's school district is ranked as the 16th largest public school system in the United States.",
        parent=super_node,
        critical=True
    )
    node_rank2 = evaluator.add_leaf(
        id="District_Ranked_2nd_In_State",
        desc="The superintendent's school district is the 2nd largest in its state.",
        parent=super_node,
        critical=True
    )
    node_hired2023 = evaluator.add_leaf(
        id="Appointed_Or_Hired_In_2023",
        desc="Superintendent was appointed or hired as superintendent in the year 2023.",
        parent=super_node,
        critical=True
    )
    node_nc_at = evaluator.add_leaf(
        id="Bachelors_And_Masters_From_NC_AT",
        desc="Superintendent earned both a bachelor's degree and a master's degree from North Carolina A&T State University.",
        parent=super_node,
        critical=True
    )
    node_prior_chief = evaluator.add_leaf(
        id="Prior_Chief_Level_Role_In_Same_District",
        desc="Before becoming superintendent, the person held a chief-level leadership position in the same school district.",
        parent=super_node,
        critical=True
    )

    # Build claims
    prior_role_phrase = f", serving as '{info.prior_role_title}'" if info.prior_role_title else ""
    state_phrase = f" in {info.state_name}" if info.state_name else ""

    claims_and_sources = [
        (
            f"{name} is the current superintendent of {district}.",
            sources,
            node_current,
            "Verify that the person is explicitly described as the current superintendent (not former or interim unless explicitly current). Allow reasonable phrasing variants such as 'superintendent' or 'district leader'."
        ),
        (
            f"{name} earned an EdD (Doctor of Education) in Educational Leadership from Gardner-Webb University.",
            sources,
            node_edd,
            "Confirm both the degree (EdD/Doctor of Education) and the program (Educational Leadership) from Gardner-Webb University. Allow minor naming variants like 'Ed.D.' or 'Doctorate in Educational Leadership'."
        ),
        (
            f"{district} is ranked the 16th largest public school system in the United States.",
            sources,
            node_rank16,
            "Verify that a credible source explicitly states the district is the 16th largest in the U.S. Accept phrasing like '16th-largest' or 'ranked No. 16'."
        ),
        (
            f"{district} is the 2nd largest school district in its state{state_phrase}.",
            sources,
            node_rank2,
            "Verify a credible statement indicating the district is the second-largest in its state. Accept phrasing variants like 'second-largest' or 'No. 2'."
        ),
        (
            f"{name} was appointed or hired as superintendent in 2023.",
            sources,
            node_hired2023,
            "Confirm appointment or hire occurred in calendar year 2023. Accept phrasing such as 'hired in 2023' or 'appointed in 2023'."
        ),
        (
            f"{name} earned both a bachelor's degree and a master's degree from North Carolina A&T State University.",
            sources,
            node_nc_at,
            "Verify both degrees (bachelor's and master's) are from North Carolina A&T State University. Accept variants like 'NC A&T', 'N.C. A&T State University'."
        ),
        (
            f"Before becoming superintendent, {name} held a chief-level leadership position in {district}{prior_role_phrase}.",
            sources,
            node_prior_chief,
            "Confirm a previous chief-level role in the SAME district (e.g., Chief of Schools, Chief Academic Officer, Chief Operating Officer, Chief of Staff). The role title can vary but must clearly be a 'chief'-level leadership position."
        ),
    ]

    # Execute verifications in parallel
    await evaluator.batch_verify(claims_and_sources)


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
) -> Dict:
    """
    Entry point for evaluating an answer for the superintendent identification task.
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_superintendent_info(),
        template_class=SuperintendentInfo,
        extraction_name="superintendent_info",
    )

    # Build verification tree and run checks
    await build_superintendent_verification(evaluator, root, extracted)

    # Optional: Record constraints for transparency
    evaluator.add_custom_info(
        info={
            "constraints": [
                "EdD in Educational Leadership from Gardner-Webb University",
                "District is 16th largest in US",
                "District is 2nd largest in its state",
                "Appointed/hired in 2023",
                "Bachelor's and Master's from North Carolina A&T State University",
                "Prior chief-level role in same district",
                "Answer provides both superintendent and district names"
            ]
        },
        info_type="constraints_summary"
    )

    # Return evaluation summary
    return evaluator.get_summary()