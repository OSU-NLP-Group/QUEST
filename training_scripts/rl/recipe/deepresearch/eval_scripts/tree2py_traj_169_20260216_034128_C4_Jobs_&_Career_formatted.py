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
TASK_ID = "univ_career_services_position"
TASK_DESCRIPTION = (
    "Identify a university career services director or assistant/associate director position in the United States "
    "that meets ALL of the following requirements:\n\n"
    "1. The university must have a Center for Career and Professional Success or equivalent career services department\n"
    "2. The university must offer a structured career readiness program for students (such as digital badge programs, "
    "multi-step career development programs, or similar formalized initiatives)\n"
    "3. The career services department must provide services through a major career platform (such as Handshake or equivalent)\n"
    "4. The position must be at the director, assistant director, or associate director level within career services\n"
    "5. The position must require or prefer candidates to hold a master's degree\n"
    "6. The position must require a minimum of 3 to 5 years of relevant experience in career services, higher education, "
    "student affairs, or a related field\n"
    "7. The university's career services department must host regular career fairs or employer engagement events\n"
    "8. The university must be located in a U.S. state that borders either the Atlantic Ocean or one of the Great Lakes\n\n"
    "For your answer, provide:\n"
    "- The name of the university\n"
    "- The specific position title\n"
    "- A reference URL to the university's career services webpage showing the services and programs\n"
    "- A reference URL to the position posting or job description"
)

# Valid states for geographic criterion
ATLANTIC_STATES = [
    "Maine", "New Hampshire", "Massachusetts", "Rhode Island", "Connecticut",
    "New York", "New Jersey", "Delaware", "Maryland", "Virginia",
    "North Carolina", "South Carolina", "Georgia", "Florida",
]
GREAT_LAKES_STATES = [
    "Minnesota", "Wisconsin", "Illinois", "Indiana", "Michigan",
    "Ohio", "Pennsylvania", "New York",
]


def _states_list_str() -> str:
    return (
        "Atlantic states: " + ", ".join(ATLANTIC_STATES) + "; "
        "Great Lakes states: " + ", ".join(GREAT_LAKES_STATES)
    )


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CareerPositionInfo(BaseModel):
    """Information required from the agent's answer."""
    university_name: Optional[str] = None
    position_title: Optional[str] = None
    career_services_url: Optional[str] = None
    job_posting_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_position_info() -> str:
    return (
        "Extract the following fields exactly as they appear from the provided answer:\n"
        "1. university_name: The name of the university.\n"
        "2. position_title: The exact position/title identified (e.g., Director of Career Services, Associate Director, Assistant Director).\n"
        "3. career_services_url: A URL to the university's career services webpage that shows services/programs.\n"
        "4. job_posting_url: A URL to the position posting or job description.\n\n"
        "Rules:\n"
        "- If a field is missing, set it to null.\n"
        "- Extract URLs exactly as provided (markdown links are allowed; extract the actual URL). "
        "Prepend http:// if protocol is missing.\n"
        "- Do not infer or fabricate any information."
    )


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_criteria(
    evaluator: Evaluator,
    parent_node,
    info: CareerPositionInfo,
) -> None:
    """
    Build and execute verification for each rubric criterion using the provided URLs.
    """

    # Create the critical main node as per rubric
    main_node = evaluator.add_parallel(
        id="University_Career_Services_Position_Criteria",
        desc="Evaluate whether the identified university career services position meets all specified requirements",
        parent=parent_node,
        critical=True,
    )

    # Convenience variables
    uni = info.university_name or "the university"
    pos_title = info.position_title or "the identified position"
    cs_url = info.career_services_url
    job_url = info.job_posting_url

    # 1) Has Career Services Center / equivalent
    leaf1 = evaluator.add_leaf(
        id="Has_Career_Services_Center",
        desc="The university has a Center for Career and Professional Success or equivalent career services department",
        parent=main_node,
        critical=True,
    )
    claim1 = (
        "This webpage is the university's official career services page or equivalent center "
        "(e.g., Center for Career and Professional Success, Career Services, Career Center, "
        "Career & Professional Development)."
    )
    await evaluator.verify(
        claim=claim1,
        node=leaf1,
        sources=cs_url,
        additional_instruction=(
            "Confirm that the page represents the university's career services department "
            "(allow reasonable naming variants such as Career Services, Career Center, "
            "Center for Career and Professional Success, Career & Professional Development). "
            "If the URL is missing or not a career services page, mark as not supported."
        ),
    )

    # 2) Offers structured career readiness program
    leaf2 = evaluator.add_leaf(
        id="Offers_Career_Readiness_Program",
        desc="The university offers a structured career readiness program for students (such as digital badge programs, multi-step career development programs, or similar formalized initiatives)",
        parent=main_node,
        critical=True,
    )
    claim2 = (
        "The career services page shows that the university offers a structured career readiness program "
        "for students (e.g., digital badges, competency badges, micro-credentials, certificates, multi-step "
        "development programs, roadmaps, passports)."
    )
    await evaluator.verify(
        claim=claim2,
        node=leaf2,
        sources=cs_url,
        additional_instruction=(
            "Look for explicit mention of a formal/structured program such as Career Readiness Badges, "
            "Digital Badges, Competency badges, Career Readiness Certificate, Micro-credential in Career Readiness, "
            "Passport, Roadmap, or a clearly defined multi-step program. "
            "General advising or one-off workshops alone are insufficient."
        ),
    )

    # 3) Uses major career platform
    leaf3 = evaluator.add_leaf(
        id="Uses_Major_Career_Platform",
        desc="The career services department provides services through a major career platform (such as Handshake or equivalent system)",
        parent=main_node,
        critical=True,
    )
    claim3 = (
        "The career services page indicates that students use a major career platform "
        "to access services, postings, or events."
    )
    await evaluator.verify(
        claim=claim3,
        node=leaf3,
        sources=cs_url,
        additional_instruction=(
            "Accept platforms such as Handshake, Symplicity, 12Twenty, GradLeaders, Purple Briefcase, JobTeaser, "
            "CareerHub/Symplicity. Identify explicit references to one of these platforms on the page."
        ),
    )

    # 4) Position level appropriate (director/assistant/associate director) within career services
    leaf4 = evaluator.add_leaf(
        id="Position_Level_Appropriate",
        desc="The position is at the director, assistant director, or associate director level in career services",
        parent=main_node,
        critical=True,
    )
    claim4 = (
        f"The job posting is for '{pos_title}', which is a Director, Assistant Director, or Associate Director "
        "role within the career services department (or equivalent unit name)."
    )
    await evaluator.verify(
        claim=claim4,
        node=leaf4,
        sources=job_url,
        additional_instruction=(
            "Check the job title and role context. Accept titles containing 'Director', 'Associate Director', "
            "'Assistant Director' that clearly pertain to career services or an equivalent unit "
            "(e.g., Career Services, Career Center, Center for Career & Professional Development, Career & Professional Success)."
        ),
    )

    # 5) Master's degree required or preferred
    leaf5 = evaluator.add_leaf(
        id="Masters_Degree_Required_Preferred",
        desc="The position requires or prefers a master's degree as stated in the job posting or position description",
        parent=main_node,
        critical=True,
    )
    claim5 = "The job posting states that a master's degree is required or preferred."
    await evaluator.verify(
        claim=claim5,
        node=leaf5,
        sources=job_url,
        additional_instruction=(
            "Look for language like 'Master's degree required', 'Master's degree preferred', or equivalent "
            "phrases ('graduate degree', MA/MS/MBA). If the posting mentions only a bachelor's, and no master's, "
            "mark as not supported."
        ),
    )

    # 6) Experience requirement: minimum of 3 to 5 years relevant experience
    leaf6 = evaluator.add_leaf(
        id="Experience_Requirement_Met",
        desc="The position requires a minimum of 3 to 5 years of relevant experience in career services, higher education, student affairs, or a related field",
        parent=main_node,
        critical=True,
    )
    claim6 = (
        "The job posting requires a minimum of 3 to 5 years of relevant experience in career services, "
        "higher education, student affairs, or a related field."
    )
    await evaluator.verify(
        claim=claim6,
        node=leaf6,
        sources=job_url,
        additional_instruction=(
            "Accept explicit ranges like '3–5 years', or minimum statements such as 'at least 3 years', 'minimum 3 years', "
            "or '5 years' if clearly within the expected minimum range. The experience must be relevant to "
            "career services/higher education/student affairs or closely related domains."
        ),
    )

    # 7) Hosts regular career fairs or employer engagement events
    leaf7 = evaluator.add_leaf(
        id="Hosts_Career_Events",
        desc="The university career services department hosts regular career fairs or employer engagement events",
        parent=main_node,
        critical=True,
    )
    claim7 = (
        "The career services department hosts regular career fairs or employer engagement events."
    )
    await evaluator.verify(
        claim=claim7,
        node=leaf7,
        sources=cs_url,
        additional_instruction=(
            "Look for 'career fair(s)', 'job fair(s)', 'internship fair(s)', 'industry career fairs', "
            "'employer info sessions', 'employer networking', 'meet-the-employer' events. "
            "A recurring or regular cadence should be implied by the page."
        ),
    )

    # 8) Geographic location: University in a state bordering Atlantic or Great Lakes
    leaf8 = evaluator.add_leaf(
        id="Geographic_Location",
        desc="The university is located in a U.S. state that borders either the Atlantic Ocean or one of the Great Lakes",
        parent=main_node,
        critical=True,
    )
    claim8 = (
        f"The university {uni} is located in a U.S. state that borders either the Atlantic Ocean "
        "or one of the Great Lakes."
    )
    geo_sources: List[str] = []
    if cs_url:
        geo_sources.append(cs_url)
    if job_url:
        geo_sources.append(job_url)
    await evaluator.verify(
        claim=claim8,
        node=leaf8,
        sources=geo_sources if geo_sources else None,
        additional_instruction=(
            "First identify the university's state from the provided page(s). Then determine if the state is in one of "
            "the following lists. " + _states_list_str() + ". "
            "If the page(s) do not reveal the state, mark as not supported."
        ),
    )

    # Helpful meta info
    evaluator.add_custom_info(
        info={
            "university_name": info.university_name,
            "position_title": info.position_title,
            "career_services_url": info.career_services_url,
            "job_posting_url": info.job_posting_url,
            "valid_states_atlantic": ATLANTIC_STATES,
            "valid_states_great_lakes": GREAT_LAKES_STATES,
        },
        info_type="extraction_summary",
        info_name="extracted_position_info",
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the university career services position criteria task.
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

    # Extract required fields from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_position_info(),
        template_class=CareerPositionInfo,
        extraction_name="position_info",
    )

    # Build and verify criteria
    await verify_criteria(evaluator, root, extracted_info)

    # Return structured result
    return evaluator.get_summary()