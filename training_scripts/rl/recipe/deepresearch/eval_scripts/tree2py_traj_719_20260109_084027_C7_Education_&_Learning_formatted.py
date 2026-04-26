import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "data_analytics_prof_cert_eval"
TASK_DESCRIPTION = (
    "Identify an online professional certificate program in Data Analytics that meets the following requirements: "
    "(1) offered by a major online learning platform (such as Coursera, edX, Udacity, LinkedIn Learning, or FutureLearn) "
    "or directly by a major technology company (such as Google, IBM, Microsoft, or Meta) or an accredited university; "
    "(2) 100% online with self-paced learning; "
    "(3) requires no prior degree or professional experience; "
    "(4) designed for entry-level learners; "
    "(5) includes hands-on projects or portfolio work; "
    "(6) can be completed in 6 months or less at approximately 10 hours per week; "
    "(7) costs less than $500 USD total; "
    "(8) includes career support resources (resume help, interview prep, or job search assistance); "
    "(9) available to US learners as of January 2026; "
    "(10) provides a professional certificate upon completion; "
    "(11) specifies expected learning outcomes or skills gained; "
    "(12) clearly indicates the number of courses or modules; and "
    "(13) provides information about ACE credit, university credit transfer, or degree stackability. "
    "For your identified program, provide the program name, provider, official program URL, monthly cost (if subscription-based) "
    "or total cost, typical completion time, number of courses/modules, and a brief description of the career support offered."
)


class ProgramExtraction(BaseModel):
    """Structured info extracted from the agent's answer for the identified program."""
    program_name: Optional[str] = None
    provider: Optional[str] = None
    platform: Optional[str] = None
    official_url: Optional[str] = None

    monthly_cost: Optional[str] = None
    total_cost: Optional[str] = None
    completion_time: Optional[str] = None
    num_courses_modules: Optional[str] = None

    career_support_summary: Optional[str] = None

    # Additional notes mentioned in the answer; used to contextualize verification but not strictly matched
    prerequisites_note: Optional[str] = None  # e.g., "No degree or experience required"
    entry_level_note: Optional[str] = None    # e.g., "Beginners/entry-level"
    projects_note: Optional[str] = None       # e.g., "Capstone or portfolio"
    outcomes_note: Optional[str] = None       # e.g., "Skills listed, outcomes specified"
    credit_info: Optional[str] = None         # e.g., "ACE credit recommended", "Transfer credit"
    online_format_note: Optional[str] = None  # e.g., "Self-paced", "100% online"
    us_availability_note: Optional[str] = None  # e.g., "Available in US"
    certificate_award_note: Optional[str] = None  # e.g., "Professional certificate upon completion"
    data_analytics_focus_note: Optional[str] = None  # e.g., "Focus on Data Analytics"


def prompt_extract_program() -> str:
    return """
    From the answer, extract a single identified professional certificate program in Data Analytics and its key details.
    Return a JSON object with the following fields (use null if missing):
    - program_name: the official program name as stated
    - provider: the institution/company providing the certificate (e.g., Google, IBM, University X)
    - platform: the platform hosting the program, if mentioned (e.g., Coursera, edX)
    - official_url: the official program page URL (prefer the primary certificate page; if multiple URLs are given, choose the official/primary program page)
    - monthly_cost: the monthly subscription price if applicable (string, as written, e.g., "$49/month")
    - total_cost: the total estimated cost if given (string, e.g., "$300 total")
    - completion_time: typical time to complete (string, e.g., "under 6 months", "3-6 months", or a specific duration)
    - num_courses_modules: the number of courses or modules (string, e.g., "8 courses")
    - career_support_summary: brief mention of the career support resources (string summary)
    - prerequisites_note: any note about prerequisites in the answer (string)
    - entry_level_note: any note indicating it's for beginners/entry-level (string)
    - projects_note: mention of hands-on projects, labs, capstone, or portfolio (string)
    - outcomes_note: mention of listed learning outcomes or skills (string)
    - credit_info: mention of ACE credit, credit transfer, or degree stackability (string)
    - online_format_note: mention of 100% online and self-paced (string)
    - us_availability_note: any statement about availability in the US (string)
    - certificate_award_note: mention of awarding a professional certificate or badge (string)
    - data_analytics_focus_note: confirmation it focuses on Data Analytics (string)

    IMPORTANT:
    - Extract only what is explicitly present in the answer. Do not invent or infer missing details.
    - For URLs, extract the actual URL string (including protocol). If multiple URLs are listed, pick the official program page for the certificate (not blog posts or third-party writeups).
    """


async def build_and_verify_requirements(
    evaluator: Evaluator,
    root_node,
    program: ProgramExtraction
) -> None:
    """
    Build leaf nodes for each requirement and perform verification using the official program URL.
    All checks are critical because the root is critical and requires all criteria to be satisfied.
    """

    # Gate: ensure we have a program name and an official URL to verify against.
    existence_node = evaluator.add_custom_node(
        result=bool(program.program_name) and bool(program.official_url),
        id="Program_Identified",
        desc="A single program is identified with an official program URL",
        parent=root_node,
        critical=True
    )

    # Convenience source
    source_url = program.official_url if program.official_url else None

    # Platform / Provider check
    node_platform = evaluator.add_leaf(
        id="Platform_Provider",
        desc="The program is offered by a major platform/company or an accredited university",
        parent=root_node,
        critical=True
    )
    claim_platform = (
        "This certificate is offered via a recognized major platform (Coursera, edX, Udacity, LinkedIn Learning, or FutureLearn), "
        "or directly by a major technology company (Google, IBM, Microsoft, Meta), or by an accredited university."
    )
    await evaluator.verify(
        claim=claim_platform,
        node=node_platform,
        sources=source_url,
        additional_instruction=(
            "Use the official program page to determine the provider/platform. "
            "If the page is hosted on coursera.org, edx.org, udacity.com, linkedin.com/learning, or futurelearn.com, mark supported. "
            "If branding clearly shows Google, IBM, Microsoft, or Meta as the issuing provider, mark supported. "
            "If it is an accredited university certificate (often on a .edu domain or clearly branded as a university certificate), mark supported."
        ),
    )

    # Online format (100% online & self-paced)
    node_online = evaluator.add_leaf(
        id="Online_Format",
        desc="The program is 100% online with self-paced learning",
        parent=root_node,
        critical=True
    )
    claim_online = "This program is delivered entirely online and allows self-paced learning with flexible deadlines."
    await evaluator.verify(
        claim=claim_online,
        node=node_online,
        sources=source_url,
        additional_instruction=(
            "Look for phrases like '100% online', 'online program', 'self-paced', 'learn at your own pace', or 'flexible deadlines'. "
            "Ignore optional live events; the core modality should be online and self-paced."
        ),
    )

    # Career field: Data Analytics focus
    node_field = evaluator.add_leaf(
        id="Career_Field",
        desc="The program focuses on Data Analytics",
        parent=root_node,
        critical=True
    )
    claim_field = "This certificate is focused on Data Analytics as a career field."
    await evaluator.verify(
        claim=claim_field,
        node=node_field,
        sources=source_url,
        additional_instruction=(
            "Confirm the page explicitly indicates a Data Analytics focus. Accept closely related wording like 'data analysis' "
            "or 'analytics' when clearly positioned as Data Analytics rather than general Data Science."
        ),
    )

    # Prerequisites: No prior degree or experience
    node_prereq = evaluator.add_leaf(
        id="Prerequisites",
        desc="No prior degree or professional experience required",
        parent=root_node,
        critical=True
    )
    claim_prereq = "The program requires no prior degree or professional experience to enroll."
    await evaluator.verify(
        claim=claim_prereq,
        node=node_prereq,
        sources=source_url,
        additional_instruction=(
            "Check the admissions or requirements section for statements like 'no prior experience required', 'no degree required', "
            "or similar phrasing indicating minimal prerequisites."
        ),
    )

    # Entry-level design
    node_entry = evaluator.add_leaf(
        id="Entry_Level",
        desc="Designed for entry-level or beginner learners",
        parent=root_node,
        critical=True
    )
    claim_entry = "The program is explicitly designed for entry-level or beginner learners seeking career preparation or transition."
    await evaluator.verify(
        claim=claim_entry,
        node=node_entry,
        sources=source_url,
        additional_instruction=(
            "Look for terms like 'beginner', 'entry-level', 'no experience', or statements that the certificate is for newcomers to the field."
        ),
    )

    # Practical projects / portfolio
    node_projects = evaluator.add_leaf(
        id="Practical_Projects",
        desc="Includes hands-on projects, labs, portfolio work, or a capstone",
        parent=root_node,
        critical=True
    )
    claim_projects = "The curriculum includes hands-on projects, labs, portfolio work, or a capstone project."
    await evaluator.verify(
        claim=claim_projects,
        node=node_projects,
        sources=source_url,
        additional_instruction=(
            "Verify mentions of 'hands-on projects', 'labs', 'portfolio', 'capstone', or 'practical assignments' as part of the certificate."
        ),
    )

    # Duration: 6 months or less at ~10 hours/week
    node_duration = evaluator.add_leaf(
        id="Duration",
        desc="Completion in 6 months or less at approximately 10 hours/week",
        parent=root_node,
        critical=True
    )
    claim_duration = "Typical completion time is 6 months or less when studying around 10 hours per week."
    await evaluator.verify(
        claim=claim_duration,
        node=node_duration,
        sources=source_url,
        additional_instruction=(
            "Check the program's estimated time-to-complete and weekly workload. The requirement is met if the page indicates 6 months or less "
            "and approximately 10 hours/week (or less). Minor variations are acceptable if clearly within that range."
        ),
    )

    # Cost: < $500 total
    node_cost = evaluator.add_leaf(
        id="Cost",
        desc="Total cost is less than $500 USD",
        parent=root_node,
        critical=True
    )
    claim_cost = (
        "The total program cost to complete the certificate is under $500 USD. "
        "For subscription-based pricing, use the listed monthly price and typical duration to estimate the total."
    )
    await evaluator.verify(
        claim=claim_cost,
        node=node_cost,
        sources=source_url,
        additional_instruction=(
            "If a monthly subscription price is shown (e.g., $39–$59/month) and typical duration is 6 months or less, "
            "multiply to estimate total. If the page indicates additional required fees that would exceed $500, mark unsupported."
        ),
    )

    # Career support resources
    node_career = evaluator.add_leaf(
        id="Career_Support",
        desc="Includes career support resources (resume, interview prep, job search assistance, or career services)",
        parent=root_node,
        critical=True
    )
    claim_career = "The program includes career support resources such as resume help, interview preparation, job search guidance, or access to career services."
    await evaluator.verify(
        claim=claim_career,
        node=node_career,
        sources=source_url,
        additional_instruction=(
            "Look for a career services section, job search resources, resume reviews, mock interviews, or similar resources explicitly included with the certificate."
        ),
    )

    # US availability as of Jan 2026
    node_us = evaluator.add_leaf(
        id="US_Availability",
        desc="Available to US learners as of January 2026",
        parent=root_node,
        critical=True
    )
    claim_us = "The certificate is available to learners in the United States as of January 2026."
    await evaluator.verify(
        claim=claim_us,
        node=node_us,
        sources=source_url,
        additional_instruction=(
            "Confirm that enrollment is open to learners in the US (no US-specific restrictions). "
            "If the page is globally accessible and does not list the US among restricted regions, treat it as available."
        ),
    )

    # Certificate award upon completion
    node_award = evaluator.add_leaf(
        id="Certificate_Award",
        desc="Provides a professional certificate or digital badge upon completion",
        parent=root_node,
        critical=True
    )
    claim_award = "Upon completion, learners receive a professional certificate, credential, or digital badge."
    await evaluator.verify(
        claim=claim_award,
        node=node_award,
        sources=source_url,
        additional_instruction=(
            "Verify explicit wording such as 'Professional Certificate', 'certificate upon completion', or 'digital badge awarded'."
        ),
    )

    # Learning outcomes / skills
    node_outcomes = evaluator.add_leaf(
        id="Learning_Outcomes",
        desc="Specifies learning outcomes or skills gained",
        parent=root_node,
        critical=True
    )
    claim_outcomes = "The program page lists expected learning outcomes, skills gained, or career benefits."
    await evaluator.verify(
        claim=claim_outcomes,
        node=node_outcomes,
        sources=source_url,
        additional_instruction=(
            "Look for a 'Skills you’ll gain' section, bullet lists of outcomes, competency statements, or similar explicit outcome descriptions."
        ),
    )

    # Course structure: number of courses/modules
    node_structure = evaluator.add_leaf(
        id="Course_Structure",
        desc="Indicates the number of courses or modules included",
        parent=root_node,
        critical=True
    )
    claim_structure = "The program page clearly indicates the number of courses or modules in the certificate."
    await evaluator.verify(
        claim=claim_structure,
        node=node_structure,
        sources=source_url,
        additional_instruction=(
            "Check for explicit counts such as '8 courses', 'X modules', or similar. Minor wording differences are acceptable."
        ),
    )

    # Credit / stackability info
    node_credit = evaluator.add_leaf(
        id="Credit_Stackability",
        desc="Provides info on ACE credit, university transfer credit, or degree stackability",
        parent=root_node,
        critical=True
    )
    claim_credit = "The program provides information about ACE credit recommendations, university credit transfer options, or stackability toward a degree."
    await evaluator.verify(
        claim=claim_credit,
        node=node_credit,
        sources=source_url,
        additional_instruction=(
            "Look for statements like 'ACE recommended credits', 'university credit transfer', 'transferable credits', 'stackable to degree', or similar."
        ),
    )


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
    Entry point: evaluate whether the identified Data Analytics certificate meets all specified requirements.
    Returns the evaluator summary dict with the verification tree and extracted info.
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

    # Root must be critical to enforce all checks are required
    root.critical = True

    # Extract the program info from the answer
    program_info = await evaluator.extract(
        prompt=prompt_extract_program(),
        template_class=ProgramExtraction,
        extraction_name="program_extraction",
    )

    # Record expected requirements in summary for clarity
    evaluator.add_custom_info(
        info={
            "requirements": [
                "Major platform/company/university provider",
                "100% online and self-paced",
                "No prior degree/experience required",
                "Entry-level design",
                "Hands-on projects/portfolio/capstone",
                "≤ 6 months at ~10 hours/week",
                "< $500 USD total cost",
                "Career support resources included",
                "Available to US learners (Jan 2026)",
                "Professional certificate upon completion",
                "Learning outcomes/skills specified",
                "Number of courses/modules indicated",
                "ACE credit / transfer credit / degree stackability info"
            ]
        },
        info_type="rubric_requirements",
    )

    # Build verification leaves and run checks
    await build_and_verify_requirements(evaluator, root, program_info)

    return evaluator.get_summary()