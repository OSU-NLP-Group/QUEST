import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "beginner_data_analyst_certificate"
TASK_DESCRIPTION = """
I am interested in transitioning into a data analyst career but have no prior experience in the field. I need to find a professional certificate program that is suitable for complete beginners (requiring no degree or prior experience) and covers all the essential technical skills needed for entry-level data analyst positions, including SQL, data visualization tools, and a programming language used in data analytics.

Please identify one such professional certificate program and provide the following information:
1. The name of the certificate program and the organization offering it
2. Confirmation that it teaches SQL, at least one data visualization tool (such as Tableau or Power BI), and a programming language (R or Python)
3. The estimated duration to complete the program
4. The number of courses included in the program
5. A link to the official program webpage
"""


# ----------------------------- Data Models --------------------------------- #
class TechnicalSkills(BaseModel):
    sql: Optional[str] = None
    visualization_tools: List[str] = Field(default_factory=list)
    programming_languages: List[str] = Field(default_factory=list)


class ProgramExtraction(BaseModel):
    program_name: Optional[str] = None
    organization: Optional[str] = None
    official_url: Optional[str] = None
    duration: Optional[str] = None
    course_count: Optional[str] = None
    technical: TechnicalSkills = Field(default_factory=TechnicalSkills)


# ---------------------------- Extraction Prompt ---------------------------- #
def prompt_extract_program_info() -> str:
    return """
    Extract exactly one professional certificate program from the answer (choose the first if multiple are mentioned). Return the following fields:

    - program_name: The full name/title of the certificate program.
    - organization: The organization/company offering the program (e.g., Google, IBM, Microsoft, a university).
    - official_url: The URL to the official program webpage provided in the answer. If multiple URLs are given, choose the one that appears to be the official program page (e.g., the organization's domain or an official partner page like Coursera/edX that clearly represents the program).
    - duration: The estimated completion time or typical duration stated in the answer (keep as a string as written in the answer).
    - course_count: The number of courses included in the program (keep as a string as written, e.g., "8 courses" or "eight courses").
    - technical: 
        - sql: If the answer explicitly states that the program teaches SQL, set to "SQL"; otherwise null.
        - visualization_tools: A list of any data visualization tools explicitly mentioned (e.g., "Tableau", "Power BI", "Looker Studio", "Qlik", "Excel"). If none are mentioned, return an empty list.
        - programming_languages: A list of any programming languages explicitly mentioned that are taught (e.g., "R", "Python"). If none are mentioned, return an empty list.

    Rules:
    - Extract only what is explicitly in the answer. Do not infer or add information.
    - For official_url, extract the exact URL string found in the answer (plain or markdown). If none is found, set to null.
    - If a field is missing or not clearly stated, set it to null (or empty list for arrays).
    """


# ---------------------------- Helper Functions ----------------------------- #
def _first_tool(tools: List[str]) -> Optional[str]:
    return tools[0] if tools else None


def _first_lang(langs: List[str]) -> Optional[str]:
    # Prefer R or Python if present
    for preferred in ["Python", "R"]:
        for l in langs:
            if l.lower() == preferred.lower():
                return preferred
    return langs[0] if langs else None


# --------------------------- Verification Builder -------------------------- #
async def build_and_verify_certificate_tree(
    evaluator: Evaluator,
    parent_root,
    info: ProgramExtraction
) -> None:
    # Create the critical root for certificate program evaluation
    cert_root = evaluator.add_parallel(
        id="Certificate_Program",
        desc="Identifies one professional certificate program suitable for complete beginners and provides all required details",
        parent=parent_root,
        critical=True
    )

    official_url = info.official_url or None

    # Program Identification
    prog_id = evaluator.add_parallel(
        id="Program_Identification",
        desc="Provides the name of the certificate program and the organization offering it",
        parent=cert_root,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.program_name and info.program_name.strip()) and bool(info.organization and info.organization.strip()),
        id="program_identification_provided",
        desc="Program name and offering organization are provided in the answer",
        parent=prog_id,
        critical=True
    )
    pid_verify = evaluator.add_leaf(
        id="program_identification_supported",
        desc="Official page shows the certificate program name and offering organization",
        parent=prog_id,
        critical=True
    )
    prog_name = info.program_name or ""
    org_name = info.organization or ""
    await evaluator.verify(
        claim=f"This webpage is the official page for the program '{prog_name}' offered by '{org_name}', and both the program name and the organization are clearly shown.",
        node=pid_verify,
        sources=official_url,
        additional_instruction="Accept partner platforms (e.g., Coursera, edX, Udacity) as official if the offering organization is explicitly indicated on the page. Verify that both the program title and the organization appear."
    )

    # Beginner Friendly
    beginner_leaf = evaluator.add_leaf(
        id="Beginner_Friendly",
        desc="Program explicitly states no degree or prior experience is required to enroll",
        parent=cert_root,
        critical=True
    )
    await evaluator.verify(
        claim="The program explicitly states that no prior experience or degree is required to enroll (suitable for complete beginners).",
        node=beginner_leaf,
        sources=official_url,
        additional_instruction="Look for statements like 'No degree or prior experience required', 'No prerequisites', 'Open to beginners', or equivalent wording on the page."
    )

    # Technical Requirements
    tech_root = evaluator.add_parallel(
        id="Technical_Requirements",
        desc="Program teaches the required technical skills: SQL, at least one data visualization tool, and R or Python",
        parent=cert_root,
        critical=True
    )

    # SQL Training
    sql_leaf = evaluator.add_leaf(
        id="SQL_Training",
        desc="Program curriculum explicitly includes SQL",
        parent=tech_root,
        critical=True
    )
    await evaluator.verify(
        claim="The program curriculum includes training in SQL (Structured Query Language).",
        node=sql_leaf,
        sources=official_url,
        additional_instruction="Check course/module titles or descriptions for SQL. Accept synonyms like 'SQL for data analysis', 'writing queries', or references to databases with SQL."
    )

    # Visualization Tool
    viz_tool_mentioned = _first_tool(info.technical.visualization_tools)
    viz_desc = "Program teaches at least one data visualization tool (e.g., Tableau or Power BI)"
    viz_leaf = evaluator.add_leaf(
        id="Visualization_Tool",
        desc=viz_desc,
        parent=tech_root,
        critical=True
    )
    viz_claim = (
        f"The program teaches the data visualization tool '{viz_tool_mentioned}'."
        if viz_tool_mentioned else
        "The program teaches at least one recognized data visualization tool such as Tableau, Power BI, Looker Studio, Qlik, or advanced Excel visualization."
    )
    await evaluator.verify(
        claim=viz_claim,
        node=viz_leaf,
        sources=official_url,
        additional_instruction="Verify that at least one recognized visualization tool is part of the curriculum. Accept Tableau, Power BI, Looker Studio (Google Data Studio), Qlik, or advanced Excel visualization."
    )

    # Programming Language (R or Python)
    lang_mentioned = _first_lang(info.technical.programming_languages)
    lang_leaf = evaluator.add_leaf(
        id="Programming_Language",
        desc="Program teaches a data-analytics programming language (R or Python)",
        parent=tech_root,
        critical=True
    )
    lang_claim = (
        f"The program teaches the programming language '{lang_mentioned}' for data analytics."
        if lang_mentioned else
        "The program teaches at least one programming language used in data analytics, specifically R or Python."
    )
    await evaluator.verify(
        claim=lang_claim,
        node=lang_leaf,
        sources=official_url,
        additional_instruction="Confirm that either 'Python' or 'R' is included in the curriculum for data analysis."
    )

    # Program Information
    info_root = evaluator.add_parallel(
        id="Program_Information",
        desc="Provides required program details: estimated duration and number of courses",
        parent=cert_root,
        critical=True
    )

    # Duration
    dur_seq = evaluator.add_sequential(
        id="Duration_Specified",
        desc="States the estimated completion time/duration for the program",
        parent=info_root,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.duration and info.duration.strip()),
        id="duration_provided",
        desc="Duration is provided in the answer",
        parent=dur_seq,
        critical=True
    )
    dur_verify = evaluator.add_leaf(
        id="duration_supported",
        desc="Official page provides an estimated duration consistent with the answer",
        parent=dur_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official webpage indicates the program's estimated completion time is '{info.duration or ''}' (allowing minor phrasing or rounding).",
        node=dur_verify,
        sources=official_url,
        additional_instruction="Allow minor variations in phrasing or rounding (e.g., '4-6 months' vs '6 months'). Match the general timeframe stated in the answer."
    )

    # Course Count
    course_seq = evaluator.add_sequential(
        id="Course_Count_Specified",
        desc="Specifies the number of courses included in the program",
        parent=info_root,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.course_count and info.course_count.strip()),
        id="course_count_provided",
        desc="Course count is provided in the answer",
        parent=course_seq,
        critical=True
    )
    course_verify = evaluator.add_leaf(
        id="course_count_supported",
        desc="Official page shows the number of courses consistent with the answer",
        parent=course_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official webpage indicates the program includes '{info.course_count or ''}' (allowing minor wording differences like 'modules' vs 'courses' when clearly equivalent).",
        node=course_verify,
        sources=official_url,
        additional_instruction="Accept small wording variations (e.g., 'courses', 'modules', 'classes') when clearly referring to discrete course units for the certificate."
    )

    # Reference URL
    ref_seq = evaluator.add_sequential(
        id="Reference_URL",
        desc="Provides a link to the official program webpage",
        parent=cert_root,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(official_url and official_url.strip()),
        id="reference_url_provided",
        desc="Official program webpage URL is provided in the answer",
        parent=ref_seq,
        critical=True
    )
    ref_verify = evaluator.add_leaf(
        id="reference_url_official",
        desc="URL corresponds to the official program webpage",
        parent=ref_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"This URL is the official program webpage for '{prog_name}' offered by '{org_name}'.",
        node=ref_verify,
        sources=official_url,
        additional_instruction="The page should present the program (title, organization, and enrollment info). Accept partner platforms (Coursera, edX, Udacity) if they are the canonical page for the program and clearly indicate the offering organization."
    )


# --------------------------- Main Evaluation API --------------------------- #
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

    # Extract structured info from answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_program_info(),
        template_class=ProgramExtraction,
        extraction_name="program_info_extraction",
    )

    # Build verification tree and run checks
    await build_and_verify_certificate_tree(evaluator, root, extracted_info)

    # Return summary
    return evaluator.get_summary()