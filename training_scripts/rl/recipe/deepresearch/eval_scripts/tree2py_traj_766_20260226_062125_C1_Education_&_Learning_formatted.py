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
TASK_ID = "md_school_district_enrollment_2024_2025"
TASK_DESCRIPTION = (
    "Identify a public school district in Maryland that has a total student enrollment "
    "between 150,000 and 170,000 students for the 2024-2025 school year. Provide the name "
    "of the district, the exact enrollment number, and a reference URL from an official "
    "district source that verifies this information."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DistrictInfo(BaseModel):
    district_name: Optional[str] = None
    enrollment: Optional[str] = None
    academic_year: Optional[str] = None
    official_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_district_info() -> str:
    return """
    From the answer, extract the following fields for a single identified public school district:

    - district_name: The full name of the public school district (e.g., "Montgomery County Public Schools").
    - enrollment: The exact total student enrollment number cited in the answer (as written, keep commas or formatting).
                  If multiple numbers appear, choose the one the answer explicitly claims as the total enrollment.
    - academic_year: The school year associated with the enrollment data as written in the answer (e.g., "2024-2025", "SY 2024-25", "2024–25").
    - official_urls: A list of URLs included in the answer that are intended as official district sources verifying the enrollment.
                     Only include URLs that are likely the district's own website or subdomains (e.g., domains like *.k12.md.us, *.org clearly belonging to the district).
                     Do not include third-party media, Wikipedia, data aggregators, or state websites unless the answer explicitly frames them as the district’s official source.

    Return null for any field not present in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_district(
    evaluator: Evaluator,
    extracted: DistrictInfo,
) -> None:
    """
    Build the verification tree according to the rubric and run the checks.
    """
    # Create the parent node that represents the overall identification verification
    parent_node = evaluator.add_parallel(
        id="School_District_Identification",
        desc="Verify that the identified school district meets all specified criteria: location in Maryland, enrollment range of 150,000-170,000 students, data from the 2024-2025 school year, and includes an official source reference.",
        parent=evaluator.root,
        critical=False,
    )

    name = extracted.district_name or ""
    enrollment = extracted.enrollment or ""
    academic_year = extracted.academic_year or ""
    urls = extracted.official_urls or []

    # 1) Maryland_Location (Critical)
    node_loc = evaluator.add_leaf(
        id="Maryland_Location",
        desc="Verify that the school district is a public school district located in Maryland.",
        parent=parent_node,
        critical=True,
    )
    if urls:
        claim_loc = f"'{name}' is a public school district located in the U.S. state of Maryland."
        await evaluator.verify(
            claim=claim_loc,
            node=node_loc,
            sources=urls,
            additional_instruction=(
                "Use the provided webpage(s) to confirm the district is in Maryland (MD). "
                "Look for mentions like 'Maryland', 'MD', or references to Maryland counties/cities. "
                "Do not rely on your own memory; rely on the page content and screenshot."
            ),
        )
    else:
        node_loc.score = 0.0
        node_loc.status = "failed"

    # 2) Enrollment_Range (Critical) - Logical/arithmetical check from the provided number
    node_range = evaluator.add_leaf(
        id="Enrollment_Range",
        desc="Verify that the school district's total student enrollment is between 150,000 and 170,000 students (inclusive).",
        parent=parent_node,
        critical=True,
    )
    claim_range = (
        f"The enrollment number '{enrollment}' is between 150,000 and 170,000 inclusive. "
        "Interpret the number by stripping non-digit characters (e.g., commas, spaces). "
        "If the string cannot be reasonably parsed into a number, or if it represents a range outside 150,000–170,000, judge this as Incorrect."
    )
    await evaluator.verify(
        claim=claim_range,
        node=node_range,
        additional_instruction=(
            "If the number includes commas (e.g., 160,123) or is written like '160k' or 'about 160,000', "
            "interpret it as approximately that integer. If approximation is used, judge whether the implied integer "
            "would fall within [150000, 170000]."
        ),
    )

    # 3) Academic_Year_2024_2025 (Critical) - Must be supported by official page(s)
    node_year = evaluator.add_leaf(
        id="Academic_Year_2024_2025",
        desc="Verify that the enrollment data provided corresponds to the 2024-2025 school year.",
        parent=parent_node,
        critical=True,
    )
    if urls:
        claim_year = (
            f"The enrollment figure for {name} pertains to the 2024-2025 school year. "
            "Accepted textual variants include '2024–2025', '2024-25', 'SY 2024-25', or 'School Year 2024-2025'."
        )
        await evaluator.verify(
            claim=claim_year,
            node=node_year,
            sources=urls,
            additional_instruction=(
                "Confirm that the enrollment information on the provided page(s) explicitly ties the number to the "
                "2024–2025 school year (allow hyphen/en dash variants, and common abbreviations like 'SY 2024-25'). "
                "If the year is missing or clearly a different year, return Incorrect."
            ),
        )
    else:
        node_year.score = 0.0
        node_year.status = "failed"

    # 4) Official_Source_Reference (Critical)
    node_source = evaluator.add_leaf(
        id="Official_Source_Reference",
        desc="Verify that the answer includes a reference URL from an official district source that verifies the enrollment information.",
        parent=parent_node,
        critical=True,
    )
    if urls:
        claim_source = (
            f"At least one of the provided URLs is an official district source for {name} "
            f"(i.e., the district’s own website or subdomain) and it states the total student enrollment is {enrollment}."
        )
        await evaluator.verify(
            claim=claim_source,
            node=node_source,
            sources=urls,
            additional_instruction=(
                "Treat a URL as an official district source if it is clearly the district’s own domain or subdomain "
                "(e.g., *.k12.md.us, *.org that is the district's official site). "
                "Do not count third-party media, Wikipedia, or general government/state portals as 'official district source'. "
                "The page content must explicitly state the total student enrollment and it should match the figure from the answer "
                "(allow minor formatting differences like commas)."
            ),
        )
    else:
        node_source.score = 0.0
        node_source.status = "failed"


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
    # Initialize evaluator with a parallel root
    evaluator = Evaluator()
    evaluator.initialize(
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_district_info(),
        template_class=DistrictInfo,
        extraction_name="district_info",
    )

    # Build and run verification according to the rubric
    await build_and_verify_district(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()