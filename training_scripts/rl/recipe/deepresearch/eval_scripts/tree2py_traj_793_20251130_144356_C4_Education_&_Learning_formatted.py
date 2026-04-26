import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# ------------------------------------------------------------
# Task constants
# ------------------------------------------------------------
TASK_ID = "ncaa_ohsaa_info_2025"
TASK_DESCRIPTION = (
    "I'm a high school junior football player in Ohio who wants to play NCAA Division I football in college. "
    "I need to understand what academic requirements I must meet to be eligible, and I'd also like to know details "
    "about where the Ohio state championships will be held this year in case my team makes it that far.\n\n"
    "Please provide the following information:\n\n"
    "For NCAA Division I Eligibility:\n"
    "1. How many total NCAA-approved core courses must I complete in high school?\n"
    "2. How many of these core courses must I complete before the start of my 7th semester (senior year), and how many of those must be in English, math, or science?\n"
    "3. What is the minimum core course GPA I need to maintain?\n"
    "4. Where do I need to register to get certified for NCAA eligibility?\n\n"
    "For the 2025 OHSAA Football State Championships:\n"
    "5. What is the name of the stadium where the games will be held?\n"
    "6. What is the complete address of this stadium?\n"
    "7. What is the seating capacity of this stadium?\n"
    "8. On what dates will the championship games take place?"
)

# Optional ground truth info (for reporting only; not directly used in scoring)
GROUND_TRUTH = {
    "ncaa": {
        "total_core_courses": "16",
        "early_core_courses_total": "10",
        "early_ems_courses": "7",
        "min_core_gpa": "2.3",
        "registration": "NCAA Eligibility Center (eligibilitycenter.org)"
    },
    "ohsaa_2025": {
        "stadium_name": "Tom Benson Hall of Fame Stadium",
        "stadium_address": "1835 Harrison Ave NW, Canton, OH 44708",
        "capacity_accepted_range": "20000-22400",
        "championship_dates": "December 4–6, 2025"
    }
}

# ------------------------------------------------------------
# Data Models
# ------------------------------------------------------------
class NCAAEligibility(BaseModel):
    total_core_courses: Optional[str] = None
    early_core_courses_total: Optional[str] = None
    early_ems_courses: Optional[str] = None
    min_core_gpa: Optional[str] = None
    registration_platform_name: Optional[str] = None
    registration_platform_url: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class OHSAAChampionships(BaseModel):
    stadium_name: Optional[str] = None
    stadium_address: Optional[str] = None
    stadium_capacity: Optional[str] = None
    championship_dates: Optional[str] = None
    venue_url: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class StudentAthletePackage(BaseModel):
    ncaa: Optional[NCAAEligibility] = None
    ohsaa: Optional[OHSAAChampionships] = None


# ------------------------------------------------------------
# Extraction Prompt
# ------------------------------------------------------------
def prompt_extract_student_package() -> str:
    return (
        "Extract the requested NCAA Division I eligibility details and the 2025 OHSAA football state championship venue/date details "
        "as they are presented in the answer. Return the following JSON fields:\n\n"
        "ncaa:\n"
        "- total_core_courses: the total number of NCAA-approved core courses the answer states are required for Division I (e.g., '16').\n"
        "- early_core_courses_total: the number of core courses the answer says must be completed before the start of the 7th semester (senior year) (e.g., '10').\n"
        "- early_ems_courses: how many of those early core courses must be in English, math, or science (e.g., '7').\n"
        "- min_core_gpa: the stated minimum core course GPA (e.g., '2.3').\n"
        "- registration_platform_name: where to register for NCAA eligibility (e.g., 'NCAA Eligibility Center').\n"
        "- registration_platform_url: the URL for the registration platform (e.g., 'https://www.eligibilitycenter.org/'). If absent, null.\n"
        "- sources: an array of all URLs explicitly provided in the answer that support any of these NCAA details. Include official NCAA pages or the Eligibility Center if present. If none, return an empty list.\n\n"
        "ohsaa:\n"
        "- stadium_name: the stadium name for the 2025 OHSAA football state championships (e.g., 'Tom Benson Hall of Fame Stadium').\n"
        "- stadium_address: the complete postal address for that stadium (e.g., '1835 Harrison Ave NW, Canton, OH 44708').\n"
        "- stadium_capacity: the seating capacity stated in the answer (string, e.g., '20,000' or '22,364').\n"
        "- championship_dates: the dates for the 2025 championship games (e.g., 'December 4–6, 2025').\n"
        "- venue_url: a URL specifically about the stadium or OHSAA championship information (if provided). If none, null.\n"
        "- sources: an array of all URLs explicitly provided in the answer that support any OHSAA stadium or date details (e.g., OHSAA.org event pages, stadium/Pro Football Hall of Fame pages). If none, return an empty list.\n\n"
        "General rules:\n"
        "- Extract only what is explicitly present in the answer. Do not invent or infer missing values.\n"
        "- For URLs, extract only valid, explicit URLs from the answer; include markdown link targets if used, and ensure they have protocol.\n"
        "- If a requested field is not present in the answer, set it to null.\n"
    )


# ------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------
def combine_sources(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if not url:
                continue
            u = url.strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                result.append(u)
    return result


# ------------------------------------------------------------
# Verification subtrees
# ------------------------------------------------------------
async def build_ncaa_subtree(evaluator: Evaluator, parent_node, ncaa: Optional[NCAAEligibility]) -> None:
    ncaa_node = evaluator.add_parallel(
        id="NCAA_Division_I_Eligibility",
        desc="All required NCAA Division I eligibility details are provided.",
        parent=parent_node,
        critical=True
    )

    # Compose sources list for NCAA checks
    ncaa_sources = combine_sources(
        ncaa.sources if ncaa else [],
        [ncaa.registration_platform_url] if (ncaa and ncaa.registration_platform_url) else None
    )

    # 1) Total core courses (expect 16)
    leaf_total = evaluator.add_leaf(
        id="NCAA_Total_Core_Courses",
        desc="States the total number of NCAA-approved core courses required (16).",
        parent=ncaa_node,
        critical=True
    )
    total_val = ncaa.total_core_courses if ncaa else None
    claim_total = f"The total number of NCAA-approved core courses required for NCAA Division I is {total_val}."
    await evaluator.verify(
        claim=claim_total,
        node=leaf_total,
        sources=ncaa_sources if ncaa_sources else None,
        additional_instruction=(
            "Judge whether the stated number matches official NCAA Division I initial-eligibility requirements. "
            "The expected correct value is 16. Allow numeric vs spelled-out formats (e.g., '16' vs 'sixteen')."
        )
    )

    # 2) Early core courses: 10 before 7th semester, 7 of those in English/math/science
    leaf_early = evaluator.add_leaf(
        id="NCAA_Early_Core_Courses",
        desc="States how many core courses must be completed before the 7th semester (10) and how many of those must be English/math/science (7).",
        parent=ncaa_node,
        critical=True
    )
    early_total = ncaa.early_core_courses_total if ncaa else None
    early_ems = ncaa.early_ems_courses if ncaa else None
    claim_early = (
        f"Before the start of the 7th semester (senior year), a student must have completed {early_total} core courses, "
        f"including {early_ems} in English, math, or science, according to NCAA Division I rules."
    )
    await evaluator.verify(
        claim=claim_early,
        node=leaf_early,
        sources=ncaa_sources if ncaa_sources else None,
        additional_instruction=(
            "Judge whether both numbers are correct per NCAA Division I initial-eligibility rules. "
            "Expected correct values are 10 total and 7 of those in English/math/science. "
            "Accept reasonable formatting variations (e.g., 'English/Math/Science' vs 'English, math or science')."
        )
    )

    # 3) Minimum core-course GPA (expect 2.3)
    leaf_gpa = evaluator.add_leaf(
        id="NCAA_Minimum_GPA",
        desc="States the minimum core course GPA required (2.3).",
        parent=ncaa_node,
        critical=True
    )
    gpa = ncaa.min_core_gpa if ncaa else None
    claim_gpa = f"The minimum core-course GPA required for NCAA Division I initial eligibility is {gpa}."
    await evaluator.verify(
        claim=claim_gpa,
        node=leaf_gpa,
        sources=ncaa_sources if ncaa_sources else None,
        additional_instruction=(
            "Verify the minimum core-course GPA requirement for NCAA Division I. The expected correct value is 2.3."
        )
    )

    # 4) Registration platform (Eligibility Center at eligibilitycenter.org)
    leaf_reg = evaluator.add_leaf(
        id="NCAA_Registration_Platform",
        desc="Identifies where the student must register for certification (NCAA Eligibility Center at eligibilitycenter.org).",
        parent=ncaa_node,
        critical=True
    )
    reg_name = (ncaa.registration_platform_name if ncaa else None) or "NCAA Eligibility Center"
    reg_url = ncaa.registration_platform_url if ncaa else None
    if reg_url:
        claim_reg = f"The official platform to register for NCAA eligibility certification is the {reg_name} at {reg_url}."
        reg_sources = combine_sources([reg_url], ncaa_sources)
    else:
        claim_reg = f"The official platform to register for NCAA eligibility certification is the {reg_name} at eligibilitycenter.org."
        reg_sources = ncaa_sources
    await evaluator.verify(
        claim=claim_reg,
        node=leaf_reg,
        sources=reg_sources if reg_sources else None,
        additional_instruction=(
            "Check that the identified registration platform is the NCAA Eligibility Center. "
            "Older term 'Clearinghouse' may appear but should refer to the NCAA Eligibility Center. "
            "If a URL is provided, it should be eligibilitycenter.org (or the official NCAA path to the Eligibility Center)."
        )
    )


async def build_ohsaa_subtree(evaluator: Evaluator, parent_node, ohsaa: Optional[OHSAAChampionships]) -> None:
    ohsaa_node = evaluator.add_parallel(
        id="OHSAA_2025_State_Championships",
        desc="All required 2025 OHSAA football state championship venue details and dates are provided.",
        parent=parent_node,
        critical=True
    )

    # Compose sources list for OHSAA checks
    ohsaa_sources = combine_sources(
        ohsaa.sources if ohsaa else [],
        [ohsaa.venue_url] if (ohsaa and ohsaa.venue_url) else None
    )

    # 1) Stadium name
    leaf_venue = evaluator.add_leaf(
        id="OHSAA_Championship_Venue",
        desc="Provides the stadium name (Tom Benson Hall of Fame Stadium).",
        parent=ohsaa_node,
        critical=True
    )
    stadium_name = ohsaa.stadium_name if ohsaa else None
    claim_venue = f"The 2025 OHSAA football state championships will be held at {stadium_name}."
    await evaluator.verify(
        claim=claim_venue,
        node=leaf_venue,
        sources=ohsaa_sources if ohsaa_sources else None,
        additional_instruction=(
            "Verify the championship venue name. Expected is 'Tom Benson Hall of Fame Stadium' in Canton, Ohio. "
            "Allow minor formatting variations (e.g., omission of 'the')."
        )
    )

    # 2) Stadium address
    leaf_address = evaluator.add_leaf(
        id="OHSAA_Venue_Address",
        desc="Provides the complete stadium address (1835 Harrison Ave NW, Canton, OH 44708).",
        parent=ohsaa_node,
        critical=True
    )
    address = ohsaa.stadium_address if ohsaa else None
    if stadium_name:
        claim_address = f"The complete address of {stadium_name} is {address}."
    else:
        claim_address = f"The complete stadium address is {address}."
    await evaluator.verify(
        claim=claim_address,
        node=leaf_address,
        sources=ohsaa_sources if ohsaa_sources else None,
        additional_instruction=(
            "Verify the venue's postal address. Accept standard abbreviation variations (e.g., Ave vs Avenue, NW vs Northwest), "
            "but the address should correspond to the correct stadium in Canton, OH 44708."
        )
    )

    # 3) Stadium capacity
    leaf_capacity = evaluator.add_leaf(
        id="OHSAA_Stadium_Capacity",
        desc="Provides the seating capacity (approximately 20,000; acceptable within 20,000–22,400 per constraints).",
        parent=ohsaa_node,
        critical=True
    )
    capacity_txt = ohsaa.stadium_capacity if ohsaa else None
    claim_capacity = (
        f"Tom Benson Hall of Fame Stadium has a seating capacity of {capacity_txt}. "
        f"Approximate values are acceptable if they fall within 20,000–22,400."
    )
    await evaluator.verify(
        claim=claim_capacity,
        node=leaf_capacity,
        sources=ohsaa_sources if ohsaa_sources else None,
        additional_instruction=(
            "Confirm the typical seating capacity reported for the venue. "
            "Treat approximate values as correct if they reasonably fall within 20,000–22,400. "
            "If the page shows a precise number like 22,364, then 'about 20,000' should still be considered acceptable."
        )
    )

    # 4) Championship dates
    leaf_dates = evaluator.add_leaf(
        id="OHSAA_Championship_Dates",
        desc="Provides the championship dates (December 4–6, 2025).",
        parent=ohsaa_node,
        critical=True
    )
    dates = ohsaa.championship_dates if ohsaa else None
    claim_dates = f"The 2025 OHSAA football state championship games will take place on {dates}."
    await evaluator.verify(
        claim=claim_dates,
        node=leaf_dates,
        sources=ohsaa_sources if ohsaa_sources else None,
        additional_instruction=(
            "Verify the championship dates. Expected range is December 4–6, 2025. "
            "Allow small format variations (e.g., 'Dec. 4-6, 2025', en dash vs hyphen)."
        )
    )


# ------------------------------------------------------------
# Main evaluation entry point
# ------------------------------------------------------------
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

    # Record ground truth information for transparency (not used directly in scoring)
    evaluator.add_ground_truth(GROUND_TRUTH, gt_type="reference_expectations")

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_student_package(),
        template_class=StudentAthletePackage,
        extraction_name="student_athlete_info"
    )

    # Build top-level critical package node
    package_node = evaluator.add_parallel(
        id="Student_Athlete_Information_Package",
        desc="Provide all required NCAA Division I eligibility requirements and 2025 OHSAA football state championship venue/date details.",
        parent=root,
        critical=True
    )

    # Build NCAA subtree
    await build_ncaa_subtree(evaluator, package_node, extracted.ncaa)

    # Build OHSAA subtree
    await build_ohsaa_subtree(evaluator, package_node, extracted.ohsaa)

    # Return evaluation summary
    return evaluator.get_summary()