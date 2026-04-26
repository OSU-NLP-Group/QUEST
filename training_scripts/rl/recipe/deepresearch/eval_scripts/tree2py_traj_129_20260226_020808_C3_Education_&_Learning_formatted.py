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
TASK_ID = "ahsaa_6a_2026_2028_third_highest"
TASK_DESCRIPTION = (
    "In the Alabama High School Athletic Association's (AHSAA) 2026-2028 reclassification, "
    "Class 6A consists of the 32 largest public high schools in the state, ranked by Average Daily Enrollment. "
    "Identify the school with the third-highest enrollment in Class 6A. Then provide: "
    "(1) that school's exact Average Daily Enrollment number, "
    "(2) the football region to which it is assigned for the 2026-2028 cycle, "
    "(3) the name of one other school assigned to the same region, and "
    "(4) that other school's Average Daily Enrollment number. "
    "Include reference URLs to official AHSAA sources or reliable news sources that document the region assignments."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TargetSchoolInfo(BaseModel):
    """Info for the identified third-highest enrollment school."""
    name: Optional[str] = None
    enrollment: Optional[str] = None  # Keep as string to allow variants like "1,234" or "1234"
    region: Optional[str] = None  # e.g., "Region 4", "6A Region 4"
    # URLs that support specific aspects of the claim
    identification_urls: List[str] = Field(default_factory=list)  # Ranking / classification sources
    enrollment_urls: List[str] = Field(default_factory=list)      # Sources documenting the ADE value
    region_urls: List[str] = Field(default_factory=list)          # AHSAA or reliable news documenting region assignments


class CoRegionSchoolInfo(BaseModel):
    """Info for another school in the same region."""
    name: Optional[str] = None
    enrollment: Optional[str] = None
    urls: List[str] = Field(default_factory=list)  # Any sources that support the co-school info (optional)


class AHSAA6AExtraction(BaseModel):
    """Top-level extraction structure."""
    target: Optional[TargetSchoolInfo] = None
    co_school: Optional[CoRegionSchoolInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_6a_info() -> str:
    return """
    Extract the structured information requested for AHSAA Class 6A (2026–2028) from the provided answer.

    Required fields:
    - target.name: The name of the school identified as having the third-highest Average Daily Enrollment (ADE) in Class 6A for 2026–2028.
    - target.enrollment: The exact ADE number for the identified school (verbatim from the answer).
    - target.region: The football region assignment for the 2026–2028 cycle (e.g., "Region 4", "6A Region 4"). Include the region label as written.
    - target.identification_urls: All URLs cited that support the identification (ranking/classification list) showing the school is third-highest in Class 6A by ADE. Include only valid, complete URLs mentioned in the answer.
    - target.enrollment_urls: All URLs cited that support the ADE number for the identified school. Include only valid, complete URLs mentioned in the answer.
    - target.region_urls: All URLs cited that document the football region assignments for 2026–2028 (prefer official AHSAA pages or credible news sources). Include only valid, complete URLs mentioned in the answer.

    Co-region school fields:
    - co_school.name: The name of one other school assigned to the same region.
    - co_school.enrollment: That other school's ADE number (verbatim).
    - co_school.urls: Any URLs mentioned that support the co-school's region assignment or ADE. Include only valid, complete URLs mentioned in the answer.

    Rules:
    - Extract information exactly as it appears in the answer; do not invent or infer missing details.
    - If any field is missing, set it to null (or empty array for URL lists).
    - For URLs, extract only actual URLs present in the answer (including markdown links), and ensure they are valid (include protocol).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def combine_sources(*lists: List[str]) -> List[str]:
    """Combine multiple URL lists, deduplicate, and filter out empties."""
    seen = set()
    result: List[str] = []
    for lst in lists:
        for url in lst or []:
            u = (url or "").strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                result.append(u)
    return result


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def build_school_identification(
    evaluator: Evaluator,
    parent_node,
    extraction: AHSAA6AExtraction,
) -> Any:
    """
    Build the 'School_Identification' branch.
    Verifies:
      - Target school name provided
      - Identification URLs provided
      - Claim: target is the 3rd-highest ADE in Class 6A (2026–2028), supported by URLs
    """
    target = extraction.target or TargetSchoolInfo()

    node = evaluator.add_sequential(
        id="School_Identification",
        desc="Correctly identify the school with the third-highest Average Daily Enrollment in AHSAA Class 6A for 2026-2028",
        parent=parent_node,
        critical=True  # Critical chain start
    )

    # Existence: target school name present
    evaluator.add_custom_node(
        result=bool(target.name and target.name.strip()),
        id="target_school_provided",
        desc="Target school name is provided (third-highest in Class 6A)",
        parent=node,
        critical=True
    )

    # Existence: identification URLs present
    evaluator.add_custom_node(
        result=bool(target.identification_urls),
        id="identification_urls_provided",
        desc="Identification source URL(s) provided for ranking/classification",
        parent=node,
        critical=True
    )

    # Verification: third-highest claim supported by URLs
    third_claim_leaf = evaluator.add_leaf(
        id="third_highest_supported",
        desc="The school identified is the 3rd-highest ADE in Class 6A (2026–2028), supported by sources",
        parent=node,
        critical=True
    )
    claim = f"The school with the third-highest Average Daily Enrollment in AHSAA Class 6A for 2026–2028 is {target.name}."
    await evaluator.verify(
        claim=claim,
        node=third_claim_leaf,
        sources=target.identification_urls,
        additional_instruction=(
            "Use the provided page(s) to check the Class 6A ranking by Average Daily Enrollment (ADM/ADE). "
            "Confirm that the listed order shows the target school ranked exactly 3rd among 32 Class 6A public schools. "
            "Allow naming variants and formatting differences; focus on ranking by ADE."
        ),
    )

    return node


async def build_enrollment_verification(
    evaluator: Evaluator,
    parent_node,
    extraction: AHSAA6AExtraction,
) -> Any:
    """
    Build the 'Enrollment_Verification' branch.
    Verifies:
      - Target enrollment value exists
      - At least one enrollment-supporting source exists (or classification source usable)
      - Claim: target ADE value is correct, supported by sources
    """
    target = extraction.target or TargetSchoolInfo()

    node = evaluator.add_sequential(
        id="Enrollment_Verification",
        desc="Provide the correct Average Daily Enrollment number for the identified school",
        parent=parent_node,
        critical=True
    )

    # Existence: enrollment provided
    evaluator.add_custom_node(
        result=bool(target.enrollment and target.enrollment.strip()),
        id="target_enrollment_provided",
        desc="Target school's Average Daily Enrollment number is provided",
        parent=node,
        critical=True
    )

    # Existence: there is at least one source to support enrollment (prefer enrollment_urls, fallback identification_urls)
    enrollment_sources = target.enrollment_urls if target.enrollment_urls else target.identification_urls
    evaluator.add_custom_node(
        result=bool(enrollment_sources),
        id="enrollment_sources_available",
        desc="Enrollment-supporting source URL(s) are available (direct enrollment sources or classification sources)",
        parent=node,
        critical=True
    )

    # Verification: ADE value supported
    ade_claim_leaf = evaluator.add_leaf(
        id="target_enrollment_supported",
        desc="The target school's ADE is correctly cited and supported by sources",
        parent=node,
        critical=True
    )
    claim = f"The Average Daily Enrollment for {target.name} is {target.enrollment}."
    await evaluator.verify(
        claim=claim,
        node=ade_claim_leaf,
        sources=enrollment_sources,
        additional_instruction=(
            "Verify the numeric ADE value for the specified 2026–2028 classification cycle. "
            "Allow minor formatting differences (commas, spacing) or reasonable rounding, but the value should match the page's stated number."
        ),
    )

    return node


async def build_region_assignment_with_references(
    evaluator: Evaluator,
    parent_node,
    extraction: AHSAA6AExtraction,
) -> Any:
    """
    Build the 'Region_Assignment_With_References' branch.
    Verifies:
      - Region label present
      - Region assignment reference URL(s) present
      - Claim: target is assigned to the stated region for 2026–2028, supported by official AHSAA or reliable news sources
    """
    target = extraction.target or TargetSchoolInfo()

    node = evaluator.add_sequential(
        id="Region_Assignment_With_References",
        desc="Correctly identify the football region assignment for the school in 2026–2028 and provide valid reference URL(s)",
        parent=parent_node,
        critical=True
    )

    # Existence: region label provided
    evaluator.add_custom_node(
        result=bool(target.region and target.region.strip()),
        id="region_label_provided",
        desc="Region label provided (e.g., 'Region 4', '6A Region 4')",
        parent=node,
        critical=True
    )

    # Existence: region reference URLs provided
    evaluator.add_custom_node(
        result=bool(target.region_urls),
        id="region_reference_urls_provided",
        desc="Region assignment reference URL(s) provided (official AHSAA or reliable news)",
        parent=node,
        critical=True
    )

    # Verification: region assignment supported
    region_claim_leaf = evaluator.add_leaf(
        id="region_assignment_supported",
        desc="The school's 2026–2028 football region assignment is correctly cited and supported by sources",
        parent=node,
        critical=True
    )
    claim = f"For the 2026–2028 cycle, {target.name} is assigned to football {target.region} in AHSAA Class 6A."
    await evaluator.verify(
        claim=claim,
        node=region_claim_leaf,
        sources=target.region_urls,
        additional_instruction=(
            "Confirm the football region assignment for 2026–2028 on the provided page(s). "
            "Prefer official AHSAA sources; reliable local news outlets are acceptable if they clearly document region assignments. "
            "Allow minor formatting differences (e.g., 'Class 6A, Region 4' vs '6A Region 4')."
        ),
    )

    return node


async def build_co_region_school_information(
    evaluator: Evaluator,
    parent_node,
    extraction: AHSAA6AExtraction,
) -> Any:
    """
    Build the 'Co_Region_School_Information' branch (placed under root as non-critical to allow partial credit).
    Verifies:
      - Another school name in the same region is provided
      - That school is indeed in the same region (supported by region sources)
      - ADE for the co-region school is correctly cited (supported by available sources)
    """
    target = extraction.target or TargetSchoolInfo()
    co = extraction.co_school or CoRegionSchoolInfo()

    node = evaluator.add_parallel(
        id="Co_Region_School_Information",
        desc="Provide information about another school in the same region, including its name and enrollment",
        parent=parent_node,
        critical=False  # Non-critical for partial credit
    )

    # Existence: co-region school name provided
    evaluator.add_custom_node(
        result=bool(co.name and co.name.strip()),
        id="co_school_name_provided",
        desc="Co-region school name is provided",
        parent=node,
        critical=True
    )

    # Verification: co-region school is in the same region
    co_region_name_leaf = evaluator.add_leaf(
        id="Co_Region_School_Name",
        desc="Name another school that is assigned to the same region for 2026-2028",
        parent=node,
        critical=True
    )
    co_region_sources = combine_sources(target.region_urls, co.urls)
    co_region_claim = f"For the 2026–2028 cycle, {co.name} is assigned to football {target.region} in AHSAA Class 6A."
    await evaluator.verify(
        claim=co_region_claim,
        node=co_region_name_leaf,
        sources=co_region_sources,
        additional_instruction=(
            "Confirm the co-region school's assignment to the same football region as the target school for 2026–2028. "
            "Prefer official AHSAA sources; reliable news acceptable. Allow minor formatting differences."
        ),
    )

    # Existence: co-region school enrollment provided
    evaluator.add_custom_node(
        result=bool(co.enrollment and co.enrollment.strip()),
        id="co_school_enrollment_provided",
        desc="Co-region school's Average Daily Enrollment number is provided",
        parent=node,
        critical=True
    )

    # Verification: co-region school ADE supported
    co_enroll_leaf = evaluator.add_leaf(
        id="Co_Region_School_Enrollment",
        desc="Provide the correct Average Daily Enrollment number for the named co-region school",
        parent=node,
        critical=True
    )
    co_enroll_sources = combine_sources(co.urls, target.identification_urls, target.enrollment_urls)
    co_enroll_claim = f"The Average Daily Enrollment for {co.name} is {co.enrollment}."
    await evaluator.verify(
        claim=co_enroll_claim,
        node=co_enroll_leaf,
        sources=co_enroll_sources if co_enroll_sources else None,
        additional_instruction=(
            "Verify the ADE value for the co-region school (for the 2026–2028 classification period if shown). "
            "Allow minor formatting differences or reasonable rounding; value should match the page."
        ),
    )

    return node


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
    Evaluate an answer for the AHSAA Class 6A 2026–2028 third-highest enrollment task.
    """
    # Initialize evaluator (root is non-critical sequential to reflect overall task flow)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured info from the agent's answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_6a_info(),
        template_class=AHSAA6AExtraction,
        extraction_name="ahsaa_6a_extraction",
    )

    # Add ground truth contextual info (no fixed values; this is contextual metadata)
    evaluator.add_ground_truth({
        "cycle": "2026–2028",
        "classification": "AHSAA Class 6A (32 largest public high schools by ADE)",
        "requirements": [
            "Identify the 3rd-highest ADE school",
            "Provide school ADE number",
            "Provide football region assignment (2026–2028)",
            "Provide one other school in same region and its ADE",
            "Include region assignment references (AHSAA or reliable news)"
        ]
    }, gt_type="task_context")

    # Build verification tree following sequential flow
    school_ident_node = await build_school_identification(evaluator, root, extraction)
    enrollment_node = await build_enrollment_verification(evaluator, school_ident_node, extraction)
    region_node = await build_region_assignment_with_references(evaluator, enrollment_node, extraction)

    # Place co-region info as a sibling under root (after region assignment) to allow partial credit
    await build_co_region_school_information(evaluator, root, extraction)

    # Return structured evaluation summary
    return evaluator.get_summary()