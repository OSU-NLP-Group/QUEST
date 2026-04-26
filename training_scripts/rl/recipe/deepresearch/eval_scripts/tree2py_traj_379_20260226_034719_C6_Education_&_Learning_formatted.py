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
TASK_ID = "tx_public_universities_pre1900_40k"
TASK_DESCRIPTION = """
Identify exactly three public universities in Texas that meet ALL of the following criteria:

1. The university must be a public (state-funded) institution located in Texas.

2. The university must have been established (founded/opened) before the year 1900.

3. The university must currently have a total enrollment of at least 40,000 students (including all campuses if part of a multi-campus system).

4. The university must be a flagship institution or main campus of its system (not a branch campus or satellite location).

For each of the three universities, provide:
- The full official name of the university
- The exact founding year
- The current total enrollment figure with the academic year specified (e.g., "Fall 2024")
- The official university website URL
- A URL reference for the founding year information (from an official university source or reliable historical database)
- A URL reference for the enrollment data (from an official university source or verifiable database)
- If applicable, the name of the university system it belongs to

All information must be verifiable through the provided URL references.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    name: Optional[str] = None
    official_website: Optional[str] = None

    founding_year: Optional[str] = None
    founding_source_urls: List[str] = Field(default_factory=list)

    enrollment_total: Optional[str] = None
    enrollment_year_label: Optional[str] = None
    enrollment_source_urls: List[str] = Field(default_factory=list)

    system_name: Optional[str] = None
    campus_designation: Optional[str] = None  # e.g., "flagship", "main campus", "primary campus"
    system_source_urls: List[str] = Field(default_factory=list)

    additional_source_urls: List[str] = Field(default_factory=list)  # any other references provided


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to three universities from the answer that the agent claims meet the stated criteria.
    For each university, extract the following fields exactly as presented in the answer. Return them in the same order as they appear in the answer, and include ONLY the first three entries if more are provided.

    For each university, extract:
    - name: The full official university name as stated.
    - official_website: The official university website URL (must be a valid URL if provided).
    - founding_year: The exact founding year stated.
    - founding_source_urls: All URLs cited specifically for founding/founded history confirmation.
    - enrollment_total: The current total enrollment figure stated (keep formatting like commas or ranges).
    - enrollment_year_label: The academic year or term label associated with the enrollment (e.g., "Fall 2024", "2023-24").
    - enrollment_source_urls: All URLs cited specifically for enrollment confirmation.
    - system_name: The name of the university system if given (e.g., "The University of Texas System").
    - campus_designation: Any explicit designation, e.g., "flagship", "main campus", "primary campus", stating it is not a branch or satellite.
    - system_source_urls: Any URLs cited for system information (if provided).
    - additional_source_urls: Any other URLs cited for this university beyond the founding and enrollment sources.

    Rules:
    - Extract URLs exactly as shown, including markdown links. If a markdown link is used, extract the actual URL.
    - If a required field is missing in the answer, set it to null (or empty list for URL lists).
    - Do not invent or infer any data not explicitly present in the answer.
    - If more than three universities are listed, include only the first three in the returned JSON.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        nu = u.strip()
        if nu and nu not in seen:
            seen.add(nu)
            out.append(nu)
    return out


def compile_sources_for_university(uni: UniversityItem) -> List[str]:
    """
    Compile a list of URLs to use for general verifications (location, public status, flagship).
    Prefer official website plus any provided references.
    """
    urls: List[str] = []
    if uni.official_website:
        urls.append(uni.official_website)
    urls.extend(uni.founding_source_urls or [])
    urls.extend(uni.enrollment_source_urls or [])
    urls.extend(uni.system_source_urls or [])
    urls.extend(uni.additional_source_urls or [])
    return _dedup_urls(urls)


def ordinal(n: int) -> str:
    return ["First", "Second", "Third"][n - 1] if 1 <= n <= 3 else f"#{n}"


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    index: int,
) -> None:
    """
    Build the verification subtree for one university and run checks.
    """
    # University container node (non-critical to allow partial credit across the three universities)
    uni_node = evaluator.add_parallel(
        id=f"university_{index}",
        desc=f"{ordinal(index)} qualifying Texas public university with complete information",
        parent=parent_node,
        critical=False,
    )

    # Name provided (critical existence)
    evaluator.add_custom_node(
        result=bool(uni.name and uni.name.strip()),
        id=f"university_{index}_university_name",
        desc="Full official name of the university is provided",
        parent=uni_node,
        critical=True,
    )

    # Official website provided (critical existence)
    evaluator.add_custom_node(
        result=bool(uni.official_website and uni.official_website.strip()),
        id=f"university_{index}_official_website",
        desc="Official university website URL is provided",
        parent=uni_node,
        critical=True,
    )

    # Founding verification (critical, sequential)
    founding_node = evaluator.add_sequential(
        id=f"university_{index}_founding_verification",
        desc="University was established before 1900",
        parent=uni_node,
        critical=True,
    )

    # Founding year provided (critical existence)
    founding_year_provided = evaluator.add_custom_node(
        result=bool(uni.founding_year and uni.founding_year.strip()),
        id=f"university_{index}_founding_year_provided",
        desc="Exact founding year is stated",
        parent=founding_node,
        critical=True,
    )

    # Founding year accuracy (critical verification via founding sources)
    founding_accuracy_leaf = evaluator.add_leaf(
        id=f"university_{index}_founding_year_accuracy",
        desc="Founding year is before 1900 and matches official records",
        parent=founding_node,
        critical=True,
    )
    founding_claim_name = uni.name or "the university"
    founding_claim_year = uni.founding_year or ""
    founding_claim = (
        f"The founding year of {founding_claim_name} is {founding_claim_year}, and this year is before 1900."
    )
    await evaluator.verify(
        claim=founding_claim,
        node=founding_accuracy_leaf,
        sources=uni.founding_source_urls if uni.founding_source_urls else None,
        additional_instruction=(
            "Verify that the page(s) explicitly state the founding/opening year equals the provided year, "
            "and confirm that the year is strictly before 1900. Allow minor formatting differences (e.g., commas)."
        ),
    )

    # Founding source URL presence (critical existence)
    evaluator.add_custom_node(
        result=bool(uni.founding_source_urls and len(uni.founding_source_urls) > 0),
        id=f"university_{index}_founding_source_url",
        desc="URL reference provided for founding information from official university source or reliable historical database",
        parent=founding_node,
        critical=True,
    )

    # Enrollment verification (critical, sequential)
    enrollment_node = evaluator.add_sequential(
        id=f"university_{index}_enrollment_verification",
        desc="Current total enrollment is at least 40,000 students",
        parent=uni_node,
        critical=True,
    )

    # Enrollment figure + year label provided (critical existence)
    enrollment_provided = evaluator.add_custom_node(
        result=bool(uni.enrollment_total and uni.enrollment_total.strip()) and bool(uni.enrollment_year_label and uni.enrollment_year_label.strip()),
        id=f"university_{index}_enrollment_figure_provided",
        desc="Current total enrollment number is stated with academic year specified",
        parent=enrollment_node,
        critical=True,
    )

    # Enrollment meets threshold (critical verification via enrollment sources)
    enrollment_meets_leaf = evaluator.add_leaf(
        id=f"university_{index}_enrollment_meets_threshold",
        desc="Enrollment figure is 40,000 or higher",
        parent=enrollment_node,
        critical=True,
    )
    enrollment_claim_name = uni.name or "the university"
    enrollment_total_str = uni.enrollment_total or ""
    enrollment_year_label = uni.enrollment_year_label or ""
    enrollment_claim = (
        f"The reported current total enrollment for {enrollment_claim_name} in {enrollment_year_label} is {enrollment_total_str}, "
        "which is at least 40,000 students."
    )
    await evaluator.verify(
        claim=enrollment_claim,
        node=enrollment_meets_leaf,
        sources=uni.enrollment_source_urls if uni.enrollment_source_urls else None,
        additional_instruction=(
            "Check the stated total enrollment equals or exceeds 40,000. Minor variations or rounding are acceptable, "
            "but the figure must be reasonably supported by the provided source(s)."
        ),
    )

    # Enrollment source URL presence (critical existence)
    evaluator.add_custom_node(
        result=bool(uni.enrollment_source_urls and len(uni.enrollment_source_urls) > 0),
        id=f"university_{index}_enrollment_source_url",
        desc="URL reference provided for enrollment data from official university source or verifiable database",
        parent=enrollment_node,
        critical=True,
    )

    # Public institution status (critical, parallel)
    public_status_node = evaluator.add_parallel(
        id=f"university_{index}_public_institution_status",
        desc="University is confirmed as a public institution in Texas",
        parent=uni_node,
        critical=True,
    )

    # Verify Texas location (critical)
    texas_location_leaf = evaluator.add_leaf(
        id=f"university_{index}_texas_location",
        desc="University is located in Texas",
        parent=public_status_node,
        critical=True,
    )

    # Verify Public status (critical)
    public_status_leaf = evaluator.add_leaf(
        id=f"university_{index}_public_status",
        desc="University is a public (state-funded) institution",
        parent=public_status_node,
        critical=True,
    )

    general_sources = compile_sources_for_university(uni)
    # Batch verify location and public status in parallel under the same parent
    await evaluator.batch_verify([
        (
            f"The university {uni.name or ''} is located in the U.S. state of Texas.",
            general_sources if general_sources else None,
            texas_location_leaf,
            "Look for indications such as 'Austin, Texas', 'College Station, Texas', 'Denton, Texas', or explicit mention of Texas. "
            "Rely on official or authoritative pages among the provided URLs."
        ),
        (
            f"The university {uni.name or ''} is a public (state-funded) institution.",
            general_sources if general_sources else None,
            public_status_leaf,
            "Look for phrases like 'public university', 'state-supported', 'public research university', or similar wording on official pages."
        ),
    ])

    # Flagship/main campus status (critical, parallel)
    flagship_node = evaluator.add_parallel(
        id=f"university_{index}_flagship_status",
        desc="University is a flagship or main campus (not a branch campus)",
        parent=uni_node,
        critical=True,  # critical criterion; all its children must be critical to satisfy framework constraint
    )

    # Verify campus designation (critical)
    campus_designation_leaf = evaluator.add_leaf(
        id=f"university_{index}_campus_designation",
        desc="University is identified as flagship, main campus, or primary institution of its system",
        parent=flagship_node,
        critical=True,
    )
    designation_text = uni.campus_designation or "flagship/main campus"
    if uni.system_name:
        flagship_claim = (
            f"{uni.name or 'The university'} is the {designation_text} of the {uni.system_name}, "
            "and is not a branch or satellite campus."
        )
    else:
        flagship_claim = (
            f"{uni.name or 'The university'} is the {designation_text} and is not a branch or satellite campus."
        )
    flagship_sources = general_sources
    await evaluator.verify(
        claim=flagship_claim,
        node=campus_designation_leaf,
        sources=flagship_sources if flagship_sources else None,
        additional_instruction=(
            "Confirm the institution is described as 'flagship', 'main campus', 'primary campus', or equivalent. "
            "Being explicitly 'not a branch or satellite campus' also suffices. Prefer system/official pages."
        ),
    )

    # System information presence (critical for framework consistency)
    evaluator.add_custom_node(
        result=bool(uni.system_name and uni.system_name.strip()),
        id=f"university_{index}_system_information",
        desc="If part of a university system, the system name is provided",
        parent=flagship_node,
        critical=True,
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
    Evaluate an answer for the Texas public universities task.
    """
    # Initialize evaluator (root set to non-critical to allow partial credit and satisfy critical-child constraint)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # three universities evaluated independently
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

    # Extract up to three universities with required fields from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Prepare exactly three items (pad if fewer)
    universities: List[UniversityItem] = list(extracted.universities[:3])
    while len(universities) < 3:
        universities.append(UniversityItem())

    # Build verification subtrees for each university
    for i, uni in enumerate(universities, start=1):
        await verify_university(evaluator, root, uni, i)

    # Return structured evaluation summary
    return evaluator.get_summary()