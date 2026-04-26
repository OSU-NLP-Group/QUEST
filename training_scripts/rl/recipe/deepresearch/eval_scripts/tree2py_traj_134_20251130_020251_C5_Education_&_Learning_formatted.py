import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "al_caep_k6_gpa"
TASK_DESCRIPTION = """
Identify four Alabama public universities that offer CAEP-accredited undergraduate elementary education (K-6) teacher certification programs where the minimum GPA requirement for admission to the teacher education program is 2.75 or lower. For each university, provide: (1) the specific minimum GPA requirement for admission to the teacher education program, (2) the specific GPA categories to which this requirement applies (e.g., overall GPA, professional studies GPA, teaching field GPA), (3) the required standardized testing for admission to the teacher education program (such as Praxis Core or Alabama Educators Certification Assessment Program), and (4) a reference URL from the official university website that documents these admission requirements.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    """One university's admission requirement info as stated in the agent's answer."""
    university_name: Optional[str] = None
    min_gpa: Optional[str] = None
    gpa_categories: List[str] = Field(default_factory=list)
    testing: Optional[str] = None
    official_urls: List[str] = Field(default_factory=list)

    # Optional supportive fields if present in the answer (used for eligibility verification)
    program_k6_statement: Optional[str] = None  # e.g., "Elementary Education (K–6) certification"
    caep_accreditation_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    """List of universities extracted from the answer."""
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to four Alabama public universities as presented in the answer that the answer claims meet ALL of the following:
    – They offer an undergraduate elementary education (K–6) teacher certification program.
    – Their teacher education program is CAEP-accredited.
    – The minimum GPA requirement for admission to the teacher education program is 2.75 or lower.
    For each university mentioned in the answer, extract:
      1. university_name: The institution’s name exactly as written in the answer.
      2. min_gpa: The specific minimum GPA requirement value (e.g., "2.75", "2.5", "2.75 overall and 2.5 teaching field"). Return null if not explicitly stated.
      3. gpa_categories: An array listing which GPA categories the minimum applies to (e.g., ["overall/cumulative", "professional studies", "teaching field"]). Return empty array if not specified.
      4. testing: The required standardized testing for admission to the teacher education program (e.g., "Praxis Core", "Alabama Educators Certification Assessment Program (AECTP)", "ACT/SAT", or "none" if explicitly stated). Return null if not specified.
      5. official_urls: All official university website URLs in the answer that document the admission requirements for the teacher education program. Only include URLs explicitly present in the answer; prefer .edu domains or clearly official subdomains. Return an empty array if none are given.
      6. program_k6_statement: Any phrase in the answer that explicitly indicates the program is Elementary Education (K–6) leading to certification/licensure. Return null if not mentioned.
      7. caep_accreditation_urls: Any official URLs in the answer that explicitly state CAEP accreditation for the teacher education program. Return an empty array if none are given.
    Notes:
    – Do not invent or infer any information; only extract what is explicitly stated in the answer.
    – If the answer lists more than four universities, extract the first four as they appear.
    – If a field is missing for a university, set it to null (or empty array for list fields).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _first_n_universities(extraction: UniversitiesExtraction, n: int = 4) -> List[UniversityItem]:
    items = extraction.universities[:n]
    # pad to length n with empty items if fewer are provided
    while len(items) < n:
        items.append(UniversityItem())
    return items


def _join_categories(categories: List[str]) -> str:
    return ", ".join([c.strip() for c in categories if c and c.strip()]) if categories else ""


def _combined_sources(item: UniversityItem) -> List[str]:
    # Combine admission requirement URLs and accreditation URLs, deduplicate while preserving order
    seen = set()
    combined: List[str] = []
    for url in (item.official_urls + item.caep_accreditation_urls):
        if url and url not in seen:
            combined.append(url)
            seen.add(url)
    return combined


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_university_verification(
    evaluator: Evaluator,
    parent_node,
    item: UniversityItem,
    index: int,
) -> None:
    """
    Build the verification subtree for a single university (sequential node),
    including eligibility checks and required outputs checks.
    """
    # Map index to rubric-friendly identifiers and descriptions
    idx_to_label = {0: "First", 1: "Second", 2: "Third", 3: "Fourth"}
    label = idx_to_label.get(index, f"University_{index+1}")

    # Top-level sequential node for this university
    uni_node = evaluator.add_sequential(
        id=f"{label}_University",
        desc=f"{label} qualifying university with required documented details",
        parent=parent_node,
        critical=False  # allow partial credit across universities
    )

    # --------------------- Eligibility (parallel, critical) ---------------------
    elig_node = evaluator.add_parallel(
        id=f"Eligibility_{index+1}",
        desc="University meets all eligibility constraints",
        parent=uni_node,
        critical=True
    )

    # Public Alabama check
    public_leaf = evaluator.add_leaf(
        id=f"Public_Alabama_{index+1}",
        desc="Institution is a public university located in Alabama",
        parent=elig_node,
        critical=True
    )
    public_claim = (
        f"The institution '{item.university_name or 'the institution'}' is a public university located in Alabama."
    )
    await evaluator.verify(
        claim=public_claim,
        node=public_leaf,
        sources=item.official_urls or None,
        additional_instruction=(
            "Verify on the provided official university pages whether the institution is located in Alabama and is public "
            "(e.g., language such as 'public university', 'state university', 'part of a state system', etc.). "
            "If the page does not clearly indicate public status or the Alabama location, judge as not supported."
        ),
    )

    # Program K–6 check
    k6_leaf = evaluator.add_leaf(
        id=f"Program_K6_{index+1}",
        desc="Offers an undergraduate elementary education (K–6) teacher certification program",
        parent=elig_node,
        critical=True
    )
    k6_claim = (
        "The institution offers an undergraduate Elementary Education teacher certification program that covers grades K–6."
    )
    await evaluator.verify(
        claim=k6_claim,
        node=k6_leaf,
        sources=item.official_urls or None,
        additional_instruction=(
            "Check the page(s) for explicit mention of Elementary Education (K–6) or equivalent phrasing indicating K–6 "
            "coverage leading to initial teacher certification/licensure. Accept reasonable equivalents like 'K-6', "
            "'elementary education certification', etc., only if clearly tied to undergraduate teacher preparation."
        ),
    )

    # CAEP accreditation check
    caep_leaf = evaluator.add_leaf(
        id=f"CAEP_{index+1}",
        desc="Teacher education program is CAEP-accredited",
        parent=elig_node,
        critical=True
    )
    caep_claim = (
        "The teacher education program is accredited by CAEP (Council for the Accreditation of Educator Preparation)."
    )
    await evaluator.verify(
        claim=caep_claim,
        node=caep_leaf,
        sources=_combined_sources(item) or None,
        additional_instruction=(
            "Look for explicit text indicating CAEP accreditation (e.g., 'accredited by CAEP', 'Council for the Accreditation of Educator Preparation'). "
            "If multiple provided official URLs exist, any single URL that clearly states CAEP accreditation suffices."
        ),
    )

    # Minimum GPA ≤ 2.75 check
    gpa_thresh_leaf = evaluator.add_leaf(
        id=f"Min_GPA_Leq_275_{index+1}",
        desc="Minimum GPA requirement for admission to the teacher education program is stated and is 2.75 or lower",
        parent=elig_node,
        critical=True
    )
    gpa_thresh_claim = (
        f"The minimum GPA required for admission to the teacher education program is {item.min_gpa or 'not specified'}, "
        "and it is 2.75 or lower."
    )
    await evaluator.verify(
        claim=gpa_thresh_claim,
        node=gpa_thresh_leaf,
        sources=item.official_urls or None,
        additional_instruction=(
            "Confirm that the page explicitly states a minimum GPA threshold for admission to the teacher education program, "
            "and that the stated threshold is a value of 2.75 or lower. If the page states a minimum higher than 2.75 (e.g., 3.0), judge as not supported."
        ),
    )

    # ------------------- Required outputs (parallel, critical) -------------------
    outputs_node = evaluator.add_parallel(
        id=f"Required_Outputs_{index+1}",
        desc="All requested admission-requirement details are provided for this university",
        parent=uni_node,
        critical=True
    )

    # Min GPA value provided and supported
    gpa_value_leaf = evaluator.add_leaf(
        id=f"Min_GPA_Value_{index+1}",
        desc="Provides the specific minimum GPA requirement value for admission to the teacher education program",
        parent=outputs_node,
        critical=True
    )
    gpa_value_claim = (
        f"The minimum GPA requirement for admission to the teacher education program is: {item.min_gpa or 'not specified'}."
    )
    await evaluator.verify(
        claim=gpa_value_claim,
        node=gpa_value_leaf,
        sources=item.official_urls or None,
        additional_instruction=(
            "Verify that the exact minimum GPA value stated in the answer appears on the provided official university page(s) "
            "for teacher education admissions."
        ),
    )

    # GPA categories supported
    categories_leaf = evaluator.add_leaf(
        id=f"GPA_Categories_{index+1}",
        desc="Specifies which GPA category/categories the requirement applies to (e.g., overall/cumulative, professional studies, teaching field)",
        parent=outputs_node,
        critical=True
    )
    categories_text = _join_categories(item.gpa_categories) or "not specified"
    categories_claim = (
        f"The minimum GPA requirement applies to these GPA category/categories: {categories_text}."
    )
    await evaluator.verify(
        claim=categories_claim,
        node=categories_leaf,
        sources=item.official_urls or None,
        additional_instruction=(
            "Confirm that the provided official page(s) explicitly indicate the GPA category or categories to which the minimum "
            "applies (e.g., overall/cumulative GPA, professional studies GPA, teaching field GPA). If categories are not "
            "explicitly stated on the page(s), judge as not supported."
        ),
    )

    # Testing requirement supported
    testing_leaf = evaluator.add_leaf(
        id=f"Testing_{index+1}",
        desc="Documents the required standardized testing for admission to the teacher education program (or explicitly states none required, if that is what the official source says)",
        parent=outputs_node,
        critical=True
    )
    testing_text = item.testing or "not specified"
    testing_claim = (
        f"The standardized testing requirement for admission to the teacher education program is: {testing_text}."
    )
    await evaluator.verify(
        claim=testing_claim,
        node=testing_leaf,
        sources=item.official_urls or None,
        additional_instruction=(
            "Verify on the provided official page(s) whether admission requires a standardized test such as Praxis Core, "
            "Alabama Educators Certification Assessment Program (AECTP), ACT/SAT, or if it explicitly states none required. "
            "If the page(s) do not mention testing requirements as claimed, judge as not supported."
        ),
    )

    # Official URL provided and official
    official_url_leaf = evaluator.add_leaf(
        id=f"Official_URL_{index+1}",
        desc="Provides an official university website URL that supports the stated GPA requirement, applicable GPA categories, and testing requirement (verifiable on the linked page)",
        parent=outputs_node,
        critical=True
    )
    official_url_claim = (
        "At least one provided URL is an official university website page (e.g., .edu domain) relevant to teacher education admissions "
        "and contains the admissions requirements information (minimum GPA and related categories and/or testing)."
    )
    await evaluator.verify(
        claim=official_url_claim,
        node=official_url_leaf,
        sources=item.official_urls or None,
        additional_instruction=(
            "Judge the official nature of the URL (e.g., .edu domain, recognizable university subdomain) and its relevance to teacher "
            "education admissions. The page should include admissions requirement information such as minimum GPA and related categories "
            "and/or testing. If none of the provided URLs qualify, judge as not supported."
        ),
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
    Evaluate an answer to the Alabama CAEP-accredited K–6 admission GPA task.
    """
    # Initialize evaluator (Root set to parallel aggregation; set critical=False to allow partial credit)
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Limit to exactly four universities (padding if fewer)
    uni_items = _first_n_universities(extracted, 4)

    # Build verification trees for the four universities
    # According to the rubric, each university subtree is sequential, and the root aggregates them in parallel.
    for idx, item in enumerate(uni_items):
        await build_university_verification(evaluator, root, item, idx)

    # Optionally record custom info
    evaluator.add_custom_info(
        {"extracted_count": len(extracted.universities)},
        info_type="extraction_stats",
        info_name="universities_extraction_stats"
    )

    # Return structured summary
    return evaluator.get_summary()