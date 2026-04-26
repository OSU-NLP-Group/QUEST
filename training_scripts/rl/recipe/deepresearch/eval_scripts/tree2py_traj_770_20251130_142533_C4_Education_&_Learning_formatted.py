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
TASK_ID = "identify_university_ny_1906_msche_business_10k_15k_multicampus"
TASK_DESCRIPTION = (
    "Identify the name of a university that meets all of the following criteria: "
    "(1) Located in New York State, "
    "(2) Founded in 1906, "
    "(3) Accredited by the Middle States Commission on Higher Education (MSCHE), "
    "(4) Originally established as a business school, "
    "(5) Has enrollment between 10,000 and 15,000 students as of fall 2024, and "
    "(6) Operates multiple campuses in the New York metropolitan area. "
    "Provide the official name of the university and a reference URL from an authoritative source "
    "(such as the university's official website or accreditation agency) that confirms at least one of these criteria."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityExtraction(BaseModel):
    university_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)

    founding_year: Optional[str] = None
    accreditation_agency: Optional[str] = None
    origin_description: Optional[str] = None
    enrollment_fall_2024: Optional[str] = None
    campuses_description: Optional[str] = None
    location_state: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_university() -> str:
    return (
        "Extract the core information about the single university identified in the answer. "
        "Return the following fields:\n"
        "1. university_name: The official name of the university chosen in the answer.\n"
        "2. reference_urls: A list of reference URLs included in the answer (official university site, .edu pages, "
        "   accreditation agency like MSCHE, or other authoritative sources). Extract actual URLs only.\n"
        "3. founding_year: The year the university was founded if stated in the answer (string).\n"
        "4. accreditation_agency: The accreditation agency stated (string). Prefer 'Middle States Commission on Higher Education' or 'MSCHE' if present.\n"
        "5. origin_description: Any statement about the institution originally being established as a business school.\n"
        "6. enrollment_fall_2024: Enrollment figure or range mentioned for fall 2024 (string; may be approximate).\n"
        "7. campuses_description: Any statement about multiple campuses in the New York metropolitan area.\n"
        "8. location_state: The U.S. state stated for the university location (e.g., 'New York', 'NY') if mentioned in the answer.\n"
        "If a field is not mentioned, set it to null. For reference_urls, return an empty list if none are provided."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _sources_or_none(urls: List[str]) -> Optional[List[str]]:
    return urls if urls else None


async def _verify_reference_supports_any_criterion(
    evaluator: Evaluator,
    urls: List[str],
    uni_name: Optional[str],
) -> bool:
    """
    Check whether at least one provided reference URL is authoritative and supports at least
    one of the specified criteria. To operationalize this, we attempt verification of several
    concrete claims against the provided URLs and return True if any succeeds.

    We do not assign results to a specific leaf node inside this helper; instead we return a boolean
    to be used in a custom node.
    """
    if not urls or not uni_name:
        return False

    claims_and_instructions: List[Dict[str, str]] = [
        {
            "claim": f"The university '{uni_name}' is accredited by the Middle States Commission on Higher Education (MSCHE).",
            "ins": (
                "Verify explicitly that the page states accreditation by MSCHE or Middle States Commission on Higher Education. "
                "Prefer authoritative sources such as the university's official website (.edu) or MSCHE's official site."
            ),
        },
        {
            "claim": f"The university '{uni_name}' was founded in 1906.",
            "ins": (
                "Check an official 'About' or 'History' page for language like 'founded in 1906' or 'established in 1906'. "
                "Treat official university sites (.edu or the institution's main domain) as authoritative."
            ),
        },
        {
            "claim": f"The university '{uni_name}' operates multiple campuses in the New York metropolitan area.",
            "ins": (
                "Confirm that at least two distinct campuses or locations are in the New York metropolitan area "
                "(e.g., Manhattan/NYC and Westchester/Long Island). Prefer official campus/location pages."
            ),
        },
        {
            "claim": f"The university '{uni_name}' is located in New York State (NY).",
            "ins": (
                "Look for address or descriptive mentions indicating the state is New York (NY). "
                "Campus addresses or 'New York, NY' count as evidence."
            ),
        },
    ]

    # Try each claim against the URL list; if any succeeds, return True.
    for ci in claims_and_instructions:
        try:
            ok = await evaluator.verify(
                claim=ci["claim"],
                node=None,  # standalone verification without binding to a node
                sources=urls,
                additional_instruction=ci["ins"],
            )
            if ok:
                return True
        except Exception:
            # Ignore errors on individual verifications; continue to next claim
            continue

    return False


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_criteria(evaluator: Evaluator, extracted: UniversityExtraction) -> None:
    """
    Build verification nodes according to the rubric and perform verifications.
    All children under Identify_University are critical.
    """
    identify_node = evaluator.add_parallel(
        id="Identify_University",
        desc="Identify a university that meets all specified criteria and provide required documentation",
        parent=evaluator.root,
        critical=True,
    )

    uni_name = extracted.university_name or ""
    sources_list = extracted.reference_urls

    # New York Location
    ny_loc_node = evaluator.add_leaf(
        id="New_York_Location",
        desc="The university must be located in New York State",
        parent=identify_node,
        critical=True,
    )
    ny_claim = f"The university '{uni_name}' is located in New York State (NY)."
    await evaluator.verify(
        claim=ny_claim,
        node=ny_loc_node,
        sources=_sources_or_none(sources_list),
        additional_instruction=(
            "Confirm via authoritative evidence on the provided page(s)—such as campus addresses, contact sections, "
            "or explicit statements—that the university is located in New York State (NY). "
            "Mentions of New York City (NYC) or other NY localities count."
        ),
    )

    # Founded in 1906
    founded_node = evaluator.add_leaf(
        id="Founded_1906",
        desc="The university must have been founded in 1906",
        parent=identify_node,
        critical=True,
    )
    founded_claim = f"The university '{uni_name}' was founded in 1906."
    await evaluator.verify(
        claim=founded_claim,
        node=founded_node,
        sources=_sources_or_none(sources_list),
        additional_instruction=(
            "Check the official 'About' or 'History' content for phrases like 'founded in 1906' or 'established in 1906'. "
            "Prefer authoritative pages (official university site)."
        ),
    )

    # MSCHE Accreditation
    msche_node = evaluator.add_leaf(
        id="MSCHE_Accreditation",
        desc="The university must be accredited by the Middle States Commission on Higher Education",
        parent=identify_node,
        critical=True,
    )
    msche_claim = f"The university '{uni_name}' is accredited by the Middle States Commission on Higher Education (MSCHE)."
    await evaluator.verify(
        claim=msche_claim,
        node=msche_node,
        sources=_sources_or_none(sources_list),
        additional_instruction=(
            "Prefer explicit accreditation listings on MSCHE's official website or the university's official accreditation page. "
            "Accept textual variants like 'Middle States Commission on Higher Education' or 'MSCHE'."
        ),
    )

    # Business School Origin
    origin_node = evaluator.add_leaf(
        id="Business_School_Origin",
        desc="The university must have been originally established as a business school",
        parent=identify_node,
        critical=True,
    )
    origin_claim = f"The university '{uni_name}' was originally established as a business school."
    await evaluator.verify(
        claim=origin_claim,
        node=origin_node,
        sources=_sources_or_none(sources_list),
        additional_instruction=(
            "Check official history pages for origin statements indicating a business-focused founding, "
            "e.g., 'founded as a business school', 'business institute', or similar language."
        ),
    )

    # Enrollment Range (Fall 2024 between 10,000 and 15,000)
    enroll_node = evaluator.add_leaf(
        id="Enrollment_Range_2024",
        desc="The university must have enrollment between 10,000 and 15,000 students as of fall 2024",
        parent=identify_node,
        critical=True,
    )
    enroll_claim = (
        f"As of fall 2024, the university '{uni_name}' has total student enrollment between 10,000 and 15,000."
    )
    await evaluator.verify(
        claim=enroll_claim,
        node=enroll_node,
        sources=_sources_or_none(sources_list),
        additional_instruction=(
            "Look for Factbook, At-a-Glance, CDS, or official statistics pages indicating fall 2024 enrollment. "
            "Allow minor rounding differences; verify total headcount is within 10,000 to 15,000."
        ),
    )

    # Multiple NYC Area Campuses
    campuses_node = evaluator.add_leaf(
        id="Multiple_NYC_Campuses",
        desc="The university must operate multiple campuses in the New York metropolitan area",
        parent=identify_node,
        critical=True,
    )
    campuses_claim = (
        f"The university '{uni_name}' operates multiple campuses within the New York metropolitan area."
    )
    await evaluator.verify(
        claim=campuses_claim,
        node=campuses_node,
        sources=_sources_or_none(sources_list),
        additional_instruction=(
            "Confirm at least two distinct campuses/locations in the NYC metro (e.g., Manhattan, Westchester, Queens, Bronx, Staten Island, Long Island). "
            "Prefer official campus/location pages."
        ),
    )

    # Reference URL Provided (break into two critical checks under a sub-node for clarity)
    ref_parent = evaluator.add_parallel(
        id="Reference_URL_Provided",
        desc="The answer must include at least one reference URL from an authoritative source that confirms at least one of the criteria",
        parent=identify_node,
        critical=True,
    )

    ref_exist_node = evaluator.add_custom_node(
        result=bool(sources_list),
        id="Reference_URL_Exists",
        desc="At least one reference URL is provided in the answer",
        parent=ref_parent,
        critical=True,
    )

    # Check that at least one provided reference URL is authoritative and supports at least one criterion
    supported = await _verify_reference_supports_any_criterion(
        evaluator=evaluator, urls=sources_list, uni_name=extracted.university_name
    )
    evaluator.add_custom_node(
        result=supported,
        id="Reference_URL_Supports_Criterion",
        desc="At least one provided reference URL is authoritative and explicitly supports one of the specified criteria",
        parent=ref_parent,
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
    Evaluate an answer for the university identification task with specified constraints.
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

    # Extract candidate university info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_university(),
        template_class=UniversityExtraction,
        extraction_name="university_extraction",
    )

    # Record criteria as ground truth requirements (not a target label, just to display requirements)
    evaluator.add_ground_truth({
        "required_criteria": [
            "Located in New York State",
            "Founded in 1906",
            "Accredited by MSCHE",
            "Originally established as a business school",
            "Enrollment between 10,000 and 15,000 students as of fall 2024",
            "Operates multiple campuses in the New York metropolitan area",
            "Provide at least one authoritative reference URL confirming a criterion",
        ]
    })

    # Build tree and perform verifications
    await build_and_verify_criteria(evaluator, extracted)

    # Return standard summary
    return evaluator.get_summary()