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
TASK_ID = "public_universities_validation"
TASK_DESCRIPTION = """Identify 4 public universities in the United States that meet ALL of the following criteria:

1. Must be a public (state-supported) university
2. Must be regionally accredited by one of the following: Higher Learning Commission (HLC), Middle States Commission on Higher Education (MSCHE), or Southern Association of Colleges and Schools Commission on Colleges (SACSCOC)
3. Must be a member institution of NCAA Division I athletics
4. In-state tuition and fees for the 2024-2025 or 2025-2026 academic year must be between $6,000 and $13,000
5. Total undergraduate enrollment must be between 20,000 and 35,000 students
6. Must have at least one ABET-accredited engineering program OR at least one AACSB-accredited business program (or both)

For each university, provide:
- Official university name
- City and state location
- Official university website URL
- Specific in-state tuition and fees amount for 2024-2025 or 2025-2026
- Specific undergraduate enrollment number
- Regional accreditor name (HLC, MSCHE, or SACSCOC)
- Program accreditation type (ABET engineering, AACSB business, or both)
- Reference URL(s) supporting the provided information
"""

ALLOWED_ACCREDITORS = {"HLC", "MSCHE", "SACSCOC"}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    """Information for a single university, as extracted from the answer."""
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    website_url: Optional[str] = None
    tuition_amount: Optional[str] = None            # Keep as string to accommodate ranges like "$12,300" or "about $10,500"
    tuition_year: Optional[str] = None              # Expected values like "2024-2025" or "2025-2026"
    enrollment: Optional[str] = None                # Keep as string to allow formats like "approx. 30,000"
    regional_accreditor: Optional[str] = None       # One of HLC, MSCHE, SACSCOC (ideally)
    program_accreditation_type: Optional[str] = None  # "ABET", "AACSB", or "both"
    reference_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    """Top-level extraction: up to 4 universities."""
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to 4 universities provided in the answer. For each university, extract the following fields exactly as stated in the answer:

    Fields:
    - name: Official university name
    - city: City where the university is located
    - state: State where the university is located (2-letter code or full name acceptable)
    - website_url: The official university website URL
    - tuition_amount: The specific in-state tuition and fees amount for either the 2024-2025 or 2025-2026 academic year; keep it as a string as written (e.g., "$12,300" or "about $10,500")
    - tuition_year: "2024-2025" or "2025-2026" as written in the answer (if available)
    - enrollment: The specific undergraduate enrollment number (keep as string; may include commas or approximations)
    - regional_accreditor: One of "HLC", "MSCHE", or "SACSCOC" if given in the answer; otherwise return the accreditor string as written or null if not provided
    - program_accreditation_type: One of "ABET", "AACSB", or "both" if specified; if both types are claimed, use "both"; if only ABET is claimed, use "ABET"; if only AACSB is claimed, use "AACSB"; if unclear or not specified, return null
    - reference_urls: All explicit URLs cited in the answer that support the provided information (tuition, enrollment, accreditation, NCAA membership, etc.). Include the official website if explicitly given as a source. If none are cited, return an empty list.

    Important extraction rules:
    - Do not invent or infer any values; only extract what is explicitly present in the answer.
    - For URLs, extract actual URL strings (including protocol). Extract URLs found in plain text or markdown links; if a markdown link is present, extract the underlying URL.
    - If a requested field is missing, set it to null (or empty list for reference_urls).
    - Return a JSON object with a single field: "universities", which is an array of up to 4 UniversityItem objects in the order they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    """Deduplicate and filter obvious non-empty URLs; ensure protocol if missing."""
    seen = set()
    result: List[str] = []
    for u in urls:
        if not u or not isinstance(u, str):
            continue
        su = u.strip()
        if not su:
            continue
        # prepend http if missing protocol and looks like a domain
        if not su.lower().startswith(("http://", "https://")) and "." in su:
            su = "http://" + su
        if su not in seen:
            seen.add(su)
            result.append(su)
    return result


def get_sources_for_university(u: UniversityItem) -> List[str]:
    """Combine official website URL and reference URLs for verification."""
    urls = []
    if u.website_url:
        urls.append(u.website_url)
    urls.extend(u.reference_urls or [])
    return _dedup_urls(urls)


# --------------------------------------------------------------------------- #
# University verification                                                     #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    index: int,
) -> None:
    """
    Build verification sub-tree and run checks for one university.
    """
    uni_node = evaluator.add_parallel(
        id=f"University_{index}",
        desc=f"University #{index} verification - all required constraints and details",
        parent=parent_node,
        critical=False,  # Allow partial credit per university
    )

    # ----------------------- Details presence (as custom checks) ----------------------- #
    details_node = evaluator.add_parallel(
        id=f"U{index}_Details",
        desc=f"U{index}: Details presence and support",
        parent=uni_node,
        critical=False
    )

    # Presence checks (some essential for meaningful verification)
    name_present = evaluator.add_custom_node(
        result=bool(uni.name and uni.name.strip()),
        id=f"U{index}_University_Name",
        desc="Provide the official name of the university",
        parent=details_node,
        critical=True  # Name is essential
    )

    location_present = evaluator.add_custom_node(
        result=bool(uni.city and uni.city.strip() and uni.state and uni.state.strip()),
        id=f"U{index}_Location",
        desc="Provide the city and state where the university is located",
        parent=details_node,
        critical=True  # Location is essential detail
    )

    website_present = evaluator.add_custom_node(
        result=bool(uni.website_url and uni.website_url.strip()),
        id=f"U{index}_Website_URL",
        desc="Provide the official university website URL",
        parent=details_node,
        critical=True  # Sources grounding relies on website
    )

    tuition_amount_present = evaluator.add_custom_node(
        result=bool(uni.tuition_amount and uni.tuition_amount.strip()),
        id=f"U{index}_Tuition_Amount",
        desc="Provide the specific in-state tuition and fees amount for 2024-2025 or 2025-2026",
        parent=details_node,
        critical=False
    )

    enrollment_present = evaluator.add_custom_node(
        result=bool(uni.enrollment and uni.enrollment.strip()),
        id=f"U{index}_Enrollment_Number",
        desc="Provide the specific undergraduate enrollment number",
        parent=details_node,
        critical=False
    )

    accreditor_present = evaluator.add_custom_node(
        result=bool(uni.regional_accreditor and uni.regional_accreditor.strip()),
        id=f"U{index}_Accreditation_Details",
        desc="Specify which regional accreditor (HLC, MSCHE, or SACSCOC)",
        parent=details_node,
        critical=True
    )

    program_type_present = evaluator.add_custom_node(
        result=bool(uni.program_accreditation_type and uni.program_accreditation_type.strip()),
        id=f"U{index}_Program_Type",
        desc="Specify whether university has ABET-accredited engineering program(s) or AACSB-accredited business program, or both",
        parent=details_node,
        critical=True
    )

    refs_present = evaluator.add_custom_node(
        result=bool(uni.reference_urls and len(uni.reference_urls) > 0),
        id=f"U{index}_Reference_URLs",
        desc="Provide reference URL(s) supporting the provided information",
        parent=details_node,
        critical=False
    )

    # ----------------------- Constraint checks (critical) ----------------------- #
    constraints_node = evaluator.add_parallel(
        id=f"U{index}_Constraints",
        desc=f"U{index}: Must satisfy ALL constraints (public, regional accreditation, NCAA DI, tuition range, enrollment range, program accreditation)",
        parent=uni_node,
        critical=False  # Children leaves will be marked critical individually
    )

    sources = get_sources_for_university(uni)

    # Public status
    node_public = evaluator.add_leaf(
        id=f"U{index}_Public_Status",
        desc="University is a public (state-supported) institution",
        parent=constraints_node,
        critical=True
    )
    claim_public = f"{uni.name} is a public (state-supported) university."
    # Batch ready: collect claims
    claims_and_sources: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []
    claims_and_sources.append((
        claim_public,
        sources,
        node_public,
        "Look for clear statements like 'public university', 'public research university', 'state university', or 'state-supported'. Allow synonyms and minor wording variants."
    ))

    # Regional accreditation (allowed + supported)
    node_accred_allowed = evaluator.add_custom_node(
        result=bool(uni.regional_accreditor and uni.regional_accreditor.strip() in ALLOWED_ACCREDITORS),
        id=f"U{index}_Regional_Accreditation_Allowed",
        desc="Regional accreditor is one of HLC, MSCHE, or SACSCOC",
        parent=constraints_node,
        critical=True
    )

    node_accred_supported = evaluator.add_leaf(
        id=f"U{index}_Regional_Accreditation",
        desc="University is regionally accredited by Higher Learning Commission (HLC), Middle States Commission on Higher Education (MSCHE), or Southern Association of Colleges and Schools Commission on Colleges (SACSCOC)",
        parent=constraints_node,
        critical=True
    )
    claim_accred = f"{uni.name} is accredited by {uni.regional_accreditor}."
    claims_and_sources.append((
        claim_accred,
        sources,
        node_accred_supported,
        "Confirm accreditation by the specified regional accreditor (HLC, MSCHE, or SACSCOC). Accept institution accreditation pages or accreditor listings."
    ))

    # NCAA Division I membership
    node_ncaa = evaluator.add_leaf(
        id=f"U{index}_NCAA_Division_I",
        desc="University is a member institution of NCAA Division I athletics",
        parent=constraints_node,
        critical=True
    )
    claim_ncaa = f"{uni.name} is a member of NCAA Division I athletics."
    claims_and_sources.append((
        claim_ncaa,
        sources,
        node_ncaa,
        "Verify that the institution participates in NCAA Division I. Accept NCAA membership lists or official athletics pages stating Division I."
    ))

    # Tuition range
    node_tuition_range = evaluator.add_leaf(
        id=f"U{index}_Tuition_Range",
        desc="In-state tuition and fees for 2024-2025 or 2025-2026 academic year fall between $6,000 and $13,000",
        parent=constraints_node,
        critical=True
    )
    year_text = uni.tuition_year if (uni.tuition_year and uni.tuition_year.strip()) else "the specified academic year"
    claim_tuition_range = f"For {year_text}, the in-state tuition and fees at {uni.name} are between $6,000 and $13,000."
    claims_and_sources.append((
        claim_tuition_range,
        sources,
        node_tuition_range,
        "Check tuition and fees for the specified academic year on official tuition/fees pages. Accept minor rounding."
    ))

    # Enrollment range
    node_enroll_range = evaluator.add_leaf(
        id=f"U{index}_Enrollment_Range",
        desc="Total undergraduate enrollment is between 20,000 and 35,000 students",
        parent=constraints_node,
        critical=True
    )
    claim_enroll_range = f"The undergraduate enrollment at {uni.name} is between 20,000 and 35,000 students."
    claims_and_sources.append((
        claim_enroll_range,
        sources,
        node_enroll_range,
        "Use official facts pages, institutional research, or authoritative sources. Allow approximations and minor rounding."
    ))

    # Program accreditation (ABET or AACSB)
    node_program_acc = evaluator.add_leaf(
        id=f"U{index}_Program_Accreditation",
        desc="University has at least one ABET-accredited engineering program OR at least one AACSB-accredited business program",
        parent=constraints_node,
        critical=True
    )
    prog_type = (uni.program_accreditation_type or "").strip().lower()
    if prog_type == "both":
        claim_prog = f"{uni.name} has at least one ABET-accredited engineering program and at least one AACSB-accredited business program."
    elif prog_type == "abet":
        claim_prog = f"{uni.name} has at least one ABET-accredited engineering program."
    elif prog_type == "aacsb":
        claim_prog = f"{uni.name} has at least one AACSB-accredited business program."
    else:
        # If not specified, state the OR condition per requirement
        claim_prog = f"{uni.name} has at least one ABET-accredited engineering program OR at least one AACSB-accredited business program."
    claims_and_sources.append((
        claim_prog,
        sources,
        node_program_acc,
        "Confirm presence of ABET-accredited engineering or AACSB-accredited business programs. Accept official program accreditation listings from ABET/AACSB or institution pages listing these accreditations."
    ))

    # Execute critical constraint verifications in batch to avoid sibling gating interference
    await evaluator.batch_verify(claims_and_sources)

    # ----------------------- Details supported by sources (non-critical) ----------------------- #
    # These checks depend on presence; they will be skipped if preconditions fail

    # Official website matches university name
    node_site_matches_name = evaluator.add_leaf(
        id=f"U{index}_Website_Official",
        desc="Official website URL corresponds to the university's official site",
        parent=details_node,
        critical=False
    )
    claim_site_name = f"The official website of {uni.name} is {uni.website_url}."
    await evaluator.verify(
        claim=claim_site_name,
        node=node_site_matches_name,
        sources=[s for s in _dedup_urls([uni.website_url])],
        additional_instruction="Check the homepage or About page to confirm the institution's name appears, indicating this is the official site."
    )

    # Location supported
    node_location_supported = evaluator.add_leaf(
        id=f"U{index}_Location_Supported",
        desc="The city and state location are supported by sources",
        parent=details_node,
        critical=False
    )
    claim_location = f"{uni.name} is located in {uni.city}, {uni.state}."
    await evaluator.verify(
        claim=claim_location,
        node=node_location_supported,
        sources=sources,
        additional_instruction="Verify the institution's location (city, state) on official pages or authoritative sources."
    )

    # Tuition amount (exact) supported
    node_tuition_exact = evaluator.add_leaf(
        id=f"U{index}_Tuition_Exact",
        desc="Specific in-state tuition and fees amount is supported by sources",
        parent=details_node,
        critical=False
    )
    claim_tuition_exact = f"For {uni.tuition_year}, the in-state tuition and fees at {uni.name} are {uni.tuition_amount}."
    await evaluator.verify(
        claim=claim_tuition_exact,
        node=node_tuition_exact,
        sources=sources,
        additional_instruction="Confirm the stated in-state tuition and fees amount for the specified academic year. Allow minor rounding and formatting differences."
    )

    # Enrollment number (exact) supported
    node_enroll_exact = evaluator.add_leaf(
        id=f"U{index}_Enrollment_Exact",
        desc="Specific undergraduate enrollment number is supported by sources",
        parent=details_node,
        critical=False
    )
    claim_enroll_exact = f"The undergraduate enrollment at {uni.name} is {uni.enrollment} students."
    await evaluator.verify(
        claim=claim_enroll_exact,
        node=node_enroll_exact,
        sources=sources,
        additional_instruction="Confirm the undergraduate enrollment figure; allow approximations or rounding."
    )

    # Accreditation details (exact) supported
    node_accred_exact = evaluator.add_leaf(
        id=f"U{index}_Accreditation_Supported",
        desc="Regional accreditor detail (HLC, MSCHE, or SACSCOC) is supported by sources",
        parent=details_node,
        critical=False
    )
    claim_accred_exact = f"{uni.name} is accredited by {uni.regional_accreditor}."
    await evaluator.verify(
        claim=claim_accred_exact,
        node=node_accred_exact,
        sources=sources,
        additional_instruction="Confirm the institution is accredited by the specified regional accreditor (HLC, MSCHE, or SACSCOC)."
    )

    # Program type details (exact) supported
    node_program_exact = evaluator.add_leaf(
        id=f"U{index}_Program_Type_Supported",
        desc="Program accreditation type (ABET/AACSB/both) is supported by sources",
        parent=details_node,
        critical=False
    )
    if prog_type == "both":
        claim_program_exact = f"{uni.name} has ABET-accredited engineering programs and AACSB-accredited business programs."
    elif prog_type == "abet":
        claim_program_exact = f"{uni.name} has ABET-accredited engineering programs."
    elif prog_type == "aacsb":
        claim_program_exact = f"{uni.name} has AACSB-accredited business programs."
    else:
        claim_program_exact = f"{uni.name} has either ABET-accredited engineering programs or AACSB-accredited business programs."
    await evaluator.verify(
        claim=claim_program_exact,
        node=node_program_exact,
        sources=sources,
        additional_instruction="Verify the claimed accreditation type(s) via ABET/AACSB listings or official program pages."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the public universities criteria task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Universities evaluated independently
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

    # Extract up to 4 universities
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Normalize to exactly 4 items (pad with empty)
    universities: List[UniversityItem] = list(extracted.universities[:4])
    while len(universities) < 4:
        universities.append(UniversityItem())

    # Build subtrees and verify
    for idx, uni in enumerate(universities, start=1):
        await verify_university(evaluator, root, uni, idx)

    # Return summary
    return evaluator.get_summary()