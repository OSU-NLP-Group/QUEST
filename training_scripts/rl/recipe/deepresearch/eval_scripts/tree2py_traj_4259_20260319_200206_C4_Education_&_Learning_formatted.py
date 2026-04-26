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
TASK_ID = "va_aacsb_largest_undergrad"
TASK_DESCRIPTION = (
    "Identify the public university in Virginia with the largest undergraduate enrollment (as of Fall 2024 or Fall 2025) among those institutions whose business schools hold AACSB accreditation. "
    "For the identified university, provide the following information: (1) The exact undergraduate enrollment number with the corresponding term and year, "
    "(2) The official name of its AACSB-accredited business school or college, (3) The city where the main campus is located, (4) The percentage of in-state students, "
    "(5) The total university enrollment figure. Provide a URL reference that verifies each piece of information."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FactSources(BaseModel):
    undergrad_enrollment: List[str] = Field(default_factory=list)
    business_school_name: List[str] = Field(default_factory=list)
    aacsb_accreditation: List[str] = Field(default_factory=list)
    campus_city: List[str] = Field(default_factory=list)
    in_state_percentage: List[str] = Field(default_factory=list)
    total_enrollment: List[str] = Field(default_factory=list)
    public_status: List[str] = Field(default_factory=list)
    largest_claim: List[str] = Field(default_factory=list)


class UniversityExtraction(BaseModel):
    university_name: Optional[str] = None
    business_school_name: Optional[str] = None
    main_campus_city: Optional[str] = None
    undergrad_enrollment_number: Optional[str] = None
    undergrad_enrollment_term: Optional[str] = None  # Expected like "Fall 2024" or "Fall 2025"
    in_state_percentage: Optional[str] = None
    total_enrollment: Optional[str] = None
    sources: FactSources = Field(default_factory=FactSources)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_university_info() -> str:
    return """
    Extract the single Virginia public university identified as the answer and all requested facts exactly as stated in the answer text.
    Required fields to extract:
    - university_name: The full official name of the university identified as the answer.
    - business_school_name: The official name of its business school or college that is AACSB-accredited.
    - main_campus_city: The city where the university's main campus is located (city only, e.g., "Blacksburg").
    - undergrad_enrollment_number: The exact undergraduate enrollment figure provided in the answer.
    - undergrad_enrollment_term: The associated term and year for that undergraduate enrollment, expected to be "Fall 2024" or "Fall 2025" (extract exactly as stated).
    - in_state_percentage: The percentage of in-state students as provided (e.g., "68%" or "68 percent"). Return the exact string.
    - total_enrollment: The total university enrollment figure as provided (exact string).
    
    For each of the following facts, extract all verifying URLs explicitly cited in the answer text. If none are provided, return an empty list:
    - sources.undergrad_enrollment: URLs that verify the undergraduate enrollment number and the specified term/year.
    - sources.business_school_name: URLs that verify the official business school/college name.
    - sources.aacsb_accreditation: URLs that verify the business school/college holds AACSB accreditation (e.g., on aacsb.edu or official university accreditation pages).
    - sources.campus_city: URLs that verify the main campus city.
    - sources.in_state_percentage: URLs that verify the in-state student percentage.
    - sources.total_enrollment: URLs that verify the total university enrollment figure.
    - sources.public_status: URLs that verify the institution is a public university (e.g., official "About" page, state higher-ed council page).
    - sources.largest_claim: URLs used to justify that this university has the largest undergraduate enrollment among Virginia public universities with AACSB-accredited business schools for Fall 2024 or Fall 2025.
    
    IMPORTANT:
    - Only extract URLs explicitly present in the answer (including markdown links). Do not invent or infer URLs.
    - Return the strings exactly as they appear in the answer; do not rewrite or normalize numbers or names.
    - If any field is missing in the answer, return null (for strings) or an empty list (for URL lists).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str]) -> str:
    return s or ""

def _unique_nonempty(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls or []:
        u2 = (u or "").strip()
        if not u2:
            continue
        if u2 not in seen:
            seen.add(u2)
            result.append(u2)
    return result

def _combine_sources(*args: List[str]) -> List[str]:
    combined: List[str] = []
    for lst in args:
        combined.extend(lst or [])
    return _unique_nonempty(combined)

def _url_is_acceptable_for_fact(url: str, fact: str) -> bool:
    """Lightweight domain heuristic for source acceptability."""
    u = (url or "").lower()
    # General official university or state council pages
    is_edu = ".edu" in u
    is_state_va = "schev.edu" in u or ".virginia.gov" in u
    # AACSB official
    is_aacsb = "aacsb.edu" in u or "aacsb.org" in u
    # Google Drive/docs or generic news often not acceptable for strict officiality
    is_generic = any(x in u for x in ["wikipedia.org", "linkedin.com", "facebook.com", "twitter.com", "x.com", "medium.com"])

    if fact == "aacsb_accreditation":
        return is_aacsb or is_edu  # AACSB page or school's official accreditation page
    if fact == "business_school_name":
        return is_edu or is_aacsb
    if fact in {"undergrad_enrollment", "campus_city", "in_state_percentage", "total_enrollment", "public_status"}:
        return is_edu or is_state_va
    if fact == "largest_claim":
        # Prefer official/statewide or university institutional research; AACSB list may help define eligible set
        return is_state_va or is_edu or is_aacsb
    # Default conservative
    return is_edu or is_aacsb or is_state_va

def _has_acceptable_citation(urls: List[str], fact: str) -> bool:
    urls = _unique_nonempty(urls)
    if not urls:
        return False
    return any(_url_is_acceptable_for_fact(u, fact) for u in urls)


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: UniversityExtraction) -> None:
    """
    Build and run all verification nodes based on the rubric.
    """
    root = evaluator.root
    # Create an overall critical parent to mirror rubric's critical root
    overall = evaluator.add_parallel(
        id="overall_evaluation",
        desc="Evaluation of the identified university and all required information",
        parent=root,
        critical=True
    )

    uni_name = _safe(extracted.university_name)
    bschool_name = _safe(extracted.business_school_name)
    city = _safe(extracted.main_campus_city)
    ug_num = _safe(extracted.undergrad_enrollment_number)
    ug_term = _safe(extracted.undergrad_enrollment_term)
    instate_pct = _safe(extracted.in_state_percentage)
    total_enroll = _safe(extracted.total_enrollment)

    # 1) university_identified_by_name (existence)
    evaluator.add_custom_node(
        result=bool(uni_name.strip()),
        id="university_identified_by_name",
        desc="A specific public university is identified by name as the answer",
        parent=overall,
        critical=True
    )

    # 2) is_public_university
    node_public = evaluator.add_leaf(
        id="is_public_university",
        desc="The identified institution is a public university",
        parent=overall,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} is a public university.",
        node=node_public,
        sources=_unique_nonempty(extracted.sources.public_status),
        additional_instruction="Judge this only if the provided URL(s) are official (university .edu or Virginia state higher-ed council) or AACSB. Do not rely on general knowledge."
    )

    # 3) is_located_in_virginia
    node_va = evaluator.add_leaf(
        id="is_located_in_virginia",
        desc="The identified institution is located in Virginia, United States",
        parent=overall,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} is located in the U.S. state of Virginia.",
        node=node_va,
        sources=_unique_nonempty(_combine_sources(extracted.sources.campus_city)),
        additional_instruction="Confirm from official sources (.edu or state council) that the university is based in Virginia."
    )

    # 4) has_aacsb_accredited_business_school
    node_aacsb = evaluator.add_leaf(
        id="has_aacsb_accredited_business_school",
        desc="The identified institution has an AACSB-accredited business school or college",
        parent=overall,
        critical=True
    )
    aacsb_claim = (
        f"The business school/college at {uni_name}"
        + (f" (named '{bschool_name}')" if bschool_name else "")
        + " holds AACSB accreditation."
    )
    await evaluator.verify(
        claim=aacsb_claim,
        node=node_aacsb,
        sources=_unique_nonempty(extracted.sources.aacsb_accreditation),
        additional_instruction="Rely on AACSB's official accreditation directory (preferred) or the university's official accreditation page. Do not accept non-official sources."
    )

    # 5) largest_undergrad_enrollment_among_qualified
    node_largest = evaluator.add_leaf(
        id="largest_undergrad_enrollment_among_qualified",
        desc="For the reported Fall 2024 or Fall 2025 term, the identified university has the largest undergraduate enrollment among the institutions that satisfy the public/VA/AACSB criteria",
        parent=overall,
        critical=True
    )
    term_display = ug_term if ug_term else "the specified term"
    largest_claim = (
        f"For {term_display}, among Virginia public universities whose business schools are AACSB-accredited, "
        f"{uni_name} has the largest undergraduate enrollment."
    )
    largest_sources = _unique_nonempty(_combine_sources(extracted.sources.largest_claim, extracted.sources.undergrad_enrollment))
    await evaluator.verify(
        claim=largest_claim,
        node=node_largest,
        sources=largest_sources,
        additional_instruction=(
            "Only consider Virginia public universities that have AACSB-accredited business schools. "
            "Use the provided URLs to confirm the set and make the comparison for the SAME term (Fall 2024 or Fall 2025). "
            "Conclude 'supported' only if the evidence explicitly establishes that the chosen university's undergraduate enrollment "
            "is the largest among that eligible group for the specified term."
        )
    )

    # 6) undergraduate_enrollment_info_correct
    node_ug = evaluator.add_leaf(
        id="undergraduate_enrollment_info_correct",
        desc="The exact undergraduate enrollment number is provided with its corresponding term/year (Fall 2024 or Fall 2025) and matches an official source for that term",
        parent=overall,
        critical=True
    )
    ug_claim = f"The undergraduate enrollment of {uni_name} in {ug_term} is {ug_num}."
    await evaluator.verify(
        claim=ug_claim,
        node=node_ug,
        sources=_unique_nonempty(extracted.sources.undergrad_enrollment),
        additional_instruction="Verify both the exact number and the specific term/year (must be Fall 2024 or Fall 2025) from an official source (university .edu or state council)."
    )

    # 7) business_school_official_name_correct
    node_bname = evaluator.add_leaf(
        id="business_school_official_name_correct",
        desc="The official name of the AACSB-accredited business school/college is correctly provided",
        parent=overall,
        critical=True
    )
    bname_claim = f"The official AACSB-accredited business school/college at {uni_name} is named '{bschool_name}'."
    await evaluator.verify(
        claim=bname_claim,
        node=node_bname,
        sources=_unique_nonempty(_combine_sources(extracted.sources.business_school_name, extracted.sources.aacsb_accreditation)),
        additional_instruction="Accept only if the official school site (.edu) or AACSB directory shows this exact official name; allow minor punctuation/capitalization variations."
    )

    # 8) main_campus_city_correct
    node_city = evaluator.add_leaf(
        id="main_campus_city_correct",
        desc="The city where the main campus is located is correctly identified",
        parent=overall,
        critical=True
    )
    city_claim = f"The main campus city for {uni_name} is {city}, Virginia."
    await evaluator.verify(
        claim=city_claim,
        node=node_city,
        sources=_unique_nonempty(extracted.sources.campus_city),
        additional_instruction="Use official sources to confirm the main campus location (city). If multiple campuses exist, ensure the main campus city is correctly identified."
    )

    # 9) in_state_percentage_correct
    node_instate = evaluator.add_leaf(
        id="in_state_percentage_correct",
        desc="The percentage of in-state students is accurately reported based on official university data for the stated context/term (if a term is specified)",
        parent=overall,
        critical=True
    )
    instate_claim = (
        f"At {uni_name}, in-state students constitute {instate_pct} of the student body."
        if instate_pct else f"The in-state percentage is correctly reported for {uni_name}."
    )
    await evaluator.verify(
        claim=instate_claim,
        node=node_instate,
        sources=_unique_nonempty(extracted.sources.in_state_percentage),
        additional_instruction="Confirm the percentage from an official institutional research/factbook/admissions (.edu) or state council page. Allow minor rounding (e.g., 66.7% ≈ 67%)."
    )

    # 10) total_university_enrollment_correct
    node_total = evaluator.add_leaf(
        id="total_university_enrollment_correct",
        desc="The total university enrollment figure is accurately provided and matches an official source for the stated context/term (if a term is specified)",
        parent=overall,
        critical=True
    )
    if ug_term and total_enroll:
        total_claim = f"The total university enrollment of {uni_name} in {ug_term} is {total_enroll}."
    elif total_enroll:
        total_claim = f"The total university enrollment of {uni_name} is {total_enroll}."
    else:
        total_claim = f"The total university enrollment figure reported for {uni_name} is accurate."
    await evaluator.verify(
        claim=total_claim,
        node=node_total,
        sources=_unique_nonempty(extracted.sources.total_enrollment),
        additional_instruction="Validate from official sources (.edu or state council). If a term is stated, ensure the number aligns with that term/year."
    )

    # 11) citations_provided_and_acceptable (parallel group)
    citations = evaluator.add_parallel(
        id="citations_provided_and_acceptable",
        desc="A verifying URL is provided for each required fact, and the sources are acceptable per constraints (official university sites and/or AACSB accreditation records)",
        parent=overall,
        critical=True
    )

    # 11a) citation_for_undergrad_enrollment
    evaluator.add_custom_node(
        result=_has_acceptable_citation(extracted.sources.undergrad_enrollment, "undergrad_enrollment"),
        id="citation_for_undergrad_enrollment",
        desc="A URL reference is provided that verifies the undergraduate enrollment number and its term/year",
        parent=citations,
        critical=True
    )

    # 11b) citation_for_business_school_name
    evaluator.add_custom_node(
        result=_has_acceptable_citation(extracted.sources.business_school_name, "business_school_name"),
        id="citation_for_business_school_name",
        desc="A URL reference is provided that verifies the official business school/college name",
        parent=citations,
        critical=True
    )

    # 11c) citation_for_aacsb_accreditation
    evaluator.add_custom_node(
        result=_has_acceptable_citation(extracted.sources.aacsb_accreditation, "aacsb_accreditation"),
        id="citation_for_aacsb_accreditation",
        desc="A URL reference is provided that verifies the business school/college holds AACSB accreditation",
        parent=citations,
        critical=True
    )

    # 11d) citation_for_campus_city
    evaluator.add_custom_node(
        result=_has_acceptable_citation(extracted.sources.campus_city, "campus_city"),
        id="citation_for_campus_city",
        desc="A URL reference is provided that verifies the main campus city",
        parent=citations,
        critical=True
    )

    # 11e) citation_for_in_state_percentage
    evaluator.add_custom_node(
        result=_has_acceptable_citation(extracted.sources.in_state_percentage, "in_state_percentage"),
        id="citation_for_in_state_percentage",
        desc="A URL reference is provided that verifies the in-state student percentage",
        parent=citations,
        critical=True
    )

    # 11f) citation_for_total_enrollment
    evaluator.add_custom_node(
        result=_has_acceptable_citation(extracted.sources.total_enrollment, "total_enrollment"),
        id="citation_for_total_enrollment",
        desc="A URL reference is provided that verifies the total university enrollment figure",
        parent=citations,
        critical=True
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
    Evaluate an answer for the 'VA AACSB largest undergraduate enrollment' task.
    Returns a structured evaluation summary dict.
    """
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_university_info(),
        template_class=UniversityExtraction,
        extraction_name="university_extraction",
    )

    # Optional: record minimal custom info for debugging
    evaluator.add_custom_info(
        info={
            "university_name": extracted.university_name,
            "ug_term": extracted.undergrad_enrollment_term,
            "ug_enrollment": extracted.undergrad_enrollment_number,
            "business_school_name": extracted.business_school_name,
            "main_campus_city": extracted.main_campus_city,
        },
        info_type="extraction_summary",
        info_name="extracted_core_fields"
    )

    # Build and run verification nodes
    await build_verification_tree(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()