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
TASK_ID = "college_of_computing_1990"
TASK_DESCRIPTION = """
In the early 1990s, a public research university in the United States established the nation's first or among the first college-level academic unit dedicated to computing, elevating computing from department status to college status. This college was officially established and began operations in 1990.

Identify this college and provide the following information:

1. The name of the public university where this college was established
2. The official name of the college of computing
3. Confirmation that it is indeed a college-level unit (not merely a department) at a public university
4. The founding dean who was appointed in 1990 and the year of their appointment
5. Information about any pre-existing research institutes or units that were incorporated into this new college at or near the time of its founding

Provide URL references from reliable sources (official university websites, academic publications, or reputable news sources) to support each piece of information.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class IncorporatedUnit(BaseModel):
    unit_name: Optional[str] = None
    description: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CollegeResearchExtraction(BaseModel):
    # Identification
    university_name: Optional[str] = None
    university_urls: List[str] = Field(default_factory=list)

    college_name: Optional[str] = None
    college_urls: List[str] = Field(default_factory=list)

    # Constraint checks
    public_research_university_urls: List[str] = Field(default_factory=list)
    college_level_confirmation_urls: List[str] = Field(default_factory=list)
    established_1990_urls: List[str] = Field(default_factory=list)
    among_first_claim_text: Optional[str] = None
    among_first_urls: List[str] = Field(default_factory=list)

    # Founding dean and tenure
    founding_dean_name: Optional[str] = None
    founding_dean_year: Optional[str] = None
    founding_dean_urls: List[str] = Field(default_factory=list)
    founding_dean_tenure_urls: List[str] = Field(default_factory=list)

    # Incorporated units at/near founding
    incorporated_units: List[IncorporatedUnit] = Field(default_factory=list)

    # PhD program in CS with requirements
    phd_cs_requirement_urls: List[str] = Field(default_factory=list)

    # Still operating as of 2025
    still_operates_year: Optional[str] = None
    still_operates_urls: List[str] = Field(default_factory=list)

    # Collected list of all sources mentioned in the answer
    all_source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_college_research() -> str:
    return """
    Extract from the answer the structured facts and the URLs that support them. Return null for any missing string field and an empty list [] for any missing URL list.

    Required fields:
    - university_name (string)
    - university_urls (array of URLs that support identifying the university)
    - college_name (string; the official college-level name, e.g., "College of Computing")
    - college_urls (array of URLs supporting the college's official name and identity)

    Constraint verification URLs (each should be an array; include the URLs the answer uses to support these points):
    - public_research_university_urls (array: URLs supporting that the university is a public research university in the USA)
    - college_level_confirmation_urls (array: URLs confirming the unit is a college-level entity, not just a department)
    - established_1990_urls (array: URLs confirming official establishment and operations began in 1990)
    - among_first_claim_text (string, the phrasing used such as "first" or "among the first"; null if not stated)
    - among_first_urls (array: URLs supporting that status)

    Founding dean:
    - founding_dean_name (string; name of founding dean)
    - founding_dean_year (string; the year of appointment)
    - founding_dean_urls (array: URLs supporting the founding dean appointment in 1990)
    - founding_dean_tenure_urls (array: URLs supporting that the founding dean served at least 10 years)

    Incorporated units (at or near founding):
    - incorporated_units (array of objects), each object:
        - unit_name (string)
        - description (string)
        - urls (array of URLs supporting incorporation into the new college at/near founding)

    PhD in CS:
    - phd_cs_requirement_urls (array: URLs pointing to official requirements or official description of requirements for the Computer Science PhD)

    Still operating as of 2025:
    - still_operates_year (string; e.g., "2025" if explicitly stated, otherwise null)
    - still_operates_urls (array: URLs supporting that the college still exists and operates as a college of computing as of 2025)

    Aggregate:
    - all_source_urls (array: every URL cited anywhere in the answer for this task; deduplicate; include only valid URLs; ensure protocol present)

    Important extraction rules:
    - Only extract URLs explicitly present in the answer. Do not invent URLs.
    - Deduplicate URLs. Include full URLs with protocol (prepend http:// if missing).
    - If an item is mentioned without a URL, leave the corresponding URL list empty.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _unique_nonempty(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls or []:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _merge_urls(*lists: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in lists:
        merged.extend(lst or [])
    return _unique_nonempty(merged)


def _fmt_url_list_for_claim(urls: List[str]) -> str:
    if not urls:
        return "[]"
    return "[" + ", ".join(urls) + "]"


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extraction: CollegeResearchExtraction
) -> None:
    # Top-level task node (critical, sequential)
    task_node = evaluator.add_sequential(
        id="College_Research_Task",
        desc="Identify the qualifying 1990-established college of computing at a U.S. public research university and provide all required details with reliable URL citations.",
        parent=evaluator.root,
        critical=True,
    )

    # 1) Identify target college (parallel, critical)
    identify_node = evaluator.add_parallel(
        id="Identify_Target_College",
        desc="Identify the specific university and the specific college of computing being claimed.",
        parent=task_node,
        critical=True,
    )

    # 1.a) University name with citation
    uni_urls = _merge_urls(extraction.university_urls, extraction.college_urls)
    uni_claim = (
        f"The public university identified is '{extraction.university_name}'. "
        f"Use the following URLs as evidence; if this URL list is empty, you must mark this claim as NOT supported: "
        f"{_fmt_url_list_for_claim(uni_urls)}"
    )
    uni_node = evaluator.add_leaf(
        id="University_Name_With_Citation",
        desc="State the name of the public university and provide a reliable URL supporting the identification.",
        parent=identify_node,
        critical=True,
    )
    await evaluator.verify(
        claim=uni_claim,
        node=uni_node,
        sources=uni_urls,
        additional_instruction=(
            "Verify the university's name from the provided URLs. "
            "If no URL is provided, or the URLs do not explicitly support the university identification, return Incorrect."
        ),
    )

    # 1.b) College name with citation
    college_urls = _unique_nonempty(extraction.college_urls)
    college_claim = (
        f"The official name of the college of computing is '{extraction.college_name}'. "
        f"Use the following URLs as evidence; if this URL list is empty, you must mark this claim as NOT supported: "
        f"{_fmt_url_list_for_claim(college_urls)}"
    )
    college_node = evaluator.add_leaf(
        id="College_Name_With_Citation",
        desc="State the official name of the college of computing and provide a reliable URL supporting the name.",
        parent=identify_node,
        critical=True,
    )
    await evaluator.verify(
        claim=college_claim,
        node=college_node,
        sources=college_urls,
        additional_instruction=(
            "Confirm the official name as shown on the provided page(s). "
            "Allow minor variations (case, punctuation) but the meaning must match. "
            "If no URL is provided, or the URLs do not explicitly support the official name, return Incorrect."
        ),
    )

    # 2) Verify constraints and provide requested details (parallel, critical)
    verify_node = evaluator.add_parallel(
        id="Verify_All_Constraints_And_Requested_Details",
        desc="Verify the institution meets all stated constraints and provide requested historical/organizational details, each supported by reliable sources.",
        parent=task_node,
        critical=True,
    )

    # 2.a) Public research university in the US
    pru_urls = _merge_urls(extraction.public_research_university_urls, extraction.university_urls)
    pru_claim = (
        f"{extraction.university_name} is a public research university in the United States. "
        f"Evidence URLs: {_fmt_url_list_for_claim(pru_urls)}. "
        f"If no valid supporting URL is provided or the pages do not confirm 'public research university' status, mark Incorrect."
    )
    pru_node = evaluator.add_leaf(
        id="Public_Research_University_US_With_Citation",
        desc="Confirm the institution is a public research university in the United States, with a reliable URL citation.",
        parent=verify_node,
        critical=True,
    )
    await evaluator.verify(
        claim=pru_claim,
        node=pru_node,
        sources=pru_urls,
        additional_instruction="Confirm both 'public' and 'research university' classification and that it is in the United States.",
    )

    # 2.b) College-level unit (not merely a department)
    clu_urls = _merge_urls(extraction.college_level_confirmation_urls, extraction.college_urls)
    clu_claim = (
        f"The unit '{extraction.college_name}' at {extraction.university_name} is a college-level entity (not merely a department). "
        f"Evidence URLs: {_fmt_url_list_for_claim(clu_urls)}. "
        f"If the URLs do not explicitly indicate college-level status (e.g., identified as a 'College' in the university's organizational structure), mark Incorrect."
    )
    clu_node = evaluator.add_leaf(
        id="College_Level_Unit_Not_Department_With_Citation",
        desc="Confirm the computing unit is organized as a college-level unit (not merely a department), with a reliable URL citation.",
        parent=verify_node,
        critical=True,
    )
    await evaluator.verify(
        claim=clu_claim,
        node=clu_node,
        sources=clu_urls,
        additional_instruction="Look for explicit mention that it is a 'College' or equivalent university-level academic unit.",
    )

    # 2.c) Established and began operations in 1990
    est_urls = _unique_nonempty(extraction.established_1990_urls)
    est_claim = (
        f"The {extraction.college_name} was officially established and began operations in 1990. "
        f"Evidence URLs: {_fmt_url_list_for_claim(est_urls)}. "
        f"If the URLs do not explicitly confirm 1990 establishment/operations, mark Incorrect."
    )
    est_node = evaluator.add_leaf(
        id="Established_And_Began_Operations_1990_With_Citation",
        desc="Confirm the college was officially established and began operations in 1990, with a reliable URL citation.",
        parent=verify_node,
        critical=True,
    )
    await evaluator.verify(
        claim=est_claim,
        node=est_node,
        sources=est_urls,
        additional_instruction="Confirm both 'official establishment' and that it began operations in 1990.",
    )

    # 2.d) Among the first to elevate computing to college status
    among_urls = _unique_nonempty(extraction.among_first_urls)
    among_text = extraction.among_first_claim_text or "among the first to elevate computing to college status"
    among_claim = (
        f"The {extraction.college_name} was {among_text} at a U.S. public university. "
        f"Evidence URLs: {_fmt_url_list_for_claim(among_urls)}. "
        f"If the URLs do not support this 'first/among the first' status, mark Incorrect."
    )
    among_node = evaluator.add_leaf(
        id="Among_First_To_Elevate_Computing_To_College_Status_With_Citation",
        desc="Confirm the college was among the first public universities in the U.S. to elevate computing to college status (separate from engineering/science colleges), with a reliable URL citation.",
        parent=verify_node,
        critical=True,
    )
    await evaluator.verify(
        claim=among_claim,
        node=among_node,
        sources=among_urls,
        additional_instruction="The page should clearly indicate 'first' or 'among the first' status for college-level computing.",
    )

    # 2.e) Founding dean appointed in 1990
    fd_urls = _unique_nonempty(extraction.founding_dean_urls)
    fd_claim = (
        f"In 1990, {extraction.founding_dean_name} was appointed as the founding dean of the {extraction.college_name}. "
        f"Evidence URLs: {_fmt_url_list_for_claim(fd_urls)}. "
        f"If the URLs do not confirm founding dean and the 1990 appointment year, mark Incorrect."
    )
    fd_node = evaluator.add_leaf(
        id="Founding_Dean_Appointed_1990_With_Citation",
        desc="Provide the founding dean’s name and confirm the appointment year is 1990, with a reliable URL citation.",
        parent=verify_node,
        critical=True,
    )
    await evaluator.verify(
        claim=fd_claim,
        node=fd_node,
        sources=fd_urls,
        additional_instruction="Confirm both the person's name and the 1990 appointment as founding dean.",
    )

    # 2.f) Founding dean served at least 10 years
    fdt_urls = _merge_urls(extraction.founding_dean_tenure_urls, extraction.founding_dean_urls)
    fdt_claim = (
        f"{extraction.founding_dean_name} served as dean for at least 10 years (through at least 2000). "
        f"Evidence URLs: {_fmt_url_list_for_claim(fdt_urls)}. "
        f"If the URLs do not support tenure length >= 10 years, mark Incorrect."
    )
    fdt_node = evaluator.add_leaf(
        id="Founding_Dean_Served_At_Least_10_Years_With_Citation",
        desc="Confirm the founding dean served for at least 10 years (through at least 2000), with a reliable URL citation.",
        parent=verify_node,
        critical=True,
    )
    await evaluator.verify(
        claim=fdt_claim,
        node=fdt_node,
        sources=fdt_urls,
        additional_instruction="Look for explicit years of service or start/end dates implying >= 10 years.",
    )

    # 2.g) Incorporated preexisting units at/near founding
    # Use the first provided incorporated unit for verification (at least one must be supported)
    inc_unit = extraction.incorporated_units[0] if extraction.incorporated_units else IncorporatedUnit()
    inc_urls = _unique_nonempty(inc_unit.urls)
    inc_name = inc_unit.unit_name or "a pre-existing research institute or unit"
    inc_claim = (
        f"The {extraction.college_name} incorporated {inc_name} at or near its founding (circa 1990). "
        f"Evidence URLs: {_fmt_url_list_for_claim(inc_urls)}. "
        f"If the URLs do not support incorporation at/near founding, mark Incorrect."
    )
    inc_node = evaluator.add_leaf(
        id="Incorporated_Preexisting_Units_With_Citation",
        desc="Describe at least one pre-existing research institute/unit or academic program that was incorporated into the new college at/near founding, with a reliable URL citation.",
        parent=verify_node,
        critical=True,
    )
    await evaluator.verify(
        claim=inc_claim,
        node=inc_node,
        sources=inc_urls,
        additional_instruction="Confirm that the referenced unit existed prior to the college and was incorporated into it at or near the time of founding.",
    )

    # 2.h) Offers PhD in Computer Science with requirements URL
    phd_urls = _unique_nonempty(extraction.phd_cs_requirement_urls)
    phd_claim = (
        f"{extraction.university_name} offers a doctoral (PhD) program in Computer Science and the provided page(s) include program requirements or official requirement descriptions. "
        f"Evidence URLs: {_fmt_url_list_for_claim(phd_urls)}. "
        f"If the URL(s) do not show an official requirements page/section, mark Incorrect."
    )
    phd_node = evaluator.add_leaf(
        id="Offers_PhD_In_Computer_Science_With_Requirements_Citation",
        desc="Confirm the institution offers a doctoral (PhD) program in computer science and provide a reliable URL pointing to program requirements (or an official description of requirements).",
        parent=verify_node,
        critical=True,
    )
    await evaluator.verify(
        claim=phd_claim,
        node=phd_node,
        sources=phd_urls,
        additional_instruction="The page should be official (university/college/school/department) and include requirement details (coursework, milestones, exams, etc.).",
    )

    # 2.i) Still operates as a college of computing as of 2025
    still_urls = _unique_nonempty(extraction.still_operates_urls)
    still_year = extraction.still_operates_year or "2025"
    still_claim = (
        f"As of {still_year}, the {extraction.college_name} still exists and operates as a college of computing. "
        f"Evidence URLs: {_fmt_url_list_for_claim(still_urls)}. "
        f"If the URLs do not plausibly indicate current operation as of 2025 (e.g., recent pages, current leadership pages, recent news), mark Incorrect."
    )
    still_node = evaluator.add_leaf(
        id="Still_Operates_As_College_Of_Computing_As_Of_2025_With_Citation",
        desc="Confirm the college still exists and operates as a college of computing as of 2025, with a reliable URL citation.",
        parent=verify_node,
        critical=True,
    )
    await evaluator.verify(
        claim=still_claim,
        node=still_node,
        sources=still_urls,
        additional_instruction="Use recency cues on pages (2024/2025 news, current faculty/leadership pages, current academic catalog) to judge current operation.",
    )

    # 2.j) Source reliability check (single leaf)
    all_urls = _merge_urls(
        extraction.all_source_urls,
        extraction.university_urls,
        extraction.college_urls,
        extraction.public_research_university_urls,
        extraction.college_level_confirmation_urls,
        extraction.established_1990_urls,
        extraction.among_first_urls,
        extraction.founding_dean_urls,
        extraction.founding_dean_tenure_urls,
        extraction.phd_cs_requirement_urls,
        extraction.still_operates_urls,
        *(u.urls for u in extraction.incorporated_units or [])
    )

    # Limit the length of the claim if too many URLs; but include as many as possible
    urls_for_claim = all_urls[:30]  # cap to keep prompt reasonable
    reliability_claim = (
        "All the following URLs are from acceptable reliable sources (official university websites, "
        "academic publications, or reputable news sources). "
        "Treat .edu domains as official university sources. Academic publications include journals, "
        "conference publishers, or institutional repositories. Reputable news includes well-known, "
        "credible media outlets. Do NOT treat personal blogs, random aggregators, or wikis as reliable. "
        f"URLs to assess: {_fmt_url_list_for_claim(urls_for_claim)}"
    )
    reliability_node = evaluator.add_leaf(
        id="Source_Reliability_Check",
        desc="All provided URLs are from acceptable reliable sources (official university websites, academic publications, or reputable news sources).",
        parent=verify_node,
        critical=True,
    )
    # Use simple verification: we ask the model to judge reliability based on domains/contexts listed in claim
    await evaluator.verify(
        claim=reliability_claim,
        node=reliability_node,
        sources=None,
        additional_instruction="Base your judgment on the domain types and the nature of the sources listed.",
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
    Evaluate an answer for the '1990 college of computing' research task using the Mind2Web2 framework.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root wrapper, we'll add a critical sequential task node under it
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_college_research(),
        template_class=CollegeResearchExtraction,
        extraction_name="college_research_extraction",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extraction)

    # Return standardized summary
    return evaluator.get_summary()