import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Set

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "big_ten_land_grant_1862"
TASK_DESCRIPTION = (
    "Identify four universities that satisfy all of the following criteria: "
    "(1) The university must be a current member of the Big Ten Conference (as of the 2025-2026 academic year); "
    "(2) The university must be an 1862 land-grant institution established under the Morrill Act of 1862; "
    "(3) The university must have been founded between 1855 and 1869 (inclusive); "
    "(4) The university's main campus must be between 1,800 and 6,500 acres in size; "
    "(5) The university must have a total student enrollment between 30,000 and 60,000 students (based on Fall 2024 or Fall 2025 data); "
    "(6) The university must have a College of Engineering that offers at least four of the following five engineering disciplines: "
    "Aerospace/Aeronautical Engineering, Mechanical Engineering, Electrical Engineering, Computer Engineering (or Computer Science Engineering), and Chemical Engineering. "
    "For each university you identify, provide: the university name, its founding year, its main campus size in acres, its current total student enrollment, "
    "the specific engineering disciplines it offers from the list above, and reference URLs supporting each piece of information."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityCitations(BaseModel):
    big_ten_urls: List[str] = Field(default_factory=list)
    land_grant_urls: List[str] = Field(default_factory=list)
    founding_year_urls: List[str] = Field(default_factory=list)
    campus_size_urls: List[str] = Field(default_factory=list)
    enrollment_urls: List[str] = Field(default_factory=list)
    engineering_urls: List[str] = Field(default_factory=list)
    public_status_urls: List[str] = Field(default_factory=list)
    single_campus_urls: List[str] = Field(default_factory=list)


class UniversityItem(BaseModel):
    name: Optional[str] = None
    founding_year: Optional[str] = None
    campus_size_acres: Optional[str] = None
    total_enrollment: Optional[str] = None
    enrollment_term: Optional[str] = None  # Expect "Fall 2024" or "Fall 2025"
    engineering_disciplines_offered: List[str] = Field(default_factory=list)  # From the specified list
    citations: UniversityCitations = Field(default_factory=UniversityCitations)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract all universities mentioned in the answer that the agent claims meet the task criteria. For each university, extract the following fields exactly as stated in the answer:

    Required Fields per University:
    1) name: The university's full name.
    2) founding_year: The founding year (a 4-digit year as written; if a range or multiple dates are given, choose the main founding year stated).
    3) campus_size_acres: The main campus size in acres as stated (allow numbers with commas, the words "acres", or approximate phrases).
    4) total_enrollment: The total student enrollment number as stated (allow numbers with commas or ranges; extract exactly as shown).
    5) enrollment_term: The enrollment term label as stated, expected to be "Fall 2024" or "Fall 2025". If an equivalent phrase is provided (e.g., "Fall 2025 census"), extract it literally.
    6) engineering_disciplines_offered: A list of discipline categories from the following set that the university offers via its College of Engineering (or equivalent):
       - "Aerospace Engineering"
       - "Aeronautical Engineering"
       - "Mechanical Engineering"
       - "Electrical Engineering"
       - "Computer Engineering"
       - "Computer Science Engineering"
       - "Chemical Engineering"
       IMPORTANT: Use the canonical names above as your list items. If the answer mentions a synonym like "Computer Science and Engineering", map it to "Computer Science Engineering". If it mentions "Electrical and Computer Engineering", include BOTH "Electrical Engineering" and "Computer Engineering".

    Citations per University (URLs explicitly present in the answer; extract actual URLs only):
    - citations.big_ten_urls: URLs that support Big Ten membership (as of 2025–2026).
    - citations.land_grant_urls: URLs that support 1862 Morrill Act land-grant status.
    - citations.founding_year_urls: URLs that support the founding year.
    - citations.campus_size_urls: URLs that support main campus size in acres.
    - citations.enrollment_urls: URLs that support the stated total enrollment for Fall 2024 or Fall 2025 (ensure the term is clear).
    - citations.engineering_urls: URLs that support the claimed engineering disciplines offered (may be multiple pages).
    - citations.public_status_urls: URLs that support that the institution is public.
    - citations.single_campus_urls: URLs that support that the institution operates a single main flagship campus rather than a multi-campus system.

    Notes:
    - Extract only what is explicitly present in the answer. Do not infer or add new information.
    - If any field is missing for a university, set that field to null or an empty list accordingly.
    - Extract all universities mentioned; the evaluator may later select the first four for verification.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_university_name(name: Optional[str]) -> str:
    if not name:
        return ""
    s = name.lower().strip()
    s = re.sub(r"[\s\-\.,&]+", " ", s)
    s = s.replace(" the ", " ")
    s = s.strip()
    return s


def parse_int_from_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    nums = re.findall(r"\d{1,6}", text.replace(",", ""))
    if not nums:
        return None
    # Choose the largest number present, which usually corresponds to totals or acreage
    try:
        return max(int(n) for n in nums)
    except Exception:
        return None


def enrollment_term_valid(term: Optional[str]) -> bool:
    if not term:
        return False
    t = term.lower()
    return "fall 2024" in t or "fall 2025" in t


ALLOWED_DISCIPLINE_CANONICAL = {
    "aerospace engineering": {"Aerospace"},
    "aeronautical engineering": {"Aerospace"},
    "mechanical engineering": {"Mechanical"},
    "electrical engineering": {"Electrical"},
    "computer engineering": {"Computer"},
    "computer science engineering": {"Computer"},
    "chemical engineering": {"Chemical"},
}


def discipline_categories_covered(disciplines: List[str]) -> Set[str]:
    covered: Set[str] = set()
    for d in disciplines:
        key = d.lower().strip()
        # Handle combined department names mapped in extraction (E&CE -> two entries)
        if key in ALLOWED_DISCIPLINE_CANONICAL:
            covered |= ALLOWED_DISCIPLINE_CANONICAL[key]
        else:
            # Be lenient for common synonyms that the extractor might not perfectly map
            if "electrical and computer" in key:
                covered |= {"Electrical", "Computer"}
            elif "computer science and engineering" in key:
                covered |= {"Computer"}
            elif "aerospace" in key or "aeronautical" in key:
                covered |= {"Aerospace"}
            elif "mechanical" in key:
                covered |= {"Mechanical"}
            elif "electrical" in key:
                covered |= {"Electrical"}
            elif "chemical" in key:
                covered |= {"Chemical"}
            elif "computer" in key:
                covered |= {"Computer"}
    return covered


def at_least_four_of_five(covered: Set[str]) -> bool:
    # The five canonical categories: Aerospace, Mechanical, Electrical, Computer, Chemical
    canonical = {"Aerospace", "Mechanical", "Electrical", "Computer", "Chemical"}
    return len(covered & canonical) >= 4


# --------------------------------------------------------------------------- #
# Verification for a single university                                        #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    index_one_based: int,
) -> None:
    # Create University node (non-critical; partial credit allowed across universities)
    uni_node = evaluator.add_parallel(
        id=f"university_{index_one_based}",
        desc=f"University #{index_one_based}: eligibility constraints satisfied; required reporting fields and citations provided",
        parent=parent_node,
        critical=False,
    )

    # --- Required outputs (critical parallel) ---
    outputs_node = evaluator.add_parallel(
        id=f"u{index_one_based}_required_outputs",
        desc=f"Required fields for University #{index_one_based} are provided",
        parent=uni_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(uni.name and uni.name.strip()),
        id=f"u{index_one_based}_output_name",
        desc=f"University #{index_one_based} name is provided",
        parent=outputs_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(uni.founding_year and uni.founding_year.strip()),
        id=f"u{index_one_based}_output_founding_year",
        desc=f"University #{index_one_based} founding year is provided",
        parent=outputs_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(uni.campus_size_acres and uni.campus_size_acres.strip()),
        id=f"u{index_one_based}_output_campus_size_acres",
        desc=f"University #{index_one_based} main campus size in acres is provided",
        parent=outputs_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(uni.total_enrollment and uni.total_enrollment.strip()) and enrollment_term_valid(uni.enrollment_term),
        id=f"u{index_one_based}_output_total_enrollment",
        desc=f"University #{index_one_based} total student enrollment is provided and labeled as Fall 2024 or Fall 2025",
        parent=outputs_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(uni.engineering_disciplines_offered),
        id=f"u{index_one_based}_output_engineering_list",
        desc=f"University #{index_one_based} includes which of the listed engineering disciplines it offers",
        parent=outputs_node,
        critical=True,
    )

    # --- Eligibility criteria (critical parallel) ---
    elig_node = evaluator.add_parallel(
        id=f"u{index_one_based}_eligibility_criteria",
        desc=f"University #{index_one_based} satisfies all eligibility constraints",
        parent=uni_node,
        critical=True,
    )

    # Big Ten membership
    big_ten_leaf = evaluator.add_leaf(
        id=f"u{index_one_based}_big_ten_membership",
        desc=f"University #{index_one_based} is a current member of the Big Ten Conference as of the 2025–2026 academic year",
        parent=elig_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"As of the 2025–2026 academic year, {uni.name or 'the university'} is a current member of the Big Ten Conference.",
        node=big_ten_leaf,
        sources=uni.citations.big_ten_urls,
        additional_instruction="If no URLs are provided, judge NOT SUPPORTED. Prefer official Big Ten site or institution/athletics pages; membership should reflect the 2025–2026 lineup.",
    )

    # Land-grant (1862 Morrill Act)
    land_grant_leaf = evaluator.add_leaf(
        id=f"u{index_one_based}_land_grant_status",
        desc=f"University #{index_one_based} is an 1862 land-grant institution established under the Morrill Act of 1862",
        parent=elig_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{uni.name or 'the university'} is an 1862 land-grant institution established under the Morrill Act of 1862.",
        node=land_grant_leaf,
        sources=uni.citations.land_grant_urls,
        additional_instruction="If no URLs are provided, judge NOT SUPPORTED. Distinguish 1862 Morrill Act land-grant from 1890 land-grant.",
    )

    # Founding year in range [1855, 1869]
    founding_year_int = parse_int_from_text(uni.founding_year)
    evaluator.add_custom_node(
        result=founding_year_int is not None and 1855 <= founding_year_int <= 1869,
        id=f"u{index_one_based}_founding_year_range",
        desc=f"University #{index_one_based} founding year is between 1855 and 1869 inclusive",
        parent=elig_node,
        critical=True,
    )

    # Campus size range [1800, 6500] acres
    campus_size_int = parse_int_from_text(uni.campus_size_acres)
    evaluator.add_custom_node(
        result=campus_size_int is not None and 1800 <= campus_size_int <= 6500,
        id=f"u{index_one_based}_campus_size_range",
        desc=f"University #{index_one_based} main campus size is between 1,800 and 6,500 acres inclusive",
        parent=elig_node,
        critical=True,
    )

    # Enrollment range [30000, 60000] and term Fall 2024 or Fall 2025
    enrollment_int = parse_int_from_text(uni.total_enrollment)
    evaluator.add_custom_node(
        result=(enrollment_int is not None and 30000 <= enrollment_int <= 60000) and enrollment_term_valid(uni.enrollment_term),
        id=f"u{index_one_based}_enrollment_range_and_term",
        desc=f"University #{index_one_based} total student enrollment is between 30,000 and 60,000 inclusive, based on Fall 2024 or Fall 2025 data",
        parent=elig_node,
        critical=True,
    )

    # Engineering disciplines: at least 4 of 5 categories
    covered = discipline_categories_covered(uni.engineering_disciplines_offered)
    evaluator.add_custom_node(
        result=at_least_four_of_five(covered),
        id=f"u{index_one_based}_engineering_disciplines",
        desc=f"University #{index_one_based} has a College of Engineering offering at least four of the specified five disciplines",
        parent=elig_node,
        critical=True,
    )

    # Public institution
    public_leaf = evaluator.add_leaf(
        id=f"u{index_one_based}_public_institution",
        desc=f"University #{index_one_based} is a public institution",
        parent=elig_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{uni.name or 'the university'} is a public institution.",
        node=public_leaf,
        sources=uni.citations.public_status_urls,
        additional_instruction="If no URLs are provided, judge NOT SUPPORTED. Accept 'public', 'state', or equivalent official classification.",
    )

    # Single flagship campus (not a multi-campus system)
    single_leaf = evaluator.add_leaf(
        id=f"u{index_one_based}_single_flagship_campus",
        desc=f"University #{index_one_based} operates a single main flagship campus rather than a multi-campus system",
        parent=elig_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{uni.name or 'the university'} operates a single main flagship campus and is not a multi-campus system.",
        node=single_leaf,
        sources=uni.citations.single_campus_urls,
        additional_instruction="If no URLs are provided, judge NOT SUPPORTED. Distinguish 'flagship campus' vs. multi-campus systems; supporting pages should reference the single main campus.",
    )

    # --- Required citations (critical parallel) ---
    cites_node = evaluator.add_parallel(
        id=f"u{index_one_based}_required_citations",
        desc=f"Reference URLs are provided to support each required piece of information for University #{index_one_based}",
        parent=uni_node,
        critical=True,
    )

    # Big Ten membership citation support
    cite_big_ten_leaf = evaluator.add_leaf(
        id=f"u{index_one_based}_cite_big_ten",
        desc=f"At least one URL supports Big Ten membership (as of 2025–2026)",
        parent=cites_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"As of 2025–2026, {uni.name or 'the university'} is a Big Ten member.",
        node=cite_big_ten_leaf,
        sources=uni.citations.big_ten_urls,
        additional_instruction="If no URLs are provided, judge NOT SUPPORTED. Verify membership status using provided URLs.",
    )

    # Land-grant citation support
    cite_land_leaf = evaluator.add_leaf(
        id=f"u{index_one_based}_cite_land_grant",
        desc=f"At least one URL supports 1862 Morrill Act land-grant status",
        parent=cites_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{uni.name or 'the university'} is an 1862 Morrill Act land-grant institution.",
        node=cite_land_leaf,
        sources=uni.citations.land_grant_urls,
        additional_instruction="If no URLs are provided, judge NOT SUPPORTED. Verify the 1862 land-grant designation using provided URLs.",
    )

    # Founding year citation support
    cite_found_leaf = evaluator.add_leaf(
        id=f"u{index_one_based}_cite_founding_year",
        desc=f"At least one URL supports the founding year",
        parent=cites_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The founding year of {uni.name or 'the university'} is {uni.founding_year or '[missing]'}.",
        node=cite_found_leaf,
        sources=uni.citations.founding_year_urls,
        additional_instruction="If no URLs are provided, judge NOT SUPPORTED. Use authoritative sources (official university history page or trusted encyclopedic pages).",
    )

    # Campus size citation support
    cite_campus_leaf = evaluator.add_leaf(
        id=f"u{index_one_based}_cite_campus_size",
        desc=f"At least one URL supports the main campus size (acres)",
        parent=cites_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The main campus size of {uni.name or 'the university'} is {uni.campus_size_acres or '[missing]'} acres.",
        node=cite_campus_leaf,
        sources=uni.citations.campus_size_urls,
        additional_instruction="If no URLs are provided, judge NOT SUPPORTED. Focus on main campus acreage only, not system-wide.",
    )

    # Enrollment citation support
    cite_enroll_leaf = evaluator.add_leaf(
        id=f"u{index_one_based}_cite_enrollment",
        desc=f"At least one URL supports the total enrollment for Fall 2024 or Fall 2025",
        parent=cites_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The total enrollment of {uni.name or 'the university'} in {uni.enrollment_term or '[missing term]'} is {uni.total_enrollment or '[missing]'} students.",
        node=cite_enroll_leaf,
        sources=uni.citations.enrollment_urls,
        additional_instruction="If no URLs are provided, judge NOT SUPPORTED. Confirm census/official fall headcount using provided URLs.",
    )

    # Engineering disciplines citation support
    cite_eng_leaf = evaluator.add_leaf(
        id=f"u{index_one_based}_cite_engineering_disciplines",
        desc=f"At least one URL supports the claimed engineering disciplines offered",
        parent=cites_node,
        critical=True,
    )
    eng_list_text = ", ".join(uni.engineering_disciplines_offered) if uni.engineering_disciplines_offered else "[none]"
    await evaluator.verify(
        claim=(
            f"The College of Engineering at {uni.name or 'the university'} offers the following disciplines: {eng_list_text}. "
            f"These map to the categories Aerospace/Aeronautical, Mechanical, Electrical, Computer (or Computer Science), and Chemical."
        ),
        node=cite_eng_leaf,
        sources=uni.citations.engineering_urls,
        additional_instruction="If no URLs are provided, judge NOT SUPPORTED. Allow synonyms like 'Electrical & Computer Engineering' (covers Electrical and Computer) and 'Computer Science and Engineering' (Computer).",
    )

    # Public status citation support
    cite_public_leaf = evaluator.add_leaf(
        id=f"u{index_one_based}_cite_public_status",
        desc=f"At least one URL supports that the institution is public",
        parent=cites_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{uni.name or 'the university'} is a public institution.",
        node=cite_public_leaf,
        sources=uni.citations.public_status_urls,
        additional_instruction="If no URLs are provided, judge NOT SUPPORTED. Confirm public/state status using provided URLs.",
    )

    # Single campus citation support
    cite_single_leaf = evaluator.add_leaf(
        id=f"u{index_one_based}_cite_single_campus_status",
        desc=f"At least one URL supports that the institution is not a multi-campus system (single main flagship campus)",
        parent=cites_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{uni.name or 'the university'} is not a multi-campus system and operates a single main flagship campus.",
        node=cite_single_leaf,
        sources=uni.citations.single_campus_urls,
        additional_instruction="If no URLs are provided, judge NOT SUPPORTED. The source should indicate a single main flagship campus rather than multiple co-equal campuses.",
    )


# --------------------------------------------------------------------------- #
# Count and distinctness checks                                               #
# --------------------------------------------------------------------------- #
def compute_distinctness(universities: List[UniversityItem]) -> bool:
    names = [normalize_university_name(u.name) for u in universities if u.name]
    if len(names) < len(universities):
        # Some missing names -> cannot be distinct; count as non-distinct
        return False
    return len(set(names)) == len(names)


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
) -> Dict[str, Any]:
    # Initialize evaluator (root is non-critical by framework design)
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

    # Extract structured universities data
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_structured",
    )

    # Count and distinctness (critical parallel group)
    cd_node = evaluator.add_parallel(
        id="count_and_distinctness",
        desc="Response includes exactly four universities and they are all distinct",
        parent=root,
        critical=True,
    )
    # Exactly four universities
    evaluator.add_custom_node(
        result=len(extracted.universities) == 4,
        id="exactly_four_universities",
        desc="Response identifies exactly four universities (not fewer or more)",
        parent=cd_node,
        critical=True,
    )
    # Distinct universities
    evaluator.add_custom_node(
        result=compute_distinctness(extracted.universities),
        id="universities_are_distinct",
        desc="All four identified universities are distinct (no duplicates/aliases of the same institution)",
        parent=cd_node,
        critical=True,
    )

    # Choose up to first four items for detailed verification; pad with empty items if fewer
    selected: List[UniversityItem] = list(extracted.universities[:4])
    while len(selected) < 4:
        selected.append(UniversityItem())

    # Build university verification subtrees (non-critical each)
    for i, uni in enumerate(selected, start=1):
        await verify_university(evaluator, root, uni, i)

    # Return structured summary
    return evaluator.get_summary()