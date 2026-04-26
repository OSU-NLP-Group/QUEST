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
TASK_ID = "ga_superintendent_gsba_2026"
TASK_DESCRIPTION = (
    "You are researching superintendent career opportunities in Georgia for professional consideration. "
    "Identify three school district superintendent positions in Georgia that are currently accepting applications with deadlines in March 2026 or later, "
    "as listed by the Georgia School Boards Association (GSBA) superintendent search services. For each position, provide the following information: "
    "Basic Information: The school district name and the county where it is located; The exact application deadline (including date and time); "
    "The current search status as indicated by GSBA. District Characteristics: The district's total student enrollment; The district's annual operating budget; "
    "The total number of schools in the district (broken down by elementary, middle, and high schools if available). "
    "Qualification Requirements: The minimum degree requirement stated in the vacancy announcement; "
    "The type of Georgia Professional Standards Commission certification required (or eligibility statement); "
    "Any stated minimum years of administrative or leadership experience required. Verification: A direct link to either the official GSBA vacancy page "
    "for that position or the district's official superintendent search page. All information must be current as of March 2026 and verifiable through "
    "official Georgia School Boards Association sources or the school districts' official websites."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PositionItem(BaseModel):
    district_name: Optional[str] = None
    county: Optional[str] = None
    deadline: Optional[str] = None  # exact application deadline as stated (date and time if provided)
    search_status: Optional[str] = None  # GSBA-indicated search status (e.g., Accepting Applications, Screening, Closed)
    student_enrollment: Optional[str] = None
    annual_operating_budget: Optional[str] = None
    num_schools_total: Optional[str] = None
    num_elem_schools: Optional[str] = None
    num_middle_schools: Optional[str] = None
    num_high_schools: Optional[str] = None
    min_degree_requirement: Optional[str] = None
    gapsc_certification: Optional[str] = None  # e.g., Educational Leadership Tier II or eligibility
    min_years_experience: Optional[str] = None
    state: Optional[str] = None  # e.g., GA or Georgia
    currently_accepting: Optional[str] = None  # e.g., "accepting applications", "open"
    application_portal: Optional[str] = None  # e.g., GSBA Revelus
    verification_url: Optional[str] = None  # GSBA vacancy page or district official search page for this position
    additional_urls: List[str] = Field(default_factory=list)  # other official district or GSBA URLs used as citations


class PositionsExtraction(BaseModel):
    positions: List[PositionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
    From the provided answer, extract up to five superintendent vacancy items described. For each position, return the following fields exactly as stated in the answer (use strings for values, do not invent or infer):

    - district_name
    - county
    - deadline  (exact application deadline text including date and time if present)
    - search_status  (GSBA-indicated search status such as "Accepting Applications", "Screening", "Closed", etc.)
    - student_enrollment  (total district enrollment; if a range or approximation is given, keep that text)
    - annual_operating_budget  (keep as provided, e.g., "$120M", "approximately $100 million")
    - num_schools_total
    - num_elem_schools
    - num_middle_schools
    - num_high_schools
    - min_degree_requirement  (e.g., Master's, Specialist/Ed.S., Doctorate/Ed.D./Ph.D.)
    - gapsc_certification  (the Georgia PSC certification requirement or eligibility statement, e.g., "Educational Leadership Tier II or eligible")
    - min_years_experience  (e.g., "3 years", "5+ years"; if the answer states no minimum, capture that text)
    - state  (if the answer mentions GA/Georgia for the district)
    - currently_accepting  (verbatim text indicating accepting applications/open if present)
    - application_portal  (e.g., "GSBA Revelus", "Revelus")
    - verification_url  (the direct, primary URL for the vacancy: the GSBA vacancy/listing page for this position or the district's official superintendent search page)
    - additional_urls  (an array of any other official GSBA or district URLs cited in the answer that support the facts. Do not include third-party news or blogs.)

    Rules:
    - Extract URLs exactly as shown in the answer. If none is given for a field, return null (or an empty array for additional_urls).
    - Do not fabricate fields. Only capture what is explicitly present in the answer.
    - If the answer includes more than three positions, still extract them; the evaluator will only consider the first three.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _normalize_name(name: Optional[str]) -> str:
    return (name or "").strip().lower()


def _urls_for(p: PositionItem) -> List[str]:
    urls: List[str] = []
    if p.verification_url and p.verification_url.strip():
        urls.append(p.verification_url.strip())
    for u in p.additional_urls or []:
        if u and isinstance(u, str) and u.strip():
            if u.strip() not in urls:
                urls.append(u.strip())
    return urls


# --------------------------------------------------------------------------- #
# Verification logic per position                                             #
# --------------------------------------------------------------------------- #
async def verify_position(
    evaluator: Evaluator,
    parent_node,
    position: PositionItem,
    index: int,
) -> None:
    # Position parent node (non-critical overall to allow partial scoring across different positions)
    pos_node = evaluator.add_parallel(
        id=f"position_{index}",
        desc=f"{ordinal(index)} superintendent position meeting criteria",
        parent=parent_node,
        critical=False,
    )

    urls_all = _urls_for(position)

    # 1) Eligibility Criteria (critical group)
    elig_node = evaluator.add_parallel(
        id=f"position_{index}_eligibility",
        desc="Position satisfies the required eligibility constraints.",
        parent=pos_node,
        critical=True,
    )

    # 1.a Located in Georgia
    leaf_loc = evaluator.add_leaf(
        id=f"position_{index}_located_in_georgia",
        desc="District is located in the state of Georgia (USA).",
        parent=elig_node,
        critical=True,
    )
    claim_loc = f"The school district '{position.district_name or 'UNKNOWN'}' is located in Georgia (GA), USA."
    await evaluator.verify(
        claim=claim_loc,
        node=leaf_loc,
        sources=urls_all,
        additional_instruction="Confirm the district is in Georgia (GA). Official district or GSBA pages should state Georgia explicitly or imply GA via address/domain."
    )

    # 1.b GSBA Listed Search Services
    leaf_gsba = evaluator.add_leaf(
        id=f"position_{index}_gsba_listed",
        desc="Position is listed by GSBA superintendent search services.",
        parent=elig_node,
        critical=True,
    )
    claim_gsba = (
        f"The superintendent vacancy for '{position.district_name or 'UNKNOWN'}' is listed by the Georgia School Boards Association (GSBA) "
        "superintendent search services (e.g., on GSBA's website or GSBA's Revelus portal)."
    )
    await evaluator.verify(
        claim=claim_gsba,
        node=leaf_gsba,
        sources=urls_all,
        additional_instruction="Treat GSBA-branded Revelus listings or pages on gsba.com (or related GSBA executive search pages) as GSBA listings."
    )

    # 1.c Currently Accepting Applications
    leaf_open = evaluator.add_leaf(
        id=f"position_{index}_currently_accepting",
        desc="Position is currently accepting applications (open/accepting applications).",
        parent=elig_node,
        critical=True,
    )
    open_text = position.currently_accepting or "accepting applications"
    claim_open = (
        f"The posting indicates that applications are currently being accepted (i.e., 'open' or 'accepting applications') for the "
        f"superintendent position at '{position.district_name or 'UNKNOWN'}'."
    )
    await evaluator.verify(
        claim=claim_open,
        node=leaf_open,
        sources=urls_all,
        additional_instruction=f"Look for phrases like 'Accepting Applications', 'Open', 'Apply by', etc., ideally referencing March 2026 or later. The answer text suggests '{open_text}'."
    )

    # 1.d Application Deadline in March 2026 or later, with exact date/time
    leaf_deadline = evaluator.add_leaf(
        id=f"position_{index}_deadline_march2026_or_later",
        desc="Provides the exact application deadline including date and time, and it is in March 2026 or later.",
        parent=elig_node,
        critical=True,
    )
    deadline_text = position.deadline or "UNKNOWN DEADLINE"
    claim_deadline = (
        f"The application deadline for the '{position.district_name or 'UNKNOWN'}' superintendent position is '{deadline_text}', "
        "and that deadline falls in March 2026 or later (>= 2026-03-01)."
    )
    await evaluator.verify(
        claim=claim_deadline,
        node=leaf_deadline,
        sources=urls_all,
        additional_instruction="Confirm that the page clearly states an application deadline that occurs on or after March 1, 2026. Minor format variations are acceptable."
    )

    # 1.e Applications submitted through GSBA Revelus
    leaf_revelus = evaluator.add_leaf(
        id=f"position_{index}_applications_revelus",
        desc="Posting indicates applications are submitted through the GSBA Revelus online application system.",
        parent=elig_node,
        critical=True,
    )
    claim_revelus = (
        f"The posting for '{position.district_name or 'UNKNOWN'}' indicates that applications are submitted via GSBA's Revelus online system."
    )
    await evaluator.verify(
        claim=claim_revelus,
        node=leaf_revelus,
        sources=urls_all,
        additional_instruction="Look for 'Revelus' branding, links to Revelus, or explicit instructions to apply via the GSBA Revelus system."
    )

    # 2) Required Position Details (critical group)
    rqd_node = evaluator.add_parallel(
        id=f"position_{index}_required_details",
        desc="Provides all required district/role details for the position.",
        parent=pos_node,
        critical=True,
    )

    # 2.a District Name
    leaf_dname = evaluator.add_leaf(
        id=f"position_{index}_district_name",
        desc="Provides the school district name.",
        parent=rqd_node,
        critical=True,
    )
    claim_dname = f"The district name for this vacancy is '{position.district_name or 'UNKNOWN'}'."
    await evaluator.verify(
        claim=claim_dname,
        node=leaf_dname,
        sources=urls_all,
        additional_instruction="Verify the district name as stated on the official GSBA vacancy page or the district's official page for the superintendent search."
    )

    # 2.b County
    leaf_county = evaluator.add_leaf(
        id=f"position_{index}_county",
        desc="Provides the county where the district is located.",
        parent=rqd_node,
        critical=True,
    )
    claim_county = f"The district is located in {position.county or 'UNKNOWN'} County, Georgia."
    await evaluator.verify(
        claim=claim_county,
        node=leaf_county,
        sources=urls_all,
        additional_instruction="Confirm the county location using official district or GSBA sources. Allow common naming variations."
    )

    # 2.c GSBA Search Status
    leaf_status = evaluator.add_leaf(
        id=f"position_{index}_gsba_search_status",
        desc="Provides the current GSBA-indicated search status for the position.",
        parent=rqd_node,
        critical=True,
    )
    claim_status = f"GSBA lists the search status for this position as '{position.search_status or 'UNKNOWN'}'."
    await evaluator.verify(
        claim=claim_status,
        node=leaf_status,
        sources=urls_all,
        additional_instruction="Look for a status label like 'Accepting Applications', 'Screening', 'Closed', or similar on the GSBA posting."
    )

    # 2.d Student Enrollment
    leaf_enroll = evaluator.add_leaf(
        id=f"position_{index}_student_enrollment",
        desc="Provides the district's total student enrollment.",
        parent=rqd_node,
        critical=True,
    )
    claim_enroll = f"The district's total student enrollment is {position.student_enrollment or 'UNKNOWN'}."
    await evaluator.verify(
        claim=claim_enroll,
        node=leaf_enroll,
        sources=urls_all,
        additional_instruction="Accept approximate figures stated on official district or GSBA pages (e.g., 'about 8,500 students')."
    )

    # 2.e Annual Operating Budget
    leaf_budget = evaluator.add_leaf(
        id=f"position_{index}_annual_budget",
        desc="Provides the district's annual operating budget.",
        parent=rqd_node,
        critical=True,
    )
    claim_budget = f"The district's annual operating budget is {position.annual_operating_budget or 'UNKNOWN'}."
    await evaluator.verify(
        claim=claim_budget,
        node=leaf_budget,
        sources=urls_all,
        additional_instruction="Accept reasonable formats such as '$120M', '$120,000,000', or 'approximately $120 million' if supported by official sources."
    )

    # 2.f Number of schools (with breakdown if available)
    leaf_schools = evaluator.add_leaf(
        id=f"position_{index}_num_schools",
        desc="Provides total number of schools and includes elementary/middle/high breakdown if available from sources.",
        parent=rqd_node,
        critical=True,
    )
    # Construct a flexible claim depending on available details
    total_txt = position.num_schools_total or "UNKNOWN"
    parts = [f"The district operates {total_txt} schools"]
    breakdown_bits = []
    if position.num_elem_schools:
        breakdown_bits.append(f"{position.num_elem_schools} elementary")
    if position.num_middle_schools:
        breakdown_bits.append(f"{position.num_middle_schools} middle")
    if position.num_high_schools:
        breakdown_bits.append(f"{position.num_high_schools} high")
    if breakdown_bits:
        parts.append("including " + ", ".join(breakdown_bits))
    claim_schools = " ".join(parts) + "."
    await evaluator.verify(
        claim=claim_schools,
        node=leaf_schools,
        sources=urls_all,
        additional_instruction="Confirm counts using official district or GSBA pages. If only total is available, that's acceptable; include breakdown when an official source provides it."
    )

    # 2.g Minimum Degree Requirement
    leaf_degree = evaluator.add_leaf(
        id=f"position_{index}_min_degree",
        desc="States the minimum degree requirement from the vacancy announcement.",
        parent=rqd_node,
        critical=True,
    )
    claim_degree = f"The minimum degree requirement stated is: {position.min_degree_requirement or 'not specified'}."
    await evaluator.verify(
        claim=claim_degree,
        node=leaf_degree,
        sources=urls_all,
        additional_instruction="Look for degree requirements such as Master's, Specialist/Ed.S., Doctorate/Ed.D./Ph.D. Prefer exact language from the posting."
    )

    # 2.h GaPSC Certification Type or Eligibility
    leaf_gapsc = evaluator.add_leaf(
        id=f"position_{index}_gapsc_cert",
        desc="States the GaPSC certification type required (or eligibility statement) from the posting, and it must be Educational Leadership Tier II certification or eligibility for such certification (per constraints).",
        parent=rqd_node,
        critical=True,
    )
    cert_text = position.gapsc_certification or "not specified"
    claim_gapsc = (
        f"The posting states a Georgia PSC Educational Leadership certification requirement or eligibility (e.g., Tier II), described as: {cert_text}."
    )
    await evaluator.verify(
        claim=claim_gapsc,
        node=leaf_gapsc,
        sources=urls_all,
        additional_instruction="Accept language indicating Educational Leadership certification or eligibility (Tier II or equivalent terminology). Check GSBA/district sources."
    )

    # 2.i Minimum Administrative/Leadership Experience
    leaf_exp = evaluator.add_leaf(
        id=f"position_{index}_min_experience",
        desc="Reports any stated minimum years of administrative/leadership experience required, or states that no minimum years are specified.",
        parent=rqd_node,
        critical=True,
    )
    exp_text = position.min_years_experience or "no minimum specified"
    claim_exp = f"The posting indicates a minimum administrative/leadership experience requirement of {exp_text}."
    await evaluator.verify(
        claim=claim_exp,
        node=leaf_exp,
        sources=urls_all,
        additional_instruction="Verify explicit minimum years (e.g., 3, 5) or confirm the posting states no minimum years specified."
    )

    # 3) Verification & Sourcing (critical group)
    ver_node = evaluator.add_parallel(
        id=f"position_{index}_verification",
        desc="Provides direct verification link(s) and traceable sourcing.",
        parent=pos_node,
        critical=True,
    )

    # 3.a Direct Verification URL is official (GSBA vacancy or district search page)
    leaf_direct = evaluator.add_leaf(
        id=f"position_{index}_direct_verification_url",
        desc="Provides a direct URL to either (a) the official GSBA vacancy/listing page for the position OR (b) the district’s official superintendent search page.",
        parent=ver_node,
        critical=True,
    )
    claim_direct = (
        "This page is an official GSBA vacancy/listing page for the superintendent position or an official page on the district's website "
        "for the superintendent search/vacancy."
    )
    await evaluator.verify(
        claim=claim_direct,
        node=leaf_direct,
        sources=position.verification_url or None,
        additional_instruction="Treat GSBA-branded Revelus or pages on gsba.com as official GSBA pages. District official domains are official district sources."
    )

    # 3.b Citations map to claims (traceability)
    leaf_citations = evaluator.add_leaf(
        id=f"position_{index}_citations_map_to_claims",
        desc="Provides official source URL citations such that each required fact is traceable to GSBA or the district’s official website.",
        parent=ver_node,
        critical=True,
    )
    claim_citations = (
        f"For the '{position.district_name or 'UNKNOWN'}' position, the combined official URLs provided contain direct evidence for all required facts "
        "(district name, county, GSBA search status, application deadline, open/accepting status, submission portal/Revelus, "
        "enrollment, budget, number of schools with breakdown if available, minimum degree, GaPSC certification, and minimum experience)."
    )
    await evaluator.verify(
        claim=claim_citations,
        node=leaf_citations,
        sources=urls_all,
        additional_instruction="Confirm that for each listed fact, at least one of the provided official URLs (GSBA or district) contains that information. "
                              "Allow reasonable numeric approximations and common naming variations."
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
    # Initialize evaluator with a parallel root strategy (per rubric root)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify three Georgia school district superintendent positions listed by GSBA with deadlines in March 2026 or later, and provide required details with official verification.",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured positions info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction",
    )

    # Record simple statistics about extracted data
    all_positions = extracted.positions or []
    evaluator.add_custom_info(
        info={
            "total_positions_found_in_answer": len(all_positions),
            "positions_used_for_scoring": min(3, len(all_positions)),
        },
        info_type="extraction_stats"
    )

    # Filter only the first 3 positions for evaluation; pad with empty if fewer
    positions: List[PositionItem] = list(all_positions[:3])
    while len(positions) < 3:
        positions.append(PositionItem())

    # Build global child nodes under root (critical checks)
    # A) Exactly three distinct positions (we accept at least three distinct in the answer as valid for scoring the first three)
    distinct_names = list({ _normalize_name(p.district_name) for p in all_positions if p.district_name and p.district_name.strip() })
    at_least_three_distinct = len(distinct_names) >= 3
    evaluator.add_custom_node(
        result=at_least_three_distinct,
        id="Provides_Exactly_Three_Distinct_Positions",
        desc="Response identifies exactly three distinct superintendent positions (no duplicates; no extra positions).",
        parent=root,
        critical=True
    )

    # B) Information current as of March 2026 (via sources)
    info_current_leaf = evaluator.add_leaf(
        id="Information_Current_As_Of_March_2026",
        desc="Response indicates (via sources and/or access dates) that the provided information is current as of March 2026 as required.",
        parent=root,
        critical=True
    )
    all_used_urls: List[str] = []
    for p in positions:
        for u in _urls_for(p):
            if u not in all_used_urls:
                all_used_urls.append(u)
    claim_current = (
        "The cited official source pages for these positions show dates (e.g., application deadlines or posting/update dates) in March 2026 or later."
    )
    await evaluator.verify(
        claim=claim_current,
        node=info_current_leaf,
        sources=all_used_urls,
        additional_instruction="Check the pages for explicit dates (deadlines or updates) on or after March 1, 2026. "
                              "If at least the positions under evaluation have such deadlines, consider this satisfied."
    )

    # C) Position subtrees (non-critical overall)
    for idx, pos in enumerate(positions, start=1):
        await verify_position(evaluator, root, pos, idx)

    # Return summary
    return evaluator.get_summary()