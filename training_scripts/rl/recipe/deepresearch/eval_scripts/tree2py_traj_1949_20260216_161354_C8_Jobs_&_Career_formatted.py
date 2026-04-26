import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "p5_admin_jobs_2026q1"
TASK_DESCRIPTION = (
    "Identify three current full-time administrative job openings in NCAA Division I athletic departments at universities "
    "that are members of Power Five conferences (Big Ten, SEC, ACC, Big 12, or Pac-12). Each position must be from a different functional area "
    "(Compliance; Ticket Operations; Academic Services). Each must have a publicly accessible job posting with an application deadline that has not yet passed "
    "as of February 16, 2026 (or is open until filled). For each: institution name and conference affiliation, exact title, 2–3 sentence responsibilities summary, "
    "confirmation of minimum bachelor’s degree, confirmation of full-time benefits-eligible, and a direct URL to the official posting."
)
AS_OF_DATE_STR = "2026-02-16"

ALLOWED_P5_CONFERENCES = {"big ten", "sec", "acc", "big 12", "pac-12", "pac 12"}

# Allowed job posting domains or patterns for "official posting" sources
ALLOWED_JOB_SOURCE_PATTERNS = [
    # Aggregators
    "ncaamarket.ncaa.org",
    "higheredjobs.com",
    "teamworkonline.com",
    "collegesports.jobs",
    # University career platforms/vendors often used officially
    "workdayjobs.com",       # myworkdayjobs.com, wd5.myworkdayjobs.com, etc.
    "myworkdayjobs.com",
    "icims.com",
    "taleo.net",
    "oraclecloud.com",
    "peopleadmin.com",
    "paycomonline.net",
    "dayforcehcm.com",
    "ultipro.com",
    "successfactors.com",
    "brassring.com",
    "greenhouse.io",
    "lever.co",
    "jobvite.com",
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PositionItem(BaseModel):
    functional_area: Optional[str] = None
    institution_name: Optional[str] = None
    conference_affiliation: Optional[str] = None
    position_title: Optional[str] = None
    responsibilities_summary: Optional[str] = None
    posting_url: Optional[str] = None
    application_deadline_text: Optional[str] = None
    full_time_indicator: Optional[str] = None  # e.g., "Full-time", "FT", or null
    benefits_indicator: Optional[str] = None   # e.g., "Benefits-eligible", "with benefits", or null


class PositionsExtraction(BaseModel):
    positions: List[PositionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
Extract all NCAA Division I athletic department administrative job openings mentioned in the answer.

For each position you find, extract the following fields exactly as they appear in the answer (do not invent):
- functional_area: Map or classify the position into exactly one of these canonical categories:
    • "Compliance" (e.g., NCAA compliance, rules education, monitoring, eligibility certification)
    • "Ticket Operations" (e.g., ticket sales, ticket operations, box office management, ticketing customer service)
    • "Academic Services" (e.g., academic advising for student-athletes, tutoring coordination, academic support)
  If the answer doesn’t explicitly say the functional area but it’s clearly implied by the title/description, classify it into one of the three canonical categories above.
- institution_name: The name of the hiring institution (e.g., "University of Michigan").
- conference_affiliation: The Power Five conference affiliation if stated (e.g., Big Ten, SEC, ACC, Big 12, Pac-12). If missing in the answer, return null.
- position_title: The exact position title text as written in the answer.
- responsibilities_summary: The 2–3 sentence responsibilities summary provided in the answer. If the answer includes more than 3 sentences, extract the best 2–3 sentence segment that represents the core responsibilities. If there is no such summary in the answer, return null.
- posting_url: The direct, publicly accessible URL to the official job posting page as given in the answer. If missing, return null.
- application_deadline_text: Any application deadline or closing info mentioned in the answer (e.g., "Apply by Feb 20, 2026", "Open until filled"). If not provided in the answer, return null.
- full_time_indicator: The exact text in the answer indicating full-time status if present (e.g., "Full-time"). Otherwise null.
- benefits_indicator: The exact text in the answer indicating benefits-eligible status if present (e.g., "with benefits", "benefits-eligible"). Otherwise null.

Return a JSON object with a 'positions' array of such objects. If the answer lists positions beyond the three required areas, include them all in the array.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def canonicalize_area(area: Optional[str]) -> Optional[str]:
    if not area:
        return None
    s = area.strip().lower()
    # Compliance synonyms
    comp = ["compliance", "ncaa compliance", "rules", "eligibility", "governance", "monitoring"]
    # Ticket Operations synonyms
    ticket = ["ticket", "ticketing", "box office", "ticket sales", "ticket operations", "ticket office", "ticketing operations"]
    # Academic Services synonyms
    acad = ["academic", "academics", "academic services", "academic advising", "learning specialist", "student-athlete academic", "tutoring", "academic support"]
    for kw in comp:
        if kw in s:
            return "Compliance"
    for kw in ticket:
        if kw in s:
            return "Ticket Operations"
    for kw in acad:
        if kw in s:
            return "Academic Services"
    # Fallback strict canonical map
    if s in {"compliance", "ticket operations", "academic services"}:
        return area.strip().title()
    return None


def is_p5_conference(conf: Optional[str]) -> bool:
    if not conf:
        return False
    s = conf.strip().lower().replace("–", "-").replace("—", "-")
    s = s.replace("conference", "").strip()
    return s in ALLOWED_P5_CONFERENCES


def parse_domain(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        parsed = urlparse(url.strip())
        return parsed.netloc.lower()
    except Exception:
        return None


def is_allowed_official_job_url(url: Optional[str]) -> bool:
    if not url:
        return False
    if not (url.startswith("http://") or url.startswith("https://")):
        return False
    domain = parse_domain(url)
    if not domain:
        return False
    # Accept university .edu or known vendor/aggregator domains
    if domain.endswith(".edu"):
        return True
    for pat in ALLOWED_JOB_SOURCE_PATTERNS:
        if pat in domain:
            return True
    return False


def normalize_url_for_dedup(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        parsed = urlparse(url.strip())
        # Normalize: lower netloc, keep path, drop query/fragment, strip trailing slash
        netloc = parsed.netloc.lower()
        path = parsed.path or ""
        norm = f"{parsed.scheme.lower()}://{netloc}{path}"
        while norm.endswith("/") and len(path) > 1:
            norm = norm[:-1]
            path = path[:-1]
        return norm
    except Exception:
        return url.strip().lower() if url else None


def title_indicates_admin_role(title: Optional[str]) -> bool:
    if not title:
        return False
    t = title.lower()
    admin_markers = [
        "director", "assistant director", "associate director", "coordinator", "manager", "administrator",
        "officer", "specialist", "analyst", "advisor", "advisor", "supervisor", "lead", "senior",
        "associate athletic director", "associate ad", "executive director", "program manager", "program coordinator"
    ]
    if "coach" in t:
        return False
    return any(mark in t for mark in admin_markers)


def select_position_by_area(positions: List[PositionItem], area: str) -> Optional[PositionItem]:
    for p in positions:
        if canonicalize_area(p.functional_area) == area:
            return p
    return None


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_set_level_requirements(
    evaluator: Evaluator,
    parent_node,
    extracted: PositionsExtraction
) -> None:
    set_node = evaluator.add_parallel(
        id="Set_Level_Requirements",
        desc="Requirements that apply to the full set of reported positions.",
        parent=parent_node,
        critical=True
    )

    # Exactly three positions and area coverage
    positions = extracted.positions or []
    recognized = [canonicalize_area(p.functional_area) for p in positions]
    recognized_non_null = [r for r in recognized if r is not None]
    coverage_set = set(recognized_non_null)
    exactly_three = len(positions) == 3
    correct_coverage = coverage_set == {"Compliance", "Ticket Operations", "Academic Services"} and len(recognized_non_null) == 3
    exact_three_and_coverage = exactly_three and correct_coverage

    evaluator.add_custom_node(
        result=exact_three_and_coverage,
        id="Exactly_Three_Positions_And_Area_Coverage",
        desc="Answer provides exactly three positions total, covering exactly these three functional areas: Compliance, Ticket Operations, Academic Services (one per area).",
        parent=set_node,
        critical=True
    )

    # Distinct job postings (use the three target areas if present; else check whatever exists)
    target_urls = []
    for area in ["Compliance", "Ticket Operations", "Academic Services"]:
        p = select_position_by_area(positions, area)
        if p and p.posting_url:
            target_urls.append(normalize_url_for_dedup(p.posting_url))

    all_three_present = len(target_urls) == 3 and all(u is not None for u in target_urls)
    distinct = len(set(u for u in target_urls if u)) == 3 if all_three_present else False

    evaluator.add_custom_node(
        result=distinct,
        id="Distinct_Job_Postings",
        desc="The three positions correspond to three distinct job postings (distinct URLs/posting IDs), not the same posting reused.",
        parent=set_node,
        critical=True
    )


async def build_position_checks(
    evaluator: Evaluator,
    parent_node,
    position: Optional[PositionItem],
    position_node_id: str,
    position_node_desc: str,
    prefix: str,
    canonical_area: str
) -> None:
    """
    Build per-position verification subtree for one functional area.
    - prefix: "P1", "P2", or "P3"
    - canonical_area: "Compliance" | "Ticket Operations" | "Academic Services"
    """
    pos_node = evaluator.add_parallel(
        id=position_node_id,
        desc=position_node_desc,
        parent=parent_node,
        critical=False
    )

    # Prepare extracted fields
    inst = position.institution_name if position else None
    conf = position.conference_affiliation if position else None
    title = position.position_title if position else None
    summary = position.responsibilities_summary if position else None
    url = position.posting_url if position else None

    # Leaf: Institution Name Provided (critical)
    evaluator.add_custom_node(
        result=bool(inst and inst.strip()),
        id=f"{prefix}_Institution_Name_Provided",
        desc="Answer states the name of the hiring institution.",
        parent=pos_node,
        critical=True
    )

    # Leaf: Conference affiliation provided and P5 (critical)
    conf_provided_and_p5 = bool(conf and is_p5_conference(conf))
    evaluator.add_custom_node(
        result=conf_provided_and_p5,
        id=f"{prefix}_Conference_Affiliation_Provided_And_P5",
        desc="Answer states the institution's conference affiliation, and it is one of: Big Ten, SEC, ACC, Big 12, Pac-12.",
        parent=pos_node,
        critical=True
    )

    # Leaf: Institution Is NCAA D1 (critical) - simple logical verification
    inst_d1_node = evaluator.add_leaf(
        id=f"{prefix}_Institution_Is_NCAA_D1",
        desc="Hiring institution is NCAA Division I.",
        parent=pos_node,
        critical=True
    )
    claim_d1 = f"{inst} is an NCAA Division I institution."
    await evaluator.verify(
        claim=claim_d1,
        node=inst_d1_node,
        additional_instruction="If the institution is a member of a Power Five conference (Big Ten, SEC, ACC, Big 12, Pac-12), then it is NCAA Division I. Use this logic to decide."
    )

    # Leaf: Position Title Exact (critical) - needs URL
    title_exact_node = evaluator.add_leaf(
        id=f"{prefix}_Position_Title_Exact",
        desc="Answer provides the exact position title as listed in the job posting.",
        parent=pos_node,
        critical=True
    )

    # Leaf: Title indicates admin/management role (critical) - code-level check
    evaluator.add_custom_node(
        result=title_indicates_admin_role(title),
        id=f"{prefix}_Title_Indicates_Admin_Management_Role",
        desc="Position title clearly indicates an administrative/management role in the functional area.",
        parent=pos_node,
        critical=True
    )

    # Leaf: Functional Area Match (critical) - needs URL
    functional_match_node = evaluator.add_leaf(
        id=f"{prefix}_Functional_Area_Match",
        desc=f"Posting responsibilities are primarily {('NCAA compliance/rules education/monitoring/eligibility certification' if canonical_area=='Compliance' else 'ticket sales/operations/customer service/box office management' if canonical_area=='Ticket Operations' else 'academic advising/tutoring coordination/academic support services for student-athletes')}.",
        parent=pos_node,
        critical=True
    )

    # Leaf: Responsibilities Summary (critical) - needs URL
    summary_node = evaluator.add_leaf(
        id=f"{prefix}_Responsibilities_Summary_2to3_Sentences_And_Based_On_Posting",
        desc="Provides a 2–3 sentence summary of the primary responsibilities, and the summary is based on (and consistent with) the job posting's job description.",
        parent=pos_node,
        critical=True
    )

    # Leaf: Bachelor's minimum (critical) - needs URL
    bachelors_node = evaluator.add_leaf(
        id=f"{prefix}_Bachelors_Minimum",
        desc="Posting confirms minimum of a bachelor's degree from an accredited institution.",
        parent=pos_node,
        critical=True
    )

    # Leaf: Full-time and benefits-eligible (critical) - needs URL
    ft_benefits_node = evaluator.add_leaf(
        id=f"{prefix}_FullTime_And_Benefits_Eligible",
        desc="Posting confirms the position is full-time and benefits-eligible.",
        parent=pos_node,
        critical=True
    )

    # Leaf: Direct public official posting URL allowed source (critical) - custom gating check
    url_allowed = is_allowed_official_job_url(url)
    url_allowed_node = evaluator.add_custom_node(
        result=url_allowed,
        id=f"{prefix}_Direct_Public_Official_Posting_URL_Allowed_Source",
        desc="Provides a direct, publicly accessible URL to an official posting on NCAA Market, HigherEdJobs, TeamWork Online, CollegeSports.jobs, or the university's official career website.",
        parent=pos_node,
        critical=True
    )

    # Leaf: Application still open (critical) - needs URL
    open_node = evaluator.add_leaf(
        id=f"{prefix}_Application_Still_Open",
        desc="Posting deadline has not passed as of Feb 16, 2026, or posting states 'open until filled' (or equivalent).",
        parent=pos_node,
        critical=True
    )

    # Optional leaf only for P1 (Compliance): relevant experience mentioned (non-critical)
    comp_exp_node = None
    if prefix == "P1":
        comp_exp_node = evaluator.add_leaf(
            id=f"{prefix}_Compliance_Experience_Mentioned",
            desc="Posting indicates relevant professional experience in collegiate athletics compliance (reflecting the 'typically require' constraint).",
            parent=pos_node,
            critical=False
        )

    # Build and run verifications that require the posting URL
    # Add precondition: depend on url_allowed_node to avoid false positives if URL is missing/invalid
    # Title exact
    await evaluator.verify(
        claim=f'The job posting shows the position title exactly as "{title}".',
        node=title_exact_node,
        sources=url,
        additional_instruction="Check the displayed posting title. Treat minor case or punctuation differences as equivalent, but reject if wording materially differs.",
        extra_prerequisites=[url_allowed_node]
    )

    # Functional area match
    if canonical_area == "Compliance":
        fam_claim = "Based on the job posting, the role's primary responsibilities are NCAA compliance, rules education, monitoring, and/or eligibility certification."
        fam_ins = "Look for terms like 'compliance', 'NCAA rules', 'education/monitoring', 'eligibility certification', 'bylaw interpretation'. Reject if primarily coaching or unrelated."
    elif canonical_area == "Ticket Operations":
        fam_claim = "Based on the job posting, the role's primary responsibilities are ticket operations such as ticket sales, ticketing operations, customer service, or box office management."
        fam_ins = "Look for terms like 'ticket operations', 'ticket sales', 'box office', 'ticketing system', 'customer service'. Reject if primarily fundraising or unrelated."
    else:  # Academic Services
        fam_claim = "Based on the job posting, the role's primary responsibilities are academic advising, tutoring coordination, or academic support services for student-athletes."
        fam_ins = "Look for terms like 'academic advising', 'student-athlete academic support', 'tutoring coordination', 'progress toward degree', 'study hall'. Reject if unrelated."
    await evaluator.verify(
        claim=fam_claim,
        node=functional_match_node,
        sources=url,
        additional_instruction=fam_ins,
        extra_prerequisites=[url_allowed_node]
    )

    # Responsibilities summary
    sum_text = summary or ""
    await evaluator.verify(
        claim=f'The following 2–3 sentence summary faithfully reflects the posting’s primary responsibilities: "{sum_text}".',
        node=summary_node,
        sources=url,
        additional_instruction="Pass only if the summary is 2–3 sentences and aligns with the posting’s core responsibilities without adding unsupported details.",
        extra_prerequisites=[url_allowed_node]
    )

    # Bachelor's minimum
    await evaluator.verify(
        claim="The posting requires at least a bachelor's degree (or equivalent) as a minimum qualification.",
        node=bachelors_node,
        sources=url,
        additional_instruction="Accept language like 'bachelor’s degree required' or 'minimum of bachelor's degree' (or equivalent phrasing). Reject if only preferred.",
        extra_prerequisites=[url_allowed_node]
    )

    # Full-time and benefits-eligible
    await evaluator.verify(
        claim="The position is full-time and benefits-eligible.",
        node=ft_benefits_node,
        sources=url,
        additional_instruction="Accept if the posting states full-time and mentions benefits eligibility (or implies standard benefits). Reject if part-time/temporary without benefits.",
        extra_prerequisites=[url_allowed_node]
    )

    # Application still open as of AS_OF_DATE_STR
    await evaluator.verify(
        claim=f"As of {AS_OF_DATE_STR}, the application deadline has not passed, or the posting states 'open until filled' (or similar).",
        node=open_node,
        sources=url,
        additional_instruction=f"Check for closing date relative to {AS_OF_DATE_STR}. If the page says 'open until filled', 'continuous', or no hard deadline, consider it open.",
        extra_prerequisites=[url_allowed_node]
    )

    # Compliance experience mentioned (non-critical, only for P1)
    if comp_exp_node is not None:
        await evaluator.verify(
            claim="The posting mentions prior experience working in NCAA compliance (e.g., compliance office, rules education/monitoring) as required or preferred.",
            node=comp_exp_node,
            sources=url,
            additional_instruction="Look for 'experience in NCAA compliance' or similar phrasing in required/preferred qualifications.",
            extra_prerequisites=[url_allowed_node]
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
    Evaluate an answer for the Power Five administrative jobs task and return a structured result.
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

    # Add a non-critical task-level root under evaluator.root to host the rubric tree
    task_root = evaluator.add_parallel(
        id="Task_Root",
        desc="Identify three current Power Five NCAA Division I admin jobs (Compliance, Ticket Ops, Academic Services) with valid postings and required details.",
        parent=root,
        critical=False
    )

    # Extract structured positions from the answer
    extracted_positions = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction"
    )

    # Record custom info for transparency
    evaluator.add_custom_info(
        info={"as_of_date": AS_OF_DATE_STR, "allowed_p5": sorted(list(ALLOWED_P5_CONFERENCES))},
        info_type="as_of_policy",
        info_name="deadline_policy"
    )

    # Build set-level requirements (critical)
    await build_set_level_requirements(evaluator, task_root, extracted_positions)

    # Select positions per required area (first match for each)
    pos_compliance = select_position_by_area(extracted_positions.positions, "Compliance")
    pos_tickets = select_position_by_area(extracted_positions.positions, "Ticket Operations")
    pos_academic = select_position_by_area(extracted_positions.positions, "Academic Services")

    # Build per-position verification subtrees
    await build_position_checks(
        evaluator=evaluator,
        parent_node=task_root,
        position=pos_compliance,
        position_node_id="Position_1_Compliance",
        position_node_desc="Compliance functional-area position details and eligibility.",
        prefix="P1",
        canonical_area="Compliance"
    )

    await build_position_checks(
        evaluator=evaluator,
        parent_node=task_root,
        position=pos_tickets,
        position_node_id="Position_2_Ticket_Operations",
        position_node_desc="Ticket Operations functional-area position details and eligibility.",
        prefix="P2",
        canonical_area="Ticket Operations"
    )

    await build_position_checks(
        evaluator=evaluator,
        parent_node=task_root,
        position=pos_academic,
        position_node_id="Position_3_Academic_Services",
        position_node_desc="Academic Services functional-area position details and eligibility.",
        prefix="P3",
        canonical_area="Academic Services"
    )

    # Return summary
    return evaluator.get_summary()