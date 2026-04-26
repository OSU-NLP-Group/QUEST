import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "hs_athletic_director_opening_2026plus"
TASK_DESCRIPTION = """
Identify at least one current high school athletic director job opening in the United States that meets all of the following criteria:

1. The position is for an athletic director at the high school level (grades 7-12 or 9-12)
2. The position is located in a U.S. state
3. The job posting requires or prefers a bachelor's degree as the minimum educational qualification
4. A salary or salary range is specified in the posting
5. The salary (or midpoint of the range if a range is given) is at least $60,000 annually
6. The position is for the 2026-2027 school year or later
7. The position description explicitly mentions budget management as a responsibility
8. The position requires or prefers knowledge of the state's interscholastic athletic association rules (such as OHSAA, UIL, CIF, PIAA, or equivalent)
9. The position includes coaching supervision or hiring responsibilities
10. The job posting must be verifiable through an official URL from a school district, state athletic association, or recognized education job board

For each position you identify, provide:
- The school/district name and state
- The specific position title
- The salary or salary range
- The school year for which the position is posted
- A brief description of key responsibilities that satisfy the criteria
- The official URL where the posting can be verified
"""

EVAL_DATE_STR = "2026-03-22"  # Used in "currently open" judgment
EVAL_DATE = datetime.strptime(EVAL_DATE_STR, "%Y-%m-%d").date()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PositionItem(BaseModel):
    school_or_district: Optional[str] = None
    state: Optional[str] = None
    position_title: Optional[str] = None
    salary: Optional[str] = None
    school_year: Optional[str] = None
    responsibilities_summary: Optional[str] = None
    url: Optional[str] = None
    application_deadline: Optional[str] = None


class PositionsExtraction(BaseModel):
    positions: List[PositionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
    Extract up to the first 3 distinct job openings that the answer presents. For each job opening, return:
    - school_or_district: The school or district name exactly as stated in the answer.
    - state: The U.S. state (full name or two-letter postal abbreviation) as stated in the answer. If unclear or missing, return null.
    - position_title: The specific position title as stated in the answer.
    - salary: The salary or salary range exactly as stated in the answer (e.g., "$65,000 - $80,000", "$72,000", or similar). If the answer does not state it, return null.
    - school_year: The school year or start term (e.g., "2026–2027", "2027-2028", "Starts July 2026") as stated in the answer. If not stated, return null.
    - responsibilities_summary: A brief responsibility/requirements snippet copied from the answer that highlights key points (budget, association rules, coaching supervision/hiring). If not present, return null.
    - url: The official verification URL for the posting as provided in the answer. Must be a URL string; if missing, return null.
    - application_deadline: If the answer states an application deadline date, copy it exactly; otherwise null.

    Rules:
    - Do not invent or infer information not explicitly present in the answer text.
    - Return exactly what the answer states for salary and school year, without normalization.
    - If the answer lists more than one position, include them in order; we will focus on the first.
    """


# --------------------------------------------------------------------------- #
# Helper for robust boolean checks                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


# --------------------------------------------------------------------------- #
# Verification builder for Position 1                                         #
# --------------------------------------------------------------------------- #
async def verify_first_position(evaluator: Evaluator, parent_node, pos: PositionItem) -> None:
    """
    Build and execute the verification subtree for the first (or only) position.
    This follows the rubric leaves closely. Where a leaf requires webpage support,
    we verify against pos.url. Where it checks the response's own reporting,
    we use custom existence checks or simple_verify against the provided summary.
    """
    # Container for Position 1 evaluation. Set as non-critical to allow optional leaves.
    position_node = evaluator.add_parallel(
        id="Position_1_Evaluation",
        desc="Evaluate the first (or only) provided position against all criteria and required reported fields.",
        parent=parent_node,
        critical=False  # Allow mixture of critical and non-critical children
    )

    # ------------------ Presence checks (critical existence) ------------------ #
    node_school_state = evaluator.add_custom_node(
        result=_nonempty(pos.school_or_district) and _nonempty(pos.state),
        id="School_or_District_Name_and_US_State_Provided",
        desc="Response provides the school/district name and a U.S. state for the position.",
        parent=position_node,
        critical=True
    )

    node_title = evaluator.add_custom_node(
        result=_nonempty(pos.position_title),
        id="Position_Title_Provided",
        desc="Response provides the specific position title.",
        parent=position_node,
        critical=True
    )

    node_official_url = evaluator.add_custom_node(
        result=_nonempty(pos.url) and (pos.url.startswith("http://") or pos.url.startswith("https://")),
        id="Official_Verifiable_URL_Provided",
        desc="Response provides an official URL from a school district, state athletic association, or recognized education job board.",
        parent=position_node,
        critical=True
    )

    node_salary_reported = evaluator.add_custom_node(
        result=_nonempty(pos.salary),
        id="Salary_or_Range_Reported_in_Response",
        desc="Response reports the salary or salary range for the position.",
        parent=position_node,
        critical=True
    )

    node_sy_reported = evaluator.add_custom_node(
        result=_nonempty(pos.school_year),
        id="School_Year_Reported_in_Response",
        desc="Response states the school year for which the position is posted.",
        parent=position_node,
        critical=True
    )

    node_resp_summary = evaluator.add_custom_node(
        result=_nonempty(pos.responsibilities_summary),
        id="Brief_Key_Responsibilities_Description_Provided",
        desc="Response includes a brief description of key responsibilities/requirements intended to show how the posting satisfies the criteria.",
        parent=position_node,
        critical=True
    )

    # ------------------ Web-grounded factual checks (critical) ---------------- #
    verifications: List[tuple[str, Optional[str], Any, Optional[str]]] = []

    def add_url_leaf(node_id: str, desc: str, claim: str, add_ins: Optional[str] = None, critical: bool = True):
        node = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=position_node,
            critical=critical
        )
        verifications.append((claim, pos.url, node, add_ins or "None"))
        return node

    # Posting_Is_Current_Open
    add_url_leaf(
        "Posting_Is_Current_Open",
        "The provided posting indicates the job is currently open/active.",
        f"The job posting page shows that the position is currently open for applications as of {EVAL_DATE_STR} "
        f"(not closed/filled/expired). If a deadline is shown, it is not past {EVAL_DATE_STR}.",
        add_ins=(
            "Use page signals such as 'Apply' buttons, 'Now hiring', 'Open until filled', or 'Accepting Applications'. "
            "Treat phrases like 'position filled', 'closed', 'no longer accepting applications', or a clearly past deadline "
            "as not open. If ambiguous, use the most reasonable interpretation from the page."
        ),
    )

    # HS_Level_Athletic_Director_Role
    add_url_leaf(
        "HS_Level_Athletic_Director_Role",
        "The posting describes an athletic director role at the high school level (grades 7–12 or 9–12).",
        "This posting is for an Athletic Director (or Director of Athletics/Activities Director) role at a high school "
        "(grades 9–12 or 7–12, or clearly on a high school campus).",
        add_ins=(
            "Accept synonyms like 'Director of Athletics' or 'Activities Director' if clearly the HS athletics leader. "
            "Reject if it's a district-only role with no HS scope, or a middle/elementary-only role."
        ),
    )

    # Bachelor's degree requirement/preference
    add_url_leaf(
        "Bachelors_Degree_Required_or_Preferred_in_Posting",
        "The posting requires or prefers a bachelor's degree as the minimum educational qualification.",
        "The posting explicitly states that a Bachelor's degree is required or preferred.",
        add_ins="Allow synonyms like 'baccalaureate degree'."
    )

    # Salary or range specified in posting
    add_url_leaf(
        "Salary_or_Range_Specified_in_Posting",
        "The posting specifies a salary or salary range.",
        "The posting provides a specific salary or a salary range for the position.",
        add_ins="Look for '$', 'salary', 'range', 'per year', or comp tables. Stipends alone do NOT count as annual salary."
    )

    # Salary midpoint >= $60,000
    add_url_leaf(
        "Salary_Midpoint_At_Least_60000",
        "The salary (or midpoint if a range is given) is at least $60,000 annually.",
        "The salary (or the midpoint of the salary range) for this position is at least $60,000 per year.",
        add_ins=(
            "If a range is given (e.g., $58,000–$68,000), compute the midpoint and check >= 60000. "
            "If only an annual salary is given, check it directly. If pay is presented per month, multiply by 12. "
            "If stipend-only or hourly without clear FTE conversion, this fails."
        )
    )

    # School year verifiably stated in posting
    add_url_leaf(
        "School_Year_Verifiably_Stated_in_Posting",
        "The posting states the school year (or start school year) for which the position is posted.",
        "The posting explicitly states the school year or start term for which the position is posted (e.g., 2026–2027, starts July 2026).",
        add_ins="Look for phrases like 'for the 2026-2027 school year', 'start date July 2026', 'SY 26-27', etc."
    )

    # School year 2026–2027 or later
    add_url_leaf(
        "School_Year_2026_2027_or_Later",
        "The posted school year is 2026–2027 or later.",
        "The position is for the 2026–2027 school year or any later school year (e.g., 2027–2028, 2028–2029), "
        "or has a start date in 2026 or later that corresponds to the 2026–2027 school year or beyond.",
        add_ins="If only a start month/year is given, infer the school year reasonably (e.g., starts July/Aug 2026 => 2026–2027)."
    )

    # Budget management mentioned in posting
    add_url_leaf(
        "Budget_Management_Mentioned_in_Posting",
        "The posting explicitly mentions budget management as a responsibility.",
        "The posting explicitly states budget-related responsibilities (e.g., budget management, financial oversight, "
        "purchasing oversight, allocating athletics funds).",
        add_ins="Accept synonyms: 'budget oversight', 'financial management', 'fiscal responsibility', 'manages athletics budget'."
    )

    # State association rules knowledge mentioned in posting
    add_url_leaf(
        "State_Association_Rules_Knowledge_in_Posting",
        "The posting requires or prefers knowledge of the state's interscholastic athletic association rules (or equivalent).",
        "The posting requires or prefers knowledge of the state's interscholastic athletic association rules "
        "(e.g., OHSAA, UIL, CIF, PIAA, GHSA, OSAA, FHSAA, MHSAA, WIAA, VHSL, NMAA, KHSAA, etc.).",
        add_ins="Accept explicit state association abbreviations or full names; reject only NFHS without state if no state association is mentioned."
    )

    # Coaching supervision or hiring in posting
    add_url_leaf(
        "Coaching_Supervision_or_Hiring_in_Posting",
        "The posting includes coaching supervision responsibilities OR coaching hiring responsibilities.",
        "The posting states responsibilities to supervise, lead, evaluate, or hire coaches.",
        add_ins="Synonyms include 'oversee coaching staff', 'evaluate coaches', 'select/hire head coaches', 'recommend hires'."
    )

    # ------------------ Response summary checks (critical, simple verify) ----- #
    # These checks rely on the answer's provided responsibilities summary.
    def add_simple_leaf(node_id: str, desc: str, claim: str, add_ins: Optional[str] = None, critical: bool = True):
        node = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=position_node,
            critical=critical
        )
        verifications.append((claim, None, node, add_ins or "None"))
        return node

    rs = pos.responsibilities_summary or ""

    add_simple_leaf(
        "Budget_Management_Noted_in_Response_Summary",
        "Response’s responsibilities/requirements summary explicitly notes budget management.",
        f"The following response responsibilities summary explicitly mentions budget or financial management concepts: '{rs}'.",
        add_ins="Accept synonyms like 'budget', 'financial oversight', 'fiscal', 'managing funds', 'purchasing', 'allocations'."
    )

    add_simple_leaf(
        "State_Association_Rules_Noted_in_Response_Summary",
        "Response’s responsibilities/requirements summary notes the state athletic association rules knowledge requirement/preference.",
        f"The following response responsibilities summary explicitly mentions knowledge of the state interscholastic "
        f"athletic association rules (e.g., UIL, OHSAA, CIF, PIAA, etc.): '{rs}'.",
        add_ins="Pass if the summary references the specific state association or clearly mentions 'state athletic association rules'."
    )

    add_simple_leaf(
        "Coaching_Supervision_or_Hiring_Noted_in_Response_Summary",
        "Response’s responsibilities/requirements summary notes coaching supervision OR coaching hiring responsibilities.",
        f"The following response responsibilities summary explicitly mentions supervising, evaluating, or hiring coaches: '{rs}'.",
        add_ins="Accept terms like 'supervise coaches', 'evaluate coaches', 'hire coaches', 'oversee coaching staff'."
    )

    # ------------------ Optional (non-critical, conditional-on-posting) ------- #
    # Single-step conditional verification using the posting and answer together.
    add_url_leaf(
        "State_Certification_if_Mentioned",
        "If the posting mentions state-specific certification/credential requirements, the response identifies them.",
        f"If the posting mentions any state-specific certification or credential requirements (e.g., teacher certification, "
        f"administrator license, AD permit), then the response identifies them in its text or summary. "
        f"Response excerpt for reference: '{rs}'. If the posting does not mention any, pass.",
        add_ins="Evaluate the posting first; only require the response to include it if the posting mentions it.",
        critical=False
    )

    add_url_leaf(
        "NIAAA_Certification_if_Mentioned",
        "If the posting mentions NIAAA certification preference/requirement, the response specifies it.",
        f"If the posting mentions NIAAA certifications (e.g., RAA, RMSAA, CAA, CMAA) as preferred/required, "
        f"then the response states that preference/requirement. "
        f"Response excerpt for reference: '{rs}'. If not mentioned in posting, pass.",
        add_ins="Check the posting for NIAAA references; only then require that the response mentions it.",
        critical=False
    )

    add_url_leaf(
        "Years_of_Experience_if_Specified",
        "If the posting specifies years of experience, the response states the requirement.",
        f"If the posting specifies required years of experience (e.g., '3–5 years'), then the response states that requirement. "
        f"Response excerpt: '{rs}'. If the posting does not specify a number of years, pass.",
        add_ins="Pass if no numeric experience requirement is present in the posting; otherwise the response must state it.",
        critical=False
    )

    add_url_leaf(
        "Teaching_Experience_if_Mentioned",
        "If the posting mentions a teaching experience requirement, the response notes it.",
        f"If the posting mentions a teaching experience requirement, then the response notes it. "
        f"Response excerpt: '{rs}'. If no such requirement exists in the posting, pass.",
        add_ins="Consider phrases like 'teaching license', 'teaching experience required', 'classroom experience'.",
        critical=False
    )

    add_url_leaf(
        "Application_Deadline_if_Available",
        "If an application deadline is available in the posting, the response provides it.",
        f"If the posting provides an application deadline, then the response includes that date somewhere "
        f"(for instance in a field or the summary). The response's extracted deadline field is: "
        f"'{pos.application_deadline or 'None'}'. If the posting has no deadline, pass.",
        add_ins="If a deadline is shown on the page but absent in the response, fail.",
        critical=False
    )

    # ------------------ Execute all verifications (with auto preconditions) --- #
    await evaluator.batch_verify(verifications)


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
    Evaluate an answer for the 'High School Athletic Director Opening (2026–2027+)' task.
    """
    # Initialize evaluator with a sequential root (so failing the first child can skip later checks)
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

    # Extract structured positions from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction",
    )

    # Determine first position (or fallback to empty)
    positions = extracted.positions or []
    pos1 = positions[0] if len(positions) > 0 else PositionItem()

    # Add "At Least One Position Provided" (critical; gates subsequent checks)
    atleast_one = evaluator.add_custom_node(
        result=(
            len(positions) >= 1 and
            _nonempty(pos1.position_title) and
            _nonempty(pos1.url)
        ),
        id="At_Least_One_Position_Provided",
        desc="Response identifies at least one distinct job opening (i.e., at least one position is present).",
        parent=root,
        critical=True
    )

    # Build and run verifications for Position 1 (will auto-skip if the above fails)
    await verify_first_position(evaluator, root, pos1)

    # Return summary with verification tree and extraction info
    return evaluator.get_summary()