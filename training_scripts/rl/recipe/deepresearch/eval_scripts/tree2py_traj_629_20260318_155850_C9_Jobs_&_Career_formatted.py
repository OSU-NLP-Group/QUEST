import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "find_three_higher_ed_leaders_2025"
TASK_DESCRIPTION = (
    "Identify three individuals who were appointed to university president or chancellor positions at public research "
    "universities in the United States in 2025, where each individual transitioned from a previous senior leadership "
    "position (dean, provost, president, or chancellor) at a different institution.\n\n"
    "For each of the three leaders, provide the following information:\n\n"
    "1. Leader Identification:\n"
    "   - Full name of the leader\n"
    "   - Current position title (President or Chancellor)\n"
    "   - Name of the current institution\n"
    "   - A URL reference from the institution's official website confirming the appointment\n\n"
    "2. Current Position Details:\n"
    "   - Confirmation that the institution is a public (state) university, with a supporting URL\n"
    "   - Evidence that the institution is a major research university (either a state flagship or enrollment over 15,000 students), with a supporting URL\n"
    "   - The specific start date (month and year) of the current position in 2025, with a supporting URL\n"
    "   - A URL to the official announcement or press release about the appointment\n\n"
    "3. Previous Position Details:\n"
    "   - The title of the previous leadership position held\n"
    "   - The name of the previous institution\n"
    "   - A URL reference confirming the previous position and institutional affiliation\n\n"
    "4. Career Transition Verification:\n"
    "   - Confirmation that the current institution is different from the previous institution\n"
    "   - A URL documenting this institutional transition\n"
    "   - Confirmation that the previous position was a senior leadership role (Dean, Provost, President, or Chancellor)\n"
    "   - A URL confirming the leadership level of the previous role\n\n"
    "All information must be verifiable through publicly available sources, with specific URLs provided for each major claim."
)


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class Leader(BaseModel):
    # Leader identification
    name: Optional[str] = None
    current_title: Optional[str] = None   # Should be President or Chancellor
    current_institution: Optional[str] = None
    appointment_confirmation_url: Optional[str] = None  # On the current institution’s official website

    # Current position details
    start_date_text: Optional[str] = None  # Month + Year (e.g., "July 2025")
    start_date_url: Optional[str] = None   # Public URL showing the start date (should indicate 2025)
    appointment_announcement_url: Optional[str] = None  # Official announcement/press release URL
    institution_public_url: Optional[str] = None  # URL proving public US university
    major_research_url: Optional[str] = None      # URL proving flagship/R1/Enrollment > 15k

    # Previous position details
    previous_title: Optional[str] = None
    previous_institution: Optional[str] = None
    previous_position_url: Optional[str] = None   # URL confirming previous role + institution
    previous_duration_text: Optional[str] = None  # Timeframe statement for previous role
    previous_duration_url: Optional[str] = None   # URL supporting timeframe for previous role

    # Transition verification
    transition_url: Optional[str] = None          # URL documenting inter-institution move


class LeadersExtraction(BaseModel):
    leaders: List[Leader] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_leaders() -> str:
    return """
    Extract ALL leader entries mentioned in the answer text (do not limit to three; include every distinct leader described).
    For each leader, return an object with the following fields (use null if missing):

    Leader identification:
    - name: full name of the leader
    - current_title: the current position title (should be President or Chancellor)
    - current_institution: the full name of the current institution
    - appointment_confirmation_url: a URL on the current institution’s official website confirming the appointment

    Current position details:
    - start_date_text: the start date (at least month and year, e.g., "July 2025")
    - start_date_url: a public URL explicitly showing this start date in 2025
    - appointment_announcement_url: a public URL to an official announcement/press release about the appointment (may be the same as the appointment_confirmation_url)
    - institution_public_url: a public URL showing the institution is a public university in the United States
    - major_research_url: a public URL showing the institution is a major research university (e.g., state flagship, R1, or enrollment > 15,000)

    Previous position details:
    - previous_title: title of the previous senior leadership position (Dean, Provost, President, or Chancellor)
    - previous_institution: name of the previous institution
    - previous_position_url: a public URL confirming the previous position and institution
    - previous_duration_text: duration/timeframe for the previous position (e.g., "2019–2024", "since 2022", "January 2020 to June 2024")
    - previous_duration_url: a public URL supporting the timeframe/duration for the previous position

    Career transition:
    - transition_url: a public URL documenting the move between different institutions (can reuse an announcement or biography page)

    Strict instructions for URL extraction:
    - Only extract URLs explicitly present in the answer text. Do not invent URLs.
    - Accept plain URLs and markdown links; always return the actual URL.
    - If a URL is missing a scheme, prepend http://
    - Prefer official pages when the instruction asks for official websites (institution .edu domains or their official subdomains).

    Output format:
    {
      "leaders": [
        { ... leader 1 object ... },
        { ... leader 2 object ... },
        ...
      ]
    }

    If some required info is missing in the answer, set those fields to null explicitly.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
_MONTH_PATTERN = re.compile(
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    r")\b", re.IGNORECASE
)
_YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")


def has_month_and_year(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(_MONTH_PATTERN.search(text)) and bool(_YEAR_PATTERN.search(text))


def is_president_or_chancellor(title: Optional[str]) -> bool:
    if not title:
        return False
    t = title.strip().lower()
    # Require the word 'president' or 'chancellor' present as a token (allow hyphens and spaces).
    return bool(re.search(r"\bpresident\b", t)) or bool(re.search(r"\bchancellor\b", t))


def previous_role_is_allowed(title: Optional[str]) -> bool:
    if not title:
        return False
    t = title.strip().lower()
    # Accept clear variants containing the key words as standalone tokens
    return any(re.search(rf"\b{kw}\b", t) for kw in ["dean", "provost", "president", "chancellor"])


def normalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return re.sub(r"\s+", " ", name).strip().lower()


def normalize_institution(inst: Optional[str]) -> str:
    if not inst:
        return ""
    s = inst.lower()
    s = re.sub(r"^the\s+", "", s)
    s = s.replace("univ.", "university")
    s = re.sub(r"[\s\-]+", " ", s)
    s = s.strip()
    return s


def first_non_empty(*vals: Optional[str]) -> Optional[str]:
    for v in vals:
        if v and str(v).strip():
            return v
    return None


# Helper to add a URL-based verification leaf or fail immediately if URL missing
def add_url_verification_or_fail(
    evaluator: Evaluator,
    *,
    node_id: str,
    desc: str,
    parent,
    critical: bool,
    claim: str,
    url: Optional[str],
    add_ins: str,
    batch: List[Tuple[str, Optional[str], Any, Optional[str]]],
) -> None:
    if url and url.strip():
        leaf = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=parent,
            critical=critical
        )
        batch.append((claim, url, leaf, add_ins))
    else:
        evaluator.add_custom_node(
            result=False,
            id=node_id,
            desc=desc,
            parent=parent,
            critical=critical
        )


def add_simple_verification(
    evaluator: Evaluator,
    *,
    node_id: str,
    desc: str,
    parent,
    critical: bool,
    claim: str,
    add_ins: str,
    batch: List[Tuple[str, Optional[str], Any, Optional[str]]],
) -> None:
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )
    batch.append((claim, None, leaf, add_ins))


# --------------------------------------------------------------------------- #
# Verification for one leader                                                 #
# --------------------------------------------------------------------------- #
async def verify_leader(
    evaluator: Evaluator,
    parent_node,
    leader: Leader,
    leader_index: int,
) -> None:
    # Parent node for this leader (non-critical; overall partial credit per leader)
    leader_node = evaluator.add_parallel(
        id=f"leader_{leader_index}",
        desc=f"Leader #{leader_index} (evaluated independently for partial credit).",
        parent=parent_node,
        critical=False
    )

    claims_batch: List[Tuple[str, Optional[str], Any, Optional[str]]] = []

    # 1) Leader Identification (critical group)
    ident_node = evaluator.add_parallel(
        id=f"leader_{leader_index}_identification",
        desc="Leader identity and current role are stated.",
        parent=leader_node,
        critical=True
    )

    # 1.1 Name provided
    evaluator.add_custom_node(
        result=bool(leader.name and leader.name.strip()),
        id=f"leader_{leader_index}_name_provided",
        desc="Provides the leader's full name.",
        parent=ident_node,
        critical=True
    )

    # 1.2 Current title provided AND is President/Chancellor
    evaluator.add_custom_node(
        result=is_president_or_chancellor(leader.current_title),
        id=f"leader_{leader_index}_current_title_provided",
        desc="Provides current position title and it is President or Chancellor.",
        parent=ident_node,
        critical=True
    )

    # 1.3 Current institution name provided
    evaluator.add_custom_node(
        result=bool(leader.current_institution and leader.current_institution.strip()),
        id=f"leader_{leader_index}_current_institution_name_provided",
        desc="Provides the name of the current institution.",
        parent=ident_node,
        critical=True
    )

    # 1.4 Official institution URL confirms appointment (URL-based)
    appointment_claim = (
        f"This page, hosted on the official website of {leader.current_institution}, confirms that "
        f"{leader.name} has been appointed {leader.current_title} of {leader.current_institution}."
    )
    add_url_verification_or_fail(
        evaluator,
        node_id=f"leader_{leader_index}_official_institution_url_confirms_appointment",
        desc="Provides a URL on the current institution’s official website confirming the appointment.",
        parent=ident_node,
        critical=True,
        claim=appointment_claim,
        url=leader.appointment_confirmation_url,
        add_ins=(
            "Treat the URL as official only if it clearly belongs to the institution (e.g., the university's .edu domain "
            "or its official subdomain or system site). The page must explicitly confirm the appointment to the role."
        ),
        batch=claims_batch
    )

    # 2) Current Position Details (critical group)
    current_det_node = evaluator.add_parallel(
        id=f"leader_{leader_index}_current_position_details",
        desc="Current position timing and official announcement are verifiable.",
        parent=leader_node,
        critical=True
    )

    # 2.1 Start date month + year provided
    evaluator.add_custom_node(
        result=has_month_and_year(leader.start_date_text),
        id=f"leader_{leader_index}_start_date_month_year_provided",
        desc="Provides the start date at least to month and year.",
        parent=current_det_node,
        critical=True
    )

    # 2.2 Start date in 2025 with URL
    start_2025_claim = (
        f"This page states that {leader.name}'s start date for the {leader.current_title} role at "
        f"{leader.current_institution} is in 2025 (month and year)."
    )
    add_url_verification_or_fail(
        evaluator,
        node_id=f"leader_{leader_index}_start_date_in_2025_with_url",
        desc="Provides a supporting public URL showing the start date is in 2025.",
        parent=current_det_node,
        critical=True,
        claim=start_2025_claim,
        url=leader.start_date_url,
        add_ins=(
            "Confirm the start/effective date is in calendar year 2025. Accept phrasing like 'effective July 1, 2025', "
            "'begins in August 2025', or a similar explicit reference to 2025 with a month."
        ),
        batch=claims_batch
    )

    # 2.3 Appointment announcement or press release URL
    ann_url = first_non_empty(leader.appointment_announcement_url, leader.appointment_confirmation_url)
    ann_claim = (
        f"This page is an official announcement or press release from {leader.current_institution} "
        f"about {leader.name}'s appointment as {leader.current_title}."
    )
    add_url_verification_or_fail(
        evaluator,
        node_id=f"leader_{leader_index}_appointment_announcement_or_press_release_url",
        desc="Provides a public URL to an official announcement/press release about the appointment (may be the same as the official institution confirmation URL).",
        parent=current_det_node,
        critical=True,
        claim=ann_claim,
        url=ann_url,
        add_ins=(
            "The page should read like an announcement, news item, or press release from the institution (or its system). "
            "It must clearly be about the appointment in question."
        ),
        batch=claims_batch
    )

    # 3) Current Institution Eligibility (critical group)
    inst_elig_node = evaluator.add_parallel(
        id=f"leader_{leader_index}_current_institution_eligibility",
        desc="Current institution meets U.S. public research university constraints, with supporting URLs.",
        parent=leader_node,
        critical=True
    )

    # 3.1 Public US university with URL
    public_claim = (
        f"This page indicates that {leader.current_institution} is a public university located in the United States."
    )
    add_url_verification_or_fail(
        evaluator,
        node_id=f"leader_{leader_index}_public_us_university_with_url",
        desc="Provides evidence with a supporting URL that the institution is in the United States and is a public (state) university (not private).",
        parent=inst_elig_node,
        critical=True,
        claim=public_claim,
        url=leader.institution_public_url,
        add_ins=(
            "Look for indications such as 'public university', 'public research university', 'state university', or "
            "clear governance by a state system within the U.S."
        ),
        batch=claims_batch
    )

    # 3.2 Major research: flagship or enrollment > 15,000 with URL
    research_claim = (
        f"This page shows that {leader.current_institution} is a major research university — for example, it is the state flagship "
        f"OR classified as R1/very high research activity (or equivalent) OR has total student enrollment over 15,000."
    )
    add_url_verification_or_fail(
        evaluator,
        node_id=f"leader_{leader_index}_major_research_flagship_or_enrollment_with_url",
        desc="Provides evidence with a supporting URL that the institution is a major research university (state flagship OR enrollment > 15,000 students).",
        parent=inst_elig_node,
        critical=True,
        claim=research_claim,
        url=leader.major_research_url,
        add_ins=(
            "Accept evidence such as 'flagship university', 'Carnegie R1 (Very High Research Activity)', or an "
            "explicit enrollment count over 15,000 students on the page."
        ),
        batch=claims_batch
    )

    # 4) Previous Position Details (critical group)
    prev_node = evaluator.add_parallel(
        id=f"leader_{leader_index}_previous_position_details",
        desc="Previous position and institution are documented and verifiable.",
        parent=leader_node,
        critical=True
    )

    # 4.1 Previous title and institution provided
    evaluator.add_custom_node(
        result=bool(leader.previous_title and leader.previous_title.strip() and leader.previous_institution and leader.previous_institution.strip()),
        id=f"leader_{leader_index}_previous_title_and_institution_provided",
        desc="Provides the title of the previous position and the name of the previous institution.",
        parent=prev_node,
        critical=True
    )

    # 4.2 Previous position verification URL (confirms role and affiliation)
    prev_pos_claim = (
        f"This page confirms that {leader.name} previously served as {leader.previous_title} at {leader.previous_institution}."
    )
    add_url_verification_or_fail(
        evaluator,
        node_id=f"leader_{leader_index}_previous_position_verification_url",
        desc="Provides an official/authoritative URL confirming the previous position and institutional affiliation.",
        parent=prev_node,
        critical=True,
        claim=prev_pos_claim,
        url=leader.previous_position_url,
        add_ins=(
            "Prefer official pages from the previous institution (e.g., leadership bio, announcement), but credible sources are acceptable "
            "if they explicitly confirm both the position title and the institution."
        ),
        batch=claims_batch
    )

    # 4.3 Previous role is allowed senior leadership (simple verify on the title string)
    role_allowed_claim = (
        f"The previous position title '{leader.previous_title}' is a senior leadership role equivalent to Dean, Provost, President, or Chancellor."
    )
    add_simple_verification(
        evaluator,
        node_id=f"leader_{leader_index}_previous_role_is_allowed_senior_leadership",
        desc="Confirms the previous position title is one of: Dean, Provost, President, or Chancellor.",
        parent=prev_node,
        critical=True,
        claim=role_allowed_claim,
        add_ins=(
            "Judge True only if the title clearly contains one of these keywords as a role: Dean, Provost, President, Chancellor. "
            "Accept reasonable variants like 'Interim President', 'Executive Vice Chancellor and Provost', "
            "'Chancellor-Designate'. Do not accept 'Vice President' alone."
        ),
        batch=claims_batch
    )

    # 4.4 Previous position duration/timeframe with URL
    if leader.previous_duration_text and leader.previous_duration_text.strip() and leader.previous_duration_url and leader.previous_duration_url.strip():
        prev_dur_claim = (
            f"This page states a timeframe or dates for {leader.name}'s service as {leader.previous_title} at {leader.previous_institution}, "
            f"such as '{leader.previous_duration_text}'."
        )
        add_url_verification_or_fail(
            evaluator,
            node_id=f"leader_{leader_index}_previous_position_duration_documentable",
            desc="Provides a duration/timeframe for the previous position, and it is supported by a public URL.",
            parent=prev_node,
            critical=True,
            claim=prev_dur_claim,
            url=leader.previous_duration_url,
            add_ins=(
                "Look for explicit years or month-year ranges (e.g., '2019–2024', 'since 2022', 'January 2020 to June 2024'). "
                "The timeframe must clearly pertain to the stated previous role at the stated previous institution."
            ),
            batch=claims_batch
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"leader_{leader_index}_previous_position_duration_documentable",
            desc="Provides a duration/timeframe for the previous position, and it is supported by a public URL.",
            parent=prev_node,
            critical=True
        )

    # 5) Career Transition Verification (critical group)
    trans_node = evaluator.add_parallel(
        id=f"leader_{leader_index}_career_transition_verification",
        desc="Move represents an inter-institution transition (not an internal promotion), with a supporting URL.",
        parent=leader_node,
        critical=True
    )

    # 5.1 Current and previous institutions are different
    inst_diff = normalize_institution(leader.current_institution) != normalize_institution(leader.previous_institution)
    evaluator.add_custom_node(
        result=inst_diff and bool(leader.current_institution) and bool(leader.previous_institution),
        id=f"leader_{leader_index}_current_and_previous_institutions_are_different",
        desc="Confirms the current institution is different from the previous institution.",
        parent=trans_node,
        critical=True
    )

    # 5.2 Transition documented with URL
    trans_url = first_non_empty(leader.transition_url, leader.appointment_announcement_url, leader.appointment_confirmation_url)
    trans_claim = (
        f"This page documents that {leader.name} moved from {leader.previous_institution} to become "
        f"{leader.current_title} at {leader.current_institution} (an inter-institution transition)."
    )
    add_url_verification_or_fail(
        evaluator,
        node_id=f"leader_{leader_index}_transition_documented_with_url",
        desc="Provides a public URL documenting the transition between different institutions (may reuse the announcement/biography URL).",
        parent=trans_node,
        critical=True,
        claim=trans_claim,
        url=trans_url,
        add_ins=(
            "The page should reference both the prior institution/role and the new appointment to President/Chancellor at the current institution."
        ),
        batch=claims_batch
    )

    # Execute all accumulated verifications in parallel
    if claims_batch:
        await evaluator.batch_verify(claims_batch)


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    # Initialize evaluator (root as parallel aggregator)
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
        default_model=model
    )

    # 1) Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_leaders(),
        template_class=LeadersExtraction,
        extraction_name="leaders_extraction"
    )

    # 2) Global critical check: exactly three distinct leaders provided
    all_leaders = extracted.leaders or []
    names_norm = [normalize_name(ld.name) for ld in all_leaders if ld.name and ld.name.strip()]
    unique_names = set(names_norm)
    exactly_three_entries = (len(all_leaders) == 3)
    three_non_empty = (len(names_norm) == 3)
    three_distinct = (len(unique_names) == 3)

    evaluator.add_custom_node(
        result=bool(exactly_three_entries and three_non_empty and three_distinct),
        id="three_distinct_leaders_provided",
        desc="Exactly three leader entries are provided and they refer to three distinct individuals (not duplicates).",
        parent=root,
        critical=True
    )

    # 3) Verify each of the three leaders (use first three; pad with empty if <3)
    leaders_to_check: List[Leader] = list(all_leaders[:3])
    while len(leaders_to_check) < 3:
        leaders_to_check.append(Leader())

    # Create per-leader parent containers and verify
    for idx, leader in enumerate(leaders_to_check, start=1):
        await verify_leader(evaluator, root, leader, idx)

    # 4) Return structured evaluation summary
    return evaluator.get_summary()