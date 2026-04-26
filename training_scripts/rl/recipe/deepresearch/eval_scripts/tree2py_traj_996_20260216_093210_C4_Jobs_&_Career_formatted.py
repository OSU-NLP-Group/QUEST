import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "b1g_special_teams_hire_jan_2026"
TASK_DESCRIPTION = (
    "In January 2026, a Big Ten Conference university hired a new assistant head coach and special teams coordinator. "
    "This coach had previously spent more than 15 years at a different Big Ten Conference institution. Identify this "
    "coaching hire by providing: (1) The name of the hiring university, (2) The coach's full name, (3) The name of the "
    "coach's immediately previous employer (university), (4) The total number of years the coach served at the previous "
    "institution, and (5) An official URL from the hiring university's athletics website that documents this hire."
)

# Authoritative pages to verify Big Ten membership (multi-URL fallback)
BIG_TEN_MEMBERSHIP_URLS = [
    "https://bigten.org/sports/2019/8/19/schools.aspx",
    "https://bigten.org/sports/2019/6/6/schools.aspx",
    "https://bigten.org/sports/2016/6/13/members.html",
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CoachingHireExtraction(BaseModel):
    hiring_university: Optional[str] = None
    coach_full_name: Optional[str] = None
    previous_institution: Optional[str] = None
    tenure_years_text: Optional[str] = None  # Keep as free-form text if provided (e.g., "16 years", "17 seasons")
    official_athletics_url: Optional[str] = None
    position_title: Optional[str] = None      # e.g., "Assistant Head Coach / Special Teams Coordinator"
    hire_date_text: Optional[str] = None      # e.g., "January 8, 2026", "Jan. 2026"
    additional_urls: List[str] = Field(default_factory=list)  # any other cited URLs in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hire_info() -> str:
    return """
    Extract the following fields from the answer. Return null for any field that is not clearly present.

    Required fields:
    - hiring_university: The full name of the university that made the hire.
    - coach_full_name: The full name of the coach who was hired.
    - previous_institution: The full name of the coach's immediately previous employer (university).
    - tenure_years_text: The total tenure length at the previous institution as stated in the answer (e.g., "16 years", "17 seasons", "over 15 years"). Keep as free-form text; DO NOT convert to a number.
    - official_athletics_url: A URL from the hiring university's official athletics website that documents this hire. It must be an explicit URL present in the answer (e.g., mgoblue.com, gopsusports.com, iuhoosiers.com, gophersports.com, hawkeyesports.com, huskers.com, umterps.com, msuspartans.com, purduesports.com, scarletknights.com, uwbadgers.com, fightingillini.com, hailstate? no — ensure it's the hiring university's official athletics domain). If multiple potential athletics URLs are in the answer, choose the one that specifically documents this hire. If none are present, return null.
    - position_title: The job title as written in the answer (e.g., "Assistant Head Coach/Special Teams Coordinator"). Include the full phrase as given.
    - hire_date_text: The announcement or effective date text for the hire if given (e.g., "January 8, 2026", "Jan. 2026"); otherwise null.
    - additional_urls: An array of any other URLs mentioned in the answer that may be relevant (e.g., previous institution profiles, news releases). Only include explicit URLs; do not invent.

    Notes:
    - Do not infer or invent any values. Only extract what the answer explicitly provides.
    - For URLs, include the full URL string. If a URL is missing the protocol, prepend http://.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str]) -> str:
    return s.strip() if isinstance(s, str) else ""


def _all_sources(extracted: CoachingHireExtraction) -> List[str]:
    urls: List[str] = []
    if extracted.official_athletics_url:
        urls.append(extracted.official_athletics_url)
    if extracted.additional_urls:
        urls.extend([u for u in extracted.additional_urls if isinstance(u, str) and u.strip()])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify(evaluator: Evaluator, extracted: CoachingHireExtraction) -> None:
    """
    Build the verification tree and run checks according to the rubric.
    """
    # Create top-level node from the rubric (parallel non-critical aggregator)
    main_node = evaluator.add_parallel(
        id="CoachingHireIdentification",
        desc="Identify the assistant head coach and special teams coordinator hire at a Big Ten university in January 2026 by a coach with extensive prior Big Ten experience",
        parent=evaluator.root,
        critical=False
    )

    hiring_university = _safe(extracted.hiring_university)
    coach_name = _safe(extracted.coach_full_name)
    previous_inst = _safe(extracted.previous_institution)
    official_url = _safe(extracted.official_athletics_url)
    position_title = _safe(extracted.position_title)
    hire_date_text = _safe(extracted.hire_date_text)

    # 1) OfficialAnnouncementURL (Critical leaf)
    official_url_node = evaluator.add_leaf(
        id="OfficialAnnouncementURL",
        desc="An official URL from the hiring university's athletics website documenting the hire is provided",
        parent=main_node,
        critical=True
    )
    official_url_claim = (
        f"This URL is an official athletics website of {hiring_university} and it contains an official announcement "
        f"documenting the hiring of {coach_name} to the football coaching staff."
    )
    await evaluator.verify(
        claim=official_url_claim,
        node=official_url_node,
        sources=official_url if official_url else None,
        additional_instruction=(
            "Verify that the page is an official athletics website page for the specified university "
            "(check domain, site header/footer branding, and organizational cues) and that it clearly "
            "documents the hiring of the named coach to the football staff. The exact role details and dates "
            "will be checked separately, so focus here on (1) official athletics site authenticity and "
            "(2) that this page documents the hire of the named coach."
        ),
    )

    # 2) HiringUniversityIdentified (Critical aggregator)
    hiring_univ_node = evaluator.add_parallel(
        id="HiringUniversityIdentified",
        desc="The hiring university is correctly identified and is a Big Ten Conference member institution",
        parent=main_node,
        critical=True
    )
    # 2.a) University matches the announcement page (critical leaf)
    hiring_univ_match_leaf = evaluator.add_leaf(
        id="HiringUniversity_NameMatches_Announcement",
        desc="The announcement page is published by the specified hiring university's athletics site",
        parent=hiring_univ_node,
        critical=True
    )
    hiring_univ_match_claim = (
        f"The announcement page is published by {hiring_university}'s official athletics website."
    )
    await evaluator.verify(
        claim=hiring_univ_match_claim,
        node=hiring_univ_match_leaf,
        sources=official_url if official_url else None,
        additional_instruction=(
            "Check site branding, domain ownership, and institutional references on the page "
            "to confirm it is indeed the athletics site of the specified university."
        ),
        extra_prerequisites=[official_url_node]
    )
    # 2.b) University is a Big Ten member (critical leaf)
    hiring_univ_b1g_leaf = evaluator.add_leaf(
        id="HiringUniversity_Is_BigTenMember",
        desc="The hiring university is a Big Ten Conference member",
        parent=hiring_univ_node,
        critical=True
    )
    hiring_univ_b1g_claim = f"{hiring_university} is a member institution of the Big Ten Conference."
    await evaluator.verify(
        claim=hiring_univ_b1g_claim,
        node=hiring_univ_b1g_leaf,
        sources=BIG_TEN_MEMBERSHIP_URLS,
        additional_instruction="Confirm that the university appears in the Big Ten Conference members list."
    )

    # 3) AssistantHeadCoachTitle (Critical leaf)
    ahc_leaf = evaluator.add_leaf(
        id="AssistantHeadCoachTitle",
        desc="The position title includes 'Assistant Head Coach' or equivalent designation",
        parent=main_node,
        critical=True
    )
    ahc_claim = (
        f"On the announcement page, the hire for {coach_name} includes an Assistant Head Coach (or equivalent) "
        f"designation in the job title."
    )
    await evaluator.verify(
        claim=ahc_claim,
        node=ahc_leaf,
        sources=official_url if official_url else None,
        additional_instruction=(
            "Accept 'Assistant Head Coach' explicitly, or reasonable equivalents like 'Associate Head Coach' "
            "or 'Assistant/Associate Head Coach'. The title may be combined with other responsibilities."
        ),
        extra_prerequisites=[official_url_node]
    )

    # 4) SpecialTeamsCoordinatorTitle (Critical leaf)
    stc_leaf = evaluator.add_leaf(
        id="SpecialTeamsCoordinatorTitle",
        desc="The position title includes 'Special Teams Coordinator' or equivalent designation",
        parent=main_node,
        critical=True
    )
    stc_claim = (
        f"On the announcement page, the hire for {coach_name} includes 'Special Teams Coordinator' (or equivalent) "
        f"in the job title."
    )
    await evaluator.verify(
        claim=stc_claim,
        node=stc_leaf,
        sources=official_url if official_url else None,
        additional_instruction=(
            "Accept variations such as 'Special Teams Coordinator', 'STC', or equivalent phrasing indicating the "
            "coach is the special teams coordinator. The title may be paired with other positional duties."
        ),
        extra_prerequisites=[official_url_node]
    )

    # 5) HireDateJanuary2026 (Critical leaf)
    date_leaf = evaluator.add_leaf(
        id="HireDateJanuary2026",
        desc="The hire was officially announced or effective in January 2026",
        parent=main_node,
        critical=True
    )
    date_claim = (
        "The hire was officially announced or has an effective date in January 2026."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=official_url if official_url else None,
        additional_instruction=(
            "Check the press release/article publication date or explicit statements like 'effective January X, 2026'. "
            "Accept standard month abbreviations (e.g., 'Jan.'), and consider the page's dateline/metadata."
        ),
        extra_prerequisites=[official_url_node]
    )

    # 6) CoachNameProvided (Critical leaf)
    name_leaf = evaluator.add_leaf(
        id="CoachNameProvided",
        desc="The coach's full name is correctly provided",
        parent=main_node,
        critical=True
    )
    name_claim = (
        f"The hired coach named in the announcement is {coach_name}."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=official_url if official_url else None,
        additional_instruction=(
            "Verify the full name appears as the subject of the hire on the announcement page. "
            "Allow minor formatting variations (e.g., middle initials)."
        ),
        extra_prerequisites=[official_url_node]
    )

    # 7) PreviousInstitutionIdentified (Critical aggregator)
    prev_inst_node = evaluator.add_parallel(
        id="PreviousInstitutionIdentified",
        desc="The coach's immediately previous institution is correctly identified and is a Big Ten Conference member",
        parent=main_node,
        critical=True
    )
    # 7.a) Previous institution correctly identified on the announcement (critical leaf)
    prev_inst_match_leaf = evaluator.add_leaf(
        id="PreviousInstitution_Matches_Announcement",
        desc="The announcement states the immediately previous institution as specified",
        parent=prev_inst_node,
        critical=True
    )
    prev_inst_match_claim = (
        f"Immediately prior to this hire, {coach_name} was employed by {previous_inst}."
    )
    await evaluator.verify(
        claim=prev_inst_match_claim,
        node=prev_inst_match_leaf,
        sources=official_url if official_url else None,
        additional_instruction=(
            "Look for language like 'comes from', 'spent the last X years at', 'served at', or 'previously at'. "
            "The identified institution must be the immediate, most recent employer."
        ),
        extra_prerequisites=[official_url_node]
    )
    # 7.b) Previous institution is a Big Ten member (critical leaf)
    prev_inst_b1g_leaf = evaluator.add_leaf(
        id="PreviousInstitution_Is_BigTenMember",
        desc="The immediately previous institution is a Big Ten Conference member",
        parent=prev_inst_node,
        critical=True
    )
    prev_inst_b1g_claim = f"{previous_inst} is a member institution of the Big Ten Conference."
    await evaluator.verify(
        claim=prev_inst_b1g_claim,
        node=prev_inst_b1g_leaf,
        sources=BIG_TEN_MEMBERSHIP_URLS,
        additional_instruction="Confirm that the university appears in the Big Ten Conference members list."
    )

    # 8) TenureMoreThan15Years (Critical leaf)
    tenure_leaf = evaluator.add_leaf(
        id="TenureMoreThan15Years",
        desc="The coach served more than 15 years total at the previous institution (from initial start date to departure date)",
        parent=main_node,
        critical=True
    )
    tenure_claim = (
        f"In total, {coach_name} spent more than 15 years at {previous_inst} before this hire."
    )
    # Use all available sources, prioritizing the official announcement; accept 'seasons' phrasing as equivalent to years when > 15
    await evaluator.verify(
        claim=tenure_claim,
        node=tenure_leaf,
        sources=_all_sources(extracted),
        additional_instruction=(
            "Look for statements like 'over 15 years', 'more than 15 seasons', '16 years', '17 seasons', etc. "
            "Treat 'seasons' as one-per-year for this verification. The total tenure must exceed 15 (i.e., ≥16)."
        ),
        extra_prerequisites=[official_url_node]
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
    Evaluate an answer for the January 2026 Big Ten assistant head coach / special teams coordinator hire.
    """
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root strategy per rubric
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_hire_info(),
        template_class=CoachingHireExtraction,
        extraction_name="coaching_hire_extraction",
    )

    # Build verification tree and run checks
    await build_and_verify(evaluator, extracted)

    # Return structured summary
    return evaluator.get_summary()