import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "aas248_abstract_prep"
TASK_DESCRIPTION = (
    "You are a planetary scientist preparing to submit an abstract to the 248th American Astronomical Society (AAS) "
    "Meeting, scheduled for June 14-18, 2026, in Pasadena, California. Your research focuses on applying Earth climate "
    "modeling techniques to understand atmospheric processes on other planets. To strengthen your abstract, you want to "
    "reference recent advances in terrestrial climate science and acknowledge leading research institutions working on ice "
    "sheet and climate dynamics.\n\n"
    "Your task is to prepare the following information for your abstract submission:\n\n"
    "1. Climate Science Insights: Identify the '10 New Insights in Climate Science 2025/2026' report. Provide: the exact report "
    "title, the release date, confirmation of its peer-reviewed publication details, valid reference URL(s) for the report, and "
    "at least 3 specific numbered insights (from 1-10) from the report that could be relevant to atmospheric or climate modeling.\n\n"
    "2. Research Institutions: Identify at least 2 (preferably 3) research institutions that conduct Greenland ice sheet or "
    "Arctic climate research. For each institution, provide the name, a description of their specific research focus or ongoing "
    "projects, and a verifiable reference URL.\n\n"
    "3. Conference Requirements: Confirm the following AAS 248 requirements: the full conference name, dates, and location; the "
    "abstract submission deadline(s); the maximum character limit for abstract body text; any registration requirements for "
    "presenting authors; and a valid reference URL for the conference.\n\n"
    "4. Submission Planning: Acknowledge that today is March 18, 2026, and verify whether the submission deadline has passed.\n\n"
    "All information must be verifiable through provided URLs and must be accurate as of March 2026."
)

# Ground-truth anchors to verify against sources (as of March 2026 per rubric)
EXPECTED_REPORT_TITLE = "10 New Insights in Climate Science 2025/2026"
EXPECTED_REPORT_RELEASE_DATE = "October 30, 2025"
EXPECTED_PEER_REVIEW_JOURNAL = "Global Sustainability"
EXPECTED_PEER_REVIEW_PUBLICATION_DATE = "January 8, 2026"

EXPECTED_CONF_NAME_VARIANTS = [
    "248th American Astronomical Society Meeting",
    "248th meeting of the American Astronomical Society",
    "AAS 248",
]
EXPECTED_CONF_DATES = "June 14–18, 2026"  # en dash variant
EXPECTED_CONF_DATES_ALT = "June 14-18, 2026"  # hyphen variant
EXPECTED_CONF_LOCATION = "Pasadena, California"
EXPECTED_DEADLINE_REGULAR = "April 20, 2026"
EXPECTED_DEADLINE_LATE = "May 24, 2026"
EXPECTED_ABSTRACT_CHAR_LIMIT_PHRASE = "2,250 characters"
TODAY_STR = "March 18, 2026"


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class InsightItem(BaseModel):
    number: Optional[str] = None  # Keep as string to be robust (e.g., "1", "Insight 1", "No. 1")
    text: Optional[str] = None
    rationale: Optional[str] = None


class ClimateReportExtraction(BaseModel):
    report_title: Optional[str] = None
    report_release_date: Optional[str] = None
    report_reference_urls: List[str] = Field(default_factory=list)
    peer_review_journal: Optional[str] = None
    peer_review_publication_date: Optional[str] = None
    peer_review_urls: List[str] = Field(default_factory=list)
    insights: List[InsightItem] = Field(default_factory=list)


class InstitutionItem(BaseModel):
    name: Optional[str] = None
    focus_or_project: Optional[str] = None
    url: Optional[str] = None


class ResearchInstitutionsExtraction(BaseModel):
    institutions: List[InstitutionItem] = Field(default_factory=list)


class ConferenceExtraction(BaseModel):
    conference_full_name: Optional[str] = None
    conference_dates: Optional[str] = None
    conference_location: Optional[str] = None
    abstract_deadline_regular: Optional[str] = None
    abstract_deadline_late: Optional[str] = None
    abstract_character_limit: Optional[str] = None
    presenting_author_registration_required: Optional[str] = None  # "yes/no" or phrase
    conference_urls: List[str] = Field(default_factory=list)


class SubmissionPlanningExtraction(BaseModel):
    current_date_ack: Optional[str] = None
    deadline_status_statement: Optional[str] = None  # E.g., "not passed", "still open", etc.


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_climate() -> str:
    return """
    Extract details about the '10 New Insights in Climate Science 2025/2026' report from the answer.
    Required fields:
    - report_title: exact title text quoted or as written in the answer
    - report_release_date: the stated report release date (keep formatting as in the answer)
    - report_reference_urls: list all URLs cited for the report, press release, or official project page(s)
    - peer_review_journal: the named peer-reviewed journal where the report was published (if any)
    - peer_review_publication_date: the stated publication date in the peer-reviewed venue
    - peer_review_urls: list all URLs that point to the peer-reviewed journal page or its publisher page for the report/paper
    - insights: list at least 3 items. For each item, include:
        * number: the insight number (e.g., '1', 'Insight 1', 'No. 1', or similar)
        * text: the text of the insight as provided in the answer (do not invent)
        * rationale: the provided explanation in the answer linking this insight to atmospheric/climate modeling relevance
    If any required information is missing in the answer, return null or an empty list for that specific field.
    """


def prompt_extract_institutions() -> str:
    return """
    Extract up to three institutions that the answer claims conduct Greenland ice sheet or Arctic climate research.
    For each institution, include:
    - name: the institution or center name (e.g., 'NASA Goddard', 'NSIDC', 'CPOM', 'UC Irvine CCI', etc.)
    - focus_or_project: a brief description of a specified research focus or ongoing project mentioned in the answer relevant to Greenland ice or Arctic climate
    - url: a verifiable URL provided in the answer that supports this institution's involvement in Greenland/Ice/Arctic research
    Gather them into an array called 'institutions'. If fewer than 3 are provided, include as many as present.
    """


def prompt_extract_conference() -> str:
    return """
    Extract AAS 248 conference details and abstract submission requirements cited in the answer.
    Required fields:
    - conference_full_name
    - conference_dates
    - conference_location
    - abstract_deadline_regular
    - abstract_deadline_late
    - abstract_character_limit
    - presenting_author_registration_required: a phrase or yes/no indicating if presenting authors must register
    - conference_urls: list all URLs cited for AAS 248 information
    If any field is missing, set it to null or an empty list.
    """


def prompt_extract_submission_planning() -> str:
    return f"""
    Extract the submission-planning statements from the answer.
    Required fields:
    - current_date_ack: the exact date acknowledged in the answer as 'today' (if any)
    - deadline_status_statement: whether the answer explicitly states if the submission deadline has passed as of {TODAY_STR}
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_valid_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    u = url.strip()
    return u.startswith("http://") or u.startswith("https://")


def _valid_url_list(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if _is_valid_url(u)]


def _parse_insight_number(num_str: Optional[str]) -> Optional[int]:
    if not num_str:
        return None
    s = num_str.strip().lower()
    # Extract first integer token 1..10
    digits = ""
    for ch in s:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    if not digits:
        return None
    try:
        val = int(digits)
        if 1 <= val <= 10:
            return val
        return None
    except Exception:
        return None


def _at_least_k_distinct_insight_numbers(insights: List[InsightItem], k: int = 3) -> bool:
    nums: Set[int] = set()
    for it in insights:
        n = _parse_insight_number(it.number)
        if n is not None:
            nums.add(n)
    return len(nums) >= k


def _all_have_rationales(insights: List[InsightItem], k: int = 3) -> bool:
    # Check first k provided items (or all if fewer)
    if not insights:
        return False
    cnt = min(k, len(insights))
    for i in range(cnt):
        r = insights[i].rationale.strip() if insights[i].rationale else ""
        if not r:
            return False
    return True


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_climate_science_insights(
    evaluator: Evaluator,
    parent_node,
    climate: ClimateReportExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Climate_Science_Insights",
        desc="Verify identification of the '10 New Insights in Climate Science 2025/2026' report and extraction of required insights with verifiable sources.",
        parent=parent_node,
        critical=False,
    )

    # URL existence nodes (critical to gate source-grounded checks)
    report_urls = _valid_url_list(climate.report_reference_urls)
    peer_urls = _valid_url_list(climate.peer_review_urls)

    report_urls_node = evaluator.add_custom_node(
        result=len(report_urls) > 0,
        id="Report_Reference_URLs",
        desc="Provides valid reference URL(s) that allow verification of the report itself.",
        parent=node,
        critical=True,
    )
    peer_review_urls_node = evaluator.add_custom_node(
        result=len(peer_urls) > 0,
        id="Peer_Reviewed_Publication_Reference_URL",
        desc="Provides a valid reference URL that allows verification of the peer-reviewed publication details (venue and publication date).",
        parent=node,
        critical=True,
    )

    # Title
    title_leaf = evaluator.add_leaf(
        id="Report_Title",
        desc="Provides the exact report title: '10 New Insights in Climate Science 2025/2026'.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The official report title is '{EXPECTED_REPORT_TITLE}', as shown on the referenced page(s).",
        node=title_leaf,
        sources=(report_urls + peer_urls),
        additional_instruction="Allow minor punctuation/capitalization variations; confirm the official title used by the report's own pages or the journal page.",
        extra_prerequisites=[report_urls_node],
    )

    # Release date
    release_leaf = evaluator.add_leaf(
        id="Report_Release_Date",
        desc="States the report release date as October 30, 2025.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The report was released on {EXPECTED_REPORT_RELEASE_DATE}.",
        node=release_leaf,
        sources=report_urls,
        additional_instruction="Verify using official report/press pages; accept if the page explicitly states this release date.",
        extra_prerequisites=[report_urls_node],
    )

    # Peer-reviewed venue
    venue_leaf = evaluator.add_leaf(
        id="Peer_Reviewed_Journal",
        desc="Identifies the peer-reviewed publication venue as Global Sustainability.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The peer-reviewed publication venue for the report is '{EXPECTED_PEER_REVIEW_JOURNAL}'.",
        node=venue_leaf,
        sources=peer_urls,
        additional_instruction="Use the journal or publisher page to confirm the venue name.",
        extra_prerequisites=[peer_review_urls_node],
    )

    # Peer-reviewed publication date
    pubdate_leaf = evaluator.add_leaf(
        id="Peer_Reviewed_Publication_Date",
        desc="States the peer-reviewed publication date as January 8, 2026.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The peer-reviewed publication date is {EXPECTED_PEER_REVIEW_PUBLICATION_DATE}.",
        node=pubdate_leaf,
        sources=peer_urls,
        additional_instruction="Confirm the publication date shown on the journal/publisher's page; allow minor formatting variants of the date string.",
        extra_prerequisites=[peer_review_urls_node],
    )

    # Insights count and numbering
    insights_ok = _at_least_k_distinct_insight_numbers(climate.insights, k=3)
    evaluator.add_custom_node(
        result=insights_ok,
        id="Insights_Provided_Count_And_Numbering",
        desc="Provides at least 3 distinct, specific numbered insights (numbers within 1–10) from the report.",
        parent=node,
        critical=True,
    )

    # Insights relevance rationales existence
    rationales_ok = _all_have_rationales(climate.insights, k=3)
    evaluator.add_custom_node(
        result=rationales_ok,
        id="Insights_Relevance_Rationales",
        desc="Provides a relevance rationale (brief explanation) linking each selected insight to atmospheric or climate modeling use in the abstract context.",
        parent=node,
        critical=True,
    )


async def verify_one_institution(
    evaluator: Evaluator,
    parent_node,
    inst: InstitutionItem,
    idx: int,
    used_names_lower: Set[str],
) -> None:
    num = idx + 1
    inst_node = evaluator.add_parallel(
        id=f"Institution_{num}",
        desc=f"Candidate institution #{num} with required details and verification.",
        parent=parent_node,
        critical=False,
    )

    # Name (must exist and be distinct)
    name_val = (inst.name or "").strip()
    is_distinct = (name_val.lower() not in used_names_lower) if name_val else False
    name_ok = bool(name_val) and is_distinct
    evaluator.add_custom_node(
        result=name_ok,
        id=f"Institution_{num}_Name",
        desc=("Provides the institution name." if num == 1 else f"Provides the institution name (distinct from prior institutions)."),
        parent=inst_node,
        critical=True,
    )
    if name_ok:
        used_names_lower.add(name_val.lower())

    # URL (must exist and be a valid URL)
    url_ok = _is_valid_url(inst.url)
    url_node = evaluator.add_custom_node(
        result=url_ok,
        id=f"Institution_{num}_URL",
        desc="Provides a verifiable reference URL supporting the stated relevant research focus/project.",
        parent=inst_node,
        critical=True,
    )

    # Focus verification grounded by the provided URL
    focus_leaf = evaluator.add_leaf(
        id=f"Institution_{num}_Focus",
        desc="Describes a specific research focus or ongoing project relevant to Greenland ice sheet or Arctic climate research.",
        parent=inst_node,
        critical=True,
    )
    focus_text = (inst.focus_or_project or "").strip()
    claim_focus = (
        f"The referenced page shows that the institution '{name_val}' conducts the described research: '{focus_text}', "
        f"and that this work is related to Greenland ice sheet and/or Arctic climate research."
    )
    await evaluator.verify(
        claim=claim_focus,
        node=focus_leaf,
        sources=inst.url if url_ok else None,
        additional_instruction="Confirm that the page supports that this institution performs Greenland ice sheet or Arctic climate research matching the described focus/project. Reject if unrelated.",
        extra_prerequisites=[url_node],
    )


async def verify_research_institutions(
    evaluator: Evaluator,
    parent_node,
    insts: ResearchInstitutionsExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Research_Institutions",
        desc="Verify that at least 2 distinct research institutions conducting Greenland ice sheet or Arctic climate research are provided; each with name, specific focus/project, and a verifiable URL.",
        parent=parent_node,
        critical=False,  # Keep non-critical to allow partial scoring; individual institution nodes enforce critical leaves
    )

    # Ensure exactly 3 positions (pad with empty if fewer)
    items = list(insts.institutions)[:3]
    while len(items) < 3:
        items.append(InstitutionItem())

    used_names_lower: Set[str] = set()
    for i in range(3):
        await verify_one_institution(evaluator, node, items[i], i, used_names_lower)


async def verify_conference_requirements(
    evaluator: Evaluator,
    parent_node,
    conf: ConferenceExtraction,
) -> Tuple[Any, Any]:
    node = evaluator.add_parallel(
        id="Conference_Requirements",
        desc="Verify AAS 248 conference details and abstract submission requirements are correctly stated with a valid reference URL.",
        parent=parent_node,
        critical=False,  # Keep non-critical aggregator; all leaves are critical
    )

    conf_urls = _valid_url_list(conf.conference_urls)
    url_exist_node = evaluator.add_custom_node(
        result=len(conf_urls) > 0,
        id="Conference_Reference_URL",
        desc="Provides at least one valid reference URL for AAS 248 conference information that supports the stated conference details/requirements.",
        parent=node,
        critical=True,
    )

    # Full name
    full_name_leaf = evaluator.add_leaf(
        id="Conference_Full_Name",
        desc="States the full conference name (248th American Astronomical Society Meeting / AAS 248).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The referenced AAS 248 page indicates the conference is the 248th American Astronomical Society Meeting (AAS 248).",
        node=full_name_leaf,
        sources=conf_urls,
        additional_instruction="Accept reasonable variants like '248th meeting of the American Astronomical Society' or 'AAS 248'.",
        extra_prerequisites=[url_exist_node],
    )

    # Dates
    dates_leaf = evaluator.add_leaf(
        id="Conference_Dates",
        desc="States the conference dates as June 14–18, 2026.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The conference dates are {EXPECTED_CONF_DATES} (or equivalently {EXPECTED_CONF_DATES_ALT}).",
        node=dates_leaf,
        sources=conf_urls,
        additional_instruction="Confirm dates on the official AAS 248 page; accept en dash or hyphen variants.",
        extra_prerequisites=[url_exist_node],
    )

    # Location
    location_leaf = evaluator.add_leaf(
        id="Conference_Location",
        desc="States the conference location as Pasadena, California.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The conference location is {EXPECTED_CONF_LOCATION}.",
        node=location_leaf,
        sources=conf_urls,
        additional_instruction="Verify that the official AAS 248 page states the host city and state.",
        extra_prerequisites=[url_exist_node],
    )

    # Regular deadline
    reg_deadline_leaf = evaluator.add_leaf(
        id="Abstract_Submission_Deadline_Regular",
        desc="States the regular abstract submission deadline as April 20, 2026.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The regular abstract submission deadline is {EXPECTED_DEADLINE_REGULAR}.",
        node=reg_deadline_leaf,
        sources=conf_urls,
        additional_instruction="Use the AAS 248 call-for-abstracts or submission page to confirm the regular deadline date.",
        extra_prerequisites=[url_exist_node],
    )

    # Late deadline
    late_deadline_leaf = evaluator.add_leaf(
        id="Abstract_Submission_Deadline_Late",
        desc="States the late abstract submission deadline as May 24, 2026.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The late abstract submission deadline is {EXPECTED_DEADLINE_LATE}.",
        node=late_deadline_leaf,
        sources=conf_urls,
        additional_instruction="Use the official AAS 248 information to confirm the late submission deadline date.",
        extra_prerequisites=[url_exist_node],
    )

    # Character limit
    char_limit_leaf = evaluator.add_leaf(
        id="Abstract_Character_Limit",
        desc="States the maximum abstract body text limit as 2,250 characters including all text elements.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The maximum abstract body text limit is {EXPECTED_ABSTRACT_CHAR_LIMIT_PHRASE}, including all text elements.",
        node=char_limit_leaf,
        sources=conf_urls,
        additional_instruction="Confirm the abstract character limits from official submission instructions; allow minor wording variations.",
        extra_prerequisites=[url_exist_node],
    )

    # Registration requirement
    reg_required_leaf = evaluator.add_leaf(
        id="Presenting_Author_Registration",
        desc="States that presenting authors must register for the AAS conference.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Presenting authors are required to register for the AAS conference in order to present.",
        node=reg_required_leaf,
        sources=conf_urls,
        additional_instruction="Verify official AAS guidance on presenter registration requirements.",
        extra_prerequisites=[url_exist_node],
    )

    # Return the two deadline nodes so Submission_Planning can depend on them
    return reg_deadline_leaf, late_deadline_leaf


async def verify_submission_planning(
    evaluator: Evaluator,
    parent_node,
    subplan: SubmissionPlanningExtraction,
    prereq_deadline_nodes: Tuple[Any, Any],
) -> None:
    node = evaluator.add_parallel(
        id="Submission_Planning",
        desc="Verify the current date acknowledgment and whether the submission deadline has passed.",
        parent=parent_node,
        critical=False,
    )

    # Current date acknowledgment (checked against the answer text)
    cur_date_leaf = evaluator.add_leaf(
        id="Current_Date",
        desc=f"Acknowledges that today is {TODAY_STR}.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The answer explicitly acknowledges that today is {TODAY_STR}.",
        node=cur_date_leaf,
        additional_instruction="Search the answer text for an explicit acknowledgment of the current date.",
    )

    # Deadline status as of current date (logic check; require the deadline nodes to have succeeded)
    deadline_status_leaf = evaluator.add_leaf(
        id="Deadline_Status_As_Of_Current_Date",
        desc=f"Correctly verifies whether the submission deadline has passed as of {TODAY_STR} (it has not passed).",
        parent=node,
        critical=True,
    )
    reg_deadline_leaf, late_deadline_leaf = prereq_deadline_nodes
    await evaluator.verify(
        claim=f"As of {TODAY_STR}, the abstract submission deadline(s) for AAS 248 have not passed.",
        node=deadline_status_leaf,
        additional_instruction=(
            "Use logical reasoning with the previously verified regular and late deadlines. "
            f"If the verified regular deadline is {EXPECTED_DEADLINE_REGULAR} and/or the late deadline is {EXPECTED_DEADLINE_LATE}, "
            f"then on {TODAY_STR} the deadline has not yet passed."
        ),
        extra_prerequisites=[reg_deadline_leaf, late_deadline_leaf],
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
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

    # Parallelize extractions
    climate_fut = evaluator.extract(
        prompt=prompt_extract_climate(),
        template_class=ClimateReportExtraction,
        extraction_name="climate_report",
    )
    inst_fut = evaluator.extract(
        prompt=prompt_extract_institutions(),
        template_class=ResearchInstitutionsExtraction,
        extraction_name="research_institutions",
    )
    conf_fut = evaluator.extract(
        prompt=prompt_extract_conference(),
        template_class=ConferenceExtraction,
        extraction_name="conference_requirements",
    )
    subplan_fut = evaluator.extract(
        prompt=prompt_extract_submission_planning(),
        template_class=SubmissionPlanningExtraction,
        extraction_name="submission_planning",
    )

    climate, insts, conf, subplan = await asyncio.gather(
        climate_fut, inst_fut, conf_fut, subplan_fut
    )

    # Optional: record ground-truth anchors for transparency
    evaluator.add_ground_truth(
        {
            "expected_report": {
                "title": EXPECTED_REPORT_TITLE,
                "release_date": EXPECTED_REPORT_RELEASE_DATE,
                "peer_review_journal": EXPECTED_PEER_REVIEW_JOURNAL,
                "peer_review_publication_date": EXPECTED_PEER_REVIEW_PUBLICATION_DATE,
            },
            "expected_conference": {
                "name_variants": EXPECTED_CONF_NAME_VARIANTS,
                "dates": EXPECTED_CONF_DATES,
                "dates_alt": EXPECTED_CONF_DATES_ALT,
                "location": EXPECTED_CONF_LOCATION,
                "deadlines": {
                    "regular": EXPECTED_DEADLINE_REGULAR,
                    "late": EXPECTED_DEADLINE_LATE,
                },
                "abstract_char_limit_phrase": EXPECTED_ABSTRACT_CHAR_LIMIT_PHRASE,
            },
            "today": TODAY_STR,
        },
        gt_type="ground_truth",
    )

    # Build top-level node (kept non-critical to satisfy framework critical-child constraints)
    top = evaluator.add_parallel(
        id="Conference_Abstract_Preparation",
        desc="Evaluate whether the required information for the AAS 248 abstract submission is provided and verifiable via URLs.",
        parent=root,
        critical=False,
    )

    # Climate insights
    await verify_climate_science_insights(evaluator, top, climate)

    # Institutions
    await verify_research_institutions(evaluator, top, insts)

    # Conference requirements
    reg_deadline_leaf, late_deadline_leaf = await verify_conference_requirements(evaluator, top, conf)

    # Submission planning (depends on the verified deadlines)
    await verify_submission_planning(evaluator, top, subplan, (reg_deadline_leaf, late_deadline_leaf))

    # Return structured evaluation summary
    return evaluator.get_summary()