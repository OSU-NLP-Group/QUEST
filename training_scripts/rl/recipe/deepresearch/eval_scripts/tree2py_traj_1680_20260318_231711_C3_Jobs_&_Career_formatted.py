import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ncaa_d2_ad_position_research"
TASK_DESCRIPTION = (
    "Identify exactly one qualifying NCAA Division II Athletic Director job posting (posted on/after Jan 17, 2026) "
    "recruited via an external search firm, and provide all required details with supporting reference URLs."
)

CUTOFF_DATE_STR = "2026-01-17"
REFERENCE_TODAY_STR = "2026-03-18"  # Provided by the user prompt for the 60-day window context


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PostingItem(BaseModel):
    institution_name: Optional[str] = None
    institution_country: Optional[str] = None
    institution_state: Optional[str] = None
    ncaa_division_status: Optional[str] = None  # e.g., "NCAA Division II", "DII"
    conference: Optional[str] = None
    position_title: Optional[str] = None
    posting_date: Optional[str] = None  # Keep as string to maximize compatibility
    search_firm_name: Optional[str] = None
    reporting_relationship: Optional[str] = None  # e.g., "Reports to the President"
    application_deadline_or_priority_date: Optional[str] = None
    search_firm_contact: Optional[str] = None  # e.g., "apply link", "email", "contact person"
    posting_urls: List[str] = Field(default_factory=list)   # direct job posting or search firm posting page(s)
    support_urls: List[str] = Field(default_factory=list)   # NCAA/conference/institution pages cited in the answer


class PositionsExtraction(BaseModel):
    postings: List[PostingItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
    Extract every distinct NCAA Division II Athletic Director job posting described in the answer text.
    Return a JSON object with a field "postings" which is an array of objects. For each posting found in the answer,
    extract the following fields exactly as written in the answer (do not infer or add new info):

    - institution_name: The institution's name.
    - institution_country: The country of the institution if explicitly stated (e.g., "United States"); otherwise null.
    - institution_state: The U.S. state (if mentioned) or region info; otherwise null.
    - ncaa_division_status: The NCAA division status for the institution if stated (prefer exact phrases like "NCAA Division II", "Division II", "DII"); otherwise null.
    - conference: The athletic conference for the institution if stated; otherwise null.
    - position_title: The exact position title from the posting (e.g., "Athletic Director", "Director of Athletics").
    - posting_date: The posting/open date mentioned (string exactly as in answer); otherwise null.
    - search_firm_name: The external search firm name handling the search if stated; otherwise null.
    - reporting_relationship: To whom the AD reports (e.g., "reports to the President") if stated; otherwise null.
    - application_deadline_or_priority_date: The application deadline or priority review date string if stated; otherwise null.
    - search_firm_contact: The contact or submission method for the search firm (email, portal link, contact person) if stated; otherwise null.
    - posting_urls: All URLs in the answer that directly point to the specific job posting page or the specific search-firm posting page for this role.
                     Include only valid URLs. If a URL is missing a protocol, prepend http://.
    - support_urls: Any additional reference URLs from the answer (e.g., NCAA membership page, conference site, institution athletics page)
                    used to support claims such as NCAA DII status, conference membership, reporting relationship, etc.

    Rules:
    - Only extract information explicitly present in the answer. If a field is not present, set it to null (or empty list for URLs).
    - If multiple postings are mentioned, you must include all of them in the "postings" array (each as its own object).
    - For URLs, accept both plain links and markdown links. Extract the actual link targets.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u or not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        # Basic normalization: ensure protocol exists
        if not (u.startswith("http://") or u.startswith("https://")):
            u = "http://" + u
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def get_sources_for_posting(posting: PostingItem) -> List[str]:
    return _dedup_preserve_order(list(posting.posting_urls or []) + list(posting.support_urls or []))


def get_primary_url(posting: PostingItem) -> Optional[str]:
    all_urls = get_sources_for_posting(posting)
    return all_urls[0] if all_urls else None


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_identification_and_access(
    evaluator: Evaluator,
    parent_node,
    extraction: PositionsExtraction,
    primary: PostingItem
) -> None:
    """
    Build the 'Identify_One_Posting_And_Accessibility' subtree and perform checks.
    """
    node = evaluator.add_parallel(
        id="Identify_One_Posting_And_Accessibility",
        desc="Provide exactly one job posting and ensure it is publicly accessible without login.",
        parent=parent_node,
        critical=True
    )

    # 1) Exactly one posting identified
    exactly_one = evaluator.add_custom_node(
        result=(len(extraction.postings) == 1),
        id="Identify_Exactly_One_Posting",
        desc="Exactly one position/posting is identified (not zero, not multiple).",
        parent=node,
        critical=True
    )

    # 2) Provide at least one URL that directly points to the job posting or the search-firm posting page
    has_direct_url = bool(primary.posting_urls) or bool(primary.support_urls)
    provide_url = evaluator.add_custom_node(
        result=has_direct_url,
        id="Provide_Posting_Or_Search_Page_URL",
        desc="Provide at least one URL that directly points to the job posting or the search-firm search page for the identified role.",
        parent=node,
        critical=True
    )

    # 3) Posting publicly accessible without login
    access_leaf = evaluator.add_leaf(
        id="Posting_Publicly_Accessible_No_Login",
        desc="The job posting/search page is publicly accessible online without requiring login credentials to view basic posting details.",
        parent=node,
        critical=True
    )
    primary_url = get_primary_url(primary)
    claim_access = (
        "The provided job posting or search-firm page for the identified Athletic Director role is publicly accessible "
        "without requiring login to view at least the job title or basic posting details."
    )
    await evaluator.verify(
        claim=claim_access,
        node=access_leaf,
        sources=primary_url,
        additional_instruction="If the page loads basic details (title, description, employer) without authentication, treat as accessible. "
                               "If a sign-in wall prevents viewing basic details, it's not publicly accessible."
    )


async def verify_all_constraints_and_fields(
    evaluator: Evaluator,
    parent_node,
    primary: PostingItem
) -> None:
    """
    Build the 'Verify_Constraints_And_Provide_All_Requested_Fields_With_Sources' subtree and perform checks.
    """
    node = evaluator.add_parallel(
        id="Verify_Constraints_And_Provide_All_Requested_Fields_With_Sources",
        desc="For the identified position, confirm all constraints and provide each requested field, each supported by reference URL(s).",
        parent=parent_node,
        critical=True
    )

    sources_all = get_sources_for_posting(primary)
    primary_url = get_primary_url(primary)

    # 1) Institution name and NCAA Division II status
    leaf_inst_dii = evaluator.add_leaf(
        id="Institution_Name_And_NCAA_DII_Status",
        desc="Provide the institution name and confirm NCAA Division II status, with supporting URL(s).",
        parent=node,
        critical=True
    )
    inst_name = primary.institution_name or "the institution for this posting"
    claim_inst_dii = f"The job posting is for {inst_name}, and this institution competes in NCAA Division II."
    await evaluator.verify(
        claim=claim_inst_dii,
        node=leaf_inst_dii,
        sources=sources_all,
        additional_instruction="Look for explicit phrases such as 'NCAA Division II', 'Division II', 'DII' on the posting or referenced pages "
                               "that clearly tie to the institution. Fuzzy match acceptable for naming variants."
    )

    # 2) Institution located in the United States
    leaf_us = evaluator.add_leaf(
        id="Institution_Located_In_United_States",
        desc="Confirm the institution is located in the United States, with supporting URL(s) or posting evidence.",
        parent=node,
        critical=True
    )
    claim_us = f"{inst_name} is located in the United States."
    await evaluator.verify(
        claim=claim_us,
        node=leaf_us,
        sources=sources_all,
        additional_instruction="Accept evidence such as city and U.S. state, 'United States', '.edu' domain with U.S. context, or the institution's official page showing U.S. address."
    )

    # 3) Position title is Athletic Director (not Assistant/Associate)
    leaf_title = evaluator.add_leaf(
        id="Position_Title_Is_Athletic_Director",
        desc="Confirm the position title is Athletic Director (not Assistant/Associate AD), with supporting URL(s).",
        parent=node,
        critical=True
    )
    pos_title = primary.position_title or "the position"
    claim_title = (
        f"The position title for this posting is the head Athletic Director role (e.g., 'Athletic Director' or 'Director of Athletics'), "
        f"not an Assistant, Associate, Deputy, or similar subordinate AD role. The listed title is '{pos_title}'."
    )
    await evaluator.verify(
        claim=claim_title,
        node=leaf_title,
        sources=primary_url or sources_all,
        additional_instruction="Treat 'Director of Athletics' or 'Athletic Director' as valid. Titles containing 'Assistant', 'Associate', 'Deputy', or 'Senior Associate' are not valid. "
                               "Composite titles like 'Vice President for Intercollegiate Athletics and Director of Athletics' are valid for the head AD role."
    )

    # 4) Posting date on or after 2026-01-17
    leaf_date = evaluator.add_leaf(
        id="Posting_Date_On_Or_After_2026_01_17",
        desc="Provide and confirm the posting date is on or after January 17, 2026, with supporting URL(s).",
        parent=node,
        critical=True
    )
    pd = primary.posting_date or "an indicated posting/open date"
    claim_date = (
        f"The job posting shows a posting/open date of {pd}, and that date is on or after {CUTOFF_DATE_STR}."
    )
    await evaluator.verify(
        claim=claim_date,
        node=leaf_date,
        sources=primary_url or sources_all,
        additional_instruction=f"Locate the posting/open date or equivalent field (e.g., 'Open Date', 'Posted on'). "
                               f"Confirm that the date is >= {CUTOFF_DATE_STR} (January 17, 2026)."
    )

    # 5) External search firm name
    leaf_firm = evaluator.add_leaf(
        id="External_Search_Firm_Name",
        desc="Provide and confirm the name of the external search firm/executive search consultant handling the search, with supporting URL(s).",
        parent=node,
        critical=True
    )
    firm_name = primary.search_firm_name or "the external search firm"
    claim_firm = (
        f"The search for this Athletic Director role is being conducted by {firm_name}."
    )
    await evaluator.verify(
        claim=claim_firm,
        node=leaf_firm,
        sources=sources_all,
        additional_instruction="The page(s) should explicitly state that the search is led by an external search firm or direct applicants to apply via the firm's portal or contact."
    )

    # 6) NCAA DII conference membership
    leaf_conf = evaluator.add_leaf(
        id="NCAA_DII_Conference_Membership",
        desc="Provide and confirm the institution's current NCAA Division II athletic conference membership (recognized DII conference), with supporting URL(s).",
        parent=node,
        critical=True
    )
    conf = primary.conference or "the stated conference"
    claim_conf = f"{inst_name} is a member of the NCAA Division II conference: {conf}."
    await evaluator.verify(
        claim=claim_conf,
        node=leaf_conf,
        sources=sources_all,
        additional_instruction="Prefer official conference or NCAA pages when available. Verify that the named conference is a recognized NCAA Division II conference and that the institution is a member."
    )

    # 7) Reporting relationship
    leaf_reporting = evaluator.add_leaf(
        id="Reporting_Relationship",
        desc="Provide and confirm who the Athletic Director reports to (explicit reporting structure), with supporting URL(s).",
        parent=node,
        critical=True
    )
    reporting_to = primary.reporting_relationship or "the stated supervisor"
    claim_reporting = f"The Athletic Director position reports to {reporting_to}."
    await evaluator.verify(
        claim=claim_reporting,
        node=leaf_reporting,
        sources=primary_url or sources_all,
        additional_instruction="Look for explicit phrases such as 'reports to the President' or 'reports to the Vice President' within the posting or referenced materials."
    )

    # 8) Application deadline or priority review date
    leaf_deadline = evaluator.add_leaf(
        id="Application_Deadline_Or_Priority_Review_Date",
        desc="Provide and confirm a stated application deadline date or priority review date, with supporting URL(s).",
        parent=node,
        critical=True
    )
    deadline = primary.application_deadline_or_priority_date or "a stated application deadline or priority review date"
    claim_deadline = f"The posting provides {deadline} for applications or priority review."
    await evaluator.verify(
        claim=claim_deadline,
        node=leaf_deadline,
        sources=primary_url or sources_all,
        additional_instruction="Accept labels such as 'application deadline', 'priority consideration date', 'first review date', or similar, as long as a concrete date is shown."
    )

    # 9) Search firm contact or submission method
    leaf_contact = evaluator.add_leaf(
        id="Search_Firm_Contact_Or_Submission_Method",
        desc="Provide and confirm the contact method or submission process for the search firm (e.g., website link, email address, or contact person info), with supporting URL(s).",
        parent=node,
        critical=True
    )
    contact = primary.search_firm_contact or "a valid contact or submission method for the search firm"
    claim_contact = f"The posting or referenced pages provide {contact} for contacting or submitting materials to the external search firm for this role."
    await evaluator.verify(
        claim=claim_contact,
        node=leaf_contact,
        sources=sources_all,
        additional_instruction="Accept an application portal link hosted by the firm, a firm email address, or a named firm contact person with contact details. "
                               "This must clearly be associated with the specific AD search."
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
    Evaluate an answer for the NCAA Division II Athletic Director position research task.
    """
    # Initialize evaluator
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

    # Extraction
    extraction = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction"
    )

    # Record ground truth/policy info
    evaluator.add_ground_truth({
        "cutoff_posting_date_on_or_after": CUTOFF_DATE_STR,
        "reference_today": REFERENCE_TODAY_STR,
        "required_role": "Athletic Director (head AD, Director of Athletics)",
        "required_division": "NCAA Division II",
        "requires_external_search_firm": True,
        "required_fields": [
            "Institution name and NCAA Division II status",
            "Institution located in United States",
            "Position title is Athletic Director (not Assistant/Associate)",
            "Posting date on or after 2026-01-17",
            "External search firm name",
            "NCAA Division II conference membership",
            "Reporting relationship (who AD reports to)",
            "Application deadline or priority review date",
            "Search firm contact or submission method"
        ]
    })

    # Choose primary posting (first one if present)
    primary_posting = extraction.postings[0] if extraction.postings else PostingItem()

    # Add custom info for transparency
    evaluator.add_custom_info(
        info={
            "num_postings_extracted": len(extraction.postings),
            "primary_posting_preview": primary_posting.dict()
        },
        info_type="extraction_summary"
    )

    # Main critical task node (mirrors rubric root)
    task_node = evaluator.add_sequential(
        id="NCAA_Division_II_AD_Position_Research",
        desc="Identify exactly one qualifying NCAA Division II Athletic Director posting (>= Jan 17, 2026) with an external search firm and provide all requested details with sources.",
        parent=root,
        critical=True
    )

    # Step 1: Identification and accessibility
    await verify_identification_and_access(evaluator, task_node, extraction, primary_posting)

    # Step 2: Verify constraints and all requested fields with sources
    await verify_all_constraints_and_fields(evaluator, task_node, primary_posting)

    # Return structured summary
    return evaluator.get_summary()