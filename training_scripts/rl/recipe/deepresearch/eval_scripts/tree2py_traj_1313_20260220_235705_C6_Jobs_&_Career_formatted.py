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
TASK_ID = "ncaa_d2_dev_big_rapids_2026"
TASK_DESCRIPTION = """
I am seeking an NCAA Division II athletic development position with very specific characteristics. Please identify a position that meets ALL of the following requirements:

Institution and Location:
- Must be at an NCAA Division II institution
- Must be located in Big Rapids, Michigan

Educational and Experience Requirements:
- Requires a Bachelor's degree as the minimum educational qualification
- Requires at least 3 years of experience in fundraising, account management, sales, or related relationship-building work

Compensation and Employment Terms:
- Offers a salary range of $52,000-$58,000 annually
- Is a full-time, 12-month continuing (permanent) position

Key Responsibilities:
- Requires conducting a minimum of 140 donor meetings annually
- Has an annual fundraising target of $800,000-$1,000,000
- Includes responsibility for developing NIL (Name, Image, Likeness) fundraising strategies
- Requires maintaining active donor portfolios and recording engagement in a CRM system

Organizational Structure:
- Has a dual reporting structure to both athletic department leadership and university advancement/foundation leadership

Application Timeline:
- Has an initial application review date of March 9, 2026

For this position, please provide:
1. The name of the university
2. The exact position title
3. The specific campus location (city and state)
4. The direct URL to the official job posting
5. Confirmation that the position meets each of the specified requirements with supporting details from the job posting
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PositionExtraction(BaseModel):
    """Structured extraction of the agent's provided position information."""
    university_name: Optional[str] = None
    position_title: Optional[str] = None
    campus_city: Optional[str] = None
    campus_state: Optional[str] = None
    job_posting_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_position_info() -> str:
    return """
    Extract the requested information about the identified NCAA Division II athletic development position from the provided answer text.

    Required fields:
    - university_name: The name of the university as stated in the answer (e.g., "Ferris State University").
    - position_title: The exact position title as stated in the answer (e.g., "Assistant Director of Development, Athletics").
    - campus_city: The city of the position's location as stated in the answer (e.g., "Big Rapids").
    - campus_state: The state of the position's location as stated in the answer (e.g., "Michigan" or "MI").
    - job_posting_url: The direct URL to the official job posting page as provided in the answer. If multiple URLs are given, pick the most direct official posting link (e.g., the university HR, advancement/foundation site, or an official university jobs portal). If none is provided, return null.
    - additional_urls: Any other URLs mentioned for context or support. Exclude the job_posting_url from this list. If none, return an empty array.

    Rules:
    - Extract only what is explicitly present in the answer; do not invent or infer.
    - For URLs, extract fully qualified URLs. If a URL lacks protocol, prepend "http://".
    - If any field is not mentioned, set it to null. For additional_urls, return an empty array if none mentioned.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _normalize_location(city: Optional[str], state: Optional[str]) -> str:
    city_str = (city or "").strip()
    state_str = (state or "").strip()
    if city_str and state_str:
        return f"{city_str}, {state_str}"
    return city_str or state_str or ""


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_job_posting_url(
    evaluator: Evaluator,
    parent_node,
    job_url: Optional[str],
    position_title: Optional[str],
    university_name: Optional[str],
) -> None:
    """
    Create a critical sequential node to ensure the job posting URL is provided and is the official posting.
    """
    jp_node = evaluator.add_sequential(
        id="Job_Posting_URL",
        desc="Provide the direct URL to the official job posting that contains all the required information",
        parent=parent_node,
        critical=True,
    )

    # Existence check (critical)
    evaluator.add_custom_node(
        result=bool(job_url and job_url.strip()),
        id="job_posting_url_provided",
        desc="Job posting URL is provided in the answer",
        parent=jp_node,
        critical=True,
    )

    # Verify the URL corresponds to an official job posting page
    verify_leaf = evaluator.add_leaf(
        id="job_posting_url_verified",
        desc="The provided URL is an official job posting for the specified position",
        parent=jp_node,
        critical=True,
    )

    title_hint = (position_title or "").strip()
    uni_hint = (university_name or "").strip()

    claim = (
        f"This webpage is the official job posting for the position "
        f"{repr(title_hint) if title_hint else 'described in the answer'} "
        f"at {repr(uni_hint) if uni_hint else 'the university described in the answer'}."
    )

    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=job_url if job_url else None,
        additional_instruction=(
            "Confirm that the page is an official job posting (e.g., university HR site, advancement/foundation, or an "
            "official jobs portal). It should display the job title, employer/university (Ferris State University), "
            "and job details. If the page is a news post or third-party aggregator without official posting details, "
            "it should not be considered official."
        ),
    )


async def verify_institution(
    evaluator: Evaluator,
    parent_node,
    job_url: Optional[str],
) -> None:
    """
    Verify the position is at the correct institution and division level and in the correct location.
    """
    inst_node = evaluator.add_parallel(
        id="Institution_Verification",
        desc="Verify the position is at the correct institution and division level",
        parent=parent_node,
        critical=True,
    )

    # University name check
    uni_leaf = evaluator.add_leaf(
        id="University_Name",
        desc="The position is at Ferris State University",
        parent=inst_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The job posting indicates the position is at Ferris State University.",
        node=uni_leaf,
        sources=job_url if job_url else None,
        additional_instruction="Look for the employer/university name. Accept 'Ferris State University' or standard abbreviations.",
    )

    # NCAA Division II check
    div_leaf = evaluator.add_leaf(
        id="NCAA_Division",
        desc="The institution is NCAA Division II",
        parent=inst_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The job posting indicates that the institution competes at the NCAA Division II level.",
        node=div_leaf,
        sources=job_url if job_url else None,
        additional_instruction=(
            "Look for explicit mentions like 'NCAA Division II', conference (e.g., GLIAC) with Division II context, "
            "or statements that clearly indicate DII status. If not mentioned, do not assume."
        ),
    )

    # Campus location check
    loc_leaf = evaluator.add_leaf(
        id="Campus_Location",
        desc="The position is located in Big Rapids, Michigan",
        parent=inst_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The job posting indicates the position is located in Big Rapids, Michigan (Big Rapids, MI).",
        node=loc_leaf,
        sources=job_url if job_url else None,
        additional_instruction="Accept standard city/state formatting variations (e.g., 'Big Rapids, MI').",
    )


async def verify_position_title(
    evaluator: Evaluator,
    parent_node,
    job_url: Optional[str],
    position_title: Optional[str],
) -> None:
    """
    Verify the exact position title matches the job posting.
    """
    title_node = evaluator.add_sequential(
        id="Position_Title",
        desc="Provide the exact position title",
        parent=parent_node,
        critical=True,
    )

    # Existence check
    evaluator.add_custom_node(
        result=bool(position_title and position_title.strip()),
        id="position_title_provided",
        desc="Position title is provided in the answer",
        parent=title_node,
        critical=True,
    )

    # Verify match on the job posting page
    match_leaf = evaluator.add_leaf(
        id="position_title_match",
        desc="The provided position title matches the official job posting",
        parent=title_node,
        critical=True,
    )
    claim = (
        f"The official job posting shows the position title as {repr((position_title or '').strip())} "
        f"or an equivalent phrasing."
    )
    await evaluator.verify(
        claim=claim,
        node=match_leaf,
        sources=job_url if job_url else None,
        additional_instruction=(
            "Focus on the job title text on the posting. Allow minor formatting differences "
            "(e.g., punctuation, capitalization) but the title should be an equivalent match."
        ),
    )


async def verify_qualification_requirements(
    evaluator: Evaluator,
    parent_node,
    job_url: Optional[str],
) -> None:
    """
    Verify the minimum education and experience requirements.
    """
    qual_node = evaluator.add_parallel(
        id="Qualification_Requirements",
        desc="Verify the position has the required qualification criteria",
        parent=parent_node,
        critical=True,
    )

    # Bachelor's degree minimum
    edu_leaf = evaluator.add_leaf(
        id="Educational_Requirement",
        desc="The position requires a Bachelor's degree as minimum education",
        parent=qual_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The job posting states that a Bachelor's degree is the minimum required education.",
        node=edu_leaf,
        sources=job_url if job_url else None,
        additional_instruction="Accept 'Bachelor's', 'baccalaureate', or equivalent language indicating minimum education.",
    )

    # Experience minimum 3 years in relevant areas
    exp_leaf = evaluator.add_leaf(
        id="Experience_Requirement",
        desc="The position requires minimum 3 years of experience in fundraising, account management, sales, or related relationship-building work",
        parent=qual_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The job posting states that at least 3 years of experience is required in fundraising, "
            "account management, sales, or related relationship-building work."
        ),
        node=exp_leaf,
        sources=job_url if job_url else None,
        additional_instruction="Accept equivalent phrasing such as 'minimum of three (3) years' and related areas listed.",
    )


async def verify_compensation_and_terms(
    evaluator: Evaluator,
    parent_node,
    job_url: Optional[str],
) -> None:
    """
    Verify salary range and employment terms.
    """
    comp_node = evaluator.add_parallel(
        id="Compensation_and_Terms",
        desc="Verify the position offers the specified compensation and employment terms",
        parent=parent_node,
        critical=True,
    )

    # Salary Range node
    salary_node = evaluator.add_parallel(
        id="Salary_Range",
        desc="Verify salary specifications",
        parent=comp_node,
        critical=True,
    )

    min_salary_leaf = evaluator.add_leaf(
        id="Minimum_Salary",
        desc="The advertised salary range minimum is $52,000",
        parent=salary_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The job posting specifies a salary minimum of $52,000.",
        node=min_salary_leaf,
        sources=job_url if job_url else None,
        additional_instruction="Accept '$52,000', '$52k', or equivalent numeric representation.",
    )

    max_salary_leaf = evaluator.add_leaf(
        id="Maximum_Salary",
        desc="The advertised salary range maximum is $58,000",
        parent=salary_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The job posting specifies a salary maximum of $58,000.",
        node=max_salary_leaf,
        sources=job_url if job_url else None,
        additional_instruction="Accept '$58,000', '$58k', or equivalent numeric representation.",
    )

    # Employment Status node
    emp_node = evaluator.add_parallel(
        id="Employment_Status",
        desc="Verify employment type and duration",
        parent=comp_node,
        critical=True,
    )

    full_time_leaf = evaluator.add_leaf(
        id="Full_Time_Status",
        desc="The position is full-time",
        parent=emp_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The job posting indicates that the position is full-time.",
        node=full_time_leaf,
        sources=job_url if job_url else None,
        additional_instruction="Look for 'full-time' or equivalent designation.",
    )

    duration_leaf = evaluator.add_leaf(
        id="Position_Duration",
        desc="The position is a 12-month continuing (permanent) position",
        parent=emp_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The job posting indicates the position is a 12-month continuing (permanent) appointment.",
        node=duration_leaf,
        sources=job_url if job_url else None,
        additional_instruction="Accept phrasing indicating 12-month and continuing/permanent status.",
    )


async def verify_key_responsibilities(
    evaluator: Evaluator,
    parent_node,
    job_url: Optional[str],
) -> None:
    """
    Verify the essential responsibilities and targets.
    """
    resp_node = evaluator.add_parallel(
        id="Key_Responsibilities",
        desc="Verify the position includes all specified essential responsibilities",
        parent=parent_node,
        critical=True,
    )

    # Minimum donor meetings
    meetings_leaf = evaluator.add_leaf(
        id="Meeting_Requirement",
        desc="The position requires a minimum of 140 donor meetings per year",
        parent=resp_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The job posting requires a minimum of 140 donor meetings annually.",
        node=meetings_leaf,
        sources=job_url if job_url else None,
        additional_instruction="Look for numeric target '140 meetings' or equivalent explicit requirement.",
    )

    # Fundraising target range
    fund_node = evaluator.add_parallel(
        id="Fundraising_Target",
        desc="Verify annual fundraising goals",
        parent=resp_node,
        critical=True,
    )

    min_fund_leaf = evaluator.add_leaf(
        id="Minimum_Fundraising_Amount",
        desc="The position requires raising at least $800,000 annually",
        parent=fund_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The job posting specifies an annual fundraising minimum target of $800,000.",
        node=min_fund_leaf,
        sources=job_url if job_url else None,
        additional_instruction="Accept '$800,000', '$800k', or equivalent representation.",
    )

    max_fund_leaf = evaluator.add_leaf(
        id="Maximum_Fundraising_Amount",
        desc="The fundraising target range extends to $1,000,000 annually",
        parent=fund_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The job posting specifies the annual fundraising target can extend up to $1,000,000.",
        node=max_fund_leaf,
        sources=job_url if job_url else None,
        additional_instruction="Accept '$1,000,000', '$1M', or equivalent representation.",
    )

    # NIL fundraising strategies
    nil_leaf = evaluator.add_leaf(
        id="NIL_Fundraising",
        desc="The position requires providing fundraising strategy for NIL supportive projects and scholarships",
        parent=resp_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The job posting includes responsibility for developing fundraising strategies for NIL (Name, Image, Likeness) initiatives or supportive projects/scholarships.",
        node=nil_leaf,
        sources=job_url if job_url else None,
        additional_instruction="Look for mentions of NIL strategy, NIL fundraising, or support for NIL scholarships/projects.",
    )

    # Donor portfolio management (portfolio + CRM)
    portfolio_node = evaluator.add_parallel(
        id="Donor_Portfolio_Management",
        desc="Verify donor relationship management requirements",
        parent=resp_node,
        critical=True,
    )

    portfolio_leaf = evaluator.add_leaf(
        id="Portfolio_Maintenance",
        desc="The position requires maintaining active donor portfolios",
        parent=portfolio_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The job posting requires maintaining active donor portfolios.",
        node=portfolio_leaf,
        sources=job_url if job_url else None,
        additional_instruction="Accept equivalent wording indicating active management of donor portfolio/assignments.",
    )

    crm_leaf = evaluator.add_leaf(
        id="CRM_System_Usage",
        desc="The position requires recording all donor engagement in a CRM system",
        parent=portfolio_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The job posting requires recording donor engagement/activity in a CRM system.",
        node=crm_leaf,
        sources=job_url if job_url else None,
        additional_instruction="Accept references to CRM or specific systems used for donor/contact tracking.",
    )

    # Dual reporting structure
    report_node = evaluator.add_parallel(
        id="Dual_Reporting_Structure",
        desc="Verify organizational reporting requirements",
        parent=resp_node,
        critical=True,
    )

    ath_leaf = evaluator.add_leaf(
        id="Athletics_Reporting",
        desc="The position reports to the Director of Athletics",
        parent=report_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The job posting states the position reports to the Director of Athletics.",
        node=ath_leaf,
        sources=job_url if job_url else None,
        additional_instruction="Accept equivalent phrasing for reporting to athletics leadership (Director of Athletics).",
    )

    adv_leaf = evaluator.add_leaf(
        id="Advancement_Reporting",
        desc="The position reports to the Associate Vice President for Advancement within the Foundation",
        parent=report_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The job posting states the position reports to the Associate Vice President for Advancement within the Foundation.",
        node=adv_leaf,
        sources=job_url if job_url else None,
        additional_instruction="Accept equivalent phrasing indicating reporting into university advancement/foundation leadership.",
    )


async def verify_application_information(
    evaluator: Evaluator,
    parent_node,
    job_url: Optional[str],
) -> None:
    """
    Verify application timeline initial review date.
    """
    app_node = evaluator.add_parallel(
        id="Application_Information",
        desc="Verify application timeline",
        parent=parent_node,
        critical=True,
    )

    review_leaf = evaluator.add_leaf(
        id="Initial_Review_Date",
        desc="The initial application review date is March 9, 2026",
        parent=app_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The job posting states that the initial application review date is March 9, 2026.",
        node=review_leaf,
        sources=job_url if job_url else None,
        additional_instruction="Allow date format variations (e.g., 'March 9, 2026', '03/09/2026').",
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
    Evaluate the agent's answer for the NCAA Division II athletic development position in Big Rapids, MI.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregator; we add a critical child node for the main rubric
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

    # Extract structured position info from the answer
    extracted: PositionExtraction = await evaluator.extract(
        prompt=prompt_extract_position_info(),
        template_class=PositionExtraction,
        extraction_name="position_extraction",
    )

    # Add rubric main node (critical)
    rubric_root = evaluator.add_parallel(
        id="Position_Identification",
        desc="Correctly identify the NCAA Division II athletic development position that meets all specified criteria",
        parent=root,
        critical=True,
    )

    # 1) Job Posting URL checks
    await verify_job_posting_url(
        evaluator=evaluator,
        parent_node=rubric_root,
        job_url=extracted.job_posting_url,
        position_title=extracted.position_title,
        university_name=extracted.university_name,
    )

    # 2) Institution verification (university, division, location)
    await verify_institution(
        evaluator=evaluator,
        parent_node=rubric_root,
        job_url=extracted.job_posting_url,
    )

    # 3) Position title match
    await verify_position_title(
        evaluator=evaluator,
        parent_node=rubric_root,
        job_url=extracted.job_posting_url,
        position_title=extracted.position_title,
    )

    # 4) Qualification requirements
    await verify_qualification_requirements(
        evaluator=evaluator,
        parent_node=rubric_root,
        job_url=extracted.job_posting_url,
    )

    # 5) Compensation and employment terms
    await verify_compensation_and_terms(
        evaluator=evaluator,
        parent_node=rubric_root,
        job_url=extracted.job_posting_url,
    )

    # 6) Key responsibilities
    await verify_key_responsibilities(
        evaluator=evaluator,
        parent_node=rubric_root,
        job_url=extracted.job_posting_url,
    )

    # 7) Application timeline
    await verify_application_information(
        evaluator=evaluator,
        parent_node=rubric_root,
        job_url=extracted.job_posting_url,
    )

    # Add some context info to summary
    evaluator.add_custom_info(
        info={
            "extracted_university": extracted.university_name,
            "extracted_position_title": extracted.position_title,
            "extracted_location": _normalize_location(extracted.campus_city, extracted.campus_state),
            "job_posting_url": extracted.job_posting_url,
            "additional_urls": extracted.additional_urls,
        },
        info_type="extracted_summary",
        info_name="extracted_answer_fields"
    )

    # Return structured evaluation summary
    return evaluator.get_summary()