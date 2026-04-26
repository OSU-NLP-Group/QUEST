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
TASK_ID = "big_east_ad_returned_alma_mater"
TASK_DESCRIPTION = (
    "Identify the current Director of Athletics at a Big East Conference university who: "
    "1. Earned both their undergraduate degree (B.A. in Communication, graduated 1997) and graduate degree "
    '(M.A. with "Communication" in the title, graduated 2005) from the same university; '
    "2. Currently works at that same university (returned to their alma mater); "
    "3. Previously worked at their alma mater in a non-athletic-director role earlier in their career; "
    "4. Served as Director of Athletics at a different Division I institution for at least 1.5 years before returning to their alma mater; "
    "5. Began their current athletic director position in July 2019 or later. "
    "Provide the person's full name and include reference URLs documenting their current position, educational background, prior athletic director role, and career history."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AthleticDirectorExtraction(BaseModel):
    # Identity
    full_name: Optional[str] = None

    # Current position
    current_employer: Optional[str] = None
    current_title: Optional[str] = None
    current_start_date: Optional[str] = None  # e.g., "July 2021" or "2021-07-01"

    # Education (bachelor's)
    bachelors_institution: Optional[str] = None
    bachelors_degree_title: Optional[str] = None  # e.g., "B.A. in Communication"
    bachelors_field: Optional[str] = None         # e.g., "Communication"
    bachelors_year: Optional[str] = None          # e.g., "1997"

    # Education (master's)
    masters_institution: Optional[str] = None
    masters_degree_title: Optional[str] = None    # e.g., "M.A. in Corporate and Public Communication"
    masters_field: Optional[str] = None           # e.g., "Communication" (part of title)
    masters_year: Optional[str] = None            # e.g., "2005"

    # Prior AD role (before current role)
    prior_ad_institution: Optional[str] = None
    prior_ad_division: Optional[str] = None       # e.g., "NCAA Division I"
    prior_ad_start: Optional[str] = None          # e.g., "2017" or "August 2017"
    prior_ad_end: Optional[str] = None            # e.g., "2020" or "January 2020"

    # Earlier career at alma mater (non-AD role)
    earlier_alma_mater_role_title: Optional[str] = None
    earlier_alma_mater_role_employer: Optional[str] = None
    earlier_alma_mater_role_dates: Optional[str] = None

    # Source URLs (must come from the answer)
    current_position_urls: List[str] = Field(default_factory=list)
    education_urls: List[str] = Field(default_factory=list)
    prior_ad_urls: List[str] = Field(default_factory=list)
    career_history_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_ad_candidate() -> str:
    return """
    Extract the single athletic director candidate described in the answer who is proposed to satisfy all constraints.
    Return the structured fields strictly based on what is explicitly stated in the answer.

    Required fields to extract:
    1) Identity
       - full_name

    2) Current position
       - current_employer                (university name)
       - current_title                   (e.g., "Director of Athletics", "VP/Director of Athletics")
       - current_start_date              (month+year or year; if specific day is given, include it)

    3) Education (Bachelor's)
       - bachelors_institution
       - bachelors_degree_title          (e.g., "B.A. in Communication")
       - bachelors_field                 (e.g., "Communication" or "Communications")
       - bachelors_year                  (e.g., "1997")

    4) Education (Master's)
       - masters_institution
       - masters_degree_title            (should include the word "Communication" if present in the answer)
       - masters_field                   (e.g., "Communication", if identifiable)
       - masters_year                    (e.g., "2005")

    5) Prior AD role (before the current role)
       - prior_ad_institution            (institution where this person previously served as AD)
       - prior_ad_division               (e.g., "NCAA Division I", "Division I", if mentioned)
       - prior_ad_start                  (start date string)
       - prior_ad_end                    (end date string)

    6) Earlier career at alma mater (non-AD role)
       - earlier_alma_mater_role_title   (e.g., "Assistant AD", "Communications staff", etc.)
       - earlier_alma_mater_role_employer
       - earlier_alma_mater_role_dates

    7) URLs (must be URLs explicitly present in the answer; do not invent or infer)
       - current_position_urls   (URLs that document current position/title and possibly conference)
       - education_urls          (URLs that document degrees, institutions, fields, and years)
       - prior_ad_urls           (URLs that document prior AD role and dates; Division I if available)
       - career_history_urls     (URLs that document earlier career, including non-AD role at alma mater; bios/press releases acceptable)

    Rules:
    - Do not add any information not present in the answer.
    - If any field is not provided in the answer, return null (for strings) or empty list (for URLs).
    - For URLs, only include the actual URLs present in the answer (including markdown links). Prepend http:// if protocol is missing.
    - Use the exact naming as presented in the answer when possible (e.g., institution names, titles).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def merge_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if not u or not isinstance(u, str):
                continue
            u = u.strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_response_requirements(evaluator: Evaluator, parent_node, data: AthleticDirectorExtraction) -> None:
    resp = evaluator.add_parallel(
        id="Response_Requirements",
        desc="Verify the answer includes the required output fields and citations.",
        parent=parent_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.full_name and data.full_name.strip()),
        id="Provide_Full_Name",
        desc="Provide the individual's full name.",
        parent=resp,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(data.current_position_urls and len(data.current_position_urls) > 0),
        id="Provide_Current_Position_URL",
        desc="Include at least one reference URL documenting the individual's current position.",
        parent=resp,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(data.education_urls and len(data.education_urls) > 0),
        id="Provide_Education_URL",
        desc="Include at least one reference URL documenting the individual's educational background.",
        parent=resp,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(data.prior_ad_urls and len(data.prior_ad_urls) > 0),
        id="Provide_Prior_AD_URL",
        desc="Include at least one reference URL documenting the individual's prior athletic director role.",
        parent=resp,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(data.career_history_urls and len(data.career_history_urls) > 0),
        id="Provide_Career_History_URL",
        desc="Include at least one reference URL documenting the individual's career history (including earlier non-AD work at the alma mater).",
        parent=resp,
        critical=True
    )


async def verify_current_position_requirements(evaluator: Evaluator, parent_node, data: AthleticDirectorExtraction) -> None:
    cur = evaluator.add_parallel(
        id="Current_Position_Requirements",
        desc="Verify the individual's current role and employer satisfy conference/title/start-date constraints.",
        parent=parent_node,
        critical=True,
    )

    # Big East membership
    node_be = evaluator.add_leaf(
        id="Big_East_Member",
        desc="The current employing institution is a current member of the Big East Conference.",
        parent=cur,
        critical=True
    )
    employer = data.current_employer or "the individual's current employing institution"
    be_claim = f"{employer} is a current member of the Big East Conference."
    await evaluator.verify(
        claim=be_claim,
        node=node_be,
        sources=merge_urls(data.current_position_urls, data.career_history_urls, data.education_urls),
        additional_instruction=(
            "Confirm that the institution is a CURRENT Big East member (not historical/previous). "
            "Accept pages that clearly indicate Big East membership (e.g., school bio, press release, athletics site). "
            "Minor variations in naming are acceptable (e.g., 'Big East' vs 'BIG EAST')."
        )
    )

    # Athletic Director title
    node_title = evaluator.add_leaf(
        id="Athletic_Director_Title",
        desc="The individual currently holds the title Director of Athletics (or equivalent athletic director position).",
        parent=cur,
        critical=True
    )
    person = data.full_name or "The individual"
    cur_title_claim = f"{person} currently serves in an athletic director role (Director of Athletics or equivalent) at {employer}."
    await evaluator.verify(
        claim=cur_title_claim,
        node=node_title,
        sources=merge_urls(data.current_position_urls, data.career_history_urls),
        additional_instruction=(
            "Accept equivalent titles such as 'Director of Athletics', 'Athletics Director', 'VP/Director of Athletics', "
            "'Vice President for Athletics' if it is clearly the top AD role. "
            "Do NOT accept assistant/associate roles."
        )
    )

    # Start date July 2019 or later
    node_start = evaluator.add_leaf(
        id="Position_Start_Date",
        desc="The individual began the current athletic director position in July 2019 or later (per constraints).",
        parent=cur,
        critical=True
    )
    start_claim = "They began their current athletic director position in July 2019 or later."
    await evaluator.verify(
        claim=start_claim,
        node=node_start,
        sources=merge_urls(data.current_position_urls, data.career_history_urls),
        additional_instruction=(
            "Use the effective/official start date on the page. If only month-year is shown, use that. "
            "If the source states the AD began after 2019-07 (inclusive of July 2019), it qualifies. "
            "If only a press release/announcement date is given, it's acceptable if it clearly indicates a start date in July 2019 or later."
        )
    )


async def verify_educational_credentials(evaluator: Evaluator, parent_node, data: AthleticDirectorExtraction) -> None:
    edu = evaluator.add_parallel(
        id="Educational_Credentials",
        desc="Verify the individual's degrees match institution/field/year requirements and the alma mater equals current employer.",
        parent=parent_node,
        critical=True,
    )

    # Same institution for both degrees
    node_same_inst = evaluator.add_leaf(
        id="Same_Institution_Both_Degrees",
        desc="Both the bachelor's and master's degrees were earned from the same university.",
        parent=edu,
        critical=True
    )
    same_inst_claim = "Both the bachelor's and master's degrees were earned from the same university."
    await evaluator.verify(
        claim=same_inst_claim,
        node=node_same_inst,
        sources=data.education_urls,
        additional_instruction=(
            "The page(s) should explicitly show both degrees and that they are from the same institution. "
            "Slight variations in naming (e.g., 'University of X' vs 'X University') are acceptable if obviously the same."
        )
    )

    # Institution is current employer (returned to alma mater)
    node_same_as_employer = evaluator.add_leaf(
        id="Institution_Is_Current_Employer",
        desc="The university where the individual earned both degrees is the same institution where they currently serve as athletic director (returned to alma mater).",
        parent=edu,
        critical=True
    )
    same_as_employer_claim = (
        "The person currently works as athletic director at the same university where they earned both degrees (returned to their alma mater)."
    )
    await evaluator.verify(
        claim=same_as_employer_claim,
        node=node_same_as_employer,
        sources=merge_urls(data.education_urls, data.current_position_urls, data.career_history_urls),
        additional_instruction=(
            "A single biography/announcement page that mentions both their current AD role and their degrees from the same school is sufficient. "
            "Statements such as 'returned to his/her alma mater' or listing both degrees alongside current role at the same school satisfy this."
        )
    )

    # Bachelor's field is Communication
    node_ba_field = evaluator.add_leaf(
        id="Bachelors_Field",
        desc="The bachelor's degree is in Communication.",
        parent=edu,
        critical=True
    )
    ba_field_claim = "The bachelor's degree is in Communication (or Communications)."
    await evaluator.verify(
        claim=ba_field_claim,
        node=node_ba_field,
        sources=data.education_urls,
        additional_instruction="Treat 'Communication' and 'Communications' as equivalent. Accept 'B.A. in Communication' or similar."
    )

    # Bachelor's graduation year is 1997
    node_ba_year = evaluator.add_leaf(
        id="Bachelors_Graduation_Year",
        desc="The bachelor's degree graduation year is 1997.",
        parent=edu,
        critical=True
    )
    ba_year_claim = "The bachelor's degree graduation year is 1997."
    await evaluator.verify(
        claim=ba_year_claim,
        node=node_ba_year,
        sources=data.education_urls,
        additional_instruction="Verify the bachelor's completion year is 1997."
    )

    # Master's degree title includes "Communication"
    node_ma_field_title = evaluator.add_leaf(
        id="Masters_Field_Title",
        desc='The master's degree title includes \"Communication\" (e.g., Corporate and Public Communication).',
        parent=edu,
        critical=True
    )
    ma_field_title_claim = 'The master’s degree title includes the word "Communication".'
    await evaluator.verify(
        claim=ma_field_title_claim,
        node=node_ma_field_title,
        sources=data.education_urls,
        additional_instruction="Accept related titles like 'Corporate and Public Communication' or similar that clearly include 'Communication'."
    )

    # Master's completion year is 2005
    node_ma_year = evaluator.add_leaf(
        id="Masters_Completion_Year",
        desc="The master's degree completion year is 2005.",
        parent=edu,
        critical=True
    )
    ma_year_claim = "The master's degree completion year is 2005."
    await evaluator.verify(
        claim=ma_year_claim,
        node=node_ma_year,
        sources=data.education_urls,
        additional_instruction="Verify the master's completion year is 2005."
    )


async def verify_career_history(evaluator: Evaluator, parent_node, data: AthleticDirectorExtraction) -> None:
    career = evaluator.add_parallel(
        id="Career_History",
        desc="Verify prior AD experience and prior alma-mater employment requirements.",
        parent=parent_node,
        critical=True,
    )

    # Prior AD role at Division I
    node_prior_ad_div1 = evaluator.add_leaf(
        id="Prior_AD_At_Division_I",
        desc="Before the current role, the individual served as Director of Athletics at a Division I institution.",
        parent=career,
        critical=True
    )
    prior_div1_claim = (
        "Before the current role, the person served as Director of Athletics at an NCAA Division I institution."
    )
    await evaluator.verify(
        claim=prior_div1_claim,
        node=node_prior_ad_div1,
        sources=data.prior_ad_urls,
        additional_instruction=(
            "Confirm both that they were the Director of Athletics and that the institution competes in NCAA Division I "
            "(explicit wording like 'NCAA Division I' or clearly-known D-I membership on the page)."
        )
    )

    # Prior AD at different institution
    node_prior_diff = evaluator.add_leaf(
        id="Prior_AD_Is_Different_Institution",
        desc="The prior athletic director institution is different from the current employer/alma mater.",
        parent=career,
        critical=True
    )
    prior_diff_claim = (
        "The prior athletic director institution is different from the current employer/alma mater."
    )
    await evaluator.verify(
        claim=prior_diff_claim,
        node=node_prior_diff,
        sources=merge_urls(data.career_history_urls, data.prior_ad_urls, data.current_position_urls),
        additional_instruction=(
            "A single bio/announcement page that references the person returning to their alma mater after serving as AD elsewhere is sufficient. "
            "If the page lists the prior AD institution and clearly indicates the current employer is a different school, this passes."
        )
    )

    # Prior AD duration at least 1.5 years
    node_prior_duration = evaluator.add_leaf(
        id="Prior_AD_Duration",
        desc="The individual served in the prior athletic director role for at least 1.5 years.",
        parent=career,
        critical=True
    )
    prior_duration_claim = "They served in the prior athletic director role for at least 1.5 years (18 months)."
    await evaluator.verify(
        claim=prior_duration_claim,
        node=node_prior_duration,
        sources=data.prior_ad_urls,
        additional_instruction=(
            "Use the prior AD start/end dates or clearly stated tenure length on the page. "
            "If the computed duration (considering months/years) is >= 18 months, pass. "
            "If the page only states 'two years', that suffices to confirm >= 1.5 years."
        )
    )

    # Earlier non-AD role at alma mater
    node_alma_non_ad = evaluator.add_leaf(
        id="Alma_Mater_Non_AD_Role_Earlier",
        desc="Earlier in their career, the individual worked at their alma mater in a non-athletic-director capacity.",
        parent=career,
        critical=True
    )
    alma_non_ad_claim = (
        "Earlier in their career, the person worked at their alma mater in a non-athletic-director role."
    )
    await evaluator.verify(
        claim=alma_non_ad_claim,
        node=node_alma_non_ad,
        sources=merge_urls(data.career_history_urls, data.education_urls),
        additional_instruction=(
            "Look for a role at the same university that granted their degrees, but earlier in their career (e.g., "
            "communications staff, development, assistant/associate AD). It must be a non-AD role."
        )
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
    # Initialize evaluator (root is a neutral container)
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

    # Extract structured candidate information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_ad_candidate(),
        template_class=AthleticDirectorExtraction,
        extraction_name="athletic_director_candidate"
    )

    # Build main critical node that mirrors the rubric root
    main = evaluator.add_parallel(
        id="Identify_Athletic_Director",
        desc="Identify the current athletics director at a Big East member institution who satisfies all specified education and career constraints, and provide required citations.",
        parent=root,
        critical=True
    )

    # Build and verify subtrees
    await verify_current_position_requirements(evaluator, main, extracted)
    await verify_educational_credentials(evaluator, main, extracted)
    await verify_career_history(evaluator, main, extracted)
    await verify_response_requirements(evaluator, main, extracted)

    # Return the complete evaluation summary
    return evaluator.get_summary()