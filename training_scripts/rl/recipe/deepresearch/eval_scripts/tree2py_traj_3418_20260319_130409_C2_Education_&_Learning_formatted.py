import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "miaa_only_private_university"
TASK_DESCRIPTION = """
Among the current member institutions of the Mid-America Intercollegiate Athletics Association (MIAA), identify the only private university. Provide the following information about this university: (1) the city and state where it is located, (2) the date when its full membership in the MIAA officially began, and (3) its enrollment figure as listed on the MIAA official member page.
"""


class UniversityExtraction(BaseModel):
    university_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    full_membership_start_date: Optional[str] = None
    enrollment_from_miaa_member_page: Optional[str] = None
    miaa_member_page_url: Optional[str] = None
    cited_urls: List[str] = Field(default_factory=list)


def prompt_extract_university_info() -> str:
    return """
    Extract the details of the university identified in the answer as the only private current member of the MIAA.
    Return the following fields:
    - university_name: the university name that is claimed to be the only private current MIAA member.
    - city: the city where the university is located (as stated in the answer).
    - state: the state where the university is located (as stated in the answer).
    - full_membership_start_date: the date when the university's full membership in the MIAA officially began (as stated in the answer).
      Accept any reasonable date format exactly as written in the answer (e.g., "July 1, 2015", "2015-07-01", or "2015").
    - enrollment_from_miaa_member_page: the enrollment figure the answer claims is on the MIAA official member page for this university.
      Extract exactly the number or string as written in the answer (e.g., "6,200", "6,200+", or "About 6,200").
    - miaa_member_page_url: the specific URL to the official MIAA member page of the identified university, if the answer includes it.
      If multiple URLs are present, choose the one that appears to be the primary MIAA member profile page for this university.
    - cited_urls: an array of all URLs cited in the answer relevant to this task (e.g., MIAA member list page, the university’s own MIAA member page, official releases, news posts).
    
    Special rules:
    - Only extract URLs explicitly present in the answer text. Include full URLs. If a URL lacks protocol, prepend "http://".
    - If a field is not present in the answer, set it to null. For arrays with no entries, return an empty list.
    """


def _uniq_nonempty(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


async def verify_university_identification(
    evaluator: Evaluator,
    parent,
    info: UniversityExtraction,
):
    node = evaluator.add_sequential(
        id="University_Identification",
        desc="Correctly identify an institution that (a) is a current MIAA member and (b) is the only private institution among current MIAA members.",
        parent=parent,
        critical=True,
    )

    name_ok = bool(info.university_name and info.university_name.strip())
    evaluator.add_custom_node(
        result=name_ok,
        id="university_name_provided",
        desc="A university name is provided in the answer.",
        parent=node,
        critical=True,
    )

    all_sources = _uniq_nonempty([info.miaa_member_page_url] + list(info.cited_urls))
    has_sources = len(all_sources) > 0
    evaluator.add_custom_node(
        result=has_sources,
        id="has_any_source_url",
        desc="At least one source URL is provided to support identification.",
        parent=node,
        critical=True,
    )

    member_leaf = evaluator.add_leaf(
        id="current_member_supported",
        desc="The identified university is a current member of the MIAA.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{info.university_name} is a current member of the Mid-America Intercollegiate Athletics Association (MIAA).",
        node=member_leaf,
        sources=all_sources,
        additional_instruction="Use the provided URLs to confirm current membership (e.g., official MIAA website pages listing current members or the school's MIAA member page). Ignore historical or past-affiliation pages.",
    )

    only_private_leaf = evaluator.add_leaf(
        id="only_private_among_current_members",
        desc="The identified university is the only private institution among current MIAA members.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Among the current MIAA member institutions, {info.university_name} is the only private university.",
        node=only_private_leaf,
        sources=all_sources,
        additional_instruction=(
            "Judge this solely from the provided sources. "
            "Accept only if the combined evidence shows (1) this university is private and (2) "
            "no other current MIAA members are private. Consider only current full members; "
            "ignore affiliates or provisional members."
        ),
    )


async def verify_required_information(
    evaluator: Evaluator,
    parent,
    info: UniversityExtraction,
):
    req_node = evaluator.add_parallel(
        id="Required_Information_Provided",
        desc="Provide all required information for the identified university.",
        parent=parent,
        critical=True,
    )

    # 1) Location (City, State)
    loc_seq = evaluator.add_sequential(
        id="Location_City_State",
        desc="State the city and state where the university is located.",
        parent=req_node,
        critical=True,
    )

    loc_ok = bool(info.city and info.city.strip()) and bool(info.state and info.state.strip())
    evaluator.add_custom_node(
        result=loc_ok,
        id="location_fields_present",
        desc="City and State are both provided.",
        parent=loc_seq,
        critical=True,
    )

    loc_leaf = evaluator.add_leaf(
        id="location_supported",
        desc="The provided city and state are correct for the identified university.",
        parent=loc_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{info.university_name} is located in {info.city}, {info.state}.",
        node=loc_leaf,
        sources=_uniq_nonempty([info.miaa_member_page_url] + list(info.cited_urls)),
        additional_instruction="Accept reasonable formatting variants (e.g., abbreviations like 'MO' for Missouri). Verify using provided sources such as the MIAA member page or the university's official pages.",
    )

    # 2) Full membership start date
    mem_seq = evaluator.add_sequential(
        id="Full_Membership_Start_Date",
        desc="State the date when the university's full membership in the MIAA officially began.",
        parent=req_node,
        critical=True,
    )

    date_ok = bool(info.full_membership_start_date and info.full_membership_start_date.strip())
    evaluator.add_custom_node(
        result=date_ok,
        id="membership_date_present",
        desc="The full membership start date is provided.",
        parent=mem_seq,
        critical=True,
    )

    date_leaf = evaluator.add_leaf(
        id="membership_date_supported",
        desc="The provided full membership start date is correct.",
        parent=mem_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The full MIAA membership for {info.university_name} officially began on {info.full_membership_start_date}.",
        node=date_leaf,
        sources=_uniq_nonempty([info.miaa_member_page_url] + list(info.cited_urls)),
        additional_instruction="Verify the official start date from provided sources (prefer MIAA announcements or official MIAA pages). Accept equivalent date formats if they refer to the same day.",
    )

    # 3) Enrollment from MIAA member page
    enr_seq = evaluator.add_sequential(
        id="Enrollment_From_MIAA_Member_Page",
        desc="Provide the university's enrollment figure as listed on the MIAA official member page.",
        parent=req_node,
        critical=True,
    )

    miaa_url_present = bool(info.miaa_member_page_url and info.miaa_member_page_url.strip())
    evaluator.add_custom_node(
        result=miaa_url_present,
        id="member_page_url_present",
        desc="The MIAA official member page URL for the university is provided.",
        parent=enr_seq,
        critical=True,
    )

    enrollment_ok = bool(info.enrollment_from_miaa_member_page and info.enrollment_from_miaa_member_page.strip())
    evaluator.add_custom_node(
        result=enrollment_ok,
        id="enrollment_value_present",
        desc="The enrollment figure is provided (claimed to be from the MIAA member page).",
        parent=enr_seq,
        critical=True,
    )

    enr_leaf = evaluator.add_leaf(
        id="enrollment_matches_member_page",
        desc="The provided enrollment matches the number on the MIAA official member page.",
        parent=enr_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The enrollment for {info.university_name} is {info.enrollment_from_miaa_member_page} on its official MIAA member page.",
        node=enr_leaf,
        sources=info.miaa_member_page_url if info.miaa_member_page_url else None,
        additional_instruction=(
            "Only use the university's official MIAA member page provided here to verify the enrollment. "
            "If the exact number is not visible in the text, inspect the screenshot. Minor formatting differences "
            "(e.g., commas or plus signs) are acceptable only if they represent the same figure."
        ),
    )


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

    extracted = await evaluator.extract(
        prompt=prompt_extract_university_info(),
        template_class=UniversityExtraction,
        extraction_name="university_info",
    )

    overall = evaluator.add_sequential(
        id="Overall_Task_Evaluation",
        desc="Evaluate whether the response correctly identifies the only private current MIAA member institution and provides the requested attributes.",
        parent=root,
        critical=True,
    )

    await verify_university_identification(evaluator, overall, extracted)
    await verify_required_information(evaluator, overall, extracted)

    evaluator.add_ground_truth({
        "task_focus": "Identify the only private university among current MIAA members, and provide city/state, official full membership start date, and enrollment as listed on the MIAA member page.",
        "notes": "Verification prioritizes the official MIAA site for membership and enrollment. Uniqueness (only private) must be evidenced from provided sources."
    })

    return evaluator.get_summary()