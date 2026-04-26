import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "me_school_weather_cutoff"
TASK_DESCRIPTION = (
    "In the state of Maine, identify a school district that has publicly documented on its official website "
    "the latest time in the morning by which weather-related school closure decisions may be made. Provide the following "
    "information: (1) the name of the school district, (2) the latest time stated for making closure decisions, and "
    "(3) the list of officials or departments that the superintendent consults with when making the closure decision, "
    "as documented by the district."
)

EVAL_DESCRIPTION = (
    "Evaluate whether the answer identifies a Maine school district whose official website documents the latest morning "
    "decision time for weather-related closures and the superintendent's consultation list, and whether the answer reports "
    "those requested fields correctly."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class MaineDistrictExtraction(BaseModel):
    district_name: Optional[str] = None
    official_urls: List[str] = Field(default_factory=list)
    latest_decision_time: Optional[str] = None
    consulted_officials: List[str] = Field(default_factory=list)

    # Optional/partial-credit process details (only if the answer mentions them; else null/empty)
    extra_decision_by_superintendent: Optional[bool] = None
    extra_monitoring_starts_day_before: Optional[bool] = None
    extra_early_morning_decision_making: Optional[bool] = None
    extra_transportation_on_duty_by_4am: Optional[bool] = None
    extra_final_decision_before_6am: Optional[bool] = None
    extra_consultation_categories: List[str] = Field(default_factory=list)  # e.g., law enforcement, public works, etc.
    extra_announcement_channels: List[str] = Field(default_factory=list)     # e.g., automated notifications, website, radio, TV


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_main() -> str:
    return """
Extract the following items from the answer. Do NOT invent anything. Return null for missing fields.

Required fields:
- district_name: The name of the specific school district identified in the answer (string).
- official_urls: A list of URL(s) explicitly provided in the answer that point to the district's own official website pages
  documenting weather-related closure decision process (e.g., transportation page, weather policy, superintendent page).
  Only include URLs that are clearly on the district’s own official domain (allow subpages). Exclude third‑party/news/social sites.
- latest_decision_time: The latest morning time by which weather-related closure decisions may be made, as stated in the answer.
  Preserve formatting exactly as written in the answer (e.g., 'by 5:45 a.m.', 'no later than 6:00 AM').
- consulted_officials: The complete list of specific officials/departments that the superintendent consults with, as listed in the answer.
  Preserve each item as a separate string. If not provided, return an empty list.

Optional/partial-credit details (extract only if the answer mentions them; else return null/empty):
- extra_decision_by_superintendent: boolean, true if the answer states decisions are made by the superintendent in consultation.
- extra_monitoring_starts_day_before: boolean, true if the answer says monitoring begins 1+ days before expected weather.
- extra_early_morning_decision_making: boolean, true if the answer says active decision-making occurs in early morning hours.
- extra_transportation_on_duty_by_4am: boolean, true if the answer states transportation staff typically on duty by 4:00 a.m.
- extra_final_decision_before_6am: boolean, true if the answer states final decision is typically before 6:00 a.m.
- extra_consultation_categories: list of categories (e.g., 'law enforcement', 'public works/highway', 'transportation', 'neighboring districts') that the answer mentions.
- extra_announcement_channels: list of channels (e.g., 'automated notification system', 'district website', 'local radio', 'local TV') that the answer mentions.

Return a single JSON object with these fields. Do not include any fields other than those specified above.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _build_source_instruction(url_count: int) -> str:
    return (
        f"Source-grounded verification required. URLs supplied: {url_count}. "
        "Rely only on the provided official district webpage(s). If 0 URLs are supplied or if a page is clearly not on the "
        "district’s own official domain, you must judge the claim as Not Supported. Allow subpages under the district’s domain. "
        "Reject third‑party/news/social sites unless the district page itself hosts the content."
    )


def _fmt_list(items: List[str]) -> str:
    if not items:
        return "(none provided)"
    return "; ".join(items)


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def _verify_district_qualification(
    evaluator: Evaluator,
    parent_node,
    extracted: MaineDistrictExtraction,
) -> None:
    """
    Builds and verifies the 'District_Qualification' subtree (parallel, critical).
    """
    district_name = (extracted.district_name or "").strip()
    official_urls = _dedup_urls(extracted.official_urls)
    time_txt = (extracted.latest_decision_time or "").strip()

    qual_node = evaluator.add_parallel(
        id="District_Qualification",
        desc="Verify that the answer identifies a qualifying Maine school district and that the required information is documented on the district's official website.",
        parent=parent_node,
        critical=True
    )

    # 1) District_Name_Provided (custom, critical)
    evaluator.add_custom_node(
        result=bool(district_name),
        id="District_Name_Provided",
        desc="Answer provides the name of a specific school district.",
        parent=qual_node,
        critical=True
    )

    # 2) District_Located_in_Maine (verify with URLs; critical)
    node_loc = evaluator.add_leaf(
        id="District_Located_in_Maine",
        desc="The named school district is located in the state of Maine.",
        parent=qual_node,
        critical=True
    )
    claim_loc = (
        f"The provided official district webpage(s) are for the {district_name} school district (or equivalent district entity) "
        "and indicate it is located in the state of Maine (ME), e.g., by address, district name context, or page text."
        if district_name
        else "The provided official district webpage(s) belong to a school district in the state of Maine (ME)."
    )
    add_ins_loc = _build_source_instruction(len(official_urls)) + " Accept evidence such as a Maine address, 'ME' abbreviation, or explicit references to Maine communities served."
    await evaluator.verify(
        claim=claim_loc,
        node=node_loc,
        sources=official_urls if official_urls else None,
        additional_instruction=add_ins_loc
    )

    # 3) Official_Website_Documents_Latest_Decision_Time (verify existence of cutoff time; critical)
    node_cutoff = evaluator.add_leaf(
        id="Official_Website_Documents_Latest_Decision_Time",
        desc="The district's official website publicly documents the latest time in the morning by which weather-related school closure decisions may be made.",
        parent=qual_node,
        critical=True
    )
    if time_txt:
        claim_cutoff = (
            f"The official district webpage(s) explicitly state a latest morning decision time for weather-related school closures, specifically '{time_txt}'."
        )
    else:
        claim_cutoff = (
            "The official district webpage(s) explicitly state a latest time in the morning by which weather-related school closure decisions may be made "
            "(e.g., 'by 5:30 a.m.' / 'no later than 6:00 AM' / 'at the latest by X a.m.')."
        )
    add_ins_cutoff = _build_source_instruction(len(official_urls)) + " Look for phrases like 'no later than', 'by X a.m.', or 'at the latest'. The time must be a morning time (before 12:00 p.m.)."
    await evaluator.verify(
        claim=claim_cutoff,
        node=node_cutoff,
        sources=official_urls if official_urls else None,
        additional_instruction=add_ins_cutoff
    )

    # 4) Official_Website_Documents_Consulted_Officials (verify existence of consultation list; critical)
    node_consult = evaluator.add_leaf(
        id="Official_Website_Documents_Consulted_Officials",
        desc="The district's official website documents which specific officials/departments the superintendent consults with during the closure decision-making process.",
        parent=qual_node,
        critical=True
    )
    claim_consult = (
        "The official district webpage(s) list the specific officials or departments that the superintendent consults with when making weather-related closure decisions. "
        "General vagueness like 'various stakeholders' without a list is not sufficient."
    )
    add_ins_consult = _build_source_instruction(len(official_urls)) + " Look for explicit lists such as 'Transportation Director', 'Police Department', 'Public Works', 'Road Commissioner', etc."
    await evaluator.verify(
        claim=claim_consult,
        node=node_consult,
        sources=official_urls if official_urls else None,
        additional_instruction=add_ins_consult
    )


async def _verify_reported_required_fields(
    evaluator: Evaluator,
    parent_node,
    extracted: MaineDistrictExtraction,
) -> None:
    """
    Builds and verifies the 'Reported_Required_Answer_Fields' subtree (parallel, critical).
    """
    official_urls = _dedup_urls(extracted.official_urls)
    time_txt = (extracted.latest_decision_time or "").strip()
    consulted_list = extracted.consulted_officials or []

    rep_node = evaluator.add_parallel(
        id="Reported_Required_Answer_Fields",
        desc="Verify the required outputs are reported accurately from the district documentation.",
        parent=parent_node,
        critical=True
    )

    # 1) Latest_Decision_Time_Reported (critical)
    node_time_reported = evaluator.add_leaf(
        id="Latest_Decision_Time_Reported",
        desc="Answer states the latest decision time exactly as documented on the district's official website.",
        parent=rep_node,
        critical=True
    )
    claim_time_reported = (
        f"The answer reports the latest morning decision time for weather-related school closures as '{time_txt}'. "
        "According to the provided official district webpage(s), that time matches exactly the documented latest decision time."
    )
    add_ins_time_reported = (
        _build_source_instruction(len(official_urls))
        + " If the answer omits the time or states a different time, judge Incorrect. "
          "Allow trivial formatting differences (e.g., 'a.m.' vs 'AM'), but not any numeric or substantive differences."
    )
    await evaluator.verify(
        claim=claim_time_reported,
        node=node_time_reported,
        sources=official_urls if official_urls else None,
        additional_instruction=add_ins_time_reported
    )

    # 2) Consulted_Officials_Reported_Completely (critical)
    node_consult_reported = evaluator.add_leaf(
        id="Consulted_Officials_Reported_Completely",
        desc="Answer lists the complete set of consulted officials/departments exactly as documented on the district's official website.",
        parent=rep_node,
        critical=True
    )
    list_str = _fmt_list(consulted_list)
    claim_consult_reported = (
        f"The answer lists the consulted officials/departments as: {list_str}. "
        "According to the provided official district webpage(s), this list is complete and contains no omissions or additions "
        "compared to the set documented on the site."
    )
    add_ins_consult_reported = (
        _build_source_instruction(len(official_urls))
        + " Judge completeness as set equivalence (ignore order and trivial formatting). "
          "Accept minor synonyms (e.g., 'PD' vs 'Police Department'). If the answer gives only a subset or includes roles not in the webpage, judge Incorrect. "
          "If the answer provides no list, judge Incorrect."
    )
    await evaluator.verify(
        claim=claim_consult_reported,
        node=node_consult_reported,
        sources=official_urls if official_urls else None,
        additional_instruction=add_ins_consult_reported
    )


async def _verify_additional_details(
    evaluator: Evaluator,
    parent_node,
    extracted: MaineDistrictExtraction,
) -> None:
    """
    Builds and verifies the optional 'Additional_Process_Details_From_Constraints' subtree (parallel, non-critical).
    These checks award partial credit if the official district documentation includes them.
    """
    official_urls = _dedup_urls(extracted.official_urls)

    add_node = evaluator.add_parallel(
        id="Additional_Process_Details_From_Constraints",
        desc="Check additional process details mentioned in the provided constraints section (treated as optional/partial-credit unless explicitly required by the question).",
        parent=parent_node,
        critical=False
    )

    common_ins = _build_source_instruction(len(official_urls)) + " If not stated on the district webpage(s), judge Not Supported."

    # Decision_Made_By_Superintendent_In_Consultation
    n1 = evaluator.add_leaf(
        id="Decision_Made_By_Superintendent_In_Consultation",
        desc="District documentation states that closure decisions are made by the superintendent in consultation with other officials/departments.",
        parent=add_node,
        critical=False
    )
    c1 = "The district documentation states that closure decisions are made by the superintendent in consultation with other officials/departments."
    await evaluator.verify(
        claim=c1, node=n1, sources=official_urls if official_urls else None, additional_instruction=common_ins
    )

    # Monitoring_Begins_At_Least_One_Day_Before
    n2 = evaluator.add_leaf(
        id="Monitoring_Begins_At_Least_One_Day_Before",
        desc="District documentation states that monitoring of forecasts begins 1 day or more before expected severe weather events.",
        parent=add_node,
        critical=False
    )
    c2 = "The district documentation states that monitoring of weather forecasts begins at least one day prior to an anticipated severe weather event."
    await evaluator.verify(
        claim=c2, node=n2, sources=official_urls if official_urls else None, additional_instruction=common_ins
    )

    # Active_Decision_Making_Early_Morning
    n3 = evaluator.add_leaf(
        id="Active_Decision_Making_Early_Morning",
        desc="District documentation states that active decision-making occurs in the early morning hours.",
        parent=add_node,
        critical=False
    )
    c3 = "The district documentation states that active closure decision-making occurs in the early morning hours (e.g., roughly between 4:00 a.m. and 6:00 a.m.)."
    await evaluator.verify(
        claim=c3, node=n3, sources=official_urls if official_urls else None, additional_instruction=common_ins
    )

    # Transportation_Staff_On_Duty_By_4am
    n4 = evaluator.add_leaf(
        id="Transportation_Staff_On_Duty_By_4am",
        desc="District documentation states that transportation staff are typically on duty by 4:00 a.m.",
        parent=add_node,
        critical=False
    )
    c4 = "The district documentation states that transportation staff are typically on duty by 4:00 a.m."
    await evaluator.verify(
        claim=c4, node=n4, sources=official_urls if official_urls else None, additional_instruction=common_ins
    )

    # Final_Decision_Typically_Before_6am
    n5 = evaluator.add_leaf(
        id="Final_Decision_Typically_Before_6am",
        desc="If stated in the district documentation, it indicates final closure decisions are typically made before 6:00 a.m.",
        parent=add_node,
        critical=False
    )
    c5 = "The district documentation indicates that final closure decisions are typically made before 6:00 a.m."
    await evaluator.verify(
        claim=c5, node=n5, sources=official_urls if official_urls else None, additional_instruction=common_ins
    )

    # Consultation_Includes_Source_Categories
    n6 = evaluator.add_leaf(
        id="Consultation_Includes_Source_Categories",
        desc="If stated in the district documentation, the consultation set includes the source categories listed in the constraints (e.g., law enforcement, public works/highway, transportation, neighboring districts).",
        parent=add_node,
        critical=False
    )
    c6 = (
        "The district documentation indicates that the superintendent consults with one or more of the following categories: "
        "law enforcement, public works/highway, transportation, or neighboring districts."
    )
    await evaluator.verify(
        claim=c6, node=n6, sources=official_urls if official_urls else None, additional_instruction=common_ins
    )

    # Announcements_Use_Stated_Channels
    n7 = evaluator.add_leaf(
        id="Announcements_Use_Stated_Channels",
        desc="If stated in the district documentation, closure announcements are communicated through the channels listed in the constraints (e.g., automated notification systems, district website, local radio, local TV).",
        parent=add_node,
        critical=False
    )
    c7 = (
        "The district documentation states that closure announcements are communicated via one or more of these channels: "
        "automated notification system, district website, local radio, or local TV."
    )
    await evaluator.verify(
        claim=c7, node=n7, sources=official_urls if official_urls else None, additional_instruction=common_ins
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
    model: str = "o4-mini",
) -> Dict:
    """
    Entry point for evaluating an answer for the Maine school district weather decision-time task.
    """
    # Initialize evaluator with a sequential root to enforce gating across major sections
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

    # Record high-level evaluation description
    evaluator.add_custom_info({"description": EVAL_DESCRIPTION}, info_type="eval_overview")

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_main(),
        template_class=MaineDistrictExtraction,
        extraction_name="extracted_fields",
    )

    # Build and verify trees according to rubric
    # 1) District Qualification (critical, parallel)
    await _verify_district_qualification(evaluator, root, extracted)

    # 2) Reported Required Fields (critical, parallel) - evaluated only if prior step passes due to sequential root
    await _verify_reported_required_fields(evaluator, root, extracted)

    # 3) Additional optional process details (non-critical, parallel)
    await _verify_additional_details(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()