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
TASK_ID = "md_school_district_identification_and_verification"
TASK_DESCRIPTION = """A Maryland school district operates exactly 44 schools and serves between 25,000 and 27,000 students. This county-level public school system is governed by a Board of Education consisting of five elected members and ranks as either the 9th or 10th largest school system in Maryland. The district has 22 elementary schools.

The current superintendent of this district holds a doctorate degree (Ed.D.) and was appointed to the position effective July 1, 2022, with an initial four-year contract term. In February 2026, the Board of Education renewed the superintendent's contract for four more years during a Wednesday meeting, which included a salary increase. The superintendent received the 2025 Deans' Recognition Award, which was announced in October 2025.

Identify this school district and provide the following information:
1. The name of the school district
2. The full name and title of the current superintendent
3. The specific salary increase amount associated with the February 2026 contract renewal
4. The exact date (including day of week and full date) of the Board of Education meeting where the contract renewal was approved

For each piece of information provided, include valid reference URLs that confirm the details.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DistrictOutput(BaseModel):
    name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class SuperintendentOutput(BaseModel):
    full_name: Optional[str] = None
    title: Optional[str] = None
    degree: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class RenewalOutput(BaseModel):
    salary_increase_amount: Optional[str] = None  # e.g., "$15,000", "5%", "$8,500"
    meeting_date_text: Optional[str] = None       # e.g., "Wednesday, February 12, 2026"
    meeting_weekday: Optional[str] = None         # e.g., "Wednesday"
    meeting_date: Optional[str] = None            # e.g., "February 12, 2026"
    urls: List[str] = Field(default_factory=list)


class ConstraintsExtraction(BaseModel):
    location_state: Optional[str] = None                  # e.g., "Maryland"
    org_level: Optional[str] = None                       # e.g., "county-level"
    system_type: Optional[str] = None                     # e.g., "public school system"
    total_schools: Optional[str] = None                   # e.g., "44"
    students_served: Optional[str] = None                 # e.g., "26,000"
    elementary_schools: Optional[str] = None              # e.g., "22"
    ranking_md: Optional[str] = None                      # e.g., "9th largest", "10th largest"
    governance_body: Optional[str] = None                 # e.g., "Board of Education"
    board_members_elected_count: Optional[str] = None     # e.g., "five elected members", "5 elected members"
    superintendent_degree: Optional[str] = None           # e.g., "Ed.D."
    appointment_effective_date: Optional[str] = None      # e.g., "July 1, 2022"
    initial_contract_term: Optional[str] = None           # e.g., "four years", "4 years"
    renewal_extension_term: Optional[str] = None          # e.g., "four years", "4 years"
    award_name: Optional[str] = None                      # e.g., "2025 Deans' Recognition Award"
    award_announcement_month_year: Optional[str] = None   # e.g., "October 2025"

    urls_stats: List[str] = Field(default_factory=list)            # counts / enrollment / ranking
    urls_governance: List[str] = Field(default_factory=list)       # board / governance pages
    urls_board_composition: List[str] = Field(default_factory=list)
    urls_appointment: List[str] = Field(default_factory=list)      # appointment & contract info
    urls_award: List[str] = Field(default_factory=list)            # award announcement/source
    urls_ranking: List[str] = Field(default_factory=list)          # ranking sources


class FullExtraction(BaseModel):
    district: Optional[DistrictOutput] = None
    superintendent: Optional[SuperintendentOutput] = None
    renewal: Optional[RenewalOutput] = None
    constraints: Optional[ConstraintsExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract the following structured information from the answer. Return null for any missing field and ensure URL fields contain only valid, complete URLs explicitly present in the answer text.

A) district:
- name: the school district name exactly as stated.
- urls: list of URLs that confirm the district identity (homepage, about page, official profiles, etc.).

B) superintendent:
- full_name: the current superintendent's full name.
- title: the superintendent's title (e.g., "Superintendent of Schools").
- degree: any terminal degree acronym explicitly stated (e.g., "Ed.D.", "Ph.D."). If multiple, include the one that supports Ed.D. if present.
- urls: list of URLs that confirm the superintendent's identity/title.

C) renewal:
- salary_increase_amount: the specific salary increase amount tied to the Feb 2026 contract renewal (e.g., "$15,000", "5%").
- meeting_date_text: the exact date string as presented in the answer that includes weekday and full date (e.g., "Wednesday, February 12, 2026"), if present.
- meeting_weekday: the weekday alone if present (e.g., "Wednesday").
- meeting_date: the full calendar date string if present (e.g., "February 12, 2026").
- urls: list of URLs that directly reference the Feb 2026 renewal action and salary increase and/or the approval meeting.

D) constraints:
- location_state: state of the district (e.g., "Maryland").
- org_level: organizational level if mentioned (e.g., "county-level").
- system_type: system type if mentioned (e.g., "public school system").
- total_schools: total number of schools operated (as stated, e.g., "44").
- students_served: number of students served (as stated, e.g., "26,000").
- elementary_schools: count of elementary schools (as stated, e.g., "22").
- ranking_md: ranking phrase if given (e.g., "9th largest", "10th largest").
- governance_body: governing body label (e.g., "Board of Education").
- board_members_elected_count: phrase or number stating the board has five elected members (e.g., "five elected members", "5 elected members").
- superintendent_degree: degree credentials text that supports Ed.D. if available.
- appointment_effective_date: the effective appointment date (e.g., "July 1, 2022").
- initial_contract_term: initial term length (e.g., "four years", "4 years").
- renewal_extension_term: renewal extension length (e.g., "four years", "4 years").
- award_name: exactly "2025 Deans' Recognition Award" if present.
- award_announcement_month_year: month-year phrase for the award announcement (e.g., "October 2025").

- urls_stats: list of URLs that support counts, enrollment, ranking, or district profile facts.
- urls_governance: list of URLs that support governance and "Board of Education".
- urls_board_composition: list of URLs that support the number of elected members.
- urls_appointment: list of URLs that support appointment date, initial term, renewal term details.
- urls_award: list of URLs that support the 2025 Deans' Recognition Award and announcement timing.
- urls_ranking: list of URLs that support the 9th/10th largest ranking.

Output JSON structure:
{
  "district": {...},
  "superintendent": {...},
  "renewal": {...},
  "constraints": {...}
}
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_urls(*url_lists: Optional[List[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if not u:
                continue
            uu = u.strip()
            if uu and uu not in seen:
                seen.add(uu)
                out.append(uu)
    return out


def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_non_output_constraints(evaluator: Evaluator, parent_node, data: FullExtraction) -> None:
    """
    Build and verify all non-output constraints under a critical parallel node.
    """
    c = data.constraints or ConstraintsExtraction()
    d = data.district or DistrictOutput()
    sup = data.superintendent or SuperintendentOutput()
    ren = data.renewal or RenewalOutput()

    constraints_node = evaluator.add_parallel(
        id="identify_correct_district_meets_non_output_constraints",
        desc="The identified district/superintendent satisfies all constraints that are not already verified via the four required outputs.",
        parent=parent_node,
        critical=True
    )

    # Aggregate URLs for constraints
    district_urls = d.urls
    sup_urls = _unique_urls(sup.urls, c.urls_appointment)
    renewal_urls = _unique_urls(ren.urls, c.urls_appointment)
    stats_urls = _unique_urls(c.urls_stats, c.urls_ranking, d.urls)
    governance_urls = _unique_urls(c.urls_governance, c.urls_board_composition, d.urls)
    award_urls = c.urls_award
    all_urls = _unique_urls(district_urls, sup_urls, renewal_urls, stats_urls, governance_urls, award_urls)

    # 1) District in Maryland
    n1 = evaluator.add_leaf(
        id="district_in_maryland",
        desc="District is located in Maryland.",
        parent=constraints_node,
        critical=True
    )
    name_part = f"'{d.name}' " if _nonempty(d.name) else ""
    claim_1 = f"The school district {name_part}is located in Maryland."
    await evaluator.verify(
        claim=claim_1,
        node=n1,
        sources=_unique_urls(district_urls, stats_urls) or all_urls,
        additional_instruction="Confirm the district is a Maryland public school district; accept if the page explicitly mentions Maryland (MD) or is an official Maryland county schools page."
    )

    # 2) County-level public school system
    n2 = evaluator.add_leaf(
        id="district_county_level_public_system",
        desc="District is a county-level public school system.",
        parent=constraints_node,
        critical=True
    )
    claim_2 = "This district is a county-level public school system."
    await evaluator.verify(
        claim=claim_2,
        node=n2,
        sources=_unique_urls(district_urls, governance_urls) or all_urls,
        additional_instruction="Accept if the site or official description indicates 'County Public Schools' or otherwise clearly identifies it as a county-level public school system."
    )

    # 3) Exactly 44 schools
    n3 = evaluator.add_leaf(
        id="district_operates_44_schools",
        desc="District operates exactly 44 schools.",
        parent=constraints_node,
        critical=True
    )
    claim_3 = "The district operates exactly 44 schools."
    await evaluator.verify(
        claim=claim_3,
        node=n3,
        sources=stats_urls or all_urls,
        additional_instruction="Look for an official profile, 'About' page, or data dashboard stating the district operates 44 schools."
    )

    # 4) Serves 25,000–27,000 students
    n4 = evaluator.add_leaf(
        id="district_serves_25000_to_27000_students",
        desc="District serves between 25,000 and 27,000 students.",
        parent=constraints_node,
        critical=True
    )
    claim_4 = "The district serves between 25,000 and 27,000 students (inclusive)."
    await evaluator.verify(
        claim=claim_4,
        node=n4,
        sources=stats_urls or all_urls,
        additional_instruction="Accept if the page shows any recent student count falling in [25,000, 27,000]; approximate phrasing like 'about 26,000' is acceptable."
    )

    # 5) Exactly 22 elementary schools
    n5 = evaluator.add_leaf(
        id="district_has_22_elementary_schools",
        desc="District has exactly 22 elementary schools.",
        parent=constraints_node,
        critical=True
    )
    claim_5 = "The district has exactly 22 elementary schools."
    await evaluator.verify(
        claim=claim_5,
        node=n5,
        sources=stats_urls or all_urls,
        additional_instruction="Look for official counts by school level indicating 22 elementary schools."
    )

    # 6) 9th or 10th largest in Maryland
    n6 = evaluator.add_leaf(
        id="district_ranking_9th_or_10th_in_md",
        desc="District ranks as either the 9th or 10th largest school system in Maryland.",
        parent=constraints_node,
        critical=True
    )
    claim_6 = "The district is the 9th or 10th largest school system in Maryland."
    await evaluator.verify(
        claim=claim_6,
        node=n6,
        sources=_unique_urls(stats_urls, c.urls_ranking) or all_urls,
        additional_instruction="Accept if the source explicitly says '9th largest' or '10th largest' in Maryland."
    )

    # 7) Governed by a Board of Education
    n7 = evaluator.add_leaf(
        id="district_governed_by_board_of_education",
        desc="District is governed by a Board of Education.",
        parent=constraints_node,
        critical=True
    )
    claim_7 = "The district is governed by a Board of Education."
    await evaluator.verify(
        claim=claim_7,
        node=n7,
        sources=governance_urls or all_urls,
        additional_instruction="Look for pages describing district governance referencing a 'Board of Education'."
    )

    # 8) Board has five elected members
    n8 = evaluator.add_leaf(
        id="district_board_consists_of_five_elected_members",
        desc="The Board of Education consists of five elected members.",
        parent=constraints_node,
        critical=True
    )
    claim_8 = "The Board of Education consists of five elected members."
    await evaluator.verify(
        claim=claim_8,
        node=n8,
        sources=_unique_urls(c.urls_board_composition, governance_urls) or all_urls,
        additional_instruction="Count only the elected adult board members; student members or ex officio members should not be counted toward the five elected members."
    )

    # 9) Superintendent holds an Ed.D.
    n9 = evaluator.add_leaf(
        id="superintendent_has_edd",
        desc="Current superintendent holds an Ed.D.",
        parent=constraints_node,
        critical=True
    )
    claim_9 = "The current superintendent holds an Ed.D. (Doctor of Education) degree."
    await evaluator.verify(
        claim=claim_9,
        node=n9,
        sources=sup_urls or all_urls,
        additional_instruction="Confirm that the superintendent's credentials include Ed.D.; accept reasonable formatting variations like 'EdD' or 'Doctor of Education'."
    )

    # 10) Appointed effective July 1, 2022
    n10 = evaluator.add_leaf(
        id="superintendent_appointed_effective_july_1_2022",
        desc="Superintendent was appointed effective July 1, 2022.",
        parent=constraints_node,
        critical=True
    )
    claim_10 = "The superintendent was appointed effective July 1, 2022."
    await evaluator.verify(
        claim=claim_10,
        node=n10,
        sources=_unique_urls(c.urls_appointment, sup_urls) or all_urls,
        additional_instruction="Look for official appointment/contract announcements or board news specifying the effective appointment date as July 1, 2022."
    )

    # 11) Initial contract term four years
    n11 = evaluator.add_leaf(
        id="superintendent_initial_contract_four_years",
        desc="Superintendent’s initial contract term was four years.",
        parent=constraints_node,
        critical=True
    )
    claim_11 = "The superintendent’s initial contract term was four years."
    await evaluator.verify(
        claim=claim_11,
        node=n11,
        sources=_unique_urls(c.urls_appointment, sup_urls) or all_urls,
        additional_instruction="Verify that the initial contract length specified was four (4) years."
    )

    # 12) Renewal extended four more years in Feb 2026
    n12 = evaluator.add_leaf(
        id="renewal_extended_four_more_years",
        desc="In February 2026, the superintendent’s contract was renewed for four more years.",
        parent=constraints_node,
        critical=True
    )
    claim_12 = "In February 2026, the superintendent’s contract was renewed for four more years."
    await evaluator.verify(
        claim=claim_12,
        node=n12,
        sources=renewal_urls or all_urls,
        additional_instruction="The source should explicitly tie the renewal action in February 2026 to an additional four-year term."
    )

    # 13) 2025 Deans' Recognition Award
    n13 = evaluator.add_leaf(
        id="deans_recognition_award_2025",
        desc="Superintendent received the 2025 Deans' Recognition Award.",
        parent=constraints_node,
        critical=True
    )
    claim_13 = "The superintendent received the 2025 Deans' Recognition Award."
    await evaluator.verify(
        claim=claim_13,
        node=n13,
        sources=award_urls or all_urls,
        additional_instruction="Confirm the award name and year exactly as '2025 Deans' Recognition Award'."
    )

    # 14) Award announced in October 2025
    n14 = evaluator.add_leaf(
        id="award_announced_october_2025",
        desc="The Deans' Recognition Award announcement occurred in October 2025.",
        parent=constraints_node,
        critical=True
    )
    claim_14 = "The Deans' Recognition Award announcement occurred in October 2025."
    await evaluator.verify(
        claim=claim_14,
        node=n14,
        sources=award_urls or all_urls,
        additional_instruction="Confirm that the announcement or news release date for this award was in October 2025."
    )


async def verify_requested_outputs(evaluator: Evaluator, parent_node, data: FullExtraction) -> None:
    """
    Verify the four explicitly requested outputs under a critical parallel node.
    Includes existence checks (critical) and source-grounded verification leaves (critical).
    """
    d = data.district or DistrictOutput()
    sup = data.superintendent or SuperintendentOutput()
    ren = data.renewal or RenewalOutput()
    c = data.constraints or ConstraintsExtraction()

    outputs_node = evaluator.add_parallel(
        id="provide_four_requested_outputs_with_citations",
        desc="Provide the four explicitly requested outputs, each supported by at least one valid reference URL that substantiates the claim.",
        parent=parent_node,
        critical=True
    )

    # URL aggregations for fallback
    all_urls = _unique_urls(
        d.urls, sup.urls, ren.urls,
        c.urls_stats, c.urls_governance, c.urls_board_composition,
        c.urls_appointment, c.urls_award, c.urls_ranking
    )

    # Output 1: District name with URL(s)
    # Existence gate (critical)
    e1 = evaluator.add_custom_node(
        result=_nonempty(d.name) and len(d.urls) > 0,
        id="output_district_name_exists",
        desc="Output #1 existence: district name present with at least one URL.",
        parent=outputs_node,
        critical=True
    )
    # Verification leaf (critical)
    o1 = evaluator.add_leaf(
        id="output_district_name_with_url",
        desc="Provide the school district name with URL(s) confirming the district identity.",
        parent=outputs_node,
        critical=True
    )
    claim_o1 = f"This webpage confirms the school district is '{d.name}'." if _nonempty(d.name) else "This webpage confirms the identified school district."
    await evaluator.verify(
        claim=claim_o1,
        node=o1,
        sources=d.urls or all_urls,
        additional_instruction="Verify that the page clearly identifies the district by this exact name (or an equivalent official naming variant), and it is a Maryland public school district."
    )

    # Output 2: Superintendent full name and title with URL(s)
    e2 = evaluator.add_custom_node(
        result=_nonempty(sup.full_name) and _nonempty(sup.title) and len(sup.urls) > 0,
        id="output_superintendent_exists",
        desc="Output #2 existence: superintendent full name and title present with at least one URL.",
        parent=outputs_node,
        critical=True
    )
    o2 = evaluator.add_leaf(
        id="output_superintendent_name_title_with_url",
        desc="Provide the current superintendent’s full name and title with URL(s) confirming both.",
        parent=outputs_node,
        critical=True
    )
    claim_o2 = f"The current superintendent is {sup.full_name} with the title '{sup.title}'."
    await evaluator.verify(
        claim=claim_o2,
        node=o2,
        sources=sup.urls or all_urls,
        additional_instruction="The page should clearly show this individual currently holds the superintendent title; accept minor formatting variations (e.g., 'Superintendent of Schools')."
    )

    # Output 3: Salary increase amount with URL(s) tied to Feb 2026 renewal
    e3 = evaluator.add_custom_node(
        result=_nonempty(ren.salary_increase_amount) and len(ren.urls) > 0,
        id="output_salary_increase_exists",
        desc="Output #3 existence: salary increase amount present with at least one renewal-related URL.",
        parent=outputs_node,
        critical=True
    )
    o3 = evaluator.add_leaf(
        id="output_salary_increase_amount_with_url",
        desc="Provide the specific salary increase amount, and URL(s) showing it is the increase associated with the February 2026 contract renewal.",
        parent=outputs_node,
        critical=True
    )
    claim_o3 = f"The salary increase associated with the February 2026 contract renewal was {ren.salary_increase_amount}."
    await evaluator.verify(
        claim=claim_o3,
        node=o3,
        sources=ren.urls or all_urls,
        additional_instruction="The source must explicitly connect this increase amount to the Feb 2026 renewal action. Accept dollar or percent expressions as long as the linkage is explicit."
    )

    # Output 4: Exact Board meeting date (weekday + full date) for approval with URL(s)
    e4 = evaluator.add_custom_node(
        result=( (_nonempty(ren.meeting_date_text) or (_nonempty(ren.meeting_weekday) and _nonempty(ren.meeting_date))) and len(ren.urls) > 0 ),
        id="output_meeting_date_exists",
        desc="Output #4 existence: meeting date (weekday + full date) for approval present with at least one URL.",
        parent=outputs_node,
        critical=True
    )
    o4 = evaluator.add_leaf(
        id="output_meeting_exact_date_with_url",
        desc="Provide the exact Board of Education meeting date (weekday + full calendar date) when the renewal was approved, and URL(s) confirming this approval date; the date must be in February 2026 and fall on a Wednesday.",
        parent=outputs_node,
        critical=True
    )
    # Build date text for claim
    if _nonempty(ren.meeting_date_text):
        date_phrase = ren.meeting_date_text.strip()
    elif _nonempty(ren.meeting_weekday) and _nonempty(ren.meeting_date):
        date_phrase = f"{ren.meeting_weekday.strip()}, {ren.meeting_date.strip()}"
    else:
        date_phrase = "a Wednesday in February 2026"  # fallback to still allow verification attempt

    claim_o4 = f"The Board of Education approved the contract renewal on {date_phrase}, and that date is a Wednesday in February 2026."
    await evaluator.verify(
        claim=claim_o4,
        node=o4,
        sources=ren.urls or all_urls,
        additional_instruction="Confirm that the renewal approval occurred on the specified date in February 2026. If the weekday is not printed on the page but the date is, you may compute the weekday from the date and accept only if it is a Wednesday."
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
    Evaluate an answer against the Maryland school district identification task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Per rubric: evaluate constraints first, then outputs
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=FullExtraction,
        extraction_name="extracted_info"
    )

    # Build a critical sequential "main" node to mirror rubric's critical root behavior
    main_node = evaluator.add_sequential(
        id="main_evaluation",
        desc="Identify the correct Maryland county-level public school district and verify all constraints and required outputs with supporting URLs.",
        parent=root,
        critical=True
    )

    # 1) Non-output constraints (critical parallel)
    await verify_non_output_constraints(evaluator, main_node, extracted)

    # 2) Provide the four requested outputs with citations (critical parallel)
    await verify_requested_outputs(evaluator, main_node, extracted)

    # Return structured summary
    return evaluator.get_summary()