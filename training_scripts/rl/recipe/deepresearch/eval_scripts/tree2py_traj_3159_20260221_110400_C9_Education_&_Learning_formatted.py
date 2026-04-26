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
TASK_ID = "big_ten_universities_criteria"
TASK_DESCRIPTION = """
Identify three public universities that are current members of the Big Ten Conference (as of the 2024-25 academic year) and satisfy ALL of the following criteria:

1. The university must be a member of the Association of American Universities (AAU), demonstrating recognized research excellence.
2. The university must be designated as a land-grant institution under the Morrill Act.
3. The university must have a total fall enrollment exceeding 40,000 students (as reported in the most recent academic year data available).
4. The university must have annual research expenditures exceeding $800 million, as documented in fiscal year 2024 or the most recent available data.
5. The university must field an NCAA Division I FBS football team competing in the Big Ten Conference.
6. The university must have participated in the NCAA Division I Men's Basketball Tournament at least once between 2015 and 2025.
7. The university's athletic department must have total annual revenues exceeding $150 million.
8. The university's current president or chancellor must have been appointed to their position after January 1, 2020.
9. The university must be located in a U.S. state that borders at least one of the Great Lakes (Lakes Superior, Michigan, Huron, Erie, or Ontario).
10. The university must offer degree programs through at least 10 different colleges or schools, including a College of Engineering, a College (or School) of Business, and a College (or School) of Education.

For each of the three universities you identify, provide:
- The full official name of the university
- The city and state where the main campus is located
- The current total enrollment figure (with academic year specified)
- The current annual research expenditure amount (with fiscal year specified)
- The name and appointment date of the current president or chancellor
- The total athletic department revenue (with fiscal year specified)
- A URL link to the university's official Big Ten Conference member page or the Big Ten Conference website confirming membership
- A URL link to the AAU website confirming the university's AAU membership
- A URL link to a USDA NIFA resource or official university source confirming land-grant designation
- A URL link documenting at least one NCAA Tournament appearance between 2015-2025
- A URL link to the university's official website listing its colleges and schools
"""

GREAT_LAKES_STATES = {
    "Minnesota", "Wisconsin", "Illinois", "Indiana", "Michigan", "Ohio", "Pennsylvania", "New York"
}
APPOINTMENT_CUTOFF_DATE = "2020-01-01"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityInfo(BaseModel):
    """Information for a single university as provided by the agent's answer."""
    full_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None

    enrollment_value: Optional[str] = None
    enrollment_year: Optional[str] = None
    enrollment_url: Optional[str] = None

    research_expenditure_value: Optional[str] = None
    research_fiscal_year: Optional[str] = None
    research_expenditure_url: Optional[str] = None

    president_or_chancellor: Optional[str] = None
    president_appointment_date: Optional[str] = None
    president_appointment_url: Optional[str] = None

    athletic_revenue_value: Optional[str] = None
    athletic_revenue_fiscal_year: Optional[str] = None
    athletic_revenue_url: Optional[str] = None

    big_ten_member_url: Optional[str] = None
    aau_membership_url: Optional[str] = None
    land_grant_url: Optional[str] = None
    ncaa_tournament_url: Optional[str] = None
    colleges_listing_url: Optional[str] = None

    # Optional supporting URLs if provided
    football_url: Optional[str] = None
    public_status_url: Optional[str] = None
    location_url: Optional[str] = None


class UniversitiesExtraction(BaseModel):
    """Top-level extraction model for up to three universities."""
    universities: List[UniversityInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to three universities listed in the answer along with all required details and URLs. Return a JSON object with a 'universities' array. For each university, extract the following fields exactly as stated in the answer; if a field is missing, set it to null:

    Identification:
    - full_name
    - city
    - state

    Enrollment (most recent academic year):
    - enrollment_value
    - enrollment_year
    - enrollment_url (institutional source or NCES link if provided)

    Research Expenditures (FY 2024 or most recent available):
    - research_expenditure_value
    - research_fiscal_year
    - research_expenditure_url (NSF HERD or institutional source)

    Leadership:
    - president_or_chancellor (current title/name)
    - president_appointment_date (ISO or textual date as stated)
    - president_appointment_url (institutional source)

    Athletics:
    - athletic_revenue_value
    - athletic_revenue_fiscal_year
    - athletic_revenue_url (USA Today, Knight Commission, or institutional source)

    Required URLs:
    - big_ten_member_url (official Big Ten Conference website or the university’s Big Ten member page)
    - aau_membership_url (AAU official website page confirming membership)
    - land_grant_url (USDA NIFA or institutional page confirming land-grant designation)
    - ncaa_tournament_url (page documenting at least one NCAA Men's Basketball Tournament appearance between 2015-2025)
    - colleges_listing_url (university official page listing colleges/schools)

    Optional supporting URLs (if present in the answer):
    - football_url (official athletics page confirming FBS football in Big Ten, if provided)
    - public_status_url (institutional page confirming public status, if provided)
    - location_url (institutional page describing campus location, if provided)

    Only include URLs explicitly present in the answer text. Do not infer or create new URLs. Normalize URLs; if protocol is missing, prepend http://.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def has_text(x: Optional[str]) -> bool:
    return bool(x and isinstance(x, str) and x.strip())


def gather_all_urls(uni: UniversityInfo) -> List[str]:
    urls = [
        uni.big_ten_member_url,
        uni.aau_membership_url,
        uni.land_grant_url,
        uni.ncaa_tournament_url,
        uni.colleges_listing_url,
        uni.research_expenditure_url,
        uni.enrollment_url,
        uni.athletic_revenue_url,
        uni.president_appointment_url,
        uni.football_url,
        uni.public_status_url,
        uni.location_url,
    ]
    return [u for u in urls if has_text(u)]


# --------------------------------------------------------------------------- #
# Verification for a single university                                        #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityInfo,
    idx: int,
) -> None:
    """
    Build verification sub-tree and run checks for a single university.
    """

    # University i node (non-critical to allow partial credit across universities)
    uni_node = evaluator.add_parallel(
        id=f"University_{idx + 1}",
        desc=f"{['First','Second','Third'][idx]} identified university meets all requirement categories",
        parent=parent_node,
        critical=False
    )

    # --------------------- Conference Membership ---------------------------
    conf_node = evaluator.add_parallel(
        id=f"U{idx+1}_Conference_Membership",
        desc="University is confirmed as a Big Ten Conference member for the 2024-25 season",
        parent=uni_node,
        critical=True
    )

    # URL existence
    evaluator.add_custom_node(
        result=has_text(uni.big_ten_member_url),
        id=f"U{idx+1}_Conference_URL",
        desc="Reference URL from official Big Ten Conference source confirming membership",
        parent=conf_node,
        critical=True
    )

    # Verify membership via URL
    conf_verify = evaluator.add_leaf(
        id=f"U{idx+1}_Conference_Verify",
        desc="Big Ten Conference membership verification (2024-25)",
        parent=conf_node,
        critical=True
    )
    claim = f"The university {uni.full_name or 'UNKNOWN'} is a current member of the Big Ten Conference for the 2024-25 academic year."
    await evaluator.verify(
        claim=claim,
        node=conf_verify,
        sources=uni.big_ten_member_url,
        additional_instruction="Confirm this university appears on the Big Ten Conference official member list for the 2024-25 season."
    )

    # --------------------- Research Classification -------------------------
    research_main = evaluator.add_parallel(
        id=f"U{idx+1}_Research_Classification",
        desc="University meets research institution requirements",
        parent=uni_node,
        critical=True
    )

    # AAU Membership
    evaluator.add_custom_node(
        result=has_text(uni.aau_membership_url),
        id=f"U{idx+1}_AAU_URL",
        desc="Reference URL from AAU official source confirming membership",
        parent=research_main,
        critical=True
    )
    aau_leaf = evaluator.add_leaf(
        id=f"U{idx+1}_AAU_Membership",
        desc="University is confirmed as an AAU member institution",
        parent=research_main,
        critical=True
    )
    aau_claim = f"The university {uni.full_name or 'UNKNOWN'} is a member of the Association of American Universities (AAU)."
    await evaluator.verify(
        claim=aau_claim,
        node=aau_leaf,
        sources=uni.aau_membership_url,
        additional_instruction="Verify the university appears on AAU's official member list."
    )

    # Research Expenditures > $800M
    evaluator.add_custom_node(
        result=has_text(uni.research_expenditure_url),
        id=f"U{idx+1}_Research_Expenditure_URL",
        desc="Reference URL from NSF HERD or institutional source documenting research expenditure amount",
        parent=research_main,
        critical=True
    )
    research_leaf = evaluator.add_leaf(
        id=f"U{idx+1}_Research_Expenditure",
        desc="University has annual research expenditures exceeding $800 million",
        parent=research_main,
        critical=True
    )
    research_claim = (
        f"The university {uni.full_name or 'UNKNOWN'} reported annual research expenditures of "
        f"{uni.research_expenditure_value or 'UNKNOWN'} in fiscal year {uni.research_fiscal_year or 'UNKNOWN'}, "
        f"which exceeds $800 million."
    )
    await evaluator.verify(
        claim=research_claim,
        node=research_leaf,
        sources=uni.research_expenditure_url,
        additional_instruction="Confirm the research expenditure figure on the provided page and determine whether it is above $800 million; if the page shows FY 2023 or latest available instead of FY 2024, it is acceptable."
    )

    # --------------------- Institution Type --------------------------------
    inst_node = evaluator.add_parallel(
        id=f"U{idx+1}_Institution_Type",
        desc="University meets institutional classification requirements",
        parent=uni_node,
        critical=True
    )

    # Public Status (no explicit URL leaf in rubric; verify using any available sources)
    public_leaf = evaluator.add_leaf(
        id=f"U{idx+1}_Public_Status",
        desc="University is confirmed as a public state university",
        parent=inst_node,
        critical=True
    )
    public_claim = f"The university {uni.full_name or 'UNKNOWN'} is a public state university."
    await evaluator.verify(
        claim=public_claim,
        node=public_leaf,
        sources=gather_all_urls(uni),
        additional_instruction="Use institutional or reputable sources among provided URLs to confirm that the university is public."
    )

    # Land-grant status
    evaluator.add_custom_node(
        result=has_text(uni.land_grant_url),
        id=f"U{idx+1}_Land_Grant_URL",
        desc="Reference URL from USDA NIFA or institutional source confirming land-grant designation",
        parent=inst_node,
        critical=True
    )
    land_leaf = evaluator.add_leaf(
        id=f"U{idx+1}_Land_Grant_Status",
        desc="University is designated as a land-grant institution",
        parent=inst_node,
        critical=True
    )
    land_claim = f"The university {uni.full_name or 'UNKNOWN'} is designated as a land-grant institution under the Morrill Act."
    await evaluator.verify(
        claim=land_claim,
        node=land_leaf,
        sources=uni.land_grant_url,
        additional_instruction="Confirm land-grant designation via USDA NIFA or an official institutional page."
    )

    # --------------------- Athletic Programs --------------------------------
    athletic_node = evaluator.add_parallel(
        id=f"U{idx+1}_Athletic_Programs",
        desc="University meets athletic program requirements",
        parent=uni_node,
        critical=True
    )

    # Football (FBS Big Ten)
    football_leaf = evaluator.add_leaf(
        id=f"U{idx+1}_Football_Program",
        desc="University fields an NCAA Division I FBS football team in the Big Ten Conference",
        parent=athletic_node,
        critical=True
    )
    football_sources = uni.football_url if has_text(uni.football_url) else uni.big_ten_member_url
    football_claim = f"The university {uni.full_name or 'UNKNOWN'} fields an NCAA Division I FBS football team that competes in the Big Ten Conference."
    await evaluator.verify(
        claim=football_claim,
        node=football_leaf,
        sources=football_sources,
        additional_instruction="Confirm the university's football team competes in the Big Ten Conference (FBS). The Big Ten official page or the school's athletics page should support this."
    )

    # NCAA Tournament appearances (2015-2025)
    evaluator.add_custom_node(
        result=has_text(uni.ncaa_tournament_url),
        id=f"U{idx+1}_Basketball_URL",
        desc="Reference URL documenting NCAA Tournament appearance(s) within the specified timeframe",
        parent=athletic_node,
        critical=True
    )
    bball_leaf = evaluator.add_leaf(
        id=f"U{idx+1}_Basketball_Tournament",
        desc="University participated in NCAA Division I Men's Basketball Tournament between 2015-2025",
        parent=athletic_node,
        critical=True
    )
    bball_claim = f"The university {uni.full_name or 'UNKNOWN'} participated in the NCAA Division I Men's Basketball Tournament at least once between 2015 and 2025."
    await evaluator.verify(
        claim=bball_claim,
        node=bball_leaf,
        sources=uni.ncaa_tournament_url,
        additional_instruction="Verify from the provided source that the university has at least one NCAA Men's Basketball Tournament appearance in the specified window (inclusive)."
    )

    # Athletic revenue > $150M
    evaluator.add_custom_node(
        result=has_text(uni.athletic_revenue_url),
        id=f"U{idx+1}_Athletic_Revenue_URL",
        desc="Reference URL from USA Today, Knight Commission, or institutional source documenting athletic revenue",
        parent=athletic_node,
        critical=True
    )
    revenue_leaf = evaluator.add_leaf(
        id=f"U{idx+1}_Athletic_Revenue",
        desc="Athletic department has total annual revenues exceeding $150 million",
        parent=athletic_node,
        critical=True
    )
    revenue_claim = (
        f"The athletic department of {uni.full_name or 'UNKNOWN'} reported total annual revenues of "
        f"{uni.athletic_revenue_value or 'UNKNOWN'} in fiscal year {uni.athletic_revenue_fiscal_year or 'UNKNOWN'}, "
        f"which exceeds $150 million."
    )
    await evaluator.verify(
        claim=revenue_claim,
        node=revenue_leaf,
        sources=uni.athletic_revenue_url,
        additional_instruction="Confirm the total athletic department revenues exceed $150 million for the given fiscal year."
    )

    # --------------------- Leadership ---------------------------------------
    leader_node = evaluator.add_parallel(
        id=f"U{idx+1}_Leadership",
        desc="University president/chancellor meets appointment timing requirement",
        parent=uni_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_text(uni.president_appointment_url),
        id=f"U{idx+1}_President_URL",
        desc="Reference URL from institutional source documenting president/chancellor appointment date",
        parent=leader_node,
        critical=True
    )
    leader_leaf = evaluator.add_leaf(
        id=f"U{idx+1}_President_Appointment",
        desc="Current president or chancellor was appointed after January 1, 2020",
        parent=leader_node,
        critical=True
    )
    leader_claim = (
        f"The current {uni.president_or_chancellor or 'UNKNOWN'} of {uni.full_name or 'UNKNOWN'} "
        f"was appointed on {uni.president_appointment_date or 'UNKNOWN'}, which is after January 1, 2020."
    )
    await evaluator.verify(
        claim=leader_claim,
        node=leader_leaf,
        sources=uni.president_appointment_url,
        additional_instruction=f"Confirm appointment date is strictly later than {APPOINTMENT_CUTOFF_DATE}. Titles may vary (President vs. Chancellor)."
    )

    # --------------------- Academic Characteristics -------------------------
    academic_node = evaluator.add_parallel(
        id=f"U{idx+1}_Academic_Characteristics",
        desc="University meets academic scale and breadth requirements",
        parent=uni_node,
        critical=True
    )

    # Enrollment > 40,000
    evaluator.add_custom_node(
        result=has_text(uni.enrollment_url),
        id=f"U{idx+1}_Enrollment_URL",
        desc="Reference URL from institutional source or NCES documenting enrollment figures",
        parent=academic_node,
        critical=True
    )
    enroll_leaf = evaluator.add_leaf(
        id=f"U{idx+1}_Enrollment",
        desc="University has total fall enrollment exceeding 40,000 students",
        parent=academic_node,
        critical=True
    )
    enrollment_claim = (
        f"The university {uni.full_name or 'UNKNOWN'} has total fall enrollment of "
        f"{uni.enrollment_value or 'UNKNOWN'} in academic year {uni.enrollment_year or 'UNKNOWN'}, "
        f"exceeding 40,000 students."
    )
    await evaluator.verify(
        claim=enrollment_claim,
        node=enroll_leaf,
        sources=uni.enrollment_url,
        additional_instruction="Confirm the total enrollment figure from the provided page and determine whether it exceeds 40,000 students."
    )

    # Academic breadth (>=10 colleges; includes Engineering, Business, Education)
    evaluator.add_custom_node(
        result=has_text(uni.colleges_listing_url),
        id=f"U{idx+1}_Academic_Breadth_URL",
        desc="Reference URL from institutional source listing colleges and schools",
        parent=academic_node,
        critical=True
    )
    breadth_leaf = evaluator.add_leaf(
        id=f"U{idx+1}_Academic_Breadth",
        desc="University offers degree programs in at least 10 colleges/schools including Engineering, Business, and Education",
        parent=academic_node,
        critical=True
    )
    breadth_claim = (
        f"The university {uni.full_name or 'UNKNOWN'} offers degree programs through at least 10 different colleges or schools, "
        f"including a College of Engineering, a College or School of Business, and a College or School of Education."
    )
    await evaluator.verify(
        claim=breadth_claim,
        node=breadth_leaf,
        sources=uni.colleges_listing_url,
        additional_instruction="Verify that the colleges/schools list includes Engineering, Business, and Education, and count indicates at least 10 distinct colleges/schools."
    )

    # --------------------- Geographic Location ------------------------------
    geo_leaf = evaluator.add_leaf(
        id=f"U{idx+1}_Geographic_Location",
        desc="University is located in a state bordering at least one Great Lake",
        parent=uni_node,
        critical=True
    )
    state_text = (uni.state or "UNKNOWN")
    geo_claim = f"The state {state_text} borders at least one of the Great Lakes."
    await evaluator.verify(
        claim=geo_claim,
        node=geo_leaf,
        sources=gather_all_urls(uni),
        additional_instruction="The valid U.S. states bordering the Great Lakes are Minnesota, Wisconsin, Illinois, Indiana, Michigan, Ohio, Pennsylvania, and New York. Confirm the university's state is one of these."
    )


# --------------------------------------------------------------------------- #
# University identification gating                                            #
# --------------------------------------------------------------------------- #
def add_identification_gate(evaluator: Evaluator, root: Any, universities: List[UniversityInfo]) -> Any:
    """
    Add the University_Identification node with checks ensuring at least 3 universities
    have names and Big Ten membership URLs.
    """
    ident_node = evaluator.add_parallel(
        id="University_Identification",
        desc="Correctly identify universities as current Big Ten Conference members eligible for evaluation",
        parent=root,
        critical=True
    )

    # Add three critical existence checks (name + membership URL)
    for i in range(3):
        uni = universities[i] if i < len(universities) else UniversityInfo()
        result = has_text(uni.full_name) and has_text(uni.big_ten_member_url)
        evaluator.add_custom_node(
            result=result,
            id=f"U{i+1}_ID_Min",
            desc=f"University #{i+1} identification has name and Big Ten membership URL",
            parent=ident_node,
            critical=True
        )

    return ident_node


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
    Evaluate an answer for the Big Ten universities criteria task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Sequential: identification gates criteria verification
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

    # Extract structured data
    extraction = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Normalize to first three universities (pad if fewer)
    universities: List[UniversityInfo] = list(extraction.universities[:3])
    while len(universities) < 3:
        universities.append(UniversityInfo())

    # Add identification gate (critical)
    add_identification_gate(evaluator, root, universities)

    # Criteria verification (non-critical to allow partial credit across universities)
    criteria_node = evaluator.add_parallel(
        id="Criteria_Verification",
        desc="Verify that each identified university satisfies all specified constraints",
        parent=root,
        critical=False
    )

    # Verify each university subtree
    for idx, uni in enumerate(universities):
        await verify_university(evaluator, criteria_node, uni, idx)

    # Return summary
    return evaluator.get_summary()