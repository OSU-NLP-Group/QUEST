import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "power_four_universities_2025_26"
TASK_DESCRIPTION = (
    "Identify four different public universities that are current members of Power Four conferences (SEC, Big Ten, Big 12, or ACC) "
    "as of the 2025-26 academic year, each meeting specific enrollment and geographic criteria. "
    "For each, provide the university's name, its conference affiliation, its total enrollment figure, and reference URL(s) that verify "
    "the conference membership and enrollment data. For the fourth university, also provide information about a large school district "
    "in the same state (>150,000 students)."
)

POWER_FOUR = {"SEC", "Southeastern Conference", "Big Ten", "Big Ten Conference", "Big 12", "Big 12 Conference", "ACC", "Atlantic Coast Conference"}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SchoolDistrictInfo(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    enrollment: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class UniversityInfo(BaseModel):
    name: Optional[str] = None
    conference: Optional[str] = None
    total_enrollment: Optional[str] = None
    state: Optional[str] = None
    conference_sources: List[str] = Field(default_factory=list)
    enrollment_sources: List[str] = Field(default_factory=list)
    public_sources: List[str] = Field(default_factory=list)


class U4Info(UniversityInfo):
    district: Optional[SchoolDistrictInfo] = None


class UniversitiesExtraction(BaseModel):
    university_1: Optional[UniversityInfo] = None
    university_2: Optional[UniversityInfo] = None
    university_3: Optional[UniversityInfo] = None
    university_4: Optional[U4Info] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
Extract exactly four universities (University 1, 2, 3, 4) from the answer, mapping them to the following structures. 
Do NOT invent information. Only extract what is explicitly present in the answer. 
For URL fields, extract actual URLs as strings. If a field is missing, return null (for strings) or [] (for lists).

For each of University 1, 2, and 3, extract:
- name: University name as stated.
- conference: Conference name as stated (e.g., "SEC", "Big Ten", "Big 12", "ACC", or full names like "Southeastern Conference").
- total_enrollment: The total enrollment figure or phrase as stated (combined undergrad+grad, if provided).
- state: The U.S. state where the university is located, if stated in the answer; else null.
- conference_sources: URLs cited that specifically support its conference membership.
- enrollment_sources: URLs cited that specifically support the total enrollment.
- public_sources: URLs cited that indicate the institution is public (e.g., official pages or third-party references). If the answer uses the same URLs to imply public status, include them here too.

For University 4, also extract a large public school district in the same state:
- name, conference, total_enrollment, state, conference_sources, enrollment_sources, public_sources (same as above)
- district: 
  - name: District name (e.g., "Los Angeles Unified School District").
  - state: The state of the district, if stated.
  - enrollment: The district's enrollment figure or phrase as stated.
  - sources: URLs cited that support the district's enrollment.

Return a JSON object with fields: university_1, university_2, university_3, university_4.
Each of these is an object with the fields specified above (university_4 uses the district sub-object).
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def first_non_empty_url_list(*lists: List[str]) -> List[str]:
    for l in lists:
        if l and len(l) > 0:
            return l
    return []


def merge_urls(*lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for l in lists:
        for u in l:
            if isinstance(u, str):
                uu = u.strip()
                if uu and uu not in seen:
                    seen.add(uu)
                    out.append(uu)
    return out


def parse_int_from_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    digits = re.findall(r"\d[\d,\.]*", text)
    if not digits:
        return None
    # Take the first number-like token, strip commas and decimals
    token = digits[0].replace(",", "")
    try:
        # Handle forms like "150000.0" or "150000+" -> strip non-digits
        token_clean = re.sub(r"[^\d]", "", token)
        if token_clean == "":
            return None
        return int(token_clean)
    except Exception:
        return None


def in_power_four(conf: Optional[str]) -> bool:
    if not conf:
        return False
    c = conf.strip()
    if c in POWER_FOUR:
        return True
    # Normalize shorthand/full name matching
    c_lower = c.lower()
    return any(
        kw in c_lower
        for kw in ["sec", "southeastern conference", "big ten", "big 12", "big12", "acc", "atlantic coast conference"]
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_university_1(evaluator: Evaluator, parent_node, u: Optional[UniversityInfo]) -> None:
    uni_node = evaluator.add_parallel(
        id="university_1",
        desc="First university: A public university in the Southeastern Conference (SEC) with total enrollment between 40,000 and 50,000 students",
        parent=parent_node,
        critical=False
    )

    name = (u.name or "the university").strip() if u else "the university"
    # Conference membership (SEC)
    conf_leaf = evaluator.add_leaf(
        id="u1_conference",
        desc="The university is a current member of the Southeastern Conference (SEC) as of the 2025-26 academic year",
        parent=uni_node,
        critical=True
    )
    claim_conf = f"{name} is a current member of the Southeastern Conference (SEC) as of the 2025–26 academic year."
    await evaluator.verify(
        claim=claim_conf,
        node=conf_leaf,
        sources=u.conference_sources if u else [],
        additional_instruction="Verify that the page explicitly indicates membership in the SEC. If the page indicates 'Southeastern Conference', treat it as SEC."
    )

    # Enrollment range check (40k–50k)
    enroll_leaf = evaluator.add_leaf(
        id="u1_enrollment",
        desc="The university's total enrollment (undergraduate plus graduate) is between 40,000 and 50,000 students",
        parent=uni_node,
        critical=True
    )
    claim_enroll = f"{name}'s total enrollment (combined undergraduate and graduate) is between 40,000 and 50,000 students."
    await evaluator.verify(
        claim=claim_enroll,
        node=enroll_leaf,
        sources=u.enrollment_sources if u else [],
        additional_instruction="Check total enrollment as stated on the page. If an exact total is provided and falls within [40,000, 50,000], consider this supported. Allow minor rounding or recent academic year variations."
    )

    # Public institution status
    public_leaf = evaluator.add_leaf(
        id="u1_public",
        desc="The university is a public institution",
        parent=uni_node,
        critical=True
    )
    public_sources = merge_urls(*(u.public_sources if u else []), *(u.conference_sources if u else []), *(u.enrollment_sources if u else []))
    claim_public = f"{name} is a public university (i.e., a public institution)."
    await evaluator.verify(
        claim=claim_public,
        node=public_leaf,
        sources=public_sources,
        additional_instruction="Accept phrases like 'public university', 'public research university', or 'public institution'."
    )

    # References provided (presence check)
    ref_ok = bool(u and u.conference_sources and len(u.conference_sources) > 0 and u.enrollment_sources and len(u.enrollment_sources) > 0)
    evaluator.add_custom_node(
        result=ref_ok,
        id="u1_reference",
        desc="Valid reference URL(s) are provided that confirm the university's SEC membership and enrollment data",
        parent=uni_node,
        critical=True
    )


async def verify_university_2(evaluator: Evaluator, parent_node, u: Optional[UniversityInfo]) -> None:
    uni_node = evaluator.add_parallel(
        id="university_2",
        desc="Second university: A public university in the Big Ten Conference with total enrollment exceeding 50,000 students",
        parent=parent_node,
        critical=False
    )

    name = (u.name or "the university").strip() if u else "the university"
    # Big Ten membership
    conf_leaf = evaluator.add_leaf(
        id="u2_conference",
        desc="The university is a current member of the Big Ten Conference as of the 2025-26 academic year",
        parent=uni_node,
        critical=True
    )
    claim_conf = f"{name} is a current member of the Big Ten Conference as of the 2025–26 academic year."
    await evaluator.verify(
        claim=claim_conf,
        node=conf_leaf,
        sources=u.conference_sources if u else [],
        additional_instruction="Verify explicit membership in the Big Ten Conference. Abbreviations like 'Big Ten' are acceptable."
    )

    # Enrollment > 50,000
    enroll_leaf = evaluator.add_leaf(
        id="u2_enrollment",
        desc="The university's total enrollment (undergraduate plus graduate) exceeds 50,000 students",
        parent=uni_node,
        critical=True
    )
    claim_enroll = f"{name}'s total enrollment (combined undergraduate and graduate) exceeds 50,000 students."
    await evaluator.verify(
        claim=claim_enroll,
        node=enroll_leaf,
        sources=u.enrollment_sources if u else [],
        additional_instruction="Check the total enrollment on the page. If it is 50,001 or higher (or clearly 'over 50,000'), consider supported. Allow minor rounding or recent academic-year variations."
    )

    # Public institution status
    public_leaf = evaluator.add_leaf(
        id="u2_public",
        desc="The university is a public institution",
        parent=uni_node,
        critical=True
    )
    public_sources = merge_urls(*(u.public_sources if u else []), *(u.conference_sources if u else []), *(u.enrollment_sources if u else []))
    claim_public = f"{name} is a public university (i.e., a public institution)."
    await evaluator.verify(
        claim=claim_public,
        node=public_leaf,
        sources=public_sources,
        additional_instruction="Accept phrases like 'public university', 'public research university', or 'public institution'."
    )

    # References provided
    ref_ok = bool(u and u.conference_sources and len(u.conference_sources) > 0 and u.enrollment_sources and len(u.enrollment_sources) > 0)
    evaluator.add_custom_node(
        result=ref_ok,
        id="u2_reference",
        desc="Valid reference URL(s) are provided that confirm the university's Big Ten membership and enrollment data",
        parent=uni_node,
        critical=True
    )


async def verify_university_3(evaluator: Evaluator, parent_node, u: Optional[UniversityInfo]) -> None:
    uni_node = evaluator.add_parallel(
        id="university_3",
        desc="Third university: A public university in the Atlantic Coast Conference (ACC) with total enrollment between 20,000 and 35,000 students",
        parent=parent_node,
        critical=False
    )

    name = (u.name or "the university").strip() if u else "the university"
    # ACC membership
    conf_leaf = evaluator.add_leaf(
        id="u3_conference",
        desc="The university is a current member of the Atlantic Coast Conference (ACC) as of the 2025-26 academic year",
        parent=uni_node,
        critical=True
    )
    claim_conf = f"{name} is a current member of the Atlantic Coast Conference (ACC) as of the 2025–26 academic year."
    await evaluator.verify(
        claim=claim_conf,
        node=conf_leaf,
        sources=u.conference_sources if u else [],
        additional_instruction="Verify explicit membership in the ACC. Abbreviations like 'ACC' are acceptable."
    )

    # Enrollment between 20k–35k
    enroll_leaf = evaluator.add_leaf(
        id="u3_enrollment",
        desc="The university's total enrollment (undergraduate plus graduate) is between 20,000 and 35,000 students",
        parent=uni_node,
        critical=True
    )
    claim_enroll = f"{name}'s total enrollment (combined undergraduate and graduate) is between 20,000 and 35,000 students."
    await evaluator.verify(
        claim=claim_enroll,
        node=enroll_leaf,
        sources=u.enrollment_sources if u else [],
        additional_instruction="Check the total enrollment on the page. If the number lies within [20,000, 35,000], consider supported. Allow minor rounding or recent academic-year variations."
    )

    # Public institution status
    public_leaf = evaluator.add_leaf(
        id="u3_public",
        desc="The university is a public institution",
        parent=uni_node,
        critical=True
    )
    public_sources = merge_urls(*(u.public_sources if u else []), *(u.conference_sources if u else []), *(u.enrollment_sources if u else []))
    claim_public = f"{name} is a public university (i.e., a public institution)."
    await evaluator.verify(
        claim=claim_public,
        node=public_leaf,
        sources=public_sources,
        additional_instruction="Accept phrases like 'public university', 'public research university', or 'public institution'."
    )

    # References provided
    ref_ok = bool(u and u.conference_sources and len(u.conference_sources) > 0 and u.enrollment_sources and len(u.enrollment_sources) > 0)
    evaluator.add_custom_node(
        result=ref_ok,
        id="u3_reference",
        desc="Valid reference URL(s) are provided that confirm the university's ACC membership and enrollment data",
        parent=uni_node,
        critical=True
    )


async def verify_university_4(evaluator: Evaluator, parent_node, u: Optional[U4Info]) -> None:
    uni_node = evaluator.add_parallel(
        id="university_4",
        desc="Fourth university: A public university in any Power Four conference (SEC, Big Ten, Big 12, or ACC) located in a state that also has a school district with over 150,000 students, with information about that district provided",
        parent=parent_node,
        critical=False
    )

    name = (u.name or "the university").strip() if u else "the university"

    # Conference membership in Power Four (use provided conference string and verify)
    conf_leaf = evaluator.add_leaf(
        id="u4_conference",
        desc="The university is a current member of one of the Power Four conferences (SEC, Big Ten, Big 12, or ACC) as of the 2025-26 academic year",
        parent=uni_node,
        critical=True
    )
    if u and u.conference:
        claim_conf = f"{name} is a current member of the {u.conference} as of the 2025–26 academic year."
    else:
        claim_conf = f"{name} is a current member of the SEC, Big Ten, Big 12, or ACC as of the 2025–26 academic year."
    await evaluator.verify(
        claim=claim_conf,
        node=conf_leaf,
        sources=u.conference_sources if u else [],
        additional_instruction="Verify that the page shows the university is a member of the specified conference. Accept full or abbreviated conference names."
    )

    # Public institution status
    public_leaf = evaluator.add_leaf(
        id="u4_public",
        desc="The university is a public institution",
        parent=uni_node,
        critical=True
    )
    public_sources = merge_urls(*(u.public_sources if u else []), *(u.conference_sources if u else []), *(u.enrollment_sources if u else []))
    claim_public = f"{name} is a public university (i.e., a public institution)."
    await evaluator.verify(
        claim=claim_public,
        node=public_leaf,
        sources=public_sources,
        additional_instruction="Accept phrases like 'public university', 'public research university', or 'public institution'."
    )

    # State-district logical check: same state and district >= 150,000
    state_ok = False
    dist_enough = False
    if u and u.state and u.district and u.district.state:
        state_ok = u.state.strip().lower() == u.district.state.strip().lower()
    if u and u.district and u.district.enrollment:
        n = parse_int_from_text(u.district.enrollment)
        dist_enough = (n is not None and n >= 150000)

    evaluator.add_custom_node(
        result=bool(state_ok and dist_enough),
        id="u4_state_district",
        desc="The university is located in a state that has at least one public school district with over 150,000 enrolled students",
        parent=uni_node,
        critical=True
    )

    # District info presence (name and enrollment provided)
    district_info_ok = bool(u and u.district and u.district.name and u.district.enrollment)
    evaluator.add_custom_node(
        result=district_info_ok,
        id="u4_district_info",
        desc="Information about the large school district (name and enrollment) in the same state is provided",
        parent=uni_node,
        critical=True
    )

    # References provided for university membership (presence)
    ref_uni_ok = bool(u and u.conference_sources and len(u.conference_sources) > 0)
    evaluator.add_custom_node(
        result=ref_uni_ok,
        id="u4_reference_university",
        desc="Valid reference URL(s) are provided that confirm the university's conference membership",
        parent=uni_node,
        critical=True
    )

    # References provided for district enrollment (verify by URLs that district has >150,000 students)
    district_leaf = evaluator.add_leaf(
        id="u4_reference_district",
        desc="Valid reference URL(s) are provided that confirm the school district's enrollment data",
        parent=uni_node,
        critical=True
    )
    if u and u.district and u.district.name:
        claim_district = f"The public school district '{u.district.name}' has over 150,000 students."
    else:
        claim_district = "The specified public school district has over 150,000 students."
    await evaluator.verify(
        claim=claim_district,
        node=district_leaf,
        sources=(u.district.sources if (u and u.district) else []),
        additional_instruction="Confirm from the page that the district's enrollment exceeds 150,000. Accept clearly stated totals or phrases like 'over 150,000'."
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
    Evaluate an answer for the Power Four universities (2025–26) task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent verification for each university
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

    # Extract structured university information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Build and verify the four universities
    await verify_university_1(evaluator, root, extracted.university_1)
    await verify_university_2(evaluator, root, extracted.university_2)
    await verify_university_3(evaluator, root, extracted.university_3)
    await verify_university_4(evaluator, root, extracted.university_4)

    # Return the evaluation summary
    return evaluator.get_summary()