import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "education_leader_identification_2026_transition"
TASK_DESCRIPTION = """Identify the education leader who satisfies all of the following conditions:

1. In 2026, this person is transitioning from a K-12 school district superintendent position to become a university president, with the university presidency beginning on July 1, 2026, and their superintendent resignation effective June 30, 2026.

2. This person worked at their K-12 school district for exactly 21 years as a public school educator, all at the same district, following this specific career progression:
   - Started as an hourly paraprofessional (2 years)
   - Served as a special education teacher (5 years)
   - Served as Director of Special Education (6 years, from 2011 to 2017)
   - Served as superintendent (exactly 9 years, from 2017 to 2026)

3. This person was also a K-12 student at the same school district for 13 years before their professional career there.

4. This person earned all of their higher education degrees from the destination university where they are becoming president:
   - Bachelor's degree in Political and Legal Studies (completed 2004)
   - Master of Education (completed 2012)
   - Educational Specialist degree (completed 2016)
   - Doctorate (completed by 2023)

5. This person served as an adjunct instructor at the destination university for exactly 7 years while working as a K-12 administrator.

6. This person delivered the commencement address at the destination university's Graduate School graduation ceremony in 2019.

7. The K-12 school district where this person served has the following characteristics:
   - Serves approximately 666 students across exactly 2 schools
   - Located in Callaway County, Missouri, specifically in southern Callaway County
   - Located approximately 11 miles from the destination university

8. Under this person's leadership, the district received state and national recognition for innovative thinking and smart budgeting practices.

9. This person received the Missouri Association of Rural Education's Outstanding Rural Administrator award in 2025.

10. The destination university where this person is becoming president has the following characteristics:
    - Founded in 1870 as the Female Orphan School by the Christian Church of Missouri
    - Originally located in Camden Point, Missouri
    - Moved to Fulton, Missouri in 1890
    - Became coeducational in 1996
    - Currently serves more than 3,500 total students, including approximately 1,100 traditional undergraduates
    - Accredited by the Higher Learning Commission
    - Maintains affiliation with the Christian Church (Disciples of Christ)

11. This person is becoming the 14th president of the destination university.

12. The current university president, Romaine Seguin, is transitioning back to serve as chair of the university's Board of Trustees when the new leader assumes the presidency on July 1, 2026.

13. The transition was publicly announced on March 4, 2026, in a joint announcement by the school district and the university.

What is the full name of this education leader?
"""


# ----------------------------- Data Models --------------------------------- #
class EducationLeaderExtraction(BaseModel):
    # Core identity
    leader_full_name: Optional[str] = None
    destination_university: Optional[str] = None
    k12_district: Optional[str] = None

    # Role transition timing
    transition_year: Optional[str] = None
    superintendent_resignation_date: Optional[str] = None
    presidency_start_date: Optional[str] = None
    role_transition_sources: List[str] = Field(default_factory=list)

    # Career at same K12 district
    educator_years_same_district: Optional[str] = None
    paraprofessional_years: Optional[str] = None
    special_ed_teacher_years: Optional[str] = None
    director_special_ed_years: Optional[str] = None
    director_special_ed_years_range: Optional[str] = None
    superintendent_years: Optional[str] = None
    superintendent_years_range: Optional[str] = None
    career_sources: List[str] = Field(default_factory=list)

    # Student at same district
    k12_student_years: Optional[str] = None
    student_sources: List[str] = Field(default_factory=list)

    # Degrees from destination university
    bachelors_field: Optional[str] = None
    bachelors_year: Optional[str] = None
    masters_name: Optional[str] = None
    masters_year: Optional[str] = None
    eds_name: Optional[str] = None
    eds_year: Optional[str] = None
    doctorate_name: Optional[str] = None
    doctorate_year_or_by: Optional[str] = None
    degrees_sources: List[str] = Field(default_factory=list)

    # Adjunct and commencement
    adjunct_years: Optional[str] = None
    adjunct_sources: List[str] = Field(default_factory=list)
    commencement_sources: List[str] = Field(default_factory=list)

    # District characteristics
    district_students_approx: Optional[str] = None
    district_schools_exact: Optional[str] = None
    district_location_county: Optional[str] = None
    district_location_region: Optional[str] = None
    district_distance_miles_to_university: Optional[str] = None
    district_characteristics_sources: List[str] = Field(default_factory=list)

    # District recognition and awards
    recognition_sources: List[str] = Field(default_factory=list)
    award_sources: List[str] = Field(default_factory=list)

    # University characteristics
    university_char_sources: List[str] = Field(default_factory=list)

    # Presidency ordinal and predecessor transition
    presidency_ordinal: Optional[str] = None
    predecessor_transition_text: Optional[str] = None
    succession_sources: List[str] = Field(default_factory=list)

    # Announcement details
    announcement_date: Optional[str] = None
    announcement_joint: Optional[str] = None
    announcement_sources: List[str] = Field(default_factory=list)

    # Fallback general sources
    general_sources: List[str] = Field(default_factory=list)


# ----------------------------- Extraction Prompt --------------------------- #
def prompt_extract_leader() -> str:
    return """
Extract the following fields from the answer. Return null for any missing field. For all URL fields, extract every URL explicitly present in the answer (including those in markdown links).

Core identity:
- leader_full_name
- destination_university
- k12_district

Role transition timing:
- transition_year (e.g., "2026")
- superintendent_resignation_date (e.g., "June 30, 2026")
- presidency_start_date (e.g., "July 1, 2026")
- role_transition_sources (list of URLs)

Career at same K12 district:
- educator_years_same_district (e.g., "21")
- paraprofessional_years (e.g., "2")
- special_ed_teacher_years (e.g., "5")
- director_special_ed_years (e.g., "6")
- director_special_ed_years_range (e.g., "2011-2017")
- superintendent_years (e.g., "9")
- superintendent_years_range (e.g., "2017-2026")
- career_sources (list of URLs)

Student at same district:
- k12_student_years (e.g., "13")
- student_sources (list of URLs)

Degrees from destination university:
- bachelors_field (e.g., "Political and Legal Studies")
- bachelors_year (e.g., "2004")
- masters_name (e.g., "Master of Education")
- masters_year (e.g., "2012")
- eds_name (e.g., "Educational Specialist")
- eds_year (e.g., "2016")
- doctorate_name (e.g., "Doctorate", "Ed.D.", "Ph.D.", etc.)
- doctorate_year_or_by (e.g., "2023", "by 2023")
- degrees_sources (list of URLs)

Adjunct and commencement:
- adjunct_years (e.g., "7")
- adjunct_sources (list of URLs)
- commencement_sources (list of URLs)

District characteristics:
- district_students_approx (e.g., "666")
- district_schools_exact (e.g., "2")
- district_location_county (e.g., "Callaway County, Missouri")
- district_location_region (e.g., "southern Callaway County")
- district_distance_miles_to_university (e.g., "11")
- district_characteristics_sources (list of URLs)

District recognition and awards:
- recognition_sources (list of URLs)
- award_sources (list of URLs)

University characteristics:
- university_char_sources (list of URLs)

Presidency ordinal and predecessor transition:
- presidency_ordinal (e.g., "14th")
- predecessor_transition_text (free text if present)
- succession_sources (list of URLs)

Announcement details:
- announcement_date (e.g., "March 4, 2026")
- announcement_joint (free text indicating joint announcement if present)
- announcement_sources (list of URLs)

Fallback:
- general_sources (list of URLs)
"""


# ----------------------------- Helpers ------------------------------------- #
def norm(s: Optional[str]) -> str:
    return s.strip() if isinstance(s, str) else ""


def has_two_words(name: Optional[str]) -> bool:
    if not name:
        return False
    parts = [p for p in name.strip().split() if p]
    return len(parts) >= 2


def uniq(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def src_union(*lists: List[str]) -> List[str]:
    all_urls: List[str] = []
    for lst in lists:
        if lst:
            all_urls.extend(lst)
    return uniq(all_urls)


async def add_and_verify_leaf(
    evaluator: Evaluator,
    *,
    node_id: str,
    desc: str,
    parent,
    claim: str,
    sources: List[str],
    add_ins: Optional[str] = None,
    critical: bool = True,
):
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources if sources else None,
        additional_instruction=add_ins or "None",
    )


# ----------------------------- Verification Tree --------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: EducationLeaderExtraction):
    # Create a critical main node under the evaluator's root
    edu_root = evaluator.add_parallel(
        id="Education_Leader_Identification",
        desc="Identify the education leader (full name) who satisfies all stated transition, career, education, district, university, recognition, and announcement constraints.",
        parent=evaluator.root,
        critical=True,
    )

    # Convenience variables
    name = norm(extracted.leader_full_name) or "the leader"
    univ = norm(extracted.destination_university) or "the destination university"
    district = norm(extracted.k12_district) or "the K-12 school district"

    # Record key extracted info as custom info
    evaluator.add_custom_info(
        info={"leader_full_name": norm(extracted.leader_full_name),
              "destination_university": norm(extracted.destination_university),
              "k12_district": norm(extracted.k12_district)},
        info_type="extracted_summary",
        info_name="extracted_core_identity"
    )

    # 1) Answer provides full name
    evaluator.add_custom_node(
        result=has_two_words(extracted.leader_full_name),
        id="Answer_Provides_Full_Name",
        desc="Answer provides the leader's full name (not just a title or partial name).",
        parent=edu_root,
        critical=True,
    )

    # 2) Role Transition Timing
    role_node = evaluator.add_parallel(
        id="Role_Transition_Timing",
        desc="Leader transitions from K-12 superintendent to university president with the specified effective dates.",
        parent=edu_root,
        critical=True,
    )
    role_sources = src_union(
        extracted.role_transition_sources,
        extracted.announcement_sources,
        extracted.succession_sources,
        extracted.general_sources,
    )

    await add_and_verify_leaf(
        evaluator,
        node_id="Transition_Is_Superintendent_To_University_President_In_2026",
        desc="Leader is transitioning from a K-12 superintendent role to a university president role in 2026.",
        parent=role_node,
        claim=f"{name}, superintendent of {district}, is transitioning to become president of {univ} in 2026.",
        sources=role_sources,
        add_ins="Verify the pages clearly indicate that the person is currently (or was) a K-12 district superintendent and is becoming a university president in 2026.",
    )
    await add_and_verify_leaf(
        evaluator,
        node_id="Superintendent_Resignation_Effective_June_30_2026",
        desc="Leader's superintendent resignation is effective June 30, 2026.",
        parent=role_node,
        claim=f"{name}'s resignation as superintendent of {district} is effective June 30, 2026.",
        sources=role_sources,
        add_ins="Look for explicit effective resignation date as superintendent: June 30, 2026.",
    )
    await add_and_verify_leaf(
        evaluator,
        node_id="University_Presidency_Begins_July_1_2026",
        desc="Leader's university presidency begins on July 1, 2026.",
        parent=role_node,
        claim=f"{name} will begin serving as president of {univ} on July 1, 2026.",
        sources=role_sources,
        add_ins="Look for explicit presidency start date: July 1, 2026.",
    )

    # 3) Career at same K12 district
    career_node = evaluator.add_parallel(
        id="Career_At_Same_K12_District",
        desc="Leader's K-12 career duration and progression at the same district match the stated constraints.",
        parent=edu_root,
        critical=True,
    )
    career_sources = src_union(extracted.career_sources, extracted.general_sources, extracted.announcement_sources)

    await add_and_verify_leaf(
        evaluator,
        node_id="K12_Educator_At_Same_District_Exactly_21_Years",
        desc="Leader served at the K-12 school district for exactly 21 years as a public school educator (all at the same district).",
        parent=career_node,
        claim=f"{name} served as a public school educator for exactly 21 years, all at {district}.",
        sources=career_sources,
        add_ins="The source should indicate 21 total years of service as an educator at the same district.",
    )
    await add_and_verify_leaf(
        evaluator,
        node_id="Paraprofessional_2_Years",
        desc="Leader started as an hourly paraprofessional for 2 years at the district.",
        parent=career_node,
        claim=f"{name} started as an hourly paraprofessional for 2 years at {district}.",
        sources=career_sources,
    )
    await add_and_verify_leaf(
        evaluator,
        node_id="Special_Education_Teacher_5_Years",
        desc="Leader served as a special education teacher for 5 years at the district.",
        parent=career_node,
        claim=f"{name} served as a special education teacher for 5 years at {district}.",
        sources=career_sources,
    )
    await add_and_verify_leaf(
        evaluator,
        node_id="Director_Special_Ed_6_Years_2011_2017",
        desc="Leader served as Director of Special Education for 6 years, from 2011 to 2017, at the district.",
        parent=career_node,
        claim=f"{name} served as Director of Special Education for 6 years (2011–2017) at {district}.",
        sources=career_sources,
        add_ins="Confirm both duration (6 years) and range (2011–2017).",
    )
    await add_and_verify_leaf(
        evaluator,
        node_id="Superintendent_9_Years_2017_2026",
        desc="Leader served as superintendent for exactly 9 years, from 2017 to 2026, at the district.",
        parent=career_node,
        claim=f"{name} served as superintendent at {district} for exactly 9 years (2017–2026).",
        sources=career_sources,
        add_ins="Confirm both duration (9 years) and range (2017–2026).",
    )

    # 4) K-12 student at same district for 13 years
    await add_and_verify_leaf(
        evaluator,
        node_id="K12_Student_At_Same_District_13_Years",
        desc="Leader attended the same K-12 district as a student for 13 years before working there.",
        parent=edu_root,
        claim=f"Before working there, {name} was a K-12 student in {district} for 13 years.",
        sources=src_union(extracted.student_sources, extracted.general_sources),
    )

    # 5) Degrees from destination university
    degrees_node = evaluator.add_parallel(
        id="Higher_Education_Degrees_From_Destination_University",
        desc="Leader earned the specified higher-education degrees from the destination university with the specified fields and completion years.",
        parent=edu_root,
        critical=True,
    )
    degree_sources = src_union(extracted.degrees_sources, extracted.announcement_sources, extracted.general_sources)

    await add_and_verify_leaf(
        evaluator,
        node_id="Bachelors_Political_And_Legal_Studies_Completed_2004",
        desc="Leader earned a Bachelor's degree in Political and Legal Studies, completed in 2004, from the destination university.",
        parent=degrees_node,
        claim=f"{name} earned a Bachelor's degree in Political and Legal Studies from {univ} in 2004.",
        sources=degree_sources,
    )
    await add_and_verify_leaf(
        evaluator,
        node_id="Master_Of_Education_Completed_2012",
        desc="Leader earned a Master of Education, completed in 2012, from the destination university.",
        parent=degrees_node,
        claim=f"{name} earned a Master of Education from {univ} in 2012.",
        sources=degree_sources,
    )
    await add_and_verify_leaf(
        evaluator,
        node_id="Educational_Specialist_Completed_2016",
        desc="Leader earned an Educational Specialist degree, completed in 2016, from the destination university.",
        parent=degrees_node,
        claim=f"{name} earned an Educational Specialist degree from {univ} in 2016.",
        sources=degree_sources,
    )
    await add_and_verify_leaf(
        evaluator,
        node_id="Doctorate_Completed_By_2023",
        desc="Leader earned a doctorate completed by 2023 from the destination university.",
        parent=degrees_node,
        claim=f"{name} completed a doctorate from {univ} by 2023.",
        sources=degree_sources,
        add_ins="It is acceptable if the source says 'by 2023' or 'in 2023' for completion timing.",
    )

    # 6) Adjunct instructor exactly 7 years
    await add_and_verify_leaf(
        evaluator,
        node_id="Adjunct_Instructor_Exactly_7_Years",
        desc="Leader served as an adjunct instructor at the destination university for exactly 7 years while working as a K-12 administrator.",
        parent=edu_root,
        claim=f"{name} served as an adjunct instructor at {univ} for exactly 7 years while working as a K-12 administrator.",
        sources=src_union(extracted.adjunct_sources, extracted.general_sources),
    )

    # 7) Graduate School commencement address in 2019
    await add_and_verify_leaf(
        evaluator,
        node_id="Graduate_School_Commencement_Address_2019",
        desc="Leader delivered the commencement address at the destination university's Graduate School graduation ceremony in 2019.",
        parent=edu_root,
        claim=f"In 2019, {name} delivered the commencement address at {univ}'s Graduate School graduation ceremony.",
        sources=src_union(extracted.commencement_sources, extracted.general_sources),
    )

    # 8) K-12 district characteristics
    district_node = evaluator.add_parallel(
        id="K12_District_Characteristics",
        desc="The K-12 district matches the stated characteristics.",
        parent=edu_root,
        critical=True,
    )
    district_sources = src_union(extracted.district_characteristics_sources, extracted.general_sources)

    await add_and_verify_leaf(
        evaluator,
        node_id="District_Serves_Approx_666_Students_Across_2_Schools",
        desc="District serves approximately 666 students across exactly 2 schools.",
        parent=district_node,
        claim=f"{district} serves approximately 666 students across exactly 2 schools.",
        sources=district_sources,
    )
    await add_and_verify_leaf(
        evaluator,
        node_id="District_Located_In_Southern_Callaway_County_Missouri",
        desc="District is located in southern Callaway County, Missouri.",
        parent=district_node,
        claim=f"{district} is located in southern Callaway County, Missouri.",
        sources=district_sources,
    )
    await add_and_verify_leaf(
        evaluator,
        node_id="District_Approximately_11_Miles_From_University",
        desc="District is located approximately 11 miles from the destination university.",
        parent=district_node,
        claim=f"{district} is located approximately 11 miles from {univ}.",
        sources=district_sources,
        add_ins="The distance can be stated approximately; variations like 'about 11 miles' are acceptable.",
    )

    # 9) District recognition under leader
    await add_and_verify_leaf(
        evaluator,
        node_id="District_Recognition_For_Innovative_Thinking_And_Smart_Budgeting",
        desc="Under the leader's leadership, the district received state and national recognition for innovative thinking and smart budgeting practices.",
        parent=edu_root,
        claim=f"Under {name}'s leadership, {district} received state and national recognition for innovative thinking and smart budgeting practices.",
        sources=src_union(extracted.recognition_sources, extracted.general_sources),
    )

    # 10) Award in 2025
    await add_and_verify_leaf(
        evaluator,
        node_id="Leader_Received_MARE_Outstanding_Rural_Administrator_2025",
        desc="Leader received the Missouri Association of Rural Education's Outstanding Rural Administrator award in 2025.",
        parent=edu_root,
        claim=f"In 2025, {name} received the Missouri Association of Rural Education's Outstanding Rural Administrator award.",
        sources=src_union(extracted.award_sources, extracted.general_sources),
    )

    # 11) Destination university characteristics
    univ_node = evaluator.add_parallel(
        id="Destination_University_Characteristics",
        desc="Destination university matches the stated historical/organizational characteristics.",
        parent=edu_root,
        critical=True,
    )
    univ_sources = src_union(extracted.university_char_sources, extracted.general_sources)

    await add_and_verify_leaf(
        evaluator,
        node_id="Founded_1870_Female_Orphan_School_By_Christian_Church_Of_Missouri",
        desc="University was founded in 1870 as the Female Orphan School by the Christian Church of Missouri.",
        parent=univ_node,
        claim=f"{univ} was founded in 1870 as the Female Orphan School by the Christian Church of Missouri.",
        sources=univ_sources,
    )
    await add_and_verify_leaf(
        evaluator,
        node_id="Originally_In_Camden_Point_Missouri",
        desc="University was originally located in Camden Point, Missouri.",
        parent=univ_node,
        claim=f"{univ} was originally located in Camden Point, Missouri.",
        sources=univ_sources,
    )
    await add_and_verify_leaf(
        evaluator,
        node_id="Moved_To_Fulton_In_1890",
        desc="University moved to Fulton, Missouri in 1890.",
        parent=univ_node,
        claim=f"{univ} moved to Fulton, Missouri in 1890.",
        sources=univ_sources,
    )
    await add_and_verify_leaf(
        evaluator,
        node_id="Became_Coeducational_1996",
        desc="University became coeducational in 1996.",
        parent=univ_node,
        claim=f"{univ} became coeducational in 1996.",
        sources=univ_sources,
    )
    await add_and_verify_leaf(
        evaluator,
        node_id="Enrollment_More_Than_3500_Total_Including_Approx_1100_Traditional_Undergrads",
        desc="University serves more than 3,500 total students, including approximately 1,100 traditional undergraduates.",
        parent=univ_node,
        claim=f"{univ} serves more than 3,500 total students, including approximately 1,100 traditional undergraduates.",
        sources=univ_sources,
        add_ins="Allow statements like 'more than 3,500' and 'about 1,100' as approximate counts.",
    )
    await add_and_verify_leaf(
        evaluator,
        node_id="Accredited_By_Higher_Learning_Commission",
        desc="University is accredited by the Higher Learning Commission.",
        parent=univ_node,
        claim=f"{univ} is accredited by the Higher Learning Commission.",
        sources=univ_sources,
    )
    await add_and_verify_leaf(
        evaluator,
        node_id="Affiliated_With_Christian_Church_Disciples_Of_Christ",
        desc="University maintains affiliation with the Christian Church (Disciples of Christ).",
        parent=univ_node,
        claim=f"{univ} maintains affiliation with the Christian Church (Disciples of Christ).",
        sources=univ_sources,
    )

    # 12) Presidential succession and predecessor transition
    succ_node = evaluator.add_parallel(
        id="Presidential_Succession_And_Predecessor_Transition",
        desc="Checks presidency ordinal and predecessor transition details.",
        parent=edu_root,
        critical=True,
    )
    succ_sources = src_union(extracted.succession_sources, extracted.announcement_sources, extracted.general_sources)

    await add_and_verify_leaf(
        evaluator,
        node_id="Leader_Becoming_14th_President",
        desc="Leader is becoming the 14th president of the destination university.",
        parent=succ_node,
        claim=f"{name} is becoming the 14th president of {univ}.",
        sources=succ_sources,
    )
    await add_and_verify_leaf(
        evaluator,
        node_id="Romaine_Seguin_Transitions_To_Board_Chair",
        desc="Current president Romaine Seguin transitions back to serve as chair of the university's Board of Trustees when the new leader assumes the presidency on July 1, 2026.",
        parent=succ_node,
        claim=f"The current president, Romaine Seguin, will transition to serve as chair of {univ}'s Board of Trustees when {name} assumes the presidency on July 1, 2026.",
        sources=succ_sources,
    )

    # 13) Announcement details
    ann_node = evaluator.add_parallel(
        id="Announcement_Details",
        desc="Transition announcement details match the stated constraints.",
        parent=edu_root,
        critical=True,
    )
    ann_sources = src_union(extracted.announcement_sources, extracted.general_sources)

    await add_and_verify_leaf(
        evaluator,
        node_id="Announcement_Date_March_4_2026",
        desc="The transition was publicly announced on March 4, 2026.",
        parent=ann_node,
        claim="The transition was publicly announced on March 4, 2026.",
        sources=ann_sources,
        add_ins="Verify that the press release or news states the date March 4, 2026.",
    )
    await add_and_verify_leaf(
        evaluator,
        node_id="Announcement_Is_Joint_District_And_University",
        desc="The announcement was a joint announcement by the school district and the university.",
        parent=ann_node,
        claim=f"The announcement of {name}'s transition was a joint announcement by {district} and {univ}.",
        sources=ann_sources,
        add_ins="Look for explicit mention that both the district and the university jointly announced the transition.",
    )


# ----------------------------- Main Entrypoint ----------------------------- #
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
    evaluator.initialize(
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_leader(),
        template_class=EducationLeaderExtraction,
        extraction_name="education_leader_extraction",
    )

    await build_verification_tree(evaluator, extracted)

    return evaluator.get_summary()