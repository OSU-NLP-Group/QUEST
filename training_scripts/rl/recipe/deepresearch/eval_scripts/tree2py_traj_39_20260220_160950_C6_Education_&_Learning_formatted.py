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
TASK_ID = "pa_superintendent_2024_act158"
TASK_DESCRIPTION = """
Identify the Pennsylvania public school district superintendent who was named the 2024 Pennsylvania Superintendent of the Year by the Pennsylvania Association of School Administrators (PASA). This superintendent must hold both a bachelor's degree and a master's degree (or higher), as required by Pennsylvania superintendent certification law (22 Pa. Code § 49.172). Additionally, the superintendent must have at least six years of satisfactory school experience, with at least three years served in an administrative or supervisory capacity. The superintendent's school district must implement Pennsylvania Act 158 graduation pathways, which became effective with the graduating class of 2023, and must specifically offer the Keystone Proficiency Pathway as a graduation option. Under this pathway, students must achieve a proficient score of 1500 or higher on each of the three required Keystone Exams: Algebra I, Biology, and Literature. Provide the superintendent's full name, the name of the school district, and URL references supporting each requirement.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class SuperintendentExtraction(BaseModel):
    superintendent_name: Optional[str] = None
    district_name: Optional[str] = None

    # Award / recognition sources
    award_announcement_urls: List[str] = Field(default_factory=list)

    # Tenure / service period (for 2023-2024)
    service_period_urls: List[str] = Field(default_factory=list)

    # District characteristics
    location_urls: List[str] = Field(default_factory=list)           # district in PA
    district_type_urls: List[str] = Field(default_factory=list)      # public school district

    # Educational qualifications
    degree_master_or_higher_urls: List[str] = Field(default_factory=list)
    degree_bachelor_urls: List[str] = Field(default_factory=list)

    # Professional experience
    experience_total_urls: List[str] = Field(default_factory=list)   # ≥ 6 years school experience
    experience_admin_urls: List[str] = Field(default_factory=list)   # ≥ 3 years admin/supervisory

    # Graduation requirements
    act158_urls: List[str] = Field(default_factory=list)                  # district implements Act 158
    keystone_proficiency_urls: List[str] = Field(default_factory=list)    # shows Keystone Proficiency score rule (1500+)
    keystone_exam_list_urls: List[str] = Field(default_factory=list)      # shows the three Keystone exams list


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_superintendent() -> str:
    return """
Extract the single superintendent and district the answer claims meet all requirements, along with URL sources that the answer cites for each requirement.

Return JSON with the following fields:
- superintendent_name: Full name of the superintendent as stated in the answer.
- district_name: Full official name of the superintendent’s school district as stated in the answer.

For each of the following, extract ALL URLs explicitly provided in the answer text that support the statement. If the same URL supports multiple items, include it in each relevant list. If none are provided for an item, return an empty list (do not invent URLs).

- award_announcement_urls: URLs that directly confirm the superintendent was named the 2024 Pennsylvania Superintendent of the Year by PASA (Pennsylvania Association of School Administrators), e.g., PASA, AASA, district, or reputable news release.
- service_period_urls: URLs that show the person was serving as superintendent during the 2023–2024 school year (or clearly “current” in 2023/2024) at the named district.
- location_urls: URLs that show the district is located in Pennsylvania (district site or PDE page acceptable).
- district_type_urls: URLs that show it is a public school district (not private/charter/cyber) — PDE “School District” pages or district “About” pages acceptable.
- degree_master_or_higher_urls: URLs that confirm the superintendent holds at least a master’s degree (M.Ed., MA/MS, Ed.D., Ph.D., etc.).
- degree_bachelor_urls: URLs that confirm the superintendent holds a bachelor’s degree.
- experience_total_urls: URLs that confirm the superintendent has at least six years of school experience in total.
- experience_admin_urls: URLs that confirm at least three years in an administrative or supervisory capacity.
- act158_urls: URLs that confirm the district implements Act 158 graduation pathways (district policy/handbook/counseling/graduation page preferred; must be district-specific).
- keystone_proficiency_urls: URLs that show/describe the Keystone Proficiency Pathway’s score threshold of 1500 or higher on each Keystone exam (can be PDE/state or district pages).
- keystone_exam_list_urls: URLs that list the three Keystone exams as Algebra I, Biology, and Literature (can be PDE/state or district pages).

Rules:
- Extract only URLs explicitly present in the answer (plain URLs or in markdown links).
- Do not infer or create any URLs.
- If a field isn’t mentioned, set it to null (for names) or [] for URL lists.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_urls(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    return any(isinstance(u, str) and len(u.strip()) > 0 for u in urls)


# --------------------------------------------------------------------------- #
# Build verification tree and run checks                                      #
# --------------------------------------------------------------------------- #
async def _build_and_verify(evaluator: Evaluator, data: SuperintendentExtraction):
    # Root aggregation under evaluator.root: a critical parallel node
    main = evaluator.add_parallel(
        id="Superintendent_Identification",
        desc="Identify a Pennsylvania school district superintendent who meets all specified professional, educational, and district criteria",
        parent=evaluator.root,
        critical=True
    )

    # Basic identity presence (critical)
    identity_present = evaluator.add_custom_node(
        result=bool(data.superintendent_name and data.superintendent_name.strip()) and
               bool(data.district_name and data.district_name.strip()),
        id="Identity_Provided",
        desc="Superintendent full name and district name are provided in the answer",
        parent=main,
        critical=True
    )

    # ---------------- Professional Recognition ---------------- #
    prof = evaluator.add_parallel(
        id="Professional_Recognition",
        desc="The superintendent must have received official recognition from PASA as the 2024 Pennsylvania Superintendent of the Year",
        parent=main,
        critical=True
    )

    # PASA 2024 Award block
    pasa_award = evaluator.add_parallel(
        id="PASA_2024_Award",
        desc="The superintendent was named the 2024 Pennsylvania Superintendent of the Year by PASA",
        parent=prof,
        critical=True
    )

    # Presence of award source URL(s) – critical under a critical parent (framework requirement)
    award_url_presence = evaluator.add_custom_node(
        result=_has_urls(data.award_announcement_urls),
        id="Award_Announcement_URL",
        desc="Provide a URL reference confirming the superintendent received the 2024 PA Superintendent of the Year award",
        parent=pasa_award,
        critical=True
    )

    # Verify Award Year = 2024 (critical)
    award_year_leaf = evaluator.add_leaf(
        id="Award_Year_2024",
        desc="The award recognition was for the 2024 academic year",
        parent=pasa_award,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.superintendent_name or ''} was recognized as the 2024 Pennsylvania Superintendent of the Year.",
        node=award_year_leaf,
        sources=data.award_announcement_urls,
        additional_instruction="Verify that the page explicitly indicates the '2024' Pennsylvania Superintendent of the Year associated with the named person. If the recognition announcement is dated late 2023 for the 2024 award, it should still count as the 2024 award."
    )

    # Verify Awarding Organization = PASA (critical)
    award_org_leaf = evaluator.add_leaf(
        id="Awarding_Organization_PASA",
        desc="The award was given by the Pennsylvania Association of School Administrators (PASA)",
        parent=pasa_award,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.superintendent_name or ''} was named Pennsylvania Superintendent of the Year by the Pennsylvania Association of School Administrators (PASA).",
        node=award_org_leaf,
        sources=data.award_announcement_urls,
        additional_instruction="Confirm that the awarding organization is PASA. The page may be hosted by PASA, AASA, the district, or reputable press, but it must attribute the honor to PASA."
    )

    # Active Service during 2023-2024
    active_service_block = evaluator.add_parallel(
        id="Active_Service_Period",
        desc="The superintendent was actively serving in the position during the 2023-2024 academic year when the award was given",
        parent=prof,
        critical=True
    )

    service_url_presence = evaluator.add_custom_node(
        result=_has_urls(data.service_period_urls),
        id="Service_Period_URL",
        desc="Provide a URL reference confirming the superintendent's tenure during the award period",
        parent=active_service_block,
        critical=True
    )

    service_leaf = evaluator.add_leaf(
        id="Active_Service_Verify",
        desc="Verification that the superintendent served during the 2023–2024 academic year",
        parent=active_service_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.superintendent_name or ''} served as superintendent of {data.district_name or ''} during the 2023–2024 academic year (or was the current superintendent in 2023/2024).",
        node=service_leaf,
        sources=data.service_period_urls,
        additional_instruction="Accept evidence that the person held the superintendent title during 2023 or 2024 (e.g., 'current superintendent' or tenure spanning these years). The page should clearly associate the person with the named district."
    )

    # ---------------- District Characteristics ---------------- #
    district_char = evaluator.add_parallel(
        id="District_Characteristics",
        desc="The school district must be a public school district located in Pennsylvania",
        parent=main,
        critical=True
    )

    # Located in Pennsylvania
    pa_loc_block = evaluator.add_parallel(
        id="Pennsylvania_Location",
        desc="The school district is located in Pennsylvania",
        parent=district_char,
        critical=True
    )

    location_url_presence = evaluator.add_custom_node(
        result=_has_urls(data.location_urls),
        id="Location_URL",
        desc="Provide a URL reference confirming the district is in Pennsylvania",
        parent=pa_loc_block,
        critical=True
    )

    pa_loc_leaf = evaluator.add_leaf(
        id="Pennsylvania_Location_Verify",
        desc="Verify district is located in Pennsylvania",
        parent=pa_loc_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.district_name or ''} is located in Pennsylvania.",
        node=pa_loc_leaf,
        sources=data.location_urls,
        additional_instruction="Accept district website pages or Pennsylvania Department of Education (PDE) pages that clearly list the district as a PA entity."
    )

    # Public school district (not private/charter/cyber)
    public_block = evaluator.add_parallel(
        id="Public_School_District",
        desc="The district is a public school district (not private, charter, or cyber)",
        parent=district_char,
        critical=True
    )

    dist_type_url_presence = evaluator.add_custom_node(
        result=_has_urls(data.district_type_urls),
        id="District_Type_URL",
        desc="Provide a URL reference confirming the district type",
        parent=public_block,
        critical=True
    )

    public_leaf = evaluator.add_leaf(
        id="Public_School_District_Verify",
        desc="Verify district is a public school district (not private/charter/cyber)",
        parent=public_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.district_name or ''} is a public school district (not private, charter, or cyber).",
        node=public_leaf,
        sources=data.district_type_urls,
        additional_instruction="PDE 'School District' pages or district 'About' pages identifying it as a public school district are acceptable. Purely charter/private/cyber pages should not satisfy this check."
    )

    # ---------------- Educational Qualifications ---------------- #
    edu = evaluator.add_parallel(
        id="Educational_Qualifications",
        desc="The superintendent must hold the educational qualifications required by Pennsylvania law for superintendent certification",
        parent=main,
        critical=True
    )

    masters_block = evaluator.add_parallel(
        id="Masters_Degree_Or_Higher",
        desc="The superintendent holds a master's degree or higher (required by 22 Pa. Code § 49.172)",
        parent=edu,
        critical=True
    )
    masters_url_presence = evaluator.add_custom_node(
        result=_has_urls(data.degree_master_or_higher_urls),
        id="Degree_Verification_URL",
        desc="Provide a URL reference confirming the superintendent's master's degree or doctoral degree",
        parent=masters_block,
        critical=True
    )
    masters_leaf = evaluator.add_leaf(
        id="Masters_Degree_Verify",
        desc="Verify that the superintendent holds a master's degree or higher",
        parent=masters_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.superintendent_name or ''} holds at least a master's degree (or a doctoral degree).",
        node=masters_leaf,
        sources=data.degree_master_or_higher_urls,
        additional_instruction="Accept degree abbreviations like M.Ed., M.A., M.S., Ed.D., Ph.D., etc., when clearly associated with the superintendent."
    )

    bachelors_block = evaluator.add_parallel(
        id="Bachelor_Degree",
        desc="The superintendent holds a bachelor's degree (base requirement for PA superintendent certification)",
        parent=edu,
        critical=True
    )
    bachelors_url_presence = evaluator.add_custom_node(
        result=_has_urls(data.degree_bachelor_urls),
        id="Bachelor_Degree_URL",
        desc="Provide a URL reference confirming the bachelor's degree",
        parent=bachelors_block,
        critical=True
    )
    bachelors_leaf = evaluator.add_leaf(
        id="Bachelor_Degree_Verify",
        desc="Verify that the superintendent holds a bachelor's degree",
        parent=bachelors_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.superintendent_name or ''} holds a bachelor's degree.",
        node=bachelors_leaf,
        sources=data.degree_bachelor_urls,
        additional_instruction="Accept any credible bio or official page that explicitly states the bachelor’s degree for the superintendent."
    )

    # ---------------- Professional Experience ---------------- #
    prof_exp = evaluator.add_parallel(
        id="Professional_Experience",
        desc="The superintendent must meet Pennsylvania's experience requirements for superintendent certification",
        parent=main,
        critical=True
    )

    six_years_block = evaluator.add_parallel(
        id="Six_Years_School_Experience",
        desc="The superintendent has at least six years of satisfactory school experience (PA requirement per 22 Pa. Code § 49.172)",
        parent=prof_exp,
        critical=True
    )
    total_exp_url_presence = evaluator.add_custom_node(
        result=_has_urls(data.experience_total_urls),
        id="Total_Experience_URL",
        desc="Provide a URL reference confirming at least six years of school experience",
        parent=six_years_block,
        critical=True
    )
    six_years_leaf = evaluator.add_leaf(
        id="Six_Years_Experience_Verify",
        desc="Verify that the superintendent has at least six years of school experience",
        parent=six_years_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.superintendent_name or ''} has at least six years of satisfactory school experience.",
        node=six_years_leaf,
        sources=data.experience_total_urls,
        additional_instruction="Use official bios, resumes, or news articles that summarize years of service in education."
    )

    three_years_block = evaluator.add_parallel(
        id="Three_Years_Administrative_Experience",
        desc="At least three of the six years must be in a supervisory or administrative capacity (PA requirement)",
        parent=prof_exp,
        critical=True
    )
    admin_exp_url_presence = evaluator.add_custom_node(
        result=_has_urls(data.experience_admin_urls),
        id="Administrative_Experience_URL",
        desc="Provide a URL reference confirming at least three years of administrative or supervisory experience",
        parent=three_years_block,
        critical=True
    )
    three_years_leaf = evaluator.add_leaf(
        id="Three_Years_Admin_Verify",
        desc="Verify at least three years in an administrative or supervisory capacity",
        parent=three_years_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.superintendent_name or ''} has at least three years of administrative or supervisory experience.",
        node=three_years_leaf,
        sources=data.experience_admin_urls,
        additional_instruction="Accept roles such as assistant principal, principal, supervisor, director, assistant superintendent, superintendent, etc., when the time span totals at least three years."
    )

    # ---------------- District Graduation Requirements ---------------- #
    grad_reqs = evaluator.add_parallel(
        id="District_Graduation_Requirements",
        desc="The school district must implement Pennsylvania's Act 158 graduation requirements with the Keystone Proficiency Pathway",
        parent=main,
        critical=True
    )

    act158_block = evaluator.add_parallel(
        id="Act_158_Compliance",
        desc="The district implements Pennsylvania Act 158 graduation pathways, which became effective with the Class of 2023",
        parent=grad_reqs,
        critical=True
    )
    act158_url_presence = evaluator.add_custom_node(
        result=_has_urls(data.act158_urls),
        id="Act_158_URL",
        desc="Provide a URL reference confirming the district implements Act 158 graduation pathways",
        parent=act158_block,
        critical=True
    )
    act158_leaf = evaluator.add_leaf(
        id="Act_158_Verify",
        desc="Verify the district implements Act 158 graduation pathways effective with the Class of 2023",
        parent=act158_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"{data.district_name or ''} implements Pennsylvania Act 158 graduation pathways (effective with the Class of 2023).",
        node=act158_leaf,
        sources=data.act158_urls,
        additional_instruction="Prefer district website pages (policies, handbooks, counseling/graduation pages) that explicitly state the district implements Act 158. Generic PDE pages without district mention are insufficient for this specific check."
    )

    kpp_block = evaluator.add_parallel(
        id="Keystone_Proficiency_Pathway",
        desc="The district offers the Keystone Proficiency Pathway as a graduation option",
        parent=grad_reqs,
        critical=True
    )

    # Proficiency score 1500 requirement
    score1500_block = evaluator.add_parallel(
        id="Proficiency_Score_1500",
        desc="The pathway requires achieving a proficient score of 1500 or higher on each Keystone Exam",
        parent=kpp_block,
        critical=True
    )
    score_url_presence = evaluator.add_custom_node(
        result=_has_urls(data.keystone_proficiency_urls),
        id="Score_Requirement_URL",
        desc="Provide a URL reference confirming the 1500 proficiency score requirement",
        parent=score1500_block,
        critical=True
    )
    score1500_leaf = evaluator.add_leaf(
        id="Proficiency_Score_1500_Verify",
        desc="Verify Keystone Proficiency Pathway requires score ≥ 1500 on each Keystone",
        parent=score1500_block,
        critical=True
    )
    await evaluator.verify(
        claim="Under the Keystone Proficiency Pathway, students must achieve a score of 1500 or higher (proficient or advanced) on each Keystone exam.",
        node=score1500_leaf,
        sources=data.keystone_proficiency_urls,
        additional_instruction="It is acceptable to use PDE/state pages or district pages that clearly specify the 1500 (proficient) threshold for each exam."
    )

    # Three required Keystone exams list
    three_exams_block = evaluator.add_parallel(
        id="Three_Keystone_Exams",
        desc="The pathway requires proficiency on all three Keystone Exams: Algebra I, Biology, and Literature",
        parent=kpp_block,
        critical=True
    )
    exam_list_url_presence = evaluator.add_custom_node(
        result=_has_urls(data.keystone_exam_list_urls),
        id="Exam_List_URL",
        desc="Provide a URL reference confirming the three required Keystone Exams",
        parent=three_exams_block,
        critical=True
    )
    three_exams_leaf = evaluator.add_leaf(
        id="Three_Keystone_Exams_Verify",
        desc="Verify the three Keystone exams are Algebra I, Biology, and Literature",
        parent=three_exams_block,
        critical=True
    )
    await evaluator.verify(
        claim="The three Keystone exams for graduation considerations are Algebra I, Biology, and Literature.",
        node=three_exams_leaf,
        sources=data.keystone_exam_list_urls,
        additional_instruction="Accept PDE/state pages or district pages that clearly list the Keystone exams as Algebra I, Biology, and Literature."
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
    Evaluate the answer for the Pennsylvania Superintendent of the Year (2024) with Act 158 and Keystone pathway requirements.
    """
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
        default_model=model
    )

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_superintendent(),
        template_class=SuperintendentExtraction,
        extraction_name="superintendent_extraction"
    )

    # Build verification tree and run checks
    await _build_and_verify(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()