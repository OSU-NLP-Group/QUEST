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
TASK_ID = "harvard_yale_planning_2027"
TASK_DESCRIPTION = """
You are a high school junior planning to apply to both Harvard University and Yale University for Fall 2027 admission. Your family's annual household income is $150,000. You are interested in studying Computer Science if you attend Harvard or Economics if you attend Yale. Create a comprehensive comparison document for a family planning meeting that includes the following four sections:

1. Application Timeline: Identify and document the Restrictive Early Action (or Single-Choice Early Action) deadline and the Regular Decision deadline for both Harvard and Yale. Provide reference URLs for these deadlines from official university sources.

2. Financial Aid Eligibility: Based on your family's annual income of $150,000, determine whether your family qualifies for free tuition at each university. Explain each university's income threshold policy and conclude whether you qualify. Provide reference URLs for the financial aid policies from official university sources.

3. Major Requirements: Document the specific academic requirements for:
   - Harvard's Computer Science concentration (including the number of core CS courses required and mathematics prerequisites)
   - Yale's Economics major (including total course requirements and the distribution of ECON vs. MATH courses)
   Provide reference URLs for these program requirements from official university sources.

4. Campus Visit Planning: Provide detailed logistics for visiting each campus, including tour duration and either registration requirements (for Harvard) or visitor center location (for Yale). Provide reference URLs for campus visit information from official university sources.
"""

# Family income context for logic checks
FAMILY_INCOME = 150_000

# Expected ground-truth style targets encoded in rubric (used as judging anchors)
EXP_HARVARD_REA = "November 1"
EXP_HARVARD_RD = "January 1"
EXP_YALE_SCEA = "November 1"
EXP_YALE_RD = "January 2"

EXP_HARVARD_CS_CORE_COUNT = "9"  # "9 core computer science courses"
EXP_HARVARD_CS_MATH_PREREQS_KEYWORDS = [
    "Linear Algebra", "Probability", "Discrete Mathematics", "Calculus"
]

EXP_YALE_ECON_TOTAL = "12"  # "12 term courses total"
# "7 ECON courses numbered above 2000 and 5 MATH courses"
EXP_YALE_ECON_DISTRIBUTION = "7 ECON courses numbered above 2000 and 5 MATH courses"

EXP_HARVARD_TOUR_DURATION = "45-60 minutes"
EXP_HARVARD_REGISTRATION_REQUIRED = "advance registration is required"

EXP_YALE_TOUR_DURATION = "60 minutes"
EXP_YALE_VISITOR_CENTER = "149 Elm Street, New Haven, CT"

# Financial aid rubric anchors
EXP_FREE_TUITION_THRESHOLD = 200_000  # Claim anchor used by the rubric


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class TimelineSchool(BaseModel):
    early_deadline: Optional[str] = None  # REA/SCEA deadline text as written in the answer
    regular_deadline: Optional[str] = None  # RD deadline text as written in the answer
    timeline_urls: List[str] = Field(default_factory=list)  # official deadline page URLs cited


class ApplicationTimelineExtraction(BaseModel):
    harvard: Optional[TimelineSchool] = None
    yale: Optional[TimelineSchool] = None


class AidSchool(BaseModel):
    policy_statement: Optional[str] = None  # summary/quote of the policy as written
    conclusion: Optional[str] = None        # e.g., "qualifies for free tuition" or similar
    urls: List[str] = Field(default_factory=list)  # official financial aid policy URLs cited


class FinancialAidExtraction(BaseModel):
    harvard: Optional[AidSchool] = None
    yale: Optional[AidSchool] = None


class HarvardCSRequirements(BaseModel):
    core_count: Optional[str] = None        # number of core CS courses as written
    math_prereqs: Optional[str] = None      # text listing math prereqs
    urls: List[str] = Field(default_factory=list)  # official CS requirements URLs cited


class YaleEconRequirements(BaseModel):
    total_courses: Optional[str] = None     # total term courses as written
    distribution: Optional[str] = None      # e.g., "7 ECON >2000 + 5 MATH"
    urls: List[str] = Field(default_factory=list)  # official Econ requirements URLs cited


class MajorRequirementsExtraction(BaseModel):
    harvard_cs: Optional[HarvardCSRequirements] = None
    yale_econ: Optional[YaleEconRequirements] = None


class HarvardVisit(BaseModel):
    tour_duration: Optional[str] = None
    registration_required: Optional[str] = None
    urls: List[str] = Field(default_factory=list)  # official Harvard visit URLs cited


class YaleVisit(BaseModel):
    tour_duration: Optional[str] = None
    visitor_center_location: Optional[str] = None
    urls: List[str] = Field(default_factory=list)  # official Yale visit URLs cited


class CampusVisitExtraction(BaseModel):
    harvard: Optional[HarvardVisit] = None
    yale: Optional[YaleVisit] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_application_timeline() -> str:
    return """
Extract the application timeline details for Harvard and Yale exactly as presented in the answer.

For each university, extract:
- early_deadline: The Restrictive Early Action (Harvard) or Single-Choice Early Action (Yale) deadline date string as written (e.g., "November 1", "Nov 1", or "11/1").
- regular_deadline: The Regular Decision deadline date string as written (e.g., "January 1", "Jan 1", or "1/1").
- timeline_urls: All official university URLs cited in the answer that reference these deadlines (Harvard College/Admissions domain or Yale Admissions domain). Return only valid, explicit URLs.

Return a JSON with this structure:
{
  "harvard": {"early_deadline": str|null, "regular_deadline": str|null, "timeline_urls": [urls...]},
  "yale": {"early_deadline": str|null, "regular_deadline": str|null, "timeline_urls": [urls...]}
}

If any field is not present in the answer, set it to null (or [] for URLs).
Do NOT invent any URLs. Extract only explicit URLs from the answer.
"""


def prompt_extract_financial_aid() -> str:
    return f"""
Extract the financial aid eligibility discussion for Harvard and Yale exactly as presented in the answer, focusing on the family income of ${FAMILY_INCOME}.

For each university, extract:
- policy_statement: The sentence/phrase summarizing the university's policy about free tuition thresholds as written in the answer (e.g., "provides free tuition for families earning $200,000 or less").
- conclusion: The conclusion in the answer about whether a family earning ${FAMILY_INCOME} qualifies for free tuition at that university, as written (e.g., "qualifies for free tuition" or "does not qualify").
- urls: All official university URLs cited in the answer that support the policy (financial aid pages or policy pages). Return only valid, explicit URLs.

Return JSON:
{{
  "harvard": {{"policy_statement": str|null, "conclusion": str|null, "urls": [urls...]}},
  "yale":    {{"policy_statement": str|null, "conclusion": str|null, "urls": [urls...]}}
}}

If anything is missing, use null (or [] for URLs). Do NOT invent URLs.
"""


def prompt_extract_major_requirements() -> str:
    return """
Extract the major requirements documentation exactly as presented in the answer.

Harvard Computer Science concentration:
- core_count: The number of "core" CS courses required as stated in the answer (e.g., "9").
- math_prereqs: The mathematics prerequisites text as listed (e.g., "Linear Algebra, Probability, Discrete Mathematics, plus preparatory Calculus").
- urls: Official Harvard CS/SEAS/College catalog URLs cited for CS requirements.

Yale Economics major:
- total_courses: Total number of term courses required as stated (e.g., "12").
- distribution: The distribution between ECON and MATH courses as stated (e.g., "7 ECON courses numbered above 2000 and 5 MATH courses").
- urls: Official Yale Economics/Yale College Program of Study URLs cited for Econ requirements.

Return JSON:
{
  "harvard_cs": {"core_count": str|null, "math_prereqs": str|null, "urls": [urls...]},
  "yale_econ":  {"total_courses": str|null, "distribution": str|null, "urls": [urls...]}
}

If anything is missing, set to null (or [] for URLs). Extract only explicit URLs from the answer.
"""


def prompt_extract_campus_visit() -> str:
    return """
Extract campus visit logistics exactly as presented in the answer.

Harvard:
- tour_duration: The stated duration for Harvard campus tours (e.g., "45-60 minutes").
- registration_required: The stated registration requirement (e.g., "advance registration required" or similar).
- urls: Official Harvard visit/admissions tour URLs cited.

Yale:
- tour_duration: The stated duration for Yale campus tours (e.g., "60 minutes").
- visitor_center_location: The stated visitor center/tour departure location (e.g., "149 Elm Street, New Haven, CT").
- urls: Official Yale visit/admissions tour URLs cited.

Return JSON:
{
  "harvard": {"tour_duration": str|null, "registration_required": str|null, "urls": [urls...]},
  "yale":    {"tour_duration": str|null, "visitor_center_location": str|null, "urls": [urls...]}
}

If anything is missing, set to null (or [] for URLs). Extract only explicit URLs from the answer.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nz_list(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_application_timeline_tree(
    evaluator: Evaluator,
    parent,
    timeline: ApplicationTimelineExtraction,
) -> None:
    # Section node (critical)
    section = evaluator.add_parallel(
        id="Application_Timeline_Section",
        desc="Complete and accurate documentation of application deadlines for both universities",
        parent=parent,
        critical=True,
    )

    # Harvard subtree
    harv = evaluator.add_parallel(
        id="Harvard_Application_Timeline",
        desc="Documentation of Harvard's complete application timeline with both deadlines and reference URL",
        parent=section,
        critical=True,
    )

    harv_dead = evaluator.add_parallel(
        id="Harvard_Deadlines",
        desc="Accurate identification of Harvard's application deadlines",
        parent=harv,
        critical=True,
    )

    harv_urls = _nz_list(timeline.harvard.timeline_urls if timeline and timeline.harvard else None)
    harv_early = (timeline.harvard.early_deadline if timeline and timeline.harvard else None) or ""
    harv_regular = (timeline.harvard.regular_deadline if timeline and timeline.harvard else None) or ""

    # Harvard REA
    leaf = evaluator.add_leaf(
        id="Harvard_REA_Deadline",
        desc="Correctly identifies Harvard's Restrictive Early Action deadline as November 1",
        parent=harv_dead,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Harvard's Restrictive Early Action (REA) deadline is {harv_early}.",
        node=leaf,
        sources=harv_urls,
        additional_instruction=f"Pass only if the cited official Harvard page explicitly supports the REA deadline "
                               f"AND the deadline equals {EXP_HARVARD_REA} (allow minor format variants like 'Nov 1', '11/1', 'November 1st'). "
                               f"If the extracted value is not {EXP_HARVARD_REA}, mark as incorrect.",
    )

    # Harvard RD
    leaf = evaluator.add_leaf(
        id="Harvard_RD_Deadline",
        desc="Correctly identifies Harvard's Regular Decision deadline as January 1",
        parent=harv_dead,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Harvard's Regular Decision (RD) deadline is {harv_regular}.",
        node=leaf,
        sources=harv_urls,
        additional_instruction=f"Pass only if the cited official Harvard page explicitly supports the RD deadline "
                               f"AND the deadline equals {EXP_HARVARD_RD} (allow minor format variants like 'Jan 1', '1/1', 'January 1st'). "
                               f"If the extracted value is not {EXP_HARVARD_RD}, mark as incorrect.",
    )

    # Harvard URL presence/officialness
    leaf = evaluator.add_leaf(
        id="Harvard_Timeline_URL",
        desc="Provides a valid reference URL for Harvard application deadlines from official Harvard sources",
        parent=harv,
        critical=True,
    )
    await evaluator.verify(
        claim="This is an official Harvard College/Admissions webpage that lists undergraduate first-year application "
              "deadlines (REA and Regular Decision).",
        node=leaf,
        sources=harv_urls,
        additional_instruction="Pass only if at least one provided URL is an official Harvard site page that clearly "
                               "lists application deadlines. If no URL is provided, mark as incorrect.",
    )

    # Yale subtree
    yale = evaluator.add_parallel(
        id="Yale_Application_Timeline",
        desc="Documentation of Yale's complete application timeline with both deadlines and reference URL",
        parent=section,
        critical=True,
    )

    yale_dead = evaluator.add_parallel(
        id="Yale_Deadlines",
        desc="Accurate identification of Yale's application deadlines",
        parent=yale,
        critical=True,
    )

    yale_urls = _nz_list(timeline.yale.timeline_urls if timeline and timeline.yale else None)
    yale_early = (timeline.yale.early_deadline if timeline and timeline.yale else None) or ""
    yale_regular = (timeline.yale.regular_deadline if timeline and timeline.yale else None) or ""

    # Yale SCEA
    leaf = evaluator.add_leaf(
        id="Yale_SCEA_Deadline",
        desc="Correctly identifies Yale's Single-Choice Early Action deadline as November 1",
        parent=yale_dead,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Yale's Single-Choice Early Action (SCEA) deadline is {yale_early}.",
        node=leaf,
        sources=yale_urls,
        additional_instruction=f"Pass only if the cited official Yale Admissions page explicitly supports the SCEA deadline "
                               f"AND the deadline equals {EXP_YALE_SCEA} (allow 'Nov 1', '11/1', 'November 1st'). "
                               f"If the extracted value is not {EXP_YALE_SCEA}, mark as incorrect.",
    )

    # Yale RD
    leaf = evaluator.add_leaf(
        id="Yale_RD_Deadline",
        desc="Correctly identifies Yale's Regular Decision deadline as January 2",
        parent=yale_dead,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Yale's Regular Decision (RD) deadline is {yale_regular}.",
        node=leaf,
        sources=yale_urls,
        additional_instruction=f"Pass only if the cited official Yale Admissions page explicitly supports the RD deadline "
                               f"AND the deadline equals {EXP_YALE_RD} (allow 'Jan 2', '1/2', 'January 2nd'). "
                               f"If the extracted value is not {EXP_YALE_RD}, mark as incorrect.",
    )

    # Yale URL presence/officialness
    leaf = evaluator.add_leaf(
        id="Yale_Timeline_URL",
        desc="Provides a valid reference URL for Yale application deadlines from official Yale sources",
        parent=yale,
        critical=True,
    )
    await evaluator.verify(
        claim="This is an official Yale Admissions webpage that lists undergraduate first-year application deadlines "
              "(Single-Choice Early Action and Regular Decision).",
        node=leaf,
        sources=yale_urls,
        additional_instruction="Pass only if at least one provided URL is an official Yale site page that clearly lists "
                               "application deadlines. If no URL is provided, mark as incorrect.",
    )


async def build_financial_aid_tree(
    evaluator: Evaluator,
    parent,
    aid: FinancialAidExtraction,
) -> None:
    section = evaluator.add_parallel(
        id="Financial_Aid_Section",
        desc="Accurate assessment and explanation of financial aid eligibility for $150,000 family income at both universities",
        parent=parent,
        critical=True,
    )

    # Harvard Aid (sequential)
    harv = evaluator.add_sequential(
        id="Harvard_Aid_Analysis",
        desc="Complete analysis of Harvard financial aid eligibility including policy, qualification determination, and reference URL",
        parent=section,
        critical=True,
    )

    harv_urls = _nz_list(aid.harvard.urls if aid and aid.harvard else None)
    harv_policy_text = (aid.harvard.policy_statement if aid and aid.harvard else None) or ""
    harv_conclusion = (aid.harvard.conclusion if aid and aid.harvard else None) or ""

    # Policy statement leaf
    leaf = evaluator.add_leaf(
        id="Harvard_Policy_Statement",
        desc="Correctly states that Harvard provides free tuition for families earning $200,000 or less",
        parent=harv,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The policy states: {harv_policy_text}",
        node=leaf,
        sources=harv_urls,
        additional_instruction=f"Pass only if the cited official Harvard financial aid page explicitly states that families "
                               f"earning up to or under ${EXP_FREE_TUITION_THRESHOLD:,} receive free tuition (allow synonyms like "
                               f"'tuition-free', 'no tuition', 'full tuition covered'). If the page does not clearly indicate "
                               f"a free-tuition threshold at ${EXP_FREE_TUITION_THRESHOLD:,}, mark as incorrect.",
    )

    # Qualification conclusion leaf (pure logic)
    leaf = evaluator.add_leaf(
        id="Harvard_Qualification_Conclusion",
        desc="Correctly concludes that a family earning $150,000 qualifies for Harvard's free tuition since $150,000 < $200,000",
        parent=harv,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Given a free-tuition threshold of ${EXP_FREE_TUITION_THRESHOLD:,} and a family income of ${FAMILY_INCOME:,}, "
              f"the conclusion '{harv_conclusion}' correctly asserts the family qualifies for free tuition at Harvard.",
        node=leaf,
        additional_instruction="This is a pure logic check: pass if and only if 150,000 < 200,000 and the conclusion asserts "
                               "qualification for free tuition. Ignore browsing and external info.",
    )

    # Aid URL officialness leaf
    leaf = evaluator.add_leaf(
        id="Harvard_Aid_URL",
        desc="Provides a valid reference URL for Harvard's financial aid policy from official Harvard sources",
        parent=harv,
        critical=True,
    )
    await evaluator.verify(
        claim="This is an official Harvard financial aid/policy webpage describing income thresholds and free tuition/no-cost eligibility.",
        node=leaf,
        sources=harv_urls,
        additional_instruction="Pass only if at least one provided URL is an official Harvard site that presents the policy. "
                               "If no URL is provided, mark as incorrect.",
    )

    # Yale Aid (sequential)
    yale = evaluator.add_sequential(
        id="Yale_Aid_Analysis",
        desc="Complete analysis of Yale financial aid eligibility including policy, qualification determination, and reference URL",
        parent=section,
        critical=True,
    )

    yale_urls = _nz_list(aid.yale.urls if aid and aid.yale else None)
    yale_policy_text = (aid.yale.policy_statement if aid and aid.yale else None) or ""
    yale_conclusion = (aid.yale.conclusion if aid and aid.yale else None) or ""

    # Policy statement leaf
    leaf = evaluator.add_leaf(
        id="Yale_Policy_Statement",
        desc="Correctly states that Yale provides free tuition for families earning under $200,000",
        parent=yale,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The policy states: {yale_policy_text}",
        node=leaf,
        sources=yale_urls,
        additional_instruction=f"Pass only if the cited official Yale financial aid page explicitly states that families "
                               f"earning under ${EXP_FREE_TUITION_THRESHOLD:,} receive free tuition (allow synonyms like "
                               f"'tuition-free', 'no tuition', 'full tuition covered'). If the page does not clearly indicate "
                               f"a free-tuition threshold near ${EXP_FREE_TUITION_THRESHOLD:,}, mark as incorrect.",
    )

    # Qualification conclusion leaf (pure logic)
    leaf = evaluator.add_leaf(
        id="Yale_Qualification_Conclusion",
        desc="Correctly concludes that a family earning $150,000 qualifies for Yale's free tuition since $150,000 < $200,000",
        parent=yale,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Given a free-tuition threshold of ${EXP_FREE_TUITION_THRESHOLD:,} and a family income of ${FAMILY_INCOME:,}, "
              f"the conclusion '{yale_conclusion}' correctly asserts the family qualifies for free tuition at Yale.",
        node=leaf,
        additional_instruction="This is a pure logic check: pass if and only if 150,000 < 200,000 and the conclusion asserts "
                               "qualification for free tuition. Ignore browsing and external info.",
    )

    # Aid URL officialness leaf
    leaf = evaluator.add_leaf(
        id="Yale_Aid_URL",
        desc="Provides a valid reference URL for Yale's financial aid policy from official Yale sources",
        parent=yale,
        critical=True,
    )
    await evaluator.verify(
        claim="This is an official Yale financial aid/policy webpage describing income thresholds and free tuition/no-cost eligibility.",
        node=leaf,
        sources=yale_urls,
        additional_instruction="Pass only if at least one provided URL is an official Yale site that presents the policy. "
                               "If no URL is provided, mark as incorrect.",
    )


async def build_major_requirements_tree(
    evaluator: Evaluator,
    parent,
    majors: MajorRequirementsExtraction,
) -> None:
    section = evaluator.add_parallel(
        id="Major_Requirements_Section",
        desc="Detailed and accurate documentation of specific major requirements at each university",
        parent=parent,
        critical=True,
    )

    # Harvard CS
    harv = evaluator.add_parallel(
        id="Harvard_CS_Documentation",
        desc="Complete documentation of Harvard Computer Science concentration requirements including core courses, math prerequisites, and reference URL",
        parent=section,
        critical=True,
    )

    harv_reqs = evaluator.add_parallel(
        id="Harvard_CS_Requirements",
        desc="Accurate documentation of Harvard CS course requirements",
        parent=harv,
        critical=True,
    )

    harv_urls = _nz_list(majors.harvard_cs.urls if majors and majors.harvard_cs else None)
    harv_core = (majors.harvard_cs.core_count if majors and majors.harvard_cs else None) or ""
    harv_math = (majors.harvard_cs.math_prereqs if majors and majors.harvard_cs else None) or ""

    # Core count
    leaf = evaluator.add_leaf(
        id="Harvard_CS_Core_Count",
        desc="Correctly identifies that Harvard CS requires 9 core computer science courses",
        parent=harv_reqs,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The number of core computer science courses required in Harvard's CS concentration is {harv_core}.",
        node=leaf,
        sources=harv_urls,
        additional_instruction=f"Pass only if the official Harvard CS requirements page supports the stated core requirement "
                               f"AND the number equals {EXP_HARVARD_CS_CORE_COUNT}. Allow minor phrasing variants like 'core'/'technical core'. "
                               f"If the extracted number is not {EXP_HARVARD_CS_CORE_COUNT}, mark as incorrect.",
    )

    # Math prerequisites
    leaf = evaluator.add_leaf(
        id="Harvard_CS_Math_Prerequisites",
        desc="Identifies mathematics prerequisites: Linear Algebra, Probability, Discrete Mathematics, plus preparatory calculus",
        parent=harv_reqs,
        critical=True,
    )
    expect_list = ", ".join(EXP_HARVARD_CS_MATH_PREREQS_KEYWORDS)
    await evaluator.verify(
        claim=f"Harvard CS mathematics prerequisites include the following areas (or clear equivalents): {harv_math}",
        node=leaf,
        sources=harv_urls,
        additional_instruction=f"Pass only if the official Harvard CS requirements page indicates math prerequisites covering: "
                               f"{expect_list} (accept equivalents like 'Math 21a/21b' mapping to Calculus/Linear Algebra; "
                               f"Probability and Discrete Mathematics can be named via course titles). "
                               f"If these areas are not all present, mark as incorrect.",
    )

    # Harvard CS URL
    leaf = evaluator.add_leaf(
        id="Harvard_CS_URL",
        desc="Provides a valid reference URL for Harvard CS concentration requirements from official Harvard sources",
        parent=harv,
        critical=True,
    )
    await evaluator.verify(
        claim="This is an official Harvard SEAS/Computer Science/College catalog webpage that lists CS concentration "
              "requirements including core courses and math prerequisites.",
        node=leaf,
        sources=harv_urls,
        additional_instruction="Pass only if at least one provided URL is an official Harvard site showing CS concentration requirements. "
                               "If no URL is provided, mark as incorrect.",
    )

    # Yale Econ
    yale = evaluator.add_parallel(
        id="Yale_Econ_Documentation",
        desc="Complete documentation of Yale Economics major requirements including course totals, distribution, and reference URL",
        parent=section,
        critical=True,
    )

    yale_reqs = evaluator.add_parallel(
        id="Yale_Econ_Requirements",
        desc="Accurate documentation of Yale Economics course requirements",
        parent=yale,
        critical=True,
    )

    yale_urls = _nz_list(majors.yale_econ.urls if majors and majors.yale_econ else None)
    yale_total = (majors.yale_econ.total_courses if majors and majors.yale_econ else None) or ""
    yale_dist = (majors.yale_econ.distribution if majors and majors.yale_econ else None) or ""

    # Total count
    leaf = evaluator.add_leaf(
        id="Yale_Econ_Total_Count",
        desc="Correctly identifies that Yale Economics requires 12 term courses total",
        parent=yale_reqs,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The total number of term courses required in Yale's Economics major is {yale_total}.",
        node=leaf,
        sources=yale_urls,
        additional_instruction=f"Pass only if the official Yale Economics/Yale College Program of Study page supports the stated total "
                               f"AND the total equals {EXP_YALE_ECON_TOTAL}. If the extracted number is not {EXP_YALE_ECON_TOTAL}, mark as incorrect.",
    )

    # Distribution
    leaf = evaluator.add_leaf(
        id="Yale_Econ_Distribution",
        desc="Correctly identifies the distribution: 7 ECON courses numbered above 2000 and 5 MATH courses",
        parent=yale_reqs,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The distribution between ECON and MATH courses in Yale's Economics major is: {yale_dist}.",
        node=leaf,
        sources=yale_urls,
        additional_instruction=f"Pass only if the official Yale page supports a distribution equivalent to: "
                               f"{EXP_YALE_ECON_DISTRIBUTION}. Allow minor phrasing variants; "
                               f"if numbering schemes differ, ensure equivalence is explicit. If not equivalent, mark as incorrect.",
    )

    # Yale Econ URL
    leaf = evaluator.add_leaf(
        id="Yale_Econ_URL",
        desc="Provides a valid reference URL for Yale Economics major requirements from official Yale sources",
        parent=yale,
        critical=True,
    )
    await evaluator.verify(
        claim="This is an official Yale Economics/Yale College catalog webpage that lists Economics major requirements, "
              "including total courses and the ECON vs. MATH distribution.",
        node=leaf,
        sources=yale_urls,
        additional_instruction="Pass only if at least one provided URL is an official Yale site showing Economics major requirements. "
                               "If no URL is provided, mark as incorrect.",
    )


async def build_campus_visit_tree(
    evaluator: Evaluator,
    parent,
    visits: CampusVisitExtraction,
) -> None:
    section = evaluator.add_parallel(
        id="Campus_Visit_Section",
        desc="Detailed and accurate campus visit logistics for both universities",
        parent=parent,
        critical=True,
    )

    # Harvard visit
    harv = evaluator.add_parallel(
        id="Harvard_Visit_Documentation",
        desc="Complete information about Harvard campus visits including duration, registration requirements, and reference URL",
        parent=section,
        critical=True,
    )

    harv_details = evaluator.add_parallel(
        id="Harvard_Visit_Details",
        desc="Specific details about Harvard campus tours",
        parent=harv,
        critical=True,
    )

    harv_urls = _nz_list(visits.harvard.urls if visits and visits.harvard else None)
    harv_dur = (visits.harvard.tour_duration if visits and visits.harvard else None) or ""
    harv_reg = (visits.harvard.registration_required if visits and visits.harvard else None) or ""

    # Harvard tour duration
    leaf = evaluator.add_leaf(
        id="Harvard_Tour_Duration",
        desc="Correctly identifies that Harvard tours are 45-60 minutes long",
        parent=harv_details,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Harvard campus tours are {harv_dur} long.",
        node=leaf,
        sources=harv_urls,
        additional_instruction=f"Pass only if the official Harvard visit page supports the stated duration and it is within "
                               f"{EXP_HARVARD_TOUR_DURATION} (allow phrasing like 'about an hour' so long as 45–60 minutes is implied). "
                               f"If not within that range, mark as incorrect.",
    )

    # Harvard registration requirement
    leaf = evaluator.add_leaf(
        id="Harvard_Registration_Requirement",
        desc="Correctly identifies that advance registration is required for Harvard campus tours",
        parent=harv_details,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The Harvard visit information indicates: {harv_reg}",
        node=leaf,
        sources=harv_urls,
        additional_instruction=f"Pass only if the official Harvard visit page states that {EXP_HARVARD_REGISTRATION_REQUIRED}. "
                               f"Allow synonymous language. If the extracted statement does not indicate advance registration is required, mark as incorrect.",
    )

    # Harvard visit URL
    leaf = evaluator.add_leaf(
        id="Harvard_Visit_URL",
        desc="Provides a valid reference URL for Harvard campus visit information from official Harvard sources",
        parent=harv,
        critical=True,
    )
    await evaluator.verify(
        claim="This is an official Harvard Admissions/Visit/Tours webpage providing campus visit logistics and tour details.",
        node=leaf,
        sources=harv_urls,
        additional_instruction="Pass only if at least one provided URL is an official Harvard site page about campus visits or tours. "
                               "If no URL is provided, mark as incorrect.",
    )

    # Yale visit
    yale = evaluator.add_parallel(
        id="Yale_Visit_Documentation",
        desc="Complete information about Yale campus visits including duration, visitor center location, and reference URL",
        parent=section,
        critical=True,
    )

    yale_details = evaluator.add_parallel(
        id="Yale_Visit_Details",
        desc="Specific details about Yale campus tours",
        parent=yale,
        critical=True,
    )

    yale_urls = _nz_list(visits.yale.urls if visits and visits.yale else None)
    yale_dur = (visits.yale.tour_duration if visits and visits.yale else None) or ""
    yale_loc = (visits.yale.visitor_center_location if visits and visits.yale else None) or ""

    # Yale tour duration
    leaf = evaluator.add_leaf(
        id="Yale_Tour_Duration",
        desc="Correctly identifies that Yale tours are 60 minutes long",
        parent=yale_details,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Yale campus tours are {yale_dur} long.",
        node=leaf,
        sources=yale_urls,
        additional_instruction=f"Pass only if the official Yale visit page supports the stated duration and it equals "
                               f"{EXP_YALE_TOUR_DURATION} (allow phrasing like 'one hour'). If not equivalent, mark as incorrect.",
    )

    # Yale Visitor Center location
    leaf = evaluator.add_leaf(
        id="Yale_Visitor_Center_Location",
        desc="Correctly identifies that Yale tours depart from Yale Visitor Center at 149 Elm Street, New Haven, CT",
        parent=yale_details,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Yale campus tours depart from the Yale Visitor Center located at {yale_loc}.",
        node=leaf,
        sources=yale_urls,
        additional_instruction=f"Pass only if the official Yale visit page supports that tours depart from the Yale Visitor Center at "
                               f"{EXP_YALE_VISITOR_CENTER} (allow minor formatting variations). If the extracted address differs, mark as incorrect.",
    )

    # Yale visit URL
    leaf = evaluator.add_leaf(
        id="Yale_Visit_URL",
        desc="Provides a valid reference URL for Yale campus visit information from official Yale sources",
        parent=yale,
        critical=True,
    )
    await evaluator.verify(
        claim="This is an official Yale Admissions/Visitor Center/Tours webpage providing campus visit logistics and tour details.",
        node=leaf,
        sources=yale_urls,
        additional_instruction="Pass only if at least one provided URL is an official Yale site page about campus visits, tours, or Visitor Center. "
                               "If no URL is provided, mark as incorrect.",
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
    Evaluate an answer for the Harvard/Yale Fall 2027 planning comparison document task.
    """
    # Initialize evaluator (root is always non-critical; create a critical top node under it)
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
        default_model=model,
    )

    # Perform extractions (in parallel)
    timeline_task = evaluator.extract(
        prompt=prompt_extract_application_timeline(),
        template_class=ApplicationTimelineExtraction,
        extraction_name="application_timeline",
    )
    aid_task = evaluator.extract(
        prompt=prompt_extract_financial_aid(),
        template_class=FinancialAidExtraction,
        extraction_name="financial_aid",
    )
    majors_task = evaluator.extract(
        prompt=prompt_extract_major_requirements(),
        template_class=MajorRequirementsExtraction,
        extraction_name="major_requirements",
    )
    visits_task = evaluator.extract(
        prompt=prompt_extract_campus_visit(),
        template_class=CampusVisitExtraction,
        extraction_name="campus_visits",
    )

    extracted_timeline, extracted_aid, extracted_majors, extracted_visits = await asyncio.gather(
        timeline_task, aid_task, majors_task, visits_task
    )

    # Add rubric ground-truth anchors for transparency (not used for scoring directly)
    evaluator.add_ground_truth({
        "deadlines_expected": {
            "harvard": {"REA": EXP_HARVARD_REA, "RD": EXP_HARVARD_RD},
            "yale": {"SCEA": EXP_YALE_SCEA, "RD": EXP_YALE_RD},
        },
        "aid_expected_free_tuition_threshold": f"${EXP_FREE_TUITION_THRESHOLD:,}",
        "harvard_cs_expected": {
            "core_count": EXP_HARVARD_CS_CORE_COUNT,
            "math_keywords": EXP_HARVARD_CS_MATH_PREREQS_KEYWORDS,
        },
        "yale_econ_expected": {
            "total_courses": EXP_YALE_ECON_TOTAL,
            "distribution": EXP_YALE_ECON_DISTRIBUTION,
        },
        "visits_expected": {
            "harvard": {"tour_duration": EXP_HARVARD_TOUR_DURATION, "registration": EXP_HARVARD_REGISTRATION_REQUIRED},
            "yale": {"tour_duration": EXP_YALE_TOUR_DURATION, "visitor_center": EXP_YALE_VISITOR_CENTER},
        }
    }, gt_type="rubric_targets")

    # Build the top-level critical node that corresponds to the rubric's document requirement
    doc_node = evaluator.add_parallel(
        id="Complete_Comparison_Document",
        desc="A comprehensive comparison document covering application deadlines, financial aid eligibility, major requirements, and campus visit logistics for both Harvard and Yale",
        parent=root,
        critical=True,
    )

    # Build and verify each rubric section
    await build_application_timeline_tree(evaluator, doc_node, extracted_timeline)
    await build_financial_aid_tree(evaluator, doc_node, extracted_aid)
    await build_major_requirements_tree(evaluator, doc_node, extracted_majors)
    await build_campus_visit_tree(evaluator, doc_node, extracted_visits)

    # Return evaluation summary
    return evaluator.get_summary()