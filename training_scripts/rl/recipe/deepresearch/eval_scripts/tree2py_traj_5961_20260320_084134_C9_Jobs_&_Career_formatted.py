import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "pro_dev_plans_four_clients"
TASK_DESCRIPTION = """
A career development consulting firm has four clients seeking to transition into licensed professional roles. For each client, create a comprehensive professional development plan that addresses all certification requirements, prerequisite qualifications, and ongoing continuing education obligations.

Client 1 - Professional Engineer (Florida):
Currently works as an engineering graduate with a bachelor's degree in mechanical engineering. Plans to obtain Professional Engineer (PE) license in Florida. Identify: (1) all prerequisite requirements for PE licensure, (2) required examinations, (3) the specific continuing education hour requirements for Florida PEs, including the total hours and renewal period.

Client 2 - Certified Financial Planner:
Currently works in financial services with a bachelor's degree in business. Plans to obtain CFP certification. Identify: (1) all education requirements including degree and CFP Board program completion, (2) examination requirements, (3) experience requirements, (4) the specific continuing education requirements for CFP professionals, including total hours per reporting period, required ethics hours, and general education hours.

Client 3 - Louisiana Teacher (Level 2):
Currently holds Louisiana Level 1 teacher certification and has been teaching for 2 years. Plans to advance to Level 2 certification. Identify: (1) the minimum teaching experience required including mentored experience, (2) the graduate degree requirement, (3) all other requirements for Level 2 certification advancement.

Client 4 - Project Management Professional:
Has an associate degree and 5 years of experience leading project teams. Plans to obtain PMP certification. Identify: (1) whether the client's education level meets PMP requirements, (2) the specific project management experience requirement in months based on their education level, (3) the timeframe within which experience must have occurred, (4) the requirement that experience involves leading and managing projects, (5) the required hours of project management education.

For each client, provide a structured plan that includes all certification prerequisites, required examinations, experience requirements, and continuing education obligations with specific hours and time periods. Include reference URLs documenting each requirement.
""".strip()


# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class Client1PEFlorida(BaseModel):
    # Structure
    structured: Optional[bool] = None  # whether sections for prerequisites, exams, CE are clearly separated

    # Prerequisites (beyond just listing exams)
    prereq_text: Optional[str] = None
    prereq_urls: List[str] = Field(default_factory=list)

    # Exams
    fe_exam_stated: Optional[bool] = None
    fe_urls: List[str] = Field(default_factory=list)
    pe_exam_stated: Optional[bool] = None
    pe_urls: List[str] = Field(default_factory=list)

    # Continuing Education
    ce_hours_stated: Optional[bool] = None     # whether hours explicitly stated in answer
    ce_hours_value: Optional[str] = None       # e.g., "18"
    ce_period_value: Optional[str] = None      # e.g., "every 2 years", "biennial"
    ce_urls: List[str] = Field(default_factory=list)


class Client2CFP(BaseModel):
    structured: Optional[bool] = None

    # Education
    bachelors_stated: Optional[bool] = None
    bachelors_urls: List[str] = Field(default_factory=list)

    cfp_program_stated: Optional[bool] = None
    cfp_program_urls: List[str] = Field(default_factory=list)

    # Exam
    exam_stated: Optional[bool] = None
    exam_urls: List[str] = Field(default_factory=list)

    # Experience
    experience_stated: Optional[bool] = None
    experience_urls: List[str] = Field(default_factory=list)

    # CE
    ce_total_stated: Optional[bool] = None
    ce_total_value: Optional[str] = None     # e.g., "30"
    ce_period: Optional[str] = None          # e.g., "every 2 years"
    ce_total_urls: List[str] = Field(default_factory=list)

    ce_ethics_stated: Optional[bool] = None
    ce_ethics_value: Optional[str] = None    # e.g., "2"
    ce_ethics_urls: List[str] = Field(default_factory=list)

    ce_general_stated: Optional[bool] = None
    ce_general_value: Optional[str] = None   # e.g., "28"
    ce_general_urls: List[str] = Field(default_factory=list)


class Client3LATeacherL2(BaseModel):
    structured: Optional[bool] = None

    level1_prereq_stated: Optional[bool] = None
    level1_urls: List[str] = Field(default_factory=list)

    exp_3yrs_stated: Optional[bool] = None
    exp_3yrs_urls: List[str] = Field(default_factory=list)

    mentored_1yr_stated: Optional[bool] = None
    mentored_urls: List[str] = Field(default_factory=list)

    masters_stated: Optional[bool] = None
    masters_urls: List[str] = Field(default_factory=list)

    # Additional requirements beyond experience and master's; if asserting none, text should reflect that
    other_requirements_text: Optional[str] = None
    other_requirements_urls: List[str] = Field(default_factory=list)


class Client4PMP(BaseModel):
    structured: Optional[bool] = None

    # Education eligibility (associate-degree path)
    education_eligibility_stated: Optional[bool] = None
    education_eligibility_affirmative: Optional[bool] = None  # True if answer states "associate degree meets a PMP eligibility path"
    education_eligibility_statement: Optional[str] = None     # as stated in the answer
    education_eligibility_urls: List[str] = Field(default_factory=list)

    # Experience months for associate-degree path
    exp_months_stated: Optional[bool] = None
    exp_months_value: Optional[str] = None     # e.g., "60 months"
    exp_months_urls: List[str] = Field(default_factory=list)

    # Timeframe within which experience must have occurred
    timeframe_stated: Optional[bool] = None
    timeframe_value: Optional[str] = None      # typically "8 years"
    timeframe_urls: List[str] = Field(default_factory=list)

    # Leading/managing nature
    leading_managing_stated: Optional[bool] = None
    leading_managing_urls: List[str] = Field(default_factory=list)

    # PM education hours
    pm_education_hours_stated: Optional[bool] = None
    pm_education_hours_value: Optional[str] = None  # e.g., "35"
    pm_education_urls: List[str] = Field(default_factory=list)

    # Exam requirement
    pmp_exam_stated: Optional[bool] = None
    pmp_exam_urls: List[str] = Field(default_factory=list)


class PlansExtraction(BaseModel):
    client1: Optional[Client1PEFlorida] = None
    client2: Optional[Client2CFP] = None
    client3: Optional[Client3LATeacherL2] = None
    client4: Optional[Client4PMP] = None


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_plans() -> str:
    return """
You will extract structured information for four client certification plans from the provided answer text. Do not invent any information. If a field is not clearly stated, set it to null (or empty list for URLs).

General rules for URLs:
- Extract only actual URLs explicitly present in the answer.
- Accept plain URLs or markdown links; capture the target URL.
- If no URLs are provided for a requirement, return an empty list.

For booleans like "*_stated", set to true ONLY if the answer explicitly states the requirement in reasonably clear terms (allow minor wording variations).

Return a JSON conforming to the following schema:

client1:
  structured: boolean (true if the plan has clearly separated sections/steps for prerequisites, exams, and continuing education)
  prereq_text: string summary of Florida PE prerequisites BEYOND just listing exams (e.g., education, EIT/FE status, supervised experience), as stated in the answer
  prereq_urls: array of URLs cited for Florida PE prerequisites
  fe_exam_stated: boolean (true if the answer states FE exam is required)
  fe_urls: array of URLs for FE requirement
  pe_exam_stated: boolean (true if the answer states PE exam is required)
  pe_urls: array of URLs for PE requirement
  ce_hours_stated: boolean (true if hours and renewal cadence are explicitly stated)
  ce_hours_value: string (e.g., "18" if stated)
  ce_period_value: string (e.g., "every 2 years", "biennial" if stated)
  ce_urls: array of URLs for CE requirement

client2:
  structured: boolean (sections for education, exam, experience, CE)
  bachelors_stated: boolean (bachelor’s degree required)
  bachelors_urls: array
  cfp_program_stated: boolean (CFP Board-registered program required)
  cfp_program_urls: array
  exam_stated: boolean (CFP exam required)
  exam_urls: array
  experience_stated: boolean (qualifying experience required)
  experience_urls: array
  ce_total_stated: boolean (total CE and period)
  ce_total_value: string (e.g., "30")
  ce_period: string (e.g., "every 2 years")
  ce_total_urls: array
  ce_ethics_stated: boolean (2 hours ethics)
  ce_ethics_value: string (e.g., "2")
  ce_ethics_urls: array
  ce_general_stated: boolean (28 hours general)
  ce_general_value: string (e.g., "28")
  ce_general_urls: array

client3:
  structured: boolean (sections for experience, degree, other requirements)
  level1_prereq_stated: boolean (Level 1 required to advance to Level 2)
  level1_urls: array
  exp_3yrs_stated: boolean (3 years successful teaching required)
  exp_3yrs_urls: array
  mentored_1yr_stated: boolean (mentored experience with at least 1 year)
  mentored_urls: array
  masters_stated: boolean (appropriate master’s degree required)
  masters_urls: array
  other_requirements_text: string (any other requirements beyond experience and master’s; if asserting none, write "none stated" or similar)
  other_requirements_urls: array (citations supporting additional requirements, or supporting that only listed requirements apply)

client4:
  structured: boolean (sections for eligibility, experience, timeframe/nature, PM education hours, examinations)
  education_eligibility_stated: boolean (answer explicitly judges whether associate degree meets a PMP eligibility path)
  education_eligibility_affirmative: boolean (true if answer says associate degree DOES meet a PMP path; false if answer says it does NOT)
  education_eligibility_statement: string (verbatim or concise paraphrase of the statement in the answer)
  education_eligibility_urls: array
  exp_months_stated: boolean (months of PM experience for associate-degree path is stated)
  exp_months_value: string (e.g., "60 months")
  exp_months_urls: array
  timeframe_stated: boolean (timeframe within which experience must have occurred is stated)
  timeframe_value: string (e.g., "8 years" if stated)
  timeframe_urls: array
  leading_managing_stated: boolean (experience must involve leading and managing projects is stated)
  leading_managing_urls: array
  pm_education_hours_stated: boolean (PM education hours are stated)
  pm_education_hours_value: string (e.g., "35")
  pm_education_urls: array
  pmp_exam_stated: boolean (PMP exam required is stated)
  pmp_exam_urls: array
""".strip()


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _urls_or_none(urls: Optional[List[str]]) -> Optional[List[str]]:
    if not urls:
        return None
    cleaned = [u for u in urls if isinstance(u, str) and u.strip()]
    return cleaned if cleaned else None


# -----------------------------------------------------------------------------
# Verification per client
# -----------------------------------------------------------------------------
async def verify_client1(evaluator: Evaluator, parent_node, c1: Optional[Client1PEFlorida]) -> None:
    node = evaluator.add_parallel(
        id="Client_1_PE_Florida",
        desc="Client 1: Florida Professional Engineer (PE) plan meets prompt requirements (prereqs, exams, CE) with citations.",
        parent=parent_node,
        critical=False
    )

    # Structured plan (critical)
    structured = bool(c1 and c1.structured)
    evaluator.add_custom_node(
        result=structured,
        id="C1_Structured_Plan",
        desc="Plan is structured (clearly separated sections/steps for prerequisites, exams, and continuing education).",
        parent=node,
        critical=True
    )

    # Prerequisites with citations (critical)
    leaf = evaluator.add_leaf(
        id="C1_Prerequisites_With_Citations",
        desc="Identifies Florida PE licensure prerequisite requirements (beyond just listing exams) and includes reference URL(s) documenting the prerequisites stated.",
        parent=node,
        critical=True
    )
    prereq_claim = (
        "The answer identifies Florida PE licensure prerequisite requirements beyond just listing exams "
        "and includes at least one reference URL. The provided source(s) support the stated prerequisites."
    )
    await evaluator.verify(
        claim=prereq_claim,
        node=leaf,
        sources=_urls_or_none(c1.prereq_urls if c1 else None),
        additional_instruction="Judge two things: (1) the answer text explicitly lists Florida PE prerequisites other than just the FE/PE exams "
                               "(e.g., education, EIT status, supervised experience, application to the board, etc.); "
                               "(2) at least one cited URL clearly supports those prerequisites. If either is missing, mark Incorrect."
    )

    # FE exam with citation (critical)
    leaf = evaluator.add_leaf(
        id="C1_Exams_FE_With_Citation",
        desc="States the Fundamentals of Engineering (FE) exam is required for Florida PE licensure and provides a supporting reference URL.",
        parent=node,
        critical=True
    )
    fe_claim = (
        "The answer states that the Fundamentals of Engineering (FE) exam is required for Florida PE licensure, "
        "and the provided source(s) support this requirement."
    )
    await evaluator.verify(
        claim=fe_claim,
        node=leaf,
        sources=_urls_or_none(c1.fe_urls if c1 else None),
        additional_instruction="Check the answer explicitly mentions the FE exam as a requirement and that at least one provided URL (e.g., FBPE, NCEES) supports it."
    )

    # PE exam with citation (critical)
    leaf = evaluator.add_leaf(
        id="C1_Exams_PE_With_Citation",
        desc="States the Professional Engineering (PE) exam is required for Florida PE licensure and provides a supporting reference URL.",
        parent=node,
        critical=True
    )
    pe_claim = (
        "The answer states that the Professional Engineering (PE) exam is required for Florida PE licensure, "
        "and the provided source(s) support this requirement."
    )
    await evaluator.verify(
        claim=pe_claim,
        node=leaf,
        sources=_urls_or_none(c1.pe_urls if c1 else None),
        additional_instruction="Check the answer explicitly mentions the PE (Principles and Practice of Engineering) exam and that at least one provided URL supports it."
    )

    # CE hours and period with citation (critical)
    leaf = evaluator.add_leaf(
        id="C1_CE_Hours_And_Period_With_Citation",
        desc="States Florida PEs must complete 18 hours of continuing education every 2 years (biennial renewal) and provides a supporting reference URL.",
        parent=node,
        critical=True
    )
    ce_claim = (
        "The answer states that Florida licensed Professional Engineers must complete 18 hours of continuing education "
        "every 2 years (biennial renewal), and the provided source(s) support this requirement."
    )
    await evaluator.verify(
        claim=ce_claim,
        node=leaf,
        sources=_urls_or_none(c1.ce_urls if c1 else None),
        additional_instruction="Allow wording equivalents such as 'biennially' or 'every two years'. If the answer does not state both the total (18 hours) "
                               "and the biennial cadence, or does not provide a supporting URL, mark Incorrect."
    )


async def verify_client2(evaluator: Evaluator, parent_node, c2: Optional[Client2CFP]) -> None:
    node = evaluator.add_parallel(
        id="Client_2_CFP",
        desc="Client 2: Certified Financial Planner (CFP) plan meets prompt requirements (education, exam, experience, CE) with citations.",
        parent=parent_node,
        critical=False
    )

    # Structured plan
    evaluator.add_custom_node(
        result=bool(c2 and c2.structured),
        id="C2_Structured_Plan",
        desc="Plan is structured (clearly separated sections/steps for education, exam, experience, and continuing education).",
        parent=node,
        critical=True
    )

    # Bachelor's degree
    leaf = evaluator.add_leaf(
        id="C2_Education_Bachelors_With_Citation",
        desc="States CFP requires a bachelor’s degree (in any discipline) and provides a supporting reference URL.",
        parent=node,
        critical=True
    )
    claim = ("The answer states that CFP certification requires a bachelor’s degree (in any discipline), "
             "and the provided source(s) support this requirement.")
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_urls_or_none(c2.bachelors_urls if c2 else None),
        additional_instruction="Check the answer explicitly mentions the bachelor's degree requirement and that at least one cited URL (e.g., CFP Board) supports it."
    )

    # CFP program
    leaf = evaluator.add_leaf(
        id="C2_Education_CFP_Program_With_Citation",
        desc="States CFP requires completion of a CFP Board-registered education program and provides a supporting reference URL.",
        parent=node,
        critical=True
    )
    claim = ("The answer states that CFP certification requires completion of a CFP Board-registered education program, "
             "and the provided source(s) support this requirement.")
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_urls_or_none(c2.cfp_program_urls if c2 else None),
        additional_instruction="Verify the answer states completion of a CFP Board-registered program and that at least one cited URL supports it."
    )

    # Exam
    leaf = evaluator.add_leaf(
        id="C2_Exam_With_Citation",
        desc="States passing the CFP exam is required and provides a supporting reference URL.",
        parent=node,
        critical=True
    )
    claim = "The answer states that passing the CFP exam is required, and the provided source(s) support this requirement."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_urls_or_none(c2.exam_urls if c2 else None),
        additional_instruction="Confirm the answer mentions the CFP exam requirement and that at least one cited URL supports it."
    )

    # Experience
    leaf = evaluator.add_leaf(
        id="C2_Experience_With_Citation",
        desc="States that demonstrating qualifying financial planning experience is required and provides a supporting reference URL.",
        parent=node,
        critical=True
    )
    claim = ("The answer states that qualifying financial planning experience is required for CFP certification, "
             "and the provided source(s) support this requirement.")
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_urls_or_none(c2.experience_urls if c2 else None),
        additional_instruction="Check that the answer mentions an experience requirement and at least one URL supports it."
    )

    # CE total and period
    leaf = evaluator.add_leaf(
        id="C2_CE_Total_Period_With_Citation",
        desc="States CFP professionals must complete 30 hours of CE every 2 years and provides a supporting reference URL.",
        parent=node,
        critical=True
    )
    claim = ("The answer states that CFP professionals must complete 30 hours of continuing education every 2 years, "
             "and the provided source(s) support this requirement.")
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_urls_or_none(c2.ce_total_urls if c2 else None),
        additional_instruction="Allow 'biennial' wording. If the answer lacks either the '30 hours' or the 'every 2 years' cadence, or lacks a citation, mark Incorrect."
    )

    # CE ethics 2 hours
    leaf = evaluator.add_leaf(
        id="C2_CE_Ethics_With_Citation",
        desc="States CFP CE includes 2 hours of ethics per reporting period and provides a supporting reference URL.",
        parent=node,
        critical=True
    )
    claim = ("The answer states that CFP continuing education includes 2 hours of ethics per reporting period, "
             "and the provided source(s) support this requirement.")
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_urls_or_none(c2.ce_ethics_urls if c2 else None),
        additional_instruction="Confirm explicit mention of 2 ethics hours and at least one URL supporting it."
    )

    # CE general 28 hours
    leaf = evaluator.add_leaf(
        id="C2_CE_General_With_Citation",
        desc="States CFP CE includes 28 hours of general CE per reporting period and provides a supporting reference URL.",
        parent=node,
        critical=True
    )
    claim = ("The answer states that CFP continuing education includes 28 hours of general CE per reporting period (in addition to ethics), "
             "and the provided source(s) support this requirement.")
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_urls_or_none(c2.ce_general_urls if c2 else None),
        additional_instruction="Confirm explicit mention of 28 general hours and at least one URL supporting it."
    )


async def verify_client3(evaluator: Evaluator, parent_node, c3: Optional[Client3LATeacherL2]) -> None:
    node = evaluator.add_parallel(
        id="Client_3_LA_Teacher_Level2",
        desc="Client 3: Louisiana Teacher Level 2 advancement plan meets prompt requirements (experience incl. mentored, master’s, other requirements) with citations.",
        parent=parent_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(c3 and c3.structured),
        id="C3_Structured_Plan",
        desc="Plan is structured (clearly separated sections/steps for experience, degree, and other advancement requirements).",
        parent=node,
        critical=True
    )

    # Level 1 prerequisite
    leaf = evaluator.add_leaf(
        id="C3_Level1_Prereq_With_Citation",
        desc="States Level 1 certification is required to advance to Level 2 and provides a supporting reference URL.",
        parent=node,
        critical=True
    )
    claim = ("The answer states that holding Level 1 certification is required to advance to Level 2 in Louisiana, "
             "and the provided source(s) support this prerequisite.")
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_urls_or_none(c3.level1_urls if c3 else None),
        additional_instruction="Confirm explicit mention of Level 1 as a prerequisite and at least one supporting URL from an authoritative source."
    )

    # 3 years experience
    leaf = evaluator.add_leaf(
        id="C3_Experience_3_Years_With_Citation",
        desc="States Level 2 advancement requires 3 years of successful teaching experience and provides a supporting reference URL.",
        parent=node,
        critical=True
    )
    claim = ("The answer states that Louisiana Level 2 certification advancement requires 3 years of successful teaching experience, "
             "and the provided source(s) support this requirement.")
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_urls_or_none(c3.exp_3yrs_urls if c3 else None),
        additional_instruction="Look for '3 years' and that the experience must be successful/effective per state guidance."
    )

    # Mentored at least 1 year
    leaf = evaluator.add_leaf(
        id="C3_Mentored_1_Year_With_Citation",
        desc="States Level 2 advancement requires mentored experience with at least 1 year mentored and provides a supporting reference URL.",
        parent=node,
        critical=True
    )
    claim = ("The answer states that Level 2 advancement requires mentored experience with at least 1 year under mentorship, "
             "and the provided source(s) support this requirement.")
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_urls_or_none(c3.mentored_urls if c3 else None),
        additional_instruction="Verify the answer mentions mentored experience and at least one year minimum, supported by cited URL(s)."
    )

    # Master's degree
    leaf = evaluator.add_leaf(
        id="C3_Masters_With_Citation",
        desc="States an appropriate master’s degree is required for Level 2 advancement and provides a supporting reference URL.",
        parent=node,
        critical=True
    )
    claim = ("The answer states that an appropriate master’s degree is required for Louisiana Level 2 certification advancement, "
             "and the provided source(s) support this requirement.")
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_urls_or_none(c3.masters_urls if c3 else None),
        additional_instruction="Confirm the answer mentions a master's degree requirement and that at least one provided URL supports it."
    )

    # Other requirements (or asserting none) with citations
    leaf = evaluator.add_leaf(
        id="C3_Other_Level2_Requirements_With_Citations",
        desc="Identifies any other Level 2 advancement requirements beyond experience and master’s degree, and provides reference URL(s) documenting each additional requirement stated; if asserting none, includes a citation indicating only the listed requirements apply.",
        parent=node,
        critical=True
    )
    if c3 and c3.other_requirements_text and c3.other_requirements_text.strip().lower() not in {"none", "none stated", "no other requirements", "no additional requirements"}:
        claim = (
            "The answer identifies additional Louisiana Level 2 advancement requirements beyond experience and a master’s degree, "
            "and the provided source(s) support these additional requirements."
        )
    else:
        claim = (
            "The answer states there are no additional Louisiana Level 2 advancement requirements beyond the listed experience and master’s degree, "
            "and the provided source(s) indicate that only those listed requirements apply."
        )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=_urls_or_none(c3.other_requirements_urls if c3 else None),
        additional_instruction="If the answer lists additional requirements, verify at least one URL supports them. If it asserts none, "
                               "verify at least one authoritative URL indicates that only the listed requirements apply."
    )


async def verify_client4(evaluator: Evaluator, parent_node, c4: Optional[Client4PMP]) -> None:
    node = evaluator.add_parallel(
        id="Client_4_PMP",
        desc="Client 4: PMP plan meets prompt requirements (eligibility by education, experience months, timeframe, leading/managing nature, PM education hours, exams) with citations.",
        parent=parent_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(c4 and c4.structured),
        id="C4_Structured_Plan",
        desc="Plan is structured (clearly separated sections/steps for eligibility, experience, timeframe/nature, PM education hours, and examinations).",
        parent=node,
        critical=True
    )

    # Education eligibility (associate-degree path)
    leaf = evaluator.add_leaf(
        id="C4_Education_Eligibility_With_Citation",
        desc="Correctly evaluates whether the client’s associate degree meets a PMP eligibility path and provides a supporting reference URL.",
        parent=node,
        critical=True
    )
    if c4 and c4.education_eligibility_affirmative is not None:
        if c4.education_eligibility_affirmative:
            edu_claim = ("The answer states that an associate degree satisfies one acceptable PMP eligibility path, "
                         "and the provided source(s) support this evaluation.")
        else:
            edu_claim = ("The answer states that an associate degree does NOT satisfy any PMP eligibility path, "
                         "and the provided source(s) support this evaluation.")
    else:
        edu_claim = ("The answer explicitly evaluates whether an associate degree meets a PMP eligibility path, "
                     "and the provided source(s) support that evaluation.")
    await evaluator.verify(
        claim=edu_claim,
        node=leaf,
        sources=_urls_or_none(c4.education_eligibility_urls if c4 else None),
        additional_instruction="Check that the answer clearly states whether the associate-degree education meets a PMP eligibility route and that at least one authoritative URL supports that judgment."
    )

    # Experience months for associate-degree path
    leaf = evaluator.add_leaf(
        id="C4_Experience_Months_With_Citation",
        desc="States the PMP project management experience requirement in months for the client’s education path (associate-degree path) and provides a supporting reference URL.",
        parent=node,
        critical=True
    )
    exp_months_value = (c4.exp_months_value if c4 and c4.exp_months_value else "the required number of")
    exp_months_claim = (
        f"The answer states that for the associate-degree eligibility path, PMP requires {exp_months_value} months "
        f"of project management experience, and the provided source(s) support this requirement."
    )
    await evaluator.verify(
        claim=exp_months_claim,
        node=leaf,
        sources=_urls_or_none(c4.exp_months_urls if c4 else None),
        additional_instruction="Verify the answer explicitly states the months of experience for the associate-degree path (e.g., 60 months) and that at least one URL supports it."
    )

    # Timeframe for experience occurrence (past 8 years)
    leaf = evaluator.add_leaf(
        id="C4_Experience_Timeframe_With_Citation",
        desc="States the timeframe within which the PMP experience must have occurred (past 8 years) and provides a supporting reference URL.",
        parent=node,
        critical=True
    )
    timeframe_claim = (
        "The answer states that the project management experience must have occurred within the past 8 years, "
        "and the provided source(s) support this requirement."
    )
    await evaluator.verify(
        claim=timeframe_claim,
        node=leaf,
        sources=_urls_or_none(c4.timeframe_urls if c4 else None),
        additional_instruction="Confirm explicit mention of the 8-year look-back window and at least one URL supporting it."
    )

    # Leading and managing nature
    leaf = evaluator.add_leaf(
        id="C4_Experience_Leading_Managing_With_Citation",
        desc="States PMP experience must involve leading and managing projects and provides a supporting reference URL.",
        parent=node,
        critical=True
    )
    lm_claim = (
        "The answer states that qualifying PMP experience must involve leading and managing projects, "
        "and the provided source(s) support this requirement."
    )
    await evaluator.verify(
        claim=lm_claim,
        node=leaf,
        sources=_urls_or_none(c4.leading_managing_urls if c4 else None),
        additional_instruction="Verify the answer explicitly mentions 'leading and managing' project experience and at least one URL supports it."
    )

    # PM education hours (35)
    leaf = evaluator.add_leaf(
        id="C4_PM_Education_Hours_With_Citation",
        desc="States PMP requires 35 hours of project management education/training and provides a supporting reference URL.",
        parent=node,
        critical=True
    )
    edu_hours_claim = (
        "The answer states that PMP eligibility requires 35 hours of project management education/training, "
        "and the provided source(s) support this requirement."
    )
    await evaluator.verify(
        claim=edu_hours_claim,
        node=leaf,
        sources=_urls_or_none(c4.pm_education_urls if c4 else None),
        additional_instruction="Confirm explicit mention of '35 hours' and at least one URL supporting it."
    )

    # PMP exam required
    leaf = evaluator.add_leaf(
        id="C4_PMP_Exam_With_Citation",
        desc="States that passing the PMP certification exam is required and provides a supporting reference URL.",
        parent=node,
        critical=True
    )
    pmp_exam_claim = (
        "The answer states that passing the PMP certification exam is required, "
        "and the provided source(s) support this requirement."
    )
    await evaluator.verify(
        claim=pmp_exam_claim,
        node=leaf,
        sources=_urls_or_none(c4.pmp_exam_urls if c4 else None),
        additional_instruction="Verify the answer mentions the PMP exam requirement and at least one cited URL supports it."
    )


# -----------------------------------------------------------------------------
# Main evaluation entry
# -----------------------------------------------------------------------------
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

    # Extract structured info once
    extraction = await evaluator.extract(
        prompt=prompt_extract_plans(),
        template_class=PlansExtraction,
        extraction_name="plans_extraction"
    )

    # Build and verify tree per client
    await verify_client1(evaluator, root, extraction.client1 if extraction else None)
    await verify_client2(evaluator, root, extraction.client2 if extraction else None)
    await verify_client3(evaluator, root, extraction.client3 if extraction else None)
    await verify_client4(evaluator, root, extraction.client4 if extraction else None)

    # Return evaluator summary
    return evaluator.get_summary()