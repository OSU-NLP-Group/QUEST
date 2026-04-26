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
TASK_ID = "pmp_requirements_2026"
TASK_DESCRIPTION = (
    "You are considering obtaining the Project Management Professional (PMP) certification to advance your career in "
    "project management. Research and document the complete eligibility requirements and key application details for "
    "the PMP certification in the United States as of February 2026.\n\n"
    "Your research must include:\n\n"
    "1. Educational Requirements: What are the two educational pathways that qualify you for PMP certification? "
    "(Specify both the higher education option and the alternative option.)\n\n"
    "2. Experience Requirements:\n"
    "   - How many months of project management experience are required if you have a four-year degree?\n"
    "   - How many months of project management experience are required if you only have a high school diploma?\n"
    "   - Within what timeframe must this experience have been obtained?\n"
    "   - Must the experience specifically involve leading and directing projects, or is any project participation sufficient?\n\n"
    "3. Training Requirements:\n"
    "   - How many contact hours of project management education or training are required before you can take the PMP exam?\n"
    "   - Can an active CAPM (Certified Associate in Project Management) certification substitute for this training requirement?\n\n"
    "4. 2026 Exam Transition:\n"
    "   - In which specific month of 2026 will the new version of the PMP exam be launched?\n"
    "   - What is the last date to take the current version of the exam before the new version launches?\n\n"
    "5. Certification Administration:\n"
    "   - What is the full name of the organization that administers and governs the PMP certification?\n"
    "   - Where (which platform or system) must you submit your PMP certification application?\n"
    "   - Is membership in this governing organization required to apply for PMP certification, or is it optional?\n\n"
    "6. Exam Quality Assurance:\n"
    "   - Has the governing organization published an official examination content outline document specifically for the 2026 PMP exam?\n"
    "   - Are PMP exam questions reviewed by subject matter experts who themselves hold valid PMP certification?\n\n"
    "For each piece of information you provide, include the reference URL where you found that information."
)

AS_OF_DATE = "February 2026"

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class PMPRequirementsExtraction(BaseModel):
    educational_higher: Optional[str] = None
    educational_higher_sources: List[str] = Field(default_factory=list)

    educational_alternative: Optional[str] = None
    educational_alternative_sources: List[str] = Field(default_factory=list)

    exp_months_bachelor: Optional[str] = None
    exp_months_bachelor_sources: List[str] = Field(default_factory=list)

    exp_months_hs_diploma: Optional[str] = None
    exp_months_hs_diploma_sources: List[str] = Field(default_factory=list)

    exp_timeframe: Optional[str] = None
    exp_timeframe_sources: List[str] = Field(default_factory=list)

    exp_leading_directing: Optional[str] = None
    exp_leading_directing_sources: List[str] = Field(default_factory=list)

    training_contact_hours: Optional[str] = None
    training_contact_hours_sources: List[str] = Field(default_factory=list)

    training_capm_substitute: Optional[str] = None
    training_capm_substitute_sources: List[str] = Field(default_factory=list)

    new_exam_launch_month: Optional[str] = None
    new_exam_launch_month_sources: List[str] = Field(default_factory=list)

    current_exam_last_date: Optional[str] = None
    current_exam_last_date_sources: List[str] = Field(default_factory=list)

    governing_org_full_name: Optional[str] = None
    governing_org_sources: List[str] = Field(default_factory=list)

    application_platform: Optional[str] = None
    application_platform_sources: List[str] = Field(default_factory=list)

    pmi_membership_required: Optional[str] = None
    pmi_membership_required_sources: List[str] = Field(default_factory=list)

    exam_content_outline_2026_published: Optional[str] = None
    exam_content_outline_sources: List[str] = Field(default_factory=list)

    sme_review_pmp_questions: Optional[str] = None
    sme_review_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pmp_requirements() -> str:
    return """
Extract the specific PMP eligibility and application details exactly as they appear in the answer. For each item, also extract the URLs (sources) the answer cites for that item. Return a JSON object matching the following schema:

Fields to extract (use exact text from the answer; if missing, set to null; for yes/no items, use 'yes' or 'no'):

1. educational_higher: The higher-education pathway (e.g., "four-year degree (bachelor's)").
   educational_higher_sources: Array of URLs cited for this item.

2. educational_alternative: The alternative educational pathway (e.g., "secondary school diploma", "high school diploma").
   educational_alternative_sources: Array of URLs.

3. exp_months_bachelor: Number of months of project management experience required if the candidate has a four-year degree. Prefer exact wording; e.g., "36 months".
   exp_months_bachelor_sources: Array of URLs.

4. exp_months_hs_diploma: Number of months of project management experience required with a high school/secondary school diploma.
   exp_months_hs_diploma_sources: Array of URLs.

5. exp_timeframe: The timeframe window within which the experience must have been obtained (e.g., "within the last 8 years").
   exp_timeframe_sources: Array of URLs.

6. exp_leading_directing: Whether the experience must specifically involve leading and directing projects, or if any project participation suffices. Use a concise phrase or 'yes'/'no' (where 'yes' means leading/directing is required).
   exp_leading_directing_sources: Array of URLs.

7. training_contact_hours: The number of contact hours of project management education/training required before taking the exam (e.g., "35 contact hours").
   training_contact_hours_sources: Array of URLs.

8. training_capm_substitute: Whether an active CAPM can substitute for the training contact hours requirement. Use 'yes' or 'no'.
   training_capm_substitute_sources: Array of URLs.

9. new_exam_launch_month: The specific month in 2026 when the new PMP exam version will be launched (e.g., "July 2026" or just "July").
   new_exam_launch_month_sources: Array of URLs.

10. current_exam_last_date: The last date to take the current version of the exam before the new version launches (e.g., "June 30, 2026").
    current_exam_last_date_sources: Array of URLs.

11. governing_org_full_name: The full name of the organization that administers and governs the PMP certification (e.g., "Project Management Institute (PMI)").
    governing_org_sources: Array of URLs.

12. application_platform: The official platform/system where candidates must submit their PMP application (e.g., "PMI.org Certification Dashboard" or "PMI Online Application").
    application_platform_sources: Array of URLs.

13. pmi_membership_required: Whether PMI membership is required to apply, or optional. Use 'required' or 'optional' if the answer uses those terms; otherwise 'yes' or 'no' ('no' meaning optional).
    pmi_membership_required_sources: Array of URLs.

14. exam_content_outline_2026_published: Whether PMI has published an official Examination Content Outline (ECO) specifically for the 2026 PMP exam. Use 'yes' or 'no'.
    exam_content_outline_sources: Array of URLs.

15. sme_review_pmp_questions: Whether PMP exam questions are reviewed by subject matter experts who hold valid PMP certification. Use 'yes' or 'no'.
    sme_review_sources: Array of URLs.

Rules for sources extraction:
- Extract only URLs explicitly present in the answer (plain or markdown links).
- Do not invent or infer URLs; if none are present, return an empty array.
- Include full URLs with protocol.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _urls_or_empty(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Basic cleanup: ensure strings and non-empty
    return [u.strip() for u in urls if isinstance(u, str) and u.strip()]


def _has_value(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _to_bool_from_yn(s: Optional[str]) -> Optional[bool]:
    if s is None:
        return None
    v = s.strip().lower()
    if v in {"yes", "true"}:
        return True
    if v in {"no", "false"}:
        return False
    if v in {"required"}:
        return True
    if v in {"optional"}:
        return False
    # Try to infer from phrasing for leading/directing
    if "lead" in v or "leading" in v or "direct" in v or "directing" in v or "manage" in v:
        return True
    if "participation" in v and "sufficient" in v:
        return False
    return None


def _add_presence_prereqs(
    evaluator: Evaluator,
    node_id_base: str,
    value_present: bool,
    sources_present: bool,
    parent_node=None
):
    vp_node = evaluator.add_custom_node(
        result=value_present,
        id=f"{node_id_base}_value_present",
        desc=f"Value for {node_id_base} is provided in the answer",
        parent=parent_node,
        critical=False
    )
    sp_node = evaluator.add_custom_node(
        result=sources_present,
        id=f"{node_id_base}_sources_present",
        desc=f"Reference URL(s) for {node_id_base} are provided in the answer",
        parent=parent_node,
        critical=False
    )
    return vp_node, sp_node


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extraction: PMPRequirementsExtraction
) -> None:
    root = evaluator.root

    # 1. Educational_Requirement_Bachelor (critical)
    edu_bach_value = extraction.educational_higher
    edu_bach_sources = _urls_or_empty(extraction.educational_higher_sources)
    vp1, sp1 = _add_presence_prereqs(
        evaluator, "Educational_Requirement_Bachelor", _has_value(edu_bach_value), bool(edu_bach_sources), root
    )
    leaf_edu_bach = evaluator.add_leaf(
        id="Educational_Requirement_Bachelor",
        desc="Verify that a four-year degree (bachelor's degree) is one of the acceptable educational qualifications for PMP certification",
        parent=root,
        critical=True
    )
    claim_edu_bach = (
        f"The higher-education pathway stated in the answer ('{edu_bach_value}') is accepted by PMI as an "
        f"eligible educational qualification for PMP candidates."
    )
    await evaluator.verify(
        claim=claim_edu_bach,
        node=leaf_edu_bach,
        sources=edu_bach_sources,
        additional_instruction=(
            f"As of {AS_OF_DATE}, check PMI's official eligibility criteria. Accept equivalent phrasing such as "
            f"'four-year degree', 'bachelor's degree', or 'global equivalent'."
        ),
        extra_prerequisites=[vp1, sp1]
    )

    # 2. Educational_Requirement_Alternative (critical)
    edu_alt_value = extraction.educational_alternative
    edu_alt_sources = _urls_or_empty(extraction.educitional_alternative_sources if hasattr(extraction, "educitional_alternative_sources") else extraction.educational_alternative_sources)
    vp2, sp2 = _add_presence_prereqs(
        evaluator, "Educational_Requirement_Alternative", _has_value(edu_alt_value), bool(edu_alt_sources), root
    )
    leaf_edu_alt = evaluator.add_leaf(
        id="Educational_Requirement_Alternative",
        desc="Verify that a high school diploma or secondary school diploma is accepted as an alternative educational qualification for PMP certification",
        parent=root,
        critical=True
    )
    claim_edu_alt = (
        f"The alternative educational pathway stated ('{edu_alt_value}') is accepted by PMI as an eligible "
        f"qualification for PMP (the pathway without a four-year degree)."
    )
    await evaluator.verify(
        claim=claim_edu_alt,
        node=leaf_edu_alt,
        sources=edu_alt_sources,
        additional_instruction=(
            f"As of {AS_OF_DATE}, look for PMI's wording such as 'secondary degree (high school diploma, "
            f"associate’s degree or global equivalent)'."
        ),
        extra_prerequisites=[vp2, sp2]
    )

    # 3. Experience_Months_Bachelor (critical)
    exp_bach_value = extraction.exp_months_bachelor
    exp_bach_sources = _urls_or_empty(extraction.exp_months_bachelor_sources)
    vp3, sp3 = _add_presence_prereqs(
        evaluator, "Experience_Months_Bachelor", _has_value(exp_bach_value), bool(exp_bach_sources), root
    )
    leaf_exp_bach = evaluator.add_leaf(
        id="Experience_Months_Bachelor",
        desc="Identify the exact number of months of project management experience required for candidates with a four-year degree",
        parent=root,
        critical=True
    )
    claim_exp_bach = (
        f"With a four-year degree, PMI requires {exp_bach_value} of project management experience for PMP eligibility "
        f"(allowing equivalences like '36 months' ≈ '3 years')."
    )
    await evaluator.verify(
        claim=claim_exp_bach,
        node=leaf_exp_bach,
        sources=exp_bach_sources,
        additional_instruction=(
            "Focus on the exact experience duration PMI requires for candidates who hold a four-year degree. "
            "Accept reasonable phrasing variants such as months vs. years."
        ),
        extra_prerequisites=[vp3, sp3]
    )

    # 4. Experience_Months_HS_Diploma (critical)
    exp_hs_value = extraction.exp_months_hs_diploma
    exp_hs_sources = _urls_or_empty(extraction.exp_months_hs_diploma_sources)
    vp4, sp4 = _add_presence_prereqs(
        evaluator, "Experience_Months_HS_Diploma", _has_value(exp_hs_value), bool(exp_hs_sources), root
    )
    leaf_exp_hs = evaluator.add_leaf(
        id="Experience_Months_HS_Diploma",
        desc="Identify the exact number of months of project management experience required for candidates with a high school diploma",
        parent=root,
        critical=True
    )
    claim_exp_hs = (
        f"With a secondary school (high school/associate) diploma, PMI requires {exp_hs_value} of project management "
        f"experience for PMP eligibility."
    )
    await evaluator.verify(
        claim=claim_exp_hs,
        node=leaf_exp_hs,
        sources=exp_hs_sources,
        additional_instruction="Check PMI's eligibility criteria for candidates without a four-year degree.",
        extra_prerequisites=[vp4, sp4]
    )

    # 5. Experience_Timeframe (critical)
    exp_timeframe_value = extraction.exp_timeframe
    exp_timeframe_sources = _urls_or_empty(extraction.exp_timeframe_sources)
    vp5, sp5 = _add_presence_prereqs(
        evaluator, "Experience_Timeframe", _has_value(exp_timeframe_value), bool(exp_timeframe_sources), root
    )
    leaf_exp_timeframe = evaluator.add_leaf(
        id="Experience_Timeframe",
        desc="Verify the timeframe within which the required project management experience must have been obtained (e.g., within the past X years)",
        parent=root,
        critical=True
    )
    claim_exp_timeframe = (
        f"PMP eligibility requires that the qualifying project management experience was obtained {exp_timeframe_value}."
    )
    await evaluator.verify(
        claim=claim_exp_timeframe,
        node=leaf_exp_timeframe,
        sources=exp_timeframe_sources,
        additional_instruction=(
            "Look for PMI's specified window (e.g., 'within the last 8 years'). Accept minor phrasing differences."
        ),
        extra_prerequisites=[vp5, sp5]
    )

    # 6. Training_Contact_Hours (critical)
    train_hours_value = extraction.training_contact_hours
    train_hours_sources = _urls_or_empty(extraction.training_contact_hours_sources)
    vp6, sp6 = _add_presence_prereqs(
        evaluator, "Training_Contact_Hours", _has_value(train_hours_value), bool(train_hours_sources), root
    )
    leaf_train_hours = evaluator.add_leaf(
        id="Training_Contact_Hours",
        desc="Identify the exact number of contact hours of project management education or training required before taking the PMP exam",
        parent=root,
        critical=True
    )
    claim_train_hours = (
        f"PMP eligibility requires {train_hours_value} of project management education/training before sitting for the exam."
    )
    await evaluator.verify(
        claim=claim_train_hours,
        node=leaf_train_hours,
        sources=train_hours_sources,
        additional_instruction="Accept equivalent labels such as 'contact hours' or 'hours of project management education'.",
        extra_prerequisites=[vp6, sp6]
    )

    # 7. Training_Alternative_CAPM (non-critical)
    capm_sub_value = extraction.training_capm_substitute
    capm_sub_bool = _to_bool_from_yn(capm_sub_value)
    capm_sub_sources = _urls_or_empty(extraction.training_capm_substitute_sources)
    vp7, sp7 = _add_presence_prereqs(
        evaluator, "Training_Alternative_CAPM", _has_value(capm_sub_value), bool(capm_sub_sources), root
    )
    leaf_capm_sub = evaluator.add_leaf(
        id="Training_Alternative_CAPM",
        desc="Verify whether holding an active CAPM (Certified Associate in Project Management) certification can substitute for the training requirement",
        parent=root,
        critical=False
    )
    if capm_sub_bool is True:
        claim_capm_sub = "An active CAPM certification can substitute for the PMP training (contact hours) requirement."
    elif capm_sub_bool is False:
        claim_capm_sub = "An active CAPM certification cannot substitute for the PMP training (contact hours) requirement."
    else:
        # Fall back to a neutral phrasing if we couldn't infer
        claim_capm_sub = (
            f"PMI's policy on whether an active CAPM substitutes for training is stated as '{capm_sub_value}'. Verify its correctness."
        )
    await evaluator.verify(
        claim=claim_capm_sub,
        node=leaf_capm_sub,
        sources=capm_sub_sources,
        additional_instruction=(
            f"As of {AS_OF_DATE}, check PMI's eligibility/training policy. The answer's statement must align with PMI's "
            "official stance; accept clear equivalences (e.g., 'CAPM satisfies/waives the 35 hours')."
        ),
        extra_prerequisites=[vp7, sp7]
    )

    # 8. New_Exam_Launch_Month (critical)
    launch_month_value = extraction.new_exam_launch_month
    launch_month_sources = _urls_or_empty(extraction.new_exam_launch_month_sources)
    vp8, sp8 = _add_presence_prereqs(
        evaluator, "New_Exam_Launch_Month", _has_value(launch_month_value), bool(launch_month_sources), root
    )
    leaf_launch_month = evaluator.add_leaf(
        id="New_Exam_Launch_Month",
        desc="Identify the specific month in 2026 when the new version of the PMP exam will be launched",
        parent=root,
        critical=True
    )
    claim_launch_month = f"The new version of the PMP exam will be launched in {launch_month_value} 2026."
    await evaluator.verify(
        claim=claim_launch_month,
        node=leaf_launch_month,
        sources=launch_month_sources,
        additional_instruction=f"Verify against PMI's official announcements as of {AS_OF_DATE}.",
        extra_prerequisites=[vp8, sp8]
    )

    # 9. Current_Exam_Deadline (critical)
    deadline_value = extraction.current_exam_last_date
    deadline_sources = _urls_or_empty(extraction.current_exam_last_date_sources)
    vp9, sp9 = _add_presence_prereqs(
        evaluator, "Current_Exam_Deadline", _has_value(deadline_value), bool(deadline_sources), root
    )
    leaf_deadline = evaluator.add_leaf(
        id="Current_Exam_Deadline",
        desc="Determine the last date to take the current version of the PMP exam before the new version launches",
        parent=root,
        critical=True
    )
    claim_deadline = f"The last date to take the current version of the PMP exam before the new version launches is {deadline_value}."
    await evaluator.verify(
        claim=claim_deadline,
        node=leaf_deadline,
        sources=deadline_sources,
        additional_instruction="Verify against PMI's official transition schedule or announcements.",
        extra_prerequisites=[vp9, sp9]
    )

    # 10. PMP_Governing_Organization (critical)
    gov_org_value = extraction.governing_org_full_name
    gov_org_sources = _urls_or_empty(extraction.governing_org_sources)
    vp10, sp10 = _add_presence_prereqs(
        evaluator, "PMP_Governing_Organization", _has_value(gov_org_value), bool(gov_org_sources), root
    )
    leaf_gov_org = evaluator.add_leaf(
        id="PMP_Governing_Organization",
        desc="Identify the full name of the organization that administers and governs the PMP certification",
        parent=root,
        critical=True
    )
    claim_gov_org = f"The organization that administers and governs the PMP certification is '{gov_org_value}'."
    await evaluator.verify(
        claim=claim_gov_org,
        node=leaf_gov_org,
        sources=gov_org_sources,
        additional_instruction="Accept 'Project Management Institute (PMI)' as full/proper name.",
        extra_prerequisites=[vp10, sp10]
    )

    # 11. PMP_Application_Platform (non-critical)
    app_platform_value = extraction.application_platform
    app_platform_sources = _urls_or_empty(extraction.application_platform_sources)
    vp11, sp11 = _add_presence_prereqs(
        evaluator, "PMP_Application_Platform", _has_value(app_platform_value), bool(app_platform_sources), root
    )
    leaf_app_platform = evaluator.add_leaf(
        id="PMP_Application_Platform",
        desc="Identify the official platform or system where candidates must submit their PMP certification application",
        parent=root,
        critical=False
    )
    claim_app_platform = f"PMP certification applications must be submitted via '{app_platform_value}'."
    await evaluator.verify(
        claim=claim_app_platform,
        node=leaf_app_platform,
        sources=app_platform_sources,
        additional_instruction="Accept equivalents such as 'PMI.org Certification Dashboard' or 'PMI online application system'.",
        extra_prerequisites=[vp11, sp11]
    )

    # 12. Experience_Leading_Projects (critical)
    lead_dir_value = extraction.exp_leading_directing
    lead_dir_bool = _to_bool_from_yn(lead_dir_value)
    lead_dir_sources = _urls_or_empty(extraction.exp_leading_directing_sources)
    vp12, sp12 = _add_presence_prereqs(
        evaluator, "Experience_Leading_Projects", _has_value(lead_dir_value), bool(lead_dir_sources), root
    )
    leaf_lead_dir = evaluator.add_leaf(
        id="Experience_Leading_Projects",
        desc="Verify that the required experience specifically involves leading and directing projects (not just participating in projects)",
        parent=root,
        critical=True
    )
    if lead_dir_bool is True:
        claim_lead_dir = "PMP eligibility requires experience that specifically involves leading and directing projects (not just participation)."
    elif lead_dir_bool is False:
        claim_lead_dir = "PMP eligibility does not strictly require leading and directing projects; any project participation experience is sufficient."
    else:
        claim_lead_dir = (
            f"The answer states experience requirement as '{lead_dir_value}'. Verify whether PMI requires leading/directing."
        )
    await evaluator.verify(
        claim=claim_lead_dir,
        node=leaf_lead_dir,
        sources=lead_dir_sources,
        additional_instruction=(
            "Look for PMI's phrasing like 'leading and directing the work of the project'. "
            "If the answer claims otherwise, verify accordingly."
        ),
        extra_prerequisites=[vp12, sp12]
    )

    # 13. PMI_Membership_Required (non-critical)
    membership_value = extraction.pmi_membership_required
    membership_bool = _to_bool_from_yn(membership_value)
    membership_sources = _urls_or_empty(extraction.pmi_membership_required_sources)
    vp13, sp13 = _add_presence_prereqs(
        evaluator, "PMI_Membership_Required", _has_value(membership_value), bool(membership_sources), root
    )
    leaf_membership = evaluator.add_leaf(
        id="PMI_Membership_Required",
        desc="Determine whether PMI (Project Management Institute) membership is required to apply for PMP certification, or if it is optional",
        parent=root,
        critical=False
    )
    if membership_bool is True:
        claim_membership = "PMI membership is required to apply for PMP certification."
    elif membership_bool is False:
        claim_membership = "PMI membership is optional (not required) to apply for PMP certification."
    else:
        claim_membership = f"Verify the statement in the answer about membership requirement: '{membership_value}'."
    await evaluator.verify(
        claim=claim_membership,
        node=leaf_membership,
        sources=membership_sources,
        additional_instruction="Check PMI's application prerequisites. Often membership is optional but may reduce fees.",
        extra_prerequisites=[vp13, sp13]
    )

    # 14. Exam_Content_Outline_Availability (non-critical)
    eco_value = extraction.exam_content_outline_2026_published
    eco_bool = _to_bool_from_yn(eco_value)
    eco_sources = _urls_or_empty(extraction.exam_content_outline_sources)
    vp14, sp14 = _add_presence_prereqs(
        evaluator, "Exam_Content_Outline_Availability", _has_value(eco_value), bool(eco_sources), root
    )
    leaf_eco = evaluator.add_leaf(
        id="Exam_Content_Outline_Availability",
        desc="Verify that PMI has published an official examination content outline document for the 2026 PMP exam",
        parent=root,
        critical=False
    )
    if eco_bool is True:
        claim_eco = "PMI has published an official Examination Content Outline (ECO) specifically for the 2026 PMP exam."
    elif eco_bool is False:
        claim_eco = "PMI has not published an official Examination Content Outline (ECO) specifically for the 2026 PMP exam."
    else:
        claim_eco = f"Verify the availability claim about the 2026 ECO stated as '{eco_value}'."
    await evaluator.verify(
        claim=claim_eco,
        node=leaf_eco,
        sources=eco_sources,
        additional_instruction=f"As of {AS_OF_DATE}, verify PMI's official ECO page or documentation.",
        extra_prerequisites=[vp14, sp14]
    )

    # 15. Subject_Matter_Expert_Review (non-critical)
    sme_value = extraction.sme_review_pmp_questions
    sme_bool = _to_bool_from_yn(sme_value)
    sme_sources = _urls_or_empty(extraction.sme_review_sources)
    vp15, sp15 = _add_presence_prereqs(
        evaluator, "Subject_Matter_Expert_Review", _has_value(sme_value), bool(sme_sources), root
    )
    leaf_sme = evaluator.add_leaf(
        id="Subject_Matter_Expert_Review",
        desc="Verify that PMP exam questions are reviewed by subject matter experts (SMEs) who hold valid PMP certification",
        parent=root,
        critical=False
    )
    if sme_bool is True:
        claim_sme = "PMP exam questions are reviewed by subject matter experts who themselves hold valid PMP certification."
    elif sme_bool is False:
        claim_sme = "PMP exam questions are not necessarily reviewed by SMEs who hold PMP certification."
    else:
        claim_sme = f"Verify the SME review policy stated as '{sme_value}'."
    await evaluator.verify(
        claim=claim_sme,
        node=leaf_sme,
        sources=sme_sources,
        additional_instruction="Check PMI's exam development and quality assurance statements.",
        extra_prerequisites=[vp15, sp15]
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
    Evaluate an answer for the PMP certification requirements and key details as of February 2026.
    """
    # Initialize evaluator with parallel aggregation at root
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

    # Extract structured info
    extraction = await evaluator.extract(
        prompt=prompt_extract_pmp_requirements(),
        template_class=PMPRequirementsExtraction,
        extraction_name="pmp_requirements_extraction"
    )

    # Record as-of date for context
    evaluator.add_custom_info({"as_of_date": AS_OF_DATE}, info_type="meta", info_name="evaluation_context")

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extraction)

    # Return structured summary
    return evaluator.get_summary()