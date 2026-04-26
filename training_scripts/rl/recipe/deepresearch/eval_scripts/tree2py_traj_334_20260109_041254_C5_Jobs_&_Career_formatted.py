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
TASK_ID = "professional_credentials_requirements"
TASK_DESCRIPTION = """
I am creating a comprehensive reference guide for professional licensing and certification requirements across different career fields. Please provide the specific requirements for the following four professional credentials:

1. Certified Financial Planner (CFP) - Standard Pathway:
   - How many hours of professional experience are required related to financial planning?
   - What is the minimum education degree required?

2. Licensed Clinical Social Worker (LCSW) in California:
   - How many total supervised hours are required?
   - What is the minimum number of weeks over which the supervised hours must be completed?
   - How many of the total hours must specifically be gained under the supervision of a licensed clinical social worker (LCSW)?

3. Project Management Professional (PMP) - For Bachelor's Degree Holders:
   - How many months of project management experience are required?
   - How many hours of project management education or training are required?

4. Texas Licensed Physician - Continuing Medical Education:
   - How many continuing medical education (CME) credits must be completed every 24 months?
   - How many of these credits must be formal AMA PRA Category 1 credits?

For each requirement, provide the specific numerical value or qualification and include a reference URL that verifies this information.
""".strip()


# Ground truth expectations for reference (recorded in summary)
GROUND_TRUTH = {
    "CFP": {
        "experience_hours": "6000",
        "education_degree": "bachelor's degree or higher"
    },
    "LCSW_CA": {
        "total_hours": "3000",
        "minimum_weeks": "104",
        "lcsw_supervision_hours": "1700"
    },
    "PMP_Bachelors": {
        "experience_months": "36",
        "education_hours": "35"
    },
    "Texas_Physician_CME": {
        "cme_total_24mo": "48",
        "cme_category1": "24"
    }
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FieldWithSources(BaseModel):
    value: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CFPRequirements(BaseModel):
    experience_hours: Optional[FieldWithSources] = None
    education_degree: Optional[FieldWithSources] = None


class LCSWRequirements(BaseModel):
    total_supervised_hours: Optional[FieldWithSources] = None
    minimum_weeks: Optional[FieldWithSources] = None
    lcsw_supervision_hours: Optional[FieldWithSources] = None


class PMPRequirements(BaseModel):
    experience_months: Optional[FieldWithSources] = None
    education_hours: Optional[FieldWithSources] = None


class TexasPhysicianCME(BaseModel):
    cme_total_24mo: Optional[FieldWithSources] = None
    cme_category1: Optional[FieldWithSources] = None


class CredentialRequirementsExtraction(BaseModel):
    cfp: Optional[CFPRequirements] = None
    lcsw_ca: Optional[LCSWRequirements] = None
    pmp_bachelors: Optional[PMPRequirements] = None
    texas_physician_cme: Optional[TexasPhysicianCME] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
    Extract the specific requirement values and their verifying reference URL(s) for the four credentials listed below, exactly as they appear in the provided answer text. Return all numeric values as strings (e.g., "6000", "36", "35"). If a value is not stated, set `value` to null. For URLs, extract only actual URLs explicitly present in the answer (including markdown links). If none are present for a requirement, return an empty list for `urls`.

    Produce a single JSON object with the following structure:
    {
      "cfp": {
        "experience_hours": { "value": string|null, "urls": [string, ...] },
        "education_degree": { "value": string|null, "urls": [string, ...] }
      },
      "lcsw_ca": {
        "total_supervised_hours": { "value": string|null, "urls": [string, ...] },
        "minimum_weeks": { "value": string|null, "urls": [string, ...] },
        "lcsw_supervision_hours": { "value": string|null, "urls": [string, ...] }
      },
      "pmp_bachelors": {
        "experience_months": { "value": string|null, "urls": [string, ...] },
        "education_hours": { "value": string|null, "urls": [string, ...] }
      },
      "texas_physician_cme": {
        "cme_total_24mo": { "value": string|null, "urls": [string, ...] },
        "cme_category1": { "value": string|null, "urls": [string, ...] }
      }
    }

    Mapping of each required field:
    - CFP (standard pathway):
      • experience_hours: the number of hours of professional experience related to financial planning.
      • education_degree: the minimum education degree requirement (e.g., "bachelor's degree or higher").
    - LCSW in California:
      • total_supervised_hours: total supervised hours required.
      • minimum_weeks: minimum number of weeks over which those hours must be completed.
      • lcsw_supervision_hours: number of hours that must be gained under the supervision of an LCSW.
    - PMP (for bachelor's degree holders):
      • experience_months: required months of project management experience (for 4-year degree holders).
      • education_hours: required hours of project management education/training.
    - Texas Licensed Physician (CME):
      • cme_total_24mo: total CME credits every 24 months.
      • cme_category1: required formal AMA PRA Category 1 credits.

    Rules:
    - Do not infer or create values or URLs; only extract what the answer explicitly states.
    - If URLs are missing for a requirement, return an empty list for that requirement's `urls`.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _urls_or_empty(field: Optional[FieldWithSources]) -> List[str]:
    if field and field.urls:
        return [u for u in field.urls if isinstance(u, str) and u.strip()]
    return []


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_cfp_nodes(evaluator: Evaluator, parent_node, data: Optional[CFPRequirements]) -> None:
    group = evaluator.add_parallel(
        id="CFP_Requirements",
        desc="Certified Financial Planner (CFP) requirements accurately documented",
        parent=parent_node,
        critical=False
    )

    # Experience Hours - URL existence (critical)
    exp_urls = _urls_or_empty(data.experience_hours) if data else []
    evaluator.add_custom_node(
        result=len(exp_urls) > 0,
        id="CFP_Experience_Reference_URL",
        desc="A reference URL is provided that verifies the CFP experience-hours requirement",
        parent=group,
        critical=True
    )

    # Experience Hours - canonical requirement (critical)
    exp_claim_node = evaluator.add_leaf(
        id="CFP_Experience_Hours",
        desc="CFP certification standard pathway requires 6,000 hours of professional experience related to financial planning",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="The CFP certification standard pathway requires 6,000 hours of professional experience related to financial planning.",
        node=exp_claim_node,
        sources=exp_urls,
        additional_instruction=(
            "Judge strictly based on the cited URL(s). Accept statements that say '6,000 hours of professional "
            "experience' or clearly equivalent wording. If a page only describes the 4,000-hour Apprenticeship Pathway, "
            "that does NOT satisfy this 'standard pathway' requirement."
        )
    )

    # Education Degree - URL existence (critical)
    edu_urls = _urls_or_empty(data.education_degree) if data else []
    evaluator.add_custom_node(
        result=len(edu_urls) > 0,
        id="CFP_Education_Reference_URL",
        desc="A reference URL is provided that verifies the CFP education-degree requirement",
        parent=group,
        critical=True
    )

    # Education Degree - canonical requirement (critical)
    edu_claim_node = evaluator.add_leaf(
        id="CFP_Education_Degree",
        desc="CFP certification requires a bachelor's degree (or higher) from an accredited institution",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="The CFP certification requires a bachelor's degree or higher from an accredited college or university.",
        node=edu_claim_node,
        sources=edu_urls,
        additional_instruction=(
            "Use only the provided URL(s). Accept equivalent phrasings like 'baccalaureate degree (or higher)' "
            "or 'degree from an accredited college or university'."
        )
    )


async def build_lcsw_nodes(evaluator: Evaluator, parent_node, data: Optional[LCSWRequirements]) -> None:
    group = evaluator.add_parallel(
        id="LCSW_Requirements",
        desc="California Licensed Clinical Social Worker (LCSW) requirements accurately documented",
        parent=parent_node,
        critical=False
    )

    # Total supervised hours
    total_urls = _urls_or_empty(data.total_supervised_hours) if data else []
    evaluator.add_custom_node(
        result=len(total_urls) > 0,
        id="LCSW_Total_Hours_Reference_URL",
        desc="A reference URL is provided that verifies the California LCSW total supervised-hours requirement",
        parent=group,
        critical=True
    )
    total_claim_node = evaluator.add_leaf(
        id="LCSW_Total_Hours",
        desc="California LCSW requires 3,000 total supervised hours",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="In California, the LCSW licensure requires a total of 3,000 supervised hours.",
        node=total_claim_node,
        sources=total_urls,
        additional_instruction=(
            "Verify specifically for California (BBS). Accept phrasing like '3,000 hours of supervised experience' "
            "or equivalent."
        )
    )

    # Minimum weeks
    weeks_urls = _urls_or_empty(data.minimum_weeks) if data else []
    evaluator.add_custom_node(
        result=len(weeks_urls) > 0,
        id="LCSW_Minimum_Weeks_Reference_URL",
        desc="A reference URL is provided that verifies the California LCSW minimum-weeks requirement",
        parent=group,
        critical=True
    )
    weeks_claim_node = evaluator.add_leaf(
        id="LCSW_Minimum_Weeks",
        desc="California LCSW supervised hours must be completed over a minimum of 104 weeks",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="In California, LCSW supervised experience must be accrued over a minimum of 104 weeks.",
        node=weeks_claim_node,
        sources=weeks_urls,
        additional_instruction="Confirm this 104-week minimum specifically for California (BBS)."
    )

    # LCSW-supervised hours
    sup_urls = _urls_or_empty(data.lcsw_supervision_hours) if data else []
    evaluator.add_custom_node(
        result=len(sup_urls) > 0,
        id="LCSW_Supervision_Type_Reference_URL",
        desc="A reference URL is provided that verifies the California LCSW LCSW-supervision-hours requirement",
        parent=group,
        critical=True
    )
    sup_claim_node = evaluator.add_leaf(
        id="LCSW_Supervision_Type",
        desc="California LCSW requires at least 1,700 hours under the supervision of a licensed clinical social worker (LCSW)",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="In California, at least 1,700 of the supervised hours must be under the supervision of a licensed clinical social worker (LCSW).",
        node=sup_claim_node,
        sources=sup_urls,
        additional_instruction="Confirm this specific 1,700-hour LCSW-supervision requirement for California."
    )


async def build_pmp_nodes(evaluator: Evaluator, parent_node, data: Optional[PMPRequirements]) -> None:
    group = evaluator.add_parallel(
        id="PMP_Requirements",
        desc="Project Management Professional (PMP) requirements for bachelor's degree holders accurately documented",
        parent=parent_node,
        critical=False
    )

    # Experience months
    exp_urls = _urls_or_empty(data.experience_months) if data else []
    evaluator.add_custom_node(
        result=len(exp_urls) > 0,
        id="PMP_Experience_Reference_URL",
        desc="A reference URL is provided that verifies the PMP project-management-experience requirement",
        parent=group,
        critical=True
    )
    exp_claim_node = evaluator.add_leaf(
        id="PMP_Experience_Months",
        desc="PMP certification requires 36 months of project management experience for candidates with a four-year degree",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="For candidates who hold a four-year degree (bachelor's or global equivalent), the PMP certification requires 36 months of project management experience.",
        node=exp_claim_node,
        sources=exp_urls,
        additional_instruction="Accept equivalent phrasing such as 'three years' experience' for bachelor's degree holders."
    )

    # Education hours
    edu_urls = _urls_or_empty(data.education_hours) if data else []
    evaluator.add_custom_node(
        result=len(edu_urls) > 0,
        id="PMP_Education_Reference_URL",
        desc="A reference URL is provided that verifies the PMP project-management-education-hours requirement",
        parent=group,
        critical=True
    )
    edu_claim_node = evaluator.add_leaf(
        id="PMP_Education_Hours",
        desc="PMP certification requires 35 hours of project management education or training",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="The PMP certification requires 35 hours of project management education or training.",
        node=edu_claim_node,
        sources=edu_urls,
        additional_instruction="Accept equivalent phrasing like '35 contact hours' of project management education."
    )


async def build_texas_physician_nodes(evaluator: Evaluator, parent_node, data: Optional[TexasPhysicianCME]) -> None:
    group = evaluator.add_parallel(
        id="Texas_Physician_Requirements",
        desc="Texas licensed physician continuing medical education requirements accurately documented",
        parent=parent_node,
        critical=False
    )

    # Total CME credits
    total_urls = _urls_or_empty(data.cme_total_24mo) if data else []
    evaluator.add_custom_node(
        result=len(total_urls) > 0,
        id="Texas_CME_Total_Reference_URL",
        desc="A reference URL is provided that verifies the Texas total CME-credits-per-24-months requirement",
        parent=group,
        critical=True
    )
    total_claim_node = evaluator.add_leaf(
        id="Texas_CME_Total",
        desc="Texas physicians must complete 48 continuing medical education (CME) credits every 24 months",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="Texas physicians must complete 48 continuing medical education (CME) credits every 24 months.",
        node=total_claim_node,
        sources=total_urls,
        additional_instruction="Verify with Texas Medical Board or other authoritative Texas sources."
    )

    # Category 1 credits
    cat1_urls = _urls_or_empty(data.cme_category1) if data else []
    evaluator.add_custom_node(
        result=len(cat1_urls) > 0,
        id="Texas_CME_Category1_Reference_URL",
        desc="A reference URL is provided that verifies the Texas AMA PRA Category 1 CME-credits requirement",
        parent=group,
        critical=True
    )
    cat1_claim_node = evaluator.add_leaf(
        id="Texas_CME_Category1",
        desc="Texas physicians must complete at least 24 formal AMA PRA Category 1 CME credits",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim="Of the total CME credits, Texas physicians must complete at least 24 formal AMA PRA Category 1 Credits.",
        node=cat1_claim_node,
        sources=cat1_urls,
        additional_instruction="Verify specifically for Texas state CME requirements."
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
    Evaluate an answer against the professional credential requirements rubric.
    """
    # Initialize evaluator (root is parallel as per rubric)
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

    # Extract structured data from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=CredentialRequirementsExtraction,
        extraction_name="credential_requirements_extraction"
    )

    # Record ground truth expectations for transparency
    evaluator.add_ground_truth(
        gt_info=GROUND_TRUTH,
        gt_type="expected_requirements"
    )

    # Build subtrees for each credential in parallel fashion
    await build_cfp_nodes(evaluator, root, extraction.cfp)
    await build_lcsw_nodes(evaluator, root, extraction.lcsw_ca)
    await build_pmp_nodes(evaluator, root, extraction.pmp_bachelors)
    await build_texas_physician_nodes(evaluator, root, extraction.texas_physician_cme)

    # Return structured summary
    return evaluator.get_summary()