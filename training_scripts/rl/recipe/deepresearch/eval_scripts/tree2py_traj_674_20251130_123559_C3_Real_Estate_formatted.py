import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "calhfa_myhome_eligibility_2025"
TASK_DESCRIPTION = (
    "Research and document the complete eligibility requirements for a first-time homebuyer to qualify "
    "for the CalHFA MyHome Assistance Program in California in 2025. Your documentation must include: "
    "(1) The specific definition of 'first-time homebuyer' as defined by CalHFA, including the exact time-period "
    "requirement for prior homeownership and any spouse-related exclusions. "
    "(2) The minimum credit score requirements for the MyHome Assistance Program. "
    "(3) The mandatory homebuyer education and counseling requirements, including all acceptable course provider options "
    "and any certificate requirements. For each of these three requirement categories, provide the specific details and "
    "include a reference URL from CalHFA's official website or another authoritative California housing authority source "
    "that documents that requirement."
)

# Ground truth expectations used for context logging
GROUND_TRUTH = {
    "scope": {
        "program": "CalHFA MyHome Assistance Program",
        "year": "2025"
    },
    "first_time_homebuyer": {
        "three_year_no_own_and_occupy": True,
        "spouse_ownership_exclusion": True
    },
    "credit_score": {
        "government_min": 640,
        "conventional_min_range": [660, 680]
    },
    "homebuyer_education": {
        "one_occupying_borrower_required": True,
        "certificate_required": True,
        "online_option": "eHomeAmerica eight-hour course + mandatory 1-on-1 counseling",
        "live_option": "NeighborWorks America or HUD-Approved Housing Counseling Agency"
    }
}

# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_authoritative_ca_url(url: str) -> bool:
    """
    Determine whether a URL is from CalHFA or another authoritative California housing authority source.
    Accepted:
      - calhfa.ca.gov
      - Any subdomain ending with .ca.gov (California government domains)
    """
    try:
        parsed = urlparse(url if (url.startswith("http://") or url.startswith("https://")) else f"http://{url}")
        host = (parsed.netloc or "").lower()
        return (
            host.endswith(".ca.gov")
            or host == "calhfa.ca.gov"
            or host.endswith(".calhfa.ca.gov")
        )
    except Exception:
        return False


def filter_authoritative_urls(urls: List[str]) -> List[str]:
    """Return only authoritative CA/CalHFA URLs."""
    return [u for u in urls if is_authoritative_ca_url(u)]


# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class ScopeExtraction(BaseModel):
    mentions_calhfa_myhome: Optional[bool] = None
    mentions_2025: Optional[bool] = None


class FirstTimeHomebuyerExtraction(BaseModel):
    includes_three_year_no_own_occupy: Optional[bool] = None
    includes_spouse_owned_occupancy_exclusion: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class CreditScoreExtraction(BaseModel):
    includes_min_640_government: Optional[bool] = None
    includes_min_660_to_680_conventional: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class EducationExtraction(BaseModel):
    includes_one_occupying_borrower_required: Optional[bool] = None
    includes_certificate_required: Optional[bool] = None
    includes_online_ehome_8hr_1on1: Optional[bool] = None
    includes_neighborworks_or_hud_live_option: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class ProgramRequirementsExtraction(BaseModel):
    scope: Optional[ScopeExtraction] = None
    first_time_homebuyer: Optional[FirstTimeHomebuyerExtraction] = None
    credit_score: Optional[CreditScoreExtraction] = None
    homebuyer_education: Optional[EducationExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
    Extract structured information from the answer for the CalHFA MyHome Assistance Program (2025).
    Return a JSON object with the following structure and rules:

    1) scope:
       - mentions_calhfa_myhome: boolean
         True if the answer specifically mentions CalHFA MyHome Assistance Program (not generic programs).
       - mentions_2025: boolean
         True if the answer frames the requirements as applicable in 2025 (explicit mention of 2025).

    2) first_time_homebuyer:
       - includes_three_year_no_own_occupy: boolean
         True if the answer states that an applicant must NOT have owned AND occupied a home in the last three (3) years.
       - includes_spouse_owned_occupancy_exclusion: boolean
         True if the answer states that an applicant must NOT have lived in a home owned by a spouse in the last three (3) years.
       - sources: array of URLs
         All URLs cited in the answer that document the first-time homebuyer definition (CalHFA or authoritative CA agency).
         If no URLs are given, return an empty array.

    3) credit_score:
       - includes_min_640_government: boolean
         True if the answer states minimum credit score 640 for CalHFA government loan programs (FHA, VA, USDA).
       - includes_min_660_to_680_conventional: boolean
         True if the answer states minimum credit score range 660–680 for CalHFA conventional programs.
       - sources: array of URLs
         All URLs cited in the answer that document credit score requirements (CalHFA or authoritative CA agency).
         If no URLs are given, return an empty array.

    4) homebuyer_education:
       - includes_one_occupying_borrower_required: boolean
         True if the answer states that only one occupying first-time borrower per loan must complete the course.
       - includes_certificate_required: boolean
         True if the answer states that a certificate of completion is required.
       - includes_online_ehome_8hr_1on1: boolean
         True if the answer identifies an approved online option as eHome's 8-hour course with mandatory 1-on-1 counseling follow-up.
       - includes_neighborworks_or_hud_live_option: boolean
         True if the answer identifies an in-person/virtual option as live counseling via NeighborWorks America or a HUD-Approved Housing Counseling Agency.
       - sources: array of URLs
         All URLs cited in the answer that document education/counseling requirements (CalHFA or authoritative CA agency).
         If no URLs are given, return an empty array.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_scope(evaluator: Evaluator, parent_node, extraction: ProgramRequirementsExtraction) -> None:
    scope_node = evaluator.add_parallel(
        id="Scope_Requirements",
        desc="Answer scope matches the asked program and year.",
        parent=parent_node,
        critical=True
    )

    # Program specificity leaf (critical)
    program_specific_node = evaluator.add_leaf(
        id="Program_Specificity",
        desc="Documentation is specific to the CalHFA MyHome Assistance Program (not generic first-time homebuyer programs).",
        parent=scope_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer is specifically about the CalHFA MyHome Assistance Program (not generic first-time homebuyer programs).",
        node=program_specific_node,
        additional_instruction="Judge based on the answer text: it should explicitly reference 'CalHFA' and 'MyHome Assistance Program'. If the answer mixes multiple programs or remains generic, mark incorrect."
    )

    # Year applicability leaf (critical)
    year_2025_node = evaluator.add_leaf(
        id="Year_Applicability_2025",
        desc="Documentation is presented as applicable in 2025 (explicitly frames the requirements for 2025).",
        parent=scope_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer frames the requirements as applicable in 2025 (explicit mention or clear framing for 2025).",
        node=year_2025_node,
        additional_instruction="Check the answer for an explicit mention of 2025 or a clear framing that the requirements provided apply to year 2025."
    )


async def verify_first_time_homebuyer(evaluator: Evaluator, parent_node, extraction: ProgramRequirementsExtraction) -> None:
    ft_node = evaluator.add_parallel(
        id="First_Time_Homebuyer_Definition",
        desc="Provide CalHFA's definition of 'first-time homebuyer' with required details and an authoritative URL.",
        parent=parent_node,
        critical=True
    )

    ftb = extraction.first_time_homebuyer or FirstTimeHomebuyerExtraction()
    auth_urls = filter_authoritative_urls(ftb.sources)

    # Reference URL presence (critical custom)
    evaluator.add_custom_node(
        result=(len(auth_urls) > 0),
        id="Reference_URL_FTB",
        desc="Includes at least one reference URL from CalHFA's official website or another authoritative California housing authority source documenting the first-time homebuyer definition.",
        parent=ft_node,
        critical=True
    )

    # Three-year no own AND occupy (critical leaf)
    three_year_node = evaluator.add_leaf(
        id="Three_Year_No_Own_And_Occupy",
        desc="Specifies that the applicant has not owned and occupied a home in the last three years.",
        parent=ft_node,
        critical=True
    )
    await evaluator.verify(
        claim="CalHFA defines a 'first-time homebuyer' as someone who has not owned AND occupied a home in the last three (3) years.",
        node=three_year_node,
        sources=auth_urls,
        additional_instruction="Using the authoritative CA/CalHFA URLs, verify that the three-year non-ownership AND occupancy requirement is explicitly stated."
    )

    # Spouse ownership exclusion (critical leaf)
    spouse_excl_node = evaluator.add_leaf(
        id="Spouse_Ownership_Exclusion",
        desc="Specifies that the applicant has not lived in a home owned by a spouse in the last three years.",
        parent=ft_node,
        critical=True
    )
    await evaluator.verify(
        claim="CalHFA's first-time homebuyer definition includes that the applicant must not have lived in a home owned by their spouse within the last three (3) years.",
        node=spouse_excl_node,
        sources=auth_urls,
        additional_instruction="Verify from authoritative CA/CalHFA sources that the spouse-owned occupancy exclusion is part of the first-time homebuyer definition."
    )


async def verify_credit_score(evaluator: Evaluator, parent_node, extraction: ProgramRequirementsExtraction) -> None:
    credit_node = evaluator.add_parallel(
        id="Credit_Score_Requirements",
        desc="Provide minimum credit score requirements for MyHome/CalHFA programs with an authoritative URL.",
        parent=parent_node,
        critical=True
    )

    credit = extraction.credit_score or CreditScoreExtraction()
    auth_urls = filter_authoritative_urls(credit.sources)

    # Reference URL presence (critical custom)
    evaluator.add_custom_node(
        result=(len(auth_urls) > 0),
        id="Reference_URL_Credit",
        desc="Includes at least one reference URL from CalHFA's official website or another authoritative California housing authority source documenting credit score requirements.",
        parent=credit_node,
        critical=True
    )

    # Government programs min 640
    gov_min_node = evaluator.add_leaf(
        id="Government_Program_Min_640",
        desc="Specifies minimum credit score of 640 for CalHFA government loan programs (FHA, VA, USDA).",
        parent=credit_node,
        critical=True
    )
    await evaluator.verify(
        claim="The minimum credit score for CalHFA government loan programs (FHA, VA, USDA) is 640.",
        node=gov_min_node,
        sources=auth_urls,
        additional_instruction="Confirm from CalHFA or CA government pages that government programs require at least a 640 FICO score."
    )

    # Conventional programs min 660–680
    conv_min_node = evaluator.add_leaf(
        id="Conventional_Program_Min_660_to_680",
        desc="Specifies minimum credit score ranging from 660 to 680 for CalHFA conventional programs.",
        parent=credit_node,
        critical=True
    )
    await evaluator.verify(
        claim="CalHFA conventional programs require a minimum credit score between 660 and 680 depending on the specific program.",
        node=conv_min_node,
        sources=auth_urls,
        additional_instruction="Confirm from CalHFA or CA government pages that conventional programs set minimum FICO thresholds of 660 or 680 (program dependent). Minor wording variation is acceptable."
    )


async def verify_homebuyer_education(evaluator: Evaluator, parent_node, extraction: ProgramRequirementsExtraction) -> None:
    edu_node = evaluator.add_parallel(
        id="Homebuyer_Education_Requirements",
        desc="Provide mandatory homebuyer education/counseling requirements, acceptable provider options, certificate requirements, and an authoritative URL.",
        parent=parent_node,
        critical=True
    )

    edu = extraction.homebuyer_education or EducationExtraction()
    auth_urls = filter_authoritative_urls(edu.sources)

    # Reference URL presence (critical custom)
    evaluator.add_custom_node(
        result=(len(auth_urls) > 0),
        id="Reference_URL_Education",
        desc="Includes at least one reference URL from CalHFA's official website or another authoritative California housing authority source documenting homebuyer education requirements.",
        parent=edu_node,
        critical=True
    )

    # Only one occupying first-time borrower per loan must complete the course
    one_borrower_node = evaluator.add_leaf(
        id="Only_One_Occupying_First_Time_Borrower",
        desc="Specifies that only one occupying first-time borrower per loan transaction must complete the course.",
        parent=edu_node,
        critical=True
    )
    await evaluator.verify(
        claim="Only one occupying first-time borrower per loan transaction must complete the required homebuyer education course.",
        node=one_borrower_node,
        sources=auth_urls,
        additional_instruction="Verify the CalHFA/CA authority rule stating that only one occupying first-time borrower needs to complete the education course."
    )

    # Certificate of completion required
    certificate_node = evaluator.add_leaf(
        id="Certificate_Of_Completion_Required",
        desc="Specifies that a certificate of completion is required.",
        parent=edu_node,
        critical=True
    )
    await evaluator.verify(
        claim="A certificate of completion is required for the homebuyer education/counseling.",
        node=certificate_node,
        sources=auth_urls,
        additional_instruction="Find language on CalHFA/CA authority sources indicating the certificate requirement."
    )

    # Online option eHome 8hr + 1-on-1
    online_ehome_node = evaluator.add_leaf(
        id="Online_Option_eHome_8hr_plus_1on1",
        desc="Identifies an online option as eHome's eight-hour course with mandatory 1-on-1 counseling follow-up.",
        parent=edu_node,
        critical=True
    )
    await evaluator.verify(
        claim="An approved online option is eHome's eight-hour course, which includes a mandatory one-on-one counseling follow-up.",
        node=online_ehome_node,
        sources=auth_urls,
        additional_instruction="Confirm approved online provider option (eHomeAmerica) and requirement for 1-on-1 counseling follow-up."
    )

    # In-person/virtual option NeighborWorks/HUD-approved agency
    live_option_node = evaluator.add_leaf(
        id="InPerson_or_Virtual_Option_NeighborWorks_or_HUD_Agency",
        desc="Identifies an in-person/virtual option as live counseling through NeighborWorks America or a HUD-Approved Housing Counseling Agency.",
        parent=edu_node,
        critical=True
    )
    await evaluator.verify(
        claim="An in-person or virtual live counseling option is available through NeighborWorks America or a HUD-Approved Housing Counseling Agency.",
        node=live_option_node,
        sources=auth_urls,
        additional_instruction="Confirm approved live counseling provider options (NeighborWorks America or HUD-Approved HCA) on CalHFA/CA authority sources."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for CalHFA MyHome Assistance Program eligibility requirements (2025).
    """
    # Initialize evaluator with parallel root
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

    # Extraction
    extraction = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=ProgramRequirementsExtraction,
        extraction_name="program_requirements_extraction"
    )

    # Add ground truth info for context
    evaluator.add_ground_truth({
        "expected": GROUND_TRUTH,
        "note": "Expected high-level requirements for CalHFA MyHome Assistance Program in 2025"
    })

    # Build top-level critical node as per rubric
    top_node = evaluator.add_parallel(
        id="CalHFA_MyHome_Program_Requirements",
        desc="Document CalHFA MyHome Assistance Program eligibility requirements (first-time homebuyer definition, credit score requirements, and homebuyer education requirements) for 2025, with authoritative source URLs.",
        parent=root,
        critical=True
    )

    # Verify Scope
    await verify_scope(evaluator, top_node, extraction)

    # Verify First-time homebuyer definition
    await verify_first_time_homebuyer(evaluator, top_node, extraction)

    # Verify Credit Score Requirements
    await verify_credit_score(evaluator, top_node, extraction)

    # Verify Homebuyer Education Requirements
    await verify_homebuyer_education(evaluator, top_node, extraction)

    # Return structured evaluation summary
    return evaluator.get_summary()