import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "multi_state_pet_care_plan"
TASK_DESCRIPTION = (
    "Develop a compliance and financial plan covering foster eligibility/commitments, "
    "New Orleans intact dog permit requirements, Rhode Island intact cat permit requirements, "
    "and correct cost calculations with supporting URLs."
)

# Ground truth / expected values based on rubric
EXPECTED = {
    "foster_min_age": 18,
    "bottle_baby_feeding_freq_desc": "every 2–6 hours depending on age for 5–8 weeks bottle-babies",
    "foster_min_daily_time_hours": 2,
    "no_initial_fee": 95,
    "no_processing_fee": 5,
    "no_renewal_fee": 20,  # assumption given
    "no_renewal_processing_fee": 5,  # assumption given
    "no_applicant_min_age": 18,
    "no_dog_threshold_months": 6,
    "ri_threshold_months": 6,
    "ri_annual_fee": 100,
}
EXPECTED["combined_first_year_total"] = (
    EXPECTED["no_initial_fee"] + EXPECTED["no_processing_fee"] + EXPECTED["ri_annual_fee"]
)
EXPECTED["combined_annual_recurring_total"] = (
    EXPECTED["no_renewal_fee"] + EXPECTED["no_renewal_processing_fee"] + EXPECTED["ri_annual_fee"]
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FosterExtraction(BaseModel):
    foster_min_age_text: Optional[str] = None
    bottle_baby_feeding_frequency_5_8_weeks: Optional[str] = None
    foster_min_daily_time_commitment_text: Optional[str] = None
    foster_reference_url: Optional[str] = None


class NewOrleansPermitExtraction(BaseModel):
    spca_reference_url: Optional[str] = None
    applicant_min_age_text: Optional[str] = None
    dog_threshold_age_text: Optional[str] = None
    required_documents: List[str] = Field(default_factory=list)
    initial_application_fee_text: Optional[str] = None
    online_processing_fee_text: Optional[str] = None
    first_year_total_text: Optional[str] = None


class RhodeIslandPermitExtraction(BaseModel):
    reference_url: Optional[str] = None
    threshold_age_text: Optional[str] = None
    annual_permit_fee_text: Optional[str] = None


class CostAnalysisExtraction(BaseModel):
    combined_first_year_total_text: Optional[str] = None
    combined_annual_recurring_total_text: Optional[str] = None


class MultiStatePlanExtraction(BaseModel):
    foster: FosterExtraction = FosterExtraction()
    new_orleans: NewOrleansPermitExtraction = NewOrleansPermitExtraction()
    rhode_island: RhodeIslandPermitExtraction = RhodeIslandPermitExtraction()
    cost_analysis: CostAnalysisExtraction = CostAnalysisExtraction()


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return """
    Extract the following structured information exactly as stated in the answer. If an item is not explicitly provided, return null (or empty list where applicable). Do NOT infer or invent values, and extract URLs as full strings when present.

    Section: Foster Care
    - foster_min_age_text: The minimum age requirement for participating in foster care programs (e.g., "18+").
    - bottle_baby_feeding_frequency_5_8_weeks: The feeding frequency requirement for bottle-baby animals ages 5–8 weeks (e.g., "every 2–6 hours depending on age").
    - foster_min_daily_time_commitment_text: The minimum daily time commitment required for foster care (e.g., "at least 2 hours/day").
    - foster_reference_url: A URL cited for foster policies or guidelines, if any.

    Section: New Orleans Intact Dog Permit (Louisiana SPCA)
    - spca_reference_url: A URL from the Louisiana SPCA website that details New Orleans intact permit requirements.
    - applicant_min_age_text: The minimum age requirement for the applicant (e.g., "18+").
    - dog_threshold_age_text: The minimum age at which an intact permit or spay/neuter is required (e.g., "6 months or older").
    - required_documents: A list of exactly the documents stated in the answer for the permit application.
    - initial_application_fee_text: The initial application fee (e.g., "$95").
    - online_processing_fee_text: The online processing fee per pet (e.g., "$5").
    - first_year_total_text: The total first-year cost explicitly stated in the answer for the New Orleans permit (sum of initial fee + processing fee), if the answer includes it.

    Section: Rhode Island Intact Cat Permit
    - reference_url: A URL describing Rhode Island's cat spay/neuter law and permit requirements.
    - threshold_age_text: The minimum age at which RI requires cats to be spayed/neutered or have a permit (e.g., "6 months").
    - annual_permit_fee_text: The annual permit fee per intact cat (e.g., "$100").

    Section: Cost Analysis
    - combined_first_year_total_text: The total first-year cost for both permits combined, as stated in the answer.
    - combined_annual_recurring_total_text: The total annual recurring cost for both permits in subsequent years as stated in the answer (assume New Orleans renewal $20 + $5 processing, Rhode Island $100/year).

    Return a single JSON object matching the MultiStatePlanExtraction schema.
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def parse_first_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_int(text: Optional[str]) -> Optional[int]:
    val = parse_first_number(text)
    return int(round(val)) if val is not None else None


def normalize_money(text: Optional[str]) -> Optional[int]:
    """Extract the first numeric value as an integer dollars amount."""
    val = parse_first_number(text)
    if val is None:
        return None
    return int(round(val))


def url_domain_contains(url: Optional[str], keyword: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    return keyword.lower() in url.lower()


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_foster_section(evaluator: Evaluator, parent_node, foster: FosterExtraction) -> None:
    foster_node = evaluator.add_parallel(
        id="Foster_Care_Eligibility_and_Commitment",
        desc="Address foster care eligibility and required bottle-baby kitten care commitments.",
        parent=parent_node,
        critical=True,
    )

    # Minimum Age + Qualification -> sequential: first state the requirement, then confirm Maria qualifies
    age_seq = evaluator.add_sequential(
        id="Foster_Minimum_Age_and_Qualification",
        desc="State the minimum foster age requirement (18+) and confirm Maria qualifies given she is 25.",
        parent=foster_node,
        critical=True,
    )

    # Leaf: Minimum age is 18+
    leaf_age_req = evaluator.add_leaf(
        id="Foster_Min_Age_18",
        desc="Minimum foster age requirement is 18+.",
        parent=age_seq,
        critical=True,
    )
    claim_age = "The minimum age requirement for foster care participation is 18 years old."
    await evaluator.verify(
        claim=claim_age,
        node=leaf_age_req,
        sources=foster.foster_reference_url if foster.foster_reference_url else None,
        additional_instruction="Verify the foster policy age requirement if a URL is provided; otherwise confirm the answer states 18+."
    )

    # Leaf: Maria qualifies (she is 25)
    leaf_maria_qualifies = evaluator.add_leaf(
        id="Foster_Maria_Qualifies",
        desc="Maria qualifies at age 25 against the 18+ requirement.",
        parent=age_seq,
        critical=True,
    )
    claim_qual = "Maria qualifies for foster care because the minimum age is 18 and she is 25 years old."
    await evaluator.verify(
        claim=claim_qual,
        node=leaf_maria_qualifies,
        sources=None,
        additional_instruction="Simple logical check: confirm the statement that 25 >= 18 is correct."
    )

    # Bottle-baby feeding frequency (5–8 weeks): every 2–6 hours depending on age
    leaf_feed = evaluator.add_leaf(
        id="Bottle_Baby_Feeding_Frequency",
        desc="State the feeding frequency requirement for bottle-baby animals ages 5–8 weeks (every 2–6 hours depending on age).",
        parent=foster_node,
        critical=True,
    )
    claim_feed = "For bottle-baby animals ages 5–8 weeks, the feeding frequency is every 2–6 hours depending on age."
    await evaluator.verify(
        claim=claim_feed,
        node=leaf_feed,
        sources=foster.foster_reference_url if foster.foster_reference_url else None,
        additional_instruction="Check foster guidelines if URL exists; otherwise validate that the answer states this range clearly."
    )

    # Minimum daily time commitment: at least 2 hours/day
    leaf_time = evaluator.add_leaf(
        id="Foster_Minimum_Daily_Time_Commitment",
        desc="State the minimum daily time commitment for foster care (at least 2 hours/day).",
        parent=foster_node,
        critical=True,
    )
    claim_time = "The minimum daily time commitment for foster care is at least 2 hours per day."
    await evaluator.verify(
        claim=claim_time,
        node=leaf_time,
        sources=foster.foster_reference_url if foster.foster_reference_url else None,
        additional_instruction="If URL exists, confirm commitment guidance; else confirm the answer states 'at least 2 hours/day'."
    )


async def verify_no_permit(evaluator: Evaluator, parent_node, no_data: NewOrleansPermitExtraction) -> None:
    no_node = evaluator.add_parallel(
        id="New_Orleans_Intact_Dog_Permit",
        desc="Provide New Orleans intact dog permit eligibility, required documents (exactly five), fees, and a Louisiana SPCA reference URL.",
        parent=parent_node,
        critical=True,
    )

    # Reference URL group: existence + page relevance
    ref_group = evaluator.add_parallel(
        id="NO_Reference_URL_Group",
        desc="Louisiana SPCA reference URL exists and describes intact permit requirements.",
        parent=no_node,
        critical=True,
    )

    # Existence & domain check (custom)
    exists_valid = evaluator.add_custom_node(
        result=(no_data.spca_reference_url is not None and url_domain_contains(no_data.spca_reference_url, "louisianaspca.org")),
        id="NO_SPCA_URL_Provided_ValidDomain",
        desc="Louisiana SPCA intact permit reference URL is provided and from louisianaspca.org.",
        parent=ref_group,
        critical=True,
    )

    # Content check via URL verification
    leaf_ref = evaluator.add_leaf(
        id="NO_Reference_URL",
        desc="Provide a Louisiana SPCA reference URL that details the New Orleans intact permit requirements.",
        parent=ref_group,
        critical=True,
    )
    claim_ref = "This page describes the New Orleans intact (unsterilized pet) permit requirements, including eligibility, documentation, and fees."
    await evaluator.verify(
        claim=claim_ref,
        node=leaf_ref,
        sources=no_data.spca_reference_url if no_data.spca_reference_url else None,
        additional_instruction="Confirm the page details the intact permit requirements for New Orleans."
    )

    # Minimum age requirements (parallel)
    age_reqs = evaluator.add_parallel(
        id="NO_Minimum_Age_Requirements",
        desc="State minimum age requirement for the applicant (18+) and for the dog (permit required if dog is 6+ months / spay-neuter-or-permit threshold).",
        parent=no_node,
        critical=True,
    )

    leaf_app_age = evaluator.add_leaf(
        id="NO_Applicant_Min_Age",
        desc="Applicant minimum age is 18+.",
        parent=age_reqs,
        critical=True,
    )
    claim_app_age = "The minimum applicant age to obtain an intact permit is 18 years old."
    await evaluator.verify(
        claim=claim_app_age,
        node=leaf_app_age,
        sources=no_data.spca_reference_url if no_data.spca_reference_url else None,
        additional_instruction="Verify applicant eligibility age on the LA SPCA page."
    )

    leaf_dog_age = evaluator.add_leaf(
        id="NO_Dog_Min_Age",
        desc="Dog threshold age is 6 months or older for spay/neuter or intact permit requirement.",
        parent=age_reqs,
        critical=True,
    )
    claim_dog_age = "Dogs 6 months or older must be spayed/neutered or have an intact permit."
    await evaluator.verify(
        claim=claim_dog_age,
        node=leaf_dog_age,
        sources=no_data.spca_reference_url if no_data.spca_reference_url else None,
        additional_instruction="Verify threshold age requirement for dogs (6+ months) on the LA SPCA page."
    )

    # Required documents list (parallel)
    doc_list = evaluator.add_parallel(
        id="NO_Five_Required_Documents_List",
        desc="List all five required documents for the New Orleans intact dog permit application.",
        parent=no_node,
        critical=True,
    )

    # Five document verification leaves (critical)
    # 1: Proof of vaccinations (Rabies, Distemper, Parvovirus).
    leaf_doc1 = evaluator.add_leaf(
        id="NO_Document_1",
        desc="Document #1: Proof of vaccinations (Rabies, Distemper, Parvovirus).",
        parent=doc_list,
        critical=True,
    )
    await evaluator.verify(
        claim="The intact permit application requires proof of current vaccinations including Rabies, Distemper, and Parvovirus.",
        node=leaf_doc1,
        sources=no_data.spca_reference_url if no_data.spca_reference_url else None,
        additional_instruction="Check the required documents list; allow minor phrasing differences like 'distemper/parvo'."
    )

    # 2: Current City license/rabies tag number.
    leaf_doc2 = evaluator.add_leaf(
        id="NO_Document_2",
        desc="Document #2: Current City license/rabies tag number.",
        parent=doc_list,
        critical=True,
    )
    await evaluator.verify(
        claim="The intact permit application requires the current City license or rabies tag number.",
        node=leaf_doc2,
        sources=no_data.spca_reference_url if no_data.spca_reference_url else None,
        additional_instruction="Look for licensing or rabies tag requirement in the documents list."
    )

    # 3: Proof of microchip.
    leaf_doc3 = evaluator.add_leaf(
        id="NO_Document_3",
        desc="Document #3: Proof of microchip.",
        parent=doc_list,
        critical=True,
    )
    await evaluator.verify(
        claim="The intact permit application requires proof of microchip.",
        node=leaf_doc3,
        sources=no_data.spca_reference_url if no_data.spca_reference_url else None,
        additional_instruction="Verify microchip requirement."
    )

    # 4: Current photo of the pet.
    leaf_doc4 = evaluator.add_leaf(
        id="NO_Document_4",
        desc="Document #4: Current photo of the pet.",
        parent=doc_list,
        critical=True,
    )
    await evaluator.verify(
        claim="The intact permit application requires a current photo of the pet.",
        node=leaf_doc4,
        sources=no_data.spca_reference_url if no_data.spca_reference_url else None,
        additional_instruction="Verify photo requirement."
    )

    # 5: Copy of owner's ID.
    leaf_doc5 = evaluator.add_leaf(
        id="NO_Document_5",
        desc="Document #5: Copy of owner's ID.",
        parent=doc_list,
        critical=True,
    )
    await evaluator.verify(
        claim="The intact permit application requires a copy of the owner's ID.",
        node=leaf_doc5,
        sources=no_data.spca_reference_url if no_data.spca_reference_url else None,
        additional_instruction="Verify owner ID requirement."
    )

    # Fees & first-year cost (sequential)
    fees_seq = evaluator.add_sequential(
        id="NO_Fees_and_First_Year_Cost",
        desc="Provide the New Orleans initial application fee, online processing fee, and correctly compute the first-year total.",
        parent=no_node,
        critical=True,
    )

    leaf_initial_fee = evaluator.add_leaf(
        id="NO_Initial_Application_Fee",
        desc="State the initial application fee is $95.",
        parent=fees_seq,
        critical=True,
    )
    await evaluator.verify(
        claim="The initial application fee for the New Orleans intact permit is $95.",
        node=leaf_initial_fee,
        sources=no_data.spca_reference_url if no_data.spca_reference_url else None,
        additional_instruction="Verify the fee amount on the LA SPCA page."
    )

    leaf_processing_fee = evaluator.add_leaf(
        id="NO_Online_Processing_Fee",
        desc="State the online processing fee is $5 per pet.",
        parent=fees_seq,
        critical=True,
    )
    await evaluator.verify(
        claim="The online processing fee per pet for the intact permit application is $5.",
        node=leaf_processing_fee,
        sources=no_data.spca_reference_url if no_data.spca_reference_url else None,
        additional_instruction="Verify the processing fee on the LA SPCA page."
    )

    # Computation check for first-year total (custom)
    calc_total = evaluator.add_custom_node(
        result=(EXPECTED["no_initial_fee"] + EXPECTED["no_processing_fee"] == EXPECTED["no_initial_fee"] + EXPECTED["no_processing_fee"]),
        id="NO_First_Year_Total_Calculated_Correctly",
        desc="Correctly calculate first-year total cost as (initial fee + processing fee).",
        parent=fees_seq,
        critical=True,
    )


async def verify_ri_permit(evaluator: Evaluator, parent_node, ri_data: RhodeIslandPermitExtraction) -> None:
    ri_node = evaluator.add_parallel(
        id="Rhode_Island_Intact_Cat_Permit",
        desc="Provide Rhode Island intact cat permit threshold age, annual fee, and a reference URL.",
        parent=parent_node,
        critical=True,
    )

    # Reference URL group: existence + content
    ri_ref_group = evaluator.add_parallel(
        id="RI_Reference_URL_Group",
        desc="RI reference URL exists and describes cat spay/neuter law and permit requirements.",
        parent=ri_node,
        critical=True,
    )

    ri_exists = evaluator.add_custom_node(
        result=(ri_data.reference_url is not None and isinstance(ri_data.reference_url, str) and len(ri_data.reference_url.strip()) > 0),
        id="RI_Reference_URL_Provided",
        desc="A Rhode Island reference URL describing cat spay/neuter law and permit is provided.",
        parent=ri_ref_group,
        critical=True,
    )

    leaf_ri_ref = evaluator.add_leaf(
        id="RI_Reference_URL",
        desc="Provide a reference URL describing Rhode Island's cat spay/neuter law and permit requirements.",
        parent=ri_ref_group,
        critical=True,
    )
    await evaluator.verify(
        claim="This page describes Rhode Island's cat spay/neuter law and intact permit requirements.",
        node=leaf_ri_ref,
        sources=ri_data.reference_url if ri_data.reference_url else None,
        additional_instruction="Confirm the page includes RI's cat spay/neuter law and intact permit details."
    )

    # Threshold age (6 months)
    leaf_ri_age = evaluator.add_leaf(
        id="RI_Cat_Threshold_Age",
        desc="State the minimum age at which Rhode Island requires cats to be spayed/neutered or have a permit (6 months or older).",
        parent=ri_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Rhode Island requires cats 6 months or older to be spayed/neutered or have an intact permit.",
        node=leaf_ri_age,
        sources=ri_data.reference_url if ri_data.reference_url else None,
        additional_instruction="Verify threshold age requirement for cats in RI."
    )

    # Annual permit fee ($100)
    leaf_ri_fee = evaluator.add_leaf(
        id="RI_Annual_Permit_Fee",
        desc="State the annual permit fee per cat is $100.",
        parent=ri_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The annual permit fee per intact cat in Rhode Island is $100.",
        node=leaf_ri_fee,
        sources=ri_data.reference_url if ri_data.reference_url else None,
        additional_instruction="Verify annual fee requirement for intact cats in RI."
    )


async def verify_cost_analysis(evaluator: Evaluator, parent_node, plan: MultiStatePlanExtraction) -> None:
    cost_node = evaluator.add_parallel(
        id="Cost_Analysis",
        desc="Correctly compute combined first-year costs and combined annual recurring costs (using given renewal assumptions).",
        parent=parent_node,
        critical=True,
    )

    # Combined first-year cost: compare the answer's stated combined total to expected (95 + 5 + 100 = 200)
    combined_first_year_node = evaluator.add_custom_node(
        result=(
            normalize_money(plan.cost_analysis.combined_first_year_total_text) == EXPECTED["combined_first_year_total"]
        ),
        id="Combined_First_Year_Cost",
        desc="Correctly calculate the total first-year cost for both permits combined.",
        parent=cost_node,
        critical=True,
    )

    # Combined annual recurring cost: compare the answer's stated combined recurring total to expected (20 + 5 + 100 = 125)
    combined_recurring_node = evaluator.add_custom_node(
        result=(
            normalize_money(plan.cost_analysis.combined_annual_recurring_total_text) == EXPECTED["combined_annual_recurring_total"]
        ),
        id="Combined_Annual_Recurring_Cost",
        desc="Correctly calculate the total annual recurring cost after the first year using: New Orleans renewal $20 + $5 processing, Rhode Island $100/year.",
        parent=cost_node,
        critical=True,
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate the multi-state pet care compliance & financial plan answer.
    """
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

    # Top-level critical plan node
    top = evaluator.add_parallel(
        id="Complete_Multi_State_Pet_Care_Plan",
        desc="Develop a compliance and financial plan covering foster eligibility/commitments, New Orleans intact dog permit requirements, Rhode Island intact cat permit requirements, and correct cost calculations with supporting URLs.",
        parent=root,
        critical=True,
    )

    # Extract all structured information in one pass
    plan: MultiStatePlanExtraction = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=MultiStatePlanExtraction,
        extraction_name="plan_extraction",
    )

    # Add expected ground truth info
    evaluator.add_ground_truth(
        {
            "expected": EXPECTED,
            "notes": "Expected numeric thresholds and fees derived from rubric; URL verifications rely on pages cited in the answer.",
        },
        gt_type="ground_truth",
    )

    # Build and verify subtrees
    await verify_foster_section(evaluator, top, plan.foster)
    await verify_no_permit(evaluator, top, plan.new_orleans)
    await verify_ri_permit(evaluator, top, plan.rhode_island)
    await verify_cost_analysis(evaluator, top, plan)

    # Return standard summary
    return evaluator.get_summary()