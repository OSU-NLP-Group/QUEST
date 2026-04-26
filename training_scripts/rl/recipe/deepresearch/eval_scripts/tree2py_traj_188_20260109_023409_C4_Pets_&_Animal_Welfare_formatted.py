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
TASK_ID = "port_orange_dog_2026"
TASK_DESCRIPTION = """
A resident of Port Orange, Florida adopts a 4-month-old puppy in January 2026. What are all the regulatory requirements they must satisfy to legally own and maintain this dog in Port Orange during the first year of ownership?
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class RabiesRequirement(BaseModel):
    mentioned: Optional[bool] = None
    mentions_four_months: Optional[bool] = None
    mentions_florida_law: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class SterilizationOrPermitRequirement(BaseModel):
    mentioned: Optional[bool] = None
    mentions_six_months_deadline: Optional[bool] = None
    states_either_sterilize_or_unaltered_permit: Optional[bool] = None
    mentions_port_orange_ordinance: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class AnnualTagRequirement(BaseModel):
    mentioned: Optional[bool] = None
    tag_current_required: Optional[bool] = None
    rabies_certificate_required: Optional[bool] = None
    sterilization_doc_or_unaltered_permit_required: Optional[bool] = None
    sterilized_fee_usd: Optional[str] = None
    unaltered_fee_usd: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PortOrangeDogRequirements(BaseModel):
    rabies: Optional[RabiesRequirement] = None
    sterilization_or_permit: Optional[SterilizationOrPermitRequirement] = None
    annual_tag: Optional[AnnualTagRequirement] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
    Extract whether the answer explicitly states the following regulatory requirements for Port Orange, Florida, for a 4‑month‑old puppy adopted in January 2026 during the first year of ownership. Do not infer beyond what is written; only capture what the answer actually states. Also extract any URLs cited by the answer that are relevant to each requirement.

    Return a JSON object with the following structure:

    {
      "rabies": {
        "mentioned": boolean | null,
        "mentions_four_months": boolean | null,
        "mentions_florida_law": boolean | null,
        "sources": string[]   // Only URLs that appear in the answer text relevant to rabies vaccination
      },
      "sterilization_or_permit": {
        "mentioned": boolean | null,
        "mentions_six_months_deadline": boolean | null,
        "states_either_sterilize_or_unaltered_permit": boolean | null,
        "mentions_port_orange_ordinance": boolean | null,
        "sources": string[]   // Only URLs that appear in the answer text relevant to sterilization or unaltered permit
      },
      "annual_tag": {
        "mentioned": boolean | null,
        "tag_current_required": boolean | null,
        "rabies_certificate_required": boolean | null,
        "sterilization_doc_or_unaltered_permit_required": boolean | null,
        "sterilized_fee_usd": string | null,                 // e.g., "$10.00" or "10"
        "unaltered_fee_usd": string | null,                  // e.g., "$15.00" or "15"
        "sources": string[]   // Only URLs that appear in the answer text relevant to the Port Orange animal tag
      }
    }

    Notes and rules:
    - mentioned: true only if the answer clearly addresses that requirement.
    - mentions_four_months: true if the answer explicitly states that rabies vaccination is required by 4 months of age (Florida requirement).
    - mentions_florida_law: true if it explicitly attributes rabies requirement to Florida state law/statute.
    - mentions_six_months_deadline: true if spay/neuter is stated as due by 6 months of age, or that an unaltered animal permit is needed by that point instead.
    - states_either_sterilize_or_unaltered_permit: true if the answer clearly states it is EITHER spay/neuter OR valid unaltered permit for compliance.
    - mentions_port_orange_ordinance: true if it clearly attributes the spay/neuter or unaltered permit policy to Port Orange ordinance/policy (city-level).
    - annual_tag.tag_current_required: true if the answer says the owner must obtain and keep current a Port Orange annual animal tag (city pet license).
    - annual_tag.rabies_certificate_required: true if the answer says a current rabies vaccination certificate from a licensed veterinarian is required to obtain the tag.
    - annual_tag.sterilization_doc_or_unaltered_permit_required: true if the answer says a sterilization certificate OR documentation of a valid unaltered animal permit is required for tag registration.
    - annual_tag.sterilized_fee_usd and annual_tag.unaltered_fee_usd: capture any explicit fee amounts stated for sterilized and unsterilized dogs (with valid unaltered permit). Keep as strings exactly as written (e.g., "$10.00", "10", "USD 10").
    - sources arrays must consist ONLY of URLs explicitly present in the answer (e.g., plain links or markdown links). Do not invent URLs.

    If any item is not present in the answer, return null for the boolean or string field as appropriate, and return an empty array for sources.
    """


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    extraction: PortOrangeDogRequirements,
) -> None:
    """
    Build the verification tree following the rubric and run checks.
    All leaves are phrased as "answer coverage" checks: whether the answer explicitly states each required item.
    """

    # Add the rubric root as a critical parallel node under evaluator.root (which is non-critical by framework design)
    compliance_root = evaluator.add_parallel(
        id="Complete_Regulatory_Compliance",
        desc="All stated regulatory requirements (per the provided constraints) for first-year dog ownership/maintenance in Port Orange, Florida are addressed.",
        parent=evaluator.root,
        critical=True
    )

    # ---------------- Requirement 1: Rabies vaccination by 4 months (FL law) ----------------
    req1_node = evaluator.add_leaf(
        id="Requirement_1_Rabies_Vaccination",
        desc="Dog has received rabies vaccination by 4 months of age as required by Florida state law.",
        parent=compliance_root,
        critical=True
    )
    # Claim focuses on whether the answer explicitly states this requirement
    claim_req1 = (
        "Within the answer text, it is explicitly stated that the dog must receive a rabies vaccination "
        "by 4 months of age, as required by Florida state law."
    )
    await evaluator.verify(
        claim=claim_req1,
        node=req1_node,
        additional_instruction=(
            "Judge only based on the answer content. Accept minor phrasing variations for 'by 4 months', "
            "such as 'no later than four (4) months' or 'at 4 months or earlier'. The key is that the answer "
            "explicitly states this requirement and attributes it to Florida state law."
        )
    )

    # ---------------- Requirement 2: Spay/neuter by 6 months OR unaltered permit (Port Orange ordinance) ----------------
    req2_node = evaluator.add_leaf(
        id="Requirement_2_Sterilization_Or_Permit",
        desc="By the time the dog is 6 months old, the dog is spayed/neutered OR the owner holds a valid unaltered animal permit (per Port Orange ordinance).",
        parent=compliance_root,
        critical=True
    )
    claim_req2 = (
        "Within the answer text, it is explicitly stated that by 6 months of age the dog must be spayed or neutered "
        "OR the owner must hold a valid unaltered animal permit, and that this is per Port Orange ordinance (city policy)."
    )
    await evaluator.verify(
        claim=claim_req2,
        node=req2_node,
        additional_instruction=(
            "Judge only based on the answer content. The answer should clearly present the EITHER-OR compliance path "
            "(sterilization OR valid unaltered permit) tied to a 6-month-old threshold and attribute this to Port Orange (city) rules."
        )
    )

    # ---------------- Requirement 3: Annual animal tag (with documentation and correct fee) ----------------
    req3_parent = evaluator.add_parallel(
        id="Requirement_3_Annual_Animal_Tag",
        desc="Owner complies with Port Orange annual animal tag requirements (tag, required documentation, and correct fee).",
        parent=compliance_root,
        critical=True
    )

    # Leaf a) Tag obtained and kept current
    req3a = evaluator.add_leaf(
        id="Annual_Animal_Tag_Obtained_And_Current",
        desc="Owner obtains/maintains a current Port Orange annual animal tag for the dog.",
        parent=req3_parent,
        critical=True
    )
    claim_req3a = (
        "Within the answer text, it is explicitly stated that the owner must obtain and maintain a current "
        "Port Orange annual animal tag (i.e., city pet license) for the dog."
    )
    # Leaf b) Rabies certificate is needed for the tag
    req3b = evaluator.add_leaf(
        id="Rabies_Certificate_Provided_For_Tag",
        desc="Owner provides a current rabies vaccination certificate from a licensed veterinarian to obtain the animal tag.",
        parent=req3_parent,
        critical=True
    )
    claim_req3b = (
        "Within the answer text, it is explicitly stated that to obtain the Port Orange animal tag, "
        "a current rabies vaccination certificate from a licensed veterinarian is required."
    )
    # Leaf c) Sterilization certificate or unaltered permit needed for tag
    req3c = evaluator.add_leaf(
        id="Sterilization_Certificate_Or_Unaltered_Permit_Provided_For_Tag",
        desc="Owner provides either a certificate of sterilization OR documentation of a valid unaltered animal permit for tag registration.",
        parent=req3_parent,
        critical=True
    )
    claim_req3c = (
        "Within the answer text, it is explicitly stated that for tag registration the owner must provide either "
        "a sterilization certificate or documentation of a valid unaltered animal permit."
    )
    # Leaf d) Correct fee amounts
    req3d = evaluator.add_leaf(
        id="Correct_Annual_License_Fee_Paid",
        desc="Owner pays the correct annual license fee: $10.00 for sterilized dogs; $15.00 for unsterilized dogs (with valid unaltered animal permit).",
        parent=req3_parent,
        critical=True
    )
    claim_req3d = (
        "Within the answer text, it is explicitly stated that the Port Orange annual animal license fee is "
        "$10.00 for sterilized dogs, and $15.00 for unsterilized dogs with a valid unaltered animal permit."
    )

    # Batch verify the four leaves under the same parent to avoid premature skipping from critical sibling gating
    await evaluator.batch_verify([
        (
            claim_req3a,
            None,
            req3a,
            "Judge only based on the answer content. Accept equivalent phrasing such as 'annual pet license', "
            "'city tag', or 'keep the tag current for the year'. It must be clearly Port Orange city, not only county."
        ),
        (
            claim_req3b,
            None,
            req3b,
            "Judge only based on the answer content. Accept equivalent wording such as 'current rabies certificate "
            "from a veterinarian is required to license/register the dog'."
        ),
        (
            claim_req3c,
            None,
            req3c,
            "Judge only based on the answer content. The answer must clearly say that either a sterilization certificate "
            "OR proof of a valid unaltered animal permit is required for the tag."
        ),
        (
            claim_req3d,
            None,
            req3d,
            "Judge only based on the answer content. Minor formatting differences are okay (e.g., '$10', '10.00 USD'), "
            "but both amounts and the condition ('with valid unaltered permit' for the $15 amount) must be present."
        ),
    ])

    # Optionally, record the extracted URLs as additional info for transparency
    evaluator.add_custom_info(
        info={
            "rabies_sources": extraction.rabies.sources if extraction and extraction.rabies else [],
            "sterilization_or_permit_sources": extraction.sterilization_or_permit.sources if extraction and extraction.sterilization_or_permit else [],
            "annual_tag_sources": extraction.annual_tag.sources if extraction and extraction.annual_tag else [],
        },
        info_type="extracted_sources",
        info_name="extracted_sources_by_requirement"
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
    Evaluate an answer for Port Orange, FL first-year dog regulatory compliance coverage.
    Returns a standard evaluation summary dict.
    """
    # Initialize evaluator with a neutral root (non-critical by framework design)
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

    # Record scenario context as custom info for transparency
    evaluator.add_custom_info(
        info={
            "adoption_month_year": "January 2026",
            "puppy_age_months_at_adoption": 4,
            "first_year_coverage": "January 2026 through January 2027 (anniversary-year logic may apply by city)",
        },
        info_type="scenario_context",
        info_name="scenario_context"
    )

    # Extract structured requirement coverage and any cited URLs from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=PortOrangeDogRequirements,
        extraction_name="requirements_extraction"
    )

    # Optional: add lightweight ground-truth expectations for clarity (not used for scoring)
    evaluator.add_ground_truth({
        "required_items": [
            "Rabies vaccination by 4 months per Florida law.",
            "By 6 months, either spay/neuter OR valid unaltered animal permit per Port Orange ordinance.",
            "Port Orange annual animal tag kept current.",
            "Rabies vaccination certificate required to obtain tag.",
            "Provide sterilization certificate OR valid unaltered animal permit to register tag.",
            "Annual license fee is $10.00 if sterilized; $15.00 if unsterilized with valid unaltered permit."
        ]
    })

    # Build rubric tree and perform verifications
    await build_and_verify_tree(evaluator, extraction)

    # Return framework-standard summary
    return evaluator.get_summary()