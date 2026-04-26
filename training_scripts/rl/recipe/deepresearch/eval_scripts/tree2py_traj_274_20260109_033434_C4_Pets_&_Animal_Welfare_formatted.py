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
TASK_ID = "texas_rescue_org_v1"
TASK_DESCRIPTION = (
    "Identify an animal rescue organization located in Texas that meets all of the following criteria: "
    "(1) has 501(c)(3) nonprofit status, (2) operates an active foster care program, "
    "(3) offers low-cost, subsidized, or free spay/neuter services to the community, "
    "(4) accepts and facilitates dog adoptions, (5) provides microchipping services, "
    "and (6) charges dog adoption fees within the typical range of $50-$200. "
    "Provide the organization's name, a brief description of how it meets each criterion, "
    "and a reference URL confirming this information."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class OrgExtraction(BaseModel):
    """
    Extract exactly one organization (the main one presented by the answer).
    All fields should be directly taken from the answer text only; do not invent.
    """
    organization_name: Optional[str] = None

    # Optional brief descriptions the answer provides (strings as-is from answer)
    location_text: Optional[str] = None
    nonprofit_501c3_text: Optional[str] = None
    foster_program_text: Optional[str] = None
    spay_neuter_text: Optional[str] = None
    dog_adoptions_text: Optional[str] = None
    microchipping_text: Optional[str] = None
    adoption_fee_text: Optional[str] = None

    # Reference URLs explicitly listed in the answer supporting claims
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_org_info() -> str:
    return """
    You will extract exactly one organization (the primary one the answer recommends or focuses on) and the supporting info the answer provides.

    Extract the following fields from the answer:
    - organization_name: The official name of the organization.
    - location_text: The phrase/sentence indicating the org is in Texas (e.g., "Austin, TX", "Texas-based", etc.). If not explicitly present, return null.
    - nonprofit_501c3_text: The phrase/sentence indicating 501(c)(3) status. If not explicitly present, return null.
    - foster_program_text: The phrase/sentence indicating an active foster program. If not explicitly present, return null.
    - spay_neuter_text: The phrase/sentence indicating low-cost/subsidized/free spay/neuter services to the community. If not explicitly present, return null.
    - dog_adoptions_text: The phrase/sentence indicating the org accepts/facilitates dog adoptions. If not explicitly present, return null.
    - microchipping_text: The phrase/sentence indicating microchipping services. If not explicitly present, return null.
    - adoption_fee_text: The phrase/sentence indicating dog adoption fee(s) or range/tiers. If not explicitly present, return null.

    - reference_urls: An array of all URLs explicitly shown in the answer that are meant to support any of the above claims (official site pages, adoption pages, services pages, fee pages, etc.).
      Rules for URLs:
      * Extract only actual URLs present in the answer (plain, markdown, etc.).
      * Include full URLs with protocol (http:// or https://). If missing, prepend http://.
      * Deduplicate exact duplicates.

    If any field is not present in the answer, return null (or empty array for reference_urls if none).
    """


# --------------------------------------------------------------------------- #
# Helper to assemble verification items                                       #
# --------------------------------------------------------------------------- #
def _safe_name(name: Optional[str]) -> str:
    return name or "the organization"


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
    Evaluate an answer for the Texas animal rescue organization task.
    """
    # Initialize evaluator (root is non-critical by framework design)
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

    # Extract structured org info from the answer
    org_info = await evaluator.extract(
        prompt=prompt_extract_org_info(),
        template_class=OrgExtraction,
        extraction_name="org_extraction",
    )

    # Optionally record some custom info for debugging
    evaluator.add_custom_info(
        info={
            "extracted_name": org_info.organization_name,
            "num_reference_urls": len(org_info.reference_urls),
        },
        info_type="debug",
        info_name="extraction_summary",
    )

    # Create a critical parallel node as the main task container
    task_node = evaluator.add_parallel(
        id="Texas_Animal_Rescue_Organization",
        desc="Identify one animal rescue organization in Texas that satisfies all listed criteria and provide the required supporting information.",
        parent=root,
        critical=True,
    )

    # Existence/Gating critical checks first (so subsequent verifications can auto-skip if they fail)
    # 1) Organization name provided
    evaluator.add_custom_node(
        result=bool(org_info.organization_name and org_info.organization_name.strip()),
        id="Organization_Name_Provided",
        desc="The response provides the organization's name.",
        parent=task_node,
        critical=True,
    )

    # 2) At least one reference URL exists (our added gating node to avoid unsupported checks)
    urls_exist_node = evaluator.add_custom_node(
        result=bool(org_info.reference_urls and len(org_info.reference_urls) > 0),
        id="Reference_URLs_Exist",
        desc="At least one reference URL is provided in the response.",
        parent=task_node,
        critical=True,
    )

    # Prepare common values
    org_name = _safe_name(org_info.organization_name)
    urls = org_info.reference_urls

    # Build all verification leaves (critical) and run verification
    # Location in Texas
    loc_node = evaluator.add_leaf(
        id="Location_Texas",
        desc="The organization is located in and operates within Texas, with a brief description supporting this claim.",
        parent=task_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{org_name} is located in Texas (TX) or explicitly operates within Texas.",
        node=loc_node,
        sources=urls,
        additional_instruction=(
            "Treat 'Austin, TX', 'Houston, TX', 'Texas-based', or explicit Texas address/service area as sufficient. "
            "The page must clearly indicate the organization is in Texas or operates in Texas."
        ),
    )

    # 501(c)(3) nonprofit status
    nonprofit_node = evaluator.add_leaf(
        id="Nonprofit_Status_501c3",
        desc="The organization has 501(c)(3) nonprofit status, with a brief description supporting this claim.",
        parent=task_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{org_name} is a 501(c)(3) nonprofit organization.",
        node=nonprofit_node,
        sources=urls,
        additional_instruction=(
            "Accept variants like '501c3' or '501(c)3'. Look for explicit statements indicating federal tax-exempt status."
        ),
    )

    # Active foster program
    foster_node = evaluator.add_leaf(
        id="Active_Foster_Program",
        desc="The organization operates an active foster care program, with a brief description supporting this claim.",
        parent=task_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{org_name} operates an active foster program for animals (e.g., they recruit fosters or provide a foster application).",
        node=foster_node,
        sources=urls,
        additional_instruction=(
            "Accept synonyms: 'foster program', 'foster network', 'become a foster', 'foster application'. "
            "The page should clearly show fostering is an active offering."
        ),
    )

    # Low-cost/subsidized/free spay/neuter services
    spay_node = evaluator.add_leaf(
        id="Spay_Neuter_Services_LowCost",
        desc="The organization offers low-cost, subsidized, or free spay/neuter services to the community, with a brief description supporting this claim.",
        parent=task_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{org_name} offers low-cost, subsidized, or free spay/neuter services to the community (directly or via vouchers/partners).",
        node=spay_node,
        sources=urls,
        additional_instruction=(
            "Look for wording like 'low-cost spay/neuter', 'spay/neuter vouchers', 'community clinic', or similar. "
            "Partner or voucher programs count if run/offered by the organization."
        ),
    )

    # Dog adoptions
    adopt_node = evaluator.add_leaf(
        id="Dog_Adoptions",
        desc="The organization accepts and facilitates dog adoptions, with a brief description supporting this claim.",
        parent=task_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{org_name} accepts and facilitates dog adoptions (e.g., adoptable dogs list, dog adoption application/process).",
        node=adopt_node,
        sources=urls,
        additional_instruction=(
            "Page should clearly indicate that dogs can be adopted from the organization. "
            "Mentions of 'adoptable dogs', 'dog adoption', or 'adoption application' with dog references are sufficient."
        ),
    )

    # Microchipping services
    microchip_node = evaluator.add_leaf(
        id="Microchipping_Services",
        desc="The organization provides microchipping services, with a brief description supporting this claim.",
        parent=task_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{org_name} provides microchipping services (as part of adoptions or as a community service).",
        node=microchip_node,
        sources=urls,
        additional_instruction=(
            "Accept phrases like 'microchipping', 'microchip', or 'pets are microchipped'. "
            "It may be offered during adoption or as a standalone clinic/service."
        ),
    )

    # Dog adoption fee range within $50–$200
    fee_node = evaluator.add_leaf(
        id="Dog_Adoption_Fee_Range",
        desc="The organization's dog adoption fees are within the range of $50–$200, with a brief description supporting this claim.",
        parent=task_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{org_name}'s dog adoption fee(s) fall within the range of $50 to $200.",
        node=fee_node,
        sources=urls,
        additional_instruction=(
            "Check the webpage(s) for stated dog adoption fee(s). If multiple tiers (e.g., puppy vs. adult), "
            "at least the standard fee for dogs should be between $50 and $200 to pass. "
            "If only cat fees are shown, or dog fees are outside this range, mark as not supported."
        ),
    )

    # Reference URL provided and relevant (validate at least one URL is valid and mentions any of the criteria)
    ref_relevant_node = evaluator.add_leaf(
        id="Reference_URL_Provided_And_Relevant",
        desc="The response provides at least one valid reference URL that supports the organization's claimed status/services/fees (i.e., is relevant evidence).",
        parent=task_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"At least one provided reference URL is a valid webpage about {org_name} that explicitly mentions "
            "at least one of the following: 501(c)(3) status, foster program, low-cost/subsidized/free spay/neuter services, "
            "dog adoptions, microchipping services, or dog adoption fees."
        ),
        node=ref_relevant_node,
        sources=urls,
        additional_instruction=(
            "Judge using only the provided URLs. A page is relevant if it is about the same organization and "
            "explicitly contains concrete information for any of the listed criteria. "
            "If the URL is broken, irrelevant, or about a different organization, the claim is not supported."
        ),
    )

    # Return summary
    return evaluator.get_summary()