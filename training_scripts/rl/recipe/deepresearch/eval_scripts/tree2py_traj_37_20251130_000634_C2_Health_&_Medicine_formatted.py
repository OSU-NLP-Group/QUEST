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
TASK_ID = "neuralink_prime_site_dallas"
TASK_DESCRIPTION = (
    "You live in Dallas, Texas and have been diagnosed with cervical spinal cord injury causing quadriplegia, "
    "which has left you with limited ability to use both hands. You are interested in participating in Neuralink's "
    "PRIME Study, a clinical trial testing a brain-computer interface device. Identify which U.S. clinical trial site "
    "currently conducting the Neuralink PRIME Study is geographically closest to your location in Dallas, Texas. "
    "For the closest site, provide the following information: (1) The facility name and location (city and state), "
    "(2) The complete physical address of the facility, (3) The main contact phone number for the facility, and "
    "(4) A reference URL from an official source (such as the institution's website or ClinicalTrials.gov) that "
    "documents this site's participation in the PRIME Study."
)

EXPECTED_NCT_ID = "NCT06429735"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ClosestSiteExtraction(BaseModel):
    facility_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    complete_address: Optional[str] = None
    main_phone_number: Optional[str] = None
    official_reference_urls: List[str] = Field(default_factory=list)
    clinicaltrials_gov_identifier: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_site_info() -> str:
    return (
        "From the answer text, extract details for the single U.S. clinical trial site the answer identifies as the "
        "closest location to Dallas, Texas for Neuralink's PRIME Study.\n"
        "Return the following fields:\n"
        "1) facility_name: The facility or institution name of the closest site.\n"
        "2) city: The city of the closest site.\n"
        "3) state: The U.S. state of the closest site (use two-letter abbreviation if provided, else full state name).\n"
        "4) complete_address: The full physical street address for the facility (include street, city, state, and ZIP if available).\n"
        "5) main_phone_number: The main contact phone number for the facility (format as provided in the answer).\n"
        "6) official_reference_urls: A list of URLs explicitly mentioned in the answer that are official sources documenting this site's participation in the PRIME Study. Examples of official sources include institution websites or ClinicalTrials.gov.\n"
        "7) clinicaltrials_gov_identifier: The ClinicalTrials.gov identifier mentioned for the PRIME Study (e.g., NCT06429735). If not mentioned, return null.\n"
        "Rules:\n"
        "- Only extract URLs explicitly present in the answer text. Do not invent or infer URLs.\n"
        "- If a field is missing, return null (or an empty list for official_reference_urls).\n"
        "- Do not add information not present in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _build_nct_url(nct_id: Optional[str]) -> Optional[str]:
    """Return the ClinicalTrials.gov study URL if an identifier is available; fallback to the expected PRIME Study id."""
    if nct_id and isinstance(nct_id, str) and nct_id.strip():
        return f"https://clinicaltrials.gov/study/{nct_id.strip()}"
    # Provide a default official trial page for verification if not present in the answer
    return f"https://clinicaltrials.gov/study/{EXPECTED_NCT_ID}"


def _unique_urls(urls: List[str]) -> List[str]:
    """Deduplicate URLs while preserving order."""
    seen = set()
    result = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree_and_run(
    evaluator: Evaluator,
    root_node,
    extracted: ClosestSiteExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """

    # Prepare consolidated sources: use answer-provided official URLs; also include ClinicalTrials.gov if available/known
    answer_sources = extracted.official_reference_urls or []
    trial_url = _build_nct_url(extracted.clinicaltrials_gov_identifier)
    combined_sources = _unique_urls(answer_sources + ([trial_url] if trial_url else []))

    # 1) Closest site identification (Critical leaf under sequential root)
    closest_leaf = evaluator.add_leaf(
        id="Closest_Site_Identification",
        desc="Correctly identify the geographically closest U.S. PRIME Study site to Dallas, TX (facility name plus city/state)",
        parent=root_node,
        critical=True,
    )
    closest_claim = (
        f"Among the U.S. locations listed for the Neuralink PRIME Study, the location closest to Dallas, Texas "
        f"is {extracted.facility_name or '[missing facility]'} in {extracted.city or '[missing city]'}, {extracted.state or '[missing state]'}."
    )
    await evaluator.verify(
        claim=closest_claim,
        node=closest_leaf,
        sources=combined_sources if combined_sources else None,
        additional_instruction=(
            "Use the official trial 'Locations' information if available (e.g., ClinicalTrials.gov) to identify all U.S. sites. "
            "Judge geographic closeness to Dallas, TX using approximate distances between major U.S. cities. "
            "Pass if the identified site is plausibly the nearest to Dallas among listed U.S. locations. "
            "Allow reasonable judgment when distances are similar."
        ),
    )

    # 2) Required site details (Critical parallel group)
    details_group = evaluator.add_parallel(
        id="Required_Site_Details",
        desc="Provide all required details for the identified closest site",
        parent=root_node,
        critical=True,
    )

    # 2.1 Complete Physical Address (Critical leaf)
    address_leaf = evaluator.add_leaf(
        id="Complete_Physical_Address",
        desc="Provide the complete physical address of the facility",
        parent=details_group,
        critical=True,
    )
    address_claim = (
        f"The complete physical address for {extracted.facility_name or '[missing facility]'} in "
        f"{extracted.city or '[missing city]'}, {extracted.state or '[missing state]'} is: "
        f"{extracted.complete_address or '[missing address]'}."
    )
    await evaluator.verify(
        claim=address_claim,
        node=address_leaf,
        sources=combined_sources if combined_sources else None,
        additional_instruction=(
            "Verify the address from the official source page (institution site or ClinicalTrials.gov). "
            "Accept minor formatting variations (e.g., 'St' vs 'Street', punctuation, ZIP+4). "
            "Reject if the page does not support the provided address."
        ),
    )

    # 2.2 Main Contact Phone Number (Critical leaf)
    phone_leaf = evaluator.add_leaf(
        id="Main_Contact_Phone_Number",
        desc="Provide the main contact phone number for the facility",
        parent=details_group,
        critical=True,
    )
    phone_claim = (
        f"The main contact phone number for {extracted.facility_name or '[missing facility]'} is "
        f"{extracted.main_phone_number or '[missing phone number]'}."
    )
    await evaluator.verify(
        claim=phone_claim,
        node=phone_leaf,
        sources=combined_sources if combined_sources else None,
        additional_instruction=(
            "Check the contact information on the official page (e.g., 'Contacts and Locations', 'Contact', or 'Phone'). "
            "Accept minor formatting differences (e.g., parentheses, hyphens, spaces). "
            "Reject if the page does not support the provided phone number."
        ),
    )

    # 2.3 Official Reference URL (Critical leaf)
    reference_leaf = evaluator.add_leaf(
        id="Official_Reference_URL",
        desc="Provide an official reference URL (e.g., institution site or ClinicalTrials.gov) that documents the site's participation in the PRIME Study",
        parent=details_group,
        critical=True,
    )
    reference_claim = (
        f"This official webpage documents that {extracted.facility_name or '[missing facility]'} in "
        f"{extracted.city or '[missing city]'}, {extracted.state or '[missing state]'} is a participating site "
        f"in Neuralink's PRIME Study."
    )
    # Prefer verifying against all provided official URLs (and trial page if included)
    await evaluator.verify(
        claim=reference_claim,
        node=reference_leaf,
        sources=combined_sources if combined_sources else None,
        additional_instruction=(
            "The page should explicitly indicate the facility participates in Neuralink's PRIME Study "
            "(e.g., mentions 'Neuralink', 'PRIME Study', 'NCT06429735', or lists the location under the trial's 'Locations'). "
            "Reject if the page is not official or does not document participation."
        ),
    )

    # 2.4 ClinicalTrials.gov Identifier (Critical leaf)
    nct_leaf = evaluator.add_leaf(
        id="ClinicalTrialsGov_Identifier",
        desc="Correctly identify the PRIME Study using the ClinicalTrials.gov identifier NCT06429735",
        parent=details_group,
        critical=True,
    )
    nct_claim = "The ClinicalTrials.gov identifier for Neuralink's PRIME Study is NCT06429735."
    nct_sources: List[str] = []
    if trial_url:
        nct_sources.append(trial_url)
    # Also include any official URLs from answer, in case they reference the NCT ID explicitly
    if answer_sources:
        nct_sources.extend(answer_sources)
    nct_sources = _unique_urls(nct_sources)
    await evaluator.verify(
        claim=nct_claim,
        node=nct_leaf,
        sources=nct_sources if nct_sources else None,
        additional_instruction=(
            "Confirm on ClinicalTrials.gov that the PRIME Study identifier is NCT06429735. "
            "If verifying via institution pages, accept only if NCT06429735 is explicitly mentioned."
        ),
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
    Evaluate an answer for the Neuralink PRIME Study closest site to Dallas, TX task.
    """

    # Initialize evaluator with sequential aggregation (identify first, then details)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract closest site information from the answer
    extracted_site = await evaluator.extract(
        prompt=prompt_extract_site_info(),
        template_class=ClosestSiteExtraction,
        extraction_name="closest_site_info",
    )

    # Add ground truth/context info for transparency
    evaluator.add_ground_truth({
        "user_location": "Dallas, Texas",
        "expected_clinicaltrials_gov_identifier": EXPECTED_NCT_ID,
        "task_focus": "Identify closest U.S. PRIME Study site and verify facility address, phone, and official documentation",
    })

    # Build verification tree and run checks
    await build_verification_tree_and_run(evaluator, root, extracted_site)

    # Return standard summary
    return evaluator.get_summary()