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
TASK_ID = "fl_state_park_south_florida"
TASK_DESCRIPTION = (
    "Find one Florida state park located in South Florida (Miami-Dade, Monroe, Broward, Palm Beach, or Collier counties) "
    "that offers full-facility camping with water and electric hookups, allows pets in the campground, and provides direct beach access. "
    "For the park you identify, provide the following information: (1) The official park name, (2) The complete physical address, "
    "(3) The total number of campsites available, (4) The nightly camping fee for full-facility sites, "
    "(5) A link to the park's page on the official Florida State Parks reservation system, (6) Confirmation that the park meets Florida State Parks pet policy, "
    "and (7) Description of the beach access facilities. Include reference URLs from official sources for verification."
)

ALLOWED_COUNTIES = [
    "Miami-Dade", "Miami Dade",
    "Monroe",
    "Broward",
    "Palm Beach",
    "Collier",
]

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ParkDetailsExtraction(BaseModel):
    park_name: Optional[str] = None
    address: Optional[str] = None
    county: Optional[str] = None
    total_campsites: Optional[str] = None
    nightly_fee: Optional[str] = None
    reservation_url: Optional[str] = None
    pet_policy_statement: Optional[str] = None
    beach_access_description: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_park_details() -> str:
    return (
        "Extract details for one specific Florida State Park identified in the answer. Extract the following fields strictly from the answer text:\n"
        "- park_name: The official Florida State Park name.\n"
        "- address: The complete physical street address of the park (include city and ZIP if available).\n"
        "- county: The county the park is in (e.g., Miami-Dade, Monroe, Broward, Palm Beach, or Collier). If not explicitly given, infer from the answer only if clearly stated.\n"
        "- total_campsites: The total number of campsites available (use the exact phrasing or number from the answer).\n"
        "- nightly_fee: The nightly camping fee for full-facility sites (use the exact price or range as shown in the answer).\n"
        "- reservation_url: The URL pointing to the park's page on the official Florida State Parks reservation system (e.g., reserve.floridastateparks.org or Florida State Parks’ official reservation vendor page). Extract only if explicitly present in the answer.\n"
        "- pet_policy_statement: A statement indicating pets are allowed in the campground under Florida State Parks pet policy (if explicitly stated in the answer).\n"
        "- beach_access_description: A short description of the beach access facilities (e.g., direct beach access, boardwalk to beach, beachside camping area).\n"
        "- reference_urls: A list of all reference URLs cited in the answer for verification. Include official Florida State Parks website URLs (floridastateparks.org) or official reservation system URLs if present. Do not fabricate URLs.\n"
        "Return null for any field that is not present in the answer text. For URLs, extract only valid URLs presented in the answer (including markdown link targets)."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _clean_url_list(urls: List[str]) -> List[str]:
    seen = set()
    cleaned = []
    for u in urls:
        if not u:
            continue
        s = u.strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            cleaned.append(s)
    return cleaned


def _build_sources(extracted: ParkDetailsExtraction) -> List[str]:
    urls: List[str] = []
    if extracted.reservation_url:
        urls.append(extracted.reservation_url)
    urls.extend(extracted.reference_urls or [])
    return _clean_url_list(urls)


def _has_text(value: Optional[str]) -> bool:
    return bool(value and value.strip())


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def add_constraints_verifications(
    evaluator: Evaluator,
    parent_node,
    extracted: ParkDetailsExtraction,
) -> None:
    """
    Create and verify the ParkMeetsAllConstraints subtree.
    """
    sources = _build_sources(extracted)

    constraints_node = evaluator.add_parallel(
        id="ParkMeetsAllConstraints",
        desc="The selected park satisfies all stated eligibility constraints (official FL State Park, correct county region, camping/pets/beach requirements)",
        parent=parent_node,
        critical=True,
    )

    # OfficialPark
    official_leaf = evaluator.add_leaf(
        id="OfficialPark",
        desc="The identified location is an official Florida State Park managed by the Florida State Parks system",
        parent=constraints_node,
        critical=True,
    )
    official_claim = (
        f"The park '{extracted.park_name or ''}' is an official Florida State Park managed by the Florida State Parks system."
    )
    await evaluator.verify(
        claim=official_claim,
        node=official_leaf,
        sources=sources,
        additional_instruction=(
            "Check the provided official sources (e.g., floridastateparks.org or official reservation system pages) "
            "to confirm that this is an official Florida State Park. The page should clearly indicate Florida State Parks branding or affiliation."
        ),
    )

    # SouthFloridaLocation
    south_fl_leaf = evaluator.add_leaf(
        id="SouthFloridaLocation",
        desc="The park is located in South Florida (Miami-Dade, Monroe, Broward, Palm Beach, or Collier counties)",
        parent=constraints_node,
        critical=True,
    )
    county_text = extracted.county or ""
    south_fl_claim = (
        f"The park is located in one of these counties: Miami-Dade, Monroe, Broward, Palm Beach, or Collier. "
        f"Stated county in the answer: '{county_text}'."
    )
    await evaluator.verify(
        claim=south_fl_claim,
        node=south_fl_leaf,
        sources=sources,
        additional_instruction=(
            "Use the official source pages to verify the county for the park. "
            "If the county is not explicitly stated, derive it from the address/location shown on the official page. "
            "Minor spelling variations (e.g., 'Miami Dade' vs 'Miami-Dade') should be considered equivalent."
        ),
    )

    # FullFacilityCamping
    camping_leaf = evaluator.add_leaf(
        id="FullFacilityCamping",
        desc="The park offers full-facacility camping that includes water hookups, electric hookups, and access to restrooms/showers",
        parent=constraints_node,
        critical=True,
    )
    camping_claim = (
        "The park offers full-facility camping including water hookups, electric hookups, and access to restrooms/showers."
    )
    await evaluator.verify(
        claim=camping_claim,
        node=camping_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm the campground amenities on official pages show water and electric hookups and access to restrooms/showers. "
            "Accept equivalent wording such as 'utility hookups', 'electricity', 'bathhouse', or 'shower facilities'."
        ),
    )

    # PetFriendly
    pet_leaf = evaluator.add_leaf(
        id="PetFriendly",
        desc="The park allows pets in the campground in accordance with Florida State Parks pet policy",
        parent=constraints_node,
        critical=True,
    )
    pet_claim = (
        "Pets are allowed in the campground at this park under the Florida State Parks pet policy."
    )
    await evaluator.verify(
        claim=pet_claim,
        node=pet_leaf,
        sources=sources,
        additional_instruction=(
            "Verify pets are allowed in the campground per Florida State Parks policy. "
            "It is acceptable if pets are restricted in certain areas (e.g., beach) but allowed in the campground."
        ),
    )

    # BeachAccess
    beach_leaf = evaluator.add_leaf(
        id="BeachAccess",
        desc="The park provides direct beach access or coastal beach facilities",
        parent=constraints_node,
        critical=True,
    )
    beach_claim = (
        "This park provides direct beach access or coastal beach facilities."
    )
    await evaluator.verify(
        claim=beach_claim,
        node=beach_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that the official sources explicitly show park access to a beach (e.g., oceanfront beach, coastal shoreline) "
            "and that visitors have direct access (e.g., boardwalk, path, or adjacent sand beach)."
        ),
    )


async def add_required_outputs_verifications(
    evaluator: Evaluator,
    parent_node,
    extracted: ParkDetailsExtraction,
) -> None:
    """
    Create the RequiredOutputsProvided subtree with existence checks and specific URL verification for the reservation page.
    """
    outputs_node = evaluator.add_parallel(
        id="RequiredOutputsProvided",
        desc="The answer provides all required information fields for the identified park",
        parent=parent_node,
        critical=True,
    )

    # OfficialParkNameProvided (existence)
    evaluator.add_custom_node(
        result=_has_text(extracted.park_name),
        id="OfficialParkNameProvided",
        desc="The official park name is provided",
        parent=outputs_node,
        critical=True,
    )

    # PhysicalAddressProvided (existence)
    evaluator.add_custom_node(
        result=_has_text(extracted.address),
        id="PhysicalAddressProvided",
        desc="The complete physical address of the park is provided",
        parent=outputs_node,
        critical=True,
    )

    # TotalCampsitesProvided (existence)
    evaluator.add_custom_node(
        result=_has_text(extracted.total_campsites),
        id="TotalCampsitesProvided",
        desc="The total number of campsites available at the park is provided",
        parent=outputs_node,
        critical=True,
    )

    # NightlyCampingFeeProvided (existence)
    evaluator.add_custom_node(
        result=_has_text(extracted.nightly_fee),
        id="NightlyCampingFeeProvided",
        desc="The nightly camping fee for full-facility sites is provided",
        parent=outputs_node,
        critical=True,
    )

    # ReservationSystemParkPageLinkProvided (verification of working/official reservation page)
    reservation_leaf = evaluator.add_leaf(
        id="ReservationSystemParkPageLinkProvided",
        desc="A working link to the park's page on the official Florida State Parks reservation system is provided",
        parent=outputs_node,
        critical=True,
    )
    reservation_claim = (
        f"The provided reservation URL is the official reservation system page for the park '{extracted.park_name or ''}', and it is accessible."
    )
    await evaluator.verify(
        claim=reservation_claim,
        node=reservation_leaf,
        sources=extracted.reservation_url,
        additional_instruction=(
            "Open the URL and verify it is an official Florida State Parks reservation page (e.g., reserve.floridastateparks.org "
            "or the official reservation vendor for Florida State Parks) and that it corresponds to the identified park."
        ),
    )

    # PetPolicyConfirmationProvided (existence of confirmation text in the answer)
    evaluator.add_custom_node(
        result=_has_text(extracted.pet_policy_statement),
        id="PetPolicyConfirmationProvided",
        desc="The answer explicitly confirms the park meets Florida State Parks pet policy (pets allowed in the campground under that policy)",
        parent=outputs_node,
        critical=True,
    )

    # BeachAccessFacilitiesDescribed (existence)
    evaluator.add_custom_node(
        result=_has_text(extracted.beach_access_description),
        id="BeachAccessFacilitiesDescribed",
        desc="The beach access facilities are described (not just that beach access exists)",
        parent=outputs_node,
        critical=True,
    )

    # OfficialSourceReferenceURLsIncluded (existence)
    evaluator.add_custom_node(
        result=bool(extracted.reference_urls and len(extracted.reference_urls) > 0),
        id="OfficialSourceReferenceURLsIncluded",
        desc="Reference URL(s) from official sources are included to verify the claims (e.g., official Florida State Parks and/or official reservation system pages)",
        parent=outputs_node,
        critical=True,
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the South Florida Florida State Park camping task.
    """
    # Initialize evaluator with sequential root
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

    # Extraction: get all required fields from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_park_details(),
        template_class=ParkDetailsExtraction,
        extraction_name="park_details",
    )

    # Build the top-level task node (critical sequential)
    find_park_node = evaluator.add_sequential(
        id="FindPark",
        desc="Identify one qualifying Florida State Park in South Florida and provide all required details with official-source references",
        parent=root,
        critical=True,
    )

    # 1) Constraints verification (critical, parallel)
    await add_constraints_verifications(evaluator, find_park_node, extracted)

    # 2) Required outputs provided (critical, parallel)
    await add_required_outputs_verifications(evaluator, find_park_node, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()