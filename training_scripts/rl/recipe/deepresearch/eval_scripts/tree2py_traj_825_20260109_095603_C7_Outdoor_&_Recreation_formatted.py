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
TASK_ID = "fl_public_campground_eval"
TASK_DESCRIPTION = """
I'm planning an extended RV camping trip to Florida and need to find a public campground that can accommodate my family's specific needs. We require a campground that offers full hookups (water, electricity, and sewer) with 50-amp electrical service for our 38-foot RV. The campground must have ADA-accessible campsites since my father uses a wheelchair, and it must be pet-friendly as we're bringing our two dogs. For family recreation, we need an on-site swimming pool and playground facilities. Essential amenities include bathhouse facilities with showers, laundry facilities, and an on-site camp store. Each campsite should have a picnic table and fire ring. Finally, the campground must accept online reservations. Can you identify a Florida public campground that meets all these requirements and provide the reference URL for verification?
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CampgroundSelection(BaseModel):
    """
    Extracted campground identification and sources from the agent's answer.
    """
    campground_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_campground_selection() -> str:
    return """
    Extract the single Florida public campground identified in the answer and all reference URLs cited for verification.

    Return a JSON object with:
    - campground_name: The name of the identified campground. If multiple campgrounds are listed, pick the first one the answer recommends and return only that name.
    - reference_urls: An array of URLs that the answer provides as references for the identified campground. Include official park pages, county/city pages, .gov pages, and reservation portals (e.g., ReserveAmerica/Florida State Parks) that are clearly tied to this campground. If the answer provides no URLs, return an empty array.

    STRICT RULES:
    - Only extract a single campground_name.
    - Extract only URLs explicitly present in the answer (including markdown links). Do not invent or infer URLs.
    - Ensure URLs are complete and valid. If a URL is missing a protocol, prepend http://
    """


# --------------------------------------------------------------------------- #
# Helper for amenity verification                                             #
# --------------------------------------------------------------------------- #
def _amenity_claims_and_instructions(campground_name: Optional[str]) -> List[Dict[str, str]]:
    name_part = f" at {campground_name}" if campground_name else ""
    return [
        {
            "id": "full_hookups_per_site",
            "desc": "Each individual RV site offers full hookups including water, electricity, and sewer connections",
            "claim": f"RV campsites{name_part} offer full hookups including water, electricity, and sewer connections.",
            "ins": "Confirm via the official amenities or site details section. Accept phrases such as 'full hookups', 'full hook‑up', or 'W/E/S'."
        },
        {
            "id": "fifty_amp_service",
            "desc": "RV sites offer 50-amp electrical service",
            "claim": f"RV sites{name_part} offer 50‑amp electrical service.",
            "ins": "Look for '50 amp', '50A', or '50-amp service' mentioned for the RV sites."
        },
        {
            "id": "rv_length_accommodation",
            "desc": "The campground can accommodate an RV length consistent with the requirement (up to 40 feet, which covers a 38-foot RV)",
            "claim": f"The RV sites{name_part} can accommodate RVs at least 38–40 feet in length.",
            "ins": "Verify maximum RV length limits from the webpage. Accept wording indicating 40 ft or higher."
        },
        {
            "id": "ada_accessible_campsites",
            "desc": "ADA-compliant campsites suitable for wheelchair access are available",
            "claim": f"ADA‑accessible or wheelchair‑accessible campsites{name_part} are available.",
            "ins": "Accept 'accessible campsites', 'ADA campsites', or wording indicating wheelchair accessibility."
        },
        {
            "id": "pet_friendly_with_leash_rules",
            "desc": "The campground is pet-friendly for dogs and enforces leash restrictions (or equivalent leash rule)",
            "claim": f"The campground{name_part} is pet‑friendly for dogs and enforces leash restrictions.",
            "ins": "Look for 'pets allowed' plus leash rules (e.g., 6‑foot leash requirement)."
        },
        {
            "id": "on_site_swimming_pool",
            "desc": "An on-site swimming pool is available for campers",
            "claim": f"An on‑site swimming pool{name_part} is available.",
            "ins": "Confirm presence of a swimming pool on‑site. Accept synonyms like 'pool'."
        },
        {
            "id": "playground_facilities",
            "desc": "Playground facilities or equipment are available for children",
            "claim": f"Playground facilities{name_part} are available.",
            "ins": "Verify a playground or play area is present."
        },
        {
            "id": "bathhouse_restrooms_and_showers",
            "desc": "Bathhouse facilities are available and include both restrooms and showers",
            "claim": f"Bathhouse facilities{name_part} include restrooms and showers.",
            "ins": "Accept mentions of restrooms and showers together, or 'bathhouse with showers'."
        },
        {
            "id": "laundry_on_site",
            "desc": "Laundry facilities are available on-site",
            "claim": f"On‑site laundry facilities{name_part} are available.",
            "ins": "Look for 'laundry' or 'laundromat' available at the campground."
        },
        {
            "id": "camp_store_sells_supplies",
            "desc": "An on-site camp store is present and sells camping supplies",
            "claim": f"An on‑site camp store{name_part} is present and sells supplies or convenience items.",
            "ins": "Accept any on‑site camp store that sells items like groceries, firewood, camping goods, or general supplies."
        },
        {
            "id": "picnic_table_per_site",
            "desc": "Individual campsites are equipped with picnic tables",
            "claim": f"Individual campsites{name_part} include picnic tables.",
            "ins": "Look for wording like 'each site has a picnic table' or similar."
        },
        {
            "id": "fire_ring_or_equivalent_per_site",
            "desc": "Individual campsites include fire rings or authorized burning containers for campfires",
            "claim": f"Individual campsites{name_part} include fire rings, fire pits, or authorized burning containers for campfires.",
            "ins": "Accept 'fire ring', 'fire pit', or equivalent authorized containers."
        },
        {
            "id": "online_reservations",
            "desc": "Reservations can be made through an online booking system",
            "claim": f"Reservations{name_part} can be made online through an online booking system.",
            "ins": "Look for an online booking link or 'Reserve' button (e.g., ReserveAmerica or Florida State Parks reservation portal)."
        },
    ]


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_identification_nodes(
    evaluator: Evaluator,
    root_node,
    selection: CampgroundSelection
) -> None:
    """
    Build and verify the identification nodes at the root level:
    1) Identify a single public Florida campground (critical verification by URL).
    2) Reference URL provided (critical existence check).
    """
    # 1) Identify a single public Florida campground
    identify_node = evaluator.add_leaf(
        id="identify_florida_public_campground",
        desc="Identifies a single campground that is both public and located in Florida",
        parent=root_node,
        critical=True
    )
    cg_name = selection.campground_name or ""
    identify_claim = (
        f"The identified campground '{cg_name}' is a public campground located in the state of Florida."
    )
    await evaluator.verify(
        claim=identify_claim,
        node=identify_node,
        sources=selection.reference_urls,
        additional_instruction=(
            "Confirm the campground is in Florida and publicly owned/managed (e.g., state, county, city, or federal). "
            "Accept Florida State Parks, county parks, or city parks as public campgrounds. Use only the provided URLs."
        ),
    )

    # 2) Reference URL provided (existence check)
    evaluator.add_custom_node(
        result=(len(selection.reference_urls) >= 1),
        id="reference_url_provided",
        desc="Provides at least one reference URL relevant to the identified campground for verification",
        parent=root_node,
        critical=True
    )


async def build_amenities_nodes(
    evaluator: Evaluator,
    root_node,
    selection: CampgroundSelection
) -> None:
    """
    Build and verify the parallel amenities node with all critical leaf checks.
    Uses batch verification to avoid inter-sibling precondition interference.
    """
    amenities_node = evaluator.add_parallel(
        id="meets_all_amenity_and_site_requirements",
        desc="The identified campground meets all specified amenity and campsite requirements",
        parent=root_node,
        critical=True
    )

    claims = _amenity_claims_and_instructions(selection.campground_name)

    # Create leaves and prepare batch verification tuples
    verify_items: List[
        tuple[str, List[str], Any, Optional[str]]
    ] = []

    for item in claims:
        leaf = evaluator.add_leaf(
            id=item["id"],
            desc=item["desc"],
            parent=amenities_node,
            critical=True
        )
        claim_text = item["claim"]
        add_ins = item["ins"]
        verify_items.append((claim_text, selection.reference_urls, leaf, add_ins))

    # Perform batch verification in parallel for amenity leaves
    await evaluator.batch_verify(verify_items)


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
    Evaluate an agent's answer for the Florida public campground requirements task.
    """
    # Initialize evaluator with a sequential root to enforce ordering:
    # identification -> reference URL -> amenities checks
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

    # Extract the campground selection and reference URLs from the answer
    selection = await evaluator.extract(
        prompt=prompt_extract_campground_selection(),
        template_class=CampgroundSelection,
        extraction_name="campground_selection",
    )

    # Build and verify identification nodes
    await build_identification_nodes(evaluator, root, selection)

    # Build and verify amenities node
    await build_amenities_nodes(evaluator, root, selection)

    # Optionally record requirement summary for context in the output
    evaluator.add_custom_info(
        info={
            "requirements": [
                "Florida public campground",
                "Full hookups (water, electricity, sewer)",
                "50-amp electrical service",
                "Accommodates 38-foot RV (up to 40 ft)",
                "ADA-accessible campsites",
                "Pet-friendly (leash rules)",
                "On-site swimming pool",
                "Playground",
                "Bathhouse (restrooms & showers)",
                "Laundry facilities",
                "On-site camp store (supplies)",
                "Picnic table per site",
                "Fire ring or equivalent per site",
                "Online reservations"
            ]
        },
        info_type="requirements_summary",
    )

    # Return the final evaluation summary
    return evaluator.get_summary()