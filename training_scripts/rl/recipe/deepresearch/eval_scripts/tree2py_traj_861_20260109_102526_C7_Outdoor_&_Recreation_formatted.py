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
TASK_ID = "az_rv_campground_requirements"
TASK_DESCRIPTION = (
    "I am planning an extended RV trip to Arizona and need to find a campground that meets all of my family's requirements. "
    "Identify one campground or RV resort in Arizona that satisfies ALL of the following criteria: "
    "(1) Provides full hookups (water, electricity, and sewer connections) at individual RV campsites, "
    "(2) Offers electrical service with either 30-amp or 50-amp hookups (or both), "
    "(3) Has restroom facilities available on-site, "
    "(4) Has shower facilities available on-site, "
    "(5) Has laundry facilities (washers and dryers) available on-site, "
    "(6) Has a pet-friendly policy that allows dogs, "
    "(7) Provides WiFi or internet connectivity, "
    "(8) Has a swimming pool facility, "
    "(9) Has a playground or playground equipment area, "
    "(10) Provides picnic tables at individual campsites, "
    "(11) Provides fire rings or fire pits at individual campsites, "
    "(12) Has an RV dump station available on-site, "
    "(13) Offers pull-through RV sites, and "
    "(14) Is big rig friendly and can accommodate large RVs. "
    "For your answer, provide the name of the campground, its location/address in Arizona, and include a link to the campground's official website or a reliable source that confirms these amenities."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CampgroundExtraction(BaseModel):
    """Structured information for one identified Arizona campground."""
    name: Optional[str] = None
    location_address: Optional[str] = None
    official_url: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_campground() -> str:
    return (
        "Extract the campground selected in the answer. Return the following fields:\n"
        "1. name: The specific name of the campground or RV resort.\n"
        "2. location_address: The location or street address in Arizona as provided in the answer.\n"
        "3. official_url: The main official website URL for the campground (if explicitly provided). "
        "If multiple URLs are provided, choose the one that appears to be the official website. If none is provided, return null.\n"
        "4. source_urls: An array of all other URLs mentioned in the answer that serve as references or sources confirming amenities. "
        "Exclude the official_url from this list to avoid duplication. If no other URLs are mentioned, return an empty array.\n"
        "Rules:\n"
        "- Extract only URLs explicitly present in the answer (plain links or markdown links). Do not invent URLs.\n"
        "- Normalize URLs and include the protocol (http:// or https://). If missing, prepend http://.\n"
        "- If any field is missing in the answer, set it to null (or empty array for source_urls)."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _collect_sources(extracted: CampgroundExtraction) -> List[str]:
    """Collect and deduplicate all URLs to be used for verification."""
    urls: List[str] = []
    if extracted.official_url and isinstance(extracted.official_url, str):
        urls.append(extracted.official_url.strip())
    for u in extracted.source_urls:
        if isinstance(u, str) and u.strip():
            urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _amenity_claim(amenity_key: str, cg_name: Optional[str]) -> str:
    """Build a human-readable claim for an amenity tied to the campground name."""
    name_part = f" at {cg_name}" if cg_name else ""
    mapping = {
        "Full_Hookups": f"The campground{name_part} provides full hookups (water, electricity, and sewer) at individual RV campsites.",
        "Electrical_Service": f"The campground{name_part} offers 30-amp or 50-amp electrical hookups (or both) at RV sites.",
        "Restroom_Facilities": f"The campground{name_part} has restroom facilities available on-site.",
        "Shower_Facilities": f"The campground{name_part} has shower facilities available on-site.",
        "Laundry_Facilities": f"The campground{name_part} has laundry facilities (washers and dryers) available on-site.",
        "Pet_Friendly": f"The campground{name_part} is pet-friendly and allows dogs.",
        "WiFi_Internet": f"The campground{name_part} provides WiFi or internet connectivity.",
        "Swimming_Pool": f"The campground{name_part} has a swimming pool facility.",
        "Playground": f"The campground{name_part} has a playground or playground equipment area.",
        "Picnic_Tables": f"The campground{name_part} provides picnic tables at individual campsites.",
        "Fire_Rings": f"The campground{name_part} provides fire rings or fire pits at individual campsites.",
        "Dump_Station": f"The campground{name_part} has an RV dump station available on-site.",
        "Pull_Through_Sites": f"The campground{name_part} offers pull-through RV sites.",
        "Big_Rig_Friendly": f"The campground{name_part} is big rig friendly and can accommodate large RVs.",
    }
    return mapping[amenity_key]


def _amenity_instruction(amenity_key: str) -> str:
    """Additional instructions to guide the verifier for each amenity."""
    instructions = {
        "Full_Hookups": "Look for phrases like 'full hookups', 'W/E/S', or 'water/electric/sewer at sites'. The page should clearly indicate hookups include sewer.",
        "Electrical_Service": "Accept mentions of '30 amp', '50 amp', '30/50-amp', or 'electric hookups'. Either 30A or 50A (or both) is sufficient.",
        "Restroom_Facilities": "Accept 'restrooms', 'bathrooms', 'bathhouse'.",
        "Shower_Facilities": "Accept 'showers', 'shower house', 'hot showers'.",
        "Laundry_Facilities": "Accept 'laundry', 'laundry room', 'washers and dryers'.",
        "Pet_Friendly": "Accept 'pets allowed', 'pet-friendly', 'dogs permitted'. Be mindful of any restrictions; at minimum dogs must be allowed.",
        "WiFi_Internet": "Accept 'Wi-Fi', 'WiFi', 'internet access', 'free WiFi'.",
        "Swimming_Pool": "Accept 'pool', 'swimming pool', 'heated pool'.",
        "Playground": "Accept 'playground', 'play area', 'kids playground'.",
        "Picnic_Tables": "Accept 'picnic table at site' or equivalent phrasing that tables are provided at campsites.",
        "Fire_Rings": "Accept 'fire rings', 'fire pits' provided at campsites; consider synonyms.",
        "Dump_Station": "Accept 'dump station' or 'RV dump' available on-site.",
        "Pull_Through_Sites": "Accept 'pull-through', 'pull thru' sites.",
        "Big_Rig_Friendly": "Accept 'big rig friendly', 'big rigs welcome', or explicit mention of accommodating large RVs/long rigs.",
    }
    return instructions[amenity_key]


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extracted: CampgroundExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and run verification checks.
    """
    # Create the rubric main node under root (critical, parallel aggregation)
    req_node = evaluator.add_parallel(
        id="Campground_Requirements",
        desc="The answer identifies a campground in Arizona and provides all required information including name, location, reference URL, and verification that it meets all specified amenity and facility requirements",
        parent=evaluator.root,
        critical=True
    )

    # Critical existence checks (must be present in the answer)
    name_exists = bool(extracted.name and extracted.name.strip())
    evaluator.add_custom_node(
        result=name_exists,
        id="Campground_Name",
        desc="The answer provides the specific name of the campground or RV resort",
        parent=req_node,
        critical=True
    )

    location_exists = bool(extracted.location_address and extracted.location_address.strip())
    evaluator.add_custom_node(
        result=location_exists,
        id="Location_Address",
        desc="The answer provides the location or address of the campground in Arizona",
        parent=req_node,
        critical=True
    )

    sources = _collect_sources(extracted)
    ref_url_exists = len(sources) > 0
    evaluator.add_custom_node(
        result=ref_url_exists,
        id="Reference_URL",
        desc="The answer includes a link to the campground's official website or a reliable source that confirms the amenities",
        parent=req_node,
        critical=True
    )

    # Prepare amenity leaf nodes
    amenity_nodes_desc_map = {
        "Full_Hookups": "The campground provides full hookups (water, electricity, and sewer) at individual campsites",
        "Electrical_Service": "The campground provides either 30-amp or 50-amp electrical service (or both) at RV sites",
        "Restroom_Facilities": "The campground has restroom facilities available on-site",
        "Shower_Facilities": "The campground has shower facilities available on-site",
        "Laundry_Facilities": "The campground has laundry facilities (washers and dryers) available on-site",
        "Pet_Friendly": "The campground has a pet-friendly policy allowing dogs",
        "WiFi_Internet": "The campground provides WiFi or internet connectivity",
        "Swimming_Pool": "The campground has a swimming pool facility available",
        "Playground": "The campground has playground equipment or a playground area",
        "Picnic_Tables": "The campground provides picnic tables at campsites",
        "Fire_Rings": "The campground provides fire rings or fire pits at campsites",
        "Dump_Station": "The campground has an RV dump station available on-site",
        "Pull_Through_Sites": "The campground offers pull-through RV sites",
        "Big_Rig_Friendly": "The campground is big rig friendly and can accommodate large RVs",
    }

    # Create leaf nodes for amenities
    leaf_nodes = {}
    for key, desc in amenity_nodes_desc_map.items():
        leaf_nodes[key] = evaluator.add_leaf(
            id=key,
            desc=desc,
            parent=req_node,
            critical=True
        )

    # If we have no sources, the subsequent verifications will be auto-skipped due to the failed Reference_URL precondition
    # Build claims and batch verify
    claims_and_sources: List[tuple[str, List[str] | None, Any, Optional[str]]] = []
    for key, node in leaf_nodes.items():
        claim = _amenity_claim(key, extracted.name)
        add_ins = _amenity_instruction(key)
        claims_and_sources.append((claim, sources if sources else None, node, add_ins))

    # Run verifications (parallelized for efficiency)
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate the answer for the Arizona RV campground requirements task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root holds a single critical rubric node
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

    # Extract campground info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_campground(),
        template_class=CampgroundExtraction,
        extraction_name="campground_info"
    )

    # Optionally record custom info for debugging
    evaluator.add_custom_info(
        info={
            "campground_name": extracted.name,
            "location_address": extracted.location_address,
            "official_url": extracted.official_url,
            "source_urls": extracted.source_urls
        },
        info_type="extracted_fields",
        info_name="extracted_campground_fields"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extracted)

    # Return standardized summary
    return evaluator.get_summary()