import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tucson_luxury_apartment_requirements"
TASK_DESCRIPTION = (
    "Name a specific luxury apartment complex in Tucson, Arizona that satisfies ALL of the following required amenities: "
    "swimming pool, fitness center, pet-friendly, and covered or reserved parking. Additionally, the complex should offer as many "
    "of the following preferred features as possible: hot tub or spa, in-unit washer and dryer, outdoor BBQ or grilling areas, "
    "gated community or controlled access, stainless steel appliances, clubhouse or resident lounge, 9-foot or higher ceilings, "
    "private balconies or patios, and modern kitchen features (such as quartz countertops). Provide the apartment complex name "
    "with supporting reference URLs demonstrating it meets the criteria."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class ApartmentInfoExtraction(BaseModel):
    """
    Structured information extracted from the answer.
    """
    complex_name: Optional[str] = None
    location_text: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_apartment_info() -> str:
    return """
    Extract the key information about the luxury apartment complex mentioned in the answer.

    Required fields:
    1) complex_name: The official name of the apartment complex identified in the answer. Provide the name exactly as written.
    2) location_text: Any text snippet from the answer indicating its location (e.g., "Tucson, AZ" or "Tucson, Arizona"). If absent, return null.
    3) source_urls: An array of all reference URLs explicitly listed or linked in the answer that support the apartment's details (official website, apartments.com, Zillow, other listing/management pages, etc.).
       - Only include URLs that are explicitly present in the answer (plain URLs or markdown links).
       - Return full valid URLs; if protocol is missing, prepend http://
       - Do not invent URLs.

    Return a single JSON object with the exact fields above. If any field is missing, set it to null (for strings) or [] (for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    cleaned = []
    for u in urls:
        if not u:
            continue
        u2 = u.strip()
        if not u2:
            continue
        if u2 not in seen:
            seen.add(u2)
            cleaned.append(u2)
    return cleaned


def _amenity_instruction_pool() -> str:
    return (
        "Verify that the page explicitly indicates a swimming pool at the property. Accept synonyms like 'pool', "
        "'resort-style pool', 'sparkling pool', or 'swimming pool'."
    )


def _amenity_instruction_fitness() -> str:
    return (
        "Verify that the page indicates a fitness center/gym at the property. Accept synonyms like 'fitness center', 'gym', "
        "'fitness studio', or 'fitness room'."
    )


def _amenity_instruction_pet_friendly() -> str:
    return (
        "Verify that pets are allowed at the property (e.g., 'pet-friendly', 'cats and dogs allowed', 'pets welcome'). "
        "It's acceptable if there are breed/weight restrictions or fees; as long as pets are allowed, it counts."
    )


def _amenity_instruction_parking() -> str:
    return (
        "Verify that the property offers either covered parking or reserved/assigned parking. Accept 'covered parking', 'carports', "
        "'garages', 'assigned parking', or 'reserved parking'. Generic mentions like 'ample parking' alone do not count."
    )


def _amenity_instruction_spa() -> str:
    return (
        "Verify a hot tub or spa is available. Accept 'spa', 'hot tub', or 'jacuzzi'."
    )


def _amenity_instruction_inunit_wd() -> str:
    return (
        "Verify that apartment homes have in-unit washer and dryer. Phrases like 'in-unit washer/dryer', 'washer and dryer included', "
        "or 'full-size washer and dryer' count. 'Washer/dryer hookups' only does NOT count."
    )


def _amenity_instruction_bbq() -> str:
    return (
        "Verify that the property offers outdoor BBQ or grilling areas. Accept 'BBQ area', 'grilling stations', 'outdoor grills', or 'barbecue area'."
    )


def _amenity_instruction_gated() -> str:
    return (
        "Verify that the community is gated or has controlled/secured access. Accept 'gated community', 'controlled access', 'limited-entry access', "
        "'access gates', or 'secure entry'."
    )


def _amenity_instruction_stainless() -> str:
    return (
        "Verify that apartment homes include stainless steel kitchen appliances. Accept 'stainless steel appliances' or close synonyms."
    )


def _amenity_instruction_clubhouse() -> str:
    return (
        "Verify that the property has a clubhouse, resident lounge, or community lounge/gathering space."
    )


def _amenity_instruction_high_ceilings() -> str:
    return (
        "Verify that apartments feature 9-foot or higher ceilings. Accept explicit numeric mentions like '9 ft' or 'nine-foot'. "
        "Mentions of 'vaulted ceilings' alone should not count unless a 9-foot-or-higher numeric height is also stated."
    )


def _amenity_instruction_balcony_patio() -> str:
    return (
        "Verify that apartments include private balconies or patios. Accept 'private patio', 'private balcony', or 'patio/balcony'."
    )


def _amenity_instruction_modern_kitchen() -> str:
    return (
        "Verify that apartments have modern kitchen features such as 'quartz countertops', 'granite countertops', or equivalent "
        "contemporary finishes explicitly stated."
    )


def _location_instruction() -> str:
    return (
        "Verify that the property is located in Tucson, Arizona. Accept 'Tucson, AZ' or 'Tucson, Arizona' or an address clearly in Tucson. "
        "Do not accept surrounding cities (e.g., Oro Valley) unless the page explicitly says the property is in Tucson."
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_and_verify_must_haves(
    evaluator: Evaluator,
    parent_node,
    complex_name: str,
    urls: List[str],
) -> None:
    """
    Build the required amenities subtree (critical) and verify each leaf against the provided URLs.
    """
    must_node = evaluator.add_parallel(
        id="must_haves",
        desc="All required amenities must be present",
        parent=parent_node,
        critical=True,
    )

    # Create leaf nodes
    leaves: List[Dict[str, Any]] = []

    # Location in Tucson, AZ
    node_loc = evaluator.add_leaf(
        id="Located_in_Tucson_AZ",
        desc="The apartment complex must be located in Tucson, Arizona",
        parent=must_node,
        critical=True,
    )
    leaves.append({
        "node": node_loc,
        "claim": f"{complex_name} is located in Tucson, Arizona.",
        "instruction": _location_instruction(),
    })

    # Swimming Pool
    node_pool = evaluator.add_leaf(
        id="Has_Swimming_Pool",
        desc="The complex has a swimming pool (resort-style or standard pool)",
        parent=must_node,
        critical=True,
    )
    leaves.append({
        "node": node_pool,
        "claim": f"{complex_name} offers a swimming pool.",
        "instruction": _amenity_instruction_pool(),
    })

    # Fitness Center
    node_fitness = evaluator.add_leaf(
        id="Has_Fitness_Center",
        desc="The complex has a fitness center",
        parent=must_node,
        critical=True,
    )
    leaves.append({
        "node": node_fitness,
        "claim": f"{complex_name} has a fitness center or gym.",
        "instruction": _amenity_instruction_fitness(),
    })

    # Pet Friendly
    node_pet = evaluator.add_leaf(
        id="Pet_Friendly",
        desc="The complex is pet-friendly",
        parent=must_node,
        critical=True,
    )
    leaves.append({
        "node": node_pet,
        "claim": f"{complex_name} is pet-friendly and allows residents to have pets.",
        "instruction": _amenity_instruction_pet_friendly(),
    })

    # Covered or Reserved Parking
    node_parking = evaluator.add_leaf(
        id="Has_Covered_Reserved_Parking",
        desc="The complex offers covered or reserved/assigned parking",
        parent=must_node,
        critical=True,
    )
    leaves.append({
        "node": node_parking,
        "claim": f"{complex_name} offers covered parking or reserved/assigned parking for residents.",
        "instruction": _amenity_instruction_parking(),
    })

    # Batch verify all required leaves
    claims_and_sources = []
    for item in leaves:
        claims_and_sources.append((
            item["claim"],
            urls,
            item["node"],
            item["instruction"]
        ))
    await evaluator.batch_verify(claims_and_sources)


async def build_and_verify_nice_to_haves(
    evaluator: Evaluator,
    parent_node,
    complex_name: str,
    urls: List[str],
) -> None:
    """
    Build the preferred features subtree (non-critical) and verify each leaf against the provided URLs.
    """
    nice_node = evaluator.add_parallel(
        id="nice_to_haves",
        desc="Preferred features (partial credit)",
        parent=parent_node,
        critical=False,
    )

    # Prepare leaf nodes and their verification specs
    leaves_specs = [
        # (id, desc, claim, instruction)
        ("Has_Hot_Tub_Spa", "The complex has a hot tub or spa",
         f"{complex_name} offers a hot tub or spa.", _amenity_instruction_spa()),
        ("Has_In_Unit_Washer_Dryer", "In-unit washer and dryer available in apartments",
         f"{complex_name} apartments include in-unit washer and dryer.", _amenity_instruction_inunit_wd()),
        ("Has_Outdoor_BBQ_Grilling", "Outdoor BBQ or grilling areas available",
         f"{complex_name} offers outdoor BBQ or grilling areas.", _amenity_instruction_bbq()),
        ("Has_Gated_Controlled_Access", "Gated community or controlled access",
         f"{complex_name} is a gated community or provides controlled/secured access.", _amenity_instruction_gated()),
        ("Has_Stainless_Steel_Appliances", "Stainless steel appliances included",
         f"{complex_name} apartments include stainless steel kitchen appliances.", _amenity_instruction_stainless()),
        ("Has_Clubhouse_Lounge", "Clubhouse or resident lounge available",
         f"{complex_name} provides a clubhouse or resident lounge/community lounge.", _amenity_instruction_clubhouse()),
        ("Has_High_Ceilings", "9-foot or higher ceilings",
         f"{complex_name} apartments feature 9-foot or higher ceilings.", _amenity_instruction_high_ceilings()),
        ("Has_Private_Balcony_Patio", "Private balconies or patios included",
         f"{complex_name} apartments include private balconies or patios.", _amenity_instruction_balcony_patio()),
        ("Has_Modern_Kitchen_Features", "Modern kitchen features (e.g., quartz countertops)",
         f"{complex_name} apartments have modern kitchen features such as quartz or granite countertops.", _amenity_instruction_modern_kitchen()),
    ]

    # Create leaves and queue verifications
    claims_and_sources = []
    for leaf_id, desc, claim, instruction in leaves_specs:
        node = evaluator.add_leaf(
            id=leaf_id,
            desc=desc,
            parent=nice_node,
            critical=False
        )
        claims_and_sources.append((claim, urls, node, instruction))

    # Batch verify all preferred leaves
    await evaluator.batch_verify(claims_and_sources)


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
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Tucson luxury apartment requirements task.
    """
    # 1) Initialize evaluator and root (non-critical to allow partial credit via subtrees)
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

    # 2) Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_apartment_info(),
        template_class=ApartmentInfoExtraction,
        extraction_name="apartment_info",
    )

    # Clean up URLs
    all_urls = _dedupe_urls(extracted.source_urls or [])

    # 3) Record ground truth / rubric info for transparency
    evaluator.add_ground_truth({
        "required": [
            "Located in Tucson, AZ",
            "Swimming Pool",
            "Fitness Center",
            "Pet-Friendly",
            "Covered or Reserved Parking"
        ],
        "preferred": [
            "Hot Tub/Spa",
            "In-unit Washer/Dryer",
            "Outdoor BBQ/Grilling",
            "Gated/Controlled Access",
            "Stainless Steel Appliances",
            "Clubhouse/Resident Lounge",
            "9-foot or Higher Ceilings",
            "Private Balconies/Patios",
            "Modern Kitchen Features (e.g., quartz countertops)"
        ]
    })

    evaluator.add_custom_info(
        {
            "extracted_complex_name": extracted.complex_name,
            "extracted_location_text": extracted.location_text,
            "source_url_count": len(all_urls)
        },
        info_type="extraction_meta"
    )

    # 4) Build verification tree

    # 4.1 Preconditions (Critical): ensure the complex name and at least one URL are provided
    pre_node = evaluator.add_parallel(
        id="preconditions",
        desc="Preconditions: complex identified and sources provided",
        parent=root,
        critical=True
    )

    name_ok = bool(extracted.complex_name and extracted.complex_name.strip())
    urls_ok = len(all_urls) > 0

    evaluator.add_custom_node(
        result=name_ok,
        id="complex_name_provided",
        desc="Apartment complex name is provided in the answer",
        parent=pre_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=urls_ok,
        id="source_urls_provided",
        desc="At least one reference URL is provided in the answer",
        parent=pre_node,
        critical=True
    )

    # If name or URLs are missing, subsequent verifications will auto-skip due to critical preconditions.
    complex_name = extracted.complex_name or ""

    # 4.2 Required amenities (Critical subtree)
    await build_and_verify_must_haves(
        evaluator=evaluator,
        parent_node=root,
        complex_name=complex_name,
        urls=all_urls
    )

    # 4.3 Preferred features (Non-critical subtree)
    await build_and_verify_nice_to_haves(
        evaluator=evaluator,
        parent_node=root,
        complex_name=complex_name,
        urls=all_urls
    )

    # 5) Return evaluation summary
    return evaluator.get_summary()