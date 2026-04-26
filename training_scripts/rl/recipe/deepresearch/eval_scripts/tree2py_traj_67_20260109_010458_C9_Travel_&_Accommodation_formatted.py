import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_beach_luxury_resorts_2"
TASK_DESCRIPTION = (
    "Identify 2 beachfront luxury resorts in California that each meet ALL of the following core requirements:\n\n"
    "1. Location & Classification: Must be located on the California coast with direct beachfront access and classified as a luxury resort (4-star or higher)\n"
    "2. Accommodations: Must have at least 250 total guest rooms\n"
    "3. Dining: Must have at least 5 distinct on-site restaurants or dining venues\n"
    "4. Wellness & Recreation (critical requirements):\n"
    "   - Must have at least 3 swimming pools on the property\n"
    "   - Must have a full-service spa facility\n"
    "   - Must have a fitness center\n"
    "5. Meeting & Events:\n"
    "   - Must have at least 10,000 square feet of indoor meeting/event space\n"
    "   - Must have at least one ballroom capable of accommodating 300+ guests\n"
    "6. Additional Amenities (preferred but not required):\n"
    "   - Water sports equipment or activities\n"
    "   - Additional major recreation facilities (golf, tennis, etc.)\n"
    "   - Pet-friendly policy\n"
    "   - ADA-compliant accessible rooms\n"
    "   - Recognized environmental certification\n\n"
    "For each resort, provide the resort name, confirmation of each requirement met, and supporting URL references from official sources."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ResortItem(BaseModel):
    """One resort entry extracted from the answer."""
    name: Optional[str] = None
    # Official or primary sources explicitly cited in the answer (property website, brand site, official fact sheets, etc.)
    source_urls: List[str] = Field(default_factory=list)

    # Optional textual snippets (verbatim from the answer) about features; not used as hard constraints but recorded
    location_text: Optional[str] = None
    beachfront_text: Optional[str] = None
    luxury_text: Optional[str] = None
    rooms_text: Optional[str] = None
    dining_text: Optional[str] = None
    pools_text: Optional[str] = None
    spa_text: Optional[str] = None
    fitness_text: Optional[str] = None
    meeting_space_text: Optional[str] = None
    ballroom_text: Optional[str] = None

    # Preferred amenities (optional)
    water_sports_text: Optional[str] = None
    major_recreation_text: Optional[str] = None
    pet_friendly_text: Optional[str] = None
    ada_rooms_text: Optional[str] = None
    environment_cert_text: Optional[str] = None


class ResortsExtraction(BaseModel):
    """List of up to two resorts extracted from the answer."""
    resorts: List[ResortItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_resorts() -> str:
    return """
    Extract up to TWO distinct California beachfront luxury resort candidates from the answer. For each resort, extract:
    - name: The exact resort name as written in the answer.
    - source_urls: All official or primary URLs explicitly cited in the answer for this resort. These should be the resort's own website pages, brand-owned pages, official fact sheets, or official press/news pages. Extract only URLs present in the answer; do not invent any URLs. Include all relevant pages (property overview, rooms, dining, spa, meetings/events, etc.) if provided.
    - location_text: Any sentence or phrase from the answer describing the resort location.
    - beachfront_text: Any sentence or phrase confirming direct beachfront/oceanfront access.
    - luxury_text: Any sentence or phrase indicating luxury classification (e.g., "luxury resort", "5-star", "4-star", or belonging to a luxury brand).
    - rooms_text: Any sentence or phrase mentioning total room count.
    - dining_text: Any sentence or phrase mentioning number of restaurants or dining venues.
    - pools_text: Any sentence or phrase mentioning the number of pools.
    - spa_text: Any sentence or phrase indicating a full-service spa exists.
    - fitness_text: Any sentence or phrase indicating a fitness center exists.
    - meeting_space_text: Any sentence or phrase mentioning total indoor meeting/event square footage.
    - ballroom_text: Any sentence or phrase indicating ballroom capacity (e.g., "300+ guests").
    - water_sports_text: Any sentence or phrase indicating water sports offerings.
    - major_recreation_text: Any sentence or phrase indicating major recreation facilities (golf, tennis, etc.).
    - pet_friendly_text: Any sentence or phrase indicating a pet-friendly policy.
    - ada_rooms_text: Any sentence or phrase indicating ADA-compliant accessible rooms.
    - environment_cert_text: Any sentence or phrase indicating recognized environmental certification (LEED, Green Key, EarthCheck, etc.).

    IMPORTANT:
    - Return ONLY information explicitly present in the provided answer text.
    - For URLs, include full valid URLs exactly as they appear. Accept markdown links [text](url) and extract the URL.
    - If a field is not present in the answer, set it to null (for text fields) or an empty list (for source_urls).
    - Do not rely on your own knowledge. Do not infer additional details.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_resort(
    evaluator: Evaluator,
    root_node,
    resort: ResortItem,
    index: int,
) -> None:
    """
    Build and verify the rubric tree for a single resort.
    index is 1-based to match rubric IDs (resort_1/resort_2).
    """
    # Parent resort node (parallel, non-critical to allow partial credit across resorts)
    resort_node = evaluator.add_parallel(
        id=f"resort_{index}",
        desc=f"Resort #{index} (one candidate resort)",
        parent=root_node,
        critical=False,
    )

    # Critical existence checks: name and at least one official source URL
    name_exists = resort.name is not None and resort.name.strip() != ""
    sources_exist = bool(resort.source_urls) and any(u.strip() for u in resort.source_urls)

    evaluator.add_custom_node(
        result=name_exists,
        id=f"resort_name_{index}",
        desc="Provide the resort name",
        parent=resort_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=sources_exist,
        id=f"official_sources_{index}",
        desc="Provide supporting URL reference(s) from official sources for the resort details",
        parent=resort_node,
        critical=True
    )

    # Core requirements node (critical) with parallel children
    core_node = evaluator.add_parallel(
        id=f"core_requirements_{index}",
        desc=f"Resort #{index} meets all core requirements",
        parent=resort_node,
        critical=True
    )

    sources = resort.source_urls

    # 1) Beachfront in California
    beachfront_leaf = evaluator.add_leaf(
        id=f"beachfront_ca_{index}",
        desc="Located on the California coast with direct beachfront access",
        parent=core_node,
        critical=True
    )
    beachfront_claim = (
        f"{resort.name} is located on the California coast and has direct beachfront (oceanfront) access."
        if resort.name else
        "The resort is located on the California coast and has direct beachfront (oceanfront) access."
    )
    await evaluator.verify(
        claim=beachfront_claim,
        node=beachfront_leaf,
        sources=sources,
        additional_instruction=(
            "Verify the property is on the California coastline and explicitly offers direct beachfront/oceanfront access. "
            "Look for phrases like 'beachfront', 'on the beach', 'oceanfront', 'steps to the beach'. "
            "Generic 'near beach' without direct access should not count."
        ),
    )

    # 2) Luxury classification (4-star or higher or equivalent luxury positioning)
    luxury_leaf = evaluator.add_leaf(
        id=f"luxury_rating_{index}",
        desc="Classified as a luxury resort (4-star or higher)",
        parent=core_node,
        critical=True
    )
    luxury_claim = (
        f"{resort.name} is a luxury resort (4-star or higher category or equivalent positioning)."
        if resort.name else
        "The resort is a luxury resort (4-star or higher category or equivalent positioning)."
    )
    await evaluator.verify(
        claim=luxury_claim,
        node=luxury_leaf,
        sources=sources,
        additional_instruction=(
            "Accept explicit 'luxury resort' wording, 4- or 5-star mentions, or membership in a widely recognized luxury brand. "
            "Use official/brand pages provided. If evidence is absent, mark as not supported."
        ),
    )

    # 3) Rooms >= 250
    rooms_leaf = evaluator.add_leaf(
        id=f"rooms_{index}",
        desc="Has at least 250 total guest rooms",
        parent=core_node,
        critical=True
    )
    rooms_claim = (
        f"{resort.name} has at least 250 total guest rooms."
        if resort.name else
        "The resort has at least 250 total guest rooms."
    )
    await evaluator.verify(
        claim=rooms_claim,
        node=rooms_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm via official property facts, overview, or accommodations pages. "
            "If a range or exact number is shown, ensure it is ≥ 250."
        ),
    )

    # 4) Dining venues >= 5
    dining_leaf = evaluator.add_leaf(
        id=f"dining_{index}",
        desc="Has at least 5 distinct on-site restaurants or dining venues",
        parent=core_node,
        critical=True
    )
    dining_claim = (
        f"{resort.name} has at least 5 distinct on-site restaurants or dining venues."
        if resort.name else
        "The resort has at least 5 distinct on-site restaurants or dining venues."
    )
    await evaluator.verify(
        claim=dining_claim,
        node=dining_leaf,
        sources=sources,
        additional_instruction=(
            "Count distinct on-site restaurants/dining venues listed on official pages. "
            "Allow venues such as restaurants, bars/lounges with food service, cafes, poolside grills, etc., if explicitly described as distinct venues."
        ),
    )

    # 5) Pools >= 3
    pools_leaf = evaluator.add_leaf(
        id=f"pools_{index}",
        desc="Has at least 3 swimming pools on the property",
        parent=core_node,
        critical=True
    )
    pools_claim = (
        f"{resort.name} has at least 3 swimming pools on the property."
        if resort.name else
        "The resort has at least 3 swimming pools on the property."
    )
    await evaluator.verify(
        claim=pools_claim,
        node=pools_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm via official site; count distinct pools (main pools, family/adult pools, spa pools, etc.) as long as they are described as swimming pools."
        ),
    )

    # 6) Full-service spa facility
    spa_leaf = evaluator.add_leaf(
        id=f"spa_{index}",
        desc="Has a full-service spa facility",
        parent=core_node,
        critical=True
    )
    spa_claim = (
        f"{resort.name} has a full-service spa facility."
        if resort.name else
        "The resort has a full-service spa facility."
    )
    await evaluator.verify(
        claim=spa_claim,
        node=spa_leaf,
        sources=sources,
        additional_instruction=(
            "Look for an official spa page or details describing treatments, services, and amenities indicating a full-service spa."
        ),
    )

    # 7) Fitness center
    fitness_leaf = evaluator.add_leaf(
        id=f"fitness_{index}",
        desc="Has a fitness center",
        parent=core_node,
        critical=True
    )
    fitness_claim = (
        f"{resort.name} has a fitness center."
        if resort.name else
        "The resort has a fitness center."
    )
    await evaluator.verify(
        claim=fitness_claim,
        node=fitness_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm presence of a fitness center/gym on official amenities pages."
        ),
    )

    # 8) Meeting space >= 10,000 sqft (indoor)
    meeting_leaf = evaluator.add_leaf(
        id=f"meeting_space_{index}",
        desc="Has at least 10,000 square feet of indoor meeting/event space",
        parent=core_node,
        critical=True
    )
    meeting_claim = (
        f"{resort.name} has at least 10,000 square feet of indoor meeting/event space."
        if resort.name else
        "The resort has at least 10,000 square feet of indoor meeting/event space."
    )
    await evaluator.verify(
        claim=meeting_claim,
        node=meeting_leaf,
        sources=sources,
        additional_instruction=(
            "Use official meetings/events pages or fact sheets. "
            "Prefer indoor space totals; if only overall is given, ensure the stated space is ≥ 10,000 sqft."
        ),
    )

    # 9) Ballroom capacity >= 300
    ballroom_leaf = evaluator.add_leaf(
        id=f"ballroom_{index}",
        desc="Has at least one ballroom capable of accommodating 300+ guests",
        parent=core_node,
        critical=True
    )
    ballroom_claim = (
        f"{resort.name} has at least one ballroom capable of accommodating 300 or more guests."
        if resort.name else
        "The resort has at least one ballroom capable of accommodating 300 or more guests."
    )
    await evaluator.verify(
        claim=ballroom_claim,
        node=ballroom_leaf,
        sources=sources,
        additional_instruction=(
            "Check official capacity charts/floor plans. "
            "Look specifically for 'ballroom' space with capacity ≥ 300 in banquet/theater/reception as commonly stated."
        ),
    )

    # Preferred (non-required) amenities node
    pref_node = evaluator.add_parallel(
        id=f"preferred_amenities_{index}",
        desc=f"Preferred (non-required) additional amenities for Resort #{index}",
        parent=resort_node,
        critical=False
    )

    # Water sports
    water_leaf = evaluator.add_leaf(
        id=f"water_sports_{index}",
        desc="Offers water sports equipment or activities",
        parent=pref_node,
        critical=False
    )
    water_claim = (
        f"{resort.name} offers water sports equipment or activities."
        if resort.name else
        "The resort offers water sports equipment or activities."
    )
    await evaluator.verify(
        claim=water_claim,
        node=water_leaf,
        sources=sources,
        additional_instruction=(
            "Look for official mentions of surfing lessons, paddleboarding, kayaking, sailing, jet skis, snorkel gear, etc."
        ),
    )

    # Major recreation (golf/tennis/etc.)
    rec_leaf = evaluator.add_leaf(
        id=f"major_recreation_{index}",
        desc="Has additional major recreation facilities (e.g., golf, tennis)",
        parent=pref_node,
        critical=False
    )
    rec_claim = (
        f"{resort.name} has additional major recreation facilities such as golf or tennis."
        if resort.name else
        "The resort has additional major recreation facilities such as golf or tennis."
    )
    await evaluator.verify(
        claim=rec_claim,
        node=rec_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm official presence of facilities like golf courses, tennis courts, etc., on property or directly affiliated."
        ),
    )

    # Pet-friendly
    pet_leaf = evaluator.add_leaf(
        id=f"pet_friendly_{index}",
        desc="Has a pet-friendly policy",
        parent=pref_node,
        critical=False
    )
    pet_claim = (
        f"{resort.name} has a pet-friendly policy."
        if resort.name else
        "The resort has a pet-friendly policy."
    )
    await evaluator.verify(
        claim=pet_claim,
        node=pet_leaf,
        sources=sources,
        additional_instruction=(
            "Look for official policy pages mentioning pets allowed, fees, restrictions."
        ),
    )

    # ADA-compliant accessible rooms
    ada_leaf = evaluator.add_leaf(
        id=f"ada_rooms_{index}",
        desc="Has ADA-compliant accessible rooms",
        parent=pref_node,
        critical=False
    )
    ada_claim = (
        f"{resort.name} has ADA-compliant accessible rooms."
        if resort.name else
        "The resort has ADA-compliant accessible rooms."
    )
    await evaluator.verify(
        claim=ada_claim,
        node=ada_leaf,
        sources=sources,
        additional_instruction=(
            "Check official accessibility statements or room descriptions indicating ADA/accessible accommodations."
        ),
    )

    # Environmental certification
    env_leaf = evaluator.add_leaf(
        id=f"environment_cert_{index}",
        desc="Holds a recognized environmental certification",
        parent=pref_node,
        critical=False
    )
    env_claim = (
        f"{resort.name} holds a recognized environmental certification (e.g., LEED, Green Key, EarthCheck)."
        if resort.name else
        "The resort holds a recognized environmental certification (e.g., LEED, Green Key, EarthCheck)."
    )
    await evaluator.verify(
        claim=env_claim,
        node=env_leaf,
        sources=sources,
        additional_instruction=(
            "Look for official mentions of certifications (LEED, Green Key, EarthCheck, etc.) on property or brand pages."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the California beachfront luxury resorts task.
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

    # Extract resort candidates from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_resorts(),
        template_class=ResortsExtraction,
        extraction_name="resorts_extraction",
    )

    # Keep only first two resorts; pad if fewer
    resorts = (extracted.resorts or [])[:2]
    while len(resorts) < 2:
        resorts.append(ResortItem())

    evaluator.add_custom_info(
        info={
            "num_resorts_in_answer": len(extracted.resorts or []),
            "num_resorts_evaluated": 2
        },
        info_type="meta",
        info_name="resort_count_info"
    )

    # Build verification subtrees for each resort (parallel under root)
    await verify_resort(evaluator, root, resorts[0], index=1)
    await verify_resort(evaluator, root, resorts[1], index=2)

    return evaluator.get_summary()