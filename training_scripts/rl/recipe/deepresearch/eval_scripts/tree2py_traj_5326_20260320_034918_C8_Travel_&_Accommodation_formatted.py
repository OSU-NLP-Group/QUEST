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
TASK_ID = "luxury_resorts_apac_wellness"
TASK_DESCRIPTION = """
I am planning a corporate wellness retreat in the Asia-Pacific region and need to identify three luxury resort properties that can accommodate our group's comprehensive requirements. For each of the three resort properties, please provide:

1. Property name and official website URL
2. Verification that the property meets the following essential criteria:
   - Is a certified or recognized 5-star luxury property
   - Provides 24-hour reception service
   - Has on-site spa and wellness facilities
   - Has on-site fine dining restaurant
   - Offers wheelchair-accessible guest rooms
   - Has roll-in showers with accessibility features (grab bars, fold-down benches)
   - Is located in the Asia-Pacific region
   - Has direct beach or waterfront access
   - Has meeting and event facilities suitable for corporate groups
   - Has a minimum of 50 guest rooms
   - Provides concierge services
   - Has fitness center facilities
   - Offers airport transfer services
   - Offers evening turndown service
   - Has accessible common areas and pathways

Please ensure all three properties are distinct and include reference URLs documenting each property's features and services.
"""

# Helpful reference text for APAC determination
APAC_REFERENCE_TEXT = (
    "For this verification, consider the Asia-Pacific (APAC) region to commonly include: "
    "Australia, New Zealand; East Asia (Japan, South Korea, China, Hong Kong, Macau, Taiwan, Mongolia); "
    "Southeast Asia (Singapore, Malaysia, Indonesia, Thailand, Vietnam, Philippines, Brunei, Cambodia, Laos, Myanmar, Timor-Leste); "
    "South Asia (India, Sri Lanka, Bangladesh, Nepal, Bhutan, Maldives, Pakistan); "
    "and Oceania/Pacific Islands (Fiji, Samoa, Tonga, Vanuatu, Papua New Guinea, Solomon Islands, Micronesia, Palau, Marshall Islands, Kiribati, Nauru, Tuvalu, French Polynesia, New Caledonia, Guam, Northern Mariana Islands). "
    "A property located in one of these countries or territories should be considered APAC."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PropertyItem(BaseModel):
    name: Optional[str] = None
    official_url: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class PropertiesExtraction(BaseModel):
    properties: List[PropertyItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_properties() -> str:
    return """
    Extract up to three distinct resort properties mentioned in the answer.

    For each property, return:
    - name: The property's official name as stated in the answer.
    - official_url: The official website URL for the property or the official brand-managed property page (e.g., hyatt.com/... for a Hyatt property). If the answer instead provides a reputable booking platform page as the main link (e.g., booking.com, marriott.com property page), extract that. If missing, set to null.
    - reference_urls: All additional URLs in the answer that document the property's features/services (e.g., spa page, accessibility page, meetings/events page). Include only URLs explicitly present in the answer text.

    Rules:
    - Extract only URLs explicitly present in the answer (plain links or markdown links). Do not invent URLs.
    - Return at most 3 properties, in the same order they appear in the answer.
    - If a particular field is not provided, set it to null (or an empty list for reference_urls).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_valid_url(u: Optional[str]) -> bool:
    if not u:
        return False
    u = u.strip()
    return u.startswith("http://") or u.startswith("https://")


def _dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not _is_valid_url(u):
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _collect_sources(prop: PropertyItem) -> List[str]:
    urls: List[str] = []
    if _is_valid_url(prop.official_url):
        urls.append(prop.official_url.strip())
    urls.extend(prop.reference_urls or [])
    return _dedupe_urls(urls)


def _prop_label(prop: PropertyItem, idx: int) -> str:
    return prop.name.strip() if prop.name else f"Property #{idx + 1}"


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def verify_property(evaluator: Evaluator, parent_node, prop: PropertyItem, idx: int) -> None:
    """
    Build verification tree for a single property and run all leaf checks.
    All criteria are critical at the leaf level, and the property node is non-critical (for partial credit across properties).
    """
    label = _prop_label(prop, idx)
    prop_node = evaluator.add_parallel(
        id=f"property_{idx + 1}",
        desc=f"{['First','Second','Third'][idx] if idx < 3 else f'Property #{idx+1}'} luxury resort property meeting all specified criteria",
        parent=parent_node,
        critical=False,
    )

    # Prepare sources
    all_sources = _collect_sources(prop)

    # 1) Official URL check (single-URL)
    if _is_valid_url(prop.official_url):
        leaf_official = evaluator.add_leaf(
            id=f"property_{idx + 1}_official_url",
            desc="Property has a verifiable official website or booking platform URL",
            parent=prop_node,
            critical=True,
        )
        await evaluator.verify(
            claim=f"This URL is the official website or an official brand-managed property page (or a reputable booking platform property page) for '{label}'.",
            node=leaf_official,
            sources=prop.official_url,
            additional_instruction=(
                "Accept property microsites owned by the brand (e.g., hyatt.com/hotel/..., marriott.com/hotels/...). "
                "Also accept reputable booking platform property pages if the answer used that as the main URL. "
                "Do not accept generic travel blogs or unrelated aggregator pages as 'official'."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"property_{idx + 1}_official_url",
            desc="Property has a verifiable official website or booking platform URL",
            parent=prop_node,
            critical=True,
        )

    # Helper to add a leaf; if no sources, mark failed directly per source-grounding policy.
    async def _add_claim_leaf(id_suffix: str, desc: str, claim: str, add_ins: str) -> None:
        if not all_sources:
            evaluator.add_custom_node(
                result=False,
                id=f"property_{idx + 1}_{id_suffix}",
                desc=desc,
                parent=prop_node,
                critical=True,
            )
            return
        node = evaluator.add_leaf(
            id=f"property_{idx + 1}_{id_suffix}",
            desc=desc,
            parent=prop_node,
            critical=True,
        )
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=all_sources,
            additional_instruction=add_ins,
        )

    # 2) Five-star rating
    await _add_claim_leaf(
        "five_star_rating",
        "Property is certified or recognized as a 5-star luxury property",
        claim=f"'{label}' is a certified or widely recognized 5-star luxury property.",
        add_ins=(
            "Look for explicit 5-star claims (e.g., '5-star', 'five-star') on the official or brand page, "
            "or recognition by reputable rating bodies (e.g., Forbes Travel Guide, AAA, national tourism authorities) "
            "or brand statements. Minor phrasing variations are acceptable."
        ),
    )

    # 3) 24-hour reception
    await _add_claim_leaf(
        "24hour_reception",
        "Property provides 24-hour reception service",
        claim=f"'{label}' provides 24-hour reception or 24-hour front desk service.",
        add_ins="Allow synonyms like '24-hour front desk', 'round-the-clock reception'.",
    )

    # 4) Evening turndown service
    await _add_claim_leaf(
        "evening_turndown",
        "Property offers evening turndown service",
        claim=f"'{label}' offers evening turndown service (e.g., daily turndown).",
        add_ins="Turndown may be provided in certain room categories or on request; that still counts.",
    )

    # 5) Spa and wellness facilities
    await _add_claim_leaf(
        "spa_wellness",
        "Property has on-site spa and wellness facilities",
        claim=f"'{label}' has on-site spa and wellness facilities (e.g., spa treatments, wellness center).",
        add_ins="Look for a dedicated spa page or references to spa treatments or wellness facilities on property.",
    )

    # 6) Fine dining restaurant
    await _add_claim_leaf(
        "fine_dining",
        "Property has on-site fine dining restaurant",
        claim=f"'{label}' has an on-site fine dining restaurant or an equivalently upscale signature dining venue.",
        add_ins="Accept phrasing like 'signature restaurant', 'fine dining', 'gourmet'. Ensure it is on-site.",
    )

    # 7) Accessible rooms
    await _add_claim_leaf(
        "accessible_rooms",
        "Property has wheelchair-accessible guest rooms available",
        claim=f"'{label}' offers wheelchair-accessible guest rooms.",
        add_ins="Look for 'accessible room(s)', 'ADA room(s)', 'wheelchair accessible', etc.",
    )

    # 8) Roll-in showers
    await _add_claim_leaf(
        "roll_in_showers",
        "Property has roll-in showers with accessibility features (grab bars, fold-down benches)",
        claim=f"'{label}' has guest rooms with roll-in (wheel-in) showers and accessibility features like grab bars or fold-down benches.",
        add_ins="Look specifically for 'roll-in shower' or very close synonyms on accessible room descriptions.",
    )

    # 9) Accessible common areas
    await _add_claim_leaf(
        "accessible_common_areas",
        "Property has accessible common areas and pathways",
        claim=f"'{label}' provides accessible common areas and accessible routes/pathways around the property.",
        add_ins="Accept references to step-free access, ramps, elevators, accessible paths of travel, etc.",
    )

    # 10) Asia-Pacific location
    await _add_claim_leaf(
        "apac_location",
        "Property is located in the Asia-Pacific region",
        claim=f"'{label}' is located in the Asia-Pacific (APAC) region.",
        add_ins=APAC_REFERENCE_TEXT + " Verify by checking the property's address/country on the provided pages.",
    )

    # 11) Beach or waterfront access
    await _add_claim_leaf(
        "beach_waterfront_access",
        "Property has direct beach or waterfront access",
        claim=f"'{label}' has direct beach or waterfront access from the resort grounds (ocean/sea/lake/lagoon/river).",
        add_ins="Look for 'beachfront', 'private beach', 'oceanfront', 'on the lagoon', 'lakefront', 'riverside' with direct access.",
    )

    # 12) Meeting and event facilities
    await _add_claim_leaf(
        "meeting_event_facilities",
        "Property has meeting/event facilities suitable for corporate groups",
        claim=f"'{label}' has on-site meeting and event facilities suitable for corporate groups.",
        add_ins="Look for 'meeting rooms', 'ballroom', 'conference', 'MICE', 'events', and group capacities/catering.",
    )

    # 13) Minimum 50 guest rooms
    await _add_claim_leaf(
        "minimum_50_rooms",
        "Property has a minimum of 50 guest rooms",
        claim=f"'{label}' has at least 50 guest rooms (including rooms, suites, or villas).",
        add_ins="Count rooms, suites, and villas if the site presents a combined total of accommodations.",
    )

    # 14) Concierge services
    await _add_claim_leaf(
        "concierge_services",
        "Property provides concierge services",
        claim=f"'{label}' provides concierge services (e.g., concierge desk, guest relations, butler acting as concierge).",
        add_ins="Accept synonyms: concierge desk, guest relations, butler service providing concierge functions.",
    )

    # 15) Fitness center
    await _add_claim_leaf(
        "fitness_center",
        "Property has fitness center facilities",
        claim=f"'{label}' has an on-site fitness center (gym) for guests.",
        add_ins="Accept 'fitness center', 'gym', 'fitness studio', 'health club' on property.",
    )

    # 16) Airport transfer services
    await _add_claim_leaf(
        "airport_transfer",
        "Property offers airport transfer services",
        claim=f"'{label}' offers airport transfer services (e.g., shuttle, car/limousine, boat transfer if applicable).",
        add_ins="Look for 'airport transfer', 'airport shuttle', 'limousine service', 'private transfer', 'speedboat transfer' (if applicable).",
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
    Evaluate an answer for the luxury APAC wellness resorts task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # 3 properties evaluated independently
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

    # Extract up to 3 properties from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_properties(),
        template_class=PropertiesExtraction,
        extraction_name="extracted_properties",
    )

    props = list(extracted.properties or [])
    # Keep only the first 3 properties, pad if fewer
    props = props[:3]
    while len(props) < 3:
        props.append(PropertyItem())

    # Add an informational record about requested count
    evaluator.add_custom_info(
        {"requested_properties": 3, "extracted_count": sum(1 for p in props if p.name)},
        info_type="meta",
        info_name="request_summary",
    )

    # Build subtrees per property
    # All property subtrees are independent (parallel under root)
    for i in range(3):
        await verify_property(evaluator, root, props[i], i)

    # Return structured summary
    return evaluator.get_summary()