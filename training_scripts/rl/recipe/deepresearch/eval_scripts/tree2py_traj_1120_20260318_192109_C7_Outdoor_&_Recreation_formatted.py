import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "accessible_gulf_coast_resort_2026"
TASK_DESCRIPTION = """
A family is planning an accessible beach reunion in July 2026 and needs to identify a fully wheelchair-accessible beachfront resort on Florida's Gulf Coast between Pensacola Beach and Panama City Beach. The resort must meet comprehensive ADA accessibility standards to accommodate family members with mobility disabilities.

The resort must provide all of the following accessibility features:
- Wheelchair-accessible guest rooms with ADA compliance
- Roll-in showers in accessible rooms
- Grab bars in bathrooms (at toilet and shower areas)
- Doorways with minimum 32-inch clear width in accessible rooms
- Complimentary beach wheelchairs available for guest use
- Swimming pool with lift access or walk-in entry
- Ramped access from the resort property to the beach
- ADA-compliant door viewers in accessible rooms
- Designated wheelchair-accessible parking spaces
- Visual fire alarm systems in accessible rooms
- Policy permitting service animals throughout the property
- Lowered door peepholes at wheelchair-user height

Additionally, the resort must be directly beachfront (with immediate beach access) and must be operational with availability for bookings in July 2026.

Identify one resort that meets all of these requirements, and provide supporting documentation showing it satisfies each accessibility feature and location criterion.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ResortExtraction(BaseModel):
    # Resort identity
    resort_names: List[str] = Field(default_factory=list, description="All distinct resort/property names presented as the final answer or candidates.")
    primary_resort_name: Optional[str] = Field(default=None, description="The explicit single resort the answer recommends. If only one name is provided, repeat it here.")
    # Common property URLs explicitly mentioned in the answer
    property_homepage: Optional[str] = None
    property_urls: List[str] = Field(default_factory=list, description="Any other official property pages cited (e.g., accessibility statement, amenities page).")
    booking_urls: List[str] = Field(default_factory=list, description="Reservation or booking engine URLs used in the answer.")
    # Location evidence
    beachfront_urls: List[str] = Field(default_factory=list, description="URLs that show the property is directly beachfront with immediate beach access.")
    gulf_coast_between_range_urls: List[str] = Field(default_factory=list, description="URLs that show the property’s location between Pensacola Beach and Panama City Beach on Florida’s Gulf Coast (address, map, locale pages).")
    july_2026_availability_urls: List[str] = Field(default_factory=list, description="URLs that evidence operational status and booking availability for dates in July 2026.")
    # Accessibility feature evidence
    accessible_rooms_urls: List[str] = Field(default_factory=list, description="Evidence of ADA/wheelchair-accessible guest rooms.")
    roll_in_showers_urls: List[str] = Field(default_factory=list, description="Evidence of roll-in showers in accessible rooms.")
    grab_bars_urls: List[str] = Field(default_factory=list, description="Evidence of grab bars at toilet and shower areas.")
    door_width_32in_urls: List[str] = Field(default_factory=list, description="Evidence of minimum 32-inch door clear width in accessible rooms.")
    beach_wheelchairs_urls: List[str] = Field(default_factory=list, description="Evidence the resort provides complimentary beach wheelchairs for guest use.")
    pool_access_urls: List[str] = Field(default_factory=list, description="Evidence of pool lift or walk-in (zero-entry/sloped) access.")
    ramped_beach_access_urls: List[str] = Field(default_factory=list, description="Evidence of a ramp/boardwalk providing access from property to beach (e.g., accessible dune walkover).")
    ada_door_viewers_urls: List[str] = Field(default_factory=list, description="Evidence of ADA door viewers/peepholes at wheelchair-user height (lowered).")
    accessible_parking_urls: List[str] = Field(default_factory=list, description="Evidence of designated accessible parking spaces.")
    visual_fire_alarms_urls: List[str] = Field(default_factory=list, description="Evidence of visual fire alarm notification systems in accessible rooms.")
    service_animals_urls: List[str] = Field(default_factory=list, description="Evidence that service animals are permitted across the property.")
    # Misc
    other_general_urls: List[str] = Field(default_factory=list, description="Any other URLs the answer cites that may support claims.")


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_resort() -> str:
    return """
Extract the structured information below only from the provided answer text. Do not invent any information or URLs. Extract only URLs that are explicitly present in the answer (plain or markdown). If a field is not provided in the answer, return null (for a single value) or an empty array (for lists).

Required fields:
- resort_names: list of all distinct resort/property names presented as candidates or the final answer.
- primary_resort_name: the single specific resort/property the answer ultimately recommends; if only one name appears, repeat it here; otherwise null if not explicitly singled out.
- property_homepage: the official homepage URL of the chosen resort if given.
- property_urls: other official resort URLs cited (e.g., accessibility statement, amenities pages).
- booking_urls: reservation/booking engine search/result URLs used in the answer.

Location evidence (URLs):
- beachfront_urls: URLs demonstrating the property is directly beachfront with immediate beach access.
- gulf_coast_between_range_urls: URLs showing the property is located on Florida's Gulf Coast between Pensacola Beach and Panama City Beach (address, map, or official locale evidence).
- july_2026_availability_urls: URLs that show operational status and booking availability for dates within July 2026.

Accessibility features evidence (URLs):
- accessible_rooms_urls: ADA/wheelchair-accessible guest rooms evidence.
- roll_in_showers_urls: evidence of roll-in showers in accessible rooms.
- grab_bars_urls: evidence of grab bars at toilet and shower areas.
- door_width_32in_urls: evidence that accessible room doorways provide at least 32-inch clear width.
- beach_wheelchairs_urls: evidence that the resort provides complimentary beach wheelchairs for guest use (not just municipality).
- pool_access_urls: evidence of pool lift or walk-in (zero-entry/sloped) entry.
- ramped_beach_access_urls: evidence of ramped access from property to the beach (e.g., accessible boardwalk/dune walkover).
- ada_door_viewers_urls: evidence of ADA-compliant door viewers/peepholes at wheelchair height (lowered).
- accessible_parking_urls: evidence of designated accessible parking spaces.
- visual_fire_alarms_urls: evidence of visual fire alarms in accessible rooms.
- service_animals_urls: evidence that service animals are permitted.

- other_general_urls: any other URLs the answer cites that might support claims.

Only include URLs that are explicitly present in the answer text.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedupe_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _merge_sources(*url_lists: List[str]) -> List[str]:
    merged: List[str] = []
    for ul in url_lists:
        if ul:
            merged.extend(ul)
    return _dedupe_preserve_order(merged)


def _nonempty(lst: Optional[List[str]]) -> bool:
    return bool(lst) and any(isinstance(x, str) and x.strip() for x in lst or [])


def _one_resort_name(extracted: ResortExtraction) -> Optional[str]:
    # If primary explicitly provided, prefer it
    if extracted.primary_resort_name and extracted.primary_resort_name.strip():
        return extracted.primary_resort_name.strip()
    # Else if exactly one resort name extracted, use that
    distinct = [n.strip() for n in (extracted.resort_names or []) if n and n.strip()]
    distinct = list(dict.fromkeys(distinct))  # dedupe while preserving order
    if len(distinct) == 1:
        return distinct[0]
    return None


# --------------------------------------------------------------------------- #
# Build and verify tree                                                       #
# --------------------------------------------------------------------------- #
async def _build_and_verify(
    evaluator: Evaluator,
    root_parent,
    extracted: ResortExtraction,
) -> None:
    """
    Build the verification nodes and run URL-grounded checks. Any check without
    supporting URLs from the answer will be marked as failed (avoid ungrounded verification).
    """
    # Create the rubric's main critical node (parallel aggregation)
    main = evaluator.add_parallel(
        id="identify_accessible_resort",
        desc="Identify one fully wheelchair-accessible, directly beachfront resort on Florida's Gulf Coast (between Pensacola Beach and Panama City Beach) that is operational and bookable in July 2026, and provide supporting documentation showing it satisfies each stated requirement.",
        parent=root_parent,
        critical=True,
    )

    # Determine chosen resort name (for claims)
    chosen_name = _one_resort_name(extracted)

    # 1) names_one_resort — critical (custom check)
    exactly_one = chosen_name is not None
    evaluator.add_custom_node(
        result=exactly_one,
        id="names_one_resort",
        desc="Response identifies exactly one specific resort/property as the answer.",
        parent=main,
        critical=True,
    )

    # Helper to add one leaf and schedule verification if sources exist; otherwise mark failed
    claims_and_sources: List[tuple[str, List[str], Any, Optional[str]]] = []

    def add_feature_leaf(node_id: str, desc: str, claim: str, sources: List[str], add_ins: Optional[str] = None):
        node = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=main,
            critical=True,
        )
        if _nonempty(sources):
            claims_and_sources.append((claim, sources, node, add_ins or "None"))
        else:
            # Fail immediately due to lack of cited sources in the answer (avoid ungrounded verification)
            node.score = 0.0
            node.status = "failed"
            evaluator.add_custom_info(
                {"missing_sources_for": node_id, "desc": desc},
                info_type="missing_sources",
                info_name=f"missing_sources_{node_id}"
            )

    # Common generic source pool (only those explicitly present in the answer)
    common_sources = _merge_sources(
        [extracted.property_homepage] if extracted.property_homepage else [],
        extracted.property_urls,
        extracted.other_general_urls,
    )

    # Feature checks (all critical)
    # 2) Wheelchair-accessible guest rooms (ADA)
    add_feature_leaf(
        "has_accessible_rooms",
        "Resort provides wheelchair-accessible guest rooms with ADA compliance.",
        claim=f"The resort{f' {chosen_name}' if chosen_name else ''} provides ADA- or wheelchair-accessible guest rooms (mobility accessible rooms) that comply with ADA requirements.",
        sources=_merge_sources(extracted.accessible_rooms_urls, common_sources),
        add_ins=(
            "Look for phrases like 'ADA accessible', 'mobility accessible room', 'wheelchair accessible'. "
            "Evidence should indicate ADA compliance or equivalent for guest rooms."
        ),
    )

    # 3) Roll-in showers
    add_feature_leaf(
        "has_roll_in_showers",
        "Accessible rooms include roll-in showers.",
        claim=f"The resort{f' {chosen_name}' if chosen_name else ''} offers accessible rooms that include roll-in showers.",
        sources=_merge_sources(extracted.roll_in_showers_urls, common_sources),
        add_ins="Look for 'roll-in shower' in accessible room descriptions or amenity lists.",
    )

    # 4) Grab bars
    add_feature_leaf(
        "has_grab_bars",
        "Bathrooms in accessible rooms have grab bars at the toilet and in the shower area.",
        claim=f"The accessible bathrooms at the resort{f' {chosen_name}' if chosen_name else ''} have grab bars at the toilet and in the shower.",
        sources=_merge_sources(extracted.grab_bars_urls, common_sources),
        add_ins="Look for 'grab bars' specifically at toilet and shower locations.",
    )

    # 5) Doorway width ≥ 32 inches
    add_feature_leaf(
        "door_width_compliant",
        "Doorways in accessible rooms have at least 32-inch clear width.",
        claim=f"The accessible room doorways at the resort{f' {chosen_name}' if chosen_name else ''} provide a minimum 32-inch clear width.",
        sources=_merge_sources(extracted.door_width_32in_urls, common_sources),
        add_ins=(
            "Look for explicit statements like '32 inches', '32-inch clear width', or 'doorway ≥ 32 inches'. "
            "Minor variations (e.g., 34 in) should still satisfy the 'at least 32 inches' requirement."
        ),
    )

    # 6) Complimentary beach wheelchairs provided by resort
    add_feature_leaf(
        "provides_beach_wheelchairs",
        "Resort provides complimentary beach wheelchairs for guest use.",
        claim=f"The resort{f' {chosen_name}' if chosen_name else ''} provides complimentary beach wheelchairs for guest use (managed by the resort or on-site).",
        sources=_merge_sources(extracted.beach_wheelchairs_urls, common_sources),
        add_ins=(
            "Evidence must indicate the resort itself provides beach wheelchairs or manages on-site access specifically for guests. "
            "Do NOT count general city/county programs unless the resort explicitly facilitates on-site guest access."
        ),
    )

    # 7) Swimming pool access (lift or walk-in)
    add_feature_leaf(
        "pool_has_accessibility",
        "Swimming pool has lift access or walk-in entry.",
        claim=f"The resort{f' {chosen_name}' if chosen_name else ''} provides accessible pool entry via a pool lift, chair hoist, or a walk-in (zero-entry/sloped) design.",
        sources=_merge_sources(extracted.pool_access_urls, common_sources),
        add_ins="Accept terms like 'pool lift', 'ADA lift', 'chair hoist', 'zero-entry', 'beach entry', or 'sloped entry'.",
    )

    # 8) Ramped access from property to beach
    add_feature_leaf(
        "ramped_beach_access",
        "Resort has ramped access from the resort property to the beach.",
        claim=f"The resort{f' {chosen_name}' if chosen_name else ''} has a ramp/accessible boardwalk providing access from the property to the beach.",
        sources=_merge_sources(extracted.ramped_beach_access_urls, common_sources),
        add_ins="Look for 'ramp', 'accessible boardwalk', 'accessible dune walkover', or 'Mobi-mat' connecting property to beach.",
    )

    # 9) ADA door viewers/peepholes at accessible height
    add_feature_leaf(
        "ada_door_viewers_at_accessible_height",
        "Accessible rooms have ADA-compliant door viewers/peepholes that are positioned at wheelchair-user height (i.e., lowered/accessible viewing height).",
        claim=f"The accessible rooms at the resort{f' {chosen_name}' if chosen_name else ''} include ADA door viewers/peepholes positioned at wheelchair-user (lowered) height.",
        sources=_merge_sources(extracted.ada_door_viewers_urls, common_sources),
        add_ins="Look for 'lowered peephole', 'door viewer at accessible height', or similar wording.",
    )

    # 10) Accessible parking
    add_feature_leaf(
        "accessible_parking",
        "Resort provides designated wheelchair-accessible parking spaces.",
        claim=f"The resort{f' {chosen_name}' if chosen_name else ''} provides designated accessible (ADA) parking spaces.",
        sources=_merge_sources(extracted.accessible_parking_urls, common_sources),
        add_ins="Look for 'accessible parking', 'ADA parking', or reserved spaces for disabled guests.",
    )

    # 11) Visual fire alarms
    add_feature_leaf(
        "visual_fire_alarms",
        "Accessible rooms have visual fire alarm notification systems.",
        claim=f"The accessible rooms at the resort{f' {chosen_name}' if chosen_name else ''} have visual fire alarm notification systems (e.g., strobe alarms).",
        sources=_merge_sources(extracted.visual_fire_alarms_urls, common_sources),
        add_ins="Accept terms like 'visual fire alarm', 'strobe alarm', 'visual notification device' in accessible rooms.",
    )

    # 12) Service animals permitted
    add_feature_leaf(
        "service_animals_permitted",
        "Resort permits service animals throughout the property.",
        claim=f"The resort{f' {chosen_name}' if chosen_name else ''} permits service animals throughout the property.",
        sources=_merge_sources(extracted.service_animals_urls, common_sources),
        add_ins="Policy should explicitly allow service animals (not just 'pets allowed').",
    )

    # 13) Directly beachfront
    add_feature_leaf(
        "beachfront_location",
        "Resort is directly beachfront with immediate beach access.",
        claim=f"The resort{f' {chosen_name}' if chosen_name else ''} is directly beachfront with immediate beach access.",
        sources=_merge_sources(extracted.beachfront_urls, common_sources),
        add_ins="Look for 'beachfront', 'on the beach', 'direct beach access'.",
    )

    # 14) On Florida's Gulf Coast between Pensacola Beach and Panama City Beach
    add_feature_leaf(
        "gulf_coast_location",
        "Resort is located on Florida's Gulf Coast between Pensacola Beach and Panama City Beach.",
        claim=f"The resort{f' {chosen_name}' if chosen_name else ''} is on Florida's Gulf Coast between Pensacola Beach and Panama City Beach (e.g., Gulf Breeze/Navarre Beach/Fort Walton Beach/Okaloosa Island/Destin/Miramar Beach/Santa Rosa Beach/30A communities).",
        sources=_merge_sources(extracted.gulf_coast_between_range_urls, common_sources),
        add_ins=(
            "Use the provided webpages to identify the city/area. You may use general U.S. geography knowledge to judge whether this city/area lies along the Gulf Coast between Pensacola Beach and Panama City Beach."
        ),
    )

    # 15) Operational and available for bookings in July 2026
    add_feature_leaf(
        "july_2026_availability",
        "Resort is operational and available for bookings in July 2026.",
        claim=f"The resort{f' {chosen_name}' if chosen_name else ''} is operational and accepting bookings for dates within July 2026 (has bookable inventory on at least one July 2026 date).",
        sources=_merge_sources(extracted.july_2026_availability_urls, extracted.booking_urls, common_sources),
        add_ins="Check booking engine or availability pages for any dates within July 2026. Evidence of open inventory on any July 2026 date satisfies this.",
    )

    # 16) Supporting documentation — critical (custom check across all criteria)
    # Must provide at least one URL for every single listed requirement and location/availability criterion.
    requirement_to_urls = {
        "has_accessible_rooms": extracted.accessible_rooms_urls,
        "has_roll_in_showers": extracted.roll_in_showers_urls,
        "has_grab_bars": extracted.grab_bars_urls,
        "door_width_compliant": extracted.door_width_32in_urls,
        "provides_beach_wheelchairs": extracted.beach_wheelchairs_urls,
        "pool_has_accessibility": extracted.pool_access_urls,
        "ramped_beach_access": extracted.ramped_beach_access_urls,
        "ada_door_viewers_at_accessible_height": extracted.ada_door_viewers_urls,
        "accessible_parking": extracted.accessible_parking_urls,
        "visual_fire_alarms": extracted.visual_fire_alarms_urls,
        "service_animals_permitted": extracted.service_animals_urls,
        "beachfront_location": extracted.beachfront_urls,
        "gulf_coast_location": extracted.gulf_coast_between_range_urls,
        "july_2026_availability": _merge_sources(extracted.july_2026_availability_urls, extracted.booking_urls),
    }
    coverage = {k: _nonempty(v) for k, v in requirement_to_urls.items()}
    all_covered = all(coverage.values())

    evaluator.add_custom_node(
        result=all_covered,
        id="supporting_documentation",
        desc="Response provides supporting documentation (e.g., credible citations/links) demonstrating the resort satisfies each listed accessibility requirement and each location/availability criterion.",
        parent=main,
        critical=True,
    )
    evaluator.add_custom_info(
        {"per_requirement_url_provided": coverage},
        info_type="support_coverage",
        info_name="documentation_coverage"
    )

    # Run all URL-grounded verifications in parallel
    if claims_and_sources:
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the 'accessible_gulf_coast_resort_2026' task.
    """
    # Initialize evaluator (root node is a non-critical container)
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

    # Extract structured data from the answer
    extracted: ResortExtraction = await evaluator.extract(
        prompt=prompt_extract_resort(),
        template_class=ResortExtraction,
        extraction_name="resort_extraction",
    )

    # Build verification tree and run checks
    await _build_and_verify(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()