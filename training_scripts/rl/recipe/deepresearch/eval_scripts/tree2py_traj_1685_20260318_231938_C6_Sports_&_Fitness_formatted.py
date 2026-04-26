import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ncaa_2026_regionals"
TASK_DESCRIPTION = """
Identify all four venues hosting the 2026 NCAA Division I Men's Basketball Tournament regional rounds (Sweet 16 and Elite Eight). For each venue, provide: (1) the complete arena name and city location, (2) the basketball seating capacity, (3) the official host university, and (4) which regional (South, West, Midwest, or East) it hosts.
"""

# Expected ground-truth constraints for each regional (used for matching and verification)
EXPECTED_REGIONALS = {
    "south": {
        "arena": "Toyota Center",
        "city": "Houston, TX",
        "capacity_desc": "approximately 18,300",
        "capacity_exact": "18,300",
        "host": "Rice University",
        "region_label": "South"
    },
    "west": {
        "arena": "SAP Center",
        "city": "San Jose, CA",
        "capacity_desc": "approximately between 17,500 and 18,000",
        "capacity_range": (17500, 18000),
        "host": "San Jose State University",
        "region_label": "West"
    },
    "midwest": {
        "arena": "United Center",
        "city": "Chicago, IL",
        "capacity_desc": "approximately between 20,000 and 21,500",
        "capacity_range": (20000, 21500),
        "host": "Northwestern University",
        "region_label": "Midwest"
    },
    "east": {
        "arena": "Capital One Arena",
        "city": "Washington, D.C.",
        "capacity_desc": "approximately 20,000",
        "capacity_range": (19500, 20500),  # tolerant band for “about 20,000”
        "host": "Georgetown University",
        "region_label": "East"
    }
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    """One venue entry as stated in the answer."""
    arena_name: Optional[str] = None
    city_location: Optional[str] = None  # e.g., "Houston, TX" or "Washington, D.C."
    basketball_seating_capacity: Optional[str] = None  # Keep as string for flexibility
    host_university: Optional[str] = None
    regional: Optional[str] = None  # Expected: South, West, Midwest, East (any case/format)
    source_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    """All venues extracted from the answer."""
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt builder                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract the venues that the answer lists for hosting the 2026 NCAA Division I Men's Basketball Tournament regional rounds (Sweet 16 and Elite Eight).
    For each venue mentioned in the answer, extract the following fields:
    - arena_name: the full official arena name exactly as written
    - city_location: the city and state (or D.C.) as written (e.g., "Houston, TX"; "Washington, D.C.")
    - basketball_seating_capacity: the basketball seating capacity as written (keep any formatting or qualifiers like 'approximately')
    - host_university: the official host university as written
    - regional: which regional (South, West, Midwest, or East) this venue is assigned to as written in the answer
    - source_urls: a list of any URLs cited in the answer that are associated with this venue entry
    Return a JSON object with a single field 'venues' which is a list of objects with the above fields.
    Do NOT invent information not explicitly in the answer. If any field is missing for a venue, set it to null (or empty list for source_urls).
    If the answer lists more than four venue entries, still extract them all; we will only evaluate the first four later.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def canonical_region(s: Optional[str]) -> Optional[str]:
    t = _norm(s)
    if not t:
        return None
    if "south" in t:
        return "south"
    if "west" in t:
        return "west"
    if "midwest" in t:
        return "midwest"
    if "east" in t:
        return "east"
    return None


def dedup_key(item: VenueItem) -> str:
    return f"{_norm(item.arena_name)}|{_norm(item.city_location)}"


def pick_first_k(venues: List[VenueItem], k: int = 4) -> List[VenueItem]:
    return venues[:k] if len(venues) > k else venues


def find_by_region(venues: List[VenueItem], region_key: str) -> Optional[VenueItem]:
    for v in venues:
        if canonical_region(v.regional) == region_key:
            return v
    return None


def has_all_required_fields(v: VenueItem) -> bool:
    return bool(
        v
        and v.arena_name and v.arena_name.strip()
        and v.city_location and v.city_location.strip()
        and v.basketball_seating_capacity and v.basketball_seating_capacity.strip()
        and v.host_university and v.host_university.strip()
        and v.regional and canonical_region(v.regional) in {"south", "west", "midwest", "east"}
    )


def capacity_match_claim_parts(region_key: str, venue: VenueItem) -> Tuple[str, str]:
    """
    Build a pair of strings:
    - match_text: compares the extracted capacity string to the expected description (simple verify).
    - support_text: a URL-backed claim about the expected capacity for the arena.
    """
    exp = EXPECTED_REGIONALS[region_key]
    cap_desc = exp.get("capacity_desc") or "a specific basketball seating capacity"
    arena = exp["arena"]
    city = exp["city"]
    # For support, include a general approximate capacity statement
    support_text = f"{arena} in {city} has a basketball seating capacity of {cap_desc}."
    # For match, compare the user's provided capacity string to expected description
    user_cap = venue.basketball_seating_capacity or ""
    match_text = f"The capacity value provided in the answer ('{user_cap}') is consistent with {cap_desc}."
    return match_text, support_text


# --------------------------------------------------------------------------- #
# Region verification logic                                                   #
# --------------------------------------------------------------------------- #
async def verify_one_region(
    evaluator: Evaluator,
    parent_node,
    region_key: str,
    extracted_venues: List[VenueItem],
) -> None:
    """
    Build and execute the verification subtree for one region (south/west/midwest/east).
    The subtree is sequential: existence -> presence of all fields -> value matches -> source-backed checks.
    """
    exp = EXPECTED_REGIONALS[region_key]
    pretty_region = exp["region_label"]
    desc = (
        f"{pretty_region} Regional entry is complete and matches constraints: "
        f"arena name {exp['arena']}; city/location {exp['city']}; "
        f"basketball seating capacity {exp['capacity_desc']}; "
        f"official host university {exp['host']}; explicitly designated as {pretty_region} Regional."
    )

    region_node = evaluator.add_sequential(
        id=f"{region_key}_regional_block",
        desc=desc,
        parent=parent_node,
        critical=False  # The region block itself is non-critical at the root level
    )

    # Locate the entry for this region
    venue = find_by_region(extracted_venues, region_key)
    found = venue is not None

    # 1) Existence check for region entry
    evaluator.add_custom_node(
        result=found,
        id=f"{region_key}_entry_found",
        desc=f"{pretty_region} Regional entry is present in the answer",
        parent=region_node,
        critical=True
    )

    # Safety if missing: construct a dummy to allow ID creation, later checks will be skipped by sequential short-circuit
    if not found:
        venue = VenueItem()

    # 2) Completeness check (all required fields present)
    evaluator.add_custom_node(
        result=has_all_required_fields(venue),
        id=f"{region_key}_entry_complete",
        desc=f"{pretty_region} entry includes arena name, city/location, capacity, host university, and regional label",
        parent=region_node,
        critical=True
    )

    # 3) Value matches vs. expected (simple, non-URL factual checks)
    # 3.1 Arena name match
    arena_match = evaluator.add_leaf(
        id=f"{region_key}_arena_match",
        desc=f"Arena name in the answer matches expected '{exp['arena']}'",
        parent=region_node,
        critical=True
    )
    arena_claim = f"The arena name provided in the answer ('{venue.arena_name}') matches the expected '{exp['arena']}' (allow minor formatting variations)."
    await evaluator.verify(
        claim=arena_claim,
        node=arena_match,
        additional_instruction="Treat minor punctuation, casing, or suffix variations as acceptable if they clearly refer to the same arena."
    )

    # 3.2 City/location match
    city_match = evaluator.add_leaf(
        id=f"{region_key}_city_match",
        desc=f"City/location in the answer matches expected '{exp['city']}'",
        parent=region_node,
        critical=True
    )
    city_claim = f"The city/location value in the answer ('{venue.city_location}') refers to '{exp['city']}' (allow common variants like full state names or standard abbreviations)."
    await evaluator.verify(
        claim=city_claim,
        node=city_match,
        additional_instruction="Allow common formatting variants such as 'Washington, DC' vs. 'Washington, D.C.' or 'Texas' vs. 'TX'."
    )

    # 3.3 Capacity match (approximate tolerance)
    cap_match = evaluator.add_leaf(
        id=f"{region_key}_capacity_match",
        desc=f"Capacity in the answer is consistent with expected {exp['capacity_desc']}",
        parent=region_node,
        critical=True
    )
    cap_match_claim, cap_support_claim = capacity_match_claim_parts(region_key, venue)
    await evaluator.verify(
        claim=cap_match_claim,
        node=cap_match,
        additional_instruction="Judge consistency with a tolerance of roughly ±5% or common rounding (e.g., 'about 20,000' vs. '20,356')."
    )

    # 3.4 Host university match
    host_match = evaluator.add_leaf(
        id=f"{region_key}_host_match",
        desc=f"Host university in the answer matches expected '{exp['host']}'",
        parent=region_node,
        critical=True
    )
    host_claim = f"The host university provided in the answer ('{venue.host_university}') matches the expected '{exp['host']}' (allow minor naming variations)."
    await evaluator.verify(
        claim=host_claim,
        node=host_match,
        additional_instruction="Accept minor formatting (e.g., 'Univ.' vs. 'University') if clearly the same institution."
    )

    # 3.5 Regional label match
    label_match = evaluator.add_leaf(
        id=f"{region_key}_label_match",
        desc=f"Regional label in the answer is '{pretty_region}'",
        parent=region_node,
        critical=True
    )
    label_claim = f"The answer labels this venue as the '{pretty_region}' Regional (allow small wording variants like 'Regional - {pretty_region}')."
    await evaluator.verify(
        claim=label_claim,
        node=label_match,
        additional_instruction="Allow minor wording variations like 'South Region' or 'South Regional' to count as '{pretty_region}'."
    )

    # 4) Source-backed checks (using any URLs associated with this venue)
    urls = venue.source_urls if venue and venue.source_urls else []

    # 4.1 Regional mapping supported by sources (2026 specificity)
    region_source_node = evaluator.add_leaf(
        id=f"{region_key}_regional_supported",
        desc=f"Sources support that the 2026 {pretty_region} Regional is hosted at {exp['arena']} in {exp['city']} by {exp['host']}",
        parent=region_node,
        critical=True
    )
    regional_support_claim = (
        f"For the 2026 NCAA Division I Men's Basketball Tournament, the {pretty_region} Regional (Sweet 16/Elite Eight) "
        f"is hosted at {exp['arena']} in {exp['city']}, and the official host institution is {exp['host']}."
    )
    await evaluator.verify(
        claim=regional_support_claim,
        node=region_source_node,
        sources=urls,
        additional_instruction="Confirm the 2026 Regional assignment, not a different year, and that the cited host institution is correct."
    )

    # 4.2 Capacity supported by sources (approximate phrasing)
    capacity_source_node = evaluator.add_leaf(
        id=f"{region_key}_capacity_supported",
        desc=f"Sources support the basketball capacity for {exp['arena']} ({exp['capacity_desc']})",
        parent=region_node,
        critical=True
    )
    await evaluator.verify(
        claim=cap_support_claim,
        node=capacity_source_node,
        sources=urls,
        additional_instruction="If multiple capacities are listed (concert/hockey/basketball), focus on basketball configuration; allow approximate phrasing."
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
    Evaluate an answer for the 2026 NCAA Regional Venues task.
    """
    # Initialize evaluator (root is PARALLEL). Set root non-critical to allow partial credit while honoring critical children.
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

    # Ground truth context
    evaluator.add_ground_truth({
        "expected_regions": EXPECTED_REGIONALS,
        "note": "Expected venues and attributes for the 2026 NCAA Division I Men's Basketball Tournament regional rounds."
    })

    # 1) Extract venue entries from the answer
    extracted: VenuesExtraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # 2) Apply standard post-processing: keep only the first 4 items (per evaluation policy)
    raw_list = extracted.venues or []
    venues_4 = pick_first_k(raw_list, 4)

    # 3) Build quick stats for custom info
    normalized_regions = [canonical_region(v.regional) for v in venues_4]
    unique_keys = {dedup_key(v) for v in venues_4}
    evaluator.add_custom_info(
        info={
            "total_extracted": len(raw_list),
            "evaluated_count": len(venues_4),
            "regions_extracted_first_4": normalized_regions,
            "unique_key_count_first_4": len(unique_keys),
        },
        info_type="extraction_summary"
    )

    # 4) Critical checks at the root level
    # 4.1 Exactly four distinct venues (within the evaluated 4)
    evaluator.add_custom_node(
        result=(len(venues_4) == 4 and len(unique_keys) == 4),
        id="Exactly_Four_Distinct_Venues_Listed",
        desc="Response lists exactly four distinct venues (within evaluated set of first four).",
        parent=root,
        critical=True
    )

    # 4.2 All four regionals covered exactly once (within evaluated 4)
    region_counts: Dict[str, int] = {"south": 0, "west": 0, "midwest": 0, "east": 0}
    for r in normalized_regions:
        if r in region_counts:
            region_counts[r] += 1
    all_once = all(region_counts[k] == 1 for k in region_counts.keys())
    evaluator.add_custom_node(
        result=all_once,
        id="All_Four_Regionals_Covered_Exactly_Once",
        desc="South, West, Midwest, and East are each assigned exactly once across the four venue entries (evaluated set).",
        parent=root,
        critical=True
    )

    # 5) Region-specific verification blocks (non-critical siblings under root)
    # South
    await verify_one_region(evaluator, root, "south", venues_4)
    # West
    await verify_one_region(evaluator, root, "west", venues_4)
    # Midwest
    await verify_one_region(evaluator, root, "midwest", venues_4)
    # East
    await verify_one_region(evaluator, root, "east", venues_4)

    # 6) Return structured summary
    return evaluator.get_summary()