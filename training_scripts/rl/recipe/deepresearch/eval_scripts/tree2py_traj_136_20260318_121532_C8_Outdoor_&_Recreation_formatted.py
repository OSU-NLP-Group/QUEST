import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_state_parks_rv_july_2026"
TASK_DESCRIPTION = """
You are planning an RV camping trip in California for July 2026 and need campgrounds with full amenities. Find 3 campgrounds within the California State Parks system that meet ALL of the following requirements:

1. Must offer RV campsites with water and electric hookups
2. Must accommodate RVs of at least 30 feet in length at hookup sites
3. Must provide restroom facilities with hot showers available to campers
4. Must use the ReserveCalifornia reservation system (reservecalifornia.com)
5. Must be open for camping during July 2026
6. Reservations must be available up to 6 months in advance

For each campground, provide:
- The park name and campground name
- Maximum RV length accommodated at hookup sites
- Electrical service specifications (amp ratings available)
- Confirmation of shower facility availability
- Reference URL from California State Parks website or ReserveCalifornia
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CampgroundItem(BaseModel):
    park_name: Optional[str] = None
    campground_name: Optional[str] = None
    max_rv_length_hookup: Optional[str] = None
    electrical_service: Optional[str] = None  # e.g., "30/50 amp", "30 amp", "20/30/50 amp"
    water_electric_hookups: Optional[str] = None  # e.g., "yes", "both", "full hookups"
    showers_available: Optional[str] = None  # e.g., "hot showers", "showers available"
    uses_reservecalifornia: Optional[str] = None  # e.g., "yes", "ReserveCalifornia"
    open_in_july_2026: Optional[str] = None  # e.g., "open year-round", "open in July"
    reference_urls: List[str] = Field(default_factory=list)


class CampgroundsExtraction(BaseModel):
    campgrounds: List[CampgroundItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_campgrounds() -> str:
    return """
    Extract up to three campgrounds (from the California State Parks system) mentioned in the answer.
    For each campground, extract the following fields exactly as they appear in the answer (do not invent):
    - park_name: The California State Park name (e.g., "San Elijo State Beach").
    - campground_name: The named campground or loop if provided (may be same as park; if not mentioned, set to null).
    - max_rv_length_hookup: The maximum RV length for hookup sites (include units if present; e.g., "35 ft", "up to 30 feet").
    - electrical_service: The electrical hookup amp ratings available (e.g., "30 amp", "50 amp", "20/30/50 amp"). If multiple, include them as one string (e.g., "30/50 amp").
    - water_electric_hookups: Whether both water and electric hookups are available at RV sites (e.g., "both", "full hookups", "water and electric"). If unclear or not mentioned, set to null.
    - showers_available: Confirmation of showers availability (e.g., "hot showers", "showers available"). If not mentioned, set to null.
    - uses_reservecalifornia: Whether reservations use ReserveCalifornia (e.g., "ReserveCalifornia", "yes"). If not mentioned, set to null.
    - open_in_july_2026: Any explicit seasonality or open status that implies July 2026 is open (e.g., "open year-round", "summer season"). If not mentioned, set to null.
    - reference_urls: All URLs cited for this campground. Include only valid full URLs that are explicitly present in the answer. Prefer URLs from 'parks.ca.gov' and 'reservecalifornia.com' if they appear, but include all URLs that are present.

    Return a JSON object with a single key 'campgrounds' that is an array of up to three campground objects with the above fields.
    If the answer mentions more than three, include the first three in the order they appear. If fewer than three, include as many as present.
    If a field is missing for a campground, set it to null (or an empty list for reference_urls).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_allowed_domain(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
        return ("parks.ca.gov" in netloc) or ("reservecalifornia.com" in netloc)
    except Exception:
        return False


def filter_allowed_sources(urls: List[str]) -> List[str]:
    """Filter to CA State Parks or ReserveCalifornia URLs; keep order and uniqueness."""
    seen = set()
    filtered = []
    for u in urls:
        if not u or not isinstance(u, str):
            continue
        if _is_allowed_domain(u):
            if u not in seen:
                filtered.append(u)
                seen.add(u)
    return filtered


def campground_label(item: CampgroundItem) -> str:
    p = (item.park_name or "").strip()
    c = (item.campground_name or "").strip()
    if p and c and c.lower() != p.lower():
        return f"'{c}' at {p}"
    return p or c or "the campground"


# --------------------------------------------------------------------------- #
# Verification logic per campground                                           #
# --------------------------------------------------------------------------- #
async def verify_campground(evaluator: Evaluator, camp_node, item: CampgroundItem, idx: int) -> None:
    """
    Build and execute verification for one campground under camp_node.
    The structure follows the rubric; all factual leaves are verified against provided URLs.
    """
    idx1 = idx + 1
    sources_allowed = filter_allowed_sources(item.reference_urls)

    # 1) Reference URL existence and validity (Critical)
    ref_node = evaluator.add_custom_node(
        result=bool(sources_allowed),
        id=f"campground_{idx1}_reference",
        desc="Valid reference URL from California State Parks website or ReserveCalifornia is provided",
        parent=camp_node,
        critical=True
    )

    # 2) Identification (Critical) - verify it's in CA State Parks system and names are correct
    ident_node = evaluator.add_leaf(
        id=f"campground_{idx1}_identification",
        desc="Campground is correctly identified as part of California State Parks system with valid park and campground names",
        parent=camp_node,
        critical=True
    )
    ident_claim_parts = []
    park = (item.park_name or "").strip()
    cg = (item.campground_name or "").strip()
    if park:
        ident_claim_parts.append(f"the park named '{park}'")
    if cg:
        ident_claim_parts.append(f"a campground (or facility/loop) named '{cg}'")
    names_text = " and ".join(ident_claim_parts) if ident_claim_parts else "the named park/campground"
    ident_claim = f"This webpage is an official California State Parks (parks.ca.gov) or ReserveCalifornia page for {names_text}, which belongs to the California State Parks system."
    await evaluator.verify(
        claim=ident_claim,
        node=ident_node,
        sources=sources_allowed,
        additional_instruction="Confirm that the page is indeed for a California State Park or its campground/facility on either parks.ca.gov or reservecalifornia.com. Allow minor name variations (punctuation/case)."
    )

    # 3) Hookups aggregator (Critical): both water+electric, and amp specs
    hookups_node = evaluator.add_parallel(
        id=f"campground_{idx1}_hookups",
        desc="Campground offers RV sites with both water and electric hookups",
        parent=camp_node,
        critical=True
    )

    # 3.a) Water + Electric presence (Critical under hookups)
    hookups_both_node = evaluator.add_leaf(
        id=f"campground_{idx1}_hookups_both",
        desc="Campground offers RV sites with both water and electric hookups",
        parent=hookups_node,
        critical=True
    )
    claim_hookups = f"{campground_label(item)} offers RV campsites with both water and electric hookups (or equivalent 'full hookups' that clearly include water and electric)."
    await evaluator.verify(
        claim=claim_hookups,
        node=hookups_both_node,
        sources=sources_allowed,
        additional_instruction="Accept phrasing like 'water and electric hookups', 'W/E', or 'full hookups' that explicitly include both water and electric. Focus on RV hookup sites, not tent-only sites."
    )

    # 3.b) Electrical specs (Critical to satisfy strict hookups info; adjusted from NON-CRITICAL to CRITICAL for consistency)
    electric_specs_node = evaluator.add_leaf(
        id=f"campground_{idx1}_electric_specs",
        desc="Accurate electrical service specifications (amp ratings) are provided",
        parent=hookups_node,
        critical=True
    )
    elec_text = (item.electrical_service or "").strip()
    if elec_text:
        claim_elec = f"The RV hookup electrical service at {campground_label(item)} includes the following amp ratings: {elec_text}."
    else:
        # If missing in the answer, still verify as likely to fail (encourages proper citation)
        claim_elec = f"The RV hookup electrical service at {campground_label(item)} includes clearly specified amp ratings."
    await evaluator.verify(
        claim=claim_elec,
        node=electric_specs_node,
        sources=sources_allowed,
        additional_instruction="Look for mentions like 20 amp, 30 amp, 50 amp, or combined '20/30/50 amp' on the referenced page(s). Minor formatting differences are acceptable."
    )

    # 4) RV max length at hookup sites (Critical)
    rv_length_node = evaluator.add_leaf(
        id=f"campground_{idx1}_rv_length",
        desc="Maximum RV length at hookup sites is at least 30 feet and accurately stated",
        parent=camp_node,
        critical=True
    )
    length_text = (item.max_rv_length_hookup or "").strip()
    if length_text:
        claim_length = f"The maximum RV length accommodated at RV hookup sites for {campground_label(item)} is {length_text}, which is at least 30 feet."
    else:
        claim_length = f"The maximum RV length accommodated at RV hookup sites for {campground_label(item)} is at least 30 feet."
    await evaluator.verify(
        claim=claim_length,
        node=rv_length_node,
        sources=sources_allowed,
        additional_instruction="Confirm the stated max vehicle length for RV hookup sites (not tent-only). Allow rounded values; ensure it meets or exceeds 30 ft."
    )

    # 5) Showers (Critical)
    showers_node = evaluator.add_leaf(
        id=f"campground_{idx1}_showers",
        desc="Campground provides restroom facilities with hot showers available to campers",
        parent=camp_node,
        critical=True
    )
    claim_showers = f"{campground_label(item)} provides restroom facilities with hot showers available to campers."
    await evaluator.verify(
        claim=claim_showers,
        node=showers_node,
        sources=sources_allowed,
        additional_instruction="On CA State Parks/ReserveCalifornia pages, 'hot showers' are often implied by 'showers'. If 'showers' are listed, assume hot unless the page says otherwise."
    )

    # 6) Reservation system (Critical)
    resys_node = evaluator.add_leaf(
        id=f"campground_{idx1}_reservation_system",
        desc="Campground uses the ReserveCalifornia reservation system",
        parent=camp_node,
        critical=True
    )
    claim_resys = f"Reservations for {campground_label(item)} are handled via ReserveCalifornia (reservecalifornia.com)."
    await evaluator.verify(
        claim=claim_resys,
        node=resys_node,
        sources=sources_allowed,
        additional_instruction="Look for explicit ReserveCalifornia mentions or direct booking links to reservecalifornia.com. Ignore third-party booking sites."
    )

    # 7) Open during July 2026 (Critical)
    july_open_node = evaluator.add_leaf(
        id=f"campground_{idx1}_july_2026_availability",
        desc="Campground is operationally open during July 2026",
        parent=camp_node,
        critical=True
    )
    claim_july = f"{campground_label(item)} is open for camping during July 2026."
    await evaluator.verify(
        claim=claim_july,
        node=july_open_node,
        sources=sources_allowed,
        additional_instruction="Accept if the page indicates the campground is 'open year-round' or otherwise clearly open in July. If the page clearly shows seasonal closure overlapping July, mark as not supported."
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
    Evaluate an answer for the California State Parks RV campground task and return a structured result.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel so each campground is independent
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

    # Extract up to 3 campgrounds from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_campgrounds(),
        template_class=CampgroundsExtraction,
        extraction_name="campgrounds_extraction",
    )

    # Normalize to exactly 3 entries (pad if needed)
    items = list(extracted.campgrounds or [])
    if len(items) < 3:
        items = items + [CampgroundItem() for _ in range(3 - len(items))]
    items = items[:3]

    # Add three top-level campground nodes
    cg_titles = ["First campground identified meets all requirements",
                 "Second campground identified meets all requirements",
                 "Third campground identified meets all requirements"]

    camp_nodes = []
    for i in range(3):
        camp_node = evaluator.add_parallel(
            id=f"campground_{i+1}",
            desc=cg_titles[i],
            parent=root,
            critical=False  # Non-critical at root level to allow partial credit across the 3
        )
        camp_nodes.append(camp_node)

    # Verify each campground subtree
    for i, (camp_node, item) in enumerate(zip(camp_nodes, items)):
        await verify_campground(evaluator, camp_node, item, i)

    # Optional: record the high-level requirements as custom info
    evaluator.add_custom_info(
        info={
            "requirements": [
                "Water + electric RV hookups",
                ">= 30 ft RV length at hookup sites",
                "Restrooms with hot showers",
                "ReserveCalifornia reservation system",
                "Open during July 2026",
                "Reservations up to 6 months in advance (global system rule, not individually re-verified here)"
            ],
            "allowed_domains": ["parks.ca.gov", "reservecalifornia.com"]
        },
        info_type="context",
        info_name="task_requirements"
    )

    return evaluator.get_summary()