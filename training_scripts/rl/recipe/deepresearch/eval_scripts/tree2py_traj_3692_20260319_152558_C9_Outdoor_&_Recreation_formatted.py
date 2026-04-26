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
TASK_ID = "rv_state_parks_requirements"
TASK_DESCRIPTION = """
A multi-family group is planning a cross-country RV camping trip and needs to identify suitable state parks that can accommodate their specific requirements. They need to find four different state parks, each located in a different US state, that meet all of the following criteria:

1. The park must offer full-hookup RV campsites that include water, electricity, and sewer connections
2. The park must provide 50-amp electrical service at RV sites
3. The park must have ADA-accessible campsites with appropriate accommodations
4. The park must accept online reservations through a reservation system
5. The park must offer either beach/waterfront access OR hiking trails that are at least 2 miles in length

For each of the four state parks you identify, provide:
- The name of the state park
- The US state where it is located
- Confirmation that it meets all five camping and recreational requirements listed above
- Reference URLs that verify the camping facilities and recreational activities information
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ParkItem(BaseModel):
    """Single park entry extracted from the answer."""
    name: Optional[str] = None
    state: Optional[str] = None
    camping_urls: List[str] = Field(default_factory=list)
    recreation_urls: List[str] = Field(default_factory=list)


class ParksExtraction(BaseModel):
    """Top-level extraction: list of parks."""
    parks: List[ParkItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_parks() -> str:
    return """
    Extract up to four state parks mentioned in the answer. For each park, return:
    - name: the state park's name, as written in the answer
    - state: the U.S. state where the park is located (prefer full state name if available; otherwise the 2-letter code)
    - camping_urls: an array of URLs explicitly listed in the answer that substantiate camping facility details (e.g., hookups, 50-amp, ADA sites, reservations)
    - recreation_urls: an array of URLs explicitly listed in the answer that substantiate recreation features (e.g., beach/waterfront access, hiking trails)

    IMPORTANT:
    - Only include URLs explicitly present in the answer (including markdown links). Do not invent or infer.
    - Normalize any missing protocol by prepending http:// if necessary.
    - Preserve order of URLs as they appear in the answer.
    - If some field is missing, set it to null (for strings) or [] (for arrays).

    Return a JSON with a 'parks' array of up to 4 ParkItem objects, in the same order as they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _norm_state(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return s.strip().lower()


def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        uu = u.strip()
        if not uu:
            continue
        if uu not in seen:
            seen.add(uu)
            out.append(uu)
    return out


def _combine_urls(*url_lists: List[str]) -> List[str]:
    combined: List[str] = []
    for lst in url_lists:
        combined.extend(lst or [])
    return _dedup_urls(combined)


# --------------------------------------------------------------------------- #
# Verification for a single park                                              #
# --------------------------------------------------------------------------- #
async def verify_single_park(
    evaluator: Evaluator,
    parent_node,
    park: ParkItem,
    idx: int,
    state_counts: Dict[str, int],
) -> None:
    """
    Build the verification subtree for one park and run URL-grounded checks.
    """
    pfx = f"park_{idx + 1}"

    # Create the park-level node (non-critical to allow partial across different parks)
    park_node = evaluator.add_parallel(
        id=pfx,
        desc=f"{['First', 'Second', 'Third', 'Fourth'][idx]} state park meets all requirements",
        parent=parent_node,
        critical=False
    )

    # Identity (critical)
    identity_node = evaluator.add_custom_node(
        result=bool(park and park.name and park.name.strip()) and bool(park and park.state and park.state.strip()),
        id=f"{pfx}_identity",
        desc="State park name and state location are provided",
        parent=park_node,
        critical=True
    )

    # Camping facilities (critical)
    camping_node = evaluator.add_parallel(
        id=f"{pfx}_camping_facilities",
        desc="Camping facilities meet all specifications",
        parent=park_node,
        critical=True
    )

    # Hookup specifications (critical)
    hookup_node = evaluator.add_parallel(
        id=f"{pfx}_hookup_specifications",
        desc="RV hookup specifications are met",
        parent=camping_node,
        critical=True
    )

    # Full-hookup (critical leaf, URL-verified)
    full_hookup_leaf = evaluator.add_leaf(
        id=f"{pfx}_full_hookup",
        desc="Park offers full-hookup RV sites with water, electricity, and sewer",
        parent=hookup_node,
        critical=True
    )
    full_hookup_claim = (
        f"The official or reservation page(s) for '{park.name}' in {park.state} indicate that the campground offers "
        f"full-hookup RV campsites (i.e., water, electricity, and sewer connections). "
        f"Accept synonyms such as 'full hookup', 'full hook-ups', 'W/E/S', or explicit listing of water + electric + sewer."
    )
    await evaluator.verify(
        claim=full_hookup_claim,
        node=full_hookup_leaf,
        sources=_dedup_urls(park.camping_urls),
        additional_instruction="Confirm that at least some RV sites are full-hookup (water + electric + sewer)."
    )

    # 50-amp service (critical leaf, URL-verified)
    fifty_amp_leaf = evaluator.add_leaf(
        id=f"{pfx}_50_amp",
        desc="Park provides 50-amp electrical service at RV sites",
        parent=hookup_node,
        critical=True
    )
    fifty_amp_claim = (
        f"The campground for '{park.name}' in {park.state} provides 50-amp electrical service at some RV sites. "
        f"Accept variants like '50 amp', '50-amp', '50A'."
    )
    await evaluator.verify(
        claim=fifty_amp_claim,
        node=fifty_amp_leaf,
        sources=_dedup_urls(park.camping_urls),
        additional_instruction="Verify that 50-amp service is available at RV sites (not just 30-amp)."
    )

    # Accessibility and online booking (critical)
    access_book_node = evaluator.add_parallel(
        id=f"{pfx}_accessibility_and_booking",
        desc="Accessibility and reservation requirements are met",
        parent=camping_node,
        critical=True
    )

    # ADA-accessible sites (critical leaf, URL-verified)
    ada_leaf = evaluator.add_leaf(
        id=f"{pfx}_ada_accessible",
        desc="Park has ADA-accessible campsites with required features",
        parent=access_book_node,
        critical=True
    )
    ada_claim = (
        f"'{park.name}' in {park.state} offers ADA-accessible or accessible-designated campsites with appropriate accommodations."
    )
    await evaluator.verify(
        claim=ada_claim,
        node=ada_leaf,
        sources=_dedup_urls(park.camping_urls),
        additional_instruction="Look for phrases like 'ADA-accessible', 'accessible campsites', 'accessible features'."
    )

    # Online reservations accepted (critical leaf, URL-verified)
    reservations_leaf = evaluator.add_leaf(
        id=f"{pfx}_reservations",
        desc="Park accepts online reservations through a booking system",
        parent=access_book_node,
        critical=True
    )
    reservations_claim = (
        f"'{park.name}' in {park.state} accepts online reservations via an official reservation portal or booking system."
    )
    await evaluator.verify(
        claim=reservations_claim,
        node=reservations_leaf,
        sources=_combine_urls(park.camping_urls, park.recreation_urls),
        additional_instruction="Accept platforms such as ReserveAmerica, state reservation portals, or official booking links."
    )

    # Camping URL provided (critical leaf - existence check only)
    camping_url_exist = evaluator.add_custom_node(
        result=bool(park.camping_urls and len(park.camping_urls) > 0),
        id=f"{pfx}_camping_url",
        desc="URL reference provided for camping facilities information",
        parent=camping_node,
        critical=True
    )

    # Recreation (critical)
    recreation_node = evaluator.add_parallel(
        id=f"{pfx}_recreation",
        desc="Park offers required recreational activities",
        parent=park_node,
        critical=True
    )

    rec_features_node = evaluator.add_parallel(
        id=f"{pfx}_recreation_features",
        desc="Recreation features meet requirements",
        parent=recreation_node,
        critical=True
    )

    # Beach or Trails >= 2 miles (critical leaf, URL-verified)
    beach_or_trails_leaf = evaluator.add_leaf(
        id=f"{pfx}_beach_or_trails",
        desc="Park has either beach/waterfront access OR hiking trails of at least 2 miles",
        parent=rec_features_node,
        critical=True
    )
    beach_or_trails_claim = (
        f"'{park.name}' in {park.state} offers either (a) beach or waterfront/lakeshore access, "
        f"or (b) hiking with at least one trail that is 2 miles (≈3.2 km) or longer."
    )
    await evaluator.verify(
        claim=beach_or_trails_claim,
        node=beach_or_trails_leaf,
        sources=_dedup_urls(park.recreation_urls if park.recreation_urls else park.camping_urls),
        additional_instruction="Accept evidence of beach, swimming beach, waterfront, shoreline, or a clearly stated trail length ≥ 2 miles (or ≥ 3.2 km)."
    )

    # Recreation URL provided (critical leaf - existence check only)
    recreation_url_exist = evaluator.add_custom_node(
        result=bool(park.recreation_urls and len(park.recreation_urls) > 0),
        id=f"{pfx}_recreation_url",
        desc="URL reference provided for recreational activities information",
        parent=recreation_node,
        critical=True
    )

    # Different state constraint (critical)
    norm_state = _norm_state(park.state)
    unique_state = bool(norm_state) and state_counts.get(norm_state, 0) == 1
    evaluator.add_custom_node(
        result=unique_state,
        id=f"{pfx}_location",
        desc="Park is located in a different state than the other three parks",
        parent=park_node,
        critical=True
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
    Evaluate an answer for the RV state parks requirements task.
    Builds a verification tree that:
      - Extracts up to 4 state parks and their verification URLs
      - Verifies all required facility and recreation claims with URL evidence
      - Ensures all four parks are in different states
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel across the four parks
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Successfully identify four qualifying state parks meeting all specified camping and recreational requirements",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract parks from answer
    extracted: ParksExtraction = await evaluator.extract(
        prompt=prompt_extract_parks(),
        template_class=ParksExtraction,
        extraction_name="parks_extraction",
    )

    # Normalize to exactly 4 parks (pad with blanks or trim)
    parks: List[ParkItem] = list(extracted.parks[:4])
    while len(parks) < 4:
        parks.append(ParkItem())

    # Build state counts (case-insensitive) for uniqueness checks
    norm_states = [_norm_state(p.state) for p in parks]
    state_counts: Dict[str, int] = {}
    for ns in norm_states:
        if not ns:
            continue
        state_counts[ns] = state_counts.get(ns, 0) + 1

    # Add custom info for debugging/visibility
    evaluator.add_custom_info(
        info={
            f"park_{i+1}": {"name": parks[i].name, "state": parks[i].state,
                            "camping_urls": parks[i].camping_urls,
                            "recreation_urls": parks[i].recreation_urls}
            for i in range(4)
        },
        info_type="extracted_parks_overview",
        info_name="extracted_parks_overview"
    )

    # Verify each park subtree
    for i in range(4):
        await verify_single_park(evaluator, root, parks[i], i, state_counts)

    return evaluator.get_summary()