import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "accessible_parks_allegiant"
TASK_DESCRIPTION = (
    "I'm planning accessible outdoor adventures and would like to identify three U.S. national parks that meet the following criteria: "
    "(1) The park must be accessible via Allegiant Air nonstop flights to a nearby airport, and (2) The park must have at least one wheelchair-accessible "
    "visitor center with indoor exhibits (not just an information desk). For each of the three national parks you identify, please provide: the name of the "
    "national park; the three-letter airport code of the Allegiant-served airport that provides access to the park; the approximate distance in miles from that "
    "airport to the park entrance or main visitor area; a reference URL from Allegiant Air's website confirming they serve this airport for access to this national park; "
    "the name of at least one wheelchair-accessible visitor center at the park; confirmation of the visitor center's wheelchair accessibility and ADA-compliant features; "
    "a description of the indoor exhibits available at that visitor center; the typical operating hours during peak season; and a reference URL from the official "
    "National Park Service website or reputable source confirming the visitor center's accessibility and exhibits."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ParkItem(BaseModel):
    park_name: Optional[str] = None
    airport_code: Optional[str] = None  # expected IATA 3-letter
    distance_miles: Optional[str] = None  # keep as string to allow ranges/approx
    allegiant_reference_url: Optional[str] = None  # preferably allegiantair.com / allegiant.com
    vc_name: Optional[str] = None  # visitor center name
    vc_accessibility: Optional[str] = None  # textual confirmation of wheelchair/ADA accessibility
    vc_exhibits: Optional[str] = None  # textual description of indoor exhibits
    vc_hours: Optional[str] = None  # typical peak season hours, as text
    vc_reference_url: Optional[str] = None  # preferably nps.gov; reputable source also ok


class ParksExtraction(BaseModel):
    parks: List[ParkItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_parks() -> str:
    return """
    Extract up to THREE U.S. national parks and their associated details exactly as presented in the answer text. 
    Only extract items explicitly present in the answer. Do not invent or infer anything not stated.

    For each park, extract the following fields (use null if missing):
    - park_name: The national park's name (e.g., "Zion National Park").
    - airport_code: The three-letter IATA airport code for the Allegiant-served airport used for access to this park (letters only, e.g., "LAS").
    - distance_miles: The approximate distance in miles from that airport to the park entrance or main visitor area (keep the text exactly or approximately as written, e.g., "45 miles", "~60 miles", "60–70 miles").
    - allegiant_reference_url: A URL from Allegiant Air's website (allegiantair.com or allegiant.com) confirming Allegiant serves this airport; if the answer instead cites an official park source for Allegiant service, extract that URL.
    - vc_name: The name of at least one wheelchair-accessible visitor center at the park (e.g., "Zion Canyon Visitor Center").
    - vc_accessibility: The text that confirms the visitor center is wheelchair accessible and has ADA-compliant features (e.g., ramps, accessible restrooms). Keep the exact or summarized phrasing from the answer.
    - vc_exhibits: The text describing indoor exhibits available at that visitor center (e.g., exhibits, museum displays, interpretive panels, films).
    - vc_hours: Typical operating hours during peak season (as described in the answer; keep the phrasing the answer uses).
    - vc_reference_url: A URL from the official National Park Service website (nps.gov) or another reputable source confirming the visitor center's accessibility and exhibits.

    Return a JSON object with one field:
    {
      "parks": [ParkItem, ParkItem, ParkItem]   // up to three items, in the same order as in the answer
    }

    Rules:
    - If the answer lists more than three parks, include only the first three.
    - If any field is not provided in the answer text, set it to null.
    - For URLs, extract only actual URLs shown in the answer (plain or markdown links). Do not fabricate URLs.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return s is not None and str(s).strip() != ""


def _collect_sources(*urls: Optional[str]) -> List[str]:
    return [u for u in urls if _nonempty(u)]


# --------------------------------------------------------------------------- #
# Verification per-park                                                       #
# --------------------------------------------------------------------------- #
async def verify_one_park(
    evaluator: Evaluator,
    parent_node,
    park: ParkItem,
    index: int,
) -> None:
    """
    Build verification sub-tree and run checks for a single park.
    """
    park_idx = index + 1
    park_node = evaluator.add_parallel(
        id=f"park_{park_idx}",
        desc=f"{['First','Second','Third'][index]} national park meeting all criteria",
        parent=parent_node,
        critical=False  # allow partial scoring among parks
    )

    # Common sources
    combined_sources = _collect_sources(park.vc_reference_url, park.allegiant_reference_url)

    # --- Park name ---
    # Add existence gating
    evaluator.add_custom_node(
        result=_nonempty(park.park_name),
        id=f"park_{park_idx}_name_exists",
        desc=f"Park #{park_idx} name is provided in the answer",
        parent=park_node,
        critical=True
    )

    name_leaf = evaluator.add_leaf(
        id=f"park_{park_idx}_name",
        desc=f"The name of park #{park_idx}",
        parent=park_node,
        critical=True
    )
    name_claim = (
        f"The national park identified is '{park.park_name}'. Verify that at least one provided source page clearly refers to this park. "
        "Minor name variants (e.g., 'National Park & Preserve') should still count as the same park."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=combined_sources,
        additional_instruction="Accept reasonable formatting variants (e.g., with or without 'National Park')."
    )

    # --- Allegiant access group (critical) ---
    allegiant_group = evaluator.add_parallel(
        id=f"park_{park_idx}_allegiant_access",
        desc=f"Allegiant Air accessibility information for park #{park_idx}",
        parent=park_node,
        critical=True
    )

    # Airport code existence
    evaluator.add_custom_node(
        result=_nonempty(park.airport_code),
        id=f"park_{park_idx}_airport_code_exists",
        desc=f"Park #{park_idx} airport code is provided",
        parent=allegiant_group,
        critical=True
    )

    # Distance existence
    evaluator.add_custom_node(
        result=_nonempty(park.distance_miles),
        id=f"park_{park_idx}_distance_exists",
        desc=f"Park #{park_idx} approximate distance is provided",
        parent=allegiant_group,
        critical=True
    )

    # Allegiant reference existence
    evaluator.add_custom_node(
        result=_nonempty(park.allegiant_reference_url),
        id=f"park_{park_idx}_allegiant_reference_exists",
        desc=f"Park #{park_idx} Allegiant reference URL is provided",
        parent=allegiant_group,
        critical=True
    )

    # Airport code verification against Allegiant page
    airport_code_leaf = evaluator.add_leaf(
        id=f"park_{park_idx}_airport_code",
        desc=f"The three-letter airport code of the Allegiant-served airport for park #{park_idx}",
        parent=allegiant_group,
        critical=True
    )
    airport_code_claim = (
        f"The Allegiant Air website page confirms service to the airport with IATA code '{park.airport_code}'. "
        "The code should appear or be inferable from the airport/city page, destinations, or route map."
    )
    await evaluator.verify(
        claim=airport_code_claim,
        node=airport_code_leaf,
        sources=park.allegiant_reference_url,
        additional_instruction="Treat Allegiant destination/airport pages and Allegiant route maps as valid evidence."
    )

    # Distance verification (allow approximate, broad tolerance)
    distance_leaf = evaluator.add_leaf(
        id=f"park_{park_idx}_distance",
        desc=f"The approximate distance from the Allegiant-served airport to park #{park_idx}",
        parent=allegiant_group,
        critical=True
    )
    distance_claim = (
        f"The approximate distance from {park.airport_code} airport to {park.park_name} main entrance or primary visitor area "
        f"is about {park.distance_miles} (in miles). Allow ±25% tolerance and accept driving distance/time statements that clearly imply a similar mileage."
    )
    await evaluator.verify(
        claim=distance_claim,
        node=distance_leaf,
        sources=combined_sources,  # try VC/NPS page or Allegiant page if it mentions distances
        additional_instruction="Use any provided page(s) to confirm a reasonable approximate distance; accept close variants or rounded values."
    )

    # Allegiant serves airport (service confirmation)
    allegiant_ref_leaf = evaluator.add_leaf(
        id=f"park_{park_idx}_allegiant_reference",
        desc=f"Reference URL confirms Allegiant serves the airport for park #{park_idx}",
        parent=allegiant_group,
        critical=True
    )
    allegiant_ref_claim = (
        f"This page confirms that Allegiant Air serves the airport with IATA code '{park.airport_code}'. "
        "The page can be a destinations list, city/airport info page, or route map on Allegiant’s official site."
    )
    await evaluator.verify(
        claim=allegiant_ref_claim,
        node=allegiant_ref_leaf,
        sources=park.allegiant_reference_url,
        additional_instruction="If the URL is not on Allegiant’s domain, check whether it still explicitly states Allegiant service to the stated airport."
    )

    # --- Visitor center group (critical) ---
    vc_group = evaluator.add_parallel(
        id=f"park_{park_idx}_visitor_center",
        desc=f"Visitor center accessibility and features for park #{park_idx}",
        parent=park_node,
        critical=True
    )

    # Existence checks for VC details
    evaluator.add_custom_node(
        result=_nonempty(park.vc_name),
        id=f"park_{park_idx}_vc_name_exists",
        desc=f"Park #{park_idx} VC name is provided",
        parent=vc_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(park.vc_reference_url),
        id=f"park_{park_idx}_vc_reference_exists",
        desc=f"Park #{park_idx} VC reference URL is provided",
        parent=vc_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(park.vc_accessibility),
        id=f"park_{park_idx}_vc_accessibility_exists",
        desc=f"Park #{park_idx} VC accessibility description is provided",
        parent=vc_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(park.vc_exhibits),
        id=f"park_{park_idx}_vc_exhibits_exists",
        desc=f"Park #{park_idx} VC exhibits description is provided",
        parent=vc_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(park.vc_hours),
        id=f"park_{park_idx}_vc_hours_exists",
        desc=f"Park #{park_idx} VC hours are provided",
        parent=vc_group,
        critical=True
    )

    # VC name verification (exists and is an indoor visitor center)
    vc_name_leaf = evaluator.add_leaf(
        id=f"park_{park_idx}_vc_name",
        desc=f"The name of a wheelchair-accessible visitor center with indoor exhibits at park #{park_idx}",
        parent=vc_group,
        critical=True
    )
    vc_name_claim = (
        f"The provided page confirms a visitor center named '{park.vc_name}' at {park.park_name}. "
        "It should be a staffed indoor visitor center (not just an outdoor contact station or information desk)."
    )
    # VC accessibility verification
    vc_access_leaf = evaluator.add_leaf(
        id=f"park_{park_idx}_vc_accessibility",
        desc=f"Confirmation that the visitor center is wheelchair accessible with ADA features for park #{park_idx}",
        parent=vc_group,
        critical=True
    )
    vc_access_claim = (
        f"The page confirms that the {park.vc_name} visitor center is wheelchair accessible and has ADA-compliant features "
        f"(e.g., accessible entrances, ramps, accessible restrooms, assistive devices)."
    )
    # VC exhibits verification
    vc_exhibits_leaf = evaluator.add_leaf(
        id=f"park_{park_idx}_vc_exhibits",
        desc=f"Indoor exhibits are available at the visitor center for park #{park_idx}",
        parent=vc_group,
        critical=True
    )
    vc_exhibits_claim = (
        f"The page describes indoor exhibits at the {park.vc_name} visitor center (e.g., interpretive displays, museum exhibits, films)."
    )
    # VC hours verification
    vc_hours_leaf = evaluator.add_leaf(
        id=f"park_{park_idx}_vc_hours",
        desc=f"Typical peak-season operating hours for the visitor center for park #{park_idx}",
        parent=vc_group,
        critical=True
    )
    vc_hours_claim = (
        f"The page provides the typical peak-season operating hours for {park.vc_name}, approximately matching: '{park.vc_hours}'. "
        "Allow for reasonable seasonal variations and common disclaimers such as 'hours subject to change'."
    )
    # VC reference verification (provenance/fitness of source)
    vc_ref_leaf = evaluator.add_leaf(
        id=f"park_{park_idx}_vc_reference",
        desc=f"Reference URL is an official NPS page or reputable source about the VC's accessibility/exhibits for park #{park_idx}",
        parent=vc_group,
        critical=True
    )
    vc_ref_claim = (
        "This page is either on the official National Park Service domain (nps.gov) or is another reputable official source, "
        f"and it provides information about the {park.vc_name} visitor center's accessibility and/or indoor exhibits."
    )

    # Prepare batch verifications for VC group (all use VC reference URL)
    vc_verify_batch = [
        (vc_name_claim, park.vc_reference_url, vc_name_leaf, "Prefer explicit mentions that confirm the VC name and that it is a proper indoor visitor center."),
        (vc_access_claim, park.vc_reference_url, vc_access_leaf, "Look for wheelchair accessibility and ADA-related features; synonymous phrases acceptable."),
        (vc_exhibits_claim, park.vc_reference_url, vc_exhibits_leaf, "Any clear indication of indoor exhibits, museum displays, interpretive exhibits, orientation film counts."),
        (vc_hours_claim, park.vc_reference_url, vc_hours_leaf, "Accept approximate/typical peak-season hours or representative schedules."),
        (vc_ref_claim, park.vc_reference_url, vc_ref_leaf, "If on nps.gov, count as official; reputable alternatives (e.g., official state/park pages) are acceptable."),
    ]
    await evaluator.batch_verify(vc_verify_batch)

    # Note: We already awaited the Allegiant/airport/distance/name leaves individually above.


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Entry point: build the verification tree and run the evaluation.
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

    # IMPORTANT: Root cannot be critical if it has non-critical children (framework constraint).
    # We'll keep root as non-critical and enforce critical checks at lower levels.

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_parks(),
        template_class=ParksExtraction,
        extraction_name="parks_extraction",
    )

    parks = list(extracted.parks) if extracted and extracted.parks else []
    # Pad or trim to exactly 3 entries
    if len(parks) < 3:
        parks = parks + [ParkItem() for _ in range(3 - len(parks))]
    else:
        parks = parks[:3]

    # Build verification subtrees for each park
    for i in range(3):
        await verify_one_park(evaluator, root, parks[i], i)

    # Return evaluation summary
    return evaluator.get_summary()