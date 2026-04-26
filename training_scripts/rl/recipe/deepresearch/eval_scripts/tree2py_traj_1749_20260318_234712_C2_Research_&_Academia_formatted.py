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
TASK_ID = "conference_hotels_2026"
TASK_DESCRIPTION = (
    "I am a researcher planning to attend an academic conference in spring 2026 and need help finding suitable "
    "accommodation. I am considering either the ACS Spring 2026 conference (March 22-26, 2026, at the Georgia World "
    "Congress Center in Atlanta, GA) or the AERA 2026 Annual Meeting (April 8-12, 2026, at the Los Angeles Convention "
    "Center in Los Angeles, CA). Please select one of these two conferences and identify three hotels that meet ALL of "
    "the following requirements: (1) The hotel must be part of the official conference housing block offering special "
    "discounted rates to conference attendees, (2) The conference discounted rate must be $250 per night or less "
    "(before taxes), (3) The hotel must be within 0.5 miles (800 meters) walking distance from the main conference "
    "venue, (4) The hotel must be bookable through the official conference housing system. For each of the three hotels "
    "you identify, please provide: the official hotel name, the conference discounted nightly rate (before taxes), the "
    "walking distance from the conference venue, and a direct link to the hotel information on the official conference "
    "website or housing page."
)

ALLOWED_CONFERENCES = [
    "ACS Spring 2026",
    "AERA 2026 Annual Meeting",
]

# Helpful reminder strings for verification prompts
ACS_KNOWN_CONTEXT = "ACS Spring 2026 (March 22–26, 2026) at the Georgia World Congress Center in Atlanta, GA."
AERA_KNOWN_CONTEXT = "AERA 2026 Annual Meeting (April 8–12, 2026) at the Los Angeles Convention Center in Los Angeles, CA."


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ConferenceInfo(BaseModel):
    selected_conference: Optional[str] = None  # e.g., "ACS Spring 2026" or "AERA 2026 Annual Meeting"
    dates_text: Optional[str] = None           # As written in the answer, e.g., "March 22-26, 2026"
    city: Optional[str] = None                 # e.g., "Atlanta" or "Los Angeles"
    state: Optional[str] = None                # e.g., "GA" or "CA"
    venue: Optional[str] = None                # e.g., "Georgia World Congress Center"
    sources: List[str] = Field(default_factory=list)  # Official conference/housing URLs mentioned


class HotelItem(BaseModel):
    name: Optional[str] = None
    discounted_rate_text: Optional[str] = None      # e.g., "$199", "from $249"
    walking_distance_text: Optional[str] = None     # e.g., "0.3 miles", "5-minute walk", "800 m"
    direct_link_url: Optional[str] = None           # direct link to official conference website/housing page for this hotel
    source_urls: List[str] = Field(default_factory=list)  # any additional URLs cited for this hotel


class ConferenceHotelsExtraction(BaseModel):
    conference: Optional[ConferenceInfo] = None
    hotels: List[HotelItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_conference_and_hotels() -> str:
    return """
    Extract the structured information that the answer explicitly presents about the selected conference and the hotels.

    You MUST:
    - Identify which single conference the answer actually chooses and proceeds with (choose only one).
    - Capture the conference details as stated in the answer: name, dates (as a single text span), city, state, venue.
    - Extract any official conference/housing URLs that the answer cites for the conference details or hotel booking.
    - Extract up to 5 hotels that the answer explicitly claims meet the requirements, in the order they are presented.

    For EACH hotel, extract:
    1) name: the official hotel name as written in the answer.
    2) discounted_rate_text: the conference discounted nightly rate before taxes, as text (e.g., "$199", "from $249", "$250").
    3) walking_distance_text: the walking distance to the conference venue, as text (e.g., "0.3 miles", "800 m", "6-minute walk").
    4) direct_link_url: a direct URL in the answer that goes to the official conference website/housing page (not the general hotel website).
    5) source_urls: any additional URLs the answer cites for this hotel's info (can be empty).

    IMPORTANT rules for URL extraction:
    - Extract only URLs that actually appear in the answer text (including markdown links).
    - If a URL lacks protocol, prepend "http://".
    - Prefer official conference/housing pages (e.g., onPeak, Maritz, ConferenceDirect, official conference site subpages).
    - Do NOT invent URLs.

    Output JSON schema:
    {
      "conference": {
        "selected_conference": string | null,
        "dates_text": string | null,
        "city": string | null,
        "state": string | null,
        "venue": string | null,
        "sources": string[]   // can be empty
      },
      "hotels": [
        {
          "name": string | null,
          "discounted_rate_text": string | null,
          "walking_distance_text": string | null,
          "direct_link_url": string | null,
          "source_urls": string[]  // can be empty
        },
        ...
      ]
    }

    If a specific field is missing in the answer, set it to null (or empty array for lists).
    """


# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[Optional[str]]) -> List[str]:
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


def _safe_text(v: Optional[str]) -> str:
    return v if v is not None else ""


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_conference_selection_subtree(
    evaluator: Evaluator,
    parent_node,
    extraction: ConferenceHotelsExtraction,
) -> None:
    """
    Build verification subtree for conference selection and basic details.
    """
    conf = extraction.conference or ConferenceInfo()

    # Parallel node: Conference_Selection (critical)
    conf_node = evaluator.add_parallel(
        id="Conference_Selection",
        desc="Select one of the two specified conferences (ACS Spring 2026 or AERA 2026) and verify it meets the stated criteria",
        parent=parent_node,
        critical=True,
    )

    # Leaf 1: Valid_Conference_Choice (critical)
    valid_choice_leaf = evaluator.add_leaf(
        id="Valid_Conference_Choice",
        desc="The selected conference must be either ACS Spring 2026 (March 22-26, 2026, Atlanta) or AERA 2026 (April 8-12, 2026, Los Angeles)",
        parent=conf_node,
        critical=True,
    )

    # Construct claim to check selected conference set-membership with leniency for naming variants
    selected_name = _safe_text(conf.selected_conference)
    claim_valid_choice = (
        f"The selected conference is one of the two allowed options: "
        f"'ACS Spring 2026' or 'AERA 2026 Annual Meeting'. "
        f"The answer selected: '{selected_name}'. Consider reasonable naming variants like "
        f"'American Chemical Society (ACS) Spring 2026' or 'AERA 2026'."
    )
    await evaluator.verify(
        claim=claim_valid_choice,
        node=valid_choice_leaf,
        additional_instruction="Judge based on the answer text; do not require external evidence for this set-membership check.",
    )

    # Leaf 2: Conference_Details_Provided (critical, with source grounding)
    details_leaf = evaluator.add_leaf(
        id="Conference_Details_Provided",
        desc="Conference name, dates, location, and venue are correctly stated",
        parent=conf_node,
        critical=True,
    )

    # Build claim using extracted details and verify against cited conference sources
    details_claim_parts = []
    if conf.selected_conference:
        details_claim_parts.append(f"Name: {conf.selected_conference}")
    if conf.dates_text:
        details_claim_parts.append(f"Dates: {conf.dates_text}")
    if conf.city and conf.state:
        details_claim_parts.append(f"Location: {conf.city}, {conf.state}")
    elif conf.city:
        details_claim_parts.append(f"Location: {conf.city}")
    if conf.venue:
        details_claim_parts.append(f"Venue: {conf.venue}")
    details_summary = "; ".join(details_claim_parts) if details_claim_parts else "No details provided in answer."

    details_claim = (
        f"The following conference details stated in the answer are accurate per the official conference website: "
        f"{details_summary}"
    )

    await evaluator.verify(
        claim=details_claim,
        node=details_leaf,
        sources=conf.sources,  # may be empty; then falls back to simple verification
        additional_instruction=(
            "Use only the provided official conference URLs (home/housing/registration pages). "
            "Accept minor format variations (e.g., 'Mar 22–26, 2026' vs 'March 22-26, 2026'). "
            f"If the selected conference is ACS, context: {ACS_KNOWN_CONTEXT} "
            f"If AERA, context: {AERA_KNOWN_CONTEXT}."
        ),
    )


async def verify_single_hotel(
    evaluator: Evaluator,
    parent_node,
    idx: int,
    hotel: HotelItem,
    conference: ConferenceInfo,
) -> None:
    """
    Build verification subtree for a single hotel.
    """
    # Node for Hotel_i (critical so that each hotel must pass all 4 leaves)
    hotel_node = evaluator.add_parallel(
        id=f"Hotel_{idx+1}",
        desc=f"Hotel #{idx+1} meeting all requirements",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Documentation (existence check)
    has_all_fields = bool(
        (hotel.name and hotel.name.strip())
        and (hotel.discounted_rate_text and hotel.discounted_rate_text.strip())
        and (hotel.walking_distance_text and hotel.walking_distance_text.strip())
        and (hotel.direct_link_url and hotel.direct_link_url.strip())
    )
    evaluator.add_custom_node(
        result=has_all_fields,
        id=f"Hotel_{idx+1}_Documentation",
        desc=(
            f"Hotel {idx+1} name, conference discounted rate, walking distance from venue, and direct link to official "
            f"conference website/housing page are all provided"
        ),
        parent=hotel_node,
        critical=True,
    )

    # Prepare sources: prefer the hotel's direct link + any hotel sources + conference sources
    sources = _dedup_urls(
        [hotel.direct_link_url] + list(hotel.source_urls or []) + list((conference.sources or []))
    )

    # Leaf: Rate <= $250 (critical)
    rate_leaf = evaluator.add_leaf(
        id=f"Hotel_{idx+1}_Rate",
        desc=f"Hotel {idx+1} conference discounted rate is ≤ $250 per night before taxes",
        parent=hotel_node,
        critical=True,
    )
    rate_claim = (
        f"The conference discounted nightly rate (before taxes) for '{_safe_text(hotel.name)}' on the official "
        f"conference housing page is '{_safe_text(hotel.discounted_rate_text)}', and it is less than or equal to $250."
    )
    await evaluator.verify(
        claim=rate_claim,
        node=rate_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that the rate is a conference-discounted rate from the official housing portal or official conference site "
            "(not the public hotel website). Accept phrasing like 'from $X' or 'starting at $X'. The threshold is $250 before taxes."
        ),
    )

    # Leaf: Proximity within 0.5 miles (critical)
    prox_leaf = evaluator.add_leaf(
        id=f"Hotel_{idx+1}_Proximity",
        desc=f"Hotel {idx+1} is within 0.5 miles (800 meters) walking distance from the conference venue",
        parent=hotel_node,
        critical=True,
    )
    prox_claim = (
        f"'{_safe_text(hotel.name)}' is within 0.5 miles (800 meters) walking distance from the venue "
        f"'{_safe_text(conference.venue)}'. The stated walking distance is '{_safe_text(hotel.walking_distance_text)}'."
    )
    await evaluator.verify(
        claim=prox_claim,
        node=prox_leaf,
        sources=sources,
        additional_instruction=(
            "Use the official housing page if it lists distance. Accept reasonable variants like '0.5 mi', '800 m', "
            "'0.80 km', or similar. If only a minute-walk is shown, pass only if it corresponds to about ≤10 minutes typical city walking."
        ),
    )

    # Leaf: Conference affiliation + bookable via official housing (critical)
    aff_leaf = evaluator.add_leaf(
        id=f"Hotel_{idx+1}_Conference_Affiliation",
        desc=f"Hotel {idx+1} is part of the official conference housing block with discounted rates",
        parent=hotel_node,
        critical=True,
    )
    conf_name = _safe_text(conference.selected_conference)
    aff_claim = (
        f"'{_safe_text(hotel.name)}' is included in the official housing block for '{conf_name}' and is bookable "
        f"through the official conference housing system (not a general public hotel site)."
    )
    await evaluator.verify(
        claim=aff_claim,
        node=aff_leaf,
        sources=sources,
        additional_instruction=(
            "Look for indicators like 'official housing', 'book now' via the conference's portal, vendor pages such as onPeak/Maritz/"
            "ConferenceDirect, and branding explicitly tied to the selected conference."
        ),
    )


async def build_hotels_subtree(
    evaluator: Evaluator,
    parent_node,
    extraction: ConferenceHotelsExtraction,
) -> None:
    """
    Build verification subtree for identifying three qualifying hotels.
    """
    conf = extraction.conference or ConferenceInfo()
    hotels_parent = evaluator.add_parallel(
        id="Hotel_Identification",
        desc="Identify at least 3 hotels that meet all specified criteria for the selected conference",
        parent=parent_node,
        critical=True,  # All required hotels must meet constraints
    )

    # Normalize hotels: take first 3; pad if fewer
    hotels: List[HotelItem] = list((extraction.hotels or [])[:3])
    while len(hotels) < 3:
        hotels.append(HotelItem())  # placeholder will fail documentation leaf

    # Verify each of the 3 hotels
    for i in range(3):
        await verify_single_hotel(evaluator, hotels_parent, i, hotels[i], conf)


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
    Evaluate an answer for the Spring 2026 conference hotels task.

    Returns:
        A standard evaluation summary dictionary produced by the Evaluator.
    """
    # Initialize evaluator with SEQUENTIAL root (root is non-critical by design in framework)
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

    # Extract structured info
    extraction: ConferenceHotelsExtraction = await evaluator.extract(
        prompt=prompt_extract_conference_and_hotels(),
        template_class=ConferenceHotelsExtraction,
        extraction_name="conference_and_hotels_extraction",
    )

    # Add GT-like context for allowed conferences (for transparency only)
    evaluator.add_ground_truth({
        "allowed_conferences": ALLOWED_CONFERENCES,
        "acs_context": ACS_KNOWN_CONTEXT,
        "aera_context": AERA_KNOWN_CONTEXT,
    }, gt_type="constraints_context")

    # Build verification tree
    await build_conference_selection_subtree(evaluator, root, extraction)
    await build_hotels_subtree(evaluator, root, extraction)

    # Return standard summary
    return evaluator.get_summary()