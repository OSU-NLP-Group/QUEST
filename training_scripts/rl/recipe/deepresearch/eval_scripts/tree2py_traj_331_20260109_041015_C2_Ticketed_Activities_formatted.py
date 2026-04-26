import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nyc_performance_venue_tours_2024"
TASK_DESCRIPTION = (
    "Identify a performance venue in New York City that has a seating capacity between "
    "2,500 and 6,000 seats and offers public guided tours. Provide the venue's name, its exact "
    "seating capacity, and the adult tour admission price as of 2024."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueSources(BaseModel):
    general: List[str] = Field(default_factory=list)
    location: List[str] = Field(default_factory=list)
    capacity: List[str] = Field(default_factory=list)
    tours: List[str] = Field(default_factory=list)
    price: List[str] = Field(default_factory=list)


class VenueExtraction(BaseModel):
    venue_name: Optional[str] = None
    seating_capacity: Optional[str] = None  # Keep as string; we'll parse
    adult_tour_price: Optional[str] = None  # Keep as string "$35", "USD 39", etc.
    price_year_reference: Optional[str] = None  # e.g., "2024" if explicitly stated
    sources: VenueSources = Field(default_factory=VenueSources)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return """
    From the provided answer, extract the details for the single primary performance venue the answer proposes.

    You must return a JSON object with the following fields:
    - venue_name: The name of the selected performance venue in New York City.
    - seating_capacity: The exact seating capacity value if the answer presents a single, specific number (e.g., "3,600", "2,750"). 
        Do NOT put a range (e.g., "2,500–3,000") or approximate text (e.g., "about 3,000") here. 
        If the answer does NOT give a single exact capacity value, return null for seating_capacity.
    - adult_tour_price: The adult tour admission price as stated in the answer (e.g., "$35", "USD 39").
        If the answer does not provide an adult price, return null.
    - price_year_reference: If the answer explicitly ties the adult tour price to a year (e.g., "as of 2024", "2024 price"), extract that year as a 4-digit string such as "2024".
        Otherwise, return null. Do NOT guess or infer a year.
    - sources: A structured grouping of the URLs explicitly provided in the answer:
        * general: All URLs cited that are relevant to the venue generally.
        * location: URLs that specifically support the location (NYC) of the venue (if provided).
        * capacity: URLs that specifically support the seating capacity (if provided).
        * tours: URLs that specifically support that public guided tours are offered (if provided).
        * price: URLs that specifically support the adult tour price (if provided).

    SPECIAL RULES FOR URL EXTRACTION:
    - Extract only actual URLs explicitly present in the answer (including markdown links).
    - Do not invent URLs. If none are provided for a category, return an empty array for that category.
    - If the answer provides a single sources section, place all URLs in 'general'.

    IMPORTANT:
    - Do not add or infer information not in the answer.
    - If multiple venues are mentioned, choose the first venue the answer ultimately selects as its recommendation, otherwise default to the first venue mentioned.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _strip_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s2 = s.strip()
    return s2 if s2 else None


def parse_single_int(text: Optional[str]) -> Optional[int]:
    """
    Parse a single integer from a seating capacity string, returning None if
    - the string is empty,
    - it appears to be a range or approximate,
    - or if there isn't a clear single integer.
    """
    if not text:
        return None
    s = text.strip().lower()

    # Heuristics for non-exact statements
    range_markers = [
        "-", "–", "—",  # hyphen/dashes indicating ranges
        " to ", "–", " between ", " and ", " thru ", " through ",
    ]
    approx_markers = ["approx", "approximately", "about", "around", "~", "circa", "ca.", "up to", "over", "more than", "less than", "nearly", "roughly"]

    # If contains obvious range markers with numbers around, treat as non-exact
    has_range_marker = any(marker in s for marker in range_markers)
    has_approx_marker = any(marker in s for marker in approx_markers)

    # Extract all numeric substrings
    nums = re.findall(r"\d[\d,]*", s)
    nums_clean = [int(n.replace(",", "")) for n in nums]

    # If approximations or clear range markers are present, reject
    if has_range_marker or has_approx_marker:
        return None

    # If multiple numbers present, it's ambiguous for a single exact capacity
    if len(nums_clean) != 1:
        return None

    return nums_clean[0]


def is_specific_capacity_value(text: Optional[str]) -> bool:
    """Check if seating_capacity text represents a single specific integer value."""
    return parse_single_int(text) is not None


def dedup_urls(url_lists: List[List[str]]) -> List[str]:
    """Deduplicate while preserving order; filter obviously invalid items."""
    seen = set()
    result: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if not isinstance(u, str):
                continue
            u2 = u.strip()
            if not u2:
                continue
            # Basic validity check; the verifier will normalize further if needed
            if not (u2.startswith("http://") or u2.startswith("https://")):
                continue
            if u2 not in seen:
                seen.add(u2)
                result.append(u2)
    return result


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_tree_and_verify(evaluator: Evaluator, extracted: VenueExtraction) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """

    # ---------------- Top-level critical node (acts as rubric root) ---------------- #
    venue_info_node = evaluator.add_parallel(
        id="VenueInformation",
        desc="Identify a NYC performance venue that meets the constraints and provide the required details (name, exact capacity, adult tour price as of 2024).",
        parent=evaluator.root,
        critical=True,
    )

    # Prepare sources buckets
    sources = extracted.sources or VenueSources()
    all_general = sources.general or []
    loc_sources = dedup_urls([sources.location, all_general, sources.capacity, sources.tours, sources.price])
    tours_sources = dedup_urls([sources.tours, all_general, sources.location])
    # You could also create others if needed later, e.g., capacity_sources, price_sources

    # ---------------- Branch: VenueMeetsConstraints (critical, parallel) ---------------- #
    constraints_node = evaluator.add_parallel(
        id="VenueMeetsConstraints",
        desc="The selected venue satisfies all stated constraints (location, capacity range, and public guided tours).",
        parent=venue_info_node,
        critical=True,
    )

    # Leaf: NYCLocation (verify via URL if available else simple)
    nyc_loc_leaf = evaluator.add_leaf(
        id="NYCLocation",
        desc="The venue is located in New York City.",
        parent=constraints_node,
        critical=True,
    )
    venue_name_for_claim = _strip_text(extracted.venue_name) or "the selected venue"
    nyc_claim = f"The venue '{venue_name_for_claim}' is located in New York City (NYC), i.e., within one of the five NYC boroughs (Manhattan, Brooklyn, Queens, the Bronx, or Staten Island)."
    await evaluator.verify(
        claim=nyc_claim,
        node=nyc_loc_leaf,
        sources=loc_sources,  # May be empty; will fall back to simple verification
        additional_instruction="Consider pages that clearly indicate the address or city. Accept mentions like 'New York, NY' or a specific NYC borough.",
    )

    # Leaf: CapacityInRange (custom check on exact numeric capacity)
    cap_in_range_bool = False
    parsed_capacity = parse_single_int(extracted.seating_capacity)
    if parsed_capacity is not None and 2500 <= parsed_capacity <= 6000:
        cap_in_range_bool = True
    evaluator.add_custom_node(
        result=cap_in_range_bool,
        id="CapacityInRange",
        desc="The venue's seating capacity is between 2,500 and 6,000 seats (inclusive).",
        parent=constraints_node,
        critical=True,
    )

    # Leaf: PublicGuidedToursOffered (verify via URL if available else simple)
    tours_leaf = evaluator.add_leaf(
        id="PublicGuidedToursOffered",
        desc="The venue offers public guided tours (not private-only and not self-guided-only).",
        parent=constraints_node,
        critical=True,
    )
    tours_claim = f"The venue '{venue_name_for_claim}' offers public guided tours (i.e., tours led by staff/guides that are open to the public, not only private or self-guided tours)."
    await evaluator.verify(
        claim=tours_claim,
        node=tours_leaf,
        sources=tours_sources,
        additional_instruction="Confirm that the venue provides scheduled or bookable public guided tours. Reject cases where tours are only private bookings or self-guided with no guide.",
    )

    # ---------------- Branch: RequiredFieldsProvided (critical, parallel) ---------------- #
    fields_node = evaluator.add_parallel(
        id="RequiredFieldsProvided",
        desc="The response provides all requested fields: venue name, exact seating capacity, and adult tour admission price as of 2024.",
        parent=venue_info_node,
        critical=True,
    )

    # VenueNameProvided (custom existence)
    name_provided = evaluator.add_custom_node(
        result=_strip_text(extracted.venue_name) is not None,
        id="VenueNameProvided",
        desc="Provides the venue's name.",
        parent=fields_node,
        critical=True,
    )

    # ExactSeatingCapacityProvided (custom exactness + single integer)
    exact_capacity_provided_bool = is_specific_capacity_value(extracted.seating_capacity)
    exact_capacity_node = evaluator.add_custom_node(
        result=exact_capacity_provided_bool,
        id="ExactSeatingCapacityProvided",
        desc="Provides the venue's exact seating capacity as a specific value (not just a range).",
        parent=fields_node,
        critical=True,
    )

    # AdultTourPriceProvided (custom existence)
    adult_price_provided_bool = _strip_text(extracted.adult_tour_price) is not None
    adult_price_node = evaluator.add_custom_node(
        result=adult_price_provided_bool,
        id="AdultTourPriceProvided",
        desc="Provides the tour admission price for adults.",
        parent=fields_node,
        critical=True,
    )

    # TourPriceAsOf2024 (simple verification against the answer text)
    price_asof_leaf = evaluator.add_leaf(
        id="TourPriceAsOf2024",
        desc="Indicates the adult tour price is as of 2024 (e.g., explicitly states 'as of 2024' or otherwise clearly ties the price to 2024).",
        parent=fields_node,
        critical=True,
    )
    # Prefer a reasoning check over the raw extracted field so the LLM examines the answer text directly
    asof_claim = "The answer explicitly indicates that the stated adult tour price is as of the year 2024 (e.g., includes 'as of 2024' or otherwise clearly ties the price to 2024)."
    await evaluator.verify(
        claim=asof_claim,
        node=price_asof_leaf,
        additional_instruction="Read the answer content and confirm that the adult tour price is clearly tied to the year 2024. If the year is not clear, mark as incorrect.",
    )

    # Record some helpful custom info (for debugging/analysis)
    evaluator.add_custom_info(
        info={
            "parsed_capacity": parsed_capacity,
            "exact_capacity_string": extracted.seating_capacity,
            "adult_tour_price": extracted.adult_tour_price,
            "price_year_reference_extracted": extracted.price_year_reference,
            "venue_name": extracted.venue_name,
            "location_sources_used": loc_sources,
            "tours_sources_used": tours_sources,
        },
        info_type="debug",
        info_name="parsed_values"
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
    Evaluate an answer for the NYC performance venue tours (2024) task and return a structured result.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    _ = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation is parallel
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

    # Extract structured information from the answer
    extracted: VenueExtraction = await evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction",
    )

    # Build the verification tree per rubric and run verifications
    await build_tree_and_verify(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()