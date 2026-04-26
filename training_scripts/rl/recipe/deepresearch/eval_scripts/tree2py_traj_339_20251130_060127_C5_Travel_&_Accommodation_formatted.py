import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "universal_orlando_planner_2025"
TASK_DESCRIPTION = (
    "I'm planning a Universal Orlando Resort vacation in 2025 and want to maximize savings and benefits. I plan to "
    "book a vacation package through American Airlines Vacations during their Cyber Week promotion "
    "(November 27 - December 6). My total package cost will be approximately $4,000.\n\n"
    "Please provide the following information:\n\n"
    "1. Name at least one Universal Orlando Premier hotel that includes complimentary Universal Express Unlimited Pass "
    "as a benefit for guests (so I don't have to purchase it separately). Confirm that this hotel also provides Early Park Admission.\n\n"
    "2. Based on my $4,000 package cost, which American Airlines Vacations Cyber Week promo code should I use, and how much will I save?\n\n"
    "3. What is the average stated daily value per person of the Universal Express Unlimited Pass benefit that's included with the Premier hotel stay?\n\n"
    "4. Since I'll be connecting through Denver International Airport on my way to Orlando, I have a Capital One Venture X card. "
    "Where is the Capital One Lounge located at Denver airport (which concourse, near which gate, and on which level)?"
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PremierHotelExtraction(BaseModel):
    hotel_name: Optional[str] = None
    hotel_sources: List[str] = Field(default_factory=list)
    includes_express_unlimited_mentioned: Optional[str] = None
    includes_early_park_admission_mentioned: Optional[str] = None


class AAVDiscountExtraction(BaseModel):
    promo_code: Optional[str] = None
    savings_amount: Optional[str] = None
    aav_sources: List[str] = Field(default_factory=list)


class ExpressPassValueExtraction(BaseModel):
    value_per_person_per_day: Optional[str] = None
    value_sources: List[str] = Field(default_factory=list)


class DENLoungeExtraction(BaseModel):
    concourse: Optional[str] = None
    near_gate: Optional[str] = None
    level: Optional[str] = None
    den_lounge_sources: List[str] = Field(default_factory=list)


class UniversalTripPlanExtraction(BaseModel):
    premier_hotel: Optional[PremierHotelExtraction] = None
    aav_discount: Optional[AAVDiscountExtraction] = None
    express_value: Optional[ExpressPassValueExtraction] = None
    den_lounge: Optional[DENLoungeExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_info() -> str:
    return """
Extract the requested information from the answer for the Universal Orlando planning scenario. Return a single JSON object following this exact schema:

{
  "premier_hotel": {
    "hotel_name": string or null,
    "hotel_sources": array of urls (strings),
    "includes_express_unlimited_mentioned": string or null,
    "includes_early_park_admission_mentioned": string or null
  },
  "aav_discount": {
    "promo_code": string or null,
    "savings_amount": string or null,
    "aav_sources": array of urls (strings)
  },
  "express_value": {
    "value_per_person_per_day": string or null,
    "value_sources": array of urls (strings)
  },
  "den_lounge": {
    "concourse": string or null,
    "near_gate": string or null,
    "level": string or null,
    "den_lounge_sources": array of urls (strings)
  }
}

Detailed instructions:
- premier_hotel.hotel_name: Extract the single hotel name the answer claims as a qualifying Universal Orlando Premier hotel. If the answer lists multiple, pick one (the first mentioned).
- premier_hotel.hotel_sources: Extract all URLs the answer cites that directly support the hotel's benefits (Express Unlimited, Early Park Admission). Include hotel brand pages or Universal pages if provided. If none are present, return [].
- aav_discount.promo_code: Extract the specific American Airlines Vacations Cyber Week promo code the answer recommends for a package around $4,000.
- aav_discount.savings_amount: Extract the stated savings amount for that ~$4,000 package with the given code (e.g., "$400", "10%", "$350"). Keep the text exactly as shown in the answer.
- aav_discount.aav_sources: All URLs cited to support the Cyber Week promotion tiers, code, or savings; [] if none.
- express_value.value_per_person_per_day: Extract the stated average per-person, per-day value for Universal Express Unlimited included with Premier hotel stays (e.g., "$129 per person per day", "up to $199 per person, per day").
- express_value.value_sources: All URLs cited that support this value; [] if none.
- den_lounge: Extract the Capital One Lounge location details at Denver (DEN): concourse (e.g., "A"), near_gate (e.g., "A34"), and level (e.g., "mezzanine level", "Level 2"). Extract any URLs cited for this info into den_lounge_sources.

Special rules for URL extraction:
- Only include URLs explicitly present in the answer text (plain URLs or markdown links).
- If the URL appears without protocol, prepend http://
- If no URLs are provided for a section, return an empty array for that section's sources.

If any field is not present in the answer, set it to null (or [] for arrays). Do not fabricate data.
"""


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_premier_hotel(
    evaluator: Evaluator,
    parent,
    extracted: UniversalTripPlanExtraction
):
    node = evaluator.add_parallel(
        id="premier_hotel_with_benefits",
        desc="Provide at least one qualifying Universal Orlando Premier hotel and confirm required on-site benefits",
        parent=parent,
        critical=True
    )

    hotel_name = None
    hotel_sources: List[str] = []
    if extracted and extracted.premier_hotel:
        hotel_name = extracted.premier_hotel.hotel_name
        hotel_sources = extracted.premier_hotel.hotel_sources or []

    # Leaf: hotel_name
    leaf_hotel_name = evaluator.add_leaf(
        id="hotel_name",
        desc="Name at least one qualifying Premier hotel (Loews Portofino Bay Hotel, Hard Rock Hotel, or Loews Royal Pacific Resort)",
        parent=node,
        critical=True
    )
    claim_hotel_name = (
        f"The hotel named '{hotel_name}' is one of Universal Orlando's Premier hotels: "
        f"Loews Portofino Bay Hotel, Hard Rock Hotel, or Loews Royal Pacific Resort."
    )
    await evaluator.verify(
        claim=claim_hotel_name,
        node=leaf_hotel_name,
        additional_instruction=(
            "Verify if the provided hotel name is one of the three Universal Orlando Premier hotels. "
            "Allow minor variations like 'at Universal Orlando' or punctuation differences. "
            "If the name is missing or is not one of these three, mark incorrect."
        )
    )

    # Leaf: includes_express_unlimited
    leaf_express = evaluator.add_leaf(
        id="includes_express_unlimited",
        desc="Confirm the hotel stay includes complimentary Universal Express Unlimited Pass (Express Unlimited) for guests",
        parent=node,
        critical=True
    )
    claim_express = (
        f"Staying at {hotel_name} includes Universal Express Unlimited access for each registered guest "
        f"as a complimentary benefit (no separate purchase required)."
    )
    await evaluator.verify(
        claim=claim_express,
        node=leaf_express,
        sources=hotel_sources,
        additional_instruction=(
            "Check the cited hotel or Universal pages for 'Universal Express Unlimited' included with the stay. "
            "Accept equivalent wording (e.g., 'complimentary Universal Express Unlimited', 'included Express Unlimited')."
        )
    )

    # Leaf: includes_early_park_admission
    leaf_early = evaluator.add_leaf(
        id="includes_early_park_admission",
        desc="Confirm the hotel provides Early Park Admission",
        parent=node,
        critical=True
    )
    claim_early = f"Guests at {hotel_name} receive Early Park Admission at Universal Orlando."
    await evaluator.verify(
        claim=claim_early,
        node=leaf_early,
        sources=hotel_sources,
        additional_instruction=(
            "Verify the benefit 'Early Park Admission' is listed for the specified hotel. "
            "Accept synonyms like 'Early Park Entry' if clearly referring to Universal Orlando."
        )
    )


async def build_and_verify_aav_discount(
    evaluator: Evaluator,
    parent,
    extracted: UniversalTripPlanExtraction
):
    node = evaluator.add_parallel(
        id="aav_cyberweek_discount",
        desc="Select the correct American Airlines Vacations Cyber Week promo code for a ~$4,000 package and state the savings amount",
        parent=parent,
        critical=True
    )

    promo_code = None
    savings_amount = None
    aav_sources: List[str] = []
    if extracted and extracted.aav_discount:
        promo_code = extracted.aav_discount.promo_code
        savings_amount = extracted.aav_discount.savings_amount
        aav_sources = extracted.aav_discount.aav_sources or []

    # Leaf: promo_code
    leaf_code = evaluator.add_leaf(
        id="promo_code",
        desc="Provide the correct Cyber Week promo code applicable to a ~$4,000 package total",
        parent=node,
        critical=True
    )
    claim_code = (
        f"For an American Airlines Vacations Cyber Week booking of around $4,000, the correct promo code is '{promo_code}'."
    )
    await evaluator.verify(
        claim=claim_code,
        node=leaf_code,
        sources=aav_sources,
        additional_instruction=(
            "Check the Cyber Week page or terms for the promo code tiers and applicability. "
            "Confirm the named code is valid during November 27–December 6 and is the code intended for the ~$4,000 tier "
            "(or a tier for which $4,000 qualifies)."
        )
    )

    # Leaf: savings_amount
    leaf_savings = evaluator.add_leaf(
        id="savings_amount",
        desc="State how much will be saved using that code for a ~$4,000 package",
        parent=node,
        critical=True
    )
    claim_savings = (
        f"Using the promo code '{promo_code}' for a ~$4,000 American Airlines Vacations Cyber Week package yields savings of {savings_amount}."
    )
    await evaluator.verify(
        claim=claim_savings,
        node=leaf_savings,
        sources=aav_sources,
        additional_instruction=(
            "Verify the savings amount for the $4,000 tier (or that $4,000 meets the 'X or more' threshold). "
            "If the source presents a tiered table (e.g., save $400 on $4,000+), the stated savings must match the tier. "
            "Allow minor rounding (e.g., $3,999 still qualifies for $4,000+)."
        )
    )


async def build_and_verify_express_value(
    evaluator: Evaluator,
    parent,
    extracted: UniversalTripPlanExtraction
):
    # Single-leaf under root
    value_str = None
    value_sources: List[str] = []
    if extracted and extracted.express_value:
        value_str = extracted.express_value.value_per_person_per_day
        value_sources = extracted.express_value.value_sources or []

    leaf = evaluator.add_leaf(
        id="express_pass_average_value",
        desc="State the average stated daily value per person of the included Universal Express Unlimited Pass benefit",
        parent=parent,
        critical=True
    )
    claim = (
        f"The stated average daily value per person for Universal Express Unlimited included with Premier hotels is {value_str}."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=value_sources,
        additional_instruction=(
            "Accept phrasing like 'a value of' or 'up to' if it clearly conveys the marketed per-person, per-day value "
            "of the Universal Express Unlimited benefit tied to Premier hotel stays."
        )
    )


async def build_and_verify_den_lounge(
    evaluator: Evaluator,
    parent,
    extracted: UniversalTripPlanExtraction
):
    node = evaluator.add_parallel(
        id="den_capital_one_lounge_location",
        desc="Provide the Capital One Lounge location at Denver International Airport (concourse, nearby gate, and level)",
        parent=parent,
        critical=True
    )

    concourse = None
    near_gate = None
    level = None
    den_sources: List[str] = []
    if extracted and extracted.den_lounge:
        concourse = extracted.den_lounge.concourse
        near_gate = extracted.den_lounge.near_gate
        level = extracted.den_lounge.level
        den_sources = extracted.den_lounge.den_lounge_sources or []

    # Leaf: concourse
    leaf_concourse = evaluator.add_leaf(
        id="concourse",
        desc="State which concourse the Capital One Lounge is in at DEN",
        parent=node,
        critical=True
    )
    claim_concourse = f"The Capital One Lounge at Denver International Airport is located in Concourse {concourse}."
    await evaluator.verify(
        claim=claim_concourse,
        node=leaf_concourse,
        sources=den_sources,
        additional_instruction=(
            "Verify the concourse (A/B/C). Allow case-insensitive matches. Prefer Capital One official page or airport site."
        )
    )

    # Leaf: near_gate
    leaf_near_gate = evaluator.add_leaf(
        id="near_gate",
        desc="State which gate (or approximate gate area) the lounge is near",
        parent=node,
        critical=True
    )
    claim_gate = f"The Capital One Lounge at DEN is near gate {near_gate}."
    await evaluator.verify(
        claim=claim_gate,
        node=leaf_near_gate,
        sources=den_sources,
        additional_instruction=(
            "Verify the approximate gate reference (e.g., 'near A34', 'across from A34'). "
            "Allow minor wording variations indicating proximity."
        )
    )

    # Leaf: level
    leaf_level = evaluator.add_leaf(
        id="level",
        desc="State what level the lounge is on (e.g., mezzanine level)",
        parent=node,
        critical=True
    )
    claim_level = f"The Capital One Lounge at DEN is on the {level}."
    await evaluator.verify(
        claim=claim_level,
        node=leaf_level,
        sources=den_sources,
        additional_instruction=(
            "Verify the level/floor description (e.g., 'mezzanine', 'Level 2'). "
            "Allow synonyms as long as they clearly refer to the same level."
        )
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Universal Orlando 2025 trip planning task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates 4 critical sections in parallel
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_trip_info(),
        template_class=UniversalTripPlanExtraction,
        extraction_name="trip_plan_extraction",
    )

    # Build and verify each critical section
    await build_and_verify_premier_hotel(evaluator, root, extracted)
    await build_and_verify_aav_discount(evaluator, root, extracted)
    await build_and_verify_express_value(evaluator, root, extracted)
    await build_and_verify_den_lounge(evaluator, root, extracted)

    # Return the standard evaluation summary
    return evaluator.get_summary()