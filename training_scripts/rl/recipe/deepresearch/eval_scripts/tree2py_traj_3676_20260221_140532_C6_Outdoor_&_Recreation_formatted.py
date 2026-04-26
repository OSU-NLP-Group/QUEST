import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "med_cruise_dining_2026"
TASK_DESCRIPTION = (
    "Find two cruise ships for a 2026 Mediterranean cruise - one from Oceania Cruises and one from Holland America Line - "
    "that visit Spanish ports. Provide details about their specialty dining options and pricing."
)

OCEANIA_ALLOWED_SHIPS = ["Riviera", "Marina"]
HAL_ALLOWED_SHIPS = ["Rotterdam", "Nieuw Statendam", "Koningsdam", "Eurodam", "Nieuw Amsterdam"]
SPANISH_PORT_EXAMPLES = [
    "Barcelona", "Valencia", "Cartagena", "Málaga", "Cadiz", "Cádiz", "Seville", "Palma de Mallorca",
    "Ibiza", "Gran Canaria", "Tenerife", "Alicante", "Bilbao", "Gijón", "Cádiz/Seville"
]
TARGET_YEAR = 2026

# ----------------------------- Data Models --------------------------------- #
class OceaniaShipInfo(BaseModel):
    name: Optional[str] = None
    identity_urls: List[str] = Field(default_factory=list)
    restaurants_present: List[str] = Field(default_factory=list)  # e.g., ["Polo Grill", "Toscana", "Jacques", "Red Ginger"]
    restaurants_urls: List[str] = Field(default_factory=list)
    dining_policy_text: Optional[str] = None
    dining_policy_urls: List[str] = Field(default_factory=list)
    itinerary_urls: List[str] = Field(default_factory=list)
    spanish_ports: List[str] = Field(default_factory=list)


class HollandShipInfo(BaseModel):
    name: Optional[str] = None
    identity_urls: List[str] = Field(default_factory=list)
    restaurants_present: List[str] = Field(default_factory=list)  # e.g., ["Tamarind", "Pinnacle Grill", "Canaletto"]
    restaurants_urls: List[str] = Field(default_factory=list)
    tamarind_price: Optional[str] = None
    pinnacle_price: Optional[str] = None
    canaletto_price: Optional[str] = None
    pricing_urls: List[str] = Field(default_factory=list)
    itinerary_urls: List[str] = Field(default_factory=list)
    spanish_ports: List[str] = Field(default_factory=list)


# --------------------------- Extraction Prompts ---------------------------- #
def prompt_extract_oceania_ship() -> str:
    return (
        "Extract details for the single Oceania Cruises ship selected in the answer.\n"
        "Rules:\n"
        "- Only extract URLs explicitly present in the answer text. Do not invent or infer URLs.\n"
        "- If multiple candidate ships are mentioned, choose the first one and extract fields for that ship.\n"
        "- Return null for any missing field; return empty list for missing arrays.\n\n"
        "Fields to extract:\n"
        "1) name: The ship name (prefer plain name without prefix); should be either 'Riviera' or 'Marina' if applicable.\n"
        "2) identity_urls: List of 1-5 URLs that the answer uses to identify/describe the ship (e.g., official ship page, ship overview pages). Extract only URLs that appear in the answer.\n"
        "3) restaurants_present: List of specialty restaurant names claimed in the answer for this ship; only include names from this set if present: ['Polo Grill', 'Toscana', 'Jacques', 'Red Ginger'].\n"
        "4) restaurants_urls: List of 1-5 URLs used in the answer to support the restaurant availability (e.g., dining venues page, deck plans).\n"
        "5) dining_policy_text: The answer's statement about whether these specialty restaurants are complimentary or require a cover charge (verbatim phrase or concise paraphrase from the answer).\n"
        "6) dining_policy_urls: List of 1-5 URLs used in the answer to support the dining policy statement.\n"
        "7) itinerary_urls: List of 1-5 URLs used in the answer to support the 2026 Mediterranean itinerary for this ship.\n"
        "8) spanish_ports: List of Spanish ports explicitly mentioned in the answer for this ship's 2026 Mediterranean itinerary "
        "(e.g., Barcelona, Valencia, Cartagena, Málaga, Cádiz/Seville, Palma de Mallorca, Ibiza)."
    )


def prompt_extract_hal_ship() -> str:
    return (
        "Extract details for the single Holland America Line ship selected in the answer.\n"
        "Rules:\n"
        "- Only extract URLs explicitly present in the answer text. Do not invent or infer URLs.\n"
        "- If multiple candidate ships are mentioned, choose the first one and extract fields for that ship.\n"
        "- Return null for any missing field; return empty list for missing arrays.\n\n"
        "Fields to extract:\n"
        "1) name: The ship name; must be one of ['Rotterdam', 'Nieuw Statendam', 'Koningsdam', 'Eurodam', 'Nieuw Amsterdam'] if the answer is correct.\n"
        "2) identity_urls: List of 1-5 URLs that the answer uses to identify/describe the ship.\n"
        "3) restaurants_present: List of specialty restaurant names claimed in the answer for this ship; include any of ['Tamarind', 'Pinnacle Grill', 'Canaletto'] that are claimed.\n"
        "4) restaurants_urls: List of 1-5 URLs used in the answer to support restaurant availability (dining pages, deck plans).\n"
        "5) tamarind_price: The claimed price for Tamarind (e.g., '$35 per person'), as a string.\n"
        "6) pinnacle_price: The claimed price for Pinnacle Grill (e.g., '$52 per person'), as a string.\n"
        "7) canaletto_price: The claimed price for Canaletto (e.g., '$25 per person'), as a string.\n"
        "8) pricing_urls: List of 1-5 URLs used in the answer to support the specialty restaurant pricing.\n"
        "9) itinerary_urls: List of 1-5 URLs used in the answer to support the 2026 Mediterranean itinerary for this ship.\n"
        "10) spanish_ports: List of Spanish ports explicitly mentioned in the answer for this ship's 2026 Mediterranean itinerary "
        "(e.g., Barcelona, Valencia, Cartagena, Málaga, Cádiz/Seville, Palma de Mallorca, Ibiza)."
    )


# ---------------------------- Helper Functions ----------------------------- #
def first_or_none(items: List[str]) -> Optional[str]:
    return items[0] if items else None


def sources_or_fallback(primary: List[str], fallback: List[str]) -> List[str]:
    return primary if primary else fallback


# ----------------------------- Verification -------------------------------- #
async def verify_oceania(evaluator: Evaluator, parent_node, data: OceaniaShipInfo) -> None:
    oce_node = evaluator.add_parallel(
        id="Oceania_Ship",
        desc="Identify one Oceania Cruises ship (Riviera or Marina class) with complete specialty dining information",
        parent=parent_node,
        critical=False
    )

    # Ship Name check
    name_node = evaluator.add_leaf(
        id="Oceania_Ship_Name",
        desc="Ship name is either Oceania Riviera or Oceania Marina",
        parent=oce_node,
        critical=True
    )
    ship_name = data.name or ""
    await evaluator.verify(
        claim=f"The ship name '{ship_name}' is either 'Riviera' or 'Marina'. Consider case-insensitive forms or variants like 'Oceania Riviera' or 'Oceania Marina' as equivalent.",
        node=name_node,
        additional_instruction="This is a simple membership check; do not rely on web sources."
    )

    # Ship Reference (identity & class if available)
    ref_node = evaluator.add_leaf(
        id="Oceania_Ship_Reference",
        desc="Reference URL confirming the ship's identity and class",
        parent=oce_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This page is about the Oceania Cruises ship {ship_name}, and if available, it indicates the ship is in the Riviera/Marina (O-Class) category.",
        node=ref_node,
        sources=data.identity_urls,
        additional_instruction="Treat official Oceania pages, well-known cruise sites (e.g., Cruise Critic, CruiseMapper), or Wikipedia as valid. It's acceptable if the page primarily confirms the ship identity; class mention is a bonus."
    )

    # Specialty Restaurants presence
    rest_node = evaluator.add_parallel(
        id="Oceania_Specialty_Restaurants",
        desc="All four specialty restaurants are present on the identified ship",
        parent=oce_node,
        critical=True
    )
    # Individual restaurant leaves
    polo_leaf = evaluator.add_leaf(
        id="Oceania_Polo_Grill",
        desc="Polo Grill steakhouse is available on the ship",
        parent=rest_node,
        critical=True
    )
    tosc_leaf = evaluator.add_leaf(
        id="Oceania_Toscana",
        desc="Toscana Italian restaurant is available on the ship",
        parent=rest_node,
        critical=True
    )
    jacq_leaf = evaluator.add_leaf(
        id="Oceania_Jacques",
        desc="Jacques French restaurant is available on the ship",
        parent=rest_node,
        critical=True
    )
    redg_leaf = evaluator.add_leaf(
        id="Oceania_Red_Ginger",
        desc="Red Ginger Asian restaurant is available on the ship",
        parent=rest_node,
        critical=True
    )
    rest_ref_leaf = evaluator.add_leaf(
        id="Oceania_Restaurants_Reference",
        desc="Reference URL confirming the four specialty restaurants on this ship",
        parent=rest_node,
        critical=True
    )

    rest_sources = sources_or_fallback(data.restaurants_urls, data.identity_urls)
    claims_sources = [
        (
            f"The ship {ship_name} has the specialty restaurant 'Polo Grill'.",
            rest_sources,
            polo_leaf,
            "Check the ship's dining venues or deck plans. Minor naming variants like 'The Polo Grill' count as a match."
        ),
        (
            f"The ship {ship_name} has the specialty restaurant 'Toscana'.",
            rest_sources,
            tosc_leaf,
            "Check dining venues or deck plans. Accept 'Toscana' or 'Toscana Italian'."
        ),
        (
            f"The ship {ship_name} has the specialty restaurant 'Jacques'.",
            rest_sources,
            jacq_leaf,
            "Accept 'Jacques' or 'Jacques Pépin' as valid mentions of the French restaurant."
        ),
        (
            f"The ship {ship_name} has the specialty restaurant 'Red Ginger'.",
            rest_sources,
            redg_leaf,
            "Check dining venues or deck plans. Accept typical mentions of 'Red Ginger'."
        ),
        (
            f"The ship {ship_name} offers all four specialty restaurants: Polo Grill, Toscana, Jacques, and Red Ginger.",
            rest_sources,
            rest_ref_leaf,
            "The supporting page should list all four venues for this ship."
        ),
    ]
    await evaluator.batch_verify(claims_and_sources=claims_sources)

    # Dining Policy (complimentary)
    policy_node = evaluator.add_parallel(
        id="Oceania_Dining_Policy",
        desc="Specialty dining policy is correctly described",
        parent=oce_node,
        critical=True
    )
    comp_leaf = evaluator.add_leaf(
        id="Oceania_Complimentary_Policy",
        desc="All four specialty restaurants are complimentary (included in cruise fare with no additional cover charges)",
        parent=policy_node,
        critical=True
    )
    comp_ref_leaf = evaluator.add_leaf(
        id="Oceania_Policy_Reference",
        desc="Reference URL confirming the complimentary specialty dining policy",
        parent=policy_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"On Oceania {ship_name}, the specialty restaurants Polo Grill, Toscana, Jacques, and Red Ginger are complimentary with no cover charge.",
        node=comp_leaf,
        sources=data.dining_policy_urls,
        additional_instruction="Look for phrases like 'no cover charge', 'included in cruise fare', or 'complimentary specialty dining'."
    )
    await evaluator.verify(
        claim="This page confirms the specialty dining venues are complimentary (no additional fee).",
        node=comp_ref_leaf,
        sources=data.dining_policy_urls,
        additional_instruction="Synonyms or equivalent phrasing indicating no cover charge are acceptable."
    )

    # Itinerary: 2026 Mediterranean with Spanish port(s)
    itin_node = evaluator.add_parallel(
        id="Oceania_Spanish_Itinerary",
        desc="Ship offers 2026 Mediterranean cruise visiting Spanish ports",
        parent=oce_node,
        critical=True
    )
    year_leaf = evaluator.add_leaf(
        id="Oceania_2026_Cruise",
        desc="Ship has a 2026 Mediterranean cruise itinerary",
        parent=itin_node,
        critical=True
    )
    port_leaf = evaluator.add_leaf(
        id="Oceania_Spanish_Port",
        desc="The itinerary includes at least one Spanish port (e.g., Barcelona, Valencia, Cartagena, Málaga, or Cádiz)",
        parent=itin_node,
        critical=True
    )
    itin_ref_leaf = evaluator.add_leaf(
        id="Oceania_Itinerary_Reference",
        desc="Reference URL confirming the 2026 Mediterranean itinerary with Spanish port(s)",
        parent=itin_node,
        critical=True
    )
    one_spanish_port = first_or_none(data.spanish_ports) or "a Spanish port"
    await evaluator.verify(
        claim=f"This page shows a Mediterranean itinerary for {ship_name} in {TARGET_YEAR}.",
        node=year_leaf,
        sources=data.itinerary_urls,
        additional_instruction=f"Confirm the year is {TARGET_YEAR} and the itinerary is in the Mediterranean region."
    )
    await evaluator.verify(
        claim=f"The {TARGET_YEAR} Mediterranean itinerary of {ship_name} includes {one_spanish_port}, which is a Spanish port.",
        node=port_leaf,
        sources=data.itinerary_urls,
        additional_instruction="Look for port calls in Spain such as Barcelona, Valencia, Cartagena, Málaga, Cádiz/Seville, Palma de Mallorca, or Ibiza."
    )
    await evaluator.verify(
        claim=f"This page confirms that {ship_name}'s {TARGET_YEAR} Mediterranean itinerary includes at least one Spanish port.",
        node=itin_ref_leaf,
        sources=data.itinerary_urls,
        additional_instruction="Any explicit Spanish port mention in the itinerary suffices."
    )


async def verify_hal(evaluator: Evaluator, parent_node, data: HollandShipInfo) -> None:
    hal_node = evaluator.add_parallel(
        id="Holland_America_Ship",
        desc="Identify one Holland America Line ship with Tamarind restaurant and complete specialty dining information",
        parent=parent_node,
        critical=False
    )

    # Ship Name membership check
    name_leaf = evaluator.add_leaf(
        id="Holland_Ship_Name",
        desc="Ship name is one of the five Holland America ships with Tamarind: Rotterdam, Nieuw Statendam, Koningsdam, Eurodam, or Nieuw Amsterdam",
        parent=hal_node,
        critical=True
    )
    hal_name = data.name or ""
    await evaluator.verify(
        claim=(
            f"The ship name '{hal_name}' is one of the following: Rotterdam, Nieuw Statendam, Koningsdam, "
            f"Eurodam, or Nieuw Amsterdam."
        ),
        node=name_leaf,
        additional_instruction="This is a simple membership check; case-insensitive matching is acceptable."
    )

    # Ship Reference (identity + Tamarind availability)
    ref_leaf = evaluator.add_leaf(
        id="Holland_Ship_Reference",
        desc="Reference URL confirming the ship's identity and Tamarind availability",
        parent=hal_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This page is about the Holland America Line ship {hal_name} and confirms that Tamarind is available on this ship.",
        node=ref_leaf,
        sources=sources_or_fallback(data.restaurants_urls, data.identity_urls),
        additional_instruction="Look for the Tamarind restaurant on the ship's dining pages or deck plans."
    )

    # Specialty Restaurants presence
    rest_node = evaluator.add_parallel(
        id="Holland_Specialty_Restaurants",
        desc="Three core specialty restaurants are present on the identified ship",
        parent=hal_node,
        critical=True
    )
    tam_leaf = evaluator.add_leaf(
        id="Holland_Tamarind",
        desc="Tamarind Pan-Asian restaurant is available on the ship",
        parent=rest_node,
        critical=True
    )
    pin_leaf = evaluator.add_leaf(
        id="Holland_Pinnacle_Grill",
        desc="Pinnacle Grill steakhouse is available on the ship",
        parent=rest_node,
        critical=True
    )
    can_leaf = evaluator.add_leaf(
        id="Holland_Canaletto",
        desc="Canaletto Italian restaurant is available on the ship",
        parent=rest_node,
        critical=True
    )
    rest_ref_leaf = evaluator.add_leaf(
        id="Holland_Restaurants_Reference",
        desc="Reference URL confirming the three specialty restaurants on this ship",
        parent=rest_node,
        critical=True
    )
    hal_rest_sources = sources_or_fallback(data.restaurants_urls, data.identity_urls)
    hal_rest_claims = [
        (
            f"The ship {hal_name} has the specialty restaurant 'Tamarind'.",
            hal_rest_sources,
            tam_leaf,
            "Confirm Tamarind appears in dining venues or deck plans."
        ),
        (
            f"The ship {hal_name} has the specialty restaurant 'Pinnacle Grill'.",
            hal_rest_sources,
            pin_leaf,
            "Confirm Pinnacle Grill is listed for this ship."
        ),
        (
            f"The ship {hal_name} has the specialty restaurant 'Canaletto'.",
            hal_rest_sources,
            can_leaf,
            "Confirm Canaletto is listed for this ship."
        ),
        (
            f"The ship {hal_name} offers Tamarind, Pinnacle Grill, and Canaletto specialty restaurants.",
            hal_rest_sources,
            rest_ref_leaf,
            "The supporting page should list all three venues for this ship."
        ),
    ]
    await evaluator.batch_verify(claims_and_sources=hal_rest_claims)

    # Dining Pricing
    pricing_node = evaluator.add_parallel(
        id="Holland_Dining_Pricing",
        desc="Specialty restaurant pricing is correctly provided",
        parent=hal_node,
        critical=True
    )
    tam_price_leaf = evaluator.add_leaf(
        id="Holland_Tamarind_Price",
        desc="Tamarind cover charge is $35 per person",
        parent=pricing_node,
        critical=True
    )
    pin_price_leaf = evaluator.add_leaf(
        id="Holland_Pinnacle_Price",
        desc="Pinnacle Grill cover charge is $52 per person",
        parent=pricing_node,
        critical=True
    )
    can_price_leaf = evaluator.add_leaf(
        id="Holland_Canaletto_Price",
        desc="Canaletto cover charge is $25 per person",
        parent=pricing_node,
        critical=True
    )
    price_ref_leaf = evaluator.add_leaf(
        id="Holland_Pricing_Reference",
        desc="Reference URL confirming the specialty restaurant pricing",
        parent=pricing_node,
        critical=True
    )
    price_sources = data.pricing_urls
    price_claims = [
        (
            "The Tamarind cover charge is $35 per person.",
            price_sources,
            tam_price_leaf,
            "Accept equivalent phrasing such as '$35 pp', 'USD 35 per person', or 'approximately $35'."
        ),
        (
            "The Pinnacle Grill cover charge is $52 per person.",
            price_sources,
            pin_price_leaf,
            "Accept equivalent phrasing such as '$52 pp', 'USD 52 per person', or 'starts at $52'."
        ),
        (
            "The Canaletto cover charge is $25 per person.",
            price_sources,
            can_price_leaf,
            "Accept equivalent phrasing such as '$25 pp' or 'USD 25 per person'."
        ),
        (
            f"This page provides the specialty restaurant pricing for {hal_name}, including fees for Tamarind, Pinnacle Grill, and Canaletto.",
            price_sources,
            price_ref_leaf,
            "A consolidated or multiple sources page listing these prices is acceptable."
        ),
    ]
    await evaluator.batch_verify(claims_and_sources=price_claims)

    # Itinerary: 2026 Mediterranean with Spanish port(s)
    itin_node = evaluator.add_parallel(
        id="Holland_Spanish_Itinerary",
        desc="Ship offers 2026 Mediterranean cruise visiting Spanish ports",
        parent=hal_node,
        critical=True
    )
    year_leaf = evaluator.add_leaf(
        id="Holland_2026_Cruise",
        desc="Ship has a 2026 Mediterranean cruise itinerary",
        parent=itin_node,
        critical=True
    )
    port_leaf = evaluator.add_leaf(
        id="Holland_Spanish_Port",
        desc="The itinerary includes at least one Spanish port (e.g., Barcelona, Valencia, Cartagena, Málaga, or Cádiz)",
        parent=itin_node,
        critical=True
    )
    itin_ref_leaf = evaluator.add_leaf(
        id="Holland_Itinerary_Reference",
        desc="Reference URL confirming the 2026 Mediterranean itinerary with Spanish port(s)",
        parent=itin_node,
        critical=True
    )
    hal_spanish_port = first_or_none(data.spanish_ports) or "a Spanish port"
    await evaluator.verify(
        claim=f"This page shows a Mediterranean itinerary for {hal_name} in {TARGET_YEAR}.",
        node=year_leaf,
        sources=data.itinerary_urls,
        additional_instruction=f"Confirm year {TARGET_YEAR} and Mediterranean region."
    )
    await evaluator.verify(
        claim=f"The {TARGET_YEAR} Mediterranean itinerary of {hal_name} includes {hal_spanish_port}, which is a Spanish port.",
        node=port_leaf,
        sources=data.itinerary_urls,
        additional_instruction="Look for Spanish port calls such as Barcelona, Valencia, Cartagena, Málaga, Cádiz/Seville, Palma de Mallorca, or Ibiza."
    )
    await evaluator.verify(
        claim=f"This page confirms that {hal_name}'s {TARGET_YEAR} Mediterranean itinerary includes at least one Spanish port.",
        node=itin_ref_leaf,
        sources=data.itinerary_urls,
        additional_instruction="Any explicit Spanish port mention in the itinerary suffices."
    )


# ---------------------------- Main Entry Point ----------------------------- #
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

    # Extract structured data
    oceania_data = await evaluator.extract(
        prompt=prompt_extract_oceania_ship(),
        template_class=OceaniaShipInfo,
        extraction_name="oceania_ship_info"
    )
    hal_data = await evaluator.extract(
        prompt=prompt_extract_hal_ship(),
        template_class=HollandShipInfo,
        extraction_name="holland_ship_info"
    )

    # Add helpful custom info
    evaluator.add_custom_info(
        info={"allowed_oceania_ships": OCEANIA_ALLOWED_SHIPS, "allowed_hal_ships": HAL_ALLOWED_SHIPS,
              "spanish_port_examples": SPANISH_PORT_EXAMPLES, "target_year": TARGET_YEAR},
        info_type="constraints",
        info_name="evaluation_constraints"
    )

    # Build verification tree
    await verify_oceania(evaluator, root, oceania_data)
    await verify_hal(evaluator, root, hal_data)

    return evaluator.get_summary()