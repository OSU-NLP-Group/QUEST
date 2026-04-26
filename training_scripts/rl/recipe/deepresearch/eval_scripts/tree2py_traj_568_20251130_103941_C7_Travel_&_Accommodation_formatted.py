import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "universal_orlando_partner_hotel_eval"
TASK_DESCRIPTION = """Identify the name of the hotel near Universal Orlando Resort that meets ALL of the following criteria:

1. Located within 2.5 miles of Universal Orlando Resort
2. Offers free hot breakfast served daily from 6:00-9:30 AM on weekdays (Monday-Friday) and 7:00-10:00 AM on weekends (Saturday-Sunday)
3. Provides free scheduled shuttle service to Universal Orlando theme parks
4. Offers free parking during guest stays
5. Provides free Wi-Fi throughout the property
6. Has standard check-in time at 3:00 PM
7. Has standard check-out time at 11:00 AM
8. Features an outdoor pool and whirlpool
9. Has a 24-hour fitness center
10. Offers complimentary evening snacks and beverages (such as a 5:30 Kickback program or similar)
11. Is classified as a Universal Partner Hotel (not an on-site Universal Orlando hotel)
12. Does NOT include Universal Express Pass benefits as a complimentary amenity for guests
13. Accepts pets (dogs and cats) with a daily fee of $50 per room plus tax
14. Has a pet policy limiting guests to a maximum of two pets with a combined weight limit of 80 pounds
15. Includes a microwave and refrigerator as standard in-room amenities in all guest rooms
16. Has a 24-hour business center

Provide the full name of the hotel.
"""


class HotelExtraction(BaseModel):
    hotel_name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


def prompt_extract_hotel() -> str:
    return """
    Extract the single final hotel's full name identified in the answer and all URLs cited in the answer.
    Fields:
    - hotel_name: The full, specific hotel name the answer ultimately identifies as meeting the criteria. If multiple are mentioned, choose the one presented as the final answer. If none, return null.
    - urls: An array of all valid URLs mentioned anywhere in the answer (including markdown links). Include official hotel pages or Universal Partner listings if present. Do not invent URLs. If no URLs, return an empty array.
    """


async def build_verification_tree(
    evaluator: Evaluator,
    parent_root,
    hotel: HotelExtraction
) -> None:
    hotel_ident_node = evaluator.add_sequential(
        id="Hotel_Identification",
        desc="Identify the hotel near Universal Orlando Resort that satisfies all specified criteria and provide its full name",
        parent=parent_root,
        critical=True
    )

    name_provided = bool(hotel.hotel_name and hotel.hotel_name.strip())
    evaluator.add_custom_node(
        result=name_provided,
        id="Provide_Hotel_Full_Name",
        desc="Answer provides the full name of a single specific hotel (i.e., identifies the hotel to be evaluated)",
        parent=hotel_ident_node,
        critical=True
    )

    criteria_node = evaluator.add_parallel(
        id="Meets_All_Criteria",
        desc="The identified hotel satisfies all listed constraints",
        parent=hotel_ident_node,
        critical=True
    )

    hotel_name = hotel.hotel_name or "the hotel"
    sources = hotel.urls if hotel.urls else None

    leaves_and_claims: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    def add_leaf_and_prepare(id_: str, desc_: str, claim_: str, add_ins: str) -> None:
        leaf = evaluator.add_leaf(
            id=id_,
            desc=desc_,
            parent=criteria_node,
            critical=True
        )
        leaves_and_claims.append((claim_, sources, leaf, add_ins))

    add_leaf_and_prepare(
        "Distance_From_Universal",
        "The hotel must be located within 2.5 miles of Universal Orlando Resort",
        f"{hotel_name} is located within 2.5 miles of Universal Orlando Resort.",
        "Verify if any provided page explicitly states the property's distance to Universal Orlando Resort is ≤ 2.5 miles. Accept variants like 'mi' and references to Universal Orlando/CityWalk. If the pages lack distance info, consider the claim not supported."
    )

    add_leaf_and_prepare(
        "Free_Breakfast_Offered",
        "The hotel must offer free hot breakfast served daily",
        f"{hotel_name} offers free hot breakfast daily.",
        "Look for 'free hot breakfast' or 'complimentary hot breakfast' mentioned on the hotel's official page or listing. Ensure it is complimentary for guests."
    )

    add_leaf_and_prepare(
        "Breakfast_Service_Hours",
        "Breakfast must be served from 6:00-9:30 AM on weekdays (Monday-Friday) and 7:00-10:00 AM on weekends (Saturday-Sunday)",
        f"Breakfast at {hotel_name} is served from 6:00–9:30 AM Monday–Friday and 7:00–10:00 AM Saturday–Sunday.",
        "Confirm the exact breakfast hours as stated. Minor punctuation variations are fine, but times and weekday/weekend mapping must match."
    )

    add_leaf_and_prepare(
        "Free_Shuttle_Service",
        "The hotel must provide free scheduled shuttle service to Universal Orlando theme parks",
        f"{hotel_name} provides free scheduled shuttle service to Universal Orlando theme parks.",
        "Find explicit mention of a complimentary, scheduled shuttle to Universal Orlando theme parks."
    )

    add_leaf_and_prepare(
        "Free_Parking",
        "The hotel must offer free parking during guest stays",
        f"{hotel_name} offers free parking during guest stays.",
        "Verify that guest parking is complimentary (no nightly fee)."
    )

    add_leaf_and_prepare(
        "Free_WiFi",
        "The hotel must provide free Wi-Fi throughout the property",
        f"{hotel_name} provides free Wi‑Fi throughout the property.",
        "Confirm that Wi‑Fi is complimentary and available property‑wide (rooms and common areas)."
    )

    add_leaf_and_prepare(
        "Check_In_Time",
        "The hotel must have standard check-in time at 3:00 PM",
        f"The standard check‑in time at {hotel_name} is 3:00 PM.",
        "Check the official policy page or FAQs for check‑in time exactly at 3:00 PM."
    )

    add_leaf_and_prepare(
        "Check_Out_Time",
        "The hotel must have standard check-out time at 11:00 AM",
        f"The standard check‑out time at {hotel_name} is 11:00 AM.",
        "Check the official policy page or FAQs for check‑out time exactly at 11:00 AM."
    )

    add_leaf_and_prepare(
        "Pool_And_Whirlpool",
        "The hotel must feature an outdoor pool and whirlpool",
        f"{hotel_name} features an outdoor pool and a whirlpool (hot tub).",
        "Look for amenities listing 'outdoor pool' and 'whirlpool' (synonyms: hot tub, spa tub). Both must be present."
    )

    add_leaf_and_prepare(
        "24Hour_Fitness_Center",
        "The hotel must have a 24-hour fitness center",
        f"{hotel_name} has a 24‑hour fitness center.",
        "Amenity description should indicate the fitness center is open 24 hours."
    )

    add_leaf_and_prepare(
        "Evening_Snacks_Beverages",
        "The hotel must offer complimentary evening snacks and beverages (such as a 5:30 Kickback program or similar)",
        f"{hotel_name} offers complimentary evening snacks and beverages such as a '5:30 Kickback' program.",
        "Confirm complimentary evening snacks and beverages. Names like '5:30 Kickback' or equivalent are acceptable."
    )

    add_leaf_and_prepare(
        "Universal_Partner_Status",
        "The hotel must be classified as a Universal Partner Hotel (not an on-site Universal Orlando hotel)",
        f"{hotel_name} is a Universal Partner Hotel and is not an on‑site Universal Orlando hotel.",
        "Check Universal Orlando's official partner hotel listing or hotel page. The property should be designated 'Universal Partner Hotel' and not one of Universal's on‑site categories."
    )

    add_leaf_and_prepare(
        "No_Express_Pass_Benefit",
        "The hotel must NOT include Universal Express Pass benefits as a complimentary amenity for guests",
        f"{hotel_name} does not include Universal Express Pass benefits as a complimentary amenity for guests.",
        "Confirm that Universal Express Pass is not complimentary for guests. If pages explicitly say it's not included or is sold separately, the claim is supported."
    )

    add_leaf_and_prepare(
        "Pet_Policy_Fee",
        "The hotel must accept pets (dogs and cats) with a daily fee of $50 per room plus tax",
        f"{hotel_name} accepts dogs and cats and charges a daily fee of $50 per room plus tax.",
        "Verify pet acceptance (dogs and cats) and the specific daily fee of $50 per room plus tax."
    )

    add_leaf_and_prepare(
        "Pet_Weight_Limit",
        "The hotel's pet policy must limit guests to a maximum of two pets with a combined weight limit of 80 pounds",
        f"The pet policy at {hotel_name} limits guests to a maximum of two pets with a combined weight limit of 80 pounds.",
        "Confirm both: max two pets and combined weight limit of 80 lbs."
    )

    add_leaf_and_prepare(
        "In_Room_Amenities",
        "Guest rooms must include a microwave and refrigerator as standard in-room amenities",
        f"All guest rooms at {hotel_name} include a microwave and a refrigerator as standard amenities.",
        "Amenity descriptions should indicate microwave and refrigerator are standard in all rooms."
    )

    add_leaf_and_prepare(
        "24Hour_Business_Center",
        "The hotel must have a 24-hour business center",
        f"{hotel_name} has a 24‑hour business center.",
        "Confirm a business center that is available 24 hours."
    )

    await evaluator.batch_verify(leaves_and_claims)


async def evaluate_answer(
    client: LLMClient,
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
        strategy=AggregationStrategy.SEQUENTIAL,
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

    hotel_info = await evaluator.extract(
        prompt=prompt_extract_hotel(),
        template_class=HotelExtraction,
        extraction_name="hotel_extraction"
    )

    evaluator.add_custom_info(
        info={
            "extracted_hotel_name": hotel_info.hotel_name,
            "extracted_urls_count": len(hotel_info.urls),
            "extracted_urls": hotel_info.urls
        },
        info_type="extraction_summary",
        info_name="hotel_extraction_summary"
    )

    await build_verification_tree(evaluator, root, hotel_info)

    return evaluator.get_summary()