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
TASK_ID = "la_state_park_eval_2024"
TASK_DESCRIPTION = (
    "Identify a Louisiana State Park that meets all of the following requirements as of 2024: "
    "(1) Offers RV campsites with full hookups (water, electric, and sewer), "
    "(2) Has a swimming beach that is open year-round, "
    "(3) Has a water playground facility, "
    "(4) The water playground operates Tuesday through Sunday from 8 a.m. to 8 p.m. (closed Mondays for cleaning), "
    "(5) Offers deluxe cabin accommodations that sleep up to 8 people, "
    "(6) The deluxe cabins cost between $175 and $262.50 per night plus tax, "
    "(7) Has a campground with bathhouse facilities that include showers, "
    "(8) The bathhouse facilities include laundry, "
    "(9) Has boat launch facilities, "
    "(10) Has premium campsites with water and electrical hookups, "
    "(11) The premium campsites cost between $33 and $49.50 per night plus tax, "
    "(12) Swimming at the beach is available without lifeguard supervision. "
    "Provide the name of the park and a reference URL supporting your answer."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ParkAnswerExtraction(BaseModel):
    park_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_park_info() -> str:
    return """
    Extract the identified park name and all reference URLs cited in the answer.

    Required fields:
    - park_name: The name of the Louisiana State Park the answer claims meets all requirements. Extract exactly as written.
    - reference_urls: An array of all URLs referenced in the answer that purportedly support the claims. Include any links provided (plain URLs or markdown links).
    
    Rules:
    - Only include URLs explicitly present in the answer.
    - If no URLs are present, return an empty array.
    - If the park name is not clearly stated, return null for park_name.
    """


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_park_nodes(
    evaluator: Evaluator,
    parent_node,
    extracted: ParkAnswerExtraction,
) -> None:
    """
    Build the verification tree under a single critical parallel node and run verifications.
    """
    top = evaluator.add_parallel(
        id="Louisiana_State_Park_Identification",
        desc="Identifies a Louisiana State Park that satisfies all specified facility and amenity requirements and provides required citation.",
        parent=parent_node,
        critical=True,
    )

    # Existence checks (critical)
    has_name = bool(extracted.park_name and extracted.park_name.strip())
    evaluator.add_custom_node(
        result=has_name,
        id="Response_Includes_Park_Name",
        desc="The response provides the name of the park.",
        parent=top,
        critical=True,
    )

    has_urls = bool(extracted.reference_urls and len(extracted.reference_urls) > 0)
    evaluator.add_custom_node(
        result=has_urls,
        id="Response_Includes_Reference_URL",
        desc="The response provides at least one reference URL supporting the answer.",
        parent=top,
        critical=True,
    )

    park_name = extracted.park_name or ""
    urls = extracted.reference_urls if extracted.reference_urls else []

    # Prepare leaf nodes + claims
    claims_and_sources: List[tuple[str, List[str], Any, Optional[str]]] = []

    # Helper for leaf creation
    def add_leaf_with_claim(node_id: str, desc: str, claim: str, add_ins: str) -> None:
        node = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=top,
            critical=True,
        )
        claims_and_sources.append((claim, urls, node, add_ins))

    # 1. Park belongs to Louisiana State Parks system
    add_leaf_with_claim(
        "Park_Is_Louisiana_State_Park",
        "The identified park is a Louisiana State Park (i.e., part of the Louisiana State Parks system).",
        f"The park '{park_name}' is part of the Louisiana State Parks system (a Louisiana State Park).",
        "Confirm that the referenced page(s) explicitly identify the park as a Louisiana State Park (operated by the Louisiana Office of State Parks). Accept reasonable equivalents such as official LA State Parks site pages.",
    )

    # 2. RV full hookups
    add_leaf_with_claim(
        "Full_Hookup_RV_Sites",
        "The park offers RV campsites with full hookups (water, electric, and sewer).",
        f"The park '{park_name}' offers RV campsites with full hookups, including water, electricity, and sewer.",
        "Look for exact mention of 'full hookups' or explicit listing of water, electric, and sewer. Accept equivalent phrasing like 'full hook-ups', 'full-service RV sites'.",
    )

    # 3. Year-round swimming beach
    add_leaf_with_claim(
        "Year_Round_Swimming_Beach",
        "The park has a swimming beach that is open year-round.",
        f"The park '{park_name}' has a swimming beach that is open year-round.",
        "Support may include phrases like 'open year-round', 'open all year', or similar. If a seasonal closure is stated, then this claim should be judged not supported.",
    )

    # 4. Water playground facility
    add_leaf_with_claim(
        "Water_Playground_Facility",
        "The park has a water playground facility.",
        f"The park '{park_name}' has a water playground facility (e.g., splash pad or water playground).",
        "Accept synonyms such as 'splash pad' or 'water playground'. The page should clearly indicate the facility exists at this park.",
    )

    # 5. Water playground schedule (Tue–Sun 8am–8pm; closed Mon)
    add_leaf_with_claim(
        "Water_Playground_Schedule",
        "The water playground operates Tuesday through Sunday from 8 a.m. to 8 p.m. and is closed Mondays (for cleaning).",
        f"At {park_name}, the water playground operates Tuesday through Sunday from 8 a.m. to 8 p.m. and is closed on Mondays for cleaning.",
        "Verify the specific hours and days. Accept reasonable time formatting variants (e.g., '8:00 am–8:00 pm', '8 AM to 8 PM'). The key is Tue–Sun open and Monday closed for cleaning.",
    )

    # 6. Deluxe cabins offered
    add_leaf_with_claim(
        "Deluxe_Cabin_Availability",
        "The park offers deluxe cabin accommodations.",
        f"The park '{park_name}' offers deluxe cabin accommodations.",
        "Look for 'deluxe cabin(s)' or equivalent cabin type labeling on official pages.",
    )

    # 7. Deluxe cabins sleep up to 8
    add_leaf_with_claim(
        "Deluxe_Cabin_Capacity",
        "The deluxe cabins sleep up to 8 people.",
        f"The deluxe cabins at '{park_name}' sleep up to 8 people.",
        "The page should specify occupancy (e.g., 'sleeps 8'). Minor phrasing variants are acceptable.",
    )

    # 8. Deluxe cabin price range $175–$262.50 plus tax
    add_leaf_with_claim(
        "Deluxe_Cabin_Price_Range",
        "The deluxe cabins cost between $175 and $262.50 per night plus tax.",
        f"The deluxe cabins at '{park_name}' cost between $175 and $262.50 per night plus tax.",
        "Focus on the numeric nightly rate range. Seasonal or weekend rate variations are fine as long as rates fall within $175–$262.50. Treat 'plus tax' as standard; explicit mention is helpful but not strictly required if rates are shown before tax.",
    )

    # 9. Bathhouse includes showers
    add_leaf_with_claim(
        "Bathhouse_Shower_Facilities",
        "The park's campground bathhouse includes shower facilities.",
        f"The campground bathhouse at '{park_name}' includes shower facilities.",
        "Accept mentions such as 'bathhouse with showers' or 'shower facilities available' tied to the campground.",
    )

    # 10. Bathhouse includes laundry
    add_leaf_with_claim(
        "Bathhouse_Laundry_Facilities",
        "The park's campground bathhouse includes laundry facilities.",
        f"The campground bathhouse at '{park_name}' includes laundry facilities.",
        "Look for 'laundry', 'washers/dryers', or similar wording indicating laundry facilities in or at the bathhouse.",
    )

    # 11. Boat launch facilities
    add_leaf_with_claim(
        "Boat_Launch_Facilities",
        "The park has boat launch facilities.",
        f"The park '{park_name}' has boat launch facilities.",
        "Accept synonyms like 'boat launch', 'launch ramp', 'boat ramp' provided they are within the park.",
    )

    # 12. Premium campsites with water & electric
    add_leaf_with_claim(
        "Premium_Campsite_Hookups",
        "The park has premium campsites with water and electrical hookups.",
        f"The park '{park_name}' offers premium campsites with water and electrical hookups.",
        "Louisiana State Parks often label 'Premium Campsites' as having water and electricity. Confirm this on the referenced page.",
    )

    # 13. Premium campsite price range $33–$49.50 plus tax
    add_leaf_with_claim(
        "Premium_Campsite_Price_Range",
        "The premium campsites cost between $33 and $49.50 per night plus tax.",
        f"The premium campsites at '{park_name}' cost between $33 and $49.50 per night plus tax.",
        "Focus on the nightly rate range for Premium Campsites. Seasonal/weekend variations are acceptable if they remain within $33–$49.50. Treat 'plus tax' as standard; explicit mention helps but is not strictly required if rates are shown pre-tax.",
    )

    # 14. No lifeguard supervision
    add_leaf_with_claim(
        "No_Lifeguard_Supervision",
        "Swimming at the beach is available without lifeguard supervision.",
        f"Swimming at the beach at '{park_name}' is available without lifeguard supervision (e.g., 'no lifeguard on duty').",
        "Accept 'no lifeguard on duty' or 'swim at your own risk' as evidence that swimming occurs without lifeguard supervision.",
    )

    # Execute all verifications in parallel to avoid unnecessary cross-sibling gating
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the Louisiana State Park identification task.
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

    # 1) Extract park name and reference URLs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_park_info(),
        template_class=ParkAnswerExtraction,
        extraction_name="park_answer_extraction",
    )

    # 2) Add custom info to summary for transparency
    evaluator.add_custom_info(
        {
            "park_name": extracted.park_name,
            "reference_urls": extracted.reference_urls,
            "total_urls": len(extracted.reference_urls),
        },
        info_type="extraction_summary",
    )

    # 3) Build verification tree and run checks
    await build_and_verify_park_nodes(evaluator, root, extracted)

    # 4) Return structured evaluation summary
    return evaluator.get_summary()