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
TASK_ID = "sd_parks_day_use_dec2025"
TASK_DESCRIPTION = (
    "A family is planning a Saturday day trip to San Diego County on December 7, 2025, "
    "and wants to compare three parks: Wilderness Gardens County Preserve, Anza-Borrego Desert State Park, "
    "and Torrey Pines State Natural Reserve. For each of these three parks, provide: "
    "(1) whether the park is open for day-use visitors on Saturdays in early December 2025 and what the operating hours are, "
    "(2) the day-use parking or entry fee per vehicle, and (3) whether dogs are allowed at the park."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ParkInfo(BaseModel):
    # Core fields requested in the task
    saturday_open_status: Optional[str] = None  # e.g., "open", "closed", "partially open"
    operating_hours: Optional[str] = None       # e.g., "Vehicle access 8 AM–4 PM Thu–Tue; pedestrian sunrise–sunset"
    fee_per_vehicle: Optional[str] = None       # e.g., "$5", "$10", "$12–$25 demand-based"
    dog_policy: Optional[str] = None            # e.g., "Dogs allowed on leash <6 feet", "Dogs not allowed"

    # Field-specific sources cited in the answer
    sources_open_hours: List[str] = Field(default_factory=list)  # URLs that support open status / hours
    sources_fee: List[str] = Field(default_factory=list)         # URLs that support fee information
    sources_dogs: List[str] = Field(default_factory=list)        # URLs that support dog policy

    # Torrey Pines specific constraint note (optional for other parks)
    trails_closure_notice: Optional[str] = None                  # e.g., "All trails closed Nov 2025 – Feb 2026"
    sources_trails: List[str] = Field(default_factory=list)      # URLs that support trail closure notice


class ParksExtraction(BaseModel):
    wilderness_gardens: Optional[ParkInfo] = None
    anza_borrego: Optional[ParkInfo] = None
    torrey_pines: Optional[ParkInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_parks() -> str:
    return """
    Extract structured information for the three parks mentioned in the task: Wilderness Gardens County Preserve,
    Anza-Borrego Desert State Park, and Torrey Pines State Natural Reserve.

    For each park, extract the following fields exactly as stated in the answer:

    1) saturday_open_status: Whether the park is open or closed for day-use on Saturdays in early December 2025.
       Use a short phrase like "open", "closed", or "partially open". If not explicitly stated, return null.

    2) operating_hours: The operating hours applicable to day-use visitors (succinct textual summary).
       Include schedule details such as gates/vehicle access times and any daily/weekday specifics relevant to Saturdays.
       If not provided, return null.

    3) fee_per_vehicle: The per-vehicle day-use parking/entry fee (as a string, e.g., "$5", "$10", "$10–$25").
       If not provided, return null.

    4) dog_policy: Whether dogs are allowed (and any constraints like leash length).
       Use a concise statement like "Dogs allowed on leash <6 feet" or "Dogs not allowed".
       If not provided, return null.

    5) sources_open_hours: All URLs cited in the answer that support open status and operating hours for the park.
       Extract actual URLs only. If none are cited, return an empty list.

    6) sources_fee: All URLs cited in the answer that support the day-use per-vehicle fee for the park.
       Extract actual URLs only. If none are cited, return an empty list.

    7) sources_dogs: All URLs cited in the answer that support the dog policy for the park.
       Extract actual URLs only. If none are cited, return an empty list.

    Additionally for Torrey Pines State Natural Reserve:
    8) trails_closure_notice: If the answer mentions any trail closure notices (e.g., "All trails closed Nov 2025 – Feb 2026"),
       extract that exact statement; else return null.
    9) sources_trails: All URLs cited that support the trail closure notice; else return an empty list.

    Return a JSON object with keys: wilderness_gardens, anza_borrego, torrey_pines.
    Each value should be an object with the fields above. If a park is not mentioned, return null for that park.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _ensure_sources(*lists: List[str]) -> List[str]:
    """Combine multiple source lists, remove duplicates, keep order."""
    seen = set()
    combined: List[str] = []
    for lst in lists:
        for url in lst:
            if url and url not in seen:
                seen.add(url)
                combined.append(url)
    return combined


def _presence_claim(field_value: Optional[str], park_name: str, field_label: str) -> str:
    """Build a presence-assertion claim for simple verification when the answer did not provide the field."""
    # Example: "The answer explicitly provides the operating hours for Wilderness Gardens County Preserve."
    return (
        f"The answer explicitly provides the {field_label} for {park_name}."
        if not field_value else f"The answer states the {field_label} for {park_name} as: {field_value}."
    )


# --------------------------------------------------------------------------- #
# Verification functions per park                                             #
# --------------------------------------------------------------------------- #
async def verify_wilderness_gardens(evaluator: Evaluator, parent_node, info: Optional[ParkInfo]) -> None:
    park_name = "Wilderness Gardens County Preserve"
    park_node = evaluator.add_parallel(
        id="wilderness_gardens",
        desc="Wilderness Gardens County Preserve information provided",
        parent=parent_node,
        critical=False
    )

    # Leaf: Saturday open status (explicitly stated in answer)
    open_node = evaluator.add_leaf(
        id="wilderness_gardens_saturday_open_status",
        desc="States whether Wilderness Gardens is open for day-use on Saturdays in early Dec 2025 (open/closed determination for Saturday explicitly stated)",
        parent=park_node,
        critical=True
    )
    open_claim = _presence_claim(
        info.saturday_open_status if info else None,
        park_name,
        "Saturday day-use open/closed status in early December 2025"
    )
    await evaluator.verify(
        claim=open_claim,
        node=open_node,
        sources=None,
        additional_instruction="Verify presence in the answer: it should clearly say whether Saturday is open or closed for day-use in early December 2025."
    )

    # Leaf: Operating hours (verify against sources; if missing, check presence)
    hours_node = evaluator.add_leaf(
        id="wilderness_gardens_operating_hours",
        desc="Provides Wilderness Gardens operating hours consistent with constraints (vehicle access 8 AM–4 PM Thu–Tue; pedestrian access sunrise–sunset daily; and closed Wednesdays for vehicle access implied by Thu–Tue schedule)",
        parent=park_node,
        critical=True
    )
    hours_claim = (
        f"The operating hours at {park_name} are: {info.operating_hours}."
        if info and info.operating_hours else
        f"The answer explicitly provides the operating hours for {park_name}."
    )
    hours_sources = _ensure_sources(info.sources_open_hours if info else [])
    await evaluator.verify(
        claim=hours_claim,
        node=hours_node,
        sources=hours_sources if hours_sources else None,
        additional_instruction=(
            "If hours are provided, verify the schedule against the cited sources. "
            "For consistency, expect statements like: vehicle access 8 AM–4 PM Thursday–Tuesday; "
            "pedestrian access sunrise–sunset daily; vehicle access closed Wednesdays."
        )
    )

    # Leaf: Fee ($5 per vehicle) — verify agent-stated fee against sources (presence fallback)
    fee_node = evaluator.add_leaf(
        id="wilderness_gardens_fee",
        desc="States Wilderness Gardens day-use parking fee per vehicle as $5",
        parent=park_node,
        critical=True
    )
    fee_claim = (
        f"The day-use parking fee per vehicle at {park_name} is {info.fee_per_vehicle}."
        if info and info.fee_per_vehicle else
        f"The answer explicitly provides the day-use per-vehicle parking fee for {park_name}."
    )
    fee_sources = _ensure_sources(info.sources_fee if info else [])
    await evaluator.verify(
        claim=fee_claim,
        node=fee_node,
        sources=fee_sources if fee_sources else None,
        additional_instruction="Check whether the cited source(s) show a $5 per-vehicle day-use fee (allow equivalent phrasing like '$5 per vehicle')."
    )

    # Leaf: Dog policy — verify against sources (presence fallback)
    dogs_node = evaluator.add_leaf(
        id="wilderness_gardens_dogs",
        desc="Explicitly states whether dogs are allowed at Wilderness Gardens (yes/no statement present)",
        parent=park_node,
        critical=True
    )
    dogs_claim = (
        f"The dog policy at {park_name} is: {info.dog_policy}."
        if info and info.dog_policy else
        f"The answer explicitly states whether dogs are allowed at {park_name}."
    )
    dogs_sources = _ensure_sources(info.sources_dogs if info else [])
    await evaluator.verify(
        claim=dogs_claim,
        node=dogs_node,
        sources=dogs_sources if dogs_sources else None,
        additional_instruction="Verify whether dogs are allowed or not, and any leash/length requirements if applicable."
    )


async def verify_anza_borrego(evaluator: Evaluator, parent_node, info: Optional[ParkInfo]) -> None:
    park_name = "Anza-Borrego Desert State Park"
    park_node = evaluator.add_parallel(
        id="anza_borrego",
        desc="Anza-Borrego Desert State Park information provided",
        parent=parent_node,
        critical=False
    )

    # Leaf: Saturday open status (explicitly stated)
    open_node = evaluator.add_leaf(
        id="anza_borrego_saturday_open_status",
        desc="States whether Anza-Borrego is open for day-use on Saturdays in early Dec 2025",
        parent=park_node,
        critical=True
    )
    open_claim = _presence_claim(
        info.saturday_open_status if info else None,
        park_name,
        "Saturday day-use open/closed status in early December 2025"
    )
    await evaluator.verify(
        claim=open_claim,
        node=open_node,
        sources=None,
        additional_instruction="Verify presence in the answer: it should clearly say whether Saturday is open or closed for day-use in early December 2025."
    )

    # Leaf: Operating hours (Visitor Center 9–5 daily Oct–May; parking lot 7–7)
    hours_node = evaluator.add_leaf(
        id="anza_borrego_operating_hours",
        desc="Provides operating hours information consistent with constraints (Visitor Center 9 AM–5 PM daily Oct 1–May 31; Visitor Center parking lot 7 AM–7 PM)",
        parent=park_node,
        critical=True
    )
    hours_claim = (
        f"The operating hours at {park_name} are: {info.operating_hours}."
        if info and info.operating_hours else
        f"The answer explicitly provides the operating hours for {park_name}."
    )
    hours_sources = _ensure_sources(info.sources_open_hours if info else [])
    await evaluator.verify(
        claim=hours_claim,
        node=hours_node,
        sources=hours_sources if hours_sources else None,
        additional_instruction=(
            "If hours are provided, verify against sources. Expect details such as: "
            "Visitor Center open 9 AM–5 PM daily from October 1 to May 31; "
            "Visitor Center parking lot open 7 AM–7 PM."
        )
    )

    # Leaf: Fee ($10 per vehicle)
    fee_node = evaluator.add_leaf(
        id="anza_borrego_fee",
        desc="States Anza-Borrego day-use entry fee per vehicle as $10",
        parent=park_node,
        critical=True
    )
    fee_claim = (
        f"The day-use entry fee per vehicle at {park_name} is {info.fee_per_vehicle}."
        if info and info.fee_per_vehicle else
        f"The answer explicitly provides the day-use per-vehicle entry fee for {park_name}."
    )
    fee_sources = _ensure_sources(info.sources_fee if info else [])
    await evaluator.verify(
        claim=fee_claim,
        node=fee_node,
        sources=fee_sources if fee_sources else None,
        additional_instruction="Verify whether the cited source(s) show a $10 per-vehicle day-use fee (allow equivalent phrasing like '$10 per vehicle')."
    )

    # Leaf: Dog policy (dogs allowed on leash <6 feet)
    dogs_node = evaluator.add_leaf(
        id="anza_borrego_dogs",
        desc="States Anza-Borrego dog policy consistent with constraints (dogs allowed on leash <6 feet)",
        parent=park_node,
        critical=True
    )
    dogs_claim = (
        f"The dog policy at {park_name} is: {info.dog_policy}."
        if info and info.dog_policy else
        f"The answer explicitly states the dog policy for {park_name}."
    )
    dogs_sources = _ensure_sources(info.sources_dogs if info else [])
    await evaluator.verify(
        claim=dogs_claim,
        node=dogs_node,
        sources=dogs_sources if dogs_sources else None,
        additional_instruction="Verify whether dogs are allowed on leash (commonly <6 feet) and any restrictions (e.g., not allowed on trails)."
    )


async def verify_torrey_pines(evaluator: Evaluator, parent_node, info: Optional[ParkInfo]) -> None:
    park_name = "Torrey Pines State Natural Reserve"
    park_node = evaluator.add_parallel(
        id="torrey_pines",
        desc="Torrey Pines State Natural Reserve information provided",
        parent=parent_node,
        critical=False
    )

    # Leaf: Saturday open status (explicitly stated)
    open_node = evaluator.add_leaf(
        id="torrey_pines_saturday_open_status",
        desc="States whether Torrey Pines is open for day-use on Saturdays in early Dec 2025",
        parent=park_node,
        critical=True
    )
    open_claim = _presence_claim(
        info.saturday_open_status if info else None,
        park_name,
        "Saturday day-use open/closed status in early December 2025"
    )
    await evaluator.verify(
        claim=open_claim,
        node=open_node,
        sources=None,
        additional_instruction="Verify presence in the answer: it should clearly say whether Saturday is open or closed for day-use in early December 2025."
    )

    # Leaf: Operating hours (gates open 7:15 AM; close at sunset)
    hours_node = evaluator.add_leaf(
        id="torrey_pines_operating_hours",
        desc="Provides Torrey Pines operating hours consistent with constraints (gates open 7:15 AM; close at sunset)",
        parent=park_node,
        critical=True
    )
    hours_claim = (
        f"The operating hours at {park_name} are: {info.operating_hours}."
        if info and info.operating_hours else
        f"The answer explicitly provides the operating hours for {park_name}."
    )
    hours_sources = _ensure_sources(info.sources_open_hours if info else [])
    await evaluator.verify(
        claim=hours_claim,
        node=hours_node,
        sources=hours_sources if hours_sources else None,
        additional_instruction="If hours are provided, verify that gates open at approximately 7:15 AM and close at sunset (allow minor phrasing or time variations)."
    )

    # Leaf: Trails closure notice (explicit note in the answer)
    closure_node = evaluator.add_leaf(
        id="torrey_pines_trails_closure_notice",
        desc="Notes the constraint that all trails are closed for construction November 2025 – February 2026",
        parent=park_node,
        critical=True
    )
    closure_claim = (
        f"The answer explicitly mentions that all trails at {park_name} are closed for construction from November 2025 through February 2026."
    )
    # This is a presence check in the answer; use simple verification
    await evaluator.verify(
        claim=closure_claim,
        node=closure_node,
        sources=None,
        additional_instruction="Check the answer text for an explicit note of the trail closure period (Nov 2025 – Feb 2026)."
    )

    # Leaf: Fee (demand-based pricing ranges)
    fee_node = evaluator.add_leaf(
        id="torrey_pines_fee",
        desc="States Torrey Pines per-vehicle day-use parking/entry fee information consistent with constraints (North Beach $10–$25; South Beach $12–$25; demand-based pricing)",
        parent=park_node,
        critical=True
    )
    fee_claim = (
        f"The day-use parking/entry fee at {park_name} is stated as: {info.fee_per_vehicle}."
        if info and info.fee_per_vehicle else
        f"The answer explicitly provides the day-use per-vehicle fee information for {park_name}."
    )
    fee_sources = _ensure_sources(info.sources_fee if info else [])
    await evaluator.verify(
        claim=fee_claim,
        node=fee_node,
        sources=fee_sources if fee_sources else None,
        additional_instruction=(
            "Verify demand-based pricing ranges (allow minor variations): examples include North Beach $10–$25, "
            "South Beach $12–$25. Confirm per-vehicle parking/entry fees."
        )
    )

    # Leaf: Dog policy (dogs not allowed)
    dogs_node = evaluator.add_leaf(
        id="torrey_pines_dogs",
        desc="States Torrey Pines dog policy consistent with constraints (dogs not allowed)",
        parent=park_node,
        critical=True
    )
    dogs_claim = (
        f"The dog policy at {park_name} is: {info.dog_policy}."
        if info and info.dog_policy else
        f"The answer explicitly states the dog policy for {park_name}."
    )
    dogs_sources = _ensure_sources(info.sources_dogs if info else [])
    await evaluator.verify(
        claim=dogs_claim,
        node=dogs_node,
        sources=dogs_sources if dogs_sources else None,
        additional_instruction="Verify whether dogs are not allowed in the reserve; allow minor phrasing variations."
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
    Evaluate the agent's answer for the San Diego County parks day-use task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Compare parks independently with partial credit
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
    parks_info = await evaluator.extract(
        prompt=prompt_extract_parks(),
        template_class=ParksExtraction,
        extraction_name="parks_extraction",
    )

    # Build verification subtrees for each park as parallel children of root
    await verify_wilderness_gardens(evaluator, root, parks_info.wilderness_gardens)
    await verify_anza_borrego(evaluator, root, parks_info.anza_borrego)
    await verify_torrey_pines(evaluator, root, parks_info.torrey_pines)

    # Optional: add contextual custom info
    evaluator.add_custom_info(
        info={
            "target_period": "Early December 2025 (Saturday focus)",
            "parks": [
                "Wilderness Gardens County Preserve",
                "Anza-Borrego Desert State Park",
                "Torrey Pines State Natural Reserve",
            ]
        },
        info_type="context",
        info_name="task_context"
    )

    return evaluator.get_summary()