import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "msc_mediterranean_shore_parks"
TASK_DESCRIPTION = """
I am planning shore excursions for an MSC Mediterranean cruise that stops at Barcelona and Valencia in Spain. I want to identify 3 Spanish national parks (Parque Nacional) that I could potentially visit as day trips or shore excursions from these ports, where I can go hiking on named trails.

For each of the 3 national parks, please provide:
1. The official name of the national park
2. Which MSC cruise port it is accessible from (Barcelona or Valencia)
3. The name of at least one specific hiking trail in that park
4. The documented distance of that trail in kilometers, specifying whether it is one-way or round-trip distance
5. Documented altitude or elevation information for the park or trail (such as maximum altitude, elevation gain, or base/summit elevations in meters)
6. Reference URLs supporting all the provided information

Each national park must:
- Be officially designated as a "Parque Nacional" (National Park) in Spain
- Be accessible as a day trip or shore excursion from Barcelona or Valencia
- Have at least one named, marked hiking trail suitable for tourist day hiking
- Have documented trail distances and altitude/elevation data available
"""


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class ParkInfo(BaseModel):
    park_name: Optional[str] = None
    access_port: Optional[str] = None  # Expected: "Barcelona" or "Valencia"
    trail_name: Optional[str] = None
    distance_km: Optional[str] = None  # e.g., "12", "12.5", "12,5"
    distance_type: Optional[str] = None  # e.g., "one-way", "round-trip", "loop", "circular"
    altitude_info: Optional[str] = None  # e.g., "elevation gain 600 m", "max altitude 2,648 m"
    difficulty: Optional[str] = None  # e.g., "easy", "moderate", "difícil", "T2"
    suitability: Optional[str] = None  # e.g., "waymarked", "sendero señalizado"
    day_hike: Optional[str] = None  # e.g., "day hike", "single day", "stage", "etapa"
    source_urls: List[str] = Field(default_factory=list)


class ParksExtraction(BaseModel):
    parks: List[ParkInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_parks() -> str:
    return """
Extract up to 3 Spanish national parks (Parques Nacionales) mentioned in the answer that are feasible as day trips from Barcelona or Valencia (MSC cruise ports). For each selected park, extract the following fields exactly as stated in the answer:

- park_name: The official name of the national park.
- access_port: Which MSC cruise port the answer claims it is accessible from. Must be either "Barcelona" or "Valencia" if explicitly stated; otherwise set to null.
- trail_name: The name of at least one specific hiking trail in that park.
- distance_km: The trail distance in kilometers as a number string (e.g., "12", "12.5", "12,5"). Do not add "km" here; only the numeric string as it appears (use comma or dot as in the answer).
- distance_type: One of "one-way" or "round-trip". If the answer uses equivalents like "loop", "circular", "ida y vuelta", map them to "round-trip". If not specified, set to null.
- altitude_info: Any documented altitude/elevation detail in meters (e.g., "elevation gain 600 m", "max altitude 2,648 m"). Include the text exactly as presented in the answer (keep number and 'm' unit).
- difficulty: Any documented difficulty rating or description (e.g., "easy", "moderate", "difícil", "media", "alta", "T2").
- suitability: Any statement indicating the route is marked/maintained and suitable for tourist hiking (e.g., "waymarked", "sendero señalizado", "marked trail", "well-maintained").
- day_hike: Whether it is explicitly a single-day hike or a stage of a multi-day route with documented stages. Use a short phrase from the answer such as "day hike", "single day", "stage", "etapa". If not stated, set to null.
- source_urls: A list of all URLs the answer cites to support the information for this park (park status, access feasibility, trail details, distances, altitude, difficulty, suitability). Include every URL explicitly mentioned for this park.

Rules:
- Do NOT invent or infer new information. Extract only what is explicitly in the answer.
- If more than 3 parks are mentioned, return only the first 3 that are tied to Barcelona or Valencia access.
- If fewer than 3 parks are provided, return all available; missing fields must be null.
- For access_port, only return "Barcelona" or "Valencia" if clearly stated for that park; otherwise null.
- For distance_type, normalize synonyms: "loop"/"circular"/"ida y vuelta" -> "round-trip".
- Always return an array 'parks' with up to 3 items in the original order they appear in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _valid_urls(urls: List[str]) -> List[str]:
    cleaned: List[str] = []
    for u in urls or []:
        if isinstance(u, str):
            uu = u.strip()
            if uu and (uu.startswith("http://") or uu.startswith("https://")):
                cleaned.append(uu)
    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for u in cleaned:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _safe(value: Optional[str], placeholder: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else placeholder


def _normalize_port(port: Optional[str]) -> Optional[str]:
    if not isinstance(port, str):
        return None
    p = port.strip().lower()
    if p == "barcelona":
        return "Barcelona"
    if p == "valencia":
        return "Valencia"
    return None


# --------------------------------------------------------------------------- #
# Verification for a single park                                              #
# --------------------------------------------------------------------------- #
async def verify_one_park(
    evaluator: Evaluator,
    parent_node,
    park: ParkInfo,
    index: int,
) -> None:
    """
    Build the verification sub-tree for one park and execute all checks.
    """
    park_idx = index + 1
    park_node = evaluator.add_parallel(
        id=f"park_{park_idx}",
        desc=f"{['First','Second','Third'][index]} national park meeting all specified criteria",
        parent=parent_node,
        critical=False,  # each park contributes partial credit
    )

    # Gather sources and add a gating check for URL presence (critical prerequisite)
    sources_list = _valid_urls(park.source_urls)
    urls_ok = len(sources_list) > 0

    # Critical gating: presence of at least one valid URL for this park bundle
    evaluator.add_custom_node(
        result=urls_ok,
        id=f"park_{park_idx}_reference_urls",
        desc="All information about the park is supported by valid reference URLs",
        parent=park_node,
        critical=True,
    )

    # Prepare common context
    p_name = _safe(park.park_name, "[park name missing]")
    t_name = _safe(park.trail_name, "[trail name missing]")
    port_norm = _normalize_port(park.access_port)
    port_label = port_norm if port_norm else _safe(park.access_port, "[port unspecified]")

    # 1) Official national park designation (Parque Nacional)
    n_loc = evaluator.add_leaf(
        id=f"park_{park_idx}_location",
        desc="The park is officially designated as a national park (Parque Nacional) in Spain",
        parent=park_node,
        critical=True,
    )
    claim_loc = f"The park '{p_name}' is officially designated as a Spanish National Park (Parque Nacional)."
    await evaluator.verify(
        claim=claim_loc,
        node=n_loc,
        sources=sources_list,
        additional_instruction="Look for explicit references like 'Parque Nacional', 'National Park (Spain)', or membership in 'Red de Parques Nacionales' on authoritative sources (official park/government pages, Wikipedia infobox, etc.). If the sources do not support this designation, mark as not supported.",
    )

    # 2) Cruise access feasibility as a day trip from Barcelona or Valencia
    n_access = evaluator.add_leaf(
        id=f"park_{park_idx}_cruise_access",
        desc="The park is accessible as a day trip from Barcelona or Valencia (MSC cruise ports in Spain)",
        parent=park_node,
        critical=True,
    )
    claim_access = (
        f"The park '{p_name}' is reasonably accessible as a day trip from the MSC cruise port of {port_label} "
        f"by ground/public transportation (car/bus/train) within a few hours, making a hike feasible during a port day."
    )
    await evaluator.verify(
        claim=claim_access,
        node=n_access,
        sources=sources_list,
        additional_instruction=(
            "Treat 'day trip' as feasible if sources show typical travel time/distance from the named port city "
            "to the park's visitor center or a common trailhead is roughly a few hours each way (≈3 hours or less). "
            "Accept indirect evidence (e.g., distance in km and highway routes, or transit connections) that clearly "
            "indicates day-trip feasibility. If no source supports reachability/time/distance from the specified port "
            "city, mark as not supported."
        ),
    )

    # 3) Named trail exists in the park
    n_trail_named = evaluator.add_leaf(
        id=f"park_{park_idx}_named_trail",
        desc="The park has at least one officially named hiking trail",
        parent=park_node,
        critical=True,
    )
    claim_named = f"The park '{p_name}' has a hiking trail named '{t_name}'."
    await evaluator.verify(
        claim=claim_named,
        node=n_trail_named,
        sources=sources_list,
        additional_instruction=(
            "Verify that the cited page(s) explicitly mention a named route within the park boundaries. "
            "Allow Spanish terms such as 'ruta', 'sendero', 'GR/PR' numbered trails."
        ),
    )

    # 4) Trail has documented distance in kilometers and distance type (one-way or round-trip)
    n_distance = evaluator.add_leaf(
        id=f"park_{park_idx}_trail_distance",
        desc="The named trail has a documented specific distance in kilometers with specification of whether it is one-way or round-trip",
        parent=park_node,
        critical=True,
    )
    dist_val = _safe(park.distance_km, "[distance missing]")
    dist_type = _safe(park.distance_type, "[type missing]")  # expect 'one-way' or 'round-trip'
    claim_distance = (
        f"The trail '{t_name}' has a documented distance of {dist_val} km and it is {dist_type}."
    )
    await evaluator.verify(
        claim=claim_distance,
        node=n_distance,
        sources=sources_list,
        additional_instruction=(
            "Confirm both: (1) an explicit numeric distance in kilometers; and (2) whether the distance is one-way "
            "or round-trip. Consider synonyms: 'loop'/'circular'/'ida y vuelta' imply round-trip; 'linear' implies one-way. "
            "Accept minor numeric rounding (e.g., 12 vs 12.1). Spanish decimals may use commas (e.g., 12,5 km). "
            "If either the distance value in km or the direction/type is missing from sources, mark as not supported."
        ),
    )

    # 5) Altitude/elevation information present
    n_alt = evaluator.add_leaf(
        id=f"park_{park_idx}_altitude_data",
        desc="The park or its trails have documented altitude/elevation information (maximum altitude, elevation gain, or base/summit elevations in meters)",
        parent=park_node,
        critical=True,
    )
    alt_text = _safe(park.altitude_info, "[altitude/elevation info missing]")
    claim_alt = (
        f"The park '{p_name}' or the trail '{t_name}' has documented altitude/elevation information: {alt_text} (meters)."
    )
    await evaluator.verify(
        claim=claim_alt,
        node=n_alt,
        sources=sources_list,
        additional_instruction=(
            "Look for altitude/elevation metrics in meters, such as 'altitud', 'desnivel', 'elevation gain', "
            "'max altitude', or 'cota'. The page should include a number with 'm'."
        ),
    )

    # 6) Trail suitability (marked/maintained; non-technical hiking)
    n_suit = evaluator.add_leaf(
        id=f"park_{park_idx}_trail_suitability",
        desc="The trail is marked/maintained and suitable for tourist hiking (not requiring technical climbing equipment)",
        parent=park_node,
        critical=True,
    )
    claim_suit = (
        f"The trail '{t_name}' is a waymarked or maintained hiking route suitable for tourists without technical climbing equipment."
    )
    await evaluator.verify(
        claim=claim_suit,
        node=n_suit,
        sources=sources_list,
        additional_instruction=(
            "Accept language indicating marked/maintained trails such as 'sendero señalizado', 'waymarked', 'signposted', "
            "'PR/GR' trails, 'well-maintained path'. Reject routes requiring ropes, harnesses, scrambling grades beyond basic hiking."
        ),
    )

    # 7) Documented difficulty present
    n_diff = evaluator.add_leaf(
        id=f"park_{park_idx}_trail_difficulty",
        desc="The trail has documented difficulty ratings or descriptions",
        parent=park_node,
        critical=True,
    )
    diff_text = _safe(park.difficulty, "[difficulty not specified]")
    claim_diff = (
        f"The trail '{t_name}' has a documented difficulty rating or description (e.g., '{diff_text}' or equivalent)."
    )
    await evaluator.verify(
        claim=claim_diff,
        node=n_diff,
        sources=sources_list,
        additional_instruction=(
            "Look for explicit difficulty wording or scale (e.g., fácil/moderado/difícil; baja/media/alta; T1/T2/T3; "
            "easy/moderate/hard). If none is present, mark as not supported."
        ),
    )

    # 8) Trail completable in a single day or is a documented stage
    n_complete = evaluator.add_leaf(
        id=f"park_{park_idx}_trail_completability",
        desc="The trail is completable in a single day or is part of multi-day hut-to-hut routes with documented stages",
        parent=park_node,
        critical=True,
    )
    claim_complete = (
        f"The trail '{t_name}' can be completed as a single-day hike, OR it is a stage of a multi-day route with documented stages."
    )
    await evaluator.verify(
        claim=claim_complete,
        node=n_complete,
        sources=sources_list,
        additional_instruction=(
            "Evidence may include typical duration estimates for a day, total loop distance/time consistent with a day hike, "
            "or explicit mention of the route being a 'stage'/'etapa' of a multi-day itinerary. If no indication is present, "
            "mark as not supported."
        ),
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
    Evaluate an answer for the MSC Mediterranean shore-excursion national parks task.
    """
    # Initialize evaluator (root is parallel: 3 parks independently contribute)
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

    # Extract up to 3 parks
    extraction = await evaluator.extract(
        prompt=prompt_extract_parks(),
        template_class=ParksExtraction,
        extraction_name="parks_extraction",
    )

    parks: List[ParkInfo] = list(extraction.parks or [])
    # Keep only first 3; if fewer than 3, pad with empty ParkInfo to maintain structure
    parks = parks[:3]
    while len(parks) < 3:
        parks.append(ParkInfo())

    # Build and verify sub-trees for each park
    for i, park in enumerate(parks[:3]):
        await verify_one_park(evaluator, root, park, i)

    # Return structured evaluation summary
    return evaluator.get_summary()