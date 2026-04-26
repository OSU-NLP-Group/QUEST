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
TASK_ID = "ea_ca_350kw_ccs1"
TASK_DESCRIPTION = """
Identify 4 public DC fast charging stations operated by Electrify America in California that support 350 kW charging capability. For each station, provide the station name or address, confirmation that it is an Electrify America location, confirmation of 350 kW charging capability, and confirmation that it has CCS Type 1 connectors.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StationCandidate(BaseModel):
    """
    Represents a single station entry extracted from the answer.
    """
    name_or_address: Optional[str] = None
    urls: List[str] = Field(default_factory=list)
    operator_mentioned: Optional[str] = None
    power_info: Optional[str] = None
    connector_info: Optional[str] = None
    public_access_mentioned: Optional[str] = None
    state_or_city: Optional[str] = None


class StationsExtraction(BaseModel):
    """
    A collection of station entries extracted from the answer.
    """
    stations: List[StationCandidate] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stations() -> str:
    return """
    Extract all station entries mentioned in the answer that appear to correspond to DC fast charging locations.
    For each station, extract the following fields from the answer text exactly as written:
    - name_or_address: The station name or its street address (return null if absent).
    - urls: A list of all URLs provided that correspond to this specific station (e.g., an official Electrify America location page, PlugShare, Google Maps, etc.). Extract only valid URLs. If none are provided, return an empty list.
    - operator_mentioned: The operator brand mentioned (e.g., "Electrify America") if present; otherwise null.
    - power_info: Any explicit mention of power capability (e.g., "350 kW", "up to 350 kW") if present; otherwise null.
    - connector_info: Any explicit mention of connector types (e.g., "CCS Type 1", "CCS1", "SAE Combo", "CCS") if present; otherwise null.
    - public_access_mentioned: Whether the station is described as public or privately accessible in the answer (e.g., "public", "open to the public", "fleet-only"); otherwise null.
    - state_or_city: Any city or state designation mentioned for the station (e.g., "San Diego, CA", "California"); otherwise null.

    Return a JSON object with a single field:
    - stations: an array of objects, each representing one station with the fields above.
    
    GENERAL RULES:
    - Extract only what is explicitly present in the answer. Do not infer or invent.
    - If any field is missing for a station, set it to null (except urls which should be an empty list).
    - Include every station entry mentioned in the answer, in the order they appear.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_station_id(s: Optional[str]) -> Optional[str]:
    """
    Normalize a station identifier (name or address) to help detect duplicates.
    Lowercase, strip, and remove common punctuation.
    """
    if not s:
        return None
    import re
    s_norm = s.lower().strip()
    s_norm = re.sub(r"[^\w\s]", "", s_norm)  # remove punctuation
    s_norm = re.sub(r"\s+", " ", s_norm)     # collapse whitespace
    return s_norm


def select_first_four_unique(stations: List[StationCandidate]) -> List[StationCandidate]:
    """
    Select the first 4 unique stations by name_or_address.
    If fewer than 4, pad with empty entries.
    """
    seen = set()
    selected: List[StationCandidate] = []
    for st in stations:
        key = normalize_station_id(st.name_or_address)
        if key and key not in seen:
            selected.append(st)
            seen.add(key)
        if len(selected) >= 4:
            break
    while len(selected) < 4:
        selected.append(StationCandidate())
    return selected


def count_distinct_nonempty(stations: List[StationCandidate]) -> int:
    """
    Count distinct non-empty station identifiers across all extracted stations.
    """
    ids = set()
    for st in stations:
        key = normalize_station_id(st.name_or_address)
        if key:
            ids.add(key)
    return len(ids)


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_station(
    evaluator: Evaluator,
    parent_node,
    station: StationCandidate,
    station_index: int,
) -> None:
    """
    Build verification subtree for a single station.
    """
    station_num = station_index + 1
    st_node = evaluator.add_parallel(
        id=f"station_{station_num}",
        desc=f"Station {station_num} satisfies all constraints and required fields.",
        parent=parent_node,
        critical=False
    )

    # Existence: station name or address provided (critical)
    name_exists = bool(station.name_or_address and station.name_or_address.strip())
    evaluator.add_custom_node(
        result=name_exists,
        id=f"station_{station_num}_name_or_address",
        desc=f"Provides a station name or an address for Station {station_num}.",
        parent=st_node,
        critical=True
    )

    # Sources list for this station (could be empty)
    sources_list = station.urls or []

    # Operated by Electrify America (critical)
    node_ea = evaluator.add_leaf(
        id=f"station_{station_num}_operated_by_ea",
        desc=f"Station {station_num} is confirmed to be operated by Electrify America.",
        parent=st_node,
        critical=True
    )
    claim_ea = (
        f"The station '{station.name_or_address or f'Station {station_num}'}' is an Electrify America location operated by Electrify America."
    )
    await evaluator.verify(
        claim=claim_ea,
        node=node_ea,
        sources=sources_list,
        additional_instruction="Verify on the provided webpage(s) that the location is branded or operated by Electrify America (EA). Look for the operator field, brand logo/text, or explicit mention."
    )

    # Located in California (critical)
    node_ca = evaluator.add_leaf(
        id=f"station_{station_num}_in_california",
        desc=f"Station {station_num} is confirmed to be located in California.",
        parent=st_node,
        critical=True
    )
    claim_ca = (
        f"The station '{station.name_or_address or f'Station {station_num}'}' is located in California (CA), United States."
    )
    await evaluator.verify(
        claim=claim_ca,
        node=node_ca,
        sources=sources_list,
        additional_instruction="Check the address or location details on the page. Accept 'CA' as California and common city names within California."
    )

    # 350 kW support (critical)
    node_350 = evaluator.add_leaf(
        id=f"station_{station_num}_350kw",
        desc=f"Station {station_num} is confirmed to support 350 kW charging capability (e.g., has 350 kW-rated dispensers).",
        parent=st_node,
        critical=True
    )
    claim_350 = (
        f"The station '{station.name_or_address or f'Station {station_num}'}' offers at least one DC fast charger rated at 350 kW (e.g., 'up to 350 kW')."
    )
    await evaluator.verify(
        claim=claim_350,
        node=node_350,
        sources=sources_list,
        additional_instruction="Confirm that the page explicitly states 350 kW capability for one or more dispensers. Accept phrases like 'up to 350 kW' or '350 kW-rated'."
    )

    # CCS Type 1 connectors (critical)
    node_ccs = evaluator.add_leaf(
        id=f"station_{station_num}_ccs_type1",
        desc=f"Station {station_num} is confirmed to have CCS Type 1 connectors.",
        parent=st_node,
        critical=True
    )
    claim_ccs = (
        f"The station '{station.name_or_address or f'Station {station_num}'}' provides CCS Type 1 connectors (also known as CCS1, SAE Combo in North America)."
    )
    await evaluator.verify(
        claim=claim_ccs,
        node=node_ccs,
        sources=sources_list,
        additional_instruction="Look for connector type details such as 'CCS', 'CCS1', 'SAE Combo'. Treat 'CCS (North America)' as CCS Type 1."
    )

    # Public access (critical)
    node_public = evaluator.add_leaf(
        id=f"station_{station_num}_public_access",
        desc=f"Station {station_num} is confirmed to be publicly accessible.",
        parent=st_node,
        critical=True
    )
    claim_public = (
        f"The station '{station.name_or_address or f'Station {station_num}'}' is publicly accessible."
    )
    await evaluator.verify(
        claim=claim_public,
        node=node_public,
        sources=sources_list,
        additional_instruction="Verify that the location is public (e.g., 'Public', 'Open to public'). If the page implies fleet-only or private access, it should NOT pass."
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
    Evaluate an answer for the Electrify America California 350kW CCS Type 1 station identification task.
    """
    # Initialize evaluator; make root non-critical to allow partial credit across station nodes
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

    # Extract all stations from the answer
    stations_extracted = await evaluator.extract(
        prompt=prompt_extract_stations(),
        template_class=StationsExtraction,
        extraction_name="stations_extraction"
    )

    # Compute exactness check: exactly 4 distinct stations in the answer (by name_or_address)
    total_nonempty = [st for st in stations_extracted.stations if st.name_or_address and st.name_or_address.strip()]
    distinct_count = count_distinct_nonempty(stations_extracted.stations)
    exactly_four_result = (len(total_nonempty) == 4) and (distinct_count == 4)

    evaluator.add_custom_node(
        result=exactly_four_result,
        id="exactly_four_stations",
        desc="Response identifies exactly 4 distinct stations (not fewer/more, and not duplicates).",
        parent=root,
        critical=True
    )

    # Record some custom info for debugging
    evaluator.add_custom_info(
        info={
            "total_stations_mentioned": len(stations_extracted.stations),
            "nonempty_station_entries": len(total_nonempty),
            "distinct_station_entries": distinct_count
        },
        info_type="extraction_stats"
    )

    # Select first 4 unique stations for detailed verification
    stations_to_check = select_first_four_unique(stations_extracted.stations)

    # Build station verification subtrees
    for idx, station in enumerate(stations_to_check):
        await verify_station(evaluator, root, station, idx)

    # Return structured summary
    return evaluator.get_summary()