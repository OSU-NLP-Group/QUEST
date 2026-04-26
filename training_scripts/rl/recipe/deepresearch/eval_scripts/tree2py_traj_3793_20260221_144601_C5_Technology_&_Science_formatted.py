import asyncio
import logging
import re
from typing import Any, Optional, List, Dict, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tech_infra_cities"
TASK_DESCRIPTION = (
    "Identify four distinct cities or metropolitan regions in the United States where each location satisfies ALL of the following requirements:\n\n"
    "1. National Research Facility: The city must host at least one U.S. Department of Energy (DOE) National Laboratory or a DOE National Quantum Information Science (QIS) Research Center.\n\n"
    "2. Data Center Infrastructure: The city must have documented existing or planned data center infrastructure with a minimum power capacity of 20 megawatts (MW).\n\n"
    "3. 5G Network Deployment: The city must be located in a region with documented 5G network infrastructure deployment.\n\n"
    "4. Research University Presence: The city must have at least one major research university that conducts technology research in areas such as semiconductors, quantum computing, artificial intelligence, or related fields.\n\n"
    "For each of the four cities you identify, provide:\n"
    "- The city name and state\n"
    "- The name of the DOE National Laboratory or QIS Research Center located there\n"
    "- Information about the data center infrastructure, including its power capacity\n"
    "- Evidence of 5G network deployment in the region\n"
    "- The name of the research university and its relevant technology research areas\n"
    "- Reference URLs that verify each of these requirements\n\n"
    "All four cities must be distinct locations (not within the same immediate metropolitan area)."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DOEFacility(BaseModel):
    name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class DataCenterInfo(BaseModel):
    name: Optional[str] = None
    capacity_mw: Optional[str] = None  # Keep as string for flexibility (e.g., "20-40 MW", "25 MW planned")
    urls: List[str] = Field(default_factory=list)


class FiveGInfo(BaseModel):
    provider: Optional[str] = None
    description: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ResearchUniversity(BaseModel):
    name: Optional[str] = None
    research_areas: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class CityInfo(BaseModel):
    city_name: Optional[str] = None
    state: Optional[str] = None
    doe_facility: Optional[DOEFacility] = None
    data_centers: List[DataCenterInfo] = Field(default_factory=list)
    fiveg: Optional[FiveGInfo] = None
    university: Optional[ResearchUniversity] = None


class CitiesExtraction(BaseModel):
    cities: List[CityInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_cities() -> str:
    return """
    Extract up to four distinct U.S. cities or metropolitan regions mentioned in the answer, along with structured information for each that satisfies all required criteria. 
    For each identified city, extract the following fields:

    - city_name: The name of the city (or metropolitan region name if provided)
    - state: The U.S. state (two-letter abbreviation or full name)
    - doe_facility:
        - name: The name of the DOE National Laboratory or DOE National Quantum Information Science (QIS) Research Center located in or near the city/metropolitan region
        - urls: An array of URLs that explicitly support the presence/location of this facility (include official DOE or lab webpages when available)
    - data_centers: An array of data center entries (extract all mentioned), each with:
        - name: The data center name or operator (e.g., "Equinix SV1")
        - capacity_mw: The documented power capacity, kept exactly as written in the answer (e.g., "25 MW", "20-40 MW", "30 MW planned")
        - urls: An array of URLs that explicitly support the data center and its capacity
    - fiveg:
        - provider: The wireless carrier or source mentioning 5G deployment (e.g., Verizon, AT&T, T-Mobile, FCC)
        - description: A short description of the 5G deployment or coverage information for this city/region
        - urls: An array of URLs that explicitly support 5G deployment in the city/region (e.g., coverage map, provider announcement, FCC page)
    - university:
        - name: The name of a major research university located in or near the city/region
        - research_areas: An array of research areas mentioned in the answer (e.g., "semiconductors", "quantum computing", "artificial intelligence")
        - urls: An array of URLs that explicitly support the university's research activities in the listed areas (e.g., department pages, lab pages, center pages)

    IMPORTANT RULES:
    - Extract ONLY information explicitly present in the provided answer text.
    - Extract URLs that are explicitly included in the answer; do NOT invent or infer URLs.
    - For each category, include all relevant URLs provided; if none are provided in the answer, return an empty array for that category.
    - If a field is missing in the answer, set it to null (or empty array, as applicable).
    - Return exactly up to four cities in the 'cities' array, preserving the order they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def parse_capacity_to_mw(capacity_str: Optional[str]) -> Optional[float]:
    """
    Best-effort parsing of capacity string to a numeric MW value.
    - Supports patterns like "25 MW", "20-40 MW", "30MW planned".
    - If multiple numbers are present, use the maximum as a conservative estimate.
    - Handles GW and kW units if explicitly present.
    """
    if not capacity_str:
        return None
    s = capacity_str.lower()
    # Determine unit multiplier
    multiplier = 1.0
    if " gw" in s or s.endswith("gw") or "gw " in s:
        multiplier = 1000.0
    elif " kw" in s or s.endswith("kw") or "kw " in s:
        multiplier = 0.001
    else:
        multiplier = 1.0  # default MW

    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if not nums:
        return None
    values = [float(n) * multiplier for n in nums]
    return max(values) if values else None


def select_best_data_center(city: CityInfo) -> Tuple[Optional[DataCenterInfo], Optional[float]]:
    """
    Select the data center with the highest parsed capacity (MW) for verification.
    Returns (DataCenterInfo, parsed_capacity_mw)
    """
    best_dc = None
    best_cap = None
    for dc in city.data_centers:
        cap = parse_capacity_to_mw(dc.capacity_mw)
        if cap is not None and (best_cap is None or cap > best_cap):
            best_dc = dc
            best_cap = cap
    # If none could be parsed, fallback to first item if present
    if best_dc is None and city.data_centers:
        best_dc = city.data_centers[0]
        best_cap = parse_capacity_to_mw(best_dc.capacity_mw)
    return best_dc, best_cap


def has_verification_urls_for_city(city: CityInfo) -> bool:
    """
    Require at least one URL for each category:
    - DOE facility
    - Data center (at least one of the listed has URLs)
    - 5G deployment
    - Research university
    """
    facility_ok = bool(city.doe_facility and city.doe_facility.urls)
    dc_ok = any(dc.urls for dc in (city.data_centers or []))
    fiveg_ok = bool(city.fiveg and city.fiveg.urls)
    uni_ok = bool(city.university and city.university.urls)
    return facility_ok and dc_ok and fiveg_ok and uni_ok


def city_label(city: CityInfo) -> str:
    name = (city.city_name or "").strip()
    state = (city.state or "").strip()
    if name and state:
        return f"{name}, {state}"
    return name or state or "Unknown City"


# --------------------------------------------------------------------------- #
# Verification for a single city                                              #
# --------------------------------------------------------------------------- #
async def verify_city(
    evaluator: Evaluator,
    parent_node,
    city: CityInfo,
    city_index: int,
) -> None:
    """
    Build verification nodes and perform checks for one city/region.
    """
    city_id = f"city_{city_index + 1}"
    city_desc = (
        "First identified city/region meets all required criteria" if city_index == 0 else
        "Second identified city/region meets all required criteria and is distinct from the first city" if city_index == 1 else
        "Third identified city/region meets all required criteria and is distinct from the first two cities" if city_index == 2 else
        "Fourth identified city/region meets all required criteria and is distinct from the first three cities"
    )

    city_node = evaluator.add_parallel(
        id=city_id,
        desc=city_desc,
        parent=parent_node,
        critical=False
    )

    # 1) National Lab / QIS Center presence (critical)
    lab_leaf = evaluator.add_leaf(
        id=f"{city_id}_national_lab_presence",
        desc="City hosts a DOE National Laboratory or DOE National Quantum Information Science Research Center",
        parent=city_node,
        critical=True
    )
    facility_name = city.doe_facility.name if city.doe_facility else None
    facility_urls = city.doe_facility.urls if (city.doe_facility and city.doe_facility.urls) else []
    claim_lab = (
        f"The location {city_label(city)} hosts the DOE facility '{facility_name}' (National Laboratory or DOE National QIS Research Center), "
        f"located in or within the metropolitan region of {city_label(city)}."
        if facility_name else
        f"The location {city_label(city)} hosts a DOE National Laboratory or DOE National QIS Research Center."
    )
    await evaluator.verify(
        claim=claim_lab,
        node=lab_leaf,
        sources=facility_urls,
        additional_instruction=(
            "Confirm that the cited DOE facility is located in or immediately adjacent to the specified city/metropolitan region. "
            "Allow metropolitan-area/locality interpretation (e.g., suburbs or neighboring towns commonly considered part of the metro). "
            "The source should be an official or credible page explicitly referencing the facility and its location."
        )
    )

    # 2) Data center infrastructure >= 20 MW (critical)
    dc_leaf = evaluator.add_leaf(
        id=f"{city_id}_data_center_infrastructure",
        desc="City has documented data center infrastructure with minimum 20 MW power capacity",
        parent=city_node,
        critical=True
    )
    best_dc, best_cap = select_best_data_center(city)
    dc_urls = best_dc.urls if (best_dc and best_dc.urls) else []
    dc_name_str = best_dc.name if best_dc and best_dc.name else "a data center"
    cap_str = best_dc.capacity_mw if best_dc and best_dc.capacity_mw else "capacity documented on the cited page"
    claim_dc = (
        f"There is documented existing or planned data center infrastructure in or near {city_label(city)} "
        f"with at least 20 MW of power capacity. Example: {dc_name_str} with capacity '{cap_str}'."
    )
    await evaluator.verify(
        claim=claim_dc,
        node=dc_leaf,
        sources=dc_urls,
        additional_instruction=(
            "Verify that the cited page explicitly documents a data center (existing or planned) in or near the specified city/region "
            "with a power capacity that is at least 20 MW. Accept ranges or planned capacities if they meet or exceed 20 MW."
        )
    )

    # 3) 5G network deployment documented (critical)
    g_leaf = evaluator.add_leaf(
        id=f"{city_id}_5g_deployment",
        desc="City is in a region with documented 5G network infrastructure deployment",
        parent=city_node,
        critical=True
    )
    fiveg_urls = city.fiveg.urls if (city.fiveg and city.fiveg.urls) else []
    provider_str = city.fiveg.provider if (city.fiveg and city.fiveg.provider) else "a provider or authoritative source"
    claim_5g = (
        f"The region encompassing {city_label(city)} has documented 5G network infrastructure deployment, as evidenced by {provider_str}."
    )
    await evaluator.verify(
        claim=claim_5g,
        node=g_leaf,
        sources=fiveg_urls,
        additional_instruction=(
            "Confirm that the source explicitly indicates 5G deployment or coverage in the specified city/metropolitan region."
        )
    )

    # 4) Research university conducting relevant technology research (critical)
    uni_leaf = evaluator.add_leaf(
        id=f"{city_id}_research_university",
        desc="City has a major research university conducting technology research in semiconductors, quantum computing, AI, or related fields",
        parent=city_node,
        critical=True
    )
    uni_name = city.university.name if city.university else None
    uni_urls = city.university.urls if (city.university and city.university.urls) else []
    areas = city.university.research_areas if (city.university and city.university.research_areas) else []
    areas_str = ", ".join(areas) if areas else "technology research areas (e.g., semiconductors, quantum computing, AI)"
    claim_uni = (
        f"The city/region {city_label(city)} has a major research university '{uni_name}' that conducts research in {areas_str}."
        if uni_name else
        f"The city/region {city_label(city)} has a major research university conducting research in {areas_str}."
    )
    await evaluator.verify(
        claim=claim_uni,
        node=uni_leaf,
        sources=uni_urls,
        additional_instruction=(
            "Verify that the university is located in or near the specified city/region and that the cited page(s) explicitly show research activities "
            "in semiconductors, quantum computing, artificial intelligence, or closely related technology areas. Department/lab/center pages are acceptable."
        )
    )

    # 5) Verification URLs presence check (critical existence check)
    urls_leaf = evaluator.add_custom_node(
        result=has_verification_urls_for_city(city),
        id=f"{city_id}_verification_urls",
        desc="Provide reference URLs verifying the national lab/QIS center, data center infrastructure, 5G deployment, and research university for this city",
        parent=city_node,
        critical=True
    )

    # Add helpful summary for this city to the evaluator info
    evaluator.add_custom_info(
        info={
            "city": city_label(city),
            "doe_facility_name": facility_name,
            "doe_facility_urls_count": len(facility_urls),
            "selected_data_center": best_dc.name if best_dc else None,
            "selected_data_center_capacity_str": best_dc.capacity_mw if best_dc else None,
            "selected_data_center_capacity_parsed_mw": best_cap,
            "selected_data_center_urls_count": len(dc_urls),
            "fiveg_provider": provider_str,
            "fiveg_urls_count": len(fiveg_urls),
            "university_name": uni_name,
            "university_areas": areas,
            "university_urls_count": len(uni_urls),
            "verification_urls_present_all_categories": urls_leaf.score == 1.0
        },
        info_type="city_summary",
        info_name=f"city_{city_index + 1}_summary"
    )


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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate the answer for the 'tech_infra_cities' task and return a structured summary.
    """
    # Initialize evaluator with a parallel root
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

    # Extract structured city information from the answer
    extraction: CitiesExtraction = await evaluator.extract(
        prompt=prompt_extract_cities(),
        template_class=CitiesExtraction,
        extraction_name="cities_extraction"
    )

    # Use only the first four cities (pad with placeholders if fewer)
    cities: List[CityInfo] = list(extraction.cities[:4])
    while len(cities) < 4:
        cities.append(CityInfo())

    # Build per-city verification subtrees
    for idx in range(4):
        await verify_city(evaluator, root, cities[idx], idx)

    # Add an overall distinctness check (critical under root)
    # We verify via simple logical reasoning (no sources) that the cities are distinct and not within the same immediate metropolitan area.
    distinct_leaf = evaluator.add_leaf(
        id="distinct_cities",
        desc="All four identified cities are distinct locations (not within the same immediate metropolitan area)",
        parent=root,
        critical=True
    )
    city_labels = [city_label(cities[i]) for i in range(4)]
    claim_distinct = (
        f"The following four U.S. locations are distinct and NOT within the same immediate metropolitan area: {', '.join(city_labels)}."
    )
    await evaluator.verify(
        claim=claim_distinct,
        node=distinct_leaf,
        additional_instruction=(
            "Judge distinctness logically based on city/state names and common knowledge of metropolitan areas. "
            "Allow suburban localities to be considered part of the same metro; the four should not be in the same immediate metro."
        )
    )

    # Add custom info about threshold policy
    evaluator.add_custom_info(
        info={"minimum_required_dc_capacity_mw": 20.0},
        info_type="policy",
        info_name="data_center_threshold_policy"
    )

    # Return structured summary
    return evaluator.get_summary()