import asyncio
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_colocation_primary_dr_selection"
TASK_DESCRIPTION = """
Your organization is planning to expand its IT infrastructure by deploying equipment in two colocation data center facilities in California to support enterprise applications with high availability requirements. You need to identify one primary facility and one disaster recovery (DR) facility that meet the following requirements:

Primary Facility Requirements:
- Must be located in the Los Angeles metropolitan area
- Must hold Uptime Institute Tier III certification (or explicitly state Tier III equivalent specifications: 99.982% uptime, N+1 redundancy, concurrent maintainability)
- Must maintain current SOC 2 Type II certification
- Must maintain current ISO 27001 certification
- Must be carrier-neutral with a meet-me-room (MMR)
- Must support minimum 10 kW per rack power density
- Must provide direct access to at least 3 independent telecommunications carriers or network service providers
- Must offer physical cross-connect services
- Must maintain 24/7 onsite security personnel
- Must be operated by an established colocation provider with multiple data center locations nationwide

Disaster Recovery Facility Requirements:
- Must be located in Northern California (San Francisco Bay Area, Silicon Valley, or Sacramento region)
- Must be located at least 100 miles from the primary Los Angeles facility for adequate geographic redundancy
- Must hold Uptime Institute Tier III certification (or explicitly state Tier III equivalent specifications: 99.982% uptime, N+1 redundancy, concurrent maintainability)
- Must maintain current SOC 2 Type II certification
- Must maintain current ISO 27001 certification
- Must be carrier-neutral with a meet-me-room (MMR)
- Must support minimum 10 kW per rack power density
- Must provide direct access to at least 3 independent telecommunications carriers or network service providers
- Must offer physical cross-connect services
- Must maintain 24/7 onsite security personnel
- Must be operated by an established colocation provider with multiple data center locations nationwide

For each facility, provide:
1. The facility name and specific data center identifier (if applicable)
2. The complete street address
3. The operating company/provider name
4. A reference URL that confirms the facility specifications and certifications
"""


# --------------------------------------------------------------------------- #
# Geography utilities                                                         #
# --------------------------------------------------------------------------- #
# Approximate lat/long for key CA cities (sufficient for 100-mile separation check)
CITY_COORDS = {
    # Los Angeles metro anchors
    "los angeles": (34.0522, -118.2437),
    "el segundo": (33.9164, -118.4040),
    "torrance": (33.8358, -118.3406),
    "burbank": (34.1808, -118.3089),
    "culver city": (34.0211, -118.3965),
    "pasadena": (34.1478, -118.1445),
    "glendale": (34.1425, -118.2551),
    "long beach": (33.7701, -118.1937),
    "irvine": (33.6846, -117.8265),
    "anaheim": (33.8366, -117.9143),
    "santa clarita": (34.3917, -118.5426),

    # Northern California (Bay Area, Silicon Valley, Sacramento)
    "san jose": (37.3382, -121.8863),
    "santa clara": (37.3541, -121.9552),
    "sunnyvale": (37.3688, -122.0363),
    "mountain view": (37.3861, -122.0839),
    "palo alto": (37.4419, -122.1430),
    "milpitas": (37.4323, -121.8996),
    "fremont": (37.5485, -121.9886),
    "redwood city": (37.4852, -122.2364),
    "san mateo": (37.5630, -122.3255),
    "menlo park": (37.4530, -122.1817),
    "san francisco": (37.7749, -122.4194),
    "oakland": (37.8044, -122.2712),
    "berkeley": (37.8715, -122.2730),
    "hayward": (37.6688, -122.0808),
    "walnut creek": (37.9101, -122.0652),
    "sacramento": (38.5816, -121.4944),
    "roseville": (38.7521, -121.2880),
    "rancho cordova": (38.5891, -121.3027),
    "folsom": (38.6779, -121.1761),
    "santa rosa": (38.4405, -122.7144),
}


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    km = R_km * c
    return km * 0.621371


def try_city_distance_miles(city_a: Optional[str], city_b: Optional[str]) -> Optional[float]:
    if not city_a or not city_b:
        return None
    a = CITY_COORDS.get(city_a.strip().lower())
    b = CITY_COORDS.get(city_b.strip().lower())
    if not a or not b:
        return None
    return haversine_miles(a[0], a[1], b[0], b[1])


def parse_city_from_address(street_address: Optional[str]) -> Optional[str]:
    if not street_address:
        return None
    parts = [p.strip() for p in street_address.split(",")]
    if len(parts) >= 2:
        return parts[-2]  # common pattern: "street, City, CA 9xxxx"
    return None


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FacilityInfo(BaseModel):
    facility_name: Optional[str] = None
    data_center_id: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    provider_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    provider_urls: List[str] = Field(default_factory=list)


class FacilitiesExtraction(BaseModel):
    primary: Optional[FacilityInfo] = None
    dr: Optional[FacilityInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facilities() -> str:
    return """
    Extract exactly two colocation facility entries from the answer: one designated as "primary" and one as "dr" (disaster recovery).
    If the answer explicitly labels them (e.g., "Primary", "DR"), follow that mapping. If not labeled, treat the first qualifying entry as primary and the second as dr.

    For each facility, extract the following fields as literally stated in the answer:
    - facility_name: The facility/site name (e.g., "Equinix LA1", "Digital Realty LAX10", etc.)
    - data_center_id: An explicit DC identifier if given (e.g., "LA1", "SV5"); else null
    - street_address: Full street address line as provided
    - city: The city name (if provided; otherwise try to infer from the address text; if unclear, set null)
    - state: The US state (e.g., "CA") if provided; else null
    - provider_name: The operating colocation company (e.g., Equinix, CoreSite, Digital Realty)
    - reference_urls: An array of one or more URLs that the answer cites to substantiate the facility’s specs/certifications. Include all such URLs. If none are provided, set to an empty array.
    - provider_urls: An array of URLs pointing to the provider’s main site or overview pages if explicitly cited in the answer. If none are provided, use an empty array.

    Output a JSON object with two top-level fields: "primary" and "dr", each being an object with the fields above.
    If any field is missing, set it to null (or an empty array for URL lists).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def non_empty_str(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def combine_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    result = []
    for urls in url_lists:
        for u in urls:
            if u and isinstance(u, str):
                key = u.strip()
                if key and key not in seen:
                    seen.add(key)
                    result.append(key)
    return result


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_primary_facility_checks(evaluator: Evaluator, parent_node, primary: FacilityInfo) -> None:
    # Group node for primary facility
    p_node = evaluator.add_parallel(
        id="primary_facility",
        desc="Primary colocation facility located in Los Angeles metropolitan area",
        parent=parent_node,
        critical=False
    )

    # Basic presence checks (critical)
    evaluator.add_custom_node(
        result=non_empty_str(primary.facility_name) or non_empty_str(primary.data_center_id),
        id="primary_facility_name_provided",
        desc="The facility name and specific data center identifier (if applicable) are provided",
        parent=p_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=non_empty_str(primary.street_address),
        id="primary_street_address_provided",
        desc="The complete street address of the facility is provided",
        parent=p_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=non_empty_str(primary.provider_name),
        id="primary_operating_company_provided",
        desc="The operating company/provider name is provided",
        parent=p_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(primary.reference_urls) > 0,
        id="primary_reference_url",
        desc="A reference URL is provided that confirms the facility specifications and certifications",
        parent=p_node,
        critical=True
    )

    # URLs to use for verification
    p_urls = combine_urls(primary.reference_urls, primary.provider_urls)

    # Facility identity text for claims
    facility_label = primary.facility_name or (primary.data_center_id or "the primary facility")
    addr_label = primary.street_address or ""
    provider_label = primary.provider_name or "the provider"

    # Location requirement (critical)
    node = evaluator.add_leaf(
        id="primary_location",
        desc="Facility is located in Los Angeles metropolitan area",
        parent=p_node,
        critical=True
    )
    claim = f"The facility '{facility_label}' at address '{addr_label}' is located in the Los Angeles metropolitan area (e.g., Los Angeles County or adjacent cities such as El Segundo, Torrance, Burbank, Culver City, Pasadena, Long Beach, Glendale)."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=p_urls,
        additional_instruction="Accept mentions of Los Angeles metro or LA County or clearly LA-adjacent cities as satisfying this requirement."
    )

    # Tier III or equivalent (critical)
    node = evaluator.add_leaf(
        id="primary_tier_certification",
        desc="Facility holds Uptime Institute Tier III certification or explicitly states Tier III equivalent specifications (99.982% uptime, N+1 redundancy, concurrent maintainability)",
        parent=p_node,
        critical=True
    )
    claim = f"The facility '{facility_label}' operated by {provider_label} is either Uptime Institute Tier III certified OR explicitly states Tier III-equivalent specifications: 99.982% availability, N+1 redundancy, and concurrent maintainability."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=p_urls,
        additional_instruction="Look for 'Tier III' or combinations of 99.982% uptime, N+1, and concurrent maintainability. Equivalent wording is acceptable."
    )

    # SOC 2 Type II (critical)
    node = evaluator.add_leaf(
        id="primary_soc2_certification",
        desc="Facility maintains current SOC 2 Type II certification",
        parent=p_node,
        critical=True
    )
    claim = f"The facility '{facility_label}' maintains a current SOC 2 Type II certification (not just Type I)."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=p_urls,
        additional_instruction="Confirm 'SOC 2 Type II' anywhere on the page(s). 'Type II' specifically is required."
    )

    # ISO 27001 (critical)
    node = evaluator.add_leaf(
        id="primary_iso27001_certification",
        desc="Facility maintains current ISO 27001 certification",
        parent=p_node,
        critical=True
    )
    claim = f"The facility '{facility_label}' maintains a current ISO/IEC 27001 certification."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=p_urls,
        additional_instruction="Wording variations such as 'ISO 27001 certified' are acceptable."
    )

    # Carrier-neutral + MMR (critical)
    node = evaluator.add_leaf(
        id="primary_carrier_neutral",
        desc="Facility is carrier-neutral with meet-me-room (MMR) for multi-carrier interconnection",
        parent=p_node,
        critical=True
    )
    claim = f"The facility '{facility_label}' is carrier-neutral and has a meet-me room (MMR) enabling interconnection with multiple carriers."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=p_urls,
        additional_instruction="Either explicit 'carrier-neutral' + 'meet-me room/MMR' or equivalent phrases that clearly indicate both should count."
    )

    # Power density >= 10 kW/rack (critical)
    node = evaluator.add_leaf(
        id="primary_power_density",
        desc="Facility supports minimum 10 kW per rack power density",
        parent=p_node,
        critical=True
    )
    claim = f"The facility '{facility_label}' supports a minimum of 10 kW per rack (or per cabinet) power density."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=p_urls,
        additional_instruction="Equivalent phrasing like '10 kW per cabinet' is acceptable. If a higher minimum is stated, it also satisfies the requirement."
    )

    # At least 3 carriers/NSPs (critical)
    node = evaluator.add_leaf(
        id="primary_network_access",
        desc="Facility provides direct access to at least 3 independent network service providers or telecommunications carriers",
        parent=p_node,
        critical=True
    )
    claim = f"The facility '{facility_label}' provides direct access to at least three independent telecommunications carriers or network service providers."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=p_urls,
        additional_instruction="Look for a carriers list or peering options that imply >= 3 distinct providers."
    )

    # Physical cross-connects (critical)
    node = evaluator.add_leaf(
        id="primary_cross_connect",
        desc="Facility offers physical cross-connect services for direct point-to-point connections",
        parent=p_node,
        critical=True
    )
    claim = f"The facility '{facility_label}' offers physical cross-connect services (e.g., copper, fiber) for direct point-to-point connections."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=p_urls,
        additional_instruction="Any explicit mention of 'cross connect(s)' counts."
    )

    # 24/7 onsite security (critical)
    node = evaluator.add_leaf(
        id="primary_physical_security",
        desc="Facility maintains 24/7 onsite security personnel",
        parent=p_node,
        critical=True
    )
    claim = f"The facility '{facility_label}' maintains 24/7 (24x7) on-site security personnel."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=p_urls,
        additional_instruction="Accept equivalent wording like '24x7 on-site security' or 'round-the-clock staffed security'."
    )

    # Provider credibility (critical)
    node = evaluator.add_leaf(
        id="primary_provider_credibility",
        desc="Facility is operated by an established colocation provider with multiple data center locations nationwide",
        parent=p_node,
        critical=True
    )
    claim = f"The provider {provider_label} operates multiple colocation data center locations nationwide in the United States (or an extensive national network), indicating it is an established colocation provider."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=p_urls,
        additional_instruction="Evidence can be on the facility page or provider pages showing many locations across the U.S. Global footprints are acceptable as stronger evidence."
    )


async def build_dr_facility_checks(evaluator: Evaluator, parent_node, primary: FacilityInfo, dr: FacilityInfo) -> None:
    # Group node for DR facility
    d_node = evaluator.add_parallel(
        id="dr_facility",
        desc="Disaster recovery colocation facility located in Northern California",
        parent=parent_node,
        critical=False
    )

    # Basic presence checks (critical)
    evaluator.add_custom_node(
        result=non_empty_str(dr.facility_name) or non_empty_str(dr.data_center_id),
        id="dr_facility_name_provided",
        desc="The facility name and specific data center identifier (if applicable) are provided",
        parent=d_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=non_empty_str(dr.street_address),
        id="dr_street_address_provided",
        desc="The complete street address of the facility is provided",
        parent=d_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=non_empty_str(dr.provider_name),
        id="dr_operating_company_provided",
        desc="The operating company/provider name is provided",
        parent=d_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(dr.reference_urls) > 0,
        id="dr_reference_url",
        desc="A reference URL is provided that confirms the facility specifications and certifications",
        parent=d_node,
        critical=True
    )

    # URLs to use for verification
    d_urls = combine_urls(dr.reference_urls, dr.provider_urls)

    # Identity labels
    facility_label = dr.facility_name or (dr.data_center_id or "the DR facility")
    addr_label = dr.street_address or ""
    provider_label = dr.provider_name or "the provider"

    # Location in Northern California (critical)
    node = evaluator.add_leaf(
        id="dr_location",
        desc="Facility is located in Northern California (San Francisco Bay Area, Silicon Valley, or Sacramento region)",
        parent=d_node,
        critical=True
    )
    claim = f"The facility '{facility_label}' at address '{addr_label}' is in Northern California (e.g., San Francisco Bay Area/Silicon Valley or the Sacramento region)."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=d_urls,
        additional_instruction="Accept mentions of Bay Area, Silicon Valley, or Sacramento region (or cities within those regions) as satisfying this requirement."
    )

    # Geographic separation >= 100 miles from LA primary (critical)
    # Try to compute with cities if available; else fallback to a simple verification claim.
    primary_city = primary.city or parse_city_from_address(primary.street_address) or "Los Angeles"
    dr_city = dr.city or parse_city_from_address(dr.street_address)
    distance = try_city_distance_miles(primary_city, dr_city)

    if distance is not None:
        # Deterministic check
        evaluator.add_custom_node(
            result=distance >= 100.0,
            id="dr_geographic_separation",
            desc=f"Facility is located at least 100 miles from the primary Los Angeles facility (estimated distance {distance:.1f} miles between {primary_city} and {dr_city})",
            parent=d_node,
            critical=True
        )
    else:
        # Fallback to a verification leaf with general-knowledge reasoning
        node = evaluator.add_leaf(
            id="dr_geographic_separation",
            desc="Facility is located at least 100 miles from the primary Los Angeles facility",
            parent=d_node,
            critical=True
        )
        los_angeles_anchor = primary_city or "Los Angeles"
        city_for_claim = dr_city or "the DR facility city"
        claim = f"The distance between {los_angeles_anchor}, CA and {city_for_claim}, CA is at least 100 miles."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=None,
            additional_instruction="Use general U.S. geography knowledge to judge approximate distance between the two cities. If {city_for_claim} is in Northern California, this condition should be satisfied."
        )

    # Tier III or equivalent (critical)
    node = evaluator.add_leaf(
        id="dr_tier_certification",
        desc="Facility holds Uptime Institute Tier III certification or explicitly states Tier III equivalent specifications (99.982% uptime, N+1 redundancy, concurrent maintainability)",
        parent=d_node,
        critical=True
    )
    claim = f"The DR facility '{facility_label}' operated by {provider_label} is either Uptime Institute Tier III certified OR explicitly states Tier III-equivalent specifications: 99.982% availability, N+1 redundancy, and concurrent maintainability."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=d_urls,
        additional_instruction="Look for 'Tier III' or combinations of 99.982% uptime, N+1, and concurrent maintainability."
    )

    # SOC 2 Type II (critical)
    node = evaluator.add_leaf(
        id="dr_soc2_certification",
        desc="Facility maintains current SOC 2 Type II certification",
        parent=d_node,
        critical=True
    )
    claim = f"The DR facility '{facility_label}' maintains a current SOC 2 Type II certification (not just Type I)."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=d_urls,
        additional_instruction="Confirm 'SOC 2 Type II' anywhere on the cited page(s)."
    )

    # ISO 27001 (critical)
    node = evaluator.add_leaf(
        id="dr_iso27001_certification",
        desc="Facility maintains current ISO 27001 certification",
        parent=d_node,
        critical=True
    )
    claim = f"The DR facility '{facility_label}' maintains a current ISO/IEC 27001 certification."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=d_urls,
        additional_instruction="Equivalent phrasing such as 'ISO 27001 certified' is acceptable."
    )

    # Carrier-neutral + MMR (critical)
    node = evaluator.add_leaf(
        id="dr_carrier_neutral",
        desc="Facility is carrier-neutral with meet-me-room (MMR) for multi-carrier interconnection",
        parent=d_node,
        critical=True
    )
    claim = f"The DR facility '{facility_label}' is carrier-neutral and has a meet-me room (MMR)."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=d_urls,
        additional_instruction="Explicit 'carrier-neutral' and 'MMR/meet-me room' (or clear equivalents) should be present."
    )

    # Power density >= 10 kW/rack (critical)
    node = evaluator.add_leaf(
        id="dr_power_density",
        desc="Facility supports minimum 10 kW per rack power density",
        parent=d_node,
        critical=True
    )
    claim = f"The DR facility '{facility_label}' supports at least 10 kW per rack (or per cabinet) power density."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=d_urls,
        additional_instruction="Higher stated minimum also satisfies the requirement."
    )

    # At least 3 carriers/NSPs (critical)
    node = evaluator.add_leaf(
        id="dr_network_access",
        desc="Facility provides direct access to at least 3 independent network service providers or telecommunications carriers",
        parent=d_node,
        critical=True
    )
    claim = f"The DR facility '{facility_label}' provides direct access to at least three independent telecommunications carriers or network service providers."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=d_urls,
        additional_instruction="Look for a carriers list or interconnection options implying >= 3 distinct providers."
    )

    # Physical cross-connects (critical)
    node = evaluator.add_leaf(
        id="dr_cross_connect",
        desc="Facility offers physical cross-connect services for direct point-to-point connections",
        parent=d_node,
        critical=True
    )
    claim = f"The DR facility '{facility_label}' offers physical cross-connect services (e.g., copper or fiber) for direct interconnection."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=d_urls,
        additional_instruction="Any explicit cross-connect mention should count."
    )

    # 24/7 onsite security (critical)
    node = evaluator.add_leaf(
        id="dr_physical_security",
        desc="Facility maintains 24/7 onsite security personnel",
        parent=d_node,
        critical=True
    )
    claim = f"The DR facility '{facility_label}' maintains 24/7 (24x7) on-site security personnel."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=d_urls,
        additional_instruction="Accept equivalent expressions such as 'round-the-clock staffed security'."
    )

    # Provider credibility (critical)
    node = evaluator.add_leaf(
        id="dr_provider_credibility",
        desc="Facility is operated by an established colocation provider with multiple data center locations nationwide",
        parent=d_node,
        critical=True
    )
    claim = f"The provider {provider_label} operates multiple colocation data center locations nationwide (U.S.) or has an extensive national footprint, indicating it is an established colocation provider."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=d_urls,
        additional_instruction="Evidence can be on the facility or provider pages indicating many U.S. locations. Global footprints are acceptable as stronger evidence."
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
    # Initialize evaluator (root as parallel; mark non-critical to avoid critical-child restriction)
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

    # Extract facilities data from the answer
    extracted: FacilitiesExtraction = await evaluator.extract(
        prompt=prompt_extract_facilities(),
        template_class=FacilitiesExtraction,
        extraction_name="facilities_extraction"
    )

    # Ensure presence of FacilityInfo placeholders to avoid attribute errors
    primary = extracted.primary or FacilityInfo()
    dr = extracted.dr or FacilityInfo()

    # Run verifications for primary and DR
    await build_primary_facility_checks(evaluator, root, primary)
    await build_dr_facility_checks(evaluator, root, primary, dr)

    # Return structured evaluation summary
    return evaluator.get_summary()