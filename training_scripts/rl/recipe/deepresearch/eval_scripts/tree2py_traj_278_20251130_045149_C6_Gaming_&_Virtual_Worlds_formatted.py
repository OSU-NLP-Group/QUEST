import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "esports_venue_oc_fortnite"
TASK_DESCRIPTION = """
Identify an esports venue located in Orange County, California that meets the following requirements for hosting a professional Fortnite tournament:

1. The venue must be physically located within Orange County, California, with a verifiable street address
2. The venue must have a minimum seating capacity of 300 spectators
3. The venue must have at least 15,000 square feet of total space
4. The venue must provide network connectivity with at least 25 Mbps download speed and 5 Mbps upload speed per gaming station (meeting industry standards for esports venues)
5. The venue's gaming stations must support Fortnite's minimum system requirements: at least 8GB RAM and Intel HD 4000 or equivalent graphics capability
6. A major gaming company headquarters must be located within 50 miles of the venue, either in Orange County or adjacent Los Angeles County

For your answer, provide:
- The name of the esports venue
- The complete physical address of the venue
- The seating capacity and square footage of the venue
- Confirmation that the venue meets the network connectivity requirements (either through explicit specifications or through its status as a professional esports facility)
- Confirmation that gaming stations meet Fortnite's minimum requirements
- The name and headquarters address of a nearby major gaming company
- The approximate distance between the venue and the gaming company headquarters

Include reference URLs that verify each piece of information.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PostalAddress(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    full_address: Optional[str] = None


class VenueExtraction(BaseModel):
    # Venue identity and address
    venue_name: Optional[str] = None
    venue_address: Optional[PostalAddress] = None
    venue_address_urls: List[str] = Field(default_factory=list)

    # Capacity and square footage
    seating_capacity: Optional[str] = None
    seating_capacity_urls: List[str] = Field(default_factory=list)

    total_space_sqft: Optional[str] = None
    total_space_urls: List[str] = Field(default_factory=list)

    # Network connectivity (per-station or equivalent evidence)
    network_download_mbps_per_station: Optional[str] = None
    network_upload_mbps_per_station: Optional[str] = None
    network_urls: List[str] = Field(default_factory=list)

    # Gaming station hardware
    hardware_ram_gb: Optional[str] = None
    hardware_graphics: Optional[str] = None
    hardware_urls: List[str] = Field(default_factory=list)

    # Nearby major gaming company
    company_name: Optional[str] = None
    company_hq_address: Optional[PostalAddress] = None
    company_hq_urls: List[str] = Field(default_factory=list)
    company_major_evidence_urls: List[str] = Field(default_factory=list)

    # Proximity distance
    approximate_distance_miles: Optional[str] = None
    distance_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_data() -> str:
    return """
    Extract the single esports venue proposed in the answer (use the first/main venue if multiple are mentioned).
    Return a JSON with the following fields. Only extract facts explicitly present in the answer; do not invent.

    1) venue_name: string (venue name)
    2) venue_address: object
       - street: string or null
       - city: string or null
       - state: string or null
       - zip_code: string or null
       - full_address: the full street address as one line if present; else null
    3) venue_address_urls: array of URL strings that support the address

    4) seating_capacity: string (as stated, e.g., "350", "300-400", "about 500") or null
    5) seating_capacity_urls: array of URL strings that support the capacity

    6) total_space_sqft: string (as stated, e.g., "20,000 sq ft", "15k", "15,000-18,000 sq ft") or null
    7) total_space_urls: array of URL strings that support the total space

    8) network_download_mbps_per_station: string (as stated, e.g., "25 Mbps per station", "1 Gbps total") or null
    9) network_upload_mbps_per_station: string (as stated) or null
    10) network_urls: array of URL strings that support the network capabilities (venue specs, provider, or professional esports facility references acceptable)

    11) hardware_ram_gb: string (as stated, e.g., "8GB", "16 GB") or null
    12) hardware_graphics: string (as stated, e.g., "Intel HD 4000 or better", "GTX 1660") or null
    13) hardware_urls: array of URL strings supporting hardware specs

    14) company_name: string (major gaming company) or null
    15) company_hq_address: object
        - street: string or null
        - city: string or null
        - state: string or null
        - zip_code: string or null
        - full_address: string or null
    16) company_hq_urls: array of URL strings supporting the HQ address/location
    17) company_major_evidence_urls: array of URL strings that show the company is a major/leading publisher/developer

    18) approximate_distance_miles: string as stated in the answer (e.g., "32 miles", "~25 mi") or null
    19) distance_urls: array of URL strings sufficient to verify the distance (e.g., a Google Maps directions link or a directions result page)

    Rules for URL extraction:
    - Only include URLs explicitly present in the answer text (plain links or markdown).
    - If a field is missing, set it to null (or an empty array for URL lists).
    """


# --------------------------------------------------------------------------- #
# Helper parsing utilities                                                    #
# --------------------------------------------------------------------------- #
_NUM_RE = re.compile(r"(?:(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?))|(\d+(?:\.\d+)?)(?:\s*\+)?", re.IGNORECASE)


def _clean_str(s: Optional[str]) -> str:
    return (s or "").strip()


def _has_any_url(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    for u in urls:
        if isinstance(u, str) and ("http://" in u or "https://" in u):
            return True
    return False


def parse_number_range(text: Optional[str]) -> Optional[Tuple[float, float]]:
    """
    Parse a numeric value or range from text.
    - Handles commas, "k" suffix, ranges like "300-400", "15k", "20,000", "300+"
    Returns (low, high). If single number, low==high.
    """
    if not text:
        return None
    t = text.lower().replace(",", " ").replace("~", " ").replace("approx", " ").replace("approximately", " ")
    t = re.sub(r"\s+", " ", t).strip()

    # Convert k suffix, e.g., "15k" -> "15000"
    def k_to_num(match):
        num = float(match.group(1))
        return str(int(num * 1000))

    t = re.sub(r"(\d+(?:\.\d+)?)\s*k\b", k_to_num, t)

    m = _NUM_RE.search(t)
    if not m:
        return None

    if m.group(1) and m.group(2):
        low = float(m.group(1))
        high = float(m.group(2))
        if low > high:
            low, high = high, low
        return low, high

    if m.group(3):
        val = float(m.group(3))
        return val, val

    return None


def parse_sqft(text: Optional[str]) -> Optional[Tuple[float, float]]:
    rng = parse_number_range(text)
    if rng is None:
        return None
    # If units indicate sqft already, just return. If mentions acres, convert? For simplicity, assume stated numbers are sqft.
    return rng


def parse_capacity(text: Optional[str]) -> Optional[Tuple[float, float]]:
    return parse_number_range(text)


def parse_speed_mbps(text: Optional[str]) -> Optional[Tuple[float, float]]:
    """
    Parse a speed value (Mbps or Gbps). Convert to Mbps.
    Return range (low, high) in Mbps if found; else None.
    """
    if not text:
        return None
    t = text.lower()
    rng = parse_number_range(t)
    if not rng:
        return None
    low, high = rng
    # Detect unit
    if "gbps" in t or "gbit" in t or "gigabit" in t or "g b p s" in t or "gbs" in t:
        low *= 1000.0
        high *= 1000.0
    # If unspecified unit, commonly Mbps in these contexts; leave as-is.
    return low, high


def parse_distance_miles(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    t = text.lower()
    rng = parse_number_range(t)
    if not rng:
        return None
    val = rng[0]  # take the lower bound as conservative
    # Convert if km
    if "km" in t:
        val = val * 0.621371
    return val


def is_complete_address(addr: Optional[PostalAddress]) -> bool:
    if not addr:
        return False
    return bool(_clean_str(addr.street) and _clean_str(addr.city) and _clean_str(addr.state) and _clean_str(addr.zip_code))


def build_full_address(addr: Optional[PostalAddress]) -> str:
    if not addr:
        return ""
    if addr.full_address:
        return addr.full_address
    parts = [p for p in [addr.street, addr.city, addr.state, addr.zip_code] if _clean_str(p)]
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root, data: VenueExtraction) -> None:
    # Top-level critical assessment node (parallel aggregation)
    top = evaluator.add_parallel(
        id="Venue_Suitability_Assessment",
        desc="Evaluate whether the response identifies a single esports venue in Orange County, CA that meets all stated constraints and provides the requested fields with verifying URLs.",
        parent=root,
        critical=True
    )

    # 1) Venue Identification and Location
    vil = evaluator.add_parallel(
        id="Venue_Identification_and_Location",
        desc="Venue is identified, has a verifiable address, and is located in Orange County, CA.",
        parent=top,
        critical=True
    )

    # Venue name provided (custom existence check)
    evaluator.add_custom_node(
        result=bool(_clean_str(data.venue_name)),
        id="Venue_Name_Provided",
        desc="The esports venue name is provided.",
        parent=vil,
        critical=True
    )

    # Full physical address provided (street, city, state, ZIP)
    evaluator.add_custom_node(
        result=is_complete_address(data.venue_address),
        id="Venue_Full_Physical_Address_Provided",
        desc="A complete venue street address (street, city, state, ZIP) is provided.",
        parent=vil,
        critical=True
    )

    # Address source URL(s) provided
    evaluator.add_custom_node(
        result=_has_any_url(data.venue_address_urls),
        id="Venue_Address_Source_URL_Provided",
        desc="At least one reference URL is provided that supports the venue’s physical address.",
        parent=vil,
        critical=True
    )

    # Address lies in Orange County, CA (verify with address sources)
    addr_claim = f"The venue at '{build_full_address(data.venue_address)}' is located in Orange County, California."
    vil_in_oc = evaluator.add_leaf(
        id="Venue_Located_In_Orange_County_CA",
        desc="The provided/cited venue location is within Orange County, California.",
        parent=vil,
        critical=True
    )
    await evaluator.verify(
        claim=addr_claim,
        node=vil_in_oc,
        sources=(data.venue_address_urls if _has_any_url(data.venue_address_urls) else None),
        additional_instruction="Accept if the address belongs to a city within Orange County or the page explicitly states 'Orange County' for the venue's location."
    )

    # 2) Capacity and Square Footage
    cap_sq = evaluator.add_parallel(
        id="Capacity_and_Square_Footage",
        desc="Venue meets minimum seating capacity and total space requirements, with sources.",
        parent=top,
        critical=True
    )

    # Seating capacity provided
    evaluator.add_custom_node(
        result=bool(_clean_str(data.seating_capacity)),
        id="Seating_Capacity_Value_Provided",
        desc="A seating capacity (or spectator capacity) value is provided.",
        parent=cap_sq,
        critical=True
    )

    # Seating capacity >= 300 (numeric check)
    cap_rng = parse_capacity(data.seating_capacity)
    cap_pass = (cap_rng is not None and cap_rng[0] >= 300.0)
    evaluator.add_custom_node(
        result=cap_pass,
        id="Seating_Capacity_At_Least_300",
        desc="The stated seating capacity is >= 300 spectators.",
        parent=cap_sq,
        critical=True
    )

    # Seating capacity supported by sources (verify)
    cap_src = evaluator.add_leaf(
        id="Seating_Capacity_Source_URL_Provided",
        desc="A reference URL is provided that supports the seating capacity value.",
        parent=cap_sq,
        critical=True
    )
    cap_claim = f"The venue has a seating/spectator capacity of approximately '{_clean_str(data.seating_capacity)}'."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_src,
        sources=(data.seating_capacity_urls if _has_any_url(data.seating_capacity_urls) else None),
        additional_instruction="Verify that the cited page(s) state the venue's seating or spectator capacity consistent with the given value (allow approximate phrasing)."
    )

    # Square footage provided
    evaluator.add_custom_node(
        result=bool(_clean_str(data.total_space_sqft)),
        id="Square_Footage_Value_Provided",
        desc="A total space / square footage value is provided.",
        parent=cap_sq,
        critical=True
    )

    # Total space >= 15,000 sqft (numeric check)
    sqft_rng = parse_sqft(data.total_space_sqft)
    sqft_pass = (sqft_rng is not None and sqft_rng[0] >= 15000.0)
    evaluator.add_custom_node(
        result=sqft_pass,
        id="Total_Space_At_Least_15000_SqFt",
        desc="The stated total space is >= 15,000 square feet.",
        parent=cap_sq,
        critical=True
    )

    # Square footage supported by sources (verify)
    sqft_src = evaluator.add_leaf(
        id="Square_Footage_Source_URL_Provided",
        desc="A reference URL is provided that supports the square footage value.",
        parent=cap_sq,
        critical=True
    )
    sqft_claim = f"The venue offers approximately '{_clean_str(data.total_space_sqft)}' of total space (in square feet)."
    await evaluator.verify(
        claim=sqft_claim,
        node=sqft_src,
        sources=(data.total_space_urls if _has_any_url(data.total_space_urls) else None),
        additional_instruction="Verify that the cited page(s) state the venue's total space/square footage consistent with the given value (allow approximate phrasing)."
    )

    # 3) Network Connectivity
    net = evaluator.add_parallel(
        id="Network_Connectivity",
        desc="Venue meets minimum per-station network speed constraints, with a supporting source.",
        parent=top,
        critical=True
    )

    # Ensure network sources exist (as a separate critical leaf before verification)
    evaluator.add_custom_node(
        result=_has_any_url(data.network_urls),
        id="Network_Speeds_Source_URL_Provided",
        desc="At least one reference URL is provided that supports the stated per-station network speed capability (either explicit venue specs or other acceptable evidence per the question).",
        parent=net,
        critical=True
    )

    # Download speed >= 25 Mbps per station (verify with sources)
    dl_leaf = evaluator.add_leaf(
        id="Download_Speed_Per_Station_At_Least_25Mbps",
        desc="The response states/claims per-station download speed meets or exceeds 25 Mbps.",
        parent=net,
        critical=True
    )
    dl_claim = "The venue provides at least 25 Mbps download speed per gaming station (either explicitly stated or implied by professional esports-grade networking, e.g., dedicated gigabit connectivity appropriately shared)."
    await evaluator.verify(
        claim=dl_claim,
        node=dl_leaf,
        sources=(data.network_urls if _has_any_url(data.network_urls) else None),
        additional_instruction="Accept if the evidence shows: (a) explicit per-station speed >=25 Mbps, or (b) a total capacity (e.g., gigabit fiber) that reasonably allows >=25 Mbps per station for typical station counts."
    )

    # Upload speed >= 5 Mbps per station (verify with sources)
    ul_leaf = evaluator.add_leaf(
        id="Upload_Speed_Per_Station_At_Least_5Mbps",
        desc="The response states/claims per-station upload speed meets or exceeds 5 Mbps.",
        parent=net,
        critical=True
    )
    ul_claim = "The venue provides at least 5 Mbps upload speed per gaming station (either explicitly stated or reasonably implied by professional esports-grade networking capacity)."
    await evaluator.verify(
        claim=ul_claim,
        node=ul_leaf,
        sources=(data.network_urls if _has_any_url(data.network_urls) else None),
        additional_instruction="Accept if the evidence shows: (a) explicit per-station upload >=5 Mbps, or (b) a total capacity that reasonably allows >=5 Mbps per station for typical station counts."
    )

    # 4) Gaming Station Hardware
    hw = evaluator.add_parallel(
        id="Gaming_Station_Hardware",
        desc="Venue gaming stations meet Fortnite minimum hardware requirements, with a supporting source.",
        parent=top,
        critical=True
    )

    # Hardware sources existence (critical)
    evaluator.add_custom_node(
        result=_has_any_url(data.hardware_urls),
        id="Hardware_Specs_Source_URL_Provided",
        desc="A reference URL is provided supporting the venue gaming station hardware claims/specs.",
        parent=hw,
        critical=True
    )

    # RAM >= 8GB (verify with sources)
    ram_leaf = evaluator.add_leaf(
        id="RAM_At_Least_8GB",
        desc="The response states/claims gaming stations have >= 8GB RAM.",
        parent=hw,
        critical=True
    )
    ram_claim = "The venue's gaming stations each have at least 8 GB of RAM."
    await evaluator.verify(
        claim=ram_claim,
        node=ram_leaf,
        sources=(data.hardware_urls if _has_any_url(data.hardware_urls) else None),
        additional_instruction="Accept if the cited specs indicate 8GB RAM or more (e.g., 16GB)."
    )

    # Graphics >= Intel HD 4000 equivalent (verify with sources)
    gpu_leaf = evaluator.add_leaf(
        id="Graphics_At_Least_Intel_HD_4000_Equivalent",
        desc="The response states/claims gaming stations have Intel HD 4000 or equivalent (or better) graphics capability.",
        parent=hw,
        critical=True
    )
    gpu_claim = "The venue's gaming stations have graphics capability at least equivalent to Intel HD 4000 or better (e.g., modern discrete GPUs easily exceeding this)."
    await evaluator.verify(
        claim=gpu_claim,
        node=gpu_leaf,
        sources=(data.hardware_urls if _has_any_url(data.hardware_urls) else None),
        additional_instruction="Accept if the cited specs list GPUs that are equal to or stronger than Intel HD 4000 (e.g., NVIDIA GTX/RTX, AMD Radeon)."
    )

    # 5) Nearby Major Gaming Company
    comp = evaluator.add_parallel(
        id="Nearby_Major_Gaming_Company",
        desc="A major gaming company HQ is identified, located in Orange County or Los Angeles County, with a verifiable HQ address.",
        parent=top,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(_clean_str(data.company_name)),
        id="Gaming_Company_Name_Provided",
        desc="A gaming company is named.",
        parent=comp,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_complete_address(data.company_hq_address),
        id="Gaming_Company_HQ_Address_Provided",
        desc="The gaming company headquarters address is provided.",
        parent=comp,
        critical=True
    )

    # HQ location is in OC or LA County (verify with HQ URLs)
    hq_in_county_leaf = evaluator.add_leaf(
        id="Gaming_Company_HQ_In_OC_or_LA_County",
        desc="The provided/cited HQ location is in Orange County or Los Angeles County.",
        parent=comp,
        critical=True
    )
    hq_addr_full = build_full_address(data.company_hq_address)
    hq_loc_claim = f"The headquarters of '{_clean_str(data.company_name)}' at '{hq_addr_full}' is located in Orange County or Los Angeles County, California."
    await evaluator.verify(
        claim=hq_loc_claim,
        node=hq_in_county_leaf,
        sources=(data.company_hq_urls if _has_any_url(data.company_hq_urls) else None),
        additional_instruction="Accept if the HQ is in any city that belongs to Orange County or Los Angeles County, or if the page explicitly states the county."
    )

    # Evidence that the company is "major" (verify)
    major_leaf = evaluator.add_leaf(
        id="Gaming_Company_Major_Evidence_Provided",
        desc="At least one cited source supports that the company is 'major' (e.g., describes it as a major/leading publisher/developer or otherwise clearly indicates large/well-known status).",
        parent=comp,
        critical=True
    )
    major_urls: List[str] = []
    if _has_any_url(data.company_major_evidence_urls):
        major_urls.extend(data.company_major_evidence_urls)
    elif _has_any_url(data.company_hq_urls):
        # fallback
        major_urls.extend(data.company_hq_urls)
    major_claim = f"'{_clean_str(data.company_name)}' is a major/leading video game developer or publisher (or otherwise widely recognized as a major company in gaming)."
    await evaluator.verify(
        claim=major_claim,
        node=major_leaf,
        sources=(major_urls if _has_any_url(major_urls) else None),
        additional_instruction="Accept if the source describes the company as major, leading, large, or otherwise clearly indicates prominent status (e.g., Wikipedia, corporate overview, notable press)."
    )

    # HQ source URL(s) provided (custom existence)
    evaluator.add_custom_node(
        result=_has_any_url(data.company_hq_urls),
        id="Gaming_Company_HQ_Source_URL_Provided",
        desc="At least one reference URL is provided that supports the HQ address/location.",
        parent=comp,
        critical=True
    )

    # 6) Venue-to-HQ Proximity
    prox = evaluator.add_parallel(
        id="Venue_to_HQ_Proximity",
        desc="Distance between the venue and the gaming company HQ is provided and is within 50 miles, with a verifiable basis.",
        parent=top,
        critical=True
    )

    # Approximate distance provided
    evaluator.add_custom_node(
        result=bool(_clean_str(data.approximate_distance_miles)),
        id="Approximate_Distance_Provided",
        desc="An approximate distance between the venue and the HQ is provided.",
        parent=prox,
        critical=True
    )

    # Distance <= 50 miles (numeric check)
    dist_val = parse_distance_miles(data.approximate_distance_miles)
    evaluator.add_custom_node(
        result=(dist_val is not None and dist_val <= 50.0),
        id="Distance_Within_50_Miles",
        desc="The provided distance is <= 50 miles.",
        parent=prox,
        critical=True
    )

    # Distance verification URL(s) provided (existence)
    evaluator.add_custom_node(
        result=_has_any_url(data.distance_urls),
        id="Distance_Verification_Source_URL_Provided",
        desc="Reference URL(s) are provided sufficient to verify the distance (e.g., a maps distance link OR both addresses with an explicit distance computation source).",
        parent=prox,
        critical=True
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
    Evaluate an answer for the Orange County esports venue suitability task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation; critical gating handled by top-level child
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

    # Extract structured data from the answer
    extracted: VenueExtraction = await evaluator.extract(
        prompt=prompt_extract_venue_data(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction"
    )

    # Build verification tree and run verifications
    await build_verification_tree(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()