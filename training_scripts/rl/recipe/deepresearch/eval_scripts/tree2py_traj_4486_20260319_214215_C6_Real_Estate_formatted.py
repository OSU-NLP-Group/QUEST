import asyncio
import logging
import math
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nashville_restaurant_properties"
TASK_DESCRIPTION = (
    "I am planning to open a restaurant in Nashville, Tennessee, and need to identify three suitable "
    "commercial properties for lease. Please find three different restaurant spaces that meet ALL of the following requirements:\n\n"
    "1. Location: The property must be located in one of these Nashville commercial districts: West End, Midtown, Downtown, or Cool Springs.\n"
    "2. Size: The space must be between 2,500 and 4,000 square feet of gross floor area.\n"
    "3. Zoning: The property must be zoned to permit restaurant use (acceptable zoning classifications include CL - Commercial Limited, or CN - Commercial Neighborhood).\n"
    "4. Parking: The property must either (a) provide adequate parking at a minimum of 1 parking space per 250 square feet of gross floor area, OR (b) be located within Nashville's Urban Zoning Overlay (UZO) where parking minimums are waived.\n"
    "5. Seating Capacity: The property must have sufficient dining area to accommodate at least 150 seated guests (15 square feet per person for dining areas → at least 2,250 square feet of usable dining space).\n"
    "6. Lease Rate: The annual lease rate must not exceed $45 per square foot per year.\n\n"
    "For each of the three properties, please provide: address, gross floor area, zoning classification, parking (spaces or UZO), estimated seating capacity or dining area basis, annual lease rate per SF, and URL references."
)

ALLOWED_DISTRICTS = {"west end", "midtown", "downtown", "cool springs"}
PARKING_SF_PER_SPACE = 250.0
DINING_SF_PER_PERSON = 15.0
MIN_SEATS = 150
MIN_DINING_SF = MIN_SEATS * DINING_SF_PER_PERSON
MAX_ANNUAL_RATE_PSF = 45.0


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PropertyItem(BaseModel):
    address: Optional[str] = None
    district: Optional[str] = None  # e.g., "Midtown", "Downtown", "West End", "Cool Springs"
    gross_sqft: Optional[str] = None  # keep as string; we will parse
    zoning: Optional[str] = None  # e.g., "CL", "CN"
    parking_spaces: Optional[str] = None  # number or ratio text like "4/1000"
    uzo_status: Optional[str] = None  # free text indicating UZO presence (e.g., "in UZO", "Urban Zoning Overlay")
    dining_area_sqft: Optional[str] = None  # string; we will parse to number if provided
    seating_capacity: Optional[str] = None  # number as string if provided
    lease_rate_psf_annual: Optional[str] = None  # e.g., "$35/SF/YR", "$3/SF/MO"
    urls: List[str] = Field(default_factory=list)  # one or more listing/source URLs for this property


class PropertiesExtraction(BaseModel):
    properties: List[PropertyItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_properties() -> str:
    return """
    Extract up to three distinct restaurant-ready commercial properties that the answer proposes.
    For each property, extract the following fields exactly as stated in the answer (use strings for numeric values):
    - address: the complete street address (if given)
    - district: the named commercial district within the Nashville area (e.g., "West End", "Midtown", "Downtown", "Cool Springs")
    - gross_sqft: the gross floor area of the space (e.g., "3,200 SF", "3,000-3,500 SF")
    - zoning: the zoning classification code (e.g., "CL", "CN")
    - parking_spaces: the provided parking information, either a number of spaces (e.g., "40 spaces") or a ratio (e.g., "4/1000")
    - uzo_status: text indicating whether the property is inside Nashville's Urban Zoning Overlay (e.g., "in UZO", "Urban Zoning Overlay", "outside UZO", or "unknown")
    - dining_area_sqft: the usable dining area square footage if provided (e.g., "2,400 SF"), else null
    - seating_capacity: the stated seating capacity if provided (e.g., "175 seats"), else null
    - lease_rate_psf_annual: the rent rate per square foot per year if given (e.g., "$35/SF/YR", "$3/SF/MO", "Negotiable")
    - urls: an array of one or more URLs that the answer cites for this property (listing pages or official sources)

    Return a JSON object with one field:
      - "properties": an array of at most three PropertyItem objects in the same order they appear in the answer.

    Rules:
    - Do not invent any data. If a field is not present, return null for that field (or empty array for urls).
    - Extract exactly the URLs that appear in the answer; include all that seem relevant for verifying the details.
    - If the answer provides more than three properties, only include the first three.
    """


# --------------------------------------------------------------------------- #
# Helper parsing utilities                                                    #
# --------------------------------------------------------------------------- #
def is_valid_url(u: Optional[str]) -> bool:
    if not u:
        return False
    try:
        p = urlparse(u.strip())
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False


def filter_valid_urls(urls: List[str]) -> List[str]:
    return [u for u in urls if is_valid_url(u)]


def parse_number(s: Optional[str]) -> Optional[float]:
    """Extract first numeric token from a string; handles thousand separators."""
    if not s:
        return None
    # Replace commas and find the first float-like number
    m = re.search(r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", s.replace("’", "").replace("–", "-"))
    if not m:
        return None
    num_str = m.group(0).replace(",", "")
    try:
        return float(num_str)
    except Exception:
        return None


def parse_sqft(s: Optional[str]) -> Optional[float]:
    """Attempt to parse a square footage value from text. If a range is given, use the first number."""
    return parse_number(s)


def parse_int(s: Optional[str]) -> Optional[int]:
    n = parse_number(s)
    return int(round(n)) if n is not None else None


def infer_parking_spaces(parking_text: Optional[str], gross_sf: Optional[float]) -> Optional[int]:
    """
    From a parking text like "40 spaces" or "4/1000", estimate number of spaces.
    - If explicit number present, return it.
    - If ratio present like "4/1000" (spaces per 1000 SF), compute and round up using gross_sf if known.
    """
    if not parking_text:
        return None

    # Direct integer present (e.g., "45 spaces")
    direct = re.search(r"(\d{1,4})\s*(?:spaces?|stalls?)", parking_text.lower())
    if direct:
        try:
            return int(direct.group(1))
        except Exception:
            pass

    # Ratio like "4/1000" or "4 per 1,000"
    ratio = re.search(r"(\d+(?:\.\d+)?)\s*(?:/|per\s*)1[, ]?000", parking_text.lower())
    if ratio and gross_sf:
        try:
            spaces = math.ceil((gross_sf / 1000.0) * float(ratio.group(1)))
            return int(spaces)
        except Exception:
            return None

    # Fallback: first number if any
    num = parse_int(parking_text)
    return num


def parse_uzo_bool(s: Optional[str]) -> Optional[bool]:
    if not s:
        return None
    text = s.strip().lower()
    positives = [
        "in uzo", "within uzo", "urban zoning overlay", "uzo", "inside uzo", "located in uzo", "parking minimums waived"
    ]
    negatives = ["outside uzo", "not in uzo", "no uzo"]
    if any(p in text for p in positives):
        return True
    if any(n in text for n in negatives):
        return False
    return None


def parse_annual_psf(rate_text: Optional[str]) -> Optional[float]:
    """
    Parse a lease rate string to annual $/SF if possible.
    Recognizes patterns:
      - $34/SF/YR, $34 / SF / YR, $34 per sf per year
      - $2.75/SF/MO (multiply by 12)
      - Ranges: use the higher value to be conservative for budget checks
    Returns None if cannot determine.
    """
    if not rate_text:
        return None
    txt = rate_text.lower().replace("per", "/").replace("year", "yr").replace("annually", "yr").replace("annual", "yr")
    txt = re.sub(r"\s+", "", txt)

    # Range yearly: $30-40/SF/YR
    m = re.search(r"\$?(\d+(?:\.\d+)?)\-?\$?(\d+(?:\.\d+)?)/s?q?f/yr", txt)
    if m:
        lo = float(m.group(1))
        hi = float(m.group(2))
        return max(lo, hi)

    # Single yearly
    m = re.search(r"\$?(\d+(?:\.\d+)?)/s?q?f/yr", txt)
    if m:
        return float(m.group(1))

    # Range monthly: $2.5-3.0/SF/MO
    m = re.search(r"\$?(\d+(?:\.\d+)?)\-?\$?(\d+(?:\.\d+)?)/s?q?f/mo", txt)
    if m:
        lo = float(m.group(1))
        hi = float(m.group(2))
        return max(lo, hi) * 12.0

    # Single monthly
    m = re.search(r"\$?(\d+(?:\.\d+)?)/s?q?f/mo", txt)
    if m:
        return float(m.group(1)) * 12.0

    # "per sf yr" written differently: $35/sf/yr could be captured above; handle "$35sfyr"
    m = re.search(r"\$?(\d+(?:\.\d+)?)[a-z]*/s?q?f[a-z]*/yr", txt)
    if m:
        return float(m.group(1))

    # "$35psfyr"
    m = re.search(r"\$?(\d+(?:\.\d+)?)psf/yr", txt)
    if m:
        return float(m.group(1))

    # "$3psf/mo"
    m = re.search(r"\$?(\d+(?:\.\d+)?)psf/mo", txt)
    if m:
        return float(m.group(1)) * 12.0

    return None


def normalized_district_name(d: Optional[str]) -> Optional[str]:
    if not d:
        return None
    return d.strip().lower()


# --------------------------------------------------------------------------- #
# Verification helpers per property                                           #
# --------------------------------------------------------------------------- #
async def verify_property(evaluator: Evaluator, parent_node, prop: PropertyItem, idx: int) -> None:
    """
    Build the verification subtree for a single property.
    """
    prop_num = idx + 1
    prop_node = evaluator.add_parallel(
        id=f"property_{prop_num}",
        desc=f"Property #{prop_num}: Restaurant space meets all requirements",
        parent=parent_node,
        critical=False  # Each property contributes partial credit independently
    )

    # Normalize/parse values used in calculations
    urls_all = filter_valid_urls(prop.urls or [])
    sqft_val = parse_sqft(prop.gross_sqft)
    dining_sf_val = parse_sqft(prop.dining_area_sqft)
    seating_val = parse_int(prop.seating_capacity)
    parking_spaces_est = infer_parking_spaces(prop.parking_spaces, sqft_val)
    in_uzo = parse_uzo_bool(prop.uzo_status)
    annual_psf = parse_annual_psf(prop.lease_rate_psf_annual)
    district_norm = normalized_district_name(prop.district)

    # ---------------- Identification (critical) ----------------
    ident_node = evaluator.add_parallel(
        id=f"property_{prop_num}_identification",
        desc=f"Property #{prop_num} address and listing source provided",
        parent=prop_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(prop.address and prop.address.strip()),
        id=f"property_{prop_num}_address",
        desc="Complete street address provided",
        parent=ident_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(urls_all) > 0,
        id=f"property_{prop_num}_listing_source",
        desc="URL reference to property listing or source",
        parent=ident_node,
        critical=True
    )

    # ---------------- Location (critical) ----------------
    loc_node = evaluator.add_parallel(
        id=f"property_{prop_num}_location",
        desc="Property location verification",
        parent=prop_node,
        critical=True
    )
    # District membership (simple check)
    loc_district_leaf = evaluator.add_leaf(
        id=f"property_{prop_num}_location_district",
        desc="Property is in West End, Midtown, Downtown, or Cool Springs",
        parent=loc_node,
        critical=True
    )
    district_claim = (
        f"The stated district '{prop.district}' is one of the allowed districts: West End, Midtown, Downtown, or Cool Springs."
    )
    await evaluator.verify(
        claim=district_claim,
        node=loc_district_leaf,
        additional_instruction="Accept common sub-neighborhood names if they are widely considered part of one of the allowed districts."
    )

    # Location reference via URL(s)
    loc_ref_leaf = evaluator.add_leaf(
        id=f"property_{prop_num}_location_reference",
        desc="URL reference confirming the location",
        parent=loc_node,
        critical=True
    )
    loc_ref_claim = (
        f"The property at '{prop.address}' is located in the '{prop.district}' area of the Nashville region "
        f"(i.e., one of West End, Midtown, Downtown, or Cool Springs)."
    )
    # We'll queue URL verifications and run in a batch below
    verifications: List[tuple] = []
    verifications.append((
        loc_ref_claim,
        urls_all,
        loc_ref_leaf,
        "Use the listing page or authoritative source on the URL to confirm the district/location area."
    ))

    # ---------------- Size (critical) ----------------
    size_node = evaluator.add_parallel(
        id=f"property_{prop_num}_size",
        desc="Property size verification",
        parent=prop_node,
        critical=True
    )
    size_range_leaf = evaluator.add_leaf(
        id=f"property_{prop_num}_size_range",
        desc="Property gross floor area is between 2,500 and 4,000 square feet",
        parent=size_node,
        critical=True
    )
    size_range_claim = (
        f"The property gross floor area is between 2,500 and 4,000 square feet. "
        f"The stated gross floor area is '{prop.gross_sqft}'."
    )
    await evaluator.verify(
        claim=size_range_claim,
        node=size_range_leaf,
        additional_instruction="If a range is stated (e.g., 3,000–3,500 SF), it's acceptable if any or typical suite size falls between 2,500 and 4,000 SF."
    )

    size_ref_leaf = evaluator.add_leaf(
        id=f"property_{prop_num}_size_reference",
        desc="URL reference confirming the square footage",
        parent=size_node,
        critical=True
    )
    size_ref_claim = (
        f"The listing/source page indicates the space size is approximately '{prop.gross_sqft}' (gross floor area)."
    )
    verifications.append((
        size_ref_claim,
        urls_all,
        size_ref_leaf,
        "Allow minor formatting differences and approximations (e.g., 3,000 SF vs 3,050 SF)."
    ))

    # ---------------- Zoning (critical) ----------------
    zoning_node = evaluator.add_parallel(
        id=f"property_{prop_num}_zoning",
        desc="Property zoning verification",
        parent=prop_node,
        critical=True
    )
    zoning_class_leaf = evaluator.add_leaf(
        id=f"property_{prop_num}_zoning_classification",
        desc="Property is zoned CL (Commercial Limited) or CN (Commercial Neighborhood), which permit restaurant uses",
        parent=zoning_node,
        critical=True
    )
    zoning_claim = (
        f"The zoning classification '{prop.zoning}' indicates the property is zoned CL or CN (acceptable for restaurant use)."
    )
    await evaluator.verify(
        claim=zoning_claim,
        node=zoning_class_leaf,
        additional_instruction="Only consider this correct if the stated zoning is exactly CL or CN (case-insensitive). Do not accept other zones."
    )

    zoning_ref_leaf = evaluator.add_leaf(
        id=f"property_{prop_num}_zoning_reference",
        desc="URL reference confirming zoning classification",
        parent=zoning_node,
        critical=True
    )
    zoning_ref_claim = f"The listing or cited source confirms the property's zoning classification is '{prop.zoning}'."
    verifications.append((
        zoning_ref_claim,
        urls_all,
        zoning_ref_leaf,
        "The source may be the listing, a parcel GIS record, or an official zoning map for the parcel."
    ))

    # ---------------- Parking (critical) ----------------
    parking_node = evaluator.add_parallel(
        id=f"property_{prop_num}_parking",
        desc="Parking requirement verification",
        parent=prop_node,
        critical=True
    )
    # Calculation/existence check (non-web; deterministic)
    required_spaces = None
    if sqft_val is not None:
        required_spaces = math.ceil(sqft_val / PARKING_SF_PER_SPACE)

    # Determine whether requirement is satisfied based on extracted numbers
    calc_satisfied = False
    if in_uzo is True:
        calc_satisfied = True
    elif in_uzo is False and required_spaces is not None and parking_spaces_est is not None:
        calc_satisfied = parking_spaces_est >= required_spaces
    elif in_uzo is None and required_spaces is not None and parking_spaces_est is not None:
        calc_satisfied = parking_spaces_est >= required_spaces
    else:
        calc_satisfied = False

    evaluator.add_custom_node(
        result=calc_satisfied,
        id=f"property_{prop_num}_parking_calculation",
        desc="Provides ≥ 1 parking space per 250 SF (or is within Urban Zoning Overlay where minimums are waived)",
        parent=parking_node,
        critical=True
    )

    parking_ref_leaf = evaluator.add_leaf(
        id=f"property_{prop_num}_parking_reference",
        desc="URL reference confirming parking availability or UZO status",
        parent=parking_node,
        critical=True
    )
    if required_spaces is not None:
        parking_ref_claim = (
            f"The listing/source confirms that the property either provides at least {required_spaces} parking spaces "
            f"for approximately {int(round(sqft_val or 0))} SF (1 per 250 SF) "
            f"or is located within Nashville's Urban Zoning Overlay (UZO) where parking minimums are waived."
        )
    else:
        parking_ref_claim = (
            "The listing/source confirms that the property either meets a minimum of 1 parking space per 250 SF of gross floor area "
            "or is located within Nashville's Urban Zoning Overlay (UZO) where parking minimums are waived."
        )
    verifications.append((
        parking_ref_claim,
        urls_all,
        parking_ref_leaf,
        "Look for explicit parking counts, ratios (e.g., 4/1000), or a clear statement about UZO coverage."
    ))

    # ---------------- Seating Capacity (critical) ----------------
    capacity_node = evaluator.add_parallel(
        id=f"property_{prop_num}_capacity",
        desc="Seating capacity verification",
        parent=prop_node,
        critical=True
    )

    # Compute feasibility: either stated seats >= 150 OR dining area >= 2,250 SF
    has_capacity = False
    if seating_val is not None and seating_val >= MIN_SEATS:
        has_capacity = True
    elif dining_sf_val is not None and dining_sf_val >= MIN_DINING_SF:
        has_capacity = True
    else:
        has_capacity = False

    evaluator.add_custom_node(
        result=has_capacity,
        id=f"property_{prop_num}_capacity_calculation",
        desc="Dining area supports ≥ 150 seated guests (≥ 2,250 SF at 15 SF/person) or stated capacity ≥ 150",
        parent=capacity_node,
        critical=True
    )

    capacity_ref_leaf = evaluator.add_leaf(
        id=f"property_{prop_num}_capacity_reference",
        desc="URL reference supporting capacity calculation or floor plan information",
        parent=capacity_node,
        critical=True
    )
    if dining_sf_val is not None:
        cap_ref_claim = (
            f"The listing/source indicates at least {int(round(dining_sf_val))} SF of dining area, "
            f"which is sufficient to seat at least 150 guests at 15 SF/person."
        )
    elif seating_val is not None:
        cap_ref_claim = (
            f"The listing/source indicates an estimated/stated seating capacity of at least {seating_val} seats, "
            f"which satisfies the minimum 150-seat requirement."
        )
    else:
        cap_ref_claim = (
            "The listing/source provides enough information (dining area square footage or explicit seating count) to support "
            "that the space can accommodate at least 150 seated guests (≥ 2,250 SF at 15 SF/person)."
        )
    verifications.append((
        cap_ref_claim,
        urls_all,
        capacity_ref_leaf,
        "Prefer explicit dining area or seating capacity on the page; floor plans or occupancy notes also acceptable."
    ))

    # ---------------- Lease Rate (critical) ----------------
    lease_node = evaluator.add_parallel(
        id=f"property_{prop_num}_lease_rate",
        desc="Lease rate budget verification",
        parent=prop_node,
        critical=True
    )

    lease_within_budget = (annual_psf is not None) and (annual_psf <= MAX_ANNUAL_RATE_PSF)
    evaluator.add_custom_node(
        result=lease_within_budget,
        id=f"property_{prop_num}_lease_amount",
        desc=f"Annual lease rate does not exceed ${MAX_ANNUAL_RATE_PSF:.0f} per SF per year",
        parent=lease_node,
        critical=True
    )

    lease_ref_leaf = evaluator.add_leaf(
        id=f"property_{prop_num}_lease_reference",
        desc="URL reference confirming the lease rate",
        parent=lease_node,
        critical=True
    )
    if annual_psf is not None:
        lease_ref_claim = (
            f"The listing/source shows a rate equivalent to approximately ${annual_psf:.2f} per SF per year, "
            f"which does not exceed ${MAX_ANNUAL_RATE_PSF:.0f}/SF/YR."
        )
    else:
        lease_ref_claim = (
            "The listing/source clearly states a lease rate per SF per year (or convertible monthly PSF), and it does not exceed $45/SF/YR."
        )
    verifications.append((
        lease_ref_claim,
        urls_all,
        lease_ref_leaf,
        "If the rate is monthly per SF, convert to annual by multiplying by 12; if a range is shown, use the higher value."
    ))

    # ---------------- Batch-verify all URL-backed leaves for this property ----------------
    if verifications:
        await evaluator.batch_verify(verifications)


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
    Evaluate an answer for the Nashville restaurant property selection task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Properties evaluated independently for partial credit
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

    # Extract up to 3 properties from the answer
    extracted: PropertiesExtraction = await evaluator.extract(
        prompt=prompt_extract_properties(),
        template_class=PropertiesExtraction,
        extraction_name="properties_extraction"
    )

    props = list(extracted.properties or [])
    # Only first 3, pad with empty items if fewer
    props = props[:3]
    while len(props) < 3:
        props.append(PropertyItem())

    # Optional: record task constraints for transparency
    evaluator.add_ground_truth({
        "allowed_districts": sorted(list(ALLOWED_DISTRICTS)),
        "size_range_sf": [2500, 4000],
        "zoning_allowed": ["CL", "CN"],
        "parking_min_ratio": "1 space per 250 SF or within UZO",
        "seating_min": 150,
        "dining_min_sf": MIN_DINING_SF,
        "lease_rate_max_psf_per_year": MAX_ANNUAL_RATE_PSF
    }, gt_type="constraints")

    # Build verification subtrees for the 3 properties
    tasks = []
    for i, prop in enumerate(props):
        tasks.append(verify_property(evaluator, root, prop, i))
    await asyncio.gather(*tasks)

    return evaluator.get_summary()