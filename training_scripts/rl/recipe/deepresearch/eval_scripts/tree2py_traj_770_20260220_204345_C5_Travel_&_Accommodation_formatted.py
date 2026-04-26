import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "bangor_trip_prep"
TASK_DESCRIPTION = (
    "I'm a U.S. resident planning a 14-day road trip from Bangor, Maine to visit the Grand Canyon and two other "
    "national parks in the southwestern United States. Before I depart, I need to gather several pieces of "
    "information to prepare for my trip:\n\n"
    "1. TSA PreCheck Enrollment: Provide the complete street address of a TSA PreCheck enrollment center in Bangor, "
    "Maine, along with the typical application fee range for new enrollments.\n\n"
    "2. Park-Sleep-Fly Hotel Packages: Find three different hotels near Bangor International Airport that offer "
    "park-sleep-fly packages. For each hotel, provide: the hotel name; confirmation that the package includes at "
    "least 14 days of parking, one night hotel stay, and airport shuttle service; a valid URL or reference for "
    "booking or package information.\n\n"
    "3. Cost Analysis: Calculate the cost of parking directly at Bangor International Airport's long-term lot for "
    "14 days, and compare this to the typical cost of a park-sleep-fly package to determine potential savings.\n\n"
    "4. National Park Pass Decision: Determine whether purchasing an America the Beautiful Annual Pass (for U.S. "
    "residents) would be more cost-effective than paying individual entrance fees if I'm visiting the Grand Canyon "
    "and two other national parks that charge $35 per private vehicle. Include: the Grand Canyon private vehicle "
    "entrance fee; the total cost for three parks at individual entrance fees; the cost of the America the Beautiful "
    "Annual Pass; the amount saved (or additional cost) by purchasing the annual pass."
)

# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _is_valid_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    return url.strip().lower().startswith(("http://", "https://"))

def _ensure_list(urls: Optional[List[str] | str]) -> List[str]:
    if urls is None:
        return []
    if isinstance(urls, str):
        return [urls]
    return [u for u in urls if isinstance(u, str)]

def _parse_money_to_float(text: Optional[str]) -> Optional[float]:
    """Extract the first monetary number (e.g., '80', '35.00') from a string."""
    if not text or not isinstance(text, str):
        return None
    # Accept inputs like "$80", "USD 80", "$78-$85" (we take the first number)
    nums = re.findall(r"(\d+(?:\.\d+)?)", text.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[0])
    except Exception:
        return None

def _compute_savings(airport_cost: Optional[float], psf_cost: Optional[float]) -> Optional[float]:
    if airport_cost is None or psf_cost is None:
        return None
    # Positive value => PSF is cheaper (savings), negative => additional cost
    return airport_cost - psf_cost

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class TSAExtraction(BaseModel):
    address: Optional[str] = None
    fee_range: Optional[str] = None
    center_urls: List[str] = Field(default_factory=list)
    fee_urls: List[str] = Field(default_factory=list)

class HotelPackageInfo(BaseModel):
    name: Optional[str] = None
    package_url: Optional[str] = None
    parking_days: Optional[str] = None
    includes_one_night: Optional[bool] = None
    includes_shuttle: Optional[bool] = None

class HotelsExtraction(BaseModel):
    hotels: List[HotelPackageInfo] = Field(default_factory=list)

class CostAnalysisExtraction(BaseModel):
    airport_parking_daily_rate: Optional[str] = None
    airport_parking_total_for_14_days: Optional[str] = None
    airport_parking_url: Optional[str] = None
    typical_psf_cost: Optional[str] = None
    psf_urls: List[str] = Field(default_factory=list)

class ParkPassExtraction(BaseModel):
    grand_canyon_fee: Optional[str] = None
    annual_pass_cost: Optional[str] = None
    three_parks_total: Optional[str] = None
    fees_urls: List[str] = Field(default_factory=list)
    annual_pass_url: Optional[str] = None

# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_tsa() -> str:
    return (
        "From the answer, extract TSA PreCheck enrollment information specific to Bangor, Maine.\n"
        "Return a JSON object with:\n"
        "- address: the complete street address of the TSA PreCheck enrollment center in Bangor, Maine\n"
        "- fee_range: the typical application fee range for new TSA PreCheck enrollments as stated in the answer (e.g., '$78–$85')\n"
        "- center_urls: an array of URLs provided that reference the enrollment center information\n"
        "- fee_urls: an array of URLs provided that reference the TSA PreCheck application fee information (if any)\n"
        "If any field is missing in the answer, set it to null (or empty array for URLs). Extract URLs exactly as provided."
    )

def prompt_extract_hotels() -> str:
    return (
        "Extract up to three hotels near Bangor International Airport that offer park-sleep-fly packages.\n"
        "Return a JSON object with a 'hotels' array (length up to 3). For each hotel, include:\n"
        "- name: the hotel name\n"
        "- package_url: the booking or package information URL provided in the answer\n"
        "- parking_days: the number of parking days included as stated (e.g., '14 days', 'up to 14', '7-14 days')\n"
        "- includes_one_night: boolean indicating if the package includes one night hotel stay (true/false)\n"
        "- includes_shuttle: boolean indicating if shuttle service to the airport is included (true/false)\n"
        "If the answer lists more than three hotels, only include the first three. If fewer, include what is available.\n"
        "If any field is missing for a hotel, set it to null."
    )

def prompt_extract_costs() -> str:
    return (
        "Extract the cost analysis details from the answer related to Bangor International Airport parking and "
        "park-sleep-fly packages.\n"
        "Return a JSON object including:\n"
        "- airport_parking_daily_rate: the quoted daily rate for the long-term lot at Bangor International Airport\n"
        "- airport_parking_total_for_14_days: the total cost for 14 days of long-term parking if provided\n"
        "- airport_parking_url: a URL provided that references the Bangor International Airport parking rates\n"
        "- typical_psf_cost: the typical cost of a park-sleep-fly package as stated in the answer\n"
        "- psf_urls: an array of URLs provided that reference the package pricing or examples\n"
        "If any field is missing in the answer, set it to null (or empty array for URLs)."
    )

def prompt_extract_park_pass() -> str:
    return (
        "Extract the national park pass and fee information mentioned in the answer.\n"
        "Return a JSON object including:\n"
        "- grand_canyon_fee: the private vehicle entrance fee for Grand Canyon National Park as stated\n"
        "- annual_pass_cost: the price of the America the Beautiful Annual Pass for U.S. residents\n"
        "- three_parks_total: the total cost for visiting three parks at $35 per park if provided; otherwise null\n"
        "- fees_urls: an array of URLs provided that reference national park entrance fees or Grand Canyon fees\n"
        "- annual_pass_url: a URL provided that references the annual pass pricing (if any)\n"
        "If any field is missing in the answer, set it to null (or empty array for URLs)."
    )

# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_tsa_section(evaluator: Evaluator, parent_node, tsa: TSAExtraction) -> None:
    tsa_node = evaluator.add_parallel(
        id="tsa_precheck_enrollment",
        desc="Provide TSA PreCheck enrollment information for Bangor, Maine",
        parent=parent_node,
        critical=False,
    )

    # Critical: Valid reference URL(s) exist
    has_ref = any(_is_valid_url(u) for u in tsa.center_urls)
    evaluator.add_custom_node(
        result=has_ref,
        id="tsa_reference_url",
        desc="Provide a valid URL reference to the TSA PreCheck enrollment center information",
        parent=tsa_node,
        critical=True,
    )

    # Critical: Verify the complete street address via the referenced URL(s)
    address_leaf = evaluator.add_leaf(
        id="enrollment_center_address",
        desc="Provide the complete street address of a TSA PreCheck enrollment center in Bangor, Maine",
        parent=tsa_node,
        critical=True,
    )
    addr_claim = f"The TSA PreCheck enrollment center in Bangor, Maine is located at: {tsa.address}."
    await evaluator.verify(
        claim=addr_claim,
        node=address_leaf,
        sources=tsa.center_urls,
        additional_instruction=(
            "Verify that the page explicitly shows the same complete street address in Bangor, ME (with street number, "
            "street name, city, state, and possibly ZIP). Minor formatting differences are acceptable."
        ),
    )

    # Non-critical: Fee range verification (grounded if fee_urls provided, otherwise still attempt with center_urls)
    fee_sources = tsa.fee_urls if tsa.fee_urls else tsa.center_urls
    fee_leaf = evaluator.add_leaf(
        id="application_fee_range",
        desc="Provide the fee range for TSA PreCheck application",
        parent=tsa_node,
        critical=False,
    )
    fee_claim = f"The typical TSA PreCheck new enrollment application fee range is: {tsa.fee_range}."
    await evaluator.verify(
        claim=fee_claim,
        node=fee_leaf,
        sources=fee_sources,
        additional_instruction=(
            "Confirm the fee range for a new TSA PreCheck enrollment (first-time application). If multiple providers "
            "have slightly different fees, a reasonable range still counts."
        ),
    )

async def verify_hotel_package(
    evaluator: Evaluator,
    parent_node,
    hotel: HotelPackageInfo,
    idx: int
) -> None:
    # Parent node for this hotel
    hotel_node = evaluator.add_parallel(
        id=f"hotel_package_{idx+1}",
        desc=f"{['First','Second','Third'][idx]} hotel with park-sleep-fly package near Bangor International Airport",
        parent=parent_node,
        critical=False,
    )

    # Gate: existence of essential info (name + URL) to avoid meaningless verification
    evaluator.add_custom_node(
        result=bool(hotel and hotel.name and hotel.package_url and _is_valid_url(hotel.package_url)),
        id=f"hotel_{idx+1}_existence",
        desc=f"Hotel #{idx+1} has a name and a valid package URL",
        parent=hotel_node,
        critical=True,
    )

    # Critical: Hotel name verified by the package URL
    name_leaf = evaluator.add_leaf(
        id=f"hotel_name_{idx+1}",
        desc=f"Provide the name of a hotel near Bangor International Airport offering park-sleep-fly packages",
        parent=hotel_node,
        critical=True,
    )
    name_claim = f"This page corresponds to the hotel named '{hotel.name}'."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=hotel.package_url,
        additional_instruction=(
            "Verify that the page clearly indicates the hotel's name. Minor differences in branding or suffixes "
            "(e.g., 'Hotel & Suites') are acceptable."
        ),
    )

    # Critical: Parking duration includes at least 14 days
    parking_leaf = evaluator.add_leaf(
        id=f"parking_duration_{idx+1}",
        desc="Verify the package includes at least 14 days of parking",
        parent=hotel_node,
        critical=True,
    )
    parking_claim = "This package includes at least 14 days of parking."
    await evaluator.verify(
        claim=parking_claim,
        node=parking_leaf,
        sources=hotel.package_url,
        additional_instruction=(
            "Look for language such as '14 days', 'two weeks', or 'up to 14 days'. Also accept packages explicitly "
            "mentioning 14 or more days of parking."
        ),
    )

    # Critical: Package components include one night stay, airport parking, and shuttle service
    components_leaf = evaluator.add_leaf(
        id=f"package_components_{idx+1}",
        desc="Verify the package includes one night hotel stay, airport parking, and shuttle service",
        parent=hotel_node,
        critical=True,
    )
    components_claim = (
        "The park-sleep-fly package includes: (1) one night hotel stay, (2) airport parking, and (3) shuttle service to the airport."
    )
    await evaluator.verify(
        claim=components_claim,
        node=components_leaf,
        sources=hotel.package_url,
        additional_instruction=(
            "Check the package description for each component. Synonyms like 'park and fly', 'stay and fly', 'airport "
            "shuttle', or 'transport to airport' are acceptable."
        ),
    )

    # Critical: Booking/reference URL validity
    evaluator.add_custom_node(
        result=_is_valid_url(hotel.package_url),
        id=f"booking_reference_{idx+1}",
        desc="Provide a valid URL or reference for booking or information about the package",
        parent=hotel_node,
        critical=True,
    )

async def verify_cost_comparison(
    evaluator: Evaluator,
    parent_node,
    cost: CostAnalysisExtraction
) -> None:
    # Sequential grouping for cost analysis
    cost_node = evaluator.add_sequential(
        id="cost_comparison_analysis",
        desc="Calculate and compare costs of park-sleep-fly packages versus direct airport parking",
        parent=parent_node,
        critical=False,
    )

    # Sub-node to separate daily-rate grounding and total-cost math
    parking_cost_node = evaluator.add_parallel(
        id="airport_parking_cost",
        desc="Calculate the cost of parking at Bangor International Airport long-term lot for 14 days using the correct daily rate",
        parent=cost_node,
        critical=False,
    )

    # Critical: Ensure airport parking URL exists
    evaluator.add_custom_node(
        result=_is_valid_url(cost.airport_parking_url),
        id="airport_parking_url_valid",
        desc="Airport parking information URL is provided and valid",
        parent=parking_cost_node,
        critical=True,
    )

    # Verify daily rate via URL
    daily_rate_leaf = evaluator.add_leaf(
        id="airport_daily_rate_supported",
        desc="Bangor International Airport long-term daily rate is correctly identified",
        parent=parking_cost_node,
        critical=True,
    )
    daily_rate_claim = f"The long-term parking daily rate at Bangor International Airport is {cost.airport_parking_daily_rate}."
    await evaluator.verify(
        claim=daily_rate_claim,
        node=daily_rate_leaf,
        sources=cost.airport_parking_url,
        additional_instruction=(
            "Confirm the long-term (not short-term) daily parking rate shown on the official airport or authoritative page."
        ),
    )

    # Verify total cost math for 14 days (simple check)
    total_cost_leaf = evaluator.add_leaf(
        id="airport_total_cost_14days_correct",
        desc="Total airport parking cost for 14 days is correctly calculated from the daily rate",
        parent=parking_cost_node,
        critical=False,
    )
    daily_rate_val = _parse_money_to_float(cost.airport_parking_daily_rate)
    provided_total_val = _parse_money_to_float(cost.airport_parking_total_for_14_days)
    computed_total_val = None
    if daily_rate_val is not None:
        computed_total_val = round(daily_rate_val * 14, 2)
    total_cost_claim = (
        f"Given a daily rate of {cost.airport_parking_daily_rate}, the correct total for 14 days is ${computed_total_val} "
        f"(rounded). The provided total is {cost.airport_parking_total_for_14_days}. These should match within reasonable rounding."
    )
    await evaluator.verify(
        claim=total_cost_claim,
        node=total_cost_leaf,
        sources=None,
        additional_instruction=(
            "Treat this as a pure arithmetic check: total = daily_rate × 14. Allow minor rounding differences (e.g., cents)."
        ),
    )

    # Compare to typical park-sleep-fly package cost (simple math check)
    comparison_leaf = evaluator.add_leaf(
        id="cost_comparison",
        desc="Compare the park-sleep-fly package costs against direct airport parking to determine potential savings",
        parent=cost_node,
        critical=False,
    )
    airport_total_for_calc = provided_total_val if provided_total_val is not None else computed_total_val
    psf_cost_val = _parse_money_to_float(cost.typical_psf_cost)
    savings_val = _compute_savings(airport_total_for_calc, psf_cost_val)
    comparison_claim = (
        f"Comparing airport parking total ${airport_total_for_calc} to a typical park-sleep-fly package cost of "
        f"{cost.typical_psf_cost}, the potential savings (airport minus package) is ${savings_val}."
    )
    await evaluator.verify(
        claim=comparison_claim,
        node=comparison_leaf,
        sources=None,
        additional_instruction=(
            "Pure arithmetic check only. If the savings value is positive, PSF is cheaper (savings). "
            "If negative, PSF is more expensive (additional cost)."
        ),
    )

async def verify_park_pass_analysis(
    evaluator: Evaluator,
    parent_node,
    pp: ParkPassExtraction
) -> None:
    pass_node = evaluator.add_sequential(
        id="national_park_pass_analysis",
        desc="Analyze whether the America the Beautiful Annual Pass is cost-effective for visiting Grand Canyon and two other similar national parks",
        parent=parent_node,
        critical=False,
    )

    # Critical: At least one valid park fees reference URL
    has_fee_ref = any(_is_valid_url(u) for u in pp.fees_urls)
    evaluator.add_custom_node(
        result=has_fee_ref,
        id="park_fees_reference_url",
        desc="Provide a valid URL reference to the national park entrance fees information",
        parent=pass_node,
        critical=True,
    )

    # Grand Canyon vehicle fee grounded by fees URLs
    gc_fee_leaf = evaluator.add_leaf(
        id="grand_canyon_vehicle_fee",
        desc="Provide the private vehicle entrance fee for Grand Canyon National Park",
        parent=pass_node,
        critical=False,
    )
    gc_fee_claim = "The private vehicle entrance fee for Grand Canyon National Park is $35."
    await evaluator.verify(
        claim=gc_fee_claim,
        node=gc_fee_leaf,
        sources=pp.fees_urls,
        additional_instruction=(
            "Confirm the private vehicle (standard non-commercial) fee listed on NPS or Grand Canyon official page."
        ),
    )

    # Three parks total cost (simple arithmetic check)
    total_three_leaf = evaluator.add_leaf(
        id="three_parks_total_cost",
        desc="Calculate the total entrance fees for visiting three national parks at $35 per park",
        parent=pass_node,
        critical=False,
    )
    provided_three_total = _parse_money_to_float(pp.three_parks_total)
    calc_three_total = 35.0 * 3
    three_claim = (
        f"Visiting 3 parks at $35 each costs ${calc_three_total} in total. "
        f"The provided total is {pp.three_parks_total}. These should match within reasonable rounding."
    )
    await evaluator.verify(
        claim=three_claim,
        node=total_three_leaf,
        sources=None,
        additional_instruction="Treat as a simple multiplication: 35 × 3 = 105.",
    )

    # Annual pass cost effectiveness combined check (simple math and decision)
    effectiveness_leaf = evaluator.add_leaf(
        id="annual_pass_cost_effectiveness",
        desc="Compare the America the Beautiful Annual Pass cost ($80 for U.S. residents) against paying individual entrance fees, and determine the savings",
        parent=pass_node,
        critical=False,
    )
    pass_cost_val = _parse_money_to_float(pp.annual_pass_cost)
    # Default to known typical scenario if missing: $80 annual pass vs $105 fees
    pass_cost_for_calc = pass_cost_val if pass_cost_val is not None else 80.0
    savings_vs_three = round(calc_three_total - pass_cost_for_calc, 2)
    effective_msg = "more cost-effective" if savings_vs_three > 0 else "not more cost-effective"
    effectiveness_claim = (
        f"The America the Beautiful Annual Pass costs {pp.annual_pass_cost if pp.annual_pass_cost else '$80'}. "
        f"Visiting three parks at $35 each totals ${calc_three_total}. The savings by purchasing the annual pass is "
        f"${savings_vs_three}, so the annual pass is {effective_msg} in this scenario."
    )
    await evaluator.verify(
        claim=effectiveness_claim,
        node=effectiveness_leaf,
        sources=None,
        additional_instruction=(
            "Pure arithmetic and logical check only. Compare $105 vs the annual pass price ($80 typical). "
            "Savings = 105 − pass_cost. If positive, annual pass is more cost-effective."
        ),
    )

# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel aggregation across sub-tasks
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

    # Extract all sections (can be parallelized)
    tsa_extraction_task = evaluator.extract(
        prompt=prompt_extract_tsa(),
        template_class=TSAExtraction,
        extraction_name="tsa_precheck_enrollment",
    )
    hotels_extraction_task = evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_psf",
    )
    cost_extraction_task = evaluator.extract(
        prompt=prompt_extract_costs(),
        template_class=CostAnalysisExtraction,
        extraction_name="cost_analysis",
    )
    park_pass_extraction_task = evaluator.extract(
        prompt=prompt_extract_park_pass(),
        template_class=ParkPassExtraction,
        extraction_name="park_pass",
    )

    tsa_extraction, hotels_extraction, cost_extraction, park_pass_extraction = await asyncio.gather(
        tsa_extraction_task, hotels_extraction_task, cost_extraction_task, park_pass_extraction_task
    )

    # TSA section verification
    await verify_tsa_section(evaluator, root, tsa_extraction)

    # Hotel packages verification (ensure exactly 3 slots)
    hotels_list = hotels_extraction.hotels[:3]
    while len(hotels_list) < 3:
        hotels_list.append(HotelPackageInfo())

    for i, hotel in enumerate(hotels_list):
        await verify_hotel_package(evaluator, root, hotel, i)

    # Cost comparison analysis verification
    await verify_cost_comparison(evaluator, root, cost_extraction)

    # National park pass analysis verification
    await verify_park_pass_analysis(evaluator, root, park_pass_extraction)

    return evaluator.get_summary()