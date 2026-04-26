import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "atl_pharmacy_shopping_coordination"
TASK_DESCRIPTION = (
    "For a resident of the Atlanta, Georgia metropolitan area coordinating their pharmacy and shopping services, provide the following information:\n\n"
    "1. The complete street address of a CVS pharmacy location in Atlanta, Georgia that operates 24 hours per day and offers drive-thru pharmacy service.\n\n"
    "2. The complete street address of a Publix supermarket in the Atlanta metropolitan area that has an in-store pharmacy, and provide the specific pharmacy operating hours on Sundays.\n\n"
    "3. The annual membership cost for a Costco Executive membership and the cashback reward percentage with its annual maximum limit for purchases.\n\n"
    "4. Which of the following three retail chains were closed on Thanksgiving Day (Thursday, November 28, 2024) in the Atlanta area: CVS Pharmacy, Publix Super Markets, and Costco Wholesale."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CVSExtraction(BaseModel):
    address: Optional[str] = None
    address_url: Optional[str] = None
    is_24_hour: Optional[str] = None
    hours_url: Optional[str] = None
    has_drive_thru: Optional[str] = None
    drive_thru_url: Optional[str] = None


class PublixExtraction(BaseModel):
    address: Optional[str] = None
    address_url: Optional[str] = None
    has_pharmacy: Optional[str] = None
    pharmacy_url: Optional[str] = None
    sunday_open_time: Optional[str] = None
    sunday_close_time: Optional[str] = None
    sunday_hours_url: Optional[str] = None


class CostcoExtraction(BaseModel):
    executive_cost: Optional[str] = None
    cost_url: Optional[str] = None
    cashback_percentage: Optional[str] = None
    cashback_percentage_url: Optional[str] = None
    cashback_maximum: Optional[str] = None
    cashback_maximum_url: Optional[str] = None


class ThanksgivingExtraction(BaseModel):
    cvs_status: Optional[str] = None
    cvs_status_url: Optional[str] = None
    publix_status: Optional[str] = None
    publix_status_url: Optional[str] = None
    costco_status: Optional[str] = None
    costco_status_url: Optional[str] = None


class FullExtraction(BaseModel):
    cvs: Optional[CVSExtraction] = None
    publix: Optional[PublixExtraction] = None
    costco: Optional[CostcoExtraction] = None
    thanksgiving: Optional[ThanksgivingExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_full() -> str:
    return """
Extract the following pieces of information from the answer. If multiple candidates are provided for a category, choose the first suitable one. If a field is not present, return null.

1) CVS pharmacy (must be in Atlanta, GA city; operates 24 hours; has drive‑thru):
- cvs.address: The complete street address string as written (include street number, street name, city, state, and ZIP if present).
- cvs.address_url: A URL that references this specific CVS store/location and shows the address.
- cvs.is_24_hour: The stated 24-hour status as text (e.g., "24 hours", "24/7", "open 24 hours", "yes", "no").
- cvs.hours_url: A URL (preferably CVS official or a reputable directory) that shows the pharmacy hours.
- cvs.has_drive_thru: The drive‑thru availability as text (e.g., "drive‑thru", "yes", "no").
- cvs.drive_thru_url: A URL that shows drive‑thru availability for this location (can be same as address or hours page if it states drive‑thru).

2) Publix supermarket (must be in the Atlanta metropolitan area; has in‑store pharmacy; provide Sunday pharmacy hours):
- publix.address: The complete street address string as written (with city, state, ZIP if present).
- publix.address_url: A URL that references this specific Publix store/location and shows the address.
- publix.has_pharmacy: The in‑store pharmacy existence as text (e.g., "pharmacy", "has pharmacy", "yes", "no").
- publix.pharmacy_url: A URL that shows the pharmacy department exists at this store (can be the store page).
- publix.sunday_open_time: The Sunday pharmacy opening time string (e.g., "9:00 AM").
- publix.sunday_close_time: The Sunday pharmacy closing time string (e.g., "7:00 PM").
- publix.sunday_hours_url: A URL that shows Sunday pharmacy hours (can be the store page if it shows pharmacy hours).

3) Costco Executive membership information:
- costco.executive_cost: The annual cost string for Executive membership (e.g., "$120").
- costco.cost_url: A Costco official URL that shows the Executive membership price.
- costco.cashback_percentage: The cashback/2% Reward percentage string for purchases (e.g., "2%").
- costco.cashback_percentage_url: A Costco official URL that states the cashback percentage.
- costco.cashback_maximum: The maximum annual Reward/limit string (e.g., "$1,000").
- costco.cashback_maximum_url: A Costco official URL that states the annual maximum.

4) Thanksgiving Day 2024 (Nov 28, 2024) closure/open status in Atlanta area:
- thanksgiving.cvs_status: "open" or "closed" for CVS Pharmacy as stated in the answer.
- thanksgiving.cvs_status_url: A URL that supports the CVS Thanksgiving 2024 status.
- thanksgiving.publix_status: "open" or "closed" for Publix Super Markets as stated.
- thanksgiving.publix_status_url: A URL that supports the Publix Thanksgiving 2024 status.
- thanksgiving.costco_status: "open" or "closed" for Costco Wholesale as stated.
- thanksgiving.costco_status_url: A URL that supports the Costco Thanksgiving 2024 status.
"""


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def is_valid_http_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return url.strip().lower().startswith("http://") or url.strip().lower().startswith("https://")


def has_complete_address_format(addr: Optional[str]) -> bool:
    if not addr:
        return False
    s = addr.strip()
    # Check has at least one digit (street number), state (GA or Georgia), and ZIP
    has_digit = any(ch.isdigit() for ch in s)
    has_state = (" GA" in s.upper()) or (" GEORGIA" in s.upper())
    has_zip = re.search(r"\b\d{5}(?:-\d{4})?\b", s) is not None
    # Minimal heuristics: digit + state + zip
    return bool(has_digit and has_state and has_zip)


def extract_city_from_address(addr: Optional[str]) -> Optional[str]:
    if not addr:
        return None
    # Try patterns like "..., City, GA 303xx" or "..., City, Georgia ..."
    m = re.search(r",\s*([A-Za-z\.\- ]+),\s*GA\b", addr, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r",\s*([A-Za-z\.\- ]+),\s*Georgia\b", addr, flags=re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return None


def normalize_open_closed(x: Optional[str]) -> Optional[str]:
    if not x:
        return None
    t = x.strip().lower()
    if "closed" in t:
        return "closed"
    if "open" in t:
        return "open"
    # If ambiguous, return original lower string
    return t


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_cvs_subtree(evaluator: Evaluator, root_node, cvs: Optional[CVSExtraction]) -> None:
    cvs_node = evaluator.add_parallel(
        id="CVS_24_Hour_Location",
        desc="Provide information about a 24-hour CVS pharmacy with drive-thru service in Atlanta",
        parent=root_node,
        critical=False
    )

    addr = cvs.address if cvs else None
    addr_url = cvs.address_url if cvs else None
    hours_url = cvs.hours_url if cvs else None
    drive_url = cvs.drive_thru_url if cvs else None

    # CVS_Address (critical)
    cvs_addr_parent = evaluator.add_parallel(
        id="CVS_Address",
        desc="Provide the complete street address of the CVS pharmacy location",
        parent=cvs_node,
        critical=True
    )

    # Address format check (custom)
    evaluator.add_custom_node(
        result=has_complete_address_format(addr),
        id="CVS_Address_Format",
        desc="The address includes street number, street name, city, state, and ZIP code",
        parent=cvs_addr_parent,
        critical=True
    )

    # Address reference presence (custom)
    evaluator.add_custom_node(
        result=is_valid_http_url(addr_url),
        id="CVS_Address_Reference",
        desc="The address can be verified through a reference URL from CVS or a reputable pharmacy directory",
        parent=cvs_addr_parent,
        critical=True
    )

    # Address in Atlanta (verify by URL)
    addr_in_atl_leaf = evaluator.add_leaf(
        id="CVS_Address_In_Atlanta",
        desc="The address is verified to be located within Atlanta, Georgia city limits or the specified geographic area",
        parent=cvs_addr_parent,
        critical=True
    )
    claim_addr_atl = f"The store's address is located in Atlanta, Georgia." if not addr else f"The store's address '{addr}' is located in Atlanta, Georgia."
    await evaluator.verify(
        claim=claim_addr_atl,
        node=addr_in_atl_leaf,
        sources=addr_url,
        additional_instruction="Verify that the store page clearly shows the city as 'Atlanta, GA' (or 'Atlanta, Georgia'). If it is not Atlanta city specifically, consider it not supported."
    )

    # CVS_24_Hour_Operation (critical)
    cvs_24_parent = evaluator.add_parallel(
        id="CVS_24_Hour_Operation",
        desc="Verify the CVS pharmacy operates 24 hours per day",
        parent=cvs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_valid_http_url(hours_url),
        id="CVS_24_Hour_Reference",
        desc="The 24-hour operation is verified through a reference URL from CVS or a reputable source",
        parent=cvs_24_parent,
        critical=True
    )

    cvs_24_leaf = evaluator.add_leaf(
        id="CVS_24_Hour_Confirmation",
        desc="The pharmacy's operating hours are explicitly stated and indicate 24-hour or continuous daily operation",
        parent=cvs_24_parent,
        critical=True
    )
    await evaluator.verify(
        claim="This CVS pharmacy operates 24 hours per day (24/7).",
        node=cvs_24_leaf,
        sources=hours_url,
        additional_instruction="Confirm that the page explicitly indicates 24-hour operation for the pharmacy. Look for 'Open 24 hours', '24/7', or equivalent phrasing."
    )

    # CVS_Drive_Thru_Service (critical)
    cvs_drive_parent = evaluator.add_parallel(
        id="CVS_Drive_Thru_Service",
        desc="Verify the CVS pharmacy offers drive-thru service",
        parent=cvs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_valid_http_url(drive_url),
        id="CVS_Drive_Thru_Reference",
        desc="The drive-thru service availability is verified through a reference URL",
        parent=cvs_drive_parent,
        critical=True
    )

    cvs_drive_leaf = evaluator.add_leaf(
        id="CVS_Drive_Thru_Confirmation",
        desc="The pharmacy is confirmed to have drive-thru pharmacy service available",
        parent=cvs_drive_parent,
        critical=True
    )
    await evaluator.verify(
        claim="This CVS pharmacy offers drive-thru pharmacy service.",
        node=cvs_drive_leaf,
        sources=drive_url,
        additional_instruction="Look for 'drive-thru pharmacy' or similar wording on the store page. Accept reasonable variants (e.g., 'Drive-Thru Pharmacy available')."
    )


async def build_publix_subtree(evaluator: Evaluator, root_node, publix: Optional[PublixExtraction]) -> None:
    publix_node = evaluator.add_parallel(
        id="Publix_Location_With_Sunday_Pharmacy",
        desc="Provide information about a Publix supermarket with in-store pharmacy and Sunday hours",
        parent=root_node,
        critical=False
    )

    addr = publix.address if publix else None
    addr_url = publix.address_url if publix else None
    pharm_url = publix.pharmacy_url if publix else None
    sunday_url = publix.sunday_hours_url if publix else None
    sunday_open = publix.sunday_open_time if publix else None
    sunday_close = publix.sunday_close_time if publix else None
    city = extract_city_from_address(addr)

    # Publix_Address (critical)
    publix_addr_parent = evaluator.add_parallel(
        id="Publix_Address",
        desc="Provide the complete street address of the Publix supermarket",
        parent=publix_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_complete_address_format(addr),
        id="Publix_Address_Format",
        desc="The address includes street number, street name, city, state, and ZIP code",
        parent=publix_addr_parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_valid_http_url(addr_url),
        id="Publix_Address_Reference",
        desc="The address can be verified through a reference URL from Publix or a reputable directory",
        parent=publix_addr_parent,
        critical=True
    )

    publix_in_metro_leaf = evaluator.add_leaf(
        id="Publix_Address_In_Metro",
        desc="The address is verified to be located within the Atlanta metropolitan area",
        parent=publix_addr_parent,
        critical=True
    )
    metro_claim = (
        f"The store is located in {city}, GA, which is part of the Atlanta metropolitan area."
        if city else
        "The store is located within the Atlanta metropolitan area."
    )
    await evaluator.verify(
        claim=metro_claim,
        node=publix_in_metro_leaf,
        sources=addr_url,
        additional_instruction=(
            "Use the city shown on the store page. Consider it within the Atlanta metro area if it is a commonly recognized suburb/city in the region "
            "(e.g., Sandy Springs, Decatur, Brookhaven, Marietta, Smyrna, Roswell, Alpharetta, Johns Creek, Duluth, Norcross, Tucker, Stone Mountain, etc.). "
            "If the location is clearly outside the Atlanta area, do not support the claim."
        )
    )

    # Publix_In_Store_Pharmacy (critical)
    publix_pharm_parent = evaluator.add_parallel(
        id="Publix_In_Store_Pharmacy",
        desc="Verify the Publix location has an in-store pharmacy",
        parent=publix_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_valid_http_url(pharm_url),
        id="Publix_Pharmacy_Reference",
        desc="The pharmacy existence is verified through a reference URL",
        parent=publix_pharm_parent,
        critical=True
    )

    publix_pharm_leaf = evaluator.add_leaf(
        id="Publix_Pharmacy_Existence",
        desc="The Publix store is confirmed to have a pharmacy department inside the supermarket",
        parent=publix_pharm_parent,
        critical=True
    )
    await evaluator.verify(
        claim="This Publix location has an in-store pharmacy.",
        node=publix_pharm_leaf,
        sources=pharm_url,
        additional_instruction="Confirm that the store page indicates an in-store pharmacy (e.g., 'Pharmacy' section, 'In-Store Pharmacy')."
    )

    # Publix_Sunday_Hours (critical)
    publix_sun_parent = evaluator.add_parallel(
        id="Publix_Sunday_Hours",
        desc="Provide the specific pharmacy operating hours on Sundays",
        parent=publix_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_valid_http_url(sunday_url),
        id="Publix_Sunday_Hours_Reference",
        desc="The Sunday pharmacy hours are verified through a reference URL from Publix or a reputable source",
        parent=publix_sun_parent,
        critical=True
    )

    open_leaf = evaluator.add_leaf(
        id="Publix_Sunday_Opening_Time",
        desc="The Sunday opening time for the pharmacy is clearly stated",
        parent=publix_sun_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"On Sundays, the in-store pharmacy opens at {sunday_open}." if sunday_open else "On Sundays, the in-store pharmacy opening time is as specified.",
        node=open_leaf,
        sources=sunday_url,
        additional_instruction="Match the Sunday pharmacy opening time. Allow formatting variations (e.g., '9 AM' vs '9:00 AM')."
    )

    close_leaf = evaluator.add_leaf(
        id="Publix_Sunday_Closing_Time",
        desc="The Sunday closing time for the pharmacy is clearly stated",
        parent=publix_sun_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"On Sundays, the in-store pharmacy closes at {sunday_close}." if sunday_close else "On Sundays, the in-store pharmacy closing time is as specified.",
        node=close_leaf,
        sources=sunday_url,
        additional_instruction="Match the Sunday pharmacy closing time. Allow formatting variations (e.g., '7 PM' vs '7:00 PM')."
    )


async def build_costco_subtree(evaluator: Evaluator, root_node, costco: Optional[CostcoExtraction]) -> None:
    costco_node = evaluator.add_parallel(
        id="Costco_Executive_Membership_Information",
        desc="Provide accurate information about Costco Executive membership costs and benefits",
        parent=root_node,
        critical=False
    )

    # Annual cost (critical)
    cost_parent = evaluator.add_parallel(
        id="Costco_Membership_Cost",
        desc="Provide the annual cost of Costco Executive membership",
        parent=costco_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_valid_http_url(costco.cost_url if costco else None),
        id="Costco_Cost_Reference",
        desc="The membership cost is verified through a reference URL from Costco's official website",
        parent=cost_parent,
        critical=True
    )

    cost_leaf = evaluator.add_leaf(
        id="Costco_Cost_Amount",
        desc="The annual membership cost is clearly stated with a specific dollar amount",
        parent=cost_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The annual cost of a Costco Executive membership is {costco.executive_cost}." if costco and costco.executive_cost else "The annual cost of a Costco Executive membership is as specified.",
        node=cost_leaf,
        sources=(costco.cost_url if costco else None),
        additional_instruction="Verify the Executive membership price on an official Costco page. Accept reasonable currency format variations (e.g., $120 vs $120.00)."
    )

    # Cashback percentage (critical)
    pct_parent = evaluator.add_parallel(
        id="Costco_Cashback_Percentage",
        desc="Provide the cashback reward percentage for Executive members",
        parent=costco_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_valid_http_url(costco.cashback_percentage_url if costco else None),
        id="Costco_Percentage_Reference",
        desc="The cashback percentage is verified through a reference URL from Costco's official website",
        parent=pct_parent,
        critical=True
    )

    pct_leaf = evaluator.add_leaf(
        id="Costco_Percentage_Amount",
        desc="The cashback percentage on qualified purchases is clearly stated",
        parent=pct_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"Costco Executive members receive {costco.cashback_percentage} Reward on qualified purchases." if costco and costco.cashback_percentage else "Costco Executive members receive the stated percentage Reward on qualified purchases.",
        node=pct_leaf,
        sources=(costco.cashback_percentage_url if costco else None),
        additional_instruction="Verify the percent (e.g., 2%) as shown on an official Costco page. Allow reasonable symbols and spacing."
    )

    # Cashback maximum (critical)
    max_parent = evaluator.add_parallel(
        id="Costco_Cashback_Maximum",
        desc="Provide the annual maximum limit for cashback rewards",
        parent=costco_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_valid_http_url(costco.cashback_maximum_url if costco else None),
        id="Costco_Maximum_Reference",
        desc="The cashback maximum is verified through a reference URL from Costco's official website",
        parent=max_parent,
        critical=True
    )

    max_leaf = evaluator.add_leaf(
        id="Costco_Maximum_Amount",
        desc="The annual maximum cashback limit is clearly stated with a specific dollar amount",
        parent=max_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The maximum annual 2% Reward for Costco Executive members is {costco.cashback_maximum}." if costco and costco.cashback_maximum else "The maximum annual 2% Reward for Costco Executive members is as specified.",
        node=max_leaf,
        sources=(costco.cashback_maximum_url if costco else None),
        additional_instruction="Verify the annual maximum amount (e.g., '$1,000') on an official Costco page. Accept reasonable formatting variants like '1000' vs '$1,000'."
    )


async def build_thanksgiving_subtree(evaluator: Evaluator, root_node, tg: Optional[ThanksgivingExtraction]) -> None:
    tg_node = evaluator.add_parallel(
        id="Thanksgiving_2024_Closure_Status",
        desc="Identify which of the three retail chains were closed on Thanksgiving Day 2024 in Atlanta",
        parent=root_node,
        critical=False
    )

    # CVS
    cvs_parent = evaluator.add_parallel(
        id="CVS_Thanksgiving_Status",
        desc="Verify whether CVS Pharmacy was open or closed on Thanksgiving Day 2024",
        parent=tg_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=is_valid_http_url(tg.cvs_status_url if tg else None),
        id="CVS_Status_Reference",
        desc="The Thanksgiving Day 2024 status for CVS is verified through a reference URL",
        parent=cvs_parent,
        critical=True
    )
    cvs_leaf = evaluator.add_leaf(
        id="CVS_Status_Determination",
        desc="The answer provides a clear statement about CVS Pharmacy's operational status on Thanksgiving Day 2024",
        parent=cvs_parent,
        critical=True
    )
    cvs_status = normalize_open_closed(tg.cvs_status if tg else None) if tg else None
    cvs_claim = (
        f"CVS Pharmacy was {cvs_status} on Thanksgiving Day 2024 (Thursday, November 28, 2024) in the Atlanta area."
        if cvs_status else
        "CVS Pharmacy's Thanksgiving Day 2024 operational status in the Atlanta area is as specified."
    )
    await evaluator.verify(
        claim=cvs_claim,
        node=cvs_leaf,
        sources=(tg.cvs_status_url if tg else None),
        additional_instruction="Verify whether CVS Pharmacy was open or closed on Nov 28, 2024. The page should clearly indicate the Thanksgiving 2024 status or announcement."
    )

    # Publix
    publix_parent = evaluator.add_parallel(
        id="Publix_Thanksgiving_Status",
        desc="Verify whether Publix Super Markets was open or closed on Thanksgiving Day 2024",
        parent=tg_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=is_valid_http_url(tg.publix_status_url if tg else None),
        id="Publix_Status_Reference",
        desc="The Thanksgiving Day 2024 closure status for Publix is verified through a reference URL",
        parent=publix_parent,
        critical=True
    )
    publix_leaf = evaluator.add_leaf(
        id="Publix_Status_Determination",
        desc="The answer provides a clear statement about Publix Super Markets' operational status on Thanksgiving Day 2024",
        parent=publix_parent,
        critical=True
    )
    publix_status = normalize_open_closed(tg.publix_status if tg else None) if tg else None
    publix_claim = (
        f"Publix Super Markets was {publix_status} on Thanksgiving Day 2024 (Thursday, November 28, 2024) in the Atlanta area."
        if publix_status else
        "Publix Super Markets' Thanksgiving Day 2024 operational status in the Atlanta area is as specified."
    )
    await evaluator.verify(
        claim=publix_claim,
        node=publix_leaf,
        sources=(tg.publix_status_url if tg else None),
        additional_instruction="Verify whether Publix was open or closed on Nov 28, 2024. Prefer official announcements or credible local news/hours pages."
    )

    # Costco
    costco_parent = evaluator.add_parallel(
        id="Costco_Thanksgiving_Status",
        desc="Verify whether Costco Wholesale was open or closed on Thanksgiving Day 2024",
        parent=tg_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=is_valid_http_url(tg.costco_status_url if tg else None),
        id="Costco_Status_Reference",
        desc="The Thanksgiving Day 2024 closure status for Costco is verified through a reference URL",
        parent=costco_parent,
        critical=True
    )
    costco_leaf = evaluator.add_leaf(
        id="Costco_Status_Determination",
        desc="The answer provides a clear statement about Costco Wholesale's operational status on Thanksgiving Day 2024",
        parent=costco_parent,
        critical=True
    )
    costco_status = normalize_open_closed(tg.costco_status if tg else None) if tg else None
    costco_claim = (
        f"Costco Wholesale was {costco_status} on Thanksgiving Day 2024 (Thursday, November 28, 2024) in the Atlanta area."
        if costco_status else
        "Costco Wholesale's Thanksgiving Day 2024 operational status in the Atlanta area is as specified."
    )
    await evaluator.verify(
        claim=costco_claim,
        node=costco_leaf,
        sources=(tg.costco_status_url if tg else None),
        additional_instruction="Verify whether Costco was open or closed on Nov 28, 2024. Prefer official announcements or credible local news/hours pages."
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates subtasks independently
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

    # Extract all needed info
    full_info = await evaluator.extract(
        prompt=prompt_extract_full(),
        template_class=FullExtraction,
        extraction_name="full_extraction"
    )

    # Build verification subtrees
    await build_cvs_subtree(evaluator, root, full_info.cvs if full_info else None)
    await build_publix_subtree(evaluator, root, full_info.publix if full_info else None)
    await build_costco_subtree(evaluator, root, full_info.costco if full_info else None)
    await build_thanksgiving_subtree(evaluator, root, full_info.thanksgiving if full_info else None)

    return evaluator.get_summary()