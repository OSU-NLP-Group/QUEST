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
TASK_ID = "phx_cvs_walgreens_services"
TASK_DESCRIPTION = """
I am researching pharmacy options in the Phoenix metropolitan area and need to identify four different CVS or Walgreens locations, each offering specific combinations of services. For each location, provide the pharmacy chain name, complete street address, and a direct link to the location's store-specific page on the official CVS or Walgreens website.

The four locations must meet the following requirements:

Location 1: Must have a 24-hour pharmacy department, drive-thru pharmacy service, and in-store photo printing services with same-day pickup.

Location 2: Must offer immunization services (including flu shots and COVID-19 vaccines), prescription delivery services (same-day or 1-2 day delivery), and curbside pickup for prescriptions and store items.

Location 3: Must have an on-site health clinic (MinuteClinic for CVS or healthcare clinic for Walgreens), in-store photo printing services with same-day pickup, and drive-thru pharmacy service.

Location 4: Must have a 24-hour pharmacy department, immunization services (including flu shots and COVID-19 vaccines), and prescription delivery services (same-day or 1-2 day delivery).
"""

PHX_METRO_CITIES = [
    "Phoenix", "Scottsdale", "Tempe", "Mesa", "Chandler", "Gilbert", "Glendale",
    "Peoria", "Surprise", "Avondale", "Goodyear", "Buckeye", "Queen Creek",
    "Paradise Valley", "Fountain Hills", "El Mirage", "Litchfield Park",
    "Tolleson", "Sun City", "Sun City West", "Sun Lakes", "Anthem",
    "Carefree", "Cave Creek", "New River", "Youngtown"
]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class LocationItem(BaseModel):
    chain: Optional[str] = None  # Expected: "CVS" or "Walgreens"
    address: Optional[str] = None  # Complete street address (street, city, state, ZIP)
    store_url: Optional[str] = None  # Direct store-specific URL on official domain


class LocationsExtraction(BaseModel):
    locations: List[LocationItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_locations() -> str:
    return """
    Extract up to four pharmacy locations (CVS or Walgreens) from the answer.
    For each location, extract:
    1) chain: The pharmacy chain name, normalized to 'CVS' or 'Walgreens' if possible.
    2) address: The complete street address as presented (street, city, state, and ZIP if available).
    3) store_url: A direct link to the location's store-specific page on the official CVS or Walgreens website.
       - The URL must be on cvs.com or walgreens.com.
       - It should be a page for a specific store location (not the general homepage or a generic locator landing page).
       - If the answer provides multiple URLs for a location, choose the one that best matches a store-specific details page.
    Return a JSON object with a 'locations' array (up to 4 items) where each item contains 'chain', 'address', and 'store_url'.
    If any field is missing for a location, set it to null. Do not invent information not present in the answer.
    If the answer contains more than four locations, include only the first four.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_chain(chain: Optional[str]) -> Optional[str]:
    if not chain:
        return None
    c = chain.strip().lower()
    if "cvs" in c:
        return "CVS"
    if "walgreens" in c or "walgreen" in c:
        return "Walgreens"
    return None


def is_official_domain(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip().lower()
    return ("cvs.com" in u) or ("walgreens.com" in u)


def expected_domain_for_chain(chain: Optional[str]) -> Optional[str]:
    if not chain:
        return None
    if chain == "CVS":
        return "cvs.com"
    if chain == "Walgreens":
        return "walgreens.com"
    return None


def get_location_desc(index: int) -> str:
    mapping = {
        1: "First pharmacy location with 24-hour pharmacy, drive-thru, and photo services",
        2: "Second pharmacy location with immunizations, delivery, and curbside pickup",
        3: "Third pharmacy location with health clinic, photo services, and drive-thru",
        4: "Fourth pharmacy location with 24-hour pharmacy, immunizations, and delivery",
    }
    return mapping.get(index, f"Pharmacy location #{index}")


def required_services_for(index: int) -> List[str]:
    mapping = {
        1: ["24hr", "drive_thru", "photo_sameday"],
        2: ["immunizations", "delivery", "curbside"],
        3: ["clinic", "photo_sameday", "drive_thru"],
        4: ["24hr", "immunizations", "delivery"],
    }
    return mapping.get(index, [])


def service_desc_node_id(index: int, service_code: str) -> (str, str):
    if index == 1:
        mapping = {
            "24hr": ("Location_1_24Hour", "The pharmacy department operates 24 hours a day"),
            "drive_thru": ("Location_1_DriveThru", "The location has a drive-through window for pharmacy services"),
            "photo_sameday": ("Location_1_Photo", "The store offers photo printing services with same-day pickup capability"),
        }
    elif index == 2:
        mapping = {
            "immunizations": ("Location_2_Immunizations", "The pharmacy provides immunization services including flu shots and COVID-19 vaccines"),
            "delivery": ("Location_2_Delivery", "The location offers prescription delivery services (same-day or 1-2 day delivery)"),
            "curbside": ("Location_2_Curbside", "The store offers curbside pickup service for prescriptions and store items"),
        }
    elif index == 3:
        mapping = {
            "clinic": ("Location_3_Clinic", "The location has an on-site health clinic (MinuteClinic for CVS or healthcare clinic for Walgreens)"),
            "photo_sameday": ("Location_3_Photo", "The store offers photo printing services with same-day pickup capability"),
            "drive_thru": ("Location_3_DriveThru", "The location has a drive-through window for pharmacy services"),
        }
    elif index == 4:
        mapping = {
            "24hr": ("Location_4_24Hour", "The pharmacy department operates 24 hours a day"),
            "immunizations": ("Location_4_Immunizations", "The pharmacy provides immunization services including flu shots and COVID-19 vaccines"),
            "delivery": ("Location_4_Delivery", "The location offers prescription delivery services (same-day or 1-2 day delivery)"),
        }
    else:
        mapping = {}
    return mapping.get(service_code, (f"Location_{index}_{service_code}", f"Service check: {service_code}"))


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def add_identity_checks(
    evaluator: Evaluator,
    parent_node,
    index: int,
    loc: LocationItem
) -> None:
    """
    Add critical identity checks: chain provided, chain matches page, address provided,
    address matches page, and city within Phoenix metro (verified via the page).
    """
    chain_norm = normalize_chain(loc.chain)
    id_node = evaluator.add_parallel(
        id=f"Location_{index}_Identity",
        desc="Correctly identifies the pharmacy chain (CVS or Walgreens) and provides the complete street address in Phoenix metro area",
        parent=parent_node,
        critical=True
    )

    # Chain provided (critical, custom)
    evaluator.add_custom_node(
        result=(chain_norm in {"CVS", "Walgreens"}),
        id=f"Location_{index}_ChainProvided",
        desc="Chain is explicitly identified as CVS or Walgreens",
        parent=id_node,
        critical=True
    )

    # Chain matches URL content (critical, LLM + URL)
    chain_leaf = evaluator.add_leaf(
        id=f"Location_{index}_ChainMatchesURL",
        desc=f"The page confirms the chain is '{chain_norm}'",
        parent=id_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This webpage is the official store page for a {chain_norm} pharmacy location.",
        node=chain_leaf,
        sources=loc.store_url,
        additional_instruction="Check the page branding and text to confirm the chain (CVS vs Walgreens). Ignore third-party directories. The domain should be official."
    )

    # Address provided (critical, custom)
    evaluator.add_custom_node(
        result=(loc.address is not None and str(loc.address).strip() != ""),
        id=f"Location_{index}_AddressProvided",
        desc="A complete street address was provided in the answer",
        parent=id_node,
        critical=True
    )

    # Address matches the page (critical, LLM + URL)
    addr_leaf = evaluator.add_leaf(
        id=f"Location_{index}_AddressMatchesPage",
        desc="The store page shows the same address as provided (allow minor formatting/abbreviation differences)",
        parent=id_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The store's address on this page matches or is equivalent to: {loc.address}",
        node=addr_leaf,
        sources=loc.store_url,
        additional_instruction="Allow minor differences such as abbreviations (e.g., Rd vs Road), punctuation, or formatting. Confirm it refers to the same physical address."
    )

    # City is within Phoenix metro (critical, LLM + URL)
    metro_leaf = evaluator.add_leaf(
        id=f"Location_{index}_CityInPhoenixMetro",
        desc="The city in the store's address is within the Phoenix metropolitan area",
        parent=id_node,
        critical=True
    )
    cities_list = ", ".join(PHX_METRO_CITIES)
    await evaluator.verify(
        claim=f"The city shown in the store's address is one of the following Phoenix metropolitan area cities: {cities_list}.",
        node=metro_leaf,
        sources=loc.store_url,
        additional_instruction="Look at the city name in the address on this page and check if it appears in the provided list."
    )


async def add_url_checks(
    evaluator: Evaluator,
    parent_node,
    index: int,
    loc: LocationItem
) -> None:
    """
    Add critical URL checks: provided, official domain, and store-specific page.
    """
    url_node = evaluator.add_parallel(
        id=f"Location_{index}_URL",
        desc="Provides a direct link to the location's store-specific page on the official CVS or Walgreens website",
        parent=parent_node,
        critical=True
    )

    # URL provided
    evaluator.add_custom_node(
        result=(loc.store_url is not None and str(loc.store_url).strip() != ""),
        id=f"Location_{index}_URLProvided",
        desc="A store-specific URL was provided",
        parent=url_node,
        critical=True
    )

    # Official domain check (simple custom)
    evaluator.add_custom_node(
        result=is_official_domain(loc.store_url),
        id=f"Location_{index}_OfficialDomain",
        desc="URL is on the official cvs.com or walgreens.com domain",
        parent=url_node,
        critical=True
    )

    # Store-specific page verification
    store_specific_leaf = evaluator.add_leaf(
        id=f"Location_{index}_StoreSpecificPage",
        desc="The URL is a store-specific details page (not a general homepage or generic locator)",
        parent=url_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page is a store-specific details page for a particular physical location (e.g., shows store address/hours). It is not just a generic locator or corporate homepage.",
        node=store_specific_leaf,
        sources=loc.store_url,
        additional_instruction="Look for a unique store identifier, full address, hours, and services for this exact location."
    )


async def add_service_check(
    evaluator: Evaluator,
    services_node,
    index: int,
    service_code: str,
    loc: LocationItem
) -> None:
    """
    Add a non-critical service verification leaf per required service, grounded on the store URL.
    """
    node_id, node_desc = service_desc_node_id(index, service_code)
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=node_desc,
        parent=services_node,
        critical=False
    )

    # Build claim and instruction per service
    if service_code == "24hr":
        claim = "This store's pharmacy department is open 24 hours a day."
        add_ins = "Focus on pharmacy hours (not just store hours). Accept explicit '24 hours' indications."
    elif service_code == "drive_thru":
        claim = "This location has a drive-thru pharmacy service."
        add_ins = "Look for 'Drive-Thru Pharmacy', 'Drive-thru', or similar phrasing indicating a pharmacy drive-through."
    elif service_code == "photo_sameday":
        claim = "This store offers photo printing services with same-day pickup."
        add_ins = "Look for 'Photo' and 'Same Day Pickup' availability at this specific store."
    elif service_code == "immunizations":
        claim = "This store provides immunization services, including flu shots and COVID-19 vaccines."
        add_ins = "Verify the page indicates vaccine services at this store. Accept mentions of scheduling flu or COVID-19 vaccines at this location."
    elif service_code == "delivery":
        claim = "This store offers prescription delivery services (same-day or 1-2 day delivery)."
        add_ins = "Look for 'Same Day Rx Delivery' or '1–2 day delivery' available for prescriptions from this store."
    elif service_code == "curbside":
        claim = "This store offers curbside pickup for prescriptions and store items."
        add_ins = "Look for 'Curbside Pickup' or similar wording tied to this specific location."
    elif service_code == "clinic":
        claim = "This location has an on-site health clinic (MinuteClinic for CVS or a healthcare clinic/Village Medical for Walgreens)."
        add_ins = "Confirm the presence of MinuteClinic (CVS) or an on-site clinic partner (Walgreens), clearly associated with this store."
    else:
        claim = f"This store offers the service: {service_code}."
        add_ins = "Verify the service is explicitly indicated as available at this store."

    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=loc.store_url,
        additional_instruction=add_ins
    )


async def verify_location(
    evaluator: Evaluator,
    root_parent,
    index: int,
    loc: LocationItem
) -> None:
    """
    Build the verification subtree for a single location, including:
    - Identity (critical)
    - URL (critical)
    - Services (non-critical)
    """
    # Parent node for the location
    location_node = evaluator.add_parallel(
        id=f"Location_{index}",
        desc=get_location_desc(index),
        parent=root_parent,
        critical=False
    )

    # Critical identity checks
    await add_identity_checks(evaluator, location_node, index, loc)

    # Critical URL checks
    await add_url_checks(evaluator, location_node, index, loc)

    # Services (non-critical)
    services_node = evaluator.add_parallel(
        id=f"Location_{index}_Services",
        desc="The location offers all required services for this slot",
        parent=location_node,
        critical=False
    )
    for svc in required_services_for(index):
        await add_service_check(evaluator, services_node, index, svc, loc)


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
    Evaluate an answer for the Phoenix CVS/Walgreens locations task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Four locations are independent
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

    # Extract up to 4 locations from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_locations(),
        template_class=LocationsExtraction,
        extraction_name="locations_extraction"
    )

    # Normalize and pad/truncate to exactly 4 items
    locs: List[LocationItem] = list(extracted.locations or [])
    if len(locs) > 4:
        locs = locs[:4]
    while len(locs) < 4:
        locs.append(LocationItem())

    # Optional: uniqueness check on non-empty addresses (non-critical)
    addr_values = [l.address.strip() for l in locs if l.address and str(l.address).strip() != ""]
    unique_nonempty = len(set(a.lower() for a in addr_values)) == len(addr_values)
    evaluator.add_custom_node(
        result=unique_nonempty,
        id="Distinct_Locations_Check",
        desc="All provided non-empty addresses are distinct (four different locations)",
        parent=root,
        critical=False
    )

    # Build verification subtrees for each of the 4 locations
    for i in range(4):
        await verify_location(evaluator, root, i + 1, locs[i])

    # Return structured summary
    return evaluator.get_summary()