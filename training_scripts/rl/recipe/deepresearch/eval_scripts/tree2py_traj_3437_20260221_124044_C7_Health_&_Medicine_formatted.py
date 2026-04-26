import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "walgreens_chicago_24hr_comprehensive_vaccination_center"
TASK_DESCRIPTION = (
    "I need to find a Walgreens pharmacy location in Chicago, IL that meets all of the following requirements:\n\n"
    "1. The pharmacy department must operate 24 hours per day, every day\n"
    "2. The retail store must also operate 24 hours\n"
    "3. Must offer COVID-19 vaccination for individuals ages 3 and older\n"
    "4. Must offer flu shots with walk-in availability (no appointment needed)\n"
    "5. Must offer Shingrix (shingles vaccine) for adults ages 50+\n"
    "6. Must be listed by the CDC as an authorized Yellow Fever vaccination clinic\n"
    "7. Must offer pneumococcal (pneumonia) vaccination services\n"
    "8. Must offer typhoid vaccination for travelers\n"
    "9. Must provide prescription refill services\n\n"
    "Please provide one specific Walgreens location that meets all these requirements, including:\n"
    "- Complete street address (street number, street name, city, state, and zip code)\n"
    "- Contact phone number\n"
    "- Confirmation that same-day pickup and curbside pickup services are available (if offered at this location)"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ClaimSources(BaseModel):
    """Claim-specific URLs explicitly cited in the answer."""
    chicago_il: List[str] = Field(default_factory=list)
    pharmacy_24h: List[str] = Field(default_factory=list)
    store_24h: List[str] = Field(default_factory=list)
    covid19_vaccination_3plus: List[str] = Field(default_factory=list)
    flu_shot_walk_in: List[str] = Field(default_factory=list)
    shingles_shingrix_50plus: List[str] = Field(default_factory=list)
    yellow_fever_cdc_listed: List[str] = Field(default_factory=list)
    pneumonia_vaccine: List[str] = Field(default_factory=list)
    typhoid_vaccine: List[str] = Field(default_factory=list)
    prescription_refill: List[str] = Field(default_factory=list)
    same_day_pickup: List[str] = Field(default_factory=list)
    curbside_pickup: List[str] = Field(default_factory=list)


class WalgreensLocationExtraction(BaseModel):
    """Structured info for one Walgreens location from the answer."""
    location_name: Optional[str] = None  # e.g., "Walgreens" or "Walgreens #1234"
    address_line: Optional[str] = None   # e.g., "1234 N Example Ave"
    city: Optional[str] = None           # e.g., "Chicago"
    state: Optional[str] = None          # e.g., "IL"
    zip_code: Optional[str] = None       # e.g., "60616"
    phone_number: Optional[str] = None

    # Optional affirmations (free text from the answer; not strictly needed for verification)
    pharmacy_hours_24_7: Optional[str] = None
    store_hours_24_7: Optional[str] = None
    covid19_age3plus: Optional[str] = None
    flu_shot_walk_in: Optional[str] = None
    shingles_shingrix_50plus: Optional[str] = None
    yellow_fever_cdc: Optional[str] = None
    pneumonia_vaccine: Optional[str] = None
    typhoid_vaccine: Optional[str] = None
    rx_refill: Optional[str] = None
    same_day_pickup: Optional[str] = None
    curbside_pickup: Optional[str] = None

    # Claim-specific URLs
    sources: ClaimSources = Field(default_factory=ClaimSources)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_walgreens_location() -> str:
    return """
Extract exactly one Walgreens location (the first location described) from the answer and the URLs the answer cites to support each requirement.

Return a JSON with these fields:
- location_name: Name/identifier of the Walgreens location if provided (e.g., "Walgreens", "Walgreens #1234"). Null if not provided.
- address_line: The street number + street name portion of the address (e.g., "151 N State St"). Null if not provided.
- city: City name. Null if not provided.
- state: Two-letter state code. Null if not provided.
- zip_code: 5-digit ZIP code. Null if not provided.
- phone_number: The location's phone number as written in the answer. Null if not provided.

Also include any short affirmation strings if present in the answer (else null):
- pharmacy_hours_24_7
- store_hours_24_7
- covid19_age3plus
- flu_shot_walk_in
- shingles_shingrix_50plus
- yellow_fever_cdc
- pneumonia_vaccine
- typhoid_vaccine
- rx_refill
- same_day_pickup
- curbside_pickup

Finally, include a 'sources' object with arrays of URLs explicitly mentioned in the answer that support each of the following claim categories. For each category, extract only valid URLs from the answer (plain URLs or markdown links). If the answer does not provide a URL for that category, return an empty list for that category.

'sources' fields to extract:
- chicago_il: URLs that show the location address/city/state is Chicago, IL
- pharmacy_24h: URLs that show the pharmacy department is open 24 hours
- store_24h: URLs that show the retail store is open 24 hours
- covid19_vaccination_3plus: URLs that show COVID-19 vaccines are offered to ages 3+
- flu_shot_walk_in: URLs that show flu shots are available with walk-in/no appointment
- shingles_shingrix_50plus: URLs that show Shingrix (shingles) for ages 50+
- yellow_fever_cdc_listed: URLs to CDC pages listing the location as an authorized Yellow Fever clinic
- pneumonia_vaccine: URLs that show pneumococcal (pneumonia) vaccines
- typhoid_vaccine: URLs that show typhoid vaccination available
- prescription_refill: URLs that show prescription refill services at this location
- same_day_pickup: URLs that show same-day pickup is available at this location
- curbside_pickup: URLs that show curbside pickup is available at this location

IMPORTANT:
- Extract only what is explicitly present in the answer text.
- Do not invent URLs. Do not add URLs that are not shown in the answer.
- If the answer lists just one store page URL that supports multiple claims, include that URL in each relevant 'sources' list.
- If any field is not present in the answer, set it to null (or an empty list for URL arrays).
"""


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def format_address(ex: WalgreensLocationExtraction) -> str:
    """Build a readable address string from the extracted fields."""
    parts = []
    if ex.address_line:
        parts.append(ex.address_line.strip())
    city_state_zip = []
    if ex.city:
        city_state_zip.append(ex.city.strip())
    if ex.state:
        city_state_zip.append(ex.state.strip())
    if ex.zip_code:
        city_state_zip.append(ex.zip_code.strip())
    if city_state_zip:
        parts.append(", ".join(city_state_zip[:-1]) + (f" {city_state_zip[-1]}" if len(city_state_zip) >= 1 else ""))
    return ", ".join(parts).strip() if parts else "the specified Walgreens location"


def get_sources_list(ex: WalgreensLocationExtraction, field_name: str) -> List[str]:
    """Safely get a claim-specific sources list."""
    if not ex or not ex.sources:
        return []
    return getattr(ex.sources, field_name, []) or []


def is_valid_zip(zip_code: Optional[str]) -> bool:
    if not zip_code:
        return False
    return bool(re.fullmatch(r"\d{5}", zip_code.strip()))


def is_valid_phone(phone: Optional[str]) -> bool:
    if not phone:
        return False
    digits = re.sub(r"\D", "", phone)
    return len(digits) >= 10


async def add_claim_group(
    evaluator: Evaluator,
    root_node,
    *,
    claim_id: str,
    claim_desc: str,
    critical: bool,
    claim_text: str,
    sources: List[str],
    add_ins: str
) -> None:
    """
    Create a sequential sub-group for a single claim:
    1) Check that the answer provided source URLs (critical, enforces source-grounding)
    2) Verify the claim against the cited sources
    """
    group = evaluator.add_sequential(
        id=f"{claim_id}_group",
        desc=f"{claim_desc} (source-backed)",
        parent=root_node,
        critical=critical
    )

    # Existence of sources for this specific claim (enforce source-grounding)
    evaluator.add_custom_node(
        result=(len(sources) > 0),
        id=f"{claim_id}_sources_provided",
        desc=f"Source URLs are provided in the answer to support: {claim_desc}",
        parent=group,
        critical=True
    )

    # Actual verification leaf
    leaf = evaluator.add_leaf(
        id=claim_id,
        desc=claim_desc,
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=claim_text,
        node=leaf,
        sources=sources,
        additional_instruction=add_ins
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Walgreens Chicago 24-hour comprehensive vaccination center task.
    """
    # Initialize evaluator with a parallel root (to independently assess each requirement)
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

    # Extract structured information from the answer
    extracted: WalgreensLocationExtraction = await evaluator.extract(
        prompt=prompt_extract_walgreens_location(),
        template_class=WalgreensLocationExtraction,
        extraction_name="walgreens_location_extraction"
    )

    # Compose address string for claim texts
    address_str = format_address(extracted)
    loc_display = extracted.location_name.strip() if (extracted and extracted.location_name) else "the Walgreens location"

    # -------------------- Content completeness checks (critical) -------------------- #
    # Complete street address (street number & name, city, state, ZIP)
    addr_complete_result = (
        bool(extracted and extracted.address_line and extracted.address_line.strip()) and
        bool(extracted and extracted.city and extracted.city.strip()) and
        bool(extracted and extracted.state and extracted.state.strip()) and
        is_valid_zip(extracted.zip_code)
    )
    evaluator.add_custom_node(
        result=addr_complete_result,
        id="Complete_Street_Address",
        desc="Provides complete street address including street number, street name, city, state, and 5-digit zip code",
        parent=root,
        critical=True
    )

    # Contact phone number provided (critical)
    phone_ok = is_valid_phone(extracted.phone_number)
    evaluator.add_custom_node(
        result=phone_ok,
        id="Contact_Phone_Number",
        desc="Provides a contact phone number for the pharmacy",
        parent=root,
        critical=True
    )

    # -------------------- Core requirement verifications (critical) -------------------- #
    # Chicago, IL location
    await add_claim_group(
        evaluator,
        root,
        claim_id="Chicago_IL_Location",
        claim_desc="The pharmacy is a Walgreens location within Chicago, IL city limits",
        critical=True,
        claim_text=f"The Walgreens location at {address_str} is located within the City of Chicago, Illinois (IL).",
        sources=get_sources_list(extracted, "chicago_il"),
        add_ins=(
            "Verify that the webpage clearly shows the city as 'Chicago, IL' (or equivalent). "
            "If the city is a separate suburb (e.g., Evanston, Skokie, Oak Park), the claim is not supported. "
            "Allow minor formatting variations (e.g., 'Chicago IL', 'Chicago, Illinois')."
        )
    )

    # 24-hour pharmacy
    await add_claim_group(
        evaluator,
        root,
        claim_id="24Hour_Pharmacy_Operation",
        claim_desc="The pharmacy department operates 24 hours per day, every day",
        critical=True,
        claim_text=f"The pharmacy department at {loc_display} at {address_str} operates 24 hours a day, every day.",
        sources=get_sources_list(extracted, "pharmacy_24h"),
        add_ins=(
            "Confirm PHARMACY hours specifically indicate 'Open 24 hours' or equivalent, not merely the retail store hours. "
            "Look for explicit 'Pharmacy hours' sections on the store page or an official Walgreens page for this location."
        )
    )

    # 24-hour retail store
    await add_claim_group(
        evaluator,
        root,
        claim_id="24Hour_Store_Operation",
        claim_desc="The retail store operates 24 hours per day",
        critical=True,
        claim_text=f"The retail store at {loc_display} at {address_str} is open 24 hours per day.",
        sources=get_sources_list(extracted, "store_24h"),
        add_ins=(
            "Confirm STORE/RETAIL hours indicate 'Open 24 hours' or equivalent. "
            "Distinguish from pharmacy hours; this check is for the store itself."
        )
    )

    # COVID-19 vaccination for ages 3+
    await add_claim_group(
        evaluator,
        root,
        claim_id="COVID19_Vaccination",
        claim_desc="Offers COVID-19 vaccination services for individuals ages 3 and older",
        critical=True,
        claim_text=f"{loc_display} at {address_str} offers COVID-19 vaccination for individuals aged 3 years and older.",
        sources=get_sources_list(extracted, "covid19_vaccination_3plus"),
        add_ins=(
            "Look for explicit mention of 'ages 3+' or '3 years and older' for COVID-19 vaccination. "
            "Accept official Walgreens policy pages if clearly applicable to this location."
        )
    )

    # Flu shots with walk-in availability
    await add_claim_group(
        evaluator,
        root,
        claim_id="Flu_Shot_Walk_In",
        claim_desc="Offers flu shots with walk-in availability (no appointment required)",
        critical=True,
        claim_text=f"{loc_display} at {address_str} offers flu shots with walk-in availability (no appointment needed).",
        sources=get_sources_list(extracted, "flu_shot_walk_in"),
        add_ins=(
            "Verify that the page mentions 'walk-in', 'no appointment needed', or similar for flu shots at this location."
        )
    )

    # Shingrix (shingles) for adults 50+
    await add_claim_group(
        evaluator,
        root,
        claim_id="Shingles_Vaccine",
        claim_desc="Offers Shingrix (shingles vaccine) for adults ages 50 and older",
        critical=True,
        claim_text=f"{loc_display} at {address_str} offers Shingrix (the shingles vaccine) for adults aged 50+.",
        sources=get_sources_list(extracted, "shingles_shingrix_50plus"),
        add_ins=(
            "Check for 'Shingrix' and age guidance '50 years and older' (or '50+'). "
            "Prefer location-specific or official Walgreens vaccine pages applicable to this location."
        )
    )

    # CDC Yellow Fever authorized clinic (CDC listing)
    await add_claim_group(
        evaluator,
        root,
        claim_id="Yellow_Fever_Vaccine_CDC_Listed",
        claim_desc="Listed by the CDC as an authorized Yellow Fever vaccination clinic",
        critical=True,
        claim_text=f"{loc_display} at {address_str} is listed by the CDC as an authorized Yellow Fever vaccination clinic.",
        sources=get_sources_list(extracted, "yellow_fever_cdc_listed"),
        add_ins=(
            "Use CDC 'Yellow Fever Vaccination Clinics' pages. Verify the specific location (matching address/city/state) "
            "appears on the CDC list. Corporate pages alone are insufficient for this claim."
        )
    )

    # Pneumococcal (pneumonia) vaccine
    await add_claim_group(
        evaluator,
        root,
        claim_id="Pneumonia_Vaccine",
        claim_desc="Offers pneumococcal (pneumonia) vaccination services",
        critical=True,
        claim_text=f"{loc_display} at {address_str} offers pneumococcal (pneumonia) vaccination services (e.g., PCV/PPSV).",
        sources=get_sources_list(extracted, "pneumonia_vaccine"),
        add_ins=(
            "Look for 'pneumococcal' (PCV, PPSV) vaccines. Accept official Walgreens vaccine pages applicable to this location."
        )
    )

    # Typhoid vaccine for travelers
    await add_claim_group(
        evaluator,
        root,
        claim_id="Typhoid_Vaccine",
        claim_desc="Offers typhoid vaccination for travelers",
        critical=True,
        claim_text=f"{loc_display} at {address_str} offers typhoid vaccination for travelers.",
        sources=get_sources_list(extracted, "typhoid_vaccine"),
        add_ins=(
            "Verify the availability of typhoid vaccine (injectable or oral) at this location or as a service that can be obtained through this location."
        )
    )

    # Prescription refill services
    await add_claim_group(
        evaluator,
        root,
        claim_id="Prescription_Refill_Services",
        claim_desc="Provides prescription refill services",
        critical=True,
        claim_text=f"{loc_display} at {address_str} provides prescription refill services.",
        sources=get_sources_list(extracted, "prescription_refill"),
        add_ins=(
            "Look for 'Refill prescriptions', 'Pharmacy services', or similar on the location page or official Walgreens page applicable to the location."
        )
    )

    # -------------------- Optional service verifications (non-critical) ------------- #
    # Same-day pickup
    await add_claim_group(
        evaluator,
        root,
        claim_id="Same_Day_Pickup",
        claim_desc="Offers same-day order pickup services for online orders",
        critical=False,
        claim_text=f"Same Day Pickup is available at {loc_display} at {address_str}.",
        sources=get_sources_list(extracted, "same_day_pickup"),
        add_ins=(
            "Verify 'Same Day Pickup' is offered at this location (commonly shown on Walgreens store pages or service pages)."
        )
    )

    # Curbside pickup
    await add_claim_group(
        evaluator,
        root,
        claim_id="Curbside_Pickup",
        claim_desc="Has curbside pickup service available",
        critical=False,
        claim_text=f"Curbside pickup service is available at {loc_display} at {address_str}.",
        sources=get_sources_list(extracted, "curbside_pickup"),
        add_ins=(
            "Verify 'Curbside pickup' service availability at this location."
        )
    )

    # Return structured summary
    return evaluator.get_summary()