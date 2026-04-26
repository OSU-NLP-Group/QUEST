import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ikea_austin_domain_2024"
TASK_DESCRIPTION = (
    'In 2024, IKEA opened a new smaller-format "Plan & Order Point with Pick-up" store at The Domain shopping center in Austin, Texas. '
    "This was IKEA's second location in the Austin area. Provide the following information about this store: "
    "(1) The complete street address (street number, street name, city, state, and ZIP code), "
    "(2) The exact square footage of the store, and (3) The specific opening date (month, day, and year)."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StoreAddress(BaseModel):
    street_number: Optional[str] = None
    street_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    full_address: Optional[str] = None


class SquareFootageInfo(BaseModel):
    square_footage_text: Optional[str] = None
    numeric_value: Optional[str] = None  # digits only, no commas; null if not exact
    units: Optional[str] = None  # e.g., "square feet", "sq ft", "ft²"


class OpeningDateInfo(BaseModel):
    opening_date_text: Optional[str] = None
    month: Optional[str] = None   # e.g., "August" or "08"
    day: Optional[str] = None     # e.g., "2"
    year: Optional[str] = None    # e.g., "2024"


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_address() -> str:
    return """
    Extract the address for IKEA's smaller-format "Plan & Order Point with Pick-up" store located at The Domain in Austin, Texas (the 2024 opening) as stated in the provided answer text.

    Return a JSON object with:
    - street_number: the numeric street number (digits only), or null if not present
    - street_name: the street name portion (e.g., "Esperanza Crossing"), excluding city/state/ZIP/suite, or null if not present
    - city: the city name, or null if not present
    - state: the U.S. state (accept "TX" or "Texas"), or null if not present
    - zip_code: the ZIP code; if a ZIP+4 is given like "78758-1234", return just the leading 5-digit ZIP (e.g., "78758"); return null if not present
    - full_address: the complete address string as it appears in the answer (if available), otherwise null

    If multiple addresses are present, extract the one that is clearly for the Plan & Order Point with Pick-up store at The Domain (Austin).
    Do not invent or infer any missing part—return null when a field is not explicitly provided.
    """


def prompt_extract_square_footage() -> str:
    return """
    Extract the exact square footage for IKEA's Plan & Order Point with Pick-up store at The Domain (Austin) from the answer text.

    Return a JSON object with:
    - square_footage_text: the square footage phrase exactly as written in the answer (e.g., "6,500 square feet")
    - numeric_value: a single exact numeric value (digits only, no commas) if the answer states an exact size; 
      If the answer uses a range (e.g., "6,000–7,000"), or approximation words ("about", "approx", "~", "around"), set this to null.
      For "6,500 square feet", return "6500".
    - units: the units string used in the answer for square footage (e.g., "square feet", "sq ft", "ft²"), or null if units not stated

    If square footage is not mentioned, return all fields as null.
    """


def prompt_extract_opening_date() -> str:
    return """
    Extract the specific opening date (month, day, and year) for IKEA's Plan & Order Point with Pick-up store at The Domain (Austin) as stated in the answer.

    Return a JSON object with:
    - opening_date_text: the date exactly as written in the answer (e.g., "July 31, 2024")
    - month: the month component (prefer the full month name like "July" or a standard abbreviation like "Jul"; numeric "07" is acceptable if that's how it appears)
    - day: the day of month as digits only (e.g., "31")
    - year: the four-digit year (e.g., "2024")

    If any component is missing in the answer, set that field to null.
    """


# --------------------------------------------------------------------------- #
# Helper validation utilities                                                 #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s2 = s.strip()
    return s2 if s2 else None


def _has_five_digit_zip(zip_code: Optional[str], fallback_text: Optional[str]) -> bool:
    # Accept either a direct 5-digit ZIP or ZIP+4; presence of 5 digits anywhere counts.
    patterns = [
        r"^\d{5}$",
        r"^\d{5}-\d{4}$",
    ]
    if zip_code:
        z = zip_code.strip()
        if any(re.match(p, z) for p in patterns):
            return True
        # If provided zip_code is not normalized, search for a 5-digit sequence
        if re.search(r"\b\d{5}\b", z):
            return True
    if fallback_text and re.search(r"\b\d{5}\b", fallback_text):
        return True
    return False


def _has_street_number_and_name(street_number: Optional[str], street_name: Optional[str]) -> bool:
    return bool(_norm(street_number) and _norm(street_name))


def _has_city_and_state(city: Optional[str], state: Optional[str]) -> bool:
    return bool(_norm(city) and _norm(state))


def _is_exact_sqft(numeric_value: Optional[str], units: Optional[str], original_text: Optional[str]) -> bool:
    # numeric_value must be digits only
    if not _norm(numeric_value):
        return False

    if not re.fullmatch(r"\d+", numeric_value.strip()):
        return False

    # Ensure units indicate square feet
    units_candidates = set()
    if units:
        units_candidates.add(units.strip().lower())
    if original_text:
        # Collect candidates from text to be safe
        text_low = original_text.lower()
        if "square feet" in text_low or "square foot" in text_low:
            units_candidates.add("square feet")
        if "sq ft" in text_low:
            units_candidates.add("sq ft")
        if "sqft" in text_low:
            units_candidates.add("sqft")
        if "ft²" in text_low or "ft^2" in text_low or "ft2" in text_low:
            units_candidates.add("ft²")
        if "s.f." in text_low or re.search(r"\bsf\b", text_low):
            units_candidates.add("sf")

        # Disallow approximations for "exact" requirement
        approx_markers = ["approx", "approximately", "about", "around", "roughly", "~", "approx."]
        if any(tok in text_low for tok in approx_markers):
            return False

        # Disallow explicit ranges (e.g., 6,000–7,000)
        if re.search(r"\d[\d,]*\s*[-–]\s*\d[\d,]*", text_low):
            return False

    valid_sqft_units = {"square feet", "square foot", "sq ft", "sqft", "ft²", "ft^2", "ft2", "s.f.", "sf"}
    if not any(u in valid_sqft_units for u in units_candidates):
        return False

    return True


def _valid_month(month_str: Optional[str]) -> bool:
    if not _norm(month_str):
        return False
    s = month_str.strip().lower().rstrip(".")
    months = {
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    }
    if s in months:
        return True
    # numeric month, 1-12 or zero-padded 01-12
    if re.fullmatch(r"0?[1-9]|1[0-2]", s):
        return True
    return False


def _valid_day(day_str: Optional[str]) -> bool:
    if not _norm(day_str):
        return False
    if not re.fullmatch(r"\d{1,2}", day_str.strip()):
        return False
    day = int(day_str)
    return 1 <= day <= 31


def _valid_year(year_str: Optional[str]) -> bool:
    if not _norm(year_str):
        return False
    return bool(re.fullmatch(r"\d{4}", year_str.strip()))


def _has_month_day_year(date_info: OpeningDateInfo) -> bool:
    return _valid_month(date_info.month) and _valid_day(date_info.day) and _valid_year(date_info.year)


# --------------------------------------------------------------------------- #
# Build verification subtrees                                                 #
# --------------------------------------------------------------------------- #
def build_address_checks(evaluator: Evaluator, parent_node, addr: StoreAddress) -> None:
    # Group node for address (critical)
    addr_node = evaluator.add_parallel(
        id="Complete_Street_Address",
        desc="Provide the complete street address including street number, street name, city, state, and ZIP code.",
        parent=parent_node,
        critical=True
    )

    # Street number and name present
    street_ok = _has_street_number_and_name(addr.street_number, addr.street_name)
    evaluator.add_custom_node(
        result=street_ok,
        id="Street_Number_and_Name_Present",
        desc="Street number and street name are present in the address.",
        parent=addr_node,
        critical=True
    )

    # City and state present
    city_state_ok = _has_city_and_state(addr.city, addr.state)
    evaluator.add_custom_node(
        result=city_state_ok,
        id="City_and_State_Present",
        desc="City and state are present in the address.",
        parent=addr_node,
        critical=True
    )

    # ZIP code present (5-digit presence)
    zip_ok = _has_five_digit_zip(addr.zip_code, addr.full_address)
    evaluator.add_custom_node(
        result=zip_ok,
        id="ZIP_Code_Present",
        desc="A 5-digit ZIP code is present in the address.",
        parent=addr_node,
        critical=True
    )


def build_sqft_checks(evaluator: Evaluator, parent_node, sqft: SquareFootageInfo) -> None:
    sqft_node = evaluator.add_parallel(
        id="Exact_Square_Footage",
        desc="Provide the exact square footage of the store.",
        parent=parent_node,
        critical=True
    )

    sqft_ok = _is_exact_sqft(sqft.numeric_value, sqft.units, sqft.square_footage_text)
    evaluator.add_custom_node(
        result=sqft_ok,
        id="Square_Footage_Value_and_Units",
        desc="Square footage is stated as an exact numeric value and expressed in square feet (e.g., 'sq ft' or 'square feet').",
        parent=sqft_node,
        critical=True
    )


def build_opening_date_checks(evaluator: Evaluator, parent_node, date_info: OpeningDateInfo) -> None:
    date_node = evaluator.add_parallel(
        id="Specific_Opening_Date",
        desc="Provide the specific opening date including month, day, and year.",
        parent=parent_node,
        critical=True
    )

    mdy_ok = _has_month_day_year(date_info)
    evaluator.add_custom_node(
        result=mdy_ok,
        id="Month_Day_Year_Present",
        desc="Opening date includes month, day, and year in an unambiguous format.",
        parent=date_node,
        critical=True
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
    Evaluate an answer for the IKEA Austin (The Domain) 2024 store info task.
    """
    # Initialize evaluator
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

    # Run extractions (in parallel)
    addr_task = evaluator.extract(
        prompt=prompt_extract_address(),
        template_class=StoreAddress,
        extraction_name="address_extraction"
    )
    sqft_task = evaluator.extract(
        prompt=prompt_extract_square_footage(),
        template_class=SquareFootageInfo,
        extraction_name="square_footage_extraction"
    )
    date_task = evaluator.extract(
        prompt=prompt_extract_opening_date(),
        template_class=OpeningDateInfo,
        extraction_name="opening_date_extraction"
    )
    address_info, sqft_info, date_info = await asyncio.gather(addr_task, sqft_task, date_task)

    # Build the rubric tree under a critical top-level node mirroring the rubric
    top_node = evaluator.add_parallel(
        id="IKEA_Austin_Store_Information",
        desc="Provide the requested information about IKEA's Plan & Order Point with Pick-up store at The Domain in Austin, Texas (2024 opening).",
        parent=root,
        critical=True
    )

    # Construct subtrees
    build_address_checks(evaluator, top_node, address_info)
    build_sqft_checks(evaluator, top_node, sqft_info)
    build_opening_date_checks(evaluator, top_node, date_info)

    # Optionally record a compact snapshot of extracted fields for debugging and transparency
    evaluator.add_custom_info(
        info={
            "address": address_info.dict(),
            "square_footage": sqft_info.dict(),
            "opening_date": date_info.dict()
        },
        info_type="extraction_snapshot",
        info_name="parsed_fields"
    )

    return evaluator.get_summary()