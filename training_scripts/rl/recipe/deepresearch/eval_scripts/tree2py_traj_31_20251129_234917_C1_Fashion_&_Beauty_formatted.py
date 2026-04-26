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
TASK_ID = "sephora_stanford_store_info"
TASK_DESCRIPTION = """
I'm planning to visit the Sephora store at Stanford Shopping Center in Palo Alto, California. Please provide the following information for this specific store location: the complete street address, the store's phone number, and the current store hours.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StoreAddress(BaseModel):
    full_address: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class StorePhone(BaseModel):
    phone: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class StoreHours(BaseModel):
    hours_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SephoraStanfordExtraction(BaseModel):
    address: Optional[StoreAddress] = None
    phone: Optional[StorePhone] = None
    hours: Optional[StoreHours] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_store_info() -> str:
    return """
    Extract the information for the Sephora store at Stanford Shopping Center in Palo Alto, California as it appears in the provided answer. Do not invent or infer details beyond what the answer states.

    Required JSON structure:
    {
      "address": {
        "full_address": string or null,
        "street": string or null,
        "city": string or null,
        "state": string or null,
        "zip_code": string or null,
        "sources": string[]  // URLs explicitly mentioned in the answer that support the address; if unclear, include any store-related URLs
      },
      "phone": {
        "phone": string or null,
        "sources": string[]  // URLs explicitly mentioned in the answer that support the phone number; if unclear, include any store-related URLs
      },
      "hours": {
        "hours_text": string or null,  // copy the hours text exactly as in the answer; if multi-line, preserve line breaks
        "sources": string[]  // URLs explicitly mentioned in the answer that support hours; if unclear, include any store-related URLs
      }
    }

    Rules:
    - "full_address" should be the complete street address string exactly as presented in the answer, if provided. If the answer lists components separately (street, city, state, zip), fill those components accordingly and construct "full_address" only if the answer explicitly presents such a combined string; otherwise leave "full_address" as null.
    - City should be "Palo Alto" or equivalent as stated; State should be "CA" or "California" as stated; Zip is the 5-digit code (allow ZIP+4 if provided).
    - Phone should be exactly as written in the answer (allow punctuation like parentheses, dashes, spaces, or a leading +1).
    - Hours: copy the hours exactly from the answer; if the answer uses a day-by-day list, preserve it as one multi-line string.
    - For sources arrays, extract only URLs explicitly present in the answer. If the answer provides store-related URLs without saying which field they support, include those in all relevant sources arrays (address/phone/hours). Do not fabricate URLs. If no URLs are present, return empty arrays.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _filter_http_urls(urls: List[str]) -> List[str]:
    return [u for u in urls if isinstance(u, str) and u.strip().lower().startswith(("http://", "https://"))]


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_address(
    evaluator: Evaluator,
    parent_node,
    address: Optional[StoreAddress],
) -> None:
    # Build the container node for Address (critical, sequential to gate checks)
    address_node = evaluator.add_sequential(
        id="Store_Address_With_Zip_And_CityState",
        desc="Provides the complete street address for the Sephora at Stanford Shopping Center, including Palo Alto, CA and the ZIP code.",
        parent=parent_node,
        critical=True,
    )

    # Existence/provided check
    provided = bool(
        address and (
            _nonempty(address.full_address)
            or (_nonempty(address.street) and _nonempty(address.city) and _nonempty(address.state) and _nonempty(address.zip_code))
        )
    )
    evaluator.add_custom_node(
        result=provided,
        id="address_provided",
        desc="Address is provided (as a full address string or as street, city, state, and ZIP).",
        parent=address_node,
        critical=True,
    )

    # Format/content check: includes Palo Alto, CA and a ZIP code
    format_leaf = evaluator.add_leaf(
        id="address_has_city_state_zip",
        desc="The provided address includes Palo Alto (city), CA (state), and a 5-digit ZIP (ZIP+4 acceptable).",
        parent=address_node,
        critical=True,
    )
    full_addr_for_check = address.full_address or " ".join(
        [p for p in [address.street, address.city, address.state, address.zip_code] if _nonempty(p)]
    ) if address else ""
    await evaluator.verify(
        claim=(
            f"The address '{full_addr_for_check}' explicitly includes 'Palo Alto' as the city, 'CA' (or 'California') as the state, "
            f"and contains a 5-digit ZIP code (ZIP+4 acceptable)."
        ),
        node=format_leaf,
        additional_instruction=(
            "Judge correctness by inspecting the given address string only. "
            "Accept 'California' as equivalent to 'CA'. Accept ZIP+4 formats like '94304-1404' as valid."
        ),
    )

    # Source-supported check
    source_leaf = evaluator.add_leaf(
        id="address_supported_by_sources",
        desc="The address corresponds to the Sephora store at Stanford Shopping Center in Palo Alto, CA per the cited sources.",
        parent=address_node,
        critical=True,
    )
    addr_sources = _filter_http_urls(address.sources if address else [])
    await evaluator.verify(
        claim=(
            f"The Sephora store at Stanford Shopping Center in Palo Alto, CA has the street address '{full_addr_for_check}'."
        ),
        node=source_leaf,
        sources=addr_sources,  # May be empty; then judge uses simple verification
        additional_instruction=(
            "Verify that the webpage explicitly lists the Sephora store (Stanford Shopping Center / Palo Alto location) and "
            "shows this address. Allow minor formatting differences (e.g., 'Ste' vs 'Suite', presence/absence of ZIP+4, punctuation)."
        ),
    )


async def verify_phone(
    evaluator: Evaluator,
    parent_node,
    phone: Optional[StorePhone],
) -> None:
    phone_node = evaluator.add_sequential(
        id="Store_Phone_Number_For_This_Location",
        desc="Provides the phone number for the Sephora location at Stanford Shopping Center (Palo Alto, CA).",
        parent=parent_node,
        critical=True,
    )

    # Existence/provided check
    provided = bool(phone and _nonempty(phone.phone))
    evaluator.add_custom_node(
        result=provided,
        id="phone_provided",
        desc="Phone number is provided.",
        parent=phone_node,
        critical=True,
    )

    # Validity/plausibility check
    valid_format_leaf = evaluator.add_leaf(
        id="phone_valid_format",
        desc="The provided phone number is a plausible US phone number format.",
        parent=phone_node,
        critical=True,
    )
    num_str = phone.phone if phone else ""
    await evaluator.verify(
        claim=(
            f"The phone number '{num_str}' is a plausible US phone number format (typically 10 digits, may include '+1', "
            f"parentheses, spaces, or hyphens)."
        ),
        node=valid_format_leaf,
        additional_instruction=(
            "This is a format plausibility check only. Do not verify external correctness here."
        ),
    )

    # Source-supported phone for this exact location
    source_leaf = evaluator.add_leaf(
        id="phone_supported_by_sources",
        desc="The phone number corresponds to the Stanford Shopping Center Sephora location per the cited sources.",
        parent=phone_node,
        critical=True,
    )
    phone_sources = _filter_http_urls(phone.sources if phone else [])
    await evaluator.verify(
        claim=(
            f"The phone number for the Sephora store at Stanford Shopping Center in Palo Alto, CA is '{num_str}'."
        ),
        node=source_leaf,
        sources=phone_sources,
        additional_instruction=(
            "Confirm that the page for the Sephora Stanford Shopping Center (Palo Alto) location lists this same phone number. "
            "Look for labels such as 'Phone', 'Call', or contact details for this location."
        ),
    )


async def verify_hours(
    evaluator: Evaluator,
    parent_node,
    hours: Optional[StoreHours],
) -> None:
    hours_node = evaluator.add_sequential(
        id="Current_Store_Hours_For_This_Location",
        desc="Provides the current store hours for the Sephora location at Stanford Shopping Center (Palo Alto, CA).",
        parent=parent_node,
        critical=True,
    )

    # Existence/provided check
    provided = bool(hours and _nonempty(hours.hours_text))
    evaluator.add_custom_node(
        result=provided,
        id="hours_provided",
        desc="Store hours are provided.",
        parent=hours_node,
        critical=True,
    )

    # Source-supported hours
    hours_leaf = evaluator.add_leaf(
        id="hours_supported_by_sources",
        desc="The provided hours match what is shown for this location on the cited sources.",
        parent=hours_node,
        critical=True,
    )
    hours_text = hours.hours_text if hours else ""
    hours_sources = _filter_http_urls(hours.sources if hours else [])
    await evaluator.verify(
        claim=(
            f"The current store hours for the Sephora store at Stanford Shopping Center in Palo Alto, CA match: {hours_text}"
        ),
        node=hours_leaf,
        sources=hours_sources,
        additional_instruction=(
            "Compare day-by-day hours on the webpage(s) with the provided hours. "
            "Allow minor formatting differences (e.g., 'Mon' vs 'Monday', punctuation, spaces). "
            "If the page shows 'Today's hours' alongside a weekly schedule, accept a match if the weekly schedule aligns. "
            "If multiple pages show hours, a clear match on any one page is sufficient."
        ),
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
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Sephora Stanford Shopping Center store information task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel at root; child node handles critical gating
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

    # Extract structured store info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_store_info(),
        template_class=SephoraStanfordExtraction,
        extraction_name="sephora_stanford_store_info",
        additional_instruction="Focus only on the Sephora store at Stanford Shopping Center, Palo Alto, CA.",
    )

    # Build main critical node as specified by the rubric
    main_node = evaluator.add_parallel(
        id="Sephora_Stanford_Store_Information",
        desc="Verify that complete and accurate information is provided for the Sephora store at Stanford Shopping Center in Palo Alto, California.",
        parent=root,
        critical=True,
    )

    # Verify each major requirement (each sub-node is critical and sequentially gated internally)
    await verify_address(evaluator, main_node, extracted.address)
    await verify_phone(evaluator, main_node, extracted.phone)
    await verify_hours(evaluator, main_node, extracted.hours)

    # Return summary
    return evaluator.get_summary()