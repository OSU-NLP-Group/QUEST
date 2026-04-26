import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "austin_office_space_single_property"
TASK_DESCRIPTION = (
    "I am relocating my company to Austin, Texas and need to find office space for my team of 20 employees. "
    "Based on standard office space planning guidelines of approximately 150 square feet per employee, I need at least 3,000 square feet of office space. "
    "My budget allows for a maximum annual lease rate of $45 per square foot.\n\n"
    "Please identify one available office property in downtown Austin, Texas that meets these requirements. For the property, provide the following details:\n"
    "- Complete property address\n"
    "- Building name (if applicable)\n"
    "- Total square footage of the available space\n"
    "- Annual lease rate per square foot\n"
    "- A direct URL link to the property listing"
)

MIN_SQFT = 3000
MAX_RENT_PER_SF_YR = 45.0


# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class PropertyItem(BaseModel):
    building_name: Optional[str] = None
    # If the answer explicitly states "not applicable", "N/A", or "unknown" for building name,
    # extract that literal note here.
    building_name_note: Optional[str] = None

    address: Optional[str] = None
    available_sqft: Optional[str] = None
    annual_lease_rate_per_sf: Optional[str] = None
    listing_url: Optional[str] = None

    # Optional extra signals from the answer text, if present
    availability_text: Optional[str] = None
    neighborhood_or_area: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None


class PropertiesExtraction(BaseModel):
    properties: List[PropertyItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_properties() -> str:
    return """
    Extract every distinct office property that the answer identifies. For each property, extract exactly these fields:

    - building_name: The building name, exactly as written in the answer text, if one is provided alongside the selected property. If the answer uses a phrase indicating not applicable or unknown (e.g., "N/A", "not applicable", "unknown"), still set building_name to that literal text if it appears in the answer as the building name value. If the answer does not mention a building name or such a phrase for the building name, return null.
    - building_name_note: If the answer explicitly notes that a building name is not applicable or unknown (e.g., "N/A", "unknown", "not applicable"), extract that phrase here. Otherwise, return null.
    - address: The complete property address as presented in the answer (street, city, state, and zip code if present). If not provided, return null.
    - available_sqft: The total square footage of the available space for the property as written, keeping any units or ranges as text (e.g., "3,200 SF", "3k-5k SF"). If not present, return null.
    - annual_lease_rate_per_sf: The lease rate per square foot per year as written (e.g., "$42/SF/yr", "$3.50/SF/mo", "$40 NNN"). Keep the text exactly, do not convert. If not present, return null.
    - listing_url: A direct URL to the property listing if provided in the answer. If more than one URL is given for a single property, include the one that appears to be the primary listing page. If no listing URL is provided, return null.
    - availability_text: Any availability wording from the answer (e.g., "Available", "For Lease", "Leased", "Off Market") if present; otherwise null.
    - neighborhood_or_area: Any area/neighborhood tag used in the answer for the property (e.g., "Downtown", "CBD"), if present; otherwise null.
    - city: The city (if provided).
    - state: The state (if provided).
    - zip_code: The postal code if provided (e.g., "78701").

    Important rules:
    - Extract only information explicitly present in the answer text. Do not infer or invent values.
    - If an item is missing from the answer, return null for that field.
    - If a URL is missing a protocol (http/https) in the answer, prepend "http://".
    - Return a JSON object with one top-level field: "properties", which is an array of objects (one per property).
    """


# --------------------------------------------------------------------------- #
# Helper Utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_na_like(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    return any(kw in t for kw in ["n/a", "na", "not applicable", "unknown", "none"])


def _get_selected_property(extraction: PropertiesExtraction) -> PropertyItem:
    # Always select the first property if present; if none, return an empty PropertyItem.
    return extraction.properties[0] if extraction.properties else PropertyItem()


# --------------------------------------------------------------------------- #
# Verification Tree Construction                                              #
# --------------------------------------------------------------------------- #
async def build_office_property_tree(
    evaluator: Evaluator,
    root_node,
    extracted: PropertiesExtraction,
) -> None:
    """
    Build the verification tree for the Austin office property task and run verifications.
    """
    # Top-level critical node mirroring the rubric's root
    top_node = evaluator.add_parallel(
        id="Office_Property_Response",
        desc="Identify exactly one available office property in downtown Austin that meets constraints and provide required listing details with a verifiable reference URL.",
        parent=root_node,
        critical=True,
    )

    # Determine count and selected property
    prop_count = len(extracted.properties)
    selected = _get_selected_property(extracted)
    selected_url = selected.listing_url if selected and selected.listing_url else None

    # For debugging and clarity, record selected property info
    evaluator.add_custom_info(
        info={
            "extracted_property_count": prop_count,
            "selected_property": selected.dict() if selected else {},
        },
        info_type="extraction_summary",
        info_name="selected_property_info"
    )

    # 1) Exactly One Property Identified (critical leaf)
    evaluator.add_custom_node(
        result=(prop_count == 1),
        id="Exactly_One_Property_Identified",
        desc="Response identifies exactly one office property (not multiple).",
        parent=top_node,
        critical=True
    )

    # 2) Eligibility Constraints (critical parallel)
    eligibility_node = evaluator.add_parallel(
        id="Eligibility_Constraints",
        desc="The selected property meets all stated eligibility constraints.",
        parent=top_node,
        critical=True
    )

    # 2.a) Located in Downtown Austin, TX (critical leaf)
    located_leaf = evaluator.add_leaf(
        id="Located_In_Downtown_Austin_TX",
        desc="Property is located in downtown Austin, Texas.",
        parent=eligibility_node,
        critical=True
    )
    await evaluator.verify(
        claim="The property in the provided listing is located in Downtown Austin, Texas.",
        node=located_leaf,
        sources=selected_url,
        additional_instruction=(
            "Use the listing page to confirm downtown location. Accept synonyms like 'Downtown Austin', 'CBD', "
            "'Central Business District', or a 78701 ZIP code as evidence for downtown. "
            "If the listing's address or tags indicate 78701 or 'Downtown'/'CBD', consider it downtown. "
            "If there is no clear evidence, mark as not supported."
        )
    )

    # 2.b) Meets Minimum Square Footage (critical leaf)
    min_sqft_leaf = evaluator.add_leaf(
        id="Meets_Minimum_Square_Footage",
        desc=f"Available office space is at least {MIN_SQFT:,} square feet.",
        parent=eligibility_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The available office space offered in the listing is at least {MIN_SQFT} square feet in a single contiguous space or in explicitly combinable suites that reach at least {MIN_SQFT} square feet.",
        node=min_sqft_leaf,
        sources=selected_url,
        additional_instruction=(
            "Check the available space sizes on the listing. If multiple suites are listed, pass if there is at least "
            "one single contiguous space >= 3,000 SF OR the listing explicitly states suites can be combined to at least 3,000 SF. "
            "If sizes are given as ranges, consider the minimum value. If only smaller separate suites are present without "
            "explicit combinability to >=3,000, mark as not supported."
        )
    )

    # 2.c) Meets Maximum Lease Rate (critical leaf)
    max_rate_leaf = evaluator.add_leaf(
        id="Meets_Maximum_Lease_Rate",
        desc=f"Annual lease rate is not greater than ${MAX_RENT_PER_SF_YR:.0f} per square foot.",
        parent=eligibility_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The annual lease rate per square foot is not greater than ${MAX_RENT_PER_SF_YR:.0f} per year.",
        node=max_rate_leaf,
        sources=selected_url,
        additional_instruction=(
            "Verify rent per SF per YEAR. If the page shows monthly ($/SF/mo), convert to annual by multiplying by 12. "
            "If a range is shown (e.g., $40-$50/SF/yr), use the upper bound. The claim is supported only if the annual "
            "rate (or converted monthly) is <= $45/SF/yr. If the page says 'Contact for pricing' or does not provide a number, "
            "this should not be considered supported."
        )
    )

    # 2.d) Property Is Available (critical leaf)
    available_leaf = evaluator.add_leaf(
        id="Property_Is_Available_Per_Listing",
        desc="The listing indicates the office space is available (not explicitly unavailable/leased).",
        parent=eligibility_node,
        critical=True
    )
    await evaluator.verify(
        claim="The listing indicates the space is currently available for lease (e.g., 'For Lease', 'Available').",
        node=available_leaf,
        sources=selected_url,
        additional_instruction=(
            "Look for indicators like 'Available', 'For Lease', or an active status. "
            "If the listing shows 'Leased', 'Off Market', or otherwise indicates unavailability, fail. "
            "If unclear, consider it not supported."
        )
    )

    # 3) Required Output Fields Provided (critical parallel)
    required_node = evaluator.add_parallel(
        id="Required_Output_Fields_Provided",
        desc="Response includes all required property details requested in the question.",
        parent=top_node,
        critical=True
    )

    # 3.a) Provides complete property address (critical leaf as custom existence check)
    evaluator.add_custom_node(
        result=bool(selected.address and selected.address.strip()),
        id="Provides_Complete_Property_Address",
        desc="Provides the complete property address.",
        parent=required_node,
        critical=True
    )

    # 3.b) Provides building name if applicable (critical leaf as custom check allowing NA-like notes)
    building_ok = bool(selected.building_name and selected.building_name.strip()) or \
                  _is_na_like(selected.building_name) or \
                  _is_na_like(selected.building_name_note)
    evaluator.add_custom_node(
        result=building_ok,
        id="Provides_Building_Name_If_Applicable",
        desc="Provides the building name if applicable (or explicitly notes if not applicable/unknown).",
        parent=required_node,
        critical=True
    )

    # 3.c) Provides total available square footage (critical leaf custom existence check)
    evaluator.add_custom_node(
        result=bool(selected.available_sqft and selected.available_sqft.strip()),
        id="Provides_Total_Available_Square_Footage",
        desc="Provides the total square footage of the available space being offered.",
        parent=required_node,
        critical=True
    )

    # 3.d) Provides annual lease rate per SF (critical leaf custom existence check)
    evaluator.add_custom_node(
        result=bool(selected.annual_lease_rate_per_sf and selected.annual_lease_rate_per_sf.strip()),
        id="Provides_Annual_Lease_Rate_Per_SF",
        desc="Provides the annual lease rate per square foot.",
        parent=required_node,
        critical=True
    )

    # 3.e) Provides direct listing URL (critical leaf custom existence check with simple validity)
    url_ok = bool(selected_url and isinstance(selected_url, str) and selected_url.strip().lower().startswith(("http://", "https://")))
    evaluator.add_custom_node(
        result=url_ok,
        id="Provides_Direct_Listing_URL",
        desc="Provides a direct URL link to the property listing.",
        parent=required_node,
        critical=True
    )

    # 4) Listing Is Verifiable Per Constraints (critical parallel)
    verifiable_node = evaluator.add_parallel(
        id="Listing_Is_Verifiable_Per_Constraints",
        desc="The provided reference listing is verifiable and contains the required evidence.",
        parent=top_node,
        critical=True
    )

    # 4.a) Listing includes property address (critical leaf; verify against listing URL)
    listing_has_address_leaf = evaluator.add_leaf(
        id="Listing_Includes_Property_Address",
        desc="The referenced listing page includes the property address.",
        parent=verifiable_node,
        critical=True
    )
    await evaluator.verify(
        claim="The listing page explicitly shows the property address.",
        node=listing_has_address_leaf,
        sources=selected_url,
        additional_instruction=(
            f"If available, look for an address matching or reasonably aligning with: '{selected.address}'. "
            "Minor variations (e.g., missing suite numbers) are acceptable. If the page does not list an address, fail."
        )
    )

    # 4.b) Listing includes available square footage (critical leaf; verify against listing URL)
    listing_has_sqft_leaf = evaluator.add_leaf(
        id="Listing_Includes_Available_Square_Footage",
        desc="The referenced listing page includes the available square footage.",
        parent=verifiable_node,
        critical=True
    )
    await evaluator.verify(
        claim="The listing page includes the available square footage for the space.",
        node=listing_has_sqft_leaf,
        sources=selected_url,
        additional_instruction=(
            "Look for any explicit square footage value(s) on the page (e.g., '3,000 SF', '3k SF'). "
            "Ranges or multiple suite sizes count as long as square footage numbers are present."
        )
    )

    # 4.c) Listing includes annual lease rate per SF (critical leaf; verify against listing URL)
    listing_has_rate_leaf = evaluator.add_leaf(
        id="Listing_Includes_Annual_Lease_Rate_Per_SF",
        desc="The referenced listing page includes the lease rate per square foot per year.",
        parent=verifiable_node,
        critical=True
    )
    await evaluator.verify(
        claim="The listing page includes a lease rate per square foot per year (or a value convertible to annual).",
        node=listing_has_rate_leaf,
        sources=selected_url,
        additional_instruction=(
            "Accept explicit $/SF/yr values, or $/SF/mo values that can be converted to annual. "
            "If the page only says 'Contact for pricing' or provides no numeric rent, fail."
        )
    )


# --------------------------------------------------------------------------- #
# Main Evaluation Entry Point                                                 #
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
    Evaluate an answer for the 'single downtown Austin office property' task.
    """
    # Initialize evaluator (root is non-critical; we add a critical child node as per rubric)
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
        default_model=model,
    )

    # Extract the property/properties described in the answer
    extracted_props = await evaluator.extract(
        prompt=prompt_extract_properties(),
        template_class=PropertiesExtraction,
        extraction_name="office_properties_extraction"
    )

    # Build verification tree and run checks
    await build_office_property_tree(evaluator, root, extracted_props)

    # Return standardized summary
    return evaluator.get_summary()