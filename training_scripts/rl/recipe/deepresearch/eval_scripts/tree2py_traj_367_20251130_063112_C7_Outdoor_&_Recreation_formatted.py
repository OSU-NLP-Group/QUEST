import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sd_state_parks_info"
TASK_DESCRIPTION = (
    "You are planning a camping trip to California State Parks in San Diego County and need to gather "
    "specific information to compare options and budget your trip. Please research and provide the following "
    "information:\n\n"
    "For Cuyamaca Rancho State Park:\n"
    "1. What is the standard vehicle day use fee?\n"
    "2. What is the standard campsite fee per night at Paso Picacho Campground?\n"
    "3. What is the physical address of the park?\n"
    "4. What is the contact phone number?\n\n"
    "For Palomar Mountain State Park:\n"
    "5. What is the standard vehicle day use fee?\n"
    "6. What is the standard campsite fee per night at Doane Valley Campground?\n"
    "7. What is the maximum trailer length allowed at the park?\n"
    "8. What is the physical address of the park?\n"
    "9. What is the contact phone number?\n\n"
    "For Anza-Borrego Desert State Park:\n"
    "10. What is the standard vehicle day use fee?\n"
    "11. What is the tent site fee per night at Borrego Palm Canyon Campground?\n"
    "12. What is the location/address of the Visitor Center?\n"
    "13. What are the Visitor Center's operating hours during the peak season (October 1 through May 31)?\n"
    "14. What is the Visitor Center contact phone number?\n\n"
    "California State Parks Pass:\n"
    "15. What is the cost of the California Explorer Vehicle Day Use Annual Pass?\n\n"
    "For each piece of information, please provide the answer along with a reference URL from the official "
    "California State Parks website or relevant official source that supports your answer."
)

# Expected ground truth values (based on rubric requirements)
EXPECTED = {
    # Cuyamaca Rancho SP
    "cuyamaca_day_use_fee": 10,
    "cuyamaca_paso_picacho_fee": 40,
    "cuyamaca_address": "13652 Highway 79, Julian, CA 92036",
    "cuyamaca_phone_digits": "7607653020",
    # Palomar Mountain SP
    "palomar_day_use_fee": 10,
    "palomar_doane_valley_fee": 40,
    "palomar_max_trailer_length_ft": 24,
    "palomar_address": "19952 State Park Drive, Palomar Mountain, CA 92060",
    "palomar_phone_digits": "7607423462",
    # Anza-Borrego Desert SP
    "anza_day_use_fee": 10,
    "borrego_palm_canyon_tent_fee": 35,
    "visitor_center_location_phrase": "west end of palm canyon drive in borrego springs",
    "visitor_center_peak_hours_phrase": "daily 9 am to 5 pm",
    "visitor_center_phone_digits": "7607674205",
    # California Explorer Pass
    "explorer_pass_cost": 195,
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ItemWithSources(BaseModel):
    """One fact value plus its supporting URLs extracted from the answer."""
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ParksExtraction(BaseModel):
    """All requested fields extracted from the answer."""
    # Cuyamaca Rancho SP
    cuyamaca_day_use_fee: Optional[ItemWithSources] = None
    cuyamaca_paso_picacho_fee: Optional[ItemWithSources] = None
    cuyamaca_address: Optional[ItemWithSources] = None
    cuyamaca_phone: Optional[ItemWithSources] = None

    # Palomar Mountain SP
    palomar_day_use_fee: Optional[ItemWithSources] = None
    palomar_doane_valley_fee: Optional[ItemWithSources] = None
    palomar_max_trailer_length: Optional[ItemWithSources] = None
    palomar_address: Optional[ItemWithSources] = None
    palomar_phone: Optional[ItemWithSources] = None

    # Anza-Borrego Desert SP
    anza_day_use_fee: Optional[ItemWithSources] = None
    borrego_palm_canyon_tent_fee: Optional[ItemWithSources] = None
    visitor_center_location: Optional[ItemWithSources] = None
    visitor_center_peak_hours: Optional[ItemWithSources] = None
    visitor_center_phone: Optional[ItemWithSources] = None

    # California Explorer Vehicle Day Use Annual Pass
    explorer_pass_cost: Optional[ItemWithSources] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract the following fields exactly as stated in the answer, plus the list of URLs cited to support each field.
    For each field below, return a JSON object with:
        - value: the answer text for the field, copied exactly from the answer
        - sources: an array of URLs cited in the answer that support this field. Include only URLs (plain or markdown).
          If no URL is present for a field, return an empty array.

    Fields to extract:
    Cuyamaca Rancho State Park:
      - cuyamaca_day_use_fee
      - cuyamaca_paso_picacho_fee
      - cuyamaca_address
      - cuyamaca_phone

    Palomar Mountain State Park:
      - palomar_day_use_fee
      - palomar_doane_valley_fee
      - palomar_max_trailer_length
      - palomar_address
      - palomar_phone

    Anza-Borrego Desert State Park & Visitor Center:
      - anza_day_use_fee
      - borrego_palm_canyon_tent_fee
      - visitor_center_location
      - visitor_center_peak_hours
      - visitor_center_phone

    California Explorer Vehicle Day Use Annual Pass:
      - explorer_pass_cost

    Rules:
    - Do not invent information. If a field is not mentioned, set value to null and sources to [].
    - Extract URLs exactly as they appear in the answer (plain or markdown).
    - If a URL is missing a protocol, prepend http://.
    """


# --------------------------------------------------------------------------- #
# Helper normalization functions                                              #
# --------------------------------------------------------------------------- #
def extract_first_int(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    m = re.search(r"(\d+)", value.replace(",", ""))
    try:
        return int(m.group(1)) if m else None
    except Exception:
        return None


def digits_only_phone(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    digits = re.sub(r"\D", "", value)
    # Keep last 10 digits if longer
    if len(digits) >= 10:
        return digits[-10:]
    return digits if digits else None


def normalize_text(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s2 = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    s2 = re.sub(r"\s+", " ", s2)
    return s2


def text_contains(extracted: Optional[str], expected_phrase: str) -> bool:
    ex_norm = normalize_text(extracted)
    exp_norm = normalize_text(expected_phrase)
    if ex_norm is None or exp_norm is None:
        return False
    return exp_norm in ex_norm


# --------------------------------------------------------------------------- #
# Verification helpers (leaf construction + verify calls)                     #
# --------------------------------------------------------------------------- #
async def verify_numeric_item(
    evaluator: Evaluator,
    parent,
    id_base: str,
    desc_item: str,
    extracted: Optional[ItemWithSources],
    expected_value: int,
    support_claim: str,
):
    """Verify a numeric/fee field: provided, sources provided, value match, and source support."""
    group = evaluator.add_parallel(
        id=id_base,
        desc=desc_item,
        parent=parent,
        critical=True,
    )

    value_str = extracted.value if extracted else None
    srcs = extracted.sources if (extracted and extracted.sources) else []

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=bool(value_str and value_str.strip()),
        id=f"{id_base}_value_provided",
        desc=f"{desc_item} - Value is provided in the answer",
        parent=group,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(srcs) > 0,
        id=f"{id_base}_sources_provided",
        desc=f"{desc_item} - At least one supporting URL is provided",
        parent=group,
        critical=True,
    )

    # Value correctness (critical)
    got_int = extract_first_int(value_str)
    evaluator.add_custom_node(
        result=(got_int == expected_value),
        id=f"{id_base}_value_correct",
        desc=f"{desc_item} - Provided value matches expected ({expected_value})",
        parent=group,
        critical=True,
    )

    # Source support (critical)
    leaf_support = evaluator.add_leaf(
        id=f"{id_base}_source_support",
        desc=f"{desc_item} - Supported by cited source(s)",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim=support_claim,
        node=leaf_support,
        sources=srcs,
        additional_instruction=(
            "Verify the claim strictly against the provided URL(s). Prefer official California State Parks sources "
            "(parks.ca.gov) or other official pages. Allow minor formatting variations (e.g., currency symbols, "
            "spaces). If any of the cited pages clearly supports the claim, mark as supported."
        ),
    )


async def verify_address_item(
    evaluator: Evaluator,
    parent,
    id_base: str,
    desc_item: str,
    extracted: Optional[ItemWithSources],
    expected_address: str,
    support_claim: str,
):
    group = evaluator.add_parallel(
        id=id_base,
        desc=desc_item,
        parent=parent,
        critical=True,
    )
    value_str = extracted.value if extracted else None
    srcs = extracted.sources if (extracted and extracted.sources) else []

    evaluator.add_custom_node(
        result=bool(value_str and value_str.strip()),
        id=f"{id_base}_value_provided",
        desc=f"{desc_item} - Address is provided in the answer",
        parent=group,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(srcs) > 0,
        id=f"{id_base}_sources_provided",
        desc=f"{desc_item} - At least one supporting URL is provided",
        parent=group,
        critical=True,
    )
    evaluator.add_custom_node(
        result=text_contains(value_str, expected_address),
        id=f"{id_base}_value_correct",
        desc=f"{desc_item} - Provided address matches expected",
        parent=group,
        critical=True,
    )

    leaf_support = evaluator.add_leaf(
        id=f"{id_base}_source_support",
        desc=f"{desc_item} - Supported by cited source(s)",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim=support_claim,
        node=leaf_support,
        sources=srcs,
        additional_instruction=(
            "Check the page for the official address text. Allow minor punctuation/casing variations. "
            "If any cited page clearly shows the expected address, mark as supported."
        ),
    )


async def verify_phone_item(
    evaluator: Evaluator,
    parent,
    id_base: str,
    desc_item: str,
    extracted: Optional[ItemWithSources],
    expected_digits: str,
    support_claim: str,
):
    group = evaluator.add_parallel(
        id=id_base,
        desc=desc_item,
        parent=parent,
        critical=True,
    )
    value_str = extracted.value if extracted else None
    srcs = extracted.sources if (extracted and extracted.sources) else []

    evaluator.add_custom_node(
        result=bool(value_str and value_str.strip()),
        id=f"{id_base}_value_provided",
        desc=f"{desc_item} - Phone is provided in the answer",
        parent=group,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(srcs) > 0,
        id=f"{id_base}_sources_provided",
        desc=f"{desc_item} - At least one supporting URL is provided",
        parent=group,
        critical=True,
    )
    got_digits = digits_only_phone(value_str)
    evaluator.add_custom_node(
        result=(got_digits == expected_digits),
        id=f"{id_base}_value_correct",
        desc=f"{desc_item} - Provided phone matches expected",
        parent=group,
        critical=True,
    )

    leaf_support = evaluator.add_leaf(
        id=f"{id_base}_source_support",
        desc=f"{desc_item} - Supported by cited source(s)",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim=support_claim,
        node=leaf_support,
        sources=srcs,
        additional_instruction=(
            "Check the page for the phone number. Minor formatting variations are acceptable; compare digits. "
            "If any cited page clearly shows the expected phone number, mark as supported."
        ),
    )


async def verify_trailer_length_item(
    evaluator: Evaluator,
    parent,
    id_base: str,
    desc_item: str,
    extracted: Optional[ItemWithSources],
    expected_length_ft: int,
    support_claim: str,
):
    group = evaluator.add_parallel(
        id=id_base,
        desc=desc_item,
        parent=parent,
        critical=True,
    )
    value_str = extracted.value if extracted else None
    srcs = extracted.sources if (extracted and extracted.sources) else []

    evaluator.add_custom_node(
        result=bool(value_str and value_str.strip()),
        id=f"{id_base}_value_provided",
        desc=f"{desc_item} - Maximum trailer length is provided",
        parent=group,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(srcs) > 0,
        id=f"{id_base}_sources_provided",
        desc=f"{desc_item} - At least one supporting URL is provided",
        parent=group,
        critical=True,
    )
    got_int = extract_first_int(value_str)
    evaluator.add_custom_node(
        result=(got_int == expected_length_ft),
        id=f"{id_base}_value_correct",
        desc=f"{desc_item} - Provided maximum trailer length matches expected ({expected_length_ft} ft)",
        parent=group,
        critical=True,
    )

    leaf_support = evaluator.add_leaf(
        id=f"{id_base}_source_support",
        desc=f"{desc_item} - Supported by cited source(s)",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim=support_claim,
        node=leaf_support,
        sources=srcs,
        additional_instruction=(
            "Check the page for maximum trailer length restrictions. If any cited page clearly states "
            f"{expected_length_ft} ft maximum, mark as supported."
        ),
    )


async def verify_hours_item(
    evaluator: Evaluator,
    parent,
    id_base: str,
    desc_item: str,
    extracted: Optional[ItemWithSources],
    support_claim: str,
):
    """Hours verification uses LLM for value correctness due to formatting variability."""
    group = evaluator.add_parallel(
        id=id_base,
        desc=desc_item,
        parent=parent,
        critical=True,
    )
    value_str = extracted.value if extracted else None
    srcs = extracted.sources if (extracted and extracted.sources) else []

    evaluator.add_custom_node(
        result=bool(value_str and value_str.strip()),
        id=f"{id_base}_value_provided",
        desc=f"{desc_item} - Hours are provided in the answer",
        parent=group,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(srcs) > 0,
        id=f"{id_base}_sources_provided",
        desc=f"{desc_item} - At least one supporting URL is provided",
        parent=group,
        critical=True,
    )

    # Value correctness via LLM (critical)
    leaf_value = evaluator.add_leaf(
        id=f"{id_base}_value_correct",
        desc=f"{desc_item} - Provided hours match expected",
        parent=group,
        critical=True,
    )
    # The claim itself is the expected hours sentence; LLM checks if the answer states it.
    await evaluator.verify(
        claim=support_claim,
        node=leaf_value,
        additional_instruction=(
            "Judge based only on the answer content. Accept minor formatting variations "
            "(e.g., 9am-5pm vs 9 AM to 5 PM). Ensure the answer clearly indicates these hours for the "
            "peak season (Oct 1–May 31)."
        ),
    )

    # Source support (critical)
    leaf_support = evaluator.add_leaf(
        id=f"{id_base}_source_support",
        desc=f"{desc_item} - Supported by cited source(s)",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim=support_claim,
        node=leaf_support,
        sources=srcs,
        additional_instruction=(
            "Verify the hours and season strictly against the provided URL(s). Prefer official California "
            "State Parks sources. Accept minor formatting variations."
        ),
    )


async def verify_location_item(
    evaluator: Evaluator,
    parent,
    id_base: str,
    desc_item: str,
    extracted: Optional[ItemWithSources],
    support_claim: str,
):
    """Visitor Center location phrased claim; value correctness via LLM due to phrasing variability."""
    group = evaluator.add_parallel(
        id=id_base,
        desc=desc_item,
        parent=parent,
        critical=True,
    )
    value_str = extracted.value if extracted else None
    srcs = extracted.sources if (extracted and extracted.sources) else []

    evaluator.add_custom_node(
        result=bool(value_str and value_str.strip()),
        id=f"{id_base}_value_provided",
        desc=f"{desc_item} - Location/address is provided in the answer",
        parent=group,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(srcs) > 0,
        id=f"{id_base}_sources_provided",
        desc=f"{desc_item} - At least one supporting URL is provided",
        parent=group,
        critical=True,
    )

    # Value correctness via LLM (critical)
    leaf_value = evaluator.add_leaf(
        id=f"{id_base}_value_correct",
        desc=f"{desc_item} - Provided location matches expected phrasing",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim=support_claim,
        node=leaf_value,
        additional_instruction=(
            "Judge based only on the answer content. The expected statement is that the Visitor Center is at "
            "the west end of Palm Canyon Drive in Borrego Springs. Accept minor rephrasings."
        ),
    )

    # Source support (critical)
    leaf_support = evaluator.add_leaf(
        id=f"{id_base}_source_support",
        desc=f"{desc_item} - Supported by cited source(s)",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim=support_claim,
        node=leaf_support,
        sources=srcs,
        additional_instruction=(
            "Verify the location strictly against the provided URL(s). Prefer official California State Parks sources."
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
) -> Dict:
    """
    Evaluate the agent's answer for San Diego County California State Parks information and citations.
    Returns the standard summary dictionary from the evaluator.
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
        default_model=model,
    )

    # Extract all fields from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=ParksExtraction,
        extraction_name="parks_extraction",
    )

    # Ground truth info
    evaluator.add_ground_truth(
        {
            "expected_values": EXPECTED,
            "note": "Rubric-required values used as ground truth for evaluation. Minor formatting variations allowed in verification.",
        },
        gt_type="ground_truth",
    )

    # Build park group nodes (critical)
    cuyamaca_node = evaluator.add_parallel(
        id="Cuyamaca_Rancho_State_Park",
        desc="All requested information for Cuyamaca Rancho State Park is provided with supporting official citations.",
        parent=root,
        critical=True,
    )
    palomar_node = evaluator.add_parallel(
        id="Palomar_Mountain_State_Park",
        desc="All requested information for Palomar Mountain State Park is provided with supporting official citations.",
        parent=root,
        critical=True,
    )
    anza_node = evaluator.add_parallel(
        id="Anza_Borrego_Desert_State_Park",
        desc="All requested information for Anza-Borrego Desert State Park and its Visitor Center is provided with supporting official citations.",
        parent=root,
        critical=True,
    )
    pass_node = evaluator.add_parallel(
        id="California_Explorer_Vehicle_Day_Use_Annual_Pass",
        desc="The requested California State Parks pass cost is provided with a supporting official citation.",
        parent=root,
        critical=True,
    )

    # ---------------------- Cuyamaca Rancho State Park ----------------------
    await verify_numeric_item(
        evaluator,
        cuyamaca_node,
        "Cuyamaca_Day_Use_Fee_With_Citation",
        "Cuyamaca Rancho SP: Standard vehicle day use fee as $10 with citation",
        extraction.cuyamaca_day_use_fee,
        EXPECTED["cuyamaca_day_use_fee"],
        "The standard vehicle day use fee at Cuyamaca Rancho State Park is $10.",
    )

    await verify_numeric_item(
        evaluator,
        cuyamaca_node,
        "Cuyamaca_Paso_Picacho_Campsite_Fee_With_Citation",
        "Cuyamaca Rancho SP: Paso Picacho Campground standard campsite fee per night as $40 with citation",
        extraction.cuyamaca_paso_picacho_fee,
        EXPECTED["cuyamaca_paso_picacho_fee"],
        "The standard campsite fee per night at Paso Picacho Campground (Cuyamaca Rancho SP) is $40.",
    )

    await verify_address_item(
        evaluator,
        cuyamaca_node,
        "Cuyamaca_Park_Address_With_Citation",
        "Cuyamaca Rancho SP: Physical address with citation",
        extraction.cuyamaca_address,
        EXPECTED["cuyamaca_address"],
        "The physical address of Cuyamaca Rancho State Park is 13652 Highway 79, Julian, CA 92036.",
    )

    await verify_phone_item(
        evaluator,
        cuyamaca_node,
        "Cuyamaca_Park_Phone_With_Citation",
        "Cuyamaca Rancho SP: Contact phone with citation",
        extraction.cuyamaca_phone,
        EXPECTED["cuyamaca_phone_digits"],
        "The contact phone number for Cuyamaca Rancho State Park is (760) 765-3020.",
    )

    # ---------------------- Palomar Mountain State Park ----------------------
    await verify_numeric_item(
        evaluator,
        palomar_node,
        "Palomar_Day_Use_Fee_With_Citation",
        "Palomar Mountain SP: Standard vehicle day use fee as $10 with citation",
        extraction.palomar_day_use_fee,
        EXPECTED["palomar_day_use_fee"],
        "The standard vehicle day use fee at Palomar Mountain State Park is $10.",
    )

    await verify_numeric_item(
        evaluator,
        palomar_node,
        "Palomar_Doane_Valley_Campsite_Fee_With_Citation",
        "Palomar Mountain SP: Doane Valley Campground standard campsite fee per night as $40 with citation",
        extraction.palomar_doane_valley_fee,
        EXPECTED["palomar_doane_valley_fee"],
        "The standard campsite fee per night at Doane Valley Campground (Palomar Mountain SP) is $40.",
    )

    await verify_trailer_length_item(
        evaluator,
        palomar_node,
        "Palomar_Max_Trailer_Length_With_Citation",
        "Palomar Mountain SP: Maximum trailer length with citation",
        extraction.palomar_max_trailer_length,
        EXPECTED["palomar_max_trailer_length_ft"],
        "The maximum trailer length allowed at Palomar Mountain State Park is 24 feet.",
    )

    await verify_address_item(
        evaluator,
        palomar_node,
        "Palomar_Park_Address_With_Citation",
        "Palomar Mountain SP: Physical address with citation",
        extraction.palomar_address,
        EXPECTED["palomar_address"],
        "The physical address of Palomar Mountain State Park is 19952 State Park Drive, Palomar Mountain, CA 92060.",
    )

    await verify_phone_item(
        evaluator,
        palomar_node,
        "Palomar_Park_Phone_With_Citation",
        "Palomar Mountain SP: Contact phone with citation",
        extraction.palomar_phone,
        EXPECTED["palomar_phone_digits"],
        "The contact phone number for Palomar Mountain State Park is (760) 742-3462.",
    )

    # ---------------------- Anza-Borrego Desert State Park -------------------
    await verify_numeric_item(
        evaluator,
        anza_node,
        "Anza_Borrego_Day_Use_Fee_With_Citation",
        "Anza-Borrego Desert SP: Standard vehicle day use fee as $10 with citation",
        extraction.anza_day_use_fee,
        EXPECTED["anza_day_use_fee"],
        "The standard vehicle day use fee at Anza-Borrego Desert State Park is $10.",
    )

    await verify_numeric_item(
        evaluator,
        anza_node,
        "Borrego_Palm_Canyon_Tent_Site_Fee_With_Citation",
        "Anza-Borrego Desert SP: Borrego Palm Canyon Campground tent site fee per night as $35 with citation",
        extraction.borrego_palm_canyon_tent_fee,
        EXPECTED["borrego_palm_canyon_tent_fee"],
        "The tent site fee per night at Borrego Palm Canyon Campground is $35.",
    )

    await verify_location_item(
        evaluator,
        anza_node,
        "Visitor_Center_Location_With_Citation",
        "Anza-Borrego Desert SP: Visitor Center location/address with citation",
        extraction.visitor_center_location,
        "The Visitor Center is located at the west end of Palm Canyon Drive in Borrego Springs.",
    )

    await verify_hours_item(
        evaluator,
        anza_node,
        "Visitor_Center_Peak_Season_Hours_With_Citation",
        "Anza-Borrego Desert SP: Visitor Center peak season hours with citation",
        extraction.visitor_center_peak_hours,
        "During peak season (October 1 through May 31), the Anza-Borrego Desert State Park Visitor Center is open daily 9 AM to 5 PM.",
    )

    await verify_phone_item(
        evaluator,
        anza_node,
        "Visitor_Center_Phone_With_Citation",
        "Anza-Borrego Desert SP: Visitor Center contact phone with citation",
        extraction.visitor_center_phone,
        EXPECTED["visitor_center_phone_digits"],
        "The Anza-Borrego Desert State Park Visitor Center contact phone number is (760) 767-4205.",
    )

    # ---------------------- California Explorer Pass -------------------------
    await verify_numeric_item(
        evaluator,
        pass_node,
        "Explorer_Pass_Cost_With_Citation",
        "California Explorer Vehicle Day Use Annual Pass: Cost with citation",
        extraction.explorer_pass_cost,
        EXPECTED["explorer_pass_cost"],
        "The California Explorer Vehicle Day Use Annual Pass costs $195.",
    )

    # Return the evaluation summary
    return evaluator.get_summary()