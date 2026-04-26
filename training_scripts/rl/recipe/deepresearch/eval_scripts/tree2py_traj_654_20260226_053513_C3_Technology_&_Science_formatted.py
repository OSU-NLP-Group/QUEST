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
TASK_ID = "laptop_battery_recall_2019"
TASK_DESCRIPTION = """
A laptop computer manufacturer headquartered in the United States announced or expanded a safety recall program for embedded lithium-ion batteries in notebook computers during 2019. The batteries affected by this recall are not customer-replaceable and must be serviced by an authorized technician. The recall program requires users to download and run a specific validation utility software tool to determine if their battery is affected (rather than simply entering a serial number on a webpage). The official recall documentation instructs users to enable a 'Battery Safety Mode' on their device while waiting for the free battery replacement service. The recall affects multiple product series or model families from this manufacturer. What is the official URL of this battery recall program?
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RecallAnswerExtraction(BaseModel):
    manufacturer: Optional[str] = None
    headquarters_country: Optional[str] = None
    official_url: Optional[str] = None
    supporting_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_recall_answer_data() -> str:
    return """
Extract the following information from the answer about the laptop battery recall:

Fields to extract:
1) manufacturer: The name of the laptop manufacturer (e.g., "HP", "HP Inc.", "Dell", "Apple"). Extract exactly as presented in the answer.
2) headquarters_country: The country where the manufacturer is headquartered, if explicitly mentioned in the answer (e.g., "United States"). If not mentioned, return null.
3) official_url: The single official recall program URL cited in the answer that best represents the recall program details/announcement (not a generic support homepage, and not a news article). If multiple URLs are present, choose the one that is the dedicated recall program page describing the program, instructions, and replacement process. If none is present, return null.
4) supporting_urls: A list of all other URLs provided in the answer that are related to this recall (e.g., download tool pages, product lists, press releases, CPSC notices, company “about” page). Include every relevant URL mentioned in the answer except the official_url (do not duplicate). Preserve them as full URLs.

Important:
- Only extract URLs explicitly present in the answer (plain URLs or markdown links). Do not invent URLs.
- If a URL is missing a protocol, prepend http://
- If any field is missing from the answer, set it to null (or empty list for supporting_urls).
"""


# --------------------------------------------------------------------------- #
# Helper to build URL lists                                                   #
# --------------------------------------------------------------------------- #
def combine_urls(primary: Optional[str], extras: List[str]) -> List[str]:
    urls = []
    if primary and primary.strip():
        urls.append(primary.strip())
    urls.extend([u for u in extras if isinstance(u, str) and u.strip()])
    return urls


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_battery_recall_tree(
    evaluator: Evaluator,
    root_node,
    extracted: RecallAnswerExtraction
) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    Root node uses sequential aggregation (already set in initialize), so later
    stages will be skipped if earlier stages fail.
    """

    manufacturer = (extracted.manufacturer or "").strip()
    hq_country = (extracted.headquarters_country or "").strip()
    official_url = (extracted.official_url or "").strip()
    supporting_urls = extracted.supporting_urls or []
    all_urls = combine_urls(official_url, supporting_urls)

    # -------------------- Manufacturer_Verification (Parallel) --------------------
    manu_node = evaluator.add_parallel(
        id="manufacturer_verification",
        desc="Verify the manufacturer identity and headquarters location",
        parent=root_node,
        critical=False
    )

    # Manufacturer Identity (Critical)
    manu_identity_leaf = evaluator.add_leaf(
        id="manufacturer_identity",
        desc="Correctly identify the manufacturer of the recalled laptop batteries",
        parent=manu_node,
        critical=True
    )
    manu_claim = (
        f"This webpage is an official recall program page from the manufacturer '{manufacturer}'. "
        f"Confirm that the page belongs to this manufacturer (allow 'Inc.'/'Co.' naming variants) "
        f"and clearly shows the manufacturer's identity."
    )
    await evaluator.verify(
        claim=manu_claim,
        node=manu_identity_leaf,
        sources=official_url if official_url else None,
        additional_instruction="Accept minor naming variants (e.g., 'HP' vs 'HP Inc.' or 'Hewlett-Packard'). Confirm via on-page branding, headers, or domain ownership."
    )

    # US Headquarters (Critical)
    hq_leaf = evaluator.add_leaf(
        id="us_headquarters",
        desc="Verify that the manufacturer is headquartered in the United States",
        parent=manu_node,
        critical=True
    )
    hq_claim = (
        f"The manufacturer '{manufacturer}' is headquartered in the United States."
    )
    await evaluator.verify(
        claim=hq_claim,
        node=hq_leaf,
        sources=all_urls if all_urls else None,
        additional_instruction="If the recall page itself does not state HQ location, use other cited sources provided in the answer (e.g., company 'About' page, Wikipedia, press releases) to confirm U.S. headquarters."
    )

    # ---------------- Recall_Timeframe_And_Scope (Parallel) ----------------------
    timeframe_scope_node = evaluator.add_parallel(
        id="recall_timeframe_scope",
        desc="Verify the recall announcement timeframe and scope of affected products",
        parent=root_node,
        critical=False
    )

    # Recall Year 2019 (Critical)
    recall_2019_leaf = evaluator.add_leaf(
        id="recall_year_2019",
        desc="Verify that the recall was announced or expanded between January 1, 2019 and December 31, 2019 (inclusive)",
        parent=timeframe_scope_node,
        critical=True
    )
    recall_2019_claim = (
        "The recall program for embedded laptop batteries was announced or expanded during the 2019 calendar year (between January 1, 2019 and December 31, 2019, inclusive)."
    )
    await evaluator.verify(
        claim=recall_2019_claim,
        node=recall_2019_leaf,
        sources=all_urls if all_urls else None,
        additional_instruction="Look for dates on the official recall page or linked official announcements/press releases confirming the recall was announced or expanded in 2019."
    )

    # Multiple Product Series (Critical)
    multiple_series_leaf = evaluator.add_leaf(
        id="multiple_product_series",
        desc="Verify that the recall affects multiple product series or model families",
        parent=timeframe_scope_node,
        critical=True
    )
    series_claim = (
        "The recall affects multiple product series or model families (i.e., more than one distinct series/family is listed as affected)."
    )
    await evaluator.verify(
        claim=series_claim,
        node=multiple_series_leaf,
        sources=official_url if official_url else None,
        additional_instruction="Check the affected products section; passing examples include multiple families like 'ProBook', 'ZBook', 'Pavilion', etc., or multiple distinct series/model families listed."
    )

    # -------------------- Battery_Characteristics (Parallel) ---------------------
    battery_char_node = evaluator.add_parallel(
        id="battery_characteristics",
        desc="Verify the battery type and replacement requirements",
        parent=root_node,
        critical=False
    )

    # Embedded Battery (Critical)
    embedded_leaf = evaluator.add_leaf(
        id="embedded_battery",
        desc="Verify that the batteries are embedded/internal and not customer-replaceable according to official documentation",
        parent=battery_char_node,
        critical=True
    )
    embedded_claim = (
        "According to the official recall documentation, the affected batteries are embedded/internal and are NOT customer-replaceable."
    )
    await evaluator.verify(
        claim=embedded_claim,
        node=embedded_leaf,
        sources=official_url if official_url else None,
        additional_instruction="Look for language like 'embedded', 'internal', 'not customer replaceable', or similar phrasing."
    )

    # Authorized Technician Service (Critical)
    authorized_service_leaf = evaluator.add_leaf(
        id="authorized_technician_service",
        desc="Verify that battery replacement must be performed by an authorized technician",
        parent=battery_char_node,
        critical=True
    )
    technician_claim = (
        "The recall documentation requires that battery replacement must be performed by an authorized technician/service provider (not self-service)."
    )
    await evaluator.verify(
        claim=technician_claim,
        node=authorized_service_leaf,
        sources=official_url if official_url else None,
        additional_instruction="Look for directives indicating only authorized service providers or technicians should replace the battery."
    )

    # ----------------------- Program_Features (Parallel) -------------------------
    program_features_node = evaluator.add_parallel(
        id="program_features",
        desc="Verify the recall program's validation method and safety features",
        parent=root_node,
        critical=False
    )

    # Validation Utility Required (Critical)
    validation_tool_leaf = evaluator.add_leaf(
        id="validation_utility_required",
        desc="Verify that the program requires users to download and run a validation utility software tool (not just web-based serial number entry)",
        parent=program_features_node,
        critical=True
    )
    validation_claim = (
        "The recall program requires users to download and run a validation utility software tool to determine if their battery is affected (rather than only using a web-based serial number entry)."
    )
    await evaluator.verify(
        claim=validation_claim,
        node=validation_tool_leaf,
        sources=official_url if official_url else None,
        additional_instruction="Look for explicit instructions to download and run a 'Battery Program Validation Utility' (or similar named tool). Distinguish from simple serial-number web form checks."
    )

    # Battery Safety Mode (Critical)
    safety_mode_leaf = evaluator.add_leaf(
        id="battery_safety_mode",
        desc="Verify that the official recall documentation explicitly mentions a 'Battery Safety Mode' feature that users should enable",
        parent=program_features_node,
        critical=True
    )
    safety_mode_claim = (
        "The official recall documentation explicitly instructs users to enable a 'Battery Safety Mode' while awaiting the free battery replacement service."
    )
    await evaluator.verify(
        claim=safety_mode_claim,
        node=safety_mode_leaf,
        sources=official_url if official_url else None,
        additional_instruction="Look for the exact phrase 'Battery Safety Mode' (or a very close variant) and instructions to enable it during the waiting period."
    )

    # ------------------------------ Official_URL ---------------------------------
    official_url_leaf = evaluator.add_leaf(
        id="official_url",
        desc="Provide the correct official recall program URL as documented in the official recall announcement",
        parent=root_node,
        critical=True
    )
    url_claim = (
        "This URL is the manufacturer's official recall program page for embedded lithium-ion batteries in notebook computers, describing the program and instructions."
    )
    await evaluator.verify(
        claim=url_claim,
        node=official_url_leaf,
        sources=official_url if official_url else None,
        additional_instruction="Confirm that the page is an official recall program page (not a third-party news article), and that it matches the described program details (validation utility, Battery Safety Mode, authorized service, multiple families)."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the 2019 embedded battery recall identification task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Follow rubric: sequential root (logical order)
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_recall_answer_data(),
        template_class=RecallAnswerExtraction,
        extraction_name="recall_answer_data"
    )

    # Record constraints as custom info for transparency
    evaluator.add_custom_info(
        info={
            "must_be_us_headquartered": True,
            "year": 2019,
            "embedded_non_customer_replaceable": True,
            "authorized_technician_required": True,
            "requires_validation_utility": True,
            "mentions_battery_safety_mode": True,
            "affects_multiple_series": True
        },
        info_type="constraints",
        info_name="required_constraints"
    )

    # Build and verify the rubric-based checks
    await build_and_verify_battery_recall_tree(evaluator, root, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()