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
TASK_ID = "compact_12v_drill_kit_model"
TASK_DESCRIPTION = (
    "What is the full model number of a compact 12V cordless drill kit that meets all of the following specifications: "
    "the drill must have a tool-only weight of 2.0 pounds or less, feature a 3/8-inch chuck size, include a brushless motor, "
    "have 15 or more adjustable clutch settings, include an integrated LED work light, come with a belt clip, and the kit must include "
    "2 or more batteries rated at 2.0Ah or higher capacity along with a battery charger?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DrillKitExtraction(BaseModel):
    """Structured information for the identified compact 12V drill kit."""
    name: Optional[str] = None
    brand: Optional[str] = None
    model_number: Optional[str] = None  # Full model identifier/string
    product_type: Optional[str] = None  # e.g., "cordless drill kit", "tool-only", "corded"
    voltage: Optional[str] = None       # e.g., "12V", "12V Max", "10.8V nominal"
    compact: Optional[str] = None       # phrase or indication that product is marketed as compact/subcompact
    weight_tool_only: Optional[str] = None  # tool-only/bare-tool weight as stated
    chuck_size: Optional[str] = None
    motor_type: Optional[str] = None    # e.g., "brushless", "brushed"
    clutch_settings: Optional[str] = None  # e.g., "16+1", "15 settings"
    led_light: Optional[str] = None     # phrase indicating integrated LED
    belt_clip: Optional[str] = None     # phrase indicating belt clip included
    battery_count: Optional[str] = None # phrase or number indicating included battery quantity
    battery_capacity: Optional[str] = None  # capacity per included battery (e.g., "2.0Ah", "2 x 2.0Ah")
    charger_included: Optional[str] = None  # phrase indicating charger included
    sources: List[str] = Field(default_factory=list)  # all URLs explicitly present in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_drill_kit_info() -> str:
    return (
        "From the provided answer, extract details for the single compact 12V cordless drill kit that the answer identifies as meeting "
        "the requirements. Extract only what is explicitly present in the answer. Do not invent or infer information.\n"
        "Return a JSON object with the following fields:\n"
        "- name: The product title/name as given in the answer\n"
        "- brand: The brand/manufacturer, if mentioned\n"
        "- model_number: The full model number/identifier string (not just the brand or product line)\n"
        "- product_type: A short phrase indicating whether it is a 'cordless drill kit', 'tool-only', 'corded', etc.\n"
        "- voltage: The voltage system, e.g., '12V', '12V Max', or '10.8V nominal'\n"
        "- compact: The exact phrase or word(s) showing it is marketed as compact (e.g., 'compact' or 'subcompact'), if mentioned\n"
        "- weight_tool_only: The tool-only (bare tool) weight if provided; if only a weight with battery is mentioned in the answer, still extract that string but include the phrase indicating context\n"
        "- chuck_size: The chuck size as stated (e.g., '3/8 in', '3/8-inch')\n"
        "- motor_type: 'brushless' or 'brushed' if explicitly stated\n"
        "- clutch_settings: The number of clutch/torque settings or a phrase like '16+1'; extract the exact text as provided\n"
        "- led_light: The phrase indicating an integrated LED work light if present\n"
        "- belt_clip: The phrase indicating that a belt clip is included, if present\n"
        "- battery_count: The quantity of batteries included in the kit as stated (e.g., '2 batteries')\n"
        "- battery_capacity: The capacity rating per included battery (e.g., '2.0Ah each', '2 x 2.0Ah'), if present\n"
        "- charger_included: The phrase indicating a charger is included, if present\n"
        "- sources: An array of all URLs explicitly shown in the answer (plain URLs or markdown links). Only include URLs actually present in the answer text.\n"
        "If any field is not mentioned, set it to null. For URLs, follow the special rules for URL extraction and return an empty array if none are provided."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_full_model_number(model_number: Optional[str], brand: Optional[str]) -> bool:
    """
    Heuristic to judge whether a model string looks like a full model identifier (not just brand or line).
    Rules:
      - Must be a non-empty string.
      - Should contain at least one digit OR a hyphen/slash common in model codes.
      - If brand exists and equals the model string (case-insensitive), treat as not a full model number.
    """
    if not model_number or not model_number.strip():
        return False
    s = model_number.strip()
    has_code_char = any(ch.isdigit() for ch in s) or ("-" in s) or ("/" in s)
    if not has_code_char:
        return False
    if brand and brand.strip().lower() == s.lower():
        return False
    return True


def _collect_sources(extracted: DrillKitExtraction) -> List[str]:
    """Return list of source URLs extracted from the answer."""
    return extracted.sources if extracted and extracted.sources else []


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify(
    evaluator: Evaluator,
    root_node,
    extracted: DrillKitExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """
    # Create a critical parallel node representing the entire kit compliance
    main_node = evaluator.add_parallel(
        id="compact_12v_drill_kit",
        desc="Identifies a compact 12V cordless drill kit and provides its full model number, meeting all specified feature and kit-inclusion requirements",
        parent=root_node,
        critical=True
    )

    # Sources for verification
    sources_list = _collect_sources(extracted)

    # 1) Provides full model number (custom check as a leaf)
    evaluator.add_custom_node(
        result=_is_full_model_number(extracted.model_number, extracted.brand),
        id="provides_full_model_number",
        desc="The answer includes the drill kit's full model number/identifier (not just brand or product line)",
        parent=main_node,
        critical=True
    )

    # 2) Cordless drill kit (not corded; not tool-only)
    node_cordless_kit = evaluator.add_leaf(
        id="cordless_drill_kit",
        desc="The identified product is a cordless drill kit (not a corded drill and not tool-only without being a kit)",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="This product listing is for a cordless drill kit (battery-powered) and is not corded or tool-only.",
        node=node_cordless_kit,
        sources=sources_list,
        additional_instruction=(
            "Confirm that the product is battery-powered (cordless) and sold as a kit (with included components). "
            "If the listing clearly indicates 'tool-only' or 'bare tool' without being a kit, or it is corded, then it fails this check."
        ),
    )

    # 3) Compact claim
    node_compact = evaluator.add_leaf(
        id="compact_claim",
        desc="The identified product is explicitly described/marketed as compact in the product information",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The product is explicitly marketed or described as 'compact' or 'subcompact' in its official product or retailer listing.",
        node=node_compact,
        sources=sources_list,
        additional_instruction=(
            "Look for the word 'compact' or closely related terms such as 'subcompact' in titles, bullets, or descriptions. "
            "Generic small size without the explicit compact claim does not qualify."
        ),
    )

    # 4) Voltage specification = 12V system
    node_voltage = evaluator.add_leaf(
        id="voltage_specification",
        desc="The drill operates on a 12V battery system",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The drill operates on a 12V system.",
        node=node_voltage,
        sources=sources_list,
        additional_instruction=(
            "Allow synonyms like '12V Max' or '10.8V nominal' which commonly correspond to 12V class systems. "
            "If the listing shows any other voltage class (e.g., 18V/20V), then fail."
        ),
    )

    # 5) Tool-only weight <= 2.0 lb
    node_weight = evaluator.add_leaf(
        id="tool_weight",
        desc="The tool-only weight is 2.0 pounds or less",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The tool-only (bare tool) weight is less than or equal to 2.0 pounds.",
        node=node_weight,
        sources=sources_list,
        additional_instruction=(
            "Prefer explicit 'tool-only' or 'bare tool' weight specs. "
            "If only a 'with battery' weight is given and it exceeds 2.0 lb, do not count it as tool-only. "
            "If the page clearly states the bare tool weight ≤ 2.0 lb, pass."
        ),
    )

    # 6) Chuck size = 3/8-inch
    node_chuck = evaluator.add_leaf(
        id="chuck_size",
        desc="The drill has a 3/8-inch chuck size",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The drill has a 3/8-inch chuck.",
        node=node_chuck,
        sources=sources_list,
        additional_instruction="Allow variants like '3/8 in' or '0.375 inch'; it must not be 1/2-inch or other sizes."
    )

    # 7) Motor type = brushless
    node_motor = evaluator.add_leaf(
        id="motor_type",
        desc="The drill features a brushless motor",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The drill uses a brushless motor.",
        node=node_motor,
        sources=sources_list,
        additional_instruction="The page must explicitly state 'brushless'; if it states 'brushed' or no mention, fail."
    )

    # 8) Clutch settings >= 15
    node_clutch = evaluator.add_leaf(
        id="clutch_settings",
        desc="The drill has 15 or more adjustable clutch settings",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The drill provides 15 or more adjustable clutch/torque settings.",
        node=node_clutch,
        sources=sources_list,
        additional_instruction=(
            "If the spec shows formats like '16+1' (16 clutch positions plus drill mode), count the clutch positions for the threshold. "
            "Values like 15, 16, 18, etc. qualify; fewer than 15 fails."
        ),
    )

    # 9) Integrated LED work light
    node_led = evaluator.add_leaf(
        id="led_light",
        desc="The drill includes an integrated LED work light",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The drill includes an integrated LED work light.",
        node=node_led,
        sources=sources_list,
        additional_instruction="Confirm that the product page mentions a built-in or integrated LED light feature."
    )

    # 10) Belt clip included
    node_belt = evaluator.add_leaf(
        id="belt_clip",
        desc="A belt clip is included with the drill",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The kit includes a belt clip for the drill.",
        node=node_belt,
        sources=sources_list,
        additional_instruction=(
            "The belt clip must be included as part of the kit contents or accessories; 'compatible belt clip sold separately' does not qualify."
        )
    )

    # 11) Battery quantity >= 2
    node_batt_qty = evaluator.add_leaf(
        id="battery_quantity",
        desc="The kit includes 2 or more batteries",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The kit includes two or more batteries.",
        node=node_batt_qty,
        sources=sources_list,
        additional_instruction="Look for explicit mentions like '2 batteries', 'two 12V batteries', or similar; a single battery fails."
    )

    # 12) Battery capacity >= 2.0Ah
    node_batt_cap = evaluator.add_leaf(
        id="battery_capacity",
        desc="The included batteries are rated at 2.0Ah or higher",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="Each included battery has a capacity rating of at least 2.0Ah.",
        node=node_batt_cap,
        sources=sources_list,
        additional_instruction=(
            "If the listing shows capacities like 2.0Ah, 2.5Ah, 3.0Ah, or 4.0Ah, these qualify. "
            "If any included battery is rated below 2.0Ah (e.g., 1.5Ah), fail."
        ),
    )

    # 13) Charger included
    node_charger = evaluator.add_leaf(
        id="charger_included",
        desc="The kit includes a battery charger",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The kit includes a battery charger.",
        node=node_charger,
        sources=sources_list,
        additional_instruction="The included items list or package contents must show a charger; absence of charger fails."
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
    """
    Evaluate an answer for the compact 12V drill kit model task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation; rubric uses parallel
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_drill_kit_info(),
        template_class=DrillKitExtraction,
        extraction_name="drill_kit_candidate"
    )

    # Record constraints as ground truth info for traceability
    evaluator.add_ground_truth({
        "required_specs": {
            "voltage": "12V system",
            "tool_only_weight_max_lb": 2.0,
            "chuck_size": "3/8-inch",
            "motor_type": "brushless",
            "clutch_settings_min": 15,
            "led_light": "integrated",
            "belt_clip": "included",
            "battery_quantity_min": 2,
            "battery_capacity_min_Ah": 2.0,
            "charger_included": True
        }
    }, gt_type="constraints")

    # Build and run verifications
    await build_and_verify(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()