import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "airpods_pro_3_specs"
TASK_DESCRIPTION = (
    "What is the battery life on a single charge with Active Noise Cancellation enabled for the Apple AirPods Pro 3, "
    "what specific heart rate sensor technology and pulse frequency does it use, and what is its official IP rating "
    "for dust and water resistance?"
)

# Expected claims according to rubric
EXPECTED_BATTERY_ANC_CLAIM = (
    "Apple AirPods Pro 3 provide up to 8 hours of listening time on a single charge "
    "with Active Noise Cancellation enabled."
)
EXPECTED_SENSOR_TYPE_CLAIM = (
    "Apple AirPods Pro 3 use a custom photoplethysmography (PPG) heart-rate sensor."
)
EXPECTED_LIGHT_TYPE_CLAIM = (
    "The PPG heart-rate sensor uses invisible infrared light."
)
EXPECTED_PULSE_FREQUENCY_CLAIM = (
    "The infrared light for the heart-rate sensor is pulsed at 256 times per second."
)
EXPECTED_IP_RATING_CLAIM = (
    "The official dust and water resistance rating for Apple AirPods Pro 3 is IP57 (for dust, sweat, and water resistance)."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BatterySpec(BaseModel):
    battery_life_anc_text: Optional[str] = None
    battery_sources: List[str] = Field(default_factory=list)


class HeartRateSpec(BaseModel):
    sensor_type: Optional[str] = None
    light_type: Optional[str] = None
    pulse_frequency: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class IPSpec(BaseModel):
    ip_rating: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AirPodsSpecs(BaseModel):
    battery: Optional[BatterySpec] = None
    heart: Optional[HeartRateSpec] = None
    ip: Optional[IPSpec] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_specs() -> str:
    return """
    Extract the specific specifications for Apple AirPods Pro 3 as stated in the answer. Return a JSON object with the following fields:

    battery:
      - battery_life_anc_text: The exact text the answer provides for battery life on a single charge when Active Noise Cancellation (ANC) is enabled (e.g., "up to 8 hours", "6 hours", etc.). If not stated, return null.
      - battery_sources: An array of URLs cited in the answer that specifically support the battery life with ANC claim. If none are provided, return [].

    heart:
      - sensor_type: The exact heart-rate sensor technology name the answer states (e.g., "PPG", "photoplethysmography"). If not stated, return null.
      - light_type: The light type the answer states the sensor uses (e.g., "infrared"). If not stated, return null.
      - pulse_frequency: The pulse frequency the answer states (e.g., "256 times per second", "256 Hz"). If not stated, return null.
      - sources: An array of URLs cited in the answer that support the heart-rate sensor details. If none are provided, return [].

    ip:
      - ip_rating: The exact IP rating stated by the answer for dust/water resistance (e.g., "IP57"). If not stated, return null.
      - sources: An array of URLs cited in the answer that support the IP rating claim. If none are provided, return [].

    RULES:
    - Only extract information explicitly mentioned in the answer.
    - For URLs, extract actual URLs shown in the answer (including plain URLs or markdown links).
    - Do not invent any information. Use null or [] if missing.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _ensure_list(maybe_list: Optional[List[str]]) -> List[str]:
    return maybe_list if isinstance(maybe_list, list) else []


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_tree_and_verify(evaluator: Evaluator, specs: AirPodsSpecs) -> None:
    """
    Build the verification tree according to the rubric and trigger all verifications.
    """

    # Main rubric node (critical, parallel)
    main_node = evaluator.add_parallel(
        id="AirPods_Pro_3_Technical_Specifications",
        desc="Verify the three required AirPods Pro 3 specifications per the provided constraints: ANC battery life, heart-rate sensor technology (including pulse frequency), and official IP rating.",
        parent=evaluator.root,
        critical=True
    )

    # ---------------- Battery Life (ANC mode) ----------------
    battery_leaf = evaluator.add_leaf(
        id="Battery_Life_ANC_Mode",
        desc="Answer states AirPods Pro 3 provide up to 8 hours of listening time on a single charge with Active Noise Cancellation enabled.",
        parent=main_node,
        critical=True,
    )

    battery_sources = _ensure_list(specs.battery.battery_sources) if specs.battery else []
    await evaluator.verify(
        claim=EXPECTED_BATTERY_ANC_CLAIM,
        node=battery_leaf,
        sources=battery_sources,
        additional_instruction=(
            "Verify that the provided URL(s) explicitly state the AirPods Pro 3 have up to 8 hours of listening time "
            "on a single charge with ANC enabled. Accept small wording variants like 'up to eight hours' but the value "
            "must be 8 and specifically with ANC enabled. If the page references a different model or a different duration, "
            "it does not support the claim."
        ),
    )

    # ---------------- Heart-Rate Sensor Technology (critical parallel group) ----------------
    hr_node = evaluator.add_parallel(
        id="Heart_Rate_Sensor_Technology",
        desc="Answer states the heart-rate sensing technology and operating details per constraints.",
        parent=main_node,
        critical=True
    )

    heart_sources = _ensure_list(specs.heart.sources) if specs.heart else []

    # Sensor Type
    sensor_type_leaf = evaluator.add_leaf(
        id="Sensor_Type",
        desc="Answer identifies the sensor as a custom PPG (photoplethysmography) sensor.",
        parent=hr_node,
        critical=True,
    )
    await evaluator.verify(
        claim=EXPECTED_SENSOR_TYPE_CLAIM,
        node=sensor_type_leaf,
        sources=heart_sources,
        additional_instruction=(
            "Confirm the page(s) explicitly state a heart-rate sensor exists in AirPods Pro 3 and that the technology is "
            "PPG/photoplethysmography. If heart-rate sensing is absent or the technology differs, the claim is not supported."
        ),
    )

    # Light Type
    light_type_leaf = evaluator.add_leaf(
        id="Light_Type",
        desc="Answer states the PPG uses invisible infrared light.",
        parent=hr_node,
        critical=True,
    )
    await evaluator.verify(
        claim=EXPECTED_LIGHT_TYPE_CLAIM,
        node=light_type_leaf,
        sources=heart_sources,
        additional_instruction=(
            "Verify that the heart-rate sensor's PPG uses invisible infrared light. If the page mentions green LEDs or any "
            "other light type instead of invisible infrared, this claim is not supported."
        ),
    )

    # Pulse Frequency
    pulse_freq_leaf = evaluator.add_leaf(
        id="Pulse_Frequency",
        desc="Answer states the infrared light is pulsed at 256 times per second.",
        parent=hr_node,
        critical=True,
    )
    await evaluator.verify(
        claim=EXPECTED_PULSE_FREQUENCY_CLAIM,
        node=pulse_freq_leaf,
        sources=heart_sources,
        additional_instruction=(
            "Confirm that the page(s) explicitly state the infrared light for heart-rate sensing is pulsed at 256 times per second "
            "(i.e., 256 Hz). Minor wording variations are acceptable, but the frequency must match 256."
        ),
    )

    # ---------------- IP Rating ----------------
    ip_leaf = evaluator.add_leaf(
        id="IP_Rating",
        desc="Answer states the official dust/water resistance rating is IP57 (for dust, sweat, and water resistance).",
        parent=main_node,
        critical=True,
    )

    ip_sources = _ensure_list(specs.ip.sources) if specs.ip else []
    await evaluator.verify(
        claim=EXPECTED_IP_RATING_CLAIM,
        node=ip_leaf,
        sources=ip_sources,
        additional_instruction=(
            "Verify the official IP rating for AirPods Pro 3 as IP57. The supporting page must clearly indicate IP57 for dust and water "
            "resistance. If the rating shown is different (e.g., IP54, IPX4) or the product page is for a different model, consider the claim unsupported."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for AirPods Pro 3 technical specifications.
    """
    # Initialize evaluator (root is created internally and is non-critical by design)
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured specs from the answer
    specs: AirPodsSpecs = await evaluator.extract(
        prompt=prompt_extract_specs(),
        template_class=AirPodsSpecs,
        extraction_name="airpods_pro_3_specs",
    )

    # Record expected ground truth (per rubric)
    evaluator.add_ground_truth({
        "expected_battery_anc_claim": EXPECTED_BATTERY_ANC_CLAIM,
        "expected_sensor_type_claim": EXPECTED_SENSOR_TYPE_CLAIM,
        "expected_light_type_claim": EXPECTED_LIGHT_TYPE_CLAIM,
        "expected_pulse_frequency_claim": EXPECTED_PULSE_FREQUENCY_CLAIM,
        "expected_ip_rating_claim": EXPECTED_IP_RATING_CLAIM,
    }, gt_type="expected_claims")

    # Build the verification tree and run checks
    await build_tree_and_verify(evaluator, specs)

    # Return summarized evaluation results
    return evaluator.get_summary()