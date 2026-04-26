import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "iphone15_specs_verizon_uwb"
TASK_DESCRIPTION = """
A business professional is evaluating the iPhone 15 for purchase and needs to verify its technical specifications to ensure it meets their requirements and is compatible with Verizon's 5G Ultra Wideband network. Provide the following specifications for the iPhone 15:

1. Display size (in inches, diagonal measurement)
2. Display resolution (pixel dimensions and pixels per inch)
3. Processor model
4. Main camera resolution (in megapixels)
5. Water resistance rating with maximum depth and duration specifications
6. Charging port connector type
7. Support for 5G band n77 (C-band) - confirm Yes or No
8. Support for 5G mmWave bands n260 and n261 - confirm Yes or No
9. Battery life for video playback (maximum hours)
10. Device weight (in grams)
11. Face ID authentication support - confirm Yes or No
12. MagSafe wireless charging support and maximum charging wattage
13. SIM card technology type
"""


# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class IPhone15Specs(BaseModel):
    # Display
    display_size: Optional[str] = None  # e.g., "6.1 inches"
    display_tech: Optional[str] = None  # e.g., "OLED", "Super Retina XDR OLED"
    display_resolution_pixels: Optional[str] = None  # e.g., "2556×1179"
    display_ppi: Optional[str] = None  # e.g., "460 ppi"

    # Processor
    processor_model: Optional[str] = None  # e.g., "A16 Bionic"

    # Camera
    main_camera_resolution_mp: Optional[str] = None  # e.g., "48MP"

    # Water resistance
    water_resistance_rating: Optional[str] = None  # e.g., "IP68"
    water_resistance_depth_m: Optional[str] = None  # e.g., "6 meters"
    water_resistance_duration_min: Optional[str] = None  # e.g., "30 minutes"

    # Charging port
    charging_port_type: Optional[str] = None  # e.g., "USB-C"

    # 5G bands
    support_5g_n77: Optional[str] = None  # "Yes" or "No"
    support_5g_mmwave_n260_n261: Optional[str] = None  # "Yes" or "No"

    # Battery life (video)
    battery_video_playback_hours: Optional[str] = None  # e.g., "20 hours"

    # Weight
    device_weight_grams: Optional[str] = None  # e.g., "171 grams"

    # Face ID
    face_id_support: Optional[str] = None  # "Yes" or "No"

    # MagSafe
    magsafe_support: Optional[str] = None  # "Yes" or "No"
    magsafe_max_wattage: Optional[str] = None  # e.g., "15W"

    # SIM technology
    sim_technology: Optional[str] = None  # e.g., "Dual eSIM (no physical SIM)"

    # Sources
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_specs() -> str:
    return """
    Extract the iPhone 15 technical specifications as presented in the answer. Focus on the base iPhone 15 model (6.1-inch), not the Plus or Pro variants.
    Return a JSON object with the following fields (use exact text from the answer without inventing anything):
    - display_size: diagonal display size in inches (e.g., "6.1 inches").
    - display_tech: display technology (e.g., "OLED", or Apple's branding such as "Super Retina XDR (OLED)").
    - display_resolution_pixels: pixel dimensions (e.g., "2556×1179" or "2556x1179").
    - display_ppi: pixel density (e.g., "460 ppi").
    - processor_model: chip model (e.g., "A16 Bionic").
    - main_camera_resolution_mp: main camera resolution (e.g., "48MP", "48 megapixels").
    - water_resistance_rating: rating string (e.g., "IP68").
    - water_resistance_depth_m: maximum depth (e.g., "6 meters").
    - water_resistance_duration_min: maximum duration (e.g., "30 minutes").
    - charging_port_type: connector type (e.g., "USB-C").
    - support_5g_n77: "Yes" or "No" for band n77 support.
    - support_5g_mmwave_n260_n261: "Yes" or "No" for mmWave bands n260 and n261 support.
    - battery_video_playback_hours: maximum hours of video playback (e.g., "20 hours").
    - device_weight_grams: device weight in grams (e.g., "171 grams").
    - face_id_support: "Yes" or "No".
    - magsafe_support: "Yes" or "No".
    - magsafe_max_wattage: maximum MagSafe charging wattage (e.g., "15W").
    - sim_technology: SIM technology (e.g., "Dual eSIM (no physical SIM)").
    - source_urls: all URLs explicitly mentioned in the answer, extracted as full URLs. Include Apple, Verizon, or other spec pages. If none are given, return an empty list.

    If any field is missing from the answer, return null for that field (or empty list for source_urls). Do not infer values.
    """


# --------------------------------------------------------------------------- #
# Helper: Normalize sources                                                   #
# --------------------------------------------------------------------------- #
def get_sources(specs: IPhone15Specs) -> Optional[List[str]]:
    return specs.source_urls if specs.source_urls else None


# --------------------------------------------------------------------------- #
# Verification Logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_specs(evaluator: Evaluator, parent_node, specs: IPhone15Specs) -> None:
    """
    Build the verification tree and perform checks for each specification. 
    The top-level node is critical; all children must pass to satisfy the rubric.
    """
    # Create the critical top-level parallel node
    top = evaluator.add_parallel(
        id="iPhone_15_Technical_Specifications",
        desc="Verify iPhone 15 technical specifications and Verizon 5G Ultra Wideband compatibility against the stated constraints",
        parent=parent_node,
        critical=True,
    )

    sources = get_sources(specs)

    # Display size and technology (split into two critical checks under a sequential group)
    display_group = evaluator.add_sequential(
        id="Display_Size_and_Technology",
        desc="Display must be 6.1 inches diagonal and use OLED technology",
        parent=top,
        critical=True,
    )

    # Existence checks (size and tech)
    evaluator.add_custom_node(
        result=bool(specs.display_size),
        id="display_size_provided",
        desc="Display size is provided in the answer",
        parent=display_group,
        critical=True,
    )
    display_size_leaf = evaluator.add_leaf(
        id="display_size_6_1",
        desc="Display is 6.1 inches diagonal",
        parent=display_group,
        critical=True,
    )
    claim_display_size = f"The iPhone 15 display size is {specs.display_size} (measured diagonally)."
    await evaluator.verify(
        claim=claim_display_size,
        node=display_size_leaf,
        sources=sources,
        additional_instruction="Verify the claim specifically for the base iPhone 15 (6.1-inch). Allow minor unit formatting differences (e.g., in vs inches)."
    )

    evaluator.add_custom_node(
        result=bool(specs.display_tech),
        id="display_tech_provided",
        desc="Display technology is provided in the answer",
        parent=display_group,
        critical=True,
    )
    display_tech_leaf = evaluator.add_leaf(
        id="display_tech_oled",
        desc="Display uses OLED technology",
        parent=display_group,
        critical=True,
    )
    claim_display_tech = f"The iPhone 15 uses {specs.display_tech} display technology."
    await evaluator.verify(
        claim=claim_display_tech,
        node=display_tech_leaf,
        sources=sources,
        additional_instruction="Treat 'Super Retina XDR' as Apple's branding for an OLED display. Confirm OLED for iPhone 15 (not Plus/Pro)."
    )

    # Display resolution: split pixel dimensions and PPI under a sequential node
    resolution_group = evaluator.add_sequential(
        id="Display_Resolution",
        desc="Display resolution must be 2556×1179 pixels at 460 ppi",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(specs.display_resolution_pixels),
        id="display_resolution_pixels_provided",
        desc="Display resolution (pixel dimensions) is provided",
        parent=resolution_group,
        critical=True,
    )
    res_pixels_leaf = evaluator.add_leaf(
        id="display_resolution_pixels_2556_1179",
        desc="Display resolution is 2556×1179 pixels",
        parent=resolution_group,
        critical=True,
    )
    claim_res_pixels = f"The iPhone 15 display resolution is {specs.display_resolution_pixels} pixels."
    await evaluator.verify(
        claim=claim_res_pixels,
        node=res_pixels_leaf,
        sources=sources,
        additional_instruction="Confirm the base iPhone 15 resolution equals 2556×1179 pixels (accept × or x)."
    )

    evaluator.add_custom_node(
        result=bool(specs.display_ppi),
        id="display_ppi_provided",
        desc="Display pixel density (ppi) is provided",
        parent=resolution_group,
        critical=True,
    )
    res_ppi_leaf = evaluator.add_leaf(
        id="display_ppi_460",
        desc="Display pixel density is 460 ppi",
        parent=resolution_group,
        critical=True,
    )
    claim_res_ppi = f"The iPhone 15 display pixel density is {specs.display_ppi}."
    await evaluator.verify(
        claim=claim_res_ppi,
        node=res_ppi_leaf,
        sources=sources,
        additional_instruction="Confirm the base iPhone 15 pixel density equals 460 ppi."
    )

    # Processor model
    processor_group = evaluator.add_sequential(
        id="Processor_Model",
        desc="Processor must be A16 Bionic chip",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(specs.processor_model),
        id="processor_provided",
        desc="Processor model is provided",
        parent=processor_group,
        critical=True,
    )
    processor_leaf = evaluator.add_leaf(
        id="processor_a16_bionic",
        desc="Processor is A16 Bionic",
        parent=processor_group,
        critical=True,
    )
    claim_processor = f"The iPhone 15 uses the {specs.processor_model} chip."
    await evaluator.verify(
        claim=claim_processor,
        node=processor_leaf,
        sources=sources,
        additional_instruction="Confirm the chip model for base iPhone 15 equals 'A16 Bionic'."
    )

    # Main camera resolution
    camera_group = evaluator.add_sequential(
        id="Main_Camera_Resolution",
        desc="Main camera must be 48MP",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(specs.main_camera_resolution_mp),
        id="camera_resolution_provided",
        desc="Main camera resolution is provided",
        parent=camera_group,
        critical=True,
    )
    camera_leaf = evaluator.add_leaf(
        id="camera_48mp",
        desc="Main camera is 48MP",
        parent=camera_group,
        critical=True,
    )
    claim_camera = f"The iPhone 15 main camera resolution is {specs.main_camera_resolution_mp}."
    await evaluator.verify(
        claim=claim_camera,
        node=camera_leaf,
        sources=sources,
        additional_instruction="Confirm that the base iPhone 15 has a 48-megapixel (48MP) main camera."
    )

    # Water resistance rating (split rating + depth + duration)
    water_group = evaluator.add_sequential(
        id="Water_Resistance_Rating",
        desc="Water resistance must be IP68 rated (maximum depth 6 meters for up to 30 minutes)",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(specs.water_resistance_rating),
        id="water_rating_provided",
        desc="Water resistance rating is provided",
        parent=water_group,
        critical=True,
    )
    water_rating_leaf = evaluator.add_leaf(
        id="water_rating_ip68",
        desc="Water resistance rating is IP68",
        parent=water_group,
        critical=True,
    )
    claim_water_rating = f"The iPhone 15 has an IP68 water resistance rating ({specs.water_resistance_rating})."
    await evaluator.verify(
        claim=claim_water_rating,
        node=water_rating_leaf,
        sources=sources,
        additional_instruction="Confirm that iPhone 15 is rated IP68."
    )

    evaluator.add_custom_node(
        result=bool(specs.water_resistance_depth_m),
        id="water_depth_provided",
        desc="Water resistance maximum depth is provided",
        parent=water_group,
        critical=True,
    )
    water_depth_leaf = evaluator.add_leaf(
        id="water_depth_6m",
        desc="Water resistance maximum depth is 6 meters",
        parent=water_group,
        critical=True,
    )
    claim_water_depth = f"The iPhone 15 water resistance maximum depth is {specs.water_resistance_depth_m}."
    await evaluator.verify(
        claim=claim_water_depth,
        node=water_depth_leaf,
        sources=sources,
        additional_instruction="Confirm the maximum depth equals 6 meters for the base iPhone 15."
    )

    evaluator.add_custom_node(
        result=bool(specs.water_resistance_duration_min),
        id="water_duration_provided",
        desc="Water resistance maximum duration is provided",
        parent=water_group,
        critical=True,
    )
    water_duration_leaf = evaluator.add_leaf(
        id="water_duration_30min",
        desc="Water resistance maximum duration is 30 minutes",
        parent=water_group,
        critical=True,
    )
    claim_water_duration = f"The iPhone 15 water resistance maximum duration is {specs.water_resistance_duration_min}."
    await evaluator.verify(
        claim=claim_water_duration,
        node=water_duration_leaf,
        sources=sources,
        additional_instruction="Confirm the maximum duration equals up to 30 minutes for the base iPhone 15."
    )

    # Charging port type
    port_group = evaluator.add_sequential(
        id="Charging_Port_Type",
        desc="Charging port must be USB-C (not Lightning)",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(specs.charging_port_type),
        id="port_type_provided",
        desc="Charging port type is provided",
        parent=port_group,
        critical=True,
    )
    port_leaf = evaluator.add_leaf(
        id="port_usb_c",
        desc="Charging port is USB-C",
        parent=port_group,
        critical=True,
    )
    claim_port = f"The iPhone 15 uses {specs.charging_port_type} as its charging/data port."
    await evaluator.verify(
        claim=claim_port,
        node=port_leaf,
        sources=sources,
        additional_instruction="Confirm that iPhone 15 uses USB‑C (and not Lightning)."
    )

    # 5G C-band n77 support
    cband_group = evaluator.add_sequential(
        id="5G_C_Band_Support_n77",
        desc="Must support 5G band n77 (C-band) for Verizon 5G Ultra Wideband",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(specs.support_5g_n77),
        id="cband_n77_provided",
        desc="Answer provides Yes/No for 5G band n77 support",
        parent=cband_group,
        critical=True,
    )
    cband_leaf = evaluator.add_leaf(
        id="cband_n77_support",
        desc="Supports 5G band n77 (C-band)",
        parent=cband_group,
        critical=True,
    )
    claim_cband = f"The iPhone 15 supports 5G band n77 (C-band): {specs.support_5g_n77}."
    await evaluator.verify(
        claim=claim_cband,
        node=cband_leaf,
        sources=sources,
        additional_instruction="Verify if iPhone 15 (base model) supports 5G band n77 (C-band). Evidence should explicitly associate n77 with iPhone 15."
    )

    # 5G mmWave n260/n261 support
    mmwave_group = evaluator.add_sequential(
        id="5G_mmWave_Support_n260_n261",
        desc="Must support 5G mmWave bands n260 and n261 for Verizon 5G Ultra Wideband",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(specs.support_5g_mmwave_n260_n261),
        id="mmwave_provided",
        desc="Answer provides Yes/No for 5G mmWave n260/n261 support",
        parent=mmwave_group,
        critical=True,
    )
    mmwave_leaf = evaluator.add_leaf(
        id="mmwave_n260_n261_support",
        desc="Supports 5G mmWave bands n260 and n261",
        parent=mmwave_group,
        critical=True,
    )
    claim_mmwave = f"The iPhone 15 supports 5G mmWave bands n260 and n261: {specs.support_5g_mmwave_n260_n261}."
    await evaluator.verify(
        claim=claim_mmwave,
        node=mmwave_leaf,
        sources=sources,
        additional_instruction="Verify explicit mmWave band support (n260 and n261) for iPhone 15 (base model). US model pages may list these bands; ensure the claim matches the evidence."
    )

    # Battery video playback
    battery_group = evaluator.add_sequential(
        id="Battery_Video_Playback",
        desc="Battery must provide up to 20 hours of video playback",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(specs.battery_video_playback_hours),
        id="battery_video_provided",
        desc="Battery life (video playback) is provided",
        parent=battery_group,
        critical=True,
    )
    battery_leaf = evaluator.add_leaf(
        id="battery_video_20h",
        desc="Battery provides up to 20 hours of video playback",
        parent=battery_group,
        critical=True,
    )
    claim_battery = f"The iPhone 15 provides up to {specs.battery_video_playback_hours} of video playback."
    await evaluator.verify(
        claim=claim_battery,
        node=battery_leaf,
        sources=sources,
        additional_instruction="Confirm Apple's stated maximum video playback for iPhone 15 equals 'up to 20 hours'."
    )

    # Device weight
    weight_group = evaluator.add_sequential(
        id="Device_Weight",
        desc="Device weight must be 171 grams (6.02 ounces)",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(specs.device_weight_grams),
        id="weight_provided",
        desc="Device weight is provided",
        parent=weight_group,
        critical=True,
    )
    weight_leaf = evaluator.add_leaf(
        id="weight_171g",
        desc="Device weight is 171 grams",
        parent=weight_group,
        critical=True,
    )
    claim_weight = f"The iPhone 15 weighs {specs.device_weight_grams}."
    await evaluator.verify(
        claim=claim_weight,
        node=weight_leaf,
        sources=sources,
        additional_instruction="Confirm the base iPhone 15 weight equals 171 grams (6.02 oz)."
    )

    # Face ID authentication
    faceid_group = evaluator.add_sequential(
        id="Face_ID_Authentication",
        desc="Must support Face ID authentication",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(specs.face_id_support),
        id="faceid_provided",
        desc="Face ID support (Yes/No) is provided",
        parent=faceid_group,
        critical=True,
    )
    faceid_leaf = evaluator.add_leaf(
        id="faceid_support_yes",
        desc="Supports Face ID authentication",
        parent=faceid_group,
        critical=True,
    )
    claim_faceid = f"The iPhone 15 supports Face ID: {specs.face_id_support}."
    await evaluator.verify(
        claim=claim_faceid,
        node=faceid_leaf,
        sources=sources,
        additional_instruction="Confirm Face ID support for iPhone 15."
    )

    # MagSafe wireless charging (support + wattage)
    magsafe_group = evaluator.add_sequential(
        id="MagSafe_Wireless_Charging",
        desc="Must support MagSafe wireless charging up to 15W",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(specs.magsafe_support),
        id="magsafe_support_provided",
        desc="MagSafe support (Yes/No) is provided",
        parent=magsafe_group,
        critical=True,
    )
    magsafe_support_leaf = evaluator.add_leaf(
        id="magsafe_support_yes",
        desc="Supports MagSafe wireless charging",
        parent=magsafe_group,
        critical=True,
    )
    claim_magsafe_support = f"The iPhone 15 supports MagSafe wireless charging: {specs.magsafe_support}."
    await evaluator.verify(
        claim=claim_magsafe_support,
        node=magsafe_support_leaf,
        sources=sources,
        additional_instruction="Confirm MagSafe wireless charging support for iPhone 15."
    )

    evaluator.add_custom_node(
        result=bool(specs.magsafe_max_wattage),
        id="magsafe_wattage_provided",
        desc="MagSafe maximum wattage is provided",
        parent=magsafe_group,
        critical=True,
    )
    magsafe_wattage_leaf = evaluator.add_leaf(
        id="magsafe_wattage_15w",
        desc="MagSafe wireless charging maximum wattage is 15W",
        parent=magsafe_group,
        critical=True,
    )
    claim_magsafe_wattage = f"The iPhone 15 MagSafe wireless charging maximum power is {specs.magsafe_max_wattage}."
    await evaluator.verify(
        claim=claim_magsafe_wattage,
        node=magsafe_wattage_leaf,
        sources=sources,
        additional_instruction="Confirm that MagSafe charging is up to 15W on iPhone 15."
    )

    # SIM technology
    sim_group = evaluator.add_sequential(
        id="SIM_Technology",
        desc="Must use dual eSIM technology (no physical SIM card)",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(specs.sim_technology),
        id="sim_tech_provided",
        desc="SIM technology is provided",
        parent=sim_group,
        critical=True,
    )
    sim_leaf = evaluator.add_leaf(
        id="sim_dual_esim_no_physical",
        desc="Uses dual eSIM technology (no physical SIM card)",
        parent=sim_group,
        critical=True,
    )
    claim_sim = f"The iPhone 15 uses SIM technology: {specs.sim_technology}."
    await evaluator.verify(
        claim=claim_sim,
        node=sim_leaf,
        sources=sources,
        additional_instruction="Confirm dual eSIM and no physical SIM tray for the base iPhone 15 (US models). If the page is US-specific, eSIM-only counts."
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
    Evaluate an answer for iPhone 15 technical specifications and Verizon 5G Ultra Wideband compatibility.
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

    # Extract the specs from the answer
    specs = await evaluator.extract(
        prompt=prompt_extract_specs(),
        template_class=IPhone15Specs,
        extraction_name="iphone15_specs",
    )

    # Optional: Add ground truth expectations (for reference only, not used for direct verification)
    evaluator.add_ground_truth({
        "expected_constraints": {
            "display_size": "6.1 inches",
            "display_tech": "OLED (Super Retina XDR)",
            "resolution_pixels": "2556×1179",
            "ppi": "460 ppi",
            "processor": "A16 Bionic",
            "main_camera": "48MP",
            "water_resistance": "IP68; 6 meters up to 30 minutes",
            "port": "USB‑C",
            "5g_n77": "Yes",
            "5g_mmwave_n260_n261": "Yes (US models)",
            "battery_video": "up to 20 hours",
            "weight": "171 grams",
            "face_id": "Yes",
            "magsafe": "Yes; up to 15W",
            "sim_tech": "dual eSIM; no physical SIM (US)",
        }
    }, gt_type="ground_truth_specs")

    # Build verification tree and run checks
    await verify_specs(evaluator, root, specs)

    # Return summary
    return evaluator.get_summary()