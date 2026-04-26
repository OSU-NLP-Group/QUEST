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
TASK_ID = "olympic_aquatics_facility_compliance"
TASK_DESCRIPTION = """
The city of Los Angeles is planning to construct a new aquatic facility for the 2028 Olympic Games swimming competitions. The facility design team has submitted their preliminary specifications for review. Evaluate whether the proposed facility meets all mandatory World Aquatics (FINA) requirements for Olympic Games swimming pool certification by verifying: 
(1) Proper measurement certification procedures are specified, including certified surveyor requirements, Total Station equipment specifications (angle accuracy 1 inch or 0.3 mgon, distance accuracy 1mm + 1.5ppm), and calibration certificate validity (maximum 1 year); 
(2) Pool geometry meets exact dimensional requirements (length 50.000m with tolerance +0.010m/-0.000m, width minimum 25.00m, orthogonality with corner angles 90° ±0.05°, diagonal equality within ±10mm, end wall verticality ±0.3°); 
(3) Depth specifications are satisfied (minimum 2.5m throughout pool, minimum 1.35m from 1.0m to at least 6.0m from end wall for starting block zones); 
(4) Lane configuration is correct (8 lanes of 2.5m width each, 2.5m outside spaces, lane ropes with 0.15m minimum float diameter, 1-1.2 kN tension, 5.0m red sections at ends, proper color scheme for 8-lane pools, distance markers at 15m and 25m); 
(5) Lane markings comply (width 0.2-0.3m, length 46.0m, dark contrasting color on pool floor); 
(6) Starting platforms meet specifications (height 0.5-0.75m above water, surface area minimum 0.5m × 0.6m, maximum slope 10°, slip-resistant surface, backstroke handgrips 0.3-0.6m above water, numbered on all four sides); 
(7) Backstroke equipment is properly specified (flagged ropes at 1.8m above water and 5.0m from each end); 
(8) Water systems meet requirements (temperature 25-28°C, constant level with turbulence test protocol, salt content less than 3 g/litre); 
(9) Lighting meets minimum intensity (not less than 1500 lux over whole pool with lighting report); 
(10) Spectator capacity is adequate (15,000 to 20,000 seats with up to one-third for VIPs, family, and media). 
Provide a comprehensive verification report indicating which requirements are satisfied and which require modification, with supporting documentation from official World Aquatics sources.
"""

# --------------------------------------------------------------------------- #
# Ground truth constraints (for logging/reference in summary)                 #
# --------------------------------------------------------------------------- #
GROUND_TRUTH_REQUIREMENTS: Dict[str, Any] = {
    "Measurement_Certification": {
        "Surveyor_Qualifications": "Certified surveyor appointed/approved by member federation of the country",
        "Total_Station_Angle_Accuracy": "Total Station angle accuracy 1 inch (0.3 mgon) for horizontal & vertical",
        "Total_Station_Distance_Accuracy": "Total Station distance accuracy 1 mm + 1.5 ppm",
        "Calibration_Certificate_Validity": "Calibration certificate validity max 1 year",
    },
    "Pool_Geometry_Structure": {
        "Touch_Panel_Length_Tolerance": "50.000 m with tolerance +0.010 m / -0.000 m (touch panel to opposite wall)",
        "Wall_to_Wall_Distance": "Without touch panels: 50.020 m to 50.030 m",
        "Tolerance_Vertical_Zone": "Tolerances apply from +0.300 m above to -0.800 m below water surface",
        "Width_Requirement": "Width ≥ 25.00 m (Olympic Games permanent pools)",
        "Corner_Angles": "Corner angles 90° ± 0.05°",
        "Diagonal_Equality": "Diagonals equal within ± 10 mm",
        "End_Wall_Verticality": "End walls vertical with tolerance ± 0.3°",
        "Minimum_Pool_Depth": "Depth ≥ 2.5 m throughout",
        "Starting_Block_Zone_Depth": "Depth ≥ 1.35 m from 1.0 m to at least 6.0 m from end wall (blocks zone)",
    },
    "Lane_Configuration_System": {
        "Number_of_Lanes": "8 lanes (9–10 lanes only with special approval)",
        "Individual_Lane_Width": "Each lane 2.5 m wide",
        "Outside_Space_Width": "Outside spaces at lanes 1 & 8 are 2.5 m each",
        "Float_Diameter": "Lane rope floats ≥ 0.15 m diameter",
        "Rope_Tension": "Lane rope tension 1–1.2 kN",
        "Red_End_Sections": "Red sections extend 5.0 m from each end",
        "Color_Scheme_8_Lanes": "8-lane color scheme: green 1&8; blue 2,3,6,7; yellow 4,5",
        "Distance_Markers_15m": "Distinct markers at 15 m from each end",
        "Distance_Markers_25m": "Distinct markers at 25 m in 50 m pools",
        "Marking_Width": "Lane marking width 0.2–0.3 m",
        "Marking_Length": "Lane marking length 46.0 m (50 m pool)",
        "Marking_Color_Placement": "Dark contrasting color at lane center on pool floor",
        "Flag_Placement": "Backstroke flags at 1.8 m above water and 5.0 m from each end",
    },
    "Starting_Platform_Equipment": {
        "Height_Above_Water": "Block height 0.5–0.75 m above water",
        "Platform_Surface_Area": "Surface area ≥ 0.5 m × 0.6 m",
        "Slip_Resistant_Surface": "Slip-resistant surface",
        "Maximum_Slope_Angle": "Maximum slope ≤ 10°",
        "Backstroke_Handgrips_Height": "Handgrips 0.3–0.6 m above water",
        "Platform_Numbering": "Numbered on all four sides",
    },
    "Water_Quality_Systems": {
        "Water_Temperature_Range": "25–28 °C",
        "Constant_Water_Level": "Constant level; no appreciable movement during competition",
        "Turbulence_Test_Protocol": "Basketball in 2.5 m square must not touch lane ropes within 60 s; test lanes 1,3,6,8 at 5 m from ends",
        "Salt_Content_Limit": "Salt content < 3 g/L",
    },
    "Lighting_Systems": {
        "Light_Intensity_Minimum": "≥ 1500 lux over whole pool",
        "Lighting_Report_Requirement": "Lighting report included as certification addendum",
    },
    "Spectator_Facilities": {
        "Seating_Capacity_Range": "15,000–20,000 seats",
        "VIP_Media_Allocation": "Up to one-third reserved for VIPs, family, media",
    }
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MeasurementCertificationExtraction(BaseModel):
    surveyor_qualifications: Optional[str] = None
    total_station_angle_accuracy: Optional[str] = None
    total_station_distance_accuracy: Optional[str] = None
    calibration_certificate_validity: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PoolGeometryExtraction(BaseModel):
    length_touch_panel_tolerance: Optional[str] = None
    wall_to_wall_distance: Optional[str] = None
    tolerance_vertical_zone: Optional[str] = None
    width_requirement: Optional[str] = None
    corner_angles: Optional[str] = None
    diagonal_equality: Optional[str] = None
    end_wall_verticality: Optional[str] = None
    minimum_pool_depth: Optional[str] = None
    starting_block_zone_depth: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class LaneConfigurationExtraction(BaseModel):
    number_of_lanes: Optional[str] = None
    individual_lane_width: Optional[str] = None
    outside_space_width: Optional[str] = None
    float_diameter: Optional[str] = None
    rope_tension: Optional[str] = None
    red_end_sections: Optional[str] = None
    color_scheme_8_lanes: Optional[str] = None
    distance_markers_15m: Optional[str] = None
    distance_markers_25m: Optional[str] = None
    marking_width: Optional[str] = None
    marking_length: Optional[str] = None
    marking_color_placement: Optional[str] = None
    backstroke_flag_placement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class StartingPlatformExtraction(BaseModel):
    height_above_water: Optional[str] = None
    platform_surface_area: Optional[str] = None
    slip_resistant_surface: Optional[str] = None
    maximum_slope_angle: Optional[str] = None
    backstroke_handgrips_height: Optional[str] = None
    platform_numbering: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class WaterQualityExtraction(BaseModel):
    water_temperature_range: Optional[str] = None
    constant_water_level: Optional[str] = None
    turbulence_test_protocol: Optional[str] = None
    salt_content_limit: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class LightingExtraction(BaseModel):
    light_intensity_minimum: Optional[str] = None
    lighting_report_requirement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SpectatorFacilitiesExtraction(BaseModel):
    seating_capacity_range: Optional[str] = None
    vip_media_allocation: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_measurement() -> str:
    return """
    Extract the facility specification statements related to measurement certification and equipment from the answer. 
    Return a JSON with:
    - surveyor_qualifications: The exact statement about certified surveyor appointment/approval (string or null)
    - total_station_angle_accuracy: The exact statement about Total Station angle accuracy for horizontal/vertical measurements (string or null)
    - total_station_distance_accuracy: The exact statement about Total Station distance accuracy (string or null)
    - calibration_certificate_validity: The exact statement about calibration certificate validity period (string or null)
    - sources: An array of official World Aquatics / FINA URLs cited for measurement requirements. Only include URLs explicitly present in the answer and only if they are official (e.g., worldaquatics.com, fina.org). If none are present, return an empty array.
    """


def prompt_extract_geometry() -> str:
    return """
    Extract the facility specification statements related to pool geometry and structure (dimensions, tolerances, orthogonality, width, depth).
    Return a JSON with:
    - length_touch_panel_tolerance: Statement about 50.000 m pool length with +0.010 / -0.000 tolerance between touch panels and opposite wall
    - wall_to_wall_distance: Statement about 50.020–50.030 m wall-to-wall distance without touch panels
    - tolerance_vertical_zone: Statement about the vertical zone (0.300 m above to 0.800 m below water surface) where tolerances apply
    - width_requirement: Statement about width ≥ 25.00 m for Olympic permanent pools
    - corner_angles: Statement about corner angles 90° ± 0.05°
    - diagonal_equality: Statement about diagonals equal within ±10 mm
    - end_wall_verticality: Statement about end wall verticality ± 0.3°
    - minimum_pool_depth: Statement about minimum depth 2.5 m throughout pool
    - starting_block_zone_depth: Statement about minimum depth 1.35 m from 1.0 m to at least 6.0 m from end wall in starting block zones
    - sources: Official WA/FINA URLs cited for geometry requirements. Only include URLs explicitly present in the answer and only official ones. Empty array if none.
    """


def prompt_extract_lane_configuration() -> str:
    return """
    Extract the facility specification statements related to lane configuration, lane ropes, floor markings, and backstroke flags.
    Return a JSON with:
    - number_of_lanes: Statement about lane count (e.g., 8 lanes)
    - individual_lane_width: Statement about lane width (2.5 m each)
    - outside_space_width: Statement about 2.5 m outside spaces at lanes 1 and 8
    - float_diameter: Statement about lane rope float diameter (≥ 0.15 m)
    - rope_tension: Statement about lane rope tension (1–1.2 kN)
    - red_end_sections: Statement about 5.0 m red sections at ends
    - color_scheme_8_lanes: Statement about 8-lane color scheme (green 1&8; blue 2,3,6,7; yellow 4,5)
    - distance_markers_15m: Statement about markers at 15 m from each end
    - distance_markers_25m: Statement about markers at 25 m in 50 m pools
    - marking_width: Statement about lane marking width (0.2–0.3 m)
    - marking_length: Statement about lane marking length (46.0 m)
    - marking_color_placement: Statement about lane markings being dark contrasting color in lane center on pool floor
    - backstroke_flag_placement: Statement about flags at 1.8 m above water and 5.0 m from ends
    - sources: Official WA/FINA URLs cited for lane/marking/flag requirements. Only include URLs explicitly present in the answer and only official ones. Empty array if none.
    """


def prompt_extract_starting_platform() -> str:
    return """
    Extract the facility specification statements related to starting platforms/blocks.
    Return a JSON with:
    - height_above_water: Statement about block height above water (0.5–0.75 m)
    - platform_surface_area: Statement about surface area (≥ 0.5 m × 0.6 m)
    - slip_resistant_surface: Statement about slip-resistant surface
    - maximum_slope_angle: Statement about maximum slope angle (≤ 10°)
    - backstroke_handgrips_height: Statement about handgrips height (0.3–0.6 m above water)
    - platform_numbering: Statement about numbering on all four sides
    - sources: Official WA/FINA URLs cited for starting platform requirements. Only include URLs explicitly present in the answer and only official ones. Empty array if none.
    """


def prompt_extract_water_quality() -> str:
    return """
    Extract the facility specification statements related to water systems and quality.
    Return a JSON with:
    - water_temperature_range: Statement about water temperature range (25–28 °C)
    - constant_water_level: Statement about constant water level/no appreciable movement during competition
    - turbulence_test_protocol: Statement about turbulence test protocol (basketball in 2.5 m square not touching ropes within 60 s; tested lanes 1,3,6,8 at 5 m from ends)
    - salt_content_limit: Statement about salt content (< 3 g/L)
    - sources: Official WA/FINA URLs cited for these water system requirements. Only include URLs explicitly present in the answer and only official ones. Empty array if none.
    """


def prompt_extract_lighting() -> str:
    return """
    Extract the facility specification statements related to lighting.
    Return a JSON with:
    - light_intensity_minimum: Statement about minimum light intensity (≥ 1500 lux over whole pool)
    - lighting_report_requirement: Statement about lighting report as an addendum to certification documentation
    - sources: Official WA/FINA URLs cited for lighting requirements. Only include URLs explicitly present in the answer and only official ones. Empty array if none.
    """


def prompt_extract_spectator() -> str:
    return """
    Extract the facility specification statements related to spectator seating capacity and allocations.
    Return a JSON with:
    - seating_capacity_range: Statement about seating capacity (15,000–20,000 seats)
    - vip_media_allocation: Statement about up to one-third reserved for VIPs, athlete family, media
    - sources: Official WA/FINA URLs cited for spectator facility requirements. Only include URLs explicitly present in the answer and only official ones. Empty array if none.
    """


# --------------------------------------------------------------------------- #
# Helper: Build item verification: existence → spec claim → WA doc support    #
# --------------------------------------------------------------------------- #
async def build_item_checks(
    evaluator: Evaluator,
    parent_node,
    item_id: str,
    item_desc: str,
    exists: bool,
    spec_claim: str,
    wa_sources: List[str],
    spec_add_ins: Optional[str] = None,
    wa_add_ins: Optional[str] = None,
) -> None:
    """
    Create a sequential sub-tree for one requirement:
    1) Existence in facility spec
    2) Spec compliance in answer text
    3) Supported by official WA documentation (URLs)
    """
    checks_node = evaluator.add_sequential(
        id=f"{item_id}_checks",
        desc=f"{item_desc} — two-step compliance (spec + WA documentation)",
        parent=parent_node,
        critical=True
    )

    # 1) Existence in facility specification
    evaluator.add_custom_node(
        result=bool(exists),
        id=f"{item_id}_exists",
        desc=f"{item_desc} is explicitly specified in the facility submission",
        parent=checks_node,
        critical=True
    )

    # 2) Spec compliance (from answer text)
    spec_leaf = evaluator.add_leaf(
        id=item_id,
        desc=item_desc,
        parent=checks_node,
        critical=True
    )
    await evaluator.verify(
        claim=spec_claim,
        node=spec_leaf,
        additional_instruction=spec_add_ins or (
            "Verify this statement against the facility's answer text. "
            "Allow minor paraphrasing or synonymy, but numeric values, units, and ranges must match or be clearly equivalent."
        )
    )

    # 3) WA documentation support
    wa_leaf = evaluator.add_leaf(
        id=f"{item_id}_WA_support",
        desc=f"{item_desc} is required by official World Aquatics (FINA) rules",
        parent=checks_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"World Aquatics (FINA) rules require: {spec_claim}",
        node=wa_leaf,
        sources=wa_sources,
        additional_instruction=wa_add_ins or (
            "Check the provided official World Aquatics/FINA webpage(s) to confirm this requirement is explicitly stated. "
            "Treat only worldaquatics.com / fina.org / official PDFs as authoritative. "
            "Numeric values and units must match exactly or be clearly equivalent."
        )
    )


# --------------------------------------------------------------------------- #
# Category verification functions                                             #
# --------------------------------------------------------------------------- #
async def verify_measurement_certification(
    evaluator: Evaluator,
    parent_node,
    data: MeasurementCertificationExtraction,
) -> None:
    cat = evaluator.add_parallel(
        id="Measurement_Certification",
        desc="Verifies required measurement certification personnel and equipment constraints are specified",
        parent=parent_node,
        critical=True
    )
    sources = data.sources or []

    items = [
        (
            "Surveyor_Qualifications",
            "Measurements are conducted by a certified surveyor appointed or approved by the member federation in the country where the venue is located",
            data.surveyor_qualifications,
            "The facility specification states that measurements are conducted by a certified surveyor appointed or approved by the member federation in the country where the venue is located."
        ),
        (
            "Total_Station_Angle_Accuracy",
            "Surveying equipment is a Total Station with angle measurement accuracy of 1 inch (0.3 mgon) for horizontal and vertical measurements",
            data.total_station_angle_accuracy,
            "The facility specification states a Total Station angle measurement accuracy of 1 inch (0.3 mgon) for both horizontal and vertical measurements."
        ),
        (
            "Total_Station_Distance_Accuracy",
            "Total Station distance measurement accuracy is 1 mm + 1.5 ppm",
            data.total_station_distance_accuracy,
            "The facility specification states a Total Station distance accuracy of 1 mm + 1.5 ppm."
        ),
        (
            "Calibration_Certificate_Validity",
            "Surveyor provides a calibration certificate with maximum validity of 1 year",
            data.calibration_certificate_validity,
            "The facility specification states a calibration certificate with a maximum validity of 1 year."
        ),
    ]
    for item_id, item_desc, field_val, spec_claim in items:
        await build_item_checks(
            evaluator, cat, item_id, item_desc, bool(field_val), spec_claim, sources
        )


async def verify_pool_geometry_structure(
    evaluator: Evaluator,
    parent_node,
    data: PoolGeometryExtraction,
) -> None:
    cat = evaluator.add_parallel(
        id="Pool_Geometry_Structure",
        desc="Evaluates all dimensional and structural pool requirements from the constraints",
        parent=parent_node,
        critical=True
    )
    sources = data.sources or []

    # Length specs group
    length_node = evaluator.add_parallel(
        id="Length_Specifications",
        desc="Verifies pool length and tolerance constraints",
        parent=cat,
        critical=True
    )
    length_items = [
        (
            "Touch_Panel_Length_Tolerance",
            "Pool length between touch panels and opposite wall is 50.000 m with tolerance +0.010 m / -0.000 m",
            data.length_touch_panel_tolerance,
            "The facility specification states the pool length between touch panels and the opposite wall is 50.000 m with tolerance +0.010 m / -0.000 m."
        ),
        (
            "Wall_to_Wall_Distance",
            "Wall-to-wall distance without touch panels is between 50.020 m and 50.030 m",
            data.wall_to_wall_distance,
            "The facility specification states wall-to-wall distance without touch panels is between 50.020 m and 50.030 m."
        ),
        (
            "Tolerance_Vertical_Zone",
            "Tolerances are consistent from 0.300 m above to 0.800 m below water surface",
            data.tolerance_vertical_zone,
            "The facility specification states the tolerance zone applies from 0.300 m above to 0.800 m below the water surface."
        ),
    ]
    for item_id, item_desc, field_val, spec_claim in length_items:
        await build_item_checks(evaluator, length_node, item_id, item_desc, bool(field_val), spec_claim, sources)

    # Width requirement
    await build_item_checks(
        evaluator, cat,
        "Width_Requirement",
        "Pool width for Olympic Games permanent pools is at least 25.00 m",
        bool(data.width_requirement),
        "The facility specification states the pool width is at least 25.00 m.",
        sources
    )

    # Orthogonality requirements group
    ortho_node = evaluator.add_parallel(
        id="Orthogonality_Requirements",
        desc="Verifies geometric accuracy constraints for angles, diagonals, and end walls",
        parent=cat,
        critical=True
    )
    ortho_items = [
        (
            "Corner_Angles",
            "Pool sides form 90° angles with tolerance of ±0.05°",
            data.corner_angles,
            "The facility specification states pool sides form 90° angles with a tolerance of ±0.05°."
        ),
        (
            "Diagonal_Equality",
            "Two pool diagonals are equal length within ±10 mm",
            data.diagonal_equality,
            "The facility specification states the two pool diagonals are equal within ±10 mm."
        ),
        (
            "End_Wall_Verticality",
            "End walls are vertical and form 90° right angles with verticality tolerance of ±0.3°",
            data.end_wall_verticality,
            "The facility specification states the end walls are vertical with a verticality tolerance of ±0.3°."
        ),
    ]
    for item_id, item_desc, field_val, spec_claim in ortho_items:
        await build_item_checks(evaluator, ortho_node, item_id, item_desc, bool(field_val), spec_claim, sources)

    # Depth requirements group
    depth_node = evaluator.add_parallel(
        id="Depth_Requirements",
        desc="Verifies Olympic minimum depth constraints",
        parent=cat,
        critical=True
    )
    depth_items = [
        (
            "Minimum_Pool_Depth",
            "Minimum water depth is 2.5 m throughout the pool",
            data.minimum_pool_depth,
            "The facility specification states a minimum depth of 2.5 m throughout the pool."
        ),
        (
            "Starting_Block_Zone_Depth",
            "For pools with starting blocks, minimum depth of 1.35 m extends from 1.0 m to at least 6.0 m from end wall",
            data.starting_block_zone_depth,
            "The facility specification states a minimum depth of 1.35 m from 1.0 m to at least 6.0 m from the end wall in starting block zones."
        ),
    ]
    for item_id, item_desc, field_val, spec_claim in depth_items:
        await build_item_checks(evaluator, depth_node, item_id, item_desc, bool(field_val), spec_claim, sources)


async def verify_lane_configuration_system(
    evaluator: Evaluator,
    parent_node,
    data: LaneConfigurationExtraction,
) -> None:
    cat = evaluator.add_parallel(
        id="Lane_Configuration_System",
        desc="Evaluates lane count/dimensions, lane rope requirements, lane markings, and backstroke flags per constraints",
        parent=parent_node,
        critical=True
    )
    sources = data.sources or []

    # Lane count & dimensions
    count_node = evaluator.add_parallel(
        id="Lane_Count_Dimensions",
        desc="Verifies lane quantity and dimensional constraints",
        parent=cat,
        critical=True
    )
    count_items = [
        (
            "Number_of_Lanes",
            "Olympic Games require 8 lanes (9–10 lanes only with special approval from Technical Swimming Committee Chair)",
            data.number_of_lanes,
            "The facility specification states the pool has 8 lanes (with 9–10 lanes only allowed by special approval)."
        ),
        (
            "Individual_Lane_Width",
            "Each lane is 2.5 m wide",
            data.individual_lane_width,
            "The facility specification states each lane is 2.5 m wide."
        ),
        (
            "Outside_Space_Width",
            "Spaces outside lanes 1 and 8 are 2.5 m wide each",
            data.outside_space_width,
            "The facility specification states the outside spaces at lanes 1 and 8 are 2.5 m wide each."
        ),
    ]
    for item_id, item_desc, field_val, spec_claim in count_items:
        await build_item_checks(evaluator, count_node, item_id, item_desc, bool(field_val), spec_claim, sources)

    # Lane rope specifications
    rope_node = evaluator.add_parallel(
        id="Lane_Rope_Specifications",
        desc="Verifies lane rope constraints (floats, tension, colors, markers)",
        parent=cat,
        critical=True
    )
    rope_items = [
        (
            "Float_Diameter",
            "Lane rope floats have minimum diameter of 0.15 m",
            data.float_diameter,
            "The facility specification states lane rope floats have a minimum diameter of 0.15 m."
        ),
        (
            "Rope_Tension",
            "Lane rope tension is between 1 kN and 1.2 kN",
            data.rope_tension,
            "The facility specification states lane rope tension is between 1 kN and 1.2 kN."
        ),
        (
            "Red_End_Sections",
            "Red colored section of lane ropes extends 5.0 m from each end of pool",
            data.red_end_sections,
            "The facility specification states the lane ropes have 5.0 m red sections at each end."
        ),
        (
            "Color_Scheme_8_Lanes",
            "8-lane pool color scheme: green ropes at lanes 1 and 8; blue ropes at lanes 2, 3, 6, 7; yellow ropes at lanes 4 and 5",
            data.color_scheme_8_lanes,
            "The facility specification states the 8-lane color scheme: green at lanes 1 & 8; blue at lanes 2,3,6,7; yellow at lanes 4 & 5."
        ),
        (
            "Distance_Markers_15m",
            "Distinct color markers are placed at 15 m from each end wall",
            data.distance_markers_15m,
            "The facility specification states distinct color markers are placed at 15 m from each end wall."
        ),
        (
            "Distance_Markers_25m",
            "Distinct color markers are placed at the 25 m mark in 50 m pools",
            data.distance_markers_25m,
            "The facility specification states distinct color markers are placed at the 25 m mark (50 m pool)."
        ),
    ]
    for item_id, item_desc, field_val, spec_claim in rope_items:
        await build_item_checks(evaluator, rope_node, item_id, item_desc, bool(field_val), spec_claim, sources)

    # Lane floor markings
    marking_node = evaluator.add_parallel(
        id="Lane_Floor_Markings",
        desc="Verifies lane marking constraints on pool floor",
        parent=cat,
        critical=True
    )
    marking_items = [
        (
            "Marking_Width",
            "Lane marking width is between 0.2 m and 0.3 m",
            data.marking_width,
            "The facility specification states lane marking width is between 0.2 m and 0.3 m."
        ),
        (
            "Marking_Length",
            "Lane marking length for 50 m pools is 46.0 m",
            data.marking_length,
            "The facility specification states lane marking length is 46.0 m."
        ),
        (
            "Marking_Color_Placement",
            "Lane markings are dark contrasting color on pool floor in center of each lane",
            data.marking_color_placement,
            "The facility specification states lane markings are a dark contrasting color in the center of each lane on the pool floor."
        ),
    ]
    for item_id, item_desc, field_val, spec_claim in marking_items:
        await build_item_checks(evaluator, marking_node, item_id, item_desc, bool(field_val), spec_claim, sources)

    # Backstroke flags
    flags_node = evaluator.add_parallel(
        id="Backstroke_Flags",
        desc="Verifies backstroke flagged rope placement constraint",
        parent=cat,
        critical=True
    )
    await build_item_checks(
        evaluator, flags_node,
        "Flag_Placement",
        "Flagged ropes are positioned at 1.8 m above water surface and 5.0 m from each end wall",
        bool(data.backstroke_flag_placement),
        "The facility specification states backstroke flagged ropes are positioned at 1.8 m above the water surface and 5.0 m from each end wall.",
        sources
    )


async def verify_starting_platform_equipment(
    evaluator: Evaluator,
    parent_node,
    data: StartingPlatformExtraction,
) -> None:
    cat = evaluator.add_parallel(
        id="Starting_Platform_Equipment",
        desc="Evaluates starting platform constraints from the provided list",
        parent=parent_node,
        critical=True
    )
    sources = data.sources or []

    items = [
        (
            "Height_Above_Water",
            "Starting platform height above water is between 0.5 m and 0.75 m",
            data.height_above_water,
            "The facility specification states starting platform height above water is between 0.5 m and 0.75 m."
        ),
        (
            "Platform_Surface_Area",
            "Platform surface area is at least 0.5 m × 0.6 m",
            data.platform_surface_area,
            "The facility specification states platform surface area is at least 0.5 m × 0.6 m."
        ),
        (
            "Slip_Resistant_Surface",
            "Platform surface is covered with slip-resistant material",
            data.slip_resistant_surface,
            "The facility specification states the platform surface is slip-resistant."
        ),
        (
            "Maximum_Slope_Angle",
            "Platform maximum slope does not exceed 10 degrees",
            data.maximum_slope_angle,
            "The facility specification states the platform maximum slope does not exceed 10°."
        ),
        (
            "Backstroke_Handgrips_Height",
            "Backstroke handgrips are positioned 0.3 m to 0.6 m above water surface",
            data.backstroke_handgrips_height,
            "The facility specification states backstroke handgrips are positioned 0.3–0.6 m above the water surface."
        ),
        (
            "Platform_Numbering",
            "Each starting block is distinctly numbered on all four sides",
            data.platform_numbering,
            "The facility specification states each starting block is numbered on all four sides."
        ),
    ]
    for item_id, item_desc, field_val, spec_claim in items:
        await build_item_checks(evaluator, cat, item_id, item_desc, bool(field_val), spec_claim, sources)


async def verify_water_quality_systems(
    evaluator: Evaluator,
    parent_node,
    data: WaterQualityExtraction,
) -> None:
    cat = evaluator.add_parallel(
        id="Water_Quality_Systems",
        desc="Evaluates water temperature, constant level, turbulence test, and salt content constraints",
        parent=parent_node,
        critical=True
    )
    sources = data.sources or []

    items = [
        (
            "Water_Temperature_Range",
            "Water temperature is between 25 °C and 28 °C",
            data.water_temperature_range,
            "The facility specification states water temperature is maintained between 25 °C and 28 °C."
        ),
        (
            "Constant_Water_Level",
            "Water is maintained at constant level with no appreciable movement during competition",
            data.constant_water_level,
            "The facility specification states a constant water level with no appreciable movement during competition."
        ),
        (
            "Turbulence_Test_Protocol",
            "Turbulence test protocol is satisfied (basketball in 2.5 m square must not touch lane ropes within 60 seconds; tested in lanes 1, 3, 6, 8 at 5 m from each end)",
            data.turbulence_test_protocol,
            "The facility specification states the turbulence test protocol: a basketball in a 2.5 m square must not touch lane ropes within 60 seconds, tested in lanes 1, 3, 6, 8 at 5 m from each end."
        ),
        (
            "Salt_Content_Limit",
            "Salt content is less than 3 g/litre",
            data.salt_content_limit,
            "The facility specification states salt content is less than 3 g/litre."
        ),
    ]
    for item_id, item_desc, field_val, spec_claim in items:
        await build_item_checks(evaluator, cat, item_id, item_desc, bool(field_val), spec_claim, sources)


async def verify_lighting_systems(
    evaluator: Evaluator,
    parent_node,
    data: LightingExtraction,
) -> None:
    cat = evaluator.add_parallel(
        id="Lighting_Systems",
        desc="Evaluates lighting intensity and documentation constraints",
        parent=parent_node,
        critical=True
    )
    sources = data.sources or []

    items = [
        (
            "Light_Intensity_Minimum",
            "Light intensity over whole pool is not less than 1500 lux",
            data.light_intensity_minimum,
            "The facility specification states light intensity over the whole pool is at least 1500 lux."
        ),
        (
            "Lighting_Report_Requirement",
            "Lighting report is provided as addendum to certification documentation",
            data.lighting_report_requirement,
            "The facility specification states a lighting report will be provided as an addendum to certification documentation."
        ),
    ]
    for item_id, item_desc, field_val, spec_claim in items:
        await build_item_checks(evaluator, cat, item_id, item_desc, bool(field_val), spec_claim, sources)


async def verify_spectator_facilities(
    evaluator: Evaluator,
    parent_node,
    data: SpectatorFacilitiesExtraction,
) -> None:
    cat = evaluator.add_parallel(
        id="Spectator_Facilities",
        desc="Evaluates spectator seating capacity constraints",
        parent=parent_node,
        critical=True
    )
    sources = data.sources or []

    items = [
        (
            "Seating_Capacity_Range",
            "Venue provides between 15,000 and 20,000 seats",
            data.seating_capacity_range,
            "The facility specification states spectator seating capacity is between 15,000 and 20,000."
        ),
        (
            "VIP_Media_Allocation",
            "Up to one-third of seats may be reserved for VIPs, athlete family members, and media",
            data.vip_media_allocation,
            "The facility specification states up to one-third of seats may be reserved for VIPs, athlete family members, and media."
        ),
    ]
    for item_id, item_desc, field_val, spec_claim in items:
        await build_item_checks(evaluator, cat, item_id, item_desc, bool(field_val), spec_claim, sources)


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
    Evaluate the facility specification against World Aquatics (FINA) requirements
    using a hierarchical verification tree with spec compliance and WA documentation support.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation parallel (JSON root is parallel)
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

    # Top-level compliance node (critical)
    compliance_root = evaluator.add_parallel(
        id="Olympic_Facility_Compliance",
        desc="Evaluates whether the proposed aquatic facility meets all World Aquatics (FINA) requirements listed in the provided constraints for Olympic Games swimming pool certification",
        parent=root,
        critical=True
    )

    # Extract structured info (parallelize extraction across categories)
    measurement_task = evaluator.extract(
        prompt=prompt_extract_measurement(),
        template_class=MeasurementCertificationExtraction,
        extraction_name="measurement_certification"
    )
    geometry_task = evaluator.extract(
        prompt=prompt_extract_geometry(),
        template_class=PoolGeometryExtraction,
        extraction_name="pool_geometry_structure"
    )
    lane_task = evaluator.extract(
        prompt=prompt_extract_lane_configuration(),
        template_class=LaneConfigurationExtraction,
        extraction_name="lane_configuration_system"
    )
    starting_platform_task = evaluator.extract(
        prompt=prompt_extract_starting_platform(),
        template_class=StartingPlatformExtraction,
        extraction_name="starting_platform_equipment"
    )
    water_quality_task = evaluator.extract(
        prompt=prompt_extract_water_quality(),
        template_class=WaterQualityExtraction,
        extraction_name="water_quality_systems"
    )
    lighting_task = evaluator.extract(
        prompt=prompt_extract_lighting(),
        template_class=LightingExtraction,
        extraction_name="lighting_systems"
    )
    spectator_task = evaluator.extract(
        prompt=prompt_extract_spectator(),
        template_class=SpectatorFacilitiesExtraction,
        extraction_name="spectator_facilities"
    )

    (
        measurement_data,
        geometry_data,
        lane_data,
        starting_platform_data,
        water_quality_data,
        lighting_data,
        spectator_data,
    ) = await asyncio.gather(
        measurement_task,
        geometry_task,
        lane_task,
        starting_platform_task,
        water_quality_task,
        lighting_task,
        spectator_task
    )

    # Add ground truth info for reference
    evaluator.add_ground_truth({
        "requirements": GROUND_TRUTH_REQUIREMENTS,
        "notes": "These constraints summarize mandatory World Aquatics (FINA) certification requirements for Olympic swimming pools, used as evaluation targets."
    }, gt_type="world_aquatics_requirements")

    # Add simple custom info: number of official URLs extracted per category
    evaluator.add_custom_info(
        {
            "measurement_sources": len(measurement_data.sources or []),
            "geometry_sources": len(geometry_data.sources or []),
            "lane_sources": len(lane_data.sources or []),
            "starting_platform_sources": len(starting_platform_data.sources or []),
            "water_quality_sources": len(water_quality_data.sources or []),
            "lighting_sources": len(lighting_data.sources or []),
            "spectator_sources": len(spectator_data.sources or []),
        },
        info_type="source_counts",
        info_name="official_source_url_counts"
    )

    # Build and verify the full tree according to JSON structure
    await verify_measurement_certification(evaluator, compliance_root, measurement_data)
    await verify_pool_geometry_structure(evaluator, compliance_root, geometry_data)
    await verify_lane_configuration_system(evaluator, compliance_root, lane_data)
    await verify_starting_platform_equipment(evaluator, compliance_root, starting_platform_data)
    await verify_water_quality_systems(evaluator, compliance_root, water_quality_data)
    await verify_lighting_systems(evaluator, compliance_root, lighting_data)
    await verify_spectator_facilities(evaluator, compliance_root, spectator_data)

    # Return structured summary
    return evaluator.get_summary()