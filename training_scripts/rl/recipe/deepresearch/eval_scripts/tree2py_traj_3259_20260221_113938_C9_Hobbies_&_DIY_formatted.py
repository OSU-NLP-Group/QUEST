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
TASK_ID = "holiday_parade_float_project"
TASK_DESCRIPTION = """
You are planning a comprehensive holiday parade float entry for your community organization's participation in the annual municipal holiday parade. The float will serve as both a competition entry and a mobile holiday display, incorporating multiple traditional craft elements to showcase holiday creativity.

Your parade float must meet the following requirements:

Float Structure:
- Comply with municipal regulations: maximum 12 feet height, 12 feet width, and 40 feet length (excluding towing vehicle)
- Built on a standard trailer platform between 7-8 feet wide and 14-24 feet long
- Include required fire safety equipment (minimum 2A:10BC rated fire extinguisher)

Integrated Display Components:

1. Gingerbread House Centerpiece: A structural gingerbread house built to construction-grade standards with:
   - Dough rolled to 1/4 inch thickness
   - Construction recipe using 1:4 butter-to-flour ratio
   - Royal icing that forms stiff peaks for assembly
   - Proper curing times (walls: 15-20 minutes before roof; full structure: 4-6 hours before decorating)

2. Christmas Village Scene: A miniature village display featuring:
   - Buildings at approximately 1:48 scale (Department 56 Snow Village standard)
   - Figures at approximately 60mm scale (1:32)
   - Mounted on a platform of minimum 24 inches square

3. Wreath Decorations: Fresh evergreen wreaths with:
   - 16-inch diameter frames
   - 22 or 24 gauge florist wire for assembly
   - Branch bundles cut to 6-8 inches
   - Final diameter of 24-28 inches with greenery

4. Christmas Lighting System: Meeting electrical code requirements:
   - Maximum 3 light strands connected end-to-end
   - Total electrical load not exceeding 80% of circuit capacity
   - GFCI protection for all outdoor connections

5. Interactive Advent Calendar: With 24 or 25 compartments, each approximately 2.5-3 inches in size

Safety and Installation:
- Outdoor components with IP65 or higher weather resistance rating
- Proper anchoring system using rebar stakes or equivalent
- Indoor display elements not exceeding 20-30% wall/ceiling coverage
- Minimum 3 feet clearance from heat sources
- Candles (if used) placed minimum 12 inches from combustible materials

Develop a detailed plan that specifies all the key construction specifications, materials, safety equipment, and reference documentation needed to successfully build and operate this parade float while complying with all technical and safety requirements.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DimensionSpec(BaseModel):
    height_ft: Optional[str] = None
    width_ft: Optional[str] = None
    length_ft: Optional[str] = None
    trailer_width_ft: Optional[str] = None
    trailer_length_ft: Optional[str] = None


class FireSafety(BaseModel):
    extinguisher_rating: Optional[str] = None
    extinguisher_accessibility: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class GingerbreadDough(BaseModel):
    thickness_in: Optional[str] = None
    butter_flour_ratio: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class GingerbreadAssembly(BaseModel):
    icing_stiff_peaks: Optional[str] = None
    wall_dry_time_min: Optional[str] = None
    full_cure_time_hours: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class GingerbreadHouse(BaseModel):
    dough: Optional[GingerbreadDough] = None
    assembly: Optional[GingerbreadAssembly] = None


class VillageScene(BaseModel):
    building_scale: Optional[str] = None
    figure_scale: Optional[str] = None
    platform_size_in: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Wreaths(BaseModel):
    frame_diameter_in: Optional[str] = None
    wire_gauge: Optional[str] = None
    branch_length_in: Optional[str] = None
    final_diameter_in: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class LightingSystem(BaseModel):
    max_strands_connected: Optional[str] = None
    load_percent: Optional[str] = None
    gfci_protection: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AdventCalendar(BaseModel):
    total_compartments: Optional[str] = None
    compartment_size_in: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class OutdoorInstall(BaseModel):
    weather_rating: Optional[str] = None
    anchor_method: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class IndoorSafety(BaseModel):
    coverage_percent: Optional[str] = None
    heat_clearance_ft: Optional[str] = None
    candle_clearance_in: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DocumentationRefs(BaseModel):
    specs_docs: List[str] = Field(default_factory=list)
    safety_docs: List[str] = Field(default_factory=list)
    materials_docs: List[str] = Field(default_factory=list)


class FloatPlanExtraction(BaseModel):
    dimensions: Optional[DimensionSpec] = None
    gingerbread_house: Optional[GingerbreadHouse] = None
    village_scene: Optional[VillageScene] = None
    wreaths: Optional[Wreaths] = None
    lighting: Optional[LightingSystem] = None
    advent_calendar: Optional[AdventCalendar] = None
    fire_safety: Optional[FireSafety] = None
    outdoor_install: Optional[OutdoorInstall] = None
    indoor_safety: Optional[IndoorSafety] = None
    documentation: Optional[DocumentationRefs] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_float_plan() -> str:
    return """
    Extract structured specifications from the answer for the parade float plan. Return values EXACTLY as stated in the answer text (use strings; do not infer numbers). If any field is missing, return null for that field. Also extract any URLs the answer cites for references.

    Return a JSON object matching this schema:
    {
      "dimensions": {
        "height_ft": string|null,                 // overall float height in feet (e.g., "11 ft" or "10.5 feet")
        "width_ft": string|null,                  // overall float width in feet
        "length_ft": string|null,                 // overall float length in feet (excluding towing vehicle)
        "trailer_width_ft": string|null,          // trailer platform width in feet
        "trailer_length_ft": string|null          // trailer platform length in feet
      },
      "gingerbread_house": {
        "dough": {
          "thickness_in": string|null,            // dough thickness (e.g., "1/4 inch", "0.25 in")
          "butter_flour_ratio": string|null,      // recipe ratio (e.g., "1:4")
          "sources": string[]                     // URLs related to recipe/dough specs
        },
        "assembly": {
          "icing_stiff_peaks": string|null,       // statement indicating "stiff peaks" or equivalent
          "wall_dry_time_min": string|null,       // wall dry time (e.g., "15-20 minutes")
          "full_cure_time_hours": string|null,    // full cure time (e.g., "4-6 hours")
          "sources": string[]                     // URLs for assembly/royal icing guidance
        }
      },
      "village_scene": {
        "building_scale": string|null,            // e.g., "1:48"
        "figure_scale": string|null,              // e.g., "60mm" or "1:32"
        "platform_size_in": string|null,          // e.g., "24 inches square", "24x24 in"
        "sources": string[]                       // URLs for village scale references
      },
      "wreaths": {
        "frame_diameter_in": string|null,         // e.g., "16 inch"
        "wire_gauge": string|null,                // e.g., "22 gauge", "24 gauge"
        "branch_length_in": string|null,          // e.g., "6-8 inches"
        "final_diameter_in": string|null,         // e.g., "24-28 inches"
        "sources": string[]                       // URLs for wreath construction references
      },
      "lighting": {
        "max_strands_connected": string|null,     // e.g., "3", "no more than three"
        "load_percent": string|null,              // e.g., "80% max", "not exceeding 80%"
        "gfci_protection": string|null,           // statement indicating GFCI protection
        "sources": string[]                       // URLs for electrical code references
      },
      "advent_calendar": {
        "total_compartments": string|null,        // e.g., "24", "25"
        "compartment_size_in": string|null,       // e.g., "2.5-3 inches"
        "sources": string[]                       // URLs for advent calendar references (if any)
      },
      "fire_safety": {
        "extinguisher_rating": string|null,       // e.g., "2A:10BC"
        "extinguisher_accessibility": string|null,// statement about accessibility
        "sources": string[]                       // URLs for fire safety references
      },
      "outdoor_install": {
        "weather_rating": string|null,            // e.g., "IP65", "IP66"
        "anchor_method": string|null,             // e.g., "rebar stakes", "equivalent anchoring"
        "sources": string[]                       // URLs for outdoor/weatherproofing references
      },
      "indoor_safety": {
        "coverage_percent": string|null,          // e.g., "≤ 30%", "20-30%"
        "heat_clearance_ft": string|null,         // e.g., "3 feet"
        "candle_clearance_in": string|null,       // e.g., "12 inches"
        "sources": string[]                       // URLs for indoor safety references
      },
      "documentation": {
        "specs_docs": string[],                   // URLs to construction specs/reference materials
        "safety_docs": string[],                  // URLs to safety guidelines/equipment requirements
        "materials_docs": string[]                // URLs to material lists/specs/sourcing
      }
    }

    Rules:
    - Extract URLs only if they are explicitly present in the answer. Use full URLs when available.
    - Preserve units and ranges; do not convert or normalize beyond copying exactly.
    - If the answer mentions equivalence (e.g., "stiff peaks"), record it as a string.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _src(sources: Optional[List[str]]) -> Optional[List[str]]:
    """Normalize empty source lists to None to avoid irrelevant URL checks."""
    if not sources:
        return None
    return sources


# --------------------------------------------------------------------------- #
# Verification functions: Build subtrees and leaf checks                      #
# --------------------------------------------------------------------------- #
async def verify_float_base_structure(evaluator: Evaluator, parent_node, dims: Optional[DimensionSpec]) -> None:
    node = evaluator.add_parallel(
        id="Float_Base_Structure",
        desc="Primary float platform meeting dimensional and structural requirements",
        parent=parent_node,
        critical=True
    )

    # Height <= 12 ft
    leaf_h = evaluator.add_leaf(
        id="Height_Compliance",
        desc="Float height does not exceed 12 feet from ground to highest point",
        parent=node,
        critical=True
    )
    height_val = dims.height_ft if dims else None
    claim_h = f"The plan explicitly states a float height that does not exceed 12 feet. Stated height: {height_val}."
    await evaluator.verify(
        claim=claim_h,
        node=leaf_h,
        additional_instruction="If the height is missing or exceeds 12 feet, mark incorrect. Allow equivalents like '≤12 ft'."
    )

    # Width <= 12 ft
    leaf_w = evaluator.add_leaf(
        id="Width_Compliance",
        desc="Float width does not exceed 12 feet",
        parent=node,
        critical=True
    )
    width_val = dims.width_ft if dims else None
    claim_w = f"The plan explicitly states the float width does not exceed 12 feet. Stated width: {width_val}."
    await evaluator.verify(
        claim=claim_w,
        node=leaf_w,
        additional_instruction="If the width is missing or exceeds 12 feet, mark incorrect."
    )

    # Length <= 40 ft (excluding towing vehicle)
    leaf_l = evaluator.add_leaf(
        id="Length_Compliance",
        desc="Float length does not exceed 40 feet excluding towing vehicle",
        parent=node,
        critical=True
    )
    length_val = dims.length_ft if dims else None
    claim_l = f"The plan explicitly states the float length (excluding towing vehicle) does not exceed 40 feet. Stated length: {length_val}."
    await evaluator.verify(
        claim=claim_l,
        node=leaf_l,
        additional_instruction="If the length is missing or exceeds 40 feet, mark incorrect."
    )

    # Platform base specs: width 7-8 ft, length 14-24 ft
    leaf_p = evaluator.add_leaf(
        id="Platform_Base_Specs",
        desc="Trailer platform dimensions between 7-8 feet wide and 14-24 feet long",
        parent=node,
        critical=True
    )
    tw = dims.trailer_width_ft if dims else None
    tl = dims.trailer_length_ft if dims else None
    claim_p = f"The plan specifies a trailer platform width in the 7–8 ft range and a length in the 14–24 ft range. Stated width: {tw}; stated length: {tl}."
    await evaluator.verify(
        claim=claim_p,
        node=leaf_p,
        additional_instruction="Pass only if both width (7–8 ft) and length (14–24 ft) are explicitly within ranges; if missing or out of range, mark incorrect."
    )


async def verify_gingerbread_house(evaluator: Evaluator, parent_node, gh: Optional[GingerbreadHouse]) -> None:
    g_node = evaluator.add_parallel(
        id="Gingerbread_House_Display",
        desc="Structural gingerbread house display as centerpiece, built to construction standards",
        parent=parent_node,
        critical=True
    )

    # Dough specifications
    dough_node = evaluator.add_parallel(
        id="Dough_Specifications",
        desc="Gingerbread dough specifications and construction-grade recipe details",
        parent=g_node,
        critical=True
    )
    dough = gh.dough if gh else None

    leaf_th = evaluator.add_leaf(
        id="Dough_Thickness",
        desc="All structural pieces rolled to 1/4 inch (0.25 inches) thickness",
        parent=dough_node,
        critical=True
    )
    th = dough.thickness_in if dough else None
    claim_th = f"The plan states structural gingerbread dough pieces are rolled to 1/4 inch (0.25 in). Stated thickness: {th}."
    await evaluator.verify(
        claim=claim_th,
        node=leaf_th,
        sources=_src(dough.sources if dough else []),
        additional_instruction="Pass if the text clearly indicates 1/4 inch (0.25 in) for structural pieces; otherwise incorrect."
    )

    leaf_ratio = evaluator.add_leaf(
        id="Butter_Flour_Ratio",
        desc="Construction grade recipe uses 1:4 butter/fat to flour ratio",
        parent=dough_node,
        critical=True
    )
    ratio = dough.butter_flour_ratio if dough else None
    claim_ratio = f"The construction-grade gingerbread recipe uses a 1:4 butter/fat to flour ratio. Stated ratio: {ratio}."
    await evaluator.verify(
        claim=claim_ratio,
        node=leaf_ratio,
        sources=_src(dough.sources if dough else []),
        additional_instruction="Allow wording like '1 part butter/fat to 4 parts flour'; mark incorrect if unspecified or different."
    )

    # Assembly requirements (sequential)
    asm_node = evaluator.add_sequential(
        id="Assembly_Requirements",
        desc="Royal icing and assembly timing specifications",
        parent=g_node,
        critical=True
    )
    assembly = gh.assembly if gh else None

    leaf_ic = evaluator.add_leaf(
        id="Icing_Consistency",
        desc="Royal icing achieves stiff peaks for structural assembly",
        parent=asm_node,
        critical=True
    )
    ic = assembly.icing_stiff_peaks if assembly else None
    claim_ic = f"The plan states royal icing achieves stiff peaks for structural assembly. Stated consistency: {ic}."
    await evaluator.verify(
        claim=claim_ic,
        node=leaf_ic,
        sources=_src(assembly.sources if assembly else []),
        additional_instruction="Pass if 'stiff peaks' (or equivalent) is explicitly mentioned."
    )

    leaf_wall = evaluator.add_leaf(
        id="Wall_Drying_Time",
        desc="Walls dry 15-20 minutes before roof attachment",
        parent=asm_node,
        critical=True
    )
    wdt = assembly.wall_dry_time_min if assembly else None
    claim_wdt = f"The plan specifies walls dry 15–20 minutes before roof attachment. Stated wall drying time: {wdt}."
    await evaluator.verify(
        claim=claim_wdt,
        node=leaf_wall,
        sources=_src(assembly.sources if assembly else []),
        additional_instruction="Pass if between 15 and 20 minutes is explicitly stated; otherwise incorrect."
    )

    leaf_cure = evaluator.add_leaf(
        id="Full_Cure_Time",
        desc="Complete structure cures 4-6 hours minimum before decorating",
        parent=asm_node,
        critical=True
    )
    fct = assembly.full_cure_time_hours if assembly else None
    claim_cure = f"The plan specifies the complete structure cures 4–6 hours minimum before decorating. Stated cure time: {fct}."
    await evaluator.verify(
        claim=claim_cure,
        node=leaf_cure,
        sources=_src(assembly.sources if assembly else []),
        additional_instruction="Pass if ≥4 hours and within the 4–6 hours window is clearly indicated."
    )


async def verify_christmas_village_scene(evaluator: Evaluator, parent_node, vs: Optional[VillageScene]) -> None:
    node = evaluator.add_parallel(
        id="Christmas_Village_Scene",
        desc="Miniature Christmas village display with scale-appropriate buildings and figures",
        parent=parent_node,
        critical=True
    )

    leaf_b = evaluator.add_leaf(
        id="Building_Scale",
        desc="Buildings approximately 1:48 scale (Department 56 Snow Village standard)",
        parent=node,
        critical=True
    )
    bscale = vs.building_scale if vs else None
    claim_b = f"The plan specifies buildings at approximately 1:48 scale. Stated building scale: {bscale}."
    await evaluator.verify(
        claim=claim_b,
        node=leaf_b,
        sources=_src(vs.sources if vs else []),
        additional_instruction="Allow approximate wording; pass if 1:48 (Snow Village) is clearly indicated."
    )

    leaf_f = evaluator.add_leaf(
        id="Figure_Scale",
        desc="Figures approximately 60mm scale (1:32) compatible with buildings",
        parent=node,
        critical=True
    )
    fscale = vs.figure_scale if vs else None
    claim_f = f"The plan specifies figures around 60mm (1:32). Stated figure scale: {fscale}."
    await evaluator.verify(
        claim=claim_f,
        node=leaf_f,
        sources=_src(vs.sources if vs else []),
        additional_instruction="Pass if the plan states ~60mm OR 1:32 for figures."
    )

    leaf_p = evaluator.add_leaf(
        id="Platform_Dimensions",
        desc="Display base minimum 24 inches square",
        parent=node,
        critical=True
    )
    psize = vs.platform_size_in if vs else None
    claim_p = f"The display base is at least 24 inches square. Stated base size: {psize}."
    await evaluator.verify(
        claim=claim_p,
        node=leaf_p,
        sources=_src(vs.sources if vs else []),
        additional_instruction="Pass only if ≥24 inches square is explicitly stated."
    )


async def verify_wreath_decorations(evaluator: Evaluator, parent_node, wr: Optional[Wreaths]) -> None:
    node = evaluator.add_parallel(
        id="Wreath_Decorations",
        desc="Fresh evergreen wreaths as decorative elements following construction standards",
        parent=parent_node,
        critical=True
    )

    leaf_frame = evaluator.add_leaf(
        id="Frame_Diameter",
        desc="16-inch diameter frames used",
        parent=node,
        critical=True
    )
    fd = wr.frame_diameter_in if wr else None
    claim_frame = f"The wreath frames used are 16 inches in diameter. Stated frame diameter: {fd}."
    await evaluator.verify(
        claim=claim_frame,
        node=leaf_frame,
        sources=_src(wr.sources if wr else []),
        additional_instruction="Pass if '16 inch' (or equivalent) frames are explicitly stated."
    )

    leaf_wire = evaluator.add_leaf(
        id="Wire_Gauge",
        desc="22 or 24 gauge florist wire used for assembly",
        parent=node,
        critical=True
    )
    wg = wr.wire_gauge if wr else None
    claim_wire = f"The assembly uses 22 or 24 gauge florist wire. Stated wire gauge: {wg}."
    await evaluator.verify(
        claim=claim_wire,
        node=leaf_wire,
        sources=_src(wr.sources if wr else []),
        additional_instruction="Pass only if '22 gauge' or '24 gauge' is clearly indicated."
    )

    leaf_branch = evaluator.add_leaf(
        id="Branch_Length",
        desc="Branch bundles cut to 6-8 inches for assembly",
        parent=node,
        critical=True
    )
    bl = wr.branch_length_in if wr else None
    claim_branch = f"Branch bundles are cut to 6–8 inches. Stated branch length: {bl}."
    await evaluator.verify(
        claim=claim_branch,
        node=leaf_branch,
        sources=_src(wr.sources if wr else []),
        additional_instruction="Pass only if '6–8 inches' (or equivalent range) is explicitly stated."
    )

    leaf_final = evaluator.add_leaf(
        id="Final_Diameter",
        desc="Completed wreaths measure 24-28 inches diameter with greenery",
        parent=node,
        critical=True
    )
    fd2 = wr.final_diameter_in if wr else None
    claim_final = f"Completed wreaths measure 24–28 inches in diameter with greenery. Stated final diameter: {fd2}."
    await evaluator.verify(
        claim=claim_final,
        node=leaf_final,
        sources=_src(wr.sources if wr else []),
        additional_instruction="Pass only if 24–28 inches is clearly indicated."
    )


async def verify_lighting_system(evaluator: Evaluator, parent_node, ls: Optional[LightingSystem]) -> None:
    node = evaluator.add_parallel(
        id="Electrical_Lighting_System",
        desc="Christmas lighting installation meeting electrical code and safety requirements",
        parent=parent_node,
        critical=True
    )

    leaf_conn = evaluator.add_leaf(
        id="Maximum_Connections",
        desc="No more than 3 light strands connected end-to-end",
        parent=node,
        critical=True
    )
    mc = ls.max_strands_connected if ls else None
    claim_conn = f"The plan limits end-to-end connections to no more than 3 light strands. Stated max: {mc}."
    await evaluator.verify(
        claim=claim_conn,
        node=leaf_conn,
        sources=_src(ls.sources if ls else []),
        additional_instruction="Pass only if '≤3 strands' (or equivalent) is explicitly stated."
    )

    leaf_load = evaluator.add_leaf(
        id="Load_Limit",
        desc="Total electrical load does not exceed 80% of circuit capacity",
        parent=node,
        critical=True
    )
    lp = ls.load_percent if ls else None
    claim_load = f"The total electrical load does not exceed 80% of circuit capacity. Stated load guidance: {lp}."
    await evaluator.verify(
        claim=claim_load,
        node=leaf_load,
        sources=_src(ls.sources if ls else []),
        additional_instruction="Pass if the answer clearly states ≤80% cap; mark incorrect if missing or exceeding."
    )

    leaf_gfci = evaluator.add_leaf(
        id="GFCI_Required",
        desc="All outdoor electrical connections use GFCI-protected outlets",
        parent=node,
        critical=True
    )
    gfci = ls.gfci_protection if ls else None
    claim_gfci = f"All outdoor electrical connections use GFCI-protected outlets. Stated GFCI provision: {gfci}."
    await evaluator.verify(
        claim=claim_gfci,
        node=leaf_gfci,
        sources=_src(ls.sources if ls else []),
        additional_instruction="Pass only if GFCI usage for outdoor connections is explicitly stated."
    )


async def verify_advent_calendar(evaluator: Evaluator, parent_node, ac: Optional[AdventCalendar]) -> None:
    node = evaluator.add_parallel(
        id="Advent_Calendar_Component",
        desc="Interactive advent calendar for community engagement",
        parent=parent_node,
        critical=True
    )

    leaf_total = evaluator.add_leaf(
        id="Total_Compartments",
        desc="24 or 25 individual compartments or boxes",
        parent=node,
        critical=True
    )
    tc = ac.total_compartments if ac else None
    claim_total = f"The advent calendar has 24 or 25 compartments. Stated total: {tc}."
    await evaluator.verify(
        claim=claim_total,
        node=leaf_total,
        sources=_src(ac.sources if ac else []),
        additional_instruction="Pass if 24 or 25 is explicitly stated; otherwise incorrect."
    )

    leaf_size = evaluator.add_leaf(
        id="Compartment_Size",
        desc="Each compartment approximately 2.5-3 inches",
        parent=node,
        critical=True
    )
    cs = ac.compartment_size_in if ac else None
    claim_size = f"Each compartment is approximately 2.5–3 inches. Stated size: {cs}."
    await evaluator.verify(
        claim=claim_size,
        node=leaf_size,
        sources=_src(ac.sources if ac else []),
        additional_instruction="Pass if ~2.5–3 inches is explicitly indicated."
    )


async def verify_fire_safety_equipment(evaluator: Evaluator, parent_node, fs: Optional[FireSafety]) -> None:
    node = evaluator.add_parallel(
        id="Fire_Safety_Equipment",
        desc="Required fire safety equipment and extinguisher specifications",
        parent=parent_node,
        critical=True
    )

    leaf_min = evaluator.add_leaf(
        id="Minimum_Rating",
        desc="Minimum 2A:10BC rated portable fire extinguisher provided",
        parent=node,
        critical=True
    )
    rt = fs.extinguisher_rating if fs else None
    claim_min = f"The plan provides at least one portable fire extinguisher rated 2A:10BC or higher. Stated rating: {rt}."
    await evaluator.verify(
        claim=claim_min,
        node=leaf_min,
        sources=_src(fs.sources if fs else []),
        additional_instruction="Pass only if '2A:10BC' (or higher equivalent) is explicitly provided."
    )

    leaf_acc = evaluator.add_leaf(
        id="Extinguisher_Accessibility",
        desc="Extinguisher readily accessible to float operators",
        parent=node,
        critical=True
    )
    acc = fs.extinguisher_accessibility if fs else None
    claim_acc = f"The extinguisher is readily accessible to float operators. Stated accessibility: {acc}."
    await evaluator.verify(
        claim=claim_acc,
        node=leaf_acc,
        sources=_src(fs.sources if fs else []),
        additional_instruction="Pass if accessibility (easy reach during operation) is explicitly indicated."
    )


async def verify_outdoor_installation_standards(evaluator: Evaluator, parent_node, oi: Optional[OutdoorInstall]) -> None:
    node = evaluator.add_parallel(
        id="Outdoor_Installation_Standards",
        desc="Weather resistance and anchoring specifications for outdoor display components",
        parent=parent_node,
        critical=True
    )

    leaf_mat = evaluator.add_leaf(
        id="Material_Rating",
        desc="Outdoor decorations rated IP65 or higher for weather resistance",
        parent=node,
        critical=True
    )
    wr = oi.weather_rating if oi else None
    claim_mat = f"Outdoor decorations are rated IP65 or higher for weather resistance. Stated rating: {wr}."
    await evaluator.verify(
        claim=claim_mat,
        node=leaf_mat,
        sources=_src(oi.sources if oi else []),
        additional_instruction="Pass only if 'IP65' or higher (e.g., IP66, IP67) is explicitly stated."
    )

    leaf_anchor = evaluator.add_leaf(
        id="Anchor_Method",
        desc="Proper anchoring system using rebar stakes or equivalent documented and implemented",
        parent=node,
        critical=True
    )
    am = oi.anchor_method if oi else None
    claim_anchor = f"The plan uses a proper anchoring system with rebar stakes or an equivalent method. Stated method: {am}."
    await evaluator.verify(
        claim=claim_anchor,
        node=leaf_anchor,
        sources=_src(oi.sources if oi else []),
        additional_instruction="Pass if rebar stakes or an equivalent anchoring method is explicitly documented."
    )


async def verify_indoor_safety_compliance(evaluator: Evaluator, parent_node, ins: Optional[IndoorSafety]) -> None:
    node = evaluator.add_parallel(
        id="Indoor_Safety_Compliance",
        desc="Indoor decoration elements meeting fire safety codes",
        parent=parent_node,
        critical=True
    )

    leaf_cov = evaluator.add_leaf(
        id="Coverage_Percentage",
        desc="Indoor decorations do not exceed 20-30% of wall/ceiling area",
        parent=node,
        critical=True
    )
    cp = ins.coverage_percent if ins else None
    claim_cov = f"Indoor decorations do not exceed 20–30% of wall/ceiling coverage. Stated coverage: {cp}."
    await evaluator.verify(
        claim=claim_cov,
        node=leaf_cov,
        sources=_src(ins.sources if ins else []),
        additional_instruction="Pass if the plan explicitly limits indoor coverage to ≤30% and within 20–30%."
    )

    leaf_clear = evaluator.add_leaf(
        id="Heat_Source_Clearance",
        desc="Minimum 3 feet clearance from fireplaces, radiators, or heat sources",
        parent=node,
        critical=True
    )
    hc = ins.heat_clearance_ft if ins else None
    claim_clear = f"The plan maintains at least 3 feet clearance from heat sources. Stated clearance: {hc}."
    await evaluator.verify(
        claim=claim_clear,
        node=leaf_clear,
        sources=_src(ins.sources if ins else []),
        additional_instruction="Pass only if '3 feet' (or greater) clearance is explicitly stated."
    )

    leaf_cand = evaluator.add_leaf(
        id="Candle_Clearance",
        desc="Candles (if used) placed minimum 12 inches from combustible materials",
        parent=node,
        critical=True
    )
    cc = ins.candle_clearance_in if ins else None
    claim_cand = f"Any candles used are placed at least 12 inches from combustible materials. Stated clearance: {cc}."
    await evaluator.verify(
        claim=claim_cand,
        node=leaf_cand,
        sources=_src(ins.sources if ins else []),
        additional_instruction="Pass only if ≥12 inches clearance is explicitly stated when candles are used."
    )


async def verify_documentation_and_references(evaluator: Evaluator, parent_node, docs: Optional[DocumentationRefs]) -> None:
    node = evaluator.add_parallel(
        id="Documentation_And_References",
        desc="Complete documentation package with all reference materials as requested",
        parent=parent_node,
        critical=True
    )

    # Specifications documentation
    leaf_specs = evaluator.add_leaf(
        id="Specifications_Documentation",
        desc="Detailed construction specifications documented for all components with supporting reference materials",
        parent=node,
        critical=True
    )
    claim_specs = "The plan includes detailed construction specifications documented for all components, supported by reference materials."
    await evaluator.verify(
        claim=claim_specs,
        node=leaf_specs,
        sources=_src(docs.specs_docs if docs else []),
        additional_instruction="Use the provided URLs (if any). If there are no URLs or documentation clearly provided, mark incorrect."
    )

    # Safety documentation
    leaf_safety = evaluator.add_leaf(
        id="Safety_Documentation",
        desc="All applicable safety guidelines and equipment requirements documented with references",
        parent=node,
        critical=True
    )
    claim_safety = "All applicable safety guidelines and equipment requirements are documented with references."
    await evaluator.verify(
        claim=claim_safety,
        node=leaf_safety,
        sources=_src(docs.safety_docs if docs else []),
        additional_instruction="Pass only if safety references are provided via explicit documentation or URLs."
    )

    # Materials documentation
    leaf_materials = evaluator.add_leaf(
        id="Materials_Documentation",
        desc="Complete material lists with specifications and sourcing information provided",
        parent=node,
        critical=True
    )
    claim_materials = "The plan provides complete material lists with specifications and sourcing information."
    await evaluator.verify(
        claim=claim_materials,
        node=leaf_materials,
        sources=_src(docs.materials_docs if docs else []),
        additional_instruction="Pass only if material lists and sourcing information are explicitly documented, preferably with URLs."
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
    Evaluate an answer for the comprehensive holiday parade float project.
    Builds a critical verification tree mirroring the rubric and returns a structured summary.
    """
    # Initialize evaluator (framework root is non-critical; we add our critical project root under it)
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

    # Extract full plan
    plan: FloatPlanExtraction = await evaluator.extract(
        prompt=prompt_extract_float_plan(),
        template_class=FloatPlanExtraction,
        extraction_name="float_plan_extraction"
    )

    # Project root (critical)
    project_root = evaluator.add_parallel(
        id="Complete_Holiday_Parade_Float_Project",
        desc="Comprehensive parade float entry with integrated holiday displays, meeting all municipal regulations and competition requirements",
        parent=root,
        critical=True
    )

    # Build subtrees
    await verify_float_base_structure(evaluator, project_root, plan.dimensions)
    await verify_gingerbread_house(evaluator, project_root, plan.gingerbread_house)
    await verify_christmas_village_scene(evaluator, project_root, plan.village_scene)
    await verify_wreath_decorations(evaluator, project_root, plan.wreaths)
    await verify_lighting_system(evaluator, project_root, plan.lighting)
    await verify_advent_calendar(evaluator, project_root, plan.advent_calendar)
    await verify_fire_safety_equipment(evaluator, project_root, plan.fire_safety)
    await verify_outdoor_installation_standards(evaluator, project_root, plan.outdoor_install)
    await verify_indoor_safety_compliance(evaluator, project_root, plan.indoor_safety)
    await verify_documentation_and_references(evaluator, project_root, plan.documentation)

    # Optionally add ground truth constraints as context info
    evaluator.add_ground_truth({
        "constraints": {
            "height_ft_max": "12",
            "width_ft_max": "12",
            "length_ft_max": "40",
            "trailer_width_ft_range": "7-8",
            "trailer_length_ft_range": "14-24",
            "gingerbread": {
                "thickness_in": "1/4",
                "butter_flour_ratio": "1:4",
                "wall_dry_time_min": "15-20",
                "full_cure_time_hours": "4-6",
                "icing": "stiff peaks"
            },
            "village": {"building_scale": "1:48", "figure_scale": "60mm (1:32)", "platform_min_square_in": "24"},
            "wreaths": {"frame_diameter_in": "16", "wire_gauge": "22 or 24", "branch_length_in": "6-8", "final_diameter_in": "24-28"},
            "lighting": {"max_strands": "3", "load_percent_max": "80%", "gfci": "required"},
            "advent": {"compartments": "24 or 25", "size_in": "2.5-3"},
            "fire_safety": {"extinguisher_rating_min": "2A:10BC"},
            "outdoor": {"weather_rating_min": "IP65", "anchor_method": "rebar or equivalent"},
            "indoor": {"coverage_percent_max": "30%", "heat_clearance_ft": "3", "candle_clearance_in": "12"}
        }
    })

    # Return summary
    return evaluator.get_summary()