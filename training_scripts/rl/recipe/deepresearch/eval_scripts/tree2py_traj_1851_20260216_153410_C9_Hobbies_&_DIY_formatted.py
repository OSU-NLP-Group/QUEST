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
TASK_ID = "diy_holiday_tutorials"
TASK_DESCRIPTION = """Find four comprehensive online DIY tutorial resources for creating handmade holiday items suitable for craft fair sales or professional gift-giving. The four tutorials must each come from a different project category:

1. An edible architectural project: A gingerbread house construction tutorial
2. A natural decorative project: A fresh evergreen wreath-making tutorial
3. A textile wearable project: A knitted Christmas stocking pattern/tutorial
4. A wooden functional project: A handmade wooden advent calendar construction tutorial

Each tutorial must provide professional-level specifications meeting the following standards:

For the gingerbread house tutorial:
- Exact wall template dimensions (height and width measurements in inches)
- Royal icing recipe with specific ingredient ratios (powdered sugar, meringue powder or egg whites, water quantities)
- Icing consistency specifications distinguishing between structural/construction consistency and decorative consistency
- Structural stability techniques (e.g., drying time requirements between assembly steps)

For the evergreen wreath tutorial:
- Specific evergreen variety recommendations that retain needles well (e.g., juniper, white pine, Douglas fir, cedar)
- Floral wire gauge specification for assembly (exact gauge number)
- Wreath form diameter measurement
- At least one preservation method to maintain freshness (water misting schedule, glycerin treatment, or foliage sealer application)

For the knitted Christmas stocking tutorial:
- Yarn weight specification (exact weight category, e.g., worsted #4, bulky #6)
- Gauge measurement (stitches per inch)
- Heel construction technique description (e.g., heel flap with gusset, short-row heel, or other method)
- Finished stocking dimensions (length from cuff to toe in inches)

For the wooden advent calendar tutorial:
- Overall finished dimensions of the calendar
- Specification that it contains exactly 24 compartments or drawers
- Base material specification (wood type, plywood grade, and/or thickness)
- Individual compartment/drawer dimensions suitable for small gift items

For each tutorial, all numerical measurements must be explicitly stated (not described as "appropriate size" or "as needed"), material types must be specifically identified (not generic terms), and construction techniques must include sufficient detail for an intermediate crafter to successfully complete the project."""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class GingerbreadExtraction(BaseModel):
    tutorial_urls: List[str] = Field(default_factory=list)
    wall_height_in: Optional[str] = None
    wall_width_in: Optional[str] = None
    dimensions_sources: List[str] = Field(default_factory=list)

    powdered_sugar_qty: Optional[str] = None
    binding_agent: Optional[str] = None  # e.g., "meringue powder" or "egg whites"
    binding_agent_qty: Optional[str] = None
    water_qty: Optional[str] = None
    icing_recipe_sources: List[str] = Field(default_factory=list)

    structural_consistency_desc: Optional[str] = None
    decorative_consistency_desc: Optional[str] = None
    consistency_sources: List[str] = Field(default_factory=list)

    stability_technique_desc: Optional[str] = None
    drying_time_desc: Optional[str] = None
    stability_sources: List[str] = Field(default_factory=list)


class WreathExtraction(BaseModel):
    tutorial_urls: List[str] = Field(default_factory=list)
    evergreen_variety: Optional[str] = None
    needle_retention_desc: Optional[str] = None
    variety_sources: List[str] = Field(default_factory=list)

    wire_gauge: Optional[str] = None
    wire_sources: List[str] = Field(default_factory=list)

    wreath_diameter_in: Optional[str] = None
    diameter_sources: List[str] = Field(default_factory=list)

    preservation_method_desc: Optional[str] = None
    preservation_sources: List[str] = Field(default_factory=list)


class StockingExtraction(BaseModel):
    tutorial_urls: List[str] = Field(default_factory=list)
    yarn_weight: Optional[str] = None
    yarn_sources: List[str] = Field(default_factory=list)

    gauge_stitches_per_inch: Optional[str] = None  # e.g., "5 sts per inch" or "20 sts per 4 inches"
    gauge_sources: List[str] = Field(default_factory=list)

    heel_technique: Optional[str] = None
    heel_sources: List[str] = Field(default_factory=list)

    finished_length_in: Optional[str] = None
    dimensions_sources: List[str] = Field(default_factory=list)


class AdventExtraction(BaseModel):
    tutorial_urls: List[str] = Field(default_factory=list)
    overall_height_in: Optional[str] = None
    overall_width_in: Optional[str] = None
    overall_dim_sources: List[str] = Field(default_factory=list)

    compartment_count: Optional[str] = None  # Expect "24" or "24 drawers"
    count_sources: List[str] = Field(default_factory=list)

    base_material_spec: Optional[str] = None  # e.g., "1/2\" birch plywood", "pine"
    material_sources: List[str] = Field(default_factory=list)

    compartment_dimensions: Optional[str] = None
    comp_dim_sources: List[str] = Field(default_factory=list)


class TutorialsExtraction(BaseModel):
    gingerbread: Optional[GingerbreadExtraction] = None
    wreath: Optional[WreathExtraction] = None
    stocking: Optional[StockingExtraction] = None
    advent: Optional[AdventExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract structured information for four separate tutorials from the answer text, one per category. Only extract what is explicitly present in the answer; do not invent anything. Include all URLs exactly as written.

For each category, provide:

gingerbread:
- tutorial_urls: list of all URLs in the answer that point to the gingerbread house tutorial or related support pages
- wall_height_in: wall height (in inches) as text (e.g., "4 inches", "4 in", "4\"")
- wall_width_in: wall width (in inches) as text
- dimensions_sources: list of URL(s) that the answer cites for these dimensions (if not separate, reuse tutorial_urls)
- powdered_sugar_qty: exact powdered sugar quantity from the royal icing recipe (e.g., "4 cups")
- binding_agent: the binding agent name ("meringue powder" or "egg whites")
- binding_agent_qty: exact quantity for the binding agent (e.g., "3 tbsp", "2 egg whites")
- water_qty: exact water quantity
- icing_recipe_sources: list of URL(s) the answer cites for the icing recipe (if not separate, reuse tutorial_urls)
- structural_consistency_desc: description of structural/construction icing consistency as in the answer
- decorative_consistency_desc: description of decorative piping icing consistency as in the answer
- consistency_sources: list of URL(s) the answer cites for icing consistency (if not separate, reuse tutorial_urls)
- stability_technique_desc: description of a structural stability technique
- drying_time_desc: drying/setting time between assembly steps as text (e.g., "30 minutes", "1 hour")
- stability_sources: list of URL(s) the answer cites for stability techniques (if not separate, reuse tutorial_urls)

wreath:
- tutorial_urls: list of all URLs for the wreath tutorial or support pages
- evergreen_variety: specific named variety recommended (e.g., "juniper", "white pine", "Douglas fir", "cedar", "arborvitae")
- needle_retention_desc: text indicating it retains needles well or is suitable for wreaths
- variety_sources: URL(s) for variety info (or reuse tutorial_urls)
- wire_gauge: exact floral wire gauge number (e.g., "22 gauge", "#22")
- wire_sources: URL(s) for wire gauge (or reuse tutorial_urls)
- wreath_diameter_in: wreath form diameter in inches as text (e.g., "12 inches", "18\"")
- diameter_sources: URL(s) for diameter (or reuse tutorial_urls)
- preservation_method_desc: a specific freshness preservation method description (e.g., water misting schedule, glycerin treatment, foliage sealer)
- preservation_sources: URL(s) for the preservation method (or reuse tutorial_urls)

stocking:
- tutorial_urls: list of URLs for the knitted stocking tutorial or support pages
- yarn_weight: yarn weight category as text (e.g., "worsted #4", "bulky #6", "DK #3")
- yarn_sources: URL(s) for yarn weight (or reuse tutorial_urls)
- gauge_stitches_per_inch: gauge as text (e.g., "5 sts per inch", "20 sts per 4 inches")
- gauge_sources: URL(s) for gauge (or reuse tutorial_urls)
- heel_technique: specific heel construction method name (e.g., "heel flap with gusset", "short-row heel")
- heel_sources: URL(s) for heel technique (or reuse tutorial_urls)
- finished_length_in: finished length from cuff to toe in inches as text (e.g., "18 inches")
- dimensions_sources: URL(s) for finished dimensions (or reuse tutorial_urls)

advent:
- tutorial_urls: list of URLs for the wooden advent calendar tutorial or support pages
- overall_height_in: overall finished height in inches as text
- overall_width_in: overall finished width in inches as text
- overall_dim_sources: URL(s) for overall dimensions (or reuse tutorial_urls)
- compartment_count: text indicating number of compartments/drawers (expect "24")
- count_sources: URL(s) for compartment count (or reuse tutorial_urls)
- base_material_spec: base material specification text (e.g., wood type, plywood grade/thickness)
- material_sources: URL(s) for material specification (or reuse tutorial_urls)
- compartment_dimensions: individual compartment/drawer dimensions as text (e.g., "2\" x 2\" x 2\"")
- comp_dim_sources: URL(s) for compartment dimensions (or reuse tutorial_urls)

Notes:
- Return null for any field not explicitly present in the answer.
- URLs must be extracted exactly as they appear; include markdown link targets as raw URLs.
- Preserve measurement units and formatting as in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_and_dedup_urls(urls: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    if not urls:
        return out
    for u in urls:
        if not isinstance(u, str):
            continue
        uu = u.strip()
        if not uu:
            continue
        if not (uu.startswith("http://") or uu.startswith("https://")):
            if "://" not in uu:
                uu = "http://" + uu
        if uu not in out:
            out.append(uu)
    return out


def _effective_sources(primary: Optional[List[str]], fallback: Optional[List[str]]) -> List[str]:
    prim = _normalize_and_dedup_urls(primary or [])
    if prim:
        return prim
    return _normalize_and_dedup_urls(fallback or [])


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_gingerbread_checks(evaluator: Evaluator, parent, g: Optional[GingerbreadExtraction]) -> None:
    node_cat = evaluator.add_parallel(
        id="Gingerbread_House_Tutorial",
        desc="A tutorial for constructing a gingerbread house with architectural and structural specifications",
        parent=parent,
        critical=False
    )

    tutorial_urls = _normalize_and_dedup_urls(g.tutorial_urls if g else [])

    # Dimensions block (critical)
    dim_node = evaluator.add_parallel(
        id="Gingerbread_Dimensions",
        desc="Wall template dimensions are provided with specific measurements",
        parent=node_cat,
        critical=True
    )

    # Dimensions reference existence (critical sibling gate)
    dim_sources = _effective_sources(g.dimensions_sources if g else [], tutorial_urls)
    evaluator.add_custom_node(
        result=len(dim_sources) > 0,
        id="Dimensions_Reference",
        desc="URL reference provided for gingerbread house dimensions",
        parent=dim_node,
        critical=True
    )

    wall_meas_node = evaluator.add_parallel(
        id="Wall_Measurements",
        desc="Wall height and width measurements are explicitly specified",
        parent=dim_node,
        critical=True
    )
    # Wall height
    height_leaf = evaluator.add_leaf(
        id="Wall_Height_Specified",
        desc="Wall height measurement is provided in inches with explicit numerical value",
        parent=wall_meas_node,
        critical=True
    )
    height_claim = f"The tutorial page lists a gingerbread wall height of '{(g.wall_height_in if g else '')}' in its template dimensions."
    await evaluator.verify(
        claim=height_claim,
        node=height_leaf,
        sources=dim_sources,
        additional_instruction="Verify that a wall height measurement matching this value appears. Accept inch markers like in, inches, \"."
    )

    # Wall width
    width_leaf = evaluator.add_leaf(
        id="Wall_Width_Specified",
        desc="Wall width measurement is provided in inches with explicit numerical value",
        parent=wall_meas_node,
        critical=True
    )
    width_claim = f"The tutorial page lists a gingerbread wall width of '{(g.wall_width_in if g else '')}' in its template dimensions."
    await evaluator.verify(
        claim=width_claim,
        node=width_leaf,
        sources=dim_sources,
        additional_instruction="Verify that a wall width measurement matching this value appears. Accept inch markers like in, inches, \"."
    )

    # Royal icing recipe (critical)
    icing_node = evaluator.add_parallel(
        id="Royal_Icing_Recipe",
        desc="Royal icing recipe with specific ingredient ratios is provided",
        parent=node_cat,
        critical=True
    )

    icing_sources = _effective_sources(g.icing_recipe_sources if g else [], tutorial_urls)
    evaluator.add_custom_node(
        result=len(icing_sources) > 0,
        id="Recipe_Reference",
        desc="URL reference provided for royal icing recipe",
        parent=icing_node,
        critical=True
    )

    ing_node = evaluator.add_parallel(
        id="Ingredient_Specifications",
        desc="Exact quantities for all required ingredients are specified",
        parent=icing_node,
        critical=True
    )

    # Powdered sugar
    ps_leaf = evaluator.add_leaf(
        id="Powdered_Sugar_Quantity",
        desc="Exact quantity of powdered sugar is specified",
        parent=ing_node,
        critical=True
    )
    ps_claim = f"The royal icing recipe specifies powdered sugar quantity as '{(g.powdered_sugar_qty if g else '')}'."
    await evaluator.verify(
        claim=ps_claim,
        node=ps_leaf,
        sources=icing_sources,
        additional_instruction="Check the recipe text for the powdered sugar amount exactly or an obvious equivalent."
    )

    # Binding agent
    bind_leaf = evaluator.add_leaf(
        id="Binding_Agent_Specified",
        desc="Meringue powder or egg whites quantity is specified",
        parent=ing_node,
        critical=True
    )
    bind_text = ""
    if g and g.binding_agent and g.binding_agent_qty:
        bind_text = f"{g.binding_agent_qty} of {g.binding_agent}"
    elif g and g.binding_agent:
        bind_text = g.binding_agent
    elif g and g.binding_agent_qty:
        bind_text = g.binding_agent_qty
    bind_claim = f"The royal icing recipe specifies binding agent as '{bind_text}'."
    await evaluator.verify(
        claim=bind_claim,
        node=bind_leaf,
        sources=icing_sources,
        additional_instruction="Binding agent should be meringue powder (with quantity) or egg whites (with quantity). Accept minor formatting variations."
    )

    # Water
    water_leaf = evaluator.add_leaf(
        id="Water_Quantity",
        desc="Exact quantity of water is specified",
        parent=ing_node,
        critical=True
    )
    water_claim = f"The royal icing recipe specifies water quantity as '{(g.water_qty if g else '')}'."
    await evaluator.verify(
        claim=water_claim,
        node=water_leaf,
        sources=icing_sources,
        additional_instruction="Verify the exact water amount appears on the page. Accept minor unit formatting variations."
    )

    # Icing consistency (critical)
    cons_node = evaluator.add_parallel(
        id="Icing_Consistency_Specifications",
        desc="Tutorial distinguishes between icing consistencies for different purposes",
        parent=node_cat,
        critical=True
    )

    cons_sources = _effective_sources(g.consistency_sources if g else [], tutorial_urls)
    evaluator.add_custom_node(
        result=len(cons_sources) > 0,
        id="Consistency_Reference",
        desc="URL reference provided for icing consistency information",
        parent=cons_node,
        critical=True
    )

    cons_desc_node = evaluator.add_parallel(
        id="Consistency_Descriptions",
        desc="Both structural and decorative icing consistencies are described",
        parent=cons_node,
        critical=True
    )

    # Structural consistency
    struct_cons_leaf = evaluator.add_leaf(
        id="Structural_Consistency_Described",
        desc="Icing consistency for structural/construction assembly is described",
        parent=cons_desc_node,
        critical=True
    )
    struct_claim = f"The structural/construction icing consistency is described as: '{(g.structural_consistency_desc if g else '')}'."
    await evaluator.verify(
        claim=struct_claim,
        node=struct_cons_leaf,
        sources=cons_sources,
        additional_instruction="Look for stiff/glue-like consistency guidance for building walls/roof. Accept equivalent phrasing."
    )

    # Decorative consistency
    deco_cons_leaf = evaluator.add_leaf(
        id="Decorative_Consistency_Described",
        desc="Icing consistency for decorative piping is described",
        parent=cons_desc_node,
        critical=True
    )
    deco_claim = f"The decorative piping icing consistency is described as: '{(g.decorative_consistency_desc if g else '')}'."
    await evaluator.verify(
        claim=deco_claim,
        node=deco_cons_leaf,
        sources=cons_sources,
        additional_instruction="Look for softer/piping/flood consistency description for decoration. Accept equivalent phrasing."
    )

    # Structural stability (critical)
    stab_node = evaluator.add_parallel(
        id="Structural_Stability_Techniques",
        desc="At least one structural stability technique is included",
        parent=node_cat,
        critical=True
    )

    stab_sources = _effective_sources(g.stability_sources if g else [], tutorial_urls)
    evaluator.add_custom_node(
        result=len(stab_sources) > 0,
        id="Stability_Reference",
        desc="URL reference provided for structural stability techniques",
        parent=stab_node,
        critical=True
    )

    stab_detail_node = evaluator.add_parallel(
        id="Stability_Details",
        desc="Structural stability technique details are provided",
        parent=stab_node,
        critical=True
    )

    drying_leaf = evaluator.add_leaf(
        id="Drying_Time_Specified",
        desc="Drying or setting time requirements between assembly steps are specified",
        parent=stab_detail_node,
        critical=True
    )
    drying_claim = f"The tutorial specifies drying/setting time between assembly steps as: '{(g.drying_time_desc if g else '')}'."
    await evaluator.verify(
        claim=drying_claim,
        node=drying_leaf,
        sources=stab_sources,
        additional_instruction="Look for explicit wait times (minutes/hours) before proceeding to next assembly step."
    )


async def build_wreath_checks(evaluator: Evaluator, parent, w: Optional[WreathExtraction]) -> None:
    node_cat = evaluator.add_parallel(
        id="Evergreen_Wreath_Tutorial",
        desc="A tutorial for making a fresh evergreen wreath with material and preservation specifications",
        parent=parent,
        critical=False
    )
    tutorial_urls = _normalize_and_dedup_urls(w.tutorial_urls if w else [])

    # Evergreen variety (critical)
    var_node = evaluator.add_parallel(
        id="Evergreen_Variety_Specification",
        desc="At least one specific evergreen variety that retains needles well is recommended",
        parent=node_cat,
        critical=True
    )
    var_sources = _effective_sources(w.variety_sources if w else [], tutorial_urls)
    evaluator.add_custom_node(
        result=len(var_sources) > 0,
        id="Variety_Reference",
        desc="URL reference provided for evergreen variety information",
        parent=var_node,
        critical=True
    )

    var_details_node = evaluator.add_parallel(
        id="Variety_Details",
        desc="Evergreen variety name and needle retention properties are specified",
        parent=var_node,
        critical=True
    )

    variety_leaf = evaluator.add_leaf(
        id="Variety_Named",
        desc="A specific evergreen variety is named (e.g., juniper, white pine, Douglas fir, cedar, arborvitae)",
        parent=var_details_node,
        critical=True
    )
    variety_claim = f"The tutorial recommends using the specific evergreen variety '{(w.evergreen_variety if w else '')}'."
    await evaluator.verify(
        claim=variety_claim,
        node=variety_leaf,
        sources=var_sources,
        additional_instruction="Confirm that the page explicitly names this variety for wreath making."
    )

    needle_leaf = evaluator.add_leaf(
        id="Needle_Retention_Property",
        desc="The tutorial indicates the variety retains needles well or is suitable for wreaths",
        parent=var_details_node,
        critical=True
    )
    needle_claim = f"The tutorial indicates that this variety retains needles well or is suitable for wreaths: '{(w.needle_retention_desc if w else '')}'."
    await evaluator.verify(
        claim=needle_claim,
        node=needle_leaf,
        sources=var_sources,
        additional_instruction="Look for statements about needle retention/longevity or suitability for wreaths."
    )

    # Wire gauge (critical)
    wire_node = evaluator.add_parallel(
        id="Floral_Wire_Gauge",
        desc="Exact gauge number of floral wire for assembly is specified",
        parent=node_cat,
        critical=True
    )
    wire_sources = _effective_sources(w.wire_sources if w else [], tutorial_urls)
    evaluator.add_custom_node(
        result=len(wire_sources) > 0,
        id="Wire_Reference",
        desc="URL reference provided for wire gauge specification",
        parent=wire_node,
        critical=True
    )

    wire_spec_node = evaluator.add_parallel(
        id="Wire_Specification",
        desc="Specific wire gauge number is provided",
        parent=wire_node,
        critical=True
    )
    gauge_leaf = evaluator.add_leaf(
        id="Gauge_Number_Provided",
        desc="A specific wire gauge number is provided (not just 'wire' or 'thin wire')",
        parent=wire_spec_node,
        critical=True
    )
    gauge_claim = f"The tutorial specifies floral wire gauge as '{(w.wire_gauge if w else '')}'."
    await evaluator.verify(
        claim=gauge_claim,
        node=gauge_leaf,
        sources=wire_sources,
        additional_instruction="Accept forms like '22 gauge', '#22', or '22-gauge'. It must indicate a specific number."
    )

    # Wreath diameter (critical)
    dia_node = evaluator.add_parallel(
        id="Wreath_Form_Diameter",
        desc="Wreath form diameter measurement is provided",
        parent=node_cat,
        critical=True
    )
    dia_sources = _effective_sources(w.diameter_sources if w else [], tutorial_urls)
    evaluator.add_custom_node(
        result=len(dia_sources) > 0,
        id="Diameter_Reference",
        desc="URL reference provided for wreath form diameter",
        parent=dia_node,
        critical=True
    )
    dia_spec_node = evaluator.add_parallel(
        id="Diameter_Specification",
        desc="Diameter measurement in inches is specified",
        parent=dia_node,
        critical=True
    )
    dia_leaf = evaluator.add_leaf(
        id="Diameter_Measurement",
        desc="Diameter is specified in inches with numerical value",
        parent=dia_spec_node,
        critical=True
    )
    dia_claim = f"The wreath form diameter is specified as '{(w.wreath_diameter_in if w else '')}'."
    await evaluator.verify(
        claim=dia_claim,
        node=dia_leaf,
        sources=dia_sources,
        additional_instruction="Confirm a numeric diameter appears (accept \", in, inches)."
    )

    # Preservation method (critical)
    pres_node = evaluator.add_parallel(
        id="Preservation_Method",
        desc="At least one method to preserve wreath freshness is included",
        parent=node_cat,
        critical=True
    )
    pres_sources = _effective_sources(w.preservation_sources if w else [], tutorial_urls)
    evaluator.add_custom_node(
        result=len(pres_sources) > 0,
        id="Preservation_Reference",
        desc="URL reference provided for preservation method",
        parent=pres_node,
        critical=True
    )
    pres_det_node = evaluator.add_parallel(
        id="Method_Details",
        desc="Specific preservation method is described",
        parent=pres_node,
        critical=True
    )
    pres_leaf = evaluator.add_leaf(
        id="Method_Described",
        desc="A specific preservation method is described (water misting, glycerin treatment, or foliage sealer)",
        parent=pres_det_node,
        critical=True
    )
    pres_claim = f"The tutorial describes a freshness preservation method: '{(w.preservation_method_desc if w else '')}'."
    await evaluator.verify(
        claim=pres_claim,
        node=pres_leaf,
        sources=pres_sources,
        additional_instruction="Accept methods like water misting schedule, glycerin treatment, or foliage sealer application."
    )


async def build_stocking_checks(evaluator: Evaluator, parent, s: Optional[StockingExtraction]) -> None:
    node_cat = evaluator.add_parallel(
        id="Knitted_Stocking_Tutorial",
        desc="A tutorial or pattern for knitting a Christmas stocking with yarn and construction specifications",
        parent=parent,
        critical=False
    )
    tutorial_urls = _normalize_and_dedup_urls(s.tutorial_urls if s else [])

    # Yarn weight (critical)
    yarn_node = evaluator.add_parallel(
        id="Yarn_Weight_Specification",
        desc="Yarn weight is specified using standard yarn weight categories",
        parent=node_cat,
        critical=True
    )
    yarn_sources = _effective_sources(s.yarn_sources if s else [], tutorial_urls)
    evaluator.add_custom_node(
        result=len(yarn_sources) > 0,
        id="Yarn_Reference",
        desc="URL reference provided for yarn weight specification",
        parent=yarn_node,
        critical=True
    )
    yarn_details_node = evaluator.add_parallel(
        id="Weight_Details",
        desc="Yarn weight category is stated",
        parent=yarn_node,
        critical=True
    )
    yarn_leaf = evaluator.add_leaf(
        id="Weight_Category_Stated",
        desc="Yarn weight category is stated (e.g., worsted #4, bulky #6, DK #3)",
        parent=yarn_details_node,
        critical=True
    )
    yarn_claim = f"The pattern specifies yarn weight category as '{(s.yarn_weight if s else '')}'."
    await evaluator.verify(
        claim=yarn_claim,
        node=yarn_leaf,
        sources=yarn_sources,
        additional_instruction="Look for standard categories like worsted (#4), bulky (#5/#6), DK (#3), etc."
    )

    # Gauge (critical)
    gauge_node = evaluator.add_parallel(
        id="Gauge_Measurement",
        desc="Gauge measurement is provided",
        parent=node_cat,
        critical=True
    )
    gauge_sources = _effective_sources(s.gauge_sources if s else [], tutorial_urls)
    evaluator.add_custom_node(
        result=len(gauge_sources) > 0,
        id="Gauge_Reference",
        desc="URL reference provided for gauge measurement",
        parent=gauge_node,
        critical=True
    )
    gauge_details_node = evaluator.add_parallel(
        id="Gauge_Details",
        desc="Gauge expressed as stitches per measurement unit",
        parent=gauge_node,
        critical=True
    )
    gauge_leaf = evaluator.add_leaf(
        id="Stitches_Per_Measurement",
        desc="Gauge is expressed as stitches per inch or stitches per 4 inches",
        parent=gauge_details_node,
        critical=True
    )
    gauge_claim = f"The pattern provides gauge as '{(s.gauge_stitches_per_inch if s else '')}'."
    await evaluator.verify(
        claim=gauge_claim,
        node=gauge_leaf,
        sources=gauge_sources,
        additional_instruction="Accept 'sts per inch' or 'sts per 4 inches' formats with matching values."
    )

    # Heel technique (critical)
    heel_node = evaluator.add_parallel(
        id="Heel_Construction_Technique",
        desc="Heel construction technique is described",
        parent=node_cat,
        critical=True
    )
    heel_sources = _effective_sources(s.heel_sources if s else [], tutorial_urls)
    evaluator.add_custom_node(
        result=len(heel_sources) > 0,
        id="Heel_Reference",
        desc="URL reference provided for heel construction technique",
        parent=heel_node,
        critical=True
    )
    heel_details_node = evaluator.add_parallel(
        id="Technique_Details",
        desc="Specific heel construction method is named",
        parent=heel_node,
        critical=True
    )
    heel_leaf = evaluator.add_leaf(
        id="Technique_Named",
        desc="A specific heel construction method is named (e.g., heel flap with gusset, short-row heel)",
        parent=heel_details_node,
        critical=True
    )
    heel_claim = f"The pattern names the heel construction method as '{(s.heel_technique if s else '')}'."
    await evaluator.verify(
        claim=heel_claim,
        node=heel_leaf,
        sources=heel_sources,
        additional_instruction="Look for specific techniques like 'heel flap with gusset', 'short-row heel', 'afterthought heel', etc."
    )

    # Finished dimensions (critical)
    dim_node = evaluator.add_parallel(
        id="Finished_Dimensions",
        desc="Finished stocking dimensions are stated",
        parent=node_cat,
        critical=True
    )
    dim_sources = _effective_sources(s.dimensions_sources if s else [], tutorial_urls)
    evaluator.add_custom_node(
        result=len(dim_sources) > 0,
        id="Dimensions_Reference_Stocking",
        desc="URL reference provided for finished dimensions",
        parent=dim_node,
        critical=True
    )
    dim_details_node = evaluator.add_parallel(
        id="Dimension_Details",
        desc="Length measurement from cuff to toe is specified",
        parent=dim_node,
        critical=True
    )
    length_leaf = evaluator.add_leaf(
        id="Length_Measurement",
        desc="Length from cuff to toe is specified in inches",
        parent=dim_details_node,
        critical=True
    )
    length_claim = f"The finished stocking length (cuff to toe) is '{(s.finished_length_in if s else '')}'."
    await evaluator.verify(
        claim=length_claim,
        node=length_leaf,
        sources=dim_sources,
        additional_instruction="Confirm a numeric inches length appears (accept in, inches, \")."
    )


async def build_advent_checks(evaluator: Evaluator, parent, a: Optional[AdventExtraction]) -> None:
    node_cat = evaluator.add_parallel(
        id="Wooden_Advent_Calendar_Tutorial",
        desc="A tutorial for constructing a wooden advent calendar with material and compartment specifications",
        parent=parent,
        critical=False
    )
    tutorial_urls = _normalize_and_dedup_urls(a.tutorial_urls if a else [])

    # Overall finished dimensions (critical)
    overall_node = evaluator.add_parallel(
        id="Overall_Finished_Dimensions",
        desc="Overall finished dimensions of the completed calendar are specified",
        parent=node_cat,
        critical=True
    )
    overall_sources = _effective_sources(a.overall_dim_sources if a else [], tutorial_urls)
    evaluator.add_custom_node(
        result=len(overall_sources) > 0,
        id="Overall_Dimensions_Reference",
        desc="URL reference provided for overall finished dimensions",
        parent=overall_node,
        critical=True
    )
    overall_spec_node = evaluator.add_parallel(
        id="Dimension_Specifications",
        desc="Height and width measurements are provided",
        parent=overall_node,
        critical=True
    )
    overall_leaf = evaluator.add_leaf(
        id="Dimensions_Provided",
        desc="Height and width (or length) measurements are provided in inches",
        parent=overall_spec_node,
        critical=True
    )
    overall_claim = f"The tutorial specifies overall finished height '{(a.overall_height_in if a else '')}' and width '{(a.overall_width_in if a else '')}'."
    await evaluator.verify(
        claim=overall_claim,
        node=overall_leaf,
        sources=overall_sources,
        additional_instruction="Confirm both height and width measurements appear (accept inches formats like in, inches, \")."
    )

    # Compartment count (critical)
    count_node = evaluator.add_parallel(
        id="Compartment_Count",
        desc="The calendar is specified to contain exactly 24 compartments or drawers",
        parent=node_cat,
        critical=True
    )
    count_sources = _effective_sources(a.count_sources if a else [], tutorial_urls)
    evaluator.add_custom_node(
        result=len(count_sources) > 0,
        id="Count_Reference",
        desc="URL reference provided for compartment count",
        parent=count_node,
        critical=True
    )
    count_spec_node = evaluator.add_parallel(
        id="Count_Specification",
        desc="Tutorial explicitly states 24 compartments/drawers",
        parent=count_node,
        critical=True
    )
    count_leaf = evaluator.add_leaf(
        id="Twenty_Four_Compartments",
        desc="Tutorial explicitly states 24 compartments/drawers for Advent countdown",
        parent=count_spec_node,
        critical=True
    )
    count_claim = "The tutorial specifies exactly 24 compartments or drawers."
    await evaluator.verify(
        claim=count_claim,
        node=count_leaf,
        sources=count_sources,
        additional_instruction="Confirm that the page explicitly mentions 24 compartments/drawers."
    )

    # Base material specification (critical)
    mat_node = evaluator.add_parallel(
        id="Base_Material_Specification",
        desc="Base construction material is specified with details",
        parent=node_cat,
        critical=True
    )
    mat_sources = _effective_sources(a.material_sources if a else [], tutorial_urls)
    evaluator.add_custom_node(
        result=len(mat_sources) > 0,
        id="Material_Reference",
        desc="URL reference provided for base material specification",
        parent=mat_node,
        critical=True
    )
    mat_detail_node = evaluator.add_parallel(
        id="Material_Details",
        desc="Wood type, plywood grade, or thickness is specified",
        parent=mat_node,
        critical=True
    )
    mat_leaf = evaluator.add_leaf(
        id="Material_Type_Identified",
        desc="Wood type, plywood grade, or material thickness is specified (not just 'wood')",
        parent=mat_detail_node,
        critical=True
    )
    mat_claim = f"The base construction material is specified as '{(a.base_material_spec if a else '')}'."
    await evaluator.verify(
        claim=mat_claim,
        node=mat_leaf,
        sources=mat_sources,
        additional_instruction="Look for specific wood types (e.g., pine, birch plywood) and/or thickness/grade."
    )

    # Individual compartment dimensions (critical)
    comp_node = evaluator.add_parallel(
        id="Individual_Compartment_Dimensions",
        desc="Individual compartment or drawer dimensions are provided",
        parent=node_cat,
        critical=True
    )
    comp_sources = _effective_sources(a.comp_dim_sources if a else [], tutorial_urls)
    evaluator.add_custom_node(
        result=len(comp_sources) > 0,
        id="Compartment_Dimensions_Reference",
        desc="URL reference provided for individual compartment dimensions",
        parent=comp_node,
        critical=True
    )
    comp_detail_node = evaluator.add_parallel(
        id="Compartment_Size_Details",
        desc="Compartment dimensions suitable for small gifts are stated",
        parent=comp_node,
        critical=True
    )
    comp_leaf = evaluator.add_leaf(
        id="Compartment_Size_Stated",
        desc="Compartment dimensions suitable for small gift items are stated",
        parent=comp_detail_node,
        critical=True
    )
    comp_claim = f"The tutorial provides individual compartment/drawer dimensions as '{(a.compartment_dimensions if a else '')}'."
    await evaluator.verify(
        claim=comp_claim,
        node=comp_leaf,
        sources=comp_sources,
        additional_instruction="Confirm a clear numeric dimension set for each compartment/drawer appears."
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
    Evaluate an answer for the DIY holiday tutorials task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Categories are independent
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

    # Important: Set root as non-critical to allow partial credit across categories,
    # because framework enforces critical parent => all children must be critical.
    root.critical = False

    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=TutorialsExtraction,
        extraction_name="tutorials_extraction"
    )

    # Build four category subtrees
    await build_gingerbread_checks(evaluator, root, extracted.gingerbread if extracted else None)
    await build_wreath_checks(evaluator, root, extracted.wreath if extracted else None)
    await build_stocking_checks(evaluator, root, extracted.stocking if extracted else None)
    await build_advent_checks(evaluator, root, extracted.advent if extracted else None)

    return evaluator.get_summary()