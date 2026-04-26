import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.verification_tree import VerificationNode

TASK_ID = "custom_bookshelf_project"
TASK_DESCRIPTION = """You are planning to build a custom freestanding bookshelf for your home that will be 6 feet tall with 5 adjustable shelves. The bookshelf needs to be sturdy enough to hold a full collection of books and should have a professional finished appearance.

Create a comprehensive build plan that includes:

1. Material Specifications: Provide complete specifications for all wood materials including the main panels (sides, top, bottom), adjustable shelves, and back panel. Specify the exact thickness, type of material, and dimensions. Include specifications for edge treatment of exposed plywood edges and the adjustable shelf pin system (including hole spacing standards). Provide supplier references or specification sheet URLs for the materials.

2. Hardware Requirements: List all required hardware including pocket hole screws (with correct sizes for the material thickness), wood glue type, shelf support pins, and fasteners for the back panel. Include URL references for hardware specifications.

3. Safety Equipment: Identify all required personal protective equipment (PPE) needed during construction, referencing OSHA or industry safety standards with URL documentation.

4. Construction Process: Detail the construction methodology including:
   - Cutting accuracy requirements
   - Joinery method with proper screw specifications based on material thickness
   - Shelf pin hole drilling specifications (spacing and setback from edge)
   - Assembly and squaring technique

5. Finishing Schedule: Provide a complete finishing plan including:
   - Sanding grit progression (starting grit, intermediate grits, and final grit)
   - Protective finish type and application process
   - Number of coats required and sanding requirements between coats
   - Drying time between coats
   - Product reference URL for recommended finish

6. Installation Requirements: Specify installation and safety requirements including:
   - Leveling procedures
   - Anti-tip hardware requirements (height threshold, number of anchor points, attachment method)
   - Safety hardware specifications (minimum weight rating)
   - Reference URL for anti-tip safety guidelines

Your plan must follow industry standards for woodworking construction, including the standard 32mm spacing for adjustable shelf pins, proper span limits for shelf thickness, and CPSC safety guidelines for furniture anchoring.
"""

# -----------------------------
# Data Models for Extraction
# -----------------------------

class Dimensions(BaseModel):
    sides: Optional[str] = None
    top: Optional[str] = None
    bottom: Optional[str] = None

class PanelsSpec(BaseModel):
    thickness: Optional[str] = None
    material: Optional[str] = None
    dimensions: Optional[Dimensions] = None

class ShelvesSpec(BaseModel):
    count: Optional[str] = None
    thickness: Optional[str] = None
    width: Optional[str] = None
    depth: Optional[str] = None
    span_limit_statement: Optional[str] = None
    respects_span_limit: Optional[bool] = None

class BackPanelSpec(BaseModel):
    thickness: Optional[str] = None
    material: Optional[str] = None
    dimensions: Optional[str] = None

class EdgeTreatmentSpec(BaseModel):
    edge_banding_included: Optional[bool] = None
    details: Optional[str] = None

class MaterialsModel(BaseModel):
    panels: Optional[PanelsSpec] = None
    shelves: Optional[ShelvesSpec] = None
    back_panel: Optional[BackPanelSpec] = None
    edge_treatment: Optional[EdgeTreatmentSpec] = None
    material_urls: List[str] = Field(default_factory=list)

class HardwareModel(BaseModel):
    pocket_screw_length: Optional[str] = None
    wood_glue_type: Optional[str] = None
    wood_glue_brand: Optional[str] = None
    shelf_support_pins_included: Optional[bool] = None
    back_panel_fasteners: Optional[str] = None
    hardware_urls: List[str] = Field(default_factory=list)

class PPEModel(BaseModel):
    eye_protection: Optional[str] = None
    respiratory_protection: Optional[str] = None
    hearing_protection: Optional[str] = None
    ppe_urls: List[str] = Field(default_factory=list)

class ConstructionModel(BaseModel):
    cutting_accuracy: Optional[str] = None
    joinery_method: Optional[str] = None
    screw_spec_for_3_4_in: Optional[str] = None
    shelf_pin_vertical_spacing_mm: Optional[str] = None
    shelf_pin_front_setback_mm: Optional[str] = None
    square_by_diagonals: Optional[bool] = None
    back_panel_maintains_square: Optional[bool] = None

class FinishingModel(BaseModel):
    sanding_start_grit: Optional[str] = None
    sanding_progression: List[str] = Field(default_factory=list)
    final_sanding_grit: Optional[str] = None
    finish_type: Optional[str] = None
    coat_count: Optional[str] = None
    dry_time_between_coats: Optional[str] = None
    intercoat_sanding: Optional[str] = None
    finish_product_url: Optional[str] = None

class InstallationModel(BaseModel):
    leveling_procedure: Optional[str] = None
    anti_tip_required_by_height: Optional[bool] = None
    minimum_anchor_points: Optional[str] = None
    attach_to_studs: Optional[bool] = None
    anchor_weight_rating_lb: Optional[str] = None
    anti_tip_attachment_method: Optional[str] = None
    anti_tip_guideline_url: Optional[str] = None

# -----------------------------
# Extraction Prompts
# -----------------------------

def prompt_extract_materials() -> str:
    return """
    Extract all material specifications for this bookshelf plan from the answer. Provide:
    - panels.thickness: thickness for sides, top, and bottom (e.g., "3/4 in")
    - panels.material: material type (e.g., "plywood")
    - panels.dimensions.sides: cut dimensions for side panels (e.g., "72 in x 12 in")
    - panels.dimensions.top: cut dimensions for top panel
    - panels.dimensions.bottom: cut dimensions for bottom panel
    - shelves.count: number of adjustable shelves
    - shelves.thickness: thickness for shelves (e.g., "3/4 in")
    - shelves.width: shelf width (inches or string)
    - shelves.depth: shelf depth (inches or string)
    - shelves.span_limit_statement: any statement about maximum unsupported span (e.g., "max 32 in unsupported or add center support")
    - shelves.respects_span_limit: true/false if the plan explicitly respects the 32 in unsupported span or adds support beyond 32 in
    - back_panel.thickness: thickness (e.g., "1/4 in")
    - back_panel.material: "plywood" or "hardboard"
    - back_panel.dimensions: back panel cut dimensions string
    - edge_treatment.edge_banding_included: true/false if edge banding for exposed plywood edges is included
    - edge_treatment.details: details of edge treatment (banding type/application)
    - material_urls: list of supplier/specification sheet URLs for wood materials
    Return all fields as strings where appropriate; URLs must be explicit in the answer.
    """

def prompt_extract_hardware() -> str:
    return """
    Extract all hardware requirements from the answer:
    - pocket_screw_length: the pocket hole screw length specified for 3/4 in stock (e.g., "1-1/4 in")
    - wood_glue_type: glue type (e.g., "PVA")
    - wood_glue_brand: brand if specified (e.g., "Titebond Original" or "Titebond II" or "Titebond III")
    - shelf_support_pins_included: true/false whether shelf support pins are included
    - back_panel_fasteners: fasteners for the back panel (e.g., "brad nails", "screws")
    - hardware_urls: list of URLs referencing hardware specifications
    """

def prompt_extract_ppe() -> str:
    return """
    Extract safety PPE requirements and references:
    - eye_protection: mention of safety goggles or face shield if present
    - respiratory_protection: mention of dust mask or respirator if present
    - hearing_protection: mention of hearing protection if present
    - ppe_urls: list of URL references to OSHA or industry safety standards for PPE
    """

def prompt_extract_construction() -> str:
    return """
    Extract construction methodology details:
    - cutting_accuracy: stated cutting accuracy/tolerance requirements
    - joinery_method: described joinery method (e.g., "pocket screws", "dowels")
    - screw_spec_for_3_4_in: screw spec used for 3/4 in stock in joinery (e.g., "1-1/4 in pocket screws")
    - shelf_pin_vertical_spacing_mm: vertical spacing specified for shelf pin holes (e.g., "32 mm")
    - shelf_pin_front_setback_mm: front setback from edge for shelf pin holes (e.g., "37 mm")
    - square_by_diagonals: true/false whether the plan verifies square by equal diagonals
    - back_panel_maintains_square: true/false whether the back panel is described as locking/maintaining square
    """

def prompt_extract_finishing() -> str:
    return """
    Extract finishing schedule details:
    - sanding_start_grit: starting grit (e.g., "100", "120")
    - sanding_progression: list of grits in order (e.g., ["120","150","180","220"])
    - final_sanding_grit: final sanding grit before finish (e.g., "220")
    - finish_type: protective finish type (e.g., "polyurethane")
    - coat_count: number of coats (e.g., "2", "3", or "2–3")
    - dry_time_between_coats: drying time between coats (e.g., "24 hours")
    - intercoat_sanding: details of sanding between coats (e.g., "220 after first, 320 after subsequent")
    - finish_product_url: URL for the recommended finish product
    """

def prompt_extract_installation() -> str:
    return """
    Extract installation and anti-tip requirements:
    - leveling_procedure: statement indicating a level tool is used
    - anti_tip_required_by_height: true/false if anti-tip is required for >30 in and applied to the 72 in bookshelf
    - minimum_anchor_points: number of anchor points specified (string acceptable; e.g., "2")
    - attach_to_studs: true/false indicating anchors attach to wall studs
    - anchor_weight_rating_lb: minimum per-anchor weight rating (e.g., "50 lbs")
    - anti_tip_attachment_method: description of how anti-tip hardware connects to bookshelf and wall
    - anti_tip_guideline_url: URL to CPSC or equivalent anti-tip safety guideline
    """

# -----------------------------
# Verification Builders
# -----------------------------

async def verify_materials(evaluator: Evaluator, parent: VerificationNode, m: MaterialsModel) -> None:
    materials_node = evaluator.add_parallel(
        id="Materials",
        desc="Wood material specifications for all panels/shelves/back plus edge treatment and material reference URLs.",
        parent=parent,
        critical=True
    )

    # Main Panels Spec
    main_panels_node = evaluator.add_parallel(
        id="Main_Panels_Spec",
        desc="Main panels (sides, top, bottom) specified as 3/4 in plywood with dimensions.",
        parent=materials_node,
        critical=True
    )

    mp_thick_leaf = evaluator.add_leaf(
        id="Main_Panels_Thickness_Material",
        desc="Main panels are specified as 3/4 inch thick plywood.",
        parent=main_panels_node,
        critical=True
    )
    thickness = m.panels.thickness if m and m.panels else None
    material = m.panels.material if m and m.panels else None
    claim_mp_thickness = "The plan specifies the main panels (sides, top, bottom) as 3/4 inch thick plywood."
    await evaluator.verify(
        claim=claim_mp_thickness,
        node=mp_thick_leaf,
        additional_instruction="Confirm the answer explicitly states both the thickness (3/4 in) and material (plywood) for the main panels."
    )

    mp_dims_leaf = evaluator.add_leaf(
        id="Main_Panels_Dimensions",
        desc="Provides cut dimensions for sides, top, and bottom panels.",
        parent=main_panels_node,
        critical=True
    )
    sides_dim = m.panels.dimensions.sides if m and m.panels and m.panels.dimensions else None
    top_dim = m.panels.dimensions.top if m and m.panels and m.panels.dimensions else None
    bottom_dim = m.panels.dimensions.bottom if m and m.panels and m.panels.dimensions else None
    claim_mp_dims = "The plan provides cut dimensions for the side panels and also for the top and bottom panels."
    await evaluator.verify(
        claim=claim_mp_dims,
        node=mp_dims_leaf,
        additional_instruction="Check for explicit dimensions for sides, top, and bottom panels in the answer."
    )

    # Adjustable Shelves Spec
    shelves_node = evaluator.add_parallel(
        id="Adjustable_Shelves_Spec",
        desc="Adjustable shelves specified with required thickness, quantity, span limit, and dimensions.",
        parent=materials_node,
        critical=True
    )

    shelf_count_leaf = evaluator.add_leaf(
        id="Shelf_Count",
        desc="Specifies exactly 5 adjustable shelves.",
        parent=shelves_node,
        critical=True
    )
    claim_shelf_count = "The plan specifies exactly 5 adjustable shelves."
    await evaluator.verify(
        claim=claim_shelf_count,
        node=shelf_count_leaf,
        additional_instruction="Confirm the answer states '5 adjustable shelves'."
    )

    shelf_thick_leaf = evaluator.add_leaf(
        id="Shelf_Thickness",
        desc="Adjustable shelves are specified as 3/4 inch thick plywood.",
        parent=shelves_node,
        critical=True
    )
    claim_shelf_thick = "The adjustable shelves are specified as 3/4 inch thick plywood."
    await evaluator.verify(
        claim=claim_shelf_thick,
        node=shelf_thick_leaf,
        additional_instruction="Check the answer explicitly says shelves are 3/4 in thick and plywood."
    )

    shelf_span_leaf = evaluator.add_leaf(
        id="Shelf_Max_Span",
        desc="Shelf design respects the maximum 32 inch unsupported span constraint (or adds center support if span would exceed it).",
        parent=shelves_node,
        critical=True
    )
    claim_shelf_span = "The plan respects a maximum unsupported shelf span of 32 inches or specifies a center support if the span would exceed 32 inches."
    await evaluator.verify(
        claim=claim_shelf_span,
        node=shelf_span_leaf,
        additional_instruction="Look for a statement about a 32 in max unsupported span or stipulating center support beyond that width."
    )

    shelf_dims_leaf = evaluator.add_leaf(
        id="Shelf_Dimensions",
        desc="Provides shelf dimensions (width and depth).",
        parent=shelves_node,
        critical=True
    )
    claim_shelf_dims = "The plan provides shelf dimensions including width and depth."
    await evaluator.verify(
        claim=claim_shelf_dims,
        node=shelf_dims_leaf,
        additional_instruction="Verify that both shelf width and depth are stated."
    )

    # Back Panel Spec
    back_panel_node = evaluator.add_parallel(
        id="Back_Panel_Spec",
        desc="Back panel specified with required thickness/material and dimensions.",
        parent=materials_node,
        critical=True
    )

    back_thick_leaf = evaluator.add_leaf(
        id="Back_Panel_Thickness_Material",
        desc="Back panel specified as 1/4 inch plywood or hardboard.",
        parent=back_panel_node,
        critical=True
    )
    claim_back_thick = "The plan specifies the back panel as 1/4 inch plywood or hardboard."
    await evaluator.verify(
        claim=claim_back_thick,
        node=back_thick_leaf,
        additional_instruction="Confirm the back panel thickness is 1/4 in and the material is plywood or hardboard."
    )

    back_dim_leaf = evaluator.add_leaf(
        id="Back_Panel_Dimensions",
        desc="Provides back panel dimensions.",
        parent=back_panel_node,
        critical=True
    )
    claim_back_dim = "The plan provides the back panel dimensions."
    await evaluator.verify(
        claim=claim_back_dim,
        node=back_dim_leaf,
        additional_instruction="Verify explicit cut dimensions for the back panel."
    )

    # Edge Treatment
    edge_treat_node = evaluator.add_parallel(
        id="Edge_Treatment",
        desc="Edge treatment for exposed plywood edges.",
        parent=materials_node,
        critical=True
    )

    edge_band_leaf = evaluator.add_leaf(
        id="Edge_Banding_Included",
        desc="Includes edge banding for exposed plywood edges (required).",
        parent=edge_treat_node,
        critical=True
    )
    claim_edge_band = "The plan includes edge banding for exposed plywood edges."
    await evaluator.verify(
        claim=claim_edge_band,
        node=edge_band_leaf,
        additional_instruction="Look for explicit mention of edge banding for exposed plywood edges."
    )

    edge_details_leaf = evaluator.add_leaf(
        id="Edge_Treatment_Spec_Details",
        desc="Specifies edge treatment details (e.g., edge banding type/application method).",
        parent=edge_treat_node,
        critical=True
    )
    claim_edge_details = "The plan specifies edge treatment details, such as the edge banding type and the application method."
    await evaluator.verify(
        claim=claim_edge_details,
        node=edge_details_leaf,
        additional_instruction="Check for type (e.g., wood veneer banding) and application (e.g., heat-activated adhesive)."
    )

    # Material Reference URLs
    materials_urls_leaf = evaluator.add_leaf(
        id="Materials_Reference_URLs",
        desc="Provides supplier/specification sheet URL references for the wood materials (main panels, shelves, back panel).",
        parent=materials_node,
        critical=True
    )
    mat_urls = m.material_urls if m else []
    claim_mat_urls = "These URLs are supplier or specification sheet pages for the wood materials used in this project (panels, shelves, back panel)."
    await evaluator.verify(
        claim=claim_mat_urls,
        node=materials_urls_leaf,
        sources=mat_urls,
        additional_instruction="Verify that at least one provided URL clearly references material specifications or supplier product pages for plywood or back panel materials."
    )

async def verify_hardware(evaluator: Evaluator, parent: VerificationNode, h: HardwareModel) -> None:
    hardware_node = evaluator.add_parallel(
        id="Hardware",
        desc="Hardware list with required specs and URL references.",
        parent=parent,
        critical=True
    )

    phs_node = evaluator.add_parallel(
        id="Pocket_Hole_Screws",
        desc="Pocket hole screws specified correctly for 3/4 in stock.",
        parent=hardware_node,
        critical=True
    )
    screw_len_leaf = evaluator.add_leaf(
        id="Screw_Length",
        desc="Pocket hole screws specified as 1-1/4 inch length for 3/4 inch stock.",
        parent=phs_node,
        critical=True
    )
    claim_screw_length = "The plan specifies pocket hole screws of 1-1/4 inch length for 3/4 inch stock."
    await evaluator.verify(
        claim=claim_screw_length,
        node=screw_len_leaf,
        additional_instruction="Confirm the correct pocket screw length is stated for 3/4 in (19 mm) material."
    )

    glue_node = evaluator.add_parallel(
        id="Wood_Glue",
        desc="Wood glue meets constraint.",
        parent=hardware_node,
        critical=True
    )
    glue_type_leaf = evaluator.add_leaf(
        id="Glue_Type_PVA",
        desc="Wood glue specified as PVA type.",
        parent=glue_node,
        critical=True
    )
    claim_glue_type = "The plan specifies a PVA (polyvinyl acetate) wood glue."
    await evaluator.verify(
        claim=claim_glue_type,
        node=glue_type_leaf,
        additional_instruction="Look for mention of PVA glue."
    )

    glue_brand_leaf = evaluator.add_leaf(
        id="Glue_Brand_Allowed",
        desc="Specifies Titebond Original, Titebond II, or Titebond III (allowed options).",
        parent=glue_node,
        critical=True
    )
    claim_glue_brand = "The plan specifies an allowed glue brand: Titebond Original, Titebond II, or Titebond III."
    await evaluator.verify(
        claim=claim_glue_brand,
        node=glue_brand_leaf,
        additional_instruction="Check the answer includes one of the listed Titebond products."
    )

    shelf_pins_leaf = evaluator.add_leaf(
        id="Shelf_Support_Pins",
        desc="Shelf support pins included.",
        parent=hardware_node,
        critical=True
    )
    claim_shelf_pins = "The plan includes adjustable shelf support pins."
    await evaluator.verify(
        claim=claim_shelf_pins,
        node=shelf_pins_leaf,
        additional_instruction="Confirm inclusion of shelf pins."
    )

    back_fasteners_leaf = evaluator.add_leaf(
        id="Back_Panel_Fasteners",
        desc="Fasteners for the back panel are listed.",
        parent=hardware_node,
        critical=True
    )
    claim_back_fasteners = "The plan lists fasteners for the back panel (e.g., brad nails, screws, or staples)."
    await evaluator.verify(
        claim=claim_back_fasteners,
        node=back_fasteners_leaf,
        additional_instruction="Verify that specific fastener types for the back panel are included."
    )

    hw_urls_leaf = evaluator.add_leaf(
        id="Hardware_Reference_URLs",
        desc="Provides URL reference(s) for hardware specifications.",
        parent=hardware_node,
        critical=True
    )
    hw_urls = h.hardware_urls if h else []
    claim_hw_urls = "These URLs reference hardware specifications relevant to this plan (pocket hole screws, shelf pins, glue, or back fasteners)."
    await evaluator.verify(
        claim=claim_hw_urls,
        node=hw_urls_leaf,
        sources=hw_urls,
        additional_instruction="Validate that the URLs point to product/spec pages for the hardware items used."
    )

async def verify_ppe(evaluator: Evaluator, parent: VerificationNode, p: PPEModel) -> None:
    ppe_node = evaluator.add_parallel(
        id="PPE_Safety",
        desc="Required PPE with OSHA/industry references.",
        parent=parent,
        critical=True
    )

    eye_leaf = evaluator.add_leaf(
        id="Eye_Protection",
        desc="Includes safety goggles or face shield for eye protection.",
        parent=ppe_node,
        critical=True
    )
    claim_eye = "The plan includes safety goggles or a face shield for eye protection."
    await evaluator.verify(
        claim=claim_eye,
        node=eye_leaf,
        additional_instruction="Confirm explicit mention of eye protection."
    )

    resp_leaf = evaluator.add_leaf(
        id="Respiratory_Protection",
        desc="Includes dust mask or respirator for respiratory protection.",
        parent=ppe_node,
        critical=True
    )
    claim_resp = "The plan includes a dust mask or a respirator for respiratory protection."
    await evaluator.verify(
        claim=claim_resp,
        node=resp_leaf,
        additional_instruction="Confirm explicit mention of respiratory PPE."
    )

    hearing_leaf = evaluator.add_leaf(
        id="Hearing_Protection",
        desc="Includes hearing protection when using power tools.",
        parent=ppe_node,
        critical=True
    )
    claim_hearing = "The plan includes hearing protection for use with power tools."
    await evaluator.verify(
        claim=claim_hearing,
        node=hearing_leaf,
        additional_instruction="Confirm mention of hearing protection."
    )

    ppe_urls_leaf = evaluator.add_leaf(
        id="PPE_Reference_URLs",
        desc="Provides URL documentation referencing OSHA or industry safety standards for PPE.",
        parent=ppe_node,
        critical=True
    )
    p_urls = p.ppe_urls if p else []
    claim_ppe_urls = "These URLs reference OSHA or recognized industry standards regarding PPE relevant to woodworking."
    await evaluator.verify(
        claim=claim_ppe_urls,
        node=ppe_urls_leaf,
        sources=p_urls,
        additional_instruction="Look for OSHA or similar authoritative standards pages about eye, respiratory, and hearing protection."
    )

async def verify_construction(evaluator: Evaluator, parent: VerificationNode, c: ConstructionModel) -> None:
    cons_node = evaluator.add_parallel(
        id="Construction_Process",
        desc="Construction methodology including cutting accuracy, joinery/screw specs, shelf pin drilling specs, and squaring/back installation.",
        parent=parent,
        critical=True
    )

    cutting_leaf = evaluator.add_leaf(
        id="Cutting_Accuracy_Requirements",
        desc="States cutting accuracy/tolerance requirements for cutting components.",
        parent=cons_node,
        critical=True
    )
    claim_cutting = "The plan states cutting accuracy or tolerance requirements for cutting components."
    await evaluator.verify(
        claim=claim_cutting,
        node=cutting_leaf,
        additional_instruction="Typical tolerances are ±1/32 in to ±1/16 in; verify that some tolerance/accuracy guidance is provided."
    )

    joinery_node = evaluator.add_parallel(
        id="Joinery_And_Fastening",
        desc="Joinery method described and uses correct screw specification for 3/4 in stock.",
        parent=cons_node,
        critical=True
    )

    joinery_leaf = evaluator.add_leaf(
        id="Joinery_Method_Described",
        desc="Describes the joinery method for assembling the bookshelf.",
        parent=joinery_node,
        critical=True
    )
    claim_joinery = "The plan describes the joinery method used to assemble the bookshelf (e.g., pocket screws, dowels, biscuits)."
    await evaluator.verify(
        claim=claim_joinery,
        node=joinery_leaf,
        additional_instruction="Confirm the method is explicitly stated."
    )

    screw_spec_leaf = evaluator.add_leaf(
        id="Screw_Spec_Matches_Thickness",
        desc="Specifies/uses 1-1/4 inch pocket hole screws for 3/4 inch stock in the joinery method.",
        parent=joinery_node,
        critical=True
    )
    claim_screw_spec = "The plan specifies using 1-1/4 inch pocket hole screws for 3/4 inch stock in the joinery."
    await evaluator.verify(
        claim=claim_screw_spec,
        node=screw_spec_leaf,
        additional_instruction="Check that screw length is correctly matched to 3/4 in stock in the joinery description."
    )

    shelf_pin_node = evaluator.add_parallel(
        id="Shelf_Pin_Hole_Drilling_Spec",
        desc="Shelf pin hole drilling meets spacing and setback constraints.",
        parent=cons_node,
        critical=True
    )

    spacing_leaf = evaluator.add_leaf(
        id="Vertical_Spacing_32mm",
        desc="Shelf pin holes use standard 32mm vertical spacing.",
        parent=shelf_pin_node,
        critical=True
    )
    claim_spacing = "The plan specifies shelf pin holes with a standard 32 mm vertical spacing."
    await evaluator.verify(
        claim=claim_spacing,
        node=spacing_leaf,
        additional_instruction="Confirm 32 mm spacing is explicitly stated."
    )

    setback_leaf = evaluator.add_leaf(
        id="Front_Setback_37mm",
        desc="Shelf pin holes positioned 37mm from the front edge.",
        parent=shelf_pin_node,
        critical=True
    )
    claim_setback = "The plan positions shelf pin holes 37 mm from the front edge."
    await evaluator.verify(
        claim=claim_setback,
        node=setback_leaf,
        additional_instruction="Confirm 37 mm front setback is explicitly stated."
    )

    square_node = evaluator.add_parallel(
        id="Squaring_And_Back_Installation",
        desc="Assembly and squaring technique, including back panel installed to maintain square.",
        parent=cons_node,
        critical=True
    )

    square_diag_leaf = evaluator.add_leaf(
        id="Square_By_Diagonals",
        desc="Verifies frame is square by measuring equal diagonals.",
        parent=square_node,
        critical=True
    )
    claim_square_diag = "The plan verifies squareness by measuring equal diagonals."
    await evaluator.verify(
        claim=claim_square_diag,
        node=square_diag_leaf,
        additional_instruction="Confirm mention of diagonal measurements for squaring."
    )

    back_square_leaf = evaluator.add_leaf(
        id="Back_Panel_Maintains_Square",
        desc="Back panel installation is described as part of maintaining/locking square assembly.",
        parent=square_node,
        critical=True
    )
    claim_back_square = "The plan describes the back panel installation as maintaining or locking the square of the assembly."
    await evaluator.verify(
        claim=claim_back_square,
        node=back_square_leaf,
        additional_instruction="Look for explicit note that the back helps hold the cabinet square."
    )

async def verify_finishing(evaluator: Evaluator, parent: VerificationNode, f: FinishingModel) -> None:
    finish_node = evaluator.add_parallel(
        id="Finishing_Schedule",
        desc="Sanding progression and polyurethane finishing requirements with product URL.",
        parent=parent,
        critical=True
    )

    sanding_node = evaluator.add_parallel(
        id="Sanding_Progression",
        desc="Sanding schedule satisfies start grit range, max-jump rule, and final grit requirement.",
        parent=finish_node,
        critical=True
    )

    start_leaf = evaluator.add_leaf(
        id="Start_Grit_Range",
        desc="Starting sanding grit is 100–120.",
        parent=sanding_node,
        critical=True
    )
    claim_start = "The plan specifies a starting sanding grit between 100 and 120."
    await evaluator.verify(
        claim=claim_start,
        node=start_leaf,
        additional_instruction="Confirm the starting grit is within 100–120."
    )

    jump_leaf = evaluator.add_leaf(
        id="Max_50Percent_Jump_Rule",
        desc="Sanding progression never jumps more than 50% in grit number.",
        parent=sanding_node,
        critical=True
    )
    claim_jump = "The plan states that the sanding progression never jumps more than 50% in grit number."
    await evaluator.verify(
        claim=claim_jump,
        node=jump_leaf,
        additional_instruction="Look for a rule or guidance limiting grit jumps to ≤50%."
    )

    final_leaf = evaluator.add_leaf(
        id="Final_220_Grit",
        desc="Final sanding before finish is 220 grit.",
        parent=sanding_node,
        critical=True
    )
    claim_final = "The plan specifies 220 grit as the final sanding before applying the finish."
    await evaluator.verify(
        claim=claim_final,
        node=final_leaf,
        additional_instruction="Confirm final sanding grit is 220."
    )

    poly_node = evaluator.add_parallel(
        id="Polyurethane_Finish",
        desc="Polyurethane finish plan satisfies coats, inter-coat sanding, and drying time constraints.",
        parent=finish_node,
        critical=True
    )

    coats_leaf = evaluator.add_leaf(
        id="Coat_Count",
        desc="Specifies 2–3 coats of polyurethane.",
        parent=poly_node,
        critical=True
    )
    claim_coats = "The plan specifies applying 2–3 coats of polyurethane."
    await evaluator.verify(
        claim=claim_coats,
        node=coats_leaf,
        additional_instruction="Confirm the plan calls for 2–3 coats."
    )

    dry_leaf = evaluator.add_leaf(
        id="Dry_Time_Between_Coats",
        desc="Specifies 24 hours drying time between coats.",
        parent=poly_node,
        critical=True
    )
    claim_dry = "The plan specifies 24 hours drying time between coats of polyurethane."
    await evaluator.verify(
        claim=claim_dry,
        node=dry_leaf,
        additional_instruction="Confirm the drying interval is 24 hours."
    )

    inter_sand_leaf = evaluator.add_leaf(
        id="Intercoat_Sanding",
        desc="Specifies sanding with 220 grit after first coat and 320 grit for subsequent coats.",
        parent=poly_node,
        critical=True
    )
    claim_inter_sand = "The plan specifies sanding with 220 grit after the first coat and 320 grit for subsequent coats."
    await evaluator.verify(
        claim=claim_inter_sand,
        node=inter_sand_leaf,
        additional_instruction="Verify the specified intercoat sanding grits are 220 then 320."
    )

    finish_url_leaf = evaluator.add_leaf(
        id="Finish_Product_URL",
        desc="Provides a product reference URL for the recommended finish.",
        parent=finish_node,
        critical=True
    )
    finish_url = f.finish_product_url if f else None
    claim_finish_url = "This URL is a product reference page for the recommended polyurethane finish."
    await evaluator.verify(
        claim=claim_finish_url,
        node=finish_url_leaf,
        sources=finish_url,
        additional_instruction="Verify the URL points to a polyurethane finish product page."
    )

async def verify_installation(evaluator: Evaluator, parent: VerificationNode, i: InstallationModel) -> None:
    install_node = evaluator.add_parallel(
        id="Installation_And_AntiTip",
        desc="Leveling and anti-tip anchoring requirements with guideline URL.",
        parent=parent,
        critical=True
    )

    level_leaf = evaluator.add_leaf(
        id="Leveling",
        desc="Bookshelf leveling procedure uses a level tool (required).",
        parent=install_node,
        critical=True
    )
    claim_level = "The plan specifies using a level tool for leveling the bookshelf."
    await evaluator.verify(
        claim=claim_level,
        node=level_leaf,
        additional_instruction="Confirm the use of a level tool is mentioned."
    )

    height_leaf = evaluator.add_leaf(
        id="AntiTip_Required_By_Height",
        desc="States anti-tip hardware is required for bookcases over 30 inches tall and applies to this 72-inch bookshelf.",
        parent=install_node,
        critical=True
    )
    claim_height = "The plan states that anti-tip hardware is required for bookcases over 30 inches tall and confirms it applies to the 72-inch bookshelf."
    await evaluator.verify(
        claim=claim_height,
        node=height_leaf,
        additional_instruction="Check for explicit height threshold (>30 in) and application to the 72 in case."
    )

    anchors_leaf = evaluator.add_leaf(
        id="Minimum_Anchor_Points",
        desc="Uses at least two anchor points.",
        parent=install_node,
        critical=True
    )
    claim_anchors = "The plan specifies using at least two anchor points."
    await evaluator.verify(
        claim=claim_anchors,
        node=anchors_leaf,
        additional_instruction="Confirm ≥2 anchors are specified."
    )

    studs_leaf = evaluator.add_leaf(
        id="Attach_To_Studs",
        desc="Anchors attach to wall studs (not just drywall).",
        parent=install_node,
        critical=True
    )
    claim_studs = "The plan specifies attaching the anchors to wall studs, not just drywall."
    await evaluator.verify(
        claim=claim_studs,
        node=studs_leaf,
        additional_instruction="Confirm studs are required for anchor attachment."
    )

    rating_leaf = evaluator.add_leaf(
        id="Anchor_Weight_Rating",
        desc="Furniture anchor straps rated for minimum 50 lbs per anchor.",
        parent=install_node,
        critical=True
    )
    claim_rating = "The plan specifies furniture anchor straps rated for a minimum of 50 pounds per anchor."
    await evaluator.verify(
        claim=claim_rating,
        node=rating_leaf,
        additional_instruction="Verify a minimum 50 lb per anchor rating is stated."
    )

    method_leaf = evaluator.add_leaf(
        id="AntiTip_Attachment_Method",
        desc="Describes the attachment method for anti-tip hardware (how it connects to the bookshelf and wall).",
        parent=install_node,
        critical=True
    )
    claim_method = "The plan describes the attachment method for anti-tip hardware, including how it connects to the bookshelf and the wall."
    await evaluator.verify(
        claim=claim_method,
        node=method_leaf,
        additional_instruction="Confirm a clear description of strap/bracket attachment between wall and bookshelf."
    )

    guideline_leaf = evaluator.add_leaf(
        id="AntiTip_Guideline_URL",
        desc="Provides a URL reference for anti-tip safety guidelines (CPSC or equivalent).",
        parent=install_node,
        critical=True
    )
    guideline_url = i.anti_tip_guideline_url if i else None
    claim_guideline = "This URL references CPSC or equivalent anti-tip safety guidelines for anchoring furniture."
    await evaluator.verify(
        claim=claim_guideline,
        node=guideline_leaf,
        sources=guideline_url,
        additional_instruction="Prefer CPSC 'Anchor It!' program or equivalent authoritative guideline page."
    )

# -----------------------------
# Main Evaluation Entry Point
# -----------------------------

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

    # Extract all sections in parallel
    materials_task = evaluator.extract(
        prompt=prompt_extract_materials(),
        template_class=MaterialsModel,
        extraction_name="materials"
    )
    hardware_task = evaluator.extract(
        prompt=prompt_extract_hardware(),
        template_class=HardwareModel,
        extraction_name="hardware"
    )
    ppe_task = evaluator.extract(
        prompt=prompt_extract_ppe(),
        template_class=PPEModel,
        extraction_name="ppe"
    )
    construction_task = evaluator.extract(
        prompt=prompt_extract_construction(),
        template_class=ConstructionModel,
        extraction_name="construction"
    )
    finishing_task = evaluator.extract(
        prompt=prompt_extract_finishing(),
        template_class=FinishingModel,
        extraction_name="finishing"
    )
    installation_task = evaluator.extract(
        prompt=prompt_extract_installation(),
        template_class=InstallationModel,
        extraction_name="installation"
    )

    materials, hardware, ppe, construction, finishing, installation = await asyncio.gather(
        materials_task, hardware_task, ppe_task, construction_task, finishing_task, installation_task
    )

    # Build the project node (critical root of rubric)
    project_node = evaluator.add_parallel(
        id="Custom_Bookshelf_Project",
        desc="Build plan for a freestanding 6 ft (72 in) tall bookshelf with 5 adjustable shelves, meeting the provided constraints and including required references.",
        parent=root,
        critical=True
    )

    # Verify each rubric section
    await verify_materials(evaluator, project_node, materials)
    await verify_hardware(evaluator, project_node, hardware)
    await verify_ppe(evaluator, project_node, ppe)
    await verify_construction(evaluator, project_node, construction)
    await verify_finishing(evaluator, project_node, finishing)
    await verify_installation(evaluator, project_node, installation)

    return evaluator.get_summary()