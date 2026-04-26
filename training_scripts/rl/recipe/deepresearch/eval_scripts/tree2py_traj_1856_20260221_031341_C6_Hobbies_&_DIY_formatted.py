import asyncio
import logging
from typing import Optional, Any, Dict

from pydantic import BaseModel

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "diy_bookshelf_plan"
TASK_DESCRIPTION = """
Find a free DIY bookshelf building plan that meets all of the following requirements:

Dimensional Requirements:
- Depth: between 10-15 inches
- Height: between 60-84 inches
- Width must be clearly specified

Material Requirements:
- Must use cabinet-grade plywood (pine, birch, or oak)
- Must specify 3/4" plywood for structural components (sides, top, shelves)
- Must specify 1/2" plywood for the back panel

Construction Requirements:
- Must use pocket hole joinery as the primary assembly method
- Must include a face frame design to cover plywood edges
- Shelf spacing must be between 8-12 inches
- Must require only basic power tools (drill, circular saw, miter saw, and pocket hole jig) - no specialized equipment like table saws or router tables

Instructions Requirements:
- Must provide a logical assembly sequence: body assembly → shelf installation → face frame installation
- Must include finishing instructions covering: sanding, staining or painting, and protective topcoat application
- Must be freely accessible online without paywall restrictions

Provide the name of the plan, the website/author it comes from, and the complete URL where it can be accessed.
"""

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class PlanExtraction(BaseModel):
    plan_name: Optional[str] = None
    website_or_author: Optional[str] = None
    plan_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return """
    Extract the identifying information for the single DIY bookshelf plan the answer proposes as satisfying all requirements. 
    If multiple plans are mentioned, choose the first one that is clearly identified as meeting the task request.

    Return:
    - plan_name: The name/title of the plan as written in the answer
    - website_or_author: The website domain and/or the author or publisher (e.g., 'Ana White', 'The Handyman's Daughter', 'Family Handyman', 'Kreg Tool', etc.)
    - plan_url: The complete URL where the plan can be accessed (must include http:// or https://)

    If any field is missing in the answer, set it to null.
    Do not invent or infer any information that is not explicitly provided in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_plan(
    evaluator: Evaluator,
    extracted: PlanExtraction,
) -> None:
    # Create a critical top-level node under the root (root is always non-critical)
    plan_eval_node = evaluator.add_parallel(
        id="DIY_Bookshelf_Plan_Evaluation",
        desc="Checks whether a single free DIY bookshelf plan satisfies all stated constraints and the required output fields are provided.",
        parent=evaluator.root,
        critical=True
    )

    # 1) Required Output Fields (critical)
    required_fields_node = evaluator.add_parallel(
        id="Required_Output_Fields",
        desc="Answer provides the required identifying information for the plan.",
        parent=plan_eval_node,
        critical=True
    )

    # Existence checks (critical custom nodes)
    evaluator.add_custom_node(
        result=bool(extracted.plan_name and extracted.plan_name.strip()),
        id="Plan_Name_Provided",
        desc="Provides the name/title of the plan.",
        parent=required_fields_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.website_or_author and extracted.website_or_author.strip()),
        id="Website_or_Author_Provided",
        desc="Provides the website and/or author the plan comes from.",
        parent=required_fields_node,
        critical=True
    )
    complete_url_ok = bool(extracted.plan_url and extracted.plan_url.strip() and ("http://" in extracted.plan_url or "https://" in extracted.plan_url))
    evaluator.add_custom_node(
        result=complete_url_ok,
        id="Complete_URL_Provided",
        desc="Provides the complete URL where the plan can be accessed.",
        parent=required_fields_node,
        critical=True
    )

    plan_url = extracted.plan_url if complete_url_ok else None

    # 2) Free Accessibility (critical leaf, verified by URL)
    free_access_node = evaluator.add_leaf(
        id="Free_Accessibility",
        desc="Plan is freely accessible online without paywall restrictions.",
        parent=plan_eval_node,
        critical=True
    )
    await evaluator.verify(
        claim="This plan webpage is freely accessible without a paywall or login requirement.",
        node=free_access_node,
        sources=plan_url,
        additional_instruction="If the content is behind a paywall, a subscription, or requires login to see full instructions/materials, mark as not supported."
    )

    # 3) Dimensional Requirements (critical)
    dim_node = evaluator.add_parallel(
        id="Dimensional_Requirements",
        desc="Bookshelf dimensions satisfy the stated constraints.",
        parent=plan_eval_node,
        critical=True
    )

    depth_node = evaluator.add_leaf(
        id="Depth_Between_10_and_15_Inches",
        desc="Plan specifies depth and it is between 10–15 inches.",
        parent=dim_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies an overall depth for the bookshelf, and that depth is between 10 and 15 inches (inclusive).",
        node=depth_node,
        sources=plan_url,
        additional_instruction="Look for 'depth' or overall cabinet depth. If metric is used, convert approximately; 254–381 mm is acceptable. If depth is a range, ensure it falls within 10–15 in."
    )

    height_node = evaluator.add_leaf(
        id="Height_Between_60_and_84_Inches",
        desc="Plan specifies height and it is between 60–84 inches.",
        parent=dim_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies an overall height for the bookshelf, and that height is between 60 and 84 inches (inclusive).",
        node=height_node,
        sources=plan_url,
        additional_instruction="Look for 'height' or overall height. If metric is used, 1524–2134 mm is acceptable. If multiple height options, at least one option must be in range."
    )

    width_node = evaluator.add_leaf(
        id="Width_Clearly_Specified",
        desc="Plan clearly specifies the width.",
        parent=dim_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan clearly states the overall width of the bookshelf (a specific dimension for width).",
        node=width_node,
        sources=plan_url,
        additional_instruction="Width should be explicitly provided as a dimension. Any unambiguous width value is acceptable."
    )

    # 4) Material Requirements (critical)
    mat_node = evaluator.add_parallel(
        id="Material_Requirements",
        desc="Materials and thicknesses match the stated constraints.",
        parent=plan_eval_node,
        critical=True
    )

    cabinet_grade_node = evaluator.add_leaf(
        id="Cabinet_Grade_Plywood_Pine_Birch_or_Oak",
        desc="Plan uses cabinet-grade plywood and the plywood species is pine, birch, or oak.",
        parent=mat_node,
        critical=True
    )
    await evaluator.verify(
        claim="The materials list calls for cabinet-grade plywood and explicitly identifies the plywood as pine, birch, or oak.",
        node=cabinet_grade_node,
        sources=plan_url,
        additional_instruction="Accept 'birch plywood', 'oak plywood', or 'pine plywood' when described as cabinet-grade or equivalent quality plywood."
    )

    structural_three_quarter_node = evaluator.add_leaf(
        id="Structural_Components_Use_ThreeQuarter_Inch_Plywood",
        desc="Plan specifies 3/4 inch plywood for structural components (sides, top, shelves).",
        parent=mat_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies 3/4-inch plywood for the structural components including the sides, top, and shelves.",
        node=structural_three_quarter_node,
        sources=plan_url,
        additional_instruction="3/4 inch may be written as 3/4 in., 0.75 in., or 19 mm. It must explicitly apply to sides/top/shelves."
    )

    back_half_inch_node = evaluator.add_leaf(
        id="Back_Uses_Half_Inch_Plywood",
        desc="Plan specifies 1/2 inch plywood for the back panel.",
        parent=mat_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies a 1/2-inch plywood back panel.",
        node=back_half_inch_node,
        sources=plan_url,
        additional_instruction="1/2 inch may be written as 1/2 in., 0.5 in., or 12 mm. It must explicitly refer to the back panel."
    )

    # 5) Construction Requirements (critical)
    cons_node = evaluator.add_parallel(
        id="Construction_Requirements",
        desc="Construction methods/features match the stated constraints.",
        parent=plan_eval_node,
        critical=True
    )

    pocket_hole_node = evaluator.add_leaf(
        id="Pocket_Hole_Joinery_Primary_Method",
        desc="Plan uses pocket hole joinery as the primary assembly method.",
        parent=cons_node,
        critical=True
    )
    await evaluator.verify(
        claim="Pocket hole joinery is the primary assembly method used in this plan for the case and shelves.",
        node=pocket_hole_node,
        sources=plan_url,
        additional_instruction="Look for repeated use of pocket holes/Kreg jig for major joints. If pocket holes are only occasional/optional while other methods dominate, then it's not primary."
    )

    face_frame_node = evaluator.add_leaf(
        id="Face_Frame_Included",
        desc="Plan includes a face frame design to cover plywood edges.",
        parent=cons_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan includes a face frame to cover plywood edges.",
        node=face_frame_node,
        sources=plan_url,
        additional_instruction="Look for 'face frame' parts list, cut list, or steps installing a face frame/stiles/rails on the front."
    )

    shelf_spacing_node = evaluator.add_leaf(
        id="Shelf_Spacing_Between_8_and_12_Inches",
        desc="Plan specifies shelf spacing and it is between 8–12 inches.",
        parent=cons_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies shelf spacing and the spacing falls between 8 and 12 inches (inclusive).",
        node=shelf_spacing_node,
        sources=plan_url,
        additional_instruction="Accept fixed or adjustable shelf spacing as long as stated spacing is in 8–12 in range. If only hole spacing is given without shelf spacing, do not count."
    )

    # 6) Tool Requirements (critical)
    tool_node = evaluator.add_parallel(
        id="Tool_Requirements",
        desc="Tool requirements satisfy the stated constraints.",
        parent=plan_eval_node,
        critical=True
    )

    only_basic_tools_node = evaluator.add_leaf(
        id="Requires_Only_Basic_Power_Tools_Listed",
        desc="Plan requires only basic power tools: drill, circular saw, miter saw, and pocket hole jig.",
        parent=tool_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan's required tools consist only of a drill/driver, circular saw, miter saw, and a pocket hole jig.",
        node=only_basic_tools_node,
        sources=plan_url,
        additional_instruction="Optional mentions of other tools are acceptable if explicitly optional; the required tools list should not include table saws, router tables, or other specialized equipment."
    )

    no_specialized_tools_node = evaluator.add_leaf(
        id="Does_Not_Require_Specialized_Equipment",
        desc="Plan does not require specialized equipment like table saws or router tables.",
        parent=tool_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan does not require specialized equipment such as a table saw or router table.",
        node=no_specialized_tools_node,
        sources=plan_url,
        additional_instruction="If the required tools list includes a table saw, router table, jointer, planer, CNC, or similar as required, mark as not supported."
    )

    # 7) Instruction Requirements (critical)
    instr_node = evaluator.add_parallel(
        id="Instruction_Requirements",
        desc="Instructions include the required assembly sequence and finishing steps.",
        parent=plan_eval_node,
        critical=True
    )

    assembly_seq_node = evaluator.add_leaf(
        id="Assembly_Sequence_Body_Then_Shelves_Then_Face_Frame",
        desc="Plan provides a logical assembly sequence: body assembly → shelf installation → face frame installation.",
        parent=instr_node,
        critical=True
    )
    await evaluator.verify(
        claim="The instructions present a logical assembly sequence in this order: body/carcass assembly first, then shelf installation, then face frame installation.",
        node=assembly_seq_node,
        sources=plan_url,
        additional_instruction="Allow synonyms like 'case' or 'carcass' for the body. The order should be unambiguous in the steps or narrative."
    )

    finish_sanding_node = evaluator.add_leaf(
        id="Finishing_Instructions_Include_Sanding",
        desc="Plan includes sanding instructions.",
        parent=instr_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan includes sanding instructions as part of the finishing process.",
        node=finish_sanding_node,
        sources=plan_url,
        additional_instruction="Look for mentions like 'sand', 'sanding', 'sand to 180/220 grit', etc."
    )

    finish_stain_paint_node = evaluator.add_leaf(
        id="Finishing_Instructions_Include_Staining_or_Painting",
        desc="Plan includes staining or painting instructions.",
        parent=instr_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan includes instructions to stain or paint the bookshelf (at least one of them).",
        node=finish_stain_paint_node,
        sources=plan_url,
        additional_instruction="Accept either staining or painting; mentions should refer to applying color/finish beyond clear coats."
    )

    finish_topcoat_node = evaluator.add_leaf(
        id="Finishing_Instructions_Include_Protective_Topcoat",
        desc="Plan includes protective topcoat application instructions.",
        parent=instr_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan includes applying a protective topcoat (e.g., polyurethane, polycrylic, varnish) as part of finishing.",
        node=finish_topcoat_node,
        sources=plan_url,
        additional_instruction="Look for clear protective finishes such as polyurethane, polycrylic, lacquer, varnish, or 'clear coat'."
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
        default_model=model
    )

    # Extract the plan identification fields
    extracted_plan = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=PlanExtraction,
        extraction_name="plan_identification"
    )

    # Record a small custom info snapshot
    evaluator.add_custom_info(
        info={
            "extracted_plan_name": extracted_plan.plan_name,
            "extracted_website_or_author": extracted_plan.website_or_author,
            "extracted_plan_url": extracted_plan.plan_url
        },
        info_type="extraction_summary"
    )

    # Build and verify according to rubric
    await build_and_verify_plan(evaluator, extracted_plan)

    # Return final summary
    return evaluator.get_summary()