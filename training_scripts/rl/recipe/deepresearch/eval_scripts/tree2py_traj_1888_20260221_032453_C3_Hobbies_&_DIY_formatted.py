import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task identifiers                                                            #
# --------------------------------------------------------------------------- #
TASK_ID = "wood_table_pocket_kreg_beginner"
TASK_DESCRIPTION = (
    "I'm a beginner looking to start my first woodworking furniture project and I want to build either a coffee table "
    "or a side table. I'd like to use pocket hole joinery (Kreg jig) since I've heard it's beginner-friendly. Find a free "
    "woodworking plan online that meets the following requirements:\n\n"
    "1. Must be a beginner-friendly coffee table or side table project\n"
    "2. Must use pocket hole joinery as the primary construction method\n"
    "3. Must specify standard dimensional lumber (like 2x4s, 2x6s, 1x3s, 1x12s, or 3/4\" project panels)\n"
    "4. Must specify the appropriate pocket hole screws for the materials used\n"
    "5. Must include finishing instructions that cover sanding with a grit progression (ending at #150, #180, or #220 grit)\n"
    "6. Must include instructions for applying polyurethane finish (at least 2-3 coats)\n\n"
    "Please provide the project name, the direct URL to the free plan, and confirm that it meets all the above requirements."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class PlanExtraction(BaseModel):
    project_name: Optional[str] = None
    plan_url: Optional[str] = None
    project_type: Optional[str] = None  # e.g., "coffee table", "side table", "end table"
    joinery_description: Optional[str] = None
    materials_list: List[str] = Field(default_factory=list)  # e.g., ["2x4", "1x3", "3/4\" plywood"]
    screws_description: Optional[str] = None  # e.g., "Use 1-1/4\" pocket hole screws ..."
    sanding_instructions: Optional[str] = None  # e.g., "Sand 80→120→220"
    sanding_final_grit: Optional[str] = None  # e.g., "220"
    poly_instructions: Optional[str] = None  # e.g., "Apply 3 coats of polyurethane"
    poly_coats: Optional[str] = None  # e.g., "2", "3 coats"
    beginner_friendly_justification: Optional[str] = None  # e.g., "Beginner/Easy"
    free_plan_indicator: Optional[str] = None  # e.g., "free", "no cost"


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return (
        "Extract the single woodworking project referenced in the answer. Return exactly the following fields:\n"
        "1) project_name: The project name as stated in the answer.\n"
        "2) plan_url: The direct URL to the free plan page on the woodworking website (not a home page or search results).\n"
        "3) project_type: If the answer states it's a coffee table or side table (including synonyms like 'end table' or 'accent table'), extract the term; otherwise null.\n"
        "4) joinery_description: The text in the answer that mentions joinery method (e.g., pocket holes, Kreg jig). If no mention, null.\n"
        "5) materials_list: A list of standard dimensional lumber or panels explicitly mentioned in the answer (examples: '2x4', '2x6', '1x3', '1x12', '3/4\" plywood', '3/4\" project panel').\n"
        "6) screws_description: Any mention of pocket hole screw sizes in the answer (e.g., '1-1/4\" pocket hole screws'). If none, null.\n"
        "7) sanding_instructions: The sanding instructions text if included in the answer; otherwise null.\n"
        "8) sanding_final_grit: If the answer mentions a final sanding grit (e.g., 150, 180, 220), extract just the number as text; otherwise null.\n"
        "9) poly_instructions: The instructions for polyurethane application if included in the answer; otherwise null.\n"
        "10) poly_coats: If the answer states the number of coats (e.g., '2', '3 coats'), extract that text; otherwise null.\n"
        "11) beginner_friendly_justification: Any text in the answer that indicates beginner-friendliness (e.g., 'Beginner', 'Easy', 'Simple'); otherwise null.\n"
        "12) free_plan_indicator: Any explicit mention that the plan is free; otherwise null.\n\n"
        "Do not invent any information. If a field is not explicitly present in the answer, return null (or an empty list for materials_list)."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_project(evaluator: Evaluator, parent_node, plan: PlanExtraction) -> None:
    """
    Build and evaluate the verification tree for a single plan according to the rubric.
    """
    # Parent node: Project identification (sequential, critical)
    project_ident_node = evaluator.add_sequential(
        id="project_identification",
        desc="Identify a qualifying beginner-friendly coffee table or side table project with free downloadable plans and provide the direct URL",
        parent=parent_node,
        critical=True
    )

    # Child: Project URL reference (sequential, critical)
    url_ref_node = evaluator.add_sequential(
        id="project_url_reference",
        desc="Provide the direct URL to the free plan page on the woodworking website",
        parent=project_ident_node,
        critical=True
    )

    # 1) Existence of direct URL (critical existence gate)
    url_present = bool(plan.plan_url and str(plan.plan_url).strip())
    evaluator.add_custom_node(
        result=url_present,
        id="url_provided",
        desc="Direct plan URL is provided in the answer",
        parent=url_ref_node,
        critical=True
    )

    # Prepare safe values
    plan_url = plan.plan_url or ""
    project_type_text = (plan.project_type or "coffee table or side table").lower()

    # 2) Verify the URL is a free plan page (critical)
    url_is_plan_leaf = evaluator.add_leaf(
        id="url_is_free_plan_page",
        desc="URL points to a free woodworking plan page with full instructions",
        parent=url_ref_node,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage provides a free woodworking plan with a materials/cut list and step-by-step build instructions accessible without payment or login.",
        node=url_is_plan_leaf,
        sources=plan_url,
        additional_instruction=(
            "Accept if the page (or linked free PDF on the same site) clearly contains a materials list (or cut list) and step-by-step build instructions. "
            "It should not be a paywalled product, store listing, forum thread, or video-only page."
        )
    )

    # 3) Verify project category is coffee/side table (critical)
    table_type_leaf = evaluator.add_leaf(
        id="is_table_project",
        desc="Project is a coffee table or side table",
        parent=url_ref_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This plan is for a {project_type_text}, i.e., a coffee table or a side table (including synonyms like 'end table' or 'accent table').",
        node=table_type_leaf,
        sources=plan_url,
        additional_instruction=(
            "Look for terms like 'coffee table', 'side table', 'end table', or 'accent table'. "
            "The primary project should clearly be one of these table types."
        )
    )

    # 4) Verify beginner-friendly (critical)
    beginner_leaf = evaluator.add_leaf(
        id="beginner_friendly",
        desc="Project is beginner-friendly (Beginner/Easy level)",
        parent=url_ref_node,
        critical=True
    )
    await evaluator.verify(
        claim="This plan is beginner-friendly or easy, suitable for someone new to woodworking.",
        node=beginner_leaf,
        sources=plan_url,
        additional_instruction=(
            "Accept if the page explicitly indicates skill level as 'Beginner', 'Easy', or uses similar language like 'simple', 'quick', or 'great first project'. "
            "If a formal skill badge is missing, judge based on clear beginner-oriented cues in the instructions."
        )
    )

    # 5) Construction specifications (parallel, critical)
    construction_node = evaluator.add_parallel(
        id="construction_specifications",
        desc="Verify the project specifies pocket hole joinery and appropriate materials",
        parent=url_ref_node,
        critical=True
    )

    # 5.a) Construction method: pocket hole joinery primary (critical leaf)
    joinery_leaf = evaluator.add_leaf(
        id="construction_method",
        desc="Verify the project uses pocket hole joinery (Kreg jig) as the primary assembly method",
        parent=construction_node,
        critical=True
    )

    # 5.b) Materials specification (parallel, critical)
    materials_node = evaluator.add_parallel(
        id="materials_specification",
        desc="Verify the project specifies standard dimensional lumber and appropriate pocket hole screws",
        parent=construction_node,
        critical=True
    )

    lumber_leaf = evaluator.add_leaf(
        id="lumber_specification",
        desc="Project uses standard dimensional lumber (2x4, 2x6, 1x3, 1x12, or 3/4\" panels)",
        parent=materials_node,
        critical=True
    )
    fastener_leaf = evaluator.add_leaf(
        id="fastener_specification",
        desc="Project specifies pocket hole screws with appropriate size (e.g., 1-1/4\" for 3/4\" material)",
        parent=materials_node,
        critical=True
    )

    # 5.c) Finishing specifications (parallel, critical)
    finish_node = evaluator.add_parallel(
        id="finishing_specifications",
        desc="Verify the project includes complete finishing instructions for sanding and polyurethane application",
        parent=construction_node,
        critical=True
    )

    sanding_leaf = evaluator.add_leaf(
        id="sanding_instructions",
        desc="Project includes sanding instructions with a grit progression ending at #150, #180, or #220 grit",
        parent=finish_node,
        critical=True
    )
    poly_leaf = evaluator.add_leaf(
        id="polyurethane_instructions",
        desc="Project includes instructions for applying polyurethane finish with at least 2-3 coats",
        parent=finish_node,
        critical=True
    )

    # Batch verify construction-related leaves (auto preconditions will handle URL existence gating)
    await evaluator.batch_verify([
        (
            "The build primarily uses pocket hole joinery (Kreg jig) for assembling the main parts.",
            plan_url,
            joinery_leaf,
            "Look for repeated use of pocket holes throughout the steps or fastener list. "
            "Minor exceptions (e.g., attaching top with countersunk screws or nails) are acceptable as long as pocket holes are the primary joinery."
        ),
        (
            "The materials list on this page specifies standard dimensional lumber (e.g., 1x2, 1x3, 1x4, 1x12, 2x2, 2x4, 2x6) and/or 3/4-inch plywood/project panels.",
            plan_url,
            lumber_leaf,
            "Any of the listed sizes qualify. Accept common board nomenclature as long as it corresponds to standard thickness/width dimensions or 3/4\" sheet goods."
        ),
        (
            "The plan specifies pocket hole screw sizes appropriate to material thickness (e.g., 1-1/4\" for 3/4\" material, 2-1/2\" for 1-1/2\" lumber).",
            plan_url,
            fastener_leaf,
            "Look for explicit pocket-hole screw length callouts (such as 1-1/4\", 1-1/2\", 2-1/2\"). "
            "Accept mention of coarse vs fine thread as additional info, but some explicit length guidance must be present."
        ),
        (
            "The finishing instructions include sanding with a grit progression that ends at 150, 180, or 220 grit.",
            plan_url,
            sanding_leaf,
            "Accept patterns like '80→120→220', 'sand to 220', 'finish sanding at 180', or 'final sanding 150'. "
            "The key requirement is that the final sanding grit is one of 150, 180, or 220."
        ),
        (
            "The finishing instructions include applying polyurethane with at least two coats.",
            plan_url,
            poly_leaf,
            "Accept any polyurethane variant (oil-based, water-based, wipe-on). "
            "The instruction must indicate at least 2 coats (e.g., '2 coats', '3 thin coats')."
        ),
    ])


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer against the woodworking plan rubric using the Mind2Web2 framework.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured information from the answer
    plan_info = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=PlanExtraction,
        extraction_name="plan_extraction"
    )

    # Add custom info describing the rubric expectations (optional but helpful)
    evaluator.add_custom_info(
        info={
            "required_project_types": ["coffee table", "side table", "end table", "accent table"],
            "primary_joinery": "pocket holes (Kreg jig)",
            "standard_lumber_examples": ["1x2", "1x3", "1x4", "1x6", "1x12", "2x2", "2x4", "2x6", "3/4\" plywood/project panel"],
            "pocket_hole_screw_examples": ["1-1/4\" for 3/4\" material", "2-1/2\" for 1-1/2\" lumber"],
            "allowed_final_sanding_grits": [150, 180, 220],
            "minimum_poly_coats": 2
        },
        info_type="rubric_requirements",
        info_name="rubric_requirements"
    )

    # Build and run verifications
    await verify_project(evaluator, root, plan_info)

    return evaluator.get_summary()