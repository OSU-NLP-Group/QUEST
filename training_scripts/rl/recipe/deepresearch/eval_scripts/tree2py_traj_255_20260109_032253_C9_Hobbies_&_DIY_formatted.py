import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "outdoor_furniture_plans"
TASK_DESCRIPTION = (
    "I am planning to start woodworking as a beginner and want to build outdoor furniture for my patio. "
    "I need to find four different free woodworking plans that will help me create a complete outdoor furniture set. "
    "Please identify four outdoor furniture plans that meet all of the following requirements:\n\n"
    "General Requirements (apply to all four plans):\n"
    "1. Each plan must be freely available for download without requiring payment or subscription\n"
    "2. The four plans must be for four different types of outdoor furniture: one chair, one bench, one table, and one sofa or sectional\n"
    "3. Each plan must be from a recognized woodworking plan website or established woodworking resource\n\n"
    "Wood and Materials Requirements (for each plan):\n"
    "4. Each plan must explicitly specify using weather-resistant wood species suitable for outdoor use (such as cedar, redwood, pressure-treated pine, white oak, or cypress)\n"
    "5. Each plan must include a complete cut list that provides specific dimensions (length, width, thickness) and quantities for every piece to be cut\n"
    "6. Each plan must include a materials list or shopping list that specifies the lumber dimensions and quantities needed to purchase\n\n"
    "Construction Requirements (for each plan):\n"
    "7. Each plan must be explicitly labeled or described as suitable for beginners or rated as \"easy\" difficulty level\n"
    "8. Each plan must clearly specify the joinery method or fastening technique to be used in construction\n"
    "9. Each plan must provide complete step-by-step building instructions (not just design sketches or diagrams)\n\n"
    "Finishing Requirements (for each plan):\n"
    "10. Each plan must include recommendations or guidance for exterior wood finish, sealer, or protective coating appropriate for outdoor weather exposure\n"
    "11. Each plan must specify the overall finished dimensions of the completed furniture piece\n\n"
    "For each of the four plans, please provide:\n"
    "- The type of furniture (chair, bench, table, or sofa/sectional)\n"
    "- A direct link to the plan page\n"
    "- The wood species specified in the plan\n"
    "- The joinery method specified\n"
    "- The difficulty level stated\n"
    "- A brief confirmation that the plan includes a complete cut list, materials list, step-by-step instructions, finish guidance, and overall dimensions"
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class PlanItem(BaseModel):
    furniture_type: Optional[str] = None
    plan_url: Optional[str] = None
    wood_species: Optional[str] = None
    joinery_method: Optional[str] = None
    difficulty: Optional[str] = None

    # "Brief confirmation" text if the answer explicitly confirms the five checklist items
    confirmation_text: Optional[str] = None

    # Optional granular confirmations if the answer lists them item-by-item
    confirm_cut_list: Optional[bool] = None
    confirm_materials_list: Optional[bool] = None
    confirm_step_by_step: Optional[bool] = None
    confirm_finish_guidance: Optional[bool] = None
    confirm_finished_dimensions: Optional[bool] = None


class PlansExtraction(BaseModel):
    plans: List[PlanItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plans() -> str:
    return """
Extract up to four outdoor furniture plan entries mentioned in the answer. If more than four are present, keep only the first four in the order they appear. For each plan, extract exactly the following fields:

- furniture_type: One of the following categories as stated by the answer text: "chair", "bench", "table", or "sofa/sectional" (keep the exact wording the answer uses; do not invent).
- plan_url: The direct URL to the specific plan page. If the answer provides a markdown link, extract the URL. If the answer references a site without a direct plan page URL, set to null.
- wood_species: The wood species named in the answer for that plan. If multiple are given, extract them as a single string exactly as written.
- joinery_method: The joinery or fastening method named in the answer (e.g., pocket screws, dowels, mortise-and-tenon, exterior screws). If not stated, set to null.
- difficulty: The difficulty label stated in the answer (e.g., beginner, easy). If not stated, set to null.

The answer also asks the responder to include a brief confirmation that each plan includes five items: a complete cut list, a materials/shopping list, step-by-step instructions, finish guidance for outdoor use, and overall finished dimensions. Capture this in two ways:
- confirmation_text: If the answer includes a brief, explicit confirmation statement that the plan contains these five items, extract that sentence or phrase. Otherwise set to null.
- confirm_cut_list: true if the answer explicitly claims the plan includes a complete cut list; false if it explicitly claims it does not; null if the answer does not say.
- confirm_materials_list: true if the answer explicitly claims the plan includes a materials/shopping list; false if it explicitly claims it does not; null if the answer does not say.
- confirm_step_by_step: true if the answer explicitly claims the plan includes step-by-step instructions; false if it explicitly claims it does not; null if the answer does not say.
- confirm_finish_guidance: true if the answer explicitly claims the plan includes exterior finish/sealer guidance; false if it explicitly claims it does not; null if the answer does not say.
- confirm_finished_dimensions: true if the answer explicitly claims the plan specifies overall finished dimensions; false if it explicitly claims it does not; null if the answer does not say.

GENERAL URL EXTRACTION RULES:
- Extract only URLs explicitly present in the answer text. Do not infer or invent URLs.
- Extract complete URLs. If a URL is missing the protocol, prepend "http://".
- Prefer the most specific, direct plan page URL if multiple are present.

Return a JSON object with a single key "plans" that is an array of plan objects of the schema above. If a field is missing from the answer, set it to null (or an empty string only when absolutely necessary).
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_url_for_distinctness(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        parts = urlsplit(url.strip())
        scheme = parts.scheme or "http"
        netloc = parts.netloc.lower()
        path = parts.path.rstrip("/")
        # Ignore query and fragment for distinctness check
        if not netloc:
            return None
        return f"{scheme}://{netloc}{path}"
    except Exception:
        return None


def classify_furniture_type(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    t = raw.strip().lower()

    # Normalize common variants and synonyms
    if any(k in t for k in ["sofa", "sectional", "couch", "loveseat", "settee"]):
        return "sofa/sectional"
    if "chair" in t:
        # catch "armchair", "adirondack chair", "lounge chair", etc.
        return "chair"
    if "bench" in t:
        return "bench"
    if "table" in t:
        # catch "coffee table", "side table", "dining table", etc.
        return "table"
    return None


def brief_confirmation_provided(plan: PlanItem) -> bool:
    text_ok = bool(plan.confirmation_text and plan.confirmation_text.strip())
    flags_ok = all([
        plan.confirm_cut_list is True,
        plan.confirm_materials_list is True,
        plan.confirm_step_by_step is True,
        plan.confirm_finish_guidance is True,
        plan.confirm_finished_dimensions is True
    ])
    # Accept either an explicit text confirmation or a full set of True flags
    return text_ok or flags_ok


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_single_plan(evaluator: Evaluator, parent_node, plan: PlanItem, index: int) -> None:
    """
    Build the verification subtree for a single plan and perform all checks.
    index is 0-based for plan numbering.
    """
    plan_no = index + 1

    # Parent node for the plan (non-critical to allow partial credit across different plans)
    plan_node = evaluator.add_parallel(
        id=f"Plan_{plan_no}",
        desc=f"Evaluation of plan #{plan_no}.",
        parent=parent_node,
        critical=False
    )

    # 1) Response fields presence (critical group)
    fields_node = evaluator.add_parallel(
        id=f"Plan_{plan_no}_Response_Fields",
        desc=f"The response includes the requested fields for plan #{plan_no}.",
        parent=plan_node,
        critical=True
    )

    # Leaf checks for presence (based on extracted answer content)
    evaluator.add_custom_node(
        result=bool(plan.furniture_type and plan.furniture_type.strip()),
        id=f"Plan_{plan_no}_Type_Provided",
        desc=f"Response states the furniture type (chair/bench/table/sofa-sectional) for plan #{plan_no}.",
        parent=fields_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(plan.plan_url and plan.plan_url.strip()),
        id=f"Plan_{plan_no}_Link_Provided",
        desc=f"Response provides a direct URL link to the plan page for plan #{plan_no}.",
        parent=fields_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(plan.wood_species and plan.wood_species.strip()),
        id=f"Plan_{plan_no}_Wood_Species_Provided",
        desc=f"Response states the wood species specified in the plan for plan #{plan_no}.",
        parent=fields_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(plan.joinery_method and plan.joinery_method.strip()),
        id=f"Plan_{plan_no}_Joinery_Provided",
        desc=f"Response states the joinery/fastening method specified in the plan for plan #{plan_no}.",
        parent=fields_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(plan.difficulty and plan.difficulty.strip()),
        id=f"Plan_{plan_no}_Difficulty_Provided",
        desc=f"Response states the difficulty level stated in the plan for plan #{plan_no}.",
        parent=fields_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=brief_confirmation_provided(plan),
        id=f"Plan_{plan_no}_Brief_Confirmation_Provided",
        desc="Response includes a brief confirmation that the plan contains: cut list, materials list, step-by-step instructions, finish guidance, and overall finished dimensions.",
        parent=fields_node,
        critical=True
    )

    # 2) Constraint compliance with the plan page (critical group)
    constraints_node = evaluator.add_parallel(
        id=f"Plan_{plan_no}_Constraint_Compliance",
        desc=f"Plan #{plan_no} meets all per-plan constraints from the prompt.",
        parent=plan_node,
        critical=True
    )

    # Create verification leaves for each constraint
    # We will batch-verify them against the single plan URL.
    url = plan.plan_url

    free_node = evaluator.add_leaf(
        id=f"Plan_{plan_no}_Free_Availability",
        desc="Plan is freely available without requiring payment or subscription.",
        parent=constraints_node,
        critical=True
    )
    recognized_node = evaluator.add_leaf(
        id=f"Plan_{plan_no}_Recognized_Source",
        desc="Plan is from a recognized woodworking plan website or established woodworking resource.",
        parent=constraints_node,
        critical=True
    )
    wood_node = evaluator.add_leaf(
        id=f"Plan_{plan_no}_Weather_Resistant_Wood",
        desc="Plan explicitly specifies a weather-resistant wood species suitable for outdoor use (cedar, redwood, pressure-treated pine, white oak, or cypress).",
        parent=constraints_node,
        critical=True
    )
    cutlist_node = evaluator.add_leaf(
        id=f"Plan_{plan_no}_Cut_List_Complete",
        desc="Plan includes a complete cut list with length/width/thickness dimensions and quantities for every piece to be cut.",
        parent=constraints_node,
        critical=True
    )
    materials_node = evaluator.add_leaf(
        id=f"Plan_{plan_no}_Materials_List_Complete",
        desc="Plan includes a materials/shopping list specifying lumber dimensions and quantities needed to purchase.",
        parent=constraints_node,
        critical=True
    )
    beginner_node = evaluator.add_leaf(
        id=f"Plan_{plan_no}_Beginner_Or_Easy",
        desc="Plan is explicitly labeled/described as beginner-suitable or rated easy difficulty.",
        parent=constraints_node,
        critical=True
    )
    joinery_node = evaluator.add_leaf(
        id=f"Plan_{plan_no}_Joinery_Specified",
        desc="Plan clearly specifies the joinery method or fastening technique used in construction.",
        parent=constraints_node,
        critical=True
    )
    steps_node = evaluator.add_leaf(
        id=f"Plan_{plan_no}_Step_By_Step_Instructions",
        desc="Plan provides complete step-by-step building instructions (not just sketches/diagrams).",
        parent=constraints_node,
        critical=True
    )
    finish_node = evaluator.add_leaf(
        id=f"Plan_{plan_no}_Finish_Guidance",
        desc="Plan includes recommendations/guidance for an exterior finish/sealer/protective coating appropriate for outdoor exposure.",
        parent=constraints_node,
        critical=True
    )
    dims_node = evaluator.add_leaf(
        id=f"Plan_{plan_no}_Finished_Dimensions",
        desc="Plan specifies the overall finished dimensions of the completed furniture piece.",
        parent=constraints_node,
        critical=True
    )

    claims_and_sources: List = []

    # Build claims with additional instructions
    claims_and_sources.append((
        "This plan page is freely viewable and provides access to the full plan content without requiring payment or a subscription.",
        url,
        free_node,
        "Judge based on the page content and any visible prompts. If the page shows the full instructions (or downloadable plan) without paywall, login, or subscription requirement, consider it free."
    ))
    claims_and_sources.append((
        "This URL is from a recognized woodworking plan website or an established woodworking resource that publishes woodworking plans.",
        url,
        recognized_node,
        "Assess whether the site is known or clearly presents itself as a reputable woodworking/DIY plan provider (e.g., Family Handyman, Ana White, BuildSomething/Kreg, Popular Woodworking, Fine Woodworking, Wood Magazine, The Spruce Crafts, Instructables with detailed plans, etc.). Use the page branding and context."
    ))
    claims_and_sources.append((
        "This plan explicitly specifies at least one weather-resistant wood species suitable for outdoor use, such as cedar, redwood, pressure-treated pine, white oak, or cypress.",
        url,
        wood_node,
        "Search for mentions of cedar, redwood, pressure-treated/preservative-treated lumber, white oak, or cypress. Equivalent clearly outdoor-suitable species (e.g., teak) are acceptable only if explicitly stated as suitable for outdoor exposure."
    ))
    claims_and_sources.append((
        "This plan includes a complete cut list (also called cutting list or parts list) that provides dimensions (length, width, thickness) and quantities for every piece.",
        url,
        cutlist_node,
        "Look for a section titled 'cut list', 'cutting list', 'parts list', or a table enumerating each component with specific dimensions and quantities."
    ))
    claims_and_sources.append((
        "This plan provides a materials list or shopping list that specifies the lumber dimensions and quantities needed to purchase.",
        url,
        materials_node,
        "Look for a 'materials list', 'shopping list', or similar section that details lumber sizes (e.g., 2x4, 1x6) and quantities to buy."
    ))
    claims_and_sources.append((
        "This plan is explicitly labeled or described as beginner-friendly or easy difficulty.",
        url,
        beginner_node,
        "Look for words like 'beginner', 'beginner-friendly', 'easy', 'skill level: easy', etc., near the title or in the plan description."
    ))
    claims_and_sources.append((
        "This plan clearly specifies the joinery or fastening methods to use (e.g., pocket holes, dowels, screws, mortise-and-tenon, lap joints, etc.).",
        url,
        joinery_node,
        "Look for explicit references to fasteners or joinery techniques (pocket hole screws/Kreg, exterior screws, dowels, half-lap, mortise and tenon, biscuit, etc.)."
    ))
    claims_and_sources.append((
        "This plan provides complete step-by-step building instructions, not just diagrams or sketches.",
        url,
        steps_node,
        "Verify that the page contains numbered steps or clearly sequenced procedural instructions with text, photos, or diagrams that guide the build from start to finish."
    ))
    claims_and_sources.append((
        "This plan includes recommendations or guidance for an exterior finish, sealer, or protective coating appropriate for outdoor exposure.",
        url,
        finish_node,
        "Search for 'finish', 'sealer', 'stain', 'paint', 'exterior polyurethane', 'spar varnish', 'deck sealer', etc., and explicit guidance for outdoor weather protection."
    ))
    claims_and_sources.append((
        "This plan specifies the overall finished dimensions of the completed furniture piece.",
        url,
        dims_node,
        "Look for a line listing overall width, depth, height (or similar) of the completed furniture. Accept a drawing or table that explicitly shows 'overall dimensions'."
    ))

    # Execute verification; this respects auto-preconditions (e.g., link provided).
    await evaluator.batch_verify(claims_and_sources)


# --------------------------------------------------------------------------- #
# Global checks                                                               #
# --------------------------------------------------------------------------- #
def build_global_checks(evaluator: Evaluator, parent_node, plans: List[PlanItem]) -> None:
    global_node = evaluator.add_parallel(
        id="Global_Requirements",
        desc="Requirements that apply to the full set of four plans.",
        parent=parent_node,
        critical=True
    )

    # Exactly four distinct plan URLs (non-null, distinct after normalization)
    normalized_urls = []
    for p in plans:
        normalized = normalize_url_for_distinctness(p.plan_url)
        if normalized:
            normalized_urls.append(normalized)

    exactly_four = (len(plans) == 4) and (len(normalized_urls) == 4) and (len(set(normalized_urls)) == 4)

    evaluator.add_custom_node(
        result=exactly_four,
        id="Exactly_Four_Plans_Provided",
        desc="Response provides exactly four distinct plans (four distinct plan pages/URLs).",
        parent=global_node,
        critical=True
    )

    # Furniture type coverage: exactly one of each required category across the four plans
    classified = [classify_furniture_type(p.furniture_type) for p in plans]
    required = {"chair", "bench", "table", "sofa/sectional"}
    counts = {k: 0 for k in required}
    valid = True
    for c in classified:
        if c is None:
            valid = False
        else:
            if c in counts:
                counts[c] += 1

    coverage_ok = valid and all(counts[k] == 1 for k in required)

    evaluator.add_custom_node(
        result=coverage_ok,
        id="Furniture_Type_Coverage",
        desc="Across the four plans, the furniture types include exactly one chair, one bench, one table, and one sofa/sectional.",
        parent=global_node,
        critical=True
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the 'outdoor_furniture_plans' task.
    """
    evaluator = Evaluator()
    # IMPORTANT: Set root as non-critical to satisfy framework's critical-child consistency constraints.
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

    # Extract structured plans from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_plans(),
        template_class=PlansExtraction,
        extraction_name="plans_extraction"
    )

    # Keep first four; pad if fewer
    plans: List[PlanItem] = list(extraction.plans[:4])
    while len(plans) < 4:
        plans.append(PlanItem())

    # Build top-level task node (non-critical root already created; emulate rubric root as critical by grouping under a node)
    task_node = evaluator.add_parallel(
        id="Outdoor_Furniture_Plans_Task",
        desc="Evaluate whether the response provides four free outdoor furniture plans meeting all stated constraints and includes the requested fields for each plan.",
        parent=root,
        critical=False  # Keep non-critical to avoid forcing all children to be critical by framework rule
    )

    # Global requirements
    build_global_checks(evaluator, task_node, plans)

    # Per-plan verification
    # Plan_1..4 groups with internal critical children
    for idx in range(4):
        await verify_single_plan(evaluator, task_node, plans[idx], idx)

    # Summary
    return evaluator.get_summary()