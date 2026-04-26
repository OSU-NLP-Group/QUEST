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
TASK_ID = "stanley4_project_recommendation"
TASK_DESCRIPTION = (
    "A beginner woodworker has acquired a Stanley #4 bench plane and wants to undertake their first project using traditional hand-cut joinery. "
    "They are looking for a project that: (1) is appropriate for a smoothing plane, (2) uses a joinery method classified as high-strength that doesn't "
    "require metal fasteners, and (3) is a traditional project type where this joinery method is commonly used. Based on woodworking resources like Fine "
    "Woodworking's hand plane guide and The Spruce's wood joinery classification, what type of project should they build and which specific joinery "
    "method should they employ?"
)

# Helpful context for ground truth style info (not used as hard constraints, but recorded)
ALLOWED_HIGH_STRENGTH_EXAMPLES = [
    "mortise-and-tenon",
    "mortise & tenon",
    "through dovetail",
    "half-blind dovetail",
    "box joint",
    "finger joint",
    "biscuit joint"
]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RecommendationExtraction(BaseModel):
    """Structured extraction of the recommended project and joinery with cited sources."""
    project_type: Optional[str] = None
    joinery_method: Optional[str] = None

    # Narrative justifications mentioned in the answer (used by simple verify)
    beginner_justification: Optional[str] = None
    smoothing_plane_suitability: Optional[str] = None
    hand_cut_feasibility: Optional[str] = None
    no_fasteners_statement: Optional[str] = None

    # Source URLs explicitly provided in the answer
    plane_sources: List[str] = Field(default_factory=list)         # e.g., Fine Woodworking hand plane guide
    joinery_sources: List[str] = Field(default_factory=list)       # e.g., The Spruce joinery classification page(s)
    association_sources: List[str] = Field(default_factory=list)   # any sources tying project type to the joint


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_recommendation() -> str:
    return (
        "From the answer, extract the following fields:\n"
        "- project_type: The specific woodworking project type recommended (e.g., keepsake box, picture frame, small table, drawer, etc.).\n"
        "- joinery_method: The explicit joinery method named (e.g., 'through dovetail', 'half-blind dovetail', 'mortise-and-tenon', 'box/finger joint', 'biscuit joint').\n"
        "- beginner_justification: Any text in the answer that justifies the project as beginner-appropriate.\n"
        "- smoothing_plane_suitability: Any text that indicates the project is suitable for finishing with a smoothing plane (like a #4).\n"
        "- hand_cut_feasibility: Any text indicating the joinery can be cut by hand tools (traditional hand-cut joinery).\n"
        "- no_fasteners_statement: Any text indicating the joinery method does not require metal fasteners (screws or nails).\n"
        "- plane_sources: URLs explicitly cited in the answer that discuss hand planes/classify the #4 as a smoothing plane or recommend it for beginners. Prefer Fine Woodworking or similar sources.\n"
        "- joinery_sources: URLs explicitly cited that discuss/confirm the strength classification of the chosen joinery method (prefer The Spruce's wood joinery article(s)).\n"
        "- association_sources: URLs explicitly cited that indicate the chosen project type traditionally uses the chosen joinery method (e.g., dovetails commonly used for boxes/drawers; box/finger joints commonly used for boxes; mortise-and-tenon commonly used for frames/tables/chairs).\n"
        "Rules:\n"
        "1) Return null for any missing field.\n"
        "2) For URL fields, extract only actual URLs shown in the answer (plain or markdown); do not invent URLs.\n"
        "3) Provide full URLs with protocol; ignore malformed URLs.\n"
        "4) Do not mix categories: if a URL is about hand plane classification, put it in plane_sources; if it's about joinery strength or traditional hand-cut feasibility, put it in joinery_sources; if it links project type to the joint, put it in association_sources.\n"
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _uniq_urls(urls: List[str]) -> List[str]:
    """Return unique, cleaned URLs (basic de-dup)."""
    seen = set()
    result = []
    for u in urls:
        if not u:
            continue
        key = u.strip()
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result


def _combine_sources(*lists: List[str]) -> List[str]:
    combined: List[str] = []
    for lst in lists:
        combined.extend(lst or [])
    return _uniq_urls(combined)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_hand_plane_validation(
    evaluator: Evaluator,
    parent_node,
    extracted: RecommendationExtraction
) -> None:
    """Build and run 'Hand_Plane_Constraint_Validation' subtree."""
    hp_node = evaluator.add_parallel(
        id="Hand_Plane_Constraint_Validation",
        desc="Validate that the answer remains consistent with the required hand plane constraints.",
        parent=parent_node,
        critical=True
    )

    # 1) Plane_Is_Stanley_4 - answer consistency check (simple verify)
    leaf_plane_is_stanley4 = evaluator.add_leaf(
        id="Plane_Is_Stanley_4",
        desc="The answer is consistent with using a Stanley #4 bench plane (does not substitute a different plane as the required tool).",
        parent=hp_node,
        critical=True
    )
    claim_plane_is_stanley4 = (
        "The answer remains consistent with using a Stanley #4 bench plane for the project, and does not suggest replacing "
        "it with a different plane as the required tool."
    )
    await evaluator.verify(
        claim=claim_plane_is_stanley4,
        node=leaf_plane_is_stanley4,
        additional_instruction=(
            "Check the answer text for any substitution that contradicts the required use of a Stanley #4 smoothing plane. "
            "Mentioning other planes as optional or supplementary is fine; the primary context must remain Stanley #4."
        )
    )

    # 2) Plane_Classified_As_Smoothing_Plane - verify via plane sources (prefer Fine Woodworking)
    leaf_plane_is_smoothing = evaluator.add_leaf(
        id="Plane_Classified_As_Smoothing_Plane",
        desc="The answer recognizes/uses the fact that a Stanley #4 falls within the smoothing-plane category.",
        parent=hp_node,
        critical=True
    )
    claim_plane_is_smoothing = (
        "According to the provided sources, the #4 bench plane is a smoothing plane."
    )
    await evaluator.verify(
        claim=claim_plane_is_smoothing,
        node=leaf_plane_is_smoothing,
        sources=_uniq_urls(extracted.plane_sources),
        additional_instruction=(
            "Prefer verifying via Fine Woodworking's hand plane guide or equivalent authoritative sources that classify the #4 as a smoothing plane."
        )
    )

    # 3) Plane_Recommended_First_Plane - verify via plane sources
    leaf_plane_first = evaluator.add_leaf(
        id="Plane_Recommended_First_Plane",
        desc="The answer recognizes/uses the fact that a Stanley #4 is recommended as a first plane for beginner woodworkers.",
        parent=hp_node,
        critical=True
    )
    claim_plane_first = (
        "According to the provided sources, a #4 smoothing plane is commonly recommended as a first hand plane for beginner woodworkers."
    )
    await evaluator.verify(
        claim=claim_plane_first,
        node=leaf_plane_first,
        sources=_uniq_urls(extracted.plane_sources),
        additional_instruction=(
            "Confirm that at least one cited source explicitly recommends a #4 smoothing plane as a beginner's first hand plane."
        )
    )


async def build_project_type_validation(
    evaluator: Evaluator,
    parent_node,
    extracted: RecommendationExtraction
) -> None:
    """Build and run 'Project_Type_Validation' subtree."""
    pt_node = evaluator.add_parallel(
        id="Project_Type_Validation",
        desc="Validate that the recommended project type fits the beginner and smoothing-plane constraints.",
        parent=parent_node,
        critical=True
    )

    # 1) Project_Named - existence check
    project_exists = bool(extracted.project_type and extracted.project_type.strip())
    evaluator.add_custom_node(
        result=project_exists,
        id="Project_Named",
        desc="The answer identifies a woodworking project type to build.",
        parent=pt_node,
        critical=True
    )

    # 2) Beginner_Appropriate - simple verify based on the answer's justification
    leaf_beginner = evaluator.add_leaf(
        id="Beginner_Appropriate",
        desc="The answer indicates or justifies that the recommended project is appropriate for a beginner skill level.",
        parent=pt_node,
        critical=True
    )
    claim_beginner = (
        f"The recommended project '{extracted.project_type or ''}' is appropriate for beginner woodworkers, "
        "as indicated or justified in the answer."
    )
    await evaluator.verify(
        claim=claim_beginner,
        node=leaf_beginner,
        additional_instruction=(
            "Use the answer context and its justification to judge beginner appropriateness. "
            "Typical beginner-friendly projects include small boxes, picture frames, stools, and simple casework."
        )
    )

    # 3) Smoothing_Plane_Suitable - simple verify that a smoothing plane is suitable for the project's surface finishing
    leaf_smoothing_ok = evaluator.add_leaf(
        id="Smoothing_Plane_Suitable",
        desc="The answer indicates or justifies that the recommended project is suitable for surface finishing with a smoothing plane.",
        parent=pt_node,
        critical=True
    )
    claim_smoothing_ok = (
        f"The project '{extracted.project_type or ''}' involves surfaces where a #4 smoothing plane is appropriate for final surface finishing."
    )
    await evaluator.verify(
        claim=claim_smoothing_ok,
        node=leaf_smoothing_ok,
        additional_instruction=(
            "Check the answer for statements that a smoothing plane is appropriate for final surface preparation of the project "
            "(e.g., flattening and refining small panels or box sides prior to finish)."
        )
    )


async def build_joinery_method_validation(
    evaluator: Evaluator,
    parent_node,
    extracted: RecommendationExtraction
) -> None:
    """Build and run 'Joinery_Method_Validation' subtree."""
    jm_node = evaluator.add_parallel(
        id="Joinery_Method_Validation",
        desc="Validate that the recommended joinery method meets the high-strength, traditional hand-cut, no-fasteners, and traditional-association constraints.",
        parent=parent_node,
        critical=True
    )

    # 1) Joinery_Method_Named - existence check
    joinery_exists = bool(extracted.joinery_method and extracted.joinery_method.strip())
    evaluator.add_custom_node(
        result=joinery_exists,
        id="Joinery_Method_Named",
        desc="The answer explicitly names a joinery method to use.",
        parent=jm_node,
        critical=True
    )

    # 2) High_Strength_Classification - verify via The Spruce (or cited joinery sources)
    leaf_high_strength = evaluator.add_leaf(
        id="High_Strength_Classification",
        desc="The named joint is classified as high-strength per the provided constraint list (mortise-and-tenon, through dovetail, half-blind dovetail, box/finger joint, or biscuit joint).",
        parent=jm_node,
        critical=True
    )
    claim_high_strength = (
        f"According to the cited sources (e.g., The Spruce wood joinery classification), the '{extracted.joinery_method or ''}' joint is classified as strong/high-strength."
    )
    await evaluator.verify(
        claim=claim_high_strength,
        node=leaf_high_strength,
        sources=_uniq_urls(extracted.joinery_sources),
        additional_instruction=(
            "Confirm that the chosen joint appears on the list of strong/high-strength joinery methods. "
            "Accept mortise-and-tenon, through dovetail, half-blind dovetail, box/finger joint, or biscuit joint as strong/high-strength per the provided classification."
        )
    )

    # 3) Traditional_Hand_Cut_Joinery - verify via sources or answer context
    leaf_handcut = evaluator.add_leaf(
        id="Traditional_Hand_Cut_Joinery",
        desc="The answer presents the joinery method as traditional hand-cut joinery (i.e., feasible to cut/execute with hand tools as described).",
        parent=jm_node,
        critical=True
    )
    claim_handcut = (
        f"The '{extracted.joinery_method or ''}' joint is traditionally hand-cut and can be executed with hand tools (e.g., chisels, saws, marking tools)."
    )
    await evaluator.verify(
        claim=claim_handcut,
        node=leaf_handcut,
        sources=_uniq_urls(extracted.joinery_sources),
        additional_instruction=(
            "Verify that the joint is traditionally cut by hand. Historical and traditional woodworking practices commonly include hand-cut dovetails, "
            "mortise-and-tenon, and box/finger joints. It is acceptable if modern variations use machines, provided the joint is traditionally feasible by hand."
        )
    )

    # 4) No_Metal_Fasteners - verify via sources or answer
    leaf_no_fasteners = evaluator.add_leaf(
        id="No_Metal_Fasteners",
        desc="The recommended joinery method does not require metal fasteners (screws or nails).",
        parent=jm_node,
        critical=True
    )
    claim_no_fasteners = (
        f"The '{extracted.joinery_method or ''}' joint does not require metal fasteners (screws or nails) for its strength."
    )
    await evaluator.verify(
        claim=claim_no_fasteners,
        node=leaf_no_fasteners,
        sources=_uniq_urls(extracted.joinery_sources),
        additional_instruction=(
            "Confirm from the cited sources that the joint relies on wood-to-wood mechanical interlock and/or glue, not on screws or nails, to hold."
        )
    )

    # 5) Traditional_Project_Association - verify via association sources (and joinery sources if helpful)
    leaf_assoc = evaluator.add_leaf(
        id="Traditional_Project_Association",
        desc="The recommended project type is traditionally associated with the chosen joinery method (the joint is commonly used for that project type).",
        parent=jm_node,
        critical=True
    )
    combined_sources = _combine_sources(extracted.association_sources, extracted.joinery_sources)
    claim_assoc = (
        f"The recommended project '{extracted.project_type or ''}' traditionally uses '{extracted.joinery_method or ''}' joints and they are commonly employed for that project type."
    )
    await evaluator.verify(
        claim=claim_assoc,
        node=leaf_assoc,
        sources=combined_sources,
        additional_instruction=(
            "Examples: dovetails for boxes and drawers; half-blind dovetails for drawer fronts; box/finger joints for boxes; mortise-and-tenon for frames, chairs, and tables. "
            "Verify that the cited sources support the association."
        )
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
    Evaluate the agent's recommendation for the Stanley #4 smoothing-plane beginner project and joinery selection.
    """
    # Initialize evaluator (root node is non-critical by framework design)
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_recommendation(),
        template_class=RecommendationExtraction,
        extraction_name="recommendation_extraction",
    )

    # Record ground-truth style info for transparency (not used to judge directly)
    evaluator.add_ground_truth({
        "allowed_high_strength_examples": ALLOWED_HIGH_STRENGTH_EXAMPLES,
        "expect_sources_domains": {
            "plane_sources": ["finewoodworking.com", "popularwoodworking.com", "leevalley.com", "woodmagazine.com"],
            "joinery_sources": ["thespruce.com"],
            "association_sources": ["thespruce.com", "finewoodworking.com", "popularwoodworking.com"]
        }
    }, gt_type="context_reference")

    # Create the critical project validation umbrella node (acts as logical root for rubric)
    project_validation_node = evaluator.add_parallel(
        id="Project_Recommendation_Validation",
        desc=("The answer must recommend a project type and a joinery method satisfying all stated constraints "
              "(Stanley #4 smoothing-plane context; beginner-appropriate and smoothing-plane-suitable project; "
              "high-strength joint per The Spruce list; no metal fasteners; traditional hand-cut joinery; "
              "traditional association between project type and joint)."),
        parent=root,
        critical=True
    )

    # Build subtrees
    await build_hand_plane_validation(evaluator, project_validation_node, extracted)
    await build_project_type_validation(evaluator, project_validation_node, extracted)
    await build_joinery_method_validation(evaluator, project_validation_node, extracted)

    # Return summary with verification tree and scores
    return evaluator.get_summary()