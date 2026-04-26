import asyncio
import logging
from typing import Optional, Dict, Any

from pydantic import BaseModel

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "bookshelf_plan_requirements_eval"
TASK_DESCRIPTION = """
I'm a beginner woodworker based in Ohio, and I want to build my first bookshelf for my home office. I need to find a free woodworking plan that meets the following requirements:

1. The plan must be completely free to access (no payment or subscription required)
2. It must be explicitly suitable for beginners (labeled as "easy" or "beginner-friendly")
3. The finished bookshelf must be 60 inches (5 feet) tall or shorter to fit comfortably in my room
4. The project must use common dimensional lumber that I can easily find at my local hardware store (such as 2x4s, 2x6s, 1x10s, 1x12s, or standard plywood) rather than exotic or specialty woods
5. The plan must include complete documentation: a materials list, a cut list, and step-by-step building instructions
6. The plan should include visual aids such as diagrams, blueprints, or photographs
7. The plan should come from a reputable, established woodworking website or DIY platform

Please provide a link to one specific bookshelf plan that meets all these requirements, along with a brief description of the plan including its exact dimensions, the primary materials used, and which website hosts the plan.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PlanExtraction(BaseModel):
    """
    Extracted fields from the agent's answer for a single bookshelf plan.
    Note: Only extract information explicitly present in the answer.
    """
    plan_url: Optional[str] = None
    plan_title: Optional[str] = None
    host_website: Optional[str] = None
    exact_dimensions: Optional[str] = None
    primary_materials: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_plan_info() -> str:
    return """
    Extract the details of one specific bookshelf plan mentioned in the answer. If multiple plans are mentioned, extract only the first one.
    You must only extract information explicitly stated in the answer text. Do not infer anything not stated.

    Required fields:
    - plan_url: The direct URL link to the bookshelf plan page. If no URL is provided, return null.
    - plan_title: The title/name of the bookshelf plan as mentioned in the answer. If absent, return null.
    - host_website: The name of the website or platform (brand/site name) that hosts the plan, as explicitly named in the answer. Do not infer from the URL; if not explicitly named, return null.
    - exact_dimensions: The plan's exact dimensions as provided in the answer (for example, height x width x depth or similar text including measurements). If absent, return null.
    - primary_materials: A short text describing the primary materials used (e.g., "2x4s and 3/4\" plywood") as stated in the answer. If absent, return null.

    Return a single JSON object with these fields.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_text(s: Optional[str]) -> bool:
    return bool(s) and bool(str(s).strip())


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: PlanExtraction) -> None:
    """
    Build the verification tree and run checks according to the rubric.
    """
    # Create the top-level critical node to represent the rubric root
    root_req = evaluator.add_parallel(
        id="Bookshelf_Plan_Requirements",
        desc="Evaluates whether the provided woodworking bookshelf plan meets all specified requirements and whether the response includes the required deliverables.",
        parent=evaluator.root,
        critical=True
    )

    # --------------------------------------------------------------------- #
    # Response Deliverables (critical, parallel)                            #
    # --------------------------------------------------------------------- #
    deliverables = evaluator.add_parallel(
        id="Response_Deliverables",
        desc="Checks that the response provides the specific requested deliverables (link and brief description fields).",
        parent=root_req,
        critical=True
    )

    # Provides_Plan_Link (existence check)
    plan_link_node = evaluator.add_custom_node(
        result=_nonempty_text(extracted.plan_url) and str(extracted.plan_url).strip().lower().startswith(("http://", "https://")),
        id="Provides_Plan_Link",
        desc="Response includes a link (URL) to the specific bookshelf plan.",
        parent=deliverables,
        critical=True
    )

    # Provides_Exact_Dimensions (existence check)
    dims_node = evaluator.add_custom_node(
        result=_nonempty_text(extracted.exact_dimensions),
        id="Provides_Exact_Dimensions",
        desc="Response includes the plan's exact dimensions (as requested).",
        parent=deliverables,
        critical=True
    )

    # Provides_Primary_Materials (existence check)
    prim_mat_node = evaluator.add_custom_node(
        result=_nonempty_text(extracted.primary_materials),
        id="Provides_Primary_Materials",
        desc="Response states the primary materials used.",
        parent=deliverables,
        critical=True
    )

    # Identifies_Host_Website (existence check)
    host_site_node = evaluator.add_custom_node(
        result=_nonempty_text(extracted.host_website),
        id="Identifies_Host_Website",
        desc="Response names which website/platform hosts the plan.",
        parent=deliverables,
        critical=True
    )

    # --------------------------------------------------------------------- #
    # Core plan requirement verifications (all critical)                    #
    # Each verification depends on having a plan link (precondition).       #
    # --------------------------------------------------------------------- #
    url_source = extracted.plan_url if _nonempty_text(extracted.plan_url) else None
    prereqs = [plan_link_node]  # Gate URL-based checks on the presence of the plan link

    # Free_Plan_Availability
    free_node = evaluator.add_leaf(
        id="Free_Plan_Availability",
        desc="Plan is available for free viewing/download with no payment or subscription required.",
        parent=root_req,
        critical=True
    )
    await evaluator.verify(
        claim="This plan page is freely accessible without payment or subscription, with the content (materials/cut list and instructions) viewable without paywalls.",
        node=free_node,
        sources=url_source,
        additional_instruction="Confirm that the plan content is accessible without any payment or login requirement.",
        extra_prerequisites=prereqs
    )

    # Beginner_Skill_Level
    beginner_node = evaluator.add_leaf(
        id="Beginner_Skill_Level",
        desc="Plan is explicitly labeled/described as suitable for beginners or rated 'easy'.",
        parent=root_req,
        critical=True
    )
    await evaluator.verify(
        claim="This bookshelf plan is explicitly labeled or described as 'beginner-friendly', 'beginner', or 'easy'.",
        node=beginner_node,
        sources=url_source,
        additional_instruction="Look for a skill rating or wording such as 'easy', 'beginner', 'simple', or similar on the plan page.",
        extra_prerequisites=prereqs
    )

    # Height_Limitation (≤ 60 inches)
    height_node = evaluator.add_leaf(
        id="Height_Limitation",
        desc="Finished bookshelf height is specified and is ≤ 60 inches (5 feet).",
        parent=root_req,
        critical=True
    )
    await evaluator.verify(
        claim="The finished bookshelf height shown in the plan is no more than 60 inches (5 feet).",
        node=height_node,
        sources=url_source,
        additional_instruction="Check the plan's dimensions; accept reasonable unit conversions (e.g., feet/inches) and ensure the overall height is ≤ 60 inches.",
        extra_prerequisites=prereqs
    )

    # Common_Lumber_Materials
    common_lumber_node = evaluator.add_leaf(
        id="Common_Lumber_Materials",
        desc="Plan primarily uses common dimensional lumber/standard plywood (not exotic/specialty woods).",
        parent=root_req,
        critical=True
    )
    await evaluator.verify(
        claim="The plan primarily uses common dimensional lumber sizes (e.g., 2x4s, 2x6s, 1x10s, 1x12s) and/or standard plywood, rather than exotic or specialty woods.",
        node=common_lumber_node,
        sources=url_source,
        additional_instruction="Check the materials list or build steps for typical dimensional lumber or plywood; allow synonyms like '2 x 4', 'two-by-four', etc.",
        extra_prerequisites=prereqs
    )

    # Complete_Documentation
    documentation_node = evaluator.add_leaf(
        id="Complete_Documentation",
        desc="Plan includes a materials list, a cut list, and step-by-step building instructions.",
        parent=root_req,
        critical=True
    )
    await evaluator.verify(
        claim="This plan includes all of: a materials list, a cut list, and step-by-step building instructions.",
        node=documentation_node,
        sources=url_source,
        additional_instruction="Confirm that all three are present (materials list, cut list, step-by-step instructions) on the plan page.",
        extra_prerequisites=prereqs
    )

    # Visual_Aids
    visual_node = evaluator.add_leaf(
        id="Visual_Aids",
        desc="Plan includes diagrams, blueprints, or photographs as visual aids.",
        parent=root_req,
        critical=True
    )
    await evaluator.verify(
        claim="The plan includes visual aids such as diagrams, blueprints, or photographs.",
        node=visual_node,
        sources=url_source,
        additional_instruction="Check for any diagrams, build drawings, step photos, or illustrative images on the plan page.",
        extra_prerequisites=prereqs
    )

    # Reputable_Source
    reputable_node = evaluator.add_leaf(
        id="Reputable_Source",
        desc="Plan is hosted on an established/reputable woodworking or recognized DIY platform.",
        parent=root_req,
        critical=True
    )
    await evaluator.verify(
        claim="This plan is hosted on a reputable, established woodworking or recognized DIY platform.",
        node=reputable_node,
        sources=url_source,
        additional_instruction="Judge reputation based on the brand/domain. Examples include Ana White, Family Handyman, Instructables, Wood Magazine, Fine Woodworking, The Spruce Crafts, Popular Mechanics, BuildSomething (Kreg), Rogue Engineer, DIY Pete, etc.",
        extra_prerequisites=prereqs
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
    Evaluate an answer against the bookshelf plan requirements rubric.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root container; we add our critical root under it
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

    # Extract plan info from the answer
    plan_info = await evaluator.extract(
        prompt=prompt_extract_plan_info(),
        template_class=PlanExtraction,
        extraction_name="plan_extraction"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, plan_info)

    # Return the structured summary
    return evaluator.get_summary()