import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "beginner_diy_bookshelf"
TASK_DESCRIPTION = """
I'm a complete beginner looking to build my first DIY bookshelf for storing hardcover books. I need a design that meets the following requirements: (1) Uses standard bookshelf depth dimensions suitable for regular books, (2) Made from wood that's explicitly recommended as beginner-friendly, (3) Can be built using only basic beginner woodworking tools (circular saw or table saw, drill, clamps, tape measure, and square), (4) Uses simple joinery techniques appropriate for beginners, (5) Has shelf spacing that accommodates various book sizes according to standard design practices, and (6) Each shelf can safely support the weight of multiple hardcover books (at least 30 lbs per shelf). Please identify a specific DIY bookshelf design or plan that satisfies all these requirements, and provide the source where this plan can be found with its key specifications.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PlanSource(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    alt_urls: List[str] = Field(default_factory=list)


class PlanSpecs(BaseModel):
    depth_in_numeric: Optional[float] = None
    depth_text: Optional[str] = None

    wood_type: Optional[str] = None
    wood_beginner_friendly_note: Optional[str] = None
    wood_beginner_friendly_explicit: Optional[bool] = None

    tools: List[str] = Field(default_factory=list)
    joinery: Optional[str] = None

    shelf_spacing_min_in: Optional[float] = None
    shelf_spacing_max_in: Optional[float] = None
    shelf_spacing_text: Optional[str] = None

    per_shelf_capacity_lbs_numeric: Optional[float] = None
    per_shelf_capacity_text: Optional[str] = None

    materials_cost_usd_numeric: Optional[float] = None
    materials_cost_text: Optional[str] = None

    completion_time_text: Optional[str] = None
    weekend_claim_explicit: Optional[bool] = None

    cited_urls: List[str] = Field(default_factory=list)


class BookshelfPlanExtraction(BaseModel):
    plan: Optional[PlanSource] = None
    specs: Optional[PlanSpecs] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return """
    Extract the single DIY bookshelf plan/design identified in the answer and its key specs as explicitly stated and attributed in the answer.

    Return a JSON object with two top-level keys: "plan" and "specs".
    - plan:
        - name: The specific name/title of the DIY bookshelf plan/design mentioned (not generic).
        - url: The main source URL for the plan (if provided).
        - alt_urls: Any additional source or citation URLs that the answer ties to this plan (e.g., additional references, calculators cited for load capacity).
    - specs:
        - depth_in_numeric: The shelf/overall depth in inches as a number if provided (e.g., 11.25). If a range is given, choose the primary stated depth.
        - depth_text: The depth text as written in the answer (e.g., "approx. 11.25 in", "1x12 nominal (11.25 in)").
        - wood_type: The wood species/type (e.g., pine, poplar, plywood) that the answer claims is used/recommended in the plan.
        - wood_beginner_friendly_note: The phrase/wording from the answer indicating the wood is beginner-friendly or easy to work.
        - wood_beginner_friendly_explicit: true if the answer explicitly attributes beginner-friendliness to the source/plan; otherwise false or null.
        - tools: List of tools the answer claims are required for the plan.
        - joinery: The joinery method stated (e.g., "butt joints with screws and glue", "dados", "pocket screws").
        - shelf_spacing_min_in: Minimum shelf opening/spacing in inches if a numeric value or adjustable minimum is provided; else null.
        - shelf_spacing_max_in: Maximum shelf opening/spacing in inches if a numeric value or adjustable maximum is provided; else null.
        - shelf_spacing_text: The shelf spacing/opening text exactly as stated (e.g., "adjustable 9–11 in", "10 in between shelves").
        - per_shelf_capacity_lbs_numeric: The per-shelf load capacity in lbs if a numeric value is provided or derived from cited method; else null.
        - per_shelf_capacity_text: The capacity text as stated (e.g., "≥30 lbs per shelf", "supports 35 lbs").
        - materials_cost_usd_numeric: The estimated materials cost in USD if stated as a number; else null.
        - materials_cost_text: The cost text as stated.
        - completion_time_text: The time to build as stated (e.g., "weekend project", "4–6 hours").
        - weekend_claim_explicit: true if the answer explicitly claims it is a weekend project or time consistent with a weekend and attributes it to the source; else false or null.
        - cited_urls: All URLs mentioned in the answer that relate to this plan/specs (include the plan URL too if present).

    IMPORTANT:
    - Extract only what is explicitly present in the answer text.
    - Do not invent values. Use null when not stated.
    - For numeric fields, extract numbers only if a concrete value is given in the answer (not your inference).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _flatten_sources(plan: Optional[PlanSource], specs: Optional[PlanSpecs]) -> List[str]:
    urls: List[str] = []
    if plan:
        if plan.url:
            urls.append(plan.url)
        if plan.alt_urls:
            urls.extend([u for u in plan.alt_urls if u])
    if specs and specs.cited_urls:
        urls.extend([u for u in specs.cited_urls if u])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u and (u not in seen):
            seen.add(u)
            deduped.append(u)
    return deduped


def _num_in_range(val: Optional[float], lo: float, hi: float) -> bool:
    if val is None:
        return False
    try:
        return lo <= float(val) <= hi
    except Exception:
        return False


def _tools_within_allowed(tools: List[str]) -> bool:
    if not tools:
        return False
    allowed = {
        "circular saw", "table saw", "saw",  # accept generic "saw" if clearly refers to circular/table
        "drill", "drill/driver", "power drill",
        "clamps", "bar clamps", "clamp",
        "tape measure", "measuring tape",
        "square", "speed square", "try square", "combination square", "carpenter's square"
    }
    benign_extras = {
        "pencil", "wood glue", "glue", "sandpaper", "sanding block", "finish", "paint", "paint brush",
        "safety glasses", "hearing protection", "dust mask", "respirator", "rags", "screws", "screwdriver"
    }

    def canon(t: str) -> str:
        t = (t or "").strip().lower()
        # simple canonicalization
        t = t.replace("-", " ").replace("electric", "").replace("cordless", "").strip()
        return t

    for t in tools:
        ct = canon(t)
        if ct in allowed:
            continue
        if ct in benign_extras:
            continue
        # very light synonyms
        if ct in {"speedsquare"}:
            continue
        if ct in {"measuring tape"}:
            continue
        # If the tool clearly indicates specialized gear (router, jigsaw, miter saw, pocket hole jig, nailer, sander power-tool), then disallow
        if any(bad in ct for bad in ["router", "jigsaw", "miter", "pocket hole jig", "kreg jig", "nailer", "brad nailer", "orbital sander", "sander"]):
            return False
        # unknown tools default to not allowed
        if ct not in allowed and ct not in benign_extras:
            return False
    return True


def _joinery_allowed(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    if ("butt" in t and "screw" in t):  # butt joints with screws (and glue)
        return True
    if "pocket" in t and ("screw" in t or "hole" in t):
        return True
    if "dado" in t or "dados" in t:
        return True
    return False


def _spacing_within_8_12(min_in: Optional[float], max_in: Optional[float]) -> bool:
    # If both provided, both must be within [8,12].
    # If only one provided, it must be within [8,12].
    if min_in is None and max_in is None:
        return False
    if min_in is not None and max_in is not None:
        return 8.0 <= min_in <= 12.0 and 8.0 <= max_in <= 12.0
    only = min_in if min_in is not None else max_in
    return _num_in_range(only, 8.0, 12.0)


def _bool_true(val: Optional[bool]) -> bool:
    return bool(val is True)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_plan_and_source_checks(
    evaluator: Evaluator,
    parent_node,
    plan: Optional[PlanSource],
) -> None:
    plan_and_source = evaluator.add_parallel(
        id="Plan_And_Source",
        desc="The answer identifies a specific plan and provides a verifiable source to locate it.",
        parent=parent_node,
        critical=True
    )

    # Specific_Plan_Identified (existence as critical custom node)
    plan_identified = evaluator.add_custom_node(
        result=bool(plan and plan.name and plan.name.strip()),
        id="Specific_Plan_Identified",
        desc="Names/identifies one specific DIY bookshelf plan/design (not generic advice or a category page).",
        parent=plan_and_source,
        critical=True
    )

    # Source_Provided: if URL present, verify it corresponds to the plan; else fail by custom node
    if plan and (plan.url or (plan.alt_urls and len(plan.alt_urls) > 0)):
        src_leaf = evaluator.add_leaf(
            id="Source_Provided",
            desc="Provides a verifiable source (URL) that clearly corresponds to the identified plan.",
            parent=plan_and_source,
            critical=True
        )
        plan_name = plan.name or "the specified DIY bookshelf plan"
        claim = f"This webpage corresponds to the DIY bookshelf plan titled or clearly identified as '{plan_name}'."
        urls = plan.url if plan.url else (plan.alt_urls[0] if plan.alt_urls else None)
        await evaluator.verify(
            claim=claim,
            node=src_leaf,
            sources=urls,
            additional_instruction="Match the plan title/name or unique plan description on the page. Allow reasonable variants and minor formatting differences."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Source_Provided",
            desc="Provides a verifiable source (URL) that clearly corresponds to the identified plan.",
            parent=plan_and_source,
            critical=True
        )


async def build_constraints_checks(
    evaluator: Evaluator,
    parent_node,
    plan: Optional[PlanSource],
    specs: Optional[PlanSpecs]
) -> None:
    constraints_root = evaluator.add_parallel(
        id="Constraint_Compliance_Using_Stated_Specs",
        desc="Using specs stated in the answer (and attributable to the plan/source), the plan meets all constraints.",
        parent=parent_node,
        critical=True
    )

    all_urls = _flatten_sources(plan, specs)

    claims_to_verify: List[Tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    # 1) Depth_Within_10_to_12_Inches
    depth_node = evaluator.add_parallel(
        id="Depth_Within_10_to_12_Inches",
        desc="Answer states the bookshelf/shelf depth as a numeric value and it is within 10–12 inches (inclusive).",
        parent=constraints_root,
        critical=True
    )
    depth_in_range = evaluator.add_custom_node(
        result=_num_in_range(specs.depth_in_numeric if specs else None, 10.0, 12.0),
        id="Depth_Within_10_to_12_Inches_value_check",
        desc="Depth numeric value is present and within 10–12 inches.",
        parent=depth_node,
        critical=True
    )
    depth_src = evaluator.add_leaf(
        id="Depth_Within_10_to_12_Inches_source_support",
        desc="Depth is supported by the cited source(s).",
        parent=depth_node,
        critical=True
    )
    depth_txt = (specs.depth_text if specs and specs.depth_text else f"{specs.depth_in_numeric} in" if specs and specs.depth_in_numeric is not None else "within 10–12 in")
    depth_claim = f"The plan page states that the bookshelf/shelf depth is {depth_txt}, which is within 10–12 inches."
    claims_to_verify.append((
        depth_claim,
        all_urls,
        depth_src,
        "Look for depth dimensions or material nominal sizes like '1x12' (actual ~11.25 in). Allow small rounding."
    ))

    # 2) Wood_Is_Explicitly_Beginner_Friendly
    wood_node = evaluator.add_parallel(
        id="Wood_Is_Explicitly_Beginner_Friendly",
        desc="Answer specifies the wood type and indicates it is beginner-friendly/easy to work with, attributable to the source.",
        parent=constraints_root,
        critical=True
    )
    wood_ok = evaluator.add_custom_node(
        result=bool(specs and specs.wood_type and _bool_true(specs.wood_beginner_friendly_explicit)),
        id="Wood_Is_Explicitly_Beginner_Friendly_value_check",
        desc="Wood type is specified and beginner-friendly is explicitly stated.",
        parent=wood_node,
        critical=True
    )
    wood_src = evaluator.add_leaf(
        id="Wood_Is_Explicitly_Beginner_Friendly_source_support",
        desc="Wood beginner-friendly claim is supported by the cited source(s).",
        parent=wood_node,
        critical=True
    )
    wood_type = specs.wood_type if specs and specs.wood_type else "the specified wood"
    wood_note = specs.wood_beginner_friendly_note if specs and specs.wood_beginner_friendly_note else "beginner-friendly or easy to work"
    wood_claim = f"The plan/source recommends using {wood_type} and explicitly indicates it is {wood_note}."
    claims_to_verify.append((
        wood_claim,
        all_urls,
        wood_src,
        "Accept phrases like 'good for beginners', 'easy to work', or 'beginner-friendly' referring to the stated wood."
    ))

    # 3) Tools_Limited_To_Specified_Beginner_Tools
    tools_node = evaluator.add_parallel(
        id="Tools_Limited_To_Specified_Beginner_Tools",
        desc="Answer states required tools and they do not go beyond circular/table saw, drill, clamps, tape measure, and square.",
        parent=constraints_root,
        critical=True
    )
    tools_ok = evaluator.add_custom_node(
        result=_tools_within_allowed(specs.tools if specs else []),
        id="Tools_Limited_To_Specified_Beginner_Tools_value_check",
        desc="Stated tool list is limited to allowed beginner tools (no specialized tools).",
        parent=tools_node,
        critical=True
    )
    tools_src = evaluator.add_leaf(
        id="Tools_Limited_To_Specified_Beginner_Tools_source_support",
        desc="Tool requirements are supported by the cited source(s).",
        parent=tools_node,
        critical=True
    )
    tools_list_str = ", ".join(specs.tools) if specs and specs.tools else "basic beginner tools (circular saw or table saw, drill, clamps, tape measure, and square)"
    tools_claim = f"According to the plan page, the required tools are limited to basic beginner tools such as {tools_list_str}, and no specialized tools are required."
    claims_to_verify.append((
        tools_claim,
        all_urls,
        tools_src,
        "Allow synonyms (e.g., speed square for square) and ignore consumables/safety gear. If the page lists only basic tools, consider it supported."
    ))

    # 4) Joinery_Is_One_Of_Allowed_Methods
    joinery_node = evaluator.add_parallel(
        id="Joinery_Is_One_Of_Allowed_Methods",
        desc="Answer specifies the joinery method and it is one of: butt joints with screws and glue, dados, or pocket screws.",
        parent=constraints_root,
        critical=True
    )
    joinery_ok = evaluator.add_custom_node(
        result=_joinery_allowed(specs.joinery if specs else None),
        id="Joinery_Is_One_Of_Allowed_Methods_value_check",
        desc="Joinery method is among allowed beginner methods.",
        parent=joinery_node,
        critical=True
    )
    joinery_src = evaluator.add_leaf(
        id="Joinery_Is_One_Of_Allowed_Methods_source_support",
        desc="Joinery method is supported by the cited source(s).",
        parent=joinery_node,
        critical=True
    )
    joinery_text = specs.joinery if specs and specs.joinery else "one of the allowed beginner joinery methods"
    joinery_claim = f"The plan uses {joinery_text}, which is an allowed beginner joinery method (butt joints with screws and glue, dados, or pocket screws)."
    claims_to_verify.append((
        joinery_claim,
        all_urls,
        joinery_src,
        "Allow minor wording variations like 'butt joints with screws and glue', 'pocket holes/screws', 'dado/dados'."
    ))

    # 5) Shelf_Spacing_Within_8_to_12_Inches
    spacing_node = evaluator.add_parallel(
        id="Shelf_Spacing_Within_8_to_12_Inches",
        desc="Answer states shelf spacing/opening height (or adjustable range) within 8–12 inches (inclusive).",
        parent=constraints_root,
        critical=True
    )
    spacing_ok = evaluator.add_custom_node(
        result=_spacing_within_8_12(
            specs.shelf_spacing_min_in if specs else None,
            specs.shelf_spacing_max_in if specs else None
        ),
        id="Shelf_Spacing_Within_8_to_12_Inches_value_check",
        desc="Shelf spacing numeric value/range falls within 8–12 inches.",
        parent=spacing_node,
        critical=True
    )
    spacing_src = evaluator.add_leaf(
        id="Shelf_Spacing_Within_8_to_12_Inches_source_support",
        desc="Shelf spacing is supported by the cited source(s).",
        parent=spacing_node,
        critical=True
    )
    spacing_txt = (
        specs.shelf_spacing_text if specs and specs.shelf_spacing_text else
        f"{specs.shelf_spacing_min_in}–{specs.shelf_spacing_max_in} in"
        if specs and (specs.shelf_spacing_min_in is not None or specs.shelf_spacing_max_in is not None) else
        "between 8 and 12 inches"
    )
    spacing_claim = f"The plan specifies shelf spacing/opening height {spacing_txt}, which falls between 8 and 12 inches for book shelves."
    claims_to_verify.append((
        spacing_claim,
        all_urls,
        spacing_src,
        "If shelves are adjustable, verify that typical recommended positions yield 8–12 inch openings."
    ))

    # 6) Per_Shelf_Capacity_At_Least_30_Lbs
    capacity_node = evaluator.add_parallel(
        id="Per_Shelf_Capacity_At_Least_30_Lbs",
        desc="Answer provides a per-shelf load capacity of at least 30 lbs tied to the plan/source.",
        parent=constraints_root,
        critical=True
    )
    capacity_ok = evaluator.add_custom_node(
        result=bool(specs and specs.per_shelf_capacity_lbs_numeric is not None and specs.per_shelf_capacity_lbs_numeric >= 30.0),
        id="Per_Shelf_Capacity_At_Least_30_Lbs_value_check",
        desc="Per-shelf capacity numeric value is present and ≥ 30 lbs.",
        parent=capacity_node,
        critical=True
    )
    capacity_src = evaluator.add_leaf(
        id="Per_Shelf_Capacity_At_Least_30_Lbs_source_support",
        desc="Per-shelf capacity claim is supported by cited source(s) (plan explicitly or via cited load/sag method).",
        parent=capacity_node,
        critical=True
    )
    cap_txt = (
        specs.per_shelf_capacity_text if specs and specs.per_shelf_capacity_text else
        f"{specs.per_shelf_capacity_lbs_numeric} lbs"
        if specs and specs.per_shelf_capacity_lbs_numeric is not None else
        "at least 30 lbs"
    )
    capacity_claim = f"The cited source(s) support that each shelf can safely support {cap_txt} (≥ 30 lbs)."
    claims_to_verify.append((
        capacity_claim,
        all_urls,
        capacity_src,
        "Support can be explicit from the plan or via a cited load/sag calculator using the plan’s material/specs."
    ))

    # 7) Materials_Cost_Under_50_USD
    cost_node = evaluator.add_parallel(
        id="Materials_Cost_Under_50_USD",
        desc="Answer states an estimated materials cost under $50, consistent with the plan/source.",
        parent=constraints_root,
        critical=True
    )
    cost_ok = evaluator.add_custom_node(
        result=bool(specs and specs.materials_cost_usd_numeric is not None and specs.materials_cost_usd_numeric < 50.0),
        id="Materials_Cost_Under_50_USD_value_check",
        desc="Estimated materials cost numeric value is present and < $50.",
        parent=cost_node,
        critical=True
    )
    cost_src = evaluator.add_leaf(
        id="Materials_Cost_Under_50_USD_source_support",
        desc="Materials cost estimate under $50 is supported by the cited source(s).",
        parent=cost_node,
        critical=True
    )
    cost_txt = (
        specs.materials_cost_text if specs and specs.materials_cost_text else
        f"${specs.materials_cost_usd_numeric:.2f}"
        if specs and specs.materials_cost_usd_numeric is not None else
        "under $50"
    )
    cost_claim = f"The plan/source indicates the estimated materials cost is {cost_txt}, which is under $50."
    claims_to_verify.append((
        cost_claim,
        all_urls,
        cost_src,
        "Look for an itemized cost estimate or a stated budget summary indicating under $50."
    ))

    # 8) Completable_As_A_Weekend_Project
    weekend_node = evaluator.add_parallel(
        id="Completable_As_A_Weekend_Project",
        desc="Answer states the build can be completed in a weekend (or time consistent with a weekend) and attributes this to the source.",
        parent=constraints_root,
        critical=True
    )
    weekend_ok = evaluator.add_custom_node(
        result=bool(specs and (_bool_true(specs.weekend_claim_explicit) or (specs.completion_time_text and "weekend" in specs.completion_time_text.lower()))),
        id="Completable_As_A_Weekend_Project_value_check",
        desc="Weekend/time-to-build claim is present and indicates a weekend-scale project.",
        parent=weekend_node,
        critical=True
    )
    weekend_src = evaluator.add_leaf(
        id="Completable_As_A_Weekend_Project_source_support",
        desc="Weekend/time-to-build claim is supported by cited source(s).",
        parent=weekend_node,
        critical=True
    )
    time_txt = specs.completion_time_text if specs and specs.completion_time_text else "a weekend-scale time estimate"
    weekend_claim = f"The plan/source indicates the build can be completed in {time_txt}, consistent with a weekend project."
    claims_to_verify.append((
        weekend_claim,
        all_urls,
        weekend_src,
        "Accept phrases like 'weekend project', 'one weekend', or total build time reasonably fitting a weekend."
    ))

    # Execute all URL-based verifications (in parallel where applicable)
    await evaluator.batch_verify(claims_to_verify)


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
    # Initialize evaluator (root is non-critical by framework design)
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
        default_model=model,
    )

    # Create a critical sequential node as the main compliance root (reflecting the rubric's root)
    compliance_root = evaluator.add_sequential(
        id="Beginner_DIY_Bookshelf_Design_Compliance",
        desc="Evaluates whether the answer identifies ONE specific DIY bookshelf plan and source, and whether that plan (as described with key specs in the answer) satisfies all stated constraints.",
        parent=root,
        critical=True
    )

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=BookshelfPlanExtraction,
        extraction_name="bookshelf_plan_extraction"
    )

    # Build "Plan & Source" checks
    await build_plan_and_source_checks(
        evaluator=evaluator,
        parent_node=compliance_root,
        plan=extraction.plan
    )

    # Build "Constraint Compliance" checks
    await build_constraints_checks(
        evaluator=evaluator,
        parent_node=compliance_root,
        plan=extraction.plan,
        specs=extraction.specs
    )

    # Return evaluation summary
    return evaluator.get_summary()