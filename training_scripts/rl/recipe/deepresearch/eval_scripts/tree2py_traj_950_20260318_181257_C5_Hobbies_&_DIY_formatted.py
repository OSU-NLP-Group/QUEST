import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "beginner_woodworking_projects_and_tools"
TASK_DESCRIPTION = """
You are starting woodworking as a beginner hobby and want to identify suitable projects and acquire the necessary tools within budget. Find 3 beginner-friendly woodworking projects that meet the following criteria:

1. Each project must have a free plan available online from an established woodworking website (such as Ana White, Kreg Tool, Woodcraft, Wood Magazine, Rockler, or similar reputable sources)
2. Each project must be completable using only basic hand tools (power tools beyond an optional drill are not permitted)
3. Each project's estimated material cost must be under $75

For each of the 3 projects, provide:
- The direct URL to the free plan
- A list of the basic hand tools required to complete the project
- An estimated material cost based on the plan specifications or reasonable estimates for the required lumber and supplies

Additionally, identify one online retailer where a complete set of basic woodworking hand tools can be purchased. Provide:
- The retailer's website URL
- Confirmation that a complete basic hand tool set (including items such as hand saw, chisels, measuring tools, and hand drill or similar essential tools) is available at this retailer for a total cost under $150
"""

# A non-exhaustive list of established/reputable woodworking sites to guide the verifier
ESTABLISHED_WOODWORKING_SITES = [
    "www.ana-white.com",
    "www.kregtool.com",
    "learn.kregtool.com",
    "www.woodcraft.com",
    "www.woodmagazine.com",
    "www.rockler.com",
    "www.familyhandyman.com",
    "www.popularwoodworking.com",
    "www.instructables.com",
    "www.thisoldhouse.com",
    "www.diydiva.net",
    "www.thewoodwhisperer.com",
    "www.finewoodworking.com",
    "www.homedit.com",
    "www.thesprucecrafts.com",
]
ESTABLISHED_SITES_NOTE = (
    "Established sites include (but are not limited to): Ana White, Kreg Tool, Woodcraft, "
    "Wood Magazine, Rockler, Family Handyman, Popular Woodworking, Instructables, This Old House, "
    "The Wood Whisperer, Fine Woodworking, The Spruce Crafts. Equivalent reputable sites are acceptable."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProjectItem(BaseModel):
    title: Optional[str] = None
    plan_url: Optional[str] = None
    tools: List[str] = Field(default_factory=list)
    estimated_material_cost: Optional[str] = None
    price_support_urls: List[str] = Field(default_factory=list)


class RetailerInfo(BaseModel):
    retailer_name: Optional[str] = None
    retailer_url: Optional[str] = None
    product_url: Optional[str] = None  # Direct page for the kit/set if provided
    total_price: Optional[str] = None
    included_tools: List[str] = Field(default_factory=list)


class WoodworkingExtraction(BaseModel):
    projects: List[ProjectItem] = Field(default_factory=list)
    retailer: Optional[RetailerInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_structured() -> str:
    return """
    Extract structured information from the answer for up to three beginner woodworking projects and one hand tool retailer.

    For projects, extract up to the first 3 mentioned. For each project, return an object with:
    - title: the project title/name as written in the answer (or null if not provided)
    - plan_url: the direct URL to the free plan page (must be a full URL if present; otherwise null)
    - tools: a list of the basic hand tools listed in the answer for this project (e.g., hand saw, chisels, measuring tape, hand drill/brace, screwdriver, hammer, block plane, marking gauge, square, clamps)
    - estimated_material_cost: the estimated materials cost text as written (e.g., "$60", "about $50-70", "under $75"); return null if not present
    - price_support_urls: any additional URLs in the answer used to justify or estimate materials cost (e.g., lumber price pages); if none, return an empty list

    Also extract one retailer for a complete basic woodworking hand tool set:
    - retailer_name: the name of the retailer (e.g., Amazon, Home Depot, Harbor Freight, Rockler, Woodcraft)
    - retailer_url: the retailer's website or relevant category URL (full URL, if present)
    - product_url: a direct product or kit URL for the set if provided; otherwise null
    - total_price: the stated or implied total price for the complete basic hand tool set as written in the answer (e.g., "$129", "about $140"); null if not provided
    - included_tools: list the tools named for this set (e.g., saw, chisels, measuring tools, hand drill/brace, square, marking tools, clamps)

    Rules:
    - Extract exactly what appears in the answer; do not infer or invent new data.
    - For URLs, extract only valid full URLs that appear in the answer. If missing, return null.
    - If a field is not present in the answer, set it to null (or empty list for arrays).

    Return a JSON object with:
    {
      "projects": [ ... up to 3 ProjectItem objects ... ],
      "retailer": RetailerInfo or null
    }
    """


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def parse_any_price_to_float(text: Optional[str]) -> Optional[float]:
    """
    Attempt to parse a reasonable single price from a free-form text.
    Heuristics:
    - Prefer the smallest explicit $-amount <= 1000
    - If a range like 50-70 is present, use the upper bound (more conservative)
    - Accept phrases like "about 60", "around 65", "approx. $70"
    - If nothing parseable, return None
    """
    if not text:
        return None

    s = text.lower().strip()

    # Normalize dashes
    s = s.replace("–", "-").replace("—", "-")

    # Direct "under $X" or "less than $X" shortcuts (if X <= 1000)
    m_under = re.search(r"(under|less than)\s*\$?\s*(\d+(?:\.\d+)?)", s)
    if m_under:
        try:
            val = float(m_under.group(2))
            if 0 < val <= 1000:
                return val  # Treat "under $75" as <= 75 for gating check
        except Exception:
            pass

    # Ranges like "$50-$70" or "50-70"
    m_range = re.search(r"\$?\s*(\d+(?:\.\d+)?)\s*-\s*\$?\s*(\d+(?:\.\d+)?)", s)
    if m_range:
        try:
            low = float(m_range.group(1))
            high = float(m_range.group(2))
            if 0 < low <= 1000 and 0 < high <= 1000:
                return max(low, high)  # conservative
        except Exception:
            pass

    # General money amounts: capture $xx.xx or plain numbers near price words
    money_candidates = []
    for m in re.finditer(r"\$?\s*(\d{1,4}(?:\.\d{1,2})?)", s):
        try:
            val = float(m.group(1))
            if 0 < val <= 1000:
                money_candidates.append(val)
        except Exception:
            continue

    if money_candidates:
        # Choose the smallest plausible amount (budget-friendly intent)
        return min(money_candidates)

    return None


def is_under_budget(text: Optional[str], threshold: float) -> bool:
    """
    Return True if the parsed (heuristic) price is <= threshold,
    or if the phrase explicitly claims 'under $threshold'.
    """
    if not text:
        return False

    s = text.lower()
    if f"under ${int(threshold)}" in s or f"under {int(threshold)}" in s or f"less than ${int(threshold)}" in s:
        return True

    price = parse_any_price_to_float(text)
    return price is not None and price <= threshold


def first_n_projects(items: List[ProjectItem], n: int = 3) -> List[ProjectItem]:
    out = items[:n]
    while len(out) < n:
        out.append(ProjectItem())
    return out


def combine_sources(*args: Optional[List[str] | str]) -> List[str]:
    urls: List[str] = []
    for a in args:
        if isinstance(a, str):
            if a.strip():
                urls.append(a.strip())
        elif isinstance(a, list):
            for u in a:
                if isinstance(u, str) and u.strip():
                    urls.append(u.strip())
    # De-duplicate while preserving order
    seen = set()
    uniq = []
    for u in urls:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_single_project(
    evaluator: Evaluator,
    parent_node,
    project: ProjectItem,
    index: int,
):
    """
    Build verification nodes and run checks for a single project.
    """
    proj_num = index + 1
    proj_node = evaluator.add_parallel(
        id=f"project_{proj_num}",
        desc=f"Project #{proj_num}: beginner woodworking project verification",
        parent=parent_node,
        critical=False,  # Project nodes allow partial credit overall
    )

    # Plan URL provided (existence)
    plan_url_ok = bool(project.plan_url and project.plan_url.strip())
    evaluator.add_custom_node(
        result=plan_url_ok,
        id=f"project_{proj_num}_plan_url_provided",
        desc=f"Project #{proj_num}: a plan URL is provided",
        parent=proj_node,
        critical=True,
    )

    # Plan source is an established site and is a free plan page (URL verification)
    plan_source_leaf = evaluator.add_leaf(
        id=f"project_{proj_num}_plan_source",
        desc="Free plan URL is provided from an established woodworking website (e.g., Ana White, Kreg Tool, Woodcraft, Wood Magazine, Rockler or similar)",
        parent=proj_node,
        critical=True,
    )
    plan_claim = (
        "This URL is a free woodworking project plan page hosted by an established/reputable woodworking website. "
        + ESTABLISHED_SITES_NOTE
    )
    await evaluator.verify(
        claim=plan_claim,
        node=plan_source_leaf,
        sources=project.plan_url,
        additional_instruction=(
            "Confirm that the linked page offers a free plan/instructions (no paywall) for the described woodworking project. "
            "Evaluate whether the hosting site is reasonably established/reputable in woodworking. "
            "Minor site variations or subdomains are acceptable."
        ),
    )

    # Tool list provided (existence check)
    tools_ok = bool(project.tools and len(project.tools) > 0)
    evaluator.add_custom_node(
        result=tools_ok,
        id=f"project_{proj_num}_tools_list_provided",
        desc="A list of the basic hand tools required to complete the project is provided",
        parent=proj_node,
        critical=True,
    )

    # Material cost estimate provided (existence check)
    cost_text_ok = bool(project.estimated_material_cost and project.estimated_material_cost.strip())
    evaluator.add_custom_node(
        result=cost_text_ok,
        id=f"project_{proj_num}_material_cost_estimate_provided",
        desc="An estimated material cost is provided",
        parent=proj_node,
        critical=True,
    )

    # Requirements sub-node (critical)
    req_node = evaluator.add_parallel(
        id=f"project_{proj_num}_requirements",
        desc="Project meets hand tool and budget constraints",
        parent=proj_node,
        critical=True,
    )

    # Hand tools only (URL verification, gated by plan_url existence automatically)
    handtools_leaf = evaluator.add_leaf(
        id=f"project_{proj_num}_hand_tools_only",
        desc="Project can be completed using only basic hand tools (no power tools required beyond an optional drill)",
        parent=req_node,
        critical=True,
    )
    handtools_claim = (
        "This project can be reasonably completed using only basic hand tools. "
        "Power tools are not required beyond an optional drill; hand-tool substitutes (hand saw vs. circular/miter saw, brace/eggbeater drill vs. power drill, hand plane vs. sander) are acceptable."
    )
    await evaluator.verify(
        claim=handtools_claim,
        node=handtools_leaf,
        sources=project.plan_url,
        additional_instruction=(
            "Evaluate the steps and tool list on the plan page. "
            "If the plan lists power tools, determine whether obvious hand-tool alternatives would be feasible for a beginner "
            "(e.g., crosscuts/rips by hand saw, smoothing by hand plane/sandpaper). "
            "If the project fundamentally requires a power-only operation with no realistic hand alternative, mark as not supported."
        ),
    )

    # Budget under $75 - value check (custom) + source-supported check (URL)
    value_under_75 = is_under_budget(project.estimated_material_cost, 75.0)
    evaluator.add_custom_node(
        result=value_under_75,
        id=f"project_{proj_num}_cost_value_under_75",
        desc="Parsed materials cost is at or under $75",
        parent=req_node,
        critical=True,
    )

    cost_supported_leaf = evaluator.add_leaf(
        id=f"project_{proj_num}_material_cost_under_75_supported",
        desc="Estimated material cost is under $75 based on plan specifications or reasonable estimates",
        parent=req_node,
        critical=True,
    )
    cost_claim = "The total materials needed for this project can be obtained for under $75 USD."
    sources = combine_sources(project.plan_url, project.price_support_urls)
    await evaluator.verify(
        claim=cost_claim,
        node=cost_supported_leaf,
        sources=sources,
        additional_instruction=(
            "If the plan explicitly states a cost under $75, that is sufficient. "
            "Otherwise, use the bill of materials (lumber dimensions and quantities) and any provided links to judge if a reasonable estimate would fall under $75. "
            "Allow approximate phrasings like 'about $60' or 'under $75'. If the evidence is clearly above $75, mark as not supported."
        ),
    )


async def verify_tool_retailer(
    evaluator: Evaluator,
    parent_node,
    retailer: Optional[RetailerInfo],
):
    """
    Verify the retailer information and budget constraint for a complete basic hand tool set.
    """
    node = evaluator.add_parallel(
        id="tool_retailer",
        desc="Identification of online retailer for purchasing complete basic hand tool set",
        parent=parent_node,
        critical=True,  # Retailer requirement is essential for the overall task
    )

    # Existence of retailer URL
    retailer_url = (retailer.retailer_url if retailer else None) or ""
    product_url = (retailer.product_url if retailer else None) or ""
    retailer_any_url_ok = bool(retailer_url or product_url)
    evaluator.add_custom_node(
        result=retailer_any_url_ok,
        id="retailer_url_provided",
        desc="A specific online retailer URL is provided where basic hand tools can be purchased",
        parent=node,
        critical=True,
    )

    # Retailer offers a complete basic hand tool set (URL verification)
    has_set_leaf = evaluator.add_leaf(
        id="retailer_has_complete_set",
        desc="Retailer offers a complete basic woodworking hand tool set",
        parent=node,
        critical=True,
    )
    has_set_claim = (
        "This retailer page offers a complete basic woodworking hand tool set suitable for beginners, "
        "including essential items such as: hand saw, chisels, measuring tools (tape/rule/square), "
        "marking tools (pencil/marking gauge), a hand drill/brace (or manual drill equivalent), and clamps or a comparable starter kit."
    )
    await evaluator.verify(
        claim=has_set_claim,
        node=has_set_leaf,
        sources=(product_url or retailer_url),
        additional_instruction=(
            "Look for a bundle/kit that clearly covers the essentials for beginner woodworking hand tools. "
            "The exact brand or composition can vary, but the kit should reasonably include cutting, measuring, marking, and fastening tools. "
            "If the page only sells individual tools with no obvious set, or the kit is power-tool focused, mark as not supported."
        ),
    )

    # Budget under $150 - value check (custom) + source-supported check (URL)
    total_price_text = retailer.total_price if retailer else None
    value_under_150 = is_under_budget(total_price_text, 150.0)
    evaluator.add_custom_node(
        result=value_under_150,
        id="retailer_budget_value_under_150",
        desc="Parsed total price for the complete set is at or under $150",
        parent=node,
        critical=True,
    )

    budget_leaf = evaluator.add_leaf(
        id="retailer_budget_under_150_supported",
        desc="The total cost for a complete set of basic hand tools at this retailer is under $150",
        parent=node,
        critical=True,
    )
    budget_claim = "The listed total price for the complete basic woodworking hand tool set is under $150 USD."
    await evaluator.verify(
        claim=budget_claim,
        node=budget_leaf,
        sources=(product_url or retailer_url),
        additional_instruction=(
            "Use the product page price. If discounts are shown, use the current selling price. "
            "If the page lists multiple options, the one clearly described as a complete beginner hand tool kit must be under $150."
        ),
    )


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the beginner woodworking projects and tools task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates projects (soft) and retailer (critical gate)
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

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_structured(),
        template_class=WoodworkingExtraction,
        extraction_name="projects_and_retailer",
    )

    # Build the top-level grading structure
    # Root is non-critical here to allow mixing of critical and non-critical children in a consistent manner.
    # The retailer subtask is added as critical (as required by rubric).
    # Project subtasks are non-critical to allow partial credit if fewer than 3 are correct.

    # Add 3 project verifications
    projects = first_n_projects(extracted.projects if extracted and extracted.projects else [], 3)
    for i, proj in enumerate(projects):
        await verify_single_project(evaluator, root, proj, i)

    # Add retailer verification
    await verify_tool_retailer(evaluator, root, extracted.retailer if extracted else None)

    # Return final structured summary with the verification tree
    return evaluator.get_summary()