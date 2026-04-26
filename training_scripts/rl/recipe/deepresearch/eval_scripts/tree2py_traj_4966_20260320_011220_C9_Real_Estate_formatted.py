import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_mixed_use_2023_2025"
TASK_DESCRIPTION = """
Identify four large-scale mixed-use development projects currently under construction or that broke ground between 2023 and 2025 in the United States, where each project meets ALL of the following criteria:

1. Each project must be located in a different U.S. state (all four projects must be in four different states)
2. The project must be valued at $1 billion or more
3. The development site must be 60 acres or larger
4. The project must include at least 700 residential units (this can include apartments, condominiums, townhomes, or single-family homes, or any combination thereof)
5. The project must include commercial or office space
6. The project must include retail space
7. The project must combine at least three different types of uses (for example: residential, commercial, retail, hotel, entertainment, public amenities, or other distinct use types)

For each of the four projects, provide:
- The official project name
- The city and state where it is located
- The total project cost
- The total site acreage
- The total number of residential units
- A description of the commercial/office component
- A description of the retail component
- A list of all the different use types included in the project
- Reference URLs that verify this information
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProjectItem(BaseModel):
    # Basic info
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # Accept full state name or 2-letter postal code
    developer: Optional[str] = None

    # Timeline (answer-provided phrasing; keep flexible strings)
    timeline_status: Optional[str] = None  # e.g., "under construction" / "broke ground Month YYYY"
    timeline_detail: Optional[str] = None  # any free-text detail (e.g., phase start, press-release reference)

    # Scale & specs (keep strings to allow ranges/estimates)
    total_cost: Optional[str] = None        # e.g., "$1.5 billion", "USD 2B", "approx $1B"
    site_acreage: Optional[str] = None      # e.g., "75 acres", "~100 acres"
    residential_units: Optional[str] = None # e.g., "1,200 units", "700+ homes"

    # Mixed-use components
    commercial_office_desc: Optional[str] = None  # short description or quotation from answer
    retail_desc: Optional[str] = None             # short description or quotation from answer
    use_types: List[str] = Field(default_factory=list)  # e.g., ["residential", "office", "retail", "hotel"]

    # Source URLs explicitly cited in the answer
    sources_overall: List[str] = Field(default_factory=list)      # general/basic info sources
    sources_specs: List[str] = Field(default_factory=list)        # specs (cost/acreage/units) sources
    sources_components: List[str] = Field(default_factory=list)   # mixed-use components sources


class ProjectsExtraction(BaseModel):
    projects: List[ProjectItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_projects() -> str:
    return """
    Extract up to FOUR mixed-use development projects listed in the answer. For each project, return the following fields.
    IMPORTANT: Extract ONLY what appears explicitly in the answer text. Do NOT invent content. If a field is missing, set it to null (for string fields) or [] (for list fields).

    For each project, extract:
    - name: Official project name, exactly as stated in the answer
    - city: City of the project
    - state: State of the project (either full name or 2-letter postal abbreviation), exactly as written in the answer
    - developer: The development firm/company, if stated

    - timeline_status: A short phrase from the answer indicating if/that it is "under construction" or "broke ground" and any phrasing around 2023–2025
    - timeline_detail: Any additional free-text timeline detail (month/year notes, phase notes, quotes), if present

    - total_cost: Total project cost/budget exactly as written (e.g., "$1 billion", "$1.2–1.4B", "USD 2B")
    - site_acreage: The total site acreage exactly as written (e.g., "60 acres", "~75 acres", "over 100 acres")
    - residential_units: The total number of residential units exactly as written (e.g., "700 units", "1,000+ homes")

    - commercial_office_desc: A short excerpt or summary that indicates the project includes commercial or office space
    - retail_desc: A short excerpt or summary that indicates the project includes retail space
    - use_types: List all distinct use types explicitly mentioned (e.g., ["residential", "office", "retail", "hotel", "entertainment", "public amenities"])

    - sources_overall: URLs explicitly cited for the project's basic information (full URLs required)
    - sources_specs: URLs explicitly cited that support the cost/acreage/units (full URLs required)
    - sources_components: URLs explicitly cited that support commercial/office, retail, and other use types (full URLs required)

    Return a JSON object with a single key "projects" whose value is an array of up to 4 objects with the fields above.
    Follow URL extraction rules: only extract URLs that are explicitly present in the answer text. Normalize missing protocols by prefixing http:// if needed.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_nonempty_str(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not _is_nonempty_str(u):
            continue
        key = u.strip()
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _all_sources(p: ProjectItem) -> List[str]:
    return _dedup_urls((p.sources_overall or []) + (p.sources_specs or []) + (p.sources_components or []))


def _prefer_or_all(primary: List[str], fallback_all: List[str]) -> List[str]:
    return _dedup_urls(primary if primary else fallback_all)


def _fmt_list(items: List[str]) -> str:
    return ", ".join([x for x in items if _is_nonempty_str(x)]) if items else ""


def _norm_state(s: Optional[str]) -> Optional[str]:
    return s.strip().lower() if _is_nonempty_str(s) else None


# --------------------------------------------------------------------------- #
# Verification for a single project                                           #
# --------------------------------------------------------------------------- #
async def verify_project(
    evaluator: Evaluator,
    parent_node,
    proj: ProjectItem,
    index: int,
    prior_states: List[str],
) -> None:
    """
    Build verification subtree for a single project according to the rubric.
    index: 0..3, Project_1..Project_4
    prior_states: normalized state strings for previously processed projects (for cross-state uniqueness checks)
    """
    n = index + 1
    proj_node = evaluator.add_parallel(
        id=f"Project_{n}",
        desc=(
            "First qualifying mixed-use development project" if n == 1 else
            ("Second qualifying mixed-use development project in a different state" if n == 2 else
             ("Third qualifying mixed-use development project in a different state" if n == 3 else
              "Fourth qualifying mixed-use development project in a different state"))
        ),
        parent=parent_node,
        critical=False
    )

    # Gather URLs
    all_urls = _all_sources(proj)

    # 1) Basic info sources presence (critical)
    evaluator.add_custom_node(
        result=len(all_urls) > 0,
        id=f"Reference_URLs_Project_{n}",
        desc=f"Provide reference URL(s) for the {'first' if n==1 else 'second' if n==2 else 'third' if n==3 else 'fourth'} project's basic information",
        parent=proj_node,
        critical=True
    )

    # 2) Project name (critical, source-grounded)
    if _is_nonempty_str(proj.name) and len(all_urls) > 0:
        nm_leaf = evaluator.add_leaf(
            id=f"Project_Name_{n}",
            desc=f"Provide the official name of the {'first' if n==1 else 'second' if n==2 else 'third' if n==3 else 'fourth'} project",
            parent=proj_node,
            critical=True
        )
        await evaluator.verify(
            claim=f"The official project name is '{proj.name}'.",
            node=nm_leaf,
            sources=all_urls,
            additional_instruction="Verify that the cited page(s) explicitly reference this official project name or an equivalent accepted variant."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"Project_Name_{n}",
            desc=f"Provide the official name of the {'first' if n==1 else 'second' if n==2 else 'third' if n==3 else 'fourth'} project",
            parent=proj_node,
            critical=True
        )

    # 3) Location (critical, source-grounded)
    if _is_nonempty_str(proj.city) and _is_nonempty_str(proj.state) and len(all_urls) > 0:
        loc_leaf = evaluator.add_leaf(
            id=f"Location_{n}",
            desc=f"Identify the city and state where the {'first' if n==1 else 'second' if n==2 else 'third' if n==3 else 'fourth'} project is located",
            parent=proj_node,
            critical=True
        )
        await evaluator.verify(
            claim=f"The project is located in {proj.city}, {proj.state} (United States).",
            node=loc_leaf,
            sources=all_urls,
            additional_instruction="Confirm the city and state on the cited page(s). Minor formatting variations are acceptable."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"Location_{n}",
            desc=f"Identify the city and state where the {'first' if n==1 else 'second' if n==2 else 'third' if n==3 else 'fourth'} project is located",
            parent=proj_node,
            critical=True
        )

    # 3b) Cross-state uniqueness for Projects 2–4 (critical, logic-only)
    # Requirement: Each project must be in a different U.S. state
    norm_state = _norm_state(proj.state)
    if n >= 2:
        unique_leaf = evaluator.add_leaf(
            id=f"State_Distinctness_{n}",
            desc=f"Project {n} is in a different U.S. state than previously listed projects",
            parent=proj_node,
            critical=True
        )
        previous_display = ", ".join(prior_states) if prior_states else "None"
        await evaluator.verify(
            claim=f"The state for project {n} is '{proj.state or ''}', and it is different from previously used states: {previous_display}.",
            node=unique_leaf,
            additional_instruction="Judge only the logical distinctness between the current state and the listed prior states. If the current state appears in the prior set, mark as incorrect. Ignore web evidence for this specific check."
        )

    # 4) Developer (critical, source-grounded)
    if _is_nonempty_str(proj.developer) and len(all_urls) > 0:
        dev_leaf = evaluator.add_leaf(
            id=f"Developer_{n}",
            desc=f"Identify the real estate development firm or company developing the {'first' if n==1 else 'second' if n==2 else 'third' if n==3 else 'fourth'} project",
            parent=proj_node,
            critical=True
        )
        await evaluator.verify(
            claim=f"The developer for this project is '{proj.developer}'.",
            node=dev_leaf,
            sources=all_urls,
            additional_instruction="Verify that the cited page(s) clearly identify this developer as developing the project (lead developer, master developer, or co-developer acceptable)."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"Developer_{n}",
            desc=f"Identify the real estate development firm or company developing the {'first' if n==1 else 'second' if n==2 else 'third' if n==3 else 'fourth'} project",
            parent=proj_node,
            critical=True
        )

    # 5) Timeline: broke ground or under construction between 2023–2025 (critical, source-grounded)
    if len(all_urls) > 0:
        tl_leaf = evaluator.add_leaf(
            id=f"Timeline_{n}",
            desc=f"Verify that the {'first' if n==1 else 'second' if n==2 else 'third' if n==3 else 'fourth'} project broke ground or is under construction between 2023-2025",
            parent=proj_node,
            critical=True
        )
        # Provide any known details to help the verifier (but still require evidence)
        detail_txt = proj.timeline_status or proj.timeline_detail or ""
        await evaluator.verify(
            claim="The project either broke ground or was under construction between 2023 and 2025 inclusive.",
            node=tl_leaf,
            sources=all_urls,
            additional_instruction=(
                "Verify on the cited page(s) that construction status (e.g., 'under construction') or groundbreaking occurred in 2023, 2024, or 2025. "
                "Accept synonyms like 'construction began in 2024', 'broke ground in 2023', or clearly ongoing construction during that window. "
                f"Answer-provided hint (may be incomplete): {detail_txt!r}"
            )
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"Timeline_{n}",
            desc=f"Verify that the {'first' if n==1 else 'second' if n==2 else 'third' if n==3 else 'fourth'} project broke ground or is under construction between 2023-2025",
            parent=proj_node,
            critical=True
        )

    # 6) Project Specifications (critical parallel group)
    specs_node = evaluator.add_parallel(
        id=f"Project_Specifications_{n}",
        desc=f"Verify that the {'first' if n==1 else 'second' if n==2 else 'third' if n==3 else 'fourth'} project meets all size and scale requirements",
        parent=proj_node,
        critical=True
    )

    specs_urls = _prefer_or_all(proj.sources_specs, all_urls)

    # 6.a) Cost >= $1B (critical)
    if len(specs_urls) > 0:
        cost_leaf = evaluator.add_leaf(
            id=f"Cost_Minimum_{n}",
            desc="The project cost must be $1 billion or more",
            parent=specs_node,
            critical=True
        )
        await evaluator.verify(
            claim="The project's total cost is at least $1 billion (USD).",
            node=cost_leaf,
            sources=specs_urls,
            additional_instruction="Confirm that the page(s) indicate a total cost/budget of $1,000,000,000 or higher. Accept approximate or range formats if the lower bound is ≥ $1B."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"Cost_Minimum_{n}",
            desc="The project cost must be $1 billion or more",
            parent=specs_node,
            critical=True
        )

    # 6.b) Acreage >= 60 acres (critical)
    if len(specs_urls) > 0:
        ac_leaf = evaluator.add_leaf(
            id=f"Acreage_Minimum_{n}",
            desc="The development site must be 60 acres or larger",
            parent=specs_node,
            critical=True
        )
        await evaluator.verify(
            claim="The project's development site is at least 60 acres.",
            node=ac_leaf,
            sources=specs_urls,
            additional_instruction="Confirm that the page(s) indicate a site area ≥ 60 acres. Accept equivalent phrasings like 'over 60 acres', 'approx 60 acres', '60+ acres'."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"Acreage_Minimum_{n}",
            desc="The development site must be 60 acres or larger",
            parent=specs_node,
            critical=True
        )

    # 6.c) Residential units >= 700 (critical)
    if len(specs_urls) > 0:
        ru_leaf = evaluator.add_leaf(
            id=f"Residential_Units_{n}",
            desc="The project must include at least 700 residential units",
            parent=specs_node,
            critical=True
        )
        await evaluator.verify(
            claim="The project includes at least 700 residential units.",
            node=ru_leaf,
            sources=specs_urls,
            additional_instruction="Verify on the page(s) that the residential unit count is ≥ 700 (including apartments, condos, townhomes, or single-family homes; cumulative across phases acceptable if clearly indicated)."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"Residential_Units_{n}",
            desc="The project must include at least 700 residential units",
            parent=specs_node,
            critical=True
        )

    # 6.d) Specs reference URLs existence (critical)
    evaluator.add_custom_node(
        result=len(proj.sources_specs) > 0,
        id=f"Reference_URLs_Specs_{n}",
        desc="Provide reference URL(s) supporting the project specifications",
        parent=specs_node,
        critical=True
    )

    # 7) Mixed-Use Components (critical parallel group)
    comp_node = evaluator.add_parallel(
        id=f"Mixed_Use_Components_{n}",
        desc=f"Verify that the {'first' if n==1 else 'second' if n==2 else 'third' if n==3 else 'fourth'} project includes all required mixed-use components",
        parent=proj_node,
        critical=True
    )
    comp_urls = _prefer_or_all(proj.sources_components, all_urls)

    # 7.a) Commercial/Office component (critical)
    if len(comp_urls) > 0:
        co_leaf = evaluator.add_leaf(
            id=f"Commercial_Office_Component_{n}",
            desc="The project includes commercial or office space",
            parent=comp_node,
            critical=True
        )
        co_hint = proj.commercial_office_desc or ""
        await evaluator.verify(
            claim="The project includes commercial space or office space as a component.",
            node=co_leaf,
            sources=comp_urls,
            additional_instruction=f"Verify on the cited page(s) that office and/or commercial space is part of the program. Hint from answer: {co_hint!r}"
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"Commercial_Office_Component_{n}",
            desc="The project includes commercial or office space",
            parent=comp_node,
            critical=True
        )

    # 7.b) Retail component (critical)
    if len(comp_urls) > 0:
        rt_leaf = evaluator.add_leaf(
            id=f"Retail_Component_{n}",
            desc="The project includes retail space",
            parent=comp_node,
            critical=True
        )
        rt_hint = proj.retail_desc or ""
        await evaluator.verify(
            claim="The project includes retail space as a component.",
            node=rt_leaf,
            sources=comp_urls,
            additional_instruction=f"Verify on the cited page(s) that retail is explicitly included. Hint from answer: {rt_hint!r}"
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"Retail_Component_{n}",
            desc="The project includes retail space",
            parent=comp_node,
            critical=True
        )

    # 7.c) At least three distinct use types (critical)
    if len(comp_urls) > 0:
        ut_leaf = evaluator.add_leaf(
            id=f"Three_Use_Types_{n}",
            desc="The project combines at least three different use types",
            parent=comp_node,
            critical=True
        )
        ut_list = _fmt_list(proj.use_types)
        await evaluator.verify(
            claim=f"The project program includes at least three distinct use types (e.g., residential, commercial/office, retail, hotel, entertainment, public amenities, etc.). Stated use types: {ut_list}.",
            node=ut_leaf,
            sources=comp_urls,
            additional_instruction="Verify that the cited page(s) substantiate three or more distinct program categories. Synonyms or closely related phrases count toward the same category."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"Three_Use_Types_{n}",
            desc="The project combines at least three different use types",
            parent=comp_node,
            critical=True
        )

    # 7.d) Components reference URLs existence (critical)
    evaluator.add_custom_node(
        result=len(proj.sources_components) > 0,
        id=f"Reference_URLs_Components_{n}",
        desc="Provide reference URL(s) supporting the mixed-use components",
        parent=comp_node,
        critical=True
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the 'four U.S. large-scale mixed-use projects (2023–2025)' task.
    """
    # Initialize evaluator (root is non-critical and parallel for independent scoring across projects)
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

    # Optional: a named top-level node to mirror the rubric's root name
    projects_root = evaluator.add_parallel(
        id="Identify_Four_Qualifying_Projects",
        desc="Identify four distinct large-scale mixed-use development projects in the United States, each located in a different state, that meet all specified criteria",
        parent=root,
        critical=False
    )

    # Extract structured project info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_projects(),
        template_class=ProjectsExtraction,
        extraction_name="projects_extraction"
    )

    # Normalize to exactly four slots (pad with empty items if fewer)
    projects: List[ProjectItem] = list(extracted.projects[:4])
    while len(projects) < 4:
        projects.append(ProjectItem())

    # Add ground-truth/requirement context (for transparency)
    evaluator.add_ground_truth({
        "requirements": {
            "distinct_states": True,
            "min_cost_usd": ">= $1,000,000,000",
            "min_site_acres": ">= 60 acres",
            "min_residential_units": ">= 700",
            "components_required": ["commercial_or_office", "retail"],
            "min_use_types": ">= 3 distinct categories",
            "timeline_window": "broke ground or under construction between 2023–2025 (inclusive)"
        }
    }, gt_type="constraints")

    # Track states to enforce cross-state uniqueness (Projects 2–4)
    prior_states_norm: List[str] = []

    # Build verification subtrees for each project
    for idx, proj in enumerate(projects):
        await verify_project(
            evaluator=evaluator,
            parent_node=projects_root,
            proj=proj,
            index=idx,
            prior_states=prior_states_norm.copy(),
        )
        # Update prior states list
        ns = _norm_state(proj.state)
        if ns:
            prior_states_norm.append(ns)

    # Return final structured summary
    return evaluator.get_summary()