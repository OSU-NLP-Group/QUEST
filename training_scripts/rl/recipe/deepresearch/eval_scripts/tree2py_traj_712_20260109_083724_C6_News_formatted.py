import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fl_infra_2026_wastewater_top3"
TASK_DESCRIPTION = (
    "On January 7, 2026, Florida Governor Ron DeSantis announced $167.5 million in infrastructure funding for 34 rural "
    "and small community projects across the state. Among these projects, several focused on wastewater and sewer treatment "
    "infrastructure improvements. Identify the three largest wastewater or sewer treatment infrastructure projects (by awarded "
    "funding amount) from this announcement. For each of the three projects, provide: (1) The recipient entity name (city, town, "
    "county, or district), (2) The exact awarded funding amount, (3) The county where the project is located, and (4) A brief "
    "description of the project's stated purpose as described in the official announcement."
)

NO_URL_FAIL_INSTRUCTION = (
    "Important: Base your judgment only on the provided URLs. If no URL is provided for this verification, mark the claim as Incorrect."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProjectItem(BaseModel):
    recipient: Optional[str] = None
    amount: Optional[str] = None
    county: Optional[str] = None
    purpose: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ProjectsExtraction(BaseModel):
    projects: List[ProjectItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_projects() -> str:
    return """
    Extract wastewater or sewer treatment infrastructure projects listed in the answer (in the same order they appear).
    Only include projects explicitly identified in the answer text. For each included project, extract:

    - recipient: The recipient entity name (e.g., city, town, county, district, authority).
    - amount: The awarded amount EXACTLY as written (keep currency symbols, commas, or wording like "million").
    - county: The county where the project is located (e.g., "Baker", "Volusia"); do not include the word "County".
    - purpose: A brief, faithful description of the project's purpose as described in the announcement (paraphrase allowed but keep meaning).
    - sources: All URLs cited in the answer that can be used to verify this project's details. Include every URL mentioned for the project.
    
    Notes:
    - Only extract projects that are wastewater-, sewer-, septic-, or wastewater treatment/collection-related.
    - If any field is missing from the answer for a project, set it to null (for strings) or [] (for sources).
    - Return up to 10 such projects if present; we'll only evaluate the first three.
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _normalize_str(s: Optional[str]) -> str:
    return (s or "").strip()


def _recipients_distinct(first_three: List[ProjectItem]) -> bool:
    names = [(_normalize_str(p.recipient)).lower() for p in first_three]
    if any(n == "" for n in names):
        return False
    return len(set(names)) == len(names)


def _amount_to_float(amount_text: Optional[str]) -> Optional[float]:
    """
    Best-effort to convert amount strings to numeric dollars.
    Handles cases like:
      - "$5,000,000"
      - "$5 million" / "$5.2 million" (case-insensitive)
      - "5 million" (without $)
    Returns None if not parseable.
    """
    if not amount_text:
        return None
    s = amount_text.lower().strip()

    # Try explicit dollar number with commas
    m = re.search(r"\$?\s*([0-9][0-9,]*)\b", s)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except Exception:
            pass

    # Try "X million"
    m2 = re.search(r"([0-9]+(\.[0-9]+)?)\s*million", s)
    if m2:
        try:
            return float(m2.group(1)) * 1_000_000.0
        except Exception:
            pass

    return None


def _rank_word(idx: int) -> str:
    return {0: "largest", 1: "second-largest", 2: "third-largest"}.get(idx, f"rank #{idx+1}")


# --------------------------------------------------------------------------- #
# Tree construction and verification                                          #
# --------------------------------------------------------------------------- #
async def _build_global_requirements(
    evaluator: Evaluator,
    parent_node,
    picked_projects: List[ProjectItem],
):
    """
    Build and verify global requirements:
      - exactly_three_distinct_projects (adapted to: at least three, first three are distinct)
      - projects_from_specified_announcement (verify each of the first three is from the Jan 7, 2026 announcement/coverage)
    """
    global_node = evaluator.add_parallel(
        id="global_requirements",
        desc="Global answer requirements",
        parent=parent_node,
        critical=True,
    )

    # Exactly three distinct (adapted: at least 3 provided; first three are distinct)
    first_three = picked_projects[:3]
    exactly_three_node = evaluator.add_custom_node(
        result=(len(first_three) == 3 and _recipients_distinct(first_three)),
        id="exactly_three_distinct_projects",
        desc="Provide exactly three distinct projects (no duplicates)",
        parent=global_node,
        critical=True,
    )

    # Projects from specified announcement: split into one leaf per project for robust checking
    from_announce_group = evaluator.add_parallel(
        id="projects_from_specified_announcement",
        desc="All listed projects are from the Jan 7, 2026 DeSantis infrastructure funding announcement described in the prompt",
        parent=global_node,
        critical=True,
    )

    for i, proj in enumerate(first_three):
        leaf = evaluator.add_leaf(
            id=f"project_{i+1}_from_announcement",
            desc=f"Project #{i+1} is part of the January 7, 2026 Florida announcement (or credible coverage of it)",
            parent=from_announce_group,
            critical=True,
        )
        rec = _normalize_str(proj.recipient)
        claim = (
            f"The project for recipient '{rec}' appears in the January 7, 2026 Florida Governor's $167.5 million "
            f"infrastructure funding announcement of 34 projects (or in a credible news report that explicitly covers that announcement)."
        )
        # Provide instruction to strongly require URL evidence
        add_ins = (
            f"{NO_URL_FAIL_INSTRUCTION} Prefer the official Florida Governor's Office/State press release page "
            f"or an official state agency page; credible Florida news outlets explicitly covering the Jan 7, 2026 announcement are acceptable. "
            f"Mark Incorrect if the page is unrelated or does not show this project as part of that announcement."
        )
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=proj.sources if proj.sources else None,
            additional_instruction=add_ins,
        )


async def _verify_single_project(
    evaluator: Evaluator,
    parent_node,
    proj: ProjectItem,
    idx: int,
):
    """
    Verify a single project block according to rubric:
      - selection_correctness_{i}
      - recipient_{i}
      - amount_{i}
      - county_{i}
      - purpose_{i}
      - verifiable_sources_{i}
    """
    proj_node = evaluator.add_parallel(
        id=f"project_{idx+1}",
        desc=(
            "Largest wastewater/sewer treatment project (rank #1 by awarded amount among wastewater/sewer projects in the announcement)"
            if idx == 0 else
            "Second-largest wastewater/sewer treatment project (rank #2 by awarded amount among wastewater/sewer projects in the announcement)"
            if idx == 1 else
            "Third-largest wastewater/sewer treatment project (rank #3 by awarded amount among wastewater/sewer projects in the announcement)"
        ),
        parent=parent_node,
        critical=False,
    )

    # Selection correctness (ranking within wastewater/sewer category)
    sel_leaf = evaluator.add_leaf(
        id=f"selection_correctness_{idx+1}",
        desc=f"Correctly select the #{idx+1} ({_rank_word(idx)}) wastewater/sewer treatment project by awarded funding amount among all wastewater/sewer projects in the announcement",
        parent=proj_node,
        critical=True,
    )
    rec = _normalize_str(proj.recipient)
    amt = _normalize_str(proj.amount)
    claim_sel = (
        f"Among wastewater or sewer treatment infrastructure projects listed in the January 7, 2026 Florida Governor's $167.5M announcement, "
        f"the project for '{rec}' has the {_rank_word(idx)} awarded funding amount."
    )
    add_ins_sel = (
        f"{NO_URL_FAIL_INSTRUCTION} Determine the {_rank_word(idx)} strictly among wastewater/sewer/septic/wastewater "
        f"treatment and collection projects only. Use the press release list of all 34 projects to compare award amounts; "
        f"if any wastewater/sewer project has a higher (for #{idx+1}) or conflicting amount ordering, mark Incorrect."
    )
    await evaluator.verify(
        claim=claim_sel,
        node=sel_leaf,
        sources=proj.sources if proj.sources else None,
        additional_instruction=add_ins_sel,
    )

    # Recipient verification
    rec_leaf = evaluator.add_leaf(
        id=f"recipient_{idx+1}",
        desc="Provide the recipient entity name (city, town, county, or district)",
        parent=proj_node,
        critical=True,
    )
    claim_rec = f"The recipient (grantee) is exactly '{rec}'."
    add_ins_rec = (
        f"{NO_URL_FAIL_INSTRUCTION} Allow minor variations like inclusion of 'City of' or 'Town of' or abbreviations, "
        f"but the entity must clearly refer to the same recipient as in the announcement."
    )
    await evaluator.verify(
        claim=claim_rec,
        node=rec_leaf,
        sources=proj.sources if proj.sources else None,
        additional_instruction=add_ins_rec,
    )

    # Amount verification
    amt_leaf = evaluator.add_leaf(
        id=f"amount_{idx+1}",
        desc="Provide the exact awarded funding amount for the project",
        parent=proj_node,
        critical=True,
    )
    claim_amt = f"The awarded funding amount for this project is '{amt}'."
    add_ins_amt = (
        f"{NO_URL_FAIL_INSTRUCTION} Match the numeric value exactly; treat '$5,000,000' as equivalent to '$5 million' "
        f"only if the numeric value is the same. Minor formatting differences (commas, 'USD') are acceptable."
    )
    await evaluator.verify(
        claim=claim_amt,
        node=amt_leaf,
        sources=proj.sources if proj.sources else None,
        additional_instruction=add_ins_amt,
    )

    # County verification
    county = _normalize_str(proj.county)
    county_leaf = evaluator.add_leaf(
        id=f"county_{idx+1}",
        desc="Provide the county where the project is located",
        parent=proj_node,
        critical=True,
    )
    claim_county = f"The project is located in {county} County, Florida."
    add_ins_county = (
        f"{NO_URL_FAIL_INSTRUCTION} If the page lists a city/town and also specifies its county, ensure the county matches. "
        f"If the county is not explicitly stated but is clearly inferable from the official page, allow it. Otherwise, mark Incorrect."
    )
    await evaluator.verify(
        claim=claim_county,
        node=county_leaf,
        sources=proj.sources if proj.sources else None,
        additional_instruction=add_ins_county,
    )

    # Purpose verification
    purpose = _normalize_str(proj.purpose)
    purpose_leaf = evaluator.add_leaf(
        id=f"purpose_{idx+1}",
        desc="Provide a brief description of the project's stated purpose as described in the official announcement",
        parent=proj_node,
        critical=True,
    )
    claim_purpose = (
        f"The official announcement describes the project's purpose as: \"{purpose}\" "
        f"(paraphrase allowed but meaning must match the announcement's description)."
    )
    add_ins_purpose = (
        f"{NO_URL_FAIL_INSTRUCTION} Accept close paraphrases that preserve the meaning; "
        f"reject if the page's description contradicts or does not support this stated purpose."
    )
    await evaluator.verify(
        claim=claim_purpose,
        node=purpose_leaf,
        sources=proj.sources if proj.sources else None,
        additional_instruction=add_ins_purpose,
    )

    # Verifiable sources sufficiency
    src_leaf = evaluator.add_leaf(
      id=f"verifiable_sources_{idx+1}",
      desc="Provide source citation(s) sufficient to verify the recipient, amount, county, and purpose from official government sources or credible news reports",
      parent=proj_node,
      critical=True,
    )
    claim_src = (
        "The provided URLs include at least one official Florida government page (e.g., Governor's Office or state agency) "
        "or a credible news report that covers the Jan 7, 2026 announcement and contains sufficient detail to verify the "
        "recipient, awarded amount, county/location, and project purpose for this project."
    )
    add_ins_src = (
        f"{NO_URL_FAIL_INSTRUCTION} Prefer the official press release page summarizing all 34 projects; "
        f"credible statewide/local outlets (ABC/CBS/NBC/FOX affiliates, major newspapers) are acceptable if they cover "
        f"the Jan 7, 2026 announcement and include the necessary details for this project."
    )
    await evaluator.verify(
        claim=claim_src,
        node=src_leaf,
        sources=proj.sources if proj.sources else None,
        additional_instruction=add_ins_src,
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
    Build the verification tree and run the evaluation.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root as parallel aggregation per rubric
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

    # Extract projects from the answer
    extracted: ProjectsExtraction = await evaluator.extract(
        prompt=prompt_extract_projects(),
        template_class=ProjectsExtraction,
        extraction_name="projects_extraction",
    )

    # Record custom info about extraction
    total_projects = len(extracted.projects)
    evaluator.add_custom_info(
        info={
            "total_projects_extracted": total_projects,
            "note": "Only the first three projects are evaluated if more are provided.",
        },
        info_type="extraction_stats",
    )

    # Select first three projects (pad with empty items if fewer)
    selected: List[ProjectItem] = list(extracted.projects[:3])
    while len(selected) < 3:
        selected.append(ProjectItem())

    # Global requirements
    await _build_global_requirements(evaluator, root, selected)

    # Project verifications
    for i in range(3):
        await _verify_single_project(evaluator, root, selected[i], i)

    # Return summary
    return evaluator.get_summary()