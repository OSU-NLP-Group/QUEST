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
TASK_ID = "cdc_flu_2025_2026_very_high_states"
TASK_DESCRIPTION = """
During the 2025-2026 influenza season in the United States, the CDC tracked state-level flu activity through its FluView surveillance system, classifying states into different activity levels including "very high." Identify four U.S. states that reached the "very high" flu activity level at any point during the 2025-2026 influenza season.

For each of the four states, provide the following information:

1. The name of the state
2. Confirmation that the state reached "very high" flu activity level according to CDC classification
3. At least one specific week during the 2025-2026 season when the state had very high flu activity (provide either the week ending date or CDC week number)
4. The state's flu vaccination coverage rate or percentage for the 2025-2026 season
5. A direct URL to the state's health department webpage that contains influenza information, surveillance data, or flu-related resources
6. A URL to a CDC FluView report, surveillance map, or dashboard page that documents the state's flu activity level

All information must be verifiable through the provided URLs and must pertain specifically to the 2025-2026 influenza season.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StateItem(BaseModel):
    """One state entry extracted from the answer."""
    name: Optional[str] = None
    very_high_week: Optional[str] = None  # accept CDC week label or week-ending date string
    vaccination_coverage: Optional[str] = None  # keep as string (e.g., "45%", "45.2%")
    state_health_dept_url: Optional[str] = None
    cdc_fluview_urls: List[str] = Field(default_factory=list)  # CDC FluView/flu weekly pages showing activity
    coverage_urls: List[str] = Field(default_factory=list)  # URLs supporting vaccination coverage (CDC/state)


class FourStatesExtraction(BaseModel):
    """A list of up to four states extracted from the answer."""
    states: List[StateItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_four_states() -> str:
    return """
    Extract up to four (4) U.S. states from the answer that are claimed to have reached the CDC "very high" flu activity level during the 2025–2026 influenza season.

    For each state, extract the following fields:
    - name: The full state name (e.g., "Texas")
    - very_high_week: A specific week label when the state had "very high" activity (CDC week number or week-ending date string) explicitly mentioned in the answer text. Return null if not provided.
    - vaccination_coverage: The influenza vaccination coverage for the 2025–2026 season as given in the answer (e.g., "45%" or "45.2%"). Return null if not mentioned.
    - state_health_dept_url: A direct URL to the state's official health department (or state government) page that provides influenza/flu information, surveillance data, or flu resources.
    - cdc_fluview_urls: An array of one or more CDC URLs (e.g., CDC FluView weekly report, maps, or dashboards) that document the state's flu activity level for the 2025–2026 season. Only include actual URLs present in the answer. If none are present, return an empty array.
    - coverage_urls: An array of URLs that support the vaccination coverage number for 2025–2026 (these may be CDC or state health department pages). If none are present in the answer, return an empty array.

    Return a JSON object with a 'states' array containing up to 4 such objects. 
    IMPORTANT:
    - Extract strictly from the provided answer text. Do not invent or infer missing values.
    - Only include valid, complete URLs (http/https). If not present, set the field to null (for single URL fields) or [] (for arrays).
    - If the answer lists more than 4 states, keep only the first 4 in the 'states' array.
    - If the answer provides fewer than 4 states, include only those available (the rest can be omitted from the array).
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _gather_sources(primary: Optional[str], extra: List[str]) -> List[str]:
    out = []
    if _non_empty_str(primary):
        out.append(primary)  # type: ignore
    if extra:
        out.extend([u for u in extra if _non_empty_str(u)])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in out:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


# --------------------------------------------------------------------------- #
# Per‑state verification                                                      #
# --------------------------------------------------------------------------- #
async def verify_one_state(
    evaluator: Evaluator,
    parent_node,
    idx: int,
    state: StateItem,
) -> None:
    """
    Build the verification subtree for a single state.
    Mirrors the rubric leaves while adding necessary gating existence checks as separate critical custom nodes.
    """
    disp_idx = idx + 1
    state_node = evaluator.add_parallel(
        id=f"state_{disp_idx}",
        desc=f"State #{disp_idx}: very high flu activity during 2025–2026 with complete documentation",
        parent=parent_node,
        critical=False,
    )

    # 1) State name existence (CRITICAL)
    evaluator.add_custom_node(
        result=_non_empty_str(state.name),
        id=f"state_{disp_idx}_name_exists",
        desc="The name of the state is provided.",
        parent=state_node,
        critical=True,
    )

    # 2) Health department URL presence (CRITICAL gate) + verification (CRITICAL)
    evaluator.add_custom_node(
        result=_non_empty_str(state.state_health_dept_url),
        id=f"state_{disp_idx}_health_dept_url_present",
        desc="A state health department URL is provided.",
        parent=state_node,
        critical=True,
    )

    health_dept_leaf = evaluator.add_leaf(
        id=f"state_{disp_idx}_health_dept_url",
        desc="URL to this state's health department webpage containing flu info or surveillance resources.",
        parent=state_node,
        critical=True,
    )
    health_claim = (
        f"This webpage is an official state (or state health department) site for {state.name}, "
        f"and it contains influenza/flu information, surveillance data, or flu-related resources "
        f"relevant to the public or providers."
    )
    await evaluator.verify(
        claim=health_claim,
        node=health_dept_leaf,
        sources=state.state_health_dept_url,
        additional_instruction=(
            "Confirm the URL is an official state government domain (e.g., .gov/.state.xx.us) or "
            "a recognized state health department site. The page content should explicitly relate "
            "to influenza (flu) information or surveillance (accept synonyms like 'influenza', 'flu', 'ILI'). "
            "General health homepages without influenza content do not qualify."
        ),
    )

    # 3) CDC FluView URL(s) presence (CRITICAL gate) + verification (CRITICAL)
    cdc_sources = [u for u in (state.cdc_fluview_urls or []) if _non_empty_str(u)]
    evaluator.add_custom_node(
        result=len(cdc_sources) > 0,
        id=f"state_{disp_idx}_cdc_urls_present",
        desc="At least one CDC FluView/surveillance URL is provided.",
        parent=state_node,
        critical=True,
    )

    cdc_url_leaf = evaluator.add_leaf(
        id=f"state_{disp_idx}_cdc_fluview_url",
        desc="URL to a CDC FluView report/map/surveillance page that documents this state's activity.",
        parent=state_node,
        critical=True,
    )
    cdc_url_claim = (
        f"This webpage is on CDC (cdc.gov) and is part of CDC's influenza surveillance (e.g., FluView, flu weekly). "
        f"It includes information about {state.name}'s influenza activity for the 2025–2026 season."
    )
    await evaluator.verify(
        claim=cdc_url_claim,
        node=cdc_url_leaf,
        sources=cdc_sources,
        additional_instruction=(
            "Accept CDC FluView Weekly, FluView Interactive, Weekly U.S. Maps, or related CDC influenza surveillance "
            "pages (cdc.gov or gis.cdc.gov). The page should clearly be about influenza surveillance and should be "
            "applicable to the 2025–2026 season, mentioning states (and preferably the target state by name or map)."
        ),
    )

    # 4) Very High activity confirmation (CRITICAL) — depends on CDC URL node
    very_high_leaf = evaluator.add_leaf(
        id=f"state_{disp_idx}_very_high_activity",
        desc="Evidence that this state reached 'very high' flu activity level per CDC during 2025–2026.",
        parent=state_node,
        critical=True,
    )
    very_high_claim = (
        f"According to CDC influenza surveillance for the 2025–2026 season, {state.name} reached a 'very high' "
        f"influenza activity level at least once."
    )
    await evaluator.verify(
        claim=very_high_claim,
        node=very_high_leaf,
        sources=cdc_sources,
        additional_instruction=(
            "Look for 'very high' classification (case-insensitive; synonyms like 'VERY HIGH' are acceptable) "
            "for the specified state at any time in the 2025–2026 season. If the page is a weekly map/report, "
            "any week showing 'very high' counts as confirmation."
        ),
    )

    # 5) Week identification (CRITICAL) — existence + week-level support via CDC URLs
    evaluator.add_custom_node(
        result=_non_empty_str(state.very_high_week),
        id=f"state_{disp_idx}_week_present",
        desc="A specific week label (week number or week-ending date) is provided.",
        parent=state_node,
        critical=True,
    )

    week_leaf = evaluator.add_leaf(
        id=f"state_{disp_idx}_week_identification",
        desc="At least one specific week is identified during which the state had 'very high' activity.",
        parent=state_node,
        critical=True,
    )
    week_claim = (
        f"In the 2025–2026 CDC reporting for week '{state.very_high_week}', {state.name} is shown as having "
        f"'very high' influenza activity."
    )
    await evaluator.verify(
        claim=week_claim,
        node=week_leaf,
        sources=cdc_sources,
        additional_instruction=(
            "Verify that the provided CDC source(s) document the state's 'very high' activity for the stated week. "
            "The week may be presented as a CDC week number (e.g., 'Week 50, 2025') or a week-ending date. "
            "Allow minor formatting variations like 'Week 50 (2025)'. If the page shows a weekly map for that week "
            "and the state is 'very high', count as supported."
        ),
    )

    # 6) Vaccination coverage (CRITICAL) — value existence + supported by coverage URLs (or fallback)
    evaluator.add_custom_node(
        result=_non_empty_str(state.vaccination_coverage),
        id=f"state_{disp_idx}_coverage_present",
        desc="A 2025–2026 influenza vaccination coverage value is provided for the state.",
        parent=state_node,
        critical=True,
    )

    coverage_sources = _gather_sources(state.state_health_dept_url, state.coverage_urls)
    # If there are absolutely no coverage-specific sources, optionally fall back to CDC pages (may or may not help)
    if not coverage_sources and cdc_sources:
        coverage_sources = cdc_sources

    coverage_leaf = evaluator.add_leaf(
        id=f"state_{disp_idx}_vaccination_coverage",
        desc="Flu vaccination coverage rate/percentage for 2025–2026 is accurately cited for this state.",
        parent=state_node,
        critical=True,
    )
    coverage_claim = (
        f"The influenza vaccination coverage for {state.name} in the 2025–2026 season is {state.vaccination_coverage} "
        f"(allowing rounding to the nearest percentage point)."
    )
    await evaluator.verify(
        claim=coverage_claim,
        node=coverage_leaf,
        sources=coverage_sources if coverage_sources else None,
        additional_instruction=(
            "Confirm the numeric vaccination coverage value corresponds to the 2025–2026 season for the given state. "
            "Accept small rounding differences (±1 percentage point) and minor formatting (presence/absence of the percent sign). "
            "Prefer explicit coverage reports (e.g., CDC FluVaxView/state dashboards or state health department pages)."
        ),
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
    Entry point to evaluate an answer for the 'very high' CDC FluView states during 2025–2026 season.
    """
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

    # 1) Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_four_states(),
        template_class=FourStatesExtraction,
        extraction_name="four_states_extraction",
    )

    # Keep only the first four states (pad with empty items if fewer)
    extracted_states = (extraction.states or [])[:4]
    while len(extracted_states) < 4:
        extracted_states.append(StateItem())

    # 2) Build the rubric root node (parallel, non‑critical)
    four_states_node = evaluator.add_parallel(
        id="Four_States_Flu_Analysis",
        desc="Identification and analysis of four U.S. states that experienced very high flu activity during 2025–2026 with documentation.",
        parent=root,
        critical=False,
    )

    # 3) Verify each state
    for i in range(4):
        await verify_one_state(evaluator, four_states_node, i, extracted_states[i])

    # 4) Return the evaluation summary
    return evaluator.get_summary()