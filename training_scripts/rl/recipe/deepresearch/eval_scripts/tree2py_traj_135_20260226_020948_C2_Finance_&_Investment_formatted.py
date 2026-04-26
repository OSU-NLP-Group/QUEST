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
TASK_ID = "steyer_powell_transitions_2026"
TASK_DESCRIPTION = (
    "For both Tom Steyer and Dina Powell McCormick, identify the investment firm each person left most recently "
    "before assuming their current position in 2026. For each identified firm, provide its headquarters location."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PersonFirmInfo(BaseModel):
    firm_name: Optional[str] = None
    headquarters: Optional[str] = None
    sources_for_firm: List[str] = Field(default_factory=list)
    sources_for_headquarters: List[str] = Field(default_factory=list)


class TransitionExtraction(BaseModel):
    tom: Optional[PersonFirmInfo] = None
    dina: Optional[PersonFirmInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_transitions() -> str:
    return """
    Extract, from the answer text, the most recent investment firm that each person left before assuming their current position in 2026, along with the firm's headquarters location. You must extract exactly and only what is stated in the answer.

    Persons:
    - Tom Steyer
    - Dina Powell McCormick

    For each person, extract:
    - firm_name: The name of the investment firm they left most recently before assuming their current 2026 role (as stated in the answer).
    - headquarters: The headquarters location for that firm as stated in the answer (city or city + state/country; if multiple cities are listed, include the full string).
    - sources_for_firm: All URLs cited in the answer that support which firm they left most recently (include links that explicitly discuss their departure or that they were formerly at that firm).
    - sources_for_headquarters: All URLs cited in the answer that support the firm's headquarters location.

    Return a JSON object with the following structure:
    {
      "tom": {
        "firm_name": <string or null>,
        "headquarters": <string or null>,
        "sources_for_firm": [<url>, ...],
        "sources_for_headquarters": [<url>, ...]
      },
      "dina": {
        "firm_name": <string or null>,
        "headquarters": <string or null>,
        "sources_for_firm": [<url>, ...],
        "sources_for_headquarters": [<url>, ...]
      }
    }

    Rules:
    - Do not invent any information. If a field is missing in the answer, return null for that field.
    - Extract only URLs explicitly present in the answer. Do not infer or create new URLs.
    - Always return arrays for the sources fields (possibly empty arrays if no URLs are present in the answer).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for u in urls:
        u = (u or "").strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def _verify_person_branch(
    evaluator: Evaluator,
    parent_node,
    person_key: str,             # "tom" or "dina"
    person_full_name: str,       # "Tom Steyer" or "Dina Powell McCormick"
    info: Optional[PersonFirmInfo],
) -> None:
    """
    Build verification subtree for one person and run checks.
    Tree structure under the critical parent 'career_transition_comparison':
      - {person_key}_firm_details (critical, parallel)
        - {person_key}_firm_group (critical, parallel)
          - {person_key}_firm_info_exists (critical, custom)
          - {person_key}_firm_name (critical, leaf, URL-verified)
        - {person_key}_hq_group (critical, parallel)
          - {person_key}_hq_info_exists (critical, custom)
          - {person_key}_firm_headquarters (critical, leaf, URL-verified)
    """

    # Person-level node (must be critical because its parent is critical)
    person_node = evaluator.add_parallel(
        id=f"{person_key}_firm_details",
        desc=f"Identify the firm {person_full_name} left most recently before 2026 and its headquarters",
        parent=parent_node,
        critical=True,
    )

    # Normalize inputs
    firm_name = (info.firm_name if info else None) or ""
    hq_text = (info.headquarters if info else None) or ""
    firm_sources = _dedup_urls(info.sources_for_firm if info else [])
    hq_sources = _dedup_urls(info.sources_for_headquarters if info else [])

    # -------- Firm group (isolate existence gating for firm-only checks) --------
    firm_group = evaluator.add_parallel(
        id=f"{person_key}_firm_group",
        desc=f"{person_full_name} firm identification group",
        parent=person_node,
        critical=True,
    )

    firm_exists = bool(firm_name.strip()) and len(firm_sources) > 0
    evaluator.add_custom_node(
        result=firm_exists,
        id=f"{person_key}_firm_info_exists",
        desc=f"Firm name and at least one supporting source are provided for {person_full_name}",
        parent=firm_group,
        critical=True,
    )

    # Leaf: verify firm name claim with URLs
    firm_leaf_desc = (
        "The name of the investment firm "
        f"{person_full_name} left most recently before assuming their current role in 2026"
    )
    firm_leaf = evaluator.add_leaf(
        id=f"{person_key}_firm_name",
        desc=firm_leaf_desc,
        parent=firm_group,
        critical=True,
    )

    firm_claim = (
        f"According to the provided sources, {person_full_name} left the investment firm '{firm_name}' "
        f"most recently before assuming their current position in 2026."
    )

    await evaluator.verify(
        claim=firm_claim,
        node=firm_leaf,
        sources=firm_sources if firm_sources else None,
        additional_instruction=(
            "Determine whether the page indicates that the person left (departed, stepped down from, resigned from) "
            "the specified investment firm and that this departure corresponds to the most recent move before their 2026 position. "
            "Evidence may be explicit (e.g., 'left in 2025 before being appointed in 2026') or implicit on a 2026 appointment page "
            "stating they were 'formerly at' the named firm just prior to the 2026 role. If the page lists multiple prior firms, "
            "ensure the named one is the immediate prior firm relative to the 2026 role. If the URL is irrelevant or does not support "
            "the claim, judge as not supported."
        ),
    )

    # -------- Headquarters group (isolate existence gating for HQ-only checks) --------
    hq_group = evaluator.add_parallel(
        id=f"{person_key}_hq_group",
        desc=f"{person_full_name} firm headquarters group",
        parent=person_node,
        critical=True,
    )

    hq_exists = bool(hq_text.strip()) and len(hq_sources) > 0
    evaluator.add_custom_node(
        result=hq_exists,
        id=f"{person_key}_hq_info_exists",
        desc=f"Headquarters location and at least one supporting source are provided for {person_full_name}'s firm",
        parent=hq_group,
        critical=True,
    )

    # Leaf: verify HQ with URLs
    if person_key == "tom":
        hq_leaf_desc = "The headquarters city of Tom Steyer's most recent previous firm"
    else:
        hq_leaf_desc = "The headquarters city (or cities) of Dina Powell McCormick's most recent previous firm"

    hq_leaf = evaluator.add_leaf(
        id=f"{person_key}_firm_headquarters",
        desc=hq_leaf_desc,
        parent=hq_group,
        critical=True,
    )

    hq_claim = (
        f"The headquarters location(s) of the firm '{firm_name}' is/are '{hq_text}'."
    )

    await evaluator.verify(
        claim=hq_claim,
        node=hq_leaf,
        sources=hq_sources if hq_sources else None,
        additional_instruction=(
            "Verify that the page supports the stated headquarters location(s) for the firm. "
            "Accept reasonable formatting variants (e.g., 'New York, NY' vs 'New York City, New York'). "
            "If the company lists multiple headquarters or a global HQ plus other offices, "
            "consider the claim supported if the stated location(s) match what the page explicitly lists as headquarters."
        ),
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
    Evaluate an answer for the Steyer/Powell 2026 transition task.
    """
    # 1) Initialize evaluator (root is non-critical by framework design)
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

    # 2) Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_transitions(),
        template_class=TransitionExtraction,
        extraction_name="transition_extraction",
    )

    # 3) Build verification tree following rubric
    comparison_node = evaluator.add_parallel(
        id="career_transition_comparison",
        desc="Analyze and compare the most recent firm transitions for Tom Steyer and Dina Powell McCormick before their 2026 career changes",
        parent=root,
        critical=True,  # Root rubric node is critical; all its children must also be critical
    )

    # 3a) Tom Steyer branch
    await _verify_person_branch(
        evaluator=evaluator,
        parent_node=comparison_node,
        person_key="tom",
        person_full_name="Tom Steyer",
        info=extracted.tom if extracted and extracted.tom else None,
    )

    # 3b) Dina Powell McCormick branch
    await _verify_person_branch(
        evaluator=evaluator,
        parent_node=comparison_node,
        person_key="dina",
        person_full_name="Dina Powell McCormick",
        info=extracted.dina if extracted and extracted.dina else None,
    )

    # 4) Return structured summary
    return evaluator.get_summary()