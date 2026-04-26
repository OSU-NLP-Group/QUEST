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
TASK_ID = "free_craft_workshops_march_2026"
TASK_DESCRIPTION = (
    "A parent in the United States is planning craft activities for their two children, aged 7 and 10, during March 2026. "
    "They are looking for free DIY craft workshops that: (1) Accommodate children ages 7-10, (2) Are held on either Saturdays "
    "or Sundays in March 2026, (3) Are completely free with no participation cost, (4) Provide all necessary materials and supplies. "
    "Identify two different free DIY craft workshop options available in March 2026 that meet these requirements. For each workshop, specify: "
    "the workshop name/provider, the age range it accommodates, the day of the week it is typically held, confirmation that it is free, whether materials "
    "are provided, and a reference URL supporting this information."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class WorkshopItem(BaseModel):
    name: Optional[str] = None
    provider: Optional[str] = None
    age_range: Optional[str] = None
    day_of_week: Optional[str] = None
    free_statement: Optional[str] = None
    materials_statement: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class WorkshopsExtraction(BaseModel):
    workshops: List[WorkshopItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_workshops() -> str:
    return """
    Extract up to the first two DIY craft workshop options described in the answer that are intended for children. 
    For each workshop, return an object with the following fields:
    - name: The workshop or program name (e.g., "Kids Workshop", "Makebreak", etc.)
    - provider: The organization or provider hosting the workshop (e.g., "The Home Depot", "Michaels", a local library, etc.)
    - age_range: The age range mentioned for participants (e.g., "ages 6–12", "ages 7 and up"). Extract exactly as written in the answer.
    - day_of_week: The day of the week the workshop is held, as mentioned in the answer (e.g., "Saturday", "Sunday"). If multiple days are stated, extract the one most relevant to the March 2026 session(s).
    - free_statement: The statement or phrase indicating that the workshop is free (e.g., "free", "no cost", "complimentary"). Extract exactly as written in the answer.
    - materials_statement: The statement or phrase indicating that materials/supplies are provided (e.g., "materials included", "supplies provided"). Extract exactly as written in the answer.
    - reference_urls: A list of all URLs cited in the answer for this workshop. Include only URLs explicitly present in the answer. 
      If a URL misses protocol, prepend http://. Do not invent or infer URLs.

    Return a JSON object with a top-level field "workshops" which is an array of at most two such objects, following the order the workshops appear in the answer.
    If any field is missing for a given workshop, set it to null (for strings) or an empty list (for URLs).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth"][n] if 0 <= n < 4 else f"{n+1}th"


def _valid_http_urls(urls: List[str]) -> List[str]:
    cleaned = []
    for u in urls:
        if not isinstance(u, str):
            continue
        s = (u or "").strip()
        if not s:
            continue
        if not (s.lower().startswith("http://") or s.lower().startswith("https://")):
            # If the extractor didn't prepend, add http:// for safety
            s = "http://" + s
        cleaned.append(s)
    return cleaned


# --------------------------------------------------------------------------- #
# Verification logic for one workshop                                         #
# --------------------------------------------------------------------------- #
async def verify_workshop(
    evaluator: Evaluator,
    parent_node,
    item: WorkshopItem,
    index: int,
) -> None:
    """
    Build the verification subtree for a single workshop option.
    """
    idx_human = ordinal(index)  # "First" or "Second"
    node = evaluator.add_parallel(
        id=f"Workshop_{index+1}",
        desc=f"{idx_human} workshop option details",
        parent=parent_node,
        critical=False
    )

    # Prepare sources (validated URLs)
    sources_list = _valid_http_urls(item.reference_urls or [])

    # 1) Name & Provider existence (critical)
    has_name_provider = bool(item and item.name and item.name.strip()) and bool(item and item.provider and item.provider.strip())
    evaluator.add_custom_node(
        result=has_name_provider,
        id=f"Workshop_{index+1}_Name_Provider",
        desc="Specify the workshop name and provider/organization",
        parent=node,
        critical=True
    )

    # 2) Reference URL existence (critical)
    ref_exists = len(sources_list) > 0
    ref_node = evaluator.add_custom_node(
        result=ref_exists,
        id=f"Workshop_{index+1}_Reference",
        desc="Provide a valid reference URL supporting the workshop details",
        parent=node,
        critical=True
    )

    # 3) Age requirement accommodates ages 7–10 (critical)
    age_leaf = evaluator.add_leaf(
        id=f"Workshop_{index+1}_Age_Requirement",
        desc="The workshop must accommodate children in the specified age range (ages 7-10)",
        parent=node,
        critical=True
    )
    age_claim = (
        f"This workshop accommodates children aged 7 to 10 (i.e., both age 7 and age 10 are allowed). "
        f"The age guidance stated on the page should include both 7 and 10. "
        f"Published age range in the answer: '{item.age_range or 'N/A'}'."
    )
    await evaluator.verify(
        claim=age_claim,
        node=age_leaf,
        sources=sources_list,
        extra_prerequisites=[ref_node],
        additional_instruction=(
            "Accept broader ranges that clearly include both ages (e.g., 'ages 6–12' or '6 and up'—which covers 7 and 10). "
            "Reject if the range excludes either age (e.g., '8 and up' excludes 7). "
            "Rely on explicit age statements on the provided page(s); do not infer beyond what's written."
        )
    )

    # 4) Cost is completely free (critical)
    cost_leaf = evaluator.add_leaf(
        id=f"Workshop_{index+1}_Cost",
        desc="The workshop must be completely free (no cost for participation)",
        parent=node,
        critical=True
    )
    cost_claim = "This workshop is completely free to attend with no participation fee."
    await evaluator.verify(
        claim=cost_claim,
        node=cost_leaf,
        sources=sources_list,
        extra_prerequisites=[ref_node],
        additional_instruction=(
            "Treat phrases like 'free', 'no cost', 'complimentary', or 'no fee' as equivalent. "
            "If the page indicates any required fee, do not consider it free."
        )
    )

    # 5) Held on Saturday or Sunday in March 2026 (critical)
    sched_leaf = evaluator.add_leaf(
        id=f"Workshop_{index+1}_Schedule",
        desc="The workshop must be held in March 2026 on either Saturday or Sunday",
        parent=node,
        critical=True
    )
    sched_claim = (
        "There is at least one session of this workshop in March 2026 that occurs on a Saturday or Sunday."
    )
    await evaluator.verify(
        claim=sched_claim,
        node=sched_leaf,
        sources=sources_list,
        extra_prerequisites=[ref_node],
        additional_instruction=(
            "Verify that the provided page(s) show an event date in March 2026 that falls on a Saturday or Sunday. "
            "If the page explicitly lists March 2026 dates with day-of-week, use that. "
            "If it states a rule like 'first Saturday of each month' together with a 2026 schedule or March 2026 listing, that is acceptable. "
            "Do not assume without explicit indication that March 2026 is covered."
        )
    )

    # 6) Materials/supplies are provided (critical)
    materials_leaf = evaluator.add_leaf(
        id=f"Workshop_{index+1}_Materials",
        desc="The workshop must provide all necessary craft materials and supplies",
        parent=node,
        critical=True
    )
    materials_claim = "All necessary materials and supplies for the workshop are provided by the host."
    await evaluator.verify(
        claim=materials_claim,
        node=materials_leaf,
        sources=sources_list,
        extra_prerequisites=[ref_node],
        additional_instruction=(
            "Look for phrasing like 'materials included', 'supplies provided', or 'all materials provided (at no cost)'. "
            "If the page says participants must bring their own essential supplies, then this requirement is not met."
        )
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the free DIY craft workshops in March 2026 task.
    """
    # Initialize evaluator with a parallel root as per rubric
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
        default_model=model
    )

    # Record ground truth constraints for context
    evaluator.add_ground_truth({
        "constraints": {
            "ages_required": "7 and 10 must both be accommodated",
            "timeframe": "March 2026",
            "day_of_week": "Saturday or Sunday",
            "cost": "Free (no participation cost)",
            "materials": "All necessary materials/supplies provided",
            "distinct_providers": True
        }
    })

    # Extract workshops information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_workshops(),
        template_class=WorkshopsExtraction,
        extraction_name="workshops_extraction"
    )

    # Ensure exactly two workshop entries for evaluation
    workshops = list(extracted.workshops[:2])
    while len(workshops) < 2:
        workshops.append(WorkshopItem())

    # Build subtrees for each of the two workshops
    await verify_workshop(evaluator, root, workshops[0], 0)
    await verify_workshop(evaluator, root, workshops[1], 1)

    # Final cross-check: different providers (critical at root)
    p1 = (workshops[0].provider or "").strip().lower()
    p2 = (workshops[1].provider or "").strip().lower()
    diff_providers = bool(p1) and bool(p2) and (p1 != p2)
    evaluator.add_custom_node(
        result=diff_providers,
        id="Different_Providers",
        desc="The two workshops must be from different providers/organizations",
        parent=root,
        critical=True
    )

    # Return the evaluation summary
    return evaluator.get_summary()