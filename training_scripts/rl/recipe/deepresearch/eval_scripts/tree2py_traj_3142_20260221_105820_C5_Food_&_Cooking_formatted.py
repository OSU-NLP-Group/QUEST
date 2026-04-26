import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "holiday_restaurants_2024"
TASK_DESCRIPTION = (
    "For travelers who need guaranteed food service options during both major holidays in late 2024, "
    "identify four different national restaurant chains that meet ALL of the following requirements:\n\n"
    "1. Each chain must operate restaurant locations in multiple U.S. states (not regional chains)\n"
    "2. Each chain must be confirmed open on Thanksgiving Day (Thursday, November 28, 2024)\n"
    "3. Each chain must be confirmed open on Christmas Day (Wednesday, December 25, 2024)\n"
    "4. At least two of the four chains must operate on a 24-hour, 7-days-a-week, 365-days-a-year basis (including both holidays)\n"
    "5. For the remaining chains, their holiday operations must be publicly confirmed through official company communications or reliable news sources\n"
    "6. Provide reference URLs confirming each chain's holiday operations\n\n"
    "For each chain, specify: (a) the chain name, (b) confirmation that it operates in multiple states, "
    "(c) confirmation of Thanksgiving 2024 operations, (d) confirmation of Christmas 2024 operations, "
    "(e) whether it operates 24/7/365 or provides documented holiday hours, and (f) reference URL(s) supporting your answer."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ChainEntry(BaseModel):
    """
    Information for a single restaurant chain as extracted from the agent's answer.
    """
    name: Optional[str] = None
    national_presence_urls: List[str] = Field(default_factory=list)
    thanksgiving_urls: List[str] = Field(default_factory=list)
    christmas_urls: List[str] = Field(default_factory=list)
    ops_category: Optional[str] = None  # e.g., "24/7/365" or "documented holiday hours"
    ops_urls: List[str] = Field(default_factory=list)  # URLs confirming 24/7 or holiday hours


class ChainsExtraction(BaseModel):
    """
    Collection of up to four chains extracted from the agent's answer.
    """
    chains: List[ChainEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_chains() -> str:
    return (
        "Extract up to four national restaurant chains from the answer that meet the task requirements. "
        "For each chain, return a JSON object with the following fields:\n"
        "- name: The chain name as stated\n"
        "- national_presence_urls: A list of URLs that help confirm the chain operates in multiple U.S. states "
        "(e.g., store locator pages, 'locations by state' pages, or credible sources evidencing nationwide/multi-state presence).\n"
        "- thanksgiving_urls: A list of URLs that explicitly confirm the chain is open on Thanksgiving Day 2024 (Nov 28, 2024) "
        "or otherwise indicate 24/7/365 operations implying it is open on Thanksgiving.\n"
        "- christmas_urls: A list of URLs that explicitly confirm the chain is open on Christmas Day 2024 (Dec 25, 2024) "
        "or otherwise indicate 24/7/365 operations implying it is open on Christmas.\n"
        "- ops_category: Either '24/7/365' if the chain operates continuously through the year (including holidays), "
        "or 'documented holiday hours' if the chain publishes specific holiday hours via official communications or reliable news sources.\n"
        "- ops_urls: A list of URLs supporting the ops_category (e.g., corporate site stating 24/7 operations, official press releases, "
        "or reliable news articles documenting holiday hours for 2024).\n\n"
        "Return a JSON object with a 'chains' array of these chain objects. "
        "If a field is not mentioned in the answer, set it to null for strings or an empty list for URLs. "
        "Use only URLs explicitly present in the answer text."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_24_7_365(category: Optional[str]) -> bool:
    """
    Determine whether a textual ops category indicates 24/7/365 operations.
    Accept common variants and synonyms.
    """
    if not category:
        return False
    s = category.strip().lower()
    tokens = [
        "24/7/365", "24-7-365", "24x7x365",
        "24 hours a day, 7 days a week, 365 days a year",
        "open 24/7", "open 24 hours", "always open", "open year-round"
    ]
    for t in tokens:
        if t in s:
            return True
    # Heuristic: contains "24", "7", and "365" anywhere
    return ("24" in s and "7" in s and "365" in s)


def _union(*lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        for url in lst:
            u = url.strip()
            if u and u not in seen:
                seen.add(u)
                out.append(u)
    return out


def _normalize_name(n: Optional[str]) -> Optional[str]:
    if not n:
        return None
    return " ".join(n.strip().lower().split())


def _build_thanksgiving_claim(name: Optional[str]) -> str:
    nm = name or "the chain"
    return (
        f"The restaurant chain '{nm}' is open on Thanksgiving Day 2024 (Thursday, November 28, 2024). "
        "If the provided source states the chain operates 24/7/365, that also counts as being open on Thanksgiving."
    )


def _build_christmas_claim(name: Optional[str]) -> str:
    nm = name or "the chain"
    return (
        f"The restaurant chain '{nm}' is open on Christmas Day 2024 (Wednesday, December 25, 2024). "
        "If the provided source states the chain operates 24/7/365, that also counts as being open on Christmas."
    )


def _build_national_presence_claim(name: Optional[str]) -> str:
    nm = name or "the chain"
    return (
        f"The restaurant chain '{nm}' operates locations in multiple U.S. states (i.e., it is not a regional-only chain)."
    )


def _build_ops_category_claim(chain: ChainEntry) -> str:
    nm = chain.name or "the chain"
    if _is_24_7_365(chain.ops_category):
        return (
            f"The restaurant chain '{nm}' operates 24 hours a day, 7 days a week, 365 days a year, "
            "including both Thanksgiving and Christmas."
        )
    else:
        return (
            f"The restaurant chain '{nm}' has publicly documented holiday hours for 2024 (Thanksgiving and/or Christmas), "
            "as confirmed by official company communications or reliable news sources."
        )


def _ops_additional_instruction(chain: ChainEntry) -> str:
    if _is_24_7_365(chain.ops_category):
        return (
            "Confirm that the provided URLs clearly state continuous 24/7/365 operations, or equivalent wording "
            "that reasonably implies always open, including major holidays."
        )
    else:
        return (
            "Confirm that the provided URLs are official company communications (e.g., corporate websites, press releases) "
            "or reliable news sources explicitly documenting holiday hours for 2024 (Thanksgiving and/or Christmas). "
            "Avoid speculative blog posts or informal forums."
        )


HOLIDAY_VERIFY_ADDI = (
    "Use the provided URLs to confirm the statement. If a page clearly states 24/7/365 operations, "
    "that qualifies as being open on both holidays. Otherwise, look for explicit 2024 Thanksgiving or Christmas opening statements or holiday hours. "
    "Prefer official company communications; reliable news sources that quote or cite official statements are acceptable."
)

NATIONAL_PRESENCE_ADDI = (
    "Use the provided URLs to confirm multi-state operations. A store locator page showing locations by state, "
    "a corporate 'locations' page listing multiple states, or a credible source indicating nationwide/multi-state presence suffices."
)


# --------------------------------------------------------------------------- #
# Verification logic per chain                                                #
# --------------------------------------------------------------------------- #
async def verify_chain(
    evaluator: Evaluator,
    parent_node,
    chain: ChainEntry,
    idx: int,
) -> None:
    """
    Build the verification subtree for a single chain and perform the leaf verifications.
    All nodes under the per-chain container are critical to satisfy task requirements.
    """
    chain_node = evaluator.add_parallel(
        id=f"Chain_{idx+1}",
        desc=f"Restaurant chain #{idx+1} meeting all requirements",
        parent=parent_node,
        critical=True
    )

    # Identity & References (critical group)
    id_refs_node = evaluator.add_parallel(
        id=f"Chain_{idx+1}_Identity_And_References",
        desc="Chain is properly identified with supporting URL references",
        parent=chain_node,
        critical=True
    )

    # Named (existence)
    evaluator.add_custom_node(
        result=bool(chain.name and chain.name.strip()),
        id=f"Chain_{idx+1}_Named",
        desc="A specific national restaurant chain is named",
        parent=id_refs_node,
        critical=True
    )

    # References Provided (existence of at least one holiday-related ref)
    holiday_ref_urls = _union(chain.thanksgiving_urls, chain.christmas_urls, chain.ops_urls)
    evaluator.add_custom_node(
        result=len(holiday_ref_urls) > 0,
        id=f"Chain_{idx+1}_References_Provided",
        desc="At least one reference URL is provided confirming the chain's holiday operations",
        parent=id_refs_node,
        critical=True
    )

    # National Presence verification (critical leaf)
    national_leaf = evaluator.add_leaf(
        id=f"Chain_{idx+1}_National_Presence",
        desc="Chain operates locations in multiple U.S. states",
        parent=chain_node,
        critical=True
    )

    # Thanksgiving Open verification (critical leaf)
    tg_leaf = evaluator.add_leaf(
        id=f"Chain_{idx+1}_Thanksgiving_Open",
        desc="Chain is confirmed open on Thanksgiving Day (November 28, 2024)",
        parent=chain_node,
        critical=True
    )

    # Christmas Open verification (critical leaf)
    xmas_leaf = evaluator.add_leaf(
        id=f"Chain_{idx+1}_Christmas_Open",
        desc="Chain is confirmed open on Christmas Day (December 25, 2024)",
        parent=chain_node,
        critical=True
    )

    # Operations Category verification (critical leaf)
    ops_leaf = evaluator.add_leaf(
        id=f"Chain_{idx+1}_Operations_Category",
        desc="Chain's operational category (24/7/365 or documented holiday hours) is specified and supported",
        parent=chain_node,
        critical=True
    )

    # Prepare claims and sources for batch verification
    claims_and_sources = [
        (
            _build_national_presence_claim(chain.name),
            chain.national_presence_urls,
            national_leaf,
            NATIONAL_PRESENCE_ADDI
        ),
        (
            _build_thanksgiving_claim(chain.name),
            _union(chain.thanksgiving_urls, chain.ops_urls),
            tg_leaf,
            HOLIDAY_VERIFY_ADDI
        ),
        (
            _build_christmas_claim(chain.name),
            _union(chain.christmas_urls, chain.ops_urls),
            xmas_leaf,
            HOLIDAY_VERIFY_ADDI
        ),
        (
            _build_ops_category_claim(chain),
            _union(chain.ops_urls, chain.thanksgiving_urls, chain.christmas_urls),
            ops_leaf,
            _ops_additional_instruction(chain)
        ),
    ]

    # Execute verifications in parallel
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate the agent's answer for the holiday restaurant chains task.
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
        default_model=model
    )

    # Create a critical task root to enforce that all subrequirements must pass
    task_root = evaluator.add_parallel(
        id="Task_Root",
        desc="Evaluate whether four qualifying national restaurant chains have been identified with complete information for each, including at least two 24/7/365 operations",
        parent=root,
        critical=True
    )

    # Extract chains from the answer
    extracted: ChainsExtraction = await evaluator.extract(
        prompt=prompt_extract_chains(),
        template_class=ChainsExtraction,
        extraction_name="chains_extraction"
    )

    # Select first four chains; pad if fewer
    selected: List[ChainEntry] = list(extracted.chains[:4])
    while len(selected) < 4:
        selected.append(ChainEntry())

    # Add custom info for debugging
    ops_summary = [
        {
            "name": c.name,
            "ops_category": c.ops_category,
            "is_24_7_365": _is_24_7_365(c.ops_category),
            "num_national_presence_urls": len(c.national_presence_urls),
            "num_thanksgiving_urls": len(c.thanksgiving_urls),
            "num_christmas_urls": len(c.christmas_urls),
            "num_ops_urls": len(c.ops_urls),
        }
        for c in selected
    ]
    evaluator.add_custom_info(
        info={"chains_selected": ops_summary},
        info_type="debug",
        info_name="selected_chains_debug"
    )

    # Build per-chain verification subtrees
    for i, chain in enumerate(selected):
        await verify_chain(evaluator, task_root, chain, i)

    # Global requirement: At least two chains operate 24/7/365
    num_24_7 = sum(1 for c in selected if _is_24_7_365(c.ops_category))
    evaluator.add_custom_node(
        result=(num_24_7 >= 2),
        id="At_Least_Two_24_7_Operations",
        desc="At least two of the four identified chains operate 24 hours a day, 7 days a week, 365 days a year (including both holidays)",
        parent=task_root,
        critical=True
    )

    # Global requirement: All chains are distinct (and all four are named)
    normalized_names = [_normalize_name(c.name) for c in selected]
    valid_names = [n for n in normalized_names if n]
    all_distinct = (len(valid_names) == 4 and len(set(valid_names)) == 4)
    evaluator.add_custom_node(
        result=all_distinct,
        id="All_Chains_Distinct",
        desc="All four identified chains are different from each other (no duplicates)",
        parent=task_root,
        critical=True
    )

    # Add additional summary info
    evaluator.add_custom_info(
        info={"count_24_7_365": num_24_7},
        info_type="metric",
        info_name="24_7_365_count"
    )

    return evaluator.get_summary()