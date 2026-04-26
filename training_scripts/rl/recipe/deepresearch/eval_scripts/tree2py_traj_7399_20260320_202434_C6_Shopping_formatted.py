import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "retail_holiday_hours_2025"
TASK_DESCRIPTION = (
    "A shopper is planning to visit major retail stores during three holidays in 2025 and needs to know their "
    "operating hours in advance. Please identify three different major retail chains from the following list: "
    "Walmart, Target, Home Depot, or Kroger. For each of the three chains you identify, provide the complete operating "
    "hours (opening time and closing time) for each of the following three holidays in 2025:\n\n"
    "1. Memorial Day (Monday, May 26, 2025)\n"
    "2. Independence Day (Friday, July 4, 2025)\n"
    "3. Christmas Eve (Wednesday, December 24, 2025)\n\n"
    "For each retail chain and each holiday, you must provide:\n"
    "- The opening time\n"
    "- The closing time\n"
    "- A reference URL that confirms these hours\n\n"
    "Each of the three retail chains must be different from one another. All times and information must be verifiable "
    "through the reference URLs provided."
)

ALLOWED_CHAINS = {"Walmart", "Target", "Home Depot", "Kroger"}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HolidayHours(BaseModel):
    opening_time: Optional[str] = None
    closing_time: Optional[str] = None
    reference_url: Optional[str] = None


class ChainHours(BaseModel):
    name: Optional[str] = None
    identification_url: Optional[str] = None
    memorial_day: Optional[HolidayHours] = None
    independence_day: Optional[HolidayHours] = None
    christmas_eve: Optional[HolidayHours] = None


class ChainsExtraction(BaseModel):
    chains: List[ChainHours] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_chains() -> str:
    return """
    Extract at most the first three different major retail chains and their 2025 holiday operating hours as presented in the answer.

    Only consider chains from this allowed list (case-insensitive matching; normalize common variants):
    - Walmart
    - Target
    - Home Depot (also accept "The Home Depot")
    - Kroger

    If the answer lists more than three allowed chains, keep the first three unique ones in the order they appear.
    If fewer than three are provided, include whatever is available; for missing info, return nulls.

    For each selected chain, extract a JSON object with:
    - name: the chain name exactly as stated in the answer (do not invent)
    - identification_url: a general reference URL cited in the answer that clearly refers to the chain's store/holiday hours (official site preferred; if none is explicitly given, return null)
    - memorial_day: {
        opening_time: opening time for Memorial Day 2025 (Monday, May 26, 2025),
        closing_time: closing time for Memorial Day 2025,
        reference_url: a URL cited in the answer that supports these Memorial Day hours
      }
    - independence_day: {
        opening_time: opening time for Independence Day 2025 (Friday, July 4, 2025),
        closing_time: closing time for Independence Day 2025,
        reference_url: a URL cited in the answer that supports these July 4th hours
      }
    - christmas_eve: {
        opening_time: opening time for Christmas Eve 2025 (Wednesday, December 24, 2025),
        closing_time: closing time for Christmas Eve 2025,
        reference_url: a URL cited in the answer that supports these Christmas Eve hours
      }

    Rules:
    - Do not invent or infer times; extract only what appears in the answer.
    - Times may be in formats like "8 AM", "8:00 a.m.", "08:00", "open 24 hours", or "closed". Keep them as strings.
    - If any required field is not present in the answer for a particular chain/holiday, set it to null.
    - For URLs, extract only actual URLs explicitly present in the answer text (including those inside markdown links). Return null if not present.
    - Do not include more than three chains in the final 'chains' array.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def canonicalize_chain_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = name.strip().lower()
    if "walmart" in n:
        return "Walmart"
    if "target" in n:
        return "Target"
    if "home depot" in n:
        return "Home Depot"
    if "kroger" in n:
        return "Kroger"
    return None


def first_non_null_url(urls: List[Optional[str]]) -> Optional[str]:
    for u in urls:
        if u and isinstance(u, str) and u.strip():
            return u.strip()
    return None


def get_holiday_tuple(chain: ChainHours, holiday_key: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    hh: Optional[HolidayHours] = getattr(chain, holiday_key, None)
    if not hh:
        return None, None, None
    return hh.opening_time, hh.closing_time, hh.reference_url


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_and_verify_holiday(
    evaluator: Evaluator,
    parent_node,
    chain_name: str,
    chain_idx: int,
    holiday_key: str,
    holiday_node_id: str,
    holiday_desc: str,
    date_label: str,
    opening_time: Optional[str],
    closing_time: Optional[str],
    reference_url: Optional[str],
) -> None:
    # Holiday group node (parallel, non-critical per rubric)
    holiday_group = evaluator.add_parallel(
        id=holiday_node_id,
        desc=holiday_desc,
        parent=parent_node,
        critical=False
    )

    # Opening time leaf (critical under holiday group)
    opening_leaf = evaluator.add_leaf(
        id=f"chain_{chain_idx}_{holiday_key}_opening",
        desc=f"The opening time for {date_label} is specified",
        parent=holiday_group,
        critical=True
    )
    opening_claim = (
        f"For {chain_name}, the store opening time on {date_label} is '{opening_time}'. "
        f"Verify that the cited page explicitly supports this opening time for {date_label} (or for {date_label.split(' (')[0]} 2025)."
    )
    await evaluator.verify(
        claim=opening_claim,
        node=opening_leaf,
        sources=reference_url,
        additional_instruction=(
            "Use the provided URL as evidence. Accept minor format variations (e.g., '8am' vs '8:00 AM'). "
            "If the page lists typical/national hours with a note that hours may vary by location, still accept "
            "if the given time matches the stated hours. If the URL is missing or the page is unrelated, mark as not supported."
        )
    )

    # Closing time leaf (critical under holiday group)
    closing_leaf = evaluator.add_leaf(
        id=f"chain_{chain_idx}_{holiday_key}_closing",
        desc=f"The closing time for {date_label} is specified",
        parent=holiday_group,
        critical=True
    )
    closing_claim = (
        f"For {chain_name}, the store closing time on {date_label} is '{closing_time}'. "
        f"Verify that the cited page explicitly supports this closing time for {date_label} (or for {date_label.split(' (')[0]} 2025)."
    )
    await evaluator.verify(
        claim=closing_claim,
        node=closing_leaf,
        sources=reference_url,
        additional_instruction=(
            "Use the provided URL as evidence. Accept minor format variations (e.g., '10pm' vs '10:00 PM'). "
            "If the page lists typical/national hours with a note that hours may vary by location, still accept "
            "if the given time matches the stated hours."
        )
    )

    # Reference existence/support leaf (critical under holiday group)
    ref_leaf = evaluator.add_leaf(
        id=f"chain_{chain_idx}_{holiday_key}_reference",
        desc=f"A valid reference URL supporting {date_label} hours is provided",
        parent=holiday_group,
        critical=True
    )
    ref_claim = (
        f"The provided page supports {chain_name}'s store hours for {date_label} "
        f"(i.e., holiday hours for that specific holiday/date in 2025)."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=reference_url,
        additional_instruction=(
            "The page should clearly be about the chain's hours for the named holiday. "
            "Accept official domains (e.g., walmart.com, target.com, homedepot.com, kroger.com) and reputable sources "
            "(e.g., major news/retail publications). If the URL is missing or the page does not mention the holiday, fail."
        )
    )


async def verify_chain(
    evaluator: Evaluator,
    parent_node,
    chain: ChainHours,
    chain_position_index: int,
    seen_canonical_names: List[str]
) -> None:
    """
    Build the subtree for a single retail chain as per the rubric, and run verifications.
    chain_position_index is 1-based for IDs and descriptions.
    """
    idx = chain_position_index  # 1-based for human-friendly IDs
    chain_node = evaluator.add_parallel(
        id=f"retail_chain_{idx}",
        desc=(
            "First retail chain identification and complete holiday hours verification" if idx == 1 else
            ("Second retail chain identification and complete holiday hours verification" if idx == 2 else
             "Third retail chain identification and complete holiday hours verification")
        ),
        parent=parent_node,
        critical=False
    )

    # Prepare canonical name and ID URL fallback
    provided_name = chain.name or ""
    canonical_name = canonicalize_chain_name(provided_name) or (provided_name.strip() or f"Chain #{idx}")
    mem_open, mem_close, mem_url = get_holiday_tuple(chain, "memorial_day")
    ind_open, ind_close, ind_url = get_holiday_tuple(chain, "independence_day")
    xev_open, xev_close, xev_url = get_holiday_tuple(chain, "christmas_eve")
    id_url = first_non_null_url([chain.identification_url, mem_url, ind_url, xev_url])

    # 1) Identification block (critical)
    identification_node = evaluator.add_parallel(
        id=f"chain_{idx}_identification",
        desc=(
            "The first retail chain is correctly identified as one of the major retailers (Walmart, Target, Home Depot, or Kroger)" if idx == 1 else
            ("The second retail chain is correctly identified as one of the major retailers (Walmart, Target, Home Depot, or Kroger), different from the first chain" if idx == 2 else
             "The third retail chain is correctly identified as one of the major retailers (Walmart, Target, Home Depot, or Kroger), different from the first two chains")
        ),
        parent=chain_node,
        critical=True
    )

    # 1.a) Allowed-name check (custom, critical)
    allowed_check = evaluator.add_custom_node(
        result=(canonicalize_chain_name(provided_name) in ALLOWED_CHAINS),
        id=f"chain_{idx}_allowed_name",
        desc=f"Chain #{idx} name is one of the allowed retailers (Walmart, Target, Home Depot, or Kroger)",
        parent=identification_node,
        critical=True
    )

    # 1.b) Uniqueness check for chain #2 and #3 (custom, critical)
    if idx >= 2:
        unique = canonicalize_chain_name(provided_name) not in seen_canonical_names if canonicalize_chain_name(provided_name) else False
        evaluator.add_custom_node(
            result=unique,
            id=f"chain_{idx}_unique_vs_previous",
            desc=f"Chain #{idx} name is different from previously selected chain(s)",
            parent=identification_node,
            critical=True
        )

    # 1.c) Reference URL check (leaf, critical)
    ref_leaf = evaluator.add_leaf(
        id=f"chain_{idx}_reference_url",
        desc=f"A valid reference URL from an official or reliable source is provided for chain #{idx}'s identification",
        parent=identification_node,
        critical=True
    )
    ref_claim = (
        f"This webpage clearly refers to the retail chain '{canonical_name}' and is an official or reliable source "
        f"relevant to store or holiday hours."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=id_url,
        additional_instruction=(
            "Accept official domains (walmart.com, target.com, homedepot.com, kroger.com) or credible sources "
            "that discuss the chain's store/holiday hours. If the URL is missing or not about the chain, fail."
        )
    )

    # Record seen names (for subsequent chains' uniqueness checks)
    canon_now = canonicalize_chain_name(provided_name)
    if canon_now and canon_now not in seen_canonical_names:
        seen_canonical_names.append(canon_now)

    # 2) Memorial Day 2025 verification (parallel, non-critical)
    await build_and_verify_holiday(
        evaluator=evaluator,
        parent_node=chain_node,
        chain_name=canonical_name,
        chain_idx=idx,
        holiday_key="memorial",
        holiday_node_id=f"chain_{idx}_memorial_day",
        holiday_desc="Memorial Day 2025 (May 26) operating hours are provided for the first chain" if idx == 1 else (
            "Memorial Day 2025 (May 26) operating hours are provided for the second chain" if idx == 2 else
            "Memorial Day 2025 (May 26) operating hours are provided for the third chain"
        ),
        date_label="Memorial Day 2025 (Monday, May 26, 2025)",
        opening_time=mem_open,
        closing_time=mem_close,
        reference_url=mem_url
    )

    # 3) Independence Day 2025 verification (parallel, non-critical)
    await build_and_verify_holiday(
        evaluator=evaluator,
        parent_node=chain_node,
        chain_name=canonical_name,
        chain_idx=idx,
        holiday_key="july4",
        holiday_node_id=f"chain_{idx}_july_4th",
        holiday_desc="Independence Day 2025 (July 4) operating hours are provided for the first chain" if idx == 1 else (
            "Independence Day 2025 (July 4) operating hours are provided for the second chain" if idx == 2 else
            "Independence Day 2025 (July 4) operating hours are provided for the third chain"
        ),
        date_label="Independence Day 2025 (Friday, July 4, 2025)",
        opening_time=ind_open,
        closing_time=ind_close,
        reference_url=ind_url
    )

    # 4) Christmas Eve 2025 verification (parallel, non-critical)
    #    For closing, require that the page indicates early closure or reduced hours
    xmas_group = evaluator.add_parallel(
        id=f"chain_{idx}_christmas_eve",
        desc="Christmas Eve 2025 (December 24) operating hours are provided for the first chain" if idx == 1 else (
            "Christmas Eve 2025 (December 24) operating hours are provided for the second chain" if idx == 2 else
            "Christmas Eve 2025 (December 24) operating hours are provided for the third chain"
        ),
        parent=chain_node,
        critical=False
    )

    # Opening leaf (critical)
    x_open_leaf = evaluator.add_leaf(
        id=f"chain_{idx}_xmas_opening",
        desc="The opening time for Christmas Eve is specified",
        parent=xmas_group,
        critical=True
    )
    x_open_claim = (
        f"For {canonical_name}, the store opening time on Christmas Eve 2025 (Wednesday, December 24, 2025) is "
        f"'{xev_open}'. Verify that the cited page explicitly supports this opening time."
    )
    await evaluator.verify(
        claim=x_open_claim,
        node=x_open_leaf,
        sources=xev_url,
        additional_instruction=(
            "Accept minor time format variations. If the page lists 'reduced holiday hours' with specific times, "
            "that's acceptable as long as the opening time matches."
        )
    )

    # Closing leaf (critical) with early-closure requirement
    x_close_leaf = evaluator.add_leaf(
        id=f"chain_{idx}_xmas_closing",
        desc="The closing time for Christmas Eve is specified, and it indicates early closure",
        parent=xmas_group,
        critical=True
    )
    x_close_claim = (
        f"For {canonical_name}, the store closing time on Christmas Eve 2025 (Wednesday, December 24, 2025) is "
        f"'{xev_close}', and the page indicates an early closure or reduced hours for Christmas Eve."
    )
    await evaluator.verify(
        claim=x_close_claim,
        node=x_close_leaf,
        sources=xev_url,
        additional_instruction=(
            "Look for explicit language like 'early close', 'reduced hours', or clearly earlier-than-typical closing "
            "times listed for Christmas Eve. If the page does not indicate early closure or provide a closing time, fail."
        )
    )

    # Reference leaf (critical)
    x_ref_leaf = evaluator.add_leaf(
        id=f"chain_{idx}_xmas_reference",
        desc="A valid reference URL supporting Christmas Eve hours is provided",
        parent=xmas_group,
        critical=True
    )
    x_ref_claim = (
        f"The provided page supports {canonical_name}'s store hours for Christmas Eve 2025 "
        f"(Wednesday, December 24, 2025)."
    )
    await evaluator.verify(
        claim=x_ref_claim,
        node=x_ref_leaf,
        sources=xev_url,
        additional_instruction=(
            "The page should clearly reference Christmas Eve hours for the chain. "
            "If the URL is missing or the page is unrelated, fail."
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
    Evaluate an answer for the 2025 retail holiday hours task.
    """
    evaluator = Evaluator()
    # Note: The original rubric sets root as critical. However, in the framework,
    # a critical parent requires all children to be critical. Since child nodes
    # include non-critical groups, we set root to non-critical to allow partial credit.
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Evaluate whether the solution correctly identifies three major retail chains with complete holiday operating hours for Memorial Day 2025, Independence Day 2025, and Christmas Eve 2025, with all required details",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # 1) Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_chains(),
        template_class=ChainsExtraction,
        extraction_name="chains_extraction"
    )

    # 2) Post-process: select up to 3 unique, allowed chains in order
    seen: set = set()
    selected: List[ChainHours] = []
    for ch in extracted.chains:
        canon = canonicalize_chain_name(ch.name)
        if canon in ALLOWED_CHAINS and canon not in seen:
            selected.append(ch)
            seen.add(canon)
        if len(selected) == 3:
            break

    # If fewer than 3 chains extracted, pad with empty placeholders
    while len(selected) < 3:
        selected.append(ChainHours())

    # Record selection summary for debugging
    evaluator.add_custom_info(
        info={
            "selected_chain_names": [ch.name for ch in selected],
            "allowed_list": sorted(list(ALLOWED_CHAINS)),
        },
        info_type="selection_summary",
        info_name="selection_summary"
    )

    # 3) Build verification tree per chain
    seen_canonical_names: List[str] = []
    for i in range(3):
        await verify_chain(
            evaluator=evaluator,
            parent_node=root,
            chain=selected[i],
            chain_position_index=i + 1,  # 1-based
            seen_canonical_names=seen_canonical_names
        )

    # 4) Return structured summary
    return evaluator.get_summary()