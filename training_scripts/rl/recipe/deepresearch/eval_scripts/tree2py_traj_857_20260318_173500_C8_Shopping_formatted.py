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
TASK_ID = "retail_blackfriday_xmas_membership_2024"
TASK_DESCRIPTION = """
Identify 4 major U.S. retail chains that opened their stores at 6:00 AM on Black Friday 2024 and closed at 6:00 PM local time on Christmas Eve 2024. For each chain, provide: (1) The name of the retail chain, (2) The name of their paid annual membership program, (3) The annual cost of that membership program in dollars, (4) The duration of the free trial period in days, (5) The U.S. state with the most locations for that chain, and (6) A reference URL supporting this information. All four chains must offer a paid membership program with a free trial period, and they must meet both the Black Friday opening time (6:00 AM) and Christmas Eve closing time (6:00 PM) criteria.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RetailerItem(BaseModel):
    """One retailer's extracted info from the agent's answer."""
    chain_name: Optional[str] = None

    # Membership info (paid annual with a free trial is required by task)
    membership_name: Optional[str] = None
    membership_annual_cost: Optional[str] = None  # Keep string for flexibility, e.g., "$98", "98", "98.00"
    free_trial_days: Optional[str] = None         # Keep string to handle "30", "30 days", or "4 weeks (~28 days)"

    # Holiday hours (as presented in answer; optional text, used for context)
    black_friday_open_time: Optional[str] = None  # e.g., "6:00 AM"
    christmas_eve_close_time: Optional[str] = None  # e.g., "6:00 PM"

    # Geography
    top_state: Optional[str] = None  # U.S. state with the most locations

    # Evidence URLs
    schedule_urls: List[str] = Field(default_factory=list)     # Holiday hours: Black Friday / Christmas Eve
    membership_urls: List[str] = Field(default_factory=list)   # Membership name/cost/trial
    locations_urls: List[str] = Field(default_factory=list)    # State with the most locations
    reference_url: Optional[str] = None                        # A general reference URL if provided


class RetailersExtraction(BaseModel):
    """Container for all extracted retailers."""
    retailers: List[RetailerItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_retailers() -> str:
    return """
    Extract up to four (4) retail chains from the answer that the author claims meet ALL of these criteria:
    - They are major U.S. retailers with many physical stores nationwide.
    - Their stores opened at 6:00 AM on Black Friday 2024.
    - Their stores closed at 6:00 PM local time on Christmas Eve 2024.
    - They offer a PAID ANNUAL membership program and that program includes a FREE TRIAL period.

    For each retailer found in the answer, extract the following fields exactly as they appear in the answer:
    - chain_name: The name of the retail chain.
    - membership_name: The name of the chain's paid annual membership program (e.g., "Walmart+", "Total", "Circle 360").
    - membership_annual_cost: The ANNUAL price (in USD) of the membership program, as presented (e.g., "$98", "98", "98/year").
    - free_trial_days: The duration of the free trial in days, as presented (e.g., "30", "30 days", "4 weeks (~28 days)").
    - black_friday_open_time: The opening time the answer claims for Black Friday 2024 (e.g., "6:00 AM", "6 am"). If not explicitly stated in the answer, return null.
    - christmas_eve_close_time: The closing time the answer claims for Christmas Eve 2024 (e.g., "6:00 PM", "6 pm"). If not explicitly stated, return null.
    - top_state: The U.S. state with the most store locations for this chain (e.g., "California"). If not in the answer, return null.
    - schedule_urls: All URLs in the answer that support 2024 holiday hours (Black Friday opening and/or Christmas Eve closing). Return an array of raw URLs.
    - membership_urls: All URLs in the answer that support the membership program name, the ANNUAL cost, and free trial. Return an array of raw URLs.
    - locations_urls: All URLs in the answer that support the "state with the most locations" info. Return an array of raw URLs.
    - reference_url: If the answer provides a single "reference" link for this retailer, return it here; otherwise null.

    Rules:
    - Do NOT invent data. If any field is missing in the answer, return null for that field or [] for URL arrays.
    - Return the URLs exactly as present in the answer (raw URLs). If a URL is present without protocol, prepend "http://".
    - If the answer lists more than four retailers, extract the first four in the order they appear.
    - If the answer lists fewer than four, extract all it lists; do not fabricate the rest.

    Return a JSON object with a single field:
    {
      "retailers": [ ... up to 4 RetailerItem objects ... ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        # Basic normalization to avoid trivial dups
        low = u.lower()
        if not (low.startswith("http://") or low.startswith("https://")):
            # Keep as-is; Extractor may already add protocol, but we won't drop if missing
            pass
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def union_sources(*lists: List[str], tail: Optional[str] = None) -> List[str]:
    urls: List[str] = []
    for lst in lists:
        urls.extend(lst or [])
    if tail:
        urls.append(tail)
    return _dedup_urls(urls)


def valid_url(u: Optional[str]) -> bool:
    if not u:
        return False
    s = u.strip().lower()
    return s.startswith("http://") or s.startswith("https://")


# --------------------------------------------------------------------------- #
# Verification for a single retailer                                          #
# --------------------------------------------------------------------------- #
async def verify_retailer(
    evaluator: Evaluator,
    parent_node,
    item: RetailerItem,
    idx_one_based: int,
) -> None:
    """
    Build and verify the subtree for a single retailer (critical node).
    """

    chain = item.chain_name or f"Retailer #{idx_one_based}"
    mem_name = item.membership_name or "the chain's paid annual membership"
    cost_str = item.membership_annual_cost or "[unknown]"
    trial_str = item.free_trial_days or "[unknown]"
    state_str = item.top_state or "[unknown]"

    # Choose sources per aspect; fallback to union if the specific bucket is empty
    all_sources = union_sources(item.schedule_urls, item.membership_urls, item.locations_urls, tail=item.reference_url)
    schedule_sources = _dedup_urls(item.schedule_urls if item.schedule_urls else all_sources)
    membership_sources = _dedup_urls(item.membership_urls if item.membership_urls else all_sources)
    locations_sources = _dedup_urls(item.locations_urls if item.locations_urls else all_sources)

    # Critical retailer node (due to global requirement that all 4 must pass)
    retailer_node = evaluator.add_parallel(
        id=f"retailer_{idx_one_based}",
        desc=f"{['First','Second','Third','Fourth'][idx_one_based-1] if 1 <= idx_one_based <= 4 else f'#{idx_one_based}'} qualifying retail chain with all required attributes",
        parent=parent_node,
        critical=True,
    )

    # 1) Major U.S. retailer (with physical stores)
    major_leaf = evaluator.add_leaf(
        id=f"retailer_{idx_one_based}_is_major_retailer",
        desc="Chain is a major U.S. national retailer with physical store locations",
        parent=retailer_node,
        critical=True,
    )
    major_claim = (
        f"{chain} is a major U.S. national retailer with many physical store locations across multiple U.S. states."
    )
    major_ins = (
        "Use the provided page(s) to confirm a national brick-and-mortar presence (e.g., store locator, about page, "
        "Wikipedia summary). This is correct only if it clearly indicates the company operates many physical U.S. stores."
    )

    # 2) Black Friday 2024 opening time = 6:00 AM
    bf_leaf = evaluator.add_leaf(
        id=f"retailer_{idx_one_based}_black_friday_opening",
        desc="Chain opened stores at 6:00 AM on Black Friday 2024",
        parent=retailer_node,
        critical=True,
    )
    bf_claim = f"On Black Friday 2024, {chain} stores opened at 6:00 AM local time."
    bf_ins = (
        "Verify 2024 Black Friday store opening time specifically. Accept reasonable variants for 6:00 AM such as "
        "'6 am', '6 a.m.', or '6AM'. If the page only mentions other years or does not specify 6:00 AM for 2024, mark incorrect. "
        "If the page states 'most stores open at 6 am' for 2024, consider that acceptable."
    )

    # 3) Christmas Eve 2024 closing time = 6:00 PM
    xmas_leaf = evaluator.add_leaf(
        id=f"retailer_{idx_one_based}_christmas_eve",
        desc="Chain closed stores at 6:00 PM local time on Christmas Eve 2024",
        parent=retailer_node,
        critical=True,
    )
    xmas_claim = f"On Christmas Eve 2024, {chain} stores closed at 6:00 PM local time."
    xmas_ins = (
        "Verify 2024 Christmas Eve closing time specifically. Accept reasonable variants for 6:00 PM such as "
        "'6 pm', '6 p.m.', or '6PM'. If the page only mentions other years or does not specify 6:00 PM for 2024, mark incorrect."
    )

    # 4) Membership program name (paid annual)
    mem_name_leaf = evaluator.add_leaf(
        id=f"retailer_{idx_one_based}_membership_name",
        desc="Provide the name of the chain's paid annual membership program",
        parent=retailer_node,
        critical=True,
    )
    mem_name_claim = f"The paid annual membership program of {chain} is called '{mem_name}'."
    mem_name_ins = (
        "Confirm the membership is paid and annual (yearly). The page should refer to the official membership name."
    )

    # 5) Annual membership cost (USD)
    mem_cost_leaf = evaluator.add_leaf(
        id=f"retailer_{idx_one_based}_membership_cost",
        desc="Provide the annual membership cost in dollars",
        parent=retailer_node,
        critical=True,
    )
    mem_cost_claim = f"The annual cost of the {mem_name} membership is {cost_str} per year."
    mem_cost_ins = (
        "Check that this is the annual (yearly) price, not monthly. If the page shows multiple tiers, "
        "ensure the claimed annual cost matches one of the official annual options."
    )

    # 6) Free trial duration (days)
    trial_leaf = evaluator.add_leaf(
        id=f"retailer_{idx_one_based}_trial_period",
        desc="Specify the free trial period duration in days",
        parent=retailer_node,
        critical=True,
    )
    trial_claim = f"The free trial period for the {mem_name} membership is {trial_str} days."
    trial_ins = (
        "Confirm there is a free trial and its duration in days. If the page uses weeks, convert mentally "
        "to days for reasonableness (e.g., 4 weeks ≈ 28 days) and mark incorrect if the claim meaningfully disagrees."
    )

    # 7) State with the most locations
    state_leaf = evaluator.add_leaf(
        id=f"retailer_{idx_one_based}_state",
        desc="Identify the U.S. state with the most locations for this chain",
        parent=retailer_node,
        critical=True,
    )
    state_claim = f"{state_str} is the U.S. state with the most {chain} store locations."
    state_ins = (
        "Use the provided evidence (e.g., counts by state, store locator statistics). "
        "If there's a tie for the most locations and the claimed state is one of the tied states, consider it acceptable. "
        "If the page doesn't provide enough evidence or shows a different top state, mark incorrect."
    )

    # 8) Reference URL existence (must be provided)
    ref_ok = valid_url(item.reference_url)
    evaluator.add_custom_node(
        result=ref_ok,
        id=f"retailer_{idx_one_based}_reference",
        desc="Provide reference URL supporting the information",
        parent=retailer_node,
        critical=True,
    )

    # Execute the 7 URL-grounded verifications (the reference existence is already handled by custom node)
    claims_and_sources = [
        (major_claim, all_sources, major_leaf, major_ins),
        (bf_claim, schedule_sources, bf_leaf, bf_ins),
        (xmas_claim, schedule_sources, xmas_leaf, xmas_ins),
        (mem_name_claim, membership_sources, mem_name_leaf, mem_name_ins),
        (mem_cost_claim, membership_sources, mem_cost_leaf, mem_cost_ins),
        (trial_claim, membership_sources, trial_leaf, trial_ins),
        (state_claim, locations_sources, state_leaf, state_ins),
    ]
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an agent's answer for: 4 major U.S. retail chains with specified holiday hours and membership details.
    The evaluation is structured so that ALL four retailers must fully pass to achieve a pass at the critical node.
    """
    # Initialize evaluator
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

    # Extract structured retailers info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_retailers(),
        template_class=RetailersExtraction,
        extraction_name="retailers_extraction",
    )

    # Enforce exactly four slots (pad with empty items if fewer; cap if more)
    retailers: List[RetailerItem] = list(extracted.retailers or [])[:4]
    while len(retailers) < 4:
        retailers.append(RetailerItem())

    # Create a critical aggregator requiring all four retailer nodes to pass
    all_retailers = evaluator.add_parallel(
        id="all_retailers",
        desc="All 4 retailers must meet: major U.S. chain, 6:00 AM (Black Friday 2024), 6:00 PM (Christmas Eve 2024), and paid annual membership with free trial, plus state-with-most-locations and a reference URL.",
        parent=root,
        critical=True,
    )

    # Build and verify each retailer subtree (all critical)
    for i, item in enumerate(retailers, start=1):
        await verify_retailer(evaluator, all_retailers, item, i)

    # Optionally record a compact summary of extracted chain names
    evaluator.add_custom_info(
        info={"chains": [r.chain_name for r in retailers]},
        info_type="extracted_overview",
        info_name="retailer_names_overview",
    )

    # Return the evaluation summary
    return evaluator.get_summary()