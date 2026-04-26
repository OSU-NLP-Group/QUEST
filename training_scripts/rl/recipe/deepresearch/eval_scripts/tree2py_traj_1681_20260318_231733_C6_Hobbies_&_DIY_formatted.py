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
TASK_ID = "ohio_craft_diy_holiday_hours_2025_2026"
TASK_DESCRIPTION = """
I'm planning several DIY craft projects during the 2025-2026 holiday season in Ohio. I need to identify four distinct craft, hobby, or DIY supply store chains that have physical locations in Ohio and will be open during key shopping days with the following specific requirements:

1. Each store must be open on New Year's Day 2026 (Thursday, January 1, 2026) and must open at or before 9:30 AM local time
2. Each store must be closed all day on Christmas Day 2025 (Thursday, December 25, 2025)
3. Each store must be open on Black Friday 2025 (Friday, November 28, 2025) and must open at or before 8:00 AM local time

For each of the four stores, please provide:
- The store chain name
- Confirmation that it has at least one physical retail location in Ohio
- The store's specific opening hours on New Year's Day 2026
- Confirmation that the store is closed on Christmas Day 2025
- The store's specific opening hours on Black Friday 2025
- A direct URL to an official source (corporate website, official press release, or official store locator page) that confirms the 2025-2026 holiday schedule for each of these dates

The four stores must be distinct national or regional chains (not different locations of the same chain). Each store's information must be from official 2025-2026 holiday schedules, not schedules from previous years.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StoreEntry(BaseModel):
    """Single store chain entry extracted from the answer."""
    name: Optional[str] = None

    # Official URLs
    type_url: Optional[str] = None                       # Official page supporting store category/type (about page, etc.)
    ohio_presence_url: Optional[str] = None              # Official store locator/state page showing OH location(s)

    # New Year's Day 2026 specifics
    new_years_hours: Optional[str] = None                # Free text such as "Open 9am–6pm"
    new_years_open_time: Optional[str] = None            # Opening time string (e.g., "9:00 AM")
    new_years_url: Optional[str] = None                  # Official URL for NYD 2026 / 2025-2026 holiday page

    # Christmas Day 2025 specifics
    christmas_closed_note: Optional[str] = None          # Free text such as "Closed all day"
    christmas_url: Optional[str] = None                  # Official URL confirming Christmas Day 2025 closure

    # Black Friday 2025 specifics
    black_friday_hours: Optional[str] = None             # Free text such as "Open 6 AM – 10 PM"
    black_friday_open_time: Optional[str] = None         # Opening time string (e.g., "6:00 AM")
    black_friday_url: Optional[str] = None               # Official URL for BF 2025 hours


class StoresExtraction(BaseModel):
    """Extraction container for up to N store chains from the answer."""
    stores: List[StoreEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stores() -> str:
    return """
    Extract information for each craft, hobby, or DIY supply store chain mentioned in the answer.
    Treat "DIY supply" broadly (e.g., craft retailers like Michaels/JOANN/Hobby Lobby, hobby/arts suppliers,
    and home-improvement/hardware retailers like Home Depot/Lowe's/Ace/Harbor Freight).

    For each chain, extract the following fields (use null for anything missing):
    - name: The chain's name
    - type_url: A direct official URL (corporate/brand domain) that supports that the chain is a craft/hobby/DIY supply retailer operating in the U.S. (e.g., About page, departments page)
    - ohio_presence_url: A direct official URL (store locator, list of locations, or brand page) that shows at least one physical store in Ohio
    - new_years_hours: The specific hours text for New Year's Day 2026 (Jan 1, 2026) as written in the answer
    - new_years_open_time: The opening time on New Year's Day 2026 (e.g., "9:00 AM") if provided; otherwise null
    - new_years_url: A direct official URL that confirms the 2025-2026 holiday schedule and/or New Year's Day 2026 hours
    - christmas_closed_note: The closure statement for Christmas Day 2025 (e.g., "Closed all day"), as written in the answer
    - christmas_url: A direct official URL that confirms Christmas Day 2025 closure (prefer 2025-2026 holiday schedule pages)
    - black_friday_hours: The specific hours text for Black Friday 2025 (Nov 28, 2025) as written in the answer
    - black_friday_open_time: The opening time on Black Friday 2025 (e.g., "6:00 AM") if provided; otherwise null
    - black_friday_url: A direct official URL that confirms Black Friday 2025 hours (prefer 2025-2026 holiday schedule pages)

    Notes:
    - "Official URL" means the corporate site, an official press/news release, or an official store locator page.
      Do not use third-party sites (e.g., Yelp, Reddit, blogs, news aggregators).
    - If a single official holiday page covers multiple dates, copy that same URL into the relevant URL fields.
    - Always include full URLs with protocol (https://).
    - Extract all stores the answer mentions; the evaluator will only use up to the first four distinct chains.

    Return JSON with a top-level field "stores": an array of objects with the fields above.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _uniq_urls(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _holiday_sources_for(store: StoreEntry) -> List[str]:
    """Collect all potential official sources provided for any holiday date."""
    return _uniq_urls([
        store.new_years_url,
        store.christmas_url,
        store.black_friday_url,
        store.ohio_presence_url,
        store.type_url,
    ])


# --------------------------------------------------------------------------- #
# Verification for a single store                                             #
# --------------------------------------------------------------------------- #
async def verify_store(evaluator: Evaluator, parent_node, store: StoreEntry, store_index: int) -> None:
    idx_human = store_index + 1
    store_name = store.name or f"Store #{idx_human}"

    # Parent node for this store (non-critical to allow partial across stores)
    store_node = evaluator.add_parallel(
        id=f"store_{store_index}",
        desc=f"Identification and verification of the {['first','second','third','fourth'][store_index] if store_index < 4 else f'{idx_human}th'} qualifying store chain",
        parent=parent_node,
        critical=False
    )

    # ---------------- Store Qualification (critical) ---------------- #
    qual_node = evaluator.add_parallel(
        id=f"store_{store_index}_qualification",
        desc="Verify the store type and geographic presence",
        parent=store_node,
        critical=True
    )

    # Store Type leaf
    type_leaf = evaluator.add_leaf(
        id=f"store_{store_index}_type",
        desc="The store is a craft, hobby, or DIY supply retailer operating in the United States",
        parent=qual_node,
        critical=True
    )
    type_sources = _uniq_urls([store.type_url, store.ohio_presence_url] + _holiday_sources_for(store))
    await evaluator.verify(
        claim=f"{store_name} is a craft, hobby, or DIY supply retail chain that operates in the United States.",
        node=type_leaf,
        sources=type_sources,
        additional_instruction=(
            "Use only official corporate sources (brand/corporate domain, official press release, "
            "or official store locator/department pages). Third-party sites are not acceptable. "
            "DIY supply includes home-improvement/hardware, hobby, or craft retailers. "
            "If no official source is provided, judge as Incorrect."
        ),
    )

    # Ohio Presence leaf
    ohio_leaf = evaluator.add_leaf(
        id=f"store_{store_index}_ohio_presence",
        desc="The store has at least one physical retail location in Ohio",
        parent=qual_node,
        critical=True
    )
    ohio_sources = _uniq_urls([store.ohio_presence_url] + _holiday_sources_for(store))
    await evaluator.verify(
        claim=f"{store_name} has at least one physical retail store location in the state of Ohio.",
        node=ohio_leaf,
        sources=ohio_sources,
        additional_instruction=(
            "Prefer an official store-locator page or official locations listing that clearly shows at least one Ohio store. "
            "If the page only states 'hours vary by location' without showing Ohio presence, or if no official source is provided, mark Incorrect. "
            "Third-party aggregators are not acceptable."
        ),
    )

    # ---------------- New Year's Day 2026 Compliance (critical) ----- #
    nyd_node = evaluator.add_parallel(
        id=f"store_{store_index}_new_years",
        desc="Verify New Year's Day 2026 operating hours compliance",
        parent=store_node,
        critical=True
    )
    nyd_sources = _uniq_urls([store.new_years_url] + _holiday_sources_for(store))

    # Open on Jan 1, 2026
    nyd_open_leaf = evaluator.add_leaf(
        id=f"store_{store_index}_open_jan1",
        desc="The store is open on New Year's Day 2026 (Thursday, January 1, 2026)",
        parent=nyd_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{store_name} is open on Thursday, January 1, 2026 (New Year's Day 2026).",
        node=nyd_open_leaf,
        sources=nyd_sources,
        additional_instruction=(
            "The supporting page must be an official 2025–2026 holiday-hours page (or equivalent) that covers New Year's Day 2026. "
            "If the source is for a different year or does not clearly pertain to the 2025–2026 season, mark Incorrect. "
            "If the page says the store is closed on New Year's Day, mark Incorrect."
        ),
    )

    # Opens by 9:30 AM on Jan 1, 2026
    nyd_time_leaf = evaluator.add_leaf(
        id=f"store_{store_index}_opens_by_930",
        desc="The store's opening time on New Year's Day 2026 is at or before 9:30 AM local time",
        parent=nyd_node,
        critical=True
    )
    claimed_time = store.new_years_open_time or "an opening time at or before 9:30 AM"
    await evaluator.verify(
        claim=(
            f"On New Year's Day 2026, {store_name} opens at or before 9:30 AM local time "
            f"(the answer cites '{claimed_time}' for that date, which must be supported by the official source)."
        ),
        node=nyd_time_leaf,
        sources=nyd_sources,
        additional_instruction=(
            "Treat 9:30 AM as inclusive. Accept equivalent forms (e.g., 9 AM, 9:00 AM, 09:30). "
            "If the official page only states 'hours vary by location' without a chain-level opening time for Jan 1, 2026, mark Incorrect. "
            "The evidence must be for the 2025–2026 season."
        ),
    )

    # New Year's URL provided (official)
    nyd_url_leaf = evaluator.add_leaf(
        id=f"store_{store_index}_new_years_url",
        desc="A direct URL to an official source confirming the store's New Year's Day 2026 hours is provided",
        parent=nyd_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The answer provides a direct official URL for {store_name} that confirms the 2025–2026 holiday schedule "
            "or the New Year's Day 2026 hours."
        ),
        node=nyd_url_leaf,
        sources=store.new_years_url,
        additional_instruction=(
            "Use the provided URL (if any). If no URL is provided, judge this claim as Incorrect. "
            "The page must be official (corporate domain, official press release, or store locator) and clearly cover the 2025–2026 season."
        ),
    )

    # ---------------- Christmas Day 2025 Compliance (critical) ------- #
    xmas_node = evaluator.add_parallel(
        id=f"store_{store_index}_christmas",
        desc="Verify Christmas Day 2025 closure compliance",
        parent=store_node,
        critical=True
    )
    xmas_sources = _uniq_urls([store.christmas_url] + _holiday_sources_for(store))

    # Closed Dec 25, 2025
    xmas_closed_leaf = evaluator.add_leaf(
        id=f"store_{store_index}_closed_dec25",
        desc="The store is closed all day on Christmas Day 2025 (Thursday, December 25, 2025)",
        parent=xmas_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{store_name} is closed all day on Thursday, December 25, 2025 (Christmas Day 2025).",
        node=xmas_closed_leaf,
        sources=xmas_sources,
        additional_instruction=(
            "The supporting page should clearly pertain to 2025 (or a 2025–2026 holiday schedule). "
            "If the page is for a different year, or the year is ambiguous and not clearly 2025–2026, mark Incorrect."
        ),
    )

    # Christmas URL provided (official)
    xmas_url_leaf = evaluator.add_leaf(
        id=f"store_{store_index}_christmas_url",
        desc="A direct URL to an official source confirming the store's Christmas Day 2025 closure is provided",
        parent=xmas_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The answer provides a direct official URL for {store_name} that confirms the 2025–2026 holiday schedule "
            "or explicitly states Christmas Day 2025 closure."
        ),
        node=xmas_url_leaf,
        sources=store.christmas_url,
        additional_instruction=(
            "Use the provided URL (if any). If no URL is provided, judge this claim as Incorrect. "
            "The page must be official (corporate domain, official press release, or store locator) and clearly cover 2025–2026."
        ),
    )

    # ---------------- Black Friday 2025 Compliance (critical) -------- #
    bf_node = evaluator.add_parallel(
        id=f"store_{store_index}_black_friday",
        desc="Verify Black Friday 2025 operating hours compliance",
        parent=store_node,
        critical=True
    )
    bf_sources = _uniq_urls([store.black_friday_url] + _holiday_sources_for(store))

    # Open on Nov 28, 2025 (Black Friday 2025)
    bf_open_leaf = evaluator.add_leaf(
        id=f"store_{store_index}_open_bf",
        desc="The store is open on Black Friday 2025 (Friday, November 28, 2025)",
        parent=bf_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{store_name} is open on Friday, November 28, 2025 (Black Friday 2025).",
        node=bf_open_leaf,
        sources=bf_sources,
        additional_instruction=(
            "The supporting page must clearly pertain to 2025 (or a 2025–2026 holiday schedule). "
            "If the page states the store is closed, mark Incorrect."
        ),
    )

    # Opens by 8:00 AM on Black Friday 2025
    bf_time_leaf = evaluator.add_leaf(
        id=f"store_{store_index}_opens_by_8",
        desc="The store's opening time on Black Friday 2025 is at or before 8:00 AM local time",
        parent=bf_node,
        critical=True
    )
    bf_claim_time = store.black_friday_open_time or "an opening time at or before 8:00 AM"
    await evaluator.verify(
        claim=(
            f"On Black Friday 2025, {store_name} opens at or before 8:00 AM local time "
            f"(the answer cites '{bf_claim_time}' for that date, which must be supported by the official source)."
        ),
        node=bf_time_leaf,
        sources=bf_sources,
        additional_instruction=(
            "Treat 8:00 AM as inclusive (e.g., 6 AM, 7 AM, 8 AM all satisfy). "
            "If the official page only states 'hours vary by location' without a chain-level opening time for Black Friday 2025, mark Incorrect. "
            "Evidence must clearly correspond to 2025."
        ),
    )

    # Black Friday URL provided (official)
    bf_url_leaf = evaluator.add_leaf(
        id=f"store_{store_index}_bf_url",
        desc="A direct URL to an official source confirming the store's Black Friday 2025 hours is provided",
        parent=bf_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The answer provides a direct official URL for {store_name} that confirms Black Friday 2025 hours "
            "or a 2025–2026 holiday schedule including Black Friday."
        ),
        node=bf_url_leaf,
        sources=store.black_friday_url,
        additional_instruction=(
            "Use the provided URL (if any). If no URL is provided, judge this claim as Incorrect. "
            "The page must be official (corporate domain, official press release, or store locator) and clearly cover 2025–2026."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Ohio craft/hobby/DIY store holiday-hours 2025–2026 task.
    """
    # Initialize evaluator (root kept non-critical to allow partial credit across stores)
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

    # Extract structured store info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_stores(),
        template_class=StoresExtraction,
        extraction_name="stores_extraction"
    )

    # Normalize to exactly 4 store entries: take first 4, pad with empty if fewer
    stores: List[StoreEntry] = list(extracted.stores[:4])
    while len(stores) < 4:
        stores.append(StoreEntry())

    # Build verification subtrees per store
    for i in range(4):
        await verify_store(evaluator, root, stores[i], i)

    # Return standardized evaluation summary
    return evaluator.get_summary()