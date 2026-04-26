import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.verification_tree import VerificationNode


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "state_ihop_cbs_thanksgiving_2026"
TASK_DESCRIPTION = """
Identify the U.S. state that is the only state without any IHOP locations and is also the only state without any Dairy Queen locations. This same state must have the lowest fast food restaurant density per capita in the United States. Once you have identified this state, provide its name.

Additionally, provide the following information about IHOP:
1. The total number of IHOP locations across the United States
2. The percentage of IHOP locations that are independently franchise-owned
3. Whether most IHOP locations remain open on Christmas Day

Next, identify the culinary competition television show that premiered on CBS in March 2026 featuring 16 elite chef competitors. For this show, provide:
1. The name of the show
2. The creator of the show
3. The prize amount for the competition winner
4. The number of Michelin-starred chefs among the 16 competitors

Finally, verify the operating status on Thanksgiving Day 2025 for:
1. Walmart stores
2. Walgreens locations (specifically noting whether most locations were closed or if 24-hour locations remained open)

All facts must be supported by URL references from reliable sources.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StateExtraction(BaseModel):
    state_name: Optional[str] = None
    sources_without_ihop: List[str] = Field(default_factory=list)
    sources_without_dairy_queen: List[str] = Field(default_factory=list)
    sources_low_fastfood_density: List[str] = Field(default_factory=list)


class IHOPExtraction(BaseModel):
    total_us_locations: Optional[str] = None
    total_locations_sources: List[str] = Field(default_factory=list)

    franchise_owned_percentage: Optional[str] = None
    franchise_percentage_sources: List[str] = Field(default_factory=list)

    # Prefer normalized yes/no; but also capture free-form statement for robustness
    christmas_most_open_yes_no: Optional[str] = None  # expected values like "yes"/"no"/"true"/"false"
    christmas_day_statement: Optional[str] = None
    christmas_day_sources: List[str] = Field(default_factory=list)


class ShowExtraction(BaseModel):
    show_name: Optional[str] = None

    # Premiere info
    premiere_month: Optional[str] = None  # e.g., "March"
    premiere_year: Optional[str] = None   # e.g., "2026"
    premiere_network: Optional[str] = None  # e.g., "CBS"
    premiere_sources: List[str] = Field(default_factory=list)

    # Exactly 16 competitors requirement
    competitor_count: Optional[str] = None
    competitor_sources: List[str] = Field(default_factory=list)

    # Creator
    creator_name: Optional[str] = None
    creator_sources: List[str] = Field(default_factory=list)

    # Prize
    prize_amount: Optional[str] = None
    prize_sources: List[str] = Field(default_factory=list)

    # Michelin-starred count among competitors
    michelin_starred_count: Optional[str] = None
    michelin_count_sources: List[str] = Field(default_factory=list)


class ThanksgivingExtraction(BaseModel):
    walmart_status: Optional[str] = None  # e.g., "closed", "open", "limited hours"
    walmart_sources: List[str] = Field(default_factory=list)

    walgreens_status: Optional[str] = None  # e.g., "most locations closed"
    walgreens_24hr_note: Optional[str] = None  # e.g., "24-hour locations remained open"
    walgreens_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_state_info() -> str:
    return """
    Extract the state identification and sources from the answer.

    Return a JSON object with the following fields:
    - state_name: the name of the U.S. state identified
    - sources_without_ihop: an array of URLs explicitly cited in the answer that support that this state has zero IHOP locations, and that it is the ONLY U.S. state with zero IHOP locations
    - sources_without_dairy_queen: an array of URLs explicitly cited in the answer that support that this state has zero Dairy Queen locations, and that it is the ONLY U.S. state with zero Dairy Queen locations
    - sources_low_fastfood_density: an array of URLs explicitly cited in the answer that support that this state has the lowest fast food restaurant density per capita among U.S. states

    For each sources field, extract only actual URLs present in the answer text (including markdown links). If no URLs are provided for a given claim, return an empty array for that field.
    """


def prompt_extract_ihop_info() -> str:
    return """
    Extract IHOP-related facts and their cited sources from the answer.

    Return a JSON object with the following fields:
    - total_us_locations: the stated total number of IHOP locations across the United States (as a string, exactly as written)
    - total_locations_sources: an array of URLs cited for the total number of locations
    - franchise_owned_percentage: the stated percentage of IHOP locations that are independently franchise-owned (as a string, e.g., "99%")
    - franchise_percentage_sources: an array of URLs cited for the franchise percentage
    - christmas_most_open_yes_no: normalize whether most IHOP locations remain open on Christmas Day; return "yes" if the answer says most remain open, "no" if the answer says most do not remain open, or null if not clearly stated
    - christmas_day_statement: the sentence or short phrase from the answer that states the operating pattern on Christmas Day (verbatim)
    - christmas_day_sources: an array of URLs cited for the Christmas Day operating pattern

    If a field is missing from the answer, set it to null (or empty array for sources).
    """


def prompt_extract_show_info() -> str:
    return """
    Extract the CBS culinary competition show details and citations from the answer.

    Return a JSON object with the following fields:
    - show_name: the name of the show
    - premiere_month: the stated premiere month (e.g., "March")
    - premiere_year: the stated premiere year (e.g., "2026")
    - premiere_network: the stated network (e.g., "CBS")
    - premiere_sources: an array of URLs cited for the premiere on CBS in March 2026
    - competitor_count: the stated number of competitors (as a string, e.g., "16")
    - competitor_sources: an array of URLs cited for the competitor count
    - creator_name: the stated creator of the show
    - creator_sources: an array of URLs cited for the creator
    - prize_amount: the stated prize amount for the competition winner (as a string, e.g., "$250,000")
    - prize_sources: an array of URLs cited for the prize amount
    - michelin_starred_count: the stated number of Michelin-starred chefs among the competitors (as a string)
    - michelin_count_sources: an array of URLs cited for the Michelin-starred count

    Extract only URLs that are explicitly present in the answer text. Use empty arrays where sources are not provided.
    """


def prompt_extract_thanksgiving_info() -> str:
    return """
    Extract Thanksgiving Day 2025 operating statuses and their sources for Walmart and Walgreens from the answer.

    Return a JSON object with the following fields:
    - walmart_status: the answer's statement for Walmart stores on Thanksgiving Day 2025 (e.g., "closed", "open", "open with limited hours"); keep it short and normalized where possible
    - walmart_sources: an array of URLs cited for the Walmart Thanksgiving 2025 status
    - walgreens_status: the answer's statement summarizing Walgreens' operating status on Thanksgiving Day 2025 (e.g., "most locations closed")
    - walgreens_24hr_note: the explicit note about 24-hour Walgreens locations (e.g., "24-hour locations remained open") if provided; otherwise null
    - walgreens_sources: an array of URLs cited for the Walgreens Thanksgiving 2025 status

    If a field is missing from the answer, set it to null (sources fields to empty array).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _truthy_yes(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"yes", "true", "y", "most open", "open"}:
        return True
    if v in {"no", "false", "n", "most closed", "closed"}:
        return False
    return None


def _nz_list(lst: Optional[List[str]]) -> List[str]:
    return [u for u in (lst or []) if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_state_verification(evaluator: Evaluator, root: VerificationNode, st: StateExtraction) -> None:
    # Parent sequential node (critical): State identification and three required properties
    state_node = evaluator.add_sequential(
        id="StateIdentificationAndVerification",
        desc="Identify the U.S. state meeting the specified uniqueness and density criteria, and provide its name.",
        parent=root,
        critical=True
    )

    # 1) State name provided (existence check)
    evaluator.add_custom_node(
        result=bool(st and st.state_name and st.state_name.strip()),
        id="StateName",
        desc="Provide the name of the identified U.S. state.",
        parent=state_node,
        critical=True
    )

    state_name = (st.state_name or "").strip()

    # 2) Only state without IHOP
    n1 = evaluator.add_leaf(
        id="OnlyStateWithoutIHOP",
        desc="The identified state is the only U.S. state with zero IHOP locations.",
        parent=state_node,
        critical=True
    )
    claim1 = f"{state_name} is the only U.S. state with zero IHOP locations."
    await evaluator.verify(
        claim=claim1,
        node=n1,
        sources=_nz_list(st.sources_without_ihop),
        additional_instruction="Confirm both zero IHOP locations in the state and that no other U.S. state also has zero IHOP locations (i.e., this state is the only one). Prefer official IHOP, reputable news, or reliable store locator data."
    )

    # 3) Only state without Dairy Queen
    n2 = evaluator.add_leaf(
        id="OnlyStateWithoutDairyQueen",
        desc="The identified state is the only U.S. state with zero Dairy Queen locations.",
        parent=state_node,
        critical=True
    )
    claim2 = f"{state_name} is the only U.S. state with zero Dairy Queen locations."
    await evaluator.verify(
        claim=claim2,
        node=n2,
        sources=_nz_list(st.sources_without_dairy_queen),
        additional_instruction="Confirm both zero Dairy Queen locations in the state and that no other U.S. state also has zero Dairy Queen locations (i.e., this state is the only one). Prefer official DQ, reputable news, or reliable store locator data."
    )

    # 4) Lowest fast food density per capita
    n3 = evaluator.add_leaf(
        id="LowestFastFoodDensityPerCapita",
        desc="The identified state has the lowest fast food restaurant density per capita in the United States.",
        parent=state_node,
        critical=True
    )
    claim3 = f"{state_name} has the lowest fast food restaurant density per capita among all U.S. states."
    await evaluator.verify(
        claim=claim3,
        node=n3,
        sources=_nz_list(st.sources_low_fastfood_density),
        additional_instruction="Verify that among U.S. states, this state ranks last in fast food restaurant density normalized by population (per capita). The source should clearly show a cross-state comparison or ranking."
    )


async def build_ihop_verification(evaluator: Evaluator, root: VerificationNode, ih: IHOPExtraction) -> None:
    # Parent parallel node (critical): each sub-claim verified independently
    ihop_node = evaluator.add_parallel(
        id="IHOPInformation",
        desc="Provide the requested IHOP facts.",
        parent=root,
        critical=True
    )

    # Total US locations
    n_total = evaluator.add_leaf(
        id="IHOPTotalUSLocations",
        desc="Provide the total number of IHOP locations across the United States.",
        parent=ihop_node,
        critical=True
    )
    total_str = (ih.total_us_locations or "").strip()
    claim_total = f"The total number of IHOP locations across the United States is {total_str}."
    await evaluator.verify(
        claim=claim_total,
        node=n_total,
        sources=_nz_list(ih.total_locations_sources),
        additional_instruction="Confirm the stated total number of IHOP locations from the cited source. Minor rounding differences are acceptable if the source supports them."
    )

    # Franchise-owned percentage
    n_pct = evaluator.add_leaf(
        id="IHOPFranchiseOwnedPercentage",
        desc="Provide the percentage of IHOP locations that are independently franchise-owned.",
        parent=ihop_node,
        critical=True
    )
    pct_str = (ih.franchise_owned_percentage or "").strip()
    claim_pct = f"The percentage of IHOP locations that are independently franchise-owned is {pct_str}."
    await evaluator.verify(
        claim=claim_pct,
        node=n_pct,
        sources=_nz_list(ih.franchise_percentage_sources),
        additional_instruction="Confirm the franchise-owned percentage from the cited source (e.g., company filings, press releases, or trusted business reporting)."
    )

    # Christmas Day operating pattern (majority open or not)
    n_xmas = evaluator.add_leaf(
        id="IHOPChristmasDayOperatingPattern",
        desc="State whether most IHOP locations remain open on Christmas Day.",
        parent=ihop_node,
        critical=True
    )
    yn = _truthy_yes(ih.christmas_most_open_yes_no)
    if yn is True:
        claim_xmas = "Most IHOP locations remain open on Christmas Day."
    elif yn is False:
        claim_xmas = "Most IHOP locations do not remain open on Christmas Day (i.e., most are closed)."
    else:
        # Fallback to provided statement if any; otherwise default a direct claim that must be supported by sources
        claim_xmas = (ih.christmas_day_statement or "Most IHOP locations remain open on Christmas Day.").strip()

    await evaluator.verify(
        claim=claim_xmas,
        node=n_xmas,
        sources=_nz_list(ih.christmas_day_sources),
        additional_instruction="Verify the typical/majority operating pattern of IHOP restaurants on Christmas Day per the cited source(s). Reasonable generalizations are acceptable only if the source states them."
    )


async def build_show_verification(evaluator: Evaluator, root: VerificationNode, sh: ShowExtraction) -> None:
    show_node = evaluator.add_sequential(
        id="CulinaryShowOnCBS_March2026",
        desc="Identify the culinary competition show that premiered on CBS in March 2026 featuring 16 elite chef competitors, and provide required attributes.",
        parent=root,
        critical=True
    )

    # Show name existence
    evaluator.add_custom_node(
        result=bool(sh and sh.show_name and sh.show_name.strip()),
        id="ShowName",
        desc="Provide the name of the show.",
        parent=show_node,
        critical=True
    )

    show_name = (sh.show_name or "").strip()
    month = (sh.premiere_month or "").strip()
    year = (sh.premiere_year or "").strip()
    network = (sh.premiere_network or "").strip()

    # Premiere on CBS in March 2026
    n_prem = evaluator.add_leaf(
        id="PremiereOnCBSInMarch2026",
        desc="The show premiered on CBS in March 2026.",
        parent=show_node,
        critical=True
    )
    # Construct strict claim as per rubric
    claim_prem = f"The show '{show_name}' premiered on CBS in March 2026."
    await evaluator.verify(
        claim=claim_prem,
        node=n_prem,
        sources=_nz_list(sh.premiere_sources),
        additional_instruction="Confirm that the show's initial premiere occurred on CBS and in March 2026 (month and year must both match). Prefer official CBS/Paramount press, reputable news, or authoritative listings."
    )

    # Exactly 16 competitors
    n_comp = evaluator.add_leaf(
        id="Exactly16Competitors",
        desc="The show features exactly 16 elite chef competitors.",
        parent=show_node,
        critical=True
    )
    claim_comp = f"The show '{show_name}' features exactly 16 elite chef competitors."
    await evaluator.verify(
        claim=claim_comp,
        node=n_comp,
        sources=_nz_list(sh.competitor_sources),
        additional_instruction="Confirm that the number of competitors is exactly 16. If sources describe 'elite chef competitors', that phrasing is acceptable if it clearly refers to the 16 chefs participating."
    )

    # Show creator
    n_creator = evaluator.add_leaf(
        id="ShowCreator",
        desc="Provide the creator of the show.",
        parent=show_node,
        critical=True
    )
    creator = (sh.creator_name or "").strip()
    claim_creator = f"The creator of the show '{show_name}' is {creator}."
    await evaluator.verify(
        claim=claim_creator,
        node=n_creator,
        sources=_nz_list(sh.creator_sources),
        additional_instruction="Verify the credited creator from the cited source."
    )

    # Prize amount
    n_prize = evaluator.add_leaf(
        id="PrizeAmount",
        desc="Provide the prize amount for the competition winner.",
        parent=show_node,
        critical=True
    )
    prize = (sh.prize_amount or "").strip()
    claim_prize = f"The prize amount for the competition winner on '{show_name}' is {prize}."
    await evaluator.verify(
        claim=claim_prize,
        node=n_prize,
        sources=_nz_list(sh.prize_sources),
        additional_instruction="Verify the exact prize amount for the winner from the cited source."
    )

    # Michelin-starred chef count among competitors
    n_mich = evaluator.add_leaf(
        id="MichelinStarredChefCount",
        desc="Provide the number of Michelin-starred chefs among the 16 competitors.",
        parent=show_node,
        critical=True
    )
    mich = (sh.michelin_starred_count or "").strip()
    claim_mich = f"Among the 16 competitors on '{show_name}', {mich} are Michelin-starred chefs."
    await evaluator.verify(
        claim=claim_mich,
        node=n_mich,
        sources=_nz_list(sh.michelin_count_sources),
        additional_instruction="Verify the number of Michelin-starred chefs among the competitors as stated by the source."
    )


async def build_thanksgiving_verification(evaluator: Evaluator, root: VerificationNode, tg: ThanksgivingExtraction) -> None:
    tg_node = evaluator.add_parallel(
        id="ThanksgivingDay2025OperatingStatus",
        desc="Verify Thanksgiving Day 2025 operating status for Walmart and Walgreens, including the requested Walgreens 24-hour distinction.",
        parent=root,
        critical=True
    )

    # Walmart status
    n_wm = evaluator.add_leaf(
        id="WalmartThanksgivingStatus",
        desc="State Walmart stores' operating status on Thanksgiving Day 2025 (open/closed/limited).",
        parent=tg_node,
        critical=True
    )
    walmart_status_text = (tg.walmart_status or "").strip()
    if walmart_status_text:
        claim_wm = f"On Thanksgiving Day 2025, Walmart stores were {walmart_status_text}."
    else:
        # Use a neutral phrasing when missing; verification will fail if sources do not support
        claim_wm = "On Thanksgiving Day 2025, Walmart stores were closed."
    await evaluator.verify(
        claim=claim_wm,
        node=n_wm,
        sources=_nz_list(tg.walmart_sources),
        additional_instruction="Confirm Walmart's official or widely reported operating status on Thanksgiving Day 2025 (e.g., closed vs open/limited). Prefer Walmart corporate announcements or reputable news."
    )

    # Walgreens status with 24-hour distinction
    n_wg = evaluator.add_leaf(
        id="WalgreensThanksgivingStatusWith24HourDistinction",
        desc="State Walgreens' operating status on Thanksgiving Day 2025, explicitly addressing whether most locations were closed and whether 24-hour locations remained open.",
        parent=tg_node,
        critical=True
    )
    walgreens_status_text = (tg.walgreens_status or "").strip()
    walgreens_24h = (tg.walgreens_24hr_note or "").strip()
    if walgreens_status_text and walgreens_24h:
        claim_wg = f"On Thanksgiving Day 2025, {walgreens_status_text}, and {walgreens_24h}."
    elif walgreens_status_text:
        claim_wg = f"On Thanksgiving Day 2025, {walgreens_status_text}."
    else:
        claim_wg = "On Thanksgiving Day 2025, most Walgreens locations were closed, but 24-hour locations remained open."
    await evaluator.verify(
        claim=claim_wg,
        node=n_wg,
        sources=_nz_list(tg.walgreens_sources),
        additional_instruction="Verify Walgreens' operating status on Thanksgiving Day 2025, explicitly noting whether most locations were closed and whether any 24-hour stores remained open. Prefer Walgreens corporate announcements or reputable news."
    )


def compute_global_sourcing_ok(
    st: Optional[StateExtraction],
    ih: Optional[IHOPExtraction],
    sh: Optional[ShowExtraction],
    tg: Optional[ThanksgivingExtraction]
) -> Dict[str, Any]:
    # Build a flat list of (label, urls_list) to check for presence of sources
    items: List[Dict[str, Any]] = []

    if st:
        items.extend([
            {"field": "state_without_ihop", "urls": _nz_list(st.sources_without_ihop)},
            {"field": "state_without_dq", "urls": _nz_list(st.sources_without_dairy_queen)},
            {"field": "state_low_ff_density", "urls": _nz_list(st.sources_low_fastfood_density)},
        ])
    if ih:
        items.extend([
            {"field": "ihop_total_locations", "urls": _nz_list(ih.total_locations_sources)},
            {"field": "ihop_franchise_pct", "urls": _nz_list(ih.franchise_percentage_sources)},
            {"field": "ihop_christmas_day", "urls": _nz_list(ih.christmas_day_sources)},
        ])
    if sh:
        items.extend([
            {"field": "show_premiere", "urls": _nz_list(sh.premiere_sources)},
            {"field": "show_competitor_count", "urls": _nz_list(sh.competitor_sources)},
            {"field": "show_creator", "urls": _nz_list(sh.creator_sources)},
            {"field": "show_prize", "urls": _nz_list(sh.prize_sources)},
            {"field": "show_michelin_count", "urls": _nz_list(sh.michelin_count_sources)},
        ])
    if tg:
        items.extend([
            {"field": "walmart_thanksgiving", "urls": _nz_list(tg.walmart_sources)},
            {"field": "walgreens_thanksgiving", "urls": _nz_list(tg.walgreens_sources)},
        ])

    missing = [it["field"] for it in items if not it["urls"]]
    all_ok = len(missing) == 0
    return {"all_ok": all_ok, "missing_fields": missing, "checked_items": len(items)}


async def add_global_sourcing_node(
    evaluator: Evaluator,
    root: VerificationNode,
    st: Optional[StateExtraction],
    ih: Optional[IHOPExtraction],
    sh: Optional[ShowExtraction],
    tg: Optional[ThanksgivingExtraction],
) -> None:
    stats = compute_global_sourcing_ok(st, ih, sh, tg)
    evaluator.add_custom_info(stats, info_type="sourcing_stats", info_name="global_sourcing_stats")

    evaluator.add_custom_node(
        result=stats["all_ok"],
        id="GlobalSourcingRequirement",
        desc="All factual claims in the answer are supported by publicly accessible URL references from reliable sources.",
        parent=root,
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate the multi-component research task:
    - Identify the unique U.S. state (IHOP/DQ absence + lowest fast food density)
    - Provide IHOP facts (total locations, franchise percentage, Christmas Day majority-open)
    - Identify CBS March 2026 culinary competition show and details
    - Verify Thanksgiving Day 2025 operating statuses (Walmart, Walgreens with 24h nuance)
    - Ensure all facts are sourced
    """
    # Initialize evaluator with a critical parallel root
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
    # Make root critical (so all children must be critical and passing)
    root.critical = True

    # Parallelize extractions
    state_extraction_task = evaluator.extract(
        prompt=prompt_extract_state_info(),
        template_class=StateExtraction,
        extraction_name="state_extraction"
    )
    ihop_extraction_task = evaluator.extract(
        prompt=prompt_extract_ihop_info(),
        template_class=IHOPExtraction,
        extraction_name="ihop_extraction"
    )
    show_extraction_task = evaluator.extract(
        prompt=prompt_extract_show_info(),
        template_class=ShowExtraction,
        extraction_name="show_extraction"
    )
    thanksgiving_extraction_task = evaluator.extract(
        prompt=prompt_extract_thanksgiving_info(),
        template_class=ThanksgivingExtraction,
        extraction_name="thanksgiving_extraction"
    )

    st, ih, sh, tg = await asyncio.gather(
        state_extraction_task,
        ihop_extraction_task,
        show_extraction_task,
        thanksgiving_extraction_task
    )

    # Build verification subtrees
    await build_state_verification(evaluator, root, st or StateExtraction())
    await build_ihop_verification(evaluator, root, ih or IHOPExtraction())
    await build_show_verification(evaluator, root, sh or ShowExtraction())
    await build_thanksgiving_verification(evaluator, root, tg or ThanksgivingExtraction())

    # Global sourcing requirement (critical)
    await add_global_sourcing_node(evaluator, root, st, ih, sh, tg)

    # Return evaluation summary
    return evaluator.get_summary()