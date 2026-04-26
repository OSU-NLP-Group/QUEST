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
TASK_ID = "retail_research_multi_area"
TASK_DESCRIPTION = """
You are assisting a business owner who is planning retail purchases across multiple categories in the northeastern United States. The owner has specific requirements and needs you to research and provide detailed information across four major areas:

Area 1: Geographic Retail Analysis
Among the 9 states where Wegmans grocery stores operate, identify which state has the most CVS pharmacy locations. Then, verify whether CVS pharmacies in general are typically open on Christmas Day.

Area 2: Warehouse Club Membership Details
The business owner wants to purchase a Sam's Club Plus membership. Provide the following information about Sam's Club Plus membership:
- The annual membership cost
- What time Plus members can begin shopping on weekdays (the start of the exclusive early shopping hour)
- How many guests each member can bring per visit
- The percentage of Sam's Cash back that Plus members earn on qualifying purchases

Area 3: Business Technology Purchasing
The company needs to purchase laptops for business use. Identify:
- The minimum RAM (in GB) recommended for business/professional laptops in 2024-2025
- The minimum processor type recommended (specify Intel Core series or AMD Ryzen series)
- Approximately how many Best Buy stores operate in the United States

Area 4: Product Availability
Determine whether Crocs brand footwear is sold at Best Buy electronics stores (answer Yes or No).

Provide specific, verifiable answers with supporting details for each requirement.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StateCVSInfo(BaseModel):
    state: Optional[str] = None
    cvs_count: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CVSChristmasPolicy(BaseModel):
    typically_open_on_christmas: Optional[str] = None  # Expect "Yes" or "No" (as string)
    note_hours_may_vary: Optional[str] = None          # Free text, e.g., "Hours may vary/reduced"
    sources: List[str] = Field(default_factory=list)


class Area1Extraction(BaseModel):
    wegmans_state_most_cvs: Optional[StateCVSInfo] = None
    holiday_policy: Optional[CVSChristmasPolicy] = None


class MembershipCost(BaseModel):
    amount: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class EarlyShoppingHours(BaseModel):
    weekday_start_time: Optional[str] = None   # e.g., "8 a.m." or "8:00 AM"
    context: Optional[str] = None              # e.g., "Plus early hours are 8–10 a.m. Mon–Fri"
    sources: List[str] = Field(default_factory=list)


class GuestPolicy(BaseModel):
    guest_limit: Optional[str] = None          # e.g., "2"
    sources: List[str] = Field(default_factory=list)


class CashBack(BaseModel):
    percentage: Optional[str] = None           # e.g., "2%"
    max_annual_reward: Optional[str] = None    # e.g., "$500 per year"
    sources: List[str] = Field(default_factory=list)


class Area2Extraction(BaseModel):
    cost: Optional[MembershipCost] = None
    early_hours: Optional[EarlyShoppingHours] = None
    guest_policy: Optional[GuestPolicy] = None
    cash_back: Optional[CashBack] = None


class LaptopRAM(BaseModel):
    minimum_recommended: Optional[str] = None  # e.g., "16 GB"
    sources: List[str] = Field(default_factory=list)


class LaptopProcessor(BaseModel):
    minimum_recommended: Optional[str] = None  # e.g., "Intel Core i5" or "AMD Ryzen 5"
    sources: List[str] = Field(default_factory=list)


class BestBuyStoreCount(BaseModel):
    approx_count: Optional[str] = None         # e.g., "approximately 950"
    sources: List[str] = Field(default_factory=list)


class Area3Extraction(BaseModel):
    laptop_ram: Optional[LaptopRAM] = None
    laptop_processor: Optional[LaptopProcessor] = None
    best_buy_store_count: Optional[BestBuyStoreCount] = None


class CrocsAvailability(BaseModel):
    crocs_at_best_buy: Optional[str] = None    # Expect "Yes" or "No" (as string)
    crocs_retailers: List[str] = Field(default_factory=list)  # Other known retailers
    sources: List[str] = Field(default_factory=list)


class Area4Extraction(BaseModel):
    crocs_availability: Optional[CrocsAvailability] = None


class FullExtraction(BaseModel):
    area1: Optional[Area1Extraction] = None
    area2: Optional[Area2Extraction] = None
    area3: Optional[Area3Extraction] = None
    area4: Optional[Area4Extraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_full() -> str:
    return """
Extract the requested information from the answer text. Return all fields as strings when applicable. For URLs, return a list of complete URLs. If something is missing from the answer, set it to null (or empty list for arrays). Do not invent information.

Structure your JSON exactly as follows:

{
  "area1": {
    "wegmans_state_most_cvs": {
      "state": "<state name>",
      "cvs_count": "<count or approximate text>",
      "sources": ["<url1>", "<url2>", ...]
    },
    "holiday_policy": {
      "typically_open_on_christmas": "Yes or No",
      "note_hours_may_vary": "<optional short note if mentioned>",
      "sources": ["<url1>", "<url2>", ...]
    }
  },
  "area2": {
    "cost": {
      "amount": "<e.g., $110>",
      "sources": ["<url1>", "<url2>", ...]
    },
    "early_hours": {
      "weekday_start_time": "<e.g., 8 a.m.>",
      "context": "<e.g., Plus early shopping hour is 8–10 a.m. Mon–Fri>",
      "sources": ["<url1>", "<url2>", ...]
    },
    "guest_policy": {
      "guest_limit": "<e.g., 2>",
      "sources": ["<url1>", "<url2>", ...]
    },
    "cash_back": {
      "percentage": "<e.g., 2%>",
      "max_annual_reward": "<e.g., $500 per year>",
      "sources": ["<url1>", "<url2>", ...]
    }
  },
  "area3": {
    "laptop_ram": {
      "minimum_recommended": "<e.g., 16 GB>",
      "sources": ["<url1>", "<url2>", ...]
    },
    "laptop_processor": {
      "minimum_recommended": "<e.g., Intel Core i5 or AMD Ryzen 5>",
      "sources": ["<url1>", "<url2>", ...]
    },
    "best_buy_store_count": {
      "approx_count": "<e.g., approximately 950>",
      "sources": ["<url1>", "<url2>", ...]
    }
  },
  "area4": {
    "crocs_availability": {
      "crocs_at_best_buy": "Yes or No",
      "crocs_retailers": ["<retailer1>", "<retailer2>", ...],
      "sources": ["<url1>", "<url2>", ...]
    }
  }
}

Guidelines:
- For boolean-type answers (e.g., Yes/No), return "Yes" or "No" exactly.
- For time fields, keep the format found in the answer (e.g., "8 a.m.", "8:00 AM").
- For amounts and percentages, keep symbols (e.g., "$110", "2%").
- For counts like store counts, you may include words like "approximately" if that's how the answer phrases it.
- For sources, return only explicit URLs present in the answer text.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _urls_or_none(urls: Optional[List[str]]) -> Optional[List[str]]:
    return urls if urls else None


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0


def _safe_str(x: Optional[str]) -> str:
    return x or ""


# --------------------------------------------------------------------------- #
# Verification builders (per area)                                            #
# --------------------------------------------------------------------------- #
async def build_area1_geographic(evaluator: Evaluator, root_node, area1: Optional[Area1Extraction]) -> None:
    # Parent node for area 1 (non-critical aggregator to allow partial scoring)
    area1_node = evaluator.add_parallel(
        id="geographic_retail_analysis",
        desc="Identify the Wegmans state with most CVS locations and verify CVS Christmas Day hours",
        parent=root_node,
        critical=False
    )

    # Sub-area: State with most CVS among Wegmans states
    state_block = evaluator.add_parallel(
        id="state_with_most_cvs",
        desc="Identify which state among the 9 Wegmans states has the most CVS pharmacy locations",
        parent=area1_node,
        critical=False
    )

    state_info = area1.wegmans_state_most_cvs if area1 else None
    state_name = _safe_str(state_info.state) if state_info else ""
    cvs_count = _safe_str(state_info.cvs_count) if state_info else ""
    state_sources = state_info.sources if state_info else []

    # Reference presence (critical)
    evaluator.add_custom_node(
        result=_has_urls(state_sources),
        id="cvs_distribution_reference",
        desc="Reference URL for CVS store distribution data",
        parent=state_block,
        critical=True
    )

    # State identification (critical)
    state_ident_node = evaluator.add_leaf(
        id="state_identification",
        desc="Correctly identify the state with the most CVS locations among the 9 states where Wegmans operates",
        parent=state_block,
        critical=True
    )
    claim_state = (
        f"Among the states where Wegmans operates, the state with the most CVS Pharmacy locations is {state_name}."
    )
    await evaluator.verify(
        claim=claim_state,
        node=state_ident_node,
        sources=_urls_or_none(state_sources),
        additional_instruction=(
            "Use the provided source page(s) that show CVS store counts by U.S. state. "
            "If the identified state ranks #1 nationally for CVS locations, it is also #1 within the Wegmans subset."
        )
    )

    # CVS count verification (non-critical)
    cvs_count_node = evaluator.add_leaf(
        id="cvs_count_verification",
        desc="Provide the CVS store count for the identified state",
        parent=state_block,
        critical=False
    )
    claim_count = (
        f"The number of CVS Pharmacy locations in {state_name} is {cvs_count} (or approximately {cvs_count})."
        if cvs_count else f"The provided sources indicate the CVS location count in {state_name}."
    )
    await evaluator.verify(
        claim=claim_count,
        node=cvs_count_node,
        sources=_urls_or_none(state_sources),
        additional_instruction=(
            "Minor differences due to updates or rounding are acceptable. "
            "Confirm that the stated figure aligns with the source(s)."
        )
    )

    # Sub-area: CVS Christmas Day operating policy
    hours_block = evaluator.add_parallel(
        id="cvs_christmas_hours",
        desc="Verify CVS Christmas Day operating policy",
        parent=area1_node,
        critical=False
    )

    policy = area1.holiday_policy if area1 else None
    christmas_answer = _safe_str(policy.typically_open_on_christmas) if policy else ""
    christmas_sources = policy.sources if policy else []
    hours_note = _safe_str(policy.note_hours_may_vary) if policy else ""

    # Reference presence (critical)
    evaluator.add_custom_node(
        result=_has_urls(christmas_sources),
        id="cvs_holiday_hours_reference",
        desc="Reference URL for CVS holiday hours policy",
        parent=hours_block,
        critical=True
    )

    # Typically open on Christmas (critical)
    christmas_open_node = evaluator.add_leaf(
        id="christmas_day_open",
        desc="Confirm that CVS pharmacies are typically open on Christmas Day",
        parent=hours_block,
        critical=True
    )
    claim_christmas = "CVS pharmacies are typically open on Christmas Day."
    await evaluator.verify(
        claim=claim_christmas,
        node=christmas_open_node,
        sources=_urls_or_none(christmas_sources),
        additional_instruction=(
            "Look for CVS holiday hours guidance. It is common that stores may be open with reduced or special hours. "
            "If the source indicates they are open (even limited), treat as 'typically open'."
        )
    )

    # Hours may vary/reduced (non-critical)
    vary_node = evaluator.add_leaf(
        id="hours_may_vary",
        desc="Note that hours may vary by location or may be reduced",
        parent=hours_block,
        critical=False
    )
    claim_vary = (
        "Holiday hours may vary by location and may be reduced on Christmas Day at CVS stores."
    )
    await evaluator.verify(
        claim=claim_vary,
        node=vary_node,
        sources=_urls_or_none(christmas_sources),
        additional_instruction="Verify that the source mentions variations by location or reduced holiday hours."
    )


async def build_area2_membership(evaluator: Evaluator, root_node, area2: Optional[Area2Extraction]) -> None:
    area2_node = evaluator.add_parallel(
        id="warehouse_club_membership",
        desc="Provide complete Sam's Club Plus membership information including cost, early hours, guest policy, and rewards",
        parent=root_node,
        critical=False
    )

    # Membership cost
    cost_block = evaluator.add_parallel(
        id="membership_cost",
        desc="Identify the annual cost of Sam's Club Plus membership",
        parent=area2_node,
        critical=False
    )
    cost = area2.cost if area2 else None
    cost_amount = _safe_str(cost.amount) if cost else ""
    cost_sources = cost.sources if cost else []

    evaluator.add_custom_node(
        result=_has_urls(cost_sources),
        id="membership_cost_reference",
        desc="Reference URL for Sam's Club membership pricing",
        parent=cost_block,
        critical=True
    )

    cost_leaf = evaluator.add_leaf(
        id="cost_amount",
        desc="State the annual cost of Sam's Club Plus membership",
        parent=cost_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"The annual cost of Sam's Club Plus membership is {cost_amount}.",
        node=cost_leaf,
        sources=_urls_or_none(cost_sources),
        additional_instruction="Prefer official Sam's Club pages. Minor formatting differences (e.g., '$110' vs '110 dollars') are acceptable."
    )

    # Early shopping hours
    early_block = evaluator.add_parallel(
        id="early_shopping_hours",
        desc="Identify when Plus members can begin shopping on weekdays",
        parent=area2_node,
        critical=False
    )
    early = area2.early_hours if area2 else None
    start_time = _safe_str(early.weekday_start_time) if early else ""
    early_context = _safe_str(early.context) if early else ""
    early_sources = early.sources if early else []

    evaluator.add_custom_node(
        result=_has_urls(early_sources),
        id="early_hours_reference",
        desc="Reference URL for Sam's Club hours and Plus member benefits",
        parent=early_block,
        critical=True
    )

    early_start_leaf = evaluator.add_leaf(
        id="early_hour_start_time",
        desc="State the time Plus members can begin shopping on weekdays",
        parent=early_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"Sam's Club Plus members can begin shopping on weekdays at {start_time} (local store time).",
        node=early_start_leaf,
        sources=_urls_or_none(early_sources),
        additional_instruction="Verify weekday Plus early shopping start time. Ignore special events and holiday exceptions."
    )

    early_ctx_leaf = evaluator.add_leaf(
        id="early_hour_context",
        desc="Clarify the duration of the exclusive early shopping hour",
        parent=early_block,
        critical=False
    )
    # Use a generic verifiable claim about exclusivity and being earlier than regular opening
    await evaluator.verify(
        claim=(
            "Sam's Club Plus members receive an exclusive early shopping period on weekdays that starts before regular club hours."
        ),
        node=early_ctx_leaf,
        sources=_urls_or_none(early_sources),
        additional_instruction="Confirm that Plus membership grants an early access period (often described as an exclusive early hour) on weekdays."
    )

    # Guest policy
    guest_block = evaluator.add_parallel(
        id="guest_policy",
        desc="Identify how many guests members can bring per visit",
        parent=area2_node,
        critical=False
    )
    guest = area2.guest_policy if area2 else None
    guest_limit = _safe_str(guest.guest_limit) if guest else ""
    guest_sources = guest.sources if guest else []

    evaluator.add_custom_node(
        result=_has_urls(guest_sources),
        id="guest_policy_reference",
        desc="Reference URL for Sam's Club guest policy",
        parent=guest_block,
        critical=True
    )

    guest_leaf = evaluator.add_leaf(
        id="guest_limit_number",
        desc="State the number of guests members can bring per visit",
        parent=guest_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"Sam's Club members can bring up to {guest_limit} guest(s) per visit.",
        node=guest_leaf,
        sources=_urls_or_none(guest_sources),
        additional_instruction="Confirm the guest policy limit per visit for Sam's Club membership."
    )

    # Cash back percentage
    cash_block = evaluator.add_parallel(
        id="cash_back_percentage",
        desc="Identify the Sam's Cash back percentage for Plus members",
        parent=area2_node,
        critical=False
    )
    cash = area2.cash_back if area2 else None
    percent = _safe_str(cash.percentage) if cash else ""
    max_reward = _safe_str(cash.max_annual_reward) if cash else ""
    cash_sources = cash.sources if cash else []

    evaluator.add_custom_node(
        result=_has_urls(cash_sources),
        id="cash_back_reference",
        desc="Reference URL for Sam's Cash rewards details",
        parent=cash_block,
        critical=True
    )

    cash_rate_leaf = evaluator.add_leaf(
        id="cash_back_rate",
        desc="State the percentage of Sam's Cash back that Plus members earn on qualifying purchases",
        parent=cash_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"Sam's Club Plus members earn {percent} Sam's Cash back on qualifying purchases.",
        node=cash_rate_leaf,
        sources=_urls_or_none(cash_sources),
        additional_instruction="Confirm the Sam's Cash earn rate for Plus membership on qualifying purchases."
    )

    max_reward_leaf = evaluator.add_leaf(
        id="maximum_annual_reward",
        desc="Note the maximum annual reward amount",
        parent=cash_block,
        critical=False
    )
    await evaluator.verify(
        claim=f"The maximum annual Sam's Cash reward for Plus members is {max_reward}.",
        node=max_reward_leaf,
        sources=_urls_or_none(cash_sources),
        additional_instruction="Confirm the annual cap/maximum Sam's Cash that can be earned by Plus members."
    )


async def build_area3_business_specs(evaluator: Evaluator, root_node, area3: Optional[Area3Extraction]) -> None:
    area3_node = evaluator.add_parallel(
        id="business_technology_specs",
        desc="Provide business laptop specifications and Best Buy store count information",
        parent=root_node,
        critical=False
    )

    # Laptop RAM requirement
    ram_block = evaluator.add_parallel(
        id="laptop_ram_requirement",
        desc="Identify minimum RAM for business laptops",
        parent=area3_node,
        critical=False
    )
    ram = area3.laptop_ram if area3 else None
    ram_min = _safe_str(ram.minimum_recommended) if ram else ""
    ram_sources = ram.sources if ram else []

    evaluator.add_custom_node(
        result=_has_urls(ram_sources),
        id="ram_reference",
        desc="Reference URL for business laptop RAM requirements",
        parent=ram_block,
        critical=True
    )

    ram_leaf = evaluator.add_leaf(
        id="ram_amount",
        desc="State the minimum RAM recommended for business/professional laptops in 2024-2025",
        parent=ram_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"A reputable 2024–2025 business laptop buying guide recommends at least {ram_min} of RAM for business/professional use.",
        node=ram_leaf,
        sources=_urls_or_none(ram_sources),
        additional_instruction="Accept phrasing like 'at least' or 'minimum' and values equal to or higher than the stated amount."
    )

    # Laptop processor requirement
    proc_block = evaluator.add_parallel(
        id="laptop_processor_requirement",
        desc="Identify minimum processor type for business laptops",
        parent=area3_node,
        critical=False
    )
    proc = area3.laptop_processor if area3 else None
    proc_min = _safe_str(proc.minimum_recommended) if proc else ""
    proc_sources = proc.sources if proc else []

    evaluator.add_custom_node(
        result=_has_urls(proc_sources),
        id="processor_reference",
        desc="Reference URL for business laptop processor requirements",
        parent=proc_block,
        critical=True
    )

    proc_leaf = evaluator.add_leaf(
        id="processor_type",
        desc="State the minimum processor type recommended (Intel Core or AMD Ryzen series)",
        parent=proc_block,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"A reputable 2024–2025 business laptop guide recommends a minimum processor such as {proc_min} "
            f"(e.g., Intel Core i5/i7 or AMD Ryzen 5/7 series) for professional use."
        ),
        node=proc_leaf,
        sources=_urls_or_none(proc_sources),
        additional_instruction="Confirm that the recommendation aligns with a modern Intel Core or AMD Ryzen series suitable for business."
    )

    # Best Buy store count
    store_block = evaluator.add_parallel(
        id="best_buy_store_count",
        desc="Provide the approximate number of Best Buy stores in the United States",
        parent=area3_node,
        critical=False
    )
    stores = area3.best_buy_store_count if area3 else None
    store_count = _safe_str(stores.approx_count) if stores else ""
    store_sources = stores.sources if stores else []

    evaluator.add_custom_node(
        result=_has_urls(store_sources),
        id="store_count_reference",
        desc="Reference URL for Best Buy store count data",
        parent=store_block,
        critical=True
    )

    store_leaf = evaluator.add_leaf(
        id="store_count_number",
        desc="State the approximate number of Best Buy stores operating in the United States",
        parent=store_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"Best Buy operates {store_count} stores (approximately) in the United States.",
        node=store_leaf,
        sources=_urls_or_none(store_sources),
        additional_instruction="Allow reasonable approximations and recent fluctuations; verify that the number aligns with the cited source."
    )


async def build_area4_product_availability(evaluator: Evaluator, root_node, area4: Optional[Area4Extraction]) -> None:
    area4_node = evaluator.add_parallel(
        id="product_availability",
        desc="Determine whether Crocs footwear is sold at Best Buy",
        parent=root_node,
        critical=False
    )

    ca = area4.crocs_availability if area4 else None
    crocs_answer = _safe_str(ca.crocs_at_best_buy) if ca else ""
    crocs_retailers = ca.crocs_retailers if ca else []
    crocs_sources = ca.sources if ca else []

    evaluator.add_custom_node(
        result=_has_urls(crocs_sources),
        id="crocs_availability_reference",
        desc="Reference URL for Crocs retail availability",
        parent=area4_node,
        critical=True
    )

    # Core yes/no verification (critical)
    crocs_leaf = evaluator.add_leaf(
        id="crocs_at_best_buy",
        desc="Verify whether Best Buy sells Crocs brand footwear (Yes or No)",
        parent=area4_node,
        critical=True
    )
    if crocs_answer.strip().lower() == "yes":
        claim_crocs = "Best Buy sells Crocs brand footwear."
    else:
        claim_crocs = "Best Buy does not sell Crocs brand footwear."
    await evaluator.verify(
        claim=claim_crocs,
        node=crocs_leaf,
        sources=_urls_or_none(crocs_sources),
        additional_instruction="Use official Best Buy site pages or credible sources. Treat the statement as false if sources are irrelevant."
    )

    # Non-critical: identify real retailers that sell Crocs
    retailers_leaf = evaluator.add_leaf(
        id="crocs_retailers",
        desc="Identify actual retailers that sell Crocs",
        parent=area4_node,
        critical=False
    )
    if crocs_retailers:
        claim_retailers = f"Crocs brand footwear is sold at at least one of the following retailers: {', '.join(crocs_retailers)}."
    else:
        claim_retailers = "Crocs brand footwear is sold by at least one well-known U.S. retailer."
    await evaluator.verify(
        claim=claim_retailers,
        node=retailers_leaf,
        sources=_urls_or_none(crocs_sources),
        additional_instruction="Confirm that at least one of the listed retailers in the answer actually sells Crocs."
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
    Evaluate an answer for the multi-area retail research task.
    """
    # Initialize evaluator (root as non-critical to avoid strict gating across areas)
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_full(),
        template_class=FullExtraction,
        extraction_name="retail_research_extraction",
    )

    # Build and run verification subtrees
    await build_area1_geographic(evaluator, root, extracted.area1)
    await build_area2_membership(evaluator, root, extracted.area2)
    await build_area3_business_specs(evaluator, root, extracted.area3)
    await build_area4_product_availability(evaluator, root, extracted.area4)

    # Return evaluator summary
    return evaluator.get_summary()