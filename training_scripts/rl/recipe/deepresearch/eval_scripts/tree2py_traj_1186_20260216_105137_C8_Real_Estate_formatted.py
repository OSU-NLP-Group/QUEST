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
TASK_ID = "reit_portfolio_2026"
TASK_DESCRIPTION = """As a real estate investment analyst, you are tasked with constructing a diversified REIT (Real Estate Investment Trust) portfolio for a client seeking exposure across multiple property sectors. The portfolio must consist of exactly 4 publicly-traded REITs, each from a different property sector, meeting the following criteria:

Industrial REIT Requirements:
- Must primarily invest in industrial properties including warehouses, distribution centers, or logistics facilities
- Must be ranked among the top 3 industrial REITs by market capitalization as of February 2026
- Provide the REIT name, ticker symbol, and current dividend yield

Healthcare REIT Requirements:
- Must primarily invest in healthcare properties including senior living facilities, hospitals, medical office buildings, or skilled nursing facilities
- Must be ranked among the top 3 healthcare REITs by market capitalization as of February 2026
- Provide the REIT name, ticker symbol, and current dividend yield

Residential/Multifamily REIT Requirements:
- Must primarily invest in multifamily residential properties including apartment buildings or residential communities
- Must report an occupancy rate of 95% or higher in its most recent fiscal year report
- Provide the REIT name, ticker symbol, and the reported occupancy rate

Specialized REIT Requirements:
- Must primarily invest in specialized properties such as data centers, cell towers, or telecommunications infrastructure
- Must be among the largest 5 REITs in its specialized property category by market capitalization as of February 2026
- Provide the REIT name, ticker symbol, and current dividend yield

For each of the 4 REITs, provide a reference URL confirming the REIT's sector classification, market capitalization ranking, and the requested financial metrics.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ReitBase(BaseModel):
    name: Optional[str] = None
    ticker: Optional[str] = None
    sector_desc: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class IndustrialReit(ReitBase):
    dividend_yield: Optional[str] = None


class HealthcareReit(ReitBase):
    dividend_yield: Optional[str] = None


class ResidentialReit(ReitBase):
    occupancy_rate: Optional[str] = None


class SpecializedReit(ReitBase):
    specialized_category: Optional[str] = None  # e.g., data center, cell tower, telecom infrastructure
    dividend_yield: Optional[str] = None


class ReitPortfolioExtraction(BaseModel):
    industrial: Optional[IndustrialReit] = None
    healthcare: Optional[HealthcareReit] = None
    residential: Optional[ResidentialReit] = None
    specialized: Optional[SpecializedReit] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_reit_portfolio() -> str:
    return """
Extract the 4 REITs selected in the answer—exactly one per sector: industrial, healthcare, residential/multifamily, and specialized (data centers, cell towers, or telecom infrastructure).

For each sector, extract the following from the answer exactly as written:
- name: The REIT's full name
- ticker: The stock ticker symbol (e.g., 'PLD', 'VTR'); if not provided, set to null
- sector_desc: A short phrase from the answer that describes the REIT’s primary property type focus (e.g., 'industrial logistics warehouses', 'healthcare medical office buildings', 'multifamily apartments')
- reference_urls: A list of ALL URLs that the answer cites for that REIT. Include any page that could support sector classification, market-cap ranking, and the requested metric (dividend yield or occupancy). If the answer includes multiple URLs for the REIT, include them all. If none are provided, return an empty list.

Additionally extract the sector-specific metric:
- industrial: dividend_yield (e.g., '2.8%')
- healthcare: dividend_yield (e.g., '5.1%')
- residential: occupancy_rate (e.g., '96%', '96.2%')
- specialized: specialized_category (e.g., 'data center', 'cell tower', 'telecom infrastructure') and dividend_yield (e.g., '2.0%')

Return a JSON object with this structure:
{
  "industrial": { "name": ..., "ticker": ..., "sector_desc": ..., "dividend_yield": ..., "reference_urls": [...] },
  "healthcare": { "name": ..., "ticker": ..., "sector_desc": ..., "dividend_yield": ..., "reference_urls": [...] },
  "residential": { "name": ..., "ticker": ..., "sector_desc": ..., "occupancy_rate": ..., "reference_urls": [...] },
  "specialized": { "name": ..., "ticker": ..., "sector_desc": ..., "specialized_category": ..., "dividend_yield": ..., "reference_urls": [...] }
}

Important:
- Do NOT invent any URLs; only extract URLs explicitly present in the answer text.
- If the answer includes more than one REIT for a sector, extract the FIRST one that appears to be included in the portfolio.
- If any field is missing, set it to null; if no URLs are provided, set 'reference_urls' to an empty list.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_str(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


# --------------------------------------------------------------------------- #
# Verification builders per sector                                            #
# --------------------------------------------------------------------------- #
async def verify_industrial(evaluator: Evaluator, parent_node, data: Optional[IndustrialReit]) -> None:
    node = evaluator.add_parallel(
        id="Industrial_REIT_Selection",
        desc="Identify one publicly-traded industrial REIT that meets all specified criteria for industrial property exposure",
        parent=parent_node,
        critical=False
    )

    name = data.name if data else None
    ticker = data.ticker if data else None
    urls = (data.reference_urls if data else []) or []
    dy = data.dividend_yield if data else None

    # Identification (name + ticker) - critical existence check
    evaluator.add_custom_node(
        result=_nonempty_str(name) and _nonempty_str(ticker),
        id="Industrial_Identification",
        desc="Provide the REIT name and ticker symbol for the identified industrial REIT",
        parent=node,
        critical=True
    )

    # Reference URL existence - critical
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="Industrial_Reference_URL",
        desc="Provide a reference URL confirming the REIT's sector classification, market cap ranking, and dividend yield",
        parent=node,
        critical=True
    )

    # Property type classification - critical, source-grounded
    pt_leaf = evaluator.add_leaf(
        id="Industrial_Property_Type",
        desc="The REIT must primarily invest in industrial properties such as warehouses, distribution centers, manufacturing facilities, or logistics facilities",
        parent=node,
        critical=True
    )
    claim_pt = f"{name} is an industrial REIT that primarily invests in industrial properties (e.g., warehouses, distribution centers, logistics facilities)."
    await evaluator.verify(
        claim=claim_pt,
        node=pt_leaf,
        sources=urls,
        additional_instruction="Confirm that the company is primarily an industrial REIT focused on logistics/warehouses/distribution. Accept clear statements from investor relations or reputable sources."
    )

    # Top rank by market cap (top 3) as of Feb 2026 - critical, source-grounded
    rank_leaf = evaluator.add_leaf(
        id="Industrial_Top_Rank",
        desc="The REIT must be ranked among the top 3 industrial REITs by market capitalization as of February 2026",
        parent=node,
        critical=True
    )
    claim_rank = f"As of February 2026, {name} is among the top 3 industrial REITs by market capitalization."
    await evaluator.verify(
        claim=claim_rank,
        node=rank_leaf,
        sources=urls,
        additional_instruction="Look for rankings, market cap comparisons, or lists around 2025–2026. Consider phrases like 'largest industrial REITs' by market cap. The page should clearly support that this REIT is within the top three."
    )

    # Dividend yield - critical, source-grounded
    dy_leaf = evaluator.add_leaf(
        id="Industrial_Dividend_Yield",
        desc="Provide the current dividend yield percentage of the identified industrial REIT",
        parent=node,
        critical=True
    )
    claim_dy = f"The current dividend yield of {name} ({ticker}) is {dy}."
    await evaluator.verify(
        claim=claim_dy,
        node=dy_leaf,
        sources=urls,
        additional_instruction="Verify that the page reports a dividend yield matching the stated value (allow minor rounding). Accept 'dividend yield' or equivalent phrasing such as distribution yield."
    )


async def verify_healthcare(evaluator: Evaluator, parent_node, data: Optional[HealthcareReit]) -> None:
    node = evaluator.add_parallel(
        id="Healthcare_REIT_Selection",
        desc="Identify one publicly-traded healthcare REIT that meets all specified criteria for healthcare property exposure",
        parent=parent_node,
        critical=False
    )

    name = data.name if data else None
    ticker = data.ticker if data else None
    urls = (data.reference_urls if data else []) or []
    dy = data.dividend_yield if data else None

    evaluator.add_custom_node(
        result=_nonempty_str(name) and _nonempty_str(ticker),
        id="Healthcare_Identification",
        desc="Provide the REIT name and ticker symbol for the identified healthcare REIT",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="Healthcare_Reference_URL",
        desc="Provide a reference URL confirming the REIT's sector classification, market cap ranking, and dividend yield",
        parent=node,
        critical=True
    )

    pt_leaf = evaluator.add_leaf(
        id="Healthcare_Property_Type",
        desc="The REIT must primarily invest in healthcare properties such as senior living facilities, hospitals, medical office buildings, or skilled nursing facilities",
        parent=node,
        critical=True
    )
    claim_pt = f"{name} is a healthcare REIT primarily investing in healthcare properties such as senior housing, hospitals, medical office buildings, or skilled nursing facilities."
    await evaluator.verify(
        claim=claim_pt,
        node=pt_leaf,
        sources=urls,
        additional_instruction="Confirm that the company is primarily focused on healthcare real estate (senior living, MOBs, hospitals, SNFs)."
    )

    rank_leaf = evaluator.add_leaf(
        id="Healthcare_Top_Rank",
        desc="The REIT must be ranked among the top 3 healthcare REITs by market capitalization as of February 2026",
        parent=node,
        critical=True
    )
    claim_rank = f"As of February 2026, {name} is among the top 3 healthcare REITs by market capitalization."
    await evaluator.verify(
        claim=claim_rank,
        node=rank_leaf,
        sources=urls,
        additional_instruction="Look for rankings or lists around 2025–2026 for healthcare REITs by market cap. The page should indicate that this REIT is in the top three."
    )

    dy_leaf = evaluator.add_leaf(
        id="Healthcare_Dividend_Yield",
        desc="Provide the current dividend yield percentage of the identified healthcare REIT",
        parent=node,
        critical=True
    )
    claim_dy = f"The current dividend yield of {name} ({ticker}) is {dy}."
    await evaluator.verify(
        claim=claim_dy,
        node=dy_leaf,
        sources=urls,
        additional_instruction="Verify that the page shows a dividend yield matching the stated figure (allow small rounding differences)."
    )


async def verify_residential(evaluator: Evaluator, parent_node, data: Optional[ResidentialReit]) -> None:
    node = evaluator.add_parallel(
        id="Residential_REIT_Selection",
        desc="Identify one publicly-traded residential/multifamily REIT that meets all specified criteria for residential property exposure",
        parent=parent_node,
        critical=False
    )

    name = data.name if data else None
    ticker = data.ticker if data else None
    urls = (data.reference_urls if data else []) or []
    occ = data.occupancy_rate if data else None

    evaluator.add_custom_node(
        result=_nonempty_str(name) and _nonempty_str(ticker),
        id="Residential_Identification",
        desc="Provide the REIT name and ticker symbol for the identified residential REIT",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="Residential_Reference_URL",
        desc="Provide a reference URL confirming the REIT's property focus and reported occupancy rate",
        parent=node,
        critical=True
    )

    pt_leaf = evaluator.add_leaf(
        id="Residential_Property_Type",
        desc="The REIT must primarily invest in multifamily residential properties such as apartment buildings or residential communities",
        parent=node,
        critical=True
    )
    claim_pt = f"{name} primarily invests in multifamily residential properties such as apartment buildings or residential communities."
    await evaluator.verify(
        claim=claim_pt,
        node=pt_leaf,
        sources=urls,
        additional_instruction="Confirm multifamily/apartment community focus from investor materials or reputable sources."
    )

    occ_leaf = evaluator.add_leaf(
        id="Residential_Occupancy_Requirement",
        desc="The REIT must report an occupancy rate of 95% or higher in its most recent fiscal year report, and the specific occupancy rate percentage must be provided",
        parent=node,
        critical=True
    )
    claim_occ = f"In its most recent fiscal year, {name} reported an occupancy rate of {occ}, which is at least 95%."
    await evaluator.verify(
        claim=claim_occ,
        node=occ_leaf,
        sources=urls,
        additional_instruction="Check the most recent annual or quarterly report summary for 'occupancy' or 'occupancy rate'. The reported rate must be >= 95%; accept minor rounding."
    )


async def verify_specialized(evaluator: Evaluator, parent_node, data: Optional[SpecializedReit]) -> None:
    node = evaluator.add_parallel(
        id="Specialized_REIT_Selection",
        desc="Identify one publicly-traded specialized REIT (data center, cell tower, or telecommunications infrastructure) that meets all specified criteria",
        parent=parent_node,
        critical=False
    )

    name = data.name if data else None
    ticker = data.ticker if data else None
    urls = (data.reference_urls if data else []) or []
    category = data.specialized_category if data else None
    dy = data.dividend_yield if data else None

    evaluator.add_custom_node(
        result=_nonempty_str(name) and _nonempty_str(ticker),
        id="Specialized_Identification",
        desc="Provide the REIT name and ticker symbol for the identified specialized REIT",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="Specialized_Reference_URL",
        desc="Provide a reference URL confirming the REIT's property focus, market position, and dividend yield",
        parent=node,
        critical=True
    )

    pt_leaf = evaluator.add_leaf(
        id="Specialized_Property_Type",
        desc="The REIT must primarily invest in specialized properties such as data centers, cell towers, or telecommunications infrastructure",
        parent=node,
        critical=True
    )
    category_text = category if _nonempty_str(category) else "its specialized property category (data centers, cell towers, or telecom infrastructure)"
    claim_pt = f"{name} primarily invests in {category_text}."
    await evaluator.verify(
        claim=claim_pt,
        node=pt_leaf,
        sources=urls,
        additional_instruction="Confirm that the REIT is specialized in data centers, cell towers, or telecom infrastructure; clear statements from investor relations or well-known profiles are acceptable."
    )

    pos_leaf = evaluator.add_leaf(
        id="Specialized_Market_Position",
        desc="The REIT must be among the largest 5 REITs in its specialized property category by market capitalization as of February 2026",
        parent=node,
        critical=True
    )
    claim_pos = f"As of February 2026, {name} is among the largest 5 REITs by market capitalization in the {category_text} category."
    await evaluator.verify(
        claim=claim_pos,
        node=pos_leaf,
        sources=urls,
        additional_instruction="Look for rankings by market cap in the specific specialized category (e.g., largest data center REITs). Evidence should indicate a top-5 position around 2025–2026."
    )

    dy_leaf = evaluator.add_leaf(
        id="Specialized_Dividend_Yield",
        desc="Provide the current dividend yield percentage of the identified specialized REIT",
        parent=node,
        critical=True
    )
    claim_dy = f"The current dividend yield of {name} ({ticker}) is {dy}."
    await evaluator.verify(
        claim=claim_dy,
        node=dy_leaf,
        sources=urls,
        additional_instruction="Verify that the page shows a dividend yield matching the stated value; small rounding differences are acceptable."
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

    # Extract structured data for the four REITs
    extracted = await evaluator.extract(
        prompt=prompt_extract_reit_portfolio(),
        template_class=ReitPortfolioExtraction,
        extraction_name="reit_portfolio_extraction"
    )

    # Create a portfolio aggregation node (non-critical to allow partial credit per sector)
    portfolio_node = evaluator.add_parallel(
        id="REIT_Portfolio_Construction",
        desc="Construct a diversified REIT portfolio by identifying 4 REITs from different property sectors, each meeting specific investment criteria",
        parent=root,
        critical=False
    )

    # Note on rubric-root criticality vs implementation constraints
    evaluator.add_custom_info(
        info={
            "rubric_root_marked_critical": True,
            "implementation_adjustment": "Top-level aggregation kept non-critical due to framework constraint that critical parents must have all critical children. Sector groups remain non-critical aggregators with critical leaves to enforce requirements."
        },
        info_type="rubric_adjustment",
        info_name="criticality_note"
    )

    # Build sector verifications
    await verify_industrial(evaluator, portfolio_node, extracted.industrial)
    await verify_healthcare(evaluator, portfolio_node, extracted.healthcare)
    await verify_residential(evaluator, portfolio_node, extracted.residential)
    await verify_specialized(evaluator, portfolio_node, extracted.specialized)

    return evaluator.get_summary()