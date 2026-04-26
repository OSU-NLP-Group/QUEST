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
TASK_ID = "gamestop_leadership"
TASK_DESCRIPTION = """
Identify the U.S. state where GameStop has the highest concentration of retail store locations. For this leading state, provide the following information: (1) the total number of GameStop stores in that state, (2) the city within that state that has the most GameStop store locations along with the specific store count for that city, (3) the location of GameStop's corporate headquarters, and (4) the typical square footage size of an average GameStop retail store. Each piece of information should be supported by reference URLs from your research.
"""

# Optional expected constraints (used for context in summary)
EXPECTED_CONSTRAINTS = {
    "expected_leading_state": "Texas",
    "expected_leading_state_store_count": "234",
    "expected_leading_state_share_percent": "≈11%",
    "expected_leading_city": "San Antonio, Texas",
    "expected_leading_city_store_count": "27",
    "expected_hq_location": "Grapevine, Texas",
    "expected_store_size_range": "1,500–1,700 sq ft",
    "expected_mall_store_avg": "≈1,200 sq ft",
    "expected_strip_center_store_avg": "≈1,500 sq ft",
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class LeadingState(BaseModel):
    name: Optional[str] = None
    total_stores: Optional[str] = None
    national_share_percent: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class LeadingCity(BaseModel):
    name: Optional[str] = None
    store_count: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class Headquarters(BaseModel):
    location: Optional[str] = None
    dallas_suburb_context: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class StoreSize(BaseModel):
    average_range_sqft: Optional[str] = None
    mall_based_avg_sqft: Optional[str] = None
    strip_center_avg_sqft: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class MarketLeadershipExtraction(BaseModel):
    leading_state: Optional[LeadingState] = None
    leading_city: Optional[LeadingCity] = None
    headquarters: Optional[Headquarters] = None
    store_size: Optional[StoreSize] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_market_leadership() -> str:
    return """
    Extract the following structured information from the answer. Return a JSON object with the following nested fields:

    leading_state:
      - name: The U.S. state identified as having the most GameStop store locations (e.g., "Texas"). If not explicitly identified, return null.
      - total_stores: The total number of GameStop stores in that leading state, as stated (e.g., "234"). Keep it as a string exactly as written in the answer; if not present, return null.
      - national_share_percent: The approximate national share this state's stores represent (e.g., "11%" or "approximately 11%"). Keep it as a string exactly as written; if not present, return null.
      - urls: An array of URLs cited in the answer that support the leading state identification and/or its figures. Extract only valid URLs. If none are provided, return an empty array.

    leading_city:
      - name: The city identified as having the most GameStop store locations in the United States (e.g., "San Antonio, Texas"). If not provided, return null.
      - store_count: The number of GameStop stores in that city (e.g., "27"). Keep it as a string; if not present, return null.
      - urls: An array of URLs cited that support the leading city identification and/or its store count. If none, return an empty array.

    headquarters:
      - location: GameStop's corporate headquarters location (e.g., "Grapevine, Texas"). If not present, return null.
      - dallas_suburb_context: The Dallas-suburb context statement if provided (e.g., "Grapevine is a suburb of Dallas"). If not present, return null.
      - urls: An array of URLs cited that support the headquarters claim. If none, return an empty array.

    store_size:
      - average_range_sqft: The typical/average store size range stated (e.g., "1,500–1,700 square feet"). Keep it as a string; if not present, return null.
      - mall_based_avg_sqft: The mall-based store size context (e.g., "around 1,200 square feet"). Keep it as a string; if not present, return null.
      - strip_center_avg_sqft: The strip shopping center store size context (e.g., "about 1,500 square feet"). Keep it as a string; if not present, return null.
      - urls: An array of URLs cited that support the typical/average store size claims (including context). If none, return an empty array.

    RULES:
    - Extract only what is explicitly in the answer. Do not invent values.
    - For URLs: extract the actual link strings (including from markdown). If a URL lacks protocol, prepend http://.
    - If a field is missing in the answer, return null for that field (or empty array for urls).
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_leading_state(
    evaluator: Evaluator,
    parent_node,
    extracted: MarketLeadershipExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Leading_State",
        desc="Checks the constrained leading U.S. state for GameStop store concentration and its associated constrained figures.",
        parent=parent_node,
        critical=True,
    )

    urls = extracted.leading_state.urls if (extracted.leading_state and extracted.leading_state.urls) else []

    # Leaf: Leading_State_Identification
    leaf_ident = evaluator.add_leaf(
        id="Leading_State_Identification",
        desc="Identifies Texas as the U.S. state with the most GameStop store locations.",
        parent=node,
        critical=True,
    )
    claim_ident = "Texas has the most GameStop store locations among U.S. states."
    await evaluator.verify(
        claim=claim_ident,
        node=leaf_ident,
        sources=urls,
        additional_instruction="Accept synonyms such as 'most stores', 'largest number of locations'. The URLs must directly support or clearly imply that Texas leads by store count.",
    )

    # Leaf: Leading_State_Store_Count
    leaf_count = evaluator.add_leaf(
        id="Leading_State_Store_Count",
        desc="Provides the total number of GameStop stores in Texas as 234.",
        parent=node,
        critical=True,
    )
    claim_count = "Texas has 234 GameStop stores."
    await evaluator.verify(
        claim=claim_count,
        node=leaf_count,
        sources=urls,
        additional_instruction="Verify that the cited source explicitly reports 234 stores for Texas (allowing phrasing variants like 'has 234 locations').",
    )

    # Leaf: Leading_State_National_Share
    leaf_share = evaluator.add_leaf(
        id="Leading_State_National_Share",
        desc="States that the 234 Texas stores represent approximately 11% of all GameStop stores nationwide.",
        parent=node,
        critical=True,
    )
    claim_share = "Texas's 234 GameStop stores represent approximately 11% of all GameStop stores nationwide."
    await evaluator.verify(
        claim=claim_share,
        node=leaf_share,
        sources=urls,
        additional_instruction="Treat 'approximately 11%' as tolerant (e.g., in the 10%–12% range). If the source provides total store count enabling inference that 234 is ≈11%, that counts as support.",
    )


async def verify_leading_city(
    evaluator: Evaluator,
    parent_node,
    extracted: MarketLeadershipExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Leading_City",
        desc="Checks the constrained leading city and store count.",
        parent=parent_node,
        critical=True,
    )

    urls = extracted.leading_city.urls if (extracted.leading_city and extracted.leading_city.urls) else []

    # Leaf: Leading_City_Identification
    leaf_city_ident = evaluator.add_leaf(
        id="Leading_City_Identification",
        desc="Identifies San Antonio, Texas as the city with the most GameStop store locations in the United States.",
        parent=node,
        critical=True,
    )
    claim_city_ident = "San Antonio, Texas has the most GameStop store locations in the United States."
    await evaluator.verify(
        claim=claim_city_ident,
        node=leaf_city_ident,
        sources=urls,
        additional_instruction="Accept equivalent phrasing indicating San Antonio leads all U.S. cities by GameStop store count.",
    )

    # Leaf: Leading_City_Store_Count
    leaf_city_count = evaluator.add_leaf(
        id="Leading_City_Store_Count",
        desc="Provides the number of GameStop stores in San Antonio as 27.",
        parent=node,
        critical=True,
    )
    claim_city_count = "San Antonio, Texas has 27 GameStop stores."
    await evaluator.verify(
        claim=claim_city_count,
        node=leaf_city_count,
        sources=urls,
        additional_instruction="Verify the cited source explicitly reports 27 stores (allow variants like '27 locations' or '27 retail stores').",
    )


async def verify_headquarters(
    evaluator: Evaluator,
    parent_node,
    extracted: MarketLeadershipExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Corporate_Headquarters",
        desc="Checks the constrained corporate headquarters location and required context.",
        parent=parent_node,
        critical=True,
    )

    urls = extracted.headquarters.urls if (extracted.headquarters and extracted.headquarters.urls) else []

    # Leaf: Headquarters_Location
    leaf_hq_loc = evaluator.add_leaf(
        id="Headquarters_Location",
        desc="Identifies GameStop's corporate headquarters as Grapevine, Texas.",
        parent=node,
        critical=True,
    )
    claim_hq_loc = "GameStop's corporate headquarters is located in Grapevine, Texas."
    await evaluator.verify(
        claim=claim_hq_loc,
        node=leaf_hq_loc,
        sources=urls,
        additional_instruction="Prefer official/company filings or reputable sources. Accept 'Grapevine, TX' equivalently.",
    )

    # Leaf: Headquarters_Context
    leaf_hq_ctx = evaluator.add_leaf(
        id="Headquarters_Context",
        desc="Notes that Grapevine is a suburb of Dallas (i.e., provides the required Dallas-suburb context).",
        parent=node,
        critical=True,
    )
    claim_hq_ctx = "Grapevine is a suburb of Dallas."
    await evaluator.verify(
        claim=claim_hq_ctx,
        node=leaf_hq_ctx,
        sources=urls,
        additional_instruction="Accept broader DFW metro context if the source explicitly refers to Grapevine as a Dallas suburb.",
    )


async def verify_store_size(
    evaluator: Evaluator,
    parent_node,
    extracted: MarketLeadershipExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Typical_Store_Size",
        desc="Checks the constrained typical/average store square-footage details.",
        parent=parent_node,
        critical=True,
    )

    urls = extracted.store_size.urls if (extracted.store_size and extracted.store_size.urls) else []

    # Leaf: Average_Store_Size_Range
    leaf_avg_range = evaluator.add_leaf(
        id="Average_Store_Size_Range",
        desc="States the typical/average GameStop store size as approximately 1,500–1,700 square feet.",
        parent=node,
        critical=True,
    )
    claim_avg_range = "The typical/average GameStop store size is approximately 1,500–1,700 square feet."
    await evaluator.verify(
        claim=claim_avg_range,
        node=leaf_avg_range,
        sources=urls,
        additional_instruction="Accept small wording variations (e.g., '~1,500 to 1,700 sq ft'), and allow approximations noted in the source.",
    )

    # Leaf: Mall_Based_Store_Size_Context
    leaf_mall_ctx = evaluator.add_leaf(
        id="Mall_Based_Store_Size_Context",
        desc="Provides the context that mall-based stores average around 1,200 square feet.",
        parent=node,
        critical=True,
    )
    claim_mall_ctx = "Mall-based GameStop stores average around 1,200 square feet."
    await evaluator.verify(
        claim=claim_mall_ctx,
        node=leaf_mall_ctx,
        sources=urls,
        additional_instruction="Accept 'around' or 'approximately' 1,200 sq ft when clearly indicated by the source and tied specifically to mall-based stores.",
    )

    # Leaf: Strip_Center_Store_Size_Context
    leaf_strip_ctx = evaluator.add_leaf(
        id="Strip_Center_Store_Size_Context",
        desc="Provides the context that strip shopping center stores average about 1,500 square feet.",
        parent=node,
        critical=True,
    )
    claim_strip_ctx = "Strip shopping center GameStop stores average about 1,500 square feet."
    await evaluator.verify(
        claim=claim_strip_ctx,
        node=leaf_strip_ctx,
        sources=urls,
        additional_instruction="Accept phrasing variants like 'about 1,500 sq ft' specifically tied to strip shopping center stores.",
    )


async def verify_citations_presence(
    evaluator: Evaluator,
    parent_node,
    extracted: MarketLeadershipExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Reference_URLs_Provided",
        desc="Ensures reference URLs are provided to support each required piece of information.",
        parent=parent_node,
        critical=True,
    )

    # Leading state citations presence
    has_state_urls = bool(extracted.leading_state and extracted.leading_state.urls and len(extracted.leading_state.urls) > 0)
    evaluator.add_custom_node(
        result=has_state_urls,
        id="Citations_For_Leading_State",
        desc="Provides at least one reference URL supporting the leading-state identification and/or its constrained figures.",
        parent=node,
        critical=True,
    )

    # Leading city citations presence
    has_city_urls = bool(extracted.leading_city and extracted.leading_city.urls and len(extracted.leading_city.urls) > 0)
    evaluator.add_custom_node(
        result=has_city_urls,
        id="Citations_For_Leading_City",
        desc="Provides at least one reference URL supporting the leading-city identification and its constrained store count.",
        parent=node,
        critical=True,
    )

    # Headquarters citations presence
    has_hq_urls = bool(extracted.headquarters and extracted.headquarters.urls and len(extracted.headquarters.urls) > 0)
    evaluator.add_custom_node(
        result=has_hq_urls,
        id="Citations_For_Headquarters",
        desc="Provides at least one reference URL supporting the headquarters location claim.",
        parent=node,
        critical=True,
    )

    # Store size citations presence
    has_size_urls = bool(extracted.store_size and extracted.store_size.urls and len(extracted.store_size.urls) > 0)
    evaluator.add_custom_node(
        result=has_size_urls,
        id="Citations_For_Store_Size",
        desc="Provides at least one reference URL supporting the typical/average store size claims (including the required context breakdowns).",
        parent=node,
        critical=True,
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
    Evaluate an answer for the GameStop market leadership information task.
    """
    # Initialize evaluator with a parallel root (aggregate across categories)
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_market_leadership(),
        template_class=MarketLeadershipExtraction,
        extraction_name="market_leadership_extraction",
    )

    # Record expected constraints in summary for transparency
    evaluator.add_ground_truth({
        "expected": EXPECTED_CONSTRAINTS,
        "note": "These are the constrained targets described by the rubric. Verification checks use cited URLs."
    })

    # Build verification tree and execute checks (in parallel branches)
    await verify_leading_state(evaluator, root, extracted)
    await verify_leading_city(evaluator, root, extracted)
    await verify_headquarters(evaluator, root, extracted)
    await verify_store_size(evaluator, root, extracted)
    await verify_citations_presence(evaluator, root, extracted)

    # Return standardized evaluation summary
    return evaluator.get_summary()