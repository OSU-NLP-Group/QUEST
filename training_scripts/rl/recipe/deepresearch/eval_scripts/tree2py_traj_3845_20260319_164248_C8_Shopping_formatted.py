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
TASK_ID = "iphone16_retailer_eval"
TASK_DESCRIPTION = """
You are planning to purchase an iPhone 16 and want to choose a major U.S. electronics retailer that offers the most comprehensive shopping experience and benefits. Identify which major U.S. electronics retailer(s) meet ALL of the following criteria:

Product Availability Requirements:
1. Sells iPhone 16 models (standard, Plus, Pro, or Pro Max versions)
2. Offers carrier-unlocked iPhone 16 versions
3. Provides Buy Online, Pick Up In Store (BOPIS) service for iPhone purchases

Service Features Requirements:
4. Offers same-day delivery service for electronics
5. Provides technical support services (such as installation, repair, or customer support)
6. Offers an electronics trade-in program where customers can trade in old phones or tablets

Financial Options Requirements:
7. Offers a store-branded credit card
8. Provides 0% APR financing or payment plans for electronics purchases of $299 or more
9. Offers extended warranty or protection plans for electronics beyond manufacturer warranties

Store Policies Requirements:
10. Offers at least a 30-day return window for unopened electronics
11. Provides extended return periods for paid membership program members (beyond the standard return window)
12. Has at least 1,000 physical store locations in the United States

For the answer, the retailer(s) that meet all twelve criteria must provide reference URLs per category (Product Availability, Service Features, Financial Options, Store Policies).
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RetailerItem(BaseModel):
    name: Optional[str] = None
    product_urls: List[str] = Field(default_factory=list)
    service_urls: List[str] = Field(default_factory=list)
    financial_urls: List[str] = Field(default_factory=list)
    policy_urls: List[str] = Field(default_factory=list)


class RetailersExtraction(BaseModel):
    retailers: List[RetailerItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_retailers() -> str:
    return """
    Extract up to the first five (5) distinct major U.S. electronics retailers explicitly mentioned in the answer.
    For each retailer, extract:
    - name: the retailer's official name as written in the answer.
    - product_urls: all URLs in the answer that support Product Availability for iPhone 16 (e.g., iPhone 16/16 Plus/16 Pro/16 Pro Max product pages, carrier‑unlocked listings, or pages that show BOPIS/pickup for iPhone purchases).
    - service_urls: all URLs that support Service Features (same‑day delivery for electronics; technical support/installation/repair; trade‑in program for phones/tablets).
    - financial_urls: all URLs that support Financial Options (store‑branded credit card; 0% APR/monthly financing or payment plans for electronics purchases of $299 or more; extended warranty/protection plans beyond manufacturer warranties).
    - policy_urls: all URLs that support Store Policies (standard return window of at least 30 days for unopened electronics; extended returns for paid membership program members; at least 1,000 U.S. physical store locations).

    Rules:
    - Only extract URLs that are explicitly present in the answer text (including markdown links). Do not invent or infer URLs.
    - Group URLs under the most relevant category; if a URL is relevant to multiple categories, include it in each relevant category (duplication allowed).
    - If a field is missing, set it to null (for name) or an empty array (for URLs).
    - Return an object with a 'retailers' array (max length 5). Each item follows the RetailerItem schema.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_and_limit(urls: List[str], limit: int = 8) -> List[str]:
    seen = set()
    cleaned: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        if not (u.startswith("http://") or u.startswith("https://")):
            # Skip obviously malformed URLs; Extractor may also handle this.
            continue
        if u not in seen:
            seen.add(u)
            cleaned.append(u)
        if len(cleaned) >= limit:
            break
    return cleaned


def _nonempty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


# --------------------------------------------------------------------------- #
# Retailer verification builder                                               #
# --------------------------------------------------------------------------- #
async def build_and_verify_retailer(
    evaluator: Evaluator,
    parent: Any,
    retailer: RetailerItem,
    retailer_idx_1based: int,
) -> Any:
    """
    Build the verification subtree for a single retailer (index shown as 1..5),
    create all required leaf nodes, and dispatch verifications.
    """
    rid = retailer_idx_1based
    retailer_node = evaluator.add_parallel(
        id=f"Retailer_{rid}",
        desc=f"Evaluation of the {rid}{'st' if rid==1 else ('nd' if rid==2 else ('rd' if rid==3 else 'th'))} retailer mentioned.",
        parent=parent,
        critical=False,  # overall retailer node allows partial credit; critical checks are done under sub-nodes
    )

    # Normalize URLs (limit to avoid excessive calls)
    product_urls = _dedup_and_limit(retailer.product_urls)
    service_urls = _dedup_and_limit(retailer.service_urls)
    financial_urls = _dedup_and_limit(retailer.financial_urls)
    policy_urls = _dedup_and_limit(retailer.policy_urls)
    retailer_name = retailer.name or ""

    # 1) Retailer name existence (critical)
    evaluator.add_custom_node(
        result=_nonempty(retailer.name),
        id=f"Retailer_{rid}_Name",
        desc=f"The {rid}{'st' if rid==1 else ('nd' if rid==2 else ('rd' if rid==3 else 'th'))} retailer is explicitly named.",
        parent=retailer_node,
        critical=True,
    )

    # 2) Product Availability (critical group)
    product_node = evaluator.add_parallel(
        id=f"Retailer_{rid}_Product_Availability",
        desc=f"Product Availability requirements for the {rid}{'st' if rid==1 else ('nd' if rid==2 else ('rd' if rid==3 else 'th'))} retailer, with supporting URL(s).",
        parent=retailer_node,
        critical=True,
    )

    # Category URLs presence (critical)
    evaluator.add_custom_node(
        result=len(product_urls) > 0,
        id=f"Retailer_{rid}_Product_Reference_URLs",
        desc="Provides reference URL(s) supporting the Product Availability claims.",
        parent=product_node,
        critical=True,
    )

    leaf_prod_iphone = evaluator.add_leaf(
        id=f"Retailer_{rid}_iPhone_16_Available",
        desc="Sells iPhone 16 models (standard, Plus, Pro, or Pro Max).",
        parent=product_node,
        critical=True,
    )
    leaf_prod_unlocked = evaluator.add_leaf(
        id=f"Retailer_{rid}_Unlocked_Versions",
        desc="Offers carrier-unlocked iPhone 16 versions.",
        parent=product_node,
        critical=True,
    )
    leaf_prod_bopis = evaluator.add_leaf(
        id=f"Retailer_{rid}_BOPIS_Service",
        desc="Provides Buy Online, Pick Up In Store (BOPIS) for iPhone purchases.",
        parent=product_node,
        critical=True,
    )

    # 3) Service Features (critical group)
    service_node = evaluator.add_parallel(
        id=f"Retailer_{rid}_Service_Features",
        desc=f"Service Features requirements for the {rid}{'st' if rid==1 else ('nd' if rid==2 else ('rd' if rid==3 else 'th'))} retailer, with supporting URL(s).",
        parent=retailer_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(service_urls) > 0,
        id=f"Retailer_{rid}_Service_Reference_URLs",
        desc="Provides reference URL(s) supporting the Service Features claims.",
        parent=service_node,
        critical=True,
    )

    leaf_serv_same_day = evaluator.add_leaf(
        id=f"Retailer_{rid}_Same_Day_Delivery",
        desc="Offers same-day delivery service for electronics.",
        parent=service_node,
        critical=True,
    )
    leaf_serv_tech = evaluator.add_leaf(
        id=f"Retailer_{rid}_Technical_Support",
        desc="Provides technical support services (installation, repair, or customer support).",
        parent=service_node,
        critical=True,
    )
    leaf_serv_tradein = evaluator.add_leaf(
        id=f"Retailer_{rid}_Trade_In_Program",
        desc="Offers an electronics trade-in program for phones/tablets.",
        parent=service_node,
        critical=True,
    )

    # 4) Financial Options (critical group)
    financial_node = evaluator.add_parallel(
        id=f"Retailer_{rid}_Financial_Options",
        desc=f"Financial Options requirements for the {rid}{'st' if rid==1 else ('nd' if rid==2 else ('rd' if rid==3 else 'th'))} retailer, with supporting URL(s).",
        parent=retailer_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(financial_urls) > 0,
        id=f"Retailer_{rid}_Financial_Reference_URLs",
        desc="Provides reference URL(s) supporting the Financial Options claims.",
        parent=financial_node,
        critical=True,
    )

    leaf_fin_card = evaluator.add_leaf(
        id=f"Retailer_{rid}_Store_Credit_Card",
        desc="Offers a store-branded credit card.",
        parent=financial_node,
        critical=True,
    )
    leaf_fin_financing = evaluator.add_leaf(
        id=f"Retailer_{rid}_Financing_Available",
        desc="Provides 0% APR financing or payment plans for electronics purchases of $299 or more.",
        parent=financial_node,
        critical=True,
    )
    leaf_fin_protection = evaluator.add_leaf(
        id=f"Retailer_{rid}_Protection_Plans",
        desc="Offers extended warranty/protection plans beyond manufacturer warranties.",
        parent=financial_node,
        critical=True,
    )

    # 5) Store Policies (critical group)
    policy_node = evaluator.add_parallel(
        id=f"Retailer_{rid}_Store_Policies",
        desc=f"Store Policies requirements for the {rid}{'st' if rid==1 else ('nd' if rid==2 else ('rd' if rid==3 else 'th'))} retailer, with supporting URL(s).",
        parent=retailer_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(policy_urls) > 0,
        id=f"Retailer_{rid}_Policy_Reference_URLs",
        desc="Provides reference URL(s) supporting the Store Policies claims.",
        parent=policy_node,
        critical=True,
    )

    leaf_pol_returns = evaluator.add_leaf(
        id=f"Retailer_{rid}_Return_Window",
        desc="Offers at least a 30-day return window for unopened electronics.",
        parent=policy_node,
        critical=True,
    )
    leaf_pol_members = evaluator.add_leaf(
        id=f"Retailer_{rid}_Extended_Returns_Members",
        desc="Provides extended return periods for paid membership program members beyond the standard return window.",
        parent=policy_node,
        critical=True,
    )
    leaf_pol_stores = evaluator.add_leaf(
        id=f"Retailer_{rid}_Physical_Stores",
        desc="Has at least 1,000 physical store locations in the United States.",
        parent=policy_node,
        critical=True,
    )

    # --------------------- Prepare batched verifications --------------------- #
    claims_and_sources: List[tuple] = []

    # Product claims
    claims_and_sources.append((
        f"The retailer {retailer_name!r} sells Apple iPhone 16 models (any of iPhone 16, 16 Plus, 16 Pro, or 16 Pro Max) that are available for purchase or pre-order.",
        product_urls,
        leaf_prod_iphone,
        "Check the provided page(s) for explicit mention of 'iPhone 16', 'iPhone 16 Plus', 'iPhone 16 Pro', or 'iPhone 16 Pro Max'. "
        "It should be a product listing, category page, or official retailer page clearly showing these models for sale or pre-order. "
        "Minor name variants are acceptable (e.g., 'Apple iPhone 16 Pro Max')."
    ))
    claims_and_sources.append((
        f"The retailer {retailer_name!r} offers carrier-unlocked (SIM-free/unlocked) iPhone 16 models.",
        product_urls,
        leaf_prod_unlocked,
        "Look for terms like 'unlocked', 'SIM-free', 'factory unlocked', or 'compatible with all major carriers' specifically for the iPhone 16 series. "
        "General unlocked statements are insufficient unless they clearly apply to iPhone 16."
    ))
    claims_and_sources.append((
        f"Buy Online, Pick Up In Store (BOPIS) is available for iPhone purchases at the retailer {retailer_name!r}, including iPhone 16 models when in stock.",
        product_urls,
        leaf_prod_bopis,
        "Accept evidence such as 'Pickup today', 'Store pickup available', 'Buy online, pick up in store', or policy pages that explicitly state BOPIS applies to phones/iPhones. "
        "It is okay if the page indicates pickup depends on store inventory."
    ))

    # Service claims
    claims_and_sources.append((
        f"The retailer {retailer_name!r} offers same-day delivery service for electronics purchases.",
        service_urls,
        leaf_serv_same_day,
        "Accept mentions of 'Same-day delivery', 'Delivery today', or third-party partner same-day services (e.g., Shipt, DoorDash, Instacart) as long as they apply to electronics."
    ))
    claims_and_sources.append((
        f"The retailer {retailer_name!r} provides technical support services such as installation, repair, or customer support for electronics.",
        service_urls,
        leaf_serv_tech,
        "Look for in-house or branded services (e.g., 'Geek Squad') or pages offering installation, repair, setup, or technical assistance for electronics."
    ))
    claims_and_sources.append((
        f"The retailer {retailer_name!r} offers an electronics trade-in program where customers can trade in old phones or tablets.",
        service_urls,
        leaf_serv_tradein,
        "The page should clearly describe a trade-in program for phones/tablets (quoting values, eligibility, or how to trade in)."
    ))

    # Financial claims
    claims_and_sources.append((
        f"The retailer {retailer_name!r} offers a store-branded credit card.",
        financial_urls,
        leaf_fin_card,
        "Look for a page that explicitly mentions the retailer's own store-branded credit card (including co-branded options)."
    ))
    claims_and_sources.append((
        f"The retailer {retailer_name!r} provides 0% APR financing or monthly payment plans for electronics purchases of $299 or more.",
        financial_urls,
        leaf_fin_financing,
        "Accept evidence of 0% APR or monthly installment/payment plans for electronics purchases at thresholds of $299 or higher (e.g., $299, $300). "
        "The threshold can be stated as 'for purchases $299+' or similar."
    ))
    claims_and_sources.append((
        f"The retailer {retailer_name!r} offers extended warranty or protection plans for electronics beyond manufacturer warranties.",
        financial_urls,
        leaf_fin_protection,
        "Look for 'protection plans', 'extended warranty', 'care plans', or similar offerings provided by the retailer or partner that go beyond the manufacturer warranty."
    ))

    # Policy claims
    claims_and_sources.append((
        f"The retailer {retailer_name!r} has a standard return policy of at least 30 days for unopened electronics.",
        policy_urls,
        leaf_pol_returns,
        "Check official return policy pages. Accept wording like '30 days' or longer for unopened electronics. "
        "It's okay if certain telecom-activated phones have shorter windows; the general consumer electronics return window should be at least 30 days."
    ))
    claims_and_sources.append((
        f"The retailer {retailer_name!r} provides extended return periods for paid membership program members beyond the standard return window.",
        policy_urls,
        leaf_pol_members,
        "Look for explicit statements that paid members (e.g., premium/plus/total members) get longer return periods than non-members."
    ))
    claims_and_sources.append((
        f"The retailer {retailer_name!r} operates at least 1,000 physical store locations in the United States.",
        policy_urls,
        leaf_pol_stores,
        "Accept statements like 'over 1,000 U.S. stores' or a store count ≥ 1,000 in the United States. "
        "If a global count is given, it must clearly indicate at least 1,000 in the U.S."
    ))

    # Dispatch verifications in parallel for this retailer
    await evaluator.batch_verify(claims_and_sources)

    return retailer_node


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
    Evaluate an answer for the iPhone 16 retailer task:
    - Extract up to 5 retailers with grouped reference URLs by category.
    - For each retailer, verify the 12 criteria using the cited webpages.
    - Also check that each category has at least one reference URL.
    - Finally, assert that at least one retailer meets all 12 criteria AND provides category URLs.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Keep root parallel to avoid sequential skip-on-partial issues
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

    # Top-level grouping node (use PARALLEL and non-critical to allow independent checks)
    top = evaluator.add_parallel(
        id="Retailer_Evaluation",
        desc="Evaluate the retailer(s) named in the answer against the 12 criteria, and check that the answer provides reference URLs per category.",
        parent=root,
        critical=False,  # Using non-critical to avoid child criticality constraint and allow partial credit
    )

    # Extract retailers and URLs
    extracted = await evaluator.extract(
        prompt=prompt_extract_retailers(),
        template_class=RetailersExtraction,
        extraction_name="retailers_extraction",
    )

    # Keep only the first 5 retailers
    retailers = (extracted.retailers or [])[:5]

    # Parent node for evaluating up to 5 retailers
    eval_retailers_parent = evaluator.add_parallel(
        id="Evaluate_Up_To_5_Listed_Retailers",
        desc="Independently evaluate up to five retailers mentioned in the answer (if fewer are listed, evaluate those provided).",
        parent=top,
        critical=False,
    )

    # Build and verify each retailer subtree
    retailer_nodes: List[Any] = []
    for idx, retailer in enumerate(retailers, start=1):
        node = await build_and_verify_retailer(evaluator, eval_retailers_parent, retailer, idx)
        retailer_nodes.append(node)

    # Compute whether at least one retailer fully meets all 12 criteria AND provided category URLs.
    # This equals the retailer node aggregating to 1.0 since all category subnodes are critical and include URL presence checks.
    at_least_one = False
    for n in retailer_nodes:
        try:
            if n.aggregated_score == 1.0:
                at_least_one = True
                break
        except Exception:
            # In case of any aggregation anomalies, treat as not fully satisfying
            pass

    evaluator.add_custom_node(
        result=at_least_one,
        id="At_Least_One_Retailer_Meets_All_12",
        desc="At least one evaluated retailer satisfies all 12 criteria and provides the required category reference URL(s).",
        parent=top,
        critical=True,
    )

    # Add auxiliary info to summary
    evaluator.add_custom_info(
        {
            "total_retailers_extracted": len(retailers),
            "retailer_ids_in_tree": [rn.id for rn in retailer_nodes],
        },
        info_type="stats",
        info_name="extraction_stats",
    )

    # Return summary
    return evaluator.get_summary()