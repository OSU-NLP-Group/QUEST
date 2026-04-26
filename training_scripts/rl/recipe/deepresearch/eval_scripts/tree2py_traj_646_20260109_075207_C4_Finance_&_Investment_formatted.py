import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "dual_spot_crypto_etfs_025_fee"
TASK_DESCRIPTION = """
Identify a major asset management firm that offers both a spot Bitcoin ETF and a spot Ethereum ETF, where both products charge an expense ratio of exactly 0.25%. Provide the official ticker symbols for both ETF products and include reference URLs that verify the fee structure for each product.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ETFInfo(BaseModel):
    """
    Information for a single ETF as claimed in the answer.
    """
    name: Optional[str] = None
    ticker: Optional[str] = None
    expense_ratio: Optional[str] = None  # Keep as string to allow variations like "0.25%" or "0.250%"
    fee_verification_urls: List[str] = Field(default_factory=list)  # URLs provided in the answer that verify the fee
    product_urls: List[str] = Field(default_factory=list)  # Optional product/fact sheet/prospectus URLs (if provided)


class FirmAndETFs(BaseModel):
    """
    Extracted structure for the firm and its Bitcoin/Ethereum spot ETFs.
    """
    firm_name: Optional[str] = None
    bitcoin: ETFInfo = Field(default_factory=ETFInfo)
    ethereum: ETFInfo = Field(default_factory=ETFInfo)


# --------------------------------------------------------------------------- #
# Extraction prompt builders                                                  #
# --------------------------------------------------------------------------- #
def prompt_extract_firm_and_etfs() -> str:
    return """
    Extract the firm and ETF details explicitly stated in the answer.

    You must extract:
    - firm_name: The identified asset management firm that offers both ETFs.
    - bitcoin:
        - name: The product name of the spot Bitcoin ETF if mentioned (e.g., "iShares Bitcoin Trust").
        - ticker: The official ticker symbol for the spot Bitcoin ETF (e.g., "IBIT").
        - expense_ratio: The stated expense ratio string for the spot Bitcoin ETF exactly as shown in the answer (e.g., "0.25%").
        - fee_verification_urls: A list of all URLs provided in the answer that explicitly verify or state the Bitcoin ETF fee/expense ratio.
        - product_urls: Any additional official product/fact sheet/prospectus page URLs for the Bitcoin ETF mentioned in the answer (if any).
    - ethereum:
        - name: The product name of the spot Ethereum ETF if mentioned.
        - ticker: The official ticker symbol for the spot Ethereum ETF (e.g., "ETHA" or similar).
        - expense_ratio: The stated expense ratio string for the spot Ethereum ETF exactly as shown in the answer (e.g., "0.25%").
        - fee_verification_urls: A list of all URLs provided in the answer that explicitly verify or state the Ethereum ETF fee/expense ratio.
        - product_urls: Any additional official product/fact sheet/prospectus page URLs for the Ethereum ETF mentioned in the answer (if any).

    Important notes:
    - Only extract URLs that appear explicitly in the answer text.
    - If a required field is not present in the answer, set it to null (or an empty list for URL arrays).
    - Do not invent or infer any information that is not explicitly provided in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def safe_nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def merge_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for url in lst:
            if not isinstance(url, str):
                continue
            u = url.strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_same_firm_checks(
    evaluator: Evaluator,
    parent,
    firm: FirmAndETFs,
) -> None:
    """
    Build and verify the "same firm for both ETFs" checks.
    Decomposed into:
      - firm_name_provided (custom, critical)
      - btc_issuer_supported (verify by bitcoin URLs, critical)
      - eth_issuer_supported (verify by ethereum URLs, critical)
    """
    same_firm_node = evaluator.add_parallel(
        id="same_firm_for_both_etfs",
        desc="Both the Bitcoin ETF and the Ethereum ETF listed are offered/issued by the same identified firm",
        parent=parent,
        critical=True,
    )

    # Ensure the firm is actually identified in the answer
    evaluator.add_custom_node(
        result=safe_nonempty(firm.firm_name),
        id="firm_name_provided",
        desc="The firm name is explicitly provided in the answer",
        parent=same_firm_node,
        critical=True
    )

    # Verify the BTC ETF is issued/offered by the identified firm (use BTC fee URLs as evidence)
    btc_issuer_node = evaluator.add_leaf(
        id="btc_issuer_supported",
        desc="The Bitcoin ETF is offered/issued by the identified firm",
        parent=same_firm_node,
        critical=True
    )
    firm_label = firm.firm_name or "the identified firm"
    btc_tkr = firm.bitcoin.ticker or "the stated Bitcoin ETF"
    btc_claim = f"The spot Bitcoin ETF with ticker '{btc_tkr}' is offered or issued by {firm_label}."
    await evaluator.verify(
        claim=btc_claim,
        node=btc_issuer_node,
        sources=merge_urls(firm.bitcoin.fee_verification_urls, firm.bitcoin.product_urls),
        additional_instruction=(
            "Use the provided URLs to check the ETF's issuer/sponsor/manager. "
            "If the brand (e.g., iShares) is clearly associated with the firm (e.g., BlackRock) on the page, "
            "consider that as the issuer being the identified firm."
        ),
    )

    # Verify the ETH ETF is issued/offered by the identified firm (use ETH fee URLs as evidence)
    eth_issuer_node = evaluator.add_leaf(
        id="eth_issuer_supported",
        desc="The Ethereum ETF is offered/issued by the identified firm",
        parent=same_firm_node,
        critical=True
    )
    eth_tkr = firm.ethereum.ticker or "the stated Ethereum ETF"
    eth_claim = f"The spot Ethereum ETF with ticker '{eth_tkr}' is offered or issued by {firm_label}."
    await evaluator.verify(
        claim=eth_claim,
        node=eth_issuer_node,
        sources=merge_urls(firm.ethereum.fee_verification_urls, firm.ethereum.product_urls),
        additional_instruction=(
            "Use the provided URLs to check the ETF's issuer/sponsor/manager. "
            "If the brand (e.g., iShares) is clearly associated with the firm (e.g., BlackRock) on the page, "
            "consider that as the issuer being the identified firm."
        ),
    )


async def build_etf_requirements(
    evaluator: Evaluator,
    parent,
    firm_name: Optional[str],
    kind: str,           # "Bitcoin" or "Ethereum"
    info: ETFInfo,
) -> None:
    """
    Build verification subtree for either the Bitcoin or Ethereum ETF.
    All children here are critical, and the group is critical.
    """
    node_id_prefix = kind.lower()
    group_node = evaluator.add_parallel(
        id=f"{node_id_prefix}_etf_requirements",
        desc=f"{kind} ETF meets all stated requirements (spot/not futures; 0.25% expense ratio; ticker; fee-verifying reference URL)",
        parent=parent,
        critical=True
    )

    # 1) Ticker provided (existence check)
    evaluator.add_custom_node(
        result=safe_nonempty(info.ticker),
        id=f"{node_id_prefix}_etf_ticker",
        desc=f"The official ticker symbol for the firm's spot {kind} ETF is provided",
        parent=group_node,
        critical=True
    )

    # 2) Expense ratio stated in the answer as exactly 0.25% (answer-level check)
    stated_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_etf_expense_ratio_stated_025",
        desc=f"The spot {kind} ETF’s expense ratio is stated as exactly 0.25% in the answer",
        parent=group_node,
        critical=True
    )
    firm_label = firm_name or "the identified firm"
    tkr_label = info.ticker or f"the {kind} ETF"
    stated_claim = (
        f"In the answer, the {firm_label} spot {kind} ETF with ticker '{tkr_label}' is explicitly stated to have an expense ratio of exactly 0.25%."
    )
    await evaluator.verify(
        claim=stated_claim,
        node=stated_leaf,
        additional_instruction=(
            "Judge only based on the provided answer text. Accept small format variants like '0.250%' or '0.25 percent' as exactly 0.25%."
        )
    )

    # 3) Fee verification URL(s) provided (existence)
    evaluator.add_custom_node(
        result=len(info.fee_verification_urls) > 0,
        id=f"{node_id_prefix}_etf_fee_url_provided",
        desc=f"A fee-verifying reference URL is provided for the {kind} ETF",
        parent=group_node,
        critical=True
    )

    # 4) Fee verified by provided URL(s)
    fee_verify_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_etf_fee_verification_supported",
        desc=f"A provided reference URL explicitly supports/verifies the {kind} ETF’s expense ratio as 0.25%",
        parent=group_node,
        critical=True
    )
    fee_claim = f"The spot {kind} ETF with ticker '{tkr_label}' has an expense ratio (fee) of exactly 0.25%."
    await evaluator.verify(
        claim=fee_claim,
        node=fee_verify_leaf,
        sources=info.fee_verification_urls,
        additional_instruction=(
            "Check the provided URL(s) to confirm the expense ratio is exactly 0.25%. "
            "Accept small textual variants like '0.25 percent' or '0.250%'."
        )
    )

    # 5) Is spot (not futures/strategy-based), verified by URLs (prefer fee URLs; also include product URLs if present)
    is_spot_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_etf_is_spot_not_futures",
        desc=f"The firm offers a spot {kind} ETF (explicitly not strategy-based or futures-based)",
        parent=group_node,
        critical=True
    )
    is_spot_claim = (
        f"The ETF with ticker '{tkr_label}' is a spot {kind} ETF (it invests directly in the underlying asset or "
        f"explicitly states 'spot'), and it is not a futures-based or purely strategy-based ETF."
    )
    await evaluator.verify(
        claim=is_spot_claim,
        node=is_spot_leaf,
        sources=merge_urls(info.fee_verification_urls, info.product_urls),
        additional_instruction=(
            "Look for explicit mentions of 'spot', 'physically backed', 'holds bitcoin/ether directly', or clearly "
            "invests directly in the asset. Reject if the evidence clearly indicates a futures-based or strategy-based ETF."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Entry point to evaluate an answer for the dual spot crypto ETF task.
    """
    # Initialize evaluator (framework root is always non-critical; we add a critical main node under it)
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
        prompt=prompt_extract_firm_and_etfs(),
        template_class=FirmAndETFs,
        extraction_name="firm_and_etfs_extraction",
    )

    # Create the main critical node (to mirror rubric "Root" critical requirement)
    main = evaluator.add_parallel(
        id="main_task_verification",
        desc="Identify a single firm with both a spot Bitcoin ETF and a spot Ethereum ETF, each charging exactly 0.25%, including tickers and fee-verifying URLs.",
        parent=root,
        critical=True
    )

    # Same firm checks
    await build_same_firm_checks(evaluator, main, extracted)

    # Bitcoin ETF requirement checks
    await build_etf_requirements(
        evaluator=evaluator,
        parent=main,
        firm_name=extracted.firm_name,
        kind="Bitcoin",
        info=extracted.bitcoin,
    )

    # Ethereum ETF requirement checks
    await build_etf_requirements(
        evaluator=evaluator,
        parent=main,
        firm_name=extracted.firm_name,
        kind="Ethereum",
        info=extracted.ethereum,
    )

    # Return evaluation summary
    return evaluator.get_summary()