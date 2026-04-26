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
TASK_ID = "solana_etf_staking_2025_oct_nov"
TASK_DESCRIPTION = """
I am a single-filer investor planning my 2026 tax strategy and interested in Solana-based exchange-traded funds that offer staking rewards. I want to identify investment options that launched recently (between October and November 2025) and have competitive fee structures.

Please identify three U.S.-listed Solana ETFs that meet all of the following criteria:

1. Launch Date: The ETF must have launched between October 1, 2025 and November 30, 2025
2. Expense Ratio: The ETF must have an expense ratio of 0.30% or lower
3. Staking Capability: The ETF must offer staking functionality and aim to stake at least 50% of its Solana holdings
4. Staking Yield: The ETF must have publicly disclosed net or gross staking reward rates
5. U.S. Exchange: The ETF must be listed on a major U.S. stock exchange (such as NYSE Arca, Nasdaq, or Cboe)

For each of the three ETFs, provide:
- ETF name and ticker symbol
- Launch/inception date
- Expense ratio
- Staking details (percentage of assets staked and net/gross staking yield)
- Exchange listing
- Reference URL(s) for verification

Additionally, explain the 2026 tax implications for a single filer investing in these staking-enabled ETFs, specifically noting the income threshold for the 0% long-term capital gains tax rate and how staking rewards might factor into tax-advantaged investing. Include a reference URL for the 2026 tax bracket information.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ETFItem(BaseModel):
    name: Optional[str] = None
    ticker: Optional[str] = None
    launch_date: Optional[str] = None
    expense_ratio: Optional[str] = None
    staking_capability: Optional[str] = None  # e.g., "yes", "offers staking", or descriptive text
    staking_target: Optional[str] = None      # e.g., "≥50%", "at least 50%", "50-80%"
    staking_yield: Optional[str] = None       # e.g., "net 5%", "gross 6-7%"
    exchange: Optional[str] = None            # e.g., "NYSE Arca", "Nasdaq", "Cboe"
    sources: List[str] = Field(default_factory=list)


class ETFListExtraction(BaseModel):
    etfs: List[ETFItem] = Field(default_factory=list)


class TaxInfoExtraction(BaseModel):
    threshold_0_ltcg_single_2026: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    explanation_text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_etfs() -> str:
    return """
    Extract up to three Solana-focused U.S.-listed ETFs mentioned in the answer. For each ETF, return:
    - name: fund name as written
    - ticker: the ticker symbol as written
    - launch_date: the stated launch/inception/listing date text as written (do not reformat)
    - expense_ratio: the stated expense ratio text as written (include % sign if present)
    - staking_capability: copy the exact wording that indicates the ETF offers staking (e.g., "offers staking", "staking-enabled"); if not stated, set to null
    - staking_target: the stated target/actual share of SOL staked (e.g., "≥50%", "at least 50%", "50-80%"); if not stated, set to null
    - staking_yield: the stated net or gross staking reward rate (a number, range, or phrase like "gross 6-7%"); if not stated, set to null
    - exchange: the U.S. exchange name (e.g., "NYSE Arca", "Nasdaq", "Cboe"), as written
    - sources: a list of all reference URLs the answer provided for this ETF (fund website pages, press releases, prospectus, exchange listing page, filings, etc.)
    
    Rules:
    - Only include ETFs that are clearly Solana-based and U.S.-listed.
    - If the answer mentions more than three, include only the first three in the same order.
    - For any missing info for an ETF, set that field to null, but still include the ETF object.
    - Extract URLs exactly as written (full URLs). Ignore malformed URLs.
    """


def prompt_extract_tax_info() -> str:
    return """
    Extract the 2026 U.S. federal long-term capital gains information for single filers as stated in the answer:
    - threshold_0_ltcg_single_2026: the 0% long-term capital gains bracket upper threshold amount for single filers in 2026, exactly as written in the answer (include currency symbol and commas if present)
    - reference_urls: list all URLs provided that support 2026 federal tax bracket information
    - explanation_text: copy the sentence(s) from the answer that explain how staking yields/rewards and these ETFs relate to tax-advantaged investing for a single filer; if no such explanation, set to null
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_any_url(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    # basic sanity check for URL-like strings
    return any(isinstance(u, str) and ("http://" in u or "https://" in u) for u in urls)


def _safe(s: Optional[str]) -> str:
    return s or ""


# --------------------------------------------------------------------------- #
# Verification for one ETF                                                    #
# --------------------------------------------------------------------------- #
async def verify_single_etf(
    evaluator: Evaluator,
    root_node,
    etf: ETFItem,
    etf_index: int,
) -> None:
    """
    Build verification sub-tree for a single ETF.
    etf_index is 1-based (1, 2, 3) to match rubric node ids.
    """
    etf_node = evaluator.add_parallel(
        id=f"etf_{etf_index}",
        desc=f"{['First','Second','Third'][etf_index-1]} qualifying Solana ETF with all required attributes",
        parent=root_node,
        critical=False,
    )

    # -------------------- Basic Info (critical) -------------------- #
    basic_info = evaluator.add_parallel(
        id=f"etf_{etf_index}_basic_info",
        desc="Basic ETF information verified",
        parent=etf_node,
        critical=True,
    )

    # Identification sub-node (critical)
    identification_grp = evaluator.add_parallel(
        id=f"etf_{etf_index}_identification",
        desc="ETF name and ticker symbol correctly identified",
        parent=basic_info,
        critical=True,
    )

    # Leaf: identification matches (name + ticker)
    ident_leaf = evaluator.add_leaf(
        id=f"etf_{etf_index}_identification_match",
        desc="ETF name and ticker symbol match the cited source(s)",
        parent=identification_grp,
        critical=True,
    )
    ident_claim = (
        f"The ETF with ticker '{_safe(etf.ticker)}' is named '{_safe(etf.name)}'. "
        f"Confirm that the cited page(s) show both this fund name and ticker."
    )
    await evaluator.verify(
        claim=ident_claim,
        node=ident_leaf,
        sources=etf.sources,
        additional_instruction=(
            "Verify the official fund name and ticker symbol on the cited page(s). "
            "Allow minor punctuation/casing differences in the name (e.g., 'ETF' vs 'Trust')."
        ),
    )

    # Custom: reference provided for identification
    evaluator.add_custom_node(
        result=_has_any_url(etf.sources),
        id=f"etf_{etf_index}_id_reference",
        desc="Reference URL provided for ETF identification",
        parent=identification_grp,
        critical=True,
    )

    # Launch date sub-node (critical)
    launch_grp = evaluator.add_parallel(
        id=f"etf_{etf_index}_launch_date",
        desc="Launch date falls within October 1 - November 30, 2025",
        parent=basic_info,
        critical=True,
    )

    # Leaf: launch date in range (verified by sources)
    launch_leaf = evaluator.add_leaf(
        id=f"etf_{etf_index}_launch_in_range",
        desc="Launch/inception date is within Oct 1–Nov 30, 2025",
        parent=launch_grp,
        critical=True,
    )
    launch_claim = (
        f"The ETF '{_safe(etf.name)}' (ticker '{_safe(etf.ticker)}') launched (or had its inception/listing) on '{_safe(etf.launch_date)}', "
        f"and this date falls between October 1, 2025 and November 30, 2025."
    )
    await evaluator.verify(
        claim=launch_claim,
        node=launch_leaf,
        sources=etf.sources,
        additional_instruction=(
            "Confirm the specific launch/inception/listing date on the cited page(s). "
            "Treat 'inception date' or 'listing date' as equivalent to launch. "
            "Verify that the date is within 2025-10-01 through 2025-11-30 inclusive."
        ),
    )

    evaluator.add_custom_node(
        result=_has_any_url(etf.sources),
        id=f"etf_{etf_index}_launch_reference",
        desc="Reference URL provided for launch date verification",
        parent=launch_grp,
        critical=True,
    )

    # Exchange listing sub-node (critical)
    exchange_grp = evaluator.add_parallel(
        id=f"etf_{etf_index}_exchange_listing",
        desc="Listed on a major U.S. exchange (NYSE Arca, Nasdaq, or Cboe)",
        parent=basic_info,
        critical=True,
    )

    exchange_leaf = evaluator.add_leaf(
        id=f"etf_{etf_index}_exchange_listed_major",
        desc="Exchange is NYSE Arca, Nasdaq, or Cboe",
        parent=exchange_grp,
        critical=True,
    )
    exchange_claim = (
        f"The ETF '{_safe(etf.name)}' (ticker '{_safe(etf.ticker)}') is listed on '{_safe(etf.exchange)}', "
        f"which is a major U.S. exchange (NYSE Arca, Nasdaq, or Cboe)."
    )
    await evaluator.verify(
        claim=exchange_claim,
        node=exchange_leaf,
        sources=etf.sources,
        additional_instruction=(
            "Confirm the listing exchange on the cited page(s). "
            "Accept only NYSE Arca, Nasdaq, or Cboe (including specific market centers like Cboe BZX)."
        ),
    )

    evaluator.add_custom_node(
        result=_has_any_url(etf.sources),
        id=f"etf_{etf_index}_exchange_reference",
        desc="Reference URL provided for exchange listing verification",
        parent=exchange_grp,
        critical=True,
    )

    # -------------------- Fee Structure (critical) -------------------- #
    fee_grp = evaluator.add_parallel(
        id=f"etf_{etf_index}_fee_structure",
        desc="Fee structure verified to meet criteria",
        parent=etf_node,
        critical=True,
    )

    expense_grp = evaluator.add_parallel(
        id=f"etf_{etf_index}_expense_ratio",
        desc="Expense ratio is 0.30% or lower",
        parent=fee_grp,
        critical=True,
    )

    expense_leaf = evaluator.add_leaf(
        id=f"etf_{etf_index}_expense_value_threshold",
        desc="Expense ratio equals the stated value and is ≤ 0.30%",
        parent=expense_grp,
        critical=True,
    )
    expense_claim = (
        f"The ETF '{_safe(etf.name)}' (ticker '{_safe(etf.ticker)}') has an expense ratio of '{_safe(etf.expense_ratio)}', "
        f"and this is 0.30% or lower."
    )
    await evaluator.verify(
        claim=expense_claim,
        node=expense_leaf,
        sources=etf.sources,
        additional_instruction=(
            "Locate the expense ratio (sometimes called 'net expense ratio', 'unitary fee', or 'management fee') on the cited page(s). "
            "Check that the stated value matches and is ≤ 0.30%. Allow minor formatting differences (e.g., 0.3% vs 0.30%)."
        ),
    )

    evaluator.add_custom_node(
        result=_has_any_url(etf.sources),
        id=f"etf_{etf_index}_expense_reference",
        desc="Reference URL provided for expense ratio verification",
        parent=expense_grp,
        critical=True,
    )

    # -------------------- Staking Features (critical) -------------------- #
    staking_grp = evaluator.add_parallel(
        id=f"etf_{etf_index}_staking_features",
        desc="Staking functionality and performance verified",
        parent=etf_node,
        critical=True,
    )

    staking_cap_grp = evaluator.add_parallel(
        id=f"etf_{etf_index}_staking_capability",
        desc="ETF offers staking functionality",
        parent=staking_grp,
        critical=True,
    )

    staking_cap_leaf = evaluator.add_leaf(
        id=f"etf_{etf_index}_staking_capability_supported",
        desc="Sources confirm the ETF offers SOL staking functionality",
        parent=staking_cap_grp,
        critical=True,
    )
    staking_cap_claim = (
        f"The ETF '{_safe(etf.name)}' (ticker '{_safe(etf.ticker)}') offers staking functionality "
        f"for its Solana holdings as indicated by: '{_safe(etf.staking_capability)}'."
    )
    await evaluator.verify(
        claim=staking_cap_claim,
        node=staking_cap_leaf,
        sources=etf.sources,
        additional_instruction=(
            "Confirm that the fund's materials explicitly mention staking of Solana (participation in staking/validators, "
            "earning staking rewards, or similar)."
        ),
    )

    evaluator.add_custom_node(
        result=_has_any_url(etf.sources),
        id=f"etf_{etf_index}_staking_reference",
        desc="Reference URL provided for staking details verification",
        parent=staking_cap_grp,
        critical=True,
    )

    # Leaf: staking target ≥ 50%
    staking_target_leaf = evaluator.add_leaf(
        id=f"etf_{etf_index}_staking_target",
        desc="ETF stakes at least 50% of assets",
        parent=staking_grp,
        critical=True,
    )
    staking_target_claim = (
        f"The ETF '{_safe(etf.name)}' (ticker '{_safe(etf.ticker)}') aims to stake at least 50% of its Solana assets, "
        f"as indicated by: '{_safe(etf.staking_target)}'."
    )
    await evaluator.verify(
        claim=staking_target_claim,
        node=staking_target_leaf,
        sources=etf.sources,
        additional_instruction=(
            "Confirm that the cited page(s) indicate a target or policy to stake ≥50% of the fund's Solana holdings. "
            "Accept equivalents like 'at least half', '≥50%', 'majority (explicitly ≥50%)', or ranges where the lower bound is ≥50%."
        ),
    )

    # Leaf: documented net or gross staking reward rate
    staking_yield_leaf = evaluator.add_leaf(
        id=f"etf_{etf_index}_staking_yield",
        desc="Documented net or gross staking reward rate provided",
        parent=staking_grp,
        critical=True,
    )
    staking_yield_claim = (
        f"The ETF '{_safe(etf.name)}' (ticker '{_safe(etf.ticker)}') has publicly disclosed a net or gross staking reward rate "
        f"of '{_safe(etf.staking_yield)}'."
    )
    await evaluator.verify(
        claim=staking_yield_claim,
        node=staking_yield_leaf,
        sources=etf.sources,
        additional_instruction=(
            "Verify that the cited page(s) explicitly mention a net or gross staking reward rate (can be a number or range). "
            "If only ranges or qualified estimates are provided, that still counts as documented."
        ),
    )


# --------------------------------------------------------------------------- #
# Verification for tax context                                                #
# --------------------------------------------------------------------------- #
async def verify_tax_context(
    evaluator: Evaluator,
    root_node,
    tax_info: TaxInfoExtraction,
) -> None:
    tax_node = evaluator.add_parallel(
        id="tax_context",
        desc="Explanation of 2026 tax implications for single filers with these investments",
        parent=root_node,
        critical=False,
    )

    # Capital gains threshold (critical)
    cap_grp = evaluator.add_parallel(
        id="capital_gains_threshold",
        desc="2026 capital gains tax information provided",
        parent=tax_node,
        critical=True,
    )

    # Leaf: verify threshold value using provided reference URL(s)
    cap_rate_leaf = evaluator.add_leaf(
        id="capital_gains_rate",
        desc="Correct 0% long-term capital gains rate threshold for single filers in 2026 is stated and supported",
        parent=cap_grp,
        critical=True,
    )
    stated_threshold = _safe(tax_info.threshold_0_ltcg_single_2026)
    cap_rate_claim = (
        f"For 2026, the 0% long-term capital gains threshold for single filers is stated as '{stated_threshold}'. "
        f"Check the cited tax reference that this value matches the official 2026 threshold."
    )
    await evaluator.verify(
        claim=cap_rate_claim,
        node=cap_rate_leaf,
        sources=tax_info.reference_urls,
        additional_instruction=(
            "Verify against the authoritative tax bracket page(s) for 2026 that the provided threshold equals the official 0% LTCG "
            "upper limit for single filers. Allow minor formatting differences like $49,450 vs 49,450."
        ),
    )

    # Custom: reference URL provided for tax verification
    evaluator.add_custom_node(
        result=_has_any_url(tax_info.reference_urls),
        id="tax_reference",
        desc="Reference URL provided for 2026 tax bracket verification",
        parent=cap_grp,
        critical=True,
    )

    # Non-critical: Ensure the answer explains tax-advantaged context for staking-enabled ETFs
    invest_suit_leaf = evaluator.add_leaf(
        id="investment_suitability",
        desc="Explanation of how staking yields and ETF features relate to tax-advantaged investing",
        parent=tax_node,
        critical=False,
    )
    suitability_claim = (
        "The answer explains how staking rewards from these ETFs interact with tax-advantaged investing for a single filer—"
        "for example, noting that staking rewards may be treated as ordinary income when received, that long-term capital gains "
        "rates apply on fund share sales, and that using tax-advantaged accounts (e.g., IRAs) can impact taxation."
    )
    await evaluator.verify(
        claim=suitability_claim,
        node=invest_suit_leaf,
        additional_instruction=(
            "Judge this by reading the provided answer text: confirm that it includes a coherent explanation connecting staking rewards "
            "and ETF features to tax-advantaged investing for a single filer in 2026."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Solana ETFs with staking and 2026 tax context task.
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
        default_model=model,
    )

    # Extract ETF list and tax info (can run sequentially; parallelization optional)
    etf_list: ETFListExtraction = await evaluator.extract(
        prompt=prompt_extract_etfs(),
        template_class=ETFListExtraction,
        extraction_name="etf_list",
    )

    tax_info: TaxInfoExtraction = await evaluator.extract(
        prompt=prompt_extract_tax_info(),
        template_class=TaxInfoExtraction,
        extraction_name="tax_info",
    )

    # Normalize to exactly three ETFs (pad with empty entries if fewer)
    etfs = list(etf_list.etfs[:3])
    while len(etfs) < 3:
        etfs.append(ETFItem())

    # Build verification sub-trees for each ETF
    for idx, etf in enumerate(etfs, start=1):
        await verify_single_etf(evaluator, root, etf, idx)

    # Verify tax context
    await verify_tax_context(evaluator, root, tax_info)

    # Optional: record constraints as ground truth/context (not used for scoring)
    evaluator.add_ground_truth({
        "constraints": {
            "launch_window": "2025-10-01 to 2025-11-30",
            "max_expense_ratio": "0.30%",
            "staking_target_min": "≥50%",
            "requires_staking_yield": True,
            "us_major_exchanges": ["NYSE Arca", "Nasdaq", "Cboe"],
            "tax_year": 2026,
            "filer_status": "single",
            "ltcg_bracket_checked": "0% threshold"
        }
    }, gt_type="task_constraints")

    return evaluator.get_summary()