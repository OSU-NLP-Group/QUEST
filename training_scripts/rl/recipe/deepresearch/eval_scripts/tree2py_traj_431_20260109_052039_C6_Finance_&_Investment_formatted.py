import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tech_etf_selection"
TASK_DESCRIPTION = (
    "An investor is seeking to add a technology sector exchange-traded fund (ETF) to their portfolio with specific "
    "characteristics to ensure cost efficiency, adequate diversification, and reasonable concentration levels. Identify "
    "one technology sector ETF that satisfies ALL of the following requirements: (1) Classified within the Technology "
    "sector (Information Technology or related technology subsectors such as Semiconductors); (2) Expense ratio of "
    "0.35% or lower; (3) Portfolio containing at least 100 individual securities; (4) Top 3 holdings representing no "
    "more than 45% of total fund assets combined; (5) Assets under management (AUM) of at least $15 billion; (6) 30-day "
    "SEC yield or 12-month trailing yield of at least 0.10%. Provide the ETF's ticker symbol, full name, fund provider, "
    "and specific verified values for each criterion (expense ratio, holdings count, top 3 concentration percentage, "
    "AUM, and yield), along with the official provider website URL as the primary reference source."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ETFIdentification(BaseModel):
    """Identification fields for the ETF."""
    ticker: Optional[str] = None
    name: Optional[str] = None
    provider: Optional[str] = None
    provider_url: Optional[str] = None


class ETFMetrics(BaseModel):
    """Metric values and classification reported in the answer."""
    sector_classification: Optional[str] = None  # e.g., "Information Technology", "Semiconductors"
    expense_ratio: Optional[str] = None          # e.g., "0.10%"
    holdings_count: Optional[str] = None         # e.g., "120"
    top3_weights: List[str] = Field(default_factory=list)  # e.g., ["12.5%", "10.3%", "8.9%"]
    top3_combined_pct: Optional[str] = None      # e.g., "31.7%"
    aum: Optional[str] = None                    # e.g., "$45B"
    yield_value: Optional[str] = None            # e.g., "0.20%"
    yield_type: Optional[str] = None             # e.g., "30-day SEC" or "12-month trailing"


class ETFExtraction(BaseModel):
    """Complete extraction schema for the ETF selection answer."""
    identification: Optional[ETFIdentification] = None
    metrics: Optional[ETFMetrics] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_etf() -> str:
    return """
    Extract exactly one ETF (the chosen fund) and its reported values from the answer. Do NOT invent or compute values.
    Return a JSON object with two nested objects: `identification` and `metrics`.

    identification:
    - ticker: the ETF ticker symbol (as written)
    - name: the full ETF name (as written)
    - provider: the fund provider/issuer (as written)
    - provider_url: the official provider webpage URL referenced (full URL; if missing protocol, prepend http://)

    metrics:
    - sector_classification: the sector classification reported (e.g., "Information Technology", "Technology", "Semiconductors")
    - expense_ratio: the expense ratio as a string exactly as written (e.g., "0.10%")
    - holdings_count: the number of holdings as written (e.g., "120")
    - top3_weights: an array of up to three weight strings for the top 3 holdings, exactly as written (e.g., ["12.5%", "10.3%", "8.9%"]). If not listed individually, return an empty array.
    - top3_combined_pct: the reported combined percentage for the top 3 holdings, exactly as written (e.g., "31.7%"). If not explicitly stated, return null (do not compute).
    - aum: the Assets Under Management value as written, including units (e.g., "$45B", "$45 billion", "$45,000,000,000")
    - yield_value: the yield value as written (e.g., "0.20%")
    - yield_type: the yield type as written (e.g., "30-day SEC", "12-month trailing", "SEC yield"). If not specified, return null.

    Rules:
    - Extract ONLY what is explicitly present in the answer.
    - Use strings for all numeric values; do not convert types.
    - If a field is missing, return null. For arrays, return [] if not provided.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_str(x: Optional[str]) -> str:
    return x if (x is not None) else ""

def _parse_percent_to_float(p: Optional[str]) -> Optional[float]:
    """Parse a percentage string like '12.5%' or '12.5 %' to float 12.5."""
    if not p:
        return None
    s = p.strip().replace("%", "").replace(" ", "")
    try:
        return float(s)
    except Exception:
        return None

def _format_pct(val: Optional[float]) -> Optional[str]:
    if val is None:
        return None
    return f"{val:.2f}%"


def _compute_top3_combined(metrics: ETFMetrics) -> Optional[str]:
    """Compute combined top 3 weights from list if present; return string like '31.70%'."""
    if not metrics or not metrics.top3_weights or len(metrics.top3_weights) < 3:
        return None
    vals = [_parse_percent_to_float(w) for w in metrics.top3_weights[:3]]
    if any(v is None for v in vals):
        return None
    combined = sum(vals)  # Already in percent units
    return _format_pct(combined)


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def _verify_identification(
    evaluator: Evaluator,
    parent_node,
    ident: ETFIdentification
) -> Dict[str, Any]:
    """
    Add identification nodes: ticker, name, provider, provider_url presence checks
    and verify provider page corresponds to the ETF ticker & name.
    """
    # Presence checks (critical)
    ticker_node = evaluator.add_custom_node(
        result=bool(_safe_str(ident.ticker).strip()),
        id="etf_ticker_symbol",
        desc="Provide the ETF ticker symbol",
        parent=parent_node,
        critical=True
    )
    name_node = evaluator.add_custom_node(
        result=bool(_safe_str(ident.name).strip()),
        id="etf_full_name",
        desc="Provide the ETF full name",
        parent=parent_node,
        critical=True
    )
    provider_node = evaluator.add_custom_node(
        result=bool(_safe_str(ident.provider).strip()),
        id="fund_provider_identification",
        desc="Provide the fund provider/issuer name",
        parent=parent_node,
        critical=True
    )
    provider_url_exists_node = evaluator.add_custom_node(
        result=bool(_safe_str(ident.provider_url).strip()),
        id="official_provider_url",
        desc="Provide the official fund provider website URL as the primary reference source",
        parent=parent_node,
        critical=True
    )

    # Verify provider page matches ETF ticker and name (critical)
    match_page_leaf = evaluator.add_leaf(
        id="provider_page_matches_etf",
        desc="Provider page corresponds to the ETF ticker and full name",
        parent=parent_node,
        critical=True
    )
    claim = (
        f"This official provider webpage corresponds to the ETF with ticker '{_safe_str(ident.ticker)}' "
        f"and full name '{_safe_str(ident.name)}', issued by '{_safe_str(ident.provider)}'."
    )
    await evaluator.verify(
        claim=claim,
        node=match_page_leaf,
        sources=_safe_str(ident.provider_url) or None,
        additional_instruction="Verify the page clearly shows the ETF ticker and full fund name and indicates the provider/issuer.",
        extra_prerequisites=[provider_url_exists_node]
    )

    # Return useful node refs for prerequisites
    return {
        "provider_url_exists_node": provider_url_exists_node
    }


async def _verify_criteria_checks(
    evaluator: Evaluator,
    parent_node,
    metrics: ETFMetrics,
    provider_url: Optional[str],
    prereq_provider_url_node
) -> Dict[str, Any]:
    """
    Create 'criteria_checks' node and verify each requirement with existence + URL-backed checks.
    """
    criteria_node = evaluator.add_parallel(
        id="criteria_checks",
        desc="ETF meets all quantitative and classification constraints, with reported values",
        parent=parent_node,
        critical=True
    )

    # 1) Technology sector classification
    classification_seq = evaluator.add_sequential(
        id="technology_sector_classification",
        desc="Verify the ETF is classified within the Technology sector (Information Technology or technology-related subsectors such as Semiconductors)",
        parent=criteria_node,
        critical=True
    )
    classification_present = evaluator.add_custom_node(
        result=bool(_safe_str(metrics.sector_classification).strip()),
        id="classification_present",
        desc="Sector/classification value is provided",
        parent=classification_seq,
        critical=True
    )
    classification_verify = evaluator.add_leaf(
        id="classification_supported_by_provider",
        desc="Sector/classification is supported by official provider page and falls within Technology",
        parent=classification_seq,
        critical=True
    )
    classification_claim = (
        f"The ETF is classified within the Technology sector (including Information Technology or technology-related "
        f"subsectors such as Semiconductors). Reported classification: '{_safe_str(metrics.sector_classification)}'."
    )
    await evaluator.verify(
        claim=classification_claim,
        node=classification_verify,
        sources=_safe_str(provider_url) or None,
        additional_instruction=(
            "Accept synonyms and common naming variants such as 'Technology', 'Information Technology', "
            "'Tech', 'Semiconductors'. Confirm the provider's page clearly indicates a technology sector classification."
        ),
        extra_prerequisites=[prereq_provider_url_node, classification_present]
    )

    # 2) Expense ratio ≤ 0.35%
    expense_seq = evaluator.add_sequential(
        id="expense_ratio_requirement",
        desc="Provide the expense ratio value and verify it is ≤ 0.35%",
        parent=criteria_node,
        critical=True
    )
    expense_present = evaluator.add_custom_node(
        result=bool(_safe_str(metrics.expense_ratio).strip()),
        id="expense_ratio_present",
        desc="Expense ratio value is provided",
        parent=expense_seq,
        critical=True
    )
    expense_verify = evaluator.add_leaf(
        id="expense_ratio_verified",
        desc="Expense ratio is verified by provider page and ≤ 0.35%",
        parent=expense_seq,
        critical=True
    )
    expense_claim = (
        f"The ETF's expense ratio is '{_safe_str(metrics.expense_ratio)}' and it is less than or equal to 0.35%."
    )
    await evaluator.verify(
        claim=expense_claim,
        node=expense_verify,
        sources=_safe_str(provider_url) or None,
        additional_instruction="Check the 'Expense ratio' field on the provider page and allow minor rounding differences.",
        extra_prerequisites=[prereq_provider_url_node, expense_present]
    )

    # 3) Holdings count ≥ 100
    holdings_seq = evaluator.add_sequential(
        id="holdings_count_requirement",
        desc="Provide the holdings count and verify it is ≥ 100 securities",
        parent=criteria_node,
        critical=True
    )
    holdings_present = evaluator.add_custom_node(
        result=bool(_safe_str(metrics.holdings_count).strip()),
        id="holdings_count_present",
        desc="Holdings count value is provided",
        parent=holdings_seq,
        critical=True
    )
    holdings_verify = evaluator.add_leaf(
        id="holdings_count_verified",
        desc="Holdings count is verified by provider page and ≥ 100",
        parent=holdings_seq,
        critical=True
    )
    holdings_claim = (
        f"The ETF holds at least 100 individual securities. Reported holdings count: '{_safe_str(metrics.holdings_count)}'."
    )
    await evaluator.verify(
        claim=holdings_claim,
        node=holdings_verify,
        sources=_safe_str(provider_url) or None,
        additional_instruction=(
            "Confirm on the provider page that the fund holds ≥ 100 securities. If the count fluctuates daily, "
            "reasonable approximations or clearly stated ranges indicating ≥ 100 are acceptable."
        ),
        extra_prerequisites=[prereq_provider_url_node, holdings_present]
    )

    # 4) Top-3 combined weight ≤ 45%
    concentration_seq = evaluator.add_sequential(
        id="concentration_limit_requirement",
        desc="Provide the combined weight of the top 3 holdings and verify it is ≤ 45% of total fund assets",
        parent=criteria_node,
        critical=True
    )
    # Determine combined value to present (either extracted or computed from weights)
    combined_value = _safe_str(metrics.top3_combined_pct).strip() or (_compute_top3_combined(metrics) or "")
    concentration_present = evaluator.add_custom_node(
        result=bool(combined_value),
        id="top3_combined_present",
        desc="Top-3 combined percentage is provided (or computable from provided top-3 weights)",
        parent=concentration_seq,
        critical=True
    )
    concentration_verify = evaluator.add_leaf(
        id="top3_concentration_verified",
        desc="Top-3 combined weight is verified by provider page and ≤ 45%",
        parent=concentration_seq,
        critical=True
    )
    # Build claim including individual weights if available
    if metrics.top3_weights and len(metrics.top3_weights) >= 3:
        w1, w2, w3 = metrics.top3_weights[:3]
        concentration_claim = (
            f"The combined weight of the top 3 holdings is at most 45% of total assets. "
            f"Reported top 3 weights: {w1}, {w2}, {w3}. Combined reported/derived: '{combined_value}'."
        )
    else:
        concentration_claim = (
            f"The combined weight of the top 3 holdings is at most 45% of total assets. "
            f"Reported combined: '{combined_value}'."
        )
    await evaluator.verify(
        claim=concentration_claim,
        node=concentration_verify,
        sources=_safe_str(provider_url) or None,
        additional_instruction=(
            "Check the provider page's 'Top holdings' section. If necessary, compute the sum of the top 3 weights "
            "from the page to confirm it is ≤ 45%. Allow minor rounding differences."
        ),
        extra_prerequisites=[prereq_provider_url_node, concentration_present]
    )

    # 5) AUM ≥ $15 billion
    aum_seq = evaluator.add_sequential(
        id="aum_requirement",
        desc="Provide the AUM value and verify it is ≥ $15 billion",
        parent=criteria_node,
        critical=True
    )
    aum_present = evaluator.add_custom_node(
        result=bool(_safe_str(metrics.aum).strip()),
        id="aum_present",
        desc="AUM value is provided",
        parent=aum_seq,
        critical=True
    )
    aum_verify = evaluator.add_leaf(
        id="aum_verified",
        desc="AUM is verified by provider page and ≥ $15 billion",
        parent=aum_seq,
        critical=True
    )
    aum_claim = (
        f"The ETF's assets under management (AUM) are at least $15 billion. Reported AUM: '{_safe_str(metrics.aum)}'."
    )
    await evaluator.verify(
        claim=aum_claim,
        node=aum_verify,
        sources=_safe_str(provider_url) or None,
        additional_instruction=(
            "Confirm AUM on the provider page. Accept common units such as $B (billions) or long-form numbers; allow "
            "minor timing or rounding differences."
        ),
        extra_prerequisites=[prereq_provider_url_node, aum_present]
    )

    # 6) Yield ≥ 0.10% (either 30-day SEC or 12-month trailing)
    yield_seq = evaluator.add_sequential(
        id="yield_requirement",
        desc="Provide either the 30-day SEC yield or the 12-month trailing yield value and verify it is ≥ 0.10%",
        parent=criteria_node,
        critical=True
    )
    yield_present = evaluator.add_custom_node(
        result=bool(_safe_str(metrics.yield_value).strip()),
        id="yield_value_present",
        desc="Yield value is provided",
        parent=yield_seq,
        critical=True
    )
    yield_verify = evaluator.add_leaf(
        id="yield_verified",
        desc="Yield is verified by provider page and ≥ 0.10% (either 30-day SEC or 12-month trailing)",
        parent=yield_seq,
        critical=True
    )
    yield_type_display = _safe_str(metrics.yield_type) or "yield"
    yield_claim = (
        f"The ETF's {yield_type_display} is '{_safe_str(metrics.yield_value)}' and is at least 0.10%."
    )
    await evaluator.verify(
        claim=yield_claim,
        node=yield_verify,
        sources=_safe_str(provider_url) or None,
        additional_instruction=(
            "Confirm either 30-day SEC yield or 12-month trailing yield on the provider page. Accept whichever is "
            "explicitly given; allow minor rounding."
        ),
        extra_prerequisites=[prereq_provider_url_node, yield_present]
    )

    return {
        "criteria_node": criteria_node,
        "classification_verify": classification_verify,
        "expense_verify": expense_verify,
        "holdings_verify": holdings_verify,
        "concentration_verify": concentration_verify,
        "aum_verify": aum_verify,
        "yield_verify": yield_verify,
    }


async def _verify_provider_site_verifiability(
    evaluator: Evaluator,
    parent_node,
    ident: ETFIdentification,
    metrics: ETFMetrics,
    prerequisites: List[Any]
) -> None:
    """
    Add a single critical leaf to assert that ALL reported characteristics are verifiable via the official provider site.
    """
    verifiability_leaf = evaluator.add_leaf(
        id="provider_site_verifiability",
        desc="All reported characteristics (sector/classification, expense ratio, holdings count, top-3 concentration, AUM, yield) are verifiable via the official fund provider website",
        parent=parent_node,
        critical=True
    )

    combined_top3 = _safe_str(metrics.top3_combined_pct).strip() or (_compute_top3_combined(metrics) or "")
    top3_weights_display = ", ".join(metrics.top3_weights[:3]) if metrics.top3_weights else ""

    claim = (
        f"On the official provider page for ETF '{_safe_str(ident.name)}' (ticker '{_safe_str(ident.ticker)}'), all of "
        f"the following characteristics are clearly verifiable: "
        f"Sector/classification: '{_safe_str(metrics.sector_classification)}'; "
        f"Expense ratio: '{_safe_str(metrics.expense_ratio)}'; "
        f"Holdings count: '{_safe_str(metrics.holdings_count)}'; "
        f"Top-3 weights: [{top3_weights_display}] with combined '{combined_top3}'; "
        f"AUM: '{_safe_str(metrics.aum)}'; "
        f"Yield ({_safe_str(metrics.yield_type) or 'yield'}): '{_safe_str(metrics.yield_value)}'."
    )

    await evaluator.verify(
        claim=claim,
        node=verifiability_leaf,
        sources=_safe_str(ident.provider_url) or None,
        additional_instruction=(
            "Verify that each reported characteristic is present and supported on the official provider page. "
            "Allow reasonable rounding differences."
        ),
        extra_prerequisites=prerequisites
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the technology-sector ETF selection task.
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

    # Create a critical top-level node under root to align with rubric (critical root behavior)
    task_main = evaluator.add_parallel(
        id="task_main",
        desc="Identify one technology-sector ETF that satisfies all specified criteria and provide the requested identification fields and provider reference",
        parent=root,
        critical=True
    )

    # Extract structured ETF info from the answer
    extracted: ETFExtraction = await evaluator.extract(
        prompt=prompt_extract_etf(),
        template_class=ETFExtraction,
        extraction_name="etf_extraction",
    )

    # Record ground truth constraints (for reference)
    evaluator.add_ground_truth({
        "requirements": {
            "sector": "Technology (Information Technology or related subsectors such as Semiconductors)",
            "expense_ratio_max": "0.35%",
            "min_holdings": "100",
            "top3_max_combined": "45%",
            "aum_min": "$15 billion",
            "yield_min": "0.10% (30-day SEC or 12-month trailing)"
        }
    }, gt_type="constraints")

    # Defaults if missing nested structures
    ident = extracted.identification or ETFIdentification()
    metrics = extracted.metrics or ETFMetrics()

    # Build identification checks
    prereqs = await _verify_identification(evaluator, task_main, ident)
    provider_url_prereq_node = prereqs["provider_url_exists_node"]

    # Build criteria checks subtree
    await _verify_criteria_checks(
        evaluator=evaluator,
        parent_node=task_main,
        metrics=metrics,
        provider_url=ident.provider_url,
        prereq_provider_url_node=provider_url_prereq_node
    )

    # Provider site verifiability summary check
    await _verify_provider_site_verifiability(
        evaluator=evaluator,
        parent_node=task_main,
        ident=ident,
        metrics=metrics,
        prerequisites=[provider_url_prereq_node]
    )

    # Return summary
    return evaluator.get_summary()