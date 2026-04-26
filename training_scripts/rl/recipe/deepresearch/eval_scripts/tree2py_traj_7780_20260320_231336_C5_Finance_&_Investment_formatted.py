import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "franklin_xrp_etf_nyse_arca_2025_11_24"
TASK_DESCRIPTION = """
Identify the Franklin Templeton cryptocurrency ETF that provides exposure to XRP and was launched on November 24, 2025, on the NYSE Arca exchange. Provide the following information:

1. The official ticker symbol
2. The expense ratio (gross and net, as percentages)
3. The benchmark index that the ETF tracks

Include reference URLs supporting each piece of information.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ETFExtraction(BaseModel):
    # Helpful identification fields
    fund_name: Optional[str] = None

    # Constraint-related fields + sources
    issuer_name: Optional[str] = None
    issuer_sources: List[str] = Field(default_factory=list)

    xrp_exposure_statement: Optional[str] = None
    xrp_sources: List[str] = Field(default_factory=list)

    launch_date: Optional[str] = None
    launch_date_sources: List[str] = Field(default_factory=list)

    exchange_listing: Optional[str] = None
    listing_sources: List[str] = Field(default_factory=list)

    # Required output fields + sources
    ticker_symbol: Optional[str] = None
    ticker_sources: List[str] = Field(default_factory=list)

    gross_expense_ratio: Optional[str] = None
    gross_expense_sources: List[str] = Field(default_factory=list)

    net_expense_ratio: Optional[str] = None
    net_expense_sources: List[str] = Field(default_factory=list)

    benchmark_index: Optional[str] = None
    benchmark_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_etf() -> str:
    return """
    Extract details about a single Franklin Templeton ETF discussed in the answer. Return the fields below exactly as they appear in the answer text. For every field that asks for sources, extract only URLs that are explicitly present in the answer (including markdown links). Do not invent anything.

    Required JSON fields to extract:
    - fund_name: Full official ETF name, if present; else null.
    - issuer_name: The issuer/sponsor/adviser name mentioned (expected to be "Franklin Templeton" or a close corporate variant); else null.
    - issuer_sources: URLs cited that support the issuer information.

    - xrp_exposure_statement: Short phrase or sentence that shows the ETF provides exposure to XRP (Ripple). Examples: "provides exposure to XRP", "XRP ETF", "tracks XRP price"; else null.
    - xrp_sources: URLs cited that support the XRP exposure claim.

    - launch_date: The date the ETF launched/began trading/inception/commenced trading; keep the exact string format as written (e.g., "November 24, 2025"); else null.
    - launch_date_sources: URLs cited that support the launch/inception/commencement date.

    - exchange_listing: The exchange listing (e.g., "NYSE Arca"); else null.
    - listing_sources: URLs cited that support the exchange listing.

    - ticker_symbol: The official ticker symbol (uppercase letters preferred if given that way); else null.
    - ticker_sources: URLs cited that confirm the ticker.

    - gross_expense_ratio: The gross expense ratio as a percentage string (e.g., "0.25%"); else null.
    - gross_expense_sources: URLs cited that confirm the gross expense ratio.

    - net_expense_ratio: The net expense ratio as a percentage string (e.g., "0.19%"); else null.
    - net_expense_sources: URLs cited that confirm the net expense ratio.

    - benchmark_index: The benchmark or index name tracked by the ETF, as written; else null.
    - benchmark_sources: URLs cited that confirm the benchmark/index name.

    Extraction notes:
    - For any field not found in the answer, set it to null. For any sources field with no URLs provided in the answer, return an empty list.
    - Preserve percentage formatting exactly as in the answer (e.g., include the % sign if present).
    - Prefer full, valid URLs. If a URL is missing a protocol, prepend http:// as needed per the SPECIAL RULES.
    - Do not normalize date formats; keep the exact date string as written in the answer (e.g., "Nov 24, 2025" or "November 24, 2025").
    """.strip()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def nonempty(s: Optional[str]) -> bool:
    return s is not None and str(s).strip() != ""


def identity_label(data: ETFExtraction) -> str:
    """
    Build a human-friendly ETF identifier snippet to include in claims,
    improving the verifier's precision.
    """
    parts = []
    if nonempty(data.fund_name):
        parts.append(f"named '{data.fund_name}'")
    if nonempty(data.ticker_symbol):
        parts.append(f"with ticker '{data.ticker_symbol}'")
    if parts:
        return "for the ETF " + " ".join(parts)
    return "for the ETF"


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def build_citation_policy_nodes(evaluator: Evaluator, parent) -> Dict[str, Any]:
    """
    Build 'Citation_Policy' node with per-claim URL existence checks (critical).
    Returns a dict mapping field keys to the created existence nodes so they
    can be used as extra prerequisites for downstream verifications.
    """
    node = evaluator.add_parallel(
        id="citation_policy",
        desc="Each required claim has at least one reference URL provided in the answer.",
        parent=parent,
        critical=True
    )

    exists_nodes = {}

    def add_url_existence(field_key: str, present: bool, desc_tail: str):
        n = evaluator.add_custom_node(
            result=present,
            id=f"url_exists__{field_key}",
            desc=f"URL(s) provided for {desc_tail}",
            parent=node,
            critical=True
        )
        exists_nodes[field_key] = n

    # Populate URL existence checks
    extracted: ETFExtraction = evaluator._extraction_results[-1]["result"]  # already recorded; but safer to pass in
    # Safer approach: We'll not rely on that internal. The caller will provide the extracted object.
    # To keep function generic, we'll just return the nodes; the caller will handle 'present'.

    return exists_nodes  # Placeholder; we'll not use this return directly (overridden below)


def add_url_existence_nodes(
    evaluator: Evaluator,
    citation_parent,
    data: ETFExtraction
) -> Dict[str, Any]:
    """
    Add actual URL existence nodes under the Citation_Policy parent and return mapping.
    """
    def has_urls(urls: List[str]) -> bool:
        return isinstance(urls, list) and len([u for u in urls if nonempty(u)]) > 0

    exists_nodes: Dict[str, Any] = {}

    def add(field_key: str, urls: List[str], human_desc: str):
        node = evaluator.add_custom_node(
            result=has_urls(urls),
            id=f"url_exists__{field_key}",
            desc=f"URL(s) provided for {human_desc}",
            parent=citation_parent,
            critical=True
        )
        exists_nodes[field_key] = node

    add("issuer", data.issuer_sources, "issuer (Franklin Templeton)")
    add("xrp", data.xrp_sources, "XRP exposure claim")
    add("launch_date", data.launch_date_sources, "launch/inception/commencement date")
    add("listing", data.listing_sources, "exchange listing (NYSE Arca)")

    add("ticker", data.ticker_sources, "ticker symbol")
    add("gross", data.gross_expense_sources, "gross expense ratio")
    add("net", data.net_expense_sources, "net expense ratio")
    add("benchmark", data.benchmark_sources, "benchmark/index")

    return exists_nodes


async def verify_constraints(
    evaluator: Evaluator,
    parent,
    data: ETFExtraction,
    url_exists_nodes: Dict[str, Any]
) -> None:
    """
    Build and verify the 'ETF_Constraint_Match' subtree.
    """
    constraints = evaluator.add_parallel(
        id="etf_constraint_match",
        desc="The ETF satisfies issuer, XRP exposure, launch date, and NYSE Arca listing constraints with supporting citations.",
        parent=parent,
        critical=True
    )

    # Issuer
    issuer_leaf = evaluator.add_leaf(
        id="issuer_is_franklin_templeton",
        desc="Cited source(s) indicate the ETF is issued/sponsored by Franklin Templeton.",
        parent=constraints,
        critical=True
    )
    issuer_claim_identity = identity_label(data)
    issuer_claim = f"The issuer/sponsor/adviser is Franklin Templeton {issuer_claim_identity}."
    await evaluator.verify(
        claim=issuer_claim,
        node=issuer_leaf,
        sources=data.issuer_sources,
        additional_instruction=(
            "Accept close corporate variants such as 'Franklin Templeton', 'Franklin Templeton Investments', "
            "'Franklin Resources, Inc.' or similar branding where it clearly indicates Franklin Templeton is the ETF's issuer/sponsor/adviser."
        ),
        extra_prerequisites=[url_exists_nodes.get("issuer")] if url_exists_nodes.get("issuer") else None
    )

    # XRP exposure
    xrp_leaf = evaluator.add_leaf(
        id="provides_xrp_exposure",
        desc="Cited source(s) indicate the ETF provides exposure to XRP.",
        parent=constraints,
        critical=True
    )
    xrp_claim_identity = identity_label(data)
    xrp_claim = (
        f"The ETF {xrp_claim_identity} provides exposure to XRP (Ripple), either by directly holding XRP or by tracking an XRP price/index."
    )
    await evaluator.verify(
        claim=xrp_claim,
        node=xrp_leaf,
        sources=data.xrp_sources,
        additional_instruction=(
            "Look for explicit mention of 'XRP', 'Ripple', 'XRP Ledger' exposure. Phrases like 'XRP ETF', "
            "'provides exposure to XRP', or 'tracks XRP index/price' should count as support."
        ),
        extra_prerequisites=[url_exists_nodes.get("xrp")] if url_exists_nodes.get("xrp") else None
    )

    # Launch date Nov 24, 2025
    launch_leaf = evaluator.add_leaf(
        id="launch_date_is_2025_11_24",
        desc="Cited source(s) indicate the ETF launched on November 24, 2025.",
        parent=constraints,
        critical=True
    )
    launch_claim_identity = identity_label(data)
    launch_claim = f"The ETF {launch_claim_identity} launched/commenced trading (or inception date) on November 24, 2025."
    await evaluator.verify(
        claim=launch_claim,
        node=launch_leaf,
        sources=data.launch_date_sources,
        additional_instruction=(
            "Treat 'launch date', 'inception date', 'commencement of trading', or 'listing date' that indicates the first trading day as acceptable. "
            "Allow small date-format variants like 'Nov 24, 2025' vs 'November 24, 2025', but the date must be 2025-11-24."
        ),
        extra_prerequisites=[url_exists_nodes.get("launch_date")] if url_exists_nodes.get("launch_date") else None
    )

    # Listed on NYSE Arca
    listing_leaf = evaluator.add_leaf(
        id="listed_on_nyse_arca",
        desc="Cited source(s) indicate the ETF is listed/traded on NYSE Arca.",
        parent=constraints,
        critical=True
    )
    listing_claim_identity = identity_label(data)
    listing_claim = f"The ETF {listing_claim_identity} is listed/traded on NYSE Arca."
    await evaluator.verify(
        claim=listing_claim,
        node=listing_leaf,
        sources=data.listing_sources,
        additional_instruction=(
            "Accept phrases like 'listed on NYSE Arca', 'trades on NYSE Arca', or similar. "
            "Other exchanges (e.g., NASDAQ, Cboe) should not be accepted."
        ),
        extra_prerequisites=[url_exists_nodes.get("listing")] if url_exists_nodes.get("listing") else None
    )


async def verify_required_fields(
    evaluator: Evaluator,
    parent,
    data: ETFExtraction,
    url_exists_nodes: Dict[str, Any]
) -> None:
    """
    Build and verify the 'Required_Output_Fields' subtree:
    - Ticker
    - Expense Ratios (gross & net)
    - Benchmark index
    Each with existence checks + URL-grounded verification leaves.
    """
    required = evaluator.add_parallel(
        id="required_output_fields",
        desc="Provides all requested ETF fields (ticker, gross & net expense ratios, benchmark index) with citations.",
        parent=parent,
        critical=True
    )

    # Ticker: existence + verification
    ticker_exists = evaluator.add_custom_node(
        result=nonempty(data.ticker_symbol),
        id="ticker_provided",
        desc="Ticker symbol is provided in the answer.",
        parent=required,
        critical=True
    )
    ticker_leaf = evaluator.add_leaf(
        id="ticker_symbol_verified",
        desc="Provides the official ticker symbol for the ETF, and a cited source confirms that ticker.",
        parent=required,
        critical=True
    )
    ticker_claim_identity = identity_label(data)
    ticker_claim = f"The ETF's official ticker symbol {ticker_claim_identity} is '{data.ticker_symbol}'."
    await evaluator.verify(
        claim=ticker_claim,
        node=ticker_leaf,
        sources=data.ticker_sources,
        additional_instruction="Verify that the cited page explicitly shows the ETF ticker as stated.",
        extra_prerequisites=[ticker_exists] + ([url_exists_nodes["ticker"]] if url_exists_nodes.get("ticker") else [])
    )

    # Expense Ratios parent
    expense_parent = evaluator.add_parallel(
        id="expense_ratios",
        desc="Expense ratios (gross and net) are provided and correctly sourced.",
        parent=required,
        critical=True
    )

    # Gross expense ratio
    gross_exists = evaluator.add_custom_node(
        result=nonempty(data.gross_expense_ratio),
        id="gross_ratio_provided",
        desc="Gross expense ratio value is provided.",
        parent=expense_parent,
        critical=True
    )
    gross_leaf = evaluator.add_leaf(
        id="gross_expense_ratio_verified",
        desc="Gross expense ratio is stated as a percentage and matches the cited source.",
        parent=expense_parent,
        critical=True
    )
    gross_claim_identity = identity_label(data)
    gross_claim = f"The ETF's gross expense ratio {gross_claim_identity} is '{data.gross_expense_ratio}'."
    await evaluator.verify(
        claim=gross_claim,
        node=gross_leaf,
        sources=data.gross_expense_sources,
        additional_instruction=(
            "Confirm the gross expense ratio percentage exactly or within trivial formatting (e.g., including % sign). "
            "Do not confuse gross with net; the page should clearly indicate 'gross'."
        ),
        extra_prerequisites=[gross_exists] + ([url_exists_nodes["gross"]] if url_exists_nodes.get("gross") else [])
    )

    # Net expense ratio
    net_exists = evaluator.add_custom_node(
        result=nonempty(data.net_expense_ratio),
        id="net_ratio_provided",
        desc="Net expense ratio value is provided.",
        parent=expense_parent,
        critical=True
    )
    net_leaf = evaluator.add_leaf(
        id="net_expense_ratio_verified",
        desc="Net expense ratio is stated as a percentage and matches the cited source.",
        parent=expense_parent,
        critical=True
    )
    net_claim_identity = identity_label(data)
    net_claim = f"The ETF's net expense ratio {net_claim_identity} is '{data.net_expense_ratio}'."
    await evaluator.verify(
        claim=net_claim,
        node=net_leaf,
        sources=data.net_expense_sources,
        additional_instruction=(
            "Confirm the net expense ratio percentage exactly or within trivial formatting (e.g., including % sign). "
            "If the source distinguishes gross vs net, ensure the 'net' figure is used."
        ),
        extra_prerequisites=[net_exists] + ([url_exists_nodes["net"]] if url_exists_nodes.get("net") else [])
    )

    # Benchmark index
    benchmark_exists = evaluator.add_custom_node(
        result=nonempty(data.benchmark_index),
        id="benchmark_provided",
        desc="Benchmark index name is provided.",
        parent=required,
        critical=True
    )
    benchmark_leaf = evaluator.add_leaf(
        id="benchmark_index_verified",
        desc="Provides the benchmark index tracked by the ETF, confirmed by a cited source.",
        parent=required,
        critical=True
    )
    benchmark_claim_identity = identity_label(data)
    benchmark_claim = f"The ETF {benchmark_claim_identity} tracks the benchmark/index named '{data.benchmark_index}'."
    await evaluator.verify(
        claim=benchmark_claim,
        node=benchmark_leaf,
        sources=data.benchmark_sources,
        additional_instruction=(
            "Accept synonyms like 'benchmark', 'index', 'reference index', or 'underlying index'. "
            "The index name on the page should clearly match the claimed name (minor formatting/case differences acceptable)."
        ),
        extra_prerequisites=[benchmark_exists] + ([url_exists_nodes["benchmark"]] if url_exists_nodes.get("benchmark") else [])
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
    Evaluate an answer for the Franklin Templeton XRP ETF identification task.
    """
    # Initialize evaluator with a critical, parallel root to mirror rubric semantics
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
    # Mark root as critical by wrapping it with a critical child that encompasses all checks
    # Instead, we keep root as non-critical per framework design; enforce criticality at top-level children.

    # Ground-truth-like constraints (for context only; actual verification is evidence-based)
    evaluator.add_ground_truth({
        "expected_constraints": {
            "issuer": "Franklin Templeton",
            "asset_exposure": "XRP",
            "launch_date": "November 24, 2025",
            "exchange": "NYSE Arca"
        },
        "required_fields": ["ticker_symbol", "gross_expense_ratio", "net_expense_ratio", "benchmark_index"]
    }, gt_type="task_expectations")

    # 1) Extract structured info from the answer
    extracted: ETFExtraction = await evaluator.extract(
        prompt=prompt_extract_etf(),
        template_class=ETFExtraction,
        extraction_name="etf_extraction"
    )

    # 2) Build top-level critical nodes to mirror rubric
    # 2.1 Citation policy node with URL-existence checks
    citation_parent = evaluator.add_parallel(
        id="citation_policy",
        desc="Each required claim (constraints + fields) includes at least one reference URL in the answer.",
        parent=root,
        critical=True
    )
    url_exists_nodes = add_url_existence_nodes(evaluator, citation_parent, extracted)

    # 2.2 Constraints verification
    await verify_constraints(evaluator, root, extracted, url_exists_nodes)

    # 2.3 Required output fields verification
    await verify_required_fields(evaluator, root, extracted, url_exists_nodes)

    # Return the aggregated result
    return evaluator.get_summary()