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
TASK_ID = "silver_etf_lowest_expense"
TASK_DESCRIPTION = """
Among all exchange-traded funds (ETFs) that hold physical silver bullion, identify the one with the lowest expense ratio. Note that your selection must be a true ETF structure, not a closed-end trust or other investment vehicle type.

For the identified ETF, provide the following information:
1. The ETF's ticker symbol and full name
2. The official sponsor fee (or expense ratio) as stated in the fund's prospectus or official documentation
3. The fund's inception date (the date when the fund first began trading)

Include reference URLs from official sources (such as the fund issuer's website or reputable financial data providers) to support your answer.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ETFCore(BaseModel):
    ticker: Optional[str] = None
    name: Optional[str] = None
    expense_ratio: Optional[str] = None  # Keep as string to allow flexible formats like "0.30%" or "0.3%"
    inception_date: Optional[str] = None
    structure_type: Optional[str] = None  # e.g., "ETF", "Trust", "ETC", "ETN", "Closed-end fund"
    backing_type: Optional[str] = None    # e.g., "physically-backed", "futures", "miners", "derivatives"
    sources: List[str] = Field(default_factory=list)


class ETFCompetitor(ETFCore):
    pass


class ETFExtraction(BaseModel):
    selected: Optional[ETFCore] = None
    competitors: List[ETFCompetitor] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_etf_selection() -> str:
    return """
    Your goal is to extract the ETF that the answer identifies as the physically-backed silver ETF with the lowest expense ratio, along with all required details and the URLs cited.

    Extract a JSON object with two parts:
    1) selected: The ETF that the answer ultimately claims is the physically-backed silver ETF with the lowest expense ratio (must be a true ETF, not a trust or ETC/ETN).
       Fields to extract exactly as stated in the answer:
       - ticker: The ETF ticker symbol (e.g., "XYZ")
       - name: The full official ETF name
       - expense_ratio: The sponsor fee / management fee / expense ratio (as a percentage string, e.g., "0.30%")
       - inception_date: The fund inception or launch date (string, keep the format as given in the answer)
       - structure_type: The structure described in the answer (e.g., "ETF", "Trust", "ETC", "ETN", "Closed-end fund")
       - backing_type: What the product holds (e.g., "physically-backed", "futures", "miners", "derivatives")
       - sources: An array of all URLs in the answer that directly support any of the above facts (issuer pages, fact sheets/prospectus, or reputable data providers). Only extract URLs explicitly present in the answer.

    2) competitors: A list (possibly empty) of any other physically-backed silver ETFs mentioned in the answer that are used for comparison.
       For each competitor, extract:
       - ticker, name, expense_ratio, inception_date (if available; otherwise null)
       - structure_type, backing_type (if mentioned)
       - sources: All URLs in the answer associated with that competitor.

    Rules:
    - Do NOT invent any information. Only extract what is explicitly present in the answer.
    - If a field is not present in the answer, set it to null (or an empty array for sources).
    - Include only valid URLs that appear in the answer text (including markdown links).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    cleaned = []
    for u in urls:
        if not u:
            continue
        uu = u.strip()
        if uu and uu not in seen:
            seen.add(uu)
            cleaned.append(uu)
    return cleaned


def _combine_all_sources(selected: Optional[ETFCore], competitors: List[ETFCompetitor]) -> List[str]:
    urls: List[str] = []
    if selected and selected.sources:
        urls.extend(selected.sources)
    for c in competitors:
        urls.extend(c.sources or [])
    return _dedup_urls(urls)


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: ETFExtraction) -> None:
    """
    Build the verification tree following the rubric and run verifications.
    """
    selected = extracted.selected or ETFCore()
    competitors = extracted.competitors or []

    # All sources for the selected ETF
    sel_sources = _dedup_urls(selected.sources or [])
    # All sources across selected + competitors (for comparison)
    all_sources = _combine_all_sources(selected, competitors)

    # Root-level rubric node (critical, sequential)
    solution_node = evaluator.add_sequential(
        id="Solution_Completeness",
        desc="The solution correctly identifies the physically-backed silver ETF with the lowest expense ratio (excluding closed-end trusts) and provides all required information including ticker symbol, full name, sponsor fee, inception date, and reference URLs.",
        parent=evaluator.root,
        critical=True
    )

    # Correct ETF Identified (critical, sequential)
    correct_etf_node = evaluator.add_sequential(
        id="Correct_ETF_Identified",
        desc="The identified ETF is the correct physically-backed silver ETF with the lowest expense ratio among true ETF structures (not closed-end trusts). The answer must include the ETF's ticker symbol and full name.",
        parent=solution_node,
        critical=True
    )

    # Structure Type Verified (leaf, critical)
    structure_leaf = evaluator.add_leaf(
        id="Structure_Type_Verified",
        desc="The identified investment vehicle is confirmed to be a true exchange-traded fund (ETF) structure, not a closed-end trust or other investment vehicle type. This must be verifiable from official fund documentation or descriptions.",
        parent=correct_etf_node,
        critical=True
    )
    structure_claim_subject = f"{(selected.ticker or '').strip()} {(selected.name or '').strip()}".strip() or "the identified product"
    structure_claim = (
        f"{structure_claim_subject} is a true exchange-traded fund (ETF) structure, "
        f"and NOT a trust, ETC, ETN, or closed-end fund."
    )
    await evaluator.verify(
        claim=structure_claim,
        node=structure_leaf,
        sources=sel_sources if sel_sources else None,
        additional_instruction=(
            "Use the provided URLs to confirm the legal structure. If sources describe it as a 'trust', 'ETC', 'ETN', or "
            "'closed-end fund', then it is NOT a true ETF. Prefer issuer documentation, prospectus, or official fund pages. "
            "If the sources are ambiguous or missing, judge this claim as unsupported."
        ),
    )

    # Backing Type Verified (leaf, critical)
    backing_leaf = evaluator.add_leaf(
        id="Backing_Type_Verified",
        desc="The identified ETF is confirmed to be physically-backed, meaning it holds actual physical silver bullion (not silver mining company stocks, silver futures contracts, or derivatives). This must be verifiable from the fund's investment objective or strategy description.",
        parent=correct_etf_node,
        critical=True
    )
    backing_claim = (
        f"{structure_claim_subject} is physically-backed and holds actual physical silver bullion (bars), "
        f"not futures, derivatives, or mining equities."
    )
    await evaluator.verify(
        claim=backing_claim,
        node=backing_leaf,
        sources=sel_sources if sel_sources else None,
        additional_instruction=(
            "Check the investment objective/strategy on the provided URLs to confirm that the fund holds physical silver bullion "
            "(e.g., vaulted silver bars). If it instead uses futures, derivatives, or invests in mining companies, this claim is false. "
            "Prefer issuer or official documents."
        ),
    )

    # Lowest Expense Ratio Verified (leaf, critical)
    lowest_leaf = evaluator.add_leaf(
        id="Lowest_Expense_Ratio_Verified",
        desc="The identified ETF has the lowest expense ratio among all physically-backed silver ETFs that are true ETF structures. This requires comparing expense ratios across eligible ETFs and confirming that the selected ETF has the minimum expense ratio.",
        parent=correct_etf_node,
        critical=True
    )
    if selected.expense_ratio and isinstance(selected.expense_ratio, str):
        lowest_claim = (
            f"Among physically-backed silver ETFs that are true ETFs, {structure_claim_subject} has the lowest expense ratio "
            f"at {selected.expense_ratio}."
        )
    else:
        lowest_claim = (
            f"Among physically-backed silver ETFs that are true ETFs, {structure_claim_subject} has the lowest expense ratio."
        )
    await evaluator.verify(
        claim=lowest_claim,
        node=lowest_leaf,
        sources=all_sources if all_sources else None,
        additional_instruction=(
            "Use the provided URLs to compare expense ratios across candidate ETFs that are both: (a) true ETF structures and "
            "(b) physically-backed by silver bullion. Exclude closed-end trusts (e.g., PSLV), exchange-traded commodities (ETCs), "
            "ETNs, and grantor trusts (e.g., SLV, SIVR) from consideration. If sources do not conclusively support that the selected "
            "ETF has the lowest expense ratio among the eligible set, judge this claim as unsupported."
        ),
    )

    # Required Information Provided (critical, parallel)
    required_info_node = evaluator.add_parallel(
        id="Required_Information_Provided",
        desc="The solution provides all required information: the ETF identification details (ticker and name), the official sponsor/management fee, the fund inception date, and reference URLs from official sources.",
        parent=correct_etf_node,
        critical=True
    )

    # ETF Details Provided (custom existence, critical)
    details_ok = bool((selected.ticker or "").strip()) and bool((selected.name or "").strip())
    evaluator.add_custom_node(
        result=details_ok,
        id="ETF_Details_Provided",
        desc="The solution provides both the ticker symbol and the full official name of the identified ETF.",
        parent=required_info_node,
        critical=True
    )

    # Sponsor Fee Accurate (leaf, critical)
    sponsor_leaf = evaluator.add_leaf(
        id="Sponsor_Fee_Accurate",
        desc="The sponsor fee (or management fee/expense ratio) is accurately provided and matches the official fee stated in the ETF's prospectus or official fund page. The fee must be expressed as a percentage.",
        parent=required_info_node,
        critical=True
    )
    fee_claim_subject = structure_claim_subject or "the identified ETF"
    fee_value = (selected.expense_ratio or "").strip()
    fee_claim = (
        f"The expense ratio (also called sponsor or management fee) of {fee_claim_subject} is {fee_value}."
        if fee_value else
        f"The expense ratio (also called sponsor or management fee) of {fee_claim_subject} is correctly stated."
    )
    await evaluator.verify(
        claim=fee_claim,
        node=sponsor_leaf,
        sources=sel_sources if sel_sources else None,
        additional_instruction=(
            "Verify the exact percentage against the official fund page, fact sheet, or prospectus. Allow minor formatting differences "
            "such as 0.30% vs 0.3%, but the numeric value must match."
        ),
    )

    # Inception Date Accurate (leaf, critical)
    inception_leaf = evaluator.add_leaf(
        id="Inception_Date_Accurate",
        desc="The fund inception date (launch date or first trading date) is accurately provided and matches the official date documented in fund materials or reliable financial databases. The date must be provided in a clear format (e.g., Month Day, Year or MM/DD/YYYY).",
        parent=required_info_node,
        critical=True
    )
    inception_value = (selected.inception_date or "").strip()
    inception_claim = (
        f"The inception date (launch date) of {fee_claim_subject} is {inception_value}."
        if inception_value else
        f"The inception date (launch date) of {fee_claim_subject} is correctly stated."
    )
    await evaluator.verify(
        claim=inception_claim,
        node=inception_leaf,
        sources=sel_sources if sel_sources else None,
        additional_instruction=(
            "Confirm the inception/launch/listing date on the official issuer page, prospectus, or a reputable database (e.g., fund factsheet). "
            "Allow reasonable date format variations but the date must correspond to the same calendar day."
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
    Evaluate an answer for the 'physically-backed silver ETF with the lowest expense ratio' task.
    """
    # Initialize evaluator with sequential root to reflect stepwise gating
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured ETF selection and any competitors from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_etf_selection(),
        template_class=ETFExtraction,
        extraction_name="etf_selection"
    )

    # Optionally record custom info for transparency
    evaluator.add_custom_info(
        info={
            "selected": extraction.selected.dict() if extraction.selected else None,
            "competitors": [c.dict() for c in extraction.competitors] if extraction.competitors else []
        },
        info_type="extracted_entities",
        info_name="extracted_etf_entities"
    )

    # Build verification tree and run verifications
    await build_and_verify_tree(evaluator, extraction)

    # Return standardized evaluation summary
    return evaluator.get_summary()