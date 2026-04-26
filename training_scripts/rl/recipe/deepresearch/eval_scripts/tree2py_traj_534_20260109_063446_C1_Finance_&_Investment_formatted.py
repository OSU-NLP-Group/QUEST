import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "xrp_etf_nov2025"
TASK_DESCRIPTION = "In November 2025, several XRP exchange-traded funds (ETFs) launched in the United States. Identify the XRP ETF that meets ALL of the following criteria: 1. Listed on NYSE Arca (not NYSE or other exchanges), 2. Has an annual sponsor fee of 0.19%, 3. Offers a fee waiver on the first $5 billion in assets until May 2026. Provide the ETF's ticker symbol and the name of the issuing company."


class ETFInfo(BaseModel):
    ticker: Optional[str] = None
    company: Optional[str] = None
    exchange: Optional[str] = None
    annual_sponsor_fee: Optional[str] = None
    fee_waiver_assets: Optional[str] = None
    fee_waiver_until: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


def prompt_extract_etf_info() -> str:
    return (
        "Extract the single XRP ETF identified in the answer as meeting all specified criteria. "
        "If multiple XRP ETFs are mentioned, choose the one that the answer explicitly claims meets ALL criteria "
        "or the one presented as the final answer. Extract the following fields exactly as stated in the answer:\n"
        "1. ticker: The ETF's ticker symbol (string)\n"
        "2. company: The issuing company name (sponsor/issuer) (string)\n"
        "3. exchange: The listing exchange as written (e.g., 'NYSE Arca') (string)\n"
        "4. annual_sponsor_fee: The annual sponsor or management fee, keep formatting as in answer (e.g., '0.19%') (string)\n"
        "5. fee_waiver_assets: Any description of fee waiver threshold (e.g., 'first $5 billion') (string)\n"
        "6. fee_waiver_until: The time-until for the fee waiver (e.g., 'May 2026', 'May 31, 2026') (string)\n"
        "7. sources: An array of all URLs cited in the answer that support this ETF and its attributes. "
        "Only include actual URLs mentioned in the answer (including markdown links); do not invent URLs.\n"
        "If a field is not provided, return null for that field. If no URLs are provided, return an empty array for sources."
    )


async def build_and_verify_tree(evaluator: Evaluator, parent_node, etf: ETFInfo) -> None:
    overall = evaluator.add_parallel(
        id="overall_etf_verification",
        desc="Identify the XRP ETF meeting all specified criteria and provide its ticker and issuing company",
        parent=parent_node,
        critical=True
    )

    has_required = bool(etf.ticker and etf.ticker.strip() and etf.company and etf.company.strip())
    evaluator.add_custom_node(
        result=has_required,
        id="answer_includes_required_fields",
        desc="Response provides both (a) the ETF ticker symbol and (b) the issuing company name",
        parent=overall,
        critical=True
    )

    is_spot_leaf = evaluator.add_leaf(
        id="is_xrp_spot_etf",
        desc="The identified ETF is an XRP exchange-traded fund (spot XRP ETF)",
        parent=overall,
        critical=True
    )
    is_spot_claim = (
        f"The ETF with ticker '{etf.ticker or ''}' issued by '{etf.company or ''}' is a spot XRP ETF that invests directly in XRP or tracks the spot price of XRP."
    )
    await evaluator.verify(
        claim=is_spot_claim,
        node=is_spot_leaf,
        sources=etf.sources,
        additional_instruction=(
            "Confirm from the cited source page(s) that the product is a spot XRP ETF (or trust) backed by XRP or tracking XRP's spot price, "
            "not a futures-based or synthetic product. Minor naming variations are acceptable as long as the product is clearly a spot XRP ETF."
        ),
    )

    launched_leaf = evaluator.add_leaf(
        id="launched_in_november_2025",
        desc="The identified ETF launched in November 2025",
        parent=overall,
        critical=True
    )
    launched_claim = (
        "This ETF's initial listing, first trading, inception, or launch date was in November 2025."
    )
    await evaluator.verify(
        claim=launched_claim,
        node=launched_leaf,
        sources=etf.sources,
        additional_instruction=(
            "Verify the launch timing using the cited source(s). Accept terms such as 'listing date', 'first trading date', 'inception date', or 'launch'. "
            "It must be in November 2025."
        ),
    )

    listed_leaf = evaluator.add_leaf(
        id="listed_on_nyse_arca",
        desc="The identified ETF is listed on NYSE Arca (not NYSE or another exchange)",
        parent=overall,
        critical=True
    )
    listed_claim = "This ETF is listed on NYSE Arca."
    await evaluator.verify(
        claim=listed_claim,
        node=listed_leaf,
        sources=etf.sources,
        additional_instruction=(
            "Confirm explicitly that the listing exchange is 'NYSE Arca'. Mentions of 'NYSE' or other exchanges do not satisfy this criterion. "
            "Minor formatting differences (e.g., 'NYSE Arca, Inc.') are acceptable."
        ),
    )

    fee_leaf = evaluator.add_leaf(
        id="annual_sponsor_fee_0_19",
        desc="The identified ETF has an annual sponsor/management fee of 0.19%",
        parent=overall,
        critical=True
    )
    fee_claim = "The ETF's annual sponsor or management fee is 0.19%."
    await evaluator.verify(
        claim=fee_claim,
        node=fee_leaf,
        sources=etf.sources,
        additional_instruction=(
            "Check the fee section or prospectus text on the cited page(s) to confirm that the annual sponsor/management fee is 0.19%. "
            "Allow minor rounding or formatting (e.g., '0.19 percent')."
        ),
    )

    waiver_assets_leaf = evaluator.add_leaf(
        id="fee_waiver_first_5b",
        desc="The identified ETF offers a fee waiver on the first $5 billion in assets",
        parent=overall,
        critical=True
    )
    waiver_assets_claim = "The ETF offers a fee waiver on the first $5 billion in assets under management."
    await evaluator.verify(
        claim=waiver_assets_claim,
        node=waiver_assets_leaf,
        sources=etf.sources,
        additional_instruction=(
            "Confirm that the sponsor fee is waived for the first $5B AUM (phrasing like 'first $5 billion' or '$5B cap' is acceptable) "
            "as stated in the cited source(s)."
        ),
    )

    waiver_until_leaf = evaluator.add_leaf(
        id="fee_waiver_until_may_2026",
        desc="The identified ETF's fee waiver is in effect until May 2026",
        parent=overall,
        critical=True
    )
    waiver_until_claim = "The ETF's fee waiver is in effect until May 2026."
    await evaluator.verify(
        claim=waiver_until_claim,
        node=waiver_until_leaf,
        sources=etf.sources,
        additional_instruction=(
            "Confirm from the cited source(s) that the fee waiver period ends in May 2026 (e.g., 'until May 2026' or a specific date in May 2026)."
        ),
    )


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
        default_model=model,
    )

    extracted_etf = await evaluator.extract(
        prompt=prompt_extract_etf_info(),
        template_class=ETFInfo,
        extraction_name="extracted_etf_info",
    )

    evaluator.add_custom_info(
        info={
            "extracted": extracted_etf.dict(),
            "source_count": len(extracted_etf.sources),
        },
        info_type="extraction_debug",
        info_name="extracted_etf_debug"
    )

    await build_and_verify_tree(evaluator, root, extracted_etf)

    return evaluator.get_summary()