import asyncio
import logging
from typing import Optional, Dict, Any

from pydantic import BaseModel

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "franklin_xrpz_etf_research"
TASK_DESCRIPTION = (
    "Research Franklin Templeton's spot XRP ETF (ticker: XRPZ) and provide the following information:\n\n"
    "1. Launch Date: The official date when XRPZ began trading on the exchange\n"
    "2. Management Fee: The base annual management fee (as a percentage)\n"
    "3. Exchange: The stock exchange where XRPZ is listed\n"
    "4. XRP Custodian: The name of the entity that serves as custodian for the XRP digital assets held by the ETF\n"
    "5. Cash Custodian/Administrator: The name of the entity that serves as the administrator, transfer agent, and/or cash custodian\n"
    "6. XRP Custodian Headquarters: The city and state where the XRP custodian's headquarters is located\n"
    "7. Cash Custodian Headquarters: The city and state where the cash custodian/administrator's headquarters is located\n\n"
    "For each piece of information, provide a reference URL that supports your answer."
)


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class XRPZAttributes(BaseModel):
    # 1. Launch Date
    launch_date: Optional[str] = None
    launch_date_url: Optional[str] = None

    # 2. Management Fee
    management_fee: Optional[str] = None
    management_fee_url: Optional[str] = None

    # 3. Exchange
    exchange: Optional[str] = None
    exchange_url: Optional[str] = None

    # 4. XRP Custodian (digital asset custodian)
    xrp_custodian: Optional[str] = None
    xrp_custodian_url: Optional[str] = None

    # 5. Cash Custodian/Administrator (administrator/transfer agent and/or cash custodian)
    cash_custodian_admin: Optional[str] = None
    cash_custodian_admin_url: Optional[str] = None

    # 6. XRP Custodian HQ (city, state)
    xrp_custodian_hq: Optional[str] = None
    xrp_custodian_hq_url: Optional[str] = None

    # 7. Cash Custodian/Administrator HQ (city, state)
    cash_custodian_admin_hq: Optional[str] = None
    cash_custodian_admin_hq_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_xrpz_attributes() -> str:
    return (
        "Extract the seven requested attributes for Franklin Templeton's spot XRP ETF (ticker: XRPZ) exactly as stated "
        "in the provided answer text. For each attribute, also extract one supporting reference URL mentioned in the "
        "answer (pick the most relevant/authoritative if multiple are given; otherwise use the first provided). "
        "Return JSON fields as below. If any field is missing, set it to null.\n\n"
        "Required JSON fields:\n"
        "- launch_date: The official date when XRPZ began trading on the exchange (free-form string, e.g., 'Jan 10, 2026')\n"
        "- launch_date_url: A URL in the answer that specifically supports the trading start date\n"
        "- management_fee: The base annual management fee (percentage or bps, e.g., '0.19%' or '19 bps')\n"
        "- management_fee_url: A URL in the answer that supports the stated management fee\n"
        "- exchange: The specific U.S. exchange where XRPZ is listed (e.g., 'Cboe BZX', 'NYSE Arca', 'Nasdaq')\n"
        "- exchange_url: A URL in the answer that supports the exchange listing\n"
        "- xrp_custodian: The entity serving as digital asset custodian for XRP held by the ETF\n"
        "- xrp_custodian_url: A URL in the answer supporting the xrp_custodian designation\n"
        "- cash_custodian_admin: The entity serving as administrator/transfer agent and/or cash custodian for the ETF\n"
        "- cash_custodian_admin_url: A URL in the answer supporting the cash_custodian_admin designation\n"
        "- xrp_custodian_hq: City and state of the XRP custodian's headquarters (e.g., 'New York, NY')\n"
        "- xrp_custodian_hq_url: A URL in the answer that supports the XRP custodian HQ city/state\n"
        "- cash_custodian_admin_hq: City and state of the cash custodian/administrator HQ (e.g., 'Pittsburgh, PA')\n"
        "- cash_custodian_admin_hq_url: A URL in the answer that supports the cash custodian/administrator HQ city/state\n\n"
        "Notes:\n"
        "- Accept synonyms: 'management fee' may be called 'expense ratio'; 'digital asset custodian' may appear as 'custodian'; "
        "'administrator' may include 'transfer agent' and/or 'cash custodian'.\n"
        "- URLs can be plain or markdown. Extract the actual URL string. If a URL lacks protocol, prepend 'http://'.\n"
        "- Do not invent values or URLs not present in the answer."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_identity(evaluator: Evaluator, parent_node) -> None:
    """
    Verify the response clearly pertains to Franklin Templeton's spot XRP ETF with ticker XRPZ.
    """
    identity_node = evaluator.add_leaf(
        id="ETF_Identity_Check",
        desc="The response clearly pertains to Franklin Templeton's spot XRP ETF with ticker XRPZ (no conflicting issuer/ticker/product).",
        parent=parent_node,
        critical=True
    )

    claim = (
        "The answer is about Franklin Templeton's spot XRP ETF with ticker XRPZ, and does not mix in a different issuer, "
        "ticker, or unrelated product."
    )
    await evaluator.verify(
        claim=claim,
        node=identity_node,
        additional_instruction=(
            "Check the answer text to ensure the discussed ETF is Franklin Templeton's spot XRP ETF and that the ticker 'XRPZ' is correct. "
            "If the answer mentions a different issuer/ticker/product (e.g., mixes other XRP-related funds or incorrect tickers), mark incorrect."
        )
    )


async def verify_launch_date(evaluator: Evaluator, parent_node, attrs: XRPZAttributes) -> None:
    group = evaluator.add_parallel(
        id="Launch_Date",
        desc="Provide the official trading start (launch) date for XRPZ and a supporting URL.",
        parent=parent_node,
        critical=True
    )

    # Value existence check
    evaluator.add_custom_node(
        result=bool(attrs.launch_date and attrs.launch_date.strip()),
        id="Launch_Date_Value",
        desc="States the official trading start date on the exchange.",
        parent=group,
        critical=True
    )

    # Reference URL verification
    ref_node = evaluator.add_leaf(
        id="Launch_Date_Reference_URL",
        desc="Provides a verifiable reference URL supporting the stated launch/trading start date.",
        parent=group,
        critical=True
    )

    if attrs.launch_date_url and attrs.launch_date_url.strip():
        claim = f"Franklin Templeton's spot XRP ETF (ticker XRPZ) began trading on {attrs.launch_date}."
        await evaluator.verify(
            claim=claim,
            node=ref_node,
            sources=attrs.launch_date_url,
            additional_instruction=(
                "Verify that the webpage explicitly states the ETF's official trading start/launch date. "
                "Accept reasonable date format variants (e.g., 'Jan 10, 2026' vs 'January 10, 2026')."
            )
        )
    else:
        ref_node.score = 0.0
        ref_node.status = "failed"


async def verify_management_fee(evaluator: Evaluator, parent_node, attrs: XRPZAttributes) -> None:
    group = evaluator.add_parallel(
        id="Management_Fee",
        desc="Provide the base annual management fee percentage and a supporting URL.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(attrs.management_fee and attrs.management_fee.strip()),
        id="Management_Fee_Value",
        desc="States the base annual management fee as a percentage.",
        parent=group,
        critical=True
    )

    ref_node = evaluator.add_leaf(
        id="Management_Fee_Reference_URL",
        desc="Provides a verifiable reference URL supporting the stated management fee.",
        parent=group,
        critical=True
    )

    if attrs.management_fee_url and attrs.management_fee_url.strip():
        claim = f"The base annual management fee (expense ratio) of XRPZ is {attrs.management_fee}."
        await evaluator.verify(
            claim=claim,
            node=ref_node,
            sources=attrs.management_fee_url,
            additional_instruction=(
                "Confirm the fee shown (management fee or expense ratio) matches the stated value. "
                "Allow 'bps' vs '%' equivalence (e.g., 19 bps == 0.19%). Ignore temporary waivers; focus on the base annual fee."
            )
        )
    else:
        ref_node.score = 0.0
        ref_node.status = "failed"


async def verify_exchange(evaluator: Evaluator, parent_node, attrs: XRPZAttributes) -> None:
    group = evaluator.add_parallel(
        id="Exchange",
        desc="Provide the specific U.S. stock exchange where XRPZ is listed and a supporting URL.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(attrs.exchange and attrs.exchange.strip()),
        id="Exchange_Value",
        desc="Identifies the specific U.S. stock exchange where XRPZ is listed.",
        parent=group,
        critical=True
    )

    ref_node = evaluator.add_leaf(
        id="Exchange_Reference_URL",
        desc="Provides a verifiable reference URL supporting the stated exchange listing.",
        parent=group,
        critical=True
    )

    if attrs.exchange_url and attrs.exchange_url.strip():
        claim = f"XRPZ is listed on {attrs.exchange}."
        await evaluator.verify(
            claim=claim,
            node=ref_node,
            sources=attrs.exchange_url,
            additional_instruction=(
                "Verify the exchange listing for XRPZ. Accept synonymous naming such as 'Cboe BZX' vs 'Cboe BZX Exchange', "
                "'NYSE Arca' vs 'NYSE Arca Exchange', and 'Nasdaq' vs 'The Nasdaq Stock Market'."
            )
        )
    else:
        ref_node.score = 0.0
        ref_node.status = "failed"


async def verify_xrp_custodian(evaluator: Evaluator, parent_node, attrs: XRPZAttributes) -> None:
    group = evaluator.add_parallel(
        id="XRP_Custodian",
        desc="Provide the XRP digital-asset custodian name and a supporting URL.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(attrs.xrp_custodian and attrs.xrp_custodian.strip()),
        id="XRP_Custodian_Value",
        desc="Identifies the entity designated as custodian for the XRP digital assets held by the ETF.",
        parent=group,
        critical=True
    )

    ref_node = evaluator.add_leaf(
        id="XRP_Custodian_Reference_URL",
        desc="Provides a verifiable reference URL supporting the XRP custodian designation.",
        parent=group,
        critical=True
    )

    if attrs.xrp_custodian_url and attrs.xrp_custodian_url.strip():
        claim = f"The XRP digital asset custodian for XRPZ is {attrs.xrp_custodian}."
        await evaluator.verify(
            claim=claim,
            node=ref_node,
            sources=attrs.xrp_custodian_url,
            additional_instruction=(
                "Confirm the page identifies the XRP (digital asset) custodian for the ETF. "
                "Allow synonyms like 'custodian' or 'digital asset custodian'."
            )
        )
    else:
        ref_node.score = 0.0
        ref_node.status = "failed"


async def verify_cash_admin(evaluator: Evaluator, parent_node, attrs: XRPZAttributes) -> None:
    group = evaluator.add_parallel(
        id="Cash_Custodian_Administrator",
        desc="Provide the cash custodian/administrator (administrator/transfer agent and/or cash custodian) name and a supporting URL.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(attrs.cash_custodian_admin and attrs.cash_custodian_admin.strip()),
        id="Cash_Custodian_Administrator_Value",
        desc="Identifies the entity designated to handle cash custody and/or administrative functions (administrator/transfer agent and/or cash custodian).",
        parent=group,
        critical=True
    )

    ref_node = evaluator.add_leaf(
        id="Cash_Custodian_Administrator_Reference_URL",
        desc="Provides a verifiable reference URL supporting the cash custodian/administrator designation.",
        parent=group,
        critical=True
    )

    if attrs.cash_custodian_admin_url and attrs.cash_custodian_admin_url.strip():
        claim = f"The administrator/transfer agent and/or cash custodian for XRPZ is {attrs.cash_custodian_admin}."
        await evaluator.verify(
            claim=claim,
            node=ref_node,
            sources=attrs.cash_custodian_admin_url,
            additional_instruction=(
                "Verify the entity serving administrative roles (administrator, transfer agent) and/or cash custody for XRPZ. "
                "Allow pages that explicitly list these operational roles for the fund."
            )
        )
    else:
        ref_node.score = 0.0
        ref_node.status = "failed"


async def verify_xrp_custodian_hq(evaluator: Evaluator, parent_node, attrs: XRPZAttributes) -> None:
    group = evaluator.add_parallel(
        id="XRP_Custodian_Headquarters",
        desc="Provide the XRP custodian headquarters location (city and state) and a supporting URL.",
        parent=parent_node,
        critical=True
    )

    # Require both HQ value and the custodian name to avoid ambiguous claims
    evaluator.add_custom_node(
        result=bool(attrs.xrp_custodian_hq and attrs.xrp_custodian_hq.strip() and attrs.xrp_custodian and attrs.xrp_custodian.strip()),
        id="XRP_Custodian_HQ_Value",
        desc="States the XRP custodian headquarters location including both city and state.",
        parent=group,
        critical=True
    )

    ref_node = evaluator.add_leaf(
        id="XRP_Custodian_HQ_Reference_URL",
        desc="Provides a verifiable reference URL supporting the XRP custodian headquarters city/state.",
        parent=group,
        critical=True
    )

    if attrs.xrp_custodian_hq_url and attrs.xrp_custodian_hq_url.strip():
        claim = f"The headquarters of {attrs.xrp_custodian} are located in {attrs.xrp_custodian_hq}."
        await evaluator.verify(
            claim=claim,
            node=ref_node,
            sources=attrs.xrp_custodian_hq_url,
            additional_instruction=(
                "Verify the HQ city/state of the named XRP custodian. Allow common variations/abbreviations "
                "(e.g., 'NY' vs 'New York', 'San Francisco, California' vs 'San Francisco, CA')."
            )
        )
    else:
        ref_node.score = 0.0
        ref_node.status = "failed"


async def verify_cash_admin_hq(evaluator: Evaluator, parent_node, attrs: XRPZAttributes) -> None:
    group = evaluator.add_parallel(
        id="Cash_Custodian_Headquarters",
        desc="Provide the cash custodian/administrator headquarters location (city and state) and a supporting URL.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(attrs.cash_custodian_admin_hq and attrs.cash_custodian_admin_hq.strip() and attrs.cash_custodian_admin and attrs.cash_custodian_admin.strip()),
        id="Cash_Custodian_HQ_Value",
        desc="States the cash custodian/administrator headquarters location including both city and state.",
        parent=group,
        critical=True
    )

    ref_node = evaluator.add_leaf(
        id="Cash_Custodian_HQ_Reference_URL",
        desc="Provides a verifiable reference URL supporting the cash custodian/administrator headquarters city/state.",
        parent=group,
        critical=True
    )

    if attrs.cash_custodian_admin_hq_url and attrs.cash_custodian_admin_hq_url.strip():
        claim = f"The headquarters of {attrs.cash_custodian_admin} are located in {attrs.cash_custodian_admin_hq}."
        await evaluator.verify(
            claim=claim,
            node=ref_node,
            sources=attrs.cash_custodian_admin_hq_url,
            additional_instruction=(
                "Verify the HQ city/state of the named cash custodian/administrator. Allow common location format variants "
                "and abbreviations (e.g., 'PA' vs 'Pennsylvania')."
            )
        )
    else:
        ref_node.score = 0.0
        ref_node.status = "failed"


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Franklin Templeton XRPZ ETF research task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root container; we'll add a critical sequential main node under it
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

    # Create top-level sequential critical node (as per rubric root)
    main = evaluator.add_sequential(
        id="Franklin_Templeton_XRPZ_ETF_Research",
        desc="Verify the response is about Franklin Templeton's spot XRP ETF (ticker XRPZ) and that it provides all 7 requested attributes, each with its own supporting reference URL.",
        parent=root,
        critical=True
    )

    # Extract all attributes from the answer
    attrs = await evaluator.extract(
        prompt=prompt_extract_xrpz_attributes(),
        template_class=XRPZAttributes,
        extraction_name="xrpz_attributes"
    )

    # 1) Identity check
    await verify_identity(evaluator, main)

    # 2) Requested attributes (parallel critical group)
    attrs_main = evaluator.add_parallel(
        id="Requested_Attributes_With_Citations",
        desc="Provide each of the 7 requested attributes and a supporting reference URL for that specific attribute.",
        parent=main,
        critical=True
    )

    # Attribute-by-attribute verification
    await verify_launch_date(evaluator, attrs_main, attrs)
    await verify_management_fee(evaluator, attrs_main, attrs)
    await verify_exchange(evaluator, attrs_main, attrs)
    await verify_xrp_custodian(evaluator, attrs_main, attrs)
    await verify_cash_admin(evaluator, attrs_main, attrs)
    await verify_xrp_custodian_hq(evaluator, attrs_main, attrs)
    await verify_cash_admin_hq(evaluator, attrs_main, attrs)

    # Return evaluation summary
    return evaluator.get_summary()