import asyncio
import logging
from typing import Any, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "franklin_xrp_etf_filing_analysis"
TASK_DESCRIPTION = (
    "Franklin Templeton filed a registration statement with the SEC for its XRP exchange-traded fund (ETF). "
    "Based on this SEC filing: 1. What is the filing date and form type of this registration statement? "
    "2. What entity is identified as the sponsor of the Franklin XRP ETF? "
    "3. What entity is identified as the XRP custodian? "
    "4. What is the registration status of the Trust under the Investment Company Act of 1940? "
    "5. What is the registration status of the Sponsor with the SEC as an investment adviser? "
    "Provide the official SEC EDGAR filing URL as your reference."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FilingExtraction(BaseModel):
    """
    Extraction model for Franklin XRP ETF SEC filing details from the agent's answer.
    """
    edgar_url: Optional[str] = None  # Must be a single URL to the official SEC EDGAR filing (sec.gov/Archives/edgar/...)
    filing_date: Optional[str] = None  # e.g., "March 11, 2025"
    form_type: Optional[str] = None  # e.g., "S-1", "S-1/A"
    sponsor_entity: Optional[str] = None  # e.g., "Franklin Holdings, LLC"
    custodian_entity: Optional[str] = None  # e.g., "Coinbase Custody Trust Company, LLC"
    trust_1940_act_status: Optional[str] = None  # short phrase, e.g., "not registered", "registered"
    sponsor_adviser_status: Optional[str] = None  # short phrase, e.g., "not registered", "registered"


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_filing_details() -> str:
    return """
    Extract the requested details about Franklin Templeton’s XRP ETF registration statement from the provided answer.
    Return the information in the following fields:

    1) edgar_url: The single official SEC EDGAR filing URL that the answer cites for this registration statement.
       Requirements:
       - It must be a direct SEC EDGAR link, typically containing "sec.gov/Archives/edgar/".
       - If multiple URLs are present, return only the one that the answer uses as the primary official SEC filing reference.
       - If the answer does not include such a URL, set this field to null.

    2) filing_date: The filing date stated in the answer, preferably in a human-readable format (e.g., "March 11, 2025").
    3) form_type: The SEC form type stated in the answer (e.g., "S-1", "S-1/A").
    4) sponsor_entity: The sponsor entity named in the answer (e.g., "Franklin Holdings, LLC").
    5) custodian_entity: The XRP custodian named in the answer (e.g., "Coinbase Custody Trust Company, LLC").
    6) trust_1940_act_status: The Trust’s status under the Investment Company Act of 1940 as described in the answer.
       Use a short phrase such as "not registered" or "registered".
    7) sponsor_adviser_status: The Sponsor’s SEC registration status as an investment adviser as described in the answer.
       Use a short phrase such as "not registered" or "registered".

    Rules:
    - Extract exactly what the answer states. Do not infer or invent any details.
    - If a requested field is not mentioned, set it to null.
    - For URLs, extract the actual URL string (plain or markdown), and ensure it includes a protocol (http/https).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_official_edgar_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.lower().strip()
    return ("sec.gov" in u) and ("/archives/edgar/" in u)


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_source_verification(
    evaluator: Evaluator,
    parent_node,
    data: FilingExtraction,
) -> None:
    """
    Build and execute the Source_Filing_and_Citation subtree:
    - Official_EDGAR_URL_Provided (critical, custom existence/pattern check)
    - URL_Refers_to_Target_Filing (critical, verify by URL)
    - All_Claims_Supported_by_Cited_Filing (critical, verify all five facts by URL)
    """
    source_node = evaluator.add_parallel(
        id="Source_Filing_and_Citation",
        desc="Provide and use an official SEC EDGAR filing URL as the reference source for the answer.",
        parent=parent_node,
        critical=False,
    )

    # Official_EDGAR_URL_Provided: existence + format check (critical)
    evaluator.add_custom_node(
        result=_is_official_edgar_url(data.edgar_url),
        id="Official_EDGAR_URL_Provided",
        desc="Provides a single official SEC EDGAR filing URL (sec.gov/Archives/edgar/...) as the reference.",
        parent=source_node,
        critical=True,
    )

    # URL_Refers_to_Target_Filing: verify that the URL is the Franklin XRP ETF registration statement
    url_target_leaf = evaluator.add_leaf(
        id="URL_Refers_to_Target_Filing",
        desc="The provided EDGAR URL corresponds to Franklin Templeton’s XRP ETF registration statement being analyzed (the same filing from which the five requested details are extracted).",
        parent=source_node,
        critical=True,
    )
    claim_target = (
        "This EDGAR page corresponds to the Franklin XRP ETF registration statement (Form S-1 or S-1/A) "
        "by Franklin Templeton or related Franklin entities. It is the filing the answer relies on for the requested details."
    )
    await evaluator.verify(
        claim=claim_target,
        node=url_target_leaf,
        sources=data.edgar_url,
        additional_instruction=(
            "Confirm by the page title and content that the document is the registration statement for the Franklin XRP ETF. "
            "Look for mentions like 'Franklin XRP ETF', sponsor names, custodian details, and form type."
        ),
    )

    # All_Claims_Supported_by_Cited_Filing: verify all five outputs are supported by the cited filing
    all_supported_leaf = evaluator.add_leaf(
        id="All_Claims_Supported_by_Cited_Filing",
        desc="All five requested outputs are presented as being supported by the cited EDGAR filing (no uncited/unsupported factual assertions).",
        parent=source_node,
        critical=True,
    )
    # Build a composite claim listing each extracted attribute
    filing_date_str = data.filing_date or "UNKNOWN"
    form_type_str = data.form_type or "UNKNOWN"
    sponsor_str = data.sponsor_entity or "UNKNOWN"
    custodian_str = data.custodian_entity or "UNKNOWN"
    trust_status_str = data.trust_1940_act_status or "UNKNOWN"
    sponsor_adv_status_str = data.sponsor_adviser_status or "UNKNOWN"

    composite_claim = (
        f"On this EDGAR filing, the following details are explicitly supported:\n"
        f"1) Form type: '{form_type_str}' and filing date: '{filing_date_str}'.\n"
        f"2) Sponsor: '{sponsor_str}'.\n"
        f"3) XRP custodian: '{custodian_str}'.\n"
        f"4) Trust status under the Investment Company Act of 1940: '{trust_status_str}'.\n"
        f"5) Sponsor’s SEC registration status as an investment adviser: '{sponsor_adv_status_str}'.\n"
        "All these are supported by the cited EDGAR document and do not rely on other sources."
    )
    await evaluator.verify(
        claim=composite_claim,
        node=all_supported_leaf,
        sources=data.edgar_url,
        additional_instruction=(
            "Check that each enumerated item appears in the filing text (title page, summary, or risk sections). "
            "If any item is missing or contradicted, mark the claim as not supported."
        ),
    )


async def build_details_verification(
    evaluator: Evaluator,
    parent_node,
    data: FilingExtraction,
) -> None:
    """
    Build and execute the Extract_Requested_Details subtree:
    - Filing_Date_and_Form_Type (critical)
    - Sponsor_Entity_and_Distinction (critical)
    - XRP_Custodian (critical)
    - Trust_1940_Act_Registration_Status (critical)
    - Sponsor_SEC_Investment_Adviser_Status (critical)
    """
    details_node = evaluator.add_parallel(
        id="Extract_Requested_Details",
        desc="Extract and report the five requested attributes from the filing (date/form, sponsor, custodian, 1940 Act status, sponsor adviser status).",
        parent=parent_node,
        critical=False,
    )

    # Filing_Date_and_Form_Type
    filing_leaf = evaluator.add_leaf(
        id="Filing_Date_and_Form_Type",
        desc="Correctly states the filing date and form type for the registration statement (Form S-1; March 11, 2025).",
        parent=details_node,
        critical=True,
    )
    filing_claim = (
        f"The EDGAR filing shows the form type is '{data.form_type or ''}' "
        f"and the filing date is '{data.filing_date or ''}'."
    )
    await evaluator.verify(
        claim=filing_claim,
        node=filing_leaf,
        sources=data.edgar_url,
        additional_instruction=(
            "Verify the form type (e.g., S-1 or S-1/A) and the filing date using the document header/cover page in EDGAR. "
            "Minor formatting variations in date text are acceptable."
        ),
    )

    # Sponsor_Entity_and_Distinction
    sponsor_leaf = evaluator.add_leaf(
        id="Sponsor_Entity_and_Distinction",
        desc="Correctly identifies the sponsor as Franklin Holdings, LLC and clearly distinguishes this sponsor entity from the broader Franklin Templeton organization/brand.",
        parent=details_node,
        critical=True,
    )
    sponsor_claim = (
        f"The EDGAR filing identifies the sponsor as '{data.sponsor_entity or ''}'. "
        "This sponsor is a specific legal entity and is distinct from the broader 'Franklin Templeton' brand."
    )
    await evaluator.verify(
        claim=sponsor_claim,
        node=sponsor_leaf,
        sources=data.edgar_url,
        additional_instruction=(
            "Locate the 'Sponsor' designation in the filing. Confirm the named entity (e.g., Franklin Holdings, LLC). "
            "It should be clear that the sponsor entity is a specific legal entity separate from the general marketing brand."
        ),
    )

    # XRP_Custodian
    custodian_leaf = evaluator.add_leaf(
        id="XRP_Custodian",
        desc="Correctly identifies the XRP custodian as Coinbase Custody Trust Company, LLC.",
        parent=details_node,
        critical=True,
    )
    custodian_claim = (
        f"The EDGAR filing identifies the XRP custodian as '{data.custodian_entity or ''}'."
    )
    await evaluator.verify(
        claim=custodian_claim,
        node=custodian_leaf,
        sources=data.edgar_url,
        additional_instruction=(
            "Find the section that names the custodian (e.g., 'Custodian' or 'XRP Custodian'). "
            "Confirm the exact entity name listed in the filing."
        ),
    )

    # Trust_1940_Act_Registration_Status
    trust_status_leaf = evaluator.add_leaf(
        id="Trust_1940_Act_Registration_Status",
        desc="Correctly states the Trust’s status under the Investment Company Act of 1940 (NOT registered).",
        parent=details_node,
        critical=True,
    )
    trust_status_claim = (
        f"The EDGAR filing states that the Trust is '{data.trust_1940_act_status or ''}' under the Investment Company Act of 1940."
    )
    await evaluator.verify(
        claim=trust_status_claim,
        node=trust_status_leaf,
        sources=data.edgar_url,
        additional_instruction=(
            "Check the filing's disclosures about the Investment Company Act of 1940. "
            "The filing should explicitly state whether the Trust is registered or not registered under the 1940 Act."
        ),
    )

    # Sponsor_SEC_Investment_Adviser_Status
    sponsor_adv_leaf = evaluator.add_leaf(
        id="Sponsor_SEC_Investment_Adviser_Status",
        desc="Correctly states the Sponsor’s SEC registration status as an investment adviser (NOT registered).",
        parent=details_node,
        critical=True,
    )
    sponsor_adv_claim = (
        f"The EDGAR filing states the Sponsor is '{data.sponsor_adviser_status or ''}' registered with the SEC as an investment adviser."
    )
    await evaluator.verify(
        claim=sponsor_adv_claim,
        node=sponsor_adv_leaf,
        sources=data.edgar_url,
        additional_instruction=(
            "Locate the filing statement regarding whether the Sponsor is registered with the SEC as an investment adviser. "
            "Confirm the status (registered or not registered)."
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
    Evaluate an answer for the Franklin XRP ETF filing analysis task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Source verification first, then details
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

    # Build root node with task description (already created by initialize).
    # Add child nodes under root according to rubric
    franklin_root = evaluator.add_sequential(
        id="Franklin_XRP_ETF_Filing_Analysis",
        desc="Evaluate whether the answer correctly extracts the five requested attributes from the official SEC EDGAR filing for Franklin Templeton’s XRP ETF and cites the official filing.",
        parent=root,
        critical=False,
    )

    # 1) Extract structured details from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_filing_details(),
        template_class=FilingExtraction,
        extraction_name="filing_extraction",
    )

    # 2) Build source verification subtree
    await build_source_verification(evaluator, franklin_root, extracted)

    # 3) Build details verification subtree
    await build_details_verification(evaluator, franklin_root, extracted)

    # 4) Return standardized summary
    return evaluator.get_summary()