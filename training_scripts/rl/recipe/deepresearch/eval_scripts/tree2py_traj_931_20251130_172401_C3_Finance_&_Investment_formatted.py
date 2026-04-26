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
TASK_ID = "ibit_sec_filing"
TASK_DESCRIPTION = (
    "Research the regulatory structure of the iShares Bitcoin Trust ETF (ticker: IBIT) using its official SEC filings. "
    "Locate the most recent Form S-1 registration statement or prospectus filed with the SEC for IBIT. From this filing, "
    "extract and provide the following information: (1) The URL of the SEC filing you are referencing, (2) The legal name of "
    "IBIT's sponsor (the entity that created and manages the trust), (3) The complete business address of the sponsor "
    "(including street address, city, state, and ZIP code), (4) The name of the Bitcoin custodian (the entity responsible for "
    "holding IBIT's Bitcoin assets), and (5) A description of the custodian's role and function in safeguarding IBIT's Bitcoin "
    "holdings. All information must be extracted directly from the SEC filing and include the specific URL reference to the document."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class IBITFilingExtraction(BaseModel):
    """
    Structured info pulled from the agent's answer text.
    """
    filing_url: Optional[str] = None

    sponsor_name: Optional[str] = None
    sponsor_address: Optional[str] = None

    custodian_name: Optional[str] = None

    # Optional short snippets or paraphrases from the answer text describing the roles;
    # They are recorded for transparency but verification is done directly on the SEC filing.
    role_safekeeping: Optional[str] = None
    role_segregated_accounts: Optional[str] = None
    role_cold_hot_storage: Optional[str] = None
    role_trading_balance_prime_broker: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_ibit_filing_info() -> str:
    return """
    Extract the information about the iShares Bitcoin Trust (IBIT) SEC filing as presented in the answer.

    You must extract the following fields if they are explicitly present in the answer:
    1) filing_url: The single, primary official SEC.gov/EDGAR URL to the referenced IBIT filing. Prefer a Form S-1, S-1/A, or a Prospectus filing page.
       - If multiple SEC URLs are provided, choose the most recent one mentioned for iShares Bitcoin Trust (IBIT); if recency is unclear, choose the first clearly relevant SEC filing URL for the Trust.
       - Only return a URL on the sec.gov domain; do not use third-party mirrors or summaries.
    2) sponsor_name: The legal name of IBIT's sponsor exactly as written in the answer.
    3) sponsor_address: The complete sponsor business address (street, city, state, ZIP) as written in the answer, ideally in a single line.
    4) custodian_name: The name of the Bitcoin custodian as written in the answer.
    5) role_safekeeping: A short snippet or paraphrase from the answer describing that the custodian is responsible for safekeeping the Trust’s bitcoin holdings (if present).
    6) role_segregated_accounts: A short snippet/paraphrase stating the Trust’s bitcoin is held in segregated accounts separate from the custodian’s own assets/other customers (if present).
    7) role_cold_hot_storage: A short snippet/paraphrase indicating custody involves both cold storage and hot storage (e.g., mentions of “Cold Vault Balance” and “Hot Vault Balance”), if present.
    8) role_trading_balance_prime_broker: A short snippet/paraphrase indicating a Trading Balance may be temporarily held with the Prime Broker (e.g., Coinbase, Inc.) for creations/redemptions and/or fees/expenses, if present.

    Rules:
    - Extract only what the answer explicitly states. Do not infer any values.
    - If a field is not present in the answer, return null for that field.
    - For URLs, extract the actual URL text. If no official SEC.gov/EDGAR URL is given, set filing_url to null.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_filing_node(
    evaluator: Evaluator,
    root_node,
    extracted: IBITFilingExtraction,
) -> None:
    """
    Build the 'locate_filing' subtree and verify the SEC filing URL is an official SEC page
    and that it represents an S-1/S-1/A or prospectus for iShares Bitcoin Trust.
    """
    locate_node = evaluator.add_sequential(
        id="locate_filing",
        desc="Locate the most recent SEC registration statement/prospectus filing for IBIT (Form S-1, S-1/A, or prospectus as allowed) on SEC EDGAR and reference it",
        parent=root_node,
        critical=True,
    )

    filing_leaf = evaluator.add_leaf(
        id="filing_url",
        desc="Provide an official SEC.gov/EDGAR URL for the referenced (most recent) IBIT Form S-1/S-1A or prospectus filing",
        parent=locate_node,
        critical=True,
    )

    url = extracted.filing_url or ""
    claim = (
        "This webpage is an official SEC.gov/EDGAR filing page for the iShares Bitcoin Trust "
        "(not a third-party site) and is a registration statement (Form S-1 or S-1/A) or a "
        "prospectus for the Trust."
    )
    add_ins = (
        "Judge the claim using the provided webpage only. "
        "Confirm the domain is sec.gov and that the page represents an official SEC filing. "
        "Accept if the page clearly shows 'Form S-1', 'S-1/A', 'Prospectus', or a prospectus form "
        "(e.g., 424B3/424B4) for the iShares Bitcoin Trust. "
        "The filing might or might not explicitly show the ticker 'IBIT'; the Trust name is sufficient."
    )

    await evaluator.verify(
        claim=claim,
        node=filing_leaf,
        sources=url if url else None,
        additional_instruction=add_ins,
    )


async def build_and_verify_information_nodes(
    evaluator: Evaluator,
    root_node,
    extracted: IBITFilingExtraction,
) -> None:
    """
    Build the 'extract_information' subtree and verify fields against the referenced SEC filing.
    """
    info_parent = evaluator.add_parallel(
        id="extract_information",
        desc="From the referenced SEC filing, extract and provide all requested fields",
        parent=root_node,
        critical=True,
    )

    filing_url = extracted.filing_url or None

    # Sponsor name
    sponsor_name_leaf = evaluator.add_leaf(
        id="sponsor_name",
        desc="Provide the legal name of IBIT's sponsor as stated in the referenced SEC filing",
        parent=info_parent,
        critical=True,
    )
    sponsor_name = extracted.sponsor_name or ""
    sponsor_name_claim = (
        f"The SEC filing states that the sponsor of the iShares Bitcoin Trust is '{sponsor_name}'."
        if sponsor_name else
        "The SEC filing clearly identifies the legal sponsor of the iShares Bitcoin Trust."
    )
    await evaluator.verify(
        claim=sponsor_name_claim,
        node=sponsor_name_leaf,
        sources=filing_url,
        additional_instruction=(
            "Look for sections that define the 'Sponsor' of the Trust. "
            "Confirm the exact legal entity named as Sponsor. "
            "Minor punctuation or capitalization differences are acceptable."
        ),
    )

    # Sponsor address
    sponsor_address_leaf = evaluator.add_leaf(
        id="sponsor_address",
        desc="Provide the sponsor's complete business address (street, city, state, ZIP) as stated in the referenced SEC filing",
        parent=info_parent,
        critical=True,
    )
    sponsor_address = extracted.sponsor_address or ""
    sponsor_addr_claim = (
        f"The SEC filing lists the sponsor's complete business address as '{sponsor_address}'."
        if sponsor_address else
        "The SEC filing lists a complete sponsor business address that includes street, city, state, and ZIP code."
    )
    await evaluator.verify(
        claim=sponsor_addr_claim,
        node=sponsor_address_leaf,
        sources=filing_url,
        additional_instruction=(
            "Verify the sponsor's business address as shown in the filing. "
            "Allow minor formatting differences (commas, line breaks). "
            "The address should include street, city, state, and ZIP code."
        ),
    )

    # Custodian name
    custodian_name_leaf = evaluator.add_leaf(
        id="custodian_name",
        desc="Provide the name of the Bitcoin custodian for IBIT as stated in the referenced SEC filing",
        parent=info_parent,
        critical=True,
    )
    custodian_name = extracted.custodian_name or ""
    custodian_name_claim = (
        f"The SEC filing states that the Trust's bitcoin custodian is '{custodian_name}'."
        if custodian_name else
        "The SEC filing identifies the entity that serves as the Trust's bitcoin custodian."
    )
    await evaluator.verify(
        claim=custodian_name_claim,
        node=custodian_name_leaf,
        sources=filing_url,
        additional_instruction=(
            "Look for terms such as 'Bitcoin Custodian' or 'Custodian' in the filing. "
            "Confirm the named entity responsible for holding the Trust's bitcoin."
        ),
    )

    # Custodian role details (parallel, all critical)
    role_parent = evaluator.add_parallel(
        id="custodian_role",
        desc="Describe the custodian's role and function in safeguarding IBIT's Bitcoin holdings, matching the SEC filing’s described mechanisms",
        parent=info_parent,
        critical=True,
    )

    # 1) Safekeeping responsibility
    role_safe_leaf = evaluator.add_leaf(
        id="role_safekeeping",
        desc="States that the Bitcoin custodian is responsible for safekeeping the Trust’s bitcoin holdings (as described in the filing)",
        parent=role_parent,
        critical=True,
    )
    safe_claim = (
        "The SEC filing states that the Bitcoin custodian is responsible for the safekeeping of the Trust’s bitcoin holdings."
    )
    await evaluator.verify(
        claim=safe_claim,
        node=role_safe_leaf,
        sources=filing_url,
        additional_instruction=(
            "Check the 'Custody' or similar sections. "
            "Accept paraphrased language indicating the custodian safeguards or holds the Trust's bitcoin on behalf of the Trust."
        ),
    )

    # 2) Segregated accounts
    role_segr_leaf = evaluator.add_leaf(
        id="role_segregated_accounts",
        desc="States that the Trust’s bitcoin is held in segregated accounts separate from the custodian’s own assets/other customers (as described in the filing)",
        parent=role_parent,
        critical=True,
    )
    segr_claim = (
        "The SEC filing states that the Trust’s bitcoin is held in segregated accounts separate from the custodian’s own assets and from other customers’ assets."
    )
    await evaluator.verify(
        claim=segr_claim,
        node=role_segr_leaf,
        sources=filing_url,
        additional_instruction=(
            "Look for explicit references to 'segregated accounts' or equivalent statements making clear that the Trust’s assets are held separately from the custodian's and other customers’ assets."
        ),
    )

    # 3) Cold and hot storage
    role_cold_hot_leaf = evaluator.add_leaf(
        id="role_cold_hot_storage",
        desc="States that custody involves cold storage and hot storage (e.g., Cold Vault Balance / Hot Vault Balance) as described in the filing",
        parent=role_parent,
        critical=True,
    )
    cold_hot_claim = (
        "The SEC filing describes that custody involves both cold storage and hot storage (for example, terms like 'Cold Vault Balance' and 'Hot Vault Balance' may be used)."
    )
    await evaluator.verify(
        claim=cold_hot_claim,
        node=role_cold_hot_leaf,
        sources=filing_url,
        additional_instruction=(
            "Confirm that both cold storage and hot storage are part of the custody framework, even if exact phrasing varies."
        ),
    )

    # 4) Trading Balance with Prime Broker
    role_trading_leaf = evaluator.add_leaf(
        id="role_trading_balance_prime_broker",
        desc="States that a Trading Balance may be temporarily held with the Prime Broker (Coinbase, Inc.) for creations/redemptions and/or fee/expense payments as described in the filing",
        parent=role_parent,
        critical=True,
    )
    trading_claim = (
        "The SEC filing states that a 'Trading Balance' may be temporarily held with the Prime Broker (such as Coinbase, Inc.) "
        "for creation/redemption activity and/or the payment of fees and expenses."
    )
    await evaluator.verify(
        claim=trading_claim,
        node=role_trading_leaf,
        sources=filing_url,
        additional_instruction=(
            "Look for references to 'Trading Balance' or an equivalent transient balance held with a prime broker for operational purposes. "
            "If the filing names Coinbase, Inc. as Prime Broker for such Trading Balance, accept it."
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
    Evaluate an answer for the IBIT SEC filing extraction and verification task.
    """
    # Initialize evaluator (root uses sequential strategy to enforce ordering: locate filing -> extract info)
    evaluator = Evaluator()
    root = evaluator.initialize(
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
        default_model=model,
    )

    # Extract claimed data from the agent's answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_ibit_filing_info(),
        template_class=IBITFilingExtraction,
        extraction_name="ibit_filing_extraction",
    )

    # Build and verify the filing URL first (sequential dependency)
    await build_and_verify_filing_node(evaluator, root, extracted)

    # Then, verify all requested fields from that filing
    await build_and_verify_information_nodes(evaluator, root, extracted)

    # Return final evaluation summary
    return evaluator.get_summary()