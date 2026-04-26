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
TASK_ID = "us_telecom_cband_5g_flagship_2024"
TASK_DESCRIPTION = (
    "Identify three major U.S. telecommunications carriers that meet all of the following criteria as of December 2023: "
    "(1) The carrier was a winning bidder in FCC Auction 107 (the C-band spectrum auction for frequencies in the 3.7-3.98 GHz range); "
    "(2) The carrier spent at least $20 billion in gross bids in this C-band auction; "
    "(3) The carrier's 5G network covers at least 300 million people in the United States; "
    "(4) The carrier has U.S. headquarters or significant U.S. operations; "
    "(5) The carrier offered for sale at least one major flagship smartphone that was officially announced between January 2024 and September 2024 (inclusive), "
    "specifically from one of these manufacturers: Apple (iPhone 16), Samsung (Galaxy S24 series), or Google (Pixel 9 series). "
    "For each carrier, provide: the carrier's name, the approximate amount spent in the C-band auction, the approximate 5G coverage (in millions of people), "
    "at least one qualifying flagship smartphone model they offered, the announcement date of that smartphone, and a reference URL confirming their C-band auction participation and spending amount."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CarrierInfo(BaseModel):
    name: Optional[str] = None
    cband_amount: Optional[str] = None  # e.g., "$45 billion", "about $23B"
    coverage_millions: Optional[str] = None  # e.g., "300+ million", "310M"
    phone_model: Optional[str] = None  # e.g., "iPhone 16 Pro", "Galaxy S24 Ultra", "Pixel 9 Pro"
    phone_manufacturer: Optional[str] = None  # Apple / Samsung / Google
    phone_announcement_date: Optional[str] = None  # e.g., "2024-09-10", "January 2024"
    cband_reference_url: Optional[str] = None  # URL confirming Auction 107 and spending amount
    cband_other_urls: List[str] = Field(default_factory=list)  # any extra URLs about auction/spending
    coverage_urls: List[str] = Field(default_factory=list)  # URLs supporting coverage
    us_presence_urls: List[str] = Field(default_factory=list)  # URLs supporting US HQ/operations
    phone_urls: List[str] = Field(default_factory=list)  # URLs supporting phone manufacturer, offering, and announcement


class CarriersExtraction(BaseModel):
    carriers: List[CarrierInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_carriers() -> str:
    return """
    Extract up to three U.S. telecommunications carriers mentioned in the answer that the answer claims meet all specified criteria. 
    For each carrier, extract the following fields exactly as stated in the answer:

    1. name: Carrier name.
    2. cband_amount: The approximate gross bids amount the carrier spent in FCC Auction 107 (the C-band auction). Keep as a string (e.g., "$45 billion", "about $23B").
    3. coverage_millions: The claimed 5G coverage in the U.S. in millions of people (e.g., "300+ million", "310M").
    4. phone_model: At least one qualifying flagship smartphone model the carrier offered for sale.
    5. phone_manufacturer: The manufacturer for the model (Apple, Samsung, or Google).
    6. phone_announcement_date: The announcement date as claimed in the answer (string format; month/year or YYYY-MM-DD is acceptable).
    7. cband_reference_url: A single URL that the answer cites to confirm the carrier's Auction 107 (C-band) participation AND its spending amount.
    8. cband_other_urls: Any other URLs related to this carrier's C-band Auction 107 bidding/spending that are cited in the answer (array; can be empty).
    9. coverage_urls: URLs cited in the answer that support the coverage claim (array; can be empty).
    10. us_presence_urls: URLs cited in the answer that support having U.S. HQ or significant U.S. operations (array; can be empty).
    11. phone_urls: URLs cited in the answer that support the carrier offered the specified flagship phone and the phone's announcement timing (array; can be empty).

    IMPORTANT:
    - Only extract URLs that are explicitly present in the answer text (including markdown links). Do not invent or infer URLs.
    - If any field is not present in the answer, set it to null (for single fields) or an empty array (for array fields).
    - Return results in a JSON object with a 'carriers' array. Include all carriers mentioned; we will later use only the first three.

    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_urls(*url_groups: List[Optional[str] | List[str]]) -> Optional[List[str]]:
    """Flatten input URL groups (mix of str or list[str]), remove None/empty, deduplicate, preserve order. Return None if result empty."""
    seen = set()
    merged: List[str] = []
    for group in url_groups:
        if group is None:
            continue
        if isinstance(group, list):
            for u in group:
                if not u:
                    continue
                s = u.strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                merged.append(s)
        elif isinstance(group, str):
            s = group.strip()
            if s and s not in seen:
                seen.add(s)
                merged.append(s)
        else:
            continue
    return merged if merged else None


def _safe_name(c: CarrierInfo, idx: int) -> str:
    return c.name.strip() if c.name else f"Carrier #{idx + 1}"


# --------------------------------------------------------------------------- #
# Verification logic for a single carrier                                     #
# --------------------------------------------------------------------------- #
async def verify_single_carrier(
    evaluator: Evaluator,
    parent_node,
    carrier: CarrierInfo,
    idx: int,
) -> None:
    """
    Build verification subtree for one carrier and run checks.
    """

    # Top-level node for this carrier (parallel aggregation; non-critical to allow partial credit across carriers)
    carrier_node = evaluator.add_parallel(
        id=f"carrier_{idx + 1}",
        desc=[
            "First qualifying carrier meets all requirements",
            "Second qualifying carrier meets all requirements",
            "Third qualifying carrier meets all requirements",
        ][idx],
        parent=parent_node,
        critical=False,
    )

    # Prepare commonly used URL bundles
    cband_urls = _normalize_urls([carrier.cband_reference_url] if carrier.cband_reference_url else [], carrier.cband_other_urls)
    coverage_urls = _normalize_urls(carrier.coverage_urls)
    us_presence_urls = _normalize_urls(carrier.us_presence_urls)
    phone_urls = _normalize_urls(carrier.phone_urls)

    # 1) FCC Auction 107 C-band participation
    node_participation = evaluator.add_leaf(
        id=f"carrier_{idx + 1}_cband_participation",
        desc="Carrier was a winning bidder in FCC Auction 107 (C-band, 3.7-3.98 GHz)",
        parent=carrier_node,
        critical=True,
    )
    claim_participation = (
        f"{_safe_name(carrier, idx)} was a winning bidder in FCC Auction 107, the C-band spectrum auction for frequencies in the 3.7–3.98 GHz range."
    )

    # 2) Auction spending ≥ $20B
    node_spending = evaluator.add_leaf(
        id=f"carrier_{idx + 1}_auction_spending",
        desc="Carrier spent at least $20 billion in gross bids in the C-band auction",
        parent=carrier_node,
        critical=True,
    )
    claim_spending = (
        f"{_safe_name(carrier, idx)} spent at least $20 billion in gross bids in FCC Auction 107."
    )

    # 3) 5G coverage ≥ 300M people
    node_coverage = evaluator.add_leaf(
        id=f"carrier_{idx + 1}_5g_coverage",
        desc="Carrier's 5G network covers at least 300 million people in the United States",
        parent=carrier_node,
        critical=True,
    )
    claim_coverage = (
        f"{_safe_name(carrier, idx)}'s 5G network covers at least 300 million people in the United States."
    )

    # 4) U.S. HQ or significant U.S. operations
    node_us_presence = evaluator.add_leaf(
        id=f"carrier_{idx + 1}_us_presence",
        desc="Carrier has U.S. headquarters or significant U.S. operations",
        parent=carrier_node,
        critical=True,
    )
    claim_us_presence = (
        f"{_safe_name(carrier, idx)} has U.S. headquarters or significant U.S. operations."
    )

    # 5) Flagship smartphone conditions (parallel sub-node, critical)
    phone_parent = evaluator.add_parallel(
        id=f"carrier_{idx + 1}_flagship_phone",
        desc="Carrier offered at least one qualifying flagship smartphone announced between January-September 2024",
        parent=carrier_node,
        critical=True,
    )

    # 5a) Announcement window January–September 2024 inclusive
    node_phone_announcement = evaluator.add_leaf(
        id=f"carrier_{idx + 1}_phone_announcement",
        desc="The flagship phone was announced between January 2024 and September 2024 (inclusive)",
        parent=phone_parent,
        critical=True,
    )
    claim_phone_announcement = (
        f"The smartphone model '{carrier.phone_model or ''}' was officially announced between January 1, 2024 and September 30, 2024 (inclusive)."
    )

    # 5b) Manufacturer eligibility AND offered by the carrier
    node_phone_mfr = evaluator.add_leaf(
        id=f"carrier_{idx + 1}_phone_manufacturer",
        desc="The flagship phone is from Apple (iPhone 16), Samsung (Galaxy S24 series), or Google (Pixel 9 series)",
        parent=phone_parent,
        critical=True,
    )
    claim_phone_mfr = (
        f"{_safe_name(carrier, idx)} offered for sale the flagship smartphone '{carrier.phone_model or ''}', "
        f"which belongs to one of the eligible series: Apple iPhone 16 (including Pro/Pro Max/Plus variants), "
        f"Samsung Galaxy S24 (including S24+/Ultra), or Google Pixel 9 (including Pro/XL variants)."
    )

    # 6) Explicit reference URL for C-band participation and spending
    node_reference_url = evaluator.add_leaf(
        id=f"carrier_{idx + 1}_reference_url",
        desc="Provide reference URL supporting the carrier's C-band auction participation and spending",
        parent=carrier_node,
        critical=True,
    )
    claim_reference_url = (
        f"The reference page confirms that {_safe_name(carrier, idx)} was a winning bidder in FCC Auction 107 and shows its gross bids amount (approximately and at least $20 billion)."
    )

    # Prepare batch verification calls for this carrier
    tasks = [
        (
            claim_participation,
            cband_urls,
            node_participation,
            "Confirm the company is listed as a winning bidder in FCC Auction 107 (C-band 3.7–3.98 GHz). "
            "Accept synonyms like 'license winner' or 'won blocks'."
        ),
        (
            claim_spending,
            cband_urls,
            node_spending,
            "Verify the page indicates the company's Auction 107 spending amount was at least $20B. "
            "Accept approximate values (e.g., '~$23B', 'over $45B')."
        ),
        (
            claim_coverage,
            coverage_urls,
            node_coverage,
            "Verify that coverage claims specify at least 300 million people in the U.S. "
            "Accept phrases like '300+ million', 'over 300 million'."
        ),
        (
            claim_us_presence,
            us_presence_urls,
            node_us_presence,
            "Confirm U.S. headquarters location or major U.S. operations (e.g., nationwide network, U.S. corporate HQ)."
        ),
        (
            claim_phone_announcement,
            phone_urls,
            node_phone_announcement,
            "Verify announcement date is within Jan 1, 2024 to Sep 30, 2024 inclusive. "
            "Use manufacturer or credible tech news pages. "
            f"If a specific date is provided ('{carrier.phone_announcement_date or ''}'), ensure it falls within the range."
        ),
        (
            claim_phone_mfr,
            phone_urls,
            node_phone_mfr,
            "Confirm the phone belongs to allowed series: iPhone 16 family (Apple), Galaxy S24 family (Samsung), or Pixel 9 family (Google). "
            "Also confirm the carrier offered/sold this phone (carrier site, press release, or product page). "
            "Allow model variants (Plus/Pro/Pro Max/Ultra/XL)."
        ),
        (
            claim_reference_url,
            carrier.cband_reference_url,
            node_reference_url,
            "The page must mention Auction 107/C-band and the company's spending figure (approximate acceptable) and confirm participation."
        ),
    ]

    # Run verifications (in parallel within this carrier scope)
    await evaluator.batch_verify(tasks)


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
    Evaluate an answer for the U.S. telecom carriers C-band/5G/flagship criteria task.
    """

    # Initialize evaluator (root is non-critical to allow partial credit across carriers)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent verification per carrier
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

    # Extract carriers from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_carriers(),
        template_class=CarriersExtraction,
        extraction_name="carriers_extraction",
    )

    # Select up to the first 3 carriers; pad with empty placeholders if fewer
    carriers: List[CarrierInfo] = list(extracted.carriers[:3])
    while len(carriers) < 3:
        carriers.append(CarrierInfo())

    # Build verification subtree for each carrier
    for idx in range(3):
        await verify_single_carrier(evaluator, root, carriers[idx], idx)

    # Return structured summary
    return evaluator.get_summary()