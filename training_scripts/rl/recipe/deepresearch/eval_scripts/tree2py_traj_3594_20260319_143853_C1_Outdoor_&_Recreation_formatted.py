import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "resident_annual_pass_2026"
TASK_DESCRIPTION = """
I am a U.S. resident planning to visit multiple national parks in 2026. What is the cost of the America the Beautiful Resident Annual Pass for 2026, and what are the two primary websites where I can purchase this pass online? For each website, specify whether it offers a physical pass or a digital pass.
"""

EXPECTED_PRICE = "$80"
EXPECTED_EFFECTIVE_DATE_NOTE = "effective January 1, 2026"
EXPECTED_SITES = {
    "usgs": {"domain": "store.usgs.gov", "expected_format": "physical"},
    "recreation": {"domain": "recreation.gov", "expected_format": "digital"},
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class WebsiteInfo(BaseModel):
    site_name: Optional[str] = None
    url: Optional[str] = None
    pass_format: Optional[str] = None  # normalized to "physical" or "digital"


class WebsitesExtraction(BaseModel):
    websites: List[WebsiteInfo] = Field(default_factory=list)


class PassCoreExtraction(BaseModel):
    pass_type: Optional[str] = None
    price: Optional[str] = None
    stated_effective_date: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_core_details() -> str:
    return """
    From the answer text, extract the core details about the requested pass:
    - pass_type: The exact pass name or identification as written in the answer. This should correspond to the "America the Beautiful" Annual/Resident Annual Pass (NOT a Senior, Military, Access, or 4th Grade pass). Return the exact wording used in the answer.
    - price: The stated price for the 2026 Resident/Annual pass as written (e.g., "$80" or "80 dollars"). If multiple prices are mentioned, pick the one that clearly refers to the 2026 resident/standard annual pass.
    - stated_effective_date: If the answer explicitly mentions when the 2026 pricing is effective (e.g., "effective January 1, 2026"), extract that phrase verbatim. Otherwise return null.

    Return null for any field that is not explicitly stated in the answer.
    """


def prompt_extract_websites() -> str:
    return """
    Extract all online purchase websites (as mentioned by the answer) where the America the Beautiful Resident/Annual Pass can be purchased.
    For each website, return:
    - site_name: Human-readable name of the store/website as quoted or implied by the answer (e.g., "USGS Online Store", "Recreation.gov").
    - url: The specific URL provided in the answer for purchasing the pass. If multiple are provided, pick the one most clearly related to buying this pass. If missing protocol, prepend http://
    - pass_format: Normalize to exactly one of: "physical" or "digital".
        • Use "physical" if the answer states or implies that a physical card is mailed/shipped.
        • Use "digital" if the answer states or implies a digital/electronic pass (e.g., delivered instantly, used in app, no physical card shipped).

    Only include websites that the answer explicitly presents as places to BUY the pass online.
    If the answer lists more than two, extract them all.
    If no purchase website is given, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_domain(u: Optional[str]) -> str:
    if not u:
        return ""
    try:
        parsed = urlparse(u if u.startswith("http") else f"http://{u}")
        return parsed.netloc.lower()
    except Exception:
        return ""


def _find_site_by_domain(websites: List[WebsiteInfo], domain_keyword: str) -> Optional[WebsiteInfo]:
    for w in websites:
        dn = _normalize_domain(w.url)
        text = f"{(w.site_name or '').lower()} {dn}"
        if domain_keyword in text:
            return w
    return None


def _collect_urls(*sites: Optional[WebsiteInfo]) -> List[str]:
    urls: List[str] = []
    for s in sites:
        if s and s.url:
            u = s.url if s.url.startswith("http") else f"http://{s.url}"
            urls.append(u)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_online_websites_checks(
    evaluator: Evaluator,
    parent_node,
    websites_extraction: WebsitesExtraction,
) -> None:
    online_node = evaluator.add_parallel(
        id="Online_Purchase_Websites",
        desc="Answer provides exactly two online purchase websites, and they are the two specified sites with correct physical/digital format per site.",
        parent=parent_node,
        critical=True,
    )

    # 1) Exactly two distinct websites
    # Distinctness: require exactly 2 entries and URLs (or names) are distinct.
    ws = websites_extraction.websites or []
    distinct_pairs = set()
    for w in ws:
        key = (_normalize_domain(w.url), (w.site_name or "").strip().lower())
        distinct_pairs.add(key)
    exactly_two = (len(ws) == 2) and (len(distinct_pairs) == 2)
    evaluator.add_custom_node(
        result=exactly_two,
        id="Lists_Exactly_Two_Websites",
        desc="Answer lists exactly two (no more, no fewer) distinct online purchase websites.",
        parent=online_node,
        critical=True,
    )

    # Identify expected sites in extracted list
    usgs = _find_site_by_domain(ws, "usgs.gov")
    recgov = _find_site_by_domain(ws, "recreation.gov")

    # 2) USGS Online Store sub-checks
    usgs_group = evaluator.add_parallel(
        id="USGS_Online_Store",
        desc="Includes the USGS Online Store (store.usgs.gov) as a purchase website and correctly states it offers a physical pass mailed to the purchaser.",
        parent=online_node,
        critical=True,
    )

    # 2.a) Included in answer (existence)
    evaluator.add_custom_node(
        result=usgs is not None,
        id="USGS_included_in_answer",
        desc="The answer includes the USGS Online Store (store.usgs.gov) as one of the purchase websites.",
        parent=usgs_group,
        critical=True,
    )

    # 2.b) Format stated correctly in answer (physical)
    evaluator.add_custom_node(
        result=(usgs is not None and (usgs.pass_format or "").lower() == "physical"),
        id="USGS_format_in_answer_correct",
        desc="For USGS, the answer correctly states the pass format as physical (mailed/shipped).",
        parent=usgs_group,
        critical=True,
    )

    # 2.c) Format supported by the cited USGS page
    usgs_format_node = evaluator.add_leaf(
        id="USGS_format_supported_by_site",
        desc="USGS Online Store sells a physical pass that is mailed/shipped (not a digital pass).",
        parent=usgs_group,
        critical=True,
    )
    usgs_claim = (
        "The USGS Online Store's America the Beautiful Annual Pass offering is a physical pass that is mailed or shipped to the purchaser (i.e., not a digital pass)."
    )
    await evaluator.verify(
        claim=usgs_claim,
        node=usgs_format_node,
        sources=usgs.url if usgs and usgs.url else None,
        additional_instruction="On the product page, look for terms like 'ships', 'mailed', 'physical card', or 'plastic card'. If the page clearly indicates mailing/shipping of a physical pass, the claim is supported.",
    )

    # 3) Recreation.gov sub-checks
    rec_group = evaluator.add_parallel(
        id="Recreation_Gov",
        desc="Includes Recreation.gov as a purchase website and correctly states it offers a digital pass.",
        parent=online_node,
        critical=True,
    )

    # 3.a) Included in answer (existence)
    evaluator.add_custom_node(
        result=recgov is not None,
        id="RecreationGov_included_in_answer",
        desc="The answer includes Recreation.gov as one of the purchase websites.",
        parent=rec_group,
        critical=True,
    )

    # 3.b) Format stated correctly in answer (digital)
    evaluator.add_custom_node(
        result=(recgov is not None and (recgov.pass_format or "").lower() == "digital"),
        id="RecreationGov_format_in_answer_correct",
        desc="For Recreation.gov, the answer correctly states the pass format as digital.",
        parent=rec_group,
        critical=True,
    )

    # 3.c) Format supported by the cited Recreation.gov page
    rec_format_node = evaluator.add_leaf(
        id="RecreationGov_format_supported_by_site",
        desc="Recreation.gov sells a digital pass (delivered electronically; no physical card is mailed).",
        parent=rec_group,
        critical=True,
    )
    rec_claim = (
        "Recreation.gov's America the Beautiful Annual Pass offering is a digital pass (delivered electronically for immediate use), and no physical card is mailed."
    )
    await evaluator.verify(
        claim=rec_claim,
        node=rec_format_node,
        sources=recgov.url if recgov and recgov.url else None,
        additional_instruction="On the Recreation.gov pass page, look for indicators like 'digital pass', 'delivered electronically', 'immediate use', or 'use in app', and the absence of shipping a physical card.",
    )


async def build_pass_cost_checks(
    evaluator: Evaluator,
    parent_node,
    core: PassCoreExtraction,
    websites: WebsitesExtraction,
) -> None:
    # Create a critical parallel group for cost checks (answer statement + site support)
    cost_group = evaluator.add_parallel(
        id="Pass_Cost",
        desc="Answer states the cost is $80 for the 2026 Resident Annual Pass.",
        parent=parent_node,
        critical=True,
    )

    # 1) Answer explicitly states $80 for the 2026 Resident Annual Pass
    cost_stated_node = evaluator.add_leaf(
        id="Pass_Cost_stated_in_answer",
        desc="The answer explicitly states the cost is $80 for the 2026 Resident Annual Pass.",
        parent=cost_group,
        critical=True,
    )
    stated_price = core.price or ""
    claim_price_in_answer = (
        f"The answer states that the 2026 price for the America the Beautiful Resident/Annual Pass is {EXPECTED_PRICE}."
    )
    await evaluator.verify(
        claim=claim_price_in_answer,
        node=cost_stated_node,
        additional_instruction="Check the answer text only (not external pages). Minor formatting variations like '$80.00' should count as $80.",
    )

    # 2) Price is supported by the purchase pages (use available URLs from extracted sites)
    usgs = _find_site_by_domain(websites.websites, "usgs.gov")
    recgov = _find_site_by_domain(websites.websites, "recreation.gov")
    price_supported_node = evaluator.add_leaf(
        id="Pass_Cost_supported_by_sites",
        desc="The price of the America the Beautiful (Resident/Annual) Pass is $80, supported by the cited purchase pages.",
        parent=cost_group,
        critical=True,
    )
    price_claim = "The America the Beautiful Annual Pass (standard resident/annual pass) costs $80."
    await evaluator.verify(
        claim=price_claim,
        node=price_supported_node,
        sources=_collect_urls(usgs, recgov),
        additional_instruction="Verify on the cited purchase page(s) that the price shown for the standard America the Beautiful Annual Pass is $80 (allow '$80.00'). Ignore shipping/processing fees.",
    )


async def build_pass_type_check(evaluator: Evaluator, parent_node) -> None:
    # Single critical leaf verifying the answer identifies the correct pass
    pass_type_node = evaluator.add_leaf(
        id="Pass_Type",
        desc="Answer identifies the pass as the America the Beautiful Resident Annual Pass for U.S. citizens or permanent residents (not another pass type).",
        parent=parent_node,
        critical=True,
    )
    pass_type_claim = (
        "The answer identifies the pass as the 'America the Beautiful' Resident Annual Pass (i.e., the standard Annual Pass) intended for U.S. citizens or permanent residents, "
        "and does not refer instead to a different pass type such as Senior, Military, Access, 4th Grade, or Volunteer."
    )
    await evaluator.verify(
        claim=pass_type_claim,
        node=pass_type_node,
        additional_instruction="Check only the answer text. Allow reasonable naming variants like 'America the Beautiful Annual Pass' or 'standard Annual Pass'."
    )


async def build_effective_date_check(evaluator: Evaluator, parent_node) -> None:
    # Single critical leaf verifying the answer notes Jan 1, 2026 effective date
    eff_date_node = evaluator.add_leaf(
        id="Pass_Effective_Date",
        desc="Answer notes the 2026 Resident Annual Pass pricing is effective January 1, 2026.",
        parent=parent_node,
        critical=True,
    )
    eff_date_claim = "The answer explicitly mentions that the 2026 pricing is effective January 1, 2026."
    await evaluator.verify(
        claim=eff_date_claim,
        node=eff_date_node,
        additional_instruction="Check the answer text only (not external pages). Accept reasonable phrasings like 'effective Jan 1, 2026' or 'starting January 1, 2026'."
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

    # Extract structured details from the answer
    core_details, websites_details = await asyncio.gather(
        evaluator.extract(
            prompt=prompt_extract_core_details(),
            template_class=PassCoreExtraction,
            extraction_name="pass_core_details",
        ),
        evaluator.extract(
            prompt=prompt_extract_websites(),
            template_class=WebsitesExtraction,
            extraction_name="purchase_websites",
        ),
    )

    # Add ground truth/reference expectations (for transparency; not used to auto-pass)
    evaluator.add_ground_truth({
        "expected_price": EXPECTED_PRICE,
        "expected_effective_date_phrase": EXPECTED_EFFECTIVE_DATE_NOTE,
        "expected_websites": {
            "USGS Online Store": {"domain": EXPECTED_SITES["usgs"]["domain"], "format": EXPECTED_SITES["usgs"]["expected_format"]},
            "Recreation.gov": {"domain": EXPECTED_SITES["recreation"]["domain"], "format": EXPECTED_SITES["recreation"]["expected_format"]},
        }
    })

    # Build top-level critical evaluation group
    top = evaluator.add_parallel(
        id="Resident_Annual_Pass_2026",
        desc="Evaluate whether the answer provides the correct 2026 cost and the two specified online purchase websites for the America the Beautiful Resident Annual Pass, including the correct pass format per website.",
        parent=root,
        critical=True,
    )

    # Build child checks (all critical under the critical parent)
    await build_pass_type_check(evaluator, top)
    await build_pass_cost_checks(evaluator, top, core_details, websites_details)
    await build_effective_date_check(evaluator, top)
    await build_online_websites_checks(evaluator, top, websites_details)

    # Return structured summary
    return evaluator.get_summary()