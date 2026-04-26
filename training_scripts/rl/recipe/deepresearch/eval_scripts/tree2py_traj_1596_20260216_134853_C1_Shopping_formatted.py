import asyncio
import logging
import re
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse, parse_qs

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "lego_enterprise_purchase_info"
TASK_DESCRIPTION = """
What is the current price in US dollars and the household purchase limit for the LEGO Star Trek U.S.S. Enterprise NCC-1701-D (set 10356) when ordering from the official LEGO website?
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PurchaseInfoExtraction(BaseModel):
    product_name: Optional[str] = None
    set_number: Optional[str] = None
    price_usd: Optional[str] = None
    availability_status: Optional[str] = None
    purchase_limit: Optional[str] = None
    as_of_phrase: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_purchase_info() -> str:
    return """
    Extract the following fields exactly as stated in the answer text. Do not infer or invent any values.
    - product_name: The product name as written (e.g., "LEGO Icons Star Trek U.S.S. Enterprise NCC-1701-D").
    - set_number: The LEGO set number mentioned (digits only if possible, e.g., "10356"). If not clearly given, return null.
    - price_usd: The current price stated in USD, including symbol if present (e.g., "$499.99", "USD 499.99"). If no price in USD is stated, return null.
    - availability_status: The availability status as stated (e.g., "In stock", "Backorder", "Out of stock", "Pre-order", "Coming soon"). If not provided, return null.
    - purchase_limit: The household purchase limit text (e.g., "Limit 2 per household", "Max 3 per customer"). If not provided, return null.
    - as_of_phrase: Any "as-of" or "checked on" date/time statement in the answer (e.g., "as of Feb 2026", "checked February 2026", "updated 2026-02"). If not present, return null.
    - source_urls: All URLs present in the answer (including markdown link targets). Extract only valid, complete URLs. If none are present, return an empty array.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_lego_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        return "lego.com" in host
    except Exception:
        return False


def prefer_us_lego_urls(urls: List[str]) -> List[str]:
    """Prefer LEGO URLs that appear to be US/English. If none match, return original list."""
    if not urls:
        return urls
    us_like = []
    for u in urls:
        try:
            p = urlparse(u)
            path_lower = (p.path or "").lower()
            query = parse_qs(p.query or "")
            query_values = " ".join([str(v).lower() for v in query.values()])
            if ("en-us" in path_lower
                or "country=us" in (p.query or "").lower()
                or "locale=en-us" in (p.query or "").lower()
                or "lang=en-us" in (p.query or "").lower()
                or "region=us" in (p.query or "").lower()
                or "us" in query_values):
                us_like.append(u)
        except Exception:
            continue
    return us_like if us_like else urls


def parse_first_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r'\d+', text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def indicates_february_2026(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    if "2026" not in t:
        return False
    # Direct month name
    if "february" in t or re.search(r'\bfeb\b', t):
        return True
    # Numeric month-year patterns like 02/2026, 02-2026, 2026-02, 2026/02, 2/2026, 2026/2
    if re.search(r'(?:^|\b)(0?2)[/\-.\s]?2026\b', t):
        return True
    if re.search(r'\b2026[/\-.\s]?(0?2)(?:\b|$)', t):
        return True
    return False


def availability_looks_valid(status: Optional[str]) -> bool:
    if not status:
        return False
    s = status.strip().lower()
    valid_tokens = [
        "in stock",
        "out of stock",
        "temporarily out of stock",
        "backorder",
        "backordered",
        "pre-order",
        "preorder",
        "coming soon",
        "available now",
        "available",
        "sold out",
        "back in stock",
        "ships in",
        "ships by",
        "pre order"
    ]
    return any(tok in s for tok in valid_tokens)


# --------------------------------------------------------------------------- #
# Verification building                                                       #
# --------------------------------------------------------------------------- #
async def verify_purchase_info(
    evaluator: Evaluator,
    root_node,
    extracted: PurchaseInfoExtraction
) -> None:
    """
    Build the verification tree according to the rubric.
    """
    # Derive LEGO URLs from provided sources
    lego_urls = [u for u in (extracted.source_urls or []) if is_lego_url(u)]
    lego_us_urls = prefer_us_lego_urls(lego_urls)

    # Add some custom info for debugging
    evaluator.add_custom_info(
        {
            "all_source_urls": extracted.source_urls,
            "lego_urls": lego_urls,
            "lego_us_urls": lego_us_urls
        },
        info_type="url_analysis",
        info_name="url_analysis"
    )

    # Main critical parallel node
    main = evaluator.add_parallel(
        id="LEGO_Enterprise_Purchase_Information",
        desc="Answer provides the requested purchase information for the specified LEGO set, consistent with the stated constraints.",
        parent=root_node,
        critical=True
    )

    # 1) Correct_Product_Identified (critical leaf)
    node_correct_product = evaluator.add_leaf(
        id="Correct_Product_Identified",
        desc="Answer identifies the product as LEGO Icons Star Trek U.S.S. Enterprise NCC-1701-D with set number 10356.",
        parent=main,
        critical=True,
    )
    # Prefer URL-based verification if official LEGO URL is present; otherwise simple verify against the answer
    if lego_us_urls or lego_urls:
        product_claim = (
            "This webpage is for the LEGO Icons Star Trek U.S.S. Enterprise NCC-1701-D "
            "(set 10356). Treat minor naming/title variations as acceptable if it's clearly the same set; "
            "the set number must be 10356."
        )
        await evaluator.verify(
            claim=product_claim,
            node=node_correct_product,
            sources=lego_us_urls or lego_urls,
            additional_instruction="Match by set number 10356 and the U.S.S. Enterprise NCC-1701-D name; 'LEGO Icons' may appear in category/title."
        )
    else:
        # Fallback: check directly against the answer content
        pname = extracted.product_name or ""
        snum = extracted.set_number or ""
        simple_claim = (
            f"The answer identifies the product as LEGO Icons Star Trek U.S.S. Enterprise NCC-1701-D "
            f"with set number 10356. Extracted product_name='{pname}', set_number='{snum}'."
        )
        await evaluator.verify(
            claim=simple_claim,
            node=node_correct_product,
            sources=None,
            additional_instruction="Use the provided answer content to determine if this statement is true."
        )

    # 2) Verified_From_Official_LEGO_Website (critical leaf) — check presence of lego.com link(s) in answer
    evaluator.add_custom_node(
        result=len(lego_urls) > 0,
        id="Verified_From_Official_LEGO_Website",
        desc="Answer indicates the pricing/availability/limit information was verified from the official LEGO website (LEGO.com).",
        parent=main,
        critical=True
    )

    # 3) As_Of_February_2026_Indicated (critical leaf) — check that the answer mentions Feb 2026
    evaluator.add_custom_node(
        result=indicates_february_2026(extracted.as_of_phrase),
        id="As_Of_February_2026_Indicated",
        desc="Answer indicates the information reflects an as-of date/time in February 2026 (e.g., 'checked Feb 2026').",
        parent=main,
        critical=True
    )

    # 4) Availability_Status_Provided (critical leaf) — check that availability status is present and plausible
    evaluator.add_custom_node(
        result=availability_looks_valid(extracted.availability_status),
        id="Availability_Status_Provided",
        desc="Answer states the product availability status as shown on LEGO.com at the time checked (e.g., in stock/backorder/out of stock).",
        parent=main,
        critical=True
    )

    # 5) Price_In_USD_Provided (critical leaf) — verify against LEGO.com page if available
    node_price = evaluator.add_leaf(
        id="Price_In_USD_Provided",
        desc="Answer states the current price in US dollars (USD).",
        parent=main,
        critical=True
    )
    price_text = (extracted.price_usd or "").strip()
    price_claim = (
        f"The price shown on the LEGO.com product page for set 10356 is '{price_text}' in USD. "
        "Minor formatting differences (e.g., $499 vs $499.99) are acceptable, but the amount must match."
    )
    await evaluator.verify(
        claim=price_claim,
        node=node_price,
        sources=lego_us_urls or lego_urls or None,
        additional_instruction="Focus on the US store/regional view (USD). Verify price near the 'Add to Bag' or main price element. Treat '$' as USD."
    )

    # 6) Household_Purchase_Limit_Numeric (critical leaf) — ensure numeric quantity is provided
    limit_num = parse_first_int(extracted.purchase_limit)
    evaluator.add_custom_node(
        result=limit_num is not None,
        id="Household_Purchase_Limit_Numeric",
        desc="Answer states the household purchase limit as a specific numeric quantity of units.",
        parent=main,
        critical=True
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
    Evaluate an answer for the LEGO Star Trek U.S.S. Enterprise (set 10356) purchase information task.
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_purchase_info(),
        template_class=PurchaseInfoExtraction,
        extraction_name="purchase_info_extraction"
    )

    # Record minor diagnostics
    evaluator.add_custom_info(
        {
            "extracted_product_name": extracted.product_name,
            "extracted_set_number": extracted.set_number,
            "extracted_price_usd": extracted.price_usd,
            "extracted_availability_status": extracted.availability_status,
            "extracted_purchase_limit": extracted.purchase_limit,
            "extracted_as_of_phrase": extracted.as_of_phrase
        },
        info_type="extracted_fields",
        info_name="extracted_fields"
    )

    # Build and run verification
    await verify_purchase_info(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()