import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "wscuc_complete_mailing_address"
TASK_DESCRIPTION = (
    "What is the complete mailing address of the WASC Senior College and University Commission (WSCUC)? "
    "Please provide the full address including street, suite number, city, state, and ZIP code, along with a "
    "reference URL from WSCUC's official website."
)
OFFICIAL_CONTACT_URL = "https://www.wscuc.org/contact/"

EXPECTED_ADDRESS = {
    "street_address": "1080 Marina Village Parkway",
    "suite": "Suite 500",
    "city": "Alameda",
    "state": "CA",
    "zip_code": "94501",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AddressExtraction(BaseModel):
    street_address: Optional[str] = None
    suite: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_address() -> str:
    return """
    Extract the mailing address for the WASC Senior College and University Commission (WSCUC) as presented in the answer.

    Return a JSON object with the following fields:
    - street_address: The street portion only (e.g., "1080 Marina Village Parkway"), without suite/city/state/zip.
    - suite: The suite portion (e.g., "Suite 500"); if not provided, return null.
    - city: The city name (e.g., "Alameda"); if not provided, return null.
    - state: The state abbreviation (e.g., "CA"); if not provided, return null.
    - zip_code: The ZIP code (e.g., "94501"); if not provided, return null.
    - source_urls: An array of all URLs explicitly cited in the answer as references.

    IMPORTANT:
    - Extract exactly what is written in the answer. Do not infer or add missing components.
    - For URLs, include full URLs that appear in the answer text (plain or markdown links). Do not invent URLs.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_official_contact_url(url: str) -> bool:
    """
    Check if the given URL points to WSCUC's official contact page.
    Accepts http/https, with/without 'www', with optional trailing slash or query/fragment.
    """
    if not url:
        return False
    try:
        parsed = urlparse(url.strip())
        if not parsed.scheme or not parsed.netloc:
            return False
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        if host != "wscuc.org":
            return False
        # Normalize path: accept '/contact' or '/contact/' and anything that starts with '/contact'
        path = (parsed.path or "").rstrip("/")
        return path == "/contact"
    except Exception:
        return False


def _has_official_contact_url(urls: List[str]) -> bool:
    return any(_is_official_contact_url(u) for u in urls or [])


async def _verify_component_with_contact_page(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    node_desc: str,
    component_label: str,
    component_value: Optional[str],
) -> None:
    """
    Create a critical leaf node under parent_node and verify the component_value against
    the official WSCUC contact page.
    """
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=True,
    )

    # Build the claim using the extracted value from the answer
    # This ensures we judge the answer’s stated component against the official page.
    val = component_value if component_value is not None else ""
    claim = f"WSCUC's official mailing address lists the {component_label} as '{val}'."

    add_ins = (
        "Verify this exact address component against the official WSCUC contact page. "
        "Allow minor formatting variations (e.g., 'Pkwy' vs 'Parkway', presence/absence of commas), "
        "but the semantic content must match. Only consider the specified component; "
        "ignore other parts of the address for this check."
    )

    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=OFFICIAL_CONTACT_URL,
        additional_instruction=add_ins,
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_address(),
        template_class=AddressExtraction,
        extraction_name="address_extraction",
    )

    # Record ground truth info (for reference in summary)
    evaluator.add_ground_truth(
        {
            "expected_street_address": EXPECTED_ADDRESS["street_address"],
            "expected_suite": EXPECTED_ADDRESS["suite"],
            "expected_city": EXPECTED_ADDRESS["city"],
            "expected_state": EXPECTED_ADDRESS["state"],
            "expected_zip_code": EXPECTED_ADDRESS["zip_code"],
            "official_contact_url": OFFICIAL_CONTACT_URL,
        },
        gt_type="expected_address",
    )

    # Build verification tree according to rubric
    # Root critical node (as per rubric)
    main_node = evaluator.add_parallel(
        id="WSCUC_Complete_Mailing_Address",
        desc="Provides the complete mailing address for WASC Senior College and University Commission (WSCUC) and cites an official WSCUC source URL.",
        parent=root,
        critical=True,
    )

    # Mailing_Address_Correctness (critical, parallel)
    mailing_correct_node = evaluator.add_parallel(
        id="Mailing_Address_Correctness",
        desc="All required address fields match the official WSCUC mailing address.",
        parent=main_node,
        critical=True,
    )

    # Street
    await _verify_component_with_contact_page(
        evaluator=evaluator,
        parent_node=mailing_correct_node,
        node_id="Street_Address_Correct",
        node_desc="Street address is 1080 Marina Village Parkway.",
        component_label="street address",
        component_value=extracted.street_address,
    )

    # Suite
    await _verify_component_with_contact_page(
        evaluator=evaluator,
        parent_node=mailing_correct_node,
        node_id="Suite_Number_Correct",
        node_desc="Suite number is Suite 500.",
        component_label="suite",
        component_value=extracted.suite,
    )

    # City
    await _verify_component_with_contact_page(
        evaluator=evaluator,
        parent_node=mailing_correct_node,
        node_id="City_Correct",
        node_desc="City is Alameda.",
        component_label="city",
        component_value=extracted.city,
    )

    # State
    await _verify_component_with_contact_page(
        evaluator=evaluator,
        parent_node=mailing_correct_node,
        node_id="State_Correct",
        node_desc="State is CA.",
        component_label="state",
        component_value=extracted.state,
    )

    # ZIP
    await _verify_component_with_contact_page(
        evaluator=evaluator,
        parent_node=mailing_correct_node,
        node_id="ZIP_Code_Correct",
        node_desc="ZIP code is 94501.",
        component_label="ZIP code",
        component_value=extracted.zip_code,
    )

    # Official_Source_Reference (critical)
    has_contact_url = _has_official_contact_url(extracted.source_urls)
    evaluator.add_custom_node(
        result=has_contact_url,
        id="Official_Source_Reference",
        desc="Provides a reference URL from WSCUC’s official website, specifically https://www.wscuc.org/contact/.",
        parent=main_node,
        critical=True,
    )

    # Provide extra diagnostic info
    evaluator.add_custom_info(
        info={
            "extracted_source_urls": extracted.source_urls,
            "official_contact_url_detected_in_answer": has_contact_url,
        },
        info_type="diagnostics",
        info_name="source_url_diagnostics",
    )

    # Return result summary
    return evaluator.get_summary()