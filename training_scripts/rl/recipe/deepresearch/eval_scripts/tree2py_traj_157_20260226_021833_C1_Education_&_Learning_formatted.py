import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tamu_main_address"
TASK_DESCRIPTION = "What is the physical address of Texas A&M University's main campus?"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AddressExtraction(BaseModel):
    institution_name: Optional[str] = None
    campus_scope: Optional[str] = None  # e.g., "main campus", "College Station campus"
    street_address: Optional[str] = None  # e.g., "400 Bizzell St"
    city: Optional[str] = None           # e.g., "College Station"
    state: Optional[str] = None          # e.g., "TX" or "Texas"
    zip_code: Optional[str] = None       # e.g., "77843" or "77843-XXXX"
    full_address: Optional[str] = None   # If a single-line address is given by the answer
    sources: List[str] = Field(default_factory=list)  # URLs explicitly cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_address() -> str:
    return """
    Extract the physical address details provided in the answer for Texas A&M University's main campus.

    Return a JSON object with the following fields:
    - institution_name: The name of the institution that the address is for (e.g., "Texas A&M University"). If the answer does not explicitly name it, return null.
    - campus_scope: The campus designation mentioned in the answer (e.g., "main campus", "College Station campus"). If not explicitly stated, return null.
    - street_address: The street number and street name (e.g., "400 Bizzell St"). If not provided, return null.
    - city: The city component (ideally "College Station"). If not provided, return null.
    - state: The state component (e.g., "Texas" or "TX"). If not provided, return null.
    - zip_code: The ZIP or ZIP+4 (e.g., "77843" or "77843-XXXX"). If not provided, return null.
    - full_address: If the answer presents a single-line address string, include it here exactly as written. If not, return null.
    - sources: An array of all URLs explicitly mentioned in the answer that are used as sources or references for this address. Extract actual URLs only; do not invent or infer any.

    If any field is missing in the answer, set it to null (or an empty array for 'sources').
    When possible, if the full address is provided, try to also parse out the components (street_address, city, state, zip_code). If unsure, leave those components as null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _compose_address_string(ex: AddressExtraction) -> Optional[str]:
    """
    Prefer the full_address if present; otherwise compose from components.
    """
    if ex.full_address and ex.full_address.strip():
        return ex.full_address.strip()

    parts: List[str] = []
    if ex.street_address and ex.street_address.strip():
        parts.append(ex.street_address.strip())

    city_state_zip: List[str] = []
    if ex.city and ex.city.strip():
        city_state_zip.append(ex.city.strip())
    if ex.state and ex.state.strip():
        # Join state and zip with space, but only if zip present
        if ex.zip_code and ex.zip_code.strip():
            city_state_zip.append(f"{ex.state.strip()} {ex.zip_code.strip()}")
        else:
            city_state_zip.append(ex.state.strip())
    else:
        if ex.zip_code and ex.zip_code.strip():
            # No state but has zip, still include zip
            city_state_zip.append(ex.zip_code.strip())

    if city_state_zip:
        parts.append(", ".join([city_state_zip[0]] + city_state_zip[1:]))

    if not parts:
        return None
    return ", ".join(parts)


def _build_reference_urls(extracted_sources: List[str]) -> List[str]:
    """
    Construct the set of URLs used for verification.
    Priority:
    1) URLs cited by the answer (if any).
    2) A small set of authoritative fallback pages likely to mention TAMU's address.
    """
    seen = set()
    urls: List[str] = []

    # 1) Include answer-cited sources first
    for u in extracted_sources or []:
        if isinstance(u, str) and u.strip() and u.strip() not in seen:
            seen.add(u.strip())
            urls.append(u.strip())

    # 2) Fallback authoritative listings (do not claim these were provided by the answer)
    fallback_urls = [
        "https://www.usnews.com/best-colleges/texas-am-3570",
        "https://www.collegedata.com/college/texas-a-and-m-university-college-station",
        "https://www.mapquest.com/us/tx/college-station/texas-am-university-8959749",
        "https://en.wikipedia.org/wiki/Texas_A%26M_University",
        "https://www.tamu.edu/",
    ]
    for u in fallback_urls:
        if u not in seen:
            seen.add(u)
            urls.append(u)

    return urls


def _has_street_like(s: Optional[str]) -> bool:
    """
    A lightweight check indicating the presence of a plausible street address.
    Requires at least one digit and a word (e.g., "St", "Street", "Rd", "Drive", etc.) or any non-empty with digits.
    """
    if not s or not s.strip():
        return False
    if not re.search(r"\d", s):
        return False
    # Very permissive: number present is often sufficient; the LLM verification will do the heavy work later.
    return True


def _has_zip_like(s: Optional[str]) -> bool:
    """
    Basic US ZIP/ZIP+4 pattern check.
    """
    if not s or not s.strip():
        return False
    return bool(re.search(r"\b\d{5}(-\d{4})?\b", s.strip()))


# --------------------------------------------------------------------------- #
# Verification subroutine                                                     #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: AddressExtraction) -> None:
    """
    Build the verification tree based on the rubric and run all checks.
    """
    # Create a critical top-level node (parallel aggregation) under the evaluator root
    tam_root = evaluator.add_parallel(
        id="Texas_A&M_Main_Campus_Physical_Address",
        desc="Evaluate whether the response provides a complete, correct, verifiable physical address for Texas A&M University's main campus.",
        parent=evaluator.root,
        critical=True,
    )

    # Prepare data
    address_str = _compose_address_string(extracted) or ""
    ref_urls = _build_reference_urls(extracted.sources)

    # Record which URLs we attempted for verification (for transparency)
    evaluator.add_custom_info(
        info={"reference_urls_used": ref_urls, "address_string_evaluated": address_str},
        info_type="reference_urls",
        info_name="verification_references"
    )

    # 1) Correct Institution
    correct_inst_node = evaluator.add_leaf(
        id="Correct_Institution",
        desc="The address is explicitly for Texas A&M University (not another institution).",
        parent=tam_root,
        critical=True,
    )
    claim_correct_inst = (
        f"The address '{address_str}' corresponds to Texas A&M University (the flagship in College Station), "
        f"not to any other institution nor to a different Texas A&M System university."
    )
    await evaluator.verify(
        claim=claim_correct_inst,
        node=correct_inst_node,
        sources=ref_urls,
        additional_instruction=(
            "Use the provided webpages to determine whether this address belongs to Texas A&M University "
            "(the flagship university). If the webpages clearly associate this address with Texas A&M University "
            "in College Station, mark as supported. Ignore minor formatting differences."
        ),
    )

    # 2) Main Campus Scope
    main_scope_node = evaluator.add_leaf(
        id="Main_Campus_Scope",
        desc="The address is specifically for Texas A&M University's main campus (not a satellite campus or another Texas A&M System institution).",
        parent=tam_root,
        critical=True,
    )
    claim_main_scope = (
        f"The address '{address_str}' is for the main campus of Texas A&M University in College Station, "
        f"not for a satellite campus such as Galveston, Corpus Christi, or others."
    )
    await evaluator.verify(
        claim=claim_main_scope,
        node=main_scope_node,
        sources=ref_urls,
        additional_instruction=(
            "Confirm from the webpages that the location is Texas A&M University in College Station "
            "(the main campus), not another university or branch campus. Minor variations like 'TX' vs 'Texas' are acceptable."
        ),
    )

    # 3) Address Components Present (critical- parallel)
    components_root = evaluator.add_parallel(
        id="Address_Components_Present",
        desc="The provided physical address includes all required components: street address, city, state, and ZIP code.",
        parent=tam_root,
        critical=True,
    )

    # 3.1) Street Address Present (custom binary check)
    street_present_node = evaluator.add_custom_node(
        result=_has_street_like(extracted.street_address),
        id="Street_Address_Present",
        desc="Includes a street address (e.g., street number + street name) suitable as a physical location.",
        parent=components_root,
        critical=True,
    )

    # 3.2) City is College Station (simple verify from the answer content)
    city_leaf = evaluator.add_leaf(
        id="City_Is_College_Station",
        desc="City is College Station.",
        parent=components_root,
        critical=True,
    )
    provided_city = (extracted.city or "").strip()
    claim_city = (
        f"The city component of the provided address is 'College Station' (provided: '{provided_city}'). "
        f"Case-insensitive comparison is acceptable."
    )
    await evaluator.verify(
        claim=claim_city,
        node=city_leaf,
        additional_instruction=(
            "Judge using the answer text: treat 'College Station' as correct even if letter casing differs "
            "or minor punctuation variations exist. If the provided city is blank or a different city, mark incorrect."
        ),
    )

    # 3.3) State is Texas or TX (simple verify from the answer content)
    state_leaf = evaluator.add_leaf(
        id="State_Is_Texas",
        desc="State is Texas (or TX).",
        parent=components_root,
        critical=True,
    )
    provided_state = (extracted.state or "").strip()
    claim_state = (
        f"The state component of the provided address is 'Texas' or 'TX' (provided: '{provided_state}'). "
        f"Case-insensitive comparison and standard abbreviation expansion are acceptable."
    )
    await evaluator.verify(
        claim=claim_state,
        node=state_leaf,
        additional_instruction=(
            "Judge using the answer text: 'TX' and 'Texas' are equivalent. If state is blank or not Texas/TX, mark incorrect."
        ),
    )

    # 3.4) ZIP Code Present (custom binary check)
    zip_present_node = evaluator.add_custom_node(
        result=_has_zip_like(extracted.zip_code),
        id="ZIP_Code_Present",
        desc="Includes a ZIP code.",
        parent=components_root,
        critical=True,
    )

    # 4) Address Is Verifiable (critical leaf)
    verifiable_leaf = evaluator.add_leaf(
        id="Address_Is_Verifiable",
        desc="The provided address can be verified against an official Texas A&M University source or an authoritative directory listing (i.e., it matches such a listing).",
        parent=tam_root,
        critical=True,
    )
    claim_verifiable = (
        f"The physical address for Texas A&M University's main campus is '{address_str}'. "
        f"At least one of the provided webpages lists this same address (allowing minor formatting differences)."
    )
    await evaluator.verify(
        claim=claim_verifiable,
        node=verifiable_leaf,
        sources=ref_urls,
        additional_instruction=(
            "Verify that at least one webpage explicitly lists the same physical address for Texas A&M University "
            "in College Station, including street number/name, city, state, and ZIP. Minor variations like 'TX' vs 'Texas' "
            "and ZIP vs ZIP+4 are acceptable. If the webpages only indicate the city/state without street or zip, "
            "and the claim requires full address, consider it insufficient."
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
    Evaluate an answer for the task: physical address of Texas A&M University's main campus.
    """
    # Initialize evaluator with a parallel root (we add our critical node under it)
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
    extracted_address = await evaluator.extract(
        prompt=prompt_extract_address(),
        template_class=AddressExtraction,
        extraction_name="address_extraction",
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, extracted_address)

    # Return summary
    return evaluator.get_summary()