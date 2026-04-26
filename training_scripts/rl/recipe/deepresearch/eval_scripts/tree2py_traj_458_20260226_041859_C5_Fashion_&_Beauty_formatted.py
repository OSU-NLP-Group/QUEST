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
TASK_ID = "bh_salon"
TASK_DESCRIPTION = (
    "I am looking for professional hair salons in Beverly Hills, California that offer comprehensive hair services. "
    "Identify four different salons, where each salon must meet all of the following requirements:\n\n"
    "1. The salon must be located in Beverly Hills, California (zip code 90210 or 90211)\n"
    "2. The salon must offer professional hair color services\n"
    "3. The salon must specifically offer balayage technique services\n"
    "4. The salon must offer at least one type of professional hair extension service\n"
    "5. The salon must have a publicly listed phone number\n"
    "6. The salon must have an official website or verified online business listing\n\n"
    "For each of the four salons, provide:\n"
    "- The complete name of the salon\n"
    "- The complete street address\n"
    "- The phone number\n"
    "- A reference URL to the salon's official website or verified business listing (such as Yelp or Google Business Profile)\n"
    "- A brief description of the services offered, confirming that the salon provides hair color, balayage, and hair extension services"
)

ZIP_ALLOWLIST = {"90210", "90211"}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SalonEntry(BaseModel):
    """Single salon entry as extracted from the agent's answer."""
    name: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    phone: Optional[str] = None
    reference_url: Optional[str] = None
    service_description: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


class SalonsExtraction(BaseModel):
    """All salons extracted from the answer."""
    salons: List[SalonEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_salons() -> str:
    return (
        "Extract up to four distinct salon entries from the answer. Each entry must include the following fields:\n"
        "- name: The complete salon name.\n"
        "- street_address: The full street address.\n"
        "- city: The city portion of the address (expect 'Beverly Hills').\n"
        "- state: The state abbreviation (expect 'CA').\n"
        "- zip_code: The 5-digit zip code.\n"
        "- phone: A publicly listed phone number. If not provided, return null.\n"
        "- reference_url: A URL to the salon’s official website or a verified business listing (Yelp or Google Business Profile). Extract the actual URL. If missing protocol, prepend http://.\n"
        "- service_description: A brief description from the answer confirming the salon offers hair color, balayage, and at least one hair extension service. If not explicitly stated, summarize available info or return null.\n"
        "- additional_urls: Any other URLs mentioned that are relevant to this salon (e.g., services page, bookings page, separate listings). Return as an array; if none, return an empty array.\n\n"
        "Rules:\n"
        "1. Only extract information explicitly present in the answer.\n"
        "2. If the answer lists more than four salons, extract only the first four.\n"
        "3. If the answer lists fewer than four salons, extract those available.\n"
        "4. Return null for any missing field.\n"
        "5. Capture URLs in plain form or within markdown; extract the actual hyperlink target.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_nonempty(text: Optional[str]) -> bool:
    return bool(text and text.strip())


def _normalize_key(s: Optional[str]) -> str:
    if not _is_nonempty(s):
        return ""
    import re
    return re.sub(r"[^a-z0-9]+", "", s.strip().lower())


def _ordinal(n: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


def _build_sources(salon: SalonEntry) -> List[str]:
    urls = []
    if _is_nonempty(salon.reference_url):
        urls.append(salon.reference_url.strip())
    for u in salon.additional_urls or []:
        if _is_nonempty(u):
            urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for u in urls:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


def _count_named_salons(salons: List[SalonEntry]) -> int:
    return sum(1 for s in salons if _is_nonempty(s.name))


def _salons_are_distinct(salons: List[SalonEntry]) -> bool:
    """
    Checks distinctness among first up to 4 salons using normalized (name, address) pairs.
    If both name and street_address are empty, that entry is ignored for distinctness check.
    """
    pairs = []
    for s in salons[:4]:
        name_key = _normalize_key(s.name)
        addr_key = _normalize_key(s.street_address)
        if not name_key and not addr_key:
            # ignore empty entry
            continue
        pairs.append((name_key, addr_key))
    # Distinct if all pairs are unique and we have exactly 4 entries with at least a name
    return len(pairs) == 4 and len(set(pairs)) == 4


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_salon(
    evaluator: Evaluator,
    parent_node,
    salon: SalonEntry,
    index: int,
) -> None:
    """
    Build verification sub-tree for a single salon.
    """
    ord_idx = _ordinal(index + 1)

    salon_node = evaluator.add_parallel(
        id=f"Salon_{index + 1}",
        desc=f"{ord_idx} salon entry (constraints + required fields).",
        parent=parent_node,
        critical=False
    )

    # Name provided (existence)
    evaluator.add_custom_node(
        result=_is_nonempty(salon.name),
        id=f"Salon_{index + 1}_Name_Provided",
        desc="Provides the complete name of the salon.",
        parent=salon_node,
        critical=True
    )

    # Phone provided (existence)
    evaluator.add_custom_node(
        result=_is_nonempty(salon.phone),
        id=f"Salon_{index + 1}_Phone_Provided",
        desc="Provides a publicly listed phone number for the salon.",
        parent=salon_node,
        critical=True
    )

    # Reference URL provided (existence)
    has_ref = _is_nonempty(salon.reference_url)
    evaluator.add_custom_node(
        result=has_ref,
        id=f"Salon_{index + 1}_Reference_URL_Provided",
        desc="Provides a reference URL to an official website or a verified online business listing (e.g., Yelp or Google Business Profile).",
        parent=salon_node,
        critical=True
    )

    # Address in Beverly Hills, CA with zip 90210 or 90211 (verified via URL)
    addr_leaf = evaluator.add_leaf(
        id=f"Salon_{index + 1}_Address_In_Beverly_Hills_90210_90211",
        desc="Provides a complete street address in Beverly Hills, CA with zip code 90210 or 90211.",
        parent=salon_node,
        critical=True
    )
    # Craft claim: allow the verifier to look for BH and zip code on page
    if _is_nonempty(salon.street_address) and _is_nonempty(salon.zip_code):
        addr_text = f"{salon.street_address}, {salon.city or ''}, {salon.state or ''} {salon.zip_code}".strip()
    else:
        addr_text = "the salon’s address"

    addr_claim = (
        f"The page shows {addr_text} located in Beverly Hills, CA, and the zip code is either 90210 or 90211."
    )
    await evaluator.verify(
        claim=addr_claim,
        node=addr_leaf,
        sources=_build_sources(salon),
        additional_instruction=(
            "Verify from the webpage/listing that the salon is in Beverly Hills, CA and the zip code is 90210 or 90211. "
            "Accept addresses in text, footer, contact page, map widgets, or listing details. Minor formatting variations are acceptable."
        )
    )

    # Currently operating (verified via URL)
    operating_leaf = evaluator.add_leaf(
        id=f"Salon_{index + 1}_Currently_Operating",
        desc="Salon is currently operating (not closed).",
        parent=salon_node,
        critical=True
    )
    operating_claim = (
        "The salon appears to be an active, currently operating business (not permanently closed)."
    )
    await evaluator.verify(
        claim=operating_claim,
        node=operating_leaf,
        sources=_build_sources(salon),
        additional_instruction=(
            "Use evidence such as active website content (services, booking), working hours, recent posts, or listing status. "
            "If the page clearly indicates 'permanently closed' or similar, then it is not operating."
        )
    )

    # Services confirm: hair color, balayage, extensions (verified via URL)
    services_leaf = evaluator.add_leaf(
        id=f"Salon_{index + 1}_Service_Description_Confirms_Required_Services",
        desc="Provides a brief description that explicitly confirms the salon offers (i) professional hair color, (ii) balayage, and (iii) at least one type of professional hair extension service.",
        parent=salon_node,
        critical=True
    )
    services_claim = (
        "This salon offers professional hair color services, balayage technique services, and at least one type of professional hair extension service."
    )
    await evaluator.verify(
        claim=services_claim,
        node=services_leaf,
        sources=_build_sources(salon),
        additional_instruction=(
            "Confirm on the site/listing that the salon offers all three categories: "
            "• Hair color (e.g., professional color, highlights, gloss) "
            "• Balayage (e.g., balayage highlights) "
            "• Hair extensions (e.g., tape-in, hand-tied, sew-in, keratin-bond). "
            "Allow reasonable synonyms. Evidence may be on a services page, menu, description, or listing details."
        )
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
    Evaluate an answer for the Beverly Hills salons task using the Mind2Web2 evaluation framework.
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
        default_model=model
    )

    # Extract salons from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_salons(),
        template_class=SalonsExtraction,
        extraction_name="salons_extraction"
    )

    # Normalize to first four salons; do not pad here to keep global checks meaningful
    salons = extracted.salons[:4]

    # Record custom info (useful for debugging)
    evaluator.add_custom_info(
        info={"allowed_zip_codes": sorted(list(ZIP_ALLOWLIST)), "num_salons_extracted": len(extracted.salons)},
        info_type="constraints",
        info_name="zip_and_count_info"
    )

    # Build main task completion node (non-critical to satisfy framework's critical consistency rule)
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Identify four different professional hair salons in Beverly Hills, CA (90210 or 90211) that meet all specified constraints, and provide the requested fields for each salon.",
        parent=root,
        critical=False
    )

    # Global requirements (critical)
    global_node = evaluator.add_parallel(
        id="Global_Requirements",
        desc="Cross-salon requirements that apply to the overall set of results.",
        parent=task_node,
        critical=True
    )

    # Global leaf: Four salons provided (name presence used to count)
    evaluator.add_custom_node(
        result=_count_named_salons(salons) == 4,
        id="Four_Salons_Provided",
        desc="Provides four salons (four entries/items are present).",
        parent=global_node,
        critical=True
    )

    # Global leaf: Salons are distinct (by normalized name/address pair)
    evaluator.add_custom_node(
        result=_salons_are_distinct(salons),
        id="Salons_Are_Distinct",
        desc="The four salons are different businesses (not duplicates of the same salon).",
        parent=global_node,
        critical=True
    )

    # Per-salon verification
    # If fewer than 4 were extracted, create placeholder empty entries so structure remains consistent
    while len(salons) < 4:
        salons.append(SalonEntry())

    for idx, salon in enumerate(salons):
        await verify_salon(evaluator, task_node, salon, idx)

    # Return structured result summary
    return evaluator.get_summary()