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
TASK_ID = "flagship_smartphones_mar2026_qi2_mmwave_512gb"
TASK_DESCRIPTION = """
I'm looking to purchase a new flagship smartphone in March 2026 that meets specific technical requirements. Please identify at least 3 different smartphone models that are currently available for purchase online and meet ALL of the following criteria:

1. Wireless Charging: Must support Qi2 wireless charging standard at 15W or higher
2. 5G Connectivity: Must support 5G mmWave (millimeter wave) bands
3. Camera: Must have a primary/main camera with at least 50MP resolution
4. Storage: Must be available in a 512GB storage configuration
5. Retail Availability: Each phone must be available from:
   - At least one major online retailer (such as Best Buy, Amazon, or directly from the manufacturer's website)
   - At least one mobile carrier (AT&T, Verizon, or T-Mobile)

For each of the 3 smartphones you identify, please provide:
- The exact model name
- Verification that it meets the wireless charging specification (Qi2 at 15W+)
- Verification that it supports 5G mmWave
- Verification of the primary camera resolution
- At least one URL from a major online retailer where the 512GB version can be purchased
- At least one URL from a mobile carrier where it's available
- The listed price for the 512GB model from at least one retailer
- Reference URL(s) confirming the technical specifications

Please ensure all information is current as of March 2026 and that the phones are actually available for purchase (not pre-order only).
"""


# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class PhoneCandidate(BaseModel):
    model_name: Optional[str] = None

    # Technical requirement evidence
    qi2_claim: Optional[str] = None
    qi2_ref_urls: List[str] = Field(default_factory=list)

    mmwave_claim: Optional[str] = None
    mmwave_ref_urls: List[str] = Field(default_factory=list)

    primary_camera_resolution: Optional[str] = None
    camera_ref_urls: List[str] = Field(default_factory=list)

    # Purchase availability and price (512GB)
    retailer_urls_512gb: List[str] = Field(default_factory=list)
    carrier_urls: List[str] = Field(default_factory=list)

    price_512gb: Optional[str] = None
    price_source_url: Optional[str] = None  # Preferably one of retailer_urls_512gb

    # Additional spec/source pages (fallbacks)
    general_spec_urls: List[str] = Field(default_factory=list)


class CandidateExtraction(BaseModel):
    candidates: List[PhoneCandidate] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_candidates() -> str:
    return """
    Parse the answer and extract up to five (5) candidate smartphone entries, preserving exactly what the answer states.
    For each candidate, extract these fields:
    - model_name: exact model name string as written in the answer.
    - qi2_claim: any text in the answer claiming Qi2 wireless charging at 15W or higher (verbatim if present).
    - qi2_ref_urls: array of URLs cited to support Qi2 15W+ (Qi2 standard) capability.
    - mmwave_claim: any text in the answer claiming 5G mmWave support (verbatim if present).
    - mmwave_ref_urls: array of URLs cited to support 5G mmWave support.
    - primary_camera_resolution: the main/primary camera megapixel value as stated (e.g., "50MP", "200 MP").
    - camera_ref_urls: array of URLs cited to support the primary camera resolution.
    - retailer_urls_512gb: array of URLs to major online retailers or official manufacturer stores for the 512GB variant purchase page
      (e.g., Best Buy, Amazon, Walmart, official manufacturer stores like apple.com, samsung.com, store.google.com, oneplus.com, motorola.com).
    - carrier_urls: array of URLs to carrier pages (must be AT&T, Verizon, or T-Mobile) showing availability for purchase (not pre-order only).
    - price_512gb: the listed price text for the 512GB variant as stated in the answer (e.g., "$1199.99"). If multiple, choose one.
    - price_source_url: a single URL (preferably from retailer_urls_512gb) where the above price is shown.
    - general_spec_urls: any other spec/reference URLs cited that could corroborate the above technical specs.

    Rules:
    - Extract only what the answer explicitly provides; do not invent or infer.
    - All URL fields must contain actual URLs present in the answer. If none, return an empty list.
    - If any requested text field is missing from the answer, set it to null.
    - Return an object with a 'candidates' array of up to 5 candidate objects.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_model_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return " ".join(name.strip().lower().split())


def _pick_sources(preferred: List[str], fallback: List[str]) -> List[str]:
    # Return preferred if any; else fallback; ensure uniqueness and non-empty strings
    merged = preferred if preferred else fallback
    seen = set()
    cleaned: List[str] = []
    for u in merged:
        if not isinstance(u, str):
            continue
        uu = u.strip()
        if not uu:
            continue
        if uu not in seen:
            seen.add(uu)
            cleaned.append(uu)
    return cleaned


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_phone_verification_subtree(
    evaluator: Evaluator,
    parent_node,
    phone: PhoneCandidate,
    phone_idx: int,
) -> None:
    """
    Build verification nodes for one phone and run all required verifications.
    phone_idx is 0-based; human-readable numbering is idx+1.
    """
    phone_no = phone_idx + 1
    phone_node = evaluator.add_parallel(
        id=f"phone_{phone_no}",
        desc=f"Candidate Phone {phone_no} meets all per-phone constraints with verifiable evidence.",
        parent=parent_node,
        critical=False  # Non-critical at top; its critical children enforce compliance
    )

    # 1) Exact model name provided (critical)
    name_ok = isinstance(phone.model_name, str) and phone.model_name.strip() != ""
    evaluator.add_custom_node(
        result=name_ok,
        id=f"phone_{phone_no}_model_name",
        desc="Exact model name is provided.",
        parent=phone_node,
        critical=True
    )

    # 2) Technical requirements (critical group)
    tech_node = evaluator.add_parallel(
        id=f"phone_{phone_no}_technical_requirements",
        desc="Required technical specs are verified via reference URL(s).",
        parent=phone_node,
        critical=True
    )

    # 2.a) Qi2 15W+
    qi2_leaf = evaluator.add_leaf(
        id=f"phone_{phone_no}_qi2_15w",
        desc="Verifies (via reference URL[s]) that the phone supports Qi2 wireless charging at 15W or higher.",
        parent=tech_node,
        critical=True
    )
    qi2_sources = _pick_sources(phone.qi2_ref_urls, phone.general_spec_urls)
    qi2_claim = f"The smartphone model '{phone.model_name or ''}' supports Qi2 (Qi 2.0) wireless charging at 15W or higher."
    qi2_ins = (
        "You must ONLY judge using the provided URL(s). "
        "Mark Incorrect if no URL is provided. "
        "Accept only if the page explicitly references the Qi2 (Qi 2.0) standard AND a power of at least 15W for wireless charging. "
        "Do NOT accept generic 'Qi' or ambiguous wording that does not say 'Qi2' or 'Qi 2.0'. "
        "Accept manufacturer official store/spec pages or reputable spec pages if they clearly state Qi2 ≥ 15W."
    )

    # 2.b) 5G mmWave
    mmw_leaf = evaluator.add_leaf(
        id=f"phone_{phone_no}_mmwave",
        desc="Verifies (via reference URL[s]) that the phone supports 5G mmWave.",
        parent=tech_node,
        critical=True
    )
    mmw_sources = _pick_sources(phone.mmwave_ref_urls, phone.general_spec_urls)
    mmw_claim = f"The smartphone model '{phone.model_name or ''}' supports 5G mmWave (millimeter wave) bands."
    mmw_ins = (
        "You must ONLY judge using the provided URL(s). "
        "Mark Incorrect if no URL is provided. "
        "Accept if the page clearly mentions 'mmWave', '5G UW' (Verizon Ultra Wideband denoting mmWave), or explicit mmWave band support. "
        "Do NOT accept generic '5G' without mmWave."
    )

    # 2.c) Primary camera ≥ 50MP
    cam_leaf = evaluator.add_leaf(
        id=f"phone_{phone_no}_primary_camera_50mp",
        desc="Verifies (via reference URL[s]) that the primary/main camera is at least 50MP.",
        parent=tech_node,
        critical=True
    )
    cam_sources = _pick_sources(phone.camera_ref_urls, phone.general_spec_urls)
    cam_claim = (
        f"The primary (main) rear camera of '{phone.model_name or ''}' has a resolution of at least 50 megapixels "
        f"(stated in the page as 50MP, 64MP, 108MP, 200MP, etc.)."
    )
    cam_ins = (
        "You must ONLY judge using the provided URL(s). "
        "Mark Incorrect if no URL is provided. "
        "Confirm that the page explicitly lists a main/primary camera ≥ 50MP. "
        "If multiple cameras are listed, ensure the main/primary meets the threshold."
    )

    # 3) Major retailer 512GB purchase URL (critical)
    retailer_leaf = evaluator.add_leaf(
        id=f"phone_{phone_no}_major_retailer_512gb_purchase_url",
        desc="Provides ≥1 URL from a major online retailer (e.g., Best Buy, Amazon, or manufacturer direct) for the 512GB variant, and the listing indicates it is available for purchase (not pre-order only).",
        parent=phone_node,
        critical=True
    )
    retailer_sources = list({u.strip() for u in (phone.retailer_urls_512gb or []) if isinstance(u, str) and u.strip()})
    retailer_claim = (
        f"At least one provided retailer URL shows the '{phone.model_name or ''}' 512GB variant available for purchase now "
        f"(not pre-order only)."
    )
    retailer_ins = (
        "Judge only using the provided retailer URL(s). Mark Incorrect if none are provided. "
        "The URL must be a major retailer (e.g., Best Buy, Amazon) or an official manufacturer store (e.g., apple.com, samsung.com, store.google.com, oneplus.com, motorola.com). "
        "The page must indicate the 512GB capacity (e.g., '512GB', '512 GB', '0.5TB') and that it is purchasable now (e.g., 'Add to cart', 'Buy now', 'In stock', 'Ships today'). "
        "Do NOT accept pages that are pre-order only or that do not clearly show 512GB."
    )

    # 4) Carrier purchase URL (critical)
    carrier_leaf = evaluator.add_leaf(
        id=f"phone_{phone_no}_carrier_purchase_url",
        desc="Provides ≥1 URL from an eligible carrier (AT&T, Verizon, or T-Mobile) showing the model is available for purchase (not pre-order only).",
        parent=phone_node,
        critical=True
    )
    carrier_sources = list({u.strip() for u in (phone.carrier_urls or []) if isinstance(u, str) and u.strip()})
    carrier_claim = (
        f"At least one provided carrier URL shows the '{phone.model_name or ''}' available for purchase now "
        f"(not pre-order only) at AT&T, Verizon, or T-Mobile."
    )
    carrier_ins = (
        "Judge only using the provided carrier URL(s). Mark Incorrect if none are provided. "
        "The carrier must be one of AT&T, Verizon, or T-Mobile. "
        "The page must show it can be purchased now (e.g., 'Buy', 'Add to cart', 'In stock'). "
        "Do NOT accept pages that are pre-order only."
    )

    # 5) Price for 512GB (critical)
    price_leaf = evaluator.add_leaf(
        id=f"phone_{phone_no}_price_for_512gb",
        desc="Provides a publicly listed, verifiable price for the 512GB model from at least one retailer listing URL.",
        parent=phone_node,
        critical=True
    )
    price_sources_list = []
    if isinstance(phone.price_source_url, str) and phone.price_source_url.strip():
        price_sources_list = [phone.price_source_url.strip()]
    elif retailer_sources:
        price_sources_list = retailer_sources

    price_claim = (
        f"The 512GB configuration of '{phone.model_name or ''}' has a publicly listed price of '{phone.price_512gb or ''}' on the provided retailer page."
    )
    price_ins = (
        "Judge only using the provided URL(s). Mark Incorrect if no URL is provided or if the price string is missing/blank. "
        "Confirm that the page shows a price for the 512GB variant matching the provided price text (allow minor formatting differences like currency symbol or comma separators). "
        "Do NOT use monthly financing amounts; use the one-time retail price. "
        "If only a different storage price is shown or price is absent, mark Incorrect."
    )

    # Batch all verifications for this phone
    verify_jobs = [
        (qi2_claim, qi2_sources, qi2_leaf, qi2_ins),
        (mmw_claim, mmw_sources, mmw_leaf, mmw_ins),
        (cam_claim, cam_sources, cam_leaf, cam_ins),
        (retailer_claim, retailer_sources, retailer_leaf, retailer_ins),
        (carrier_claim, carrier_sources, carrier_leaf, carrier_ins),
        (price_claim, price_sources_list, price_leaf, price_ins),
    ]
    await evaluator.batch_verify(verify_jobs)


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the flagship smartphone selection task (March 2026).
    """
    # Initialize evaluator
    evaluator = Evaluator()
    # Use PARALLEL at root to avoid sequential short-circuiting and to allow independent checks
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

    # Extract candidates
    extracted: CandidateExtraction = await evaluator.extract(
        prompt=prompt_extract_candidates(),
        template_class=CandidateExtraction,
        extraction_name="phone_candidates",
    )

    # Candidate set requirements (critical)
    cand_req_node = evaluator.add_parallel(
        id="candidate_set_requirements",
        desc="Response provides a valid set of candidate phones to evaluate.",
        parent=root,
        critical=True
    )

    total_candidates = len(extracted.candidates or [])
    evaluator.add_custom_node(
        result=total_candidates >= 3,
        id="candidate_count_at_least_3",
        desc="Provides at least 3 candidate smartphone models.",
        parent=cand_req_node,
        critical=True
    )

    # Distinctness among up to first five candidates with non-empty names
    first_five = (extracted.candidates or [])[:5]
    names_norm = [
        _normalize_model_name(c.model_name)
        for c in first_five
        if isinstance(c.model_name, str) and c.model_name.strip()
    ]
    distinct_ok = len(names_norm) == len(set(names_norm)) if names_norm else False
    evaluator.add_custom_node(
        result=distinct_ok,
        id="candidates_distinct",
        desc="All provided candidate smartphone models are different (no repeats).",
        parent=cand_req_node,
        critical=True
    )

    # Evaluate up to five candidates (non-critical)
    eval_node = evaluator.add_parallel(
        id="evaluate_up_to_five_candidates",
        desc="Evaluate candidate phone entries (up to five can be scored) against all per-phone constraints.",
        parent=root,
        critical=False
    )

    # Build per-phone verification subtrees
    phone_node_ids: List[str] = []
    for idx, cand in enumerate(first_five):
        await build_phone_verification_subtree(evaluator, eval_node, cand, idx)
        phone_node_ids.append(f"phone_{idx + 1}")

    # After all phone verifications, compute how many fully compliant (aggregated_score == 1.0)
    compliant_count = 0
    for node_id in phone_node_ids:
        node = evaluator.find_node(node_id)
        if node and node.aggregated_score == 1.0:
            compliant_count += 1

    # Add final critical check: at least three compliant
    evaluator.add_custom_node(
        result=compliant_count >= 3,
        id="at_least_three_compliant",
        desc="At least 3 candidate phone entries satisfy all per-phone critical requirements (Qi2 15W+, mmWave, ≥50MP main camera, 512GB retailer purchase URL with in-stock/not-preorder status, carrier purchase URL with in-stock/not-preorder status, and verifiable 512GB price).",
        parent=root,
        critical=True
    )

    # Add custom info for transparency
    evaluator.add_custom_info(
        info={
            "total_candidates_extracted": total_candidates,
            "evaluated_candidates": len(first_five),
            "compliant_count": compliant_count,
            "evaluated_phone_node_ids": phone_node_ids,
        },
        info_type="evaluation_stats",
        info_name="evaluation_statistics"
    )

    return evaluator.get_summary()