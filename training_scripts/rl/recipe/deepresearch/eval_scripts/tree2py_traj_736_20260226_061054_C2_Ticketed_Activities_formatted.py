import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "tn_aquarium_admission_pricing"
TASK_DESCRIPTION = """
What is the admission pricing structure at the Tennessee Aquarium in Chattanooga, Tennessee, including the adult admission cost, the youth (ages 5-17) admission cost, and the age threshold below which children receive free admission?
"""


# ----------------------------- Data Models --------------------------------- #
class AdmissionExtraction(BaseModel):
    # Venue identification
    venue_name: Optional[str] = None
    location_city: Optional[str] = None
    location_state: Optional[str] = None

    # Pricing details extracted verbatim from the answer
    adult_price: Optional[str] = None          # e.g., "$39.95"
    adult_age_bracket: Optional[str] = None    # e.g., "18+"
    youth_price: Optional[str] = None          # e.g., "$29.95"
    youth_age_bracket: Optional[str] = None    # e.g., "5-17"

    # Free admission policy
    free_admission_policy: Optional[str] = None      # e.g., "Children ages 4 and under receive free admission"
    free_age_threshold: Optional[str] = None         # e.g., "3 and under" or "4 and under"

    # All URLs mentioned in the answer (including any tnaqua.org pages)
    source_urls: List[str] = Field(default_factory=list)


# --------------------------- Extraction Prompt ----------------------------- #
def prompt_extract_admission_info() -> str:
    return """
    Extract the Tennessee Aquarium admission information as presented in the answer.

    Required fields:
    - venue_name: The venue name explicitly mentioned (e.g., "Tennessee Aquarium").
    - location_city: The city mentioned for the venue (e.g., "Chattanooga").
    - location_state: The state mentioned (e.g., "Tennessee" or "TN").
    - adult_price: The adult admission price exactly as stated in the answer text (include currency symbol if present).
    - adult_age_bracket: The age definition for adult admission if provided (e.g., "18+").
    - youth_price: The youth admission price exactly as stated (include currency symbol if present).
    - youth_age_bracket: The age definition for youth admission if provided (e.g., "5-17").
    - free_admission_policy: The text that states the free admission rule for children, verbatim from the answer.
    - free_age_threshold: The threshold phrase if specified (e.g., "3 and under" or "4 and under"). If not clearly stated, return null.
    - source_urls: All URLs explicitly listed in the answer (including markdown links). Extract only valid URLs.

    Notes:
    - Do not infer or invent any information; extract only what is explicitly stated in the answer.
    - If any field is missing in the answer, return null for that field.
    - For URLs, return full, valid URLs exactly as presented or normalized with http(s) if missing.
    """


# ------------------------------ Helpers ------------------------------------ #
def filter_urls_by_domain(urls: List[str], domain_keyword: str) -> List[str]:
    """Filter URLs that contain a given domain keyword."""
    if not urls:
        return []
    domain_keyword = domain_keyword.lower()
    return [u for u in urls if isinstance(u, str) and domain_keyword in u.lower()]


def choose_sources(all_urls: List[str], preferred_domain: str) -> List[str]:
    """Prefer URLs from a specific domain; fall back to all URLs if none."""
    preferred = filter_urls_by_domain(all_urls, preferred_domain)
    return preferred if preferred else all_urls


def _nonempty(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


def _normalize_location(city: Optional[str], state: Optional[str]) -> str:
    c = (city or "").strip()
    s = (state or "").strip()
    if not c and not s:
        return ""
    if c and s:
        return f"{c}, {s}"
    return c or s


# -------------------------- Verification Logic ----------------------------- #
async def verify_identify_venue(
    evaluator: Evaluator,
    parent_node,
    extracted: AdmissionExtraction
) -> None:
    """
    Build and verify the 'Identify_Venue' subtree:
    - Check the answer identifies the venue correctly.
    - Ensure a tnaqua.org URL is present.
    - Verify the venue/location is supported by a tnaqua.org page.
    """
    identify_node = evaluator.add_parallel(
        id="Identify_Venue",
        desc="Correctly identify the venue as the Tennessee Aquarium located in Chattanooga, Tennessee",
        parent=parent_node,
        critical=True
    )

    # 1) Check the answer itself identifies the venue and location
    venue_leaf = evaluator.add_leaf(
        id="Venue_Identified_In_Answer",
        desc="The answer identifies the venue as the Tennessee Aquarium located in Chattanooga, Tennessee",
        parent=identify_node,
        critical=True
    )
    venue_claim = "The answer identifies the venue as the Tennessee Aquarium located in Chattanooga, Tennessee."
    await evaluator.verify(
        claim=venue_claim,
        node=venue_leaf,
        additional_instruction="Focus on the answer text; consider minor variations in state name (e.g., TN vs Tennessee) acceptable."
    )

    # 2) Require at least one tnaqua.org URL
    has_tnaqua_url = len(filter_urls_by_domain(extracted.source_urls, "tnaqua.org")) > 0
    evaluator.add_custom_node(
        result=has_tnaqua_url,
        id="Has_TNAQUA_URL",
        desc="At least one provided reference URL is from tnaqua.org",
        parent=identify_node,
        critical=True
    )

    # 3) Verify the venue/location via tnaqua.org sources
    ref_leaf = evaluator.add_leaf(
        id="Reference_URL",
        desc="Provide a valid reference URL from tnaqua.org confirming the venue and location",
        parent=identify_node,
        critical=True
    )
    tnaqua_urls = filter_urls_by_domain(extracted.source_urls, "tnaqua.org")
    location_text = _normalize_location(extracted.location_city, extracted.location_state)
    # Construct a world-fact claim supported by official site
    loc_claim = "The Tennessee Aquarium is located in Chattanooga, Tennessee."
    await evaluator.verify(
        claim=loc_claim,
        node=ref_leaf,
        sources=tnaqua_urls if tnaqua_urls else extracted.source_urls,
        additional_instruction="Use official pages on tnaqua.org that mention Chattanooga, Tennessee explicitly (e.g., Visit, Directions, Hours & Tickets)."
    )


async def verify_admission_pricing(
    evaluator: Evaluator,
    parent_node,
    extracted: AdmissionExtraction
) -> None:
    """
    Build and verify the 'Provide_Admission_Pricing' subtree:
    - Adult price (ages 18+) as stated in the answer, supported by tnaqua.org.
    - Youth price (ages 5-17) as stated in the answer, supported by tnaqua.org.
    - Free admission age policy (3 and under or 4 and under), supported by tnaqua.org.
    """
    pricing_node = evaluator.add_parallel(
        id="Provide_Admission_Pricing",
        desc="Provide accurate admission pricing information for different age categories",
        parent=parent_node,
        critical=True
    )

    # Preferred sources: use tnaqua.org when available
    preferred_sources = choose_sources(extracted.source_urls, "tnaqua.org")

    # Adult price check
    evaluator.add_custom_node(
        result=_nonempty(extracted.adult_price),
        id="Adult_Price_Provided",
        desc="Adult price is provided in the answer",
        parent=pricing_node,
        critical=True
    )
    adult_leaf = evaluator.add_leaf(
        id="Adult_Admission_Price",
        desc="State that adult admission (ages 18+) costs $39.95",
        parent=pricing_node,
        critical=True
    )
    adult_age = extracted.adult_age_bracket if _nonempty(extracted.adult_age_bracket) else "18+"
    adult_price_text = (extracted.adult_price or "").strip()
    adult_claim = f"Adult admission (ages {adult_age}) costs {adult_price_text}."
    await evaluator.verify(
        claim=adult_claim,
        node=adult_leaf,
        sources=preferred_sources,
        additional_instruction=(
            "Verify the base general admission price for an adult (ages 18+)."
            " Accept minor formatting like 'plus tax'. Prefer official pricing/tickets pages on tnaqua.org."
        )
    )

    # Youth price check
    evaluator.add_custom_node(
        result=_nonempty(extracted.youth_price),
        id="Youth_Price_Provided",
        desc="Youth price is provided in the answer",
        parent=pricing_node,
        critical=True
    )
    youth_leaf = evaluator.add_leaf(
        id="Youth_Admission_Price",
        desc="State that youth admission (ages 5-17) costs $29.95",
        parent=pricing_node,
        critical=True
    )
    youth_age = extracted.youth_age_bracket if _nonempty(extracted.youth_age_bracket) else "5-17"
    youth_price_text = (extracted.youth_price or "").strip()
    youth_claim = f"Youth admission (ages {youth_age}) costs {youth_price_text}."
    await evaluator.verify(
        claim=youth_claim,
        node=youth_leaf,
        sources=preferred_sources,
        additional_instruction=(
            "Verify the base general admission price for youth (ages 5-17)."
            " Accept minor formatting like 'plus tax'. Prefer official pricing/tickets pages on tnaqua.org."
        )
    )

    # Free admission policy check
    evaluator.add_custom_node(
        result=_nonempty(extracted.free_admission_policy) or _nonempty(extracted.free_age_threshold),
        id="Free_Policy_Provided",
        desc="Free admission policy for young children is provided in the answer",
        parent=pricing_node,
        critical=True
    )
    free_leaf = evaluator.add_leaf(
        id="Free_Admission_Age_Policy",
        desc="State that children ages 3 and under (or 4 and under) receive free admission",
        parent=pricing_node,
        critical=True
    )
    # Prefer threshold; fallback to policy text
    free_threshold = (extracted.free_age_threshold or "").strip()
    if not free_threshold and _nonempty(extracted.free_admission_policy):
        # Try to reuse the policy short form or just use the policy text as-is
        free_threshold = extracted.free_admission_policy.strip()

    # Build the claim; handle subset logic in instruction:
    free_claim = f"Children {free_threshold} receive free admission."
    await evaluator.verify(
        claim=free_claim,
        node=free_leaf,
        sources=preferred_sources,
        additional_instruction=(
            "Confirm the official free-admission policy for young children on tnaqua.org."
            " If the official page states '4 and under free', then a claim '3 and under free' should be considered supported"
            " (subset is logically true). However, if the official page states '3 and under free', then a claim '4 and under free'"
            " should not be considered supported."
        )
    )


# --------------------------- Main Evaluation ------------------------------- #
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
    Evaluate an answer for Tennessee Aquarium admission pricing.
    """
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_admission_info(),
        template_class=AdmissionExtraction,
        extraction_name="admission_extraction"
    )

    # Create the main critical node (since root is non-critical by design)
    main_node = evaluator.add_sequential(
        id="Provide_Tennessee_Aquarium_Admission_Information",
        desc="Provide accurate admission pricing information for the Tennessee Aquarium in Chattanooga, including the venue identification, free admission age policy, adult pricing, and youth pricing",
        parent=root,
        critical=True
    )

    # Add ground truth info for transparency (used only for reporting)
    evaluator.add_ground_truth({
        "expected": {
            "adult_price": "$39.95",
            "youth_price": "$29.95",
            "free_age_threshold_allowed": ["3 and under", "4 and under"],
            "adult_age_bracket": "18+",
            "youth_age_bracket": "5-17",
            "official_domain": "tnaqua.org"
        }
    }, gt_type="pricing_expectations")

    # Build and run verifications
    await verify_identify_venue(evaluator, main_node, extracted)
    await verify_admission_pricing(evaluator, main_node, extracted)

    return evaluator.get_summary()