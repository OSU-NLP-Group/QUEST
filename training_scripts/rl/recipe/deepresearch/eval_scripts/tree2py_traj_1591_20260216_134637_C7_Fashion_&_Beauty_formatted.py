import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fashion_beauty_celebrities_2023_2026"
TASK_DESCRIPTION = (
    "Identify four celebrities from the fashion and beauty industry who meet the following criteria:\n\n"
    "1. One celebrity who was officially announced as a global or international makeup brand ambassador for a luxury beauty brand between January 2025 and December 2025.\n\n"
    "2. One celebrity who was officially announced as a brand ambassador and became the face of high jewelry or fine jewelry collections for a luxury fashion house between January 2023 and December 2023.\n\n"
    "3. One celebrity who was featured as a brand ambassador or face of a Spring/Summer 2026 fashion campaign for a luxury fashion or outerwear brand, with the campaign announcement occurring between October 2025 and January 2026.\n\n"
    "4. One celebrity who founded or co-founded their own beauty or skincare brand that is currently active and commercially available as of 2024-2025.\n\n"
    "For each celebrity, provide their name and a reference URL from official brand sources or credible fashion publications that confirms their role and announcement details."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CelebrityItem(BaseModel):
    """Information for one celebrity-role instance."""
    name: Optional[str] = None  # Celebrity's full name
    brand: Optional[str] = None  # Brand/house involved
    role: Optional[str] = None  # e.g., "Global makeup ambassador", "High jewelry ambassador", "SS26 campaign face", "Founder"
    collection_or_campaign: Optional[str] = None  # e.g., "High Jewelry", "Spring/Summer 2026"
    announcement_date: Optional[str] = None  # e.g., "Nov 2025" or "2023-06-14" (string flexible)
    reference_urls: List[str] = Field(default_factory=list)  # URLs that confirm the announcement/role
    brand_website: Optional[str] = None  # Optional official brand site relevant to the role/brand


class CelebritiesExtraction(BaseModel):
    """The four required celebrities, one per category."""
    beauty_ambassador_2025: Optional[CelebrityItem] = None
    jewelry_ambassador_2023: Optional[CelebrityItem] = None
    ss26_campaign_ambassador: Optional[CelebrityItem] = None
    beauty_brand_founder: Optional[CelebrityItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_celebrities() -> str:
    return """
    Extract exactly four celebrity entries from the answer, one for each of the following categories. If a category is not covered in the answer, return `null` for that category.

    Category A (beauty_ambassador_2025):
    - A celebrity who was officially announced as a global or international makeup brand ambassador for a luxury beauty brand.
    - The announcement must occur between January 2025 and December 2025.

    Category B (jewelry_ambassador_2023):
    - A celebrity who was officially announced as a brand ambassador and became the face of high jewelry or fine jewelry collections for a luxury fashion house.
    - The announcement and first jewelry campaign must occur between January 2023 and December 2023.

    Category C (ss26_campaign_ambassador):
    - A celebrity who was featured as a brand ambassador or face of a Spring/Summer 2026 fashion campaign for a luxury fashion or outerwear brand.
    - The campaign announcement must occur between October 2025 and January 2026.

    Category D (beauty_brand_founder):
    - A celebrity who founded or co-founded their own beauty or skincare brand.
    - The brand must be currently active and commercially available as of 2024-2025.

    For each category, extract a JSON object with fields:
    - name: celebrity full name (string)
    - brand: brand or house associated with the role, or the brand they founded (string)
    - role: short description of the role (e.g., "Global makeup ambassador", "High jewelry ambassador", "SS26 campaign face", "Founder") (string)
    - collection_or_campaign: collection or campaign name when applicable (e.g., "High Jewelry", "Spring/Summer 2026"); otherwise null
    - announcement_date: text of the announcement date or month/year provided in the answer (string; flexible format). If absent, set null.
    - reference_urls: an array of URLs explicitly present in the answer that confirm this category’s role and announcement details. Only include valid URLs. If none are present, return an empty array.
    - brand_website: official brand/house website URL directly relevant to this role/category if present; otherwise null.

    Map the four selected entries to these fields in a single JSON object:
    {
      "beauty_ambassador_2025": CelebrityItem or null,
      "jewelry_ambassador_2023": CelebrityItem or null,
      "ss26_campaign_ambassador": CelebrityItem or null,
      "beauty_brand_founder": CelebrityItem or null
    }

    Selection rules:
    - If more than one candidate fits a category, select the best-matching one per the timeframe and role specificity.
    - Extract URLs exactly as they appear (plain or markdown). Do not invent URLs.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _all_sources(item: Optional[CelebrityItem]) -> List[str]:
    """Combine reference URLs and brand website (if any)."""
    if not item:
        return []
    urls = list(item.reference_urls or [])
    if item.brand_website:
        urls.append(item.brand_website)
    # Filter obvious empties
    return [u for u in urls if isinstance(u, str) and u.strip() != ""]


def _credibility_instruction() -> str:
    """Instruction snippet for judging credible sources."""
    return (
        "Consider official brand/house websites and press rooms as authoritative. "
        "Also consider major, credible fashion publications such as WWD, Vogue, Harper's Bazaar, Elle, Business of Fashion, "
        "The Fashion Law, GQ, Vanity Fair, and similar tier publications as credible sources. "
        "Reject low-credibility blogs, forums, or non-authoritative sites. "
        "Only judge as supported if the source explicitly mentions the role/announcement."
    )


# --------------------------------------------------------------------------- #
# Verification functions per category                                         #
# --------------------------------------------------------------------------- #
async def verify_beauty_ambassador_2025(evaluator: Evaluator, parent_node, item: Optional[CelebrityItem]) -> None:
    """
    Celebrity #1:
    - Global/international makeup brand ambassador for a luxury beauty brand.
    - Announcement timeframe: January 2025 – December 2025.
    - Must have verifiable sources.
    """
    node = evaluator.add_parallel(
        id="celebrity_1",
        desc="First celebrity who became a luxury beauty brand ambassador with a global/international makeup role announced in 2025",
        parent=parent_node,
        critical=False
    )

    # Reference existence (extra gating)
    ref_exist = evaluator.add_custom_node(
        result=bool(item and item.reference_urls),
        id="celebrity_1_reference_provided",
        desc="Reference URL(s) provided in the answer for celebrity #1",
        parent=node,
        critical=True
    )

    # Reference credibility and confirmation
    ref_node = evaluator.add_leaf(
        id="celebrity_1_reference",
        desc="A verifiable reference URL from official brand sources or credible fashion publications confirms the announcement",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided sources confirm that {item.name if item else '[unknown]'} was announced in an official capacity related to a makeup ambassador role for {item.brand if item else '[unknown brand]'}",
        node=ref_node,
        sources=item.reference_urls if item else [],
        additional_instruction=_credibility_instruction()
    )

    # Ambassador announcement specifics
    ann_node = evaluator.add_leaf(
        id="celebrity_1_ambassador_announcement",
        desc="The celebrity was officially announced as a global or international makeup brand ambassador for a luxury beauty brand",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"{item.name if item else '[unknown]'} was officially announced as a global or international makeup brand ambassador "
            f"for {item.brand if item else '[unknown brand]'}."
        ),
        node=ann_node,
        sources=_all_sources(item),
        additional_instruction="The source must explicitly state global/international makeup ambassador status (or equivalent wording)."
    )

    # Timeframe check: Jan–Dec 2025
    timeframe_node = evaluator.add_leaf(
        id="celebrity_1_timeframe",
        desc="The ambassador announcement occurred between January 2025 and December 2025",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The announcement occurred in 2025 (between January 1, 2025 and December 31, 2025).",
        node=timeframe_node,
        sources=_all_sources(item),
        additional_instruction=(
            f"If available, the extracted announcement date is: {item.announcement_date if item else 'null'}. "
            "Judge supported only if the source clearly shows a 2025 announcement date."
        )
    )

    # Brand category check: recognized luxury beauty house
    brand_cat_node = evaluator.add_leaf(
        id="celebrity_1_brand_category",
        desc="The brand is a recognized luxury beauty house (such as Dior Beauty, Chanel Beauty, YSL Beauty, or equivalent tier)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"{item.brand if item else '[unknown brand]'} is a recognized luxury beauty house comparable in tier to Dior Beauty, "
            "Chanel Beauty, or YSL Beauty."
        ),
        node=brand_cat_node,
        sources=_all_sources(item),
        additional_instruction=(
            "Use general industry understanding alongside the provided source(s). "
            "It is acceptable if the page itself does not explicitly state 'luxury'; "
            "judge based on brand identity and industry tier (e.g., Dior, Chanel, YSL)."
        )
    )


async def verify_jewelry_ambassador_2023(evaluator: Evaluator, parent_node, item: Optional[CelebrityItem]) -> None:
    """
    Celebrity #2:
    - Brand ambassador and face of high/fine jewelry collections for a luxury fashion house.
    - Announcement timeframe: January 2023 – December 2023.
    - Must have verifiable sources.
    """
    node = evaluator.add_parallel(
        id="celebrity_2",
        desc="Second celebrity who became a luxury fashion brand ambassador for high jewelry collections announced in 2023",
        parent=parent_node,
        critical=False
    )

    ref_exist = evaluator.add_custom_node(
        result=bool(item and item.reference_urls),
        id="celebrity_2_reference_provided",
        desc="Reference URL(s) provided in the answer for celebrity #2",
        parent=node,
        critical=True
    )

    ref_node = evaluator.add_leaf(
        id="celebrity_2_reference",
        desc="A verifiable reference URL from official brand sources or credible fashion publications confirms the announcement",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided sources confirm that {item.name if item else '[unknown]'} became the face/ambassador of high or fine jewelry for {item.brand if item else '[unknown brand]'}",
        node=ref_node,
        sources=item.reference_urls if item else [],
        additional_instruction=_credibility_instruction()
    )

    ann_node = evaluator.add_leaf(
        id="celebrity_2_ambassador_announcement",
        desc="The celebrity was officially announced as a brand ambassador and became the face of high jewelry or fine jewelry collections",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"{item.name if item else '[unknown]'} was officially announced as a brand ambassador and face of high or fine jewelry "
            f"for {item.brand if item else '[unknown brand]'}."
        ),
        node=ann_node,
        sources=_all_sources(item),
        additional_instruction="The source should clearly indicate ambassador status tied to high/fine jewelry."
    )

    timeframe_node = evaluator.add_leaf(
        id="celebrity_2_timeframe",
        desc="The ambassador announcement and first jewelry campaign occurred between January 2023 and December 2023",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The announcement and first jewelry campaign occurred during calendar year 2023.",
        node=timeframe_node,
        sources=_all_sources(item),
        additional_instruction=(
            f"If available, the extracted announcement date is: {item.announcement_date if item else 'null'}. "
            "Judge supported only if the source shows a 2023 announcement/campaign date."
        )
    )

    brand_cat_node = evaluator.add_leaf(
        id="celebrity_2_brand_category",
        desc="The brand is a recognized luxury fashion house with high jewelry collections (such as Louis Vuitton, Dior, Bulgari, or equivalent tier)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"{item.brand if item else '[unknown brand]'} is a recognized luxury fashion house known for high/fine jewelry collections "
            "(e.g., Louis Vuitton, Dior, Bulgari)."
        ),
        node=brand_cat_node,
        sources=_all_sources(item),
        additional_instruction=(
            "Use industry understanding and the provided sources. "
            "It is acceptable if the page itself does not literally say 'luxury'; judge based on brand identity/tier."
        )
    )


async def verify_ss26_campaign_ambassador(evaluator: Evaluator, parent_node, item: Optional[CelebrityItem]) -> None:
    """
    Celebrity #3:
    - Ambassador or face of a Spring/Summer 2026 fashion campaign for a luxury fashion or outerwear brand.
    - Announcement timeframe: October 2025 – January 2026.
    - Must have verifiable sources.
    """
    node = evaluator.add_parallel(
        id="celebrity_3",
        desc="Third celebrity who became a fashion brand campaign ambassador for Spring/Summer 2026 collections announced in late 2025 or early 2026",
        parent=parent_node,
        critical=False
    )

    ref_exist = evaluator.add_custom_node(
        result=bool(item and item.reference_urls),
        id="celebrity_3_reference_provided",
        desc="Reference URL(s) provided in the answer for celebrity #3",
        parent=node,
        critical=True
    )

    ref_node = evaluator.add_leaf(
        id="celebrity_3_reference",
        desc="A verifiable reference URL from official brand sources or credible fashion publications confirms the campaign announcement",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided sources confirm that {item.name if item else '[unknown]'} was announced as a face/ambassador for a Spring/Summer 2026 campaign for {item.brand if item else '[unknown brand]'}",
        node=ref_node,
        sources=item.reference_urls if item else [],
        additional_instruction=_credibility_instruction()
    )

    role_node = evaluator.add_leaf(
        id="celebrity_3_campaign_role",
        desc="The celebrity was featured as a brand ambassador or face of a Spring/Summer 2026 fashion campaign",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"{item.name if item else '[unknown]'} was featured as a brand ambassador or campaign face for Spring/Summer 2026 for "
            f"{item.brand if item else '[unknown brand]'}."
        ),
        node=role_node,
        sources=_all_sources(item),
        additional_instruction="The source must clearly indicate SS26 involvement as ambassador/face."
    )

    timeframe_node = evaluator.add_leaf(
        id="celebrity_3_timeframe",
        desc="The campaign announcement occurred between October 2025 and January 2026",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The campaign announcement occurred between October 1, 2025 and January 31, 2026.",
        node=timeframe_node,
        sources=_all_sources(item),
        additional_instruction=(
            f"If available, the extracted announcement date is: {item.announcement_date if item else 'null'}. "
            "Judge supported only if the source shows an announcement date in Oct 2025–Jan 2026."
        )
    )

    brand_cat_node = evaluator.add_leaf(
        id="celebrity_3_brand_category",
        desc="The brand is a recognized luxury fashion or outerwear brand (such as Mackage, Burberry, Moncler, or equivalent tier)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"{item.brand if item else '[unknown brand]'} is a recognized luxury fashion/outerwear brand comparable to Mackage, "
            "Burberry, or Moncler."
        ),
        node=brand_cat_node,
        sources=_all_sources(item),
        additional_instruction=(
            "Use industry understanding alongside the provided source(s). "
            "It is acceptable if the page itself does not literally say 'luxury'; judge based on brand tier/identity."
        )
    )


async def verify_beauty_brand_founder(evaluator: Evaluator, parent_node, item: Optional[CelebrityItem]) -> None:
    """
    Celebrity #4:
    - Founder or co-founder of a beauty/skincare brand.
    - Brand is currently active and commercially available (as of 2024–2025).
    - Must have verifiable sources.
    """
    node = evaluator.add_parallel(
        id="celebrity_4",
        desc="Fourth celebrity who founded or co-founded their own beauty or skincare brand that is currently active and commercially available",
        parent=parent_node,
        critical=False
    )

    ref_exist = evaluator.add_custom_node(
        result=bool(item and item.reference_urls),
        id="celebrity_4_reference_provided",
        desc="Reference URL(s) provided in the answer for celebrity #4",
        parent=node,
        critical=True
    )

    ref_node = evaluator.add_leaf(
        id="celebrity_4_reference",
        desc="A verifiable reference URL confirms the celebrity's role as founder and the brand's current commercial availability",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The provided sources confirm that {item.name if item else '[unknown]'} is the founder or co-founder of "
            f"{item.brand if item else '[unknown brand]'}, and that the brand is commercially available."
        ),
        node=ref_node,
        sources=item.reference_urls if item else [],
        additional_instruction=_credibility_instruction()
    )

    founder_node = evaluator.add_leaf(
        id="celebrity_4_brand_founder",
        desc="The celebrity is the founder or co-founder of a beauty, skincare, or cosmetics brand (not just an ambassador)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{item.name if item else '[unknown]'} is the founder or co-founder of the brand {item.brand if item else '[unknown brand]'}.",
        node=founder_node,
        sources=_all_sources(item),
        additional_instruction="Source should explicitly state founder/co-founder role; ambassador-only roles should not be accepted."
    )

    active_node = evaluator.add_leaf(
        id="celebrity_4_brand_active",
        desc="The brand is currently active and sells products commercially (as of 2024-2025)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The brand {item.brand if item else '[unknown brand]'} is currently active and selling products (commercially available) as of 2024–2025.",
        node=active_node,
        sources=_all_sources(item),
        additional_instruction=(
            "Look for evidence of current product listings, e-commerce functionality, or recent press in 2024–2025 indicating availability."
        )
    )

    brand_cat_node = evaluator.add_leaf(
        id="celebrity_4_brand_category",
        desc="The brand specializes in skincare, makeup, or beauty products",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The brand {item.brand if item else '[unknown brand]'} specializes in skincare, makeup, or beauty products.",
        node=brand_cat_node,
        sources=_all_sources(item),
        additional_instruction="The source should indicate product types in skincare, makeup, or beauty."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the fashion & beauty celebrity ambassador/founder task.
    Returns the evaluator's structured summary.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root node aggregates categories independently
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

    # Extract the four category entries from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_celebrities(),
        template_class=CelebritiesExtraction,
        extraction_name="celebrities_selection"
    )

    # Build the rubric tree and run verifications for each category
    await verify_beauty_ambassador_2025(evaluator, root, extraction.beauty_ambassador_2025)
    await verify_jewelry_ambassador_2023(evaluator, root, extraction.jewelry_ambassador_2023)
    await verify_ss26_campaign_ambassador(evaluator, root, extraction.ss26_campaign_ambassador)
    await verify_beauty_brand_founder(evaluator, root, extraction.beauty_brand_founder)

    # Return structured summary with tree, extractions, and score
    return evaluator.get_summary()