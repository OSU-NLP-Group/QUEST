import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "global_fragrance_ambassador_2024_2025"
TASK_DESCRIPTION = (
    "Identify one professional actor or actress who was announced as a Global Brand Ambassador "
    "(or equivalent senior ambassadorship title) for a luxury beauty brand between January 2024 and "
    "December 2025. The luxury beauty brand must be part of a major luxury conglomerate (such as LVMH, "
    "Shiseido, L'Oréal Luxe, or Estée Lauder Companies) or be an established independent luxury beauty "
    "house. The ambassadorship must explicitly include fragrance or perfume products as part of the partnership. "
    "For your identified celebrity, provide: (1) The celebrity's full name, (2) The luxury beauty brand name, "
    "(3) The official announcement date (month and year) of the ambassadorship, (4) The name of at least one "
    "specific fragrance or fragrance line associated with their ambassadorship, and (5) A reference URL from an "
    "official brand source or major beauty industry publication documenting the partnership."
)

# Allowed date range (inclusive)
ALLOWED_START = (2024, 1)
ALLOWED_END = (2025, 12)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CelebrityAmbassadorItem(BaseModel):
    celebrity_full_name: Optional[str] = None
    brand_name: Optional[str] = None
    ambassador_title: Optional[str] = None  # e.g., Global Brand Ambassador, Global Face, House Ambassador
    announcement_date: Optional[str] = None  # month and year string (e.g., "March 2024", "2024-03", "03/2024")
    fragrance_name_or_line: Optional[str] = None  # e.g., "Dior Sauvage", "Gucci Bloom", "J'Adore"
    reference_urls: List[str] = Field(default_factory=list)


class CelebrityAmbassadorExtraction(BaseModel):
    items: List[CelebrityAmbassadorItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_celebrity_ambassador() -> str:
    return """
Extract the information explicitly provided in the answer about celebrity global fragrance ambassadorship(s).

Rules:
- Return a list named "items". Each element corresponds to one identified celebrity ambassadorship in the answer.
- Extract only what is explicitly in the answer; do not invent or infer missing information.
- For URLs, extract full URLs explicitly present in the answer (including those in markdown links). If none, use an empty list.

For each item, extract:
- celebrity_full_name: Full name of the celebrity.
- brand_name: The luxury beauty brand's name.
- ambassador_title: The formal title used (e.g., "Global Brand Ambassador", "Global Face", "House Ambassador", "Global Spokesperson"). If absent, set null.
- announcement_date: The official announcement date as month and year if available (e.g., "March 2024", "2024-03", "03/2024"). If present in another clear variant, extract that string. If absent, set null.
- fragrance_name_or_line: The name of at least one specific fragrance or fragrance line associated with the ambassadorship (e.g., "Sauvage", "J'Adore"). If absent, set null.
- reference_urls: An array of all reference URLs in the answer that document the partnership (brand press release, brand site page, or major beauty industry publication). If none provided, return [].

Return JSON strictly matching the schema.
"""


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _parse_month_year(s: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    Parse a string for a (year, month) tuple.
    Accepts formats like:
      - "March 2024", "Mar 2024", "Mar. 2024"
      - "2024-03", "2024/03", "2024.03"
      - "03/2024", "03-2024", "03.2024"
    Returns (year, month) or None if cannot parse.
    """
    if not s:
        return None
    txt = s.strip().lower()

    # e.g., "March 2024", "Mar 2024", "Mar. 2024"
    m = re.search(r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
                  r"sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+(\d{4})\b", txt, re.I)
    if m:
        month_str = m.group(1).replace(".", "").lower()
        year = int(m.group(2))
        month = _MONTHS.get(month_str, None)
        if month:
            return (year, month)

    # e.g., "2024-03", "2024/3", "2024.3"
    m = re.search(r"\b(20\d{2})[-/\.](\d{1,2})\b", txt)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        if 1 <= month <= 12:
            return (year, month)

    # e.g., "03/2024", "3-2024", "03.2024"
    m = re.search(r"\b(\d{1,2})[-/\.](20\d{2})\b", txt)
    if m:
        month = int(m.group(1))
        year = int(m.group(2))
        if 1 <= month <= 12:
            return (year, month)

    return None


def _in_allowed_range(ym: Optional[Tuple[int, int]], start: Tuple[int, int] = ALLOWED_START,
                      end: Tuple[int, int] = ALLOWED_END) -> bool:
    if ym is None:
        return False
    y, m = ym
    sy, sm = start
    ey, em = end
    return (y, m) >= (sy, sm) and (y, m) <= (ey, em)


def _first_or_placeholder(items: List[CelebrityAmbassadorItem]) -> CelebrityAmbassadorItem:
    return items[0] if items else CelebrityAmbassadorItem()


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    root_node,
    extracted: CelebrityAmbassadorExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and perform verifications.
    We will:
    - Enforce exactly one item (fail if multiple)
    - Enforce required fields exist
    - Verify source validity and content support
    - Verify eligibility criteria with URL evidence
    """
    # Create main critical node
    main = evaluator.add_parallel(
        id="celebrity_global_fragrance_ambassador_identification",
        desc=("Verify the response identifies exactly one qualifying professional actor/actress announced (Jan 2024–Dec 2025) "
              "as a Global Brand Ambassador (or equivalent) for a luxury beauty brand that explicitly covers fragrance, "
              "and provides all required fields with an appropriate citation URL."),
        parent=root_node,
        critical=True,
    )

    # Select the first item for detailed verification but check exact count
    items = extracted.items or []
    selected = _first_or_placeholder(items)
    urls = selected.reference_urls or []

    # Exactly one celebrity (critical leaf)
    evaluator.add_custom_node(
        result=(len(items) == 1),
        id="exactly_one_celebrity",
        desc="The response names exactly one celebrity as the identified ambassador (not multiple candidates).",
        parent=main,
        critical=True
    )

    # Required Output Fields group (critical)
    req = evaluator.add_parallel(
        id="required_output_fields",
        desc="Check that all required output fields are present.",
        parent=main,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(selected.celebrity_full_name and selected.celebrity_full_name.strip()),
        id="celebrity_full_name_provided",
        desc="The response provides the celebrity's full name.",
        parent=req,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(selected.brand_name and selected.brand_name.strip()),
        id="luxury_beauty_brand_name_provided",
        desc="The response provides the luxury beauty brand name.",
        parent=req,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(selected.announcement_date and selected.announcement_date.strip()),
        id="announcement_date_month_year_provided",
        desc="The response provides the official announcement date of the ambassadorship in month and year format.",
        parent=req,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(selected.fragrance_name_or_line and selected.fragrance_name_or_line.strip()),
        id="specific_fragrance_or_line_provided",
        desc="The response names at least one specific fragrance or fragrance line associated with the ambassadorship.",
        parent=req,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(urls and len(urls) > 0),
        id="reference_url_provided",
        desc="The response provides at least one reference URL documenting the partnership.",
        parent=req,
        critical=True
    )

    # Source and Citation Validity (critical)
    src = evaluator.add_parallel(
        id="source_and_citation_validity",
        desc="Check that the citation meets source-type requirements and supports the claimed partnership details.",
        parent=main,
        critical=True
    )

    # Leaf: Source type validity
    leaf_src_type = evaluator.add_leaf(
        id="source_type_official_or_major_beauty_publication",
        desc="The reference URL is from an official brand source or a major beauty industry publication.",
        parent=src,
        critical=True
    )
    claim_src_type = (
        "At least one of the provided URLs is either: "
        "(a) an official brand source (e.g., the brand's own domain, corporate press release), or "
        "(b) a major beauty industry publication (e.g., WWD, Business of Fashion, Vogue Beauty, Elle Beauty, "
        "Harper's Bazaar Beauty, Allure, Cosmetics Business, Beauty Packaging, The Perfume Society, L'Officiel)."
    )
    await evaluator.verify(
        claim=claim_src_type,
        node=leaf_src_type,
        sources=urls,
        additional_instruction=(
            "Judge based on the domain and the site's identity/content. Official brand sources include the brand's "
            "domains or corporate press pages. Major industry publications include globally recognized fashion/beauty "
            "media as listed. If none of the URLs are official or major publications, the claim is not supported."
        )
    )

    # Leaf: Citation supports key details (ambassador + fragrance explicit)
    leaf_citation_support = evaluator.add_leaf(
        id="citation_supports_key_partnership_details",
        desc=("The provided reference URL(s) substantively document the partnership and explicitly support "
              "(at minimum) the ambassador relationship and the inclusion of fragrance/perfume in the partnership."),
        parent=src,
        critical=True
    )
    celeb = selected.celebrity_full_name or "the celebrity"
    brand = selected.brand_name or "the brand"
    claim_citation_support = (
        f"The provided page(s) explicitly document that {celeb} was announced as a Global Brand Ambassador or "
        f"equivalent senior ambassadorial role for {brand}, and that the partnership explicitly includes fragrance/perfume."
    )
    await evaluator.verify(
        claim=claim_citation_support,
        node=leaf_citation_support,
        sources=urls,
        additional_instruction=(
            "Look for clear statements like 'Global Brand Ambassador', 'Global Face', 'House Ambassador', 'Global Spokesperson', "
            "or equivalent senior-level title, and explicit mention of 'fragrance', 'perfume', 'eau de parfum', 'cologne', or 'scent'. "
            "If the page only mentions makeup/skincare and not fragrance, do not support this claim."
        )
    )

    # Eligibility Criteria (critical)
    elig = evaluator.add_parallel(
        id="eligibility_criteria",
        desc="Check that the identified celebrity, brand, and partnership satisfy all stated constraints.",
        parent=main,
        critical=True
    )

    # Professional actor/actress
    leaf_actor = evaluator.add_leaf(
        id="professional_actor_or_actress",
        desc=("The identified person is an actor or actress with film and/or television credits "
              "(i.e., is a professional actor/actress rather than a non-acting celebrity)."),
        parent=elig,
        critical=True
    )
    claim_actor = (
        f"{celeb} is a professional actor or actress with film and/or television credits."
    )
    await evaluator.verify(
        claim=claim_actor,
        node=leaf_actor,
        sources=urls,
        additional_instruction=(
            "Accept if the page refers to the person as an actor or actress (e.g., 'actor', 'actress', 'film star', "
            "'screen star', 'actor-producer', etc.) or lists film/TV credits. If the page clearly positions them solely "
            "as a non-acting celebrity (e.g., only musician or only model) without acting credits, mark unsupported."
        )
    )

    # Global Ambassador title or equivalent
    leaf_title = evaluator.add_leaf(
        id="global_ambassador_title_or_equivalent",
        desc=("The person holds an official 'Global Brand Ambassador' title (or an equivalent senior ambassadorship title) "
              "for the luxury beauty brand."),
        parent=elig,
        critical=True
    )
    title_text = selected.ambassador_title or "a senior ambassadorial title"
    claim_title = (
        f"{celeb} was announced as a Global Brand Ambassador (or an equivalent senior ambassadorship title such as "
        f"'Global Face', 'Global Spokesperson', 'House Ambassador', or 'International Ambassador') for {brand}. "
        f"If available, the specific title is '{title_text}'."
    )
    await evaluator.verify(
        claim=claim_title,
        node=leaf_title,
        sources=urls,
        additional_instruction=(
            "Support if the page clearly states a global/senior ambassadorial role (e.g., 'Global Brand Ambassador', "
            "'Global Face', 'Global Spokesperson', 'House Ambassador', 'International Ambassador'). "
            "Do not support if only a regional/limited scope (e.g., 'Asia-Pacific ambassador') unless it is explicitly global."
        )
    )

    # Luxury brand qualification
    leaf_luxury = evaluator.add_leaf(
        id="luxury_brand_qualification",
        desc=("The luxury beauty brand is part of a major luxury conglomerate (e.g., LVMH, Shiseido, L'Oréal Luxe, "
              "Estée Lauder Companies) OR is an established independent luxury beauty house."),
        parent=elig,
        critical=True
    )
    claim_luxury = (
        f"{brand} is a luxury beauty brand that is either part of a major luxury conglomerate (such as LVMH, Shiseido, "
        f"L'Oréal Luxe, Estée Lauder Companies, Kering Beauté, Puig) or is an established independent luxury beauty house."
    )
    await evaluator.verify(
        claim=claim_luxury,
        node=leaf_luxury,
        sources=urls,
        additional_instruction=(
            "Use the content and the site's identity to determine whether the brand is a prestige/luxury beauty brand. "
            "Evidence can include explicit statements of the parent group (e.g., LVMH/ELC/L'Oréal Luxe/Shiseido/Puig/Kering Beauté) "
            "or clear positioning as a luxury house (e.g., 'Maison', 'haute parfumerie', 'luxury fragrance house'). "
            "If the page does not substantiate luxury/parent-group status, mark unsupported."
        )
    )

    # Partnership explicitly includes fragrance
    leaf_includes_fragrance = evaluator.add_leaf(
        id="partnership_explicitly_includes_fragrance",
        desc="The ambassadorship explicitly includes fragrance/perfume products as part of the partnership.",
        parent=elig,
        critical=True
    )
    claim_includes_fragrance = (
        "The ambassadorship explicitly includes fragrance/perfume products (e.g., fragrance, perfume, eau de parfum, EDP, cologne, scent)."
    )
    await evaluator.verify(
        claim=claim_includes_fragrance,
        node=leaf_includes_fragrance,
        sources=urls,
        additional_instruction=(
            "Look for explicit mentions tying the ambassadorial role to fragrance/perfume products. "
            "If only makeup/skincare is mentioned with no fragrance, mark unsupported."
        )
    )

    # Official announcement date within Jan 2024 - Dec 2025 (custom check on provided date string)
    parsed_ym = _parse_month_year(selected.announcement_date)
    evaluator.add_custom_node(
        result=_in_allowed_range(parsed_ym, ALLOWED_START, ALLOWED_END),
        id="official_announcement_between_jan_2024_and_dec_2025",
        desc="The official announcement date of the ambassadorship is between January 2024 and December 2025 (inclusive).",
        parent=elig,
        critical=True
    )

    # Appeared in fragrance campaign or advertising
    leaf_campaign = evaluator.add_leaf(
        id="appeared_in_fragrance_campaign_or_advertising",
        desc=("There is public documentation that the celebrity appeared in at least one campaign or advertising "
              "material for the referenced fragrance/fragrance line."),
        parent=elig,
        critical=True
    )
    fragrance = selected.fragrance_name_or_line or "the referenced fragrance/fragrance line"
    claim_campaign = (
        f"{celeb} appeared in at least one campaign or advertising (e.g., film, image campaign, visual assets) "
        f"for {fragrance}."
    )
    await evaluator.verify(
        claim=claim_campaign,
        node=leaf_campaign,
        sources=urls,
        additional_instruction=(
            "Look for explicit statements, images, or video descriptors that {celeb} appears in campaign/advertising materials "
            "for the named fragrance or line. Words like 'campaign', 'ad', 'film', 'visuals', 'spot', 'campaign images' are signals."
        )
    )

    # Partnership currently active (no termination indicated in provided documentation)
    leaf_active = evaluator.add_leaf(
        id="partnership_currently_active",
        desc=("There is no indication (in the provided documentation) that the partnership has been terminated or expired at the time of the response."),
        parent=elig,
        critical=True
    )
    claim_active = (
        "The provided page(s) do not indicate that the ambassadorship has ended, expired, or been terminated; "
        "they present the partnership as current or ongoing at the time of publication/update."
    )
    await evaluator.verify(
        claim=claim_active,
        node=leaf_active,
        sources=urls,
        additional_instruction=(
            "Check for terms like 'former', 'ended', 'expired', 'no longer', 'past ambassador'. "
            "If such indications exist, the claim is unsupported. If the page presents the role in present tense with no "
            "termination mentioned, support the claim."
        )
    )

    # Record some helpful custom info for debugging
    evaluator.add_custom_info(
        info={
            "extracted_item_count": len(items),
            "selected_item": selected.dict(),
            "parsed_announcement_year_month": parsed_ym,
            "reference_url_count": len(urls),
        },
        info_type="debug_info"
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
    Evaluate an answer for the global fragrance ambassador (2024–2025) task.
    """
    # Initialize evaluator with a parallel root (we'll add a critical child as the main rubric root)
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_celebrity_ambassador(),
        template_class=CelebrityAmbassadorExtraction,
        extraction_name="celebrity_ambassador_extraction"
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()