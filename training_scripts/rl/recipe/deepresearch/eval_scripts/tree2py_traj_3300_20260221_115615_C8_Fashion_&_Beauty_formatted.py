import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "celebrity_fashion_beauty_partnerships_2024"
TASK_DESCRIPTION = (
    "Identify four celebrities who had distinct fashion or beauty brand partnerships announced or prominently featured in 2024, "
    "meeting the following criteria:\n"
    "1) A male celebrity who became a fragrance brand ambassador in 2024 (provide celebrity name, fragrance brand, specific fragrance product, and confirm 2024 announcement).\n"
    "2) A male celebrity who owns an apparel/clothing brand (provide celebrity name, brand name, what the brand offers, and confirm the brand is operational with products available).\n"
    "3) A female celebrity who announced a fashion design collaboration in 2024 (provide celebrity name, collaborating partner, type of collaboration, and the announcement month/year in 2024).\n"
    "4) A female celebrity with an official beauty brand partnership or ambassador role (provide celebrity name, beauty brand, partnership type, and the product category such as skincare, makeup, or fragrance).\n"
    "For each celebrity, provide supporting reference URLs from credible fashion/beauty industry sources."
)


# --------------------------------------------------------------------------- #
# Credible source helpers                                                     #
# --------------------------------------------------------------------------- #
_CREDIBLE_DOMAIN_KEYWORDS = [
    "vogue.com",
    "wwd.com",
    "fashionista.com",
    "harpersbazaar.com",
    "elle.com",
    "gq.com",
    "allure.com",
    "refinery29.com",
    "glamour.com",
    "instyle.com",
    "cosmopolitan.com",
    "people.com",
    "thecut.com",
    "nylon.com",
    "dazeddigital.com",
    "vanityfair.com",
    "sephora.com",
    "ulta.com",
    # Official brand or fashion house/beauty sites
    "loreal.com",
    "lancome.com",
    "lancome-usa.com",
    "chanel.com",
    "dior.com",
    "ysl.com",
    "yslbeauty.com",
    "gucci.com",
    "prada.com",
    "lvmh.com",
    "fentybeauty.com",
    "rarebeauty.com",
    "kyliecosmetics.com",
]


def _normalize_url(u: str) -> Optional[str]:
    if not u:
        return None
    url = u.strip().strip("()[]<>")
    if not url:
        return None
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "http://" + url
    try:
        p = urlparse(url)
        if not p.netloc:
            return None
    except Exception:
        return None
    return url


def normalize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    seen = set()
    out: List[str] = []
    for u in urls:
        nu = _normalize_url(u)
        if nu and nu not in seen:
            seen.add(nu)
            out.append(nu)
    return out


def is_credible_url(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    for kw in _CREDIBLE_DOMAIN_KEYWORDS:
        if kw in host:
            return True
    return False


def any_credible_url(urls: List[str]) -> bool:
    return any(is_credible_url(u) for u in urls)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Celebrity1FragranceAmbassador(BaseModel):
    name: Optional[str] = None
    fragrance_brand: Optional[str] = None
    fragrance_name: Optional[str] = None
    announcement_year: Optional[str] = None  # Expect "2024"
    reference_urls: List[str] = Field(default_factory=list)


class Celebrity2OwnedClothingBrand(BaseModel):
    name: Optional[str] = None
    brand: Optional[str] = None
    brand_description: Optional[str] = None
    brand_availability: Optional[str] = None  # e.g., "shop live", "products available", or sentence from the answer
    reference_urls: List[str] = Field(default_factory=list)


class Celebrity3FashionCollab(BaseModel):
    name: Optional[str] = None
    collaborating_partner: Optional[str] = None
    collaboration_type: Optional[str] = None
    announcement_month: Optional[str] = None  # e.g., "March"
    announcement_year: Optional[str] = None  # Expect "2024"
    reference_urls: List[str] = Field(default_factory=list)


class Celebrity4BeautyPartnership(BaseModel):
    name: Optional[str] = None
    beauty_brand: Optional[str] = None
    partnership_type: Optional[str] = None  # e.g., brand ambassador, spokesperson
    product_category: Optional[str] = None  # e.g., skincare, makeup, fragrance
    reference_urls: List[str] = Field(default_factory=list)


class CelebritiesExtraction(BaseModel):
    celebrity_1_fragrance_ambassador: Optional[Celebrity1FragranceAmbassador] = None
    celebrity_2_owned_clothing_brand: Optional[Celebrity2OwnedClothingBrand] = None
    celebrity_3_fashion_collaboration: Optional[Celebrity3FashionCollab] = None
    celebrity_4_beauty_brand_partnership: Optional[Celebrity4BeautyPartnership] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_celebrities() -> str:
    return """
Extract the four celebrity-brand cases from the answer, strictly following these categories and fields. Return exactly the following JSON schema:

- celebrity_1_fragrance_ambassador:
  - name: string | null  (male celebrity who became a fragrance brand ambassador in 2024)
  - fragrance_brand: string | null
  - fragrance_name: string | null
  - announcement_year: string | null  (must be '2024' if present)
  - reference_urls: string[]  (URLs cited in the answer that support this case)

- celebrity_2_owned_clothing_brand:
  - name: string | null  (male celebrity who owns an apparel/clothing brand)
  - brand: string | null
  - brand_description: string | null  (what the brand offers, e.g., streetwear, athletic apparel, basics, etc.)
  - brand_availability: string | null  (a phrase from the answer indicating the brand has products available for purchase, if provided)
  - reference_urls: string[]  (URLs cited in the answer that support this case; can include brand/retailer pages)

- celebrity_3_fashion_collaboration:
  - name: string | null  (female celebrity with a 2024 fashion design collaboration)
  - collaborating_partner: string | null  (brand or designer)
  - collaboration_type: string | null  (e.g., capsule collection, design partnership)
  - announcement_month: string | null  (month name as written in the answer, e.g., 'March')
  - announcement_year: string | null  (must be '2024' if present)
  - reference_urls: string[]  (URLs cited in the answer that support this case)

- celebrity_4_beauty_brand_partnership:
  - name: string | null  (female celebrity with official beauty partnership/ambassador role)
  - beauty_brand: string | null
  - partnership_type: string | null  (e.g., brand ambassador, spokesperson, face of campaign)
  - product_category: string | null  (e.g., skincare, makeup, fragrance)
  - reference_urls: string[]  (URLs cited in the answer that support this case)

Rules:
- Extract ONLY what is explicitly present in the answer. Do not invent information.
- If any field is missing in the answer, set it to null.
- For URL fields, extract actual URLs listed in the answer (including those in markdown). If a URL is missing 'http://' or 'https://', include it as-is; the evaluator will normalize it later.
- If multiple candidates are given for any category, choose the first complete case mentioned in the answer and ignore the rest.
- For 'announcement_year' in celebrity_1, prefer the stated year of announcement/launch; it should be 2024 for validity in this task.
- For celebrity_3 announcement, prefer the specific month and year (e.g., 'April 2024') if present; otherwise set missing parts to null.
    """.strip()


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_celeb_1_fragrance_ambassador(evaluator: Evaluator, parent) -> None:
    ex: CelebritiesExtraction = evaluator._extraction_results[-1]["result"]  # last extraction
    # Rebuild model object from dict for convenience
    c1 = Celebrity1FragranceAmbassador(**(ex.get("celebrity_1_fragrance_ambassador") or {}))

    node = evaluator.add_parallel(
        id="celebrity_1_fragrance_ambassador",
        desc="A male celebrity who became a fragrance brand ambassador in 2024",
        parent=parent,
        critical=False
    )

    urls = normalize_urls(c1.reference_urls)
    credible = any_credible_url(urls)
    evaluator.add_custom_node(
        result=(len(urls) > 0 and credible),
        id="celebrity_1_reference_url",
        desc="Provision of supporting reference URL(s) from credible fashion or beauty industry sources",
        parent=node,
        critical=True
    )

    # Name
    name_leaf = evaluator.add_leaf(
        id="celebrity_1_name",
        desc="The correct name of the male celebrity fragrance ambassador",
        parent=node,
        critical=True
    )
    claim_name = f"The pages support that the celebrity named '{c1.name}' is involved in this fragrance brand ambassadorship."
    await evaluator.verify(
        claim=claim_name,
        node=name_leaf,
        sources=urls,
        additional_instruction="Verify the pages explicitly connect the named celebrity to the fragrance ambassadorship or campaign."
    )

    # Brand
    brand_leaf = evaluator.add_leaf(
        id="celebrity_1_brand",
        desc="The correct fragrance brand name",
        parent=node,
        critical=True
    )
    claim_brand = f"The pages support that the fragrance brand is '{c1.fragrance_brand}'."
    await evaluator.verify(
        claim=claim_brand,
        node=brand_leaf,
        sources=urls,
        additional_instruction="Verify the pages clearly name this fragrance brand for the celebrity's ambassadorship."
    )

    # Fragrance product name
    frag_name_leaf = evaluator.add_leaf(
        id="celebrity_1_fragrance_name",
        desc="The specific fragrance or cologne name",
        parent=node,
        critical=True
    )
    claim_frag = f"The pages support that the specific fragrance product is '{c1.fragrance_name}'."
    await evaluator.verify(
        claim=claim_frag,
        node=frag_name_leaf,
        sources=urls,
        additional_instruction="Verify the pages explicitly mention this specific fragrance/cologne product name."
    )

    # Announcement year = 2024
    year_leaf = evaluator.add_leaf(
        id="celebrity_1_announcement_year",
        desc="Confirmation that the partnership was announced or became active in 2024",
        parent=node,
        critical=True
    )
    claim_year = "The ambassadorship campaign was announced or made official in 2024."
    await evaluator.verify(
        claim=claim_year,
        node=year_leaf,
        sources=urls,
        additional_instruction="Check the article date or explicit text; allow phrasing such as 'in 2024' or dated press coverage in 2024."
    )


async def verify_celeb_2_owned_clothing_brand(evaluator: Evaluator, parent) -> None:
    ex: CelebritiesExtraction = evaluator._extraction_results[-1]["result"]
    c2 = Celebrity2OwnedClothingBrand(**(ex.get("celebrity_2_owned_clothing_brand") or {}))

    node = evaluator.add_parallel(
        id="celebrity_2_owned_clothing_brand",
        desc="A male celebrity who owns a clothing brand",
        parent=parent,
        critical=False
    )

    urls = normalize_urls(c2.reference_urls)
    credible = any_credible_url(urls)
    evaluator.add_custom_node(
        result=(len(urls) > 0 and credible),
        id="celebrity_2_reference_url",
        desc="Provision of supporting reference URL(s) from credible fashion or beauty industry sources",
        parent=node,
        critical=True
    )

    # Name (ownership link)
    name_leaf = evaluator.add_leaf(
        id="celebrity_2_name",
        desc="The correct name of the male celebrity who owns a clothing brand",
        parent=node,
        critical=True
    )
    claim_name = f"The pages support that '{c2.name}' owns or founded the clothing brand '{c2.brand}'."
    await evaluator.verify(
        claim=claim_name,
        node=name_leaf,
        sources=urls,
        additional_instruction="Confirm that the pages explicitly link the named celebrity to ownership/founding of the brand."
    )

    # Brand name
    brand_leaf = evaluator.add_leaf(
        id="celebrity_2_brand",
        desc="The correct clothing brand name owned by the celebrity",
        parent=node,
        critical=True
    )
    claim_brand = f"The clothing brand is named '{c2.brand}'."
    await evaluator.verify(
        claim=claim_brand,
        node=brand_leaf,
        sources=urls,
        additional_instruction="Verify the brand name as stated on official brand/retailer or credible media pages."
    )

    # Brand description
    desc_leaf = evaluator.add_leaf(
        id="celebrity_2_brand_description",
        desc="Accurate description of what the clothing brand offers",
        parent=node,
        critical=True
    )
    claim_desc = f"The brand '{c2.brand}' offers: {c2.brand_description}."
    await evaluator.verify(
        claim=claim_desc,
        node=desc_leaf,
        sources=urls,
        additional_instruction="Check product/category descriptions; minor wording differences are acceptable if meaning matches."
    )

    # Availability (operational with products available)
    avail_leaf = evaluator.add_leaf(
        id="celebrity_2_brand_availability",
        desc="Confirmation that the brand has products currently available for purchase",
        parent=node,
        critical=True
    )
    claim_avail = (
        f"The brand '{c2.brand}' currently has products available for purchase "
        f"(e.g., a live shop page with items or products listed for sale)."
    )
    await evaluator.verify(
        claim=claim_avail,
        node=avail_leaf,
        sources=urls,
        additional_instruction="Look for a working shop page, product listings, or in-stock items; retailer listings also count."
    )


async def verify_celeb_3_fashion_collaboration(evaluator: Evaluator, parent) -> None:
    ex: CelebritiesExtraction = evaluator._extraction_results[-1]["result"]
    c3 = Celebrity3FashionCollab(**(ex.get("celebrity_3_fashion_collaboration") or {}))

    node = evaluator.add_parallel(
        id="celebrity_3_fashion_collaboration",
        desc="A female celebrity who had a fashion collaboration announced in 2024",
        parent=parent,
        critical=False
    )

    urls = normalize_urls(c3.reference_urls)
    credible = any_credible_url(urls)
    evaluator.add_custom_node(
        result=(len(urls) > 0 and credible),
        id="celebrity_3_reference_url",
        desc="Provision of supporting reference URL(s) from credible fashion or beauty industry sources",
        parent=node,
        critical=True
    )

    # Name
    name_leaf = evaluator.add_leaf(
        id="celebrity_3_name",
        desc="The correct name of the female celebrity with fashion collaboration",
        parent=node,
        critical=True
    )
    claim_name = f"The pages support that the celebrity in this 2024 fashion collaboration is '{c3.name}'."
    await evaluator.verify(
        claim=claim_name,
        node=name_leaf,
        sources=urls,
        additional_instruction="Verify that the pages explicitly name this celebrity as part of the collaboration."
    )

    # Partner
    partner_leaf = evaluator.add_leaf(
        id="celebrity_3_collaborating_partner",
        desc="The correct fashion brand or designer name for the collaboration",
        parent=node,
        critical=True
    )
    claim_partner = f"The collaboration partner is '{c3.collaborating_partner}'."
    await evaluator.verify(
        claim=claim_partner,
        node=partner_leaf,
        sources=urls,
        additional_instruction="Check that the pages clearly identify this partner brand/designer for the collaboration."
    )

    # Collaboration type
    collab_type_leaf = evaluator.add_leaf(
        id="celebrity_3_collaboration_type",
        desc="Accurate description of the type of collaboration (collection, design partnership, etc.)",
        parent=node,
        critical=True
    )
    claim_type = f"The type of collaboration is '{c3.collaboration_type}'."
    await evaluator.verify(
        claim=claim_type,
        node=collab_type_leaf,
        sources=urls,
        additional_instruction="Confirm the collaboration is described as the specified type (e.g., capsule collection, design partnership)."
    )

    # Announcement date (month + 2024)
    ann_date_leaf = evaluator.add_leaf(
        id="celebrity_3_announcement_date",
        desc="The correct month and year the collaboration was announced in 2024",
        parent=node,
        critical=True
    )
    month = c3.announcement_month or "UNKNOWN"
    year = c3.announcement_year or "UNKNOWN"
    claim_date = f"The collaboration was announced in {month} {year}."
    await evaluator.verify(
        claim=claim_date,
        node=ann_date_leaf,
        sources=urls,
        additional_instruction="Verify that the pages indicate both the month and the year of the announcement; the year must be 2024."
    )


async def verify_celeb_4_beauty_partnership(evaluator: Evaluator, parent) -> None:
    ex: CelebritiesExtraction = evaluator._extraction_results[-1]["result"]
    c4 = Celebrity4BeautyPartnership(**(ex.get("celebrity_4_beauty_brand_partnership") or {}))

    node = evaluator.add_parallel(
        id="celebrity_4_beauty_brand_partnership",
        desc="A female celebrity with an official beauty brand partnership or ambassador role",
        parent=parent,
        critical=False
    )

    urls = normalize_urls(c4.reference_urls)
    credible = any_credible_url(urls)
    evaluator.add_custom_node(
        result=(len(urls) > 0 and credible),
        id="celebrity_4_reference_url",
        desc="Provision of supporting reference URL(s) from credible fashion or beauty industry sources",
        parent=node,
        critical=True
    )

    # Name
    name_leaf = evaluator.add_leaf(
        id="celebrity_4_name",
        desc="The correct name of the female celebrity with beauty brand partnership",
        parent=node,
        critical=True
    )
    claim_name = f"The pages support that the celebrity in this beauty brand partnership is '{c4.name}'."
    await evaluator.verify(
        claim=claim_name,
        node=name_leaf,
        sources=urls,
        additional_instruction="Verify the pages explicitly connect this celebrity with the beauty brand partnership."
    )

    # Beauty brand name
    brand_leaf = evaluator.add_leaf(
        id="celebrity_4_beauty_brand",
        desc="The correct beauty brand name",
        parent=node,
        critical=True
    )
    claim_brand = f"The beauty brand involved is '{c4.beauty_brand}'."
    await evaluator.verify(
        claim=claim_brand,
        node=brand_leaf,
        sources=urls,
        additional_instruction="Confirm that the pages name this beauty brand for the partnership."
    )

    # Partnership type
    ptype_leaf = evaluator.add_leaf(
        id="celebrity_4_partnership_type",
        desc="Accurate description of the partnership type (brand ambassador, spokesperson, face of campaign, etc.)",
        parent=node,
        critical=True
    )
    claim_ptype = f"The partnership type/role is '{c4.partnership_type}'."
    await evaluator.verify(
        claim=claim_ptype,
        node=ptype_leaf,
        sources=urls,
        additional_instruction="Verify that the pages describe the role as stated (e.g., brand ambassador, spokesperson, face of campaign)."
    )

    # Product category
    pcat_leaf = evaluator.add_leaf(
        id="celebrity_4_product_category",
        desc="The correct beauty product category (skincare, makeup, fragrance, etc.)",
        parent=node,
        critical=True
    )
    claim_pcat = f"The partnership relates to the '{c4.product_category}' product category."
    await evaluator.verify(
        claim=claim_pcat,
        node=pcat_leaf,
        sources=urls,
        additional_instruction="Confirm the pages link the partnership to this product category (e.g., skincare, makeup, fragrance)."
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
    Evaluate an answer for the 'celebrity_fashion_beauty_partnerships_2024' task.
    Returns the evaluation summary dictionary.
    """
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_celebrities(),
        template_class=CelebritiesExtraction,
        extraction_name="celebrity_partnerships_extraction"
    )

    # Add a parent grouping node (optional; root is already parallel, but this adds clarity mirroring the rubric)
    parent_group = evaluator.add_parallel(
        id="celebrity_fashion_beauty_partnerships",
        desc="Identify four celebrities with distinct fashion or beauty brand partnerships, with one celebrity per partnership category",
        parent=root,
        critical=False
    )

    # Run verifications for each category
    await verify_celeb_1_fragrance_ambassador(evaluator, parent_group)
    await verify_celeb_2_owned_clothing_brand(evaluator, parent_group)
    await verify_celeb_3_fashion_collaboration(evaluator, parent_group)
    await verify_celeb_4_beauty_partnership(evaluator, parent_group)

    # Record some custom info about reference URL credibility coverage
    try:
        ex_dict = extraction.dict()
        stats: List[Tuple[str, int, int]] = []
        for key in [
            "celebrity_1_fragrance_ambassador",
            "celebrity_2_owned_clothing_brand",
            "celebrity_3_fashion_collaboration",
            "celebrity_4_beauty_brand_partnership",
        ]:
            urls = normalize_urls((ex_dict.get(key) or {}).get("reference_urls") or [])
            cred_count = sum(1 for u in urls if is_credible_url(u))
            stats.append((key, len(urls), cred_count))
        evaluator.add_custom_info(
            info={
                "reference_url_coverage": [
                    {"category": k, "total_urls": t, "credible_urls": c}
                    for (k, t, c) in stats
                ]
            },
            info_type="reference_url_stats",
        )
    except Exception:
        pass

    return evaluator.get_summary()