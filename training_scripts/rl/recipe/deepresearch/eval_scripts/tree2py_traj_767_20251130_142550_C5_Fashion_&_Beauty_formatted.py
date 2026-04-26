import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "celebrity_haircare_brand_target_2024"
TASK_DESCRIPTION = (
    "Identify the celebrity-founded haircare brand that meets ALL of the following criteria:\n\n"
    "1. Launched exclusively at Target (both Target stores and Target.com) in 2024\n"
    "2. The initial product lineup consisted of exactly 8 products\n"
    "3. All products are priced between $18.99 and $24.99\n"
    "4. The brand is both vegan and cruelty-free\n"
    "5. Products are formulated without: SLS/SLES, silicones, parabens, phthalates, synthetic dyes, and mineral oil\n"
    "6. The launch occurred in August 2024\n"
    "7. The brand achieved the distinction of being Target's biggest hair care launch on record\n\n"
    "Provide the following information:\n"
    "- The name of the brand\n"
    "- The name of the celebrity founder\n"
    "- The specific launch date (month, day, and year)\n"
    "- Supporting URL references for each major claim"
)


class BrandExtraction(BaseModel):
    brand_name: Optional[str] = None
    celebrity_founder: Optional[str] = None
    specific_launch_date: Optional[str] = None

    brand_name_urls: List[str] = Field(default_factory=list)
    celebrity_founder_urls: List[str] = Field(default_factory=list)
    target_exclusive_urls: List[str] = Field(default_factory=list)
    launch_date_urls: List[str] = Field(default_factory=list)
    initial_product_count_urls: List[str] = Field(default_factory=list)
    price_range_urls: List[str] = Field(default_factory=list)
    vegan_cruelty_free_urls: List[str] = Field(default_factory=list)
    formulation_exclusions_urls: List[str] = Field(default_factory=list)
    target_record_achievement_urls: List[str] = Field(default_factory=list)


def prompt_extract_brand_identification() -> str:
    return """
Extract the requested fields from the answer. Do not invent information. If something is missing, return null (for single values) or an empty array (for URL lists).

Required fields:
1. brand_name: The name of the brand identified in the answer.
2. celebrity_founder: The name of the celebrity founder for the brand.
3. specific_launch_date: The specific launch date as written in the answer, including month, day, and year (e.g., "August 27, 2024"). If the answer does not specify the day, still return what is written; do not invent a day.

Supporting URL references for each major claim (extract actual URLs as they appear in the answer):
- brand_name_urls: URLs that support the brand name (e.g., brand site, press release, retailer page).
- celebrity_founder_urls: URLs that support the identity of the celebrity founder.
- target_exclusive_urls: URLs that support that the launch was exclusive to Target (Target stores and Target.com).
- launch_date_urls: URLs that support the specific launch date (month/day/year) or the launch timing.
- initial_product_count_urls: URLs that support that the initial lineup consisted of exactly 8 products.
- price_range_urls: URLs that support all products are priced in the range $18.99–$24.99.
- vegan_cruelty_free_urls: URLs that support the brand is both vegan and cruelty-free.
- formulation_exclusions_urls: URLs that support that products are formulated without SLS/SLES, silicones, parabens, phthalates, synthetic dyes, and mineral oil.
- target_record_achievement_urls: URLs that support the claim that it was Target's biggest hair care launch on record.

Rules for URLs:
- Extract only actual URLs present in the answer text (including markdown links).
- Do not invent or infer URLs.
- If a URL is missing a protocol, prepend http://.
- If multiple URLs are provided for a claim, include them all.
"""


def has_month_day_year(date_str: Optional[str]) -> bool:
    if not date_str or not isinstance(date_str, str):
        return False
    month_pattern = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    year_pattern = r"\b(19|20)\d{2}\b"
    day_pattern = r"\b([1-9]|[12]\d|3[01])\b"
    has_month = re.search(month_pattern, date_str, flags=re.IGNORECASE) is not None
    has_year = re.search(year_pattern, date_str) is not None
    has_day = re.search(day_pattern, date_str) is not None
    return has_month and has_year and has_day


def merge_sources(*source_lists: List[str]) -> List[str]:
    uniq = []
    seen = set()
    for lst in source_lists:
        for url in lst:
            u = (url or "").strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                uniq.append(u)
    return uniq


async def build_verification_tree(evaluator: Evaluator, extracted: BrandExtraction) -> None:
    main_node = evaluator.add_parallel(
        id="Celebrity_Haircare_Brand_Identification",
        desc="Identify the celebrity-founded haircare brand that matches all constraints and provide required fields with supporting URLs.",
        parent=evaluator.root,
        critical=True
    )

    # Required response fields
    required_fields = evaluator.add_parallel(
        id="Required_Response_Fields",
        desc="Check that the response includes all requested fields.",
        parent=main_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.brand_name and extracted.brand_name.strip()),
        id="Brand_Name_Provided",
        desc="Provide the name of the brand.",
        parent=required_fields,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.celebrity_founder and extracted.celebrity_founder.strip()),
        id="Celebrity_Founder_Name_Provided",
        desc="Provide the name of the celebrity founder.",
        parent=required_fields,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_month_day_year(extracted.specific_launch_date),
        id="Specific_Launch_Date_Provided",
        desc="Provide the specific launch date including month, day, and year.",
        parent=required_fields,
        critical=True
    )

    # Supporting URL references (existence checks)
    refs_node = evaluator.add_parallel(
        id="Supporting_URL_References",
        desc="Provide supporting URL references for each major claim.",
        parent=main_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(extracted.brand_name_urls) > 0,
        id="URL_For_Brand_Name",
        desc="Provide at least one URL supporting the brand name.",
        parent=refs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(extracted.celebrity_founder_urls) > 0,
        id="URL_For_Celebrity_Founder",
        desc="Provide at least one URL supporting the celebrity founder identity.",
        parent=refs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(extracted.target_exclusive_urls) > 0,
        id="URL_For_Target_Exclusive_Launch",
        desc="Provide at least one URL supporting that the launch was exclusive to Target (stores and Target.com).",
        parent=refs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(extracted.launch_date_urls) > 0,
        id="URL_For_Specific_Launch_Date",
        desc="Provide at least one URL supporting the specific launch date (month, day, year).",
        parent=refs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(extracted.initial_product_count_urls) > 0,
        id="URL_For_Initial_Product_Count",
        desc="Provide at least one URL supporting that the initial lineup consisted of exactly 8 products.",
        parent=refs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(extracted.price_range_urls) > 0,
        id="URL_For_Price_Range",
        desc="Provide at least one URL supporting the $18.99–$24.99 price range for the products.",
        parent=refs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(extracted.vegan_cruelty_free_urls) > 0,
        id="URL_For_Vegan_And_Cruelty_Free",
        desc="Provide at least one URL supporting that the brand is vegan and cruelty-free.",
        parent=refs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(extracted.formulation_exclusions_urls) > 0,
        id="URL_For_Formulation_Exclusions",
        desc="Provide at least one URL supporting the formulation exclusions (SLS/SLES, silicones, parabens, phthalates, synthetic dyes, mineral oil).",
        parent=refs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(extracted.target_record_achievement_urls) > 0,
        id="URL_For_Target_Record_Achievement",
        desc="Provide at least one URL supporting the claim that it was Target's biggest hair care launch on record.",
        parent=refs_node,
        critical=True
    )

    # Eligibility criteria verifications
    eligibility = evaluator.add_parallel(
        id="Eligibility_Criteria",
        desc="Verify the brand satisfies all stated constraints.",
        parent=main_node,
        critical=True
    )

    # Prepare leaf nodes
    haircare_node = evaluator.add_leaf(
        id="Product_Category_Haircare",
        desc="The brand must be a haircare line (not skincare, makeup, or fragrance).",
        parent=eligibility,
        critical=True
    )
    celeb_founded_node = evaluator.add_leaf(
        id="Celebrity_Founded",
        desc="The brand must be founded/created by a celebrity.",
        parent=eligibility,
        critical=True
    )
    exclusive_target_node = evaluator.add_leaf(
        id="Launch_Exclusive_At_Target",
        desc="The brand must have launched exclusively at Target (Target stores and Target.com).",
        parent=eligibility,
        critical=True
    )
    launch_year_node = evaluator.add_leaf(
        id="Launch_Year_2024",
        desc="The brand must have launched in 2024.",
        parent=eligibility,
        critical=True
    )
    launch_month_node = evaluator.add_leaf(
        id="Launch_Month_August",
        desc="The brand's launch month must be August.",
        parent=eligibility,
        critical=True
    )
    product_count_node = evaluator.add_leaf(
        id="Initial_Product_Lineup_Exactly_8",
        desc="The initial product lineup must consist of exactly 8 products.",
        parent=eligibility,
        critical=True
    )
    prices_range_node = evaluator.add_leaf(
        id="All_Product_Prices_In_Range",
        desc="All products must be priced between $18.99 and $24.99.",
        parent=eligibility,
        critical=True
    )
    vegan_cf_node = evaluator.add_leaf(
        id="Vegan_And_Cruelty_Free",
        desc="The brand must be both vegan and cruelty-free.",
        parent=eligibility,
        critical=True
    )
    formulation_exclusions_node = evaluator.add_leaf(
        id="Formulated_Without_Listed_Ingredients",
        desc="Products must be formulated without: SLS/SLES, silicones, parabens, phthalates, synthetic dyes, and mineral oil.",
        parent=eligibility,
        critical=True
    )
    target_record_node = evaluator.add_leaf(
        id="Target_Biggest_Haircare_Launch_On_Record",
        desc="The brand must be documented as Target's biggest hair care launch on record.",
        parent=eligibility,
        critical=True
    )

    brand_name = extracted.brand_name or "the brand"
    founder_name = extracted.celebrity_founder or "the founder"

    # Sources per claim
    haircare_sources = merge_sources(extracted.brand_name_urls, extracted.target_exclusive_urls)
    founder_sources = merge_sources(extracted.celebrity_founder_urls)
    exclusive_sources = merge_sources(extracted.target_exclusive_urls)
    launch_date_sources = merge_sources(extracted.launch_date_urls)
    product_count_sources = merge_sources(extracted.initial_product_count_urls)
    price_range_sources = merge_sources(extracted.price_range_urls)
    vegan_cf_sources = merge_sources(extracted.vegan_cruelty_free_urls)
    exclusions_sources = merge_sources(extracted.formulation_exclusions_urls)
    target_record_sources = merge_sources(extracted.target_record_achievement_urls)

    # Build batch verification list
    claims_and_sources: List[tuple[str, List[str] | None, Any, Optional[str]]] = [
        (
            f"The brand {brand_name} is a haircare line (hair care products), not skincare, makeup, or fragrance.",
            haircare_sources if haircare_sources else None,
            haircare_node,
            "Confirm the product category is hair care/haircare. Accept equivalent phrasing such as 'hair care brand' or 'hair styling products'."
        ),
        (
            f"The brand {brand_name} was founded by celebrity {founder_name}.",
            founder_sources if founder_sources else None,
            celeb_founded_node,
            "Verify that the named founder is a widely recognized celebrity (e.g., singer, actor, influencer), and the brand is attributed to them as founder/creator."
        ),
        (
            "The brand launched exclusively at Target (Target stores and Target.com) at launch.",
            exclusive_sources if exclusive_sources else None,
            exclusive_target_node,
            "Look for phrasing like 'exclusive to Target', 'launched exclusively at Target', available only at Target stores and Target.com at launch."
        ),
        (
            "The brand launched in 2024.",
            launch_date_sources if launch_date_sources else None,
            launch_year_node,
            "Use the launch/press release or retailer page to confirm the launch year is 2024."
        ),
        (
            "The brand launched in August 2024.",
            launch_date_sources if launch_date_sources else None,
            launch_month_node,
            "Confirm the month is August and the year is 2024 from the page; minor phrasing variations are acceptable."
        ),
        (
            "The initial product lineup consisted of exactly 8 products.",
            product_count_sources if product_count_sources else None,
            product_count_node,
            "Find an explicit count or clear listing that totals eight products at launch."
        ),
        (
            "All launch products are priced between $18.99 and $24.99.",
            price_range_sources if price_range_sources else None,
            prices_range_node,
            "Check the prices shown; allow minor formatting variations (e.g., $19 vs $19.00). All listed products must fall within [18.99, 24.99]."
        ),
        (
            "The brand is both vegan and cruelty-free.",
            vegan_cf_sources if vegan_cf_sources else None,
            vegan_cf_node,
            "Look for explicit statements that the brand is vegan (no animal-derived ingredients) and cruelty-free (no animal testing)."
        ),
        (
            "The products are formulated without SLS/SLES, silicones, parabens, phthalates, synthetic dyes, and mineral oil.",
            exclusions_sources if exclusions_sources else None,
            formulation_exclusions_node,
            "Match the exclusion list. Allow equivalent wording such as 'sulfates (SLS/SLES)' and 'no silicones/parabens/phthalates/synthetic dyes/mineral oil'."
        ),
        (
            "This was Target's biggest hair care launch on record.",
            target_record_sources if target_record_sources else None,
            target_record_node,
            "Accept paraphrases like 'largest haircare launch ever at Target' or 'biggest hair care launch on record'."
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)


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
    evaluator = Evaluator()
    evaluator.initialize(
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_brand_identification(),
        template_class=BrandExtraction,
        extraction_name="brand_identification"
    )

    await build_verification_tree(evaluator, extracted)

    return evaluator.get_summary()