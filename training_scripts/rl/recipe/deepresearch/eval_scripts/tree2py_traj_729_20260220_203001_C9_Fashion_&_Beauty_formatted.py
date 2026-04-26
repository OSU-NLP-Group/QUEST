import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "celeb_fashion_beauty_2024_2025"
TASK_DESCRIPTION = """
Identify 4 celebrities who launched or announced major fashion and beauty partnerships between February 2024 and February 2025, meeting the following specific criteria:

Celebrity 1: Founded and launched a haircare brand in February 2024 with a foundation collection containing exactly 8 products, and achieved a record-breaking retail partnership with a major US beauty retailer (becoming their largest haircare launch ever).

Celebrity 2: Founded and launched a haircare brand in June 2024 with a launch collection containing exactly 9 products (including shampoo, conditioners, treatment, stylers, and a tool), as an extension of their existing beauty brand empire.

Celebrity 3: Was announced as a US brand ambassador for an Italian lingerie brand in November 2024, and also launched a 7-piece blazer capsule collection with an Italian fashion brand in February 2025.

Celebrity 4: Was announced as a brand ambassador for L'Oréal Paris (a major drugstore beauty brand) in May 2024, and participated in mother-daughter lingerie brand campaigns with an Italian lingerie brand in 2024.

For each celebrity, provide their name, the relevant brand names, specific announcement/launch dates, and reference URLs confirming each requirement.
"""


# ----------------------------- Data Models --------------------------------- #
class Celebrity1Data(BaseModel):
    name: Optional[str] = None
    haircare_brand: Optional[str] = None
    launch_date: Optional[str] = None  # e.g., "February 20, 2024"
    launch_urls: List[str] = Field(default_factory=list)

    founder_urls: List[str] = Field(default_factory=list)

    products_urls: List[str] = Field(default_factory=list)

    retailer_name: Optional[str] = None
    retail_urls: List[str] = Field(default_factory=list)


class Celebrity2Data(BaseModel):
    name: Optional[str] = None
    haircare_brand: Optional[str] = None
    launch_date: Optional[str] = None  # e.g., "June 13, 2024"
    launch_urls: List[str] = Field(default_factory=list)

    founder_urls: List[str] = Field(default_factory=list)

    products_urls: List[str] = Field(default_factory=list)

    existing_beauty_brand: Optional[str] = None
    empire_urls: List[str] = Field(default_factory=list)


class Celebrity3Data(BaseModel):
    name: Optional[str] = None
    lingerie_brand: Optional[str] = None
    lingerie_announcement_date: Optional[str] = None  # e.g., "November 5, 2024"
    lingerie_urls: List[str] = Field(default_factory=list)
    us_market_urls: List[str] = Field(default_factory=list)
    brand_origin_urls: List[str] = Field(default_factory=list)

    fashion_brand: Optional[str] = None
    capsule_launch_date: Optional[str] = None  # e.g., "February 12, 2025"
    capsule_urls: List[str] = Field(default_factory=list)
    fashion_origin_urls: List[str] = Field(default_factory=list)


class Celebrity4Data(BaseModel):
    name: Optional[str] = None
    loreal_announcement_date: Optional[str] = None  # e.g., "May 7, 2024"
    loreal_urls: List[str] = Field(default_factory=list)

    # Mother-daughter lingerie campaign details
    campaign_brand: Optional[str] = None
    daughter_name: Optional[str] = None
    campaign_urls: List[str] = Field(default_factory=list)
    brand_origin_urls: List[str] = Field(default_factory=list)


class CelebritiesExtraction(BaseModel):
    celebrity1: Optional[Celebrity1Data] = None
    celebrity2: Optional[Celebrity2Data] = None
    celebrity3: Optional[Celebrity3Data] = None
    celebrity4: Optional[Celebrity4Data] = None


# ------------------------- Extraction Prompt ------------------------------- #
def prompt_extract_celebrities() -> str:
    return """
    Extract structured information for exactly 4 celebrities that meet the task's criteria window (Feb 2024–Feb 2025). 
    Return a JSON object with fields 'celebrity1', 'celebrity2', 'celebrity3', 'celebrity4'.
    For each celebrity, extract only what is explicitly present in the answer text; do not invent data.

    celebrity1 (Haircare – Feb 2024; 8-product foundation; record retail achievement):
      - name: Full name of the celebrity
      - haircare_brand: Brand name they founded/launched
      - launch_date: Specific launch date string in February 2024 (e.g., "February 20, 2024"); if missing, null
      - launch_urls: URLs that confirm the brand launched in Feb 2024
      - founder_urls: URLs that confirm the celebrity is the founder/owner (not just an ambassador)
      - products_urls: URLs that list/confirm the foundation collection contains exactly 8 products
      - retailer_name: Name of the major US beauty retailer with the record achievement (e.g., Sephora, Ulta Beauty); if missing, null
      - retail_urls: URLs confirming the record-breaking claim (e.g., "largest haircare launch ever" at the retailer)

    celebrity2 (Haircare – June 2024; 9-product launch; category breakdown; extension of existing beauty empire):
      - name: Full name of the celebrity
      - haircare_brand: Brand name they founded/launched
      - launch_date: Specific launch date string in June 2024; if missing, null
      - launch_urls: URLs confirming the brand launched in June 2024
      - founder_urls: URLs confirming the celebrity founded/owns the haircare brand
      - products_urls: URLs confirming the launch collection contains exactly 9 products and the categories (shampoo, conditioners, treatment, stylers, tool)
      - existing_beauty_brand: Name of the pre-existing beauty brand (empire) this haircare extends; if missing, null
      - empire_urls: URLs confirming the connection/extension to the existing beauty brand

    celebrity3 (Italian lingerie US ambassador – Nov 2024; 7-piece blazer capsule – Feb 2025):
      - name: Full name of the celebrity
      - lingerie_brand: Italian lingerie brand name
      - lingerie_announcement_date: Specific announcement date string in November 2024; if missing, null
      - lingerie_urls: URLs confirming the ambassadorship announcement in Nov 2024
      - us_market_urls: URLs confirming the role is specifically for the US market
      - brand_origin_urls: URLs confirming the lingerie brand is Italian
      - fashion_brand: Italian fashion brand name for the blazer capsule
      - capsule_launch_date: Specific launch date string in February 2025; if missing, null
      - capsule_urls: URLs confirming the 7-piece blazer capsule launch in Feb 2025
      - fashion_origin_urls: URLs confirming the fashion brand is Italian

    celebrity4 (L'Oréal Paris ambassador – May 2024; mother-daughter lingerie campaigns – 2024):
      - name: Full name of the celebrity
      - loreal_announcement_date: Specific announcement date string in May 2024; if missing, null
      - loreal_urls: URLs confirming the L'Oréal Paris ambassadorship announcement in May 2024
      - campaign_brand: Italian lingerie brand name involved in mother-daughter campaigns; if missing, null
      - daughter_name: Name of the celebrity's daughter who participated; if missing, null
      - campaign_urls: URLs confirming mother-daughter campaign participation and timing (year 2024)
      - brand_origin_urls: URLs confirming the lingerie brand is Italian

    Rules:
    - Extract only URLs explicitly present in the answer (plain, markdown, etc.). If none, return [].
    - Do not infer dates; if the answer doesn't give a specific date string, set the date field to null.
    - Preserve letter casing for names/brands as in the answer.
    """


# ------------------------- Verification Helpers ---------------------------- #
def safe_str(value: Optional[str], fallback: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() != "" else fallback


# ------------------------- Celebrity 1 Verification ------------------------ #
async def verify_celebrity_1(evaluator: Evaluator, parent_node, c: Celebrity1Data) -> None:
    celeb_node = evaluator.add_parallel(
        id="celebrity_1",
        desc="Celebrity who launched a haircare brand in February 2024 as founder with 8 products and record US retail achievement",
        parent=parent_node,
        critical=False
    )

    brand = safe_str(c.haircare_brand, "the haircare brand")
    celeb_name = safe_str(c.name, "the celebrity")
    retailer = safe_str(c.retailer_name, "the major US beauty retailer")

    # Launch timing (Feb 2024)
    launch_timing = evaluator.add_parallel(
        id="c1_launch_timing",
        desc="Haircare brand launched in February 2024",
        parent=celeb_node,
        critical=True
    )
    node = evaluator.add_leaf(
        id="c1_launch_month_verification",
        desc="Provide specific launch date in February 2024",
        parent=launch_timing,
        critical=True
    )
    claim = f"The haircare brand '{brand}' launched in February 2024 (date: {safe_str(c.launch_date, 'unspecified')})."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.launch_urls,
        additional_instruction="Verify the month/year is February 2024; the exact day is provided if available."
    )

    node = evaluator.add_leaf(
        id="c1_launch_timing_reference",
        desc="Provide URL confirming February 2024 launch date",
        parent=launch_timing,
        critical=True
    )
    claim = f"The provided sources explicitly confirm that '{brand}' launched in February 2024."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.launch_urls,
        additional_instruction="Look for an explicit statement of launch timing (February 2024)."
    )

    # Founder status
    founder_status = evaluator.add_parallel(
        id="c1_founder_status",
        desc="Celebrity is the founder/owner of the haircare brand",
        parent=celeb_node,
        critical=True
    )
    node = evaluator.add_leaf(
        id="c1_brand_ownership_verification",
        desc="Confirm celebrity founded and owns the brand",
        parent=founder_status,
        critical=True
    )
    claim = f"{celeb_name} is the founder/owner (not merely an ambassador) of '{brand}'."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.founder_urls,
        additional_instruction="Check for 'founder', 'co-founder', 'owns', or equivalent language confirming ownership."
    )
    node = evaluator.add_leaf(
        id="c1_founder_reference",
        desc="Provide URL confirming founder status",
        parent=founder_status,
        critical=True
    )
    claim = f"The sources confirm that {celeb_name} founded and owns '{brand}'."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.founder_urls,
        additional_instruction="Ensure the founder/ownership status is explicitly stated in the source."
    )

    # Product collection: exactly 8 products
    product_collection = evaluator.add_parallel(
        id="c1_product_collection",
        desc="Foundation collection contains exactly 8 products",
        parent=celeb_node,
        critical=True
    )
    node = evaluator.add_leaf(
        id="c1_product_count_verification",
        desc="Confirm the foundation collection has 8 products",
        parent=product_collection,
        critical=True
    )
    claim = f"The foundation collection of '{brand}' contains exactly 8 products."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.products_urls,
        additional_instruction="Count/list in the source should clearly indicate 8 distinct products in the foundation collection."
    )
    node = evaluator.add_leaf(
        id="c1_product_collection_reference",
        desc="Provide URL listing the 8 products",
        parent=product_collection,
        critical=True
    )
    claim = f"The provided sources list or enumerate the 8 products in the foundation collection of '{brand}'."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.products_urls,
        additional_instruction="Source should list product names or present a clearly countable set of 8 items."
    )

    # Retail achievement: record-breaking at major US beauty retailer
    retail = evaluator.add_parallel(
        id="c1_retail_achievement",
        desc="Achieved record-breaking retail partnership with major US beauty retailer",
        parent=celeb_node,
        critical=True
    )
    node = evaluator.add_leaf(
        id="c1_retail_record_verification",
        desc="Confirm the brand achieved largest haircare launch record at the retail partner",
        parent=retail,
        critical=True
    )
    claim = f"At {retailer}, '{brand}' achieved a record such as 'largest haircare launch ever' or equivalent."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.retail_urls,
        additional_instruction="Look for explicit phrasing like 'largest haircare launch ever', 'record-breaking', or similar at the named retailer."
    )
    node = evaluator.add_leaf(
        id="c1_retail_achievement_reference",
        desc="Provide URL confirming the record-breaking retail partnership",
        parent=retail,
        critical=True
    )
    claim = f"The sources explicitly confirm a record-breaking retail partnership at {retailer} for '{brand}'."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.retail_urls,
        additional_instruction="Confirm the retailer is a major US beauty retailer and the record claim is explicit."
    )


# ------------------------- Celebrity 2 Verification ------------------------ #
async def verify_celebrity_2(evaluator: Evaluator, parent_node, c: Celebrity2Data) -> None:
    celeb_node = evaluator.add_parallel(
        id="celebrity_2",
        desc="Celebrity who launched a haircare brand in June 2024 as founder with 9 products and category breakdown; extension of existing beauty empire",
        parent=parent_node,
        critical=False
    )

    brand = safe_str(c.haircare_brand, "the haircare brand")
    celeb_name = safe_str(c.name, "the celebrity")
    empire_brand = safe_str(c.existing_beauty_brand, "the existing beauty brand")

    # Launch timing (June 2024)
    launch_timing = evaluator.add_parallel(
        id="c2_launch_timing",
        desc="Haircare brand launched in June 2024",
        parent=celeb_node,
        critical=True
    )
    node = evaluator.add_leaf(
        id="c2_launch_month_verification",
        desc="Provide specific launch date in June 2024",
        parent=launch_timing,
        critical=True
    )
    claim = f"The haircare brand '{brand}' launched in June 2024 (date: {safe_str(c.launch_date, 'unspecified')})."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.launch_urls,
        additional_instruction="Verify month/year is June 2024; day provided if available."
    )
    node = evaluator.add_leaf(
        id="c2_launch_timing_reference",
        desc="Provide URL confirming June 2024 launch date",
        parent=launch_timing,
        critical=True
    )
    claim = f"The sources explicitly confirm that '{brand}' launched in June 2024."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.launch_urls,
        additional_instruction="Look for explicit launch timing statements."
    )

    # Founder status
    founder_status = evaluator.add_parallel(
        id="c2_founder_status",
        desc="Celebrity is the founder/owner of the haircare brand",
        parent=celeb_node,
        critical=True
    )
    node = evaluator.add_leaf(
        id="c2_brand_ownership_verification",
        desc="Confirm celebrity founded and owns the brand",
        parent=founder_status,
        critical=True
    )
    claim = f"{celeb_name} is the founder/owner (not merely an ambassador) of '{brand}'."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.founder_urls,
        additional_instruction="Check for explicit founder/owner language."
    )
    node = evaluator.add_leaf(
        id="c2_founder_reference",
        desc="Provide URL confirming founder status",
        parent=founder_status,
        critical=True
    )
    claim = f"The sources confirm that {celeb_name} founded and owns '{brand}'."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.founder_urls,
        additional_instruction="Ensure ownership status is explicit."
    )

    # Product collection: exactly 9 products + categories
    product_collection = evaluator.add_parallel(
        id="c2_product_collection",
        desc="Launch collection contains exactly 9 products",
        parent=celeb_node,
        critical=True
    )
    node = evaluator.add_leaf(
        id="c2_product_count_verification",
        desc="Confirm the launch collection has 9 products",
        parent=product_collection,
        critical=True
    )
    claim = f"The launch collection of '{brand}' contains exactly 9 products."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.products_urls,
        additional_instruction="The source should clearly indicate 9 distinct products."
    )
    node = evaluator.add_leaf(
        id="c2_product_breakdown_verification",
        desc="Confirm product breakdown includes shampoos, conditioners, treatment, stylers, and tool",
        parent=product_collection,
        critical=True
    )
    claim = f"The 9 products for '{brand}' include shampoos, conditioners, a treatment, stylers, and a tool."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.products_urls,
        additional_instruction="Look for category breakdown including all specified categories."
    )
    node = evaluator.add_leaf(
        id="c2_product_collection_reference",
        desc="Provide URL listing the 9 products and their categories",
        parent=product_collection,
        critical=True
    )
    claim = f"The sources list the 9 products for '{brand}' and show their categories."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.products_urls,
        additional_instruction="Source should list product names and/or classify them into the specified categories."
    )

    # Beauty empire connection
    empire = evaluator.add_parallel(
        id="c2_beauty_empire_connection",
        desc="Haircare brand is an extension of celebrity's existing beauty brand empire",
        parent=celeb_node,
        critical=True
    )
    node = evaluator.add_leaf(
        id="c2_existing_beauty_brand_verification",
        desc="Confirm celebrity has a pre-existing major beauty brand",
        parent=empire,
        critical=True
    )
    claim = f"{celeb_name} has an existing major beauty brand '{empire_brand}'."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.empire_urls,
        additional_instruction="Verify the existence and prominence of the existing beauty brand."
    )
    node = evaluator.add_leaf(
        id="c2_beauty_empire_reference",
        desc="Provide URL confirming connection to existing beauty brand",
        parent=empire,
        critical=True
    )
    claim = f"'{brand}' is explicitly described as an extension/expansion of the existing beauty brand '{empire_brand}'."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.empire_urls,
        additional_instruction="Source should link the haircare brand to the celebrity's existing beauty brand."
    )


# ------------------------- Celebrity 3 Verification ------------------------ #
async def verify_celebrity_3(evaluator: Evaluator, parent_node, c: Celebrity3Data) -> None:
    celeb_node = evaluator.add_parallel(
        id="celebrity_3",
        desc="US brand ambassador for Italian lingerie brand in Nov 2024; 7-piece blazer capsule with Italian fashion brand in Feb 2025",
        parent=parent_node,
        critical=False
    )

    celeb_name = safe_str(c.name, "the celebrity")
    lingerie_brand = safe_str(c.lingerie_brand, "the lingerie brand")
    fashion_brand = safe_str(c.fashion_brand, "the fashion brand")

    # Lingerie ambassadorship
    amb = evaluator.add_parallel(
        id="c3_lingerie_ambassadorship",
        desc="Announced as US brand ambassador for Italian lingerie brand in November 2024",
        parent=celeb_node,
        critical=True
    )

    # Announcement timing
    ann_timing = evaluator.add_parallel(
        id="c3_announcement_timing",
        desc="Ambassadorship announced specifically in November 2024",
        parent=amb,
        critical=True
    )
    node = evaluator.add_leaf(
        id="c3_announcement_date_verification",
        desc="Provide specific announcement date in November 2024",
        parent=ann_timing,
        critical=True
    )
    claim = f"{celeb_name} was announced as a brand ambassador for '{lingerie_brand}' in November 2024 (date: {safe_str(c.lingerie_announcement_date, 'unspecified')})."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.lingerie_urls,
        additional_instruction="Confirm November 2024; the precise date is used if available."
    )
    node = evaluator.add_leaf(
        id="c3_announcement_reference",
        desc="Provide URL confirming November 2024 announcement",
        parent=ann_timing,
        critical=True
    )
    claim = f"The sources explicitly confirm the ambassadorship announcement for '{lingerie_brand}' occurred in November 2024."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.lingerie_urls,
        additional_instruction="Look for explicit month/year statements."
    )

    # US market specification
    us_spec = evaluator.add_parallel(
        id="c3_us_market_specification",
        desc="Ambassador role is specifically for the US market",
        parent=amb,
        critical=True
    )
    node = evaluator.add_leaf(
        id="c3_us_market_verification",
        desc="Confirm the ambassadorship is designated as US brand ambassador",
        parent=us_spec,
        critical=True
    )
    claim = f"{celeb_name}'s ambassadorship for '{lingerie_brand}' is specifically designated for the US market."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.us_market_urls or c.lingerie_urls,
        additional_instruction="The source should explicitly mention 'US' or 'United States' market/ambassador designation."
    )
    node = evaluator.add_leaf(
        id="c3_us_market_reference",
        desc="Provide URL confirming US market specification",
        parent=us_spec,
        critical=True
    )
    claim = f"The provided sources confirm the role is specifically for the US market."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.us_market_urls or c.lingerie_urls,
        additional_instruction="Look for wording like 'US brand ambassador' or equivalent."
    )

    # Italian brand verification
    brand_origin = evaluator.add_parallel(
        id="c3_italian_brand_verification",
        desc="Lingerie brand is Italian",
        parent=amb,
        critical=True
    )
    node = evaluator.add_leaf(
        id="c3_brand_origin_verification",
        desc="Confirm the lingerie brand is Italian",
        parent=brand_origin,
        critical=True
    )
    claim = f"'{lingerie_brand}' is an Italian lingerie brand."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.brand_origin_urls or c.lingerie_urls,
        additional_instruction="Source should state the brand's Italian origin (country of origin/headquartered in Italy)."
    )
    node = evaluator.add_leaf(
        id="c3_brand_origin_reference",
        desc="Provide URL confirming Italian origin",
        parent=brand_origin,
        critical=True
    )
    claim = f"The sources explicitly confirm that '{lingerie_brand}' is Italian."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.brand_origin_urls or c.lingerie_urls,
        additional_instruction="Look for explicit 'Italian' origin statements."
    )

    # Fashion capsule collaboration
    capsule = evaluator.add_parallel(
        id="c3_fashion_capsule_collaboration",
        desc="Launched 7-piece blazer capsule collection with Italian fashion brand in February 2025",
        parent=celeb_node,
        critical=True
    )

    # Launch timing (Feb 2025)
    cap_launch = evaluator.add_parallel(
        id="c3_capsule_launch_timing",
        desc="Capsule collection launched in February 2025",
        parent=capsule,
        critical=True
    )
    node = evaluator.add_leaf(
        id="c3_launch_date_verification",
        desc="Provide specific launch date in February 2025",
        parent=cap_launch,
        critical=True
    )
    claim = f"{celeb_name} launched a capsule collection with '{fashion_brand}' in February 2025 (date: {safe_str(c.capsule_launch_date, 'unspecified')})."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.capsule_urls,
        additional_instruction="Confirm month/year is February 2025; the exact date if available."
    )
    node = evaluator.add_leaf(
        id="c3_launch_timing_reference",
        desc="Provide URL confirming February 2025 launch date",
        parent=cap_launch,
        critical=True
    )
    claim = f"The sources explicitly confirm that the capsule launch occurred in February 2025."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.capsule_urls,
        additional_instruction="Look for explicit month/year statements."
    )

    # Collection specifications (7 blazers)
    specs = evaluator.add_parallel(
        id="c3_collection_specifications",
        desc="Collection contains exactly 7 blazer pieces",
        parent=capsule,
        critical=True
    )
    node = evaluator.add_leaf(
        id="c3_piece_count_verification",
        desc="Confirm collection has 7 pieces",
        parent=specs,
        critical=True
    )
    claim = f"The capsule collection with '{fashion_brand}' contains exactly 7 pieces."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.capsule_urls,
        additional_instruction="The source should enumerate a set of 7 items."
    )
    node = evaluator.add_leaf(
        id="c3_blazer_focus_verification",
        desc="Confirm collection focuses on blazers",
        parent=specs,
        critical=True
    )
    claim = f"The capsule is a blazer-focused collection."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.capsule_urls,
        additional_instruction="The source should explicitly indicate the focus on blazers."
    )
    node = evaluator.add_leaf(
        id="c3_collection_reference",
        desc="Provide URL confirming 7-piece blazer collection",
        parent=specs,
        critical=True
    )
    claim = f"The sources confirm a 7-piece blazer capsule."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.capsule_urls,
        additional_instruction="Look for wording that matches both '7-piece' and 'blazer capsule'."
    )

    # Italian fashion brand origin
    italian_fashion = evaluator.add_parallel(
        id="c3_italian_fashion_brand",
        desc="Fashion brand is Italian",
        parent=capsule,
        critical=True
    )
    node = evaluator.add_leaf(
        id="c3_fashion_brand_origin_verification",
        desc="Confirm the fashion brand is Italian",
        parent=italian_fashion,
        critical=True
    )
    claim = f"'{fashion_brand}' is an Italian fashion brand."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.fashion_origin_urls or c.capsule_urls,
        additional_instruction="Source should explicitly indicate the brand's Italian origin."
    )
    node = evaluator.add_leaf(
        id="c3_fashion_brand_reference",
        desc="Provide URL confirming Italian fashion brand",
        parent=italian_fashion,
        critical=True
    )
    claim = f"The sources confirm that '{fashion_brand}' is Italian."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.fashion_origin_urls or c.capsule_urls,
        additional_instruction="Look for explicit confirmation of Italian origin."
    )


# ------------------------- Celebrity 4 Verification ------------------------ #
async def verify_celebrity_4(evaluator: Evaluator, parent_node, c: Celebrity4Data) -> None:
    celeb_node = evaluator.add_parallel(
        id="celebrity_4",
        desc="L'Oréal Paris ambassador in May 2024; mother-daughter lingerie campaigns with Italian brand in 2024",
        parent=parent_node,
        critical=False
    )

    celeb_name = safe_str(c.name, "the celebrity")
    daughter = safe_str(c.daughter_name, "the daughter")
    lingerie_brand = safe_str(c.campaign_brand, "the lingerie brand")

    # Drugstore beauty ambassadorship (L'Oréal Paris)
    amb = evaluator.add_parallel(
        id="c4_drugstore_beauty_ambassadorship",
        desc="Announced as brand ambassador for L'Oréal Paris in May 2024",
        parent=celeb_node,
        critical=True
    )

    # Announcement timing (May 2024)
    ann_timing = evaluator.add_parallel(
        id="c4_announcement_timing",
        desc="Ambassadorship announced in May 2024",
        parent=amb,
        critical=True
    )
    node = evaluator.add_leaf(
        id="c4_may_announcement_verification",
        desc="Provide specific announcement date in May 2024",
        parent=ann_timing,
        critical=True
    )
    claim = f"{celeb_name} was announced as a L'Oréal Paris brand ambassador in May 2024 (date: {safe_str(c.loreal_announcement_date, 'unspecified')})."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.loreal_urls,
        additional_instruction="Confirm month/year is May 2024; include the exact date if present."
    )
    node = evaluator.add_leaf(
        id="c4_announcement_timing_reference",
        desc="Provide URL confirming May 2024 announcement",
        parent=ann_timing,
        critical=True
    )
    claim = f"The sources explicitly confirm the L'Oréal Paris ambassadorship announcement in May 2024."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.loreal_urls,
        additional_instruction="Look for explicit month/year statements."
    )

    # L'Oréal Paris identity
    loreal_verify = evaluator.add_parallel(
        id="c4_loreal_paris_verification",
        desc="Brand is specifically L'Oréal Paris",
        parent=amb,
        critical=True
    )
    node = evaluator.add_leaf(
        id="c4_brand_identity_verification",
        desc="Confirm the brand is L'Oréal Paris",
        parent=loreal_verify,
        critical=True
    )
    claim = "The brand is specifically L'Oréal Paris."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.loreal_urls,
        additional_instruction="Ensure the brand named in the source is precisely 'L'Oréal Paris'."
    )
    node = evaluator.add_leaf(
        id="c4_loreal_reference",
        desc="Provide URL confirming L'Oréal Paris ambassadorship",
        parent=loreal_verify,
        critical=True
    )
    claim = f"The sources confirm that {celeb_name} is a L'Oréal Paris brand ambassador."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.loreal_urls,
        additional_instruction="Look for explicit ambassadorship wording."
    )

    # Drugstore brand nature (adjusted to critical to satisfy framework constraints)
    drugstore = evaluator.add_parallel(
        id="c4_drugstore_brand_nature",
        desc="L'Oréal Paris is a major drugstore/accessible beauty brand",
        parent=amb,
        critical=True  # Adjusted to satisfy critical-parent constraint
    )
    node = evaluator.add_leaf(
        id="c4_drugstore_classification_verification",
        desc="Confirm L'Oréal Paris is a drugstore beauty brand",
        parent=drugstore,
        critical=True  # Adjusted to satisfy critical-parent constraint
    )
    claim = "L'Oréal Paris is a major drugstore/accessible beauty brand."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.loreal_urls,
        additional_instruction="Look for classification indicating 'drugstore' or widely accessible mass-market positioning."
    )

    # Mother-daughter lingerie campaign (2024)
    campaign = evaluator.add_parallel(
        id="c4_mother_daughter_campaign",
        desc="Participated in mother-daughter lingerie campaigns with Italian brand in 2024",
        parent=celeb_node,
        critical=True
    )

    family = evaluator.add_parallel(
        id="c4_family_collaboration",
        desc="Campaigns featured celebrity and her daughter",
        parent=campaign,
        critical=True
    )
    node = evaluator.add_leaf(
        id="c4_mother_daughter_verification",
        desc="Confirm campaigns featured the celebrity and her daughter together",
        parent=family,
        critical=True
    )
    claim = f"The lingerie campaign for '{lingerie_brand}' featured {celeb_name} and her daughter together."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.campaign_urls,
        additional_instruction="The source should show or state that both the mother and daughter participated together."
    )
    node = evaluator.add_leaf(
        id="c4_daughter_identity_verification",
        desc="Provide name of celebrity's daughter who participated",
        parent=family,
        critical=True  # Adjusted to satisfy critical-parent constraint
    )
    claim = f"The daughter's name who participated is '{daughter}'."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.campaign_urls,
        additional_instruction="Verify the daughter's name explicitly in the campaign source."
    )
    node = evaluator.add_leaf(
        id="c4_family_campaign_reference",
        desc="Provide URL confirming mother-daughter campaign participation",
        parent=family,
        critical=True
    )
    claim = "The provided sources confirm mother-daughter participation in the lingerie campaign."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.campaign_urls,
        additional_instruction="Look for explicit participation statements or visuals."
    )

    # Campaign timing (2024)
    timing = evaluator.add_parallel(
        id="c4_campaign_timing",
        desc="Campaigns occurred in 2024",
        parent=campaign,
        critical=True
    )
    node = evaluator.add_leaf(
        id="c4_year_verification",
        desc="Confirm campaigns were in 2024",
        parent=timing,
        critical=True
    )
    claim = "The mother-daughter lingerie campaign occurred in 2024."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.campaign_urls,
        additional_instruction="Confirm the campaign timing falls within calendar year 2024."
    )
    node = evaluator.add_leaf(
        id="c4_campaign_timing_reference",
        desc="Provide URL confirming 2024 campaign timing",
        parent=timing,
        critical=True
    )
    claim = "The sources explicitly state the 2024 timing of the campaign."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.campaign_urls,
        additional_instruction="Look for explicit year mention."
    )

    # Italian lingerie brand origin
    origin = evaluator.add_parallel(
        id="c4_italian_lingerie_brand",
        desc="Lingerie brand is Italian",
        parent=campaign,
        critical=True
    )
    node = evaluator.add_leaf(
        id="c4_lingerie_brand_origin",
        desc="Confirm the lingerie brand is Italian",
        parent=origin,
        critical=True
    )
    claim = f"'{lingerie_brand}' is an Italian lingerie brand."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.brand_origin_urls or c.campaign_urls,
        additional_instruction="Source should explicitly indicate Italian origin."
    )
    node = evaluator.add_leaf(
        id="c4_lingerie_brand_reference",
        desc="Provide URL confirming Italian lingerie brand",
        parent=origin,
        critical=True
    )
    claim = f"The sources confirm that '{lingerie_brand}' is Italian."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=c.brand_origin_urls or c.campaign_urls,
        additional_instruction="Look for 'Italian' explicitly."
    )


# ------------------------- Main Evaluation --------------------------------- #
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_celebrities(),
        template_class=CelebritiesExtraction,
        extraction_name="celebrities_extraction"
    )

    # Build verification tree for each celebrity (create nodes even if data missing; verification will fail/skip accordingly)
    await verify_celebrity_1(evaluator, root, extracted.celebrity1 or Celebrity1Data())
    await verify_celebrity_2(evaluator, root, extracted.celebrity2 or Celebrity2Data())
    await verify_celebrity_3(evaluator, root, extracted.celebrity3 or Celebrity3Data())
    await verify_celebrity_4(evaluator, root, extracted.celebrity4 or Celebrity4Data())

    return evaluator.get_summary()