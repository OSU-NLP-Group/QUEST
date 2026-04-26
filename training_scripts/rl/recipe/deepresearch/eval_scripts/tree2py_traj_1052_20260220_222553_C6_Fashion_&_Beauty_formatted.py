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
TASK_ID = "rare_beauty_ulta_expansion_2026"
TASK_DESCRIPTION = (
    "Research Selena Gomez's beauty brand Rare Beauty and provide comprehensive information about its recent expansion "
    "into Ulta Beauty stores. Include launch details, Ulta-exclusive products for Feb 1, 2026 and March 1, 2026, "
    "cruelty-free and vegan certifications (Leaping Bunny and PETA), packaging sustainability practices, and optional "
    "signature blush line information. Provide supporting reference URLs from official or reputable sources for each section."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RetailLaunchInfo(BaseModel):
    launch_date: Optional[str] = None
    store_count_nationwide: Optional[str] = None
    sephora_us_presence: Optional[str] = None
    international_other_presence: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class FebExclusivesInfo(BaseModel):
    exclusive_1_name: Optional[str] = None  # Expecting "Selena's Most Loved 3-Piece Set"
    exclusive_2_name: Optional[str] = None  # Expecting "Selena's Lash & Brow Duo"
    reference_urls: List[str] = Field(default_factory=list)


class MarchExclusivesInfo(BaseModel):
    product_list: List[str] = Field(default_factory=list)  # Expect exactly three names
    product_descriptions: List[str] = Field(default_factory=list)  # Brief descriptions aligned by index
    first_eyeshadow_palette_note: Optional[str] = None  # If the claim is present in the answer
    reference_urls: List[str] = Field(default_factory=list)


class LeapingBunnyInfo(BaseModel):
    status: Optional[str] = None  # e.g., "Leaping Bunny certified"
    certified_since: Optional[str] = None  # date or timeframe
    significance: Optional[str] = None  # what the certification signifies
    reference_urls: List[str] = Field(default_factory=list)


class PETAInfo(BaseModel):
    vegan_status: Optional[str] = None  # e.g., "100% vegan"
    peta_status: Optional[str] = None  # e.g., "Listed by PETA Beauty Without Bunnies"
    peta_timing: Optional[str] = None  # date or timeframe if available
    peta_significance: Optional[str] = None  # meaning of PETA listing/certification
    reference_urls: List[str] = Field(default_factory=list)


class CertificationsInfo(BaseModel):
    leaping_bunny: LeapingBunnyInfo = Field(default_factory=LeapingBunnyInfo)
    peta: PETAInfo = Field(default_factory=PETAInfo)


class SustainabilityInfo(BaseModel):
    outer_box_materials: Optional[str] = None
    box_recyclability: Optional[str] = None
    ink_type: Optional[str] = None
    pcr_percentage_primary_packaging: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class ProductLineOptionalInfo(BaseModel):
    blush_formulation_types: Optional[str] = None  # e.g., "liquid and powder"
    soft_pinch_liquid_blush_shade_count: Optional[str] = None  # e.g., "13 shades" or a number string
    reference_urls: List[str] = Field(default_factory=list)


class RareBeautyResearchExtraction(BaseModel):
    retail_launch: RetailLaunchInfo = Field(default_factory=RetailLaunchInfo)
    feb_exclusives: FebExclusivesInfo = Field(default_factory=FebExclusivesInfo)
    march_exclusives: MarchExclusivesInfo = Field(default_factory=MarchExclusivesInfo)
    certifications: CertificationsInfo = Field(default_factory=CertificationsInfo)
    sustainability: SustainabilityInfo = Field(default_factory=SustainabilityInfo)
    product_line_optional: ProductLineOptionalInfo = Field(default_factory=ProductLineOptionalInfo)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_research() -> str:
    return (
        "Extract structured information from the answer about Rare Beauty's expansion into Ulta Beauty. "
        "Return JSON exactly matching the following fields:\n\n"
        "retail_launch:\n"
        "- launch_date: When Rare Beauty launched at Ulta Beauty (string; e.g., 'February 1, 2026').\n"
        "- store_count_nationwide: How many Ulta Beauty stores carried Rare Beauty at launch/expansion (string; allow ranges or approximations).\n"
        "- sephora_us_presence: Describe prior U.S. Sephora availability before the Ulta expansion (string).\n"
        "- international_other_presence: Describe any international/other retailer presence prior to Ulta (string).\n"
        "- reference_urls: Array of URLs supporting launch timing/scale and prior presence.\n\n"
        "feb_exclusives:\n"
        "- exclusive_1_name: Name of Feb 1, 2026 Ulta-exclusive product/set (string). Expected: \"Selena's Most Loved 3-Piece Set\".\n"
        "- exclusive_2_name: Name of Feb 1, 2026 Ulta-exclusive product/set (string). Expected: \"Selena's Lash & Brow Duo\".\n"
        "- reference_urls: Array of URLs supporting Feb 1, 2026 exclusives.\n\n"
        "march_exclusives:\n"
        "- product_list: Array of three product names scheduled to launch exclusively at Ulta on March 1, 2026.\n"
        "- product_descriptions: Array of brief descriptions corresponding to product_list indices.\n"
        "- first_eyeshadow_palette_note: If the answer claims one product is Rare Beauty’s first eyeshadow palette, include that note (string); otherwise null.\n"
        "- reference_urls: Array of URLs supporting March 1, 2026 exclusives and product details.\n\n"
        "certifications:\n"
        "- leaping_bunny:\n"
        "  - status: Whether Rare Beauty is Leaping Bunny certified (string).\n"
        "  - certified_since: When certification was obtained (string).\n"
        "  - significance: What Leaping Bunny certification signifies (string).\n"
        "  - reference_urls: Array of URLs (e.g., leapingbunny.org, rarebeauty.com) supporting status and timing.\n"
        "- peta:\n"
        "  - vegan_status: Whether Rare Beauty is vegan (string; e.g., '100% vegan').\n"
        "  - peta_status: Whether Rare Beauty is listed/certified by PETA’s Beauty Without Bunnies program (string).\n"
        "  - peta_timing: When listing/certification timing was obtained if available (string).\n"
        "  - peta_significance: What PETA listing/certification signifies (string).\n"
        "  - reference_urls: Array of URLs (e.g., peta.org, rarebeauty.com) supporting status and timing.\n\n"
        "sustainability:\n"
        "- outer_box_materials: Materials used for outer boxes (string; e.g., FSC-certified).\n"
        "- box_recyclability: Whether packaging is recyclable and to what extent (string).\n"
        "- ink_type: Type of ink used (string; e.g., water-based).\n"
        "- pcr_percentage_primary_packaging: Percentage or quantified claim of PCR materials in primary packaging (string).\n"
        "- reference_urls: Array of URLs supporting the sustainability claims.\n\n"
        "product_line_optional:\n"
        "- blush_formulation_types: Blush formulation types (string; e.g., liquid and powder).\n"
        "- soft_pinch_liquid_blush_shade_count: Shade count for Soft Pinch Liquid Blush (string).\n"
        "- reference_urls: Array of URLs supporting blush formulation and shade count.\n\n"
        "Rules:\n"
        "- Extract only what is explicitly present in the answer; if missing, set fields to null or empty arrays as appropriate.\n"
        "- For URL arrays, include only valid URLs explicitly mentioned in the answer (plain or markdown links)."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_retail_launch(evaluator: Evaluator, parent_node, info: RetailLaunchInfo) -> None:
    retail_node = evaluator.add_parallel(
        id="Retail_Launch_Information",
        desc="Launch timing, scale, and prior retail presence before the Ulta expansion, with supporting citations.",
        parent=parent_node,
        critical=True,
    )

    # Launch Date
    launch_date_node = evaluator.add_leaf(
        id="Launch_Date",
        desc="States when Rare Beauty launched at Ulta Beauty.",
        parent=retail_node,
        critical=True,
    )
    launch_claim = f"Rare Beauty launched at Ulta Beauty on {info.launch_date}."
    await evaluator.verify(
        claim=launch_claim,
        node=launch_date_node,
        sources=info.reference_urls,
        additional_instruction="Verify the launch date using the provided sources (Ulta, Rare Beauty, or reputable publications). Accept phrasing like 'available starting <date>'.",
    )

    # Store Count Nationwide
    store_count_node = evaluator.add_leaf(
        id="Store_Count_Nationwide",
        desc="States how many Ulta Beauty stores nationwide carried Rare Beauty at launch/expansion.",
        parent=retail_node,
        critical=True,
    )
    store_claim = f"At launch/expansion, Rare Beauty was made available in {info.store_count_nationwide} Ulta Beauty stores nationwide."
    await evaluator.verify(
        claim=store_claim,
        node=store_count_node,
        sources=info.reference_urls,
        additional_instruction="Verify that the stated store count (or approximation) is supported by the cited sources.",
    )

    # Previous Retail Presence
    prev_presence_node = evaluator.add_parallel(
        id="Previous_Retail_Presence_Before_Ulta",
        desc="Describes Rare Beauty’s retail presence prior to the Ulta expansion.",
        parent=retail_node,
        critical=True,
    )

    sephora_node = evaluator.add_leaf(
        id="Sephora_US_Presence",
        desc="Describes prior U.S. Sephora availability (as applicable).",
        parent=prev_presence_node,
        critical=True,
    )
    sephora_claim = (
        f"Prior to the Ulta expansion, Rare Beauty's U.S. retail presence included Sephora. Details: {info.sephora_us_presence}."
    )
    await evaluator.verify(
        claim=sephora_claim,
        node=sephora_node,
        sources=info.reference_urls,
        additional_instruction="Confirm via sources that Rare Beauty retailed at Sephora in the U.S. before expanding to Ulta.",
    )

    intl_other_node = evaluator.add_leaf(
        id="International_And_Other_Retailers",
        desc="Describes any international and/or other retailer presence prior to the Ulta expansion (as applicable).",
        parent=prev_presence_node,
        critical=True,
    )
    intl_claim = (
        f"Before Ulta expansion, Rare Beauty had international and/or other retailer presence: {info.international_other_presence}."
    )
    await evaluator.verify(
        claim=intl_claim,
        node=intl_other_node,
        sources=info.reference_urls,
        additional_instruction="Verify any mentioned international or other retailers (outside Sephora US) using the cited sources.",
    )

    # At least one supporting URL exists
    evaluator.add_custom_node(
        result=bool(info.reference_urls),
        id="Retail_Launch_Reference_URL",
        desc="Provides at least one valid URL from an official or reputable source supporting the retail launch details (launch timing/scale and prior presence).",
        parent=retail_node,
        critical=True,
    )


async def verify_feb_exclusives(evaluator: Evaluator, parent_node, info: FebExclusivesInfo) -> None:
    feb_node = evaluator.add_parallel(
        id="February_1_2026_Exclusives",
        desc="Identifies the specific products/sets that launched exclusively at Ulta on Feb 1, 2026.",
        parent=parent_node,
        critical=True,
    )

    feb_ex1_node = evaluator.add_leaf(
        id="Feb_Exclusive_1",
        desc="Identifies 'Selena's Most Loved 3-Piece Set' as an Ulta-exclusive for Feb 1, 2026.",
        parent=feb_node,
        critical=True,
    )
    ex1_name = info.exclusive_1_name or "Selena's Most Loved 3-Piece Set"
    ex1_claim = f"The product/set '{ex1_name}' launched exclusively at Ulta Beauty on February 1, 2026."
    await evaluator.verify(
        claim=ex1_claim,
        node=feb_ex1_node,
        sources=info.reference_urls,
        additional_instruction="Use the provided URLs (Ulta, Rare Beauty, or reputable publications) to confirm exclusivity and the Feb 1, 2026 date.",
    )

    feb_ex2_node = evaluator.add_leaf(
        id="Feb_Exclusive_2",
        desc="Identifies 'Selena's Lash & Brow Duo' as an Ulta-exclusive for Feb 1, 2026.",
        parent=feb_node,
        critical=True,
    )
    ex2_name = info.exclusive_2_name or "Selena's Lash & Brow Duo"
    ex2_claim = f"The product/set '{ex2_name}' launched exclusively at Ulta Beauty on February 1, 2026."
    await evaluator.verify(
        claim=ex2_claim,
        node=feb_ex2_node,
        sources=info.reference_urls,
        additional_instruction="Verify exclusivity and launch date via the provided URLs.",
    )

    evaluator.add_custom_node(
        result=bool(info.reference_urls),
        id="February_Exclusives_Reference_URL",
        desc="Provides at least one valid URL supporting the Feb 1, 2026 exclusives.",
        parent=feb_node,
        critical=True,
    )


async def verify_march_exclusives(evaluator: Evaluator, parent_node, info: MarchExclusivesInfo) -> None:
    march_node = evaluator.add_parallel(
        id="March_1_2026_Exclusives",
        desc="Identifies and describes the new products scheduled to launch exclusively at Ulta on March 1, 2026, with citations.",
        parent=parent_node,
        critical=True,
    )

    march_list_node = evaluator.add_leaf(
        id="March_Exclusive_Product_List",
        desc="Names and briefly describes all three March 1, 2026 Ulta-exclusive products (three distinct products are identified), including noting if one is Rare Beauty’s first eyeshadow palette.",
        parent=march_node,
        critical=True,
    )
    names_str = ", ".join(info.product_list) if info.product_list else "three products"
    descs_str = "; ".join(info.product_descriptions) if info.product_descriptions else ""
    palette_note = (
        f" Note: {info.first_eyeshadow_palette_note}."
        if info.first_eyeshadow_palette_note and info.first_eyeshadow_palette_note.strip()
        else ""
    )
    march_claim = (
        f"On March 1, 2026, Rare Beauty launched the following three products exclusively at Ulta Beauty: {names_str}. "
        f"Brief descriptions: {descs_str}.{palette_note}"
    )
    await evaluator.verify(
        claim=march_claim,
        node=march_list_node,
        sources=info.reference_urls,
        additional_instruction="Confirm that exactly three distinct products are named and scheduled for Ulta-exclusive launch on March 1, 2026; verify descriptions and any 'first eyeshadow palette' claim if present.",
    )

    evaluator.add_custom_node(
        result=bool(info.reference_urls),
        id="March_Exclusives_Reference_URL",
        desc="Provides at least one valid URL supporting the March 1, 2026 exclusives (including the product list).",
        parent=march_node,
        critical=True,
    )


async def verify_certifications(evaluator: Evaluator, parent_node, info: CertificationsInfo) -> None:
    certs_node = evaluator.add_parallel(
        id="Certifications_and_Ethics",
        desc="Cruelty-free and vegan certifications, including timing and meaning, with citations.",
        parent=parent_node,
        critical=True,
    )

    # Leaping Bunny
    lb_node = evaluator.add_parallel(
        id="Leaping_Bunny_Certification",
        desc="Verifies Leaping Bunny certification status, timing, and what it signifies, with citations.",
        parent=certs_node,
        critical=True,
    )

    lb_status_leaf = evaluator.add_leaf(
        id="Leaping_Bunny_Status",
        desc="States whether Rare Beauty is Leaping Bunny certified.",
        parent=lb_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Rare Beauty Leaping Bunny certification status: {info.leaping_bunny.status}.",
        node=lb_status_leaf,
        sources=info.leaping_bunny.reference_urls,
        additional_instruction="Verify via Leaping Bunny or official brand sources whether Rare Beauty is Leaping Bunny certified.",
    )

    lb_since_leaf = evaluator.add_leaf(
        id="Leaping_Bunny_Certified_Since",
        desc="States when the Leaping Bunny certification was obtained (timing).",
        parent=lb_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Rare Beauty obtained Leaping Bunny certification in {info.leaping_bunny.certified_since}.",
        node=lb_since_leaf,
        sources=info.leaping_bunny.reference_urls,
        additional_instruction="Verify the timing (date or timeframe) from Leaping Bunny or official sources.",
    )

    lb_sig_leaf = evaluator.add_leaf(
        id="Leaping_Bunny_Significance",
        desc="Explains what Leaping Bunny certification signifies.",
        parent=lb_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Leaping Bunny certification signifies: {info.leaping_bunny.significance}.",
        node=lb_sig_leaf,
        sources=info.leaping_bunny.reference_urls,
        additional_instruction="Confirm the meaning (e.g., no animal testing across the supply chain) via official program pages.",
    )

    evaluator.add_custom_node(
        result=bool(info.leaping_bunny.reference_urls),
        id="Leaping_Bunny_Reference_URL",
        desc="Provides at least one valid URL from Leaping Bunny or another official/reputable source supporting the Leaping Bunny status and timing.",
        parent=lb_node,
        critical=True,
    )

    # PETA / Vegan
    peta_node = evaluator.add_parallel(
        id="Vegan_and_PETA_Certification",
        desc="Verifies vegan status and PETA certification, including timing and meaning, with citations.",
        parent=certs_node,
        critical=True,
    )

    vegan_status_leaf = evaluator.add_leaf(
        id="Vegan_Status",
        desc="States whether Rare Beauty is vegan (e.g., 100% vegan, if claimed).",
        parent=peta_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Rare Beauty vegan status: {info.peta.vegan_status}.",
        node=vegan_status_leaf,
        sources=info.peta.reference_urls,
        additional_instruction="Verify vegan status via official brand pages or PETA resources.",
    )

    peta_status_leaf = evaluator.add_leaf(
        id="PETA_Beauty_Without_Bunnies_Status",
        desc="States whether Rare Beauty is certified/listed by PETA’s Beauty Without Bunnies program (as applicable).",
        parent=peta_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Rare Beauty PETA Beauty Without Bunnies status: {info.peta.peta_status}.",
        node=peta_status_leaf,
        sources=info.peta.reference_urls,
        additional_instruction="Verify listing/certification via PETA’s Beauty Without Bunnies page or other official sources.",
    )

    peta_timing_leaf = evaluator.add_leaf(
        id="PETA_Timing",
        desc="States when the PETA certification/listing (or vegan certification claim timing) was obtained, if a credible source provides a date/timeframe.",
        parent=peta_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"PETA certification/listing timing for Rare Beauty: {info.peta.peta_timing}.",
        node=peta_timing_leaf,
        sources=info.peta.reference_urls,
        additional_instruction="Confirm any available date/timeframe for PETA listing/certification.",
    )

    peta_sig_leaf = evaluator.add_leaf(
        id="PETA_Vegan_Significance",
        desc="Explains what the stated PETA certification/listing and/or vegan certification signifies.",
        parent=peta_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"PETA certification/listing signifies: {info.peta.peta_significance}.",
        node=peta_sig_leaf,
        sources=info.peta.reference_urls,
        additional_instruction="Confirm the meaning of PETA listing/certification via official PETA resources.",
    )

    evaluator.add_custom_node(
        result=bool(info.peta.reference_urls),
        id="Vegan_PETA_Reference_URL",
        desc="Provides at least one valid URL supporting the vegan/PETA claims (status and timing, if stated).",
        parent=peta_node,
        critical=True,
    )


async def verify_sustainability(evaluator: Evaluator, parent_node, info: SustainabilityInfo) -> None:
    sust_node = evaluator.add_parallel(
        id="Sustainability_Practices",
        desc="Packaging sustainability practices (outer box materials, recyclability, ink type, PCR %), with citations.",
        parent=parent_node,
        critical=True,
    )

    outer_box_leaf = evaluator.add_leaf(
        id="Outer_Box_Materials",
        desc="States the materials used for outer boxes (e.g., FSC-certified materials, if claimed).",
        parent=sust_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Rare Beauty outer box materials: {info.outer_box_materials}.",
        node=outer_box_leaf,
        sources=info.reference_urls,
        additional_instruction="Verify materials (e.g., FSC-certified paper) via official brand sustainability pages or packaging documentation.",
    )

    recyclability_leaf = evaluator.add_leaf(
        id="Box_Recyclability",
        desc="States whether the outer box/packaging is recyclable (and to what extent, if stated).",
        parent=sust_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Rare Beauty packaging recyclability: {info.box_recyclability}.",
        node=recyclability_leaf,
        sources=info.reference_urls,
        additional_instruction="Confirm recyclability statements via official sources.",
    )

    ink_type_leaf = evaluator.add_leaf(
        id="Ink_Type",
        desc="States the type of ink used for printing on packaging (e.g., water-based, if claimed).",
        parent=sust_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Rare Beauty packaging ink type: {info.ink_type}.",
        node=ink_type_leaf,
        sources=info.reference_urls,
        additional_instruction="Verify ink type details via official sustainability or packaging information.",
    )

    pcr_leaf = evaluator.add_leaf(
        id="PCR_Percentage_Primary_Packaging",
        desc="States the percentage (or quantified claim) of post-consumer recycled (PCR) materials used in primary packaging.",
        parent=sust_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Rare Beauty primary packaging PCR usage: {info.pcr_percentage_primary_packaging}.",
        node=pcr_leaf,
        sources=info.reference_urls,
        additional_instruction="Confirm PCR material percentage or quantified claim via official sources.",
    )

    evaluator.add_custom_node(
        result=bool(info.reference_urls),
        id="Sustainability_Reference_URL",
        desc="Provides at least one valid URL supporting the sustainability/packaging claims.",
        parent=sust_node,
        critical=True,
    )


async def verify_product_line_optional(evaluator: Evaluator, parent_node, info: ProductLineOptionalInfo) -> None:
    opt_node = evaluator.add_parallel(
        id="Product_Line_Information_Optional",
        desc="Optional: signature blush line details (formulation types and shade count), with citations.",
        parent=parent_node,
        critical=False,
    )

    formulation_leaf = evaluator.add_leaf(
        id="Blush_Formulation_Types",
        desc="States the blush formulation types offered (e.g., liquid and powder, if claimed).",
        parent=opt_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"Rare Beauty blush formulation types: {info.blush_formulation_types}.",
        node=formulation_leaf,
        sources=info.reference_urls,
        additional_instruction="Confirm blush formulation types via official product pages or reputable beauty publications.",
    )

    shade_count_leaf = evaluator.add_leaf(
        id="Soft_Pinch_Liquid_Blush_Shade_Count",
        desc="States the shade count (or minimum shade count claim) for Soft Pinch Liquid Blush.",
        parent=opt_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"Soft Pinch Liquid Blush shade count: {info.soft_pinch_liquid_blush_shade_count}.",
        node=shade_count_leaf,
        sources=info.reference_urls,
        additional_instruction="Verify shade count via official product pages.",
    )

    evaluator.add_custom_node(
        result=bool(info.reference_urls),
        id="Blush_Line_Reference_URL",
        desc="Provides at least one valid URL supporting the blush formulation and shade-count claims.",
        parent=opt_node,
        critical=False,
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
    Evaluate an answer for the Rare Beauty Ulta expansion research task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel sections with critical gating per section
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

    # Extract structured info once
    extraction = await evaluator.extract(
        prompt=prompt_extract_research(),
        template_class=RareBeautyResearchExtraction,
        extraction_name="rare_beauty_ulta_expansion_extraction",
    )

    # Build top-level sections under root (root is non-critical to allow optional content)
    # 1) Retail launch
    await verify_retail_launch(evaluator, root, extraction.retail_launch)

    # 2) Ulta-exclusive products
    ulta_exclusives_node = evaluator.add_parallel(
        id="Ulta_Exclusive_Products",
        desc="Ulta-exclusive products for Feb 1, 2026 and March 1, 2026, with citations.",
        parent=root,
        critical=True,
    )
    await verify_feb_exclusives(evaluator, ulta_exclusives_node, extraction.feb_exclusives)
    await verify_march_exclusives(evaluator, ulta_exclusives_node, extraction.march_exclusives)

    # 3) Certifications and ethics
    await verify_certifications(evaluator, root, extraction.certifications)

    # 4) Sustainability practices
    await verify_sustainability(evaluator, root, extraction.sustainability)

    # 5) Optional product line info
    await verify_product_line_optional(evaluator, root, extraction.product_line_optional)

    # Return the summary with verification tree
    return evaluator.get_summary()