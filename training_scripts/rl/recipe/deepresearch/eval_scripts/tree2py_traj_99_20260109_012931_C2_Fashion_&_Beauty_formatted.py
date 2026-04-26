import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "designer_skincare_2022_lvmh"
TASK_DESCRIPTION = (
    "A fashion designer founded an eponymous fashion house between 2000 and 2005 (inclusive) as a joint venture "
    "partnership with a major luxury group, with the first collection shown in Paris. This same designer later "
    "launched a skincare line in September 2022 in partnership with LVMH's Beauty Division. The skincare line "
    "consists of exactly three products, and each product is made from at least 99% natural-origin ingredients and "
    "is vegan, cruelty-free, and refillable. What is the full name of this fashion designer?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FashionHouseInfo(BaseModel):
    name: Optional[str] = None
    founding_year: Optional[str] = None
    joint_venture_group: Optional[str] = None
    first_collection_location: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SkincareLineInfo(BaseModel):
    line_name: Optional[str] = None
    launch_month_year: Optional[str] = None
    partner: Optional[str] = None
    number_of_products: Optional[str] = None
    natural_origin_claim: Optional[str] = None
    vegan: Optional[str] = None
    cruelty_free: Optional[str] = None
    refillable: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DesignerExtraction(BaseModel):
    designer_full_name: Optional[str] = None
    fashion_house: Optional[FashionHouseInfo] = None
    skincare_line: Optional[SkincareLineInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_designer_and_brands() -> str:
    return """
    Extract the key information explicitly stated in the answer about the designer, the eponymous fashion house, and the skincare line.

    Return a JSON object with:
    - designer_full_name: The full name of the designer (e.g., "Stella McCartney"). If only a brand name is given without a person's name, return null.
    - fashion_house: An object with:
        * name: The name of the fashion house/brand (e.g., "Stella McCartney").
        * founding_year: The founding year of the eponymous house if stated (e.g., "2001").
        * joint_venture_group: The luxury group partner for the launch, if stated (e.g., "Gucci Group" or "Kering" or "LVMH").
        * first_collection_location: The city where the first collection was shown, if stated (e.g., "Paris").
        * sources: All URLs cited in the answer that support any of these fashion house details. Include only valid URLs mentioned in the answer. If none are provided, return an empty list.
    - skincare_line: An object with:
        * line_name: The skincare line name (e.g., "STELLA by Stella McCartney"), if stated.
        * launch_month_year: The stated launch timing (e.g., "September 2022").
        * partner: The partner organization for the skincare line (e.g., "LVMH Beauty Division" or "LVMH Beauty"), if stated.
        * number_of_products: The number of products in the line if stated (e.g., "3").
        * natural_origin_claim: The stated claim about natural-origin percentage (e.g., "at least 99%"), if stated.
        * vegan: If the answer explicitly claims the products are vegan, set to "yes"; if not stated, null.
        * cruelty_free: If the answer explicitly claims the products are cruelty-free, set to "yes"; if not stated, null.
        * refillable: If the answer explicitly claims the products are refillable, set to "yes"; if not stated, null.
        * sources: All URLs cited in the answer that support any of these skincare details. Include only valid URLs mentioned in the answer. If none are provided, return an empty list.

    Rules:
    - Extract exactly what the answer states; do not invent facts.
    - For any missing field, return null (or an empty array for sources).
    - URLs can appear as plain links or markdown links; extract the actual URL strings.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def has_full_name(name: Optional[str]) -> bool:
    if not name:
        return False
    # A simple heuristic for "full name": at least two tokens with alphabetic characters
    tokens = [t for t in name.strip().split() if any(ch.isalpha() for ch in t)]
    return len(tokens) >= 2


def preferred_label_for_skincare(designer_name: Optional[str], line_name: Optional[str]) -> str:
    if line_name and line_name.strip():
        return f"the skincare line '{line_name.strip()}'"
    if designer_name and designer_name.strip():
        return f"{designer_name.strip()}'s skincare line"
    return "the skincare line"


def safe(val: Optional[str], fallback: str = "") -> str:
    return val.strip() if isinstance(val, str) else fallback


def sources_or_none(urls: Optional[List[str]]) -> List[str] | None:
    if not urls:
        return None
    # filter obvious garbage
    cleaned = [u for u in urls if isinstance(u, str) and len(u) >= 5]
    return cleaned if cleaned else None


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: DesignerExtraction) -> None:
    # Root-level critical node for designer identification (parallel aggregation of sub-criteria)
    designer_node = evaluator.add_parallel(
        id="Designer_Identification",
        desc="Identify the fashion designer who satisfies all stated fashion house and skincare line constraints",
        parent=evaluator.root,
        critical=True
    )

    # 1) Answer provides full name (existence/format check)
    full_name_ok = has_full_name(extracted.designer_full_name)
    evaluator.add_custom_node(
        result=full_name_ok,
        id="Answer_Provides_Full_Name",
        desc="Response provides the designer’s full name (not a brand name only)",
        parent=designer_node,
        critical=True
    )

    # Prepare convenience variables
    dname = safe(extracted.designer_full_name)
    fh = extracted.fashion_house or FashionHouseInfo()
    sk = extracted.skincare_line or SkincareLineInfo()

    fh_sources = sources_or_none(fh.sources)
    sk_sources = sources_or_none(sk.sources)

    # 2) Fashion House Requirements (critical parallel group)
    fh_node = evaluator.add_parallel(
        id="Fashion_House_Requirements",
        desc="Designer’s eponymous fashion house satisfies all stated founding and debut constraints",
        parent=designer_node,
        critical=True
    )

    # 2.a) Founded + Eponymous + Year in [2000, 2005] – split into two leaf checks under an internal critical node
    fh_found_epo_node = evaluator.add_parallel(
        id="Fashion_House_Founded_Eponymous_2000_2005",
        desc="Designer founded an eponymous fashion house between 2000 and 2005 inclusive",
        parent=fh_node,
        critical=True
    )

    # 2.a.i) Eponymous check
    leaf_fh_eponymous = evaluator.add_leaf(
        id="Fashion_House_Eponymous",
        desc="The fashion house is eponymous to the designer (brand name includes the designer’s name)",
        parent=fh_found_epo_node,
        critical=True
    )
    claim_eponymous = (
        f"The fashion house '{safe(fh.name, 'the fashion house')}' is eponymous to the designer '{dname}', "
        f"meaning the brand name includes the designer's name."
    )
    await evaluator.verify(
        claim=claim_eponymous,
        node=leaf_fh_eponymous,
        sources=fh_sources,
        additional_instruction=(
            "Support this if the brand/house name clearly contains the designer's name. "
            "Accept reasonable variants like 'Maison <Name>' or similar forms that still incorporate the designer's name."
        )
    )

    # 2.a.ii) Founding year in [2000, 2005] inclusive
    leaf_fh_year = evaluator.add_leaf(
        id="Fashion_House_Founded_Year_In_Range",
        desc="The eponymous fashion house was founded between 2000 and 2005 (inclusive)",
        parent=fh_found_epo_node,
        critical=True
    )
    if fh.founding_year and fh.founding_year.isdigit():
        claim_year = (
            f"The eponymous fashion house was founded in {fh.founding_year}, "
            f"which is between 2000 and 2005 inclusive."
        )
    else:
        claim_year = (
            "The eponymous fashion house was founded between 2000 and 2005 inclusive."
        )
    await evaluator.verify(
        claim=claim_year,
        node=leaf_fh_year,
        sources=fh_sources,
        additional_instruction="Verify that the page states a founding year within the 2000–2005 range."
    )

    # 2.b) Joint venture with a major luxury group
    leaf_fh_jv = evaluator.add_leaf(
        id="Fashion_House_Joint_Venture_Major_Luxury_Group",
        desc="Fashion house was launched as a joint venture partnership with a major luxury group",
        parent=fh_node,
        critical=True
    )
    if fh.joint_venture_group:
        claim_jv = (
            f"The fashion house was launched as a joint venture partnership with {fh.joint_venture_group}, "
            f"a major luxury group."
        )
    else:
        claim_jv = (
            "The fashion house was launched as a joint venture partnership with a major luxury group."
        )
    await evaluator.verify(
        claim=claim_jv,
        node=leaf_fh_jv,
        sources=fh_sources,
        additional_instruction=(
            "Accept this if the page states a joint venture with a well-known luxury group such as LVMH, "
            "Kering (formerly Gucci Group), Richemont, or similar top-tier groups."
        )
    )

    # 2.c) First collection shown in Paris
    leaf_fh_paris = evaluator.add_leaf(
        id="First_Collection_Shown_In_Paris",
        desc="Designer’s first collection for the fashion house was shown in Paris",
        parent=fh_node,
        critical=True
    )
    claim_paris = (
        f"The first collection for the fashion house '{safe(fh.name, 'the fashion house')}' "
        f"was shown in Paris."
    )
    await evaluator.verify(
        claim=claim_paris,
        node=leaf_fh_paris,
        sources=fh_sources,
        additional_instruction=(
            "Match phrases like 'first show in Paris', 'debut collection in Paris', or equivalent wording. "
            "Minor phrasing differences are acceptable."
        )
    )

    # 3) Skincare Line Requirements (critical parallel group)
    sk_node = evaluator.add_parallel(
        id="Skincare_Line_Requirements",
        desc="Designer’s skincare line satisfies all stated launch timing, partnership, and product constraints",
        parent=designer_node,
        critical=True
    )

    line_label = preferred_label_for_skincare(dname, sk.line_name)

    # 3.a) Skincare debuted in September 2022
    leaf_sk_sept2022 = evaluator.add_leaf(
        id="Skincare_Debut_September_2022",
        desc="Skincare line debuted in September 2022",
        parent=sk_node,
        critical=True
    )
    claim_sept2022 = f"The designer {dname} launched {line_label} in September 2022."
    await evaluator.verify(
        claim=claim_sept2022,
        node=leaf_sk_sept2022,
        sources=sk_sources,
        additional_instruction="The page should explicitly indicate a launch in September 2022."
    )

    # 3.b) Partner is LVMH Beauty Division
    leaf_sk_lvmh = evaluator.add_leaf(
        id="Skincare_Partner_LVMH_Beauty_Division",
        desc="Skincare line was developed in partnership with LVMH's Beauty Division",
        parent=sk_node,
        critical=True
    )
    claim_lvmh = f"{line_label.capitalize()} was developed in partnership with LVMH's Beauty Division (also called LVMH Beauty)."
    await evaluator.verify(
        claim=claim_lvmh,
        node=leaf_sk_lvmh,
        sources=sk_sources,
        additional_instruction=(
            "Accept 'LVMH Beauty Division', 'LVMH Beauty', or references to LVMH's Perfumes & Cosmetics division "
            "as the partner for the skincare line."
        )
    )

    # 3.c) Exactly three products
    leaf_sk_three = evaluator.add_leaf(
        id="Skincare_Exactly_Three_Products",
        desc="Skincare line consists of exactly three products",
        parent=sk_node,
        critical=True
    )
    claim_three = f"{line_label.capitalize()} consists of exactly three products."
    await evaluator.verify(
        claim=claim_three,
        node=leaf_sk_three,
        sources=sk_sources,
        additional_instruction="Look for explicit mention that the range has exactly three products or a 'trio'."
    )

    # 3.d) At least 99% natural-origin ingredients for each product
    leaf_sk_99 = evaluator.add_leaf(
        id="Products_At_Least_99pct_Natural_Origin",
        desc="Each product is made from at least 99% natural-origin ingredients",
        parent=sk_node,
        critical=True
    )
    claim_99 = f"Each product in {line_label} is made from at least 99% natural-origin ingredients."
    await evaluator.verify(
        claim=claim_99,
        node=leaf_sk_99,
        sources=sk_sources,
        additional_instruction=(
            "Support this if the page states 99% natural-origin (or higher) per product; "
            "allow phrasing like 'at least 99%' or '99% natural-origin'."
        )
    )

    # 3.e) Vegan, cruelty-free, and refillable
    leaf_sk_attrs = evaluator.add_leaf(
        id="Products_Vegan_CrueltyFree_Refillable",
        desc="All products are vegan, cruelty-free, and refillable",
        parent=sk_node,
        critical=True
    )
    claim_attrs = f"All products in {line_label} are vegan, cruelty-free, and refillable."
    await evaluator.verify(
        claim=claim_attrs,
        node=leaf_sk_attrs,
        sources=sk_sources,
        additional_instruction=(
            "The page should explicitly state that products are vegan, cruelty-free, and refillable. "
            "Equivalent phrases like 'no animal-derived ingredients' or 'not tested on animals' qualify for vegan/cruelty-free."
        )
    )

    # Add some custom info for debugging
    evaluator.add_custom_info(
        info={
            "extracted_designer_full_name": dname,
            "extracted_fashion_house": fh.dict(),
            "extracted_skincare_line": sk.dict(),
        },
        info_type="extraction_debug",
        info_name="extraction_debug_info"
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
    Evaluate an answer for the designer identification + skincare verification task.
    """
    # Initialize evaluator (root is non-critical; we add our own critical child as per rubric)
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
        default_model=model,
    )

    # Extract structured data from the answer text
    extracted = await evaluator.extract(
        prompt=prompt_extract_designer_and_brands(),
        template_class=DesignerExtraction,
        extraction_name="designer_and_brands_extraction",
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, extracted)

    # Return evaluator's standardized summary
    return evaluator.get_summary()