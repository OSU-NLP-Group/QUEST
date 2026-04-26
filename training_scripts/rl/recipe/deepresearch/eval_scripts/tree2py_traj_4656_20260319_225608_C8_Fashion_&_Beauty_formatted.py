import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "olivia_culpo_late_2025_collections"
TASK_DESCRIPTION = (
    "A fashion blogger is creating a comprehensive shopping guide for celebrity-designed fashion collaborations from late 2025. "
    "They are specifically focusing on model and entrepreneur Olivia Culpo's two major fashion partnerships that launched in October and December 2025.\n\n"
    "Research and document the following information for both collaborations:\n\n"
    "1. For the cashmere collection (launched in October 2025):\n"
    "- Partner brand name\n- Total number of pieces in the collection\n- Material composition\n- Price range (minimum and maximum prices)\n"
    "- Exact launch date\n- Available size range\n\n"
    "2. For the NFL-themed collection (launched in December 2025):\n"
    "- Partner brand name\n- Total number of pieces in the collection\n- Which NFL team the collection features\n- The three specific item types included\n"
    "- Price of the bomber jacket\n- Exact launch date\n- Official team colors featured\n\n"
    "3. Explain the personal connection between Olivia Culpo and the NFL team featured in the December collection by providing:\n"
    "- The name of her husband (married June 29, 2024)\n- His NFL team\n- His position\n- His jersey number\n\n"
    "4. Provide care instructions for the 100% cashmere items, including:\n"
    "- Proper washing method\n- Recommended detergent type\n- Water temperature guidelines\n- Drying method\n\n"
    "5. Document retail availability for both collections (where customers can purchase these items).\n\n"
    "For all factual claims, provide reference URLs from your research."
)

# Expected facts (ground truth targets for verification)
CASHMERE_EXPECTED = {
    "partner_brand": "NAADAM",
    "piece_count": "14",
    "material": "100% cashmere",
    "price_min": "$138",
    "price_max": "$448",
    "launch_date": "October 21, 2025",
    "size_range_phrase": "women’s XXS–3X and men’s up to XXL",
}

NFL_EXPECTED = {
    "partner_brand": "Abercrombie & Fitch",
    "piece_count": "3",
    "featured_team": "San Francisco 49ers",
    "item_types_phrase": "bomber jacket, tank top, and a crewneck/sweatshirt",
    "bomber_price": "$220",
    "launch_date": "December 11, 2025",
    "team_colors": ["red", "gold"],
}

PERSONAL_EXPECTED = {
    "husband_name": "Christian McCaffrey",
    "husband_team": "San Francisco 49ers",
    "husband_position": "running back",
    "husband_jersey_number": "23",
}

RETAIL_EXPECTED = {
    "cashmere_availability_phrase": "naadam.co and select retailers",
    "nfl_availability_phrase": "Abercrombie & Fitch stores and online",
}


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class CashmereSection(BaseModel):
    partner_brand: Optional[str] = None
    piece_count: Optional[str] = None
    material: Optional[str] = None
    price_min: Optional[str] = None
    price_max: Optional[str] = None
    launch_date: Optional[str] = None
    size_range: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class NFLSection(BaseModel):
    partner_brand: Optional[str] = None
    piece_count: Optional[str] = None
    featured_team: Optional[str] = None
    item_types: List[str] = Field(default_factory=list)
    bomber_price: Optional[str] = None
    launch_date: Optional[str] = None
    team_colors: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class PersonalConnectionSection(BaseModel):
    husband_name: Optional[str] = None
    husband_team: Optional[str] = None
    husband_position: Optional[str] = None
    husband_jersey_number: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CareInstructionsSection(BaseModel):
    washing_method: Optional[str] = None
    detergent_type: Optional[str] = None
    water_temperature: Optional[str] = None
    drying_method: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class RetailAvailabilitySection(BaseModel):
    cashmere_availability: Optional[str] = None
    nfl_availability: Optional[str] = None
    cashmere_sources: List[str] = Field(default_factory=list)
    nfl_sources: List[str] = Field(default_factory=list)


class CitationsSection(BaseModel):
    cashmere: List[str] = Field(default_factory=list)
    nfl: List[str] = Field(default_factory=list)
    personal: List[str] = Field(default_factory=list)
    care: List[str] = Field(default_factory=list)
    retail: List[str] = Field(default_factory=list)


class ShoppingGuideExtraction(BaseModel):
    cashmere: Optional[CashmereSection] = None
    nfl: Optional[NFLSection] = None
    personal: Optional[PersonalConnectionSection] = None
    care: Optional[CareInstructionsSection] = None
    retail: Optional[RetailAvailabilitySection] = None
    citations: Optional[CitationsSection] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_shopping_guide() -> str:
    return """
Extract the following fields exactly as they appear in the answer. If a field is not provided in the answer, return null (or empty list for arrays). Also extract the reference URLs grouped by section.

1) Cashmere collection (October 2025) — object 'cashmere':
- partner_brand: string (e.g., "NAADAM")
- piece_count: string (e.g., "14")
- material: string (e.g., "100% cashmere")
- price_min: string (minimal price, include currency symbol if shown; e.g., "$138")
- price_max: string (maximal price; e.g., "$448")
- launch_date: string (e.g., "October 21, 2025")
- size_range: string (e.g., "women’s XXS–3X and men’s up to XXL")
- sources: array of URLs explicitly cited for the above facts for this collection

2) NFL-themed collection (December 2025) — object 'nfl':
- partner_brand: string (e.g., "Abercrombie & Fitch")
- piece_count: string (e.g., "3")
- featured_team: string (e.g., "San Francisco 49ers")
- item_types: array of strings listing the three item types (e.g., ["bomber jacket", "tank top", "crewneck"])
- bomber_price: string (e.g., "$220")
- launch_date: string (e.g., "December 11, 2025")
- team_colors: array of strings (e.g., ["red", "gold"])
- sources: array of URLs explicitly cited for the above facts for this collection

3) Personal connection — object 'personal':
- husband_name: string (e.g., "Christian McCaffrey")
- husband_team: string (e.g., "San Francisco 49ers")
- husband_position: string (e.g., "running back")
- husband_jersey_number: string (e.g., "23")
- sources: array of URLs for the above personal-connection facts

4) Cashmere care instructions — object 'care':
- washing_method: string (e.g., "hand wash" or "cold/delicate machine wash")
- detergent_type: string (e.g., "gentle/mild/wool-friendly detergent")
- water_temperature: string (e.g., "cold or under 30°C/86°F")
- drying_method: string (e.g., "air dry; do not tumble dry")
- sources: array of URLs for these care instructions

5) Retail availability — object 'retail':
- cashmere_availability: string describing where to buy the cashmere collection
- nfl_availability: string describing where to buy the NFL collection
- cashmere_sources: array of URLs for cashmere retail availability
- nfl_sources: array of URLs for NFL retail availability

6) Citations — object 'citations':
- cashmere: array of URLs cited anywhere in the answer to support the cashmere collection facts
- nfl: array of URLs cited anywhere in the answer to support the NFL collection facts
- personal: array of URLs cited for the personal-connection facts
- care: array of URLs cited for the cashmere care instructions
- retail: array of URLs cited for retail availability

Reminder:
- Return only data explicitly present in the answer.
- Extract actual URLs (full URLs). If a source is referenced without a URL, omit it.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(values: Optional[List[str]]) -> List[str]:
    return values or []


def _any_sources(*candidates: Optional[List[str]]) -> List[str]:
    for c in candidates:
        if c and len(c) > 0:
            return c
    return []


# --------------------------------------------------------------------------- #
# Verification routines                                                       #
# --------------------------------------------------------------------------- #
async def verify_cashmere_section(evaluator: Evaluator, parent_node, ext: ShoppingGuideExtraction) -> None:
    cash = ext.cashmere or CashmereSection()
    node = evaluator.add_parallel(
        id="cashmere_collection_oct_2025",
        desc="Cashmere collection (October 2025): required collaboration details are provided correctly",
        parent=parent_node,
        critical=True,
    )

    # Gate: at least one URL present for this section
    cash_sources_node = evaluator.add_custom_node(
        result=bool(cash.sources),
        id="cashmere_sources_present",
        desc="Cashmere section includes at least one reference URL",
        parent=node,
        critical=True,
    )

    # Partner brand
    provided_partner = evaluator.add_custom_node(
        result=bool(cash.partner_brand),
        id="cashmere_partner_brand_provided",
        desc="Cashmere partner brand is provided in the answer",
        parent=node,
        critical=True,
    )
    # The answer explicitly states it
    stated_partner = evaluator.add_leaf(
        id="cashmere_partner_brand_stated",
        desc="Answer explicitly states: Partner brand is NAADAM",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the partner brand for the October 2025 cashmere collection is NAADAM.",
        node=stated_partner,
        additional_instruction="Check the answer text only. Allow minor casing or punctuation differences for 'NAADAM'.",
        extra_prerequisites=[provided_partner],
    )
    # Supported by sources (JSON leaf mapping)
    partner_brand_leaf = evaluator.add_leaf(
        id="cashmere_partner_brand",
        desc="Partner brand name is NAADAM",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The partner brand for Olivia Culpo’s October 2025 cashmere collection is NAADAM.",
        node=partner_brand_leaf,
        sources=cash.sources,
        additional_instruction="Verify on the cited pages that NAADAM is the collaboration partner for the October 2025 cashmere collection.",
        extra_prerequisites=[cash_sources_node, provided_partner],
    )

    # Piece count
    provided_pieces = evaluator.add_custom_node(
        result=bool(cash.piece_count),
        id="cashmere_piece_count_provided",
        desc="Cashmere piece count is provided in the answer",
        parent=node,
        critical=True,
    )
    stated_pieces = evaluator.add_leaf(
        id="cashmere_piece_count_stated",
        desc="Answer explicitly states: Total pieces is 14",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the cashmere collection contains 14 pieces.",
        node=stated_pieces,
        additional_instruction="Check the answer text only. Allow '14' whether written as a numeral or word; treat them as equivalent.",
        extra_prerequisites=[provided_pieces],
    )
    piece_leaf = evaluator.add_leaf(
        id="cashmere_piece_count",
        desc="Total number of pieces is 14",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The October 2025 cashmere collection contains 14 pieces.",
        node=piece_leaf,
        sources=cash.sources,
        additional_instruction="Confirm that the cited page(s) explicitly indicate a 14-piece collection.",
        extra_prerequisites=[cash_sources_node, provided_pieces],
    )

    # Material
    provided_material = evaluator.add_custom_node(
        result=bool(cash.material),
        id="cashmere_material_provided",
        desc="Cashmere material composition is provided in the answer",
        parent=node,
        critical=True,
    )
    stated_material = evaluator.add_leaf(
        id="cashmere_material_stated",
        desc="Answer explicitly states: Material composition is 100% cashmere",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the material composition is 100% cashmere.",
        node=stated_material,
        additional_instruction="Check the answer text only. Allow minor formatting like '100 percent cashmere' vs '100% cashmere'.",
        extra_prerequisites=[provided_material],
    )
    material_leaf = evaluator.add_leaf(
        id="cashmere_material_composition",
        desc="Material composition is 100% cashmere",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The collection’s pieces are made of 100% cashmere.",
        node=material_leaf,
        sources=cash.sources,
        additional_instruction="Confirm that the cited page(s) indicate 100% cashmere composition.",
        extra_prerequisites=[cash_sources_node, provided_material],
    )

    # Price range
    provided_prices = evaluator.add_custom_node(
        result=bool(cash.price_min) and bool(cash.price_max),
        id="cashmere_price_range_provided",
        desc="Cashmere price range (min and max) is provided in the answer",
        parent=node,
        critical=True,
    )
    stated_prices = evaluator.add_leaf(
        id="cashmere_price_range_stated",
        desc="Answer explicitly states: Price range is $138 to $448",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the cashmere collection price range is from $138 to $448.",
        node=stated_prices,
        additional_instruction="Check the answer text only. Accept minor formatting variants like $138.00 or an en dash.",
        extra_prerequisites=[provided_prices],
    )
    price_leaf = evaluator.add_leaf(
        id="cashmere_price_range",
        desc="Price range is $138 (min) to $448 (max)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The price range for the October 2025 cashmere collection is from $138 to $448.",
        node=price_leaf,
        sources=cash.sources,
        additional_instruction="Verify that the cited source(s) explicitly list prices spanning $138 (min) to $448 (max). Allow tiny formatting variations.",
        extra_prerequisites=[cash_sources_node, provided_prices],
    )

    # Launch date
    provided_launch = evaluator.add_custom_node(
        result=bool(cash.launch_date),
        id="cashmere_launch_date_provided",
        desc="Cashmere launch date is provided in the answer",
        parent=node,
        critical=True,
    )
    stated_launch = evaluator.add_leaf(
        id="cashmere_launch_date_stated",
        desc="Answer explicitly states: Launch date is October 21, 2025",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the cashmere collection launched on October 21, 2025.",
        node=stated_launch,
        additional_instruction="Check the answer text only. Accept common date formatting variants such as 'Oct. 21, 2025'.",
        extra_prerequisites=[provided_launch],
    )
    launch_leaf = evaluator.add_leaf(
        id="cashmere_launch_date",
        desc="Exact launch date is October 21, 2025",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The October 2025 cashmere collection launched on October 21, 2025.",
        node=launch_leaf,
        sources=cash.sources,
        additional_instruction="Confirm the launch date on the cited page(s). Accept typical date-format variants.",
        extra_prerequisites=[cash_sources_node, provided_launch],
    )

    # Size range
    provided_sizes = evaluator.add_custom_node(
        result=bool(cash.size_range),
        id="cashmere_size_range_provided",
        desc="Cashmere size range is provided in the answer",
        parent=node,
        critical=True,
    )
    stated_sizes = evaluator.add_leaf(
        id="cashmere_size_range_stated",
        desc="Answer explicitly states: Sizes are women’s XXS–3X and men’s up to XXL",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that sizes include women’s XXS to 3X and men’s up to XXL.",
        node=stated_sizes,
        additional_instruction="Check the answer text only. Allow hyphen/en-dash variants (XXS-3X/XXS–3X) and 'XXL'/'2XL' as equivalent.",
        extra_prerequisites=[provided_sizes],
    )
    sizes_leaf = evaluator.add_leaf(
        id="cashmere_size_range",
        desc="Available size range is XXS–3X for women's and up to XXL for men's",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Available sizes for the October 2025 cashmere collection include women’s XXS–3X and men’s up to XXL.",
        node=sizes_leaf,
        sources=cash.sources,
        additional_instruction="Verify wording/meaning on the cited page(s). Allow minor formatting differences and '2XL' as equivalent to 'XXL' for men.",
        extra_prerequisites=[cash_sources_node, provided_sizes],
    )


async def verify_nfl_section(evaluator: Evaluator, parent_node, ext: ShoppingGuideExtraction) -> None:
    nfl = ext.nfl or NFLSection()
    node = evaluator.add_parallel(
        id="nfl_collection_dec_2025",
        desc="NFL-themed collection (December 2025): required collaboration details are provided correctly",
        parent=parent_node,
        critical=True,
    )

    nfl_sources_node = evaluator.add_custom_node(
        result=bool(nfl.sources),
        id="nfl_sources_present",
        desc="NFL-themed section includes at least one reference URL",
        parent=node,
        critical=True,
    )

    # Partner brand
    provided_partner = evaluator.add_custom_node(
        result=bool(nfl.partner_brand),
        id="nfl_partner_brand_provided",
        desc="NFL partner brand is provided in the answer",
        parent=node,
        critical=True,
    )
    stated_partner = evaluator.add_leaf(
        id="nfl_partner_brand_stated",
        desc="Answer explicitly states: Partner brand is Abercrombie & Fitch",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the partner brand for the December 2025 NFL-themed collection is Abercrombie & Fitch.",
        node=stated_partner,
        additional_instruction="Check the answer text only. Allow 'Abercrombie' as shorthand if clearly referring to Abercrombie & Fitch.",
        extra_prerequisites=[provided_partner],
    )
    brand_leaf = evaluator.add_leaf(
        id="nfl_partner_brand",
        desc="Partner brand name is Abercrombie & Fitch",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The December 2025 NFL-themed collection is a collaboration with Abercrombie & Fitch.",
        node=brand_leaf,
        sources=nfl.sources,
        additional_instruction="Confirm on the cited page(s) that Abercrombie & Fitch is the collaboration partner.",
        extra_prerequisites=[nfl_sources_node, provided_partner],
    )

    # Piece count
    provided_pieces = evaluator.add_custom_node(
        result=bool(nfl.piece_count),
        id="nfl_piece_count_provided",
        desc="NFL piece count is provided in the answer",
        parent=node,
        critical=True,
    )
    stated_pieces = evaluator.add_leaf(
        id="nfl_piece_count_stated",
        desc="Answer explicitly states: Total number of pieces is 3",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the NFL-themed collection contains 3 pieces.",
        node=stated_pieces,
        additional_instruction="Check the answer text only. Treat 'three' as equivalent to '3'.",
        extra_prerequisites=[provided_pieces],
    )
    pieces_leaf = evaluator.add_leaf(
        id="nfl_piece_count",
        desc="Total number of pieces is 3",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The December 2025 NFL-themed collection consists of 3 pieces.",
        node=pieces_leaf,
        sources=nfl.sources,
        additional_instruction="Confirm the number of pieces on the cited page(s).",
        extra_prerequisites=[nfl_sources_node, provided_pieces],
    )

    # Featured team
    provided_team = evaluator.add_custom_node(
        result=bool(nfl.featured_team),
        id="nfl_featured_team_provided",
        desc="Featured NFL team is provided in the answer",
        parent=node,
        critical=True,
    )
    stated_team = evaluator.add_leaf(
        id="nfl_featured_team_stated",
        desc="Answer explicitly states: Featured team is the San Francisco 49ers",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the featured NFL team is the San Francisco 49ers.",
        node=stated_team,
        additional_instruction="Check the answer text only. Allow '49ers' as equivalent to 'San Francisco 49ers'.",
        extra_prerequisites=[provided_team],
    )
    team_leaf = evaluator.add_leaf(
        id="nfl_featured_team",
        desc="Featured NFL team is the San Francisco 49ers",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The December 2025 NFL-themed collection features the San Francisco 49ers.",
        node=team_leaf,
        sources=nfl.sources,
        additional_instruction="Verify that the cited page(s) explicitly name the San Francisco 49ers as the featured team.",
        extra_prerequisites=[nfl_sources_node, provided_team],
    )

    # Item types (three)
    provided_items = evaluator.add_custom_node(
        result=bool(nfl.item_types) and len(nfl.item_types) >= 1,
        id="nfl_item_types_provided",
        desc="NFL item types are provided in the answer",
        parent=node,
        critical=True,
    )
    stated_items = evaluator.add_leaf(
        id="nfl_item_types_stated",
        desc="Answer explicitly states: items are bomber jacket, tank top, and crewneck/sweatshirt",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer explicitly states that the three item types included are bomber jacket, tank top, and a crewneck/sweatshirt."
        ),
        node=stated_items,
        additional_instruction="Check the answer text only. Treat 'crewneck' and 'sweatshirt' as acceptable variants.",
        extra_prerequisites=[provided_items],
    )
    items_leaf = evaluator.add_leaf(
        id="nfl_item_types",
        desc="The three item types included are: bomber jacket, tank top, and crewneck/sweatshirt",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The December 2025 NFL-themed collection includes exactly three item types: a bomber jacket, a tank top, and a crewneck/sweatshirt.",
        node=items_leaf,
        sources=nfl.sources,
        additional_instruction="Confirm these exact three item types on the cited page(s). Accept 'crewneck' vs 'sweatshirt' wording.",
        extra_prerequisites=[nfl_sources_node, provided_items],
    )

    # Bomber jacket price
    provided_bomber_price = evaluator.add_custom_node(
        result=bool(nfl.bomber_price),
        id="nfl_bomber_price_provided",
        desc="NFL bomber jacket price is provided in the answer",
        parent=node,
        critical=True,
    )
    stated_bomber_price = evaluator.add_leaf(
        id="nfl_bomber_price_stated",
        desc="Answer explicitly states: bomber jacket price is $220",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the bomber jacket is priced at $220.",
        node=stated_bomber_price,
        additional_instruction="Check the answer text only. Accept $220.00 as equivalent.",
        extra_prerequisites=[provided_bomber_price],
    )
    bomber_price_leaf = evaluator.add_leaf(
        id="nfl_bomber_price",
        desc="Price of the bomber jacket is $220",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The bomber jacket in the December 2025 NFL-themed collection costs $220.",
        node=bomber_price_leaf,
        sources=nfl.sources,
        additional_instruction="Verify the bomber jacket price = $220 on the cited source(s).",
        extra_prerequisites=[nfl_sources_node, provided_bomber_price],
    )

    # Launch date
    provided_launch = evaluator.add_custom_node(
        result=bool(nfl.launch_date),
        id="nfl_launch_date_provided",
        desc="NFL launch date is provided in the answer",
        parent=node,
        critical=True,
    )
    stated_launch = evaluator.add_leaf(
        id="nfl_launch_date_stated",
        desc="Answer explicitly states: launch date is December 11, 2025",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the NFL-themed collection launched on December 11, 2025.",
        node=stated_launch,
        additional_instruction="Check the answer text only. Accept typical date-format variants.",
        extra_prerequisites=[provided_launch],
    )
    launch_leaf = evaluator.add_leaf(
        id="nfl_launch_date",
        desc="Exact launch date is December 11, 2025",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The NFL-themed collection launched on December 11, 2025.",
        node=launch_leaf,
        sources=nfl.sources,
        additional_instruction="Confirm the stated launch date on the cited page(s).",
        extra_prerequisites=[nfl_sources_node, provided_launch],
    )

    # Team colors
    provided_colors = evaluator.add_custom_node(
        result=bool(nfl.team_colors),
        id="nfl_team_colors_provided",
        desc="NFL team colors are provided in the answer",
        parent=node,
        critical=True,
    )
    stated_colors = evaluator.add_leaf(
        id="nfl_team_colors_stated",
        desc="Answer explicitly states: team colors are red and gold",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the official team colors featured are red and gold.",
        node=stated_colors,
        additional_instruction="Check the answer text only. Treat 'scarlet' as a shade of red; allow case and punctuation variations.",
        extra_prerequisites=[provided_colors],
    )
    colors_leaf = evaluator.add_leaf(
        id="nfl_team_colors",
        desc="Official team colors featured are Red and Gold",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The collection features the San Francisco 49ers’ official colors: red and gold.",
        node=colors_leaf,
        sources=nfl.sources,
        additional_instruction="Confirm via cited source(s). Allow 'scarlet' as equivalent to red.",
        extra_prerequisites=[nfl_sources_node, provided_colors],
    )


async def verify_personal_connection(evaluator: Evaluator, parent_node, ext: ShoppingGuideExtraction) -> None:
    pers = ext.personal or PersonalConnectionSection()
    node = evaluator.add_parallel(
        id="personal_connection_to_team",
        desc="Personal connection between Olivia Culpo and the featured NFL team is explained using the required facts",
        parent=parent_node,
        critical=True,
    )

    personal_sources_node = evaluator.add_custom_node(
        result=bool(pers.sources),
        id="personal_sources_present",
        desc="Personal-connection section includes at least one reference URL",
        parent=node,
        critical=True,
    )

    # Husband name
    provided_name = evaluator.add_custom_node(
        result=bool(pers.husband_name),
        id="husband_name_provided",
        desc="Husband name is provided in the answer",
        parent=node,
        critical=True,
    )
    name_leaf = evaluator.add_leaf(
        id="husband_name",
        desc="Olivia Culpo's husband is Christian McCaffrey",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Olivia Culpo’s husband is Christian McCaffrey.",
        node=name_leaf,
        sources=pers.sources,
        additional_instruction="Verify via the cited page(s).",
        extra_prerequisites=[personal_sources_node, provided_name],
    )

    # Husband team
    provided_team = evaluator.add_custom_node(
        result=bool(pers.husband_team),
        id="husband_team_provided",
        desc="Husband NFL team is provided in the answer",
        parent=node,
        critical=True,
    )
    team_leaf = evaluator.add_leaf(
        id="husband_team",
        desc="Her husband's NFL team is the San Francisco 49ers",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Christian McCaffrey plays for the San Francisco 49ers.",
        node=team_leaf,
        sources=pers.sources,
        additional_instruction="Verify via the cited page(s).",
        extra_prerequisites=[personal_sources_node, provided_team],
    )

    # Husband position
    provided_position = evaluator.add_custom_node(
        result=bool(pers.husband_position),
        id="husband_position_provided",
        desc="Husband position is provided in the answer",
        parent=node,
        critical=True,
    )
    position_leaf = evaluator.add_leaf(
        id="husband_position",
        desc="Her husband's position is running back",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Christian McCaffrey’s position is running back.",
        node=position_leaf,
        sources=pers.sources,
        additional_instruction="Verify via the cited page(s).",
        extra_prerequisites=[personal_sources_node, provided_position],
    )

    # Husband jersey number
    provided_number = evaluator.add_custom_node(
        result=bool(pers.husband_jersey_number),
        id="husband_jersey_number_provided",
        desc="Husband jersey number is provided in the answer",
        parent=node,
        critical=True,
    )
    number_leaf = evaluator.add_leaf(
        id="husband_jersey_number",
        desc="Her husband's jersey number is 23",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Christian McCaffrey’s jersey number is 23.",
        node=number_leaf,
        sources=pers.sources,
        additional_instruction="Verify via the cited page(s).",
        extra_prerequisites=[personal_sources_node, provided_number],
    )

    # Connection explanation (explicit in the answer text)
    connection_explained = evaluator.add_leaf(
        id="connection_explanation",
        desc="Explanation explicitly states that the December collection’s featured team is the same team her husband plays for (i.e., links featured_team to husband_team)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer explicitly states that the December 2025 collection’s featured team is the same team that "
            "Olivia Culpo’s husband (Christian McCaffrey) plays for (the San Francisco 49ers)."
        ),
        node=connection_explained,
        additional_instruction="Check the answer text only for an explicit linkage statement.",
    )


async def verify_care_instructions(evaluator: Evaluator, parent_node, ext: ShoppingGuideExtraction) -> None:
    care = ext.care or CareInstructionsSection()
    node = evaluator.add_parallel(
        id="cashmere_care_instructions",
        desc="Care instructions for 100% cashmere items are provided",
        parent=parent_node,
        critical=True,
    )

    care_sources_node = evaluator.add_custom_node(
        result=bool(care.sources),
        id="care_sources_present",
        desc="Care instructions section includes at least one reference URL",
        parent=node,
        critical=True,
    )

    # Washing method
    provided_wash = evaluator.add_custom_node(
        result=bool(care.washing_method),
        id="care_washing_method_provided",
        desc="Washing method is provided in the answer",
        parent=node,
        critical=True,
    )
    washing_leaf = evaluator.add_leaf(
        id="care_washing_method",
        desc="Proper washing method is hand washing or cold/delicate machine washing",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="For 100% cashmere, proper washing is hand wash or a cold/delicate machine cycle.",
        node=washing_leaf,
        sources=care.sources,
        additional_instruction="Confirm via the cited source(s). Accept equivalent wording like 'gentle cycle' for delicate.",
        extra_prerequisites=[care_sources_node, provided_wash],
    )

    # Detergent type
    provided_det = evaluator.add_custom_node(
        result=bool(care.detergent_type),
        id="care_detergent_type_provided",
        desc="Detergent type is provided in the answer",
        parent=node,
        critical=True,
    )
    detergent_leaf = evaluator.add_leaf(
        id="care_detergent_type",
        desc="Recommended detergent type is gentle/mild/wool-friendly detergent",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="For 100% cashmere, use a gentle/mild/wool-friendly detergent.",
        node=detergent_leaf,
        sources=care.sources,
        additional_instruction="Confirm via the cited source(s). Accept 'cashmere detergent' or 'wool wash' as equivalent.",
        extra_prerequisites=[care_sources_node, provided_det],
    )

    # Water temperature
    provided_temp = evaluator.add_custom_node(
        result=bool(care.water_temperature),
        id="care_water_temperature_provided",
        desc="Water temperature guideline is provided in the answer",
        parent=node,
        critical=True,
    )
    temp_leaf = evaluator.add_leaf(
        id="care_water_temperature",
        desc="Water temperature guideline is cold or lukewarm (under 30°C/86°F)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="For 100% cashmere, water should be cold or lukewarm (under 30°C/86°F).",
        node=temp_leaf,
        sources=care.sources,
        additional_instruction="Confirm via the cited source(s). Accept small numeric formatting differences.",
        extra_prerequisites=[care_sources_node, provided_temp],
    )

    # Drying method
    provided_dry = evaluator.add_custom_node(
        result=bool(care.drying_method),
        id="care_drying_method_provided",
        desc="Drying method is provided in the answer",
        parent=node,
        critical=True,
    )
    drying_leaf = evaluator.add_leaf(
        id="care_drying_method",
        desc="Drying method is air drying (no tumble dry)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="For 100% cashmere, lay flat to air dry; do not tumble dry.",
        node=drying_leaf,
        sources=care.sources,
        additional_instruction="Confirm via the cited source(s). Accept equivalent wording like 'flat dry' or 'no dryer'.",
        extra_prerequisites=[care_sources_node, provided_dry],
    )


async def verify_retail_availability(evaluator: Evaluator, parent_node, ext: ShoppingGuideExtraction) -> None:
    retail = ext.retail or RetailAvailabilitySection()
    node = evaluator.add_parallel(
        id="retail_availability",
        desc="Retail availability for both collections is documented",
        parent=parent_node,
        critical=True,
    )

    # Cashmere retail
    provided_cash_retail = evaluator.add_custom_node(
        result=bool(retail.cashmere_availability),
        id="retail_cashmere_availability_provided",
        desc="Cashmere retail availability is provided in the answer",
        parent=node,
        critical=True,
    )
    cash_retail_leaf = evaluator.add_leaf(
        id="naadam_availability",
        desc="NAADAM collection availability includes naadam.co and select retailers",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The cashmere collection is available at naadam.co and select retailers.",
        node=cash_retail_leaf,
        sources=_any_sources(retail.cashmere_sources),
        additional_instruction="Verify via the cited source(s). Accept 'NAADAM website' as naadam.co; 'select retailers' can be phrased as partner/selected retailers.",
        extra_prerequisites=[provided_cash_retail],
    )

    # NFL retail
    provided_nfl_retail = evaluator.add_custom_node(
        result=bool(retail.nfl_availability),
        id="retail_nfl_availability_provided",
        desc="NFL retail availability is provided in the answer",
        parent=node,
        critical=True,
    )
    nfl_retail_leaf = evaluator.add_leaf(
        id="abercrombie_availability",
        desc="Abercrombie collection availability includes Abercrombie & Fitch stores and online",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The NFL-themed collection is available at Abercrombie & Fitch stores and online.",
        node=nfl_retail_leaf,
        sources=_any_sources(retail.nfl_sources),
        additional_instruction="Confirm via the cited source(s). Accept 'in stores and online' phrasing.",
        extra_prerequisites=[provided_nfl_retail],
    )


async def verify_citations_coverage(evaluator: Evaluator, parent_node, ext: ShoppingGuideExtraction) -> None:
    # Build grouped citation pools (fall back to per-section sources if explicit 'citations' absent)
    cashmere_urls = _any_sources(
        ext.citations.cashmere if ext.citations else None,
        (ext.cashmere.sources if ext.cashmere else None),
    )
    nfl_urls = _any_sources(
        ext.citations.nfl if ext.citations else None,
        (ext.nfl.sources if ext.nfl else None),
    )
    personal_urls = _any_sources(
        ext.citations.personal if ext.citations else None,
        (ext.personal.sources if ext.personal else None),
    )
    care_urls = _any_sources(
        ext.citations.care if ext.citations else None,
        (ext.care.sources if ext.care else None),
    )
    retail_urls = _any_sources(
        ext.citations.retail if ext.citations else None,
        _safe_list((ext.retail.cashmere_sources if ext.retail else []) + (ext.retail.nfl_sources if ext.retail else [])),
    )

    node = evaluator.add_parallel(
        id="citations",
        desc="Reference URLs are provided to support the factual claims across all sections",
        parent=parent_node,
        critical=True,
    )

    # Existence-only checks for citations per section (critical)
    evaluator.add_custom_node(
        result=bool(cashmere_urls),
        id="cashmere_citations",
        desc="Includes reference URL(s) supporting the cashmere collection facts",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(nfl_urls),
        id="nfl_citations",
        desc="Includes reference URL(s) supporting the NFL-themed collection facts",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(personal_urls),
        id="personal_connection_citations",
        desc="Includes reference URL(s) supporting the personal-connection facts",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(care_urls),
        id="care_citations",
        desc="Includes reference URL(s) supporting the cashmere care instructions",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(retail_urls),
        id="retail_citations",
        desc="Includes reference URL(s) supporting the retail availability claims",
        parent=node,
        critical=True,
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
    Evaluate an answer for Olivia Culpo's late-2025 collaborations shopping guide.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # root container (non-critical) per framework design
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

    # Critical top-level node that mirrors the rubric's root requirement
    task_root = evaluator.add_parallel(
        id="task_requirements",
        desc="Shopping guide documents Olivia Culpo's two late-2025 fashion collaborations and all requested fields, plus care instructions, retail availability, and reference URLs",
        parent=root,
        critical=True,
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_shopping_guide(),
        template_class=ShoppingGuideExtraction,
        extraction_name="shopping_guide_extraction",
    )

    # Record expected ground-truth target facts for transparency
    evaluator.add_ground_truth({
        "cashmere_expected": CASHMERE_EXPECTED,
        "nfl_expected": NFL_EXPECTED,
        "personal_expected": PERSONAL_EXPECTED,
        "retail_expected": RETAIL_EXPECTED,
    })

    # Build and verify sections according to rubric
    await verify_cashmere_section(evaluator, task_root, extracted)
    await verify_nfl_section(evaluator, task_root, extracted)
    await verify_personal_connection(evaluator, task_root, extracted)
    await verify_care_instructions(evaluator, task_root, extracted)
    await verify_retail_availability(evaluator, task_root, extracted)
    await verify_citations_coverage(evaluator, task_root, extracted)

    # Return structured summary with verification tree
    return evaluator.get_summary()