import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "holiday_gifts_2025_2026"
TASK_DESCRIPTION = """
You are planning to purchase three specific tech and collectible gifts for the 2025-2026 holiday season. For each gift category below, provide complete verification details including product specifications, pricing, availability, and purchase logistics:

Gift 1: Apple Watch Series 11
- Specify the available case size options and their corresponding retail prices for both GPS-only and GPS+Cellular models
- Confirm the minimum iPhone model and iOS version required for compatibility
- List key health monitoring features (must include any blood pressure related features)
- Identify a major electronics retailer where this product can be purchased and provide their Black Friday 2025 (November 28) store hours

Gift 2: Nintendo Switch 2
- Confirm the official release date and standalone console price
- Identify the game included in the official launch bundle and verify the bundle price
- Specify the handheld screen resolution
- Confirm backward compatibility with original Nintendo Switch games
- Identify a major retailer where this console can be purchased and provide their Christmas Eve 2025 (December 24) closing time

Gift 3: LEGO Pokemon Sets
- Confirm the official release date for the LEGO Pokemon collection
- For each of the three available sets (Eevee, Pikachu and Poke Ball, Venusaur/Charizard/Blastoise), provide: set number, piece count, and retail price
- Confirm whether pre-orders were available and if so, when they began
- Identify official purchase locations for these sets

For all product information, provide reference URLs that support your answers.
"""


# =========================
# Extraction Models
# =========================

class AppleWatchExtraction(BaseModel):
    # Model specifications
    size_options: List[str] = Field(default_factory=list)
    connectivity_options: List[str] = Field(default_factory=list)
    model_reference_urls: List[str] = Field(default_factory=list)

    # Pricing
    pricing_42mm_gps: Optional[str] = None
    pricing_42mm_cellular: Optional[str] = None
    pricing_46mm_gps: Optional[str] = None
    pricing_46mm_cellular: Optional[str] = None
    pricing_reference_urls: List[str] = Field(default_factory=list)

    # Compatibility
    min_iphone_model: Optional[str] = None
    required_ios_version: Optional[str] = None
    compatibility_reference_urls: List[str] = Field(default_factory=list)

    # Health features
    health_features: List[str] = Field(default_factory=list)
    health_reference_urls: List[str] = Field(default_factory=list)

    # Retailer information
    retailer_name: Optional[str] = None
    retailer_product_url: Optional[str] = None
    black_friday_2025_open_time: Optional[str] = None
    black_friday_2025_close_time: Optional[str] = None
    retailer_hours_urls: List[str] = Field(default_factory=list)


class Switch2Extraction(BaseModel):
    # Release information
    release_date: Optional[str] = None
    release_reference_urls: List[str] = Field(default_factory=list)

    # Pricing details
    standalone_price: Optional[str] = None
    bundle_game: Optional[str] = None
    bundle_price: Optional[str] = None
    pricing_reference_urls: List[str] = Field(default_factory=list)

    # Technical specifications
    handheld_screen_resolution: Optional[str] = None
    backward_compatibility: Optional[str] = None  # free-form string: e.g., "Yes / No / partial"
    tech_reference_urls: List[str] = Field(default_factory=list)

    # Retailer information
    retailer_name: Optional[str] = None
    retailer_product_url: Optional[str] = None
    christmas_eve_2025_closing_time: Optional[str] = None
    retailer_hours_urls: List[str] = Field(default_factory=list)


class LegoPokemonExtraction(BaseModel):
    # Collection release
    collection_release_date: Optional[str] = None
    release_reference_urls: List[str] = Field(default_factory=list)

    # Set details (3 sets)
    eevee_set_number: Optional[str] = None
    eevee_piece_count: Optional[str] = None
    eevee_price: Optional[str] = None

    pikachu_set_number: Optional[str] = None
    pikachu_piece_count: Optional[str] = None
    pikachu_price: Optional[str] = None

    starters_set_number: Optional[str] = None
    starters_piece_count: Optional[str] = None
    starters_price: Optional[str] = None

    sets_reference_urls: List[str] = Field(default_factory=list)

    # Pre-order details
    preorder_available: Optional[str] = None  # "Yes" / "No" / free-form
    preorder_start_date: Optional[str] = None
    preorder_reference_urls: List[str] = Field(default_factory=list)

    # Purchase locations
    purchase_locations: List[str] = Field(default_factory=list)
    purchase_location_urls: List[str] = Field(default_factory=list)
    purchase_reference_urls: List[str] = Field(default_factory=list)


# =========================
# Extraction Prompts
# =========================

def prompt_extract_apple_watch() -> str:
    return """
    Extract all Apple Watch Series 11 information explicitly provided in the answer. Return a JSON object with the following fields:

    Model specifications:
    - size_options: array of strings listing all case sizes mentioned (e.g., ["42mm","46mm"]; preserve units text if shown).
    - connectivity_options: array of strings (e.g., ["GPS-only","GPS+Cellular"]).
    - model_reference_urls: array of URLs that specifically reference Series 11 model specs (sizes and connectivity).

    Pricing:
    - pricing_42mm_gps: string price for the 42mm GPS model (e.g., "$399"). If not provided, null.
    - pricing_42mm_cellular: string price for the 42mm GPS+Cellular model.
    - pricing_46mm_gps: string price for the 46mm GPS model.
    - pricing_46mm_cellular: string price for the 46mm GPS+Cellular model.
    - pricing_reference_urls: array of URLs that explicitly state Series 11 pricing.

    Compatibility:
    - min_iphone_model: string naming the minimum iPhone model required (e.g., "iPhone X").
    - required_ios_version: string naming the minimum iOS version required (e.g., "iOS 18").
    - compatibility_reference_urls: array of URLs confirming Apple Watch Series 11 iPhone/iOS compatibility.

    Health features:
    - health_features: array of strings naming key health features listed in the answer (e.g., ["ECG","Blood oxygen","Blood pressure trend"]).
    - health_reference_urls: array of URLs that describe these health features (prefer official Apple pages or major retailers).

    Retailer & hours:
    - retailer_name: string for a major electronics retailer selling Series 11 (e.g., "Best Buy").
    - retailer_product_url: URL to the Series 11 product page at that retailer.
    - black_friday_2025_open_time: string time for Nov 28, 2025 opening hours at that retailer (e.g., "5:00 AM").
    - black_friday_2025_close_time: string time for Nov 28, 2025 closing hours (e.g., "10:00 PM").
    - retailer_hours_urls: array of URLs (store locator or holiday hours pages) supporting the Black Friday hours.

    Rules:
    - Only extract URLs present in the answer.
    - For any missing field, return null or empty array accordingly.
    """


def prompt_extract_switch2() -> str:
    return """
    Extract all Nintendo Switch 2 information from the answer. Return a JSON object with:

    Release information:
    - release_date: string official release date (e.g., "September 12, 2025").
    - release_reference_urls: array of URLs confirming the release date.

    Pricing details:
    - standalone_price: string price for the console alone (e.g., "$399").
    - bundle_game: string name of the game included in the official launch bundle.
    - bundle_price: string total price for the bundle (e.g., "$449").
    - pricing_reference_urls: array of URLs confirming standalone and bundle pricing.

    Technical specs:
    - handheld_screen_resolution: string resolution for handheld mode (e.g., "1080p", "1920×1080").
    - backward_compatibility: free-form string (e.g., "Yes, supports original Switch games" or "No").
    - tech_reference_urls: array of URLs confirming resolution and backward compatibility.

    Retailer & hours:
    - retailer_name: string of a major retailer selling Switch 2 (e.g., "GameStop" or "Walmart").
    - retailer_product_url: URL to the Switch 2 product page at that retailer.
    - christmas_eve_2025_closing_time: string closing time for Dec 24, 2025 (e.g., "6:00 PM").
    - retailer_hours_urls: array of URLs (store pages or hours pages) supporting the Christmas Eve closing time.

    Rules:
    - Extract only what’s explicitly in the answer; use URLs shown in the answer as sources.
    - Missing fields -> null or empty array as appropriate.
    """


def prompt_extract_lego_pokemon() -> str:
    return """
    Extract all LEGO Pokémon collection information. Return a JSON object with:

    Collection release:
    - collection_release_date: string official release date for the LEGO Pokémon collection.
    - release_reference_urls: array of URLs confirming the release date.

    Set details:
    For each of the three sets (Eevee; Pikachu and Poke Ball; Venusaur, Charizard & Blastoise):
    - eevee_set_number, eevee_piece_count, eevee_price
    - pikachu_set_number, pikachu_piece_count, pikachu_price
    - starters_set_number, starters_piece_count, starters_price
    - sets_reference_urls: array of URLs confirming set numbers, piece counts, and prices (one or more pages covering all sets is acceptable).

    Pre-order details:
    - preorder_available: string stating if pre-orders were available ("Yes"/"No"/free-form).
    - preorder_start_date: string date pre-orders began (if applicable).
    - preorder_reference_urls: array of URLs confirming preorder status and date.

    Purchase locations:
    - purchase_locations: array of retailer names or "LEGO.com".
    - purchase_location_urls: array of URLs to product pages or official listings where sets can be purchased.
    - purchase_reference_urls: array of URLs confirming official purchase locations (can duplicate purchase_location_urls).

    Rules:
    - Include only URLs present in the answer.
    - If data is missing, use null or empty arrays accordingly.
    """


# =========================
# Helper Functions
# =========================

def _join_list(items: List[str]) -> str:
    return ", ".join([s.strip() for s in items if s and s.strip()]) if items else ""


def _affirmative(text: Optional[str]) -> Optional[bool]:
    if text is None:
        return None
    t = text.lower()
    if any(k in t for k in ["yes", "supported", "backward compatible", "plays", "compatible", "works with"]):
        return True
    if any(k in t for k in ["no", "not backward", "incompatible", "does not", "cannot"]):
        return False
    return None


# =========================
# Verification Builders
# =========================

async def build_smartwatch_verification(evaluator: Evaluator, parent_node, aw: AppleWatchExtraction) -> None:
    # Category node
    watch_node = evaluator.add_parallel(
        id="Smartwatch_Gift",
        desc="Verify specifications and purchase details for an Apple Watch Series 11 gift",
        parent=parent_node,
        critical=False
    )

    # Model Specifications (critical)
    specs_node = evaluator.add_parallel(
        id="Model_Specifications",
        desc="Verify Apple Watch Series 11 model specifications including size options and connectivity types",
        parent=watch_node,
        critical=True
    )

    # Available sizes
    sizes_leaf = evaluator.add_leaf(
        id="Available_Sizes",
        desc="Identify all available case size options for Apple Watch Series 11",
        parent=specs_node,
        critical=True
    )
    sizes_claim = f"Apple Watch Series 11 offers case sizes: {_join_list(aw.size_options)}."
    await evaluator.verify(
        claim=sizes_claim,
        node=sizes_leaf,
        sources=aw.model_reference_urls,
        additional_instruction="Confirm the case sizes for Series 11 on the provided page(s). Allow minor formatting variations like including 'mm'."
    )

    # Connectivity types
    conn_leaf = evaluator.add_leaf(
        id="Connectivity_Types",
        desc="Identify the connectivity options available (GPS-only and/or GPS+Cellular)",
        parent=specs_node,
        critical=True
    )
    conn_claim = f"Apple Watch Series 11 is available in connectivity options: {_join_list(aw.connectivity_options)}."
    await evaluator.verify(
        claim=conn_claim,
        node=conn_leaf,
        sources=aw.model_reference_urls,
        additional_instruction="The page should mention GPS-only and/or GPS+Cellular options for Series 11."
    )

    # Model reference URL(s) validity
    model_ref_leaf = evaluator.add_leaf(
        id="Model_Reference_URL",
        desc="Provide reference URL confirming Apple Watch Series 11 model specifications",
        parent=specs_node,
        critical=True
    )
    model_ref_claim = "This page contains Apple Watch Series 11 model specifications (case sizes and connectivity options)."
    await evaluator.verify(
        claim=model_ref_claim,
        node=model_ref_leaf,
        sources=aw.model_reference_urls,
        additional_instruction="Verify that the linked page(s) clearly list Series 11 specs including size(s) and connectivity."
    )

    # Pricing Information (critical)
    pricing_node = evaluator.add_parallel(
        id="Pricing_Information",
        desc="Verify pricing for all Apple Watch Series 11 model configurations",
        parent=watch_node,
        critical=True
    )

    p42_gps_leaf = evaluator.add_leaf(
        id="Size_42mm_GPS_Price",
        desc="Provide the retail price for 42mm GPS model",
        parent=pricing_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The retail price for the Apple Watch Series 11 42mm GPS model is {aw.pricing_42mm_gps}.",
        node=p42_gps_leaf,
        sources=aw.pricing_reference_urls,
        additional_instruction="Match the listed price for the 42mm GPS configuration. Allow currency symbols and minor formatting variations."
    )

    p42_cell_leaf = evaluator.add_leaf(
        id="Size_42mm_Cellular_Price",
        desc="Provide the retail price for 42mm GPS+Cellular model",
        parent=pricing_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The retail price for the Apple Watch Series 11 42mm GPS+Cellular model is {aw.pricing_42mm_cellular}.",
        node=p42_cell_leaf,
        sources=aw.pricing_reference_urls,
        additional_instruction="Confirm the price for the 42mm GPS+Cellular model."
    )

    p46_gps_leaf = evaluator.add_leaf(
        id="Size_46mm_GPS_Price",
        desc="Provide the retail price for 46mm GPS model",
        parent=pricing_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The retail price for the Apple Watch Series 11 46mm GPS model is {aw.pricing_46mm_gps}.",
        node=p46_gps_leaf,
        sources=aw.pricing_reference_urls,
        additional_instruction="Confirm the price for the 46mm GPS model."
    )

    p46_cell_leaf = evaluator.add_leaf(
        id="Size_46mm_Cellular_Price",
        desc="Provide the retail price for 46mm GPS+Cellular model",
        parent=pricing_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The retail price for the Apple Watch Series 11 46mm GPS+Cellular model is {aw.pricing_46mm_cellular}.",
        node=p46_cell_leaf,
        sources=aw.pricing_reference_urls,
        additional_instruction="Confirm the price for the 46mm GPS+Cellular model."
    )

    p_ref_leaf = evaluator.add_leaf(
        id="Pricing_Reference_URL",
        desc="Provide reference URL confirming Apple Watch Series 11 pricing",
        parent=pricing_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page provides official or retailer-listed pricing for Apple Watch Series 11 configurations.",
        node=p_ref_leaf,
        sources=aw.pricing_reference_urls,
        additional_instruction="The linked page(s) should list Series 11 prices by size and/or connectivity."
    )

    # Compatibility Requirements (critical)
    comp_node = evaluator.add_parallel(
        id="Compatibility_Requirements",
        desc="Verify iPhone compatibility requirements for Apple Watch Series 11",
        parent=watch_node,
        critical=True
    )

    min_iphone_leaf = evaluator.add_leaf(
        id="Minimum_iPhone_Model",
        desc="Specify the minimum iPhone model required for compatibility",
        parent=comp_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The minimum iPhone model required for Apple Watch Series 11 is {aw.min_iphone_model}.",
        node=min_iphone_leaf,
        sources=aw.compatibility_reference_urls,
        additional_instruction="Confirm the minimum iPhone requirement stated on the page."
    )

    req_ios_leaf = evaluator.add_leaf(
        id="Required_iOS_Version",
        desc="Specify the minimum iOS version required for compatibility",
        parent=comp_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The minimum iOS version required for Apple Watch Series 11 is {aw.required_ios_version}.",
        node=req_ios_leaf,
        sources=aw.compatibility_reference_urls,
        additional_instruction="Confirm the minimum iOS version requirement stated on the page."
    )

    comp_ref_leaf = evaluator.add_leaf(
        id="Compatibility_Reference_URL",
        desc="Provide reference URL confirming compatibility requirements",
        parent=comp_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page confirms Apple Watch Series 11 compatibility requirements (minimum iPhone model and iOS version).",
        node=comp_ref_leaf,
        sources=aw.compatibility_reference_urls,
        additional_instruction="The page should mention both iPhone and iOS requirements for Series 11."
    )

    # Health Features (non-critical)
    health_leaf = evaluator.add_leaf(
        id="Health_Features",
        desc="List key health monitoring features, ensuring blood pressure related features are included",
        parent=watch_node,
        critical=False
    )
    health_claim = f"Key Apple Watch Series 11 health features include: {_join_list(aw.health_features)}. At least one listed feature is blood-pressure-related."
    await evaluator.verify(
        claim=health_claim,
        node=health_leaf,
        sources=aw.health_reference_urls,
        additional_instruction="Confirm the listed health features. If the product includes any blood-pressure-related functionality (measurement, trends, notifications), ensure the page mentions it."
    )

    # Retailer Information (non-critical)
    retail_node = evaluator.add_parallel(
        id="Retailer_Information",
        desc="Identify purchase location and Black Friday shopping hours",
        parent=watch_node,
        critical=False
    )

    retailer_leaf = evaluator.add_leaf(
        id="Electronics_Retailer",
        desc="Identify at least one major electronics retailer where Apple Watch Series 11 can be purchased",
        parent=retail_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Apple Watch Series 11 can be purchased at {aw.retailer_name}.",
        node=retailer_leaf,
        sources=aw.retailer_product_url,
        additional_instruction="Confirm that the linked retailer page sells Apple Watch Series 11 (product name or model clearly shown)."
    )

    bf_open_leaf = evaluator.add_leaf(
        id="Black_Friday_Opening",
        desc="Provide the store opening time on Black Friday 2025 (November 28) for the identified retailer",
        parent=retail_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The store opening time on Black Friday 2025 (Nov 28) for {aw.retailer_name} is {aw.black_friday_2025_open_time}.",
        node=bf_open_leaf,
        sources=aw.retailer_hours_urls,
        additional_instruction="Use the retailer's official hours page or store locator; confirm the specific Black Friday 2025 opening time."
    )

    bf_close_leaf = evaluator.add_leaf(
        id="Black_Friday_Closing",
        desc="Provide the store closing time on Black Friday 2025 (November 28) for the identified retailer",
        parent=retail_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The store closing time on Black Friday 2025 (Nov 28) for {aw.retailer_name} is {aw.black_friday_2025_close_time}.",
        node=bf_close_leaf,
        sources=aw.retailer_hours_urls,
        additional_instruction="Use the retailer's official hours page or store locator; confirm the specific Black Friday 2025 closing time."
    )


async def build_switch_verification(evaluator: Evaluator, parent_node, ns: Switch2Extraction) -> None:
    # Category node
    switch_node = evaluator.add_parallel(
        id="Gaming_Console_Gift",
        desc="Verify specifications and purchase details for a Nintendo Switch 2 console gift",
        parent=parent_node,
        critical=False
    )

    # Release Information (critical)
    rel_node = evaluator.add_parallel(
        id="Release_Information",
        desc="Verify Nintendo Switch 2 release date",
        parent=switch_node,
        critical=True
    )

    rel_date_leaf = evaluator.add_leaf(
        id="Official_Release_Date",
        desc="Confirm the official release date of Nintendo Switch 2",
        parent=rel_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official release date for Nintendo Switch 2 is {ns.release_date}.",
        node=rel_date_leaf,
        sources=ns.release_reference_urls,
        additional_instruction="Confirm the release date from official Nintendo or major reputable sources."
    )

    rel_ref_leaf = evaluator.add_leaf(
        id="Release_Reference_URL",
        desc="Provide reference URL confirming Nintendo Switch 2 release date",
        parent=rel_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page confirms the official release date of Nintendo Switch 2.",
        node=rel_ref_leaf,
        sources=ns.release_reference_urls,
        additional_instruction="The page should explicitly state the Switch 2 release date."
    )

    # Pricing Details (critical)
    price_node = evaluator.add_parallel(
        id="Pricing_Details",
        desc="Verify pricing for Nintendo Switch 2 standalone and bundle options",
        parent=switch_node,
        critical=True
    )

    standalone_leaf = evaluator.add_leaf(
        id="Standalone_Console_Price",
        desc="Verify the retail price for Nintendo Switch 2 console standalone (without bundle)",
        parent=price_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The retail price for the Nintendo Switch 2 standalone console is {ns.standalone_price}.",
        node=standalone_leaf,
        sources=ns.pricing_reference_urls,
        additional_instruction="Confirm the standalone console price on the provided page(s)."
    )

    bundle_game_leaf = evaluator.add_leaf(
        id="Bundle_Game_Identity",
        desc="Identify the game included in the official Nintendo Switch 2 launch bundle",
        parent=price_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official Nintendo Switch 2 launch bundle includes the game '{ns.bundle_game}'.",
        node=bundle_game_leaf,
        sources=ns.pricing_reference_urls,
        additional_instruction="Confirm the bundled game's title."
    )

    bundle_price_leaf = evaluator.add_leaf(
        id="Bundle_Total_Price",
        desc="Verify the retail price for the Nintendo Switch 2 bundle including the game",
        parent=price_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The retail price for the Nintendo Switch 2 launch bundle (with '{ns.bundle_game}') is {ns.bundle_price}.",
        node=bundle_price_leaf,
        sources=ns.pricing_reference_urls,
        additional_instruction="Confirm the price of the official launch bundle."
    )

    price_ref_leaf = evaluator.add_leaf(
        id="Pricing_Reference_URL",
        desc="Provide reference URL confirming Nintendo Switch 2 pricing and bundle details",
        parent=price_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page provides official pricing for Nintendo Switch 2 (standalone and launch bundle).",
        node=price_ref_leaf,
        sources=ns.pricing_reference_urls,
        additional_instruction="The page(s) should clearly list standalone and bundle pricing."
    )

    # Technical Specifications (non-critical)
    tech_node = evaluator.add_parallel(
        id="Technical_Specifications",
        desc="Verify technical specifications for Nintendo Switch 2",
        parent=switch_node,
        critical=False
    )

    res_leaf = evaluator.add_leaf(
        id="Handheld_Screen_Resolution",
        desc="Specify the handheld screen resolution for Nintendo Switch 2",
        parent=tech_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The handheld screen resolution for Nintendo Switch 2 is {ns.handheld_screen_resolution}.",
        node=res_leaf,
        sources=ns.tech_reference_urls,
        additional_instruction="Confirm the handheld-mode resolution value."
    )

    bc_leaf = evaluator.add_leaf(
        id="Backward_Compatibility_Status",
        desc="Confirm whether Nintendo Switch 2 can play original Nintendo Switch games",
        parent=tech_node,
        critical=False
    )
    bc_bool = _affirmative(ns.backward_compatibility)
    if bc_bool is True:
        bc_claim = "Nintendo Switch 2 is backward compatible and can play original Nintendo Switch games."
    elif bc_bool is False:
        bc_claim = "Nintendo Switch 2 is not backward compatible with original Nintendo Switch games."
    else:
        bc_claim = f"Backward compatibility status for Nintendo Switch 2 is: {ns.backward_compatibility}."
    await evaluator.verify(
        claim=bc_claim,
        node=bc_leaf,
        sources=ns.tech_reference_urls,
        additional_instruction="Confirm the statement about backward compatibility with original Switch titles."
    )

    # Retailer Information (non-critical)
    s_retail_node = evaluator.add_parallel(
        id="Retailer_Information",
        desc="Identify purchase location and Christmas Eve shopping hours",
        parent=switch_node,
        critical=False
    )

    s_retailer_leaf = evaluator.add_leaf(
        id="Gaming_Retailer",
        desc="Identify at least one major retailer where Nintendo Switch 2 can be purchased",
        parent=s_retail_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Nintendo Switch 2 can be purchased at {ns.retailer_name}.",
        node=s_retailer_leaf,
        sources=ns.retailer_product_url,
        additional_instruction="Confirm the product page is for Nintendo Switch 2 and available for purchase."
    )

    xmas_close_leaf = evaluator.add_leaf(
        id="Christmas_Eve_Closing_Time",
        desc="Provide the store closing time on Christmas Eve 2025 (December 24) for the identified retailer",
        parent=s_retail_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The store closing time on December 24, 2025 for {ns.retailer_name} is {ns.christmas_eve_2025_closing_time}.",
        node=xmas_close_leaf,
        sources=ns.retailer_hours_urls,
        additional_instruction="Use official retailer hours page or store locator; confirm the specific Christmas Eve 2025 closing time."
    )


async def build_lego_verification(evaluator: Evaluator, parent_node, lp: LegoPokemonExtraction) -> None:
    # Category node
    lego_node = evaluator.add_parallel(
        id="Collectible_Set_Gift",
        desc="Verify specifications and purchase details for LEGO Pokemon sets",
        parent=parent_node,
        critical=False
    )

    # Collection Release (critical)
    coll_node = evaluator.add_parallel(
        id="Collection_Release",
        desc="Verify the release date for LEGO Pokemon sets collection",
        parent=lego_node,
        critical=True
    )

    coll_date_leaf = evaluator.add_leaf(
        id="Collection_Release_Date",
        desc="Confirm the official release date for LEGO Pokemon sets",
        parent=coll_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official release date for the LEGO Pokémon collection is {lp.collection_release_date}.",
        node=coll_date_leaf,
        sources=lp.release_reference_urls,
        additional_instruction="Confirm the collection release date from LEGO.com or reputable sources."
    )

    coll_ref_leaf = evaluator.add_leaf(
        id="Release_Reference_URL",
        desc="Provide reference URL confirming LEGO Pokemon sets release date",
        parent=coll_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page confirms the official release date for the LEGO Pokémon collection.",
        node=coll_ref_leaf,
        sources=lp.release_reference_urls,
        additional_instruction="The page should explicitly state the collection release date."
    )

    # Set Details (critical)
    sets_node = evaluator.add_parallel(
        id="Set_Details",
        desc="Provide detailed specifications for all three LEGO Pokemon sets",
        parent=lego_node,
        critical=True
    )

    # Eevee set (critical)
    eevee_node = evaluator.add_parallel(
        id="Eevee_Set",
        desc="Provide complete details for the LEGO Pokemon Eevee set",
        parent=sets_node,
        critical=True
    )
    ee_num_leaf = evaluator.add_leaf(
        id="Eevee_Set_Number",
        desc="Provide the set number for the Eevee set",
        parent=eevee_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The LEGO Pokémon Eevee set number is {lp.eevee_set_number}.",
        node=ee_num_leaf,
        sources=lp.sets_reference_urls,
        additional_instruction="Confirm the Eevee set number."
    )
    ee_piece_leaf = evaluator.add_leaf(
        id="Eevee_Piece_Count",
        desc="Provide the piece count for the Eevee set",
        parent=eevee_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The LEGO Pokémon Eevee set piece count is {lp.eevee_piece_count}.",
        node=ee_piece_leaf,
        sources=lp.sets_reference_urls,
        additional_instruction="Confirm the Eevee set piece count."
    )
    ee_price_leaf = evaluator.add_leaf(
        id="Eevee_Price",
        desc="Provide the retail price for the Eevee set",
        parent=eevee_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The retail price for the LEGO Pokémon Eevee set is {lp.eevee_price}.",
        node=ee_price_leaf,
        sources=lp.sets_reference_urls,
        additional_instruction="Confirm the Eevee set price."
    )

    # Pikachu & Poke Ball set (critical)
    pika_node = evaluator.add_parallel(
        id="Pikachu_Set",
        desc="Provide complete details for the LEGO Pokemon Pikachu and Poke Ball set",
        parent=sets_node,
        critical=True
    )
    pika_num_leaf = evaluator.add_leaf(
        id="Pikachu_Set_Number",
        desc="Provide the set number for the Pikachu and Poke Ball set",
        parent=pika_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The LEGO Pokémon Pikachu and Poke Ball set number is {lp.pikachu_set_number}.",
        node=pika_num_leaf,
        sources=lp.sets_reference_urls,
        additional_instruction="Confirm the Pikachu and Poke Ball set number."
    )
    pika_piece_leaf = evaluator.add_leaf(
        id="Pikachu_Piece_Count",
        desc="Provide the piece count for the Pikachu and Poke Ball set",
        parent=pika_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The LEGO Pokémon Pikachu and Poke Ball set piece count is {lp.pikachu_piece_count}.",
        node=pika_piece_leaf,
        sources=lp.sets_reference_urls,
        additional_instruction="Confirm the Pikachu and Poke Ball set piece count."
    )
    pika_price_leaf = evaluator.add_leaf(
        id="Pikachu_Price",
        desc="Provide the retail price for the Pikachu and Poke Ball set",
        parent=pika_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The retail price for the LEGO Pokémon Pikachu and Poke Ball set is {lp.pikachu_price}.",
        node=pika_price_leaf,
        sources=lp.sets_reference_urls,
        additional_instruction="Confirm the Pikachu and Poke Ball set price."
    )

    # Starters set (critical)
    starters_node = evaluator.add_parallel(
        id="Starter_Pokemon_Set",
        desc="Provide complete details for the LEGO Pokemon Venusaur, Charizard and Blastoise set",
        parent=sets_node,
        critical=True
    )
    st_num_leaf = evaluator.add_leaf(
        id="Starters_Set_Number",
        desc="Provide the set number for the Venusaur, Charizard and Blastoise set",
        parent=starters_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The LEGO Pokémon Venusaur, Charizard and Blastoise set number is {lp.starters_set_number}.",
        node=st_num_leaf,
        sources=lp.sets_reference_urls,
        additional_instruction="Confirm the starters set number."
    )
    st_piece_leaf = evaluator.add_leaf(
        id="Starters_Piece_Count",
        desc="Provide the piece count for the Venusaur, Charizard and Blastoise set",
        parent=starters_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The LEGO Pokémon Venusaur, Charizard and Blastoise set piece count is {lp.starters_piece_count}.",
        node=st_piece_leaf,
        sources=lp.sets_reference_urls,
        additional_instruction="Confirm the starters set piece count."
    )
    st_price_leaf = evaluator.add_leaf(
        id="Starters_Price",
        desc="Provide the retail price for the Venusaur, Charizard and Blastoise set",
        parent=starters_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The retail price for the LEGO Pokémon Venusaur, Charizard and Blastoise set is {lp.starters_price}.",
        node=st_price_leaf,
        sources=lp.sets_reference_urls,
        additional_instruction="Confirm the starters set price."
    )

    # Sets Reference URL leaf (critical)
    sets_ref_leaf = evaluator.add_leaf(
        id="Sets_Reference_URL",
        desc="Provide reference URL confirming LEGO Pokemon sets specifications and pricing",
        parent=sets_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page provides LEGO Pokémon sets specifications (set numbers, piece counts) and pricing.",
        node=sets_ref_leaf,
        sources=lp.sets_reference_urls,
        additional_instruction="The linked page(s) should contain specs and prices for the Pokémon sets."
    )

    # Pre-order details (non-critical)
    preorder_node = evaluator.add_parallel(
        id="Pre_Order_Details",
        desc="Verify pre-order availability and timing information",
        parent=lego_node,
        critical=False
    )

    pre_avail_leaf = evaluator.add_leaf(
        id="Pre_Order_Available",
        desc="Confirm whether pre-orders for LEGO Pokemon sets were available",
        parent=preorder_node,
        critical=False
    )
    pa = _affirmative(lp.preorder_available)
    if pa is True:
        pre_avail_claim = "Pre-orders for LEGO Pokémon sets were available."
    elif pa is False:
        pre_avail_claim = "Pre-orders for LEGO Pokémon sets were not available."
    else:
        pre_avail_claim = f"Pre-order availability status for LEGO Pokémon sets: {lp.preorder_available}."
    await evaluator.verify(
        claim=pre_avail_claim,
        node=pre_avail_leaf,
        sources=lp.preorder_reference_urls,
        additional_instruction="Confirm whether the collection was available for pre-order."
    )

    pre_start_leaf = evaluator.add_leaf(
        id="Pre_Order_Start",
        desc="If pre-orders were available, specify when they began",
        parent=preorder_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"Pre-orders for LEGO Pokémon sets began on {lp.preorder_start_date}.",
        node=pre_start_leaf,
        sources=lp.preorder_reference_urls,
        additional_instruction="Confirm the pre-order start date (if applicable)."
    )

    # Purchase locations (critical)
    purchase_node = evaluator.add_parallel(
        id="Purchase_Locations",
        desc="Identify official purchase locations for LEGO Pokemon sets",
        parent=lego_node,
        critical=True
    )

    purchase_leaf = evaluator.add_leaf(
        id="Official_Retailers",
        desc="Identify official locations where LEGO Pokemon sets can be purchased",
        parent=purchase_node,
        critical=True
    )
    purchase_names = _join_list(lp.purchase_locations)
    await evaluator.verify(
        claim=f"Official purchase locations for LEGO Pokémon sets include: {purchase_names}.",
        node=purchase_leaf,
        sources=lp.purchase_location_urls or lp.purchase_reference_urls,
        additional_instruction="Confirm that the linked pages are official retailer or LEGO pages where the sets can be purchased."
    )

    purchase_ref_leaf = evaluator.add_leaf(
        id="Purchase_Reference_URL",
        desc="Provide reference URL confirming LEGO Pokemon sets purchase locations",
        parent=purchase_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page confirms official purchase locations for the LEGO Pokémon sets.",
        node=purchase_ref_leaf,
        sources=lp.purchase_reference_urls or lp.purchase_location_urls,
        additional_instruction="The page(s) should clearly indicate purchasing availability for the sets."
    )


# =========================
# Main Evaluation Entrypoint
# =========================

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
    # Initialize Evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # categories evaluated independently
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

    # Optional: Create a top-level plan node to mirror rubric root
    plan_node = evaluator.add_parallel(
        id="Holiday_Gift_Shopping_Plan",
        desc="Validate a complete holiday gift shopping plan including product selection, specifications verification, store availability, and purchase timing across three different gift categories",
        parent=root,
        critical=False
    )

    # Extract data for each category (in parallel)
    apple_task = evaluator.extract(
        prompt=prompt_extract_apple_watch(),
        template_class=AppleWatchExtraction,
        extraction_name="apple_watch_series_11"
    )
    switch_task = evaluator.extract(
        prompt=prompt_extract_switch2(),
        template_class=Switch2Extraction,
        extraction_name="nintendo_switch_2"
    )
    lego_task = evaluator.extract(
        prompt=prompt_extract_lego_pokemon(),
        template_class=LegoPokemonExtraction,
        extraction_name="lego_pokemon_sets"
    )

    aw, ns, lp = await asyncio.gather(apple_task, switch_task, lego_task)

    # Build verification subtrees
    await build_smartwatch_verification(evaluator, plan_node, aw)
    await build_switch_verification(evaluator, plan_node, ns)
    await build_lego_verification(evaluator, plan_node, lp)

    # Return summary
    return evaluator.get_summary()