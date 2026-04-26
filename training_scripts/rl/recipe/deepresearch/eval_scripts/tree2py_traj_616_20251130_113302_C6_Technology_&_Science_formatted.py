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
TASK_ID = "ipad_a16_report"
TASK_DESCRIPTION = (
    "A business technology consultant in California is preparing a comprehensive product research report for a client "
    "who wants to purchase iPad tablets with the A16 chip for their organization's employees. The consultant needs to "
    "compile the following information:\n\n"
    "Product Details:\n"
    "- Official model name, generation number, release date, and chip designation\n"
    "- Complete display specifications: screen size, resolution in pixels, pixel density (ppi), and display type\n"
    "- Processor architecture: CPU core count and GPU core count\n"
    "- Battery capacity in watt-hours\n"
    "- Primary wireless connectivity standard (Wi-Fi generation)\n\n"
    "Purchasing Information:\n"
    "- All available storage capacity options (in GB)\n"
    "- Official Apple retail price for each storage tier in Wi-Fi-only models\n"
    "- Official Apple retail price for each storage tier in Wi-Fi + Cellular models\n"
    "- Available color options\n\n"
    "California Retail Availability:\n"
    "- The exact number of Best Buy store locations in California\n"
    "- Names of at least two other major retailers where this iPad model can be purchased\n\n"
    "Provide all the requested information with supporting reference URLs from official Apple pages or reliable "
    "retail/industry sources for verification."
)

# Expected values according to rubric
EXPECTED = {
    "official_model_name": "11-inch iPad",
    "generation": "11th generation",
    "release_date": "March 12, 2025",
    "chip": "A16",
    "display": {
        "screen_size": "11-inch",
        "display_type": "Liquid Retina",
        "resolution": "2360-by-1640",
        "pixel_density": "264 ppi",
    },
    "processor": {
        "cpu_cores": "5-core",
        "gpu_cores": "4-core",
    },
    "battery": {
        "capacity_wh": "28.93 watt-hours",
        "battery_type": "rechargeable lithium-polymer",
    },
    "wifi": {
        "standard": "Wi‑Fi 6 (802.11ax)",
        "mimo": "2x2 MIMO",
    },
    "purchasing": {
        "storage_options": ["128GB", "256GB", "512GB"],
        "wifi_only_prices": {
            "gb_128": "$349",
            "gb_256": "$449",
            "gb_512": "$649",
        },
        "wifi_cellular_prices": {
            "gb_128": "$499",
            "gb_256": "$599",
            "gb_512": "$799",
        },
        "colors": ["Silver", "Blue", "Pink", "Yellow"],
    },
    "retail": {
        "bestbuy_ca_store_count": "146",
    }
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProductDetails(BaseModel):
    official_model_name: Optional[str] = None
    generation: Optional[str] = None
    release_date: Optional[str] = None
    chip: Optional[str] = None


class DisplaySpecs(BaseModel):
    screen_size: Optional[str] = None
    display_type: Optional[str] = None
    resolution: Optional[str] = None
    pixel_density: Optional[str] = None


class ProcessorSpecs(BaseModel):
    cpu_cores: Optional[str] = None
    gpu_cores: Optional[str] = None


class BatterySpecs(BaseModel):
    capacity_wh: Optional[str] = None
    battery_type: Optional[str] = None


class WifiSpecs(BaseModel):
    standard: Optional[str] = None
    mimo: Optional[str] = None


class PriceTiers(BaseModel):
    gb_128: Optional[str] = None
    gb_256: Optional[str] = None
    gb_512: Optional[str] = None


class PurchasingInfo(BaseModel):
    storage_options: List[str] = Field(default_factory=list)
    wifi_only_prices: PriceTiers = PriceTiers()
    wifi_cellular_prices: PriceTiers = PriceTiers()
    colors: List[str] = Field(default_factory=list)


class RetailInfo(BaseModel):
    bestbuy_ca_store_count: Optional[str] = None
    other_retailers: List[str] = Field(default_factory=list)


class SourceURLs(BaseModel):
    apple_urls: List[str] = Field(default_factory=list)
    retail_urls: List[str] = Field(default_factory=list)


class IpadA16ReportExtraction(BaseModel):
    product_details: ProductDetails = ProductDetails()
    display: DisplaySpecs = DisplaySpecs()
    processor: ProcessorSpecs = ProcessorSpecs()
    battery: BatterySpecs = BatterySpecs()
    wifi: WifiSpecs = WifiSpecs()
    purchasing: PurchasingInfo = PurchasingInfo()
    retail: RetailInfo = RetailInfo()
    sources: SourceURLs = SourceURLs()


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_report() -> str:
    return (
        "Extract structured information from the answer for the requested iPad report. "
        "Return JSON with the following top-level keys: product_details, display, processor, battery, wifi, "
        "purchasing, retail, sources.\n\n"
        "product_details:\n"
        "- official_model_name: exact model name string as stated (e.g., \"11-inch iPad\")\n"
        "- generation: generation string as stated (e.g., \"11th generation\")\n"
        "- release_date: exact release date string as stated (e.g., \"March 12, 2025\")\n"
        "- chip: chip designation string as stated (e.g., \"A16\")\n\n"
        "display:\n"
        "- screen_size: exact string (e.g., \"11-inch\")\n"
        "- display_type: exact string (e.g., \"Liquid Retina\")\n"
        "- resolution: exact string (e.g., \"2360-by-1640\")\n"
        "- pixel_density: exact string (e.g., \"264 ppi\")\n\n"
        "processor:\n"
        "- cpu_cores: exact string (e.g., \"5-core\")\n"
        "- gpu_cores: exact string (e.g., \"4-core\")\n\n"
        "battery:\n"
        "- capacity_wh: exact string including unit (e.g., \"28.93 watt-hours\")\n"
        "- battery_type: exact string (e.g., \"rechargeable lithium-polymer\")\n\n"
        "wifi:\n"
        "- standard: exact Wi‑Fi generation string (e.g., \"Wi‑Fi 6 (802.11ax)\")\n"
        "- mimo: exact MIMO spec if stated (e.g., \"2x2 MIMO\")\n\n"
        "purchasing:\n"
        "- storage_options: list of storage option strings (e.g., [\"128GB\",\"256GB\",\"512GB\"]) exactly as stated\n"
        "- wifi_only_prices: object with gb_128, gb_256, gb_512 as price strings (e.g., \"$349\")\n"
        "- wifi_cellular_prices: object with gb_128, gb_256, gb_512 as price strings (e.g., \"$499\")\n"
        "- colors: list of color strings exactly as stated\n\n"
        "retail:\n"
        "- bestbuy_ca_store_count: exact string number as stated (e.g., \"146\")\n"
        "- other_retailers: list of retailer names (exclude Best Buy)\n\n"
        "sources:\n"
        "- apple_urls: all official Apple URLs mentioned in the answer\n"
        "- retail_urls: all other reliable retail/industry source URLs mentioned (e.g., bestbuy.com, target.com)\n\n"
        "Rules:\n"
        "- Extract only what is explicitly present in the answer; do not invent.\n"
        "- Use strings for numbers and prices; preserve formatting and units.\n"
        "- If an item is missing, set it to null or [] as appropriate.\n"
        "- For URLs, extract actual full URLs; if missing protocol, prepend http://."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _list_nonempty(lst: Optional[List[str]]) -> bool:
    return bool(lst and len(lst) > 0)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_supporting_references(
    evaluator: Evaluator,
    parent_node,
    ext: IpadA16ReportExtraction,
) -> None:
    refs_node = evaluator.add_parallel(
        id="Supporting_References",
        desc="Includes supporting reference URLs from official Apple pages and/or reliable retail/industry sources that substantively support the report’s claims.",
        parent=parent_node,
        critical=True
    )

    # Apple_Source_URLs leaf: verify that at least one provided URL is an official Apple page
    apple_leaf = evaluator.add_leaf(
        id="Apple_Source_URLs",
        desc="Provides at least one official Apple URL that supports Apple-specific claims in the report (e.g., model identification/specs and/or Apple pricing/configurations).",
        parent=refs_node,
        critical=True
    )
    apple_urls = ext.sources.apple_urls or []
    await evaluator.verify(
        claim="This URL is an official Apple webpage relevant to the iPad model/specs/pricing.",
        node=apple_leaf,
        sources=apple_urls,
        additional_instruction="Pass if the page is clearly from apple.com (Apple official site) and pertains to iPad model/specs/pricing. Minor URL variations are acceptable."
    )

    # Retail_Industry_Source_URLs leaf: verify at least one reliable retail/industry source URL
    retail_leaf = evaluator.add_leaf(
        id="Retail_Industry_Source_URLs",
        desc="Provides at least one reliable retail/industry source URL supporting non-Apple claims (e.g., Best Buy California store count and/or California retail availability statements).",
        parent=refs_node,
        critical=True
    )
    retail_urls = ext.sources.retail_urls or []
    await evaluator.verify(
        claim="This URL is a reliable retail/industry source (e.g., a major retailer or reputable industry site) relevant to availability or store counts.",
        node=retail_leaf,
        sources=retail_urls,
        additional_instruction="Accept pages from major retailers (bestbuy.com, target.com, walmart.com, costco.com, etc.), carrier sites, or reputable industry media. The content should be relevant to availability or store counts."
    )


async def build_product_identification(
    evaluator: Evaluator,
    parent_node,
    ext: IpadA16ReportExtraction,
) -> None:
    prod_node = evaluator.add_parallel(
        id="Product_Identification",
        desc="Correctly identifies the required iPad model per constraints.",
        parent=parent_node,
        critical=True
    )

    # Existence gate
    prod_exists = evaluator.add_custom_node(
        result=(
            _nonempty(ext.product_details.official_model_name)
            and _nonempty(ext.product_details.generation)
            and _nonempty(ext.product_details.release_date)
            and _nonempty(ext.product_details.chip)
        ),
        id="Product_Identification_Provided",
        desc="Product identification details are provided (name, generation, release date, chip).",
        parent=prod_node,
        critical=True
    )

    # Correct_Model_And_Release leaf
    model_leaf = evaluator.add_leaf(
        id="Correct_Model_And_Release",
        desc="Identifies the model as the 11-inch iPad (A16), 11th generation, released on March 12, 2025, and states the chip designation as A16.",
        parent=prod_node,
        critical=True
    )
    claim_model = (
        "The iPad model is the 11-inch iPad (A16), 11th generation; it was released on March 12, 2025; "
        "the chip designation is A16."
    )
    await evaluator.verify(
        claim=claim_model,
        node=model_leaf,
        sources=ext.sources.apple_urls or [],
        additional_instruction="Verify against Apple's official pages; allow minor naming variants (e.g., punctuation or hyphenation), but the substance must match exactly."
    )


async def build_technical_specifications(
    evaluator: Evaluator,
    parent_node,
    ext: IpadA16ReportExtraction,
) -> None:
    tech_node = evaluator.add_parallel(
        id="Technical_Specifications",
        desc="Provides the required technical specifications per constraints.",
        parent=parent_node,
        critical=True
    )

    # Existence gate for specs
    tech_exists = evaluator.add_custom_node(
        result=(
            _nonempty(ext.display.screen_size)
            and _nonempty(ext.display.display_type)
            and _nonempty(ext.display.resolution)
            and _nonempty(ext.display.pixel_density)
            and _nonempty(ext.processor.cpu_cores)
            and _nonempty(ext.processor.gpu_cores)
            and _nonempty(ext.battery.capacity_wh)
            and _nonempty(ext.battery.battery_type)
            and _nonempty(ext.wifi.standard)
        ),
        id="Technical_Specs_Provided",
        desc="Technical specifications are provided (display, CPU/GPU cores, battery capacity/type, Wi‑Fi standard).",
        parent=tech_node,
        critical=True
    )

    # Display_Specifications leaf
    display_leaf = evaluator.add_leaf(
        id="Display_Specifications",
        desc="States display size, display type, resolution, and pixel density matching: 11-inch Liquid Retina, 2360-by-1640 pixels, 264 ppi.",
        parent=tech_node,
        critical=True
    )
    claim_display = (
        "The display is 11-inch Liquid Retina with a resolution of 2360-by-1640 pixels and a pixel density of 264 ppi."
    )
    await evaluator.verify(
        claim=claim_display,
        node=display_leaf,
        sources=ext.sources.apple_urls or [],
        additional_instruction="Use Apple's tech specs page(s) for the iPad model; accept minor formatting variants."
    )

    # Processor_Architecture leaf
    processor_leaf = evaluator.add_leaf(
        id="Processor_Architecture",
        desc="States CPU core count and GPU core count matching: 5-core CPU and 4-core GPU.",
        parent=tech_node,
        critical=True
    )
    claim_cpu_gpu = "The processor architecture has a 5-core CPU and a 4-core GPU."
    await evaluator.verify(
        claim=claim_cpu_gpu,
        node=processor_leaf,
        sources=ext.sources.apple_urls or [],
        additional_instruction="Confirm the core counts from Apple's official pages for this model/chip; minor wording differences are acceptable."
    )

    # Battery_Capacity leaf
    battery_leaf = evaluator.add_leaf(
        id="Battery_Capacity",
        desc="States battery capacity matching: 28.93 watt-hours (rechargeable lithium-polymer).",
        parent=tech_node,
        critical=True
    )
    claim_battery = "The battery capacity is 28.93 watt-hours and the type is rechargeable lithium-polymer."
    await evaluator.verify(
        claim=claim_battery,
        node=battery_leaf,
        sources=ext.sources.apple_urls or [],
        additional_instruction="Use Apple's official tech specs; accept minor hyphenation or punctuation variants."
    )

    # WiFi_Standard leaf
    wifi_leaf = evaluator.add_leaf(
        id="WiFi_Standard",
        desc="States primary Wi‑Fi standard matching: Wi‑Fi 6 (802.11ax) with 2x2 MIMO.",
        parent=tech_node,
        critical=True
    )
    claim_wifi = "The primary wireless connectivity is Wi‑Fi 6 (802.11ax) with 2x2 MIMO."
    await evaluator.verify(
        claim=claim_wifi,
        node=wifi_leaf,
        sources=ext.sources.apple_urls or [],
        additional_instruction="Confirm with Apple's official specs for wireless standards; minor formatting differences are acceptable."
    )


async def build_purchasing_information(
    evaluator: Evaluator,
    parent_node,
    ext: IpadA16ReportExtraction,
) -> None:
    purchase_node = evaluator.add_parallel(
        id="Purchasing_Information",
        desc="Provides storage options, Apple retail pricing by tier for Wi‑Fi and Wi‑Fi+Cellular, and colors per constraints.",
        parent=parent_node,
        critical=True
    )

    # Existence gate for purchasing info
    purchase_exists = evaluator.add_custom_node(
        result=(
            _list_nonempty(ext.purchasing.storage_options)
            and _nonempty(ext.purchasing.wifi_only_prices.gb_128)
            and _nonempty(ext.purchasing.wifi_only_prices.gb_256)
            and _nonempty(ext.purchasing.wifi_only_prices.gb_512)
            and _nonempty(ext.purchasing.wifi_cellular_prices.gb_128)
            and _nonempty(ext.purchasing.wifi_cellular_prices.gb_256)
            and _nonempty(ext.purchasing.wifi_cellular_prices.gb_512)
            and _list_nonempty(ext.purchasing.colors)
        ),
        id="Purchasing_Info_Provided",
        desc="Purchasing information is provided (storage options, prices by tier for Wi‑Fi and Wi‑Fi+Cellular, colors).",
        parent=purchase_node,
        critical=True
    )

    # Storage_Options leaf
    storage_leaf = evaluator.add_leaf(
        id="Storage_Options",
        desc="Lists all available storage options: 128GB, 256GB, and 512GB.",
        parent=purchase_node,
        critical=True
    )
    claim_storage = "The storage options are 128GB, 256GB, and 512GB."
    await evaluator.verify(
        claim=claim_storage,
        node=storage_leaf,
        sources=ext.sources.apple_urls or [],
        additional_instruction="Confirm storage capacities from Apple's product/config pages."
    )

    # Apple_Pricing_WiFi_Only leaf
    wifi_only_leaf = evaluator.add_leaf(
        id="Apple_Pricing_WiFi_Only",
        desc="Provides official Apple retail prices for Wi‑Fi-only models for each tier: 128GB $349, 256GB $449, 512GB $649.",
        parent=purchase_node,
        critical=True
    )
    claim_wifi_only = (
        "The official Apple retail prices for Wi‑Fi-only models are: 128GB $349, 256GB $449, and 512GB $649."
    )
    await evaluator.verify(
        claim=claim_wifi_only,
        node=wifi_only_leaf,
        sources=ext.sources.apple_urls or [],
        additional_instruction="Verify prices on Apple's Store/product page; ignore temporary promotions/trade-in."
    )

    # Apple_Pricing_WiFi_Plus_Cellular leaf
    wifi_cell_leaf = evaluator.add_leaf(
        id="Apple_Pricing_WiFi_Plus_Cellular",
        desc="Provides official Apple retail prices for Wi‑Fi + Cellular models for each tier: 128GB $499, 256GB $599, 512GB $799.",
        parent=purchase_node,
        critical=True
    )
    claim_wifi_cell = (
        "The official Apple retail prices for Wi‑Fi + Cellular models are: 128GB $499, 256GB $599, and 512GB $799."
    )
    await evaluator.verify(
        claim=claim_wifi_cell,
        node=wifi_cell_leaf,
        sources=ext.sources.apple_urls or [],
        additional_instruction="Verify prices on Apple's Store/product page; ignore temporary promotions/trade-in."
    )

    # Color_Options leaf
    colors_leaf = evaluator.add_leaf(
        id="Color_Options",
        desc="Lists available colors: Silver, Blue, Pink, and Yellow.",
        parent=purchase_node,
        critical=True
    )
    claim_colors = "The available color options are Silver, Blue, Pink, and Yellow."
    await evaluator.verify(
        claim=claim_colors,
        node=colors_leaf,
        sources=ext.sources.apple_urls or [],
        additional_instruction="Confirm color options from Apple's product page; allow minor naming variants."
    )


async def build_california_availability(
    evaluator: Evaluator,
    parent_node,
    ext: IpadA16ReportExtraction,
) -> None:
    ca_node = evaluator.add_parallel(
        id="California_Retail_Availability",
        desc="Provides California-specific retail availability info per constraints.",
        parent=parent_node,
        critical=True
    )

    # Existence gate for CA retail info
    ca_exists = evaluator.add_custom_node(
        result=(_nonempty(ext.retail.bestbuy_ca_store_count) and len(ext.retail.other_retailers) >= 2),
        id="CA_Retail_Info_Provided",
        desc="California retail availability info is provided (Best Buy CA store count and at least two other major retailers).",
        parent=ca_node,
        critical=True
    )

    # BestBuy_CA_Store_Count leaf
    bb_leaf = evaluator.add_leaf(
        id="BestBuy_CA_Store_Count",
        desc="States the exact number of Best Buy store locations in California as 146.",
        parent=ca_node,
        critical=True
    )
    claim_bb = "There are 146 Best Buy store locations in California."
    await evaluator.verify(
        claim=claim_bb,
        node=bb_leaf,
        sources=ext.sources.retail_urls or [],
        additional_instruction="Verify via Best Buy store locator or other reputable sources; the number must match exactly."
    )

    # Other_Retailers leaf
    or_leaf = evaluator.add_leaf(
        id="Other_Retailers",
        desc="Names at least two other (non–Best Buy) major retailers where the specified iPad model can be purchased.",
        parent=ca_node,
        critical=True
    )
    retailers_list = ext.retail.other_retailers or []
    claim_other = (
        f"The answer names at least two other major retailers (excluding Best Buy) for purchasing this iPad model. "
        f"Extracted names: {retailers_list}."
    )
    await evaluator.verify(
        claim=claim_other,
        node=or_leaf,
        sources=None,
        additional_instruction="Pass if the answer includes at least two recognizable major U.S. retailers other than Best Buy (e.g., Apple, Target, Walmart, Amazon, Costco, carriers, Staples)."
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
    Evaluate the iPad A16 research report answer against the rubric using Mind2Web2.
    """
    # Initialize evaluator
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
        default_model=model,
    )

    # Create critical task node under root to mirror rubric root
    report_node = evaluator.add_parallel(
        id="iPad_A16_Research_Report",
        desc="Complete product research report for the specified iPad model, including required specs, pricing/configurations, California retail availability, and supporting reference URLs.",
        parent=root,
        critical=True
    )

    # Extract structured info from the answer
    ext = await evaluator.extract(
        prompt=prompt_extract_report(),
        template_class=IpadA16ReportExtraction,
        extraction_name="ipad_a16_report_extraction"
    )

    # Add ground truth expectations to summary
    evaluator.add_ground_truth({
        "expected": EXPECTED,
        "notes": "All critical leaves must match these expected values per rubric; verification should be supported by official Apple pages or reputable retail/industry sources."
    })

    # Build and verify subtrees.
    # Verify supporting references first so other verifications can auto-skip on failed prerequisite.
    await build_supporting_references(evaluator, report_node, ext)
    await build_product_identification(evaluator, report_node, ext)
    await build_technical_specifications(evaluator, report_node, ext)
    await build_purchasing_information(evaluator, report_node, ext)
    await build_california_availability(evaluator, report_node, ext)

    # Return structured result summary
    return evaluator.get_summary()