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
TASK_ID = "clicks_products_research"
TASK_DESCRIPTION = (
    "In early January 2026, Clicks Technology announced multiple new physical keyboard products for smartphones at CES 2026. "
    "Research these product announcements and provide detailed specifications for the following three products:\n\n"
    "1. Clicks Communicator (standalone smartphone with built-in physical QWERTY keyboard):\n"
    "   - Display size (in inches) and resolution\n"
    "   - Battery capacity (in mAh) and battery technology type\n"
    "   - Base storage capacity (in GB) and maximum expandable storage capacity via MicroSD\n"
    "   - Rear camera megapixel count and front camera megapixel count\n"
    "   - Reservation pricing options (both deposit amount and full early bird reservation price in USD)\n\n"
    "2. Clicks Power Keyboard (magnetic wireless keyboard accessory):\n"
    "   - Total onboard battery capacity (in mAh)\n"
    "   - Device dimensions (length × width × height in mm) and weight (in grams)\n"
    "   - Bluetooth version specification\n"
    "   - Pre-order early bird price and regular MSRP (both in USD)\n"
    "   - Expected shipping timeframe (season and year)\n\n"
    "3. Clicks Keyboard Cases (wrap-around keyboard cases):\n"
    "   - Complete list of iPhone model generations that are compatible (e.g., iPhone 14, 15, 16, etc.)\n"
    "   - List of specific Android phone models officially compatible (brand and model names)\n"
    "   - Price range for Clicks keyboard cases (minimum and maximum prices in USD)\n\n"
    "For each specification, provide the official source URL from Clicks Technology's website (www.clicks.tech) where this information can be verified."
)


# --------------------------------------------------------------------------- #
# Ground truth expectations (from rubric)                                     #
# --------------------------------------------------------------------------- #
EXPECTED_SPECS = {
    "communicator": {
        "display": {"size_inches": "4.03 inches", "resolution": "1080 x 1200"},
        "battery": {"capacity_mAh": "4000 mAh", "technology": "silicon-carbon"},
        "storage": {"base": "256GB", "max_microsd": "2TB"},
        "camera": {"rear": "50MP (with OIS)", "front": "24MP"},
        "pricing": {"deposit_usd": "$199", "early_bird_usd": "$399"},
    },
    "power_keyboard": {
        "battery": {"capacity_mAh": "2150 mAh"},
        "dimensions": {"dimensions_mm": "119.7 × 76.6 × 15.2mm", "weight_g": "180 grams"},
        "bluetooth": {"version": "BLE 5.4"},
        "pricing": {"early_bird_usd": "$79", "msrp_usd": "$109"},
        "availability": {"shipping": "Spring 2026"},
    },
    "keyboard_cases": {
        "iphone": {"generations": ["iPhone 17", "iPhone 16", "iPhone 15", "iPhone 14"]},
        "android": {
            "models": [
                "Google Pixel 9", "Google Pixel 9 Pro",
                "Samsung Galaxy S25",
                "Motorola Razr+ (2024)", "Motorola Razr (2024)"
            ]
        },
        "pricing": {"min_usd": "$139", "max_usd": "$159"},
    }
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
# We primarily extract the official source URLs for each specification area
# because the leaf verifications will check the answer’s statements against expected
# values and then verify support on clicks.tech via those URLs.

class CommunicatorDisplaySources(BaseModel):
    urls: List[str] = Field(default_factory=list)


class CommunicatorBatterySources(BaseModel):
    urls: List[str] = Field(default_factory=list)


class CommunicatorStorageSources(BaseModel):
    urls: List[str] = Field(default_factory=list)


class CommunicatorCameraSources(BaseModel):
    urls: List[str] = Field(default_factory=list)


class CommunicatorPricingSources(BaseModel):
    urls: List[str] = Field(default_factory=list)


class CommunicatorSources(BaseModel):
    display: Optional[CommunicatorDisplaySources] = None
    battery: Optional[CommunicatorBatterySources] = None
    storage: Optional[CommunicatorStorageSources] = None
    camera: Optional[CommunicatorCameraSources] = None
    pricing: Optional[CommunicatorPricingSources] = None


class PowerKeyboardBatterySources(BaseModel):
    urls: List[str] = Field(default_factory=list)


class PowerKeyboardDimensionsSources(BaseModel):
    urls: List[str] = Field(default_factory=list)


class PowerKeyboardBluetoothSources(BaseModel):
    urls: List[str] = Field(default_factory=list)


class PowerKeyboardPricingSources(BaseModel):
    urls: List[str] = Field(default_factory=list)


class PowerKeyboardAvailabilitySources(BaseModel):
    urls: List[str] = Field(default_factory=list)


class PowerKeyboardSources(BaseModel):
    battery: Optional[PowerKeyboardBatterySources] = None
    dimensions: Optional[PowerKeyboardDimensionsSources] = None
    bluetooth: Optional[PowerKeyboardBluetoothSources] = None
    pricing: Optional[PowerKeyboardPricingSources] = None
    availability: Optional[PowerKeyboardAvailabilitySources] = None


class KeyboardCasesiPhoneSources(BaseModel):
    urls: List[str] = Field(default_factory=list)


class KeyboardCasesAndroidSources(BaseModel):
    urls: List[str] = Field(default_factory=list)


class KeyboardCasesPricingSources(BaseModel):
    urls: List[str] = Field(default_factory=list)


class KeyboardCasesSources(BaseModel):
    iphone: Optional[KeyboardCasesiPhoneSources] = None
    android: Optional[KeyboardCasesAndroidSources] = None
    pricing: Optional[KeyboardCasesPricingSources] = None


class ClicksProductsExtraction(BaseModel):
    communicator: Optional[CommunicatorSources] = None
    power_keyboard: Optional[PowerKeyboardSources] = None
    keyboard_cases: Optional[KeyboardCasesSources] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_clicks_sources() -> str:
    return """
    Extract the official source URLs from the answer for each specification area below. Only include URLs explicitly present in the answer text, and prioritize URLs from clicks.tech (the official site). If multiple URLs are provided, include all of them. If no URL is given for an area, return an empty list for that area.

    Organize the URLs as follows:

    communicator:
      display.urls: URLs that support the Clicks Communicator display size and resolution
      battery.urls: URLs that support the Clicks Communicator battery capacity and technology type
      storage.urls: URLs that support the Clicks Communicator base storage and maximum expandable storage (MicroSD)
      camera.urls: URLs that support the Clicks Communicator rear/front camera specifications
      pricing.urls: URLs that support the Clicks Communicator reservation pricing (deposit and early bird price)

    power_keyboard:
      battery.urls: URLs for the onboard battery capacity
      dimensions.urls: URLs for device dimensions (mm) and weight (grams)
      bluetooth.urls: URLs for the Bluetooth version
      pricing.urls: URLs for pre-order early bird price and MSRP
      availability.urls: URLs for the expected shipping timeframe

    keyboard_cases:
      iphone.urls: URLs for iPhone generations compatibility
      android.urls: URLs for Android models compatibility
      pricing.urls: URLs for the keyboard cases price range

    Rules:
    - Extract only URLs explicitly shown in the answer (plain text or markdown links). Do not invent URLs.
    - Prefer URLs from clicks.tech; if the answer includes non-clicks.tech sources, still include them, but we will verify primarily against clicks.tech.
    - If a URL lacks a protocol, prepend http://.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _filter_clicks_urls(urls: List[str]) -> List[str]:
    """Return URLs that appear to be from Clicks Technology official site."""
    return [u for u in urls if isinstance(u, str) and ("clicks.tech" in u.lower())]


def _has_clicks_source(urls: Optional[List[str]]) -> bool:
    """Check existence of at least one clicks.tech URL."""
    if not urls:
        return False
    return len(_filter_clicks_urls(urls)) > 0


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_communicator(evaluator: Evaluator, parent_node, sources: Optional[CommunicatorSources]) -> None:
    product_node = evaluator.add_parallel(
        id="Clicks_Communicator",
        desc="Verify specifications for the Clicks Communicator standalone smartphone with built-in physical QWERTY keyboard",
        parent=parent_node,
        critical=False
    )

    # Display
    display_node = evaluator.add_parallel(
        id="Display_Specification",
        desc="Verify the Clicks Communicator display size is 4.03 inches and resolution is 1080 x 1200 pixels, with supporting URL from clicks.tech",
        parent=product_node,
        critical=False
    )
    display_urls = sources.display.urls if (sources and sources.display) else []
    evaluator.add_custom_node(
        result=_has_clicks_source(display_urls),
        id="communicator_display_sources_exist",
        desc="Clicks Communicator display spec has official clicks.tech source URL(s) provided",
        parent=display_node,
        critical=True
    )
    display_match = evaluator.add_leaf(
        id="communicator_display_value_match",
        desc="Answer states Communicator display is 4.03 inches and 1080 x 1200 resolution",
        parent=display_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states the Clicks Communicator display size is 4.03 inches and the resolution is 1080 x 1200 pixels.",
        node=display_match,
        additional_instruction="Judge based on the answer text. Allow minor formatting differences like '1080x1200' vs '1080 x 1200'. Orientation reversal should not be considered a match."
    )
    display_supported = evaluator.add_leaf(
        id="communicator_display_source_supported",
        desc="Official clicks.tech page supports Communicator 4.03-inch display and 1080 x 1200 resolution",
        parent=display_node,
        critical=True
    )
    await evaluator.verify(
        claim="Clicks Communicator has a 4.03-inch display and a resolution of 1080 x 1200 pixels.",
        node=display_supported,
        sources=_filter_clicks_urls(display_urls),
        additional_instruction="Verify this claim using the official Clicks Technology page(s). Minor formatting differences are acceptable."
    )

    # Battery
    battery_node = evaluator.add_parallel(
        id="Communicator_Battery_Specification",
        desc="Verify the Clicks Communicator battery capacity is 4,000 mAh and technology type is silicon-carbon, with supporting URL from clicks.tech",
        parent=product_node,
        critical=False
    )
    battery_urls = sources.battery.urls if (sources and sources.battery) else []
    evaluator.add_custom_node(
        result=_has_clicks_source(battery_urls),
        id="communicator_battery_sources_exist",
        desc="Clicks Communicator battery spec has official clicks.tech source URL(s) provided",
        parent=battery_node,
        critical=True
    )
    battery_match = evaluator.add_leaf(
        id="communicator_battery_value_match",
        desc="Answer states Communicator battery is 4,000 mAh and silicon-carbon technology",
        parent=battery_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states the Clicks Communicator battery capacity is 4,000 mAh and uses silicon-carbon technology.",
        node=battery_match,
        additional_instruction="Judge based on the answer text; allow minor formatting variations like '4000 mAh' vs '4,000 mAh'."
    )
    battery_supported = evaluator.add_leaf(
        id="communicator_battery_source_supported",
        desc="Official clicks.tech page supports Communicator 4,000 mAh silicon-carbon battery",
        parent=battery_node,
        critical=True
    )
    await evaluator.verify(
        claim="Clicks Communicator has a 4,000 mAh battery that uses silicon-carbon technology.",
        node=battery_supported,
        sources=_filter_clicks_urls(battery_urls),
        additional_instruction="Confirm both capacity and the battery technology type are explicitly supported on clicks.tech."
    )

    # Storage
    storage_node = evaluator.add_parallel(
        id="Communicator_Storage_Specification",
        desc="Verify the Clicks Communicator base storage is 256GB and maximum expandable storage via MicroSD is 2TB, with supporting URL from clicks.tech",
        parent=product_node,
        critical=False
    )
    storage_urls = sources.storage.urls if (sources and sources.storage) else []
    evaluator.add_custom_node(
        result=_has_clicks_source(storage_urls),
        id="communicator_storage_sources_exist",
        desc="Clicks Communicator storage spec has official clicks.tech source URL(s) provided",
        parent=storage_node,
        critical=True
    )
    storage_match = evaluator.add_leaf(
        id="communicator_storage_value_match",
        desc="Answer states Communicator base storage is 256GB and max MicroSD is 2TB",
        parent=storage_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states the Clicks Communicator base storage is 256GB and the maximum MicroSD expandable storage is 2TB.",
        node=storage_match,
        additional_instruction="Judge based on the answer text; allow minor formatting differences (e.g., '2 TB' vs '2TB')."
    )
    storage_supported = evaluator.add_leaf(
        id="communicator_storage_source_supported",
        desc="Official clicks.tech page supports Communicator 256GB base and up to 2TB MicroSD",
        parent=storage_node,
        critical=True
    )
    await evaluator.verify(
        claim="Clicks Communicator provides 256GB base storage and supports up to 2TB expandable storage via MicroSD.",
        node=storage_supported,
        sources=_filter_clicks_urls(storage_urls),
        additional_instruction="Confirm both base storage and the maximum MicroSD capacity are explicitly mentioned on clicks.tech."
    )

    # Camera
    camera_node = evaluator.add_parallel(
        id="Communicator_Camera_Specification",
        desc="Verify the Clicks Communicator rear camera is 50MP (with OIS) and front camera is 24MP, with supporting URL from clicks.tech",
        parent=product_node,
        critical=False
    )
    camera_urls = sources.camera.urls if (sources and sources.camera) else []
    evaluator.add_custom_node(
        result=_has_clicks_source(camera_urls),
        id="communicator_camera_sources_exist",
        desc="Clicks Communicator camera spec has official clicks.tech source URL(s) provided",
        parent=camera_node,
        critical=True
    )
    camera_match = evaluator.add_leaf(
        id="communicator_camera_value_match",
        desc="Answer states Communicator rear camera is 50MP (with OIS) and front camera is 24MP",
        parent=camera_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states the Clicks Communicator rear camera is 50MP with OIS and the front camera is 24MP.",
        node=camera_match,
        additional_instruction="Judge based on the answer text; recognize 'OIS' as optical image stabilization."
    )
    camera_supported = evaluator.add_leaf(
        id="communicator_camera_source_supported",
        desc="Official clicks.tech page supports Communicator 50MP OIS rear and 24MP front cameras",
        parent=camera_node,
        critical=True
    )
    await evaluator.verify(
        claim="Clicks Communicator has a 50MP rear camera with OIS and a 24MP front camera.",
        node=camera_supported,
        sources=_filter_clicks_urls(camera_urls),
        additional_instruction="Confirm both megapixel counts and OIS on the rear camera are explicitly stated on clicks.tech."
    )

    # Pricing
    pricing_node = evaluator.add_parallel(
        id="Communicator_Pricing_Specification",
        desc="Verify the Clicks Communicator reservation options are $199 USD deposit and $399 USD full early bird reservation, with supporting URL from clicks.tech",
        parent=product_node,
        critical=False
    )
    pricing_urls = sources.pricing.urls if (sources and sources.pricing) else []
    evaluator.add_custom_node(
        result=_has_clicks_source(pricing_urls),
        id="communicator_pricing_sources_exist",
        desc="Clicks Communicator pricing has official clicks.tech source URL(s) provided",
        parent=pricing_node,
        critical=True
    )
    pricing_match = evaluator.add_leaf(
        id="communicator_pricing_value_match",
        desc="Answer states Communicator reservation deposit is $199 and early bird full reservation price is $399",
        parent=pricing_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states the Clicks Communicator reservation requires a $199 deposit and the full early bird reservation price is $399.",
        node=pricing_match,
        additional_instruction="Judge based on the answer text; currency formatting variations like 'USD' are acceptable."
    )
    pricing_supported = evaluator.add_leaf(
        id="communicator_pricing_source_supported",
        desc="Official clicks.tech page supports Communicator $199 deposit and $399 early bird",
        parent=pricing_node,
        critical=True
    )
    await evaluator.verify(
        claim="Clicks Communicator reservations include a $199 deposit and a $399 full early bird reservation price.",
        node=pricing_supported,
        sources=_filter_clicks_urls(pricing_urls),
        additional_instruction="Confirm both deposit and early bird pricing are explicitly supported on clicks.tech."
    )


async def verify_power_keyboard(evaluator: Evaluator, parent_node, sources: Optional[PowerKeyboardSources]) -> None:
    product_node = evaluator.add_parallel(
        id="Clicks_Power_Keyboard",
        desc="Verify specifications for the Clicks Power Keyboard magnetic wireless keyboard accessory",
        parent=parent_node,
        critical=False
    )

    # Battery
    battery_node = evaluator.add_parallel(
        id="Power_Keyboard_Battery_Specification",
        desc="Verify the Clicks Power Keyboard onboard battery capacity is 2150 mAh, with supporting URL from clicks.tech",
        parent=product_node,
        critical=False
    )
    battery_urls = sources.battery.urls if (sources and sources.battery) else []
    evaluator.add_custom_node(
        result=_has_clicks_source(battery_urls),
        id="power_keyboard_battery_sources_exist",
        desc="Clicks Power Keyboard battery spec has official clicks.tech source URL(s) provided",
        parent=battery_node,
        critical=True
    )
    battery_match = evaluator.add_leaf(
        id="power_keyboard_battery_value_match",
        desc="Answer states Power Keyboard onboard battery capacity is 2150 mAh",
        parent=battery_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states the Clicks Power Keyboard onboard battery capacity is 2150 mAh.",
        node=battery_match
    )
    battery_supported = evaluator.add_leaf(
        id="power_keyboard_battery_source_supported",
        desc="Official clicks.tech page supports Power Keyboard 2150 mAh battery",
        parent=battery_node,
        critical=True
    )
    await evaluator.verify(
        claim="Clicks Power Keyboard has an onboard battery capacity of 2150 mAh.",
        node=battery_supported,
        sources=_filter_clicks_urls(battery_urls)
    )

    # Dimensions
    dimensions_node = evaluator.add_parallel(
        id="Power_Keyboard_Dimensions_Specification",
        desc="Verify the Clicks Power Keyboard dimensions are 119.7 × 76.6 × 15.2mm and weight is 180 grams, with supporting URL from clicks.tech",
        parent=product_node,
        critical=False
    )
    dimensions_urls = sources.dimensions.urls if (sources and sources.dimensions) else []
    evaluator.add_custom_node(
        result=_has_clicks_source(dimensions_urls),
        id="power_keyboard_dimensions_sources_exist",
        desc="Clicks Power Keyboard dimensions/weight have official clicks.tech source URL(s)",
        parent=dimensions_node,
        critical=True
    )
    dimensions_match = evaluator.add_leaf(
        id="power_keyboard_dimensions_value_match",
        desc="Answer states device dimensions are 119.7 × 76.6 × 15.2mm and weight is 180 grams",
        parent=dimensions_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states the Clicks Power Keyboard dimensions are 119.7 × 76.6 × 15.2mm and its weight is 180 grams.",
        node=dimensions_match,
        additional_instruction="Allow minor variations in the multiplication sign '×' vs 'x' and spacing."
    )
    dimensions_supported = evaluator.add_leaf(
        id="power_keyboard_dimensions_source_supported",
        desc="Official clicks.tech page supports 119.7 × 76.6 × 15.2mm and 180 grams",
        parent=dimensions_node,
        critical=True
    )
    await evaluator.verify(
        claim="Clicks Power Keyboard dimensions are 119.7 × 76.6 × 15.2mm and weight is 180 grams.",
        node=dimensions_supported,
        sources=_filter_clicks_urls(dimensions_urls)
    )

    # Bluetooth
    bt_node = evaluator.add_parallel(
        id="Power_Keyboard_Bluetooth_Specification",
        desc="Verify the Clicks Power Keyboard uses BLE 5.4 (Bluetooth Low Energy version 5.4), with supporting URL from clicks.tech",
        parent=product_node,
        critical=False
    )
    bt_urls = sources.bluetooth.urls if (sources and sources.bluetooth) else []
    evaluator.add_custom_node(
        result=_has_clicks_source(bt_urls),
        id="power_keyboard_bt_sources_exist",
        desc="Clicks Power Keyboard Bluetooth spec has official clicks.tech source URL(s)",
        parent=bt_node,
        critical=True
    )
    bt_match = evaluator.add_leaf(
        id="power_keyboard_bt_value_match",
        desc="Answer states Power Keyboard uses BLE 5.4",
        parent=bt_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states the Clicks Power Keyboard uses BLE 5.4 (Bluetooth Low Energy 5.4).",
        node=bt_match
    )
    bt_supported = evaluator.add_leaf(
        id="power_keyboard_bt_source_supported",
        desc="Official clicks.tech page supports BLE 5.4 on Power Keyboard",
        parent=bt_node,
        critical=True
    )
    await evaluator.verify(
        claim="Clicks Power Keyboard uses BLE 5.4 (Bluetooth Low Energy 5.4).",
        node=bt_supported,
        sources=_filter_clicks_urls(bt_urls)
    )

    # Pricing
    pricing_node = evaluator.add_parallel(
        id="Power_Keyboard_Pricing_Specification",
        desc="Verify the Clicks Power Keyboard early bird pre-order price is $79 USD and regular MSRP is $109 USD, with supporting URL from clicks.tech",
        parent=product_node,
        critical=False
    )
    pricing_urls = sources.pricing.urls if (sources and sources.pricing) else []
    evaluator.add_custom_node(
        result=_has_clicks_source(pricing_urls),
        id="power_keyboard_pricing_sources_exist",
        desc="Clicks Power Keyboard pricing has official clicks.tech source URL(s)",
        parent=pricing_node,
        critical=True
    )
    pricing_match = evaluator.add_leaf(
        id="power_keyboard_pricing_value_match",
        desc="Answer states Power Keyboard early bird is $79 and MSRP is $109",
        parent=pricing_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states the Clicks Power Keyboard early bird pre-order price is $79 and the regular MSRP is $109.",
        node=pricing_match
    )
    pricing_supported = evaluator.add_leaf(
        id="power_keyboard_pricing_source_supported",
        desc="Official clicks.tech page supports $79 early bird and $109 MSRP for Power Keyboard",
        parent=pricing_node,
        critical=True
    )
    await evaluator.verify(
        claim="Clicks Power Keyboard has an early bird pre-order price of $79 and a regular MSRP of $109.",
        node=pricing_supported,
        sources=_filter_clicks_urls(pricing_urls)
    )

    # Availability
    availability_node = evaluator.add_parallel(
        id="Power_Keyboard_Availability_Specification",
        desc="Verify the Clicks Power Keyboard is scheduled to ship in Spring 2026, with supporting URL from clicks.tech",
        parent=product_node,
        critical=False
    )
    availability_urls = sources.availability.urls if (sources and sources.availability) else []
    evaluator.add_custom_node(
        result=_has_clicks_source(availability_urls),
        id="power_keyboard_availability_sources_exist",
        desc="Clicks Power Keyboard availability has official clicks.tech source URL(s)",
        parent=availability_node,
        critical=True
    )
    availability_match = evaluator.add_leaf(
        id="power_keyboard_availability_value_match",
        desc="Answer states Power Keyboard ships in Spring 2026",
        parent=availability_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states the Clicks Power Keyboard is scheduled to ship in Spring 2026.",
        node=availability_match
    )
    availability_supported = evaluator.add_leaf(
        id="power_keyboard_availability_source_supported",
        desc="Official clicks.tech page supports Power Keyboard shipping in Spring 2026",
        parent=availability_node,
        critical=True
    )
    await evaluator.verify(
        claim="Clicks Power Keyboard is scheduled to ship in Spring 2026.",
        node=availability_supported,
        sources=_filter_clicks_urls(availability_urls)
    )


async def verify_keyboard_cases(evaluator: Evaluator, parent_node, sources: Optional[KeyboardCasesSources]) -> None:
    product_node = evaluator.add_parallel(
        id="Clicks_Keyboard_Cases",
        desc="Verify specifications for Clicks wrap-around keyboard cases",
        parent=parent_node,
        critical=False
    )

    # iPhone compatibility
    iphone_node = evaluator.add_parallel(
        id="Keyboard_Cases_iPhone_Compatibility",
        desc="Verify Clicks keyboard cases are compatible with iPhone generations 17, 16, 15, and 14 (including all variants: standard, Plus, Pro, Pro Max), with supporting URL from clicks.tech",
        parent=product_node,
        critical=False
    )
    iphone_urls = sources.iphone.urls if (sources and sources.iphone) else []
    evaluator.add_custom_node(
        result=_has_clicks_source(iphone_urls),
        id="keyboard_cases_iphone_sources_exist",
        desc="Clicks keyboard cases iPhone compatibility has official clicks.tech source URL(s)",
        parent=iphone_node,
        critical=True
    )
    iphone_match = evaluator.add_leaf(
        id="keyboard_cases_iphone_value_match",
        desc="Answer states iPhone generations 17, 16, 15, and 14 are compatible",
        parent=iphone_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that Clicks keyboard cases are compatible with iPhone generations 17, 16, 15, and 14.",
        node=iphone_match,
        additional_instruction="High-level generation coverage is sufficient; explicit listing of all variants is not required for this check."
    )
    iphone_supported = evaluator.add_leaf(
        id="keyboard_cases_iphone_source_supported",
        desc="Official clicks.tech page supports iPhone 17/16/15/14 compatibility (incl. standard, Plus, Pro, Pro Max variants)",
        parent=iphone_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "Clicks keyboard cases are compatible with iPhone 17, 16, 15, and 14 generations, including the standard, Plus, Pro, and Pro Max variants."
        ),
        node=iphone_supported,
        sources=_filter_clicks_urls(iphone_urls)
    )

    # Android compatibility
    android_node = evaluator.add_parallel(
        id="Keyboard_Cases_Android_Compatibility",
        desc="Verify Clicks keyboard cases for Android are compatible with Google Pixel 9/9 Pro, Samsung Galaxy S25, and Motorola Razr+ (2024) and Razr (2024), with supporting URL from clicks.tech",
        parent=product_node,
        critical=False
    )
    android_urls = sources.android.urls if (sources and sources.android) else []
    evaluator.add_custom_node(
        result=_has_clicks_source(android_urls),
        id="keyboard_cases_android_sources_exist",
        desc="Clicks keyboard cases Android compatibility has official clicks.tech source URL(s)",
        parent=android_node,
        critical=True
    )
    android_match = evaluator.add_leaf(
        id="keyboard_cases_android_value_match",
        desc="Answer states Android compatibility includes Pixel 9/9 Pro, Galaxy S25, Razr+ (2024), Razr (2024)",
        parent=android_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that Android compatibility includes Google Pixel 9 and 9 Pro, Samsung Galaxy S25, and Motorola Razr+ (2024) and Razr (2024).",
        node=android_match
    )
    android_supported = evaluator.add_leaf(
        id="keyboard_cases_android_source_supported",
        desc="Official clicks.tech page supports Android compatibility list (Pixel 9/9 Pro, Galaxy S25, Razr+ 2024, Razr 2024)",
        parent=android_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "Clicks keyboard cases for Android are compatible with Google Pixel 9 and 9 Pro, Samsung Galaxy S25, and Motorola Razr+ (2024) and Razr (2024)."
        ),
        node=android_supported,
        sources=_filter_clicks_urls(android_urls)
    )

    # Pricing range
    pricing_node = evaluator.add_parallel(
        id="Keyboard_Cases_Pricing_Range",
        desc="Verify Clicks keyboard cases are priced from $139 USD (for standard models) to $159 USD (for Plus/Pro Max models), with supporting URL from clicks.tech",
        parent=product_node,
        critical=False
    )
    kc_pricing_urls = sources.pricing.urls if (sources and sources.pricing) else []
    evaluator.add_custom_node(
        result=_has_clicks_source(kc_pricing_urls),
        id="keyboard_cases_pricing_sources_exist",
        desc="Clicks keyboard cases pricing range has official clicks.tech source URL(s)",
        parent=pricing_node,
        critical=True
    )
    kc_pricing_match = evaluator.add_leaf(
        id="keyboard_cases_pricing_value_match",
        desc="Answer states keyboard cases priced from $139 to $159",
        parent=pricing_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that Clicks keyboard cases are priced from $139 to $159.",
        node=kc_pricing_match
    )
    kc_pricing_supported = evaluator.add_leaf(
        id="keyboard_cases_pricing_source_supported",
        desc="Official clicks.tech page supports keyboard cases priced $139 (standard) to $159 (Plus/Pro Max)",
        parent=pricing_node,
        critical=True
    )
    await evaluator.verify(
        claim="Clicks keyboard cases are priced from $139 USD (standard models) to $159 USD (Plus/Pro Max models).",
        node=kc_pricing_supported,
        sources=_filter_clicks_urls(kc_pricing_urls)
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
    Evaluate an answer for the Clicks Technology products research and verification task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates independently across three products
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

    # Extract official source URLs from the answer for each specification area
    extracted_sources = await evaluator.extract(
        prompt=prompt_extract_clicks_sources(),
        template_class=ClicksProductsExtraction,
        extraction_name="clicks_sources_extraction"
    )

    # Add ground truth expectations to summary for transparency
    evaluator.add_ground_truth({
        "expected": EXPECTED_SPECS,
        "note": "Expected values are derived from the rubric and should be supported by official Clicks Technology (clicks.tech) sources."
    })

    # Build verification tree
    main_node = evaluator.add_parallel(
        id="Clicks_Technology_Products_Research",
        desc="Research and verify detailed specifications for three physical keyboard products announced by Clicks Technology in January 2026",
        parent=root,
        critical=False
    )

    # Verify Communicator
    await verify_communicator(evaluator, main_node, extracted_sources.communicator or CommunicatorSources())

    # Verify Power Keyboard
    await verify_power_keyboard(evaluator, main_node, extracted_sources.power_keyboard or PowerKeyboardSources())

    # Verify Keyboard Cases
    await verify_keyboard_cases(evaluator, main_node, extracted_sources.keyboard_cases or KeyboardCasesSources())

    # Return structured result
    return evaluator.get_summary()