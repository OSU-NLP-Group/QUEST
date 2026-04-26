import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "macbook_march_2026_aseries"
TASK_DESCRIPTION = (
    "In March 2026, Apple announced a new MacBook model at an unprecedented starting price point "
    "for the Mac lineup. This model uses a chip from Apple's iPhone processor line rather than "
    "the M-series chips typically found in Mac computers.\n\n"
    "Identify this MacBook model and provide the following information:\n\n"
    "1. The official announcement date and the date when the product became available for purchase (began shipping to customers)\n"
    "2. The specific chip powering this MacBook, including its exact CPU core configuration (how many performance cores and how many efficiency cores)\n"
    "3. A detailed description of both USB-C ports on this MacBook, including:\n"
    "   - The data transfer speed capabilities of each port (which port supports which speed)\n"
    "   - Which port (left or right, when viewing from the left side of the device) supports external display connectivity\n"
    "   - The maximum external display resolution and refresh rate supported\n"
    "4. The built-in display specifications, including screen size, resolution (in pixels), and brightness (in nits)\n\n"
    "For each piece of information, provide a reference URL from Apple's official website (apple.com) that supports your answer."
)

# Expected values (from rubric)
EXPECTED_ANNOUNCEMENT_DATE = "March 4, 2026"
EXPECTED_AVAILABILITY_DATE = "March 11, 2026"

EXPECTED_CHIP_NAME = "Apple A18 Pro"
EXPECTED_CPU_CONFIG_TEXT = "6-core CPU with 2 performance cores and 4 efficiency cores"  # normalized sentence form

EXPECTED_LEFT_PORT_SPEED_PHRASE = "usb 3 up to 10 gb/s"   # normalized phrase (we'll fuzzy-check)
EXPECTED_RIGHT_PORT_SPEED_PHRASE = "usb 2 up to 480 mb/s"

EXPECTED_EXT_DISPLAY_PORT = "left"
EXPECTED_EXT_DISPLAY_RES_PHRASES = ["4k", "3840×2160", "3840x2160", "3840-2160"]  # accept common variants
EXPECTED_EXT_DISPLAY_REFRESH_PHRASES = ["60hz", "60 hz"]

EXPECTED_DISPLAY_SIZE_PHRASES = ["13-inch", '13″', '13"']  # allow common variants
EXPECTED_DISPLAY_RESOLUTION_DIGITS = ("2408", "1506")
EXPECTED_DISPLAY_BRIGHTNESS_PHRASE = "500 nits"


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_apple_url(url: str) -> bool:
    if not url:
        return False
    u = url.strip().lower()
    return "apple.com" in u and u.startswith(("http://", "https://"))


def filter_apple_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if _is_apple_url(u)]


def norm_text(s: Optional[str]) -> str:
    if not s:
        return ""
    # Normalize: lowercase, replace unicode x, remove extra spaces and punctuation-like chars (keep digits/letters)
    t = s.lower().strip()
    t = t.replace("×", "x").replace("‑", "-").replace("–", "-").replace("—", "-")
    t = re.sub(r"\s+", " ", t)
    return t


def contains_all(text: str, phrases: List[str]) -> bool:
    t = norm_text(text)
    return all(p in t for p in phrases)


def any_contains(text: str, phrases: List[str]) -> bool:
    t = norm_text(text)
    return any(p in t for p in phrases)


def to_left_or_right(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    t = norm_text(s)
    if "left" in t:
        return "left"
    if "right" in t:
        return "right"
    return None


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class ProductInfo(BaseModel):
    model_name: Optional[str] = None
    starting_price_usd: Optional[str] = None
    general_urls: List[str] = Field(default_factory=list)


class AnnouncementInfo(BaseModel):
    announcement_date: Optional[str] = None
    availability_date: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ChipInfo(BaseModel):
    chip_name: Optional[str] = None
    cpu_core_config: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class PortsInfo(BaseModel):
    left_port_speed: Optional[str] = None
    right_port_speed: Optional[str] = None
    external_display_port_side: Optional[str] = None
    external_display_max_resolution: Optional[str] = None
    external_display_max_refresh_rate: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class DisplayInfo(BaseModel):
    display_size_inches: Optional[str] = None
    display_resolution_pixels: Optional[str] = None
    display_brightness_nits: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class MacBookExtraction(BaseModel):
    product: Optional[ProductInfo] = None
    announcement: Optional[AnnouncementInfo] = None
    chip: Optional[ChipInfo] = None
    ports: Optional[PortsInfo] = None
    display: Optional[DisplayInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_macbook_info() -> str:
    return """
    From the answer, extract the structured information for the single MacBook model being described. 
    Return JSON with these top-level objects: product, announcement, chip, ports, display.

    Field details:
    - product.model_name: The exact MacBook model name as stated by Apple (string).
    - product.starting_price_usd: The starting price as written in the answer (e.g., "$599") (string).
    - product.general_urls: Any Apple.com URLs cited that generally refer to the product (array of strings).

    - announcement.announcement_date: The announcement date as written (e.g., "March 4, 2026") (string).
    - announcement.availability_date: The date it became available/began shipping (e.g., "March 11, 2026") (string).
    - announcement.urls: Apple.com URLs that directly support dates (array of strings).

    - chip.chip_name: The specific chip (e.g., "Apple A18 Pro") (string).
    - chip.cpu_core_config: The CPU core configuration as written (e.g., "6-core CPU (2 performance cores and 4 efficiency cores)") (string).
    - chip.urls: Apple.com URLs that support the chip and core configuration (array of strings).

    - ports.left_port_speed: The left USB-C port speed capability as written (e.g., "USB 3 up to 10 Gb/s") (string).
    - ports.right_port_speed: The right USB-C port speed capability as written (e.g., "USB 2 up to 480 Mb/s") (string).
    - ports.external_display_port_side: Which port supports external display connectivity ("left" or "right") (string).
    - ports.external_display_max_resolution: Max external display resolution as written (e.g., "4K" or "3840-by-2160") (string).
    - ports.external_display_max_refresh_rate: Max external display refresh rate as written (e.g., "60Hz") (string).
    - ports.urls: Apple.com URLs that support the port speeds and external display capability (array of strings).

    - display.display_size_inches: Built-in display size as written (e.g., "13-inch") (string).
    - display.display_resolution_pixels: Built-in display resolution as written (e.g., "2408-by-1506") (string).
    - display.display_brightness_nits: Built-in display brightness as written (e.g., "500 nits") (string).
    - display.urls: Apple.com URLs that support the built-in display specs (array of strings).

    Important:
    - Extract exactly what appears in the answer; do not infer or fabricate.
    - For URL fields, include ONLY URLs explicitly present in the answer and ONLY if they are on apple.com.
    - If a required field is missing in the answer, set it to null (or an empty array for URL lists).
    """


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_product_identification_tree(evaluator: Evaluator, parent, data: MacBookExtraction) -> None:
    """
    Product Identification gate (sequential): ensure the answer identifies a MacBook model.
    This acts as a gate before checking detailed specs.
    """
    node = evaluator.add_sequential(
        id="product_identification",
        desc="Correctly identifies a MacBook model announced by Apple in March 2026 with a starting price of $599 USD",
        parent=parent,
        critical=False  # Keep non-critical here to allow mixed scoring downstream; we'll add critical gate leaves inside
    )

    # Critical gate: model identified and is a MacBook
    model_name = (data.product.model_name if data and data.product else None) or ""
    model_is_macbook = "macbook" in norm_text(model_name)
    evaluator.add_custom_node(
        result=bool(model_name.strip()) and model_is_macbook,
        id="product_identified_gate",
        desc="The answer identifies a specific MacBook model",
        parent=node,
        critical=True
    )


async def build_dates_tree(evaluator: Evaluator, parent, data: MacBookExtraction) -> None:
    ann = data.announcement if data else None
    ann_urls_all = filter_apple_urls(ann.urls if ann else [])

    # Container for dates info
    ann_info_node = evaluator.add_sequential(
        id="announcement_and_availability_information",
        desc="Provides announcement and availability dates for the product",
        parent=parent,
        critical=False
    )

    # A critical subnode that groups checks for dates
    dates_group = evaluator.add_sequential(
        id="announcement_and_availability_dates",
        desc="Provides correct announcement date (March 4, 2026) and availability start date (March 11, 2026)",
        parent=ann_info_node,
        critical=True
    )

    # 1) Value match against expected (custom, critical)
    ann_date = (ann.announcement_date if ann else None) or ""
    avail_date = (ann.availability_date if ann else None) or ""
    match_expected = (norm_text(ann_date) == norm_text(EXPECTED_ANNOUNCEMENT_DATE)) and (
        norm_text(avail_date) == norm_text(EXPECTED_AVAILABILITY_DATE)
    )
    evaluator.add_custom_node(
        result=match_expected,
        id="dates_match_expected",
        desc=f"Announcement date is '{EXPECTED_ANNOUNCEMENT_DATE}' and availability date is '{EXPECTED_AVAILABILITY_DATE}'",
        parent=dates_group,
        critical=True
    )

    # 2) Apple.com reference provided (custom, critical)
    evaluator.add_custom_node(
        result=len(ann_urls_all) > 0,
        id="reference_url_announcement_present",
        desc="Provides a valid apple.com URL supporting the announcement and availability",
        parent=dates_group,
        critical=True
    )

    # 3) Dates supported by the provided Apple URL(s) (critical)
    dates_supported_leaf = evaluator.add_leaf(
        id="dates_supported_by_urls",
        desc="Announcement and availability dates are supported by the cited Apple URL(s)",
        parent=dates_group,
        critical=True
    )
    claim_dates = (
        f"The product was announced on {EXPECTED_ANNOUNCEMENT_DATE}, and became available/began shipping on {EXPECTED_AVAILABILITY_DATE}."
    )
    await evaluator.verify(
        claim=claim_dates,
        node=dates_supported_leaf,
        sources=ann_urls_all,
        additional_instruction="Verify the dates on the cited Apple page(s). Minor wording differences are acceptable as long as the dates match exactly."
    )


async def build_chip_tree(evaluator: Evaluator, parent, data: MacBookExtraction) -> None:
    chip = data.chip if data else None
    chip_urls = filter_apple_urls(chip.urls if chip else [])

    chip_node = evaluator.add_sequential(
        id="chip_information",
        desc="Provides chip specifications for the product",
        parent=parent,
        critical=False
    )

    chip_arch_group = evaluator.add_sequential(
        id="chip_architecture",
        desc="Correctly identifies that the product is powered by Apple A18 Pro chip with 6-core CPU (2 performance cores and 4 efficiency cores)",
        parent=chip_node,
        critical=True
    )

    # 1) Match expected (custom, critical)
    chip_name = (chip.chip_name if chip else None) or ""
    cpu_cfg = (chip.cpu_core_config if chip else None) or ""
    name_ok = EXPECTED_CHIP_NAME.lower() in norm_text(chip_name)
    # Check CPU config: 6-core + 2 performance + 4 efficiency (robust)
    cfg_text = norm_text(cpu_cfg)
    six_core = any(p in cfg_text for p in ["6-core", "6 core", "six-core", "six core"])
    two_perf = any(p in cfg_text for p in ["2 performance", "two performance"])
    four_eff = any(p in cfg_text for p in ["4 efficiency", "four efficiency"])
    cfg_ok = six_core and two_perf and four_eff
    evaluator.add_custom_node(
        result=name_ok and cfg_ok,
        id="chip_match_expected",
        desc="Chip is Apple A18 Pro with a 6-core CPU (2 performance cores and 4 efficiency cores)",
        parent=chip_arch_group,
        critical=True
    )

    # 2) Apple.com reference provided (custom, critical)
    evaluator.add_custom_node(
        result=len(chip_urls) > 0,
        id="reference_url_chip_present",
        desc="Provides a valid apple.com URL supporting the chip specifications",
        parent=chip_arch_group,
        critical=True
    )

    # 3) Supported by Apple URL(s)
    chip_supported_leaf = evaluator.add_leaf(
        id="chip_supported_by_urls",
        desc="Chip name and CPU core configuration are supported by cited Apple URL(s)",
        parent=chip_arch_group,
        critical=True
    )
    claim_chip = (
        f"The MacBook is powered by {EXPECTED_CHIP_NAME} with a 6-core CPU consisting of 2 performance cores and 4 efficiency cores."
    )
    await evaluator.verify(
        claim=claim_chip,
        node=chip_supported_leaf,
        sources=chip_urls,
        additional_instruction="Check the cited Apple page(s) for the chip name and CPU core breakdown; allow minor formatting variations."
    )


async def build_ports_display_tree(evaluator: Evaluator, parent, data: MacBookExtraction) -> None:
    ports = data.ports if data else None
    ports_urls = filter_apple_urls(ports.urls if ports else [])

    top = evaluator.add_sequential(
        id="port_and_display_connectivity_information",
        desc="Provides port specifications and external display connectivity information",
        parent=parent,
        critical=False
    )

    port_cfg = evaluator.add_sequential(
        id="port_configuration",
        desc="Correctly describes both USB-C ports: left port supports USB 3 (up to 10 Gb/s); right port supports USB 2 (up to 480 Mb/s)",
        parent=top,
        critical=True
    )

    # 1) Port speeds match expected (custom, critical)
    left_speed = (ports.left_port_speed if ports else None) or ""
    right_speed = (ports.right_port_speed if ports else None) or ""
    left_ok = ("usb 3" in norm_text(left_speed)) and (("10 gb/s" in norm_text(left_speed)) or ("10gb/s" in norm_text(left_speed)) or ("10gbps" in norm_text(left_speed)) or ("10 gbps" in norm_text(left_speed)))
    right_ok = ("usb 2" in norm_text(right_speed)) and (("480 mb/s" in norm_text(right_speed)) or ("480mb/s" in norm_text(right_speed)) or ("480mbps" in norm_text(right_speed)) or ("480 mbps" in norm_text(right_speed)))
    evaluator.add_custom_node(
        result=left_ok and right_ok,
        id="port_speeds_match_expected",
        desc="Left port: USB 3 up to 10 Gb/s; Right port: USB 2 up to 480 Mb/s",
        parent=port_cfg,
        critical=True
    )

    # 2) External display support group under port configuration
    ext_disp = evaluator.add_sequential(
        id="external_display_support",
        desc="Correctly states that external display connectivity is supported only through the left USB-C port with maximum resolution of 4K at 60Hz",
        parent=port_cfg,
        critical=True
    )

    # 2.1) External display expectations match (custom, critical)
    side = to_left_or_right(ports.external_display_port_side if ports else None)
    res = (ports.external_display_max_resolution if ports else None) or ""
    hz = (ports.external_display_max_refresh_rate if ports else None) or ""
    side_ok = (side == EXPECTED_EXT_DISPLAY_PORT)
    res_ok = any_contains(res, EXPECTED_EXT_DISPLAY_RES_PHRASES)
    hz_ok = any_contains(hz, EXPECTED_EXT_DISPLAY_REFRESH_PHRASES)
    evaluator.add_custom_node(
        result=side_ok and res_ok and hz_ok,
        id="external_display_match_expected",
        desc="External display works only via the left USB-C port, up to 4K at 60Hz",
        parent=ext_disp,
        critical=True
    )

    # 2.2) Apple.com reference provided (custom, critical)
    evaluator.add_custom_node(
        result=len(ports_urls) > 0,
        id="reference_url_ports_and_display_present",
        desc="Provides a valid apple.com URL supporting port and external display specs",
        parent=ext_disp,
        critical=True
    )

    # 2.3) Supported by Apple URL(s) (critical)
    ports_supported_leaf = evaluator.add_leaf(
        id="ports_and_display_supported_by_urls",
        desc="Port speeds and external display support are supported by cited Apple URL(s)",
        parent=ext_disp,
        critical=True
    )
    claim_ports = (
        "This MacBook has two USB-C ports: the left USB-C port supports USB 3 up to 10 Gb/s, "
        "while the right USB-C port supports USB 2 up to 480 Mb/s. External display is supported only "
        "through the left USB-C port, up to 4K at 60Hz."
    )
    await evaluator.verify(
        claim=claim_ports,
        node=ports_supported_leaf,
        sources=ports_urls,
        additional_instruction="Verify both port speed differences and the external display limit (4K@60Hz) using the Apple page(s). Allow minor phrasing differences."
    )


async def build_display_tree(evaluator: Evaluator, parent, data: MacBookExtraction) -> None:
    disp = data.display if data else None
    disp_urls = filter_apple_urls(disp.urls if disp else [])

    disp_node = evaluator.add_sequential(
        id="built_in_display_information",
        desc="Provides built-in display specifications",
        parent=parent,
        critical=False
    )

    specs_group = evaluator.add_sequential(
        id="display_specifications",
        desc="Correctly provides the built-in display specifications: 13-inch Liquid Retina display with 2408-by-1506 resolution and 500 nits brightness",
        parent=disp_node,
        critical=True
    )

    # 1) Match expected (custom, critical)
    size = (disp.display_size_inches if disp else None) or ""
    res = (disp.display_resolution_pixels if disp else None) or ""
    brt = (disp.display_brightness_nits if disp else None) or ""
    size_ok = any(any_contains(size, [p.lower()]) for p in EXPECTED_DISPLAY_SIZE_PHRASES) and ("liquid retina" in norm_text(size) or "liquid retina" in norm_text(res))
    res_ok = (EXPECTED_DISPLAY_RESOLUTION_DIGITS[0] in norm_text(res)) and (EXPECTED_DISPLAY_RESOLUTION_DIGITS[1] in norm_text(res))
    brt_ok = "500" in norm_text(brt) and "nit" in norm_text(brt)
    evaluator.add_custom_node(
        result=size_ok and res_ok and brt_ok,
        id="display_specs_match_expected",
        desc="Built-in display is 13-inch Liquid Retina, 2408-by-1506 resolution, 500 nits brightness",
        parent=specs_group,
        critical=True
    )

    # 2) Apple.com reference provided (custom, critical)
    evaluator.add_custom_node(
        result=len(disp_urls) > 0,
        id="reference_url_display_present",
        desc="Provides a valid apple.com URL supporting built-in display specs",
        parent=specs_group,
        critical=True
    )

    # 3) Supported by Apple URL(s) (critical)
    display_supported_leaf = evaluator.add_leaf(
        id="display_specs_supported_by_urls",
        desc="Built-in display specs are supported by cited Apple URL(s)",
        parent=specs_group,
        critical=True
    )
    claim_disp = (
        "The built-in display is a 13-inch Liquid Retina display with a 2408-by-1506 resolution and 500 nits brightness."
    )
    await evaluator.verify(
        claim=claim_disp,
        node=display_supported_leaf,
        sources=disp_urls,
        additional_instruction="Verify the display size, 'Liquid Retina' branding, resolution digits (2408-by-1506), and brightness (500 nits) on the Apple page(s)."
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
    Evaluate an answer for the March 2026 MacBook (A-series) task.
    """
    # Initialize evaluator with a sequential root to honor gating
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured info
    extraction: MacBookExtraction = await evaluator.extract(
        prompt=prompt_extract_macbook_info(),
        template_class=MacBookExtraction,
        extraction_name="macbook_extraction",
    )

    # Record expected info as ground truth notes (for transparency)
    evaluator.add_ground_truth({
        "expected": {
            "announcement_date": EXPECTED_ANNOUNCEMENT_DATE,
            "availability_date": EXPECTED_AVAILABILITY_DATE,
            "chip": {
                "name": EXPECTED_CHIP_NAME,
                "cpu": EXPECTED_CPU_CONFIG_TEXT
            },
            "ports": {
                "left": "USB 3 up to 10 Gb/s",
                "right": "USB 2 up to 480 Mb/s",
                "external_display": {
                    "port": "left",
                    "max": "4K at 60Hz"
                }
            },
            "display": {
                "size": "13-inch Liquid Retina",
                "resolution": "2408-by-1506",
                "brightness": "500 nits"
            }
        }
    })

    # Build Product Identification (gate)
    await build_product_identification_tree(evaluator, root, extraction)

    # Product Information Set (parallel) - evaluated after gate because root is SEQUENTIAL:
    info_set = evaluator.add_parallel(
        id="product_information_set",
        desc="Provides the required information about the identified MacBook model",
        parent=root,
        critical=False
    )

    # Announcement & Availability
    await build_dates_tree(evaluator, info_set, extraction)

    # Chip
    await build_chip_tree(evaluator, info_set, extraction)

    # Ports & External Display
    await build_ports_display_tree(evaluator, info_set, extraction)

    # Built-in Display
    await build_display_tree(evaluator, info_set, extraction)

    # Return structured evaluation summary
    return evaluator.get_summary()