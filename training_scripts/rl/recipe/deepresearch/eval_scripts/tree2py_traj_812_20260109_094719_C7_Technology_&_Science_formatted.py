import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "flagship_smartphone_2024_2026"
TASK_DESCRIPTION = (
    "Identify a flagship smartphone released between 2024 and 2026 that meets all of the following technical specifications: "
    "Must use a current-generation flagship processor (such as Snapdragon 8 Elite series, Apple A19 Pro, Google Tensor G5, or equivalent flagship-grade chipset); "
    "Must have at least 12 GB of RAM; Must offer at least a 512 GB storage configuration option; Must feature an LTPO OLED or equivalent adaptive refresh rate display; "
    "Display resolution must be at least 1200 pixels on the shorter dimension; Must support 120Hz display refresh rate; Primary rear camera must be at least 48 MP; "
    "Must include a telephoto camera with at least 3x optical zoom; Battery capacity must be at least 4500 mAh; Must support at least 25W wired fast charging; "
    "Must support wireless charging; Must have IP68 water and dust resistance rating or better; Must run a current-generation operating system (Android 15/16 or iOS 26); "
    "Must support 5G network connectivity. Provide the model name, manufacturer, and specific technical specifications that satisfy each requirement, along with reference URLs "
    "from official manufacturer websites or reputable technology review sites (such as GSMArena) that verify each specification."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SmartphoneSpecs(BaseModel):
    """Structured extraction of the selected smartphone and its key specs"""
    model_name: Optional[str] = None
    manufacturer: Optional[str] = None
    release_info: Optional[str] = None  # e.g., "Released September 2025", "Announced 2024"
    processor: Optional[str] = None     # e.g., "Snapdragon 8 Elite", "Apple A19 Pro", "Google Tensor G5"
    ram: Optional[str] = None           # e.g., "12 GB", "16GB"
    storage_options: List[str] = Field(default_factory=list)  # e.g., ["256GB", "512GB", "1TB"]
    display_tech: Optional[str] = None  # e.g., "LTPO OLED", "LTPO AMOLED", "Super Retina XDR (LTPO)"
    resolution: Optional[str] = None    # e.g., "1440 x 3120", "1290 x 2796"
    refresh_rate: Optional[str] = None  # e.g., "120Hz", "adaptive up to 120Hz"
    main_camera: Optional[str] = None   # e.g., "48 MP", "50MP"
    telephoto_camera: Optional[str] = None  # e.g., "3x optical zoom", "5x optical zoom"
    battery_capacity: Optional[str] = None  # e.g., "5000 mAh"
    wired_charging: Optional[str] = None    # e.g., "45W wired", "USB PD 30W"
    wireless_charging: Optional[str] = None # e.g., "Qi wireless charging 15W", "MagSafe"
    ip_rating: Optional[str] = None         # e.g., "IP68", "IP68 dust/water resistant"
    os_version: Optional[str] = None        # e.g., "Android 15", "iOS 26"
    connectivity: Optional[str] = None      # e.g., "5G", "5G SA/NSA"
    reference_urls: List[str] = Field(default_factory=list)  # List of URLs in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_phone_specs() -> str:
    return """
    From the provided answer, extract the single smartphone model proposed to meet the task requirements and its key specifications exactly as stated in the answer. Also extract the reference URLs provided in the answer.

    Return a JSON object matching this schema:
    {
      "model_name": string|null,
      "manufacturer": string|null,
      "release_info": string|null,
      "processor": string|null,
      "ram": string|null,
      "storage_options": string[] (list all storage configurations explicitly mentioned),
      "display_tech": string|null,
      "resolution": string|null,                // Keep the full resolution string exactly as written (e.g., "1290 x 2796")
      "refresh_rate": string|null,
      "main_camera": string|null,
      "telephoto_camera": string|null,
      "battery_capacity": string|null,
      "wired_charging": string|null,
      "wireless_charging": string|null,
      "ip_rating": string|null,
      "os_version": string|null,
      "connectivity": string|null,
      "reference_urls": string[]               // Only include URLs explicitly present in the answer; accept plain URLs or markdown links
    }

    Rules:
    - Do not invent any data. If an item is not present in the answer, set it to null (or empty list for storage_options/reference_urls).
    - Preserve units and names exactly as written (e.g., "5000 mAh", "120Hz").
    - For storage_options, include every capacity explicitly listed for the model (e.g., "512GB", "1TB").
    - For reference_urls, extract only valid-looking URLs or markdown link targets. If the answer mentions a source without a URL, do not include it.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def phone_label(specs: SmartphoneSpecs) -> str:
    """Return a friendly phone label for claims."""
    if specs.manufacturer and specs.model_name:
        return f"{specs.manufacturer} {specs.model_name}"
    if specs.model_name:
        return specs.model_name
    if specs.manufacturer:
        return f"{specs.manufacturer} smartphone"
    return "the selected smartphone model"

def urls_text(urls: List[str]) -> str:
    if not urls:
        return "None"
    return "; ".join(urls)

# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, specs: SmartphoneSpecs) -> None:
    """
    Build the verification tree per rubric:
    - Create a critical parallel node "Identified_Flagship_Smartphone"
    - Add critical leaf/custom checks for each requirement
    - Create nested critical parallel node for "Verification_Sources"
    - The node "References_Verify_All_Listed_Requirements" is computed after spec verification leaves
    """
    # Root-level critical node aggregating all requirements
    smartphone_node = evaluator.add_parallel(
        id="Identified_Flagship_Smartphone",
        desc="A smartphone solution that satisfies all explicitly stated requirements and provides identifying info and verification sources.",
        parent=evaluator.root,
        critical=True
    )

    # ----- Identification fields (existence checks) -----
    evaluator.add_custom_node(
        result=(specs.model_name is not None and specs.model_name.strip() != ""),
        id="Model_Name_Provided",
        desc="The solution provides the specific model name of the smartphone.",
        parent=smartphone_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=(specs.manufacturer is not None and specs.manufacturer.strip() != ""),
        id="Manufacturer_Provided",
        desc="The solution provides the manufacturer name of the smartphone.",
        parent=smartphone_node,
        critical=True
    )

    # ----- Verification sources node (critical parallel) -----
    sources_node = evaluator.add_parallel(
        id="Verification_Sources",
        desc="The solution provides reference URLs that are acceptable and that substantiate the required specifications.",
        parent=smartphone_node,
        critical=True
    )

    # 1) Acceptable sources check (simple verification based on domains list)
    acceptable_sources_leaf = evaluator.add_leaf(
        id="Reference_URLs_From_Acceptable_Sources",
        desc="Reference URLs are from official manufacturer sites or reputable technology review/specification sources (e.g., GSMArena).",
        parent=sources_node,
        critical=True
    )
    acceptable_claim = (
        f"These URLs are acceptable references (official manufacturer or reputable tech review/specification sites like GSMArena):\n"
        f"{urls_text(specs.reference_urls)}"
    )
    await evaluator.verify(
        claim=acceptable_claim,
        node=acceptable_sources_leaf,
        additional_instruction=(
            "Judge acceptability by domain names only. Accept official manufacturer domains (e.g., apple.com, samsung.com, "
            "oneplus.com, google.com/pixel, motorola.com), and reputable specification/review sites such as gsmarena.com, "
            "phonearena.com, androidauthority.com, theverge.com, tomsguide.com, notebookcheck.net, etc."
        ),
    )

    # ----- Spec verification leaves (all critical) -----
    phone = phone_label(specs)
    refs = specs.reference_urls  # may be empty, verify_by_urls will attempt all

    spec_leaf_nodes: Dict[str, Any] = {}

    # Release window 2024–2026
    release_leaf = evaluator.add_leaf(
        id="Release_Window_2024_2026",
        desc="The solution identifies a smartphone released between 2024 and 2026 (inclusive).",
        parent=smartphone_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{phone} was released (or made available) between 2024 and 2026 (inclusive).",
        node=release_leaf,
        sources=refs,
        additional_instruction="Check the release or availability date on the provided sources; announcements alone are acceptable if they correspond to retail availability in that window."
    )
    spec_leaf_nodes["Release_Window_2024_2026"] = release_leaf

    # Processor generation (flagship 2024–2026)
    processor_leaf = evaluator.add_leaf(
        id="Processor_Generation",
        desc="The smartphone uses a current-generation flagship processor (Snapdragon 8 Elite series, Apple A19 Pro, Google Tensor G5, or equivalent flagship processor from 2024–2026).",
        parent=smartphone_node,
        critical=True
    )
    proc_str = specs.processor or "a current-generation flagship SoC"
    await evaluator.verify(
        claim=f"{phone} is powered by {proc_str}, which is a current-generation flagship chipset for 2024–2026.",
        node=processor_leaf,
        sources=refs,
        additional_instruction=(
            "Verify the SoC name on the page. Treat Snapdragon 8 Elite / 8 Gen (latest), Apple A19 Pro, Google Tensor G5, "
            "or similarly positioned 2024–2026 flagship chipsets (including equivalent Exynos flagships) as valid."
        )
    )
    spec_leaf_nodes["Processor_Generation"] = processor_leaf

    # RAM ≥ 12 GB
    ram_leaf = evaluator.add_leaf(
        id="RAM_Capacity",
        desc="The smartphone has at least 12 GB of RAM.",
        parent=smartphone_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{phone} offers at least 12 GB of RAM.",
        node=ram_leaf,
        sources=refs,
        additional_instruction="Accept any config with RAM ≥ 12 GB (e.g., 12GB, 16GB, 24GB)."
    )
    spec_leaf_nodes["RAM_Capacity"] = ram_leaf

    # Storage option ≥ 512 GB
    storage_leaf = evaluator.add_leaf(
        id="Storage_Option",
        desc="The smartphone offers at least a 512 GB storage configuration option.",
        parent=smartphone_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{phone} has at least one storage configuration of 512 GB or higher (e.g., 512GB or 1TB).",
        node=storage_leaf,
        sources=refs,
        additional_instruction="Look for storage options; accept 512GB, 1TB, or higher capacities."
    )
    spec_leaf_nodes["Storage_Option"] = storage_leaf

    # Display technology: LTPO OLED or equivalent adaptive refresh
    disptech_leaf = evaluator.add_leaf(
        id="Display_Technology",
        desc="The smartphone features LTPO OLED or equivalent adaptive refresh rate display technology.",
        parent=smartphone_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{phone} features LTPO OLED (or equivalent LTPO adaptive display technology).",
        node=disptech_leaf,
        sources=refs,
        additional_instruction="Accept variations like LTPO OLED/AMOLED or equivalent adaptive refresh technologies mentioned."
    )
    spec_leaf_nodes["Display_Technology"] = disptech_leaf

    # Display resolution: shorter dimension ≥ 1200 px
    resolution_leaf = evaluator.add_leaf(
        id="Display_Resolution",
        desc="The display resolution has at least 1200 pixels on the shorter dimension.",
        parent=smartphone_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The display resolution of {phone} has at least 1200 pixels on its shorter side.",
        node=resolution_leaf,
        sources=refs,
        additional_instruction=(
            "Use the resolution listed to compute the shorter side (e.g., for 1290 x 2796, shorter is 1290). "
            "Pass if the shorter side ≥ 1200."
        )
    )
    spec_leaf_nodes["Display_Resolution"] = resolution_leaf

    # Refresh rate: 120Hz support
    refreshrate_leaf = evaluator.add_leaf(
        id="Refresh_Rate",
        desc="The display supports a 120Hz refresh rate.",
        parent=smartphone_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{phone} supports a 120Hz display refresh rate.",
        node=refreshrate_leaf,
        sources=refs,
        additional_instruction="Look for explicit 120Hz support; adaptive up to 120Hz is acceptable."
    )
    spec_leaf_nodes["Refresh_Rate"] = refreshrate_leaf

    # Main camera ≥ 48 MP
    maincam_leaf = evaluator.add_leaf(
        id="Main_Camera_Resolution",
        desc="The primary rear camera has at least 48 MP resolution.",
        parent=smartphone_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The primary rear camera on {phone} is at least 48 MP.",
        node=maincam_leaf,
        sources=refs,
        additional_instruction="Accept 48MP, 50MP, or greater for the main camera module."
    )
    spec_leaf_nodes["Main_Camera_Resolution"] = maincam_leaf

    # Telephoto camera ≥ 3x optical zoom
    tele_leaf = evaluator.add_leaf(
        id="Telephoto_Camera",
        desc="The smartphone includes a telephoto camera with at least 3x optical zoom capability.",
        parent=smartphone_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{phone} includes a telephoto camera with at least 3x optical zoom.",
        node=tele_leaf,
        sources=refs,
        additional_instruction="Optical zoom of 3x or higher (e.g., 3x, 5x) passes; digital zoom alone is not sufficient."
    )
    spec_leaf_nodes["Telephoto_Camera"] = tele_leaf

    # Battery capacity ≥ 4500 mAh
    battery_leaf = evaluator.add_leaf(
        id="Battery_Capacity",
        desc="The battery capacity is at least 4500 mAh.",
        parent=smartphone_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{phone} has a battery capacity of at least 4500 mAh.",
        node=battery_leaf,
        sources=refs,
        additional_instruction="Accept capacities ≥ 4500 mAh."
    )
    spec_leaf_nodes["Battery_Capacity"] = battery_leaf

    # Wired fast charging ≥ 25W
    wired_leaf = evaluator.add_leaf(
        id="Fast_Charging_Wired",
        desc="The smartphone supports at least 25W wired fast charging.",
        parent=smartphone_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{phone} supports at least 25W wired fast charging.",
        node=wired_leaf,
        sources=refs,
        additional_instruction="Look for stated wattage for wired charging; pass if ≥ 25W."
    )
    spec_leaf_nodes["Fast_Charging_Wired"] = wired_leaf

    # Wireless charging support
    wireless_leaf = evaluator.add_leaf(
        id="Wireless_Charging",
        desc="The smartphone supports wireless charging functionality.",
        parent=smartphone_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{phone} supports wireless charging.",
        node=wireless_leaf,
        sources=refs,
        additional_instruction="Any standard wireless charging (e.g., Qi, MagSafe) is acceptable."
    )
    spec_leaf_nodes["Wireless_Charging"] = wireless_leaf

    # Water resistance: IP68 or better
    ip_leaf = evaluator.add_leaf(
        id="Water_Resistance",
        desc="The smartphone has an IP68 water and dust resistance rating or better.",
        parent=smartphone_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{phone} is rated IP68 or better for water and dust resistance.",
        node=ip_leaf,
        sources=refs,
        additional_instruction="Look for 'IP68' or higher (e.g., IP68/69) protection."
    )
    spec_leaf_nodes["Water_Resistance"] = ip_leaf

    # OS: Android 15/16 or iOS 26
    os_leaf = evaluator.add_leaf(
        id="Operating_System",
        desc="The smartphone runs a current-generation operating system (Android 15/16 or iOS 26).",
        parent=smartphone_node,
        critical=True
    )
    os_str = specs.os_version or "current-generation OS"
    await evaluator.verify(
        claim=f"{phone} runs {os_str}, which counts as a current-generation OS (Android 15/16 or iOS 26).",
        node=os_leaf,
        sources=refs,
        additional_instruction="Confirm OS version on the page; accept Android 15/16 or iOS 26."
    )
    spec_leaf_nodes["Operating_System"] = os_leaf

    # 5G connectivity
    fiveg_leaf = evaluator.add_leaf(
        id="5G_Connectivity",
        desc="The smartphone supports 5G network connectivity.",
        parent=smartphone_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{phone} supports 5G connectivity.",
        node=fiveg_leaf,
        sources=refs,
        additional_instruction="Look for explicit 5G support; SA/NSA bands, sub-6 or mmWave are acceptable."
    )
    spec_leaf_nodes["5G_Connectivity"] = fiveg_leaf

    # 2) References collectively verify all listed requirements
    # Compute after spec leaves executed: pass only if all spec leaves passed.
    all_specs_passed = all(node.status == "passed" for node in spec_leaf_nodes.values())
    evaluator.add_custom_node(
        result=all_specs_passed,
        id="References_Verify_All_Listed_Requirements",
        desc="Provided references collectively verify each listed required specification/feature (processor, RAM, storage option, display tech, resolution, refresh rate, cameras, battery, charging, IP rating, OS, and 5G).",
        parent=sources_node,
        critical=True
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
    Evaluate an answer for the flagship smartphone specification task.

    Returns a structured evaluation summary including the verification tree and aggregated score.
    """
    # Initialize evaluator
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

    # Extract structured smartphone specs and URLs
    specs = await evaluator.extract(
        prompt=prompt_extract_phone_specs(),
        template_class=SmartphoneSpecs,
        extraction_name="smartphone_specs"
    )

    # Optional: record constraints into the summary
    evaluator.add_custom_info(
        {
            "release_window": "2024–2026 inclusive",
            "required_specs": [
                "Flagship processor (Snapdragon 8 Elite / Apple A19 Pro / Google Tensor G5 / equivalent)",
                "≥ 12 GB RAM",
                "≥ 512 GB storage option",
                "LTPO OLED or equivalent adaptive display tech",
                "Shorter display dimension ≥ 1200 px",
                "120Hz refresh rate",
                "Main camera ≥ 48 MP",
                "Telephoto camera with ≥ 3x optical zoom",
                "Battery ≥ 4500 mAh",
                "Wired fast charging ≥ 25W",
                "Wireless charging support",
                "IP68 rating or better",
                "Current-gen OS (Android 15/16 or iOS 26)",
                "5G connectivity"
            ]
        },
        info_type="constraints",
        info_name="task_requirements"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, specs)

    # Return final summary
    return evaluator.get_summary()