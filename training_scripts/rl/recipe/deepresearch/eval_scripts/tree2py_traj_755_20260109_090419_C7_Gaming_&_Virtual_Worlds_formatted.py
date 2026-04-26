import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "gaming_monitor_240hz_qhd_27in_2023_2025"
TASK_DESCRIPTION = (
    "Identify a gaming monitor that meets ALL of the following specifications:\n\n"
    "1. Has a refresh rate of exactly 240Hz\n"
    "2. Has a response time of 1ms (GTG - Gray to Gray)\n"
    "3. Uses an IPS panel technology\n"
    "4. Has a resolution of 2560×1440 pixels (QHD/1440p)\n"
    "5. Has a screen size of 27 inches\n"
    "6. Supports NVIDIA G-SYNC Compatible technology\n"
    "7. Supports HDR with VESA DisplayHDR 400 certification\n"
    "8. Has at least one DisplayPort 1.4 connection\n"
    "9. Has at least two HDMI 2.1 input connections\n"
    "10. Includes a stand with at least 100mm of height adjustment range\n"
    "11. Supports VESA mounting with 100mm×100mm pattern\n"
    "12. Covers at least 95% of the DCI-P3 color gamut\n"
    "13. Is manufactured by a company that had a presence at Gamescom 2025\n"
    "14. Was announced or released between 2023 and 2025\n\n"
    "Provide the exact model name and model number of the monitor, along with the manufacturer name."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MonitorSpecs(BaseModel):
    """Structured extraction of the monitor identification and specifications from the answer."""
    # Identification
    model_name: Optional[str] = None
    model_number: Optional[str] = None
    manufacturer: Optional[str] = None

    # Core specs (keep as strings for flexibility)
    refresh_rate: Optional[str] = None
    response_time: Optional[str] = None
    panel_type: Optional[str] = None
    resolution: Optional[str] = None
    screen_size: Optional[str] = None
    gsync_compatible: Optional[str] = None
    hdr_certification: Optional[str] = None

    # Connectivity
    displayport_1_4_count: Optional[str] = None
    hdmi_2_1_count: Optional[str] = None

    # Ergonomics & mounting
    height_adjustment_range_mm: Optional[str] = None
    vesa_mount_pattern: Optional[str] = None

    # Color
    dci_p3_coverage: Optional[str] = None

    # Release timing
    release_or_announce_date: Optional[str] = None
    release_year: Optional[str] = None

    # URLs cited in the answer
    product_page_url: Optional[str] = None
    manufacturer_page_url: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)
    gamescom_2025_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_monitor_specs() -> str:
    return """
    Extract the gaming monitor identification and all listed specifications exactly as stated in the answer.

    Required fields:
    1) model_name: The exact model name (e.g., "ROG Swift PG27AQN")
    2) model_number: The exact model number or SKU (e.g., "PG27AQN")
    3) manufacturer: The brand/manufacturer name (e.g., "ASUS" or "LG")
    4) refresh_rate: The stated refresh rate (e.g., "240Hz")
    5) response_time: The stated response time (e.g., "1ms GTG" or "1 ms (Gray-to-Gray)")
    6) panel_type: The panel technology (e.g., "IPS", "Fast IPS")
    7) resolution: The resolution string (e.g., "2560×1440", "2560 x 1440", "QHD", "1440p")
    8) screen_size: The screen size (e.g., "27 inches", "27\"")
    9) gsync_compatible: The statement about NVIDIA G-SYNC Compatible support (string)
    10) hdr_certification: The HDR certification string (e.g., "VESA DisplayHDR 400", "DisplayHDR 400")
    11) displayport_1_4_count: The stated count for DisplayPort 1.4 inputs (if given; otherwise the phrase as-is)
    12) hdmi_2_1_count: The stated count for HDMI 2.1 inputs (if given; otherwise the phrase as-is)
    13) height_adjustment_range_mm: Stated height adjustment range (e.g., "100mm", "110 mm")
    14) vesa_mount_pattern: The VESA mount pattern string (e.g., "100x100", "100 mm × 100 mm")
    15) dci_p3_coverage: The DCI-P3 coverage statement/value (e.g., "95%", "97% DCI-P3")
    16) release_or_announce_date: A date or month-year string if provided (e.g., "June 2024")
    17) release_year: A year if provided (e.g., "2024")
    18) product_page_url: The official product page URL if provided
    19) manufacturer_page_url: The general manufacturer or support page URL if provided
    20) source_urls: All other URLs cited in the answer that are relevant to this monitor and its specs (list all)
    21) gamescom_2025_urls: Any URL(s) cited that specifically relate to the manufacturer's presence at Gamescom 2025 (exhibitor page, press release, schedule, news)

    Rules:
    - Return null for any field not mentioned.
    - Do not invent information.
    - For URL fields and URL lists, extract complete, valid URLs explicitly present in the answer (including those in markdown link format).
    - Keep all values as strings (do not convert units or numbers).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[Optional[str]]) -> List[str]:
    """Deduplicate while preserving order; ignore None/empty."""
    seen = set()
    ordered: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


def collect_all_sources(specs: MonitorSpecs) -> List[str]:
    """Collect all relevant URLs from the extracted specs."""
    return _dedup_urls(
        (specs.source_urls or [])
        + ([specs.product_page_url] if specs.product_page_url else [])
        + ([specs.manufacturer_page_url] if specs.manufacturer_page_url else [])
        + (specs.gamescom_2025_urls or [])
    )


def identity_string(specs: MonitorSpecs) -> str:
    """Build a human-friendly identity string for claims."""
    mfg = specs.manufacturer or "the manufacturer"
    name = specs.model_name or "the monitor"
    num = specs.model_number or ""
    if num and num.strip():
        return f"{mfg} {name} ({num})"
    return f"{mfg} {name}"


# --------------------------------------------------------------------------- #
# Verification tree builder                                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, specs: MonitorSpecs) -> None:
    """
    Build the verification tree according to the rubric and execute verifications.
    """
    # Create top-level critical parallel node under root
    top_node = evaluator.add_parallel(
        id="Gaming_Monitor_Identification",
        desc="Identifies a gaming monitor that meets all specified technical requirements and provides the required identification details in the answer",
        parent=evaluator.root,
        critical=True
    )

    # Existence checks (critical siblings to gate subsequent verifications)
    has_model_and_number = bool(specs.model_name and specs.model_name.strip()) and bool(specs.model_number and specs.model_number.strip())
    evaluator.add_custom_node(
        result=has_model_and_number,
        id="Provides_Model_Name_And_Number",
        desc="The answer provides the exact model name and model number of the monitor",
        parent=top_node,
        critical=True
    )

    has_manufacturer = bool(specs.manufacturer and specs.manufacturer.strip())
    evaluator.add_custom_node(
        result=has_manufacturer,
        id="Provides_Manufacturer_Name",
        desc="The answer provides the manufacturer name",
        parent=top_node,
        critical=True
    )

    # Collect all cited URLs for verification
    all_urls = collect_all_sources(specs)
    evaluator.add_custom_info(
        info={"urls_used_for_verification": all_urls},
        info_type="url_bundle",
        info_name="urls_used_for_verification"
    )

    # Prepare spec leaf nodes
    id_str = identity_string(specs)

    # 1. Refresh rate exactly 240Hz
    leaf_refresh = evaluator.add_leaf(
        id="Refresh_Rate_240Hz",
        desc="The monitor has a refresh rate of exactly 240Hz",
        parent=top_node,
        critical=True
    )
    claim_refresh = f"The monitor {id_str} supports a refresh rate of exactly 240 Hz (maximum/native 240Hz)."
    add_ins_refresh = (
        "Verify the product page or reliable specifications page explicitly showing 240 Hz. "
        "Minor phrasing variations like 'up to 240Hz' or 'supports 240Hz' count as meeting exactly 240Hz."
    )

    # 2. Response time 1ms GTG
    leaf_response = evaluator.add_leaf(
        id="Response_Time_1ms_GTG",
        desc="The monitor has a response time of 1ms (Gray to Gray)",
        parent=top_node,
        critical=True
    )
    claim_response = f"The monitor {id_str} specifies a 1 ms Gray-to-Gray (GTG) response time."
    add_ins_response = "Look for '1ms GTG' or equivalent wording like '1 ms (G2G)'."

    # 3. IPS panel type
    leaf_ips = evaluator.add_leaf(
        id="IPS_Panel_Type",
        desc="The monitor uses an IPS (In-Plane Switching) panel technology",
        parent=top_node,
        critical=True
    )
    claim_ips = f"The monitor {id_str} uses an IPS panel technology (including variants such as 'Fast IPS')."
    add_ins_ips = "Accept synonyms such as 'IPS', 'Fast IPS', 'Nano IPS'. VA or TN would not satisfy this."

    # 4. Resolution 2560×1440
    leaf_res = evaluator.add_leaf(
        id="Resolution_1440p",
        desc="The monitor has a resolution of 2560×1440 pixels (QHD/1440p)",
        parent=top_node,
        critical=True
    )
    claim_res = f"The monitor {id_str} has a native resolution of 2560×1440 (QHD/1440p)."
    add_ins_res = "Accept formatting variations like '2560 x 1440' and mentions of 'QHD' or '1440p'."

    # 5. Screen size 27 inches
    leaf_size = evaluator.add_leaf(
        id="Screen_Size_27_Inch",
        desc="The monitor has a screen size of 27 inches (diagonal measurement)",
        parent=top_node,
        critical=True
    )
    claim_size = f"The monitor {id_str} has a 27-inch screen (27\" diagonal)."
    add_ins_size = "Accept reasonable variants like '27 inch' or '27-inch class'."

    # 6. NVIDIA G-SYNC Compatible
    leaf_gsync = evaluator.add_leaf(
        id="NVIDIA_G_SYNC_Compatible",
        desc="The monitor supports NVIDIA G-SYNC Compatible technology",
        parent=top_node,
        critical=True
    )
    claim_gsync = f"The monitor {id_str} is NVIDIA G-SYNC Compatible."
    add_ins_gsync = "Look for 'G-SYNC Compatible' on spec sheets or product pages; adaptive sync generic mentions alone are insufficient."

    # 7. HDR DisplayHDR 400
    leaf_hdr = evaluator.add_leaf(
        id="HDR_DisplayHDR_400",
        desc="The monitor supports HDR with VESA DisplayHDR 400 certification",
        parent=top_node,
        critical=True
    )
    claim_hdr = f"The monitor {id_str} carries VESA DisplayHDR 400 certification."
    add_ins_hdr = "Accept 'DisplayHDR 400' or 'VESA DisplayHDR 400'. Generic 'HDR' without 'DisplayHDR 400' does not satisfy this."

    # 8. At least one DisplayPort 1.4
    leaf_dp = evaluator.add_leaf(
        id="DisplayPort_1_4_Connection",
        desc="The monitor has at least one DisplayPort 1.4 input connection",
        parent=top_node,
        critical=True
    )
    claim_dp = f"The monitor {id_str} includes at least one DisplayPort 1.4 input."
    add_ins_dp = "Look for 'DP 1.4' or 'DisplayPort 1.4' in I/O specs."

    # 9. At least two HDMI 2.1
    leaf_hdmi = evaluator.add_leaf(
        id="HDMI_2_1_Dual_Connection",
        desc="The monitor has at least two HDMI 2.1 input connections",
        parent=top_node,
        critical=True
    )
    claim_hdmi = f"The monitor {id_str} provides at least two HDMI 2.1 input ports."
    add_ins_hdmi = "Confirm that there are two or more HDMI ports explicitly labeled 'HDMI 2.1'."

    # 10. Height-adjustable stand ≥ 100mm
    leaf_height = evaluator.add_leaf(
        id="Height_Adjustable_Stand",
        desc="The monitor includes a stand with at least 100mm of height adjustment range",
        parent=top_node,
        critical=True
    )
    claim_height = f"The monitor {id_str} includes a stand with height adjustment of at least 100 mm."
    add_ins_height = "Check ergonomics specs; accept '≥100mm', '100mm', or greater values."

    # 11. VESA mount 100×100
    leaf_vesa = evaluator.add_leaf(
        id="VESA_Mount_100x100",
        desc="The monitor supports VESA mounting with 100mm×100mm pattern",
        parent=top_node,
        critical=True
    )
    claim_vesa = f"The monitor {id_str} supports a 100×100 mm VESA mount pattern."
    add_ins_vesa = "Accept '100x100', '100 mm × 100 mm', or equivalent notation."

    # 12. DCI-P3 ≥ 95%
    leaf_p3 = evaluator.add_leaf(
        id="DCI_P3_95_Percent_Coverage",
        desc="The monitor covers at least 95% of the DCI-P3 color gamut",
        parent=top_node,
        critical=True
    )
    claim_p3 = f"The monitor {id_str} covers at least 95% of the DCI-P3 color gamut."
    add_ins_p3 = "Color coverage must be explicitly DCI-P3; accept values like 95%, 96%, 97%."

    # 13. Manufacturer presence at Gamescom 2025
    leaf_gamescom = evaluator.add_leaf(
        id="Gamescom_2025_Manufacturer",
        desc="The monitor manufacturer had a presence at Gamescom 2025",
        parent=top_node,
        critical=True
    )
    mfg = specs.manufacturer or "The manufacturer"
    claim_gamescom = f"{mfg} had a presence at Gamescom 2025 (e.g., exhibitor listing, booth, official announcement, or press coverage)."
    add_ins_gamescom = (
        "Look for official exhibitor lists, event pages, press releases, or credible news confirming presence at Gamescom 2025. "
        "Presence includes exhibiting, a booth, scheduled showcase, or official partnership."
    )

    # 14. Release/announce period 2023–2025
    leaf_release = evaluator.add_leaf(
        id="Release_Period_2023_2025",
        desc="The monitor was announced or released between 2023 and 2025",
        parent=top_node,
        critical=True
    )
    claim_release = f"The monitor {id_str} was announced or released between January 1, 2023 and December 31, 2025 (inclusive)."
    add_ins_release = (
        "Accept announcement or release dates within 2023, 2024, or 2025 on official pages, press releases, or reputable reviews. "
        "If multiple dates are shown, choose the earliest announcement or public release."
    )

    # Batch verification of all spec leaves
    claims_and_sources = [
        (claim_refresh, all_urls, leaf_refresh, add_ins_refresh),
        (claim_response, all_urls, leaf_response, add_ins_response),
        (claim_ips, all_urls, leaf_ips, add_ins_ips),
        (claim_res, all_urls, leaf_res, add_ins_res),
        (claim_size, all_urls, leaf_size, add_ins_size),
        (claim_gsync, all_urls, leaf_gsync, add_ins_gsync),
        (claim_hdr, all_urls, leaf_hdr, add_ins_hdr),
        (claim_dp, all_urls, leaf_dp, add_ins_dp),
        (claim_hdmi, all_urls, leaf_hdmi, add_ins_hdmi),
        (claim_height, all_urls, leaf_height, add_ins_height),
        (claim_vesa, all_urls, leaf_vesa, add_ins_vesa),
        (claim_p3, all_urls, leaf_p3, add_ins_p3),
        (claim_gamescom, all_urls, leaf_gamescom, add_ins_gamescom),
        (claim_release, all_urls, leaf_release, add_ins_release),
    ]

    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the gaming monitor identification task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation (top-level node will be critical parallel)
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

    # Extract structured monitor specs from the answer
    specs: MonitorSpecs = await evaluator.extract(
        prompt=prompt_extract_monitor_specs(),
        template_class=MonitorSpecs,
        extraction_name="monitor_specs",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, specs)

    # Return structured evaluation summary
    return evaluator.get_summary()