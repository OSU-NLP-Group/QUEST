import asyncio
import logging
import re
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "ai_accelerator_hbm_2025_2026"
TASK_DESCRIPTION = (
    "Identify one AI accelerator chip that was announced or released between January 1, 2025, and January 9, 2026, "
    "and that features at least 250 GB of high-bandwidth memory (HBM). For the chip you identify, provide the following "
    "information: (1) The chip's name and model designation, (2) The total HBM memory capacity in GB, (3) The memory "
    "bandwidth in TB/s, (4) The specific HBM generation (e.g., HBM3E, HBM4), (5) The manufacturer, (6) The architecture "
    "type or code name, (7) At least one AI-related performance metric (such as PFLOPS, TOPS, or inference performance), "
    "and (8) An official announcement or product page URL from the manufacturer. The chip must be from a major "
    "semiconductor manufacturer (NVIDIA, AMD, Intel, or Google) and must be specifically designed for AI/ML workloads."
)

ALLOWED_MANUFACTURERS = {"nvidia", "amd", "intel", "google"}
MANUFACTURER_DOMAINS = {
    "nvidia": ["nvidia.com", "developer.nvidia.com", "nvidianews.nvidia.com"],
    "amd": ["amd.com", "community.amd.com", "newsroom.amd.com"],
    "intel": ["intel.com", "newsroom.intel.com"],
    "google": ["google.com", "blog.google", "cloud.google.com", "ai.google.dev", "ai.google"]
}
TIMEFRAME_START = "2025-01-01"
TIMEFRAME_END = "2026-01-09"


# ----------------------------- Data Models --------------------------------- #
class ChipSpec(BaseModel):
    chip_name: Optional[str] = None
    model_designation: Optional[str] = None
    hbm_capacity_gb: Optional[str] = None
    mem_bandwidth_tbps: Optional[str] = None
    hbm_generation: Optional[str] = None
    manufacturer: Optional[str] = None
    architecture_codename: Optional[str] = None
    ai_performance_metric: Optional[str] = None
    official_url: Optional[str] = None
    announcement_date: Optional[str] = None  # Optional: as given in the answer, e.g., "Nov 2025"
    support_urls: List[str] = Field(default_factory=list)  # additional URLs mentioned in the answer
    other_chips_mentioned: List[str] = Field(default_factory=list)  # other chip names if multiple are mentioned


# ----------------------------- Extraction Prompt --------------------------- #
def prompt_extract_chip_spec() -> str:
    return (
        "Extract exactly one target AI accelerator chip and its fields from the answer.\n"
        "Return a JSON object with the following fields (use strings; if missing, set null; if lists missing, return empty list):\n"
        "- chip_name: The chip's product name.\n"
        "- model_designation: The model designation (SKU or variant name).\n"
        "- hbm_capacity_gb: Total HBM capacity as expressed in the answer (include units like 'GB' or 'TB' if present).\n"
        "- mem_bandwidth_tbps: Memory bandwidth value as expressed (ideally in TB/s; include units if present).\n"
        "- hbm_generation: The HBM generation string (e.g., 'HBM3E', 'HBM4').\n"
        "- manufacturer: Manufacturer name (e.g., 'NVIDIA', 'AMD', 'Intel', 'Google').\n"
        "- architecture_codename: Architecture type or code name.\n"
        "- ai_performance_metric: At least one AI-related performance metric string (e.g., '20 PFLOPS FP8', '2000 TOPS', or an inference benchmark).\n"
        "- official_url: The official announcement or product page URL from the manufacturer.\n"
        "- announcement_date: If the answer mentions an announcement or release date, extract it verbatim.\n"
        "- support_urls: List of any other URLs in the answer that support the claims for this chip (can include manufacturer blog/newsroom pages).\n"
        "- other_chips_mentioned: List of names of any other chips that are mentioned in the answer (if multiple candidates are listed)."
    )


# ----------------------------- Helper Functions ---------------------------- #
def _safe_str(s: Optional[str]) -> str:
    return s.strip() if isinstance(s, str) else ""


def _normalize_manufacturer(name: Optional[str]) -> str:
    n = _safe_str(name).lower()
    if "nvidia" in n:
        return "nvidia"
    if "advanced micro devices" in n or "amd" in n:
        return "amd"
    if "intel" in n:
        return "intel"
    if "google" in n:
        return "google"
    return n  # return as-is; may fail allowed set check


def _first_number(text: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)", text.replace(",", ""))
    return float(m.group(1)) if m else None


def _capacity_to_gb(capacity_text: Optional[str]) -> Optional[float]:
    """
    Convert capacity string to GB if possible. Handles GB and TB units.
    Uses 1000 as conversion factor for TB -> GB for tolerance.
    """
    if not capacity_text:
        return None
    t = capacity_text.lower()
    num = _first_number(t)
    if num is None:
        return None
    if "tb" in t:
        return num * 1000.0
    return num  # assume GB if 'gb' or unit not specified


def _bandwidth_to_tbps(bw_text: Optional[str]) -> Optional[float]:
    """
    Convert bandwidth string to TB/s if possible. Handles TB/s and GB/s.
    Uses 1000 as conversion factor GB/s -> TB/s for tolerance.
    """
    if not bw_text:
        return None
    t = bw_text.lower()
    num = _first_number(t)
    if num is None:
        return None
    if "tb" in t:
        return num
    if "gb" in t:
        return num / 1000.0
    # If unit not clear, assume TB/s to be lenient
    return num


def _is_manufacturer_domain(url: Optional[str], manufacturer_norm: str) -> bool:
    if not url or manufacturer_norm not in MANUFACTURER_DOMAINS:
        return False
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return False
    allowed_suffixes = MANUFACTURER_DOMAINS[manufacturer_norm]
    return any(netloc.endswith(suffix) for suffix in allowed_suffixes)


def _sources_list(spec: ChipSpec) -> List[str]:
    urls: List[str] = []
    if spec.official_url:
        urls.append(spec.official_url)
    if spec.support_urls:
        urls.extend([u for u in spec.support_urls if _safe_str(u)])
    return urls


# ----------------------------- Verification Builder ------------------------ #
async def build_and_verify(evaluator: Evaluator, root_node, spec: ChipSpec) -> None:
    """
    Build the verification tree under a critical task node and perform verifications.
    """
    task_node = evaluator.add_parallel(
        id="task_core",
        desc="Identify exactly one qualifying AI accelerator chip (per constraints) and provide all required specification fields with an official manufacturer URL.",
        parent=root_node,
        critical=True  # Gate: any failure should fail the whole task
    )

    # 1) Exactly one chip identified
    only_one_chip = bool(_safe_str(spec.chip_name) or _safe_str(spec.model_designation)) and bool(_safe_str(spec.official_url)) and len(spec.other_chips_mentioned or []) == 0
    evaluator.add_custom_node(
        result=only_one_chip,
        id="single_chip_identified",
        desc="Exactly one chip is identified (not multiple candidates).",
        parent=task_node,
        critical=True
    )

    # 2) Announcement timeframe check (via manufacturer/support URLs)
    timeframe_node = evaluator.add_leaf(
        id="announcement_timeframe",
        desc="The chip was announced or released between January 1, 2025 and January 9, 2026.",
        parent=task_node,
        critical=True
    )
    ann_date_text = _safe_str(spec.announcement_date)
    timeframe_claim = (
        f"The chip was announced or released between {TIMEFRAME_START} and {TIMEFRAME_END}."
        if not ann_date_text
        else f"The chip was announced or released on '{ann_date_text}', which falls between {TIMEFRAME_START} and {TIMEFRAME_END}."
    )
    await evaluator.verify(
        claim=timeframe_claim,
        node=timeframe_node,
        sources=_sources_list(spec),
        additional_instruction=(
            "Check the announcement/release date on the manufacturer page(s) or clearly referenced date within the page(s). "
            "Pass only if the date explicitly lies within the specified timeframe."
        )
    )

    # 3) Eligible manufacturer
    manufacturer_norm = _normalize_manufacturer(spec.manufacturer)
    is_allowed_mfr = manufacturer_norm in ALLOWED_MANUFACTURERS
    evaluator.add_custom_node(
        result=is_allowed_mfr,
        id="eligible_manufacturer",
        desc="The chip's manufacturer is one of: NVIDIA, AMD, Intel, or Google.",
        parent=task_node,
        critical=True
    )

    # 4) AI/ML-specific design (verify the product page describes AI/ML usage)
    ai_design_node = evaluator.add_leaf(
        id="ai_ml_specific_design",
        desc="The chip is specifically designed for AI/ML workloads (e.g., AI accelerator GPU or AI-focused ASIC), not merely a general-purpose processor.",
        parent=task_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "This chip is specifically designed for AI/ML workloads (e.g., AI accelerator GPU or AI-focused ASIC) rather than a general-purpose processor."
        ),
        node=ai_design_node,
        sources=_sources_list(spec),
        additional_instruction=(
            "Look for phrases indicating AI acceleration, AI training/inference, ML workloads, or an AI-focused architecture on the page(s)."
        )
    )

    # 5) Chip name and model
    name_model_block = evaluator.add_parallel(
        id="chip_name_and_model_block",
        desc="Chip name and model designation checks",
        parent=task_node,
        critical=True
    )
    name_provided = bool(_safe_str(spec.chip_name))
    model_provided = bool(_safe_str(spec.model_designation))
    evaluator.add_custom_node(
        result=(name_provided and model_provided),
        id="chip_name_and_model_provided",
        desc="The chip's name and model designation are provided.",
        parent=name_model_block,
        critical=True
    )
    name_model_verify_node = evaluator.add_leaf(
        id="chip_name_and_model_on_page",
        desc="The official page shows the chip name and model designation as provided.",
        parent=name_model_block,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The official manufacturer page mentions the chip name '{_safe_str(spec.chip_name)}' and the model designation '{_safe_str(spec.model_designation)}'."
        ),
        node=name_model_verify_node,
        sources=_sources_list(spec),
        additional_instruction=(
            "Allow minor formatting differences, letter casing, or suffixes/prefixes. The page should clearly show both the name and the model designation."
        )
    )

    # 6) HBM capacity and threshold
    hbm_block = evaluator.add_parallel(
        id="hbm_capacity_and_threshold_block",
        desc="HBM capacity presence and threshold checks",
        parent=task_node,
        critical=True
    )
    capacity_text = _safe_str(spec.hbm_capacity_gb)
    capacity_provided = bool(capacity_text)
    evaluator.add_custom_node(
        result=capacity_provided,
        id="hbm_capacity_provided",
        desc="Total HBM capacity is provided in GB.",
        parent=hbm_block,
        critical=True
    )
    capacity_gb_val = _capacity_to_gb(capacity_text)
    evaluator.add_custom_node(
        result=(capacity_gb_val is not None and capacity_gb_val >= 250.0),
        id="hbm_capacity_meets_threshold",
        desc="Total HBM capacity is at least 250 GB.",
        parent=hbm_block,
        critical=True
    )

    # 7) Memory bandwidth
    bw_block = evaluator.add_parallel(
        id="memory_bandwidth_block",
        desc="Memory bandwidth checks",
        parent=task_node,
        critical=True
    )
    bw_text = _safe_str(spec.mem_bandwidth_tbps)
    bw_provided = bool(bw_text)
    evaluator.add_custom_node(
        result=bw_provided,
        id="memory_bandwidth_provided",
        desc="Memory bandwidth is provided.",
        parent=bw_block,
        critical=True
    )
    bw_tbps_val = _bandwidth_to_tbps(bw_text)
    bw_tbps_unit_ok = ("tb" in bw_text.lower()) or (bw_tbps_val is not None)
    evaluator.add_custom_node(
        result=bw_tbps_unit_ok,
        id="memory_bandwidth_tbps_unit",
        desc="Memory bandwidth is expressed or convertible to TB/s.",
        parent=bw_block,
        critical=True
    )

    # 8) HBM generation
    hbm_gen_block = evaluator.add_parallel(
        id="hbm_generation_block",
        desc="HBM generation checks",
        parent=task_node,
        critical=True
    )
    hbm_gen_text = _safe_str(spec.hbm_generation)
    evaluator.add_custom_node(
        result=bool(hbm_gen_text),
        id="hbm_generation_provided",
        desc="The specific HBM generation is identified (e.g., HBM3E, HBM4).",
        parent=hbm_gen_block,
        critical=True
    )
    hbm_gen_verify_node = evaluator.add_leaf(
        id="hbm_generation_on_page",
        desc="HBM generation is supported by the official or support page(s).",
        parent=hbm_gen_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"The chip uses {_safe_str(spec.hbm_generation)} memory.",
        node=hbm_gen_verify_node,
        sources=_sources_list(spec),
        additional_instruction="Check that the page mentions the specific HBM generation, such as HBM3E or HBM4."
    )

    # 9) Architecture code name
    arch_block = evaluator.add_parallel(
        id="architecture_codename_block",
        desc="Architecture/code name checks",
        parent=task_node,
        critical=True
    )
    arch_text = _safe_str(spec.architecture_codename)
    evaluator.add_custom_node(
        result=bool(arch_text),
        id="architecture_codename_provided",
        desc="The architecture type or code name is specified.",
        parent=arch_block,
        critical=True
    )
    arch_verify_node = evaluator.add_leaf(
        id="architecture_codename_on_page",
        desc="Architecture type or code name is supported by the official or support page(s).",
        parent=arch_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"The chip's architecture type or code name is '{arch_text}'.",
        node=arch_verify_node,
        sources=_sources_list(spec),
        additional_instruction="Confirm the architecture type or code name on the page(s), allowing minor formatting differences."
    )

    # 10) AI performance metric
    perf_block = evaluator.add_parallel(
        id="ai_performance_metric_block",
        desc="AI-related performance metric checks",
        parent=task_node,
        critical=True
    )
    perf_text = _safe_str(spec.ai_performance_metric)
    evaluator.add_custom_node(
        result=bool(perf_text),
        id="ai_performance_metric_provided",
        desc="At least one AI-related performance metric is provided (e.g., PFLOPS, TOPS, or inference performance).",
        parent=perf_block,
        critical=True
    )
    perf_verify_node = evaluator.add_leaf(
        id="ai_performance_metric_on_page",
        desc="The AI-related performance metric is supported by the official or support page(s).",
        parent=perf_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"The chip has the following AI-related performance metric: '{perf_text}'.",
        node=perf_verify_node,
        sources=_sources_list(spec),
        additional_instruction="Verify that the metric (e.g., PFLOPS, TOPS, or inference performance) is present on the page(s). Allow minor unit variations."
    )

    # 11) Official manufacturer URL with specs (domain and presence of capacity/bandwidth on that page)
    official_block = evaluator.add_parallel(
        id="official_manufacturer_url_block",
        desc="Official manufacturer URL and specs presence checks",
        parent=task_node,
        critical=True
    )
    official_url_present = bool(_safe_str(spec.official_url))
    evaluator.add_custom_node(
        result=official_url_present,
        id="official_url_provided",
        desc="An official manufacturer announcement/product page URL is provided.",
        parent=official_block,
        critical=True
    )
    domain_ok = _is_manufacturer_domain(spec.official_url, manufacturer_norm)
    evaluator.add_custom_node(
        result=domain_ok,
        id="official_url_is_manufacturer_domain",
        desc="The provided URL belongs to the manufacturer's official domain.",
        parent=official_block,
        critical=True
    )
    official_capacity_verify = evaluator.add_leaf(
        id="official_url_supports_hbm_capacity",
        desc="Official manufacturer page contains the HBM capacity specification.",
        parent=official_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official manufacturer page states that the chip has {_safe_str(spec.hbm_capacity_gb)} of HBM memory.",
        node=official_capacity_verify,
        sources=spec.official_url,
        additional_instruction="Confirm that the page explicitly shows the HBM capacity (allow minor formatting differences)."
    )
    official_bw_verify = evaluator.add_leaf(
        id="official_url_supports_memory_bandwidth",
        desc="Official manufacturer page contains the memory bandwidth specification.",
        parent=official_block,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official manufacturer page states that the chip's memory bandwidth is {_safe_str(spec.mem_bandwidth_tbps)}.",
        node=official_bw_verify,
        sources=spec.official_url,
        additional_instruction="Confirm that the page explicitly shows the memory bandwidth (allow unit wording variants like TB/s, TBps)."
    )


# ----------------------------- Main Entrypoint ------------------------------ #
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate the agent's answer for the AI accelerator chip HBM specification task.
    """
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

    # Extract chip specification from the answer
    spec: ChipSpec = await evaluator.extract(
        prompt=prompt_extract_chip_spec(),
        template_class=ChipSpec,
        extraction_name="chip_spec_extraction"
    )

    # Record ground truth constraints for context
    evaluator.add_ground_truth({
        "timeframe_start": TIMEFRAME_START,
        "timeframe_end": TIMEFRAME_END,
        "allowed_manufacturers": sorted(list(ALLOWED_MANUFACTURERS)),
        "hbm_min_capacity_gb": 250
    }, gt_type="constraints")

    # Build verification tree and run checks
    await build_and_verify(evaluator, root, spec)

    # Return structured summary
    return evaluator.get_summary()