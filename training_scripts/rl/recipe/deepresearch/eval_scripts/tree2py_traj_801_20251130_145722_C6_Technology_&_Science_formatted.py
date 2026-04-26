import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "console_setup_2025"
TASK_DESCRIPTION = """A gaming enthusiast in California wants to build a high-performance console gaming setup for 2025 and needs your recommendation. They require a current-generation gaming console that meets the following specifications:

Console Requirements:
- GPU performance of at least 12 TFLOPS
- Minimum 1TB internal storage
- HDMI 2.1 support for 4K@120Hz gaming
- Variable Refresh Rate (VRR) support
- Dedicated hardware ray tracing capability

Storage and Services:
- Additional 1TB or more storage expansion compatible with the selected console
- A subscription service that includes cloud gaming/streaming functionality

Display Compatibility:
- Identify the display requirements needed for optimal gaming (HDMI version, HDR, VRR)

Budget Analysis:
- Calculate the total first-year cost including: console purchase, storage expansion, and annual subscription service

Provide a complete recommendation that includes:
1. The specific console model that meets all requirements
2. A compatible storage expansion solution with technical specifications
3. The appropriate subscription service tier with cloud gaming
4. Display requirements for optimal experience
5. Total first-year cost breakdown

Include reference URLs for all specifications to verify your recommendations."""


# ------------------------------ Data Models --------------------------------- #
class ConsoleSpecs(BaseModel):
    model_name: Optional[str] = None
    gpu_tflops: Optional[str] = None
    internal_storage: Optional[str] = None
    hdmi_21: Optional[bool] = None
    supports_4k120: Optional[bool] = None
    vrr: Optional[bool] = None
    ray_tracing_hw: Optional[bool] = None
    reference_urls: List[str] = Field(default_factory=list)


class StorageExpansion(BaseModel):
    product_name: Optional[str] = None
    capacity_str: Optional[str] = None
    capacity_tb_num: Optional[float] = None
    interface: Optional[str] = None
    read_speed_mb_s: Optional[float] = None
    proprietary_card: Optional[bool] = None
    reference_urls: List[str] = Field(default_factory=list)


class SubscriptionService(BaseModel):
    service_name: Optional[str] = None
    tier_name: Optional[str] = None
    includes_cloud_gaming: Optional[bool] = None
    compatible_platform: Optional[str] = None
    annual_price_usd: Optional[float] = None
    reference_urls: List[str] = Field(default_factory=list)


class DisplayRequirements(BaseModel):
    hdmi_21_required: Optional[bool] = None
    hdr_required: Optional[bool] = None
    vrr_required: Optional[bool] = None
    reference_urls: List[str] = Field(default_factory=list)


class BudgetInfo(BaseModel):
    console_price_usd: Optional[float] = None
    storage_price_usd: Optional[float] = None
    subscription_annual_usd: Optional[float] = None
    total_first_year_usd: Optional[float] = None


# ---------------------------- Extraction Prompts ---------------------------- #
def prompt_extract_console() -> str:
    return """
    Extract the selected console and its cited specification references from the answer.

    Return fields:
    - model_name: the exact console model name (e.g., "Xbox Series X", "PlayStation 5 Pro")
    - gpu_tflops: the GPU performance as stated (e.g., "12 TFLOPS", "33.5 TFLOPS"); if not stated, null
    - internal_storage: internal storage as stated (e.g., "1TB", "825GB"); if not stated, null
    - hdmi_21: boolean, whether the answer explicitly claims HDMI 2.1 support
    - supports_4k120: boolean, whether the answer claims 4K@120Hz support
    - vrr: boolean, whether the answer claims VRR support
    - ray_tracing_hw: boolean, whether the answer claims hardware ray tracing capability
    - reference_urls: list of URLs provided in the answer that serve as references for the console specifications

    Rules:
    - Only extract URLs explicitly present in the answer. If none are provided, return an empty list.
    - Parse the booleans based solely on answer statements.
    """


def prompt_extract_storage() -> str:
    return """
    Extract the storage expansion recommendation and its specification references from the answer.

    Return fields:
    - product_name: the exact storage expansion product/model name
    - capacity_str: capacity as stated (e.g., "1TB", "2TB")
    - capacity_tb_num: numeric capacity in terabytes if stated (e.g., 1.0, 2.0); else null
    - interface: interface/form factor as stated (e.g., "M.2 NVMe PCIe Gen4 x4", "proprietary expansion card")
    - read_speed_mb_s: numeric sequential read speed MB/s if stated (e.g., 5500); else null
    - proprietary_card: boolean indicating claimed proprietary expansion card (True) if relevant (e.g., Xbox Series expansion card)
    - reference_urls: list of URLs provided in the answer that verify storage specs/compatibility

    Rules:
    - Only extract URLs explicitly present in the answer.
    """


def prompt_extract_subscription() -> str:
    return """
    Extract the subscription service recommendation and its references from the answer.

    Return fields:
    - service_name: e.g., "Xbox Game Pass Ultimate", "PlayStation Plus Premium"
    - tier_name: specific tier name if applicable
    - includes_cloud_gaming: boolean indicating claimed inclusion of cloud gaming/streaming
    - compatible_platform: platform mentioned for compatibility (e.g., "Xbox", "PlayStation")
    - annual_price_usd: numeric annual price in USD if stated; else null
    - reference_urls: list of URLs provided in the answer that verify subscription features

    Rules:
    - Only extract URLs explicitly present in the answer.
    - Convert the annual price to a number, stripping symbols.
    """


def prompt_extract_display() -> str:
    return """
    Extract the display requirements statements and references from the answer.

    Return fields:
    - hdmi_21_required: boolean, whether the answer states HDMI 2.1 is required for 4K@120Hz
    - hdr_required: boolean, whether the answer states HDR support is required/recommended for optimal experience
    - vrr_required: boolean, whether the answer states VRR compatibility is required/recommended for optimal experience
    - reference_urls: list of URLs provided in the answer that support these display requirements

    Rules:
    - Only extract URLs explicitly present in the answer.
    """


def prompt_extract_budget() -> str:
    return """
    Extract the first-year cost breakdown from the answer.

    Return fields:
    - console_price_usd: numeric console purchase price in USD
    - storage_price_usd: numeric storage expansion cost in USD
    - subscription_annual_usd: numeric annual subscription cost in USD
    - total_first_year_usd: numeric total first-year cost in USD as stated by the answer

    Rules:
    - Convert all amounts to numbers in USD; strip currency symbols and commas.
    - If any component is not present, return null for that field.
    """


# ------------------------------ Helper Utils -------------------------------- #
def detect_platform(model_name: Optional[str]) -> str:
    if not model_name:
        return "unknown"
    m = model_name.lower()
    if "xbox" in m or "series x" in m or "series s" in m:
        return "xbox"
    if "playstation" in m or "ps5" in m or "ps 5" in m:
        return "ps5"
    return "unknown"


def format_platform(platform: str) -> str:
    return "Xbox Series" if platform == "xbox" else ("PlayStation 5" if platform == "ps5" else "unknown platform")


# --------------------------- Verification Builders -------------------------- #
async def verify_console_selection(
    evaluator: Evaluator,
    parent_node,
    console: ConsoleSpecs,
) -> None:
    task_node = evaluator.add_parallel(
        id="console_selection",
        desc="Selected console meets required technical specifications and provides verifiable references.",
        parent=parent_node,
        critical=True,
    )

    # Reference URLs existence (gate others)
    refs_exist = evaluator.add_custom_node(
        result=bool(console.reference_urls),
        id="console_reference_urls",
        desc="Provides reference URL(s) verifying the console specifications.",
        parent=task_node,
        critical=True,
    )

    # GPU >= 12 TFLOPS
    gpu_leaf = evaluator.add_leaf(
        id="gpu_performance",
        desc="Console GPU performance is at least 12 TFLOPS.",
        parent=task_node,
        critical=True,
    )
    gpu_claim = f"The console {console.model_name or 'selected console'} has GPU performance of at least 12 TFLOPS."
    await evaluator.verify(
        claim=gpu_claim,
        node=gpu_leaf,
        sources=console.reference_urls,
        additional_instruction="Confirm that the page states GPU performance ≥ 12 TFLOPS; allow equivalent wording like '12 teraflops' or greater.",
    )

    # Internal storage >= 1TB
    storage_leaf = evaluator.add_leaf(
        id="internal_storage",
        desc="Console has at least 1TB internal storage.",
        parent=task_node,
        critical=True,
    )
    storage_claim = f"The console {console.model_name or 'selected console'} includes at least 1 TB of internal storage."
    await evaluator.verify(
        claim=storage_claim,
        node=storage_leaf,
        sources=console.reference_urls,
        additional_instruction="Look for '1TB' or '1 TB' internal storage spec; greater capacities also satisfy the requirement.",
    )

    # HDMI 2.1 for 4K@120Hz
    hdmi_leaf = evaluator.add_leaf(
        id="hdmi_21_4k120",
        desc="Console supports HDMI 2.1 for 4K@120Hz output.",
        parent=task_node,
        critical=True,
    )
    hdmi_claim = f"The console {console.model_name or 'selected console'} supports HDMI 2.1 enabling 4K at 120 Hz output."
    await evaluator.verify(
        claim=hdmi_claim,
        node=hdmi_leaf,
        sources=console.reference_urls,
        additional_instruction="Accept specs stating HDMI 2.1 and 4K/120Hz capability; minor wording differences are fine.",
    )

    # VRR support
    vrr_leaf = evaluator.add_leaf(
        id="vrr_support",
        desc="Console supports Variable Refresh Rate (VRR).",
        parent=task_node,
        critical=True,
    )
    vrr_claim = f"The console {console.model_name or 'selected console'} supports Variable Refresh Rate (VRR)."
    await evaluator.verify(
        claim=vrr_claim,
        node=vrr_leaf,
        sources=console.reference_urls,
        additional_instruction="Look for mentions of VRR support; allow equivalent phrases like 'variable refresh rate'.",
    )

    # Hardware ray tracing
    rt_leaf = evaluator.add_leaf(
        id="hardware_ray_tracing",
        desc="Console includes dedicated hardware ray tracing capability.",
        parent=task_node,
        critical=True,
    )
    rt_claim = f"The console {console.model_name or 'selected console'} provides hardware-accelerated ray tracing."
    await evaluator.verify(
        claim=rt_claim,
        node=rt_leaf,
        sources=console.reference_urls,
        additional_instruction="Confirm the page mentions ray tracing, hardware acceleration, or equivalent wording.",
    )


async def verify_storage_expansion(
    evaluator: Evaluator,
    parent_node,
    storage: StorageExpansion,
    platform: str,
) -> None:
    task_node = evaluator.add_parallel(
        id="storage_expansion",
        desc="Storage expansion is compatible with the chosen console, provides at least 1TB additional capacity, and includes verifiable references.",
        parent=parent_node,
        critical=True,
    )

    # Reference URLs existence (gate others)
    refs_exist = evaluator.add_custom_node(
        result=bool(storage.reference_urls),
        id="storage_reference_urls",
        desc="Provides reference URL(s) verifying the storage expansion compatibility/specifications.",
        parent=task_node,
        critical=True,
    )

    # Capacity >= 1TB
    cap_leaf = evaluator.add_leaf(
        id="capacity_requirement",
        desc="Storage expansion provides at least 1TB additional capacity.",
        parent=task_node,
        critical=True,
    )
    cap_claim = f"The storage expansion {storage.product_name or 'selected expansion'} provides at least 1 TB of additional capacity."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=storage.reference_urls,
        additional_instruction="Accept capacity statements of 1TB or greater (e.g., 2TB).",
    )

    # Platform-specific compatibility
    compat_leaf = evaluator.add_leaf(
        id="platform_specific_compatibility",
        desc="Storage expansion matches platform-specific constraints (PS5: PCIe Gen 4 x4 M.2 SSD with ≥5,500 MB/s sequential read; Xbox Series X: proprietary Seagate Storage Expansion Card).",
        parent=task_node,
        critical=True,
    )

    if platform == "ps5":
        compat_claim = (
            f"The storage expansion {storage.product_name or 'selected expansion'} is an M.2 NVMe PCIe Gen 4 x4 SSD "
            f"compatible with PS5 and has sequential read speed of at least 5,500 MB/s."
        )
        add_ins = "Confirm PS5 compatibility and NVMe PCIe Gen4 x4 interface; speed ≥ 5,500 MB/s satisfies PS5 spec."
    elif platform == "xbox":
        compat_claim = (
            f"The storage expansion {storage.product_name or 'selected expansion'} is the official proprietary "
            f"storage expansion card compatible with Xbox Series X/S."
        )
        add_ins = "Allow equivalent official expansion cards (e.g., Seagate or WD) for Xbox Series X/S proprietary storage."
    else:
        compat_claim = (
            f"The storage expansion {storage.product_name or 'selected expansion'} matches the platform-specific "
            f"requirements of the selected console."
        )
        add_ins = "Verify compatibility based on the referenced page; platform may be unknown."

    await evaluator.verify(
        claim=compat_claim,
        node=compat_leaf,
        sources=storage.reference_urls,
        additional_instruction=add_ins,
    )


async def verify_subscription_service(
    evaluator: Evaluator,
    parent_node,
    subscription: SubscriptionService,
    platform: str,
) -> None:
    task_node = evaluator.add_parallel(
        id="subscription_service",
        desc="Subscription service is compatible with the chosen console platform, includes cloud gaming/streaming, and includes verifiable references.",
        parent=parent_node,
        critical=True,
    )

    # Reference URLs existence (gate others)
    refs_exist = evaluator.add_custom_node(
        result=bool(subscription.reference_urls),
        id="subscription_reference_urls",
        desc="Provides reference URL(s) verifying the subscription features (including cloud gaming).",
        parent=task_node,
        critical=True,
    )

    # Service compatibility
    compat_leaf = evaluator.add_leaf(
        id="service_compatibility",
        desc="Subscription service is compatible with the selected console/platform.",
        parent=task_node,
        critical=True,
    )
    plat_text = format_platform(platform)
    compat_claim = (
        f"The subscription service {subscription.service_name or 'selected service'} "
        f"is available and compatible with {plat_text}."
    )
    await evaluator.verify(
        claim=compat_claim,
        node=compat_leaf,
        sources=subscription.reference_urls,
        additional_instruction="Confirm that the service is offered for the specified console platform.",
    )

    # Cloud gaming included
    cloud_leaf = evaluator.add_leaf(
        id="cloud_gaming_included",
        desc="Subscription includes cloud gaming/streaming functionality.",
        parent=task_node,
        critical=True,
    )
    cloud_claim = (
        f"The subscription tier {subscription.tier_name or subscription.service_name or 'selected tier'} includes cloud gaming or cloud streaming."
    )
    await evaluator.verify(
        claim=cloud_claim,
        node=cloud_leaf,
        sources=subscription.reference_urls,
        additional_instruction="Look for mentions of cloud gaming, cloud streaming, or play over the cloud; allow equivalent phrasing.",
    )


async def verify_display_requirements(
    evaluator: Evaluator,
    parent_node,
    display: DisplayRequirements,
) -> None:
    task_node = evaluator.add_parallel(
        id="display_requirements",
        desc="Display requirements for optimal gaming are identified and include verifiable references.",
        parent=parent_node,
        critical=True,
    )

    # Reference URLs existence (gate others)
    refs_exist = evaluator.add_custom_node(
        result=bool(display.reference_urls),
        id="display_reference_urls",
        desc="Provides reference URL(s) supporting the stated display requirements.",
        parent=task_node,
        critical=True,
    )

    # HDMI 2.1 required
    hdmi_leaf = evaluator.add_leaf(
        id="hdmi_21_required",
        desc="Identifies HDMI 2.1 as required for 4K@120Hz.",
        parent=task_node,
        critical=True,
    )
    hdmi_claim = "HDMI 2.1 is required for consistent 4K at 120 Hz console gaming."
    await evaluator.verify(
        claim=hdmi_claim,
        node=hdmi_leaf,
        sources=display.reference_urls,
        additional_instruction="Accept statements that 4K@120Hz requires HDMI 2.1 on TVs/monitors for consoles.",
    )

    # HDR required
    hdr_leaf = evaluator.add_leaf(
        id="hdr_required",
        desc="Identifies HDR support as required.",
        parent=task_node,
        critical=True,
    )
    hdr_claim = "HDR support is required or strongly recommended for optimal console gaming visuals."
    await evaluator.verify(
        claim=hdr_claim,
        node=hdr_leaf,
        sources=display.reference_urls,
        additional_instruction="Allow 'required' or 'recommended' phrasing; confirm HDR improves dynamic range and is an expected feature for optimal experience.",
    )

    # VRR required
    vrr_leaf = evaluator.add_leaf(
        id="vrr_required",
        desc="Identifies VRR compatibility as required.",
        parent=task_node,
        critical=True,
    )
    vrr_claim = "VRR compatibility is required or strongly recommended for optimal console gaming smoothness at variable frame rates."
    await evaluator.verify(
        claim=vrr_claim,
        node=vrr_leaf,
        sources=display.reference_urls,
        additional_instruction="Allow 'required' or 'recommended' phrasing; confirm VRR reduces tearing by matching refresh to frame rate.",
    )


async def verify_budget_analysis(
    evaluator: Evaluator,
    parent_node,
    budget: BudgetInfo,
) -> None:
    task_node = evaluator.add_parallel(
        id="budget_analysis",
        desc="Computes total first-year cost including console, storage expansion, and annual subscription, with a clear breakdown.",
        parent=parent_node,
        critical=True,
    )

    # Cost breakdown presence
    has_breakdown = all([
        budget.console_price_usd is not None,
        budget.storage_price_usd is not None,
        budget.subscription_annual_usd is not None,
        budget.total_first_year_usd is not None,
    ])
    evaluator.add_custom_node(
        result=has_breakdown,
        id="cost_breakdown_present",
        desc="Provides a first-year cost breakdown including console purchase price, storage expansion cost, and annual subscription cost.",
        parent=task_node,
        critical=True,
    )

    # Correct total check (tolerance ±1.0 USD)
    sum_components = None
    total_correct = False
    if has_breakdown:
        try:
            sum_components = (
                float(budget.console_price_usd) +
                float(budget.storage_price_usd) +
                float(budget.subscription_annual_usd)
            )
            total_correct = abs(sum_components - float(budget.total_first_year_usd)) <= 1.0
        except Exception:
            total_correct = False

    evaluator.add_custom_node(
        result=total_correct,
        id="total_first_year_cost_correct",
        desc="Total first-year cost equals (console + storage expansion + annual subscription) based on the stated component costs.",
        parent=task_node,
        critical=True,
    )

    # Record custom info for transparency
    evaluator.add_custom_info(
        info={
            "console_price_usd": budget.console_price_usd,
            "storage_price_usd": budget.storage_price_usd,
            "subscription_annual_usd": budget.subscription_annual_usd,
            "stated_total_first_year_usd": budget.total_first_year_usd,
            "computed_sum_usd": sum_components,
            "within_tolerance": total_correct
        },
        info_type="budget_check",
        info_name="budget_analysis_details"
    )


# ---------------------------- Main Evaluation API --------------------------- #
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

    # Create a critical top-level task node (since Evaluator root is non-critical by default)
    task_root = evaluator.add_parallel(
        id="task_main",
        desc="Complete gaming console setup recommendation meets all specified requirements (console specs, storage expansion, subscription, display requirements, cost calculation, and reference URLs).",
        parent=root,
        critical=True,
    )

    # Parallel extractions
    console_specs_task = evaluator.extract(
        prompt=prompt_extract_console(),
        template_class=ConsoleSpecs,
        extraction_name="console_specs",
    )
    storage_task = evaluator.extract(
        prompt=prompt_extract_storage(),
        template_class=StorageExpansion,
        extraction_name="storage_expansion",
    )
    subscription_task = evaluator.extract(
        prompt=prompt_extract_subscription(),
        template_class=SubscriptionService,
        extraction_name="subscription_service",
    )
    display_task = evaluator.extract(
        prompt=prompt_extract_display(),
        template_class=DisplayRequirements,
        extraction_name="display_requirements",
    )
    budget_task = evaluator.extract(
        prompt=prompt_extract_budget(),
        template_class=BudgetInfo,
        extraction_name="budget_info",
    )

    console_specs, storage_info, subscription_info, display_info, budget_info = await asyncio.gather(
        console_specs_task, storage_task, subscription_task, display_task, budget_task
    )

    # Platform detection from console model
    platform = detect_platform(console_specs.model_name)
    evaluator.add_custom_info(
        info={"console_model": console_specs.model_name, "detected_platform": platform},
        info_type="platform_detection",
        info_name="console_platform_info"
    )

    # Build verification tree per rubric
    await verify_console_selection(evaluator, task_root, console_specs)
    await verify_storage_expansion(evaluator, task_root, storage_info, platform)
    await verify_subscription_service(evaluator, task_root, subscription_info, platform)
    await verify_display_requirements(evaluator, task_root, display_info)
    await verify_budget_analysis(evaluator, task_root, budget_info)

    return evaluator.get_summary()