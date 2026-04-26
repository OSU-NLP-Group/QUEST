import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "thinkpad_t14_gen5_ultra5_125u_specs"
TASK_DESCRIPTION = """
I am researching business laptops and need detailed specifications for a specific Lenovo ThinkPad model. Please find and provide comprehensive specifications for the Lenovo ThinkPad T14 Gen 5 model that is powered by the Intel Core Ultra 5 125U processor.

Your report must include the following information with supporting reference URLs:

1. Model Identification: Confirm the exact model name and processor
2. Processor Architecture: Specify the number of cores, number of threads, and maximum turbo boost frequency
3. Memory System: Specify the memory technology type, maximum memory capacity supported, and number of memory slots with their configuration
4. Physical Specifications: Specify the available display resolution option(s), battery capacity in watt-hours, and the laptop's weight
5. Purchase and Support: Indicate the availability of Lenovo Premier Support warranty upgrade option and provide at least one online retailer (Best Buy or Lenovo official website) where this specific configuration can be purchased

Please provide reference URLs from official Lenovo specifications pages or authorized retailers to verify the information.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ModelIdentification(BaseModel):
    model_name: Optional[str] = None
    cpu_name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ProcessorArchitecture(BaseModel):
    cores: Optional[str] = None
    threads: Optional[str] = None
    max_turbo: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class MemorySystem(BaseModel):
    memory_tech: Optional[str] = None
    max_capacity: Optional[str] = None
    slots: Optional[str] = None
    dual_channel: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class PhysicalSpecifications(BaseModel):
    display_resolutions: List[str] = Field(default_factory=list)
    battery_capacity_wh: Optional[str] = None
    weight: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class PurchaseAndSupport(BaseModel):
    premier_support_available: Optional[str] = None
    premier_urls: List[str] = Field(default_factory=list)
    purchase_urls: List[str] = Field(default_factory=list)


class SpecReportExtraction(BaseModel):
    model: Optional[ModelIdentification] = None
    processor: Optional[ProcessorArchitecture] = None
    memory: Optional[MemorySystem] = None
    physical: Optional[PhysicalSpecifications] = None
    purchase: Optional[PurchaseAndSupport] = None
    all_reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_spec_report() -> str:
    return """
    Extract the structured information provided in the answer for the Lenovo ThinkPad T14 Gen 5 with Intel Core Ultra 5 125U.
    Capture exactly what is stated in the answer and list all reference URLs mentioned.

    Return a JSON object with the following structure:
    {
      "model": {
        "model_name": "...",
        "cpu_name": "...",
        "urls": ["...", "..."]
      },
      "processor": {
        "cores": "...",
        "threads": "...",
        "max_turbo": "...",
        "urls": ["...", "..."]
      },
      "memory": {
        "memory_tech": "...",
        "max_capacity": "...",
        "slots": "...",
        "dual_channel": "...",
        "urls": ["...", "..."]
      },
      "physical": {
        "display_resolutions": ["...", "..."],
        "battery_capacity_wh": "...",
        "weight": "...",
        "urls": ["...", "..."]
      },
      "purchase": {
        "premier_support_available": "...",
        "premier_urls": ["...", "..."],
        "purchase_urls": ["...", "..."]
      },
      "all_reference_urls": ["...", "..."]
    }

    Notes:
    - urls arrays should include all URLs explicitly present in the answer for that section.
    - all_reference_urls should include all URLs mentioned anywhere in the answer (deduplicate if possible).
    - For numbers or units, keep the exact text as written in the answer (e.g., "52.5Wh", "2.86 lb", "DDR5-5600").
    - If any field is not mentioned in the answer, return null (or an empty list for arrays).
    - Extract only URLs explicitly present in the answer. If a URL is missing protocol, prepend http://
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def union_urls(*url_lists: Optional[List[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if not u or not isinstance(u, str):
                continue
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


def collect_all_urls(spec: SpecReportExtraction) -> List[str]:
    return union_urls(
        spec.all_reference_urls if spec else [],
        spec.model.urls if spec and spec.model else [],
        spec.processor.urls if spec and spec.processor else [],
        spec.memory.urls if spec and spec.memory else [],
        spec.physical.urls if spec and spec.physical else [],
        spec.purchase.premier_urls if spec and spec.purchase else [],
        spec.purchase.purchase_urls if spec and spec.purchase else [],
    )


def is_allowed_url(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc.endswith("lenovo.com") or netloc.endswith("bestbuy.com")
    except Exception:
        return False


def has_any_allowed_purchase_link(purchase_urls: List[str]) -> bool:
    for u in purchase_urls or []:
        if is_allowed_url(u):
            return True
    return False


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_model_identification(
    evaluator: Evaluator,
    parent: VerificationNode,
    spec: SpecReportExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Model_Identification",
        desc="Confirm the exact model name and processor for the requested configuration, with supporting references.",
        parent=parent,
        critical=True,
    )

    model_urls = union_urls(
        spec.model.urls if spec and spec.model else [],
        spec.purchase.purchase_urls if spec and spec.purchase else [],
    )

    # Presence/gating: at least one supporting reference URL
    evaluator.add_custom_node(
        result=len(model_urls) > 0,
        id="Model_Identification_Has_Supporting_URL",
        desc="Provides at least one supporting reference URL corroborating the model and CPU configuration.",
        parent=node,
        critical=True,
    )

    # Model is ThinkPad T14 Gen 5
    leaf_model = evaluator.add_leaf(
        id="Model_Is_T14_Gen5",
        desc="States the laptop model as Lenovo ThinkPad T14 Gen 5 (Intel variant acceptable if specified).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The laptop model shown on the referenced page(s) is Lenovo ThinkPad T14 Gen 5. Allow reasonable variants like 'ThinkPad T14 Gen 5 (Intel)'.",
        node=leaf_model,
        sources=model_urls,
        additional_instruction="Accept minor naming variants such as 'Gen5' vs 'Gen 5', and optional 'Intel' suffix. Reject if it is not T14 Gen 5.",
    )

    # CPU is Intel Core Ultra 5 125U
    leaf_cpu = evaluator.add_leaf(
        id="CPU_Is_Intel_Core_Ultra_5_125U",
        desc="States the processor as Intel Core Ultra 5 125U.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This configuration uses an Intel Core Ultra 5 125U processor.",
        node=leaf_cpu,
        sources=model_urls,
        additional_instruction="The page must clearly indicate 'Intel Core Ultra 5 125U' for the CPU in the configuration or specs.",
    )


async def build_processor_architecture(
    evaluator: Evaluator,
    parent: VerificationNode,
    spec: SpecReportExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Processor_Architecture",
        desc="Provide processor architecture details meeting the constraints, with supporting references.",
        parent=parent,
        critical=True,
    )

    proc_urls = union_urls(
        spec.processor.urls if spec and spec.processor else [],
        spec.model.urls if spec and spec.model else [],
        spec.purchase.purchase_urls if spec and spec.purchase else [],
    )

    # Presence/gating
    evaluator.add_custom_node(
        result=len(proc_urls) > 0,
        id="Processor_Architecture_Has_Supporting_URL",
        desc="Provides at least one supporting reference URL for the processor core/thread/turbo claims.",
        parent=node,
        critical=True,
    )

    # 12 cores
    cores_leaf = evaluator.add_leaf(
        id="Processor_Cores_Exactly_12",
        desc="States the processor has exactly 12 cores.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The Intel Core Ultra 5 125U processor has 12 cores.",
        node=cores_leaf,
        sources=proc_urls,
        additional_instruction="Accept breakdowns like '12 cores total (e.g., 2P+8E+2LP-E)' as confirming 12 cores.",
    )

    # 14 threads
    threads_leaf = evaluator.add_leaf(
        id="Processor_Threads_Exactly_14",
        desc="States the processor has exactly 14 threads.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The Intel Core Ultra 5 125U processor has 14 threads.",
        node=threads_leaf,
        sources=proc_urls,
        additional_instruction="If threads count is explicitly 14 (total), accept.",
    )

    # Max turbo up to 4.3 GHz
    turbo_leaf = evaluator.add_leaf(
        id="Max_Turbo_Up_To_4_3_GHz",
        desc="States maximum turbo/boost frequency is up to 4.3 GHz (allow equivalent formatting such as 4.30 GHz).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The Intel Core Ultra 5 125U has a maximum turbo/boost frequency up to 4.3 GHz.",
        node=turbo_leaf,
        sources=proc_urls,
        additional_instruction="Accept formatting like 4.30 GHz. Reject values that are higher or lower than 4.3 GHz if clearly stated.",
    )


async def build_memory_system(
    evaluator: Evaluator,
    parent: VerificationNode,
    spec: SpecReportExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Memory_System",
        desc="Provide memory technology, max capacity, and slot configuration meeting the constraints, with supporting references.",
        parent=parent,
        critical=True,
    )

    mem_urls = union_urls(
        spec.memory.urls if spec and spec.memory else [],
        spec.model.urls if spec and spec.model else [],
        spec.purchase.purchase_urls if spec and spec.purchase else [],
    )

    # Presence/gating
    evaluator.add_custom_node(
        result=len(mem_urls) > 0,
        id="Memory_System_Has_Supporting_URL",
        desc="Provides at least one supporting reference URL for the memory technology/capacity/slot claims.",
        parent=node,
        critical=True,
    )

    # DDR5-5600
    tech_leaf = evaluator.add_leaf(
        id="Memory_Technology_DDR5_5600",
        desc="States the system supports DDR5-5600 memory technology.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This ThinkPad T14 Gen 5 (Intel) supports DDR5-5600 memory.",
        node=tech_leaf,
        sources=mem_urls,
        additional_instruction="Accept equivalent notation like 'DDR5 5600' or 'DDR5-5600MHz'.",
    )

    # Max 64GB
    max_leaf = evaluator.add_leaf(
        id="Max_Memory_64GB",
        desc="States the maximum supported memory capacity is 64GB.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The maximum supported memory capacity is 64GB.",
        node=max_leaf,
        sources=mem_urls,
        additional_instruction="Accept variants like 'up to 64 GB' or 'max 64GB', including configurations of 2x32GB.",
    )

    # Two SO-DIMM Dual Channel
    slots_leaf = evaluator.add_leaf(
        id="Two_SO_DIMM_Dual_Channel",
        desc="States the system has 2 SO-DIMM slots and indicates dual-channel configuration support.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This system has 2 SO-DIMM memory slots and supports dual-channel memory.",
        node=slots_leaf,
        sources=mem_urls,
        additional_instruction="Accept expressions like '2x SODIMM', 'dual-channel', '2 slots'.",
    )


async def build_physical_specifications(
    evaluator: Evaluator,
    parent: VerificationNode,
    spec: SpecReportExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Physical_Specifications",
        desc="Provide display resolution option(s), battery capacity (Wh), and weight meeting the constraints, with supporting references.",
        parent=parent,
        critical=True,
    )

    phys_urls = union_urls(
        spec.physical.urls if spec and spec.physical else [],
        spec.model.urls if spec and spec.model else [],
        spec.purchase.purchase_urls if spec and spec.purchase else [],
    )

    # Presence/gating
    evaluator.add_custom_node(
        result=len(phys_urls) > 0,
        id="Physical_Specs_Has_Supporting_URL",
        desc="Provides at least one supporting reference URL for display/battery/weight claims.",
        parent=node,
        critical=True,
    )

    # WUXGA option
    wuxga_leaf = evaluator.add_leaf(
        id="Includes_WUXGA_1920x1200_Option",
        desc="States that WUXGA (1920×1200) is an available display resolution option (it may list additional options, but must include WUXGA 1920×1200).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="WUXGA (1920×1200) is one of the available display options for this model.",
        node=wuxga_leaf,
        sources=phys_urls,
        additional_instruction="Accept '1920 x 1200' or 'WUXGA 1920x1200' or similar formatting.",
    )

    # Battery 52.5Wh
    battery_leaf = evaluator.add_leaf(
        id="Battery_Capacity_52_5Wh",
        desc="States the battery capacity is 52.5Wh.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The battery capacity is 52.5Wh.",
        node=battery_leaf,
        sources=phys_urls,
        additional_instruction="Accept notations like '52.5 Wh' or '52.5Whr'. Reject substantially different capacities.",
    )

    # Weight range
    weight_leaf = evaluator.add_leaf(
        id="Weight_In_2_86_to_3_15_lb_Range",
        desc="States laptop weight (with units) consistent with approximately 2.86 to 3.15 pounds (may be a single value within the range or a stated range overlapping it).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The laptop's weight is within approximately 2.86 to 3.15 pounds.",
        node=weight_leaf,
        sources=phys_urls,
        additional_instruction=(
            "If weight is provided in kg, convert to pounds (1 kg ≈ 2.20462 lb). "
            "Accept single values within the range or ranges that overlap with 2.86–3.15 lb. "
            "Allow minor rounding differences."
        ),
    )


async def build_purchase_and_support(
    evaluator: Evaluator,
    parent: VerificationNode,
    spec: SpecReportExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Purchase_And_Support",
        desc="Indicate Premier Support upgrade availability and provide an allowed purchase link, with supporting references where required.",
        parent=parent,
        critical=True,
    )

    premier_urls = (spec.purchase.premier_urls if spec and spec.purchase else []) or []
    purchase_urls = (spec.purchase.purchase_urls if spec and spec.purchase else []) or []

    # Premier Support presence/gating
    evaluator.add_custom_node(
        result=len(premier_urls) > 0,
        id="Premier_Support_Has_Supporting_URL",
        desc="Provides at least one supporting reference URL corroborating Premier Support upgrade availability.",
        parent=node,
        critical=True,
    )

    # Premier Support availability
    premier_leaf = evaluator.add_leaf(
        id="Premier_Support_Upgrade_Available",
        desc="Indicates that a Lenovo Premier Support warranty upgrade option is available for this model/configuration.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="A Lenovo Premier Support warranty upgrade option is available for this model/configuration.",
        node=premier_leaf,
        sources=premier_urls,
        additional_instruction="Accept references to 'Premier Support' or 'Premier Support Plus' as valid upgrade options for warranty/support.",
    )

    # Purchase link allowed source (Best Buy or Lenovo)
    evaluator.add_custom_node(
        result=has_any_allowed_purchase_link(purchase_urls),
        id="Purchase_Link_Is_From_BestBuy_Or_Lenovo",
        desc="Provides at least one purchase link from either Best Buy or the official Lenovo website for the specified model/CPU configuration.",
        parent=node,
        critical=True,
    )


def add_url_source_compliance(
    evaluator: Evaluator,
    parent: VerificationNode,
    spec: SpecReportExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="URL_Source_Compliance",
        desc="All reference URLs comply with the allowed-source requirement.",
        parent=parent,
        critical=True,
    )

    all_urls = collect_all_urls(spec)
    # Vacuously true if no URLs; other nodes should fail on missing URLs.
    all_allowed = all(is_allowed_url(u) for u in all_urls) if all_urls else True

    evaluator.add_custom_node(
        result=all_allowed,
        id="All_URLs_Are_Lenovo_Specs_Or_Authorized_Retailers",
        desc="All provided reference URLs are from official Lenovo specifications pages or authorized retailers (including Best Buy / official Lenovo store pages) as requested.",
        parent=node,
        critical=True,
    )

    # Optionally record diagnostic info
    evaluator.add_custom_info(
        info={
            "total_urls_collected": len(all_urls),
            "disallowed_urls": [u for u in all_urls if not is_allowed_url(u)],
        },
        info_type="url_compliance",
        info_name="url_source_compliance_details",
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
    # Initialize evaluator with parallel root
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

    # Extract structured information from the answer
    extracted: SpecReportExtraction = await evaluator.extract(
        prompt=prompt_extract_spec_report(),
        template_class=SpecReportExtraction,
        extraction_name="spec_report_extraction",
    )

    # Build a top-level critical node for the report
    report_node = evaluator.add_parallel(
        id="ThinkPad_T14_Gen5_Ultra5_125U_Spec_Report",
        desc="Provide comprehensive specifications for Lenovo ThinkPad T14 Gen 5 with Intel Core Ultra 5 125U, meeting all listed constraints and including supporting reference URLs from allowed sources.",
        parent=root,
        critical=True,
    )

    # Build each section (all critical under the report)
    await build_model_identification(evaluator, report_node, extracted)
    await build_processor_architecture(evaluator, report_node, extracted)
    await build_memory_system(evaluator, report_node, extracted)
    await build_physical_specifications(evaluator, report_node, extracted)
    await build_purchase_and_support(evaluator, report_node, extracted)
    add_url_source_compliance(evaluator, report_node, extracted)

    # Return summary
    return evaluator.get_summary()