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
TASK_ID = "gaming_laptop_upgrade_warranty"
TASK_DESCRIPTION = (
    "I am looking to purchase a gaming laptop that meets specific hardware upgradeability and manufacturer support "
    "requirements. Find one laptop model that satisfies ALL of the following criteria:\n\n"
    "1. Must feature an NVIDIA GeForce RTX 4070 mobile GPU (8GB GDDR6)\n"
    "2. Must have user-accessible, upgradeable RAM slots (SO-DIMM slots, not soldered RAM)\n"
    "3. Must have at least 2 M.2 SSD slots for storage expansion (both PCIe NVMe compatible)\n"
    "4. The manufacturer must offer International Warranty Service that covers North America\n"
    "5. The manufacturer must have at least one authorized service center in New York City, New York\n"
    "6. The manufacturer must offer on-site warranty service as an available option\n\n"
    "For your answer, provide:\n"
    "- The specific laptop model name and manufacturer\n"
    "- Confirmation of the RTX 4070 GPU specification\n"
    "- Confirmation of upgradeable RAM configuration (number and type of slots)\n"
    "- Confirmation of the number of M.2 SSD slots\n"
    "- A link to official information about the manufacturer's international warranty coverage for North America\n"
    "- The address and contact information for at least one authorized service center in New York City\n"
    "- A link to official information about the availability of on-site warranty service"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ServiceCenterInfo(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    url: Optional[str] = None


class LaptopSelection(BaseModel):
    # Identification
    model_name: Optional[str] = None
    manufacturer: Optional[str] = None

    # Official product/spec sources (manufacturer-owned pages)
    product_urls: List[str] = Field(default_factory=list)

    # GPU details
    gpu_model: Optional[str] = None           # e.g., "NVIDIA GeForce RTX 4070 Laptop GPU"
    gpu_vram: Optional[str] = None            # e.g., "8GB GDDR6"

    # RAM upgradeability
    ram_slots_count: Optional[str] = None     # e.g., "2"
    ram_slot_type: Optional[str] = None       # e.g., "DDR5 SO-DIMM"
    ram_user_upgradeable: Optional[str] = None  # "yes"/"no" or textual confirmation

    # Storage expansion
    m2_slots_count: Optional[str] = None      # e.g., "2"
    m2_nvme_for_all: Optional[str] = None     # e.g., "yes" or text confirming both NVMe-compatible

    # Warranty & service URLs
    iws_url: Optional[str] = None             # International Warranty Service official page URL
    onsite_service_url: Optional[str] = None  # On-site warranty service official page URL

    # Authorized service center in NYC
    service_center: Optional[ServiceCenterInfo] = None

    # Any additional official URLs provided in the answer (support/manual/spec)
    extra_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_laptop_info() -> str:
    return (
        "Extract the single recommended gaming laptop model and all required supporting details as presented in the answer. "
        "Return a JSON object matching the following fields:\n"
        "- model_name: The exact laptop model name the answer recommends (string). If multiple models are mentioned, pick the single one the answer ultimately recommends; otherwise, pick the first mentioned model.\n"
        "- manufacturer: The manufacturer/brand for the selected model (string).\n"
        "- product_urls: An array of official manufacturer URLs relevant to this model (e.g., product page, specifications, support/manual). Only include manufacturer-owned pages; exclude news/reviews/retailers unless the answer explicitly uses them as official sources.\n"
        "- gpu_model: The GPU model stated for the selected laptop (string), e.g., 'NVIDIA GeForce RTX 4070 Laptop GPU'.\n"
        "- gpu_vram: The VRAM capacity/type stated for the selected laptop GPU (string), e.g., '8GB GDDR6'.\n"
        "- ram_slots_count: The number of user-accessible RAM slots mentioned (string). If described qualitatively (e.g., 'two slots'), normalize to a simple string like '2' when possible; otherwise keep the original phrasing.\n"
        "- ram_slot_type: The RAM slot type (string), e.g., 'DDR5 SO-DIMM'.\n"
        "- ram_user_upgradeable: A simple 'yes'/'no' string (lowercase) indicating if RAM is user-accessible/upgradeable (from the answer).\n"
        "- m2_slots_count: The number of M.2 SSD slots (string); normalize to a simple count when possible.\n"
        "- m2_nvme_for_all: A simple 'yes'/'no' string or short text indicating whether all M.2 slots support PCIe NVMe.\n"
        "- iws_url: A single official URL to the manufacturer's International Warranty Service page that covers North America (string). If multiple are present, choose the most specific.\n"
        "- onsite_service_url: A single official URL to the manufacturer's page confirming on-site warranty service availability (string).\n"
        "- service_center: An object with fields {name, address, phone, email, url} for at least one authorized service center in New York City (NYC). All fields are strings; any missing field should be null. The 'url' should be an official manufacturer page showing the service center.\n"
        "- extra_urls: Any additional official URLs present in the answer that may support hardware specs (array of strings).\n\n"
        "Rules:\n"
        "1) Extract only what the answer explicitly provides; do not invent details. If a field is missing, return null (or empty array where appropriate).\n"
        "2) For URLs, include only valid, complete URLs. If a URL is missing a protocol, prepend 'http://'.\n"
        "3) Do not include duplicates in product_urls or extra_urls.\n"
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(*url_groups: List[str]) -> List[str]:
    """Merge and de-duplicate URL lists, preserving order."""
    seen = set()
    merged: List[str] = []
    for group in url_groups:
        for url in group:
            if not url:
                continue
            if url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


def _non_empty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root_node,
    info: LaptopSelection,
) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    All children under Laptop_Solution are marked critical to enforce ALL constraints.
    """
    # Parent critical node for the overall solution
    solution_node = evaluator.add_parallel(
        id="Laptop_Solution",
        desc="Answer identifies one gaming laptop model that satisfies all hardware and manufacturer support requirements and includes the required supporting information/links.",
        parent=root_node,
        critical=True,
    )

    # Collect general spec sources for hardware checks
    spec_sources = _merge_sources(info.product_urls, info.extra_urls)

    # 1) Model and Manufacturer Provided
    node_model = evaluator.add_leaf(
        id="Model_And_Manufacturer_Provided",
        desc="Provides exactly one specific laptop model name and its manufacturer.",
        parent=solution_node,
        critical=True,
    )
    mm_model = info.model_name or ""
    mm_mfr = info.manufacturer or ""
    claim_model = (
        f"The answer provides exactly one specific laptop model and its manufacturer, namely '{mm_model}' by '{mm_mfr}'."
    )
    await evaluator.verify(
        claim=claim_model,
        node=node_model,
        sources=None,  # This check is based on the answer text itself
        additional_instruction=(
            "Check the entire answer. The answer must provide a single, specific laptop model and its manufacturer. "
            "If more than one model is proposed or if either the model name or the manufacturer is missing/unclear, judge incorrect."
        ),
    )

    # 2) GPU Specification
    node_gpu = evaluator.add_leaf(
        id="GPU_Specification",
        desc="Confirms the laptop features an NVIDIA GeForce RTX 4070 mobile GPU with 8GB GDDR6.",
        parent=solution_node,
        critical=True,
    )
    claim_gpu = (
        "The selected laptop features an NVIDIA GeForce RTX 4070 Laptop (mobile) GPU with 8GB GDDR6 VRAM."
    )
    await evaluator.verify(
        claim=claim_gpu,
        node=node_gpu,
        sources=spec_sources if spec_sources else None,
        additional_instruction=(
            "Use official product/spec pages if available. Allow reasonable naming variants such as "
            "'GeForce RTX 4070 Laptop GPU' or 'mobile GPU'. Explicitly confirm 8GB GDDR6 VRAM. "
            "If official sources are unavailable but the answer text clearly states the correct GPU and VRAM, it may be acceptable."
        ),
    )

    # 3) RAM Upgradeability via SO-DIMM (not soldered), include number/type of slots
    node_ram = evaluator.add_leaf(
        id="RAM_Upgradeability",
        desc="Confirms the laptop has user-accessible, upgradeable RAM via SO-DIMM slots (not soldered), including number/type of slots.",
        parent=solution_node,
        critical=True,
    )
    slots_text = info.ram_slots_count or "the required number of"
    slot_type_text = info.ram_slot_type or "SO-DIMM"
    claim_ram = (
        f"The laptop has user-accessible, upgradeable RAM via SO-DIMM slots (not soldered). "
        f"It offers {slots_text} {slot_type_text} slots that the user can access and upgrade."
    )
    await evaluator.verify(
        claim=claim_ram,
        node=node_ram,
        sources=spec_sources if spec_sources else None,
        additional_instruction=(
            "Verify that RAM is not soldered and uses SO-DIMM slots that are user-accessible/upgradeable. "
            "Confirm the number of slots and the slot type where stated on official pages (product specs or service manual). "
            "Minor phrasing variations are acceptable as long as the requirement is clearly satisfied."
        ),
    )

    # 4) Dual M.2 Slots (both NVMe compatible)
    node_m2 = evaluator.add_leaf(
        id="Dual_M2_Slots",
        desc="Confirms the laptop has at least 2 M.2 SSD slots and that both are PCIe NVMe compatible.",
        parent=solution_node,
        critical=True,
    )
    claim_m2 = (
        "The laptop has at least 2 M.2 SSD slots and both are PCIe NVMe compatible."
    )
    await evaluator.verify(
        claim=claim_m2,
        node=node_m2,
        sources=spec_sources if spec_sources else None,
        additional_instruction=(
            "Confirm that there are two (or more) M.2 slots and that both support PCIe NVMe (not just SATA). "
            "Use official product/spec pages or service manuals if provided."
        ),
    )

    # 5) International Warranty Service covering North America, with official link
    node_iws = evaluator.add_leaf(
        id="International_Warranty_North_America_With_Official_Link",
        desc="Confirms the manufacturer offers International Warranty Service covering North America and provides a link to official information documenting this.",
        parent=solution_node,
        critical=True,
    )
    claim_iws = (
        "The manufacturer offers International Warranty Service that covers North America, and the answer includes at least one official URL documenting this."
    )
    await evaluator.verify(
        claim=claim_iws,
        node=node_iws,
        sources=info.iws_url if _non_empty(info.iws_url) else None,
        additional_instruction=(
            "Use only official manufacturer webpages. The page must explicitly indicate international warranty coverage "
            "and that North America (including the USA) is covered. If no official link is provided in the answer, judge incorrect."
        ),
    )

    # 6) NYC authorized service center details (address + contact info)
    node_nyc = evaluator.add_leaf(
        id="NYC_Authorized_Service_Center_Details",
        desc="Provides the address and contact information for at least one manufacturer-authorized service center located in New York City, New York.",
        parent=solution_node,
        critical=True,
    )
    sc = info.service_center or ServiceCenterInfo()
    contact = sc.phone or sc.email or ""
    claim_nyc = (
        f"The manufacturer has at least one authorized service center located in New York City, New York. "
        f"The answer provides one with address and contact information: name '{sc.name or ''}', "
        f"address '{sc.address or ''}', contact '{contact}'."
    )
    await evaluator.verify(
        claim=claim_nyc,
        node=node_nyc,
        sources=sc.url if _non_empty(sc.url) else None,
        additional_instruction=(
            "Confirm that the listed service center is manufacturer-authorized and located in NYC (New York, NY). "
            "The answer must include both an address and at least one contact method (phone or email). "
            "If an official service locator page URL is provided, use it; otherwise, rely on the answer text. "
            "If either address or contact info is missing, judge incorrect."
        ),
    )

    # 7) On-site warranty service available, with official link
    node_onsite = evaluator.add_leaf(
        id="Onsite_Warranty_Service_With_Official_Link",
        desc="Confirms on-site warranty service is available as an option from the manufacturer and provides a link to official information documenting availability.",
        parent=solution_node,
        critical=True,
    )
    claim_onsite = (
        "The manufacturer offers on-site warranty service as an available option, and the answer includes at least one official URL documenting this."
    )
    await evaluator.verify(
        claim=claim_onsite,
        node=node_onsite,
        sources=info.onsite_service_url if _non_empty(info.onsite_service_url) else None,
        additional_instruction=(
            "Use only official manufacturer webpages. The page must explicitly indicate on-site warranty service availability. "
            "If no official link is provided in the answer, judge incorrect."
        ),
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
    Evaluate an answer for the gaming laptop upgradeability and manufacturer support requirements.
    """
    # Initialize evaluator with a parallel root
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

    # Extraction
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_laptop_info(),
        template_class=LaptopSelection,
        extraction_name="laptop_selection",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, root, extracted_info)

    # Return structured summary
    return evaluator.get_summary()