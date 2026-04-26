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
TASK_ID = "ceas_ceas_laptop_2024_2025"
TASK_DESCRIPTION = """A student has been admitted to the University of Cincinnati's College of Engineering and Applied Science (CEAS) for the Fall 2024 academic year. According to CEAS's published minimum laptop requirements for 2024-2025, the student must purchase a laptop that meets ALL of the following mandatory specifications:

- Processor: Intel 13th Generation i7 OR Intel 14th Generation i7
- Memory: Minimum 16GB DDR4 RAM
- Graphics: Dedicated graphics card with minimum 4GB VRAM (not integrated graphics)
- Storage: Minimum 500GB SSD (solid-state drive)
- Battery Life: Minimum 5 hours rated battery life
- Display: Minimum 1920x1080 (1080p) resolution
- Operating System: Must support Windows 10 or Windows 11

Identify ONE specific laptop model currently available for purchase that meets all of these requirements. For your answer, provide:

1. The exact laptop model name and manufacturer
2. Verification of each specification requirement with the specific values for that model
3. Reference URL(s) from the manufacturer's official website or authorized retailer (such as Best Buy, Amazon, HP Store, etc.) that confirm all specifications
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class LaptopCandidate(BaseModel):
    manufacturer: Optional[str] = None
    model_name: Optional[str] = None

    # Raw spec strings exactly as stated in the answer (be flexible in format)
    processor: Optional[str] = None
    memory: Optional[str] = None          # e.g., "16GB DDR5", "16 GB LPDDR5"
    graphics: Optional[str] = None        # e.g., "NVIDIA GeForce RTX 3050 4GB", "Intel Arc A370M 4GB"
    storage: Optional[str] = None         # e.g., "512GB SSD", "1TB NVMe SSD"
    battery_life: Optional[str] = None    # e.g., "8 hours", "up to 7.5 hours"
    display_resolution: Optional[str] = None  # e.g., "1920x1080", "2560x1600"
    operating_system: Optional[str] = None    # e.g., "Windows 11 Home"

    # All citation URLs provided in the answer (manufacturer or authorized retailers)
    citations: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_laptop() -> str:
    return """
Extract exactly one specific laptop model proposed in the answer (if multiple are listed, extract the first one only). Return a JSON object with:

- manufacturer: The laptop manufacturer/brand (e.g., Dell, HP, Lenovo, ASUS, Acer, MSI, etc.)
- model_name: The exact model designation (e.g., "G15 5530", "Victus 15-fa1xxx", "ThinkPad P1 Gen 6", etc.)
- processor: The CPU as written (e.g., "Intel Core i7-13700H", "Intel Core i7 14th Gen")
- memory: The RAM spec as written (e.g., "16GB DDR4", "16 GB LPDDR5", "16GB DDR5-5200")
- graphics: The GPU and VRAM info as written (e.g., "NVIDIA GeForce RTX 3050 4GB", "AMD Radeon RX 6500M 4GB", "Intel Arc A370M 4GB")
- storage: The storage spec as written (e.g., "512GB SSD", "1TB NVMe SSD")
- battery_life: The rated battery life as written (e.g., "up to 7 hours", "8 hours")
- display_resolution: The display resolution as written (e.g., "1920x1080", "1920 x 1200", "2560x1600", "4K (3840x2160)")
- operating_system: The OS as written (e.g., "Windows 11 Home", "Windows 11 Pro", "Windows 10 Pro")
- citations: An array of all URLs cited in the answer that are intended to verify the specs or availability. Only include actual URLs present in the answer text. Prefer official manufacturer or known authorized retailers (e.g., manufacturer.com, bestbuy.com, amazon.com, microsoft.com, lenovo.com, hp.com, dell.com, asus.com, acer.com, msi.com, walmart.com, target.com, newegg.com, microcenter.com, bhphotovideo.com, adorama.com, costco.com, staples.com, officedepot.com). If none are present, return an empty array.

Important:
- Do not invent or infer values. Extract exactly what appears in the answer.
- If any field is missing in the answer, set it to null (for strings) or [] (for citations).
- Keep values as strings exactly as written; do not normalize or convert units.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def pretty_model_name(specs: LaptopCandidate) -> str:
    maker = (specs.manufacturer or "").strip()
    model = (specs.model_name or "").strip()
    if maker and model:
        return f"{maker} {model}"
    if maker:
        return maker
    if model:
        return model
    return "the identified laptop model"


def sources_or_none(urls: Optional[List[str]]) -> Optional[List[str]]:
    if not urls:
        return None
    cleaned = [u for u in urls if isinstance(u, str) and u.strip()]
    return cleaned if cleaned else None


def urls_as_bulleted_list(urls: List[str]) -> str:
    if not urls:
        return "(no URLs provided)"
    return "\n".join(f"- {u}" for u in urls)


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_laptop_candidate(evaluator: Evaluator, parent_node, specs: LaptopCandidate) -> None:
    # Create the main compliance node (critical, parallel aggregation)
    compliance_node = evaluator.add_parallel(
        id="CEAS_Engineering_Laptop_Compliance",
        desc="Identify one currently purchasable laptop that satisfies all CEAS minimum specs and provide verifiable citations from official/authorized sources",
        parent=parent_node,
        critical=True
    )

    # 1) Laptop Identification
    ident_node = evaluator.add_parallel(
        id="Laptop_Identification",
        desc="Answer clearly identifies the proposed laptop",
        parent=compliance_node,
        critical=True
    )
    # Leaf: Model name and manufacturer provided
    evaluator.add_custom_node(
        result=bool((specs.manufacturer or "").strip()) and bool((specs.model_name or "").strip()),
        id="Model_Name_And_Manufacturer_Provided",
        desc="Provides the exact laptop model name and manufacturer",
        parent=ident_node,
        critical=True
    )

    # 2) Meets Technical Specifications
    specs_node = evaluator.add_parallel(
        id="Meets_Technical_Specifications",
        desc="Laptop meets all mandatory technical specifications (with model-specific values provided)",
        parent=compliance_node,
        critical=True
    )

    # Collect created spec leaf nodes to serve as prerequisites later (for citation coverage)
    spec_leaves: List[Any] = []

    # 2.a) Processor requirement
    cpu_leaf = evaluator.add_leaf(
        id="Processor_Requirement",
        desc="Processor is Intel 13th Gen Core i7 OR Intel 14th Gen Core i7 (not i3/i5/i9)",
        parent=specs_node,
        critical=True
    )
    spec_leaves.append(cpu_leaf)
    cpu_claim = (
        f"The product page(s) for {pretty_model_name(specs)} indicate the CPU is an Intel Core i7 from either the 13th or 14th generation. "
        f"Accept explicit forms like 'Intel Core i7-13xxx' or 'Intel Core i7 13th Gen', or 'Intel Core i7-14xxx'/'14th Gen'. "
        f"Reject i3/i5/i9, AMD, Apple, or anything other than Intel Core i7 13th/14th Gen. "
        f"Extracted answer CPU text (for context): '{(specs.processor or '').strip()}'."
    )
    await evaluator.verify(
        claim=cpu_claim,
        node=cpu_leaf,
        sources=sources_or_none(specs.citations),
        additional_instruction="Check the CPU field carefully on the provided pages; if ambiguous or missing, answer Incorrect."
    )

    # 2.b) Memory requirement (>=16GB and DDR4 or DDR5; allow LPDDR4x/LPDDR5 as DDR family equivalents)
    mem_leaf = evaluator.add_leaf(
        id="Memory_Requirement",
        desc="System RAM is at least 16GB and is DDR4 or DDR5",
        parent=specs_node,
        critical=True
    )
    spec_leaves.append(mem_leaf)
    mem_claim = (
        f"The product page(s) for {pretty_model_name(specs)} show system memory is at least 16 GB and uses DDR4-family or DDR5-family memory. "
        f"Accept 'DDR4', 'DDR5', 'LPDDR4x', or 'LPDDR5' (these are DDR families). "
        f"Reject if <16 GB or a non-DDR memory not equivalent to DDR4/DDR5. "
        f"Extracted answer memory text (for context): '{(specs.memory or '').strip()}'."
    )
    await evaluator.verify(
        claim=mem_claim,
        node=mem_leaf,
        sources=sources_or_none(specs.citations),
        additional_instruction="Allow LPDDR4x/LPDDR5 as DDR4/DDR5 families; ensure capacity is ≥16 GB."
    )

    # 2.c) Graphics requirement (discrete/dedicated with >=4GB VRAM)
    gpu_leaf = evaluator.add_leaf(
        id="Graphics_Requirement",
        desc="Has dedicated/discrete graphics (not integrated-only) with at least 4GB VRAM",
        parent=specs_node,
        critical=True
    )
    spec_leaves.append(gpu_leaf)
    gpu_claim = (
        f"The product page(s) for {pretty_model_name(specs)} indicate a discrete/dedicated GPU (not integrated-only) with at least 4 GB of dedicated VRAM. "
        f"Look for mentions like 'NVIDIA GeForce ... 4GB', 'AMD Radeon ... 4GB', or 'Intel Arc ... 4GB'. "
        f"Reject if only integrated graphics (e.g., Intel Iris Xe) or if VRAM is not specified as ≥4 GB. "
        f"Extracted answer graphics text (for context): '{(specs.graphics or '').strip()}'."
    )
    await evaluator.verify(
        claim=gpu_claim,
        node=gpu_leaf,
        sources=sources_or_none(specs.citations),
        additional_instruction="If the page lists both integrated and discrete, ensure the discrete GPU exists and has ≥4GB VRAM."
    )

    # 2.d) Storage requirement (>=500GB SSD)
    storage_leaf = evaluator.add_leaf(
        id="Storage_Requirement",
        desc="Has at least 500GB SSD storage (not HDD)",
        parent=specs_node,
        critical=True
    )
    spec_leaves.append(storage_leaf)
    storage_claim = (
        f"The product page(s) for {pretty_model_name(specs)} indicate at least 500 GB of SSD storage (e.g., 512 GB, 1 TB NVMe SSD). "
        f"Reject HDD-only configurations or SSD capacities under 500 GB. "
        f"Extracted answer storage text (for context): '{(specs.storage or '').strip()}'."
    )
    await evaluator.verify(
        claim=storage_claim,
        node=storage_leaf,
        sources=sources_or_none(specs.citations),
        additional_instruction="Confirm the presence of SSD (PCIe/NVMe acceptable) with capacity ≥500 GB."
    )

    # 2.e) Battery requirement (≥5 hours)
    battery_leaf = evaluator.add_leaf(
        id="Battery_Requirement",
        desc="Rated battery life is at least 5 hours",
        parent=specs_node,
        critical=True
    )
    spec_leaves.append(battery_leaf)
    battery_claim = (
        f"The product page(s) for {pretty_model_name(specs)} state a rated battery life of at least 5 hours (e.g., 'up to 5 hours' or higher). "
        f"Extracted answer battery text (for context): '{(specs.battery_life or '').strip()}'."
    )
    await evaluator.verify(
        claim=battery_claim,
        node=battery_leaf,
        sources=sources_or_none(specs.citations),
        additional_instruction="Use the official or retailer-listed rating; if unspecified or clearly under 5 hours, answer Incorrect."
    )

    # 2.f) Display requirement (>= 1920x1080)
    display_leaf = evaluator.add_leaf(
        id="Display_Requirement",
        desc="Display resolution is at least 1920x1080",
        parent=specs_node,
        critical=True
    )
    spec_leaves.append(display_leaf)
    display_claim = (
        f"The product page(s) for {pretty_model_name(specs)} indicate a display resolution of at least 1920×1080. "
        f"Accept 1920×1080 (FHD), 1920×1200 (WUXGA), 2560×1440 (QHD), 2560×1600, 3840×2160 (4K), etc. "
        f"Reject anything below 1920×1080. "
        f"Extracted answer display text (for context): '{(specs.display_resolution or '').strip()}'."
    )
    await evaluator.verify(
        claim=display_claim,
        node=display_leaf,
        sources=sources_or_none(specs.citations),
        additional_instruction="If only 'FHD' is stated, treat as 1920×1080 unless contradicted."
    )

    # 2.g) OS requirement (Windows 10 or 11)
    os_leaf = evaluator.add_leaf(
        id="Operating_System_Requirement",
        desc="Supports Windows 10 or Windows 11 (64-bit)",
        parent=specs_node,
        critical=True
    )
    spec_leaves.append(os_leaf)
    os_claim = (
        f"The product page(s) for {pretty_model_name(specs)} indicate support for Windows 10 or Windows 11 (64-bit). "
        f"Presence of 'Windows 11 Home/Pro' is sufficient to satisfy this requirement. "
        f"Extracted answer OS text (for context): '{(specs.operating_system or '').strip()}'."
    )
    await evaluator.verify(
        claim=os_claim,
        node=os_leaf,
        sources=sources_or_none(specs.citations),
        additional_instruction="If only Windows 11 is listed, that is acceptable; if only non-Windows OS is listed, answer Incorrect."
    )

    # 3) Availability
    availability_node = evaluator.add_parallel(
        id="Availability",
        desc="Laptop is currently available for purchase (not discontinued)",
        parent=compliance_node,
        critical=True
    )
    avail_leaf = evaluator.add_leaf(
        id="Currently_For_Sale",
        desc="At least one provided source indicates the laptop/listing is currently purchasable",
        parent=availability_node,
        critical=True
    )
    avail_claim = (
        f"At least one of the provided listings for {pretty_model_name(specs)} shows the laptop is currently purchasable (e.g., 'Add to Cart', 'Buy Now', 'In Stock', 'Ships', active price). "
        f"Reject pages that clearly state 'Out of Stock', 'Discontinued', or no buying option."
    )
    await evaluator.verify(
        claim=avail_claim,
        node=avail_leaf,
        sources=sources_or_none(specs.citations),
        additional_instruction="If any citation shows a live purchasable listing, pass; otherwise, fail."
    )

    # 4) Citations and Verifiability
    cites_node = evaluator.add_parallel(
        id="Citations_and_Verifiability",
        desc="Specifications are verifiable via official manufacturer site or authorized retailer URLs",
        parent=compliance_node,
        critical=True
    )

    # 4.a) Authorized or Official Sources
    auth_leaf = evaluator.add_leaf(
        id="Authorized_or_Official_Sources",
        desc="All citation URLs are from the manufacturer’s official website or an authorized retailer",
        parent=cites_node,
        critical=True
    )

    all_urls_str = urls_as_bulleted_list(specs.citations)
    auth_claim = (
        "All of the citation URLs provided in the answer are from the official manufacturer domain or from an authorized/legitimate retailer for laptop sales. "
        "If any URL is not from an official or authorized retailer domain, the claim is Incorrect.\n"
        f"URLs to check:\n{all_urls_str}"
    )
    await evaluator.verify(
        claim=auth_claim,
        node=auth_leaf,
        additional_instruction=(
            "Judge using common retailer knowledge and the website content. Examples of commonly authorized/legitimate retailers include (non-exhaustive): "
            "bestbuy.com, amazon.com, microsoft.com, lenovo.com, hp.com, dell.com, asus.com, acer.com, msi.com, walmart.com, target.com, newegg.com, "
            "microcenter.com, bhphotovideo.com, adorama.com, costco.com, staples.com, officedepot.com. "
            "The official manufacturer domains also qualify. If any provided URL appears dubious or not a recognized retailer/manufacturer, mark Incorrect. "
            "If no URLs are provided, mark Incorrect."
        )
    )

    # 4.b) Citations cover all required specs
    cover_leaf = evaluator.add_leaf(
        id="Citations_Cover_All_Required_Specs",
        desc="Provided citations collectively substantiate each required spec for the identified model (processor, RAM, GPU/VRAM, storage, battery, display, OS)",
        parent=cites_node,
        critical=True
    )
    cover_claim = (
        f"The provided citations collectively substantiate each required spec for {pretty_model_name(specs)}: "
        "processor (Intel Core i7 13th/14th gen), RAM (≥16GB DDR family), discrete GPU with ≥4GB VRAM, SSD storage ≥500GB, "
        "battery life ≥5 hours, display ≥1920×1080, and Windows 10/11 support. "
        "If any required spec cannot be found or confirmed across the provided citations, the claim is Incorrect."
    )
    await evaluator.verify(
        claim=cover_claim,
        node=cover_leaf,
        sources=sources_or_none(specs.citations),
        additional_instruction="Look across all provided pages; collectively they must confirm every required spec.",
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
    Evaluate an answer for the CEAS laptop compliance task (2024–2025).
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

    # Extract the proposed laptop and all cited URLs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_laptop(),
        template_class=LaptopCandidate,
        extraction_name="laptop_candidate"
    )

    # Optional: record simple custom info
    evaluator.add_custom_info(
        info={
            "extracted_model": {
                "manufacturer": extracted.manufacturer,
                "model_name": extracted.model_name,
            },
            "num_citations": len(extracted.citations),
        },
        info_type="metadata",
        info_name="extraction_summary"
    )

    # Build verification tree and run checks
    await verify_laptop_candidate(evaluator, root, extracted)

    # Return final structured summary
    return evaluator.get_summary()