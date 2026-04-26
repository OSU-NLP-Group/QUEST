import asyncio
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nextgen_console_tech_analysis_2024Q4"
TASK_DESCRIPTION = (
    "A gaming hardware reviewer needs to write a comprehensive technical analysis of a next-generation gaming console "
    "for their publication. They require a console that meets ALL of the following criteria:\n\n"
    "1. Released between September 1, 2024 and December 31, 2024\n"
    "2. GPU performance of at least 15 TFLOPS\n"
    "3. Base storage capacity of at least 1TB\n"
    "4. Supports storage expansion beyond the base capacity\n"
    "5. Official retail price under $800 USD\n"
    "6. Has backward compatibility with previous generation games\n"
    "7. Can play Red Dead Redemption 2\n\n"
    "For your analysis, provide the following information with source URLs:\n\n"
    "Console Identification:\n"
    "- Console model name\n"
    "- Manufacturer name\n\n"
    "Technical Specifications (with source URLs for each):\n"
    "- GPU performance in TFLOPS\n"
    "- Base storage capacity\n"
    "- Official USD retail price\n"
    "- Release date\n"
    "- Storage expansion interface type (e.g., M.2 NVMe SSD, proprietary expansion card)\n"
    "- Minimum speed requirements for expansion storage (if applicable)\n"
    "- Maximum supported expansion storage capacity\n\n"
    "Compatibility:\n"
    "- Confirmation that the console has backward compatibility\n"
    "- Confirmation that the console can play Red Dead Redemption 2\n\n"
    "Publisher Information (with source URLs for each):\n"
    "- Name of the publisher of the Grand Theft Auto series\n"
    "- At least 3 worldwide studio locations of this publisher (include city and country/region)\n"
    "- Official release date of Grand Theft Auto VI (GTA 6)\n\n"
    "Storage Calculations:\n"
    "- Total maximum storage capacity (base storage + maximum expansion capacity)\n"
    "- Analysis: Can the base storage alone hold 3 copies of Red Dead Redemption 2? (Note: Red Dead Redemption 2 requires 150GB per copy)\n"
    "  - Show your calculation for total storage needed\n"
    "  - State whether base storage is sufficient\n\n"
    "All factual claims must be supported by source URLs from your research."
)


# --------------------------------------------------------------------------- #
# Pydantic models for data extraction                                         #
# --------------------------------------------------------------------------- #
class StudioLocation(BaseModel):
    city: Optional[str] = None
    country_or_region: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ConsoleAnalysisExtraction(BaseModel):
    # Console Identification
    model_name: Optional[str] = None
    model_name_sources: List[str] = Field(default_factory=list)
    manufacturer_name: Optional[str] = None
    manufacturer_sources: List[str] = Field(default_factory=list)

    # Technical Specs
    gpu_tflops: Optional[str] = None
    gpu_sources: List[str] = Field(default_factory=list)

    base_storage_capacity: Optional[str] = None
    base_storage_sources: List[str] = Field(default_factory=list)

    official_usd_price: Optional[str] = None
    price_sources: List[str] = Field(default_factory=list)

    release_date: Optional[str] = None
    release_date_sources: List[str] = Field(default_factory=list)

    expansion_supported: Optional[str] = None  # "yes/no" or textual confirmation
    expansion_supported_sources: List[str] = Field(default_factory=list)

    expansion_interface_type: Optional[str] = None
    expansion_interface_sources: List[str] = Field(default_factory=list)

    expansion_min_speed_requirement: Optional[str] = None  # Allow "N/A" or "none"
    expansion_min_speed_sources: List[str] = Field(default_factory=list)

    maximum_expansion_capacity: Optional[str] = None
    maximum_expansion_sources: List[str] = Field(default_factory=list)

    # Compatibility
    backward_compatibility: Optional[str] = None
    backward_compatibility_sources: List[str] = Field(default_factory=list)

    rdr2_playable: Optional[str] = None
    rdr2_sources: List[str] = Field(default_factory=list)

    # Publisher Information (GTA series)
    gta_publisher_name: Optional[str] = None
    gta_publisher_sources: List[str] = Field(default_factory=list)
    publisher_studio_locations: List[StudioLocation] = Field(default_factory=list)
    gta6_release_date: Optional[str] = None
    gta6_date_sources: List[str] = Field(default_factory=list)

    # Storage Calculations claimed in the answer
    total_max_storage_capacity: Optional[str] = None  # e.g., "3 TB" or "3072 GB"
    stated_total_needed_for_3_rdr2: Optional[str] = None  # e.g., "450 GB"
    base_storage_sufficient_for_3_rdr2: Optional[str] = None  # yes/no/true/false/sufficient/insufficient


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extraction() -> str:
    return """
Extract the following structured information exactly as stated in the answer. For every field that asks for sources, return the array of the exact URLs present in the answer text (do not invent, and do not include non-URL references). If a field is not stated in the answer, return null (or empty array for sources).

Return a single JSON object with these fields:

Console Identification:
- model_name (string)
- model_name_sources (array of url strings)
- manufacturer_name (string)
- manufacturer_sources (array of url strings)

Technical Specifications:
- gpu_tflops (string; e.g., "15.6 TFLOPS" or "~16 TFLOPS")
- gpu_sources (array of url strings)
- base_storage_capacity (string; e.g., "1 TB", "1024 GB")
- base_storage_sources (array of url strings)
- official_usd_price (string; e.g., "$699", "699 USD")
- price_sources (array of url strings)
- release_date (string; e.g., "November 15, 2024")
- release_date_sources (array of url strings)
- expansion_supported (string; a short confirmation like "yes", "supported", or "no")
- expansion_supported_sources (array of url strings)
- expansion_interface_type (string; e.g., "M.2 NVMe SSD", "proprietary expansion card")
- expansion_interface_sources (array of url strings)
- expansion_min_speed_requirement (string; e.g., "PCIe Gen4 x4, 7000 MB/s minimum"; use "N/A" or "none" if explicitly stated to be not applicable)
- expansion_min_speed_sources (array of url strings)
- maximum_expansion_capacity (string; e.g., "2 TB")
- maximum_expansion_sources (array of url strings)

Compatibility:
- backward_compatibility (string; e.g., "yes"/"no" or a short confirmation)
- backward_compatibility_sources (array of url strings)
- rdr2_playable (string; "yes"/"no" or a short confirmation that Red Dead Redemption 2 can be played)
- rdr2_sources (array of url strings)

Publisher Information (GTA series):
- gta_publisher_name (string)
- gta_publisher_sources (array of url strings)
- publisher_studio_locations (array of objects), each object has:
  - city (string)
  - country_or_region (string)
  - sources (array of url strings)  // specific sources that support this location
  Provide at least 3 if the answer includes them.
- gta6_release_date (string; official release date as stated in the answer)
- gta6_date_sources (array of url strings)

Storage Calculations (as stated in the answer):
- total_max_storage_capacity (string; the answer's stated total max storage = base + max expansion; e.g., "3 TB" or "3072 GB")
- stated_total_needed_for_3_rdr2 (string; the answer’s stated total needed for 3 copies, e.g., "450 GB")
- base_storage_sufficient_for_3_rdr2 (string; "yes"/"no" or "sufficient"/"insufficient")

Rules:
- Do not infer or compute values that are not explicitly given in the answer text (except the storage calculation fields which should reflect what the answer states).
- For URL sources, extract raw URL strings (prepend http:// if protocol missing).
- Keep the original units/formatting for values (do not convert).
    """


# --------------------------------------------------------------------------- #
# Utility parsing helpers                                                     #
# --------------------------------------------------------------------------- #
def _first_number(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    try:
        cleaned = s.replace(",", " ")
    except Exception:
        cleaned = s
    m = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_tflops(value: Optional[str]) -> Optional[float]:
    # Extract first float; assume it's TFLOPS
    return _first_number(value)


def parse_price_usd(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    # Try to locate USD-like number; prioritize $ or USD
    s = value.upper().replace(",", "")
    m = re.search(r"\$?\s*(\d+(?:\.\d+)?)\s*(USD)?", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_capacity_to_gb(value: Optional[str]) -> Optional[float]:
    """
    Parse capacity strings like "1 TB", "1TB", "1024 GB", "2 TiB", "750 GB" into decimal GB.
    Rules:
    - TB/T: 1 TB = 1000 GB (decimal)
    - TiB: 1 TiB = 1024 GiB ≈ 1024 GB (we'll treat as 1024)
    - GB/G: 1 GB = 1 GB
    - GiB: treat as GB (1 GiB ≈ 1.0737 GB) but we approximate to 1.0 for simplicity
    """
    if not value:
        return None
    s = value.strip().lower()
    number = _first_number(s)
    if number is None:
        return None
    # Determine unit
    unit = None
    if "tib" in s:
        unit = "tib"
    elif re.search(r"\bti?\b", s):  # unlikely to occur alone; fallback
        unit = "tib"
    elif "tb" in s or re.search(r"\bt\b", s):
        unit = "tb"
    elif "gib" in s:
        unit = "gib"
    elif "gb" in s or re.search(r"\bg\b", s):
        unit = "gb"
    elif "mb" in s:
        unit = "mb"
    else:
        # No explicit unit; cannot determine reliably; return number as GB guess if it's > 10
        if number > 10:
            return number
        return None

    if unit == "tb":
        return number * 1000.0
    if unit == "tib":
        return number * 1024.0
    if unit == "gib":
        # Approximate 1 GiB as 1 GB for our evaluation tolerance
        return number * 1.0
    if unit == "gb":
        return number * 1.0
    if unit == "mb":
        return number / 1000.0
    return None


def parse_yes_no(s: Optional[str]) -> Optional[bool]:
    if not s:
        return None
    v = s.strip().lower()
    truthy = {"yes", "true", "supported", "support", "compatible", "sufficient"}
    falsy = {"no", "false", "not supported", "unsupported", "incompatible", "insufficient"}
    if v in truthy:
        return True
    if v in falsy:
        return False
    # Fallback: look for positive/negative words
    if any(x in v for x in ["yes", "true", "support", "compatible", "sufficient"]):
        return True
    if any(x in v for x in ["no", "false", "not", "incompatible", "insufficient"]):
        return False
    return None


def non_empty_str(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def has_sources(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


def within_tolerance(a: float, b: float, abs_tol: float = 10.0, rel_tol: float = 0.05) -> bool:
    """
    Returns True if |a-b| <= max(abs_tol, rel_tol * max(|a|, |b|))
    Default abs_tol=10 GB, rel_tol=5%.
    """
    diff = abs(a - b)
    threshold = max(abs_tol, rel_tol * max(abs(a), abs(b), 1.0))
    return diff <= threshold


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_console_identification(evaluator: Evaluator, parent, ex: ConsoleAnalysisExtraction) -> None:
    node = evaluator.add_parallel(
        id="console_identification",
        desc="Console identification provided with supporting sources",
        parent=parent,
        critical=True
    )

    # Model name existence (value + at least one source)
    evaluator.add_custom_node(
        result=(non_empty_str(ex.model_name) and has_sources(ex.model_name_sources)),
        id="model_name_has_value_and_source",
        desc="Console model name is provided and at least one source URL is present in the answer",
        parent=node,
        critical=True
    )

    # Model name supported by sources
    model_leaf = evaluator.add_leaf(
        id="model_name_with_source",
        desc="Console model name is provided and supported by at least one source URL",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The console model name is '{ex.model_name}'.",
        node=model_leaf,
        sources=ex.model_name_sources,
        additional_instruction="Verify that the provided webpages explicitly mention the console's official model name. Allow minor formatting variants."
    )

    # Manufacturer existence
    evaluator.add_custom_node(
        result=(non_empty_str(ex.manufacturer_name) and has_sources(ex.manufacturer_sources)),
        id="manufacturer_has_value_and_source",
        desc="Manufacturer name is provided and at least one source URL is present in the answer",
        parent=node,
        critical=True
    )

    # Manufacturer supported by sources
    manu_leaf = evaluator.add_leaf(
        id="manufacturer_with_source",
        desc="Manufacturer name is provided and supported by at least one source URL",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The manufacturer of the console is '{ex.manufacturer_name}'.",
        node=manu_leaf,
        sources=ex.manufacturer_sources,
        additional_instruction="Accept reasonable publisher/manufacturer label variants (e.g., brand vs corporate entity) if they refer to the same organization."
    )


async def build_console_requirements_and_specs(evaluator: Evaluator, parent, ex: ConsoleAnalysisExtraction) -> None:
    node = evaluator.add_parallel(
        id="console_requirements_and_specs",
        desc="Console satisfies all stated constraints and required specs are reported with sources",
        parent=parent,
        critical=True
    )

    # Release date within window
    evaluator.add_custom_node(
        result=(non_empty_str(ex.release_date) and has_sources(ex.release_date_sources)),
        id="release_date_value_and_sources_present",
        desc="Release date is stated and includes at least one source URL",
        parent=node,
        critical=True
    )

    release_leaf = evaluator.add_leaf(
        id="release_date_within_window_with_source",
        desc="Release date is stated, has a source URL, and falls between Sep 1, 2024 and Dec 31, 2024 (inclusive)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The console's release date occurred between September 1, 2024 and December 31, 2024 (inclusive).",
        node=release_leaf,
        sources=ex.release_date_sources,
        additional_instruction="If a source indicates a specific date or 'September/October/November/December 2024', treat it as within range. If date is outside 2024 Q4 (inclusive of Sep 1 to Dec 31), mark as not supported."
    )

    # GPU >= 15 TFLOPS
    evaluator.add_custom_node(
        result=(non_empty_str(ex.gpu_tflops) and has_sources(ex.gpu_sources)),
        id="gpu_value_and_sources_present",
        desc="GPU performance is stated and at least one source URL is present",
        parent=node,
        critical=True
    )

    gpu_supported_leaf = evaluator.add_leaf(
        id="gpu_value_supported_by_source",
        desc="GPU performance in TFLOPS is supported by cited sources",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The console's GPU performance is {ex.gpu_tflops}.",
        node=gpu_supported_leaf,
        sources=ex.gpu_sources,
        additional_instruction="Verify the TFLOPS performance as stated. Allow approximate phrasing like ~ or up to."
    )
    # Threshold check (custom)
    gpu_num = parse_tflops(ex.gpu_tflops)
    evaluator.add_custom_node(
        result=(gpu_num is not None and gpu_num >= 15.0),
        id="gpu_meets_threshold_with_source",
        desc="GPU performance in TFLOPS is ≥ 15 (based on the stated value)",
        parent=node,
        critical=True
    )

    # Base storage >= 1TB
    evaluator.add_custom_node(
        result=(non_empty_str(ex.base_storage_capacity) and has_sources(ex.base_storage_sources)),
        id="base_storage_value_and_sources_present",
        desc="Base storage capacity is stated and at least one source URL is present",
        parent=node,
        critical=True
    )

    base_storage_supported_leaf = evaluator.add_leaf(
        id="base_storage_supported_by_source",
        desc="Base storage capacity is supported by cited sources",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The base storage capacity is {ex.base_storage_capacity}.",
        node=base_storage_supported_leaf,
        sources=ex.base_storage_sources,
        additional_instruction="Verify the base storage capacity exactly as stated (e.g., 1 TB, 1024 GB)."
    )
    base_gb = parse_capacity_to_gb(ex.base_storage_capacity)
    evaluator.add_custom_node(
        result=(base_gb is not None and base_gb >= 1000.0),
        id="base_storage_meets_minimum_with_source",
        desc="Base storage capacity is ≥ 1 TB (1000 GB) based on the stated value",
        parent=node,
        critical=True
    )

    # Price < $800
    evaluator.add_custom_node(
        result=(non_empty_str(ex.official_usd_price) and has_sources(ex.price_sources)),
        id="price_value_and_sources_present",
        desc="Official USD retail price is stated and at least one source URL is present",
        parent=node,
        critical=True
    )

    price_supported_leaf = evaluator.add_leaf(
        id="price_supported_by_source",
        desc="Official USD retail price is supported by cited sources",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official USD retail price is {ex.official_usd_price}.",
        node=price_supported_leaf,
        sources=ex.price_sources,
        additional_instruction="Verify the official price in USD. Allow formatting like $699 or 699 USD."
    )
    price_num = parse_price_usd(ex.official_usd_price)
    evaluator.add_custom_node(
        result=(price_num is not None and price_num < 800.0),
        id="price_under_800_with_source",
        desc="Official USD retail price is < $800 (based on the stated value)",
        parent=node,
        critical=True
    )

    # Storage expansion support
    evaluator.add_custom_node(
        result=(non_empty_str(ex.expansion_supported) and has_sources(ex.expansion_supported_sources)),
        id="expansion_supported_value_and_sources_present",
        desc="Expansion support is stated and at least one source URL is present",
        parent=node,
        critical=True
    )

    exp_supported_leaf = evaluator.add_leaf(
        id="storage_expansion_supported_with_source",
        desc="Confirms storage expansion beyond base capacity is supported and provides a source URL",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The console supports storage expansion beyond the base capacity.",
        node=exp_supported_leaf,
        sources=ex.expansion_supported_sources,
        additional_instruction="The source must indicate that additional storage beyond the base internal storage is supported (via SSD, expansion card, etc.)."
    )

    # Expansion interface type
    evaluator.add_custom_node(
        result=(non_empty_str(ex.expansion_interface_type) and has_sources(ex.expansion_interface_sources)),
        id="expansion_interface_value_and_sources_present",
        desc="Expansion interface type is stated and at least one source URL is present",
        parent=node,
        critical=True
    )

    exp_iface_leaf = evaluator.add_leaf(
        id="expansion_interface_type_with_source",
        desc="Storage expansion interface type is stated and supported by a source URL",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The storage expansion interface type is {ex.expansion_interface_type}.",
        node=exp_iface_leaf,
        sources=ex.expansion_interface_sources,
        additional_instruction="Verify the specific interface (e.g., M.2 NVMe SSD, PCIe Gen4 M.2, proprietary expansion card)."
    )

    # Expansion minimum speed requirement (if applicable)
    min_speed_present = non_empty_str(ex.expansion_min_speed_requirement) or (
        ex.expansion_min_speed_requirement is not None and ex.expansion_min_speed_requirement.strip().lower() in {"n/a", "none", "not applicable"}
    )
    evaluator.add_custom_node(
        result=(min_speed_present and has_sources(ex.expansion_min_speed_sources)),
        id="expansion_min_speed_value_and_sources_present",
        desc="Minimum speed requirement is stated if applicable (or explicitly 'N/A'), with at least one source URL",
        parent=node,
        critical=True
    )

    min_speed_leaf = evaluator.add_leaf(
        id="expansion_min_speed_requirement_if_applicable_with_source",
        desc="Minimum speed requirements for expansion storage are stated if applicable, supported by a source URL",
        parent=node,
        critical=True
    )
    if ex.expansion_min_speed_requirement and ex.expansion_min_speed_requirement.strip().lower() in {"n/a", "none", "not applicable"}:
        claim_min_speed = "There is no explicit minimum speed requirement for expansion storage stated by the manufacturer."
    else:
        claim_min_speed = f"The minimum speed requirement for expansion storage is {ex.expansion_min_speed_requirement}."
    await evaluator.verify(
        claim=claim_min_speed,
        node=min_speed_leaf,
        sources=ex.expansion_min_speed_sources,
        additional_instruction="If the manufacturer states no explicit minimum speed requirement, that also counts as supported."
    )

    # Maximum supported expansion capacity
    evaluator.add_custom_node(
        result=(non_empty_str(ex.maximum_expansion_capacity) and has_sources(ex.maximum_expansion_sources)),
        id="maximum_expansion_value_and_sources_present",
        desc="Maximum supported expansion storage capacity is stated and at least one source URL is present",
        parent=node,
        critical=True
    )

    max_exp_leaf = evaluator.add_leaf(
        id="maximum_expansion_capacity_with_source",
        desc="Maximum supported expansion storage capacity is stated and supported by a source URL",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The maximum supported expansion storage capacity is {ex.maximum_expansion_capacity}.",
        node=max_exp_leaf,
        sources=ex.maximum_expansion_sources,
        additional_instruction="Verify the maximum capacity limit as stated by official or authoritative sources."
    )

    # Backward compatibility
    evaluator.add_custom_node(
        result=(non_empty_str(ex.backward_compatibility) and has_sources(ex.backward_compatibility_sources)),
        id="backward_compatibility_value_and_sources_present",
        desc="Backward compatibility is stated and at least one source URL is present",
        parent=node,
        critical=True
    )

    bc_leaf = evaluator.add_leaf(
        id="backward_compatibility_with_source",
        desc="Confirms backward compatibility with previous generation games and provides a supporting source URL",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The console is backward compatible with previous generation games.",
        node=bc_leaf,
        sources=ex.backward_compatibility_sources,
        additional_instruction="The source should explicitly mention backward compatibility with prior-generation titles."
    )

    # RDR2 playable
    evaluator.add_custom_node(
        result=(non_empty_str(ex.rdr2_playable) and has_sources(ex.rdr2_sources)),
        id="rdr2_playable_value_and_sources_present",
        desc="RDR2 playability is stated and at least one source URL is present",
        parent=node,
        critical=True
    )

    rdr2_leaf = evaluator.add_leaf(
        id="rdr2_playable_with_source",
        desc="Confirms the console can play Red Dead Redemption 2 and provides a supporting source URL",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The console can play Red Dead Redemption 2.",
        node=rdr2_leaf,
        sources=ex.rdr2_sources,
        additional_instruction="Verification should confirm either native availability, backward compatibility, or official support for RDR2 on this console."
    )


async def build_publisher_information(evaluator: Evaluator, parent, ex: ConsoleAnalysisExtraction) -> None:
    node = evaluator.add_parallel(
        id="publisher_information",
        desc="Publisher information (GTA series) provided with required details and sources",
        parent=parent,
        critical=True
    )

    # GTA publisher
    evaluator.add_custom_node(
        result=(non_empty_str(ex.gta_publisher_name) and has_sources(ex.gta_publisher_sources)),
        id="gta_publisher_value_and_sources_present",
        desc="GTA publisher name is stated and at least one source URL is present",
        parent=node,
        critical=True
    )
    publisher_leaf = evaluator.add_leaf(
        id="gta_publisher_with_source",
        desc="Name of the publisher of the Grand Theft Auto series is stated and supported by a source URL",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The publisher of the Grand Theft Auto series is {ex.gta_publisher_name}.",
        node=publisher_leaf,
        sources=ex.gta_publisher_sources,
        additional_instruction="Accept if the cited source clearly attributes publishing of the GTA franchise to this publisher."
    )

    # Publisher studio locations (>=3)
    # Check count and presence of city+country for at least three entries
    valid_locations = [
        loc for loc in ex.publisher_studio_locations
        if non_empty_str(loc.city) and non_empty_str(loc.country_or_region)
    ]
    evaluator.add_custom_node(
        result=(len(valid_locations) >= 3),
        id="publisher_studio_locations_3plus_count",
        desc="At least 3 studio locations (city + country/region) are listed in the answer",
        parent=node,
        critical=True
    )

    # Verify locations supported by sources (use first 3)
    first_three = valid_locations[:3]
    # Aggregate sources for the first three (require at least one source overall)
    agg_sources: List[str] = []
    for loc in first_three:
        agg_sources.extend(loc.sources or [])
    evaluator.add_custom_node(
        result=(len(agg_sources) > 0),
        id="publisher_locations_sources_present",
        desc="At least one source URL is provided for the listed studio locations",
        parent=node,
        critical=True
    )

    loc_strs = [f"{loc.city}, {loc.country_or_region}" for loc in first_three]
    locations_claim = "The publisher has studio locations in " + "; ".join(loc_strs) + "."
    loc_leaf = evaluator.add_leaf(
        id="publisher_studio_locations_3plus_with_source",
        desc="Lists ≥3 worldwide studio locations of the publisher (city and country/region) supported by source URLs",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=locations_claim,
        node=loc_leaf,
        sources=agg_sources,
        additional_instruction="The webpages should corroborate that the publisher operates studios in the listed cities/countries (studio names like Rockstar North, etc., are acceptable as long as city and country match)."
    )

    # GTA 6 release date
    evaluator.add_custom_node(
        result=(non_empty_str(ex.gta6_release_date) and has_sources(ex.gta6_date_sources)),
        id="gta6_release_date_value_and_sources_present",
        desc="GTA 6 official release date is stated and at least one source URL is present",
        parent=node,
        critical=True
    )
    gta6_date_leaf = evaluator.add_leaf(
        id="gta6_release_date_with_source",
        desc="Official release date of GTA 6 is stated and supported by a source URL",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official release date of Grand Theft Auto VI (GTA 6) is {ex.gta6_release_date}.",
        node=gta6_date_leaf,
        sources=ex.gta6_date_sources,
        additional_instruction="Prefer official announcements or publisher press releases; reputable gaming press is acceptable if citing the official date."
    )


async def build_storage_calculations(evaluator: Evaluator, parent, ex: ConsoleAnalysisExtraction) -> None:
    node = evaluator.add_parallel(
        id="storage_calculations",
        desc="Required storage calculations and RDR2 base-storage analysis are correct",
        parent=parent,
        critical=True
    )

    # Calculate expected total max storage from base + max expansion
    base_gb = parse_capacity_to_gb(ex.base_storage_capacity)
    max_exp_gb = parse_capacity_to_gb(ex.maximum_expansion_capacity)
    claimed_total_gb = parse_capacity_to_gb(ex.total_max_storage_capacity)

    total_calc_ok = (
        base_gb is not None and
        max_exp_gb is not None and
        claimed_total_gb is not None and
        within_tolerance(base_gb + max_exp_gb, claimed_total_gb, abs_tol=10.0, rel_tol=0.05)
    )

    evaluator.add_custom_node(
        result=total_calc_ok,
        id="total_max_storage_calculated",
        desc="Correctly calculates total maximum storage capacity as (base storage + maximum expansion capacity)",
        parent=node,
        critical=True
    )

    # RDR2 3-copy analysis: 3 x 150GB = 450GB
    stated_needed_gb = parse_capacity_to_gb(ex.stated_total_needed_for_3_rdr2)
    calc_needed_ok = (stated_needed_gb is not None and within_tolerance(stated_needed_gb, 450.0, abs_tol=1.0, rel_tol=0.02))
    base_sufficient_bool = (base_gb is not None and base_gb >= 450.0)
    stated_sufficient = parse_yes_no(ex.base_storage_sufficient_for_3_rdr2)

    rdr2_analysis_ok = (calc_needed_ok and stated_sufficient is not None and stated_sufficient == base_sufficient_bool)

    evaluator.add_custom_node(
        result=rdr2_analysis_ok,
        id="rdr2_three_copy_analysis_correct",
        desc="Correctly computes 3 × 150GB = 450GB and states whether base storage alone is sufficient",
        parent=node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the next-generation console technical analysis task.
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

    # Extract structured data from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extraction(),
        template_class=ConsoleAnalysisExtraction,
        extraction_name="console_analysis_extraction"
    )

    # Build top-level critical task node to aggregate all criteria
    task_node = evaluator.add_parallel(
        id="task_overall",
        desc="Complete next-generation gaming console technical analysis and verification (including citations)",
        parent=root,
        critical=True
    )

    # Build subtrees
    await build_console_identification(evaluator, task_node, extraction)
    await build_console_requirements_and_specs(evaluator, task_node, extraction)
    await build_publisher_information(evaluator, task_node, extraction)
    await build_storage_calculations(evaluator, task_node, extraction)

    # Return final structured summary
    return evaluator.get_summary()