import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cs_prof_tech_setup"
TASK_DESCRIPTION = """
A computer science professor at the University of Colorado Denver is setting up their technology infrastructure for research, teaching, and frequent travel to conferences. They need to select four essential technology items that meet specific professional requirements:

Device Requirements:

1. Primary Laptop:
- Must support computational research tasks requiring at least 16GB RAM
- Must have dedicated GPU with minimum 8GB VRAM for AI/ML work
- Must include a minimum 1TB SSD storage
- Must have a display of at least 15.6 inches
- Must weigh less than 5 pounds for portability

2. Smartphone:
- Must support 5G connectivity for field research in Denver, Colorado
- Must have minimum 8GB RAM for multitasking
- Must include IP68 water/dust resistance rating
- Must have battery capacity of at least 4,500 mAh
- Must support NFC for contactless payments

3. Wireless Earbuds:
- Must feature active noise cancellation (ANC)
- Must provide minimum 8 hours of playback per charge with ANC enabled
- Must include charging case with total battery life of at least 24 hours
- Must support Bluetooth 5.0 or higher

4. Smartwatch:
- Must include 24/7 heart rate monitoring
- Must include blood oxygen (SpO2) monitoring
- Must include sleep tracking functionality
- Must have minimum 24-hour battery life on a single charge
- Must have water resistance rating of at least IP68 or 5ATM

Additional Requirements:
- All devices must support either Apple ecosystem (iOS/macOS) OR Android/Windows ecosystem for seamless integration
- The smartphone must be from a manufacturer providing minimum 1-year warranty
- All wireless devices must be FCC certified for use in the United States

Identify four specific product models (one from each category) that satisfy all stated requirements, and provide reference URLs supporting your selections.
"""

REQUIREMENTS_SUMMARY = {
    "laptop": {
        "ram": ">=16GB",
        "gpu_vram": ">=8GB (dedicated/discrete GPU)",
        "storage": ">=1TB SSD",
        "display": ">=15.6 inches",
        "weight": "<5 lb",
    },
    "smartphone": {
        "5g": "supports 5G",
        "denver": "compatible with Denver, CO 5G networks (US carrier 5G bands)",
        "ram": ">=8GB",
        "battery": ">=4500 mAh",
        "ip68": "IP68 water/dust resistance",
        "nfc": "supports NFC",
        "warranty": ">= 1-year manufacturer warranty",
        "fcc": "FCC certified"
    },
    "earbuds": {
        "anc": "Active Noise Cancellation (ANC)",
        "playback_anc": ">=8 hours per charge with ANC enabled",
        "total_battery": ">=24 hours total with charging case",
        "bluetooth": "Bluetooth 5.0 or higher",
        "fcc": "FCC certified"
    },
    "smartwatch": {
        "hr_247": "24/7 heart rate monitoring",
        "spo2": "blood oxygen (SpO2) monitoring",
        "sleep": "sleep tracking",
        "battery": ">=24 hours battery life",
        "water": ">=IP68 or >=5ATM water resistance",
        "fcc": "FCC certified"
    },
    "ecosystem": {
        "platform": "All devices support Apple (iOS/macOS) OR Android/Windows"
    }
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class LaptopSources(BaseModel):
    general: List[str] = Field(default_factory=list)
    ram: List[str] = Field(default_factory=list)
    gpu: List[str] = Field(default_factory=list)
    storage: List[str] = Field(default_factory=list)
    display: List[str] = Field(default_factory=list)
    weight: List[str] = Field(default_factory=list)
    ecosystem: List[str] = Field(default_factory=list)


class SmartphoneSources(BaseModel):
    general: List[str] = Field(default_factory=list)
    s5g: List[str] = Field(default_factory=list)
    denver: List[str] = Field(default_factory=list)
    ram: List[str] = Field(default_factory=list)
    battery: List[str] = Field(default_factory=list)
    ip68: List[str] = Field(default_factory=list)
    nfc: List[str] = Field(default_factory=list)
    warranty: List[str] = Field(default_factory=list)
    fcc: List[str] = Field(default_factory=list)
    ecosystem: List[str] = Field(default_factory=list)


class EarbudsSources(BaseModel):
    general: List[str] = Field(default_factory=list)
    anc: List[str] = Field(default_factory=list)
    playback: List[str] = Field(default_factory=list)
    total_battery: List[str] = Field(default_factory=list)
    bluetooth: List[str] = Field(default_factory=list)
    fcc: List[str] = Field(default_factory=list)
    ecosystem: List[str] = Field(default_factory=list)


class SmartwatchSources(BaseModel):
    general: List[str] = Field(default_factory=list)
    hr: List[str] = Field(default_factory=list)
    spo2: List[str] = Field(default_factory=list)
    sleep: List[str] = Field(default_factory=list)
    battery: List[str] = Field(default_factory=list)
    water: List[str] = Field(default_factory=list)
    fcc: List[str] = Field(default_factory=list)
    ecosystem: List[str] = Field(default_factory=list)


class LaptopInfo(BaseModel):
    model: Optional[str] = None
    ram: Optional[str] = None
    gpu: Optional[str] = None
    vram: Optional[str] = None
    storage: Optional[str] = None
    display_size: Optional[str] = None
    weight: Optional[str] = None
    os: Optional[str] = None  # e.g., Windows 11, macOS
    ecosystem: Optional[str] = None  # e.g., Windows/macOS
    sources: LaptopSources = Field(default_factory=LaptopSources)


class SmartphoneInfo(BaseModel):
    model: Optional[str] = None
    ram: Optional[str] = None
    battery_capacity: Optional[str] = None
    ip_rating: Optional[str] = None
    nfc: Optional[str] = None
    connectivity_5g: Optional[str] = None
    denver_compatibility: Optional[str] = None
    os: Optional[str] = None
    warranty: Optional[str] = None
    fcc_id: Optional[str] = None
    sources: SmartphoneSources = Field(default_factory=SmartphoneSources)


class EarbudsInfo(BaseModel):
    model: Optional[str] = None
    anc: Optional[str] = None
    playback_hours_with_anc: Optional[str] = None
    total_battery_hours: Optional[str] = None
    bluetooth_version: Optional[str] = None
    ecosystem_compat: Optional[str] = None  # e.g., Works with iOS/Android
    sources: EarbudsSources = Field(default_factory=EarbudsSources)


class SmartwatchInfo(BaseModel):
    model: Optional[str] = None
    hr_247: Optional[str] = None
    spo2: Optional[str] = None
    sleep_tracking: Optional[str] = None
    battery_life_hours: Optional[str] = None
    water_resistance: Optional[str] = None
    os: Optional[str] = None
    ecosystem_compat: Optional[str] = None
    sources: SmartwatchSources = Field(default_factory=SmartwatchSources)


class TechSetupExtraction(BaseModel):
    laptop: Optional[LaptopInfo] = None
    smartphone: Optional[SmartphoneInfo] = None
    earbuds: Optional[EarbudsInfo] = None
    smartwatch: Optional[SmartwatchInfo] = None
    chosen_ecosystem: Optional[str] = None  # "Apple" or "Android/Windows"


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_tech_setup() -> str:
    return """
Extract structured details about the four selected products (one laptop, one smartphone, one pair of wireless earbuds, and one smartwatch) from the answer. Extract EXACTLY what the answer states and the URLs it cites.

Return JSON with this structure (null for any missing field):

{
  "laptop": {
    "model": str|null,
    "ram": str|null,
    "gpu": str|null,
    "vram": str|null,
    "storage": str|null,
    "display_size": str|null,
    "weight": str|null,
    "os": str|null,
    "ecosystem": str|null,
    "sources": {
      "general": [url,...],
      "ram": [url,...],
      "gpu": [url,...],
      "storage": [url,...],
      "display": [url,...],
      "weight": [url,...],
      "ecosystem": [url,...]
    }
  },
  "smartphone": {
    "model": str|null,
    "ram": str|null,
    "battery_capacity": str|null,
    "ip_rating": str|null,
    "nfc": str|null,
    "connectivity_5g": str|null,
    "denver_compatibility": str|null,
    "os": str|null,
    "warranty": str|null,
    "fcc_id": str|null,
    "sources": {
      "general": [url,...],
      "s5g": [url,...],
      "denver": [url,...],
      "ram": [url,...],
      "battery": [url,...],
      "ip68": [url,...],
      "nfc": [url,...],
      "warranty": [url,...],
      "fcc": [url,...],
      "ecosystem": [url,...]
    }
  },
  "earbuds": {
    "model": str|null,
    "anc": str|null,
    "playback_hours_with_anc": str|null,
    "total_battery_hours": str|null,
    "bluetooth_version": str|null,
    "ecosystem_compat": str|null,
    "sources": {
      "general": [url,...],
      "anc": [url,...],
      "playback": [url,...],
      "total_battery": [url,...],
      "bluetooth": [url,...],
      "fcc": [url,...],
      "ecosystem": [url,...]
    }
  },
  "smartwatch": {
    "model": str|null,
    "hr_247": str|null,
    "spo2": str|null,
    "sleep_tracking": str|null,
    "battery_life_hours": str|null,
    "water_resistance": str|null,
    "os": str|null,
    "ecosystem_compat": str|null,
    "sources": {
      "general": [url,...],
      "hr": [url,...],
      "spo2": [url,...],
      "sleep": [url,...],
      "battery": [url,...],
      "water": [url,...],
      "fcc": [url,...],
      "ecosystem": [url,...]
    }
  },
  "chosen_ecosystem": str|null
}

Rules:
- Extract only URLs explicitly present in the answer (plain or markdown links). Do not invent URLs.
- Include up to 3 relevant URLs per spec list if available; otherwise an empty list.
- For any value the answer doesn't state, set it to null.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(v: Optional[List[str]]) -> List[str]:
    return [u.strip() for u in (v or []) if isinstance(u, str) and u.strip()]


def _combine_sources(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for lst in lists:
        for url in _safe_list(lst):
            if url not in seen:
                seen.add(url)
                combined.append(url)
    return combined


def _name_or_placeholder(name: Optional[str], placeholder: str) -> str:
    return name.strip() if name else placeholder


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_laptop(evaluator: Evaluator, parent, laptop: LaptopInfo) -> None:
    # Primary_Laptop_Selection
    node_main = evaluator.add_parallel(
        id="Primary_Laptop_Selection",
        desc="Verify the selected laptop meets all computational and portability requirements",
        parent=parent,
        critical=False
    )

    # Computational_Performance
    node_compute = evaluator.add_parallel(
        id="Laptop_Computational_Performance",
        desc="Laptop meets computational performance requirements for research tasks",
        parent=node_main,
        critical=True
    )

    # RAM_Specification
    node_ram = evaluator.add_parallel(
        id="Laptop_RAM_Specification",
        desc="Verify laptop has minimum 16GB RAM",
        parent=node_compute,
        critical=True
    )
    ram_sources = _combine_sources(laptop.sources.ram, laptop.sources.general)
    evaluator.add_custom_node(
        result=len(ram_sources) > 0,
        id="Laptop_RAM_Reference_URL",
        desc="Reference URL confirming RAM specification",
        parent=node_ram,
        critical=True
    )
    leaf_ram = evaluator.add_leaf(
        id="Laptop_RAM_Value_Check",
        desc="RAM capacity is at least 16GB",
        parent=node_ram,
        critical=True
    )
    ram_claim = f"The laptop '{_name_or_placeholder(laptop.model, 'the selected laptop')}' has {laptop.ram or 'at least 16GB'} RAM and meets or exceeds 16GB."
    await evaluator.verify(
        claim=ram_claim,
        node=leaf_ram,
        sources=ram_sources,
        additional_instruction="Verify memory capacity is at least 16 GB. Accept equivalents like '16 GB DDR5', '32 GB', etc."
    )

    # GPU_Specification
    node_gpu = evaluator.add_parallel(
        id="Laptop_GPU_Specification",
        desc="Verify laptop has dedicated GPU with minimum 8GB VRAM",
        parent=node_compute,
        critical=True
    )
    gpu_sources = _combine_sources(laptop.sources.gpu, laptop.sources.general)
    evaluator.add_custom_node(
        result=len(gpu_sources) > 0,
        id="Laptop_GPU_Reference_URL",
        desc="Reference URL confirming GPU and VRAM specification",
        parent=node_gpu,
        critical=True
    )
    leaf_gpu = evaluator.add_leaf(
        id="Laptop_GPU_VRAM_Check",
        desc="Dedicated GPU has at least 8GB VRAM",
        parent=node_gpu,
        critical=True
    )
    gpu_claim = (
        f"The laptop '{_name_or_placeholder(laptop.model, 'the selected laptop')}' includes a dedicated (discrete) GPU "
        f"{'('+laptop.gpu+')' if laptop.gpu else ''} with at least 8 GB of VRAM {('('+laptop.vram+')' if laptop.vram else '')}."
    )
    await evaluator.verify(
        claim=gpu_claim,
        node=leaf_gpu,
        sources=gpu_sources,
        additional_instruction="Confirm the GPU is discrete/dedicated (not integrated) and has >= 8GB VRAM."
    )

    # Storage_Performance
    node_storage = evaluator.add_parallel(
        id="Laptop_Storage_Performance",
        desc="Laptop meets storage requirements",
        parent=node_main,
        critical=True
    )
    storage_sources = _combine_sources(laptop.sources.storage, laptop.sources.general)
    evaluator.add_custom_node(
        result=len(storage_sources) > 0,
        id="Laptop_Storage_Reference_URL",
        desc="Reference URL confirming storage specification",
        parent=node_storage,
        critical=True
    )
    leaf_storage = evaluator.add_leaf(
        id="Laptop_Storage_Capacity_Check",
        desc="Laptop has minimum 1TB SSD storage",
        parent=node_storage,
        critical=True
    )
    storage_claim = f"The laptop '{_name_or_placeholder(laptop.model, 'the selected laptop')}' provides at least 1 TB SSD storage ({laptop.storage or '>=1 TB SSD'})."
    await evaluator.verify(
        claim=storage_claim,
        node=leaf_storage,
        sources=storage_sources,
        additional_instruction="Check storage is an SSD and capacity >= 1 TB (accept '1 TB', '1000 GB')."
    )

    # Display_Specification
    node_display = evaluator.add_parallel(
        id="Laptop_Display_Specification",
        desc="Laptop meets display size requirements",
        parent=node_main,
        critical=True
    )
    display_sources = _combine_sources(laptop.sources.display, laptop.sources.general)
    evaluator.add_custom_node(
        result=len(display_sources) > 0,
        id="Laptop_Display_Reference_URL",
        desc="Reference URL confirming display specification",
        parent=node_display,
        critical=True
    )
    leaf_display = evaluator.add_leaf(
        id="Laptop_Display_Size_Check",
        desc="Laptop has minimum 15.6-inch display",
        parent=node_display,
        critical=True
    )
    display_claim = f"The laptop '{_name_or_placeholder(laptop.model, 'the selected laptop')}' has a display of at least 15.6 inches ({laptop.display_size or '>=15.6\"'})."
    await evaluator.verify(
        claim=display_claim,
        node=leaf_display,
        sources=display_sources,
        additional_instruction="Confirm screen size >= 15.6 inches; allow larger sizes (e.g., 16\", 17\")."
    )

    # Portability_Requirement
    node_weight = evaluator.add_parallel(
        id="Laptop_Portability_Requirement",
        desc="Laptop meets portability weight requirement",
        parent=node_main,
        critical=True
    )
    weight_sources = _combine_sources(laptop.sources.weight, laptop.sources.general)
    evaluator.add_custom_node(
        result=len(weight_sources) > 0,
        id="Laptop_Weight_Reference_URL",
        desc="Reference URL confirming weight specification",
        parent=node_weight,
        critical=True
    )
    leaf_weight = evaluator.add_leaf(
        id="Laptop_Weight_Check",
        desc="Laptop weighs less than 5 pounds",
        parent=node_weight,
        critical=True
    )
    weight_claim = f"The laptop '{_name_or_placeholder(laptop.model, 'the selected laptop')}' weighs less than 5 pounds ({laptop.weight or '< 5 lb'})."
    await evaluator.verify(
        claim=weight_claim,
        node=leaf_weight,
        sources=weight_sources,
        additional_instruction="Confirm weight < 5.0 lb. If listed in kg, allow unit conversion (5 lb ≈ 2.27 kg)."
    )


async def verify_smartphone(evaluator: Evaluator, parent, phone: SmartphoneInfo) -> None:
    node_main = evaluator.add_parallel(
        id="Smartphone_Selection",
        desc="Verify the selected smartphone meets all connectivity, durability, and performance requirements",
        parent=parent,
        critical=False
    )

    # Network_Connectivity
    node_net = evaluator.add_parallel(
        id="Smartphone_Network_Connectivity",
        desc="Smartphone meets network connectivity requirements",
        parent=node_main,
        critical=True
    )
    # 5G capability
    node_5g = evaluator.add_parallel(
        id="Smartphone_5G_Capability",
        desc="Verify smartphone supports 5G connectivity",
        parent=node_net,
        critical=True
    )
    s5g_sources = _combine_sources(phone.sources.s5g, phone.sources.general)
    evaluator.add_custom_node(
        result=len(s5g_sources) > 0,
        id="Smartphone_5G_Reference_URL",
        desc="Reference URL confirming 5G support",
        parent=node_5g,
        critical=True
    )
    leaf_5g = evaluator.add_leaf(
        id="Smartphone_5G_Support_Check",
        desc="Smartphone has 5G capability",
        parent=node_5g,
        critical=True
    )
    claim_5g = f"The smartphone '{_name_or_placeholder(phone.model, 'the selected smartphone')}' supports 5G connectivity ({phone.connectivity_5g or '5G support'})."
    await evaluator.verify(
        claim=claim_5g,
        node=leaf_5g,
        sources=s5g_sources,
        additional_instruction="Verify the phone supports 5G (sub-6 and/or mmWave)."
    )

    # Denver compatibility
    node_den = evaluator.add_parallel(
        id="Smartphone_Denver_Compatibility",
        desc="Verify smartphone works with Denver 5G networks",
        parent=node_net,
        critical=True
    )
    denver_sources = _combine_sources(phone.sources.denver, phone.sources.s5g, phone.sources.general)
    evaluator.add_custom_node(
        result=len(denver_sources) > 0,
        id="Smartphone_Network_Reference_URL",
        desc="Reference URL confirming network compatibility",
        parent=node_den,
        critical=True
    )
    leaf_den = evaluator.add_leaf(
        id="Smartphone_Denver_5G_Check",
        desc="Smartphone is compatible with 5G networks in Denver, Colorado",
        parent=node_den,
        critical=True
    )
    claim_den = (
        f"The smartphone '{_name_or_placeholder(phone.model, 'the selected smartphone')}' is compatible with 5G networks used by major US carriers "
        f"in Denver, Colorado (e.g., AT&T, Verizon, T‑Mobile), based on supported US 5G bands."
    )
    await evaluator.verify(
        claim=claim_den,
        node=leaf_den,
        sources=denver_sources,
        additional_instruction="Confirm US 5G carrier band support (e.g., n5, n41, n66, n71, n260, n261) or explicit carrier compatibility in Denver."
    )

    # Performance_Specifications
    node_perf = evaluator.add_parallel(
        id="Smartphone_Performance_Specifications",
        desc="Smartphone meets performance requirements",
        parent=node_main,
        critical=True
    )
    # RAM >= 8GB
    node_sram = evaluator.add_parallel(
        id="Smartphone_RAM_Capacity",
        desc="Verify smartphone has sufficient RAM",
        parent=node_perf,
        critical=True
    )
    sram_sources = _combine_sources(phone.sources.ram, phone.sources.general)
    evaluator.add_custom_node(
        result=len(sram_sources) > 0,
        id="Smartphone_RAM_Spec_URL",
        desc="Reference URL confirming RAM specification",
        parent=node_sram,
        critical=True
    )
    leaf_sram = evaluator.add_leaf(
        id="Smartphone_RAM_Amount_Check",
        desc="Smartphone has minimum 8GB RAM",
        parent=node_sram,
        critical=True
    )
    claim_sram = f"The smartphone '{_name_or_placeholder(phone.model, 'the selected smartphone')}' has {phone.ram or '>= 8GB'} RAM, meeting the minimum 8 GB requirement."
    await evaluator.verify(
        claim=claim_sram,
        node=leaf_sram,
        sources=sram_sources,
        additional_instruction="Confirm RAM >= 8 GB."
    )

    # Battery >= 4500 mAh
    node_batt = evaluator.add_parallel(
        id="Smartphone_Battery_Specification",
        desc="Verify smartphone has adequate battery capacity",
        parent=node_perf,
        critical=True
    )
    batt_sources = _combine_sources(phone.sources.battery, phone.sources.general)
    evaluator.add_custom_node(
        result=len(batt_sources) > 0,
        id="Smartphone_Battery_Reference_URL",
        desc="Reference URL confirming battery capacity",
        parent=node_batt,
        critical=True
    )
    leaf_batt = evaluator.add_leaf(
        id="Smartphone_Battery_Capacity_Check",
        desc="Smartphone has battery capacity of at least 4,500 mAh",
        parent=node_batt,
        critical=True
    )
    claim_batt = f"The smartphone '{_name_or_placeholder(phone.model, 'the selected smartphone')}' has a battery capacity of at least 4,500 mAh ({phone.battery_capacity or '≥4500 mAh'})."
    await evaluator.verify(
        claim=claim_batt,
        node=leaf_batt,
        sources=batt_sources,
        additional_instruction="Confirm capacity >= 4500 mAh."
    )

    # Durability_Features (IP68)
    node_dura = evaluator.add_parallel(
        id="Smartphone_Durability_Features",
        desc="Smartphone meets durability requirements",
        parent=node_main,
        critical=True
    )
    ip_sources = _combine_sources(phone.sources.ip68, phone.sources.general)
    evaluator.add_custom_node(
        result=len(ip_sources) > 0,
        id="Smartphone_Durability_Reference_URL",
        desc="Reference URL confirming IP68 rating",
        parent=node_dura,
        critical=True
    )
    leaf_ip = evaluator.add_leaf(
        id="Smartphone_Water_Resistance_Check",
        desc="Smartphone has IP68 water and dust resistance rating",
        parent=node_dura,
        critical=True
    )
    claim_ip = f"The smartphone '{_name_or_placeholder(phone.model, 'the selected smartphone')}' has an IP68 water/dust resistance rating ({phone.ip_rating or 'IP68'})."
    await evaluator.verify(
        claim=claim_ip,
        node=leaf_ip,
        sources=ip_sources,
        additional_instruction="Confirm explicit 'IP68'."
    )

    # Payment_Capability (NFC)
    node_pay = evaluator.add_parallel(
        id="Smartphone_Payment_Capability",
        desc="Smartphone supports contactless payment",
        parent=node_main,
        critical=True
    )
    nfc_sources = _combine_sources(phone.sources.nfc, phone.sources.general)
    evaluator.add_custom_node(
        result=len(nfc_sources) > 0,
        id="Smartphone_NFC_Reference_URL",
        desc="Reference URL confirming NFC support",
        parent=node_pay,
        critical=True
    )
    leaf_nfc = evaluator.add_leaf(
        id="Smartphone_NFC_Feature_Check",
        desc="Smartphone supports NFC for contactless payments",
        parent=node_pay,
        critical=True
    )
    claim_nfc = f"The smartphone '{_name_or_placeholder(phone.model, 'the selected smartphone')}' supports NFC for contactless payments."
    await evaluator.verify(
        claim=claim_nfc,
        node=leaf_nfc,
        sources=nfc_sources,
        additional_instruction="Confirm NFC support suitable for mobile payments."
    )

    # Compliance_Requirements (warranty + FCC)
    node_comp = evaluator.add_parallel(
        id="Smartphone_Compliance_Requirements",
        desc="Smartphone meets regulatory and warranty requirements",
        parent=node_main,
        critical=True
    )
    # Warranty
    node_war = evaluator.add_parallel(
        id="Smartphone_Warranty_Coverage",
        desc="Verify manufacturer warranty coverage",
        parent=node_comp,
        critical=True
    )
    war_sources = _combine_sources(phone.sources.warranty, phone.sources.general)
    evaluator.add_custom_node(
        result=len(war_sources) > 0,
        id="Smartphone_Warranty_Reference_URL",
        desc="Reference URL confirming warranty terms",
        parent=node_war,
        critical=True
    )
    leaf_war = evaluator.add_leaf(
        id="Smartphone_Warranty_Duration_Check",
        desc="Manufacturer provides minimum 1-year warranty",
        parent=node_war,
        critical=True
    )
    claim_war = f"The manufacturer of '{_name_or_placeholder(phone.model, 'the selected smartphone')}' provides a minimum 1-year warranty ({phone.warranty or '≥1 year'})."
    await evaluator.verify(
        claim=claim_war,
        node=leaf_war,
        sources=war_sources,
        additional_instruction="Confirm at least one (1) year limited warranty from the manufacturer."
    )
    # FCC
    node_fcc = evaluator.add_parallel(
        id="Smartphone_FCC_Certification",
        desc="Verify FCC certification for US use",
        parent=node_comp,
        critical=True
    )
    fcc_sources = _combine_sources(phone.sources.fcc, phone.sources.general)
    evaluator.add_custom_node(
        result=len(fcc_sources) > 0,
        id="Smartphone_FCC_Reference_URL",
        desc="Reference URL confirming FCC certification",
        parent=node_fcc,
        critical=True
    )
    leaf_fcc = evaluator.add_leaf(
        id="Smartphone_FCC_Status_Check",
        desc="Smartphone is FCC certified for use in United States",
        parent=node_fcc,
        critical=True
    )
    claim_fcc = f"The smartphone '{_name_or_placeholder(phone.model, 'the selected smartphone')}' is FCC certified for use in the United States {('with FCC ID '+phone.fcc_id) if phone.fcc_id else ''}."
    await evaluator.verify(
        claim=claim_fcc,
        node=leaf_fcc,
        sources=fcc_sources,
        additional_instruction="Prefer an official FCC ID database page or an official compliance statement explicitly indicating FCC certification."
    )


async def verify_earbuds(evaluator: Evaluator, parent, buds: EarbudsInfo) -> None:
    node_main = evaluator.add_parallel(
        id="Wireless_Earbuds_Selection",
        desc="Verify the selected wireless earbuds meet all audio quality and battery requirements",
        parent=parent,
        critical=False
    )

    # Audio_Features (ANC)
    node_audio = evaluator.add_parallel(
        id="Earbuds_Audio_Features",
        desc="Earbuds meet audio quality requirements",
        parent=node_main,
        critical=True
    )
    anc_sources = _combine_sources(buds.sources.anc, buds.sources.general)
    evaluator.add_custom_node(
        result=len(anc_sources) > 0,
        id="Earbuds_ANC_Reference_URL",
        desc="Reference URL confirming ANC feature",
        parent=node_audio,
        critical=True
    )
    leaf_anc = evaluator.add_leaf(
        id="Earbuds_ANC_Capability",
        desc="Earbuds feature active noise cancellation",
        parent=node_audio,
        critical=True
    )
    claim_anc = f"The earbuds '{_name_or_placeholder(buds.model, 'the selected earbuds')}' provide Active Noise Cancellation (ANC)."
    await evaluator.verify(
        claim=claim_anc,
        node=leaf_anc,
        sources=anc_sources,
        additional_instruction="Confirm explicit ANC support (Active Noise Cancellation)."
    )

    # Battery_Performance
    node_batt = evaluator.add_parallel(
        id="Earbuds_Battery_Performance",
        desc="Earbuds meet battery life requirements",
        parent=node_main,
        critical=True
    )
    # Single-charge playback with ANC >= 8 h
    node_single = evaluator.add_parallel(
        id="Earbuds_Single_Charge_Battery",
        desc="Verify earbuds provide adequate playback time",
        parent=node_batt,
        critical=True
    )
    single_sources = _combine_sources(buds.sources.playback, buds.sources.general)
    evaluator.add_custom_node(
        result=len(single_sources) > 0,
        id="Earbuds_Playback_Reference_URL",
        desc="Reference URL confirming playback battery life",
        parent=node_single,
        critical=True
    )
    leaf_play = evaluator.add_leaf(
        id="Earbuds_Playback_Duration_Check",
        desc="Earbuds provide minimum 8 hours playback per charge with ANC enabled",
        parent=node_single,
        critical=True
    )
    claim_play = (
        f"The earbuds '{_name_or_placeholder(buds.model, 'the selected earbuds')}' provide at least 8 hours of playback per charge "
        f"with ANC enabled ({buds.playback_hours_with_anc or '≥8 hours with ANC'})."
    )
    await evaluator.verify(
        claim=claim_play,
        node=leaf_play,
        sources=single_sources,
        additional_instruction="The 8-hour claim must explicitly correspond to playback with ANC turned on."
    )

    # Total battery with case >= 24 h
    node_total = evaluator.add_parallel(
        id="Earbuds_Total_Battery_Life",
        desc="Verify total battery life with charging case",
        parent=node_batt,
        critical=True
    )
    total_sources = _combine_sources(buds.sources.total_battery, buds.sources.general)
    evaluator.add_custom_node(
        result=len(total_sources) > 0,
        id="Earbuds_Total_Battery_Reference_URL",
        desc="Reference URL confirming total battery life",
        parent=node_total,
        critical=True
    )
    leaf_total = evaluator.add_leaf(
        id="Earbuds_Total_Duration_Check",
        desc="Charging case provides total battery life of at least 24 hours",
        parent=node_total,
        critical=True
    )
    claim_total = f"The earbuds '{_name_or_placeholder(buds.model, 'the selected earbuds')}' offer at least 24 hours of total listening time with the charging case ({buds.total_battery_hours or '≥24 hours total'})."
    await evaluator.verify(
        claim=claim_total,
        node=leaf_total,
        sources=total_sources,
        additional_instruction="Confirm total battery life (buds + case) >= 24 hours."
    )

    # Connectivity_Standard (Bluetooth 5.0+)
    node_bt = evaluator.add_parallel(
        id="Earbuds_Connectivity_Standard",
        desc="Earbuds meet wireless connectivity requirements",
        parent=node_main,
        critical=True
    )
    bt_sources = _combine_sources(buds.sources.bluetooth, buds.sources.general)
    evaluator.add_custom_node(
        result=len(bt_sources) > 0,
        id="Earbuds_Bluetooth_Reference_URL",
        desc="Reference URL confirming Bluetooth version",
        parent=node_bt,
        critical=True
    )
    leaf_bt = evaluator.add_leaf(
        id="Earbuds_Bluetooth_Version_Check",
        desc="Earbuds support Bluetooth 5.0 or higher",
        parent=node_bt,
        critical=True
    )
    claim_bt = f"The earbuds '{_name_or_placeholder(buds.model, 'the selected earbuds')}' support Bluetooth 5.0 or higher ({buds.bluetooth_version or '>=5.0'})."
    await evaluator.verify(
        claim=claim_bt,
        node=leaf_bt,
        sources=bt_sources,
        additional_instruction="Confirm Bluetooth version is 5.0, 5.1, 5.2, 5.3, etc."
    )

    # FCC_Compliance
    node_fcc = evaluator.add_parallel(
        id="Earbuds_FCC_Compliance",
        desc="Earbuds meet FCC certification requirements",
        parent=node_main,
        critical=True
    )
    fcc_sources = _combine_sources(buds.sources.fcc, buds.sources.general)
    evaluator.add_custom_node(
        result=len(fcc_sources) > 0,
        id="Earbuds_FCC_Reference_URL",
        desc="Reference URL confirming FCC certification",
        parent=node_fcc,
        critical=True
    )
    leaf_fcc = evaluator.add_leaf(
        id="Earbuds_FCC_Certification_Check",
        desc="Earbuds are FCC certified for use in United States",
        parent=node_fcc,
        critical=True
    )
    claim_fcc = f"The earbuds '{_name_or_placeholder(buds.model, 'the selected earbuds')}' are FCC certified for use in the United States."
    await evaluator.verify(
        claim=claim_fcc,
        node=leaf_fcc,
        sources=fcc_sources,
        additional_instruction="Prefer official FCC ID database or manufacturer compliance statements explicitly indicating FCC certification."
    )


async def verify_smartwatch(evaluator: Evaluator, parent, watch: SmartwatchInfo) -> None:
    node_main = evaluator.add_parallel(
        id="Smartwatch_Selection",
        desc="Verify the selected smartwatch meets all health monitoring and durability requirements",
        parent=parent,
        critical=False
    )

    # Health_Monitoring_Features
    node_health = evaluator.add_parallel(
        id="Smartwatch_Health_Monitoring_Features",
        desc="Smartwatch meets health monitoring requirements",
        parent=node_main,
        critical=True
    )

    # Heart Rate
    node_hr = evaluator.add_parallel(
        id="Smartwatch_Heart_Rate_Feature",
        desc="Verify 24/7 heart rate monitoring capability",
        parent=node_health,
        critical=True
    )
    hr_sources = _combine_sources(watch.sources.hr, watch.sources.general)
    evaluator.add_custom_node(
        result=len(hr_sources) > 0,
        id="Smartwatch_Heart_Rate_Reference_URL",
        desc="Reference URL confirming heart rate monitoring",
        parent=node_hr,
        critical=True
    )
    leaf_hr = evaluator.add_leaf(
        id="Smartwatch_Heart_Rate_Check",
        desc="Smartwatch includes 24/7 heart rate monitoring",
        parent=node_hr,
        critical=True
    )
    claim_hr = f"The smartwatch '{_name_or_placeholder(watch.model, 'the selected smartwatch')}' provides 24/7 heart rate monitoring."
    await evaluator.verify(
        claim=claim_hr,
        node=leaf_hr,
        sources=hr_sources,
        additional_instruction="Confirm continuous (24/7) heart rate monitoring capability."
    )

    # SpO2
    node_spo2 = evaluator.add_parallel(
        id="Smartwatch_Blood_Oxygen_Feature",
        desc="Verify blood oxygen monitoring capability",
        parent=node_health,
        critical=True
    )
    spo2_sources = _combine_sources(watch.sources.spo2, watch.sources.general)
    evaluator.add_custom_node(
        result=len(spo2_sources) > 0,
        id="Smartwatch_SpO2_Reference_URL",
        desc="Reference URL confirming SpO2 monitoring",
        parent=node_spo2,
        critical=True
    )
    leaf_spo2 = evaluator.add_leaf(
        id="Smartwatch_SpO2_Check",
        desc="Smartwatch includes blood oxygen (SpO2) monitoring",
        parent=node_spo2,
        critical=True
    )
    claim_spo2 = f"The smartwatch '{_name_or_placeholder(watch.model, 'the selected smartwatch')}' measures blood oxygen (SpO2)."
    await evaluator.verify(
        claim=claim_spo2,
        node=leaf_spo2,
        sources=spo2_sources,
        additional_instruction="Confirm SpO2 measurement capability."
    )

    # Sleep Tracking
    node_sleep = evaluator.add_parallel(
        id="Smartwatch_Sleep_Tracking_Feature",
        desc="Verify sleep tracking capability",
        parent=node_health,
        critical=True
    )
    sleep_sources = _combine_sources(watch.sources.sleep, watch.sources.general)
    evaluator.add_custom_node(
        result=len(sleep_sources) > 0,
        id="Smartwatch_Sleep_Reference_URL",
        desc="Reference URL confirming sleep tracking",
        parent=node_sleep,
        critical=True
    )
    leaf_sleep = evaluator.add_leaf(
        id="Smartwatch_Sleep_Tracking_Check",
        desc="Smartwatch includes sleep tracking functionality",
        parent=node_sleep,
        critical=True
    )
    claim_sleep = f"The smartwatch '{_name_or_placeholder(watch.model, 'the selected smartwatch')}' includes sleep tracking."
    await evaluator.verify(
        claim=claim_sleep,
        node=leaf_sleep,
        sources=sleep_sources,
        additional_instruction="Confirm that the watch supports sleep tracking."
    )

    # Battery_Endurance
    node_batt = evaluator.add_parallel(
        id="Smartwatch_Battery_Endurance",
        desc="Smartwatch meets battery life requirements",
        parent=node_main,
        critical=True
    )
    batt_sources = _combine_sources(watch.sources.battery, watch.sources.general)
    evaluator.add_custom_node(
        result=len(batt_sources) > 0,
        id="Smartwatch_Battery_Life_Reference_URL",
        desc="Reference URL confirming battery life",
        parent=node_batt,
        critical=True
    )
    leaf_batt = evaluator.add_leaf(
        id="Smartwatch_Battery_Duration_Check",
        desc="Smartwatch has minimum 24-hour battery life on single charge",
        parent=node_batt,
        critical=True
    )
    claim_batt = f"The smartwatch '{_name_or_placeholder(watch.model, 'the selected smartwatch')}' has at least 24 hours of battery life on a single charge ({watch.battery_life_hours or '≥24 hours'})."
    await evaluator.verify(
        claim=claim_batt,
        node=leaf_batt,
        sources=batt_sources,
        additional_instruction="Confirm battery life is at least 24 hours."
    )

    # Water_Resistance
    node_water = evaluator.add_parallel(
        id="Smartwatch_Water_Resistance",
        desc="Smartwatch meets water resistance requirements",
        parent=node_main,
        critical=True
    )
    water_sources = _combine_sources(watch.sources.water, watch.sources.general)
    evaluator.add_custom_node(
        result=len(water_sources) > 0,
        id="Smartwatch_Water_Resistance_Reference_URL",
        desc="Reference URL confirming water resistance rating",
        parent=node_water,
        critical=True
    )
    leaf_water = evaluator.add_leaf(
        id="Smartwatch_Water_Rating_Check",
        desc="Smartwatch has water resistance rating of at least IP68 or 5ATM",
        parent=node_water,
        critical=True
    )
    claim_water = (
        f"The smartwatch '{_name_or_placeholder(watch.model, 'the selected smartwatch')}' has at least IP68 or 5ATM water resistance "
        f"({watch.water_resistance or '≥IP68 or ≥5ATM'})."
    )
    await evaluator.verify(
        claim=claim_water,
        node=leaf_water,
        sources=water_sources,
        additional_instruction="Confirm rating is IP68 or 5ATM (or higher like 10ATM)."
    )

    # FCC_Compliance
    node_fcc = evaluator.add_parallel(
        id="Smartwatch_FCC_Compliance",
        desc="Smartwatch meets FCC certification requirements",
        parent=node_main,
        critical=True
    )
    fcc_sources = _combine_sources(watch.sources.fcc, watch.sources.general)
    evaluator.add_custom_node(
        result=len(fcc_sources) > 0,
        id="Smartwatch_FCC_Reference_URL",
        desc="Reference URL confirming FCC certification",
        parent=node_fcc,
        critical=True
    )
    leaf_fcc = evaluator.add_leaf(
        id="Smartwatch_FCC_Certification_Check",
        desc="Smartwatch is FCC certified for use in United States",
        parent=node_fcc,
        critical=True
    )
    claim_fcc = f"The smartwatch '{_name_or_placeholder(watch.model, 'the selected smartwatch')}' is FCC certified for use in the United States."
    await evaluator.verify(
        claim=claim_fcc,
        node=leaf_fcc,
        sources=fcc_sources,
        additional_instruction="Prefer official FCC ID database or explicit manufacturer FCC compliance statement."
    )


async def verify_ecosystem(
    evaluator: Evaluator,
    parent,
    chosen_ecosystem: Optional[str],
    laptop: LaptopInfo,
    phone: SmartphoneInfo,
    buds: EarbudsInfo,
    watch: SmartwatchInfo
) -> None:
    # Ecosystem_Integration (sequential critical)
    node_main = evaluator.add_sequential(
        id="Ecosystem_Integration",
        desc="Verify all devices support a unified ecosystem for seamless integration",
        parent=parent,
        critical=True
    )

    # Ecosystem_Compatibility_Check (parallel group)
    node_compat = evaluator.add_parallel(
        id="Ecosystem_Compatibility_Check",
        desc="Verify all devices support the same ecosystem",
        parent=node_main,
        critical=True
    )

    eco_sources_union = _combine_sources(
        laptop.sources.ecosystem, laptop.sources.general,
        phone.sources.ecosystem, phone.sources.general,
        buds.sources.ecosystem, buds.sources.general,
        watch.sources.ecosystem, watch.sources.general,
    )

    # Ecosystem reference URLs existence for all devices
    has_ecosystem_refs = all([
        len(_safe_list(laptop.sources.ecosystem)) > 0,
        len(_safe_list(phone.sources.ecosystem)) > 0,
        len(_safe_list(buds.sources.ecosystem)) > 0,
        len(_safe_list(watch.sources.ecosystem)) > 0,
    ])
    evaluator.add_custom_node(
        result=has_ecosystem_refs,
        id="Ecosystem_Reference_URL",
        desc="Reference URLs confirming ecosystem compatibility for all devices",
        parent=node_compat,
        critical=True
    )

    # Unified platform verification
    leaf_unified = evaluator.add_leaf(
        id="Unified_Platform",
        desc="All devices support either Apple ecosystem (iOS/macOS) OR Android/Windows ecosystem",
        parent=node_compat,
        critical=True
    )
    eco = (chosen_ecosystem or "").strip() or "the chosen ecosystem"
    claim_unified = (
        f"All four devices support {eco}: "
        f"laptop '{_name_or_placeholder(laptop.model, 'laptop')}' runs {laptop.os or laptop.ecosystem or 'the chosen platform'}, "
        f"smartphone '{_name_or_placeholder(phone.model, 'smartphone')}' runs {phone.os or 'the chosen platform'}, "
        f"earbuds '{_name_or_placeholder(buds.model, 'earbuds')}' are compatible with {buds.ecosystem_compat or eco}, "
        f"and smartwatch '{_name_or_placeholder(watch.model, 'smartwatch')}' supports {watch.os or watch.ecosystem_compat or eco}."
    )
    await evaluator.verify(
        claim=claim_unified,
        node=leaf_unified,
        sources=eco_sources_union,
        additional_instruction="Confirm each device supports the same ecosystem (Apple: iOS/macOS; or Android/Windows). For earbuds/smartwatch, compatibility with the ecosystem (app support or 'works with') is sufficient."
    )

    # Cross_Device_Functionality (depends on prior, sequential will skip if fail)
    node_cross = evaluator.add_parallel(
        id="Cross_Device_Functionality",
        desc="Devices can integrate and sync within chosen ecosystem",
        parent=node_main,
        critical=True
    )
    leaf_integration = evaluator.add_leaf(
        id="Integration_Capability",
        desc="Devices support cross-device data syncing and integration",
        parent=node_cross,
        critical=True
    )
    claim_integration = (
        f"Within {eco}, these devices support cross‑device syncing/integration (e.g., notifications, health/fitness data, audio pairing, file sharing, or continuity features)."
    )
    await evaluator.verify(
        claim=claim_integration,
        node=leaf_integration,
        sources=eco_sources_union,
        additional_instruction="Look for official statements/features that indicate cross-device integration within the chosen ecosystem (e.g., Apple Health, iCloud/Continuity, Fast Pair/Phone Link/Google Fit/Windows)."
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
    Evaluate an answer for the complete technology setup task and return a structured result dictionary.
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
        default_model=model,
    )

    # Root container for rubric tree (make non-critical to avoid framework constraint that critical parents must have all critical children)
    tree_root = evaluator.add_parallel(
        id="Complete_Technology_Setup",
        desc="Evaluate whether the complete four-device technology setup meets all specified requirements",
        parent=root,
        critical=False  # Adjusted to satisfy framework's critical-child constraint
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_tech_setup(),
        template_class=TechSetupExtraction,
        extraction_name="tech_setup_extraction"
    )

    # Add requirements (ground truth expectations) for transparency
    evaluator.add_ground_truth({
        "requirements": REQUIREMENTS_SUMMARY
    }, gt_type="requirements_summary")

    # Build device verification subtrees
    laptop = extracted.laptop or LaptopInfo()
    smartphone = extracted.smartphone or SmartphoneInfo()
    earbuds = extracted.earbuds or EarbudsInfo()
    smartwatch = extracted.smartwatch or SmartwatchInfo()

    await verify_laptop(evaluator, tree_root, laptop)
    await verify_smartphone(evaluator, tree_root, smartphone)
    await verify_earbuds(evaluator, tree_root, earbuds)
    await verify_smartwatch(evaluator, tree_root, smartwatch)
    await verify_ecosystem(evaluator, tree_root, extracted.chosen_ecosystem, laptop, smartphone, earbuds, smartwatch)

    return evaluator.get_summary()