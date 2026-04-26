import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tx_level1_trauma_centers"
TASK_DESCRIPTION = (
    "Identify three distinct ACS-verified Level I trauma centers currently operating in Texas, each meeting core "
    "Level I requirements and its assigned specialized capability set, and provide required documentation and URLs."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CapabilityEvidence(BaseModel):
    detail: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ACSVerification(CapabilityEvidence):
    level: Optional[str] = None  # e.g., "Level I (Adult)", "Level I (Pediatric)", etc.


class OperatingStatus(CapabilityEvidence):
    status: Optional[str] = None  # e.g., "currently operating", "active", etc.


class VolumeRequirement(CapabilityEvidence):
    metric: Optional[str] = None  # e.g., "≥1,200 cases/year" or "≥240 ISS>15"
    value: Optional[str] = None   # free-form string if provided


class PediatricVolume(CapabilityEvidence):
    annual_under15_volume: Optional[str] = None  # leave as string (e.g., "≥200", "at least 200")
    level: Optional[str] = None  # e.g., "Level I pediatric trauma services"


class Rotations(CapabilityEvidence):
    specialties: List[str] = Field(default_factory=list)  # e.g., ["general surgery", "orthopedic", "neurosurgery", "emergency medicine"]


class RegistryStaffing(CapabilityEvidence):
    fte_per_entries: Optional[str] = None       # e.g., "0.5 FTE per 200-300 entries"
    annual_entries: Optional[str] = None        # e.g., "600 entries/year"
    fte_value: Optional[str] = None             # e.g., "1.0 FTE"


class PIStaffing(CapabilityEvidence):
    fte_value: Optional[str] = None             # e.g., "0.5 FTE" or "1.0 FTE"
    annual_volume: Optional[str] = None         # e.g., ">500", ">1000"


class CoreRequirements(BaseModel):
    general_surgery_24hr: CapabilityEvidence = Field(default_factory=CapabilityEvidence)
    icu_director_critical_care: CapabilityEvidence = Field(default_factory=CapabilityEvidence)
    anesthesia_15min: CapabilityEvidence = Field(default_factory=CapabilityEvidence)
    neurosurgery_30min: CapabilityEvidence = Field(default_factory=CapabilityEvidence)
    radiology_30min: CapabilityEvidence = Field(default_factory=CapabilityEvidence)
    volume_requirement: VolumeRequirement = Field(default_factory=VolumeRequirement)
    quality_assessment: CapabilityEvidence = Field(default_factory=CapabilityEvidence)
    substance_abuse: CapabilityEvidence = Field(default_factory=CapabilityEvidence)


class CenterSpecialized1(BaseModel):
    # Center 1: pediatric volume + research + rotations
    peds_volume: PediatricVolume = Field(default_factory=PediatricVolume)
    research_program: CapabilityEvidence = Field(default_factory=CapabilityEvidence)
    rotations: Rotations = Field(default_factory=Rotations)


class CenterSpecialized2(BaseModel):
    # Center 2: burn verification + dedicated ortho OR + CPB or transfer
    burn_center_verified: CapabilityEvidence = Field(default_factory=CapabilityEvidence)
    dedicated_ortho_or: CapabilityEvidence = Field(default_factory=CapabilityEvidence)
    cpb_or_transfer: CapabilityEvidence = Field(default_factory=CapabilityEvidence)


class CenterSpecialized3(BaseModel):
    # Center 3: neurotrauma + registry staffing + PI staffing
    neurotrauma_tbi: CapabilityEvidence = Field(default_factory=CapabilityEvidence)
    neurosurgeons: CapabilityEvidence = Field(default_factory=CapabilityEvidence)
    registry_staffing: RegistryStaffing = Field(default_factory=RegistryStaffing)
    pi_staffing: PIStaffing = Field(default_factory=PIStaffing)


class CenterInfo(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    operating: OperatingStatus = Field(default_factory=OperatingStatus)
    acs_verification: ACSVerification = Field(default_factory=ACSVerification)
    core: CoreRequirements = Field(default_factory=CoreRequirements)
    specialized1: CenterSpecialized1 = Field(default_factory=CenterSpecialized1)
    specialized2: CenterSpecialized2 = Field(default_factory=CenterSpecialized2)
    specialized3: CenterSpecialized3 = Field(default_factory=CenterSpecialized3)


class TraumaCentersExtraction(BaseModel):
    centers: List[CenterInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_centers() -> str:
    return """
    Extract up to three Texas trauma centers from the answer that the author intends to use to satisfy the task.
    Map them into Center 1, Center 2, and Center 3, corresponding to the specialized capability sets below.
    IMPORTANT: Do not invent or infer facts or URLs not explicitly present in the answer. If something is missing, set it to null or an empty list as appropriate.

    Center mapping and required specialized fields:
    - Center 1 specializes in pediatric + research + resident rotations:
      • specialized1.peds_volume: Provide details and URLs; include annual_under15_volume if mentioned (e.g., "≥200 under age 15 per year") and level if specified (e.g., "Level I pediatric").
      • specialized1.research_program: Provide details and URLs to peer-reviewed publications or statements about an organized research program.
      • specialized1.rotations: Provide details and URLs; list specialties in rotations.specialties, expecting general surgery, orthopedic surgery, neurosurgery, and emergency medicine if claimed.

    - Center 2 specializes in burn + dedicated orthopaedic trauma OR + CPB or transfer:
      • specialized2.burn_center_verified: Provide details and URLs; note verifying authority (e.g., American Burn Association).
      • specialized2.dedicated_ortho_or: Provide details and URLs.
      • specialized2.cpb_or_transfer: Provide details and URLs (equipment on-site or a documented transfer agreement).

    - Center 3 specializes in neurotrauma TBI + registry staffing + PI staffing:
      • specialized3.neurotrauma_tbi: Provide details and URLs; include GCS threshold if stated.
      • specialized3.neurosurgeons: Provide details and URLs about board-certified or board-eligible neurosurgeons.
      • specialized3.registry_staffing: Provide details and URLs; include fte_per_entries, annual_entries, and/or fte_value if provided.
      • specialized3.pi_staffing: Provide details and URLs; include fte_value and annual_volume if provided.

    For each center, also extract:
    - name: Hospital name (string)
    - city: City in Texas (string as written in the answer)
    - operating: Current operating status evidence (detail and urls)
    - acs_verification: ACS Level I verification evidence (detail, level if given, and urls)
    - core: Evidence for all core Level I requirements with URLs when available:
      • general_surgery_24hr: 24-hour in-house general surgeon coverage
      • icu_director_critical_care: ICU surgical director is board-certified or board-eligible in surgical critical care
      • anesthesia_15min: Anesthesia services available within 15 minutes of request
      • neurosurgery_30min: Neurosurgery evaluation available within 30 minutes for specified injuries
      • radiology_30min: Radiologist available within 30 minutes to access patient images
      • volume_requirement: Meets ≥1,200 trauma cases annually OR ≥240 patients with ISS > 15 (capture any stated metric/value)
      • quality_assessment: Comprehensive quality assessment program
      • substance_abuse: Substance abuse screening and intervention program

    Rules:
    - Extract URLs exactly as shown in the answer (including markdown links). If a URL is missing, leave the urls list empty.
    - Prefer strings for numeric values (e.g., "≥200") rather than converting to numbers.
    - Return up to 3 centers, in the order the answer implies for the 3 specialized sets. If the answer includes more than three, only include the first three that match the specialized sets. If fewer are present, return as many as available.

    Output JSON structure:
    {
      "centers": [
        {
          "name": str | null,
          "city": str | null,
          "operating": {"status": str | null, "detail": str | null, "urls": [str, ...]},
          "acs_verification": {"level": str | null, "detail": str | null, "urls": [str, ...]},
          "core": {
            "general_surgery_24hr": {"detail": str | null, "urls": [str, ...]},
            "icu_director_critical_care": {"detail": str | null, "urls": [str, ...]},
            "anesthesia_15min": {"detail": str | null, "urls": [str, ...]},
            "neurosurgery_30min": {"detail": str | null, "urls": [str, ...]},
            "radiology_30min": {"detail": str | null, "urls": [str, ...]},
            "volume_requirement": {"metric": str | null, "value": str | null, "detail": str | null, "urls": [str, ...]},
            "quality_assessment": {"detail": str | null, "urls": [str, ...]},
            "substance_abuse": {"detail": str | null, "urls": [str, ...]}
          },
          "specialized1": {
            "peds_volume": {"annual_under15_volume": str | null, "level": str | null, "detail": str | null, "urls": [str, ...]},
            "research_program": {"detail": str | null, "urls": [str, ...]},
            "rotations": {"specialties": [str, ...], "detail": str | null, "urls": [str, ...]}
          },
          "specialized2": {
            "burn_center_verified": {"detail": str | null, "urls": [str, ...]},
            "dedicated_ortho_or": {"detail": str | null, "urls": [str, ...]},
            "cpb_or_transfer": {"detail": str | null, "urls": [str, ...]}
          },
          "specialized3": {
            "neurotrauma_tbi": {"detail": str | null, "urls": [str, ...]},
            "neurosurgeons": {"detail": str | null, "urls": [str, ...]},
            "registry_staffing": {"fte_per_entries": str | null, "annual_entries": str | null, "fte_value": str | null, "detail": str | null, "urls": [str, ...]},
            "pi_staffing": {"fte_value": str | null, "annual_volume": str | null, "detail": str | null, "urls": [str, ...]}
          }
        }
      ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Filter out obvious non-urls if extractor included, but keep simple
    return [u for u in urls if isinstance(u, str) and len(u.strip()) > 0]


def _all_core_staffing_timing_urls(center: CenterInfo) -> List[str]:
    urls: List[str] = []
    urls += _safe_urls(center.core.general_surgery_24hr.urls)
    urls += _safe_urls(center.core.anesthesia_15min.urls)
    urls += _safe_urls(center.core.neurosurgery_30min.urls)
    urls += _safe_urls(center.core.radiology_30min.urls)
    return urls


def _all_core_urls(center: CenterInfo) -> List[str]:
    urls: List[str] = []
    urls += _safe_urls(center.core.general_surgery_24hr.urls)
    urls += _safe_urls(center.core.icu_director_critical_care.urls)
    urls += _safe_urls(center.core.anesthesia_15min.urls)
    urls += _safe_urls(center.core.neurosurgery_30min.urls)
    urls += _safe_urls(center.core.radiology_30min.urls)
    urls += _safe_urls(center.core.volume_requirement.urls)
    urls += _safe_urls(center.core.quality_assessment.urls)
    urls += _safe_urls(center.core.substance_abuse.urls)
    return urls


def _specialized_urls_for_index(center: CenterInfo, idx1_based: int) -> List[str]:
    urls: List[str] = []
    if idx1_based == 1:
        urls += _safe_urls(center.specialized1.peds_volume.urls)
        urls += _safe_urls(center.specialized1.research_program.urls)
        urls += _safe_urls(center.specialized1.rotations.urls)
    elif idx1_based == 2:
        urls += _safe_urls(center.specialized2.burn_center_verified.urls)
        urls += _safe_urls(center.specialized2.dedicated_ortho_or.urls)
        urls += _safe_urls(center.specialized2.cpb_or_transfer.urls)
    elif idx1_based == 3:
        urls += _safe_urls(center.specialized3.neurotrauma_tbi.urls)
        urls += _safe_urls(center.specialized3.neurosurgeons.urls)
        urls += _safe_urls(center.specialized3.registry_staffing.urls)
        urls += _safe_urls(center.specialized3.pi_staffing.urls)
    return urls


def _all_center_urls(center: CenterInfo, idx1_based: int) -> List[str]:
    urls: List[str] = []
    urls += _safe_urls(center.operating.urls)
    urls += _safe_urls(center.acs_verification.urls)
    urls += _all_core_urls(center)
    urls += _specialized_urls_for_index(center, idx1_based)
    # De-duplicate while preserving order
    deduped = []
    seen = set()
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _normalize_name(name: Optional[str]) -> str:
    return (name or "").strip().lower()


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def _verify_leaf(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    desc: str,
    claim: str,
    urls: Optional[List[str]],
    critical: bool,
    add_ins: str = "None",
):
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls or [],
        additional_instruction=add_ins,
    )


async def _add_url_presence_checks(
    evaluator: Evaluator,
    parent_node,
    idx1_based: int,
    center: CenterInfo,
):
    # Parent node for URL references (critical, parallel)
    url_refs_node = evaluator.add_parallel(
        id=f"H{idx1_based}_URL_References",
        desc=f"Provides URL references supporting each major capability category for Center {idx1_based} (ACS status, specialized capabilities, core staffing/time requirements, volume, and quality/substance programs).",
        parent=parent_node,
        critical=True
    )

    # ACS status URLs present
    evaluator.add_custom_node(
        result=len(_safe_urls(center.acs_verification.urls)) > 0,
        id=f"H{idx1_based}_URL_ACS_Status",
        desc=f"Center {idx1_based} provides URLs for ACS Level I verification.",
        parent=url_refs_node,
        critical=True
    )

    # Core staffing/time URLs present (any of the staffing/time categories)
    evaluator.add_custom_node(
        result=len(_all_core_staffing_timing_urls(center)) > 0,
        id=f"H{idx1_based}_URL_Core_Staffing_Time",
        desc=f"Center {idx1_based} provides URLs for core staffing/time requirements (GS 24h, anesthesia 15m, neuro 30m, radiology 30m).",
        parent=url_refs_node,
        critical=True
    )

    # Volume URLs present
    evaluator.add_custom_node(
        result=len(_safe_urls(center.core.volume_requirement.urls)) > 0,
        id=f"H{idx1_based}_URL_Volume",
        desc=f"Center {idx1_based} provides URLs for volume requirement evidence.",
        parent=url_refs_node,
        critical=True
    )

    # Quality assessment URLs present
    evaluator.add_custom_node(
        result=len(_safe_urls(center.core.quality_assessment.urls)) > 0,
        id=f"H{idx1_based}_URL_Quality",
        desc=f"Center {idx1_based} provides URLs for comprehensive quality assessment program.",
        parent=url_refs_node,
        critical=True
    )

    # Substance abuse program URLs present
    evaluator.add_custom_node(
        result=len(_safe_urls(center.core.substance_abuse.urls)) > 0,
        id=f"H{idx1_based}_URL_Substance",
        desc=f"Center {idx1_based} provides URLs for substance abuse screening and intervention program.",
        parent=url_refs_node,
        critical=True
    )

    # Specialized URLs present (depending on center index)
    evaluator.add_custom_node(
        result=len(_specialized_urls_for_index(center, idx1_based)) > 0,
        id=f"H{idx1_based}_URL_Specialized",
        desc=f"Center {idx1_based} provides URLs for its specialized capability set.",
        parent=url_refs_node,
        critical=True
    )


async def _verify_core_requirements(
    evaluator: Evaluator,
    parent_node,
    idx1_based: int,
    center: CenterInfo
):
    core_node = evaluator.add_parallel(
        id=f"H{idx1_based}_Core_LevelI_Requirements",
        desc=f"Center {idx1_based} meets all core Level I trauma center requirements listed in the question.",
        parent=parent_node,
        critical=True
    )

    name = center.name or "the hospital"
    # General Surgery 24hr
    await _verify_leaf(
        evaluator, core_node,
        node_id=f"H{idx1_based}_General_Surgery_24hr",
        desc="24-hour in-house general surgeon coverage.",
        claim=f"The trauma center at {name} provides 24-hour in-house general surgeon coverage.",
        urls=_safe_urls(center.core.general_surgery_24hr.urls),
        critical=True,
        add_ins="Evidence may be on the hospital's trauma services page, ACS verification details, or official policy documents."
    )

    # ICU Director board-certified/eligible in surgical critical care
    await _verify_leaf(
        evaluator, core_node,
        node_id=f"H{idx1_based}_ICU_Director",
        desc="ICU surgical director is board-certified or board-eligible in surgical critical care.",
        claim=f"The ICU surgical director at {name} is board-certified or board-eligible in surgical critical care.",
        urls=_safe_urls(center.core.icu_director_critical_care.urls),
        critical=True,
        add_ins="Look for trauma/ICU leadership bios, credential statements, or ACS review details."
    )

    # Anesthesia within 15 minutes
    await _verify_leaf(
        evaluator, core_node,
        node_id=f"H{idx1_based}_Anesthesia_15min",
        desc="Anesthesia services available within 15 minutes of request.",
        claim=f"Anesthesia services at {name} are available within 15 minutes of request.",
        urls=_safe_urls(center.core.anesthesia_15min.urls),
        critical=True,
        add_ins="Accept explicit policies or official statements matching this timing requirement."
    )

    # Neurosurgery within 30 minutes
    await _verify_leaf(
        evaluator, core_node,
        node_id=f"H{idx1_based}_Neurosurgery_30min",
        desc="Neurosurgery evaluation available within 30 minutes for specified injuries.",
        claim=f"Neurosurgery evaluation at {name} is available within 30 minutes for specified injuries.",
        urls=_safe_urls(center.core.neurosurgery_30min.urls),
        critical=True,
        add_ins="Look for trauma coverage model descriptions or ACS verification summaries specifying timing."
    )

    # Radiology within 30 minutes
    await _verify_leaf(
        evaluator, core_node,
        node_id=f"H{idx1_based}_Radiology_30min",
        desc="Radiologist available within 30 minutes to access patient images.",
        claim=f"Radiology at {name} provides radiologist availability within 30 minutes to access patient images.",
        urls=_safe_urls(center.core.radiology_30min.urls),
        critical=True,
        add_ins="Accept statements regarding time-to-read/coverage that reasonably imply this standard."
    )

    # Volume requirement
    await _verify_leaf(
        evaluator, core_node,
        node_id=f"H{idx1_based}_Volume",
        desc="Annual volume is at least 1,200 trauma cases OR at least 240 patients with ISS > 15.",
        claim=f"{name} meets the ACS Level I volume requirement (≥1,200 trauma cases annually OR ≥240 patients with ISS > 15).",
        urls=_safe_urls(center.core.volume_requirement.urls),
        critical=True,
        add_ins="Look for annual trauma volumes or ISS>15 counts; either threshold suffices."
    )

    # Quality assessment program
    await _verify_leaf(
        evaluator, core_node,
        node_id=f"H{idx1_based}_Quality_Assessment",
        desc="Comprehensive quality assessment program is present.",
        claim=f"{name} has a comprehensive trauma quality assessment and performance improvement program.",
        urls=_safe_urls(center.core.quality_assessment.urls),
        critical=True,
        add_ins="Accept explicit references to trauma quality/performance improvement programs or ACS verification details."
    )

    # Substance abuse screening and intervention
    await _verify_leaf(
        evaluator, core_node,
        node_id=f"H{idx1_based}_Substance_Abuse",
        desc="Substance abuse screening and intervention program is present.",
        claim=f"{name} has a substance abuse screening and intervention program for trauma patients.",
        urls=_safe_urls(center.core.substance_abuse.urls),
        critical=True,
        add_ins="Accept SBIRT programs or equivalent substance use screening/intervention references."
    )


async def _verify_specialized_requirements(
    evaluator: Evaluator,
    parent_node,
    idx1_based: int,
    center: CenterInfo
):
    spec_node = evaluator.add_parallel(
        id=f"H{idx1_based}_Specialized_Requirements",
        desc=f"Center {idx1_based} fulfills the Center-{idx1_based} specialized capability set.",
        parent=parent_node,
        critical=True
    )
    name = center.name or "the hospital"

    if idx1_based == 1:
        # Pediatric volume (≥200 under age 15) and Level I pediatric trauma services
        await _verify_leaf(
            evaluator, spec_node,
            node_id="H1_Peds_Volume",
            desc="Provides Level I pediatric trauma services treating at least 200 trauma patients under age 15 annually.",
            claim=f"{name} provides Level I pediatric trauma services and treats at least 200 trauma patients under age 15 annually.",
            urls=_safe_urls(center.specialized1.peds_volume.urls),
            critical=True,
            add_ins="Accept explicit pediatric Level I trauma designation and clear statement that ≥200 under age 15 are treated annually."
        )
        # Research program with peer-reviewed publications
        await _verify_leaf(
            evaluator, spec_node,
            node_id="H1_Research",
            desc="Has an organized research program with peer-reviewed publications in indexed journals.",
            claim=f"{name} has an organized trauma research program with peer-reviewed publications in indexed journals.",
            urls=_safe_urls(center.specialized1.research_program.urls),
            critical=True,
            add_ins="Look for mentions of research programs and publications in peer‑reviewed indexed journals (e.g., PubMed-listed)."
        )
        # Trauma rotations for residents in specified specialties
        await _verify_leaf(
            evaluator, spec_node,
            node_id="H1_Trauma_Rotations",
            desc="Trauma rotations available to residents in general surgery, orthopedic surgery, neurosurgery, and emergency medicine.",
            claim=f"{name} offers trauma rotations for residents in general surgery, orthopedic surgery, neurosurgery, and emergency medicine.",
            urls=_safe_urls(center.specialized1.rotations.urls),
            critical=True,
            add_ins="Accept GME/Residency rotation descriptions that explicitly include all four specialties."
        )

    elif idx1_based == 2:
        # Burn center verified (ABA or equivalent)
        await _verify_leaf(
            evaluator, spec_node,
            node_id="H2_Burn_Verified",
            desc="Operates a burn center verified by the American Burn Association or equivalent authority.",
            claim=f"{name} operates a burn center verified by the American Burn Association (ABA) or an equivalent recognized authority.",
            urls=_safe_urls(center.specialized2.burn_center_verified.urls),
            critical=True,
            add_ins="Prefer ABA verification listings; accept equivalent recognized verification if clearly stated."
        )
        # Dedicated OR prioritized for ortho trauma fracture care
        await _verify_leaf(
            evaluator, spec_node,
            node_id="H2_Dedicated_OR",
            desc="Has a dedicated operating room prioritized for orthopaedic trauma fracture care.",
            claim=f"The trauma center at {name} has a dedicated operating room prioritized for orthopedic trauma fracture care.",
            urls=_safe_urls(center.specialized2.dedicated_ortho_or.urls),
            critical=True,
            add_ins="Look for perioperative scheduling policies or facilities pages stating a dedicated ortho trauma OR."
        )
        # CPB equipment or transfer agreement
        await _verify_leaf(
            evaluator, spec_node,
            node_id="H2_CPB",
            desc="Has cardiopulmonary bypass equipment or a documented transfer agreement for such services.",
            claim=f"{name} has cardiopulmonary bypass equipment on site or a documented transfer agreement for cardiopulmonary bypass services.",
            urls=_safe_urls(center.specialized2.cpb_or_transfer.urls),
            critical=True,
            add_ins="Accept either on-site capability or a formal transfer/affiliation agreement that explicitly covers CPB."
        )

    elif idx1_based == 3:
        # Specialized neurotrauma care for moderate to severe TBI (GCS ≤12)
        await _verify_leaf(
            evaluator, spec_node,
            node_id="H3_TBI_Care",
            desc="Provides specialized neurotrauma care for moderate to severe TBI (GCS ≤12).",
            claim=f"{name} provides specialized neurotrauma care for moderate to severe TBI (GCS ≤ 12).",
            urls=_safe_urls(center.specialized3.neurotrauma_tbi.urls),
            critical=True,
            add_ins="Accept neurotrauma programs describing care pathways for moderate/severe TBI and GCS-based triage."
        )
        # Board-certified or board-eligible neurosurgeons
        await _verify_leaf(
            evaluator, spec_node,
            node_id="H3_Neurosurgeons",
            desc="Has board-certified or board-eligible neurosurgeons on staff.",
            claim=f"{name} has board-certified or board-eligible neurosurgeons on staff for trauma care.",
            urls=_safe_urls(center.specialized3.neurosurgeons.urls),
            critical=True,
            add_ins="Verify neurosurgery coverage with credential statements or physician bios indicating certification/eligibility."
        )
        # Trauma registry staffing ratio
        await _verify_leaf(
            evaluator, spec_node,
            node_id="H3_Registry_Staffing",
            desc="Maintains trauma registry operations with 0.5 FTE dedicated registry professional per 200–300 annual patient entries.",
            claim=f"The trauma registry at {name} is staffed at least at a ratio of 0.5 FTE per 200–300 annual patient entries.",
            urls=_safe_urls(center.specialized3.registry_staffing.urls),
            critical=True,
            add_ins="Accept clear staffing statements or ACS review excerpts indicating compliance with this ratio."
        )
        # Performance improvement staffing thresholds
        await _verify_leaf(
            evaluator, spec_node,
            node_id="H3_PI_Staffing",
            desc="Performance improvement staffing meets ACS requirements (≥0.5 FTE if annual volume >500; ≥1.0 FTE if volume >1,000).",
            claim=f"The performance improvement program staffing at {name} meets ACS thresholds (≥0.5 FTE if annual volume >500; ≥1.0 FTE if volume >1,000).",
            urls=_safe_urls(center.specialized3.pi_staffing.urls),
            critical=True,
            add_ins="Accept explicit PI staffing FTE statements tied to trauma volumes or ACS verification documentation."
        )


async def _verify_center(
    evaluator: Evaluator,
    root,
    center: CenterInfo,
    idx1_based: int
):
    # Parent center node (parallel, non-critical per rubric)
    center_node = evaluator.add_parallel(
        id=f"Hospital_{idx1_based}",
        desc=(
            "Center 1: Level I trauma center in Texas with pediatric volume + research + resident rotations specialization; includes required fields and evidence/URLs."
            if idx1_based == 1 else
            "Center 2: Level I trauma center in Texas with burn verification + dedicated ortho trauma OR + cardiopulmonary bypass/transfer specialization; includes required fields and evidence/URLs."
            if idx1_based == 2 else
            "Center 3: Level I trauma center in Texas with neurotrauma + registry staffing + PI staffing specialization; includes required fields and evidence/URLs."
        ),
        parent=root,
        critical=False
    )

    # Output fields (critical): hospital name + city
    output_fields_node = evaluator.add_parallel(
        id=f"H{idx1_based}_Output_Fields",
        desc=f"Provides hospital name and Texas city location for Center {idx1_based}.",
        parent=center_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(center.name and center.name.strip()),
        id=f"H{idx1_based}_Hospital_Name",
        desc="Hospital name is provided.",
        parent=output_fields_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(center.city and center.city.strip()),
        id=f"H{idx1_based}_City",
        desc="City in Texas is provided.",
        parent=output_fields_node,
        critical=True
    )

    # Currently operating (critical verification)
    await _verify_leaf(
        evaluator, center_node,
        node_id=f"H{idx1_based}_Currently_Operating",
        desc=f"Center {idx1_based} is currently operating.",
        claim=f"{center.name or 'The hospital'} is currently operating as an active hospital (not closed/inactive).",
        urls=_safe_urls(center.operating.urls) or _all_center_urls(center, idx1_based),
        critical=True,
        add_ins="Evidence can include an active hospital website, ACS facility listing, or state registry indicating current operation."
    )

    # ACS verification (critical verification)
    await _verify_leaf(
        evaluator, center_node,
        node_id=f"H{idx1_based}_ACS_Verification",
        desc=f"Center {idx1_based} is verified by the American College of Surgeons (ACS) as a Level I trauma center.",
        claim=f"{center.name or 'The hospital'} is verified by the American College of Surgeons (ACS) as a Level I trauma center.",
        urls=_safe_urls(center.acs_verification.urls),
        critical=True,
        add_ins="Prefer ACS (facs.org) listing pages; accept official hospital pages clearly citing current ACS Level I verification."
    )

    # Core Level I requirements (critical, parallel with multiple leaves)
    await _verify_core_requirements(evaluator, center_node, idx1_based, center)

    # Specialized requirements (critical, parallel with multiple leaves)
    await _verify_specialized_requirements(evaluator, center_node, idx1_based, center)

    # URL references presence checks (critical, parallel)
    await _add_url_presence_checks(evaluator, center_node, idx1_based, center)


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
    Evaluate an answer for the Texas Level I trauma centers task.
    Notes:
    - Root node is initialized as non-critical to allow mixed criticality children (framework requires critical parents
      to have only critical children). All required sub-criteria are marked critical at their respective nodes.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root follows rubric's parallel aggregation
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

    # 1) Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_centers(),
        template_class=TraumaCentersExtraction,
        extraction_name="trauma_centers_extraction"
    )

    # Prepare exactly 3 centers (pad with empty objects if fewer)
    centers = list(extracted.centers[:3])
    while len(centers) < 3:
        centers.append(CenterInfo())

    # 2) Set completeness check (critical): exactly three distinct centers (by name), no duplicates
    valid_names = [c.name for c in centers if c.name and c.name.strip()]
    unique_name_count = len({ _normalize_name(n) for n in valid_names })
    set_complete = (len(valid_names) == 3) and (unique_name_count == 3)

    evaluator.add_custom_node(
        result=set_complete,
        id="Set_Completeness",
        desc="Response provides exactly three distinct trauma centers (no duplicates).",
        parent=root,
        critical=True
    )

    # 3) Build verification subtrees for each center
    for idx, center in enumerate(centers, start=1):
        await _verify_center(evaluator, root, center, idx)

    # 4) Return standardized summary
    return evaluator.get_summary()