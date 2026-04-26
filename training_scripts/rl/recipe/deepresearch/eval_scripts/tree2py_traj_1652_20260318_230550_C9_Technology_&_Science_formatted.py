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
TASK_ID = "tech_deployment_2026"
TASK_DESCRIPTION = """
Identify 4 countries or regions that in 2026 simultaneously meet ALL of the following technology deployment criteria:

1. Advanced Semiconductor Manufacturing: Operates semiconductor fabrication facilities producing chips at 3-nanometer (3nm) process node or smaller (including 2nm-class processes), with volume production started or planned for 2026-2027.

2. 5G Standalone Network Deployment: Achieved either:
   - 5G SA sample share exceeding 20% of mobile connections, OR
   - Median 5G SA download speed exceeding 400 Mbps (based on Q4 2025 or early 2026 data)

3. Advanced Battery Technology: Engaged in manufacturing or development of electric vehicle batteries with gravimetric energy density of at least 170 Wh/kg.

For each country or region identified, provide:
- The country/region name
- A reference URL supporting EACH of the three criteria (minimum 3 URLs per country)
- A brief description explaining how each criterion is satisfied
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CriterionEvidence(BaseModel):
    explanation: Optional[str] = None
    urls: List[str] = Field(default_factory=list)
    timeframe: Optional[str] = None
    metric: Optional[str] = None


class CountryItem(BaseModel):
    name: Optional[str] = None
    semiconductor: Optional[CriterionEvidence] = None
    fiveg: Optional[CriterionEvidence] = None
    battery: Optional[CriterionEvidence] = None


class CountriesExtraction(BaseModel):
    countries: List[CountryItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_countries() -> str:
    return """
    Extract up to the first 4 countries or regions listed in the answer that claim to meet ALL three criteria:
    (1) Advanced Semiconductor Manufacturing (3nm or smaller, incl. 2nm-class; volume production started or planned 2026–2027),
    (2) 5G Standalone deployment thresholds (SA sample share >20% OR median SA download >400 Mbps) with Q4 2025 or early 2026 data,
    (3) Advanced EV battery technology with gravimetric energy density ≥170 Wh/kg.

    For each selected country, return:
    - name: Country/region name as written in the answer text.
    - semiconductor: 
        - explanation: brief note from the answer about 3nm/2nm-class capability and timeline.
        - urls: all URLs cited in the answer that support the semiconductor claim.
        - timeframe: any mentioned production timing (e.g., "Q4 2025", "2026", "2027"), if present.
        - metric: any useful note (e.g., "N3 in volume", "2nm ramp 2026"), if present.
    - fiveg:
        - explanation: brief note about 5G SA status from the answer.
        - urls: all URLs cited that support the 5G SA thresholds.
        - timeframe: the data period mentioned (e.g., "Q4 2025", "Q1 2026"), if present.
        - metric: mention which threshold (e.g., "SA share 23%", "median SA 450 Mbps"), if present.
    - battery:
        - explanation: brief note about ≥170 Wh/kg EV battery tech from the answer.
        - urls: all URLs cited that support the battery claim.
        - timeframe: any commercialization/operational timing (e.g., "in 2026"), if present.
        - metric: any energy density note (e.g., "180 Wh/kg NCM"), if present.

    IMPORTANT RULES:
    - Extract only what appears in the provided answer. Do not invent URLs or names.
    - For each criterion (semiconductor, fiveg, battery), include all URLs explicitly present in the answer text for that country.
    - Return at most 4 country entries in the 'countries' array. If more are in the answer, keep only the first 4.
    - If a field is not mentioned, set it to null (or [] for urls).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    if not urls:
        return []
    # Preserve order while removing duplicates
    seen = set()
    out = []
    for u in urls:
        if u and u not in seen:
            out.append(u)
            seen.add(u)
    return out


def _country_display_name(idx: int, item: Optional[CountryItem]) -> str:
    if item and item.name:
        return item.name
    return f"Country/Region #{idx}"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def _verify_semiconductor(
    evaluator: Evaluator,
    parent_node,
    country_idx: int,
    country_name: str,
    evidence: Optional[CriterionEvidence],
):
    sem_node = evaluator.add_parallel(
        id=f"country_{country_idx}_semiconductor",
        desc=f"Semiconductor manufacturing criterion for {'first' if country_idx==1 else ('second' if country_idx==2 else ('third' if country_idx==3 else 'fourth'))} country",
        parent=parent_node,
        critical=True,
    )

    urls = _dedup_urls(evidence.urls if evidence else [])

    # URL existence (critical)
    url_exist_node = evaluator.add_custom_node(
        result=len(urls) > 0,
        id=f"country_{country_idx}_semiconductor_url",
        desc="Reference URL provided supporting semiconductor manufacturing claim",
        parent=sem_node,
        critical=True,
    )

    # Facility capability at 3nm or smaller (critical)
    facility_node = evaluator.add_leaf(
        id=f"country_{country_idx}_semiconductor_facility",
        desc="Country operates facilities producing semiconductors at 3nm node or smaller",
        parent=sem_node,
        critical=True,
    )
    claim_facility = (
        f"{country_name} operates or hosts semiconductor fabrication facilities capable of producing chips "
        f"at 3nm or smaller nodes (including 2nm-class). The cited source(s) explicitly indicate 3nm/2nm-class capability."
    )
    await evaluator.verify(
        claim=claim_facility,
        node=facility_node,
        sources=urls,
        additional_instruction=(
            "Accept synonyms such as 'N3', '3 nm-class', '2 nm', '2 nm-class'. "
            "The support can be from foundry roadmaps, official announcements, or reliable tech reports indicating 3nm/2nm capability."
        ),
    )

    # Volume production timing (critical)
    volume_node = evaluator.add_leaf(
        id=f"country_{country_idx}_semiconductor_volume",
        desc="Volume production started by Q4 2025 or planned for 2026-2027",
        parent=sem_node,
        critical=True,
    )
    claim_volume = (
        f"In {country_name}, 3nm-class (3nm or 2nm) volume production either started by Q4 2025 "
        f"or is clearly planned/scheduled for 2026 or 2027, per the cited sources."
    )
    await evaluator.verify(
        claim=claim_volume,
        node=volume_node,
        sources=urls,
        additional_instruction=(
            "Look for clear phrasing like 'volume production', 'high-volume manufacturing', 'HVM', "
            "'ramp in 2026', 'mass production in 2027', or similar schedule confirmations. "
            "If the evidence only suggests R&D with no schedule, consider it unsupported."
        ),
    )


async def _verify_5g(
    evaluator: Evaluator,
    parent_node,
    country_idx: int,
    country_name: str,
    evidence: Optional[CriterionEvidence],
):
    fiveg_node = evaluator.add_parallel(
        id=f"country_{country_idx}_5g",
        desc=f"5G Standalone deployment criterion for {'first' if country_idx==1 else ('second' if country_idx==2 else ('third' if country_idx==3 else 'fourth'))} country",
        parent=parent_node,
        critical=True,
    )

    urls = _dedup_urls(evidence.urls if evidence else [])

    # URL existence (critical)
    url_exist_node = evaluator.add_custom_node(
        result=len(urls) > 0,
        id=f"country_{country_idx}_5g_url",
        desc="Reference URL provided supporting 5G SA deployment claim",
        parent=fiveg_node,
        critical=True,
    )

    # Threshold satisfied (critical)
    threshold_node = evaluator.add_leaf(
        id=f"country_{country_idx}_5g_threshold",
        desc="Country meets at least one 5G SA threshold: >20% sample share OR >400 Mbps median download speed",
        parent=fiveg_node,
        critical=True,
    )
    claim_threshold = (
        f"In {country_name}, according to the cited sources, 5G Standalone (SA) meets at least one threshold: "
        f"SA sample share exceeds 20% of mobile connections, OR median SA download speed exceeds 400 Mbps."
    )
    await evaluator.verify(
        claim=claim_threshold,
        node=threshold_node,
        sources=urls,
        additional_instruction=(
            "Ensure it is 5G Standalone (NR SA) specifically, not NSA. "
            "Accept equivalent expressions like 'median SA download speed > 400 Mbps' or 'SA share > 20%'. "
            "If multiple datasets exist, any one credible dataset suffices."
        ),
    )

    # Data recency (critical)
    data_source_node = evaluator.add_leaf(
        id=f"country_{country_idx}_5g_data_source",
        desc="5G SA statistics based on Q4 2025 or early 2026 data",
        parent=fiveg_node,
        critical=True,
    )
    claim_timeframe = (
        f"The cited 5G SA statistics for {country_name} correspond to Q4 2025 or early 2026 "
        f"(e.g., Q1 2026 or Jan–Mar 2026)."
    )
    await evaluator.verify(
        claim=claim_timeframe,
        node=data_source_node,
        sources=urls,
        additional_instruction=(
            "Look for date stamps like 'Q4 2025', 'Dec 2025', 'Q1 2026', or similar indications. "
            "If the source clearly states data for Q4 2025 or early 2026, mark as supported. "
            "If dates fall far outside this window, mark unsupported."
        ),
    )


async def _verify_battery(
    evaluator: Evaluator,
    parent_node,
    country_idx: int,
    country_name: str,
    evidence: Optional[CriterionEvidence],
):
    battery_node = evaluator.add_parallel(
        id=f"country_{country_idx}_battery",
        desc=f"Advanced battery technology criterion for {'first' if country_idx==1 else ('second' if country_idx==2 else ('third' if country_idx==3 else 'fourth'))} country",
        parent=parent_node,
        critical=True,
    )

    urls = _dedup_urls(evidence.urls if evidence else [])

    # URL existence (critical)
    url_exist_node = evaluator.add_custom_node(
        result=len(urls) > 0,
        id=f"country_{country_idx}_battery_url",
        desc="Reference URL provided supporting battery technology claim",
        parent=battery_node,
        critical=True,
    )

    # Energy density threshold (critical)
    density_node = evaluator.add_leaf(
        id=f"country_{country_idx}_battery_density",
        desc="Country engaged in EV battery manufacturing/development with energy density ≥170 Wh/kg",
        parent=battery_node,
        critical=True,
    )
    claim_density = (
        f"In {country_name}, companies are manufacturing or developing EV batteries with gravimetric energy density "
        f"of at least 170 Wh/kg, as supported by the cited sources."
    )
    await evaluator.verify(
        claim=claim_density,
        node=density_node,
        sources=urls,
        additional_instruction=(
            "Focus on gravimetric energy density (Wh/kg), not volumetric (Wh/L). "
            "Accept chemistry types like NCM, NCA, LFP (if ≥170 Wh/kg), LMFP, solid-state etc., if the Wh/kg threshold is met. "
            "Spec sheets, OEM announcements, or credible reports are acceptable."
        ),
    )

    # Operational/commercialization status (critical)
    status_node = evaluator.add_leaf(
        id=f"country_{country_idx}_battery_status",
        desc="Battery technology operational or in advanced commercialization as of 2026",
        parent=battery_node,
        critical=True,
    )
    claim_status = (
        f"As of 2026, the ≥170 Wh/kg EV battery technology in {country_name} is operational or in advanced commercialization "
        f"(e.g., mass production, vehicle integration/shipments, or pilot lines with committed 2026 deliveries)."
    )
    await evaluator.verify(
        claim=claim_status,
        node=status_node,
        sources=urls,
        additional_instruction=(
            "Look for language like 'mass production', 'commercialized', 'entered production', 'shipping in 2026', "
            "'gigafactory ramp', or confirmed deployments. Pure lab results without commercialization signals should be considered insufficient."
        ),
    )


async def _verify_country(
    evaluator: Evaluator,
    root,
    country_item: Optional[CountryItem],
    country_idx: int,
):
    idx_str = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}.get(country_idx, f"Country {country_idx}")
    country_node = evaluator.add_parallel(
        id=f"country_{country_idx}",
        desc=f"{idx_str.lower()} country/region evaluation" if country_idx > 1 else "First country/region evaluation",
        parent=root,
        critical=False,
    )

    country_name = _country_display_name(country_idx, country_item)

    # Semiconductor criterion
    await _verify_semiconductor(
        evaluator=evaluator,
        parent_node=country_node,
        country_idx=country_idx,
        country_name=country_name,
        evidence=country_item.semiconductor if country_item else None,
    )

    # 5G SA criterion
    await _verify_5g(
        evaluator=evaluator,
        parent_node=country_node,
        country_idx=country_idx,
        country_name=country_name,
        evidence=country_item.fiveg if country_item else None,
    )

    # Battery criterion
    await _verify_battery(
        evaluator=evaluator,
        parent_node=country_node,
        country_idx=country_idx,
        country_name=country_name,
        evidence=country_item.battery if country_item else None,
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
    Evaluate an answer for the 2026 technology deployment criteria across 4 countries/regions.
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

    # Extract structured country evidence
    extracted = await evaluator.extract(
        prompt=prompt_extract_countries(),
        template_class=CountriesExtraction,
        extraction_name="countries_extraction",
    )

    # Keep only first 4 countries; pad with empty if fewer
    items: List[CountryItem] = list(extracted.countries[:4])
    while len(items) < 4:
        items.append(CountryItem())

    # Build verification tree per country
    # Root description per JSON
    root.desc = "Evaluation of 4 countries/regions meeting advanced semiconductor, 5G SA, and battery technology criteria in 2026"

    for i in range(4):
        await _verify_country(
            evaluator=evaluator,
            root=root,
            country_item=items[i],
            country_idx=i + 1,
        )

    return evaluator.get_summary()