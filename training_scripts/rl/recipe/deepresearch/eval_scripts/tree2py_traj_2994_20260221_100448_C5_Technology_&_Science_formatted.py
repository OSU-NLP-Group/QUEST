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
TASK_ID = "us_colocation_tier3_eval"
TASK_DESCRIPTION = """
Identify a colocation data center facility in the United States that meets all of the following technical infrastructure requirements: 
(1) The facility must have Tier III certification (or higher) from the Uptime Institute, 
(2) The facility must guarantee an uptime of at least 99.982%, 
(3) The facility must support concurrent maintainability, allowing maintenance activities without service disruption, 
(4) The facility's power systems must use at least N+1 redundancy configuration, 
(5) The facility's cooling systems must use at least N+1 redundancy configuration, 
(6) The facility must have single-mode fiber optic cable infrastructure with minimum 12-strand fiber connections between critical telecommunications points, and 
(7) All technical specifications (tier certification, uptime guarantee, redundancy configurations, and fiber connectivity) must be verifiable through publicly accessible documentation with valid URLs. 
Provide the facility name, location, and reference URLs documenting each of the required specifications.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FacilitySpecSources(BaseModel):
    """Per-requirement source URLs explicitly cited in the answer."""
    tier_cert_urls: List[str] = Field(default_factory=list)
    uptime_urls: List[str] = Field(default_factory=list)
    concurrent_maint_urls: List[str] = Field(default_factory=list)
    power_redundancy_urls: List[str] = Field(default_factory=list)
    cooling_redundancy_urls: List[str] = Field(default_factory=list)
    fiber_infrastructure_urls: List[str] = Field(default_factory=list)
    fiber_strand_urls: List[str] = Field(default_factory=list)
    location_urls: List[str] = Field(default_factory=list)
    general_urls: List[str] = Field(default_factory=list)  # All URLs mentioned in the answer


class FacilityExtraction(BaseModel):
    """Facility identification and per-spec sources extracted from the answer."""
    facility_name: Optional[str] = None
    location_text: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None

    # Optional claimed values/descriptions as stated in the answer (free text)
    tier_level: Optional[str] = None
    uptime_percentage: Optional[str] = None
    power_redundancy_desc: Optional[str] = None
    cooling_redundancy_desc: Optional[str] = None
    fiber_infrastructure_desc: Optional[str] = None
    fiber_strand_desc: Optional[str] = None

    sources: FacilitySpecSources = Field(default_factory=FacilitySpecSources)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_facility_and_sources() -> str:
    return """
    Extract the data center facility identification details and categorize all URLs presented in the answer according to the specified technical requirements. You must only extract information explicitly mentioned in the answer text.

    Required fields:
    - facility_name: The name of the specific data center facility (e.g., a particular building/campus/site). If the answer only gives a brand/provider without a specific facility name, return the brand/provider name as given.
    - location_text: The location string as provided (e.g., "Ashburn, VA, USA"). If not provided, return null.
    - city: Extract the city if explicitly stated; otherwise null.
    - state: Extract the state or region (e.g., "VA", "California"); otherwise null.
    - country: Extract the country if explicitly stated (e.g., "United States"); otherwise null.

    Optional claimed values/descriptions (free text exactly as stated in the answer):
    - tier_level: The Tier level mentioned (e.g., "Tier III", "Tier IV").
    - uptime_percentage: The uptime percentage mentioned (e.g., "99.982%", "99.99%").
    - power_redundancy_desc: Any description of power redundancy (e.g., "N+1 UPS", "2N power").
    - cooling_redundancy_desc: Any description of cooling redundancy (e.g., "N+1 cooling").
    - fiber_infrastructure_desc: Any description of single-mode fiber infrastructure as stated.
    - fiber_strand_desc: Any description mentioning minimum strand count (e.g., "12-strand fiber").

    URL categorization (extract real URLs explicitly present in the answer; do not invent):
    - sources.tier_cert_urls: URLs that the answer cites as evidence for Uptime Institute Tier III (or higher) certification.
    - sources.uptime_urls: URLs that the answer cites as evidence for the uptime guarantee/SLA (>= 99.982%).
    - sources.concurrent_maint_urls: URLs cited as evidence for concurrent maintainability (maintenance without service disruption).
    - sources.power_redundancy_urls: URLs cited as evidence for power systems redundancy (at least N+1).
    - sources.cooling_redundancy_urls: URLs cited as evidence for cooling systems redundancy (at least N+1).
    - sources.fiber_infrastructure_urls: URLs cited as evidence for single-mode fiber optic infrastructure.
    - sources.fiber_strand_urls: URLs cited as evidence for minimum 12-strand fiber connections between critical telecommunications points.
    - sources.location_urls: URLs cited as evidence of the facility being located in the United States (e.g., address/contact page for the specific facility).
    - sources.general_urls: All URLs mentioned anywhere in the answer (including the above; duplicates allowed, but prefer unique URLs if possible).

    Special rules for URL extraction:
    - Extract only valid URLs explicitly present in the answer (including markdown links). Do not infer or create URLs.
    - If a URL is missing the protocol, prepend "http://".
    - If the answer provides a generic provider page and implies it supports multiple specs, include that URL in the relevant categories based on the answer's own statements.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _facility_ref_name(extracted: FacilityExtraction) -> str:
    return extracted.facility_name.strip() if extracted.facility_name else "the facility"


def _pick_sources(primary: List[str], fallback: List[str]) -> List[str]:
    """Use spec-specific URLs if present; otherwise fall back to general URLs."""
    if primary and len(primary) > 0:
        return primary
    return fallback if fallback else []


def _exists(value: Optional[str]) -> bool:
    return bool(value and value.strip())


def _location_info_present(extracted: FacilityExtraction) -> bool:
    return _exists(extracted.location_text) or (_exists(extracted.city) and _exists(extracted.state)) or _exists(extracted.country)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_tier_certification(
    evaluator: Evaluator,
    parent_node,
    extracted: FacilityExtraction,
) -> None:
    tier_node = evaluator.add_parallel(
        id="tier_certification",
        desc="Data center must have Tier III or higher certification from Uptime Institute, verifiable through documentation",
        parent=parent_node,
        critical=True
    )
    sources = _pick_sources(extracted.sources.tier_cert_urls, extracted.sources.general_urls)

    evaluator.add_custom_node(
        result=bool(sources),
        id="tier_certification_sources_exist",
        desc="Sources for Tier certification are provided",
        parent=tier_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="tier_certification_verify",
        desc="Tier III (or higher) certification from Uptime Institute is supported by cited sources",
        parent=tier_node,
        critical=True
    )
    claim = f"The facility {_facility_ref_name(extracted)} has Uptime Institute Tier III or higher certification."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that the cited page(s) explicitly state an Uptime Institute Tier certification for the specific facility. "
            "Accept variants such as 'Tier III Design Certification', 'Tier III Constructed Facility', 'Tier III Gold', 'Tier IV', etc. "
            "Mentions must refer to Uptime Institute certification, not just generic 'Tier-3-like'."
        )
    )


async def verify_uptime_guarantee(
    evaluator: Evaluator,
    parent_node,
    extracted: FacilityExtraction,
) -> None:
    uptime_node = evaluator.add_parallel(
        id="uptime_guarantee",
        desc="Data center must provide uptime guarantee of at least 99.982%, verifiable through publicly accessible SLA documentation",
        parent=parent_node,
        critical=True
    )
    sources = _pick_sources(extracted.sources.uptime_urls, extracted.sources.general_urls)

    evaluator.add_custom_node(
        result=bool(sources),
        id="uptime_sources_exist",
        desc="SLA/uptime guarantee sources are provided",
        parent=uptime_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="uptime_verify",
        desc="Uptime guarantee ≥ 99.982% is supported by cited sources",
        parent=uptime_node,
        critical=True
    )
    claim = f"The facility {_facility_ref_name(extracted)} provides an uptime guarantee of at least 99.982%."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=(
            "Check the SLA/availability commitment on the cited page(s). "
            "Accept any uptime percentage ≥ 99.982% (e.g., 99.99%, 99.995%, 99.999%). "
            "Marketing phrases must be backed by explicit SLA or availability figures."
        )
    )


async def verify_concurrent_maintainability(
    evaluator: Evaluator,
    parent_node,
    extracted: FacilityExtraction,
) -> None:
    cm_node = evaluator.add_parallel(
        id="concurrent_maintainability",
        desc="Data center must support concurrent maintainability (maintenance without service disruption), verifiable through documentation",
        parent=parent_node,
        critical=True
    )
    sources = _pick_sources(extracted.sources.concurrent_maint_urls, extracted.sources.general_urls)

    evaluator.add_custom_node(
        result=bool(sources),
        id="concurrent_maint_sources_exist",
        desc="Sources for concurrent maintainability are provided",
        parent=cm_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="concurrent_maint_verify",
        desc="Concurrent maintainability support is evidenced by cited sources",
        parent=cm_node,
        critical=True
    )
    claim = f"The facility {_facility_ref_name(extracted)} supports concurrent maintainability, allowing maintenance without service disruption."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=(
            "Accept phrases like 'concurrently maintainable', 'maintenance without service interruption', or equivalent wording. "
            "If the page merely implies Tier III properties, ensure it explicitly mentions concurrent maintainability."
        )
    )


async def verify_redundancy_configuration(
    evaluator: Evaluator,
    parent_node,
    extracted: FacilityExtraction,
) -> None:
    red_node = evaluator.add_parallel(
        id="redundancy_configuration",
        desc="Infrastructure redundancy specifications must meet minimum requirements and be documented",
        parent=parent_node,
        critical=True
    )

    # Power redundancy
    p_node = evaluator.add_parallel(
        id="power_redundancy",
        desc="Power systems must have at least N+1 redundancy configuration, verifiable through documentation",
        parent=red_node,
        critical=True
    )
    p_sources = _pick_sources(extracted.sources.power_redundancy_urls, extracted.sources.general_urls)

    evaluator.add_custom_node(
        result=bool(p_sources),
        id="power_redundancy_sources_exist",
        desc="Sources for power redundancy are provided",
        parent=p_node,
        critical=True
    )

    p_leaf = evaluator.add_leaf(
        id="power_redundancy_verify",
        desc="Power systems N+1 (or higher) redundancy is supported by cited sources",
        parent=p_node,
        critical=True
    )
    p_claim = f"The facility {_facility_ref_name(extracted)} has power systems configured with at least N+1 redundancy."
    await evaluator.verify(
        claim=p_claim,
        node=p_leaf,
        sources=p_sources,
        additional_instruction=(
            "Confirm at least N+1 redundancy for power systems (UPS, generators, power feeds/paths). "
            "Stronger configurations (e.g., N+2, 2N, 2N+1) also satisfy the 'at least N+1' requirement."
        )
    )

    # Cooling redundancy
    c_node = evaluator.add_parallel(
        id="cooling_redundancy",
        desc="Cooling systems must have at least N+1 redundancy configuration, verifiable through documentation",
        parent=red_node,
        critical=True
    )
    c_sources = _pick_sources(extracted.sources.cooling_redundancy_urls, extracted.sources.general_urls)

    evaluator.add_custom_node(
        result=bool(c_sources),
        id="cooling_redundancy_sources_exist",
        desc="Sources for cooling redundancy are provided",
        parent=c_node,
        critical=True
    )

    c_leaf = evaluator.add_leaf(
        id="cooling_redundancy_verify",
        desc="Cooling systems N+1 (or higher) redundancy is supported by cited sources",
        parent=c_node,
        critical=True
    )
    c_claim = f"The facility {_facility_ref_name(extracted)} has cooling systems configured with at least N+1 redundancy."
    await evaluator.verify(
        claim=c_claim,
        node=c_leaf,
        sources=c_sources,
        additional_instruction=(
            "Confirm at least N+1 redundancy for cooling systems (CRAC/CRAH/chillers). "
            "Stronger configurations (e.g., N+2, 2N, 2N+1) also satisfy the 'at least N+1' requirement."
        )
    )


async def verify_fiber_connectivity(
    evaluator: Evaluator,
    parent_node,
    extracted: FacilityExtraction,
) -> None:
    fiber_node = evaluator.add_parallel(
        id="fiber_connectivity",
        desc="Facility must have single-mode fiber optic connectivity meeting minimum specifications",
        parent=parent_node,
        critical=True
    )

    # Single-mode fiber infrastructure
    fi_sources = _pick_sources(extracted.sources.fiber_infrastructure_urls, extracted.sources.general_urls)

    fi_exist = evaluator.add_custom_node(
        result=bool(fi_sources),
        id="fiber_infrastructure_sources_exist",
        desc="Sources for single-mode fiber infrastructure are provided",
        parent=fiber_node,
        critical=True
    )

    fi_leaf = evaluator.add_leaf(
        id="fiber_infrastructure_verify",
        desc="Single-mode fiber optic cable infrastructure is supported by cited sources",
        parent=fiber_node,
        critical=True
    )
    fi_claim = f"The facility {_facility_ref_name(extracted)} has single-mode fiber optic cable infrastructure."
    await evaluator.verify(
        claim=fi_claim,
        node=fi_leaf,
        sources=fi_sources,
        additional_instruction=(
            "Look for explicit mentions of single-mode fiber (SMF). "
            "Accept synonyms like 'singlemode' or 'SMF'. The statement must clearly indicate single-mode fiber availability."
        )
    )

    # Minimum 12-strand connections
    fs_sources = _pick_sources(extracted.sources.fiber_strand_urls, extracted.sources.general_urls)

    evaluator.add_custom_node(
        result=bool(fs_sources),
        id="minimum_strand_sources_exist",
        desc="Sources for minimum 12-strand fiber connections are provided",
        parent=fiber_node,
        critical=True
    )

    fs_leaf = evaluator.add_leaf(
        id="minimum_strand_count",
        desc="Minimum 12-strand fiber connections between critical telecommunications points are supported by cited sources",
        parent=fiber_node,
        critical=True
    )
    fs_claim = f"The facility {_facility_ref_name(extracted)} provides minimum 12-strand fiber connections between critical telecommunications points."
    await evaluator.verify(
        claim=fs_claim,
        node=fs_leaf,
        sources=fs_sources,
        additional_instruction=(
            "Confirm language indicating '12-strand', '12-core', or equivalent fiber bundles between critical telecom points "
            "(e.g., meet-me rooms, core distribution frames). Accept higher strand counts if clearly minimum ≥ 12."
        )
    )


async def verify_geographic_location(
    evaluator: Evaluator,
    parent_node,
    extracted: FacilityExtraction,
) -> None:
    geo_node = evaluator.add_parallel(
        id="geographic_location",
        desc="Facility must be located in the United States",
        parent=parent_node,
        critical=True
    )

    # Facility identification existence (name + some location info)
    evaluator.add_custom_node(
        result=(_exists(extracted.facility_name) and _location_info_present(extracted)),
        id="facility_identified",
        desc="Facility name and some location information are provided in the answer",
        parent=geo_node,
        critical=True
    )

    # Sources existence for location/US verification (use fallback to general URLs)
    loc_sources = _pick_sources(extracted.sources.location_urls, extracted.sources.general_urls)
    evaluator.add_custom_node(
        result=bool(loc_sources),
        id="location_sources_exist",
        desc="Location/address sources are provided",
        parent=geo_node,
        critical=True
    )

    # Verify located in the United States (via cited URLs)
    geo_leaf = evaluator.add_leaf(
        id="geographic_location_verify",
        desc="United States location is supported by cited sources",
        parent=geo_node,
        critical=True
    )
    claim = f"The facility {_facility_ref_name(extracted)} is located in the United States."
    await evaluator.verify(
        claim=claim,
        node=geo_leaf,
        sources=loc_sources,
        additional_instruction=(
            "Check address or location indicators. Accept variants like 'USA', 'United States', 'U.S.', or state names/abbreviations "
            "paired with city names typical to U.S. addresses. Ensure the page clearly pertains to the specified facility."
        )
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the US colocation Tier III (or higher) facility requirements task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root combines children in parallel
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

    # Extract facility identification and categorized URLs
    extracted = await evaluator.extract(
        prompt=prompt_extract_facility_and_sources(),
        template_class=FacilityExtraction,
        extraction_name="facility_and_sources",
    )

    # Add ground truth policy/thresholds for transparency
    evaluator.add_ground_truth({
        "minimum_requirements": {
            "tier": "Uptime Institute Tier III or higher",
            "uptime": ">= 99.982%",
            "concurrent_maintainability": True,
            "power_redundancy": ">= N+1",
            "cooling_redundancy": ">= N+1",
            "fiber_single_mode": True,
            "fiber_strand_minimum": ">= 12 strands",
            "geography": "United States"
        }
    })

    # Create a critical wrapper node under root (since root itself is non‑critical in framework)
    requirements_node = evaluator.add_parallel(
        id="overall_requirements",
        desc="All specified technical infrastructure requirements must be met and supported by valid documentation",
        parent=root,
        critical=True
    )

    # Build verification subtrees
    await verify_tier_certification(evaluator, requirements_node, extracted)
    await verify_uptime_guarantee(evaluator, requirements_node, extracted)
    await verify_concurrent_maintainability(evaluator, requirements_node, extracted)
    await verify_redundancy_configuration(evaluator, requirements_node, extracted)
    await verify_fiber_connectivity(evaluator, requirements_node, extracted)
    await verify_geographic_location(evaluator, requirements_node, extracted)

    # Return structured result
    return evaluator.get_summary()