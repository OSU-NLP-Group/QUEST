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
TASK_ID = "us_semiconductor_facilities_ai_2026"
TASK_DESCRIPTION = """
Identify three semiconductor manufacturing facilities in the United States that meet ALL of the following criteria:

1. Geographic Proximity to AI Conference: The facility must be located within 500 miles driving distance from a venue hosting a major AI or machine learning conference in 2026 (such as CVPR in Denver, Colorado; The AI Conference in San Francisco, California; or Ai4 in Las Vegas, Nevada).

2. 5G Network Coverage: The facility's city or metropolitan area must have 5G network coverage from at least two of the following major carriers: T-Mobile, Verizon, or AT&T.

3. ML Framework Ecosystem Involvement: The facility must be operated by a company that either directly maintains or is the primary contributor to TensorFlow (licensed under Apache 2.0) or PyTorch (licensed under BSD-3-Clause), OR manufactures semiconductors that are specifically optimized for and widely used in AI/ML workloads running these frameworks.

4. Operational Status: The facility must be operational or in high-volume production as of 2026.

5. Production Scale: The facility must be a major advanced semiconductor manufacturing facility capable of producing modern process nodes.

For each facility, provide:
- Facility name and operating company
- Specific location (city, state, with address or campus name)
- The nearest AI/ML conference in 2026 and its venue
- At least two specific carriers (from T-Mobile, Verizon, AT&T) providing 5G coverage in the area
- The relevant ML framework (TensorFlow or PyTorch) and its license type (Apache 2.0 or BSD-3-Clause)
- How the operating company relates to the framework (as maintainer/contributor or as hardware manufacturer optimized for the framework)
- Reference URLs for: facility location, 5G coverage information, framework/license information, and operational status
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FacilityLocation(BaseModel):
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)


class ConferenceInfo(BaseModel):
    name: Optional[str] = None
    venue: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    url: Optional[str] = None


class CoverageInfo(BaseModel):
    carriers: List[str] = Field(default_factory=list)  # Expect canonical: "T-Mobile", "Verizon", "AT&T"
    coverage_urls: List[str] = Field(default_factory=list)


class FrameworkInfo(BaseModel):
    framework: Optional[str] = None  # "TensorFlow" or "PyTorch"
    license: Optional[str] = None    # "Apache 2.0" or "BSD-3-Clause"
    relationship: Optional[str] = None  # maintainer/contributor or hardware optimized widely used
    framework_urls: List[str] = Field(default_factory=list)


class OperationInfo(BaseModel):
    operational_status: Optional[str] = None  # e.g., "operational", "high-volume production"
    status_urls: List[str] = Field(default_factory=list)


class ProductionInfo(BaseModel):
    modern_node_capability: Optional[str] = None  # e.g., "5nm", "7nm", "3nm", "advanced nodes"
    production_urls: List[str] = Field(default_factory=list)


class Facility(BaseModel):
    facility_name: Optional[str] = None
    company_name: Optional[str] = None
    facility_urls: List[str] = Field(default_factory=list)

    location: FacilityLocation = Field(default_factory=FacilityLocation)
    conference: ConferenceInfo = Field(default_factory=ConferenceInfo)
    coverage: CoverageInfo = Field(default_factory=CoverageInfo)
    framework_info: FrameworkInfo = Field(default_factory=FrameworkInfo)
    operation: OperationInfo = Field(default_factory=OperationInfo)
    production: ProductionInfo = Field(default_factory=ProductionInfo)


class FacilitiesExtraction(BaseModel):
    facilities: List[Facility] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_facilities() -> str:
    return """
    Extract up to three (3) semiconductor manufacturing facilities mentioned in the answer that are located in the United States.
    For each facility, return an object with the following fields. If any field is missing in the answer, set it to null (or [] for arrays).

    Fields to extract for each facility:
    1) facility_name: The official name of the semiconductor manufacturing facility (e.g., "Fab 8", "Chandler Fab", "D1X", etc.).
    2) company_name: The operating company (e.g., Intel, TSMC, Samsung, GlobalFoundries, Micron, Texas Instruments, etc.).
    3) facility_urls: URLs that directly reference the facility (official pages, company sites, press releases, Wikipedia, or reliable news sources). Array.
    
    4) location: { 
         address: A specific street address or campus/fab name if provided (e.g., "Ocotillo Campus", "Fab 8"),
         city: City name,
         state: Two-letter state or full state name,
         location_urls: URLs that support the facility location/address/campus. Array.
       }

    5) conference: {
         name: The nearest major AI/ML conference in 2026 (e.g., CVPR, ICML, NeurIPS, ICLR, The AI Conference, Ai4),
         venue: Venue name (e.g., "Colorado Convention Center"),
         city: City of the venue,
         state: State of the venue,
         url: A URL that states the 2026 conference location/venue details (official conference page, reputable event page).
       }

    6) coverage: {
         carriers: An array of carriers providing 5G in the facility area. Only use exact canonical names from this set: "T-Mobile", "Verizon", "AT&T".
                   If the answer mentions variants (e.g., "att", "AT&T Wireless"), normalize to "AT&T".
         coverage_urls: URLs supporting 5G coverage (official carrier coverage maps or reputable third-party coverage aggregators). Array.
       }

    7) framework_info: {
         framework: "TensorFlow" or "PyTorch" (choose the one relevant per the answer),
         license: "Apache 2.0" for TensorFlow OR "BSD-3-Clause" for PyTorch (normalize spelling to exactly one of these),
         relationship: A short sentence of how the operating company relates to the framework (e.g., "primary maintainer of TensorFlow"; or "manufactures AI chips optimized for PyTorch/TensorFlow and widely used"),
         framework_urls: URLs that support the framework/license/relationship claims (official docs, GitHub repo/license page, vendor docs). Array.
       }

    8) operation: {
         operational_status: A short phrase indicating status as of 2026 (e.g., "operational", "in high-volume production"),
         status_urls: URLs supporting the operational status as of 2026 (company announcements, reliable news, or reports). Array.
       }

    9) production: {
         modern_node_capability: A short phrase stating capability for modern process nodes (e.g., "5nm", "7nm", "advanced logic nodes"),
         production_urls: URLs supporting production scale/capability (company pages, reputable news/reports). Array.
       }

    Requirements:
    - Only include United States facilities.
    - Normalize carrier names strictly to: "T-Mobile", "Verizon", "AT&T".
    - Normalize license strictly to "Apache 2.0" (TensorFlow) or "BSD-3-Clause" (PyTorch).
    - If more than 3 facilities are present in the answer, still extract them all; the evaluator will use only the first 3.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
CANONICAL_CARRIERS = {"t-mobile": "T-Mobile", "tmobile": "T-Mobile", "t mobile": "T-Mobile",
                      "verizon": "Verizon",
                      "at&t": "AT&T", "att": "AT&T", "at & t": "AT&T"}


def canonicalize_carrier(name: str) -> Optional[str]:
    if not name:
        return None
    key = name.strip().lower()
    return CANONICAL_CARRIERS.get(key, None)


def normalize_carriers(carriers: List[str]) -> List[str]:
    normed = []
    for c in carriers or []:
        canon = canonicalize_carrier(c)
        if canon and canon not in normed:
            normed.append(canon)
    return normed


def dedup_urls(*lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        for u in lst or []:
            if isinstance(u, str) and u.strip():
                if u not in seen:
                    seen.add(u)
                    out.append(u)
    return out


def safe(s: Optional[str]) -> str:
    return s or ""


# --------------------------------------------------------------------------- #
# Verification logic per facility                                             #
# --------------------------------------------------------------------------- #
async def verify_facility(evaluator: Evaluator, parent_node, facility: Facility, index: int) -> None:
    fidx = index + 1
    fid = f"f{fidx}"

    # Facility node (non-critical; overall facility success depends on internal critical nodes)
    facility_node = evaluator.add_parallel(
        id=f"facility_{fidx}",
        desc=f"{['First','Second','Third'][index]} facility meeting all criteria",
        parent=parent_node,
        critical=False
    )

    # ---------------- Basic identification and location --------------------
    basic_node = evaluator.add_parallel(
        id=f"{fid}_basic_information",
        desc=f"Basic identification and location information for {['first','second','third'][index]} facility",
        parent=facility_node,
        critical=True
    )

    ident_node = evaluator.add_parallel(
        id=f"{fid}_identification",
        desc="Facility name and operating company",
        parent=basic_node,
        critical=True
    )

    # Sources for identification/company checks
    id_sources = dedup_urls(facility.facility_urls, facility.location.location_urls, facility.production.production_urls)

    # Facility name leaf
    leaf = evaluator.add_leaf(
        id=f"{fid}_facility_name",
        desc="Correct facility name provided",
        parent=ident_node,
        critical=True
    )
    claim = f"The official semiconductor manufacturing facility name referenced in the answer is '{safe(facility.facility_name)}'. If the name is missing or not provided, this claim should be judged incorrect."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=id_sources if id_sources else None,
        additional_instruction="Check the cited source(s) for the explicit facility name or unambiguous naming (e.g., Fab ID or campus/fab name). If the answer omits the name, mark incorrect."
    )

    # Company name leaf
    leaf = evaluator.add_leaf(
        id=f"{fid}_company_name",
        desc="Correct operating company identified",
        parent=ident_node,
        critical=True
    )
    claim = f"The operating company for the facility is '{safe(facility.company_name)}'. If the company name is missing or not provided, judge incorrect."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=id_sources if id_sources else None,
        additional_instruction="Confirm the page ties the facility to the stated company (owner/operator). Allow known brand/subsidiary/parent equivalences."
    )

    # Location details
    loc_node = evaluator.add_parallel(
        id=f"{fid}_location",
        desc="Physical location details",
        parent=basic_node,
        critical=True
    )

    # Address/campus
    leaf = evaluator.add_leaf(
        id=f"{fid}_address",
        desc="Specific physical address or campus location provided",
        parent=loc_node,
        critical=True
    )
    claim = f"The facility's specific physical address or campus location is '{safe(facility.location.address)}'. If only a campus/fab name is given (e.g., 'Ocotillo Campus'), that is acceptable. If missing, judge incorrect."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=facility.location.location_urls if facility.location.location_urls else id_sources if id_sources else None,
        additional_instruction="Accept a campus/fab name if that is how the site is officially identified. Prefer explicit address when available."
    )

    # City/state
    leaf = evaluator.add_leaf(
        id=f"{fid}_city_state_info",
        desc="City and state correctly identified",
        parent=loc_node,
        critical=True
    )
    claim = f"The facility is located in {safe(facility.location.city)}, {safe(facility.location.state)}, United States."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=facility.location.location_urls if facility.location.location_urls else id_sources if id_sources else None,
        additional_instruction="Confirm city and state on the cited page. Implicit US location is required."
    )

    # Location reference (existence of URL evidence)
    evaluator.add_custom_node(
        result=bool(facility.location.location_urls and len(facility.location.location_urls) > 0),
        id=f"{fid}_location_reference",
        desc="URL reference for facility location",
        parent=loc_node,
        critical=True
    )

    # ---------------- Technical and operational requirements ---------------
    tech_node = evaluator.add_parallel(
        id=f"{fid}_technical_requirements",
        desc=f"Technical and operational requirements for {['first','second','third'][index]} facility",
        parent=facility_node,
        critical=True
    )

    # Conference proximity
    conf_node = evaluator.add_parallel(
        id=f"{fid}_conference_proximity",
        desc="Proximity to AI/ML conference verification",
        parent=tech_node,
        critical=True
    )

    # Conference identified
    leaf = evaluator.add_leaf(
        id=f"{fid}_conference_identified",
        desc="Specific AI/ML conference in 2026 identified",
        parent=conf_node,
        critical=True
    )
    conf_claim = f"There is a major AI/ML conference in 2026 named '{safe(facility.conference.name)}' located in {safe(facility.conference.city)}, {safe(facility.conference.state)}."
    await evaluator.verify(
        claim=conf_claim,
        node=leaf,
        sources=facility.conference.url if facility.conference.url else None,
        additional_instruction="Treat conferences such as CVPR, ICML, ICLR, NeurIPS, The AI Conference, or Ai4 as 'major'. The URL should indicate 2026 details if available."
    )

    # Venue identified
    leaf = evaluator.add_leaf(
        id=f"{fid}_venue_identified",
        desc="Conference venue name provided",
        parent=conf_node,
        critical=True
    )
    venue_claim = f"The venue for {safe(facility.conference.name)} is '{safe(facility.conference.venue)}' in {safe(facility.conference.city)}, {safe(facility.conference.state)}."
    await evaluator.verify(
        claim=venue_claim,
        node=leaf,
        sources=facility.conference.url if facility.conference.url else None,
        additional_instruction="Verify that the page states the venue name (e.g., convention center or hotel) for the 2026 edition."
    )

    # Within 500 miles (driving distance) — compute via reasoning (no single-page evidence typically states distance)
    leaf = evaluator.add_leaf(
        id=f"{fid}_within_500_miles",
        desc="Facility is within 500 miles driving distance of the conference venue",
        parent=conf_node,
        critical=True
    )
    dist_claim = (
        f"The driving distance from {safe(facility.location.city)}, {safe(facility.location.state)} to "
        f"{safe(facility.conference.venue)} in {safe(facility.conference.city)}, {safe(facility.conference.state)} "
        f"is within 500 miles."
    )
    await evaluator.verify(
        claim=dist_claim,
        node=leaf,
        sources=None,
        additional_instruction=(
            "Use reasonable geographic reasoning with typical US city-to-city driving distances. "
            "If either city/state is missing or the venues are obviously far apart (e.g., coast-to-coast), judge incorrect. "
            "Do not require an exact route; an informed estimate is acceptable."
        )
    )

    # 5G coverage
    cov_node = evaluator.add_parallel(
        id=f"{fid}_5g_coverage",
        desc="5G network coverage verification",
        parent=tech_node,
        critical=True
    )

    # Normalize carriers and compute minimum check
    carriers_norm = normalize_carriers(facility.coverage.carriers)
    min_two = len(carriers_norm) >= 2

    evaluator.add_custom_node(
        result=min_two,
        id=f"{fid}_carrier_minimum",
        desc="At least two major carriers (T-Mobile, Verizon, or AT&T) provide 5G coverage in facility area",
        parent=cov_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id=f"{fid}_carrier_names",
        desc="Specific carrier names providing coverage listed",
        parent=cov_node,
        critical=True
    )
    carriers_str = ", ".join(carriers_norm) if carriers_norm else ""
    cov_claim = f"In {safe(facility.location.city)}, {safe(facility.location.state)}, 5G coverage is available from the following carriers: {carriers_str} (subset of T-Mobile, Verizon, AT&T)."
    await evaluator.verify(
        claim=cov_claim,
        node=leaf,
        sources=facility.coverage.coverage_urls if facility.coverage.coverage_urls else None,
        additional_instruction="Confirm on official carrier maps or reliable coverage aggregators that the listed carriers provide 5G in the stated area."
    )

    evaluator.add_custom_node(
        result=bool(facility.coverage.coverage_urls and len(facility.coverage.coverage_urls) > 0),
        id=f"{fid}_coverage_reference",
        desc="URL reference for 5G coverage information",
        parent=cov_node,
        critical=True
    )

    # ML framework involvement
    ml_node = evaluator.add_parallel(
        id=f"{fid}_ml_framework",
        desc="ML framework ecosystem involvement",
        parent=tech_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=facility.framework_info.framework in {"TensorFlow", "PyTorch"},
        id=f"{fid}_framework_identified",
        desc="TensorFlow or PyTorch framework connection identified",
        parent=ml_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id=f"{fid}_license_type",
        desc="Correct license type (Apache 2.0 for TensorFlow, BSD-3-Clause for PyTorch) specified",
        parent=ml_node,
        critical=True
    )
    lic_claim = f"The {safe(facility.framework_info.framework)} framework is licensed under {safe(facility.framework_info.license)}."
    await evaluator.verify(
        claim=lic_claim,
        node=leaf,
        sources=facility.framework_info.framework_urls if facility.framework_info.framework_urls else None,
        additional_instruction="Accept the canonical licenses: TensorFlow -> Apache License 2.0; PyTorch -> BSD-3-Clause. Allow exact naming variations like 'Apache License 2.0'."
    )

    leaf = evaluator.add_leaf(
        id=f"{fid}_relationship_type",
        desc="Company relationship to framework specified (as maintainer/contributor or as hardware manufacturer optimized for the framework)",
        parent=ml_node,
        critical=True
    )
    rel_sources = dedup_urls(facility.framework_info.framework_urls, facility.production.production_urls, facility.facility_urls)
    rel_claim = (
        f"The operating company {safe(facility.company_name)} either is a primary maintainer/contributor of {safe(facility.framework_info.framework)}, "
        f"or manufactures semiconductors optimized for and widely used in AI/ML workloads running {safe(facility.framework_info.framework)}. "
        f"The answer describes the relationship as: '{safe(facility.framework_info.relationship)}'."
    )
    await evaluator.verify(
        claim=rel_claim,
        node=leaf,
        sources=rel_sources if rel_sources else None,
        additional_instruction="Use vendor docs, framework org pages, or credible reports to confirm either (a) maintainer/contributor role or (b) hardware optimized and widely used by the framework."
    )

    evaluator.add_custom_node(
        result=bool(facility.framework_info.framework_urls and len(facility.framework_info.framework_urls) > 0),
        id=f"{fid}_framework_reference",
        desc="URL reference for framework and license information",
        parent=ml_node,
        critical=True
    )

    # Production scale
    prod_node = evaluator.add_parallel(
        id=f"{fid}_production_scale",
        desc="Production scale verification",
        parent=tech_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id=f"{fid}_is_major_facility",
        desc="Facility is described as a major advanced semiconductor manufacturing facility capable of producing modern process nodes (not just a research lab or small-scale operation)",
        parent=prod_node,
        critical=True
    )
    prod_sources = dedup_urls(facility.production.production_urls, facility.facility_urls)
    prod_claim = (
        f"The facility is a major advanced semiconductor manufacturing site capable of producing modern process nodes "
        f"(e.g., advanced logic like {safe(facility.production.modern_node_capability)}), not merely a small research line."
    )
    await evaluator.verify(
        claim=prod_claim,
        node=leaf,
        sources=prod_sources if prod_sources else None,
        additional_instruction="Look for mentions of advanced nodes (e.g., 3nm/5nm/7nm/advanced nodes) or high-volume/large-scale manufacturing capacity."
    )

    # Operational status
    ops_node = evaluator.add_parallel(
        id=f"{fid}_operational_status",
        desc="Facility operational status",
        parent=tech_node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id=f"{fid}_is_operational_2026",
        desc="Facility is operational or in high-volume production as of 2026",
        parent=ops_node,
        critical=True
    )
    ops_sources = dedup_urls(facility.operation.status_urls, facility.facility_urls)
    ops_claim = f"As of 2026, the facility is '{safe(facility.operation.operational_status)}' (operational or in high-volume production)."
    await evaluator.verify(
        claim=ops_claim,
        node=leaf,
        sources=ops_sources if ops_sources else None,
        additional_instruction="Confirm via 2025–2026 company press releases, news, or official statements. If status is unclear or outdated, judge incorrect."
    )

    evaluator.add_custom_node(
        result=bool(facility.operation.status_urls and len(facility.operation.status_urls) > 0),
        id=f"{fid}_status_reference",
        desc="URL reference for operational status information",
        parent=ops_node,
        critical=True
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
    # Initialize evaluator (root is non-critical by default; we aggregate in parallel)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify three semiconductor manufacturing facilities in the United States that meet all specified criteria",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract facilities
    extraction = await evaluator.extract(
        prompt=prompt_extract_facilities(),
        template_class=FacilitiesExtraction,
        extraction_name="facilities_extraction",
    )

    # Keep the first 3 facilities; pad with empty if fewer
    facilities: List[Facility] = list(extraction.facilities[:3])
    while len(facilities) < 3:
        facilities.append(Facility())

    # Record a small custom info snapshot
    evaluator.add_custom_info(
        info={
            "num_facilities_in_answer": len(extraction.facilities),
            "used_facilities": min(3, len(extraction.facilities))
        },
        info_type="extraction_stats"
    )

    # Build and verify subtrees for each facility
    for i in range(3):
        await verify_facility(evaluator, root, facilities[i], i)

    # Return summary
    return evaluator.get_summary()