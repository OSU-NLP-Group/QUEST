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
TASK_ID = "us_telecom_infra_2024_2026"
TASK_DESCRIPTION = (
    "Identify a major telecommunications infrastructure project or deployment in the United States that was announced, "
    "initiated, or reached a significant milestone between January 1, 2024, and February 26, 2026. The project must meet all "
    "of the following criteria: (1) Located within or primarily serving the United States; (2) Involves telecommunications "
    "infrastructure technology (such as fiber optic networks, 5G deployment, data centers, cable landing stations, or similar "
    "infrastructure); (3) Has documented financial investment or funding commitment; (4) Includes specific technical "
    "specifications or capacity information; (5) Is associated with, operated by, or serves identifiable telecommunications "
    "service providers; (6) Serves a clearly identified geographic market, region, city, or state; (7) Has a clearly "
    "documented purpose (such as network expansion, modernization, capacity increase, or new service deployment); "
    "(8) Has a clearly stated operational status (planning phase, under construction, partially operational, or fully "
    "operational); (9) Has publicly accessible official documentation or announcements; (10) Includes quantifiable "
    "infrastructure metrics (such as number of locations, miles of fiber, geographic coverage area, capacity specifications, "
    "or similar measurable scale indicators); (11) Identifies the implementing organization, company, or consortium responsible "
    "for deployment. Provide the project name, a description of the project, and reference URLs that verify each criterion."
)

TIMEFRAME_START_STR = "2024-01-01"
TIMEFRAME_END_STR = "2026-02-26"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProjectExtraction(BaseModel):
    # Identification
    project_name: Optional[str] = None
    project_description: Optional[str] = None

    # Implementing and providers
    implementing_entity: Optional[str] = None
    service_providers: List[str] = Field(default_factory=list)

    # Geography and timeframe
    geographic_location: Optional[str] = None
    geographic_market: Optional[str] = None
    timeframe_statement: Optional[str] = None
    timeframe_date: Optional[str] = None  # Keep as string for flexibility

    # Technical infrastructure
    infrastructure_type: Optional[str] = None
    technology_specifications: Optional[str] = None  # capacity/technical specs, speeds, wavelengths, etc.
    infrastructure_scale: Optional[str] = None       # quantifiable metrics like miles of fiber, number of sites
    network_architecture: Optional[str] = None       # backbone, metro rings, POPs, redundancy, etc.

    # Operational characteristics
    project_purpose: Optional[str] = None
    operational_status: Optional[str] = None
    service_objectives: Optional[str] = None

    # Business and regulatory context
    investment_scale: Optional[str] = None           # funding amount or commitment wording
    regulatory_context: Optional[str] = None         # approvals, compliance, standards

    # Reference URLs by criterion category (one primary URL per category if available)
    project_documentation_url: Optional[str] = None
    scope_documentation_url: Optional[str] = None
    technical_documentation_url: Optional[str] = None
    operational_documentation_url: Optional[str] = None
    business_documentation_url: Optional[str] = None

    # Any other URLs cited in the answer
    additional_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_project() -> str:
    return f"""
    Extract structured information about a single telecommunications infrastructure project described in the answer.
    Only extract information explicitly present in the answer text. If a field is not mentioned, return null or an empty list.

    Required fields to extract (as strings unless noted):
    - project_name: The project name or commonly used identifier.
    - project_description: A concise textual description.
    - implementing_entity: The organization/company/consortium responsible for deployment.
    - service_providers: List of telecom service providers associated with, operating, or served by the project (if mentioned).
    - geographic_location: Country-level or general geographic scope (e.g., "United States").
    - geographic_market: Specific target market/region/city/state served (e.g., "Arizona", "Chicago metro", "Pacific Northwest").
    - timeframe_statement: The wording describing announcement/initiation/milestone timing (e.g., "announced in Jan 2025").
    - timeframe_date: A specific date or month/year explicitly mentioned for announcement/initiation/milestone (e.g., "2025-01-15", "March 2024").
    - infrastructure_type: The infrastructure involved (e.g., "fiber optic network", "5G deployment", "data center", "cable landing station").
    - technology_specifications: Specific technical specs or capacity details (e.g., "400G wavelengths", "100 Tbps", "multi-tenant Tier III").
    - infrastructure_scale: Quantifiable metrics (e.g., "250 miles of fiber", "20 cell sites", "covers 12 counties").
    - network_architecture: Architecture or design features (e.g., "ring topology", "redundant backbone", "POP to POP connectivity").
    - project_purpose: The documented purpose (e.g., "network expansion", "modernization", "capacity increase", "new service deployment").
    - operational_status: Current status (e.g., "planning", "under construction", "partially operational", "fully operational").
    - service_objectives: Intended service improvements, coverage goals, or customer benefits (if mentioned).
    - investment_scale: Documented funding or investment commitment (e.g., "$500M", "grants totaling $120M").
    - regulatory_context: References to regulatory approvals/compliance/standards (e.g., FCC, PUC dockets, NEPA) if mentioned.

    Reference URLs: Select the most relevant official or authoritative URL for each category if present.
    - project_documentation_url: A URL documenting the project overall (prefer official company/government press release or announcement).
    - scope_documentation_url: A URL documenting geographic scope and/or timing (dates).
    - technical_documentation_url: A URL documenting technical details/specifications/capacity/scale.
    - operational_documentation_url: A URL documenting purpose and operational status.
    - business_documentation_url: A URL documenting investment/funding and service provider associations.
    - additional_urls: Any other URLs cited in the answer (exclude duplicates; include full protocols).

    Important:
    - Choose URLs that best match the category descriptions when multiple are present.
    - Always return full URLs with protocol (http/https).
    - If a field or URL is not present in the answer, set it to null (or [] for lists).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(*sources: List[Optional[str] | List[str] | None]) -> List[str]:
    """Combine and deduplicate various URL inputs into a single list."""
    combined: List[str] = []
    for s in sources:
        if s is None:
            continue
        if isinstance(s, list):
            for url in s:
                if isinstance(url, str) and url.strip():
                    if url.strip() not in combined:
                        combined.append(url.strip())
        elif isinstance(s, str):
            if s.strip() and s.strip() not in combined:
                combined.append(s.strip())
    return combined


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_and_verify_project_identification(
    evaluator: Evaluator,
    parent: Any,
    info: ProjectExtraction,
) -> None:
    """
    Project identification section:
    - project_documentation_url (existence)
    - public_documentation
    - implementation_entity
    """
    node = evaluator.add_parallel(
        id="project_identification",
        desc="Project can be clearly identified with official documentation and implementing entity",
        parent=parent,
        critical=True,  # Parent marked critical; all children must be critical
    )

    # Existence of a documentation URL
    evaluator.add_custom_node(
        result=bool(info.project_documentation_url and info.project_documentation_url.strip()),
        id="project_documentation_url",
        desc="Valid reference URL provided that documents the project",
        parent=node,
        critical=True
    )

    # Publicly accessible official documentation or announcements
    public_doc_leaf = evaluator.add_leaf(
        id="public_documentation",
        desc="Project has publicly accessible official documentation or announcements",
        parent=node,
        critical=True
    )
    doc_sources = _combine_sources(info.project_documentation_url, info.additional_urls)
    await evaluator.verify(
        claim="The provided URL(s) include a publicly accessible official documentation or announcement of the project.",
        node=public_doc_leaf,
        sources=doc_sources,
        additional_instruction=(
            "Consider company/government press releases, official blogs/newsrooms, regulatory dockets, or authoritative "
            "project pages. The page should clearly document the project itself."
        ),
    )

    # Implementing entity verification
    impl_leaf = evaluator.add_leaf(
        id="implementation_entity",
        desc="Project identifies the implementing organization, company, or consortium responsible for deployment",
        parent=node,
        critical=True
    )
    impl_sources = _combine_sources(info.project_documentation_url, info.business_documentation_url, info.additional_urls)
    implementing_entity_text = info.implementing_entity or ""
    await evaluator.verify(
        claim=f"The implementing organization/company/consortium responsible for deployment is '{implementing_entity_text}'.",
        node=impl_leaf,
        sources=impl_sources,
        additional_instruction=(
            "Verify that the page explicitly names the implementing entity (operator, builder, lead company, or consortium) "
            "responsible for the project's deployment."
        ),
    )


async def build_and_verify_geographic_and_temporal_scope(
    evaluator: Evaluator,
    parent: Any,
    info: ProjectExtraction,
) -> None:
    """
    Geographic and temporal scope:
    - scope_documentation_url (existence)
    - geographic_location
    - geographic_market
    - timeframe
    """
    node = evaluator.add_parallel(
        id="geographic_and_temporal_scope",
        desc="Project has clearly defined geographic scope and falls within the required timeframe",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(info.scope_documentation_url and info.scope_documentation_url.strip()),
        id="scope_documentation_url",
        desc="Valid reference URL provided that documents the geographic and temporal scope",
        parent=node,
        critical=True
    )

    # Geographic location (US)
    geo_loc_leaf = evaluator.add_leaf(
        id="geographic_location",
        desc="Project is located within or primarily serves the United States",
        parent=node,
        critical=True
    )
    geo_sources = _combine_sources(info.scope_documentation_url, info.project_documentation_url, info.additional_urls)
    await evaluator.verify(
        claim="The project is located within or primarily serves the United States.",
        node=geo_loc_leaf,
        sources=geo_sources,
        additional_instruction=(
            "Confirm that the documentation indicates the project operates in the United States. If multi-country, "
            "the page should show that U.S. is a primary service area or location."
        ),
    )

    # Geographic market (region/city/state)
    geo_market_leaf = evaluator.add_leaf(
        id="geographic_market",
        desc="Project serves a clearly identified geographic market, region, city, or state",
        parent=node,
        critical=True
    )
    market_text = info.geographic_market or ""
    await evaluator.verify(
        claim=f"The project serves the identified geographic market/region/city/state: '{market_text}'.",
        node=geo_market_leaf,
        sources=geo_sources,
        additional_instruction=(
            "Verify that the documentation explicitly names the target market area (state, county, metro, city, region)."
        ),
    )

    # Timeframe (between Jan 1, 2024 and Feb 26, 2026)
    timeframe_leaf = evaluator.add_leaf(
        id="timeframe",
        desc=f"Project was announced, initiated, or reached a major milestone between {TIMEFRAME_START_STR} and {TIMEFRAME_END_STR}",
        parent=node,
        critical=True
    )
    timeframe_text = info.timeframe_statement or info.timeframe_date or ""
    await evaluator.verify(
        claim=(
            f"The project's announcement/initiation/milestone timing ('{timeframe_text}') falls between "
            f"{TIMEFRAME_START_STR} and {TIMEFRAME_END_STR}."
        ),
        node=timeframe_leaf,
        sources=geo_sources,
        additional_instruction=(
            "Look for explicit dates on the page (press release date, announcement date, groundbreaking, go-live, completion, "
            "contract award, construction start). Accept month/year formats if clearly within range."
        ),
    )


async def build_and_verify_technical_infrastructure(
    evaluator: Evaluator,
    parent: Any,
    info: ProjectExtraction,
) -> None:
    """
    Technical infrastructure:
    - technical_documentation_url (existence)
    - infrastructure_type
    - technology_specifications
    - infrastructure_scale
    - network_architecture (treated as critical in code due to framework constraints)
    """
    node = evaluator.add_parallel(
        id="technical_infrastructure",
        desc="Project involves specific telecommunications infrastructure with documented technical characteristics",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(info.technical_documentation_url and info.technical_documentation_url.strip()),
        id="technical_documentation_url",
        desc="Valid reference URL provided that documents the technical infrastructure details",
        parent=node,
        critical=True
    )

    tech_sources = _combine_sources(info.technical_documentation_url, info.project_documentation_url, info.additional_urls)

    # Infrastructure type
    infra_type_leaf = evaluator.add_leaf(
        id="infrastructure_type",
        desc="Project involves telecommunications infrastructure (fiber optic network, 5G deployment, data center, cable landing station, or similar technology infrastructure)",
        parent=node,
        critical=True
    )
    infra_type_text = info.infrastructure_type or ""
    await evaluator.verify(
        claim=f"This project involves '{infra_type_text}', which is telecommunications infrastructure.",
        node=infra_type_leaf,
        sources=tech_sources,
        additional_instruction=(
            "Confirm that the page describes telecom infrastructure such as fiber optic networks, 5G deployments, data centers, "
            "cable landing stations, or closely related infrastructure."
        ),
    )

    # Technology specifications / capacity
    tech_spec_leaf = evaluator.add_leaf(
        id="technology_specifications",
        desc="Project includes specific technical specifications or capacity information",
        parent=node,
        critical=True
    )
    tech_spec_text = info.technology_specifications or ""
    await evaluator.verify(
        claim=f"The project includes specific technical specifications or capacity details such as: '{tech_spec_text}'.",
        node=tech_spec_leaf,
        sources=tech_sources,
        additional_instruction=(
            "Look for capacities, speeds (e.g., Gbps/Tbps), wavelengths, tier levels, throughput, latency targets, fiber counts, etc."
        ),
    )

    # Infrastructure scale metrics
    infra_scale_leaf = evaluator.add_leaf(
        id="infrastructure_scale",
        desc="Project includes quantifiable infrastructure metrics (number of locations, miles of fiber, geographic coverage area, capacity specifications, or similar measurable scale indicators)",
        parent=node,
        critical=True
    )
    infra_scale_text = info.infrastructure_scale or ""
    await evaluator.verify(
        claim=f"The project includes quantifiable infrastructure metrics such as: '{infra_scale_text}'.",
        node=infra_scale_leaf,
        sources=tech_sources,
        additional_instruction=(
            "Verify that the page provides measurable scale indicators: miles of fiber, number of sites/nodes, coverage area, "
            "capacity figures, etc."
        ),
    )

    # Network architecture / connectivity features (treated as critical due to framework constraints)
    net_arch_leaf = evaluator.add_leaf(
        id="network_architecture",
        desc="Project describes network architecture, connectivity features, or infrastructure design elements",
        parent=node,
        critical=True  # Adjusted to critical because critical parent cannot have non-critical children
    )
    net_arch_text = info.network_architecture or ""
    await evaluator.verify(
        claim=f"The project documentation describes architecture/design/connectivity features such as: '{net_arch_text}'.",
        node=net_arch_leaf,
        sources=tech_sources,
        additional_instruction=(
            "Look for topology (ring/mesh/star), backbone/redundancy, POPs, interconnects, diverse paths, failover, routing domains."
        ),
    )


async def build_and_verify_operational_characteristics(
    evaluator: Evaluator,
    parent: Any,
    info: ProjectExtraction,
) -> None:
    """
    Operational characteristics:
    - operational_documentation_url (existence)
    - project_purpose
    - operational_status
    - service_objectives (treated as critical due to framework constraints)
    """
    node = evaluator.add_parallel(
        id="operational_characteristics",
        desc="Project has clearly defined operational purpose and status",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(info.operational_documentation_url and info.operational_documentation_url.strip()),
        id="operational_documentation_url",
        desc="Valid reference URL provided that documents the operational characteristics",
        parent=node,
        critical=True
    )

    op_sources = _combine_sources(info.operational_documentation_url, info.project_documentation_url, info.additional_urls)

    # Project purpose
    purpose_leaf = evaluator.add_leaf(
        id="project_purpose",
        desc="Project has a clearly documented purpose (network expansion, modernization, capacity increase, new service deployment, etc.)",
        parent=node,
        critical=True
    )
    purpose_text = info.project_purpose or ""
    await evaluator.verify(
        claim=f"The project has a clearly documented purpose, such as: '{purpose_text}'.",
        node=purpose_leaf,
        sources=op_sources,
        additional_instruction=(
            "Verify that the page describes goals like expansion, modernization, capacity increase, or new services."
        ),
    )

    # Operational status
    status_leaf = evaluator.add_leaf(
        id="operational_status",
        desc="Project has a clearly stated operational status (planning phase, under construction, partially operational, or fully operational)",
        parent=node,
        critical=True
    )
    status_text = info.operational_status or ""
    await evaluator.verify(
        claim=f"The project's operational status is: '{status_text}'.",
        node=status_leaf,
        sources=op_sources,
        additional_instruction=(
            "Confirm a clear status label such as planning, under construction, partial operations, or fully operational."
        ),
    )

    # Service objectives (treated as critical due to framework constraints)
    serv_obj_leaf = evaluator.add_leaf(
        id="service_objectives",
        desc="Project articulates intended service improvements, coverage goals, or customer benefits",
        parent=node,
        critical=True  # Adjusted to critical because critical parent cannot have non-critical children
    )
    serv_obj_text = info.service_objectives or ""
    await evaluator.verify(
        claim=f"The project articulates service objectives such as: '{serv_obj_text}'.",
        node=serv_obj_leaf,
        sources=op_sources,
        additional_instruction=(
            "Look for coverage goals, improved speeds/reliability, customer benefits, business service aims."
        ),
    )


async def build_and_verify_business_and_regulatory_context(
    evaluator: Evaluator,
    parent: Any,
    info: ProjectExtraction,
) -> None:
    """
    Business and regulatory context:
    - business_documentation_url (existence)
    - investment_scale
    - service_provider_association
    - regulatory_context (treated as critical due to framework constraints)
    """
    node = evaluator.add_parallel(
        id="business_and_regulatory_context",
        desc="Project has documented financial investment and service provider associations",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(info.business_documentation_url and info.business_documentation_url.strip()),
        id="business_documentation_url",
        desc="Valid reference URL provided that documents the business and financial context",
        parent=node,
        critical=True
    )

    biz_sources = _combine_sources(
        info.business_documentation_url,
        info.project_documentation_url,
        info.additional_urls
    )

    # Investment scale
    invest_leaf = evaluator.add_leaf(
        id="investment_scale",
        desc="Project involves documented financial investment or funding commitment",
        parent=node,
        critical=True
    )
    invest_text = info.investment_scale or ""
    await evaluator.verify(
        claim=f"The project has documented financial investment or funding commitment: '{invest_text}'.",
        node=invest_leaf,
        sources=biz_sources,
        additional_instruction=(
            "Confirm that the page documents investment amounts, budgets, grants, funding commitments, or similar financial scale."
        ),
    )

    # Service provider association
    provider_leaf = evaluator.add_leaf(
        id="service_provider_association",
        desc="Project is associated with, operated by, or serves identifiable telecommunications service providers",
        parent=node,
        critical=True
    )
    providers_text = ", ".join(info.service_providers) if info.service_providers else ""
    await evaluator.verify(
        claim=f"The project is associated with or serves the following telecommunications service providers: '{providers_text}'.",
        node=provider_leaf,
        sources=biz_sources,
        additional_instruction=(
            "Verify that the page mentions carriers/ISPs/telecom operators or enterprise providers linked to this project."
        ),
    )

    # Regulatory context (treated as critical due to framework constraints)
    reg_leaf = evaluator.add_leaf(
        id="regulatory_context",
        desc="Project documentation references applicable regulatory requirements, approvals, or compliance standards",
        parent=node,
        critical=True  # Adjusted to critical because critical parent cannot have non-critical children
    )
    reg_text = info.regulatory_context or ""
    await evaluator.verify(
        claim=f"The documentation references regulatory approvals/compliance/standards such as: '{reg_text}'.",
        node=reg_leaf,
        sources=biz_sources,
        additional_instruction=(
            "Look for references to FCC/PUC dockets, environmental reviews, permits, standards compliance, or similar regulatory context."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the U.S. telecommunications infrastructure project criteria (2024-2026).
    """
    # Initialize evaluator with a critical root (as specified by rubric). Note:
    # In the framework, critical parents require critical children. Some rubric items
    # marked non-critical are treated as critical to satisfy framework constraints.
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identifies a major telecommunications infrastructure project or deployment from 2024-2026 meeting all specified criteria",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Mark root as critical by adding a critical wrapper node under root
    # to satisfy "critical root" intent while allowing tree construction
    critical_root = evaluator.add_parallel(
        id="critical_root",
        desc="Overall verification of a U.S. telecommunications infrastructure project meeting criteria (2024-2026)",
        parent=root,
        critical=True
    )

    # Extract structured project info from the answer
    project_info: ProjectExtraction = await evaluator.extract(
        prompt=prompt_extract_project(),
        template_class=ProjectExtraction,
        extraction_name="project_extraction",
    )

    # Add a small custom info block to aid debugging/traceability
    evaluator.add_custom_info(
        info={
            "timeframe_window": {"start": TIMEFRAME_START_STR, "end": TIMEFRAME_END_STR},
            "extracted_project_name": project_info.project_name,
            "primary_urls": {
                "project": project_info.project_documentation_url,
                "scope": project_info.scope_documentation_url,
                "technical": project_info.technical_documentation_url,
                "operational": project_info.operational_documentation_url,
                "business": project_info.business_documentation_url,
            },
            "additional_urls_count": len(project_info.additional_urls or [])
        },
        info_type="context",
        info_name="evaluation_context"
    )

    # Build verification subsections
    await build_and_verify_project_identification(evaluator, critical_root, project_info)
    await build_and_verify_geographic_and_temporal_scope(evaluator, critical_root, project_info)
    await build_and_verify_technical_infrastructure(evaluator, critical_root, project_info)
    await build_and_verify_operational_characteristics(evaluator, critical_root, project_info)
    await build_and_verify_business_and_regulatory_context(evaluator, critical_root, project_info)

    # Return structured summary
    return evaluator.get_summary()