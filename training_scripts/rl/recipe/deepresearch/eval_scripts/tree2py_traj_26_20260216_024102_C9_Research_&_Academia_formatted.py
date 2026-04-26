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
TASK_ID = "mit_erc_lead"
TASK_DESCRIPTION = """
Evaluate whether the Massachusetts Institute of Technology (MIT) meets all comprehensive institutional requirements and possesses the necessary research infrastructure, administrative support systems, faculty expertise, and collaborative research capacity to qualify as a lead institution for a federally-funded, multi-institutional Engineering Research Center (ERC) focused on artificial intelligence and robotics research.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class RankingsExtraction(BaseModel):
    qs_rank_or_tier: Optional[str] = None
    qs_url: Optional[str] = None
    the_rank_or_tier: Optional[str] = None
    the_url: Optional[str] = None


class NamedEntityWithSources(BaseModel):
    name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ServicesWithSources(BaseModel):
    services: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class MITERCExtraction(BaseModel):
    rankings: Optional[RankingsExtraction] = None

    ai_robotics_labs: List[NamedEntityWithSources] = Field(default_factory=list)

    hpc_facility: Optional[NamedEntityWithSources] = None
    core_facility: Optional[NamedEntityWithSources] = None

    irb: Optional[NamedEntityWithSources] = None
    sponsored_research_office: Optional[NamedEntityWithSources] = None
    cs_ee_unit: Optional[NamedEntityWithSources] = None
    research_compliance_office: Optional[NamedEntityWithSources] = None

    data_management_support: Optional[ServicesWithSources] = None
    research_admin_support: Optional[ServicesWithSources] = None

    collaboration_programs: List[NamedEntityWithSources] = Field(default_factory=list)

    faculty_productivity_example: Optional[NamedEntityWithSources] = None

    library_support: Optional[ServicesWithSources] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_mit_erc_evidence() -> str:
    return """
    Extract from the answer all structured evidence that MIT qualifies as a lead institution for a federally-funded, multi-institutional Engineering Research Center (ERC) focused on AI and robotics. Return a single JSON object containing the following fields. Do NOT invent information; only extract what is explicitly present in the answer.

    1) rankings:
       - qs_rank_or_tier: MIT's position or tier in QS World University Rankings 2026 as stated in the answer (e.g., "#1", "Top 5", "top tier"). If not provided, set to null.
       - qs_url: The URL cited for the QS 2026 ranking. If absent, set to null.
       - the_rank_or_tier: MIT's position or tier in Times Higher Education (THE) World University Rankings 2026 as stated in the answer. If not provided, set to null.
       - the_url: The URL cited for the THE 2026 ranking. If absent, set to null.

    2) ai_robotics_labs: Array of labs/units clearly focused on AI/robotics. Each item:
       - name: Lab/unit name (e.g., "CSAIL", "MIT Robotics Institute"). If missing, set to null.
       - urls: All URLs cited for that lab/unit. If none, use an empty array.

    3) hpc_facility: An HPC facility or program accessible to researchers for AI/ML computation:
       - name
       - urls (array)

    4) core_facility: A core/shared research facility program:
       - name
       - urls (array)

    5) irb: The human-subjects review body:
       - name
       - urls (array)

    6) sponsored_research_office: The sponsored research/grants/contracts administration office:
       - name
       - urls (array)

    7) cs_ee_unit: A department or unit covering computer science/electrical engineering:
       - name
       - urls (array)

    8) research_compliance_office: Office/role overseeing research compliance:
       - name
       - urls (array)

    9) data_management_support: Data management support services:
       - services: array of service names mentioned (e.g., "data management planning", "DMP support").
       - urls: array of URLs cited for these services.

    10) research_admin_support: Research administration support services:
       - services: array of service names mentioned.
       - urls: array of URLs cited.

    11) collaboration_programs: Array of centers/programs showing multi-institution collaboration capacity:
       - name
       - urls (array)

    12) faculty_productivity_example: One concrete AI/robotics faculty/group example with research output evidence:
       - name (faculty/lab/group)
       - urls (array)

    13) library_support: Library/publication support:
       - services: array of services mentioned (e.g., "journal subscriptions", "publication support").
       - urls: array of URLs cited.

    If any field or subfield is missing in the answer, use null or an empty array as appropriate. Do not infer or create URLs not present in the answer. Preserve the URLs exactly as they appear (plain or markdown).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_entity_with_sources(entity: Optional[NamedEntityWithSources]) -> bool:
    return bool(entity and entity.name and entity.name.strip() and entity.urls and len(entity.urls) > 0)


def _first_entity_with_sources(items: List[NamedEntityWithSources]) -> Optional[NamedEntityWithSources]:
    for it in items:
        if _has_entity_with_sources(it):
            return it
    return None


def _has_services_with_sources(svc: Optional[ServicesWithSources]) -> bool:
    return bool(svc and svc.services and len(svc.services) > 0 and svc.urls and len(svc.urls) > 0)


def _safe_urls(urls: Optional[List[str] | str]) -> List[str]:
    if urls is None:
        return []
    if isinstance(urls, list):
        return urls
    return [urls]


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def _verify_global_ranking(evaluator: Evaluator, parent_node, ext: MITERCExtraction) -> None:
    node = evaluator.add_parallel(
        id="Global_Ranking_QS_AND_THE_2026",
        desc="MIT is ranked among top universities globally in BOTH QS 2026 and THE 2026 with ranks/tiers and sources.",
        parent=parent_node,
        critical=True
    )

    has_qs = bool(ext.rankings and ext.rankings.qs_rank_or_tier and ext.rankings.qs_url)
    has_the = bool(ext.rankings and ext.rankings.the_rank_or_tier and ext.rankings.the_url)
    evaluator.add_custom_node(
        result=has_qs and has_the,
        id="Global_Ranking_QS_AND_THE_2026_exists",
        desc="Answer provides QS 2026 and THE 2026 ranks/tiers for MIT with sources.",
        parent=node,
        critical=True
    )

    qs_leaf = evaluator.add_leaf(
        id="Global_Ranking_QS_AND_THE_2026_QS_supported",
        desc="QS 2026 rank/tier for MIT matches the cited QS source.",
        parent=node,
        critical=True
    )
    qs_claim = f"MIT's QS World University Rankings 2026 position/tier is '{ext.rankings.qs_rank_or_tier or ''}'."
    await evaluator.verify(
        claim=qs_claim,
        node=qs_leaf,
        sources=ext.rankings.qs_url if ext.rankings else None,
        additional_instruction=(
            "Verify the QS 2026 ranking/tier for MIT as stated. Accept reasonable phrasing variants "
            "(e.g., '#1', 'Top 5', 'top tier'). Ensure the year is 2026 or the clearly corresponding edition."
        )
    )

    the_leaf = evaluator.add_leaf(
        id="Global_Ranking_QS_AND_THE_2026_THE_supported",
        desc="THE 2026 rank/tier for MIT matches the cited THE source.",
        parent=node,
        critical=True
    )
    the_claim = f"MIT's Times Higher Education World University Rankings 2026 position/tier is '{ext.rankings.the_rank_or_tier or ''}'."
    await evaluator.verify(
        claim=the_claim,
        node=the_leaf,
        sources=ext.rankings.the_url if ext.rankings else None,
        additional_instruction=(
            "Verify the THE 2026 ranking/tier for MIT as stated. Accept reasonable phrasing variants. "
            "Ensure the year is 2026 or the corresponding edition explicitly indicated."
        )
    )


async def _verify_ai_robotics_labs(evaluator: Evaluator, parent_node, ext: MITERCExtraction) -> None:
    node = evaluator.add_parallel(
        id="Dedicated_AI_And_Robotics_Labs",
        desc="MIT operates dedicated AI/robotics labs with at least one identified and cited.",
        parent=parent_node,
        critical=True
    )

    first_lab = _first_entity_with_sources(ext.ai_robotics_labs)
    evaluator.add_custom_node(
        result=_has_entity_with_sources(first_lab),
        id="Dedicated_AI_And_Robotics_Labs_exists",
        desc="At least one AI or robotics lab/unit is identified with source(s).",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Dedicated_AI_And_Robotics_Labs_supported",
        desc="Cited lab/unit page supports that it is focused on AI and/or robotics and operated by MIT.",
        parent=node,
        critical=True
    )
    lab_name = first_lab.name if first_lab else ""
    claim = f"MIT operates a dedicated research laboratory or unit focused on artificial intelligence or robotics named '{lab_name}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=first_lab.urls if first_lab else [],
        additional_instruction=(
            "Confirm the page is an official MIT lab/unit and explicitly indicates focus on AI and/or robotics. "
            "Name variants or abbreviations are acceptable."
        )
    )


async def _verify_hpc_access(evaluator: Evaluator, parent_node, ext: MITERCExtraction) -> None:
    node = evaluator.add_parallel(
        id="HPC_Access_For_AI_ML",
        desc="MIT provides HPC facilities accessible to researchers for computational AI/ML.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_entity_with_sources(ext.hpc_facility),
        id="HPC_Access_For_AI_ML_exists",
        desc="An HPC facility/program is identified with source(s).",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="HPC_Access_For_AI_ML_supported",
        desc="Cited HPC page supports accessible high-performance computing for research (suitable for AI/ML).",
        parent=node,
        critical=True
    )
    name = ext.hpc_facility.name if ext.hpc_facility else ""
    claim = f"MIT provides high-performance computing resources accessible to researchers for computational AI/ML via '{name}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=ext.hpc_facility.urls if ext.hpc_facility else [],
        additional_instruction=(
            "Verify the page states HPC capabilities (e.g., clusters, GPUs) available to MIT researchers; "
            "access may require affiliation or application."
        )
    )


async def _verify_core_shared_facilities(evaluator: Evaluator, parent_node, ext: MITERCExtraction) -> None:
    node = evaluator.add_parallel(
        id="Core_Shared_Research_Facilities",
        desc="MIT operates core/shared research facilities with shared access to specialized equipment/tech.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_entity_with_sources(ext.core_facility),
        id="Core_Shared_Research_Facilities_exists",
        desc="A core/shared facility is identified with source(s).",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Core_Shared_Research_Facilities_supported",
        desc="Cited page supports shared/core facility access to specialized equipment/technologies.",
        parent=node,
        critical=True
    )
    name = ext.core_facility.name if ext.core_facility else ""
    claim = f"MIT operates core/shared research facilities providing shared access to specialized equipment/technologies, for example '{name}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=ext.core_facility.urls if ext.core_facility else [],
        additional_instruction="Confirm the program offers shared/core facility access to specialized instruments/technologies."
    )


async def _verify_irb(evaluator: Evaluator, parent_node, ext: MITERCExtraction) -> None:
    node = evaluator.add_parallel(
        id="Active_IRB",
        desc="MIT has an active IRB or equivalent human-subjects ethics review body.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_entity_with_sources(ext.irb),
        id="Active_IRB_exists",
        desc="IRB or equivalent body is identified with source(s).",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Active_IRB_supported",
        desc="Cited page supports the existence of an active IRB or equivalent at MIT.",
        parent=node,
        critical=True
    )
    name = ext.irb.name if ext.irb else ""
    claim = f"MIT maintains an active Institutional Review Board (IRB) or equivalent human-subjects review body called '{name}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=ext.irb.urls if ext.irb else [],
        additional_instruction="Confirm the body is an IRB or equivalent and is active for MIT research oversight."
    )


async def _verify_sponsored_research_office(evaluator: Evaluator, parent_node, ext: MITERCExtraction) -> None:
    node = evaluator.add_parallel(
        id="Sponsored_Research_Administration_Office",
        desc="MIT has an office that manages sponsored research programs, grants, and contracts.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_entity_with_sources(ext.sponsored_research_office),
        id="Sponsored_Research_Administration_Office_exists",
        desc="Sponsored research administration office is identified with source(s).",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Sponsored_Research_Administration_Office_supported",
        desc="Cited page supports that the office manages sponsored research/grants/contracts.",
        parent=node,
        critical=True
    )
    name = ext.sponsored_research_office.name if ext.sponsored_research_office else ""
    claim = f"MIT has an administrative office responsible for managing sponsored research programs, grants, and contracts, namely '{name}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=ext.sponsored_research_office.urls if ext.sponsored_research_office else [],
        additional_instruction="Confirm the office handles sponsored research administration (grants, contracts, proposals)."
    )


async def _verify_department_cs_ee(evaluator: Evaluator, parent_node, ext: MITERCExtraction) -> None:
    node = evaluator.add_parallel(
        id="Department_Covering_CS_EE_And_Related_Fields",
        desc="MIT has a substantial department/unit covering computer science and electrical engineering.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_entity_with_sources(ext.cs_ee_unit),
        id="Department_Covering_CS_EE_And_Related_Fields_exists",
        desc="A CS/EE (or related fields) department/unit identified with source(s).",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Department_Covering_CS_EE_And_Related_Fields_supported",
        desc="Cited page supports the unit coverage (CS/EE) and expert faculty presence.",
        parent=node,
        critical=True
    )
    name = ext.cs_ee_unit.name if ext.cs_ee_unit else ""
    claim = f"MIT has a substantial academic department or unit covering computer science and electrical engineering with expert faculty, specifically '{name}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=ext.cs_ee_unit.urls if ext.cs_ee_unit else [],
        additional_instruction="Confirm the unit covers CS/EE fields and indicates faculty expertise or breadth."
    )


async def _verify_research_compliance(evaluator: Evaluator, parent_node, ext: MITERCExtraction) -> None:
    node = evaluator.add_parallel(
        id="Research_Compliance_Infrastructure",
        desc="MIT has an established research compliance infrastructure (office/officer).",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_entity_with_sources(ext.research_compliance_office),
        id="Research_Compliance_Infrastructure_exists",
        desc="Research compliance office/officer identified with source(s).",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Research_Compliance_Infrastructure_supported",
        desc="Cited page supports an established research compliance infrastructure at MIT.",
        parent=node,
        critical=True
    )
    name = ext.research_compliance_office.name if ext.research_compliance_office else ""
    claim = f"MIT has an established research compliance infrastructure including a designated office or officer, for example '{name}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=ext.research_compliance_office.urls if ext.research_compliance_office else [],
        additional_instruction="Confirm the existence and role of research compliance oversight (policies, training, monitoring)."
    )


async def _verify_data_and_admin_support(evaluator: Evaluator, parent_node, ext: MITERCExtraction) -> None:
    node = evaluator.add_parallel(
        id="Data_Management_And_Research_Administration_Support_Services",
        desc="MIT provides data management and research administration support services.",
        parent=parent_node,
        critical=True
    )

    has_data = _has_services_with_sources(ext.data_management_support)
    has_admin = _has_services_with_sources(ext.research_admin_support)
    evaluator.add_custom_node(
        result=has_data and has_admin,
        id="Data_Management_And_Research_Administration_Support_Services_exists",
        desc="Both data management and research administration support services are identified with source(s).",
        parent=node,
        critical=True
    )

    data_leaf = evaluator.add_leaf(
        id="Data_Management_Services_supported",
        desc="Cited page(s) support MIT data management support services.",
        parent=node,
        critical=True
    )
    data_claim = "MIT provides data management support services required by federal funding agencies."
    await evaluator.verify(
        claim=data_claim,
        node=data_leaf,
        sources=ext.data_management_support.urls if ext.data_management_support else [],
        additional_instruction="Confirm services like data management planning, DMP support, storage, curation, or related offerings."
    )

    admin_leaf = evaluator.add_leaf(
        id="Research_Administration_Services_supported",
        desc="Cited page(s) support MIT research administration support services.",
        parent=node,
        critical=True
    )
    admin_claim = "MIT provides research administration support services required by federal funding agencies."
    await evaluator.verify(
        claim=admin_claim,
        node=admin_leaf,
        sources=ext.research_admin_support.urls if ext.research_admin_support else [],
        additional_instruction="Confirm services such as proposal development, grants management, compliance guidance, or contract support."
    )


async def _verify_collaboration_capacity(evaluator: Evaluator, parent_node, ext: MITERCExtraction) -> None:
    node = evaluator.add_parallel(
        id="Multi_Institution_Collaboration_Capacity",
        desc="MIT demonstrates capacity for multi-institution collaborative research via centers/programs.",
        parent=parent_node,
        critical=True
    )

    first_prog = _first_entity_with_sources(ext.collaboration_programs)
    evaluator.add_custom_node(
        result=_has_entity_with_sources(first_prog),
        id="Multi_Institution_Collaboration_Capacity_exists",
        desc="At least one collaborative research center/program is identified with source(s).",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Multi_Institution_Collaboration_Capacity_supported",
        desc="Cited page supports multi-institution collaboration capacity (partners/consortia).",
        parent=node,
        critical=True
    )
    name = first_prog.name if first_prog else ""
    claim = f"MIT demonstrates capacity for multi-institution collaborative research via '{name}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=first_prog.urls if first_prog else [],
        additional_instruction=(
            "Confirm that the program/center involves collaboration with external institutions "
            "(partners, consortia, joint centers), not solely intra-MIT."
        )
    )


async def _verify_faculty_productivity(evaluator: Evaluator, parent_node, ext: MITERCExtraction) -> None:
    node = evaluator.add_parallel(
        id="AI_And_Robotics_Faculty_Productivity_Evidence",
        desc="MIT has substantial AI/robotics faculty with evidence of research productivity.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_entity_with_sources(ext.faculty_productivity_example),
        id="AI_And_Robotics_Faculty_Productivity_Evidence_exists",
        desc="At least one concrete AI/robotics faculty/group example with evidence is identified with source(s).",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="AI_And_Robotics_Faculty_Productivity_Evidence_supported",
        desc="Cited page shows research outputs (e.g., publications, projects) for the example.",
        parent=node,
        critical=True
    )
    name = ext.faculty_productivity_example.name if ext.faculty_productivity_example else ""
    claim = f"MIT has substantial AI/robotics faculty with research productivity; for example, '{name}' shows research outputs."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=ext.faculty_productivity_example.urls if ext.faculty_productivity_example else [],
        additional_instruction=(
            "Confirm the page shows evidence of productivity (publications, funded projects, awards, major lab/group activities) "
            "in AI/robotics."
        )
    )


async def _verify_library_support(evaluator: Evaluator, parent_node, ext: MITERCExtraction) -> None:
    node = evaluator.add_parallel(
        id="Library_And_Publication_Support",
        desc="MIT provides necessary library resources, journal subscriptions, and publication support services.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_services_with_sources(ext.library_support),
        id="Library_And_Publication_Support_exists",
        desc="Library/publication support services are identified with source(s).",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="Library_And_Publication_Support_supported",
        desc="Cited page(s) support the availability of library resources, journal access, and publication support.",
        parent=node,
        critical=True
    )
    claim = "MIT provides access to library resources, journal subscriptions, and publication support services for researchers."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=ext.library_support.urls if ext.library_support else [],
        additional_instruction="Confirm services like journal databases, subscriptions, publishing support, open access guidance, or similar."
    )


# --------------------------------------------------------------------------- #
# Main verification tree construction                                         #
# --------------------------------------------------------------------------- #
async def build_erc_tree(evaluator: Evaluator, ext: MITERCExtraction) -> None:
    # Critical overall ERC qualification node (child of evaluator root)
    erc_root = evaluator.add_parallel(
        id="ERC_Lead_Institution_Qualification",
        desc="Evaluate whether MIT satisfies all stated requirements to qualify as ERC lead for AI/robotics.",
        parent=evaluator.root,
        critical=True
    )

    # Build each criterion sub-tree (all critical under the ERC node)
    await _verify_global_ranking(evaluator, erc_root, ext)
    await _verify_ai_robotics_labs(evaluator, erc_root, ext)
    await _verify_hpc_access(evaluator, erc_root, ext)
    await _verify_core_shared_facilities(evaluator, erc_root, ext)
    await _verify_irb(evaluator, erc_root, ext)
    await _verify_sponsored_research_office(evaluator, erc_root, ext)
    await _verify_department_cs_ee(evaluator, erc_root, ext)
    await _verify_research_compliance(evaluator, erc_root, ext)
    await _verify_data_and_admin_support(evaluator, erc_root, ext)
    await _verify_collaboration_capacity(evaluator, erc_root, ext)
    await _verify_faculty_productivity(evaluator, erc_root, ext)
    await _verify_library_support(evaluator, erc_root, ext)


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
    Evaluate whether MIT qualifies as an ERC lead institution for AI/robotics based on the provided answer.
    """
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation
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

    # Extract structured evidence from the answer
    ext = await evaluator.extract(
        prompt=prompt_extract_mit_erc_evidence(),
        template_class=MITERCExtraction,
        extraction_name="mit_erc_evidence"
    )

    # Build and execute verification tree
    await build_erc_tree(evaluator, ext)

    # Return standardized summary
    return evaluator.get_summary()