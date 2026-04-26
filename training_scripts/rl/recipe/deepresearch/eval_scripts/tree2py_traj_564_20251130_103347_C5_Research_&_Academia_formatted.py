import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nsf_apsa_collab_2025"
TASK_DESCRIPTION = (
    "A research team at the University of California, San Diego is planning to submit an NSF collaborative research "
    "proposal in 2025 that will establish a partnership with Stanford University and follow the APSA Research Partnerships "
    "Program guidelines for team composition. What are the specific documentation requirements and specifications for this "
    "proposal, including: (1) the page limit and mandatory content elements for the data management and sharing plan; "
    "(2) the page limit for the budget justification; (3) the page limits for project descriptions in both standard "
    "small/medium proposals and large proposals; (4) the minimum team composition requirements under APSA guidelines "
    "specifying academic and applied members; and (5) the typical duration for seed grant or pilot collaboration projects?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DmpElements(BaseModel):
    data_types_and_materials_required: Optional[bool] = None
    standards_for_data_and_metadata_required: Optional[bool] = None
    access_and_sharing_policies_required: Optional[bool] = None
    reuse_and_redistribution_policies_required: Optional[bool] = None
    archiving_and_preservation_plans_required: Optional[bool] = None


class DMPRequirements(BaseModel):
    page_limit: Optional[str] = None
    elements: Optional[DmpElements] = None
    sources: List[str] = Field(default_factory=list)


class BudgetJustificationInfo(BaseModel):
    page_limit: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ProjectDescriptionInfo(BaseModel):
    small_medium_page_limit: Optional[str] = None
    large_page_limit: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class APSATeamCompositionRequirements(BaseModel):
    min_academic_members: Optional[str] = None
    min_applied_members: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ProjectDurationInfo(BaseModel):
    typical_seed_or_pilot_duration: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ProposalRequirementsExtraction(BaseModel):
    dmp: Optional[DMPRequirements] = None
    budget: Optional[BudgetJustificationInfo] = None
    description_limits: Optional[ProjectDescriptionInfo] = None
    apsa_team: Optional[APSATeamCompositionRequirements] = None
    duration: Optional[ProjectDurationInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_proposal_requirements() -> str:
    return (
        "Extract the specific documentation requirements mentioned in the answer for an NSF collaborative research "
        "proposal (2025 timeframe), along with any URLs cited as sources supporting each requirement. "
        "Return a JSON object with the following structure:\n\n"
        "dmp: {\n"
        "  page_limit: string or null,\n"
        "  elements: {\n"
        "    data_types_and_materials_required: boolean or null,\n"
        "    standards_for_data_and_metadata_required: boolean or null,\n"
        "    access_and_sharing_policies_required: boolean or null,\n"
        "    reuse_and_redistribution_policies_required: boolean or null,\n"
        "    archiving_and_preservation_plans_required: boolean or null\n"
        "  },\n"
        "  sources: [list of URLs explicitly cited for the Data Management and Sharing Plan]\n"
        "},\n\n"
        "budget: {\n"
        "  page_limit: string or null,\n"
        "  sources: [list of URLs explicitly cited for Budget Justification]\n"
        "},\n\n"
        "description_limits: {\n"
        "  small_medium_page_limit: string or null,\n"
        "  large_page_limit: string or null,\n"
        "  sources: [list of URLs explicitly cited for Project Description page limits]\n"
        "},\n\n"
        "apsa_team: {\n"
        "  min_academic_members: string or null,\n"
        "  min_applied_members: string or null,\n"
        "  sources: [list of URLs explicitly cited for APSA Research Partnerships Program team composition]\n"
        "},\n\n"
        "duration: {\n"
        "  typical_seed_or_pilot_duration: string or null,\n"
        "  sources: [list of URLs explicitly cited for typical seed grant/pilot duration]\n"
        "}\n\n"
        "Rules:\n"
        "- Extract only what is explicitly mentioned in the answer; do not invent.\n"
        "- For URL fields, include only actual URLs present in the answer (plain or markdown link forms).\n"
        "- If any field is not present in the answer, set it to null and use an empty list for its sources.\n"
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_dmp(
    evaluator: Evaluator,
    parent_node,
    dmp: Optional[DMPRequirements],
) -> None:
    group_node = evaluator.add_parallel(
        id="Data_Management_Plan_Specifications",
        desc="Verify data management and sharing plan requirements",
        parent=parent_node,
        critical=True,
    )

    sources = dmp.sources if dmp and dmp.sources else []

    # Leaf nodes for DMP page limit and required content elements
    node_dmp_limit = evaluator.add_leaf(
        id="DMP_Page_Limit",
        desc="Data management plan must not exceed 2 pages",
        parent=group_node,
        critical=True,
    )

    node_dmp_data_types = evaluator.add_leaf(
        id="DMP_Data_Types_Content",
        desc="Plan must specify types of data, samples, physical collections, software, curriculum materials, and other materials to be produced",
        parent=group_node,
        critical=True,
    )

    node_dmp_standards = evaluator.add_leaf(
        id="DMP_Standards_Content",
        desc="Plan must specify standards for data and metadata format and content",
        parent=group_node,
        critical=True,
    )

    node_dmp_access = evaluator.add_leaf(
        id="DMP_Access_Sharing_Policies",
        desc="Plan must include policies for data access and sharing, including provisions for appropriate protection of privacy, confidentiality, security, intellectual property or other rights or requirements",
        parent=group_node,
        critical=True,
    )

    node_dmp_reuse = evaluator.add_leaf(
        id="DMP_Reuse_Provisions",
        desc="Plan must include policies and provisions for data reuse, redistribution and the production of derivatives",
        parent=group_node,
        critical=True,
    )

    node_dmp_archiving = evaluator.add_leaf(
        id="DMP_Archiving_Plans",
        desc="Plan must include plans for archiving data, samples and other research products, and for preserving access to them",
        parent=group_node,
        critical=True,
    )

    claims = [
        (
            "NSF requires that the Data Management and Sharing Plan (DMSP/DMP) must not exceed 2 pages.",
            sources,
            node_dmp_limit,
            "Verify against the NSF PAPPG Data Management and Sharing Plan section (e.g., Chapter II.D.2(i) or equivalent). "
            "Accept 'DMP'/'DMSP' synonyms; confirm the explicit 2-page maximum."
        ),
        (
            "NSF requires the DMSP to specify the types of data, samples, physical collections, software, curriculum materials, and other materials to be produced.",
            sources,
            node_dmp_data_types,
            "Check the DMSP content requirements in the NSF PAPPG. Confirm that these material types are explicitly listed as required content."
        ),
        (
            "NSF requires the DMSP to specify the standards for data and metadata format and content.",
            sources,
            node_dmp_standards,
            "Confirm that standards for data and metadata (format/content) are explicitly required in the DMSP per the PAPPG."
        ),
        (
            "NSF requires the DMSP to include policies for data access and sharing, with appropriate provisions for privacy, confidentiality, security, intellectual property, and other rights or requirements.",
            sources,
            node_dmp_access,
            "Verify the access/sharing policy requirements for the DMSP and the protections (privacy, confidentiality, security, IP, etc.) in the PAPPG."
        ),
        (
            "NSF requires the DMSP to include policies and provisions for data reuse, redistribution, and the production of derivatives.",
            sources,
            node_dmp_reuse,
            "Confirm that reuse, redistribution, and derivatives policies are explicit DMSP requirements in the PAPPG."
        ),
        (
            "NSF requires the DMSP to include plans for archiving data, samples, and other research products, and preserving access to them.",
            sources,
            node_dmp_archiving,
            "Confirm archiving and preservation requirements are explicitly part of the DMSP per the PAPPG."
        ),
    ]

    await evaluator.batch_verify(claims)


async def verify_budget(
    evaluator: Evaluator,
    parent_node,
    budget: Optional[BudgetJustificationInfo],
) -> None:
    group_node = evaluator.add_parallel(
        id="Budget_Justification_Specifications",
        desc="Verify budget justification page limit requirement",
        parent=parent_node,
        critical=True,
    )

    sources = budget.sources if budget and budget.sources else []

    node_budget_limit = evaluator.add_leaf(
        id="Budget_Page_Limit",
        desc="Budget justification must not exceed 5 pages unless otherwise specified in the program solicitation",
        parent=group_node,
        critical=True,
    )

    claim = (
        "NSF requires that the Budget Justification section must not exceed 5 pages, "
        "unless a specific program solicitation states otherwise."
    )
    await evaluator.verify(
        claim=claim,
        node=node_budget_limit,
        sources=sources,
        additional_instruction="Verify in the NSF PAPPG Budget Justification guidance (e.g., Chapter II.C.2.g or equivalent) "
                              "that the standard page limit is 5 pages, with solicitation-based exceptions allowed."
    )


async def verify_project_description(
    evaluator: Evaluator,
    parent_node,
    desc_info: Optional[ProjectDescriptionInfo],
) -> None:
    group_node = evaluator.add_parallel(
        id="Project_Description_Specifications",
        desc="Verify project description page limit requirements",
        parent=parent_node,
        critical=True,
    )

    sources = desc_info.sources if desc_info and desc_info.sources else []

    node_small_medium = evaluator.add_leaf(
        id="Standard_Proposal_Page_Limit",
        desc="Project description for small/medium proposals must not exceed 15 pages",
        parent=group_node,
        critical=True,
    )
    node_large = evaluator.add_leaf(
        id="Large_Proposal_Page_Limit",
        desc="Project description for large proposals must not exceed 20 pages",
        parent=group_node,
        critical=True,
    )

    claims = [
        (
            "For standard NSF proposals (Small/Medium), the Project Description must not exceed 15 pages unless overridden by a specific solicitation.",
            sources,
            node_small_medium,
            "Verify the 15-page limit for Small/Medium proposals in the NSF PAPPG. Allow for solicitation-specific overrides."
        ),
        (
            "For Large NSF proposals, the Project Description must not exceed 20 pages unless overridden by a specific solicitation.",
            sources,
            node_large,
            "Verify the 20-page limit for Large proposals in the NSF PAPPG. Allow for solicitation-specific overrides."
        ),
    ]

    await evaluator.batch_verify(claims)


async def verify_apsa_team(
    evaluator: Evaluator,
    parent_node,
    apsa: Optional[APSATeamCompositionRequirements],
) -> None:
    group_node = evaluator.add_parallel(
        id="APSA_Team_Composition",
        desc="Verify APSA Research Partnership minimum team composition requirements",
        parent=parent_node,
        critical=True,
    )

    sources = apsa.sources if apsa and apsa.sources else []

    node_min_academic = evaluator.add_leaf(
        id="Minimum_Academic_Members",
        desc="Team must include minimum of 4 academic members",
        parent=group_node,
        critical=True,
    )
    node_min_applied = evaluator.add_leaf(
        id="Minimum_Applied_Members",
        desc="Team must include minimum of 4 applied team members (practitioners/experts)",
        parent=group_node,
        critical=True,
    )

    claims = [
        (
            "Under APSA Research Partnerships Program guidelines, teams must include a minimum of 4 academic members.",
            sources,
            node_min_academic,
            "Verify the APSA RPP guideline specifying at least 4 academic members on the team."
        ),
        (
            "Under APSA Research Partnerships Program guidelines, teams must include a minimum of 4 applied team members (practitioners/experts).",
            sources,
            node_min_applied,
            "Verify the APSA RPP guideline specifying at least 4 applied/practitioner members on the team."
        ),
    ]

    await evaluator.batch_verify(claims)


async def verify_duration(
    evaluator: Evaluator,
    parent_node,
    duration_info: Optional[ProjectDurationInfo],
) -> None:
    group_node = evaluator.add_parallel(
        id="Typical_Project_Duration",
        desc="Verify typical duration for seed grant/pilot collaboration projects",
        parent=parent_node,
        critical=True,
    )

    sources = duration_info.sources if duration_info and duration_info.sources else []

    node_duration = evaluator.add_leaf(
        id="Standard_Seed_Grant_Duration",
        desc="Typical seed grant or pilot collaboration projects have a duration of 12 months",
        parent=group_node,
        critical=True,
    )

    claim = (
        "Typical seed grant or pilot collaboration projects under APSA-style research partnerships have a duration of approximately 12 months."
    )

    await evaluator.verify(
        claim=claim,
        node=node_duration,
        sources=sources,
        additional_instruction="Verify the typical duration (around 12 months) for APSA Research Partnerships seed/pilot projects "
                              "from APSA program pages or official descriptions; allow reasonable wording variants like 'one year'."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for NSF collaborative proposal requirements following APSA Research Partnerships Program guidelines.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation per rubric
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

    # Extract structured information from the answer
    extracted: ProposalRequirementsExtraction = await evaluator.extract(
        prompt=prompt_extract_proposal_requirements(),
        template_class=ProposalRequirementsExtraction,
        extraction_name="proposal_requirements_extraction",
    )

    # Create the main critical parent node as specified in the rubric
    main_node = evaluator.add_parallel(
        id="NSF_APSA_Collaborative_Proposal_Requirements",
        desc="Verify all required specifications for an NSF collaborative research proposal following APSA Research Partnership guidelines",
        parent=root,
        critical=True,
    )

    # Build and verify each specification group (all critical under main node)
    await verify_dmp(evaluator, main_node, extracted.dmp)
    await verify_budget(evaluator, main_node, extracted.budget)
    await verify_project_description(evaluator, main_node, extracted.description_limits)
    await verify_apsa_team(evaluator, main_node, extracted.apsa_team)
    await verify_duration(evaluator, main_node, extracted.duration)

    # Return the structured evaluation summary
    return evaluator.get_summary()