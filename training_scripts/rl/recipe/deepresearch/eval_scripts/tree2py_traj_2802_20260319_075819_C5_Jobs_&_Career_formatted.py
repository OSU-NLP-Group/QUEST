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
TASK_ID = "pa_md_career_milestones"
TASK_DESCRIPTION = """
You are advising a teacher who currently holds a Pennsylvania Level I teaching certificate and is planning their long-term career in education. They want to understand the complete requirements for two career milestones: (1) converting their Pennsylvania Level I certificate to a Level II certificate, and (2) eventually becoming eligible for a principal position at Prince George's County Public Schools (PGCPS) in Maryland.

Provide a comprehensive analysis that identifies and explains:

A. Pennsylvania Level II Conversion Requirements - All requirements needed to convert from Level I to Level II certification, organized into the following categories:
   - Timeline constraints (including certificate validity period and conversion window)
   - Educational requirements (including total credit hours and subject-specific credit requirements)
   - Performance evaluation requirements (including number of evaluations and rating standards)
   - Program completion requirements
   - Application and documentation procedures

B. Maryland PGCPS Principal Eligibility Requirements - All requirements needed to become eligible for a principal position at PGCPS, including:
   - Educational degree requirements
   - Teaching experience requirements
   - Special education coursework requirements
   - Administrative certification pathway options
   - PGCPS-specific certification requirements

For each requirement identified, provide the specific details (such as numerical values, timeframes, or procedural steps) and include reference URL(s) to support your answer.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RequirementWithSources(BaseModel):
    claim: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PAExtraction(BaseModel):
    # Timeline
    timeline_validity_6_years: Optional[RequirementWithSources] = None
    timeline_no_renewal: Optional[RequirementWithSources] = None
    timeline_min_3_years: Optional[RequirementWithSources] = None
    timeline_window_3_to_6: Optional[RequirementWithSources] = None

    # Educational
    ed_24_credits: Optional[RequirementWithSources] = None
    ed_min6_related: Optional[RequirementWithSources] = None

    # Performance evaluation
    perf_6_evals: Optional[RequirementWithSources] = None

    # Program completion
    program_induction_verified: Optional[RequirementWithSources] = None

    # Application & documentation
    app_submit_tims: Optional[RequirementWithSources] = None
    app_transcripts: Optional[RequirementWithSources] = None
    app_pde427: Optional[RequirementWithSources] = None


class MDPathwaysExtraction(BaseModel):
    pathway_a_dept_program: Optional[RequirementWithSources] = None
    pathway_b_oot_practicum: Optional[RequirementWithSources] = None
    pathway_c_18hrs_240practicum: Optional[RequirementWithSources] = None
    pathway_d_other_license_5yrs: Optional[RequirementWithSources] = None
    slla_required: Optional[RequirementWithSources] = None


class MDExtraction(BaseModel):
    degree_masters: Optional[RequirementWithSources] = None
    experience_27_months: Optional[RequirementWithSources] = None
    special_ed_coursework_3sh_or_cpd: Optional[RequirementWithSources] = None
    administrative_certification_pathways: Optional[MDPathwaysExtraction] = None
    pgcps_admin2_apc: Optional[RequirementWithSources] = None


class CareerMilestonesExtraction(BaseModel):
    pa: Optional[PAExtraction] = None
    md: Optional[MDExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_career_milestones() -> str:
    return """
Extract from the answer the exact requirement statements (as concise declarative sentences) and the specific URL(s) that the answer cites for each requirement below. Do NOT invent details. If the answer omits a requirement or any URLs for it, set claim to null and sources to an empty array.

Return JSON matching this schema exactly:

{
  "pa": {
    "timeline_validity_6_years": {"claim": string|null, "sources": string[]},
    "timeline_no_renewal": {"claim": string|null, "sources": string[]},
    "timeline_min_3_years": {"claim": string|null, "sources": string[]},
    "timeline_window_3_to_6": {"claim": string|null, "sources": string[]},

    "ed_24_credits": {"claim": string|null, "sources": string[]},
    "ed_min6_related": {"claim": string|null, "sources": string[]},

    "perf_6_evals": {"claim": string|null, "sources": string[]},

    "program_induction_verified": {"claim": string|null, "sources": string[]},

    "app_submit_tims": {"claim": string|null, "sources": string[]},
    "app_transcripts": {"claim": string|null, "sources": string[]},
    "app_pde427": {"claim": string|null, "sources": string[]}
  },
  "md": {
    "degree_masters": {"claim": string|null, "sources": string[]},
    "experience_27_months": {"claim": string|null, "sources": string[]},
    "special_ed_coursework_3sh_or_cpd": {"claim": string|null, "sources": string[]},

    "administrative_certification_pathways": {
      "pathway_a_dept_program": {"claim": string|null, "sources": string[]},
      "pathway_b_oot_practicum": {"claim": string|null, "sources": string[]},
      "pathway_c_18hrs_240practicum": {"claim": string|null, "sources": string[]},
      "pathway_d_other_license_5yrs": {"claim": string|null, "sources": string[]},
      "slla_required": {"claim": string|null, "sources": string[]}
    },

    "pgcps_admin2_apc": {"claim": string|null, "sources": string[]}
  }
}

Guidance for the 'claim' text:
- Copy the statement from the answer succinctly and include the key numeric/timeframe or procedural specifics (e.g., “6 years of service (not calendar years)”, “24 post‑baccalaureate credits”, “6 semi‑annual satisfactory evaluations”, “PDE Form 427”, “TIMS online”, “3 semester hours in special education or approved CPD”, “27 months effective teaching/specialist experience”, “Administrator II + APC for PGCPS”).
- If the answer does not include that requirement, set claim to null.
- The 'sources' must be actual URLs explicitly present in the answer for that requirement (plain or markdown links). If none provided, set an empty list.
""".strip()


# --------------------------------------------------------------------------- #
# Helper: uniform verification for a single requirement leaf                  #
# --------------------------------------------------------------------------- #
async def verify_requirement_with_urls(
    evaluator: Evaluator,
    *,
    parent,
    node_id: str,
    desc: str,
    req: Optional[RequirementWithSources],
    critical: bool = True,
    additional_instruction: Optional[str] = None,
) -> None:
    """
    Create a leaf node for a requirement and verify its claim using the extracted URLs.
    If the claim or sources are missing, mark the leaf as failed without LLM verification.
    """
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )

    claim_text = (req.claim.strip() if (req and req.claim) else None)
    srcs = (req.sources if (req and req.sources) else [])

    if not claim_text or len(srcs) == 0:
        # Missing either the statement or the supporting URLs → fail this leaf
        leaf.score = 0.0
        leaf.status = "failed"
        return

    await evaluator.verify(
        claim=claim_text,
        node=leaf,
        sources=srcs,
        additional_instruction=additional_instruction or "None",
    )


# --------------------------------------------------------------------------- #
# Build PA verification subtree                                               #
# --------------------------------------------------------------------------- #
async def build_pa_subtree(evaluator: Evaluator, parent, pa: Optional[PAExtraction]) -> None:
    pa_node = evaluator.add_parallel(
        id="Milestone_1_PA_Level_II_Conversion",
        desc="All requirements to convert PA Level I to Level II, organized by the requested categories, with specific details and supporting URL(s) per requirement.",
        parent=parent,
        critical=True,
    )

    # Group: Timeline constraints
    tl_node = evaluator.add_parallel(
        id="PA_Timeline_Constraints",
        desc="Timeline constraints for PA Level II conversion.",
        parent=pa_node,
        critical=True,
    )
    await verify_requirement_with_urls(
        evaluator,
        parent=tl_node,
        node_id="PA_Timeline_Validity_6_Years_Service",
        desc="States Level I is valid for exactly 6 years of service (not calendar years) and provides supporting URL(s).",
        req=pa.timeline_validity_6_years if pa else None,
        additional_instruction="Verify the page explicitly states a Pennsylvania Level I teaching certificate is valid for 6 years of SERVICE (not calendar years). Accept synonymous wording such as 'six years of service' or '6 years of satisfactory service.'",
    )
    await verify_requirement_with_urls(
        evaluator,
        parent=tl_node,
        node_id="PA_Timeline_No_Renewal_After_Expiry",
        desc="States Level I cannot be renewed after the 6-year validity period and must be converted before expiration; includes supporting URL(s).",
        req=pa.timeline_no_renewal if pa else None,
        additional_instruction="Look for statements that Level I certificates are non-renewable and must be converted to Level II before or upon reaching the service limit.",
    )
    await verify_requirement_with_urls(
        evaluator,
        parent=tl_node,
        node_id="PA_Timeline_Min_3_Years_Creditable_Service",
        desc="States a minimum of 3 years of creditable service is required before applying for Level II; includes supporting URL(s).",
        req=pa.timeline_min_3_years if pa else None,
        additional_instruction="The evidence should indicate at least 3 years of creditable/satisfactory service are required for Level II.",
    )
    await verify_requirement_with_urls(
        evaluator,
        parent=tl_node,
        node_id="PA_Timeline_Conversion_Window_3_to_6_Years",
        desc="States conversion must occur between 3 and 6 years of service on Level I; includes supporting URL(s).",
        req=pa.timeline_window_3_to_6 if pa else None,
        additional_instruction="Confirm sources specify a conversion window between the 3rd and 6th years of service (inclusive language acceptable).",
    )

    # Group: Educational requirements
    ed_node = evaluator.add_parallel(
        id="PA_Educational_Requirements",
        desc="Educational requirements for PA Level II conversion.",
        parent=pa_node,
        critical=True,
    )
    await verify_requirement_with_urls(
        evaluator,
        parent=ed_node,
        node_id="PA_Ed_24_PostBacc_Credits",
        desc="States 24 post-baccalaureate credits are required; includes supporting URL(s).",
        req=pa.ed_24_credits if pa else None,
        additional_instruction="The page should clearly state that 24 post-baccalaureate credits are required for Level II.",
    )
    await verify_requirement_with_urls(
        evaluator,
        parent=ed_node,
        node_id="PA_Ed_Min_6_Credits_Related_or_Enhancing",
        desc="States at least 6 of the 24 credits must be related to the certification area or designed to enhance professional practice; includes supporting URL(s).",
        req=pa.ed_min6_related if pa else None,
        additional_instruction="Confirm the 6-credit subset rule (related to certification area or designed to enhance professional practice).",
    )

    # Group: Performance evaluation requirements
    perf_node = evaluator.add_parallel(
        id="PA_Performance_Evaluation_Requirements",
        desc="Performance evaluation requirements for PA Level II conversion.",
        parent=pa_node,
        critical=True,
    )
    await verify_requirement_with_urls(
        evaluator,
        parent=perf_node,
        node_id="PA_Perf_6_Semiannual_Satisfactory_Evals",
        desc="States 6 semi-annual evaluations with satisfactory ratings are required (representing 3 years of satisfactory service) and provides supporting URL(s).",
        req=pa.perf_6_evals if pa else None,
        additional_instruction="Check that the evidence mentions six semiannual evaluations (or equivalent) with satisfactory ratings over three years.",
    )

    # Group: Program completion requirements
    prog_node = evaluator.add_parallel(
        id="PA_Program_Completion_Requirements",
        desc="Program completion requirements for PA Level II conversion.",
        parent=pa_node,
        critical=True,
    )
    await verify_requirement_with_urls(
        evaluator,
        parent=prog_node,
        node_id="PA_Program_PDE_Approved_Induction_Verified",
        desc="States completion of a PDE-approved induction program is required and must be verified by the chief school administrator; includes supporting URL(s).",
        req=pa.program_induction_verified if pa else None,
        additional_instruction="Look for a PDE-approved induction program requirement and that verification must be provided (e.g., by a chief school administrator).",
    )

    # Group: Application & documentation
    app_node = evaluator.add_parallel(
        id="PA_Application_and_Documentation_Procedures",
        desc="Application/documentation procedures for PA Level II conversion.",
        parent=pa_node,
        critical=True,
    )
    await verify_requirement_with_urls(
        evaluator,
        parent=app_node,
        node_id="PA_App_Submit_Via_TIMS",
        desc="States the application must be submitted via TIMS (online) and includes supporting URL(s).",
        req=pa.app_submit_tims if pa else None,
        additional_instruction="Verify that TIMS (online Teacher Information Management System) is the required submission method.",
    )
    await verify_requirement_with_urls(
        evaluator,
        parent=app_node,
        node_id="PA_App_Official_Transcripts_Submission_Instructions",
        desc="States official transcripts must be sent to PDE and includes the electronic transcript option to ra-teachercert@pa.gov (if available), with supporting URL(s).",
        req=pa.app_transcripts if pa else None,
        additional_instruction="Confirm official transcripts must be sent to PDE. If the claim mentions the option to send electronic transcripts to ra-teachercert@pa.gov, verify that detail as well.",
    )
    await verify_requirement_with_urls(
        evaluator,
        parent=app_node,
        node_id="PA_App_PDE_Form_427_Summative_Evaluation",
        desc="States supervisor must complete PDE Form 427 Level I→Level II Summative Evaluation and includes supporting URL(s).",
        req=pa.app_pde427 if pa else None,
        additional_instruction="Look for PDE Form 427 (Level I to Level II Summative Evaluation) being required/completed by the supervisor/administrator.",
    )


# --------------------------------------------------------------------------- #
# Build MD/PGCPS verification subtree                                         #
# --------------------------------------------------------------------------- #
async def build_md_subtree(evaluator: Evaluator, parent, md: Optional[MDExtraction]) -> None:
    md_node = evaluator.add_parallel(
        id="Milestone_2_MD_PGCPS_Principal_Eligibility",
        desc="All requirements to become eligible for a principal position at PGCPS, including MD administrative licensure requirements and PGCPS-specific certification requirements, with specific details and supporting URL(s) per requirement.",
        parent=parent,
        critical=True,
    )

    # Core MD requirements
    await verify_requirement_with_urls(
        evaluator,
        parent=md_node,
        node_id="MD_Degree_Requirement",
        desc="States a master’s degree is required for Maryland administrative licensure (as specified) and includes supporting URL(s).",
        req=md.degree_masters if md else None,
        additional_instruction="Confirm that a master's degree is required for Maryland administrative licensure (e.g., Administrator I/II pathway prerequisites).",
    )
    await verify_requirement_with_urls(
        evaluator,
        parent=md_node,
        node_id="MD_Teaching_or_Specialist_Experience_27_Months",
        desc="States 27 months of effective teaching performance or effective performance as a licensed specialist is required and includes supporting URL(s).",
        req=md.experience_27_months if md else None,
        additional_instruction="Evidence should specify 27 months of effective performance as a teacher or as a licensed specialist.",
    )
    await verify_requirement_with_urls(
        evaluator,
        parent=md_node,
        node_id="MD_Special_Education_Coursework_3_Semester_Hours_or_CPD",
        desc="States 3 semester hours or State-approved CPD credits in special education coursework are required and includes supporting URL(s).",
        req=md.special_ed_coursework_3sh_or_cpd if md else None,
        additional_instruction="Look for requirement of 3 semester hours (or equivalent State-approved CPD credits) in special education coursework.",
    )

    # Administrative certification pathways (with SLLA condition)
    pathways_node = evaluator.add_parallel(
        id="MD_Administrative_Certification_Pathways",
        desc="Describes the four Maryland administrative licensure pathway options and the SLLA condition where applicable, with supporting URL(s).",
        parent=md_node,
        critical=True,
    )
    mdp = md.administrative_certification_pathways if md else None

    await verify_requirement_with_urls(
        evaluator,
        parent=pathways_node,
        node_id="MD_Pathway_A_Department_Approved_Program",
        desc="Includes pathway (a): Department-approved administrative program; includes supporting URL(s).",
        req=mdp.pathway_a_dept_program if mdp else None,
        additional_instruction="Verify inclusion of a Maryland Department-approved administrator preparation program as a valid pathway.",
    )
    await verify_requirement_with_urls(
        evaluator,
        parent=pathways_node,
        node_id="MD_Pathway_B_Out_of_State_Program_with_Practicum",
        desc="Includes pathway (b): approved out-of-state program with supervised clinical practicum; includes supporting URL(s).",
        req=mdp.pathway_b_oot_practicum if mdp else None,
        additional_instruction="Confirm an approved out-of-state administrator program with supervised clinical practicum is an option.",
    )
    await verify_requirement_with_urls(
        evaluator,
        parent=pathways_node,
        node_id="MD_Pathway_C_18_Graduate_Hours_plus_240_Hour_Practicum",
        desc="Includes pathway (c): 18 graduate semester hours in specified categories plus a 240-hour supervised clinical practicum; includes supporting URL(s).",
        req=mdp.pathway_c_18hrs_240practicum if mdp else None,
        additional_instruction="Check for the 18 graduate semester hours in specified domains AND a 240-hour supervised clinical practicum.",
    )
    await verify_requirement_with_urls(
        evaluator,
        parent=pathways_node,
        node_id="MD_Pathway_D_Other_License_plus_5_Years_Admin_Experience",
        desc="Includes pathway (d): valid professional administrative license from another state/country plus 5 years of effective PK-12 administrative experience; includes supporting URL(s).",
        req=mdp.pathway_d_other_license_5yrs if mdp else None,
        additional_instruction="Look for a valid professional admin license from another jurisdiction AND 5 years of effective PK–12 administrative experience.",
    )
    await verify_requirement_with_urls(
        evaluator,
        parent=pathways_node,
        node_id="MD_SLLA_Required_for_Specified_Pathways",
        desc="States SLLA qualifying score is required for the specified pathways (out-of-state program and/or 18-hour coursework pathway) and includes supporting URL(s).",
        req=mdp.slla_required if mdp else None,
        additional_instruction="Verify that a qualifying SLLA score is required for the designated pathways (commonly for out-of-state and/or the 18-hour coursework pathway).",
    )

    # PGCPS-specific requirement
    await verify_requirement_with_urls(
        evaluator,
        parent=md_node,
        node_id="PGCPS_Specific_Certification_Administrator_II_APC",
        desc="States PGCPS principal eligibility requires a Maryland Advanced Professional Certificate with Administrator II certification attached to the application; includes supporting URL(s).",
        req=md.pgcps_admin2_apc if md else None,
        additional_instruction="Check PGCPS job postings or HR pages stating candidates must hold a Maryland Advanced Professional Certificate (APC) with Administrator II certification at application or by hire.",
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
    Evaluate an answer for the PA Level I→II conversion and MD/PGCPS principal eligibility requirements task.
    """
    # Initialize evaluator with parallel aggregation at root
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_career_milestones(),
        template_class=CareerMilestonesExtraction,
        extraction_name="career_milestones_extraction",
    )

    # Create a top-level critical node as specified by the rubric
    top = evaluator.add_parallel(
        id="Complete_Career_Milestones_Response",
        desc="Covers BOTH requested career milestones (PA Level I→II conversion; MD/PGCPS principal eligibility) and provides the required specific details with supporting reference URL(s).",
        parent=root,
        critical=True,
    )

    # Build PA subtree
    await build_pa_subtree(evaluator, top, extraction.pa if extraction else None)

    # Build MD/PGCPS subtree
    await build_md_subtree(evaluator, top, extraction.md if extraction else None)

    # Return the structured evaluation summary
    return evaluator.get_summary()