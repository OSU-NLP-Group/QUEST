import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "va_hqwbl_advanced_studies_cybersecurity"
TASK_DESCRIPTION = (
    "A high school student in Fairfax County, Virginia, who entered 9th grade in 2020-2021 and is pursuing an Advanced Studies Diploma, "
    "plans to complete a High-Quality Work-Based Learning (HQWBL) experience as their additional graduation requirement instead of earning "
    "a CTE credential or taking an AP/honors course. The student is considering a 100-hour internship in cybersecurity at a technology company "
    "in Virginia, where they will work under the direct supervision of a certified information security professional. The internship will provide "
    "the student with the opportunity to earn an industry-recognized cybersecurity certification upon completion. According to Virginia Department of "
    "Education regulations for HQWBL experiences, verify whether this proposed internship meets all the mandatory requirements to satisfy the Advanced "
    "Studies Diploma's additional graduation requirement. Your answer must confirm: (1) whether the duration meets the minimum requirement, "
    "(2) whether cybersecurity qualifies as a high-demand field according to Virginia's official occupational classification, "
    "(3) whether the credential opportunity requirement is satisfied, (4) whether the supervision arrangement constitutes an authentic worksite, "
    "and (5) what documentation must be completed to verify the experience meets all HQWBL criteria including the daily application of Virginia's 5 Cs."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class InternshipDetails(BaseModel):
    """Extracted details about the proposed internship from the answer."""
    duration_hours: Optional[str] = None  # e.g., "100 hours"
    field: Optional[str] = None           # e.g., "cybersecurity"
    supervision_arrangement: Optional[str] = None  # e.g., "direct supervision by a certified information security professional"
    credential_opportunity: Optional[str] = None   # e.g., "opportunity to earn an industry-recognized cybersecurity certification"
    worksite_company: Optional[str] = None         # e.g., "technology company in Virginia"


class HQWBLSources(BaseModel):
    """URLs cited in the answer to support HQWBL requirements."""
    program_structure_urls: List[str] = Field(default_factory=list)   # VDOE pages describing HQWBL program structure requirements
    implementation_urls: List[str] = Field(default_factory=list)      # VDOE pages describing HQWBL implementation and documentation
    voee_urls: List[str] = Field(default_factory=list)                # VOEE High Demand Occupations Dashboard or relevant VOEE pages
    documentation_urls: List[str] = Field(default_factory=list)       # VDOE WBL Student Evaluation form or documentation requirements pages
    credential_urls: List[str] = Field(default_factory=list)          # Company/program pages stating certification opportunity or VDOE-approved credential info


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_internship_details() -> str:
    return """
    Extract the proposed internship details as stated in the answer. Return:
    - duration_hours: the stated duration (e.g., "100 hours"). If not explicitly stated, return null.
    - field: the field of the internship (e.g., "cybersecurity"). If not explicitly stated, return null.
    - supervision_arrangement: how supervision is arranged (e.g., "direct supervision by a certified information security professional"). If not explicitly stated, return null.
    - credential_opportunity: text describing any opportunity to earn an industry-recognized certification or similar credential (e.g., "opportunity to earn an industry-recognized cybersecurity certification upon completion"). If not explicitly stated, return null.
    - worksite_company: the name or description of the worksite (e.g., "technology company in Virginia"). If not explicitly stated, return null.
    Only extract what is explicitly present in the answer. Do not invent or infer.
    """


def prompt_extract_hqwbl_sources() -> str:
    return """
    Extract all URLs cited in the answer that are relevant to verifying HQWBL requirements. Categorize them as:
    - program_structure_urls: Virginia Department of Education (VDOE) pages documenting HQWBL program structure requirements (e.g., definitions, minimum hours).
    - implementation_urls: VDOE pages documenting HQWBL implementation, authentic worksites, and documentation requirements.
    - voee_urls: Virginia Office of Education Economics (VOEE) High Demand Occupations Dashboard or VOEE pages evidencing high-demand fields.
    - documentation_urls: VDOE Work-Based Learning (WBL) Student Evaluation form or official VDOE documentation pages that specify required forms and 5 Cs documentation.
    - credential_urls: Company/program pages or official sources that state the internship provides an opportunity to earn an industry-recognized cybersecurity certification (or VDOE-approved credential lists).
    Special rules:
    - Extract only URLs explicitly present in the answer (including markdown links). Do not invent any URLs.
    - If a category has no URLs in the answer, return an empty list for that category.
    - Prefer official Virginia sources for policy (e.g., domains containing "virginia.gov" or "doe.virginia.gov") where applicable.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_vdoe_domain(urls: List[str]) -> bool:
    """Check if any URL likely belongs to Virginia Department of Education."""
    for u in urls:
        lu = u.lower()
        if "virginia.gov" in lu or "doe.virginia.gov" in lu or "vdoe" in lu:
            return True
    return False


def _non_empty(urls: List[str]) -> bool:
    return bool(urls) and len(urls) > 0


def _contains_certification_text(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(k in t for k in ["certification", "credential", "certificate", "postsecondary credit"])


def _mentions_cybersecurity(text: Optional[str]) -> bool:
    if not text:
        return False
    return "cybersecurity" in text.lower() or "information security" in text.lower()


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_program_structure_requirements(
    evaluator: Evaluator,
    parent_node,
    details: InternshipDetails,
    sources: HQWBLSources,
) -> None:
    """
    Build and verify the 'Program_Structure_Requirements' subtree.
    """
    prog_node = evaluator.add_parallel(
        id="Program_Structure_Requirements",
        desc="Verify the internship meets structural requirements including duration, field classification, and credential opportunity",
        parent=parent_node,
        critical=True
    )

    # Reference URL existence for program structure
    ref_prog_exist = evaluator.add_custom_node(
        result=_non_empty(sources.program_structure_urls) and _has_vdoe_domain(sources.program_structure_urls),
        id="Reference_URL_Program_Structure_Provided",
        desc="Reference URL from Virginia Department of Education for HQWBL program structure requirements is provided in the answer",
        parent=prog_node,
        critical=True
    )

    # Reference URL content verification (policy page actually documents program structure)
    ref_prog_content = evaluator.add_leaf(
        id="Reference_URL_Program_Structure",
        desc="Provide reference URL from Virginia Department of Education documenting HQWBL program structure requirements",
        parent=prog_node,
        critical=True
    )
    await evaluator.verify(
        claim="This VDOE source documents HQWBL program structure requirements for experiences used to satisfy the Advanced Studies Diploma additional graduation requirement.",
        node=ref_prog_content,
        sources=sources.program_structure_urls,
        additional_instruction="Confirm the page is an official VDOE resource describing HQWBL program structure (e.g., definitions, minimum hours for non-clinical experiences, acceptable options)."
    )

    # Duration provided in the answer (existence gate)
    duration_provided = evaluator.add_custom_node(
        result=bool(details.duration_hours and "hour" in (details.duration_hours or "").lower()),
        id="Duration_Provided_In_Answer",
        desc="The internship duration is explicitly stated in the answer",
        parent=prog_node,
        critical=True
    )

    # Duration verification: 100 hours meets >= 90 hours minimum for non-clinical experiences
    duration_leaf = evaluator.add_leaf(
        id="Duration_Verification",
        desc="Verify the internship duration of 100 hours meets or exceeds the minimum HQWBL requirement of at least 90 hours for non-clinical experiences",
        parent=prog_node,
        critical=True
    )
    stated_hours = details.duration_hours or "100 hours"
    await evaluator.verify(
        claim=f"For non-clinical HQWBL experiences, Virginia requires at least 90 hours. The proposed {stated_hours} internship meets or exceeds this minimum.",
        node=duration_leaf,
        sources=sources.program_structure_urls,
        additional_instruction="Use the VDOE program structure policy page(s) to confirm the ≥90-hour requirement for non-clinical HQWBL experiences. Then affirm that the stated duration meets this threshold."
    )

    # High-demand field verification: cybersecurity via VOEE dashboard
    voee_exist = evaluator.add_custom_node(
        result=_non_empty(sources.voee_urls),
        id="VOEE_URL_Provided",
        desc="VOEE High Demand Occupations Dashboard (or equivalent VOEE reference) URL is provided in the answer",
        parent=prog_node,
        critical=True
    )

    high_demand_leaf = evaluator.add_leaf(
        id="High_Demand_Field_Verification",
        desc="Verify that cybersecurity qualifies as a high-demand field according to the Virginia Office of Education Economics (VOEE) High Demand Occupations Dashboard",
        parent=prog_node,
        critical=True
    )
    await evaluator.verify(
        claim="Cybersecurity (e.g., information security analysts) is identified as a high-demand field in Virginia by the VOEE High Demand Occupations Dashboard.",
        node=high_demand_leaf,
        sources=sources.voee_urls,
        additional_instruction="Allow reasonable synonyms (e.g., 'information security', 'security analysts'). Confirm the VOEE page marks these roles or the cybersecurity domain as high demand in Virginia."
    )

    # Credential opportunity verification: industry-recognized cybersecurity certification available
    credential_exist = evaluator.add_custom_node(
        result=_non_empty(sources.credential_urls) and _contains_certification_text(details.credential_opportunity),
        id="Credential_Opportunity_Provided",
        desc="The answer states an opportunity to earn an industry-recognized cybersecurity certification and cites a supporting source URL",
        parent=prog_node,
        critical=True
    )

    credential_leaf = evaluator.add_leaf(
        id="Credential_Opportunity_Verification",
        desc="Verify the internship provides opportunity to earn an industry-recognized cybersecurity certification (national certificate of completion, industry recognized credential, and/or postsecondary credit)",
        parent=prog_node,
        critical=True
    )
    await evaluator.verify(
        claim="The internship provides an opportunity to earn an industry‑recognized cybersecurity certification as part of the experience (e.g., national certificate, industry credential, or postsecondary credit).",
        node=credential_leaf,
        sources=sources.credential_urls,
        additional_instruction="Confirm the cited source(s) explicitly state that participants can earn an industry-recognized cybersecurity certification or equivalent credential/credit through the internship/program."
    )


async def verify_implementation_and_documentation(
    evaluator: Evaluator,
    parent_node,
    details: InternshipDetails,
    sources: HQWBLSources,
) -> None:
    """
    Build and verify the 'Implementation_and_Documentation' subtree.
    """
    impl_node = evaluator.add_parallel(
        id="Implementation_and_Documentation",
        desc="Verify the internship implementation meets worksite authenticity and documentation requirements",
        parent=parent_node,
        critical=True
    )

    # Reference URL existence for implementation/documentation requirements
    ref_impl_exist = evaluator.add_custom_node(
        result=_non_empty(sources.implementation_urls) and _has_vdoe_domain(sources.implementation_urls),
        id="Reference_URL_Implementation_Provided",
        desc="Reference URL from VDOE for HQWBL implementation and documentation requirements is provided in the answer",
        parent=impl_node,
        critical=True
    )

    # Reference URL content verification
    ref_impl_content = evaluator.add_leaf(
        id="Reference_URL_Implementation",
        desc="Provide reference URL from Virginia Department of Education documenting HQWBL implementation and documentation requirements",
        parent=impl_node,
        critical=True
    )
    await evaluator.verify(
        claim="This VDOE source documents HQWBL implementation and documentation requirements, including authentic worksite definitions and required forms.",
        node=ref_impl_content,
        sources=sources.implementation_urls,
        additional_instruction="Confirm the page is an official VDOE resource detailing HQWBL implementation (e.g., definitions of authentic worksites, supervision expectations) and documentation requirements."
    )

    # Authentic worksite verification
    authentic_leaf = evaluator.add_leaf(
        id="Authentic_Worksite_Verification",
        desc="Verify the technology company with certified information security professional supervision constitutes an authentic worksite (in-person, virtual, or simulated environment administered and supervised by an industry professional or postsecondary subject matter expert)",
        parent=impl_node,
        critical=True
    )
    supervision_text = details.supervision_arrangement or "direct supervision by a certified information security professional"
    await evaluator.verify(
        claim=f"A technology company in Virginia with {supervision_text} qualifies as an authentic worksite under HQWBL (in-person, virtual, or simulated) when administered and supervised by an industry professional or postsecondary subject-matter expert.",
        node=authentic_leaf,
        sources=sources.implementation_urls,
        additional_instruction="Verify that VDOE’s HQWBL implementation policy recognizes industry/professional supervision as authentic worksite oversight for HQWBL, regardless of delivery mode (in-person, virtual, or simulated)."
    )

    # Required documentation subtree
    docs_node = evaluator.add_parallel(
        id="Required_Documentation",
        desc="Verify what documentation must be completed to meet all HQWBL criteria",
        parent=impl_node,
        critical=True
    )

    # Documentation URLs existence gate
    docs_exist = evaluator.add_custom_node(
        result=_non_empty(sources.documentation_urls) or _non_empty(sources.implementation_urls),
        id="Documentation_URLs_Provided",
        desc="Documentation references (e.g., WBL Student Evaluation form) are provided in the answer",
        parent=docs_node,
        critical=True
    )

    # Mentorship documentation verification
    mentor_doc_leaf = evaluator.add_leaf(
        id="Mentorship_Documentation",
        desc="Verify that structured mentorship, supervision, and feedback from the industry professional must be documented on the WBL Student Evaluation form",
        parent=docs_node,
        critical=True
    )
    mentor_sources = sources.documentation_urls if _non_empty(sources.documentation_urls) else sources.implementation_urls
    await evaluator.verify(
        claim="Structured mentorship, supervision, and feedback from the workplace mentor/industry professional must be documented on the VDOE WBL Student Evaluation form.",
        node=mentor_doc_leaf,
        sources=mentor_sources,
        additional_instruction="Confirm the cited VDOE documentation specifies recording mentor supervision/feedback on an official WBL Student Evaluation form (or equivalent)."
    )

    # Five Cs documentation verification
    fivecs_doc_leaf = evaluator.add_leaf(
        id="Five_Cs_Documentation",
        desc="Verify that daily opportunities for students to apply Virginia's 5 Cs (critical thinking, collaboration, communication, creative thinking, and citizenship) must be documented on the WBL Student Evaluation form with attainment confirmed",
        parent=docs_node,
        critical=True
    )
    fivecs_sources = sources.documentation_urls if _non_empty(sources.documentation_urls) else sources.implementation_urls
    await evaluator.verify(
        claim="The VDOE WBL Student Evaluation form requires documenting daily opportunities for students to apply Virginia’s 5 Cs with attainment confirmed.",
        node=fivecs_doc_leaf,
        sources=fivecs_sources,
        additional_instruction="Confirm that the evaluation/documentation materials explicitly reference the 5 Cs and require recording/confirmation of attainment during HQWBL."
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
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate the answer for the Virginia HQWBL Advanced Studies Diploma requirement verification task.
    """
    # Initialize evaluator with a root node (non-critical by design)
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

    # Create a critical top-level node to match rubric root semantics
    main_node = evaluator.add_parallel(
        id="HQWBL_Requirements_Verification",
        desc="Verify that the proposed 100-hour cybersecurity internship meets all mandatory requirements to satisfy the Virginia Advanced Studies Diploma additional graduation requirement through High-Quality Work-Based Learning",
        parent=root,
        critical=True
    )

    # Extract needed information and sources from the answer
    details_task = evaluator.extract(
        prompt=prompt_extract_internship_details(),
        template_class=InternshipDetails,
        extraction_name="internship_details"
    )
    sources_task = evaluator.extract(
        prompt=prompt_extract_hqwbl_sources(),
        template_class=HQWBLSources,
        extraction_name="hqwbl_sources"
    )
    details, sources = await asyncio.gather(details_task, sources_task)

    # Build and verify subtrees
    await verify_program_structure_requirements(evaluator, main_node, details, sources)
    await verify_implementation_and_documentation(evaluator, main_node, details, sources)

    # Return aggregated summary
    return evaluator.get_summary()