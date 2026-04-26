import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ncaa_d2_assistant_football_requirements_checklist"
TASK_DESCRIPTION = (
    "You are a high school football coach in Ohio with 3 years of varsity coaching experience and OHSAA certification. "
    "You want to transition to college coaching and are interested in applying for assistant football coaching positions "
    "at NCAA Division II programs in the Northeast region (such as those in the Northeast-10 Conference). "
    "Prepare a comprehensive requirements checklist to guide your application preparation. Your checklist must include three sections: "
    "(1) Minimum Qualifications - Document the educational credentials and coaching experience typically required for Division II assistant coaching positions, "
    "including both the specific degree requirement and the typical years of experience needed; "
    "(2) Required Certifications and Clearances - Document the mandatory certifications and background clearances you must obtain or verify before being eligible for hire, "
    "including specific details about safety certifications and screening requirements; "
    "(3) Application Materials - Document the standard materials you need to prepare for your application package, including specific details about documentation requirements. "
    "For each section, provide specific requirement details and at least one reference URL from your research that supports the requirements you've documented."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MinimumQualificationsExtraction(BaseModel):
    """Information extracted for minimum qualifications."""
    education_requirement_text: Optional[str] = None
    education_requirement_urls: List[str] = Field(default_factory=list)
    experience_requirement_text: Optional[str] = None
    experience_requirement_urls: List[str] = Field(default_factory=list)


class CertsClearancesExtraction(BaseModel):
    """Information extracted for certifications and clearances."""
    cpr_first_aid_text: Optional[str] = None
    cpr_first_aid_urls: List[str] = Field(default_factory=list)
    background_check_text: Optional[str] = None
    background_check_urls: List[str] = Field(default_factory=list)


class ApplicationMaterialsExtraction(BaseModel):
    """Information extracted for application materials."""
    resume_text: Optional[str] = None
    resume_urls: List[str] = Field(default_factory=list)
    transcripts_text: Optional[str] = None
    transcripts_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_minimum_qualifications() -> str:
    return (
        "Extract from the answer the details and reference URLs related to Minimum Qualifications for NCAA Division II assistant football coaching positions.\n"
        "Return a JSON object with the following fields:\n"
        "1) education_requirement_text: The exact sentence(s) from the answer that state a Bachelor's degree from an accredited institution is required.\n"
        "2) education_requirement_urls: An array of URLs cited in the answer that specifically support the Bachelor's degree requirement.\n"
        "3) experience_requirement_text: The exact sentence(s) from the answer that state the typical experience requirement: 2–5 years of coaching experience, "
        "and that qualifying experience can be collegiate, high school varsity, or professional.\n"
        "4) experience_requirement_urls: An array of URLs cited in the answer that support the typical 2–5 years requirement and/or the acceptable experience levels.\n\n"
        "Rules:\n"
        "- Only extract text that appears in the answer. Do not invent or paraphrase beyond what is explicitly stated.\n"
        "- For URLs, extract only actual URLs (including those in markdown links) explicitly present in the answer; ignore non-URL references.\n"
        "- If any field is not present, set it to null; if no URLs are present for a requirement, return an empty array for that URLs field."
    )


def prompt_extract_certs_clearances() -> str:
    return (
        "Extract from the answer the details and reference URLs related to Required Certifications and Clearances.\n"
        "Return a JSON object with the following fields:\n"
        "1) cpr_first_aid_text: The exact sentence(s) stating that valid/current CPR and First Aid certification is mandatory.\n"
        "2) cpr_first_aid_urls: An array of URLs cited in the answer that support the CPR/First Aid certification requirement.\n"
        "3) background_check_text: The exact sentence(s) stating that a criminal background check is required and includes the typical completion time of 2–7 business days "
        "and that it is performed through NCAA-approved vendors (or institution/athletics-approved vendors aligned with NCAA compliance practices).\n"
        "4) background_check_urls: An array of URLs cited in the answer that support the background check requirement and the included screening details.\n\n"
        "Rules:\n"
        "- Only extract text explicitly present in the answer.\n"
        "- Extract only valid URLs that are explicitly cited.\n"
        "- If any field is not present, set it to null; if no URLs are present for a requirement, return an empty array."
    )


def prompt_extract_application_materials() -> str:
    return (
        "Extract from the answer the details and reference URLs related to Application Materials.\n"
        "Return a JSON object with the following fields:\n"
        "1) resume_text: The exact sentence(s) stating that a professional coaching resume highlighting experience/accomplishments/qualifications must be submitted.\n"
        "2) resume_urls: An array of URLs cited in the answer that support the resume submission requirement.\n"
        "3) transcripts_text: The exact sentence(s) stating that official transcripts showing degree completion must be submitted as part of the application package.\n"
        "4) transcripts_urls: An array of URLs cited in the answer that support the transcripts submission requirement.\n\n"
        "Rules:\n"
        "- Only extract text explicitly present in the answer.\n"
        "- Extract only valid URLs explicitly cited in the answer.\n"
        "- If any field is not present, set it to null; if no URLs are present for a requirement, return an empty array."
    )


# --------------------------------------------------------------------------- #
# Helper: verify supporting URLs (enforce non-empty list)                     #
# --------------------------------------------------------------------------- #
async def _verify_supporting_urls(
    evaluator: Evaluator,
    claim: str,
    node,
    urls: List[str],
    additional_instruction: str
) -> bool:
    """
    Verify that at least one of the provided URLs supports the claim.
    If URLs list is empty, directly fail the node (since the rubric requires ≥1 supporting URL).
    """
    if not urls:
        node.score = 0.0
        node.status = "failed"
        return False
    return await evaluator.verify(
        claim=claim,
        node=node,
        sources=urls,
        additional_instruction=additional_instruction
    )


# --------------------------------------------------------------------------- #
# Verification functions for each section                                     #
# --------------------------------------------------------------------------- #
async def verify_minimum_qualifications(
    evaluator: Evaluator,
    parent_node,
    extracted: MinimumQualificationsExtraction
) -> None:
    """
    Build and verify the 'Minimum Qualifications' section.
    """
    section_node = evaluator.add_parallel(
        id="Minimum_Qualifications_Section",
        desc="Minimum Qualifications section is present and documents required education and typical experience.",
        parent=parent_node,
        critical=True
    )

    # Educational Requirement
    edu_node = evaluator.add_parallel(
        id="Educational_Requirement",
        desc="Documents the educational requirement: Bachelor's degree from an accredited institution.",
        parent=section_node,
        critical=True
    )

    edu_detail_leaf = evaluator.add_leaf(
        id="Education_Detail_Documented",
        desc="States that a Bachelor's degree from an accredited institution is required.",
        parent=edu_node,
        critical=True
    )
    edu_detail_claim = (
        "The checklist explicitly states that a Bachelor's degree from an accredited institution is required for NCAA Division II assistant football coaching positions."
    )
    await evaluator.verify(
        claim=edu_detail_claim,
        node=edu_detail_leaf,
        additional_instruction="Judge this solely based on the answer text. Accept equivalent phrasing like 'bachelor’s degree required' or 'degree from an accredited university'."
    )

    edu_urls_leaf = evaluator.add_leaf(
        id="Education_Supporting_URL_Provided",
        desc="Provides ≥1 reference URL supporting the Bachelor's degree requirement.",
        parent=edu_node,
        critical=True
    )
    edu_urls_claim = (
        "NCAA Division II assistant football coaching job postings typically require a Bachelor's degree from an accredited institution."
    )
    await _verify_supporting_urls(
        evaluator,
        claim=edu_urls_claim,
        node=edu_urls_leaf,
        urls=extracted.education_requirement_urls if extracted.education_requirement_urls else [],
        additional_instruction=(
            "Verify that at least one cited URL (e.g., institutional HR/job postings or NCAA-related hiring pages) explicitly supports the bachelor's degree requirement."
        )
    )

    # Experience Requirement
    exp_node = evaluator.add_parallel(
        id="Experience_Requirement",
        desc="Documents the typical coaching experience requirement (range and acceptable levels).",
        parent=section_node,
        critical=True
    )

    exp_detail_leaf = evaluator.add_leaf(
        id="Experience_Detail_Documented",
        desc="States that typically 2–5 years of coaching experience is required and that qualifying experience can be collegiate, high school varsity, or professional.",
        parent=exp_node,
        critical=True
    )
    exp_detail_claim = (
        "The checklist explicitly states that typically 2–5 years of coaching experience is required and that qualifying experience can be collegiate, high school varsity, or professional."
    )
    await evaluator.verify(
        claim=exp_detail_claim,
        node=exp_detail_leaf,
        additional_instruction="Judge based on the answer text; allow minor phrasing variations for the 2–5 year range and acceptable experience levels."
    )

    exp_urls_leaf = evaluator.add_leaf(
        id="Experience_Supporting_URL_Provided",
        desc="Provides ≥1 reference URL supporting the typical 2–5 years experience requirement (and/or the acceptable experience levels).",
        parent=exp_node,
        critical=True
    )
    exp_urls_claim = (
        "Division II assistant football coaching job postings typically require around 2–5 years of prior coaching experience; acceptable experience includes collegiate, high school varsity, or professional levels."
    )
    await _verify_supporting_urls(
        evaluator,
        claim=exp_urls_claim,
        node=exp_urls_leaf,
        urls=extracted.experience_requirement_urls if extracted.experience_requirement_urls else [],
        additional_instruction=(
            "Confirm from at least one cited job posting or HR page that the typical experience requirement is ~2–5 years and/or that collegiate, HS varsity, or professional coaching experience is accepted."
        )
    )


async def verify_certs_clearances(
    evaluator: Evaluator,
    parent_node,
    extracted: CertsClearancesExtraction
) -> None:
    """
    Build and verify the 'Required Certifications and Clearances' section.
    """
    section_node = evaluator.add_parallel(
        id="Required_Certifications_and_Clearances_Section",
        desc="Required Certifications and Clearances section is present and documents mandatory safety certifications and screening/clearance requirements.",
        parent=parent_node,
        critical=True
    )

    # CPR/First Aid Certification
    cpr_node = evaluator.add_parallel(
        id="CPR_First_Aid_Certification",
        desc="Documents the mandatory CPR/First Aid certification requirement.",
        parent=section_node,
        critical=True
    )

    cpr_detail_leaf = evaluator.add_leaf(
        id="CPR_FirstAid_Detail_Documented",
        desc="States that valid/current CPR and First Aid certification is mandatory.",
        parent=cpr_node,
        critical=True
    )
    cpr_detail_claim = "The checklist states that valid/current CPR and First Aid certification is mandatory."
    await evaluator.verify(
        claim=cpr_detail_claim,
        node=cpr_detail_leaf,
        additional_instruction="Judge based on the answer text; allow equivalent naming like 'First Aid/CPR certification' or 'CPR & First Aid'."
    )

    cpr_urls_leaf = evaluator.add_leaf(
        id="CPR_FirstAid_Supporting_URL_Provided",
        desc="Provides ≥1 reference URL supporting the CPR/First Aid certification requirement.",
        parent=cpr_node,
        critical=True
    )
    cpr_urls_claim = "Assistant football coach job postings typically require current CPR and First Aid certifications."
    await _verify_supporting_urls(
        evaluator,
        claim=cpr_urls_claim,
        node=cpr_urls_leaf,
        urls=extracted.cpr_first_aid_urls if extracted.cpr_first_aid_urls else [],
        additional_instruction="Verify at least one cited posting or HR page clearly requires or mandates current CPR and First Aid certification."
    )

    # Background Check Requirement
    bg_node = evaluator.add_parallel(
        id="Background_Check_Requirement",
        desc="Documents the mandatory criminal background check requirement with required constraint-level specifics.",
        parent=section_node,
        critical=True
    )

    bg_detail_leaf = evaluator.add_leaf(
        id="BackgroundCheck_Detail_Documented",
        desc="States that a criminal background check is required and includes: typical completion time of 2–7 business days and that it is done through NCAA-approved vendors.",
        parent=bg_node,
        critical=True
    )
    bg_detail_claim = (
        "The checklist states that a criminal background check is required and includes the typical completion time of 2–7 business days and that the screening is performed through NCAA-approved vendors."
    )
    await evaluator.verify(
        claim=bg_detail_claim,
        node=bg_detail_leaf,
        additional_instruction=(
            "Judge based on the answer text. Accept equivalent phrasing such as 'institution or athletics department approved vendors aligned with NCAA compliance' "
            "for 'NCAA-approved vendors'. The statement must include both the background check requirement and the 2–7 business days timeframe."
        )
    )

    bg_urls_leaf = evaluator.add_leaf(
        id="BackgroundCheck_Supporting_URL_Provided",
        desc="Provides ≥1 reference URL supporting the background check requirement (including the screening requirement details used).",
        parent=bg_node,
        critical=True
    )
    bg_urls_claim = (
        "Assistant coach hiring requires a criminal background check; sources indicate the requirement and include either the typical completion timeframe (about 2–7 business days) and/or that screening is handled by NCAA-approved or institution-approved vendors."
    )
    await _verify_supporting_urls(
        evaluator,
        claim=bg_urls_claim,
        node=bg_urls_leaf,
        urls=extracted.background_check_urls if extracted.background_check_urls else [],
        additional_instruction=(
            "Verify that at least one cited URL supports the background check requirement and mentions either the 2–7 business day timeframe or vendor/approved screening details consistent with NCAA compliance practices."
        )
    )


async def verify_application_materials(
    evaluator: Evaluator,
    parent_node,
    extracted: ApplicationMaterialsExtraction
) -> None:
    """
    Build and verify the 'Application Materials' section.
    """
    section_node = evaluator.add_parallel(
        id="Application_Materials_Section",
        desc="Application Materials section is present and documents standard materials to prepare, including documentation requirements.",
        parent=parent_node,
        critical=True
    )

    # Resume Requirement
    resume_node = evaluator.add_parallel(
        id="Resume_Requirement",
        desc="Documents that a professional coaching resume must be submitted.",
        parent=section_node,
        critical=True
    )

    resume_detail_leaf = evaluator.add_leaf(
        id="Resume_Detail_Documented",
        desc="States that a professional coaching resume highlighting experience/accomplishments/qualifications must be submitted.",
        parent=resume_node,
        critical=True
    )
    resume_detail_claim = (
        "The checklist states that a professional coaching resume highlighting experience, accomplishments, and qualifications must be submitted."
    )
    await evaluator.verify(
        claim=resume_detail_claim,
        node=resume_detail_leaf,
        additional_instruction="Judge based on the answer text; treat 'coaching CV' or 'curriculum vitae' as equivalent to 'resume'."
    )

    resume_urls_leaf = evaluator.add_leaf(
        id="Resume_Supporting_URL_Provided",
        desc="Provides ≥1 reference URL supporting the resume submission requirement.",
        parent=resume_node,
        critical=True
    )
    resume_urls_claim = "Assistant coach application instructions require submitting a resume or CV."
    await _verify_supporting_urls(
        evaluator,
        claim=resume_urls_claim,
        node=resume_urls_leaf,
        urls=extracted.resume_urls if extracted.resume_urls else [],
        additional_instruction="Verify that at least one cited URL clearly lists a resume/CV as a required application material for assistant coach positions."
    )

    # Transcripts Requirement
    transcripts_node = evaluator.add_parallel(
        id="Transcripts_Requirement",
        desc="Documents that official transcripts proving degree completion must be submitted.",
        parent=section_node,
        critical=True
    )

    transcripts_detail_leaf = evaluator.add_leaf(
        id="Transcripts_Detail_Documented",
        desc="States that official transcripts showing degree completion must be submitted as part of the application package.",
        parent=transcripts_node,
        critical=True
    )
    transcripts_detail_claim = (
        "The checklist states that official transcripts showing degree completion must be submitted as part of the application package."
    )
    await evaluator.verify(
        claim=transcripts_detail_claim,
        node=transcripts_detail_leaf,
        additional_instruction="Judge based on the answer text; allow equivalent phrasing like 'official academic transcripts required to verify degree completion'."
    )

    transcripts_urls_leaf = evaluator.add_leaf(
        id="Transcripts_Supporting_URL_Provided",
        desc="Provides ≥1 reference URL supporting the transcripts submission requirement.",
        parent=transcripts_node,
        critical=True
    )
    transcripts_urls_claim = "Assistant coach application instructions list official academic transcripts as a required document."
    await _verify_supporting_urls(
        evaluator,
        claim=transcripts_urls_claim,
        node=transcripts_urls_leaf,
        urls=extracted.transcripts_urls if extracted.transcripts_urls else [],
        additional_instruction="Verify that at least one cited URL clearly requires official transcripts (or equivalent documentation) as part of the application process."
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
    Evaluate the answer against the NCAA Division II assistant football coaching requirements checklist rubric.
    """
    # Initialize evaluator
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

    # Create the top-level critical node (as per rubric root requirements)
    top_node = evaluator.add_parallel(
        id="Requirements_Checklist_Complete",
        desc="Comprehensive requirements checklist with the three required sections for NCAA Division II assistant football coaching applications, covering all stated constraints.",
        parent=root,
        critical=True
    )

    # Extract all sections concurrently
    minq_task = evaluator.extract(
        prompt=prompt_extract_minimum_qualifications(),
        template_class=MinimumQualificationsExtraction,
        extraction_name="minimum_qualifications"
    )
    certs_task = evaluator.extract(
        prompt=prompt_extract_certs_clearances(),
        template_class=CertsClearancesExtraction,
        extraction_name="certifications_clearances"
    )
    materials_task = evaluator.extract(
        prompt=prompt_extract_application_materials(),
        template_class=ApplicationMaterialsExtraction,
        extraction_name="application_materials"
    )

    minq, certs, materials = await asyncio.gather(minq_task, certs_task, materials_task)

    # Build and verify each section under the top-level critical node
    await verify_minimum_qualifications(evaluator, top_node, minq)
    await verify_certs_clearances(evaluator, top_node, certs)
    await verify_application_materials(evaluator, top_node, materials)

    # Return summary
    return evaluator.get_summary()