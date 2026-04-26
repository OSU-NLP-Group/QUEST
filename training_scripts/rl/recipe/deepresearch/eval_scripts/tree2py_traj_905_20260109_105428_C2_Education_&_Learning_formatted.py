import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "unc_edd_k12_superintendent"
TASK_DESCRIPTION = (
    "A working education professional in North Carolina is seeking a doctoral program to advance to a superintendent position. "
    "They need a program that meets the following requirements: "
    "(1) The program must be offered by an institution in the University of North Carolina system that is accredited by the Southern Association of Colleges and Schools Commission on Colleges (SACSCOC). "
    "(2) The program must be a Doctor of Education (EdD) in Educational Leadership. "
    "(3) The program must offer a concentration specifically in K-12 Leadership. "
    "(4) The program must lead to North Carolina superintendent licensure upon completion. "
    "(5) For admission, the program must require a master's degree from an accredited institution. "
    "(6) For admission, the program must require a minimum graduate GPA of 3.0 or lower on a 4.0 scale. "
    "(7) The program must not require GRE (Graduate Record Examination) scores for admission. "
    "Which UNC system university offers a program that satisfies all of these requirements?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramSourceBundle(BaseModel):
    program_urls: List[str] = Field(default_factory=list)
    admissions_urls: List[str] = Field(default_factory=list)
    licensure_urls: List[str] = Field(default_factory=list)
    accreditation_urls: List[str] = Field(default_factory=list)  # SACSCOC or institution accreditation page(s)
    unc_system_urls: List[str] = Field(default_factory=list)      # UNC System membership page(s)
    institution_urls: List[str] = Field(default_factory=list)     # General institution pages (about, academics, etc.)


class ProgramExtraction(BaseModel):
    institution_name: Optional[str] = None
    program_name: Optional[str] = None
    degree: Optional[str] = None            # e.g., "EdD"
    field: Optional[str] = None             # e.g., "Educational Leadership"
    concentrations: List[str] = Field(default_factory=list)  # e.g., ["K-12 Leadership", "Higher Education"]
    requires_masters_degree_statement: Optional[str] = None  # any text or yes/no stated in the answer
    minimum_gpa_statement: Optional[str] = None              # e.g., "3.0", "2.75", or text
    gre_requirement_statement: Optional[str] = None          # e.g., "GRE not required", "GRE waived", etc.
    sources: ProgramSourceBundle = Field(default_factory=ProgramSourceBundle)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_program() -> str:
    return """
    Extract from the answer the single UNC-system institution and program the user recommended, along with all URLs provided as sources.

    You must strictly extract what is explicitly stated in the answer. Do not invent or infer details not present in the text.

    Return the following fields:
    - institution_name: The name of the university (e.g., "East Carolina University", "UNC Charlotte", etc.)
    - program_name: The program name (e.g., "Doctor of Education (EdD) in Educational Leadership")
    - degree: The degree abbreviation/name (e.g., "EdD")
    - field: The program field/major (e.g., "Educational Leadership")
    - concentrations: List of concentration names stated for this program (e.g., ["K-12 Leadership", "Higher Education"])
    - requires_masters_degree_statement: The statement or indication about requiring a master's degree for admission, as presented in the answer; can be text like "Master's degree required"
    - minimum_gpa_statement: The stated minimum GPA requirement (e.g., "3.0", "2.75 on a 4.0 scale", "3.0 for last 60 credits"), or null if not mentioned
    - gre_requirement_statement: The statement regarding GRE requirement (e.g., "GRE not required", "GRE waived", "GRE optional"), or null if not mentioned

    Also extract all URLs mentioned in the answer and categorize them:
    - sources.program_urls: URLs to the main program page(s) describing the EdD in Educational Leadership and its concentrations
    - sources.admissions_urls: URLs that specifically list admission requirements for the program
    - sources.licensure_urls: URLs explicitly discussing superintendent licensure outcomes for the program
    - sources.accreditation_urls: URLs that show institutional accreditation by SACSCOC (e.g., SACSCOC institutional listing, or the institution's accreditation page)
    - sources.unc_system_urls: URLs that confirm the institution is part of the UNC System (e.g., UNC System "Our Institutions" page or a page stating membership)
    - sources.institution_urls: Any other institution pages cited in the answer (e.g., general "About" page)

    SPECIAL RULES FOR URL EXTRACTION:
    - Extract only URLs present in the answer (includes plain URLs or markdown links).
    - If a URL is missing a protocol, prepend http://
    - Return empty lists if a category has no URLs mentioned.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(*lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    for lst in lists:
        if lst:
            for url in lst:
                if url and isinstance(url, str):
                    u = url.strip()
                    if u and u not in merged:
                        merged.append(u)
    return merged


def _name_or_generic(name: Optional[str], generic_label: str) -> str:
    return name.strip() if (name and name.strip()) else generic_label


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root: Any,
    extracted: ProgramExtraction,
) -> None:
    """
    Build the verification tree and run checks based on the rubric.
    """
    institution = _name_or_generic(extracted.institution_name, "the institution")
    program_name = _name_or_generic(extracted.program_name, "the program")

    # Top-level: Program Identification (critical, parallel)
    program_ident_node = evaluator.add_parallel(
        id="Program_Identification",
        desc="Identify a UNC system university offering an EdD in Educational Leadership with specific characteristics",
        parent=root,
        critical=True
    )

    # Group 1: Institution Verification (critical, parallel)
    institution_ver_node = evaluator.add_parallel(
        id="Institution_Verification",
        desc="Verify the identified institution meets accreditation and system requirements",
        parent=program_ident_node,
        critical=True
    )

    # Add a gating existence node to ensure we have at least some sources to check institution-level claims
    inst_sources_available = evaluator.add_custom_node(
        result=len(_merge_sources(
            extracted.sources.accreditation_urls,
            extracted.sources.unc_system_urls,
            extracted.sources.institution_urls,
        )) > 0,
        id="Institution_Sources_Available",
        desc="Institution-related sources are provided in the answer",
        parent=institution_ver_node,
        critical=True
    )

    # Leaf: SACSCOC Accreditation
    sacs_leaf = evaluator.add_leaf(
        id="SACSCOC_Accreditation",
        desc="The institution is accredited by the Southern Association of Colleges and Schools Commission on Colleges (SACSCOC)",
        parent=institution_ver_node,
        critical=True
    )
    sacs_claim = f"{institution} is accredited by the Southern Association of Colleges and Schools Commission on Colleges (SACSCOC)."
    sacs_sources = _merge_sources(
        extracted.sources.accreditation_urls,
        extracted.sources.institution_urls
    )
    await evaluator.verify(
        claim=sacs_claim,
        node=sacs_leaf,
        sources=sacs_sources,
        additional_instruction=(
            "Verify institutional accreditation specifically by SACSCOC (not programmatic accreditation). "
            "Accept explicit statements on SACSCOC listing pages or the institution's accreditation page clearly stating SACSCOC."
        ),
        extra_prerequisites=[inst_sources_available]
    )

    # Leaf: UNC System Membership
    unc_leaf = evaluator.add_leaf(
        id="UNC_System_Membership",
        desc="The institution is a public university within the University of North Carolina system",
        parent=institution_ver_node,
        critical=True
    )
    unc_claim = f"{institution} is a public university within the University of North Carolina (UNC) system."
    unc_sources = _merge_sources(
        extracted.sources.unc_system_urls,
        extracted.sources.institution_urls
    )
    await evaluator.verify(
        claim=unc_claim,
        node=unc_leaf,
        sources=unc_sources,
        additional_instruction=(
            "Confirm that the institution is one of the UNC System universities (e.g., listed on the UNC System 'Our Institutions' page) "
            "or a credible institution page explicitly stating UNC System membership."
        ),
        extra_prerequisites=[inst_sources_available]
    )

    # Group 2: Program Characteristics (critical, parallel)
    program_char_node = evaluator.add_parallel(
        id="Program_Characteristics",
        desc="Verify the program has the required characteristics and offerings",
        parent=program_ident_node,
        critical=True
    )

    # Gating existence: program pages exist
    prog_sources_available = evaluator.add_custom_node(
        result=len(_merge_sources(extracted.sources.program_urls)) > 0,
        id="Program_Sources_Available",
        desc="Program pages are provided in the answer",
        parent=program_char_node,
        critical=True
    )

    # Leaf: Degree and Field = EdD in Educational Leadership
    degree_leaf = evaluator.add_leaf(
        id="Degree_and_Field",
        desc="The program is a Doctor of Education (EdD) in Educational Leadership or equivalent field",
        parent=program_char_node,
        critical=True
    )
    degree_claim = (
        f"{program_name} is a Doctor of Education (EdD) in Educational Leadership."
    )
    degree_sources = _merge_sources(extracted.sources.program_urls)
    await evaluator.verify(
        claim=degree_claim,
        node=degree_leaf,
        sources=degree_sources,
        additional_instruction=(
            "Verify the degree is a Doctor of Education (EdD) and the program field is Educational Leadership. "
            "Minor naming variants like 'Educational Leadership & Administration' or 'Educational Leadership (K-12)' are acceptable as equivalent field labels if clearly within Educational Leadership."
        ),
        extra_prerequisites=[prog_sources_available]
    )

    # Leaf: K-12 Leadership Concentration
    k12_leaf = evaluator.add_leaf(
        id="K12_Leadership_Concentration",
        desc="The program offers a concentration in K-12 Leadership",
        parent=program_char_node,
        critical=True
    )
    k12_claim = (
        f"{program_name} offers a concentration specifically in K-12 Leadership."
    )
    k12_sources = _merge_sources(extracted.sources.program_urls)
    await evaluator.verify(
        claim=k12_claim,
        node=k12_leaf,
        sources=k12_sources,
        additional_instruction=(
            "Confirm the program offers a concentration named 'K-12 Leadership'. "
            "Accept variants such as 'K–12 Leadership', 'PK–12 Leadership', 'PreK–12 Leadership', or 'P–12 Leadership' if clearly equivalent."
        ),
        extra_prerequisites=[prog_sources_available]
    )

    # Leaf: Superintendent Licensure
    lic_leaf = evaluator.add_leaf(
        id="Superintendent_Licensure",
        desc="The program leads to North Carolina superintendent licensure",
        parent=program_char_node,
        critical=True
    )
    lic_claim = (
        f"Completion of {program_name} leads to North Carolina superintendent licensure."
    )
    lic_sources = _merge_sources(
        extracted.sources.licensure_urls,
        extracted.sources.program_urls
    )
    await evaluator.verify(
        claim=lic_claim,
        node=lic_leaf,
        sources=lic_sources,
        additional_instruction=(
            "Verify that the program explicitly states it leads to or qualifies graduates for North Carolina superintendent licensure. "
            "Accept statements that clearly indicate eligibility or licensure recommendation for NC superintendent credentials."
        ),
        extra_prerequisites=[prog_sources_available]
    )

    # Group 3: Admission Requirements (critical, parallel)
    admission_node = evaluator.add_parallel(
        id="Admission_Requirements",
        desc="Verify the program's admission requirements match the specified criteria",
        parent=program_ident_node,
        critical=True
    )

    # Gating existence: admissions sources exist
    adm_sources_available = evaluator.add_custom_node(
        result=len(_merge_sources(extracted.sources.admissions_urls, extracted.sources.program_urls)) > 0,
        id="Admission_Sources_Available",
        desc="Admissions requirements sources are provided in the answer",
        parent=admission_node,
        critical=True
    )

    # Leaf: Master's degree required
    masters_leaf = evaluator.add_leaf(
        id="Masters_Degree_Required",
        desc="The program requires applicants to hold a master's degree from an accredited institution",
        parent=admission_node,
        critical=True
    )
    masters_claim = (
        f"Admission to {program_name} requires applicants to hold a master's degree from an accredited institution."
    )
    masters_sources = _merge_sources(extracted.sources.admissions_urls, extracted.sources.program_urls)
    await evaluator.verify(
        claim=masters_claim,
        node=masters_leaf,
        sources=masters_sources,
        additional_instruction=(
            "Confirm that a master's degree is required for admission. "
            "Look for explicit language such as 'master's degree required' or equivalent."
        ),
        extra_prerequisites=[adm_sources_available]
    )

    # Leaf: GPA ≥ requirement (3.0 or lower)
    gpa_leaf = evaluator.add_leaf(
        id="GPA_Requirement",
        desc="The program requires a minimum graduate GPA of 3.0 or lower on a 4.0 scale",
        parent=admission_node,
        critical=True
    )
    gpa_claim = (
        f"The minimum required graduate GPA for admission to {program_name} is 3.0 on a 4.0 scale or lower."
    )
    gpa_sources = _merge_sources(extracted.sources.admissions_urls, extracted.sources.program_urls)
    await evaluator.verify(
        claim=gpa_claim,
        node=gpa_leaf,
        sources=gpa_sources,
        additional_instruction=(
            "Support this claim if the page states a minimum graduate GPA requirement of 3.0 (on 4.0 scale) or a lower threshold (e.g., 2.75). "
            "References to last 60 credits at 3.0 also count if clearly stated as the minimum."
        ),
        extra_prerequisites=[adm_sources_available]
    )

    # Leaf: GRE not required
    gre_leaf = evaluator.add_leaf(
        id="GRE_Waived",
        desc="The program does not require GRE scores for admission",
        parent=admission_node,
        critical=True
    )
    gre_claim = (
        f"GRE scores are not required for admission to {program_name}."
    )
    gre_sources = _merge_sources(extracted.sources.admissions_urls, extracted.sources.program_urls)
    await evaluator.verify(
        claim=gre_claim,
        node=gre_leaf,
        sources=gre_sources,
        additional_instruction=(
            "Confirm that GRE is not required. Accept statements like 'GRE not required', 'GRE waived', or 'GRE optional'. "
            "If GRE is required under certain exceptions only (e.g., low GPA), the general statement 'GRE not required' should not be considered supported unless the page clearly conveys general non-requirement."
        ),
        extra_prerequisites=[adm_sources_available]
    )

    # Record a small custom info block for convenience
    evaluator.add_custom_info(
        info={
            "institution_name": extracted.institution_name,
            "program_name": extracted.program_name,
            "degree": extracted.degree,
            "field": extracted.field,
            "concentrations": extracted.concentrations,
            "source_counts": {
                "program_urls": len(extracted.sources.program_urls),
                "admissions_urls": len(extracted.sources.admissions_urls),
                "licensure_urls": len(extracted.sources.licensure_urls),
                "accreditation_urls": len(extracted.sources.accreditation_urls),
                "unc_system_urls": len(extracted.sources.unc_system_urls),
                "institution_urls": len(extracted.sources.institution_urls),
            }
        },
        info_type="extracted_summary",
        info_name="extracted_program_summary"
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
    Evaluate the agent's answer for the UNC EdD in Educational Leadership with K-12 Leadership concentration
    leading to NC superintendent licensure and specific admissions requirements.
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
        default_model=model,
    )

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_program(),
        template_class=ProgramExtraction,
        extraction_name="program_extraction",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()