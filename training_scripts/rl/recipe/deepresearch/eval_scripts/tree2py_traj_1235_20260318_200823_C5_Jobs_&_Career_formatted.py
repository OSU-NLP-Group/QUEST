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
TASK_ID = "career_services_standards"
TASK_DESCRIPTION = (
    "A public university in Ohio is updating its Career Center hiring guidelines and professional development "
    "framework for staff. The Human Resources department needs to document industry-standard minimum qualifications "
    "for two key positions in the career services hierarchy (Career Services Coordinator and Assistant Director of "
    "Career Services), as well as understand the requirements for nationally recognized professional certifications "
    "that career services staff can pursue for professional development.\n\n"
    "Based on standards from leading professional associations such as NACE (National Association of Colleges and "
    "Employers) and NCDA (National Career Development Association), provide the following information:\n\n"
    "1. What are the standard minimum education and experience requirements for a Career Services Coordinator position "
    "in higher education?\n\n"
    "2. What are the standard minimum education and experience requirements for an Assistant Director of Career "
    "Services position? (Include alternative qualification pathways if they exist.)\n\n"
    "3. What are the complete requirements for obtaining the GCDF (Global Career Development Facilitator) certification, "
    "including both training hours and experience hours required based on different education levels?\n\n"
    "4. What are the complete requirements for obtaining the NCDA Certified Career Counselor (CCC) credential, including "
    "the education requirement and the available options for fulfilling the career specialization requirement?\n\n"
    "5. What is the name of the leading professional association that connects college career services professionals "
    "and employers in the United States?\n\n"
    "6. What coaching certification program options does this professional association offer to help career services "
    "professionals pursue advanced credentials?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CoordinatorInfo(BaseModel):
    education: Optional[str] = None
    experience: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AssistantDirectorInfo(BaseModel):
    education_options: List[str] = Field(default_factory=list)  # e.g., ["Master's + 3 yrs", "Bachelor's + 5 yrs"]
    degree_field: Optional[str] = None  # e.g., "counseling, higher education, student affairs, or related"
    sources: List[str] = Field(default_factory=list)


class GCDFInfo(BaseModel):
    training_hours: Optional[str] = None  # e.g., "120 hours from a CCE Registered Credential Training Provider"
    experience_hours_by_education: List[str] = Field(default_factory=list)  # e.g., ["Graduate: 1400", "Bachelor's: 2800", ...]
    sources: List[str] = Field(default_factory=list)


class CCCInfo(BaseModel):
    education: Optional[str] = None  # e.g., "Master's degree or higher in counselor education, ..."
    career_specialization_options: List[str] = Field(default_factory=list)  # e.g., ["600 hours supervised...", "60 hours CE...", "NCDA FCD course"]
    sources: List[str] = Field(default_factory=list)


class AssociationInfo(BaseModel):
    name: Optional[str] = None  # e.g., "National Association of Colleges and Employers (NACE)"
    role: Optional[str] = None  # e.g., "connects college career services professionals and employers"
    network_size: Optional[str] = None  # e.g., "over 17,600 professionals" or a number-like string
    sources: List[str] = Field(default_factory=list)


class NACECoachingInfo(BaseModel):
    tracks: List[str] = Field(default_factory=list)  # e.g., ["30-hour", "60-hour"]
    bcc_preparation: Optional[str] = None  # e.g., "prepares candidates for BCC via CCE"
    sources: List[str] = Field(default_factory=list)


class StandardsExtraction(BaseModel):
    coordinator: Optional[CoordinatorInfo] = None
    assistant_director: Optional[AssistantDirectorInfo] = None
    gcdf: Optional[GCDFInfo] = None
    ccc: Optional[CCCInfo] = None
    association: Optional[AssociationInfo] = None
    coaching_program: Optional[NACECoachingInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_standards() -> str:
    return """
    Extract structured information from the answer for the following items. Only extract what is explicitly present
    in the answer. When extracting URLs, include only full, valid URLs that are explicitly written in the answer
    (plain links or markdown links), not inferred references.

    Required JSON schema:

    {
      "coordinator": {
        "education": string | null,
        "experience": string | null,
        "sources": string[]    // all URLs cited in the answer that support coordinator minimum qualifications
      },
      "assistant_director": {
        "education_options": string[],  // e.g., "Master's + 3 years", "Bachelor's + 5 years"
        "degree_field": string | null,  // e.g., "counseling, higher education, student affairs, or related"
        "sources": string[]             // URLs supporting assistant director minimum qualifications
      },
      "gcdf": {
        "training_hours": string | null,  // e.g., "120 hours from a CCE Registered Credential Training Provider"
        "experience_hours_by_education": string[], // e.g., ["Graduate: 1400", "Bachelor's: 2800", "Associate: 4200", "High School: 5600"]
        "sources": string[]               // URLs supporting GCDF requirements (ideally CCE/GCDF official pages)
      },
      "ccc": {
        "education": string | null,  // e.g., "Master's degree or higher in counselor education or closely related"
        "career_specialization_options": string[], // list each distinct option verbatim as stated
        "sources": string[]           // URLs supporting NCDA CCC requirements
      },
      "association": {
        "name": string | null,   // e.g., "National Association of Colleges and Employers (NACE)"
        "role": string | null,   // e.g., "connects college career services professionals and employers"
        "network_size": string | null, // e.g., "over 17,600 professionals", "17,600+", "more than 17,600"
        "sources": string[]      // URLs supporting the association identification and stats
      },
      "coaching_program": {
        "tracks": string[],           // e.g., ["30-hour", "60-hour"]
        "bcc_preparation": string | null, // e.g., "prepares for BCC through CCE"
        "sources": string[]           // URLs supporting NACE Coaching Certification Program details
      }
    }

    Additional rules:
    - Do not invent data. If any field is not stated in the answer, set it to null (for strings) or [] (for lists).
    - For sources arrays, include only URLs that are specifically tied to the corresponding section in the answer.
    - Preserve units and numeric values exactly as shown in the answer when populating strings (e.g., "120 hours").
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_coordinator_requirements(evaluator: Evaluator, parent_node, info: Optional[CoordinatorInfo]) -> None:
    node = evaluator.add_parallel(
        id="Coordinator_Requirements",
        desc="Minimum qualifications for Career Services Coordinator position",
        parent=parent_node,
        critical=False
    )

    sources = (info.sources if info else []) or []

    # Education leaf
    edu_node = evaluator.add_leaf(
        id="Coordinator_Education",
        desc="Requires a bachelor's degree in a relevant field such as Counseling, Education, Business, or related area",
        parent=node,
        critical=True
    )
    edu_claim = (
        "Industry-standard minimum education for a Career Services Coordinator in higher education is a bachelor's "
        "degree in a relevant field (e.g., counseling, education, business, or a closely related discipline)."
    )
    await evaluator.verify(
        claim=edu_claim,
        node=edu_node,
        sources=sources,
        additional_instruction=(
            "Verify that the cited page(s) indicate a bachelor's degree is the typical minimum for a Career Services "
            "Coordinator (or similar entry/intermediate career services role) in higher education. Allow phrasing like "
            "'Bachelor's required; Master's preferred' to count as supporting a bachelor's minimum."
        )
    )

    # Experience leaf
    exp_node = evaluator.add_leaf(
        id="Coordinator_Experience",
        desc="Requires a minimum of 2–3 years of experience in career services or a related field",
        parent=node,
        critical=True
    )
    exp_claim = (
        "The standard minimum experience for a Career Services Coordinator is in the 2–3 years range in career services "
        "or a related student services/employer relations field."
    )
    await evaluator.verify(
        claim=exp_claim,
        node=exp_node,
        sources=sources,
        additional_instruction=(
            "Confirm that the page(s) specify a minimum of about two to three years of relevant experience for a "
            "Career Services Coordinator or an equivalent role. Accept close variants such as '2 years minimum' or "
            "'2-3 years preferred/required' as support."
        )
    )


async def verify_assistant_director_requirements(evaluator: Evaluator, parent_node, info: Optional[AssistantDirectorInfo]) -> None:
    node = evaluator.add_parallel(
        id="Assistant_Director_Requirements",
        desc="Minimum qualifications for Assistant Director of Career Services position (including alternative pathways)",
        parent=parent_node,
        critical=False
    )

    sources = (info.sources if info else []) or []

    # Education options leaf (alternative pathways)
    edu_opts_node = evaluator.add_leaf(
        id="Assistant_Director_Education_Options",
        desc="Alternative qualification pathways: either (a) Master's degree with 3 years of related experience, OR (b) Bachelor's degree with 5 years of related experience",
        parent=node,
        critical=True
    )
    edu_opts_claim = (
        "For an Assistant Director of Career Services, industry-standard minimum pathways commonly include either: "
        "(a) a master's degree with approximately 3 years of relevant experience, or (b) a bachelor's degree with "
        "approximately 5 years of relevant experience."
    )
    await evaluator.verify(
        claim=edu_opts_claim,
        node=edu_opts_node,
        sources=sources,
        additional_instruction=(
            "Validate that credible sources (e.g., professional associations, representative university HR/job standards) "
            "explicitly support the two alternative minimum pathways: Master's + ~3 years OR Bachelor's + ~5 years. "
            "Minor variations in wording are acceptable if clearly equivalent in level and years."
        )
    )

    # Degree field appropriateness leaf
    field_node = evaluator.add_leaf(
        id="Assistant_Director_Field",
        desc="Degree field must be appropriate (e.g., counseling, higher education, student affairs, or related areas)",
        parent=node,
        critical=True
    )
    field_claim = (
        "Assistant Director of Career Services minimum qualifications specify an appropriate degree field such as "
        "counseling, higher education/student affairs, or a closely related area."
    )
    await evaluator.verify(
        claim=field_claim,
        node=field_node,
        sources=sources,
        additional_instruction=(
            "Look for language indicating degree fields commonly required/accepted for this role, including counseling, "
            "higher education, student affairs, or related areas. Synonyms or near-equivalents are acceptable."
        )
    )


async def verify_gcdf_requirements(evaluator: Evaluator, parent_node, info: Optional[GCDFInfo]) -> None:
    node = evaluator.add_parallel(
        id="GCDF_Certification",
        desc="Complete requirements for Global Career Development Facilitator (GCDF) certification",
        parent=parent_node,
        critical=False
    )

    sources = (info.sources if info else []) or []

    # Training hours leaf
    training_node = evaluator.add_leaf(
        id="GCDF_Training",
        desc="Requires a minimum of 120 hours of comprehensive training from a CCE Registered Credential Training Provider",
        parent=node,
        critical=True
    )
    training_claim = (
        "The GCDF credential requires a minimum of 120 hours of comprehensive training delivered by a CCE Registered "
        "Credential Training Provider (RCTP)."
    )
    await evaluator.verify(
        claim=training_claim,
        node=training_node,
        sources=sources,
        additional_instruction=(
            "Use official GCDF/CCE sources where possible. Confirm that 120 training hours and the RCTP requirement are "
            "explicitly stated."
        )
    )

    # Experience hours by education level leaf
    exp_node = evaluator.add_leaf(
        id="GCDF_Experience",
        desc="Requires career development work experience hours that vary by education level: 1,400 (graduate), 2,800 (bachelor's), 4,200 (associate), or 5,600 (high school diploma)",
        parent=node,
        critical=True
    )
    exp_claim = (
        "GCDF experience requirements vary by education level as follows: 1,400 hours with a graduate degree; 2,800 hours "
        "with a bachelor's degree; 4,200 hours with an associate degree; or 5,600 hours with a high school diploma."
    )
    await evaluator.verify(
        claim=exp_claim,
        node=exp_node,
        sources=sources,
        additional_instruction=(
            "Rely on GCDF/CCE official requirement pages. Minor formatting differences (commas, wording) are acceptable, "
            "but the numeric hours by education level must match."
        )
    )


async def verify_ccc_requirements(evaluator: Evaluator, parent_node, info: Optional[CCCInfo]) -> None:
    node = evaluator.add_parallel(
        id="NCDA_CCC_Certification",
        desc="Complete requirements for NCDA Certified Career Counselor (CCC) credential",
        parent=parent_node,
        critical=False
    )

    sources = (info.sources if info else []) or []

    # Education requirement leaf
    edu_node = evaluator.add_leaf(
        id="CCC_Education",
        desc="Requires a master's degree or higher in counselor education, counseling psychology, or a closely related counseling degree",
        parent=node,
        critical=True
    )
    edu_claim = (
        "The NCDA Certified Career Counselor (CCC) credential requires a master's degree or higher in counselor "
        "education, counseling psychology, or a closely related counseling degree."
    )
    await evaluator.verify(
        claim=edu_claim,
        node=edu_node,
        sources=sources,
        additional_instruction=(
            "Use NCDA official CCC documentation where available. Accept closely equivalent phrasing that clearly "
            "specifies a master's degree or higher in a counseling-related field."
        )
    )

    # Career specialization requirement options leaf
    spec_node = evaluator.add_leaf(
        id="CCC_Career_Specialization",
        desc="Requires meeting one of three career specialization options: (1) minimum 600 hours supervised clinical experience in career counseling, OR (2) minimum 60 hours approved continuing education in career development, OR (3) completion of the U.S. NCDA Facilitating Career Development course",
        parent=node,
        critical=True
    )
    spec_claim = (
        "To fulfill the CCC career specialization requirement, applicants must meet one of the following: "
        "(1) at least 600 hours of supervised clinical experience in career counseling; "
        "(2) at least 60 hours of approved continuing education in career development; or "
        "(3) completion of the U.S. NCDA Facilitating Career Development (FCD) course."
    )
    await evaluator.verify(
        claim=spec_claim,
        node=spec_node,
        sources=sources,
        additional_instruction=(
            "Prefer NCDA official sources. Accept reasonable wording variations, but the three options (600 supervised "
            "hours; 60 CE hours; completion of U.S. NCDA FCD) must be explicit."
        )
    )


async def verify_primary_association_info(evaluator: Evaluator, parent_node, info: Optional[AssociationInfo]) -> None:
    node = evaluator.add_parallel(
        id="Primary_Association_Info",
        desc="Identifies the leading professional association and required identifying details",
        parent=parent_node,
        critical=False
    )

    sources = (info.sources if info else []) or []

    # Association name leaf
    name_node = evaluator.add_leaf(
        id="Primary_Association_Name",
        desc="Identifies the National Association of Colleges and Employers (NACE) as the leading professional association",
        parent=node,
        critical=True
    )
    name_claim = (
        "The leading professional association connecting U.S. college career services professionals and employers is the "
        "National Association of Colleges and Employers (NACE)."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_node,
        sources=sources,
        additional_instruction=(
            "Confirm that the source identifies NACE as the primary/leading association connecting college career "
            "services professionals and employers in the U.S."
        )
    )

    # Association role leaf
    role_node = evaluator.add_leaf(
        id="Primary_Association_Role",
        desc="States that NACE connects U.S. college career services professionals and employers",
        parent=node,
        critical=True
    )
    role_claim = "NACE connects U.S. college career services professionals and employers."
    await evaluator.verify(
        claim=role_claim,
        node=role_node,
        sources=sources,
        additional_instruction="Allow synonyms like 'links,' 'bridges,' or 'brings together' to indicate the same function."
    )

    # Network size leaf
    size_node = evaluator.add_leaf(
        id="Primary_Association_Network_Size",
        desc="States that NACE has over 17,600 professionals in its network",
        parent=node,
        critical=True
    )
    size_claim = "NACE has over 17,600 professionals in its network."
    await evaluator.verify(
        claim=size_claim,
        node=size_node,
        sources=sources,
        additional_instruction=(
            "Validate that the page provides a membership/network size around 17,600+ (e.g., 'over 17,600', '17,600+'). "
            "Minor rounding or phrasing differences are acceptable if clearly equivalent."
        )
    )


async def verify_nace_coaching_programs(evaluator: Evaluator, parent_node, info: Optional[NACECoachingInfo]) -> None:
    node = evaluator.add_parallel(
        id="NACE_Coaching_Programs",
        desc="Describes NACE Coaching Certification Program (CCP) offerings",
        parent=parent_node,
        critical=False
    )

    sources = (info.sources if info else []) or []

    # CCP tracks leaf
    tracks_node = evaluator.add_leaf(
        id="NACE_CCP_Tracks",
        desc="States that NACE offers a Coaching Certification Program (CCP) with two tracks: a 30-hour program and a 60-hour program",
        parent=node,
        critical=True
    )
    tracks_claim = (
        "NACE offers a Coaching Certification Program (CCP) with two tracks: a 30-hour program and a 60-hour program."
    )
    await evaluator.verify(
        claim=tracks_claim,
        node=tracks_node,
        sources=sources,
        additional_instruction=(
            "Confirm explicit mention of two tracks and their hour lengths (30-hour and 60-hour). Accept variants like "
            "'30 hours' and '60 hours'."
        )
    )

    # BCC preparation leaf
    bcc_node = evaluator.add_leaf(
        id="NACE_CCP_BCC_Preparation",
        desc="States that the CCP tracks prepare candidates for the Board Certified Coach (BCC) credential through the Center for Credentialing & Education (CCE)",
        parent=node,
        critical=True
    )
    bcc_claim = (
        "The NACE CCP tracks prepare candidates for the Board Certified Coach (BCC) credential through the Center for "
        "Credentialing & Education (CCE)."
    )
    await evaluator.verify(
        claim=bcc_claim,
        node=bcc_node,
        sources=sources,
        additional_instruction=(
            "Validate that the program is aligned with or prepares candidates for the BCC credential from CCE. "
            "Explicit reference to CCE/BCC is expected."
        )
    )


# --------------------------------------------------------------------------- #
# Orchestrators for each rubric subtree                                       #
# --------------------------------------------------------------------------- #
async def build_position_minimum_qualifications(
    evaluator: Evaluator,
    parent_node,
    extracted: StandardsExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Position_Minimum_Qualifications",
        desc="Standard minimum qualifications for career services positions based on industry norms",
        parent=parent_node,
        critical=False
    )
    await verify_coordinator_requirements(evaluator, node, extracted.coordinator)
    await verify_assistant_director_requirements(evaluator, node, extracted.assistant_director)


async def build_professional_certifications(
    evaluator: Evaluator,
    parent_node,
    extracted: StandardsExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Professional_Certifications",
        desc="Requirements for recognized professional certifications in career services",
        parent=parent_node,
        critical=False
    )
    await verify_gcdf_requirements(evaluator, node, extracted.gcdf)
    await verify_ccc_requirements(evaluator, node, extracted.ccc)


async def build_professional_association_resources(
    evaluator: Evaluator,
    parent_node,
    extracted: StandardsExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Professional_Association_Resources",
        desc="Leading professional association and its coaching certification program options",
        parent=parent_node,
        critical=False
    )
    await verify_primary_association_info(evaluator, node, extracted.association)
    await verify_nace_coaching_programs(evaluator, node, extracted.coaching_program)


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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root strategy
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

    # Create rubric root node under framework root
    rubric_root = evaluator.add_parallel(
        id="Career_Services_Standards_Documentation",
        desc="Complete documentation of standard industry requirements for university career services positions and professional certifications",
        parent=root,
        critical=False
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_standards(),
        template_class=StandardsExtraction,
        extraction_name="extracted_standards",
    )

    # Add reference expectations to summary (for transparency/debugging)
    evaluator.add_ground_truth({
        "positions": {
            "coordinator": {
                "education_min": "Bachelor's degree in a relevant field",
                "experience_min": "Approximately 2–3 years in career services or related"
            },
            "assistant_director": {
                "pathways": ["Master's + ~3 years", "Bachelor's + ~5 years"],
                "degree_field": "Counseling, higher education/student affairs, or related"
            }
        },
        "gcdf": {
            "training_hours": "120 hours from CCE Registered Credential Training Provider",
            "experience_by_education": {
                "graduate": 1400,
                "bachelor": 2800,
                "associate": 4200,
                "high_school": 5600
            }
        },
        "ccc": {
            "education": "Master's degree or higher in counseling-related field",
            "specialization_options": [
                "600 hours supervised clinical experience in career counseling",
                "60 hours approved continuing education in career development",
                "Completion of U.S. NCDA Facilitating Career Development (FCD) course"
            ]
        },
        "association": {
            "name": "National Association of Colleges and Employers (NACE)",
            "role": "Connects U.S. college career services professionals and employers",
            "network_size": "Over 17,600 professionals"
        },
        "nace_coaching": {
            "tracks": ["30-hour", "60-hour"],
            "bcc_preparation": "Prepares for BCC credential through CCE"
        }
    }, gt_type="expected_requirements")

    # Build and verify each major subtree
    await build_position_minimum_qualifications(evaluator, rubric_root, extracted)
    await build_professional_certifications(evaluator, rubric_root, extracted)
    await build_professional_association_resources(evaluator, rubric_root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()