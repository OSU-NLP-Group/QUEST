import asyncio
import logging
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "certificate_program_compliance"
TASK_DESCRIPTION = """
Identify a certificate program offered by a U.S. university extension, professional education division, or recognized educational organization that meets all of the following requirements:

1. The program must be in the field of e-learning, instructional design, or online teaching
2. The program must consist of exactly four courses
3. The program must have a specified standard completion duration of 8 months
4. The program must be available 100% online
5. The program must require applicants to have completed a minimum of two years of college education
6. The program must require applicants to have at least one year of work experience in education, training, or a related field where they have demonstrated ability to apply learning principles

Provide the name of the certificate program, the institution offering it, and a URL to the official program webpage.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProgramExtraction(BaseModel):
    """Information we need from the agent's answer."""
    program_name: Optional[str] = None
    institution_name: Optional[str] = None
    program_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_program_info() -> str:
    return """
    Extract exactly one certificate program from the answer text that the user intends to present.
    Return the following fields:
    - program_name: The official name/title of the certificate program.
    - institution_name: The institution or organization offering the program.
    - program_url: The URL to the official program webpage (not a general institution homepage or a news article).
    
    Rules:
    - If multiple programs are mentioned, extract the first one only.
    - Extract URLs exactly as shown in the answer. If a URL is missing a protocol, prepend http://.
    - If any field is missing in the answer, return null for that field.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_program_identity(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    extracted: ProgramExtraction,
) -> VerificationNode:
    """
    Build the Program Identity subtree:
    - Program_Verification_URL (existence check - critical)
    - Institution_Type (leaf - critical, verify by URL)
    - Subject_Area (leaf - critical, verify by URL)
    """
    identity_node = evaluator.add_parallel(
        id="Program_Identity",
        desc="Verify the program's institutional affiliation, subject area, and that an official program URL is provided",
        parent=parent_node,
        critical=True
    )

    # URL existence (critical custom node)
    url_ok = bool(extracted.program_url and extracted.program_url.strip())
    program_url_node = evaluator.add_custom_node(
        result=url_ok,
        id="Program_Verification_URL",
        desc="A URL to the official program webpage is provided",
        parent=identity_node,
        critical=True
    )

    # Institution type verification (critical leaf)
    inst_leaf = evaluator.add_leaf(
        id="Institution_Type",
        desc="The program is offered by a U.S. university extension, professional education division, or recognized educational organization",
        parent=identity_node,
        critical=True
    )
    inst_claim = (
        "This certificate program is offered by a U.S.-based university extension or professional/continuing education "
        "division, or by a recognized U.S. educational organization."
    )
    await evaluator.verify(
        claim=inst_claim,
        node=inst_leaf,
        sources=extracted.program_url,
        extra_prerequisites=[program_url_node],
        additional_instruction=(
            "Use the webpage content and its branding/header/footer to judge institutional affiliation. "
            "Accept synonyms like 'Extension', 'Continuing Education', 'Professional & Continuing Education', "
            "'School of Professional Studies', 'Continuing Studies'. The page should clearly indicate a U.S.-based provider."
        )
    )

    # Subject area verification (critical leaf)
    subj_leaf = evaluator.add_leaf(
        id="Subject_Area",
        desc="The program is in the field of e-learning, instructional design, or online teaching",
        parent=identity_node,
        critical=True
    )
    subj_claim = (
        "This program is in the field of e-learning, instructional design, or online teaching."
    )
    await evaluator.verify(
        claim=subj_claim,
        node=subj_leaf,
        sources=extracted.program_url,
        extra_prerequisites=[program_url_node],
        additional_instruction=(
            "Check the title/overview/curriculum to ensure the program focuses on e-learning, instructional design, "
            "or online teaching. Accept close synonyms such as 'learning design', 'course design', 'online pedagogy', "
            "or 'instructional systems design'. Do not pass if the focus is unrelated (e.g., general education leadership "
            "without explicit focus on e-learning/instructional design/online teaching)."
        )
    )

    return program_url_node


async def build_program_structure(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    program_url: Optional[str],
    url_prereq: VerificationNode,
) -> None:
    """
    Build the Program Structure subtree:
    - Course_Count (leaf - critical)
    - Program_Duration (leaf - critical)
    - Online_Delivery (leaf - critical)
    """
    structure_node = evaluator.add_parallel(
        id="Program_Structure",
        desc="Verify the program meets structural and delivery requirements",
        parent=parent_node,
        critical=True
    )

    # Course count (exactly 4) - critical leaf
    course_leaf = evaluator.add_leaf(
        id="Course_Count",
        desc="The program consists of exactly four courses",
        parent=structure_node,
        critical=True
    )
    course_claim = "The program consists of exactly four courses."
    await evaluator.verify(
        claim=course_claim,
        node=course_leaf,
        sources=program_url,
        extra_prerequisites=[url_prereq],
        additional_instruction=(
            "Pass only if the page clearly states that 4 courses (or 4 classes/modules) are required. "
            "Do not pass if the page shows a range (e.g., 3–5 courses), 'at least 4', or 'up to 4'."
        )
    )

    # Standard completion duration of 8 months - critical leaf
    duration_leaf = evaluator.add_leaf(
        id="Program_Duration",
        desc="The program has a specified standard completion duration of 8 months",
        parent=structure_node,
        critical=True
    )
    duration_claim = "The program has a standard completion duration of 8 months."
    await evaluator.verify(
        claim=duration_claim,
        node=duration_leaf,
        sources=program_url,
        extra_prerequisites=[url_prereq],
        additional_instruction=(
            "Pass only if the page explicitly indicates an 8-month completion timeline (e.g., 'complete in 8 months', "
            "'8-month program'). If the page lists a wider range (e.g., 6–12 months) without a specific 8-month standard, do not pass."
        )
    )

    # 100% online - critical leaf
    online_leaf = evaluator.add_leaf(
        id="Online_Delivery",
        desc="The program is available 100% online",
        parent=structure_node,
        critical=True
    )
    online_claim = "The program is available 100% online."
    await evaluator.verify(
        claim=online_claim,
        node=online_leaf,
        sources=program_url,
        extra_prerequisites=[url_prereq],
        additional_instruction=(
            "Accept phrasing like 'fully online', '100% online', 'entirely online'. "
            "Do not pass if any required in-person/on-campus component is indicated."
        )
    )


async def build_admission_requirements(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    program_url: Optional[str],
    url_prereq: VerificationNode,
) -> None:
    """
    Build the Admission Requirements subtree:
    - Education_Prerequisite (leaf - critical)
    - Work_Experience_Requirement (parallel - critical) with 3 critical leaves:
        - Experience_Duration
        - Experience_Field
        - Demonstrated_Ability_To_Apply_Learning_Principles
    """
    adm_node = evaluator.add_parallel(
        id="Admission_Requirements",
        desc="Verify the program has the stated admission prerequisites",
        parent=parent_node,
        critical=True
    )

    # Education prerequisite (minimum two years of college) - critical leaf
    edu_leaf = evaluator.add_leaf(
        id="Education_Prerequisite",
        desc="The program requires applicants to have completed a minimum of two years of college education",
        parent=adm_node,
        critical=True
    )
    edu_claim = "Applicants must have completed a minimum of two years of college education."
    await evaluator.verify(
        claim=edu_claim,
        node=edu_leaf,
        sources=program_url,
        extra_prerequisites=[url_prereq],
        additional_instruction=(
            "Look for phrases like 'minimum two years of college', 'at least sophomore standing', "
            "'60 semester credits', or equivalent indications. The requirement must be explicit."
        )
    )

    # Work experience requirement (parallel critical subnode)
    work_node = evaluator.add_parallel(
        id="Work_Experience_Requirement",
        desc="Verify the stated work-experience prerequisite (duration, field, and demonstrated ability requirement)",
        parent=adm_node,
        critical=True
    )

    # Experience_Duration - critical leaf
    exp_dur_leaf = evaluator.add_leaf(
        id="Experience_Duration",
        desc="The program requires at least one year of work experience",
        parent=work_node,
        critical=True
    )
    exp_dur_claim = "Applicants must have at least one year of work experience."
    await evaluator.verify(
        claim=exp_dur_claim,
        node=exp_dur_leaf,
        sources=program_url,
        extra_prerequisites=[url_prereq],
        additional_instruction=(
            "Accept 'at least 1 year', '1+ years', or 'minimum one year'. The requirement must be explicit."
        )
    )

    # Experience_Field - critical leaf
    exp_field_leaf = evaluator.add_leaf(
        id="Experience_Field",
        desc="The required work experience is in education, training, or a related field",
        parent=work_node,
        critical=True
    )
    exp_field_claim = "The required work experience must be in education, training, or a related field."
    await evaluator.verify(
        claim=exp_field_claim,
        node=exp_field_leaf,
        sources=program_url,
        extra_prerequisites=[url_prereq],
        additional_instruction=(
            "Accept fields like teaching, instructional design, learning & development, corporate training, HR training, "
            "or closely related areas. The page should clearly specify the relevant field."
        )
    )

    # Demonstrated ability to apply learning principles - critical leaf
    ability_leaf = evaluator.add_leaf(
        id="Demonstrated_Ability_To_Apply_Learning_Principles",
        desc="The work-experience requirement includes demonstrated ability to apply learning principles",
        parent=work_node,
        critical=True
    )
    ability_claim = (
        "The work-experience requirement includes demonstrated ability to apply learning principles."
    )
    await evaluator.verify(
        claim=ability_claim,
        node=ability_leaf,
        sources=program_url,
        extra_prerequisites=[url_prereq],
        additional_instruction=(
            "Look for wording such as 'demonstrated ability to apply learning principles', "
            "'experience applying instructional design or learning theory', or 'teaching practice applying pedagogy'. "
            "The page must explicitly include this aspect within the work-experience requirement."
        )
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
    Evaluate the agent's answer for the certificate program compliance task.
    """
    # Initialize evaluator (root is non-critical by design)
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

    # Extract the program info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_program_info(),
        template_class=ProgramExtraction,
        extraction_name="program_extraction",
    )

    # Create the main compliance node (critical)
    compliance_node = evaluator.add_parallel(
        id="Certificate_Program_Compliance",
        desc="Verify that the identified certificate program satisfies all stated requirements",
        parent=root,
        critical=True
    )

    # Build subtrees
    url_prereq_node = await build_program_identity(evaluator, compliance_node, extracted)
    await build_program_structure(evaluator, compliance_node, extracted.program_url, url_prereq_node)
    await build_admission_requirements(evaluator, compliance_node, extracted.program_url, url_prereq_node)

    # Return final summary
    return evaluator.get_summary()