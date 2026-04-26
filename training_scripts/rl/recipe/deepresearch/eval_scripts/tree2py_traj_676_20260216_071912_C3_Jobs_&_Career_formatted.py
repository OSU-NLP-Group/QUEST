import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "ga_em_career_path"
TASK_DESCRIPTION = """
For someone in Georgia who wants to pursue a career as a Director of Emergency Management in a school district, identify one specific bachelor's degree program at a public university in Georgia that satisfies all of the following requirements:

1. The program must offer a bachelor's degree specifically in Emergency Management (not just a related field)
2. According to IAEM CEM certification requirements, holding a bachelor's degree specifically in Emergency Management must qualify the holder for the reduced 2-year comprehensive work experience requirement (instead of the standard 3-year requirement) for CEM certification eligibility
3. The bachelor's degree in Emergency Management must meet the preferred educational qualification stated for school district Director of Emergency Management positions

Provide the following information:
- The name of the public university in Georgia
- The exact name of the bachelor's degree program
- A URL reference to the university's official program page that confirms this bachelor's degree in Emergency Management is offered
"""


class ProgramSelection(BaseModel):
    university_name: Optional[str] = None
    program_name: Optional[str] = None
    degree_level: Optional[str] = None
    program_url: Optional[str] = None


class EvidenceExtraction(BaseModel):
    iaem_urls: List[str] = Field(default_factory=list)
    director_urls: List[str] = Field(default_factory=list)


def prompt_extract_program_selection() -> str:
    return """
    From the provided answer, extract the specific bachelor's degree program information at a public university in Georgia.
    Return a JSON object with fields:
    - university_name: The full name of the public university in Georgia.
    - program_name: The exact name of the bachelor's degree program as stated (do not paraphrase).
    - degree_level: The degree level string if mentioned (e.g., "Bachelor of Science", "BS", "Bachelor of Arts").
    - program_url: The official university program webpage URL that confirms the bachelor's degree in Emergency Management is offered.
    Rules:
    - Extract exactly what appears in the answer text. If any field is missing, set it to null.
    - For URLs, extract the actual URL even if it is embedded in markdown; ensure a valid absolute URL.
    """


def prompt_extract_supporting_evidence() -> str:
    return """
    From the provided answer, extract supporting URLs for the following claims:
    - iaem_urls: All URLs that point to official IAEM pages or authoritative sources that state the IAEM CEM work experience can be reduced to 2 years if the candidate holds a bachelor's degree specifically in Emergency Management.
    - director_urls: URLs to job descriptions or school district documents that show Director of Emergency Management positions list a bachelor's in Emergency Management (or explicitly 'Emergency Management' among preferred degrees) as preferred/required education.
    Rules:
    - Return both fields as arrays of URLs. If none are provided, return empty arrays.
    - Extract only URLs explicitly mentioned in the answer.
    """


def _clean_urls(urls: List[str]) -> List[str]:
    return [u.strip() for u in urls if isinstance(u, str) and u.strip()]


async def verify_career_pathway(
    evaluator: Evaluator,
    root_node,
    program: ProgramSelection,
    evidence: EvidenceExtraction
) -> None:
    # Career_Pathway_Verification (root already exists with sequential strategy)
    # University_Program_Identification (critical sequential)
    uni_prog_node = evaluator.add_sequential(
        id="University_Program_Identification",
        desc="Identify a specific bachelor's degree program at a public university in Georgia",
        parent=root_node,
        critical=True
    )

    # Existence check for essential fields
    program_info_provided = evaluator.add_custom_node(
        result=bool(program.university_name and program.program_name and program.program_url),
        id="Program_Info_Provided",
        desc="Program information provided: university_name, program_name, program_url",
        parent=uni_prog_node,
        critical=True
    )

    # Optional geographic check (kept critical per parent requirement; verified via program_url)
    georgia_check_leaf = evaluator.add_leaf(
        id="University_in_Georgia",
        desc="Verify the university is located in Georgia, USA",
        parent=uni_prog_node,
        critical=True
    )
    georgia_claim = f"The university '{program.university_name or ''}' is located in Georgia, USA."
    await evaluator.verify(
        claim=georgia_claim,
        node=georgia_check_leaf,
        sources=program.program_url,
        additional_instruction="Check the program page header/footer, address, campus location, or 'About' information for evidence that the institution is in Georgia."
    )

    # Emergency_Management_Degree_Verification (critical sequential)
    em_verify_node = evaluator.add_sequential(
        id="Emergency_Management_Degree_Verification",
        desc="Verify that the program offers a bachelor's degree specifically in Emergency Management (not just a related field)",
        parent=uni_prog_node,
        critical=True
    )

    # Degree level must be bachelor's
    degree_is_bachelors_leaf = evaluator.add_leaf(
        id="Degree_is_Bachelors",
        desc="Program confers a bachelor's degree (e.g., BS/BA)",
        parent=em_verify_node,
        critical=True
    )
    degree_claim = f"The program '{program.program_name or ''}' offered by '{program.university_name or ''}' confers a bachelor's degree (e.g., Bachelor of Science or Bachelor of Arts)."
    await evaluator.verify(
        claim=degree_claim,
        node=degree_is_bachelors_leaf,
        sources=program.program_url,
        additional_instruction="Confirm that the program is explicitly a bachelor's degree (BS/BA). If only certificates or graduate programs are shown, this should fail."
    )

    # Field is specifically Emergency Management
    field_is_em_specific_leaf = evaluator.add_leaf(
        id="Field_is_Emergency_Management_Specific",
        desc="Program is specifically in 'Emergency Management' (not just a related field)",
        parent=em_verify_node,
        critical=True
    )
    field_claim = f"The program '{program.program_name or ''}' is specifically in 'Emergency Management' (the degree name or major explicitly contains 'Emergency Management')."
    await evaluator.verify(
        claim=field_claim,
        node=field_is_em_specific_leaf,
        sources=program.program_url,
        additional_instruction="Be strict: the program should explicitly be titled as 'Emergency Management' (e.g., 'B.S. in Emergency Management'). Do not accept only related-fields naming such as Homeland Security or Public Safety unless 'Emergency Management' is clearly the named major."
    )

    # Degree_Evidence_and_Properties (critical sequential)
    degree_props_node = evaluator.add_sequential(
        id="Degree_Evidence_and_Properties",
        desc="Verify that evidence is provided and the degree meets all required properties",
        parent=em_verify_node,
        critical=True
    )

    # Degree_Requirements_Verification (critical parallel)
    degree_reqs_node = evaluator.add_parallel(
        id="Degree_Requirements_Verification",
        desc="Verify all independent requirements about the Emergency Management degree",
        parent=degree_props_node,
        critical=True
    )

    # IAEM_CEM_Eligibility_Verification (critical leaf)
    iaem_leaf = evaluator.add_leaf(
        id="IAEM_CEM_Eligibility_Verification",
        desc="Holding this specific Emergency Management bachelor's degree qualifies for the reduced 2-year (instead of 3-year) comprehensive EM work experience for IAEM CEM",
        parent=degree_reqs_node,
        critical=True
    )
    iaem_claim = "According to official IAEM CEM certification requirements, holding a bachelor's degree specifically in Emergency Management qualifies the candidate for a reduced two-year comprehensive emergency management work experience requirement (instead of the standard three years)."
    iaem_sources = _clean_urls(evidence.iaem_urls)
    await evaluator.verify(
        claim=iaem_claim,
        node=iaem_leaf,
        sources=iaem_sources,
        additional_instruction="Verify on official IAEM sources. Look for explicit language about a bachelor's degree in Emergency Management reducing the comprehensive EM work experience requirement from 3 years to 2 years."
    )

    # School_Director_Qualification_Verification (critical leaf)
    director_leaf = evaluator.add_leaf(
        id="School_Director_Qualification_Verification",
        desc="Bachelor's in Emergency Management meets preferred educational qualification for school district Director of Emergency Management roles",
        parent=degree_reqs_node,
        critical=True
    )
    director_claim = "School district Director of Emergency Management job descriptions list a bachelor's degree in Emergency Management (or explicitly include 'Emergency Management' among preferred degrees) as a preferred or required educational qualification."
    director_sources = _clean_urls(evidence.director_urls)
    await evaluator.verify(
        claim=director_claim,
        node=director_leaf,
        sources=director_sources,
        additional_instruction="Check the education/qualification section of the job description. It should explicitly include 'Emergency Management' (potentially among related fields). If the posting only mentions unrelated fields and not Emergency Management, this should fail."
    )

    # Program_URL_Reference (critical leaf)
    program_url_leaf = evaluator.add_leaf(
        id="Program_URL_Reference",
        desc="Valid official program page URL confirms the bachelor's degree in Emergency Management is offered",
        parent=degree_reqs_node,
        critical=True
    )
    program_url_claim = f"The official program page confirms that a bachelor's degree in Emergency Management is offered and that the program is named '{program.program_name or ''}'."
    await evaluator.verify(
        claim=program_url_claim,
        node=program_url_leaf,
        sources=program.program_url,
        additional_instruction="Confirm the page is an official university page and explicitly states the offering of a bachelor's degree in Emergency Management (by title or description)."
    )


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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Root node mirrors rubric root description
    root.desc = "Verify that the identified educational pathway in Georgia enables both IAEM CEM eligibility with reduced experience and meets school district director qualifications"

    # Extract program selection and supporting evidence concurrently
    program_task = evaluator.extract(
        prompt=prompt_extract_program_selection(),
        template_class=ProgramSelection,
        extraction_name="program_selection"
    )
    evidence_task = evaluator.extract(
        prompt=prompt_extract_supporting_evidence(),
        template_class=EvidenceExtraction,
        extraction_name="supporting_evidence"
    )
    program_result, evidence_result = await asyncio.gather(program_task, evidence_task)

    # Build verification tree and run checks
    await verify_career_pathway(
        evaluator=evaluator,
        root_node=root,
        program=program_result,
        evidence=evidence_result
    )

    # Return evaluation summary
    return evaluator.get_summary()