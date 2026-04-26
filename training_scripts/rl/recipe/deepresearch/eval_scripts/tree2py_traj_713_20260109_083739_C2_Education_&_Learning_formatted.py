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
TASK_ID = "ny_1887_adv_cert_college_advising"
TASK_DESCRIPTION = """
Identify the institution of higher education in New York State that was founded in 1887, is accredited by the Middle States Commission on Higher Education, and offers an online Advanced Certificate in College Advising. For this certificate program, provide: (1) the institution's name, (2) a URL to the program's official page, (3) the number of graduate-level credits required, and (4) the degree requirement for admission.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class ExtractedAnswer(BaseModel):
    # Institution facts
    institution_name: Optional[str] = None
    founded_year: Optional[str] = None
    accreditation_text: Optional[str] = None
    state_or_location: Optional[str] = None
    institution_urls: List[str] = Field(default_factory=list)

    # Program facts
    program_name: Optional[str] = None  # e.g., "Advanced Certificate in College Advising"
    program_url: Optional[str] = None
    credits_required: Optional[str] = None  # keep as string to allow "12", "12 credits", "12–15", etc.
    admission_degree_requirement: Optional[str] = None  # e.g., "Bachelor's degree"
    additional_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_answer() -> str:
    return """
    Extract, from the provided answer text, the single institution and its program details that best satisfy the task:
    - The institution is located in New York State (NY), was founded in 1887, is accredited by the Middle States Commission on Higher Education (MSCHE), and offers an online Advanced Certificate in College Advising.
    - If the answer mentions multiple institutions or programs, pick the one the answer ultimately uses to satisfy the task.

    Return the following fields:
    1) institution_name: The institution’s name (string).
    2) founded_year: The founding year of the institution as written in the answer (string).
    3) accreditation_text: Any accreditation phrase(s) as written in the answer (string), ideally referencing MSCHE.
    4) state_or_location: The geographic location as written (string) (e.g., "New York, NY" or "New York State").
    5) institution_urls: Array of any URLs in the answer that are about the institution in general (about, accreditation, overview pages). Do not include the specific program URL here.
    6) program_name: The program name as written (string). Prefer the exact phrasing used by the answer (e.g., "Advanced Certificate in College Advising").
    7) program_url: The official URL to the program’s page (string URL).
    8) credits_required: The number of graduate-level credits required as written (string).
    9) admission_degree_requirement: The degree requirement for admission as written (string), e.g., "Bachelor's degree".
    10) additional_urls: Array of any other URLs mentioned in the answer that are relevant to verifying the institution’s facts or the program, excluding the program_url already captured.

    Special rules for URLs:
    - Extract only URLs explicitly present in the answer.
    - Include full URLs (with http/https).
    - If no such URLs are present for a field, return an empty array.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _sanitize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    dedup = []
    seen = set()
    for u in urls:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        # Normalize missing protocol if needed (Extractor may already handle this)
        if not (u.startswith("http://") or u.startswith("https://")):
            u = "http://" + u
        if u not in seen:
            seen.add(u)
            dedup.append(u)
    return dedup


def _combine_sources(*url_lists: List[str]) -> List[str]:
    combined: List[str] = []
    seen = set()
    for urls in url_lists:
        for u in urls:
            if u not in seen:
                seen.add(u)
                combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def build_institution_identification_checks(
    evaluator: Evaluator,
    parent_node,
    data: ExtractedAnswer,
) -> None:
    """
    Build and run all verification checks for InstitutionIdentification as a critical parallel node.
    Checks:
      - Institution name exists (custom)
      - Founded in 1887 (URL-supported)
      - Located in New York State (URL-supported)
      - Accredited by Middle States Commission on Higher Education (URL-supported)
      - Offers an online Advanced Certificate in College Advising (URL-supported, typically program page)
    """
    inst_node = evaluator.add_parallel(
        id="InstitutionIdentification",
        desc="Identify the institution founded in 1887 in New York State that is accredited by the Middle States Commission on Higher Education and offers an online Advanced Certificate in College Advising",
        parent=parent_node,
        critical=True
    )

    # Existence gate: Institution name must be provided
    inst_exists = evaluator.add_custom_node(
        result=(data.institution_name is not None and str(data.institution_name).strip() != ""),
        id="InstitutionNameProvided",
        desc="Institution name is provided",
        parent=inst_node,
        critical=True
    )

    # Collect sources
    program_url_list = _sanitize_urls([data.program_url] if data.program_url else [])
    inst_urls = _sanitize_urls(data.institution_urls)
    addl_urls = _sanitize_urls(data.additional_urls)
    all_inst_sources = _combine_sources(inst_urls, addl_urls, program_url_list)

    # 1) Founded year is 1887
    founded_leaf = evaluator.add_leaf(
        id="FoundedIn1887",
        desc="The institution was founded in 1887 (supported by cited sources)",
        parent=inst_node,
        critical=True
    )
    founded_claim = f"The institution named '{data.institution_name or ''}' was founded in 1887."
    await evaluator.verify(
        claim=founded_claim,
        node=founded_leaf,
        sources=all_inst_sources,
        additional_instruction="Check the provided webpages for the institution's founding year. Accept clear references such as 'founded in 1887' or equivalent."
    )

    # 2) Located in New York State
    location_leaf = evaluator.add_leaf(
        id="LocatedInNYState",
        desc="The institution is located in New York State (supported by cited sources)",
        parent=inst_node,
        critical=True
    )
    location_claim = f"The institution named '{data.institution_name or ''}' is located in New York State (NY), United States."
    await evaluator.verify(
        claim=location_claim,
        node=location_leaf,
        sources=all_inst_sources,
        additional_instruction="Accept mentions of a city in New York (e.g., New York, NY) or explicit statements that the institution is in New York State."
    )

    # 3) Accredited by MSCHE
    accred_leaf = evaluator.add_leaf(
        id="AccreditedByMSCHE",
        desc="The institution is accredited by the Middle States Commission on Higher Education (supported by cited sources)",
        parent=inst_node,
        critical=True
    )
    accred_claim = f"The institution named '{data.institution_name or ''}' is accredited by the Middle States Commission on Higher Education (MSCHE)."
    await evaluator.verify(
        claim=accred_claim,
        node=accred_leaf,
        sources=all_inst_sources,
        additional_instruction="Look for accreditation statements including 'Middle States Commission on Higher Education' or 'MSCHE'. Footers or accreditation pages are acceptable."
    )

    # 4) Offers an online Advanced Certificate in College Advising
    offers_leaf = evaluator.add_leaf(
        id="OffersOnlineAdvCertCollegeAdvising",
        desc="The institution offers an online Advanced Certificate in College Advising (supported by cited sources)",
        parent=inst_node,
        critical=True
    )
    offers_claim = f"The institution named '{data.institution_name or ''}' offers an online Advanced Certificate in College Advising."
    await evaluator.verify(
        claim=offers_claim,
        node=offers_leaf,
        sources=program_url_list if program_url_list else all_inst_sources,
        additional_instruction="Verify on the program page (or official institutional pages) that the program is both: (1) 'Advanced Certificate' (or an equivalent graduate/post-baccalaureate advanced certificate) and (2) online/fully online/distance. The title should clearly center on 'College Advising'."
    )


async def build_program_details_checks(
    evaluator: Evaluator,
    parent_node,
    data: ExtractedAnswer,
) -> None:
    """
    Build and run all verification checks for ProgramDetails as a critical parallel node.
    Children:
      - ProgramURLReference (critical sequential):
          * URL provided (custom)
          * URL points to the official program page for (online) Advanced Certificate in College Advising
      - CreditRequirement (critical leaf): number of graduate-level credits required is supported by program page
      - AdmissionRequirement (critical leaf): degree requirement for admission is supported by program page
    """
    details_node = evaluator.add_parallel(
        id="ProgramDetails",
        desc="Provide complete details about the online Advanced Certificate in College Advising program",
        parent=parent_node,
        critical=True
    )

    # Program URL reference subnode (sequential)
    url_ref_node = evaluator.add_sequential(
        id="ProgramURLReference",
        desc="Provide a valid URL reference to the program's official page",
        parent=details_node,
        critical=True
    )

    # a) URL provided (gate)
    url_provided = evaluator.add_custom_node(
        result=(data.program_url is not None and str(data.program_url).strip() != ""),
        id="ProgramURLProvided",
        desc="Program URL is provided",
        parent=url_ref_node,
        critical=True
    )

    # b) URL is official program page for the target program
    url_valid_leaf = evaluator.add_leaf(
        id="ProgramURLOfficialAndRelevant",
        desc="The provided URL is the official page for the (online) Advanced Certificate in College Advising",
        parent=url_ref_node,
        critical=True
    )
    url_valid_claim = (
        f"This webpage is the official program page for the "
        f"{data.program_name or 'Advanced Certificate in College Advising'} at {data.institution_name or ''}, "
        f"and it describes an online format."
    )
    await evaluator.verify(
        claim=url_valid_claim,
        node=url_valid_leaf,
        sources=data.program_url,
        additional_instruction="Verify that the page is hosted by the institution and clearly describes the Advanced Certificate in College Advising program; confirm that it indicates an online/fully online/distance format."
    )

    # Credit requirement verification
    credits_leaf = evaluator.add_leaf(
        id="CreditRequirement",
        desc="State the number of graduate-level credits required for the Advanced Certificate in College Advising",
        parent=details_node,
        critical=True
    )
    credits_claim = f"The program requires {data.credits_required or ''} graduate-level credits to complete."
    await evaluator.verify(
        claim=credits_claim,
        node=credits_leaf,
        sources=data.program_url,
        additional_instruction="Check the program page for the total required graduate credits. Accept reasonable formatting variants, e.g., '12 credits', '12 graduate credits', or spelled-out numbers."
    )

    # Admission degree requirement verification
    admission_leaf = evaluator.add_leaf(
        id="AdmissionRequirement",
        desc="State the degree requirement for admission to the Advanced Certificate program",
        parent=details_node,
        critical=True
    )
    admission_claim = (
        f"The degree requirement for admission to the program is '{data.admission_degree_requirement or ''}'."
    )
    await evaluator.verify(
        claim=admission_claim,
        node=admission_leaf,
        sources=data.program_url,
        additional_instruction="Verify the minimum degree requirement (e.g., Bachelor's degree) from the admissions section of the program page. Focus on degree requirement, not GPA or other criteria."
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
    Evaluate an answer for the NY 1887 online Advanced Certificate in College Advising task.
    Returns a structured summary containing the verification tree and final score.
    """
    # Initialize evaluator and root node following rubric's root "TaskCompletion" (sequential)
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
        default_model=model,
    )

    # Extract structured info from the answer
    extracted: ExtractedAnswer = await evaluator.extract(
        prompt=prompt_extract_answer(),
        template_class=ExtractedAnswer,
        extraction_name="extracted_answer",
    )

    # Build the rubric tree nodes and run verifications

    # 1) InstitutionIdentification (critical)
    await build_institution_identification_checks(evaluator, root, extracted)

    # 2) ProgramDetails (critical, evaluated after identification due to sequential root)
    await build_program_details_checks(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()