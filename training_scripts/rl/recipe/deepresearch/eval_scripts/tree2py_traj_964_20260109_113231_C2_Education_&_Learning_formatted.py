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
TASK_ID = "etl_university_dc"
TASK_DESCRIPTION = (
    "Identify a university in the United States that offers a fully online master's degree program in Educational "
    "Technology Leadership and is accredited by the Middle States Commission on Higher Education. The program must "
    "consist of exactly 12 credit hours for the graduate certificate version. Additionally, the university must be "
    "located in Washington, D.C. Provide the full name of the university."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityExtraction(BaseModel):
    university_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    university_urls: List[str] = Field(default_factory=list)


class ProgramExtraction(BaseModel):
    program_name: Optional[str] = None
    program_type: Optional[str] = None  # e.g., "Master's", "M.A.", "M.Ed."
    modality: Optional[str] = None      # e.g., "fully online", "online", "hybrid", "on-campus"
    program_urls: List[str] = Field(default_factory=list)


class CertificateExtraction(BaseModel):
    certificate_name: Optional[str] = None
    credit_hours: Optional[str] = None  # keep as string to allow "12 credit hours"
    certificate_urls: List[str] = Field(default_factory=list)


class AccreditationExtraction(BaseModel):
    accreditor_name: Optional[str] = None  # e.g., "Middle States Commission on Higher Education"
    accreditor_abbrev: Optional[str] = None  # e.g., "MSCHE"
    accreditation_urls: List[str] = Field(default_factory=list)


class ETLUniversityExtraction(BaseModel):
    university: Optional[UniversityExtraction] = None
    program: Optional[ProgramExtraction] = None
    certificate: Optional[CertificateExtraction] = None
    accreditation: Optional[AccreditationExtraction] = None
    general_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_etl_university_info() -> str:
    return """
    From the provided answer, extract the structured information for a university that (according to the answer) satisfies all the constraints.
    Extract the following fields. If any field is missing, set it to null or an empty list as appropriate.

    university:
      - university_name: The full official name of the university (e.g., "The George Washington University").
      - city: City where the university is located (e.g., "Washington").
      - state: State or district abbreviation/name (e.g., "DC" or "District of Columbia").
      - country: The country (e.g., "United States").
      - university_urls: All URLs in the answer that point to the university's official website (homepage, about page, academics).

    program:
      - program_name: The name of the master's program if mentioned (should relate to "Educational Technology Leadership").
      - program_type: The master's credential type if mentioned (e.g., "Master's", "M.A.", "M.Ed.").
      - modality: How the program is delivered if explicitly stated (e.g., "fully online", "online", "hybrid").
      - program_urls: All URLs that specifically describe or are dedicated to the Educational Technology Leadership master's program.

    certificate:
      - certificate_name: The name of the graduate certificate program related to Educational Technology Leadership, if provided.
      - credit_hours: The total credit hours for the certificate as stated (e.g., "12", "12 credit hours").
      - certificate_urls: All URLs that specifically describe the graduate certificate program and its credit requirements.

    accreditation:
      - accreditor_name: The accrediting body name if mentioned (e.g., "Middle States Commission on Higher Education").
      - accreditor_abbrev: The accrediting body abbreviation if mentioned (e.g., "MSCHE").
      - accreditation_urls: URLs that support accreditation (e.g., an MSCHE institution page or the university's accreditation page mentioning MSCHE).

    general_sources:
      - Any other URLs present in the answer that are relevant but not already included above.

    SPECIAL RULES FOR URL SOURCES:
    - Extract only URLs explicitly present in the answer (including markdown links).
    - Prefer official pages (university domains, MSCHE.org) when available.
    - Do not invent URLs.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls:
        if not isinstance(u, str):
            continue
        uu = u.strip()
        if not uu:
            continue
        if uu not in seen:
            seen.add(uu)
            result.append(uu)
    return result


def collect_all_sources(extracted: ETLUniversityExtraction) -> List[str]:
    urls: List[str] = []
    if extracted.university and extracted.university.university_urls:
        urls.extend(extracted.university.university_urls)
    if extracted.program and extracted.program.program_urls:
        urls.extend(extracted.program.program_urls)
    if extracted.certificate and extracted.certificate.certificate_urls:
        urls.extend(extracted.certificate.certificate_urls)
    if extracted.accreditation and extracted.accreditation.accreditation_urls:
        urls.extend(extracted.accreditation.accreditation_urls)
    if extracted.general_sources:
        urls.extend(extracted.general_sources)
    return _dedupe_urls(urls)


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: ETLUniversityExtraction, root_node) -> None:
    # Create a critical parallel node under the root to represent the task-level constraints
    main_node = evaluator.add_parallel(
        id="task_main",
        desc="Response identifies a university that satisfies all stated constraints",
        parent=root_node,
        critical=True
    )

    # Extract key data
    uni_name = extracted.university.university_name.strip() if (extracted.university and extracted.university.university_name) else ""
    city = extracted.university.city.strip() if (extracted.university and extracted.university.city) else ""
    state = extracted.university.state.strip() if (extracted.university and extracted.university.state) else ""
    country = extracted.university.country.strip() if (extracted.university and extracted.university.country) else ""
    program_name = extracted.program.program_name.strip() if (extracted.program and extracted.program.program_name) else ""
    modality = extracted.program.modality.strip() if (extracted.program and extracted.program.modality) else ""
    certificate_name = extracted.certificate.certificate_name.strip() if (extracted.certificate and extracted.certificate.certificate_name) else ""
    credit_hours = extracted.certificate.credit_hours.strip() if (extracted.certificate and extracted.certificate.credit_hours) else ""
    accreditor_name = extracted.accreditation.accreditor_name.strip() if (extracted.accreditation and extracted.accreditation.accreditor_name) else ""
    accreditor_abbrev = extracted.accreditation.accreditor_abbrev.strip() if (extracted.accreditation and extracted.accreditation.accreditor_abbrev) else ""

    university_sources = _dedupe_urls(extracted.university.university_urls if extracted.university else [])
    program_sources = _dedupe_urls(extracted.program.program_urls if extracted.program else [])
    certificate_sources = _dedupe_urls(extracted.certificate.certificate_urls if extracted.certificate else [])
    accreditation_sources = _dedupe_urls(extracted.accreditation.accreditation_urls if extracted.accreditation else [])
    all_sources = collect_all_sources(extracted)

    # 1) Provide the full name of the university (existence check)
    name_exists = bool(uni_name)
    name_node = evaluator.add_custom_node(
        result=name_exists,
        id="provides_full_university_name",
        desc="Provide the full name of the university",
        parent=main_node,
        critical=True
    )

    # 2) The identified institution is a university in the United States
    us_node = evaluator.add_leaf(
        id="university_in_united_states",
        desc="The identified institution is a university in the United States",
        parent=main_node,
        critical=True
    )
    us_claim = (
        f"'{uni_name}' is a university in the United States."
        if uni_name else "The identified institution is a university in the United States."
    )
    await evaluator.verify(
        claim=us_claim,
        node=us_node,
        sources=university_sources or all_sources or None,
        extra_prerequisites=[name_node],
        additional_instruction="Use the provided webpages to confirm the institution is a US university (e.g., location, accreditor, or official about page)."
    )

    # 3) The university is located in Washington, D.C.
    loc_node = evaluator.add_leaf(
        id="location_washington_dc",
        desc="The university is located in Washington, D.C.",
        parent=main_node,
        critical=True
    )
    loc_claim = (
        f"'{uni_name}' is located in Washington, D.C."
        if uni_name else "The university is located in Washington, D.C."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=university_sources or all_sources or None,
        extra_prerequisites=[name_node],
        additional_instruction="Verify the institution's location is Washington, D.C. Accept variants like 'Washington, DC', 'District of Columbia', or 'Washington, D.C.'."
    )

    # 4) Offers a fully online master's degree program in Educational Technology Leadership
    etl_node = evaluator.add_leaf(
        id="offers_fully_online_masters_etl",
        desc="The university offers a fully online master's degree program in Educational Technology Leadership",
        parent=main_node,
        critical=True
    )
    etl_claim = (
        f"'{uni_name}' offers a fully online master's degree program in Educational Technology Leadership."
        if uni_name else "The university offers a fully online master's degree program in Educational Technology Leadership."
    )
    await evaluator.verify(
        claim=etl_claim,
        node=etl_node,
        sources=program_sources or all_sources or None,
        extra_prerequisites=[name_node],
        additional_instruction=(
            "Confirm the program is master's-level (e.g., MA, MS, M.Ed.) specifically in 'Educational Technology Leadership' "
            "and that it is fully online. Accept synonyms like 'entirely online', '100% online', or 'distance/online learning'."
        )
    )

    # 5) Accredited by MSCHE
    msche_node = evaluator.add_leaf(
        id="msche_accreditation",
        desc="The university is accredited by the Middle States Commission on Higher Education (MSCHE)",
        parent=main_node,
        critical=True
    )
    msche_claim = (
        f"'{uni_name}' is accredited by the Middle States Commission on Higher Education (MSCHE)."
        if uni_name else "The university is accredited by the Middle States Commission on Higher Education (MSCHE)."
    )
    await evaluator.verify(
        claim=msche_claim,
        node=msche_node,
        sources=accreditation_sources or all_sources or None,
        extra_prerequisites=[name_node],
        additional_instruction=(
            "Prefer official accreditation evidence (e.g., MSCHE.org institution page or the university's accreditation page "
            "explicitly naming 'Middle States Commission on Higher Education' or 'MSCHE')."
        )
    )

    # 6) Graduate certificate version consists of exactly 12 credit hours
    credits_node = evaluator.add_leaf(
        id="graduate_certificate_exactly_12_credits",
        desc="The graduate certificate version of the Educational Technology Leadership program consists of exactly 12 credit hours",
        parent=main_node,
        critical=True
    )
    cert_phrase = certificate_name if certificate_name else "the graduate certificate in Educational Technology Leadership"
    credits_claim = (
        f"{cert_phrase} at '{uni_name}' consists of exactly 12 credit hours."
        if uni_name else f"{cert_phrase} consists of exactly 12 credit hours."
    )
    await evaluator.verify(
        claim=credits_claim,
        node=credits_node,
        sources=certificate_sources or program_sources or all_sources or None,
        extra_prerequisites=[name_node],
        additional_instruction=(
            "Verify that the certificate credit requirement is exactly 12 credits. Accept phrasing like '12 credits', "
            "'12 credit hours', or 'total of 12 credits'."
        )
    )

    # Add a compact custom info summary for debugging
    evaluator.add_custom_info(
        info={
            "extracted_university_name": uni_name,
            "extracted_city": city,
            "extracted_state": state,
            "extracted_country": country,
            "program_name": program_name,
            "program_modality": modality,
            "certificate_name": certificate_name,
            "certificate_credit_hours": credit_hours,
            "accreditor_name": accreditor_name,
            "accreditor_abbrev": accreditor_abbrev,
            "total_sources_collected": len(all_sources),
        },
        info_type="extraction_summary"
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
) -> Dict[str, Any]:
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root per rubric is parallel
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_etl_university_info(),
        template_class=ETLUniversityExtraction,
        extraction_name="etl_university_info"
    )

    await build_and_verify_tree(evaluator, extracted, root)

    return evaluator.get_summary()