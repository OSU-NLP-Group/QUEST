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
TASK_ID = "grad_program_selection_2026"
TASK_DESCRIPTION = (
    "You are an international student planning to pursue graduate studies in Computer Science or a closely related "
    "STEM field (such as Data Science, Computer Engineering, or Information Science) in the United States for Fall "
    "2026 enrollment. You need to identify a US university and specific graduate program that meet all of the "
    "following requirements:\n\n"
    "Program and Field:\n"
    "- Must be a graduate degree program (Master's or PhD) in Computer Science, Data Science, or a closely related STEM field\n"
    "- The university must be accredited by a recognized US regional accrediting body\n\n"
    "Financial Support:\n"
    "- The university must offer graduate assistantships (teaching, research, or general assistantship positions) to graduate students in the program\n"
    "- The minimum stipend for a full-time (approximately 20 hours per week) graduate assistantship must be at least $30,000 for a 9-month academic year\n"
    "- Graduate assistants must be required to maintain a minimum cumulative GPA of 3.0 to remain eligible\n\n"
    "English Proficiency:\n"
    "- The program must accept TOEFL iBT scores to demonstrate English language proficiency\n"
    "- The minimum TOEFL iBT score required for admission must be 90 or lower\n\n"
    "International Students:\n"
    "- The program must accept applications from international students\n"
    "- The university must be able to sponsor F-1 student visas for international graduate students\n\n"
    "Your Task:\n"
    "Identify one US university that meets all these requirements. Provide:\n"
    "1. The complete official name of the university\n"
    "2. The specific graduate program name\n"
    "3. Evidence that all requirements are satisfied, including reference URLs for:\n"
    "   - Graduate assistantship stipend rates and requirements\n"
    "   - English proficiency requirements (TOEFL scores)\n"
    "   - University accreditation status"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityProgram(BaseModel):
    university_name: Optional[str] = None
    program_name: Optional[str] = None
    degree_level: Optional[str] = None  # e.g., "MS", "M.S.", "Master of Science", "PhD", "Ph.D."
    field: Optional[str] = None
    program_urls: List[str] = Field(default_factory=list)  # official program/department/admissions page URLs


class AccreditationInfo(BaseModel):
    accreditation_urls: List[str] = Field(default_factory=list)


class AssistantshipInfo(BaseModel):
    assistantship_urls: List[str] = Field(default_factory=list)
    offers_assistantships: Optional[bool] = None
    stipend_min_9mo: Optional[str] = None  # string as written in the answer (e.g., "$31,500", "≥ $30,000")
    fulltime_hours_per_week: Optional[str] = None  # e.g., "20"
    gpa_minimum: Optional[str] = None  # e.g., "3.0"


class EnglishInfo(BaseModel):
    english_urls: List[str] = Field(default_factory=list)
    toefl_accepted: Optional[bool] = None
    toefl_min_score: Optional[str] = None  # e.g., "80", "90"


class InternationalInfo(BaseModel):
    international_urls: List[str] = Field(default_factory=list)
    accepts_international: Optional[bool] = None
    f1_visa_sponsorship: Optional[bool] = None  # or "issues I-20" etc.


class AllExtraction(BaseModel):
    university: UniversityProgram = UniversityProgram()
    accreditation: AccreditationInfo = AccreditationInfo()
    assistantship: AssistantshipInfo = AssistantshipInfo()
    english: EnglishInfo = EnglishInfo()
    international: InternationalInfo = InternationalInfo()


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract the selected university and graduate program information and the evidence URLs provided in the answer. 
    Only extract information explicitly present in the answer. Do not infer or add any missing information.

    Return a JSON object with the following structure and fields:

    {
      "university": {
        "university_name": string|null,      // The complete official name of the university as written in the answer
        "program_name": string|null,         // The specific graduate program name as written in the answer
        "degree_level": string|null,         // e.g., "MS", "Master of Science", "PhD", "Ph.D.", etc.
        "field": string|null,                // e.g., "Computer Science", "Data Science", "Computer Engineering", "Information Science", "AI", etc.
        "program_urls": string[]             // URLs to official program/department/admissions pages cited in the answer
      },
      "accreditation": {
        "accreditation_urls": string[]       // URLs that document the institution’s regional accreditation status
      },
      "assistantship": {
        "assistantship_urls": string[],      // URLs that document assistantship availability/requirements and stipend info
        "offers_assistantships": boolean|null,     // If explicitly stated in the answer text
        "stipend_min_9mo": string|null,            // Minimum 9-month stipend amount mentioned (as text), if present
        "fulltime_hours_per_week": string|null,    // Hours/week for full-time (e.g., "20"), if present in the answer
        "gpa_minimum": string|null                 // GPA threshold for eligibility (e.g., "3.0") if present
      },
      "english": {
        "english_urls": string[],            // URLs that document English proficiency requirements/TOEFL acceptance & minimums
        "toefl_accepted": boolean|null,      // If explicitly stated in the answer text
        "toefl_min_score": string|null       // Extract the minimum TOEFL iBT score cited (e.g., "80", "90")
      },
      "international": {
        "international_urls": string[],      // URLs that document international applicant info and F-1 sponsorship/I-20 issuance
        "accepts_international": boolean|null,
        "f1_visa_sponsorship": boolean|null
      }
    }

    IMPORTANT:
    - For URLs: extract only those explicitly present in the answer; include full URLs. If none are present for a section, return an empty array.
    - For booleans and numbers represented in text, only set them if explicitly stated in the answer; otherwise, set to null.
    - Preserve program and university names exactly as written in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _validate_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    seen = set()
    out: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        su = u.strip()
        if not su:
            continue
        # Basic validity: must look like a URL with protocol
        if not (su.startswith("http://") or su.startswith("https://")):
            continue
        if su not in seen:
            seen.add(su)
            out.append(su)
    return out


def _combine_sources(*lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        for u in lst:
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extraction: AllExtraction) -> None:
    # Sanitize URL lists
    program_urls = _validate_urls(extraction.university.program_urls)
    accreditation_urls = _validate_urls(extraction.accreditation.accreditation_urls)
    assistantship_urls = _validate_urls(extraction.assistantship.assistantship_urls)
    english_urls = _validate_urls(extraction.english.english_urls)
    international_urls = _validate_urls(extraction.international.international_urls)

    # Common fallback sources for program identity checks
    fallback_sources = _combine_sources(program_urls, accreditation_urls, assistantship_urls, english_urls, international_urls)

    # Root child: Graduate_Program_Identification (critical, parallel)
    gpi_node = evaluator.add_parallel(
        id="Graduate_Program_Identification",
        desc="Identify one US university and specific graduate program meeting all stated requirements for an international student, and provide required evidence URLs.",
        parent=evaluator.root,
        critical=True
    )

    # 1) University and Program (critical, parallel)
    up_node = evaluator.add_parallel(
        id="University_and_Program",
        desc="Identify the university and a qualifying graduate program in the required field and level.",
        parent=gpi_node,
        critical=True
    )

    # 1.a) University_Name (leaf)
    uni_name_node = evaluator.add_leaf(
        id="University_Name",
        desc="Provide the complete official name of the university.",
        parent=up_node,
        critical=True
    )
    uni_name = extraction.university.university_name or ""
    await evaluator.verify(
        claim=f"The official name of the university is '{uni_name}'.",
        node=uni_name_node,
        sources=fallback_sources,
        additional_instruction="Check the referenced official pages to confirm the institution's official name. "
                               "Allow stylistic variants like 'The University of X' vs 'University of X' if obviously the same institution."
    )

    # 1.b) Program_Name (leaf)
    prog_name_node = evaluator.add_leaf(
        id="Program_Name",
        desc="Provide the specific graduate program name.",
        parent=up_node,
        critical=True
    )
    program_name = extraction.university.program_name or ""
    await evaluator.verify(
        claim=f"The specific graduate program is '{program_name}'.",
        node=prog_name_node,
        sources=program_urls if program_urls else fallback_sources,
        additional_instruction="Verify that the program page or official university page clearly names the program as stated. "
                               "Equivalent formulations (e.g., 'M.S. in Computer Science' vs 'Master of Science in Computer Science') are acceptable."
    )

    # 1.c) Graduate_Degree_Level (leaf)
    degree_level_node = evaluator.add_leaf(
        id="Graduate_Degree_Level",
        desc="Confirm the program is a graduate degree program (Master’s or PhD).",
        parent=up_node,
        critical=True
    )
    await evaluator.verify(
        claim="This program is a graduate degree program at the Master's or PhD level.",
        node=degree_level_node,
        sources=program_urls if program_urls else fallback_sources,
        additional_instruction="Look for signals such as 'Master of Science (MS/M.S.)', 'Master of Engineering (MEng/M.Eng.)', "
                               "'Doctor of Philosophy (PhD/Ph.D.)', or explicit statements that it is a graduate program."
    )

    # 1.d) Program_Field (leaf)
    program_field_node = evaluator.add_leaf(
        id="Program_Field",
        desc="Confirm the program is in Computer Science, Data Science, or a closely related STEM field.",
        parent=up_node,
        critical=True
    )
    await evaluator.verify(
        claim="This program is in Computer Science, Data Science, Computer Engineering, Information Science, or a closely related STEM field.",
        node=program_field_node,
        sources=program_urls if program_urls else fallback_sources,
        additional_instruction="Accept closely related STEM graduate programs such as Artificial Intelligence, Software Engineering, "
                               "Machine Learning, Computer Engineering, Information Science, etc., if the connection is clear on the official page."
    )

    # 2) Accreditation_Status (critical, parallel)
    accred_node = evaluator.add_parallel(
        id="Accreditation_Status",
        desc="Verify the university is accredited by a recognized US regional accrediting body.",
        parent=gpi_node,
        critical=True
    )

    # 2.a) Regional_Accreditation (leaf)
    regional_accred_node = evaluator.add_leaf(
        id="Regional_Accreditation",
        desc="Confirm the university is accredited by a recognized US regional accrediting body (as specified in the constraints).",
        parent=accred_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The university '{uni_name}' is accredited by a recognized US regional accrediting body.",
        node=regional_accred_node,
        sources=accreditation_urls,
        additional_instruction=(
            "Confirm accreditation by any of the recognized regional bodies, including: "
            "HLC (Higher Learning Commission), MSCHE (Middle States Commission on Higher Education), "
            "SACSCOC (Southern Association of Colleges and Schools Commission on Colleges), "
            "WSCUC/WASC (WASC Senior College and University Commission), "
            "NECHE (New England Commission of Higher Education), or NWCCU (Northwest Commission on Colleges and Universities). "
            "The accreditation page should explicitly list the institution as accredited."
        )
    )

    # 2.b) Accreditation_Reference_URL (existence check)
    evaluator.add_custom_node(
        result=len(accreditation_urls) > 0,
        id="Accreditation_Reference_URL",
        desc="Provide a valid reference URL documenting the accreditation status.",
        parent=accred_node,
        critical=True
    )

    # 3) Graduate_Assistantship_Requirements (critical, parallel)
    ga_node = evaluator.add_parallel(
        id="Graduate_Assistantship_Requirements",
        desc="Verify graduate assistantship availability and the stated assistantship constraints.",
        parent=gpi_node,
        critical=True
    )

    # Combined sources for assistantship (program page + assistantship page can both be relevant)
    assistantship_sources = _combine_sources(assistantship_urls, program_urls)

    # 3.a) Assistantship_Availability (leaf)
    ga_avail_node = evaluator.add_leaf(
        id="Assistantship_Availability",
        desc="Confirm the university offers graduate assistantships (TA/RA/GA) to graduate students in the specified program.",
        parent=ga_node,
        critical=True
    )
    await evaluator.verify(
        claim="The university offers graduate assistantships (TA/RA/GA) to graduate students in this program or the department that houses it.",
        node=ga_avail_node,
        sources=assistantship_sources,
        additional_instruction="Look for explicit mentions of Teaching Assistantships (TA), Research Assistantships (RA), "
                               "or Graduate Assistantships (GA) available to graduate students in the program/department."
    )

    # 3.b) Minimum_Stipend (leaf)
    stipend_node = evaluator.add_leaf(
        id="Minimum_Stipend",
        desc="Verify the minimum 9-month stipend for a full-time graduate assistantship is at least $30,000.",
        parent=ga_node,
        critical=True
    )
    await evaluator.verify(
        claim="The minimum stipend for a full-time (approximately 20 hours/week) graduate assistantship for a 9-month academic year is at least $30,000.",
        node=stipend_node,
        sources=assistantship_urls,
        additional_instruction=(
            "Use the official assistantship/funding page(s). If amounts are monthly or per-semester, convert to a 9-month total. "
            "If a 12-month stipend is given, check whether the 9-month equivalent would be ≥ $30,000 (i.e., annual ≥ $40,000 implies 9-month ≥ $30,000). "
            "If multiple ranges/levels exist, the minimum for a full-time TA/RA/GA must be ≥ $30,000 for 9 months."
        )
    )

    # 3.c) Work_Hours_Specification (leaf)
    hours_node = evaluator.add_leaf(
        id="Work_Hours_Specification",
        desc="Confirm that a full-time graduate assistantship is approximately 20 hours per week.",
        parent=ga_node,
        critical=True
    )
    await evaluator.verify(
        claim="A full-time graduate assistantship is approximately 20 hours per week.",
        node=hours_node,
        sources=assistantship_urls,
        additional_instruction="Accept formulations such as '20 hours/week', 'no more than 20 hours/week', "
                               "'0.50 FTE equals 20 hours/week', or similar official language."
    )

    # 3.d) GPA_Maintenance_Requirement (leaf)
    gpa_node = evaluator.add_leaf(
        id="GPA_Maintenance_Requirement",
        desc="Verify graduate assistants must maintain a minimum cumulative GPA of 3.0 to remain eligible.",
        parent=ga_node,
        critical=True
    )
    await evaluator.verify(
        claim="Graduate assistants must maintain a minimum cumulative GPA of 3.0 to remain eligible.",
        node=gpa_node,
        sources=assistantship_urls,
        additional_instruction="Accept equivalent language such as 'maintain good academic standing defined as GPA ≥ 3.0' or explicit GPA 3.0 requirement in assistantship policies."
    )

    # 3.e) Assistantship_Reference_URL (existence check)
    evaluator.add_custom_node(
        result=len(assistantship_urls) > 0,
        id="Assistantship_Reference_URL",
        desc="Provide a valid reference URL documenting assistantship stipend rates and requirements.",
        parent=ga_node,
        critical=True
    )

    # 4) English_Proficiency_Requirements (critical, parallel)
    eng_node = evaluator.add_parallel(
        id="English_Proficiency_Requirements",
        desc="Verify TOEFL iBT acceptance and score threshold.",
        parent=gpi_node,
        critical=True
    )

    # 4.a) TOEFL_Acceptance (leaf)
    toefl_accept_node = evaluator.add_leaf(
        id="TOEFL_Acceptance",
        desc="Confirm the program accepts TOEFL iBT scores for English proficiency.",
        parent=eng_node,
        critical=True
    )
    await evaluator.verify(
        claim="The program/university accepts TOEFL iBT scores as proof of English proficiency for graduate admission.",
        node=toefl_accept_node,
        sources=english_urls,
        additional_instruction="Prefer program-level or graduate school policy pages. Acceptance of TOEFL iBT should be explicit."
    )

    # 4.b) TOEFL_Minimum_Score (leaf)
    toefl_min_node = evaluator.add_leaf(
        id="TOEFL_Minimum_Score",
        desc="Verify the minimum TOEFL iBT score required for admission is 90 or lower.",
        parent=eng_node,
        critical=True
    )
    await evaluator.verify(
        claim="The minimum required TOEFL iBT score for admission is 90 or lower.",
        node=toefl_min_node,
        sources=english_urls,
        additional_instruction="If the program/department specifies a different minimum than the graduate school, use the program-specific minimum. "
                               "If only an overall graduate minimum is listed, use that. Subscores can be ignored; focus on the total iBT minimum. "
                               "Accept values like 80, 85, 90; reject if strictly > 90."
    )

    # 4.c) English_Proficiency_Reference_URL (existence check)
    evaluator.add_custom_node(
        result=len(english_urls) > 0,
        id="English_Proficiency_Reference_URL",
        desc="Provide a valid reference URL documenting the English proficiency requirements.",
        parent=eng_node,
        critical=True
    )

    # 5) International_Student_Support (critical, parallel)
    intl_node = evaluator.add_parallel(
        id="International_Student_Support",
        desc="Verify the program/university supports international students per constraints.",
        parent=gpi_node,
        critical=True
    )

    # 5.a) International_Admission (leaf)
    intl_adm_node = evaluator.add_leaf(
        id="International_Admission",
        desc="Confirm the program accepts applications from international students.",
        parent=intl_node,
        critical=True
    )
    intl_sources = _combine_sources(international_urls, english_urls, program_urls)
    await evaluator.verify(
        claim="International students are accepted to apply for this graduate program.",
        node=intl_adm_node,
        sources=intl_sources,
        additional_instruction="Look for explicit statements that international applicants are accepted, or the presence of an 'International Applicants' section for the program/graduate admissions."
    )

    # 5.b) F1_Visa_Support (leaf)
    f1_node = evaluator.add_leaf(
        id="F1_Visa_Support",
        desc="Confirm the university can sponsor F-1 student visas for international graduate students.",
        parent=intl_node,
        critical=True
    )
    await evaluator.verify(
        claim="The university can sponsor F-1 student visas (issues I-20 for F-1) for international graduate students.",
        node=f1_node,
        sources=international_urls,
        additional_instruction="Accept clear indications of SEVP certification, I-20 issuance for F-1, or explicit statements that the institution sponsors F-1 visas for graduate students."
    )

    # Record some custom info to aid debugging
    evaluator.add_custom_info(
        info={
            "university_name": uni_name,
            "program_name": program_name,
            "degree_level": extraction.university.degree_level,
            "field": extraction.university.field,
            "url_counts": {
                "program_urls": len(program_urls),
                "accreditation_urls": len(accreditation_urls),
                "assistantship_urls": len(assistantship_urls),
                "english_urls": len(english_urls),
                "international_urls": len(international_urls),
            }
        },
        info_type="summary",
        info_name="extraction_summary"
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
    # Initialize evaluator with a parallel root
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extraction
    extraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=AllExtraction,
        extraction_name="selection_extraction"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extraction)

    # Return summary
    return evaluator.get_summary()