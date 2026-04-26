import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "or_grad_teacher_prep"
TASK_DESCRIPTION = (
    "Identify three Oregon public universities that offer graduate-level education programs leading to an Oregon preliminary teaching license. "
    "For each university, the program must meet all of the following requirements: "
    "(1) The institution must be one of Oregon's seven public universities, "
    "(2) The institution must offer graduate programs in education, "
    "(3) The educator preparation programs must hold national accreditation from either CAEP or AAQEP, "
    "(4) The program must be approved by Oregon's Teacher Standards and Practices Commission (TSPC), "
    "(5) Admission must require a bachelor's degree, "
    "(6) Admission must require a minimum 3.0 GPA, "
    "(7) The program must require a minimum of 45 graduate credits for completion, "
    "(8) The program must require at least two terms of study, "
    "(9) The program must lead to an Oregon preliminary teaching license, "
    "(10) The curriculum must include professional development coursework, "
    "(11) The curriculum must include a research component, "
    "(12) The institution must accept online applications or the Common Application. "
    "For each of the three universities, provide the institution name, verify that it meets all twelve requirements, and include a reference URL that supports your findings."
)

OREGON_PUBLIC_UNIVERSITIES = [
    "Eastern Oregon University",
    "Oregon Institute of Technology",
    "Oregon State University",
    "Portland State University",
    "Southern Oregon University",
    "University of Oregon",
    "Western Oregon University",
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramItem(BaseModel):
    institution_name: Optional[str] = None
    program_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class ProgramListExtraction(BaseModel):
    items: List[ProgramItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return (
        "Extract up to three Oregon public universities with graduate-level education programs that the answer presents. "
        "For each qualifying university mentioned in the answer, extract:\n"
        "1) institution_name: The full institution name as written in the answer.\n"
        "2) program_name: The program name or type (e.g., MAT, MEd, MS in Education) if provided; otherwise null.\n"
        "3) reference_urls: Collect all URLs in the answer that are associated with this university's education program, licensure, accreditation, or admissions. "
        "Include program pages, accreditation pages (CAEP or AAQEP), Oregon TSPC pages, and application/admissions pages if present. "
        "Only include URLs explicitly present in the answer text. Do not invent URLs.\n"
        "Return a JSON object with an array field 'items'. Each item should have the three fields above. "
        "If the answer contains more than three universities, keep the first three mentioned. "
        "If some fields are missing for a university, set them to null or empty list as appropriate."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_valid_url(u: Optional[str]) -> bool:
    if not u or not isinstance(u, str):
        return False
    ul = u.strip().lower()
    return ul.startswith("http://") or ul.startswith("https://")


def _clean_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if _is_valid_url(u)]


def _public_universities_instruction() -> str:
    return (
        "Consider the following to be Oregon's seven public universities:\n"
        f"{', '.join(OREGON_PUBLIC_UNIVERSITIES)}.\n"
        "Treat the check as satisfied if the webpage clearly corresponds to one of these institutions and it is a public university in Oregon. "
        "Allow reasonable naming variations (e.g., abbreviations, acronyms). Focus on whether the page shows the institution name and that it is a public university in Oregon."
    )


# --------------------------------------------------------------------------- #
# Verification per university                                                 #
# --------------------------------------------------------------------------- #
async def verify_one_university(
    evaluator: Evaluator,
    parent,
    uni_index: int,
    item: ProgramItem,
) -> None:
    inst_name = item.institution_name or ""
    prog_name = item.program_name or ""
    urls = _clean_urls(item.reference_urls)

    # University node (non-critical to allow partial across universities)
    uni_node = evaluator.add_parallel(
        id=f"university_{uni_index+1}",
        desc=f"{['First','Second','Third'][uni_index]} qualifying Oregon public university with graduate education program",
        parent=parent,
        critical=False,
    )

    # Reference URL existence (Critical leaf via custom node)
    ref_node = evaluator.add_custom_node(
        result=len(urls) > 0,
        id=f"u{uni_index+1}_reference_url",
        desc="Provide a valid reference URL for the program information",
        parent=uni_node,
        critical=True,
    )

    # 1. Public University Status
    leaf_pub = evaluator.add_leaf(
        id=f"u{uni_index+1}_public_university_status",
        desc="Verify the institution is one of Oregon's seven public universities",
        parent=uni_node,
        critical=True,
    )
    claim_pub = (
        f"The institution shown on the provided webpage(s) is '{inst_name}', and it is one of Oregon's public universities."
        if inst_name else
        "The institution on the provided webpage(s) is one of Oregon's public universities."
    )
    await evaluator.verify(
        claim=claim_pub,
        node=leaf_pub,
        sources=urls,
        extra_prerequisites=[ref_node],
        additional_instruction=_public_universities_instruction(),
    )

    # 2. Graduate Education Program Exists
    leaf_grad_prog = evaluator.add_leaf(
        id=f"u{uni_index+1}_graduate_education_program_exists",
        desc="Verify the institution offers graduate programs in education",
        parent=uni_node,
        critical=True,
    )
    claim_grad_prog = (
        f"The institution offers graduate programs in education (e.g., MAT, MEd, MS, etc.)."
    )
    await evaluator.verify(
        claim=claim_grad_prog,
        node=leaf_grad_prog,
        sources=urls,
        extra_prerequisites=[ref_node],
        additional_instruction=(
            "Look for explicit evidence of a graduate-level education program (such as Master of Arts in Teaching, Master of Education, "
            "or similar). Phrases like 'graduate program in education', 'MEd', 'MAT', or 'graduate teacher preparation' should count."
        ),
    )

    # 3. National Accreditation (CAEP or AAQEP)
    leaf_accred = evaluator.add_leaf(
        id=f"u{uni_index+1}_national_accreditation_status",
        desc="Verify the educator preparation programs hold national accreditation from CAEP or AAQEP",
        parent=uni_node,
        critical=True,
    )
    claim_accred = (
        "The educator preparation program is nationally accredited by CAEP or AAQEP (either organization satisfies this requirement)."
    )
    await evaluator.verify(
        claim=claim_accred,
        node=leaf_accred,
        sources=urls,
        extra_prerequisites=[ref_node],
        additional_instruction=(
            "Accept any explicit statement that the college/school/unit of education or educator preparation programs are accredited by CAEP "
            "(Council for Accreditation of Educator Preparation) or AAQEP (Association for Advancing Quality in Educator Preparation). "
            "Equivalent wording is acceptable."
        ),
    )

    # 4. TSPC Approval
    leaf_tspc = evaluator.add_leaf(
        id=f"u{uni_index+1}_tspc_approval_status",
        desc="Verify the program is approved by Oregon's Teacher Standards and Practices Commission (TSPC)",
        parent=uni_node,
        critical=True,
    )
    claim_tspc = "The educator preparation program is approved by Oregon's Teacher Standards and Practices Commission (TSPC)."
    await evaluator.verify(
        claim=claim_tspc,
        node=leaf_tspc,
        sources=urls,
        extra_prerequisites=[ref_node],
        additional_instruction=(
            "Look for an explicit statement that the program is TSPC-approved or a listing on an official TSPC page. "
            "If the source is a TSPC provider/program list page showing the institution/program, that satisfies the requirement."
        ),
    )

    # 5. Bachelor's Degree Requirement
    leaf_bach = evaluator.add_leaf(
        id=f"u{uni_index+1}_bachelor_degree_requirement",
        desc="Verify the program requires a bachelor's degree for admission",
        parent=uni_node,
        critical=True,
    )
    claim_bach = "Admission to the program requires a bachelor's degree."
    await evaluator.verify(
        claim=claim_bach,
        node=leaf_bach,
        sources=urls,
        extra_prerequisites=[ref_node],
        additional_instruction=(
            "Accept equivalent phrasing such as 'baccalaureate degree' or 'earned bachelor's degree required for admission'."
        ),
    )

    # 6. Minimum GPA 3.0 Requirement
    leaf_gpa = evaluator.add_leaf(
        id=f"u{uni_index+1}_minimum_gpa_requirement",
        desc="Verify the program requires a minimum 3.0 GPA for graduate admission",
        parent=uni_node,
        critical=True,
    )
    claim_gpa = "Admission requires a minimum GPA of 3.0 (3.00) for graduate admission."
    await evaluator.verify(
        claim=claim_gpa,
        node=leaf_gpa,
        sources=urls,
        extra_prerequisites=[ref_node],
        additional_instruction=(
            "Look for 'minimum 3.0 GPA' or 'GPA of 3.00 or higher'. If the page states a different minimum (e.g., 2.75), do not consider it satisfied."
        ),
    )

    # 7. Minimum Credits ≥ 45
    leaf_credits = evaluator.add_leaf(
        id=f"u{uni_index+1}_minimum_credits_requirement",
        desc="Verify the program requires a minimum of 45 graduate credits for degree completion",
        parent=uni_node,
        critical=True,
    )
    claim_credits = "The program requires at least 45 graduate credits (or credit hours) for completion."
    await evaluator.verify(
        claim=claim_credits,
        node=leaf_credits,
        sources=urls,
        extra_prerequisites=[ref_node],
        additional_instruction=(
            "Accept wording such as '45 credits', 'minimum of 45 graduate credits', or any requirement clearly indicating 45 or more graduate credits. "
            "Treat 'credits' and 'credit hours' as equivalent; do not attempt to convert between quarter and semester credits—accept if the page plainly says 45 or more."
        ),
    )

    # 8. Minimum Study Duration ≥ 2 Terms
    leaf_terms = evaluator.add_leaf(
        id=f"u{uni_index+1}_minimum_study_duration",
        desc="Verify the program requires at least two terms of study",
        parent=uni_node,
        critical=True,
    )
    claim_terms = "The program requires at least two academic terms (quarters or semesters) of study."
    await evaluator.verify(
        claim=claim_terms,
        node=leaf_terms,
        sources=urls,
        extra_prerequisites=[ref_node],
        additional_instruction=(
            "Accept explicit mentions of duration such as 'two terms', 'two quarters', 'two semesters', or any structure that clearly implies at least two academic terms."
        ),
    )

    # 9. Leads to Oregon Preliminary Teaching License
    leaf_license = evaluator.add_leaf(
        id=f"u{uni_index+1}_teaching_license_pathway",
        desc="Verify the program leads to an Oregon preliminary teaching license",
        parent=uni_node,
        critical=True,
    )
    claim_license = "This program leads to (or results in recommendation for) an Oregon Preliminary Teaching License."
    await evaluator.verify(
        claim=claim_license,
        node=leaf_license,
        sources=urls,
        extra_prerequisites=[ref_node],
        additional_instruction=(
            "Accept phrasing like 'leads to Oregon preliminary teaching licensure', 'recommends graduates for the Oregon Preliminary Teaching License', "
            "or equivalent language explicitly connecting the program to Oregon preliminary licensure."
        ),
    )

    # 10. Professional Development Coursework Included
    leaf_pd = evaluator.add_leaf(
        id=f"u{uni_index+1}_professional_development_coursework",
        desc="Verify the program includes required professional development courses",
        parent=uni_node,
        critical=True,
    )
    claim_pd = "The program's curriculum includes professional development coursework."
    await evaluator.verify(
        claim=claim_pd,
        node=leaf_pd,
        sources=urls,
        extra_prerequisites=[ref_node],
        additional_instruction=(
            "Look for 'professional development' course listings or requirements, or equivalent terms such as 'professional practice' courses. "
            "If the curriculum explicitly includes professional development components, count it as satisfied."
        ),
    )

    # 11. Research Component Included
    leaf_research = evaluator.add_leaf(
        id=f"u{uni_index+1}_research_component",
        desc="Verify the program includes a research component",
        parent=uni_node,
        critical=True,
    )
    claim_research = "The curriculum includes a research component."
    await evaluator.verify(
        claim=claim_research,
        node=leaf_research,
        sources=urls,
        extra_prerequisites=[ref_node],
        additional_instruction=(
            "Accept components such as 'research methods', 'action research', 'capstone research project', 'thesis', or similar research-oriented requirements."
        ),
    )

    # 12. Online Applications or Common Application
    leaf_apply = evaluator.add_leaf(
        id=f"u{uni_index+1}_application_system_acceptance",
        desc="Verify the institution accepts online applications or Common Application",
        parent=uni_node,
        critical=True,
    )
    claim_apply = "The institution accepts online applications or uses the Common Application for this program."
    await evaluator.verify(
        claim=claim_apply,
        node=leaf_apply,
        sources=urls,
        extra_prerequisites=[ref_node],
        additional_instruction=(
            "For graduate programs, an official online application portal (e.g., 'Apply Now' or Slate) satisfies this requirement. "
            "If Common Application is specifically accepted for this level, that also satisfies it."
        ),
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

    # Extract up to 3 universities with their associated URLs
    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramListExtraction,
        extraction_name="extracted_university_programs",
    )

    # Record helper info about Oregon public universities to aid interpretation
    evaluator.add_custom_info(
        info={"oregon_public_universities_list": OREGON_PUBLIC_UNIVERSITIES},
        info_type="reference",
        info_name="oregon_public_universities_reference",
    )

    # Prepare up to 3 items (pad with empty if fewer)
    items: List[ProgramItem] = list(extracted.items[:3])
    while len(items) < 3:
        items.append(ProgramItem())

    # Build three university subtrees
    for i in range(3):
        await verify_one_university(evaluator, root, i, items[i])

    return evaluator.get_summary()