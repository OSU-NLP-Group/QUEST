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
TASK_ID = "edu_leadership_recruitment_4_positions"
TASK_DESCRIPTION = """
You are assisting a national educational leadership recruitment firm that is filling four concurrent positions. For each of the following four leadership positions, identify one qualified candidate who meets all the minimum requirements for that specific role. Each candidate must be a real person whose qualifications can be verified through publicly available information.

Position 1: Superintendent for a public school district in Virginia
- Requirements: Master's degree minimum in educational leadership or related field; at least 5 years of educational experience with at least 2 years of full-time teaching; eligible for or listed on Virginia's "Eligible List of Division Superintendents"

Position 2: Superintendent for a public school district in Colorado
- Requirements: Master's degree minimum in educational leadership or related field; at least 2-5 years of administrative/supervisory experience; holds or is eligible for Colorado superintendent licensure

Position 3: Athletic Director for an NCAA Division I university
- Requirements: Bachelor's degree minimum (Master's preferred) in sports management, physical education, or related field; at least 3-5 years of collegiate athletic administration or coaching experience; demonstrated knowledge of NCAA rules and compliance

Position 4: Provost (Chief Academic Officer) for a four-year university
- Requirements: Doctoral degree (Ph.D. or Ed.D.); prior experience as a tenured faculty member; prior senior administrative experience (such as dean or department chair); established record of teaching and scholarly publications

For each position, provide:
1. The candidate's name
2. A brief description of their qualifications demonstrating they meet all minimum requirements for that specific position
3. At least one reference URL that verifies their qualifications

Note: The same person cannot be proposed for multiple positions. Each candidate must currently hold qualifications that would make them eligible to apply for their respective position, based on publicly verifiable information.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CandidateInfo(BaseModel):
    name: Optional[str] = None
    qualification_summary: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CandidatesExtraction(BaseModel):
    position_1: Optional[CandidateInfo] = None  # Virginia Superintendent
    position_2: Optional[CandidateInfo] = None  # Colorado Superintendent
    position_3: Optional[CandidateInfo] = None  # NCAA Division I Athletic Director
    position_4: Optional[CandidateInfo] = None  # University Provost


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_candidates() -> str:
    return """
    From the provided answer, extract exactly one candidate for each of the four positions. If multiple are mentioned for a position, select the first one listed. For each position, extract:
    - name: Candidate's full name as written in the answer
    - qualification_summary: A brief summary (as written in the answer) describing how they meet the minimum requirements
    - urls: A list of all reference URLs explicitly provided in the answer for that candidate (do not invent any URLs; include valid http/https links; accept raw URLs or markdown links)

    Return a JSON object with these four fields at the top level:
    - position_1
    - position_2
    - position_3
    - position_4

    Each of those fields should be a JSON object with keys: name, qualification_summary, urls (array of strings).
    If a position does not have a candidate or any field is missing, set that field to null (or empty list for urls).

    IMPORTANT:
    - Only extract URLs explicitly present in the answer text. Do not infer or fabricate.
    - Normalize markdown links to the actual URL.
    - Keep the candidate's name exactly as written in the answer (minus obvious typos).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_name(name: Optional[str]) -> str:
    return name or "the candidate"


def _source_list(cand: Optional[CandidateInfo]) -> List[str]:
    if not cand:
        return []
    return [u for u in cand.urls if isinstance(u, str) and len(u.strip()) > 0][:10]


# --------------------------------------------------------------------------- #
# Position-specific verification builders                                     #
# --------------------------------------------------------------------------- #
async def verify_position_1_virginia_superintendent(
    evaluator: Evaluator,
    parent_node,
    cand: Optional[CandidateInfo],
) -> None:
    """
    Position 1: Superintendent (Virginia)
    Leaves:
      - position_1_education_credential (critical)
      - position_1_teaching_experience (critical)
      - position_1_total_experience (critical)
      - position_1_virginia_eligibility (critical)
      - position_1_reference_url (critical)
    """
    pos_node = evaluator.add_parallel(
        id="position_1_virginia_superintendent",
        desc="Evaluate the candidate identified for the Virginia public school superintendent position",
        parent=parent_node,
        critical=True  # Child of critical root → must be critical
    )

    name = _safe_name(cand.name if cand else None)
    urls = _source_list(cand)

    # 1) Reference URL (evaluate first to act as a gating critical sibling)
    ref_leaf = evaluator.add_leaf(
        id="position_1_reference_url",
        desc="A valid reference URL is provided that verifies the candidate's qualifications",
        parent=pos_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"At least one of these webpages is a publicly accessible page that mentions {name} and provides biographical or professional qualifications.",
        node=ref_leaf,
        sources=urls,
        additional_instruction="Accept university, district, government, or reputable news/association bios. Page must mention the person and discuss qualifications or roles."
    )

    # 2) Education credential
    edu_leaf = evaluator.add_leaf(
        id="position_1_education_credential",
        desc="The candidate holds at minimum a master's degree in educational leadership, educational administration, or a closely related field from an accredited institution",
        parent=pos_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} holds at least a master's degree in educational leadership, educational administration, curriculum and instruction, public administration, or a closely related education field.",
        node=edu_leaf,
        sources=urls,
        additional_instruction="Look for degrees such as M.Ed., M.A., M.S., Ed.S., or equivalent in education-related domains; university/official bios preferred."
    )

    # 3) At least two years of full-time teaching experience
    teach_leaf = evaluator.add_leaf(
        id="position_1_teaching_experience",
        desc="The candidate has completed at least two years of full-time teaching experience in a public or accredited nonpublic school",
        parent=pos_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The webpages indicate {name} has at least two years of full-time K-12 classroom teaching experience.",
        node=teach_leaf,
        sources=urls,
        additional_instruction="Evidence can include explicit years as a teacher or multiple school-year roles as a teacher. Coaching-only or central-office roles alone do not satisfy this requirement."
    )

    # 4) At least five years of successful educational experience
    total_exp_leaf = evaluator.add_leaf(
        id="position_1_total_experience",
        desc="The candidate has completed at least five years of successful educational experience in public or accredited nonpublic schools",
        parent=pos_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The webpages show that {name} has at least five total years of professional experience in K-12 education (teaching and/or administrative).",
        node=total_exp_leaf,
        sources=urls,
        additional_instruction="Count combined years across teaching, school-based admin (e.g., principal), and district roles. News or official bios acceptable."
    )

    # 5) Virginia eligibility / listing
    va_elig_leaf = evaluator.add_leaf(
        id="position_1_virginia_eligibility",
        desc="The candidate is eligible for or currently listed on the Virginia Board of Education's 'Eligible List of Division Superintendents' (or meets all requirements to obtain such eligibility)",
        parent=pos_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The webpages indicate that {name} is either listed on, or eligible for, the Virginia Board of Education 'Eligible List of Division Superintendents' (or otherwise meets all state requirements for such eligibility).",
        node=va_elig_leaf,
        sources=urls,
        additional_instruction="Accept explicit statements of eligibility, licensure, or active service as a Virginia division superintendent that implies such eligibility. Prefer VA DOE pages, district announcements, or official statements."
    )


async def verify_position_2_colorado_superintendent(
    evaluator: Evaluator,
    parent_node,
    cand: Optional[CandidateInfo],
) -> None:
    """
    Position 2: Superintendent (Colorado)
    Leaves:
      - position_2_education_credential (critical)
      - position_2_administrative_experience (critical)
      - position_2_colorado_licensure (critical)
      - position_2_reference_url (critical)
    """
    pos_node = evaluator.add_parallel(
        id="position_2_colorado_superintendent",
        desc="Evaluate the candidate identified for the Colorado public school superintendent position",
        parent=parent_node,
        critical=True
    )

    name = _safe_name(cand.name if cand else None)
    urls = _source_list(cand)

    # 1) Reference URL (gate)
    ref_leaf = evaluator.add_leaf(
        id="position_2_reference_url",
        desc="A valid reference URL is provided that verifies the candidate's qualifications",
        parent=pos_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"At least one of these webpages is a publicly accessible page that mentions {name} and provides biographical or professional qualifications.",
        node=ref_leaf,
        sources=urls,
        additional_instruction="Accept official district/university/government bios, press releases, reputable media, or professional profiles with verifiable credentials."
    )

    # 2) Education credential
    edu_leaf = evaluator.add_leaf(
        id="position_2_education_credential",
        desc="The candidate holds at minimum a master's degree in educational leadership, educational administration, or a closely related field from an accredited institution",
        parent=pos_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} holds at least a master's degree in educational leadership, educational administration, curriculum & instruction, public administration, or a closely related education field.",
        node=edu_leaf,
        sources=urls,
        additional_instruction="Look for M.Ed., M.A., M.S., Ed.S., or similar in education or closely related fields from accredited institutions."
    )

    # 3) Administrative/supervisory experience (≥ 2 years)
    admin_exp_leaf = evaluator.add_leaf(
        id="position_2_administrative_experience",
        desc="The candidate has at least 2-5 years of school administrative or supervisory experience",
        parent=pos_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The webpages show that {name} has at least two years of school administrative or supervisory experience (e.g., principal, assistant principal, director, assistant superintendent, superintendent).",
        node=admin_exp_leaf,
        sources=urls,
        additional_instruction="Evidence may include multiple school years in admin roles; titles and dates should indicate at least two years."
    )

    # 4) Colorado superintendent licensure (holds or eligible)
    co_lic_leaf = evaluator.add_leaf(
        id="position_2_colorado_licensure",
        desc="The candidate holds or is eligible to obtain Colorado superintendent licensure/certification",
        parent=pos_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The webpages indicate that {name} holds, previously held, or is eligible to obtain Colorado superintendent licensure/certification.",
        node=co_lic_leaf,
        sources=urls,
        additional_instruction="Accept explicit statements of licensure or eligibility in Colorado, or district/BOE materials indicating superintendent appointment consistent with CO requirements."
    )


async def verify_position_3_ncaa_athletic_director(
    evaluator: Evaluator,
    parent_node,
    cand: Optional[CandidateInfo],
) -> None:
    """
    Position 3: NCAA Division I Athletic Director
    Leaves:
      - position_3_education_credential (critical)
      - position_3_experience (critical)
      - position_3_ncaa_knowledge (critical)
      - position_3_reference_url (critical)
    """
    pos_node = evaluator.add_parallel(
        id="position_3_ncaa_athletic_director",
        desc="Evaluate the candidate identified for the NCAA Division I athletic director position",
        parent=parent_node,
        critical=True
    )

    name = _safe_name(cand.name if cand else None)
    urls = _source_list(cand)

    # 1) Reference URL (gate)
    ref_leaf = evaluator.add_leaf(
        id="position_3_reference_url",
        desc="A valid reference URL is provided that verifies the candidate's qualifications",
        parent=pos_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"At least one of these webpages is a publicly accessible page that mentions {name} and provides biographical or professional qualifications.",
        node=ref_leaf,
        sources=urls,
        additional_instruction="Prefer official university athletics bios, NCAA/conference pages, or reputable media profiles."
    )

    # 2) Education credential (≥ bachelor's in relevant field)
    edu_leaf = evaluator.add_leaf(
        id="position_3_education_credential",
        desc="The candidate holds at minimum a bachelor's degree in sports management, physical education, business administration, or a related field from an accredited institution",
        parent=pos_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} holds at least a bachelor's degree in a relevant field (e.g., sports management, physical education, kinesiology, business administration, or a related area).",
        node=edu_leaf,
        sources=urls,
        additional_instruction="A master's degree is acceptable and also satisfies the minimum; confirm institution and field are consistent."
    )

    # 3) Experience (≥ 3 years collegiate athletic admin/coaching)
    exp_leaf = evaluator.add_leaf(
        id="position_3_experience",
        desc="The candidate has at least 3-5 years of experience in athletic administration, coaching, or sports management at the collegiate level",
        parent=pos_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The webpages indicate {name} has at least three years of collegiate (NCAA) athletic administration, compliance, operations, or coaching experience.",
        node=exp_leaf,
        sources=urls,
        additional_instruction="Titles like assistant/associate AD, compliance director, operations, or multi-year collegiate coaching count; ensure collegiate context."
    )

    # 4) Demonstrated knowledge of NCAA rules/compliance
    ncaa_leaf = evaluator.add_leaf(
        id="position_3_ncaa_knowledge",
        desc="The candidate demonstrates knowledge of NCAA rules, regulations, and compliance requirements",
        parent=pos_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The webpages demonstrate that {name} has knowledge of NCAA rules and compliance (e.g., oversight responsibility, prior compliance role, or explicit mention of NCAA compliance expertise).",
        node=ncaa_leaf,
        sources=urls,
        additional_instruction="Look for phrases like 'NCAA compliance', 'rules education', 'compliance oversight', or job duties involving NCAA governance."
    )


async def verify_position_4_university_provost(
    evaluator: Evaluator,
    parent_node,
    cand: Optional[CandidateInfo],
) -> None:
    """
    Position 4: University Provost (Chief Academic Officer)
    Leaves:
      - position_4_doctoral_degree (critical)
      - position_4_faculty_experience (critical)
      - position_4_administrative_experience (critical)
      - position_4_scholarly_record (critical)
      - position_4_reference_url (critical)
    """
    pos_node = evaluator.add_parallel(
        id="position_4_university_provost",
        desc="Evaluate the candidate identified for the university provost position",
        parent=parent_node,
        critical=True
    )

    name = _safe_name(cand.name if cand else None)
    urls = _source_list(cand)

    # 1) Reference URL (gate)
    ref_leaf = evaluator.add_leaf(
        id="position_4_reference_url",
        desc="A valid reference URL is provided that verifies the candidate's qualifications",
        parent=pos_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"At least one of these webpages is a publicly accessible page that mentions {name} and provides biographical or professional qualifications.",
        node=ref_leaf,
        sources=urls,
        additional_instruction="Prefer official university leadership bios, faculty pages, or reputable academic/professional profiles."
    )

    # 2) Doctoral degree
    doc_leaf = evaluator.add_leaf(
        id="position_4_doctoral_degree",
        desc="The candidate holds a doctoral degree (Ph.D., Ed.D., or equivalent terminal degree) from an accredited institution",
        parent=pos_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The webpages indicate {name} holds a doctoral or equivalent terminal degree (e.g., Ph.D., Ed.D., or similar).",
        node=doc_leaf,
        sources=urls,
        additional_instruction="Degree field and granting institution should be indicated or clearly implied."
    )

    # 3) Tenured faculty experience
    fac_leaf = evaluator.add_leaf(
        id="position_4_faculty_experience",
        desc="The candidate has prior experience as a tenured faculty member at an accredited college or university",
        parent=pos_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The webpages show that {name} has served as a tenured faculty member (e.g., tenured associate or full professor).",
        node=fac_leaf,
        sources=urls,
        additional_instruction="Look for explicit 'tenured' or 'with tenure'; tenure-track alone without tenure does not satisfy unless tenure is clearly achieved."
    )

    # 4) Senior administrative experience
    admin_leaf = evaluator.add_leaf(
        id="position_4_administrative_experience",
        desc="The candidate has prior senior administrative experience in higher education, such as serving as a dean, associate provost, or department chair",
        parent=pos_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The webpages indicate {name} has held senior academic administrative roles such as dean, associate/vice provost, or department chair.",
        node=admin_leaf,
        sources=urls,
        additional_instruction="Other comparable senior roles (e.g., vice dean, executive director overseeing academics) also count if responsibilities are at senior academic leadership level."
    )

    # 5) Established record of teaching and scholarly publications
    schol_leaf = evaluator.add_leaf(
        id="position_4_scholarly_record",
        desc="The candidate has an established record of teaching and scholarly publications in their academic discipline",
        parent=pos_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The webpages show {name} has a record of university-level teaching and scholarly publications (e.g., peer-reviewed articles, books, chapters).",
        node=schol_leaf,
        sources=urls,
        additional_instruction="Accept mention of multiple peer-reviewed publications, books, or equivalent scholarship plus teaching history."
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
    Evaluate an answer for the four-position educational leadership recruitment task.
    """
    # Initialize evaluator (root node is created non-critical by design)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # As per rubric root aggregation
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Evaluate whether all four educational leadership positions have been correctly matched with qualified candidates who meet the respective minimum requirements",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Create a critical gate node under root to reflect rubric's critical root requirement
    task_gate = evaluator.add_parallel(
        id="task_gate_root",
        desc="Evaluate whether all four educational leadership positions have been correctly matched with qualified candidates who meet the respective minimum requirements",
        parent=root,
        critical=True
    )

    # Extract structured candidates info
    extracted = await evaluator.extract(
        prompt=prompt_extract_candidates(),
        template_class=CandidatesExtraction,
        extraction_name="candidates_extraction"
    )

    # Optionally record simple custom info (counts)
    try:
        evaluator.add_custom_info(
            info={
                "p1_urls_count": len(extracted.position_1.urls) if extracted.position_1 and extracted.position_1.urls else 0,
                "p2_urls_count": len(extracted.position_2.urls) if extracted.position_2 and extracted.position_2.urls else 0,
                "p3_urls_count": len(extracted.position_3.urls) if extracted.position_3 and extracted.position_3.urls else 0,
                "p4_urls_count": len(extracted.position_4.urls) if extracted.position_4 and extracted.position_4.urls else 0,
            },
            info_type="url_counts",
            info_name="url_counts_summary"
        )
    except Exception:
        pass

    # Build verification subtrees for each position
    await verify_position_1_virginia_superintendent(evaluator, task_gate, extracted.position_1)
    await verify_position_2_colorado_superintendent(evaluator, task_gate, extracted.position_2)
    await verify_position_3_ncaa_athletic_director(evaluator, task_gate, extracted.position_3)
    await verify_position_4_university_provost(evaluator, task_gate, extracted.position_4)

    # Return structured evaluation summary
    return evaluator.get_summary()