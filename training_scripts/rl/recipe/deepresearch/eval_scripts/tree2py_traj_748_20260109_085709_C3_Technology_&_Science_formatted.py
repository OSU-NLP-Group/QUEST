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
TASK_ID = "princeton_nature_transmon_2025"
TASK_DESCRIPTION = (
    "In 2025, researchers at Princeton University achieved a major breakthrough in quantum computing by developing "
    "a transmon qubit with a coherence time exceeding 1 millisecond. This work was published in Nature journal on "
    "November 5, 2025. Identify the PhD student who served as one of the two co-lead authors of this Nature publication, "
    "confirm their department affiliation at Princeton University, and identify one of their faculty advisors along "
    "with that advisor's administrative position at the university. Your answer must include: (1) The full name of the "
    "PhD student co-lead author, (2) The specific department name where this student is enrolled, (3) The full name of "
    "one faculty advisor, (4) The administrative position held by that faculty advisor."
)

EXPECTED_TITLE = "Millisecond lifetimes and coherence times in 2D transmon qubits"
EXPECTED_JOURNAL = "Nature"
EXPECTED_DATE_STR = "November 5, 2025"
ECE_CANONICAL_NAME = "Department of Electrical and Computer Engineering"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class NaturePaperTaskExtraction(BaseModel):
    """
    Structured extraction from the agent's answer for the Princeton/Nature (Nov 5, 2025) transmon-qubit task.

    This includes the four required output fields and the key publication/sources needed for verification.
    """
    # Required output fields
    student_full_name: Optional[str] = None
    student_department: Optional[str] = None
    advisor_full_name: Optional[str] = None
    advisor_admin_position: Optional[str] = None

    # Publication details referenced in the answer (if provided)
    paper_title: Optional[str] = None
    journal_name: Optional[str] = None
    publication_date: Optional[str] = None

    # URLs cited in the answer
    nature_paper_urls: List[str] = Field(default_factory=list)
    other_source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_nature_task() -> str:
    return """
    Extract the following structured information from the answer as it is explicitly written. Do not invent or infer.

    Required output fields (return null if missing):
    1. student_full_name: The full name of the PhD student co-lead author.
    2. student_department: The specific Princeton department where the student is enrolled or affiliated.
    3. advisor_full_name: The full name of one faculty advisor of that student (e.g., Andrew Houck or Nathalie de Leon).
    4. advisor_admin_position: The administrative position held by that advisor at Princeton University (e.g., Dean of the School of Engineering and Applied Science).

    Publication details referenced in the answer (return null if missing):
    5. paper_title: The paper title, if the answer states it.
    6. journal_name: The journal name, if the answer states it.
    7. publication_date: The publication date as stated in the answer (any reasonable format as given).

    URL sources explicitly cited in the answer:
    8. nature_paper_urls: All URLs pointing to the Nature journal page for the paper (e.g., nature.com articles or DOI landing pages).
    9. other_source_urls: Any other URLs cited in the answer (Princeton department pages, press releases, official announcements, faculty profiles, etc.).

    Rules for URLs:
    - Extract only valid URLs mentioned in the answer (plain URLs or inside markdown).
    - If a URL is missing a protocol, prepend http://.
    - Do not add URLs not present in the answer.

    Return a single JSON object with the above fields exactly as specified.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(data: NaturePaperTaskExtraction) -> List[str]:
    """Combine Nature paper URLs and other sources; remove duplicates while preserving order."""
    seen = set()
    combined: List[str] = []
    for url in (data.nature_paper_urls + data.other_source_urls):
        if url and url not in seen:
            seen.add(url)
            combined.append(url)
    return combined


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_paper_and_attribution_constraints(
    evaluator: Evaluator,
    parent_node,
    data: NaturePaperTaskExtraction,
) -> None:
    """
    Build the 'Paper_And_Attribution_Constraint_Verification' parallel critical subtree and verify each constraint.
    """
    sources_nature = data.nature_paper_urls
    sources_all = _combine_sources(data)

    # Critical parallel node for constraints
    constraints_node = evaluator.add_parallel(
        id="Paper_And_Attribution_Constraint_Verification",
        desc="Confirm the referenced work and attribution details match all stated constraints.",
        parent=parent_node,
        critical=True
    )

    # 1) Transmon qubit coherence > 1 ms
    node_res_perf = evaluator.add_leaf(
        id="Research_Type_and_Performance",
        desc="Work involves a transmon qubit with coherence time exceeding 1 millisecond.",
        parent=constraints_node,
        critical=True
    )
    claim_res_perf = (
        "The described Nature paper reports a transmon qubit with coherence time exceeding 1 millisecond "
        "(i.e., millisecond lifetimes or coherence times)."
    )
    await evaluator.verify(
        claim=claim_res_perf,
        node=node_res_perf,
        sources=sources_nature if sources_nature else sources_all,
        additional_instruction="Check the Nature page or associated official descriptions for 'millisecond lifetimes' or coherence times > 1 ms."
    )

    # 2) Published in Nature on November 5, 2025
    node_venue_date = evaluator.add_leaf(
        id="Publication_Venue_and_Date",
        desc="Published in Nature on November 5, 2025.",
        parent=constraints_node,
        critical=True
    )
    claim_venue_date = f"The paper was published in {EXPECTED_JOURNAL} on {EXPECTED_DATE_STR}."
    await evaluator.verify(
        claim=claim_venue_date,
        node=node_venue_date,
        sources=sources_nature if sources_nature else sources_all,
        additional_instruction="Verify both the venue (Nature) and the specific publication date (November 5, 2025) shown on the official page."
    )

    # 3) Exact title match
    node_title = evaluator.add_leaf(
        id="Paper_Title_Match",
        desc=f"Paper title is '{EXPECTED_TITLE}'.",
        parent=constraints_node,
        critical=True
    )
    claim_title = f"The Nature paper's title is exactly '{EXPECTED_TITLE}'."
    await evaluator.verify(
        claim=claim_title,
        node=node_title,
        sources=sources_nature if sources_nature else sources_all,
        additional_instruction="Match the title text on the Nature page; allow minor punctuation variations but require the same wording."
    )

    # 4) Two co-lead authors (co-first authors / equal contribution)
    node_two_colead = evaluator.add_leaf(
        id="Two_Co_Lead_Authors",
        desc="The paper has two co-lead authors.",
        parent=constraints_node,
        critical=True
    )
    claim_two_colead = "The paper explicitly indicates two co-lead (co-first, equal-contribution) authors."
    await evaluator.verify(
        claim=claim_two_colead,
        node=node_two_colead,
        sources=sources_nature if sources_nature else sources_all,
        additional_instruction="Look for 'These authors contributed equally' or similar equal-contribution note naming two authors."
    )

    # 5) One co-lead author is Matthew P. Bland (a PhD student)
    node_bland_colead = evaluator.add_leaf(
        id="Co_Lead_Author_Includes_Matthew_P_Bland",
        desc="One co-lead author is Matthew P. Bland (a PhD student).",
        parent=constraints_node,
        critical=True
    )
    claim_bland_colead = (
        "Matthew P. Bland is named as one of the two co-lead (co-first) authors on the Nature paper, and he is a PhD student at Princeton."
    )
    await evaluator.verify(
        claim=claim_bland_colead,
        node=node_bland_colead,
        sources=sources_all if sources_all else sources_nature,
        additional_instruction="Treat 'graduate student' as 'PhD student' when clearly in a PhD program context."
    )

    # 6) Bland department affiliation: Princeton ECE
    node_bland_ece = evaluator.add_leaf(
        id="Bland_Department_Affiliation_ECE",
        desc="Matthew P. Bland is affiliated with Princeton University's Department of Electrical and Computer Engineering.",
        parent=constraints_node,
        critical=True
    )
    claim_bland_ece = (
        "Matthew P. Bland is affiliated/enrolled in Princeton University's Department of Electrical and Computer Engineering (ECE)."
    )
    await evaluator.verify(
        claim=claim_bland_ece,
        node=node_bland_ece,
        sources=sources_all if sources_all else sources_nature,
        additional_instruction="Allow synonyms like 'Electrical Engineering' historically; confirm modern department naming is ECE at Princeton."
    )

    # 7) Bland advisors include Andrew Houck and Nathalie de Leon
    node_bland_advisors = evaluator.add_leaf(
        id="Bland_Advisors_Include_Houck_And_DeLeon",
        desc="Matthew P. Bland's advisors include Andrew Houck and Nathalie de Leon.",
        parent=constraints_node,
        critical=True
    )
    claim_bland_advisors = "Matthew P. Bland's faculty advisors include Andrew Houck and Nathalie de Leon."
    await evaluator.verify(
        claim=claim_bland_advisors,
        node=node_bland_advisors,
        sources=sources_all if sources_all else sources_nature,
        additional_instruction="Accept minor name variants: 'de León' vs 'de Leon'. Verify via Princeton official pages or press releases when available."
    )

    # 8) Houck administrative position: Dean of SEAS
    node_houck_dean = evaluator.add_leaf(
        id="Houck_Admin_Position_Dean_SEAS",
        desc="Andrew Houck holds the administrative position of Dean of the School of Engineering and Applied Science.",
        parent=constraints_node,
        critical=True
    )
    claim_houck_dean = "Andrew Houck is the Dean of Princeton's School of Engineering and Applied Science."
    await evaluator.verify(
        claim=claim_houck_dean,
        node=node_houck_dean,
        sources=sources_all if sources_all else sources_nature,
        additional_instruction="Verify via Princeton official announcements or leadership pages; allow phrasing variants like 'Dean of Engineering' for SEAS."
    )

    # 9) Primary funding source: DOE via C2QA
    node_funding = evaluator.add_leaf(
        id="Primary_Funding_Source",
        desc="Primary funding source is the U.S. Department of Energy through the Co-design Center for Quantum Advantage (C2QA).",
        parent=constraints_node,
        critical=True
    )
    claim_funding = (
        "The primary funding source for the work is the U.S. Department of Energy through the Co-design Center for Quantum Advantage (C2QA)."
    )
    await evaluator.verify(
        claim=claim_funding,
        node=node_funding,
        sources=sources_all if sources_all else sources_nature,
        additional_instruction="Check acknowledgements on the Nature page or official press releases for 'DOE' and 'C2QA'."
    )


async def verify_required_output_fields(
    evaluator: Evaluator,
    parent_node,
    data: NaturePaperTaskExtraction,
) -> None:
    """
    Build the 'Required_Output_Fields' parallel critical subtree to ensure the answer includes all four fields
    and that they are consistent with the verified constraints via source checks.
    """
    sources_nature = data.nature_paper_urls
    sources_all = _combine_sources(data)

    req_node = evaluator.add_parallel(
        id="Required_Output_Fields",
        desc="Ensure the response includes all four requested fields and that they are consistent with the verified constraints above.",
        parent=parent_node,
        critical=True
    )

    # 1) Student name present
    student_present = evaluator.add_custom_node(
        result=bool(data.student_full_name and data.student_full_name.strip()),
        id="Student_Co_Lead_Author_Full_Name_Present",
        desc="Response includes the full name of the PhD student co-lead author (consistent with the verified co-lead author constraint).",
        parent=req_node,
        critical=True
    )
    # Student consistency with co-lead author requirement
    student_consistency = evaluator.add_leaf(
        id="Student_Co_Lead_Author_Consistency",
        desc="Named student is a co-lead author of the specified Nature paper.",
        parent=req_node,
        critical=True
    )
    claim_student_consistency = (
        f"The student named in the answer ('{data.student_full_name or ''}') is one of the two co-lead (co-first) authors of the Nature paper."
    )
    await evaluator.verify(
        claim=claim_student_consistency,
        node=student_consistency,
        sources=sources_nature if sources_nature else sources_all,
        additional_instruction="Check the author list and equal-contribution note on the Nature page.",
        extra_prerequisites=[student_present]
    )

    # 2) Student department present
    dept_present = evaluator.add_custom_node(
        result=bool(data.student_department and data.student_department.strip()),
        id="Student_Department_Present",
        desc="Response includes the specific Princeton department where the student is enrolled (consistent with the verified department-affiliation constraint).",
        parent=req_node,
        critical=True
    )
    # Department consistency with ECE
    dept_consistency = evaluator.add_leaf(
        id="Student_Department_Consistency",
        desc="The named student's department matches Princeton ECE (allow reasonable naming variants).",
        parent=req_node,
        critical=True
    )
    claim_dept_consistency = (
        f"The student's department given in the answer ('{data.student_department or ''}') corresponds to Princeton University's Department of Electrical and Computer Engineering (ECE)."
    )
    await evaluator.verify(
        claim=claim_dept_consistency,
        node=dept_consistency,
        sources=sources_all if sources_all else sources_nature,
        additional_instruction="Allow variants such as 'Electrical Engineering' or 'Electrical & Computer Engineering' when clearly referring to Princeton ECE.",
        extra_prerequisites=[dept_present]
    )

    # 3) Advisor full name present
    advisor_present = evaluator.add_custom_node(
        result=bool(data.advisor_full_name and data.advisor_full_name.strip()),
        id="Advisor_Full_Name_Present",
        desc="Response includes the full name of one faculty advisor (must be one of the verified advisors).",
        parent=req_node,
        critical=True
    )
    # Advisor consistency (must be Houck or de Leon)
    advisor_consistency = evaluator.add_leaf(
        id="Advisor_Name_Consistency",
        desc="Named advisor is among the student's verified faculty advisors (Houck or de Leon).",
        parent=req_node,
        critical=True
    )
    claim_advisor_consistency = (
        f"The advisor named in the answer ('{data.advisor_full_name or ''}') is among the student's faculty advisors (Andrew Houck or Nathalie de Leon)."
    )
    await evaluator.verify(
        claim=claim_advisor_consistency,
        node=advisor_consistency,
        sources=sources_all if sources_all else sources_nature,
        additional_instruction="Accept minor name variants (e.g., 'de León' vs 'de Leon'). Confirm via Princeton official sources.",
        extra_prerequisites=[advisor_present]
    )

    # 4) Advisor administrative position present
    advisor_pos_present = evaluator.add_custom_node(
        result=bool(data.advisor_admin_position and data.advisor_admin_position.strip()),
        id="Advisor_Administrative_Position_Present",
        desc="Response includes an administrative position held by the named advisor at Princeton University; if the advisor is Andrew Houck, the position must match the verified dean position constraint.",
        parent=req_node,
        critical=True
    )
    # Advisor admin position consistency
    advisor_pos_consistency = evaluator.add_leaf(
        id="Advisor_Administrative_Position_Consistency",
        desc="Advisor administrative position matches official Princeton role; if advisor is Houck, it must be Dean of SEAS.",
        parent=req_node,
        critical=True
    )
    claim_advisor_pos_consistency = (
        f"The administrative position provided in the answer for '{data.advisor_full_name or ''}' "
        f"('{data.advisor_admin_position or ''}') matches their official Princeton role; "
        f"if the advisor is Andrew Houck, it should be 'Dean of the School of Engineering and Applied Science' (or equivalent phrasing)."
    )
    await evaluator.verify(
        claim=claim_advisor_pos_consistency,
        node=advisor_pos_consistency,
        sources=sources_all if sources_all else sources_nature,
        additional_instruction="Use Princeton official pages or announcements; allow minor wording variants like 'Dean of Engineering' for SEAS.",
        extra_prerequisites=[advisor_pos_present, advisor_present]
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
    Evaluate the agent's answer for the Princeton/Nature transmon-qubit (Nov 5, 2025) task.
    """
    # Initialize evaluator with sequential root to reflect the overall flow
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

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_nature_task(),
        template_class=NaturePaperTaskExtraction,
        extraction_name="nature_task_extraction"
    )

    # Record expected constants for transparency
    evaluator.add_ground_truth({
        "expected_title": EXPECTED_TITLE,
        "expected_journal": EXPECTED_JOURNAL,
        "expected_date": EXPECTED_DATE_STR,
        "expected_dept_name_canonical": ECE_CANONICAL_NAME
    }, gt_type="expected_constants")

    # Build and run verification subtrees
    # 1) Paper & attribution constraints (critical parallel)
    await verify_paper_and_attribution_constraints(evaluator, root, extracted)

    # 2) Required output fields presence + consistency (critical parallel)
    await verify_required_output_fields(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()