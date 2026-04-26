import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "lsu_flores_mba_checklist_fall_2026"
TASK_DESCRIPTION = """
Provide a complete application checklist for a domestic student applying to the Louisiana State University (LSU) Flores Full-Time MBA program for Fall 2026 admission. The checklist must include: all academic eligibility requirements, all required application materials and documents that must be submitted, all applicable fees, relevant deadlines, and any additional requirements or conditions. Please provide reference URLs to official LSU sources for verification.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class LSUChecklistItem(BaseModel):
    """Single checklist item extracted from the answer."""
    statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class LSUChecklistExtraction(BaseModel):
    """Complete extraction for the LSU Flores MBA application checklist."""
    # Official references mentioned anywhere in the answer
    all_urls_in_answer: List[str] = Field(default_factory=list)

    # Eligibility
    eligibility_bachelors: Optional[LSUChecklistItem] = None
    eligibility_min_gpa: Optional[LSUChecklistItem] = None

    # Application materials and actions
    material_grad_school_online_app: Optional[LSUChecklistItem] = None
    material_select_summer_2026: Optional[LSUChecklistItem] = None
    material_two_letters: Optional[LSUChecklistItem] = None
    material_personal_statement: Optional[LSUChecklistItem] = None
    material_professional_resume: Optional[LSUChecklistItem] = None
    material_cv_resume_grad_school: Optional[LSUChecklistItem] = None
    material_transcripts_all: Optional[LSUChecklistItem] = None

    # Transcript policy
    transcript_policy_official_and_provisional: Optional[LSUChecklistItem] = None

    # Fees
    fee_domestic_50: Optional[LSUChecklistItem] = None
    fee_late_25_after_deadline: Optional[LSUChecklistItem] = None

    # Deadline
    deadline_july_1_2026_domestic: Optional[LSUChecklistItem] = None

    # Additional conditions
    condition_full_time_residency: Optional[LSUChecklistItem] = None
    condition_interview_may_be_selected: Optional[LSUChecklistItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_checklist() -> str:
    return """
    Extract the LSU Flores Full-Time MBA application checklist information PRESENTED IN THE ANSWER.
    For each required item below:
    - statement: Return the exact sentence or phrase in the answer that corresponds to this requirement. If the answer does not include it, return null.
    - sources: Return an array of all URLs that the answer explicitly cites as references for this specific item. Only include URLs; if none are cited, return an empty array.

    Also extract a global list:
    - all_urls_in_answer: Array of ALL URLs (including markdown links) mentioned anywhere in the answer. Include full URLs with protocol.

    Items to extract, keyed exactly as follows:

    1) eligibility_bachelors: Bachelor's degree from an accredited U.S. institution.
    2) eligibility_min_gpa: Minimum GPA of 3.0 on a 4.0 scale on all previous coursework.
    3) material_grad_school_online_app: Complete the LSU Graduate School online application.
    4) material_select_summer_2026: Select "Summer 2026" as the program of interest when applying (for fall start).
    5) material_two_letters: Submit two letters of recommendation, preferably from past/current employers or professors.
    6) material_personal_statement: Provide a personal statement detailing career development goals and unique attributes.
    7) material_professional_resume: Upload a professional resume with the application.
    8) material_cv_resume_grad_school: Submit CV or resume with complete chronological outline of college-level education to the Graduate School.
    9) material_transcripts_all: Submit transcripts from all institutions attended.
    10) transcript_policy_official_and_provisional: Official transcripts (with seal and registrar signature) are required; unofficial may be used initially for provisional admission, but official transcripts must be submitted within 30 days after classes begin.
    11) fee_domestic_50: Pay the $50 domestic application fee.
    12) fee_late_25_after_deadline: Additional $25 late fee applies if application is received after the deadline.
    13) deadline_july_1_2026_domestic: Application deadline is July 1, 2026 for Fall 2026 admission (domestic students).
    14) condition_full_time_residency: Commit to full-time residency.
    15) condition_interview_may_be_selected: Applicant may be selected for an in-person or Zoom interview.

    IMPORTANT:
    - Do NOT invent or infer any URLs; only include those explicitly present in the answer.
    - Return null for a 'statement' if the answer does not include the item.
    - Return an empty array for 'sources' if the answer does not provide references for that item.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_official_lsu_url(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.endswith("lsu.edu")
    except Exception:
        return False


def gather_official_sources(item: Optional[LSUChecklistItem]) -> List[str]:
    if not item or not item.sources:
        return []
    return [u for u in item.sources if is_official_lsu_url(u)]


def union_sources_from_extraction(data: LSUChecklistExtraction) -> List[str]:
    urls: List[str] = []
    urls.extend(data.all_urls_in_answer or [])

    def add_item_sources(it: Optional[LSUChecklistItem]):
        if it and it.sources:
            urls.extend(it.sources)

    add_item_sources(data.eligibility_bachelors)
    add_item_sources(data.eligibility_min_gpa)
    add_item_sources(data.material_grad_school_online_app)
    add_item_sources(data.material_select_summer_2026)
    add_item_sources(data.material_two_letters)
    add_item_sources(data.material_personal_statement)
    add_item_sources(data.material_professional_resume)
    add_item_sources(data.material_cv_resume_grad_school)
    add_item_sources(data.material_transcripts_all)
    add_item_sources(data.transcript_policy_official_and_provisional)
    add_item_sources(data.fee_domestic_50)
    add_item_sources(data.fee_late_25_after_deadline)
    add_item_sources(data.deadline_july_1_2026_domestic)
    add_item_sources(data.condition_full_time_residency)
    add_item_sources(data.condition_interview_may_be_selected)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_official_urls_node(
    evaluator: Evaluator,
    parent,
    extracted: LSUChecklistExtraction,
) -> None:
    """
    Build the 'Provides_Official_LSU_Reference_URLs' critical check.
    Checks that the answer provides at least one official LSU URL (domain endswith 'lsu.edu').
    """
    all_urls = union_sources_from_extraction(extracted)
    official_urls = [u for u in all_urls if is_official_lsu_url(u)]
    has_official = len(official_urls) > 0

    evaluator.add_custom_info(
        info={
            "total_urls_in_answer": len(all_urls),
            "official_lsu_urls_count": len(official_urls),
            "official_lsu_urls": official_urls[:10],  # truncate preview
        },
        info_type="url_stats",
        info_name="official_url_statistics"
    )

    evaluator.add_custom_node(
        result=has_official,
        id="Provides_Official_LSU_Reference_URLs",
        desc="Provides reference URL(s) to official LSU sources for verification of the checklist requirements.",
        parent=parent,
        critical=True
    )


async def build_requirement_node(
    evaluator: Evaluator,
    parent,
    node_id: str,
    node_desc: str,
    item: Optional[LSUChecklistItem],
    claim: str,
    additional_instruction: str,
) -> None:
    """
    For each requirement:
    - Create a critical parallel sub-node.
    - Add a critical existence leaf: statement present AND at least one official LSU source URL present.
    - Add a critical verification leaf: claim must be supported by cited official LSU source(s).
    """
    item_node = evaluator.add_parallel(
        id=node_id,
        desc=node_desc,
        parent=parent,
        critical=True
    )

    text_present = bool(item and item.statement and item.statement.strip())
    official_sources = gather_official_sources(item)
    has_official_source = len(official_sources) > 0

    evaluator.add_custom_node(
        result=(text_present and has_official_source),
        id=f"{node_id}_exists",
        desc="Answer includes this requirement and provides official LSU source URL(s).",
        parent=item_node,
        critical=True
    )

    verify_leaf = evaluator.add_leaf(
        id=f"{node_id}_supported",
        desc=f"{node_desc} — supported by cited official LSU source(s).",
        parent=item_node,
        critical=True
    )

    # The evaluator's auto precondition logic will gate on the critical existence sibling.
    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=official_sources,
        additional_instruction=additional_instruction
    )


# --------------------------------------------------------------------------- #
# Specification map for items                                                 #
# --------------------------------------------------------------------------- #
def get_items_spec() -> List[Dict[str, str]]:
    return [
        {
            "id": "Eligibility_Bachelors_Degree_Accredited_US_Institution",
            "field": "eligibility_bachelors",
            "desc": "Includes the requirement: Bachelor's degree from an accredited U.S. institution.",
            "claim": "LSU Flores Full-Time MBA eligibility requires a bachelor's degree from an accredited U.S. institution.",
            "ins": "Verify the eligibility section on official LSU pages for the Flores Full-Time MBA or LSU Graduate School admissions stating the bachelor's degree requirement. Allow equivalent wording."
        },
        {
            "id": "Eligibility_Minimum_GPA_3_0_on_4_0_Scale",
            "field": "eligibility_min_gpa",
            "desc": "Includes the requirement: minimum GPA of 3.0 on a 4.0 scale on all previous coursework.",
            "claim": "LSU Flores Full-Time MBA eligibility requires a minimum GPA of 3.0 on a 4.0 scale on all previous coursework.",
            "ins": "Confirm the minimum GPA requirement on official LSU sites. Minor phrasing variations are acceptable."
        },
        {
            "id": "Application_Material_LSU_Graduate_School_Online_Application",
            "field": "material_grad_school_online_app",
            "desc": "Includes the requirement: complete the LSU Graduate School online application.",
            "claim": "Applicants must complete the LSU Graduate School online application.",
            "ins": "Check LSU Graduate School application instructions or MBA admissions pages indicating the online application requirement."
        },
        {
            "id": "Application_Material_Select_Summer_2026_Program_of_Interest",
            "field": "material_select_summer_2026",
            "desc": "Includes the requirement: select 'Summer 2026' as the program of interest when applying (for fall start).",
            "claim": "Applicants must select 'Summer 2026' as the program of interest when applying for a fall start.",
            "ins": "Some LSU MBA application portals may require selecting the Summer term for Fall cohort starts. Treat official instructions that indicate selecting Summer for Fall start as supporting evidence; if a page states generally 'select Summer for Fall start' without the year, consider it equivalent for 2026."
        },
        {
            "id": "Application_Material_Two_Letters_of_Recommendation_With_Preference_Noted",
            "field": "material_two_letters",
            "desc": "Includes the requirement: submit two letters of recommendation, preferably from past/current employers or professors.",
            "claim": "Applicants must submit two letters of recommendation, preferably from past or current employers or professors.",
            "ins": "Verify recommendation requirements (number and preference) on official LSU MBA or Graduate School pages."
        },
        {
            "id": "Application_Material_Personal_Statement_Career_Goals_and_Unique_Attributes",
            "field": "material_personal_statement",
            "desc": "Includes the requirement: provide a personal statement detailing career development goals and unique attributes.",
            "claim": "Applicants must provide a personal statement detailing career development goals and unique attributes.",
            "ins": "Check official LSU pages for personal statement or essay requirements; allow synonymous phrasing."
        },
        {
            "id": "Application_Material_Professional_Resume_Uploaded",
            "field": "material_professional_resume",
            "desc": "Includes the requirement: upload a professional resume with the application.",
            "claim": "Applicants must upload a professional resume with the application.",
            "ins": "Verify that a resume is required in the official LSU Graduate School or MBA application checklist."
        },
        {
            "id": "Application_Material_Graduate_School_CV_or_Resume_Chronological_Outline",
            "field": "material_cv_resume_grad_school",
            "desc": "Includes the requirement: submit CV or resume with complete chronological outline of college-level education to the Graduate School.",
            "claim": "Applicants must submit a CV or resume to the Graduate School that includes a complete chronological outline of college-level education.",
            "ins": "Look for Graduate School documentation detailing CV/resume formatting (chronological outline). Paraphrases are acceptable."
        },
        {
            "id": "Application_Material_Transcripts_All_Institutions_Attended",
            "field": "material_transcripts_all",
            "desc": "Includes the requirement: submit transcripts from all institutions attended.",
            "claim": "Applicants must submit transcripts from all institutions attended.",
            "ins": "Confirm transcript submission policies on official LSU Graduate School or MBA admissions pages."
        },
        {
            "id": "Transcript_Policy_Official_Transcripts_And_Provisional_Unofficial_Allowance",
            "field": "transcript_policy_official_and_provisional",
            "desc": "Includes the requirement: official transcripts (with seal and registrar signature) are required; unofficial may be used initially for provisional admission but official transcripts must be submitted within 30 days after classes begin.",
            "claim": "Official transcripts (with institutional seal and registrar signature) are required; unofficial transcripts may be used initially for provisional admission, but official transcripts must be submitted within 30 days after classes begin.",
            "ins": "Verify transcript policy details on official Graduate School admissions pages; minor variations in wording are acceptable."
        },
        {
            "id": "Fee_Domestic_Application_Fee_50",
            "field": "fee_domestic_50",
            "desc": "Includes the requirement: pay the $50 domestic application fee.",
            "claim": "Domestic applicants must pay a $50 application fee.",
            "ins": "Check the LSU Graduate School application fee schedule for domestic applicants."
        },
        {
            "id": "Fee_Late_Fee_25_After_Deadline",
            "field": "fee_late_25_after_deadline",
            "desc": "Includes the requirement: additional $25 late fee applies if application received after deadline.",
            "claim": "An additional $25 late fee applies if the application is received after the deadline.",
            "ins": "Confirm late fee policy on official LSU Graduate School admissions or fee pages."
        },
        {
            "id": "Deadline_Application_Deadline_July_1_2026",
            "field": "deadline_july_1_2026_domestic",
            "desc": "Includes the requirement: application deadline is July 1, 2026 for Fall 2026 admission (domestic students).",
            "claim": "The application deadline for Fall 2026 admission (domestic students) is July 1, 2026.",
            "ins": "If an official LSU page states 'Fall (Domestic): July 1' without a year, treat that as supporting the July 1, 2026 deadline for Fall 2026."
        },
        {
            "id": "Condition_Full_Time_Residency_Commitment",
            "field": "condition_full_time_residency",
            "desc": "Includes the requirement: commit to full-time residency.",
            "claim": "Applicants must commit to full-time residency.",
            "ins": "Verify that the Flores Full-Time MBA program requires full-time residency."
        },
        {
            "id": "Condition_Interview_Possibility_In_Person_or_Zoom",
            "field": "condition_interview_may_be_selected",
            "desc": "Includes the condition: applicant may be selected for an in-person or Zoom interview.",
            "claim": "Applicants may be selected for an interview conducted in-person or via Zoom.",
            "ins": "Check official LSU MBA admissions pages regarding interviews and allowed modalities."
        },
    ]


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
    Evaluate an answer for the LSU Flores Full-Time MBA (Fall 2026 domestic) application checklist.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel checks across checklist components
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

    # Extract checklist information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_checklist(),
        template_class=LSUChecklistExtraction,
        extraction_name="lsu_flores_checklist"
    )

    # Add ground truth info (names of expected items)
    evaluator.add_ground_truth({
        "expected_items": [spec["id"] for spec in get_items_spec()],
        "program": "LSU Flores Full-Time MBA (Domestic, Fall 2026)"
    })

    # 1) Check presence of at least one official LSU reference URL in the entire answer
    await build_official_urls_node(evaluator, root, extraction)

    # 2) Build verification nodes for each requirement
    specs = get_items_spec()

    # Map field name -> extracted item
    field_to_item: Dict[str, Optional[LSUChecklistItem]] = {
        "eligibility_bachelors": extraction.eligibility_bachelors,
        "eligibility_min_gpa": extraction.eligibility_min_gpa,
        "material_grad_school_online_app": extraction.material_grad_school_online_app,
        "material_select_summer_2026": extraction.material_select_summer_2026,
        "material_two_letters": extraction.material_two_letters,
        "material_personal_statement": extraction.material_personal_statement,
        "material_professional_resume": extraction.material_professional_resume,
        "material_cv_resume_grad_school": extraction.material_cv_resume_grad_school,
        "material_transcripts_all": extraction.material_transcripts_all,
        "transcript_policy_official_and_provisional": extraction.transcript_policy_official_and_provisional,
        "fee_domestic_50": extraction.fee_domestic_50,
        "fee_late_25_after_deadline": extraction.fee_late_25_after_deadline,
        "deadline_july_1_2026_domestic": extraction.deadline_july_1_2026_domestic,
        "condition_full_time_residency": extraction.condition_full_time_residency,
        "condition_interview_may_be_selected": extraction.condition_interview_may_be_selected,
    }

    # Build all requirement nodes (critical)
    for spec in specs:
        item = field_to_item.get(spec["field"])
        await build_requirement_node(
            evaluator=evaluator,
            parent=root,
            node_id=spec["id"],
            node_desc=spec["desc"],
            item=item,
            claim=spec["claim"],
            additional_instruction=spec["ins"]
        )

    # Return structured summary
    return evaluator.get_summary()