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
TASK_ID = "ccc_transfer_planner"
TASK_DESCRIPTION = """You are helping a high school senior in California plan their community college pathway to transfer to a four-year university. To provide comprehensive options, identify four different California community colleges that meet all of the following requirements:

1. System Membership: Each college must be one of the 116 colleges in the California Community Colleges system, and you must identify which of the 73 community college districts each college belongs to.

2. Accreditation: Each college must be currently accredited by ACCJC (the Accrediting Commission for Community and Junior Colleges, part of WASC).

3. UC Transfer Pathway: Each college must participate in the UC Transfer Admission Guarantee (TAG) program with at least one of the six TAG-eligible UC campuses (UC Davis, UC Irvine, UC Merced, UC Riverside, UC Santa Barbara, or UC Santa Cruz). Specify at least one TAG-eligible campus for each college.

4. CSU Transfer Pathway: Each college must offer Associate Degree for Transfer (ADT) programs, which guarantee admission to the California State University system. Identify at least one ADT program (either AA-T or AS-T) offered by each college.

5. General Education Certification: Each college must offer both IGETC (Intersegmental General Education Transfer Curriculum) pattern certification and CSU GE-Breadth pattern certification for transfer students.

For each of the four colleges, provide:
- The official college name
- The community college district it belongs to
- Confirmation of ACCJC accreditation status
- At least one UC campus for which the college offers TAG
- At least one ADT program offered
- Confirmation of IGETC and CSU GE-Breadth certification availability
- Supporting reference URLs for each piece of information (including links to official college websites, ACCJC.org, ASSIST.org, UC TAG resources, CSU ADT resources, or California Community Colleges Chancellor's Office listings)

Your response should help the student understand the breadth of transfer options available across different California community colleges.
"""

ALLOWED_TAG_CAMPUSES = {
    "UC Davis", "UC Irvine", "UC Merced", "UC Riverside", "UC Santa Barbara", "UC Santa Cruz"
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CollegeItem(BaseModel):
    name: Optional[str] = None
    district: Optional[str] = None

    # Status/flags as free-form strings (e.g., "yes"/"no", or textual confirmation)
    accjc_accredited: Optional[str] = None
    igetc_available: Optional[str] = None
    csu_ge_available: Optional[str] = None

    # Program lists
    tag_campuses: List[str] = Field(default_factory=list)
    adt_programs: List[str] = Field(default_factory=list)

    # Supporting URLs (explicitly cited in the answer)
    info_urls: List[str] = Field(default_factory=list)             # Official college site or CCCCO listing links
    accreditation_urls: List[str] = Field(default_factory=list)    # accjc.org listing or college accreditation page
    tag_urls: List[str] = Field(default_factory=list)              # UC TAG matrix / college transfer center / ASSIST
    adt_urls: List[str] = Field(default_factory=list)              # CSU ADT resources / degree lists / catalog
    ge_urls: List[str] = Field(default_factory=list)               # ASSIST / catalog / transfer center GE pages


class TransferCollegesExtraction(BaseModel):
    colleges: List[CollegeItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_transfer_colleges() -> str:
    return """
    Extract up to six distinct California community colleges mentioned in the answer, along with transfer-related details and cited URLs.

    For each identified college, extract the following fields exactly as stated in the answer:
    - name: The official college name (string).
    - district: The college's community college district name (string).
    - accjc_accredited: Whether the answer explicitly confirms ACCJC accreditation (use a short string like "yes", "no", "accredited", or null if not stated).
    - igetc_available: Whether the answer explicitly confirms IGETC certification availability (use "yes"/"no" or null if not stated).
    - csu_ge_available: Whether the answer explicitly confirms CSU GE-Breadth certification availability (use "yes"/"no" or null if not stated).
    - tag_campuses: List of UC campuses explicitly named in the answer for which the college offers TAG. Only include campuses if they are explicitly named in the answer; otherwise leave empty. Do NOT invent campuses. Campus names should match one of: ["UC Davis", "UC Irvine", "UC Merced", "UC Riverside", "UC Santa Barbara", "UC Santa Cruz"] when possible; if variants are used (e.g., "UC Irvine (UCI)"), keep the text exactly as in the answer.
    - adt_programs: List of ADT programs (AA-T or AS-T) explicitly named in the answer for the college (e.g., "Psychology AA-T", "Computer Science AS-T"). If none are named, leave empty.
    - info_urls: List of URLs cited for basic college information (official college website or CCCCO listing). If none are cited, leave empty.
    - accreditation_urls: List of URLs cited to confirm ACCJC accreditation (e.g., accjc.org directory or college accreditation page). If none are cited, leave empty.
    - tag_urls: List of URLs cited to confirm UC TAG participation for the college (e.g., UC TAG matrix, college transfer center, or ASSIST). If none are cited, leave empty.
    - adt_urls: List of URLs cited to list ADT programs (e.g., CSU ADT resources, college catalog pages). If none are cited, leave empty.
    - ge_urls: List of URLs cited to confirm IGETC and/or CSU GE-Breadth certification availability (e.g., ASSIST, college catalog, transfer center pages). If none are cited, leave empty.

    IMPORTANT:
    - Extract ONLY what is explicitly present in the answer. Do not infer or invent information.
    - For all URL fields, extract actual URLs (including protocol). Accept plain URLs or markdown links; normalize them to plain URLs.
    - If any field is missing in the answer for a college, return null (for single fields) or an empty list (for array fields).
    - Return a JSON object with a 'colleges' array; each element is one college with the fields above.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_list_str(items: List[str]) -> List[str]:
    return [s.strip() for s in items if isinstance(s, str) and s.strip()]


def _pick_first_allowed_uc(tag_campuses: List[str]) -> Optional[str]:
    # Try to map to allowed campuses via case-insensitive match
    normalized = [c.strip() for c in tag_campuses if isinstance(c, str)]
    for campus in normalized:
        # Basic normalization for matching
        low = campus.lower()
        if "davis" in low and "uc" in low:
            return "UC Davis"
        if ("irvine" in low or "uci" in low) and "uc" in low:
            return "UC Irvine"
        if "merced" in low and "uc" in low:
            return "UC Merced"
        if ("riverside" in low or "ucr" in low) and "uc" in low:
            return "UC Riverside"
        if ("santa barbara" in low or "ucsb" in low) and "uc" in low:
            return "UC Santa Barbara"
        if ("santa cruz" in low or "ucsc" in low) and "uc" in low:
            return "UC Santa Cruz"
        # Exact match fallback
        if campus in ALLOWED_TAG_CAMPUSES:
            return campus
    return None


def _first_non_empty(items: List[str]) -> Optional[str]:
    for s in items:
        if s and s.strip():
            return s.strip()
    return None


def _ensure_four_items(colleges: List[CollegeItem]) -> List[CollegeItem]:
    # Select first four; pad with empty placeholders if fewer
    selected = colleges[:4]
    while len(selected) < 4:
        selected.append(CollegeItem())
    return selected


# --------------------------------------------------------------------------- #
# Verification for one college                                                #
# --------------------------------------------------------------------------- #
async def verify_one_college(
    evaluator: Evaluator,
    parent_node,
    college: CollegeItem,
    index_one_based: int
) -> None:
    """
    Build the verification sub-tree and run evidence-grounded checks for one college.
    """
    ordinal_map = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}
    ordinal = ordinal_map.get(index_one_based, f"College #{index_one_based}")

    # College node (critical under root to satisfy critical children constraint)
    college_node = evaluator.add_parallel(
        id=f"College_{index_one_based}",
        desc=f"{ordinal} identified California community college meeting all requirements",
        parent=parent_node,
        critical=True
    )

    # ------- Basic Info ------- #
    basic_info = evaluator.add_parallel(
        id=f"College_{index_one_based}_Basic_Info",
        desc=f"College {index_one_based} is identified with its official name and is confirmed as part of the California Community Colleges system",
        parent=college_node,
        critical=True
    )

    # Info URL existence (as gating prerequisite for basic info verifications)
    info_urls = _normalize_list_str(college.info_urls)
    info_url_exists = evaluator.add_custom_node(
        result=len(info_urls) > 0,
        id=f"College_{index_one_based}_Info_URL",
        desc=f"A valid official URL (college website or CCCCO listing) is provided as reference for College {index_one_based}'s basic information",
        parent=basic_info,
        critical=True
    )

    # CCC System membership
    ccc_leaf = evaluator.add_leaf(
        id=f"College_{index_one_based}_CCC_System",
        desc=f"College {index_one_based} is one of the 116 colleges in the California Community Colleges system",
        parent=basic_info,
        critical=True
    )
    college_name = college.name or f"College {index_one_based}"
    ccc_claim = f"'{college_name}' is a member of the California Community Colleges system (one of the 116 colleges)."
    await evaluator.verify(
        claim=ccc_claim,
        node=ccc_leaf,
        sources=info_urls,
        additional_instruction="Use the provided official college website or the California Community Colleges Chancellor's Office listing to confirm membership.",
        extra_prerequisites=[info_url_exists]
    )

    # District affiliation
    district_leaf = evaluator.add_leaf(
        id=f"College_{index_one_based}_District",
        desc=f"College {index_one_based}'s district affiliation is identified (must be one of the 73 CCC districts)",
        parent=basic_info,
        critical=True
    )
    district_name = (college.district or "").strip()
    district_claim = f"'{college_name}' belongs to the '{district_name}' community college district."
    await evaluator.verify(
        claim=district_claim,
        node=district_leaf,
        sources=info_urls,
        additional_instruction="Confirm that the named district is the governing community college district for the college. Allow minor naming variations (e.g., inclusion of 'Community College District').",
        extra_prerequisites=[info_url_exists]
    )

    # ------- Accreditation ------- #
    accreditation_node = evaluator.add_parallel(
        id=f"College_{index_one_based}_Accreditation",
        desc=f"College {index_one_based}'s accreditation status is verified",
        parent=college_node,
        critical=True
    )

    accreditation_urls = _normalize_list_str(college.accreditation_urls)
    accreditation_url_exists = evaluator.add_custom_node(
        result=len(accreditation_urls) > 0,
        id=f"College_{index_one_based}_Accreditation_URL",
        desc=f"A valid URL from ACCJC.org or the college's accreditation page confirming current accreditation status is provided",
        parent=accreditation_node,
        critical=True
    )

    accjc_leaf = evaluator.add_leaf(
        id=f"College_{index_one_based}_ACCJC_Status",
        desc=f"College {index_one_based} is accredited by ACCJC (Accrediting Commission for Community and Junior Colleges)",
        parent=accreditation_node,
        critical=True
    )
    accjc_claim = f"'{college_name}' is currently accredited by ACCJC (the Accrediting Commission for Community and Junior Colleges)."
    await evaluator.verify(
        claim=accjc_claim,
        node=accjc_leaf,
        sources=accreditation_urls,
        additional_instruction="Prefer evidence on accjc.org directory pages; a college accreditation page is acceptable if it clearly states current ACCJC accreditation.",
        extra_prerequisites=[accreditation_url_exists]
    )

    # ------- Transfer Programs ------- #
    transfer_node = evaluator.add_parallel(
        id=f"College_{index_one_based}_Transfer_Programs",
        desc=f"College {index_one_based} offers transfer pathways to UC and CSU systems",
        parent=college_node,
        critical=True
    )

    # UC TAG cluster
    uc_tag_node = evaluator.add_parallel(
        id=f"College_{index_one_based}_UC_TAG",
        desc=f"College {index_one_based} participates in UC TAG (Transfer Admission Guarantee) program with at least one eligible UC campus",
        parent=transfer_node,
        critical=True
    )

    tag_urls = _normalize_list_str(college.tag_urls)
    tag_url_exists = evaluator.add_custom_node(
        result=len(tag_urls) > 0,
        id=f"College_{index_one_based}_TAG_URL",
        desc=f"A valid URL confirming College {index_one_based}'s participation in UC TAG is provided",
        parent=uc_tag_node,
        critical=True
    )

    tag_campus_leaf = evaluator.add_leaf(
        id=f"College_{index_one_based}_TAG_Campus",
        desc=f"At least one UC TAG-eligible campus is identified (must be from: UC Davis, UC Irvine, UC Merced, UC Riverside, UC Santa Barbara, or UC Santa Cruz)",
        parent=uc_tag_node,
        critical=True
    )
    selected_tag_campus = _pick_first_allowed_uc(college.tag_campuses)
    if selected_tag_campus:
        tag_campus_claim = f"'{college_name}' participates in UC TAG with {selected_tag_campus}."
        tag_add_ins = ("Confirm that the college participates in UC TAG with the named campus. "
                       "Only the following campuses are eligible: UC Davis, UC Irvine, UC Merced, UC Riverside, UC Santa Barbara, UC Santa Cruz.")
    else:
        # No eligible campus explicitly identified in the answer; per requirements, this should fail.
        tag_campus_claim = ("No eligible TAG campus was explicitly identified in the answer for this college. "
                            "Per the task requirement, at least one of the following must be named: "
                            "UC Davis, UC Irvine, UC Merced, UC Riverside, UC Santa Barbara, UC Santa Cruz.")
        tag_add_ins = ("Since the answer did not specify an eligible campus, mark this verification as Incorrect, "
                       "even if the provided page shows UC TAG participation.")
    await evaluator.verify(
        claim=tag_campus_claim,
        node=tag_campus_leaf,
        sources=tag_urls,
        additional_instruction=tag_add_ins,
        extra_prerequisites=[tag_url_exists]
    )

    # ADT cluster
    adt_node = evaluator.add_parallel(
        id=f"College_{index_one_based}_ADT",
        desc=f"College {index_one_based} offers Associate Degree for Transfer (ADT) programs",
        parent=transfer_node,
        critical=True
    )

    adt_urls = _normalize_list_str(college.adt_urls)
    adt_url_exists = evaluator.add_custom_node(
        result=len(adt_urls) > 0,
        id=f"College_{index_one_based}_ADT_URL",
        desc=f"A valid URL listing College {index_one_based}'s ADT programs is provided (from college catalog, CSU website, or ASSIST.org)",
        parent=adt_node,
        critical=True
    )

    adt_available_leaf = evaluator.add_leaf(
        id=f"College_{index_one_based}_ADT_Available",
        desc=f"At least one ADT program (AA-T or AS-T) is offered by College {index_one_based}",
        parent=adt_node,
        critical=True
    )
    example_adt = _first_non_empty(college.adt_programs) or ""
    adt_claim = f"'{college_name}' offers at least one ADT program (AA-T or AS-T), for example '{example_adt}'."
    adt_add_ins = ("Verify that the page lists ADT programs for the college (AA-T or AS-T). "
                   "If the answer did not name any specific ADT program, mark Incorrect.")
    await evaluator.verify(
        claim=adt_claim,
        node=adt_available_leaf,
        sources=adt_urls,
        additional_instruction=adt_add_ins,
        extra_prerequisites=[adt_url_exists]
    )

    # ------- General Education ------- #
    ge_node = evaluator.add_parallel(
        id=f"College_{index_one_based}_General_Education",
        desc=f"College {index_one_based} offers courses that fulfill IGETC and CSU GE-Breadth requirements",
        parent=college_node,
        critical=True
    )

    ge_urls = _normalize_list_str(college.ge_urls)
    ge_url_exists = evaluator.add_custom_node(
        result=len(ge_urls) > 0,
        id=f"College_{index_one_based}_GE_URL",
        desc=f"A valid URL confirming College {index_one_based}'s IGETC and/or CSU GE-Breadth certification availability is provided",
        parent=ge_node,
        critical=True
    )

    igetc_leaf = evaluator.add_leaf(
        id=f"College_{index_one_based}_IGETC_Certification",
        desc=f"College {index_one_based} offers IGETC (Intersegmental General Education Transfer Curriculum) pattern certification",
        parent=ge_node,
        critical=True
    )
    igetc_claim = f"'{college_name}' offers IGETC pattern certification for transfer students."
    await evaluator.verify(
        claim=igetc_claim,
        node=igetc_leaf,
        sources=ge_urls,
        additional_instruction="Confirm that the page indicates IGETC certification or IGETC pattern certification is available at the college.",
        extra_prerequisites=[ge_url_exists]
    )

    csu_ge_leaf = evaluator.add_leaf(
        id=f"College_{index_one_based}_CSU_GE",
        desc=f"College {index_one_based} offers CSU GE-Breadth pattern certification",
        parent=ge_node,
        critical=True
    )
    csu_ge_claim = f"'{college_name}' offers CSU GE-Breadth pattern certification for transfer students."
    await evaluator.verify(
        claim=csu_ge_claim,
        node=csu_ge_leaf,
        sources=ge_urls,
        additional_instruction="Confirm that the page indicates CSU GE-Breadth certification is available at the college.",
        extra_prerequisites=[ge_url_exists]
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
    Evaluate an answer for the California Community Colleges transfer planning task.
    Builds a verification tree per the rubric and returns a structured summary.
    """
    # Initialize evaluator with a CRITICAL root (parallel aggregation)
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

    # The root node is critical per rubric; to satisfy framework constraint,
    # all direct children must be critical as well (we will set them later).

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_transfer_colleges(),
        template_class=TransferCollegesExtraction,
        extraction_name="transfer_colleges_extraction"
    )

    # Record allowed campuses as custom info
    evaluator.add_custom_info(
        info={"allowed_tag_campuses": sorted(list(ALLOWED_TAG_CAMPUSES))},
        info_type="allowed_uc_tag_campuses",
        info_name="allowed_uc_tag_campuses"
    )

    # Select the first four colleges (pad if fewer)
    selected_colleges = _ensure_four_items(extraction.colleges)

    # Add top-level college nodes under root (critical children to satisfy constraint)
    top_nodes = []
    ordinal_desc = {
        1: "First identified California community college meeting all requirements",
        2: "Second identified California community college meeting all requirements",
        3: "Third identified California community college meeting all requirements",
        4: "Fourth identified California community college meeting all requirements"
    }
    for i in range(4):
        node = evaluator.add_parallel(
            id=f"College_{i+1}_Container",
            desc=ordinal_desc[i+1],
            parent=root,
            critical=True
        )
        top_nodes.append(node)

    # Verify each selected college
    for idx, college in enumerate(selected_colleges, start=1):
        await verify_one_college(evaluator, top_nodes[idx - 1], college, idx)

    # Return summary
    return evaluator.get_summary()