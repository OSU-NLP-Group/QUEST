import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "univ_criteria_all"
TASK_DESCRIPTION = (
    "Identify the public research university in the United States that satisfies ALL of the following criteria: "
    "(1) It must be a public (state-funded) university, not a private institution; "
    "(2) It must be currently classified as an R1 research university in the Carnegie Classification system "
    "(Doctoral Universities – Very high research activity); "
    "(3) It must be a current member institution of the Big Ten Conference (as of 2024); "
    "(4) It must hold land-grant university status under the Morrill Act of 1862 or 1890; "
    "(5) It must have been founded or established during the 19th century (between 1800 and 1899); "
    "(6) It must be located in a U.S. state that has multiple R1 research universities; "
    "(7) It must have a total student enrollment (undergraduate and graduate combined) exceeding 40,000 students; "
    "(8) It must have an on-campus football stadium with a seating capacity of at least 80,000; "
    "(9) It must have annual research expenditures exceeding $500 million according to NSF Higher Education Research "
    "and Development (HERD) survey data; "
    "(10) Its library system must be a member of the Association of Research Libraries (ARL); "
    "(11) Its main campus must exceed 1,000 acres in size; "
    "(12) It must be recognized as the flagship university of its state. Provide the name of the university and "
    "include supporting reference URLs for each criterion."
)


# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class UniversityExtraction(BaseModel):
    """Structured extraction of the university and the criterion-specific sources from the agent's answer."""
    university_name: Optional[str] = None
    state: Optional[str] = None
    founding_year: Optional[str] = None
    enrollment_total: Optional[str] = None
    stadium_name: Optional[str] = None
    stadium_capacity: Optional[str] = None
    research_expenditures: Optional[str] = None
    campus_acres: Optional[str] = None

    public_status_sources: List[str] = Field(default_factory=list)
    r1_sources: List[str] = Field(default_factory=list)
    big_ten_sources: List[str] = Field(default_factory=list)
    land_grant_sources: List[str] = Field(default_factory=list)
    founding_sources: List[str] = Field(default_factory=list)
    multi_r1_state_sources: List[str] = Field(default_factory=list)
    enrollment_sources: List[str] = Field(default_factory=list)
    stadium_sources: List[str] = Field(default_factory=list)
    research_spending_sources: List[str] = Field(default_factory=list)
    arl_sources: List[str] = Field(default_factory=list)
    campus_size_sources: List[str] = Field(default_factory=list)
    flagship_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_university_criteria() -> str:
    return (
        "From the answer, extract the one identified university and the criterion-specific supporting URLs.\n"
        "Return a JSON object with the following fields:\n"
        "1) university_name: The full name of the chosen university.\n"
        "2) state: The U.S. state where the university's main campus is located.\n"
        "3) founding_year: The year the university was founded (string). If not provided, null.\n"
        "4) enrollment_total: The total student enrollment number mentioned (string, can be approximate). If not provided, null.\n"
        "5) stadium_name: The name of the on-campus football stadium (string). If not provided, null.\n"
        "6) stadium_capacity: The seating capacity figure mentioned (string). If not provided, null.\n"
        "7) research_expenditures: The annual research expenditure value mentioned (string). If not provided, null.\n"
        "8) campus_acres: The main campus acreage mentioned (string). If not provided, null.\n"
        "\n"
        "For EACH criterion below, extract ONLY the URLs explicitly present in the answer as support for that criterion "
        "(plain URLs or markdown links). If the answer does not include any URL for a criterion, return an empty list.\n"
        "9) public_status_sources: URLs confirming it is a public (state-funded) university.\n"
        "10) r1_sources: URLs confirming Carnegie R1 classification (Doctoral Universities – Very high research activity).\n"
        "11) big_ten_sources: URLs confirming current Big Ten Conference membership (as of 2024).\n"
        "12) land_grant_sources: URLs confirming land-grant status under the Morrill Act of 1862 or 1890.\n"
        "13) founding_sources: URLs confirming founding/establishment date.\n"
        "14) multi_r1_state_sources: URLs confirming the state has multiple R1 universities.\n"
        "15) enrollment_sources: URLs confirming total student enrollment exceeds 40,000.\n"
        "16) stadium_sources: URLs confirming on-campus football stadium capacity is ≥ 80,000.\n"
        "17) research_spending_sources: URLs confirming annual research expenditures exceed $500M per NSF HERD.\n"
        "18) arl_sources: URLs confirming the library system is a member of ARL.\n"
        "19) campus_size_sources: URLs confirming main campus exceeds 1,000 acres.\n"
        "20) flagship_sources: URLs confirming recognition as the state's flagship university.\n"
        "\n"
        "Rules:\n"
        "- Extract only URLs explicitly present in the answer; do not invent any URLs.\n"
        "- If a field is missing in the answer, set it to null (strings) or [] (sources lists).\n"
        "- Include full URLs; for markdown links, return the target URL.\n"
        "- Do not combine or deduplicate across criteria; place each URL only under its relevant criterion.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_univ_name(ext: UniversityExtraction) -> str:
    return ext.university_name or "the identified university"


def _safe_state(ext: UniversityExtraction) -> str:
    return ext.state or "its state"


def _require_url_support_instruction(extra: str) -> str:
    base = (
        "Your judgment must be based on explicit support from at least one of the provided URLs. "
        "If no URLs are provided, or if the URLs are irrelevant or inaccessible, judge the claim as NOT SUPPORTED. "
        "Do not rely on your own knowledge. Allow minor wording variations (e.g., synonyms or approximate figures). "
    )
    return base + extra


async def _add_check_and_doc(
    evaluator: Evaluator,
    parent: Any,
    criterion_node_id: str,
    criterion_node_desc: str,
    verification_node_id: str,
    verification_node_desc: str,
    check_node_id: str,
    check_node_desc: str,
    check_leaf_id: str,
    check_leaf_desc: str,
    doc_node_id: str,
    doc_node_desc: str,
    doc_leaf_id: str,
    doc_leaf_desc: str,
    claim_text: str,
    sources_list: List[str],
    add_ins: str,
) -> None:
    """
    Generic builder for a sequential criterion subtree:
    Criterion -> Verification -> (Check -> leaf) then (Documentation -> existence leaf)
    All nodes are critical to match rubric and enforce ALL-criteria requirement.
    """
    # Criterion node (sequential, critical)
    crit_node = evaluator.add_sequential(
        id=criterion_node_id,
        desc=criterion_node_desc,
        parent=parent,
        critical=True,
    )

    # Verification group (sequential, critical)
    ver_node = evaluator.add_sequential(
        id=verification_node_id,
        desc=verification_node_desc,
        parent=crit_node,
        critical=True,
    )

    # Status/Classification/Membership/etc check container (sequential, critical)
    check_node = evaluator.add_sequential(
        id=check_node_id,
        desc=check_node_desc,
        parent=ver_node,
        critical=True,
    )

    # Actual factual verification leaf (critical); prefer URL-grounded verification via evaluator.verify
    check_leaf = evaluator.add_leaf(
        id=check_leaf_id,
        desc=check_leaf_desc,
        parent=check_node,
        critical=True,
    )
    await evaluator.verify(
        claim=claim_text,
        node=check_leaf,
        sources=sources_list,  # Routed automatically: NONE -> simple_verify; 1 url -> single; multi -> multi_url
        additional_instruction=add_ins,
    )

    # Documentation container (sequential, critical)
    doc_node = evaluator.add_sequential(
        id=doc_node_id,
        desc=doc_node_desc,
        parent=ver_node,
        critical=True,
    )

    # Documentation existence leaf (critical): ensure answer provided URL(s) for this criterion
    doc_leaf = evaluator.add_custom_node(
        result=bool(sources_list),
        id=doc_leaf_id,
        desc=doc_leaf_desc,
        parent=doc_node,
        critical=True,
    )
    # No verify call needed: this is a pure existence check


# --------------------------------------------------------------------------- #
# Verification builder functions per criterion                                #
# --------------------------------------------------------------------------- #
async def build_institutional_type(evaluator: Evaluator, parent: Any, ext: UniversityExtraction) -> None:
    univ = _safe_univ_name(ext)
    claim = f"{univ} is a public (state-funded) university."
    add_ins = _require_url_support_instruction(
        "Confirm the institution is PUBLIC (state/government funded), not private. "
        "Accept authoritative sources such as the university's official site, state higher education pages, or Wikipedia infobox if it clearly states 'Public'."
    )
    await _add_check_and_doc(
        evaluator,
        parent,
        "Institutional_Type",
        "Verify the institution is a public university (state-funded)",
        "Type_Verification",
        "Verify and document public university status",
        "Status_Check",
        "Check institutional type classification",
        "Public_University_Confirmed",
        "Confirmed as a public state university, not private",
        "Documentation",
        "Provide supporting documentation",
        "Public_Status_Reference",
        "Provide URL reference confirming public university status",
        ext.public_status_sources,
        claim,
        add_ins,
    )


async def build_research_classification(evaluator: Evaluator, parent: Any, ext: UniversityExtraction) -> None:
    univ = _safe_univ_name(ext)
    claim = f"{univ} is currently classified as R1 (Doctoral Universities – Very high research activity)."
    add_ins = _require_url_support_instruction(
        "Prefer the official Carnegie Classification website or authoritative references that explicitly list the institution as R1."
    )
    await _add_check_and_doc(
        evaluator,
        parent,
        "Research_Classification",
        "Verify Carnegie Classification as R1 research university",
        "Classification_Verification",
        "Verify and document R1 classification status",
        "Classification_Check",
        "Check Carnegie Classification designation",
        "R1_Status_Confirmed",
        "Currently classified as R1: Doctoral Universities – Very high research activity",
        "Documentation",
        "Provide supporting documentation",
        "R1_Reference",
        "Provide URL reference confirming R1 classification",
        ext.r1_sources,
        claim,
        add_ins,
    )


async def build_athletic_membership(evaluator: Evaluator, parent: Any, ext: UniversityExtraction) -> None:
    univ = _safe_univ_name(ext)
    claim = f"{univ} is a current member institution of the Big Ten Conference (as of 2024)."
    add_ins = _require_url_support_instruction(
        "Use the official Big Ten website or credible sources that list member institutions as of 2024."
    )
    await _add_check_and_doc(
        evaluator,
        parent,
        "Athletic_Conference_Membership",
        "Verify current Big Ten Conference membership",
        "Membership_Verification",
        "Verify and document Big Ten Conference membership",
        "Membership_Check",
        "Check conference affiliation status",
        "Big_Ten_Member_Confirmed",
        "Currently a member institution of the Big Ten Conference",
        "Documentation",
        "Provide supporting documentation",
        "Conference_Reference",
        "Provide URL reference confirming Big Ten membership",
        ext.big_ten_sources,
        claim,
        add_ins,
    )


async def build_land_grant(evaluator: Evaluator, parent: Any, ext: UniversityExtraction) -> None:
    univ = _safe_univ_name(ext)
    claim = f"{univ} is designated as a land-grant university under the Morrill Act of 1862 or 1890."
    add_ins = _require_url_support_instruction(
        "Use credible sources (e.g., university history pages, land-grant lists) explicitly stating land-grant designation."
    )
    await _add_check_and_doc(
        evaluator,
        parent,
        "Land_Grant_Status",
        "Verify land-grant university designation under Morrill Act",
        "Land_Grant_Verification",
        "Verify and document land-grant designation",
        "Designation_Check",
        "Check Morrill Act land-grant status",
        "Morrill_Act_Designation_Confirmed",
        "Designated as a land-grant university under the Morrill Act of 1862 or 1890",
        "Documentation",
        "Provide supporting documentation",
        "Land_Grant_Reference",
        "Provide URL reference confirming land-grant status",
        ext.land_grant_sources,
        claim,
        add_ins,
    )


async def build_founding_period(evaluator: Evaluator, parent: Any, ext: UniversityExtraction) -> None:
    univ = _safe_univ_name(ext)
    if ext.founding_year:
        claim = f"{univ} was founded in {ext.founding_year}, which is between 1800 and 1899."
    else:
        claim = f"{univ} was founded or established between 1800 and 1899."
    add_ins = _require_url_support_instruction(
        "Confirm the founding/establishment date falls in the 19th century (1800–1899). Use authoritative sources."
    )
    await _add_check_and_doc(
        evaluator,
        parent,
        "Founding_Period",
        "Verify founding/establishment date in the 19th century",
        "Founding_Verification",
        "Verify and document founding date",
        "Date_Check",
        "Check founding/establishment date",
        "Nineteenth_Century_Founding_Confirmed",
        "Founded or established between 1800 and 1899",
        "Documentation",
        "Provide supporting documentation",
        "Founding_Date_Reference",
        "Provide URL reference confirming founding date",
        ext.founding_sources,
        claim,
        add_ins,
    )


async def build_geographic_context(evaluator: Evaluator, parent: Any, ext: UniversityExtraction) -> None:
    state = _safe_state(ext)
    claim = f"The state of {state} has multiple R1 research universities."
    add_ins = _require_url_support_instruction(
        "Use Carnegie Classification or credible sources that list multiple R1 institutions within the specified state."
    )
    await _add_check_and_doc(
        evaluator,
        parent,
        "Geographic_Context",
        "Verify location in a state with multiple R1 universities",
        "Location_Verification",
        "Verify and document state location and R1 context",
        "State_Context_Check",
        "Check state's R1 university count",
        "Multiple_R1_State_Confirmed",
        "Located in a U.S. state that has multiple R1 research universities",
        "Documentation",
        "Provide supporting documentation",
        "State_R1_Count_Reference",
        "Provide URL reference confirming the state has multiple R1 institutions",
        ext.multi_r1_state_sources,
        claim,
        add_ins,
    )


async def build_enrollment_scale(evaluator: Evaluator, parent: Any, ext: UniversityExtraction) -> None:
    univ = _safe_univ_name(ext)
    claim = f"{univ} has total student enrollment (undergraduate + graduate) exceeding 40,000."
    add_ins = _require_url_support_instruction(
        "Confirm TOTAL enrollment exceeds 40,000. Use official institutional statistics pages or credible sources. "
        "If only approximate terms like 'over 40,000' are shown, that is acceptable."
    )
    await _add_check_and_doc(
        evaluator,
        parent,
        "Enrollment_Scale",
        "Verify total student enrollment meets minimum threshold",
        "Enrollment_Verification",
        "Verify and document enrollment numbers",
        "Enrollment_Check",
        "Check total enrollment figures",
        "Enrollment_Over_40000_Confirmed",
        "Total student enrollment (undergraduate and graduate combined) exceeds 40,000",
        "Documentation",
        "Provide supporting documentation",
        "Enrollment_Reference",
        "Provide URL reference with current enrollment data",
        ext.enrollment_sources,
        claim,
        add_ins,
    )


async def build_stadium_capacity(evaluator: Evaluator, parent: Any, ext: UniversityExtraction) -> None:
    univ = _safe_univ_name(ext)
    stadium_part = f" ({ext.stadium_name})" if ext.stadium_name else ""
    claim = f"{univ}'s on-campus football stadium{stadium_part} has seating capacity of at least 80,000."
    add_ins = _require_url_support_instruction(
        "Confirm the facility is the university's on-campus football stadium and its seating capacity is ≥ 80,000. "
        "Use authoritative stadium pages or credible sources."
    )
    await _add_check_and_doc(
        evaluator,
        parent,
        "Football_Stadium_Capacity",
        "Verify on-campus football stadium capacity meets minimum threshold",
        "Stadium_Verification",
        "Verify and document stadium capacity",
        "Capacity_Check",
        "Check football stadium seating capacity",
        "Stadium_Capacity_80000_Confirmed",
        "On-campus football stadium has seating capacity of at least 80,000",
        "Documentation",
        "Provide supporting documentation",
        "Stadium_Reference",
        "Provide URL reference with stadium capacity information",
        ext.stadium_sources,
        claim,
        add_ins,
    )


async def build_research_expenditures(evaluator: Evaluator, parent: Any, ext: UniversityExtraction) -> None:
    univ = _safe_univ_name(ext)
    claim = f"{univ} has annual research expenditures exceeding $500 million according to NSF HERD survey data."
    add_ins = _require_url_support_instruction(
        "Prefer the official NSF HERD survey data pages or credible summaries that explicitly state research expenditures > $500M."
    )
    await _add_check_and_doc(
        evaluator,
        parent,
        "Research_Expenditures",
        "Verify annual research spending meets minimum threshold",
        "Expenditure_Verification",
        "Verify and document research expenditure levels",
        "Spending_Check",
        "Check annual research expenditure amounts",
        "Expenditures_Over_500M_Confirmed",
        "Annual research expenditures exceed $500 million according to NSF HERD data",
        "Documentation",
        "Provide supporting documentation",
        "Research_Spending_Reference",
        "Provide URL reference with research expenditure data",
        ext.research_spending_sources,
        claim,
        add_ins,
    )


async def build_arl_membership(evaluator: Evaluator, parent: Any, ext: UniversityExtraction) -> None:
    univ = _safe_univ_name(ext)
    claim = f"The library system of {univ} is a member of the Association of Research Libraries (ARL)."
    add_ins = _require_url_support_instruction(
        "Use the ARL official membership list or credible institutional sources that explicitly state ARL membership."
    )
    await _add_check_and_doc(
        evaluator,
        parent,
        "Library_System_Membership",
        "Verify membership in Association of Research Libraries",
        "ARL_Verification",
        "Verify and document ARL membership",
        "Membership_Check",
        "Check ARL membership status",
        "ARL_Member_Confirmed",
        "The university's library system is a member of the Association of Research Libraries (ARL)",
        "Documentation",
        "Provide supporting documentation",
        "ARL_Reference",
        "Provide URL reference confirming ARL membership",
        ext.arl_sources,
        claim,
        add_ins,
    )


async def build_campus_size(evaluator: Evaluator, parent: Any, ext: UniversityExtraction) -> None:
    univ = _safe_univ_name(ext)
    claim = f"The main campus of {univ} exceeds 1,000 acres in size."
    add_ins = _require_url_support_instruction(
        "Use authoritative institutional sources or credible references that state main campus acreage > 1,000 acres."
    )
    await _add_check_and_doc(
        evaluator,
        parent,
        "Campus_Physical_Size",
        "Verify main campus acreage meets minimum threshold",
        "Acreage_Verification",
        "Verify and document campus size",
        "Size_Check",
        "Check main campus acreage",
        "Campus_Over_1000_Acres_Confirmed",
        "Main campus size exceeds 1,000 acres",
        "Documentation",
        "Provide supporting documentation",
        "Campus_Size_Reference",
        "Provide URL reference with campus acreage information",
        ext.campus_size_sources,
        claim,
        add_ins,
    )


async def build_flagship_status(evaluator: Evaluator, parent: Any, ext: UniversityExtraction) -> None:
    univ = _safe_univ_name(ext)
    state = _safe_state(ext)
    claim = f"{univ} is recognized as the flagship university of the state of {state}."
    add_ins = _require_url_support_instruction(
        "Use credible sources (e.g., state higher education boards, university/institutional references, or authoritative publications) "
        "that explicitly recognize the institution as the state's flagship university."
    )
    await _add_check_and_doc(
        evaluator,
        parent,
        "State_Flagship_Designation",
        "Verify recognition as the state's flagship university",
        "Flagship_Verification",
        "Verify and document flagship status",
        "Status_Check",
        "Check flagship university designation",
        "Flagship_Status_Confirmed",
        "Recognized as the flagship university of its state",
        "Documentation",
        "Provide supporting documentation",
        "Flagship_Reference",
        "Provide URL reference confirming flagship status",
        ext.flagship_sources,
        claim,
        add_ins,
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
    Evaluate the agent's answer for the university identification task with all specified criteria.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # root container; we will add a critical child node per rubric
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

    # Extract structured info and URLs from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_university_criteria(),
        template_class=UniversityExtraction,
        extraction_name="university_extraction",
    )

    # Top-level rubric node (critical, parallel aggregation over all criteria)
    univ_root = evaluator.add_parallel(
        id="University_Identification",
        desc="Identify a university that satisfies all specified institutional characteristics across research, athletics, history, and operational dimensions",
        parent=root,
        critical=True,
    )

    # Build and verify each criterion subtree
    await build_institutional_type(evaluator, univ_root, extraction)
    await build_research_classification(evaluator, univ_root, extraction)
    await build_athletic_membership(evaluator, univ_root, extraction)
    await build_land_grant(evaluator, univ_root, extraction)
    await build_founding_period(evaluator, univ_root, extraction)
    await build_geographic_context(evaluator, univ_root, extraction)
    await build_enrollment_scale(evaluator, univ_root, extraction)
    await build_stadium_capacity(evaluator, univ_root, extraction)
    await build_research_expenditures(evaluator, univ_root, extraction)
    await build_arl_membership(evaluator, univ_root, extraction)
    await build_campus_size(evaluator, univ_root, extraction)
    await build_flagship_status(evaluator, univ_root, extraction)

    # Optional: record thresholds used for transparency
    evaluator.add_custom_info(
        info={
            "enrollment_threshold": "> 40,000 total students",
            "stadium_capacity_threshold": ">= 80,000 seats",
            "research_expenditures_threshold": "> $500 million (NSF HERD)",
            "campus_size_threshold": "> 1,000 acres",
            "century_requirement": "19th century (1800–1899)",
            "classification_requirement": "Carnegie R1",
            "conference_requirement": "Big Ten (as of 2024)",
            "land_grant_requirement": "Morrill Act (1862 or 1890)",
            "library_requirement": "ARL membership",
            "flagship_requirement": "Recognized as state's flagship",
            "state_context_requirement": "Located in a state with multiple R1 universities",
        },
        info_type="thresholds",
        info_name="evaluation_thresholds",
    )

    # Return evaluation summary
    return evaluator.get_summary()