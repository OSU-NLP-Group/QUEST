import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_presidents_chancellors_2025_2026"
TASK_DESCRIPTION = """
Identify at least four recent appointments of presidents or chancellors at U.S. research universities that were officially announced between December 2025 and February 2026 (inclusive). For each appointment, provide the following information: (1) Appointee's Full Name: The complete name of the person appointed; (2) University/Institution Name: The full name of the university or institution; (3) Official Announcement Date: The date when the appointment was officially announced (must fall between December 2025 and February 2026); (4) Expected Start Date: The date when the appointee is expected to begin or began their role as president or chancellor; (5) Previous Position: The appointee's immediately previous position and the institution where they held that position; (6) Reference URL: A link to an official university announcement, press release, or credible news article confirming the appointment. Each appointment must be for a different individual at a different institution. The positions must be for the top executive role (President or Chancellor) at accredited four-year research universities in the United States.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AppointmentItem(BaseModel):
    full_name: Optional[str] = None
    institution: Optional[str] = None
    role_title: Optional[str] = None
    announcement_date: Optional[str] = None
    start_date: Optional[str] = None
    previous_position_title: Optional[str] = None
    previous_institution: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class AppointmentsExtraction(BaseModel):
    appointments: List[AppointmentItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_appointments() -> str:
    return """
    Extract up to five (5) appointment items for U.S. university Presidents or Chancellors mentioned in the answer.
    For each appointment, return the following fields:
    - full_name: The complete appointee name as provided in the answer.
    - institution: The full name of the university or institution as provided in the answer.
    - role_title: The role title string as provided (e.g., "President", "Chancellor", "president-designate", "president-elect", "interim president", etc.).
    - announcement_date: The official announcement date string as provided in the answer (e.g., "December 15, 2025"). If multiple dates are mentioned, choose the one that clearly corresponds to the appointment announcement.
    - start_date: The expected or actual start date string as provided (e.g., "July 1, 2026", "effective July 2026", etc.).
    - previous_position_title: The immediately previous position title (e.g., "Provost", "Dean of X", "President of Y").
    - previous_institution: The institution where that previous position was held.
    - reference_urls: An array of URLs explicitly present in the answer that confirm the appointment (official university press release preferred; credible news articles acceptable). Include all relevant URLs up to 3 per appointment.
    
    Rules:
    1) Extract ONLY what is explicitly stated in the answer text. Do not invent or infer missing information.
    2) If any field is missing for an item, set it to null (or an empty array for reference_urls).
    3) Only extract valid, well-formed URLs for reference_urls.
    4) Preserve the original text format for dates (strings).
    5) Return the items in the same order they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def norm_key(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def appointment_node_id(index: int) -> Tuple[str, str]:
    """Return node id and description label based on index (1-based for first 4, 5th = optional)."""
    if index <= 4:
        return f"appointment_{index}", f"Appointment item {index} (candidate valid appointment)"
    else:
        return "appointment_5_optional", "Appointment item 5 (optional extra; may be used to reach ≥4 valid appointments if one of the first four is invalid)"


# --------------------------------------------------------------------------- #
# Verification for each appointment                                           #
# --------------------------------------------------------------------------- #
async def verify_appointment(
    evaluator: Evaluator,
    parent: VerificationNode,
    app: AppointmentItem,
    index: int,
) -> VerificationNode:
    """
    Build verification subtree for a single appointment and perform verifications.
    index: 1-based index for the first 4 required, 5th is optional.
    """
    appt_id, appt_desc = appointment_node_id(index)

    # Parent node for this appointment: parallel, non-critical (per rubric)
    appt_node = evaluator.add_parallel(
        id=appt_id,
        desc=appt_desc,
        parent=parent,
        critical=False
    )

    # 1) Presence checks (critical)
    # a{i}_full_name
    evaluator.add_custom_node(
        result=bool(app.full_name and app.full_name.strip()),
        id=f"a{index}_full_name",
        desc="Provides the appointee's full name",
        parent=appt_node,
        critical=True
    )

    # a{i}_institution_name
    evaluator.add_custom_node(
        result=bool(app.institution and app.institution.strip()),
        id=f"a{index}_institution_name",
        desc="Provides the university/institution name",
        parent=appt_node,
        critical=True
    )

    # a{i}_reference_url (presence only)
    evaluator.add_custom_node(
        result=bool(app.reference_urls and len(app.reference_urls) > 0),
        id=f"a{index}_reference_url",
        desc="Provides a reference URL to an official university announcement/press release or a credible news source confirming the appointment",
        parent=appt_node,
        critical=True
    )

    # 2) Role is top executive (President/Chancellor) – verify with sources
    role_leaf = evaluator.add_leaf(
        id=f"a{index}_role_top_executive",
        desc="Appointment is for the institution's top executive role and is titled President or Chancellor",
        parent=appt_node,
        critical=True
    )
    # Build claim
    name_str = app.full_name or ""
    inst_str = app.institution or ""
    role_claim = (
        f"According to the provided source(s), {name_str} was officially appointed as the top executive "
        f"of {inst_str}, with the title being either 'President' or 'Chancellor' (including reasonable variants "
        f"like 'president-designate', 'president-elect', 'chancellor-elect', or 'interim president/chancellor')."
    )
    await evaluator.verify(
        claim=role_claim,
        node=role_leaf,
        sources=app.reference_urls,
        additional_instruction=(
            "Focus on whether the page explicitly or clearly implies that the role is the institution's top executive "
            "(President or Chancellor). Titles like 'president-elect', 'president-designate', or 'interim president' "
            "are acceptable as they still refer to the top executive role. Do not accept provosts, vice presidents, "
            "deans, or chancellors of a multi-campus system unless the context clearly indicates campus top executive."
        )
    )

    # 3) Institution eligibility – verify with sources
    inst_leaf = evaluator.add_leaf(
        id=f"a{index}_institution_eligibility",
        desc="Institution is an accredited four-year research university in the United States",
        parent=appt_node,
        critical=True
    )
    inst_claim = (
        f"'{inst_str}' is a U.S.-based accredited four-year research university."
    )
    await evaluator.verify(
        claim=inst_claim,
        node=inst_leaf,
        sources=app.reference_urls,
        additional_instruction=(
            "Use evidence on the provided page(s) to assess that the institution is a U.S. university. "
            "Accept clear indicators such as .edu domain, mentions of U.S. states or locations, or phrases like "
            "'university' that reasonably imply a four-year research university. If the page is a credible news outlet, "
            "it may describe the institution as a U.S. university. If no evidence supports U.S. four-year research "
            "university status, judge as not supported."
        )
    )

    # 4) Announcement date within Dec 2025 – Feb 2026 – verify with sources
    ann_leaf = evaluator.add_leaf(
        id=f"a{index}_announcement_date_window",
        desc="Provides an official announcement date, and it falls between Dec 2025 and Feb 2026 (inclusive)",
        parent=appt_node,
        critical=True
    )
    ann_str = app.announcement_date or ""
    ann_claim = (
        f"According to the source(s), the appointment was officially announced on '{ann_str}', and that official "
        f"announcement date falls in December 2025, January 2026, or February 2026 (inclusive)."
    )
    await evaluator.verify(
        claim=ann_claim,
        node=ann_leaf,
        sources=app.reference_urls,
        additional_instruction=(
            "Check the press release or article's publication/announcement date and ensure it is within the window: "
            "December 2025, January 2026, or February 2026. If the extracted date doesn't match the page date or "
            "the date falls outside this range, mark as not supported. Allow minor format differences (e.g., 'Dec.' vs 'December')."
        )
    )

    # 5) Start date – verify with sources
    start_leaf = evaluator.add_leaf(
        id=f"a{index}_start_date",
        desc="Provides the expected or actual start date of the presidency/chancellorship",
        parent=appt_node,
        critical=True
    )
    start_str = app.start_date or ""
    start_claim = (
        f"The source(s) state that {name_str}'s expected or actual start date in the role at {inst_str} is '{start_str}'."
    )
    await evaluator.verify(
        claim=start_claim,
        node=start_leaf,
        sources=app.reference_urls,
        additional_instruction=(
            "Look for phrases like 'effective', 'begins', 'will start', or 'assumes office on'. "
            "Allow reasonable month/year formats. If the page does not specify a start date matching the claim, mark as not supported."
        )
    )

    # 6) Previous position + institution – verify with sources
    prev_leaf = evaluator.add_leaf(
        id=f"a{index}_previous_position",
        desc="Provides the appointee's immediately previous position AND the institution where they held it",
        parent=appt_node,
        critical=True
    )
    prev_title = app.previous_position_title or ""
    prev_inst = app.previous_institution or ""
    prev_claim = (
        f"Before this appointment, {name_str} served as '{prev_title}' at '{prev_inst}'."
    )
    await evaluator.verify(
        claim=prev_claim,
        node=prev_leaf,
        sources=app.reference_urls,
        additional_instruction=(
            "Verify that the page explicitly mentions the immediately previous position and the associated institution. "
            "Allow minor phrasing variations but the role and institution should match the claim."
        )
    )

    return appt_node


# --------------------------------------------------------------------------- #
# Set-level constraints                                                       #
# --------------------------------------------------------------------------- #
def compute_valid_appointments_and_uniqueness(
    appt_nodes: List[VerificationNode],
    appt_items: List[AppointmentItem]
) -> Dict[str, Any]:
    """
    Determine how many appointment nodes are fully valid (all critical checks passed),
    and whether the counted valid appointments are distinct in both person and institution.
    """
    valid_indices = []
    for i, node in enumerate(appt_nodes):
        try:
            score = node.compute_score(mutate=False)
        except Exception:
            score = 0.0
        if score == 1.0:
            valid_indices.append(i)

    valid_count = len(valid_indices)

    names = []
    insts = []
    for i in valid_indices:
        names.append(norm_key(appt_items[i].full_name))
        insts.append(norm_key(appt_items[i].institution))

    unique_names = len(set([n for n in names if n]))
    unique_insts = len(set([x for x in insts if x]))

    distinct_ok = (unique_names == valid_count) and (unique_insts == valid_count)

    return {
        "valid_indices": valid_indices,
        "valid_count": valid_count,
        "unique_names": unique_names,
        "unique_institutions": unique_insts,
        "distinct_ok": distinct_ok
    }


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate the answer for U.S. research university President/Chancellor appointments announced Dec 2025–Feb 2026.
    """
    # Initialize evaluator with parallel root
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

    # Extract appointments
    extracted = await evaluator.extract(
        prompt=prompt_extract_appointments(),
        template_class=AppointmentsExtraction,
        extraction_name="appointments_extraction"
    )

    # Keep up to 5 items; pad with empties if fewer
    apps: List[AppointmentItem] = list(extracted.appointments[:5])
    while len(apps) < 5:
        apps.append(AppointmentItem())

    # Build per-appointment subtrees for first 4 required + 5th optional
    appt_nodes: List[VerificationNode] = []
    for i in range(5):
        # Indexing for labels: first 4 are 1..4; 5th is optional
        label_index = i + 1
        node = await verify_appointment(evaluator, root, apps[i], label_index)
        appt_nodes.append(node)

    # Set-level constraints parent node (critical)
    set_node = evaluator.add_parallel(
        id="set_level_constraints",
        desc="Check constraints that apply across the full set of provided appointments",
        parent=root,
        critical=True
    )

    # Compute set-level results based on the verification outcomes
    agg_info = compute_valid_appointments_and_uniqueness(appt_nodes, apps)

    # minimum_valid_appointments (critical)
    evaluator.add_custom_node(
        result=agg_info["valid_count"] >= 4,
        id="minimum_valid_appointments",
        desc="At least four (4) of the provided appointment items are valid (i.e., each passes all per-item critical checks)",
        parent=set_node,
        critical=True
    )

    # distinct_people_and_institutions (critical)
    evaluator.add_custom_node(
        result=bool(agg_info["distinct_ok"]),
        id="distinct_people_and_institutions",
        desc="Each counted valid appointment is for a different individual AND a different institution (no duplicates)",
        parent=set_node,
        critical=True
    )

    # Add useful custom info for debugging
    evaluator.add_custom_info(
        info={
            "valid_indices": agg_info["valid_indices"],
            "valid_count": agg_info["valid_count"],
            "unique_names": agg_info["unique_names"],
            "unique_institutions": agg_info["unique_institutions"]
        },
        info_type="set_level_summary",
        info_name="set_level_summary"
    )

    # Return structured summary
    return evaluator.get_summary()