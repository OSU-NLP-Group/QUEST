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
TASK_ID = "us_universities_astronomy_planetarium_eclipse_2026"
TASK_DESCRIPTION = (
    "Identify 4 universities or colleges in the United States that meet ALL of the following criteria: "
    "(1) The institution has an active astronomy or astrophysics academic program (undergraduate, graduate, or both); "
    "(2) The institution operates a planetarium facility that offers public programs; "
    "(3) The institution hosted, organized, or participated in a specific event or activity related to the total lunar eclipse that occurred on March 3, 2026. "
    "For each institution, provide: the institution's name and location (city and state), the name and type of astronomy program, the program's website URL, "
    "the name of the planetarium and its website URL, confirmation that the planetarium offers public programs, a description of the March 2026 eclipse-related event, "
    "the date of the event, and a URL documenting the event."
)

# Ground truth guidance for acceptable eclipse date window.
ECLIPSE_TARGET_UTC_DATE = "2026-03-03"
ECLIPSE_ACCEPTABLE_WINDOW_DESC = "Accept March 1–4, 2026 (inclusive) due to local time vs UTC/day-boundary effects; most US events may occur on Mar 2 local time."


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class InstitutionItem(BaseModel):
    institution_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None

    astronomy_program_name: Optional[str] = None
    astronomy_program_type: Optional[str] = None  # e.g., undergraduate, graduate, both
    astronomy_program_url: Optional[str] = None

    planetarium_name: Optional[str] = None
    planetarium_url: Optional[str] = None

    eclipse_event_description: Optional[str] = None
    eclipse_event_date: Optional[str] = None  # as written in the answer (free-form)
    eclipse_event_url: Optional[str] = None


class InstitutionsExtraction(BaseModel):
    institutions: List[InstitutionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_institutions() -> str:
    return """
Extract up to 6 candidate institutions mentioned in the answer that potentially meet the specified criteria.
For each institution, extract the following fields exactly as written in the answer text (do NOT invent or infer):
- institution_name: The university/college name
- city: City name where the institution (or main campus) is located
- state: Two-letter state code or full state name
- astronomy_program_name: The official astronomy/astrophysics program or department name (e.g., 'Department of Astronomy', 'Department of Physics & Astronomy')
- astronomy_program_type: The level mentioned (e.g., 'undergraduate', 'graduate', 'both', 'bachelor', 'BS', 'MS', 'PhD', 'minor', etc.). If unspecified but a program is clearly stated, return a concise descriptor like 'not specified'.
- astronomy_program_url: A URL in the answer that points to the official astronomy/astrophysics program or department webpage for this institution.
- planetarium_name: The planetarium's official name (as given in the answer)
- planetarium_url: A URL in the answer that points to the official webpage for the planetarium
- eclipse_event_description: A brief description of the institution’s event/activity related to the total lunar eclipse in March 2026.
- eclipse_event_date: The date of that event/activity as written in the answer (free-form, e.g., 'March 2, 2026').
- eclipse_event_url: A URL in the answer documenting that eclipse-related event/activity.

Rules:
1) Extract only what explicitly appears in the answer text. If any field is missing, set it to null.
2) All URL fields must be actual URLs that appear in the answer (plain links or markdown links). If no URL provided, set to null.
3) Do not add extra text or commentary. Keep values concise.

Return a JSON object with a top-level key 'institutions' that is an array of objects with the above fields.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _bool_present(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


# --------------------------------------------------------------------------- #
# Verification for one institution                                            #
# --------------------------------------------------------------------------- #
async def verify_institution(
    evaluator: Evaluator,
    parent_node,
    inst: InstitutionItem,
    idx_zero_based: int,
) -> None:
    i = idx_zero_based + 1
    inst_node = evaluator.add_parallel(
        id=f"institution_{i}",
        desc=["First", "Second", "Third", "Fourth", "Fifth", "Sixth"][idx_zero_based] + " qualifying institution",
        parent=parent_node,
        critical=False  # Non-critical at institution level to allow partial credit across institutions
    )

    # ------------------------ Basic Info ---------------------------------- #
    basic_node = evaluator.add_parallel(
        id=f"inst{i}_basic_info",
        desc="Basic institutional information",
        parent=inst_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_bool_present(inst.institution_name),
        id=f"inst{i}_name",
        desc="Institution name provided",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_bool_present(inst.city) and _bool_present(inst.state),
        id=f"inst{i}_location",
        desc="City and state provided",
        parent=basic_node,
        critical=True
    )

    # --------------------- Astronomy Program ------------------------------ #
    prog_main = evaluator.add_parallel(
        id=f"inst{i}_astronomy_program",
        desc="Astronomy/astrophysics academic program verification",
        parent=inst_node,
        critical=True
    )

    prog_exist = evaluator.add_parallel(
        id=f"inst{i}_program_existence",
        desc="Institution has an astronomy or astrophysics program",
        parent=prog_main,
        critical=True
    )
    evaluator.add_custom_node(
        result=_bool_present(inst.astronomy_program_type),
        id=f"inst{i}_program_type",
        desc="Program level identified (undergraduate, graduate, or both)",
        parent=prog_exist,
        critical=True
    )
    evaluator.add_custom_node(
        result=_bool_present(inst.astronomy_program_name),
        id=f"inst{i}_department_name",
        desc="Official department or program name provided",
        parent=prog_exist,
        critical=True
    )

    prog_verify = evaluator.add_parallel(
        id=f"inst{i}_program_verification",
        desc="Program information verified through official source",
        parent=prog_main,
        critical=True
    )
    # Treat this as an evidence-based verification using the provided URL
    prog_url_leaf = evaluator.add_leaf(
        id=f"inst{i}_program_url",
        desc="URL to astronomy/astrophysics program webpage provided and supports the program’s existence",
        parent=prog_verify,
        critical=True
    )
    prog_claim = (
        f"This webpage is an official page for the astronomy or astrophysics academic program at "
        f"{inst.institution_name or 'the institution'}. It should clearly indicate an astronomy/astrophysics "
        f"program or department (e.g., Department of Astronomy or Physics & Astronomy), confirming an active academic program. "
        f"If the URL is missing or invalid, mark as not supported."
    )
    await evaluator.verify(
        claim=prog_claim,
        node=prog_url_leaf,
        sources=inst.astronomy_program_url,
        additional_instruction="Only pass if the page explicitly represents an astronomy/astrophysics program or department at the institution. If no valid URL is provided, return Incorrect."
    )

    # --------------------- Planetarium ------------------------------------ #
    pl_main = evaluator.add_parallel(
        id=f"inst{i}_planetarium",
        desc="Planetarium facility verification",
        parent=inst_node,
        critical=True
    )

    pl_exist = evaluator.add_parallel(
        id=f"inst{i}_planetarium_existence",
        desc="Institution operates a planetarium",
        parent=pl_main,
        critical=True
    )
    evaluator.add_custom_node(
        result=_bool_present(inst.planetarium_name),
        id=f"inst{i}_planetarium_name",
        desc="Planetarium name provided",
        parent=pl_exist,
        critical=True
    )
    # Public programs verification (must be supported by planetarium URL)
    pl_public_leaf = evaluator.add_leaf(
        id=f"inst{i}_planetarium_public_programs",
        desc="Planetarium offers public programs or shows",
        parent=pl_exist,
        critical=True
    )
    pl_public_claim = (
        f"The planetarium at {inst.institution_name or 'the institution'} offers public programs or shows for the general public "
        f"(e.g., public shows, show schedule, ticketing for public, 'open to the public'). If no valid URL is provided, mark as not supported."
    )
    await evaluator.verify(
        claim=pl_public_claim,
        node=pl_public_leaf,
        sources=inst.planetarium_url,
        additional_instruction="Look for language like 'public shows', 'public programs', 'open to the public', ticketing/schedule for general public. If no valid URL is provided, return Incorrect."
    )

    pl_verify = evaluator.add_parallel(
        id=f"inst{i}_planetarium_verification",
        desc="Planetarium information verified through official source",
        parent=pl_main,
        critical=True
    )
    pl_url_leaf = evaluator.add_leaf(
        id=f"inst{i}_planetarium_url",
        desc="URL to planetarium webpage provided and supports that this is the institution’s planetarium",
        parent=pl_verify,
        critical=True
    )
    pl_url_claim = (
        f"This webpage is an official page for the planetarium facility operated by {inst.institution_name or 'the institution'} "
        f"(it may be housed within a department, museum, or outreach unit). If the URL is missing or invalid, mark as not supported."
    )
    await evaluator.verify(
        claim=pl_url_claim,
        node=pl_url_leaf,
        sources=inst.planetarium_url,
        additional_instruction="Pass only if the page clearly represents the institution’s planetarium. If no valid URL is provided, return Incorrect."
    )

    # --------------------- Eclipse Event (March 2026) --------------------- #
    ev_main = evaluator.add_parallel(
        id=f"inst{i}_eclipse_event",
        desc="March 2026 lunar eclipse event participation",
        parent=inst_node,
        critical=True
    )

    ev_details = evaluator.add_parallel(
        id=f"inst{i}_event_details",
        desc="Eclipse event details provided",
        parent=ev_main,
        critical=True
    )
    evaluator.add_custom_node(
        result=_bool_present(inst.eclipse_event_description),
        id=f"inst{i}_event_description",
        desc="Description of the eclipse-related event or activity",
        parent=ev_details,
        critical=True
    )
    # Event date must be around Mar 2–3, 2026 (allow window)
    ev_date_leaf = evaluator.add_leaf(
        id=f"inst{i}_event_date",
        desc="Event date is March 2-3, 2026 or immediately surrounding dates",
        parent=ev_details,
        critical=True
    )
    ev_date_claim = (
        "The event documented on the cited page took place on or around March 2–3, 2026. "
        "Accept events dated March 1–4, 2026 (inclusive), accounting for time zone differences between local time and UTC. "
        "If no valid URL is provided, mark as not supported."
    )
    await evaluator.verify(
        claim=ev_date_claim,
        node=ev_date_leaf,
        sources=inst.eclipse_event_url,
        additional_instruction="Check the date presented on the page. If it falls within Mar 1–4, 2026 inclusive, pass. If no valid URL is provided, return Incorrect."
    )

    ev_verify = evaluator.add_parallel(
        id=f"inst{i}_event_verification",
        desc="Eclipse event documented through official source",
        parent=ev_main,
        critical=True
    )
    ev_url_leaf = evaluator.add_leaf(
        id=f"inst{i}_event_url",
        desc="URL documenting the eclipse event provided and supports participation/hosting by the institution",
        parent=ev_verify,
        critical=True
    )
    ev_url_claim = (
        f"The cited page documents that {inst.institution_name or 'the institution'} hosted, organized, or participated in an event or activity "
        f"related to the total lunar eclipse of March 2026 (UTC date March 3, 2026; may appear as March 2 locally). "
        f"If no valid URL is provided, mark as not supported."
    )
    await evaluator.verify(
        claim=ev_url_claim,
        node=ev_url_leaf,
        sources=inst.eclipse_event_url,
        additional_instruction="Look for explicit mention of a (total) lunar eclipse event in early March 2026 and the institution’s involvement. If no valid URL is provided, return Incorrect."
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
    Evaluate an answer for the astronomy/planetarium/March 2026 eclipse institutions task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Institutions evaluated independently
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

    # Record GT guidance (non-scoring metadata)
    evaluator.add_ground_truth({
        "eclipse_target_utc_date": ECLIPSE_TARGET_UTC_DATE,
        "acceptable_window_note": ECLIPSE_ACCEPTABLE_WINDOW_DESC
    }, gt_type="ground_truth_guidance")

    # Extract institutions from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_institutions(),
        template_class=InstitutionsExtraction,
        extraction_name="institutions_extraction"
    )

    # Normalize to exactly 4 institutions: first 4 if more; pad with blanks if fewer
    institutions: List[InstitutionItem] = list(extracted.institutions or [])
    if len(institutions) > 4:
        institutions = institutions[:4]
    while len(institutions) < 4:
        institutions.append(InstitutionItem())

    # Verify each institution subtree
    for idx, inst in enumerate(institutions):
        await verify_institution(evaluator, root, inst, idx)

    return evaluator.get_summary()