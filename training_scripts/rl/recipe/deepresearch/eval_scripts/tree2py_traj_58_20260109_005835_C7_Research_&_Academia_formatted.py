import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "aaai_2026_planning"
TASK_DESCRIPTION = (
    "I am planning to attend the 2026 AAAI Conference on Artificial Intelligence. "
    "Please provide comprehensive planning information including: "
    "(1) the official full conference name, "
    "(2) the host city, "
    "(3) the specific venue name, "
    "(4) the overall conference start date, "
    "(5) the overall conference end date, "
    "(6) the main technical conference start date, "
    "(7) the main technical conference end date, "
    "(8) the official website URL, "
    "(9) the Bridge Program start date, "
    "(10) the Bridge Program end date, "
    "(11) the names of co-located conferences held alongside AAAI-26, "
    "(12) whether student activities are available, "
    "(13) information about the multi-track structure and track-specific deadlines, and "
    "(14) whether registration is currently available."
)

# Expected/ground-truth values used for verification claims
ALLOWED_OFFICIAL_NAMES = [
    "The 40th Annual AAAI Conference on Artificial Intelligence",
    "AAAI-26",
]
EXPECTED_HOST_CITY = "Singapore"
EXPECTED_VENUE = "Singapore EXPO"
EXPECTED_OVERALL_START = "January 20, 2026"
EXPECTED_OVERALL_END = "January 27, 2026"
EXPECTED_MAIN_START = "January 22, 2026"
EXPECTED_MAIN_END = "January 25, 2026"
OFFICIAL_URL = "https://aaai.org/conference/aaai/aaai-26/"
EXPECTED_BRIDGE_START = "January 20, 2026"
EXPECTED_BRIDGE_END = "January 21, 2026"
EXPECTED_COLOCATED = ["IAAI-26", "EAAI-26"]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ConferenceInfoExtraction(BaseModel):
    official_conference_name: Optional[str] = None
    host_city: Optional[str] = None
    venue_name: Optional[str] = None
    overall_start_date: Optional[str] = None
    overall_end_date: Optional[str] = None
    main_technical_start_date: Optional[str] = None
    main_technical_end_date: Optional[str] = None
    official_website_url: Optional[str] = None
    bridge_program_start_date: Optional[str] = None
    bridge_program_end_date: Optional[str] = None
    co_located_conferences: List[str] = Field(default_factory=list)
    student_activities_available: Optional[str] = None  # use "yes"/"no"/"unknown"
    multiple_tracks: Optional[str] = None               # "yes"/"no"/"unknown"
    track_specific_deadlines: Optional[str] = None      # "yes"/"no"/"unknown"
    registration_available: Optional[str] = None        # "yes"/"no"/"unknown"


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_conference_info() -> str:
    return """
    Extract the AAAI-26 planning information explicitly stated in the answer.
    Return a JSON with the following fields, using the exact text as it appears in the answer whenever applicable:

    - official_conference_name: string or null
    - host_city: string or null
    - venue_name: string or null
    - overall_start_date: string or null  (accept any format, e.g., "January 20, 2026", "2026-01-20")
    - overall_end_date: string or null
    - main_technical_start_date: string or null
    - main_technical_end_date: string or null
    - official_website_url: string or null (must be an explicit URL shown in the answer; do not invent)
    - bridge_program_start_date: string or null
    - bridge_program_end_date: string or null
    - co_located_conferences: array of strings (each item as written in the answer; can include acronyms like "IAAI-26" or "EAAI-26")
    - student_activities_available: "yes" | "no" | "unknown"
    - multiple_tracks: "yes" | "no" | "unknown"   (does the answer state there are multiple tracks?)
    - track_specific_deadlines: "yes" | "no" | "unknown"  (does the answer state deadlines can differ by track?)
    - registration_available: "yes" | "no" | "unknown"  (does the answer state registration is currently available?)

    General rules:
    - Only extract what is explicitly present in the answer.
    - For booleans, normalize to the strings "yes", "no", or "unknown".
    - For URLs, extract the exact URL(s) as stated, including protocol and trailing slash if present.
    - If a requested piece of information is not found, use null (or "unknown" for the boolean-like fields).
    """


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_conference_checks(
    evaluator: Evaluator,
    root,
    extracted: ConferenceInfoExtraction
) -> None:
    """
    Build the rubric tree and run verifications.
    Following the rubric, we use a critical parallel parent node, and each concrete check is a critical leaf.
    The claims are phrased to verify that the answer explicitly provides the expected information.
    """

    # Top-level critical node matching rubric's "AAAI_2026_Conference_Information"
    info_root = evaluator.add_parallel(
        id="AAAI_2026_Conference_Information",
        desc="Complete and constraint-satisfying information package for AAAI-26 planning",
        parent=root,
        critical=True
    )

    claims_and_nodes = []

    # 1) Official Conference Name
    node_official_name = evaluator.add_leaf(
        id="Official_Conference_Name",
        desc="Provides the official conference name using an allowed form",
        parent=info_root,
        critical=True
    )
    claim_official_name = (
        "The answer explicitly states the official conference name as either "
        "'The 40th Annual AAAI Conference on Artificial Intelligence' or 'AAAI-26'."
    )
    add_ins_official_name = (
        "Accept minor variations in punctuation or the presence of both forms together "
        "(e.g., 'AAAI-26: The 40th Annual AAAI Conference on Artificial Intelligence'), "
        "but the wording must clearly correspond to one of the allowed forms and be explicitly shown in the answer."
    )
    claims_and_nodes.append((claim_official_name, None, node_official_name, add_ins_official_name))

    # 2) Host City
    node_host_city = evaluator.add_leaf(
        id="Host_City",
        desc="States the host city as Singapore",
        parent=info_root,
        critical=True
    )
    claim_host_city = "The answer explicitly states the host city as Singapore."
    add_ins_host_city = "Accept small format variants like 'Singapore, Singapore'."
    claims_and_nodes.append((claim_host_city, None, node_host_city, add_ins_host_city))

    # 3) Venue Name
    node_venue = evaluator.add_leaf(
        id="Venue_Name",
        desc="States the venue as Singapore EXPO",
        parent=info_root,
        critical=True
    )
    claim_venue = "The answer explicitly states the venue as Singapore EXPO."
    add_ins_venue = (
        "Allow equivalent naming such as 'Singapore Expo' or "
        "'SINGAPORE EXPO Convention & Exhibition Centre' as valid references to the same venue."
    )
    claims_and_nodes.append((claim_venue, None, node_venue, add_ins_venue))

    # 4) Overall Conference Start Date
    node_overall_start = evaluator.add_leaf(
        id="Overall_Conference_Start_Date",
        desc="States the overall conference start date as January 20, 2026",
        parent=info_root,
        critical=True
    )
    claim_overall_start = "The answer explicitly states the overall conference start date as January 20, 2026."
    add_ins_overall_start = "Allow different date formats referring to the same calendar date (e.g., 'Jan 20, 2026' or '2026-01-20')."
    claims_and_nodes.append((claim_overall_start, None, node_overall_start, add_ins_overall_start))

    # 5) Overall Conference End Date
    node_overall_end = evaluator.add_leaf(
        id="Overall_Conference_End_Date",
        desc="States the overall conference end date as January 27, 2026",
        parent=info_root,
        critical=True
    )
    claim_overall_end = "The answer explicitly states the overall conference end date as January 27, 2026."
    add_ins_overall_end = "Allow different date formats referring to the same calendar date."
    claims_and_nodes.append((claim_overall_end, None, node_overall_end, add_ins_overall_end))

    # 6) Main Technical Conference Start Date
    node_main_start = evaluator.add_leaf(
        id="Main_Technical_Conference_Start_Date",
        desc="States the main technical conference start date as January 22, 2026",
        parent=info_root,
        critical=True
    )
    claim_main_start = "The answer explicitly states the main technical conference start date as January 22, 2026."
    add_ins_main_start = "Allow different date formats referring to the same calendar date."
    claims_and_nodes.append((claim_main_start, None, node_main_start, add_ins_main_start))

    # 7) Main Technical Conference End Date
    node_main_end = evaluator.add_leaf(
        id="Main_Technical_Conference_End_Date",
        desc="States the main technical conference end date as January 25, 2026",
        parent=info_root,
        critical=True
    )
    claim_main_end = "The answer explicitly states the main technical conference end date as January 25, 2026."
    add_ins_main_end = "Allow different date formats referring to the same calendar date."
    claims_and_nodes.append((claim_main_end, None, node_main_end, add_ins_main_end))

    # 8) Official Website URL (exact)
    node_official_url = evaluator.add_leaf(
        id="Official_Website_URL",
        desc=f"Provides the official website URL exactly as {OFFICIAL_URL}",
        parent=info_root,
        critical=True
    )
    claim_official_url = f"The answer provides the official AAAI-26 website URL exactly as {OFFICIAL_URL}"
    add_ins_official_url = "The match must be exact, including the 'https' protocol and trailing slash."
    claims_and_nodes.append((claim_official_url, None, node_official_url, add_ins_official_url))

    # 9) Bridge Program Start Date
    node_bridge_start = evaluator.add_leaf(
        id="Bridge_Program_Start_Date",
        desc="States the Bridge Program start date as January 20, 2026",
        parent=info_root,
        critical=True
    )
    claim_bridge_start = "The answer explicitly states the Bridge Program start date as January 20, 2026."
    add_ins_bridge_start = "Allow different date formats referring to the same calendar date."
    claims_and_nodes.append((claim_bridge_start, None, node_bridge_start, add_ins_bridge_start))

    # 10) Bridge Program End Date
    node_bridge_end = evaluator.add_leaf(
        id="Bridge_Program_End_Date",
        desc="States the Bridge Program end date as January 21, 2026",
        parent=info_root,
        critical=True
    )
    claim_bridge_end = "The answer explicitly states the Bridge Program end date as January 21, 2026."
    add_ins_bridge_end = "Allow different date formats referring to the same calendar date."
    claims_and_nodes.append((claim_bridge_end, None, node_bridge_end, add_ins_bridge_end))

    # 11) Co-Located Conferences: IAAI-26 and EAAI-26 (both must be present)
    node_colocated = evaluator.add_leaf(
        id="Co_Located_Conferences",
        desc="Identifies the co-located conferences as IAAI-26 and EAAI-26 (both included)",
        parent=info_root,
        critical=True
    )
    claim_colocated = "The answer identifies the co-located conferences as IAAI-26 and EAAI-26, and both names appear explicitly."
    add_ins_colocated = "Accept equivalent formatting such as 'IAAI 2026'/'EAAI 2026' as the same conferences."
    claims_and_nodes.append((claim_colocated, None, node_colocated, add_ins_colocated))

    # 12) Student Activities Availability (affirmative)
    node_student = evaluator.add_leaf(
        id="Student_Activities_Availability",
        desc="Confirms that student activities/programs are available",
        parent=info_root,
        critical=True
    )
    claim_student = "The answer confirms that student activities or student programs are available."
    add_ins_student = "Accept synonyms such as 'student program', 'student activities', 'student volunteers program', or similar."
    claims_and_nodes.append((claim_student, None, node_student, add_ins_student))

    # 13) Multiple Tracks Structure (split into two atomic critical leaves)
    multi_parent = evaluator.add_parallel(
        id="Multiple_Tracks_Structure",
        desc="States that the conference has multiple tracks and that deadlines are track-specific",
        parent=info_root,
        critical=True
    )

    node_multi_tracks = evaluator.add_leaf(
        id="Multiple_Tracks_Present",
        desc="States that the conference has multiple tracks",
        parent=multi_parent,
        critical=True
    )
    claim_multi_tracks = "The answer states that the conference has multiple tracks (e.g., multiple submission tracks)."
    add_ins_multi_tracks = "Accept synonyms like 'tracks', 'categories', or 'themes' indicating multiple distinct tracks."
    claims_and_nodes.append((claim_multi_tracks, None, node_multi_tracks, add_ins_multi_tracks))

    node_track_deadlines = evaluator.add_leaf(
        id="Track_Specific_Deadlines",
        desc="States that deadlines may differ by track (track-specific deadlines)",
        parent=multi_parent,
        critical=True
    )
    claim_track_deadlines = "The answer states that deadlines are track-specific (they can differ by track)."
    add_ins_track_deadlines = "Look for explicit mention that different tracks can have different deadlines or timeline variations."
    claims_and_nodes.append((claim_track_deadlines, None, node_track_deadlines, add_ins_track_deadlines))

    # 14) Registration Availability (affirmative)
    node_registration = evaluator.add_leaf(
        id="Registration_Availability",
        desc="Confirms that registration is currently available",
        parent=info_root,
        critical=True
    )
    claim_registration = "The answer confirms that registration is currently available (open)."
    add_ins_registration = "Accept phrases such as 'registration is open', 'registration available', 'now open'."
    claims_and_nodes.append((claim_registration, None, node_registration, add_ins_registration))

    # Run all verifications in parallel
    await evaluator.batch_verify(claims_and_nodes)


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
    """
    Evaluate an answer for the AAAI-26 planning information task.
    Returns the standard Mind2Web2 evaluation summary dictionary.
    """
    # Initialize evaluator
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

    # Extract structured information from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_conference_info(),
        template_class=ConferenceInfoExtraction,
        extraction_name="conference_info_extraction",
    )

    # Add ground truth metadata to the summary
    evaluator.add_ground_truth({
        "allowed_official_names": ALLOWED_OFFICIAL_NAMES,
        "host_city": EXPECTED_HOST_CITY,
        "venue": EXPECTED_VENUE,
        "overall_dates": {"start": EXPECTED_OVERALL_START, "end": EXPECTED_OVERALL_END},
        "main_technical_dates": {"start": EXPECTED_MAIN_START, "end": EXPECTED_MAIN_END},
        "official_website_url": OFFICIAL_URL,
        "bridge_program_dates": {"start": EXPECTED_BRIDGE_START, "end": EXPECTED_BRIDGE_END},
        "co_located_conferences": EXPECTED_COLOCATED,
        "student_activities_required": True,
        "multiple_tracks_required": True,
        "track_specific_deadlines_required": True,
        "registration_available_required": True
    }, gt_type="expected_values")

    # Build verification tree and verify
    await build_and_verify_conference_checks(evaluator, root, extracted_info)

    # Return the evaluation summary
    return evaluator.get_summary()