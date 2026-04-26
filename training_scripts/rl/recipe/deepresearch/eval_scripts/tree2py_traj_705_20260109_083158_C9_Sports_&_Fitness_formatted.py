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
TASK_ID = "boston_2026_w45_49_masters"
TASK_DESCRIPTION = (
    "Identify a female runner who successfully qualified for the 2026 Boston Marathon in the 45-49 age group "
    "and is eligible to compete on a masters team. The runner must have: (1) achieved a qualifying time during the "
    "official qualifying window, (2) run at a properly certified marathon that meets all Boston Marathon certification "
    "requirements, (3) used official chip/net timing for qualification, (4) be age-eligible for both the 45-49 age group "
    "(based on age on race day April 20, 2026) and masters team competition (40+ years old), (5) met the qualifying "
    "standard for women aged 45-49, (6) run at a full marathon distance (not shorter, not virtual, not indoor), "
    "(7) have a qualifying time that would allow registration during the September 8-12, 2025 registration window, "
    "and (8) be eligible for a team affiliated with either USATF or RRCA. Provide the runner's full name, the specific "
    "marathon race name where they achieved their qualifying time, the exact qualifying time they achieved, the date of "
    "their qualifying race, and confirmation that the race was certified by USATF, AIMS, or an equivalent national governing body."
)

# Ground truth policy references (for recording purposes)
GROUND_TRUTH_INFO = {
    "age_group_race_day": "Age on race day April 20, 2026 determines division (45–49 target).",
    "masters_threshold": "Masters team eligibility requires age 40+.",
    "women_45_49_standard": "3:45:00 (HH:MM:SS).",
    "qualifying_window": "Sept 1, 2024 through Sept 12, 2025 (inclusive).",
    "registration_window": "Sept 8–12, 2025.",
    "accepted_certifications": "USATF, AIMS, or equivalent national governing body.",
    "race_format": "Full marathon (26.2 miles), not shorter, not virtual, not indoor.",
    "timing_method": "Official chip/net time must be used for qualification.",
    "team_affiliation": "Team must be affiliated with USATF or RRCA (or equivalent foreign NGB).",
}


# --------------------------------------------------------------------------- #
# Data model for extracted information                                        #
# --------------------------------------------------------------------------- #
class QualificationExtraction(BaseModel):
    # Runner identity and gender
    runner_name: Optional[str] = None
    gender: Optional[str] = None
    runner_urls: List[str] = Field(default_factory=list)

    # Age eligibility (based on race day 2026-04-20)
    date_of_birth: Optional[str] = None
    age_on_2026_04_20: Optional[str] = None
    age_urls: List[str] = Field(default_factory=list)

    # Qualifying race details
    race_name: Optional[str] = None
    race_date: Optional[str] = None
    distance: Optional[str] = None
    format_notes: Optional[str] = None
    race_urls: List[str] = Field(default_factory=list)

    # Certification confirmation
    certification_body: Optional[str] = None  # e.g., USATF, AIMS, NGB name
    certification_urls: List[str] = Field(default_factory=list)

    # Qualifying time and timing method
    qualifying_time: Optional[str] = None  # HH:MM:SS or similar
    timing_method: Optional[str] = None  # e.g., "chip/net", "gun", etc.
    time_urls: List[str] = Field(default_factory=list)

    # Team affiliation eligibility
    team_affiliation: Optional[str] = None  # e.g., club name or membership statement
    team_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_qualification() -> str:
    return (
        "From the answer, extract exactly the following fields for the identified female runner who qualified for the "
        "2026 Boston Marathon in the 45–49 age group and is eligible for masters team competition. Do not invent data; "
        "return null for any missing field. Also extract all explicitly mentioned source URLs per field.\n\n"
        "Required JSON fields:\n"
        "- runner_name: Full name of the runner.\n"
        "- gender: Gender label as written (e.g., 'Female', 'Woman', 'F').\n"
        "- runner_urls: Array of URLs that substantiate the runner’s identity/gender.\n\n"
        "- date_of_birth: Runner’s date of birth as stated (if provided).\n"
        "- age_on_2026_04_20: Runner’s age on April 20, 2026 (if stated explicitly in the answer).\n"
        "- age_urls: Array of URLs that substantiate DOB/age info.\n\n"
        "- race_name: Specific marathon race name where the qualifying time was achieved.\n"
        "- race_date: Exact date of the qualifying race.\n"
        "- distance: Distance descriptor (e.g., '26.2 miles', 'full marathon') as written.\n"
        "- format_notes: Any notes indicating not virtual, not indoor, or other format clarifications.\n"
        "- race_urls: Array of URLs related to the race details (name/date/distance/format), including the race’s official page or results page.\n\n"
        "- certification_body: Name of certifying body (USATF, AIMS, or equivalent) as stated.\n"
        "- certification_urls: Array of URLs that substantiate the course certification.\n\n"
        "- qualifying_time: Exact qualifying time achieved (HH:MM:SS or similar).\n"
        "- timing_method: Timing method label used for qualification (e.g., 'chip/net', 'gun').\n"
        "- time_urls: Array of URLs that substantiate the qualifying performance/time and timing method (race results page, official timing page, etc.).\n\n"
        "- team_affiliation: Stated team/club affiliation or eligibility statement (USATF/RRCA).\n"
        "- team_urls: Array of URLs that substantiate the team affiliation or eligibility.\n\n"
        "URL extraction rules:\n"
        "• Extract only actual URLs present in the answer (plain URLs or markdown links). If none are provided for a field, return an empty array for that field.\n"
        "• Do not infer or construct URLs beyond what is in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def dedup_urls(*url_lists: List[str]) -> List[str]:
    """Deduplicate and sanitize a set of URL lists."""
    seen = set()
    result: List[str] = []
    for urls in url_lists:
        for u in urls or []:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                result.append(u)
    return result


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_runner_identity(evaluator: Evaluator, parent_node, info: QualificationExtraction) -> None:
    # Group node: Runner_Identity (critical)
    group = evaluator.add_parallel(
        id="Runner_Identity",
        desc="Runner identity and gender category are provided as required to apply the correct qualifying standards.",
        parent=parent_node,
        critical=True,
    )

    # Full_Name_Provided (critical existence)
    evaluator.add_custom_node(
        result=bool(info.runner_name and info.runner_name.strip()),
        id="Full_Name_Provided",
        desc="Runner's full name is provided.",
        parent=group,
        critical=True,
    )

    # Female_Category_Confirmed (critical verification)
    female_node = evaluator.add_leaf(
        id="Female_Category_Confirmed",
        desc="Runner is identified/confirmed as female (to apply women's standards).",
        parent=group,
        critical=True,
    )

    gender_claim = f"The runner '{info.runner_name or ''}' is female."
    urls = info.runner_urls
    await evaluator.verify(
        claim=gender_claim,
        node=female_node,
        sources=urls if urls else None,
        additional_instruction=(
            "Confirm that the runner is female based on the provided sources. "
            "Accept reasonable variants such as 'F', 'Woman', or 'Female'."
        ),
    )


async def verify_age_eligibility(evaluator: Evaluator, parent_node, info: QualificationExtraction) -> None:
    # Group node: Age_Eligibility (critical)
    group = evaluator.add_parallel(
        id="Age_Eligibility",
        desc="Runner meets age-based eligibility requirements for the 45–49 group on April 20, 2026 and for masters (40+).",
        parent=parent_node,
        critical=True,
    )

    # Age_45_49_On_April_20_2026
    age4549_node = evaluator.add_leaf(
        id="Age_45_49_On_April_20_2026",
        desc="Runner is age-eligible for the 45–49 group based on age on April 20, 2026.",
        parent=group,
        critical=True,
    )
    claim_4549 = (
        f"On April 20, 2026, the runner '{info.runner_name or ''}' is between ages 45 and 49 (inclusive). "
        f"If a date of birth is available, compute age as of 2026-04-20."
    )
    await evaluator.verify(
        claim=claim_4549,
        node=age4549_node,
        sources=info.age_urls if info.age_urls else None,
        additional_instruction=(
            "Use the provided date of birth or explicit age statements in the sources to determine age as of April 20, 2026. "
            "If multiple sources provide DOB, resolve any minor discrepancies; pass only if age falls in 45–49 inclusive."
        ),
    )

    # Masters_40_Plus
    masters_node = evaluator.add_leaf(
        id="Masters_40_Plus",
        desc="Runner is at least 40 years old (masters eligibility).",
        parent=group,
        critical=True,
    )
    claim_masters = (
        f"On April 20, 2026, the runner '{info.runner_name or ''}' is at least 40 years old, satisfying masters eligibility."
    )
    await evaluator.verify(
        claim=claim_masters,
        node=masters_node,
        sources=info.age_urls if info.age_urls else None,
        additional_instruction="Verify age >= 40 as of 2026-04-20 using the sources.",
    )


async def verify_qualifying_race(evaluator: Evaluator, parent_node, info: QualificationExtraction) -> None:
    # Group node: Qualifying_Race (critical)
    group = evaluator.add_parallel(
        id="Qualifying_Race",
        desc="Qualifying performance occurred at an acceptable marathon event; required race details and certification confirmation are provided.",
        parent=parent_node,
        critical=True,
    )

    # Race_Name_Provided
    evaluator.add_custom_node(
        result=bool(info.race_name and info.race_name.strip()),
        id="Race_Name_Provided",
        desc="Specific marathon race name where the qualifying time was achieved is provided.",
        parent=group,
        critical=True,
    )

    # Race_Date_Provided
    evaluator.add_custom_node(
        result=bool(info.race_date and info.race_date.strip()),
        id="Race_Date_Provided",
        desc="Exact date of the qualifying race is provided.",
        parent=group,
        critical=True,
    )

    # Within_Official_Qualifying_Window
    window_node = evaluator.add_leaf(
        id="Within_Official_Qualifying_Window",
        desc="Qualifying race date is within the official qualifying window: Sept 1, 2024 through Sept 12, 2025 (inclusive).",
        parent=group,
        critical=True,
    )
    claim_window = (
        f"The qualifying race date '{info.race_date or ''}' falls within Sept 1, 2024 through Sept 12, 2025 (inclusive)."
    )
    await evaluator.verify(
        claim=claim_window,
        node=window_node,
        sources=info.race_urls if info.race_urls else None,
        additional_instruction=(
            "Use the provided race date in the sources to confirm it lies within the stated official qualifying window."
        ),
    )

    # Marathon_Format_Eligible
    format_node = evaluator.add_leaf(
        id="Marathon_Format_Eligible",
        desc="Qualifying race is an acceptable full marathon (not shorter than a full marathon, not virtual, and not indoor).",
        parent=group,
        critical=True,
    )
    claim_format = (
        f"The qualifying event '{info.race_name or ''}' is a full marathon (26.2 miles), not virtual, and not indoor."
    )
    await evaluator.verify(
        claim=claim_format,
        node=format_node,
        sources=info.race_urls if info.race_urls else None,
        additional_instruction=(
            "Confirm from race information or results pages that the event is a certified full marathon distance (26.2 miles) "
            "and is not virtual or indoor."
        ),
    )

    # Race_Certified_By_Authorized_Body
    cert_node = evaluator.add_leaf(
        id="Race_Certified_By_Authorized_Body",
        desc="Race/course is certified by USATF, AIMS, or an equivalent national governing body (and the answer provides confirmation of this).",
        parent=group,
        critical=True,
    )
    claim_cert = (
        f"The race/course for '{info.race_name or ''}' is certified by {info.certification_body or 'an authorized body'} "
        f"(USATF, AIMS, or equivalent national governing body)."
    )
    cert_urls = info.certification_urls
    await evaluator.verify(
        claim=claim_cert,
        node=cert_node,
        sources=cert_urls if cert_urls else (info.race_urls if info.race_urls else None),
        additional_instruction=(
            "Look for explicit certification statements or official listings indicating USATF, AIMS, or equivalent national governing body certification."
        ),
    )


async def verify_qualifying_time(evaluator: Evaluator, parent_node, info: QualificationExtraction) -> None:
    # Group node: Qualifying_Time (critical)
    group = evaluator.add_parallel(
        id="Qualifying_Time",
        desc="Qualifying time is provided, properly timed, and meets the women's 45–49 standard.",
        parent=parent_node,
        critical=True,
    )

    # Exact_Qualifying_Time_Provided
    evaluator.add_custom_node(
        result=bool(info.qualifying_time and info.qualifying_time.strip()),
        id="Exact_Qualifying_Time_Provided",
        desc="Exact qualifying time achieved is provided.",
        parent=group,
        critical=True,
    )

    # Official_Chip_Net_Time_Used
    chip_node = evaluator.add_leaf(
        id="Official_Chip_Net_Time_Used",
        desc="Qualifying time is based on official chip/net time (not gun time).",
        parent=group,
        critical=True,
    )
    claim_chip = (
        f"The qualifying time for '{info.runner_name or ''}' was based on official chip/net time (not gun time)."
    )
    await evaluator.verify(
        claim=claim_chip,
        node=chip_node,
        sources=info.time_urls if info.time_urls else (info.race_urls if info.race_urls else None),
        additional_instruction=(
            "From the results/timing page, confirm that the qualifying time cited corresponds to the official chip/net time."
        ),
    )

    # Meets_W45_49_Standard
    standard_node = evaluator.add_leaf(
        id="Meets_W45_49_Standard",
        desc="Qualifying time is faster than or equal to 3:45:00 (women age 45–49 standard).",
        parent=group,
        critical=True,
    )
    claim_standard = (
        f"The qualifying time '{info.qualifying_time or ''}' is faster than or equal to 3:45:00 for women aged 45–49."
    )
    await evaluator.verify(
        claim=claim_standard,
        node=standard_node,
        sources=info.time_urls if info.time_urls else None,
        additional_instruction=(
            "Treat 3:45:00 (HH:MM:SS) as the standard for women 45–49. Compare the cited qualifying time numerically; "
            "pass only if the qualifying time is <= 3:45:00."
        ),
    )


async def verify_team_affiliation(evaluator: Evaluator, parent_node, info: QualificationExtraction) -> None:
    # Group node: Team_Affiliation_Eligibility (critical)
    group = evaluator.add_parallel(
        id="Team_Affiliation_Eligibility",
        desc="Runner is eligible for a team affiliated with USATF or RRCA (or equivalent foreign organization), per constraints.",
        parent=parent_node,
        critical=True,
    )

    # Eligible_For_USATF_or_RRCA_Affiliated_Team
    team_node = evaluator.add_leaf(
        id="Eligible_For_USATF_or_RRCA_Affiliated_Team",
        desc="Answer confirms the runner is eligible for a team affiliated with a club holding current membership with USATF or RRCA (or equivalent foreign national athletics organization).",
        parent=group,
        critical=True,
    )
    claim_team = (
        f"The runner '{info.runner_name or ''}' is eligible for a team affiliated with USATF or RRCA (or equivalent foreign national athletics organization)."
    )
    await evaluator.verify(
        claim=claim_team,
        node=team_node,
        sources=info.team_urls if info.team_urls else None,
        additional_instruction=(
            "Confirm eligibility or membership with a club that has current USATF or RRCA affiliation. "
            "Accept equivalent foreign national athletics organization if explicitly stated."
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
    """
    Evaluate the provided answer for the Boston Marathon 2026 W45–49 masters eligibility task.
    Returns a structured summary with the verification tree and final score.
    """
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

    # Create a main critical node under the non-critical root to reflect rubric's critical root
    main_node = evaluator.add_parallel(
        id="Main_Rubric",
        desc="Verify the identified runner qualifies for the 2026 Boston Marathon as a female in the 45–49 age group and is eligible for masters team competition, satisfying all stated qualifying-window, race-format, timing, standard, certification, and team-affiliation constraints; and that all required output fields are provided.",
        parent=root,
        critical=True,
    )

    # Record ground truth/policy context
    evaluator.add_ground_truth(GROUND_TRUTH_INFO, gt_type="policy_reference")

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_qualification(),
        template_class=QualificationExtraction,
        extraction_name="qualification_entry",
    )

    # Build and verify rubric subtrees
    await verify_runner_identity(evaluator, main_node, extracted)
    await verify_age_eligibility(evaluator, main_node, extracted)
    await verify_qualifying_race(evaluator, main_node, extracted)
    await verify_qualifying_time(evaluator, main_node, extracted)
    await verify_team_affiliation(evaluator, main_node, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()