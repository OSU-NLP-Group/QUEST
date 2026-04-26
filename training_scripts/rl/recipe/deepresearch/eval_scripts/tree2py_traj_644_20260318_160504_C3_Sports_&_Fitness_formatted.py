import asyncio
import logging
import re
from datetime import datetime, date
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "2028_trials_qualification"
TASK_DESCRIPTION = """
Sarah Martinez, a 28-year-old female marathon runner from California, completed the 2026 Chicago Marathon on October 12, 2026, with the following performance details:

- Gun time: 2:36:45
- Chip time: 2:36:20
- Course: USATF-certified marathon course
- Course elevation: Total elevation loss of 85 meters over the 42.195 km distance
- Timing system: Electronic chip timing

Based on the 2028 U.S. Olympic Marathon Trials qualifying standards announced by USATF in June 2025, does Sarah Martinez qualify for the 2028 U.S. Olympic Marathon Trials? Provide a determination with justification for each criterion: (1) time standard, (2) timing method, (3) course certification, (4) elevation requirements, and (5) qualifying window compliance.
"""

# Constants used for logic checks (do not assume truth without source verification leaves)
WOMEN_B_STANDARD_HMS = "2:37:00"  # For logic combination; verified separately via sources
QUAL_WINDOW_START_ISO = "2025-09-01"  # Verified separately via sources
DEFAULT_MARATHON_KM = 42.195


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Runner(BaseModel):
    name: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[str] = None
    home_state: Optional[str] = None


class RacePerformance(BaseModel):
    event_name: Optional[str] = None
    event_date: Optional[str] = None  # Keep as string; verification + parsing downstream
    location: Optional[str] = None
    gun_time: Optional[str] = None
    chip_time: Optional[str] = None
    timing_method: Optional[str] = None  # e.g., "Electronic chip timing"
    course_certification: Optional[str] = None  # e.g., "USATF-certified marathon course"
    course_elevation_loss_meters: Optional[str] = None  # keep string for robustness
    course_distance_km: Optional[str] = None  # may be missing; default to 42.195 if needed
    performance_sources: List[str] = Field(default_factory=list)  # URLs that support performance facts


class StandardsInfo(BaseModel):
    women_marathon_b_standard: Optional[str] = None  # e.g., "2:37:00"
    timing_method_requirement: Optional[str] = None  # e.g., "chip time required"
    course_certification_requirement: Optional[str] = None  # e.g., "USATF/WA/AIMS"
    elevation_rule_description: Optional[str] = None  # e.g., "≤3.3 meters per km"
    qualifying_window_description: Optional[str] = None  # e.g., "Sep 1, 2025 to 60 days before Trials"
    standards_sources: List[str] = Field(default_factory=list)  # URLs (USATF/WA) that state 2028 Trials standards


class AnswerExtraction(BaseModel):
    runner: Optional[Runner] = None
    race: Optional[RacePerformance] = None
    standards: Optional[StandardsInfo] = None
    final_determination: Optional[str] = None  # e.g., "Qualifies" / "Does not qualify" (free text)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_main() -> str:
    return """
Extract the following structured information exactly as stated in the answer text. Do not infer or invent anything.

1) runner:
   - name
   - gender
   - age
   - home_state

2) race (the specific qualifying performance):
   - event_name (e.g., "Chicago Marathon")
   - event_date (keep original wording, e.g., "October 12, 2026")
   - location (city/state if present)
   - gun_time (verbatim string if present)
   - chip_time (verbatim string if present)
   - timing_method (verbatim phrase, e.g., "Electronic chip timing", if present)
   - course_certification (verbatim phrase describing certification, e.g., "USATF-certified marathon course", if present)
   - course_elevation_loss_meters (verbatim phrase/number for total elevation loss; keep units if present)
   - course_distance_km (verbatim text for the course distance if stated, otherwise null)
   - performance_sources (collect all URLs in the answer that support the performance facts or course details; return as an array of URLs)

3) standards (USATF or relevant standards referenced):
   - women_marathon_b_standard (verbatim string if the answer states a women's B standard time)
   - timing_method_requirement (verbatim phrase if the answer states the timing policy, e.g., "chip time required")
   - course_certification_requirement (verbatim phrase if the answer states accepted certifying bodies, e.g., "USATF, World Athletics, or AIMS")
   - elevation_rule_description (verbatim phrase of the elevation rule if present, e.g., "no greater than 3.3 meters per kilometer")
   - qualifying_window_description (verbatim phrase for the qualifying window if present)
   - standards_sources (collect all URLs in the answer that support the standards/policies; return as an array of URLs)

4) final_determination:
   - final_determination (verbatim text if the answer states whether the runner qualifies; else null)

Rules:
- Extract only URLs actually present in the answer; return as performance_sources and standards_sources, respectively.
- For time and date fields, keep them as strings exactly as written.
- If any requested field is not given in the answer, set it to null or empty array as appropriate.
"""


# --------------------------------------------------------------------------- #
# Parsing helpers                                                             #
# --------------------------------------------------------------------------- #
def parse_time_to_seconds(t: Optional[str]) -> Optional[int]:
    if not t:
        return None
    s = t.strip().lower()

    # HH:MM:SS
    m = re.search(r'(\d{1,2}):(\d{1,2}):(\d{1,2})', s)
    if m:
        h, m1, s1 = map(int, m.groups())
        return h * 3600 + m1 * 60 + s1

    # H:MM (assume H:MM)
    m = re.search(r'(\d{1,2}):(\d{2})', s)
    if m:
        h, m1 = map(int, m.groups())
        return h * 3600 + m1 * 60

    # 2h 36m 20s or similar
    h = re.search(r'(\d+)\s*h', s)
    mi = re.search(r'(\d+)\s*m', s)
    se = re.search(r'(\d+)\s*s', s)
    if h or mi or se:
        hh = int(h.group(1)) if h else 0
        mm = int(mi.group(1)) if mi else 0
        ss = int(se.group(1)) if se else 0
        return hh * 3600 + mm * 60 + ss

    # Fallback: pick first numbers groups if it's like "2 36 20"
    nums = re.findall(r'\d+', s)
    if len(nums) == 3:
        try:
            hh, mm, ss = map(int, nums[:3])
            return hh * 3600 + mm * 60 + ss
        except Exception:
            return None
    return None


def parse_float_first(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    m = re.search(r'[-+]?\d*\.?\d+', s.replace(',', ''))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def parse_date_str(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    s = s.strip()
    fmts = [
        "%B %d, %Y",      # October 12, 2026
        "%b %d, %Y",      # Oct 12, 2026
        "%Y-%m-%d",       # 2026-10-12
        "%m/%d/%Y",       # 10/12/2026
        "%Y/%m/%d",       # 2026/10/12
        "%d %B %Y",       # 12 October 2026
        "%Y.%m.%d",       # 2026.10.12
    ]
    for f in fmts:
        try:
            return datetime.strptime(s, f).date()
        except Exception:
            continue
    # Try to extract month/day/year roughly
    try:
        # Handle cases like "Oct. 12, 2026"
        s2 = re.sub(r'(\b[A-Za-z]{3,}\.)', lambda m: m.group(1)[:-1], s)
        for f in ["%b %d, %Y", "%B %d, %Y"]:
            try:
                return datetime.strptime(s2, f).date()
            except Exception:
                pass
    except Exception:
        pass
    return None


def str_contains_any(hay: Optional[str], needles: List[str]) -> bool:
    if not hay:
        return False
    h = hay.lower()
    return any(n.lower() in h for n in needles)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_time_standard_met(evaluator: Evaluator, parent, ext: AnswerExtraction) -> None:
    node = evaluator.add_parallel(
        id="Time_Standard_Met",
        desc="Verifies that the runner's marathon time meets the USATF B standard (2:16:00 or faster for men, 2:37:00 or faster for women)",
        parent=parent,
        critical=True
    )

    standards_urls = (ext.standards.standards_sources if ext and ext.standards else []) or []
    perf_urls = (ext.race.performance_sources if ext and ext.race else []) or []

    # 1) Verify the standard value for women's marathon B standard (source-grounded)
    leaf_std = evaluator.add_leaf(
        id="time_standard_policy_women_b",
        desc="Policy: Women's marathon B standard for 2028 Trials is 2:37:00 or faster",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="For the 2028 U.S. Olympic Marathon Trials, the women's marathon qualifying B standard is 2:37:00 (2 hours, 37 minutes, 00 seconds) or faster.",
        node=leaf_std,
        sources=standards_urls,
        additional_instruction="Look for the official USATF Trials qualifying standards announced for 2028, specifically the women's marathon B standard."
    )

    # 2) Verify the athlete's chip time (source-grounded)
    chip_time_str = ext.race.chip_time if ext and ext.race else None
    leaf_time_val = evaluator.add_leaf(
        id="chip_time_value_verified",
        desc=f"Sarah Martinez's chip time at the 2026 Chicago Marathon is '{chip_time_str or '[missing]'}'",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Sarah Martinez's chip (net) time at the 2026 Chicago Marathon was {chip_time_str}.",
        node=leaf_time_val,
        sources=perf_urls,
        additional_instruction="Use official results or event timing pages to confirm the individual's chip (net) time."
    )

    # 3) Logic: Chip time meets or is faster than the B standard
    std_secs = parse_time_to_seconds(WOMEN_B_STANDARD_HMS)
    chip_secs = parse_time_to_seconds(chip_time_str)
    meets = (chip_secs is not None and std_secs is not None and chip_secs <= std_secs)
    evaluator.add_custom_node(
        result=bool(meets),
        id="chip_time_meets_b_standard_logic",
        desc=f"Logic: Chip time ({chip_time_str or 'N/A'}) is equal to or faster than women's B standard ({WOMEN_B_STANDARD_HMS})",
        parent=node,
        critical=True
    )


async def verify_timing_method(evaluator: Evaluator, parent, ext: AnswerExtraction) -> None:
    node = evaluator.add_parallel(
        id="Timing_Method_Chip_Time",
        desc="Confirms that the qualifying time was recorded using chip time (not gun time), as required by USATF for 2028 trials qualification",
        parent=parent,
        critical=True
    )

    standards_urls = (ext.standards.standards_sources if ext and ext.standards else []) or []
    perf_urls = (ext.race.performance_sources if ext and ext.race else []) or []

    # 1) Verify the policy that chip/net time is required
    leaf_policy = evaluator.add_leaf(
        id="timing_policy_chip_required",
        desc="Policy: USATF requires chip/net time (not gun time) for 2028 Trials qualification",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="For the 2028 U.S. Olympic Marathon Trials, qualifying times must be based on chip (net) time, not gun time.",
        node=leaf_policy,
        sources=standards_urls,
        additional_instruction="Look for explicit language that the 'chip time' (also called net time) is used for qualification."
    )

    # 2) Verify that the race uses chip timing / that a chip (net) time is recorded
    timing_method_str = ext.race.timing_method if ext and ext.race else None
    leaf_event_chip = evaluator.add_leaf(
        id="event_uses_chip_timing",
        desc=f"Event timing method confirms chip timing ('{timing_method_str or '[missing]'}') and results include chip/net times",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The 2026 Chicago Marathon uses electronic chip timing and provides chip/net times in the official results.",
        node=leaf_event_chip,
        sources=perf_urls,
        additional_instruction="Look for mentions of 'chip', 'net time', or transponder timing in the official results or race information."
    )

    # 3) Logic: The qualifying time used in the answer is a chip time
    chip_time_present = (ext and ext.race and bool(ext.race.chip_time))
    timing_mentions_chip = str_contains_any(timing_method_str, ["chip", "net"])
    evaluator.add_custom_node(
        result=bool(chip_time_present and timing_mentions_chip),
        id="qualifying_time_is_chip_logic",
        desc="Logic: A chip (net) time is present and timing method references chip/net timing",
        parent=node,
        critical=True
    )


async def verify_course_certification(evaluator: Evaluator, parent, ext: AnswerExtraction) -> None:
    node = evaluator.add_parallel(
        id="Course_Certification_Valid",
        desc="Verifies that the marathon was run on a course certified by USATF, World Athletics, or AIMS",
        parent=parent,
        critical=True
    )

    standards_urls = (ext.standards.standards_sources if ext and ext.standards else []) or []
    perf_urls = (ext.race.performance_sources if ext and ext.race else []) or []
    course_cert_str = ext.race.course_certification if ext and ext.race else None

    # 1) Verify the policy for accepted certification bodies
    leaf_policy = evaluator.add_leaf(
        id="cert_policy_orgs",
        desc="Policy: Trials-qualifying performances must be on courses certified by USATF, World Athletics, or AIMS",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="For the 2028 U.S. Olympic Marathon Trials, courses must be certified by USATF, World Athletics, or AIMS for performances to qualify.",
        node=leaf_policy,
        sources=standards_urls,
        additional_instruction="Look for explicit mention of accepted certifying bodies (USATF, World Athletics, AIMS)."
    )

    # 2) Verify that the Chicago Marathon course is certified by one of the accepted bodies
    leaf_course_cert = evaluator.add_leaf(
        id="course_cert_verified",
        desc=f"Event course certification is valid (stated as: '{course_cert_str or '[missing]'}')",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Chicago Marathon course is certified by at least one of USATF, World Athletics, or AIMS.",
        node=leaf_course_cert,
        sources=perf_urls,
        additional_instruction="Accept evidence such as 'USATF-certified', 'World Athletics Elite/Label', AIMS recognition, or a course certification number."
    )

    # 3) Logic: The extracted certification text indicates a valid certifying body
    logic_ok = str_contains_any(
        course_cert_str,
        ["usatf", "world athletics", "aims", "certified"]
    )
    evaluator.add_custom_node(
        result=bool(logic_ok),
        id="course_cert_logic",
        desc="Logic: Extracted course certification string indicates valid certification (USATF/WA/AIMS)",
        parent=node,
        critical=True
    )


async def verify_elevation_requirement(evaluator: Evaluator, parent, ext: AnswerExtraction) -> None:
    node = evaluator.add_parallel(
        id="Elevation_Loss_Requirement",
        desc="Confirms that the course has an elevation loss of no greater than 3.3 meters per kilometer, as required for trials qualification",
        parent=parent,
        critical=True
    )

    standards_urls = (ext.standards.standards_sources if ext and ext.standards else []) or []
    perf_urls = (ext.race.performance_sources if ext and ext.race else []) or []

    elev_str = ext.race.course_elevation_loss_meters if ext and ext.race else None
    dist_km_str = ext.race.course_distance_km if ext and ext.race else None
    dist_km = parse_float_first(dist_km_str) or DEFAULT_MARATHON_KM
    elev_m = parse_float_first(elev_str)
    m_per_km = (elev_m / dist_km) if (elev_m is not None and dist_km > 0) else None

    # 1) Verify the policy threshold (≤ 3.3 m/km)
    leaf_policy = evaluator.add_leaf(
        id="elev_policy_threshold",
        desc="Policy: Net elevation loss must not exceed 3.3 meters per kilometer",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="For the 2028 U.S. Olympic Marathon Trials, the course's net elevation loss must not exceed 3.3 meters per kilometer.",
        node=leaf_policy,
        sources=standards_urls,
        additional_instruction="Look for specific numeric elevation-drop limit, e.g., 3.3 m/km."
    )

    # 2) Verify the course elevation figure (source-grounded)
    leaf_elev_val = evaluator.add_leaf(
        id="elev_value_verified",
        desc=f"Course total elevation loss is '{elev_str or '[missing]'}' over ~{dist_km:.3f} km",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Chicago Marathon course has a total elevation loss of {elev_str} over approximately {dist_km:.3f} km.",
        node=leaf_elev_val,
        sources=perf_urls,
        additional_instruction="Use official course maps/profiles or authoritative sources cited in the answer to confirm the total elevation loss."
    )

    # 3) Logic: Check m/km <= 3.3
    logic_ok = (m_per_km is not None and m_per_km <= 3.3)
    evaluator.add_custom_node(
        result=bool(logic_ok),
        id="elev_logic_ok",
        desc=f"Logic: Elevation loss per km ({(m_per_km if m_per_km is not None else 'N/A')}) ≤ 3.3 m/km",
        parent=node,
        critical=True
    )

    # Record debug info
    evaluator.add_custom_info(
        info={
            "parsed_elevation_m": elev_m,
            "parsed_distance_km": dist_km,
            "computed_m_per_km": m_per_km
        },
        info_type="debug",
        info_name="elevation_computation"
    )


async def verify_qualifying_window(evaluator: Evaluator, parent, ext: AnswerExtraction) -> None:
    node = evaluator.add_parallel(
        id="Within_Qualifying_Window",
        desc="Verifies that the qualifying performance occurred within the official qualifying window (September 1, 2025 to 60 days before the trials date)",
        parent=parent,
        critical=True
    )

    standards_urls = (ext.standards.standards_sources if ext and ext.standards else []) or []
    perf_urls = (ext.race.performance_sources if ext and ext.race else []) or []

    event_date_str = ext.race.event_date if ext and ext.race else None
    event_date_parsed = parse_date_str(event_date_str)
    window_start_parsed = parse_date_str(QUAL_WINDOW_START_ISO)

    # 1) Verify the policy for the qualifying window dates
    leaf_policy = evaluator.add_leaf(
        id="qual_window_policy",
        desc="Policy: The qualifying window opens Sep 1, 2025 and closes 60 days before the Trials date",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="For the 2028 U.S. Olympic Marathon Trials, the qualifying window opens on September 1, 2025 and closes 60 days before the Trials date.",
        node=leaf_policy,
        sources=standards_urls,
        additional_instruction="Confirm explicit window start date and the rule that it closes 60 days prior to the Trials."
    )

    # 2) Verify the performance date
    leaf_perf_date = evaluator.add_leaf(
        id="performance_date_verified",
        desc=f"Performance date is '{event_date_str or '[missing]'}'",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The 2026 Chicago Marathon took place on {event_date_str}.",
        node=leaf_perf_date,
        sources=perf_urls,
        additional_instruction="Use official event site, results page, or authoritative source cited in the answer to confirm the event date."
    )

    # 3) Logic: Event occurred within window (>= Sep 1, 2025; and logically before 60-days-prior end for 2028 Trials)
    # Since the Trials are in 2028, any 2026 date that is on/after Sep 1, 2025 will be within window.
    logic_ok = False
    if event_date_parsed and window_start_parsed:
        if event_date_parsed >= window_start_parsed:
            # Upper bound logic: 60 days before a 2028 Trials date will fall in late 2027 or 2028.
            # Any date in 2025-09-01 through 2026 should satisfy the 'before end' condition.
            if event_date_parsed.year in (2025, 2026, 2027):
                logic_ok = True
            else:
                # If it's 2028, we cannot assert without exact Trials date; conservatively mark False
                logic_ok = (event_date_parsed < date(2027, 12, 31))
    evaluator.add_custom_node(
        result=bool(logic_ok),
        id="performance_within_window_logic",
        desc=f"Logic: Event date ({event_date_parsed or 'N/A'}) is within the qualifying window (≥ 2025-09-01 and logically before the '60 days prior' end for 2028)",
        parent=node,
        critical=True
    )

    # Record debug info
    evaluator.add_custom_info(
        info={
            "event_date_str": event_date_str,
            "event_date_parsed": str(event_date_parsed) if event_date_parsed else None,
            "window_start_parsed": str(window_start_parsed) if window_start_parsed else None
        },
        info_type="debug",
        info_name="window_computation"
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
    Evaluate whether the answer shows that Sarah Martinez qualifies for the 2028 U.S. Olympic Marathon Trials
    across five criteria:
    (1) time standard, (2) timing method, (3) course certification, (4) elevation requirement, (5) qualifying window.
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_main(),
        template_class=AnswerExtraction,
        extraction_name="extracted_information"
    )

    # Add the main (critical) qualification node as per rubric
    qual_node = evaluator.add_parallel(
        id="2028_US_Olympic_Marathon_Trials_Qualification",
        desc="Determines whether a runner qualifies for the 2028 U.S. Olympic Marathon Trials based on all required criteria",
        parent=root,
        critical=True
    )

    # Build all five critical verification subtrees
    await verify_time_standard_met(evaluator, qual_node, extracted)
    await verify_timing_method(evaluator, qual_node, extracted)
    await verify_course_certification(evaluator, qual_node, extracted)
    await verify_elevation_requirement(evaluator, qual_node, extracted)
    await verify_qualifying_window(evaluator, qual_node, extracted)

    # Optional: record parsed basics for transparency
    chip_secs = parse_time_to_seconds(extracted.race.chip_time if extracted and extracted.race else None)
    std_secs = parse_time_to_seconds(WOMEN_B_STANDARD_HMS)
    evaluator.add_custom_info(
        info={
            "chip_time_raw": extracted.race.chip_time if extracted and extracted.race else None,
            "chip_time_seconds": chip_secs,
            "women_b_standard_raw": WOMEN_B_STANDARD_HMS,
            "women_b_standard_seconds": std_secs
        },
        info_type="debug",
        info_name="time_parsing"
    )

    return evaluator.get_summary()