import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "crew12_eclipse_window_est"
TASK_DESCRIPTION = (
    "NASA's SpaceX Crew-12 mission launched on February 13, 2026, at 5:15 a.m. EST from Cape Canaveral Space Force Station, "
    "carrying a four-person crew to the International Space Station. The Dragon spacecraft docked to the ISS on February 14, 2026, "
    "at approximately 3:15 p.m. Identify the NASA astronaut who serves as the pilot of this Crew-12 mission. Then, determine the "
    "specific time window (in Eastern Standard Time) during which this pilot can observe the totality phase of the March 3, 2026 "
    "total lunar eclipse while aboard the International Space Station. Your answer must include: (1) The full name of the pilot, "
    "(2) The start time of the totality observation window (in EST), (3) The end time of the totality observation window (in EST), "
    "and (4) Reference URLs supporting your answer."
)

# Grounded time claims for the eclipse totality window (EST)
TOTALITY_DATE = "March 3, 2026"
TOTALITY_START_EST = "6:04 a.m. EST"
TOTALITY_END_EST = "7:03 a.m. EST"

# Mission timeline constants
LAUNCH_CLAIM = "NASA's SpaceX Crew-12 mission launched on February 13, 2026 at 5:15 a.m. EST from Cape Canaveral Space Force Station."
DOCKING_CLAIM = "The Dragon spacecraft for Crew-12 docked to the International Space Station on February 14, 2026 at approximately 3:15 p.m. (Eastern Time)."

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MissionExtraction(BaseModel):
    """
    Information about Crew-12 pilot and mission references extracted from the answer.
    """
    pilot_full_name: Optional[str] = None
    pilot_role_label: Optional[str] = None
    mission_urls: List[str] = Field(default_factory=list)     # URLs supporting pilot identity and mission details (launch, crew roles, etc.)
    docking_urls: List[str] = Field(default_factory=list)     # URLs supporting docking details (can overlap with mission_urls)


class EclipseExtraction(BaseModel):
    """
    Information about eclipse timing and observation window extracted from the answer.
    """
    totality_start_est: Optional[str] = None   # As stated in the answer (string)
    totality_end_est: Optional[str] = None     # As stated in the answer (string)
    observation_window_start_est: Optional[str] = None  # Final stated window start in the answer (EST)
    observation_window_end_est: Optional[str] = None    # Final stated window end in the answer (EST)
    eclipse_urls: List[str] = Field(default_factory=list)       # URLs supporting eclipse timing
    iss_visibility_urls: List[str] = Field(default_factory=list)  # URLs supporting ISS/eclipse visibility concepts (optional)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_mission() -> str:
    return """
    From the provided answer, extract the following fields about NASA's SpaceX Crew-12 mission and the pilot:

    - pilot_full_name: The full name of the NASA astronaut who serves as the pilot of Crew-12, exactly as written in the answer.
    - pilot_role_label: The role label used in the answer for this astronaut (e.g., "pilot", "spacecraft pilot"). If not explicitly stated, return null.
    - mission_urls: A list of all URLs cited in the answer that support the Crew-12 mission details and/or the pilot's identity and role. Include pages such as NASA press releases, mission pages, crew bios, or credible news posts that confirm the pilot and launch details.
    - docking_urls: A list of all URLs cited in the answer that support the docking date/time details. If no dedicated docking URLs are given, return an empty list.

    Rules:
    - Only extract URLs that actually appear in the answer. Do not invent any.
    - If a URL appears in markdown link format [text](url), extract the actual URL.
    - If no suitable URLs are provided, return an empty list for that field.
    - Return null for any missing string fields.
    """


def prompt_extract_eclipse() -> str:
    return """
    From the provided answer, extract the following fields about the March 3, 2026 total lunar eclipse and the final observation window (EST) for the pilot aboard the ISS:

    - totality_start_est: The time the answer claims totality begins (in EST). If the answer gives ET instead of EST, extract it as provided. If not provided, return null.
    - totality_end_est: The time the answer claims totality ends (in EST/ET). If not provided, return null.
    - observation_window_start_est: The final start time of the observation window for the pilot to observe totality from the ISS (as presented in the answer, in EST if available). If not provided, return null.
    - observation_window_end_est: The final end time of the observation window (as presented in the answer, in EST if available). If not provided, return null.
    - eclipse_urls: A list of all URLs cited in the answer that specifically support the eclipse timing (start/end of totality). Include NASA, timeanddate.com, USNO, or other credible astronomical references.
    - iss_visibility_urls: A list of any URLs cited that support eclipse visibility from orbit/ISS or general visibility principles for lunar eclipses. If none are included, return an empty list.

    Rules:
    - Only extract URLs that actually appear in the answer. Do not invent any.
    - For time strings, keep exactly the format as it appears in the answer (e.g., "6:04 a.m. EST", "7:03 AM ET").
    - If not provided in the answer, return null for time fields and an empty list for URL arrays.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _has_valid_url(urls: List[str]) -> bool:
    return any(isinstance(u, str) and u.strip().lower().startswith(("http://", "https://")) for u in urls)


def _merge_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for urls in url_lists:
        for u in urls:
            key = u.strip()
            if key and key not in seen:
                seen.add(key)
                merged.append(key)
    return merged


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_crew12_pilot_identification(
    evaluator: Evaluator,
    parent_node,
    mission: MissionExtraction,
) -> None:
    """
    Build and verify the Crew-12 pilot identification subtree.
    Mirrors the rubric:
      Crew_12_Pilot_Identification (critical, sequential)
        - Mission_Verification (critical, parallel)
            • Launch_Date_Verification (critical leaf)
            • Role_Verification (critical leaf)
            • Mission_Reference_URL (critical leaf → existence)
        - ISS_Arrival_Timeline (critical, sequential)
            • Docking_Date_Time (critical leaf)
    """
    # Parent sequential critical node
    pilot_id_node = evaluator.add_sequential(
        id="Crew_12_Pilot_Identification",
        desc="Correctly identify the pilot of NASA's SpaceX Crew-12 mission.",
        parent=parent_node,
        critical=True,
    )

    # Mission_Verification (parallel, critical)
    mission_verif_node = evaluator.add_parallel(
        id="Mission_Verification",
        desc="Verify that the identified astronaut is part of the Crew-12 mission that launched on February 13, 2026.",
        parent=pilot_id_node,
        critical=True,
    )

    # Launch_Date_Verification (critical leaf)
    launch_leaf = evaluator.add_leaf(
        id="Launch_Date_Verification",
        desc="Confirm the mission launched on February 13, 2026, at 5:15 a.m. EST.",
        parent=mission_verif_node,
        critical=True,
    )
    await evaluator.verify(
        claim=LAUNCH_CLAIM,
        node=launch_leaf,
        sources=_merge_urls(mission.mission_urls, mission.docking_urls),
        additional_instruction=(
            "The page should indicate the Crew-12 launch occurred on Feb 13, 2026 at 5:15 a.m. in the U.S. Eastern time zone. "
            "Treat ET/Eastern the same as EST for this task. Minor formatting variations like 'AM' vs 'a.m.' are acceptable."
        ),
    )

    # Role_Verification (critical leaf)
    role_leaf = evaluator.add_leaf(
        id="Role_Verification",
        desc="Confirm the identified astronaut serves as the pilot of the mission.",
        parent=mission_verif_node,
        critical=True,
    )
    pilot_name_for_claim = mission.pilot_full_name or ""
    await evaluator.verify(
        claim=f"{pilot_name_for_claim} serves as the pilot of NASA's SpaceX Crew-12 mission.",
        node=role_leaf,
        sources=mission.mission_urls,
        additional_instruction=(
            "Verify that the page explicitly lists this astronaut as the 'pilot' (or equivalent wording) for Crew-12. "
            "Minor naming variations (e.g., middle initials) are acceptable."
        ),
    )

    # Mission_Reference_URL (existence check as a critical leaf)
    evaluator.add_custom_node(
        result=_has_valid_url(mission.mission_urls),
        id="Mission_Reference_URL",
        desc="Provide a valid reference URL confirming the pilot's identity and mission details.",
        parent=mission_verif_node,
        critical=True,
    )

    # ISS_Arrival_Timeline (critical, sequential)
    iss_timeline_node = evaluator.add_sequential(
        id="ISS_Arrival_Timeline",
        desc="Verify the crew's arrival timeline at the International Space Station.",
        parent=pilot_id_node,
        critical=True,
    )

    # Docking_Date_Time (critical leaf)
    docking_leaf = evaluator.add_leaf(
        id="Docking_Date_Time",
        desc="Confirm Dragon docked to the ISS on February 14, 2026, at approximately 3:15 p.m.",
        parent=iss_timeline_node,
        critical=True,
    )
    docking_sources = mission.docking_urls if mission.docking_urls else mission.mission_urls
    await evaluator.verify(
        claim=DOCKING_CLAIM,
        node=docking_leaf,
        sources=docking_sources,
        additional_instruction=(
            "Confirm that Crew-12 (Dragon) docking occurred on Feb 14, 2026. The time is 'approximately 3:15 p.m.' "
            "so a small tolerance (±15 minutes) is acceptable. Treat ET/Eastern the same as EST for this task."
        ),
    )


async def build_eclipse_observation_window(
    evaluator: Evaluator,
    parent_node,
    mission: MissionExtraction,
    eclipse: EclipseExtraction,
) -> None:
    """
    Build and verify the eclipse observation window subtree.
    Mirrors the rubric:
      Eclipse_Observation_Window (critical, sequential)
        - Eclipse_Timing_Identification (critical, parallel)
            • Totality_Start_Time (critical leaf)
            • Totality_End_Time (critical leaf)
            • Eclipse_Reference_URL (critical leaf → existence)
        - Observation_Feasibility_Analysis (critical, sequential)
            • Mission_Timeline_Overlap (critical leaf)
            • Visibility_From_ISS (critical leaf)
        - Final_Time_Window_EST (critical leaf)
    """
    obs_node = evaluator.add_sequential(
        id="Eclipse_Observation_Window",
        desc="Determine the time window during which the pilot can observe the total lunar eclipse from the ISS.",
        parent=parent_node,
        critical=True,
    )

    # Eclipse_Timing_Identification (parallel, critical)
    timing_node = evaluator.add_parallel(
        id="Eclipse_Timing_Identification",
        desc="Identify the correct timing of the March 3, 2026 total lunar eclipse.",
        parent=obs_node,
        critical=True,
    )

    # Totality_Start_Time (critical leaf)
    tot_start_leaf = evaluator.add_leaf(
        id="Totality_Start_Time",
        desc="Identify that totality begins at 6:04 a.m. EST on March 3, 2026.",
        parent=timing_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"On {TOTALITY_DATE}, totality begins at {TOTALITY_START_EST}.",
        node=tot_start_leaf,
        sources=eclipse.eclipse_urls,
        additional_instruction=(
            "The source(s) should explicitly state the start of totality in Eastern time (EST/ET). "
            "If a source uses ET/Eastern instead of EST, consider it equivalent for this task. "
            "Minor format variations like 'AM' vs 'a.m.' are acceptable if the time is 6:04."
        ),
    )

    # Totality_End_Time (critical leaf)
    tot_end_leaf = evaluator.add_leaf(
        id="Totality_End_Time",
        desc="Identify that totality ends at 7:03 a.m. EST on March 3, 2026.",
        parent=timing_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"On {TOTALITY_DATE}, totality ends at {TOTALITY_END_EST}.",
        node=tot_end_leaf,
        sources=eclipse.eclipse_urls,
        additional_instruction=(
            "The source(s) should explicitly state the end of totality in Eastern time (EST/ET). "
            "If a source uses ET/Eastern instead of EST, consider it equivalent for this task. "
            "Minor format variations like 'AM' vs 'a.m.' are acceptable if the time is 7:03."
        ),
    )

    # Eclipse_Reference_URL (existence)
    evaluator.add_custom_node(
        result=_has_valid_url(eclipse.eclipse_urls),
        id="Eclipse_Reference_URL",
        desc="Provide a valid reference URL confirming the eclipse timing details.",
        parent=timing_node,
        critical=True,
    )

    # Observation_Feasibility_Analysis (critical, sequential)
    feas_node = evaluator.add_sequential(
        id="Observation_Feasibility_Analysis",
        desc="Determine that the pilot is aboard the ISS during the eclipse and can observe it.",
        parent=obs_node,
        critical=True,
    )

    # Mission_Timeline_Overlap (critical leaf)
    overlap_leaf = evaluator.add_leaf(
        id="Mission_Timeline_Overlap",
        desc="Verify that the pilot is aboard the ISS on March 3, 2026 (between docking on Feb 14 and the eclipse date).",
        parent=feas_node,
        critical=True,
    )
    # Merge mission/docking sources for overlap reasoning
    overlap_sources = _merge_urls(mission.mission_urls, mission.docking_urls)
    pilot_name_for_claim = mission.pilot_full_name or "the Crew-12 pilot"
    await evaluator.verify(
        claim=(
            f"{pilot_name_for_claim} is aboard the International Space Station on March 3, 2026 as part of the "
            "Crew-12 mission, which docked on February 14, 2026 and remains on-orbit during March 2026."
        ),
        node=overlap_leaf,
        sources=overlap_sources,
        additional_instruction=(
            "Use the mission pages to check the rotation/expedition duration. If the sources indicate a multi-month "
            "mission (typical for ISS crew rotations), then March 3, 2026 (about ~17 days after docking) should fall "
            "within the on-orbit period. Minor wording variations are acceptable as long as the inference is supported."
        ),
    )

    # Visibility_From_ISS (critical leaf)
    vis_leaf = evaluator.add_leaf(
        id="Visibility_From_ISS",
        desc="Determine the visibility of the eclipse from the ISS orbital path during the totality window.",
        parent=feas_node,
        critical=True,
    )
    vis_sources = _merge_urls(eclipse.eclipse_urls, eclipse.iss_visibility_urls)
    await evaluator.verify(
        claim=(
            "A total lunar eclipse is visible from locations on Earth's night side, which includes observers in low Earth orbit "
            "such as the International Space Station when it is on the night side during the totality window."
        ),
        node=vis_leaf,
        sources=vis_sources,
        additional_instruction=(
            "The provided sources should support the general principle that lunar eclipses are visible from the entire night side "
            "of Earth (not limited to a narrow band), and that observers in space/LEO (e.g., ISS) can see it when in view. "
            "It's acceptable if one source explains eclipse visibility in general and another mentions orbital/space vantage."
        ),
    )

    # Final_Time_Window_EST (critical leaf)
    final_window_leaf = evaluator.add_leaf(
        id="Final_Time_Window_EST",
        desc="Provide the specific time window in EST format (start time to end time) during totality when observation is possible.",
        parent=obs_node,
        critical=True,
    )
    # Validate that the provided final window matches the totality window and is in EST/ET
    final_start = eclipse.observation_window_start_est or ""
    final_end = eclipse.observation_window_end_est or ""
    await evaluator.verify(
        claim=(
            f"The final observation window in the answer for the pilot aboard the ISS is from '{final_start}' to '{final_end}' on {TOTALITY_DATE}, "
            f"and it matches the totality window '{TOTALITY_START_EST}' to '{TOTALITY_END_EST}' (allow ET/Eastern as equivalent to EST; "
            "accept minor format variations like AM vs a.m.)."
        ),
        node=final_window_leaf,
        additional_instruction=(
            "Compare the answer's stated final window against the target totality window. Accept case-insensitive time markers "
            "and 'ET' vs 'EST' equivalence. This is a consistency check based on the answer text and previously established times."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Crew-12 pilot identification and eclipse observation window task.
    """
    # Initialize evaluator with a sequential root (the task is multi-step)
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
        default_model=model,
    )

    # Extract mission and eclipse info concurrently
    mission_extraction_task = evaluator.extract(
        prompt=prompt_extract_mission(),
        template_class=MissionExtraction,
        extraction_name="mission_extraction",
    )
    eclipse_extraction_task = evaluator.extract(
        prompt=prompt_extract_eclipse(),
        template_class=EclipseExtraction,
        extraction_name="eclipse_extraction",
    )
    mission_info, eclipse_info = await asyncio.gather(mission_extraction_task, eclipse_extraction_task)

    # Add a critical Task_Completion node (as the rubric root under our framework's root)
    task_node = evaluator.add_sequential(
        id="Task_Completion",
        desc="Correctly identify the NASA astronaut from Crew-12 who serves as pilot and determine the specific time window (in EST) during which they can observe the total lunar eclipse from the ISS.",
        parent=root,
        critical=True,
    )

    # Build verification subtrees
    await build_crew12_pilot_identification(evaluator, task_node, mission_info)
    await build_eclipse_observation_window(evaluator, task_node, mission_info, eclipse_info)

    # Record reference ground truth guidance (for transparency)
    evaluator.add_ground_truth({
        "expected_eclipse_totality_window_est": {
            "date": TOTALITY_DATE,
            "start_est": TOTALITY_START_EST,
            "end_est": TOTALITY_END_EST
        },
        "expected_mission_events": {
            "launch": "February 13, 2026 at 5:15 a.m. EST",
            "docking": "February 14, 2026 at approximately 3:15 p.m. Eastern"
        }
    }, gt_type="ground_truth")

    # Return final evaluation summary
    return evaluator.get_summary()