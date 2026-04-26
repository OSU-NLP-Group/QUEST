import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fifth_ranked_state_craft_info"
TASK_DESCRIPTION = """As of January 2026, identify the US state that ranks 5th in the number of Michaels craft store locations. For this state, provide the following information:

1. The state's name
2. The exact number of Michaels stores in this state
3. Confirmation of its 5th-place national ranking for Michaels store count
4. Michaels' operating hours on Thanksgiving Day
5. Michaels' operating hours on Christmas Day
6. Michaels' operating hours on New Year's Eve (December 31)
7. Michaels' operating hours on New Year's Day (January 1)
8. Hobby Lobby's operating hours on Christmas Eve (December 24)
9. The ANSI certification standard number for woodworking safety glasses
10. The filtration percentage of N95 respirator masks used for woodworking dust protection
11. The decibel (dB) threshold at which hearing protection is required for woodworking
12. The standard width range (in inches) for quilting cotton fabric
13. The typical yardage requirement for beginner-level sewing projects

Provide at least one reference URL that supports your identification of the state and its Michaels store count.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class URLBuckets(BaseModel):
    state_count_urls: List[str] = Field(default_factory=list, description="URLs supporting state identification and Michaels store count/ranking")
    michaels_hours_urls: List[str] = Field(default_factory=list, description="URLs cited for Michaels holiday hours")
    hobby_lobby_hours_urls: List[str] = Field(default_factory=list, description="URLs cited for Hobby Lobby holiday hours")
    safety_glasses_urls: List[str] = Field(default_factory=list, description="URLs cited for ANSI eyewear standard")
    n95_urls: List[str] = Field(default_factory=list, description="URLs cited for N95 filtration percentage")
    hearing_urls: List[str] = Field(default_factory=list, description="URLs cited for hearing protection threshold")
    quilting_width_urls: List[str] = Field(default_factory=list, description="URLs cited for quilting cotton width")
    beginner_yardage_urls: List[str] = Field(default_factory=list, description="URLs cited for beginner sewing yardage")


class CraftTaskExtraction(BaseModel):
    # Core state identification
    state_name: Optional[str] = None
    michaels_store_count: Optional[str] = None
    ranking_position: Optional[str] = None

    # Holiday hours
    michaels_hours_thanksgiving: Optional[str] = None
    michaels_hours_christmas: Optional[str] = None
    michaels_hours_new_years_eve: Optional[str] = None
    michaels_hours_new_years_day: Optional[str] = None
    hobby_lobby_hours_christmas_eve: Optional[str] = None

    # Safety/standards and craft fabric info
    safety_glasses_standard: Optional[str] = None
    n95_filtration_percentage: Optional[str] = None
    hearing_protection_threshold_db: Optional[str] = None
    quilting_cotton_width_inches: Optional[str] = None
    beginner_sewing_yardage: Optional[str] = None

    # URL buckets
    urls: URLBuckets = Field(default_factory=URLBuckets)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all_fields() -> str:
    return """
    Extract the following fields EXACTLY as stated in the answer. If any field is not present, set it to null. Do not infer or invent values.

    Required fields:
    - state_name: The US state identified as ranking 5th by number of Michaels craft store locations (as of January 2026).
    - michaels_store_count: The exact number of Michaels locations in that state, as stated in the answer.
    - ranking_position: The ranking position for that state, as expressed (e.g., "5", "5th", "fifth").

    Michaels holiday hours:
    - michaels_hours_thanksgiving: Text of Michaels' Thanksgiving Day hours.
    - michaels_hours_christmas: Text of Michaels' Christmas Day hours.
    - michaels_hours_new_years_eve: Text of Michaels' New Year's Eve (Dec 31) hours.
    - michaels_hours_new_years_day: Text of Michaels' New Year's Day (Jan 1) hours.

    Hobby Lobby holiday hours:
    - hobby_lobby_hours_christmas_eve: Text of Hobby Lobby's Christmas Eve (Dec 24) hours.

    Safety / standards / fabric info:
    - safety_glasses_standard: The ANSI certification standard mentioned for woodworking safety glasses (e.g., "ANSI Z87.1").
    - n95_filtration_percentage: The filtration percentage for N95 respirators (e.g., "95%").
    - hearing_protection_threshold_db: The dB threshold requiring hearing protection (e.g., "85 dB").
    - quilting_cotton_width_inches: The standard quilting cotton width range in inches (e.g., "44-45 inches").
    - beginner_sewing_yardage: The typical yardage for beginner-level sewing projects (e.g., "3-4 yards").

    URLs (extract only URLs explicitly present in the answer; do not invent):
    - urls.state_count_urls: URLs used to support the identified state and its Michaels store count/ranking.
    - urls.michaels_hours_urls: URLs cited for Michaels holiday hours.
    - urls.hobby_lobby_hours_urls: URLs cited for Hobby Lobby holiday hours.
    - urls.safety_glasses_urls: URLs cited for the ANSI eyewear standard.
    - urls.n95_urls: URLs cited for N95 filtration percentage.
    - urls.hearing_urls: URLs cited for the hearing protection threshold.
    - urls.quilting_width_urls: URLs cited for quilting cotton width.
    - urls.beginner_yardage_urls: URLs cited for beginner sewing yardage.

    Return a single JSON object matching the provided schema.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _normalize(s: Optional[str]) -> str:
    return (s or "").strip()


# --------------------------------------------------------------------------- #
# Main verification logic                                                     #
# --------------------------------------------------------------------------- #
async def _build_and_verify_tree(evaluator: Evaluator, data: CraftTaskExtraction) -> None:
    """
    Build the verification tree based on the rubric and run verifications.
    Root must be critical with all children critical as per rubric.
    """
    # Root node (critical, parallel aggregation)
    root = evaluator.add_parallel(
        id="Fifth_Ranked_State_Craft_Information",
        desc="Task requires identifying the US state that ranks 5th in Michaels store count (as of January 2026) and providing comprehensive craft store and woodworking safety information",
        parent=evaluator.root,
        critical=True
    )

    # ----------------- State Identity (presence in the answer) -----------------
    state_identity_leaf = evaluator.add_leaf(
        id="State_Identity",
        desc="The correct state ranking 5th in Michaels store count is identified",
        parent=root,
        critical=True
    )
    state_name = _normalize(data.state_name)
    await evaluator.verify(
        claim=f"In the answer, the US state identified as ranking 5th by number of Michaels craft store locations (as of January 2026) is '{state_name}'.",
        node=state_identity_leaf,
        additional_instruction="Judge whether the answer explicitly names a single US state and asserts it is 5th by Michaels locations. If state_name is missing or empty, mark as incorrect."
    )

    # ----------------- State Michaels Count (supported by sources if provided) -----------------
    state_count_leaf = evaluator.add_leaf(
        id="State_Michaels_Count",
        desc="The exact number of Michaels stores in this state is provided",
        parent=root,
        critical=True
    )
    count_str = _normalize(data.michaels_store_count)
    await evaluator.verify(
        claim=f"As of January 2026, the number of Michaels craft store locations in {state_name if state_name else 'the identified state'} is {count_str}.",
        node=state_count_leaf,
        sources=data.urls.state_count_urls,
        additional_instruction="Verify this numeric count using the provided URL(s). Allow minor formatting differences like commas. If sources are irrelevant or absent, mark unsupported."
    )

    # ----------------- National ranking confirmation (supported by sources) -----------------
    ranking_leaf = evaluator.add_leaf(
        id="National_Ranking_Confirmation",
        desc="Confirmation that this state ranks 5th nationally for Michaels store count",
        parent=root,
        critical=True
    )
    await evaluator.verify(
        claim=f"As of January 2026, {state_name if state_name else 'the identified state'} ranks 5th among US states by number of Michaels store locations.",
        node=ranking_leaf,
        sources=data.urls.state_count_urls,
        additional_instruction="Check the provided URL(s) for a ranking or a list that clearly supports a 5th-place position for the identified state. If no source substantiates this, mark unsupported."
    )

    # ----------------- Michaels Thanksgiving hours (provided) -----------------
    thx_leaf = evaluator.add_custom_node(
        result=bool(_normalize(data.michaels_hours_thanksgiving)),
        id="Michaels_Thanksgiving_Hours",
        desc="Michaels' Thanksgiving Day operating hours are provided",
        parent=root,
        critical=True
    )

    # ----------------- Michaels Christmas hours (provided) -----------------
    xmas_leaf = evaluator.add_custom_node(
        result=bool(_normalize(data.michaels_hours_christmas)),
        id="Michaels_Christmas_Hours",
        desc="Michaels' Christmas Day operating hours are provided",
        parent=root,
        critical=True
    )

    # ----------------- Michaels New Year's Eve hours (must be 9 AM - 6 PM) -----------------
    nye_leaf = evaluator.add_leaf(
        id="Michaels_New_Years_Eve_Hours",
        desc="Michaels' New Year's Eve operating hours (9 AM - 6 PM) are provided",
        parent=root,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, Michaels' New Year's Eve (December 31) hours are given as 9 AM - 6 PM (or an obviously equivalent phrasing like 9am–6pm).",
        node=nye_leaf,
        additional_instruction="Check the answer content only for a report of NYE hours equal to 9 AM–6 PM. Minor formatting variants are acceptable."
    )

    # ----------------- Michaels New Year's Day hours (must be 9 AM - 7 PM) -----------------
    nyd_leaf = evaluator.add_leaf(
        id="Michaels_New_Years_Day_Hours",
        desc="Michaels' New Year's Day operating hours (9 AM - 7 PM) are provided",
        parent=root,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, Michaels' New Year's Day (January 1) hours are given as 9 AM - 7 PM (or an obviously equivalent phrasing like 9am–7pm).",
        node=nyd_leaf,
        additional_instruction="Check the answer content only for a report of NYD hours equal to 9 AM–7 PM. Minor formatting variants are acceptable."
    )

    # ----------------- Hobby Lobby Christmas Eve hours (must be 9 AM - 5:30 PM) -----------------
    hl_xmas_eve_leaf = evaluator.add_leaf(
        id="Hobby_Lobby_Christmas_Eve_Hours",
        desc="Hobby Lobby's Christmas Eve operating hours (9 AM - 5:30 PM) are provided",
        parent=root,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, Hobby Lobby's Christmas Eve (December 24) hours are given as 9 AM - 5:30 PM (or an obviously equivalent phrasing like 9am–5:30pm).",
        node=hl_xmas_eve_leaf,
        additional_instruction="Check the answer content only for a report of Christmas Eve hours equal to 9 AM–5:30 PM. Minor formatting variants are acceptable."
    )

    # ----------------- Safety Glasses Standard (ANSI Z87.1) -----------------
    eyewear_leaf = evaluator.add_leaf(
        id="Safety_Glasses_Standard",
        desc="The ANSI Z87.1 certification standard for woodworking safety glasses is mentioned",
        parent=root,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, the ANSI certification standard stated for woodworking safety glasses is Z87.1 (ANSI Z87.1).",
        node=eyewear_leaf,
        additional_instruction="Accept variants like 'ANSI Z87', but it must clearly specify Z87.1 as the standard."
    )

    # ----------------- N95 filtration percentage (95%) -----------------
    n95_leaf = evaluator.add_leaf(
        id="Dust_Mask_Filtration",
        desc="The N95 mask filtration percentage (95% of airborne particles) is provided",
        parent=root,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, N95 respirators are stated to filter 95% of airborne particles.",
        node=n95_leaf,
        additional_instruction="The answer should explicitly indicate 95% (N95 = 95%). Allow minor wording variations."
    )

    # ----------------- Hearing protection threshold (85 dB) -----------------
    hearing_leaf = evaluator.add_leaf(
        id="Hearing_Protection_Threshold",
        desc="The 85 decibel (dB) threshold for requiring hearing protection is provided",
        parent=root,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, the threshold at which hearing protection is required is 85 dB.",
        node=hearing_leaf,
        additional_instruction="Allow equivalent phrasing such as '≥ 85 dB' or 'at 85 dB'."
    )

    # ----------------- Quilting cotton width (44–45 inches) -----------------
    quilting_leaf = evaluator.add_leaf(
        id="Quilting_Cotton_Width",
        desc="The standard quilting cotton fabric width (44-45 inches) is provided",
        parent=root,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, the standard quilting cotton fabric width is stated as 44–45 inches.",
        node=quilting_leaf,
        additional_instruction="Accept small variants like '44 to 45 inches', '44-45 in', or 'approximately 44 inches wide (commonly 44–45 inches)'."
    )

    # ----------------- Beginner fabric yardage (3–4 yards) -----------------
    yardage_leaf = evaluator.add_leaf(
        id="Beginner_Fabric_Requirement",
        desc="The typical fabric requirement for beginner sewing projects (3-4 yards) is provided",
        parent=root,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, typical beginner-level sewing projects require about 3–4 yards of fabric.",
        node=yardage_leaf,
        additional_instruction="Accept reasonable phrasing variations like '3 to 4 yards', 'around 3–4 yds'."
    )

    # ----------------- Reference URL presence (state identification & count) -----------------
    ref_url_leaf = evaluator.add_custom_node(
        result=len(data.urls.state_count_urls) >= 1,
        id="Reference_URL",
        desc="At least one reference URL supporting the state identification and store count is provided",
        parent=root,
        critical=True
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
    Evaluate an answer for the 'Fifth_Ranked_State_Craft_Information' task using the Mind2Web2 framework.
    """
    # Initialize evaluator (root inside evaluator is a container; we'll attach our true root under it)
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured info from the answer
    extraction: CraftTaskExtraction = await evaluator.extract(
        prompt=prompt_extract_all_fields(),
        template_class=CraftTaskExtraction,
        extraction_name="craft_task_extraction",
    )

    # Add explicit ground-truth expectations that are fixed by rubric for transparency
    evaluator.add_ground_truth(
        {
            "expected_fixed_values": {
                "Michaels_New_Years_Eve_Hours": "9 AM - 6 PM",
                "Michaels_New_Years_Day_Hours": "9 AM - 7 PM",
                "Hobby_Lobby_Christmas_Eve_Hours": "9 AM - 5:30 PM",
                "Safety_Glasses_Standard": "ANSI Z87.1",
                "N95_Filtration": "95%",
                "Hearing_Protection_Threshold": "85 dB",
                "Quilting_Cotton_Width": "44–45 inches",
                "Beginner_Fabric_Requirement": "3–4 yards",
            },
            "notes": "Other items (state identity, Michaels store count, 5th-place confirmation) must be supported by at least one provided URL."
        },
        gt_type="rubric_expectations",
    )

    # Build verification tree and run checks
    await _build_and_verify_tree(evaluator, extraction)

    # Return structured evaluation summary
    return evaluator.get_summary()