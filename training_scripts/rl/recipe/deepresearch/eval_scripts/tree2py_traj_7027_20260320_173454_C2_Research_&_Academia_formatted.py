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
TASK_ID = "lunar_eclipse_2026_planning"
TASK_DESCRIPTION = """
A university astronomy research team is planning an observation campaign for the March 3, 2026 total lunar eclipse and needs to coordinate with multiple observatories across the United States. For their planning documentation, determine: (1) the precise duration of the totality phase in minutes, calculated from the UTC start and end times, and (2) which region of the continental United States (Western, Central, or Eastern) offers the best viewing opportunity to observe the complete totality phase before sunrise.
"""

# Ground truth (used for reporting and guidance)
EXPECTED_START_UTC = "11:04"
EXPECTED_END_UTC = "12:03"
EXPECTED_DURATION_MIN = "59"
EXPECTED_BEST_REGION = "Western"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EclipseDuration(BaseModel):
    start_utc: Optional[str] = None
    end_utc: Optional[str] = None
    duration_minutes: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class USVisibility(BaseModel):
    best_region: Optional[str] = None  # Expect one of: Western, Central, Eastern
    mentions_before_sunrise: Optional[bool] = None
    mentions_above_horizon: Optional[bool] = None
    source_urls: List[str] = Field(default_factory=list)


class EclipsePlanningExtraction(BaseModel):
    duration: Optional[EclipseDuration] = None
    visibility: Optional[USVisibility] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_eclipse_planning() -> str:
    return """
Extract the specific items below from the answer text for the March 3, 2026 total lunar eclipse planning task.

Section 1: Totality duration (UTC-based)
- duration.start_utc: The UTC time at which totality begins (U2), as stated in the answer. Keep the exact string as written (e.g., "11:04 UTC" or "11:04").
- duration.end_utc: The UTC time at which totality ends (U3), as stated in the answer. Keep the exact string.
- duration.duration_minutes: The reported/computed totality duration in minutes as a number string (digits only, no units). If the answer states "59 minutes" or "about 59 minutes", extract "59". If not stated, return null.
- duration.source_urls: All URLs in the answer that directly support totality timing/duration (e.g., NASA, timeanddate.com). Return an array of URLs.

Section 2: U.S. visibility assessment
- visibility.best_region: The selected region among exactly these choices: "Western", "Central", or "Eastern" (case-insensitive mapping is allowed; output exactly one of those tokens). If no selection is made, return null.
- visibility.mentions_before_sunrise: true if the answer explicitly states that the complete totality can be observed before sunrise / in the early morning (accept phrases like "pre-dawn", "before dawn", "early morning before sunrise"); false if the answer explicitly states otherwise; null if not mentioned.
- visibility.mentions_above_horizon: true if the answer explicitly mentions that to observe the complete totality, the Moon must remain above the horizon for the entire totality (e.g., says "above the horizon", "does not set during totality", "remains up"); false if explicitly contradicted; null if not mentioned.
- visibility.source_urls: All URLs that support the viewing/visibility assessment for U.S. observers. Return an array of URLs.

Rules:
- Do not invent information. Extract only what is explicitly present in the answer.
- If URLs are provided as markdown links, return the actual URLs.
- If a field is missing, set it to null (for strings/bools) or [] (for URL arrays).
"""


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_totality_duration(evaluator: Evaluator, parent_node, extracted: EclipsePlanningExtraction) -> None:
    """
    Build and verify the totality duration subtree.
    We split the check into:
    - The answer explicitly reports "59 minutes" (content check).
    - The fact "totality lasts 59 minutes from 11:04–12:03 UTC" is supported by cited sources (evidence check).
    """
    duration_node = evaluator.add_parallel(
        id="Totality_Duration_Analysis",
        desc="Determine the duration of the totality phase in minutes from the provided UTC start/end times.",
        parent=parent_node,
        critical=True
    )

    # Leaf 1: The answer reports the correct duration in minutes (59)
    leaf_report_59 = evaluator.add_leaf(
        id="Computes_And_Reports_Duration_In_Minutes",
        desc="Computes and reports the totality duration (in minutes) from 11:04 UTC to 12:03 UTC (i.e., 59 minutes).",
        parent=duration_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly reports the totality duration as 59 minutes (accept phrasing like 'about 59 minutes' or '≈59 minutes').",
        node=leaf_report_59,
        additional_instruction="Focus only on whether the answer text reports 59 minutes (allowing small qualifiers like about/approximately). Do not infer from unstated numbers."
    )

    # Leaf 2: The stated duration and UTC endpoints are supported by cited sources
    duration_sources = []
    if extracted and extracted.duration and extracted.duration.source_urls:
        duration_sources.extend(extracted.duration.source_urls)
    if extracted and extracted.visibility and extracted.visibility.source_urls:
        # Sometimes users cite one set of sources for multiple claims; include them as well.
        for url in extracted.visibility.source_urls:
            if url not in duration_sources:
                duration_sources.append(url)

    leaf_supported = evaluator.add_leaf(
        id="Totality_Duration_Supported_By_Sources",
        desc="The 59-minute totality (11:04–12:03 UTC) is supported by cited sources.",
        parent=duration_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"For the March 3, 2026 total lunar eclipse, the totality phase lasts {EXPECTED_DURATION_MIN} minutes, from {EXPECTED_START_UTC} UTC to {EXPECTED_END_UTC} UTC.",
        node=leaf_supported,
        sources=duration_sources,
        additional_instruction="Verify on the provided pages that the totality (U2–U3) spans roughly 59 minutes with U2 near 11:04 UTC and U3 near 12:03 UTC. Allow ±1 minute tolerance and seconds-level differences across sources."
    )


async def verify_us_visibility_required(evaluator: Evaluator, parent_node, extracted: EclipsePlanningExtraction) -> None:
    """
    Build and verify the required U.S. visibility subtree:
    - Selects 'Western' from allowed choices (Western/Central/Eastern).
    - Mentions 'before sunrise' / 'early morning' viewing condition.
    """
    visibility_req_node = evaluator.add_parallel(
        id="US_Visibility_Assessment",
        desc="Identify which region of the continental U.S. (Western/Central/Eastern) offers the best opportunity to observe complete totality before sunrise.",
        parent=parent_node,
        critical=True
    )

    # Leaf 1: Region selection is made from allowed choices and identifies Western as best
    leaf_region = evaluator.add_leaf(
        id="Selects_Region_From_Allowed_Choices",
        desc="Selects a single region from the allowed choices (Western, Central, or Eastern) and identifies Western as best per the provided constraints.",
        parent=visibility_req_node,
        critical=True
    )

    await evaluator.verify(
        claim="The answer selects 'Western' as the best region among the allowed choices Western, Central, and Eastern.",
        node=leaf_region,
        additional_instruction="Treat 'Western U.S.', 'the West', 'West Coast/Mountain time regions' as Western. Focus only on what the answer selects."
    )

    # Leaf 2: Mentions before-sunrise viewing condition
    leaf_before_sunrise = evaluator.add_leaf(
        id="Mentions_Before_Sunrise_Viewing_Condition",
        desc="States that the complete-totality viewing opportunity for U.S. observers is in the early morning / before sunrise.",
        parent=visibility_req_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that the complete totality can be observed before sunrise (i.e., in the early morning/pre-dawn).",
        node=leaf_before_sunrise,
        additional_instruction="Accept synonyms like 'pre-dawn', 'before dawn', 'early morning before sunrise', or similar clear wording."
    )


async def verify_us_visibility_optional(evaluator: Evaluator, root_node, extracted: EclipsePlanningExtraction) -> None:
    """
    Optional, non-critical check:
    - Mentions that the Moon must remain above the horizon during the entire totality to observe the complete totality.
    Placed under the non-critical root to satisfy framework constraints (critical parents cannot have non-critical children).
    """
    leaf_optional_horizon = evaluator.add_leaf(
        id="Optional_Mentions_Moon_Above_Horizon_Criterion",
        desc="Optionally mentions that observing the complete totality requires the Moon to be above the horizon during the entire totality phase.",
        parent=root_node,
        critical=False
    )
    await evaluator.verify(
        claim="The answer explicitly mentions that to observe the complete or full totality, the Moon must remain above the horizon (i.e., it does not set) for the entire totality.",
        node=leaf_optional_horizon,
        additional_instruction="Look for phrases like 'above the horizon', 'doesn't set during totality', 'remains up', 'no moonset during totality'. Focus on explicit mention in the answer."
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
    Evaluate an answer for the March 3, 2026 total lunar eclipse planning task.
    """
    # Initialize evaluator (root is always non-critical by design)
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
        default_model=model
    )

    # Extract structured fields from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_eclipse_planning(),
        template_class=EclipsePlanningExtraction,
        extraction_name="eclipse_planning_extraction"
    )

    # Record ground truth information for transparency
    evaluator.add_ground_truth({
        "expected_totality_start_utc": EXPECTED_START_UTC,
        "expected_totality_end_utc": EXPECTED_END_UTC,
        "expected_totality_duration_minutes": EXPECTED_DURATION_MIN,
        "expected_best_region": EXPECTED_BEST_REGION
    }, gt_type="ground_truth")

    # Build main critical planning node (as per rubric)
    main_node = evaluator.add_parallel(
        id="Lunar_Eclipse_Research_Planning",
        desc=("Provide both required determinations for the March 3, 2026 total lunar eclipse: "
              "(1) totality duration in minutes from UTC times, and "
              "(2) best U.S. region (Western/Central/Eastern) to observe complete totality before sunrise."),
        parent=root,
        critical=True
    )

    # Verify duration subtree (critical)
    await verify_totality_duration(evaluator, main_node, extracted)

    # Verify U.S. visibility required subtree (critical)
    await verify_us_visibility_required(evaluator, main_node, extracted)

    # Optional non-critical mention about the horizon criterion
    await verify_us_visibility_optional(evaluator, root, extracted)

    # Return final structured summary
    return evaluator.get_summary()