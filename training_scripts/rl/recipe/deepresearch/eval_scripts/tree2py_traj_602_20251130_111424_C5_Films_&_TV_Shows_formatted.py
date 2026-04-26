import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "hallmark_series_verification"
TASK_DESCRIPTION = """
Identify the name of a Hallmark Channel original series that meets all of the following criteria: 
- The series is classified as a fantasy or time travel drama;
- The series is filmed in Ontario, Canada (in locations such as Port Perry, Uxbridge, or the Toronto area);
- The series was created by a mother-daughter team who serve as showrunners or creators;
- The series premiered between January 2020 and December 2024, inclusive;
- The series has at least 3 completed seasons as of November 2025;
- Each season of the series contains exactly 10 episodes;
- The series is produced by Hallmark Channel in collaboration with at least two other production companies.
Provide the series name along with supporting reference URLs that verify each of these criteria.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SeriesExtraction(BaseModel):
    """
    Structured extraction for the candidate Hallmark Channel series from the agent's answer.
    All URL fields should contain actual URLs explicitly present in the answer text (plain or markdown).
    """
    series_name: Optional[str] = None

    # Criterion-specific URL buckets (URLs that purportedly verify the criterion)
    hallmark_original_urls: List[str] = Field(default_factory=list)
    genre_urls: List[str] = Field(default_factory=list)
    filming_urls: List[str] = Field(default_factory=list)
    creators_urls: List[str] = Field(default_factory=list)
    premiere_urls: List[str] = Field(default_factory=list)
    seasons_urls: List[str] = Field(default_factory=list)
    episodes_urls: List[str] = Field(default_factory=list)
    production_urls: List[str] = Field(default_factory=list)

    # Optional descriptive fields extracted (used to craft clearer claims; not strictly required)
    classification: Optional[str] = None  # e.g., "fantasy", "time travel", "fantasy drama", etc.
    filming_locations: List[str] = Field(default_factory=list)  # e.g., ["Port Perry", "Uxbridge", "Toronto"]
    creators: List[str] = Field(default_factory=list)  # names of the mother-daughter team (if provided)
    premiere_date: Optional[str] = None  # a date string as provided (e.g., "January 15, 2023")
    seasons_completed_as_of_nov_2025: Optional[str] = None  # e.g., "3", "Season 1–3 completed"
    episodes_per_season_statement: Optional[str] = None  # e.g., "10 episodes per season"
    production_companies: List[str] = Field(default_factory=list)  # list of production companies mentioned


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_series() -> str:
    return """
    Extract the single candidate Hallmark Channel original series mentioned in the answer along with criterion-specific supporting URLs.
    If multiple series are mentioned, select the FIRST one and ignore the rest.

    Required fields:
    - series_name: The series name, exactly as provided.
    
    Criterion URL buckets (extract all explicitly mentioned URLs that claim to verify the criterion):
    - hallmark_original_urls: URLs that indicate the series is a Hallmark Channel original series.
    - genre_urls: URLs that classify the series as "fantasy" or "time travel" (or both), or describe it as a fantasy/time travel drama.
    - filming_urls: URLs that show the series is filmed in Ontario, Canada (e.g., Port Perry, Uxbridge, Toronto area).
    - creators_urls: URLs that indicate the series was created by a mother-daughter team who serve as showrunners or creators.
    - premiere_urls: URLs that show the premiere date, which must be between January 2020 and December 2024 inclusive.
    - seasons_urls: URLs that support that the series has at least 3 completed seasons as of November 2025.
    - episodes_urls: URLs that support that each season contains exactly 10 episodes.
    - production_urls: URLs that list production companies showing Hallmark Channel plus AT LEAST TWO other production companies.

    Optional descriptive fields (if explicitly provided in the answer; otherwise set to null or empty):
    - classification: The classification label(s) (e.g., "fantasy", "time travel", "fantasy drama").
    - filming_locations: Locations mentioned (e.g., "Port Perry", "Uxbridge", "Toronto").
    - creators: Names of the creators/showrunners if the mother-daughter team is named.
    - premiere_date: Premiere date string as provided.
    - seasons_completed_as_of_nov_2025: The series' completed seasons count/status as of November 2025 (use the exact phrasing in the answer).
    - episodes_per_season_statement: Any statement indicating the per-season episode count (e.g., "each season has 10 episodes").
    - production_companies: List of production companies named in the answer.

    Rules for URLs:
    - Extract actual URLs explicitly present in the answer (including markdown links).
    - Do not invent URLs; if none are provided for a bucket, return an empty list.
    - If a URL is missing a protocol, prepend "http://".

    Return a single JSON object with these fields.
    """


# --------------------------------------------------------------------------- #
# Helper for building criterion subtrees                                      #
# --------------------------------------------------------------------------- #
async def build_criterion_with_urls(
    evaluator: Evaluator,
    parent_node,
    criterion_id: str,
    criterion_desc: str,
    url_list: List[str],
    claim_text: str,
    additional_instruction: str,
) -> None:
    """
    Build a small sequential subtree for a single criterion:
      1) URL presence check (critical)
      2) Claim verification by URLs (critical)
    """
    # Container node for the criterion (critical)
    crit_node = evaluator.add_sequential(
        id=criterion_id,
        desc=criterion_desc,
        parent=parent_node,
        critical=True
    )

    # Step 1: URL presence check
    urls_provided = bool(url_list) and len(url_list) > 0
    evaluator.add_custom_node(
        result=urls_provided,
        id=f"{criterion_id}_urls_provided",
        desc=f"At least one verifying URL is provided for: {criterion_desc}",
        parent=crit_node,
        critical=True
    )

    # Step 2: Verification by URL(s)
    verify_leaf = evaluator.add_leaf(
        id=f"{criterion_id}_supported",
        desc=f"Claim supported by the provided URL(s): {criterion_desc}",
        parent=crit_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_text,
        node=verify_leaf,
        sources=url_list,
        additional_instruction=additional_instruction
    )


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root_node,
    extracted: SeriesExtraction
) -> None:
    """
    Construct the verification tree according to the rubric.
    """
    # Top-level critical sequential node
    series_node = evaluator.add_sequential(
        id="Series_Identification",
        desc="Identifies a Hallmark Channel original series satisfying all stated criteria and provides URLs that verify each criterion",
        parent=root_node,
        critical=True
    )

    # 1) Series name existence (critical leaf/custom node)
    series_name_provided = extracted.series_name is not None and str(extracted.series_name).strip() != ""
    evaluator.add_custom_node(
        result=series_name_provided,
        id="Series_Name",
        desc="Provides the name of a specific series (the candidate Hallmark Channel original series)",
        parent=series_node,
        critical=True
    )

    # 2) Criteria verification block (critical parallel node)
    criteria_block = evaluator.add_parallel(
        id="Criteria_Verification",
        desc="Each required criterion is satisfied AND supported by at least one reference URL that verifies it",
        parent=series_node,
        critical=True
    )

    series_name = extracted.series_name or "the series"

    # Criterion A: Hallmark Channel original
    await build_criterion_with_urls(
        evaluator=evaluator,
        parent_node=criteria_block,
        criterion_id="Hallmark_Channel_Original_With_URL",
        criterion_desc="Series is a Hallmark Channel original series (with supporting URL)",
        url_list=extracted.hallmark_original_urls,
        claim_text=f"{series_name} is a Hallmark Channel original series.",
        additional_instruction="Use the provided page(s) to confirm the network/origin that the series is produced for. The evidence should clearly indicate Hallmark Channel as the originating network or brand."
    )

    # Criterion B: Fantasy or time travel drama
    genre_label = extracted.classification or "fantasy or time travel drama"
    await build_criterion_with_urls(
        evaluator=evaluator,
        parent_node=criteria_block,
        criterion_id="Fantasy_Or_Time_Travel_Drama_With_URL",
        criterion_desc="Series is classified as a fantasy or time travel drama (with supporting URL)",
        url_list=extracted.genre_urls,
        claim_text=f"{series_name} is classified as a {genre_label}, and is a fantasy or time travel drama.",
        additional_instruction="Confirm from the page(s) that the series is described as 'fantasy' and/or 'time travel' or equivalent genre classification. Allow reasonable wording variants (e.g., 'time-travel family drama')."
    )

    # Criterion C: Filmed in Ontario (Port Perry/Uxbridge/Toronto area)
    loc_stmt = ", ".join(extracted.filming_locations) if extracted.filming_locations else "Ontario (e.g., Port Perry, Uxbridge, Toronto area)"
    await build_criterion_with_urls(
        evaluator=evaluator,
        parent_node=criteria_block,
        criterion_id="Filmed_In_Ontario_With_URL",
        criterion_desc="Series is filmed in Ontario, Canada (with supporting URL)",
        url_list=extracted.filming_urls,
        claim_text=f"{series_name} is filmed in Ontario, Canada (including locations such as {loc_stmt}).",
        additional_instruction="From the page(s), verify the filming locations are in Ontario, Canada. References to Port Perry, Uxbridge, and Toronto area are considered supportive, but any Ontario filming confirmation is acceptable."
    )

    # Criterion D: Mother-daughter creators/showrunners
    creators_list = ", ".join(extracted.creators) if extracted.creators else "a mother-daughter team"
    await build_criterion_with_urls(
        evaluator=evaluator,
        parent_node=criteria_block,
        criterion_id="Mother_Daughter_Creators_With_URL",
        criterion_desc="Series was created by a mother-daughter team serving as showrunners or creators (with supporting URL)",
        url_list=extracted.creators_urls,
        claim_text=f"{series_name} was created by {creators_list} who are a mother-daughter team serving as showrunners or creators.",
        additional_instruction="The evidence should explicitly indicate a mother-daughter relationship AND that they serve as creators or showrunners. Minor name or title variations are acceptable if the relationship and role are clear."
    )

    # Criterion E: Premiere within 2020–2024 inclusive
    prem_stmt = extracted.premiere_date or "a premiere date between January 2020 and December 2024"
    await build_criterion_with_urls(
        evaluator=evaluator,
        parent_node=criteria_block,
        criterion_id="Premiere_Within_2020_2024_With_URL",
        criterion_desc="Series premiered between Jan 2020 and Dec 2024 inclusive (with supporting URL)",
        url_list=extracted.premiere_urls,
        claim_text=f"{series_name} premiered between January 1, 2020 and December 31, 2024 inclusive (premiere date noted as {prem_stmt}).",
        additional_instruction="Use the premiere date shown on the page to check whether it falls within 2020-01-01 to 2024-12-31 inclusive. If multiple dates are shown, use the first official premiere date."
    )

    # Criterion F: At least 3 completed seasons as of Nov 2025
    seasons_stmt = extracted.seasons_completed_as_of_nov_2025 or "at least 3 completed seasons as of November 2025"
    await build_criterion_with_urls(
        evaluator=evaluator,
        parent_node=criteria_block,
        criterion_id="At_Least_3_Completed_Seasons_As_Of_Nov_2025_With_URL",
        criterion_desc="Series has at least 3 completed seasons as of Nov 2025 (with supporting URL)",
        url_list=extracted.seasons_urls,
        claim_text=f"As of November 2025, {series_name} has at least 3 completed seasons (stated as {seasons_stmt}).",
        additional_instruction="Check season counts or completion status as of November 2025 using the provided page(s). A reliable listing or official source indicating 3+ completed seasons suffices."
    )

    # Criterion G: Exactly 10 episodes per season
    eps_stmt = extracted.episodes_per_season_statement or "each season has exactly 10 episodes"
    await build_criterion_with_urls(
        evaluator=evaluator,
        parent_node=criteria_block,
        criterion_id="Exactly_10_Episodes_Per_Season_With_URL",
        criterion_desc="Each season contains exactly 10 episodes (with supporting URL)",
        url_list=extracted.episodes_urls,
        claim_text=f"Each season of {series_name} contains exactly 10 episodes (statement: {eps_stmt}).",
        additional_instruction="Verify from episode lists or season summaries that every season has exactly 10 episodes. If multiple seasons are detailed, ensure each listed season shows 10 episodes."
    )

    # Criterion H: Hallmark plus at least two other production companies
    prod_names = ", ".join(extracted.production_companies) if extracted.production_companies else "Hallmark Channel and other production companies"
    await build_criterion_with_urls(
        evaluator=evaluator,
        parent_node=criteria_block,
        criterion_id="Hallmark_Plus_At_Least_Two_Other_Production_Companies_With_URL",
        criterion_desc="Produced by Hallmark Channel with at least two other production companies (with supporting URL)",
        url_list=extracted.production_urls,
        claim_text=f"{series_name} is produced by Hallmark Channel in collaboration with at least two other production companies (named: {prod_names}).",
        additional_instruction="Use the provided page(s) to identify listed production companies. Confirm Hallmark Channel involvement plus at least two other distinct companies."
    )

    # Record some helpful custom info for debugging/tracing
    evaluator.add_custom_info(
        info={
            "series_name": extracted.series_name,
            "url_counts": {
                "hallmark_original_urls": len(extracted.hallmark_original_urls),
                "genre_urls": len(extracted.genre_urls),
                "filming_urls": len(extracted.filming_urls),
                "creators_urls": len(extracted.creators_urls),
                "premiere_urls": len(extracted.premiere_urls),
                "seasons_urls": len(extracted.seasons_urls),
                "episodes_urls": len(extracted.episodes_urls),
                "production_urls": len(extracted.production_urls),
            }
        },
        info_type="extraction_summary",
        info_name="series_extraction_summary"
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
    Evaluate an answer for the Hallmark Channel series verification task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # overall top-level; our main verification subtree is sequential
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

    # Extract structured information from the agent's answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_series(),
        template_class=SeriesExtraction,
        extraction_name="series_candidate"
    )

    # Build the verification tree and execute checks
    await build_verification_tree(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()