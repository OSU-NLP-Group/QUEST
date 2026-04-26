import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "bob_huggins_akron_record"
TASK_DESCRIPTION = (
    "What was Bob Huggins' overall win-loss record during his tenure as head basketball coach at the University of Akron, "
    "and how many seasons did he coach there?"
)

# Ground truth info (for logging context only; not directly used for scoring)
GROUND_TRUTH = {
    "overall_record": "97–46",
    "tenure_years": "1984–1989",
    "start_year": "1984",
    "end_year": "1989",
    "seasons_coached": "5",
    "first_season_record": "12–14",
    "four_consecutive_21_win_seasons": True,
}


# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class HugginsAkronExtraction(BaseModel):
    """
    Information that the answer explicitly states about Bob Huggins' Akron tenure.
    All fields should reflect exactly what the answer claims (do not infer).
    """
    # Overall head-coaching record at Akron, such as "97–46" or "97-46" or "97 wins and 46 losses"
    overall_record: Optional[str] = None

    # Tenure years - both split and combined textual form are allowed
    tenure_start_year: Optional[str] = None
    tenure_end_year: Optional[str] = None
    tenure_years_text: Optional[str] = None  # e.g., "1984–1989", "1984-1989", "from 1984 to 1989"

    # Seasons coached, as stated in the answer. Keep as text to allow "five" or "5".
    seasons_coached: Optional[str] = None

    # First season (1984–85) record at Akron, e.g., "12–14" or "12-14"
    first_season_record: Optional[str] = None

    # Whether the answer explicitly claims: "after the first season, he had four consecutive seasons with at least 21 wins each".
    # true: explicitly claimed; false: explicitly denied; null: not mentioned.
    four_consecutive_21_win_seasons: Optional[bool] = None

    # All URLs cited in the answer for supporting these facts (extract as-is)
    sources: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_huggins_akron() -> str:
    return """
    Extract exactly what the answer explicitly states about Bob Huggins' tenure at the University of Akron.
    Do not infer or calculate; only capture what is clearly stated in the answer text itself.

    Return the following fields:
    - overall_record: The overall Akron head-coaching record string exactly as stated (e.g., "97–46", "97-46", or "97 wins and 46 losses"). If absent, return null.
    - tenure_start_year: The 4-digit starting year of his Akron head-coach tenure if explicitly stated. If absent, return null.
    - tenure_end_year: The 4-digit ending year of his Akron head-coach tenure if explicitly stated. If absent, return null.
    - tenure_years_text: The text span that states his Akron tenure years range (e.g., "1984–1989", "1984-1989", or "from 1984 to 1989"). If absent, return null.
    - seasons_coached: The number of seasons he coached at Akron as stated (allow formats like "five" or "5"). If absent, return null.
    - first_season_record: The record of his first season at Akron (1984–85) as stated (e.g., "12–14", "12-14"). If absent, return null.
    - four_consecutive_21_win_seasons: true if the answer explicitly claims that after his first season, he had four consecutive seasons with at least 21 wins each; false if it explicitly claims the opposite; null if not mentioned.
    - sources: An array of all URLs that the answer cites as evidence for these facts (only include valid URLs explicitly present in the answer; include markdown link targets if present).

    Important:
    - Follow SPECIAL RULES FOR URL SOURCES EXTRACTION: extract only actual URLs present in the answer.
    - If any field is not explicitly present in the answer, return null for that field.
    """


# -----------------------------------------------------------------------------
# Verification helper
# -----------------------------------------------------------------------------
def _placeholder(value: Optional[str], placeholder: str = "<missing>") -> str:
    return value.strip() if isinstance(value, str) and value.strip() else placeholder


def _years_text(ex: HugginsAkronExtraction) -> str:
    """
    Build a canonical textual years string from extracted fields when possible.
    Prefer tenure_years_text; otherwise use "start to end" if both present; else "<missing>".
    """
    if ex.tenure_years_text and ex.tenure_years_text.strip():
        return ex.tenure_years_text.strip()
    if ex.tenure_start_year and ex.tenure_end_year:
        return f"{ex.tenure_start_year.strip()} to {ex.tenure_end_year.strip()}"
    return "<missing>"


# -----------------------------------------------------------------------------
# Build verification tree and run checks
# -----------------------------------------------------------------------------
async def verify_bob_huggins_akron(
    evaluator: Evaluator,
    parent_node,
    extracted: HugginsAkronExtraction,
) -> None:
    """
    Build the verification tree based on the rubric and run evidence-based checks.
    """
    # Top-level rubric node (critical, parallel aggregation)
    main_node = evaluator.add_parallel(
        id="Bob_Huggins_Akron_Record",
        desc="Verify the answer reports Bob Huggins' Akron coaching tenure details and record per the stated constraints, with verifiable sourcing.",
        parent=parent_node,
        critical=True,
    )

    # Common sources list extracted from the answer
    sources_list: List[str] = extracted.sources or []

    # Prepare claims and nodes for batch verification
    claims_and_sources = []

    # 1) Overall Win-Loss Record (expects 97–46 if the answer is correct)
    overall_record_value = _placeholder(extracted.overall_record)
    overall_node = evaluator.add_leaf(
        id="Overall_Win_Loss_Record",
        desc="States Bob Huggins' overall Akron head-coaching record as 97 wins and 46 losses (97–46).",
        parent=main_node,
        critical=True,
    )
    overall_claim = (
        f"Bob Huggins' overall win-loss record as head coach at the University of Akron was {overall_record_value}."
    )
    overall_instruction = (
        "Use ONLY the provided URLs to verify this exact overall record for his Akron head-coaching tenure. "
        "If no source URLs are provided or if the answer did not explicitly state an overall record (shown as '<missing>'), "
        "you must mark this as Incorrect. Treat '97–46', '97-46', and '97 wins and 46 losses' as equivalent."
    )
    claims_and_sources.append((overall_claim, sources_list, overall_node, overall_instruction))

    # 2) Tenure Years (expects 1984–1989)
    years_text_value = _years_text(extracted)
    tenure_node = evaluator.add_leaf(
        id="Tenure_Years",
        desc="States Bob Huggins served as head coach at the University of Akron from 1984 to 1989.",
        parent=main_node,
        critical=True,
    )
    tenure_claim = (
        f"Bob Huggins served as head coach at the University of Akron from {years_text_value}."
    )
    tenure_instruction = (
        "Verify using ONLY the provided URLs that his Akron head-coach tenure years correspond to 1984–1989 "
        "(accept formatting variants like '1984-1989' or 'from 1984 to 1989'). "
        "If no URLs are provided or the answer did not explicitly state the years (shown as '<missing>'), mark as Incorrect."
    )
    claims_and_sources.append((tenure_claim, sources_list, tenure_node, tenure_instruction))

    # 3) Seasons Coached (expects five)
    seasons_value = _placeholder(extracted.seasons_coached)
    seasons_node = evaluator.add_leaf(
        id="Seasons_Coached",
        desc="States his Akron tenure lasted five seasons.",
        parent=main_node,
        critical=True,
    )
    seasons_claim = f"Bob Huggins coached {seasons_value} seasons at the University of Akron."
    seasons_instruction = (
        "Verify using ONLY the provided URLs that he coached five seasons at Akron. "
        "Treat 'five' and '5' as equivalent. "
        "If no URLs are provided or the answer did not explicitly state the number of seasons (shown as '<missing>'), mark as Incorrect."
    )
    claims_and_sources.append((seasons_claim, sources_list, seasons_node, seasons_instruction))

    # 4) First Season Record (expects 12–14)
    first_season_value = _placeholder(extracted.first_season_record)
    first_season_node = evaluator.add_leaf(
        id="First_Season_Record",
        desc="States his first season record at Akron was 12–14.",
        parent=main_node,
        critical=True,
    )
    first_season_claim = (
        f"Bob Huggins' first season (1984–85) record at the University of Akron was {first_season_value}."
    )
    first_season_instruction = (
        "Verify using ONLY the provided URLs that his first season (1984–85) record at Akron was 12–14 "
        "(accept '12-14' vs '12–14'). "
        "If no URLs are provided or the answer did not explicitly state this record (shown as '<missing>'), mark as Incorrect."
    )
    claims_and_sources.append((first_season_claim, sources_list, first_season_node, first_season_instruction))

    # 5) Four Consecutive 21+ Win Seasons (expects true)
    four_21_text = "four consecutive seasons with at least 21 wins each"
    four21_node = evaluator.add_leaf(
        id="Four_Consecutive_21_Win_Seasons",
        desc="States that after the first season, he had four consecutive seasons with at least 21 wins each.",
        parent=main_node,
        critical=True,
    )
    # We state the standardized target claim. Additional instruction enforces that the answer must have claimed this.
    four21_claim = f"After his first season at Akron, Bob Huggins had {four_21_text}."
    # Reflect whether the answer claimed this; if not explicitly true -> must fail
    claimed_flag = extracted.four_consecutive_21_win_seasons is True
    four21_instruction = (
        "Verify using ONLY the provided URLs whether this statement is correct using season-by-season records. "
        "Additionally, PASS only if the answer explicitly claimed this fact; if the answer did not explicitly claim it, FAIL. "
        f"In this case, 'explicitly claimed by the answer' = {claimed_flag}. "
        "If no URLs are provided, mark as Incorrect."
    )
    claims_and_sources.append((four21_claim, sources_list, four21_node, four21_instruction))

    # 6) Verifiable & Reliable Sources present
    sources_node = evaluator.add_leaf(
        id="Verifiable_Reliable_Sources",
        desc="Provides citations/URLs to reliable sources that document and support the stated Akron tenure and record details.",
        parent=main_node,
        critical=True,
    )
    # We phrase reliability + coverage in a practical way: at least one reliable site and the set of URLs allows verification
    sources_claim = (
        "The answer provides at least one reliable source URL (e.g., official university/athletics site, NCAA, "
        "Sports Reference, or a reputable news organization) that documents Bob Huggins' Akron coaching tenure/records, "
        "so the stated details can be verified."
    )
    sources_instruction = (
        "Evaluate reliability using the domains and page content. PASS if the provided URLs include at least one "
        "official or reputable site relevant to Bob Huggins' Akron coaching history. It's acceptable if different URLs "
        "cover different sub-details. If there are no URLs in the answer, or the URLs are clearly unreliable/irrelevant, FAIL."
    )
    claims_and_sources.append((sources_claim, sources_list, sources_node, sources_instruction))

    # Execute all verifications in parallel
    await evaluator.batch_verify(claims_and_sources)


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the Bob Huggins Akron record task.
    """
    # Initialize evaluator and root
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
        prompt=prompt_extract_huggins_akron(),
        template_class=HugginsAkronExtraction,
        extraction_name="huggins_akron_extraction",
    )

    # Add ground truth to summary for reference
    evaluator.add_ground_truth(
        {
            "expected_overall_record": GROUND_TRUTH["overall_record"],
            "expected_tenure_years": GROUND_TRUTH["tenure_years"],
            "expected_seasons_coached": GROUND_TRUTH["seasons_coached"],
            "expected_first_season_record": GROUND_TRUTH["first_season_record"],
            "expected_four_consecutive_21_win_seasons": GROUND_TRUTH["four_consecutive_21_win_seasons"],
        },
        gt_type="ground_truth_huggins_akron",
    )

    # Build verification tree and perform checks
    await verify_bob_huggins_akron(evaluator, root, extracted)

    # Return final structured summary
    return evaluator.get_summary()