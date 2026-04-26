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
TASK_ID = "boston_marathon_2026_info"
TASK_DESCRIPTION = (
    "I am a 42-year-old male runner interested in qualifying for the 2026 Boston Marathon. "
    "Please provide the following information: (1) When does the 2026 Boston Marathon take place? "
    "(2) What is the qualifying time standard I need to meet based on my age and gender? "
    "(3) Include a reference URL from the Boston Athletic Association (BAA) official website that confirms this information."
)

EXPECTED_RACE_DATE_TEXT = "April 20, 2026"  # Patriots' Day 2026


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class MarathonInfoExtraction(BaseModel):
    """
    Extract fields directly from the agent answer:
    - race_date_text: the stated 2026 Boston Marathon race date (as text in the answer)
    - qualifying_time_text: the stated qualifying time for a male in age group 40–44 (as text)
    - age_group_text: the age/gender group string the answer claims to use (e.g., 'Men 40–44', 'Male 40-44')
    - referenced_urls: all URLs cited in the answer (we will filter to BAA domain later)
    """
    race_date_text: Optional[str] = None
    qualifying_time_text: Optional[str] = None
    age_group_text: Optional[str] = None
    referenced_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_marathon_info() -> str:
    return """
    Extract the requested fields from the answer text about the 2026 Boston Marathon:

    Required fields:
    1) race_date_text: The date the answer states for when the 2026 Boston Marathon takes place. Return the exact text span mentioned (e.g., "April 20, 2026"). If not stated, return null.
    2) qualifying_time_text: The qualifying time the answer states for a male runner in the 40–44 age group. Return the exact string (e.g., "3:10:00"). If not stated, return null.
    3) age_group_text: The age/gender group that the answer says applies to the runner (e.g., "Men 40–44", "Male 40-44"). If not stated, return null.
    4) referenced_urls: A list of all URLs cited in the answer. Include only valid URLs actually shown in the answer (plain links or markdown). If none, return an empty list.

    Notes:
    - Do not invent or infer values not present in the answer text.
    - Preserve the qualifying time string as-is from the answer (do not reformat).
    - For referenced_urls, extract all URLs mentioned, preserving their full string (prepend http:// if missing).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def filter_baa_urls(urls: List[str]) -> List[str]:
    """Return only URLs that appear to be from the official BAA website."""
    baa_list = []
    for u in urls:
        if not u:
            continue
        lu = u.strip().lower()
        if "baa.org" in lu:
            # ensure protocol for safety
            if lu.startswith("http://") or lu.startswith("https://"):
                baa_list.append(u)
            else:
                baa_list.append("https://" + u.lstrip("/"))
    return baa_list


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extraction: MarathonInfoExtraction) -> None:
    """
    Construct the verification tree and perform verifications according to the rubric.
    JSON rubric (adapted to enforce single-step leaves where needed):

    Boston_Marathon_2026_Information (critical, parallel)
    ├── Race_Date (critical, leaf): Answer states the 2026 race date as April 20, 2026.
    ├── Qualifying_Standard_For_Runner (critical, parallel)
    │   ├── Age_Group_Application (critical, leaf): Answer uses age-on-race-day rule and applies male 40–44 for a 42-year-old.
    │   ├── Qualifying_Time_Correct_For_Group (critical, leaf): Time matches BAA standard for male 40–44 (verified via BAA URL).
    │   └── Qualifying_Time_Format (critical, leaf): Time is in HH:MM:SS format (allow reasonable variants).
    └── Reference_URL (critical, parallel)
        ├── BAA_URL_Present (critical, leaf/custom): At least one BAA URL is provided.
        ├── Race_Date_Supported_By_BAA (critical, leaf): BAA URL(s) confirm April 20, 2026 date.
        └── Qualifying_Time_Supported_By_BAA (critical, leaf): BAA URL(s) confirm the stated male 40–44 standard.
    """
    # Top-level critical node
    top = evaluator.add_parallel(
        id="Boston_Marathon_2026_Information",
        desc="Verify the 2026 Boston Marathon race date, the qualifying standard for male 40–44, and an official BAA reference URL supporting the information.",
        parent=evaluator.root,
        critical=True
    )

    # Prepare fields
    race_date_text = (extraction.race_date_text or "").strip()
    qualifying_time_text = (extraction.qualifying_time_text or "").strip()
    age_group_text = (extraction.age_group_text or "").strip()
    baa_urls = filter_baa_urls(extraction.referenced_urls or [])

    # ---------------------- Race Date leaf --------------------------------
    race_date_leaf = evaluator.add_leaf(
        id="Race_Date",
        desc="States the 2026 Boston Marathon race date as April 20, 2026.",
        parent=top,
        critical=True
    )
    race_date_claim = (
        "The answer explicitly states that the 2026 Boston Marathon race date is April 20, 2026 "
        "(accept equivalent phrasings like 'Monday, April 20, 2026')."
    )
    await evaluator.verify(
        claim=race_date_claim,
        node=race_date_leaf,
        additional_instruction="Check only the answer text for whether it states 'April 20, 2026' as the 2026 Boston Marathon date."
    )

    # ---------------------- Qualifying Standard group ---------------------
    qual_group = evaluator.add_parallel(
        id="Qualifying_Standard_For_Runner",
        desc="Provides the correct qualifying time standard applicable to a male runner in the 40–44 age group, using the age-on-race-day rule.",
        parent=top,
        critical=True
    )

    # Age group application
    age_group_leaf = evaluator.add_leaf(
        id="Age_Group_Application",
        desc="Uses age on race day and applies the male 40–44 age group for a 42-year-old on April 20, 2026.",
        parent=qual_group,
        critical=True
    )
    age_group_claim = (
        "Based on a 42-year-old male runner and the race occurring on April 20, 2026, "
        "the answer determines (or clearly uses) the age group by age on race day and applies the male 40–44 group."
    )
    await evaluator.verify(
        claim=age_group_claim,
        node=age_group_leaf,
        additional_instruction=(
            "Evaluate the answer text for correct application of the 'age on race day' rule. "
            "Minor wording variations are fine if the logic used clearly corresponds to 'age on race day' "
            "and the male 40–44 group is applied (e.g., 'Men 40–44', 'Male 40-44')."
        )
    )

    # Qualifying time is correct for group (verified by BAA URLs)
    time_correct_leaf = evaluator.add_leaf(
        id="Qualifying_Time_Correct_For_Group",
        desc="Gives a qualifying time value that matches the official BAA qualifying standard for male 40–44 for the 2026 Boston Marathon.",
        parent=qual_group,
        critical=True
    )
    time_correct_claim = (
        f"For the 2026 Boston Marathon, the official BAA qualifying standard for male runners aged 40–44 is '{qualifying_time_text}'."
    )
    await evaluator.verify(
        claim=time_correct_claim,
        node=time_correct_leaf,
        sources=baa_urls,
        additional_instruction=(
            "Check the BAA page(s) for the qualifying standards table or text. "
            "Allow reasonable textual variants (e.g., '3hrs 10min 00sec' equivalent to '3:10:00'). "
            "The claim should match the standard for the male 40–44 category. "
            "If the provided URLs are irrelevant or not from the official BAA site, mark as not supported."
        )
    )

    # Qualifying time format (HH:MM:SS)
    time_format_leaf = evaluator.add_leaf(
        id="Qualifying_Time_Format",
        desc="Formats the qualifying time in hours:minutes:seconds (HH:MM:SS) format.",
        parent=qual_group,
        critical=True
    )
    time_format_claim = (
        f"The qualifying time value '{qualifying_time_text}' is presented in HH:MM:SS format "
        "(minor variations like omitting a leading zero are acceptable, e.g., '3:10:00' or '03:10:00')."
    )
    await evaluator.verify(
        claim=time_format_claim,
        node=time_format_leaf,
        additional_instruction=(
            "Judge the formatting of the time string only. Accept common, unambiguous equivalents "
            "such as '3:10:00' or '03:10:00'. Do not accept vague text (e.g., '3 hours and 10 minutes') unless it is also given as HH:MM:SS."
        )
    )

    # ---------------------- Reference URL group ---------------------------
    ref_group = evaluator.add_parallel(
        id="Reference_URL",
        desc="Provides a URL on the official BAA website (baa.org) that supports the stated race date and qualifying time.",
        parent=top,
        critical=True
    )

    # At least one BAA URL provided (existence check)
    baa_present_node = evaluator.add_custom_node(
        result=len(baa_urls) > 0,
        id="BAA_URL_Present",
        desc="At least one URL from the official BAA website (baa.org) is provided in the answer.",
        parent=ref_group,
        critical=True
    )

    # Race date supported by BAA
    ref_date_leaf = evaluator.add_leaf(
        id="Race_Date_Supported_By_BAA",
        desc="BAA URL(s) confirm the 2026 Boston Marathon race date April 20, 2026.",
        parent=ref_group,
        critical=True
    )
    ref_date_claim = (
        "The official BAA page(s) confirm that the 2026 Boston Marathon takes place on April 20, 2026."
    )
    await evaluator.verify(
        claim=ref_date_claim,
        node=ref_date_leaf,
        sources=baa_urls,
        additional_instruction=(
            "Look for event date information for the 2026 Boston Marathon on the BAA page(s). "
            "If the URLs are invalid, irrelevant, or do not explicitly confirm the date, mark as not supported."
        )
    )

    # Qualifying time supported by BAA
    ref_time_leaf = evaluator.add_leaf(
        id="Qualifying_Time_Supported_By_BAA",
        desc="BAA URL(s) confirm the stated qualifying time standard for male 40–44.",
        parent=ref_group,
        critical=True
    )
    ref_time_claim = (
        f"The official BAA page(s) state that the qualifying standard for the male 40–44 category is '{qualifying_time_text}'."
    )
    await evaluator.verify(
        claim=ref_time_claim,
        node=ref_time_leaf,
        sources=baa_urls,
        additional_instruction=(
            "Find the qualifying standards table or description for the Boston Marathon on the BAA page(s). "
            "Confirm that the male 40–44 standard matches the stated time. Allow for equivalent formatting."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the Boston Marathon 2026 information task and return a structured summary.
    """
    # Initialize evaluator with a parallel root (root remains non-critical)
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
        default_model=model
    )

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_marathon_info(),
        template_class=MarathonInfoExtraction,
        extraction_name="marathon_info_extraction"
    )

    # Add helpful ground-truth context (for reporting only; verification relies on BAA URLs)
    evaluator.add_ground_truth(
        {
            "expected_race_date_text": EXPECTED_RACE_DATE_TEXT,
            "target_runner_profile": {"age": 42, "gender": "male"},
            "target_age_group": "male 40–44 (age on race day)",
            "notes": "Official confirmation must come from baa.org URLs cited in the answer."
        },
        gt_type="reference_context"
    )

    # Build verification tree and run verifications
    await build_verification_tree(evaluator, extraction)

    # Return full summary
    return evaluator.get_summary()