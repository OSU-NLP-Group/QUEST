import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "miss_universe_2025_info"
TASK_DESCRIPTION = (
    "You are assisting a fashion academy that is preparing educational materials about international beauty pageants. "
    "For Miss Universe 2025, which took place in Thailand, compile the following key information for their reference guide:\n"
    "1. Contestant eligibility requirements: What was the minimum age requirement, and what educational or professional "
    "qualifications were required?\n"
    "2. Competition schedule: What were the specific dates for the Preliminary Competition, the Swimsuit Fashion Show, "
    "the Close Door Interview, and the Final Competition?\n"
    "3. Venue details: Where exactly was the Final Competition held (provide complete venue name and location), and where "
    "was the Swimsuit Fashion Show held?\n"
    "4. Competition format: How many semifinalists were selected from the preliminary round, and describe how contestants "
    "progressed from the swimsuit segment to the Top 12, and from the evening gown segment to the Top 5."
)

# Ground truth expectation snapshot (for reporting/reference only)
GROUND_TRUTH_EXPECTATIONS = {
    "eligibility": {
        "age_minimum": "18 years or older",
        "education_or_profession": "At least a university degree holder or a working professional",
    },
    "key_dates": {
        "preliminary": "November 19, 2025",
        "swimsuit": "November 14, 2025",
        "interview": "November 15, 2025",
        "final": "November 21, 2025",
    },
    "venues": {
        "final": "Impact Challenger Hall in Pak Kret, Nonthaburi, Thailand",
        "swimsuit": "Aquaverse Pattaya",
    },
    "progression": {
        "semifinalists_count": "30",
        "top_12_after_swimsuit": True,
        "top_5_after_gown": True,
    },
}


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class EligibilityExtraction(BaseModel):
    age_requirement: Optional[str] = None
    education_professional_requirement: Optional[str] = None
    age_sources: List[str] = Field(default_factory=list)
    edu_prof_sources: List[str] = Field(default_factory=list)


class ScheduleExtraction(BaseModel):
    preliminary_date: Optional[str] = None
    swimsuit_show_date: Optional[str] = None
    interview_date: Optional[str] = None
    final_date: Optional[str] = None
    preliminary_sources: List[str] = Field(default_factory=list)
    swimsuit_sources: List[str] = Field(default_factory=list)
    interview_sources: List[str] = Field(default_factory=list)
    final_sources: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    final_venue_name: Optional[str] = None
    final_venue_location: Optional[str] = None
    final_venue_sources: List[str] = Field(default_factory=list)
    swimsuit_venue_name: Optional[str] = None
    swimsuit_venue_location: Optional[str] = None
    swimsuit_venue_sources: List[str] = Field(default_factory=list)


class FormatExtraction(BaseModel):
    semifinalists_count: Optional[str] = None
    top12_progression: Optional[str] = None
    top5_progression: Optional[str] = None
    semifinalists_sources: List[str] = Field(default_factory=list)
    top12_sources: List[str] = Field(default_factory=list)
    top5_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_eligibility() -> str:
    return """
    From the provided answer text, extract the specific contestant eligibility requirements for Miss Universe 2025.
    Return the following fields:
    - age_requirement: The minimum age requirement as explicitly stated (e.g., "18 years or older", "at least 18").
    - education_professional_requirement: The educational or professional qualification as explicitly stated 
      (e.g., "at least a university degree holder or a working professional").
    - age_sources: An array of URLs explicitly provided in the answer that support the age requirement.
    - edu_prof_sources: An array of URLs explicitly provided in the answer that support the education/professional requirement.
    If any field is not present in the answer, set it to null (for strings) or [] (for url arrays).
    Extract only URLs that are explicitly present in the answer.
    """


def prompt_extract_schedule() -> str:
    return """
    From the provided answer text, extract the key competition dates for Miss Universe 2025.
    Return the following fields:
    - preliminary_date: The date of the Preliminary Competition (as stated, e.g., "November 19, 2025").
    - swimsuit_show_date: The date of the Swimsuit Fashion Show (e.g., "November 14, 2025").
    - interview_date: The date of the Close Door (or Closed-Door) Interview (e.g., "November 15, 2025").
    - final_date: The date of the Final Competition (e.g., "November 21, 2025").
    - preliminary_sources: URLs supporting the preliminary date (array).
    - swimsuit_sources: URLs supporting the swimsuit show date (array).
    - interview_sources: URLs supporting the interview date (array).
    - final_sources: URLs supporting the final date (array).
    If any field is not present in the answer, set it to null (for strings) or [] (for url arrays).
    Extract only URLs that are explicitly present in the answer.
    """


def prompt_extract_venues() -> str:
    return """
    From the provided answer text, extract venue information for Miss Universe 2025.
    Return the following fields:
    - final_venue_name: Full venue name where the Final Competition was held (e.g., "IMPACT Challenger Hall").
    - final_venue_location: The city/province/country location (e.g., "Pak Kret, Nonthaburi, Thailand").
    - final_venue_sources: URLs supporting the final venue (array).
    - swimsuit_venue_name: Full venue name where the Swimsuit Fashion Show was held (e.g., "Aquaverse Pattaya").
    - swimsuit_venue_location: The city/province/country location if present (string or null).
    - swimsuit_venue_sources: URLs supporting the swimsuit venue (array).
    If any field is not present in the answer, set it to null (for strings) or [] (for url arrays).
    Extract only URLs that are explicitly present in the answer.
    """


def prompt_extract_format() -> str:
    return """
    From the provided answer text, extract the competition progression format for Miss Universe 2025.
    Return the following fields:
    - semifinalists_count: The number of semifinalists selected from the preliminary round (string, e.g., "30").
    - top12_progression: A brief phrase describing that Top 12 were selected after the swimsuit segment.
    - top5_progression: A brief phrase describing that Top 5 were selected after the evening gown segment.
    - semifinalists_sources: URLs supporting the semifinalists count (array).
    - top12_sources: URLs supporting the Top 12 after swimsuit statement (array).
    - top5_sources: URLs supporting the Top 5 after evening gown statement (array).
    If any field is not present in the answer, set it to null (for strings) or [] (for url arrays).
    Extract only URLs that are explicitly present in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification helpers (category subtrees)                                    #
# --------------------------------------------------------------------------- #
async def verify_eligibility(
    evaluator: Evaluator,
    parent_node,
    extracted: EligibilityExtraction,
) -> None:
    cat_node = evaluator.add_parallel(
        id="Contestant_Eligibility",
        desc="Contestant eligibility requirements for Miss Universe 2025",
        parent=parent_node,
        critical=True,
    )

    # Leaves
    age_node = evaluator.add_leaf(
        id="Age_Requirement",
        desc="Minimum age requirement is 18 years or older",
        parent=cat_node,
        critical=True,
    )
    edu_node = evaluator.add_leaf(
        id="Education_Professional_Requirement",
        desc="Must be at least a university degree holder or working professional",
        parent=cat_node,
        critical=True,
    )

    claims = [
        (
            "Miss Universe 2025 minimum age requirement is 18 years or older.",
            extracted.age_sources if extracted and extracted.age_sources else None,
            age_node,
            "Accept equivalent phrasings such as '18+', 'at least 18', or '18 and above'. "
            "Ensure the page context refers to Miss Universe 2025 eligibility and not a different year or pageant.",
        ),
        (
            "Miss Universe 2025 requires contestants to be at least a university degree holder or a working professional.",
            extracted.edu_prof_sources if extracted and extracted.edu_prof_sources else None,
            edu_node,
            "Accept equivalent phrasings such as 'must possess a university degree' or 'must be a working professional'. "
            "If the page explicitly states no degree/professional requirement, then the claim is not supported.",
        ),
    ]
    await evaluator.batch_verify(claims)


async def verify_schedule(
    evaluator: Evaluator,
    parent_node,
    extracted: ScheduleExtraction,
) -> None:
    cat_node = evaluator.add_parallel(
        id="Competition_Dates",
        desc="Key dates for major competition events",
        parent=parent_node,
        critical=True,
    )

    prelim_node = evaluator.add_leaf(
        id="Preliminary_Date",
        desc="Preliminary Competition held on November 19, 2025",
        parent=cat_node,
        critical=True,
    )
    swim_node = evaluator.add_leaf(
        id="Swimsuit_Show_Date",
        desc="Swimsuit Fashion Show held on November 14, 2025",
        parent=cat_node,
        critical=True,
    )
    interview_node = evaluator.add_leaf(
        id="Interview_Date",
        desc="Close Door Interview held on November 15, 2025",
        parent=cat_node,
        critical=True,
    )
    final_node = evaluator.add_leaf(
        id="Final_Date",
        desc="Final Competition held on November 21, 2025",
        parent=cat_node,
        critical=True,
    )

    claims = [
        (
            "The Preliminary Competition took place on November 19, 2025.",
            extracted.preliminary_sources if extracted and extracted.preliminary_sources else None,
            prelim_node,
            "Allow minor variations in date formatting and consider Thailand local time (Bangkok).",
        ),
        (
            "The Swimsuit Fashion Show took place on November 14, 2025.",
            extracted.swimsuit_sources if extracted and extracted.swimsuit_sources else None,
            swim_node,
            "Allow minor variations in date formatting and consider Thailand local time.",
        ),
        (
            "The Close Door (Closed-Door) Interview took place on November 15, 2025.",
            extracted.interview_sources if extracted and extracted.interview_sources else None,
            interview_node,
            "Treat 'Close Door' and 'Closed-Door' as equivalent phrasing. Consider Thailand local time.",
        ),
        (
            "The Final Competition took place on November 21, 2025.",
            extracted.final_sources if extracted and extracted.final_sources else None,
            final_node,
            "Allow 'Final' or 'Final Competition' phrasing variations. Consider Thailand local time.",
        ),
    ]
    await evaluator.batch_verify(claims)


async def verify_venues(
    evaluator: Evaluator,
    parent_node,
    extracted: VenuesExtraction,
) -> None:
    cat_node = evaluator.add_parallel(
        id="Competition_Venues",
        desc="Venue locations for key competition events",
        parent=parent_node,
        critical=True,
    )

    final_venue_node = evaluator.add_leaf(
        id="Final_Venue",
        desc="Final competition venue at Impact Challenger Hall in Pak Kret, Nonthaburi, Thailand",
        parent=cat_node,
        critical=True,
    )
    swimsuit_venue_node = evaluator.add_leaf(
        id="Swimsuit_Venue",
        desc="Swimsuit Fashion Show venue at Aquaverse Pattaya",
        parent=cat_node,
        critical=True,
    )

    final_claim = (
        "The Miss Universe 2025 Final Competition was held at IMPACT Challenger Hall in Pak Kret, Nonthaburi, Thailand."
    )
    swimsuit_claim = (
        "The Swimsuit Fashion Show was held at Aquaverse Pattaya."
    )

    claims = [
        (
            final_claim,
            extracted.final_venue_sources if extracted and extracted.final_venue_sources else None,
            final_venue_node,
            "Accept minor name variants like 'IMPACT Challenger Hall' (case-insensitive). "
            "Location phrasing can vary but should clearly indicate Pak Kret, Nonthaburi, Thailand.",
        ),
        (
            swimsuit_claim,
            extracted.swimsuit_venue_sources if extracted and extracted.swimsuit_venue_sources else None,
            swimsuit_venue_node,
            "Accept branding variants such as 'Columbia Pictures Aquaverse' for Aquaverse Pattaya if clearly the same venue.",
        ),
    ]
    await evaluator.batch_verify(claims)


async def verify_format(
    evaluator: Evaluator,
    parent_node,
    extracted: FormatExtraction,
) -> None:
    cat_node = evaluator.add_parallel(
        id="Progression_Format",
        desc="Competition round progression format from preliminary to finals",
        parent=parent_node,
        critical=True,
    )

    semi_node = evaluator.add_leaf(
        id="Semifinalists_Count",
        desc="30 semifinalists selected from preliminary competition",
        parent=cat_node,
        critical=True,
    )
    top12_node = evaluator.add_leaf(
        id="Top_12_Selection",
        desc="Top 12 contestants selected after swimsuit segment",
        parent=cat_node,
        critical=True,
    )
    top5_node = evaluator.add_leaf(
        id="Top_5_Selection",
        desc="Top 5 contestants selected after evening gown segment",
        parent=cat_node,
        critical=True,
    )

    claims = [
        (
            "30 semifinalists were selected from the preliminary competition.",
            extracted.semifinalists_sources if extracted and extracted.semifinalists_sources else None,
            semi_node,
            "Allow equivalent expressions like 'Top 30' as the semifinalists count.",
        ),
        (
            "After the swimsuit segment, the Top 12 contestants were selected.",
            extracted.top12_sources if extracted and extracted.top12_sources else None,
            top12_node,
            "Accept equivalent phrasing indicating that the swimsuit segment determines or narrows to the Top 12.",
        ),
        (
            "After the evening gown segment, the Top 5 contestants were selected.",
            extracted.top5_sources if extracted and extracted.top5_sources else None,
            top5_node,
            "Accept equivalent phrasing indicating that the evening gown segment determines or narrows to the Top 5.",
        ),
    ]
    await evaluator.batch_verify(claims)


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
) -> Dict[str, Any]:
    # Initialize evaluator (root node is non-critical aggregator)
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

    # Extract all structured info in parallel
    elig_task = evaluator.extract(
        prompt=prompt_extract_eligibility(),
        template_class=EligibilityExtraction,
        extraction_name="eligibility_info",
    )
    schedule_task = evaluator.extract(
        prompt=prompt_extract_schedule(),
        template_class=ScheduleExtraction,
        extraction_name="schedule_info",
    )
    venues_task = evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_info",
    )
    format_task = evaluator.extract(
        prompt=prompt_extract_format(),
        template_class=FormatExtraction,
        extraction_name="format_info",
    )
    eligibility, schedule, venues, format_info = await asyncio.gather(
        elig_task, schedule_task, venues_task, format_task
    )

    # Add ground truth snapshot for reference
    evaluator.add_ground_truth(
        {
            "expected": GROUND_TRUTH_EXPECTATIONS,
            "notes": "These values represent the expected claims to be verified against sources cited in the answer.",
        },
        gt_type="ground_truth_expectations",
    )

    # Build rubric root node (non-critical) under the evaluator root
    rubric_root = evaluator.add_parallel(
        id="Miss_Universe_2025_Information",
        desc="Complete information requirements about Miss Universe 2025 competition",
        parent=root,
        critical=False,
    )

    # Verify each category subtree
    await verify_eligibility(evaluator, rubric_root, eligibility)
    await verify_schedule(evaluator, rubric_root, schedule)
    await verify_venues(evaluator, rubric_root, venues)
    await verify_format(evaluator, rubric_root, format_info)

    # Return structured summary
    return evaluator.get_summary()