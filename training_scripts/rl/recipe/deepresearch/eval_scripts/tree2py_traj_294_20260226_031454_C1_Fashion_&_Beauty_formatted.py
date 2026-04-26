import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pfw_2025_milestone"
TASK_DESCRIPTION = (
    "In September 2025, a supermodel closed a show at Paris Fashion Week for the first time "
    "in her 33-year career. Which model achieved this milestone, and for which designer's "
    "Spring/Summer 2026 show did she close the runway?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MilestoneExtraction(BaseModel):
    """
    Structured extraction of the answer content relevant to the milestone.
    Prefer strings for robustness; URLs must come from the answer text.
    """
    model_name: Optional[str] = None                      # e.g., "Naomi Campbell"
    designer_name: Optional[str] = None                   # e.g., "Vivienne Westwood"
    season: Optional[str] = None                          # e.g., "Spring/Summer 2026", "SS26"
    event_name: Optional[str] = None                      # e.g., "Paris Fashion Week"
    month_year: Optional[str] = None                      # e.g., "September 2025"
    claimed_action: Optional[str] = None                  # e.g., "closed", "opened", "walked finale"
    first_time_pfw_close: Optional[str] = None            # e.g., "first time closing at Paris Fashion Week"
    career_length_years: Optional[str] = None             # e.g., "33", "33-year", "33 years"
    source_urls: List[str] = Field(default_factory=list)  # explicit URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_milestone() -> str:
    return """
    Extract the key details the answer provides about the Paris Fashion Week milestone.

    Return a JSON object with the following fields:
    - model_name: The specific supermodel who achieved the milestone (string).
    - designer_name: The designer/brand of the show (string; e.g., "Vivienne Westwood").
    - season: The fashion season mentioned (string; e.g., "Spring/Summer 2026" or "SS26").
    - event_name: The event name if mentioned (string; ideally "Paris Fashion Week").
    - month_year: The month and year of the show if mentioned (string; e.g., "September 2025").
    - claimed_action: The role the model had in the show (string; e.g., "closed", "finale", or "opened").
    - first_time_pfw_close: The answer's phrasing indicating it was the model's first time closing at Paris Fashion Week (string, exact phrase if mentioned; otherwise null).
    - career_length_years: The career length as described (string; include the wording if available, e.g., "33-year career" or just "33").
    - source_urls: An array of all URLs explicitly cited in the answer that support this event. Extract only actual URLs.

    IMPORTANT:
    - Do not invent or infer any information; extract exactly what appears in the answer.
    - For 'season', accept either "Spring/Summer 2026" or abbreviations like "SS26".
    - For 'claimed_action', capture terms like "closed", "walked the finale", "opening look", etc.
    - For 'career_length_years', prefer the literal phrasing (e.g., "33-year career") if present; otherwise the numeric token such as "33".
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extraction: MilestoneExtraction) -> None:
    """
    Build the verification tree to judge the answer according to the rubric.
    The top-level rubric node is critical and aggregates three critical sub-areas in parallel.
    """
    # Create the top-level rubric node (critical, parallel)
    top = evaluator.add_parallel(
        id="Model_and_Milestone_Verification",
        desc="Verifies that a specific model is identified and that the Paris Fashion Week milestone in September 2025 is accurately described according to all specified constraints",
        parent=evaluator.root,
        critical=True
    )

    # ----------------------------- Model Identified ----------------------------- #
    # Critical: The answer identifies a specific supermodel (existence check on extraction)
    evaluator.add_custom_node(
        result=(extraction.model_name is not None and str(extraction.model_name).strip() != ""),
        id="Model_Identified",
        desc="The answer identifies a specific supermodel who achieved this milestone",
        parent=top,
        critical=True
    )

    # ------------------------------- Show Details ------------------------------- #
    # Critical group: All details about the show must be accurate
    show_node = evaluator.add_parallel(
        id="Show_Details",
        desc="The answer accurately states the show was Vivienne Westwood Spring/Summer 2026, took place in September 2025 at Paris Fashion Week, and that the model closed (not opened) the show",
        parent=top,
        critical=True
    )

    # Prepare sources (can be empty; verification will still run but may fail)
    sources = extraction.source_urls if extraction and extraction.source_urls else []

    # 1) Designer is Vivienne Westwood
    designer_leaf = evaluator.add_leaf(
        id="Show_Designer_Vivienne_Westwood",
        desc="Designer is Vivienne Westwood",
        parent=show_node,
        critical=True
    )
    model_ref = extraction.model_name or "the model"
    await evaluator.verify(
        claim=f"{model_ref} closed a Vivienne Westwood runway show.",
        node=designer_leaf,
        sources=sources,
        additional_instruction="Verify that the runway show in question is by Vivienne Westwood. If sources indicate the model closed a Vivienne Westwood show, mark as supported."
    )

    # 2) Season is Spring/Summer 2026 (SS26)
    season_leaf = evaluator.add_leaf(
        id="Show_Season_SS26",
        desc="Season is Spring/Summer 2026 (SS26)",
        parent=show_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Vivienne Westwood show closed by {model_ref} was for the Spring/Summer 2026 season (often abbreviated as SS26).",
        node=season_leaf,
        sources=sources,
        additional_instruction="Treat 'Spring/Summer 2026' and 'SS26' as equivalent. The claim is supported if the sources clearly indicate the show is SS26."
    )

    # 3) Took place at Paris Fashion Week
    pfw_leaf = evaluator.add_leaf(
        id="Show_Event_Paris_Fashion_Week",
        desc="Took place at Paris Fashion Week",
        parent=show_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Vivienne Westwood SS26 show that {model_ref} closed took place during Paris Fashion Week.",
        node=pfw_leaf,
        sources=sources,
        additional_instruction="Confirm that this runway show was part of Paris Fashion Week (PFW)."
    )

    # 4) Occurred in September 2025
    month_year_leaf = evaluator.add_leaf(
        id="Show_Date_September_2025",
        desc="Took place in September 2025",
        parent=show_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Vivienne Westwood Spring/Summer 2026 show took place in September 2025.",
        node=month_year_leaf,
        sources=sources,
        additional_instruction="Check the timing of the SS26 show; Paris Fashion Week SS26 is held in September 2025. Accept reasonable phrasing such as 'Sept 2025'."
    )

    # 5) The model closed (not opened) the show
    closed_leaf = evaluator.add_leaf(
        id="Show_Role_Closed_Not_Opened",
        desc="The model closed (not opened) the show",
        parent=show_node,
        critical=True
    )
    # Prefer 'closed' wording; accept synonyms: 'finale', 'walked the finale', 'closing look'
    await evaluator.verify(
        claim=f"{model_ref} closed the show (i.e., walked the finale), and did not open it.",
        node=closed_leaf,
        sources=sources,
        additional_instruction="Treat 'closed', 'finale', or 'closing look' as equivalent indications. The claim is supported if the sources clearly indicate she closed rather than opened."
    )

    # ---------------------------- Career Significance --------------------------- #
    # Critical group: First time at PFW and 33-year career claim
    career_node = evaluator.add_parallel(
        id="Career_Significance",
        desc="The answer correctly states this was the model's first time closing at Paris Fashion Week in a 33-year career",
        parent=top,
        critical=True
    )

    # 1) First time closing at Paris Fashion Week
    first_time_leaf = evaluator.add_leaf(
        id="Career_First_Time_PFW_Close",
        desc="This was the model's first time closing at Paris Fashion Week",
        parent=career_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This was {model_ref}'s first time closing a show at Paris Fashion Week.",
        node=first_time_leaf,
        sources=sources,
        additional_instruction="Confirm that sources explicitly describe it as the model's first time closing at PFW."
    )

    # 2) 33-year career mentioned
    career_len_leaf = evaluator.add_leaf(
        id="Career_Length_33_Years",
        desc="The milestone is described as occurring in her 33-year career",
        parent=career_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Sources describe this milestone as occurring in {model_ref}'s 33-year career.",
        node=career_len_leaf,
        sources=sources,
        additional_instruction="Accept phrasing like 'in her 33-year career' or equivalent; the description should clearly indicate 33 years."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Paris Fashion Week SS26 milestone question.
    """
    # Initialize evaluator with a parallel root (we add a critical rubric node under it)
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_milestone(),
        template_class=MilestoneExtraction,
        extraction_name="milestone_extraction"
    )

    # Optional: record expected constants as GT aids (no strict enforcement here)
    evaluator.add_ground_truth({
        "expected_designer": "Vivienne Westwood",
        "expected_season": "Spring/Summer 2026 (SS26)",
        "expected_event": "Paris Fashion Week",
        "expected_month_year": "September 2025",
        "expected_milestone": "First time closing at PFW in a 33-year career"
    }, gt_type="expected_context")

    # Build verification tree and run verifications
    await build_verification_tree(evaluator, extracted)

    # Return structured summary
    return evaluator.get_summary()