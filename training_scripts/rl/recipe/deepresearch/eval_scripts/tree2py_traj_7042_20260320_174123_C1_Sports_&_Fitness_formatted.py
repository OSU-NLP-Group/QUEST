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
TASK_ID = "team_usa_jersey_honor_2026"
TASK_DESCRIPTION = """
What jersey number did Team USA honor after winning the gold medal in men's ice hockey at the 2026 Winter Olympics?
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AnswerExtraction(BaseModel):
    """
    Extract key details as they explicitly appear in the answer text.
    Prefer strings over numbers to be robust to formatting.
    """
    honored_number: Optional[str] = None
    honored_person: Optional[str] = None
    honoring_action_sentence: Optional[str] = None
    gaudreau_teamusa_number_statement: Optional[str] = None
    event_context_statement: Optional[str] = None
    result_statement: Optional[str] = None
    game_date_statement: Optional[str] = None
    gaudreau_death_date_statement: Optional[str] = None
    venue_statement: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_answer_facts() -> str:
    return """
    From the answer text, extract the following fields exactly as they appear (use null if missing):
    - honored_number: The jersey number that the answer claims Team USA honored.
    - honored_person: The person whose jersey was honored (e.g., Johnny Gaudreau), if mentioned.
    - honoring_action_sentence: A sentence or quoted fragment that describes the honoring action (e.g., carrying a jersey onto the ice), if present.
    - gaudreau_teamusa_number_statement: Any explicit statement that Johnny Gaudreau wore jersey number 13 for Team USA, if present.
    - event_context_statement: A sentence/fragment that says the honoring occurred after Team USA won gold in men's ice hockey at the 2026 Winter Olympics (Milano Cortina 2026), if present.
    - result_statement: A sentence/fragment that states Team USA defeated Canada 2-1 in overtime in the gold medal game, if present.
    - game_date_statement: A sentence/fragment that states the gold medal game date as February 22, 2026 (accept reasonable variants like 'Feb 22, 2026'), if present.
    - gaudreau_death_date_statement: A sentence/fragment that states Johnny Gaudreau died on August 29, 2024 (accept reasonable variants like 'Aug. 29, 2024'), if present.
    - venue_statement: A sentence/fragment that states the venue/location as Santagiulia Arena in Milan (accept variants like 'Santa Giulia Arena', 'Milan, Italy'), if present.
    - source_urls: All URLs explicitly present in the answer text; include full URLs and ignore malformed ones.

    Do not infer or add any information that is not explicitly stated in the answer.
    Return the result strictly in the specified JSON schema.
    """


# --------------------------------------------------------------------------- #
# Verification helper                                                         #
# --------------------------------------------------------------------------- #
async def build_and_verify_honor_tree(
    evaluator: Evaluator,
    parent_node,
) -> None:
    """
    Build the rubric tree as specified and run simple (answer-text) verifications
    for each critical leaf criterion.
    """
    # Create the rubric root node under the global root
    honor_node = evaluator.add_parallel(
        id="Jersey_Number_Honor_Identification",
        desc="Verifies the answer identifies the honored jersey number and satisfies all provided constraints about the honoring context.",
        parent=parent_node,
        critical=True
    )

    # 1) Honoring action + number 13 present
    node_action_number = evaluator.add_leaf(
        id="Honoring_Action_And_Number",
        desc="States Team USA honored Johnny Gaudreau by carrying his No. 13 jersey onto the ice after winning gold (includes both the honoring action and the jersey number 13).",
        parent=honor_node,
        critical=True
    )
    claim_action_number = (
        "The answer explicitly states that after winning the gold medal, Team USA honored Johnny Gaudreau by "
        "carrying (or equivalently bringing/holding/raising) his No. 13 jersey onto the ice."
    )
    await evaluator.verify(
        claim=claim_action_number,
        node=node_action_number,
        additional_instruction=(
            "Judge only based on the answer text. Accept reasonable paraphrases for the honoring action "
            "such as 'brought out', 'held up', 'skated with', etc. Accept number formatting variants like "
            "'No. 13', 'No 13', 'number 13', or '#13'. It's okay if the 'after winning gold' context is "
            "implied within a victory celebration statement."
        )
    )

    # 2) Gaudreau jersey number for Team USA stated
    node_gaudreau_num = evaluator.add_leaf(
        id="Gaudreau_TeamUSA_Jersey_Number",
        desc="States Johnny Gaudreau wore jersey number 13 for Team USA.",
        parent=honor_node,
        critical=True
    )
    claim_gaudreau_num = (
        "The answer states (explicitly or by clear implication) that Johnny Gaudreau wore jersey number 13 for Team USA."
    )
    await evaluator.verify(
        claim=claim_gaudreau_num,
        node=node_gaudreau_num,
        additional_instruction=(
            "Judge only from the answer text. Allow variants like 'No. 13', 'No 13', 'jersey 13', or '#13'. "
            "It's acceptable if this is mentioned alongside the honoring action as long as it clearly conveys "
            "that 13 was Gaudreau's Team USA number."
        )
    )

    # 3) Event context: after winning gold at the 2026 Winter Olympics (Milano Cortina 2026)
    node_event_context = evaluator.add_leaf(
        id="Event_Context_2026_Olympics",
        desc="States the honoring occurred after Team USA won gold in men's ice hockey at the 2026 Winter Olympics (Milano Cortina 2026).",
        parent=honor_node,
        critical=True
    )
    claim_event_context = (
        "The answer states that the honoring occurred after Team USA won gold in men's ice hockey at the 2026 Winter Olympics "
        "(also known as Milano Cortina 2026)."
    )
    await evaluator.verify(
        claim=claim_event_context,
        node=node_event_context,
        additional_instruction=(
            "Judge only from the answer text. Accept reasonable paraphrases like 'after clinching gold', "
            "'following their gold-medal win', etc. Accept naming variants like 'Milano Cortina 2026', "
            "'the 2026 Winter Olympics', or 'the 2026 Games'."
        )
    )

    # 4) Gold medal game result
    node_result = evaluator.add_leaf(
        id="Gold_Medal_Game_Result",
        desc="States Team USA defeated Canada 2-1 in overtime in the gold medal game.",
        parent=honor_node,
        critical=True
    )
    claim_result = "The answer states that Team USA defeated Canada 2-1 in overtime in the gold medal game."
    await evaluator.verify(
        claim=claim_result,
        node=node_result,
        additional_instruction=(
            "Judge only from the answer text. Accept numeric formatting variants like '2–1', '2 to 1', or '2 : 1'. "
            "Also accept 'OT' as equivalent to 'overtime'."
        )
    )

    # 5) Gold medal game date
    node_date = evaluator.add_leaf(
        id="Gold_Medal_Game_Date",
        desc="States the gold medal game was played on February 22, 2026.",
        parent=honor_node,
        critical=True
    )
    claim_date = "The answer states that the gold medal game took place on February 22, 2026."
    await evaluator.verify(
        claim=claim_date,
        node=node_date,
        additional_instruction=(
            "Judge only from the answer text. Accept common date format variants such as 'Feb 22, 2026' "
            "or '22 February 2026'."
        )
    )

    # 6) Gaudreau death date
    node_death = evaluator.add_leaf(
        id="Gaudreau_Death_Date",
        desc="States Johnny Gaudreau died on August 29, 2024.",
        parent=honor_node,
        critical=True
    )
    claim_death = "The answer states that Johnny Gaudreau died on August 29, 2024."
    await evaluator.verify(
        claim=claim_death,
        node=node_death,
        additional_instruction=(
            "Judge only from the answer text. Accept variants such as 'Aug. 29, 2024' or '29 August 2024'. "
            "Look for explicit mention or an unambiguous equivalent phrasing like 'passed away on'."
        )
    )

    # 7) Venue location
    node_venue = evaluator.add_leaf(
        id="Venue_Location",
        desc="States the game was played at Santagiulia Arena in Milan.",
        parent=honor_node,
        critical=True
    )
    claim_venue = "The answer states that the game was played at Santagiulia Arena in Milan."
    await evaluator.verify(
        claim=claim_venue,
        node=node_venue,
        additional_instruction=(
            "Judge only from the answer text. Accept minor naming variants like 'Santa Giulia Arena' or "
            "'Arena di Santa Giulia', and accept 'Milan' or 'Milano, Italy' as equivalent location mentions."
        )
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
    Evaluate an answer for the Team USA 2026 jersey honor task.
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
        default_model=model
    )

    # Extract structured information (for transparency/debugging; verification is answer-text based)
    extracted = await evaluator.extract(
        prompt=prompt_extract_answer_facts(),
        template_class=AnswerExtraction,
        extraction_name="answer_extraction"
    )

    # Optionally add lightweight GT info to summary for clarity
    evaluator.add_ground_truth({
        "expected_honored_number": "13",
        "person": "Johnny Gaudreau",
        "context": {
            "event": "2026 Winter Olympics (Milano Cortina 2026)",
            "opponent": "Canada",
            "result": "2-1 OT",
            "date": "February 22, 2026",
            "venue": "Santagiulia Arena, Milan",
            "gaudreau_death_date": "August 29, 2024"
        }
    })

    # Build and verify the rubric tree
    await build_and_verify_honor_tree(evaluator, root)

    return evaluator.get_summary()