import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_politics_research_2024_2026"
TASK_DESCRIPTION = (
    "Who delivered the Republican response to the 2024 State of the Union address, "
    "which state do they represent, and what was the dollar amount of the 'warrior dividend' "
    "announced by President Trump in his 2026 State of the Union address?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ResearchExtraction(BaseModel):
    # Republican response (2024 SOTU)
    senator_name: Optional[str] = None
    senator_state: Optional[str] = None
    senator_sources: List[str] = Field(default_factory=list)

    # Warrior dividend (2026 SOTU)
    warrior_dividend_amount: Optional[str] = None
    warrior_dividend_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_research() -> str:
    return """
    Extract the following information exactly as stated in the answer. Do not infer or correct the content; simply extract what appears.

    Fields to extract:
    1) senator_name: The full name of the person the answer claims delivered the Republican response to the 2024 State of the Union address.
    2) senator_state: The U.S. state that this senator represents (as written in the answer, e.g., "Alabama", "AL", or "R-AL").
    3) senator_sources: A list of all URLs cited in the answer that support the identification of the senator and/or their state. These may include news articles, official bios, or announcements. Extract only actual URLs present in the answer.

    4) warrior_dividend_amount: The dollar amount stated in the answer for the "warrior dividend" that President Trump announced in his 2026 State of the Union address (e.g., "$2,500" or "2500").
    5) warrior_dividend_sources: A list of all URLs cited in the answer that support the claim about the "warrior dividend" amount (e.g., transcripts, reputable news coverage). Extract only actual URLs present in the answer.

    Notes:
    - If any field is missing, set it to null (or empty list for sources).
    - For URL fields, extract only valid URLs explicitly present in the answer (plain URLs or markdown links).
    - If a URL is missing a protocol, prepend "http://".
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the research task about the 2024 GOP response and the 2026 'warrior dividend'.
    Builds a verification tree with critical gating and URL-grounded checks.
    """
    # 1) Initialize evaluator (root is non-critical by framework design)
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

    # 2) Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_research(),
        template_class=ResearchExtraction,
        extraction_name="extracted_research_fields",
    )

    # 3) Build verification tree according to rubric

    # Top-level critical node (mirror rubric "Research_Task" which is critical/parallel)
    research_node = evaluator.add_parallel(
        id="Research_Task",
        desc="Complete accurate research on recent U.S. political events",
        parent=root,
        critical=True,
    )

    # 3.1 Republican response (2024 SOTU) subtree (critical/parallel)
    gop_resp_node = evaluator.add_parallel(
        id="Republican_Response_2024",
        desc="Identify the Senator who delivered the Republican response to the 2024 State of the Union address",
        parent=research_node,
        critical=True,
    )

    # 3.1.a Senator Name verification as a sequential gated node
    sen_name_node = evaluator.add_sequential(
        id="Senator_Name",
        desc="Provide the correct name of the senator who delivered the Republican response to the 2024 State of the Union address",
        parent=gop_resp_node,
        critical=True,
    )

    # Existence + sources check (critical gate)
    evaluator.add_custom_node(
        result=(_nonempty(extracted.senator_name) and len(extracted.senator_sources) > 0),
        id="Senator_Name_provided",
        desc="Senator name and supporting sources are provided",
        parent=sen_name_node,
        critical=True,
    )

    # Support check (leaf)
    sen_name_supported_leaf = evaluator.add_leaf(
        id="Senator_Name_supported",
        desc="Claim about who delivered the GOP response in 2024 is supported by cited sources",
        parent=sen_name_node,
        critical=True,
    )

    sen_name_claim = (
        f"The senator who delivered the Republican response to the 2024 State of the Union address is {extracted.senator_name or ''}."
    )
    await evaluator.verify(
        claim=sen_name_claim,
        node=sen_name_supported_leaf,
        sources=extracted.senator_sources,
        additional_instruction=(
            "Verify that the page(s) explicitly state that this person delivered the "
            "Republican response (also known as the GOP response or Republican rebuttal) "
            "to the 2024 State of the Union (2024 SOTU). Minor name variations (middle names, "
            "initials) are acceptable."
        ),
    )

    # 3.1.b Senator State verification as a sequential gated node
    sen_state_node = evaluator.add_sequential(
        id="Senator_State",
        desc="Provide the correct state that the senator represents in the U.S. Senate",
        parent=gop_resp_node,
        critical=True,
    )

    # Existence + sources check (critical gate)
    evaluator.add_custom_node(
        result=(_nonempty(extracted.senator_state) and len(extracted.senator_sources) > 0),
        id="Senator_State_provided",
        desc="Senator state and supporting sources are provided",
        parent=sen_state_node,
        critical=True,
    )

    # Support check (leaf)
    sen_state_supported_leaf = evaluator.add_leaf(
        id="Senator_State_supported",
        desc="Claim about the senator's represented state is supported by cited sources",
        parent=sen_state_node,
        critical=True,
    )

    # Use both name and state if available; allow flexible matching like R-AL or 'Republican from Alabama'
    sen_state_claim = (
        f"{extracted.senator_name or 'The named senator'} represents the state of {extracted.senator_state or ''} in the U.S. Senate."
    )
    await evaluator.verify(
        claim=sen_state_claim,
        node=sen_state_supported_leaf,
        sources=extracted.senator_sources,
        additional_instruction=(
            "Verify that the page(s) indicate this person's current U.S. Senate state. "
            "Accept formats like 'R-AL', 'Republican from Alabama', 'Senator from Alabama', "
            "or equivalent. Minor name variations are acceptable."
        ),
    )

    # 3.2 Warrior Dividend Amount (critical) as sequential gated node
    warrior_node = evaluator.add_sequential(
        id="Warrior_Dividend_Amount",
        desc="Provide the correct dollar amount of the 'warrior dividend' announced by President Trump in his 2026 State of the Union address",
        parent=research_node,
        critical=True,
    )

    # Existence + sources check (critical gate)
    evaluator.add_custom_node(
        result=(_nonempty(extracted.warrior_dividend_amount) and len(extracted.warrior_dividend_sources) > 0),
        id="Warrior_Dividend_Provided",
        desc="Warrior dividend amount and supporting sources are provided",
        parent=warrior_node,
        critical=True,
    )

    # Support check (leaf)
    warrior_supported_leaf = evaluator.add_leaf(
        id="Warrior_Dividend_Supported",
        desc="Claim about the 2026 SOTU 'warrior dividend' amount is supported by cited sources",
        parent=warrior_node,
        critical=True,
    )

    warrior_claim = (
        f"In the 2026 State of the Union address, President Trump announced a 'warrior dividend' amount of {extracted.warrior_dividend_amount or ''}."
    )
    await evaluator.verify(
        claim=warrior_claim,
        node=warrior_supported_leaf,
        sources=extracted.warrior_dividend_sources,
        additional_instruction=(
            "Verify that the page(s) clearly attribute to President Trump, in the 2026 State of the Union address, "
            "the announcement of a 'warrior dividend' of the specified amount. Allow formatting differences such as "
            "currency symbols, commas, or phrasing like '$2,500', '2,500 dollars', or 'two thousand five hundred dollars'. "
            "The year 2026 should be explicit or clearly implied by the source context."
        ),
    )

    # 4) Return summary
    return evaluator.get_summary()