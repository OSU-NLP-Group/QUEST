import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "edinburgh_zoo_last_entry_february"
TASK_DESCRIPTION = "What is the last entry time for visitors to Edinburgh Zoo in February?"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class LastEntryExtraction(BaseModel):
    """
    Extraction of what the answer claims and the URLs it cites.
    """
    last_entry_time: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_last_entry_info() -> str:
    return """
    Extract from the answer:
    1) last_entry_time: The last entry time for visitors to Edinburgh Zoo in February as explicitly stated in the answer. 
       - Return the time string exactly as written in the answer (e.g., "3pm", "3 pm", "3:00 pm", "15:00", "15.00").
       - If the answer gives a seasonal/month range that includes February (e.g., "winter months" or "November–February"), extract the last entry time that applies for that period.
       - If the answer does not clearly state a last entry time for February, return null.
    2) source_urls: Extract all URLs present in the answer (including markdown links). Do not invent URLs. If none, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def filter_official_urls(urls: List[str]) -> List[str]:
    """
    Return only URLs that appear to be from the official Edinburgh Zoo domain.
    """
    official_substr = "edinburghzoo.org.uk"
    return [u for u in urls if isinstance(u, str) and official_substr in u.lower()]


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_last_entry_time_february(evaluator: Evaluator, root_node, extracted: LastEntryExtraction) -> None:
    """
    Build the rubric tree and run the two checks:
    - Reference_URL (critical): An official Edinburgh Zoo URL that contains opening hours/last entry info is provided.
    - Correct_Time_Stated (critical): The February last entry time is correctly 3 pm (15:00), supported by the cited official source(s), and the answer aligns.
    """
    # Parent node for this task (critical, parallel as per rubric)
    task_node = evaluator.add_parallel(
        id="Last_Entry_Time_February",
        desc="Correctly identify the last entry time for Edinburgh Zoo in February",
        parent=root_node,
        critical=True,
    )

    # 1) Reference_URL (critical)
    ref_node = evaluator.add_leaf(
        id="Reference_URL",
        desc="A reference URL from the official Edinburgh Zoo website (edinburghzoo.org.uk) supporting the opening hours information is provided",
        parent=task_node,
        critical=True,
    )

    all_urls = extracted.source_urls or []
    if len(all_urls) == 0:
        # No URLs provided; mark as failed directly
        ref_node.score = 0.0
        ref_node.status = "failed"
    else:
        # Verify at least one of the provided URLs is official AND contains opening hours / last entry info
        claim_ref = (
            "This webpage is part of the official Edinburgh Zoo website (domain includes edinburghzoo.org.uk) "
            "and it contains information about opening hours, closing time, or last entry details for visiting the zoo."
        )
        await evaluator.verify(
            claim=claim_ref,
            node=ref_node,
            sources=all_urls,
            additional_instruction=(
                "Pass only if the page is clearly on edinburghzoo.org.uk and explicitly shows opening times or last entry information. "
                "Pages unrelated to visiting hours should not pass."
            ),
        )

    # 2) Correct_Time_Stated (critical)
    correct_node = evaluator.add_leaf(
        id="Correct_Time_Stated",
        desc="The last entry time is correctly stated as 3pm (or 15:00)",
        parent=task_node,
        critical=True,
    )

    stated_time = extracted.last_entry_time or ""
    official_urls = filter_official_urls(all_urls)

    claim_correct = (
        f"The last entry time for visitors to Edinburgh Zoo in February is 3 pm (15:00). "
        f"The answer states it as '{stated_time}'. "
        "Verify that: (1) the provided webpage(s) from the official site explicitly support 3 pm (15:00) as the last entry time in February "
        "(a schedule that includes February, e.g., 'winter months' or 'November–February', counts as supporting February), "
        "and (2) the answer's stated time is equivalent to 3 pm (allow minor formatting variants like '3pm', '3 pm', '3:00pm', '3:00 pm', '15:00', or '15.00')."
    )

    # Even if official_urls is empty, we still call verify(); because Reference_URL is a critical sibling,
    # if it failed above, this node will be automatically skipped by the evaluator's precondition logic.
    await evaluator.verify(
        claim=claim_correct,
        node=correct_node,
        sources=official_urls,
        additional_instruction=(
            "Do not rely on your own knowledge. Use the provided official Edinburgh Zoo page(s). "
            "If the page conveys that February is within a period whose last entry is 3 pm, treat it as supporting February."
        ),
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Edinburgh Zoo February last entry time task.
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
        default_model=model,
    )

    # 1) Extract what the answer states and its sources
    extracted = await evaluator.extract(
        prompt=prompt_extract_last_entry_info(),
        template_class=LastEntryExtraction,
        extraction_name="last_entry_extraction",
    )

    # 2) Build verification sub-tree and run checks
    await verify_last_entry_time_february(evaluator, root, extracted)

    # 3) Return evaluation summary
    return evaluator.get_summary()