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
TASK_ID = "hobby_lobby_nyd_2026_closing_time"
TASK_DESCRIPTION = "What time does Hobby Lobby close on New Year's Day 2026?"

EXPECTED_STORE = "Hobby Lobby"
EXPECTED_HOLIDAY = "New Year's Day 2026"
EXPECTED_DATE_ISO = "2026-01-01"
EXPECTED_CLOSING_TIME_530PM = "5:30 p.m. (17:30)"  # For ground truth info logging


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HolidayClosingInfo(BaseModel):
    """
    Structured extraction of the user's answer regarding Hobby Lobby's New Year's Day 2026 closing time.
    """
    store_name: Optional[str] = None
    holiday_name: Optional[str] = None  # e.g., "New Year's Day 2026"
    holiday_date_text: Optional[str] = None  # e.g., "January 1, 2026" (keep as text for flexibility)
    closing_time: Optional[str] = None  # e.g., "5:30 PM", "17:30", "5:30 p.m."
    source_urls: List[str] = Field(default_factory=list)  # URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_closing_info() -> str:
    return """
    Extract structured information from the answer specifically about Hobby Lobby’s closing time on New Year's Day 2026.

    Return a JSON object with the following fields:
    - store_name: The store chain mentioned (e.g., "Hobby Lobby").
    - holiday_name: The holiday explicitly referenced (e.g., "New Year's Day 2026").
    - holiday_date_text: The date string as written (e.g., "January 1, 2026", "Jan 1, 2026")—do not convert to a numeric date, keep the text.
    - closing_time: The specific closing time stated (e.g., "5:30 PM", "5:30 p.m.", "17:30"). If multiple times are given, choose the one directly tied to New Year's Day 2026.
    - source_urls: An array of full URLs explicitly cited in the answer that purportedly support the closing time for Hobby Lobby on New Year's Day 2026.

    Rules:
    - Only extract what is explicitly present in the answer text.
    - Do not invent any URL. If the answer names a source but no URL is provided, do not include it in source_urls.
    - Include markdown links' URL targets (e.g., [text](http://example.com)) as URLs in source_urls.
    - If a field is missing, return null (or empty array for source_urls).
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    root_node,
    info: HolidayClosingInfo
) -> None:
    """
    Build the verification tree under a critical parallel node and perform verifications.
    """

    # Main critical node representing the rubric root
    main_node = evaluator.add_parallel(
        id="Hobby_Lobby_New_Years_Day_2026_Closing_Time",
        desc="Report what time Hobby Lobby closes on New Year's Day 2026.",
        parent=root_node,
        critical=True
    )

    # 1) Store check: Answer pertains to Hobby Lobby
    store_node = evaluator.add_leaf(
        id="Store_Is_Hobby_Lobby",
        desc="Answer pertains to Hobby Lobby.",
        parent=main_node,
        critical=True
    )
    store_claim = "The answer is specifically about Hobby Lobby (the arts-and-crafts retailer), not a different store."
    await evaluator.verify(
        claim=store_claim,
        node=store_node,
        additional_instruction="Judge based on the answer text only. Accept reasonable variants like 'HobbyLobby' as referring to Hobby Lobby."
    )

    # 2) Date check: Answer pertains specifically to New Year's Day 2026 (January 1, 2026)
    date_node = evaluator.add_leaf(
        id="Date_Is_New_Years_Day_2026",
        desc="Answer pertains specifically to New Year's Day 2026 (January 1, 2026).",
        parent=main_node,
        critical=True
    )
    date_claim = (
        "The closing time discussed in the answer is specifically for New Year's Day 2026, i.e., January 1, 2026."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        additional_instruction="Check that the answer ties the closing time explicitly to New Year's Day 2026 (Jan 1, 2026), not to another date or holiday."
    )

    # 3) Closing time provided: existence check (critical)
    closing_time_exists = bool(info.closing_time and info.closing_time.strip())
    evaluator.add_custom_node(
        result=closing_time_exists,
        id="Closing_Time_Provided",
        desc="States a specific closing time for Hobby Lobby on that date.",
        parent=main_node,
        critical=True
    )

    # 4) Closing time equals 5:30 p.m. (match within the answer)
    time_match_node = evaluator.add_leaf(
        id="Closing_Time_Is_5_30_PM",
        desc="Closing time is accurately stated as 5:30 p.m. (17:30).",
        parent=main_node,
        critical=True
    )
    # This check ensures the answer itself states 5:30 PM (not just sources).
    match_claim = (
        "The closing time stated in the answer corresponds to 5:30 p.m. (17:30)."
    )
    await evaluator.verify(
        claim=match_claim,
        node=time_match_node,
        additional_instruction=(
            "Evaluate only the answer text. Consider minor formatting variants equivalent (e.g., '5:30 PM', '5:30 p.m.', '17:30', '5:30pm'). "
            "If the answer clearly indicates Hobby Lobby closes at or by 5:30 on New Year's Day 2026, mark as correct."
        )
    )

    # 5) Verifiable source provided (existence of URLs that can support the stated closing time)
    has_sources = bool(info.source_urls and len(info.source_urls) > 0)
    evaluator.add_custom_node(
        result=has_sources,
        id="Verifiable_Source_Provided",
        desc="Provides a verifiable citation (e.g., URL) from Hobby Lobby official sources or a reliable news source supporting the stated closing time.",
        parent=main_node,
        critical=True
    )

    # Source-grounded verification for the closing time being 5:30 p.m.
    # Even though the rubric's last node is an existence check, we still add a source-grounded verification
    # to ensure the factual claim is supported by the cited webpage(s).
    source_support_node = evaluator.add_leaf(
        id="Closing_Time_5_30_PM_Supported_By_Sources",
        desc="The 5:30 p.m. closing time on New Year's Day 2026 is supported by cited sources.",
        parent=main_node,
        critical=True
    )
    support_claim = "On January 1, 2026 (New Year's Day), Hobby Lobby closes at 5:30 p.m. (17:30)."
    await evaluator.verify(
        claim=support_claim,
        node=source_support_node,
        sources=info.source_urls,  # May be empty; in that case this verification will be skipped due to the critical sibling failing
        additional_instruction=(
            "Verify using the cited URL(s). Prefer official Hobby Lobby sources (e.g., hobbylobby.com) or credible news outlets. "
            "Allow equivalent time formats (e.g., 5:30 PM, 5:30 p.m., 17:30). "
            "If multiple sources conflict, prioritize official sources and the most authoritative/updated information."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
) -> Dict[str, Any]:
    """
    Evaluate an answer to: "What time does Hobby Lobby close on New Year's Day 2026?"
    """

    # Initialize evaluator
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

    # Extraction
    info = await evaluator.extract(
        prompt=prompt_extract_closing_info(),
        template_class=HolidayClosingInfo,
        extraction_name="holiday_closing_info"
    )

    # Ground truth info (for transparency; not used as hard assertions)
    evaluator.add_ground_truth({
        "expected_store": EXPECTED_STORE,
        "expected_holiday": EXPECTED_HOLIDAY,
        "expected_date_iso": EXPECTED_DATE_ISO,
        "expected_closing_time": EXPECTED_CLOSING_TIME_530PM
    }, gt_type="expected_facts")

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, root, info)

    # Return summary
    return evaluator.get_summary()