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
TASK_ID = "meijer_thanksgiving_hours_2025"
TASK_DESCRIPTION = "What are the operating hours for Meijer grocery stores on Thanksgiving Day 2025?"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HoursExtraction(BaseModel):
    """
    Structured information extracted from the agent's answer about Meijer Thanksgiving 2025 hours.
    """
    retailer: Optional[str] = None  # Expected: "Meijer"
    holiday_name: Optional[str] = None  # Expected: "Thanksgiving Day", "Thanksgiving"
    holiday_year: Optional[str] = None  # Expected: "2025"
    holiday_date_str: Optional[str] = None  # e.g., "Thursday, November 27, 2025"
    status: Optional[str] = None  # Expected normalized: "open" or "closed"
    opening_time: Optional[str] = None  # e.g., "6 AM", "6 a.m.", "7:00 AM"
    closing_time: Optional[str] = None  # e.g., "4 PM", "5 p.m.", "3:00 PM"
    hours_note: Optional[str] = None  # e.g., "hours may vary by location", "reduced hours"
    sources: List[str] = Field(default_factory=list)  # URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_meijer_thanksgiving_hours() -> str:
    return """
    Extract the specific Thanksgiving Day 2025 store-hours information presented in the answer.

    Return a JSON object with these fields:
    - retailer: The retailer name the answer is about (expected "Meijer"); if unclear, return null.
    - holiday_name: The holiday name specifically referenced (e.g., "Thanksgiving", "Thanksgiving Day"); if unclear, return null.
    - holiday_year: The explicit holiday year mentioned (expected "2025"). If the year is not explicitly mentioned, return null.
    - holiday_date_str: Any explicit date string provided for Thanksgiving 2025, such as "Thursday, November 27, 2025". If not provided, return null.
    - status: Normalize to one of: "open" or "closed", based on what the answer says about Meijer on Thanksgiving Day 2025. If not stated, return null.
    - opening_time: If the answer states Meijer is open, extract the opening time exactly as written (e.g., "6 AM", "6 a.m.", "7:00 AM"). If not provided, return null.
    - closing_time: If the answer states Meijer is open, extract the closing time exactly as written (e.g., "4 PM", "5 p.m.", "3:00 PM"). If not provided, return null.
    - hours_note: Any qualifier about hours (e.g., "hours may vary by location", "reduced hours"). If none, return null.
    - sources: An array of all URL(s) cited in the answer that purportedly support the Thanksgiving 2025 status/hours. Only include actual URLs present in the answer (including markdown links). Do not invent URLs.

    Special rules:
    - Do not infer or guess missing values; if the answer does not explicitly provide an item, return null.
    - If the answer states stores are "closed", then opening_time and closing_time should be null.
    - If the answer states stores are "open", the opening_time and closing_time should reflect any time range given. If the answer only provides partial information (e.g., "closes early"), return null for any missing time.
    - Sources must be actual URLs mentioned in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_status(status: Optional[str]) -> Optional[str]:
    if not status:
        return None
    s = status.strip().lower()
    if "closed" in s:
        return "closed"
    if "open" in s:
        return "open"
    return status.strip().lower()


def _build_source_support_claim(extracted: HoursExtraction) -> str:
    """
    Build a specific claim to verify against cited URLs.
    """
    normalized = _normalize_status(extracted.status)
    date_phrase = "Thanksgiving Day 2025 (Thursday, November 27, 2025)"
    if normalized == "open":
        if extracted.opening_time and extracted.closing_time:
            return (
                f"Meijer grocery stores are open on {date_phrase} with special hours from "
                f"{extracted.opening_time} to {extracted.closing_time} (local time; hours may vary by location)."
            )
        else:
            # Fallback if hours missing (this should usually be gated by a previous node)
            return f"Meijer grocery stores are open on {date_phrase}."
    if normalized == "closed":
        return f"Meijer grocery stores are closed on {date_phrase}."
    # Generic fallback if status is unclear; usually will be skipped due to prerequisites
    return (
        f"At least one cited source explicitly states Meijer’s store status (open or closed) and any hours "
        f"for {date_phrase}."
    )


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extracted: HoursExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """
    # Main critical container for this task
    main_node = evaluator.add_parallel(
        id="Meijer_Thanksgiving_Hours_2025",
        desc="Verify the Thanksgiving Day 2025 operating hours information for Meijer grocery stores",
        parent=evaluator.root,
        critical=True
    )

    # 1) Applies_To_Meijer_US_Stores (critical leaf)
    applies_meijer_leaf = evaluator.add_leaf(
        id="Applies_To_Meijer_US_Stores",
        desc="Answer explicitly pertains to Meijer grocery stores (the Meijer chain operating in the United States)",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly pertains to Meijer grocery stores (the US-based Meijer retail chain), not any other retailer or country.",
        node=applies_meijer_leaf,
        additional_instruction=(
            "Pass this check only if the answer is clearly about the Meijer grocery chain that operates in the US. "
            "Mentions of non-Meijer retailers or non-US chains should fail. "
            "Implicitly referring to 'Meijer' without contradiction is acceptable."
        )
    )

    # 2) Applies_To_Thanksgiving_Day_2025 (critical leaf)
    applies_thanksgiving_leaf = evaluator.add_leaf(
        id="Applies_To_Thanksgiving_Day_2025",
        desc="Answer explicitly pertains to Thanksgiving Day 2025 (Thursday, November 27, 2025)",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states hours for Thanksgiving Day 2025 (Thursday, November 27, 2025). Stating 'Thanksgiving 2025' counts.",
        node=applies_thanksgiving_leaf,
        additional_instruction=(
            "This should fail if the answer refers to a different year or only 'Thanksgiving' without indicating 2025. "
            "If the exact date 'Thursday, November 27, 2025' is used, that counts as explicitly 2025."
        )
    )

    # 3) Status_And_Hours (critical sequential)
    status_and_hours_node = evaluator.add_sequential(
        id="Status_And_Hours",
        desc="Answer provides Meijer’s open/closed status and the corresponding operating-hours information for Thanksgiving Day 2025",
        parent=main_node,
        critical=True
    )

    # 3.1) States_Open_Or_Closed_Status (critical leaf)
    states_status_leaf = evaluator.add_leaf(
        id="States_Open_Or_Closed_Status",
        desc="States whether Meijer stores are open or closed on Thanksgiving Day 2025",
        parent=status_and_hours_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer clearly states whether Meijer stores are open or closed on Thanksgiving Day 2025.",
        node=states_status_leaf,
        additional_instruction=(
            "This is a meta-check: scan the answer text to see if it explicitly claims 'open' or 'closed' for Thanksgiving 2025. "
            "Vague language that fails to indicate open vs. closed should fail."
        )
    )

    # 3.2) Provides_Hours_If_Open_Or_Closure_If_Closed (critical leaf via custom logic)
    normalized_status = _normalize_status(extracted.status)
    if normalized_status == "open":
        provides_ok = bool(extracted.opening_time and extracted.opening_time.strip()) and bool(
            extracted.closing_time and extracted.closing_time.strip()
        )
    elif normalized_status == "closed":
        # If closed, they should indicate closure and not give operating hours (opening/closing times should be absent)
        provides_ok = not (bool(extracted.opening_time and extracted.opening_time.strip()) or
                           bool(extracted.closing_time and extracted.closing_time.strip()))
    else:
        provides_ok = False

    evaluator.add_custom_node(
        result=provides_ok,
        id="Provides_Hours_If_Open_Or_Closure_If_Closed",
        desc="If stated open, provides both opening time and closing time for Thanksgiving Day 2025; if stated closed, explicitly indicates closure (i.e., no operating hours given)",
        parent=status_and_hours_node,
        critical=True
    )

    # 4) Verifiable_Reliable_Sourcing (critical parallel) with two sub-checks
    sourcing_parent = evaluator.add_parallel(
        id="Verifiable_Reliable_Sourcing",
        desc="Provides verifiable support from reliable sources (e.g., citations/URLs) for the stated open/closed status and any stated hours",
        parent=main_node,
        critical=True
    )

    # 4.1) Sources_Provided (critical existence check)
    sources_provided = evaluator.add_custom_node(
        result=bool(extracted.sources and len(extracted.sources) > 0),
        id="Sources_Provided",
        desc="At least one citation/URL is provided in the answer to support the Thanksgiving 2025 status/hours",
        parent=sourcing_parent,
        critical=True
    )

    # 4.2) Sources_Support_Claim (critical verification against URLs)
    sources_support_leaf = evaluator.add_leaf(
        id="Sources_Support_Status_And_Hours",
        desc="At least one cited source supports the stated open/closed status and any stated hours for Thanksgiving Day 2025",
        parent=sourcing_parent,
        critical=True
    )
    claim_to_verify = _build_source_support_claim(extracted)
    await evaluator.verify(
        claim=claim_to_verify,
        node=sources_support_leaf,
        sources=extracted.sources,  # may be multiple URLs
        additional_instruction=(
            "Verify the claim strictly against the provided URL(s). "
            "It is acceptable if only one source supports the claim. "
            "Prioritize official Meijer announcements/pages or reputable news outlets. "
            "If the source references a different year (e.g., 2024 or 2026), this should fail. "
            "Allow reasonable phrases like 'most stores' and 'hours may vary by location' as support for typical ranges. "
            "Pharmacy, gas station, or curbside hours are not the same as general store hours; ensure the source supports general store status/hours."
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
    Evaluate an answer for Meijer Thanksgiving Day 2025 operating hours.
    """
    # Initialize evaluator
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

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_meijer_thanksgiving_hours(),
        template_class=HoursExtraction,
        extraction_name="meijer_thanksgiving_hours_extraction"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()