import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ms_console_2020_digital_only"
TASK_DESCRIPTION = (
    "What is the name and model of the Microsoft gaming console that was released in November 2020, "
    "features 512GB of internal SSD storage, had an original launch price of $299 USD, and is designed "
    "as a digital-only console without an optical disc drive?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class ConsoleExtraction(BaseModel):
    """
    Structured information extracted from the answer text about the identified console.
    All fields should be taken exactly from the answer text when available.
    """
    name_model: Optional[str] = None
    manufacturer: Optional[str] = None
    release_date: Optional[str] = None
    storage: Optional[str] = None
    launch_price_usd: Optional[str] = None
    digital_only: Optional[str] = None  # e.g., "digital-only", "no optical drive", "disc-less", "unknown"
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_console_info() -> str:
    return """
    From the provided answer, extract the details about the single Microsoft gaming console the answer claims
    to satisfy ALL the following constraints:
    – Released in November 2020.
    – Has 512GB internal SSD storage.
    – Original launch price is $299 USD.
    – Designed as a digital-only console without an optical disc drive.

    RULES:
    1) Extract EXACTLY what the answer states. Do not infer or invent missing facts.
    2) If multiple consoles are mentioned, choose the one the answer identifies as matching ALL constraints.
    3) If any requested field is not explicitly stated in the answer, set it to null.
    4) Also extract every URL explicitly present in the answer text (source_urls). Only include valid URLs.

    Return a JSON object with these fields:
    - name_model: The console name/model as stated (e.g., "Xbox Series S").
    - manufacturer: The manufacturer as stated (e.g., "Microsoft" or "Microsoft Xbox").
    - release_date: The release date string as written (e.g., "November 10, 2020" or "November 2020"), or null.
    - storage: The internal storage string as written (e.g., "512GB SSD" or "512 GB SSD"), or null.
    - launch_price_usd: The launch price as written (e.g., "$299" or "$299 USD" or "USD 299"), or null.
    - digital_only: The digital-only/drive status as written (e.g., "digital-only", "no optical drive", "disc-less").
                     If not explicitly stated, return null.
    - source_urls: Array of URLs cited in the answer text (can be empty if none were provided).
    """


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_console(evaluator: Evaluator, extraction: ConsoleExtraction) -> None:
    """
    Build the verification tree according to the rubric and run the verifications.
    """
    # Top-level rubric node (critical, parallel aggregation)
    console_node = evaluator.add_parallel(
        id="Console_Identification",
        desc="Verify that the identified Microsoft gaming console satisfies all stated constraints (name/model + release date + storage + launch price + digital-only).",
        parent=evaluator.root,
        critical=True
    )

    # Prepare commonly used values
    console_name = extraction.name_model.strip() if extraction.name_model else None
    sources_list = extraction.source_urls if extraction.source_urls else []

    # 1) Console Name and Model — check presence (existence)
    evaluator.add_custom_node(
        result=bool(console_name),
        id="Console_Name_And_Model",
        desc="Provides the console’s name and model designation (i.e., clearly identifies the specific console).",
        parent=console_node,
        critical=True
    )

    # 2) Manufacturer is Microsoft
    manufacturer_node = evaluator.add_leaf(
        id="Manufacturer_Is_Microsoft",
        desc="Console is manufactured by Microsoft.",
        parent=console_node,
        critical=True
    )
    manufacturer_claim = (
        f"The console {console_name} is manufactured by Microsoft."
        if console_name else
        "The console in question is manufactured by Microsoft."
    )

    # 3) Release date is November 2020
    release_node = evaluator.add_leaf(
        id="Release_Date_Is_November_2020",
        desc="Console release date is in November 2020 (meets the stated launch timing; if specified, matches Nov 10, 2020 worldwide launch).",
        parent=console_node,
        critical=True
    )
    release_claim = (
        f"The console {console_name} was released in November 2020."
        if console_name else
        "The console was released in November 2020."
    )

    # 4) Internal SSD storage is 512GB
    storage_node = evaluator.add_leaf(
        id="Internal_SSD_Storage_512GB",
        desc="Console has 512GB of internal SSD storage.",
        parent=console_node,
        critical=True
    )
    storage_claim = (
        f"The console {console_name} has 512 GB of internal SSD storage."
        if console_name else
        "The console has 512 GB of internal SSD storage."
    )

    # 5) Original launch price was $299 USD
    price_node = evaluator.add_leaf(
        id="Original_Launch_Price_299_USD",
        desc="Console’s original launch price was $299 USD.",
        parent=console_node,
        critical=True
    )
    price_claim = (
        f"The original launch price of the console {console_name} was $299 USD."
        if console_name else
        "The original launch price of the console was $299 USD."
    )

    # 6) Digital-only: no optical drive
    digital_node = evaluator.add_leaf(
        id="Digital_Only_No_Optical_Drive",
        desc="Console is digital-only and has no optical disc drive.",
        parent=console_node,
        critical=True
    )
    digital_claim = (
        f"The console {console_name} is a digital-only console and does not include an optical disc drive."
        if console_name else
        "The console is a digital-only console and does not include an optical disc drive."
    )

    # Batch verify the five factual constraints via sources (if provided)
    verify_items = [
        (
            manufacturer_claim,
            sources_list,
            manufacturer_node,
            "Verify the console on the cited page(s) is manufactured by Microsoft. "
            "Accept 'Microsoft', 'Microsoft Xbox', or equivalent phrasing."
        ),
        (
            release_claim,
            sources_list,
            release_node,
            "Confirm the console's release occurred in November 2020. "
            "If a specific date like 'November 10, 2020' is stated, that satisfies 'November 2020'."
        ),
        (
            storage_claim,
            sources_list,
            storage_node,
            "Confirm the internal storage capacity is 512 GB and that it is SSD (solid-state drive). "
            "Treat '512GB', '512 GB', or minor formatting variants as equivalent."
        ),
        (
            price_claim,
            sources_list,
            price_node,
            "Confirm the original launch MSRP was $299 USD. "
            "Allow minor variants like $299.99 to count as $299. Do not use discounted or later promotional prices."
        ),
        (
            digital_claim,
            sources_list,
            digital_node,
            "Confirm that the console is digital-only (disc-less) and explicitly lacks an optical disc drive. "
            "Statements like 'no disc drive', 'digital only', or 'disc-less' satisfy this."
        ),
    ]

    await evaluator.batch_verify(verify_items)


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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Microsoft console identification task.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation
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
    extraction: ConsoleExtraction = await evaluator.extract(
        prompt=prompt_extract_console_info(),
        template_class=ConsoleExtraction,
        extraction_name="console_extraction"
    )

    # Build verification tree and verify
    await build_and_verify_console(evaluator, extraction)

    # Return evaluation summary
    return evaluator.get_summary()