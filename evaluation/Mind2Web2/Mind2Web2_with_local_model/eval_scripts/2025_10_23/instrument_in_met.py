import asyncio
import logging
from typing import Optional, List, Dict, Any

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "instrument_in_met"
TASK_DESCRIPTION = """
I am currently researching string instruments from East Asia dating from the 19th century or earlier. Can you find five examples that are on display at the Met in New York City, and provide the link to each instrument's page on the Met Collection website?
"""

JUDGE_MODEL = "o4-mini"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class InstrumentName(BaseModel):
    """Model for instrument name extraction."""
    name: Optional[str]


class InstrumentNames(BaseModel):
    """Model for all extracted instrument names."""
    instruments: List[InstrumentName] = Field(default_factory=list)


class InstrumentDetail(BaseModel):
    """Model for a single instrument's details with multiple URLs."""
    name: Optional[str]
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_instrument_names() -> str:
    return """
    Please extract the names of the instruments mentioned in the answer, in the order they appear in the text.
    Return a list of instruments with their names. If fewer than 5 instruments are mentioned, just extract what is available.
    """


def prompt_extract_instrument_urls(instrument_name: str) -> str:
    return f"""
    For the instrument named "{instrument_name}" mentioned in the answer,
    please extract ALL URLs associated with this instrument that link to the Met Collection website.
    Return the name and a list of all relevant URLs in a structured format.
    If no URLs are explicitly mentioned for this instrument, return an empty list for urls.
    """


# --------------------------------------------------------------------------- #
# Instrument verification functions                                           #
# --------------------------------------------------------------------------- #
async def verify_instrument(
        evaluator: Evaluator,
        parent_node,
        instrument: InstrumentDetail,
        index: int,
) -> None:
    """
    Verify a single instrument entry according to the task requirements.
    Verifies:
    1. Whether it's an East Asian string instrument from 19th century or earlier
    2. Whether it's on display at the Met
    3. Whether any of the URLs point to the instrument's page on the Met Collection website
    """
    instrument_node = evaluator.add_parallel(
        id=f"instrument_{index}",
        desc=f"Verification of instrument #{index + 1}: {instrument.name or 'Missing instrument'}",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial credit
    )

    # Combined existence check for instrument info and URLs
    instrument_exists_node = evaluator.add_custom_node(
        result=(instrument.name is not None and instrument.name.strip() != "") and bool(instrument.urls),
        id=f"instrument_{index}_info_exists",
        desc=f"Check if instrument #{index + 1} has both name and URLs provided",
        parent=instrument_node,
        critical=True
    )

    # Verify if it's an East Asian string instrument from 19th century or earlier
    criteria_node = evaluator.add_leaf(
        id=f"instrument_{index}_criteria",
        desc=f"Instrument #{index + 1} ({instrument.name or 'Missing'}) is an East Asian string instrument from 19th century or earlier",
        parent=instrument_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"The instrument '{instrument.name or 'Missing'}' is a string instrument from East Asia (China, Japan, Korea, Taiwan, Mongolia, etc.) dating from the 19th century or earlier (before 1900).",
        node=criteria_node,
        sources=instrument.urls,
    )

    # Verify if the instrument is on display at the Met
    on_display_node = evaluator.add_leaf(
        id=f"instrument_{index}_on_display",
        desc=f"Instrument #{index + 1} ({instrument.name or 'Missing'}) is on display at the Met",
        parent=instrument_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"The instrument '{instrument.name or 'Missing'}' is currently on display at the Metropolitan Museum of Art. Look for indications such as 'On view', 'On display', gallery location information, or any other evidence that the instrument can be seen by visitors at the museum.",
        node=on_display_node,
        sources=instrument.urls,
        additional_instruction="Check the webpage for any indication that this instrument is on display at the Met. Look for phrases like 'On view', 'On display', gallery location, room numbers, or any text indicating the instrument can be physically seen at the museum. If the page says 'Not on view' or similar, the instrument is NOT on display."
    )

    # Verify if any URL is from the Met Collection website and points to this instrument
    url_node = evaluator.add_leaf(
        id=f"instrument_{index}_url",
        desc=f"At least one URL for instrument #{index + 1} ({instrument.name or 'Missing'}) is from the Met Collection website and contains information about this instrument",
        parent=instrument_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"This is a page on the Metropolitan Museum of Art's collection website that contains information about the instrument '{instrument.name or 'Missing'}'.",
        node=url_node,
        sources=instrument.urls,
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: openai.AsyncAzureOpenAI,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer to the East Asian string instruments at the Met task.

    First extracts instrument names, then for each instrument:
    1. Extracts all associated URLs
    2. Verifies if it's an East Asian string instrument from 19th century or earlier
    3. Verifies if it's on display at the Met
    4. Verifies if any URL points to the instrument's page on the Met Collection website
    """
    # -------- 1. Set up evaluator ---------------------------------------- #
    evaluator = Evaluator()
    
    # Initialize evaluator
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
        default_model=model if model else JUDGE_MODEL
    )

    # -------- 2. Extract structured info from the answer ---------------- #
    # Extract instrument names from the answer
    instrument_names = await evaluator.extract(
        prompt=prompt_extract_instrument_names(),
        template_class=InstrumentNames,
        extraction_name="instrument_names"
    )

    # Limit to first 5 instruments
    instrument_names.instruments = instrument_names.instruments[:5]

    # Extract details (including all URLs) for each instrument
    instrument_details = []
    for i, instrument in enumerate(instrument_names.instruments):
        if instrument.name:
            detail = await evaluator.extract(
                prompt=prompt_extract_instrument_urls(instrument.name),
                template_class=InstrumentDetail,
                extraction_name=f"instrument_{i+1}_details"
            )
            instrument_details.append(detail)

    # Pad with empty InstrumentDetail objects to ensure we always have 5
    while len(instrument_details) < 5:
        instrument_details.append(InstrumentDetail(name=None, urls=[]))

    # -------- 3. Build verification tree -------------------------------- #
    # Verify all 5 instruments (existing or missing)
    for i, detail in enumerate(instrument_details):
        await verify_instrument(
            evaluator=evaluator,
            parent_node=root,
            instrument=detail,
            index=i,
        )

    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()