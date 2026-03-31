import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "mazda_production"
TASK_DESCRIPTION = """
What were the annual global production numbers (in units) for the Mazda3, as reported by Mazda, for each of the past ten years up to the most recently available official data?
"""

JUDGE_MODEL = "o4-mini"

# Years we need to check (latest ten years up to the most recent official data)


def determine_latest_available_year(today: Optional[datetime] = None) -> int:
    """
    Mazda releases the previous year's production figures each January 30.
    Before that date the latest official year is two years ago; on/after it,
    the latest data advances to last year.
    """
    if today is None:
        today = datetime.now()

    refresh_date = today.replace(month=1, day=30, hour=0, minute=0, second=0, microsecond=0)
    if today >= refresh_date:
        return today.year - 1
    return today.year - 2


LATEST_AVAILABLE_YEAR = determine_latest_available_year()
YEARS_SPAN = 10
START_YEAR = LATEST_AVAILABLE_YEAR - (YEARS_SPAN - 1)
TARGET_YEARS = list(range(START_YEAR, LATEST_AVAILABLE_YEAR + 1))


# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class MentionedYears(BaseModel):
    """Model for years mentioned in the answer"""
    years: List[int] = Field(default_factory=list)


class ProductionRecord(BaseModel):
    """Model for a single year's production data"""
    production_units: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_mentioned_years() -> str:
    """Extract years within the target range that are mentioned in the answer"""
    return f"""
    Extract all years between {START_YEAR} and {LATEST_AVAILABLE_YEAR} (inclusive) that are mentioned in the answer in relation to Mazda3 production numbers.

    Return these as an array of integers representing the years. Only include years where the answer specifically mentions production data for the Mazda3 in that year.

    For example, if the answer discusses Mazda3 production for 2015, 2016, and 2019, you should return [2015, 2016, 2019].

    Only include years between {START_YEAR} and {LATEST_AVAILABLE_YEAR}. If a year outside this range is mentioned, ignore it.
    """


def prompt_extract_year_production(year: int) -> str:
    """Extract production data for a specific year"""
    return f"""
    Extract the annual global production number for Mazda3 for the year {year}, as mentioned in the answer.

    Extract:
    1. The production units (as a string, exactly as it appears in the text)
    2. All URLs/sources mentioned that are associated with this year's production number

    For production units, extract the exact text from the answer. It could be:
    - A single number (e.g., "120,000")
    - Separate domestic and overseas numbers (e.g., "50,000 Domestic and 70,000 Overseas")
    - A number with text (e.g., "approximately 120,000 units")

    If production units are not clearly specified for year {year}, use null.
    If no URLs are mentioned for {year}'s data, return an empty list for urls.

    Note: Only extract information that is explicitly mentioned in the answer for year {year}. Do not infer or calculate missing values.
    """


# --------------------------------------------------------------------------- #
# Year-by-year verification functions                                         #
# --------------------------------------------------------------------------- #
async def verify_year_production(
        evaluator: Evaluator,
        parent_node,
        year: int,
        year_data: Optional[ProductionRecord],
) -> None:
    """
    Verify production data for a specific year.
    Creates a verification node for the year and adds it to the parent node.
    Always creates consistent child nodes (presence, precise, provenance) regardless of data availability.
    """
    # Create a node for this year's verification
    year_node = evaluator.add_parallel(
        id=f"year_{year}",
        desc=f"Verification of Mazda3 production data for year {year}",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial credit
    )

    # Prepare data for verification
    has_production_units = year_data is not None and year_data.production_units is not None and year_data.production_units.strip() != ""
    has_urls = year_data is not None and bool(year_data.urls)

    # 1. Combined existence check
    existence_node = evaluator.add_custom_node(
        result=has_production_units and has_urls,
        id=f"year_{year}_data_exists",
        desc=f"Check if both production data and source URLs are provided for year {year}",
        parent=year_node,
        critical=True
    )

    # 2. Precise verification - check if the production data is precise (not vague)
    precise_node = evaluator.add_leaf(
        id=f"year_{year}_precise",
        desc=f"The production data for year {year} is precise and not vague",
        parent=year_node,
        critical=True,
    )

    production_units = year_data.production_units if year_data else None
    claim = f"The production data '{production_units}' for Mazda3 in {year} is a precise number or combination of precise numbers (like Domestic + Overseas), not a vague estimate or range."
    
    await evaluator.verify(
        claim=claim,
        node=precise_node,
        additional_instruction="""
       Determine if the production number is precise or vague:

       Examples of precise data:
       - "120,000 units"
       - "50,000 Domestic and 70,000 Overseas"
       - "120,000"
       - "165,833 units"
       - "Domestic: 85,000; Overseas: 75,000"

       Examples of vague data:
       - "about 120,000 units" 
       - "over 100,000 units"
       - "between 100,000-150,000 units"
       - "decreased from previous year"
       - "approximately 120,000"
       - "around 120,000 units"
       - "more than 100,000 units"

       IMPORTANT: 
       1. If the data provides separate precise numbers for Domestic and Overseas production (like "50,000 Domestic and 70,000 Overseas"), this IS considered precise and should pass.
       2. The data is precise only if it gives exact numbers, not estimates or ranges.
       3. Words like "about", "approximately", "around", "over", "more than" indicate vague data and should fail.
       """
    )

    # 3. Provenance verification - check if cited sources support the data
    provenance_node = evaluator.add_leaf(
        id=f"year_{year}_provenance",
        desc=f"The answer cites sources for Mazda3 production data for year {year} that actually contain this information",
        parent=year_node,
        critical=True,
    )

    urls = year_data.urls if year_data else []
    claim = f"The global production number for Mazda3 in {year} was {production_units}, as reported by Mazda (This page should indicate that this data is from Mazda officially, for example, if this page is directly a webpage from Mazda official news site.)."
    
    await evaluator.verify(
        claim=claim,
        node=provenance_node,
        sources=urls,
        additional_instruction=f"""
       Verify if any of the provided webpages contain information that the Mazda3 production volume for year {year} was {production_units} units.

       IMPORTANT VALIDATION RULES:
       1. Look for annual reports, production data tables, or press releases from Mazda that mention this specific production figure.
       2. The data must be about PRODUCTION numbers (not sales figures) for the Mazda3 (which might also be called Mazda 3, Mazda-3, or Axela in some markets) in year {year}.
       3. The claimed production value '{production_units}' is considered correct if:
          - The webpage shows the exact same total number, OR
          - The webpage shows separate figures (like Domestic and Overseas) that are mentioned in the claim, OR
          - The claim provides separate figures (Domestic and Overseas) that when added match the total global figure shown on the webpage.

       Only verify the accuracy of the numbers - don't worry about the exact wording. For example, if the claim says "120,000 units" and the webpage says "120,000 vehicles produced", this is considered correct.

       Be cautious to distinguish between:
       - Production numbers (what we're looking for)
       - Sales numbers (not what we're looking for)
       - Production capacity (not what we're looking for)

       Look carefully at tables, charts, or sections specifically about production volumes or manufacturing output. Mazda often uses terms like "Domestic" for Japan production and "Overseas" for international production.
       """
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
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Set up evaluator ---------------------------------------- #
    evaluator = Evaluator()
    
    # Initialize evaluator with parallel strategy for root
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

    # -------- 2. Extract information from the answer -------------------- #
    # First, extract which years are mentioned in the answer
    mentioned_years_data = await evaluator.extract(
        prompt=prompt_extract_mentioned_years(),
        template_class=MentionedYears,
        extraction_name="mentioned_years"
    )

    # Filter to only include years in our target range
    mentioned_years = [year for year in mentioned_years_data.years if year in TARGET_YEARS]

    # For each mentioned year, extract specific production data
    year_data_map = {}
    for year in mentioned_years:
        year_data = await evaluator.extract(
            prompt=prompt_extract_year_production(year),
            template_class=ProductionRecord,
            extraction_name=f"year_{year}_production"
        )
        year_data_map[year] = year_data

    # Pad missing years with empty ProductionRecord instances
    for year in TARGET_YEARS:
        if year not in year_data_map:
            year_data_map[year] = ProductionRecord()

    # -------- 3. Build verification tree -------------------------------- #
    # Create verification nodes for each target year
    verification_tasks = []
    for year in TARGET_YEARS:
        verification_tasks.append(
            verify_year_production(
                evaluator,
                root,
                year,
                year_data_map[year]
            )
        )

    # Execute all verification tasks concurrently
    await asyncio.gather(*verification_tasks)

    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()
