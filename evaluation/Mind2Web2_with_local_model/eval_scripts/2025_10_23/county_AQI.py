import asyncio
import logging
from typing import Optional, List, Dict

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator
from mind2web2.llm_client.base_client import LLMClient

from datetime import datetime

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "county_AQI"
TASK_DESCRIPTION = """
Identify three counties in the state of Pennsylvania that had more than 300 days with an Air Quality Index (AQI) rating of 0–50 ("Good") in the year 2024, according to the official Air Quality Index Report published by the U.S. Environmental Protection Agency (EPA). For each selected county, provide the following information:
* Number of "Good" AQI days recorded in 2024.
* The county's recent population within the last 10 years based on official data from the U.S. Census Bureau, explicitly stating the year.
* The county's recent Gross Domestic Product (GDP) within the last 10 years according to the Federal Reserve Bank of St. Louis, explicitly stating the year of the GDP data.
"""

# Ground truth data
QUALIFYING_COUNTIES = {
    "Franklin County": 305,
    "Tioga County": 316,
    "Wyoming County": 319,
    "Susquehanna County": 321,
    "Somerset County": 323,
    "Monroe County": 329,
    "Elk County": 343,
}

OFFICIAL_AQI_REPORT_URL = "https://www.epa.gov/outdoor-air-quality-data/air-quality-index-report"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CountyInfo(BaseModel):
    name: Optional[str] = None
    good_aqi_days: Optional[int] = None
    population: Optional[str] = None
    population_year: Optional[str] = None
    gdp: Optional[str] = None
    gdp_year: Optional[str] = None


class ExtractedCounties(BaseModel):
    counties: List[CountyInfo] = Field(default_factory=list)


class ExtractedAQIUrls(BaseModel):
    urls: List[str] = Field(default_factory=list)

class ExtractedCountyUrls(BaseModel):
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_counties() -> str:
    return """
    Extract information about the Pennsylvania counties mentioned in the answer. For each county, extract:
    - name: The exact county name as mentioned in the answer
    - good_aqi_days: The extact number of "Good" AQI days mentioned for 2024
    - population: The population figure mentioned
    - population_year: The year for which the population data is provided
    - gdp: The GDP value mentioned
    - gdp_year: The year for which the GDP data is provided

    If any information is missing for a county, set the corresponding field to null.
    Extract all counties mentioned, even if more than three are provided.
    """


def prompt_extract_aqi_urls() -> str:
    return """
    Extract all URLs that are cited as sources for Air Quality Index (AQI) data or EPA air quality reports.
    Look for URLs that appear to be from the EPA or relate to air quality data.
    """


def prompt_extract_population_urls_for_county(county_name: str) -> str:
    return f"""
    Extract all URLs that are cited as sources for population data specifically for {county_name}.
    Look for URLs that appear to be from the U.S. Census Bureau or relate to population statistics.
    Only include URLs that are explicitly associated with population data for {county_name}.
    """


def prompt_extract_gdp_urls_for_county(county_name: str) -> str:
    return f"""
    Extract all URLs that are cited as sources for GDP (Gross Domestic Product) data specifically for {county_name}.
    Look for URLs that appear to be from the Federal Reserve Bank of St. Louis or relate to economic data.
    Only include URLs that are explicitly associated with GDP data for {county_name}.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                           #
# --------------------------------------------------------------------------- #
def normalize_county_name(county_name: str) -> str:
    """Normalize county name for comparison."""
    if not county_name:
        return ""
    # Remove "County" and "PA" suffixes, strip whitespace
    name = county_name.replace("County", "").replace("PA", "").replace(",", "").strip()
    return name.title()


def is_qualifying_county(county_name: str) -> bool:
    """Check if county name matches one of the qualifying counties."""
    if not county_name:
        return False
    normalized = normalize_county_name(county_name)
    return any(normalized in qualifying.replace("County", "").strip()
               for qualifying in QUALIFYING_COUNTIES.keys())


def get_expected_aqi_days(county_name: str) -> Optional[int]:
    """Get expected AQI days for a qualifying county."""
    if not county_name:
        return None
    normalized = normalize_county_name(county_name)
    for qualifying, days in QUALIFYING_COUNTIES.items():
        if normalized in qualifying.replace("County", "").strip():
            return days
    return None


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_county_selection(
        evaluator: Evaluator,
        parent_node,
        counties: List[CountyInfo],
) -> None:
    """Verify that exactly three qualifying counties are selected."""
    
    # Check if exactly 3 counties are provided (take first 3 if more)
    selected_counties = counties[:3]
    
    # Verify each selected county is qualifying
    all_qualifying = True
    for county in selected_counties:
        if not is_qualifying_county(county.name):
            all_qualifying = False
            break
    
    # We require at least one county and all selected must be qualifying
    success = len(selected_counties) >= 1 and all_qualifying
    
    evaluator.add_custom_node(
        result=success,
        id="county_selection",
        desc="Exactly three counties are selected and all are from the qualifying list",
        parent=parent_node,
        critical=True
    )


async def verify_aqi_source(
        evaluator: Evaluator,
        parent_node,
        aqi_urls: List[str],
) -> None:
    """Verify that the official EPA AQI report URL is cited."""
    
    # Check if the official URL is present
    official_url_present = any(OFFICIAL_AQI_REPORT_URL in url for url in aqi_urls)
    
    evaluator.add_custom_node(
        result=official_url_present,
        id="aqi_source_verification",
        desc="The official EPA Air Quality Index Report URL is cited as a source",
        parent=parent_node,
        critical=True
    )


async def verify_county_aqi_data(
        evaluator: Evaluator,
        parent_node,
        county: CountyInfo,
        county_index: int,
        aqi_urls: List[str],
) -> None:
    """Verify AQI data for a single county."""
    
    # Create parent node for this county's AQI verification
    aqi_node = evaluator.add_parallel(
        id=f"county_{county_index}_aqi",
        desc=f"AQI data verification for county {county_index + 1}: {county.name or 'Unknown'}",
        parent=parent_node,
    )
    
    # Add existence check
    has_data = bool(county.name and county.good_aqi_days)
    evaluator.add_custom_node(
        result=has_data and bool(aqi_urls),
        id=f"county_{county_index}_aqi_exists",
        desc=f"County {county_index + 1} has AQI data and sources",
        parent=aqi_node,
        critical=True
    )
    
    # Correctness verification
    correctness_node = evaluator.add_leaf(
        id=f"county_{county_index}_aqi_correctness",
        desc=f"County {county_index + 1} AQI days count matches expected value from ground truth",
        parent=aqi_node,
        critical=True
    )
    
    # Verify correctness against ground truth
    expected_days = get_expected_aqi_days(county.name) if county.name else None
    
    await evaluator.verify(
        claim=f"County {county.name} had {county.good_aqi_days} days with Good AQI in 2024 (expected: {expected_days})",
        node=correctness_node,
        additional_instruction=f"The expected value according to ground truth is {expected_days} days. Accept exact matches only."
    )


async def verify_county_population_data(
        evaluator: Evaluator,
        parent_node,
        county: CountyInfo,
        county_index: int,
        population_urls: List[str],
) -> None:
    """Verify population data for a single county."""
    
    # Create parent node for this county's population verification
    population_node = evaluator.add_parallel(
        id=f"county_{county_index}_population",
        desc=f"Population data verification for county {county_index + 1}: {county.name or 'Unknown'}",
        parent=parent_node
    )
    
    # Check data completeness
    has_complete_data = bool(county.population and county.population_year)
    
    # Add existence check (covers both completeness and sources)
    evaluator.add_custom_node(
        result=has_complete_data and bool(population_urls),
        id=f"county_{county_index}_population_exists",
        desc=f"County {county_index + 1} has complete population data and sources",
        parent=population_node,
        critical=True
    )
    
    # Year requirement verification
    # Get current year and calculate 10-year range
    current_year = datetime.now().year
    min_year = current_year - 9  # 10 years including current year

    # Year requirement verification
    year_node = evaluator.add_leaf(
        id=f"county_{county_index}_population_year",
        desc=f"County {county_index + 1} population data is from within the last 10 years ({min_year}-{current_year})",
        parent=population_node,
        critical=True
    )

    await evaluator.verify(
        claim=f"The year {county.population_year} is within {min_year}-{current_year}.",
        node=year_node,
        additional_instruction=f"Check if the year is between {min_year} and {current_year} inclusive"
    )
    
    # Provenance verification
    provenance_node = evaluator.add_leaf(
        id=f"county_{county_index}_population_provenance",
        desc=f"County {county_index + 1} population data is substantiated by U.S. Census Bureau sources",
        parent=population_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"The page provides the population of {county.name} from the U.S. Census Bureau, and the population of {county.name} was {county.population} in {county.population_year}",
        node=provenance_node,
        sources=population_urls
    )


async def verify_county_gdp_data(
        evaluator: Evaluator,
        parent_node,
        county: CountyInfo,
        county_index: int,
        gdp_urls: List[str],
) -> None:
    """Verify GDP data for a single county."""
    
    # Create parent node for this county's GDP verification
    gdp_node = evaluator.add_parallel(
        id=f"county_{county_index}_gdp",
        desc=f"GDP data verification for county {county_index + 1}: {county.name or 'Unknown'}",
        parent=parent_node
    )
    
    # Check data completeness
    has_complete_data = bool(county.gdp and county.gdp_year)
    
    # Add existence check
    evaluator.add_custom_node(
        result=has_complete_data and bool(gdp_urls),
        id=f"county_{county_index}_gdp_exists",
        desc=f"County {county_index + 1} has complete GDP data and sources",
        parent=gdp_node,
        critical=True
    )
    
    # Year requirement verification

    # Get current year and calculate 10-year range
    current_year = datetime.now().year
    min_year = current_year - 9  # 10 years including current year

    # Year requirement verification
    year_node = evaluator.add_leaf(
        id=f"county_{county_index}_gdp_year",
        desc=f"County {county_index + 1} population data is from within the last 10 years ({min_year}-{current_year})",
        parent=gdp_node,
        critical=True
    )

    await evaluator.verify(
        claim=f"The year {county.population_year} is within {min_year}-{current_year}.",
        node=year_node,
        additional_instruction=f"Check if the year is between {min_year} and {current_year} inclusive"
    )
    
    
    # Provenance verification
    provenance_node = evaluator.add_leaf(
        id=f"county_{county_index}_gdp_provenance",
        desc=f"County {county_index + 1} GDP data is substantiated by Federal Reserve Bank sources",
        parent=gdp_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"The page provides the GDP of {county.name} from the Federal Reserve Bank of St. Louis, and the GDP of {county.name} was {county.gdp} in {county.gdp_year}",
        node=provenance_node,
        sources=gdp_urls
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: LLMClient,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Initialize evaluator ------------------------------------ #
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )
    
    # -------- 2. Extract structured info from the answer ----------------- #
    extracted_counties = await evaluator.extract(
        prompt=prompt_extract_counties(),
        template_class=ExtractedCounties,
        extraction_name="counties"
    )
    
    extracted_aqi_urls = await evaluator.extract(
        prompt=prompt_extract_aqi_urls(),
        template_class=ExtractedAQIUrls,
        extraction_name="aqi_urls"
    )
    
    # -------- 3. Build verification tree -------------------------------- #
    
    # AQI source verification (critical gating condition)
    await verify_aqi_source(evaluator, root, extracted_aqi_urls.urls)
    
    # Ensure we have exactly 3 counties to verify (pad with empty if needed)
    counties_to_verify = extracted_counties.counties[:3]
    while len(counties_to_verify) < 3:
        counties_to_verify.append(CountyInfo())  # Empty county
    
    # Individual county data verification
    for i, county in enumerate(counties_to_verify):

        if county.name:
            # Extract population URLs for this specific county
            population_urls_result = await evaluator.extract(
                prompt=prompt_extract_population_urls_for_county(county.name),
                template_class=ExtractedCountyUrls,
                extraction_name=f"county_{i}_population_urls"
            )
            population_urls = population_urls_result.urls
            
            # Extract GDP URLs for this specific county
            gdp_urls_result = await evaluator.extract(
                prompt=prompt_extract_gdp_urls_for_county(county.name),
                template_class=ExtractedCountyUrls,
                extraction_name=f"county_{i}_gdp_urls"
            )
            gdp_urls = gdp_urls_result.urls
        else:
            # Empty county has no URLs
            population_urls = []
            gdp_urls = []
        
        evaluator.add_custom_info(
            {
                "population_urls": population_urls,
                "gdp_urls": gdp_urls
            },
            info_type="county_population_gdp_urls",
            info_name=f"county_{i}_population_gdp_urls"
        )

        county_node = evaluator.add_parallel(
            id=f"county_{i}",
            desc=f"Verification for county {i}: {county.name or 'Missing'}",
            critical=False
        )

        evaluator.add_custom_node(
            result=is_qualifying_county(county.name),
            id=f"county_{i}_qualifying",
            desc=f"Verify whether the county is qualifying according to the ground truth",
            parent=county_node,
            critical=True
        )

        # AQI data verification
        await verify_county_aqi_data(
            evaluator, county_node, county, i, extracted_aqi_urls.urls
        )
        
        # Population data verification
        await verify_county_population_data(
            evaluator, county_node, county, i, population_urls
        )
        
        # GDP data verification
        await verify_county_gdp_data(
            evaluator, county_node, county, i, gdp_urls
        )
    
    # -------- 4. Aggregate score ---------------------------------------- #
    final_score = evaluator.score()  # triggers recursive aggregation
    
    # -------- 5. Return structured result ------------------------------- #
    return evaluator.get_summary()