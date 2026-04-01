import asyncio
import logging
from typing import Optional, List, Dict, Any

import openai
from pydantic import BaseModel, Field
from datetime import datetime

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nobel_nationalities"
TASK_DESCRIPTION = """
Retrieve the list of Nobel Prize winners in Physics for each of the latest 20 available years. For each laureate, identify their nationality and place of birth. Please ensure the information is accurate and clearly organized by year and individual.
"""

JUDGE_MODEL = "o4-mini"

# Ground truth data - laureates by year
NOBEL_LAUREATES_BY_YEAR = {
    2004: ["David Gross", "Hugh David Politzer", "Frank Wilczek"],
    2005: ["Roy J. Glauber", "John L. Hall", "Theodor W. Hänsch"],
    2006: ["John C. Mather", "George Smoot"],
    2007: ["Albert Fert", "Peter Grünberg"],
    2008: ["Makoto Kobayashi", "Toshihide Maskawa", "Yoichiro Nambu"],
    2009: ["Charles K. Kao", "Willard S. Boyle", "George E. Smith"],
    2010: ["Andre Geim", "Konstantin Novoselov"],
    2011: ["Saul Perlmutter", "Brian P. Schmidt", "Adam G. Riess"],
    2012: ["Serge Haroche", "David J. Wineland"],
    2013: ["François Englert", "Peter Higgs"],
    2014: ["Isamu Akasaki", "Hiroshi Amano", "Shuji Nakamura"],
    2015: ["Takaaki Kajita", "Arthur B. McDonald"],
    2016: ["David J. Thouless", "Duncan Haldane", "John M. Kosterlitz"],
    2017: ["Rainer Weiss", "Kip Thorne", "Barry Barish"],
    2018: ["Arthur Ashkin", "Gérard Mourou", "Donna Strickland"],
    2019: ["James Peebles", "Michel Mayor", "Didier Queloz"],
    2020: ["Roger Penrose", "Reinhard Genzel", "Andrea M. Ghez"],
    2021: ["Syukuro Manabe", "Klaus Hasselmann", "Giorgio Parisi"],
    2022: ["Alain Aspect", "John Clauser", "Anton Zeilinger"],
    2023: ["Anne L'Huillier", "Ferenc Krausz", "Pierre Agostini"],
    2024: ["John Hopfield", "Geoffrey Hinton"],
    2025: ["John Clarke", "Michel H. Devoret", "John M. Martinis"]
}

AVAILABLE_YEARS = sorted(NOBEL_LAUREATES_BY_YEAR.keys())
YEARS_TO_EVALUATE = AVAILABLE_YEARS[-20:] if len(AVAILABLE_YEARS) >= 20 else AVAILABLE_YEARS

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class LaureateInfo(BaseModel):
    name: Optional[str] = None
    nationality: Optional[str] = None
    birth_place: Optional[str] = None


class LaureatesForYear(BaseModel):
    laureates: List[LaureateInfo] = Field(default_factory=list)


class LaureateSourceUrls(BaseModel):
    urls: List[str] = Field(default_factory=list)


class NameMatchResult(BaseModel):
    reasoning: str
    is_match: bool


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_laureates_for_year(year: int) -> str:
    return f"""
    Extract all Nobel Prize winners in Physics as mentioned in the answer for the year {year} from the answer.

    For each laureate mentioned for {year}, extract:
    1. Their name
    2. Their nationality (as mentioned in the answer)
    3. Their place of birth (as mentioned in the answer)

    If any information is missing for a laureate, return null for that field.
    If the answer doesn't mention any laureates for {year}, return an empty list.
    """


def prompt_extract_laureate_specific_urls(laureate_name: str, year: int) -> str:
    return f"""
    Extract all URLs provided in the answer that specifically relate to {laureate_name}.

    Look for URLs that:
    1. Specifically mention {laureate_name} in the context or description
    2. Are cited as sources for information about {laureate_name}
    3. Appear to be biographical or Nobel Prize-related pages about {laureate_name}

    Return an empty list if no relevant URLs are found.
    """


def prompt_name_matching(expected_name: str, candidate_names: List[str]) -> str:
    candidate_names_str = ", ".join([f'"{name}"' for name in candidate_names])
    return f"""
    Determine if any of the candidate names refers to the same person as the expected name.

    Expected name: "{expected_name}"
    Candidate names: {candidate_names_str}

    Consider variations in:
    - Name order (first name, last name)
    - Middle names or initials
    - Titles or honorifics
    - Minor spelling variations
    - Cultural name variations

    Return true if any candidate name clearly refers to the same person as the expected name, false otherwise.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
async def find_matching_laureate(
        expected_name: str,
        laureates_in_answer: LaureatesForYear,
        evaluator: Evaluator,
        model: str
) -> Optional[LaureateInfo]:
    """
    Find a laureate in the answer that matches the expected name using LLM-based verification.
    """
    if not laureates_in_answer.laureates:
        return None

    # Get all non-null names from the answer
    candidate_names = [l.name for l in laureates_in_answer.laureates if l.name is not None]

    if not candidate_names:
        return None

    # Use LLM to determine if any candidate matches the expected name
    match_result = await evaluator.verifier.call_llm_with_semaphore(
        model=model,
        messages=[{"role": "user", "content": prompt_name_matching(expected_name, candidate_names)}],
        response_format=NameMatchResult,
    )

    if not match_result.is_match:
        return None

    # Find the best matching laureate
    for laureate in laureates_in_answer.laureates:
        if laureate.name is not None:
            # Verify this specific pairing
            individual_match = await evaluator.verifier.call_llm_with_semaphore(
                model=model,
                messages=[{"role": "user", "content": prompt_name_matching(expected_name, [laureate.name])}],
                response_format=NameMatchResult,
            )
            if individual_match.is_match:
                return laureate

    return None


def prepare_laureates_with_urls(
        expected_laureates: List[str],
        laureates_in_answer: LaureatesForYear,
        evaluator: Evaluator
) -> List[tuple[str, Optional[LaureateInfo], LaureateSourceUrls]]:
    """
    Prepare a list of laureates with their extracted info and URLs.
    Pads missing laureates with empty objects to ensure consistent verification.
    """
    result = []
    
    for laureate_name in expected_laureates:
        # Create a placeholder entry that will be filled if found
        result.append((laureate_name, None, LaureateSourceUrls()))
    
    return result


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_year_completeness(
        evaluator: Evaluator,
        parent_node,
        year: int,
        laureates_in_answer: LaureatesForYear
) -> bool:
    """
    Verify that all required laureates for a year are mentioned in the answer.
    This is a critical check.

    Returns:
        bool: True if all laureates are mentioned, False otherwise
    """
    completeness_node = evaluator.add_leaf(
        id=f"completeness_{year}",
        desc=f"All Nobel Physics laureates for {year} are mentioned",
        parent=parent_node,
        critical=True,  # Critical - must have all laureates
    )

    # Get ground truth laureates for this year
    expected_laureates = NOBEL_LAUREATES_BY_YEAR.get(year, [])

    # Get names of laureates mentioned in the answer as a string
    mentioned_laureates_str = ", ".join([l.name for l in laureates_in_answer.laureates if l.name is not None])

    # Create a claim for verification that includes all expected laureates
    expected_laureates_str = ", ".join(expected_laureates)
    claim = f"The name list '{expected_laureates_str}' matches the name list '{mentioned_laureates_str}'."

    verified = await evaluator.verify(
        claim=claim,
        node=completeness_node,
        additional_instruction="For the names in the two lists, allow minor variations. For example:\n" +
                             "- Name order (first name, last name)\n" +
                             "- Middle names or initials\n" +
                             "- Titles or honorifics\n" +
                             "- Minor spelling variations\n" +
                             "- Cultural name variations"
    )

    return verified


async def verify_laureate(
        evaluator: Evaluator,
        parent_node,
        laureate_name: str,
        year: int,
        laureate_info: Optional[LaureateInfo],
        laureate_urls: LaureateSourceUrls
) -> None:
    """
    Verify information about a specific laureate.
    Uses unified logic for both present and missing laureates.
    """
    # Create a node for this laureate
    laureate_node = evaluator.add_parallel(
        id=f"laureate_{laureate_name.replace(' ', '_')}_{year}",
        desc=f"Information about {laureate_name}, Nobel Prize in Physics {year}",
        parent=parent_node
    )

    # Create parent node for mention verification
    mention_parent = evaluator.add_parallel(
        id=f"mention_parent_{laureate_name.replace(' ', '_')}_{year}",
        desc=f"Mention verification for {laureate_name}",
        parent=laureate_node,
        critical=False
    )

    # Add existence check for laureate info and URLs
    mention_exists = evaluator.add_custom_node(
        result=(laureate_info is not None) and bool(laureate_urls.urls),
        id=f"mention_exists_{laureate_name.replace(' ', '_')}_{year}",
        desc=f"Check if {laureate_name} is mentioned with source URLs",
        parent=mention_parent,
        critical=True
    )

    # Verify mention
    mention_node = evaluator.add_leaf(
        id=f"mention_{laureate_name.replace(' ', '_')}_{year}",
        desc=f"Verify that {laureate_name} is mentioned as a Nobel Prize winner in Physics for {year}",
        parent=mention_parent,
        critical=True
    )

    claim = f"{laureate_name} was awarded the Nobel Prize in Physics in {year}."
    await evaluator.verify(
        claim=claim,
        node=mention_node,
        sources=laureate_urls.urls,
        additional_instruction="For the laureate name, allow minor variations such as:\n" +
                             "- Name order (first name, last name)\n" +
                             "- Middle names or initials\n" +
                             "- Titles or honorifics\n" +
                             "- Minor spelling variations\n" +
                             "- Cultural name variations"
    )

    # Verify nationality
    await verify_laureate_nationality(
        evaluator=evaluator,
        parent_node=laureate_node,
        laureate_name=laureate_name,
        nationality=laureate_info.nationality if laureate_info else None,
        laureate_urls=laureate_urls.urls
    )

    # Verify birth place
    await verify_laureate_birth_place(
        evaluator=evaluator,
        parent_node=laureate_node,
        laureate_name=laureate_name,
        birth_place=laureate_info.birth_place if laureate_info else None,
        laureate_urls=laureate_urls.urls
    )


async def verify_laureate_nationality(
        evaluator: Evaluator,
        parent_node,
        laureate_name: str,
        nationality: Optional[str],
        laureate_urls: List[str]
) -> None:
    """
    Verify the laureate's nationality.
    """
    # Create parent node for nationality verification
    nationality_parent = evaluator.add_parallel(
        id=f"nationality_parent_{laureate_name.replace(' ', '_')}",
        desc=f"Nationality verification for {laureate_name}",
        parent=parent_node,
        critical=False
    )

    # Add existence check
    nationality_exists = evaluator.add_custom_node(
        result=(nationality is not None and nationality.strip() != "") and bool(laureate_urls),
        id=f"nationality_exists_{laureate_name.replace(' ', '_')}",
        desc=f"Check if nationality and source URLs are provided for {laureate_name}",
        parent=nationality_parent,
        critical=True
    )

    # Verify nationality
    nationality_node = evaluator.add_leaf(
        id=f"nationality_{laureate_name.replace(' ', '_')}",
        desc=f"Verify {laureate_name}'s nationality",
        parent=nationality_parent,
        critical=True
    )

    claim = f"{laureate_name}'s nationality is {nationality}."
    await evaluator.verify(
        claim=claim,
        node=nationality_node,
        sources=laureate_urls,
        additional_instruction="Notice, the page should have direct evidence about his/her nationality information. " +
                             "Inferring from birth place is NOT allowed! For the laureate name, you should really allow minor variations. For example:\n" +
                             "- Name order (first name, last name)\n" +
                             "- Middle names or initials\n" +
                             "- Titles or honorifics\n" +
                             "- Minor spelling variations\n" +
                             "- Cultural name variations"
    )


async def verify_laureate_birth_place(
        evaluator: Evaluator,
        parent_node,
        laureate_name: str,
        birth_place: Optional[str],
        laureate_urls: List[str]
) -> None:
    """
    Verify the laureate's place of birth.
    """
    # Create parent node for birth place verification
    birth_place_parent = evaluator.add_parallel(
        id=f"birth_place_parent_{laureate_name.replace(' ', '_')}",
        desc=f"Birth place verification for {laureate_name}",
        parent=parent_node,
        critical=False
    )

    # Add existence check
    birth_place_exists = evaluator.add_custom_node(
        result=(birth_place is not None and birth_place.strip() != "") and bool(laureate_urls),
        id=f"birth_place_exists_{laureate_name.replace(' ', '_')}",
        desc=f"Check if birth place and source URLs are provided for {laureate_name}",
        parent=birth_place_parent,
        critical=True
    )

    # Verify birth place
    birth_place_node = evaluator.add_leaf(
        id=f"birth_place_{laureate_name.replace(' ', '_')}",
        desc=f"Verify {laureate_name}'s place of birth",
        parent=birth_place_parent,
        critical=True
    )

    claim = f"{laureate_name} was born in {birth_place}."
    await evaluator.verify(
        claim=claim,
        node=birth_place_node,
        sources=laureate_urls,
        additional_instruction="For the laureate name, you should really allow minor variations. For example:\n" +
                             "- Name order (first name, last name)\n" +
                             "- Middle names or initials\n" +
                             "- Titles or honorifics\n" +
                             "- Minor spelling variations\n" +
                             "- Cultural name variations"
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: openai.AsyncAzureOpenAI,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = None
) -> Dict[str, Any]:
    """
    Evaluate a single answer and return a structured result dictionary.
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
        default_model=model if model else JUDGE_MODEL
    )

    # Track extracted data for debugging
    extracted_info_by_year = {}

    # Process each year as a separate verification unit
    for year in YEARS_TO_EVALUATE:
        # Create a node for this year using sequential strategy
        year_node = evaluator.add_sequential(
            id=f"year_{year}",
            desc=f"Information about Nobel Prize in Physics winners for {year}",
            parent=root
        )

        # Extract laureates for this year
        laureates_for_year = await evaluator.extract(
            prompt=prompt_extract_laureates_for_year(year),
            template_class=LaureatesForYear,
            extraction_name=f"laureates_{year}"
        )

        # Store for debugging
        extracted_info_by_year[year] = {
            "laureates": laureates_for_year.dict()
        }

        # Get the expected laureates for this year
        expected_laureates = NOBEL_LAUREATES_BY_YEAR.get(year, [])

        # Verify completeness first (all required laureates mentioned)
        completeness_passed = await verify_year_completeness(
            evaluator=evaluator,
            parent_node=year_node,
            year=year,
            laureates_in_answer=laureates_for_year
        )

        # Process all expected laureates uniformly, regardless of completeness check
        # The sequential strategy will automatically handle skipping if completeness failed
        for laureate_name in expected_laureates:
            # Find matching laureate info and extract URLs
            laureate_info = await find_matching_laureate(laureate_name, laureates_for_year, evaluator, model)
            
            # Extract URLs specifically for this laureate
            laureate_urls = await evaluator.extract(
                prompt=prompt_extract_laureate_specific_urls(laureate_name, year),
                template_class=LaureateSourceUrls,
                extraction_name=f"urls_for_{laureate_name}_{year}"
            )
            
            # Verify laureate with unified logic
            await verify_laureate(
                evaluator=evaluator,
                parent_node=year_node,
                laureate_name=laureate_name,
                year=year,
                laureate_info=laureate_info,
                laureate_urls=laureate_urls
            )

    # Add extracted info to custom info
    evaluator.add_custom_info(
        {"extracted_info_by_year": extracted_info_by_year},
        "extraction_results"
    )

    # Return structured result
    return evaluator.get_summary()
