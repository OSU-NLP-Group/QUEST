import asyncio
import logging
from typing import Optional, List, Dict, Union

from pydantic import BaseModel, Field

from mind2web2.eval_toolkit import create_evaluator
from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.evaluator import (
    Evaluator,
    AggregationStrategy,
)
from mind2web2.verification_tree import VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ai_job_seeking"
TASK_DESCRIPTION = """
I am a recent PhD graduate in AI actively seeking full-time job opportunities at OpenAI, Google DeepMind, and Meta. I am specifically interested in roles such as Research Scientist, Machine Learning Engineer, and Research Engineer (or equivalent) working on AI. My search is limited to positions based in the United States.

For each company, please identify five currently open positions for me to apply. Provide the direct links to these positions on the company's official careers website, and include the location (office base) for each.
"""

# List of target companies specified in the task
COMPANIES = ["OpenAI", "Google DeepMind", "Meta"]
POSITIONS_PER_COMPANY = 5

# DeepMind job links validation (from evaluation instructions)
DEEPMIND_URL_PREFIX = "https://job-boards.greenhouse.io/deepmind/jobs/"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class JobPosition(BaseModel):
    """Model for a single job position."""
    title: Optional[str] = None
    url: Optional[str] = None
    location: Optional[str] = None


class CompanyList(BaseModel):
    """Model for extracting company names from the answer."""
    companies: List[str] = Field(default_factory=list)


class CompanyJobs(BaseModel):
    """Model for jobs at a specific company."""
    positions: List[JobPosition] = Field(default_factory=list)


class JobTitles(BaseModel):
    """Model for extracting just job titles."""
    titles: List[Optional[str]] = Field(default_factory=list)


class SingleValue(BaseModel):
    """Model for extracting a single value (URL or location)."""
    value: Optional[str] = None


class URLList(BaseModel):
    """Model for extracting a list of URLs."""
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_companies() -> str:
    """Create prompt to extract the list of companies mentioned in the answer."""
    return """
    Extract the names of companies for which job positions are provided in the answer.
    The task specifically asked about OpenAI, Google DeepMind, and Meta.

    Return a list of company names that appear in the answer with job listings.
    """


def prompt_extract_job_titles_for_company(company: str) -> str:
    """Create prompt to extract just the job titles for a specific company."""
    return f"""
    Extract only the job titles (position names) mentioned for {company} in the answer.
    These should be AI-related positions like Research Scientist, Machine Learning Engineer, etc.

    Extract all job titles mentioned for {company}, even if there are more than five.
    Return null for any positions where the title is not clearly provided.
    """


def prompt_extract_url_for_job(company: str, job_title: str, index: int) -> str:
    """Create prompt to extract the URL for a specific job position."""
    return f"""
    For the job position #{index + 1} at {company} with title "{job_title}", 
    extract the complete URL link provided in the answer that points to this specific job posting.

    Extract only the URL for this specific position. If no URL is provided, return null.
    """


def prompt_extract_location_for_job(company: str, job_title: str, index: int) -> str:
    """Create prompt to extract the location for a specific job position."""
    return f"""
    For the job position #{index + 1} at {company} with title "{job_title}", 
    extract the location (office base) mentioned in the answer.

    Extract only the location for this specific position. If no location is provided, return null.
    """


def prompt_extract_urls_for_job(company: str, job_title: str) -> str:
    """Create prompt to extract all potential URLs that might be associated with a job."""
    return f"""
    Extract all URLs mentioned in the answer that might be related to the job position 
    "{job_title}" at {company}. These URLs should point to the job posting on the company's 
    official careers website.

    Return only the URLs as a list.
    """


def prompt_extract_job_details(company: str, job_index: int) -> str:
    """Create prompt to extract all details for a specific job position by index."""
    return f"""
    Extract the details for job position #{job_index + 1} at {company} from the answer.
    Return the following fields:
    - title: The job title (e.g., "Research Scientist", "Machine Learning Engineer")
    - url: The complete URL link to the job posting
    - location: The location mentioned for the job (e.g., "San Francisco, CA", "New York")

    Return null for any fields where the information is not clearly provided.

    Note: This should be the #{job_index + 1} job listed for {company} in the answer.
    """


# --------------------------------------------------------------------------- #
# Single verification step functions - each responsible for one verification   #
# --------------------------------------------------------------------------- #
async def verify_job_completeness(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        job_title: Optional[str],
        job_url: Optional[str],
        job_location: Optional[str],
        company: str,
        position_index: int
) -> bool:
    """
    Verify that all required information for a job position is present.
    Returns True if all information is complete, False otherwise.
    """
    node_id = f"{company.lower().replace(' ', '_')}_job_{position_index + 1}_completeness"
    
    is_complete = job_title is not None and job_url is not None and job_location is not None
    
    completeness_node = evaluator.add_custom_node(
        result=is_complete,
        id=node_id,
        desc=f"Job position #{position_index + 1} at {company} has complete information (title, URL, and location).",
        parent=parent_node,
        critical=True
    )

    return is_complete


async def verify_deepmind_url_pattern(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        job_url: Optional[str],
        position_index: int
) -> bool:
    """
    Verify that a DeepMind job URL follows the required pattern.
    Only used for DeepMind job positions.
    """
    node_id = f"deepmind_job_{position_index + 1}_url_pattern"

    # Check if URL follows the required pattern
    pattern_check = await evaluator.verify(
        claim=f"The URL '{job_url}' begins with the required prefix '{DEEPMIND_URL_PREFIX}'.",
        node=evaluator.add_leaf(
            id=node_id,
            desc=f"DeepMind job position #{position_index + 1} URL follows the required pattern starting with '{DEEPMIND_URL_PREFIX}'.",
            parent=parent_node,
            critical=True
        ),
        additional_instruction=f"Verify the URL matches this exact prefix pattern."
    )

    return pattern_check


async def verify_job_url_validity(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        job_url: Optional[str],
        company: str,
        position_index: int,
        urls_to_check: List[str]
) -> bool:
    """
    Verify that a job URL is valid and points to the company's official careers page.
    """
    node_id = f"{company.lower().replace(' ', '_')}_job_{position_index + 1}_url_validity"

    # Create appropriate verification instruction based on company
    if company == "Google DeepMind":
        additional_instruction = f"Verify this URL belongs to {company}'s official recruitment website or job board. Note that {company} uses Greenhouse (job-boards.greenhouse.io/deepmind) as their official recruitment platform, so URLs from this platform should be considered as official {company} job postings. Check for company branding, official domain names, and job posting information."
    else:
        additional_instruction = f"Verify this URL belongs to {company}'s official recruitment website or job board. Check for company branding, official domain names, and job posting information."

    # Verify URL is valid and belongs to company
    url_valid = await evaluator.verify(
        claim=f"The URL '{job_url}' is a valid job posting link from {company}'s official careers website.",
        node=evaluator.add_leaf(
            id=node_id,
            desc=f"The URL for {company} job position #{position_index + 1} is valid and points to the company's official careers website.",
            parent=parent_node,
            critical=True
        ),
        sources=urls_to_check,
        additional_instruction=additional_instruction
    )

    return url_valid


async def verify_job_title_consistency(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        job_title: Optional[str],
        company: str,
        position_index: int,
        urls_to_check: List[str]
) -> bool:
    """
    Verify that the job title from the answer matches the title in the job posting URL.
    """
    node_id = f"{company.lower().replace(' ', '_')}_job_{position_index + 1}_title_consistency"

    # Verify title consistency in job posting
    title_consistent = await evaluator.verify(
        claim=f"The job posting confirms the position title is '{job_title}' or a substantially similar title.",
        node=evaluator.add_leaf(
            id=node_id,
            desc=f"Job title '{job_title}' for {company} position #{position_index + 1} is consistent with the title shown in the job posting.",
            parent=parent_node,
            critical=True
        ),
        sources=urls_to_check,
        additional_instruction="Examine the job posting to confirm the job title matches what was stated in the answer. Allow for minor variations in formatting, but the core job title should be consistent."
    )

    return title_consistent


async def verify_job_location_consistency(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        job_location: Optional[str],
        company: str,
        position_index: int,
        urls_to_check: List[str]
) -> bool:
    """
    Verify that the job location from the answer matches the location in the job posting URL.
    """
    node_id = f"{company.lower().replace(' ', '_')}_job_{position_index + 1}_location_consistency"

    # Verify location consistency in job posting
    location_consistent = await evaluator.verify(
        claim=f"The job posting confirms the position location is '{job_location}' or a substantially similar location.",
        node=evaluator.add_leaf(
            id=node_id,
            desc=f"Job location '{job_location}' for {company} position #{position_index + 1} is consistent with the location shown in the job posting.",
            parent=parent_node,
            critical=True
        ),
        sources=urls_to_check,
        additional_instruction="Examine the job posting to confirm the job location matches what was stated in the answer. Allow for minor variations in formatting (e.g., 'San Francisco, CA' vs 'San Francisco, California'), but the core location should be consistent."
    )

    return location_consistent


async def verify_job_us_location(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        job_title: Optional[str],
        job_location: Optional[str],
        company: str,
        position_index: int,
        urls_to_check: List[str]
) -> bool:
    """
    Verify that a job position is located in the United States.
    """
    node_id = f"{company.lower().replace(' ', '_')}_job_{position_index + 1}_us_location"

    # Check if location string indicates a US location
    location_format_check = await evaluator.verify(
        claim=f"The location '{job_location}' refers to a location within the United States.",
        node=evaluator.add_leaf(
            id=node_id,
            desc=f"Job position '{job_title or 'Missing'}' (#{position_index + 1}) at {company} is located in the United States at '{job_location or 'Missing'}'.",
            parent=parent_node,
            critical=True
        ),
        additional_instruction="Verify if this location string indicates a US-based position. Look for US city names, state abbreviations, or explicit 'United States' mentions. Note that remote positions that specify 'US' or 'United States' also qualify."
    )

    return location_format_check


async def verify_job_ai_relevance(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        job_title: Optional[str],
        company: str,
        position_index: int,
        urls_to_check: List[str]
) -> bool:
    """
    Verify that a job position is related to AI research, engineering, or development.
    """
    node_id = f"{company.lower().replace(' ', '_')}_job_{position_index + 1}_ai_relevance"

    ai_node = evaluator.add_leaf(
        id=f"ai_title_{company.lower()}_{position_index}",
        desc=f"Job position '{job_title or 'Missing'}' (#{position_index + 1}) at {company} is related to AI research, engineering, or development.",
        parent=parent_node,
        critical=True
    )

    # Check if job title suggests AI relevance
    title_ai_check = await evaluator.verify(
        claim=f"The job title '{job_title}' suggests a role focused on AI, machine learning, or related technologies.",
        node=ai_node,
        additional_instruction="Determine if this job title indicates an AI-related position. Look for terms like 'Research Scientist', 'Machine Learning Engineer', 'AI', 'Neural Networks', 'Deep Learning', etc."
    )

    if title_ai_check:
        return True

    # Verify AI focus in job description
    description_ai_check = await evaluator.verify(
        claim=f"The job posting confirms this is an AI-related position involving research, engineering, or development of AI technologies.",
        node=ai_node,
        sources=urls_to_check,
        additional_instruction="Examine the job description to confirm this is an AI-focused role. Look for mentions of AI technologies, machine learning, neural networks, NLP, computer vision, or other AI-related skills and responsibilities."
    )

    return description_ai_check


# --------------------------------------------------------------------------- #
# Job position verification function                                          #
# --------------------------------------------------------------------------- #
async def verify_job_position(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        job_title: Optional[str],
        job_url: Optional[str],
        job_location: Optional[str],
        company: str,
        position_index: int,
) -> VerificationNode:
    """
    Create a node for a job position and verify all its aspects.
    All verification nodes are created upfront, and skipped status is set based on conditions.
    """
    # Create a node for this specific job position
    position_node = evaluator.add_parallel(
        id=f"{company.lower().replace(' ', '_')}_job_{position_index + 1}",
        desc=f"[Job #{position_index + 1}] {company} job position: '{job_title or 'Missing'}' located in '{job_location or 'Missing'}' meets all requirements.",
        parent=parent_node
    )

    # Step 1: Verify completeness of job information
    is_complete = await verify_job_completeness(
        evaluator, position_node, job_title, job_url, job_location, company, position_index
    )

    # Extract URLs for provenance checking (if complete)
    urls_to_check = []
    if is_complete and job_title:
        urls_extraction = await evaluator.extract(
            prompt=prompt_extract_urls_for_job(company, job_title),
            template_class=URLList,
            extraction_name=f"urls_for_{company}_{position_index}"
        )
        urls_to_check = urls_extraction.urls

        # Ensure the main URL is included
        if job_url and job_url not in urls_to_check:
            urls_to_check.append(job_url)

    # Step 2: Verify URL validity (for all companies)
    url_valid = await verify_job_url_validity(
        evaluator, position_node, job_url, company, position_index, urls_to_check
    )

    # Step 3: Verify title consistency
    title_consistent = await verify_job_title_consistency(
        evaluator, position_node, job_title, company, position_index, urls_to_check
    )

    # Step 4: Verify location consistency
    location_consistent = await verify_job_location_consistency(
        evaluator, position_node, job_location, company, position_index, urls_to_check
    )

    # Step 5: Verify US location
    us_location_valid = await verify_job_us_location(
        evaluator, position_node, job_title, job_location, company, position_index, urls_to_check
    )

    # Step 6: Verify AI relevance
    await verify_job_ai_relevance(
        evaluator, position_node, job_title, company, position_index, urls_to_check
    )

    return position_node


# --------------------------------------------------------------------------- #
# Company jobs verification function                                          #
# --------------------------------------------------------------------------- #
async def verify_company_jobs(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        company: str,
        job_titles: List[Optional[str]],
        job_urls: List[Optional[str]],
        job_locations: List[Optional[str]],
) -> VerificationNode:
    """
    Verify all job positions for a specific company.
    Creates nodes for each position and handles missing positions.
    """
    # Create a node for this company's job listings
    company_node = evaluator.add_parallel(
        id=f"{company.lower().replace(' ', '_')}_jobs",
        desc=f"Company: {company} - Identified five valid AI job positions in the United States with proper links.",
        parent=parent_node
    )

    # Determine number of positions provided (may be fewer than requested 5)
    num_positions = min(len(job_titles), POSITIONS_PER_COMPANY)

    # Verify each provided position
    for i in range(num_positions):
        title = job_titles[i] if i < len(job_titles) else None
        url = job_urls[i] if i < len(job_urls) else None
        location = job_locations[i] if i < len(job_locations) else None

        await verify_job_position(
            evaluator,
            company_node,
            title,
            url,
            location,
            company,
            i
        )

    # Create placeholder nodes for missing positions
    for i in range(num_positions, POSITIONS_PER_COMPANY):
        await verify_job_position(
            evaluator,
            company_node,
            None,
            None,
            None,
            company,
            i
        )

    return company_node


# --------------------------------------------------------------------------- #
# Extract job information functions                                           #
# --------------------------------------------------------------------------- #
async def extract_job_details_for_company(
        company: str,
        evaluator: Evaluator,
        logger: logging.Logger
) -> dict:
    """
    Extract job details for a company:
    1. First extract all job titles to determine how many positions to process
    2. Then extract complete details for each job position individually
    """
    # Step 1: Extract job titles to determine how many positions exist
    titles_extraction = await evaluator.extract(
        prompt=prompt_extract_job_titles_for_company(company),
        template_class=JobTitles,
        extraction_name=f"job_titles_{company}"
    )

    # Initialize result lists
    job_titles = []
    job_urls = []
    job_locations = []
    positions_info = []

    if titles_extraction and titles_extraction.titles:
        num_positions = len(titles_extraction.titles)
        logger.info(f"Found {num_positions} job titles for {company}")

        # Step 2: Extract complete details for each position individually
        for i in range(min(num_positions, POSITIONS_PER_COMPANY)):
            job_position = await evaluator.extract(
                prompt=prompt_extract_job_details(company, i),
                template_class=JobPosition,
                extraction_name=f"job_details_{company}_{i}"
            )

            # Store the extracted details
            job_titles.append(job_position.title)
            job_urls.append(job_position.url)
            job_locations.append(job_position.location)

            positions_info.append({
                "title": job_position.title,
                "url": job_position.url,
                "location": job_position.location
            })

            logger.info(f"Extracted details for {company} job #{i + 1}: {job_position.title}")
    else:
        logger.info(f"No job titles found for {company}")

    return {
        "company": company,
        "job_titles": job_titles,
        "job_urls": job_urls,
        "job_locations": job_locations,
        "positions_info": positions_info
    }


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client,
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

    The evaluation follows these steps:
    1. Extract companies mentioned in the answer
    2. For each company, extract job details in a stepwise manner
    3. Verify each company's job positions against all criteria
    4. Aggregate scores for a final evaluation
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
        default_model=model
    )

    # Step 1: Extract companies mentioned in the answer
    companies_extraction = await evaluator.extract(
        prompt=prompt_extract_companies(),
        template_class=CompanyList,
        extraction_name="companies_mentioned"
    )

    # Initialize list to track which target companies are mentioned
    mentioned_companies = []

    # Use LLM-based verification instead of string matching
    for company in COMPANIES:
        # Skip if this company is already confirmed to be mentioned
        if company in mentioned_companies:
            continue

        # For each extracted company name, check if it corresponds to this target company
        for extracted_company in companies_extraction.companies:
            if extracted_company:  # Skip empty entries
                is_match = await evaluator.verify(
                    claim=f"The company name '{extracted_company}' refers to {company}.",
                    node=None,
                    additional_instruction=f"Verify if the extracted company name refers to {company}, considering variations in formatting, abbreviations, or alternative names."
                )
                if is_match:
                    mentioned_companies.append(company)
                    # Once we've confirmed this target company is mentioned, no need to check other extractions
                    break

    logger.info(f"Companies mentioned in the answer: {mentioned_companies}")

    # Initialize storage for extracted information to return in results
    extracted_info = {
        "companies": []
    }

    # For each target company (whether mentioned or not)
    for company in COMPANIES:
        # Default empty data for companies not mentioned
        company_info = {
            "company": company,
            "mentioned": company in mentioned_companies,
            "positions": []
        }
        job_titles = []
        job_urls = []
        job_locations = []

        # If company was mentioned, extract job details
        if company in mentioned_companies:
            # Extract all job details for this company
            company_extraction = await extract_job_details_for_company(
                company, evaluator, logger
            )

            # Update variables with extraction results
            job_titles = company_extraction["job_titles"]
            job_urls = company_extraction["job_urls"]
            job_locations = company_extraction["job_locations"]
            company_info["positions"] = company_extraction["positions_info"]

        # Add to overall extracted info
        extracted_info["companies"].append(company_info)

        # Verify all job positions for this company
        await verify_company_jobs(
            evaluator,
            root,
            company,
            job_titles,
            job_urls,
            job_locations
        )

    # Add extracted info as custom info
    evaluator.add_custom_info(extracted_info, "extraction_results")

    # Return structured result using the new get_summary method
    return evaluator.get_summary()