import asyncio
import logging
from typing import Optional, List, Dict, Set

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.llm_client.base_client import LLMClient
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "boeing_swe_jobs"
TASK_DESCRIPTION = """
Find 5 distinct software engineering job postings at Boeing located in California, Washington, or Missouri, each in a different city. For each job, provide the direct official Boeing job listing link, job title, primary city location, Boeing Job ID, and the estimated monthly cost of living for a single person in that city. Do not list the same posting more than once.
"""

# Number of required job postings
NUM_REQUIRED_JOBS = 5


# --------------------------------------------------------------------------- #
# Data models for information extraction                                      #
# --------------------------------------------------------------------------- #
class JobTitles(BaseModel):
    """Model for extracting job titles only."""
    job_titles: List[str] = Field(default_factory=list)


class JobDetails(BaseModel):
    """Model for detailed information about a specific job."""
    job_urls: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    job_id: Optional[str] = None


class CostOfLivingInfo(BaseModel):
    """Model for cost of living information."""
    cost_of_living: Optional[str] = None
    cost_of_living_urls: List[str] = Field(default_factory=list)


class JobPosting(BaseModel):
    """Complete job posting information."""
    job_title: str = None
    job_urls: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    job_id: Optional[str] = None
    cost_of_living: Optional[str] = None
    cost_of_living_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_job_titles() -> str:
    return """
    Extract all Boeing software engineering job titles mentioned in the answer. 

    Only extract the job titles/names, nothing else. Return them as a list of strings.
    If no job titles are mentioned, return an empty list.

    Example: ["Software Engineer", "Senior Software Developer", "Application Developer"]
    """


def prompt_extract_job_details(job_title: str, job_index: int) -> str:
    return f"""
    For the job posting "{job_title}" (job #{job_index + 1}) mentioned in the answer, extract the following specific information:

    1. job_urls: ALL links/URLs mentioned for this specific job posting (extract as a list, even if just one URL)
    2. location: The location/city information provided for this specific job
    3. job_id: The Boeing Job ID number for this specific job

    Only extract information that clearly belongs to the job titled "{job_title}". 
    If any information is missing or unclear for this specific job, set it to null or empty list.

    Be very careful to distinguish between different jobs if multiple jobs are mentioned.
    """


def prompt_extract_cost_of_living(job_title: str, location: str, job_index: int) -> str:
    return f"""
    For the job "{job_title}" in location "{location}" (job #{job_index + 1}), extract cost of living information:

    1. cost_of_living: The estimated monthly cost of living amount for a single person in {location}
    2. cost_of_living_urls: ALL URLs/sources mentioned for this cost of living information (extract as a list)

    Only extract cost of living information that is specifically mentioned for {location} or this job.
    If no cost of living information is provided for this location, set both fields to null/empty.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #

def extract_city(location: str) -> str:
    """Extract city from a location string."""
    if not location:
        return ""
    # Simple extraction - take the first part before a comma if present
    parts = location.split(',')
    if len(parts) > 0:
        return parts[0].strip()
    return location.strip()


# --------------------------------------------------------------------------- #
# Job posting verification functions                                          #
# --------------------------------------------------------------------------- #
async def verify_job_posting(
        evaluator: Evaluator,
        job: JobPosting,
        job_index: int,
        all_job_locations: List[str],  # All job locations for uniqueness check
) -> None:
    """Verify a single Boeing job posting with 2-step sequential verification."""

    city = extract_city(job.location) if job.location else "Unknown location"

    # Main job node - non-critical to allow partial scoring across jobs
    job_node = evaluator.add_sequential(
        id=f"job_{job_index + 1}",
        desc=f"Job Posting #{job_index + 1}: {job.job_title} in {city}",
        critical=False
    )

    # Step 1: Find Job - All job-related verifications
    step1_node = evaluator.add_parallel(
        id=f"job_{job_index + 1}_step1_find_job",
        desc=f"Step 1: Find and verify job posting #{job_index + 1}",
        parent=job_node,
        critical=False
    )

    # 1.0 Job details existence
    job_existence_node = evaluator.add_custom_node(
        result=bool(job.job_title and job.location and job.job_urls and job.job_id),
        id=f"job_{job_index + 1}_existence",
        desc=f"Job #{job_index + 1} has required basic information (title, location, URLs)",
        parent=step1_node,
        critical=True
    )

    # 1.1 Job details verification (title, location, job_id consistency)
    job_details_node = evaluator.add_leaf(
        id=f"job_{job_index + 1}_job_details",
        desc=f"Job #{job_index + 1} title, location, and job ID are consistent with official posting",
        parent=step1_node,
        critical=True
    )

    # Combine all job detail checks into one verification
    combined_claim = f"""
    This job posting matches the following details:
    - Job Title: {job.job_title}
    - Location: {job.location}
    - Boeing Job ID: {job.job_id}

    All three pieces of information should be consistent with what appears on this official Boeing job posting.
    """

    await evaluator.verify(
        claim=combined_claim,
        node=job_details_node,
        sources=job.job_urls,
    )

    # 1.2 Unique city verification
    unique_city_node = evaluator.add_leaf(
        id=f"job_{job_index + 1}_unique_city",
        desc=f"Job #{job_index + 1} is in a city not already listed in other jobs",
        parent=step1_node,
        critical=True
    )

    # Use simple_verify to check uniqueness
    uniqueness_claim = f"""
    The job is located in {job.location}. 
    Here are all the job locations provided in the answer: {', '.join(all_job_locations)}.

    Verify that the city '{city}' appears only once in this list of locations, meaning this job is in a unique city compared to other jobs.
    """

    await evaluator.verify(
        claim=uniqueness_claim,
        node=unique_city_node,
    )

    # 1.3 Software engineering & Boeing official & valid job verification
    swe_boeing_node = evaluator.add_leaf(
        id=f"job_{job_index + 1}_swe_boeing_valid",
        desc=f"Job #{job_index + 1} is a software engineering position on official Boeing site and still valid",
        parent=step1_node,
        critical=True
    )

    combined_swe_boeing_claim = f"""
    This webpage contains:
    1. A software engineering job posting (or closely related software development role)
    2. On an official Boeing careers/jobs website
    3. The job posting is currently active and accessible (not expired or removed)

    All three conditions must be met for this verification to pass.
    """

    await evaluator.verify(
        claim=combined_swe_boeing_claim,
        node=swe_boeing_node,
        sources=job.job_urls,
        additional_instruction="Software engineering roles include titles like Software Engineer, Software Developer, Software Architect, Application Developer, etc. The job must involve writing code or software development. Boeing official sites include boeing.com, jobs.boeing.com, careers.boeing.com, etc."
    )

    # Step 2: Find Cost of Living - Only if Step 1 passes
    step2_node = evaluator.add_parallel(
        id=f"job_{job_index + 1}_step2_cost_of_living",
        desc=f"Step 2: Verify cost of living information for job #{job_index + 1}",
        parent=job_node,
        critical=False  # Non-critical to allow partial scoring
    )


    # Cost of living verification
    # Only proceed if Step 1 passed (will be handled by sequential logic)

    # existence node
    step2_existence_node = evaluator.add_custom_node(
        result=bool(job.cost_of_living and job.cost_of_living_urls and job.location),
        id=f"job_{job_index + 1}_cost_of_living_existence",
        desc=f"Job #{job_index + 1} has required cost-of-living information",
        parent=step2_node,
        critical=True
    )

    step2_verification = evaluator.add_leaf(
        id=f"job_{job_index + 1}_step2_cost_of_living_verification",
        desc=f"Job #{job_index + 1} Step 2: Verify cost of living information for job #{job_index + 1}",
        parent=step2_node,
        critical=True  # Non-critical to allow partial scoring
    )

    city_name = extract_city(job.location)
    col_claim = f"""
    The estimated monthly cost of living for a single person in {city_name}, {job.location} is {job.cost_of_living}.

    This should be:
    1. Consistent with the cost information shown on this webpage
    2. Specifically for {city_name} or the same metropolitan area
    3. For a single person (not family, couple, or multiple people)
    """

    await evaluator.verify(
        claim=col_claim,
        node=step2_verification,
        sources=job.cost_of_living_urls,
    )

# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
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
) -> Dict:
    """
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Set up evaluator ----------------------------- #
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

    # -------- 2. Step-by-step extraction -------------------------------- #

    # 2.1 First, extract just the job titles
    job_titles_info = await evaluator.extract(
        prompt=prompt_extract_job_titles(),
        template_class=JobTitles,
        extraction_name="job_titles"
    )

    # Take first 5 job titles for evaluation (following guideline about handling extra items)
    job_titles_to_evaluate = job_titles_info.job_titles[:NUM_REQUIRED_JOBS]

    # 2.2 For each job title, extract detailed information
    complete_jobs = []
    for i, job_title in enumerate(job_titles_to_evaluate):
        # Extract job details (URLs, location, job ID)
        job_details = await evaluator.extract(
            prompt=prompt_extract_job_details(job_title, i),
            template_class=JobDetails,
            extraction_name=f"job_{i+1}_details"
        )

        # Extract cost of living information
        location = job_details.location if job_details.location else "Unknown location"
        cost_info = await evaluator.extract(
            prompt=prompt_extract_cost_of_living(job_title, location, i),
            template_class=CostOfLivingInfo,
            extraction_name=f"job_{i+1}_cost"
        )

        # Combine into complete job posting
        complete_job = JobPosting(
            job_title=job_title,
            job_urls=job_details.job_urls,
            location=job_details.location,
            job_id=job_details.job_id,
            cost_of_living=cost_info.cost_of_living,
            cost_of_living_urls=cost_info.cost_of_living_urls
        )
        complete_jobs.append(complete_job)

    # Add extraction results as custom info
    evaluator.add_custom_info(
        {
            "total_job_titles_found": len(job_titles_info.job_titles),
            "job_titles_evaluated": len(job_titles_to_evaluate),
            "complete_job_details": [job.dict() for job in complete_jobs]
        },
        "extraction_summary"
    )

    # -------- 3. Build verification tree -------------------------------- #

    # Collect all job locations for uniqueness checking
    all_job_locations = [job.location for job in complete_jobs if job.location]
    evaluator.add_custom_info(
        {"all_job_locations": all_job_locations},
        "location_analysis"
    )

    # Verify each job posting
    for i, job in enumerate(complete_jobs):
        await verify_job_posting(
            evaluator=evaluator,
            job=job,
            job_index=i,
            all_job_locations=all_job_locations,
        )

    # Add placeholder nodes for missing job postings
    for i in range(len(complete_jobs), NUM_REQUIRED_JOBS):
        await verify_job_posting(
            evaluator=evaluator,
            job=JobPosting(),
            job_index=i,
            all_job_locations=all_job_locations,
        )

    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()