import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "bls_occupations"
TASK_DESCRIPTION = """
Using the latest data from the U.S. Bureau of Labor Statistics (BLS), identify all occupations listed among the *fastest-growing occupations* that have a median annual salary greater than $120,000.

For each qualifying occupation, provide the following details (as listed on the BLS website):
- Occupation Name
- Median Annual Salary (exact value from BLS)
- Number of Jobs (exact value from BLS)

Additionally, for each identified occupation, find one currently active job listing from publicly accessible platforms (does not require login), that meets the following criteria:
- The job title clearly matches the occupation.
- Provide the direct URL to the job (not just search results).
"""

BLS_FASTEST_GROWING_URL = "https://www.bls.gov/ooh/fastest-growing.htm"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BLSOccupation(BaseModel):
    """Occupation data from BLS website"""
    name: str
    median_annual_salary: str
    number_of_jobs: Optional[str] = None
    salary_numeric: float = 0.0  # Numeric value for comparison


class BLSOccupations(BaseModel):
    """List of occupations from BLS website"""
    occupations: List[BLSOccupation] = Field(default_factory=list)


class JobListing(BaseModel):
    """Job listing information"""
    url: Optional[str] = None
    job_title: Optional[str] = None


class OccupationDetails(BaseModel):
    """Details about an occupation extracted from the answer"""
    name: Optional[str] = None
    median_annual_salary: Optional[str] = None
    number_of_jobs: Optional[str] = None
    job_listing: Optional[JobListing] = None


class OccupationNames(BaseModel):
    """List of occupation names"""
    names: List[str] = Field(default_factory=list)


class URLList(BaseModel):
    """List of URLs"""
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_bls_occupations() -> str:
    return """
    Extract all occupations from the BLS fastest-growing occupations webpage that have a median annual salary greater than $120,000.

    For each qualifying occupation, extract:
    1. The exact occupation name as shown on the website
    2. The exact median annual salary as shown on the website (including $ sign and formatting)
    3. The exact number of jobs as shown on the website (if available)

    Also, convert each median annual salary to a numeric value (removing $ and commas) and store it in the salary_numeric field.

    Return all qualifying occupations in a list.
    """


def prompt_extract_occupation_names_from_answer() -> str:
    return """
    Extract the names of all occupations that are mentioned in the answer as being among the fastest-growing occupations with median annual salary greater than $120,000.

    Return just the names of these occupations as a list of strings.
    """


def prompt_extract_occupation_details(occupation_name: str) -> str:
    return f"""
    Extract the details provided in the answer for the occupation "{occupation_name}".

    Extract:
    1. The median annual salary (including $ and formatting). if there are multiple provided, keep all of them for the extracted string. 
    2. The number of jobs (including any formatting)
    3. Any job listing information (URL and job title)

    If any field is missing, set it to null.
    Return the information in the specified JSON format.
    """


def prompt_extract_urls_for_claim(occupation_name: str, claim_type: str, claim_value: str) -> str:
    claim_description = "median annual salary" if claim_type == "salary" else "number of jobs"

    return f"""
    Extract all URLs mentioned in the answer that are used to support or provide evidence for the claim that the {claim_description} for "{occupation_name}" is {claim_value}.

    Focus only on URLs that:
    1. Are explicitly linked to this specific occupation's {claim_description} information
    2. Appear to be used as sources for this specific {claim_description} claim

    Return these URLs as a list of strings. If no relevant URLs are found, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Verification functions for occupation BLS data                              #
# --------------------------------------------------------------------------- #
async def verify_occupation_bls_data(
        evaluator: Evaluator,
        parent_node,
        answer_occupation: OccupationDetails,
        bls_occupations: BLSOccupations,
) -> None:
    """
    Verify that the occupation details match BLS data.
    """
    bls_data_node = evaluator.add_sequential(
        id=f"{answer_occupation.name}_bls_data",
        desc=f"Verify BLS data for occupation '{answer_occupation.name}'",
        parent=parent_node,
        critical=False
    )

    # Step 1: Check if this occupation is in the BLS list
    bls_list_str = ", ".join([occ.name for occ in bls_occupations.occupations])

    valid_occupation_node = evaluator.add_leaf(
        id=f"{answer_occupation.name}_valid_occupation",
        desc=f"Verify that '{answer_occupation.name}' is in this list of fastest-growing occupations with salary > $120,000",
        parent=bls_data_node,
        critical=True
    )

    # Use verifier to check if occupation matches any in the BLS list
    await evaluator.verify(
        claim=f"Is the occupation '{answer_occupation.name}' in the list of {bls_list_str}",
        node=valid_occupation_node,
    )

    # Step 2: Verify occupation details
    details_node = evaluator.add_parallel(
        id=f"{answer_occupation.name}_details",
        desc=f"Verify details for occupation '{answer_occupation.name}'",
        parent=bls_data_node,
        critical=True
    )

    # 2.1: Verify salary is provided and ≥ $120K
    salary_check_node = evaluator.add_leaf(
        id=f"{answer_occupation.name}_salary_check",
        desc=f"Verify that '{answer_occupation.name}' has median annual salary provided and ≥ $120,000",
        parent=details_node,
        critical=True
    )

    await evaluator.verify(
        claim=f"The number '{answer_occupation.median_annual_salary or 'None'}' is equal to or greater than 120,000.",
        node=salary_check_node,
        additional_instruction="Regardless of formatting. If no salary is provided, this should fail."
    )

    # 2.2: Verify number of jobs is provided
    jobs_provided_node = evaluator.add_custom_node(
        result=bool(answer_occupation.number_of_jobs),
        id=f"{answer_occupation.name}_jobs_provided",
        desc=f"Check that number of jobs is provided for '{answer_occupation.name}'",
        parent=details_node,
        critical=True
    )

    # Step 3: Verify data accuracy using source URLs
    accuracy_node = evaluator.add_parallel(
        id=f"{answer_occupation.name}_accuracy",
        desc=f"Verify accuracy of data for '{answer_occupation.name}' using source URLs",
        parent=bls_data_node,
        critical=True
    )

    # 3.1: Extract and verify salary source URLs
    salary_urls = await evaluator.extract(
        prompt=prompt_extract_urls_for_claim(
            answer_occupation.name,
            "salary",
            answer_occupation.median_annual_salary or ""
        ),
        template_class=URLList,
        extraction_name="salary_urls"
    )

    # 3.2: Verify salary accuracy
    salary_accuracy_node = evaluator.add_leaf(
        id=f"{answer_occupation.name}_salary_accuracy",
        desc=f"Verify that the salary '{answer_occupation.median_annual_salary}' for '{answer_occupation.name}' matches exactly with BLS data",
        parent=accuracy_node,
        critical=True
    )

    # Add BLS URL to salary verification if no URLs were found
    verification_urls = salary_urls.urls[:]
    if not verification_urls:
        verification_urls = [BLS_FASTEST_GROWING_URL]
    elif BLS_FASTEST_GROWING_URL not in verification_urls:
        verification_urls.append(BLS_FASTEST_GROWING_URL)

    salary_claim = f"The median annual salary for '{answer_occupation.name}' shown on BLS is '{answer_occupation.median_annual_salary or 'not provided'}'."

    await evaluator.verify(
        claim=salary_claim,
        node=salary_accuracy_node,
        sources=verification_urls,
        additional_instruction=f"Regardless of formatting. BTW, if there are multiple salary numbers provided here ({answer_occupation.median_annual_salary}), as long as any of them match the number on the page, treat it as a success."
    )

    # 3.3: Extract and verify jobs count source URLs
    jobs_urls = await evaluator.extract(
        prompt=prompt_extract_urls_for_claim(
            answer_occupation.name,
            "jobs",
            answer_occupation.number_of_jobs or ""
        ),
        template_class=URLList,
        extraction_name="jobs_urls"
    )

    # 3.4: Verify number of jobs accuracy
    jobs_accuracy_node = evaluator.add_leaf(
        id=f"{answer_occupation.name}_jobs_accuracy",
        desc=f"Verify that the number of jobs '{answer_occupation.number_of_jobs}' for '{answer_occupation.name}' matches exactly with BLS data",
        parent=accuracy_node,
        critical=True
    )

    # Add BLS URL to jobs verification if no URLs were found
    verification_urls = jobs_urls.urls[:]
    if not verification_urls:
        verification_urls = [BLS_FASTEST_GROWING_URL]
    elif BLS_FASTEST_GROWING_URL not in verification_urls:
        verification_urls.append(BLS_FASTEST_GROWING_URL)

    jobs_claim = f"The number of jobs for '{answer_occupation.name}' shown on BLS page is '{answer_occupation.number_of_jobs or 'not provided'}'."
    
    await evaluator.verify(
        claim=jobs_claim,
        node=jobs_accuracy_node,
        sources=verification_urls,
        additional_instruction="Regardless of formatting."
    )


# --------------------------------------------------------------------------- #
# Verification functions for job listing                                      #
# --------------------------------------------------------------------------- #
async def verify_job_listing(
        evaluator: Evaluator,
        parent_node,
        answer_occupation: OccupationDetails,
) -> None:
    """
    Verify that a valid job listing is provided for the occupation.
    """
    job_listing_node = evaluator.add_sequential(
        id=f"{answer_occupation.name}_job_listing",
        desc=f"Verify job listing for '{answer_occupation.name}'",
        parent=parent_node,
        critical=False  # Not critical to overall task success
    )

    # 1: Check if job listing URL exists
    url_node = evaluator.add_custom_node(
        result=bool(answer_occupation.job_listing and answer_occupation.job_listing.url),
        id=f"{answer_occupation.name}_job_url",
        desc=f"Check if job listing URL is provided for '{answer_occupation.name}'",
        parent=job_listing_node,
        critical=True
    )

    # Verify job listing criteria as parallel nodes
    criteria_node = evaluator.add_parallel(
        id=f"{answer_occupation.name}_job_criteria",
        desc=f"Verify job listing criteria for '{answer_occupation.name}'",
        parent=job_listing_node,
        critical=True
    )

    # 2: Check if job title matches occupation
    title_node = evaluator.add_leaf(
        id=f"{answer_occupation.name}_job_title_match",
        desc=f"Verify job title matches occupation '{answer_occupation.name}'",
        parent=criteria_node,
        critical=True,
    )

    job_url = answer_occupation.job_listing.url if answer_occupation.job_listing else None
    
    await evaluator.verify(
        claim=f"The webpage shows a specific job listing, not search results or a job board homepage. And, the job listing on this webpage is a good match for occupation '{answer_occupation.name}'.",
        node=title_node,
        sources=job_url,
        additional_instruction="Consider the job title a match if it roughly refers to the same occupation, even if the wording is not identical. Or this is a very suitable or clearly related job."
    )

    # 3: Check if the job listing is active (NEW - as per feedback)
    active_listing_node = evaluator.add_leaf(
        id=f"{answer_occupation.name}_job_active",
        desc=f"Verify job listing is currently active for '{answer_occupation.name}'",
        parent=criteria_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"The job listing on this webpage is currently active and accepting applications (not expired, closed, or filled).",
        node=active_listing_node,
        sources=job_url,
        additional_instruction="Look for indicators that the job is still open such as 'Apply Now' buttons, open application deadlines, or absence of 'Position Filled' or 'Expired' notices."
    )


# --------------------------------------------------------------------------- #
# Main verification function for each occupation                              #
# --------------------------------------------------------------------------- #
async def verify_occupation(
        evaluator: Evaluator,
        parent_node,
        answer_occupation: OccupationDetails,
        bls_occupations: BLSOccupations,
) -> None:
    """
    Verify occupation with two sequential steps: BLS data and job listing.
    Each step is non-critical to allow partial credit.
    """
    # Main occupation node (sequential strategy)
    occupation_node = evaluator.add_sequential(
        id=f"occupation_{answer_occupation.name}",
        desc=f"Verify occupation '{answer_occupation.name}' - both BLS data and job listing",
        parent=parent_node,
        critical=False
    )

    # Step 1: Verify BLS data (non-critical to allow partial credit)
    await verify_occupation_bls_data(
        evaluator=evaluator,
        parent_node=occupation_node,
        answer_occupation=answer_occupation,
        bls_occupations=bls_occupations,
    )

    # Step 2: Verify job listing (non-critical to allow partial credit)
    await verify_job_listing(
        evaluator=evaluator,
        parent_node=occupation_node,
        answer_occupation=answer_occupation,
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
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Set up evaluator ---------------------------------------- #
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

    # -------- 2. Extract ground truth from BLS website ------------------ #
    # 2.1 Extract occupations from BLS website with salary > $120,000
    bls_occupations = await evaluator.extract(
        prompt=prompt_extract_bls_occupations(),
        source=BLS_FASTEST_GROWING_URL,
        template_class=BLSOccupations,
        extraction_name="bls_occupations",
        use_screenshot=True
    )

    logger.info(f"Found {len(bls_occupations.occupations)} qualifying occupations from BLS website")

    # -------- 3. Extract occupation information from the answer ---------- #
    # 3.1 Extract occupation names from the answer
    answer_occupation_names = await evaluator.extract(
        prompt=prompt_extract_occupation_names_from_answer(),
        template_class=OccupationNames,
        extraction_name="answer_occupation_names"
    )

    logger.info(f"Found {len(answer_occupation_names.names)} occupations in the answer")

    # 3.2 Extract details for each occupation name
    answer_occupations = []
    for name in answer_occupation_names.names:
        # Extract basic details for the occupation
        occupation_details = await evaluator.extract(
            prompt=prompt_extract_occupation_details(name),
            template_class=OccupationDetails,
            extraction_name=f"occupation_details_{name}"
        )

        # Store the occupation details
        if occupation_details.name:  # Only add if name was successfully extracted
            answer_occupations.append(occupation_details)

    # -------- 4. Handle cases where answer has too few or too many occupations ---- #
    expected_count = len(bls_occupations.occupations)
    actual_count = len(answer_occupations)

    # If too few occupations in answer, add empty placeholders to match expected count
    if actual_count < expected_count:
        logger.info(f"Answer has {actual_count} occupations, expected {expected_count}. Adding empty placeholders.")
        for i in range(actual_count, expected_count):
            # Create empty placeholder
            placeholder = OccupationDetails(name=f"Missing Occupation #{i + 1}")
            answer_occupations.append(placeholder)

    # If too many occupations in answer, keep only the expected count
    elif actual_count > expected_count:
        logger.info(f"Answer has {actual_count} occupations, expected {expected_count}. Trimming to expected count.")
        answer_occupations = answer_occupations[:expected_count]

    # -------- 5. Build verification tree -------------------------------- #
    # Verification for each answer occupation
    for answer_occupation in answer_occupations:
        # Verify each occupation
        await verify_occupation(
            evaluator=evaluator,
            parent_node=root,
            answer_occupation=answer_occupation,
            bls_occupations=bls_occupations,
        )

    # -------- 6. Return structured result ------------------------------- #
    return evaluator.get_summary()