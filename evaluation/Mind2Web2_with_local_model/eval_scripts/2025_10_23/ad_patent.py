import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict

from pydantic import BaseModel, Field
from mind2web2.llm_client.base_client import LLMClient

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ad_patent"
TASK_DESCRIPTION = """
Find three U.S. patents or patent applications in the field of autonomous driving that were published in the last year, regardless of whether they have been granted. For each, include the title, the inventors' names, the organization or assignee, the filing date, the publication date, and a link to the corresponding Google Patents page.
"""

LAST_YEAR = datetime.utcnow().year - 1

# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class PatentTitle(BaseModel):
    """Basic patent identification."""
    title: Optional[str] = None
    google_patents_url: Optional[str] = None


class PatentBasicInfo(BaseModel):
    """List of patent titles identified in the answer."""
    patents: List[PatentTitle] = Field(default_factory=list)


class PatentInventors(BaseModel):
    """Inventors for a specific patent."""
    inventors: List[str] = Field(default_factory=list)


class PatentAssignee(BaseModel):
    """Assignee for a specific patent."""
    assignee: Optional[str] = None


class PatentDates(BaseModel):
    """Dates for a specific patent."""
    filing_date: Optional[str] = None
    publication_date: Optional[str] = None


class PatentURLs(BaseModel):
    """URLs associated with a specific patent."""
    source_urls: List[str] = Field(default_factory=list)


class PatentInfo(BaseModel):
    """Complete information for a single patent."""
    title: Optional[str] = None
    inventors: List[str] = Field(default_factory=list)
    assignee: Optional[str] = None
    filing_date: Optional[str] = None
    publication_date: Optional[str] = None
    google_patents_url: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_basic_patent_info() -> str:
    return """
    Extract the titles and Google Patents URLs of each patent or patent application mentioned in the answer.

    For each patent, extract:
    1. title: The title of the patent
    2. google_patents_url: The URL to the Google Patents page

    Return a JSON object with a 'patents' array containing objects with these two fields.
    If any field is missing for a patent, set it to null.
    Extract all patents mentioned in the answer.
    """


def prompt_extract_inventors(patent_title: str, patent_index: int) -> str:
    return f"""
    For the patent titled "{patent_title}" (patent #{patent_index + 1} in the answer), extract the list of inventors' names.
    Return a JSON object with an 'inventors' array containing the names of all inventors mentioned for this specific patent.
    If no inventors are mentioned, return an empty array.
    """


def prompt_extract_assignee(patent_title: str, patent_index: int) -> str:
    return f"""
    For the patent titled "{patent_title}" (patent #{patent_index + 1} in the answer), extract the organization or assignee.
    Return a JSON object with an 'assignee' field containing the name of the organization or company to which the patent is assigned.
    If no assignee is mentioned, set it to null.
    """


def prompt_extract_dates(patent_title: str, patent_index: int) -> str:
    return f"""
    For the patent titled "{patent_title}" (patent #{patent_index + 1} in the answer), extract the filing date and publication date.
    Return a JSON object with:
    1. filing_date: The date when the patent was filed
    2. publication_date: The date when the patent was published

    If either date is missing, set it to null.
    """


def prompt_extract_urls(patent_title: str, patent_index: int) -> str:
    return f"""
    For the patent titled "{patent_title}", extract all source URLs mentioned for it.
    Return a JSON object with a 'source_urls' array containing all URLs that are specifically associated with this patent.
    Do not include the Google Patents URL, as it has already been extracted separately.
    If no additional source URLs are mentioned, return an empty array.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_patent(
        evaluator: Evaluator,
        parent_node,
        patent: PatentInfo,
        patent_index: int,
) -> None:
    """
    Verify all aspects of a single patent with the new tree structure.
    """
    # 1. Required Information Check (Critical)
    required_info_node = evaluator.add_custom_node(
        result=(
            patent.title is not None and 
            patent.title.strip() != "" and 
            patent.google_patents_url is not None and
            patent.google_patents_url.strip() != ""
        ),
        id=f"patent_{patent_index}_required_info",
        desc=f"Patent #{patent_index + 1} has required information (title and Google Patents URL)",
        parent=parent_node,
        critical=True
    )

    # 2. Constraints Check (Non-Critical)
    constraints_node = evaluator.add_parallel(
        id=f"patent_{patent_index}_constraints",
        desc=f"Patent #{patent_index + 1} meets all constraints",
        parent=parent_node,
        critical=False
    )

    # 2.1 Title-URL Match (Critical under constraints)
    title_url_node = evaluator.add_leaf(
        id=f"patent_{patent_index}_title_url_match",
        desc=f"Google Patents URL corresponds to the patent with the claimed title",
        parent=constraints_node,
        critical=True
    )
    
    claim = f"This Google Patents page corresponds to a patent with the title '{patent.title}'."
    await evaluator.verify(
        claim=claim,
        node=title_url_node,
        sources=patent.google_patents_url,
        additional_instruction="Check if the title on the Google Patents page matches or is equivalent to the extracted title."
    )

    # 2.2 Autonomous Driving Field (Critical under constraints)
    ad_field_node = evaluator.add_leaf(
        id=f"patent_{patent_index}_autonomous_driving",
        desc=f"Patent is in the field of autonomous driving",
        parent=constraints_node,
        critical=True
    )
    
    claim = "This patent page shows a patent that is related to autonomous driving, self-driving vehicles, or automated driving systems."
    await evaluator.verify(
        claim=claim,
        node=ad_field_node,
        sources=patent.google_patents_url,
        additional_instruction="Check the patent content, abstract, or claims to verify it relates to autonomous driving technology."
    )

    # 2.3 Published in the last year (Critical under constraints)
    published_last_year_node = evaluator.add_leaf(
        id=f"patent_{patent_index}_published_{LAST_YEAR}",
        desc=f"Patent was published in {LAST_YEAR}",
        parent=constraints_node,
        critical=True
    )
    
    claim = f"This patent was published in {LAST_YEAR}."
    await evaluator.verify(
        claim=claim,
        node=published_last_year_node,
        sources=patent.google_patents_url,
        additional_instruction=f"Check the publication date on the Google Patents page to verify it was published in {LAST_YEAR}."
    )

    # 3. Additional Information Check (Non-Critical)
    additional_info_node = evaluator.add_parallel(
        id=f"patent_{patent_index}_additional_info",
        desc=f"Patent #{patent_index + 1} additional information verification",
        parent=parent_node,
        critical=False
    )

    # 3.1 Inventors Information (Non-Critical)
    inventors_main_node = evaluator.add_parallel(
        id=f"patent_{patent_index}_inventors_main",
        desc=f"Inventors information for patent #{patent_index + 1}",
        parent=additional_info_node,
        critical=False
    )

    # 3.1.1 Inventors Provided (Critical under inventors)
    inventors_provided_node = evaluator.add_custom_node(
        result=(bool(patent.inventors) and len(patent.inventors) > 0),
        id=f"patent_{patent_index}_inventors_provided",
        desc=f"Inventors information is provided for patent #{patent_index + 1}",
        parent=inventors_main_node,
        critical=True
    )

    # 3.1.2 Inventors Accurate (Critical under inventors)
    inventors_accurate_node = evaluator.add_leaf(
        id=f"patent_{patent_index}_inventors_accurate",
        desc="Inventors are accurately cited",
        parent=inventors_main_node,
        critical=True
    )

    all_urls = [patent.google_patents_url] + patent.source_urls
    claim = f"The inventors {patent.inventors} are accurate for this patent."
    await evaluator.verify(
        claim=claim,
        node=inventors_accurate_node,
        sources=all_urls,
        additional_instruction="Verify that the listed inventors match those shown on the patent page."
    )

    # 3.2 Organization/Assignee Information (Non-Critical)
    assignee_main_node = evaluator.add_parallel(
        id=f"patent_{patent_index}_assignee_main",
        desc=f"Organization/assignee information for patent #{patent_index + 1}",
        parent=additional_info_node,
        critical=False
    )

    # 3.2.1 Assignee Provided (Critical under assignee)
    assignee_provided_node = evaluator.add_custom_node(
        result=(patent.assignee is not None and patent.assignee.strip() != ""),
        id=f"patent_{patent_index}_assignee_provided",
        desc=f"Organization/assignee information is provided for patent #{patent_index + 1}",
        parent=assignee_main_node,
        critical=True
    )

    # 3.2.2 Assignee Accurate (Critical under assignee)
    assignee_accurate_node = evaluator.add_leaf(
        id=f"patent_{patent_index}_assignee_accurate",
        desc="Assignee is accurately cited",
        parent=assignee_main_node,
        critical=True
    )

    claim = f"The assignee '{patent.assignee}' is accurate for this patent."
    await evaluator.verify(
        claim=claim,
        node=assignee_accurate_node,
        sources=all_urls,
        additional_instruction="Verify that the assignee/organization matches what is shown on the patent page."
    )

    # 3.3 Filing Date Information (Non-Critical)
    filing_date_main_node = evaluator.add_parallel(
        id=f"patent_{patent_index}_filing_date_main",
        desc=f"Filing date information for patent #{patent_index + 1}",
        parent=additional_info_node,
        critical=False
    )

    # 3.3.1 Filing Date Provided (Critical under filing date)
    filing_date_provided_node = evaluator.add_custom_node(
        result=(patent.filing_date is not None and patent.filing_date.strip() != ""),
        id=f"patent_{patent_index}_filing_date_provided",
        desc=f"Filing date information is provided for patent #{patent_index + 1}",
        parent=filing_date_main_node,
        critical=True
    )

    # 3.3.2 Filing Date Accurate (Critical under filing date)
    filing_date_accurate_node = evaluator.add_leaf(
        id=f"patent_{patent_index}_filing_date_accurate",
        desc="Filing date is accurately cited",
        parent=filing_date_main_node,
        critical=True
    )

    claim = f"The filing date '{patent.filing_date}' is accurate for this patent."
    await evaluator.verify(
        claim=claim,
        node=filing_date_accurate_node,
        sources=all_urls,
        additional_instruction="Verify that the filing date matches what is shown on the patent page."
    )

    # 3.4 Publication Date Information (Non-Critical)
    pub_date_main_node = evaluator.add_parallel(
        id=f"patent_{patent_index}_pub_date_main",
        desc=f"Publication date information for patent #{patent_index + 1}",
        parent=additional_info_node,
        critical=False
    )

    # 3.4.1 Publication Date Provided (Critical under publication date)
    pub_date_provided_node = evaluator.add_custom_node(
        result=(patent.publication_date is not None and patent.publication_date.strip() != ""),
        id=f"patent_{patent_index}_pub_date_provided",
        desc=f"Publication date information is provided for patent #{patent_index + 1}",
        parent=pub_date_main_node,
        critical=True
    )

    # 3.4.2 Publication Date Accurate (Critical under publication date)
    pub_date_accurate_node = evaluator.add_leaf(
        id=f"patent_{patent_index}_pub_date_accurate",
        desc="Publication date is accurately cited",
        parent=pub_date_main_node,
        critical=True
    )

    claim = f"The publication date '{patent.publication_date}' is accurate for this patent."
    await evaluator.verify(
        claim=claim,
        node=pub_date_accurate_node,
        sources=all_urls,
        additional_instruction="Verify that the publication date matches what is shown on the patent page."
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
) -> Dict:
    """
    Evaluate a single answer for the find_ad_patent task and return a structured result dictionary.
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
        default_model=model
    )

    # -------- 2. Extract structured info from the answer ---------------- #
    # Extract basic patent information (titles and Google Patents URLs)
    basic_patent_info = await evaluator.extract(
        prompt=prompt_extract_basic_patent_info(),
        template_class=PatentBasicInfo,
        extraction_name="basic_patent_info"
    )

    # Initialize a list to store complete patent information
    complete_patents = []

    # Extract additional details for each patent
    for i, basic_patent in enumerate(basic_patent_info.patents[:3]):  # Limit to first 3 patents
        if not basic_patent.title:
            logger.warning(f"Patent #{i + 1} missing title, skipping detailed extraction")
            # Add empty patent for padding
            complete_patents.append(PatentInfo())
            continue

        # Create a complete patent info object
        patent = PatentInfo(
            title=basic_patent.title,
            google_patents_url=basic_patent.google_patents_url,
        )

        # Extract inventors
        inventors_info = await evaluator.extract(
            prompt=prompt_extract_inventors(patent.title, i),
            template_class=PatentInventors,
            extraction_name=f"patent_{i}_inventors"
        )
        patent.inventors = inventors_info.inventors

        # Extract assignee
        assignee_info = await evaluator.extract(
            prompt=prompt_extract_assignee(patent.title, i),
            template_class=PatentAssignee,
            extraction_name=f"patent_{i}_assignee"
        )
        patent.assignee = assignee_info.assignee

        # Extract dates
        dates_info = await evaluator.extract(
            prompt=prompt_extract_dates(patent.title, i),
            template_class=PatentDates,
            extraction_name=f"patent_{i}_dates"
        )
        patent.filing_date = dates_info.filing_date
        patent.publication_date = dates_info.publication_date

        # Extract additional source URLs
        urls_info = await evaluator.extract(
            prompt=prompt_extract_urls(patent.title, i),
            template_class=PatentURLs,
            extraction_name=f"patent_{i}_urls"
        )
        patent.source_urls = urls_info.source_urls

        # Add the complete patent info to our list
        complete_patents.append(patent)

    # Pad the list to ensure we have exactly 3 patents
    while len(complete_patents) < 3:
        complete_patents.append(PatentInfo())

    # -------- 3. Build verification tree with new structure ------------- #
    # Add sequential nodes for each of the 3 required patents
    for i in range(3):
        patent_node = evaluator.add_sequential(
            id=f"patent_{i}",
            desc=f"Patent #{i + 1} verification",
            parent=root,
            critical=False,  # Allow partial scoring between patents
        )

        # Get the patent
        patent = complete_patents[i]

        # Verify all aspects of the patent with new structure
        await verify_patent(evaluator, patent_node, patent, i)

    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()
