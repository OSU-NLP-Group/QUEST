import asyncio
import logging
from typing import Optional, List, Dict

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "drone_delivery"
TASK_DESCRIPTION = """
Identify two unmanned drone delivery companies that are currently operating or have officially announced plans to operate in Africa. The selected companies must originate from two different countries. For each company, clearly provide the company name, the country of origin (where it was founded), and a link to an article about their operation or plans in Africa.
"""

JUDGE_MODEL = "o4-mini"

# --------------------------------------------------------------------------- #
# Data models for extracted info                                             #
# --------------------------------------------------------------------------- #
class DroneCompany(BaseModel):
    """Information about a single drone delivery company."""
    company_name: Optional[str] = None
    country_of_origin: Optional[str] = None
    supporting_urls: List[str] = Field(default_factory=list)
    description: Optional[str] = None  # Any additional context about Africa operations


class ExtractedCompanies(BaseModel):
    """All drone delivery companies mentioned in the answer."""
    companies: List[DroneCompany] = Field(default_factory=list)


class CompanyUrls(BaseModel):
    """URLs supporting various aspects of a company."""
    urls: List[str] = Field(default_factory=list)


class AfricaArticleUrl(BaseModel):
    """The specific article URL about Africa operations."""
    article_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                         #
# --------------------------------------------------------------------------- #
def prompt_extract_companies() -> str:
    return """
    Extract all drone delivery companies mentioned in the answer that are stated to operate or have plans to operate in Africa.

    For each company, extract:
    - company_name: The exact name of the company as mentioned
    - country_of_origin: The country where the company was founded/originated
    - supporting_urls: Any URLs/links provided that discuss the company's Africa operations
    - description: Brief description of their Africa operations/plans if mentioned

    Only extract companies that are explicitly mentioned as having drone delivery operations or plans in Africa.
    If information is missing or unclear, set the field to null.
    """


def prompt_extract_all_company_urls(company_name: str) -> str:
    return f"""
    Extract ALL URLs/links from the answer that are related to {company_name} in any way.
    
    This includes URLs that:
    - Discuss the company's operations or plans in Africa
    - Provide information about the company's origin or founding
    - Contain general information about the company
    - Mention the company in any relevant context
    
    Extract complete, valid URLs including the protocol (http:// or https://).
    Include any URL that mentions or relates to {company_name}, regardless of the specific topic.
    """


def prompt_extract_africa_article_url(company_name: str) -> str:
    return f"""
    Extract the specific URL/link that is presented in the answer as the article about {company_name}'s 
    drone delivery operations or plans in Africa.
    
    The task asks for "a link to an article about their operation or plans in Africa" for each company.
    Extract only the URL that is specifically provided to fulfill this requirement.
    
    Do NOT extract general URLs about the company - only the one specifically mentioned as the 
    article about their Africa operations.
    
    Return null if no such specific article URL is provided.
    """


# --------------------------------------------------------------------------- #
# Company verification functions                                             #
# --------------------------------------------------------------------------- #
async def verify_single_company(
        evaluator: Evaluator,
        parent_node,
        company: DroneCompany,
        company_index: int,
        companies: List[DroneCompany],
) -> None:
    """Verify all requirements for a single company."""
    company_node = evaluator.add_parallel(
        id=f"company_{company_index}",
        desc=f"Company {company_index + 1} meets all requirements",
        parent=parent_node,
        critical=False  # Allow partial scoring at root level
    )

    # Extract ALL URLs related to this company (not just Africa-specific)
    all_company_urls = await evaluator.extract(
        prompt=prompt_extract_all_company_urls(company.company_name) if company.company_name else "Extract any URLs from the answer",
        template_class=CompanyUrls,
        extraction_name=f"company_{company_index}_all_urls"
    )
    
    # Combine with supporting_urls from initial extraction and remove duplicates
    all_urls = list(set(company.supporting_urls + all_company_urls.urls))

    # Extract the specific Africa article URL
    africa_article = await evaluator.extract(
        prompt=prompt_extract_africa_article_url(company.company_name) if company.company_name else "Extract the article URL about Africa operations",
        template_class=AfricaArticleUrl,
        extraction_name=f"company_{company_index}_africa_article_url"
    )

    # 1. Verify company info extraction (name, origin, links, and article URL)
    info_extracted_node = evaluator.add_custom_node(
        result=(
            company.company_name is not None and company.company_name.strip() != "" and
            company.country_of_origin is not None and company.country_of_origin.strip() != "" and
            bool(all_urls) and
            africa_article.article_url is not None  # Must have specific article URL
        ),
        id=f"company_{company_index}_info_extracted",
        desc=f"Company {company_index + 1} has extracted company name, country of origin, supporting links, and article URL",
        parent=company_node,
        critical=True
    )

    # 2. Verify company origin
    origin_node = evaluator.add_leaf(
        id=f"company_{company_index}_origin_verification",
        desc=f"Company {company_index + 1} ({company.company_name}) originates from {company.country_of_origin}",
        parent=company_node,
        critical=True
    )

    claim = f"{company.company_name} originates from {company.country_of_origin}"
    await evaluator.verify(
        claim=claim,
        node=origin_node,
        sources=all_urls,
        additional_instruction=f"Check if {company.company_name} was founded in or originates from {company.country_of_origin}. Look for information about the company's founding location, headquarters, or country of origin."
    )

    # 3. Verify Africa operations
    africa_ops_node = evaluator.add_leaf(
        id=f"company_{company_index}_africa_operations",
        desc=f"Company {company_index + 1} ({company.company_name}) has unmanned drone delivery operations or plans in Africa",
        parent=company_node,
        critical=True
    )

    claim = f"{company.company_name} has unmanned drone delivery operations or officially announced plans to operate in Africa"
    await evaluator.verify(
        claim=claim,
        node=africa_ops_node,
        sources=all_urls,
        additional_instruction="Check if the company specifically has unmanned drone delivery operations or announced plans in Africa. Look for mentions of African countries, Africa continent, or specific African drone delivery operations/plans."
    )

    # 4. Verify article about Africa operations (using only the specific article URL)
    article_node = evaluator.add_leaf(
        id=f"company_{company_index}_article_verification",
        desc=f"The provided article link discusses {company.company_name}'s operation or plans in Africa",
        parent=company_node,
        critical=True
    )

    claim = f"This page is an article about {company.company_name}'s operation or plans in Africa"
    await evaluator.verify(
        claim=claim,
        node=article_node,
        sources=[africa_article.article_url] if africa_article.article_url else [],
        additional_instruction="Check if this webpage/page is an article (news article, blog post, press release, etc.) that discusses the company's drone delivery operations or plans specifically in Africa."
    )

    # 5. Verify different country requirement for Company 2
    if company_index == 1:  # Only check for the second company
        company1 = companies[0]
        company2 = companies[1]
        
        # Verify they are different countries
        different_country_node = evaluator.add_leaf(
            id=f"company_{company_index}_different_country",
            desc=f"Company {company_index + 1} originates from a different country than Company 1",
            parent=company_node,
            critical=True  # Critical requirement
        )

        country1 = company1.country_of_origin
        country2 = company2.country_of_origin

        claim = f"'{country1}' and '{country2}' are different countries"
        await evaluator.verify(
            claim=claim,
            node=different_country_node,
            additional_instruction="Check if these represent two distinct countries. Consider variations in naming (e.g., 'USA' vs 'United States') as the same country."
        )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                #
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

    # -------- 2. Extract companies from the answer ---------------------- #
    extracted_companies = await evaluator.extract(
        prompt=prompt_extract_companies(),
        template_class=ExtractedCompanies,
        extraction_name="extracted_companies"
    )

    # -------- 3. Pad companies list to exactly 2 ----------------------- #
    companies = extracted_companies.companies[:]  # Make a copy
    while len(companies) < 2:
        companies.append(DroneCompany())  # Pad with empty companies

    # -------- 4. Verify exactly first 2 companies ---------------------- #
    for i in range(2):
        await verify_single_company(evaluator, root, companies[i], i, companies)

    # -------- 5. Return structured result ------------------------------- #
    return evaluator.get_summary()