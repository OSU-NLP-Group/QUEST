import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator
from mind2web2.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "startups_series_b"
TASK_DESCRIPTION = """
Could you identify a startup founded no earlier than 2022 by computer science professors that have already reached Series B funding? Please provide: 1. The company name   2. The founding year 3. The founders listed on the company's homepage 4. The academic profile or homepage of the founder who is a computer science faculty member   5. A link to a report verifying the Series B funding
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StartupInfo(BaseModel):
    """Main information about the startup extracted from the answer."""
    company_name: Optional[str] = None
    founding_year: Optional[str] = None
    founders: Optional[List[str]] = Field(default_factory=list)
    cs_professor_name: Optional[str] = None
    cs_professor_url: Optional[str] = None
    company_url: Optional[str] = None
    series_b_url: Optional[str] = None


class CompanyUrls(BaseModel):
    """URLs related to the company."""
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_startup_info() -> str:
    return """
    Extract the following information from the answer:
    1. Company name (company_name): The name of the startup mentioned in the answer
    2. Founding year (founding_year): The year when the startup was founded, as mentioned in the answer
    3. Founders (founders): A list of all founders mentioned in the answer
    4. CS professor name (cs_professor_name): The name of the founder who is a computer science professor
    5. CS professor URL (cs_professor_url): The URL to the academic profile or homepage of the CS professor founder
    6. Company URL (company_url): The URL to the company's homepage
    7. Series B URL (series_b_url): The URL to a report or article verifying the Series B funding

    For each field, if the information is not explicitly mentioned in the answer, return null.
    Return the extracted information in the specified JSON format.
    """


def prompt_extract_company_urls() -> str:
    return """
    Extract all URLs from the answer that might be related to the company's homepage, about page, team page, 
    or any other page that might contain information about the company's founding year or founders.

    Return the URLs as an array in the "urls" field.
    If no relevant URLs are found, return an empty array.
    """


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: Any,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: Any,
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
        default_model=model,
        strategy=AggregationStrategy.PARALLEL
    )

    # -------- 2. Extract structured info from the answer ----------------- #
    startup_info = await evaluator.extract(
        prompt=prompt_extract_startup_info(),
        template_class=StartupInfo,
        extraction_name="startup_info"
    )

    # Extract all potentially relevant company URLs
    company_urls_info = await evaluator.extract(
        prompt=prompt_extract_company_urls(),
        template_class=CompanyUrls,
        extraction_name="company_urls"
    )

    # Combine all potential company URLs
    all_company_urls = []
    if startup_info.company_url:
        all_company_urls.append(startup_info.company_url)
    if company_urls_info.urls:
        all_company_urls.extend(company_urls_info.urls)

    # Remove duplicates while preserving order
    all_company_urls = list(dict.fromkeys(all_company_urls))

    # Add custom info about collected URLs
    evaluator.add_custom_info(
        {"all_company_urls": all_company_urls},
        "url_collection"
    )

    # -------- 3. Critical requirements verification ---------------------- #

    # necessary info verification (critical)
    evaluator.add_custom_node(
        result=bool(startup_info.company_name) and bool(startup_info.founding_year) and bool(all_company_urls) and bool(startup_info.series_b_url),
        id="company_name",
        desc="The answer provides necessary info.",
        critical=True
    )

    # Founding year verification (critical)
    # Add actual founding year verification
    founding_year_node = evaluator.add_leaf(
        id="founding_year_check",
        desc="The company was founded in 2022 or later, as verified by the company website",
        critical=True
    )

    # Always verify founding year
    claim = f"The company {startup_info.company_name} was founded in {startup_info.founding_year}, which is 2022 or later."
    await evaluator.verify(
        claim=claim,
        node=founding_year_node,
        sources=all_company_urls,
        additional_instruction="Look for explicit or implicit mentions of the founding year. The company must have been founded in 2022 or later to meet the criteria."
    )

    # Series B funding verification (critical)
    # Add actual Series B verification
    series_b_node = evaluator.add_leaf(
        id="series_b_check",
        desc="The company has reached Series B funding, as verified by the provided funding link",
        critical=True
    )

    # Always verify Series B
    claim = f"{startup_info.company_name} has received Series B funding."
    await evaluator.verify(
        claim=claim,
        node=series_b_node,
        sources=startup_info.series_b_url,
        additional_instruction="Focus specifically on Series B funding, not Series A, seed funding, or other rounds. The page must confirm Series B funding for the company."
    )

    # -------- 4. founder and professor verification ----------- #

    # verification pipeline for founder list and academic profile
    founder_professor_pipeline = evaluator.add_parallel(
        id="founder_professor_pipeline",
        desc="Verification of founder list and CS professor academic profile",
        critical=True 
    )

    # Step 1: Verify founders list matches company homepage
    founders_verification_parent = evaluator.add_parallel(
        id="founders_verification_parent",
        desc="Verify founders list matches company homepage",
        parent=founder_professor_pipeline,
        critical=True
    )

    # Add existence check for founders data
    evaluator.add_custom_node(
        result=bool(startup_info.founders) and len(startup_info.founders) > 0,
        id="founders_data_exists",
        desc="Founders list, company name, and company URLs are provided",
        parent=founders_verification_parent,
        critical=True
    )

    # Add actual founders verification
    founders_verification_node = evaluator.add_leaf(
        id="founders_check",
        desc="The founders listed in the answer match those found on the company website",
        parent=founders_verification_parent,
        critical=True
    )

    # Always verify founders
    founders_list = ", ".join(startup_info.founders) if startup_info.founders else ""
    claim = f"The founders of {startup_info.company_name} are {founders_list}, as listed on the company's homepage."
    await evaluator.verify(
        claim=claim,
        node=founders_verification_node,
        sources=all_company_urls,
        additional_instruction="Verify that the founders mentioned in the answer match those listed on the company website. Allow name variations and different formatting."
    )

    # Step 2: Verify CS professor is one of the founders AND has valid academic profile
    cs_professor_parent = evaluator.add_parallel(
        id="cs_professor_verification_parent",
        desc="Verify CS professor founder and academic profile",
        parent=founder_professor_pipeline,
        critical=True
    )

    # Add existence check for CS professor data
    evaluator.add_custom_node(
        result=bool(startup_info.cs_professor_name) and bool(startup_info.cs_professor_url) and bool(startup_info.founders),
        id="cs_professor_data_exists",
        desc="CS professor name, URL, and founders list are provided",
        parent=cs_professor_parent,
        critical=True
    )

    # Sub-verification 1: Is this person one of the founders?
    professor_founder_node = evaluator.add_leaf(
        id="professor_is_founder",
        desc=f"{startup_info.cs_professor_name} is one of the founders",
        parent=cs_professor_parent,
        critical=True
    )

    # Check if professor name matches any founder name
    is_name_match = False
    if startup_info.cs_professor_name and startup_info.founders:
        for founder in startup_info.founders:
            if (startup_info.cs_professor_name.lower() in founder.lower() or
                    founder.lower() in startup_info.cs_professor_name.lower()):
                is_name_match = True
                break

    if is_name_match:
        # Direct name match found
        evaluator.add_custom_node(
            result=True,
            id="professor_name_match",
            desc=f"{startup_info.cs_professor_name} matches one of the listed founders",
            parent=professor_founder_node
        )
        professor_founder_node.score = 1.0
        professor_founder_node.status = "passed"
    else:
        # Verify using company URLs
        founder_claim = f"{startup_info.cs_professor_name} is one of the founders of {startup_info.company_name}."
        await evaluator.verify(
            claim=founder_claim,
            node=professor_founder_node,
            sources=all_company_urls
        )

    # Sub-verification 2: Is this person a CS professor?
    cs_professor_node = evaluator.add_leaf(
        id="cs_professor_academic",
        desc=f"{startup_info.cs_professor_name} is a CS professor with valid academic profile",
        parent=cs_professor_parent,
        critical=True
    )

    professor_claim = f"{startup_info.cs_professor_name} is a computer science professor or faculty member according to the page. And this page is a valid academic profile page of {startup_info.cs_professor_name}."
    await evaluator.verify(
        claim=professor_claim,
        node=cs_professor_node,
        sources=startup_info.cs_professor_url,
        additional_instruction="Verify that the person is explicitly identified as a professor (or faculty member with professor rank) in computer science or a very closely related field."
    )

    # -------- 5. Get final results --------------------------------------- #
    return evaluator.get_summary()