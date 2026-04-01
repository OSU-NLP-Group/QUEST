import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2 import CacheFileSys, Evaluator, VerificationNode, AggregationStrategy, LLMClient

TASK_ID = "active_stock"
TASK_DESCRIPTION = """
Find 10 U.S. exchange-listed companies with an average trading volume of over 10 million shares and a market capitalization of no more than $15 billion. For each company, provide its stock ticker symbol, CIK code, and also a webpage that display its trading volume and market cap information.
"""

EVAL_NOTES = ""
GROUND_TRUTH = {}


class CompanyNames(BaseModel):
    """List of company names extracted from the answer"""
    companies: List[str] = Field(default_factory=list, description="List of company names mentioned in the answer")


class CompanyInfo(BaseModel):
    """Detailed information for a single company"""
    name: Optional[str] = Field(default=None, description="Company name")
    ticker: Optional[str] = Field(default=None, description="Stock ticker symbol")
    cik: Optional[str] = Field(default=None, description="CIK code")
    urls: List[str] = Field(default_factory=list, description="All URLs provided for this company")


def prompt_extract_company_names() -> str:
    """Extract all company names mentioned in the answer"""
    return """
    Extract all company names mentioned in the answer as potential candidates for U.S. exchange-listed companies.

    Look for any mention of company names, whether they are presented in a list, table, or narrative format.
    Extract company names exactly as they appear in the text.
    Include all companies mentioned, even if more than 10 are listed.

    Return a list of company names in the order they appear.
    """


def prompt_extract_company_info(company_name: str) -> str:
    """Extract detailed information for a specific company"""
    return f"""
    Extract detailed information for the company "{company_name}" from the answer.

    Look for:
    - name: The company name as it appears in the answer
    - ticker: The stock ticker symbol (e.g., AAPL, MSFT)
    - cik: The CIK (Central Index Key) code
    - urls: ALL URLs provided for this company (whether they are for trading volume, market cap, or any other information about the company)

    Extract information exactly as it appears in the text.
    If any field is not mentioned, set it to null.
    For the urls field, extract ALL URLs associated with this company, regardless of what information they are meant to show.
    """


async def verify_company(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        company_name: str,
        company_info: CompanyInfo,
        company_index: int,
) -> None:
    """Verify all aspects of a single company"""

    # Create company node
    company_node = evaluator.add_parallel(
        id=f"company_{company_index}",
        desc=f"Company {company_index}: {company_name}",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial credit
    )

    # Single existence check for all required fields
    all_fields_exist = evaluator.add_custom_node(
        result=bool(
            company_info.ticker and company_info.ticker.strip() and
            company_info.cik and company_info.cik.strip() and
            company_info.urls and len(company_info.urls) > 0
        ),
        id=f"company_{company_index}_exists",
        desc=f"All required information exists for {company_name} (ticker, CIK, and URLs)",
        parent=company_node,
        critical=True,  # Critical - without basic info, other checks are meaningless
    )

    # Trading Volume Verification
    cik_node = evaluator.add_leaf(
        id=f"company_{company_index}_info_provenance",
        desc=f"CIK and ticker verified by URLs",
        parent=company_node,
        critical=True,
    )

    cik_claim = f"This page contains information about company {company_name}. And it confirms that its CIK code is {company_info.cik}"
    await evaluator.verify(
        claim=cik_claim,
        node=cik_node,
        sources=company_info.urls,
        additional_instruction="Some special acceptable cases if the webpage does not explicitly mention the CIK code: (1) it is a page of this company from the SEC website. In this case, you can assume that the CIK code is correct and give it a pass. (2) it is a page of this company, and the url contains the cik code and match the one to check"
    )










    # ********* #

    # Trading Volume Verification
    ticker_node = evaluator.add_leaf(
        id=f"company_{company_index}_info_provenance",
        desc=f"CIK and ticker verified by URLs",
        parent=company_node,
        critical=True,
    )

    ticker_claim = f"This page contains information about company {company_name}. And it confirms that its ticker symbol is {company_info.ticker}"
    await evaluator.verify(
        claim=ticker_claim,
        node=ticker_node,
        sources=company_info.urls,
        additional_instruction="Some special acceptable cases if the webpage does not explicitly mention the ticker symbol: (1) it is a page of this company from the SEC website. In this case, you can assume that the ticker symbol is correct and give it a pass. (2) it is a page of this company, and the url contains the ticker symbol and match the one to check"
    )





    # Trading Volume Verification
    volume_node = evaluator.add_leaf(
        id=f"company_{company_index}_volume_verify",
        desc=f"Trading volume > 10M shares verified by URLs",
        parent=company_node,
        critical=True,
    )

    volume_claim = f"This page confirms that {company_name} has an average daily trading volume of over 10 million shares"
    await evaluator.verify(
        claim=volume_claim,
        node=volume_node,
        sources=company_info.urls,
        additional_instruction="""
        Note that trading volume should be dynamic data. The webpage should be showing real-time information (you don't need to check whether it's from the data of exactly today tho). In other words, it should not be a static page that doesn't update, for example, some news.
        
        Look for any indication of average daily volume, trading volume, or similar metrics showing >10M shares.
        Common formats include: "10 million", "10M", "10,000,000", "15.5M shares", etc.
        
        By the way, allow minor reasonable violations (+ 0.2 million), such as "10.1 million"
        """
    )

    # Market Cap Verification
    market_cap_node = evaluator.add_leaf(
        id=f"company_{company_index}_market_cap_verify",
        desc=f"Market cap ≤ $15B verified by URLs",
        parent=company_node,
        critical=True,
    )

    market_cap_claim = f"{company_name}  has a market capitalization of no more than $15 billion"
    await evaluator.verify(
        claim=market_cap_claim,
        node=market_cap_node,
        sources=company_info.urls,
        additional_instruction="""
        Note that market cap should be dynamic data. The webpage should be showing real-time information (you don't need to check whether it's from the data of exactly today tho). In other words, it should not be a static page that doesn't update, for example, some news.
        
        Look for any indication of market cap, market value, or similar metrics showing ≤$15B.
        Common formats include: "$10 billion", "$10B", "$10,000,000,000", "$5.5B", "10B USD", etc.
        
        By the way, allow minor reasonable violations (+ 0.2 billion), such as "15.1 million"
        """
    )


async def create_placeholder_company(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        company_index: int,
) -> None:
    """Create placeholder nodes for missing companies"""

    # Create company node
    company_node = evaluator.add_parallel(
        id=f"company_{company_index}",
        desc=f"Company {company_index}: [Missing]",
        parent=parent_node,
        critical=False,
    )

    # All sub-nodes should be marked as skipped
    company_node.score = 0.0
    company_node.status = "skipped"


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                               #
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
) -> Dict[str, Any]:
    """
    Main evaluation function for active_stock task.

    Evaluates whether the answer provides 10 U.S. exchange-listed companies
    meeting the specified criteria with proper verification URLs.
    """

    # -------- 1. Initialize evaluator ----------------------------- #
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Companies are evaluated in parallel
        agent_name=agent_name,
        answer_name=answer_name,
        # Evaluator creation parameters
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # -------- 2. Extract company names first ---------------------- #
    company_names_info = await evaluator.extract(
        prompt=prompt_extract_company_names(),
        template_class=CompanyNames,
        extraction_name="company_names",
    )

    # -------- 3. Extract detailed info for each company ----------- #
    companies_to_verify = company_names_info.companies[:10]  # Only verify first 10

    for i in range(10):
        if i < len(companies_to_verify):
            company_name = companies_to_verify[i]

            # Extract detailed info for this company
            company_info = await evaluator.extract(
                prompt=prompt_extract_company_info(company_name),
                template_class=CompanyInfo,
                extraction_name=f"company_{i + 1}_info",
            )

            # Verify this company
            await verify_company(
                evaluator=evaluator,
                parent_node=root,
                company_name=company_name,
                company_info=company_info,
                company_index=i + 1,
            )
        else:
            # Create placeholder for missing company
            await create_placeholder_company(
                evaluator=evaluator,
                parent_node=root,
                company_index=i + 1,
            )

    # -------- 4. Return evaluation results ------------------------ #
    return evaluator.get_summary()