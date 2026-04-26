import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "mortgage_hq_address"
TASK_DESCRIPTION = (
    "What is the street address of the headquarters of the mortgage lender that was founded in 1985 by Dan Gilbert, "
    "was originally named Rock Financial, and is currently located in Detroit, Michigan?"
)


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class CompanyHQExtraction(BaseModel):
    """
    Structured information extracted from the agent's answer for the mortgage lender identification and HQ address.
    """
    company_name: Optional[str] = None
    founded_year: Optional[str] = None
    founder_name: Optional[str] = None
    original_name: Optional[str] = None

    headquarters_address: Optional[str] = None
    address_city: Optional[str] = None
    address_state: Optional[str] = None

    founding_sources: List[str] = Field(default_factory=list)
    original_name_sources: List[str] = Field(default_factory=list)
    address_sources: List[str] = Field(default_factory=list)
    general_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_company_hq() -> str:
    return """
    Identify the mortgage lender referenced in the answer that satisfies ALL of the following criteria:
    - Founded in 1985 by Dan Gilbert
    - Originally named "Rock Financial" when founded
    - Currently headquartered in Detroit, Michigan

    Extract the following fields STRICTLY from the answer text (do not invent information):
    1. company_name: The name of the mortgage lender identified in the answer.
    2. founded_year: The year the company was founded (as written in the answer; keep it as a string).
    3. founder_name: The name of the founder (or primary founder) as stated in the answer; use a single string. If multiple founders are listed, include Dan Gilbert prominently if present (e.g., "Dan Gilbert and others").
    4. original_name: The original name at founding (should be "Rock Financial" or close variants if that is what the answer states).
    5. headquarters_address: The complete street address of the company's headquarters as provided in the answer (e.g., "1050 Woodward Ave, Detroit, MI 48226").
    6. address_city: The city part of the HQ address (e.g., "Detroit") if explicitly present in the answer; otherwise null.
    7. address_state: The state part of the HQ address (e.g., "MI" or "Michigan") if explicitly present in the answer; otherwise null.

    Sources extraction (URLs only; do not infer or guess):
    8. founding_sources: All URLs in the answer that support the founding details (1985 and Dan Gilbert).
    9. original_name_sources: All URLs that support the original name "Rock Financial".
    10. address_sources: All URLs that support the headquarters street address.
    11. general_sources: Any other URLs mentioned in the answer that may be relevant.

    SPECIAL RULES FOR URL EXTRACTION:
    - Extract only valid full URLs (including http:// or https://). If a URL is missing protocol, prepend http://.
    - If the answer references a source without a URL (e.g., "according to Wikipedia"), do NOT invent a URL; simply omit it.
    - Return empty arrays when no URLs are provided for a category.

    If any field is missing in the answer, set it to null (for strings) or an empty list (for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(*lists: List[str]) -> List[str]:
    """Combine lists of URLs and deduplicate while preserving order."""
    seen = set()
    combined: List[str] = []
    for lst in lists:
        for url in lst:
            if url and url not in seen:
                seen.add(url)
                combined.append(url)
    return combined


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_company_identification(
    evaluator: Evaluator,
    parent_node,
    info: CompanyHQExtraction,
) -> None:
    """
    Build and verify the 'company_identification' critical parallel node:
    - founding_details: "The company was founded in 1985 by Dan Gilbert"
    - original_name: "The company was originally named Rock Financial"
    """
    company_node = evaluator.add_parallel(
        id="company_identification",
        desc="The mortgage lender identified meets all the founding and historical criteria",
        parent=parent_node,
        critical=True
    )

    # founding_details leaf
    founding_leaf = evaluator.add_leaf(
        id="founding_details",
        desc="The company was founded in 1985 by Dan Gilbert",
        parent=company_node,
        critical=True
    )
    company_part = f" {info.company_name}" if info.company_name else ""
    founding_claim = f"The mortgage lender{company_part} was founded in 1985 by Dan Gilbert."
    founding_urls = _combine_sources(info.founding_sources, info.general_sources)

    await evaluator.verify(
        claim=founding_claim,
        node=founding_leaf,
        sources=founding_urls,
        additional_instruction=(
            "Verify that the cited source(s) clearly state the company was founded in 1985 by Dan Gilbert. "
            "Accept wording such as 'founded by Dan Gilbert in 1985', 'co-founded by Dan Gilbert in 1985', "
            "or similar equivalent phrasing."
        ),
    )

    # original_name leaf
    original_name_leaf = evaluator.add_leaf(
        id="original_name",
        desc="The company was originally named Rock Financial when founded",
        parent=company_node,
        critical=True
    )
    original_claim = (
        f"The mortgage lender{company_part} was originally named 'Rock Financial' when founded."
    )
    original_urls = _combine_sources(info.original_name_sources, info.general_sources)

    await evaluator.verify(
        claim=original_claim,
        node=original_name_leaf,
        sources=original_urls,
        additional_instruction=(
            "Verify that the source(s) explicitly indicate the company's original name was Rock Financial. "
            "Allow minor variants such as 'Rock Financial Corp.' or 'Rock Financial, Inc.' so long as they "
            "clearly refer to the same original entity."
        ),
    )


async def verify_headquarters_address(
    evaluator: Evaluator,
    parent_node,
    info: CompanyHQExtraction,
) -> None:
    """
    Build and verify the 'headquarters_address' critical parallel node:
    - location_verification: "The headquarters address is located in Detroit, Michigan"
    - address_completeness: "A complete street address is provided"
    """
    hq_node = evaluator.add_parallel(
        id="headquarters_address",
        desc="The correct headquarters address is provided",
        parent=parent_node,
        critical=True
    )

    # location_verification leaf
    location_leaf = evaluator.add_leaf(
        id="location_verification",
        desc="The headquarters address is located in Detroit, Michigan",
        parent=hq_node,
        critical=True
    )
    address_str = info.headquarters_address or ""
    location_claim = f"The headquarters address '{address_str}' is located in Detroit, Michigan."
    location_urls = _combine_sources(info.address_sources, info.general_sources)

    await evaluator.verify(
        claim=location_claim,
        node=location_leaf,
        sources=location_urls,
        additional_instruction=(
            "Confirm that the provided address is in Detroit, Michigan (Detroit, MI). "
            "The supporting page should show the address and its city/state. "
            "Minor formatting differences (e.g., 'Detroit, MI 48226') are acceptable."
        ),
    )

    # address_completeness leaf (simple logical verification; no URL required)
    completeness_leaf = evaluator.add_leaf(
        id="address_completeness",
        desc="A complete street address is provided",
        parent=hq_node,
        critical=True
    )
    completeness_claim = (
        f"The provided headquarters address '{address_str}' is a complete U.S. street address that "
        f"includes a street number and name, city, state, and ZIP code."
    )

    await evaluator.verify(
        claim=completeness_claim,
        node=completeness_leaf,
        sources=None,
        additional_instruction=(
            "Judge completeness based on the address string itself. A complete U.S. street address should include "
            "a street number and street name (e.g., '1050 Woodward Ave'), city (e.g., 'Detroit'), state (e.g., 'MI' or 'Michigan'), "
            "and ZIP code (e.g., '48226'). If any of these components are missing, mark as incorrect."
        ),
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an agent's answer for identifying the correct mortgage lender (founded in 1985 by Dan Gilbert, originally Rock Financial)
    and its headquarters street address in Detroit, Michigan.
    """
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
        default_model=model,
    )

    # Extract structured info from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_company_hq(),
        template_class=CompanyHQExtraction,
        extraction_name="company_hq_extraction",
    )

    # Optional: add minimal ground-truth expectations (criteria only; no fixed company name)
    evaluator.add_ground_truth({
        "criteria": {
            "founded_year": "1985",
            "founder": "Dan Gilbert",
            "original_name": "Rock Financial",
            "hq_city_state": "Detroit, Michigan"
        },
        "note": "Verification must be supported by the URLs provided in the answer when available."
    })

    # Build verification subtrees
    await verify_company_identification(evaluator, root, extracted_info)
    await verify_headquarters_address(evaluator, root, extracted_info)

    # Return summary
    return evaluator.get_summary()