import asyncio
import logging
import re
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ohio_statehouse_zip4"
TASK_DESCRIPTION = "What is the complete postal code (ZIP+4 format) of the Ohio Statehouse in Columbus, Ohio?"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PostalCodeExtraction(BaseModel):
    zip_plus_4: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_postal_code() -> str:
    return """
    Extract the complete postal code (ZIP+4) for the Ohio Statehouse as provided in the answer, along with any cited source URLs.

    Return a JSON object with:
    - zip_plus_4: the ZIP+4 code exactly as written in the answer (e.g., "43215-1234"). 
      If the answer only gives a 5-digit ZIP (e.g., "43215") or anything else not in ZIP+4 format, still return it verbatim.
      If no postal code is provided, return null.
    - source_urls: an array of all URLs mentioned in the answer as sources supporting the postal code for the Ohio Statehouse.
      Include full URLs (with protocol). If none are provided, return an empty array.

    Notes:
    - The Ohio Statehouse's address is commonly written as "1 Capitol Square, Columbus, OH".
    - Do not infer or invent URLs. Only include those explicitly present in the answer.
    """

# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_zip_plus_4_format(code: Optional[str]) -> bool:
    if not code or not isinstance(code, str):
        return False
    return bool(re.fullmatch(r"\d{5}-\d{4}", code.strip()))

# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_zip4_verification_tree(
    evaluator: Evaluator,
    root_node,
    extracted: PostalCodeExtraction,
) -> None:
    """
    Build and execute verification checks according to the rubric.
    """
    # Parent node: critical parallel aggregation for the three checks
    provision_node = evaluator.add_parallel(
        id="Complete_Postal_Code_Provision",
        desc="Verify that the answer provides the official ZIP+4 postal code for the Ohio Statehouse and supports it with an authoritative citation.",
        parent=root_node,
        critical=True
    )

    # 1) ZIP+4 format check (critical; direct binary via custom node)
    evaluator.add_custom_node(
        result=is_zip_plus_4_format(extracted.zip_plus_4),
        id="ZIP_Plus_4_Format",
        desc="The postal code is provided in ZIP+4 format: 5 digits, a hyphen, then 4 digits.",
        parent=provision_node,
        critical=True
    )

    # 2) Official mailing code for specified address (critical; verify by URLs if available)
    official_node = evaluator.add_leaf(
        id="Official_Mailing_Code_For_Specified_Address",
        desc="The provided ZIP+4 is the official mailing code for the Ohio Statehouse at 1 Capitol Square, Columbus, Ohio (i.e., it matches what the cited authoritative source states for that location).",
        parent=provision_node,
        critical=True
    )

    claim_official = (
        f"The ZIP+4 code for the Ohio Statehouse at 1 Capitol Square, Columbus, Ohio is '{extracted.zip_plus_4}'."
        if extracted.zip_plus_4 else
        "The ZIP+4 code for the Ohio Statehouse at 1 Capitol Square, Columbus, Ohio is provided and correct."
    )

    # If no sources, mark as failed immediately to avoid multi-URL assertion
    if not extracted.source_urls:
        official_node.score = 0.0
        official_node.status = "failed"
    else:
        await evaluator.verify(
            claim=claim_official,
            node=official_node,
            sources=extracted.source_urls,
            additional_instruction=(
                "Verify strictly that the page explicitly provides the mailing address for the Ohio Statehouse "
                "(look for 'Ohio Statehouse' and '1 Capitol Square, Columbus, OH') and shows the ZIP+4 that exactly "
                f"matches '{extracted.zip_plus_4}'. Allow minor stylistic variations such as 'Columbus, Ohio' vs 'Columbus, OH', "
                "but the ZIP+4 digits must match exactly (5 digits, hyphen, 4 digits). If the page is irrelevant or does not "
                "explicitly confirm the ZIP+4 for the Ohio Statehouse, conclude not supported."
            )
        )

    # 3) Authoritative source URL provided and supports the ZIP+4 (critical; verify by URLs)
    auth_node = evaluator.add_leaf(
        id="Authoritative_Source_URL_Provided",
        desc="A reference URL from an authoritative source (e.g., the official Ohio Statehouse website) is provided and it supports the stated ZIP+4 for the Ohio Statehouse.",
        parent=provision_node,
        critical=True
    )

    # If no sources, mark as failed immediately
    if not extracted.source_urls:
        auth_node.score = 0.0
        auth_node.status = "failed"
    else:
        claim_authoritative = (
            f"At least one of these URLs is an authoritative source (e.g., ohiostatehouse.org, ohio.gov domains like legislature.ohio.gov, or usps.com) "
            f"and it explicitly supports that the Ohio Statehouse ZIP+4 is '{extracted.zip_plus_4}'."
            if extracted.zip_plus_4 else
            "At least one of these URLs is an authoritative source (e.g., ohiostatehouse.org, ohio.gov domains like legislature.ohio.gov, or usps.com) and it explicitly shows the Ohio Statehouse ZIP+4."
        )
        await evaluator.verify(
            claim=claim_authoritative,
            node=auth_node,
            sources=extracted.source_urls,
            additional_instruction=(
                "Treat a URL as authoritative only if its domain is clearly official: "
                "• ohiostatehouse.org (official Ohio Statehouse site), "
                "• ohio.gov (and subdomains, e.g., legislature.ohio.gov), or "
                "• usps.com (United States Postal Service, including tools.usps.com). "
                "If the URL is not from these domains, return not supported. For URLs that qualify, confirm the page "
                f"explicitly presents the ZIP+4 '{extracted.zip_plus_4}' for the Ohio Statehouse. If not explicitly shown, return not supported."
            )
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
    Evaluate an answer to the Ohio Statehouse ZIP+4 task using the Mind2Web2 framework.
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

    # Extract postal code and sources from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_postal_code(),
        template_class=PostalCodeExtraction,
        extraction_name="postal_code_extraction"
    )

    # Add some custom info for visibility
    evaluator.add_custom_info(
        info={
            "extracted_zip_plus_4": extracted.zip_plus_4,
            "extracted_source_urls": extracted.source_urls
        },
        info_type="extraction_summary",
        info_name="extraction_summary"
    )

    # Build verification tree and run checks
    await build_zip4_verification_tree(evaluator, root, extracted)

    # Return the structured summary
    return evaluator.get_summary()