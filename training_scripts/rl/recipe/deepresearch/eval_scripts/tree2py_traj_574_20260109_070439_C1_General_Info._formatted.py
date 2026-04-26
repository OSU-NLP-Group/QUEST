import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ohio_averys_law_primary_sponsor"
TASK_DESCRIPTION = (
    "On December 19, 2025, Ohio Governor Mike DeWine signed a bill known as 'Avery's Law' "
    "that makes changes to the laws governing dogs, including dangerous and vicious dogs. "
    "What is the full name of the primary sponsor of this bill?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AnswerExtraction(BaseModel):
    """
    Core information we need to verify the answer for Avery's Law (HB 247).
    """
    sponsor_full_name: Optional[str] = None
    bill_number_text: Optional[str] = None  # e.g., "HB 247" or "House Bill 247"
    mentions_averys_law: Optional[bool] = None  # Whether "Avery's Law" is explicitly mentioned
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_core() -> str:
    return """
    Extract the following fields from the answer text about Ohio's 'Avery’s Law':
    1) sponsor_full_name: The full name of the primary sponsor of the bill discussed (at least first and last name). If the answer provides only a last name or ambiguous name, set this to null. Include any honorifics or titles (e.g., "Rep.", "Representative") only if given alongside the full name.
    2) bill_number_text: The exact textual mention of the bill number if present (e.g., "HB 247", "House Bill 247"). If the answer references a different bill or does not include a bill number, set this to null.
    3) mentions_averys_law: A boolean flag. Set to true only if the answer explicitly refers to the bill/event as “Avery’s Law” (allow reasonable punctuation variants like Avery’s/Avery's).
    4) source_urls: List all URLs explicitly provided in the answer (including markdown links). Return only valid URLs. If none are provided, return an empty list.

    Important:
    - Do not invent information. Extract strictly from the answer.
    - If sponsor_full_name is present but includes more than one possible person, choose the one explicitly described as the primary sponsor of HB 247/Avery’s Law.
    - For bill_number_text, prefer the exact string provided (e.g., "HB 247").
    - For source_urls, include all URLs as they appear, even if not official.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
OFFICIAL_OHIO_HOSTS = {
    "governor.ohio.gov",
    "ohiohouse.gov",
    "ohiosenate.gov",
    "ohiogeneralassembly.gov",
    "legislature.ohio.gov",
    "codes.ohio.gov",
    "ohio.gov",
}


def is_official_ohio_url(url: str) -> bool:
    """
    Determine whether a URL belongs to an official Ohio government source.
    Accepts *.ohio.gov and specific legislative/government hosts.
    """
    try:
        parsed = urlparse(url.strip())
        host = (parsed.netloc or "").lower()
        if not host:
            return False
        # Exact host match
        if host in OFFICIAL_OHIO_HOSTS:
            return True
        # Subdomains of *.ohio.gov (e.g., www.ohio.gov, governor.ohio.gov, something.ohio.gov)
        if host.endswith(".ohio.gov"):
            return True
        return False
    except Exception:
        return False


def dedupe_preserve_order(urls: List[str]) -> List[str]:
    """
    Deduplicate URLs preserving original order.
    """
    seen = set()
    result = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    root_node,
    extraction: AnswerExtraction,
) -> None:
    """
    Build the verification tree based on the rubric and run verifications.
    """

    # Create the main critical parent node that aggregates all constraints in parallel
    primary_node = evaluator.add_parallel(
        id="Primary_Sponsor_Identification",
        desc="Verify the response identifies the primary sponsor of the specified bill/event and satisfies all stated constraints.",
        parent=root_node,
        critical=True,
    )

    # 1) The answer explicitly identifies the bill as House Bill 247 and refers to it as “Avery’s Law.”
    bill_id_leaf = evaluator.add_leaf(
        id="Bill_Identified_As_HB247_And_Averys_Law",
        desc="The answer explicitly identifies the bill as House Bill 247 and refers to it as “Avery’s Law.”",
        parent=primary_node,
        critical=True,
    )
    bill_claim = (
        "The answer explicitly identifies the bill as House Bill 247 (or HB 247) and refers to it as 'Avery’s Law'. "
        "Minor variations in punctuation or casing should be accepted."
    )
    await evaluator.verify(
        claim=bill_claim,
        node=bill_id_leaf,
        additional_instruction=(
            "Check the answer text itself. Accept reasonable variants such as 'HB 247' for House Bill 247 and "
            "'Avery's Law' or 'Avery’s Law'. Both the bill number and the 'Avery’s Law' moniker must be present."
        ),
    )

    # 2) The answer provides the primary sponsor’s full name (at minimum first and last name).
    sponsor_provided_leaf = evaluator.add_leaf(
        id="Primary_Sponsor_Full_Name_Provided",
        desc="The answer provides the primary sponsor’s full name (at minimum first and last name).",
        parent=primary_node,
        critical=True,
    )
    sponsor_claim = (
        "The answer provides the primary sponsor’s full name (at least first and last name). "
        f"If present, the extracted name is: {extraction.sponsor_full_name!r}."
    )
    await evaluator.verify(
        claim=sponsor_claim,
        node=sponsor_provided_leaf,
        additional_instruction=(
            "Judge purely from the answer text. Titles such as 'Rep.' or 'Representative' are fine, "
            "but there must be a clear first and last name of the primary sponsor."
        ),
    )

    # 3) The answer provides at least one official Ohio government source URL.
    all_urls = dedupe_preserve_order(extraction.source_urls or [])
    official_urls = [u for u in all_urls if is_official_ohio_url(u)]
    evaluator.add_custom_info(
        {"all_extracted_urls": all_urls, "recognized_official_urls": official_urls},
        info_type="url_extraction",
        info_name="url_extraction_summary",
    )

    official_source_node = evaluator.add_custom_node(
        result=len(official_urls) > 0,
        id="Official_Ohio_Government_Source_URL_Provided",
        desc="The answer provides at least one reference URL from an official Ohio government source.",
        parent=primary_node,
        critical=True,
    )

    # 4) Official source confirms signing and subject context (date, location, governor, subject area)
    context_leaf = evaluator.add_leaf(
        id="Source_Confirms_Signing_And_Subject_Context",
        desc=("The cited official Ohio government source confirms the contextual constraints: "
              "associated with Avery’s Law / HB 247, signed on December 19, 2025, in Columbus, Ohio, "
              "by Governor Mike DeWine, and pertains to laws governing dogs (including dangerous/vicious dogs)."),
        parent=primary_node,
        critical=True,
    )
    context_claim = (
        "An official Ohio government source confirms that House Bill 247 (Avery’s Law) was signed on December 19, 2025 "
        "in Columbus, Ohio by Governor Mike DeWine and that the bill pertains to laws governing dogs, including "
        "dangerous and vicious dogs."
    )
    await evaluator.verify(
        claim=context_claim,
        node=context_leaf,
        sources=official_urls,
        additional_instruction=(
            "Verify that the official Ohio government page(s) explicitly support all parts of the claim: "
            "HB 247 (Avery’s Law), signing by Governor Mike DeWine, date (Dec 19, 2025), location (Columbus, Ohio), "
            "and subject matter (dogs, dangerous/vicious dogs). Minor formatting variations are acceptable."
        ),
    )

    # 5) Official source confirms the named individual is the primary sponsor of House Bill 247
    sponsor_confirm_leaf = evaluator.add_leaf(
        id="Source_Confirms_Named_Individual_Is_Primary_Sponsor",
        desc="The cited official Ohio government source confirms that the named individual is listed as the primary sponsor of House Bill 247.",
        parent=primary_node,
        critical=True,
    )
    sponsor_name_for_claim = extraction.sponsor_full_name or "UNKNOWN PERSON"
    sponsor_confirm_claim = (
        f"{sponsor_name_for_claim} is the primary sponsor of Ohio House Bill 247 (Avery’s Law) according to official "
        "Ohio government sources."
    )
    await evaluator.verify(
        claim=sponsor_confirm_claim,
        node=sponsor_confirm_leaf,
        sources=official_urls,
        additional_instruction=(
            "Check official Ohio government legislative or governor sources to confirm the individual's role as "
            "the 'Primary Sponsor' (or equivalent phrasing). Accept minor variations (e.g., 'Primary Sponsor(s)') "
            "but ensure the named individual appears in that role for HB 247."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer to the Avery's Law (HB 247) primary sponsor question using Mind2Web2.
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
        default_model=model,
    )

    # Extract core information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_core(),
        template_class=AnswerExtraction,
        extraction_name="core_extraction",
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()