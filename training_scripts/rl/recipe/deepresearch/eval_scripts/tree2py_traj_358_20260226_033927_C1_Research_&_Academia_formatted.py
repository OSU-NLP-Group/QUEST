import asyncio
import logging
from typing import Any, List, Optional, Dict
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "chem_nobel_2025_citation"
TASK_DESCRIPTION = (
    "What is the official citation (reason for the award) given by the Nobel Foundation for the 2025 Nobel Prize in Chemistry? "
    "Provide the exact wording as stated in the official announcement and include a reference URL from the Nobel Prize website (nobelprize.org)."
)

EXPECTED_OFFICIAL_PHRASE = "for the development of metal-organic frameworks"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CitationExtraction(BaseModel):
    """
    Structured extraction from the agent's answer:
    - citation_text: exact wording of the official citation the answer claims
    - reference_urls: all URLs provided in the answer that point to nobelprize.org
    """
    citation_text: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_citation_and_urls() -> str:
    return """
    From the provided answer, extract:
    1) citation_text: The exact wording of the official citation (reason for the award) for the Nobel Prize in Chemistry 2025 as quoted in the answer. 
       This should be the precise phrase the answer claims is the Nobel Foundation's official citation. 
       Do not rewrite or paraphrase—return it exactly as it appears in the answer, including punctuation and casing.
    2) reference_urls: A list of all URLs included in the answer that originate from the Nobel Prize website (nobelprize.org) and are intended to support or confirm this citation.
       Extract only full URLs (plain URL or markdown links). Return only URLs from the nobelprize.org domain.

    If the citation_text is not explicitly provided, set it to null.
    If no Nobel Prize URLs are included, return an empty list for reference_urls.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_nobel_url(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
        host = (parsed.netloc or "").lower()
        return "nobelprize.org" in host
    except Exception:
        return False


def filter_nobel_urls(urls: List[str]) -> List[str]:
    seen = set()
    filtered: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        if is_nobel_url(u):
            # De-duplicate while preserving order
            if u not in seen:
                filtered.append(u)
                seen.add(u)
    return filtered


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extracted: CitationExtraction,
    root: Any
) -> None:
    """
    Build and execute the verification tree for the Nobel Chemistry 2025 citation task.
    """
    # Prepare sources (only nobelprize.org)
    nobel_urls: List[str] = filter_nobel_urls(extracted.reference_urls)

    # Create the main critical node (acts as the rubric root)
    main_node = evaluator.add_parallel(
        id="Chemistry_Nobel_2025_Citation",
        desc="Correctly identify and reference the official citation for the 2025 Nobel Prize in Chemistry",
        parent=root,
        critical=True
    )

    # Leaf 1: Citation_Text_Accuracy — Ensure the answer includes the exact target phrase
    citation_text_accuracy = evaluator.add_leaf(
        id="Citation_Text_Accuracy",
        desc="The citation must state 'for the development of metal-organic frameworks' as the official reason for the 2025 Nobel Prize in Chemistry",
        parent=main_node,
        critical=True
    )
    phrase_claim = (
        "The answer contains the exact phrase 'for the development of metal-organic frameworks' as the stated official citation "
        "for the Nobel Prize in Chemistry 2025."
    )
    await evaluator.verify(
        claim=phrase_claim,
        node=citation_text_accuracy,
        additional_instruction=(
            "Verify within the answer text whether the phrase appears exactly as written. "
            "Allow minor differences in letter casing but the word sequence must match. "
            "Quotation marks or surrounding punctuation are fine as long as the core phrase is present."
        )
    )

    # Group: Reference_URL_Validity — validate URL(s) and confirm the citation via official sources
    ref_group = evaluator.add_parallel(
        id="Reference_URL_Validity",
        desc="The answer must include a valid reference URL from nobelprize.org that confirms the 2025 Chemistry Nobel Prize citation",
        parent=main_node,
        critical=True
    )

    # Leaf 2a (custom): URL domain validity check
    url_domain_valid = evaluator.add_custom_node(
        result=(len(nobel_urls) > 0),
        id="Reference_URL_Domain_Valid",
        desc="At least one valid nobelprize.org URL is included in the answer",
        parent=ref_group,
        critical=True
    )

    # Leaf 2b: URL supports the citation (evidence-backed verification)
    url_supports_citation = evaluator.add_leaf(
        id="Reference_URL_Confirms_Citation",
        desc="Provided Nobel Prize URL(s) confirm that the 2025 Chemistry citation is 'for the development of metal-organic frameworks'",
        parent=ref_group,
        critical=True
    )
    support_claim = (
        "The official Nobel Prize website page(s) confirm that the citation (motivation) for the Nobel Prize in Chemistry 2025 "
        "is 'for the development of metal-organic frameworks'."
    )
    await evaluator.verify(
        claim=support_claim,
        node=url_supports_citation,
        sources=nobel_urls,
        additional_instruction=(
            "On nobelprize.org, locate the official announcement or prize page for the Nobel Prize in Chemistry 2025. "
            "Confirm that the motivation/citation text explicitly states 'for the development of metal-organic frameworks'. "
            "Ensure the page references the year 2025 and the Chemistry category. "
            "Minor punctuation or casing differences are acceptable, but the wording must match the phrase."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the 2025 Nobel Prize in Chemistry citation task.
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

    # Extract citation text and URLs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_citation_and_urls(),
        template_class=CitationExtraction,
        extraction_name="citation_extraction",
    )

    # Add ground truth information (expected phrase for checking clarity)
    evaluator.add_ground_truth({
        "expected_phrase": EXPECTED_OFFICIAL_PHRASE,
        "category": "Chemistry",
        "year": 2025
    }, gt_type="expected_official_citation")

    # Build and execute verification tree
    await build_verification_tree(evaluator, extracted, root)

    # Return structured result
    return evaluator.get_summary()