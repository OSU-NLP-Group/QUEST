import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "freddie_mac_rate_2026_02_19"
TASK_DESCRIPTION = "What was the 30-year fixed-rate mortgage rate reported by Freddie Mac on February 19, 2026?"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class MortgageRateExtraction(BaseModel):
    """
    Extract:
    - primary_rate: the single numeric percentage that the answer claims is the Freddie Mac‑reported
      30-year fixed-rate mortgage rate for February 19, 2026 (verbatim from the answer).
    - source_urls: all URLs cited in the answer as supporting evidence for this rate.
    """
    primary_rate: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_rate_and_sources() -> str:
    return """
    Your goal is to extract the specific mortgage rate and any supporting URLs cited in the answer.

    Extract the following from the answer text:
    1) primary_rate: The single numeric percentage value that the answer claims is the
       Freddie Mac–reported 30-year fixed-rate mortgage rate for the date February 19, 2026.
       - Return it exactly as written in the answer (e.g., "6.77%" or "6.8 percent").
       - If multiple percentages are present in the answer, pick the one explicitly tied to the
         30-year fixed-rate mortgage on February 19, 2026.
       - If the answer does not clearly state such a value, return null.

    2) source_urls: A list of all URLs that the answer presents as citations/sources for this rate.
       - Include URLs shown in plain text or inside markdown links [text](url).
       - Only include valid URLs explicitly present in the answer.
       - If no URLs are provided, return an empty list.

    Return a JSON object with fields:
    - primary_rate (string or null)
    - source_urls (array of strings)
    """


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_answer_correctness(
    evaluator: Evaluator,
    parent_node,
    extracted: MortgageRateExtraction
) -> None:
    """
    Builds the verification tree corresponding to the rubric.
    """

    # Top-level rubric node (set non-critical to allow mixed criticality children)
    answer_correctness = evaluator.add_parallel(
        id="answer_correctness",
        desc="Evaluates whether the answer provides the correct Freddie Mac-reported 30-year fixed-rate mortgage rate for February 19, 2026.",
        parent=parent_node,
        critical=False
    )

    # Critical path: correctness of the stated rate for the specified context
    correct_rate_group = evaluator.add_sequential(
        id="correct_rate_for_specified_context",
        desc="Answer states a single percentage value that matches Freddie Mac’s reported 30-year fixed-rate mortgage rate for the date February 19, 2026.",
        parent=answer_correctness,
        critical=True
    )

    # Leaf 1 (critical): The answer states a single percentage value for the asked context
    single_value_leaf = evaluator.add_leaf(
        id="single_value_stated",
        desc="Answer states a single percentage value that pertains to the 30-year fixed-rate mortgage on February 19, 2026.",
        parent=correct_rate_group,
        critical=True
    )
    claim_single_value = (
        "In the answer text, there is exactly one numeric percentage that the answer presents as the "
        "30-year fixed-rate mortgage rate for February 19, 2026. Ignore any other percentages explicitly "
        "tied to different products (e.g., 15-year), different dates (e.g., previous week), or unrelated metrics. "
        "Ranges such as '6.7%–6.8%' count as multiple values and should be considered incorrect."
    )
    await evaluator.verify(
        claim=claim_single_value,
        node=single_value_leaf,
        additional_instruction=(
            "Judge based solely on the provided answer text. "
            "Pass only if the answer clearly singles out a single numeric percentage for the 30-year fixed-rate on the specified date."
        )
    )

    # Leaf 2 (critical): The value matches Freddie Mac’s reported rate (must be supported by cited URLs)
    value_text = extracted.primary_rate if extracted and extracted.primary_rate else ""
    rate_match_leaf = evaluator.add_leaf(
        id="rate_matches_freddie_mac",
        desc="The stated percentage value matches Freddie Mac’s reported 30-year fixed-rate mortgage rate for February 19, 2026, supported by the cited sources.",
        parent=correct_rate_group,
        critical=True
    )
    claim_rate_match = (
        f"According to the provided source(s), Freddie Mac’s 30-year fixed-rate mortgage rate on February 19, 2026 "
        f"was {value_text}."
    )
    await evaluator.verify(
        claim=claim_rate_match,
        node=rate_match_leaf,
        sources=extracted.source_urls if extracted else [],
        additional_instruction=(
            "Use ONLY the provided URLs as evidence. The page(s) should clearly indicate both: "
            "(a) the data is from Freddie Mac (directly or a reputable page explicitly quoting Freddie Mac), and "
            "(b) the 30-year fixed-rate mortgage rate for the date 2026-02-19 equals the stated value. "
            "Allow minor rounding differences (e.g., 6.77% vs 6.8%). "
            "If no supporting URL is provided or the page(s) do not clearly support the claim, mark as not supported."
        )
    )

    # Non-critical leaf: The answer mentions Freddie Mac as the source
    mentions_freddie_leaf = evaluator.add_leaf(
        id="mentions_freddie_mac_as_source",
        desc="Answer explicitly attributes the rate to Freddie Mac (at minimum by naming Freddie Mac).",
        parent=answer_correctness,
        critical=False
    )
    await evaluator.verify(
        claim="The answer explicitly attributes the rate to Freddie Mac (e.g., mentions 'Freddie Mac' or 'Freddie Mac PMMS').",
        node=mentions_freddie_leaf,
        additional_instruction="Minor wording variations are acceptable as long as it's clear the source is Freddie Mac."
    )

    # Non-critical leaf: The answer indicates 30-year fixed
    mentions_30yr_leaf = evaluator.add_leaf(
        id="mentions_30yr_fixed",
        desc="Answer explicitly indicates the rate is for a 30-year fixed-rate mortgage.",
        parent=answer_correctness,
        critical=False
    )
    await evaluator.verify(
        claim="The answer explicitly indicates the rate is for a 30-year fixed-rate mortgage (e.g., '30-year fixed', '30yr fixed', or equivalent).",
        node=mentions_30yr_leaf,
        additional_instruction="Accept obvious abbreviations or hyphenation variants like '30-year fixed-rate'."
    )

    # Non-critical leaf: The answer mentions the date
    mentions_date_leaf = evaluator.add_leaf(
        id="mentions_date",
        desc="Answer explicitly mentions the date February 19, 2026.",
        parent=answer_correctness,
        critical=False
    )
    await evaluator.verify(
        claim="The answer explicitly mentions the date 'February 19, 2026' or an equivalent format (e.g., 'Feb. 19, 2026', '2/19/2026', or '2026-02-19').",
        node=mentions_date_leaf,
        additional_instruction="Any clear, unambiguous notation of that calendar date should be accepted."
    )

    # Non-critical leaf: Provides citation/link (objective check for at least one URL in the answer)
    provides_citation_node = evaluator.add_custom_node(
        result=bool(extracted and extracted.source_urls and len(extracted.source_urls) > 0),
        id="provides_citation",
        desc="Answer provides a citation/link where the rate can be verified.",
        parent=answer_correctness,
        critical=False
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Freddie Mac 30-year fixed-rate mortgage rate on February 19, 2026.
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

    # Extract the stated rate and source URLs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_rate_and_sources(),
        template_class=MortgageRateExtraction,
        extraction_name="rate_and_sources"
    )

    # Optionally add custom info to help downstream inspection
    evaluator.add_custom_info(
        info={
            "target_date": "2026-02-19",
            "extracted_primary_rate": extracted.primary_rate,
            "extracted_source_urls": extracted.source_urls
        },
        info_type="context",
        info_name="target_and_extracted"
    )

    # Build and run verification
    await build_answer_correctness(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()