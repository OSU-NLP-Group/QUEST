import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cdc_dog_import_age_aug2024"
TASK_DESCRIPTION = """
What is the minimum age requirement for dogs being imported into the United States under the CDC regulations that became effective in August 2024? Provide the specific age requirement in months and cite an official CDC source.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class AgeRequirementExtraction(BaseModel):
    """
    Structured extraction from the agent's answer for CDC dog import age requirement.
    """
    stated_age_text: Optional[str] = None  # The exact quoted/phrased age requirement as written in the answer.
    age_months_str: Optional[str] = None   # Only the numeric months value as a string, e.g., "6" if the answer includes months.
    effective_date_text: Optional[str] = None  # The text snippet mentioning the effective date, if any.
    sources: List[str] = Field(default_factory=list)  # All URLs cited in the answer (valid, full URLs).


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_age_requirement() -> str:
    return """
    Extract from the answer the key details about the CDC dog import minimum age requirement effective August 2024.

    Required fields:
    1) stated_age_text:
       - Extract the exact phrase or sentence from the answer that states the minimum age requirement (e.g., “at least 6 months old”).
       - If the answer does not clearly state a minimum age, set to null.

    2) age_months_str:
       - If the answer explicitly expresses the minimum age in months (e.g., “6 months”, “six months”, “6 mo”, “≥6 months”), extract ONLY the numeric value as a string (e.g., "6").
       - If the answer gives the age in weeks or days but ALSO includes an explicit conversion in months (e.g., “180 days (6 months)”), extract the months numeric value (e.g., "6").
       - If the answer never expresses the age in months (and does not include a months conversion), set this to null. Do NOT invent or compute conversions on your own.

    3) effective_date_text:
       - If the answer mentions that the regulation/effective date is August 1, 2024 (or a very close equivalent phrasing like “effective Aug. 1, 2024” or “effective in August 2024”), extract that mention verbatim.
       - Otherwise, set to null.

    4) sources:
       - Extract all URLs cited in the answer text (including Markdown links). Return the full URL with protocol; if a URL is missing protocol, prepend http://.
       - Only include URLs that actually appear in the answer text.

    If any field cannot be found, set it to null (or empty list for sources). Do not add or infer any information not present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def filter_cdc_urls(urls: List[str]) -> List[str]:
    """Return only URLs that belong to the CDC domain."""
    cdc_list = []
    for u in urls:
        if not isinstance(u, str):
            continue
        low = u.strip().lower()
        if "cdc.gov" in low:
            cdc_list.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for u in cdc_list:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: AgeRequirementExtraction) -> None:
    """
    Build the rubric tree for CDC dog import minimum age requirement and run verifications.
    """
    # Add the top-level critical node (parallel) that aggregates all four critical checks
    root_node = evaluator.add_parallel(
        id="Dog_Import_Age_Requirement",
        desc="Correctly identifies the minimum age requirement for importing dogs into the United States under CDC regulations effective August 1, 2024, and supports it with an official CDC source.",
        parent=evaluator.root,
        critical=True
    )

    # Prepare data
    all_urls = extracted.sources or []
    cdc_urls = filter_cdc_urls(all_urls)
    evaluator.add_custom_info(
        info={
            "stated_age_text": extracted.stated_age_text,
            "age_months_str": extracted.age_months_str,
            "effective_date_text": extracted.effective_date_text,
            "all_urls": all_urls,
            "cdc_urls": cdc_urls,
        },
        info_type="extraction_postprocess",
        info_name="post_extraction_summary"
    )

    # Leaf 4 (from rubric): Valid_CDC_Source_Cited — custom boolean check
    cdc_source_leaf = evaluator.add_custom_node(
        result=(len(cdc_urls) > 0),
        id="Valid_CDC_Source_Cited",
        desc="Cites at least one official CDC website source on the cdc.gov domain that supports the stated minimum age requirement.",
        parent=root_node,
        critical=True
    )

    # Leaf 2 (from rubric): Age_Expressed_In_Months — custom boolean check based on extraction
    age_in_months_leaf = evaluator.add_custom_node(
        result=(extracted.age_months_str is not None and str(extracted.age_months_str).strip() != ""),
        id="Age_Expressed_In_Months",
        desc="Expresses the minimum age requirement in months (not only in years/days, unless also converted to months).",
        parent=root_node,
        critical=True
    )

    # Leaf 3 (from rubric): Effective_Date_Mentioned — verify in the answer text
    effective_date_leaf = evaluator.add_leaf(
        id="Effective_Date_Mentioned",
        desc="Mentions that the relevant CDC regulation became effective on August 1, 2024.",
        parent=root_node,
        critical=True
    )
    effective_date_claim = "The answer explicitly mentions that the CDC dog import regulation became effective on August 1, 2024."
    await evaluator.verify(
        claim=effective_date_claim,
        node=effective_date_leaf,
        additional_instruction="Check if the answer text clearly references the effective date as 'August 1, 2024' (including reasonable variants like 'Aug. 1, 2024'). Simply referencing 'August 2024' without indicating it as the effective date should not count unless it clearly implies the rule took effect then."
    )

    # Leaf 1 (from rubric): Minimum_Age_Is_Correct_Per_CDC_Aug2024 — verify by CDC URLs
    min_age_leaf = evaluator.add_leaf(
        id="Minimum_Age_Is_Correct_Per_CDC_Aug2024",
        desc="States the minimum age requirement that is specified by the CDC dog importation regulation effective August 1, 2024 (i.e., the stated age matches what the cited CDC source says).",
        parent=root_node,
        critical=True
    )

    # Construct the claim using the extracted months value if available, otherwise fall back to the stated phrase.
    if extracted.age_months_str and str(extracted.age_months_str).strip():
        age_clause = f"at least {str(extracted.age_months_str).strip()} months old"
    elif extracted.stated_age_text and extracted.stated_age_text.strip():
        # Less precise, but still derived from the answer. This node will be blocked if months wasn't expressed.
        age_clause = extracted.stated_age_text.strip()
    else:
        # If neither is available, the prerequisite 'Age_Expressed_In_Months' should fail and gate this check.
        age_clause = "a specific minimum age as stated in the answer"

    min_age_claim = f"CDC dog importation regulations effective August 1, 2024 require dogs to be {age_clause} to be imported into the United States."

    await evaluator.verify(
        claim=min_age_claim,
        node=min_age_leaf,
        sources=cdc_urls if cdc_urls else None,
        extra_prerequisites=[cdc_source_leaf, age_in_months_leaf],
        additional_instruction=(
            "Verify this claim strictly against the provided CDC webpage(s). "
            "Look for language on the CDC page indicating the minimum age for imported dogs as of the policy effective August 1, 2024. "
            "Treat phrasings like 'at least 6 months old' or 'must be ≥ 6 months' as equivalent. "
            "If the page states a different age, the claim is not supported."
        )
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
) -> Dict:
    """
    Evaluate an answer for the CDC dog import minimum age requirement (effective August 2024).
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation strategy (non-critical root)
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_age_requirement(),
        template_class=AgeRequirementExtraction,
        extraction_name="age_requirement_extraction"
    )

    # Build tree and run verifications
    await build_and_verify_tree(evaluator, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()