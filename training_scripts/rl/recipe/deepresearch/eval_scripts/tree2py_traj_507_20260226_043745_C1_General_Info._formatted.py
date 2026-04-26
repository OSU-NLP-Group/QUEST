import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "deion_partner_birth_year"
TASK_DESCRIPTION = """
Provide the correct birth year of the person who is in a relationship with Deion Sanders as of 2025 and was the first person of Asian Pacific American descent to win a Daytime Emmy for Lead Actress.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PersonExtraction(BaseModel):
    """
    Structured data extracted from the answer.
    """
    person_name: Optional[str] = None
    birth_year: Optional[str] = None

    # URLs explicitly cited in the answer to support each required verification
    relationship_sources: List[str] = Field(default_factory=list)
    emmy_sources: List[str] = Field(default_factory=list)
    birth_year_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_person_info() -> str:
    return """
    From the answer, extract the following fields about the identified person:
    - person_name: The full name of the person the answer identifies.
    - birth_year: The 4-digit birth year provided for that person (string). If absent, return null.
    - relationship_sources: A list of URLs explicitly cited in the answer that support the claim that this person is/was in a relationship with former NFL player Deion Sanders (as of 2025). Include only URLs actually present in the answer.
    - emmy_sources: A list of URLs explicitly cited in the answer that support the claim that this person was the first person of Asian Pacific American descent to win a Daytime Emmy Award for Lead Actress (also known as “Outstanding Performance by a Lead Actress in a Daytime Fiction Program”). Include only URLs actually present in the answer.
    - birth_year_sources: A list of URLs explicitly cited in the answer that support the person’s birth year (or date of birth). Include only URLs actually present in the answer.

    Rules for URL extraction:
    1) Extract only URLs explicitly present in the answer text (including markdown links).
    2) If the answer lists a combined “Sources” section, assign relevant URLs to the lists above as best as possible; a URL may appear in more than one list if the answer indicates it supports multiple claims.
    3) If the answer does not provide any URL for a given field, return an empty list for that field. Do not fabricate URLs.
    4) Return the year as a string (e.g., "1988"), not a number.

    Return a single JSON object with keys:
    person_name, birth_year, relationship_sources, emmy_sources, birth_year_sources.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedupe_urls(urls: List[str]) -> List[str]:
    """Deduplicate URLs while preserving order."""
    seen = set()
    deduped: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u_stripped = u.strip()
        if not u_stripped:
            continue
        if u_stripped not in seen:
            seen.add(u_stripped)
            deduped.append(u_stripped)
    return deduped


async def _verify_with_sources_or_fail(
    evaluator: Evaluator,
    node,
    claim: str,
    sources: List[str],
    additional_instruction: str,
) -> bool:
    """
    Enforce source-grounding: if no URLs are provided, mark the node failed without LLM verification.
    """
    srcs = _dedupe_urls(sources or [])
    if len(srcs) == 0:
        # No evidence provided -> treat as failure per source-grounding policy
        node.score = 0.0
        node.status = "failed"
        return False

    # Use the built-in verification with the provided URLs
    return await evaluator.verify(
        claim=claim,
        node=node,
        sources=srcs,
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Verification logic (tree construction)                                      #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root,
    extracted: PersonExtraction,
) -> None:
    """
    Build the verification tree as per rubric and run verifications.
    """

    person_name = extracted.person_name or "the identified person"
    birth_year = extracted.birth_year or ""

    # Leaf 1: Relationship constraint (Critical)
    # The person identified is in a relationship with former NFL player Deion Sanders as of 2025
    relationship_node = evaluator.add_leaf(
        id="relationship_constraint",
        desc="The person identified is in a relationship with former NFL player Deion Sanders as of 2025",
        parent=root,
        critical=True,
    )

    relationship_claim = (
        f"As of 2025, {person_name} is or was in a romantic relationship with former NFL player Deion Sanders "
        f"(e.g., dating, fiancée/engaged, or publicly acknowledged partner)."
    )
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node=relationship_node,
        claim=relationship_claim,
        sources=extracted.relationship_sources,
        additional_instruction=(
            "Use only the provided URLs. Accept synonyms like 'dating', 'girlfriend/boyfriend', 'partner', or 'fiancée/engaged'. "
            "The source must clearly tie the same person named in the answer to Deion Sanders as a romantic partner. "
            "If the evidence is outdated and indicates a breakup before 2025, consider the claim not supported."
        ),
    )

    # Leaf 2: Emmy award constraint (Critical)
    # The person is the first person of Asian Pacific American descent to win a Daytime Emmy Award for Lead Actress
    emmy_node = evaluator.add_leaf(
        id="emmy_award_constraint",
        desc="The person identified is the first person of Asian Pacific American descent to win a Daytime Emmy Award for Lead Actress (Outstanding Performance by a Lead Actress in a Daytime Fiction Program)",
        parent=root,
        critical=True,
    )

    emmy_claim = (
        f"{person_name} was the first person of Asian Pacific American descent to win a Daytime Emmy Award for Lead Actress, "
        f"also described as 'Outstanding Performance by a Lead Actress in a Daytime Fiction Program'."
    )
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node=emmy_node,
        claim=emmy_claim,
        sources=extracted.emmy_sources,
        additional_instruction=(
            "Use only the provided URLs. Validate both parts: (1) the person indeed won a Daytime Emmy for a Lead Actress category "
            "(the category name may appear as 'Outstanding Performance by a Lead Actress in a Daytime Fiction Program'), "
            "and (2) that this win made them the first Asian Pacific American to achieve this particular Lead Actress Daytime Emmy. "
            "If the page only mentions 'first' without specifying Asian Pacific American or the Lead Actress category, treat as insufficient."
        ),
    )

    # Leaf 3: Birth year accuracy (Critical)
    # The birth year provided is correct for the identified person
    birth_year_node = evaluator.add_leaf(
        id="birth_year_accuracy",
        desc="The birth year provided is correct for the identified person",
        parent=root,
        critical=True,
    )

    birth_claim = f"{person_name} was born in {birth_year}."
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node=birth_year_node,
        claim=birth_claim,
        sources=extracted.birth_year_sources,
        additional_instruction=(
            "Use only the provided URLs. Confirm the person's date of birth and ensure the year matches exactly. "
            "Minor variations in formatting (e.g., month/day) are acceptable; focus on the year."
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
    Evaluate an answer for the Deion Sanders partner birth-year task using the Mind2Web2 framework.
    """
    # Initialize evaluator and root
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_person_info(),
        template_class=PersonExtraction,
        extraction_name="person_extraction",
    )

    # Record what we extracted as custom info to aid debugging
    evaluator.add_custom_info(
        info={
            "person_name": extracted.person_name,
            "birth_year": extracted.birth_year,
            "relationship_sources_count": len(extracted.relationship_sources),
            "emmy_sources_count": len(extracted.emmy_sources),
            "birth_year_sources_count": len(extracted.birth_year_sources),
        },
        info_type="extraction_summary",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, root, extracted)

    # Return standardized summary
    return evaluator.get_summary()