import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "gloria_estefan_producer_2025"
TASK_DESCRIPTION = """
Gloria Estefan released a Spanish-language album in 2025. Who was the primary producer of this album, how many Grammy Awards has this producer won in their career, and what is this producer's date of birth?
"""

EXPECTED_PRODUCER_NAME = "Emilio Estefan Jr."
EXPECTED_GRAMMY_COUNT = "19"
EXPECTED_BIRTHDATE = "March 4, 1953"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProducerExtraction(BaseModel):
    """
    Extract information from the agent's answer about:
    - the album title (if stated)
    - the primary producer's name
    - the producer's total number of Grammy Awards (not Latin Grammys)
    - the producer's birth date
    - and the URLs (if any) explicitly cited to support each of the above
    """
    album_title: Optional[str] = None

    producer_name: Optional[str] = None
    producer_sources: List[str] = Field(default_factory=list)

    grammy_awards_count: Optional[str] = None
    grammy_sources: List[str] = Field(default_factory=list)

    producer_birth_date: Optional[str] = None
    birthdate_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_producer_info() -> str:
    return """
    From the answer text, extract the following fields related to Gloria Estefan's 2025 Spanish-language album:

    - album_title: The album title explicitly mentioned by the answer (if any). If not stated, return null.
    - producer_name: The name the answer claims is the primary producer of the album. Return exactly as written in the answer. If missing, return null.
    - producer_sources: An array of URLs explicitly cited in the answer that substantiate who the primary producer is. If none are cited, return an empty array.

    - grammy_awards_count: The number of Grammy Awards (NOT Latin Grammys) the answer claims this producer has won in their career. Return exactly as stated in the answer (e.g., "19"). If missing, return null.
    - grammy_sources: An array of URLs explicitly cited in the answer to support the Grammy Award count for the producer. If none are cited, return an empty array.

    - producer_birth_date: The producer's date of birth as stated in the answer (e.g., "March 4, 1953" or "1953-03-04"). If missing, return null.
    - birthdate_sources: An array of URLs explicitly cited in the answer to support the producer's birth date. If none are cited, return an empty array.

    IMPORTANT:
    - Extract only what is explicitly stated in the answer; do not invent or infer new information.
    - For URL fields, include only valid URLs that appear in the answer (plain or markdown links). If none are present, return empty arrays.
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_producer_information(evaluator: Evaluator, parent_node, extracted: ProducerExtraction) -> None:
    """
    Build the verification tree per rubric and run checks.
    The parent node ("Producer_Information") is critical and parallel; all children are critical.
    """
    # Create the main critical parallel node
    producer_info_node = evaluator.add_parallel(
        id="Producer_Information",
        desc="Verify all required information about the producer of Gloria Estefan's 2025 album 'Raíces'",
        parent=parent_node,
        critical=True
    )

    # Prepare safe extracted values
    extracted_name = (extracted.producer_name or "").strip()
    extracted_grammys = (extracted.grammy_awards_count or "").strip()
    extracted_birthdate = (extracted.producer_birth_date or "").strip()

    # Sources (may be empty)
    grammy_sources = extracted.grammy_sources if extracted.grammy_sources else []
    birth_sources = extracted.birthdate_sources if extracted.birthdate_sources else []
    producer_sources = extracted.producer_sources if extracted.producer_sources else []

    # 1) Producer_Name leaf (critical)
    node_name = evaluator.add_leaf(
        id="Producer_Name",
        desc="The producer is correctly identified as Emilio Estefan Jr.",
        parent=producer_info_node,
        critical=True
    )
    claim_producer_match = (
        f"The producer named in the answer is '{extracted_name}', and this refers to '{EXPECTED_PRODUCER_NAME}'. "
        f"Treat minor variations (e.g., punctuation, middle initials, or omission/addition of 'Jr.') as equivalent if they clearly refer to the same person."
    )
    # Simple logical check; no need for URL evidence because this leaf checks alignment between the answer and expected identity
    await evaluator.verify(
        claim=claim_producer_match,
        node=node_name,
        additional_instruction=(
            "Focus on whether the two names refer to the same person. "
            "Allow reasonable variants including different casing, presence/absence of 'Jr.', middle initials, or minor diacritics."
        ),
    )

    # 2) Grammy_Awards_Count leaf (critical)
    node_grammys = evaluator.add_leaf(
        id="Grammy_Awards_Count",
        desc="The number of Grammy Awards won by the producer is correctly stated as 19",
        parent=producer_info_node,
        critical=True
    )
    claim_grammys = (
        f"The producer's career Grammy Awards count stated in the answer is '{extracted_grammys}', and this equals {EXPECTED_GRAMMY_COUNT}. "
        f"This count must refer specifically to Grammy Awards (not Latin Grammy Awards)."
    )
    # Prefer verifying against URLs the answer cited (if any). If none, the system will perform a simple logical check.
    await evaluator.verify(
        claim=claim_grammys,
        node=node_grammys,
        sources=grammy_sources if len(grammy_sources) > 0 else None,
        additional_instruction=(
            "When checking the webpages, ensure the number is explicitly for 'Grammy Awards' (the U.S.-based Recording Academy awards), "
            "not 'Latin Grammy Awards'. If sources only mention Latin Grammys, the claim should not be considered supported."
        ),
    )

    # 3) Birth_Date leaf (critical)
    node_birth = evaluator.add_leaf(
        id="Birth_Date",
        desc="The producer's birth date is correctly stated as March 4, 1953",
        parent=producer_info_node,
        critical=True
    )
    claim_birthdate = (
        f"The producer's date of birth stated in the answer is '{extracted_birthdate}', and this equals '{EXPECTED_BIRTHDATE}'. "
        f"Accept reasonable date formatting variations (e.g., '1953-03-04' vs 'March 4, 1953') that represent the same date."
    )
    await evaluator.verify(
        claim=claim_birthdate,
        node=node_birth,
        sources=birth_sources if len(birth_sources) > 0 else None,
        additional_instruction=(
            "If webpages show the same calendar date using different formats (e.g., '1953-03-04', 'March 4, 1953'), treat them as equivalent."
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
    Evaluate an answer for the Gloria Estefan 2025 album producer task.
    """
    # Initialize evaluator (root is non-critical by design)
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
        prompt=prompt_extract_producer_info(),
        template_class=ProducerExtraction,
        extraction_name="producer_extraction",
    )

    # Add ground truth context for transparency
    evaluator.add_ground_truth({
        "expected_producer_name": EXPECTED_PRODUCER_NAME,
        "expected_grammy_awards_count": EXPECTED_GRAMMY_COUNT,
        "expected_birth_date": EXPECTED_BIRTHDATE,
        "note": "Counts refer to Grammy Awards (Recording Academy), not Latin Grammys."
    })

    # Build tree and verify
    await verify_producer_information(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()