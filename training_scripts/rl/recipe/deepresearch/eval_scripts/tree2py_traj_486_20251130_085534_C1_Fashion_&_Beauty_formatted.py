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
TASK_ID = "dior_beauty_soho_us_first"
TASK_DESCRIPTION = (
    "What is the complete street address and opening date of Dior's first standalone beauty boutique in the United States, "
    "which is located in SoHo, New York City?"
)

EXPECTED_ADDRESS = "109 Greene Street, New York, NY 10012"
EXPECTED_OPENING_DATE = "October 30, 2024"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BoutiqueExtraction(BaseModel):
    """Structured extraction from the agent's answer."""
    store_name: Optional[str] = None
    us_first_standalone_phrase: Optional[str] = None  # exact phrase indicating "first standalone beauty boutique in the US"
    location_neighborhood: Optional[str] = None       # e.g., "SoHo"
    city: Optional[str] = None                        # e.g., "New York City"
    state: Optional[str] = None                       # e.g., "NY"
    street_address: Optional[str] = None              # full address as stated in the answer
    opening_date: Optional[str] = None                # date as stated in the answer
    source_urls: List[str] = Field(default_factory=list)  # all URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_boutique_info() -> str:
    return """
    Extract from the answer the details about Dior's first standalone beauty boutique in the United States.

    Return a JSON object with the following fields:
    1. store_name: The store name as written (e.g., "Dior Beauty", "Dior Beauty SoHo"), if present.
    2. us_first_standalone_phrase: The exact phrase or sentence in the answer that asserts this is Dior’s first standalone beauty boutique in the United States. If not present, return null.
    3. location_neighborhood: The neighborhood name if provided (e.g., "SoHo"); else null.
    4. city: The city name if provided (e.g., "New York City"); else null.
    5. state: The state if provided (e.g., "NY"); else null.
    6. street_address: The complete street address string as written in the answer. If no address is stated, return null.
    7. opening_date: The opening date string as written in the answer (e.g., "October 30, 2024"); else null.
    8. source_urls: All URLs explicitly mentioned in the answer that relate to this boutique (including press releases, news articles, brand pages, etc.). If none are present, return an empty list.

    IMPORTANT:
    - Do not infer or invent information. Extract exactly what appears in the answer text.
    - For URLs, include only valid URLs (plain URLs or markdown links).
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_nodes(
    evaluator: Evaluator,
    root: Any,
    extraction: BoutiqueExtraction
) -> None:
    """
    Build the verification nodes according to the rubric and run the checks.
    Root is non-critical by design; we create a critical child node that mirrors the rubric root.
    """

    # Create the rubric's top-level critical node
    top_node = evaluator.add_parallel(
        id="Dior_First_US_Beauty_Boutique_SoHo",
        desc=(
            "Answer identifies Dior's first standalone beauty boutique in the United States located in SoHo, NYC, "
            "and provides the required complete street address and opening date."
        ),
        parent=root,
        critical=True
    )

    # Leaf 1: Boutique_Identification
    node_boutique_ident = evaluator.add_leaf(
        id="Boutique_Identification",
        desc="Identifies the store as Dior’s first standalone beauty boutique in the United States.",
        parent=top_node,
        critical=True
    )
    claim_boutique_ident = (
        "The answer explicitly identifies the store as Dior’s first standalone beauty boutique in the United States."
    )
    await evaluator.verify(
        claim=claim_boutique_ident,
        node=node_boutique_ident,
        additional_instruction=(
            "Judge only whether the answer text itself asserts that this store is Dior’s first standalone beauty boutique in the United States. "
            "Look for explicit phrasing such as 'first standalone beauty boutique' and 'United States'. "
            "Do not rely on external sources for this check; focus on the answer text."
        ),
    )

    # Leaf 2: Location_Constraint
    node_location = evaluator.add_leaf(
        id="Location_Constraint",
        desc="States that the boutique is located in the SoHo neighborhood of New York City.",
        parent=top_node,
        critical=True
    )
    claim_location = (
        "The answer states that the boutique is located in the SoHo neighborhood of New York City."
    )
    await evaluator.verify(
        claim=claim_location,
        node=node_location,
        additional_instruction=(
            "Check that the answer text explicitly contains 'SoHo' and also references 'New York City' (or equivalent forms such as 'NYC'). "
            "Minor variations in casing are acceptable for 'SoHo' and 'NYC', but both concepts must be clearly present."
        ),
    )

    # Leaf 3: Street_Address_Exact
    node_address = evaluator.add_leaf(
        id="Street_Address_Exact",
        desc='Provides the complete street address exactly as required: "109 Greene Street, New York, NY 10012".',
        parent=top_node,
        critical=True
    )
    claim_address = (
        f'The answer provides the complete street address exactly as: "{EXPECTED_ADDRESS}".'
    )
    await evaluator.verify(
        claim=claim_address,
        node=node_address,
        additional_instruction=(
            "Be STRICT about exactness for this check. The answer must include the exact string "
            f'"{EXPECTED_ADDRESS}" with the same punctuation and spacing. '
            "Abbreviations (e.g., 'Greene St.') or minor formatted variants should be considered incorrect for this exact-match requirement."
        ),
    )

    # Leaf 4: Opening_Date_Exact
    node_opening_date = evaluator.add_leaf(
        id="Opening_Date_Exact",
        desc='Provides the opening date exactly as required: "October 30, 2024".',
        parent=top_node,
        critical=True
    )
    claim_opening_date = (
        f'The answer provides the opening date exactly as: "{EXPECTED_OPENING_DATE}".'
    )
    await evaluator.verify(
        claim=claim_opening_date,
        node=node_opening_date,
        additional_instruction=(
            "Be STRICT about exactness for this check. The answer must include the exact date string "
            f'"{EXPECTED_OPENING_DATE}" (month name, day, and year). '
            "Minor variations (e.g., different date formats or missing components) should be considered incorrect for this exact-match requirement."
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
    Evaluate an agent's answer for the Dior SoHo beauty boutique task.
    Returns a standardized summary dictionary from the evaluator.
    """
    # Initialize evaluator (root is non-critical by framework design)
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

    # Extraction
    extraction = await evaluator.extract(
        prompt=prompt_extract_boutique_info(),
        template_class=BoutiqueExtraction,
        extraction_name="boutique_extraction",
    )

    # Optional: record ground truth for transparency
    evaluator.add_ground_truth({
        "expected_address": EXPECTED_ADDRESS,
        "expected_opening_date": EXPECTED_OPENING_DATE,
        "expected_neighborhood": "SoHo",
        "expected_city": "New York City",
        "identity_requirement": "First standalone beauty boutique in the United States"
    }, gt_type="ground_truth")

    # Build verification tree and run checks
    await build_and_verify_nodes(evaluator, root, extraction)

    # Return structured summary
    return evaluator.get_summary()