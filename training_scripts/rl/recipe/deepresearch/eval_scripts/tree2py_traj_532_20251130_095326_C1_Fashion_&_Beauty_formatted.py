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
TASK_ID = "mu2024_bncc"
TASK_DESCRIPTION = (
    "Who designed the Best National Costume winner at Miss Universe 2024, "
    "which was held on November 16, 2024, in Mexico City, Mexico, and what was the costume named?"
)

# Expected event context (for reporting and instruction to the judge model)
EXPECTED_CONTEXT = {
    "edition": "73rd",
    "year": "2024",
    "final_date": "November 16, 2024",
    "venue": "Arena CDMX",
    "city": "Mexico City",
    "country": "Mexico",
}

# --------------------------------------------------------------------------- #
# Data model for extraction                                                   #
# --------------------------------------------------------------------------- #
class BNCCExtraction(BaseModel):
    """Information about Best National Costume at Miss Universe 2024 extracted from the answer."""
    # Event context (as mentioned in the answer; may be partial or missing)
    event_label: Optional[str] = None           # e.g., "Miss Universe 2024", "73rd edition"
    event_year: Optional[str] = None            # e.g., "2024"
    edition_number: Optional[str] = None        # e.g., "73rd"
    event_date: Optional[str] = None            # any date string mentioned
    venue: Optional[str] = None                 # e.g., "Arena CDMX"
    city: Optional[str] = None                  # e.g., "Mexico City"
    country: Optional[str] = None               # e.g., "Mexico"

    # Winner info (contestant and/or country/representation)
    winner_name: Optional[str] = None           # contestant name (if mentioned)
    winner_country: Optional[str] = None        # country/representation (if mentioned)

    # Required task answers
    designer_name: Optional[str] = None         # who designed the winning costume
    costume_name: Optional[str] = None          # the name of the winning costume

    # Source URLs cited in the answer
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_bncc() -> str:
    return """
    Extract the following information exactly as it appears in the provided answer text (do not invent or infer):

    Event context (as explicitly mentioned in the answer, if any):
    - event_label: A phrase like "Miss Universe 2024" or "73rd edition".
    - event_year: The year of the event, e.g., "2024".
    - edition_number: The edition number, e.g., "73rd".
    - event_date: The final competition date mentioned in the answer, if any (as a free-form string).
    - venue: The venue mentioned (e.g., "Arena CDMX"), if any.
    - city: The city mentioned (e.g., "Mexico City"), if any.
    - country: The country mentioned (e.g., "Mexico"), if any.

    Winner information:
    - winner_name: The contestant name of the Best National Costume winner, if mentioned.
    - winner_country: The country/representation (e.g., "Vietnam") for the BNCC winner, if mentioned.

    Required task answers:
    - designer_name: The name of the designer who created the winning Best National Costume (as provided in the answer).
    - costume_name: The name of the winning costume (as provided in the answer).

    Source URLs:
    - source_urls: An array of all URLs explicitly provided in the answer that support or relate to the BNCC winner, the designer, or the costume name.
      Include plain URLs and URLs inside markdown links. Do not invent URLs. If none are present, return an empty array.

    If a field is not mentioned in the answer, return null for that field (or an empty array for source_urls).
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_bncc_tree(
    evaluator: Evaluator,
    parent_node,
    extracted: BNCCExtraction,
) -> None:
    """
    Build the verification tree and perform checks according to the rubric.
    """
    # Top-level node (critical)
    main_node = evaluator.add_parallel(
        id="Best_National_Costume_Information",
        desc="Answer provides the required information about the Best National Costume at Miss Universe 2024 (73rd edition) with the specified event context.",
        parent=parent_node,
        critical=True,
    )

    # Event context checks (critical, parallel)
    event_node = evaluator.add_parallel(
        id="Event_Context_Matches_Constraints",
        desc="Answer is explicitly about Miss Universe 2024 (73rd edition) and does not conflict with the stated event context (date/venue/location).",
        parent=main_node,
        critical=True,
    )

    # Leaf: Event_Is_Miss_Universe_2024_73rd
    event_year = extracted.event_year or ""
    edition_num = extracted.edition_number or ""
    event_label = extracted.event_label or ""
    event_mu_leaf = evaluator.add_leaf(
        id="Event_Is_Miss_Universe_2024_73rd",
        desc="The answer pertains to Miss Universe 2024 (the 73rd edition), not a different year/edition.",
        parent=event_node,
        critical=True,
    )
    mu_claim = (
        "The answer explicitly pertains to Miss Universe 2024 (the 73rd edition), "
        "and does not reference a different year or edition."
    )
    await evaluator.verify(
        claim=mu_claim,
        node=event_mu_leaf,
        additional_instruction=(
            "Check the answer text itself. This check requires explicit mention that it is about "
            "'Miss Universe 2024' and/or 'the 73rd edition'. If the answer is silent about the year/edition, "
            "mark this as Incorrect. Allow minor phrasing variants (e.g., 'MU 2024', '73rd Miss Universe')."
        ),
    )

    # Leaf: Event_Date_Matches
    event_date_leaf = evaluator.add_leaf(
        id="Event_Date_Matches",
        desc="The answer is consistent with the final competition date being November 16, 2024 (i.e., does not reference a different final date for this event).",
        parent=event_node,
        critical=True,
    )
    date_claim = (
        "The answer is consistent with the final competition date being November 16, 2024; "
        "it does not reference a different final date for this event."
    )
    await evaluator.verify(
        claim=date_claim,
        node=event_date_leaf,
        additional_instruction=(
            "We only require non-conflict for the date. If the answer is silent on the date, consider it Correct. "
            "If it mentions a different final date for Miss Universe 2024, mark Incorrect."
        ),
    )

    # Leaf: Event_Venue_Location_Matches
    venue_loc_leaf = evaluator.add_leaf(
        id="Event_Venue_Location_Matches",
        desc="The answer is consistent with the event being held at Arena CDMX in Mexico City, Mexico (i.e., does not reference a different venue/location for this event).",
        parent=event_node,
        critical=True,
    )
    venue_claim = (
        "The answer is consistent with the event being held at Arena CDMX in Mexico City, Mexico; "
        "it does not reference a different venue or location."
    )
    await evaluator.verify(
        claim=venue_claim,
        node=venue_loc_leaf,
        additional_instruction=(
            "We only require non-conflict for venue/location. If the answer is silent on venue/location, consider it Correct. "
            "If it mentions a different venue or city/country for Miss Universe 2024, mark Incorrect."
        ),
    )

    # Winner identified (existence check)
    winner_identified = evaluator.add_custom_node(
        result=bool((extracted.winner_name and extracted.winner_name.strip()) or (extracted.winner_country and extracted.winner_country.strip())),
        id="Winner_Identified",
        desc="The Best National Costume winner is identified (e.g., contestant and/or country/representation).",
        parent=main_node,
        critical=True,
    )

    # Designer Name verification (by URLs if provided, else simple)
    designer_leaf = evaluator.add_leaf(
        id="Designer_Name",
        desc="The name of the designer who created the winning costume is provided.",
        parent=main_node,
        critical=True,
    )
    designer_name = extracted.designer_name or ""
    designer_claim = (
        f"The designer who created the winning Best National Costume at Miss Universe 2024 is '{designer_name}'."
    )
    await evaluator.verify(
        claim=designer_claim,
        node=designer_leaf,
        sources=extracted.source_urls if extracted.source_urls else None,
        additional_instruction=(
            "Verify the designer's name against any cited sources (URLs) if available. "
            "Allow minor variations in spelling, accents, or ordering of names. "
            "If multiple designers are credited by the sources and the answer includes one of them, consider it Correct. "
            "If no sources are provided, judge based on the answer content only."
        ),
    )

    # Costume Name verification (by URLs if provided, else simple)
    costume_leaf = evaluator.add_leaf(
        id="Costume_Name",
        desc="The name of the winning costume is provided.",
        parent=main_node,
        critical=True,
    )
    costume_name = extracted.costume_name or ""
    costume_claim = f"The winning Best National Costume was named '{costume_name}'."
    await evaluator.verify(
        claim=costume_claim,
        node=costume_leaf,
        sources=extracted.source_urls if extracted.source_urls else None,
        additional_instruction=(
            "Verify the costume name with cited sources when available. Allow minor variations (case differences, articles, "
            "diacritics). If the sources clearly indicate the same name, consider it Correct. "
            "If no sources are provided, judge based on the answer content only."
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
    Evaluate the answer for the Miss Universe 2024 Best National Costume designer and costume name task.
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

    # Record expected context for transparency in summary
    evaluator.add_ground_truth(
        {
            "expected_event_context": EXPECTED_CONTEXT,
            "rubric_focus": [
                "Event explicitly Miss Universe 2024 (73rd edition)",
                "No conflict with date (Nov 16, 2024)",
                "No conflict with venue/location (Arena CDMX, Mexico City, Mexico)",
                "Winner identified (contestant and/or country)",
                "Designer name provided",
                "Costume name provided",
            ],
        },
        gt_type="expected_context",
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_bncc(),
        template_class=BNCCExtraction,
        extraction_name="bncc_extraction",
    )

    # Build verification tree and run checks
    await build_bncc_tree(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()