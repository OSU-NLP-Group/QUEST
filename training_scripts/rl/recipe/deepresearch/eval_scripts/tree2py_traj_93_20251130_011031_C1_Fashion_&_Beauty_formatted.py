import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "mu2025_best_national_costume"
TASK_DESCRIPTION = (
    "At the Miss Universe 2025 pageant held in Thailand in November 2025, who won the Best National Costume award, "
    "who designed the winning costume, and what was the official title or name of the costume?"
)

# Ground truth expectations for rubric checks
EXPECTED_WINNER_NAME = "Ahtisa Manalo"
EXPECTED_WINNER_COUNTRY = "Philippines"
EXPECTED_DESIGNER = "Mak Tumang"
EXPECTED_COSTUME_TITLE = "Festejada: Queen of Philippine Festivals"

# Event context ground truth for the non-critical contradiction check
GT_EVENT_DATE = "November 21, 2025"
GT_EVENT_VENUE = "Impact Challenger Hall"
GT_EVENT_LOCATION = "Pak Kret, Nonthaburi, Thailand"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class MU2025BestNationalCostumeExtraction(BaseModel):
    winner_name: Optional[str] = None
    winner_country: Optional[str] = None
    designer_name: Optional[str] = None
    costume_title: Optional[str] = None
    event_date: Optional[str] = None
    event_venue: Optional[str] = None
    event_location: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_mu2025_bnc() -> str:
    return (
        "Extract the information the answer provides specifically about the Miss Universe 2025 Best National Costume.\n"
        "Return a JSON object with the following fields (use null when the answer does not explicitly provide it):\n"
        "• winner_name: The person named as the Best National Costume winner in the answer.\n"
        "• winner_country: The country/territory the answer says the winner represented.\n"
        "• designer_name: The designer credited for the winning costume in the answer.\n"
        "• costume_title: The official title or name of the winning costume as written in the answer.\n"
        "• event_date: If the answer mentions a specific event date for Miss Universe 2025, extract it as written (otherwise null).\n"
        "• event_venue: If the answer mentions a venue/hall/arena for Miss Universe 2025, extract it as written (otherwise null).\n"
        "• event_location: If the answer mentions a city/area/province/country location for Miss Universe 2025, extract it as written (otherwise null).\n"
        "Important:\n"
        "- Extract exactly what the answer states; do not infer or correct.\n"
        "- Keep capitalization and punctuation as they appear in the answer.\n"
        "- If multiple variants are present, choose the main/official phrasing as presented in the answer.\n"
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Miss Universe 2025 Best National Costume task.
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_mu2025_bnc(),
        template_class=MU2025BestNationalCostumeExtraction,
        extraction_name="mu2025_bnc_extraction",
    )

    # Add ground truth info for traceability
    evaluator.add_ground_truth(
        {
            "expected_winner": f"{EXPECTED_WINNER_NAME} ({EXPECTED_WINNER_COUNTRY})",
            "expected_designer": EXPECTED_DESIGNER,
            "expected_costume_title": EXPECTED_COSTUME_TITLE,
            "expected_event_context": {
                "date": GT_EVENT_DATE,
                "venue": GT_EVENT_VENUE,
                "location": GT_EVENT_LOCATION,
            },
        },
        gt_type="ground_truth",
    )

    # Build top-level node from rubric
    top_node = evaluator.add_parallel(
        id="Miss_Universe_2025_Best_National_Costume_Information",
        desc="Check that the response provides the Best National Costume winner, costume designer, and official costume title/name for Miss Universe 2025.",
        parent=root,
        critical=True,
    )

    # 1) Winner check (critical)
    winner_leaf = evaluator.add_leaf(
        id="Winner_Is_Ahtisa_Manalo_Philippines",
        desc="Response identifies the Best National Costume winner as Ahtisa Manalo (representing the Philippines).",
        parent=top_node,
        critical=True,
    )
    wn = extracted.winner_name if extracted and extracted.winner_name else "UNSPECIFIED"
    wc = extracted.winner_country if extracted and extracted.winner_country else "UNSPECIFIED"
    winner_claim = (
        f"The answer states the Best National Costume winner at Miss Universe 2025 is '{wn}' "
        f"representing '{wc}'. Verify that this matches '{EXPECTED_WINNER_NAME}' representing 'Philippines'."
    )
    await evaluator.verify(
        claim=winner_claim,
        node=winner_leaf,
        additional_instruction=(
            "Judge if both the person and the country match. "
            "For the person, allow minor variations (case, spacing, middle names/initials) as long as it clearly refers to the same person. "
            "For the country, treat the following as equivalent to 'Philippines': 'the Philippines', 'Republic of the Philippines', "
            "'PH', 'PHL', 'Pilipinas'. If either field is unspecified/blank or refers to a different person/country, treat as a mismatch."
        ),
    )

    # 2) Designer check (critical)
    designer_leaf = evaluator.add_leaf(
        id="Designer_Is_Mak_Tumang",
        desc="Response identifies the costume designer as Mak Tumang.",
        parent=top_node,
        critical=True,
    )
    dn = extracted.designer_name if extracted and extracted.designer_name else "UNSPECIFIED"
    designer_claim = (
        f"The answer states the designer of the winning costume is '{dn}'. "
        f"Verify that this refers to '{EXPECTED_DESIGNER}'."
    )
    await evaluator.verify(
        claim=designer_claim,
        node=designer_leaf,
        additional_instruction=(
            "Allow minor variations in spacing/casing and presence/absence of diacritics. "
            "If the provided name is unspecified/blank or clearly a different person, mark incorrect."
        ),
    )

    # 3) Costume title check (critical)
    title_leaf = evaluator.add_leaf(
        id="Costume_Title_Is_Festejada_Queen_of_Philippine_Festivals",
        desc="Response gives the official costume title/name as 'Festejada: Queen of Philippine Festivals'.",
        parent=top_node,
        critical=True,
    )
    ct = extracted.costume_title if extracted and extracted.costume_title else "UNSPECIFIED"
    title_claim = (
        f"The answer states the official title/name of the winning costume is '{ct}'. "
        f"Verify that it matches '{EXPECTED_COSTUME_TITLE}'."
    )
    await evaluator.verify(
        claim=title_claim,
        node=title_leaf,
        additional_instruction=(
            "Compare for semantic and textual equivalence; allow minor punctuation/casing differences (e.g., colon vs dash), "
            "but the key components must be present and in the same sense: 'Festejada' and 'Queen of Philippine Festivals'. "
            "Do not accept paraphrases that drop or abbreviate essential words (e.g., 'PH' instead of 'Philippine') as exact matches. "
            "If unspecified/blank, mark incorrect."
        ),
    )

    # 4) Event context non-contradiction (non-critical)
    event_leaf = evaluator.add_leaf(
        id="Event_Context_No_Contradiction_If_Mentioned",
        desc="If the response mentions event date/venue details, they must not contradict: Miss Universe 2025 held Nov 21, 2025 at Impact Challenger Hall in Pak Kret, Nonthaburi, Thailand.",
        parent=top_node,
        critical=False,
    )
    ed = extracted.event_date if extracted and extracted.event_date else "UNSPECIFIED"
    ev = extracted.event_venue if extracted and extracted.event_venue else "UNSPECIFIED"
    el = extracted.event_location if extracted and extracted.event_location else "UNSPECIFIED"
    event_claim = (
        "Based on the answer, the event details for Miss Universe 2025 are: "
        f"date='{ed}', venue='{ev}', location='{el}'. "
        "Judge whether these details, if provided, do not contradict the ground truth: "
        f"held on {GT_EVENT_DATE} at {GT_EVENT_VENUE} in {GT_EVENT_LOCATION}. "
        "If a field is unspecified/blank in the answer, treat it as 'no contradiction'. "
        "If the answer explicitly states a different date (e.g., another day), a different venue (e.g., IMPACT Arena rather than Impact Challenger Hall), "
        "or a different location (e.g., Bangkok instead of Pak Kret, Nonthaburi), treat it as a contradiction."
    )
    await evaluator.verify(
        claim=event_claim,
        node=event_leaf,
        additional_instruction=(
            "Return Supported/Correct if there is no contradiction between the answer's mentioned details and the stated ground truth. "
            "If any mentioned detail conflicts, return Incorrect. "
            "Consider reasonable formatting/casing variants for the same venue (e.g., 'IMPACT Challenger Hall' vs 'Impact Challenger Hall') as equivalent, "
            "but do not equate 'IMPACT Arena' with 'Impact Challenger Hall'."
        ),
    )

    # Return structured summary
    return evaluator.get_summary()