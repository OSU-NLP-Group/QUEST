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
TASK_ID = "the_harder_they_come_london_autumn_2025"
TASK_DESCRIPTION = (
    "I'm planning to take a school group to see the reggae musical 'The Harder They Come' in London during autumn 2025. "
    "I need to identify the venue name and complete address, confirm the show dates, understand the running time and age guidance, "
    "and obtain the box office phone number for booking inquiries."
)

# Ground truth (expected values to verify)
GROUND_TRUTH = {
    "production_title": "The Harder They Come",
    "venue_name": "Theatre Royal Stratford East",
    "venue_address": "Gerry Raffles Square, London E15 1BN",
    "show_dates": {
        "start": "13 September 2025",
        "end": "1 November 2025",
        "canonical": "from September 13, 2025 to November 1, 2025"
    },
    "running_time": "approximately 2 hours 30 minutes plus an interval",
    "age_guidance": "14+",
    "box_office_phone": "020 8534 0310",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProductionDetailsExtraction(BaseModel):
    """Structured information extracted from the agent's answer for the 2025 London production."""
    production_title: Optional[str] = None
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None
    show_start_date: Optional[str] = None
    show_end_date: Optional[str] = None
    running_time: Optional[str] = None
    age_guidance: Optional[str] = None
    box_office_phone: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_production_details() -> str:
    return """
    Extract the production details for the London run of 'The Harder They Come' in autumn 2025 as stated in the answer.

    Return a JSON object with the following fields (use null if a field is not explicitly present in the answer):
    - production_title: the title of the production or musical
    - venue_name: the venue/theatre name
    - venue_address: the complete postal address of the venue
    - show_start_date: the first date of the run (as written in the answer)
    - show_end_date: the final date of the run (as written in the answer)
    - running_time: the running time text (e.g., 'approximately 2 hours 30 minutes plus an interval')
    - age_guidance: the age guidance (e.g., '14+')
    - box_office_phone: the box office phone number text
    - source_urls: array of URL(s) explicitly cited in the answer that support these details (production page, venue page, etc.). Include only URLs shown in the answer (plain URLs or in markdown). If none are present, return an empty array.

    Do not invent data. Extract exactly what appears in the answer text.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_production_details(
    evaluator: Evaluator,
    parent_node,
    extracted: ProductionDetailsExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and run evidence-based checks.
    """
    # Create the main node as a critical parallel aggregator
    prod_root = evaluator.add_parallel(
        id="The_Harder_They_Come_Production_Details",
        desc="Verify all required details for the London production of 'The Harder They Come' in autumn 2025.",
        parent=parent_node,
        critical=True,
    )

    # Venue details group (critical parallel)
    venue_group = evaluator.add_parallel(
        id="Venue_Details",
        desc="Confirms the venue name and complete address.",
        parent=prod_root,
        critical=True,
    )

    # Prepare sources (URLs explicitly provided in the answer)
    sources = extracted.source_urls if extracted.source_urls else None

    # Leaf nodes creation
    production_title_node = evaluator.add_leaf(
        id="Production_Title",
        desc="Confirms the theatrical production is 'The Harder They Come' musical.",
        parent=prod_root,
        critical=True,
    )
    venue_name_node = evaluator.add_leaf(
        id="Venue_Name",
        desc="Venue name is Theatre Royal Stratford East.",
        parent=venue_group,
        critical=True,
    )
    venue_address_node = evaluator.add_leaf(
        id="Venue_Address",
        desc="Venue complete address is Gerry Raffles Square, London E15 1BN.",
        parent=venue_group,
        critical=True,
    )
    show_dates_node = evaluator.add_leaf(
        id="Show_Dates",
        desc="Show dates are from September 13 to November 1, 2025.",
        parent=prod_root,
        critical=True,
    )
    running_time_node = evaluator.add_leaf(
        id="Running_Time",
        desc="Running time is approximately 2 hours 30 minutes plus an interval.",
        parent=prod_root,
        critical=True,
    )
    age_guidance_node = evaluator.add_leaf(
        id="Age_Guidance",
        desc="Age guidance is 14+.",
        parent=prod_root,
        critical=True,
    )
    box_office_contact_node = evaluator.add_leaf(
        id="Box_Office_Contact",
        desc="Box office phone number is 020 8534 0310.",
        parent=prod_root,
        critical=True,
    )

    # Build claims aligned with rubric expectations
    claim_title = "The theatrical production is 'The Harder They Come' (the reggae stage musical)."
    claim_venue_name = "The venue name for this production is Theatre Royal Stratford East."
    claim_venue_address = "The venue's postal address is Gerry Raffles Square, London E15 1BN."
    claim_show_dates = "The production runs from September 13, 2025 to November 1, 2025."
    claim_running_time = "The running time is approximately 2 hours 30 minutes plus an interval."
    claim_age_guidance = "The age guidance is 14+."
    claim_box_office = "The box office phone number is 020 8534 0310."

    # Additional instructions per check to guide the LLM judge
    addins_title = (
        "Use the cited webpage(s) to confirm the production title is 'The Harder They Come'. "
        "This refers to the reggae musical adaptation. Also ensure the answer itself asserts this exact title; "
        "minor phrasing variants are acceptable."
    )
    addins_venue_name = (
        "Confirm the official venue for this production is Theatre Royal Stratford East. "
        "Allow minor naming variations (e.g., 'TRSE', 'Theatre Royal, Stratford East'), but they must refer to the same theatre. "
        "If the sources imply another venue, mark incorrect."
    )
    addins_venue_address = (
        "Confirm the theatre's postal address is 'Gerry Raffles Square, London E15 1BN'. "
        "Allow minor formatting variations (commas/spaces), but the address must match."
    )
    addins_show_dates = (
        "Confirm that the run dates are from 13 September 2025 to 1 November 2025. "
        "Equivalent formats (e.g., '13 Sep – 1 Nov 2025', '13/09/2025 to 01/11/2025') are acceptable."
    )
    addins_running_time = (
        "Confirm the running time is about 2 hours 30 minutes plus an interval. "
        "Accept near-equivalent phrasings such as 'about 2h30 inc. interval' or '2 hours 30 minutes (incl. interval)'."
    )
    addins_age_guidance = (
        "Confirm that the age guidance is 14+. "
        "Accept 'recommended age 14+' or 'strictly 14+'."
    )
    addins_box_office = (
        "Confirm the Theatre Royal Stratford East box office phone number is '020 8534 0310'. "
        "Allow UK formatting/spaces variations, e.g., '020-8534-0310' or '020 85340310'."
    )

    # Batch verify in parallel where appropriate
    await evaluator.batch_verify([
        (claim_title, sources, production_title_node, addins_title),
        (claim_venue_name, sources, venue_name_node, addins_venue_name),
        (claim_venue_address, sources, venue_address_node, addins_venue_address),
        (claim_show_dates, sources, show_dates_node, addins_show_dates),
        (claim_running_time, sources, running_time_node, addins_running_time),
        (claim_age_guidance, sources, age_guidance_node, addins_age_guidance),
        (claim_box_office, sources, box_office_contact_node, addins_box_office),
    ])


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
    Evaluate the agent's answer for the London production details of 'The Harder They Come' in autumn 2025.
    Returns a structured summary with verification tree and final score.
    """
    # Initialize evaluator (root is non-critical by design; we'll add a critical child node)
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

    # Extract structured details from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_production_details(),
        template_class=ProductionDetailsExtraction,
        extraction_name="production_details",
    )

    # Record ground truth expectations
    evaluator.add_ground_truth(
        gt_info=GROUND_TRUTH,
        gt_type="expected_details",
    )

    # Build verification tree and run checks
    await verify_production_details(evaluator, root, extracted)

    # Return final summary
    return evaluator.get_summary()