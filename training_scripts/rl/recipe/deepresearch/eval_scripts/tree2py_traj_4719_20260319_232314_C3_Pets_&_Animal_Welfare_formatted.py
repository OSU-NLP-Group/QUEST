import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ehv1_tx_outbreak_2025"
TASK_DESCRIPTION = (
    "In November 2025, a significant multi-state outbreak of Equine Herpesvirus-1 (EHV-1) and its neurologic form "
    "(EHM) was traced back to a major equine event held in Texas. As an equine facility manager evaluating potential "
    "exposure risks for your horses, you need to compile a comprehensive report documenting this outbreak's origin "
    "and the regulatory response.\n\n"
    "Conduct a thorough investigation to identify and document the following information, with each piece building "
    "upon the previous finding:\n\n"
    "1. The official name of the equine event in Texas where the November 2025 EHV-1 outbreak originated\n"
    "2. The specific venue name and city where this event was held\n"
    "3. The exact dates when the event took place\n"
    "4. The quarantine hold period (in days) that Texas authorities implemented for horses that attended this event\n"
    "5. The specific date when horses from this event would be cleared from quarantine if they showed no clinical signs "
    "(fever, respiratory problems, or neurologic symptoms), as stated in official veterinary guidance\n\n"
    "For each piece of information, provide a reference URL from official sources, veterinary organizations, equine "
    "disease tracking centers, news reports, or the event's official communications that confirms your findings."
)

# Ground-truth style expectations (for guidance and partial checks)
EXPECTED_EVENT_NAME_OPTIONS = [
    "WPRA World Finals",
    "Women’s Professional Rodeo Association World Finals",
    "Women's Professional Rodeo Association World Finals",
    "WPRA Women’s World Finals",
    "WPRA Women's World Finals",
]
EXPECTED_VENUE_OPTIONS = [
    "Extraco Events Center",
    "Extraco Event Center",  # accept both common variants
]
EXPECTED_CITY_OPTIONS = [
    "Waco, Texas",
    "Waco, TX",
]
EXPECTED_EVENT_DATES = "November 5-9, 2025"
EXPECTED_QUARANTINE_DAYS = "21"
EXPECTED_CLEARANCE_DATE = "December 2, 2025"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class OutbreakExtraction(BaseModel):
    # Event identification
    event_name: Optional[str] = None
    event_urls: List[str] = Field(default_factory=list)

    # Venue and city
    venue_name: Optional[str] = None
    city: Optional[str] = None
    venue_urls: List[str] = Field(default_factory=list)

    # Event dates
    event_dates: Optional[str] = None
    date_urls: List[str] = Field(default_factory=list)

    # Quarantine protocol
    quarantine_days: Optional[str] = None
    protocol_urls: List[str] = Field(default_factory=list)

    # Clearance timeline
    clearance_date: Optional[str] = None
    clearance_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_outbreak() -> str:
    return (
        "Extract the required structured information about the November 2025 EHV-1/EHM outbreak linked to a Texas event "
        "from the provided answer text. Return JSON fields exactly as they appear in the answer (do not invent data). "
        "If a field is missing in the answer, set it to null (for strings) or an empty array (for URL lists).\n\n"
        "Fields to extract:\n"
        "1) event_name: The official name of the event (e.g., 'WPRA World Finals' or 'Women's Professional Rodeo Association World Finals').\n"
        "2) event_urls: An array of URLs that confirm the event name and its connection to the EHV-1/EHM outbreak.\n"
        "3) venue_name: The specific venue name (e.g., 'Extraco Event(s) Center').\n"
        "4) city: The city and state (e.g., 'Waco, Texas' or 'Waco, TX').\n"
        "5) venue_urls: An array of URLs confirming the venue and location details for the event.\n"
        "6) event_dates: The exact dates when the event took place (e.g., 'November 5-9, 2025').\n"
        "7) date_urls: An array of URLs confirming the event dates.\n"
        "8) quarantine_days: The quarantine hold period in days implemented by Texas authorities for horses that attended the event (e.g., '21' or '21 days').\n"
        "9) protocol_urls: An array of URLs confirming the Texas quarantine protocol and duration for attendees.\n"
        "10) clearance_date: The specific date when asymptomatic horses from this event would be cleared from quarantine (e.g., 'December 2, 2025').\n"
        "11) clearance_urls: An array of URLs confirming the clearance date for asymptomatic horses.\n\n"
        "URL extraction rules:\n"
        "- Only include URLs explicitly present in the answer text. Do not infer new URLs.\n"
        "- Accept plain URLs or markdown links; extract the actual link target.\n"
        "- Ensure each URL is complete and valid; prepend http:// if protocol missing.\n"
    )


# --------------------------------------------------------------------------- #
# Helper for URL-required verification                                        #
# --------------------------------------------------------------------------- #
async def verify_with_required_urls(
    evaluator: Evaluator,
    *,
    claim: str,
    node,
    urls: Optional[List[str]],
    additional_instruction: str = "None",
) -> None:
    """
    Verify a claim against provided URLs. If no URLs are provided, mark the node as failed directly.
    This enforces the source-grounding requirement for reference URL checks.
    """
    if urls and len(urls) > 0:
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction=additional_instruction,
        )
    else:
        node.score = 0.0
        node.status = "failed"


# --------------------------------------------------------------------------- #
# Tree construction and verification                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, parent_node, ext: OutbreakExtraction) -> None:
    """
    Build the verification tree according to the rubric and run all checks.
    All nodes under the 'complete' node are marked critical and organized sequentially.
    """

    # Complete investigation (critical sequential)
    complete_node = evaluator.add_sequential(
        id="Complete_EHV1_Outbreak_Investigation",
        desc="Successfully complete a comprehensive investigation of the November 2025 EHV-1 outbreak in Texas, tracing from the originating event through specific quarantine timeline details",
        parent=parent_node,
        critical=True,
    )

    # 1) Event Identification (critical sequential)
    event_ident = evaluator.add_sequential(
        id="Event_Identification",
        desc="Identify the specific equine event in Texas where the November 2025 EHV-1 outbreak originated",
        parent=complete_node,
        critical=True,
    )

    # 1.a) Event_Name (leaf, critical)
    event_name_leaf = evaluator.add_leaf(
        id="Event_Name",
        desc="Provide the official name of the event (WPRA World Finals or Women's Professional Rodeo Association World Finals)",
        parent=event_ident,
        critical=True,
    )
    # Simple textual equivalence check against accepted variants
    name_claim = (
        f"The extracted event name is acceptable. It must be equivalent to one of these canonical names: "
        f"{EXPECTED_EVENT_NAME_OPTIONS}. The extracted value is: '{ext.event_name}'. "
        f"Treat common abbreviations (e.g., 'WPRA World Finals', 'WPRA Women's World Finals') and minor punctuation/casing "
        f"differences as equivalent."
    )
    await evaluator.verify(
        claim=name_claim,
        node=event_name_leaf,
        additional_instruction="Judge based solely on string equivalence to the listed canonical variants with lenient matching rules; do not require external knowledge.",
    )

    # 1.b) Event_Reference_URL (leaf, critical)
    event_ref_leaf = evaluator.add_leaf(
        id="Event_Reference_URL",
        desc="Provide a valid reference URL confirming the event name and its connection to the EHV-1 outbreak",
        parent=event_ident,
        critical=True,
    )
    event_ref_claim = (
        f"At least one provided source explicitly indicates that the November 2025 EHV-1/EHM outbreak was associated "
        f"with or traced to the event '{ext.event_name}' in Texas."
    )
    await verify_with_required_urls(
        evaluator,
        claim=event_ref_claim,
        node=event_ref_leaf,
        urls=ext.event_urls,
        additional_instruction=(
            "Look for explicit linkage between the EHV-1/EHM outbreak (November 2025 timeframe) and the named event. "
            "Accept language like 'linked to', 'associated with', 'traced to', or 'connected to'."
        ),
    )

    # 2) Venue Identification (critical sequential)
    venue_ident = evaluator.add_sequential(
        id="Venue_Identification",
        desc="Identify the specific venue and city where the event was held",
        parent=event_ident,
        critical=True,
    )

    # 2.a) Venue_Details (leaf, critical)
    venue_details_leaf = evaluator.add_leaf(
        id="Venue_Details",
        desc="Provide the venue name (Extraco Event Center) and city (Waco, Texas)",
        parent=venue_ident,
        critical=True,
    )
    venue_claim = (
        f"The extracted venue '{ext.venue_name}' and city '{ext.city}' correspond to the Extraco Events Center "
        f"in Waco, Texas (accept 'Extraco Event Center' as a common variant; accept 'Waco, TX' for 'Waco, Texas')."
    )
    await evaluator.verify(
        claim=venue_claim,
        node=venue_details_leaf,
        additional_instruction="Judge based on textual equivalence with lenient matching (casing, pluralization 'Event(s)', and 'TX' vs 'Texas').",
    )

    # 2.b) Venue_Reference_URL (leaf, critical)
    venue_ref_leaf = evaluator.add_leaf(
        id="Venue_Reference_URL",
        desc="Provide a valid reference URL confirming the venue and location details",
        parent=venue_ident,
        critical=True,
    )
    venue_ref_claim = (
        f"At least one provided source explicitly confirms that the event '{ext.event_name}' took place at "
        f"'{ext.venue_name}' in '{ext.city}'."
    )
    await verify_with_required_urls(
        evaluator,
        claim=venue_ref_claim,
        node=venue_ref_leaf,
        urls=ext.venue_urls,
        additional_instruction="The source should mention both the venue and the city/state for the event in question.",
    )

    # 3) Event Timing (critical sequential)
    event_timing = evaluator.add_sequential(
        id="Event_Timing",
        desc="Determine when the event took place",
        parent=venue_ident,
        critical=True,
    )

    # 3.a) Event_Dates (leaf, critical)
    dates_leaf = evaluator.add_leaf(
        id="Event_Dates",
        desc="Provide the specific dates of the event (November 5-9, 2025)",
        parent=event_timing,
        critical=True,
    )
    dates_claim = (
        f"The extracted event dates '{ext.event_dates}' match the official dates '{EXPECTED_EVENT_DATES}'. "
        f"Allow reasonable formatting variations such as en-dash, hyphen, or 'Nov.' abbreviation."
    )
    await evaluator.verify(
        claim=dates_claim,
        node=dates_leaf,
        additional_instruction="Judge based on textual equivalence with minor formatting variations allowed.",
    )

    # 3.b) Dates_Reference_URL (leaf, critical)
    dates_ref_leaf = evaluator.add_leaf(
        id="Dates_Reference_URL",
        desc="Provide a valid reference URL confirming the event dates",
        parent=event_timing,
        critical=True,
    )
    dates_ref_claim = (
        f"At least one provided source explicitly confirms that the event '{ext.event_name}' took place on "
        f"'{EXPECTED_EVENT_DATES}' (or equivalent formatting)."
    )
    await verify_with_required_urls(
        evaluator,
        claim=dates_ref_claim,
        node=dates_ref_leaf,
        urls=ext.date_urls,
        additional_instruction="Confirm the event schedule matches the specified date range; accept formatting variants.",
    )

    # 4) Quarantine Protocol (critical sequential)
    quarantine = evaluator.add_sequential(
        id="Quarantine_Protocol",
        desc="Identify the quarantine duration implemented by Texas authorities for horses that attended the event",
        parent=event_timing,
        critical=True,
    )

    # 4.a) Quarantine_Duration (leaf, critical)
    quarantine_duration_leaf = evaluator.add_leaf(
        id="Quarantine_Duration",
        desc="Specify the quarantine hold period in days (21 days)",
        parent=quarantine,
        critical=True,
    )
    q_claim = (
        f"The extracted quarantine hold period '{ext.quarantine_days}' indicates a 21-day duration "
        f"(accept '21', '21 days', or '21-day')."
    )
    await evaluator.verify(
        claim=q_claim,
        node=quarantine_duration_leaf,
        additional_instruction="Judge true if the extracted text clearly denotes a 21-day quarantine period.",
    )

    # 4.b) Protocol_Reference_URL (leaf, critical)
    protocol_ref_leaf = evaluator.add_leaf(
        id="Protocol_Reference_URL",
        desc="Provide a valid reference URL confirming the Texas quarantine protocol and duration",
        parent=quarantine,
        critical=True,
    )
    protocol_ref_claim = (
        f"At least one provided source (preferably from Texas animal health or veterinary authorities, or equivalent "
        f"authoritative entities) explicitly states that horses that attended '{ext.event_name}' are to complete a "
        f"21-day quarantine/hold period."
    )
    await verify_with_required_urls(
        evaluator,
        claim=protocol_ref_claim,
        node=protocol_ref_leaf,
        urls=ext.protocol_urls,
        additional_instruction="Accept 'quarantine', 'isolation', 'movement hold', or equivalent language explicitly tied to 21 days.",
    )

    # 5) Clearance Timeline (critical sequential)
    clearance = evaluator.add_sequential(
        id="Clearance_Timeline",
        desc="Determine the specific date when horses without clinical signs would be cleared from quarantine",
        parent=quarantine,
        critical=True,
    )

    # 5.a) Clearance_Date (leaf, critical)
    clearance_date_leaf = evaluator.add_leaf(
        id="Clearance_Date",
        desc="Provide the specific clearance date as stated in official veterinary guidance (December 2, 2025)",
        parent=clearance,
        critical=True,
    )
    c_claim = (
        f"The extracted clearance date '{ext.clearance_date}' matches the expected clearance date "
        f"'{EXPECTED_CLEARANCE_DATE}' for asymptomatic horses associated with the event."
    )
    await evaluator.verify(
        claim=c_claim,
        node=clearance_date_leaf,
        additional_instruction="Judge true if the extracted date equals the expected date allowing minor formatting differences (e.g., 'Dec 2, 2025').",
    )

    # 5.b) Timeline_Reference_URL (leaf, critical)
    timeline_ref_leaf = evaluator.add_leaf(
        id="Timeline_Reference_URL",
        desc="Provide a valid reference URL confirming the clearance date for asymptomatic horses",
        parent=clearance,
        critical=True,
    )
    timeline_ref_claim = (
        f"At least one provided source explicitly states that horses from '{ext.event_name}' without clinical signs "
        f"(no fever, respiratory, or neurologic symptoms) would be cleared from quarantine on '{EXPECTED_CLEARANCE_DATE}'."
    )
    await verify_with_required_urls(
        evaluator,
        claim=timeline_ref_claim,
        node=timeline_ref_leaf,
        urls=ext.clearance_urls,
        additional_instruction="Source should explicitly mention clearance timing for asymptomatic horses; minor wording variations are acceptable.",
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
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the November 2025 Texas EHV-1 outbreak investigation task.
    Returns the standardized evaluation summary dictionary.
    """
    # Initialize evaluator (root is a non-critical container)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_outbreak(),
        template_class=OutbreakExtraction,
        extraction_name="outbreak_extraction",
    )

    # Add ground-truth expectations for transparency (not used to auto-grade)
    evaluator.add_ground_truth(
        {
            "expected_event_name_options": EXPECTED_EVENT_NAME_OPTIONS,
            "expected_venue_options": EXPECTED_VENUE_OPTIONS,
            "expected_city_options": EXPECTED_CITY_OPTIONS,
            "expected_event_dates": EXPECTED_EVENT_DATES,
            "expected_quarantine_days": EXPECTED_QUARANTINE_DAYS,
            "expected_clearance_date": EXPECTED_CLEARANCE_DATE,
        },
        gt_type="expected_values",
    )

    # Build tree and verify
    await build_and_verify_tree(evaluator, root, extracted)

    # Return standard summary
    return evaluator.get_summary()