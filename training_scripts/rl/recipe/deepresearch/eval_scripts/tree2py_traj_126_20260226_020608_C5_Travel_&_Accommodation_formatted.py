import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pre_travel_turkish_jfk_sg"
TASK_DESCRIPTION = (
    "I am a US citizen planning to fly from New York's John F. Kennedy International Airport (JFK) to Singapore "
    "with a connection in Istanbul, traveling on Turkish Airlines in Economy Class. I need to prepare for my trip and "
    "would like to know: (1) Which terminal does Turkish Airlines use at JFK, and where is their lounge located within that terminal? "
    "(2) What are the baggage allowances for Economy Class, including both checked baggage weight limits and carry-on baggage weight and "
    "dimension limits? (3) What are the entry requirements for US citizens traveling to Singapore, including visa requirements, passport "
    "validity requirements, and any arrival documentation that must be submitted? (4) What are the recommended timing guidelines for arriving "
    "at JFK before an international flight and the check-in deadline? Please provide reference URLs to support each category of information."
)


# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class TerminalInfo(BaseModel):
    terminal: Optional[str] = None
    lounge_location: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BaggageInfo(BaseModel):
    checked_baggage_limit: Optional[str] = None
    carry_on_weight_limit: Optional[str] = None
    carry_on_dimensions_limit: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class EntryRequirements(BaseModel):
    visa_requirement: Optional[str] = None
    passport_validity: Optional[str] = None
    sg_arrival_card: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DepartureTiming(BaseModel):
    airport_arrival_time: Optional[str] = None
    check_in_deadline: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PreTravelExtraction(BaseModel):
    terminal_info: Optional[TerminalInfo] = None
    baggage_info: Optional[BaggageInfo] = None
    entry_info: Optional[EntryRequirements] = None
    timing_info: Optional[DepartureTiming] = None


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pretravel() -> str:
    return (
        "Extract the structured pre-travel details the answer provides, organized into four categories. "
        "For each category, extract both the specific statements and the URL sources that support that category. "
        "If any item is missing, return null for that field or an empty list for sources.\n\n"
        "Categories and fields to extract:\n"
        "1) terminal_info:\n"
        "   - terminal: The JFK terminal Turkish Airlines uses (e.g., 'Terminal 1').\n"
        "   - lounge_location: The described location of the Turkish Airlines lounge within that terminal (e.g., 'between gates 2 and 3').\n"
        "   - sources: Array of URLs the answer cites specifically for terminal and lounge information.\n\n"
        "2) baggage_info:\n"
        "   - checked_baggage_limit: The Economy Class checked baggage limit described (e.g., '23kg per piece').\n"
        "   - carry_on_weight_limit: The Economy Class carry-on weight limit (e.g., '8kg').\n"
        "   - carry_on_dimensions_limit: The Economy Class carry-on dimension limits (e.g., '55x40x23 cm').\n"
        "   - sources: Array of URLs the answer cites for Turkish Airlines baggage policy.\n\n"
        "3) entry_info:\n"
        "   - visa_requirement: The visa rule for US citizens entering Singapore (e.g., 'No visa required for stays under 90 days').\n"
        "   - passport_validity: The passport validity requirement (e.g., 'Passport must be valid for at least 6 months from arrival').\n"
        "   - sg_arrival_card: The SG Arrival Card submission window (e.g., 'must be submitted within 3 days before arrival').\n"
        "   - sources: Array of URLs the answer cites for Singapore entry requirements.\n\n"
        "4) timing_info:\n"
        "   - airport_arrival_time: Recommended advance arrival time at JFK before an international flight (e.g., 'at least 3 hours').\n"
        "   - check_in_deadline: The check-in deadline prior to departure (e.g., 'at least 1 hour before scheduled departure').\n"
        "   - sources: Array of URLs the answer cites for timing recommendations.\n\n"
        "URL extraction rules:\n"
        "- Extract only actual URLs explicitly present in the answer (including Markdown links). Do not infer or invent URLs.\n"
        "- Normalize URLs and include protocol. If missing, prepend 'http://'.\n"
        "- Place each URL under the relevant category's 'sources' array; do not mix categories.\n"
    )


# --------------------------------------------------------------------------- #
# Helper Functions                                                            #
# --------------------------------------------------------------------------- #
def has_valid_urls(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    return any(isinstance(u, str) and u.strip().lower().startswith(("http://", "https://")) for u in urls)


# --------------------------------------------------------------------------- #
# Verification Subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_terminal_information(
    evaluator: Evaluator,
    parent_node,
    terminal_info: Optional[TerminalInfo],
) -> None:
    """
    Build and verify the 'Airport_Terminal_Information' subtree.
    """
    node = evaluator.add_parallel(
        id="Airport_Terminal_Information",
        desc="Identifies the correct JFK terminal and lounge location for Turkish Airlines",
        parent=parent_node,
        critical=False,
    )

    sources = terminal_info.sources if terminal_info else []

    # Terminal_Reference_URL (existence/validity of sources) - critical
    url_ok = has_valid_urls(sources)
    url_node = evaluator.add_custom_node(
        result=url_ok,
        id="Terminal_Reference_URL",
        desc="Provides a valid URL supporting the terminal information",
        parent=node,
        critical=True
    )

    # JFK_Terminal - critical, verify with sources
    term_leaf = evaluator.add_leaf(
        id="JFK_Terminal",
        desc="Specifies that Turkish Airlines operates from Terminal 1 at JFK Airport",
        parent=node,
        critical=True
    )
    term_claim = "Turkish Airlines operates from Terminal 1 at John F. Kennedy International Airport (JFK)."
    await evaluator.verify(
        claim=term_claim,
        node=term_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm the airline's terminal assignment at JFK on official sources (airport or airline). "
            "Treat 'T1' and 'Terminal 1' equivalently."
        ),
        extra_prerequisites=[url_node]
    )

    # Lounge_Location - critical, verify with sources
    lounge_leaf = evaluator.add_leaf(
        id="Lounge_Location",
        desc="States the Turkish Airlines lounge is located between gates 2 and 3 in Terminal 1",
        parent=node,
        critical=True
    )
    lounge_claim = "The Turkish Airlines lounge in JFK Terminal 1 is located between gates 2 and 3."
    await evaluator.verify(
        claim=lounge_claim,
        node=lounge_leaf,
        sources=sources,
        additional_instruction=(
            "Look for official lounge details indicating location between gates 2 and 3 in Terminal 1. "
            "Allow minor wording variations like 'near gates 2 & 3'."
        ),
        extra_prerequisites=[url_node]
    )


async def verify_baggage_allowances(
    evaluator: Evaluator,
    parent_node,
    baggage_info: Optional[BaggageInfo],
) -> None:
    """
    Build and verify the 'Baggage_Allowances' subtree.
    """
    node = evaluator.add_parallel(
        id="Baggage_Allowances",
        desc="Specifies Economy Class baggage limits for checked and carry-on luggage",
        parent=parent_node,
        critical=False
    )

    sources = baggage_info.sources if baggage_info else []

    # Baggage_Reference_URL - critical existence/validity
    url_ok = has_valid_urls(sources)
    url_node = evaluator.add_custom_node(
        result=url_ok,
        id="Baggage_Reference_URL",
        desc="Provides a valid URL supporting the baggage allowance information",
        parent=node,
        critical=True
    )

    # Checked_Baggage_Limit - critical
    checked_leaf = evaluator.add_leaf(
        id="Checked_Baggage_Limit",
        desc="States the checked baggage weight limit is maximum 23kg per piece for Economy Class",
        parent=node,
        critical=True
    )
    checked_claim = "For Economy Class, the checked baggage weight limit is a maximum of 23 kg per piece."
    await evaluator.verify(
        claim=checked_claim,
        node=checked_leaf,
        sources=sources,
        additional_instruction=(
            "Verify on Turkish Airlines official baggage policy pages. Minor phrasing variations are acceptable "
            "as long as the per-piece limit is clearly 23 kg for Economy."
        ),
        extra_prerequisites=[url_node]
    )

    # Carry_On_Weight - critical
    carry_weight_leaf = evaluator.add_leaf(
        id="Carry_On_Weight",
        desc="States the carry-on baggage weight limit is maximum 8kg",
        parent=node,
        critical=True
    )
    carry_weight_claim = "The carry-on baggage weight limit for Economy Class is a maximum of 8 kg."
    await evaluator.verify(
        claim=carry_weight_claim,
        node=carry_weight_leaf,
        sources=sources,
        additional_instruction=(
            "Use Turkish Airlines cabin baggage policy. Accept that some routes or fares may differ, "
            "but the general carry-on weight cap should be 8 kg for Economy."
        ),
        extra_prerequisites=[url_node]
    )

    # Carry_On_Dimensions - critical
    carry_dim_leaf = evaluator.add_leaf(
        id="Carry_On_Dimensions",
        desc="States the carry-on baggage dimensions must not exceed 55x40x23 cm",
        parent=node,
        critical=True
    )
    carry_dim_claim = "Carry-on baggage dimensions must not exceed 55 x 40 x 23 cm."
    await evaluator.verify(
        claim=carry_dim_claim,
        node=carry_dim_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm on official Turkish Airlines sources. Accept minor wording variations and equivalently formatted dimensions."
        ),
        extra_prerequisites=[url_node]
    )


async def verify_entry_requirements(
    evaluator: Evaluator,
    parent_node,
    entry_info: Optional[EntryRequirements],
) -> None:
    """
    Build and verify the 'Singapore_Entry_Requirements' subtree.
    """
    node = evaluator.add_parallel(
        id="Singapore_Entry_Requirements",
        desc="Details the entry requirements for US citizens traveling to Singapore",
        parent=parent_node,
        critical=False
    )

    sources = entry_info.sources if entry_info else []

    # Entry_Reference_URL - critical existence/validity
    url_ok = has_valid_urls(sources)
    url_node = evaluator.add_custom_node(
        result=url_ok,
        id="Entry_Reference_URL",
        desc="Provides a valid URL supporting the Singapore entry requirements",
        parent=node,
        critical=True
    )

    # Visa_Requirement - critical
    visa_leaf = evaluator.add_leaf(
        id="Visa_Requirement",
        desc="States that US citizens do not require a visa for Singapore stays under 90 days",
        parent=node,
        critical=True
    )
    visa_claim = "US citizens do not require a visa for short stays in Singapore under 90 days."
    await evaluator.verify(
        claim=visa_claim,
        node=visa_leaf,
        sources=sources,
        additional_instruction=(
            "Prefer Singapore ICA or official government sources. Allow mention such as 'tourist stays' or 'short-term visits' "
            "as long as the policy indicates visa-free entry for US citizens up to 90 days."
        ),
        extra_prerequisites=[url_node]
    )

    # Passport_Validity - critical
    passport_leaf = evaluator.add_leaf(
        id="Passport_Validity",
        desc="States that passports must be valid for at least 6 months from arrival date",
        parent=node,
        critical=True
    )
    passport_claim = "Passports must be valid for at least six months from the arrival date in Singapore."
    await evaluator.verify(
        claim=passport_claim,
        node=passport_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm 6-month passport validity on ICA or government sources. Minor phrasing differences are acceptable."
        ),
        extra_prerequisites=[url_node]
    )

    # SG_Arrival_Card - critical
    sgac_leaf = evaluator.add_leaf(
        id="SG_Arrival_Card",
        desc="States that the SG Arrival Card must be submitted within 3 days before arrival",
        parent=node,
        critical=True
    )
    sgac_claim = "The SG Arrival Card must be submitted within three days before arrival in Singapore."
    await evaluator.verify(
        claim=sgac_claim,
        node=sgac_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm on official ICA sources. Allow 'up to 3 days' or 'within 3 days' phrasing variants."
        ),
        extra_prerequisites=[url_node]
    )


async def verify_departure_timing(
    evaluator: Evaluator,
    parent_node,
    timing_info: Optional[DepartureTiming],
) -> None:
    """
    Build and verify the 'Departure_Timing' subtree.
    """
    node = evaluator.add_parallel(
        id="Departure_Timing",
        desc="Provides recommended arrival and check-in timing for international departure from JFK",
        parent=parent_node,
        critical=False
    )

    sources = timing_info.sources if timing_info else []

    # Timing_Reference_URL - critical existence/validity
    url_ok = has_valid_urls(sources)
    url_node = evaluator.add_custom_node(
        result=url_ok,
        id="Timing_Reference_URL",
        desc="Provides a valid URL supporting the timing recommendations",
        parent=node,
        critical=True
    )

    # Airport_Arrival_Time - critical
    arrival_leaf = evaluator.add_leaf(
        id="Airport_Arrival_Time",
        desc="States that passengers should arrive at JFK at least 3 hours before international flight departure",
        parent=node,
        critical=True
    )
    arrival_claim = "Passengers should arrive at JFK at least three hours before an international flight departure."
    await evaluator.verify(
        claim=arrival_claim,
        node=arrival_leaf,
        sources=sources,
        additional_instruction=(
            "Prefer official airline or airport guidance (e.g., Turkish Airlines, JFK). Accept 'approximately 3 hours' or 'at least 3 hours'."
        ),
        extra_prerequisites=[url_node]
    )

    # Check_In_Deadline - critical
    checkin_leaf = evaluator.add_leaf(
        id="Check_In_Deadline",
        desc="States that check-in must be completed at least 1 hour before scheduled departure",
        parent=node,
        critical=True
    )
    checkin_claim = "Check-in must be completed at least one hour before the scheduled departure time."
    await evaluator.verify(
        claim=checkin_claim,
        node=checkin_leaf,
        sources=sources,
        additional_instruction=(
            "Verify via Turkish Airlines or JFK guidance. Allow minor phrasing variants such as 'no later than 60 minutes'."
        ),
        extra_prerequisites=[url_node]
    )


# --------------------------------------------------------------------------- #
# Main Evaluation Entry Point                                                 #
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
    Evaluate an answer for the comprehensive pre-travel information task (Turkish Airlines JFK to Singapore).
    """
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

    # Perform a single structured extraction covering all categories
    extracted = await evaluator.extract(
        prompt=prompt_extract_pretravel(),
        template_class=PreTravelExtraction,
        extraction_name="pre_travel_extraction",
    )

    # Create a top-level aggregation node to mirror the rubric root
    top_node = evaluator.add_parallel(
        id="Complete_Pre_Travel_Information",
        desc="Provides comprehensive pre-departure information for a US citizen flying JFK to Singapore via Istanbul on Turkish Airlines Economy Class",
        parent=root,
        critical=False  # Adjusted from rubric to comply with framework constraints
    )

    # Build and verify each category subtree
    await verify_terminal_information(evaluator, top_node, extracted.terminal_info)
    await verify_baggage_allowances(evaluator, top_node, extracted.baggage_info)
    await verify_entry_requirements(evaluator, top_node, extracted.entry_info)
    await verify_departure_timing(evaluator, top_node, extracted.timing_info)

    # Optional: add rubric expectations for transparency
    evaluator.add_ground_truth({
        "expected": {
            "terminal": "Terminal 1",
            "lounge_location": "Between gates 2 and 3 (Terminal 1)",
            "checked_baggage_limit": "23 kg per piece (Economy)",
            "carry_on_weight_limit": "8 kg",
            "carry_on_dimensions_limit": "55 x 40 x 23 cm",
            "visa_requirement": "US citizens visa-free for stays under 90 days",
            "passport_validity": "At least 6 months from arrival",
            "sg_arrival_card": "Submit within 3 days before arrival",
            "airport_arrival_time": "Arrive at least 3 hours before international departure",
            "check_in_deadline": "Complete check-in at least 1 hour before departure",
        }
    }, gt_type="rubric_expectations")

    # Return evaluation summary
    return evaluator.get_summary()