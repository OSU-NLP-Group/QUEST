import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pace_emergency_notification_system"
TASK_DESCRIPTION = """Pace University, located in New York, maintains an emergency notification system for campus closures and other critical situations. According to Pace University's official emergency response policies, identify the following:

1. Which specific university position(s) or department(s) have the authority to make decisions about initiating emergency notifications, including campus closures?

2. What are the primary communication channels used by Pace University's emergency notification system? List at least three different types of communication methods that are used simultaneously during emergency alerts.

3. How frequently (how many times per year) is Pace University's emergency notification system required to be tested?

For each answer, provide a reference URL to official Pace University documentation that supports your response.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DecisionAuthority(BaseModel):
    positions: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class CommunicationChannels(BaseModel):
    channels: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class TestingPolicy(BaseModel):
    frequency_text: Optional[str] = None  # e.g., "once per semester", "annually"
    tests_per_year: Optional[str] = None  # keep as string to be robust (e.g., "2", "twice")
    sources: List[str] = Field(default_factory=list)


class PaceEmergencyExtraction(BaseModel):
    decision_authority: Optional[DecisionAuthority] = None
    communication: Optional[CommunicationChannels] = None
    testing: Optional[TestingPolicy] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pace_emergency() -> str:
    return """
    Extract the specific information asked about Pace University's emergency notification system from the answer text.

    You must return a JSON object with the following fields:
    - decision_authority: 
        - positions: an array of specific university position titles or department names that the answer claims have the authority to initiate emergency notifications (including campus closures).
        - sources: an array of URLs cited in the answer that support the authority identification. Prefer official Pace University URLs (pace.edu or closely related official subdomains).
    - communication:
        - channels: an array of distinct communication methods claimed to be used (e.g., text messages/SMS, email, voice calls/phone, website alerts, social media, mobile app/push notifications, siren/public address, digital signage).
        - sources: an array of URLs cited in the answer that describe these communication channels; prefer official Pace University documentation.
    - testing:
        - frequency_text: the exact phrase about testing frequency as stated in the answer (e.g., "once per semester", "twice per year", "annually").
        - tests_per_year: a normalized textual number of tests per year mentioned in the answer (e.g., "2", "two", "at least two"); if unclear or missing, set to null.
        - sources: an array of URLs cited in the answer that discuss the testing frequency; prefer official Pace University documentation.

    Rules:
    - Only extract items explicitly mentioned in the answer text.
    - For URLs: extract actual URLs present (plain or in markdown). If none are present for a section, return an empty array.
    - Avoid inventing or inferring any information not explicitly present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _list_to_english(items: List[str]) -> str:
    items = [s.strip() for s in items if s and s.strip()]
    if not items:
        return "none"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _has_sources(sources: Optional[List[str]]) -> bool:
    return bool(sources and len([u for u in sources if isinstance(u, str) and u.strip()]))


def _no_sources_instruction() -> str:
    return ("No source URLs were provided in the answer for this check. "
            "You must consider the claim not supported and return Incorrect.")


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_decision_authority_subtree(
    evaluator: Evaluator,
    parent_node,
    extracted: PaceEmergencyExtraction
) -> None:
    # Create node for Decision Authority (critical)
    da_node = evaluator.add_parallel(
        id="Decision_Authority",
        desc="Verify proper university officials with designated authority are identified for emergency notification decisions",
        parent=parent_node,
        critical=True
    )

    positions: List[str] = []
    sources: List[str] = []
    if extracted.decision_authority:
        positions = extracted.decision_authority.positions or []
        sources = extracted.decision_authority.sources or []

    positions_text = _list_to_english(positions)

    # Leaf: Authority Identification (critical)
    auth_id_leaf = evaluator.add_leaf(
        id="Authority_Identification",
        desc="Identify the specific university position(s) or department(s) with authority to make emergency notification decisions",
        parent=da_node,
        critical=True
    )
    claim_auth = (
        f"According to official Pace University documentation, the following position(s) or department(s) have authority "
        f"to initiate emergency notifications (including campus closures): {positions_text}."
    )
    add_ins_auth = (
        "Confirm the page(s) explicitly identify which positions or departments can make decisions to initiate "
        "emergency notifications or campus closures. Accept reasonable synonyms (e.g., Public Safety/University Safety, "
        "Office of Emergency Management/Emergency Management)."
    )
    if not _has_sources(sources):
        add_ins_auth = add_ins_auth + " " + _no_sources_instruction()

    await evaluator.verify(
        claim=claim_auth,
        node=auth_id_leaf,
        sources=sources if _has_sources(sources) else None,
        additional_instruction=add_ins_auth
    )

    # Leaf: Authority Reference (critical)
    auth_ref_leaf = evaluator.add_leaf(
        id="Authority_Reference",
        desc="Provide URL reference to official university documentation supporting the authority identification",
        parent=da_node,
        critical=True
    )
    claim_ref_auth = (
        "This URL is an official Pace University documentation page that states who has the authority to initiate "
        "emergency notifications or campus closures."
    )
    add_ins_ref_auth = (
        "Confirm the page is official Pace University documentation (typically pace.edu domain) and it discusses "
        "decision authority for emergency notification/closure initiation."
    )
    if not _has_sources(sources):
        add_ins_ref_auth = add_ins_ref_auth + " " + _no_sources_instruction()

    await evaluator.verify(
        claim=claim_ref_auth,
        node=auth_ref_leaf,
        sources=sources if _has_sources(sources) else None,
        additional_instruction=add_ins_ref_auth
    )


async def build_multi_channel_subtree(
    evaluator: Evaluator,
    parent_node,
    extracted: PaceEmergencyExtraction
) -> None:
    # Create node for Multi-Channel Communication (critical)
    mcc_node = evaluator.add_parallel(
        id="Multi_Channel_Communication",
        desc="Verify that the system uses multiple communication channels simultaneously",
        parent=parent_node,
        critical=True
    )

    channels: List[str] = []
    sources: List[str] = []
    if extracted.communication:
        channels = extracted.communication.channels or []
        sources = extracted.communication.sources or []

    channels_text = _list_to_english(channels)

    # Leaf: Minimum Channel Count (critical)
    min_count_leaf = evaluator.add_leaf(
        id="Minimum_Channel_Count",
        desc="Verify that at least three different types of communication methods are identified",
        parent=mcc_node,
        critical=True
    )
    claim_min_channels = (
        f"Pace University's emergency notification system uses at least three distinct communication methods "
        f"simultaneously during alerts, such as: {channels_text}."
    )
    add_ins_min_channels = (
        "Verify from the source(s) that there are three or more distinct alert channels used in a multi-channel manner "
        "(e.g., SMS/text, email, phone/voice, website, app/push, siren/PA, digital signage, social media). "
        "Treat obvious synonyms as the same channel; count distinct types. Confirm multi-channel/simultaneous usage."
    )
    if not _has_sources(sources):
        add_ins_min_channels = add_ins_min_channels + " " + _no_sources_instruction()

    await evaluator.verify(
        claim=claim_min_channels,
        node=min_count_leaf,
        sources=sources if _has_sources(sources) else None,
        additional_instruction=add_ins_min_channels
    )

    # Leaf: Communication Reference (critical)
    comm_ref_leaf = evaluator.add_leaf(
        id="Communication_Reference",
        desc="Provide URL reference to official university documentation describing the communication channels",
        parent=mcc_node,
        critical=True
    )
    claim_comm_ref = (
        "This is official Pace University documentation that lists the emergency alert communication channels and "
        "describes their multi-channel/simultaneous distribution."
    )
    add_ins_comm_ref = (
        "Confirm the page is official Pace University documentation and specifically mentions multiple alert channels "
        "and that alerts are distributed through them in a coordinated or simultaneous manner."
    )
    if not _has_sources(sources):
        add_ins_comm_ref = add_ins_comm_ref + " " + _no_sources_instruction()

    await evaluator.verify(
        claim=claim_comm_ref,
        node=comm_ref_leaf,
        sources=sources if _has_sources(sources) else None,
        additional_instruction=add_ins_comm_ref
    )


async def build_testing_requirement_subtree(
    evaluator: Evaluator,
    parent_node,
    extracted: PaceEmergencyExtraction
) -> None:
    # Create node for Testing Requirement (critical)
    tr_node = evaluator.add_parallel(
        id="Testing_Requirement",
        desc="Verify the testing frequency of the emergency notification system",
        parent=parent_node,
        critical=True
    )

    freq_text: Optional[str] = None
    sources: List[str] = []
    if extracted.testing:
        freq_text = extracted.testing.frequency_text
        sources = extracted.testing.sources or []

    # Leaf: Testing Frequency (critical)
    test_freq_leaf = evaluator.add_leaf(
        id="Testing_Frequency",
        desc="Verify that the system is tested at least twice per year",
        parent=tr_node,
        critical=True
    )
    claim_test_freq = (
        "Pace University's emergency notification system is required to be tested at least twice per year "
        "(for example, once each semester)."
    )
    add_ins_test_freq = (
        "Read the official documentation to determine the required testing frequency. "
        "If it states 'once per semester' or 'twice per year' or 'at least semiannually', mark Correct. "
        "If it only states 'annually' or less than twice per year, mark Incorrect."
    )
    if not _has_sources(sources):
        add_ins_test_freq = add_ins_test_freq + " " + _no_sources_instruction()

    await evaluator.verify(
        claim=claim_test_freq,
        node=test_freq_leaf,
        sources=sources if _has_sources(sources) else None,
        additional_instruction=add_ins_test_freq
    )

    # Leaf: Testing Reference (critical)
    test_ref_leaf = evaluator.add_leaf(
        id="Testing_Reference",
        desc="Provide URL reference to official university documentation supporting the testing frequency",
        parent=tr_node,
        critical=True
    )
    claim_test_ref = (
        "This URL is official Pace University documentation that explicitly states the required testing frequency "
        "for the emergency notification system."
    )
    add_ins_test_ref = (
        "Confirm the page is official Pace University documentation (pace.edu domain) and it clearly describes "
        "how often the emergency notification system must be tested."
    )
    if not _has_sources(sources):
        add_ins_test_ref = add_ins_test_ref + " " + _no_sources_instruction()

    await evaluator.verify(
        claim=claim_test_ref,
        node=test_ref_leaf,
        sources=sources if _has_sources(sources) else None,
        additional_instruction=add_ins_test_ref
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Pace University emergency notification system policy compliance task.
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_pace_emergency(),
        template_class=PaceEmergencyExtraction,
        extraction_name="pace_emergency_extraction",
    )

    # Build compliance root node (critical, parallel aggregation)
    compliance_node = evaluator.add_parallel(
        id="Emergency_Notification_System_Compliance",
        desc="Evaluate whether the described emergency notification system meets the requirements for university emergency communications",
        parent=root,
        critical=True
    )

    # Build subtrees
    await build_decision_authority_subtree(evaluator, compliance_node, extracted)
    await build_multi_channel_subtree(evaluator, compliance_node, extracted)
    await build_testing_requirement_subtree(evaluator, compliance_node, extracted)

    # Return structured summary
    return evaluator.get_summary()