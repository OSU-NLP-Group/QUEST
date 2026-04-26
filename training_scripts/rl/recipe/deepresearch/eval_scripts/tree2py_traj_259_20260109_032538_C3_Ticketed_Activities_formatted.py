import asyncio
import logging
from typing import Optional, List, Dict

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "kennedy_center_largest_venue"
TASK_DESCRIPTION = """
In Washington, DC, there is a performance venue that serves as the largest performance space within the Kennedy Center complex. This venue was renovated in 1997 to create state-of-the-art acoustic facilities specifically designed for orchestral performances. Identify this venue by its official name and confirm its exact seating capacity. Provide a URL reference from an authoritative source (such as the Kennedy Center's official website) that confirms: (1) the venue is part of the Kennedy Center, (2) it is the largest performance space in the facility, (3) its seating capacity of 2,465, and (4) the 1997 renovation for acoustic improvements.
"""


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    """
    Extracted information from the agent's answer.
    """
    official_name: Optional[str] = None
    seating_capacity: Optional[str] = None

    # URLs categorized by the claim they support
    membership_urls: List[str] = Field(default_factory=list)             # confirms it is a Kennedy Center venue in DC
    largest_urls: List[str] = Field(default_factory=list)                # confirms it's the largest performance space in the facility
    capacity_urls: List[str] = Field(default_factory=list)               # confirms exact seating capacity (2,465)
    renovation_urls: List[str] = Field(default_factory=list)             # confirms 1997 renovation for acoustic improvements
    orchestral_design_urls: List[str] = Field(default_factory=list)      # confirms designed for orchestral/classical performances


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return """
    Extract the key details and the cited URLs from the answer.

    Return a JSON object with the following fields:
    - official_name: The official name of the venue identified in the answer (string or null).
    - seating_capacity: The seating capacity the answer claims (string, keep punctuation like commas, or null if not provided).
    - membership_urls: Array of URLs that explicitly confirm the venue is part of the Kennedy Center in Washington, DC.
    - largest_urls: Array of URLs that explicitly confirm the venue is the largest performance space within the Kennedy Center complex.
    - capacity_urls: Array of URLs that explicitly confirm the venue’s exact seating capacity of 2,465 (or 2465).
    - renovation_urls: Array of URLs that explicitly confirm the venue was renovated in 1997 for acoustic improvements (state-of-the-art acoustic facilities for orchestral performance).
    - orchestral_design_urls: Array of URLs that explicitly confirm the venue is designed for orchestral/classical music performances.

    IMPORTANT:
    - Extract only URLs explicitly present in the answer (plain URLs or markdown links).
    - Do not invent URLs.
    - If a required item is missing from the answer, set it to null (for strings) or [] (for URL arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _any_authoritative(urls: List[str]) -> bool:
    """
    Simple authority heuristic: at least one URL from Kennedy Center's official domain.
    """
    if not urls:
        return False
    for u in urls:
        if not u:
            continue
        lu = u.lower()
        if "kennedy-center.org" in lu:
            return True
    return False


def _clean_name(name: Optional[str]) -> str:
    return name.strip() if name else ""


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def _add_authoritative_check_and_verify(
    evaluator: Evaluator,
    parent_node,
    node_id_prefix: str,
    group_desc: str,
    urls: List[str],
    claim: str,
    add_ins: str,
) -> None:
    """
    Build a critical sub-node that:
    - checks an authoritative URL is provided (custom leaf),
    - verifies the claim content against the provided URLs (leaf -> verify()).
    """
    group = evaluator.add_parallel(
        id=node_id_prefix,
        desc=group_desc,
        parent=parent_node,
        critical=True
    )

    # Existence + authority check (critical)
    has_authoritative = _any_authoritative(urls)
    evaluator.add_custom_node(
        result=has_authoritative,
        id=f"{node_id_prefix}_authoritative_url_present",
        desc="At least one authoritative URL (Kennedy Center official domain) is provided for this claim",
        parent=group,
        critical=True
    )

    # Content verification leaf (critical)
    verify_node = evaluator.add_leaf(
        id=f"{node_id_prefix}_supported",
        desc="Claim is supported by the provided authoritative URL(s)",
        parent=group,
        critical=True
    )

    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=urls,  # Must be non-empty and authoritative to avoid being skipped by precondition
        additional_instruction=add_ins
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the Kennedy Center largest venue task.
    """
    # Initialize evaluator (root is non-critical container)
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
        default_model=model
    )

    # Extract structured info from the answer
    extracted: VenueExtraction = await evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction"
    )

    # Build the critical main node for this evaluation
    main = evaluator.add_parallel(
        id="venue_identification",
        desc="Identify the correct performance venue and verify all required specifications with authoritative URL evidence",
        parent=root,
        critical=True
    )

    # 1) Official name must be provided (critical)
    official_name_ok = bool(_clean_name(extracted.official_name))
    evaluator.add_custom_node(
        result=official_name_ok,
        id="venue_official_name",
        desc="Provides the official name of the venue (the specific venue being identified).",
        parent=main,
        critical=True
    )

    # Prepare name token for human-readable claims (will still be skipped if official name missing)
    venue_name = extracted.official_name or "the venue"

    # 2) Venue is part of the Kennedy Center in Washington, DC (critical)
    await _add_authoritative_check_and_verify(
        evaluator=evaluator,
        parent_node=main,
        node_id_prefix="kennedy_center_location_with_authoritative_url",
        group_desc="Includes an authoritative URL confirming the venue is part of the Kennedy Center in Washington, DC.",
        urls=extracted.membership_urls,
        claim=f"The venue '{venue_name}' is a venue within the John F. Kennedy Center for the Performing Arts in Washington, DC.",
        add_ins=(
            "Use only the content from the provided URL(s). Confirm that the page explicitly indicates that this venue "
            "is a venue (or performance space) of the Kennedy Center located in Washington, DC. "
            "If the page is not on the official Kennedy Center domain (kennedy-center.org) or does not explicitly support "
            "the claim, mark it as not supported."
        )
    )

    # 3) Largest performance space within the Kennedy Center (critical)
    await _add_authoritative_check_and_verify(
        evaluator=evaluator,
        parent_node=main,
        node_id_prefix="largest_performance_space_with_authoritative_url",
        group_desc="Includes an authoritative URL confirming the venue is the largest performance space within the Kennedy Center complex.",
        urls=extracted.largest_urls,
        claim=f"The venue '{venue_name}' is the largest performance space within the Kennedy Center complex.",
        add_ins=(
            "The webpage must clearly state that this is the largest performance space in the entire Kennedy Center complex. "
            "Accept equivalent phrasings like 'largest venue' if it is unambiguously about performance spaces. "
            "Do not infer from capacity alone; it must be explicit or clearly stated on the page."
        )
    )

    # 4) Seating capacity of 2,465 (critical)
    await _add_authoritative_check_and_verify(
        evaluator=evaluator,
        parent_node=main,
        node_id_prefix="seating_capacity_2465_with_authoritative_url",
        group_desc="Includes an authoritative URL confirming the venue’s exact seating capacity is 2,465.",
        urls=extracted.capacity_urls,
        claim=f"The seating capacity of '{venue_name}' is 2,465 (i.e., 2465 seats).",
        add_ins=(
            "Confirm that the page explicitly states the seating capacity as 2,465 (allow forms like '2,465 seats', '2465'). "
            "If the number on the page differs or is not stated, mark as not supported."
        )
    )

    # 5) Renovated in 1997 for acoustic improvements (critical)
    await _add_authoritative_check_and_verify(
        evaluator=evaluator,
        parent_node=main,
        node_id_prefix="renovated_1997_acoustic_improvements_with_authoritative_url",
        group_desc="Includes an authoritative URL confirming the venue's 1997 renovation for state-of-the-art acoustic facilities for orchestral performance.",
        urls=extracted.renovation_urls,
        claim=(
            f"In 1997, the venue '{venue_name}' underwent a renovation that created state-of-the-art acoustic facilities, "
            f"specifically improving acoustics for orchestral performance."
        ),
        add_ins=(
            "The page should explicitly reference a 1997 renovation and connect it to acoustic improvements "
            "or the creation of state-of-the-art acoustic facilities for orchestral performance. "
            "If the date or the acoustic purpose is missing, mark as not supported."
        )
    )

    # 6) Designed for orchestral/classical music performances (critical)
    await _add_authoritative_check_and_verify(
        evaluator=evaluator,
        parent_node=main,
        node_id_prefix="designed_for_orchestral_classical_with_authoritative_url",
        group_desc="Includes an authoritative URL confirming the venue is designed for orchestral/classical performances.",
        urls=extracted.orchestral_design_urls,
        claim=f"The venue '{venue_name}' is designed for orchestral or classical music performances.",
        add_ins=(
            "The page should explicitly indicate that the venue is intended for orchestral/classical music performance, "
            "or is the home/performance venue for an orchestra (e.g., National Symphony Orchestra), "
            "or otherwise directly designed for symphonic/orchestral concerts."
        )
    )

    # Optionally store some contextual info about the evaluation
    evaluator.add_custom_info(
        info={
            "note": "Authority heuristic accepts only kennedy-center.org domain as authoritative for this task.",
            "accepted_authoritative_domain": "kennedy-center.org"
        },
        info_type="policy",
        info_name="authority_criteria"
    )

    # Return the final structured summary
    return evaluator.get_summary()