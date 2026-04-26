import asyncio
import logging
import math
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "veto_override_state_and_vote_counts"
TASK_DESCRIPTION = """
In the United States, identify the state whose constitution establishes the following specific veto override requirements: 
(1) The state must have a bicameral legislature (two separate legislative chambers); 
(2) The state constitution must specify different veto override vote thresholds depending on the type of legislation being considered; 
(3) For regular (non-appropriation) bills, the veto override threshold must be a simple majority of the members elected to each chamber; 
(4) For budget bills or appropriation bills specifically, the veto override threshold must be two-thirds of the members elected to each chamber; 
(5) Provide the exact number of votes required in each legislative chamber to override a gubernatorial veto on an appropriation bill.
"""


# ----------------------------- Data Models --------------------------------- #
class ChamberInfo(BaseModel):
    """Information about a legislative chamber as provided in the answer."""
    name: Optional[str] = None
    members_elected_total: Optional[str] = None  # Keep as string to be robust; we'll parse to int
    appropriation_override_votes: Optional[str] = None  # String for robustness; parse to int
    sources: List[str] = Field(default_factory=list)


class VetoOverrideExtraction(BaseModel):
    """Structured extraction for veto override requirements."""
    state: Optional[str] = None
    bicameral_claim: Optional[str] = None  # Free text claim if present
    thresholds_based_on_elected_members: Optional[str] = None  # Free text claim if present
    regular_threshold_desc: Optional[str] = None  # e.g., "simple majority of members elected"
    appropriation_threshold_desc: Optional[str] = None  # e.g., "two-thirds of members elected"
    upper_chamber: Optional[ChamberInfo] = None
    lower_chamber: Optional[ChamberInfo] = None
    global_sources: List[str] = Field(default_factory=list)


# --------------------------- Extraction Prompt ----------------------------- #
def prompt_extract_veto_override_info() -> str:
    return """
    From the provided answer, extract the structured information needed to evaluate constitutional veto-override rules and exact vote counts.

    Extract the following fields:

    1) state: The specific U.S. state named in the answer as the candidate that satisfies the requirements.

    2) bicameral_claim: If the answer states or implies the legislature is bicameral, extract the sentence or phrase; otherwise null.

    3) thresholds_based_on_elected_members: If the answer states that override thresholds are based on members elected (not merely present), extract the statement; otherwise null.

    4) regular_threshold_desc: The description of the override threshold for regular (non-appropriation) bills (e.g., "simple majority of the members elected to each chamber").

    5) appropriation_threshold_desc: The description of the override threshold for appropriation or budget bills (e.g., "two-thirds of the members elected to each chamber").

    6) upper_chamber:
       - name: The name of the upper chamber (e.g., "Senate").
       - members_elected_total: The total number of elected members of the upper chamber (extract as a string exactly as written; do not infer). If not provided, null.
       - appropriation_override_votes: The exact number of votes required in the upper chamber to override a gubernatorial veto on an appropriation/budget bill (extract as a string exactly as written; do not infer). If not provided, null.
       - sources: All URLs in the answer that specifically support upper chamber details (constitution sections, statutes, chamber webpages, etc.). If none, return an empty list.

    7) lower_chamber:
       - name: The name of the lower chamber (e.g., "House of Representatives" or "Assembly").
       - members_elected_total: The total number of elected members of the lower chamber (string). If not provided, null.
       - appropriation_override_votes: The exact number of votes required in the lower chamber to override a gubernatorial veto on an appropriation/budget bill (string). If not provided, null.
       - sources: All URLs in the answer that specifically support lower chamber details. If none, return an empty list.

    8) global_sources: All other URLs mentioned in the answer that support the constitutional veto override rules generally (including any constitution citations, legislative manuals, or official state sites). Return as an array of URLs. If none, return an empty list.

    IMPORTANT:
    - Extract only information explicitly present in the answer. If any field is not stated, use null or empty list as specified.
    - For URLs, include full valid URLs. If missing protocol, prepend http://.
    - Do NOT invent or compute numbers yourself; only extract what the answer provides.
    """


# ---------------------------- Helper Functions ----------------------------- #
def parse_int(value: Optional[str]) -> Optional[int]:
    """Parse an integer from a string. Returns None if not possible."""
    if value is None:
        return None
    # Find first integer in the string
    m = re.search(r"\d+", value.replace(",", ""))
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def ceil_two_thirds(n: int) -> int:
    """Compute ceil(2/3 * n)"""
    return math.ceil(2 * n / 3)


def collect_all_sources(extracted: VetoOverrideExtraction) -> List[str]:
    """Collect and deduplicate all URLs from global and chamber-specific sources."""
    urls: List[str] = []
    if extracted.global_sources:
        urls.extend(extracted.global_sources)
    if extracted.upper_chamber and extracted.upper_chamber.sources:
        urls.extend(extracted.upper_chamber.sources)
    if extracted.lower_chamber and extracted.lower_chamber.sources:
        urls.extend(extracted.lower_chamber.sources)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------- Verification Builders ------------------------- #
async def build_and_verify_requirements(
    evaluator: Evaluator,
    parent_node,
    extracted: VetoOverrideExtraction,
) -> None:
    """
    Build the 'Constitutional_Requirements_Met' parallel critical node and verify child leaves.
    """
    state = extracted.state or "the state"
    all_sources = collect_all_sources(extracted)

    req_node = evaluator.add_parallel(
        id="Constitutional_Requirements_Met",
        desc="The identified state's constitution satisfies all stated structural and threshold requirements for veto overrides.",
        parent=parent_node,
        critical=True
    )

    # Bicameral legislature
    bicameral_leaf = evaluator.add_leaf(
        id="Bicameral_Legislature",
        desc="The state has a bicameral legislature (two separate legislative chambers).",
        parent=req_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{state} has a bicameral legislature with two separate chambers.",
        node=bicameral_leaf,
        sources=all_sources,
        additional_instruction="Verify via cited sources that the state's legislature comprises two distinct chambers (e.g., Senate and House/Assembly). If sources contradict or show unicameral, mark incorrect."
    )

    # Regular bills override threshold: simple majority of members elected
    regular_leaf = evaluator.add_leaf(
        id="Regular_Bills_Override_Threshold",
        desc="For regular (non-appropriation) bills, the constitution requires a veto override by a simple majority of the members elected to each chamber.",
        parent=req_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In {state}, for regular (non-appropriation) bills, overriding a governor's veto requires a simple majority of the members elected to each chamber.",
        node=regular_leaf,
        sources=all_sources,
        additional_instruction="Confirm the constitution explicitly uses 'members elected' for regular bills override and specifies a simple majority (not three-fifths, two-thirds, or 'members present')."
    )

    # Appropriation/budget bills override threshold: two-thirds of members elected
    appropriation_leaf = evaluator.add_leaf(
        id="Appropriation_Bills_Override_Threshold",
        desc="For appropriation/budget bills, the constitution requires a veto override by two-thirds of the members elected to each chamber.",
        parent=req_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In {state}, overriding a veto on an appropriation or budget bill requires two-thirds of the members elected to each chamber.",
        node=appropriation_leaf,
        sources=all_sources,
        additional_instruction="Check cited constitution or official authority to ensure the higher threshold specifically applies to appropriation/budget bills and is defined as two-thirds of members elected."
    )

    # Thresholds based on elected members
    elected_basis_leaf = evaluator.add_leaf(
        id="Thresholds_Based_On_Elected_Members_Not_Present",
        desc="The override thresholds are explicitly based on the total number of members elected to each chamber (not merely members present).",
        parent=req_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In {state}, veto override thresholds are based on the number of members elected to each chamber, not just those present.",
        node=elected_basis_leaf,
        sources=all_sources,
        additional_instruction="Look for explicit language like 'members elected' or 'of all members elected' in the cited constitution/authority; if only 'members present' is mentioned, mark incorrect."
    )


async def build_and_verify_membership_fixed(
    evaluator: Evaluator,
    parent_node,
    extracted: VetoOverrideExtraction,
) -> None:
    """
    Build the 'Chamber_Membership_Fixed_And_Used_For_Calculation' leaf and verify when possible.
    If membership numbers are not provided, fail the leaf.
    """
    state = extracted.state or "the state"
    all_sources = collect_all_sources(extracted)

    membership_leaf = evaluator.add_leaf(
        id="Chamber_Membership_Fixed_And_Used_For_Calculation",
        desc="The answer provides (or otherwise makes explicit via cited authority) the fixed total membership of each chamber (as defined by constitution or statute) used to compute the two-thirds vote requirement.",
        parent=parent_node,
        critical=True
    )

    upper_name = (extracted.upper_chamber.name if extracted.upper_chamber else None) or "upper chamber"
    lower_name = (extracted.lower_chamber.name if extracted.lower_chamber else None) or "lower chamber"

    upper_total_str = extracted.upper_chamber.members_elected_total if extracted.upper_chamber else None
    lower_total_str = extracted.lower_chamber.members_elected_total if extracted.lower_chamber else None
    upper_total_int = parse_int(upper_total_str)
    lower_total_int = parse_int(lower_total_str)

    # If either membership count is missing or unparsable, mark failed without web verification
    if upper_total_int is None or lower_total_int is None:
        membership_leaf.score = 0.0
        membership_leaf.status = "failed"
        return

    # Verify that these are fixed totals used for computation (via sources)
    claim = (
        f"In {state}, the total number of elected members is {upper_total_int} in the {upper_name} "
        f"and {lower_total_int} in the {lower_name}, as defined by constitution or statute."
    )
    await evaluator.verify(
        claim=claim,
        node=membership_leaf,
        sources=all_sources,
        additional_instruction="Verify that cited authority provides fixed chamber membership totals (not variable attendance) and these totals are the basis for computing a two‑thirds override count."
    )


async def build_and_verify_vote_counts(
    evaluator: Evaluator,
    parent_node,
    extracted: VetoOverrideExtraction,
) -> None:
    """
    Build the 'Appropriation_Bill_Override_Vote_Counts' parallel critical node and check computation correctness
    for upper and lower chambers via custom computation nodes. This strictly checks that provided counts equal
    ceil(2/3 * total elected members).
    """
    counts_node = evaluator.add_parallel(
        id="Appropriation_Bill_Override_Vote_Counts",
        desc="Exact vote counts required in each chamber to override a veto on an appropriation bill, computed as two-thirds of total elected members (rounded up).",
        parent=parent_node,
        critical=True
    )

    # Extract and compute for upper chamber
    upper_name = (extracted.upper_chamber.name if extracted.upper_chamber else None) or "upper chamber"
    upper_total_int = parse_int(extracted.upper_chamber.members_elected_total if extracted.upper_chamber else None)
    upper_votes_provided_int = parse_int(extracted.upper_chamber.appropriation_override_votes if extracted.upper_chamber else None)

    upper_correct = (
        upper_total_int is not None and
        upper_votes_provided_int is not None and
        upper_votes_provided_int == ceil_two_thirds(upper_total_int)
    )

    evaluator.add_custom_node(
        result=upper_correct,
        id="Upper_Chamber_Vote_Count",
        desc=f"Provides the exact upper-chamber vote count for appropriation-bill veto override, correctly computed as ceil((2/3) * {upper_total_int if upper_total_int is not None else 'N/A'}) for {upper_name}.",
        parent=counts_node,
        critical=True
    )

    # Extract and compute for lower chamber
    lower_name = (extracted.lower_chamber.name if extracted.lower_chamber else None) or "lower chamber"
    lower_total_int = parse_int(extracted.lower_chamber.members_elected_total if extracted.lower_chamber else None)
    lower_votes_provided_int = parse_int(extracted.lower_chamber.appropriation_override_votes if extracted.lower_chamber else None)

    lower_correct = (
        lower_total_int is not None and
        lower_votes_provided_int is not None and
        lower_votes_provided_int == ceil_two_thirds(lower_total_int)
    )

    evaluator.add_custom_node(
        result=lower_correct,
        id="Lower_Chamber_Vote_Count",
        desc=f"Provides the exact lower-chamber vote count for appropriation-bill veto override, correctly computed as ceil((2/3) * {lower_total_int if lower_total_int is not None else 'N/A'}) for {lower_name}.",
        parent=counts_node,
        critical=True
    )

    # Record computed values for transparency
    evaluator.add_custom_info(
        info={
            "upper": {
                "name": upper_name,
                "members_elected_total_str": extracted.upper_chamber.members_elected_total if extracted.upper_chamber else None,
                "members_elected_total_int": upper_total_int,
                "provided_override_votes_str": extracted.upper_chamber.appropriation_override_votes if extracted.upper_chamber else None,
                "provided_override_votes_int": upper_votes_provided_int,
                "computed_two_thirds_ceiling": ceil_two_thirds(upper_total_int) if upper_total_int is not None else None,
                "match": upper_correct
            },
            "lower": {
                "name": lower_name,
                "members_elected_total_str": extracted.lower_chamber.members_elected_total if extracted.lower_chamber else None,
                "members_elected_total_int": lower_total_int,
                "provided_override_votes_str": extracted.lower_chamber.appropriation_override_votes if extracted.lower_chamber else None,
                "provided_override_votes_int": lower_votes_provided_int,
                "computed_two_thirds_ceiling": ceil_two_thirds(lower_total_int) if lower_total_int is not None else None,
                "match": lower_correct
            }
        },
        info_type="computed_vote_counts",
        info_name="appropriation_override_vote_counts_computation"
    )


# ------------------------------ Main Entry -------------------------------- #
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
    Evaluate an agent's answer for the Veto Override State and Vote Counts task
    using the obj_task_eval framework and the rubric tree.
    """
    # Initialize evaluator (root node is non-critical by default)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # We want sequential gating at the top level
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
    extracted: VetoOverrideExtraction = await evaluator.extract(
        prompt=prompt_extract_veto_override_info(),
        template_class=VetoOverrideExtraction,
        extraction_name="veto_override_extraction"
    )

    # Create the top-level critical sequential node to mirror the rubric root
    top_node = evaluator.add_sequential(
        id="Veto_Override_State_And_Vote_Counts",
        desc="Identify the US state matching the specified constitutional veto-override rules and provide the required appropriation-bill override vote counts for each chamber.",
        parent=root,
        critical=True
    )

    # 1) State identified (critical). Use existence check to ensure a specific state is named.
    state_present = bool(extracted.state and extracted.state.strip())
    evaluator.add_custom_node(
        result=state_present,
        id="State_Identified",
        desc="The answer names a specific U.S. state as the candidate that satisfies all stated requirements.",
        parent=top_node,
        critical=True
    )

    # 2) Constitutional requirements met (parallel, critical).
    await build_and_verify_requirements(evaluator, top_node, extracted)

    # 3) Chamber membership fixed and used for calculation (leaf, critical).
    await build_and_verify_membership_fixed(evaluator, top_node, extracted)

    # 4) Appropriation bill override vote counts (parallel, critical).
    await build_and_verify_vote_counts(evaluator, top_node, extracted)

    # Return summary
    return evaluator.get_summary()