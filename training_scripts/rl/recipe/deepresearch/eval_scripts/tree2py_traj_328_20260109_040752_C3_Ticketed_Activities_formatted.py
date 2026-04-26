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
TASK_ID = "broadway_theater_constraints"
TASK_DESCRIPTION = (
    "Identify the Broadway theater that satisfies all of the following conditions: "
    "(1) it opened in the 1970s, "
    "(2) it was originally named after a real estate developer or development company, "
    "(3) it was later renamed during a Tony Awards ceremony to honor a pair of creative collaborators, "
    "(4) it currently houses a hall of fame dedicated to theater, and "
    "(5) it has been continuously hosting the same musical production since 2003. "
    "For this theater, provide: (a) its current official name, (b) the exact year it opened, "
    "(c) its original name, (d) the specific year when it was renamed, and (e) the name of the musical "
    "that has been running there since 2003."
)

# Ground truth info for reference (used only for summary/debug; not for verification)
GROUND_TRUTH = {
    "expected_current_official_name": "Gershwin Theatre",
    "expected_opening_year": "1972",
    "expected_original_name": "Uris Theater",
    "expected_renaming_year": "1983",
    "expected_musical_name_since_2003": "Wicked",
    "notes": {
        "broadway_theater_requirement": "Broadway theater (≥500 seats) in NYC Theater District.",
        "original_name_developer_requirement": "Original name honored Uris Buildings Corporation (real estate developers).",
        "renamed_during_tonys_requirement": "Renamed during the Tony Awards ceremony.",
        "renaming_honors_pair_requirement": "Renamed to honor George and Ira Gershwin (pair of creative collaborators).",
        "hall_of_fame_requirement": "Houses the American Theater Hall of Fame.",
        "same_musical_since_2003_requirement": "Wicked has run continuously since 2003 at the Gershwin Theatre."
    }
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TheaterDetails(BaseModel):
    """Information about the identified theater, extracted from the answer."""
    current_official_name: Optional[str] = None
    opening_year: Optional[str] = None
    original_name: Optional[str] = None
    renaming_year: Optional[str] = None
    renaming_honorees: List[str] = Field(default_factory=list)
    musical_name_since_2003: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_theater_details() -> str:
    return (
        "Extract the single Broadway theater identified in the answer that is claimed to satisfy ALL of the following conditions:\n"
        "1) Opened in the 1970s; 2) Originally named after a real estate developer or development company; "
        "3) Later renamed during a Tony Awards ceremony to honor a pair of creative collaborators; "
        "4) Currently houses a hall of fame dedicated to theater; 5) Has continuously hosted the same musical production since 2003.\n\n"
        "Return a JSON object with the following fields:\n"
        "- current_official_name: The theater's current official name.\n"
        "- opening_year: The exact year the theater opened (as a string). If the answer provides a range or uncertain value, extract what is stated verbatim.\n"
        "- original_name: The theater's original name.\n"
        "- renaming_year: The specific year when the theater was renamed (as a string).\n"
        "- renaming_honorees: An array of the names of the creative collaborators honored by the renaming (e.g., two names).\n"
        "- musical_name_since_2003: The name of the musical that has been running there continuously since 2003.\n"
        "- sources: An array of URL(s) that the answer explicitly cites for this theater or any of the above facts. Include all URLs mentioned in the answer "
        "that are relevant to this theater or its details. Extract actual URLs only; do not invent any URLs.\n\n"
        "Important rules:\n"
        "- If multiple theaters are discussed, pick the one that best matches the constraints and is presented as the correct answer. If unclear, pick the first mentioned.\n"
        "- If any field is not mentioned, set it to null (or empty array for renaming_honorees/sources).\n"
        "- Extract only what the answer states; do not infer or add information."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_str(val: Optional[str]) -> bool:
    return bool(val) and bool(val.strip())


def _get_sources(info: TheaterDetails) -> List[str]:
    # Normalize sources: ensure list of valid-looking strings; leave as-is per framework rules
    return info.sources or []


def _safe_name(info: TheaterDetails) -> str:
    return info.current_official_name.strip() if _nonempty_str(info.current_official_name) else "the theater identified in the answer"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_theater_identification_checks(evaluator: Evaluator, parent_node, info: TheaterDetails) -> None:
    """
    Build and execute verification checks for the identification constraints.
    All children under this node are critical as per rubric.
    """
    ident_node = evaluator.add_parallel(
        id="theater_identification",
        desc="Identifies a Broadway theater that satisfies all stated conditions",
        parent=parent_node,
        critical=True
    )

    name_for_claim = _safe_name(info)
    sources = _get_sources(info)

    # 1) Broadway theater requirement (≥500 seats)
    node_bt = evaluator.add_leaf(
        id="broadway_theater_requirement",
        desc="The identified theater is a Broadway theater (minimum 500 seats per given constraint)",
        parent=ident_node,
        critical=True
    )
    claim_bt = (
        f"{name_for_claim} is a Broadway theater in New York City and has a seating capacity of at least 500."
    )
    await evaluator.verify(
        claim=claim_bt,
        node=node_bt,
        sources=sources,
        additional_instruction="Verify the theatre is classified as a Broadway theatre (not Off-Broadway) and has at least 500 seats."
    )

    # 2) Opened in the 1970s (1970–1979)
    node_70s = evaluator.add_leaf(
        id="opened_in_1970s_requirement",
        desc="The theater opened in the 1970s (1970–1979 inclusive)",
        parent=ident_node,
        critical=True
    )
    if _nonempty_str(info.opening_year):
        claim_70s = f"{name_for_claim} opened in {info.opening_year}, which falls within the 1970s."
    else:
        claim_70s = f"{name_for_claim} opened in the 1970s."
    await evaluator.verify(
        claim=claim_70s,
        node=node_70s,
        sources=sources,
        additional_instruction="Check the opening year on the cited source(s) and confirm it is between 1970 and 1979 inclusive."
    )

    # 3) Original name based on real estate developer/company
    node_dev = evaluator.add_leaf(
        id="original_name_developer_requirement",
        desc="The theater's original name was based on a real estate developer or development company",
        parent=ident_node,
        critical=True
    )
    if _nonempty_str(info.original_name):
        claim_dev = (
            f"The theater's original name was '{info.original_name}', and that name was based on a real estate developer or development company."
        )
    else:
        claim_dev = (
            "The theater's original name was based on a real estate developer or development company."
        )
    await evaluator.verify(
        claim=claim_dev,
        node=node_dev,
        sources=sources,
        additional_instruction="Confirm the original namesake was a real estate developer or development firm (e.g., Uris Buildings Corporation)."
    )

    # 4) Renamed during a Tony Awards ceremony
    node_tonys = evaluator.add_leaf(
        id="renamed_during_tonys_requirement",
        desc="The theater was renamed during a Tony Awards ceremony",
        parent=ident_node,
        critical=True
    )
    if _nonempty_str(info.renaming_year):
        claim_tonys = f"{name_for_claim} was renamed during the Tony Awards ceremony in {info.renaming_year}."
    else:
        claim_tonys = f"{name_for_claim} was renamed during a Tony Awards ceremony."
    await evaluator.verify(
        claim=claim_tonys,
        node=node_tonys,
        sources=sources,
        additional_instruction="Look for language indicating the renaming occurred as part of the Tony Awards ceremony."
    )

    # 5) Renaming honored a pair of creative collaborators
    node_pair = evaluator.add_leaf(
        id="renaming_honors_pair_requirement",
        desc="The renaming honored a pair of creative collaborators (not a single person)",
        parent=ident_node,
        critical=True
    )
    if info.renaming_honorees and len(info.renaming_honorees) >= 2:
        honorees_str = ", ".join(info.renaming_honorees)
        claim_pair = f"The renaming honored a pair of creative collaborators: {honorees_str}."
    else:
        claim_pair = "The renaming honored a pair of creative collaborators (two individuals)."
    await evaluator.verify(
        claim=claim_pair,
        node=node_pair,
        sources=sources,
        additional_instruction="Verify that the renaming recognized two collaborators together (e.g., siblings or a duo), not just one person."
    )

    # 6) Theater currently houses a hall of fame dedicated to theater
    node_hof = evaluator.add_leaf(
        id="hall_of_fame_requirement",
        desc="The theater currently houses a hall of fame dedicated to theater",
        parent=ident_node,
        critical=True
    )
    claim_hof = f"{name_for_claim} currently houses a hall of fame dedicated to theater."
    await evaluator.verify(
        claim=claim_hof,
        node=node_hof,
        sources=sources,
        additional_instruction="Confirm that the theater houses the American Theater Hall of Fame or equivalent hall of fame."
    )

    # 7) Same musical since 2003 continuously
    node_musical = evaluator.add_leaf(
        id="same_musical_since_2003_requirement",
        desc="The theater has been continuously hosting the same musical production since 2003",
        parent=ident_node,
        critical=True
    )
    if _nonempty_str(info.musical_name_since_2003):
        claim_musical = (
            f"Since 2003, {name_for_claim} has continuously hosted the musical '{info.musical_name_since_2003}'."
        )
    else:
        claim_musical = f"Since 2003, {name_for_claim} has continuously hosted the same musical production."
    await evaluator.verify(
        claim=claim_musical,
        node=node_musical,
        sources=sources,
        additional_instruction="Verify that the musical noted has been running continuously at the theater since 2003."
    )


async def build_required_info_checks(evaluator: Evaluator, parent_node, info: TheaterDetails) -> None:
    """
    Build existence checks for all required facts.
    All nodes here are critical, as per rubric.
    """
    req_node = evaluator.add_parallel(
        id="required_information_provided",
        desc="Provides all requested facts for the identified theater",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_nonempty_str(info.current_official_name),
        id="current_official_name_provided",
        desc="Provides the theater's current official name",
        parent=req_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_nonempty_str(info.opening_year),
        id="opening_year_provided",
        desc="Provides the exact year the theater opened",
        parent=req_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_nonempty_str(info.original_name),
        id="original_name_provided",
        desc="Provides the theater's original name",
        parent=req_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_nonempty_str(info.renaming_year),
        id="renaming_year_provided",
        desc="Provides the specific year when the theater was renamed",
        parent=req_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_nonempty_str(info.musical_name_since_2003),
        id="musical_name_provided",
        desc="Provides the name of the musical that has been running there since 2003",
        parent=req_node,
        critical=True
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
    Evaluate an answer for the Broadway theater constraints task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Identification first, then required info
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

    # Extract theater details from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_theater_details(),
        template_class=TheaterDetails,
        extraction_name="theater_details"
    )

    # Add ground truth info for debugging/reference
    evaluator.add_ground_truth(GROUND_TRUTH, gt_type="expected_reference")

    # Build identification verification checks (critical)
    await build_theater_identification_checks(evaluator, root, extracted_info)

    # Build provided information existence checks (critical)
    await build_required_info_checks(evaluator, root, extracted_info)

    # Return structured evaluation summary
    return evaluator.get_summary()