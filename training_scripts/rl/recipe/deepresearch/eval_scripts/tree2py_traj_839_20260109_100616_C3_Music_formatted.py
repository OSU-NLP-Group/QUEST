import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "purchase_concert_hall_capacity"
TASK_DESCRIPTION = (
    "What is the total seating capacity of the main concert hall at the performing arts center of the college in "
    "Westchester County, New York, where the 2023 Grammy Award winner for Best New Artist studied jazz?"
)

# Ground truth expectations (for info/reference; verification still follows rubric tree)
EXPECTED_WINNER = "Samara Joy"
EXPECTED_COLLEGE_CANONICAL = "Purchase College, State University of New York"
EXPECTED_COLLEGE_ALIASES = [
    "SUNY Purchase College",
    "SUNY Purchase",
    "Purchase College (SUNY)",
    "Purchase College"
]
EXPECTED_PAC_NAME = "The Performing Arts Center, Purchase College"
EXPECTED_MAIN_HALL = "Concert Hall"
EXPECTED_CAPACITY = "1,372"  # The textual canonical form (without enforcing separators)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CapacityEntityExtraction(BaseModel):
    # Core entities mentioned in the answer
    winner_name: Optional[str] = None
    studied_field: Optional[str] = None
    college_name: Optional[str] = None
    college_location: Optional[str] = None
    pac_name: Optional[str] = None
    main_hall_name: Optional[str] = None
    capacity: Optional[str] = None

    # Per-claim/source URLs (from the answer text)
    winner_sources: List[str] = Field(default_factory=list)
    college_sources: List[str] = Field(default_factory=list)
    pac_sources: List[str] = Field(default_factory=list)
    hall_sources: List[str] = Field(default_factory=list)
    capacity_sources: List[str] = Field(default_factory=list)

    # General or miscellaneous sources, if provided
    general_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_core() -> str:
    return """
    Extract from the answer the specific entities, attributes, and any URLs provided. Only extract information explicitly present in the answer text.

    Fields to extract:
    1) winner_name: the person identified as the 2023 Grammy Award winner for Best New Artist.
    2) studied_field: the field of study (e.g., "jazz") associated with where the winner studied.
    3) college_name: the college where the winner studied jazz (e.g., "Purchase College, State University of New York" or "SUNY Purchase").
    4) college_location: any location text associated with the college (e.g., "Westchester County, New York").
    5) pac_name: the name of the performing arts center at that college (e.g., "The Performing Arts Center, Purchase College").
    6) main_hall_name: the name of the main concert hall at that performing arts center (e.g., "Concert Hall").
    7) capacity: the total seating capacity number or phrase for that main concert hall (e.g., "1,372 seats"). Keep the text exactly as stated in the answer.

    Additionally, extract any URLs explicitly present in the answer and map them to the most relevant fields:
    - winner_sources: URLs supporting the identification of the Best New Artist winner or related.
    - college_sources: URLs supporting where the winner studied (the college) and/or its location.
    - pac_sources: URLs specifically about the performing arts center at the college.
    - hall_sources: URLs specifically about the main concert hall at that performing arts center.
    - capacity_sources: URLs specifically stating or supporting the capacity of the main concert hall.
    - general_sources: any other URLs cited that are relevant but not clearly mapped above.

    SPECIAL RULES FOR URL EXTRACTION:
    - Extract only URLs explicitly present in the answer text (including markdown links).
    - Do not fabricate URLs.
    - Include full URLs with protocol.

    If any field is not mentioned in the answer, set it to null (for strings) or an empty list (for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def first_non_empty_list(*lists: List[str]) -> List[str]:
    """Return the first non-empty list among the provided lists; otherwise empty list."""
    for lst in lists:
        if lst and len(lst) > 0:
            # Also deduplicate while preserving order
            seen = set()
            unique = []
            for x in lst:
                if x and x not in seen:
                    unique.append(x)
                    seen.add(x)
            return unique
    return []


def safe_str(val: Optional[str]) -> str:
    return val if isinstance(val, str) and val.strip() != "" else ""


# --------------------------------------------------------------------------- #
# Verification logic construction                                             #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extracted: CapacityEntityExtraction
) -> None:
    """
    Build and execute the verification tree according to the provided rubric.
    """

    # Root critical sequential node (Complete Task)
    complete_task_node = evaluator.add_sequential(
        id="Complete_Task",
        desc=(
            "Determine the total seating capacity of the main concert hall at the performing arts center of the "
            "Westchester County, NY college where the 2023 Best New Artist Grammy winner studied jazz, satisfying all "
            "provided constraints."
        ),
        parent=evaluator.root,
        critical=True
    )

    # 1) Verify Target Entity Constraints (parallel, critical)
    verify_entities_node = evaluator.add_parallel(
        id="Verify_Target_Entity_Constraints",
        desc="Verify the referenced artist/college/venue entities match the constraint-defined targets.",
        parent=complete_task_node,
        critical=True
    )

    # 1.a) Winner_Is_Samara_Joy
    winner_leaf = evaluator.add_leaf(
        id="Winner_Is_Samara_Joy",
        desc="The 2023 Grammy Award winner for Best New Artist is identified as Samara Joy.",
        parent=verify_entities_node,
        critical=True
    )
    winner_claim = f"The extracted winner '{safe_str(extracted.winner_name)}' and 'Samara Joy' refer to the same person."
    await evaluator.verify(
        claim=winner_claim,
        node=winner_leaf,
        additional_instruction="Judge whether the two names refer to the same person. Allow minor variations and case-insensitivity."
    )

    # 1.b) Studied_Jazz_At_SUNY_Purchase
    studied_purchase_leaf = evaluator.add_leaf(
        id="Studied_Jazz_At_SUNY_Purchase",
        desc="The winner is stated/confirmed to have studied jazz at SUNY Purchase College (Purchase College, State University of New York).",
        parent=verify_entities_node,
        critical=True
    )
    studied_purchase_claim = (
        "The 2023 Best New Artist winner (Samara Joy) studied jazz at Purchase College, State University of New York "
        "(also known as SUNY Purchase)."
    )
    studied_purchase_sources = first_non_empty_list(
        extracted.college_sources, extracted.winner_sources, extracted.general_sources
    )
    await evaluator.verify(
        claim=studied_purchase_claim,
        node=studied_purchase_leaf,
        sources=studied_purchase_sources,
        additional_instruction=(
            "Verify that the provided source(s) explicitly support that Samara Joy studied jazz at Purchase College "
            "(SUNY Purchase). Accept synonyms like 'Purchase College', 'SUNY Purchase College', or 'Purchase College, "
            "State University of New York'."
        )
    )

    # 1.c) College_In_Westchester_County_NY
    college_in_westchester_leaf = evaluator.add_leaf(
        id="College_In_Westchester_County_NY",
        desc="The college is stated/confirmed to be located in Westchester County, New York.",
        parent=verify_entities_node,
        critical=True
    )
    college_westchester_claim = (
        "Purchase College, State University of New York (SUNY Purchase) is located in Westchester County, New York."
    )
    college_westchester_sources = first_non_empty_list(
        extracted.college_sources, extracted.pac_sources, extracted.general_sources
    )
    await evaluator.verify(
        claim=college_westchester_claim,
        node=college_in_westchester_leaf,
        sources=college_westchester_sources,
        additional_instruction="Verify the location claim is explicitly supported by the provided source(s)."
    )

    # 1.d) Performing_Arts_Center_Is_PAC_Purchase
    pac_is_correct_leaf = evaluator.add_leaf(
        id="Performing_Arts_Center_Is_PAC_Purchase",
        desc="The performing arts center is identified as The Performing Arts Center, Purchase College.",
        parent=verify_entities_node,
        critical=True
    )
    pac_name_claim = (
        "The performing arts center at Purchase College is named 'The Performing Arts Center, Purchase College' "
        "(sometimes written as 'The Performing Arts Center at Purchase College')."
    )
    pac_sources = first_non_empty_list(
        extracted.pac_sources, extracted.college_sources, extracted.general_sources
    )
    await evaluator.verify(
        claim=pac_name_claim,
        node=pac_is_correct_leaf,
        sources=pac_sources,
        additional_instruction="Treat minor naming variations as equivalent (e.g., 'at Purchase College')."
    )

    # 1.e) Main_Concert_Hall_Is_Concert_Hall
    main_hall_is_correct_leaf = evaluator.add_leaf(
        id="Main_Concert_Hall_Is_Concert_Hall",
        desc="The main concert hall is identified as the Concert Hall at The Performing Arts Center, Purchase College.",
        parent=verify_entities_node,
        critical=True
    )
    main_hall_claim = (
        "The main concert hall at The Performing Arts Center, Purchase College is called 'Concert Hall' "
        "(also referred to as 'The Concert Hall')."
    )
    hall_sources = first_non_empty_list(
        extracted.hall_sources, extracted.pac_sources, extracted.general_sources
    )
    await evaluator.verify(
        claim=main_hall_claim,
        node=main_hall_is_correct_leaf,
        sources=hall_sources,
        additional_instruction="Treat 'Concert Hall' and 'The Concert Hall' as equivalent."
    )

    # 2) Report Correct Seating Capacity (parallel, critical)
    report_capacity_node = evaluator.add_parallel(
        id="Report_Correct_Seating_Capacity",
        desc="State the total seating capacity for that main concert hall.",
        parent=complete_task_node,
        critical=True
    )

    # 2.a) Capacity_Equals_1372_Seats
    capacity_leaf = evaluator.add_leaf(
        id="Capacity_Equals_1372_Seats",
        desc="The total seating capacity is given as 1,372 seats.",
        parent=report_capacity_node,
        critical=True
    )

    # Build the claim from the extracted capacity so we are judging the agent-stated figure against sources.
    extracted_capacity_text = safe_str(extracted.capacity)
    capacity_claim = (
        f"The total seating capacity of the Concert Hall at The Performing Arts Center, Purchase College is "
        f"{extracted_capacity_text if extracted_capacity_text else 'UNKNOWN'}."
    )

    capacity_sources = first_non_empty_list(
        extracted.capacity_sources, extracted.hall_sources, extracted.pac_sources, extracted.general_sources
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=capacity_sources,
        additional_instruction=(
            "Verify whether the provided source(s) explicitly state the total seating capacity. "
            "Treat '1372' and '1,372' as equivalent numeric values and allow typical formatting (e.g., '1,372 seats'). "
            "If the claim's number does not match what the source states (expected 1,372), mark as not supported."
        )
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Purchase College Concert Hall capacity task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # The internal root is a wrapper; we add our own critical sequential node
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
        prompt=prompt_extract_core(),
        template_class=CapacityEntityExtraction,
        extraction_name="core_entities_and_sources",
    )

    # Add ground truth info for reference in final summary
    evaluator.add_ground_truth({
        "expected_winner": EXPECTED_WINNER,
        "expected_college_canonical": EXPECTED_COLLEGE_CANONICAL,
        "expected_college_aliases": EXPECTED_COLLEGE_ALIASES,
        "expected_pac_name": EXPECTED_PAC_NAME,
        "expected_main_hall": EXPECTED_MAIN_HALL,
        "expected_capacity": EXPECTED_CAPACITY
    })

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()