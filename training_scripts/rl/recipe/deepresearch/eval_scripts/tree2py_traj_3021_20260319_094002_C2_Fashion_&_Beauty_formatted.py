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
TASK_ID = "wnba_beauty_product_launch_jan2025"
TASK_DESCRIPTION = (
    "A forward who was selected 2nd overall in the 2024 WNBA Draft became a brand ambassador for a beauty company. "
    "This player graduated from a Pac-12 Conference university in December 2024 with a major in Communications. "
    "The beauty company has been an official sponsor of the player's WNBA team for two consecutive seasons (2023 and 2024), "
    "and the player signed a multi-year brand ambassador agreement with them, announced in October 2024. "
    "In January 2025, the beauty company launched a collaborative liquid illuminator product with this player at an event in New York City. "
    "What is the name of this product, and on what specific date did the launch event take place?"
)

EXPECTED_LAUNCH_DATE = "January 14, 2025"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TaskExtraction(BaseModel):
    # Core entities
    player_name: Optional[str] = None
    team_name: Optional[str] = None
    brand_name: Optional[str] = None

    # Player facts
    draft_overall_pick: Optional[str] = None
    draft_year: Optional[str] = None
    position: Optional[str] = None
    graduation_university: Optional[str] = None
    graduation_date: Optional[str] = None
    major: Optional[str] = None
    conference: Optional[str] = None

    # Product facts
    product_name: Optional[str] = None
    launch_event_date: Optional[str] = None
    launch_event_city: Optional[str] = None

    # Source URLs (explicitly cited in the answer)
    sources_draft: List[str] = Field(default_factory=list)
    sources_position: List[str] = Field(default_factory=list)
    sources_graduation: List[str] = Field(default_factory=list)
    sources_brand_sponsor_2023: List[str] = Field(default_factory=list)
    sources_brand_sponsor_2024: List[str] = Field(default_factory=list)
    sources_brand_ambassador: List[str] = Field(default_factory=list)
    sources_product: List[str] = Field(default_factory=list)
    sources_launch_event: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_task_fields() -> str:
    return """
Extract the following information exactly as stated in the provided answer. Do not infer or invent any information. If an item is not present in the answer, set it to null (or an empty array for URL lists).

Required fields:
1) player_name: The full name of the player referenced.
2) team_name: The player's WNBA team referenced.
3) brand_name: The beauty company/brand referenced.

Player details:
4) draft_overall_pick: The overall pick number for the 2024 WNBA Draft (e.g., "2", "2nd", or "No. 2").
5) draft_year: The draft year (should be "2024" if present).
6) position: The player's position (e.g., "forward", "forward/center", "F").
7) graduation_university: The name of the player's university from which they graduated.
8) graduation_date: The graduation date or month/year if given (e.g., "December 2024").
9) major: The player's major (e.g., "Communications").
10) conference: The athletic conference of the university if stated (e.g., "Pac-12 Conference").

Product details:
11) product_name: The official name of the collaborative liquid illuminator/luminizer product mentioned.
12) launch_event_date: The specific launch event date (e.g., "January 14, 2025").
13) launch_event_city: The city of the launch event (e.g., "New York City"), if stated.

Source URLs:
- sources_draft: All URLs cited that support the draft details (pick number and year).
- sources_position: All URLs cited that support the player's position as a forward.
- sources_graduation: All URLs cited that support graduation details (university, date, major, and Pac-12).
- sources_brand_sponsor_2023: All URLs cited that support the brand being an official sponsor/partner of the team in 2023.
- sources_brand_sponsor_2024: All URLs cited that support the brand being an official sponsor/partner of the team in 2024.
- sources_brand_ambassador: All URLs cited that support the multi-year brand ambassador agreement announced in October 2024.
- sources_product: All URLs cited that support the collaborative liquid illuminator product name with the player.
- sources_launch_event: All URLs cited that support the launch event date (and location if included).

Rules for URLs:
- Extract only URLs explicitly present in the answer (including markdown links).
- Do not fabricate URLs. If no URL is provided for a category, return an empty array for that category.

Return a single JSON object matching the TaskExtraction schema.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_player_identification(evaluator: Evaluator, parent_node, data: TaskExtraction) -> None:
    """
    Build and run checks for player identification and academic details.
    """
    node = evaluator.add_parallel(
        id="Player_Identification",
        desc="Correctly identifies the player who was drafted 2nd overall in 2024 as a forward and graduated from a Pac-12 university in Dec 2024 with a Communications major",
        parent=parent_node,
        critical=True
    )

    # Existence of player name (gate)
    evaluator.add_custom_node(
        result=bool(data.player_name and data.player_name.strip()),
        id="player_name_present",
        desc="Player name is explicitly provided",
        parent=node,
        critical=True
    )

    # Draft details: 2nd overall in 2024 WNBA Draft
    draft_leaf = evaluator.add_leaf(
        id="drafted_2nd_overall_2024",
        desc="Player was selected 2nd overall in the 2024 WNBA Draft",
        parent=node,
        critical=True
    )
    claim_draft = f"{data.player_name} was selected 2nd overall in the 2024 WNBA Draft."
    await evaluator.verify(
        claim=claim_draft,
        node=draft_leaf,
        sources=data.sources_draft,
        additional_instruction="Accept variants like 'No. 2 pick', 'second overall', or '2nd overall'. Ensure the year is 2024."
    )

    # Position: forward (allow F, F/C, forward/center etc.)
    pos_leaf = evaluator.add_leaf(
        id="position_is_forward",
        desc="Player's position is forward",
        parent=node,
        critical=True
    )
    claim_pos = f"{data.player_name} plays as a forward."
    await evaluator.verify(
        claim=claim_pos,
        node=pos_leaf,
        sources=data.sources_position,
        additional_instruction="Allow reasonable variants like 'F', 'forward/center', or 'F/C' to count as 'forward'."
    )

    # Graduation details (Dec 2024, Communications, from a Pac-12 university)
    grad_leaf = evaluator.add_leaf(
        id="graduated_dec2024_communications",
        desc="Player graduated in December 2024 with a major in Communications from the stated university",
        parent=node,
        critical=True
    )
    uni = data.graduation_university or "the referenced university"
    claim_grad = f"{data.player_name} graduated in December 2024 from {uni} with a major in Communications."
    await evaluator.verify(
        claim=claim_grad,
        node=grad_leaf,
        sources=data.sources_graduation,
        additional_instruction="Minor phrasing differences are acceptable as long as December 2024 and a Communications major are clearly supported."
    )

    # University is in Pac-12 Conference
    pac12_leaf = evaluator.add_leaf(
        id="university_is_pac12",
        desc="The referenced university is a Pac-12 Conference member",
        parent=node,
        critical=True
    )
    claim_pac12 = f"{uni} is in the Pac-12 Conference."
    await evaluator.verify(
        claim=claim_pac12,
        node=pac12_leaf,
        sources=data.sources_graduation,
        additional_instruction="Allow variants like 'Pac-12', 'Pac 12', or 'Pacific-12'."
    )


async def verify_brand_identification(evaluator: Evaluator, parent_node, data: TaskExtraction) -> None:
    """
    Build and run checks for brand identification and sponsorship/ambassador details.
    """
    node = evaluator.add_parallel(
        id="Beauty_Brand_Identification",
        desc="Identifies the beauty brand sponsoring the team in 2023 and 2024 and the player's multi-year ambassador deal announced Oct 2024",
        parent=parent_node,
        critical=True
    )

    # Existence of brand name (gate)
    evaluator.add_custom_node(
        result=bool(data.brand_name and data.brand_name.strip()),
        id="brand_name_present",
        desc="Beauty brand name is explicitly provided",
        parent=node,
        critical=True
    )

    # Sponsor in 2023
    sponsor_2023_leaf = evaluator.add_leaf(
        id="brand_sponsor_2023",
        desc="Brand was an official sponsor/partner of the team in 2023",
        parent=node,
        critical=True
    )
    team = data.team_name or "the player's WNBA team"
    claim_sponsor_2023 = f"{data.brand_name} was an official sponsor or official partner of {team} in the 2023 season."
    await evaluator.verify(
        claim=claim_sponsor_2023,
        node=sponsor_2023_leaf,
        sources=data.sources_brand_sponsor_2023,
        additional_instruction="Accept synonymous phrasings like 'official sponsor', 'official partner', or 'official beauty partner'."
    )

    # Sponsor in 2024
    sponsor_2024_leaf = evaluator.add_leaf(
        id="brand_sponsor_2024",
        desc="Brand was an official sponsor/partner of the team in 2024",
        parent=node,
        critical=True
    )
    claim_sponsor_2024 = f"{data.brand_name} was an official sponsor or official partner of {team} in the 2024 season."
    await evaluator.verify(
        claim=claim_sponsor_2024,
        node=sponsor_2024_leaf,
        sources=data.sources_brand_sponsor_2024,
        additional_instruction="Accept synonymous phrasings like 'official sponsor', 'official partner', or 'official beauty partner'."
    )

    # Multi-year brand ambassador announced in October 2024
    ambassador_leaf = evaluator.add_leaf(
        id="brand_ambassador_oct2024_multiyear",
        desc="Player signed a multi-year brand ambassador agreement announced in October 2024",
        parent=node,
        critical=True
    )
    claim_ambassador = f"In October 2024, {data.player_name} signed a multi-year brand ambassador agreement with {data.brand_name}."
    await evaluator.verify(
        claim=claim_ambassador,
        node=ambassador_leaf,
        sources=data.sources_brand_ambassador,
        additional_instruction="Focus on both 'multi-year' and 'October 2024' being clearly indicated in the provided source(s)."
    )


async def verify_product_information(evaluator: Evaluator, parent_node, data: TaskExtraction) -> None:
    """
    Build and run checks for product name and event date.
    """
    node = evaluator.add_parallel(
        id="Product_Information",
        desc="Correctly provides both the product name and the launch event date",
        parent=parent_node,
        critical=True
    )

    # Product name presence (gate)
    evaluator.add_custom_node(
        result=bool(data.product_name and data.product_name.strip() and data.sources_product),
        id="product_name_present",
        desc="Product name is provided with at least one cited source",
        parent=node,
        critical=True
    )

    # Product name correctness (URL-grounded)
    product_name_leaf = evaluator.add_leaf(
        id="Product_Name",
        desc="Correctly provides the name of the collaborative liquid illuminator/luminizer product",
        parent=node,
        critical=True
    )
    claim_product = f"The collaborative liquid illuminator (or luminizer) product with {data.player_name} and {data.brand_name} is named '{data.product_name}'."
    await evaluator.verify(
        claim=claim_product,
        node=product_name_leaf,
        sources=data.sources_product,
        additional_instruction="Ensure the source explicitly names the collaborative product; focus on the specific product name (not a bundle). Allow 'illuminator' vs 'luminizer' variants."
    )

    # Launch event date presence (gate)
    evaluator.add_custom_node(
        result=bool(data.launch_event_date and data.launch_event_date.strip() and data.sources_launch_event),
        id="launch_event_date_present",
        desc="Launch event date is provided with at least one cited source",
        parent=node,
        critical=True
    )

    # Launch event date correctness (URL-grounded; must be January 14, 2025)
    launch_date_leaf = evaluator.add_leaf(
        id="Launch_Event_Date",
        desc="Correctly states that the launch event took place on January 14, 2025",
        parent=node,
        critical=True
    )
    claim_launch_date = "The launch event took place on January 14, 2025."
    await evaluator.verify(
        claim=claim_launch_date,
        node=launch_date_leaf,
        sources=data.sources_launch_event,
        additional_instruction="Accept reasonable date formatting variants like 'Jan. 14, 2025' or '1/14/2025'. The claim must clearly match this date."
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
    Evaluate an answer for the WNBA beauty product launch task and return a structured summary.
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_task_fields(),
        template_class=TaskExtraction,
        extraction_name="key_facts_extraction",
    )

    # Ground truth info (for transparency; not used directly for scoring)
    evaluator.add_ground_truth(
        {
            "expected_launch_event_date": EXPECTED_LAUNCH_DATE,
            "notes": "Product name varies by player/brand; must be supported by provided sources.",
        },
        gt_type="expected_values",
    )

    # Build the top-level critical node reflecting the rubric's 'Complete_Answer'
    complete_node = evaluator.add_parallel(
        id="Complete_Answer",
        desc="The answer correctly identifies the player, beauty brand, product name, and launch event date based on all specified constraints",
        parent=root,
        critical=True,
    )

    # Sub-verifications
    await verify_player_identification(evaluator, complete_node, extracted)
    await verify_brand_identification(evaluator, complete_node, extracted)
    await verify_product_information(evaluator, complete_node, extracted)

    # Return evaluation summary
    return evaluator.get_summary()