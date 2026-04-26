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
TASK_ID = "breakfast_chain_info_24_7"
TASK_DESCRIPTION = (
    "Provide comprehensive information about the three major 24/7 breakfast restaurant chains in the United States: "
    "IHOP, Denny's, and Waffle House. Your response should include: "
    "1. Which chain has the most locations in the United States overall, "
    "2. Which chain operates in the most US states, "
    "3. How many US states does Waffle House operate in, "
    "4. Which chain has the most locations in Texas, "
    "5. Which chain has the most locations in Georgia, "
    "6. The approximate number of IHOP locations in the United States, "
    "7. The approximate number of Waffle House locations in the United States, "
    "8. Which chain (between IHOP and Denny's) has the lower franchise fee, "
    "9. The franchise fee amount for a single IHOP restaurant agreement, "
    "10. Which chain (between IHOP and Denny's) requires a higher minimum net worth for franchising, "
    "11. The minimum net worth requirement for IHOP domestic franchising, "
    "12. Confirmation that all three chains (IHOP, Denny's, and Waffle House) operate 24 hours a day, 7 days a week."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BreakfastChainsExtraction(BaseModel):
    # 1. Most US locations overall
    most_us_locations_chain: Optional[str] = None
    sources_most_us_locations_chain: List[str] = Field(default_factory=list)

    # 2. Operates in most US states
    most_states_chain: Optional[str] = None
    sources_most_states_chain: List[str] = Field(default_factory=list)

    # 3. Waffle House state count
    waffle_house_state_count: Optional[str] = None
    sources_waffle_house_state_count: List[str] = Field(default_factory=list)

    # 4. Texas most locations
    texas_most_locations_chain: Optional[str] = None
    sources_texas_most_locations_chain: List[str] = Field(default_factory=list)

    # 5. Georgia most locations
    georgia_most_locations_chain: Optional[str] = None
    sources_georgia_most_locations_chain: List[str] = Field(default_factory=list)

    # 6. IHOP approximate number of US locations
    ihop_location_count: Optional[str] = None
    sources_ihop_location_count: List[str] = Field(default_factory=list)

    # 7. Waffle House approximate number of US locations
    waffle_house_location_count: Optional[str] = None
    sources_waffle_house_location_count: List[str] = Field(default_factory=list)

    # 8. Lower franchise fee (IHOP vs Denny's)
    lower_franchise_fee_chain: Optional[str] = None
    sources_lower_franchise_fee_chain: List[str] = Field(default_factory=list)

    # 9. IHOP franchise fee amount (single restaurant agreement)
    ihop_franchise_fee_amount: Optional[str] = None
    sources_ihop_franchise_fee_amount: List[str] = Field(default_factory=list)

    # 10. Higher minimum net worth requirement (IHOP vs Denny's)
    higher_net_worth_requirement_chain: Optional[str] = None
    sources_higher_net_worth_requirement_chain: List[str] = Field(default_factory=list)

    # 11. IHOP minimum net worth requirement (domestic franchising)
    ihop_min_net_worth_requirement: Optional[str] = None
    sources_ihop_min_net_worth_requirement: List[str] = Field(default_factory=list)

    # 12. All three operate 24/7 confirmation (extract as free text like "yes"/"true"/"no")
    all_three_24_7: Optional[str] = None
    sources_all_three_24_7: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_breakfast_chain_info() -> str:
    return """
    Extract the following information as explicitly stated in the answer text, along with the specific URL sources cited for each item. Only include URLs that are clearly associated with the particular claim. If a claim is not present, set its value to null. If no source URLs are provided for a claim, return an empty array for that claim's sources.

    Fields to extract:
    1. most_us_locations_chain: Which chain (among IHOP, Denny's, and Waffle House) has the most locations in the United States overall. Use the chain name as it appears in the answer (e.g., "IHOP", "Denny's", "Waffle House").
       sources_most_us_locations_chain: URLs supporting this claim.

    2. most_states_chain: Which chain operates in the most US states (among IHOP, Denny's, and Waffle House).
       sources_most_states_chain: URLs supporting this claim.

    3. waffle_house_state_count: The number of US states where Waffle House operates (can be exact or approximate as stated).
       sources_waffle_house_state_count: URLs supporting this claim.

    4. texas_most_locations_chain: Which chain has the most locations in Texas (among IHOP, Denny's, and Waffle House).
       sources_texas_most_locations_chain: URLs supporting this claim.

    5. georgia_most_locations_chain: Which chain has the most locations in Georgia (among IHOP, Denny's, and Waffle House).
       sources_georgia_most_locations_chain: URLs supporting this claim.

    6. ihop_location_count: The approximate number of IHOP locations in the United States (as stated in the answer; can be approximate phrases like "around 1,800").
       sources_ihop_location_count: URLs supporting this claim.

    7. waffle_house_location_count: The approximate number of Waffle House locations in the United States.
       sources_waffle_house_location_count: URLs supporting this claim.

    8. lower_franchise_fee_chain: Between IHOP and Denny's, which one has the lower franchise fee.
       sources_lower_franchise_fee_chain: URLs supporting the comparison (e.g., franchise disclosure documents, official franchise pages, reputable aggregators).

    9. ihop_franchise_fee_amount: The franchise fee amount for a single IHOP restaurant agreement (as stated in the answer).
       sources_ihop_franchise_fee_amount: URLs supporting this fee amount.

    10. higher_net_worth_requirement_chain: Between IHOP and Denny's, which requires the higher minimum net worth for franchising.
        sources_higher_net_worth_requirement_chain: URLs supporting this comparison.

    11. ihop_min_net_worth_requirement: The minimum net worth requirement for IHOP domestic franchising (as stated).
        sources_ihop_min_net_worth_requirement: URLs supporting this requirement.

    12. all_three_24_7: Confirmation text indicating whether all three chains (IHOP, Denny's, and Waffle House) operate 24/7 (e.g., "yes", "no", "true", "false"). Extract exactly what the answer claims. Prefer "yes"/"no" or "true"/"false" if present.
        sources_all_three_24_7: URLs supporting this claim.

    Extraction rules:
    - Source URLs must be explicitly present in the answer (plain URLs or in markdown links). Do not invent URLs.
    - If a URL is missing a protocol, prepend http://.
    - Deduplicate source URLs for each field.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _exists_with_sources(value: Optional[str], sources: Optional[List[str]]) -> bool:
    return bool(value and str(value).strip()) and bool(sources and len(sources) > 0)


def _safe_sources(sources: Optional[List[str]]) -> List[str]:
    return sources or []


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def add_seq_verification(
    evaluator: Evaluator,
    parent,
    id_base: str,
    existence_desc: str,
    verify_desc: str,
    value: Optional[str],
    sources: Optional[List[str]],
    claim_builder,
    additional_instruction: str,
) -> None:
    """
    Create a sequential node with:
    - Critical existence-and-sources check
    - Critical verification leaf using claim constructed from `value`
    """
    seq_node = evaluator.add_sequential(
        id=id_base,
        desc=verify_desc,
        parent=parent,
        critical=False,
    )

    exists_node = evaluator.add_custom_node(
        result=_exists_with_sources(value, sources),
        id=f"{id_base}_exists",
        desc=existence_desc,
        parent=seq_node,
        critical=True,
    )

    # Verification leaf (auto-skipped if exists_node failed due to Evaluator prerequisites logic)
    verify_leaf = evaluator.add_leaf(
        id=f"{id_base}_verify",
        desc=verify_desc,
        parent=seq_node,
        critical=True,
    )

    claim_text = claim_builder(value)
    await evaluator.verify(
        claim=claim_text,
        node=verify_leaf,
        sources=_safe_sources(sources),
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Main verification builder                                                   #
# --------------------------------------------------------------------------- #
async def build_chain_verifications(evaluator: Evaluator, root, info: BreakfastChainsExtraction) -> None:
    # 1. Chain with most US locations overall
    await add_seq_verification(
        evaluator=evaluator,
        parent=root,
        id_base="most_us_locations_chain",
        existence_desc="The answer identifies which chain has the most US locations and provides sources",
        verify_desc="Identify which breakfast restaurant chain has the most locations in the United States overall",
        value=info.most_us_locations_chain,
        sources=info.sources_most_us_locations_chain,
        claim_builder=lambda v: f"Among IHOP, Denny's, and Waffle House, {v} has the most locations in the United States.",
        additional_instruction="Use the provided sources to confirm location counts or credible comparative statements. Allow reasonable rounding/approximation in counts.",
    )

    # 2. Chain operating in most US states
    await add_seq_verification(
        evaluator=evaluator,
        parent=root,
        id_base="most_states_chain",
        existence_desc="The answer identifies which chain operates in the most US states and provides sources",
        verify_desc="Identify which breakfast restaurant chain operates in the most US states",
        value=info.most_states_chain,
        sources=info.sources_most_states_chain,
        claim_builder=lambda v: f"Among IHOP, Denny's, and Waffle House, {v} operates in the most US states.",
        additional_instruction="Verify coverage by states from the sources. If sources list state counts, compare them; allow minor discrepancies if clearly approximate.",
    )

    # 3. Waffle House state count
    await add_seq_verification(
        evaluator=evaluator,
        parent=root,
        id_base="waffle_house_state_count",
        existence_desc="The answer provides Waffle House's US state count with sources",
        verify_desc="Provide the number of US states where Waffle House operates",
        value=info.waffle_house_state_count,
        sources=info.sources_waffle_house_state_count,
        claim_builder=lambda v: f"Waffle House operates in {v} US states.",
        additional_instruction="Confirm the stated number (or approximation) of US states where Waffle House operates. Accept reasonable wording like 'about', 'approximately'.",
    )

    # 4. Texas: chain with most locations
    await add_seq_verification(
        evaluator=evaluator,
        parent=root,
        id_base="texas_most_locations_chain",
        existence_desc="The answer identifies which chain has the most locations in Texas and provides sources",
        verify_desc="Identify which breakfast restaurant chain has the most locations in Texas",
        value=info.texas_most_locations_chain,
        sources=info.sources_texas_most_locations_chain,
        claim_builder=lambda v: f"In Texas, {v} has the most locations among IHOP, Denny's, and Waffle House.",
        additional_instruction="From the sources, determine counts or credible statements specific to Texas for each chain, then confirm the identified chain is highest.",
    )

    # 5. Georgia: chain with most locations
    await add_seq_verification(
        evaluator=evaluator,
        parent=root,
        id_base="georgia_most_locations_chain",
        existence_desc="The answer identifies which chain has the most locations in Georgia and provides sources",
        verify_desc="Identify which breakfast restaurant chain has the most locations in Georgia",
        value=info.georgia_most_locations_chain,
        sources=info.sources_georgia_most_locations_chain,
        claim_builder=lambda v: f"In Georgia, {v} has the most locations among IHOP, Denny's, and Waffle House.",
        additional_instruction="Use the sources to compare Georgia-specific counts or credible statements for each chain.",
    )

    # 6. IHOP approximate number of US locations
    await add_seq_verification(
        evaluator=evaluator,
        parent=root,
        id_base="ihop_location_count",
        existence_desc="The answer provides the approximate number of IHOP US locations and sources",
        verify_desc="Provide the approximate number of IHOP locations in the United States",
        value=info.ihop_location_count,
        sources=info.sources_ihop_location_count,
        claim_builder=lambda v: f"IHOP has approximately {v} locations in the United States.",
        additional_instruction="Confirm the approximate IHOP US location count from sources. Allow rounding or minor variance if clearly approximate.",
    )

    # 7. Waffle House approximate number of US locations
    await add_seq_verification(
        evaluator=evaluator,
        parent=root,
        id_base="waffle_house_location_count",
        existence_desc="The answer provides the approximate number of Waffle House US locations and sources",
        verify_desc="Provide the approximate number of Waffle House locations in the United States",
        value=info.waffle_house_location_count,
        sources=info.sources_waffle_house_location_count,
        claim_builder=lambda v: f"Waffle House has approximately {v} locations in the United States.",
        additional_instruction="Confirm the approximate Waffle House US location count from sources. Allow rounding or minor variance if clearly approximate.",
    )

    # 8. Lower franchise fee between IHOP and Denny's
    await add_seq_verification(
        evaluator=evaluator,
        parent=root,
        id_base="lower_franchise_fee_chain",
        existence_desc="The answer identifies which chain has the lower franchise fee (IHOP vs Denny's) and provides sources",
        verify_desc="Identify which chain (between IHOP and Denny's) has the lower franchise fee",
        value=info.lower_franchise_fee_chain,
        sources=info.sources_lower_franchise_fee_chain,
        claim_builder=lambda v: f"Between IHOP and Denny's, {v} has the lower franchise fee.",
        additional_instruction="From the sources, read the franchise fee amounts for IHOP and Denny's and confirm that the identified chain's fee is lower. Focus on U.S. single-unit fee figures.",
    )

    # 9. IHOP franchise fee amount (single restaurant agreement)
    await add_seq_verification(
        evaluator=evaluator,
        parent=root,
        id_base="ihop_franchise_fee_amount",
        existence_desc="The answer provides IHOP's franchise fee amount (single restaurant agreement) with sources",
        verify_desc="Provide the franchise fee amount for IHOP (single restaurant agreement)",
        value=info.ihop_franchise_fee_amount,
        sources=info.sources_ihop_franchise_fee_amount,
        claim_builder=lambda v: f"The franchise fee for a single IHOP restaurant agreement is {v}.",
        additional_instruction="Verify IHOP's single-unit franchise fee from official or reputable franchise sources. Accept reasonable currency formatting and ranges if explicitly stated.",
    )

    # 10. Higher minimum net worth requirement (IHOP vs Denny's)
    await add_seq_verification(
        evaluator=evaluator,
        parent=root,
        id_base="higher_net_worth_requirement_chain",
        existence_desc="The answer identifies which chain requires the higher minimum net worth for franchising (IHOP vs Denny's) and provides sources",
        verify_desc="Identify which chain (between IHOP and Denny's) requires the higher minimum net worth for franchising",
        value=info.higher_net_worth_requirement_chain,
        sources=info.sources_higher_net_worth_requirement_chain,
        claim_builder=lambda v: f"Between IHOP and Denny's, {v} requires the higher minimum net worth for franchising.",
        additional_instruction="From the sources, find stated minimum net worth requirements for IHOP and Denny's franchise candidates and confirm the comparison.",
    )

    # 11. IHOP minimum net worth requirement (domestic franchising)
    await add_seq_verification(
        evaluator=evaluator,
        parent=root,
        id_base="ihop_min_net_worth_requirement",
        existence_desc="The answer provides IHOP's minimum net worth requirement for domestic franchising with sources",
        verify_desc="Provide the minimum net worth requirement for IHOP domestic franchising",
        value=info.ihop_min_net_worth_requirement,
        sources=info.sources_ihop_min_net_worth_requirement,
        claim_builder=lambda v: f"The minimum net worth requirement for IHOP domestic franchising is {v}.",
        additional_instruction="Confirm IHOP's minimum net worth requirement (U.S. domestic franchising) from official or reputable franchise references.",
    )

    # 12. All three operate 24/7 confirmation
    await add_seq_verification(
        evaluator=evaluator,
        parent=root,
        id_base="all_three_24_7",
        existence_desc="The answer confirms whether all three chains operate 24/7 and provides sources",
        verify_desc="Confirm whether IHOP, Denny's, and Waffle House all operate 24 hours a day, 7 days a week",
        value=info.all_three_24_7,
        sources=info.sources_all_three_24_7,
        claim_builder=lambda v: "IHOP, Denny's, and Waffle House all operate 24 hours a day, 7 days a week.",
        additional_instruction="Check the sources to confirm whether each chain is described or promoted as operating 24/7. If sources indicate variability by location (not universally 24/7), then the claim that 'all operate 24/7' should be considered not supported.",
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
    Evaluate an answer for the breakfast chains information task using the Mind2Web2 framework.
    """
    # Initialize evaluator with a parallel root as the rubric's overall aggregation
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
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_breakfast_chain_info(),
        template_class=BreakfastChainsExtraction,
        extraction_name="breakfast_chain_info",
    )

    # Build verification tree and execute checks
    await build_chain_verifications(evaluator, root, extracted_info)

    # Return standardized evaluation summary
    return evaluator.get_summary()