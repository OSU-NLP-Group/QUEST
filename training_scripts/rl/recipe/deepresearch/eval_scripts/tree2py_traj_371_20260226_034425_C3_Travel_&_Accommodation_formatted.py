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
TASK_ID = "delta_gold_medallion_baggage_atl_puj"
TASK_DESCRIPTION = (
    "What is the total checked baggage allowance for a Delta Gold Medallion member traveling in Main Cabin from "
    "Atlanta (ATL) to Punta Cana (PUJ), including: (1) the number of free checked bags, (2) the maximum weight per bag, "
    "(3) any applicable fees for additional bags, and (4) whether a standard acoustic guitar can be checked as baggage "
    "and under what conditions? Provide reference URLs to support your answer."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class BaggageExtraction(BaseModel):
    # Numbers/limits extracted exactly as stated in the answer (prefer strings for flexibility)
    free_checked_bags_count: Optional[str] = None
    max_weight_per_bag_lbs: Optional[str] = None
    max_size_per_bag_linear_in: Optional[str] = None
    second_bag_fee_us_to_caribbean: Optional[str] = None

    # Musical instrument/guitar policy details extracted from the answer
    guitar_can_be_checked: Optional[str] = None  # "yes" / "no" or similar
    guitar_conditions: Optional[str] = None      # e.g., "within 150 linear inches and 100 lbs; hard case"

    # All source URLs the answer cited
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helper                                                    #
# --------------------------------------------------------------------------- #
def prompt_extract_baggage() -> str:
    return """
    Extract the following items exactly as stated in the answer text. If an item is not clearly stated, return null.
    - free_checked_bags_count: The number or phrase indicating how many free checked bags a Delta Gold Medallion member
      gets when traveling in Main Cabin on the ATL→PUJ route (e.g., "1", "one free checked bag").
    - max_weight_per_bag_lbs: The maximum allowable weight per checked bag for this scenario, including unit if present
      (e.g., "70 lbs").
    - max_size_per_bag_linear_in: The standard maximum checked-baggage size limit in linear inches (L+W+H),
      including unit if present (e.g., "62 linear inches").
    - second_bag_fee_us_to_caribbean: The fee for the second checked bag for U.S.→Caribbean itineraries such as ATL→PUJ,
      including currency symbol if present (e.g., "$45").
    - guitar_can_be_checked: Whether a standard acoustic guitar can be checked as baggage ("yes" or "no" if clearly
      implied; otherwise return null).
    - guitar_conditions: Any conditions the answer states for checking a guitar (e.g., size/weight limits like
      "within 150 linear inches and 100 lbs", or case requirements). Provide a concise phrase; if not mentioned, return null.
    - reference_urls: An array of all URLs the answer cites as sources. Extract actual URLs even if presented in markdown.
    """


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify(
    evaluator: Evaluator,
    extracted: BaggageExtraction,
    root_node
) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """
    # Main rubric node (critical, parallel aggregation)
    complete_node = evaluator.add_parallel(
        id="Complete_Baggage_Allowance_Answer",
        desc="Answer provides the total checked baggage allowance details for a Delta Gold Medallion member traveling in Main Cabin from ATL to PUJ, covering the requested aspects and providing supporting reference URL(s).",
        parent=root_node,
        critical=True
    )

    # 1) Reference URLs check (critical; gates all other verifications implicitly as a critical sibling)
    refs_ok = bool(extracted.reference_urls)
    evaluator.add_custom_node(
        result=refs_ok,
        id="Reference_URLs_Provided",
        desc="Answer provides reference URL(s) supporting the baggage policy information it states.",
        parent=complete_node,
        critical=True
    )

    # 2) Scope matches trip (critical; simple verification against the answer text)
    scope_node = evaluator.add_leaf(
        id="Scope_Matches_Trip",
        desc="Answer explicitly addresses the stated scenario: Delta Gold Medallion member traveling in Main Cabin from Atlanta (ATL) to Punta Cana (PUJ).",
        parent=complete_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly addresses a Delta Gold Medallion member traveling in Main Cabin from Atlanta (ATL) to Punta Cana (PUJ).",
        node=scope_node,
        additional_instruction=(
            "Check the answer text to confirm that it explicitly references (1) Delta Gold Medallion status, "
            "(2) Main Cabin, and (3) the route from Atlanta (ATL) to Punta Cana (PUJ) or equivalent phrasing."
        )
    )

    # Helper to add a verification leaf or a direct failure if the answer omitted the value
    async def verify_with_value_or_fail(
        node_id: str,
        node_desc: str,
        claim_text_builder,
        sources: List[str],
        addl_instruction: str
    ):
        """
        If claim_text_builder returns None (i.e., the answer didn't state the required value),
        mark this criterion as failed directly. Otherwise, create a leaf and verify with sources.
        """
        claim_text = claim_text_builder()
        if claim_text is None:
            evaluator.add_custom_node(
                result=False,
                id=node_id,
                desc=node_desc,
                parent=complete_node,
                critical=True
            )
            return
        leaf = evaluator.add_leaf(
            id=node_id,
            desc=node_desc,
            parent=complete_node,
            critical=True
        )
        await evaluator.verify(
            claim=claim_text,
            node=leaf,
            sources=sources,
            additional_instruction=addl_instruction
        )

    # 3) Free checked bags count (critical, must be supported by sources)
    await verify_with_value_or_fail(
        node_id="Free_Checked_Bags_Count",
        node_desc="Answer states the number of free checked bags for a Gold Medallion member (per constraints: one free checked bag).",
        claim_text_builder=lambda: (
            None if not extracted.free_checked_bags_count else
            f"A Delta Gold Medallion member traveling in Main Cabin receives {extracted.free_checked_bags_count} free checked bag(s) for an itinerary like ATL to PUJ (U.S. to Caribbean)."
        ),
        sources=extracted.reference_urls,
        addl_instruction=(
            "Confirm using Delta's baggage allowance or Medallion benefits pages that a Gold Medallion member traveling in Main Cabin "
            "receives the stated number of free checked bags. The itinerary is Atlanta (ATL) to Punta Cana (PUJ), which is a U.S. to Caribbean routing."
        )
    )

    # 4) Maximum weight per bag (critical, must be supported by sources)
    await verify_with_value_or_fail(
        node_id="Maximum_Weight_Per_Bag",
        node_desc="Answer states the maximum allowable weight per checked bag for a Gold Medallion member (per constraints: up to 70 lbs per bag).",
        claim_text_builder=lambda: (
            None if not extracted.max_weight_per_bag_lbs else
            f"The maximum allowable weight per checked bag for a Delta Gold Medallion member traveling in Main Cabin is {extracted.max_weight_per_bag_lbs}."
        ),
        sources=extracted.reference_urls,
        addl_instruction=(
            "Verify the weight limit per bag for Gold Medallion members in Main Cabin per Delta's policy. "
            "Many pages quote a standard 50 lbs, but Medallion benefits can increase the limit; confirm the exact figure stated in the answer."
        )
    )

    # 5) Maximum size per bag (critical, must be supported by sources)
    await verify_with_value_or_fail(
        node_id="Maximum_Size_Per_Bag",
        node_desc="Answer states the standard maximum checked-baggage size limit (per constraints: must not exceed 62 linear inches, L+W+H).",
        claim_text_builder=lambda: (
            None if not extracted.max_size_per_bag_linear_in else
            f"The standard maximum checked-baggage size limit on Delta is {extracted.max_size_per_bag_linear_in} (linear inches: L+W+H)."
        ),
        sources=extracted.reference_urls,
        addl_instruction=(
            "Check Delta's standard checked baggage size policy (linear inches: L+W+H) and confirm the value stated in the answer."
        )
    )

    # 6) Fees for additional bags (critical, must be supported by sources)
    await verify_with_value_or_fail(
        node_id="Fees_For_Additional_Bags",
        node_desc="Answer provides applicable fees for additional checked bags beyond the free allowance, including the second checked bag fee for US-to-Caribbean flights (per constraints: $45 for the second bag).",
        claim_text_builder=lambda: (
            None if not extracted.second_bag_fee_us_to_caribbean else
            f"For a U.S.-to-Caribbean itinerary such as ATL to PUJ, the fee for the second checked bag on Delta is {extracted.second_bag_fee_us_to_caribbean}."
        ),
        sources=extracted.reference_urls,
        addl_instruction=(
            "Confirm the second checked bag fee for U.S. to Caribbean routes (e.g., Atlanta to Punta Cana) on Delta's official baggage fees pages "
            "or calculators. Focus on the second bag fee as stated in the answer."
        )
    )

    # 7) Guitar checked as baggage and conditions (critical; split into sub-criteria for clarity)
    guitar_main = evaluator.add_parallel(
        id="Guitar_Checked_As_Baggage_And_Conditions",
        desc="Answer states whether a standard acoustic guitar can be checked as baggage and under what conditions, consistent with Delta's policy.",
        parent=complete_node,
        critical=True
    )

    # 7a) Whether a guitar can be checked (existence + correctness)
    if not extracted.guitar_can_be_checked:
        evaluator.add_custom_node(
            result=False,
            id="Guitar_Can_Be_Checked",
            desc="Answer explicitly states whether a standard acoustic guitar can be checked as baggage.",
            parent=guitar_main,
            critical=True
        )
    else:
        can_checked_statement = extracted.guitar_can_be_checked.strip().lower()
        # Formulate claim according to the answer's stance
        if can_checked_statement in ("yes", "y", "true"):
            claim_guitar_accept = "Delta accepts guitars (musical instruments) as checked baggage, subject to instrument policy limits."
        elif can_checked_statement in ("no", "n", "false"):
            claim_guitar_accept = "Delta does not accept guitars as checked baggage."
        else:
            # If ambiguous, still try to verify acceptance claim as the answer implies
            claim_guitar_accept = f"The answer claims that checking a standard acoustic guitar as baggage on Delta is '{extracted.guitar_can_be_checked}'. Verify whether this is correct per Delta policy."

        guitar_accept_node = evaluator.add_leaf(
            id="Guitar_Can_Be_Checked",
            desc="Answer explicitly states whether a standard acoustic guitar can be checked as baggage.",
            parent=guitar_main,
            critical=True
        )
        await evaluator.verify(
            claim=claim_guitar_accept,
            node=guitar_accept_node,
            sources=extracted.reference_urls,
            additional_instruction=(
                "Use Delta's musical instrument policy page(s) to judge whether guitars can be accepted as checked baggage."
            )
        )

    # 7b) Guitar conditions (existence + correctness of the stated conditions vs sources)
    if not extracted.guitar_conditions:
        evaluator.add_custom_node(
            result=False,
            id="Guitar_Conditions_Supported",
            desc="Answer states the conditions under which a standard acoustic guitar can be checked (e.g., size/weight limits).",
            parent=guitar_main,
            critical=True
        )
    else:
        # Verify that the conditions are supported by sources (e.g., up to 150 linear inches and 100 lbs)
        guitar_cond_node = evaluator.add_leaf(
            id="Guitar_Conditions_Supported",
            desc="Answer states the conditions under which a standard acoustic guitar can be checked (e.g., size/weight limits).",
            parent=guitar_main,
            critical=True
        )
        await evaluator.verify(
            claim=f"Delta's policy supports these conditions for checking a guitar: {extracted.guitar_conditions}.",
            node=guitar_cond_node,
            sources=extracted.reference_urls,
            additional_instruction=(
                "Check Delta's pages for musical instruments as checked baggage. Common limits include up to 150 total linear inches and up to 100 lbs. "
                "Verify the conditions as stated in the answer are supported by the provided sources."
            )
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
    Evaluate an answer for the Delta Gold Medallion baggage allowance task.
    """
    # Initialize evaluator and root
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_baggage(),
        template_class=BaggageExtraction,
        extraction_name="baggage_extraction"
    )

    # Build verification tree and run checks
    await build_and_verify(evaluator, extracted, root)

    # Return the final structured summary
    return evaluator.get_summary()