import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "bilt_home_purchase_program"
TASK_DESCRIPTION = """
I'm considering purchasing a home in the United States and I've heard about Bilt Rewards' partnership with eXp Realty that allows homebuyers to earn rewards points on their home purchase. I need comprehensive information about this program to understand if it would be beneficial for my situation.

Please provide the following details about Bilt Rewards' home purchase program:

1. What is the earning rate for Bilt Points when purchasing a home through this program?
2. What are the specific requirements regarding the real estate agent for earning these rewards?
3. Are there any limitations on how many times I can earn rewards through home purchases?
4. What is the geographic coverage of eXp Realty across the United States?
5. Are there any restrictions on the type of property that qualifies for earning Bilt Points?
6. What are the redemption options available for Bilt Points once earned?

For each piece of information, please include reference URLs from official Bilt Rewards sources or reputable financial news sources.
"""

# Ground-truth expectations encoded in the rubric
GROUND_TRUTH = {
    "earning_rate": "1 point per $2 of the total home purchase/closing price",
    "agent_requirement": "You must be connected to and work with an eXp Realty agent through the Bilt platform (not merely any eXp agent)",
    "purchase_limitations": "Only one redemption allowed per home purchase",
    "exp_coverage": "eXp Realty operates across all 50 U.S. states",
    "property_type_eligibility": "No restrictions on qualifying property type",
    "redemption_travel_11": "Points can be transferred 1:1 to airline/hotel travel partners",
    "redemption_dpa": "Points can be redeemed toward a future home down payment",
    "redemption_rent_credit": "Points can be redeemed as rent credits",
    "redemption_gift_cards": "Points can be redeemed for gift cards",
}


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class BiltHomePurchaseExtraction(BaseModel):
    # Core program details
    earning_rate: Optional[str] = None
    earning_rate_urls: List[str] = Field(default_factory=list)

    agent_requirement: Optional[str] = None
    agent_requirement_urls: List[str] = Field(default_factory=list)

    purchase_limitations: Optional[str] = None
    purchase_limitations_urls: List[str] = Field(default_factory=list)

    exp_coverage: Optional[str] = None
    exp_coverage_urls: List[str] = Field(default_factory=list)

    property_type_eligibility: Optional[str] = None
    property_type_eligibility_urls: List[str] = Field(default_factory=list)

    # Redemption options
    redemption_travel: Optional[str] = None  # e.g., “transfer to airline/hotel partners at 1:1”
    redemption_down_payment: Optional[str] = None
    redemption_rent_credit: Optional[str] = None
    redemption_gift_cards: Optional[str] = None
    redemption_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_bilt_home_purchase_info() -> str:
    return """
    Extract the requested details about the Bilt Rewards + eXp Realty home purchase program from the answer text. 
    Return the exact phrasing as it appears in the answer for each field. If a field is not mentioned, return null for that field.
    Also extract the reference URL(s) explicitly provided in the answer for each field (if any). Only extract URLs that are explicitly present in the answer.

    Extract the following fields:
    - earning_rate: text describing the earning rate for Bilt Points on a qualifying home purchase.
    - earning_rate_urls: array of URLs specifically cited for the earning rate.

    - agent_requirement: text describing any agent/connection requirement for earning the rewards (e.g., must be connected through Bilt to an eXp agent).
    - agent_requirement_urls: array of URLs specifically cited for the agent requirement.

    - purchase_limitations: text describing limitations on earning/redeeming via a home purchase (e.g., only one redemption per home purchase).
    - purchase_limitations_urls: array of URLs specifically cited for the home purchase limitation.

    - exp_coverage: text describing the geographic coverage of eXp Realty in the U.S. (e.g., operates in all 50 states).
    - exp_coverage_urls: array of URLs specifically cited for eXp Realty’s coverage.

    - property_type_eligibility: text describing whether there are restrictions on the property types that qualify.
    - property_type_eligibility_urls: array of URLs specifically cited for property type eligibility.

    - redemption_travel: text describing travel-related redemption via airline/hotel partners (include any mention of a transfer ratio like 1:1 if stated).
    - redemption_down_payment: text describing down payment assistance as a redemption option.
    - redemption_rent_credit: text describing rent credit as a redemption option.
    - redemption_gift_cards: text describing gift cards as a redemption option.
    - redemption_urls: array of URLs specifically cited for redemption options (any/all of the redemption items above).

    SPECIAL RULES FOR URL EXTRACTION:
    - Extract only complete, valid URLs that are explicitly present in the answer (plain URLs or markdown links).
    - Do not fabricate or infer URLs. If a URL is missing a protocol, prepend http://.
    - If no URL is provided for a given item, return an empty array for that item’s URL list.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _present(text: Optional[str]) -> bool:
    return bool(text and isinstance(text, str) and text.strip() != "")


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and len(urls) > 0)


async def _verify_sequential_fact(
    evaluator: Evaluator,
    parent_node,
    base_id: str,
    existence_desc: str,
    claim_desc: str,
    urls: List[str],
    existence_condition: bool,
    additional_instruction: str = "None",
    critical: bool = True,
):
    """
    Build a sequential verification subtree:
    1) Existence check (custom node)
    2) Evidence-backed verification of the claim (leaf node with URL verification)
    """
    seq_node = evaluator.add_sequential(
        id=base_id,
        desc=claim_desc,
        parent=parent_node,
        critical=critical
    )

    # 1) Existence
    evaluator.add_custom_node(
        result=existence_condition,
        id=f"{base_id}_provided",
        desc=existence_desc,
        parent=seq_node,
        critical=True
    )

    # 2) Supported by cited sources
    verify_node = evaluator.add_leaf(
        id=f"{base_id}_supported",
        desc=claim_desc,
        parent=seq_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_desc,
        node=verify_node,
        sources=urls if urls else None,
        additional_instruction=additional_instruction
    )


# --------------------------------------------------------------------------- #
# Main evaluation logic                                                       #
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
    Evaluate an answer for the Bilt Rewards + eXp Realty home purchase program information task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # overall aggregation (we'll add a critical wrapper)
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

    # Critical wrapper node (reflecting rubric root)
    rubric_root = evaluator.add_parallel(
        id="Bilt_Home_Purchase_Program_Information",
        desc="Accurate identification of key details about Bilt Rewards' home purchase program with eXp Realty, including required citations.",
        parent=root,
        critical=True
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_bilt_home_purchase_info(),
        template_class=BiltHomePurchaseExtraction,
        extraction_name="bilt_home_purchase_extraction"
    )

    # Record rubric-based ground truth expectations (reference only)
    evaluator.add_ground_truth(
        {
            "expected": GROUND_TRUTH
        },
        gt_type="rubric_expectations"
    )

    # ------------------------------------------------------------------- #
    # 1) Reward_Earning_Rate                                              #
    # ------------------------------------------------------------------- #
    await _verify_sequential_fact(
        evaluator=evaluator,
        parent_node=rubric_root,
        base_id="Reward_Earning_Rate",
        existence_desc="Earning rate is stated with at least one supporting reference URL",
        claim_desc="The program awards 1 Bilt Point per $2 of the total home purchase (closing) price for qualifying purchases made through the Bilt + eXp Realty program.",
        urls=extracted.earning_rate_urls,
        existence_condition=_present(extracted.earning_rate) and _has_urls(extracted.earning_rate_urls),
        additional_instruction="Allow equivalent phrasings such as '1 point for every $2 spent' or '0.5x per dollar'. If a cap (e.g., up to a maximum number of points) is mentioned, it does not contradict the 1:2 earning rate."
    )

    # ------------------------------------------------------------------- #
    # 2) Required_Agent_Partnership                                       #
    # ------------------------------------------------------------------- #
    await _verify_sequential_fact(
        evaluator=evaluator,
        parent_node=rubric_root,
        base_id="Required_Agent_Partnership",
        existence_desc="Agent requirement is stated with at least one supporting reference URL",
        claim_desc="To earn Bilt Points on a home purchase, the buyer must work with an eXp Realty agent who is connected to the member via the Bilt platform; using an eXp agent outside the Bilt connection does not qualify.",
        urls=extracted.agent_requirement_urls,
        existence_condition=_present(extracted.agent_requirement) and _has_urls(extracted.agent_requirement_urls),
        additional_instruction="Look for requirements that the agent engagement be initiated or connected through Bilt (e.g., via the Bilt app/website) rather than any eXp agent relationship established independently."
    )

    # ------------------------------------------------------------------- #
    # 3) Home_Purchase_Limitations                                        #
    # ------------------------------------------------------------------- #
    await _verify_sequential_fact(
        evaluator=evaluator,
        parent_node=rubric_root,
        base_id="Home_Purchase_Limitations",
        existence_desc="Home purchase limitation is stated with at least one supporting reference URL",
        claim_desc="Only one redemption related to earning or applying Bilt Points is allowed per home purchase transaction.",
        urls=extracted.purchase_limitations_urls,
        existence_condition=_present(extracted.purchase_limitations) and _has_urls(extracted.purchase_limitations_urls),
        additional_instruction="Minor wording variations are acceptable as long as they clearly limit the redemption/award to one per home purchase."
    )

    # ------------------------------------------------------------------- #
    # 4) eXp_Realty_Coverage                                              #
    # ------------------------------------------------------------------- #
    await _verify_sequential_fact(
        evaluator=evaluator,
        parent_node=rubric_root,
        base_id="eXp_Realty_Coverage",
        existence_desc="eXp Realty geographic coverage is stated with at least one supporting reference URL",
        claim_desc="eXp Realty operates across all 50 U.S. states.",
        urls=extracted.exp_coverage_urls,
        existence_condition=_present(extracted.exp_coverage) and _has_urls(extracted.exp_coverage_urls),
        additional_instruction="If the source mentions operation in all 50 states (with or without additional regions like DC), consider that as supporting evidence."
    )

    # ------------------------------------------------------------------- #
    # 5) Property_Type_Eligibility                                        #
    # ------------------------------------------------------------------- #
    await _verify_sequential_fact(
        evaluator=evaluator,
        parent_node=rubric_root,
        base_id="Property_Type_Eligibility",
        existence_desc="Property type eligibility is stated with at least one supporting reference URL",
        claim_desc="There are no restrictions on qualifying property type for earning Bilt Points in the Bilt + eXp Realty home purchase program.",
        urls=extracted.property_type_eligibility_urls,
        existence_condition=_present(extracted.property_type_eligibility) and _has_urls(extracted.property_type_eligibility_urls),
        additional_instruction="Accept variations such as 'all home property types qualify' or explicit lists implying no restriction (e.g., single-family, condo, townhouse, etc.)."
    )

    # ------------------------------------------------------------------- #
    # 6) Point_Redemption_Options (parallel)                              #
    # ------------------------------------------------------------------- #
    redemption_main = evaluator.add_parallel(
        id="Point_Redemption_Options",
        desc="Describes the available redemption options for Bilt Points consistent with the constraints.",
        parent=rubric_root,
        critical=True
    )

    # 6a) Travel_Redemption_via_Partners (mentions 1:1 transfers)
    await _verify_sequential_fact(
        evaluator=evaluator,
        parent_node=redemption_main,
        base_id="Travel_Redemption_via_Partners",
        existence_desc="Travel redemption via airline/hotel partners is stated (including transfer ratio if claimed) with at least one supporting reference URL",
        claim_desc="Bilt Points can be transferred to airline and/or hotel partners at a 1:1 ratio.",
        urls=extracted.redemption_urls,
        existence_condition=_present(extracted.redemption_travel) and _has_urls(extracted.redemption_urls),
        additional_instruction="Look for phrasing like 'transfer partners', 'airline/hotel partners', and '1:1 transfers'. Minor variants are acceptable if substantively equivalent."
    )

    # 6b) Down_Payment_Assistance
    await _verify_sequential_fact(
        evaluator=evaluator,
        parent_node=redemption_main,
        base_id="Down_Payment_Assistance",
        existence_desc="Down payment assistance is stated as a redemption option with at least one supporting reference URL",
        claim_desc="Bilt Points can be redeemed toward a future home down payment.",
        urls=extracted.redemption_urls,
        existence_condition=_present(extracted.redemption_down_payment) and _has_urls(extracted.redemption_urls),
        additional_instruction="Wording like 'use points toward a down payment' or 'down payment assistance' should count as support."
    )

    # 6c) Rent_Credit
    await _verify_sequential_fact(
        evaluator=evaluator,
        parent_node=redemption_main,
        base_id="Rent_Credit",
        existence_desc="Rent credit is stated as a redemption option with at least one supporting reference URL",
        claim_desc="Bilt Points can be redeemed for rent credits.",
        urls=extracted.redemption_urls,
        existence_condition=_present(extracted.redemption_rent_credit) and _has_urls(extracted.redemption_urls),
        additional_instruction="Accept phrasing like 'redeem for rent' or 'apply points toward rent'."
    )

    # 6d) Gift_Cards
    await _verify_sequential_fact(
        evaluator=evaluator,
        parent_node=redemption_main,
        base_id="Gift_Cards",
        existence_desc="Gift cards are stated as a redemption option with at least one supporting reference URL",
        claim_desc="Bilt Points can be redeemed for gift cards.",
        urls=extracted.redemption_urls,
        existence_condition=_present(extracted.redemption_gift_cards) and _has_urls(extracted.redemption_urls),
        additional_instruction="Any clear mention that gift cards are an available redemption option suffices."
    )

    # ------------------------------------------------------------------- #
    # 7) Reference_URLs (parallel) — check presence of URLs per section   #
    #     Note: The factual support is already verified above with URLs.  #
    #     Here we only enforce that at least one URL was provided.        #
    # ------------------------------------------------------------------- #
    refs_main = evaluator.add_parallel(
        id="Reference_URLs",
        desc="Provides supporting reference URL(s) from official Bilt sources or reputable financial news sources for each requested detail.",
        parent=rubric_root,
        critical=True
    )

    # Helper to add existence-only URL checks as custom nodes
    def _add_ref_presence_node(node_id: str, desc: str, urls: List[str]):
        evaluator.add_custom_node(
            result=_has_urls(urls),
            id=node_id,
            desc=desc,
            parent=refs_main,
            critical=True
        )

    _add_ref_presence_node(
        "Reference_URL_for_Reward_Earning_Rate",
        "Provides at least one supporting reference URL for the earning rate detail.",
        extracted.earning_rate_urls
    )
    _add_ref_presence_node(
        "Reference_URL_for_Required_Agent_Partnership",
        "Provides at least one supporting reference URL for the agent requirement detail.",
        extracted.agent_requirement_urls
    )
    _add_ref_presence_node(
        "Reference_URL_for_Home_Purchase_Limitations",
        "Provides at least one supporting reference URL for the home purchase limitation detail.",
        extracted.purchase_limitations_urls
    )
    _add_ref_presence_node(
        "Reference_URL_for_eXp_Realty_Coverage",
        "Provides at least one supporting reference URL for eXp Realty geographic coverage detail.",
        extracted.exp_coverage_urls
    )
    _add_ref_presence_node(
        "Reference_URL_for_Property_Type_Eligibility",
        "Provides at least one supporting reference URL for property type eligibility detail.",
        extracted.property_type_eligibility_urls
    )
    _add_ref_presence_node(
        "Reference_URL_for_Point_Redemption_Options",
        "Provides at least one supporting reference URL for redemption options detail.",
        extracted.redemption_urls
    )

    # Return compiled evaluation summary
    return evaluator.get_summary()