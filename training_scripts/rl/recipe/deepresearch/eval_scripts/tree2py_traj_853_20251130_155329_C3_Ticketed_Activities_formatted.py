import asyncio
import logging
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "broadway_lottery_digital_rules"
TASK_DESCRIPTION = (
    "You are interested in attending a Broadway musical and want to use the digital lottery system to save money on tickets. "
    "Your task is to:\n\n"
    "1. Identify ONE currently-running Broadway musical that offers digital lottery tickets priced at $60 or less.\n\n"
    "2. For Broadway digital lotteries in general, determine the following mandatory requirements:\n"
    "- What is the minimum age requirement to enter a Broadway digital lottery?\n"
    "- What form of identification is required to pick up lottery tickets at the box office?\n"
    "- How long do lottery winners typically have to claim and purchase their tickets after being notified?\n"
    "- Are digital lottery tickets transferable to another person?\n\n"
    "Provide your answer with specific details and include reference URLs to support each requirement."
)

CURRENT_DATE = "2026-01-11"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RequirementEntry(BaseModel):
    """Generic requirement entry with value text and supporting URLs extracted from the answer."""
    value: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ShowInfo(BaseModel):
    """Information about the identified Broadway musical and its digital lottery details."""
    name: Optional[str] = None
    lottery_price: Optional[str] = None  # Keep as string to allow flexible formats (e.g., '$59', 'up to $60', '$40–$60')
    urls: List[str] = Field(default_factory=list)  # URLs supporting that the show is running, has a digital lottery, and the price


class LotteryExtraction(BaseModel):
    """Complete extraction structure for the Broadway digital lottery task."""
    show: Optional[ShowInfo] = None
    age_requirement: RequirementEntry = Field(default_factory=RequirementEntry)
    photo_id_requirement: RequirementEntry = Field(default_factory=RequirementEntry)
    name_match_requirement: RequirementEntry = Field(default_factory=RequirementEntry)
    claim_time_requirement: RequirementEntry = Field(default_factory=RequirementEntry)
    transfer_policy: RequirementEntry = Field(default_factory=RequirementEntry)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_lottery_requirements() -> str:
    return (
        "Extract the following structured information from the provided answer:\n\n"
        "A) One currently-running Broadway musical with a qualifying digital lottery:\n"
        "   - show.name: The name of ONE Broadway musical explicitly mentioned in the answer. If multiple are listed, choose the FIRST.\n"
        "   - show.lottery_price: The stated digital lottery ticket price for that show (keep formatting as in the answer; e.g., '$59', 'up to $60', '$40–$60').\n"
        "   - show.urls: 1–3 URLs that the answer explicitly provides to support that the show is currently running, offers a digital lottery, and the price (include only URLs actually present in the answer).\n\n"
        "B) General Broadway digital lottery requirements, each with supporting URLs:\n"
        "   - age_requirement.value: The stated minimum age requirement to enter digital lotteries (e.g., '18+').\n"
        "   - age_requirement.urls: 1–3 URLs in the answer that support this age requirement.\n"
        "   - photo_id_requirement.value: The stated form of identification required to pick up lottery tickets (e.g., 'valid photo ID', 'government-issued photo ID').\n"
        "   - photo_id_requirement.urls: 1–3 URLs in the answer that support this requirement.\n"
        "   - name_match_requirement.value: The stated rule that the name on the lottery account/entry must match the name on the photo ID.\n"
        "   - name_match_requirement.urls: 1–3 URLs in the answer that support this rule.\n"
        "   - claim_time_requirement.value: The stated typical time window winners have to claim/purchase tickets after notification (e.g., '1 hour', '2 hours', 'up to 5 hours').\n"
        "   - claim_time_requirement.urls: 1–3 URLs in the answer that support this timing.\n"
        "   - transfer_policy.value: The stated rule about transferability of digital lottery tickets (e.g., 'non-transferable').\n"
        "   - transfer_policy.urls: 1–3 URLs in the answer that support this policy.\n\n"
        "Extraction rules:\n"
        "1. Only extract data explicitly present in the answer. Do not invent or infer.\n"
        "2. If any field is missing in the answer, set it to null. If URLs are not provided for an item, return an empty list.\n"
        "3. For URLs, extract actual URLs (including those inside markdown links). If the answer lists more than 3 URLs for an item, include the first 3.\n"
        "4. For the show, always select the first explicitly mentioned show if multiple shows are present.\n"
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_show(
    evaluator: Evaluator,
    parent_node,
    show: Optional[ShowInfo],
) -> None:
    """
    Verify the identified show satisfies all required conditions:
    - It is a Broadway musical (not a play).
    - It is currently running as of CURRENT_DATE.
    - It offers a digital lottery.
    - Its digital lottery ticket price is $60 or less.
    All checks require supporting URLs (existence gated).
    """
    # Container for the show verification with sequential gating
    show_node = evaluator.add_sequential(
        id="identify_qualifying_show",
        desc="Provide ONE currently-running Broadway musical that offers a digital lottery ticket price of $60 or less, with supporting reference URLs.",
        parent=parent_node,
        critical=True,  # Critical under the main requirements
    )

    # Existence & URLs provided
    has_show = bool(show and show.name and show.name.strip())
    has_urls = bool(show and show.urls and len(show.urls) > 0)
    evaluator.add_custom_node(
        result=(has_show and has_urls),
        id="show_provided_with_urls",
        desc="A show is identified and at least one supporting URL is provided.",
        parent=show_node,
        critical=True
    )

    # Verify it is a Broadway musical
    musical_leaf = evaluator.add_leaf(
        id="show_is_broadway_musical",
        desc="The identified show is a Broadway musical.",
        parent=show_node,
        critical=True
    )
    show_name = show.name if show else ""
    await evaluator.verify(
        claim=f"'{show_name}' is a Broadway musical (not a play).",
        node=musical_leaf,
        sources=(show.urls if show else []),
        additional_instruction=(
            "Check the provided URLs to confirm the show is categorized as a musical on Broadway. "
            "Accept 'musical', 'musical revival', or similar. If the pages indicate it's a play or Off-Broadway, mark incorrect."
        )
    )

    # Verify it is currently running
    running_leaf = evaluator.add_leaf(
        id="show_currently_running",
        desc=f"The show is currently running on Broadway as of {CURRENT_DATE}.",
        parent=show_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{show_name}' is currently running on Broadway as of {CURRENT_DATE}.",
        node=running_leaf,
        sources=(show.urls if show else []),
        additional_instruction=(
            f"Use the provided URLs to determine current status around {CURRENT_DATE}. "
            "If the pages clearly indicate the show is closed, not started yet, or otherwise not currently running, mark incorrect. "
            "If status is ambiguous, mark incorrect."
        )
    )

    # Verify it offers a digital lottery
    lottery_leaf = evaluator.add_leaf(
        id="show_offers_digital_lottery",
        desc="The show offers a digital lottery for tickets.",
        parent=show_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{show_name}' offers a digital lottery for tickets.",
        node=lottery_leaf,
        sources=(show.urls if show else []),
        additional_instruction=(
            "Look for mentions of 'digital lottery', 'lottery', 'Broadway Direct lottery', 'Lucky Seat', or similar on the provided URLs."
        )
    )

    # Verify the price is $60 or less
    price_leaf = evaluator.add_leaf(
        id="show_lottery_price_60_or_less",
        desc="The show's digital lottery ticket price is $60 or less.",
        parent=show_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The digital lottery ticket price for '{show_name}' is $60 or less.",
        node=price_leaf,
        sources=(show.urls if show else []),
        additional_instruction=(
            "Find the stated lottery ticket price on the provided URLs. "
            "Accept phrasing like 'up to $60', '$59', '$40–$60', or similar, excluding extra fees unless explicitly stated to exceed $60."
        )
    )


async def verify_requirement_with_urls(
    evaluator: Evaluator,
    parent_node,
    req: RequirementEntry,
    node_id: str,
    node_desc: str,
    claim_text: str,
    add_instruction: str,
) -> None:
    """
    Generic pattern:
    - Create a sequential node for the requirement (critical).
    - Gate with existence of at least one URL.
    - Verify the claim against the provided URLs.
    """
    req_node = evaluator.add_sequential(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=True
    )

    # Existence of supporting URLs is mandatory
    urls_exist_leaf = evaluator.add_custom_node(
        result=(bool(req.urls) and len(req.urls) > 0),
        id=f"{node_id}_urls_provided",
        desc=f"At least one supporting URL is provided for {node_id}.",
        parent=req_node,
        critical=True
    )

    # Verify the requirement using the URLs
    verify_leaf = evaluator.add_leaf(
        id=f"{node_id}_supported",
        desc=node_desc,
        parent=req_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_text,
        node=verify_leaf,
        sources=req.urls,
        additional_instruction=add_instruction
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
    Evaluate the answer for the Broadway digital lottery requirements task.
    """
    # Initialize evaluator (framework root is always non-critical)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall we have independent sub-requirements
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

    # Create the task's main critical node under the framework root
    main_node = evaluator.add_parallel(
        id="broadway_lottery_requirements",
        desc="Verify the answer identifies one qualifying currently-running Broadway musical with a qualifying digital lottery price, and states the required general Broadway digital lottery rules with supporting URLs.",
        parent=root,
        critical=True
    )

    # Extract structured information
    extraction = await evaluator.extract(
        prompt=prompt_extract_lottery_requirements(),
        template_class=LotteryExtraction,
        extraction_name="lottery_requirements_extraction"
    )

    # Record minimal custom info for clarity (optional)
    evaluator.add_custom_info(
        info={
            "selected_show_name": extraction.show.name if extraction and extraction.show else None,
            "selected_show_price_text": extraction.show.lottery_price if extraction and extraction.show else None,
            "selected_show_urls_count": len(extraction.show.urls) if extraction and extraction.show else 0,
        },
        info_type="extraction_summary",
        info_name="extraction_summary"
    )

    # Verify the identified qualifying show
    await verify_show(evaluator, main_node, extraction.show)

    # Verify general lottery requirements
    await verify_requirement_with_urls(
        evaluator,
        main_node,
        extraction.age_requirement,
        node_id="age_eligibility_requirement",
        node_desc="State the minimum age requirement to enter Broadway digital lotteries (18+), with a reference URL supporting it.",
        claim_text="The minimum age to enter Broadway digital lotteries is 18 years old.",
        add_instruction="Use the provided URLs to confirm that lottery entrants must be at least 18 years old. Accept '18+' or 'must be 18'."
    )

    await verify_requirement_with_urls(
        evaluator,
        main_node,
        extraction.photo_id_requirement,
        node_id="photo_id_required_for_pickup",
        node_desc="State that photo ID is required to pick up lottery tickets at the box office, with a reference URL supporting it.",
        claim_text="A valid photo ID is required to pick up Broadway digital lottery tickets at the box office.",
        add_instruction="Confirm that the policy explicitly requires a photo ID at pickup. Accept 'government-issued photo ID' or equivalent wording."
    )

    await verify_requirement_with_urls(
        evaluator,
        main_node,
        extraction.name_match_requirement,
        node_id="name_must_match_id",
        node_desc="State that the name on the lottery account/entry must match the name on the photo ID used for pickup, with a reference URL supporting it.",
        claim_text="The name on the winner's lottery account/entry must match the name on the photo ID used for pickup.",
        add_instruction="Look for wording that the name on the ID/credit card must match the winner's name or account entry."
    )

    await verify_requirement_with_urls(
        evaluator,
        main_node,
        extraction.claim_time_requirement,
        node_id="claim_time_window",
        node_desc="State the typical claim/purchase time window after winner notification (typically 60 minutes to 5 hours depending on the show), with a reference URL supporting it.",
        claim_text="Broadway digital lottery winners typically have between 60 minutes and 5 hours to claim and purchase tickets after notification; the window varies by show.",
        add_instruction=(
            "Confirm that the provided URLs indicate time windows for winners (e.g., 1 hour, 2 hours, up to several hours). "
            "The claim is about typical ranges, not an exact universal rule, and should be supported by examples showing windows within 1–5 hours."
        )
    )

    await verify_requirement_with_urls(
        evaluator,
        main_node,
        extraction.transfer_policy,
        node_id="ticket_transfer_policy",
        node_desc="State that digital lottery tickets are non-transferable (cannot be given/sold to another person), with a reference URL supporting it.",
        claim_text="Broadway digital lottery tickets are non-transferable and cannot be transferred to another person.",
        add_instruction="Look for explicit 'non-transferable' language or wording that prohibits transferring or picking up tickets under a different name."
    )

    # Return structured evaluation summary
    return evaluator.get_summary()