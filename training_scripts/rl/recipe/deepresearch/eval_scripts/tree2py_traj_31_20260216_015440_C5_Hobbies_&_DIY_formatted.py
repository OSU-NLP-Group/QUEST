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
TASK_ID = "va_beach_diy_workshops_tool_rental_mar_2026"
TASK_DESCRIPTION = (
    "A parent in Virginia Beach, Virginia, wants to enroll their 6-year-old child in free monthly DIY/craft workshops "
    "during March 2026 to develop hands-on building skills. They are specifically interested in workshops offered by major "
    "home improvement retailers. Additionally, the parent plans to complete a beginner woodworking project at home that will "
    "require renting power tools for approximately 5 hours. Please provide: (1) Two different workshop program options from "
    "major home improvement retailers that accept 6-year-olds and offer free monthly workshops, including their monthly "
    "schedule patterns (which Saturday they occur), time windows, and registration requirements; (2) A comparison of key differences "
    "between these two workshop programs, including their scheduling patterns and any distinguishing features; (3) An analysis of tool "
    "rental options from these retailers for the 5-hour project, including the minimum rental period available, an appropriate rental "
    "period recommendation for this project, and any cost comparison information between retailers."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class WorkshopOption(BaseModel):
    retailer_name: Optional[str] = None
    workshop_program_name: Optional[str] = None
    schedule_pattern: Optional[str] = None  # e.g., "first Saturday", "second Saturday"
    time_window: Optional[str] = None       # e.g., "9:00am–12:00pm"
    age_min: Optional[str] = None           # strings to allow flexible formats (e.g., "5", "5+", "ages 5-12")
    age_max: Optional[str] = None
    age_eligibility_text: Optional[str] = None  # free-form description from the answer
    registration_method: Optional[str] = None   # e.g., "online preregistration", "walk-in", "in-store sign-up"
    membership_requirement: Optional[str] = None  # e.g., "none", "requires membership", "MVP loyalty"
    duration: Optional[str] = None               # e.g., "60–90 minutes", "about 2 hours"
    cost_text: Optional[str] = None              # e.g., "free", "no cost"
    urls: List[str] = Field(default_factory=list)


class WorkshopsExtraction(BaseModel):
    options: List[WorkshopOption] = Field(default_factory=list)


class ToolRentalOption(BaseModel):
    retailer_name: Optional[str] = None
    min_rental_period: Optional[str] = None         # e.g., "4 hours", "half-day"
    rental_periods_available: List[str] = Field(default_factory=list)  # e.g., ["4-hour", "24-hour", "weekly"]
    pricing_summary: Optional[str] = None           # free-form short pricing notes
    urls: List[str] = Field(default_factory=list)


class ToolRentalExtraction(BaseModel):
    options: List[ToolRentalOption] = Field(default_factory=list)
    recommended_period: Optional[str] = None        # e.g., "24-hour (1 day)"
    cheaper_retailer: Optional[str] = None          # e.g., "Home Depot", "Lowe's"
    rental_flexibility_text: Optional[str] = None   # free-form notes on flexibility


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_workshops() -> str:
    return """
    Extract TWO different kids DIY/craft workshop program options from MAJOR U.S. HOME IMPROVEMENT RETAILERS (examples include The Home Depot, Lowe’s, Ace Hardware, Menards). They must be free monthly workshops and accept 6-year-old children.

    For each option found in the answer, extract the following fields:
    - retailer_name: The retailer brand (e.g., "The Home Depot", "Lowe's")
    - workshop_program_name: The official program name (e.g., "Kids Workshop")
    - schedule_pattern: The monthly schedule pattern (e.g., "first Saturday", "second Saturday", "third Saturday")
    - time_window: Typical time window (e.g., "9:00am–12:00pm")
    - age_min: Minimum age if stated (string)
    - age_max: Maximum age if stated (string)
    - age_eligibility_text: The text describing age eligibility exactly as presented in the answer (string), if available.
    - registration_method: Registration requirement/method (e.g., "online registration required", "walk-in", "in-store sign-up")
    - membership_requirement: Any membership requirement (e.g., "none", "MVP required", "loyalty program required"). If not applicable, use "none".
    - duration: Approximate session length (e.g., "60–90 minutes", "about 2 hours")
    - cost_text: Cost description (e.g., "free", "no cost")
    - urls: All URLs provided in the answer that support this workshop program. Must be actual URLs; include multiple if present.

    Rules:
    - Only include MAJOR home improvement retailers. Do not include craft stores or general community organizations.
    - If some field is not mentioned, set it to null (for singular values) or [] (for urls).
    - Extract exactly what the answer states without inventing.
    - Return a JSON object with a single key "options" which is an array of these objects.
    """


def prompt_extract_tool_rentals() -> str:
    return """
    Extract tool rental information from MAJOR home improvement retailers relevant to a beginner woodworking project (~5 hours).

    Return:
    - options: An array of objects, each with:
        • retailer_name
        • min_rental_period: Minimum rental period offered for power tools (string, e.g., "4 hours")
        • rental_periods_available: List of available rental periods beyond minimum (e.g., ["4-hour", "24-hour", "weekly"])
        • pricing_summary: Short summary of pricing (if the answer provided specifics; otherwise null)
        • urls: All URLs provided in the answer that support tool rental info for this retailer (list of URLs)

    - recommended_period: The recommended rental period for a ~5-hour project based on the available rental options (string)
    - cheaper_retailer: If the answer compares pricing, which retailer generally offers more competitive short-term pricing (string, else null)
    - rental_flexibility_text: A brief description of available rental period flexibility beyond the minimum (string or null)

    Rules:
    - Only extract what the answer states; do not invent.
    - If URLs are missing for a retailer, leave the list empty.
    - If any field is unavailable, set null or [] accordingly.
    """


# --------------------------------------------------------------------------- #
# Helper verification builders                                                #
# --------------------------------------------------------------------------- #
async def verify_workshop_option(
    evaluator: Evaluator,
    parent_node,
    option: WorkshopOption,
    option_index: int
) -> Dict[str, Any]:
    """
    Build and verify the subtree for a single workshop option.

    Returns a mapping of key leaf nodes for downstream prerequisites.
    """
    opt_node = evaluator.add_parallel(
        id=f"workshop_option_{option_index + 1}",
        desc=(
            "Analysis of {} major home improvement retailer's kids workshop program"
            .format("first" if option_index == 0 else "second")
        ),
        parent=parent_node,
        critical=False
    )

    # Prepare common values
    retailer = option.retailer_name or "unknown retailer"
    program_name = option.workshop_program_name or "kids workshop program"
    schedule = option.schedule_pattern or "unspecified schedule pattern"
    time_window = option.time_window or "unspecified time window"
    registration = option.registration_method or "unspecified registration method"
    duration = option.duration or "unspecified duration"
    membership_req = option.membership_requirement or "none"
    cost_text = (option.cost_text or "").lower().strip()
    urls = option.urls if option.urls else []

    # 1) Retailer identification (critical)
    retailer_leaf = evaluator.add_leaf(
        id=f"workshop_{option_index + 1}_retailer_identification",
        desc=(
            "Correctly identifies a major home improvement retailer that offers free monthly kids workshops"
        ),
        parent=opt_node,
        critical=True
    )
    claim_ident = (
        f"The kids workshop program '{program_name}' is offered by {retailer}, a major home improvement retailer, "
        f"and the workshops are free and occur monthly."
    )
    await evaluator.verify(
        claim=claim_ident,
        node=retailer_leaf,
        sources=urls if urls else None,
        additional_instruction=(
            "Confirm the page(s) describe an official kids workshop program run by the named retailer, that it is free, "
            "and that it occurs on a monthly cadence. Allow reasonable phrasing variants."
        )
    )

    # 2) Age eligibility (critical)
    age_leaf = evaluator.add_leaf(
        id=f"workshop_{option_index + 1}_age_eligibility",
        desc="Verifies that the workshop accepts 6-year-old children",
        parent=opt_node,
        critical=True
    )
    claim_age = (
        "This workshop accepts 6-year-old children (i.e., 6 is within the stated age range or explicitly allowed)."
    )
    await evaluator.verify(
        claim=claim_age,
        node=age_leaf,
        sources=urls if urls else None,
        additional_instruction=(
            "If the page states an age range (e.g., ages 5–12), consider 6 as accepted. Minor phrasing differences are acceptable."
        )
    )

    # 3) Schedule timing (critical)
    sched_leaf = evaluator.add_leaf(
        id=f"workshop_{option_index + 1}_schedule_timing",
        desc="Provides the correct monthly schedule pattern (which Saturday of the month)",
        parent=opt_node,
        critical=True
    )
    claim_sched = f"The workshop occurs on the '{schedule}' of each month."
    await evaluator.verify(
        claim=claim_sched,
        node=sched_leaf,
        sources=urls if urls else None,
        additional_instruction=(
            "Confirm the monthly pattern, such as 'first Saturday', 'second Saturday', etc. Accept equivalent phrasing."
        )
    )

    # 4) Time window (non-critical)
    time_leaf = evaluator.add_leaf(
        id=f"workshop_{option_index + 1}_time_window",
        desc="Specifies the correct time window for workshop availability",
        parent=opt_node,
        critical=False
    )
    claim_time = f"The typical time window for the workshop is '{time_window}'."
    await evaluator.verify(
        claim=claim_time,
        node=time_leaf,
        sources=urls if urls else None,
        additional_instruction=(
            "Confirm a typical start-end window (e.g., 9:00am–12:00pm). If multiple windows exist, the claim can be correct if one standard window is cited."
        )
    )

    # 5) Registration process (non-critical)
    reg_leaf = evaluator.add_leaf(
        id=f"workshop_{option_index + 1}_registration_process",
        desc="Describes the registration requirement and method",
        parent=opt_node,
        critical=False
    )
    claim_reg = f"Registration requirement and method: {registration}."
    await evaluator.verify(
        claim=claim_reg,
        node=reg_leaf,
        sources=urls if urls else None,
        additional_instruction=(
            "Verify that the method (e.g., online preregistration, walk-in, in-store sign-up) matches what is stated on the page."
        )
    )

    # 6) Membership requirement (only explicitly checked for option 2; still fine to verify non-crit for both if available)
    member_leaf = None
    if option_index == 1:
        member_leaf = evaluator.add_leaf(
            id=f"workshop_{option_index + 1}_membership_requirement",
            desc="If applicable, identifies any membership requirements for registration",
            parent=opt_node,
            critical=False
        )
        claim_member = (
            f"Membership requirement for registration is correctly identified as: {membership_req}."
        )
        await evaluator.verify(
            claim=claim_member,
            node=member_leaf,
            sources=urls if urls else None,
            additional_instruction=(
                "If the program requires any membership or loyalty sign-in, confirm it. If not required, 'none' is acceptable."
            )
        )

    # 7) Workshop duration (non-critical, explicitly for option 2 per rubric)
    duration_leaf = None
    if option_index == 1:
        duration_leaf = evaluator.add_leaf(
            id=f"workshop_{option_index + 1}_workshop_duration",
            desc="Provides the approximate duration of the workshop session",
            parent=opt_node,
            critical=False
        )
        claim_duration = f"The approximate workshop duration is '{duration}'."
        await evaluator.verify(
            claim=claim_duration,
            node=duration_leaf,
            sources=urls if urls else None,
            additional_instruction=(
                "Confirm an approximate session length (e.g., 60–90 minutes, 2 hours). Allow reasonable approximation."
            )
        )

    # 8) Reference URL validity (critical)
    ref_leaf = evaluator.add_leaf(
        id=f"workshop_{option_index + 1}_reference_url",
        desc="Provides valid URL supporting the workshop information",
        parent=opt_node,
        critical=True
    )
    if urls:
        claim_ref = (
            f"At least one of these URLs is a valid, relevant page that describes {retailer}'s kids workshop program details "
            f"(schedule, registration, age eligibility, or cost)."
        )
        await evaluator.verify(
            claim=claim_ref,
            node=ref_leaf,
            sources=urls,
            additional_instruction=(
                "Pass this check if any provided URL explicitly describes the official kids workshop program details."
            )
        )
    else:
        # No URL provided — this should fail
        claim_ref = "No URL was provided to support the workshop information."
        await evaluator.verify(
            claim=claim_ref,
            node=ref_leaf,
            sources=None,
            additional_instruction="This should be marked incorrect because supporting URLs are required."
        )

    # Return key nodes for dependency usage
    return {
        "retailer_leaf": retailer_leaf,
        "age_leaf": age_leaf,
        "schedule_leaf": sched_leaf,
        "ref_leaf": ref_leaf
    }


async def verify_workshop_comparison(
    evaluator: Evaluator,
    parent_node,
    option1: WorkshopOption,
    option2: WorkshopOption,
    prereq_nodes_opt1: Dict[str, Any],
    prereq_nodes_opt2: Dict[str, Any]
) -> None:
    """
    Build and verify the comparison subtree between two workshop options.
    """
    comp_node = evaluator.add_parallel(
        id="workshop_comparison",
        desc="Comparative analysis between the two workshop options",
        parent=parent_node,
        critical=False
    )

    # Collect schedules and cost info
    sched1 = (option1.schedule_pattern or "").strip()
    sched2 = (option2.schedule_pattern or "").strip()
    cost1 = (option1.cost_text or "").strip().lower()
    cost2 = (option2.cost_text or "").strip().lower()

    # 1) Schedule difference (critical)
    sched_diff_leaf = evaluator.add_leaf(
        id="schedule_difference",
        desc="Correctly identifies the difference in monthly scheduling between the two programs",
        parent=comp_node,
        critical=True
    )
    claim_sched_diff = (
        f"Program 1 schedule pattern: '{sched1}'. Program 2 schedule pattern: '{sched2}'. These two scheduling patterns are different."
    )
    await evaluator.verify(
        claim=claim_sched_diff,
        node=sched_diff_leaf,
        additional_instruction=(
            "Mark correct only if the two schedule patterns are not the same (case-insensitive, normalize synonyms)."
        ),
        extra_prerequisites=[
            prereq_nodes_opt1["schedule_leaf"],
            prereq_nodes_opt2["schedule_leaf"]
        ]
    )

    # 2) Cost verification (critical)
    cost_leaf = evaluator.add_leaf(
        id="cost_verification",
        desc="Verifies that both workshop options are free",
        parent=comp_node,
        critical=True
    )
    claim_cost = "Both workshop options are free to attend."
    # Use simple verification with prerequisites to avoid multi-URL logical issues
    await evaluator.verify(
        claim=claim_cost,
        node=cost_leaf,
        additional_instruction=(
            "Consider 'free', 'no cost', or equivalent phrasing as free. If either program indicates a fee, mark incorrect."
        ),
        extra_prerequisites=[
            prereq_nodes_opt1["retailer_leaf"],  # retailer identification claim includes 'free monthly'
            prereq_nodes_opt2["retailer_leaf"]
        ]
    )

    # 3) Distinguishing features (non-critical)
    features_leaf = evaluator.add_leaf(
        id="distinguishing_features",
        desc=(
            "Identifies at least one distinguishing feature between the programs (e.g., duration, rewards program, registration requirements)"
        ),
        parent=comp_node,
        critical=False
    )
    feat_candidates = []
    if (option1.registration_method or "") != (option2.registration_method or ""):
        feat_candidates.append("different registration methods")
    if (option1.duration or "") != (option2.duration or ""):
        feat_candidates.append("different session durations")
    if (option1.membership_requirement or "none").lower() != (option2.membership_requirement or "none").lower():
        feat_candidates.append("different membership requirements")

    distinguishing_text = ", ".join(feat_candidates) if feat_candidates else "unspecified distinguishing feature"
    claim_features = (
        f"At least one distinguishing feature is identified between the programs (e.g., {distinguishing_text})."
    )
    await evaluator.verify(
        claim=claim_features,
        node=features_leaf,
        additional_instruction=(
            "If any notable difference exists (duration, registration, membership, rewards), mark correct. "
            "The exact phrasing may vary; allow reasonable variants."
        ),
        extra_prerequisites=[
            prereq_nodes_opt1["ref_leaf"],
            prereq_nodes_opt2["ref_leaf"]
        ]
    )


async def verify_tool_rental_analysis(
    evaluator: Evaluator,
    parent_node,
    tool_info: ToolRentalExtraction
) -> None:
    """
    Build and verify the tool rental analysis subtree.
    """
    rent_node = evaluator.add_parallel(
        id="tool_rental_analysis",
        desc="Analysis of tool rental options for a 5-hour beginner woodworking project",
        parent=parent_node,
        critical=False
    )

    # Aggregate URLs from all rental options
    all_rental_urls: List[str] = []
    retailer_names: List[str] = []
    min_periods: List[str] = []
    flex_periods: List[str] = []
    for opt in tool_info.options:
        retailer_names.append(opt.retailer_name or "unknown retailer")
        if opt.min_rental_period:
            min_periods.append(opt.min_rental_period)
        if opt.rental_periods_available:
            flex_periods.extend(opt.rental_periods_available)
        if opt.urls:
            all_rental_urls.extend(opt.urls)

    # 1) Minimum rental period (critical)
    min_leaf = evaluator.add_leaf(
        id="minimum_rental_period",
        desc="Correctly identifies the minimum rental period offered by major home improvement retailers",
        parent=rent_node,
        critical=True
    )
    min_period_summary = ", ".join(set([p.strip() for p in min_periods if p])) or "unspecified minimum period"
    claim_min = (
        f"Major home improvement retailers offer a minimum tool rental period such as: {min_period_summary}."
    )
    await evaluator.verify(
        claim=claim_min,
        node=min_leaf,
        sources=all_rental_urls if all_rental_urls else None,
        additional_instruction=(
            "Confirm minimum rental period(s) (e.g., 4-hour minimum). Allow retailer-specific variations; "
            "the claim is correct if typical minimums are supported by any of the provided pages."
        )
    )

    # 2) Rental period recommendation (critical)
    rec_leaf = evaluator.add_leaf(
        id="rental_period_recommendation",
        desc="Recommends an appropriate rental period for a 5-hour project based on available options",
        parent=rent_node,
        critical=True
    )
    recommended = (tool_info.recommended_period or "").strip() or "unspecified recommendation"
    claim_rec = (
        f"For a ~5-hour beginner woodworking project, the recommended rental period is '{recommended}'."
    )
    await evaluator.verify(
        claim=claim_rec,
        node=rec_leaf,
        additional_instruction=(
            "Reason logically from typical options (e.g., 4-hour minimum may be insufficient for 5 hours; "
            "24-hour (1 day) is often appropriate). Mark correct if the recommendation matches reasonable practice."
        )
    )

    # 3) Cost comparison (non-critical)
    cost_cmp_leaf = evaluator.add_leaf(
        id="cost_comparison",
        desc="Identifies which retailer generally offers more competitive pricing for short-term tool rentals",
        parent=rent_node,
        critical=False
    )
    cheaper = (tool_info.cheaper_retailer or "").strip() or "unspecified"
    claim_cost_cmp = (
        f"For short-term tool rentals, {cheaper} generally offers more competitive pricing among the retailers considered."
    )
    await evaluator.verify(
        claim=claim_cost_cmp,
        node=cost_cmp_leaf,
        sources=all_rental_urls if all_rental_urls else None,
        additional_instruction=(
            "If the provided pages indicate price differences, confirm which retailer is generally cheaper for short-term rentals. "
            "If ambiguous, this may be incorrect."
        )
    )

    # 4) Rental flexibility (non-critical)
    flex_leaf = evaluator.add_leaf(
        id="rental_flexibility",
        desc="Describes the available rental period options beyond the minimum (e.g., daily, weekly)",
        parent=rent_node,
        critical=False
    )
    flexibility_text = (tool_info.rental_flexibility_text or "").strip()
    # Fallback from collected options if not in free text
    if not flexibility_text and flex_periods:
        flexibility_text = ", ".join(sorted(set([p.strip() for p in flex_periods if p])))

    claim_flex = (
        f"Available rental period options beyond the minimum include: {flexibility_text}."
    )
    await evaluator.verify(
        claim=claim_flex,
        node=flex_leaf,
        sources=all_rental_urls if all_rental_urls else None,
        additional_instruction=(
            "Confirm additional rental periods such as 24-hour (daily) or weekly options as shown on the pages."
        )
    )

    # 5) Reference URL validity (critical)
    ref3_leaf = evaluator.add_leaf(
        id="reference_url_3",
        desc="Provides valid URL supporting the tool rental information",
        parent=rent_node,
        critical=True
    )
    if all_rental_urls:
        claim_ref3 = (
            "At least one of the provided URLs is a valid, relevant page that describes tool rental terms or pricing."
        )
        await evaluator.verify(
            claim=claim_ref3,
            node=ref3_leaf,
            sources=all_rental_urls,
            additional_instruction=(
                "Pass if any URL clearly describes rental options, minimums, or pricing at a major home improvement retailer."
            )
        )
    else:
        claim_ref3 = "No tool rental URL was provided to support the information."
        await evaluator.verify(
            claim=claim_ref3,
            node=ref3_leaf,
            sources=None,
            additional_instruction="This should be marked incorrect because supporting URLs are required."
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
    Evaluate an answer for the Virginia Beach DIY workshops and tool rental analysis task.
    """
    # Initialize evaluator (root is non-critical by design in framework; use parallel aggregation)
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

    # Extract workshop options and tool rental info
    workshops = await evaluator.extract(
        prompt=prompt_extract_workshops(),
        template_class=WorkshopsExtraction,
        extraction_name="workshops_extraction"
    )

    tool_rentals = await evaluator.extract(
        prompt=prompt_extract_tool_rentals(),
        template_class=ToolRentalExtraction,
        extraction_name="tool_rental_extraction"
    )

    # Normalize: ensure exactly two workshop options by padding or truncation
    options = workshops.options if workshops.options else []
    if len(options) < 2:
        # Pad with empty options
        options = options + [WorkshopOption() for _ in range(2 - len(options))]
    else:
        options = options[:2]

    opt1, opt2 = options[0], options[1]

    # Build workshop option subtrees
    prereq_opt1 = await verify_workshop_option(evaluator, root, opt1, 0)
    prereq_opt2 = await verify_workshop_option(evaluator, root, opt2, 1)

    # Build comparison subtree
    await verify_workshop_comparison(
        evaluator, root, opt1, opt2, prereq_opt1, prereq_opt2
    )

    # Build tool rental analysis subtree
    await verify_tool_rental_analysis(evaluator, root, tool_rentals)

    # Return structured evaluation summary
    return evaluator.get_summary()