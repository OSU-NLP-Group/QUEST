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
TASK_ID = "tsa_touchless_id_lga_apple_digital_id"
TASK_DESCRIPTION = (
    "I'm planning to fly out of LaGuardia Airport (LGA) and want to use the TSA PreCheck Touchless ID feature for faster "
    "security screening. I also want to create an Apple Digital ID using my U.S. passport. Please provide the following "
    "information: (1) Which airlines from the TSA PreCheck Touchless ID participating airlines list support this feature "
    "at LaGuardia Airport? (2) What are the complete device requirements for creating an Apple Digital ID with a U.S. passport? "
    "Include both iPhone and Apple Watch specifications. (3) What are all the eligibility requirements I need to meet to use "
    "TSA PreCheck Touchless ID?"
)

CANONICAL_AIRLINES = [
    "Alaska Airlines",
    "American Airlines",
    "Delta Air Lines",
    "Southwest Airlines",
    "United Airlines",
]

AIRLINE_SHORT = {
    "Alaska Airlines": "alaska",
    "American Airlines": "american",
    "Delta Air Lines": "delta",
    "Southwest Airlines": "southwest",
    "United Airlines": "united",
}


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class AirlineSupport(BaseModel):
    airline: Optional[str] = None  # Should be one of the five canonical names above (case-insensitive allowed)
    sources: List[str] = Field(default_factory=list)  # URLs supporting that this airline supports Touchless ID at LGA


class TouchlessIDExtraction(BaseModel):
    # Airlines at LGA
    airlines: List[AirlineSupport] = Field(default_factory=list)

    # Apple Digital ID device requirements (U.S. passport)
    iphone_requirement: Optional[str] = None
    iphone_sources: List[str] = Field(default_factory=list)
    watch_requirement: Optional[str] = None
    watch_sources: List[str] = Field(default_factory=list)

    # TSA PreCheck Touchless ID eligibility
    membership_requirement: Optional[str] = None
    membership_sources: List[str] = Field(default_factory=list)
    passport_upload_requirement: Optional[str] = None
    passport_upload_sources: List[str] = Field(default_factory=list)
    opt_in_requirement: Optional[str] = None
    opt_in_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    You must extract structured information from the answer about TSA PreCheck Touchless ID at LaGuardia (LGA) and Apple Digital ID setup using a U.S. passport.

    PART 1 — Airlines at LGA:
    - Consider only these 5 TSA PreCheck Touchless ID participating airlines: Alaska Airlines, American Airlines, Delta Air Lines, Southwest Airlines, United Airlines.
    - From the answer text, identify which of these airlines the answer claims SUPPORT the TSA PreCheck Touchless ID feature specifically at LaGuardia Airport (LGA).
    - For each claimed airline, extract:
        • airline: The airline name exactly as one of the five canonical names above (use best match if the answer uses a variant).
        • sources: All URLs mentioned in the answer that the answer uses to support that this airline supports Touchless ID at LGA.
      If no supporting URLs are provided for that airline, return an empty array for sources.

    PART 2 — Apple Digital ID device requirements (U.S. passport):
    - Extract the device requirements text for iPhone (include model and iOS version if stated) for creating an Apple Digital ID with a U.S. passport.
      • iphone_requirement: the requirement text exactly as stated in the answer.
      • iphone_sources: all URLs the answer cites for that iPhone requirement.
    - Extract the device requirements text for Apple Watch (include model and watchOS version if stated) for creating an Apple Digital ID with a U.S. passport.
      • watch_requirement: the requirement text exactly as stated in the answer.
      • watch_sources: all URLs the answer cites for that Apple Watch requirement.
    - If not mentioned, set the text to null; if no URLs are provided, set the corresponding sources array to empty.

    PART 3 — TSA PreCheck Touchless ID eligibility requirements:
    - Extract three distinct requirements as stated in the answer, with their respective sources:
      • membership_requirement: The requirement to be a TSA PreCheck traveler with a Known Traveler Number (KTN), if the answer mentions it. Use the answer's phrasing.
      • membership_sources: URLs cited for that membership requirement.
      • passport_upload_requirement: The requirement to have valid passport info uploaded/saved in the participating airline profile, if the answer mentions it. Use the answer's phrasing.
      • passport_upload_sources: URLs cited for that passport upload requirement.
      • opt_in_requirement: The requirement to opt-in (via airline profile or at check-in) and the indicator showing on the mobile boarding pass, if the answer mentions it. Use the answer's phrasing.
      • opt_in_sources: URLs cited for that opt-in requirement.
    - If any of these three are not mentioned, set the text to null; if there are no URLs, set the sources array to empty.

    RULES:
    - Do not invent URLs. Extract only URLs explicitly present in the answer text; accept raw URLs or markdown links.
    - Normalize airline names to one of: Alaska Airlines, American Airlines, Delta Air Lines, Southwest Airlines, United Airlines.
    - If an airline is mentioned as a general participant but NOT specifically at LGA, do NOT include it.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_airline_to_canonical(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = name.strip().lower()
    # Simple heuristics to map common variants to canonical names
    if "alaska" in n:
        return "Alaska Airlines"
    if "american" in n:
        return "American Airlines"
    if "delta" in n:
        return "Delta Air Lines"
    if "southwest" in n:
        return "Southwest Airlines"
    if "united" in n:
        return "United Airlines"
    return None


def _find_airline_entry(extraction: TouchlessIDExtraction, canonical_name: str) -> Optional[AirlineSupport]:
    for it in extraction.airlines:
        mapped = _normalize_airline_to_canonical(it.airline)
        if mapped == canonical_name:
            return it
    return None


def _nonempty_text(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


def _nonempty_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and len([u for u in urls if isinstance(u, str) and u.strip()]) > 0)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_airlines_at_lga(evaluator: Evaluator, parent_node, extraction: TouchlessIDExtraction) -> None:
    """
    Build verification for: Which of the five participating airlines support TSA PreCheck Touchless ID at LGA.
    """
    airlines_node = evaluator.add_parallel(
        id="participating_airlines_lga",
        desc="Participating airlines that support TSA PreCheck Touchless ID at LaGuardia (LGA)",
        parent=parent_node,
        critical=False,
    )

    for airline in CANONICAL_AIRLINES:
        short = AIRLINE_SHORT[airline]
        sub = evaluator.add_parallel(
            id=f"airline_{short}",
            desc=f"{airline}: Support for TSA PreCheck Touchless ID at LGA",
            parent=airlines_node,
            critical=False,
        )

        entry = _find_airline_entry(extraction, airline)
        claimed = entry is not None
        sources = (entry.sources if entry else []) if entry else []
        sources = [u for u in sources if isinstance(u, str) and u.strip()]

        # Whether the answer actually claimed this airline supports at LGA (non-critical informational check)
        evaluator.add_custom_node(
            result=claimed,
            id=f"airline_{short}_claimed_in_answer",
            desc=f"Answer claims {airline} supports TSA PreCheck Touchless ID at LGA",
            parent=sub,
            critical=False,
        )

        # Require sources to be provided (critical within this airline)
        src_provided = evaluator.add_custom_node(
            result=_nonempty_urls(sources),
            id=f"airline_{short}_sources_provided",
            desc=f"Sources provided for {airline} support at LGA",
            parent=sub,
            critical=True,
        )

        # Verify that the sources actually support the claim (critical within this airline)
        supported_leaf = evaluator.add_leaf(
            id=f"airline_{short}_supported_by_sources",
            desc=f"{airline} supports TSA PreCheck Touchless ID at LGA (supported by cited sources)",
            parent=sub,
            critical=True,
        )
        claim = f"{airline} supports TSA PreCheck Touchless ID at LaGuardia Airport (LGA)."
        # This leaf automatically depends on the critical sibling src_provided via evaluator.verify precondition resolution.
        await evaluator.verify(
            claim=claim,
            node=supported_leaf,
            sources=sources,
            additional_instruction=(
                "The claim is correct only if the cited page(s) explicitly indicate that TSA PreCheck Touchless ID "
                "(aka TSA Identity Verification using a Digital ID/mobile ID) is offered for this specific airline at LGA. "
                "Accept official TSA, airline, or airport pages. Generic participation pages without LGA context are insufficient."
            ),
        )


async def verify_apple_device_requirements(evaluator: Evaluator, parent_node, extraction: TouchlessIDExtraction) -> None:
    """
    Build verification for Apple Digital ID device requirements using a U.S. passport:
    - iPhone requirements
    - Apple Watch requirements
    Both are critical under this category according to the rubric.
    """
    apple_node = evaluator.add_parallel(
        id="apple_digital_id_device_requirements",
        desc="Apple Digital ID device requirements for U.S. passport (iPhone and Apple Watch)",
        parent=parent_node,
        critical=False,
    )

    # iPhone requirements (Critical)
    iphone_node = evaluator.add_parallel(
        id="iphone_requirements",
        desc="iPhone requirements for creating Apple Digital ID using a U.S. passport",
        parent=apple_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty_text(extraction.iphone_requirement),
        id="iphone_req_text_provided",
        desc="Answer provides iPhone requirement text",
        parent=iphone_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty_urls(extraction.iphone_sources),
        id="iphone_req_sources_provided",
        desc="Sources provided for iPhone requirements",
        parent=iphone_node,
        critical=True,
    )
    iphone_verify = evaluator.add_leaf(
        id="iphone_req_supported_by_sources",
        desc="iPhone requirement text is supported by cited sources",
        parent=iphone_node,
        critical=True,
    )
    iphone_claim = (
        f"The device requirements for creating an Apple Digital ID with a U.S. passport on iPhone are: "
        f"{extraction.iphone_requirement or ''}"
    )
    await evaluator.verify(
        claim=iphone_claim,
        node=iphone_verify,
        sources=extraction.iphone_sources,
        additional_instruction=(
            "Verify that the cited page(s) explicitly state the iPhone device requirements for adding/creating a Digital ID "
            "with a U.S. passport in Apple Wallet, including both the minimum iPhone model and minimum iOS version if present. "
            "Minor wording differences are acceptable, but the meaning must align."
        ),
    )

    # Apple Watch requirements (Critical)
    watch_node = evaluator.add_parallel(
        id="apple_watch_requirements",
        desc="Apple Watch requirements for creating Apple Digital ID using a U.S. passport",
        parent=apple_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty_text(extraction.watch_requirement),
        id="watch_req_text_provided",
        desc="Answer provides Apple Watch requirement text",
        parent=watch_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty_urls(extraction.watch_sources),
        id="watch_req_sources_provided",
        desc="Sources provided for Apple Watch requirements",
        parent=watch_node,
        critical=True,
    )
    watch_verify = evaluator.add_leaf(
        id="watch_req_supported_by_sources",
        desc="Apple Watch requirement text is supported by cited sources",
        parent=watch_node,
        critical=True,
    )
    watch_claim = (
        f"The device requirements for creating an Apple Digital ID with a U.S. passport on Apple Watch are: "
        f"{extraction.watch_requirement or ''}"
    )
    await evaluator.verify(
        claim=watch_claim,
        node=watch_verify,
        sources=extraction.watch_sources,
        additional_instruction=(
            "Verify that the cited page(s) explicitly state the Apple Watch device requirements for adding/creating a Digital ID "
            "with a U.S. passport in Apple Wallet, including both the minimum Apple Watch model and minimum watchOS version if present. "
            "Minor wording differences are acceptable, but the meaning must align."
        ),
    )


async def verify_tsa_eligibility(evaluator: Evaluator, parent_node, extraction: TouchlessIDExtraction) -> None:
    """
    Build verification for TSA PreCheck Touchless ID eligibility requirements:
    - TSA PreCheck membership with KTN
    - Passport info uploaded to participating airline profile
    - Opt-in via airline profile or check-in; indicator on mobile boarding pass
    Each is critical under this category according to the rubric.
    """
    elig_node = evaluator.add_parallel(
        id="tsa_precheck_touchless_id_eligibility",
        desc="Eligibility requirements for using TSA PreCheck Touchless ID",
        parent=parent_node,
        critical=False,
    )

    # 1) TSA PreCheck membership with KTN (Critical)
    mem_node = evaluator.add_parallel(
        id="tsa_precheck_membership",
        desc="Requirement: Be a current TSA PreCheck traveler with a Known Traveler Number (KTN)",
        parent=elig_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty_text(extraction.membership_requirement),
        id="membership_req_text_provided",
        desc="Answer provides membership requirement text",
        parent=mem_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty_urls(extraction.membership_sources),
        id="membership_req_sources_provided",
        desc="Sources provided for membership requirement",
        parent=mem_node,
        critical=True,
    )
    mem_verify = evaluator.add_leaf(
        id="membership_req_supported_by_sources",
        desc="Membership requirement (with KTN) is supported by cited sources",
        parent=mem_node,
        critical=True,
    )
    mem_claim = (
        f"Eligibility requires: {extraction.membership_requirement or ''}"
    )
    await evaluator.verify(
        claim=mem_claim,
        node=mem_verify,
        sources=extraction.membership_sources,
        additional_instruction=(
            "Confirm that the cited page(s) clearly state that to use TSA PreCheck Touchless ID you must be an active TSA PreCheck traveler "
            "and have a Known Traveler Number (KTN)."
        ),
    )

    # 2) Passport info uploaded to airline profile (Critical)
    pass_node = evaluator.add_parallel(
        id="passport_upload_requirement",
        desc="Requirement: Valid passport info uploaded/saved to participating airline profile",
        parent=elig_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty_text(extraction.passport_upload_requirement),
        id="passport_upload_text_provided",
        desc="Answer provides passport upload requirement text",
        parent=pass_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty_urls(extraction.passport_upload_sources),
        id="passport_upload_sources_provided",
        desc="Sources provided for passport upload requirement",
        parent=pass_node,
        critical=True,
    )
    pass_verify = evaluator.add_leaf(
        id="passport_upload_supported_by_sources",
        desc="Passport upload requirement is supported by cited sources",
        parent=pass_node,
        critical=True,
    )
    pass_claim = (
        f"Eligibility requires: {extraction.passport_upload_requirement or ''}"
    )
    await evaluator.verify(
        claim=pass_claim,
        node=pass_verify,
        sources=extraction.passport_upload_sources,
        additional_instruction=(
            "Confirm that the cited page(s) clearly state that to use TSA PreCheck Touchless ID, valid passport information "
            "must be saved/uploaded in your participating airline profile."
        ),
    )

    # 3) Opt-in and indicator on mobile boarding pass (Critical)
    opt_node = evaluator.add_parallel(
        id="opt_in_requirement",
        desc="Requirement: Opt-in via airline profile or at check-in; indicator on mobile boarding pass",
        parent=elig_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty_text(extraction.opt_in_requirement),
        id="opt_in_text_provided",
        desc="Answer provides opt-in requirement text (and indicator on mobile boarding pass)",
        parent=opt_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty_urls(extraction.opt_in_sources),
        id="opt_in_sources_provided",
        desc="Sources provided for opt-in requirement",
        parent=opt_node,
        critical=True,
    )
    opt_verify = evaluator.add_leaf(
        id="opt_in_supported_by_sources",
        desc="Opt-in requirement and mobile boarding pass indicator are supported by cited sources",
        parent=opt_node,
        critical=True,
    )
    opt_claim = (
        f"Eligibility requires: {extraction.opt_in_requirement or ''}"
    )
    await evaluator.verify(
        claim=opt_claim,
        node=opt_verify,
        sources=extraction.opt_in_sources,
        additional_instruction=(
            "Confirm that the cited page(s) clearly state that you must opt-in either via your airline profile or at check-in, "
            "and that a TSA PreCheck Touchless ID indicator appears on your mobile boarding pass."
        ),
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
    Entry point for evaluating an answer on TSA PreCheck Touchless ID at LGA and Apple Digital ID device requirements.
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

    # Extract all relevant structured data from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=TouchlessIDExtraction,
        extraction_name="touchless_id_extraction",
    )

    # Optional: add canonical list info for reference
    evaluator.add_ground_truth(
        {
            "canonical_participating_airlines": CANONICAL_AIRLINES,
            "categories": [
                "Participating airlines at LGA",
                "Apple Digital ID device requirements (iPhone + Apple Watch)",
                "TSA PreCheck Touchless ID eligibility requirements",
            ],
        },
        gt_type="ground_truth",
    )

    # Build verification subtrees
    await verify_airlines_at_lga(evaluator, root, extraction)
    await verify_apple_device_requirements(evaluator, root, extraction)
    await verify_tsa_eligibility(evaluator, root, extraction)

    # Return structured summary
    return evaluator.get_summary()