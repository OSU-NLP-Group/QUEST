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
TASK_ID = "colbert_survivor_prof_standup_venue_capacity"
TASK_DESCRIPTION = """
A contestant from CBS's reality show "Survivor" who is also a robotics professor at a university in Florida performed stand-up comedy on a major late-night talk show hosted by Stephen Colbert in February 2026. This person appeared on both Season 37 (subtitled "David vs. Goliath") and Season 50 of Survivor. Identify the name of the venue where this stand-up performance took place, and provide its seating capacity specifically configured for TV audience purposes (as opposed to its original theater capacity). Include a reference URL that confirms the venue's TV audience seating capacity.
"""

EXPECTED_CONTEXT = {
    "expected_venue_name": "Ed Sullivan Theater",
    "expected_address": "1697 Broadway, New York, NY",
    "tv_capacity_approx_range": "about 400–461 seats",
    "show_name": "The Late Show with Stephen Colbert",
    "expected_date": "February 27, 2026"
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueInfo(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    tv_audience_capacity: Optional[str] = None
    tv_capacity_source_urls: List[str] = Field(default_factory=list)

    original_capacity_statement: Optional[str] = None
    original_capacity_urls: List[str] = Field(default_factory=list)

    tv_vs_original_clarity: Optional[bool] = None
    tv_vs_original_statement: Optional[str] = None


class PerformanceContext(BaseModel):
    show_name: Optional[str] = None
    performance_type: Optional[str] = None
    performance_date: Optional[str] = None
    context_urls: List[str] = Field(default_factory=list)


class PerformerInfo(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AnswerExtraction(BaseModel):
    venue: VenueInfo = VenueInfo()
    performance: PerformanceContext = PerformanceContext()
    performer: PerformerInfo = PerformerInfo()


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_answer() -> str:
    return """
    Extract the following structured information strictly from the answer text.

    1) venue:
       - name: The venue name stated for the performance.
       - address: The venue address/location if given (e.g., "1697 Broadway, New York, NY").
       - tv_audience_capacity: The specific seating capacity number/text the answer claims for the TV/studio audience configuration (NOT the original Broadway theater capacity). Keep it verbatim as written (e.g., "461", "about 450", "around 400").
       - tv_capacity_source_urls: A list of URL(s) explicitly cited in the answer that directly confirm the TV/studio audience seating capacity (not the original capacity). Include only URLs actually present in the answer.
       - original_capacity_statement: Any sentence or phrase the answer uses to describe the venue's original theater capacity (e.g., "original capacity over 1,500", "1,760 seats", etc.). Use null if not provided.
       - original_capacity_urls: Any URL(s) cited in the answer specifically about the original theater capacity. Use an empty list if none.
       - tv_vs_original_clarity: true if the answer explicitly clarifies that the provided capacity is for the TV/studio audience configuration (and not the original theater capacity). Otherwise false or null.
       - tv_vs_original_statement: The exact phrase (if any) where the answer clarifies the distinction between TV/studio audience capacity and original theater capacity. Null if not present.

    2) performance:
       - show_name: The show name stated (e.g., "The Late Show with Stephen Colbert"), if present.
       - performance_type: The performance type if stated (e.g., "stand-up", "stand-up comedy").
       - performance_date: Any performance date stated (e.g., "February 27, 2026"), else null.
       - context_urls: Any other URLs in the answer about the performance/show/appearance (e.g., official pages, YouTube clips, press coverage). Exclude the URLs already listed under tv_capacity_source_urls and original_capacity_urls; avoid duplication.

    3) performer:
       - name: The performer’s name if stated.
       - sources: Any URLs cited in the answer specifically about the performer's identity (e.g., Survivor seasons info, university profile). Use an empty list if none.

    Rules for URL fields:
    - Include only URLs explicitly present in the answer. Do NOT invent URLs.
    - Accept plain URLs or markdown links; extract the actual URL.
    - If a URL is missing a protocol, prepend "http://".
    - Remove duplicates across fields; keep each URL under the most relevant field.

    If any field is not present in the answer, set it to null or an empty list as appropriate.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_venue_and_capacity_checks(evaluator: Evaluator, parent, ex: AnswerExtraction) -> None:
    """
    Build critical checks for venue and TV-audience capacity deliverables.
    According to rubric, all children here are critical.
    """
    node = evaluator.add_parallel(
        id="Venue_And_Capacity_Required_Output",
        desc="Response provides the required venue and TV-audience capacity deliverables, consistent with constraints.",
        parent=parent,
        critical=True
    )

    # 1) Venue name is Ed Sullivan Theater (answer-internal check)
    leaf_name = evaluator.add_leaf(
        id="Venue_Name_Is_Ed_Sullivan_Theater",
        desc="Venue name is Ed Sullivan Theater.",
        parent=node,
        critical=True
    )
    provided_name = ex.venue.name or ""
    claim_name = f"The venue named in the answer ('{provided_name}') refers to the Ed Sullivan Theater (allowing minor variants like 'The Ed Sullivan Theater')."
    await evaluator.verify(
        claim=claim_name,
        node=leaf_name,
        additional_instruction="Judge based on the answer text. Consider reasonable variants/capitalization of 'Ed Sullivan Theater' as equivalent."
    )

    # 2) Venue address/location is 1697 Broadway, New York, NY (NYC) (answer-internal check)
    leaf_addr = evaluator.add_leaf(
        id="Venue_Address_Is_1697_Broadway_NYC",
        desc="Venue address/location is 1697 Broadway, New York, NY (New York City).",
        parent=node,
        critical=True
    )
    claim_addr = "The answer states the venue's location as 1697 Broadway, New York, NY (New York City), or an equivalent phrasing/address."
    await evaluator.verify(
        claim=claim_addr,
        node=leaf_addr,
        additional_instruction="Only evaluate the answer text. Accept equivalent address variants that clearly denote '1697 Broadway, New York, NY' (NYC)."
    )

    # 3) TV audience capacity in approximate range about 400–461 (answer-internal range compliance)
    leaf_range = evaluator.add_leaf(
        id="TV_Audience_Capacity_In_Approx_Range",
        desc="States a TV-audience seating capacity that falls within the constrained approximate range (about 400–461 seats).",
        parent=node,
        critical=True
    )
    cap_txt = ex.venue.tv_audience_capacity or ""
    claim_range = f"The provided TV/studio audience seating capacity value in the answer ('{cap_txt}') falls within approximately 400 to 461 seats (inclusive), allowing for words like 'about' or 'around'."
    await evaluator.verify(
        claim=claim_range,
        node=leaf_range,
        additional_instruction="Base your judgment on the answer text. If a numeric value is extractable from the phrase, check if it is in [400, 461]. If multiple values appear, the explicit one for TV audience should be considered."
    )

    # 4) Not conflated with original theater capacity (answer-internal clarity)
    leaf_not_conflate = evaluator.add_leaf(
        id="TV_Audience_Capacity_Not_Conflated_With_Original",
        desc="Clearly identifies the stated capacity as the TV-audience configuration (not the original theater capacity).",
        parent=node,
        critical=True
    )
    clarity_phrase = ex.venue.tv_vs_original_statement or ""
    claim_not_conflate = (
        "In the answer, the seating capacity provided is explicitly identified as the TV/studio audience configuration "
        "for the show taping and is not presented as the historical/original Broadway theater capacity."
    )
    await evaluator.verify(
        claim=claim_not_conflate,
        node=leaf_not_conflate,
        additional_instruction="Look for explicit wording such as 'TV audience', 'studio audience', 'for tapings', 'television configuration', etc. The answer must make the distinction clear."
    )

    # 5) Original Broadway theater capacity is over 1,500 (answer-internal presence/claim)
    leaf_orig_over = evaluator.add_leaf(
        id="Original_Theater_Capacity_Over_1500",
        desc="States that the venue's original Broadway theater capacity is over 1,500 seats.",
        parent=node,
        critical=True
    )
    orig_stmt = ex.venue.original_capacity_statement or ""
    claim_orig_over = (
        "The answer states or indicates that the venue’s original/historic Broadway theater capacity is over 1,500 seats "
        "(e.g., mentions numbers like 1,600–1,800+, 'over 1,500', etc.)."
    )
    await evaluator.verify(
        claim=claim_orig_over,
        node=leaf_orig_over,
        additional_instruction="Only judge based on the answer text. We are not asking you to verify this against the web here—just whether the answer clearly states such a fact."
    )

    # 6) TV capacity smaller than original (answer-internal relationship)
    leaf_smaller = evaluator.add_leaf(
        id="TV_Capacity_Smaller_Than_Original",
        desc="States (or otherwise makes clear) that the TV-audience capacity is smaller than the original theater capacity.",
        parent=node,
        critical=True
    )
    claim_smaller = (
        "The answer makes it clear that the TV/studio audience capacity is smaller than the original theater capacity."
    )
    await evaluator.verify(
        claim=claim_smaller,
        node=leaf_smaller,
        additional_instruction="Only evaluate the answer text. It's sufficient if the answer explicitly says 'smaller' or if both numbers are present and the TV capacity is numerically smaller."
    )

    # 7) Reference URL confirms TV capacity (source-grounded)
    tv_urls = ex.venue.tv_capacity_source_urls or []
    if tv_urls:
        leaf_ref = evaluator.add_leaf(
            id="Reference_URL_Confirms_TV_Capacity",
            desc="Provides at least one reference URL that explicitly supports the stated TV-audience seating capacity figure/configuration.",
            parent=node,
            critical=True
        )
        cap_claim_value = ex.venue.tv_audience_capacity or "the claimed TV/studio audience capacity"
        claim_ref = (
            f"The provided source(s) explicitly support that the Ed Sullivan Theater's TV/studio audience seating capacity "
            f"for The Late Show with Stephen Colbert is {cap_claim_value} (or an unambiguous equivalent)."
        )
        await evaluator.verify(
            claim=claim_ref,
            node=leaf_ref,
            sources=tv_urls,
            additional_instruction=(
                "Confirm that the page(s) explicitly describe the TV/studio audience seating capacity for the show taping "
                "(not the historical/original Broadway capacity). Allow for minor variations like 'about' or 'approximately'."
            )
        )
    else:
        # No TV capacity source URL provided – this is a hard failure for this leaf per rubric
        evaluator.add_custom_node(
            result=False,
            id="Reference_URL_Confirms_TV_Capacity",
            desc="Provides at least one reference URL that explicitly supports the stated TV-audience seating capacity figure/configuration.",
            parent=node,
            critical=True
        )


async def build_performance_context_checks(evaluator: Evaluator, parent, ex: AnswerExtraction) -> None:
    """
    Non-critical performance context consistency checks.
    If certain details are not mentioned in the answer, we pass by default as 'not applicable' to respect the rubric phrasing.
    """
    node = evaluator.add_parallel(
        id="Performance_Context_Consistency",
        desc="Response is consistent with the constrained performance context (Late Show with Colbert; stand-up; Feb 27, 2026).",
        parent=parent,
        critical=False
    )

    # Show identification
    if ex.performance.show_name and ex.performance.show_name.strip():
        leaf_show = evaluator.add_leaf(
            id="Context_Late_Show_With_Colbert",
            desc="Identifies the performance as on The Late Show with Stephen Colbert (if the show is mentioned).",
            parent=node,
            critical=False
        )
        claim_show = (
            "The answer identifies the performance as on 'The Late Show with Stephen Colbert' "
            "(allowing reasonable variants like 'Late Show with Stephen Colbert')."
        )
        await evaluator.verify(
            claim=claim_show,
            node=leaf_show,
            additional_instruction="Judge based on the answer text. Accept minor naming variants of the show."
        )
    else:
        # Not mentioned; mark as not applicable pass to avoid undue penalty
        evaluator.add_custom_node(
            result=True,
            id="Context_Late_Show_With_Colbert",
            desc="Identifies the performance as on The Late Show with Stephen Colbert (if the show is mentioned).",
            parent=node,
            critical=False
        )

    # Performance type (stand-up)
    if ex.performance.performance_type and ex.performance.performance_type.strip():
        leaf_type = evaluator.add_leaf(
            id="Context_Standup_Comedy",
            desc="Identifies the performance as stand-up comedy (if the performance type is mentioned).",
            parent=node,
            critical=False
        )
        perf_type_txt = ex.performance.performance_type or ""
        claim_type = f"The answer identifies the performance type as stand-up comedy (or an equivalent), as indicated by '{perf_type_txt}'."
        await evaluator.verify(
            claim=claim_type,
            node=leaf_type,
            additional_instruction="Judge based on the answer text. Accept synonyms like 'stand-up', 'standup', 'stand-up comedy'."
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="Context_Standup_Comedy",
            desc="Identifies the performance as stand-up comedy (if the performance type is mentioned).",
            parent=node,
            critical=False
        )

    # Date (Feb 27, 2026) if stated
    if ex.performance.performance_date and ex.performance.performance_date.strip():
        leaf_date = evaluator.add_leaf(
            id="Context_Date_Matches_Feb_27_2026_If_Stated",
            desc="If a performance date is stated, it matches February 27, 2026.",
            parent=node,
            critical=False
        )
        date_txt = ex.performance.performance_date
        claim_date = (
            f"If a date is mentioned in the answer ('{date_txt}'), it matches February 27, 2026 "
            f"(allowing minor format variants like 'Feb 27, 2026', '27 Feb 2026', etc.)."
        )
        await evaluator.verify(
            claim=claim_date,
            node=leaf_date,
            additional_instruction="If the date in the answer corresponds to February 27, 2026 (in any common format), mark Correct; otherwise Incorrect."
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="Context_Date_Matches_Feb_27_2026_If_Stated",
            desc="If a performance date is stated, it matches February 27, 2026.",
            parent=node,
            critical=False
        )


async def build_performer_identity_checks(evaluator: Evaluator, parent, ex: AnswerExtraction) -> None:
    """
    Non-critical performer identity consistency checks.
    If the performer is not named in the answer, these checks pass by default (as 'not applicable').
    """
    node = evaluator.add_parallel(
        id="Performer_Identity_Consistency",
        desc="If the response names the performer, that identity is consistent with the constrained performer description.",
        parent=parent,
        critical=False
    )

    performer_named = ex.performer.name is not None and ex.performer.name.strip() != ""
    perf_name = ex.performer.name or ""
    perf_sources = ex.performer.sources or []

    # Survivor Season 37 (DvG)
    if performer_named:
        leaf_s37 = evaluator.add_leaf(
            id="Performer_Survivor_Season_37_DvG",
            desc="If performer is named, they are a contestant from Survivor Season 37 (David vs. Goliath).",
            parent=node,
            critical=False
        )
        claim_s37 = f"{perf_name} was a contestant on Survivor Season 37 (David vs. Goliath)."
        sources = perf_sources
        await evaluator.verify(
            claim=claim_s37,
            node=leaf_s37,
            sources=sources if sources else None,
            additional_instruction="If sources are provided, verify against them. Allow reasonable name variants. If no sources, judge based on the answer and common-sense name equivalence only."
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="Performer_Survivor_Season_37_DvG",
            desc="If performer is named, they are a contestant from Survivor Season 37 (David vs. Goliath).",
            parent=node,
            critical=False
        )

    # Survivor Season 50
    if performer_named:
        leaf_s50 = evaluator.add_leaf(
            id="Performer_Survivor_Season_50",
            desc="If performer is named, they also appeared on Survivor Season 50.",
            parent=node,
            critical=False
        )
        claim_s50 = f"{perf_name} appeared on Survivor Season 50."
        await evaluator.verify(
            claim=claim_s50,
            node=leaf_s50,
            sources=perf_sources if perf_sources else None,
            additional_instruction="If sources are provided, verify against them. Allow reasonable name variants. If no sources, judge based on the answer and common-sense name equivalence only."
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="Performer_Survivor_Season_50",
            desc="If performer is named, they also appeared on Survivor Season 50.",
            parent=node,
            critical=False
        )

    # Robotics professor at FAMU-FSU College of Engineering
    if performer_named:
        leaf_prof = evaluator.add_leaf(
            id="Performer_Robotics_Professor_FAMU_FSU",
            desc="If performer is named, they are a robotics professor at a Florida university, specifically FAMU-FSU College of Engineering.",
            parent=node,
            critical=False
        )
        claim_prof = f"{perf_name} is a robotics professor at the FAMU-FSU College of Engineering in Florida."
        await evaluator.verify(
            claim=claim_prof,
            node=leaf_prof,
            sources=perf_sources if perf_sources else None,
            additional_instruction="If sources are provided (e.g., university profile), verify against them. Accept minor title/name variants."
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="Performer_Robotics_Professor_FAMU_FSU",
            desc="If performer is named, they are a robotics professor at a Florida university, specifically FAMU-FSU College of Engineering.",
            parent=node,
            critical=False
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
    Evaluate an answer for the Colbert/Survivor/venue-capacity task and return a structured result dictionary.
    """
    # Initialize evaluator (root kept non-critical to allow partial credit for non-critical groups)
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

    # Record expected context as ground truth info (for transparency only; not used for hard comparisons)
    evaluator.add_ground_truth({
        "expected": EXPECTED_CONTEXT
    })

    # Extraction
    extraction = await evaluator.extract(
        prompt=prompt_extract_answer(),
        template_class=AnswerExtraction,
        extraction_name="answer_extraction"
    )

    # Build checks
    await build_venue_and_capacity_checks(evaluator, root, extraction)
    await build_performance_context_checks(evaluator, root, extraction)
    await build_performer_identity_checks(evaluator, root, extraction)

    # Return final structured summary
    return evaluator.get_summary()