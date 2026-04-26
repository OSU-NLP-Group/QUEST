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
TASK_ID = "gnb_2025_takeover_eval"
TASK_DESCRIPTION = (
    "Following the recent military takeover in Guinea-Bissau in late November 2025, compile a factual summary that includes: "
    "(1) the specific date when the military announced taking control, "
    "(2) the official name of the military governing body that was established, "
    "(3) the duration of the transition period announced by the military, "
    "(4) the date when the African Union suspended Guinea-Bissau from its activities, and "
    "(5) the specific policy principle cited by the African Union in its suspension resolution. "
    "Provide at least one credible news source URL to support your information."
)

# Expected canonical facts from rubric (used to check answer correctness)
EXPECTED_TAKEOVER_DATE = "November 26, 2025"
EXPECTED_MILITARY_BODY = "High Military Command for the Restoration of Order"
EXPECTED_TRANSITION_PERIOD = "one year"
EXPECTED_AU_SUSPENSION_DATE = "November 29, 2025"
EXPECTED_AU_POLICY = "zero tolerance on unconstitutional changes of government"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CoupInfoExtraction(BaseModel):
    takeover_date: Optional[str] = None
    military_body_name: Optional[str] = None
    transition_period: Optional[str] = None
    au_suspension_date: Optional[str] = None
    au_policy_principle: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_coup_info() -> str:
    return """
    Extract the following fields exactly as they are stated in the provided answer text. Do not invent or infer.

    Fields:
    - takeover_date: The specific calendar date when the military announced taking control.
    - military_body_name: The official name of the military governing body established (as written in the answer; preserve capitalization/punctuation).
    - transition_period: The duration of the transition period announced by the military (e.g., "one year", "12 months").
    - au_suspension_date: The specific calendar date when the African Union suspended Guinea-Bissau from its activities.
    - au_policy_principle: The specific policy principle cited by the African Union in its suspension resolution (e.g., phrase like "zero tolerance on unconstitutional changes of government").
    - sources: A list of all URLs explicitly shown in the answer that are used as references. Include only valid URLs (plain or markdown links). 
               If a URL is missing a protocol, prepend http://. Do not include duplicate URLs.

    If any field is missing in the answer, set it to null. If no sources are present, return an empty array for sources.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def limit_urls(urls: List[str], max_urls: int = 10) -> List[str]:
    return urls[:max_urls]


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def build_source_reference_checks(
    evaluator: Evaluator,
    parent_node,
    urls: List[str],
) -> None:
    """
    Build the critical 'Source_References' checks:
      - existence of at least one source URL
      - credibility of at least one URL as a news article about the event
    """
    source_refs_main = evaluator.add_sequential(
        id="Source_References",
        desc="Provide at least one credible news source URL supporting the information",
        parent=parent_node,
        critical=True
    )

    # Existence check (critical)
    exists_node = evaluator.add_custom_node(
        result=bool(urls),
        id="source_exists",
        desc="At least one source URL is provided in the answer",
        parent=source_refs_main,
        critical=True
    )

    # Credibility/relevance check (critical)
    credible_node = evaluator.add_leaf(
        id="source_credible",
        desc="At least one provided URL is a credible news article reporting on the event",
        parent=source_refs_main,
        critical=True
    )

    credible_claim = (
        "This webpage is a news article from a credible news organization (recognized media outlet) "
        "reporting on the Guinea-Bissau military takeover and/or the African Union's response in late November 2025."
    )

    add_ins = (
        "Assess credibility based on the page content and publisher identity (e.g., major international/regional news outlets). "
        "If the URL points to a personal blog, content farm, user-generated platform, aggregator page, generic home page, or a non-news organization page, mark as not credible. "
        "If no URLs are provided, respond Incorrect."
    )

    # If URLs empty, the Verify will use simple mode; the instruction forces Incorrect
    await evaluator.verify(
        claim=credible_claim,
        node=credible_node,
        sources=urls if urls else None,
        additional_instruction=add_ins
    )


async def build_takeover_date_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: CoupInfoExtraction,
    urls: List[str],
) -> None:
    node = evaluator.add_sequential(
        id="Takeover_Date",
        desc="The date when the military announced taking control (should be November 26, 2025)",
        parent=parent_node,
        critical=True
    )

    # Presence
    evaluator.add_custom_node(
        result=bool(extracted.takeover_date and extracted.takeover_date.strip()),
        id="Takeover_Date_present",
        desc="The answer includes the takeover announcement date",
        parent=node,
        critical=True
    )

    # Match expected in the answer (simple verify)
    leaf_match = evaluator.add_leaf(
        id="Takeover_Date_match_expected",
        desc=f"The answer states the announcement date is {EXPECTED_TAKEOVER_DATE}",
        parent=node,
        critical=True
    )
    claim_match = f"The answer states that the date when the military announced taking control is {EXPECTED_TAKEOVER_DATE}."
    await evaluator.verify(
        claim=claim_match,
        node=leaf_match,
        additional_instruction=(
            "Look only at the answer text to see if it explicitly states this date (allow variants like '26 November 2025', "
            "'Nov. 26, 2025', or day/month order differences). If the answer states a different date or omits it, return Incorrect."
        )
    )

    # Supported by sources (URL verify)
    leaf_support = evaluator.add_leaf(
        id="Takeover_Date_supported",
        desc=f"The November 26, 2025 takeover announcement date is supported by cited sources",
        parent=node,
        critical=True
    )
    claim_support = "The military announced taking control on November 26, 2025."
    await evaluator.verify(
        claim=claim_support,
        node=leaf_support,
        sources=urls if urls else None,
        additional_instruction=(
            "Verify the date using the provided webpage(s). Accept minor date format variants (e.g., '26 November 2025'). "
            "If URLs are missing or do not explicitly support this date, respond Incorrect."
        )
    )


async def build_military_body_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: CoupInfoExtraction,
    urls: List[str],
) -> None:
    node = evaluator.add_sequential(
        id="Military_Body_Name",
        desc="The official military governing body name is correctly provided",
        parent=parent_node,
        critical=True
    )

    # Presence
    evaluator.add_custom_node(
        result=bool(extracted.military_body_name and extracted.military_body_name.strip()),
        id="Military_Body_Name_present",
        desc="The answer includes the official military governing body name",
        parent=node,
        critical=True
    )

    # Match expected in the answer
    leaf_match = evaluator.add_leaf(
        id="Military_Body_Name_match_expected",
        desc=f"The answer states the body is '{EXPECTED_MILITARY_BODY}'",
        parent=node,
        critical=True
    )
    claim_match = f"The answer states that the established military governing body is '{EXPECTED_MILITARY_BODY}'."
    await evaluator.verify(
        claim=claim_match,
        node=leaf_match,
        additional_instruction=(
            "Check only the answer text. Allow capitalization differences and minor punctuation variants. "
            "If the answer uses an equivalent translation or close paraphrase (e.g., 'High Military Command for Restoring Order'), "
            "consider it correct. If it names a different body or omits it, respond Incorrect."
        )
    )

    # Supported by sources
    leaf_support = evaluator.add_leaf(
        id="Military_Body_Name_supported",
        desc="The body name is supported by the cited sources",
        parent=node,
        critical=True
    )
    claim_support = "The official military governing body established was named 'High Military Command for the Restoration of Order'."
    await evaluator.verify(
        claim=claim_support,
        node=leaf_support,
        sources=urls if urls else None,
        additional_instruction=(
            "Verify the body's official name on the provided webpage(s). Allow capitalization differences and close paraphrases. "
            "If URLs are missing or the pages do not support this name, respond Incorrect."
        )
    )


async def build_transition_period_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: CoupInfoExtraction,
    urls: List[str],
) -> None:
    node = evaluator.add_sequential(
        id="Transition_Period",
        desc="The duration of the announced transition period is correctly provided",
        parent=parent_node,
        critical=True
    )

    # Presence
    evaluator.add_custom_node(
        result=bool(extracted.transition_period and extracted.transition_period.strip()),
        id="Transition_Period_present",
        desc="The answer includes the announced transition period",
        parent=node,
        critical=True
    )

    # Match expected in the answer
    leaf_match = evaluator.add_leaf(
        id="Transition_Period_match_expected",
        desc=f"The answer states the transition period is {EXPECTED_TRANSITION_PERIOD}",
        parent=node,
        critical=True
    )
    claim_match = f"The answer states that the transition period announced by the military is {EXPECTED_TRANSITION_PERIOD}."
    await evaluator.verify(
        claim=claim_match,
        node=leaf_match,
        additional_instruction=(
            "Check only the answer text. Consider 'one year' equivalent to 'a year' or '12 months'. "
            "If the answer cites a different duration or omits it, respond Incorrect."
        )
    )

    # Supported by sources
    leaf_support = evaluator.add_leaf(
        id="Transition_Period_supported",
        desc="The one-year transition period is supported by cited sources",
        parent=node,
        critical=True
    )
    claim_support = "The military announced a one-year transition period."
    await evaluator.verify(
        claim=claim_support,
        node=leaf_support,
        sources=urls if urls else None,
        additional_instruction=(
            "Verify using the provided webpage(s). Accept 'a year' or '12 months' as equivalent. "
            "If URLs are missing or the pages do not support this duration, respond Incorrect."
        )
    )


async def build_au_suspension_date_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: CoupInfoExtraction,
    urls: List[str],
) -> None:
    node = evaluator.add_sequential(
        id="AU_Suspension_Date",
        desc="The date when the African Union suspended Guinea-Bissau is correctly provided",
        parent=parent_node,
        critical=True
    )

    # Presence
    evaluator.add_custom_node(
        result=bool(extracted.au_suspension_date and extracted.au_suspension_date.strip()),
        id="AU_Suspension_Date_present",
        desc="The answer includes the AU suspension date",
        parent=node,
        critical=True
    )

    # Match expected in the answer
    leaf_match = evaluator.add_leaf(
        id="AU_Suspension_Date_match_expected",
        desc=f"The answer states the AU suspension date is {EXPECTED_AU_SUSPENSION_DATE}",
        parent=node,
        critical=True
    )
    claim_match = f"The answer states that the African Union suspended Guinea-Bissau on {EXPECTED_AU_SUSPENSION_DATE}."
    await evaluator.verify(
        claim=claim_match,
        node=leaf_match,
        additional_instruction=(
            "Check only the answer text. Allow variants like '29 November 2025' or 'Nov. 29, 2025'. "
            "If the answer cites a different date or omits it, respond Incorrect."
        )
    )

    # Supported by sources
    leaf_support = evaluator.add_leaf(
        id="AU_Suspension_Date_supported",
        desc="The AU suspension date is supported by cited sources",
        parent=node,
        critical=True
    )
    claim_support = "The African Union suspended Guinea-Bissau on November 29, 2025."
    await evaluator.verify(
        claim=claim_support,
        node=leaf_support,
        sources=urls if urls else None,
        additional_instruction=(
            "Verify the suspension date using the provided webpage(s). Accept minor date format variants. "
            "If URLs are missing or do not clearly support this date, respond Incorrect."
        )
    )


async def build_au_policy_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: CoupInfoExtraction,
    urls: List[str],
) -> None:
    node = evaluator.add_sequential(
        id="AU_Policy_Statement",
        desc="The AU policy principle cited in the suspension resolution is correctly provided",
        parent=parent_node,
        critical=True
    )

    # Presence
    evaluator.add_custom_node(
        result=bool(extracted.au_policy_principle and extracted.au_policy_principle.strip()),
        id="AU_Policy_Statement_present",
        desc="The answer includes the AU policy principle cited",
        parent=node,
        critical=True
    )

    # Match expected in the answer
    leaf_match = evaluator.add_leaf(
        id="AU_Policy_Statement_match_expected",
        desc=f"The answer cites the AU policy principle '{EXPECTED_AU_POLICY}'",
        parent=node,
        critical=True
    )
    claim_match = f"The answer states that the AU cited the policy principle of '{EXPECTED_AU_POLICY}'."
    await evaluator.verify(
        claim=claim_match,
        node=leaf_match,
        additional_instruction=(
            "Check only the answer text. Accept close paraphrases and the variant 'zero tolerance for unconstitutional changes of government'. "
            "If the answer cites a different principle or omits it, respond Incorrect."
        )
    )

    # Supported by sources
    leaf_support = evaluator.add_leaf(
        id="AU_Policy_Statement_supported",
        desc="The cited AU policy principle is supported by the sources",
        parent=node,
        critical=True
    )
    claim_support = "In its suspension resolution, the African Union cited the policy principle of zero tolerance on unconstitutional changes of government."
    await evaluator.verify(
        claim=claim_support,
        node=leaf_support,
        sources=urls if urls else None,
        additional_instruction=(
            "Verify this policy principle on the provided webpage(s). Accept the preposition 'for' as equivalent to 'on'. "
            "If URLs are missing or the pages do not support this, respond Incorrect."
        )
    )


# --------------------------------------------------------------------------- #
# Orchestrators for rubric subtrees                                           #
# --------------------------------------------------------------------------- #
async def verify_military_takeover_details(
    evaluator: Evaluator,
    parent_node,
    extracted: CoupInfoExtraction,
    urls: List[str],
) -> None:
    military_node = evaluator.add_parallel(
        id="Military_Takeover_Details",
        desc="Core facts about the military takeover event",
        parent=parent_node,
        critical=False
    )

    await build_takeover_date_checks(evaluator, military_node, extracted, urls)
    await build_military_body_checks(evaluator, military_node, extracted, urls)
    await build_transition_period_checks(evaluator, military_node, extracted, urls)


async def verify_international_response(
    evaluator: Evaluator,
    parent_node,
    extracted: CoupInfoExtraction,
    urls: List[str],
) -> None:
    intl_node = evaluator.add_parallel(
        id="International_Response",
        desc="Actions taken by international organizations",
        parent=parent_node,
        critical=False
    )

    await build_au_suspension_date_checks(evaluator, intl_node, extracted, urls)
    await build_au_policy_checks(evaluator, intl_node, extracted, urls)


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
    Evaluate an answer for the Guinea-Bissau 2025 takeover factual summary task.
    """
    # Initialize evaluator (root = parallel as per rubric)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Verify all required information about the Guinea-Bissau military takeover and international response",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract structured fields from the answer
    extracted: CoupInfoExtraction = await evaluator.extract(
        prompt=prompt_extract_coup_info(),
        template_class=CoupInfoExtraction,
        extraction_name="extracted_coup_info"
    )

    # Clean and limit URLs
    urls = limit_urls(dedupe_preserve_order(extracted.sources or []), max_urls=10)

    # Record ground truth for transparency
    evaluator.add_ground_truth({
        "expected": {
            "takeover_date": EXPECTED_TAKEOVER_DATE,
            "military_body_name": EXPECTED_MILITARY_BODY,
            "transition_period": EXPECTED_TRANSITION_PERIOD,
            "au_suspension_date": EXPECTED_AU_SUSPENSION_DATE,
            "au_policy_principle": EXPECTED_AU_POLICY
        }
    }, gt_type="expected_facts")

    # Build verification subtrees
    await verify_military_takeover_details(evaluator, root, extracted, urls)
    await verify_international_response(evaluator, root, extracted, urls)
    await build_source_reference_checks(evaluator, root, urls)

    # Return structured summary
    return evaluator.get_summary()