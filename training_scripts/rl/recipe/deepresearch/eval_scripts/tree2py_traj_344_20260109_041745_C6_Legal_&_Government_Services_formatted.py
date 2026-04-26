import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "nov_2024_lrca"
TASK_DESCRIPTION = (
    "Identify three legislatively-referred constitutional amendments (LRCA) that appeared on state ballots in the "
    "November 2024 election and were approved by voters with at least 70% of the vote. For each amendment, provide:\n"
    "1. The state where the amendment appeared\n"
    "2. The official amendment number or designation (e.g., \"Amendment 1\", \"Proposition X\")\n"
    "3. A brief description of what the amendment does\n"
    "4. The exact percentage of voter approval the amendment received\n"
    "5. A link to the official election results page showing the amendment's approval\n"
    "6. The policy area the amendment addresses (must be one of: veterans' benefits, voting/citizenship requirements, or government structure/operations)\n"
    "7. Confirmation that it was legislatively-referred (LRCA) rather than citizen-initiated (CICA)\n"
    "8. A link to a page describing the state's constitutional amendment process or the specific legislative action that placed this amendment on the ballot\n"
    "All three amendments must be from different states and must address different specific topics within their policy areas."
)

ALLOWED_POLICY_AREAS = [
    "veterans' benefits",
    "voting/citizenship requirements",
    "government structure/operations",
]


class AmendmentItem(BaseModel):
    state: Optional[str] = None
    designation: Optional[str] = None
    description: Optional[str] = None
    approval_percentage: Optional[str] = None
    official_results_url: Optional[str] = None
    policy_area: Optional[str] = None
    lrca_confirmed: Optional[bool] = None
    legislative_process_url: Optional[str] = None
    ballot_month_year: Optional[str] = None
    topic_detail: Optional[str] = None


class AmendmentsExtraction(BaseModel):
    amendments: List[AmendmentItem] = Field(default_factory=list)


def prompt_extract_amendments() -> str:
    return (
        "Extract up to the first three constitutional amendments (LRCA) that the answer claims meet the criteria. "
        "For each amendment mentioned, return an object with these fields:\n"
        "- state: The U.S. state where the amendment appeared.\n"
        "- designation: The official amendment number or designation (e.g., \"Amendment 1\", \"Proposition X\").\n"
        "- description: A brief description of what the amendment does, as stated in the answer.\n"
        "- approval_percentage: The exact percentage of voter approval stated in the answer (e.g., \"72.5%\", \"70%\"). Use the exact text from the answer.\n"
        "- official_results_url: A URL to the official election results page that shows the amendment's approval.\n"
        "- policy_area: The policy area classification claimed in the answer. Must be one of: "
        f"{', '.join(ALLOWED_POLICY_AREAS)}. Use the exact category term if present.\n"
        "- lrca_confirmed: Return true only if the answer explicitly states or implies the measure was legislatively-referred (LRCA), "
        "false if the answer implies it was citizen-initiated (CICA) or not LRCA, and null if not stated.\n"
        "- legislative_process_url: A URL that documents the state's constitutional amendment process or the legislative action placing the amendment on the ballot.\n"
        "- ballot_month_year: If the answer mentions the ballot timing, extract the month and year (e.g., \"November 2024\"). Otherwise, null.\n"
        "- topic_detail: A short phrase capturing the specific topic within the policy area (e.g., \"noncitizen voting ban\", \"disabled veterans property tax exemption\"). "
        "If not explicitly stated, infer a concise topic phrase from the description.\n\n"
        "Return a JSON object with a single field 'amendments' which is an array of up to three such objects. "
        "If any field is missing from the answer for an amendment, set it to null. Extract only URLs explicitly present in the answer."
    )


def parse_percentage(perc_text: Optional[str]) -> Optional[float]:
    if not perc_text:
        return None
    m = re.search(r'(\d{1,3}(?:\.\d+)?)\s*%', perc_text)
    if m:
        try:
            v = float(m.group(1))
            if 0 <= v <= 100:
                return v
        except Exception:
            return None
    m2 = re.search(r'(\d{1,3}(?:\.\d+)?)', perc_text)
    if m2:
        try:
            v = float(m2.group(1))
            if 0 <= v <= 100:
                return v
        except Exception:
            return None
    return None


def _safe_urls(*urls: Optional[str]) -> List[str]:
    return [u for u in urls if u and isinstance(u, str) and u.strip()]


async def verify_amendment(
    evaluator: Evaluator,
    parent_node,
    item: AmendmentItem,
    idx: int,
) -> None:
    ordinal = ["First", "Second", "Third"]
    node = evaluator.add_parallel(
        id=f"amendment_{idx + 1}",
        desc=f"{ordinal[idx]} constitutional amendment meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    # Existence checks for URLs (critical)
    results_url_exists = evaluator.add_custom_node(
        result=bool(item.official_results_url and item.official_results_url.strip()),
        id=f"amendment_{idx + 1}_official_results_url",
        desc="A URL to the official election results page showing the amendment's approval is provided",
        parent=node,
        critical=True
    )
    legis_url_exists = evaluator.add_custom_node(
        result=bool(item.legislative_process_url and item.legislative_process_url.strip()),
        id=f"amendment_{idx + 1}_legislative_process_url",
        desc="A URL documenting the state's constitutional amendment process or the legislative action placing the amendment on the ballot is provided",
        parent=node,
        critical=True
    )

    # State Identification
    state_leaf = evaluator.add_leaf(
        id=f"amendment_{idx + 1}_state_identification",
        desc="The state where the amendment appeared is correctly identified",
        parent=node,
        critical=True
    )
    state_claim = f"The amendment with designation '{item.designation}' appeared on the statewide ballot in {item.state}."
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=item.official_results_url,
        additional_instruction="Verify that the official election results page corresponds to the specified state and this amendment.",
        extra_prerequisites=[results_url_exists]
    )

    # Official Amendment Designation
    designation_leaf = evaluator.add_leaf(
        id=f"amendment_{idx + 1}_official_amendment_designation",
        desc="The official amendment number or designation is correctly provided",
        parent=node,
        critical=True
    )
    designation_claim = f"The official designation of this measure is '{item.designation}'."
    await evaluator.verify(
        claim=designation_claim,
        node=designation_leaf,
        sources=item.official_results_url,
        additional_instruction="Check the measure number/name on the official results page. Allow minor formatting variations.",
        extra_prerequisites=[results_url_exists]
    )

    # Amendment Type: LRCA (not CICA)
    lrca_leaf = evaluator.add_leaf(
        id=f"amendment_{idx + 1}_amendment_type_lrca",
        desc="The amendment is confirmed to be legislatively-referred (LRCA), not citizen-initiated (CICA)",
        parent=node,
        critical=True
    )
    lrca_claim = "This measure is a legislatively-referred constitutional amendment (LRCA), not a citizen-initiated measure (CICA)."
    await evaluator.verify(
        claim=lrca_claim,
        node=lrca_leaf,
        sources=_safe_urls(item.official_results_url, item.legislative_process_url),
        additional_instruction="Confirm any wording such as 'legislatively referred', 'referred to the people by the legislature', or 'legislative referendum'. If it is citizen-initiated/initiative/referendum by voters, it's not LRCA.",
        extra_prerequisites=[results_url_exists, legis_url_exists]
    )

    # Policy Area Classification
    policy_leaf = evaluator.add_leaf(
        id=f"amendment_{idx + 1}_policy_area_classification",
        desc="The amendment is correctly classified as veterans' benefits, voting/citizenship requirements, or government structure/operations",
        parent=node,
        critical=True
    )
    policy_claim = f"The amendment is correctly classified under the policy area '{item.policy_area}'."
    await evaluator.verify(
        claim=policy_claim,
        node=policy_leaf,
        sources=_safe_urls(item.official_results_url, item.legislative_process_url),
        additional_instruction=f"Use the measure text/summary to judge classification. The allowed categories are: {', '.join(ALLOWED_POLICY_AREAS)}. Mark incorrect if classification is outside these or mismatched.",
        extra_prerequisites=[results_url_exists]
    )

    # November 2024 Ballot
    nov_leaf = evaluator.add_leaf(
        id=f"amendment_{idx + 1}_november_2024_ballot",
        desc="The amendment is confirmed to have appeared on the November 2024 statewide ballot",
        parent=node,
        critical=True
    )
    nov_claim = f"This amendment appeared on the November 2024 statewide ballot in {item.state}."
    await evaluator.verify(
        claim=nov_claim,
        node=nov_leaf,
        sources=item.official_results_url,
        additional_instruction="Check the election date on the official page (e.g., 'General Election November 2024', 'November 5, 2024').",
        extra_prerequisites=[results_url_exists]
    )

    # Voter Approval Requirements (parallel group)
    approval_group = evaluator.add_parallel(
        id=f"amendment_{idx + 1}_voter_approval_requirements",
        desc="Verification of voter approval and meeting the 70% threshold",
        parent=node,
        critical=True
    )

    # Approved by voters
    approved_leaf = evaluator.add_leaf(
        id=f"amendment_{idx + 1}_approved_by_voters",
        desc="The amendment was approved by voters",
        parent=approval_group,
        critical=True
    )
    approved_claim = "The 'Yes' or 'Approve' vote prevailed for this amendment."
    await evaluator.verify(
        claim=approved_claim,
        node=approved_leaf,
        sources=item.official_results_url,
        additional_instruction="Verify approval on the official results page. If the measure failed or results are absent, mark incorrect.",
        extra_prerequisites=[results_url_exists]
    )

    # Approval percentage provided
    approval_pct_exists = evaluator.add_custom_node(
        result=bool(item.approval_percentage and item.approval_percentage.strip()),
        id=f"amendment_{idx + 1}_approval_percentage_provided",
        desc="The exact voter approval percentage is provided",
        parent=approval_group,
        critical=True
    )

    # 70% threshold
    pct_value = parse_percentage(item.approval_percentage)
    seventy_node = evaluator.add_custom_node(
        result=(pct_value is not None and pct_value >= 70.0),
        id=f"amendment_{idx + 1}_seventy_percent_threshold",
        desc="The approval percentage is at least 70%",
        parent=approval_group,
        critical=True
    )

    # Amendment description accuracy
    desc_leaf = evaluator.add_leaf(
        id=f"amendment_{idx + 1}_amendment_description",
        desc="A description of what the amendment does is provided",
        parent=node,
        critical=True
    )
    desc_claim = f"The measure can be accurately described as: \"{item.description}\"."
    await evaluator.verify(
        claim=desc_claim,
        node=desc_leaf,
        sources=_safe_urls(item.official_results_url, item.legislative_process_url),
        additional_instruction="Judge whether the provided description captures the core substance of the measure per official sources. Allow concise paraphrases.",
        extra_prerequisites=[results_url_exists]
    )


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

    extracted = await evaluator.extract(
        prompt=prompt_extract_amendments(),
        template_class=AmendmentsExtraction,
        extraction_name="amendments_extraction"
    )

    # Keep exactly three items: first 3 or pad with empty ones
    items: List[AmendmentItem] = list(extracted.amendments[:3])
    while len(items) < 3:
        items.append(AmendmentItem())

    # Verify each amendment block
    for i, item in enumerate(items):
        await verify_amendment(evaluator, root, item, i)

    # Global requirement: Different states
    states = [itm.state.strip() for itm in items if itm.state and itm.state.strip()]
    unique_states = set(s.lower() for s in states)
    diff_states_node = evaluator.add_custom_node(
        result=(len(states) == 3 and len(unique_states) == 3),
        id="different_states_requirement",
        desc="All three amendments are from different states",
        parent=root,
        critical=True
    )

    # Global requirement: Different topics within policy areas
    topics = []
    for itm in items:
        if itm.topic_detail and itm.topic_detail.strip():
            topics.append(itm.topic_detail.strip().lower())
        elif itm.description and itm.description.strip():
            topics.append(itm.description.strip().lower())
        else:
            topics.append("")

    topics_claim = (
        f"The three amendments address different specific topics within their policy areas:\n"
        f"1) {items[0].state} {items[0].designation}: {items[0].topic_detail or items[0].description}\n"
        f"2) {items[1].state} {items[1].designation}: {items[1].topic_detail or items[1].description}\n"
        f"3) {items[2].state} {items[2].designation}: {items[2].topic_detail or items[2].description}\n"
        "No two of these measures target the same specific topic (e.g., multiple noncitizen voting bans would violate this)."
    )
    diff_topics_leaf = evaluator.add_leaf(
        id="different_topics_requirement",
        desc="All three amendments address different specific topics within their policy areas (e.g., not all about noncitizen voting bans)",
        parent=root,
        critical=True
    )
    await evaluator.verify(
        claim=topics_claim,
        node=diff_topics_leaf,
        sources=None,
        additional_instruction="Rely on the provided descriptions/topics in the answer. Determine if any two are substantively about the same specific topic. If yes, mark incorrect."
    )

    evaluator.add_custom_info(
        info={"allowed_policy_areas": ALLOWED_POLICY_AREAS},
        info_type="allowed_policy_areas",
        info_name="allowed_policy_areas"
    )

    return evaluator.get_summary()