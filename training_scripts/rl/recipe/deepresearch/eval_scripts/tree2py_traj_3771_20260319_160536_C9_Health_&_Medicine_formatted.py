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
TASK_ID = "healthcare_events_2025_analysis"
TASK_DESCRIPTION = """
Identify and provide detailed information about five significant healthcare-related events, policy changes, and medical developments from 2025, following these specific requirements:

Event 1 - FDA Vaccine Policy Leadership: Identify the FDA official who released a controversial vaccine policy memo on November 28 or 29, 2025. Provide: (a) the official's name, (b) their position (which must be director of CBER), (c) the key claim in the memo (regarding at least 10 children's deaths from COVID-19 vaccines), (d) their planned departure timing (end of April 2026), (e) the policy changes proposed (stricter vaccine approval processes), and (f) a supporting URL.

Event 2 - CMS Hospital Star Rating Policy: Describe the CMS policy change affecting hospital star ratings based on safety performance. Provide: (a) the year the policy became effective (2026 Star Ratings), (b) which performance quartile is affected (lowest quartile of Safety of Care), (c) the star rating cap imposed (maximum 4 stars), (d) the minimum measures requirement (at least three safety measures), (e) what this prevents (achieving 5-star ratings), (f) the 2027 policy change (automatic 1-star reduction), and (g) a supporting URL.

Event 3 - Purdue Pharma Opioid Settlement: Detail the Purdue Pharma and Sackler family opioid settlement finalized in 2025. Provide: (a) total settlement amount ($7.4 billion), (b) Sackler family contribution amount (up to $6.5 billion), (c) payment duration (15 years), (d) initial Sackler payment ($1.5 billion), (e) second payment ($500 million after one year), (f) third payment ($500 million after two years), (g) Purdue Pharma contribution (nearly $900 million), (h) state/territory approval (all 55 in June 2025), (i) court approval date (November 18, 2025), and (j) supporting URLs.

Event 4 - FDA Oncology Drug Approvals: Provide statistics on FDA oncology drug approvals in 2025. Include: (a) total novel drugs approved in 2025 (46), (b) oncology approval count (16), (c) percentage oncology represents (30%), (d) oncology approvals in the last six weeks of 2025 (16), (e) specific KRAS-targeted drug combination approved (sotorasib with panitumumab), (f) that drug's approval date (January 16, 2025), (g) its indication (KRAS G12C-mutated colorectal cancer), and (h) supporting URLs.

Event 5 - Medical Device Recall: Detail the Fresenius Kabi infusion pump recall. Provide: (a) device name (Ivenix Large Volume Pump), (b) affected software version (5.10.1 and earlier), (c) product code (LVP-SW-0005), (d) recall classification (Class I), (e) injury/death count as of November 18, 2025 (2 serious injuries, 0 deaths), (f) initial notification date (November 14, 2025), (g) FDA software correction announcement date (February 25, 2026), and (h) a supporting URL.

For each event, all specified details must be provided with appropriate reference URLs to support the information.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class Event1Info(BaseModel):
    official_name: Optional[str] = None
    position: Optional[str] = None
    memo_release_date: Optional[str] = None
    key_claim: Optional[str] = None
    departure_timing: Optional[str] = None
    policy_changes: Optional[str] = None
    identity_urls: List[str] = Field(default_factory=list)
    memo_urls: List[str] = Field(default_factory=list)


class Event2Info(BaseModel):
    effective_year: Optional[str] = None
    affected_quartile: Optional[str] = None
    star_rating_cap: Optional[str] = None
    min_measures_requirement: Optional[str] = None
    prevents: Optional[str] = None
    change_2027: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class Event3Info(BaseModel):
    total_settlement_amount: Optional[str] = None
    sackler_total: Optional[str] = None
    payment_duration: Optional[str] = None
    initial_payment: Optional[str] = None
    second_payment: Optional[str] = None
    third_payment: Optional[str] = None
    purdue_contribution: Optional[str] = None
    state_territory_approval: Optional[str] = None
    court_approval_date: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class Event4Info(BaseModel):
    total_novel_drugs: Optional[str] = None
    oncology_count: Optional[str] = None
    oncology_percentage: Optional[str] = None
    last_six_weeks_oncology_approvals: Optional[str] = None
    kras_drug_combo: Optional[str] = None
    kras_approval_date: Optional[str] = None
    kras_indication: Optional[str] = None
    stats_urls: List[str] = Field(default_factory=list)
    kras_urls: List[str] = Field(default_factory=list)


class Event5Info(BaseModel):
    device_name: Optional[str] = None
    affected_software_version: Optional[str] = None
    product_code: Optional[str] = None
    recall_classification: Optional[str] = None
    injury_count: Optional[str] = None
    death_count: Optional[str] = None
    adverse_events_as_of_date: Optional[str] = None
    initial_notification_date: Optional[str] = None
    fda_correction_announcement_date: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class HealthcareEvents2025Extraction(BaseModel):
    event1: Optional[Event1Info] = None
    event2: Optional[Event2Info] = None
    event3: Optional[Event3Info] = None
    event4: Optional[Event4Info] = None
    event5: Optional[Event5Info] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_healthcare_events() -> str:
    return """
Extract structured information from the answer for five events. Do not invent content; extract exactly what the answer claims. If a field is not present, set it to null or an empty array accordingly. Extract all URLs explicitly shown in the answer text (plain or markdown); if none are given for a URL field, return an empty list.

Return a JSON object with the following structure and exact field names:

{
  "event1": {
    "official_name": str | null,
    "position": str | null,
    "memo_release_date": str | null,
    "key_claim": str | null,
    "departure_timing": str | null,
    "policy_changes": str | null,
    "identity_urls": [url, ...],
    "memo_urls": [url, ...]
  },
  "event2": {
    "effective_year": str | null,
    "affected_quartile": str | null,
    "star_rating_cap": str | null,
    "min_measures_requirement": str | null,
    "prevents": str | null,
    "change_2027": str | null,
    "urls": [url, ...]
  },
  "event3": {
    "total_settlement_amount": str | null,
    "sackler_total": str | null,
    "payment_duration": str | null,
    "initial_payment": str | null,
    "second_payment": str | null,
    "third_payment": str | null,
    "purdue_contribution": str | null,
    "state_territory_approval": str | null,
    "court_approval_date": str | null,
    "urls": [url, ...]
  },
  "event4": {
    "total_novel_drugs": str | null,
    "oncology_count": str | null,
    "oncology_percentage": str | null,
    "last_six_weeks_oncology_approvals": str | null,
    "kras_drug_combo": str | null,
    "kras_approval_date": str | null,
    "kras_indication": str | null,
    "stats_urls": [url, ...],
    "kras_urls": [url, ...]
  },
  "event5": {
    "device_name": str | null,
    "affected_software_version": str | null,
    "product_code": str | null,
    "recall_classification": str | null,
    "injury_count": str | null,
    "death_count": str | null,
    "adverse_events_as_of_date": str | null,
    "initial_notification_date": str | null,
    "fda_correction_announcement_date": str | null,
    "urls": [url, ...]
  }
}

Rules:
- Use strings for numbers/dates as they appear in the answer (e.g., "46", "30%", "January 16, 2025", "$7.4 billion").
- For URLs, include only valid, explicit URLs from the answer; do not infer or create new links.
- If the answer provides multiple supporting URLs for a sub-item, include them all.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str]) -> str:
    return s or ""


def _merge_sources(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for l in lists:
        if not l:
            continue
        for u in l:
            if isinstance(u, str):
                if u not in seen and u.strip():
                    seen.add(u)
                    out.append(u.strip())
    return out


# --------------------------------------------------------------------------- #
# Verification builders per event                                             #
# --------------------------------------------------------------------------- #
async def verify_event1(evaluator: Evaluator, parent, ex: Optional[Event1Info]) -> None:
    node_event1 = evaluator.add_parallel(
        id="event1_fda_vaccine_policy_official",
        desc="Information about the FDA official who issued a controversial vaccine policy memo in late 2025",
        parent=parent,
        critical=False
    )

    official_name = _safe(ex.official_name) if ex else ""
    position = _safe(ex.position) if ex else ""
    memo_release_date = _safe(ex.memo_release_date) if ex else ""
    key_claim = _safe(ex.key_claim) if ex else ""
    departure_timing = _safe(ex.departure_timing) if ex else ""
    policy_changes = _safe(ex.policy_changes) if ex else ""
    identity_urls = ex.identity_urls if ex else []
    memo_urls = ex.memo_urls if ex else []
    all_urls = _merge_sources(identity_urls, memo_urls)

    # Official Identity and Position (critical)
    group_identity = evaluator.add_parallel(
        id="event1_identity_and_position",
        desc="Verification of the official's name and organizational role",
        parent=node_event1,
        critical=True
    )

    # Identity Reference URL - existence (critical gating)
    evaluator.add_custom_node(
        result=len(identity_urls) > 0,
        id="event1_identity_reference_url",
        desc="Provide valid URL reference supporting the official's identity and position",
        parent=group_identity,
        critical=True
    )

    # Official Identity (critical)
    leaf_identity = evaluator.add_leaf(
        id="event1_official_identity",
        desc="Provide the name of the FDA official who issued this memo",
        parent=group_identity,
        critical=True
    )
    claim_identity = f"The FDA official who issued the controversial vaccine policy memo in late November 2025 was {official_name}."
    await evaluator.verify(
        claim=claim_identity,
        node=leaf_identity,
        sources=_merge_sources(memo_urls, identity_urls),
        additional_instruction="Verify that the cited page(s) explicitly name the official responsible for the memo. Allow minor name variants (middle initials, capitalization)."
    )

    # Official Position (critical)
    leaf_position = evaluator.add_leaf(
        id="event1_official_position",
        desc="The official held the position of director of CBER (Center for Biologics Evaluation and Research) at FDA",
        parent=group_identity,
        critical=True
    )
    claim_position = f"At the time of the memo, {official_name} held the position of Director of the Center for Biologics Evaluation and Research (CBER) at the U.S. FDA."
    await evaluator.verify(
        claim=claim_position,
        node=leaf_position,
        sources=identity_urls,
        additional_instruction="Confirm that the page states the person was Director of CBER at FDA at the relevant time. Accept equivalent phrasing like 'CBER director'."
    )

    # Memo Details (critical)
    group_memo = evaluator.add_parallel(
        id="event1_memo_details",
        desc="Verification of the memo's release date and content",
        parent=node_event1,
        critical=True
    )

    # Memo Reference URL - existence (critical gating)
    evaluator.add_custom_node(
        result=len(memo_urls) > 0,
        id="event1_memo_reference_url",
        desc="Provide valid URL reference supporting the memo details",
        parent=group_memo,
        critical=True
    )

    # Memo Release Date (critical)
    leaf_release = evaluator.add_leaf(
        id="event1_memo_release_date",
        desc="The vaccine policy memo was released on November 28 or 29, 2025",
        parent=group_memo,
        critical=True
    )
    claim_release = "The vaccine policy memo was released on November 28 or November 29, 2025."
    await evaluator.verify(
        claim=claim_release,
        node=leaf_release,
        sources=memo_urls,
        additional_instruction="If the page lists either November 28 or November 29, 2025 (accounting for time zones or publication timestamps), consider it correct."
    )

    # Memo Content Claim (critical)
    leaf_content_claim = evaluator.add_leaf(
        id="event1_memo_content_claim",
        desc="The memo claimed that at least 10 children died from COVID-19 vaccines",
        parent=group_memo,
        critical=True
    )
    claim_content = "The memo claimed that at least 10 children had died from COVID-19 vaccines."
    await evaluator.verify(
        claim=claim_content,
        node=leaf_content_claim,
        sources=memo_urls,
        additional_instruction="Look for explicit language asserting that 10 or more children died from COVID-19 vaccines."
    )

    # Policy Proposal (critical)
    leaf_policy = evaluator.add_leaf(
        id="event1_policy_proposal",
        desc="The memo proposed stricter vaccine approval processes",
        parent=group_memo,
        critical=True
    )
    claim_policy = "The memo proposed stricter vaccine approval processes."
    await evaluator.verify(
        claim=claim_policy,
        node=leaf_policy,
        sources=memo_urls,
        additional_instruction="Look for recommendations or proposals that tighten vaccine approval requirements (e.g., evidentiary standards, process changes)."
    )

    # Departure Timeline (critical)
    leaf_departure = evaluator.add_leaf(
        id="event1_departure_timeline",
        desc="The official planned to depart the FDA at the end of April 2026",
        parent=node_event1,
        critical=True
    )
    claim_departure = "The official planned to depart the FDA at the end of April 2026."
    await evaluator.verify(
        claim=claim_departure,
        node=leaf_departure,
        sources=all_urls,
        additional_instruction="Find a statement indicating the official planned to leave around the end of April 2026. Accept equivalent phrasing (e.g., 'by the end of April 2026')."
    )


async def verify_event2(evaluator: Evaluator, parent, ex: Optional[Event2Info]) -> None:
    node_event2 = evaluator.add_parallel(
        id="event2_cms_star_rating_policy",
        desc="Information about the CMS policy change affecting hospital star ratings based on safety performance",
        parent=parent,
        critical=False
    )

    urls = ex.urls if ex else []

    # Source Reference (critical gating at event level)
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="event2_source_reference",
        desc="Provide valid URL reference supporting the CMS policy details",
        parent=node_event2,
        critical=True
    )

    # Policy Implementation Details (critical)
    group_impl = evaluator.add_parallel(
        id="event2_policy_implementation_details",
        desc="Details about when and how the policy was implemented",
        parent=node_event2,
        critical=True
    )

    # Policy Effective Year (critical)
    leaf_effective = evaluator.add_leaf(
        id="event2_policy_effective_year",
        desc="The policy became effective with the 2026 Star Ratings",
        parent=group_impl,
        critical=True
    )
    claim_effective = f"The policy became effective with the 2026 Star Ratings."
    await evaluator.verify(
        claim=claim_effective,
        node=leaf_effective,
        sources=urls,
        additional_instruction="Confirm that CMS applies this safety cap beginning with the 2026 Hospital Star Ratings program year."
    )

    # Performance Quartile (critical)
    leaf_quartile = evaluator.add_leaf(
        id="event2_performance_quartile",
        desc="The policy affects hospitals in the lowest quartile of Safety of Care measure group performance",
        parent=group_impl,
        critical=True
    )
    claim_quartile = "The policy affects hospitals in the lowest quartile of Safety of Care measure group performance."
    await evaluator.verify(
        claim=claim_quartile,
        node=leaf_quartile,
        sources=urls,
        additional_instruction="Look for 'lowest quartile' or equivalent phrasing about Safety of Care group performance."
    )

    # Minimum Measures Requirement (critical)
    leaf_min_measures = evaluator.add_leaf(
        id="event2_minimum_measures_requirement",
        desc="The cap applies only to hospitals reporting at least three safety measures",
        parent=group_impl,
        critical=True
    )
    claim_min_measures = "The cap applies only to hospitals reporting at least three safety measures."
    await evaluator.verify(
        claim=claim_min_measures,
        node=leaf_min_measures,
        sources=urls,
        additional_instruction="Confirm that the cap conditionally applies when ≥3 Safety of Care measures are reported."
    )

    # Rating Cap Effects (critical)
    group_effects = evaluator.add_parallel(
        id="event2_rating_cap_effects",
        desc="The effects and limitations imposed by the policy",
        parent=node_event2,
        critical=True
    )

    # Star Rating Cap (critical)
    leaf_cap = evaluator.add_leaf(
        id="event2_star_rating_cap",
        desc="Affected hospitals are capped at a maximum of 4 stars",
        parent=group_effects,
        critical=True
    )
    claim_cap = "Affected hospitals are capped at a maximum of 4 stars."
    await evaluator.verify(
        claim=claim_cap,
        node=leaf_cap,
        sources=urls,
        additional_instruction="Verify explicit mention of a 4-star cap for affected hospitals."
    )

    # Five-Star Prevention (critical)
    leaf_no_five = evaluator.add_leaf(
        id="event2_five_star_prevention",
        desc="This policy prevents lowest quartile hospitals from achieving 5-star ratings",
        parent=group_effects,
        critical=True
    )
    claim_no_five = "This policy prevents lowest quartile hospitals from achieving 5-star ratings."
    await evaluator.verify(
        claim=claim_no_five,
        node=leaf_no_five,
        sources=urls,
        additional_instruction="Confirm that hospitals in the lowest Safety of Care quartile cannot receive 5 stars due to this cap."
    )

    # Future Policy Change 2027 (critical)
    leaf_2027 = evaluator.add_leaf(
        id="event2_future_policy_change_2027",
        desc="In 2027, an automatic 1-star reduction will apply to lowest quartile hospitals",
        parent=node_event2,
        critical=True
    )
    claim_2027 = "Beginning in 2027, an automatic 1-star reduction will apply to hospitals in the lowest Safety of Care quartile."
    await evaluator.verify(
        claim=claim_2027,
        node=leaf_2027,
        sources=urls,
        additional_instruction="Look for an explicit statement about a one-star decrease in the 2027 ratings for the lowest Safety of Care quartile."
    )


async def verify_event3(evaluator: Evaluator, parent, ex: Optional[Event3Info]) -> None:
    node_event3 = evaluator.add_parallel(
        id="event3_purdue_opioid_settlement",
        desc="Details of the Purdue Pharma and Sackler family opioid litigation settlement finalized in 2025",
        parent=parent,
        critical=False
    )

    urls = ex.urls if ex else []

    # Settlement Reference URL (critical gating)
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="event3_settlement_reference_url",
        desc="Provide valid URL reference supporting the settlement approval details",
        parent=node_event3,
        critical=True
    )

    # Settlement Financial Terms (critical)
    group_financial = evaluator.add_parallel(
        id="event3_settlement_financial_terms",
        desc="Financial details of the settlement including total amount and contributor breakdowns",
        parent=node_event3,
        critical=True
    )

    # Total Settlement Amount (critical)
    leaf_total = evaluator.add_leaf(
        id="event3_total_settlement_amount",
        desc="The total settlement amount is $7.4 billion",
        parent=group_financial,
        critical=True
    )
    claim_total = "The total settlement amount is $7.4 billion."
    await evaluator.verify(
        claim=claim_total,
        node=leaf_total,
        sources=urls,
        additional_instruction="Confirm that the total settlement figure cited is $7.4 billion."
    )

    # Sackler Family Contribution Details (critical)
    group_sackler = evaluator.add_parallel(
        id="event3_sackler_contribution_details",
        desc="Specific details about the Sackler family's financial contribution to the settlement",
        parent=group_financial,
        critical=True
    )

    # Sackler Contribution Terms (critical)
    group_sackler_terms = evaluator.add_parallel(
        id="event3_sackler_contribution_terms",
        desc="The amount and duration of Sackler family payments",
        parent=group_sackler,
        critical=True
    )

    # Sackler Total Amount (critical)
    leaf_sackler_total = evaluator.add_leaf(
        id="event3_sackler_total_amount",
        desc="The Sackler family will contribute up to $6.5 billion",
        parent=group_sackler_terms,
        critical=True
    )
    claim_sackler_total = "The Sackler family will contribute up to $6.5 billion."
    await evaluator.verify(
        claim=claim_sackler_total,
        node=leaf_sackler_total,
        sources=urls,
        additional_instruction="Accept 'up to $6.5 billion' or equivalent phrasing indicating a cap."
    )

    # Sackler Payment Duration (critical)
    leaf_sackler_duration = evaluator.add_leaf(
        id="event3_sackler_payment_duration",
        desc="The Sackler payments will be made over 15 years",
        parent=group_sackler_terms,
        critical=True
    )
    claim_sackler_duration = "The Sackler payments will be made over 15 years."
    await evaluator.verify(
        claim=claim_sackler_duration,
        node=leaf_sackler_duration,
        sources=urls,
        additional_instruction="Look for the payment schedule duration of 15 years."
    )

    # Sackler Payment Schedule (critical)
    group_sackler_schedule = evaluator.add_parallel(
        id="event3_sackler_payment_schedule",
        desc="The specific payment amounts and timing from the Sackler family",
        parent=group_sackler,
        critical=True
    )

    # Initial Payment (critical)
    leaf_initial = evaluator.add_leaf(
        id="event3_sackler_initial_payment",
        desc="The initial payment from the Sacklers is $1.5 billion",
        parent=group_sackler_schedule,
        critical=True
    )
    claim_initial = "The initial payment from the Sacklers is $1.5 billion."
    await evaluator.verify(
        claim=claim_initial,
        node=leaf_initial,
        sources=urls,
        additional_instruction="Confirm the first installment is $1.5 billion."
    )

    # Second Payment (critical)
    leaf_second = evaluator.add_leaf(
        id="event3_sackler_second_payment",
        desc="The second payment is $500 million after one year",
        parent=group_sackler_schedule,
        critical=True
    )
    claim_second = "The second payment is $500 million after one year."
    await evaluator.verify(
        claim=claim_second,
        node=leaf_second,
        sources=urls,
        additional_instruction="Verify that the second installment of $500 million occurs one year after the initial payment."
    )

    # Third Payment (critical)
    leaf_third = evaluator.add_leaf(
        id="event3_sackler_third_payment",
        desc="The third payment is $500 million after two years",
        parent=group_sackler_schedule,
        critical=True
    )
    claim_third = "The third payment is $500 million after two years."
    await evaluator.verify(
        claim=claim_third,
        node=leaf_third,
        sources=urls,
        additional_instruction="Verify that the third installment of $500 million occurs two years after the initial payment."
    )

    # Sackler Reference URL existence (critical)
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="event3_sackler_reference_url",
        desc="Provide valid URL reference supporting Sackler payment details",
        parent=group_sackler,
        critical=True
    )

    # Purdue Pharma Contribution (critical)
    leaf_purdue = evaluator.add_leaf(
        id="event3_purdue_contribution",
        desc="Purdue Pharma will contribute nearly $900 million to the settlement",
        parent=group_financial,
        critical=True
    )
    claim_purdue = "Purdue Pharma will contribute nearly $900 million to the settlement."
    await evaluator.verify(
        claim=claim_purdue,
        node=leaf_purdue,
        sources=urls,
        additional_instruction="Accept phrasing like 'approximately $900 million' or 'nearly $900 million' if clearly tied to Purdue Pharma's contribution."
    )

    # Settlement Approval Process (critical)
    group_approval = evaluator.add_parallel(
        id="event3_settlement_approval_process",
        desc="Details of the approval process by states, territories, and courts",
        parent=node_event3,
        critical=True
    )

    # State and Territory Approval (critical)
    leaf_states = evaluator.add_leaf(
        id="event3_state_territory_approval",
        desc="All 55 U.S. states and territories approved the settlement in June 2025",
        parent=group_approval,
        critical=True
    )
    claim_states = "All 55 U.S. states and territories approved the settlement in June 2025."
    await evaluator.verify(
        claim=claim_states,
        node=leaf_states,
        sources=urls,
        additional_instruction="Look for explicit mention that all 55 states and territories approved in June 2025."
    )

    # Court Formal Approval Date (critical)
    leaf_court = evaluator.add_leaf(
        id="event3_court_formal_approval_date",
        desc="A bankruptcy court judge formally approved the settlement on November 18, 2025",
        parent=group_approval,
        critical=True
    )
    claim_court = "A bankruptcy court judge formally approved the settlement on November 18, 2025."
    await evaluator.verify(
        claim=claim_court,
        node=leaf_court,
        sources=urls,
        additional_instruction="Confirm the formal approval date is November 18, 2025."
    )


async def verify_event4(evaluator: Evaluator, parent, ex: Optional[Event4Info]) -> None:
    node_event4 = evaluator.add_parallel(
        id="event4_fda_oncology_approvals_2025",
        desc="Statistics and specific examples of FDA oncology drug approvals during 2025",
        parent=parent,
        critical=False
    )

    stats_urls = ex.stats_urls if ex else []
    kras_urls = ex.kras_urls if ex else []

    # Reference URL existence (critical gating)
    evaluator.add_custom_node(
        result=len(stats_urls) > 0,
        id="event4_oncology_statistics_reference_url",
        desc="Provide valid URL reference supporting the 2025 oncology approval statistics",
        parent=node_event4,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(kras_urls) > 0,
        id="event4_drug_approval_reference_url",
        desc="Provide valid URL reference supporting the specific drug approval details",
        parent=node_event4,
        critical=True
    )

    # Overall Approval Statistics (critical)
    group_stats = evaluator.add_parallel(
        id="event4_overall_approval_statistics",
        desc="Summary statistics of all FDA novel drug approvals in 2025",
        parent=node_event4,
        critical=True
    )

    # Total Novel Drugs 2025 (critical)
    leaf_total = evaluator.add_leaf(
        id="event4_total_novel_drugs_2025",
        desc="The FDA approved 46 novel drugs in 2025",
        parent=group_stats,
        critical=True
    )
    claim_total = "The FDA approved 46 novel drugs in 2025."
    await evaluator.verify(
        claim=claim_total,
        node=leaf_total,
        sources=stats_urls,
        additional_instruction="Verify total count of 'novel drugs' for calendar year 2025 equals 46."
    )

    # Oncology Approval Count (critical)
    leaf_onc_count = evaluator.add_leaf(
        id="event4_oncology_approval_count",
        desc="Oncology accounted for 16 drug approvals in 2025",
        parent=group_stats,
        critical=True
    )
    claim_onc_count = "Oncology accounted for 16 drug approvals in 2025."
    await evaluator.verify(
        claim=claim_onc_count,
        node=leaf_onc_count,
        sources=stats_urls,
        additional_instruction="Confirm that 16 approvals are categorized as oncology."
    )

    # Oncology Percentage (critical)
    leaf_onc_pct = evaluator.add_leaf(
        id="event4_oncology_percentage",
        desc="Oncology approvals represented 30% of total novel drug approvals",
        parent=group_stats,
        critical=True
    )
    claim_onc_pct = "Oncology approvals represented 30% of total novel drug approvals in 2025."
    await evaluator.verify(
        claim=claim_onc_pct,
        node=leaf_onc_pct,
        sources=stats_urls,
        additional_instruction="The share should be 30% of the total novel drug approvals."
    )

    # Oncology Approval Timing (critical)
    group_timing = evaluator.add_parallel(
        id="event4_oncology_approval_timing",
        desc="Specific timing information about oncology approvals",
        parent=node_event4,
        critical=True
    )

    leaf_last6 = evaluator.add_leaf(
        id="event4_last_six_weeks_approvals",
        desc="In the last six weeks of 2025 (approximately November-December), FDA approved 16 oncology drugs",
        parent=group_timing,
        critical=True
    )
    claim_last6 = "In the last six weeks of 2025, the FDA approved 16 oncology drugs."
    await evaluator.verify(
        claim=claim_last6,
        node=leaf_last6,
        sources=stats_urls,
        additional_instruction="Confirm a burst of oncology approvals totaling 16 occurred in the final six weeks of 2025."
    )

    # Specific KRAS Drug Approval (critical)
    group_kras = evaluator.add_parallel(
        id="event4_specific_kras_drug_approval",
        desc="Details of a specific KRAS-targeted colorectal cancer drug combination approval",
        parent=node_event4,
        critical=True
    )

    # Drug Identification (critical)
    group_kras_id = evaluator.add_parallel(
        id="event4_kras_drug_identification",
        desc="The identity of the approved drug combination",
        parent=group_kras,
        critical=True
    )

    # Drug Combination Names (critical)
    leaf_combo = evaluator.add_leaf(
        id="event4_kras_drug_combination_names",
        desc="The approved combination consists of sotorasib with panitumumab",
        parent=group_kras_id,
        critical=True
    )
    claim_combo = "The approved KRAS-targeted colorectal cancer combination consists of sotorasib with panitumumab."
    await evaluator.verify(
        claim=claim_combo,
        node=leaf_combo,
        sources=kras_urls,
        additional_instruction="Confirm that the specific combination is sotorasib plus panitumumab."
    )

    # Approval Date (critical)
    leaf_kras_date = evaluator.add_leaf(
        id="event4_kras_approval_date",
        desc="The approval date was January 16, 2025",
        parent=group_kras_id,
        critical=True
    )
    claim_kras_date = "The approval date for sotorasib with panitumumab in colorectal cancer was January 16, 2025."
    await evaluator.verify(
        claim=claim_kras_date,
        node=leaf_kras_date,
        sources=kras_urls,
        additional_instruction="Verify the official FDA approval date (January 16, 2025)."
    )

    # Drug Indication (critical)
    group_kras_ind = evaluator.add_parallel(
        id="event4_kras_drug_indication",
        desc="The medical indication for which the drug was approved",
        parent=group_kras,
        critical=True
    )
    leaf_ind = evaluator.add_leaf(
        id="event4_kras_indication",
        desc="The indication is for KRAS G12C-mutated colorectal cancer",
        parent=group_kras_ind,
        critical=True
    )
    claim_ind = "The approved indication is KRAS G12C–mutated colorectal cancer."
    await evaluator.verify(
        claim=claim_ind,
        node=leaf_ind,
        sources=kras_urls,
        additional_instruction="Confirm the indication text clearly states KRAS G12C–mutated colorectal cancer."
    )


async def verify_event5(evaluator: Evaluator, parent, ex: Optional[Event5Info]) -> None:
    node_event5 = evaluator.add_parallel(
        id="event5_fresenius_kabi_pump_recall",
        desc="Details of the Class I medical device recall for Fresenius Kabi's Ivenix infusion pump system",
        parent=parent,
        critical=False
    )

    urls = ex.urls if ex else []

    # Recall Reference URL (critical gating)
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="event5_recall_reference_url",
        desc="Provide valid URL reference supporting the recall details",
        parent=node_event5,
        critical=True
    )

    # Device Identification (critical)
    group_device = evaluator.add_parallel(
        id="event5_device_identification",
        desc="Identification details of the recalled device",
        parent=node_event5,
        critical=True
    )

    # Device Product Name (critical)
    leaf_device_name = evaluator.add_leaf(
        id="event5_device_product_name",
        desc="The recalled device is the Ivenix Large Volume Pump",
        parent=group_device,
        critical=True
    )
    claim_device_name = "The recalled device is the Ivenix Large Volume Pump."
    await evaluator.verify(
        claim=claim_device_name,
        node=leaf_device_name,
        sources=urls,
        additional_instruction="Confirm that the recall specifically names 'Ivenix Large Volume Pump'."
    )

    # Affected Software Version (critical)
    leaf_software = evaluator.add_leaf(
        id="event5_affected_software_version",
        desc="The affected software version is 5.10.1 and earlier",
        parent=group_device,
        critical=True
    )
    claim_software = "The affected software version is 5.10.1 and earlier."
    await evaluator.verify(
        claim=claim_software,
        node=leaf_software,
        sources=urls,
        additional_instruction="Verify that the affected versions include 5.10.1 and any earlier releases."
    )

    # Product Code (critical)
    leaf_product_code = evaluator.add_leaf(
        id="event5_product_code",
        desc="The product code is LVP-SW-0005",
        parent=group_device,
        critical=True
    )
    claim_product_code = "The product code is LVP-SW-0005."
    await evaluator.verify(
        claim=claim_product_code,
        node=leaf_product_code,
        sources=urls,
        additional_instruction="Look for an explicit internal product code 'LVP-SW-0005'."
    )

    # Recall Classification (critical)
    group_class = evaluator.add_parallel(
        id="event5_recall_classification",
        desc="The severity classification of the recall",
        parent=node_event5,
        critical=True
    )
    leaf_class = evaluator.add_leaf(
        id="event5_recall_class",
        desc="The recall is classified as Class I, which is the most serious type",
        parent=group_class,
        critical=True
    )
    claim_class = "The recall is classified as Class I (the most serious type)."
    await evaluator.verify(
        claim=claim_class,
        node=leaf_class,
        sources=urls,
        additional_instruction="Confirm that FDA labels this as a Class I recall."
    )

    # Adverse Events Information (critical)
    group_adverse = evaluator.add_parallel(
        id="event5_adverse_events_information",
        desc="Information about injuries and deaths associated with the device",
        parent=node_event5,
        critical=True
    )
    # Adverse Events Count (critical)
    leaf_adverse_count = evaluator.add_leaf(
        id="event5_adverse_events_count",
        desc="As of November 18, 2025, there were 2 serious injuries and 0 deaths reported",
        parent=group_adverse,
        critical=True
    )
    claim_adverse_count = "As of November 18, 2025, there were 2 serious injuries and 0 deaths reported for this issue."
    await evaluator.verify(
        claim=claim_adverse_count,
        node=leaf_adverse_count,
        sources=urls,
        additional_instruction="Verify the reported counts exactly: 2 serious injuries, 0 deaths, with the 'as of' date specified."
    )

    # Adverse Events Date (critical)
    leaf_adverse_date = evaluator.add_leaf(
        id="event5_adverse_events_date",
        desc="The injury/death count was reported as of November 18, 2025",
        parent=group_adverse,
        critical=True
    )
    claim_adverse_date = "The injury/death count was reported as of November 18, 2025."
    await evaluator.verify(
        claim=claim_adverse_date,
        node=leaf_adverse_date,
        sources=urls,
        additional_instruction="Confirm the 'as of' date is clearly stated as November 18, 2025."
    )

    # Recall Timeline (critical)
    group_timeline = evaluator.add_parallel(
        id="event5_recall_timeline",
        desc="Key dates in the recall process",
        parent=node_event5,
        critical=True
    )
    # Initial Recall Notification Date (critical)
    leaf_initial_date = evaluator.add_leaf(
        id="event5_initial_recall_notification_date",
        desc="The initial recall notification was sent on November 14, 2025",
        parent=group_timeline,
        critical=True
    )
    claim_initial_date = "The initial recall notification was sent on November 14, 2025."
    await evaluator.verify(
        claim=claim_initial_date,
        node=leaf_initial_date,
        sources=urls,
        additional_instruction="Look for the first customer communication/notice date of November 14, 2025."
    )

    # FDA Software Correction Announcement (critical)
    leaf_correction_date = evaluator.add_leaf(
        id="event5_fda_software_correction_announcement",
        desc="The FDA announced the software correction on February 25, 2026",
        parent=group_timeline,
        critical=True
    )
    claim_correction_date = "The FDA announced the software correction on February 25, 2026."
    await evaluator.verify(
        claim=claim_correction_date,
        node=leaf_correction_date,
        sources=urls,
        additional_instruction="Find the FDA communication indicating a correction announcement on February 25, 2026."
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
    # Initialize evaluator (root is parallel aggregation)
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

    # Top-level node per rubric
    top = evaluator.add_parallel(
        id="healthcare_policy_medical_events_2025",
        desc="Comprehensive analysis of five major healthcare-related events, policy changes, and medical developments from 2025, each with multiple verifiable attributes",
        parent=root,
        critical=False
    )

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_healthcare_events(),
        template_class=HealthcareEvents2025Extraction,
        extraction_name="healthcare_events_2025_extraction"
    )

    # Build and verify subtrees for each event
    await verify_event1(evaluator, top, extracted.event1 if extracted else None)
    await verify_event2(evaluator, top, extracted.event2 if extracted else None)
    await verify_event3(evaluator, top, extracted.event3 if extracted else None)
    await verify_event4(evaluator, top, extracted.event4 if extracted else None)
    await verify_event5(evaluator, top, extracted.event5 if extracted else None)

    # Return structured summary
    return evaluator.get_summary()