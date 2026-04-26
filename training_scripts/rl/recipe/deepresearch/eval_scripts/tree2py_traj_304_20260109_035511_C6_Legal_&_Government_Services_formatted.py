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
TASK_ID = "ak_veto_override_process"
TASK_DESCRIPTION = (
    "Provide a comprehensive analysis of Alaska's legislative veto override process. Your analysis must include: "
    "(1) the specific constitutional provision that governs veto override procedures, "
    "(2) the exact vote thresholds required to override vetoes for both appropriation/revenue bills and other bills, "
    "including both the fractional requirement and the specific number of votes needed out of the total legislature membership, "
    "(3) the procedural requirements for conducting override votes including whether chambers meet separately or jointly, "
    "the timing requirements when a veto is received during a regular session, and the timeline requirements for reconsidering "
    "bills vetoed after adjournment of the first regular session and after adjournment of the second regular session, and "
    "(4) verification that the Alaska Legislature conducted a special session in August 2025 where at least one gubernatorial "
    "veto was successfully overridden. For each major component, provide verifiable URL references to support your findings."
)

# Expected reference values (for simple internal checks and for recording GT info)
EXPECTED_INFO = {
    "constitution_article_section": "Alaska Constitution, Article II, Section 16",
    "membership_total": "60 (20 Senators, 40 Representatives)",
    "appropriation_threshold": {
        "fraction": "three-fourths (3/4)",
        "votes": "45 out of 60"
    },
    "other_bills_threshold": {
        "fraction": "two-thirds (2/3)",
        "votes": "40 out of 60"
    },
    "procedural_core": [
        "Joint session required for veto overrides",
        "If a veto is received during a regular session, the legislature must meet immediately in joint session to reconsider",
        "Bills vetoed after adjournment of the first regular session must be reconsidered no later than the fifth day of the next regular or special session",
        "Bills vetoed after adjournment of the second regular session must be reconsidered no later than the fifth day of a special session (if called)",
        "Override vote must be entered in the journals of both houses"
    ],
    "august_2025": {
        "special_session": True,
        "vetoes_overridden": "two"
    }
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ConstitutionalInfo(BaseModel):
    article_section: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ThresholdsInfo(BaseModel):
    membership_total: Optional[str] = None
    membership_breakdown: Optional[str] = None
    membership_urls: List[str] = Field(default_factory=list)

    appropriation_fraction: Optional[str] = None
    appropriation_vote_count: Optional[str] = None
    appropriation_urls: List[str] = Field(default_factory=list)

    other_fraction: Optional[str] = None
    other_vote_count: Optional[str] = None
    other_urls: List[str] = Field(default_factory=list)


class ProceduralInfo(BaseModel):
    joint_session_requirement: Optional[str] = None
    immediate_session_requirement: Optional[str] = None
    post_adj_first_session_timeline: Optional[str] = None
    post_adj_second_session_timeline: Optional[str] = None
    journal_entry_requirement: Optional[str] = None
    procedural_urls: List[str] = Field(default_factory=list)


class August2025Info(BaseModel):
    special_session_august_2025: Optional[str] = None
    num_vetoes_overridden: Optional[str] = None
    event_urls: List[str] = Field(default_factory=list)


class AlaskaVetoOverrideExtraction(BaseModel):
    constitutional: Optional[ConstitutionalInfo] = None
    thresholds: Optional[ThresholdsInfo] = None
    procedural: Optional[ProceduralInfo] = None
    event: Optional[August2025Info] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_ak_veto_override() -> str:
    return """
Extract the following structured information from the provided answer text about Alaska's legislative veto override process. Return JSON strictly following the schema below. Do not invent information not explicitly stated in the answer. For each 'urls' field, extract the actual URLs explicitly present in the answer text (including markdown links), preserving full URLs.

Return an object with these fields:

- constitutional:
  - article_section: The cited constitutional provision governing veto overrides (e.g., "Alaska Constitution, Article II, Section 16"). If not explicitly stated, return null.
  - urls: Array of URLs that specifically reference or display the constitutional provision. If none provided, return [].

- thresholds:
  - membership_total: The stated total membership of the Alaska Legislature (e.g., "60"). If not stated, return null.
  - membership_breakdown: If stated, the breakdown by chamber (e.g., "20 Senators, 40 Representatives"); else null.
  - membership_urls: Array of URLs supporting the membership figure/breakdown. If none provided, return [].

  - appropriation_fraction: The fraction threshold for overriding vetoes of appropriation bills or bills to raise revenue (e.g., "three-fourths (3/4)" or "3/4"). If not stated, return null.
  - appropriation_vote_count: The specific number of votes out of the total membership required for appropriation/revenue bills (e.g., "45 out of 60", "45/60", or "45"). If not stated, return null.
  - appropriation_urls: Array of URLs that support the appropriation/revenue override threshold. If none provided, return [].

  - other_fraction: The fraction threshold for overriding vetoes of all other bills (e.g., "two-thirds (2/3)" or "2/3"). If not stated, return null.
  - other_vote_count: The specific number of votes out of the total membership required for other bills (e.g., "40 out of 60", "40/60", or "40"). If not stated, return null.
  - other_urls: Array of URLs that support the other-bills override threshold. If none provided, return [].

- procedural:
  - joint_session_requirement: The statement about whether override votes occur in joint session (both chambers together). If not stated, return null.
  - immediate_session_requirement: The statement about what happens if a veto message is received during a regular session (e.g., must meet immediately in joint session). If not stated, return null.
  - post_adj_first_session_timeline: The stated timeline for reconsidering bills vetoed after adjournment of the first regular session (e.g., no later than the fifth day of the next regular or special session). If not stated, return null.
  - post_adj_second_session_timeline: The stated timeline for reconsidering bills vetoed after adjournment of the second regular session (e.g., no later than the fifth day of a special session if called). If not stated, return null.
  - journal_entry_requirement: The statement that the override vote must be entered in the journals of both houses (if mentioned). If not stated, return null.
  - procedural_urls: Array of URLs that support the procedural requirements above. If none provided, return [].

- event:
  - special_session_august_2025: A statement that there was a special session in August 2025 (if mentioned). If not stated, return null.
  - num_vetoes_overridden: The number of gubernatorial vetoes overridden during that August 2025 special session, if stated (e.g., "two", "2", or similar). If not stated, return null.
  - event_urls: Array of URLs that document the August 2025 special session and the veto override(s). If none provided, return [].
"""


# --------------------------------------------------------------------------- #
# Helper for existence check nodes                                            #
# --------------------------------------------------------------------------- #
def add_urls_existence_node(
    evaluator: Evaluator,
    urls: Optional[List[str]],
    node_id: str,
    desc: str,
    parent,
    critical: bool = True
):
    urls = urls or []
    return evaluator.add_custom_node(
        result=len([u for u in urls if isinstance(u, str) and u.strip()]) > 0,
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_constitutional_authority(
    evaluator: Evaluator,
    parent,
    data: Optional[ConstitutionalInfo]
):
    node = evaluator.add_parallel(
        id="Constitutional_Authority",
        desc="Correct identification of the constitutional provision governing veto override procedures, with a verifiable URL reference",
        parent=parent,
        critical=True
    )

    urls = (data.urls if data else []) if data else []
    add_urls_existence_node(
        evaluator,
        urls,
        node_id="Constitutional_Authority_URL_exists",
        desc="Constitutional authority URLs are provided",
        parent=node,
        critical=True
    )

    # Leaf: Constitution_Article_Section
    leaf_article = evaluator.add_leaf(
        id="Constitution_Article_Section",
        desc="Specifies the governing provision as Alaska Constitution, Article II, Section 16",
        parent=node,
        critical=True
    )
    claim_article = "Alaska Constitution, Article II, Section 16 governs the veto and veto override procedures."
    await evaluator.verify(
        claim=claim_article,
        node=leaf_article,
        sources=urls,
        additional_instruction="Confirm that the source displays Alaska Constitution Article II, Section 16 and that it addresses veto and override procedures."
    )

    # Leaf: Constitutional_Authority_URL
    leaf_const_url = evaluator.add_leaf(
        id="Constitutional_Authority_URL",
        desc="Provides a verifiable URL reference supporting Alaska Constitution Article II, Section 16 as the authority",
        parent=node,
        critical=True
    )
    claim_const_url = "This source explicitly contains Alaska Constitution Article II, Section 16 and identifies it as governing veto override procedures."
    await evaluator.verify(
        claim=claim_const_url,
        node=leaf_const_url,
        sources=urls,
        additional_instruction="If multiple URLs are provided, any one that clearly shows Article II, Section 16 on veto/override is sufficient."
    )


async def verify_vote_threshold_requirements(
    evaluator: Evaluator,
    parent,
    data: Optional[ThresholdsInfo]
):
    node = evaluator.add_parallel(
        id="Vote_Threshold_Requirements",
        desc="Accurate specification of veto override vote thresholds (fractions and vote counts out of total membership), with verifiable URL references",
        parent=parent,
        critical=True
    )

    # Leaf: Legislature_Total_Membership
    leaf_membership = evaluator.add_leaf(
        id="Legislature_Total_Membership",
        desc="States the Alaska Legislature total membership is 60 (20 Senators, 40 Representatives)",
        parent=node,
        critical=True
    )
    membership_urls = data.membership_urls if data else []
    claim_membership = "The Alaska Legislature has a total membership of 60, consisting of 20 Senators and 40 Representatives."
    await evaluator.verify(
        claim=claim_membership,
        node=leaf_membership,
        sources=membership_urls,
        additional_instruction="Verify using an official or reliable source that the Alaska Legislature totals 60 members (20 senate, 40 house)."
    )

    # Subnode: Appropriation/Revnue Bills Threshold
    appr_node = evaluator.add_parallel(
        id="Appropriation_Revenue_Bills_Threshold",
        desc="Override threshold for appropriation bills or bills to raise revenue (fraction and vote count), with URL support",
        parent=node,
        critical=True
    )
    appr_urls = (data.appropriation_urls if data else []) if data else []
    add_urls_existence_node(
        evaluator,
        appr_urls,
        node_id="Appropriation_Threshold_URL_exists",
        desc="Appropriation/revenue threshold URLs are provided",
        parent=appr_node,
        critical=True
    )

    # Leaf: Appropriation_Fraction
    leaf_appr_frac = evaluator.add_leaf(
        id="Appropriation_Fraction",
        desc="States the fraction required is three-fourths (3/4) of the membership",
        parent=appr_node,
        critical=True
    )
    claim_appr_frac = "Overriding a veto of an appropriation bill or a bill to raise revenue requires a three-fourths (3/4) vote of the membership."
    await evaluator.verify(
        claim=claim_appr_frac,
        node=leaf_appr_frac,
        sources=appr_urls,
        additional_instruction="Confirm the page clearly states a 3/4 (three-fourths) requirement for appropriation or revenue bill veto overrides."
    )

    # Leaf: Appropriation_Specific_Vote_Count
    leaf_appr_votes = evaluator.add_leaf(
        id="Appropriation_Specific_Vote_Count",
        desc="States the specific votes required are 45 out of 60",
        parent=appr_node,
        critical=True
    )
    claim_appr_votes = "The specific number of votes required to override such vetoes is 45 out of the 60-member legislature."
    await evaluator.verify(
        claim=claim_appr_votes,
        node=leaf_appr_votes,
        sources=appr_urls,
        additional_instruction="The source should make it clear that three-fourths of 60 equals 45, either explicitly or implicitly."
    )

    # Leaf: Appropriation_Threshold_URL
    leaf_appr_url = evaluator.add_leaf(
        id="Appropriation_Threshold_URL",
        desc="Provides a verifiable URL reference for the appropriation/revenue override threshold",
        parent=appr_node,
        critical=True
    )
    claim_appr_url = "This source explicitly states the three-fourths (3/4) threshold (equating to 45 out of 60) for overriding vetoes of appropriation or revenue bills in Alaska."
    await evaluator.verify(
        claim=claim_appr_url,
        node=leaf_appr_url,
        sources=appr_urls,
        additional_instruction="Any one provided URL suffices if it clearly documents the 3/4 (45 of 60) requirement for appropriation/revenue bills."
    )

    # Subnode: Other Bills Threshold
    other_node = evaluator.add_parallel(
        id="Other_Bills_Threshold",
        desc="Override threshold for all other vetoed bills (fraction and vote count), with URL support",
        parent=node,
        critical=True
    )
    other_urls = (data.other_urls if data else []) if data else []
    add_urls_existence_node(
        evaluator,
        other_urls,
        node_id="Other_Bills_Threshold_URL_exists",
        desc="Other-bills threshold URLs are provided",
        parent=other_node,
        critical=True
    )

    # Leaf: Other_Bills_Fraction
    leaf_other_frac = evaluator.add_leaf(
        id="Other_Bills_Fraction",
        desc="States the fraction required is two-thirds (2/3) of the membership",
        parent=other_node,
        critical=True
    )
    claim_other_frac = "Overriding vetoes of all other bills requires a two-thirds (2/3) vote of the membership."
    await evaluator.verify(
        claim=claim_other_frac,
        node=leaf_other_frac,
        sources=other_urls,
        additional_instruction="Confirm the source states a 2/3 requirement for veto overrides of bills other than appropriation/revenue bills."
    )

    # Leaf: Other_Bills_Specific_Vote_Count
    leaf_other_votes = evaluator.add_leaf(
        id="Other_Bills_Specific_Vote_Count",
        desc="States the specific votes required are 40 out of 60",
        parent=other_node,
        critical=True
    )
    claim_other_votes = "The specific number of votes required to override such vetoes is 40 out of the 60-member legislature."
    await evaluator.verify(
        claim=claim_other_votes,
        node=leaf_other_votes,
        sources=other_urls,
        additional_instruction="The source should make it clear that two-thirds of 60 equals 40, either directly or by implication."
    )

    # Leaf: Other_Bills_Threshold_URL
    leaf_other_url = evaluator.add_leaf(
        id="Other_Bills_Threshold_URL",
        desc="Provides a verifiable URL reference for the other-bills override threshold",
        parent=other_node,
        critical=True
    )
    claim_other_url = "This source explicitly states the two-thirds (2/3) threshold (equating to 40 out of 60) for overriding vetoes of other bills in Alaska."
    await evaluator.verify(
        claim=claim_other_url,
        node=leaf_other_url,
        sources=other_urls,
        additional_instruction="Any one provided URL suffices if it clearly documents the 2/3 (40 of 60) requirement for other bills."
    )


async def verify_procedural_requirements(
    evaluator: Evaluator,
    parent,
    data: Optional[ProceduralInfo]
):
    node = evaluator.add_parallel(
        id="Procedural_Requirements",
        desc="Accurate description of procedural requirements for veto override votes (joint vs separate, timing rules, timelines, journal entry), with a verifiable URL reference",
        parent=parent,
        critical=True
    )

    urls = (data.procedural_urls if data else []) if data else []
    add_urls_existence_node(
        evaluator,
        urls,
        node_id="Procedural_Requirements_URL_exists",
        desc="Procedural requirements URLs are provided",
        parent=node,
        critical=True
    )

    # Leaf: Joint_Session_Requirement
    leaf_joint = evaluator.add_leaf(
        id="Joint_Session_Requirement",
        desc="States override votes occur in joint session (both chambers meeting together as one body)",
        parent=node,
        critical=True
    )
    claim_joint = "The Alaska Legislature conducts veto override votes in a joint session of both houses (meeting together as one body)."
    await evaluator.verify(
        claim=claim_joint,
        node=leaf_joint,
        sources=urls,
        additional_instruction="Verify the source explicitly indicates joint session for veto overrides."
    )

    # Leaf: Immediate_Session_Requirement
    leaf_immediate = evaluator.add_leaf(
        id="Immediate_Session_Requirement",
        desc="States that when a veto message is received during a regular session, the legislature must meet immediately in joint session to reconsider",
        parent=node,
        critical=True
    )
    claim_immediate = "When a veto message is received during a regular session, the legislature must meet immediately in joint session to reconsider the veto."
    await evaluator.verify(
        claim=claim_immediate,
        node=leaf_immediate,
        sources=urls,
        additional_instruction="Check the constitution/statute/procedural rule text for the 'immediate' joint session requirement upon receiving a veto during session."
    )

    # Leaf: Post_Adjournment_Timeline_First_Session
    leaf_first = evaluator.add_leaf(
        id="Post_Adjournment_Timeline_First_Session",
        desc="States bills vetoed after adjournment of the first regular session must be reconsidered no later than the fifth day of the next regular or special session",
        parent=node,
        critical=True
    )
    claim_first = "Bills vetoed after adjournment of the first regular session must be reconsidered no later than the fifth day of the next regular session or of a special session."
    await evaluator.verify(
        claim=claim_first,
        node=leaf_first,
        sources=urls,
        additional_instruction="Look for the explicit 'no later than the fifth day' timeline for bills vetoed after the first regular session."
    )

    # Leaf: Post_Adjournment_Timeline_Second_Session
    leaf_second = evaluator.add_leaf(
        id="Post_Adjournment_Timeline_Second_Session",
        desc="States bills vetoed after adjournment of the second regular session must be reconsidered no later than the fifth day of a special session (if called)",
        parent=node,
        critical=True
    )
    claim_second = "Bills vetoed after adjournment of the second regular session must be reconsidered no later than the fifth day of a special session, if one is called."
    await evaluator.verify(
        claim=claim_second,
        node=leaf_second,
        sources=urls,
        additional_instruction="Look for the explicit 'no later than the fifth day' timeline for bills vetoed after the second regular session."
    )

    # Leaf: Journal_Entry_Requirement
    leaf_journal = evaluator.add_leaf(
        id="Journal_Entry_Requirement",
        desc="States the override vote must be entered in the journals of both houses",
        parent=node,
        critical=True
    )
    claim_journal = "The override vote must be entered in the journals of both houses."
    await evaluator.verify(
        claim=claim_journal,
        node=leaf_journal,
        sources=urls,
        additional_instruction="Confirm the source states that the vote is recorded/entered in both houses' journals."
    )

    # Leaf: Procedural_Requirements_URL
    leaf_proc_url = evaluator.add_leaf(
        id="Procedural_Requirements_URL",
        desc="Provides a verifiable URL reference supporting the procedural requirements described",
        parent=node,
        critical=True
    )
    claim_proc_url = "This source explicitly supports the joint session requirement, the immediate meeting upon veto during session, the fifth-day timelines after adjournment of the first and second regular sessions, and the journal entry requirement."
    await evaluator.verify(
        claim=claim_proc_url,
        node=leaf_proc_url,
        sources=urls,
        additional_instruction="Any one provided URL is sufficient if it clearly documents all the listed procedural requirements."
    )


async def verify_august_2025_event(
    evaluator: Evaluator,
    parent,
    data: Optional[August2025Info]
):
    node = evaluator.add_parallel(
        id="August_2025_Override_Event",
        desc="Verification of the August 2025 special session and successful veto override(s), with a verifiable URL reference",
        parent=parent,
        critical=True
    )

    urls = (data.event_urls if data else []) if data else []
    add_urls_existence_node(
        evaluator,
        urls,
        node_id="August_2025_Event_URL_exists",
        desc="URLs documenting the August 2025 special session and veto override(s) are provided",
        parent=node,
        critical=True
    )

    # Leaf: Special_Session_August_2025
    leaf_special = evaluator.add_leaf(
        id="Special_Session_August_2025",
        desc="Verifies that the Alaska Legislature conducted a special session in August 2025",
        parent=node,
        critical=True
    )
    claim_special = "The Alaska Legislature conducted a special session in August 2025."
    await evaluator.verify(
        claim=claim_special,
        node=leaf_special,
        sources=urls,
        additional_instruction="Verify that the source clearly identifies a special session in August 2025."
    )

    # Leaf: Two_Vetoes_Overridden_August_2025
    leaf_two = evaluator.add_leaf(
        id="Two_Vetoes_Overridden_August_2025",
        desc="Verifies that two gubernatorial vetoes were successfully overridden during the August 2025 special session",
        parent=node,
        critical=True
    )
    claim_two = "During the August 2025 special session, two gubernatorial vetoes were successfully overridden."
    await evaluator.verify(
        claim=claim_two,
        node=leaf_two,
        sources=urls,
        additional_instruction="The source should state clearly that two vetoes were overridden in that special session."
    )

    # Leaf: August_2025_Event_URL
    leaf_event_url = evaluator.add_leaf(
        id="August_2025_Event_URL",
        desc="Provides a verifiable URL reference documenting the August 2025 special session veto override(s)",
        parent=node,
        critical=True
    )
    claim_event_url = "This source documents the August 2025 special session and the veto override(s) that occurred during it."
    await evaluator.verify(
        claim=claim_event_url,
        node=leaf_event_url,
        sources=urls,
        additional_instruction="Any one provided URL is sufficient if it clearly documents the session and the overrides."
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
        prompt=prompt_extract_ak_veto_override(),
        template_class=AlaskaVetoOverrideExtraction,
        extraction_name="ak_veto_override_extraction"
    )

    # Record expected info as "ground truth info" for transparency (not used for scoring)
    evaluator.add_ground_truth({
        "expected": EXPECTED_INFO
    }, gt_type="reference_expectations")

    # Build main critical analysis node
    main = evaluator.add_parallel(
        id="Alaska_Veto_Override_Process_Analysis",
        desc="Complete and accurate analysis of Alaska's legislative veto override process including constitutional provisions, vote requirements, procedural rules, and August 2025 special-session override verification, with verifiable URL references for each major component",
        parent=root,
        critical=True
    )

    # Sub-verifications according to rubric tree
    await verify_constitutional_authority(evaluator, main, extracted.constitutional)
    await verify_vote_threshold_requirements(evaluator, main, extracted.thresholds)
    await verify_procedural_requirements(evaluator, main, extracted.procedural)
    await verify_august_2025_event(evaluator, main, extracted.event)

    return evaluator.get_summary()