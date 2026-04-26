import asyncio
import logging
import math
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "wv_veto_override"
TASK_DESCRIPTION = (
    "Under the West Virginia Constitution, the procedures and vote thresholds for overriding a governor's veto differ "
    "depending on the type of bill. For each of the following bill types, identify the specific constitutional article "
    "and section that governs the veto override procedure, state the vote threshold required, and calculate the minimum "
    "number of votes needed in both the House of Delegates and the Senate to successfully override a gubernatorial veto:\n\n"
    "1. A regular bill (non-appropriations legislation)\n"
    "2. An appropriations bill (budget or supplementary appropriation bill)\n\n"
    "For each bill type, your answer should include:\n"
    "- The constitutional citation (Article and Section number)\n"
    "- The vote threshold formula (e.g., simple majority, two-thirds, etc.)\n"
    "- The specific minimum number of votes required in the House of Delegates (out of 100 members)\n"
    "- The specific minimum number of votes required in the Senate (out of 34 members)\n\n"
    "Provide a URL reference to the relevant constitutional provision for each bill type."
)

# Expected ground truth references and computed vote counts
EXPECTED_REGULAR_CITATION = "Article VII, Section 14"
EXPECTED_APPROPRIATIONS_CITATION = "Article VI, Section 51"

HOUSE_MEMBERS = 100
SENATE_MEMBERS = 34

REGULAR_HOUSE_MIN = (HOUSE_MEMBERS // 2) + 1  # strict majority
REGULAR_SENATE_MIN = (SENATE_MEMBERS // 2) + 1  # strict majority

APPROPRIATIONS_HOUSE_MIN = math.ceil(HOUSE_MEMBERS * 2 / 3)
APPROPRIATIONS_SENATE_MIN = math.ceil(SENATE_MEMBERS * 2 / 3)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class GeneralConstraintsExtraction(BaseModel):
    based_on_members_elected_not_present: Optional[bool] = None
    recorded_yeas_and_nays: Optional[bool] = None
    both_houses_must_meet_threshold: Optional[bool] = None


class RegularBillInfo(BaseModel):
    citation: Optional[str] = None
    url: Optional[str] = None
    threshold_formula: Optional[str] = None
    house_min_votes: Optional[str] = None
    senate_min_votes: Optional[str] = None


class AppropriationsBillInfo(BaseModel):
    citation: Optional[str] = None
    url: Optional[str] = None
    threshold_formula: Optional[str] = None
    house_min_votes: Optional[str] = None
    senate_min_votes: Optional[str] = None


class WVVetoExtraction(BaseModel):
    general: Optional[GeneralConstraintsExtraction] = None
    regular_bill: Optional[RegularBillInfo] = None
    appropriations_bill: Optional[AppropriationsBillInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_veto_info() -> str:
    return (
        "Extract the information the answer provides for West Virginia veto overrides, organized into three parts: "
        "general constraints, regular bill (non-appropriations), and appropriations bill.\n\n"
        "Return a JSON object with keys 'general', 'regular_bill', and 'appropriations_bill'.\n\n"
        "1) general:\n"
        "- based_on_members_elected_not_present: true if the answer explicitly states that veto-override thresholds are "
        "based on members elected to each house (not members present); otherwise false or null if not stated.\n"
        "- recorded_yeas_and_nays: true if the answer explicitly states that the override vote must be by yeas and nays "
        "and entered in each house journal; otherwise false or null if not stated.\n"
        "- both_houses_must_meet_threshold: true if the answer explicitly states that both chambers must separately meet "
        "the threshold to override; otherwise false or null if not stated.\n\n"
        "2) regular_bill:\n"
        "- citation: the Article and Section citation text exactly as written in the answer for the governing provision "
        "for regular bill veto overrides (e.g., 'Article VII, Section 14', 'Art. VII §14'); null if missing.\n"
        "- url: the URL provided in the answer that references the relevant constitutional provision for regular bills; "
        "must be a valid URL string; null if missing.\n"
        "- threshold_formula: the textual formula stated in the answer for the vote threshold for regular bills "
        "(e.g., 'simple majority of the members elected to each house', 'strict majority'); null if missing.\n"
        "- house_min_votes: the specific minimum number the answer gives for the House (out of 100); extract the digits "
        "as they appear (e.g., '51', '51 votes'); null if missing.\n"
        "- senate_min_votes: the specific minimum number the answer gives for the Senate (out of 34); extract the digits "
        "as they appear (e.g., '18', '18 votes'); null if missing.\n\n"
        "3) appropriations_bill:\n"
        "- citation: the Article and Section citation text exactly as written in the answer for appropriations/budget "
        "veto overrides (e.g., 'Article VI, Section 51', 'Art. VI §51'); null if missing.\n"
        "- url: the URL provided in the answer that references the relevant constitutional provision for appropriations "
        "bills; must be a valid URL string; null if missing.\n"
        "- threshold_formula: the textual formula stated in the answer for the vote threshold for appropriations bills "
        "(e.g., 'two-thirds of the members elected to each house'); null if missing.\n"
        "- house_min_votes: the specific minimum number the answer gives for the House (out of 100); extract the digits "
        "as they appear (e.g., '67', '67 votes'); null if missing.\n"
        "- senate_min_votes: the specific minimum number the answer gives for the Senate (out of 34); extract the digits "
        "as they appear (e.g., '23', '23 votes'); null if missing.\n\n"
        "Important:\n"
        "- Do not invent any information; extract only what is explicitly present in the answer text.\n"
        "- For URL fields, extract only actual URLs present in the answer (including markdown URLs); if none, return null.\n"
        "- For vote counts, prefer the numeric digits if present; if embedded in text (e.g., '51 votes'), still extract the digits."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def parse_first_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"\d{1,4}", text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_general_constraints(
    evaluator: Evaluator,
    parent_node,
    general: Optional[GeneralConstraintsExtraction],
) -> None:
    node = evaluator.add_parallel(
        id="general_procedural_constraints",
        desc="General procedural constraints about how overrides are determined (apply to both bill types).",
        parent=parent_node,
        critical=True,
    )

    # Leaf: based_on_members_elected_not_present
    leaf1 = evaluator.add_leaf(
        id="based_on_members_elected_not_present",
        desc="States that override thresholds are based on members elected to each house (not members present).",
        parent=node,
        critical=True,
    )
    claim1 = (
        "The answer explicitly states that veto override thresholds are based on the members elected to each house, "
        "not the members present."
    )
    await evaluator.verify(
        claim=claim1,
        node=leaf1,
        additional_instruction=(
            "Only pass if the answer text itself explicitly includes this statement. Do NOT infer or assume."
        ),
    )

    # Leaf: recorded_yeas_and_nays
    leaf2 = evaluator.add_leaf(
        id="recorded_yeas_and_nays",
        desc="States that the override vote must be by yeas and nays and entered in each house journal.",
        parent=node,
        critical=True,
    )
    claim2 = (
        "The answer explicitly states that the veto-override vote must be taken by yeas and nays and entered in the "
        "journal of each house."
    )
    await evaluator.verify(
        claim=claim2,
        node=leaf2,
        additional_instruction=(
            "Only pass if the answer text itself explicitly includes this statement. Do NOT infer or assume."
        ),
    )

    # Leaf: both_houses_must_meet_threshold
    leaf3 = evaluator.add_leaf(
        id="both_houses_must_meet_threshold",
        desc="States that both House of Delegates and Senate must separately meet the threshold to override.",
        parent=node,
        critical=True,
    )
    claim3 = (
        "The answer explicitly states that both the House of Delegates and the Senate must each independently meet the "
        "required threshold to override a governor's veto."
    )
    await evaluator.verify(
        claim=claim3,
        node=leaf3,
        additional_instruction=(
            "Only pass if the answer text itself explicitly includes this statement. Do NOT infer or assume."
        ),
    )


async def verify_regular_bill_requirements(
    evaluator: Evaluator,
    parent_node,
    reg: Optional[RegularBillInfo],
) -> None:
    node = evaluator.add_parallel(
        id="regular_bill_requirements",
        desc="Regular bill (non-appropriations) override requirements.",
        parent=parent_node,
        critical=False,
    )

    citation_val = reg.citation if reg else None
    url_val = reg.url if reg else None
    house_votes_text = reg.house_min_votes if reg else None
    senate_votes_text = reg.senate_min_votes if reg else None

    # Leaf: citation identification
    leaf_cite = evaluator.add_leaf(
        id="regular_bill_citation",
        desc="Identifies WV Constitution Article VII, Section 14 as the governing provision for regular bill veto overrides.",
        parent=node,
        critical=True,
    )
    claim_cite = (
        f"The governing West Virginia constitutional provision for veto overrides of regular (non-appropriations) bills "
        f"is Article VII, Section 14. The answer's citation text is '{citation_val}'."
    )
    await evaluator.verify(
        claim=claim_cite,
        node=leaf_cite,
        additional_instruction=(
            "Pass only if the answer includes a citation equivalent to 'Article VII, Section 14' "
            "(allow minor variants like 'Art. VII §14', 'Article 7, Section 14'). If the answer does not include a "
            "citation, mark incorrect."
        ),
    )

    # Leaf: URL presence and relevance
    leaf_url = evaluator.add_leaf(
        id="regular_bill_url",
        desc="Provides a URL reference to the relevant constitutional provision for regular bills.",
        parent=node,
        critical=True,
    )
    claim_url = (
        "This webpage is the West Virginia Constitution's Article VII, Section 14 (Governor's approval/veto of bills) "
        "or an authoritative page reproducing that text."
    )
    await evaluator.verify(
        claim=claim_url,
        node=leaf_url,
        sources=url_val,
        additional_instruction=(
            "Verify that the provided URL points to the WV Constitution provision governing veto overrides of regular bills "
            "(Article VII, Section 14). If no URL was provided in the answer, mark incorrect."
        ),
    )

    # Leaf: threshold formula
    leaf_thresh = evaluator.add_leaf(
        id="regular_bill_threshold",
        desc="States the vote threshold for regular bills is a simple majority of the members elected to each house (i.e., strictly more than half).",
        parent=node,
        critical=True,
    )
    claim_thresh = (
        "Under WV Const. Article VII, Section 14, overriding a veto of a regular bill requires a majority of the members "
        "elected to each house (strictly more than half)."
    )
    await evaluator.verify(
        claim=claim_thresh,
        node=leaf_thresh,
        sources=url_val,
        additional_instruction=(
            "Confirm that the page states 'a majority of the members elected to each house' or equivalent language. "
            "Allow minor phrasing variations. If no URL was provided, you should still judge based on the claim; "
            "however, missing URL in the dedicated URL leaf should cause that leaf to fail."
        ),
    )

    # Leaf: specific minimum House votes (strict majority of 100 -> 51)
    leaf_house = evaluator.add_leaf(
        id="regular_bill_house_min_votes",
        desc="Gives a specific minimum House vote count consistent with a strict majority of 100 members elected (i.e., equals floor(100/2) + 1).",
        parent=node,
        critical=True,
    )
    claim_house = (
        f"The answer states a specific minimum House vote count for overriding a regular-bill veto, and it is "
        f"{REGULAR_HOUSE_MIN} out of 100 members."
    )
    await evaluator.verify(
        claim=claim_house,
        node=leaf_house,
        additional_instruction=(
            "Pass only if the answer explicitly provides a numeric House minimum and it equals 51 "
            "(a strict majority of 100 members). If the answer omitted a number, mark incorrect."
        ),
    )

    # Leaf: specific minimum Senate votes (strict majority of 34 -> 18)
    leaf_senate = evaluator.add_leaf(
        id="regular_bill_senate_min_votes",
        desc="Gives a specific minimum Senate vote count consistent with a strict majority of 34 members elected (i.e., equals floor(34/2) + 1).",
        parent=node,
        critical=True,
    )
    claim_senate = (
        f"The answer states a specific minimum Senate vote count for overriding a regular-bill veto, and it is "
        f"{REGULAR_SENATE_MIN} out of {SENATE_MEMBERS} members."
    )
    await evaluator.verify(
        claim=claim_senate,
        node=leaf_senate,
        additional_instruction=(
            "Pass only if the answer explicitly provides a numeric Senate minimum and it equals 18 "
            "(a strict majority of 34 members). If the answer omitted a number, mark incorrect."
        ),
    )


async def verify_appropriations_bill_requirements(
    evaluator: Evaluator,
    parent_node,
    appr: Optional[AppropriationsBillInfo],
) -> None:
    node = evaluator.add_parallel(
        id="appropriations_bill_requirements",
        desc="Appropriations bill (budget or supplementary appropriation) override requirements.",
        parent=parent_node,
        critical=False,
    )

    citation_val = appr.citation if appr else None
    url_val = appr.url if appr else None
    house_votes_text = appr.house_min_votes if appr else None
    senate_votes_text = appr.senate_min_votes if appr else None

    # Leaf: citation identification
    leaf_cite = evaluator.add_leaf(
        id="appropriations_bill_citation",
        desc="Identifies WV Constitution Article VI, Section 51 as the governing provision for appropriations/budget/supplementary appropriation bill veto overrides.",
        parent=node,
        critical=True,
    )
    claim_cite = (
        f"The governing West Virginia constitutional provision for veto overrides of appropriations/budget bills "
        f"is Article VI, Section 51. The answer's citation text is '{citation_val}'."
    )
    await evaluator.verify(
        claim=claim_cite,
        node=leaf_cite,
        additional_instruction=(
            "Pass only if the answer includes a citation equivalent to 'Article VI, Section 51' "
            "(allow minor variants like 'Art. VI §51', 'Article 6, Section 51'). If the answer does not include a "
            "citation, mark incorrect."
        ),
    )

    # Leaf: URL presence and relevance
    leaf_url = evaluator.add_leaf(
        id="appropriations_bill_url",
        desc="Provides a URL reference to the relevant constitutional provision for appropriations bills.",
        parent=node,
        critical=True,
    )
    claim_url = (
        "This webpage is the West Virginia Constitution's Article VI, Section 51 (Budget and supplementary appropriation "
        "bills) or an authoritative page reproducing that text."
    )
    await evaluator.verify(
        claim=claim_url,
        node=leaf_url,
        sources=url_val,
        additional_instruction=(
            "Verify that the provided URL points to the WV Constitution provision governing veto overrides for "
            "appropriations/budget/supplementary appropriation bills (Article VI, Section 51). "
            "If no URL was provided in the answer, mark incorrect."
        ),
    )

    # Leaf: threshold formula (two-thirds)
    leaf_thresh = evaluator.add_leaf(
        id="appropriations_bill_threshold",
        desc="States the vote threshold for appropriations bills is two-thirds of the members elected to each house.",
        parent=node,
        critical=True,
    )
    claim_thresh = (
        "Under WV Const. Article VI, Section 51, overriding a veto for appropriations/budget bills requires "
        "two-thirds of the members elected to each house."
    )
    await evaluator.verify(
        claim=claim_thresh,
        node=leaf_thresh,
        sources=url_val,
        additional_instruction=(
            "Confirm that the page states 'two-thirds of the members elected to each house' or equivalent language. "
            "Allow minor phrasing variations."
        ),
    )

    # Leaf: specific minimum House votes (two-thirds of 100 -> 67)
    leaf_house = evaluator.add_leaf(
        id="appropriations_bill_house_min_votes",
        desc="Gives a specific minimum House vote count consistent with two-thirds of 100 members elected (i.e., equals ceil(100 * 2/3)).",
        parent=node,
        critical=True,
    )
    claim_house = (
        f"The answer states a specific minimum House vote count for overriding an appropriations-bill veto, and it is "
        f"{APPROPRIATIONS_HOUSE_MIN} out of 100 members."
    )
    await evaluator.verify(
        claim=claim_house,
        node=leaf_house,
        additional_instruction=(
            "Pass only if the answer explicitly provides a numeric House minimum and it equals 67 "
            "(two-thirds of 100, rounded up). If the answer omitted a number, mark incorrect."
        ),
    )

    # Leaf: specific minimum Senate votes (two-thirds of 34 -> 23)
    leaf_senate = evaluator.add_leaf(
        id="appropriations_bill_senate_min_votes",
        desc="Gives a specific minimum Senate vote count consistent with two-thirds of 34 members elected (i.e., equals ceil(34 * 2/3)).",
        parent=node,
        critical=True,
    )
    claim_senate = (
        f"The answer states a specific minimum Senate vote count for overriding an appropriations-bill veto, and it is "
        f"{APPROPRIATIONS_SENATE_MIN} out of {SENATE_MEMBERS} members."
    )
    await evaluator.verify(
        claim=claim_senate,
        node=leaf_senate,
        additional_instruction=(
            "Pass only if the answer explicitly provides a numeric Senate minimum and it equals 23 "
            "(two-thirds of 34, rounded up). If the answer omitted a number, mark incorrect."
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
    Evaluate an answer for the West Virginia veto override requirements task.
    """
    evaluator = Evaluator()

    # Note: Although the rubric marks the root as critical, obj_task_eval enforces that critical parents must have all
    # critical children. We use a non-critical root to allow partial credit on sub-requirements while enforcing
    # critical constraints at appropriate child nodes.
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

    # Record the adjustment decision as custom info for transparency
    evaluator.add_custom_info(
        info={"root_critical_adjusted": True, "reason": "Parent critical requires all children critical in framework; using non-critical root allows partial scoring while enforcing critical checks within subtrees."},
        info_type="critical_policy",
        info_name="critical_adjustment"
    )

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_veto_info(),
        template_class=WVVetoExtraction,
        extraction_name="wv_veto_info",
    )

    # Add ground truth references
    evaluator.add_ground_truth({
        "expected_regular_citation": EXPECTED_REGULAR_CITATION,
        "expected_appropriations_citation": EXPECTED_APPROPRIATIONS_CITATION,
        "thresholds": {
            "regular": "majority of members elected (strictly more than half)",
            "appropriations": "two-thirds of members elected",
        },
        "computed_min_votes": {
            "house_regular": REGULAR_HOUSE_MIN,
            "senate_regular": REGULAR_SENATE_MIN,
            "house_appropriations": APPRECIATIONS_HOUSE_MIN if 'APPRECIATIONS_HOUSE_MIN' in globals() else APPROPRIATIONS_HOUSE_MIN,
            "senate_appropriations": APPROPRIATIONS_SENATE_MIN,
            "house_members": HOUSE_MEMBERS,
            "senate_members": SENATE_MEMBERS,
        }
    })

    # Build and verify general constraints
    await verify_general_constraints(
        evaluator=evaluator,
        parent_node=root,
        general=extracted.general,
    )

    # Build and verify regular bill requirements
    await verify_regular_bill_requirements(
        evaluator=evaluator,
        parent_node=root,
        reg=extracted.regular_bill,
    )

    # Build and verify appropriations bill requirements
    await verify_appropriations_bill_requirements(
        evaluator=evaluator,
        parent_node=root,
        appr=extracted.appropriations_bill,
    )

    # Return final structured summary
    return evaluator.get_summary()