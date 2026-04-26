import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# -----------------------------------------------------------------------------
# Task Metadata
# -----------------------------------------------------------------------------
TASK_ID = "ivy_athlete_aid_policy"
TASK_DESCRIPTION = (
    "A high school student-athlete is considering applying to Ivy League universities and needs to understand the financial aid policies specific to athletes. "
    "Research and provide the following information: (1) Are athletic scholarships available at Ivy League schools? (2) What type of financial aid is offered to "
    "student-athletes at Ivy League schools instead? (3) Which office or department has the authority to determine and issue financial aid packages to students?"
)

EIGHT_IVY_SCHOOLS = [
    "Brown University",
    "Columbia University",
    "Cornell University",
    "Dartmouth College",
    "Harvard University",
    "University of Pennsylvania",
    "Princeton University",
    "Yale University",
]

# -----------------------------------------------------------------------------
# Extraction Models
# -----------------------------------------------------------------------------
class PolicyStatements(BaseModel):
    """
    Extracted statements (verbatim excerpts) and URLs from the agent's answer.
    All text fields should contain the exact sentence/phrase the answer used, if present.
    """
    athletic_scholarships_statement: Optional[str] = None
    merit_scholarships_statement: Optional[str] = None
    need_based_aid_statement: Optional[str] = None
    policy_applies_all_eight_statement: Optional[str] = None

    authority_office_statement: Optional[str] = None
    coach_may_request_estimate_statement: Optional[str] = None
    coach_cannot_determine_or_commit_statement: Optional[str] = None
    verbal_commitment_not_offer_statement: Optional[str] = None

    policy_urls: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction Prompt
# -----------------------------------------------------------------------------
def prompt_extract_policy_statements() -> str:
    return """
    Your task is to extract exact statements and any policy-related URLs from the provided answer text about Ivy League financial aid policies for student-athletes.

    Extract the following fields:
    1) athletic_scholarships_statement: The exact sentence/phrase indicating whether Ivy League schools offer athletic scholarships. If it says there are none (or explicitly prohibits), extract that. Otherwise, return null.
    2) merit_scholarships_statement: The exact sentence/phrase indicating whether Ivy League schools offer merit-based (e.g., academic/merit) scholarships. Return null if not mentioned.
    3) need_based_aid_statement: The exact sentence/phrase indicating that Ivy League financial aid is need-based (and/or need-based only). Return null if not mentioned.
    4) policy_applies_all_eight_statement: The exact sentence/phrase indicating that the policy (no athletic/merit scholarships, need-based only) applies to all eight Ivy League institutions as a league-wide policy. Return null if not mentioned.

    5) authority_office_statement: The exact sentence/phrase indicating which office/department determines and issues financial aid packages (e.g., "only the Financial Aid Office determines and issues financial aid"). Return null if not mentioned.
    6) coach_may_request_estimate_statement: The exact sentence/phrase indicating that coaches may request financial aid estimates (e.g., "pre-reads") for prospective student-athletes. Return null if not mentioned.
    7) coach_cannot_determine_or_commit_statement: The exact sentence/phrase stating coaches cannot determine aid amounts or make financial aid commitments. Return null if not mentioned.
    8) verbal_commitment_not_offer_statement: The exact sentence/phrase stating that a coach's verbal commitment/support is not a financial aid offer or guarantee. Return null if not mentioned.

    9) policy_urls: Extract all URLs explicitly mentioned in the answer that are used as sources or references for Ivy League financial aid/athletics aid policy. 
       - Include Ivy League official pages or university Financial Aid/Athletics policy pages if present.
       - Only include URLs actually present in the answer (plain or markdown links). Do not invent any.
       - Return a list; if none are present, return an empty list.

    Return a JSON object containing all fields listed above. If a specific statement is not present in the answer, set it to null.
    """


# -----------------------------------------------------------------------------
# Verification Helpers
# -----------------------------------------------------------------------------
def _normalize_sources_for_verification(statements: PolicyStatements) -> List[str]:
    # Use all policy-related URLs extracted from the answer
    return statements.policy_urls or []


# -----------------------------------------------------------------------------
# Build Verification Tree and Run Checks
# -----------------------------------------------------------------------------
async def _verify_policy_nodes(evaluator: Evaluator, root, statements: PolicyStatements) -> None:
    """
    Build the verification tree based on the rubric and run verifications.
    This implements a faithful and debuggable tree with single-purpose leaf checks.
    """
    sources_list = _normalize_sources_for_verification(statements)

    # ----------------------------- Branch 1 -----------------------------------
    # ScholarshipAndAidPolicy (Critical, Parallel)
    policy_node = evaluator.add_parallel(
        id="ScholarshipAndAidPolicy",
        desc="Correctly describes whether scholarships exist and what aid replaces them for student-athletes in the Ivy League.",
        parent=root,
        critical=True,
    )

    # NoAthleticOrMeritScholarships (split into two single-purpose leaves under a parallel critical node)
    no_ath_or_merit_parent = evaluator.add_parallel(
        id="NoAthleticOrMeritPolicy",
        desc="States that Ivy League schools do not offer athletic scholarships and do not offer merit-based scholarships.",
        parent=policy_node,
        critical=True,
    )

    # Leaf: No athletic scholarships
    leaf_no_athletic = evaluator.add_leaf(
        id="NoAthleticScholarships",
        desc="Ivy League schools do not offer athletic scholarships.",
        parent=no_ath_or_merit_parent,
        critical=True,
    )
    claim_no_athletic = "Ivy League schools do not offer athletic scholarships."
    await evaluator.verify(
        claim=claim_no_athletic,
        node=leaf_no_athletic,
        sources=sources_list,
        additional_instruction=(
            "Check the provided webpage(s) for explicit statements such as 'Ivy League schools do not offer athletic scholarships' "
            "or 'no athletic scholarships are awarded in the Ivy League.' Any single supporting source is sufficient."
        ),
    )

    # Leaf: No merit scholarships
    leaf_no_merit = evaluator.add_leaf(
        id="NoMeritScholarships",
        desc="Ivy League schools do not offer merit-based (academic) scholarships.",
        parent=no_ath_or_merit_parent,
        critical=True,
    )
    claim_no_merit = "Ivy League schools do not offer merit-based scholarships."
    await evaluator.verify(
        claim=claim_no_merit,
        node=leaf_no_merit,
        sources=sources_list,
        additional_instruction=(
            "Look for language indicating 'no merit/academic scholarships' or 'no merit-based aid' in the Ivy League. "
            "If a page states only need-based aid is offered and that merit/academic scholarships are not provided, that supports this claim."
        ),
    )

    # Leaf: Need-based aid only
    leaf_need_only = evaluator.add_leaf(
        id="NeedBasedAidOnly",
        desc="Identifies need-based financial aid as the exclusive form of financial support for students, including student-athletes.",
        parent=policy_node,
        critical=True,
    )
    claim_need_only = "Ivy League schools offer only need-based financial aid (not athletic or merit scholarships)."
    await evaluator.verify(
        claim=claim_need_only,
        node=leaf_need_only,
        sources=sources_list,
        additional_instruction=(
            "Verify that the page(s) clearly state Ivy League aid is need-based only. "
            "Language like 'financial aid is based on demonstrated financial need' and explicitly excluding merit/athletic scholarships is supportive."
        ),
    )

    # Leaf: Policy applies across all eight Ivy League institutions
    leaf_all_eight = evaluator.add_leaf(
        id="PolicyAppliesToAllEight",
        desc="The no-athletic-scholarship/need-based-only policy applies across all eight Ivy League member institutions.",
        parent=policy_node,
        critical=True,
    )
    claim_all_eight = (
        "The Ivy League's no-athletic-scholarship and need-based-only aid policy is a league-wide policy that applies to all eight Ivy League institutions."
    )
    await evaluator.verify(
        claim=claim_all_eight,
        node=leaf_all_eight,
        sources=sources_list,
        additional_instruction=(
            "Support can include phrases like 'the Ivy League does not award athletic scholarships' or 'Ivy League institutions provide only need-based aid' "
            "that clearly indicate a league-wide rule, not a single-school exception."
        ),
    )

    # ----------------------------- Branch 2 -----------------------------------
    # FinancialAidAuthorityAndCoachRole (Critical, Parallel)
    authority_node = evaluator.add_parallel(
        id="FinancialAidAuthorityAndCoachRole",
        desc="Correctly identifies who has authority over aid packages and clarifies coaches’ lack of authority/commitment power.",
        parent=root,
        critical=True,
    )

    # Leaf: Financial Aid Office sole authority
    leaf_fa_office = evaluator.add_leaf(
        id="FinancialAidOfficeSoleAuthority",
        desc="Only the Financial Aid Office determines and issues financial aid packages.",
        parent=authority_node,
        critical=True,
    )
    claim_fa_office = "Only the university's Financial Aid Office determines and issues financial aid packages; athletics staff and coaches do not determine aid."
    await evaluator.verify(
        claim=claim_fa_office,
        node=leaf_fa_office,
        sources=sources_list,
        additional_instruction=(
            "Seek explicit wording that the Financial Aid Office is the sole authority on financial aid decisions or awards."
        ),
    )

    # Leaf: Coach may request estimate (pre-read)
    leaf_coach_estimate = evaluator.add_leaf(
        id="CoachMayRequestEstimate",
        desc="Coaches may request financial aid estimates for prospective student-athletes.",
        parent=authority_node,
        critical=True,
    )
    claim_coach_estimate = "Coaches may request financial aid estimates (such as financial aid 'pre-reads') for prospective student-athletes."
    await evaluator.verify(
        claim=claim_coach_estimate,
        node=leaf_coach_estimate,
        sources=sources_list,
        additional_instruction=(
            "Look for terms like 'pre-read' or 'estimate' of financial aid that coaches can request from the Financial Aid Office for prospective student-athletes."
        ),
    )

    # Leaf: Coach cannot determine or commit aid
    leaf_coach_no_commit = evaluator.add_leaf(
        id="CoachCannotDetermineOrCommitAid",
        desc="Coaches cannot determine aid amounts or make financial aid commitments.",
        parent=authority_node,
        critical=True,
    )
    claim_coach_no_commit = "Coaches cannot determine financial aid amounts or make any binding financial aid commitments."
    await evaluator.verify(
        claim=claim_coach_no_commit,
        node=leaf_coach_no_commit,
        sources=sources_list,
        additional_instruction=(
            "Confirm that authority over aid amounts and offers lies with the Financial Aid Office, not coaches."
        ),
    )

    # Leaf: Verbal commitment is not an aid offer
    leaf_verbal_not_offer = evaluator.add_leaf(
        id="VerbalCommitmentNotAidOffer",
        desc="A coach’s verbal commitment/support is not a financial aid offer or guarantee.",
        parent=authority_node,
        critical=True,
    )
    claim_verbal_not_offer = "A coach’s verbal commitment or support does not constitute a financial aid offer or guarantee."
    await evaluator.verify(
        claim=claim_verbal_not_offer,
        node=leaf_verbal_not_offer,
        sources=sources_list,
        additional_instruction=(
            "Seek explicit disclaimers that only official communications from the Financial Aid Office constitute aid offers; "
            "verbal commitments from coaches are not binding aid offers."
        ),
    )


# -----------------------------------------------------------------------------
# Main Evaluation Entry
# -----------------------------------------------------------------------------
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
    Entry point to evaluate an agent's answer for Ivy League student-athlete financial aid policies.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Two major branches checked independently
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

    # Extract statements and policy URLs from the answer
    statements: PolicyStatements = await evaluator.extract(
        prompt=prompt_extract_policy_statements(),
        template_class=PolicyStatements,
        extraction_name="policy_statements",
    )

    # Record helpful ground-truth context (not used for scoring, only for report)
    evaluator.add_ground_truth(
        {
            "ivy_member_schools": EIGHT_IVY_SCHOOLS,
            "expected_policy_elements": [
                "No athletic scholarships",
                "No merit-based (academic) scholarships",
                "Financial aid is need-based only",
                "Policy applies league-wide across all eight Ivy League institutions",
                "Financial Aid Office determines and issues financial aid packages",
                "Coaches may request estimates (pre-reads)",
                "Coaches cannot determine or commit aid",
                "Coach verbal commitment is not a financial aid offer/guarantee",
            ],
        },
        gt_type="policy_expectations",
    )

    # Add custom info: count of URLs referenced in the answer
    evaluator.add_custom_info(
        info={"policy_url_count": len(statements.policy_urls), "policy_urls": statements.policy_urls},
        info_type="extracted_sources",
        info_name="policy_sources_overview",
    )

    # Build tree and run verifications according to rubric
    await _verify_policy_nodes(evaluator, root, statements)

    # Return structured summary
    return evaluator.get_summary()