import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "mt_whitney_2026_permit_lottery"
TASK_DESCRIPTION = (
    "A group of experienced hikers from Colorado is planning to summit Mount Whitney in California via the classic "
    "Mt. Whitney Trail during August 2026. They intend to do an overnight backpacking trip and need to apply through "
    "the annual permit lottery system. To prepare their application, they need the following specific information:\n\n"
    "1. What are the exact start and end dates of the lottery application period for the 2026 season?\n"
    "2. On what date will they learn whether they won the lottery?\n"
    "3. If they win, by what date and time must they claim their awarded permit to secure it?\n"
    "4. What is the total cost per person they must pay, including all required fees?\n"
    "5. If one of their group members has a scheduling conflict after winning, can they transfer the permit to a replacement hiker?\n\n"
    "Provide detailed answers to all five questions with supporting URL references from official sources."
)

# Ground-truth expectations used for simple checks against the agent's answer text.
EXPECTED_Q1_START = "February 1, 2026"
EXPECTED_Q1_END = "March 1, 2026"
EXPECTED_Q2_RESULTS_DATE = "March 15, 2026"
EXPECTED_Q3_CLAIM_DEADLINE = "April 21, 2026 at 9:00 PM Pacific Time"
EXPECTED_Q3_CLAIM_DEADLINE_ALT_ET = "midnight Eastern Time"
EXPECTED_Q4_APP_FEE = "$6"  # non-refundable application/reservation fee per application
EXPECTED_Q4_REC_FEE = "$15"  # recreation fee per person
EXPECTED_Q5_NON_TRANSFERABLE = "Permits cannot be resold or transferred"


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class Q1Info(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Q2Info(BaseModel):
    results_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Q3Info(BaseModel):
    claim_deadline: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Q4Info(BaseModel):
    total_cost_per_person: Optional[str] = None
    application_fee: Optional[str] = None
    recreation_fee_per_person: Optional[str] = None
    breakdown_explanation: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Q5Info(BaseModel):
    transferable: Optional[str] = None
    explanation: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class MWPermitExtraction(BaseModel):
    q1: Optional[Q1Info] = None
    q2: Optional[Q2Info] = None
    q3: Optional[Q3Info] = None
    q4: Optional[Q4Info] = None
    q5: Optional[Q5Info] = None
    global_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_mw_permit_info() -> str:
    return """
    Extract the specific information the answer provides for each of the five questions about the Mount Whitney 2026 permit lottery.
    You must only extract what is explicitly stated in the answer. If something is missing, set it to null.
    Also extract any explicit URL(s) the answer cites for each question separately (q1..q5), plus any general citations as global_sources.

    Return JSON with the following structure:
    {
      "q1": {
        "start_date": string | null,
        "end_date": string | null,
        "sources": string[]   // URLs explicitly cited for Q1 in the answer
      },
      "q2": {
        "results_date": string | null,
        "sources": string[]   // URLs explicitly cited for Q2 in the answer
      },
      "q3": {
        "claim_deadline": string | null,  // include both date and time if stated; keep timezone wording if given
        "sources": string[]               // URLs explicitly cited for Q3
      },
      "q4": {
        "total_cost_per_person": string | null,             // exact per-person total as stated, if any
        "application_fee": string | null,                   // e.g., "$6" reservation/application fee per application
        "recreation_fee_per_person": string | null,         // e.g., "$15 per person"
        "breakdown_explanation": string | null,             // any explanation showing how per-person cost was computed
        "sources": string[]                                 // URLs explicitly cited for Q4
      },
      "q5": {
        "transferable": string | null,      // e.g., "no", "not transferable", or a clear statement of transferability
        "explanation": string | null,       // any explanation about permit holder/alternate leader policy
        "sources": string[]                 // URLs explicitly cited for Q5
      },
      "global_sources": string[]            // other URLs cited but not tied to a specific question
    }

    SPECIAL URL RULES:
    - Only extract URLs explicitly present in the answer text (including markdown links). Do not invent URLs.
    - Include full URLs with protocol. If protocol is missing, prepend "http://".
    - For each question's "sources", include only URLs the answer appears to cite for that specific information.
    - Place any remaining citations into "global_sources".

    Notes:
    - Dates may appear in different formats (e.g., "Feb 1, 2026", "February 1, 2026", "2/1/2026"). Extract exactly as written.
    - Timezone information (e.g., "PT", "PDT", "Pacific Time", "ET", "Eastern Time") should be preserved in the extracted string if included.
    - If multiple URLs are cited for a question, extract all of them.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(local: Optional[List[str]], global_sources: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for coll in (local or []), (global_sources or []):
        for s in coll:
            if not isinstance(s, str):
                continue
            url = s.strip()
            if not url:
                continue
            if url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


def _collect_all_sources(extracted: MWPermitExtraction) -> List[str]:
    all_srcs: List[str] = []
    for block in [extracted.q1, extracted.q2, extracted.q3, extracted.q4, extracted.q5]:
        if block and getattr(block, "sources", None):
            all_srcs.extend(block.sources)
    all_srcs.extend(extracted.global_sources or [])
    # de-duplicate
    deduped = []
    seen = set()
    for url in all_srcs:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def _has_official_source(urls: List[str]) -> bool:
    official_domains = [
        "recreation.gov",
        "fs.usda.gov",   # USDA Forest Service (Inyo National Forest)
        "usda.gov",
        "nps.gov",
        "ca.gov",
    ]
    for u in urls:
        low = u.lower()
        for dom in official_domains:
            if dom in low:
                return True
    return False


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_q1(evaluator: Evaluator, parent_node, extracted: MWPermitExtraction) -> None:
    node = evaluator.add_parallel(
        id="Q1_Application_Period_Dates",
        desc="States the exact lottery application start and end dates for the 2026 season (Feb 1, 2026 through Mar 1, 2026).",
        parent=parent_node,
        critical=True,
    )

    q1 = extracted.q1 or Q1Info()
    q1_sources = _merge_sources(q1.sources, extracted.global_sources)

    # Critical: at least one supporting source URL for Q1
    evaluator.add_custom_node(
        result=len(q1_sources) > 0,
        id="Q1_sources_provided",
        desc="Q1 includes at least one supporting URL citation.",
        parent=node,
        critical=True,
    )

    # Critical: the answer states the expected dates
    leaf_answer_states = evaluator.add_leaf(
        id="Q1_answer_states_expected_dates",
        desc="Answer states: application period is February 1, 2026 through March 1, 2026 (inclusive).",
        parent=node,
        critical=True,
    )
    claim_answer = (
        "In the answer, the agent explicitly states that the Mt. Whitney permit lottery application period for the 2026 season "
        "runs from February 1, 2026 through March 1, 2026 (inclusive)."
    )
    await evaluator.verify(
        claim=claim_answer,
        node=leaf_answer_states,
        additional_instruction=(
            "Judge only whether the answer text itself states these exact start and end dates. "
            "Allow equivalent formats (e.g., 'Feb 1–Mar 1, 2026', '2/1/2026 to 3/1/2026'). "
            "Both dates must match; otherwise mark incorrect."
        ),
    )

    # Critical: the sources support those dates
    leaf_sources_support = evaluator.add_leaf(
        id="Q1_sources_support_dates",
        desc="Cited source(s) support that the application period is Feb 1, 2026 through Mar 1, 2026.",
        parent=node,
        critical=True,
    )
    claim_sources = (
        "The Mt. Whitney permit lottery application period for the 2026 season runs from February 1, 2026 through March 1, 2026."
    )
    await evaluator.verify(
        claim=claim_sources,
        node=leaf_sources_support,
        sources=q1_sources,
        additional_instruction=(
            "Only pass if at least one cited webpage explicitly shows these start and end dates. "
            "If the page is irrelevant or inaccessible, mark as not supported."
        ),
    )


async def verify_q2(evaluator: Evaluator, parent_node, extracted: MWPermitExtraction) -> None:
    node = evaluator.add_parallel(
        id="Q2_Results_Announcement_Date",
        desc="States the date lottery results are posted (Mar 15, 2026).",
        parent=parent_node,
        critical=True,
    )

    q2 = extracted.q2 or Q2Info()
    q2_sources = _merge_sources(q2.sources, extracted.global_sources)

    evaluator.add_custom_node(
        result=len(q2_sources) > 0,
        id="Q2_sources_provided",
        desc="Q2 includes at least one supporting URL citation.",
        parent=node,
        critical=True,
    )

    leaf_answer_states = evaluator.add_leaf(
        id="Q2_answer_states_expected_date",
        desc="Answer states: lottery results are posted on March 15, 2026.",
        parent=node,
        critical=True,
    )
    claim_answer = (
        "In the answer, the agent explicitly states that Mt. Whitney lottery results are posted on March 15, 2026."
    )
    await evaluator.verify(
        claim=claim_answer,
        node=leaf_answer_states,
        additional_instruction=(
            "Judge only whether the answer text itself states March 15, 2026 (format variants allowed)."
        ),
    )

    leaf_sources_support = evaluator.add_leaf(
        id="Q2_sources_support_date",
        desc="Cited source(s) support that results are posted on March 15, 2026.",
        parent=node,
        critical=True,
    )
    claim_sources = "Mt. Whitney lottery results are posted on March 15, 2026."
    await evaluator.verify(
        claim=claim_sources,
        node=leaf_sources_support,
        sources=q2_sources,
        additional_instruction=(
            "Only pass if the cited page explicitly shows the March 15, 2026 results announcement date."
        ),
    )


async def verify_q3(evaluator: Evaluator, parent_node, extracted: MWPermitExtraction) -> None:
    node = evaluator.add_parallel(
        id="Q3_Claim_Deadline_Date_Time",
        desc="States the deadline to claim an awarded permit, including date and time with timezone context (Apr 21, 2026 at 9:00 PM Pacific Time / midnight Eastern Time).",
        parent=parent_node,
        critical=True,
    )

    q3 = extracted.q3 or Q3Info()
    q3_sources = _merge_sources(q3.sources, extracted.global_sources)

    evaluator.add_custom_node(
        result=len(q3_sources) > 0,
        id="Q3_sources_provided",
        desc="Q3 includes at least one supporting URL citation.",
        parent=node,
        critical=True,
    )

    leaf_answer_states = evaluator.add_leaf(
        id="Q3_answer_states_expected_deadline",
        desc="Answer states: claim deadline is April 21, 2026 at 9:00 PM Pacific Time (midnight Eastern Time).",
        parent=node,
        critical=True,
    )
    claim_answer = (
        "In the answer, the agent explicitly states that the deadline to claim an awarded permit is April 21, 2026 "
        "at 9:00 PM Pacific Time (which corresponds to midnight Eastern Time)."
    )
    await evaluator.verify(
        claim=claim_answer,
        node=leaf_answer_states,
        additional_instruction=(
            "Judge only whether the answer text itself states the April 21, 2026 9:00 PM Pacific Time deadline, "
            "and includes a midnight Eastern Time equivalent phrasing (e.g., 12:00 AM ET). "
            "Allow minor wording variations (PT/PDT; ET/EST/EDT) as long as the intent is clear."
        ),
    )

    leaf_sources_support = evaluator.add_leaf(
        id="Q3_sources_support_deadline",
        desc="Cited source(s) support the deadline (Apr 21, 2026 at 9:00 PM PT).",
        parent=node,
        critical=True,
    )
    claim_sources = (
        "The deadline to claim an awarded Mt. Whitney permit from the lottery is April 21, 2026 at 9:00 PM Pacific Time."
    )
    await evaluator.verify(
        claim=claim_sources,
        node=leaf_sources_support,
        sources=q3_sources,
        additional_instruction=(
            "Only pass if the cited page explicitly shows the claim deadline as April 21, 2026 at 9:00 PM Pacific Time "
            "or an equivalent clear expression."
        ),
    )


async def verify_q4(evaluator: Evaluator, parent_node, extracted: MWPermitExtraction) -> None:
    node = evaluator.add_parallel(
        id="Q4_Total_Cost_Per_Person",
        desc="Provides the total cost per person including all required fees, consistent with fee structure ($6 application + $15 per person).",
        parent=parent_node,
        critical=True,
    )

    q4 = extracted.q4 or Q4Info()
    q4_sources = _merge_sources(q4.sources, extracted.global_sources)

    evaluator.add_custom_node(
        result=len(q4_sources) > 0,
        id="Q4_sources_provided",
        desc="Q4 includes at least one supporting URL citation.",
        parent=node,
        critical=True,
    )

    # The answer should state fee structure ($6 per application + $15 per person)
    leaf_answer_fee_structure = evaluator.add_leaf(
        id="Q4_answer_states_fee_structure",
        desc="Answer states fee structure: $6 non-refundable application fee per application + $15 recreation fee per person.",
        parent=node,
        critical=True,
    )
    claim_answer_structure = (
        "In the answer, the agent explicitly states that there is a $6 non-refundable reservation/application fee per application "
        "and a $15 recreation fee per person."
    )
    await evaluator.verify(
        claim=claim_answer_structure,
        node=leaf_answer_fee_structure,
        additional_instruction="Judge only based on the answer text. Accept reasonable wording variants as long as both fees and their scopes are clear.",
    )

    # The cited sources should support those fees
    leaf_sources_support_fees = evaluator.add_leaf(
        id="Q4_sources_support_fee_structure",
        desc="Cited source(s) support: $6 application fee per application and $15 per person recreation fee.",
        parent=node,
        critical=True,
    )
    claim_sources_structure = (
        "The Mt. Whitney permit process charges a $6 non-refundable application/reservation fee per lottery application and a $15 per person recreation fee."
    )
    await evaluator.verify(
        claim=claim_sources_structure,
        node=leaf_sources_support_fees,
        sources=q4_sources,
        additional_instruction="Only pass if a cited official page explicitly lists these fees with the same amounts and scope.",
    )

    # Ensure the answer provides a per-person total or a clear explanation including both required fees
    per_person_total_provided = (
        (q4.total_cost_per_person is not None and str(q4.total_cost_per_person).strip() != "") or
        ((q4.application_fee is not None and str(q4.application_fee).strip() != "") and
         (q4.recreation_fee_per_person is not None and str(q4.recreation_fee_per_person).strip() != ""))
    )
    evaluator.add_custom_node(
        result=per_person_total_provided,
        id="Q4_per_person_total_provided",
        desc="Answer provides a per-person total or clearly explains the per-person total including all required fees.",
        parent=node,
        critical=True,
    )

    # If a specific per-person total is given, ensure it's consistent with the fee structure
    leaf_total_consistent = evaluator.add_leaf(
        id="Q4_per_person_total_consistent",
        desc="If a numeric per-person total is given, it is consistent with: $15 per person, and $6 is a one-time application fee (not per person).",
        parent=node,
        critical=True,
    )
    claim_total_consistency = (
        "In the answer, the per-person total (if explicitly stated as a number) is consistent with the fee structure: "
        "$15 per person recreation fee and a separate $6 non-refundable application fee per application (not per person). "
        "If only the breakdown is provided without a single numeric total, consider this consistent."
    )
    await evaluator.verify(
        claim=claim_total_consistency,
        node=leaf_total_consistent,
        additional_instruction=(
            "Judge only from the answer text. Accept if the answer either: "
            "(a) gives a numeric per-person total consistent with $15 per person and treats $6 as per application, or "
            "(b) clearly explains the per-person cost includes the $15 per person and mentions the required $6 application fee separately."
        ),
    )


async def verify_q5(evaluator: Evaluator, parent_node, extracted: MWPermitExtraction) -> None:
    node = evaluator.add_parallel(
        id="Q5_Permit_Transferability",
        desc="Answers whether the permit can be transferred to a replacement hiker, consistent with no resale/transfer policy.",
        parent=parent_node,
        critical=True,
    )

    q5 = extracted.q5 or Q5Info()
    q5_sources = _merge_sources(q5.sources, extracted.global_sources)

    evaluator.add_custom_node(
        result=len(q5_sources) > 0,
        id="Q5_sources_provided",
        desc="Q5 includes at least one supporting URL citation.",
        parent=node,
        critical=True,
    )

    leaf_answer_states = evaluator.add_leaf(
        id="Q5_answer_states_non_transferable",
        desc="Answer states permits cannot be resold or transferred to a replacement hiker (only permit holder or designated alternate may use).",
        parent=node,
        critical=True,
    )
    claim_answer = (
        "In the answer, the agent explicitly states that Mt. Whitney permits cannot be resold or transferred to a replacement hiker. "
        "Only the named permit holder or a pre-designated alternate leader may use the permit."
    )
    await evaluator.verify(
        claim=claim_answer,
        node=leaf_answer_states,
        additional_instruction=(
            "Judge only from the answer text. Accept wording variants conveying the same non-transferable policy. "
            "Mention of pre-designated alternate leader is acceptable."
        ),
    )

    leaf_sources_support = evaluator.add_leaf(
        id="Q5_sources_support_non_transferable",
        desc="Cited source(s) support the non-transferability (no resale/transfer) policy.",
        parent=node,
        critical=True,
    )
    claim_sources = (
        "Mt. Whitney permits cannot be resold or transferred to a different person; only the permit holder or a pre-designated alternate may use the permit."
    )
    await evaluator.verify(
        claim=claim_sources,
        node=leaf_sources_support,
        sources=q5_sources,
        additional_instruction="Only pass if an official cited page explicitly states or clearly implies no transfer/resale.",
    )


async def verify_official_citations(evaluator: Evaluator, parent_node, extracted: MWPermitExtraction) -> None:
    node = evaluator.add_parallel(
        id="Official_URL_Citations",
        desc="Provides supporting URL reference(s) from official sources for the information given (e.g., official agency pages and/or Recreation.gov).",
        parent=parent_node,
        critical=True,
    )

    all_sources = _collect_all_sources(extracted)
    evaluator.add_custom_node(
        result=len(all_sources) > 0,
        id="Citations_any_present",
        desc="At least one supporting URL citation is present in the answer.",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_official_source(all_sources),
        id="Citations_official_present",
        desc="At least one citation is an official source (e.g., Recreation.gov or USDA Forest Service).",
        parent=node,
        critical=True,
    )

    leaf_answer_lists_citations = evaluator.add_leaf(
        id="Citations_listed_in_answer",
        desc="Answer includes explicit supporting URL references for its statements.",
        parent=node,
        critical=True,
    )
    claim_answer = "In the answer, the agent includes explicit supporting URL references for the information provided."
    await evaluator.verify(
        claim=claim_answer,
        node=leaf_answer_lists_citations,
        additional_instruction="Judge only from the answer text. Accept citations listed inline or in a references section.",
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_mw_permit_info(),
        template_class=MWPermitExtraction,
        extraction_name="mw_permit_extraction",
    )

    # Add expected values as ground-truth context (for transparency in report)
    evaluator.add_ground_truth({
        "expected_q1": {
            "start_date": EXPECTED_Q1_START,
            "end_date": EXPECTED_Q1_END
        },
        "expected_q2": {
            "results_date": EXPECTED_Q2_RESULTS_DATE
        },
        "expected_q3": {
            "claim_deadline_pt": EXPECTED_Q3_CLAIM_DEADLINE,
            "alt_et_note": EXPECTED_Q3_CLAIM_DEADLINE_ALT_ET
        },
        "expected_q4": {
            "application_fee": EXPECTED_Q4_APP_FEE + " per application (non-refundable)",
            "recreation_fee_per_person": EXPECTED_Q4_REC_FEE + " per person"
        },
        "expected_q5": {
            "transferability": EXPECTED_Q5_NON_TRANSFERABLE
        }
    }, gt_type="ground_truth")

    # Build a critical, parallel top-level node that mirrors the rubric's main node
    mw_root = evaluator.add_parallel(
        id="Mount_Whitney_Permit_Lottery_Information",
        desc="Verify complete and accurate answers to the five Mount Whitney 2026 permit-lottery questions, including official supporting URLs.",
        parent=root,
        critical=True,
    )

    # Run verifications for each question and the citations node
    await verify_q1(evaluator, mw_root, extracted)
    await verify_q2(evaluator, mw_root, extracted)
    await verify_q3(evaluator, mw_root, extracted)
    await verify_q4(evaluator, mw_root, extracted)
    await verify_q5(evaluator, mw_root, extracted)
    await verify_official_citations(evaluator, mw_root, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()