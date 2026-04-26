import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "mcps_tn_transition_research"
TASK_DESCRIPTION = (
    "A certified teacher with 4 years of full-time teaching experience (earned between 2019-2023) is planning to relocate from Tennessee to Maryland. "
    "They hold a bachelor's degree in education from a regionally-accredited university and are seeking information about substitute teaching opportunities at Montgomery County Public Schools (MCPS). "
    "Research and provide the following information: "
    "(1) What is the minimum educational credential required to qualify as a substitute teacher at MCPS? "
    "(2) What is the job posting ID or reference number for MCPS substitute teacher positions for the 2025-2026 school year? "
    "(3) What is the daily compensation rate for substitute teachers at MCPS? "
    "(4) What specific Maryland state clearance (beyond standard background checks) is required for MCPS substitute teachers? "
    "(5) How many years of teaching experience are required before a teacher can apply for a Professional Teacher License in Tennessee? "
    "(6) Based on the teacher's 4 years of experience, are they eligible to apply for a Tennessee Professional Teacher License before relocating? "
    "For each answer, provide the supporting URL reference from official sources."
)

# Optional ground truth hints to help audit
GROUND_TRUTH_HINTS = {
    "Q1": "MCPS typically requires either an associate degree or at least 60 college credits, often specifying from an accredited institution.",
    "Q4": "Common Maryland-specific requirement referenced by MCPS is Child Protective Services (CPS) clearance.",
    "Q5": "Tennessee Professional Teacher License requires 3 years of qualifying teaching experience.",
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Q1Info(BaseModel):
    min_credential: Optional[str] = None  # e.g., "associate degree or 60 college credits"
    mentions_accredited: Optional[bool] = None  # whether the answer mentions accreditation
    source_urls: List[str] = Field(default_factory=list)  # MCPS official sources


class Q2Info(BaseModel):
    posting_id: Optional[str] = None  # job posting ID or reference number
    school_year: Optional[str] = None  # e.g., "2025-2026"
    posting_url: Optional[str] = None  # direct link to MCPS job posting


class Q3Info(BaseModel):
    daily_rate: Optional[str] = None  # e.g., "$140 per day" or "not publicly available"
    source_url: Optional[str] = None  # official MCPS page for compensation info


class Q4Info(BaseModel):
    clearance_name: Optional[str] = None  # e.g., "Child Protective Services (CPS) clearance"
    mentions_state_requirement: Optional[bool] = None  # whether it notes this is a State of Maryland requirement
    source_urls: List[str] = Field(default_factory=list)  # official MCPS sources


class Q5Info(BaseModel):
    required_years: Optional[str] = None  # e.g., "3 years"
    mentions_qualifying_experience: Optional[bool] = None  # answer notes "qualifying" experience requirement
    source_urls: List[str] = Field(default_factory=list)  # TN Dept of Education or official TN sources


class Q6Info(BaseModel):
    eligibility_answer: Optional[str] = None  # e.g., "Yes, eligible"
    reasoning: Optional[str] = None  # explanation that 4 >= 3
    source_urls: List[str] = Field(default_factory=list)  # supporting links (can reuse Q5 sources)


class ExtractAll(BaseModel):
    q1: Optional[Q1Info] = None
    q2: Optional[Q2Info] = None
    q3: Optional[Q3Info] = None
    q4: Optional[Q4Info] = None
    q5: Optional[Q5Info] = None
    q6: Optional[Q6Info] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return (
        "Extract the six requested answers exactly as presented in the provided answer text, "
        "and include any cited supporting URLs.\n\n"
        "For each question, extract the fields below. If a field is not present in the answer text, return null for that field "
        "or an empty array when appropriate. Do not invent information or URLs.\n\n"
        "Q1 (MCPS minimum educational credential):\n"
        "- min_credential: The minimum educational credential stated (e.g., 'associate degree or 60 college credits').\n"
        "- mentions_accredited: true/false indicating whether the answer explicitly mentions 'accredited' institution.\n"
        "- source_urls: Array of URLs cited for Q1; prefer official MCPS domains if provided.\n\n"
        "Q2 (MCPS substitute posting ID for 2025-2026):\n"
        "- posting_id: The specific job posting ID or reference number stated.\n"
        "- school_year: The school year stated (ideally '2025-2026').\n"
        "- posting_url: The URL to the specific MCPS job posting page.\n\n"
        "Q3 (MCPS daily compensation rate for substitutes):\n"
        "- daily_rate: The daily rate amount stated (e.g., '$140 per day'). If the answer states that the rate is not publicly available, extract that phrase exactly.\n"
        "- source_url: A URL cited for compensation info (prefer official MCPS page).\n\n"
        "Q4 (Maryland-specific clearance beyond background checks):\n"
        "- clearance_name: The specific clearance named (e.g., 'Child Protective Services (CPS) clearance').\n"
        "- mentions_state_requirement: true/false indicating whether the answer specifies this is a State of Maryland requirement.\n"
        "- source_urls: Array of URLs cited for Q4; prefer official MCPS domains.\n\n"
        "Q5 (Years of teaching experience required for Tennessee Professional Teacher License):\n"
        "- required_years: The number of years stated (e.g., '3 years').\n"
        "- mentions_qualifying_experience: true/false indicating whether the answer notes it must be qualifying teaching experience.\n"
        "- source_urls: Array of URLs cited for Q5; prefer official Tennessee government or Dept of Education domains.\n\n"
        "Q6 (Eligibility given 4 years of experience):\n"
        "- eligibility_answer: The answer stated (e.g., 'Yes, eligible' or 'No, not eligible').\n"
        "- reasoning: Brief explanation comparing 4 years to the required years.\n"
        "- source_urls: Array of URLs cited to support this determination (can reuse Q5 sources if applicable).\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_official_mcps_url(url: Optional[str]) -> bool:
    if not url:
        return False
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return False
    return (
        netloc.endswith("montgomeryschoolsmd.org") or
        netloc.endswith("mcps.taleo.net")
    )


def is_official_tennessee_url(url: Optional[str]) -> bool:
    if not url:
        return False
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return False
    return (
        netloc.endswith("tn.gov") or
        netloc.endswith("tncompass.org")
    )


def non_empty_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_q1_verification(evaluator: Evaluator, parent_node, q1: Optional[Q1Info]) -> None:
    q1_node = evaluator.add_parallel(
        id="Q1_MCPS_Educational_Requirement",
        desc="Answer question 1: Minimum educational credential required for MCPS substitute teachers",
        parent=parent_node,
        critical=False
    )

    # References block
    q1_ref_node = evaluator.add_parallel(
        id="Q1_Reference_URL",
        desc="Provide valid URL reference from official MCPS source documenting the educational requirement",
        parent=q1_node,
        critical=True
    )

    q1_urls = non_empty_urls(q1.source_urls if q1 else None)

    q1_url_provided = evaluator.add_custom_node(
        result=len(q1_urls) > 0,
        id="Q1_URL_Provided",
        desc="A URL is provided for this answer",
        parent=q1_ref_node,
        critical=True
    )

    q1_url_official = evaluator.add_custom_node(
        result=any(is_official_mcps_url(u) for u in q1_urls),
        id="Q1_URL_Official",
        desc="The URL is from an official MCPS domain (montgomeryschoolsmd.org or mcps.taleo.net)",
        parent=q1_ref_node,
        critical=True
    )

    # Credential answer content
    q1_cred_node = evaluator.add_parallel(
        id="Q1_Credential_Answer",
        desc="Correctly identify that MCPS requires either an associate degree OR 60 college credits from an accredited institution",
        parent=q1_node,
        critical=True
    )

    q1_content_leaf = evaluator.add_leaf(
        id="Q1_Answer_Content",
        desc="The answer states the minimum requirement as either associate degree or 60 college credits",
        parent=q1_cred_node,
        critical=True
    )

    claim_req = "MCPS requires substitute teachers to have either an associate degree or at least 60 college credits."
    await evaluator.verify(
        claim=claim_req,
        node=q1_content_leaf,
        sources=q1_urls,
        additional_instruction="Verify on official MCPS pages that the minimum educational credential for substitutes is either an associate (AA/AS) degree or at least 60 college credits.",
        extra_prerequisites=[q1_url_provided]
    )

    q1_accred_leaf = evaluator.add_leaf(
        id="Q1_Accreditation_Noted",
        desc="The answer mentions that the institution must be accredited",
        parent=q1_cred_node,
        critical=True
    )

    claim_accred = "The provided answer explicitly mentions that the required degree or credits must be from an accredited institution."
    await evaluator.verify(
        claim=claim_accred,
        node=q1_accred_leaf,
        additional_instruction="Check the provided answer text itself to confirm it mentions 'accredited' institution. Do not rely on external sources for this check."
    )


async def build_q2_verification(evaluator: Evaluator, parent_node, q2: Optional[Q2Info]) -> None:
    q2_node = evaluator.add_parallel(
        id="Q2_MCPS_Job_Posting_ID",
        desc="Answer question 2: Job posting ID or reference number for MCPS substitute teacher positions 2025-2026",
        parent=parent_node,
        critical=False
    )

    # References
    q2_ref_node = evaluator.add_parallel(
        id="Q2_Reference_URL",
        desc="Provide valid URL reference showing the job posting with the ID",
        parent=q2_node,
        critical=True
    )

    posting_url = q2.posting_url if q2 else None

    q2_url_provided = evaluator.add_custom_node(
        result=bool(posting_url),
        id="Q2_URL_Provided",
        desc="A URL is provided that links to the job posting",
        parent=q2_ref_node,
        critical=True
    )

    q2_url_shows_id_leaf = evaluator.add_leaf(
        id="Q2_URL_Shows_ID",
        desc="The URL leads to a page that displays the job posting ID",
        parent=q2_ref_node,
        critical=True
    )

    claim_shows_id = f"The job posting page displays the job posting ID or reference number '{q2.posting_id if q2 and q2.posting_id else ''}'."
    await evaluator.verify(
        claim=claim_shows_id,
        node=q2_url_shows_id_leaf,
        sources=posting_url,
        additional_instruction="Look for 'Job ID', 'Requisition ID', 'Posting Number', or similar on the page that matches the stated ID.",
        extra_prerequisites=[q2_url_provided]
    )

    # Posting ID answer content
    q2_posting_answer_node = evaluator.add_parallel(
        id="Q2_Posting_ID_Answer",
        desc="Provide the specific job posting ID or reference number for the 2025-2026 substitute teacher positions",
        parent=q2_node,
        critical=True
    )

    q2_id_provided = evaluator.add_custom_node(
        result=bool(q2 and q2.posting_id and q2.posting_id.strip()),
        id="Q2_ID_Provided",
        desc="A specific job posting ID or reference number is stated",
        parent=q2_posting_answer_node,
        critical=True
    )

    q2_year_correct_leaf = evaluator.add_leaf(
        id="Q2_Year_Correct",
        desc="The posting is confirmed to be for the 2025-2026 school year",
        parent=q2_posting_answer_node,
        critical=True
    )

    claim_year = "This job posting is for the 2025-2026 school year."
    await evaluator.verify(
        claim=claim_year,
        node=q2_year_correct_leaf,
        sources=posting_url,
        additional_instruction="Confirm the school year on the posting page (e.g., '2025-2026', 'SY 2025-26', or similar phrasing).",
        extra_prerequisites=[q2_url_provided]
    )


async def build_q3_verification(evaluator: Evaluator, parent_node, q3: Optional[Q3Info]) -> None:
    q3_node = evaluator.add_parallel(
        id="Q3_MCPS_Compensation",
        desc="Answer question 3: Daily compensation rate for MCPS substitute teachers",
        parent=parent_node,
        critical=False
    )

    # Daily rate stated
    q3_daily_rate_node = evaluator.add_parallel(
        id="Q3_Daily_Rate_Answer",
        desc="Provide the daily compensation rate for MCPS substitute teachers",
        parent=q3_node,
        critical=False
    )

    q3_rate_leaf = evaluator.add_leaf(
        id="Q3_Rate_Stated",
        desc="A specific daily rate amount is provided, or a clear statement that the rate is not publicly available",
        parent=q3_daily_rate_node,
        critical=False
    )

    compensation_url = q3.source_url if q3 else None
    rate_text = q3.daily_rate if q3 and q3.daily_rate else ""

    claim_rate = (
        f"The daily compensation rate for MCPS substitute teachers is {rate_text}."
        if rate_text and "not publicly available" not in rate_text.lower()
        else "The MCPS page indicates the daily rate is not publicly posted or publicly available."
    )
    await evaluator.verify(
        claim=claim_rate,
        node=q3_rate_leaf,
        sources=compensation_url,
        additional_instruction="Verify the stated rate (or lack thereof) on the referenced official MCPS page."
    )

    # References
    q3_ref_node = evaluator.add_parallel(
        id="Q3_Reference_URL",
        desc="Provide URL reference for compensation information",
        parent=q3_node,
        critical=False
    )

    q3_url_provided = evaluator.add_custom_node(
        result=bool(compensation_url),
        id="Q3_URL_Provided",
        desc="A URL reference is provided",
        parent=q3_ref_node,
        critical=False
    )

    q3_url_official = evaluator.add_custom_node(
        result=is_official_mcps_url(compensation_url),
        id="Q3_URL_Official",
        desc="The URL is from an official MCPS source",
        parent=q3_ref_node,
        critical=True
    )


async def build_q4_verification(evaluator: Evaluator, parent_node, q4: Optional[Q4Info]) -> None:
    q4_node = evaluator.add_parallel(
        id="Q4_Maryland_Clearance",
        desc="Answer question 4: Specific Maryland state clearance required for MCPS substitute teachers",
        parent=parent_node,
        critical=False
    )

    # References
    q4_ref_node = evaluator.add_parallel(
        id="Q4_Reference_URL",
        desc="Provide valid URL reference from official MCPS source documenting the clearance requirement",
        parent=q4_node,
        critical=True
    )

    q4_urls = non_empty_urls(q4.source_urls if q4 else None)

    q4_url_provided = evaluator.add_custom_node(
        result=len(q4_urls) > 0,
        id="Q4_URL_Provided",
        desc="A URL is provided for this requirement",
        parent=q4_ref_node,
        critical=True
    )

    q4_url_official = evaluator.add_custom_node(
        result=any(is_official_mcps_url(u) for u in q4_urls),
        id="Q4_URL_Official",
        desc="The URL is from an official MCPS domain",
        parent=q4_ref_node,
        critical=True
    )

    # Clearance answer
    q4_answer_node = evaluator.add_parallel(
        id="Q4_Clearance_Answer",
        desc="Identify the specific Maryland clearance required beyond standard background checks",
        parent=q4_node,
        critical=True
    )

    q4_cps_leaf = evaluator.add_leaf(
        id="Q4_CPS_Clearance",
        desc="The answer identifies Child Protective Services clearance as the required Maryland-specific clearance",
        parent=q4_answer_node,
        critical=True
    )

    claim_cps = "MCPS requires Child Protective Services (CPS) clearance for substitute teachers, beyond standard background checks."
    await evaluator.verify(
        claim=claim_cps,
        node=q4_cps_leaf,
        sources=q4_urls,
        additional_instruction="Verify on the official MCPS page(s) that CPS clearance is required.",
        extra_prerequisites=[q4_url_provided]
    )

    q4_state_leaf = evaluator.add_leaf(
        id="Q4_State_Specified",
        desc="The answer specifies this is a State of Maryland requirement",
        parent=q4_answer_node,
        critical=True
    )

    claim_state = "The CPS clearance mentioned is a State of Maryland requirement."
    await evaluator.verify(
        claim=claim_state,
        node=q4_state_leaf,
        sources=q4_urls,
        additional_instruction="Confirm that the clearance is specifically a State of Maryland requirement as indicated by the MCPS page.",
        extra_prerequisites=[q4_url_provided]
    )


async def build_q5_verification(evaluator: Evaluator, parent_node, q5: Optional[Q5Info]) -> None:
    q5_node = evaluator.add_parallel(
        id="Q5_Tennessee_Experience_Requirement",
        desc="Answer question 5: Years of teaching experience required for Tennessee Professional Teacher License",
        parent=parent_node,
        critical=False
    )

    # References
    q5_ref_node = evaluator.add_parallel(
        id="Q5_Reference_URL",
        desc="Provide valid URL reference from official Tennessee Department of Education source",
        parent=q5_node,
        critical=True
    )

    q5_urls = non_empty_urls(q5.source_urls if q5 else None)

    q5_url_provided = evaluator.add_custom_node(
        result=len(q5_urls) > 0,
        id="Q5_URL_Provided",
        desc="A URL is provided for this requirement",
        parent=q5_ref_node,
        critical=True
    )

    q5_url_official = evaluator.add_custom_node(
        result=any(is_official_tennessee_url(u) for u in q5_urls),
        id="Q5_URL_Official",
        desc="The URL is from an official Tennessee government or education department domain",
        parent=q5_ref_node,
        critical=True
    )

    # Years answer
    q5_years_node = evaluator.add_parallel(
        id="Q5_Years_Answer",
        desc="State the number of years of teaching experience required for Tennessee Professional Teacher License",
        parent=q5_node,
        critical=True
    )

    q5_three_leaf = evaluator.add_leaf(
        id="Q5_Three_Years",
        desc="The answer correctly identifies 3 years as the requirement",
        parent=q5_years_node,
        critical=True
    )

    claim_three = "Tennessee Professional Teacher License requires at least 3 years of teaching experience."
    await evaluator.verify(
        claim=claim_three,
        node=q5_three_leaf,
        sources=q5_urls,
        additional_instruction="Verify on official Tennessee Department of Education pages (tn.gov or TNCompass) that 3 years of teaching experience are required."
    )

    q5_qual_leaf = evaluator.add_leaf(
        id="Q5_Qualifying_Experience",
        desc="The answer notes this must be qualifying teaching experience",
        parent=q5_years_node,
        critical=True
    )

    claim_qual = "The experience requirement refers to qualifying teaching experience as defined by Tennessee licensure guidelines."
    await evaluator.verify(
        claim=claim_qual,
        node=q5_qual_leaf,
        sources=q5_urls,
        additional_instruction="Confirm on official Tennessee sources that the years required are qualifying teaching experience (not just any service).",
        extra_prerequisites=[q5_url_provided]
    )


async def build_q6_verification(evaluator: Evaluator, parent_node, q6: Optional[Q6Info], q5: Optional[Q5Info]) -> None:
    q6_node = evaluator.add_parallel(
        id="Q6_Candidate_Eligibility",
        desc="Answer question 6: Whether a teacher with 4 years of experience is eligible for Tennessee Professional Teacher License",
        parent=parent_node,
        critical=False
    )

    q6_answer_node = evaluator.add_parallel(
        id="Q6_Eligibility_Answer",
        desc="Determine and state whether the candidate with 4 years of experience qualifies",
        parent=q6_node,
        critical=True
    )

    q6_yes_leaf = evaluator.add_leaf(
        id="Q6_Yes_Eligible",
        desc="The answer correctly concludes that 4 years meets the 3-year minimum requirement",
        parent=q6_answer_node,
        critical=True
    )

    ref_urls_for_q6 = non_empty_urls((q6.source_urls if q6 else None) or (q5.source_urls if q5 else None))

    claim_yes = "Given a minimum requirement of 3 years, a candidate with 4 years of full-time qualifying teaching experience is eligible to apply for a Tennessee Professional Teacher License."
    await evaluator.verify(
        claim=claim_yes,
        node=q6_yes_leaf,
        sources=ref_urls_for_q6 if ref_urls_for_q6 else None,
        additional_instruction="Use the Tennessee requirement verified in Q5 (3 years) to determine eligibility for 4 years."
    )

    q6_reason_leaf = evaluator.add_leaf(
        id="Q6_Reasoning_Provided",
        desc="The answer explains the comparison between the 4-year experience and 3-year requirement",
        parent=q6_answer_node,
        critical=True
    )

    claim_reason = "The provided answer explicitly explains that because 4 years is equal to or greater than the 3-year requirement, the candidate is eligible."
    await evaluator.verify(
        claim=claim_reason,
        node=q6_reason_leaf,
        additional_instruction="Check the answer text for a clear explanation that 4 ≥ 3 leading to eligibility. Do not rely on external sources for this check."
    )

    q6_ref_node = evaluator.add_parallel(
        id="Q6_Reference_URL",
        desc="Provide URL reference supporting the eligibility determination",
        parent=q6_node,
        critical=False
    )

    q6_url_provided = evaluator.add_custom_node(
        result=len(non_empty_urls(q6.source_urls if q6 else None)) > 0,
        id="Q6_URL_Provided",
        desc="A URL reference is provided to support the eligibility determination",
        parent=q6_ref_node,
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
    # Initialize evaluator with a non-critical root to allow mixed critical children
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=ExtractAll,
        extraction_name="extracted_answer_struct"
    )

    # Add ground truth hints (optional)
    evaluator.add_ground_truth({"hints": GROUND_TRUTH_HINTS}, gt_type="ground_truth_hints")

    # Build subtrees for Q1-Q6
    await build_q1_verification(evaluator, root, extracted.q1)
    await build_q2_verification(evaluator, root, extracted.q2)
    await build_q3_verification(evaluator, root, extracted.q3)
    await build_q4_verification(evaluator, root, extracted.q4)
    await build_q5_verification(evaluator, root, extracted.q5)
    await build_q6_verification(evaluator, root, extracted.q6, extracted.q5)

    # Return structured summary
    return evaluator.get_summary()