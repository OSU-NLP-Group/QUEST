import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "big_ten_university_multicriteria"
TASK_DESCRIPTION = (
    "Identify the Big Ten Conference university that satisfies all of the following criteria: "
    "(1) The university was founded in the 19th century, specifically before 1850. "
    "(2) The university was one of the seven founding members of the Big Ten Conference when it was officially established in 1896. "
    "(3) The university reported total research expenditures exceeding $2 billion in fiscal year 2024. "
    "(4) The university sponsors exactly 27 varsity sports teams. "
    "(5) The university's main campus encompasses more than 3,000 acres. "
    "(6) The university had total student enrollment exceeding 52,000 students in fall 2024. "
    "(7) The university holds R1 research classification from the Carnegie Classification of Institutions of Higher Education. "
    "Provide the name of this university along with supporting URL references that verify each of these criteria."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityEvidence(BaseModel):
    university_name: Optional[str] = None

    big_ten_membership_urls: List[str] = Field(default_factory=list)

    founding_year: Optional[str] = None
    founding_year_urls: List[str] = Field(default_factory=list)

    founding_member_1896_urls: List[str] = Field(default_factory=list)

    research_expenditures_fy2024: Optional[str] = None
    research_expenditures_urls: List[str] = Field(default_factory=list)

    varsity_sports_count: Optional[str] = None
    varsity_sports_urls: List[str] = Field(default_factory=list)

    campus_acreage: Optional[str] = None
    campus_acreage_urls: List[str] = Field(default_factory=list)

    enrollment_fall2024: Optional[str] = None
    enrollment_urls: List[str] = Field(default_factory=list)

    carnegie_r1_urls: List[str] = Field(default_factory=list)

    general_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_university_evidence() -> str:
    return """
    Extract the single university that the answer claims satisfies all criteria and the URLs used to justify each criterion.
    Return a JSON object with the following fields:
    - university_name: The name of the single university presented as the answer.
    - big_ten_membership_urls: Array of URLs that support that the university is a current member of the Big Ten Conference.
    - founding_year: The founding year mentioned (if any) in the answer for the university; keep as a string if present; otherwise null.
    - founding_year_urls: Array of URLs supporting the founding year/date of the university.
    - founding_member_1896_urls: Array of URLs supporting that the university was among the seven founding members when the Big Ten (then Western Conference) was established in 1896.
    - research_expenditures_fy2024: The FY2024 total research expenditure amount mentioned (string; keep formatting as-is) if present; else null.
    - research_expenditures_urls: Array of URLs supporting the FY2024 total research expenditures, preferably showing it exceeds $2B.
    - varsity_sports_count: The varsity sports team count stated in the answer (string), if present; else null.
    - varsity_sports_urls: Array of URLs supporting that the university sponsors exactly 27 varsity sports teams.
    - campus_acreage: The main campus acreage value stated in the answer (string), if present; else null.
    - campus_acreage_urls: Array of URLs supporting that the university's main campus covers more than 3,000 acres.
    - enrollment_fall2024: The total student enrollment stated for Fall 2024 (string), if present; else null.
    - enrollment_urls: Array of URLs supporting that Fall 2024 total enrollment exceeded 52,000 students.
    - carnegie_r1_urls: Array of URLs supporting that the university is classified as Carnegie R1 (Very High Research Activity).
    - general_sources: Array of any additional URLs cited in the answer (e.g., if the answer provides a combined sources/references section not mapped above).
    
    IMPORTANT URL EXTRACTION RULES:
    - Only extract URLs that are explicitly present in the answer text. Do not invent or infer URLs.
    - Accept URLs in plain text or markdown link formats; always extract the actual link target.
    - If a URL is missing a protocol (http:// or https://), prepend http://.
    - If a criterion-specific URL list is not explicitly provided in the answer, leave that list empty. Put any remaining general sources into general_sources.
    """


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def _sanitize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    cleaned: List[str] = []
    seen = set()
    for u in urls:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not s:
            continue
        if not re.match(r"^https?://", s):
            s = "http://" + s
        if s not in seen:
            cleaned.append(s)
            seen.add(s)
    return cleaned


def _merged_sources(primary: List[str], fallback: List[str]) -> List[str]:
    primary_clean = _sanitize_urls(primary)
    if primary_clean:
        return primary_clean
    return _sanitize_urls(fallback)


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_and_verify_university_tree(
    evaluator: Evaluator,
    root,
    extracted: UniversityEvidence,
) -> None:
    # Top-level critical node: University Identification and all criteria
    uni_node = evaluator.add_parallel(
        id="University_Identification",
        desc="Identify one university and provide URL evidence that it satisfies all stated criteria.",
        parent=root,
        critical=True
    )

    # University name provided (critical)
    name_ok = bool(extracted.university_name and extracted.university_name.strip())
    evaluator.add_custom_node(
        result=name_ok,
        id="University_Name_Provided",
        desc="Provides the name of a single university as the answer.",
        parent=uni_node,
        critical=True
    )

    name = extracted.university_name or "the university"

    # ---------------------- Big Ten Membership ---------------------- #
    bigten_node = evaluator.add_parallel(
        id="Big_Ten_Membership",
        desc="The university is (currently) a Big Ten Conference member institution (not only historically).",
        parent=uni_node,
        critical=True
    )
    membership_urls = _merged_sources(extracted.big_ten_membership_urls, extracted.general_sources)

    evaluator.add_custom_node(
        result=len(membership_urls) > 0,
        id="Membership_Reference_URL",
        desc="Provides at least one URL that explicitly supports Big Ten membership.",
        parent=bigten_node,
        critical=True
    )

    membership_leaf = evaluator.add_leaf(
        id="Membership_Verification",
        desc="Evidence shows the institution is a Big Ten Conference member (as of the relevant timeframe).",
        parent=bigten_node,
        critical=True
    )
    membership_claim = f"The university named '{name}' is a current member of the Big Ten Conference."
    await evaluator.verify(
        claim=membership_claim,
        node=membership_leaf,
        sources=membership_urls,
        additional_instruction=(
            "Verify that the page explicitly indicates the institution is a current Big Ten member. "
            "Accept official Big Ten pages, the institution's athletics pages, or reliable references listing current members."
        ),
    )

    # ---------------------- Founded before 1850 --------------------- #
    founded_node = evaluator.add_parallel(
        id="Founded_19th_Century_Before_1850",
        desc="Founded in the 19th century and before 1850 (i.e., founding year is 1801–1849 inclusive).",
        parent=uni_node,
        critical=True
    )
    founding_urls = _merged_sources(extracted.founding_year_urls, extracted.general_sources)

    evaluator.add_custom_node(
        result=len(founding_urls) > 0,
        id="Founding_Date_Reference_URL",
        desc="Provides at least one URL that explicitly supports the founding year/date.",
        parent=founded_node,
        critical=True
    )

    founding_leaf = evaluator.add_leaf(
        id="Founding_Year_Check",
        desc="Founding year is between 1801 and 1849 inclusive.",
        parent=founded_node,
        critical=True
    )
    if extracted.founding_year and extracted.founding_year.strip():
        founding_claim = (
            f"The university named '{name}' was founded in {extracted.founding_year}, "
            "which is before 1850 (i.e., between 1801 and 1849 inclusive)."
        )
    else:
        founding_claim = (
            f"The university named '{name}' was founded before 1850 (in the 19th century)."
        )

    await evaluator.verify(
        claim=founding_claim,
        node=founding_leaf,
        sources=founding_urls,
        additional_instruction=(
            "Support the claim if the founding year shown on the page is between 1801 and 1849 inclusive. "
            "If multiple dates are presented (e.g., chartered vs. classes opened), use the earliest founding date."
        ),
    )

    # ---------------------- Founding member in 1896 ----------------- #
    fm_node = evaluator.add_parallel(
        id="Founding_Member_1896",
        desc="One of the seven founding members when the Big Ten (Western Conference) was established in 1896.",
        parent=uni_node,
        critical=True
    )
    fm_urls = _merged_sources(extracted.founding_member_1896_urls, extracted.general_sources)

    evaluator.add_custom_node(
        result=len(fm_urls) > 0,
        id="Founding_Member_Reference_URL",
        desc="Provides at least one URL that explicitly supports founding-member status in 1896.",
        parent=fm_node,
        critical=True
    )

    fm_leaf = evaluator.add_leaf(
        id="Founding_Member_Check",
        desc="Evidence shows the university was a founding member in 1896 (among the initial seven).",
        parent=fm_node,
        critical=True
    )
    fm_claim = (
        f"In 1896, the university named '{name}' was among the seven founding members of the Western Conference "
        "(later known as the Big Ten Conference)."
    )
    await evaluator.verify(
        claim=fm_claim,
        node=fm_leaf,
        sources=fm_urls,
        additional_instruction=(
            "The page should explicitly state that the school was a founding member in 1896 (initial seven) "
            "of the Western Conference/Big Ten."
        ),
    )

    # --------------- Research expenditures FY2024 > $2B ------------- #
    exp_node = evaluator.add_parallel(
        id="Research_Expenditures_FY2024_Over_2B",
        desc="Total research expenditures exceed $2 billion in fiscal year 2024.",
        parent=uni_node,
        critical=True
    )
    exp_urls = _merged_sources(extracted.research_expenditures_urls, extracted.general_sources)

    evaluator.add_custom_node(
        result=len(exp_urls) > 0,
        id="Expenditures_Reference_URL",
        desc="Provides at least one URL that explicitly supports the FY2024 research expenditure figure.",
        parent=exp_node,
        critical=True
    )

    exp_leaf = evaluator.add_leaf(
        id="Expenditures_Check",
        desc="Evidence shows FY2024 research expenditures > $2,000,000,000.",
        parent=exp_node,
        critical=True
    )
    exp_claim = (
        f"In fiscal year 2024 (FY2024), {name} reported total research expenditures exceeding $2 billion."
    )
    await evaluator.verify(
        claim=exp_claim,
        node=exp_leaf,
        sources=exp_urls,
        additional_instruction=(
            "Confirm that the page refers to FY2024 total research expenditures and that the amount is greater than $2,000,000,000. "
            "Accept paraphrases like 'over $2 billion' or numeric values above 2,000,000,000 USD."
        ),
    )

    # ---------------- Varsity sports exactly 27 teams ---------------- #
    sports_node = evaluator.add_parallel(
        id="Varsity_Sports_Exactly_27",
        desc="Sponsors exactly 27 varsity sports teams.",
        parent=uni_node,
        critical=True
    )
    sports_urls = _merged_sources(extracted.varsity_sports_urls, extracted.general_sources)

    evaluator.add_custom_node(
        result=len(sports_urls) > 0,
        id="Sports_Count_Reference_URL",
        desc="Provides at least one URL that explicitly supports the varsity sports count.",
        parent=sports_node,
        critical=True
    )

    sports_leaf = evaluator.add_leaf(
        id="Sports_Count_Check",
        desc="Evidence shows the varsity sports team count is exactly 27.",
        parent=sports_node,
        critical=True
    )
    sports_claim = f"{name} sponsors exactly 27 varsity sports teams."
    await evaluator.verify(
        claim=sports_claim,
        node=sports_leaf,
        sources=sports_urls,
        additional_instruction=(
            "Verify that the page explicitly indicates the university sponsors 27 varsity sports teams. "
            "If it lists men's and women's counts, the sum must equal 27."
        ),
    )

    # ---------------- Main campus > 3,000 acres ---------------------- #
    campus_node = evaluator.add_parallel(
        id="Main_Campus_Over_3000_Acres",
        desc="Main campus encompasses more than 3,000 acres.",
        parent=uni_node,
        critical=True
    )
    campus_urls = _merged_sources(extracted.campus_acreage_urls, extracted.general_sources)

    evaluator.add_custom_node(
        result=len(campus_urls) > 0,
        id="Campus_Acreage_Reference_URL",
        desc="Provides at least one URL that explicitly supports the main-campus acreage claim.",
        parent=campus_node,
        critical=True
    )

    campus_leaf = evaluator.add_leaf(
        id="Campus_Acreage_Check",
        desc="Evidence shows main-campus acreage > 3,000 acres.",
        parent=campus_node,
        critical=True
    )
    campus_claim = f"The main campus of {name} encompasses more than 3,000 acres."
    await evaluator.verify(
        claim=campus_claim,
        node=campus_leaf,
        sources=campus_urls,
        additional_instruction=(
            "Confirm that the page specifies main-campus (not system-wide) acreage above 3,000. "
            "Accept approximate figures clearly over 3,000 (e.g., ~3,200 acres)."
        ),
    )

    # ------------- Enrollment Fall 2024 > 52,000 students ------------ #
    enroll_node = evaluator.add_parallel(
        id="Enrollment_Fall2024_Over_52000",
        desc="Total student enrollment exceeds 52,000 students in fall 2024.",
        parent=uni_node,
        critical=True
    )
    enrollment_urls = _merged_sources(extracted.enrollment_urls, extracted.general_sources)

    evaluator.add_custom_node(
        result=len(enrollment_urls) > 0,
        id="Enrollment_Reference_URL",
        desc="Provides at least one URL that explicitly supports the fall 2024 enrollment figure.",
        parent=enroll_node,
        critical=True
    )

    enroll_leaf = evaluator.add_leaf(
        id="Enrollment_Check",
        desc="Evidence shows fall 2024 total enrollment > 52,000.",
        parent=enroll_node,
        critical=True
    )
    enroll_claim = f"In Fall 2024, total student enrollment at {name} exceeded 52,000 students."
    await evaluator.verify(
        claim=enroll_claim,
        node=enroll_leaf,
        sources=enrollment_urls,
        additional_instruction=(
            "The page should clearly indicate 'Fall 2024' (or equivalent) and 'total' student enrollment exceeding 52,000."
        ),
    )

    # -------------------- Carnegie R1 classification ----------------- #
    r1_node = evaluator.add_parallel(
        id="Carnegie_R1",
        desc="Holds Carnegie R1 classification.",
        parent=uni_node,
        critical=True
    )
    r1_urls = _merged_sources(extracted.carnegie_r1_urls, extracted.general_sources)

    evaluator.add_custom_node(
        result=len(r1_urls) > 0,
        id="R1_Reference_URL",
        desc="Provides at least one URL that explicitly supports the Carnegie R1 classification.",
        parent=r1_node,
        critical=True
    )

    r1_leaf = evaluator.add_leaf(
        id="R1_Check",
        desc="Evidence shows the institution is classified as Carnegie R1 (Very High Research Activity).",
        parent=r1_node,
        critical=True
    )
    r1_claim = (
        f"{name} is classified as R1: Doctoral Universities – Very High Research Activity in the Carnegie Classification."
    )
    await evaluator.verify(
        claim=r1_claim,
        node=r1_leaf,
        sources=r1_urls,
        additional_instruction=(
            "Confirm that the page indicates the institution has R1 classification (Very High Research Activity). "
            "Accept official Carnegie Classification pages or credible institutional references that clearly state R1 status."
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
    model: str = "o4-mini"
) -> Dict:
    # Initialize evaluator (root is non-critical; we add a critical child node per rubric)
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

    # Extract structured evidence from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_university_evidence(),
        template_class=UniversityEvidence,
        extraction_name="university_evidence"
    )

    # Add brief criteria info for transparency
    evaluator.add_custom_info(
        {
            "criteria": [
                "Current Big Ten member",
                "Founded before 1850 (1801–1849 inclusive)",
                "Founding member in 1896 (initial seven)",
                "FY2024 total research expenditures > $2B",
                "Exactly 27 varsity sports teams",
                "Main campus > 3,000 acres",
                "Fall 2024 total enrollment > 52,000",
                "Carnegie Classification: R1"
            ]
        },
        info_type="context",
        info_name="evaluation_criteria"
    )

    # Build verification tree and run checks
    await build_and_verify_university_tree(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()