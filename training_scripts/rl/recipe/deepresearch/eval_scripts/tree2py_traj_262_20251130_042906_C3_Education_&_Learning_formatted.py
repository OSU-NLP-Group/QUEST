import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "coach_ttu_history_1993_chair_book_2022_awards"
TASK_DESCRIPTION = (
    "In 2022, a college football coach received multiple national Coach of the Year awards, including the AP Coach of the Year, the Bear Bryant Award, and the Walter Camp Coach of the Year Award. "
    "This coach earned a bachelor's degree in history from a specific university in 1993. Please identify: "
    "(1) the full name of this coach, "
    "(2) the university where he earned his bachelor's degree, "
    "(3) the current chair of the history department at that university, and "
    "(4) a book authored by that department chair, including the book's full title, publisher, and year of publication."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CoachExtraction(BaseModel):
    full_name: Optional[str] = None
    # Awards the answer claims (free-form names as strings)
    awards_2022: List[str] = Field(default_factory=list)
    # URLs that the answer cites for the coach's 2022 awards
    awards_source_urls: List[str] = Field(default_factory=list)

    # Education details (as stated in the answer)
    degree_institution: Optional[str] = None
    degree_type: Optional[str] = None  # e.g., "Bachelor's", "BA", "B.A."
    major: Optional[str] = None        # e.g., "history"
    graduation_year: Optional[str] = None  # keep as string to tolerate formats like "1993" or "1993 (BA)"

    # URLs that the answer cites for the coach's education
    education_source_urls: List[str] = Field(default_factory=list)


class DepartmentChairExtraction(BaseModel):
    name: Optional[str] = None
    # URLs that the answer cites for verifying the chair role at TTU History
    source_urls: List[str] = Field(default_factory=list)


class BookExtraction(BaseModel):
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)  # as listed by the answer
    publisher: Optional[str] = None
    publication_year: Optional[str] = None
    # URLs that the answer cites for verifying book authorship and publication details
    source_urls: List[str] = Field(default_factory=list)


class FullExtraction(BaseModel):
    coach: Optional[CoachExtraction] = None
    chair: Optional[DepartmentChairExtraction] = None
    book: Optional[BookExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract the following information exactly as it appears in the provided answer. Do NOT invent or infer anything that is not explicitly stated.

1) coach:
   - full_name: The full name of the coach identified by the answer.
   - awards_2022: A list of the specific national Coach of the Year awards the answer associates with this coach for 2022 (e.g., "AP Coach of the Year", "Bear Bryant Award", "Walter Camp Coach of the Year").
   - awards_source_urls: A list of URLs cited in the answer that support the awards (explicit URLs only).
   - degree_institution: The university the answer states he earned his bachelor's degree from.
   - degree_type: The degree type (e.g., "Bachelor's", "BA", "B.A.") as written in the answer.
   - major: The major/field (e.g., "history") as written in the answer.
   - graduation_year: The graduation year as written in the answer (keep any formatting; do not normalize).
   - education_source_urls: A list of URLs cited in the answer that support the education details.

2) chair:
   - name: The full name of the current chair of the History Department at the same university identified in coach.degree_institution (as claimed by the answer).
   - source_urls: A list of URLs cited in the answer that confirm that chair role.

3) book:
   - title: The full title of one book authored or co-authored by the identified chair (if the answer lists more than one, extract the first one mentioned).
   - authors: The list of authors as stated in the answer (if not stated, return an empty list).
   - publisher: The publisher of the book as stated in the answer.
   - publication_year: The publication year as stated in the answer (do not normalize; keep as free-form string).
   - source_urls: A list of URLs cited in the answer that support the book's title, authorship, publisher, and publication year.

SPECIAL RULES FOR URL EXTRACTION:
- Only extract URLs explicitly present in the answer. Do not infer or construct any URLs.
- Accept URLs in plain form or markdown link form; extract the actual URL.
- If a URL appears without the protocol, you may prepend http://.

Return a single JSON object with fields: coach, chair, and book, following the schema exactly.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _list_non_empty(lst: Optional[List[str]]) -> bool:
    return bool(lst and len(lst) > 0)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_coach_and_education_checks(
    evaluator: Evaluator,
    parent_node,
    coach: CoachExtraction
) -> None:
    """
    Build and execute verification checks for:
    - Coach full name existence
    - 2022 awards (AP, Bear Bryant, Walter Camp) verified by sources
    - Education details (institution TTU, major history, bachelor's degree, year 1993) verified by sources
    - Existence of sources for both awards and education
    """
    # Parallel, critical group for coach + education
    coach_group = evaluator.add_parallel(
        id="Coach_and_Education",
        desc="Identify the correct coach and verify awards + required education details.",
        parent=parent_node,
        critical=True
    )

    # 0) Sources existence node (create first to be an explicit prerequisite for other checks)
    sources_node = evaluator.add_parallel(
        id="Coach_Awards_and_Education_Sources",
        desc="Provide publicly accessible URL source(s) that support the coach's required 2022 awards and the required education credential (Texas Tech, history, 1993).",
        parent=coach_group,
        critical=True
    )

    awards_sources_exist = evaluator.add_custom_node(
        result=_list_non_empty(coach.awards_source_urls),
        id="Coach_Awards_Sources_Exist",
        desc="Awards sources are provided (at least one URL).",
        parent=sources_node,
        critical=True
    )

    edu_sources_exist = evaluator.add_custom_node(
        result=_list_non_empty(coach.education_source_urls),
        id="Coach_Education_Sources_Exist",
        desc="Education sources are provided (at least one URL).",
        parent=sources_node,
        critical=True
    )

    # 1) Coach full name existence
    evaluator.add_custom_node(
        result=_non_empty(coach.full_name),
        id="Coach_Full_Name",
        desc="Provide the coach's full name.",
        parent=coach_group,
        critical=True
    )

    # 2) Awards required set - break into three atomic leaves
    awards_set_node = evaluator.add_parallel(
        id="Coach_Awards_2022_Required_Set",
        desc="Verify the coach received (at minimum) the AP Coach of the Year, Bear Bryant Award, and Walter Camp Coach of the Year Award in 2022.",
        parent=coach_group,
        critical=True
    )

    # AP Coach of the Year (Associated Press)
    ap_award_leaf = evaluator.add_leaf(
        id="Coach_Award_AP_2022",
        desc="In 2022, the coach received the AP College Football Coach of the Year award.",
        parent=awards_set_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In 2022, {coach.full_name or 'the identified coach'} received the AP (Associated Press) College Football Coach of the Year award.",
        node=ap_award_leaf,
        sources=coach.awards_source_urls,
        additional_instruction=(
            "Treat 'AP Coach of the Year' and 'Associated Press College Football Coach of the Year' as equivalent. "
            "The award can be described as for the 2022 season even if presented in late 2022 or early 2023. "
            "If no valid supporting URL is provided, this should be considered not supported."
        ),
        extra_prerequisites=[awards_sources_exist]
    )

    # Bear Bryant Coach of the Year Award
    bryant_award_leaf = evaluator.add_leaf(
        id="Coach_Award_Bear_Bryant_2022",
        desc="In 2022, the coach received the Paul 'Bear' Bryant Coach of the Year award.",
        parent=awards_set_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In 2022, {coach.full_name or 'the identified coach'} received the Paul 'Bear' Bryant Coach of the Year award.",
        node=bryant_award_leaf,
        sources=coach.awards_source_urls,
        additional_instruction=(
            "Treat 'Bear Bryant Award' and 'Paul \"Bear\" Bryant Coach of the Year' as equivalent. "
            "The award can be described as recognizing the 2022 season even if presented at a different date. "
            "If no valid supporting URL is provided, this should be considered not supported."
        ),
        extra_prerequisites=[awards_sources_exist]
    )

    # Walter Camp Coach of the Year Award
    walter_award_leaf = evaluator.add_leaf(
        id="Coach_Award_Walter_Camp_2022",
        desc="In 2022, the coach received the Walter Camp Coach of the Year award.",
        parent=awards_set_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In 2022, {coach.full_name or 'the identified coach'} received the Walter Camp Coach of the Year award.",
        node=walter_award_leaf,
        sources=coach.awards_source_urls,
        additional_instruction=(
            "Treat 'Walter Camp Coach of the Year' and 'Walter Camp Coach of the Year Award' as equivalent. "
            "The award can be described as recognizing the 2022 season. "
            "If no valid supporting URL is provided, this should be considered not supported."
        ),
        extra_prerequisites=[awards_sources_exist]
    )

    # 3) Education verification (atomic leaves)
    education_node = evaluator.add_parallel(
        id="Coach_Education_TTU_History_1993",
        desc="Verify the coach earned a bachelor's degree in history from Texas Tech University in 1993.",
        parent=coach_group,
        critical=True
    )

    # Institution is Texas Tech University
    edu_institution_leaf = evaluator.add_leaf(
        id="Edu_Institution_TTU",
        desc="University where he earned the bachelor's degree matches what the answer claims.",
        parent=education_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The university where {coach.full_name or 'the identified coach'} earned his bachelor's degree is '{coach.degree_institution or ''}'.",
        node=edu_institution_leaf,
        sources=coach.education_source_urls,
        additional_instruction=(
            "Verify the institution as stated in the answer. Expected is Texas Tech University (TTU). "
            "Allow minor naming variations (e.g., 'Texas Tech'). If the provided URLs do not support the institution, mark as not supported."
        ),
        extra_prerequisites=[edu_sources_exist]
    )

    # Major is History
    edu_major_leaf = evaluator.add_leaf(
        id="Edu_Major_History",
        desc="Bachelor's major/field is correctly stated.",
        parent=education_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The bachelor's major/field was '{coach.major or ''}'.",
        node=edu_major_leaf,
        sources=coach.education_source_urls,
        additional_instruction=(
            "Verify the major as stated in the answer. Expected is 'History'. "
            "Allow case-insensitive matching and minor formatting differences."
        ),
        extra_prerequisites=[edu_sources_exist]
    )

    # Degree type is Bachelor's (BA or equivalent)
    edu_degree_leaf = evaluator.add_leaf(
        id="Edu_Degree_Bachelors",
        desc="Degree type is a bachelor's degree (BA or equivalent).",
        parent=education_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The degree type earned was '{coach.degree_type or ''}'.",
        node=edu_degree_leaf,
        sources=coach.education_source_urls,
        additional_instruction=(
            "Verify the degree type as stated in the answer. Expected is a bachelor's degree (e.g., 'BA', 'B.A.', 'Bachelor of Arts'). "
            "Allow minor formatting variations."
        ),
        extra_prerequisites=[edu_sources_exist]
    )

    # Graduation year is 1993
    edu_year_leaf = evaluator.add_leaf(
        id="Edu_Grad_Year_1993",
        desc="Graduation year is correctly stated.",
        parent=education_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The graduation year for the bachelor's degree was '{coach.graduation_year or ''}'.",
        node=edu_year_leaf,
        sources=coach.education_source_urls,
        additional_instruction=(
            "Verify the graduation year as stated in the answer. Expected is 1993. "
            "If the URLs indicate a different year than the answer, mark as not supported."
        ),
        extra_prerequisites=[edu_sources_exist]
    )


async def build_department_chair_checks(
    evaluator: Evaluator,
    parent_node,
    chair: DepartmentChairExtraction,
    university_for_context: Optional[str]
) -> None:
    """
    Build and execute verification checks for:
    - Department chair full name (existence)
    - Source URL existence
    - Role verification that the person is the current chair of the History Department at the specified university
    """
    dept_node = evaluator.add_parallel(
        id="Department_Chair",
        desc="Identify and verify the current chair of the History Department at Texas Tech University.",
        parent=parent_node,
        critical=True
    )

    # Existence of chair full name
    evaluator.add_custom_node(
        result=_non_empty(chair.name),
        id="Department_Chair_Full_Name",
        desc="Provide the full name of the current chair of the Texas Tech University History Department.",
        parent=dept_node,
        critical=True
    )

    # Source URL existence
    chair_src_exists = evaluator.add_custom_node(
        result=_list_non_empty(chair.source_urls),
        id="Chair_Source_URL",
        desc="Provide a publicly accessible URL confirming the chair role at Texas Tech University's History Department.",
        parent=dept_node,
        critical=True
    )

    # Role verification
    role_leaf = evaluator.add_leaf(
        id="Chair_Role_Verification",
        desc="Verify the person is the current chair of the Texas Tech University History Department (not another role or past chair).",
        parent=dept_node,
        critical=True
    )

    uni_name = university_for_context or "Texas Tech University"
    await evaluator.verify(
        claim=f"{chair.name or 'The identified person'} is the current chair of the Department of History at {uni_name}.",
        node=role_leaf,
        sources=chair.source_urls,
        additional_instruction=(
            "Confirm that the person holds the current chair role (e.g., 'Chair', 'Department Chair', 'Interim Chair' may qualify if clearly current). "
            "If the page indicates a past chair or a different role, or does not clearly support 'current chair', mark as not supported."
        ),
        extra_prerequisites=[chair_src_exists]
    )


async def build_book_checks(
    evaluator: Evaluator,
    parent_node,
    book: BookExtraction,
    chair_name: Optional[str]
) -> None:
    """
    Build and execute verification checks for:
    - Book sources provided
    - Book authored by the identified chair
    - Book's full title, publisher, and publication year (each verified)
    """
    book_node = evaluator.add_parallel(
        id="Chair_Authored_Book_With_Details",
        desc="Provide one book authored by the chair with full publication details.",
        parent=parent_node,
        critical=True
    )

    # Source URL existence
    book_src_exists = evaluator.add_custom_node(
        result=_list_non_empty(book.source_urls),
        id="Book_Source_URL",
        desc="Provide a publicly accessible URL that supports the book's authorship and publication details (title, publisher, year).",
        parent=book_node,
        critical=True
    )

    # Authored by chair
    authored_leaf = evaluator.add_leaf(
        id="Book_Authored_By_Chair",
        desc="Verify the identified book is authored (or co-authored) by the identified department chair.",
        parent=book_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The book titled '{book.title or ''}' is authored or co-authored by {chair_name or 'the identified chair'}.",
        node=authored_leaf,
        sources=book.source_urls,
        additional_instruction=(
            "Confirm that the chair is listed among the authors. Co-authorship qualifies as 'authored'. "
            "If the chair is not an author, mark as not supported."
        ),
        extra_prerequisites=[book_src_exists]
    )

    # Full title verification
    title_leaf = evaluator.add_leaf(
        id="Book_Full_Title",
        desc="Provide the book's full title.",
        parent=book_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The book's full title is '{book.title or ''}'.",
        node=title_leaf,
        sources=book.source_urls,
        additional_instruction=(
            "Verify the title exactly as provided in the answer. Allow minor punctuation or subtitle formatting differences. "
            "If the cited page shows a different title, mark as not supported."
        ),
        extra_prerequisites=[book_src_exists]
    )

    # Publisher verification
    publisher_leaf = evaluator.add_leaf(
        id="Book_Publisher",
        desc="Provide the publisher name for the book.",
        parent=book_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The publisher of the book '{book.title or ''}' is '{book.publisher or ''}'.",
        node=publisher_leaf,
        sources=book.source_urls,
        additional_instruction=(
            "Verify that the cited source lists the same publisher as the answer. "
            "If multiple editions/publishers exist, match the edition implied by the answer if possible."
        ),
        extra_prerequisites=[book_src_exists]
    )

    # Publication year verification
    pub_year_leaf = evaluator.add_leaf(
        id="Book_Publication_Year",
        desc="Provide the year of publication for the book.",
        parent=book_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The publication year of the book '{book.title or ''}' is '{book.publication_year or ''}'.",
        node=pub_year_leaf,
        sources=book.source_urls,
        additional_instruction=(
            "Verify the publication year as stated in the answer. "
            "If the source lists multiple years (e.g., original vs. new edition), prefer the year matching the answer's edition if clearly indicated."
        ),
        extra_prerequisites=[book_src_exists]
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
    Entry point to evaluate an agent's answer for the 2022 coach + TTU history + current chair + chair-authored book task.
    Returns a structured evaluation summary dictionary.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # The top-level task is sequential
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

    # Build a critical sequential node for the entire task (to enforce all four sub-requirements in order)
    task_root = evaluator.add_sequential(
        id="Complete_Research_Task",
        desc="Identify the 2022 multi-award-winning college football coach, confirm his 1993 BA history degree institution, identify the current history department chair at that university, and provide a book by that chair with publication details (all verifiable via public sources).",
        parent=root,
        critical=True
    )

    # Extract all structured information from the answer
    extracted: FullExtraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=FullExtraction,
        extraction_name="core_extraction"
    )

    # Record helpful expected cues as ground truth info (for transparency only; not used to auto-judge)
    evaluator.add_ground_truth({
        "expected_awards_2022": [
            "AP Coach of the Year",
            "Paul 'Bear' Bryant Coach of the Year",
            "Walter Camp Coach of the Year"
        ],
        "expected_education": {
            "institution": "Texas Tech University",
            "major": "History",
            "degree_type": "Bachelor's (e.g., BA)",
            "year": "1993"
        },
        "notes": "Verification relies on URLs provided in the answer; leaves are atomic checks."
    }, gt_type="expected_criteria")

    # Prepare defaults to avoid attribute errors
    coach = extracted.coach or CoachExtraction()
    chair = extracted.chair or DepartmentChairExtraction()
    book = extracted.book or BookExtraction()

    # 1) Coach + Education checks
    await build_coach_and_education_checks(evaluator, task_root, coach)

    # 2) Department Chair checks (for TTU History)
    # Use the coach.degree_institution for context; default to TTU
    await build_department_chair_checks(
        evaluator,
        task_root,
        chair,
        university_for_context=coach.degree_institution or "Texas Tech University"
    )

    # 3) Chair-authored book checks
    await build_book_checks(
        evaluator,
        task_root,
        book,
        chair_name=chair.name
    )

    # Final summary
    return evaluator.get_summary()