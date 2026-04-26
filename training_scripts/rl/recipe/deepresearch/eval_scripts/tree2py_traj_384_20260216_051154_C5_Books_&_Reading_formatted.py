import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "author_double_awards"
TASK_DESCRIPTION = """
An author won both the 2024 National Book Award for Fiction and the 2025 Pulitzer Prize for Fiction for the same novel. Identify this author and provide the following information with supporting reference URLs: 
(1) Award-Winning Novel Details: the title of the novel, the publication date (month and year) and publisher, and what classic American novel this work reimagines; 
(2) Author's Background: the year the author was born, the author's birthplace (city and state), and where the author grew up (city and state); 
(3) Current Academic Position: the university where the author currently works and the author's specific academic title/position; 
(4) Previous Literary Works: approximately how many books the author has published in total, the title and publication year of the author's novel that was shortlisted for the 2022 Booker Prize, and confirmation that the author published a novel titled 'Erasure' with the year it was published.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AwardSources(BaseModel):
    nba_2024_fiction_urls: List[str] = Field(default_factory=list)
    pulitzer_2025_fiction_urls: List[str] = Field(default_factory=list)


class NovelDetails(BaseModel):
    title: Optional[str] = None
    publication_month_year: Optional[str] = None
    publisher: Optional[str] = None
    reimagined_classic: Optional[str] = None

    publication_date_urls: List[str] = Field(default_factory=list)
    publisher_urls: List[str] = Field(default_factory=list)
    reimagined_urls: List[str] = Field(default_factory=list)


class AuthorBio(BaseModel):
    name: Optional[str] = None
    birth_year: Optional[str] = None
    birthplace_city_state: Optional[str] = None
    grew_up_city_state: Optional[str] = None

    birth_year_urls: List[str] = Field(default_factory=list)
    birthplace_urls: List[str] = Field(default_factory=list)
    grew_up_urls: List[str] = Field(default_factory=list)


class AcademicPosition(BaseModel):
    university: Optional[str] = None
    title: Optional[str] = None

    university_urls: List[str] = Field(default_factory=list)
    title_urls: List[str] = Field(default_factory=list)


class PriorWorks(BaseModel):
    approx_total_books: Optional[str] = None
    total_books_urls: List[str] = Field(default_factory=list)

    booker_shortlisted_title: Optional[str] = None
    booker_shortlisted_year: Optional[str] = None
    booker_urls: List[str] = Field(default_factory=list)

    erasure_year: Optional[str] = None
    erasure_urls: List[str] = Field(default_factory=list)


class TaskExtraction(BaseModel):
    author: Optional[AuthorBio] = None
    awards: Optional[AwardSources] = None
    novel: Optional[NovelDetails] = None
    academic: Optional[AcademicPosition] = None
    prior: Optional[PriorWorks] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_task_data() -> str:
    return """
    Extract the structured information explicitly provided in the answer regarding the author and the award-winning novel. 
    Return a JSON object with the following nested structure. Only extract values explicitly stated in the answer; if a field is missing, set it to null (for strings) or [] (for URL arrays).

    {
      "author": {
        "name": string | null,
        "birth_year": string | null,
        "birthplace_city_state": string | null,
        "grew_up_city_state": string | null,
        "birth_year_urls": string[] (supporting URLs for birth year),
        "birthplace_urls": string[] (supporting URLs for birthplace),
        "grew_up_urls": string[] (supporting URLs for where the author grew up)
      },
      "awards": {
        "nba_2024_fiction_urls": string[] (URLs that explicitly show the author won the 2024 National Book Award for Fiction for the specified novel),
        "pulitzer_2025_fiction_urls": string[] (URLs that explicitly show the author won the 2025 Pulitzer Prize for Fiction for the same novel)
      },
      "novel": {
        "title": string | null,
        "publication_month_year": string | null,  // e.g., "March 2024"
        "publisher": string | null,
        "reimagined_classic": string | null,     // the classic American novel this work reimagines
        "publication_date_urls": string[] (supporting URLs for publication month/year),
        "publisher_urls": string[] (supporting URLs for publisher),
        "reimagined_urls": string[] (supporting URLs for the reimagined-classic claim)
      },
      "academic": {
        "university": string | null,
        "title": string | null,
        "university_urls": string[] (supporting URLs for current university),
        "title_urls": string[] (supporting URLs for academic title/position)
      },
      "prior": {
        "approx_total_books": string | null,     // e.g., "over 30", "more than thirty", etc.
        "total_books_urls": string[] (supporting URLs for total books),
        "booker_shortlisted_title": string | null,
        "booker_shortlisted_year": string | null,
        "booker_urls": string[] (supporting URLs confirming book and year shortlisted for the 2022 Booker Prize),
        "erasure_year": string | null,
        "erasure_urls": string[] (supporting URLs confirming the author published 'Erasure' and its year)
      }
    }

    Special rules for URLs:
    - Extract only URLs explicitly present in the answer (plain URLs or URLs in markdown links).
    - Normalize any URLs missing protocol by prepending http:// if needed.
    - Do not invent URLs; if not present, return an empty array.

    Notes:
    - The answer may use abbreviations like "USC" for University of Southern California; extract the text as-is for "academic.university".
    - For counts like the total number of books, keep the textual approximation (e.g., "over 30") rather than attempting a precise number.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any((u or "").strip() for u in urls)


def _nonempty(s: Optional[str]) -> bool:
    return bool(s) and bool(s.strip())


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_awards_section(evaluator: Evaluator, parent_node, data: TaskExtraction) -> None:
    # Parallel node under root for identification and awards verification
    node = evaluator.add_parallel(
        id="identify_author_novel_and_awards",
        desc="Identify the author and the novel, and verify the award conditions (same novel won both awards)",
        parent=parent_node,
        critical=True
    )

    # Existence checks for author name and novel title
    author_name = data.author.name if data.author else None
    novel_title = data.novel.title if data.novel else None

    evaluator.add_custom_node(
        result=_nonempty(author_name),
        id="author_name_provided",
        desc="Answer provides the author's name",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_nonempty(novel_title),
        id="novel_title_provided",
        desc="Answer provides the title of the novel that won both awards",
        parent=node,
        critical=True
    )

    # NBA 2024 Fiction verification (must have supporting URL)
    nba_urls = data.awards.nba_2024_fiction_urls if (data.awards and data.awards.nba_2024_fiction_urls) else []
    if _has_urls(nba_urls):
        nba_node = evaluator.add_leaf(
            id="nba_2024_fiction_for_that_novel_verified_with_url",
            desc="Provides a supporting URL showing the author won the 2024 National Book Award for Fiction for that novel",
            parent=node,
            critical=True
        )
        claim_nba = f"{author_name or 'The author'} won the 2024 National Book Award for Fiction for the novel '{novel_title or ''}'."
        await evaluator.verify(
            claim=claim_nba,
            node=nba_node,
            sources=nba_urls,
            additional_instruction="Confirm that the page explicitly states: (a) National Book Award, (b) year 2024, (c) category Fiction, and (d) the award is for the specified novel."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="nba_2024_fiction_for_that_novel_verified_with_url",
            desc="Provides a supporting URL showing the author won the 2024 National Book Award for Fiction for that novel",
            parent=node,
            critical=True
        )

    # Pulitzer 2025 Fiction verification (must have supporting URL)
    pulitzer_urls = data.awards.pulitzer_2025_fiction_urls if (data.awards and data.awards.pulitzer_2025_fiction_urls) else []
    if _has_urls(pulitzer_urls):
        pul_node = evaluator.add_leaf(
            id="pulitzer_2025_fiction_for_that_novel_verified_with_url",
            desc="Provides a supporting URL showing the author won the 2025 Pulitzer Prize for Fiction for that same novel",
            parent=node,
            critical=True
        )
        claim_pul = f"{author_name or 'The author'} won the 2025 Pulitzer Prize for Fiction for the novel '{novel_title or ''}'."
        await evaluator.verify(
            claim=claim_pul,
            node=pul_node,
            sources=pulitzer_urls,
            additional_instruction="Confirm that the page explicitly states: (a) Pulitzer Prize, (b) year 2025, (c) category Fiction, and (d) the award is for the same novel."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="pulitzer_2025_fiction_for_that_novel_verified_with_url",
            desc="Provides a supporting URL showing the author won the 2025 Pulitzer Prize for Fiction for that same novel",
            parent=node,
            critical=True
        )

    # Explicit same-novel mapping within the answer text
    same_mapping_node = evaluator.add_leaf(
        id="explicit_same_novel_mapping",
        desc="Answer explicitly indicates both awards were for the same novel (clear award ↔ novel mapping)",
        parent=node,
        critical=True
    )
    mapping_claim = f"The answer explicitly states that both the 2024 National Book Award for Fiction and the 2025 Pulitzer Prize for Fiction were awarded for the novel '{novel_title or ''}'."
    await evaluator.verify(
        claim=mapping_claim,
        node=same_mapping_node,
        additional_instruction="Judge based solely on the answer text. It must clearly state that both awards were for the same specific novel. Allow reasonable synonyms or abbreviations for the award names."
    )


async def build_novel_details_section(evaluator: Evaluator, parent_node, data: TaskExtraction) -> None:
    node = evaluator.add_parallel(
        id="award_winning_novel_details",
        desc="Provide required details about the award-winning novel and satisfy the novel-related constraints",
        parent=parent_node,
        critical=True
    )

    novel_title = data.novel.title if data.novel else None

    # Publication month/year equals March 2024 (must have supporting URL)
    pub_urls = data.novel.publication_date_urls if (data.novel and data.novel.publication_date_urls) else []
    if _has_urls(pub_urls):
        pub_node = evaluator.add_leaf(
            id="publication_month_year_equals_march_2024",
            desc="States the novel's publication date (month and year) and it matches the constraint: March 2024 (with supporting URL)",
            parent=node,
            critical=True
        )
        claim_pub = f"The publication date of the novel '{novel_title or ''}' is March 2024."
        await evaluator.verify(
            claim=claim_pub,
            node=pub_node,
            sources=pub_urls,
            additional_instruction="Verify the source explicitly indicates the book's publication month and year as March 2024 (month-year is sufficient)."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="publication_month_year_equals_march_2024",
            desc="States the novel's publication date (month and year) and it matches the constraint: March 2024 (with supporting URL)",
            parent=node,
            critical=True
        )

    # Publisher (must have value and supporting URL)
    publisher = data.novel.publisher if data.novel else None
    publisher_urls = data.novel.publisher_urls if (data.novel and data.novel.publisher_urls) else []
    if _nonempty(publisher) and _has_urls(publisher_urls):
        pubr_node = evaluator.add_leaf(
            id="publisher_provided_with_url",
            desc="Provides the novel's publisher with a supporting URL",
            parent=node,
            critical=True
        )
        claim_publisher = f"The publisher of the novel '{novel_title or ''}' is {publisher}."
        await evaluator.verify(
            claim=claim_publisher,
            node=pubr_node,
            sources=publisher_urls,
            additional_instruction="Confirm the named entity is the primary publisher of the book (trade publisher or imprint linked to a publisher). Minor imprint naming variations are acceptable."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="publisher_provided_with_url",
            desc="Provides the novel's publisher with a supporting URL",
            parent=node,
            critical=True
        )

    # Reimagines Huckleberry Finn (must have supporting URL)
    reimag_urls = data.novel.reimagined_urls if (data.novel and data.novel.reimagined_urls) else []
    if _has_urls(reimag_urls):
        re_node = evaluator.add_leaf(
            id="reimagines_huckleberry_finn",
            desc="Identifies the classic American novel reimagined and it matches the constraint: Mark Twain's 'Adventures of Huckleberry Finn' (with supporting URL)",
            parent=node,
            critical=True
        )
        claim_re = f"The novel '{novel_title or ''}' reimagines Mark Twain's 'Adventures of Huckleberry Finn'."
        await evaluator.verify(
            claim=claim_re,
            node=re_node,
            sources=reimag_urls,
            additional_instruction="The source should clearly state that the work is a reimagining/retelling of 'Adventures of Huckleberry Finn'. Allow phrasing variations such as 'retelling', 'reworking', or 'reinterpretation'."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="reimagines_huckleberry_finn",
            desc="Identifies the classic American novel reimagined and it matches the constraint: Mark Twain's 'Adventures of Huckleberry Finn' (with supporting URL)",
            parent=node,
            critical=True
        )


async def build_author_background_section(evaluator: Evaluator, parent_node, data: TaskExtraction) -> None:
    node = evaluator.add_parallel(
        id="author_background",
        desc="Provide the author's background details and satisfy the bio-related constraints",
        parent=parent_node,
        critical=True
    )

    # Birth year equals 1956
    by_urls = data.author.birth_year_urls if (data.author and data.author.birth_year_urls) else []
    if _has_urls(by_urls):
        by_node = evaluator.add_leaf(
            id="birth_year_equals_1956",
            desc="Provides the author's birth year and it matches the constraint: 1956 (with supporting URL)",
            parent=node,
            critical=True
        )
        claim_by = "The author's birth year is 1956."
        await evaluator.verify(
            claim=claim_by,
            node=by_node,
            sources=by_urls,
            additional_instruction="Confirm the page explicitly states the author was born in 1956."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="birth_year_equals_1956",
            desc="Provides the author's birth year and it matches the constraint: 1956 (with supporting URL)",
            parent=node,
            critical=True
        )

    # Birthplace equals Fort Gordon, Georgia
    bp_urls = data.author.birthplace_urls if (data.author and data.author.birthplace_urls) else []
    if _has_urls(bp_urls):
        bp_node = evaluator.add_leaf(
            id="birthplace_equals_fort_gordon_georgia",
            desc="Provides the author's birthplace (as place + state) and it matches the constraint: Fort Gordon, Georgia (with supporting URL)",
            parent=node,
            critical=True
        )
        claim_bp = "The author's birthplace is Fort Gordon, Georgia."
        await evaluator.verify(
            claim=claim_bp,
            node=bp_node,
            sources=bp_urls,
            additional_instruction="Note: Fort Gordon was renamed Fort Eisenhower in 2023; sources may refer to either name. Accept references making clear the same location in Georgia."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="birthplace_equals_fort_gordon_georgia",
            desc="Provides the author's birthplace (as place + state) and it matches the constraint: Fort Gordon, Georgia (with supporting URL)",
            parent=node,
            critical=True
        )

    # Grew up equals Columbia, South Carolina
    gu_urls = data.author.grew_up_urls if (data.author and data.author.grew_up_urls) else []
    if _has_urls(gu_urls):
        gu_node = evaluator.add_leaf(
            id="grew_up_equals_columbia_south_carolina",
            desc="Provides where the author grew up (city and state) and it matches the constraint: Columbia, South Carolina (with supporting URL)",
            parent=node,
            critical=True
        )
        claim_gu = "The author grew up in Columbia, South Carolina."
        await evaluator.verify(
            claim=claim_gu,
            node=gu_node,
            sources=gu_urls,
            additional_instruction="Confirm the page states the author grew up in Columbia, South Carolina (allow phrasing variations like 'raised in')."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="grew_up_equals_columbia_south_carolina",
            desc="Provides where the author grew up (city and state) and it matches the constraint: Columbia, South Carolina (with supporting URL)",
            parent=node,
            critical=True
        )


async def build_academic_position_section(evaluator: Evaluator, parent_node, data: TaskExtraction) -> None:
    node = evaluator.add_parallel(
        id="current_academic_position",
        desc="Provide the author's current academic employment details and satisfy the position-related constraints",
        parent=parent_node,
        critical=True
    )

    # Current university equals USC
    univ_urls = data.academic.university_urls if (data.academic and data.academic.university_urls) else []
    if _has_urls(univ_urls):
        univ_node = evaluator.add_leaf(
            id="current_university_equals_usc",
            desc="Provides the university where the author currently works and it matches the constraint: University of Southern California (with supporting URL)",
            parent=node,
            critical=True
        )
        claim_univ = "The author currently works at the University of Southern California (USC)."
        await evaluator.verify(
            claim=claim_univ,
            node=univ_node,
            sources=univ_urls,
            additional_instruction="Accept 'USC' as abbreviation for University of Southern California. The source should clearly identify present affiliation (e.g., faculty profile)."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="current_university_equals_usc",
            desc="Provides the university where the author currently works and it matches the constraint: University of Southern California (with supporting URL)",
            parent=node,
            critical=True
        )

    # Academic title equals Distinguished Professor of English
    title_urls = data.academic.title_urls if (data.academic and data.academic.title_urls) else []
    if _has_urls(title_urls):
        title_node = evaluator.add_leaf(
            id="academic_title_equals_distinguished_professor_of_english",
            desc="Provides the author's specific academic title/position and it matches the constraint: Distinguished Professor of English (with supporting URL)",
            parent=node,
            critical=True
        )
        claim_title = "The author's specific academic title/position is Distinguished Professor of English."
        await evaluator.verify(
            claim=claim_title,
            node=title_node,
            sources=title_urls,
            additional_instruction="Allow minor phrasing variants such as 'Distinguished Professor of English at USC' or 'USC Distinguished Professor of English'."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="academic_title_equals_distinguished_professor_of_english",
            desc="Provides the author's specific academic title/position and it matches the constraint: Distinguished Professor of English (with supporting URL)",
            parent=node,
            critical=True
        )


async def build_prior_works_section(evaluator: Evaluator, parent_node, data: TaskExtraction) -> None:
    node = evaluator.add_parallel(
        id="previous_literary_works",
        desc="Provide required details about the author's prior works and satisfy the prior-works constraints",
        parent=parent_node,
        critical=True
    )

    # Total books published over 30
    tb_urls = data.prior.total_books_urls if (data.prior and data.prior.total_books_urls) else []
    if _has_urls(tb_urls):
        tb_node = evaluator.add_leaf(
            id="total_books_published_over_30",
            desc="Provides an approximate total count of books published and it matches the constraint: over 30 books (with supporting URL)",
            parent=node,
            critical=True
        )
        claim_tb = "The author has published over 30 books in total."
        await evaluator.verify(
            claim=claim_tb,
            node=tb_node,
            sources=tb_urls,
            additional_instruction="The source should state a total count exceeding 30. Allow approximations such as 'over thirty', 'more than 30', or 'about 35'."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="total_books_published_over_30",
            desc="Provides an approximate total count of books published and it matches the constraint: over 30 books (with supporting URL)",
            parent=node,
            critical=True
        )

    # Booker 2022 shortlisted novel is 'The Trees' (2021)
    bk_urls = data.prior.booker_urls if (data.prior and data.prior.booker_urls) else []
    if _has_urls(bk_urls):
        bk_node = evaluator.add_leaf(
            id="booker_2022_shortlisted_novel_is_the_trees_2021",
            desc="Provides the title and publication year of the novel shortlisted for the 2022 Booker Prize and it matches the constraints: 'The Trees' (2021) (with supporting URL)",
            parent=node,
            critical=True
        )
        claim_bk = "The author's novel 'The Trees' (published in 2021) was shortlisted for the 2022 Booker Prize."
        await evaluator.verify(
            claim=claim_bk,
            node=bk_node,
            sources=bk_urls,
            additional_instruction="The source should confirm both the shortlist status for the 2022 Booker Prize and that 'The Trees' was published in 2021."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="booker_2022_shortlisted_novel_is_the_trees_2021",
            desc="Provides the title and publication year of the novel shortlisted for the 2022 Booker Prize and it matches the constraints: 'The Trees' (2021) (with supporting URL)",
            parent=node,
            critical=True
        )

    # 'Erasure' confirmed and year 2001
    er_urls = data.prior.erasure_urls if (data.prior and data.prior.erasure_urls) else []
    if _has_urls(er_urls):
        er_node = evaluator.add_leaf(
            id="erasure_confirmed_and_year_2001",
            desc="Confirms the author published a novel titled 'Erasure' and provides the publication year matching the constraint: 2001 (with supporting URL)",
            parent=node,
            critical=True
        )
        claim_er = "The author published a novel titled 'Erasure' in 2001."
        await evaluator.verify(
            claim=claim_er,
            node=er_node,
            sources=er_urls,
            additional_instruction="Confirm both the title 'Erasure' and the year 2001 for its publication."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="erasure_confirmed_and_year_2001",
            desc="Confirms the author published a novel titled 'Erasure' and provides the publication year matching the constraint: 2001 (with supporting URL)",
            parent=node,
            critical=True
        )


def _collect_all_required_url_sets(data: TaskExtraction) -> Dict[str, List[str]]:
    """Collect all required URL categories for the final supporting URLs existence check."""
    result = {
        "nba_2024_fiction_urls": (data.awards.nba_2024_fiction_urls if data.awards else []),
        "pulitzer_2025_fiction_urls": (data.awards.pulitzer_2025_fiction_urls if data.awards else []),
        "publication_date_urls": (data.novel.publication_date_urls if data.novel else []),
        "publisher_urls": (data.novel.publisher_urls if data.novel else []),
        "reimagined_urls": (data.novel.reimagined_urls if data.novel else []),
        "birth_year_urls": (data.author.birth_year_urls if data.author else []),
        "birthplace_urls": (data.author.birthplace_urls if data.author else []),
        "grew_up_urls": (data.author.grew_up_urls if data.author else []),
        "university_urls": (data.academic.university_urls if data.academic else []),
        "title_urls": (data.academic.title_urls if data.academic else []),
        "total_books_urls": (data.prior.total_books_urls if data.prior else []),
        "booker_urls": (data.prior.booker_urls if data.prior else []),
        "erasure_urls": (data.prior.erasure_urls if data.prior else []),
    }
    return result


async def build_supporting_urls_presence_node(evaluator: Evaluator, parent_node, data: TaskExtraction) -> None:
    url_sets = _collect_all_required_url_sets(data)
    # Require at least one URL for every key claim category
    all_present = all(_has_urls(urls) for urls in url_sets.values())

    evaluator.add_custom_node(
        result=all_present,
        id="supporting_reference_urls_present",
        desc="Provides supporting reference URLs substantiating the key claims (awards, novel details, bio, academic position, and prior works)",
        parent=parent_node,
        critical=True
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
    Evaluate the answer for the author double-awards task.
    """
    # Initialize evaluator with sequential root to enforce order of major sections
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured information from the answer
    data = await evaluator.extract(
        prompt=prompt_extract_task_data(),
        template_class=TaskExtraction,
        extraction_name="extracted_author_and_novel_data",
    )

    # Add ground truth constraint info for context (not used for scoring)
    evaluator.add_ground_truth({
        "constraints": {
            "awards": ["2024 National Book Award for Fiction", "2025 Pulitzer Prize for Fiction"],
            "novel_publication_month_year": "March 2024",
            "reimagined_classic": "Adventures of Huckleberry Finn",
            "bio": {
                "birth_year": "1956",
                "birthplace": "Fort Gordon, Georgia",
                "grew_up": "Columbia, South Carolina"
            },
            "academic": {
                "university": "University of Southern California (USC)",
                "title": "Distinguished Professor of English"
            },
            "prior_works": {
                "total_books": "over 30",
                "booker_shortlist": {"title": "The Trees", "year": "2021", "prize_year": "2022"},
                "erasure": {"year": "2001"}
            }
        }
    }, gt_type="ground_truth_constraints")

    # Build verification tree sections in order (sequential root will short-circuit on failures)
    await build_awards_section(evaluator, root, data)
    await build_novel_details_section(evaluator, root, data)
    await build_author_background_section(evaluator, root, data)
    await build_academic_position_section(evaluator, root, data)
    await build_prior_works_section(evaluator, root, data)
    await build_supporting_urls_presence_node(evaluator, root, data)

    # Return summary
    return evaluator.get_summary()