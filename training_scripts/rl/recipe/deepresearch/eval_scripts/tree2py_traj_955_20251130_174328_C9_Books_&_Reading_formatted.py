import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "literary_fiction_appalachia_2023_knopf_pulitzer_nba"
TASK_DESCRIPTION = (
    "Identify a literary fiction novel published in 2023 by Knopf that won the 2024 Pulitzer Prize for Fiction and "
    "was longlisted for the 2024 National Book Award, whose author was born in an Appalachian U.S. state in the 1950s, "
    "earned a Bachelor's degree from a university in their home state (graduating by the end of the 1970s), and "
    "completed an M.F.A. from the University of Iowa Writers' Workshop in the 1970s. The novel must be set in West "
    "Virginia within the Appalachian region during or after the American Civil War era (1860s-1870s) and must be set "
    "at least partially in a historical institution or asylum. Provide the book title, author name, and supporting "
    "reference URLs for: (1) author biographical information, (2) book publication and setting details, and "
    "(3) award recognition."
)

APPALACHIAN_STATES = [
    "Alabama", "Georgia", "Kentucky", "Maryland", "Mississippi", "New York",
    "North Carolina", "Ohio", "Pennsylvania", "South Carolina", "Tennessee",
    "Virginia", "West Virginia"
]

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ReferenceURLs(BaseModel):
    author_bio_urls: List[str] = Field(default_factory=list)
    book_pub_setting_urls: List[str] = Field(default_factory=list)
    awards_urls: List[str] = Field(default_factory=list)


class AuthorEducation(BaseModel):
    birth_state: Optional[str] = None
    birth_year: Optional[str] = None

    ba_university: Optional[str] = None
    ba_state: Optional[str] = None
    ba_grad_year: Optional[str] = None

    mfa_program: Optional[str] = None
    mfa_university: Optional[str] = None
    mfa_grad_year: Optional[str] = None


class BookDetails(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    pub_year: Optional[str] = None
    genre: Optional[str] = None
    setting_locations: List[str] = Field(default_factory=list)
    setting_region: Optional[str] = None
    setting_period: Optional[str] = None
    institution_asylum: Optional[str] = None


class AwardsInfo(BaseModel):
    pulitzer_2024_fiction_winner: Optional[str] = None
    nba_2024_longlist: Optional[str] = None


class MainExtraction(BaseModel):
    book: BookDetails = BookDetails()
    author: AuthorEducation = AuthorEducation()
    awards: AwardsInfo = AwardsInfo()
    references: ReferenceURLs = ReferenceURLs()


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_main() -> str:
    return (
        "Extract the key information about a single book and its author from the provided answer. "
        "Return the following fields exactly as presented:\n"
        "book:\n"
        "  - title: the book title\n"
        "  - author: the author name\n"
        "  - publisher: publisher/imprint (e.g., 'Alfred A. Knopf', 'Knopf')\n"
        "  - pub_year: publication year (string)\n"
        "  - genre: genre or format (e.g., 'novel', 'literary fiction')\n"
        "  - setting_locations: list of locations/states/cities (e.g., includes 'West Virginia')\n"
        "  - setting_region: if stated (e.g., 'Appalachian region')\n"
        "  - setting_period: historical period or years stated (e.g., '1860s', 'post-Civil War')\n"
        "  - institution_asylum: name/description if an institution/asylum setting is mentioned (string)\n"
        "author:\n"
        "  - birth_state: U.S. state of birth\n"
        "  - birth_year: year of birth (string)\n"
        "  - ba_university: name of BA university\n"
        "  - ba_state: state of BA university\n"
        "  - ba_grad_year: BA graduation year (string)\n"
        "  - mfa_program: program name (e.g., 'Iowa Writers' Workshop')\n"
        "  - mfa_university: university name (e.g., 'University of Iowa')\n"
        "  - mfa_grad_year: MFA completion year (string)\n"
        "awards:\n"
        "  - pulitzer_2024_fiction_winner: state if it won (e.g., 'won', 'winner', or any claim string)\n"
        "  - nba_2024_longlist: state if it was longlisted (e.g., 'longlisted', or any claim string)\n"
        "references:\n"
        "  - author_bio_urls: list of URLs cited for author bio/education\n"
        "  - book_pub_setting_urls: list of URLs cited for book publication/setting details\n"
        "  - awards_urls: list of URLs cited for awards recognition\n\n"
        "Rules:\n"
        "1) Extract only what is explicitly present in the answer; if missing, use null for strings or [] for lists.\n"
        "2) For URLs, include full URLs. If none are present for a category, return an empty list.\n"
        "3) Use strings for years and any numeric-like fields; do not convert to numbers.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_urls(urls: List[str]) -> Optional[List[str]]:
    return urls if (urls and len([u for u in urls if isinstance(u, str) and u.strip()])) else None


def _safe(s: Optional[str]) -> str:
    return s or ""


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_required_output_fields(
    evaluator: Evaluator,
    parent_node,
    ext: MainExtraction
) -> None:
    req_out = evaluator.add_parallel(
        id="Required_Output_Fields",
        desc="Answer includes the requested core fields.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(ext.book.title and ext.book.title.strip()),
        id="Provides_Book_Title",
        desc="Provides the book title.",
        parent=req_out,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(ext.book.author and ext.book.author.strip()),
        id="Provides_Author_Name",
        desc="Provides the author name.",
        parent=req_out,
        critical=True
    )


async def build_required_references(
    evaluator: Evaluator,
    parent_node,
    ext: MainExtraction
) -> None:
    req_refs = evaluator.add_parallel(
        id="Required_References",
        desc="Provides supporting reference URLs for each required category.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(_non_empty_urls(ext.references.author_bio_urls)),
        id="Author_Bio_Education_URLs",
        desc="Provides reference URL(s) supporting the author's biographical and education information.",
        parent=req_refs,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(_non_empty_urls(ext.references.book_pub_setting_urls)),
        id="Book_Publication_Setting_URLs",
        desc="Provides reference URL(s) supporting the book’s publication details and setting details.",
        parent=req_refs,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(_non_empty_urls(ext.references.awards_urls)),
        id="Awards_Recognition_URLs",
        desc="Provides reference URL(s) supporting the book’s award recognition.",
        parent=req_refs,
        critical=True
    )


async def build_author_criteria(
    evaluator: Evaluator,
    parent_node,
    ext: MainExtraction
) -> None:
    author_node = evaluator.add_parallel(
        id="Author_Criteria",
        desc="Author meets all specified biographical and educational constraints.",
        parent=parent_node,
        critical=True
    )

    author_urls = _non_empty_urls(ext.references.author_bio_urls)
    author_name = _safe(ext.book.author)
    birth_state = _safe(ext.author.birth_state)
    birth_year = _safe(ext.author.birth_year)
    ba_univ = _safe(ext.author.ba_university)
    ba_state = _safe(ext.author.ba_state)
    ba_year = _safe(ext.author.ba_grad_year)
    mfa_prog = _safe(ext.author.mfa_program)
    mfa_univ = _safe(ext.author.mfa_university)
    mfa_year = _safe(ext.author.mfa_grad_year)

    # Born in Appalachian State
    n1 = evaluator.add_leaf(
        id="Born_in_Appalachian_State",
        desc="Author was born in a U.S. state located in the Appalachian region.",
        parent=author_node,
        critical=True
    )
    claim1 = (
        f"The author {author_name} was born in {birth_state}, and {birth_state} is recognized as being in the Appalachian region."
    )
    add_ins1 = (
        "For Appalachian classification, treat the following U.S. states as in the Appalachian region (per ARC): "
        + ", ".join(APPALACHIAN_STATES)
        + ". Rely on the provided URL(s) to confirm birthplace; the Appalachian-state mapping is given here."
    )
    await evaluator.verify(
        claim=claim1,
        node=n1,
        sources=author_urls,
        additional_instruction=add_ins1
    )

    # Born in 1950s
    n2 = evaluator.add_leaf(
        id="Born_in_1950s",
        desc="Author's birth year is in the 1950s.",
        parent=author_node,
        critical=True
    )
    claim2 = (
        f"The author {author_name} was born in the 1950s; the birth year reported is {birth_year}, which should be between 1950 and 1959 inclusive."
    )
    add_ins2 = "Consider '1950s' as any year from 1950 to 1959 inclusive."
    await evaluator.verify(
        claim=claim2,
        node=n2,
        sources=author_urls,
        additional_instruction=add_ins2
    )

    # BA in home state, graduated by end of 1970s
    n3 = evaluator.add_leaf(
        id="BA_Home_State_University_Completed_by_End_of_1970s",
        desc="Author earned a Bachelor's degree from a university in their home state, graduating by the end of the 1970s.",
        parent=author_node,
        critical=True
    )
    claim3 = (
        f"The author {author_name} earned a Bachelor's degree from {ba_univ} in {ba_state}, the author's home state ({birth_state}), "
        f"and graduated by the end of the 1970s (graduation year {ba_year})."
    )
    add_ins3 = (
        "Home state refers to the birthplace state. Accept if the BA university is located in the same state as the birthplace. "
        "Graduated by the end of the 1970s means graduation year is 1979 or earlier."
    )
    await evaluator.verify(
        claim=claim3,
        node=n3,
        sources=author_urls,
        additional_instruction=add_ins3
    )

    # MFA Iowa Writers' Workshop in 1970s
    n4 = evaluator.add_leaf(
        id="MFA_Iowa_Writers_Workshop_Completed_in_1970s",
        desc="Author completed an M.F.A. from the University of Iowa Writers' Workshop, with completion/graduation in the 1970s.",
        parent=author_node,
        critical=True
    )
    claim4 = (
        f"The author {author_name} completed an M.F.A. from the University of Iowa Writers' Workshop ({mfa_prog} at {mfa_univ}) "
        f"in the 1970s (completion year {mfa_year})."
    )
    add_ins4 = (
        "Accept mentions of 'Iowa Writers' Workshop' or 'University of Iowa MFA' indicating completion in the 1970s (1970–1979). "
        "Allow minor phrasing variations."
    )
    await evaluator.verify(
        claim=claim4,
        node=n4,
        sources=author_urls,
        additional_instruction=add_ins4
    )


async def build_book_criteria(
    evaluator: Evaluator,
    parent_node,
    ext: MainExtraction
) -> None:
    book_node = evaluator.add_parallel(
        id="Book_Criteria",
        desc="Book meets all specified publication, genre/format, and setting constraints.",
        parent=parent_node,
        critical=True
    )

    book_urls = _non_empty_urls(ext.references.book_pub_setting_urls)
    title = _safe(ext.book.title)
    publisher = _safe(ext.book.publisher)
    pub_year = _safe(ext.book.pub_year)
    genre = _safe(ext.book.genre)
    setting_region = _safe(ext.book.setting_region)
    setting_period = _safe(ext.book.setting_period)
    institution_asylum = _safe(ext.book.institution_asylum)

    # Published in 2023
    b1 = evaluator.add_leaf(
        id="Published_in_2023",
        desc="Book was published in 2023.",
        parent=book_node,
        critical=True
    )
    claim_b1 = f"The book '{title}' was published in 2023 (reported year: {pub_year})."
    add_ins_b1 = "Check the publication year; accept any credible page statement that the publication year is 2023."
    await evaluator.verify(
        claim=claim_b1,
        node=b1,
        sources=book_urls,
        additional_instruction=add_ins_b1
    )

    # Published by Knopf
    b2 = evaluator.add_leaf(
        id="Published_by_Knopf",
        desc="Book was published by Knopf (Alfred A. Knopf).",
        parent=book_node,
        critical=True
    )
    claim_b2 = f"The book '{title}' was published by Alfred A. Knopf (Knopf). Reported publisher: {publisher}."
    add_ins_b2 = (
        "Accept 'Alfred A. Knopf', 'Knopf', or 'Knopf Doubleday' branding indicating Knopf as the publisher/imprint."
    )
    await evaluator.verify(
        claim=claim_b2,
        node=b2,
        sources=book_urls,
        additional_instruction=add_ins_b2
    )

    # Literary fiction novel
    b3 = evaluator.add_leaf(
        id="Is_Literary_Fiction_Novel",
        desc="Book is a literary fiction novel (not a memoir, short story collection, etc.).",
        parent=book_node,
        critical=True
    )
    claim_b3 = f"The book '{title}' is a literary fiction novel. Reported genre/format: {genre}."
    add_ins_b3 = (
        "Accept if the page describes the work as a 'novel' or 'literary fiction'. Do not accept 'memoir', 'short story collection', or other non-novel formats."
    )
    await evaluator.verify(
        claim=claim_b3,
        node=b3,
        sources=book_urls,
        additional_instruction=add_ins_b3
    )

    # Set in West Virginia
    b4 = evaluator.add_leaf(
        id="Set_in_West_Virginia",
        desc="Book is set in West Virginia.",
        parent=book_node,
        critical=True
    )
    claim_b4 = f"The book '{title}' is set in West Virginia."
    add_ins_b4 = (
        "Accept references to West Virginia locales (e.g., 'Weston State Hospital', 'Trans-Allegheny Lunatic Asylum', 'West Virginia') indicating the setting."
    )
    await evaluator.verify(
        claim=claim_b4,
        node=b4,
        sources=book_urls,
        additional_instruction=add_ins_b4
    )

    # Set in Appalachian region
    b5 = evaluator.add_leaf(
        id="Set_in_Appalachian_Region",
        desc="Book is set in the Appalachian region.",
        parent=book_node,
        critical=True
    )
    claim_b5 = f"The book '{title}' is set in the Appalachian region. Stated region: {setting_region}."
    add_ins_b5 = (
        "If the book is set in West Virginia or any of these states, consider it Appalachian: "
        + ", ".join(APPALACHIAN_STATES)
        + ". Use the URL(s) to confirm the setting location; Appalachian mapping provided here."
    )
    await evaluator.verify(
        claim=claim_b5,
        node=b5,
        sources=book_urls,
        additional_instruction=add_ins_b5
    )

    # Set during or after Civil War era (1860s–1870s)
    b6 = evaluator.add_leaf(
        id="Set_During_or_After_Civil_War_Era",
        desc="Book is set during or after the American Civil War era (1860s–1870s).",
        parent=book_node,
        critical=True
    )
    claim_b6 = (
        f"The book '{title}' is set during or after the American Civil War era (1860s–1870s). Reported period: {setting_period}."
    )
    add_ins_b6 = (
        "Accept if the setting explicitly mentions the Civil War period (circa 1861–1865), reconstruction/postbellum, "
        "or years in the 1860s–1870s, or later while clearly described as post–Civil War era context."
    )
    await evaluator.verify(
        claim=claim_b6,
        node=b6,
        sources=book_urls,
        additional_instruction=add_ins_b6
    )

    # Includes historical institution or asylum setting
    b7 = evaluator.add_leaf(
        id="Includes_Historical_Institution_or_Asylum_Setting",
        desc="Book is set at least partially in a historical institution or asylum.",
        parent=book_node,
        critical=True
    )
    claim_b7 = (
        f"The book '{title}' is set at least partially in a historical institution or asylum (e.g., {institution_asylum})."
    )
    add_ins_b7 = (
        "Qualifying examples include 'asylum', 'lunatic asylum', 'state hospital', 'sanatorium', or 'mental institution'. "
        "Partial setting is sufficient."
    )
    await evaluator.verify(
        claim=claim_b7,
        node=b7,
        sources=book_urls,
        additional_instruction=add_ins_b7
    )


async def build_awards_criteria(
    evaluator: Evaluator,
    parent_node,
    ext: MainExtraction
) -> None:
    awards_node = evaluator.add_parallel(
        id="Awards_Criteria",
        desc="Book meets all specified award/recognition constraints.",
        parent=parent_node,
        critical=True
    )

    awards_urls = _non_empty_urls(ext.references.awards_urls)
    title = _safe(ext.book.title)

    # Pulitzer Prize for Fiction (2024) winner
    a1 = evaluator.add_leaf(
        id="Won_2024_Pulitzer_Prize_for_Fiction",
        desc="Book won the 2024 Pulitzer Prize for Fiction.",
        parent=awards_node,
        critical=True
    )
    claim_a1 = f"The book '{title}' won the 2024 Pulitzer Prize for Fiction."
    add_ins_a1 = (
        "Confirm on authoritative sources (e.g., pulitzer.org or reputable news sites) that this title is listed as the 2024 winner in Fiction."
    )
    await evaluator.verify(
        claim=claim_a1,
        node=a1,
        sources=awards_urls,
        additional_instruction=add_ins_a1
    )

    # National Book Award longlist (2024)
    a2 = evaluator.add_leaf(
        id="Longlisted_2024_National_Book_Award",
        desc="Book was longlisted for the 2024 National Book Award.",
        parent=awards_node,
        critical=True
    )
    claim_a2 = f"The book '{title}' was longlisted for the 2024 National Book Award."
    add_ins_a2 = (
        "Confirm on the National Book Foundation site or reputable sources that this title appears on the 2024 longlist."
    )
    await evaluator.verify(
        claim=claim_a2,
        node=a2,
        sources=awards_urls,
        additional_instruction=add_ins_a2
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the complex literary fiction identification task with author, book, and awards constraints.
    """
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

    # Extract structured information from the answer
    ext = await evaluator.extract(
        prompt=prompt_extract_main(),
        template_class=MainExtraction,
        extraction_name="main_extraction"
    )

    # Record helper info for transparency
    evaluator.add_custom_info(
        info={"appalachian_states": APPALACHIAN_STATES},
        info_type="reference_list",
        info_name="appalachian_states_list"
    )

    # Build the critical task node
    task_node = evaluator.add_parallel(
        id="Book_Identification_Task",
        desc="Identify one literary fiction novel meeting all specified author/background, book/publication/setting, and awards constraints, and provide required outputs with supporting URLs.",
        parent=root,
        critical=True
    )

    # Required output fields
    await build_required_output_fields(evaluator, task_node, ext)

    # Required references
    await build_required_references(evaluator, task_node, ext)

    # Author criteria checks
    await build_author_criteria(evaluator, task_node, ext)

    # Book criteria checks
    await build_book_criteria(evaluator, task_node, ext)

    # Awards criteria checks
    await build_awards_criteria(evaluator, task_node, ext)

    # Return the full evaluation summary
    return evaluator.get_summary()