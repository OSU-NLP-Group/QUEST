import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "litfic_2024_awards_triple"
TASK_DESCRIPTION = """
Identify the literary fiction book published in 2024 that meets ALL of the following criteria:

Award Achievements:
- Won the 2024 National Book Award for Fiction (with the ceremony held at Cipriani Wall Street in New York City)
- Won the 2025 Pulitzer Prize for Fiction (announced by Columbia University)
- Was shortlisted for the 2024 Booker Prize

Publisher Requirements:
- Published by an imprint of one of the Big Five publishing houses (Penguin Random House, HarperCollins, Simon & Schuster, Hachette Book Group, or Macmillan)
- The imprint must specialize in or be known for publishing literary fiction

Author Requirements:
- The author must currently hold a professor or teaching position at a university
- The author must reside in the United States

Literary Festival Connection:
- The book or author must have been featured at a major U.S. book festival
- This book festival must take place at or near a U.S. state capital

For your answer, provide:
1. The book title
2. The author's name
3. The publication year
4. The specific imprint that published the book
5. The parent company (Big Five publisher) of that imprint
6. The university where the author teaches
7. The U.S. state where the author resides
8. The name of the book festival and the state capital where it is held

Include URL references from reliable sources to verify each major claim.
"""

BIG_FIVE_PUBLISHERS = [
    "Penguin Random House",
    "HarperCollins",
    "Simon & Schuster",
    "Hachette Book Group",
    "Macmillan",
    "Macmillan Publishers"
]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BookCandidate(BaseModel):
    # Core fields
    book_title: Optional[str] = None
    author_name: Optional[str] = None
    publication_year: Optional[str] = None
    imprint: Optional[str] = None
    parent_company: Optional[str] = None
    university: Optional[str] = None
    author_us_state: Optional[str] = None
    festival_name: Optional[str] = None
    festival_state_capital: Optional[str] = None

    # URL references for verification
    bibliographic_urls: List[str] = Field(default_factory=list)  # title/author/year support
    classification_urls: List[str] = Field(default_factory=list)  # literary fiction classification
    nba_win_urls: List[str] = Field(default_factory=list)  # 2024 NBA win for Fiction
    nba_ceremony_urls: List[str] = Field(default_factory=list)  # Cipriani Wall Street venue evidence
    pulitzer_win_urls: List[str] = Field(default_factory=list)  # 2025 Pulitzer Fiction win
    pulitzer_announcer_urls: List[str] = Field(default_factory=list)  # Columbia University announcement/admin
    booker_shortlist_urls: List[str] = Field(default_factory=list)  # 2024 Booker shortlist
    imprint_parent_urls: List[str] = Field(default_factory=list)  # imprint belongs to parent company
    imprint_lit_urls: List[str] = Field(default_factory=list)  # imprint known for literary fiction
    author_teaching_urls: List[str] = Field(default_factory=list)  # author's university position
    author_residence_urls: List[str] = Field(default_factory=list)  # author's U.S. residence evidence
    festival_feature_urls: List[str] = Field(default_factory=list)  # author/book featured at festival
    festival_location_urls: List[str] = Field(default_factory=list)  # festival at/near capital evidence


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_book_candidate() -> str:
    return """
    Extract exactly one (1) candidate book from the answer that best matches the task requirements. 
    If multiple books are presented, select the first valid one and extract only that single record.

    Return a JSON object with these fields (use null for any that are missing):
    - book_title: The exact title of the identified book.
    - author_name: The author's full name.
    - publication_year: The stated publication year for the edition relevant to the awards (prefer the first US publication year if multiple are listed).
    - imprint: The specific publishing imprint for the book.
    - parent_company: The Big Five publisher that owns that imprint (e.g., Penguin Random House, HarperCollins, Simon & Schuster, Hachette Book Group, or Macmillan).
    - university: The university where the author currently teaches (e.g., faculty page).
    - author_us_state: The U.S. state where the author resides.
    - festival_name: The name of a major U.S. book festival where the author or the book was featured.
    - festival_state_capital: The U.S. state capital where that festival takes place (or near which it is held).
    
    Also extract URL lists that support each major claim from the answer (include only URLs explicitly present in the answer; do not invent any):
    - bibliographic_urls: URLs supporting title/author/year (e.g., publisher page, reputable media, book database).
    - classification_urls: URLs that classify/describe the book as literary fiction (or a literary novel in fiction).
    - nba_win_urls: URLs confirming the book won the 2024 National Book Award for Fiction.
    - nba_ceremony_urls: URLs confirming the 2024 National Book Awards ceremony took place at Cipriani Wall Street in New York City.
    - pulitzer_win_urls: URLs confirming the book won the 2025 Pulitzer Prize for Fiction.
    - pulitzer_announcer_urls: URLs indicating that the Pulitzer Prize is announced/administered by Columbia University (or clearly announced by Columbia University for 2025 winners).
    - booker_shortlist_urls: URLs confirming the book was shortlisted for the 2024 Booker Prize.
    - imprint_parent_urls: URLs showing the imprint belongs to the stated parent company.
    - imprint_lit_urls: URLs showing the imprint is known for or specializes in literary fiction.
    - author_teaching_urls: URLs proving the author currently holds a professor/teaching position at a university.
    - author_residence_urls: URLs indicating the author resides in the United States (e.g., "lives in [City], [State]" or similar).
    - festival_feature_urls: URLs confirming the festival appearance/feature of the author or the book.
    - festival_location_urls: URLs confirming the festival takes place at or near the stated U.S. state capital.
    
    SPECIAL RULES FOR URL EXTRACTION:
    - Extract only URLs explicitly present in the answer. Do not create or infer any URLs.
    - Extract valid and complete URLs. If a URL is missing a protocol, prepend http://
    - If no URL is provided for a category, return an empty array for that category.

    Ensure all arrays are present in the JSON output even if empty.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(lst: Optional[List[str]]) -> List[str]:
    return lst if lst else []


def _safe_text(text: Optional[str], fallback: str = "") -> str:
    return text.strip() if text else fallback


def _unique_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in url_lists:
        for u in lst:
            if u and u not in seen:
                seen.add(u)
                out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_content_constraints(evaluator: Evaluator, parent_node, book: BookCandidate) -> None:
    """
    Build and verify the 'Book_Meets_All_Content_Constraints' subtree.
    """
    content_node = evaluator.add_parallel(
        id="Book_Meets_All_Content_Constraints",
        desc="The identified book satisfies every constraint stated in the proposed question/constraints list.",
        parent=parent_node,
        critical=True
    )

    # 1) Published in 2024
    pub2024_node = evaluator.add_leaf(
        id="Published_In_2024",
        desc="The book's publication year is 2024.",
        parent=content_node,
        critical=True
    )
    claim_pub = f"The publication year of '{_safe_text(book.book_title, 'the book')}' by {_safe_text(book.author_name, 'the author')} is 2024."
    await evaluator.verify(
        claim=claim_pub,
        node=pub2024_node,
        sources=_safe_list(book.bibliographic_urls),
        additional_instruction="Verify the publication year explicitly. If multiple editions exist, prioritize the 2024 U.S. publication date for the relevant edition."
    )

    # 2) Literary fiction classification
    litfic_node = evaluator.add_leaf(
        id="Book_Is_Literary_Fiction",
        desc="A reliable source classifies/describes the book as literary fiction (or clearly as a literary novel in the fiction category).",
        parent=content_node,
        critical=True
    )
    claim_lit = f"'{_safe_text(book.book_title, 'This book')}' is a literary fiction novel (or clearly categorized as literary fiction)."
    await evaluator.verify(
        claim=claim_lit,
        node=litfic_node,
        sources=_safe_list(book.classification_urls),
        additional_instruction="Accept formulations like 'literary novel' or 'literary fiction'. Do not accept solely 'genre fiction' without literary descriptors."
    )

    # 3) Award achievements
    awards_node = evaluator.add_parallel(
        id="Award_Achievements_Met",
        desc="Book meets all award constraints (NBA win + venue detail, Pulitzer win + Columbia announcement detail, Booker shortlist).",
        parent=content_node,
        critical=True
    )

    # 3.1) National Book Award win 2024 (Fiction)
    nba_win_node = evaluator.add_leaf(
        id="Won_2024_National_Book_Award_Fiction",
        desc="Book won the 2024 National Book Award for Fiction.",
        parent=awards_node,
        critical=True
    )
    claim_nba_win = f"'{_safe_text(book.book_title, 'This book')}' won the 2024 National Book Award for Fiction."
    await evaluator.verify(
        claim=claim_nba_win,
        node=nba_win_node,
        sources=_safe_list(book.nba_win_urls),
        additional_instruction="Prefer official National Book Foundation pages or authoritative coverage explicitly naming the winner."
    )

    # 3.2) NBA ceremony venue: Cipriani Wall Street (NYC)
    nba_ceremony_node = evaluator.add_leaf(
        id="NBA_Ceremony_Held_At_Cipriani_Wall_Street_NYC",
        desc="The 2024 National Book Awards ceremony took place at Cipriani Wall Street in New York City.",
        parent=awards_node,
        critical=True
    )
    claim_nba_ceremony = "The 2024 National Book Awards ceremony took place at Cipriani Wall Street in New York City."
    await evaluator.verify(
        claim=claim_nba_ceremony,
        node=nba_ceremony_node,
        sources=_safe_list(book.nba_ceremony_urls),
        additional_instruction="Look for event/venue details stating 'Cipriani Wall Street' in NYC for the 2024 ceremony."
    )

    # 3.3) Pulitzer Prize for Fiction 2025: winner
    pulitzer_win_node = evaluator.add_leaf(
        id="Won_2025_Pulitzer_Prize_Fiction",
        desc="Book won the 2025 Pulitzer Prize for Fiction.",
        parent=awards_node,
        critical=True
    )
    claim_pulitzer_win = f"'{_safe_text(book.book_title, 'This book')}' won the 2025 Pulitzer Prize for Fiction."
    await evaluator.verify(
        claim=claim_pulitzer_win,
        node=pulitzer_win_node,
        sources=_safe_list(book.pulitzer_win_urls),
        additional_instruction="Confirm that the book is the winner (not finalist) for the Fiction category in 2025."
    )

    # 3.4) Pulitzer announced by Columbia University
    pulitzer_announce_node = evaluator.add_leaf(
        id="Pulitzer_Announced_By_Columbia_University",
        desc="The 2025 Pulitzer Prize for Fiction was announced by Columbia University.",
        parent=awards_node,
        critical=True
    )
    claim_pulitzer_announce = "The 2025 Pulitzer Prize for Fiction was announced by Columbia University."
    await evaluator.verify(
        claim=claim_pulitzer_announce,
        node=pulitzer_announce_node,
        sources=_safe_list(book.pulitzer_announcer_urls),
        additional_instruction="Evidence like 'administered by Columbia University' or explicit announcement attribution to Columbia University is acceptable."
    )

    # 3.5) Booker Prize 2024 shortlist
    booker_shortlist_node = evaluator.add_leaf(
        id="Shortlisted_For_2024_Booker_Prize",
        desc="Book was shortlisted for the 2024 Booker Prize.",
        parent=awards_node,
        critical=True
    )
    claim_booker = f"'{_safe_text(book.book_title, 'This book')}' was shortlisted for the 2024 Booker Prize."
    await evaluator.verify(
        claim=claim_booker,
        node=booker_shortlist_node,
        sources=_safe_list(book.booker_shortlist_urls),
        additional_instruction="Ensure it is the 'shortlist' specifically (not merely 'longlist' or 'nominated'). Prefer official Booker Prize website."
    )

    # 4) Publisher requirements
    publisher_node = evaluator.add_parallel(
        id="Publisher_Requirements_Met",
        desc="Publishing imprint and parent company satisfy Big Five + literary fiction imprint constraints.",
        parent=content_node,
        critical=True
    )

    # 4.1) Imprint under Big Five parent
    imprint_parent_node = evaluator.add_leaf(
        id="Imprint_Is_Under_Big_Five_Parent",
        desc="The book's imprint is an imprint of one of the Big Five publishing houses listed in the question.",
        parent=publisher_node,
        critical=True
    )
    claim_imprint_parent = (
        f"The imprint '{_safe_text(book.imprint, 'the imprint')}' is an imprint of "
        f"'{_safe_text(book.parent_company, 'the parent company')}', which is one of the Big Five publishers "
        f"(Penguin Random House, HarperCollins, Simon & Schuster, Hachette Book Group, or Macmillan)."
    )
    await evaluator.verify(
        claim=claim_imprint_parent,
        node=imprint_parent_node,
        sources=_safe_list(book.imprint_parent_urls),
        additional_instruction="Use the provided URLs to confirm the imprint-parent relationship. You may use general knowledge to determine whether the named parent company is in the Big Five list provided."
    )

    # 4.2) Imprint known for literary fiction
    imprint_lit_node = evaluator.add_leaf(
        id="Imprint_Known_For_Literary_Fiction",
        desc="The imprint specializes in or is known for publishing literary fiction.",
        parent=publisher_node,
        critical=True
    )
    claim_imprint_lit = f"The imprint '{_safe_text(book.imprint, 'the imprint')}' specializes in or is known for publishing literary fiction."
    await evaluator.verify(
        claim=claim_imprint_lit,
        node=imprint_lit_node,
        sources=_safe_list(book.imprint_lit_urls),
        additional_instruction="Accept language indicating a primary or strong focus on literary fiction on publisher/imprint pages or reputable sources."
    )

    # 5) Author requirements
    author_node = evaluator.add_parallel(
        id="Author_Requirements_Met",
        desc="Author satisfies teaching-position and U.S.-residency constraints.",
        parent=content_node,
        critical=True
    )

    # 5.1) Current university teaching position
    teaching_node = evaluator.add_leaf(
        id="Author_Has_Current_University_Teaching_Position",
        desc="Author currently holds a professor or teaching position at a university.",
        parent=author_node,
        critical=True
    )
    claim_teach = f"{_safe_text(book.author_name, 'The author')} currently holds a professor or teaching position at a university."
    await evaluator.verify(
        claim=claim_teach,
        node=teaching_node,
        sources=_safe_list(book.author_teaching_urls),
        additional_instruction="Accept titles such as professor (any rank), lecturer, instructor, visiting professor, writer-in-residence, or similar active teaching/appointment roles."
    )

    # 5.2) Author resides in the United States
    resides_node = evaluator.add_leaf(
        id="Author_Resides_In_United_States",
        desc="Author resides in the United States.",
        parent=author_node,
        critical=True
    )
    claim_resides = f"{_safe_text(book.author_name, 'The author')} resides in the United States."
    await evaluator.verify(
        claim=claim_resides,
        node=resides_node,
        sources=_safe_list(book.author_residence_urls),
        additional_instruction="Evidence like 'lives in [City], [State]' or 'U.S.-based' is acceptable if clearly tied to current residence."
    )

    # 6) Festival connection
    festival_node = evaluator.add_parallel(
        id="Festival_Connection_Requirements_Met",
        desc="Book/author festival feature + state-capital proximity constraints are met.",
        parent=content_node,
        critical=True
    )

    # 6.1) Featured at a major U.S. book festival
    featured_node = evaluator.add_leaf(
        id="Featured_At_Major_US_Book_Festival",
        desc="The book or the author was featured at a major U.S. book festival.",
        parent=festival_node,
        critical=True
    )
    claim_featured = (
        f"The book '{_safe_text(book.book_title, 'the book')}' or author '{_safe_text(book.author_name, 'the author')}' "
        f"was featured/appeared at the { _safe_text(book.festival_name, 'named festival') }."
    )
    await evaluator.verify(
        claim=claim_featured,
        node=featured_node,
        sources=_safe_list(book.festival_feature_urls),
        additional_instruction="Look for official festival schedules, programs, or reputable press indicating 'featured', 'appeared', 'panelist', or similar."
    )

    # 6.2) Festival at or near a U.S. state capital
    capital_node = evaluator.add_leaf(
        id="Festival_At_Or_Near_State_Capital",
        desc="That festival takes place at or near a U.S. state capital.",
        parent=festival_node,
        critical=True
    )
    claim_capital = (
        f"The festival '{_safe_text(book.festival_name, 'the festival')}' takes place at or near the U.S. state capital "
        f"{_safe_text(book.festival_state_capital, 'the stated capital')}."
    )
    await evaluator.verify(
        claim=claim_capital,
        node=capital_node,
        sources=_safe_list(book.festival_location_urls),
        additional_instruction="Confirm that the festival's primary location is the named capital city or within its metropolitan area. Explicit statements of location suffice."
    )


def add_required_fields_checks(evaluator: Evaluator, parent_node, book: BookCandidate) -> None:
    """
    Build 'Required_Output_Fields_Provided' subtree with presence checks (custom leaf nodes).
    """
    required_node = evaluator.add_parallel(
        id="Required_Output_Fields_Provided",
        desc="All requested answer fields are present (as specified in the question).",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(_safe_text(book.book_title)),
        id="Book_Title_Provided",
        desc="Book title is provided.",
        parent=required_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(_safe_text(book.author_name)),
        id="Author_Name_Provided",
        desc="Author name is provided.",
        parent=required_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(_safe_text(book.publication_year)),
        id="Publication_Year_Provided",
        desc="Publication year is provided.",
        parent=required_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(_safe_text(book.imprint)),
        id="Imprint_Provided",
        desc="Specific publishing imprint is provided.",
        parent=required_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(_safe_text(book.parent_company)),
        id="Parent_Company_Provided",
        desc="Parent company (Big Five publisher) of that imprint is provided.",
        parent=required_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(_safe_text(book.university)),
        id="University_Provided",
        desc="University where the author teaches is provided.",
        parent=required_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(_safe_text(book.author_us_state)),
        id="Author_US_State_Provided",
        desc="U.S. state where the author resides is provided.",
        parent=required_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(_safe_text(book.festival_name)),
        id="Festival_Name_Provided",
        desc="Name of the book festival is provided.",
        parent=required_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(_safe_text(book.festival_state_capital)),
        id="Festival_State_Capital_Provided",
        desc="Name of the state capital where the festival is held is provided.",
        parent=required_node,
        critical=True
    )


async def build_url_references_checks(evaluator: Evaluator, parent_node, book: BookCandidate) -> None:
    """
    Build 'URL_References_Provided_For_Each_Major_Claim' subtree.
    Mostly presence checks for URL categories, plus a reliability assessment leaf.
    """
    urls_node = evaluator.add_parallel(
        id="URL_References_Provided_For_Each_Major_Claim",
        desc="URLs are provided to support each major claim required by the question.",
        parent=parent_node,
        critical=True
    )

    # Reliability assessment (LLM simple verify), lists all collected URLs
    all_urls = _unique_urls(
        _safe_list(book.bibliographic_urls),
        _safe_list(book.classification_urls),
        _safe_list(book.nba_win_urls),
        _safe_list(book.nba_ceremony_urls),
        _safe_list(book.pulitzer_win_urls),
        _safe_list(book.pulitzer_announcer_urls),
        _safe_list(book.booker_shortlist_urls),
        _safe_list(book.imprint_parent_urls),
        _safe_list(book.imprint_lit_urls),
        _safe_list(book.author_teaching_urls),
        _safe_list(book.author_residence_urls),
        _safe_list(book.festival_feature_urls),
        _safe_list(book.festival_location_urls),
    )

    reliability_leaf = evaluator.add_leaf(
        id="Sources_Are_Reliable",
        desc="Provided URLs come from reliable/authoritative sources (e.g., official award sites, publisher/imprint pages, university pages, reputable news/press).",
        parent=urls_node,
        critical=True
    )
    claim_reliable = f"The following URLs are generally reliable/authoritative sources: {', '.join(all_urls)}"
    await evaluator.verify(
        claim=claim_reliable,
        node=reliability_leaf,
        additional_instruction="Judge reliability using domain reputation (e.g., official award sites like nationalbook.org, pulitzer.org, thebookerprizes.com; publishers/imprints; .edu university domains; and reputable news/press)."
    )

    # Presence checks for each required URL category
    evaluator.add_custom_node(
        result=len(_safe_list(book.bibliographic_urls)) > 0,
        id="URL_For_Bibliographic_Info",
        desc="At least one URL supports the book's title/author/publication year.",
        parent=urls_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_safe_list(book.classification_urls)) > 0,
        id="URL_For_Literary_Fiction_Classification",
        desc="At least one URL supports that the book is literary fiction (or a literary novel in fiction).",
        parent=urls_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_safe_list(book.nba_win_urls)) > 0,
        id="URL_For_NBA_Win",
        desc="At least one URL supports that the book won the 2024 National Book Award for Fiction.",
        parent=urls_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_safe_list(book.nba_ceremony_urls)) > 0,
        id="URL_For_NBA_Ceremony_Venue",
        desc="At least one URL supports that the 2024 National Book Awards ceremony took place at Cipriani Wall Street in New York City.",
        parent=urls_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_safe_list(book.pulitzer_win_urls)) > 0,
        id="URL_For_Pulitzer_Win",
        desc="At least one URL supports that the book won the 2025 Pulitzer Prize for Fiction.",
        parent=urls_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_safe_list(book.pulitzer_announcer_urls)) > 0,
        id="URL_For_Pulitzer_Announcer",
        desc="At least one URL supports that the 2025 Pulitzer Prize for Fiction was announced by Columbia University.",
        parent=urls_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_safe_list(book.booker_shortlist_urls)) > 0,
        id="URL_For_Booker_Shortlist",
        desc="At least one URL supports that the book was shortlisted for the 2024 Booker Prize.",
        parent=urls_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_safe_list(book.imprint_parent_urls)) > 0,
        id="URL_For_Imprint_And_Parent_Company",
        desc="At least one URL supports the specific imprint and its Big Five parent company relationship.",
        parent=urls_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_safe_list(book.imprint_lit_urls)) > 0,
        id="URL_For_Imprint_Literary_Fiction_Reputation",
        desc="At least one URL supports that the imprint specializes in or is known for publishing literary fiction.",
        parent=urls_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_safe_list(book.author_teaching_urls)) > 0,
        id="URL_For_Author_Teaching_Position",
        desc="At least one URL supports the author's current professor/teaching position at a university.",
        parent=urls_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_safe_list(book.author_residence_urls)) > 0,
        id="URL_For_Author_US_Residency",
        desc="At least one URL supports that the author resides in the United States.",
        parent=urls_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_safe_list(book.festival_feature_urls)) > 0,
        id="URL_For_Festival_Feature",
        desc="At least one URL supports that the book or author was featured at the named major U.S. book festival.",
        parent=urls_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_safe_list(book.festival_location_urls)) > 0,
        id="URL_For_Festival_State_Capital_Location",
        desc="At least one URL supports that the festival takes place at or near the stated U.S. state capital.",
        parent=urls_node,
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
    Evaluate an answer for the 2024 literary fiction triple-award task.
    """
    # Initialize evaluator
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

    # Extraction
    extracted_book = await evaluator.extract(
        prompt=prompt_extract_book_candidate(),
        template_class=BookCandidate,
        extraction_name="book_candidate"
    )

    # Add Big Five context as ground truth info (reference list only)
    evaluator.add_ground_truth({
        "big_five_publishers": BIG_FIVE_PUBLISHERS,
        "note": "Parent company is expected to be one of the Big Five."
    })

    # Build main critical node per rubric
    main_node = evaluator.add_parallel(
        id="Book_Identification_and_Verification",
        desc="Identify one literary fiction book published in 2024 that satisfies all award, publisher, author, and festival constraints; provide all requested fields with supporting URLs.",
        parent=root,
        critical=True
    )

    # Subtrees
    await build_content_constraints(evaluator, main_node, extracted_book)
    add_required_fields_checks(evaluator, main_node, extracted_book)
    await build_url_references_checks(evaluator, main_node, extracted_book)

    # Return summary
    return evaluator.get_summary()