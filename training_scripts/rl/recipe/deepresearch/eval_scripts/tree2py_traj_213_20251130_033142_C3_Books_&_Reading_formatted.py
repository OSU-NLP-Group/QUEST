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
TASK_ID = "sff_2024_actor_coauthor"
TASK_DESCRIPTION = (
    "Identify a science fiction or fantasy book that was published in hardcover in 2024 and co-authored by a well-known "
    "Hollywood actor together with an established professional science fiction/fantasy author. The publisher must be a recognized "
    "science fiction and fantasy imprint. Provide the following information: "
    "(1) The title of the 2024 book and both co-authors' names; "
    "(2) The name of the publishing imprint and a URL reference to the publisher's official page for the book; "
    "(3) The page count and the exact publication date (month, day, and year) of the hardcover edition as listed by the publisher; "
    "(4) The name of one standalone science fiction or fantasy novel written by the professional co-author (not the actor) that was published before 2024, along with its publication year and a URL reference; "
    "(5) The name of a major speculative fiction literary award (Hugo Award, Nebula Award, Arthur C. Clarke Award, Locus Award, or British Science Fiction Association Award) for which that previous novel either won or received a nomination; "
    "(6) The year of that award win or nomination, the specific award category, whether it won or was only nominated, and a URL reference documenting the award recognition."
)

ALLOWED_AWARDS = {
    "hugo award",
    "nebula award",
    "arthur c. clarke award",
    "locus award",
    "british science fiction association award",
    "bsfa award"  # BSFA shorthand
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MainBookInfo(BaseModel):
    book_title: Optional[str] = None
    actor_name: Optional[str] = None
    professional_author_name: Optional[str] = None
    publisher_official_url: Optional[str] = None
    imprint_name: Optional[str] = None
    imprint_verification_url: Optional[str] = None
    page_count: Optional[str] = None
    hardcover_pub_date: Optional[str] = None


class PreviousNovelInfo(BaseModel):
    novel_title: Optional[str] = None
    novel_publication_year: Optional[str] = None
    novel_url: Optional[str] = None


class AwardInfo(BaseModel):
    award_name: Optional[str] = None
    award_year: Optional[str] = None
    award_category: Optional[str] = None
    award_outcome: Optional[str] = None  # e.g., "Won" or "Nominated"
    award_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_main_book_info() -> str:
    return """
    From the answer, extract ALL of the following fields for the 2024 hardcover SFF book:

    - book_title: The exact title of the book (string).
    - actor_name: The Hollywood actor co-author’s name (string).
    - professional_author_name: The professional science fiction/fantasy author's name (string).
    - publisher_official_url: A URL to the publisher’s official page for this book. Must be a publisher domain, not a retailer or third-party site. If absent, return null.
    - imprint_name: The publishing imprint name (string). If absent, return null.
    - imprint_verification_url: A URL (preferably on the publisher’s domain) that verifies the imprint is a science fiction/fantasy imprint. If the book page itself clearly indicates the imprint’s SFF focus, you may repeat the book’s publisher URL here; otherwise provide a dedicated imprint page URL. If absent, return null.
    - page_count: The page count as listed by the publisher (string; extract exactly as stated, e.g., "432" or "432 pages").
    - hardcover_pub_date: The exact publication date of the hardcover edition (month, day, and year) as listed by the publisher (string; extract exactly as stated).

    IMPORTANT:
    - Extract only what is explicitly present in the answer text. Do NOT invent or guess.
    - For any missing field, return null.
    - For URLs, extract the actual URL string mentioned; if a markdown link is given, return the URL part.
    """


def prompt_extract_previous_novel(pro_author_name: Optional[str]) -> str:
    pro = pro_author_name or "the professional co-author"
    return f"""
    From the answer, extract ONE standalone science fiction or fantasy novel by {pro} (not the actor) that was published before 2024, with:
    - novel_title: The novel’s title (string).
    - novel_publication_year: The publication year (string exactly as written; do NOT convert).
    - novel_url: A URL reference supporting the identification and/or publication year (string; if absent, return null).

    NOTES:
    - Extract only the information explicitly present in the answer.
    - Ensure the selected work is a standalone novel (not a short story or novella) according to the answer’s content.
    - If multiple are listed, pick the first one mentioned.
    """


def prompt_extract_award(novel_title: Optional[str]) -> str:
    nt = novel_title or "the previous novel"
    return f"""
    From the answer, extract award recognition details for {nt}:
    - award_name: The name of the award (string). Must be one of: Hugo Award, Nebula Award, Arthur C. Clarke Award, Locus Award, or British Science Fiction Association (BSFA) Award. If a variant spelling is shown, extract exactly as written.
    - award_year: The year of the win/nomination (string).
    - award_category: The specific award category (string).
    - award_outcome: Whether the novel won or was only nominated (string; e.g., "Won", "Nominated", or the exact wording used).
    - award_url: A URL documenting this recognition (string; if absent, return null).

    IMPORTANT:
    - Extract only information explicitly present in the answer.
    - If any field is missing, return null for that field.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_nonempty(value: Optional[str]) -> bool:
    return bool(value and str(value).strip())


def _is_allowed_award(name: Optional[str]) -> bool:
    if not _is_nonempty(name):
        return False
    lowered = name.strip().lower()
    # normalize some common variants
    if "bsfa" in lowered and "award" in lowered:
        return True
    return lowered in ALLOWED_AWARDS


# --------------------------------------------------------------------------- #
# Verification section builders                                               #
# --------------------------------------------------------------------------- #
async def verify_section_A(evaluator: Evaluator, parent_node, main: MainBookInfo) -> None:
    """
    Section A: Identify the 2024 book and both co-authors as requested.
    """
    a_node = evaluator.add_parallel(
        id="A_2024_Book_Identification",
        desc="Identify the 2024 book and both co-authors as requested.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_nonempty(main.book_title),
        id="A1_Book_Title_Provided",
        desc="Provide the title of the 2024 book.",
        parent=a_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_nonempty(main.actor_name),
        id="A2_Actor_Coauthor_Name_Provided",
        desc="Provide the actor co-author’s name.",
        parent=a_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_nonempty(main.professional_author_name),
        id="A3_Professional_Coauthor_Name_Provided",
        desc="Provide the professional (non-actor) co-author’s name.",
        parent=a_node,
        critical=True
    )


async def verify_section_B(evaluator: Evaluator, parent_node, main: MainBookInfo) -> None:
    """
    Section B: Publisher/imprint info and publisher-official verification for hardcover details.
    """
    b_node = evaluator.add_parallel(
        id="B_Publisher_Official_Book_And_Imprint_Details",
        desc="Provide publisher/imprint info and publisher-official verification for the hardcover edition details.",
        parent=parent_node,
        critical=True
    )

    # B1: Official publisher page URL provided
    evaluator.add_custom_node(
        result=_is_nonempty(main.publisher_official_url),
        id="B1_Official_Publisher_Page_URL_Provided",
        desc="Provide a URL to the publisher’s official page for the book.",
        parent=b_node,
        critical=True
    )

    # B2: Publisher page lists both co-authors
    b2_node = evaluator.add_leaf(
        id="B2_Publisher_Page_Lists_Both_Coauthors",
        desc="The publisher’s official book page lists both named co-authors for the book.",
        parent=b_node,
        critical=True
    )
    claim_b2 = (
        f"The publisher’s official book page lists both co-authors: '{main.actor_name}' and '{main.professional_author_name}'. "
        f"Minor variations in name formatting (middle initials, casing) should still be treated as matches."
    )
    await evaluator.verify(
        claim=claim_b2,
        node=b2_node,
        sources=main.publisher_official_url,
        additional_instruction="Check the author/co-author credits on the publisher page. Allow minor formatting variations."
    )

    # B3: Book is science fiction or fantasy per publisher
    b3_node = evaluator.add_leaf(
        id="B3_Book_Is_ScienceFiction_Or_Fantasy_Per_Publisher",
        desc="The publisher’s official book page indicates the book is science fiction or fantasy.",
        parent=b_node,
        critical=True
    )
    claim_b3 = (
        "This book is categorized or clearly described as science fiction or fantasy on the publisher’s official page "
        "(e.g., genre tag, imprint context, marketing copy, or series classification indicating SFF)."
    )
    await evaluator.verify(
        claim=claim_b3,
        node=b3_node,
        sources=main.publisher_official_url,
        additional_instruction="Accept genre labels such as 'Science Fiction', 'Fantasy', 'Speculative Fiction', or clear SFF marketing language. "
                               "Listing under a dedicated SFF imprint page also qualifies."
    )

    # B4: Imprint name provided
    evaluator.add_custom_node(
        result=_is_nonempty(main.imprint_name),
        id="B4_Imprint_Name_Provided",
        desc="Provide the name of the publishing imprint for the book.",
        parent=b_node,
        critical=True
    )

    # B5: Imprint is SFF imprint verifiable
    b5_node = evaluator.add_leaf(
        id="B5_Imprint_Is_SFF_Imprint_Verifiable",
        desc="The imprint can be verified as a science fiction and/or fantasy imprint using official publisher information.",
        parent=b_node,
        critical=True
    )
    imprint_sources: List[str] = []
    if _is_nonempty(main.imprint_verification_url):
        imprint_sources.append(main.imprint_verification_url)  # preferred if provided
    if _is_nonempty(main.publisher_official_url):
        imprint_sources.append(main.publisher_official_url)    # fallback/context

    claim_b5 = f"The publishing imprint '{main.imprint_name}' is a science fiction and/or fantasy imprint."
    await evaluator.verify(
        claim=claim_b5,
        node=b5_node,
        sources=imprint_sources if imprint_sources else None,
        additional_instruction="Use official publisher pages. It suffices if the book page explicitly places the imprint within SFF, or if the imprint’s own page states it focuses on SFF."
    )

    # B6: Page count from publisher official info
    b6_node = evaluator.add_leaf(
        id="B6_Page_Count_From_Publisher_Official_Info",
        desc="Provide the page count as listed by the publisher’s official information.",
        parent=b_node,
        critical=True
    )
    claim_b6 = f"The book has a page count of '{main.page_count}' as listed by the publisher."
    await evaluator.verify(
        claim=claim_b6,
        node=b6_node,
        sources=main.publisher_official_url,
        additional_instruction="Locate 'Pages' or equivalent metadata on the official publisher page. "
                               "Allow minor variations like '432 pages' vs '432'."
    )

    # B7: Exact hardcover publication date from publisher official info and is 2024
    b7_node = evaluator.add_leaf(
        id="B7_Exact_Hardcover_Publication_Date_From_Publisher_Official_Info_And_Is_2024",
        desc="Provide the exact hardcover publication date (MM/DD/YYYY or Month Day, Year) from publisher; year must be 2024.",
        parent=b_node,
        critical=True
    )
    claim_b7 = (
        f"The hardcover publication date is '{main.hardcover_pub_date}', and the year is 2024, as listed on the publisher’s official page."
    )
    await evaluator.verify(
        claim=claim_b7,
        node=b7_node,
        sources=main.publisher_official_url,
        additional_instruction="Confirm the date pertains specifically to the hardcover edition (not paperback or ebook). "
                               "Allow minor formatting differences; ensure the year equals 2024."
    )


async def verify_section_C(evaluator: Evaluator, parent_node, prev: PreviousNovelInfo) -> None:
    """
    Section C: Provide one standalone pre-2024 SFF novel by the professional co-author, with year and a URL.
    """
    c_node = evaluator.add_parallel(
        id="C_Professional_Coauthor_Pre2024_Standalone_Novel",
        desc="Provide one standalone pre-2024 SFF novel by the professional co-author, with year and a URL reference.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_nonempty(prev.novel_title),
        id="C1_Pre2024_Standalone_SFF_Novel_Title_Provided",
        desc="Name one standalone science fiction or fantasy novel written by the professional co-author (not the actor).",
        parent=c_node,
        critical=True
    )

    # C2: Year provided and is before 2024 — verify against the provided novel URL
    c2_node = evaluator.add_leaf(
        id="C2_Previous_Novel_Publication_Year_Provided_And_Is_Before_2024",
        desc="Provide the publication year for that novel, and it is before 2024.",
        parent=c_node,
        critical=True
    )
    claim_c2 = (
        f"The previous novel '{prev.novel_title}' was published in {prev.novel_publication_year}, "
        f"which is before 2024."
    )
    await evaluator.verify(
        claim=claim_c2,
        node=c2_node,
        sources=prev.novel_url,
        additional_instruction="Confirm the publication year on the referenced page; basic arithmetic suffices to confirm it is earlier than 2024. "
                               "Allow reasonable page structures (e.g., title pages, metadata sidebars)."
    )

    evaluator.add_custom_node(
        result=_is_nonempty(prev.novel_url),
        id="C3_Previous_Novel_URL_Reference_Provided",
        desc="Provide a URL reference supporting the previous novel identification and/or its publication year.",
        parent=c_node,
        critical=True
    )


async def verify_section_D(evaluator: Evaluator, parent_node, award: AwardInfo, prev: PreviousNovelInfo) -> None:
    """
    Section D: Provide one qualifying major-award win/nomination for that previous novel with all requested details and a URL.
    """
    d_node = evaluator.add_parallel(
        id="D_Award_Recognition_For_Previous_Novel",
        desc="Provide one qualifying major-award win/nomination for that previous novel with all requested details and a URL reference.",
        parent=parent_node,
        critical=True
    )

    # D1: Award name must be from allowed list
    evaluator.add_custom_node(
        result=_is_allowed_award(award.award_name),
        id="D1_Major_Award_Name_From_Allowed_List",
        desc="Provide an award name and it must be one of the allowed major awards.",
        parent=d_node,
        critical=True
    )

    # D2: Award recognition pertains to the identified previous novel
    d2_node = evaluator.add_leaf(
        id="D2_Award_Recognition_Is_For_That_Novel",
        desc="The award win/nomination pertains to the identified previous novel (not a different work).",
        parent=d_node,
        critical=True
    )
    claim_d2 = (
        f"The documented award recognition (win or nomination) is for the novel '{prev.novel_title}'. "
        f"Minor title formatting variations are acceptable."
    )
    await evaluator.verify(
        claim=claim_d2,
        node=d2_node,
        sources=award.award_url,
        additional_instruction="Confirm the award page explicitly references the novel (title) as the recognized work. "
                               "If multiple works are listed, ensure the specified novel is the one recognized."
    )

    # D3: Award year provided (existence)
    evaluator.add_custom_node(
        result=_is_nonempty(award.award_year),
        id="D3_Award_Year_Provided",
        desc="Provide the year of the award win or nomination.",
        parent=d_node,
        critical=True
    )

    # D4: Award category provided (existence)
    evaluator.add_custom_node(
        result=_is_nonempty(award.award_category),
        id="D4_Award_Category_Provided",
        desc="Provide the specific award category.",
        parent=d_node,
        critical=True
    )

    # D5: Award outcome provided (existence)
    evaluator.add_custom_node(
        result=_is_nonempty(award.award_outcome),
        id="D5_Award_Outcome_Provided",
        desc="State whether it won or was only nominated.",
        parent=d_node,
        critical=True
    )

    # D6: Award documentation URL provided (existence)
    evaluator.add_custom_node(
        result=_is_nonempty(award.award_url),
        id="D6_Award_Documentation_URL_Provided",
        desc="Provide a URL reference documenting the award win/nomination for the novel.",
        parent=d_node,
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
    Evaluate an answer for the SFF 2024 actor+author co-authored hardcover book task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Sequential: A -> B -> C -> D
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

    # Extract structured info
    main_info = await evaluator.extract(
        prompt=prompt_extract_main_book_info(),
        template_class=MainBookInfo,
        extraction_name="main_book_info",
    )

    prev_info = await evaluator.extract(
        prompt=prompt_extract_previous_novel(main_info.professional_author_name),
        template_class=PreviousNovelInfo,
        extraction_name="previous_novel_info",
    )

    award_info = await evaluator.extract(
        prompt=prompt_extract_award(prev_info.novel_title),
        template_class=AwardInfo,
        extraction_name="award_info",
    )

    # Add custom info to summary (e.g., allowed awards for transparency)
    evaluator.add_custom_info(
        info={"allowed_awards": sorted(list(ALLOWED_AWARDS))},
        info_type="policy",
        info_name="allowed_awards_list"
    )

    # Build and verify the tree
    # Root is critical sequential; all children must be critical
    await verify_section_A(evaluator, root, main_info)
    await verify_section_B(evaluator, root, main_info)
    await verify_section_C(evaluator, root, prev_info)
    await verify_section_D(evaluator, root, award_info, prev_info)

    # Return summary
    return evaluator.get_summary()