import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "awards_fiction_2024"
TASK_DESCRIPTION = (
    "Identify the novels that won the 2024 Pulitzer Prize for Fiction and the 2024 National Book Award for Fiction, "
    "and provide the specified information about each, with accurate, verifiable sources."
)


# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class PulitzerInfo(BaseModel):
    # Core winner info
    award_name: Optional[str] = None
    book_title: Optional[str] = None
    author_full_name: Optional[str] = None

    # Education and book details
    undergrad_institution: Optional[str] = None
    undergrad_grad_year: Optional[str] = None
    grad_institution_program: Optional[str] = None
    grad_grad_year: Optional[str] = None
    setting_location: Optional[str] = None
    publisher: Optional[str] = None

    # Sources (URLs cited in the answer)
    sources_award_name: List[str] = Field(default_factory=list)
    sources_book_title: List[str] = Field(default_factory=list)
    sources_author_name: List[str] = Field(default_factory=list)
    sources_undergrad_institution: List[str] = Field(default_factory=list)
    sources_undergrad_grad_year: List[str] = Field(default_factory=list)
    sources_grad_institution_program: List[str] = Field(default_factory=list)
    sources_grad_grad_year: List[str] = Field(default_factory=list)
    sources_setting_location: List[str] = Field(default_factory=list)
    sources_publisher: List[str] = Field(default_factory=list)
    sources_general: List[str] = Field(default_factory=list)


class NBAInfo(BaseModel):
    # Core winner info
    award_name: Optional[str] = None
    book_title: Optional[str] = None
    author_full_name: Optional[str] = None

    # Education details
    undergrad_institution: Optional[str] = None
    undergrad_degree_field: Optional[str] = None
    grad_institution: Optional[str] = None

    # Sources (URLs cited in the answer)
    sources_award_name: List[str] = Field(default_factory=list)
    sources_book_title: List[str] = Field(default_factory=list)
    sources_author_name: List[str] = Field(default_factory=list)
    sources_undergrad_institution: List[str] = Field(default_factory=list)
    sources_undergrad_degree_field: List[str] = Field(default_factory=list)
    sources_grad_institution: List[str] = Field(default_factory=list)
    sources_general: List[str] = Field(default_factory=list)


class AwardsExtraction(BaseModel):
    pulitzer: Optional[PulitzerInfo] = None
    nba: Optional[NBAInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_award_info() -> str:
    return """
Extract structured information for two 2024 fiction award winners as explicitly presented in the answer text, along with the exact source URLs cited in the answer.

You must extract two objects: "pulitzer" and "nba".

pulitzer:
- award_name: The formal award name as provided in the answer (e.g., "Pulitzer Prize for Fiction").
- book_title: The title of the 2024 Pulitzer Prize for Fiction winner.
- author_full_name: The author's full name of the winning book.
- undergrad_institution: The author's undergraduate institution (university/college).
- undergrad_grad_year: The author's undergraduate graduation year.
- grad_institution_program: The author's graduate institution and program, as one combined string (e.g., "University X, MFA in Creative Writing").
- grad_grad_year: The author's graduate graduation year.
- setting_location: The primary geographic location where the novel is set.
- publisher: The publisher or imprint for the winning book.

- sources_award_name: All URLs cited in the answer that support the identification of the award name.
- sources_book_title: All URLs cited that support the winner book title for the award.
- sources_author_name: All URLs cited that support the winner author name for the award.
- sources_undergrad_institution: All URLs cited that support the author's undergrad institution.
- sources_undergrad_grad_year: All URLs cited that support the author's undergrad graduation year.
- sources_grad_institution_program: All URLs cited that support the author's graduate institution/program information.
- sources_grad_grad_year: All URLs cited that support the author's graduate graduation year.
- sources_setting_location: All URLs cited that support the book's primary setting/location.
- sources_publisher: All URLs cited that support the publisher/imprint.
- sources_general: Any other URLs cited in the answer related to the Pulitzer winner (e.g., prize announcement pages, publisher pages) that can support general winner facts.

nba:
- award_name: The formal award name as provided in the answer (e.g., "National Book Award for Fiction").
- book_title: The title of the 2024 National Book Award for Fiction winner.
- author_full_name: The author's full name of the winning book.
- undergrad_institution: The author's undergraduate institution.
- undergrad_degree_field: The field/major of the author's undergraduate degree.
- grad_institution: The author's graduate institution (if present in the answer).

- sources_award_name: All URLs cited that support the award name.
- sources_book_title: All URLs cited that support the winner book title.
- sources_author_name: All URLs cited that support the winner author name.
- sources_undergrad_institution: All URLs cited that support the author's undergrad institution.
- sources_undergrad_degree_field: All URLs cited that support the author's undergrad degree field/major.
- sources_grad_institution: All URLs cited that support the author's graduate institution.
- sources_general: Any other URLs cited for the NBA winner.

IMPORTANT:
- Only extract information explicitly present in the answer text.
- For URLs, extract the actual URL strings present in the answer (including those in markdown links).
- If a specific piece of information is not present, set it to null (or an empty array for sources).
- Do not invent or infer any information or URLs.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def choose_sources(*lists: List[str]) -> List[str]:
    """Return the first non-empty list among provided lists; if all empty, return empty list."""
    for lst in lists:
        if lst and len(lst) > 0:
            # Deduplicate while preserving order
            seen = set()
            dedup = []
            for u in lst:
                if u and u not in seen:
                    dedup.append(u)
                    seen.add(u)
            return dedup
    return []


def _non_empty_str(s: Optional[str]) -> str:
    return s if (s is not None and str(s).strip() != "") else ""


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_pulitzer_tree(evaluator: Evaluator, parent_node, info: Optional[PulitzerInfo]) -> None:
    node = evaluator.add_parallel(
        id="Pulitzer_Fiction_2024_Book",
        desc="Information about the book that won the 2024 Pulitzer Prize for Fiction, including the award name, book title, author details, author's educational background, book setting, and publisher.",
        parent=parent_node,
        critical=False
    )

    # Existence & basic sources check (gating)
    has_min_fields = bool(info and _non_empty_str(info.award_name) and _non_empty_str(info.book_title) and _non_empty_str(info.author_full_name))
    has_any_sources = bool(info and any([
        info.sources_award_name, info.sources_book_title, info.sources_author_name, info.sources_general
    ]))
    evaluator.add_custom_node(
        result=(has_min_fields and has_any_sources),
        id="Pulitzer_Fiction_2024_Required_Fields_Provided",
        desc="Pulitzer 2024: Required core fields (award name, book title, author) and at least one supporting source are provided",
        parent=node,
        critical=True
    )

    # Prepare safe strings
    award_name = _non_empty_str(info.award_name if info else None)
    title = _non_empty_str(info.book_title if info else None)
    author = _non_empty_str(info.author_full_name if info else None)

    # Award Name (critical)
    n_award = evaluator.add_leaf(
        id="Pulitzer_Fiction_2024_Award_Name",
        desc="The correct name of the award won is identified as 'Pulitzer Prize for Fiction' or equivalent formal designation for the 2024 fiction award.",
        parent=node,
        critical=True
    )
    claim_award = "The award is the 'Pulitzer Prize for Fiction' for the year 2024."
    src_award = choose_sources(
        info.sources_award_name if info else [],
        info.sources_general if info else [],
        (info.sources_book_title if info else []),
        (info.sources_author_name if info else [])
    )
    await evaluator.verify(
        claim=claim_award,
        node=n_award,
        sources=src_award,
        additional_instruction="Verify that the page explicitly indicates the 2024 Pulitzer Prize for Fiction (allow minor wording variations like 'Pulitzer Prize in Fiction')."
    )

    # Book Title (critical)
    n_title = evaluator.add_leaf(
        id="Pulitzer_Fiction_2024_Book_Title",
        desc="The title of the book that won the 2024 Pulitzer Prize for Fiction is correctly identified.",
        parent=node,
        critical=True
    )
    claim_title = f"The winner of the 2024 Pulitzer Prize for Fiction is the book titled '{title}'."
    src_title = choose_sources(
        info.sources_book_title if info else [],
        info.sources_general if info else [],
        (info.sources_award_name if info else []),
        (info.sources_author_name if info else [])
    )
    await evaluator.verify(
        claim=claim_title,
        node=n_title,
        sources=src_title,
        additional_instruction="Verify that the page names the 2024 Pulitzer Prize for Fiction winner with the provided book title (allow minor punctuation or capitalization differences)."
    )

    # Author Name (critical)
    n_author = evaluator.add_leaf(
        id="Pulitzer_Fiction_2024_Author_Name",
        desc="The full name of the author of the 2024 Pulitzer Prize Fiction winner is correctly provided.",
        parent=node,
        critical=True
    )
    claim_author = f"The author of the 2024 Pulitzer Prize for Fiction-winning book '{title}' is '{author}'."
    src_author = choose_sources(
        info.sources_author_name if info else [],
        info.sources_general if info else [],
        (info.sources_book_title if info else []),
        (info.sources_award_name if info else [])
    )
    await evaluator.verify(
        claim=claim_author,
        node=n_author,
        sources=src_author,
        additional_instruction="Verify that the page shows the winner’s author for the 2024 Pulitzer Prize for Fiction; allow minor variations (middle initials, hyphenation)."
    )

    # Undergrad Institution (critical)
    n_ug_inst = evaluator.add_leaf(
        id="Pulitzer_Fiction_2024_Author_Undergrad_Institution",
        desc="The name of the university where the author earned their undergraduate degree is correctly identified.",
        parent=node,
        critical=True
    )
    ug_inst = _non_empty_str(info.undergrad_institution if info else None)
    claim_ug_inst = f"The author {author} earned an undergraduate degree from {ug_inst}."
    src_ug_inst = choose_sources(
        info.sources_undergrad_institution if info else [],
        (info.sources_author_name if info else []),
        (info.sources_general if info else [])
    )
    await evaluator.verify(
        claim=claim_ug_inst,
        node=n_ug_inst,
        sources=src_ug_inst,
        additional_instruction="Verify the author's undergraduate institution from biographies, university pages, or credible profiles."
    )

    # Undergrad Graduation Year (critical)
    n_ug_year = evaluator.add_leaf(
        id="Pulitzer_Fiction_2024_Author_Undergrad_Graduation_Year",
        desc="The year the author graduated with their undergraduate degree is correctly provided.",
        parent=node,
        critical=True
    )
    ug_year = _non_empty_str(info.undergrad_grad_year if info else None)
    claim_ug_year = f"The author {author} completed their undergraduate degree in {ug_year}."
    src_ug_year = choose_sources(
        info.sources_undergrad_grad_year if info else [],
        (info.sources_undergrad_institution if info else []),
        (info.sources_author_name if info else []),
        (info.sources_general if info else [])
    )
    await evaluator.verify(
        claim=claim_ug_year,
        node=n_ug_year,
        sources=src_ug_year,
        additional_instruction="Verify the undergraduate graduation year explicitly; minor phrasing differences are acceptable as long as the year matches."
    )

    # Graduate Institution and Program (critical)
    n_grad_inst_prog = evaluator.add_leaf(
        id="Pulitzer_Fiction_2024_Author_Graduate_Institution",
        desc="The name of the institution where the author earned their graduate degree is correctly identified, including the specific program if applicable.",
        parent=node,
        critical=True
    )
    grad_inst_prog = _non_empty_str(info.grad_institution_program if info else None)
    claim_grad_inst_prog = f"The author {author} earned a graduate degree as described: {grad_inst_prog}."
    src_grad_inst_prog = choose_sources(
        info.sources_grad_institution_program if info else [],
        (info.sources_author_name if info else []),
        (info.sources_general if info else [])
    )
    await evaluator.verify(
        claim=claim_grad_inst_prog,
        node=n_grad_inst_prog,
        sources=src_grad_inst_prog,
        additional_instruction="Verify both the graduate institution and the program/degree (e.g., MFA) as provided in the answer."
    )

    # Graduate Graduation Year (critical)
    n_grad_year = evaluator.add_leaf(
        id="Pulitzer_Fiction_2024_Author_Graduate_Graduation_Year",
        desc="The year the author earned their graduate degree is correctly provided.",
        parent=node,
        critical=True
    )
    grad_year = _non_empty_str(info.grad_grad_year if info else None)
    claim_grad_year = f"The author {author} earned their graduate degree in {grad_year}."
    src_grad_year = choose_sources(
        info.sources_grad_grad_year if info else [],
        (info.sources_grad_institution_program if info else []),
        (info.sources_author_name if info else []),
        (info.sources_general if info else [])
    )
    await evaluator.verify(
        claim=claim_grad_year,
        node=n_grad_year,
        sources=src_grad_year,
        additional_instruction="Verify the graduate degree completion year explicitly; allow minor contextual phrasing."
    )

    # Book Setting Location (critical)
    n_setting = evaluator.add_leaf(
        id="Pulitzer_Fiction_2024_Book_Setting_Location",
        desc="The primary geographic location where the book is set is correctly identified.",
        parent=node,
        critical=True
    )
    setting_loc = _non_empty_str(info.setting_location if info else None)
    claim_setting = f"The primary setting of the novel '{title}' is {setting_loc}."
    src_setting = choose_sources(
        info.sources_setting_location if info else [],
        (info.sources_general if info else []),
        (info.sources_book_title if info else [])
    )
    await evaluator.verify(
        claim=claim_setting,
        node=n_setting,
        sources=src_setting,
        additional_instruction="Verify the primary setting as described in publisher pages, reviews, or credible summaries; minor variations in phrasing are acceptable."
    )

    # Publisher (non-critical)
    n_publisher = evaluator.add_leaf(
        id="Pulitzer_Fiction_2024_Publisher",
        desc="The name of the publisher or publishing imprint is provided.",
        parent=node,
        critical=False
    )
    publisher = _non_empty_str(info.publisher if info else None)
    claim_publisher = f"The publisher or imprint of '{title}' is '{publisher}'."
    src_publisher = choose_sources(
        info.sources_publisher if info else [],
        (info.sources_general if info else []),
        (info.sources_book_title if info else [])
    )
    await evaluator.verify(
        claim=claim_publisher,
        node=n_publisher,
        sources=src_publisher,
        additional_instruction="Verify the publisher or imprint from credible sources such as the publisher site, book retailers, or library catalogs."
    )


async def build_nba_tree(evaluator: Evaluator, parent_node, info: Optional[NBAInfo]) -> None:
    node = evaluator.add_parallel(
        id="NBA_Fiction_2024_Book",
        desc="Information about the book that won the 2024 National Book Award for Fiction, including the award name, book title, author details, and author's educational background.",
        parent=parent_node,
        critical=False
    )

    # Existence & basic sources check (gating)
    has_min_fields = bool(info and _non_empty_str(info.award_name) and _non_empty_str(info.book_title) and _non_empty_str(info.author_full_name))
    has_any_sources = bool(info and any([
        info.sources_award_name, info.sources_book_title, info.sources_author_name, info.sources_general
    ]))
    evaluator.add_custom_node(
        result=(has_min_fields and has_any_sources),
        id="NBA_Fiction_2024_Required_Fields_Provided",
        desc="NBA 2024: Required core fields (award name, book title, author) and at least one supporting source are provided",
        parent=node,
        critical=True
    )

    # Prepare safe strings
    award_name = _non_empty_str(info.award_name if info else None)
    title = _non_empty_str(info.book_title if info else None)
    author = _non_empty_str(info.author_full_name if info else None)

    # Award Name (critical)
    n_award = evaluator.add_leaf(
        id="NBA_Fiction_2024_Award_Name",
        desc="The correct name of the award won is identified as 'National Book Award for Fiction' or equivalent formal designation for the 2024 fiction award.",
        parent=node,
        critical=True
    )
    claim_award = "The award is the 'National Book Award for Fiction' for the year 2024."
    src_award = choose_sources(
        info.sources_award_name if info else [],
        (info.sources_general if info else []),
        (info.sources_book_title if info else []),
        (info.sources_author_name if info else [])
    )
    await evaluator.verify(
        claim=claim_award,
        node=n_award,
        sources=src_award,
        additional_instruction="Verify that the page explicitly indicates the 2024 National Book Award for Fiction (allow minor wording variations)."
    )

    # Book Title (critical)
    n_title = evaluator.add_leaf(
        id="NBA_Fiction_2024_Book_Title",
        desc="The title of the book that won the 2024 National Book Award for Fiction is correctly identified.",
        parent=node,
        critical=True
    )
    claim_title = f"The winner of the 2024 National Book Award for Fiction is the book titled '{title}'."
    src_title = choose_sources(
        info.sources_book_title if info else [],
        (info.sources_general if info else []),
        (info.sources_award_name if info else []),
        (info.sources_author_name if info else [])
    )
    await evaluator.verify(
        claim=claim_title,
        node=n_title,
        sources=src_title,
        additional_instruction="Verify that the page names the 2024 National Book Award for Fiction winner with the provided book title."
    )

    # Author Name (critical)
    n_author = evaluator.add_leaf(
        id="NBA_Fiction_2024_Author_Name",
        desc="The full name of the author of the 2024 National Book Award Fiction winner is correctly provided.",
        parent=node,
        critical=True
    )
    claim_author = f"The author of the 2024 National Book Award for Fiction-winning book '{title}' is '{author}'."
    src_author = choose_sources(
        info.sources_author_name if info else [],
        (info.sources_general if info else []),
        (info.sources_book_title if info else []),
        (info.sources_award_name if info else [])
    )
    await evaluator.verify(
        claim=claim_author,
        node=n_author,
        sources=src_author,
        additional_instruction="Verify that the page shows the winner’s author for the 2024 National Book Award for Fiction; allow minor variations (middle initials, hyphenation)."
    )

    # Undergrad Institution (critical)
    n_ug_inst = evaluator.add_leaf(
        id="NBA_Fiction_2024_Author_Undergrad_Institution",
        desc="The name of the university where the author earned their undergraduate degree is correctly identified.",
        parent=node,
        critical=True
    )
    ug_inst = _non_empty_str(info.undergrad_institution if info else None)
    claim_ug_inst = f"The author {author} earned an undergraduate degree from {ug_inst}."
    src_ug_inst = choose_sources(
        info.sources_undergrad_institution if info else [],
        (info.sources_author_name if info else []),
        (info.sources_general if info else [])
    )
    await evaluator.verify(
        claim=claim_ug_inst,
        node=n_ug_inst,
        sources=src_ug_inst,
        additional_instruction="Verify the author's undergraduate institution from credible bios or university pages."
    )

    # Undergrad Degree Field (critical)
    n_ug_field = evaluator.add_leaf(
        id="NBA_Fiction_2024_Author_Undergrad_Degree_Field",
        desc="The field or major of the author's undergraduate degree is correctly provided.",
        parent=node,
        critical=True
    )
    ug_field = _non_empty_str(info.undergrad_degree_field if info else None)
    claim_ug_field = f"The author's undergraduate degree field/major is '{ug_field}'."
    src_ug_field = choose_sources(
        info.sources_undergrad_degree_field if info else [],
        (info.sources_undergrad_institution if info else []),
        (info.sources_author_name if info else []),
        (info.sources_general if info else [])
    )
    await evaluator.verify(
        claim=claim_ug_field,
        node=n_ug_field,
        sources=src_ug_field,
        additional_instruction="Verify the specific undergraduate field/major stated for the author."
    )

    # Graduate Institution (critical)
    n_grad_inst = evaluator.add_leaf(
        id="NBA_Fiction_2024_Author_Graduate_Institution",
        desc="The name of the institution where the author earned their graduate degree is correctly identified.",
        parent=node,
        critical=True
    )
    grad_inst = _non_empty_str(info.grad_institution if info else None)
    claim_grad_inst = f"The author {author} earned a graduate degree from {grad_inst}."
    src_grad_inst = choose_sources(
        info.sources_grad_institution if info else [],
        (info.sources_author_name if info else []),
        (info.sources_general if info else [])
    )
    await evaluator.verify(
        claim=claim_grad_inst,
        node=n_grad_inst,
        sources=src_grad_inst,
        additional_instruction="Verify the author's graduate institution as stated in reputable sources."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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

    extraction = await evaluator.extract(
        prompt=prompt_extract_award_info(),
        template_class=AwardsExtraction,
        extraction_name="awards_extraction"
    )

    # Root node: Award_Winning_Fiction_Books_2024
    main_node = evaluator.add_parallel(
        id="Award_Winning_Fiction_Books_2024",
        desc="Provide complete information about the fiction books that won the 2024 Pulitzer Prize for Fiction and the 2024 National Book Award for Fiction, including details about each book, its author's educational background, and publication information.",
        parent=root,
        critical=False
    )

    # Build subtrees
    await build_pulitzer_tree(evaluator, main_node, extraction.pulitzer)
    await build_nba_tree(evaluator, main_node, extraction.nba)

    return evaluator.get_summary()