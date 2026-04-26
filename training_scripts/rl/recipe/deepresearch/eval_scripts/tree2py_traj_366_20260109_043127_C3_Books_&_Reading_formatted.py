import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants and ground truths                                   #
# --------------------------------------------------------------------------- #
TASK_ID = "booker_2024_research"
TASK_DESCRIPTION = (
    "I am researching the 2024 Booker Prize-winning novel for a literary database entry. "
    "Please identify the winning novel and its author, then provide: "
    "(1) the UK hardcover first edition's publisher name, publication date, and ISBN-13; "
    "(2) details about the Japanese astronaut character in the novel, specifically what significant personal event happened to this character while in space and what metaphorical role this character is described as fulfilling among the crew; "
    "(3) the author's birthplace (county and country), birth year, and the two universities where the author studied philosophy; "
    "and (4) the title of the author's debut novel published in 2009 and the award it won that same year."
)

# Ground-truth assertions for verification
GT_WINNER_TITLE = "Orbital"
GT_WINNER_AUTHOR = "Samantha Harvey"

GT_UK_PUBLISHER = "Jonathan Cape"
GT_UK_PUBLICATION_DATE = "2 November 2023"  # Accept also "November 2, 2023"
GT_UK_PUBLICATION_DATE_US = "November 2, 2023"
GT_UK_ISBN13 = "978-1787334342"

GT_SIX_ASTRONAUTS = "Six astronauts from different countries"
GT_JP_EVENT = "Japanese astronaut's mother died while the astronaut was in space"
GT_JP_METAPHOR = "Japanese astronaut is the craft's 'conscience'"

GT_AUTHOR_BIRTHPLACE_COUNTY = "Kent"
GT_AUTHOR_BIRTHPLACE_COUNTRY = "England"
GT_AUTHOR_BIRTH_YEAR = "1975"
GT_AUTHOR_UNIS = ["University of York", "University of Sheffield"]

GT_DEBUT_TITLE = "The Wilderness"
GT_DEBUT_YEAR = "2009"
GT_DEBUT_AWARD = "Betty Trask Prize"
GT_DEBUT_AWARD_YEAR = "2009"


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class WinnerSection(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class UKHardcoverSection(BaseModel):
    publisher: Optional[str] = None
    publication_date: Optional[str] = None
    isbn13: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CharacterSection(BaseModel):
    six_astronauts_statement: Optional[str] = None
    japanese_personal_event: Optional[str] = None
    japanese_metaphorical_role: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AuthorBackgroundSection(BaseModel):
    birthplace_county: Optional[str] = None
    birthplace_country: Optional[str] = None
    birth_year: Optional[str] = None
    philosophy_universities: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class DebutSection(BaseModel):
    title: Optional[str] = None
    year: Optional[str] = None
    award: Optional[str] = None
    award_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AnswerExtraction(BaseModel):
    winner: Optional[WinnerSection] = None
    uk_hardcover: Optional[UKHardcoverSection] = None
    characters: Optional[CharacterSection] = None
    author_background: Optional[AuthorBackgroundSection] = None
    debut: Optional[DebutSection] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract the requested information exactly as it appears in the answer. Return null for any missing field. Additionally, for each section, extract any URLs explicitly present in the answer that support that section (as a list of URLs).

Fields to extract:

1) winner:
   - title: the title of the 2024 Booker Prize-winning novel
   - author: the author of the 2024 Booker Prize-winning novel
   - sources: URLs cited for the winner info

2) uk_hardcover (UK hardcover first edition of the winning novel only):
   - publisher: publisher name
   - publication_date: publication date as written (e.g., "2 November 2023" or "November 2, 2023")
   - isbn13: ISBN-13 as written (keep any hyphens)
   - sources: URLs cited for these UK hardcover first edition details

3) characters (the astronaut crew facts):
   - six_astronauts_statement: the statement about six astronauts from different countries (record string if present)
   - japanese_personal_event: what significant personal event happened to the Japanese astronaut while in space
   - japanese_metaphorical_role: how the Japanese astronaut is described metaphorically among the crew (e.g., "the craft's 'conscience'")
   - sources: URLs cited for these character facts

4) author_background:
   - birthplace_county: the county (e.g., "Kent")
   - birthplace_country: the country (e.g., "England")
   - birth_year: birth year (e.g., "1975")
   - philosophy_universities: list of universities where the author studied philosophy (strings)
   - sources: URLs cited for the author background

5) debut:
   - title: the author’s debut novel title
   - year: the publication year of the debut novel
   - award: the award the debut won
   - award_year: the year the debut won the award
   - sources: URLs cited for the debut info

SPECIAL RULES:
- Extract only information explicitly present in the answer; do not infer or invent.
- For URLs, extract actual hyperlinks explicitly present (plain URLs or markdown links). If no URLs for a section, return an empty list.
- Keep strings as written in the answer (do not normalize dates or numbers).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _list_nonempty(lst: Optional[List[str]]) -> bool:
    return bool(lst and len(lst) > 0)


def _get_sources(maybe_section) -> Optional[List[str]]:
    if maybe_section and hasattr(maybe_section, "sources"):
        return list(maybe_section.sources or [])
    return None


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_winner_checks(evaluator: Evaluator, parent_node, extracted: AnswerExtraction) -> None:
    """
    Booker_Winner_Identification (parallel, critical):
      - Existence checks for title and author (critical, custom)
      - Winning_Novel_Title (critical, leaf)
      - Winning_Novel_Author (critical, leaf)
    """
    node = evaluator.add_parallel(
        id="Booker_Winner_Identification",
        desc="Correctly identify the 2024 Booker Prize-winning novel and its author",
        parent=parent_node,
        critical=True
    )

    title_present = evaluator.add_custom_node(
        result=_nonempty(extracted.winner.title) if extracted.winner else False,
        id="Winning_Novel_Title_Provided",
        desc="Winner title is provided in the answer",
        parent=node,
        critical=True
    )

    author_present = evaluator.add_custom_node(
        result=_nonempty(extracted.winner.author) if extracted.winner else False,
        id="Winning_Novel_Author_Provided",
        desc="Winner author is provided in the answer",
        parent=node,
        critical=True
    )

    # Title verification
    title_leaf = evaluator.add_leaf(
        id="Winning_Novel_Title",
        desc="Provide the title of the 2024 Booker Prize-winning novel (must be 'Orbital')",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The 2024 Booker Prize-winning novel is titled 'Orbital'.",
        node=title_leaf,
        sources=_get_sources(extracted.winner),
        additional_instruction="Use authoritative sources (e.g., The Booker Prizes official site). Allow minor punctuation/casing variants; focus on the main title 'Orbital'.",
        extra_prerequisites=[title_present]
    )

    # Author verification
    author_leaf = evaluator.add_leaf(
        id="Winning_Novel_Author",
        desc="Provide the author of the 2024 Booker Prize-winning novel (must be Samantha Harvey)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The author of the 2024 Booker Prize-winning novel is Samantha Harvey.",
        node=author_leaf,
        sources=_get_sources(extracted.winner),
        additional_instruction="Use authoritative sources (e.g., The Booker Prizes official site). Allow minor name variants (e.g., middle names/initials) as equivalent.",
        extra_prerequisites=[author_present]
    )


async def build_uk_hardcover_checks(evaluator: Evaluator, parent_node, extracted: AnswerExtraction) -> None:
    """
    UK_Hardcover_First_Edition_Details (parallel, critical):
      - Existence checks for publisher, date, isbn (custom, critical)
      - UK_Hardcover_Publisher (critical, leaf)
      - UK_Hardcover_Publication_Date (critical, leaf)
      - UK_Hardcover_ISBN13 (critical, leaf)
    """
    node = evaluator.add_parallel(
        id="UK_Hardcover_First_Edition_Details",
        desc="Provide UK hardcover first edition details matching the constraints",
        parent=parent_node,
        critical=True
    )

    # Existence
    pub_present = evaluator.add_custom_node(
        result=_nonempty(extracted.uk_hardcover.publisher) if extracted.uk_hardcover else False,
        id="UK_Hardcover_Publisher_Provided",
        desc="Publisher name for the UK hardcover first edition is provided in the answer",
        parent=node,
        critical=True
    )
    date_present = evaluator.add_custom_node(
        result=_nonempty(extracted.uk_hardcover.publication_date) if extracted.uk_hardcover else False,
        id="UK_Hardcover_Publication_Date_Provided",
        desc="Publication date for the UK hardcover first edition is provided in the answer",
        parent=node,
        critical=True
    )
    isbn_present = evaluator.add_custom_node(
        result=_nonempty(extracted.uk_hardcover.isbn13) if extracted.uk_hardcover else False,
        id="UK_Hardcover_ISBN13_Provided",
        desc="ISBN-13 for the UK hardcover first edition is provided in the answer",
        parent=node,
        critical=True
    )

    # Publisher
    pub_leaf = evaluator.add_leaf(
        id="UK_Hardcover_Publisher",
        desc="Publisher name is Jonathan Cape (UK hardcover first edition)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The UK hardcover first edition of 'Orbital' was published by Jonathan Cape.",
        node=pub_leaf,
        sources=_get_sources(extracted.uk_hardcover),
        additional_instruction="Verify the UK hardcover first edition publisher specifically. Allow imprint phrasing like 'Jonathan Cape' or 'Vintage Jonathan Cape'.",
        extra_prerequisites=[pub_present]
    )

    # Publication date
    date_leaf = evaluator.add_leaf(
        id="UK_Hardcover_Publication_Date",
        desc="Publication date is November 2, 2023 (UK hardcover first edition)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The UK hardcover first edition of 'Orbital' was published on 2 November 2023 (also written as November 2, 2023).",
        node=date_leaf,
        sources=_get_sources(extracted.uk_hardcover),
        additional_instruction="Verify the UK hardcover first edition publication date. Accept UK or US date formats (2 November 2023 or November 2, 2023).",
        extra_prerequisites=[date_present]
    )

    # ISBN-13
    isbn_leaf = evaluator.add_leaf(
        id="UK_Hardcover_ISBN13",
        desc="ISBN-13 is 978-1787334342 (UK hardcover first edition)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The UK hardcover first edition of 'Orbital' has ISBN-13 978-1787334342.",
        node=isbn_leaf,
        sources=_get_sources(extracted.uk_hardcover),
        additional_instruction="Verify the 13-digit ISBN for the UK hardcover first edition. Ignore hyphenation differences; the digits must match.",
        extra_prerequisites=[isbn_present]
    )


async def build_character_checks(evaluator: Evaluator, parent_node, extracted: AnswerExtraction) -> None:
    """
    Novel_Character_Constraints (parallel, critical):
      - Existence checks (custom, critical) for all three statements
      - Six_Astronauts_Different_Countries (critical, leaf)
      - Japanese_Astronaut_Personal_Event (critical, leaf)
      - Japanese_Astronaut_Metaphorical_Role (critical, leaf)
    """
    node = evaluator.add_parallel(
        id="Novel_Character_Constraints",
        desc="Provide the required constrained facts about the astronaut characters",
        parent=parent_node,
        critical=True
    )

    six_present = evaluator.add_custom_node(
        result=_nonempty(extracted.characters.six_astronauts_statement) if extracted.characters else False,
        id="Six_Astronauts_Provided",
        desc="Statement about six astronauts from different countries is provided in the answer",
        parent=node,
        critical=True
    )
    jp_event_present = evaluator.add_custom_node(
        result=_nonempty(extracted.characters.japanese_personal_event) if extracted.characters else False,
        id="Japanese_Astronaut_Personal_Event_Provided",
        desc="Japanese astronaut personal event is provided in the answer",
        parent=node,
        critical=True
    )
    jp_role_present = evaluator.add_custom_node(
        result=_nonempty(extracted.characters.japanese_metaphorical_role) if extracted.characters else False,
        id="Japanese_Astronaut_Metaphorical_Role_Provided",
        desc="Japanese astronaut metaphorical role is provided in the answer",
        parent=node,
        critical=True
    )

    # Six astronauts statement
    six_leaf = evaluator.add_leaf(
        id="Six_Astronauts_Different_Countries",
        desc="State that the novel contains six astronaut characters from different countries",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The novel 'Orbital' features six astronaut characters from different countries.",
        node=six_leaf,
        sources=_get_sources(extracted.characters),
        additional_instruction="Verify that the crew consists of six astronauts and they are from different countries. Synonyms or paraphrases are acceptable.",
        extra_prerequisites=[six_present]
    )

    # Japanese astronaut personal event
    jp_event_leaf = evaluator.add_leaf(
        id="Japanese_Astronaut_Personal_Event",
        desc="State that the Japanese astronaut's mother died while the astronaut was in space",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="In the novel 'Orbital', the Japanese astronaut's mother died while the astronaut was in space.",
        node=jp_event_leaf,
        sources=_get_sources(extracted.characters),
        additional_instruction="Check the relevant plot/character description. Allow paraphrases but the fact must be clear.",
        extra_prerequisites=[jp_event_present]
    )

    # Japanese astronaut metaphorical role
    jp_role_leaf = evaluator.add_leaf(
        id="Japanese_Astronaut_Metaphorical_Role",
        desc="State that the Japanese astronaut is described as the craft's 'conscience'",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="In the novel 'Orbital', the Japanese astronaut is described as the craft's 'conscience' among the crew.",
        node=jp_role_leaf,
        sources=_get_sources(extracted.characters),
        additional_instruction="Paraphrases such as 'acts as the conscience of the craft' should count as equivalent.",
        extra_prerequisites=[jp_role_present]
    )


async def build_author_background_checks(evaluator: Evaluator, parent_node, extracted: AnswerExtraction) -> None:
    """
    Author_Background_Constraints (parallel, critical):
      - Existence checks for birth facts and universities (custom, critical)
      - Author_Birth_Facts (critical, leaf)
      - Author_Philosophy_Universities (critical, leaf)
    """
    node = evaluator.add_parallel(
        id="Author_Background_Constraints",
        desc="Provide the author's constrained birth and education facts",
        parent=parent_node,
        critical=True
    )

    birth_present = evaluator.add_custom_node(
        result=(
            (_nonempty(extracted.author_background.birthplace_county) and
             _nonempty(extracted.author_background.birthplace_country) and
             _nonempty(extracted.author_background.birth_year))
            if extracted.author_background else False
        ),
        id="Author_Birth_Facts_Provided",
        desc="Author birthplace county, country, and birth year are provided in the answer",
        parent=node,
        critical=True
    )

    unis_present = evaluator.add_custom_node(
        result=(
            bool(extracted.author_background and extracted.author_background.philosophy_universities and
                 len(extracted.author_background.philosophy_universities) >= 2)
        ),
        id="Author_Philosophy_Universities_Provided",
        desc="Author philosophy universities are provided in the answer",
        parent=node,
        critical=True
    )

    # Author birth facts
    birth_leaf = evaluator.add_leaf(
        id="Author_Birth_Facts",
        desc="State the author was born in Kent, England in 1975",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Samantha Harvey was born in Kent, England in 1975.",
        node=birth_leaf,
        sources=_get_sources(extracted.author_background),
        additional_instruction="Allow minor variations in wording. The county must be Kent, the country England, and the year 1975.",
        extra_prerequisites=[birth_present]
    )

    # Universities
    unis_leaf = evaluator.add_leaf(
        id="Author_Philosophy_Universities",
        desc="State the author studied philosophy at both the University of York and the University of Sheffield",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Samantha Harvey studied philosophy at both the University of York and the University of Sheffield.",
        node=unis_leaf,
        sources=_get_sources(extracted.author_background),
        additional_instruction="Accept close variants of the university names (e.g., 'York' or 'Sheffield') as long as they clearly refer to the universities.",
        extra_prerequisites=[unis_present]
    )


async def build_debut_checks(evaluator: Evaluator, parent_node, extracted: AnswerExtraction) -> None:
    """
    Debut_Novel_Constraints (parallel, critical):
      - Existence checks (custom, critical) for title+year and award+year
      - Debut_Novel_Title_And_Year (critical, leaf)
      - Debut_Novel_Award_And_Year (critical, leaf)
    """
    node = evaluator.add_parallel(
        id="Debut_Novel_Constraints",
        desc="Provide the author's constrained debut novel facts",
        parent=parent_node,
        critical=True
    )

    title_year_present = evaluator.add_custom_node(
        result=(
            (_nonempty(extracted.debut.title) and _nonempty(extracted.debut.year))
            if extracted.debut else False
        ),
        id="Debut_Novel_Title_And_Year_Provided",
        desc="Debut novel title and year are provided in the answer",
        parent=node,
        critical=True
    )

    award_year_present = evaluator.add_custom_node(
        result=(
            (_nonempty(extracted.debut.award) and _nonempty(extracted.debut.award_year))
            if extracted.debut else False
        ),
        id="Debut_Novel_Award_And_Year_Provided",
        desc="Debut novel award and year are provided in the answer",
        parent=node,
        critical=True
    )

    # Debut title and year
    debut_title_leaf = evaluator.add_leaf(
        id="Debut_Novel_Title_And_Year",
        desc="State the debut novel title is 'The Wilderness' and that it was published in 2009",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Samantha Harvey's debut novel is 'The Wilderness', published in 2009.",
        node=debut_title_leaf,
        sources=_get_sources(extracted.debut),
        additional_instruction="Allow minor punctuation/casing variants. Ensure the publication year is 2009.",
        extra_prerequisites=[title_year_present]
    )

    # Debut award and year
    debut_award_leaf = evaluator.add_leaf(
        id="Debut_Novel_Award_And_Year",
        desc="State the debut novel won the Betty Trask Prize in 2009",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Samantha Harvey's 'The Wilderness' won the Betty Trask Prize in 2009.",
        node=debut_award_leaf,
        sources=_get_sources(extracted.debut),
        additional_instruction="Accept 'Betty Trask Prize' or 'Betty Trask Award' naming variants if the same prize is clearly indicated; the year must be 2009.",
        extra_prerequisites=[award_year_present]
    )


async def build_required_details(evaluator: Evaluator, parent_node, extracted: AnswerExtraction) -> None:
    """
    Required_Details (parallel, critical):
      - UK_Hardcover_First_Edition_Details
      - Novel_Character_Constraints
      - Author_Background_Constraints
      - Debut_Novel_Constraints
    """
    node = evaluator.add_parallel(
        id="Required_Details",
        desc="Provide publication details, required character facts, author background, and debut novel facts per constraints",
        parent=parent_node,
        critical=True
    )

    await build_uk_hardcover_checks(evaluator, node, extracted)
    await build_character_checks(evaluator, node, extracted)
    await build_author_background_checks(evaluator, node, extracted)
    await build_debut_checks(evaluator, node, extracted)


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
    Evaluate an answer for the 2024 Booker Prize research task.
    """
    # Initialize evaluator and root node
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

    # Extract all structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=AnswerExtraction,
        extraction_name="answer_extraction"
    )

    # Add ground truth info for transparency
    evaluator.add_ground_truth({
        "winner": {"title": GT_WINNER_TITLE, "author": GT_WINNER_AUTHOR},
        "uk_hardcover": {
            "publisher": GT_UK_PUBLISHER,
            "publication_date_accept": [GT_UK_PUBLICATION_DATE, GT_UK_PUBLICATION_DATE_US],
            "isbn13": GT_UK_ISBN13
        },
        "characters": {
            "six_astronauts": True,
            "japanese_event": "mother died while in space",
            "japanese_metaphor": "craft's conscience"
        },
        "author_background": {
            "birthplace_county": GT_AUTHOR_BIRTHPLACE_COUNTY,
            "birthplace_country": GT_AUTHOR_BIRTHPLACE_COUNTRY,
            "birth_year": GT_AUTHOR_BIRTH_YEAR,
            "philosophy_universities": GT_AUTHOR_UNIS
        },
        "debut": {
            "title": GT_DEBUT_TITLE,
            "year": GT_DEBUT_YEAR,
            "award": GT_DEBUT_AWARD,
            "award_year": GT_DEBUT_AWARD_YEAR
        }
    }, gt_type="ground_truth")

    # Build top-level sequential critical node to enforce ordering:
    # If winner identification is wrong, subsequent details are meaningless and should be skipped.
    complete_task = evaluator.add_sequential(
        id="Complete_Research_Task",
        desc="Answer all required parts about the 2024 Booker Prize-winning novel, consistent with the provided constraints",
        parent=root,
        critical=True
    )

    # 1) Winner identification (parallel, critical)
    await build_winner_checks(evaluator, complete_task, extracted)

    # 2) Required details (parallel, critical)
    await build_required_details(evaluator, complete_task, extracted)

    # Return evaluation summary
    return evaluator.get_summary()