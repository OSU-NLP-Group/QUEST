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
TASK_ID = "literary_awards_2025"
TASK_DESCRIPTION = """
Identify the winners of four major literary fiction awards announced in 2025: the Booker Prize for Fiction, the National Book Award for Fiction, the Pulitzer Prize for Fiction, and the Women's Prize for Fiction.

For each of the four award winners, provide the following comprehensive information:

1. Author's full name (the award winner)
2. Book title (the winning novel)
3. Publisher information:
   - Publisher name
   - Parent company or imprint affiliation
   - A URL reference confirming the publisher
4. Author biographical information:
   - Birth year and birthplace (city and country)
   - Nationality/citizenship
   - Educational background (university name, specific college if applicable, subject/degree)
   - Current residence or professional position (where relevant)
   - URL references confirming biographical details
5. Award announcement date with URL reference

All information must be supported by URL references to verifiable sources such as official award websites, reputable news outlets, publisher pages, or author biography pages.
"""

# Expected ground-truth facts per rubric
EXPECTED = {
    "booker": {
        "award_name": "Booker Prize for Fiction",
        "year": "2025",
        "winner": "David Szalay",
        "book": "Flesh",
        "publisher": {
            "name": "Jonathan Cape",
            "imprint": "Vintage Publishing UK",
            "parent": "Penguin Random House",
        },
        "bio": {
            "birth_year": "1974",
            "birthplace_city": "Montreal",
            "birthplace_country": "Canada",
            "citizenships": ["Canadian", "Hungarian", "British"],  # UK/British acceptable
            "education_contains": ["University of Oxford", "Brasenose College", "English"],
            "current_contains": ["Vienna", "Austria"],
        },
        "announcement_date": "November 10, 2025",
    },
    "nba": {
        "award_name": "National Book Award for Fiction",
        "year": "2025",
        "winner": "Rabih Alameddine",
        "book": "The True True Story of Raja the Gullible (and His Mother)",
        "publisher": {
            "name": "Grove Press",
            "imprint": "Grove Atlantic",
            "parent": None,
        },
        "bio": {
            "birth_year": "1959",
            "birthplace_city": "Amman",
            "birthplace_country": "Jordan",
            "nationalities": ["Lebanese"],
        },
        "announcement_date": "November 20, 2025",
    },
    "pulitzer": {
        "award_name": "Pulitzer Prize for Fiction",
        "year": "2025",
        "winner": "Percival Everett",
        "book": "James",
        "publisher": {
            "name": "Doubleday",
            "imprint": None,
            "parent": "Penguin Random House",
        },
        "bio": {
            "dob_text": "December 22, 1956",
            "education_checks": [
                {"institution": "University of Miami", "subject_or_degree": "philosophy"},
                {"institution": "Brown University", "subject_or_degree": "master", "year": "1982"},
            ],
            "position_contains": ["Distinguished Professor", "University of Southern California"],
        },
        "announcement_date": "May 5, 2025",
    },
    "womens": {
        "award_name": "Women's Prize for Fiction",
        "year": "2025",
        "winner": "Yael van der Wouden",
        "book": "The Safekeep",
        "publisher": {
            "name": "Avid Reader Press / Simon & Schuster",
            "imprint": "Simon & Schuster",
            "parent": None,
        },
        "bio": {
            "nationality": "Dutch",
            "debut_claim": True,  # "The Safekeep" is her debut novel
        },
        "announcement_date": "June 12, 2025",
    },
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PublisherInfo(BaseModel):
    name: Optional[str] = None
    imprint_or_parent: Optional[str] = None  # e.g., "Vintage Publishing UK", "Grove Atlantic", "Simon & Schuster"
    parent_company: Optional[str] = None  # e.g., "Penguin Random House"
    urls: List[str] = Field(default_factory=list)


class BioInfo(BaseModel):
    birth_year: Optional[str] = None
    birthplace_city: Optional[str] = None
    birthplace_country: Optional[str] = None
    nationality_or_citizenship: Optional[str] = None  # free-form string or comma-separated list
    education: Optional[str] = None  # free-form description
    current_residence_or_position: Optional[str] = None  # free-form
    urls: List[str] = Field(default_factory=list)


class AwardItem(BaseModel):
    award_name: Optional[str] = None
    winner_author: Optional[str] = None
    book_title: Optional[str] = None
    winner_urls: List[str] = Field(default_factory=list)

    publisher: Optional[PublisherInfo] = None
    bio: Optional[BioInfo] = None

    announcement_date: Optional[str] = None
    announcement_url: Optional[str] = None


class AwardsExtraction(BaseModel):
    booker: Optional[AwardItem] = None
    national_book_award: Optional[AwardItem] = None
    pulitzer: Optional[AwardItem] = None
    womens_prize: Optional[AwardItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_awards() -> str:
    return """
Extract structured information for FOUR specific 2025 literary fiction awards from the provided answer text. Map each award to the designated field exactly as instructed below. Extract ONLY what is explicitly stated in the answer.

For EACH award, extract the following fields:

Common fields:
- award_name: The award name as it appears in the answer (e.g., "Booker Prize for Fiction")
- winner_author: Full name of the winning author
- book_title: Title of the winning novel
- winner_urls: An array of URLs explicitly cited in the answer that directly support the winner and the winning book information

Publisher fields (nested object "publisher"):
- name: Publisher name of the winning novel
- imprint_or_parent: The imprint or immediate parent of the publisher (if mentioned)
- parent_company: The broader parent company (if mentioned)
- urls: An array of URLs explicitly cited that confirm the book’s publisher and/or the imprint/parent relationship

Author biography fields (nested object "bio"):
- birth_year: Year of birth (if stated)
- birthplace_city: Birthplace city (if stated)
- birthplace_country: Birthplace country (if stated)
- nationality_or_citizenship: Nationality/citizenship as phrased in the answer (e.g., "Canadian, Hungarian, and British")
- education: Education summary as phrased in the answer (e.g., "read English at University of Oxford (Brasenose College)")
- current_residence_or_position: Current residence or professional position as phrased (e.g., "lives in Vienna, Austria" / "Distinguished Professor at USC")
- urls: An array of URLs explicitly cited in the answer that support the biographical details

Award announcement:
- announcement_date: The date the winner was announced (as presented in the answer)
- announcement_url: A single URL that specifically supports the winners announcement date

Return a JSON object with these top-level keys:
- "booker": AwardItem for the 2025 Booker Prize for Fiction
- "national_book_award": AwardItem for the 2025 National Book Award for Fiction
- "pulitzer": AwardItem for the 2025 Pulitzer Prize for Fiction
- "womens_prize": AwardItem for the 2025 Women's Prize for Fiction

If any field is missing in the answer, set it to null (or an empty array for URL lists). Do NOT invent URLs; only include those explicitly present in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    return urls if urls else []


def _nonempty(s: Optional[str]) -> bool:
    return s is not None and str(s).strip() != ""


# --------------------------------------------------------------------------- #
# Verification builder for a single award package                             #
# --------------------------------------------------------------------------- #
async def verify_award_package(
    evaluator: Evaluator,
    parent_node,
    item_id: str,
    item_desc: str,
    award_key: str,
    extracted: Optional[AwardItem],
) -> None:
    """
    Build verification sub-tree and verify a single award package against expected facts and cited URLs.
    Each core requirement is broken down into binary leaf checks with URL grounding where applicable.
    """
    expected = EXPECTED[award_key]

    # Item node (non-critical, allows partial scoring across awards)
    item_node = evaluator.add_parallel(
        id=item_id,
        desc=item_desc,
        parent=parent_node,
        critical=False,
    )

    # ----------------- Winner and Book ----------------- #
    winner_group = evaluator.add_parallel(
        id=f"{item_id}_Winner_and_Book_Correct",
        desc=f"{expected['award_name']} {expected['year']}: Winner and book are correct and supported",
        parent=item_node,
        critical=True,
    )

    winner_urls = _safe_urls(extracted.winner_urls if extracted else [])
    winner_exists = (
        extracted is not None
        and _nonempty(extracted.winner_author)
        and _nonempty(extracted.book_title)
        and len(winner_urls) > 0
    )
    evaluator.add_custom_node(
        result=winner_exists,
        id=f"{item_id}_winner_book_fields_present",
        desc="Winner, book title, and at least one URL are provided in the answer",
        parent=winner_group,
        critical=True,
    )

    # Winner name matches expected (simple check)
    winner_match_leaf = evaluator.add_leaf(
        id=f"{item_id}_winner_name_matches_expected",
        desc=f"Winner name in answer matches expected '{expected['winner']}'",
        parent=winner_group,
        critical=True,
    )
    extracted_winner = extracted.winner_author if extracted else ""
    await evaluator.verify(
        claim=f"The names '{extracted_winner}' and '{expected['winner']}' refer to the same person.",
        node=winner_match_leaf,
        additional_instruction="Allow minor variations (case, middle names/initials). Consider common variants equivalent.",
    )

    # Book title matches expected (simple check)
    book_match_leaf = evaluator.add_leaf(
        id=f"{item_id}_book_title_matches_expected",
        desc=f"Book title in answer matches expected '{expected['book']}'",
        parent=winner_group,
        critical=True,
    )
    extracted_book = extracted.book_title if extracted else ""
    await evaluator.verify(
        claim=f"The book title '{extracted_book}' matches the expected title '{expected['book']}'.",
        node=book_match_leaf,
        additional_instruction="Allow minor punctuation/case variations; treat equivalent titles as a match.",
    )

    # Winner/book supported by cited URLs
    winner_supported_leaf = evaluator.add_leaf(
        id=f"{item_id}_winner_book_supported_by_urls",
        desc=f"The winner and winning book are supported by cited source(s)",
        parent=winner_group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The {expected['year']} {expected['award_name']} was awarded to {extracted_winner} for '{extracted_book}'.",
        node=winner_supported_leaf,
        sources=winner_urls,
        additional_instruction="Verify the page explicitly indicates this winner and winning title for the specified award/year.",
    )

    # ----------------- Publisher Info ----------------- #
    pub_group = evaluator.add_parallel(
        id=f"{item_id}_Publisher_Info_Correct",
        desc=f"{expected['award_name']} {expected['year']}: Publisher info correct and supported",
        parent=item_node,
        critical=True,
    )

    pub = extracted.publisher if extracted else None
    pub_urls = _safe_urls(pub.urls if pub else [])

    pub_exists = (
        pub is not None
        and _nonempty(pub.name)
        and len(pub_urls) > 0
    )
    evaluator.add_custom_node(
        result=pub_exists,
        id=f"{item_id}_publisher_fields_present",
        desc="Publisher name and at least one publisher-related URL are provided",
        parent=pub_group,
        critical=True,
    )

    # Publisher name matches expected
    pub_name_leaf = evaluator.add_leaf(
        id=f"{item_id}_publisher_name_matches_expected",
        desc=f"Publisher name matches expected '{expected['publisher']['name']}'",
        parent=pub_group,
        critical=True,
    )
    extracted_pub_name = pub.name if pub else ""
    # For Women's Prize, accept 'Avid Reader Press' or 'Avid Reader Press / Simon & Schuster'
    womens_flex_note = ""
    if award_key == "womens":
        womens_flex_note = " If the answer lists 'Avid Reader Press' alone or 'Avid Reader Press / Simon & Schuster', accept as correct."

    await evaluator.verify(
        claim=f"The publisher name '{extracted_pub_name}' matches the expected '{expected['publisher']['name']}'.",
        node=pub_name_leaf,
        additional_instruction="Allow minor formatting/casing variations." + womens_flex_note,
    )

    # Book published by this publisher (URL grounded)
    pub_book_supported_leaf = evaluator.add_leaf(
        id=f"{item_id}_book_published_by_publisher_supported",
        desc="The book's stated publisher is supported by cited source(s)",
        parent=pub_group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The novel '{extracted_book}' was published by {extracted_pub_name}.",
        node=pub_book_supported_leaf,
        sources=pub_urls,
        additional_instruction="Verify the page indicates this specific book was published by the stated publisher.",
    )

    # Imprint/parent relationships as applicable
    if expected["publisher"].get("imprint"):
        # Imprint relation match (simple) and supported (URL)
        exp_imprint = expected["publisher"]["imprint"]
        impr_match_leaf = evaluator.add_leaf(
            id=f"{item_id}_imprint_matches_expected",
            desc=f"Imprint/affiliation matches expected '{exp_imprint}'",
            parent=pub_group,
            critical=True,
        )
        extracted_imprint = pub.imprint_or_parent if pub else ""
        await evaluator.verify(
            claim=f"The extracted imprint/affiliation '{extracted_imprint}' matches the expected '{exp_imprint}'.",
            node=impr_match_leaf,
            additional_instruction="Allow reasonable naming variants (e.g., 'Vintage', 'Vintage Publishing (UK)').",
        )

        impr_supported_leaf = evaluator.add_leaf(
            id=f"{item_id}_imprint_supported_by_urls",
            desc="Imprint relationship is supported by cited source(s)",
            parent=pub_group,
            critical=True,
        )
        # Use extracted values for grounding
        await evaluator.verify(
            claim=f"{extracted_pub_name} is an imprint of {extracted_imprint}.",
            node=impr_supported_leaf,
            sources=pub_urls,
            additional_instruction="The page should clearly state the imprint relationship (e.g., 'X is an imprint of Y').",
        )

    if expected["publisher"].get("parent"):
        exp_parent = expected["publisher"]["parent"]
        parent_match_leaf = evaluator.add_leaf(
            id=f"{item_id}_parent_matches_expected",
            desc=f"Parent company matches expected '{exp_parent}'",
            parent=pub_group,
            critical=True,
        )
        extracted_parent = pub.parent_company if pub else ""
        await evaluator.verify(
            claim=f"The extracted parent company '{extracted_parent}' matches the expected '{exp_parent}'.",
            node=parent_match_leaf,
            additional_instruction="Allow minor naming variants (e.g., 'PRH' for 'Penguin Random House').",
        )

        parent_supported_leaf = evaluator.add_leaf(
            id=f"{item_id}_parent_supported_by_urls",
            desc="Parent company relationship is supported by cited source(s)",
            parent=pub_group,
            critical=True,
        )
        await evaluator.verify(
            claim=f"{extracted_pub_name} is part of {extracted_parent} or within a division owned by {extracted_parent}.",
            node=parent_supported_leaf,
            sources=pub_urls,
            additional_instruction="The page should indicate affiliation with the stated parent (accept equivalent phrasings).",
        )

    # ----------------- Author Bio ----------------- #
    bio_group = evaluator.add_parallel(
        id=f"{item_id}_Author_Bio_Correct",
        desc=f"{expected['award_name']} {expected['year']}: Author bio facts are correct and supported",
        parent=item_node,
        critical=True,
    )

    bio = extracted.bio if extracted else None
    bio_urls = _safe_urls(bio.urls if bio else [])

    bio_exists = (bio is not None and len(bio_urls) > 0)
    evaluator.add_custom_node(
        result=bio_exists,
        id=f"{item_id}_bio_urls_present",
        desc="At least one biographical URL is provided in the answer",
        parent=bio_group,
        critical=True,
    )

    # Award-specific bio checks
    if award_key == "booker":
        # Birth year & birthplace (match expected and supported by URLs)
        by_match_leaf = evaluator.add_leaf(
            id=f"{item_id}_birth_year_matches_expected",
            desc="Birth year matches expected (1974)",
            parent=bio_group,
            critical=True,
        )
        await evaluator.verify(
            claim=f"The extracted birth year '{bio.birth_year if bio else ''}' equals '1974'.",
            node=by_match_leaf,
            additional_instruction="Treat '1974' as the expected value; allow minor formatting issues.",
        )

        bpl_match_leaf = evaluator.add_leaf(
            id=f"{item_id}_birthplace_matches_expected",
            desc="Birthplace matches expected (Montreal, Canada)",
            parent=bio_group,
            critical=True,
        )
        await evaluator.verify(
            claim=f"The extracted birthplace is '{(bio.birthplace_city if bio else '')}, {(bio.birthplace_country if bio else '')}', which should match 'Montreal, Canada'.",
            node=bpl_match_leaf,
            additional_instruction="Allow reasonable variants (e.g., 'Montréal').",
        )

        birth_supported_leaf = evaluator.add_leaf(
            id=f"{item_id}_birth_supported_by_urls",
            desc="Birth year and place supported by cited bio URL(s)",
            parent=bio_group,
            critical=True,
        )
        await evaluator.verify(
            claim=f"David Szalay was born in 1974 in Montreal, Canada.",
            node=birth_supported_leaf,
            sources=bio_urls,
            additional_instruction="The page should clearly indicate both the year and city/country of birth.",
        )

        # Citizenship (match and support)
        cit_match_leaf = evaluator.add_leaf(
            id=f"{item_id}_citizenship_matches_expected",
            desc="Citizenship includes Canadian, Hungarian, and UK/British",
            parent=bio_group,
            critical=True,
        )
        await evaluator.verify(
            claim=f"The extracted nationality/citizenship '{bio.nationality_or_citizenship if bio else ''}' includes Canadian, Hungarian, and UK/British.",
            node=cit_match_leaf,
            additional_instruction="Return Correct only if all three are included (British/UK acceptable).",
        )

        cit_supported_leaf = evaluator.add_leaf(
            id=f"{item_id}_citizenship_supported_by_urls",
            desc="Citizenship is supported by cited bio URL(s)",
            parent=bio_group,
            critical=True,
        )
        await evaluator.verify(
            claim="David Szalay holds Canadian, Hungarian, and British/UK citizenship.",
            node=cit_supported_leaf,
            sources=bio_urls,
            additional_instruction="Accept 'British' for 'UK' citizenship.",
        )

        # Education (match and support)
        edu_match_leaf = evaluator.add_leaf(
            id=f"{item_id}_education_matches_expected",
            desc="Education matches expected (Oxford/ Brasenose College, English)",
            parent=bio_group,
            critical=True,
        )
        await evaluator.verify(
            claim=f"The extracted education '{bio.education if bio else ''}' indicates University of Oxford (Brasenose College) and English.",
            node=edu_match_leaf,
            additional_instruction="Accept phrasings like 'read English at Oxford' / 'Brasenose College'.",
        )

        edu_supported_leaf = evaluator.add_leaf(
            id=f"{item_id}_education_supported_by_urls",
            desc="Education supported by cited bio URL(s)",
            parent=bio_group,
            critical=True,
        )
        await evaluator.verify(
            claim="David Szalay studied English at the University of Oxford (Brasenose College).",
            node=edu_supported_leaf,
            sources=bio_urls,
            additional_instruction="Accept equivalent wordings.",
        )

        # Current residence (match and support)
        res_match_leaf = evaluator.add_leaf(
            id=f"{item_id}_residence_matches_expected",
            desc="Current residence matches expected (Vienna, Austria)",
            parent=bio_group,
            critical=True,
        )
        await evaluator.verify(
            claim=f"The extracted current residence/position '{bio.current_residence_or_position if bio else ''}' indicates that he lives in Vienna, Austria.",
            node=res_match_leaf,
            additional_instruction="Accept variants like 'based in Vienna'.",
        )

        res_supported_leaf = evaluator.add_leaf(
            id=f"{item_id}_residence_supported_by_urls",
            desc="Current residence supported by cited bio URL(s)",
            parent=bio_group,
            critical=True,
        )
        await evaluator.verify(
            claim="David Szalay currently lives in Vienna, Austria.",
            node=res_supported_leaf,
            sources=bio_urls,
            additional_instruction="Accept equivalent statements that clearly imply residence in Vienna.",
        )

    elif award_key == "nba":
        # Birth year and birthplace
        by_match_leaf = evaluator.add_leaf(
            id=f"{item_id}_birth_year_matches_expected",
            desc="Birth year matches expected (1959)",
            parent=bio_group,
            critical=True,
        )
        await evaluator.verify(
            claim=f"The extracted birth year '{bio.birth_year if bio else ''}' equals '1959'.",
            node=by_match_leaf,
            additional_instruction="Treat '1959' as the expected value.",
        )

        bpl_match_leaf = evaluator.add_leaf(
            id=f"{item_id}_birthplace_matches_expected",
            desc="Birthplace matches expected (Amman, Jordan)",
            parent=bio_group,
            critical=True,
        )
        await evaluator.verify(
            claim=f"The extracted birthplace is '{(bio.birthplace_city if bio else '')}, {(bio.birthplace_country if bio else '')}', which should match 'Amman, Jordan'.",
            node=bpl_match_leaf,
            additional_instruction="Allow minor variants in spelling.",
        )

        birth_supported_leaf = evaluator.add_leaf(
            id=f"{item_id}_birth_supported_by_urls",
            desc="Birth year and place supported by cited bio URL(s)",
            parent=bio_group,
            critical=True,
        )
        await evaluator.verify(
            claim="Rabih Alameddine was born in 1959 in Amman, Jordan.",
            node=birth_supported_leaf,
            sources=bio_urls,
            additional_instruction="The page should indicate both year and place of birth.",
        )

        # Nationality
        nat_match_leaf = evaluator.add_leaf(
            id=f"{item_id}_nationality_matches_expected",
            desc="Nationality matches expected (Lebanese)",
            parent=bio_group,
            critical=True,
        )
        await evaluator.verify(
            claim=f"The extracted nationality/citizenship '{bio.nationality_or_citizenship if bio else ''}' indicates Lebanese nationality.",
            node=nat_match_leaf,
            additional_instruction="Accept phrasing like 'Lebanese author' or 'of Lebanese nationality'.",
        )

        nat_supported_leaf = evaluator.add_leaf(
            id=f"{item_id}_nationality_supported_by_urls",
            desc="Nationality supported by cited bio URL(s)",
            parent=bio_group,
            critical=True,
        )
        await evaluator.verify(
            claim="Rabih Alameddine is of Lebanese nationality.",
            node=nat_supported_leaf,
            sources=bio_urls,
            additional_instruction="Accept equivalent wording.",
        )

    elif award_key == "pulitzer":
        # Date of birth
        dob_supported_leaf = evaluator.add_leaf(
            id=f"{item_id}_dob_supported_by_urls",
            desc="Date of birth supported by cited bio URL(s)",
            parent=bio_group,
            critical=True,
        )
        await evaluator.verify(
            claim="Percival Everett was born on December 22, 1956.",
            node=dob_supported_leaf,
            sources=bio_urls,
            additional_instruction="Accept '22 December 1956' or similar formatting variations.",
        )

        # Education: University of Miami philosophy
        edu_miami_match_leaf = evaluator.add_leaf(
            id=f"{item_id}_edu_miami_matches_expected",
            desc="Education mentions University of Miami and philosophy",
            parent=bio_group,
            critical=True,
        )
        await evaluator.verify(
            claim=f"The extracted education '{bio.education if bio else ''}' indicates study of philosophy at the University of Miami (bachelor's).",
            node=edu_miami_match_leaf,
            additional_instruction="Accept equivalent phrasing like 'BA in Philosophy from the University of Miami'.",
        )

        edu_miami_supported_leaf = evaluator.add_leaf(
            id=f"{item_id}_edu_miami_supported_by_urls",
            desc="University of Miami (philosophy) supported by cited bio URL(s)",
            parent=bio_group,
            critical=True,
        )
        await evaluator.verify(
            claim="Percival Everett studied philosophy at the University of Miami (bachelor's degree).",
            node=edu_miami_supported_leaf,
            sources=bio_urls,
            additional_instruction="Accept equivalent wording.",
        )

        # Education: Brown University master's in fiction, 1982
        edu_brown_match_leaf = evaluator.add_leaf(
            id=f"{item_id}_edu_brown_matches_expected",
            desc="Education mentions Brown University (master's in fiction, 1982)",
            parent=bio_group,
            critical=True,
        )
        await evaluator.verify(
            claim=f"The extracted education '{bio.education if bio else ''}' indicates a master's in fiction from Brown University in 1982.",
            node=edu_brown_match_leaf,
            additional_instruction="Accept equivalent phrasing; the key facts are Brown University, master's in fiction, and year 1982.",
        )

        edu_brown_supported_leaf = evaluator.add_leaf(
            id=f"{item_id}_edu_brown_supported_by_urls",
            desc="Brown University master's (1982) supported by cited bio URL(s)",
            parent=bio_group,
            critical=True,
        )
        await evaluator.verify(
            claim="Percival Everett earned a master's degree in fiction from Brown University in 1982.",
            node=edu_brown_supported_leaf,
            sources=bio_urls,
            additional_instruction="Accept equivalent wording.",
        )

        # Professional position: Distinguished Professor at USC
        pos_match_leaf = evaluator.add_leaf(
            id=f"{item_id}_position_matches_expected",
            desc="Professional position mentions Distinguished Professor at USC",
            parent=bio_group,
            critical=True,
        )
        await evaluator.verify(
            claim=f"The extracted current residence/position '{bio.current_residence_or_position if bio else ''}' indicates he is a Distinguished Professor at the University of Southern California.",
            node=pos_match_leaf,
            additional_instruction="Accept 'Distinguished Professor of English at USC'.",
        )

        pos_supported_leaf = evaluator.add_leaf(
            id=f"{item_id}_position_supported_by_urls",
            desc="USC Distinguished Professor position supported by cited bio URL(s)",
            parent=bio_group,
            critical=True,
        )
        await evaluator.verify(
            claim="Percival Everett is a Distinguished Professor (of English) at the University of Southern California.",
            node=pos_supported_leaf,
            sources=bio_urls,
            additional_instruction="Accept equivalent phrasing; affiliation with USC must be clear.",
        )

    elif award_key == "womens":
        # Nationality (Dutch)
        nat_match_leaf = evaluator.add_leaf(
            id=f"{item_id}_nationality_matches_expected",
            desc="Nationality matches expected (Dutch)",
            parent=bio_group,
            critical=True,
        )
        await evaluator.verify(
            claim=f"The extracted nationality/citizenship '{bio.nationality_or_citizenship if bio else ''}' indicates she is Dutch.",
            node=nat_match_leaf,
            additional_instruction="Accept phrasing like 'Dutch author'.",
        )

        nat_supported_leaf = evaluator.add_leaf(
            id=f"{item_id}_nationality_supported_by_urls",
            desc="Dutch nationality supported by cited bio URL(s)",
            parent=bio_group,
            critical=True,
        )
        await evaluator.verify(
            claim="Yael van der Wouden is a Dutch author.",
            node=nat_supported_leaf,
            sources=bio_urls,
            additional_instruction="Accept equivalent wording.",
        )

        # Debut novel claim
        debut_supported_leaf = evaluator.add_leaf(
            id=f"{item_id}_debut_supported_by_urls",
            desc="Debut novel claim is supported by cited URL(s)",
            parent=bio_group,
            critical=True,
        )
        await evaluator.verify(
            claim="The Safekeep is Yael van der Wouden's debut novel.",
            node=debut_supported_leaf,
            sources=bio_urls,
            additional_instruction="The page should clearly indicate 'debut' or first novel.",
        )

    # ----------------- Announcement Date ----------------- #
    ann_leaf_group = evaluator.add_parallel(
        id=f"{item_id}_Announcement_Date_Correct",
        desc=f"{expected['award_name']} {expected['year']}: Announcement date is correct and supported",
        parent=item_node,
        critical=True,
    )

    ann_fields_present = (
        extracted is not None
        and _nonempty(extracted.announcement_date)
        and _nonempty(extracted.announcement_url)
    )
    evaluator.add_custom_node(
        result=ann_fields_present,
        id=f"{item_id}_announcement_fields_present",
        desc="Announcement date and its URL are provided in the answer",
        parent=ann_leaf_group,
        critical=True,
    )

    ann_match_leaf = evaluator.add_leaf(
        id=f"{item_id}_announcement_date_matches_expected",
        desc=f"Announcement date matches expected '{expected['announcement_date']}'",
        parent=ann_leaf_group,
        critical=True,
    )
    extracted_ann_date = extracted.announcement_date if extracted else ""
    await evaluator.verify(
        claim=f"The extracted announcement date '{extracted_ann_date}' equals '{expected['announcement_date']}'.",
        node=ann_match_leaf,
        additional_instruction="Allow minor formatting variants (e.g., abbreviations for months).",
    )

    ann_supported_leaf = evaluator.add_leaf(
        id=f"{item_id}_announcement_supported_by_url",
        desc="Announcement date is supported by the cited URL",
        parent=ann_leaf_group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The {expected['award_name']} {expected['year']} winner(s) were announced on {extracted_ann_date}.",
        node=ann_supported_leaf,
        sources=extracted.announcement_url if extracted else None,
        additional_instruction="The page should clearly show the announcement date; equivalent date formats acceptable.",
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the 2025 literary awards winners task.
    """
    # Initialize evaluator (root as parallel, non-critical to allow partial credit across awards)
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

    # Extract the structured information
    extracted_awards = await evaluator.extract(
        prompt=prompt_extract_awards(),
        template_class=AwardsExtraction,
        extraction_name="awards_extraction",
    )

    # Add ground truth information for transparency
    evaluator.add_ground_truth(
        {
            "expected": {
                "booker": {
                    "award_name": EXPECTED["booker"]["award_name"],
                    "year": EXPECTED["booker"]["year"],
                    "winner": EXPECTED["booker"]["winner"],
                    "book": EXPECTED["booker"]["book"],
                    "publisher": EXPECTED["booker"]["publisher"],
                    "bio": EXPECTED["booker"]["bio"],
                    "announcement_date": EXPECTED["booker"]["announcement_date"],
                },
                "national_book_award": {
                    "award_name": EXPECTED["nba"]["award_name"],
                    "year": EXPECTED["nba"]["year"],
                    "winner": EXPECTED["nba"]["winner"],
                    "book": EXPECTED["nba"]["book"],
                    "publisher": EXPECTED["nba"]["publisher"],
                    "bio": EXPECTED["nba"]["bio"],
                    "announcement_date": EXPECTED["nba"]["announcement_date"],
                },
                "pulitzer": {
                    "award_name": EXPECTED["pulitzer"]["award_name"],
                    "year": EXPECTED["pulitzer"]["year"],
                    "winner": EXPECTED["pulitzer"]["winner"],
                    "book": EXPECTED["pulitzer"]["book"],
                    "publisher": EXPECTED["pulitzer"]["publisher"],
                    "bio": EXPECTED["pulitzer"]["bio"],
                    "announcement_date": EXPECTED["pulitzer"]["announcement_date"],
                },
                "womens_prize": {
                    "award_name": EXPECTED["womens"]["award_name"],
                    "year": EXPECTED["womens"]["year"],
                    "winner": EXPECTED["womens"]["winner"],
                    "book": EXPECTED["womens"]["book"],
                    "publisher": EXPECTED["womens"]["publisher"],
                    "bio": EXPECTED["womens"]["bio"],
                    "announcement_date": EXPECTED["womens"]["announcement_date"],
                },
            }
        },
        gt_type="ground_truth",
    )

    # Build verification for each item
    await verify_award_package(
        evaluator=evaluator,
        parent_node=root,
        item_id="Item_1_Booker_Prize_for_Fiction_2025",
        item_desc="Booker Prize for Fiction (2025) winner package matches all constraints and includes required URLs.",
        award_key="booker",
        extracted=extracted_awards.booker if extracted_awards else None,
    )

    await verify_award_package(
        evaluator=evaluator,
        parent_node=root,
        item_id="Item_2_National_Book_Award_for_Fiction_2025",
        item_desc="National Book Award for Fiction (2025) winner package matches all constraints and includes required URLs.",
        award_key="nba",
        extracted=extracted_awards.national_book_award if extracted_awards else None,
    )

    await verify_award_package(
        evaluator=evaluator,
        parent_node=root,
        item_id="Item_3_Pulitzer_Prize_for_Fiction_2025",
        item_desc="Pulitzer Prize for Fiction (2025) winner package matches all constraints and includes required URLs.",
        award_key="pulitzer",
        extracted=extracted_awards.pulitzer if extracted_awards else None,
    )

    await verify_award_package(
        evaluator=evaluator,
        parent_node=root,
        item_id="Item_4_Womens_Prize_for_Fiction_2025",
        item_desc="Women's Prize for Fiction (2025) winner package matches all constraints and includes required URLs.",
        award_key="womens",
        extracted=extracted_awards.womens_prize if extracted_awards else None,
    )

    # Return evaluation summary
    return evaluator.get_summary()