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
TASK_ID = "three_major_awards_2025"
TASK_DESCRIPTION = (
    "Identify the books that won the following three major literary awards in 2025: "
    "(1) the Booker Prize, (2) the National Book Award for Fiction, and (3) the Pulitzer Prize for Fiction. "
    "For each winning book, provide: the complete title, the author's full name, the publisher, the date when the award was announced, "
    "and the location or venue where the award ceremony took place."
)

# Ground-truth constraints expected by the rubric
EXPECTED = {
    "booker": {
        "title": "Flesh",
        "author": "David Szalay",
        "publisher": "Jonathan Cape",
        "date": "November 10, 2025",
        "venue_exact": "Old Billingsgate in London",
    },
    "nba_fiction": {
        "title": "The True True Story of Raja the Gullible (and His Mother)",
        "author": "Rabih Alameddine",
        "publisher": "Grove Press / Grove Atlantic",
        "date": "November 20, 2025",
        # Only need presence for venue
    },
    "pulitzer_fiction": {
        "title": "James",
        "author": "Percival Everett",
        "publisher": "Doubleday",
        "date": "May 5, 2025",
        # Only need presence for venue
    },
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AwardInfo(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    announcement_date: Optional[str] = None
    ceremony_location: Optional[str] = None
    general_sources: List[str] = Field(default_factory=list)
    date_sources: List[str] = Field(default_factory=list)
    venue_sources: List[str] = Field(default_factory=list)


class AwardsExtraction(BaseModel):
    booker: Optional[AwardInfo] = None
    national_book_award_fiction: Optional[AwardInfo] = None
    pulitzer_prize_fiction: Optional[AwardInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_awards() -> str:
    return """
Extract the specific details for the three awards mentioned in the answer for the year 2025. Map the information into the following JSON schema:

{
  "booker": {
    "title": string or null,
    "author": string or null,
    "publisher": string or null,
    "announcement_date": string or null,
    "ceremony_location": string or null,
    "general_sources": string[] (URLs explicitly present in the answer),
    "date_sources": string[] (URLs that the answer explicitly ties to the announcement date, if any),
    "venue_sources": string[] (URLs that the answer explicitly ties to the ceremony location/venue, if any)
  },
  "national_book_award_fiction": {
    "title": string or null,
    "author": string or null,
    "publisher": string or null,
    "announcement_date": string or null,
    "ceremony_location": string or null,
    "general_sources": string[],
    "date_sources": string[],
    "venue_sources": string[]
  },
  "pulitzer_prize_fiction": {
    "title": string or null,
    "author": string or null,
    "publisher": string or null,
    "announcement_date": string or null,
    "ceremony_location": string or null,
    "general_sources": string[],
    "date_sources": string[],
    "venue_sources": string[]
  }
}

Guidelines:
- Extract only from the provided answer text (do not invent).
- Interpret "Booker Prize" as the primary Booker Prize (fiction/novel) award for 2025.
- Interpret "National Book Award for Fiction" as the fiction category.
- Interpret "Pulitzer Prize for Fiction" as that specific prize.
- If the answer lists sources or references in plain text or markdown links, collect the actual URLs into the arrays.
- If the answer ties a specific URL to the date or venue, place it into the corresponding 'date_sources' or 'venue_sources' array; otherwise, place URLs in 'general_sources'.
- If any field is missing, set it to null (or empty arrays for URL lists).
"""


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _pick_date_sources(info: Optional[AwardInfo]) -> List[str]:
    if not info:
        return []
    if info.date_sources:
        return list(dict.fromkeys(info.date_sources))
    if info.general_sources:
        return list(dict.fromkeys(info.general_sources))
    return []


def _strict_exact_instruction(field_label: str) -> str:
    return (
        f"Check ONLY the answer text. Consider the claim correct only if the {field_label} exactly matches the given string, "
        "including letter case, punctuation, and spacing. Do not allow minor variants or alternate spellings."
    )


def _authoritative_date_instruction(award_label: str, expected_date: str) -> str:
    return (
        f"Verify that at least one provided URL explicitly states that the {award_label} was announced on {expected_date}. "
        "Prefer official award sites (e.g., thebookerprizes.com, nationalbook.org, pulitzer.org) or reputable press outlets. "
        "If URLs are missing, irrelevant, inaccessible, or do not explicitly show that date in the relevant award context, "
        "judge this as NOT supported."
    )


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_booker(evaluator: Evaluator, parent_node, extracted: AwardsExtraction) -> None:
    info = extracted.booker if extracted else None
    award_node = evaluator.add_parallel(
        id="Booker_Prize_2025",
        desc="2025 Booker Prize winning book details must be provided.",
        parent=parent_node,
        critical=True,
    )

    # Title exact
    leaf_title = evaluator.add_leaf(
        id="Booker_Title_Matches_Constraint",
        desc="Book title is exactly 'Flesh' (matching the constraints and the exact-official-wording requirement).",
        parent=award_node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the 2025 Booker Prize winning book title is exactly 'Flesh'.",
        node=leaf_title,
        additional_instruction=_strict_exact_instruction("title"),
    )

    # Author exact
    leaf_author = evaluator.add_leaf(
        id="Booker_Author_Matches_Constraint",
        desc="Author full name is 'David Szalay'.",
        parent=award_node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the 2025 Booker Prize winner's author is exactly 'David Szalay'.",
        node=leaf_author,
        additional_instruction=_strict_exact_instruction("author name"),
    )

    # Publisher exact
    leaf_publisher = evaluator.add_leaf(
        id="Booker_Publisher_Matches_Constraint",
        desc="Publisher is 'Jonathan Cape'.",
        parent=award_node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the publisher for the 2025 Booker Prize winning book is exactly 'Jonathan Cape'.",
        node=leaf_publisher,
        additional_instruction=_strict_exact_instruction("publisher"),
    )

    # Date verification (sequential: equality -> source provided -> supported by source)
    date_seq = evaluator.add_sequential(
        id="Booker_Announcement_Date_Accurate_And_Verifiable",
        desc="Award announcement date is 'November 10, 2025' AND an authoritative source reference is provided supporting this date.",
        parent=award_node,
        critical=True,
    )

    leaf_date_exact = evaluator.add_leaf(
        id="Booker_Announcement_Date_Exact",
        desc="The answer lists the Booker announcement date exactly as 'November 10, 2025'.",
        parent=date_seq,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the 2025 Booker Prize announcement date is exactly 'November 10, 2025'.",
        node=leaf_date_exact,
        additional_instruction=_strict_exact_instruction("announcement date"),
    )

    date_sources = _pick_date_sources(info)
    leaf_date_src_provided = evaluator.add_custom_node(
        result=bool(date_sources),
        id="Booker_Date_Source_Provided",
        desc="At least one URL source is provided in the answer to support the Booker announcement date.",
        parent=date_seq,
        critical=True,
    )

    leaf_date_supported = evaluator.add_leaf(
        id="Booker_Announcement_Date_Supported_By_Sources",
        desc="The provided sources support that the Booker Prize 2025 was announced on November 10, 2025.",
        parent=date_seq,
        critical=True,
    )
    await evaluator.verify(
        claim="The 2025 Booker Prize winner was announced on November 10, 2025.",
        node=leaf_date_supported,
        sources=date_sources,
        additional_instruction=_authoritative_date_instruction("2025 Booker Prize", EXPECTED["booker"]["date"]),
    )

    # Venue exact
    leaf_venue = evaluator.add_leaf(
        id="Booker_Ceremony_Location_Matches_Constraint",
        desc="Ceremony location/venue is 'Old Billingsgate in London'.",
        parent=award_node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the 2025 Booker Prize ceremony location/venue is exactly 'Old Billingsgate in London'.",
        node=leaf_venue,
        additional_instruction=_strict_exact_instruction("ceremony location/venue"),
    )


async def verify_nba(evaluator: Evaluator, parent_node, extracted: AwardsExtraction) -> None:
    info = extracted.national_book_award_fiction if extracted else None
    award_node = evaluator.add_parallel(
        id="National_Book_Award_Fiction_2025",
        desc="2025 National Book Award for Fiction winning book details must be provided.",
        parent=parent_node,
        critical=True,
    )

    # Title exact
    leaf_title = evaluator.add_leaf(
        id="NBA_Title_Matches_Constraint",
        desc="Book title is exactly 'The True True Story of Raja the Gullible (and His Mother)'.",
        parent=award_node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the 2025 National Book Award for Fiction winning book title is exactly 'The True True Story of Raja the Gullible (and His Mother)'.",
        node=leaf_title,
        additional_instruction=_strict_exact_instruction("title"),
    )

    # Author exact
    leaf_author = evaluator.add_leaf(
        id="NBA_Author_Matches_Constraint",
        desc="Author full name is 'Rabih Alameddine'.",
        parent=award_node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the 2025 National Book Award for Fiction winner's author is exactly 'Rabih Alameddine'.",
        node=leaf_author,
        additional_instruction=_strict_exact_instruction("author name"),
    )

    # Publisher exact
    leaf_publisher = evaluator.add_leaf(
        id="NBA_Publisher_Matches_Constraint",
        desc="Publisher is 'Grove Press / Grove Atlantic'.",
        parent=award_node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the publisher for the 2025 National Book Award for Fiction winning book is exactly 'Grove Press / Grove Atlantic'.",
        node=leaf_publisher,
        additional_instruction=_strict_exact_instruction("publisher"),
    )

    # Date verification
    date_seq = evaluator.add_sequential(
        id="NBA_Announcement_Date_Accurate_And_Verifiable",
        desc="Award announcement date is 'November 20, 2025' AND an authoritative source reference is provided supporting this date.",
        parent=award_node,
        critical=True,
    )

    leaf_date_exact = evaluator.add_leaf(
        id="NBA_Announcement_Date_Exact",
        desc="The answer lists the NBA (Fiction) announcement date exactly as 'November 20, 2025'.",
        parent=date_seq,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the 2025 National Book Award for Fiction announcement date is exactly 'November 20, 2025'.",
        node=leaf_date_exact,
        additional_instruction=_strict_exact_instruction("announcement date"),
    )

    date_sources = _pick_date_sources(info)
    leaf_date_src_provided = evaluator.add_custom_node(
        result=bool(date_sources),
        id="NBA_Date_Source_Provided",
        desc="At least one URL source is provided in the answer to support the NBA (Fiction) announcement date.",
        parent=date_seq,
        critical=True,
    )

    leaf_date_supported = evaluator.add_leaf(
        id="NBA_Announcement_Date_Supported_By_Sources",
        desc="The provided sources support that the National Book Award for Fiction 2025 was announced on November 20, 2025.",
        parent=date_seq,
        critical=True,
    )
    await evaluator.verify(
        claim="The 2025 National Book Award for Fiction winner was announced on November 20, 2025.",
        node=leaf_date_supported,
        sources=date_sources,
        additional_instruction=_authoritative_date_instruction("2025 National Book Award for Fiction", EXPECTED["nba_fiction"]["date"]),
    )

    # Venue provided (existence only)
    leaf_venue_provided = evaluator.add_custom_node(
        result=bool(info and info.ceremony_location and info.ceremony_location.strip()),
        id="NBA_Ceremony_Location_Provided",
        desc="A ceremony location or venue is provided for the National Book Award for Fiction winner (not omitted).",
        parent=award_node,
        critical=True,
    )


async def verify_pulitzer(evaluator: Evaluator, parent_node, extracted: AwardsExtraction) -> None:
    info = extracted.pulitzer_prize_fiction if extracted else None
    award_node = evaluator.add_parallel(
        id="Pulitzer_Prize_Fiction_2025",
        desc="2025 Pulitzer Prize for Fiction winning book details must be provided.",
        parent=parent_node,
        critical=True,
    )

    # Title exact
    leaf_title = evaluator.add_leaf(
        id="Pulitzer_Title_Matches_Constraint",
        desc="Book title is exactly 'James'.",
        parent=award_node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the 2025 Pulitzer Prize for Fiction winning book title is exactly 'James'.",
        node=leaf_title,
        additional_instruction=_strict_exact_instruction("title"),
    )

    # Author exact
    leaf_author = evaluator.add_leaf(
        id="Pulitzer_Author_Matches_Constraint",
        desc="Author full name is 'Percival Everett'.",
        parent=award_node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the 2025 Pulitzer Prize for Fiction winner's author is exactly 'Percival Everett'.",
        node=leaf_author,
        additional_instruction=_strict_exact_instruction("author name"),
    )

    # Publisher exact
    leaf_publisher = evaluator.add_leaf(
        id="Pulitzer_Publisher_Matches_Constraint",
        desc="Publisher is 'Doubleday'.",
        parent=award_node,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the publisher for the 2025 Pulitzer Prize for Fiction winning book is exactly 'Doubleday'.",
        node=leaf_publisher,
        additional_instruction=_strict_exact_instruction("publisher"),
    )

    # Date verification
    date_seq = evaluator.add_sequential(
        id="Pulitzer_Announcement_Date_Accurate_And_Verifiable",
        desc="Award announcement date is 'May 5, 2025' AND an authoritative source reference is provided supporting this date.",
        parent=award_node,
        critical=True,
    )

    leaf_date_exact = evaluator.add_leaf(
        id="Pulitzer_Announcement_Date_Exact",
        desc="The answer lists the Pulitzer (Fiction) announcement date exactly as 'May 5, 2025'.",
        parent=date_seq,
        critical=True,
    )
    await evaluator.verify(
        claim="In the answer, the 2025 Pulitzer Prize for Fiction announcement date is exactly 'May 5, 2025'.",
        node=leaf_date_exact,
        additional_instruction=_strict_exact_instruction("announcement date"),
    )

    date_sources = _pick_date_sources(info)
    leaf_date_src_provided = evaluator.add_custom_node(
        result=bool(date_sources),
        id="Pulitzer_Date_Source_Provided",
        desc="At least one URL source is provided in the answer to support the Pulitzer (Fiction) announcement date.",
        parent=date_seq,
        critical=True,
    )

    leaf_date_supported = evaluator.add_leaf(
        id="Pulitzer_Announcement_Date_Supported_By_Sources",
        desc="The provided sources support that the Pulitzer Prize for Fiction 2025 was announced on May 5, 2025.",
        parent=date_seq,
        critical=True,
    )
    await evaluator.verify(
        claim="The 2025 Pulitzer Prize for Fiction winner was announced on May 5, 2025.",
        node=leaf_date_supported,
        sources=date_sources,
        additional_instruction=_authoritative_date_instruction("2025 Pulitzer Prize for Fiction", EXPECTED["pulitzer_fiction"]["date"]),
    )

    # Venue provided (existence only)
    leaf_venue_provided = evaluator.add_custom_node(
        result=bool(info and info.ceremony_location and info.ceremony_location.strip()),
        id="Pulitzer_Ceremony_Location_Provided",
        desc="A ceremony location or venue is provided for the Pulitzer Prize for Fiction winner (not omitted).",
        parent=award_node,
        critical=True,
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

    # Extract structured award info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_awards(),
        template_class=AwardsExtraction,
        extraction_name="awards_extraction",
    )

    # Ground truth info for transparency
    evaluator.add_ground_truth({
        "expected": {
            "booker": {
                "title": EXPECTED["booker"]["title"],
                "author": EXPECTED["booker"]["author"],
                "publisher": EXPECTED["booker"]["publisher"],
                "announcement_date": EXPECTED["booker"]["date"],
                "venue": EXPECTED["booker"]["venue_exact"],
            },
            "national_book_award_fiction": {
                "title": EXPECTED["nba_fiction"]["title"],
                "author": EXPECTED["nba_fiction"]["author"],
                "publisher": EXPECTED["nba_fiction"]["publisher"],
                "announcement_date": EXPECTED["nba_fiction"]["date"],
                "venue": "provided (existence required)",
            },
            "pulitzer_prize_fiction": {
                "title": EXPECTED["pulitzer_fiction"]["title"],
                "author": EXPECTED["pulitzer_fiction"]["author"],
                "publisher": EXPECTED["pulitzer_fiction"]["publisher"],
                "announcement_date": EXPECTED["pulitzer_fiction"]["date"],
                "venue": "provided (existence required)",
            }
        }
    })

    # Create a critical top-level node (since evaluator's root is non-critical by design)
    awards_main = evaluator.add_parallel(
        id="Three_Major_2025_Literary_Award_Winners",
        desc="Identify the books that won the 2025 Booker Prize, National Book Award for Fiction, and Pulitzer Prize for Fiction; for each, provide title, author full name, publisher, announcement date (verifiable from authoritative sources), and ceremony location/venue.",
        parent=root,
        critical=True,
    )

    # Verify each award subtree (all critical under the critical parent)
    await verify_booker(evaluator, awards_main, extracted)
    await verify_nba(evaluator, awards_main, extracted)
    await verify_pulitzer(evaluator, awards_main, extracted)

    return evaluator.get_summary()