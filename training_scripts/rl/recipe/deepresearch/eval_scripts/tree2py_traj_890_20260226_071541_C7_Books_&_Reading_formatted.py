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
TASK_ID = "major_fiction_awards_2025"
TASK_DESCRIPTION = (
    "You are organizing a major literary event celebrating the best fiction of 2025 and need to compile a comprehensive "
    "information sheet about the three most prestigious English-language fiction awards: the National Book Award for "
    "Fiction, the Pulitzer Prize for Fiction, and the Booker Prize. For each of these three awards, identify the 2025 "
    "winner and provide the following information: (1) The complete book title, (2) The author's full name, (3) The "
    "publisher, and (4) The author's current primary residence location (specify city and state/country). Present your "
    "findings in a structured format with clear attribution to each award category."
)

# Display names for award categories
AWARD_DISPLAY_NAMES = {
    "nba": "National Book Award for Fiction",
    "pulitzer": "Pulitzer Prize for Fiction",
    "booker": "Booker Prize",
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AwardDetails(BaseModel):
    """Structured information for one award winner."""
    book_title: Optional[str] = None
    author_name: Optional[str] = None
    publisher: Optional[str] = None
    author_residence_city: Optional[str] = None
    author_residence_region: Optional[str] = None  # state (US) or country (non-US)
    sources: List[str] = Field(default_factory=list)


class AwardsExtraction(BaseModel):
    """Top-level extraction for three awards."""
    nba: Optional[AwardDetails] = None
    pulitzer: Optional[AwardDetails] = None
    booker: Optional[AwardDetails] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_awards_info() -> str:
    return (
        "Extract structured information for the three 2025 fiction award winners mentioned in the answer. For each award, "
        "return the following fields exactly as presented in the answer text:\n"
        "— book_title: The complete title of the winning book (string)\n"
        "— author_name: The full name of the winning author (string)\n"
        "— publisher: The publisher of the winning book (string)\n"
        "— author_residence_city: The author's current primary residence city (string)\n"
        "— author_residence_region: The matching state (if US) or country (if non-US) (string)\n"
        "— sources: An array of all URLs explicitly cited in the answer that support any of the above facts for this award. "
        "Include official award pages or reputable news/publisher bios if provided. Extract only URLs explicitly present in the answer.\n\n"
        "Organize the JSON as:\n"
        "{\n"
        '  "nba": { ... },            // National Book Award for Fiction (2025)\n'
        '  "pulitzer": { ... },       // Pulitzer Prize for Fiction (2025)\n'
        '  "booker": { ... }          // Booker Prize (2025)\n'
        "}\n\n"
        "Rules:\n"
        "1) Do not invent or infer any values; use only what is explicitly given in the answer.\n"
        "2) If a specific field is missing in the answer, set it to null.\n"
        "3) For 'sources', include every URL explicitly cited in the answer for that award; if none are cited, return an empty list.\n"
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def format_location(city: Optional[str], region: Optional[str]) -> Optional[str]:
    """Combine city and region into 'City, Region' if both exist."""
    city_val = (city or "").strip()
    region_val = (region or "").strip()
    if city_val and region_val:
        return f"{city_val}, {region_val}"
    return None


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def _verify_award_field_group(
    evaluator: Evaluator,
    parent_node,
    group_id: str,
    group_desc: str,
    existence_condition: bool,
    verification_leaf_id: str,
    verification_leaf_desc: str,
    claim: str,
    sources: List[str],
    additional_instruction: str,
) -> None:
    """
    Create a Sequential field group:
      1) Critical existence check (custom node)
      2) Critical source-based verification leaf
    """
    # Group node: sequential, non-critical (allows partial credit within award)
    group_node = evaluator.add_sequential(
        id=group_id,
        desc=group_desc,
        parent=parent_node,
        critical=False,
    )

    # Existence + source availability gate
    evaluator.add_custom_node(
        result=existence_condition and bool(sources),
        id=f"{group_id}_exists",
        desc=f"{group_desc} - data and sources present",
        parent=group_node,
        critical=True,
    )

    # Verification leaf (critical under group)
    leaf = evaluator.add_leaf(
        id=verification_leaf_id,
        desc=verification_leaf_desc,
        parent=group_node,
        critical=True,
    )

    # Verify against provided sources
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=additional_instruction,
    )


async def verify_award(
    evaluator: Evaluator,
    parent_node,
    award_key: str,
    award_node_id: str,
    award_node_desc: str,
    details: Optional[AwardDetails],
    leaf_ids: Dict[str, str],
    leaf_descs: Dict[str, str],
) -> None:
    """
    Build the verification subtree for one award.
    """
    # Award node (parallel, non-critical to allow partial credit across awards)
    award_node = evaluator.add_parallel(
        id=award_node_id,
        desc=award_node_desc,
        parent=parent_node,
        critical=False,
    )

    # If no details extracted, create groups that will fail at existence gate
    info = details or AwardDetails()

    display_award_name = AWARD_DISPLAY_NAMES[award_key]
    sources = info.sources or []

    # 1) Book Title
    book_title = (info.book_title or "").strip()
    await _verify_award_field_group(
        evaluator=evaluator,
        parent_node=award_node,
        group_id=f"{award_key}_book_title_group",
        group_desc=f"{display_award_name} - Book title",
        existence_condition=bool(book_title),
        verification_leaf_id=leaf_ids["title"],
        verification_leaf_desc=leaf_descs["title"],
        claim=f"The book that won the 2025 {display_award_name} is titled '{book_title}'.",
        sources=sources,
        additional_instruction=(
            "Verify that the cited page(s) explicitly indicate the 2025 winner for this award and that the "
            "book title matches (allow minor punctuation/casing variants). Prefer official award announcement pages "
            "or reputable outlets. If multiple pages are provided, any one page is sufficient if it clearly supports the claim."
        ),
    )

    # 2) Author Name
    author_name = (info.author_name or "").strip()
    await _verify_award_field_group(
        evaluator=evaluator,
        parent_node=award_node,
        group_id=f"{award_key}_author_name_group",
        group_desc=f"{display_award_name} - Author name",
        existence_condition=bool(author_name),
        verification_leaf_id=leaf_ids["author"],
        verification_leaf_desc=leaf_descs["author"],
        claim=(
            f"The author of '{book_title}', the 2025 {display_award_name} winner, is '{author_name}'."
            if book_title
            else f"The author who won the 2025 {display_award_name} for Fiction is '{author_name}'."
        ),
        sources=sources,
        additional_instruction=(
            "Check that the cited page(s) associate the winning book and award year with the specified author. "
            "Allow reasonable variants (middle initials/names, diacritics, casing)."
        ),
    )

    # 3) Publisher
    publisher = (info.publisher or "").strip()
    await _verify_award_field_group(
        evaluator=evaluator,
        parent_node=award_node,
        group_id=f"{award_key}_publisher_group",
        group_desc=f"{display_award_name} - Publisher",
        existence_condition=bool(publisher),
        verification_leaf_id=leaf_ids["publisher"],
        verification_leaf_desc=leaf_descs["publisher"],
        claim=(
            f"The publisher of '{book_title}', the 2025 {display_award_name} winner, is '{publisher}'."
            if book_title
            else f"The publisher for the 2025 {display_award_name} winner is '{publisher}'."
        ),
        sources=sources,
        additional_instruction=(
            "Confirm the book's publisher on the cited page(s). Accept imprints or divisions when clearly identified as the publishing entity. "
            "If multiple publisher names appear (e.g., imprint vs. parent), the imprint listed on the book's page is acceptable."
        ),
    )

    # 4) Author Residence (city + state/country)
    residence_str = format_location(info.author_residence_city, info.author_residence_region) or ""
    await _verify_award_field_group(
        evaluator=evaluator,
        parent_node=award_node,
        group_id=f"{award_key}_residence_group",
        group_desc=f"{display_award_name} - Author residence",
        existence_condition=bool(residence_str),
        verification_leaf_id=leaf_ids["residence"],
        verification_leaf_desc=leaf_descs["residence"],
        claim=(
            f"The author's current primary residence is {residence_str}."
            if author_name == ""
            else f"The author {author_name}'s current primary residence is {residence_str}."
        ),
        sources=sources,
        additional_instruction=(
            "Verify that the cited page(s) explicitly indicate the author's current location (e.g., 'based in', 'lives in'). "
            "Allow reasonable naming variants (e.g., 'NYC' vs 'New York, NY'). Prefer recent/official bios or the award announcement."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the '2025 Major Fiction Awards' task.
    """
    # Initialize evaluator with a parallel root (we'll add a task-specific parent node under it)
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

    # Add task-specific top-level node
    main_node = evaluator.add_parallel(
        id="2025_Major_Fiction_Awards",
        desc=(
            "Evaluate whether information about the three major 2025 fiction award winners "
            "(National Book Award, Pulitzer Prize, Booker Prize) has been correctly identified and documented"
        ),
        parent=root,
        critical=False,
    )

    # Extract award information
    extracted_awards = await evaluator.extract(
        prompt=prompt_extract_awards_info(),
        template_class=AwardsExtraction,
        extraction_name="awards_2025_info",
    )

    # Build verification subtrees for each award
    # National Book Award for Fiction
    await verify_award(
        evaluator=evaluator,
        parent_node=main_node,
        award_key="nba",
        award_node_id="National_Book_Award_Fiction_2025",
        award_node_desc="Information about the 2025 National Book Award Fiction winner",
        details=extracted_awards.nba,
        leaf_ids={
            "title": "NBA_Book_Title",
            "author": "NBA_Author_Name",
            "publisher": "NBA_Publisher",
            "residence": "NBA_Author_Residence",
        },
        leaf_descs={
            "title": "The complete title of the book that won the 2025 National Book Award for Fiction is provided",
            "author": "The full name of the author who won the 2025 National Book Award for Fiction is provided",
            "publisher": "The publisher of the 2025 National Book Award Fiction winner is provided",
            "residence": "The author's current primary residence location (city and state/country) is provided",
        },
    )

    # Pulitzer Prize for Fiction
    await verify_award(
        evaluator=evaluator,
        parent_node=main_node,
        award_key="pulitzer",
        award_node_id="Pulitzer_Prize_Fiction_2025",
        award_node_desc="Information about the 2025 Pulitzer Prize Fiction winner",
        details=extracted_awards.pulitzer,
        leaf_ids={
            "title": "Pulitzer_Book_Title",
            "author": "Pulitzer_Author_Name",
            "publisher": "Pulitzer_Publisher",
            "residence": "Pulitzer_Author_Residence",
        },
        leaf_descs={
            "title": "The complete title of the book that won the 2025 Pulitzer Prize for Fiction is provided",
            "author": "The full name of the author who won the 2025 Pulitzer Prize for Fiction is provided",
            "publisher": "The publisher of the 2025 Pulitzer Prize Fiction winner is provided",
            "residence": "The author's current primary residence location (city and state/country) is provided",
        },
    )

    # Booker Prize
    await verify_award(
        evaluator=evaluator,
        parent_node=main_node,
        award_key="booker",
        award_node_id="Booker_Prize_2025",
        award_node_desc="Information about the 2025 Booker Prize winner",
        details=extracted_awards.booker,
        leaf_ids={
            "title": "Booker_Book_Title",
            "author": "Booker_Author_Name",
            "publisher": "Booker_Publisher",
            "residence": "Booker_Author_Residence",
        },
        leaf_descs={
            "title": "The complete title of the book that won the 2025 Booker Prize is provided",
            "author": "The full name of the author who won the 2025 Booker Prize is provided",
            "publisher": "The publisher of the 2025 Booker Prize winner is provided",
            "residence": "The author's current primary residence location (city and state/country) is provided",
        },
    )

    # Return the summary
    return evaluator.get_summary()