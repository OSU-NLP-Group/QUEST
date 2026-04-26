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
TASK_ID = "chef_profile_award_building"
TASK_DESCRIPTION = """
A chef graduated magna cum laude with a bachelor's degree in literature from New York University. This chef is the owner of a restaurant called 610 Magnolia in Louisville, Kentucky, and appeared as a contestant on Top Chef Season 9, finishing in fifth place. The chef authored a book that was published in April 2018 and won the 2019 James Beard Foundation Award for Writing. In May 2023, this chef opened a modern Korean steakhouse in Louisville, Kentucky, located at 835 East Main Street. What is: (1) The chef's full name, (2) The title of the award-winning book, (3) The name of the modern Korean steakhouse opened in May 2023, (4) The name of the building at 835 East Main Street where the restaurant is located, and (5) How many stories does this building have?
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AnswerExtraction(BaseModel):
    """Model for the extracted fields and supporting sources from the answer."""
    # Requested outputs
    chef_name: Optional[str] = None
    book_title: Optional[str] = None
    steakhouse_name: Optional[str] = None
    building_name: Optional[str] = None
    building_story_count: Optional[str] = None

    # Source URLs grouped by aspect
    chef_sources: List[str] = Field(default_factory=list)
    book_sources: List[str] = Field(default_factory=list)
    steakhouse_sources: List[str] = Field(default_factory=list)
    building_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_main_data() -> str:
    return """
    Extract the following five output fields exactly as they appear in the provided answer text, along with grouped source URLs to support each aspect. Return null for any field not mentioned. For URLs, extract only URLs explicitly present in the answer (plain URLs or markdown links). If a URL lacks a protocol, prepend http://. Remove duplicates.

    Output fields to extract:
    1) chef_name: The chef’s full name.
    2) book_title: The full title of the chef’s book that was published in April 2018 and won the 2019 James Beard Foundation Award for Writing.
    3) steakhouse_name: The name of the modern Korean steakhouse opened in May 2023.
    4) building_name: The official or commonly referenced name of the building at 835 East Main Street where the restaurant is located.
    5) building_story_count: The number of stories the building has (return exactly as stated in the answer; do not normalize; it may be a number or a phrase like “five-story”).

    Grouped source URLs:
    - chef_sources: URLs that support the chef’s education (magna cum laude, BA in literature from NYU), ownership of 610 Magnolia (Louisville, KY), and Top Chef Season 9 result (5th place).
    - book_sources: URLs that support the book’s title, its April 2018 publication date, and the 2019 James Beard Foundation Award for Writing.
    - steakhouse_sources: URLs that support the Korean steakhouse’s name, May 2023 opening, and the address at 835 East Main Street in Louisville, KY.
    - building_sources: URLs that describe the building at 835 East Main Street, including its name, mixed-use nature (apartments + retail), and the number of stories.

    Return a JSON object with these fields:
    {
      "chef_name": string|null,
      "book_title": string|null,
      "steakhouse_name": string|null,
      "building_name": string|null,
      "building_story_count": string|null,
      "chef_sources": string[] (may be empty),
      "book_sources": string[] (may be empty),
      "steakhouse_sources": string[] (may be empty),
      "building_sources": string[] (may be empty)
    }
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _has_value(text: Optional[str]) -> bool:
    return bool(text and str(text).strip())


def _combine_sources(*lists: List[str]) -> List[str]:
    combined = []
    for lst in lists:
        if lst:
            combined.extend(lst)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for url in combined:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


# --------------------------------------------------------------------------- #
# Build verification subtrees                                                 #
# --------------------------------------------------------------------------- #
async def build_constraints_checks(
    evaluator: Evaluator,
    parent_node,
    data: AnswerExtraction,
) -> None:
    """
    Build the 'Chef_And_Context_Match_Constraints' parallel critical subtree with leaf verifications.
    """
    constraints_node = evaluator.add_parallel(
        id="Chef_And_Context_Match_Constraints",
        desc="The chef/entity and related book/restaurant/building referenced by the answer satisfy all listed constraints.",
        parent=parent_node,
        critical=True
    )

    # Chef Education: magna cum laude BA in literature from NYU
    node_edu = evaluator.add_leaf(
        id="Chef_Education",
        desc="Chef graduated magna cum laude with a bachelor's degree in literature from New York University.",
        parent=constraints_node,
        critical=True
    )
    claim_edu = (
        f"{data.chef_name or 'The chef'} graduated magna cum laude with a bachelor's degree in literature from New York University (NYU)."
    )
    await evaluator.verify(
        claim=claim_edu,
        node=node_edu,
        sources=data.chef_sources or None,
        additional_instruction=(
            "Confirm the Latin honors 'magna cum laude', the bachelor's degree in literature (or English literature or similar), "
            "and that the institution is New York University (NYU). Minor naming variations are acceptable."
        ),
    )

    # Chef owns 610 Magnolia in Louisville, Kentucky
    node_610 = evaluator.add_leaf(
        id="Chef_Owns_610_Magnolia",
        desc="Chef is the owner of the restaurant 610 Magnolia in Louisville, Kentucky.",
        parent=constraints_node,
        critical=True
    )
    claim_610 = (
        f"{data.chef_name or 'The chef'} is the owner (or chef-owner/co-owner) of 610 Magnolia in Louisville, Kentucky."
    )
    await evaluator.verify(
        claim=claim_610,
        node=node_610,
        sources=data.chef_sources or None,
        additional_instruction="Check that the chef is listed as owner/chef-owner of 610 Magnolia in Louisville, KY.",
    )

    # Top Chef Season 9 finished in fifth place
    node_topchef = evaluator.add_leaf(
        id="Top_Chef_S9_Result",
        desc="Chef appeared on Top Chef Season 9 and finished in fifth place.",
        parent=constraints_node,
        critical=True
    )
    claim_topchef = (
        f"{data.chef_name or 'The chef'} appeared as a contestant on Top Chef Season 9 and finished in fifth place."
    )
    await evaluator.verify(
        claim=claim_topchef,
        node=node_topchef,
        sources=data.chef_sources or None,
        additional_instruction="Verify Season 9 participation and that the chef placed fifth.",
    )

    # Book published in April 2018
    node_book_pub = evaluator.add_leaf(
        id="Book_Published_April_2018",
        desc="Chef authored a book that was published in April 2018.",
        parent=constraints_node,
        critical=True
    )
    claim_book_pub = (
        f"The book '{data.book_title or 'the book'}' authored by {data.chef_name or 'the chef'} was published in April 2018."
    )
    await evaluator.verify(
        claim=claim_book_pub,
        node=node_book_pub,
        sources=data.book_sources or None,
        additional_instruction="Confirm the publication month is April and the year is 2018 (day can vary).",
    )

    # Book won 2019 James Beard Foundation Award for Writing
    node_book_award = evaluator.add_leaf(
        id="Book_Won_2019_JBF_Writing_Award",
        desc="That book won the 2019 James Beard Foundation Award for Writing.",
        parent=constraints_node,
        critical=True
    )
    claim_book_award = (
        f"The book '{data.book_title or 'the book'}' won the 2019 James Beard Foundation Award for Writing."
    )
    await evaluator.verify(
        claim=claim_book_award,
        node=node_book_award,
        sources=data.book_sources or None,
        additional_instruction=(
            "Confirm the James Beard Foundation (JBF) Media Award year is 2019 and the category is Writing "
            "(minor phrasing variations acceptable)."
        ),
    )

    # Opened modern Korean steakhouse in May 2023
    node_open_steakhouse = evaluator.add_leaf(
        id="Opened_Modern_Korean_Steakhouse_May_2023",
        desc="Chef opened a modern Korean steakhouse in May 2023.",
        parent=constraints_node,
        critical=True
    )
    claim_open_steakhouse = (
        f"{data.chef_name or 'The chef'} opened a modern Korean steakhouse named '{data.steakhouse_name or 'the restaurant'}' in May 2023."
    )
    await evaluator.verify(
        claim=claim_open_steakhouse,
        node=node_open_steakhouse,
        sources=data.steakhouse_sources or None,
        additional_instruction="Verify that the opening date is May 2023 and the concept is a modern Korean steakhouse.",
    )

    # Steakhouse address: 835 East Main Street in Louisville, Kentucky
    node_steakhouse_addr = evaluator.add_leaf(
        id="Steakhouse_Address_835_E_Main_St",
        desc="The modern Korean steakhouse is located at 835 East Main Street in Louisville, Kentucky.",
        parent=constraints_node,
        critical=True
    )
    claim_steakhouse_addr = (
        f"The restaurant '{data.steakhouse_name or 'the restaurant'}' is located at 835 East Main Street, Louisville, Kentucky."
    )
    await evaluator.verify(
        claim=claim_steakhouse_addr,
        node=node_steakhouse_addr,
        sources=_combine_sources(data.steakhouse_sources, data.building_sources) or None,
        additional_instruction="Confirm the street address and city/state for the restaurant.",
    )

    # Building mixed-use with apartments and retail, multi-story
    node_building_mixed = evaluator.add_leaf(
        id="Building_Mixed_Use_Apts_And_Retail",
        desc="The building at 835 East Main Street is a multi-story mixed-use structure with apartments and retail space.",
        parent=constraints_node,
        critical=True
    )
    claim_building_mixed = (
        "The building at 835 East Main Street is multi-story and mixed-use, containing apartments and retail space."
    )
    await evaluator.verify(
        claim=claim_building_mixed,
        node=node_building_mixed,
        sources=data.building_sources or None,
        additional_instruction="Verify the building description mentions both apartments and retail, and indicates multiple stories.",
    )


async def build_requested_outputs_checks(
    evaluator: Evaluator,
    parent_node,
    data: AnswerExtraction,
) -> None:
    """
    Build the 'Requested_Outputs_Present' parallel critical subtree with presence checks.
    """
    outputs_node = evaluator.add_parallel(
        id="Requested_Outputs_Present",
        desc="The response provides each requested output field.",
        parent=parent_node,
        critical=True
    )

    # Presence checks (critical) for each requested field
    evaluator.add_custom_node(
        result=_has_value(data.chef_name),
        id="Provide_Chef_Full_Name",
        desc="Provides the chef's full name.",
        parent=outputs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_value(data.book_title),
        id="Provide_Award_Winning_Book_Title",
        desc="Provides the title of the chef's book that won the 2019 James Beard Foundation Award for Writing.",
        parent=outputs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_value(data.steakhouse_name),
        id="Provide_2023_Steakhouse_Name",
        desc="Provides the name of the modern Korean steakhouse opened in May 2023.",
        parent=outputs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_value(data.building_name),
        id="Provide_Building_Name",
        desc="Provides the name of the building at 835 East Main Street where the restaurant is located.",
        parent=outputs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_value(data.building_story_count),
        id="Provide_Building_Story_Count",
        desc="Provides how many stories the building has.",
        parent=outputs_node,
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
    Evaluate an answer for the chef profile, book, and building details task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Root-level sequential to reflect overall task flow
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

    # Create a critical sequential node representing "Complete_Answer"
    complete_answer_node = evaluator.add_sequential(
        id="Complete_Answer",
        desc="Answer identifies the correct chef/profile and provides all requested fields (chef name, book title, 2023 steakhouse name, building name, and building story count) consistent with the given constraints.",
        parent=root,
        critical=True
    )

    # Extract all required fields and sources
    extracted = await evaluator.extract(
        prompt=prompt_extract_main_data(),
        template_class=AnswerExtraction,
        extraction_name="extracted_main_data"
    )

    # Build constraints subtree (critical parallel)
    await build_constraints_checks(evaluator, complete_answer_node, extracted)

    # Build requested outputs presence subtree (critical parallel)
    await build_requested_outputs_checks(evaluator, complete_answer_node, extracted)

    # Optionally record some custom info for debugging (counts of URLs)
    evaluator.add_custom_info(
        info={
            "chef_sources_count": len(extracted.chef_sources or []),
            "book_sources_count": len(extracted.book_sources or []),
            "steakhouse_sources_count": len(extracted.steakhouse_sources or []),
            "building_sources_count": len(extracted.building_sources or [])
        },
        info_type="source_counts",
        info_name="source_counts_summary"
    )

    # Return structured result using the evaluator's summary
    return evaluator.get_summary()