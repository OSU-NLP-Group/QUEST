import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "jcim_letters_length_requirements"
TASK_DESCRIPTION = """
What are the manuscript length requirements for submitting a Letter to the Journal of Chemical Information and Modeling (JCIM)? Specifically, provide: (1) the total word limit for the manuscript, (2) the word limit for the abstract, and (3) how graphics (figures, schemes, and tables) are counted toward the word limit.
"""


class JCIMLetterRequirements(BaseModel):
    manuscript_word_limit: Optional[str] = None
    included_components: List[str] = Field(default_factory=list)
    abstract_word_limit: Optional[str] = None
    graphics_counting_description: Optional[str] = None
    official_sources: List[str] = Field(default_factory=list)


def prompt_extract_requirements() -> str:
    return """
    Extract the JCIM Letters manuscript length requirements that the answer provides. Return the following fields:
    1. manuscript_word_limit: The stated overall manuscript length limit for a JCIM Letter, preferably in the form "four journal pages (approximately 3500 words)" or similar wording.
    2. included_components: A list of the components that are explicitly stated to count toward the manuscript length limit. Typical components may include "text", "references", "author names", "graphics", "figures", "schemes", or "tables". Extract only those explicitly mentioned in the answer.
    3. abstract_word_limit: The abstract word limit for JCIM Letters (e.g., "75 words").
    4. graphics_counting_description: How graphics count toward the word limit, e.g., "single-column = 300 words; double-column = 600 words".
    5. official_sources: All URLs cited in the answer that are presented as the official JCIM/ACS author guidelines or official journal instructions pages. Only include explicit URLs present in the answer text (plain or markdown).
    
    If any item is not present in the answer, set that field to null (or empty list for included_components/official_sources).
    """


def list_to_english(items: List[str]) -> str:
    vals = [i.strip() for i in items if i and i.strip()]
    if not vals:
        return ""
    if len(vals) == 1:
        return vals[0]
    return ", ".join(vals[:-1]) + f", and {vals[-1]}"


async def build_and_verify_jcim_requirements(
    evaluator: Evaluator,
    root_node,
    extracted: JCIMLetterRequirements,
) -> None:
    main_node = evaluator.add_parallel(
        id="JCIM_Letters_Requirements",
        desc="Verify that the answer correctly provides JCIM Letters manuscript length requirements and uses the official JCIM author guidelines as the source.",
        parent=root_node,
        critical=True,
    )

    # Word Limit group
    word_group = evaluator.add_parallel(
        id="Word_Limit",
        desc="States the total manuscript length limit for a JCIM Letter (four journal pages, approximately 3500 words) and indicates the included components (text, references, author names, and graphics).",
        parent=main_node,
        critical=True,
    )
    word_exists = evaluator.add_custom_node(
        result=bool(extracted.manuscript_word_limit),
        id="Word_Limit_exists",
        desc="Answer provides an overall JCIM Letters manuscript length limit",
        parent=word_group,
        critical=True,
    )
    inclusions_exist = evaluator.add_custom_node(
        result=bool(extracted.included_components),
        id="Word_Limit_inclusions_exist",
        desc="Answer indicates components that count toward the limit",
        parent=word_group,
        critical=True,
    )
    word_leaf = evaluator.add_leaf(
        id="Word_Limit_value_and_inclusions",
        desc="JCIM Letters overall limit and included components are correct per official guidelines",
        parent=word_group,
        critical=True,
    )
    wl = extracted.manuscript_word_limit or ""
    comps_str = list_to_english(extracted.included_components)
    claim_word = (
        f"For JCIM Letters, the manuscript length limit is '{wl}', and the components that count toward this limit include {comps_str}."
        if comps_str
        else f"For JCIM Letters, the manuscript length limit is '{wl}'."
    )
    await evaluator.verify(
        claim=claim_word,
        node=word_leaf,
        sources=extracted.official_sources,
        additional_instruction=(
            "Verify against the official JCIM (ACS Publications) author guidelines or instructions. "
            "The statement should reflect JCIM Letters length limit (commonly four journal pages, approximately 3500 words) "
            "and that the limit includes items like text, references, author names, and graphics (including figures, schemes, tables). "
            "Allow minor wording variations; focus on correctness and official support."
        ),
    )

    # Abstract Limit group
    abstract_group = evaluator.add_parallel(
        id="Abstract_Limit",
        desc="States the abstract word limit for JCIM Letters (75 words).",
        parent=main_node,
        critical=True,
    )
    abstract_exists = evaluator.add_custom_node(
        result=bool(extracted.abstract_word_limit),
        id="Abstract_Limit_exists",
        desc="Answer provides the abstract word limit for JCIM Letters",
        parent=abstract_group,
        critical=True,
    )
    abstract_leaf = evaluator.add_leaf(
        id="Abstract_Limit_value",
        desc="JCIM Letters abstract word limit is correct per official guidelines",
        parent=abstract_group,
        critical=True,
    )
    al = extracted.abstract_word_limit or ""
    claim_abstract = f"The abstract word limit for JCIM Letters is '{al}'."
    await evaluator.verify(
        claim=claim_abstract,
        node=abstract_leaf,
        sources=extracted.official_sources,
        additional_instruction=(
            "Verify that JCIM Letters abstracts have a 75-word limit per the official JCIM (ACS) author guidelines. "
            "Allow minor formatting differences, but confirm the correct number per official source."
        ),
    )

    # Graphics counting group
    graphics_group = evaluator.add_parallel(
        id="Graphics_Counting",
        desc="Explains how graphics count toward the word limit (single-column = 300 words; double-column = 600 words).",
        parent=main_node,
        critical=True,
    )
    graphics_exists = evaluator.add_custom_node(
        result=bool(extracted.graphics_counting_description),
        id="Graphics_Counting_exists",
        desc="Answer provides graphics counting rules toward the word limit",
        parent=graphics_group,
        critical=True,
    )
    graphics_leaf = evaluator.add_leaf(
        id="Graphics_Counting_rule",
        desc="JCIM graphics word-equivalence rules are correct per official guidelines",
        parent=graphics_group,
        critical=True,
    )
    gc = extracted.graphics_counting_description or ""
    claim_graphics = (
        "In JCIM Letters, a single-column figure/scheme/table counts as 300 words and a double-column counts as 600 words toward the manuscript limit."
        if not gc
        else f"In JCIM Letters, graphics count toward the manuscript limit as follows: {gc}."
    )
    await evaluator.verify(
        claim=claim_graphics,
        node=graphics_leaf,
        sources=extracted.official_sources,
        additional_instruction=(
            "Confirm from the official JCIM (ACS Publications) author guidelines that single-column graphics count as 300 words "
            "and double-column graphics count as 600 words toward the word limit. Graphics include figures, schemes, and tables."
        ),
    )

    # Official source cited group
    source_group = evaluator.add_parallel(
        id="Official_Source_Cited",
        desc="Provides a citation/link or clearly attributes the requirements to the official JCIM author guidelines (not a third-party source).",
        parent=main_node,
        critical=True,
    )
    has_sources = evaluator.add_custom_node(
        result=bool(extracted.official_sources),
        id="Official_Source_exists",
        desc="Answer cites at least one official source URL",
        parent=source_group,
        critical=True,
    )
    source_leaf = evaluator.add_leaf(
        id="Official_Source_is_official",
        desc="Cited source is the official JCIM/ACS author guidelines or official instructions page",
        parent=source_group,
        critical=True,
    )
    claim_official = (
        "This page is the official author guidelines or instructions page for the Journal of Chemical Information and Modeling (JCIM) by ACS Publications."
    )
    await evaluator.verify(
        claim=claim_official,
        node=source_leaf,
        sources=extracted.official_sources,
        additional_instruction=(
            "Verify that at least one cited URL corresponds to the official JCIM author guidelines or official instructions page hosted by ACS Publications "
            "(e.g., on pubs.acs.org) for JCIM. Third-party summaries (blogs, forums, etc.) should not count as official."
        ),
    )


async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=JCIMLetterRequirements,
        extraction_name="jcim_letters_requirements",
    )

    await build_and_verify_jcim_requirements(evaluator, root, extracted)

    return evaluator.get_summary()