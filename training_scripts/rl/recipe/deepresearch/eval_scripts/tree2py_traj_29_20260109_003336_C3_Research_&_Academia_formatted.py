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
TASK_ID = "cornell_dissertation_requirements"
TASK_DESCRIPTION = (
    "What are the complete doctoral dissertation requirements at Cornell University's Graduate School, "
    "specifically including: (1) the minimum number of committee members required and whether at least one member must be from outside the student's department, "
    "(2) the minimum advance time required to distribute the dissertation to committee members before the defense and the maximum time allowed after the defense to submit the final dissertation, "
    "(3) the margin specifications and line spacing requirements for the document, and (4) the maximum word count allowed for the dissertation abstract?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RequirementsExtraction(BaseModel):
    # Committee
    min_committee_members: Optional[str] = None
    min_committee_members_sources: List[str] = Field(default_factory=list)
    outside_department_requirement: Optional[str] = None
    outside_department_sources: List[str] = Field(default_factory=list)

    # Defense timeline
    pre_defense_distribution: Optional[str] = None
    pre_defense_sources: List[str] = Field(default_factory=list)
    post_defense_submission: Optional[str] = None
    post_defense_sources: List[str] = Field(default_factory=list)

    # Formatting
    margin_specifications: Optional[str] = None
    margins_sources: List[str] = Field(default_factory=list)
    line_spacing_specifications: Optional[str] = None
    line_spacing_sources: List[str] = Field(default_factory=list)

    # Abstract
    abstract_max_word_count: Optional[str] = None
    abstract_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
Extract from the answer the specific statements (as text) and the cited URL sources for each of the following dissertation requirement items at Cornell University's Graduate School.

Important:
- Extract the exact phrasing the answer uses for each item (do not normalize, paraphrase, or infer beyond the text).
- For sources, extract only actual URLs explicitly present in the answer (plain links or markdown links). If multiple URLs support an item, include all of them in that item's source list.
- If an item is missing or not stated, set the text field to null and the corresponding sources list to an empty array.
- If the answer uses synonyms or equivalent time units (e.g., "six weeks" vs. "42 days"), keep exactly what the answer wrote.

Fields to extract:
1) Committee requirements:
   - min_committee_members: the stated minimum number and characterization of required committee members (e.g., "minimum of three graduate faculty members", "at least three members", etc.)
   - min_committee_members_sources: URL(s) cited for that statement
   - outside_department_requirement: the stated rule about outside-department/field representation or minor subjects (e.g., "at least two members represent minor subjects (typically outside the major department)" or "at least one member from outside the student's department", etc.)
   - outside_department_sources: URL(s) cited for that statement

2) Defense timeline:
   - pre_defense_distribution: the minimum advance time to distribute the dissertation to committee before defense (e.g., "at least 6 weeks", "at least 42 days", etc.)
   - pre_defense_sources: URL(s) cited for that statement
   - post_defense_submission: the maximum time allowed after the defense to submit the final dissertation (e.g., "within 60 days", etc.)
   - post_defense_sources: URL(s) cited for that statement

3) Document formatting:
   - margin_specifications: the required margin specifications (e.g., "at least 1 inch on all sides")
   - margins_sources: URL(s) cited for that statement
   - line_spacing_specifications: the line spacing requirement and any noted exceptions if the answer included them (e.g., "main body double-spaced; exceptions allowed for quotations, footnotes, tables")
   - line_spacing_sources: URL(s) cited for that statement

4) Abstract:
   - abstract_max_word_count: the maximum allowed word count for the dissertation abstract (e.g., "no more than 350 words")
   - abstract_sources: URL(s) cited for that statement

Return a single JSON object with exactly these fields.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_nonempty_text_and_sources(text: Optional[str], sources: List[str]) -> bool:
    return bool(text and text.strip()) and bool(sources and len(sources) > 0)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_committee_verification(
    evaluator: Evaluator,
    parent_node,
    ext: RequirementsExtraction,
) -> None:
    """
    Committee size and outside-department representation requirements.
    """
    committee_node = evaluator.add_parallel(
        id="Committee_Composition_Requirements",
        desc="Committee size and outside-department representation requirements",
        parent=parent_node,
        critical=True,
    )

    # Existence checks to gate follow-up verifications (critical)
    evaluator.add_custom_node(
        result=_has_nonempty_text_and_sources(ext.min_committee_members, ext.min_committee_members_sources),
        id="Minimum_Committee_Members_sources_exist",
        desc="Minimum committee members claim and sources are provided in the answer",
        parent=committee_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_nonempty_text_and_sources(ext.outside_department_requirement, ext.outside_department_sources),
        id="Outside_Department_Representation_sources_exist",
        desc="Outside-department/minor-subjects requirement claim and sources are provided in the answer",
        parent=committee_node,
        critical=True,
    )

    # Leaf verifications (critical)
    min_members_leaf = evaluator.add_leaf(
        id="Minimum_Committee_Members",
        desc="States the minimum number of committee members required (minimum of three graduate faculty members)",
        parent=committee_node,
        critical=True,
    )
    outside_dept_leaf = evaluator.add_leaf(
        id="Outside_Department_Representation",
        desc="States the minor-member requirement, including that at least two members represent minor subjects (typically from outside the student's major department)",
        parent=committee_node,
        critical=True,
    )

    # Prepare claims
    min_members_claim = (
        f"Cornell University Graduate School Ph.D. special committee minimum size requirement is stated as: "
        f"'{ext.min_committee_members}'. This exact requirement is supported by the provided source(s)."
    )
    outside_dept_claim = (
        f"Cornell University Graduate School policy on outside-department/minor-subject representation is stated as: "
        f"'{ext.outside_department_requirement}'. This exact requirement is supported by the provided source(s)."
    )

    await evaluator.batch_verify(
        [
            (
                min_members_claim,
                ext.min_committee_members_sources,
                min_members_leaf,
                "Verify that the cited Cornell policy page(s) explicitly support the stated minimum committee membership. "
                "Allow minor wording differences (e.g., 'at least three' vs. 'minimum of three'). "
                "Treat 'graduate faculty' vs. 'committee members' as equivalent if the page clearly implies graduate faculty membership."
            ),
            (
                outside_dept_claim,
                ext.outside_department_sources,
                outside_dept_leaf,
                "Verify that the cited Cornell policy page(s) explicitly support the stated outside-department/minor-subject representation requirement. "
                "Accept reasonable phrasing variants (e.g., 'minor members' vs. 'members representing minor subjects')."
            ),
        ]
    )


async def build_defense_timeline_verification(
    evaluator: Evaluator,
    parent_node,
    ext: RequirementsExtraction,
) -> None:
    """
    Timeline requirements before and after the defense.
    """
    timeline_node = evaluator.add_parallel(
        id="Defense_Timeline_Requirements",
        desc="Timeline requirements before and after the defense",
        parent=parent_node,
        critical=True,
    )

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=_has_nonempty_text_and_sources(ext.pre_defense_distribution, ext.pre_defense_sources),
        id="Pre_Defense_Distribution_sources_exist",
        desc="Pre-defense distribution timeframe claim and sources are provided in the answer",
        parent=timeline_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_nonempty_text_and_sources(ext.post_defense_submission, ext.post_defense_sources),
        id="Post_Defense_Submission_sources_exist",
        desc="Post-defense submission timeframe claim and sources are provided in the answer",
        parent=timeline_node,
        critical=True,
    )

    # Leaf verifications (critical)
    pre_defense_leaf = evaluator.add_leaf(
        id="Pre_Defense_Distribution",
        desc="States the minimum advance time to distribute the dissertation to committee members before the defense (at least 6 weeks / 42 days)",
        parent=timeline_node,
        critical=True,
    )
    post_defense_leaf = evaluator.add_leaf(
        id="Post_Defense_Submission",
        desc="States the maximum time allowed after the defense to submit the final dissertation (within 60 days)",
        parent=timeline_node,
        critical=True,
    )

    # Claims
    pre_defense_claim = (
        f"Students must distribute the dissertation to the full special committee at least '{ext.pre_defense_distribution}' before the defense, "
        f"as required by Cornell University Graduate School policy."
    )
    post_defense_claim = (
        f"The final dissertation must be submitted to the Graduate School within '{ext.post_defense_submission}' after the defense, "
        f"as required by Cornell University Graduate School policy."
    )

    await evaluator.batch_verify(
        [
            (
                pre_defense_claim,
                ext.pre_defense_sources,
                pre_defense_leaf,
                "Treat 'six weeks' and '42 days' as equivalent where applicable. Verify that the cited policy page(s) clearly state the minimum lead time."
            ),
            (
                post_defense_claim,
                ext.post_defense_sources,
                post_defense_leaf,
                "Verify that the cited policy page(s) clearly state the maximum allowed time window after the defense (e.g., 'within 60 days')."
            ),
        ]
    )


async def build_formatting_verification(
    evaluator: Evaluator,
    parent_node,
    ext: RequirementsExtraction,
) -> None:
    """
    Document margins and line spacing requirements.
    """
    formatting_node = evaluator.add_parallel(
        id="Document_Formatting_Requirements",
        desc="Document margins and line spacing requirements",
        parent=parent_node,
        critical=True,
    )

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=_has_nonempty_text_and_sources(ext.margin_specifications, ext.margins_sources),
        id="Margin_Specifications_sources_exist",
        desc="Margin specifications claim and sources are provided in the answer",
        parent=formatting_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_nonempty_text_and_sources(ext.line_spacing_specifications, ext.line_spacing_sources),
        id="Line_Spacing_Specifications_sources_exist",
        desc="Line spacing requirements claim and sources are provided in the answer",
        parent=formatting_node,
        critical=True,
    )

    # Leaf verifications (critical)
    margins_leaf = evaluator.add_leaf(
        id="Margin_Specifications",
        desc="States the required margin specifications (at least 1 inch on all sides)",
        parent=formatting_node,
        critical=True,
    )
    spacing_leaf = evaluator.add_leaf(
        id="Line_Spacing_Specifications",
        desc="States the line spacing requirements (main body double-spaced) and notes that specified exceptions are allowed (e.g., quotations, footnotes, tables, etc.)",
        parent=formatting_node,
        critical=True,
    )

    # Claims
    margins_claim = (
        f"The dissertation must have margins meeting this specification: '{ext.margin_specifications}'. "
        f"This requirement is supported by Cornell University Graduate School policy."
    )
    spacing_claim = (
        f"The dissertation line spacing requirement is: '{ext.line_spacing_specifications}'. "
        f"This requirement (including any stated exceptions if mentioned) is supported by Cornell University Graduate School policy."
    )

    await evaluator.batch_verify(
        [
            (
                margins_claim,
                ext.margins_sources,
                margins_leaf,
                "Verify that the cited policy page(s) explicitly require margins of at least 1 inch on all sides, allowing minor phrasing variants (e.g., 2.54 cm)."
            ),
            (
                spacing_claim,
                ext.line_spacing_sources,
                spacing_leaf,
                "Verify that the cited policy page(s) indicate main body text is double-spaced and that specified exceptions (e.g., block quotations, footnotes, tables) are allowed if the answer claims them."
            ),
        ]
    )


async def build_abstract_verification(
    evaluator: Evaluator,
    parent_node,
    ext: RequirementsExtraction,
) -> None:
    """
    Dissertation abstract word limit requirement.
    """
    abstract_node = evaluator.add_parallel(
        id="Abstract_Requirements",
        desc="Dissertation abstract word limit requirement",
        parent=parent_node,
        critical=True,
    )

    # Existence check (critical)
    evaluator.add_custom_node(
        result=_has_nonempty_text_and_sources(ext.abstract_max_word_count, ext.abstract_sources),
        id="Abstract_Maximum_Word_Count_sources_exist",
        desc="Abstract maximum word count claim and sources are provided in the answer",
        parent=abstract_node,
        critical=True,
    )

    # Leaf verification (critical)
    abstract_leaf = evaluator.add_leaf(
        id="Abstract_Maximum_Word_Count",
        desc="States the maximum allowed word count for the dissertation abstract (no more than 350 words)",
        parent=abstract_node,
        critical=True,
    )

    # Claim
    abstract_claim = (
        f"The dissertation abstract must be '{ext.abstract_max_word_count}'. "
        f"This requirement is supported by Cornell University Graduate School policy."
    )

    await evaluator.verify(
        claim=abstract_claim,
        node=abstract_leaf,
        sources=ext.abstract_sources,
        additional_instruction="Verify that the policy page(s) explicitly state the abstract limit (e.g., 'no more than 350 words'), allowing small wording differences."
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
    Evaluate an answer for Cornell University's Graduate School dissertation requirements.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level aggregate across requirement groups
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

    # Extract all requirement statements and their cited sources from the answer
    extracted: RequirementsExtraction = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=RequirementsExtraction,
        extraction_name="dissertation_requirements_extraction",
    )

    # Build the rubric tree as specified
    top_node = evaluator.add_parallel(
        id="Complete_Dissertation_Requirements",
        desc="Verify all required doctoral dissertation requirements specified in the question/constraints for Cornell Graduate School",
        parent=root,
        critical=True,
    )

    # Add optional ground truth info (for context only; not used in scoring)
    evaluator.add_ground_truth(
        {
            "items_required": [
                "Minimum committee members",
                "Outside-department/minor-subject representation",
                "Pre-defense distribution lead time",
                "Post-defense submission deadline",
                "Margins",
                "Line spacing (with any noted exceptions)",
                "Abstract maximum word count",
            ],
            "notes": "Evaluation checks both the presence of claims with cited URLs in the answer and whether those URLs support the claims.",
        },
        gt_type="rubric_requirements",
    )

    # Build subtrees
    await build_committee_verification(evaluator, top_node, extracted)
    await build_defense_timeline_verification(evaluator, top_node, extracted)
    await build_formatting_verification(evaluator, top_node, extracted)
    await build_abstract_verification(evaluator, top_node, extracted)

    # Return evaluator summary
    return evaluator.get_summary()