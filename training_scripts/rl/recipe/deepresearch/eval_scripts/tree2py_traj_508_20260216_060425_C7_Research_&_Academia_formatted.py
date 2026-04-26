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
TASK_ID = "biomed_journal_requirements"
TASK_DESCRIPTION = (
    "What are the specific requirements and standards that must be met when establishing a new open-access academic "
    "journal in the biomedical sciences that seeks DOAJ (Directory of Open Access Journals) indexing and deposits articles "
    "in PubMed Central? Please provide the following information: (1) the minimum publishing history OR minimum number of "
    "published articles required for DOAJ indexing, (2) the standard peer review timeline average for medical, public health, "
    "and natural science journals, (3) the educational qualification requirements for editorial board members, (4) the "
    "publication record requirements for editorial board members, (5) whether a data sharing policy must be established, "
    "(6) the specific IRB regulatory reference requiring assurance that risks to human subjects are minimized, (7) the "
    "standard definition of 'human subjects' in research, (8) the maximum size limit for supplementary datasets when "
    "depositing in PubMed Central, (9) the types of open access licenses that should be specified, (10) whether documentation "
    "of informed consent must be required for research involving human participants, (11) whether peer review procedures "
    "must be publicly described, (12) whether guidelines for multi-institutional collaboration coordination must be provided, "
    "and (13) what recognized publication ethics standards should be adopted."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ItemInfo(BaseModel):
    text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class JournalRequirementsExtraction(BaseModel):
    publishing_history_requirement: Optional[ItemInfo] = None
    peer_review_timeline_standard: Optional[ItemInfo] = None
    editorial_board_doctoral_qualification: Optional[ItemInfo] = None
    editorial_board_publication_record: Optional[ItemInfo] = None
    data_sharing_policy: Optional[ItemInfo] = None
    irb_ethics_requirement: Optional[ItemInfo] = None
    human_subjects_definition: Optional[ItemInfo] = None
    supplementary_data_size_limit: Optional[ItemInfo] = None
    open_access_licensing: Optional[ItemInfo] = None
    author_consent_documentation: Optional[ItemInfo] = None
    peer_review_process_description: Optional[ItemInfo] = None
    multi_institutional_collaboration_guidelines: Optional[ItemInfo] = None
    publication_ethics_standards: Optional[ItemInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_journal_requirements() -> str:
    return """
    Extract the journal requirements/standards and the explicitly cited source URLs for each of the following 13 items from the provided answer text.
    For each item, return:
      - text: the exact statement or summary the answer gives for that item (use the phrasing from the answer; keep it concise but faithful).
      - sources: all URLs explicitly associated with that item in the answer. Extract URLs as full URLs; include only URLs that are actually present in the answer.

    Items to extract (use these exact field keys in your JSON):
      1) publishing_history_requirement
         - What the answer claims as the minimum publishing history OR minimum number of published articles for DOAJ indexing.
      2) peer_review_timeline_standard
         - The standard/typical peer review timeline (e.g., average time for initial decision) for medical, public health, and natural science journals.
      3) editorial_board_doctoral_qualification
         - Educational qualification requirements for editorial board members (e.g., doctoral degree).
      4) editorial_board_publication_record
         - Publication record requirements for editorial board members (e.g., prior scholarly publications).
      5) data_sharing_policy
         - Whether a data sharing policy must be established and what it should cover.
      6) irb_ethics_requirement
         - The IRB requirement and specific regulatory reference about minimizing risks to human subjects (e.g., 21 CFR 56.111(a)(1)).
      7) human_subjects_definition
         - The standard definition of "human subjects" in research (e.g., consistent with US federal regs).
      8) supplementary_data_size_limit
         - The maximum size limit for supplementary datasets when depositing in PubMed Central (PMC).
      9) open_access_licensing
         - The open access licensing options that should be specified (e.g., CC BY, CC0).
      10) author_consent_documentation
          - Whether documentation of informed consent must be required for research involving human participants.
      11) peer_review_process_description
          - Whether peer review procedures must be publicly described.
      12) multi_institutional_collaboration_guidelines
          - Whether guidelines for coordination/management of multi-institutional collaborations must be provided.
      13) publication_ethics_standards
          - What recognized publication ethics standards should be adopted (e.g., COPE).

    Requirements:
    - If an item is not mentioned in the answer, set its 'text' to null and 'sources' to an empty list.
    - The 'sources' arrays should include only URLs that are present in the answer. Do not invent URLs.
    - Keep each 'text' faithful to the answer and self-contained, not relying on other parts of the answer for context.
    """


# --------------------------------------------------------------------------- #
# Additional instructions for verification per item                           #
# --------------------------------------------------------------------------- #
def get_additional_instruction_for_item(item_key: str) -> str:
    instructions = {
        "publishing_history_requirement": (
            "Focus on DOAJ application criteria. Verify whether the webpage(s) explicitly describe the minimum publishing "
            "history and/or minimum number of published articles needed for DOAJ indexing eligibility. Allow equivalent "
            "wording (e.g., 'one year' vs '12 months'). If the page is not about DOAJ criteria, do not accept."
        ),
        "peer_review_timeline_standard": (
            "Check if the source(s) describe a typical or average initial peer-review timeline in medical, public health, or "
            "natural sciences that matches the statement (e.g., around the stated number of weeks). Reasonable ranges or "
            "equivalents are acceptable."
        ),
        "editorial_board_doctoral_qualification": (
            "Check if the source(s) state that editorial board members must/should have doctoral-level qualifications "
            "(e.g., PhD, MD, DrPH). Minor wording differences are acceptable if the meaning is clearly equivalent."
        ),
        "editorial_board_publication_record": (
            "Check if the source(s) indicate that editorial board members should have a track record of scholarly publications. "
            "Equivalent phrasing is acceptable."
        ),
        "data_sharing_policy": (
            "Check whether the source(s) clearly require or recommend that journals establish a data sharing policy and "
            "explain expectations for making research data available."
        ),
        "irb_ethics_requirement": (
            "Verify that the source(s) cite or paraphrase 21 CFR 56.111(a)(1) regarding IRB criteria that risks to subjects "
            "are minimized. Equivalent wording is acceptable if clearly referring to that regulation."
        ),
        "human_subjects_definition": (
            "Verify that the source(s) define 'human subject' consistently with standard definitions (e.g., 45 CFR 46.102(e)(1)): "
            "a living individual about whom an investigator obtains data through intervention/interaction or identifiable private information."
        ),
        "supplementary_data_size_limit": (
            "Verify that the source(s), preferably from PubMed Central (PMC/NLM/NCBI), specify the maximum permissible size "
            "for supplementary datasets and that it matches the statement."
        ),
        "open_access_licensing": (
            "Verify that the source(s) indicate specifying open licenses (e.g., Creative Commons) such as CC BY or CC0 for "
            "published content. Accept equivalent CC license families if clearly stated."
        ),
        "author_consent_documentation": (
            "Verify that the source(s) require documentation of informed consent for human participant research when applicable. "
            "Equivalent policy language is acceptable."
        ),
        "peer_review_process_description": (
            "Verify that the source(s) require or recommend that journals publicly describe their peer-review procedures."
        ),
        "multi_institutional_collaboration_guidelines": (
            "Verify that the source(s) require or recommend that authors describe the coordination and management for "
            "multi-institutional collaborations."
        ),
        "publication_ethics_standards": (
            "Verify that the source(s) require adopting recognized publication ethics standards (e.g., COPE; ICMJE or WAME also acceptable "
            "if the statement says so)."
        ),
    }
    # Default generic instruction if key not found
    return instructions.get(
        item_key,
        "Determine whether the webpages explicitly support the statement. Minor wording differences are acceptable if the meaning matches."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_item_with_sources(
    evaluator: Evaluator,
    parent_node,
    item_key: str,
    item_desc: str,
    item: Optional[ItemInfo],
) -> None:
    """
    Create a sub-tree for a single requirement:
    - Parent (non-critical, parallel): item description
    - Critical existence node: Requires non-empty text and at least one source URL
    - Critical verification leaf: Statement supported by the cited sources
    """
    # Parent group node for this requirement
    item_node = evaluator.add_parallel(
        id=item_key,
        desc=item_desc,
        parent=parent_node,
        critical=False
    )

    # Existence check: statement present AND at least one source URL
    has_text = bool(item and item.text and item.text.strip())
    has_sources = bool(item and item.sources and len(item.sources) > 0)
    evaluator.add_custom_node(
        result=(has_text and has_sources),
        id=f"{item_key}_provided",
        desc=f"Answer provides a statement and at least one source URL for: {item_key}",
        parent=item_node,
        critical=True
    )

    # Source-grounded verification leaf
    verify_leaf = evaluator.add_leaf(
        id=f"{item_key}_source_support",
        desc=f"The statement for {item_key} is supported by the cited sources",
        parent=item_node,
        critical=True
    )

    statement_text = item.text if item and item.text else ""
    claim = f"The following statement is accurate and supported by the provided sources: '{statement_text}'."

    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=(item.sources if item else []),
        additional_instruction=get_additional_instruction_for_item(item_key)
    )


# --------------------------------------------------------------------------- #
# Main verification assembly                                                  #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    root,
    extracted: JournalRequirementsExtraction
) -> None:
    """
    Build the verification tree according to the rubric and verify each item using the extracted content.
    """
    # Map the rubric items (IDs to descriptions) to the extracted fields
    items: List[Dict[str, Any]] = [
        {
            "id": "publishing_history_requirement",
            "desc": "The journal must demonstrate either a publishing history of more than one year OR have published at least ten open access research articles to qualify for DOAJ indexing",
            "value": extracted.publishing_history_requirement,
        },
        {
            "id": "peer_review_timeline_standard",
            "desc": "The journal should establish a peer review timeline that averages 12-14 weeks for initial review in medical, public health, and natural science fields",
            "value": extracted.peer_review_timeline_standard,
        },
        {
            "id": "editorial_board_doctoral_qualification",
            "desc": "Editorial board members must have completed a doctoral programme",
            "value": extracted.editorial_board_doctoral_qualification,
        },
        {
            "id": "editorial_board_publication_record",
            "desc": "Editorial board members must have a record of published research in scholarly journals",
            "value": extracted.editorial_board_publication_record,
        },
        {
            "id": "data_sharing_policy",
            "desc": "The journal must establish a data sharing policy that specifies how authors should make research data available",
            "value": extracted.data_sharing_policy,
        },
        {
            "id": "irb_ethics_requirement",
            "desc": "The journal must require authors to confirm that research involving human subjects received IRB approval with assurance that risks to subjects are minimized per 21 CFR 56.111(a)(1)",
            "value": extracted.irb_ethics_requirement,
        },
        {
            "id": "human_subjects_definition",
            "desc": "The journal's ethics policy must define human subjects research using the standard definition: a living individual about whom an investigator obtains data through intervention or interaction or identifiable private information",
            "value": extracted.human_subjects_definition,
        },
        {
            "id": "supplementary_data_size_limit",
            "desc": "If the journal deposits articles in PubMed Central, supplementary datasets must be limited to 2 GB in size",
            "value": extracted.supplementary_data_size_limit,
        },
        {
            "id": "open_access_licensing",
            "desc": "The journal must specify open access licensing options such as CC-0 or CC-BY licenses for published content",
            "value": extracted.open_access_licensing,
        },
        {
            "id": "author_consent_documentation",
            "desc": "The journal must require proper documentation of informed consent from research participants when applicable",
            "value": extracted.author_consent_documentation,
        },
        {
            "id": "peer_review_process_description",
            "desc": "The journal must publicly describe its peer review procedures",
            "value": extracted.peer_review_process_description,
        },
        {
            "id": "multi_institutional_collaboration_guidelines",
            "desc": "For studies involving multiple institutions, the journal must require authors to describe coordination and management of the collaborative effort",
            "value": extracted.multi_institutional_collaboration_guidelines,
        },
        {
            "id": "publication_ethics_standards",
            "desc": "The journal must adopt recognized publication ethics standards such as those from COPE (Committee on Publication Ethics)",
            "value": extracted.publication_ethics_standards,
        },
    ]

    # Root already created with parallel aggregation; iterate over items
    for item in items:
        await verify_item_with_sources(
            evaluator=evaluator,
            parent_node=root,
            item_key=item["id"],
            item_desc=item["desc"],
            item=item["value"],
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the biomedical journal requirements task.
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

    # Extract the structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_journal_requirements(),
        template_class=JournalRequirementsExtraction,
        extraction_name="journal_requirements_extraction"
    )

    # Build tree and verify all items
    await build_and_verify_tree(evaluator, root, extraction)

    # Return summarized evaluation
    return evaluator.get_summary()