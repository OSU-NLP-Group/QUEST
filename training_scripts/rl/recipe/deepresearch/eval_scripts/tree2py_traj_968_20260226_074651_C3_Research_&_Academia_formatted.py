import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "chi_2026_paper_submission_requirements"
TASK_DESCRIPTION = (
    "A PhD student in Human-Computer Interaction has completed a research study on accessibility technologies designed "
    "to improve screen reader experiences for people with visual disabilities. They want to submit their findings as a paper "
    "to the CHI 2026 conference (ACM Conference on Human Factors in Computing Systems). Please provide the following complete "
    "submission requirements information for CHI 2026: (1) Subcommittee Selection: Which CHI 2026 paper subcommittee is most appropriate "
    "for this research topic? Provide the subcommittee name and include the reference URL to the official CHI 2026 page that lists and "
    "describes all available subcommittees. (2) Paper Length Requirements: What are the word count limits for CHI 2026 papers? Specifically "
    "state the maximum word count for Short papers, the word count range for Standard-length papers, the word count threshold above which papers "
    "will be desk-rejected, and what elements are excluded from the word count. Include the reference URL to the official CHI 2026 page specifying "
    "these length requirements. (3) Format and Template: What submission format and template must be used for CHI 2026 papers during the review phase? "
    "Also explain the 'stand-alone' requirement for papers. Include the reference URL to the official CHI 2026 page with formatting guidelines. "
    "(4) Anonymization Policy: What are the anonymization requirements for CHI 2026 paper submissions? Specify requirements for both the main paper "
    "and any supplementary materials. Include the reference URL to the official CHI 2026 page describing anonymization requirements. (5) Submission Deadlines: "
    "What are the key submission deadlines for CHI 2026 papers (in Anywhere on Earth timezone)? Include the abstract/metadata deadline (and note the maximum abstract "
    "length and any restrictions), full paper deadline, and optional video figures and supplementary materials deadline. Include the reference URL to the official "
    "CHI 2026 page listing these deadlines."
)

# Optional ground-truth guidance (used only for context in summary)
GROUND_TRUTH_GUIDE = {
    "short_paper_max": "5,000 words or less",
    "standard_paper_range": "5,000 to 12,000 words",
    "desk_reject_threshold": "exceeding 12,000 words",
    "word_count_exclusions": ["references", "figure/table captions", "appendices"],
    "deadlines_expected_AoE": True,
    "abstract_metadata_date": "Thursday, September 4, 2025, AoE",
    "full_paper_deadline": "Thursday, September 11, 2025, AoE",
    "video_supp_deadline": "Thursday, September 18, 2025, AoE",
    "abstract_word_limit": "150 words max",
    "author_list_restriction": "Author list cannot be changed after abstract/metadata deadline",
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SubcommitteeInfo(BaseModel):
    subcommittee_name: Optional[str] = None
    subcommittee_ref_url: Optional[str] = None


class LengthRequirements(BaseModel):
    short_max_words: Optional[str] = None
    standard_range_words: Optional[str] = None
    desk_reject_threshold: Optional[str] = None
    exclusions_list: List[str] = Field(default_factory=list)
    length_ref_url: Optional[str] = None


class FormatTemplateInfo(BaseModel):
    review_template_desc: Optional[str] = None
    stand_alone_requirement_desc: Optional[str] = None
    format_ref_url: Optional[str] = None


class AnonymizationPolicy(BaseModel):
    paper_anonymization_desc: Optional[str] = None
    supplementary_anonymization_desc: Optional[str] = None
    anonymization_ref_url: Optional[str] = None


class SubmissionDeadlines(BaseModel):
    abstract_metadata_date: Optional[str] = None
    abstract_word_limit: Optional[str] = None
    author_list_restrictions_desc: Optional[str] = None
    full_paper_deadline_date: Optional[str] = None
    video_supplementary_deadline_date: Optional[str] = None
    deadline_ref_url: Optional[str] = None


class CHIRequirementsExtraction(BaseModel):
    subcommittee: Optional[SubcommitteeInfo] = None
    length: Optional[LengthRequirements] = None
    format_template: Optional[FormatTemplateInfo] = None
    anonymization: Optional[AnonymizationPolicy] = None
    deadlines: Optional[SubmissionDeadlines] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_chi_requirements() -> str:
    return (
        "Extract all CHI 2026 paper submission requirements explicitly stated in the answer. "
        "Return a JSON with the following structure and fields, using strings for values and arrays for lists:\n"
        "- subcommittee:\n"
        "  • subcommittee_name: the chosen CHI 2026 subcommittee name for the paper\n"
        "  • subcommittee_ref_url: the URL to the official CHI 2026 page listing and describing subcommittees\n"
        "- length:\n"
        "  • short_max_words: the maximum word count for Short papers (e.g., '5,000 words or less')\n"
        "  • standard_range_words: the word count range for Standard-length papers (e.g., '5,000 to 12,000 words')\n"
        "  • desk_reject_threshold: the word count threshold above which papers are desk-rejected (e.g., 'exceeding 12,000 words')\n"
        "  • exclusions_list: array of elements excluded from the word count (e.g., ['references','figure/table captions','appendices'])\n"
        "  • length_ref_url: the URL to the official CHI 2026 page specifying length requirements\n"
        "- format_template:\n"
        "  • review_template_desc: the required review-phase submission format/template (e.g., 'single-column ACM template')\n"
        "  • stand_alone_requirement_desc: the stand-alone requirement description in the answer\n"
        "  • format_ref_url: the URL to the official CHI 2026 page with formatting/template guidelines\n"
        "- anonymization:\n"
        "  • paper_anonymization_desc: anonymization requirement for the main paper (string)\n"
        "  • supplementary_anonymization_desc: anonymization requirement for supplementary materials (string)\n"
        "  • anonymization_ref_url: the URL to the official CHI 2026 page describing anonymization requirements\n"
        "- deadlines:\n"
        "  • abstract_metadata_date: abstract/metadata deadline date string (include AoE if provided)\n"
        "  • abstract_word_limit: the abstract max word length (e.g., '150 words')\n"
        "  • author_list_restrictions_desc: author list change restriction statement (string)\n"
        "  • full_paper_deadline_date: full paper submission deadline date string (include AoE if provided)\n"
        "  • video_supplementary_deadline_date: optional video figures/supplementary materials deadline date string (include AoE if provided)\n"
        "  • deadline_ref_url: the URL to the official CHI 2026 page listing deadlines\n"
        "If any field is missing in the answer, set it to null or an empty array (for exclusions_list). "
        "Extract only URLs explicitly present in the answer (plain or markdown)."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _url_provided(url: Optional[str]) -> bool:
    return bool(url and isinstance(url, str) and url.strip())


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_subcommittee(
    evaluator: Evaluator,
    parent_node,
    data: Optional[SubcommitteeInfo],
) -> None:
    node = evaluator.add_parallel(
        id="Subcommittee_Selection",
        desc="Provide the appropriate CHI 2026 subcommittee for research on accessibility technologies for screen reader users, including reference URL",
        parent=parent_node,
        critical=False
    )

    # Existence check for URL (critical gate)
    evaluator.add_custom_node(
        result=_url_provided(data.subcommittee_ref_url if data else None),
        id="Subcommittee_URL_Provided",
        desc="Subcommittee reference URL is provided",
        parent=node,
        critical=True
    )

    # Leaf: Subcommittee Reference URL (critical)
    subcommittee_url_leaf = evaluator.add_leaf(
        id="Subcommittee_Reference_URL",
        desc="Provide the official CHI 2026 webpage that lists and describes the subcommittees",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage lists and describes CHI 2026 paper subcommittees and is an official CHI 2026 page.",
        node=subcommittee_url_leaf,
        sources=data.subcommittee_ref_url if data else None,
        additional_instruction="Verify that the page is from the official CHI 2026 website and includes subcommittee descriptions."
    )

    # Leaf: Correct Subcommittee Identified (critical)
    correct_subcommittee_leaf = evaluator.add_leaf(
        id="Correct_Subcommittee_Identified",
        desc="The selected subcommittee is appropriate for research on accessibility technologies for people with visual disabilities using screen readers",
        parent=node,
        critical=True
    )
    name = (data.subcommittee_name or "").strip() if data else ""
    await evaluator.verify(
        claim=f"The CHI 2026 subcommittee '{name}' is appropriate for research on accessibility technologies for screen reader users (people with visual disabilities).",
        node=correct_subcommittee_leaf,
        sources=data.subcommittee_ref_url if data else None,
        additional_instruction="Check whether the subcommittee scope mentions accessibility, assistive technology, screen readers, or visual disabilities."
    )


async def verify_length_requirements(
    evaluator: Evaluator,
    parent_node,
    data: Optional[LengthRequirements],
) -> None:
    node = evaluator.add_parallel(
        id="Paper_Length_Requirements",
        desc="Provide complete word count requirements for CHI 2026 papers, including reference URL",
        parent=parent_node,
        critical=False
    )

    # Existence check for URL (critical gate)
    evaluator.add_custom_node(
        result=_url_provided(data.length_ref_url if data else None),
        id="Length_URL_Provided",
        desc="Length requirements reference URL is provided",
        parent=node,
        critical=True
    )

    # Reference URL leaf (critical)
    length_ref_leaf = evaluator.add_leaf(
        id="Length_Reference_URL",
        desc="Provide the official CHI 2026 webpage that specifies paper length requirements",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage specifies CHI 2026 paper word count requirements.",
        node=length_ref_leaf,
        sources=data.length_ref_url if data else None,
        additional_instruction="Verify that this is an official CHI 2026 page detailing word count policies for papers."
    )

    # Short paper limit (critical)
    short_leaf = evaluator.add_leaf(
        id="Short_Paper_Limit",
        desc="Correctly state the maximum word count for Short papers (5,000 words or less)",
        parent=node,
        critical=True
    )
    short_text = (data.short_max_words or "").strip() if data else ""
    await evaluator.verify(
        claim=f"The maximum word count for CHI 2026 Short papers is '{short_text}'.",
        node=short_leaf,
        sources=data.length_ref_url if data else None,
        additional_instruction="Confirm that short papers have a maximum of 5,000 words (or less) as per the official CHI 2026 policy."
    )

    # Standard paper range (critical)
    standard_leaf = evaluator.add_leaf(
        id="Standard_Paper_Range",
        desc="Correctly state the word count range for Standard-length papers (5,000 to 12,000 words)",
        parent=node,
        critical=True
    )
    standard_text = (data.standard_range_words or "").strip() if data else ""
    await evaluator.verify(
        claim=f"The word count range for CHI 2026 Standard-length papers is '{standard_text}'.",
        node=standard_leaf,
        sources=data.length_ref_url if data else None,
        additional_instruction="Confirm that standard-length papers are between 5,000 and 12,000 words inclusive."
    )

    # Excessive length threshold (critical)
    threshold_leaf = evaluator.add_leaf(
        id="Excessive_Length_Threshold",
        desc="Correctly state the word count threshold above which papers face desk rejection (exceeding 12,000 words)",
        parent=node,
        critical=True
    )
    threshold_text = (data.desk_reject_threshold or "").strip() if data else ""
    await evaluator.verify(
        claim=f"Papers exceeding '{threshold_text}' words will be desk-rejected by CHI 2026.",
        node=threshold_leaf,
        sources=data.length_ref_url if data else None,
        additional_instruction="Verify that the desk-reject threshold is exceeding 12,000 words according to the official policy."
    )

    # Word count exclusions (critical)
    exclusions_leaf = evaluator.add_leaf(
        id="Word_Count_Exclusions",
        desc="Correctly state what elements are excluded from the word count (references, figure/table captions, and appendices)",
        parent=node,
        critical=True
    )
    excl_list = (data.exclusions_list if data and data.exclusions_list else [])
    excl_text = ", ".join(excl_list) if excl_list else ""
    await evaluator.verify(
        claim=f"The CHI 2026 word count excludes: {excl_text}.",
        node=exclusions_leaf,
        sources=data.length_ref_url if data else None,
        additional_instruction="Confirm that references, figure/table captions, and appendices are excluded from word count (minor wording variants acceptable)."
    )


async def verify_format_template(
    evaluator: Evaluator,
    parent_node,
    data: Optional[FormatTemplateInfo],
) -> None:
    node = evaluator.add_parallel(
        id="Format_and_Template_Requirements",
        desc="Provide format, template, and stand-alone requirements for CHI 2026 papers, including reference URL",
        parent=parent_node,
        critical=False
    )

    # Existence check for URL (critical gate)
    evaluator.add_custom_node(
        result=_url_provided(data.format_ref_url if data else None),
        id="Format_URL_Provided",
        desc="Formatting/template reference URL is provided",
        parent=node,
        critical=True
    )

    # Reference URL leaf (critical)
    format_ref_leaf = evaluator.add_leaf(
        id="Format_Reference_URL",
        desc="Provide the official CHI 2026 webpage that specifies formatting and template requirements",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage specifies the formatting and template requirements for CHI 2026 paper submissions.",
        node=format_ref_leaf,
        sources=data.format_ref_url if data else None,
        additional_instruction="Verify that this is an official CHI 2026 page and covers formatting/template for the review phase."
    )

    # Required template format (critical)
    template_leaf = evaluator.add_leaf(
        id="Required_Template_Format",
        desc="Correctly identify that the single-column ACM template must be used for the review phase",
        parent=node,
        critical=True
    )
    template_text = (data.review_template_desc or "").strip() if data else ""
    await evaluator.verify(
        claim=f"During review, CHI 2026 papers must use the single-column ACM template (answer states: '{template_text}').",
        node=template_leaf,
        sources=data.format_ref_url if data else None,
        additional_instruction="Confirm the review-phase template is ACM single-column; allow reasonable wording variants."
    )

    # Stand-alone requirement (critical)
    standalone_leaf = evaluator.add_leaf(
        id="Stand_Alone_Requirement",
        desc="State that the paper must be stand-alone with all essential information in the main PDF, and that reviewers are not required to read appendices or supplementary materials",
        parent=node,
        critical=True
    )
    standalone_text = (data.stand_alone_requirement_desc or "").strip() if data else ""
    await evaluator.verify(
        claim=f"CHI 2026 papers must be stand-alone: essential information must be in the main PDF, and reviewers are not required to read appendices or supplementary materials (answer states: '{standalone_text}').",
        node=standalone_leaf,
        sources=data.format_ref_url if data else None,
        additional_instruction="Check the official guideline for the stand-alone requirement; allow minor wording variation."
    )


async def verify_anonymization(
    evaluator: Evaluator,
    parent_node,
    data: Optional[AnonymizationPolicy],
) -> None:
    node = evaluator.add_parallel(
        id="Anonymization_Requirements",
        desc="Provide anonymization requirements for CHI 2026 submissions, including reference URL",
        parent=parent_node,
        critical=False
    )

    # Existence check for URL (critical gate)
    evaluator.add_custom_node(
        result=_url_provided(data.anonymization_ref_url if data else None),
        id="Anonymization_URL_Provided",
        desc="Anonymization policy reference URL is provided",
        parent=node,
        critical=True
    )

    # Reference URL leaf (critical)
    anonymization_ref_leaf = evaluator.add_leaf(
        id="Anonymization_Reference_URL",
        desc="Provide the official CHI 2026 webpage that describes anonymization requirements",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage describes anonymization requirements for CHI 2026 paper submissions.",
        node=anonymization_ref_leaf,
        sources=data.anonymization_ref_url if data else None,
        additional_instruction="Verify that this is an official CHI 2026 anonymization policy page."
    )

    # Paper anonymization (critical)
    paper_anon_leaf = evaluator.add_leaf(
        id="Paper_Anonymization",
        desc="State that the paper must be fully anonymized throughout the review process",
        parent=node,
        critical=True
    )
    paper_anon_text = (data.paper_anonymization_desc or "").strip() if data else ""
    await evaluator.verify(
        claim=f"The main paper must be fully anonymized throughout the review process (answer states: '{paper_anon_text}').",
        node=paper_anon_leaf,
        sources=data.anonymization_ref_url if data else None,
        additional_instruction="Confirm that author identities and affiliations are removed per the anonymization policy."
    )

    # Supplementary materials anonymization (critical)
    supp_anon_leaf = evaluator.add_leaf(
        id="Supplementary_Materials_Anonymization",
        desc="State that all supplementary materials, including video figures, must also be anonymized",
        parent=node,
        critical=True
    )
    supp_anon_text = (data.supplementary_anonymization_desc or "").strip() if data else ""
    await evaluator.verify(
        claim=f"All supplementary materials (including video figures) must be anonymized (answer states: '{supp_anon_text}').",
        node=supp_anon_leaf,
        sources=data.anonymization_ref_url if data else None,
        additional_instruction="Confirm that videos and supplementary files must avoid revealing author identities."
    )


async def verify_submission_deadlines(
    evaluator: Evaluator,
    parent_node,
    data: Optional[SubmissionDeadlines],
) -> None:
    node = evaluator.add_parallel(
        id="Submission_Deadlines",
        desc="Provide all key submission deadlines for CHI 2026 papers in Anywhere on Earth timezone, including reference URL",
        parent=parent_node,
        critical=False
    )

    # Existence check for URL (critical gate)
    evaluator.add_custom_node(
        result=_url_provided(data.deadline_ref_url if data else None),
        id="Deadline_URL_Provided",
        desc="Deadlines reference URL is provided",
        parent=node,
        critical=True
    )

    # Reference URL leaf (critical)
    deadline_ref_leaf = evaluator.add_leaf(
        id="Deadline_Reference_URL",
        desc="Provide the official CHI 2026 webpage that lists all submission deadlines",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage lists CHI 2026 paper submission deadlines (AoE).",
        node=deadline_ref_leaf,
        sources=data.deadline_ref_url if data else None,
        additional_instruction="Verify that the page is official CHI 2026 and includes deadlines in AoE."
    )

    # Abstract/metadata deadline date (critical)
    abstract_deadline_leaf = evaluator.add_leaf(
        id="Abstract_Metadata_Deadline_Date",
        desc="Correctly state the abstract/metadata deadline date (Thursday, September 4, 2025, AoE)",
        parent=node,
        critical=True
    )
    abs_date = (data.abstract_metadata_date or "").strip() if data else ""
    await evaluator.verify(
        claim=f"The abstract/metadata deadline is '{abs_date}' (AoE).",
        node=abstract_deadline_leaf,
        sources=data.deadline_ref_url if data else None,
        additional_instruction="Confirm the abstract/metadata date and AoE timezone on the official page."
    )

    # Abstract word limit (critical)
    abstract_word_leaf = evaluator.add_leaf(
        id="Abstract_Word_Limit",
        desc="State that the abstract must be maximum 150 words",
        parent=node,
        critical=True
    )
    abs_words = (data.abstract_word_limit or "").strip() if data else ""
    await evaluator.verify(
        claim=f"The abstract must be a maximum of '{abs_words}'.",
        node=abstract_word_leaf,
        sources=data.deadline_ref_url if data else None,
        additional_instruction="Verify the abstract word limit (expected 150 words max) per official guidance."
    )

    # Author list restriction (critical)
    author_list_leaf = evaluator.add_leaf(
        id="Author_List_Restriction",
        desc="State that the author list cannot be changed after the abstract/metadata deadline",
        parent=node,
        critical=True
    )
    author_restr = (data.author_list_restrictions_desc or "").strip() if data else ""
    await evaluator.verify(
        claim=f"The author list cannot be changed after the abstract/metadata deadline (answer states: '{author_restr}').",
        node=author_list_leaf,
        sources=data.deadline_ref_url if data else None,
        additional_instruction="Confirm that the author list is frozen after the abstract/metadata deadline."
    )

    # Full paper deadline (critical)
    full_paper_leaf = evaluator.add_leaf(
        id="Full_Paper_Deadline",
        desc="Correctly state the full paper submission deadline date (Thursday, September 11, 2025, AoE)",
        parent=node,
        critical=True
    )
    full_date = (data.full_paper_deadline_date or "").strip() if data else ""
    await evaluator.verify(
        claim=f"The full paper submission deadline is '{full_date}' (AoE).",
        node=full_paper_leaf,
        sources=data.deadline_ref_url if data else None,
        additional_instruction="Confirm the full paper deadline and AoE timezone on the official page."
    )

    # Video/supplementary deadline (non-critical)
    video_supp_leaf = evaluator.add_leaf(
        id="Video_Supplementary_Deadline",
        desc="Correctly state the deadline for optional video figures and supplementary materials (Thursday, September 18, 2025, AoE)",
        parent=node,
        critical=False
    )
    video_date = (data.video_supplementary_deadline_date or "").strip() if data else ""
    await evaluator.verify(
        claim=f"The deadline for optional video figures and supplementary materials is '{video_date}' (AoE).",
        node=video_supp_leaf,
        sources=data.deadline_ref_url if data else None,
        additional_instruction="Confirm the optional videos/supplementary deadline on the official page."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
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
    """
    Evaluate an answer for the CHI 2026 paper submission requirements task.
    """
    # Initialize evaluator (root is non-critical by framework design)
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

    # Top-level node (set to non-critical to allow partial credit without violating the framework's critical-child constraint)
    top = evaluator.add_parallel(
        id="CHI_2026_Paper_Submission_Requirements",
        desc="Verify complete compliance with CHI 2026 paper submission requirements for a research paper on accessibility technologies for screen reader users",
        parent=root,
        critical=False
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_chi_requirements(),
        template_class=CHIRequirementsExtraction,
        extraction_name="chi_2026_requirements"
    )

    # Add guidance info to summary for transparency (not used in scoring)
    evaluator.add_ground_truth({"guide": GROUND_TRUTH_GUIDE}, gt_type="expected_policy_guidance")

    # Build and verify subtrees
    await verify_subcommittee(evaluator, top, extracted.subcommittee)
    await verify_length_requirements(evaluator, top, extracted.length)
    await verify_format_template(evaluator, top, extracted.format_template)
    await verify_anonymization(evaluator, top, extracted.anonymization)
    await verify_submission_deadlines(evaluator, top, extracted.deadlines)

    # Return structured evaluation summary
    return evaluator.get_summary()