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
TASK_ID = "oh_house_written_testimony"
TASK_DESCRIPTION = (
    "You are planning to submit written-only testimony to an Ohio House of Representatives standing committee hearing. "
    "Research and provide the complete procedural requirements you must follow to properly submit your testimony. "
    "Your response must include: (1) The advance submission deadline: Specify how far in advance of the committee hearing you must submit your testimony. "
    "(2) Required documents: List all documents you must submit, including any specific format requirements (such as file type) and where to obtain any necessary forms. "
    "(3) Submission protocol: Describe the method by which you must submit your testimony and identify who the recipient should be. "
    "(4) Committee notice timing: Explain when committee hearing notices are typically posted. "
    "For each procedural requirement you identify, provide a reference URL to an official Ohio government or legislative source that confirms this information."
)

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class TestimonyRequirementsExtraction(BaseModel):
    # Submission timing
    advance_deadline: Optional[str] = None
    advance_deadline_urls: List[str] = Field(default_factory=list)

    notice_timing: Optional[str] = None
    notice_timing_urls: List[str] = Field(default_factory=list)

    # Required documents
    witness_slip_required: Optional[str] = None
    witness_slip_urls: List[str] = Field(default_factory=list)
    witness_slip_form_urls: List[str] = Field(default_factory=list)  # where to obtain the standardized form

    testimony_format: Optional[str] = None  # e.g., "PDF"
    testimony_format_urls: List[str] = Field(default_factory=list)

    # Submission protocol
    submission_method: Optional[str] = None  # e.g., "email"
    submission_method_urls: List[str] = Field(default_factory=list)

    recipient_information: Optional[str] = None  # e.g., "send to the committee chair's office; email in notice"
    recipient_information_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
Extract from the answer the specific procedural requirements for submitting written-only testimony to an Ohio House of Representatives standing committee hearing.

Return a single JSON object with the following fields (use null for missing strings and [] for missing URL arrays):

1) advance_deadline: The exact timing requirement described in the answer for when written testimony must be submitted (e.g., "24 hours in advance", "one business day prior", etc.).
2) advance_deadline_urls: All official URLs the answer cites for this deadline requirement. Only include URLs that explicitly appear in the answer text.

3) notice_timing: The timing pattern described in the answer for when committee hearing notices are typically posted (e.g., "Thursday or Friday of the week before the hearing").
4) notice_timing_urls: All official URLs the answer cites for this notice timing.

5) witness_slip_required: The answer’s statement regarding whether a witness slip is required and whether a standardized witness slip form is used.
6) witness_slip_urls: All official URLs the answer cites that substantiate the witness slip requirement and/or standardized form usage.
7) witness_slip_form_urls: The URL(s) in the answer where the standardized witness slip form can be obtained (or the page hosting the form).

8) testimony_format: The answer’s statement about file format requirements for written testimony (e.g., "PDF").
9) testimony_format_urls: All official URLs the answer cites for the format requirement.

10) submission_method: The answer’s statement on how testimony must be submitted (e.g., "via email").
11) submission_method_urls: All official URLs the answer cites for the submission method.

12) recipient_information: The answer’s statement about who/where to send the testimony (e.g., "to the committee chair's office" and that the specific email is in the committee notice).
13) recipient_information_urls: All official URLs the answer cites for recipient information.

STRICT RULES:
- Extract exactly what the answer states; do not infer or add new information.
- For each *_urls field, include only URLs explicitly present in the answer (plain links or markdown links). Ignore non-URL references.
- Prefer official Ohio government/legislative domains (e.g., ohiohouse.gov, legislature.ohio.gov, ohio.gov) if they are present in the answer; still include any other cited URLs as long as they are explicitly in the answer.
- If the answer lists multiple URLs for any requirement, include them all in the corresponding array.
"""


# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #
def sanitize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if isinstance(u, str):
            v = u.strip()
            if v and v not in cleaned:
                cleaned.append(v)
    return cleaned


def official_source_instruction() -> str:
    return (
        "Only count as supported if at least one cited page is an official Ohio government or legislative source "
        "(e.g., domains like ohiohouse.gov, legislature.ohio.gov, ohio.gov). If none of the given URLs are "
        "official or if the page content does not support the claim, mark it as NOT SUPPORTED. Rely on the webpage "
        "content (and screenshot) rather than your own knowledge. Allow reasonable phrasing variants (e.g., '24 hours', "
        "'one day', '24 hours before the hearing start time'; 'witness slip' vs 'committee witness information form')."
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_submission_timing_checks(evaluator: Evaluator, parent_node, data: TestimonyRequirementsExtraction) -> None:
    timing_root = evaluator.add_parallel(
        id="Submission_Timing",
        desc="Identification of time-related requirements for testimony submission",
        parent=parent_node,
        critical=True,
    )

    # Advance deadline node
    adv_node = evaluator.add_parallel(
        id="Advance_Deadline",
        desc="Correctly identifies that written testimony must be submitted 24 hours in advance of the committee hearing",
        parent=timing_root,
        critical=True,
    )

    # Leaf: Answer claims 24 hours (or close equivalents)
    adv_claim_leaf = evaluator.add_leaf(
        id="Advance_Deadline_Claimed",
        desc="Answer states that written testimony must be submitted at least 24 hours before the committee hearing",
        parent=adv_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that written testimony must be submitted at least 24 hours before the committee hearing.",
        node=adv_claim_leaf,
        additional_instruction=(
            "Judge only by the provided answer text. Accept equivalents like 'one day prior', "
            "'24 hours prior', or similar phrasing. If the answer instead states a different deadline "
            "(e.g., 48 hours) or omits the timing, mark as incorrect."
        ),
    )

    # Custom: URLs provided for this requirement
    adv_urls = sanitize_urls(data.advance_deadline_urls)
    evaluator.add_custom_node(
        result=len(adv_urls) > 0,
        id="Advance_Deadline_Sources_Provided",
        desc="At least one official source URL is provided in the answer for the advance deadline requirement",
        parent=adv_node,
        critical=True,
    )

    # Leaf: Supported by official source(s)
    adv_supported_leaf = evaluator.add_leaf(
        id="Advance_Deadline_Supported",
        desc="The 24-hour advance submission requirement is supported by official sources",
        parent=adv_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Written testimony to an Ohio House standing committee must be submitted at least 24 hours before the hearing.",
        node=adv_supported_leaf,
        sources=adv_urls,
        additional_instruction=official_source_instruction()
        + " If a source uses business-day phrasing that clearly implies 24 hours (e.g., 'one business day prior'), treat as acceptable.",
    )

    # Notice timing node
    notice_node = evaluator.add_parallel(
        id="Notice_Timing",
        desc="Identifies that committee notices are typically posted on Thursday or Friday of the week before the hearing",
        parent=timing_root,
        critical=True,
    )

    # Leaf: Answer claims the Thursday/Friday timing
    notice_claim_leaf = evaluator.add_leaf(
        id="Notice_Timing_Claimed",
        desc="Answer states that committee hearing notices are typically posted on Thursday or Friday of the week before the hearing",
        parent=notice_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that committee hearing notices are typically posted on Thursday or Friday of the week before the hearing.",
        node=notice_claim_leaf,
        additional_instruction=(
            "Judge only by the answer content. Accept small paraphrases that clearly mean 'Thursday or Friday of the prior week'. "
            "If the answer gives a different timing pattern or omits this, mark as incorrect."
        ),
    )

    # Custom: URLs provided for notice timing
    notice_urls = sanitize_urls(data.notice_timing_urls)
    evaluator.add_custom_node(
        result=len(notice_urls) > 0,
        id="Notice_Timing_Sources_Provided",
        desc="At least one official source URL is provided in the answer for the committee notice timing",
        parent=notice_node,
        critical=True,
    )

    # Leaf: Supported by official source(s)
    notice_supported_leaf = evaluator.add_leaf(
        id="Notice_Timing_Supported",
        desc="The Thursday/Friday prior week posting pattern is supported by official sources",
        parent=notice_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Committee hearing notices are typically posted on Thursday or Friday of the week before the hearing.",
        node=notice_supported_leaf,
        sources=notice_urls,
        additional_instruction=official_source_instruction()
        + " Allow close paraphrases that indicate notices are posted late in the week before the hearing.",
    )


async def build_required_documents_checks(evaluator: Evaluator, parent_node, data: TestimonyRequirementsExtraction) -> None:
    docs_root = evaluator.add_parallel(
        id="Required_Documents",
        desc="Identification of all mandatory documents that must be submitted",
        parent=parent_node,
        critical=True,
    )

    # Witness slip node
    ws_node = evaluator.add_parallel(
        id="Witness_Slip",
        desc="Identifies that a witness slip is required and that the Ohio House uses a standardized witness slip form",
        parent=docs_root,
        critical=True,
    )

    # Leaf: Answer claims witness slip required (and standardized form)
    ws_claim_leaf = evaluator.add_leaf(
        id="Witness_Slip_Claimed",
        desc="Answer states a witness slip is required and a standardized witness slip form is used",
        parent=ws_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that a witness slip is required for submitting testimony to an Ohio House standing committee and that a standardized witness slip form is used.",
        node=ws_claim_leaf,
        additional_instruction=(
            "Judge only by the answer content. Accept 'witness slip', 'witness form', or 'committee witness information form' "
            "as equivalent. If the answer does not clearly assert the requirement and standardized form, mark as incorrect."
        ),
    )

    # Custom: URLs provided for witness slip requirement/form usage
    ws_urls = sanitize_urls(data.witness_slip_urls)
    evaluator.add_custom_node(
        result=len(ws_urls) > 0,
        id="Witness_Slip_Sources_Provided",
        desc="At least one official source URL is provided for the witness slip requirement/standardized form",
        parent=ws_node,
        critical=True,
    )

    # Leaf: Supported by official source(s)
    ws_supported_leaf = evaluator.add_leaf(
        id="Witness_Slip_Supported",
        desc="The witness slip requirement and standardized form usage are supported by official sources",
        parent=ws_node,
        critical=True,
    )
    await evaluator.verify(
        claim="An Ohio House standing committee requires a witness slip for testimony submissions and uses a standardized witness slip form.",
        node=ws_supported_leaf,
        sources=ws_urls,
        additional_instruction=official_source_instruction()
        + " Accept equivalent phrasing such as 'committee witness information form' or 'witness information form'.",
    )

    # Custom: Form URL(s) provided (where to obtain the form)
    form_urls = sanitize_urls(data.witness_slip_form_urls)
    evaluator.add_custom_node(
        result=len(form_urls) > 0,
        id="Witness_Slip_Form_URLs_Provided",
        desc="The answer includes at least one URL where the standardized witness slip form can be obtained",
        parent=ws_node,
        critical=True,
    )

    # Leaf: The provided form URL is an official source and is indeed the form or the official page to obtain it
    ws_form_official_leaf = evaluator.add_leaf(
        id="Witness_Slip_Form_Official",
        desc="Provided witness slip form URL points to the official standardized form or the official page to obtain it",
        parent=ws_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This URL is the official Ohio House standardized witness slip form or the official page to obtain it.",
        node=ws_form_official_leaf,
        sources=form_urls,
        additional_instruction=official_source_instruction()
        + " If the page provides the form (PDF/online) or clearly links to the official standardized form, treat as supported.",
    )

    # Written testimony PDF node
    pdf_node = evaluator.add_parallel(
        id="Written_Testimony_PDF",
        desc="Identifies that written testimony must be submitted as a PDF document",
        parent=docs_root,
        critical=True,
    )

    # Leaf: Answer claims PDF format
    pdf_claim_leaf = evaluator.add_leaf(
        id="Written_Testimony_PDF_Claimed",
        desc="Answer states that written testimony must be submitted as a PDF document",
        parent=pdf_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that written testimony must be submitted as a PDF document.",
        node=pdf_claim_leaf,
        additional_instruction="Judge only by the answer text. Accept phrasing like 'submit testimony in PDF format' or 'PDF attachment required'.",
    )

    # Custom: URLs provided for PDF requirement
    pdf_urls = sanitize_urls(data.testimony_format_urls)
    evaluator.add_custom_node(
        result=len(pdf_urls) > 0,
        id="Written_Testimony_PDF_Sources_Provided",
        desc="At least one official source URL is provided for the PDF requirement",
        parent=pdf_node,
        critical=True,
    )

    # Leaf: Supported by official source(s)
    pdf_supported_leaf = evaluator.add_leaf(
        id="Written_Testimony_PDF_Supported",
        desc="The PDF format requirement is supported by official sources",
        parent=pdf_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Written-only testimony for Ohio House standing committees must be submitted as a PDF document.",
        node=pdf_supported_leaf,
        sources=pdf_urls,
        additional_instruction=official_source_instruction(),
    )


async def build_submission_protocol_checks(evaluator: Evaluator, parent_node, data: TestimonyRequirementsExtraction) -> None:
    proto_root = evaluator.add_parallel(
        id="Submission_Protocol",
        desc="Identification of how and where to submit testimony materials",
        parent=parent_node,
        critical=True,
    )

    # Submission method node (email)
    method_node = evaluator.add_parallel(
        id="Submission_Method",
        desc="Identifies that testimony must be submitted via email",
        parent=proto_root,
        critical=True,
    )

    # Leaf: Answer claims email submission
    method_claim_leaf = evaluator.add_leaf(
        id="Submission_Method_Claimed",
        desc="Answer states that testimony must be submitted via email",
        parent=method_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that testimony must be submitted via email.",
        node=method_claim_leaf,
        additional_instruction="Judge only by the answer text. Accept 'email' or 'emailed to the committee' as equivalent.",
    )

    # Custom: URLs provided for submission method
    method_urls = sanitize_urls(data.submission_method_urls)
    evaluator.add_custom_node(
        result=len(method_urls) > 0,
        id="Submission_Method_Sources_Provided",
        desc="At least one official source URL is provided for the submission method",
        parent=method_node,
        critical=True,
    )

    # Leaf: Supported by official source(s)
    method_supported_leaf = evaluator.add_leaf(
        id="Submission_Method_Supported",
        desc="The email submission method is supported by official sources",
        parent=method_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Written testimony for Ohio House standing committees must be submitted via email.",
        node=method_supported_leaf,
        sources=method_urls,
        additional_instruction=official_source_instruction()
        + " Accept phrasing like 'email your testimony' or 'send testimony by email'.",
    )

    # Recipient information node
    recip_node = evaluator.add_parallel(
        id="Recipient_Information",
        desc="Identifies that testimony must be sent to the committee chair's office, with the specific email address provided in the committee notice",
        parent=proto_root,
        critical=True,
    )

    # Leaf: Answer claims correct recipient information
    recip_claim_leaf = evaluator.add_leaf(
        id="Recipient_Information_Claimed",
        desc="Answer states testimony should be sent to the committee chair's office and that the specific email address is provided in the committee notice",
        parent=recip_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that testimony should be sent to the committee chair's office and that the specific email address to use is provided in the official committee notice.",
        node=recip_claim_leaf,
        additional_instruction="Judge only by the answer text. Accept equivalents like 'chair's office' or 'as listed in the committee notice'.",
    )

    # Custom: URLs provided for recipient info
    recip_urls = sanitize_urls(data.recipient_information_urls)
    evaluator.add_custom_node(
        result=len(recip_urls) > 0,
        id="Recipient_Information_Sources_Provided",
        desc="At least one official source URL is provided for recipient details",
        parent=recip_node,
        critical=True,
    )

    # Leaf: Supported by official source(s)
    recip_supported_leaf = evaluator.add_leaf(
        id="Recipient_Information_Supported",
        desc="Recipient requirements (chair's office / notice-provided email) are supported by official sources",
        parent=recip_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Written testimony should be sent to the committee chair's office or to the email address specified in the official committee notice.",
        node=recip_supported_leaf,
        sources=recip_urls,
        additional_instruction=official_source_instruction()
        + " Accept phrasing that indicates the notice provides the correct recipient email (chair/committee).",
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

    # Create a critical top-level node reflecting the rubric root
    rubric_root = evaluator.add_parallel(
        id="Ohio_House_Testimony_Requirements",
        desc="Complete identification of procedural requirements for submitting written-only testimony to an Ohio House standing committee hearing",
        parent=root,
        critical=True,
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=TestimonyRequirementsExtraction,
        extraction_name="requirements_extraction",
    )

    # Build verification subtrees
    await build_submission_timing_checks(evaluator, rubric_root, extracted)
    await build_required_documents_checks(evaluator, rubric_root, extracted)
    await build_submission_protocol_checks(evaluator, rubric_root, extracted)

    return evaluator.get_summary()