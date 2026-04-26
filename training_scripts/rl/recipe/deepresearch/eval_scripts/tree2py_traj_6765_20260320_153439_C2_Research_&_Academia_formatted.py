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
TASK_ID = "warner_prize_2026"
TASK_DESCRIPTION = """
The American Astronomical Society announced the recipient of the 2026 Helen B. Warner Prize for Astronomy in January 2026. Identify the recipient of this award, provide their current institutional affiliation (including both the university and the academic department), and state the official award citation describing the research contribution for which they received the prize. Include reference URLs for your answer.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RecipientInfo(BaseModel):
    name: Optional[str] = None
    university: Optional[str] = None
    department: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class CitationInfo(BaseModel):
    text: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class PrizeExtraction(BaseModel):
    recipient: Optional[RecipientInfo] = None
    citation: Optional[CitationInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_warner_2026() -> str:
    return """
    Extract the structured information about the 2026 Helen B. Warner Prize for Astronomy from the provided answer text.

    Required fields:
    - recipient:
        - name: The full name of the 2026 Helen B. Warner Prize for Astronomy recipient, exactly as written in the answer.
        - university: The recipient’s current university/institution as stated in the answer (e.g., "University of X", "Harvard University").
        - department: The recipient’s academic department or equivalent unit as stated in the answer (e.g., "Department of Astronomy", "School of Physics and Astronomy"). Use the wording from the answer; do not invent.
        - reference_urls: All URLs (if any) that the answer cites to support the recipient identification and/or affiliation. Include only URLs explicitly shown in the answer. Return an empty list if none.
    - citation:
        - text: The official award citation describing why the recipient received the prize, as presented in the answer. If a direct quote is shown, extract the quote text verbatim (without the surrounding quotation marks). If not quoted, extract the exact sentence/phrase labeled or described as the official citation. If missing, set to null.
        - reference_urls: All URLs explicitly cited in the answer that support the official citation. Return an empty list if none.

    Rules:
    - Do not invent any information not present in the answer.
    - If any field is missing, set it to null (or [] for lists).
    - For URLs: extract only valid-looking URLs explicitly present in the answer (including markdown links). If a URL is missing a protocol, prepend "http://".
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip()
    return u.startswith("http://") or u.startswith("https://")


def _filter_valid_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u.strip() for u in urls if _is_valid_url(u)]


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def build_and_verify_recipient_section(
    evaluator: Evaluator,
    parent,
    recipient: Optional[RecipientInfo],
) -> None:
    # Create the Recipient_Details aggregate node (critical)
    rec_node = evaluator.add_parallel(
        id="Recipient_Details",
        desc="Correct identification of the award recipient including their name and current institutional affiliation",
        parent=parent,
        critical=True,
    )

    # Prepare data
    name = recipient.name if recipient else None
    university = recipient.university if recipient else None
    department = recipient.department if recipient else None
    recipient_urls_all = _filter_valid_urls(recipient.reference_urls if recipient else [])

    # Recipient_Reference_URL (existence/validity check)
    has_recipient_url = len(recipient_urls_all) > 0
    evaluator.add_custom_node(
        result=has_recipient_url,
        id="Recipient_Reference_URL",
        desc="A valid reference URL supporting the recipient identification and institutional affiliation is provided",
        parent=rec_node,
        critical=True,
    )

    # Recipient_Name (verify against cited URLs)
    name_leaf = evaluator.add_leaf(
        id="Recipient_Name",
        desc="The full name of the 2026 Helen B. Warner Prize recipient is provided",
        parent=rec_node,
        critical=True,
    )
    name_claim = f"{name} is the recipient of the 2026 Helen B. Warner Prize for Astronomy awarded by the American Astronomical Society."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=recipient_urls_all,
        additional_instruction=(
            "Verify that the provided webpage(s) explicitly identify this person as the 2026 recipient "
            "of the Helen B. Warner Prize for Astronomy. Allow minor variations in formatting (e.g., titles, middle initials). "
            "Ensure the year is 2026."
        ),
    )

    # Institutional_Affiliation (aggregate)
    aff_node = evaluator.add_parallel(
        id="Institutional_Affiliation",
        desc="The recipient's current institutional affiliation including both university and department",
        parent=rec_node,
        critical=True,
    )

    # University (verify against cited URLs)
    uni_leaf = evaluator.add_leaf(
        id="University",
        desc="The correct university where the recipient currently holds a position is identified",
        parent=aff_node,
        critical=True,
    )
    uni_claim = f"{name} currently holds a position at {university}."
    await evaluator.verify(
        claim=uni_claim,
        node=uni_leaf,
        sources=recipient_urls_all,
        additional_instruction=(
            "Verify that the cited page(s) explicitly state the person's current university/institution. "
            "Minor naming variants (e.g., abbreviations) are acceptable. If multiple affiliations are listed, "
            "it should still correctly include the stated university."
        ),
    )

    # Department (verify against cited URLs)
    dept_leaf = evaluator.add_leaf(
        id="Department",
        desc="The correct academic department within the university is specified",
        parent=aff_node,
        critical=True,
    )
    dept_claim = f"{name}'s academic department is {department} at {university}."
    await evaluator.verify(
        claim=dept_claim,
        node=dept_leaf,
        sources=recipient_urls_all,
        additional_instruction=(
            "Verify that the cited page(s) explicitly mention the person's department (or functionally equivalent unit, "
            "e.g., 'School of X', 'Institute of Y'). Allow minor wording variants like 'Dept.' vs 'Department'."
        ),
    )


async def build_and_verify_citation_section(
    evaluator: Evaluator,
    parent,
    citation: Optional[CitationInfo],
    recipient: Optional[RecipientInfo],
) -> None:
    # Create the Award_Citation aggregate node (critical)
    cit_node = evaluator.add_parallel(
        id="Award_Citation",
        desc="The official award citation describing why the recipient received the prize",
        parent=parent,
        critical=True,
    )

    # Prepare data
    citation_text = citation.text if citation else None
    citation_urls_all = _filter_valid_urls(citation.reference_urls if citation else [])

    # Citation_Reference_URL (existence/validity check)
    has_citation_url = len(citation_urls_all) > 0
    evaluator.add_custom_node(
        result=has_citation_url,
        id="Citation_Reference_URL",
        desc="A valid reference URL supporting the award citation is provided",
        parent=cit_node,
        critical=True,
    )

    # Citation_Text (verify accuracy against cited URLs)
    citation_leaf = evaluator.add_leaf(
        id="Citation_Text",
        desc="The official award citation text describing the recipient's research contribution is provided accurately",
        parent=cit_node,
        critical=True,
    )

    # Build the citation claim; allow minor paraphrase/format differences.
    # If recipient name is available, it can help the judge reason, but focus on citation text matching the page(s).
    recipient_name = recipient.name if recipient else "the recipient"
    cit_claim = (
        f"The official award citation for the 2026 Helen B. Warner Prize for Astronomy, as announced by the American "
        f"Astronomical Society, is: \"{citation_text}\" (for {recipient_name})."
    )

    await evaluator.verify(
        claim=cit_claim,
        node=citation_leaf,
        sources=citation_urls_all,
        additional_instruction=(
            "Check that the webpage(s) provide the official award citation text (often beginning with 'for ...'). "
            "A close textual match is expected; allow minor formatting or punctuation differences. "
            "If the page provides a clearly labeled 'citation' or 'award citation', that text should match the provided citation."
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
    Evaluate an answer for the 2026 Helen B. Warner Prize for Astronomy task.
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
        default_model=model,
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_warner_2026(),
        template_class=PrizeExtraction,
        extraction_name="warner_2026_extraction",
    )

    # Record a concise summary of extracted values
    evaluator.add_custom_info(
        info={
            "recipient": (extracted.recipient.dict() if extracted.recipient else None),
            "citation": (extracted.citation.dict() if extracted.citation else None),
        },
        info_type="extracted_summary",
        info_name="extracted_summary",
    )

    # Build the top-level critical node (as per rubric)
    main_node = evaluator.add_parallel(
        id="Helen_B_Warner_Prize_2026_Information",
        desc="Complete and accurate information about the 2026 Helen B. Warner Prize recipient and their research recognition",
        parent=root,
        critical=True,
    )

    # Recipient details branch (critical)
    await build_and_verify_recipient_section(
        evaluator=evaluator,
        parent=main_node,
        recipient=extracted.recipient,
    )

    # Award citation branch (critical)
    await build_and_verify_citation_section(
        evaluator=evaluator,
        parent=main_node,
        citation=extracted.citation,
        recipient=extracted.recipient,
    )

    # Return final summary
    return evaluator.get_summary()