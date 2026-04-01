import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "federal_reserve"
TASK_DESCRIPTION = """
Find the official Federal Reserve webpage links as well as PDF links for all speeches delivered live in person by Federal Reserve Chair Jerome H. Powell, between September 1, 2024, and December 31, 2024. For each speech, find a Reuters news article that specifically covers that particular speech.
"""

# Ground truth data for the two expected speeches
GROUND_TRUTH_SPEECHES = [
    {
        "date": "September 30, 2024",
        "title": "Economic Outlook",
        "webpage_link": "https://www.federalreserve.gov/newsevents/speech/powell20240930a.htm",
        "pdf_link": "https://www.federalreserve.gov/newsevents/speech/files/powell20240930a.pdf",
    },
    {
        "date": "November 14, 2024",
        "title": "Economic Outlook",
        "webpage_link": "https://www.federalreserve.gov/newsevents/speech/powell20241114a.htm",
        "pdf_link": "https://www.federalreserve.gov/newsevents/speech/files/powell20241114a.pdf",
    }
]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SpeechInfo(BaseModel):
    """Information about a single speech by Jerome Powell"""
    webpage_link: Optional[str] = None
    pdf_link: Optional[str] = None
    reuters_link: Optional[str] = None


class ExtractedSpeeches(BaseModel):
    """Speeches extracted for each ground truth speech"""
    september_30_speech: Optional[SpeechInfo] = None
    november_14_speech: Optional[SpeechInfo] = None


def prompt_extract_speeches() -> str:
    """Prompt to extract speech information targeting the two ground truth speeches"""
    return """
    Extract information for the two specific Jerome Powell speeches that occurred in the specified date range:

    1. Speech on September 30, 2024 (Economic Outlook)
    2. Speech on November 14, 2024 (Economic Outlook)

    For each speech (if mentioned in the answer), extract:
    - The Federal Reserve webpage link for the speech
    - The PDF link for the speech  
    - The Reuters news article link covering the speech

    IMPORTANT EXTRACTION RULE: If the answer provides multiple speeches, only extract from the FIRST TWO speeches mentioned in the answer, regardless of their dates or content. Completely ignore any speeches mentioned after the first two, even if they match the target dates (September 30, 2024 or November 14, 2024). This rule is designed to penalize answers that provide excessive or incorrect information.

    Matching process:
    1. Identify the first speech mentioned in the answer - try to match it to either September 30, 2024 or November 14, 2024 speech
    2. Identify the second speech mentioned in the answer - try to match it to the remaining target speech
    3. Ignore all speeches beyond the first two, regardless of their accuracy

    If a speech from the first two cannot be matched to either target date, or if any information is missing, set the corresponding field to null.
    If an entire target speech cannot be matched from the first two speeches in the answer, set the whole speech object to null.

    Return the information in the specified format with september_30_speech and november_14_speech fields.
    """


# --------------------------------------------------------------------------- #
# Helper functions for URL comparison                                         #
# --------------------------------------------------------------------------- #
def normalize_url(url: str) -> str:
    """Normalize URL for comparison by removing trailing slash and converting to lowercase"""
    if not url:
        return ""
    normalized = url.lower().strip()
    if normalized.endswith('/'):
        normalized = normalized[:-1]
    return normalized


def urls_match(url1: str, url2: str) -> bool:
    """Check if two URLs match with minor variations allowed"""
    if not url1 or not url2:
        return False
    return normalize_url(url1) == normalize_url(url2)


# --------------------------------------------------------------------------- #
# Complete Speech Verification                                               #
# --------------------------------------------------------------------------- #
async def verify_complete_speech(
        evaluator: Evaluator,
        parent_node,
        speech: Optional[SpeechInfo],
        gt_speech: Dict,
        speech_label: str,
) -> None:
    """
    Verify all aspects of a single speech with parallel scoring.
    Two main components: Federal Reserve links and Reuters news.
    """
    # Create a parallel node for this speech
    speech_node = evaluator.add_parallel(
        id=f"speech_{speech_label}",
        desc=f"All required information is provided for the {speech_label} speech",
        parent=parent_node,
        critical=False,
    )

    # If speech is None, pad with empty SpeechInfo
    if speech is None:
        speech = SpeechInfo()

    # Create parent node for Federal Reserve links
    fed_links_node = evaluator.add_parallel(
        id=f"federal_reserve_links_{speech_label}",
        desc=f"Federal Reserve webpage and PDF links are provided for the {speech_label} speech",
        parent=speech_node,
        critical=False,
    )

    # Verify webpage link with wrapper node
    webpage_wrapper = evaluator.add_parallel(
        id=f"webpage_verification_{speech_label}",
        desc=f"Webpage verification for {speech_label} speech",
        parent=fed_links_node,
        critical=False,  # Non-critical to allow partial scoring
    )

    webpage_exists = evaluator.add_custom_node(
        result=bool(speech.webpage_link),
        id=f"webpage_exists_{speech_label}",
        desc=f"Webpage link exists for {speech_label} speech",
        parent=webpage_wrapper,
        critical=True
    )

    webpage_node = evaluator.add_leaf(
        id=f"webpage_link_{speech_label}",
        desc=f"Federal Reserve webpage link is correctly provided for the {speech_label} speech",
        parent=webpage_wrapper,
        critical=True,
    )

    # First try shortcut matching
    if speech.webpage_link and urls_match(speech.webpage_link, gt_speech["webpage_link"]):
        webpage_node.score = 1.0
        webpage_node.status = "passed"
    else:
        # Verify by content
        claim = f"This webpage contains the text of Jerome Powell's {gt_speech['title']} speech delivered on {gt_speech['date']}"
        await evaluator.verify(
            claim=claim,
            node=webpage_node,
            sources=speech.webpage_link,
        )

    # Verify PDF link with wrapper node
    pdf_wrapper = evaluator.add_parallel(
        id=f"pdf_verification_{speech_label}",
        desc=f"PDF verification for {speech_label} speech",
        parent=fed_links_node,
        critical=False,  # Non-critical to allow partial scoring
    )

    pdf_exists = evaluator.add_custom_node(
        result=bool(speech.pdf_link),
        id=f"pdf_exists_{speech_label}",
        desc=f"PDF link exists for {speech_label} speech",
        parent=pdf_wrapper,
        critical=True
    )

    pdf_node = evaluator.add_leaf(
        id=f"pdf_link_{speech_label}",
        desc=f"PDF link is correctly provided for the {speech_label} speech",
        parent=pdf_wrapper,
        critical=True,
    )

    # First try shortcut matching
    if speech.pdf_link and urls_match(speech.pdf_link, gt_speech["pdf_link"]):
        pdf_node.score = 1.0
        pdf_node.status = "passed"
    else:
        # Verify by content
        claim = f"This URL provides a PDF document or full text document containing Jerome Powell's {gt_speech['title']} speech from {gt_speech['date']}. The document should contain the complete speech text, similar to what would be found in a PDF document."
        await evaluator.verify(
            claim=claim,
            node=pdf_node,
            sources=speech.pdf_link,
        )

    # Verify Reuters news with wrapper node - directly under speech_node
    reuters_wrapper = evaluator.add_parallel(
        id=f"reuters_verification_{speech_label}",
        desc=f"Reuters verification for {speech_label} speech",
        parent=speech_node,
        critical=False,  # Non-critical to allow partial scoring
    )

    reuters_exists = evaluator.add_custom_node(
        result=bool(speech.reuters_link),
        id=f"reuters_exists_{speech_label}",
        desc=f"Reuters link exists for {speech_label} speech",
        parent=reuters_wrapper,
        critical=True
    )

    reuters_node = evaluator.add_leaf(
        id=f"reuters_news_{speech_label}",
        desc=f"Reuters news article covering the {speech_label} speech is correctly provided",
        parent=reuters_wrapper,
        critical=True,
    )

    claim = f"This is a Reuters news article that specifically covers Jerome Powell's {gt_speech['title']} speech delivered on {gt_speech['date']}"
    await evaluator.verify(
        claim=claim,
        node=reuters_node,
        sources=speech.reuters_link,
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
    Evaluate a single answer for the find_federal_reserve task.
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

    # Extract speeches information
    extracted_speeches = await evaluator.extract(
        prompt=prompt_extract_speeches(),
        template_class=ExtractedSpeeches,
        extraction_name="extracted_speeches"
    )

    # Add ground truth info
    evaluator.add_ground_truth({"ground_truth_speeches": GROUND_TRUTH_SPEECHES})

    # Verify September 30, 2024 speech
    await verify_complete_speech(
        evaluator,
        root,
        extracted_speeches.september_30_speech,
        GROUND_TRUTH_SPEECHES[0],
        "september_30_2024"
    )

    # Verify November 14, 2024 speech
    await verify_complete_speech(
        evaluator,
        root,
        extracted_speeches.november_14_speech,
        GROUND_TRUTH_SPEECHES[1],
        "november_14_2024"
    )

    # Return structured result
    return evaluator.get_summary()