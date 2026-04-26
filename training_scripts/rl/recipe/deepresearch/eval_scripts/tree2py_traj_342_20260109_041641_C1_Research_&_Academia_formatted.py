import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "chi_2025_info"
TASK_DESCRIPTION = (
    "I am planning to attend the CHI 2025 conference in person. Please provide the following information from official "
    "CHI 2025 conference sources: (1) The name of the venue where the in-person conference will be held, (2) The name "
    "of the opening keynote speaker, and (3) The name of the closing keynote speaker. For each piece of information, "
    "include a direct link to the official CHI 2025 page where this information is published."
)


# -----------------------------------------------------------------------------
# Data models for extraction
# -----------------------------------------------------------------------------
class VenueInfo(BaseModel):
    name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class KeynoteInfo(BaseModel):
    name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CHI2025Extraction(BaseModel):
    venue: Optional[VenueInfo] = None
    opening_keynote: Optional[KeynoteInfo] = None
    closing_keynote: Optional[KeynoteInfo] = None


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_chi2025_info() -> str:
    return """
    Extract the CHI 2025 in-person conference information as presented in the answer. You must not invent any details.
    Specifically extract:
    - venue.name: The name of the venue where the in-person CHI 2025 conference will be held (e.g., a convention center).
    - venue.urls: All URLs in the answer that directly publish the venue information for CHI 2025.
    - opening_keynote.name: The name of the opening keynote speaker for CHI 2025.
    - opening_keynote.urls: All URLs in the answer that directly publish the opening keynote speaker information.
    - closing_keynote.name: The name of the closing keynote speaker for CHI 2025.
    - closing_keynote.urls: All URLs in the answer that directly publish the closing keynote speaker information.

    Rules for URL extraction:
    - Extract only URLs that are explicitly present in the answer text. Do not infer or fabricate any URLs.
    - Include full URLs. If a URL is missing a protocol, prepend 'http://'.
    - Do not deduplicate; return all URLs that are tied to each specific item (venue, opening keynote, closing keynote) in the answer.
    - The answer may include both official and non-official URLs. Extract them all as given.

    If any field is not present in the answer, set it to null (for a single value) or [] (for an array).
    Return a single JSON object with keys: venue, opening_keynote, closing_keynote.
    """


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
ALLOWED_OFFICIAL_DOMAINS = [
    "chi2025.acm.org",
    "sigchi.org",
]


def is_official_chi2025_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        host = (parsed.netloc or "").lower()
        if not host:
            return False
        # Allow main domain and subdomains for sigchi.org; only chi2025.acm.org for CHI 2025 site
        if host == "chi2025.acm.org":
            return True
        if host.endswith(".sigchi.org") or host == "sigchi.org" or host == "www.sigchi.org":
            return True
        return False
    except Exception:
        return False


def filter_official_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u if u.startswith("http://") or u.startswith("https://") else f"http://{u}" for u in urls if is_official_chi2025_url(u)]


def non_empty_str(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


# -----------------------------------------------------------------------------
# Verification builders
# -----------------------------------------------------------------------------
async def build_venue_section(evaluator: Evaluator, parent_node, extraction: CHI2025Extraction) -> None:
    venue_info = extraction.venue or VenueInfo()
    official_urls = filter_official_urls(venue_info.urls)

    venue_node = evaluator.add_parallel(
        id="CHI_2025_Venue_Info",
        desc="Venue information is provided for the CHI 2025 in-person conference and meets all stated constraints.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Venue name provided
    name_provided_result = non_empty_str(venue_info.name)
    name_node = evaluator.add_custom_node(
        result=name_provided_result,
        id="Venue_Name_Provided",
        desc="Provides the venue name for the CHI 2025 in-person conference.",
        parent=venue_node,
        critical=True,
    )

    # Leaf: Direct official link exists (on chi2025.acm.org or sigchi.org)
    has_official_link = len(official_urls) > 0
    direct_link_node = evaluator.add_custom_node(
        result=has_official_link,
        id="Venue_Direct_Official_Link",
        desc="Includes a direct URL to the official page where the venue information is published, and the URL is on chi2025.acm.org or sigchi.org.",
        parent=venue_node,
        critical=True,
    )

    # Leaf: Venue located in Yokohama, Japan (verify against the official URL(s))
    located_node = evaluator.add_leaf(
        id="Venue_Located_In_Yokohama_Japan",
        desc="Indicates (and is consistent with the cited official source) that the venue is located in Yokohama, Japan.",
        parent=venue_node,
        critical=True,
    )
    claim_loc = "The CHI 2025 in-person conference venue is located in Yokohama, Japan."
    await evaluator.verify(
        claim=claim_loc,
        node=located_node,
        sources=official_urls if official_urls else None,
        additional_instruction="Judge strictly based on the cited official CHI 2025 page(s). Accept minor variants like 'Yokohama' or 'Yokohama, Kanagawa, Japan'. If no official page is provided, consider the claim unsupported.",
        extra_prerequisites=[name_node, direct_link_node],
    )

    # Record some custom info for transparency
    evaluator.add_custom_info(
        info={"extracted_venue_name": venue_info.name, "venue_urls_all": venue_info.urls, "venue_urls_official": official_urls},
        info_type="extraction_debug",
        info_name="venue_debug"
    )


async def build_keynote_section(
    evaluator: Evaluator,
    parent_node,
    section_id: str,
    section_desc: str,
    name_leaf_id: str,
    announced_leaf_id: str,
    link_leaf_id: str,
    role_phrase: str,
    keynote: Optional[KeynoteInfo],
) -> None:
    keynote_data = keynote or KeynoteInfo()
    official_urls = filter_official_urls(keynote_data.urls)

    kn_node = evaluator.add_parallel(
        id=section_id,
        desc=section_desc,
        parent=parent_node,
        critical=True,
    )

    # Leaf: Name provided
    name_provided = non_empty_str(keynote_data.name)
    name_node = evaluator.add_custom_node(
        result=name_provided,
        id=name_leaf_id,
        desc=f"Provides the name of the {role_phrase} keynote speaker for CHI 2025.",
        parent=kn_node,
        critical=True,
    )

    # Leaf: Direct official link exists (on chi2025.acm.org or sigchi.org)
    has_official_link = len(official_urls) > 0
    link_node = evaluator.add_custom_node(
        result=has_official_link,
        id=link_leaf_id,
        desc=f"Includes a direct URL to the official page where the {role_phrase} keynote speaker information is published, and the URL is on chi2025.acm.org or sigchi.org.",
        parent=kn_node,
        critical=True,
    )

    # Leaf: Officially announced (verify against the official URL(s))
    announced_node = evaluator.add_leaf(
        id=announced_leaf_id,
        desc=f"The cited official source explicitly identifies the provided person as the {role_phrase} keynote speaker for CHI 2025.",
        parent=kn_node,
        critical=True,
    )
    # Use a robust claim phrasing; allow minor text variations in the page (e.g., 'opening keynote', 'opening plenary')
    speaker_name = keynote_data.name or ""
    claim = f"The CHI 2025 {role_phrase} keynote speaker is {speaker_name}."
    await evaluator.verify(
        claim=claim,
        node=announced_node,
        sources=official_urls if official_urls else None,
        additional_instruction=f"Verify that the cited official page(s) explicitly associate {speaker_name} with the {role_phrase} keynote at CHI 2025. Accept minor wording variants like '{role_phrase} keynote', '{role_phrase} plenary', or similar. If no official page is provided, consider the claim unsupported.",
        extra_prerequisites=[name_node, link_node],
    )

    # Record some custom info for transparency
    evaluator.add_custom_info(
        info={
            "extracted_name": keynote_data.name,
            "urls_all": keynote_data.urls,
            "urls_official": official_urls,
            "role": role_phrase
        },
        info_type="extraction_debug",
        info_name=f"{section_id}_debug"
    )


# -----------------------------------------------------------------------------
# Main evaluation entry
# -----------------------------------------------------------------------------
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
    # Initialize evaluator/root
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_chi2025_info(),
        template_class=CHI2025Extraction,
        extraction_name="chi2025_info_extraction",
    )

    # Build the "CHI_2025_Information" critical aggregator node
    chi_root = evaluator.add_parallel(
        id="CHI_2025_Information",
        desc="Verify that all required CHI 2025 in-person conference information is provided and each item is supported by an official CHI 2025 source link.",
        parent=root,
        critical=True,
    )

    # Venue section
    await build_venue_section(evaluator, chi_root, extraction)

    # Opening keynote section
    await build_keynote_section(
        evaluator=evaluator,
        parent_node=chi_root,
        section_id="Opening_Keynote_Info",
        section_desc="Opening keynote speaker information is provided for CHI 2025 and is supported by an official CHI 2025 source link.",
        name_leaf_id="Opening_Keynote_Speaker_Name_Provided",
        announced_leaf_id="Opening_Keynote_Is_Officially_Announced",
        link_leaf_id="Opening_Keynote_Direct_Official_Link",
        role_phrase="opening",
        keynote=(extraction.opening_keynote if extraction and extraction.opening_keynote else KeynoteInfo()),
    )

    # Closing keynote section
    await build_keynote_section(
        evaluator=evaluator,
        parent_node=chi_root,
        section_id="Closing_Keynote_Info",
        section_desc="Closing keynote speaker information is provided for CHI 2025 and is supported by an official CHI 2025 source link.",
        name_leaf_id="Closing_Keynote_Speaker_Name_Provided",
        announced_leaf_id="Closing_Keynote_Is_Officially_Announced",
        link_leaf_id="Closing_Keynote_Direct_Official_Link",
        role_phrase="closing",
        keynote=(extraction.closing_keynote if extraction and extraction.closing_keynote else KeynoteInfo()),
    )

    # Return evaluation summary
    return evaluator.get_summary()