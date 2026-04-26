import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cs_conf_2025_na"
TASK_DESCRIPTION = (
    "Identify one major computer science conference scheduled for 2025 in North America that meets the following "
    "requirements: The conference must focus on computer science topics, specifically in artificial intelligence, "
    "machine learning, computer vision, or data science. The conference must be held in the United States or Canada. "
    "The conference must be recognized as a top-tier venue with a CORE A* or A ranking (or equivalent recognition in its field). "
    "The conference proceedings must be indexed in Scopus and/or IEEE Xplore. The paper submission deadline must be after January 15, 2025. "
    "The conference must publish official proceedings. For the identified conference, provide the following information: "
    "(1) Conference name and official website URL, (2) Specific field/topic area, (3) City and venue name (convention center or facility), "
    "(4) Conference dates (start and end dates), (5) Evidence of ranking/tier status, (6) Indexing information (Scopus and/or IEEE Xplore), "
    "(7) Paper page limit for full submissions (excluding references), (8) Abstract word count limit, (9) Paper submission deadline, "
    "(10) Early bird registration deadline, (11) Accepted presentation formats (oral, poster, etc.), (12) Poster dimensions (if posters are accepted), "
    "(13) Session organization format, (14) Proceedings publication details."
)

# Business rules and guidance
DEADLINE_THRESHOLD_TEXT = "January 15, 2025"
ALLOWED_COUNTRIES = {"United States", "USA", "U.S.", "U.S.A.", "US", "United States of America", "Canada"}
ALLOWED_TOPIC_KEYWORDS = ["artificial intelligence", "ai", "machine learning", "ml", "computer vision", "cv", "data science", "data mining", "data analytics"]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ConferenceExtraction(BaseModel):
    conference_name: Optional[str] = None
    official_url: Optional[str] = None

    topic_area: Optional[str] = None

    city: Optional[str] = None
    country: Optional[str] = None
    venue_name: Optional[str] = None

    start_date: Optional[str] = None
    end_date: Optional[str] = None

    ranking_tier: Optional[str] = None
    ranking_evidence_urls: List[str] = Field(default_factory=list)

    indexing_services: List[str] = Field(default_factory=list)  # e.g., ["Scopus", "IEEE Xplore"]
    indexing_evidence_urls: List[str] = Field(default_factory=list)

    paper_submission_deadline: Optional[str] = None

    page_limit_full_papers_excl_refs: Optional[str] = None
    abstract_word_limit: Optional[str] = None
    early_bird_deadline: Optional[str] = None

    presentation_formats: List[str] = Field(default_factory=list)  # e.g., ["oral", "poster"]
    poster_dimensions: Optional[str] = None
    session_structure: Optional[str] = None  # e.g., "single track", "parallel sessions"

    proceedings_publication_details: Optional[str] = None  # e.g., "Published in IEEE Xplore / ACM DL / LNCS"
    proceedings_evidence_urls: List[str] = Field(default_factory=list)

    other_supporting_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_conference() -> str:
    return """
Extract the single conference described in the answer as structured fields. You must only extract exactly what the answer text explicitly provides. Do not invent any values.

Return a JSON object with fields:
- conference_name: The conference's full official name as presented.
- official_url: The official website URL of the conference (not a third-party site).
- topic_area: The specific field/topic area the conference claims to focus on (e.g., "machine learning", "computer vision", "artificial intelligence", "data science"). Use the phrasing as shown in the answer.
- city: The host city for the 2025 edition.
- country: The country for the 2025 edition (e.g., "United States" or "Canada").
- venue_name: The venue/facility name (e.g., convention center name).
- start_date: The start date of the main conference (as written).
- end_date: The end date of the main conference (as written).
- ranking_tier: The ranking or tier (e.g., "CORE A*", "CORE A", or equivalent) cited in the answer text. Use the exact phrasing.
- ranking_evidence_urls: An array of URLs cited that support the ranking claim (e.g., CORE ranking page, authoritative sources). Only include valid URLs present in the answer.
- indexing_services: An array of the indexing services claimed (e.g., ["Scopus", "IEEE Xplore"]). Use the exact names as provided by the answer.
- indexing_evidence_urls: An array of URLs cited to support indexing claims. Only include valid URLs present in the answer.
- paper_submission_deadline: The full paper submission deadline date (as written in the answer).
- page_limit_full_papers_excl_refs: The full-paper page limit for submissions excluding references, as written (e.g., "8 pages + references excluded" or "10 pages (excluding references)"). Preserve the exact text in the answer.
- abstract_word_limit: The abstract word count limit as written (e.g., "200 words", "max 250 words"). Preserve the text.
- early_bird_deadline: The early-bird registration deadline date as written.
- presentation_formats: An array listing the accepted presentation formats (e.g., ["oral", "poster", "spotlight"]). Use the exact text from the answer.
- poster_dimensions: The required or recommended poster size or dimensions (e.g., "A0 portrait", "36x48 inches"), if provided in the answer. If not provided, return null.
- session_structure: The session organization format text (e.g., "single track", "parallel sessions", "multiple tracks"), as written. If not provided, return null.
- proceedings_publication_details: The proceedings publication details as quoted in the answer (e.g., "Proceedings published in IEEE Xplore / ACM Digital Library / Springer LNCS"), as written.
- proceedings_evidence_urls: An array of URLs cited to support the proceedings publication claim. Only include valid URLs in the answer.
- other_supporting_urls: An array of any other URLs cited in the answer that relate to 2025-specific details (e.g., CFP/Important Dates/Registration/Author Guidelines pages). Only include valid URLs in the answer.

Rules:
- Extract only URLs explicitly present in the answer text. If a URL is missing protocol, prepend http://.
- If a field is not present in the answer, return null (or [] for arrays).
- Do not infer data or rely on your own knowledge.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _collect_urls(*parts: List[Optional[str] | List[str] | None]) -> List[str]:
    """Flatten and deduplicate possibly mixed lists/strings of URLs."""
    collected: List[str] = []
    for p in parts:
        if not p:
            continue
        if isinstance(p, list):
            for x in p:
                if isinstance(x, str) and x.strip():
                    collected.append(x.strip())
        elif isinstance(p, str):
            if p.strip():
                collected.append(p.strip())
    # dedupe while preserving order
    seen = set()
    unique: List[str] = []
    for u in collected:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def _formats_contain_poster(presentation_formats: List[str]) -> bool:
    for f in presentation_formats or []:
        if "poster" in f.lower():
            return True
    return False


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_conference_tree(
    evaluator: Evaluator,
    parent_node,
    info: ConferenceExtraction
) -> None:
    """
    Build the verification subtree for the conference and execute all verification checks.
    All children here are critical since the parent node is critical.
    """
    # Main critical node for the overall task
    main_node = evaluator.add_parallel(
        id="conference_identification",
        desc="Identify one major computer science conference scheduled in 2025 in the US or Canada that satisfies all stated constraints and provide all requested details",
        parent=parent_node,
        critical=True
    )

    # Prepare common URL pools
    official_only = _collect_urls(info.official_url)
    all_supporting_urls = _collect_urls(
        info.official_url,
        info.other_supporting_urls,
        info.ranking_evidence_urls,
        info.indexing_evidence_urls,
        info.proceedings_evidence_urls
    )

    # 1) Conference identity
    evaluator.add_custom_node(
        result=bool(info.conference_name and info.official_url),
        id="conference_identity_exists",
        desc="Conference identity info provided: name and official website URL",
        parent=main_node,
        critical=True
    )
    node_identity = evaluator.add_leaf(
        id="conference_identity",
        desc="Provide the conference name and official website URL",
        parent=main_node,
        critical=True
    )
    identity_claim = f"The official website URL corresponds to the conference named '{info.conference_name or ''}'."
    await evaluator.verify(
        claim=identity_claim,
        node=node_identity,
        sources=official_only,
        additional_instruction="Verify that the provided URL is the official site and that it names the conference accordingly."
    )

    # 2) Topic area within AI/ML/CV/Data Science
    evaluator.add_custom_node(
        result=bool(info.topic_area),
        id="topic_area_exists",
        desc="Topic area is provided",
        parent=main_node,
        critical=True
    )
    node_topic = evaluator.add_leaf(
        id="topic_area",
        desc="Provide the specific field/topic area and verify it is within AI, machine learning, computer vision, or data science",
        parent=main_node,
        critical=True
    )
    topic_claim = (
        f"The conference focuses on {info.topic_area or ''}, which falls within artificial intelligence, "
        f"machine learning, computer vision, or data science."
    )
    await evaluator.verify(
        claim=topic_claim,
        node=node_topic,
        sources=official_only,
        additional_instruction=(
            "Accept synonyms and close variants. The topic must be reasonably categorized into AI, machine learning, "
            "computer vision, or data science."
        )
    )

    # 3) Location and venue in US/Canada
    evaluator.add_custom_node(
        result=bool(info.city and (info.country in ALLOWED_COUNTRIES) and info.venue_name),
        id="location_and_venue_exists",
        desc="City, country, and venue name are provided, country is United States or Canada",
        parent=main_node,
        critical=True
    )
    node_loc = evaluator.add_leaf(
        id="location_and_venue",
        desc="Provide the city and venue name and verify the conference is held in the United States or Canada (North America)",
        parent=main_node,
        critical=True
    )
    loc_claim = (
        f"The 2025 edition is held in {info.city or ''} at {info.venue_name or ''}, located in the United States or Canada."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=node_loc,
        sources=official_only,
        additional_instruction="Verify that the location and venue correspond to the 2025 edition and that the country is the US or Canada."
    )

    # 4) Dates in 2025
    evaluator.add_custom_node(
        result=bool(info.start_date and info.end_date),
        id="dates_in_2025_exists",
        desc="Start and end dates are provided",
        parent=main_node,
        critical=True
    )
    node_dates = evaluator.add_leaf(
        id="dates_in_2025",
        desc="Provide start and end dates and verify the conference takes place in 2025",
        parent=main_node,
        critical=True
    )
    dates_claim = (
        f"The 2025 conference dates are {info.start_date or ''} to {info.end_date or ''}, and it takes place in 2025."
    )
    await evaluator.verify(
        claim=dates_claim,
        node=node_dates,
        sources=official_only,
        additional_instruction="Ensure the dates correspond to the 2025 edition (not workshops/tutorials unless they define the main conference dates)."
    )

    # 5) Ranking evidence: CORE A*/A or equivalent
    evaluator.add_custom_node(
        result=bool(info.ranking_tier) and (len(info.ranking_evidence_urls) > 0),
        id="ranking_urls_provided",
        desc="Ranking/tier info and at least one ranking evidence URL provided",
        parent=main_node,
        critical=True
    )
    node_rank = evaluator.add_leaf(
        id="ranking_evidence",
        desc="Provide evidence of top-tier status (CORE A*/A or equivalent recognition) and verify the conference meets this ranking/tier requirement",
        parent=main_node,
        critical=True
    )
    rank_claim = (
        f"The conference '{info.conference_name or ''}' is ranked CORE A* or CORE A, or is otherwise recognized as a top-tier venue in its field."
    )
    await evaluator.verify(
        claim=rank_claim,
        node=node_rank,
        sources=_collect_urls(info.ranking_evidence_urls, info.other_supporting_urls),
        additional_instruction="Use authoritative ranking sources (e.g., CORE portal). If 'equivalent recognition' is claimed, it must be explicit and credible."
    )

    # 6) Indexing in Scopus and/or IEEE Xplore
    evaluator.add_custom_node(
        result=(len(info.indexing_evidence_urls) > 0) or (len(info.proceedings_evidence_urls) > 0),
        id="indexing_urls_provided",
        desc="Indexing evidence URLs and/or proceedings URLs provided",
        parent=main_node,
        critical=True
    )
    node_index = evaluator.add_leaf(
        id="indexing_evidence",
        desc="Provide indexing information and verify proceedings are indexed in Scopus and/or IEEE Xplore",
        parent=main_node,
        critical=True
    )
    indexing_claim = "The conference proceedings are indexed in Scopus and/or IEEE Xplore."
    await evaluator.verify(
        claim=indexing_claim,
        node=node_index,
        sources=_collect_urls(info.indexing_evidence_urls, info.proceedings_evidence_urls, info.other_supporting_urls),
        additional_instruction="A direct statement from credible sources (publisher or index) is required. Either Scopus or IEEE Xplore suffices."
    )

    # 7) Paper submission deadline after Jan 15, 2025
    evaluator.add_custom_node(
        result=bool(info.paper_submission_deadline),
        id="paper_submission_deadline_exists",
        desc="Paper submission deadline is provided",
        parent=main_node,
        critical=True
    )
    node_deadline = evaluator.add_leaf(
        id="paper_submission_deadline",
        desc="Provide the paper submission deadline date and verify it is after January 15, 2025",
        parent=main_node,
        critical=True
    )
    deadline_claim = (
        f"The paper submission deadline is {info.paper_submission_deadline or ''} and it is after {DEADLINE_THRESHOLD_TEXT}."
    )
    await evaluator.verify(
        claim=deadline_claim,
        node=node_deadline,
        sources=_collect_urls(official_only, info.other_supporting_urls),
        additional_instruction="Confirm the deadline on the official CFP/Important Dates page and verify it falls strictly after January 15, 2025."
    )

    # 8) Paper page limit (excluding references)
    evaluator.add_custom_node(
        result=bool(info.page_limit_full_papers_excl_refs),
        id="paper_page_limit_exists",
        desc="Full-paper page limit (excluding references) is provided",
        parent=main_node,
        critical=True
    )
    node_page_limit = evaluator.add_leaf(
        id="paper_page_limit",
        desc="Provide the full-paper page limit for submissions (excluding references)",
        parent=main_node,
        critical=True
    )
    page_limit_claim = (
        f"The full‑paper page limit for submissions (excluding references) is '{info.page_limit_full_papers_excl_refs or ''}'."
    )
    await evaluator.verify(
        claim=page_limit_claim,
        node=node_page_limit,
        sources=_collect_urls(official_only, info.other_supporting_urls),
        additional_instruction="Check author guidelines/call for papers pages for page limit instructions, excluding references."
    )

    # 9) Abstract word limit
    evaluator.add_custom_node(
        result=bool(info.abstract_word_limit),
        id="abstract_word_limit_exists",
        desc="Abstract word count limit is provided",
        parent=main_node,
        critical=True
    )
    node_abs = evaluator.add_leaf(
        id="abstract_word_limit",
        desc="Provide the abstract word count limit",
        parent=main_node,
        critical=True
    )
    abstract_claim = f"The abstract word count limit is '{info.abstract_word_limit or ''}'."
    await evaluator.verify(
        claim=abstract_claim,
        node=node_abs,
        sources=_collect_urls(official_only, info.other_supporting_urls),
        additional_instruction="Check submission instructions; accept reasonable variants such as 'up to 200 words'."
    )

    # 10) Early-bird registration deadline
    evaluator.add_custom_node(
        result=bool(info.early_bird_deadline),
        id="early_bird_registration_deadline_exists",
        desc="Early bird registration deadline is provided",
        parent=main_node,
        critical=True
    )
    node_early = evaluator.add_leaf(
        id="early_bird_registration_deadline",
        desc="Provide the early bird registration deadline",
        parent=main_node,
        critical=True
    )
    early_claim = f"The early bird registration deadline is {info.early_bird_deadline or ''}."
    await evaluator.verify(
        claim=early_claim,
        node=node_early,
        sources=_collect_urls(official_only, info.other_supporting_urls),
        additional_instruction="Verify on the official registration/fees page."
    )

    # 11) Presentation formats
    evaluator.add_custom_node(
        result=bool(info.presentation_formats and len(info.presentation_formats) > 0),
        id="presentation_formats_exists",
        desc="Accepted presentation formats are provided",
        parent=main_node,
        critical=True
    )
    node_formats = evaluator.add_leaf(
        id="presentation_formats",
        desc="List the accepted presentation formats (e.g., oral, poster, etc.)",
        parent=main_node,
        critical=True
    )
    formats_text = ", ".join(info.presentation_formats) if info.presentation_formats else ""
    formats_claim = f"The accepted presentation formats include: {formats_text}."
    await evaluator.verify(
        claim=formats_claim,
        node=node_formats,
        sources=_collect_urls(official_only, info.other_supporting_urls),
        additional_instruction="Accept synonyms (e.g., 'spotlight talk' as a type of oral). Verify via program/author instructions."
    )

    # 12) Poster dimensions (if posters are accepted)
    posters_accepted = _formats_contain_poster(info.presentation_formats)
    evaluator.add_custom_node(
        result=(not posters_accepted) or bool(info.poster_dimensions),
        id="poster_dimensions_required_if_poster",
        desc="If posters are accepted, poster dimensions must be provided",
        parent=main_node,
        critical=True
    )
    node_poster = evaluator.add_leaf(
        id="poster_dimensions",
        desc="If posters are accepted, provide the required or recommended poster dimensions",
        parent=main_node,
        critical=True
    )
    if posters_accepted:
        poster_claim = f"The required or recommended poster dimensions are '{info.poster_dimensions or ''}'."
        poster_sources = _collect_urls(official_only, info.other_supporting_urls)
        poster_instruction = "Verify on the poster guidelines or author instructions page."
    else:
        poster_claim = "The conference does not accept poster presentations for the 2025 edition."
        poster_sources = _collect_urls(official_only, info.other_supporting_urls)
        poster_instruction = "Confirm via the official program/CFP/author instructions that posters are not accepted."
    await evaluator.verify(
        claim=poster_claim,
        node=node_poster,
        sources=poster_sources,
        additional_instruction=poster_instruction
    )

    # 13) Session organization format
    evaluator.add_custom_node(
        result=bool(info.session_structure),
        id="session_structure_exists",
        desc="Session organization format is provided",
        parent=main_node,
        critical=True
    )
    node_session = evaluator.add_leaf(
        id="session_structure",
        desc="Describe the session organization format (e.g., single track, parallel sessions, etc.)",
        parent=main_node,
        critical=True
    )
    session_claim = f"The session organization format is '{info.session_structure or ''}' for the 2025 conference."
    await evaluator.verify(
        claim=session_claim,
        node=node_session,
        sources=_collect_urls(official_only, info.other_supporting_urls),
        additional_instruction="Look for program overview or format statement. Accept typical phrasing like 'single track' or 'parallel sessions'."
    )

    # 14) Proceedings publication details and verification
    evaluator.add_custom_node(
        result=bool(info.proceedings_publication_details),
        id="proceedings_publication_details_exists",
        desc="Proceedings publication details are provided",
        parent=main_node,
        critical=True
    )
    node_proceedings = evaluator.add_leaf(
        id="proceedings_publication_details",
        desc="Provide proceedings publication details (e.g., publisher/series/access location) and verify the conference publishes official proceedings",
        parent=main_node,
        critical=True
    )
    proceedings_claim = (
        f"The conference publishes official proceedings: {info.proceedings_publication_details or ''}."
    )
    await evaluator.verify(
        claim=proceedings_claim,
        node=node_proceedings,
        sources=_collect_urls(info.proceedings_evidence_urls, official_only, info.other_supporting_urls),
        additional_instruction="Publisher or digital library links (e.g., IEEE Xplore, ACM DL, Springer) or official statements are valid evidence."
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
    Evaluate an answer for the 'cs_conf_2025_na' task using a verification tree and URL-backed fact checking.
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

    # Extract structured conference info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_conference(),
        template_class=ConferenceExtraction,
        extraction_name="conference_extraction"
    )

    # Record useful ground-truth constraints/context for transparency (not used as scoring)
    evaluator.add_custom_info(
        info={
            "allowed_countries": list(ALLOWED_COUNTRIES),
            "allowed_topic_keywords_examples": ALLOWED_TOPIC_KEYWORDS,
            "deadline_threshold": DEADLINE_THRESHOLD_TEXT
        },
        info_type="constraints",
        info_name="constraints_context"
    )

    # Build verification tree and run checks
    await build_and_verify_conference_tree(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()