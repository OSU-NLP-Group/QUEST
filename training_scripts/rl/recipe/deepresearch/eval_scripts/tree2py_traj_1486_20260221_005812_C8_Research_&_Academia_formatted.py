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
TASK_ID = "cs_conf_spring_2026"
TASK_DESCRIPTION = (
    "I am a computer science graduate student planning to submit my research to academic conferences in spring 2026. "
    "I need to identify 2 suitable conferences that meet the following requirements:\n\n"
    "Mandatory Requirements:\n"
    "1. Conference dates: Must take place between April 1, 2026 and June 30, 2026\n"
    "2. Location: Must be held in either North America or Europe\n"
    "3. Submission type: Must accept full research papers (not just extended abstracts or posters only)\n"
    "4. Paper length: Full papers must be allowed to be at least 6,000 words or 8 pages in length\n"
    "5. Format: Must use either IEEE or ACM standard paper formats\n"
    "6. Deadline: Paper submission deadline must be no later than March 1, 2026\n"
    "7. Peer review: Must have a formal peer review process\n"
    "8. Publication: Papers must be published in official conference proceedings\n"
    "9. Cost: Early bird student registration fee must be no more than $500\n\n"
    "Preferred (but not mandatory) Requirements:\n"
    "10. Should offer poster presentation as an alternative submission option\n"
    "11. Main conference venue should accommodate at least 200 attendees\n"
    "12. Venue should meet ADA accessibility standards (or equivalent)\n"
    "13. Should offer travel grants or funding support for students\n\n"
    "For each conference you identify, please provide:\n"
    "- Conference name\n"
    "- Conference dates\n"
    "- Location (city and country)\n"
    "- A reference URL that confirms the conference details\n\n"
    "Please identify 2 conferences that satisfy all mandatory requirements and as many preferred requirements as possible."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SingleConferenceExtraction(BaseModel):
    name: Optional[str] = None
    dates: Optional[str] = None
    location_city: Optional[str] = None
    location_country: Optional[str] = None
    reference_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)

    # Optional fields (verbatim text if present in answer)
    submission_deadline: Optional[str] = None
    accepts_full_papers: Optional[str] = None
    pages_or_words: Optional[str] = None
    format_standard: Optional[str] = None
    peer_review_statement: Optional[str] = None
    proceedings_publication: Optional[str] = None
    early_bird_student_fee: Optional[str] = None

    # Preferred non-mandatory
    poster_option: Optional[str] = None
    venue_capacity: Optional[str] = None
    accessibility: Optional[str] = None
    travel_grants: Optional[str] = None


class ConferencesExtraction(BaseModel):
    conferences: List[SingleConferenceExtraction] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_conferences() -> str:
    return """
    Extract up to the first 2 conferences mentioned in the answer, capturing the following fields for each conference exactly as presented in the answer text.

    For each conference, extract these fields (use null if missing):
    - name: Full conference name
    - dates: Conference dates as a free-form string (e.g., "June 12–15, 2026")
    - location_city: City name
    - location_country: Country name
    - reference_url: The primary URL provided to confirm conference details (a single URL)
    - additional_urls: Any other URLs mentioned for this conference (array; exclude duplicates and the primary reference_url)
    - submission_deadline: The paper submission (full paper) deadline string if provided
    - accepts_full_papers: Any explicit wording indicating acceptance of full research papers (e.g., "Full paper submissions are accepted")
    - pages_or_words: The full paper length policy (e.g., "8 pages", "10 pages excluding references", or "6000 words")
    - format_standard: The required paper format standard if stated (e.g., "IEEE", "ACM", "ACM SIGCONF")
    - peer_review_statement: Any explicit mention of peer review (e.g., "peer-reviewed", "double-blind review")
    - proceedings_publication: Any explicit mention of official proceedings publication (e.g., "published in IEEE Xplore", "ACM Digital Library")
    - early_bird_student_fee: Early bird student registration fee amount with currency if provided (e.g., "$450", "€400")

    Preferred but optional (use null if missing):
    - poster_option: Any mention that poster presentations are offered
    - venue_capacity: Any mention of venue capacity (e.g., "up to 500 attendees")
    - accessibility: Any mention of ADA or equivalent accessibility standards
    - travel_grants: Any mention of travel grants or student funding support

    Return a JSON object with key 'conferences' which is an array of objects containing the fields above for each of the first 2 conferences found in the answer.
    Do not invent information; only extract what is explicitly present in the answer. If URLs are in markdown-style links, extract the actual URL.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def gather_sources(conf: SingleConferenceExtraction) -> List[str]:
    urls: List[str] = []
    if conf.reference_url and conf.reference_url.strip():
        urls.append(conf.reference_url.strip())
    for u in conf.additional_urls or []:
        if u and u.strip() and u.strip() not in urls:
            urls.append(u.strip())
    return urls


def _safe_join_location(city: Optional[str], country: Optional[str]) -> str:
    parts = [p for p in [city, country] if p and p.strip()]
    return ", ".join(parts) if parts else "the stated location"


# --------------------------------------------------------------------------- #
# Verification for a single conference                                        #
# --------------------------------------------------------------------------- #
async def verify_conference(evaluator: Evaluator, parent_node, conf: SingleConferenceExtraction, idx: int) -> None:
    conf_num = idx + 1
    conf_node = evaluator.add_parallel(
        id=f"conference_{conf_num}",
        desc=f"{'First' if conf_num == 1 else 'Second'} suitable conference meeting all mandatory requirements",
        parent=parent_node,
        critical=False
    )

    # Existence of name and reference URL (critical gate)
    evaluator.add_custom_node(
        result=bool(conf.name and conf.name.strip()) and bool(conf.reference_url and conf.reference_url.strip()),
        id=f"conf{conf_num}_name_and_reference",
        desc="Provide the conference name and a reference URL",
        parent=conf_node,
        critical=True
    )

    # Prepare sources
    sources = gather_sources(conf)
    loc_text = _safe_join_location(conf.location_city, conf.location_country)

    # Create mandatory leaf nodes
    node_dates = evaluator.add_leaf(
        id=f"conf{conf_num}_dates",
        desc="Conference must take place between April 1, 2026 and June 30, 2026",
        parent=conf_node,
        critical=True
    )
    node_location = evaluator.add_leaf(
        id=f"conf{conf_num}_location",
        desc="Conference must be held in North America or Europe",
        parent=conf_node,
        critical=True
    )
    node_full_papers = evaluator.add_leaf(
        id=f"conf{conf_num}_full_papers",
        desc="Conference must accept full research papers (not just abstracts)",
        parent=conf_node,
        critical=True
    )
    node_paper_length = evaluator.add_leaf(
        id=f"conf{conf_num}_paper_length",
        desc="Full papers must be allowed to be at least 6,000 words or 8 pages",
        parent=conf_node,
        critical=True
    )
    node_format = evaluator.add_leaf(
        id=f"conf{conf_num}_format",
        desc="Conference must use IEEE or ACM paper format standards",
        parent=conf_node,
        critical=True
    )
    node_deadline = evaluator.add_leaf(
        id=f"conf{conf_num}_deadline",
        desc="Paper submission deadline must be no later than March 1, 2026",
        parent=conf_node,
        critical=True
    )
    node_peer_review = evaluator.add_leaf(
        id=f"conf{conf_num}_peer_review",
        desc="Conference must have a formal peer review process",
        parent=conf_node,
        critical=True
    )
    node_proceedings = evaluator.add_leaf(
        id=f"conf{conf_num}_proceedings",
        desc="Papers must be published in official conference proceedings",
        parent=conf_node,
        critical=True
    )
    node_fee = evaluator.add_leaf(
        id=f"conf{conf_num}_registration_fee",
        desc="Early bird student registration must be no more than $500",
        parent=conf_node,
        critical=True
    )

    # Preferred (non-critical) leaf nodes
    node_poster = evaluator.add_leaf(
        id=f"conf{conf_num}_poster_option",
        desc="Conference must offer poster presentations as an option",
        parent=conf_node,
        critical=False
    )
    node_capacity = evaluator.add_leaf(
        id=f"conf{conf_num}_venue_capacity",
        desc="Main venue must accommodate at least 200 attendees",
        parent=conf_node,
        critical=False
    )
    node_accessibility = evaluator.add_leaf(
        id=f"conf{conf_num}_accessibility",
        desc="Venue must meet ADA accessibility standards or equivalent",
        parent=conf_node,
        critical=False
    )
    node_travel = evaluator.add_leaf(
        id=f"conf{conf_num}_travel_grants",
        desc="Conference must offer travel grants or funding support for students",
        parent=conf_node,
        critical=False
    )

    # Build claims and run batch verification for the leaves (excluding existence which is custom)
    claims_and_sources: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    # Dates within range (inclusive)
    dates_claim = "This conference takes place between 2026-04-01 and 2026-06-30 (inclusive)."
    dates_ins = (
        "Check the event dates on the page. The claim is supported if all scheduled conference days fall within "
        "April 1, 2026 and June 30, 2026, inclusive. If multiple events (workshops, main track) are listed, use the main conference dates."
    )
    claims_and_sources.append((dates_claim, sources, node_dates, dates_ins))

    # Location in NA or Europe
    if conf.location_city or conf.location_country:
        loc_claim = f"The conference is held in {loc_text}, which is located in either North America or Europe."
    else:
        loc_claim = "The conference venue is located in either North America or Europe."
    loc_ins = (
        "Use the city/country stated on the page to determine the continent. It is acceptable if the page only lists "
        "city and country; you may infer the continent from that information. The claim is supported if the country is in Europe or North America."
    )
    claims_and_sources.append((loc_claim, sources, node_location, loc_ins))

    # Accepts full research papers
    fp_claim = "The conference accepts full research papers (not just extended abstracts or posters)."
    fp_ins = (
        "Look for explicit mention of 'full papers', 'research papers', or equivalent. Calls for full paper submissions, "
        "or proceedings papers, satisfy this requirement."
    )
    claims_and_sources.append((fp_claim, sources, node_full_papers, fp_ins))

    # Paper length policy (>= 8 pages or >= 6,000 words)
    length_claim = (
        "The author guidelines permit full research papers of at least 8 pages or at least 6,000 words (excluding references if specified)."
    )
    length_ins = (
        "Verify the full paper page/word limits in the author guidelines. The claim is supported if the stated limit "
        "is 8+ pages or 6000+ words. If references are excluded from the page count, that is acceptable."
    )
    claims_and_sources.append((length_claim, sources, node_paper_length, length_ins))

    # Format IEEE or ACM
    format_claim = "The required paper format uses IEEE or ACM standard templates."
    format_ins = (
        "Look for mentions of 'IEEE conference template', 'IEEE Xplore formatting', 'ACM SIGCONF', or 'ACM template'. "
        "Any explicit IEEE or ACM template requirement supports the claim."
    )
    claims_and_sources.append((format_claim, sources, node_format, format_ins))

    # Submission deadline no later than March 1, 2026
    if conf.submission_deadline and conf.submission_deadline.strip():
        ddl_claim = f"The paper submission deadline is {conf.submission_deadline.strip()}, which is on or before March 1, 2026."
    else:
        ddl_claim = "The paper submission deadline is on or before March 1, 2026."
    ddl_ins = (
        "Use the full paper submission deadline (not abstract-only deadlines). The claim is supported if the full paper deadline date is "
        "March 1, 2026 or earlier. If multiple rounds exist, use the earliest full paper deadline."
    )
    claims_and_sources.append((ddl_claim, sources, node_deadline, ddl_ins))

    # Peer review process
    pr_claim = "The conference employs a formal peer review process (e.g., single-blind or double-blind review)."
    pr_ins = (
        "Look for text such as 'peer-reviewed', 'review process', 'double-blind', or 'single-blind'. Any explicit mention of peer review supports the claim."
    )
    claims_and_sources.append((pr_claim, sources, node_peer_review, pr_ins))

    # Proceedings publication
    proc_claim = "Accepted papers are published in official conference proceedings."
    proc_ins = (
        "Look for statements like 'published in the proceedings', 'IEEE Xplore', 'ACM Digital Library', or equivalent official proceedings."
    )
    claims_and_sources.append((proc_claim, sources, node_proceedings, proc_ins))

    # Registration fee (early bird student <= $500)
    fee_claim = "The early bird student registration fee is at most $500 (USD)."
    fee_ins = (
        "Verify a fee table or registration information indicating an early bird student rate of $500 or less. "
        "If fees are listed in USD, compare directly. If listed in EUR/GBP, you may accept clearly lower nominal values (e.g., €450) as within $500."
    )
    claims_and_sources.append((fee_claim, sources, node_fee, fee_ins))

    # Preferred checks (non-critical)
    poster_claim = "The conference offers poster presentations as a submission or presentation option."
    poster_ins = "Look for 'poster session', 'posters', or guidance on poster submissions."
    claims_and_sources.append((poster_claim, sources, node_poster, poster_ins))

    capacity_claim = "The main conference venue accommodates at least 200 attendees."
    capacity_ins = (
        "Look for venue capacity information, expected attendance numbers, or venue specifications indicating capacity ≥ 200."
    )
    claims_and_sources.append((capacity_claim, sources, node_capacity, capacity_ins))

    accessibility_claim = "The conference venue meets ADA accessibility standards or an equivalent accessibility standard."
    accessibility_ins = (
        "Look for statements on accessibility (ADA, wheelchair access, or equivalent national standards) in venue or conference info."
    )
    claims_and_sources.append((accessibility_claim, sources, node_accessibility, accessibility_ins))

    travel_claim = "The conference offers travel grants or funding support for students."
    travel_ins = "Look for 'student travel grants', 'travel support', 'scholarships', or similar programs."
    claims_and_sources.append((travel_claim, sources, node_travel, travel_ins))

    await evaluator.batch_verify(claims_and_sources)


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
    # Initialize evaluator with parallel root (two conferences evaluated independently)
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

    # Extract up to two conferences
    extracted = await evaluator.extract(
        prompt=prompt_extract_conferences(),
        template_class=ConferencesExtraction,
        extraction_name="conferences_extraction"
    )

    # Normalize to exactly two entries (pad with empty if needed, slice if more)
    conferences: List[SingleConferenceExtraction] = list(extracted.conferences or [])
    if len(conferences) > 2:
        conferences = conferences[:2]
    while len(conferences) < 2:
        conferences.append(SingleConferenceExtraction())

    # Add custom info for debugging
    evaluator.add_custom_info(
        {"extracted_conferences_count": len(extracted.conferences or [])},
        info_type="extraction_stats",
        info_name="extraction_stats"
    )

    # Build verification subtrees for each conference
    # The parent is the root (parallel)
    await verify_conference(evaluator, root, conferences[0], 0)
    await verify_conference(evaluator, root, conferences[1], 1)

    # Return the summary
    return evaluator.get_summary()