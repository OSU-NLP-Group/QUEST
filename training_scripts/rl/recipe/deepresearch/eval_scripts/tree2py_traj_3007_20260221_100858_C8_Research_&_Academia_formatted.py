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
TASK_ID = "conf_2026_parallel_algorithms"
TASK_DESCRIPTION = (
    "A computational science researcher has completed a study on parallel algorithms for climate modeling in early "
    "January 2026 and needs to submit full research papers to four suitable international conferences in 2026. "
    "Identify four conferences that satisfy all of the following requirements:\n\n"
    "1. Timing Requirements: The conference submission deadline must be on or after January 15, 2026, the conference "
    "must take place between June 1 and August 31, 2026, and the submission deadline must be at least 3 months before "
    "the conference dates to allow for peer review.\n\n"
    "2. Quality Indicators: The conference proceedings must be indexed in either Scopus or Web of Science, and the "
    "conference must be organized or sponsored by a recognized professional organization such as IEEE, ACM, or an "
    "equivalent established conference series with Springer or similar reputable publishers.\n\n"
    "3. Field Alignment: The conference's primary focus must be on computational science, high-performance computing (HPC), "
    "parallel computing, scientific computing, or closely related computer science topics appropriate for research on parallel algorithms.\n\n"
    "4. Submission Requirements: The conference must accept full research paper submissions (not just abstracts or posters) "
    "with a page limit between 8 and 15 pages for the initial submission.\n\n"
    "For each conference, provide: (a) the official conference name and acronym, (b) the exact conference dates, "
    "(c) the conference location (city and country), (d) the paper submission deadline, (e) confirmation of indexing database, "
    "(f) the organizing body, (g) the conference website URL, and (h) evidence that it accepts papers in the relevant field."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SingleConference(BaseModel):
    """Structured representation of a single conference as extracted from the answer."""
    name: Optional[str] = None
    acronym: Optional[str] = None

    # Core schedule
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    submission_deadline: Optional[str] = None

    # Location
    city: Optional[str] = None
    country: Optional[str] = None

    # Quality / Organization
    indexing_db: Optional[str] = None  # e.g., "Scopus", "Web of Science", or both
    organizing_body: Optional[str] = None  # e.g., "IEEE", "ACM", "Springer LNCS", etc.

    # URLs
    website_url: Optional[str] = None
    timing_urls: List[str] = Field(default_factory=list)   # Explicit URLs for dates/deadlines
    quality_urls: List[str] = Field(default_factory=list)  # Explicit URLs for indexing/organization
    field_urls: List[str] = Field(default_factory=list)    # Explicit URLs for fields/topics
    format_urls: List[str] = Field(default_factory=list)   # Explicit URLs for submission format/page limit

    # Optional topics list, if the answer lists them
    topics: List[str] = Field(default_factory=list)


class ConferencesExtraction(BaseModel):
    """Top-level extraction model: a list of up to four conferences."""
    conferences: List[SingleConference] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_conferences() -> str:
    return """
    Extract up to FOUR conferences mentioned in the answer that the researcher could submit to in 2026. 
    For each conference, return a JSON object with the following fields (extract exactly as written in the answer):

    - name: Official conference name (string)
    - acronym: Official acronym if stated (string; null if not present)
    - start_date: Exact conference start date (string as presented)
    - end_date: Exact conference end date (string as presented)
    - city: Conference city (string)
    - country: Conference country (string)
    - submission_deadline: Exact paper submission deadline date (string as presented)
    - indexing_db: The indexing database name if stated (e.g., "Scopus", "Web of Science", or similar; string)
    - organizing_body: The organizing/sponsoring body (e.g., IEEE, ACM, Springer LNCS series; string)
    - website_url: The official conference website URL (string)
    - timing_urls: All URLs in the answer that specifically mention dates/deadlines (array of URLs; empty if none)
    - quality_urls: All URLs in the answer that specifically mention indexing or organizing body details (array of URLs; empty if none)
    - field_urls: All URLs in the answer that specifically mention topics/fields/aims/scope (array of URLs; empty if none)
    - format_urls: All URLs in the answer that specifically mention submission format, page limits, or paper categories (array of URLs; empty if none)
    - topics: Any topic keywords or phrases listed for the conference (array of strings; empty if none)

    IMPORTANT:
    - Extract only what is explicitly present in the answer. Do not invent or infer any values.
    - Dates must be extracted exactly as they appear in the answer text.
    - For URLs, include only valid URLs present in the answer (including markdown links). Do not fabricate links.
    - If any field is missing for a conference, return null for that field or an empty array, as appropriate.

    Return the result as: {"conferences": [ ... up to 4 items ... ]}.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _valid_url(url: Optional[str]) -> bool:
    return bool(url and url.strip() and url.strip().lower().startswith(("http://", "https://")))


def compile_sources(conf: SingleConference, url_kinds: List[str]) -> List[str]:
    """
    Collect and deduplicate URLs across requested kinds and always include the main website if present.
    url_kinds elements should be one of: "timing", "quality", "field", "format".
    """
    collected: List[str] = []
    for kind in url_kinds:
        if kind == "timing":
            collected.extend(conf.timing_urls or [])
        elif kind == "quality":
            collected.extend(conf.quality_urls or [])
        elif kind == "field":
            collected.extend(conf.field_urls or [])
        elif kind == "format":
            collected.extend(conf.format_urls or [])
    if conf.website_url and _valid_url(conf.website_url):
        collected.append(conf.website_url)

    # Deduplicate, preserve order
    seen = set()
    final: List[str] = []
    for u in collected:
        if not u:
            continue
        uu = u.strip()
        if not _valid_url(uu):
            continue
        if uu not in seen:
            seen.add(uu)
            final.append(uu)
    return final


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_single_conference(
    evaluator: Evaluator,
    parent_node,
    conf: SingleConference,
    index: int
) -> None:
    """
    Build verification sub-tree for one conference and run verifications.
    index: 0-based index; used for node IDs.
    """
    conf_num = index + 1

    # Top-level conference node (parallel aggregation, non-critical to allow partial credit across items)
    conf_node = evaluator.add_parallel(
        id=f"conference_{conf_num}",
        desc=f"Conference #{conf_num} meeting all requirements",
        parent=parent_node,
        critical=False,
    )

    # 1) Information provided (non-critical collection checks)
    info_node = evaluator.add_parallel(
        id=f"conference_{conf_num}_information",
        desc=f"Required information provided for conference #{conf_num}",
        parent=conf_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(conf.name and conf.name.strip()) and bool(conf.acronym and conf.acronym.strip()),
        id=f"conference_{conf_num}_info_name",
        desc="Official conference name and acronym provided",
        parent=info_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=bool(conf.start_date and conf.start_date.strip()) and bool(conf.end_date and conf.end_date.strip()),
        id=f"conference_{conf_num}_info_dates",
        desc="Exact conference dates provided",
        parent=info_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=bool(conf.city and conf.city.strip()) and bool(conf.country and conf.country.strip()),
        id=f"conference_{conf_num}_info_location",
        desc="Conference location (city and country) provided",
        parent=info_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=bool(conf.submission_deadline and conf.submission_deadline.strip()),
        id=f"conference_{conf_num}_info_deadline",
        desc="Paper submission deadline provided",
        parent=info_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=bool(conf.organizing_body and conf.organizing_body.strip()),
        id=f"conference_{conf_num}_info_organizing_body",
        desc="Organizing body name provided",
        parent=info_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_valid_url(conf.website_url),
        id=f"conference_{conf_num}_info_url",
        desc="Conference website URL provided",
        parent=info_node,
        critical=False
    )

    # 2) Timing requirements (critical)
    timing_node = evaluator.add_parallel(
        id=f"conference_{conf_num}_timing",
        desc=f"Timing requirements for conference #{conf_num}",
        parent=conf_node,
        critical=True
    )

    # Timing sources compile
    timing_sources = compile_sources(conf, ["timing"])

    # 2.1 Submission deadline on/after Jan 15, 2026
    leaf_deadline_ok = evaluator.add_leaf(
        id=f"conference_{conf_num}_timing_submission_deadline",
        desc="Submission deadline is on or after January 15, 2026",
        parent=timing_node,
        critical=True
    )
    claim_deadline = (
        f"The paper submission deadline ({conf.submission_deadline or 'unknown'}) is on or after January 15, 2026."
    )
    await evaluator.verify(
        claim=claim_deadline,
        node=leaf_deadline_ok,
        sources=timing_sources,
        additional_instruction="Use the cited call-for-papers or important dates page to confirm the deadline date is >= Jan 15, 2026."
    )

    # 2.2 Conference dates between June 1 and August 31, 2026
    leaf_dates_window = evaluator.add_leaf(
        id=f"conference_{conf_num}_timing_conference_dates",
        desc="Conference dates fall between June 1 and August 31, 2026",
        parent=timing_node,
        critical=True
    )
    claim_dates_window = (
        f"The conference takes place between June 1 and August 31, 2026. "
        f"Start date: {conf.start_date or 'unknown'}, End date: {conf.end_date or 'unknown'}."
    )
    await evaluator.verify(
        claim=claim_dates_window,
        node=leaf_dates_window,
        sources=timing_sources,
        additional_instruction="Verify the official schedule shows the event dates within June–August 2026 (inclusive)."
    )

    # 2.3 Review period: deadline at least 3 months before start date
    leaf_review_period = evaluator.add_leaf(
        id=f"conference_{conf_num}_timing_review_period",
        desc="Submission deadline is at least 3 months before the conference start date",
        parent=timing_node,
        critical=True
    )
    claim_review_period = (
        f"The submission deadline ({conf.submission_deadline or 'unknown'}) is at least 3 months (≈90 days) before the "
        f"conference start date ({conf.start_date or 'unknown'})."
    )
    await evaluator.verify(
        claim=claim_review_period,
        node=leaf_review_period,
        sources=timing_sources,
        additional_instruction="Confirm from the official dates page that the interval between the submission deadline and the start date is ≥ 3 months (~90 days)."
    )

    # 2.4 Timing URL reference provided (critical for consistency under a critical parent)
    evaluator.add_custom_node(
        result=len(timing_sources) > 0,
        id=f"conference_{conf_num}_timing_url",
        desc="URL reference provided for timing information",
        parent=timing_node,
        critical=True
    )

    # 3) Quality indicators (critical)
    quality_node = evaluator.add_parallel(
        id=f"conference_{conf_num}_quality",
        desc=f"Quality indicators for conference #{conf_num}",
        parent=conf_node,
        critical=True
    )
    quality_sources = compile_sources(conf, ["quality"])

    # 3.1 Indexing in Scopus or Web of Science
    leaf_indexing = evaluator.add_leaf(
        id=f"conference_{conf_num}_quality_indexing",
        desc="Conference proceedings are indexed in Scopus or Web of Science",
        parent=quality_node,
        critical=True
    )
    idx_desc = conf.indexing_db or "Scopus or Web of Science"
    claim_indexing = (
        f"The conference proceedings are indexed in {idx_desc}, specifically Scopus or Web of Science."
    )
    await evaluator.verify(
        claim=claim_indexing,
        node=leaf_indexing,
        sources=quality_sources,
        additional_instruction=(
            "Accept credible evidence that the proceedings or the series (e.g., IEEE Xplore, Springer LNCS) are indexed "
            "by Scopus or Web of Science. Generic assertions without reliable source should not be accepted."
        )
    )

    # 3.2 Organized/sponsored by recognized org (IEEE/ACM or equivalent established publisher/series)
    leaf_org = evaluator.add_leaf(
        id=f"conference_{conf_num}_quality_organization",
        desc="Conference is organized or sponsored by IEEE, ACM, or equivalent recognized international organization",
        parent=quality_node,
        critical=True
    )
    org_name = conf.organizing_body or "a recognized professional organization or established publisher series"
    claim_org = (
        f"The conference is organized or sponsored by {org_name}, which is recognized internationally "
        "(e.g., IEEE, ACM, Springer LNCS, Elsevier, SIAM)."
    )
    await evaluator.verify(
        claim=claim_org,
        node=leaf_org,
        sources=quality_sources,
        additional_instruction="Confirm organizing/sponsoring body on the official site or credible publisher/series page."
    )

    # 3.3 Quality URL reference provided (critical under a critical parent)
    evaluator.add_custom_node(
        result=len(quality_sources) > 0,
        id=f"conference_{conf_num}_quality_url",
        desc="URL reference provided for quality indicators",
        parent=quality_node,
        critical=True
    )

    # 4) Field alignment (critical)
    field_node = evaluator.add_parallel(
        id=f"conference_{conf_num}_field",
        desc=f"Field alignment for conference #{conf_num}",
        parent=conf_node,
        critical=True
    )
    field_sources = compile_sources(conf, ["field"])

    leaf_field = evaluator.add_leaf(
        id=f"conference_{conf_num}_field_topic",
        desc="Primary conference topic is computational science, HPC, parallel computing, scientific computing, or closely related field",
        parent=field_node,
        critical=True
    )
    # Build a concise claim for field relevance
    topics_text = ", ".join(conf.topics) if conf.topics else "computational science/HPC/parallel/scientific computing"
    claim_field = (
        f"The conference's primary focus includes {topics_text}, appropriate for research on parallel algorithms."
    )
    await evaluator.verify(
        claim=claim_field,
        node=leaf_field,
        sources=field_sources,
        additional_instruction="Use the 'scope', 'topics', or 'call for papers' page to confirm relevance to computational science/HPC/parallel computing."
    )

    evaluator.add_custom_node(
        result=len(field_sources) > 0,
        id=f"conference_{conf_num}_field_url",
        desc="URL reference provided for field information",
        parent=field_node,
        critical=True
    )

    # 5) Submission format requirements (critical)
    format_node = evaluator.add_parallel(
        id=f"conference_{conf_num}_format",
        desc=f"Submission format requirements for conference #{conf_num}",
        parent=conf_node,
        critical=True
    )
    format_sources = compile_sources(conf, ["format"])

    # 5.1 Accepts full research papers
    leaf_full_paper = evaluator.add_leaf(
        id=f"conference_{conf_num}_format_full_paper",
        desc="Conference accepts full research paper submissions (not just abstracts or posters)",
        parent=format_node,
        critical=True
    )
    claim_full_paper = "The conference accepts full research paper (regular paper) submissions, not only abstracts or posters."
    await evaluator.verify(
        claim=claim_full_paper,
        node=leaf_full_paper,
        sources=format_sources,
        additional_instruction="Check the submission categories on the official site (e.g., 'full papers', 'regular papers')."
    )

    # 5.2 Page limit between 8 and 15 pages for initial submission
    leaf_page_limit = evaluator.add_leaf(
        id=f"conference_{conf_num}_format_page_limit",
        desc="Initial submission page limit is between 8 and 15 pages",
        parent=format_node,
        critical=True
    )
    claim_page_limit = "The initial submission page limit for full papers is between 8 and 15 pages (inclusive)."
    await evaluator.verify(
        claim=claim_page_limit,
        node=leaf_page_limit,
        sources=format_sources,
        additional_instruction="Confirm the page limit range on the submissions/guidelines page; accept variants like '8–12 pages' or 'up to 15 pages'."
    )

    evaluator.add_custom_node(
        result=len(format_sources) > 0,
        id=f"conference_{conf_num}_format_url",
        desc="URL reference provided for format requirements",
        parent=format_node,
        critical=True
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
    Evaluate an answer for the 2026 conferences task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Conferences evaluated independently
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

    # IMPORTANT: Root must be non-critical to allow non-critical children (framework constraint)
    # We already set root as non-critical by default in initialize().

    # Extract conferences from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_conferences(),
        template_class=ConferencesExtraction,
        extraction_name="conferences_extraction"
    )

    # Prepare exactly four conference entries (pad with blanks or truncate)
    confs: List[SingleConference] = list(extracted.conferences[:4])
    while len(confs) < 4:
        confs.append(SingleConference())

    # Build verification subtrees for each of the four conferences
    for i in range(4):
        await verify_single_conference(evaluator, root, confs[i], i)

    # Return structured summary
    return evaluator.get_summary()