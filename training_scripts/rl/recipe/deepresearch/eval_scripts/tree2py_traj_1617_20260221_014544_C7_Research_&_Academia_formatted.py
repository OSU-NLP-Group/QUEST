import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "conference_selection_2026_europe_ai"
TASK_DESCRIPTION = """I am a researcher in artificial intelligence and machine learning seeking to submit my research work to an academic conference. Please identify an upcoming international conference that meets the following requirements:

1. The conference must be held in Europe between June 2026 and September 2026
2. The submission deadline must be after March 1, 2026 (so I still have time to prepare my submission)
3. The conference must focus on artificial intelligence, machine learning, or data science as its primary research area
4. Submissions must undergo a peer review process
5. The conference must publish proceedings of accepted papers
6. The proceedings must be indexed in recognized academic databases (such as IEEE Xplore, ACM Digital Library, Scopus, or Web of Science)
7. The conference must accept full research paper submissions (not limited to abstracts or posters only)
8. Registration information and fees must be publicly available
9. The conference must be organized by a recognized academic institution, professional society, or established conference series
10. The conference should preferably offer hybrid or virtual participation options
11. It would be preferable if the conference has been held at least once before
12. The conference must accept submissions in English
13. Clear submission guidelines including page limits or word counts must be specified
14. The conference must have an official website with comprehensive information

Please provide the conference name, dates, location, and reference URLs that verify each requirement.
"""


class RequirementSources(BaseModel):
    location_urls: List[str] = Field(default_factory=list)
    dates_urls: List[str] = Field(default_factory=list)
    deadline_urls: List[str] = Field(default_factory=list)
    field_urls: List[str] = Field(default_factory=list)
    peer_review_urls: List[str] = Field(default_factory=list)
    proceedings_urls: List[str] = Field(default_factory=list)
    indexing_urls: List[str] = Field(default_factory=list)
    full_papers_urls: List[str] = Field(default_factory=list)
    registration_urls: List[str] = Field(default_factory=list)
    organizing_urls: List[str] = Field(default_factory=list)
    hybrid_urls: List[str] = Field(default_factory=list)
    history_urls: List[str] = Field(default_factory=list)
    language_urls: List[str] = Field(default_factory=list)
    guidelines_urls: List[str] = Field(default_factory=list)
    website_urls: List[str] = Field(default_factory=list)
    general_urls: List[str] = Field(default_factory=list)


class ConferenceItem(BaseModel):
    name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    location: Optional[str] = None
    submission_deadline: Optional[str] = None
    official_website: Optional[str] = None
    research_area: Optional[str] = None
    page_limits: Optional[str] = None
    registration_fees: Optional[str] = None
    organizer: Optional[str] = None
    accepts_full_papers: Optional[str] = None
    indexing_databases: Optional[str] = None
    peer_review_process: Optional[str] = None
    proceedings_publication: Optional[str] = None
    hybrid_virtual_option: Optional[str] = None
    history: Optional[str] = None
    language: Optional[str] = None


class ConferenceExtraction(BaseModel):
    conferences: List[ConferenceItem] = Field(default_factory=list)
    sources: RequirementSources = Field(default_factory=RequirementSources)


def prompt_extract_conference() -> str:
    return (
        "Extract all conferences mentioned in the answer and their key attributes. "
        "For each conference, extract these fields exactly as written in the answer (use strings; if missing, set to null):\n"
        "- name\n"
        "- start_date\n"
        "- end_date\n"
        "- location\n"
        "- submission_deadline\n"
        "- official_website\n"
        "- research_area\n"
        "- page_limits (e.g., '10 pages', '8000 words')\n"
        "- registration_fees (text mentioning fee amounts or categories, if present)\n"
        "- organizer (organizing entity)\n"
        "- accepts_full_papers (statement indicating full paper submissions are accepted)\n"
        "- indexing_databases (text mentioning IEEE Xplore, ACM Digital Library, Scopus, Web of Science, etc.)\n"
        "- peer_review_process (statement indicating peer review)\n"
        "- proceedings_publication (statement indicating proceedings publication)\n"
        "- hybrid_virtual_option (statement indicating hybrid or virtual options)\n"
        "- history (statement indicating previous editions)\n"
        "- language (statement indicating submissions accepted in English)\n\n"
        "Also extract requirement-specific URLs that the answer cites to support each requirement. Only include valid URLs explicitly present in the answer. "
        "Populate the following arrays with URLs (if not present, return empty arrays):\n"
        "- location_urls\n"
        "- dates_urls\n"
        "- deadline_urls\n"
        "- field_urls\n"
        "- peer_review_urls\n"
        "- proceedings_urls\n"
        "- indexing_urls\n"
        "- full_papers_urls\n"
        "- registration_urls\n"
        "- organizing_urls\n"
        "- hybrid_urls\n"
        "- history_urls\n"
        "- language_urls\n"
        "- guidelines_urls\n"
        "- website_urls\n"
        "- general_urls\n\n"
        "Return JSON with two top-level keys: 'conferences' (array of objects with the above fields) and 'sources' (object with the URL arrays). "
        "If multiple conferences are provided, include them all; the evaluator will use the first."
    )


def _unique_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _merge_sources(primary: List[str], official: Optional[str], general: List[str]) -> List[str]:
    merged: List[str] = []
    merged.extend(primary or [])
    if official:
        merged.append(official)
    merged.extend(general or [])
    return _unique_urls(merged)


async def _add_requirement_verification(
    evaluator: Evaluator,
    parent,
    node_id: str,
    node_desc: str,
    claim_text: str,
    sources: List[str],
    critical: bool,
    additional_instruction: str,
) -> None:
    group = evaluator.add_sequential(
        id=f"{node_id}_group",
        desc=node_desc,
        parent=parent,
        critical=critical,
    )
    sources_exist = len(sources) > 0
    evaluator.add_custom_node(
        result=sources_exist,
        id=f"{node_id}_sources_provided",
        desc=f"Sources provided for {node_desc}",
        parent=group,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=node_desc,
        parent=group,
        critical=True if critical else False,
    )
    await evaluator.verify(
        claim=claim_text,
        node=leaf,
        sources=sources if sources_exist else None,
        additional_instruction=additional_instruction,
    )


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

    extraction = await evaluator.extract(
        prompt=prompt_extract_conference(),
        template_class=ConferenceExtraction,
        extraction_name="conference_extraction",
    )

    selected: ConferenceItem = extraction.conferences[0] if extraction.conferences else ConferenceItem()

    evaluator.add_custom_info(
        info={
            "selected_conference": {
                "name": selected.name,
                "start_date": selected.start_date,
                "end_date": selected.end_date,
                "location": selected.location,
                "submission_deadline": selected.submission_deadline,
                "official_website": selected.official_website,
                "organizer": selected.organizer,
            }
        },
        info_type="selection",
        info_name="selected_conference",
    )

    # Conference_Identification root node (parallel aggregator)
    conf_root = evaluator.add_parallel(
        id="Conference_Identification",
        desc="Identifies an academic conference that satisfies the specified research and professional requirements",
        parent=root,
        critical=False,
    )

    general_urls = extraction.sources.general_urls
    official = selected.official_website

    # Geographic_Location (Critical)
    loc_sources = _merge_sources(extraction.sources.location_urls, official, general_urls)
    loc_claim = f"The conference '{selected.name}' is held in Europe; the location listed is '{selected.location}'."
    await _add_requirement_verification(
        evaluator,
        conf_root,
        "Geographic_Location",
        "The conference is held in Europe",
        loc_claim,
        loc_sources,
        True,
        "Verify that the conference location (city/country) shown on the provided webpages is in Europe. Reasonable geographic knowledge is allowed.",
    )

    # Conference_Dates (Critical)
    dates_sources = _merge_sources(extraction.sources.dates_urls, official, general_urls)
    dates_claim = (
        f"The conference '{selected.name}' takes place between June 2026 and September 2026; "
        f"the dates provided are '{selected.start_date}' to '{selected.end_date}'."
    )
    await _add_requirement_verification(
        evaluator,
        conf_root,
        "Conference_Dates",
        "The conference takes place between June 2026 and September 2026",
        dates_claim,
        dates_sources,
        True,
        "Confirm the event dates on the official site or program pages fall within June 1, 2026 and September 30, 2026.",
    )

    # Submission_Deadline (Critical)
    deadline_sources = _merge_sources(extraction.sources.deadline_urls, official, general_urls)
    deadline_claim = (
        f"The submission deadline for '{selected.name}' is after March 1, 2026; specifically '{selected.submission_deadline}'."
    )
    await _add_requirement_verification(
        evaluator,
        conf_root,
        "Submission_Deadline",
        "The submission deadline is after March 1, 2026",
        deadline_claim,
        deadline_sources,
        True,
        "Verify the submission deadline date stated on the call for papers or submission page is strictly after March 1, 2026.",
    )

    # Research_Field (Critical)
    field_sources = _merge_sources(extraction.sources.field_urls, official, general_urls)
    field_claim = (
        f"The conference '{selected.name}' focuses on artificial intelligence, machine learning, or data science; "
        f"the stated research area is '{selected.research_area}'."
    )
    await _add_requirement_verification(
        evaluator,
        conf_root,
        "Research_Field",
        "The conference focuses on artificial intelligence, machine learning, or data science",
        field_claim,
        field_sources,
        True,
        "Check the scope or topics sections to confirm AI/ML/data science are primary focus areas.",
    )

    # Peer_Review_Process (Critical)
    review_sources = _merge_sources(extraction.sources.peer_review_urls, official, general_urls)
    review_claim = (
        f"The conference '{selected.name}' explicitly states that submissions undergo peer review; "
        f"evidence includes '{selected.peer_review_process}'."
    )
    await _add_requirement_verification(
        evaluator,
        conf_root,
        "Peer_Review_Process",
        "The conference explicitly states that submissions undergo peer review",
        review_claim,
        review_sources,
        True,
        "Confirm the site mentions peer review (e.g., blind review, reviewers, program committee review).",
    )

    # Proceedings_Publication (Critical)
    proceedings_sources = _merge_sources(extraction.sources.proceedings_urls, official, general_urls)
    proceedings_claim = (
        f"The conference '{selected.name}' publishes proceedings of accepted papers; "
        f"evidence includes '{selected.proceedings_publication}'."
    )
    await _add_requirement_verification(
        evaluator,
        conf_root,
        "Proceedings_Publication",
        "The conference publishes proceedings of accepted papers",
        proceedings_claim,
        proceedings_sources,
        True,
        "Confirm the conference states proceedings publication (e.g., Springer LNCS, IEEE, ACM, or similar).",
    )

    # Indexing_Information (Critical)
    indexing_sources = _merge_sources(extraction.sources.indexing_urls, official, general_urls)
    indexing_claim = (
        f"The proceedings of '{selected.name}' are indexed in recognized academic databases; "
        f"the mentioned indexing is '{selected.indexing_databases}'."
    )
    await _add_requirement_verification(
        evaluator,
        conf_root,
        "Indexing_Information",
        "The conference proceedings are indexed in recognized academic databases (e.g., IEEE Xplore, ACM Digital Library, Scopus, Web of Science)",
        indexing_claim,
        indexing_sources,
        True,
        "Confirm explicit indexing statements (IEEE Xplore, ACM DL, Scopus, Web of Science). If the site states Springer LNCS indexing, it commonly implies Scopus/WoS but prefer explicit mentions.",
    )

    # Presentation_Formats (Critical)
    full_paper_sources = _merge_sources(extraction.sources.full_papers_urls, official, general_urls)
    full_paper_claim = (
        f"The conference '{selected.name}' accepts full research paper submissions; "
        f"evidence includes '{selected.accepts_full_papers}'."
    )
    await _add_requirement_verification(
        evaluator,
        conf_root,
        "Presentation_Formats",
        "The conference accepts full paper submissions (not just abstracts or posters only)",
        full_paper_claim,
        full_paper_sources,
        True,
        "Check submission tracks/types to confirm 'full papers' are accepted (not limited to abstracts or posters).",
    )

    # Registration_Details (Critical)
    reg_sources = _merge_sources(extraction.sources.registration_urls, official, general_urls)
    reg_claim = (
        f"Registration information and fees for '{selected.name}' are publicly available; "
        f"the answer mentions '{selected.registration_fees}'."
    )
    await _add_requirement_verification(
        evaluator,
        conf_root,
        "Registration_Details",
        "Registration information and fees are publicly available on the conference website",
        reg_claim,
        reg_sources,
        True,
        "Verify that the registration page shows fee amounts or a fee table. If fees are not yet posted or 'TBA', do not consider supported.",
    )

    # Organizing_Entity (Critical)
    org_sources = _merge_sources(extraction.sources.organizing_urls, official, general_urls)
    org_claim = (
        f"The conference '{selected.name}' is organized by a recognized academic institution, professional society, or established series; "
        f"the organizer stated is '{selected.organizer}'."
    )
    await _add_requirement_verification(
        evaluator,
        conf_root,
        "Organizing_Entity",
        "The conference is organized by a recognized academic institution, professional society, or established conference series",
        org_claim,
        org_sources,
        True,
        "Confirm the organizing entity is a university department, society (e.g., IEEE/ACM), or an established conference series.",
    )

    # Hybrid_Virtual_Option (Non-Critical)
    hybrid_sources = _merge_sources(extraction.sources.hybrid_urls, official, general_urls)
    hybrid_claim = (
        f"The conference '{selected.name}' offers hybrid or virtual participation options; "
        f"evidence includes '{selected.hybrid_virtual_option}'."
    )
    await _add_requirement_verification(
        evaluator,
        conf_root,
        "Hybrid_Virtual_Option",
        "The conference offers hybrid or virtual participation options",
        hybrid_claim,
        hybrid_sources,
        False,
        "Check if the site mentions online, hybrid, or remote participation options.",
    )

    # Conference_History (Non-Critical)
    history_sources = _merge_sources(extraction.sources.history_urls, official, general_urls)
    history_claim = (
        f"The conference '{selected.name}' has been held at least once before; "
        f"evidence includes '{selected.history}'."
    )
    await _add_requirement_verification(
        evaluator,
        conf_root,
        "Conference_History",
        "The conference has been held at least once before (not a first-edition conference)",
        history_claim,
        history_sources,
        False,
        "Verify previous years/editions page, archive, or history statement.",
    )

    # Language_Requirement (Critical)
    language_sources = _merge_sources(extraction.sources.language_urls, official, general_urls)
    language_claim = (
        f"The conference '{selected.name}' accepts submissions in English; "
        f"the language statement is '{selected.language}'."
    )
    await _add_requirement_verification(
        evaluator,
        conf_root,
        "Language_Requirement",
        "The conference accepts submissions in English",
        language_claim,
        language_sources,
        True,
        "Confirm the author guidelines or submission page explicitly state English submissions are accepted.",
    )

    # Submission_Guidelines (Critical)
    guide_sources = _merge_sources(extraction.sources.guidelines_urls, official, general_urls)
    guide_claim = (
        f"The conference '{selected.name}' provides clear submission guidelines including page limits or word counts; "
        f"the stated limit is '{selected.page_limits}'."
    )
    await _add_requirement_verification(
        evaluator,
        conf_root,
        "Submission_Guidelines",
        "Clear submission guidelines including page limits or word counts are specified",
        guide_claim,
        guide_sources,
        True,
        "Check author guidelines/CfP for explicit page limits or word count constraints.",
    )

    # Conference_Website (Critical)
    website_sources = _merge_sources(extraction.sources.website_urls, official, general_urls)
    website_claim = (
        f"The conference '{selected.name}' has an official website with detailed information; the official URL is '{selected.official_website}'."
    )
    await _add_requirement_verification(
        evaluator,
        conf_root,
        "Conference_Website",
        "The conference has an official website with detailed information",
        website_claim,
        website_sources if website_sources else ([selected.official_website] if selected.official_website else []),
        True,
        "Verify that the given URL appears to be the official conference website and includes major sections (CFP, dates, location, submission, registration).",
    )

    return evaluator.get_summary()