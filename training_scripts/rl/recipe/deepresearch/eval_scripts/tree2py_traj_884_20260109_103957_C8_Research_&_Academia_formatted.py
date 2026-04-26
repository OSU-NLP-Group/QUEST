import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "ieee_conf_2025_eval"
TASK_DESCRIPTION = """Identify an IEEE-sponsored international academic conference scheduled for 2025 that meets ALL of the following requirements:

Venue & Logistics:
- Hosted at a university or academic institution venue
- Venue capacity of at least 800-1,000 attendees
- Conference duration of 3-4 days
- Attracts participants from multiple countries internationally

Registration & Attendance:
- Early-bird IEEE member registration fee between $800-$1,200
- Student member registration available at reduced rates ($400-$600)
- Offers virtual or hybrid attendance options

Program Structure:
- Features multi-track parallel sessions with at least 4-6 concurrent tracks
- Includes keynote speaker presentations
- Offers both oral presentation and poster presentation formats

Submission & Review:
- Has a formal peer review process with multiple reviewers per paper
- Published acceptance rate between 20-40%
- Clear submission deadlines with paper length requirements of 6-10 pages

Publication:
- Conference proceedings published by IEEE, ACM, or Springer
- Proceedings indexed in major databases (Scopus, Web of Science, or similar)
- Accepted papers receive DOI assignment
- Has an established program committee with at least 10 members

Provide the conference name, website URL, and specific details demonstrating how it meets each of these requirements.
"""


class FieldWithSources(BaseModel):
    text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ConferenceBasic(BaseModel):
    conference_name: Optional[str] = None
    website_url: Optional[str] = None
    candidates: List[str] = Field(default_factory=list)
    year: FieldWithSources = Field(default_factory=FieldWithSources)


class SponsorshipInfo(BaseModel):
    ieee_sponsorship: FieldWithSources = Field(default_factory=FieldWithSources)


class VenueInfo(BaseModel):
    academic_venue: FieldWithSources = Field(default_factory=FieldWithSources)
    capacity: FieldWithSources = Field(default_factory=FieldWithSources)
    duration: FieldWithSources = Field(default_factory=FieldWithSources)
    international_scope: FieldWithSources = Field(default_factory=FieldWithSources)


class RegistrationInfo(BaseModel):
    early_member_fee: FieldWithSources = Field(default_factory=FieldWithSources)
    student_member_fee: FieldWithSources = Field(default_factory=FieldWithSources)
    virtual_or_hybrid: FieldWithSources = Field(default_factory=FieldWithSources)


class ProgramStructureInfo(BaseModel):
    concurrent_tracks: FieldWithSources = Field(default_factory=FieldWithSources)
    keynotes: FieldWithSources = Field(default_factory=FieldWithSources)
    oral_and_poster: FieldWithSources = Field(default_factory=FieldWithSources)


class SubmissionReviewInfo(BaseModel):
    peer_review: FieldWithSources = Field(default_factory=FieldWithSources)
    acceptance_rate: FieldWithSources = Field(default_factory=FieldWithSources)
    deadlines: FieldWithSources = Field(default_factory=FieldWithSources)
    paper_length: FieldWithSources = Field(default_factory=FieldWithSources)


class PublicationInfo(BaseModel):
    proceedings_publisher: FieldWithSources = Field(default_factory=FieldWithSources)
    indexing: FieldWithSources = Field(default_factory=FieldWithSources)
    doi_assignment: FieldWithSources = Field(default_factory=FieldWithSources)
    program_committee_size: FieldWithSources = Field(default_factory=FieldWithSources)


class ConferenceExtraction(BaseModel):
    basic: ConferenceBasic = Field(default_factory=ConferenceBasic)
    sponsorship: SponsorshipInfo = Field(default_factory=SponsorshipInfo)
    venue: VenueInfo = Field(default_factory=VenueInfo)
    registration: RegistrationInfo = Field(default_factory=RegistrationInfo)
    program: ProgramStructureInfo = Field(default_factory=ProgramStructureInfo)
    submission: SubmissionReviewInfo = Field(default_factory=SubmissionReviewInfo)
    publication: PublicationInfo = Field(default_factory=PublicationInfo)
    constraint_urls_global: List[str] = Field(default_factory=list)


def prompt_extract_conference() -> str:
    return """
    Extract structured information about the single conference identified in the answer. Return JSON that follows the schema below.
    Rules:
    - Extract only what is explicitly stated in the answer.
    - For each FieldWithSources, 'text' should be a short quote or paraphrase from the answer showing the claimed detail; 'urls' must be URLs listed in the answer that support that detail (registration page, CFP/guidelines, program page, venue page, publisher page, etc.).
    - If a required detail is missing, set 'text' to null and 'urls' to an empty list.

    Schema:
    {
      "basic": {
        "conference_name": string|null,
        "website_url": string|null,
        "candidates": [string],   // all conference names mentioned; primary target should be the one in 'conference_name'
        "year": { "text": string|null, "urls": [string] }
      },
      "sponsorship": {
        "ieee_sponsorship": { "text": string|null, "urls": [string] }
      },
      "venue": {
        "academic_venue": { "text": string|null, "urls": [string] },     // shows venue is a university/academic institution
        "capacity": { "text": string|null, "urls": [string] },           // shows venue capacity (numbers or capacity statement)
        "duration": { "text": string|null, "urls": [string] },           // shows 3–4 days duration (dates or schedule)
        "international_scope": { "text": string|null, "urls": [string] } // shows international participants/multi-country
      },
      "registration": {
        "early_member_fee": { "text": string|null, "urls": [string] },   // IEEE member early-bird fee
        "student_member_fee": { "text": string|null, "urls": [string] }, // student member fee
        "virtual_or_hybrid": { "text": string|null, "urls": [string] }   // virtual/hybrid options
      },
      "program": {
        "concurrent_tracks": { "text": string|null, "urls": [string] },  // 4–6 concurrent tracks
        "keynotes": { "text": string|null, "urls": [string] },           // keynote talks
        "oral_and_poster": { "text": string|null, "urls": [string] }     // oral and poster formats
      },
      "submission": {
        "peer_review": { "text": string|null, "urls": [string] },        // multiple reviewers per paper
        "acceptance_rate": { "text": string|null, "urls": [string] },    // acceptance rate shown (20–40%)
        "deadlines": { "text": string|null, "urls": [string] },          // clear submission deadlines
        "paper_length": { "text": string|null, "urls": [string] }        // 6–10 pages
      },
      "publication": {
        "proceedings_publisher": { "text": string|null, "urls": [string] }, // IEEE/ACM/Springer
        "indexing": { "text": string|null, "urls": [string] },               // Scopus/Web of Science or similar
        "doi_assignment": { "text": string|null, "urls": [string] },         // DOI assignment mentioned
        "program_committee_size": { "text": string|null, "urls": [string] }  // shows at least 10 PC members
      },
      "constraint_urls_global": [string] // any extra URLs that support constraints, if provided in the answer
    }

    Extract URLs exactly as shown in the answer; include full URLs with protocol.
    """


def _combine_sources(*sources_lists: List[str], extra: Optional[List[str]] = None, base_url: Optional[str] = None) -> List[str]:
    urls: List[str] = []
    for lst in sources_lists:
        if lst:
            urls.extend([u for u in lst if isinstance(u, str) and u.strip() != ""])
    if extra:
        urls.extend([u for u in extra if isinstance(u, str) and u.strip() != ""])
    if base_url and isinstance(base_url, str) and base_url.strip() != "":
        urls.append(base_url)
    # deduplicate while preserving order
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


def _has_evidence(f: FieldWithSources) -> bool:
    return bool(f and f.text and f.text.strip() and f.urls and len(f.urls) > 0)


async def _build_response_completeness(
    evaluator: Evaluator,
    parent_node,
    data: ConferenceExtraction
) -> None:
    resp_node = evaluator.add_parallel(
        id="response_completeness",
        desc="Response includes required identification fields and supporting details.",
        parent=parent_node,
        critical=True
    )

    # Single conference identified
    candidates = data.basic.candidates or []
    name = data.basic.conference_name or ""
    single_conf = False
    if name.strip():
        # If candidates explicitly listed: must be exactly one unique name
        if len({c.strip() for c in candidates if c and c.strip()}) <= 1:
            single_conf = True
        else:
            single_conf = False
    else:
        single_conf = False

    evaluator.add_custom_node(
        result=single_conf,
        id="single_conference_identified",
        desc="Identifies one specific conference (not multiple candidates).",
        parent=resp_node,
        critical=True
    )

    # Conference Name Provided
    evaluator.add_custom_node(
        result=bool(name and name.strip()),
        id="conference_name_provided",
        desc="Provides the conference name.",
        parent=resp_node,
        critical=True
    )

    # Website URL Provided
    website = data.basic.website_url or ""
    evaluator.add_custom_node(
        result=bool(website and website.strip().startswith(("http://", "https://"))),
        id="website_url_provided",
        desc="Provides a conference website URL.",
        parent=resp_node,
        critical=True
    )

    # Evidence for each constraint provided
    evidence_ok = all([
        _has_evidence(data.sponsorship.ieee_sponsorship),
        _has_evidence(data.basic.year),
        _has_evidence(data.venue.academic_venue),
        _has_evidence(data.venue.capacity),
        _has_evidence(data.venue.duration),
        _has_evidence(data.venue.international_scope),
        _has_evidence(data.registration.early_member_fee),
        _has_evidence(data.registration.student_member_fee),
        _has_evidence(data.registration.virtual_or_hybrid),
        _has_evidence(data.program.concurrent_tracks),
        _has_evidence(data.program.keynotes),
        _has_evidence(data.program.oral_and_poster),
        _has_evidence(data.submission.peer_review),
        _has_evidence(data.submission.acceptance_rate),
        _has_evidence(data.submission.deadlines),
        _has_evidence(data.submission.paper_length),
        _has_evidence(data.publication.proceedings_publisher),
        _has_evidence(data.publication.indexing),
        _has_evidence(data.publication.doi_assignment),
        _has_evidence(data.publication.program_committee_size),
    ])

    evaluator.add_custom_node(
        result=evidence_ok,
        id="evidence_for_each_constraint_provided",
        desc="Provides specific supporting details demonstrating how the conference meets each listed requirement (not merely asserting compliance).",
        parent=resp_node,
        critical=True
    )


async def _build_conference_constraints(
    evaluator: Evaluator,
    parent_node,
    data: ConferenceExtraction
) -> None:
    cons_node = evaluator.add_parallel(
        id="conference_constraints",
        desc="Conference satisfies all stated constraints.",
        parent=parent_node,
        critical=True
    )

    # IEEE Sponsored
    node_ieee = evaluator.add_leaf(
        id="ieee_sponsored",
        desc="Conference is IEEE-sponsored (as stated by IEEE or official conference materials).",
        parent=cons_node,
        critical=True
    )
    ieee_sources = _combine_sources(
        data.sponsorship.ieee_sponsorship.urls,
        extra=data.constraint_urls_global,
        base_url=data.basic.website_url
    )
    claim_ieee = f"The conference '{data.basic.conference_name or 'the conference'}' is IEEE-sponsored or technically co-sponsored by IEEE."
    await evaluator.verify(
        claim=claim_ieee,
        node=node_ieee,
        sources=ieee_sources,
        additional_instruction="Look for official statements such as 'sponsored by IEEE', 'technically co-sponsored by IEEE', or IEEE logos on official materials."
    )

    # Conference Year 2025
    node_year = evaluator.add_leaf(
        id="conference_year_2025",
        desc="Conference is scheduled for 2025.",
        parent=cons_node,
        critical=True
    )
    year_sources = _combine_sources(
        data.basic.year.urls,
        extra=data.constraint_urls_global,
        base_url=data.basic.website_url
    )
    claim_year = "The event is scheduled in calendar year 2025."
    await evaluator.verify(
        claim=claim_year,
        node=node_year,
        sources=year_sources,
        additional_instruction="Confirm the event dates (start/end) or schedule clearly fall within the year 2025."
    )

    # Venue Academic Institution
    node_venue_acad = evaluator.add_leaf(
        id="venue_academic_institution",
        desc="Hosted at a university or academic institution venue.",
        parent=cons_node,
        critical=True
    )
    venue_acad_sources = _combine_sources(
        data.venue.academic_venue.urls,
        extra=data.constraint_urls_global,
        base_url=data.basic.website_url
    )
    claim_venue_acad = "The conference venue is a university or academic institution (campus or facility owned/operated by such an institution)."
    await evaluator.verify(
        claim=claim_venue_acad,
        node=node_venue_acad,
        sources=venue_acad_sources,
        additional_instruction="Confirm venue affiliation with a university or academic institution (e.g., 'University Hall', 'Campus Center', official university website pages)."
    )

    # Venue Capacity 800–1000
    node_capacity = evaluator.add_leaf(
        id="venue_capacity_range_800_1000",
        desc="Venue capacity is within the stated 800–1,000 attendee range (inclusive), as required by the prompt’s capacity specification.",
        parent=cons_node,
        critical=True
    )
    capacity_sources = _combine_sources(
        data.venue.capacity.urls,
        extra=data.constraint_urls_global,
        base_url=data.basic.website_url
    )
    claim_capacity = "The venue capacity (for the main conference space or combined spaces used) is within 800 to 1,000 attendees (inclusive)."
    await evaluator.verify(
        claim=claim_capacity,
        node=node_capacity,
        sources=capacity_sources,
        additional_instruction="Use official venue specs or conference logistics pages. Approximations are acceptable if clearly stated on official sources and within the range."
    )

    # International Scope
    node_international = evaluator.add_leaf(
        id="international_scope",
        desc="International conference attracting participants from multiple countries.",
        parent=cons_node,
        critical=True
    )
    intl_sources = _combine_sources(
        data.venue.international_scope.urls,
        extra=data.constraint_urls_global,
        base_url=data.basic.website_url
    )
    claim_international = "It is an international conference attracting participants from multiple countries."
    await evaluator.verify(
        claim=claim_international,
        node=node_international,
        sources=intl_sources,
        additional_instruction="Look for statements like 'international', lists of international committees, past attendees from multiple countries, or global participation indicators on official pages."
    )

    # Conference Duration 3–4 days
    node_duration = evaluator.add_leaf(
        id="conference_duration",
        desc="Conference duration is 3–4 days.",
        parent=cons_node,
        critical=True
    )
    duration_sources = _combine_sources(
        data.venue.duration.urls,
        extra=data.constraint_urls_global,
        base_url=data.basic.website_url
    )
    claim_duration = "The conference runs for 3 to 4 days (inclusive), based on official schedule or dates."
    await evaluator.verify(
        claim=claim_duration,
        node=node_duration,
        sources=duration_sources,
        additional_instruction="Check the official program schedule or event dates; count consecutive days of the main conference program."
    )

    # Registration & Attendance
    reg_node = evaluator.add_parallel(
        id="registration_fees",
        desc="Registration fee structure meets stated ranges and attendance-mode requirement.",
        parent=cons_node,
        critical=True
    )

    node_early = evaluator.add_leaf(
        id="early_bird_ieee_member_fee_range",
        desc="Early-bird IEEE member registration fee is between $800 and $1,200.",
        parent=reg_node,
        critical=True
    )
    reg_sources = _combine_sources(
        data.registration.early_member_fee.urls,
        data.registration.student_member_fee.urls,
        data.registration.virtual_or_hybrid.urls,
        extra=data.constraint_urls_global,
        base_url=data.basic.website_url
    )
    claim_early = "The early-bird IEEE member registration fee falls between $800 and $1,200."
    await evaluator.verify(
        claim=claim_early,
        node=node_early,
        sources=reg_sources,
        additional_instruction="Use the official registration/fees page; confirm member category, early-bird timing, and fee within the range."
    )

    node_student = evaluator.add_leaf(
        id="student_member_fee_range",
        desc="Student member registration is available at a reduced rate between $400 and $600.",
        parent=reg_node,
        critical=True
    )
    claim_student = "A student member registration option is offered at a reduced rate between $400 and $600."
    await evaluator.verify(
        claim=claim_student,
        node=node_student,
        sources=reg_sources,
        additional_instruction="Confirm student category fees and IEEE member status if applicable; verify amount within the stated range."
    )

    node_virtual = evaluator.add_leaf(
        id="virtual_or_hybrid_option",
        desc="Offers virtual or hybrid attendance options.",
        parent=reg_node,
        critical=True
    )
    claim_virtual = "The conference offers virtual or hybrid attendance options."
    await evaluator.verify(
        claim=claim_virtual,
        node=node_virtual,
        sources=reg_sources,
        additional_instruction="Look for any official mention of remote/online participation, hybrid format, or virtual attendance options."
    )

    # Program Structure
    prog_node = evaluator.add_parallel(
        id="program_structure",
        desc="Program structure meets track, keynote, and presentation-format requirements.",
        parent=cons_node,
        critical=True
    )
    prog_sources = _combine_sources(
        data.program.concurrent_tracks.urls,
        data.program.keynotes.urls,
        data.program.oral_and_poster.urls,
        extra=data.constraint_urls_global,
        base_url=data.basic.website_url
    )

    node_tracks = evaluator.add_leaf(
        id="multi_track_concurrent_tracks_range_4_6",
        desc="Features multi-track parallel sessions with 4–6 concurrent tracks (inclusive), matching the prompt’s stated track concurrency requirement.",
        parent=prog_node,
        critical=True
    )
    claim_tracks = "The program features multi-track parallel sessions with between 4 and 6 concurrent tracks (inclusive)."
    await evaluator.verify(
        claim=claim_tracks,
        node=node_tracks,
        sources=prog_sources,
        additional_instruction="Confirm number of simultaneous tracks/sessions from program overview or schedule; synonyms like 'streams' or 'parallel tracks' are acceptable."
    )

    node_keynotes = evaluator.add_leaf(
        id="keynote_presentations",
        desc="Includes keynote speaker presentations.",
        parent=prog_node,
        critical=True
    )
    claim_keynotes = "The conference includes keynote speaker presentations."
    await evaluator.verify(
        claim=claim_keynotes,
        node=node_keynotes,
        sources=prog_sources,
        additional_instruction="Look for 'Keynote' pages, invited keynote lists, or program schedule explicitly featuring keynotes."
    )

    node_formats = evaluator.add_leaf(
        id="oral_and_poster_formats",
        desc="Offers both oral presentation and poster presentation formats.",
        parent=prog_node,
        critical=True
    )
    claim_formats = "Both oral and poster presentation formats are offered."
    await evaluator.verify(
        claim=claim_formats,
        node=node_formats,
        sources=prog_sources,
        additional_instruction="Check author guidelines or program info mentioning oral talks and poster sessions."
    )

    # Submission & Review
    sub_node = evaluator.add_parallel(
        id="submission_and_review",
        desc="Submission and peer review constraints are satisfied.",
        parent=cons_node,
        critical=True
    )
    sub_sources = _combine_sources(
        data.submission.peer_review.urls,
        data.submission.acceptance_rate.urls,
        data.submission.deadlines.urls,
        data.submission.paper_length.urls,
        extra=data.constraint_urls_global,
        base_url=data.basic.website_url
    )

    node_peer = evaluator.add_leaf(
        id="peer_review_multiple_reviewers",
        desc="Has a formal peer review process with multiple reviewers per paper.",
        parent=sub_node,
        critical=True
    )
    claim_peer = "Each paper undergoes a formal peer review process with multiple reviewers per submission."
    await evaluator.verify(
        claim=claim_peer,
        node=node_peer,
        sources=sub_sources,
        additional_instruction="Look for statements about TPC review, at least two or three reviewers per paper, or standard peer-review language in CFP/guidelines."
    )

    node_accept = evaluator.add_leaf(
        id="acceptance_rate_range",
        desc="Published acceptance rate is between 20% and 40%.",
        parent=sub_node,
        critical=True
    )
    claim_accept = "The published acceptance rate for the conference is between 20% and 40%."
    await evaluator.verify(
        claim=claim_accept,
        node=node_accept,
        sources=sub_sources,
        additional_instruction="Confirm an explicitly stated acceptance rate (historical or current) within the 20–40% range."
    )

    node_deadlines = evaluator.add_leaf(
        id="submission_deadlines_clear",
        desc="Has clear submission deadlines.",
        parent=sub_node,
        critical=True
    )
    claim_deadlines = "The conference has clear submission deadlines stated on official pages."
    await evaluator.verify(
        claim=claim_deadlines,
        node=node_deadlines,
        sources=sub_sources,
        additional_instruction="Verify calendar or key dates page shows deadlines for paper submission, notification, camera-ready, etc."
    )

    node_pages = evaluator.add_leaf(
        id="paper_length_requirement",
        desc="Paper length requirements are stated as 6–10 pages.",
        parent=sub_node,
        critical=True
    )
    claim_pages = "The paper length requirements are between 6 and 10 pages (inclusive), excluding references when specified."
    await evaluator.verify(
        claim=claim_pages,
        node=node_pages,
        sources=sub_sources,
        additional_instruction="Check author guidelines; allow variants like 'up to 8 pages' or '6–10 pages' and note if references are excluded; requirement must fall within the 6–10 range."
    )

    # Publication
    pub_node = evaluator.add_parallel(
        id="publication",
        desc="Publication and committee constraints are satisfied.",
        parent=cons_node,
        critical=True
    )
    pub_sources = _combine_sources(
        data.publication.proceedings_publisher.urls,
        data.publication.indexing.urls,
        data.publication.doi_assignment.urls,
        data.publication.program_committee_size.urls,
        extra=data.constraint_urls_global,
        base_url=data.basic.website_url
    )

    node_publisher = evaluator.add_leaf(
        id="proceedings_publisher",
        desc="Proceedings are published by IEEE, ACM, or Springer.",
        parent=pub_node,
        critical=True
    )
    claim_publisher = "The conference proceedings are published by IEEE, ACM, or Springer."
    await evaluator.verify(
        claim=claim_publisher,
        node=node_publisher,
        sources=pub_sources,
        additional_instruction="Look for explicit publisher statements (IEEE Xplore, ACM ICPS, Springer LNCS/LNEE, etc.)."
    )

    node_indexing = evaluator.add_leaf(
        id="proceedings_indexed",
        desc="Proceedings are indexed in major databases (Scopus, Web of Science, or similar).",
        parent=pub_node,
        critical=True
    )
    claim_indexing = "The proceedings are indexed in major databases such as Scopus or Web of Science (or similar recognized indices)."
    await evaluator.verify(
        claim=claim_indexing,
        node=node_indexing,
        sources=pub_sources,
        additional_instruction="Confirm indexing statements; acceptable synonyms include 'ISI Web of Science', 'Scopus', 'EI Compendex'."
    )

    node_doi = evaluator.add_leaf(
        id="doi_assigned",
        desc="Accepted papers receive DOI assignment.",
        parent=pub_node,
        critical=True
    )
    claim_doi = "Accepted papers receive DOI assignment."
    await evaluator.verify(
        claim=claim_doi,
        node=node_doi,
        sources=pub_sources,
        additional_instruction="Check publisher policies or proceedings info indicating DOIs (e.g., IEEE Xplore assigns DOIs to published papers)."
    )

    node_pc = evaluator.add_leaf(
        id="program_committee_size",
        desc="Has an established program committee with at least 10 members.",
        parent=pub_node,
        critical=True
    )
    claim_pc = "The program committee (TPC) has at least 10 members."
    await evaluator.verify(
        claim=claim_pc,
        node=node_pc,
        sources=pub_sources,
        additional_instruction="Look for a TPC page listing names; count should be at least 10."
    )


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

    extracted = await evaluator.extract(
        prompt=prompt_extract_conference(),
        template_class=ConferenceExtraction,
        extraction_name="conference_extraction"
    )

    conf_eval_node = evaluator.add_parallel(
        id="conference_evaluation",
        desc="Evaluate whether the response identifies one IEEE-sponsored international academic conference scheduled for 2025 and demonstrates it meets all listed constraints, including required output fields.",
        parent=root,
        critical=True
    )

    await _build_response_completeness(evaluator, conf_eval_node, extracted)
    await _build_conference_constraints(evaluator, conf_eval_node, extracted)

    # Optional: add custom info for debugging
    evaluator.add_custom_info(
        {
            "conference_name": extracted.basic.conference_name,
            "website_url": extracted.basic.website_url,
            "total_extracted_urls": sum([
                len(extracted.sponsorship.ieee_sponsorship.urls),
                len(extracted.basic.year.urls),
                len(extracted.venue.academic_venue.urls),
                len(extracted.venue.capacity.urls),
                len(extracted.venue.duration.urls),
                len(extracted.venue.international_scope.urls),
                len(extracted.registration.early_member_fee.urls),
                len(extracted.registration.student_member_fee.urls),
                len(extracted.registration.virtual_or_hybrid.urls),
                len(extracted.program.concurrent_tracks.urls),
                len(extracted.program.keynotes.urls),
                len(extracted.program.oral_and_poster.urls),
                len(extracted.submission.peer_review.urls),
                len(extracted.submission.acceptance_rate.urls),
                len(extracted.submission.deadlines.urls),
                len(extracted.submission.paper_length.urls),
                len(extracted.publication.proceedings_publisher.urls),
                len(extracted.publication.indexing.urls),
                len(extracted.publication.doi_assignment.urls),
                len(extracted.publication.program_committee_size.urls),
                len(extracted.constraint_urls_global),
            ])
        },
        info_type="debug",
        info_name="extraction_summary"
    )

    return evaluator.get_summary()