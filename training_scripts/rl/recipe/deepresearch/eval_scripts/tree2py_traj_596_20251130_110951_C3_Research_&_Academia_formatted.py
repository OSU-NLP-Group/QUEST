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
TASK_ID = "mars_electrical_discharges_nature_2025"
TASK_DESCRIPTION = (
    "In November 2025, a research team published a paper in the journal Nature reporting the first in situ detection "
    "of atmospheric electrical discharges on Mars, identified through acoustic measurements captured by NASA's "
    "Perseverance rover's SuperCam microphone. Identify this publication and provide the following information: "
    "(1) the complete publication reference including journal name, volume, page numbers, and the full paper title; "
    "(2) a reference URL to the publication; (3) the name of the lead (first) author of the paper; "
    "(4) the full name of the lead author's primary institutional affiliation; "
    "(5) the city and country where this research institution is located; "
    "(6) the total number of hours of microphone recordings that were analyzed in the study and the timespan these "
    "recordings covered (expressed in both Martian years and Earth days); and (7) the total number of electrical "
    "discharge events that were detected in the study."
)

# Expected ground-truth values (for verification claims)
EXPECTED_TITLE = "Detection of triboelectric discharges during dust events on Mars"
EXPECTED_JOURNAL = "Nature"
EXPECTED_VOLUME = "647"
EXPECTED_PAGES = "865–869"  # Allow hyphen/en dash variants during verification
EXPECTED_PUB_MONTH = "November"
EXPECTED_PUB_YEAR = "2025"
EXPECTED_LEAD_AUTHOR = "Baptiste Chide"
EXPECTED_PRIMARY_AFFILIATION = "Institut de Recherche en Astrophysique et Planétologie (IRAP)"
EXPECTED_CITY = "Toulouse"
EXPECTED_COUNTRY = "France"
EXPECTED_RECORDING_HOURS = "28"  # hours
EXPECTED_TIMESPAN_MARTIAN_YEARS = "2"  # two Martian years
EXPECTED_TIMESPAN_EARTH_DAYS = "1374"  # 1,374 Earth days (allow comma variants)
EXPECTED_EVENT_COUNT = "55"

# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class PublicationInfo(BaseModel):
    title: Optional[str] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    pages: Optional[str] = None
    publication_month: Optional[str] = None
    publication_year: Optional[str] = None
    publication_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


class AuthorInstitutionInfo(BaseModel):
    lead_author_name: Optional[str] = None
    lead_author_primary_affiliation: Optional[str] = None
    institution_city: Optional[str] = None
    institution_country: Optional[str] = None


class StudyQuantDetails(BaseModel):
    recording_hours_analyzed: Optional[str] = None
    timespan_martian_years: Optional[str] = None
    timespan_earth_days: Optional[str] = None
    discharge_event_count: Optional[str] = None


class ScopeMethodInfo(BaseModel):
    phenomenon_phrase: Optional[str] = None
    method_phrase: Optional[str] = None
    event_association_phrase: Optional[str] = None


class MarsDischargePaperExtraction(BaseModel):
    publication: Optional[PublicationInfo] = None
    author_institution: Optional[AuthorInstitutionInfo] = None
    study: Optional[StudyQuantDetails] = None
    scope_method: Optional[ScopeMethodInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_paper_info() -> str:
    return """
    Extract the requested structured information from the provided answer about the specified Nature paper (Nov 2025).
    IMPORTANT: Extract only what is explicitly present in the answer text. Do not invent or infer missing items.

    Return a JSON object with the following nested structure:

    {
      "publication": {
        "title": string | null,                          // full paper title as written in the answer
        "journal": string | null,                        // e.g., "Nature"
        "volume": string | null,                         // e.g., "647"
        "pages": string | null,                          // preserve formatting e.g., "865–869" or "865-869"
        "publication_month": string | null,              // e.g., "November"
        "publication_year": string | null,               // e.g., "2025"
        "publication_url": string | null,                // the primary reference URL to the publication (from the answer)
        "additional_urls": string[]                      // any other URLs the answer cites relevant to the paper (e.g., Nature, Nature article page, arXiv, NASA press pages)
      },
      "author_institution": {
        "lead_author_name": string | null,               // lead (first) author name as stated
        "lead_author_primary_affiliation": string | null,// full name of lead author's primary institution (e.g., "Institut de Recherche en Astrophysique et Planétologie (IRAP)")
        "institution_city": string | null,               // e.g., "Toulouse"
        "institution_country": string | null             // e.g., "France"
      },
      "study": {
        "recording_hours_analyzed": string | null,       // e.g., "28 hours" or "28"
        "timespan_martian_years": string | null,         // e.g., "2" or "two"
        "timespan_earth_days": string | null,            // e.g., "1374" or "1,374"
        "discharge_event_count": string | null           // e.g., "55"
      },
      "scope_method": {
        "phenomenon_phrase": string | null,              // e.g., "triboelectric discharges" or equivalent phrase in the answer
        "method_phrase": string | null,                  // phrase describing the method, e.g., "acoustic measurements from Perseverance's SuperCam microphone"
        "event_association_phrase": string | null        // phrase noting association with dust devils and dust storm convective fronts
      }
    }

    GUIDANCE:
    - For URLs, extract the actual links as they appear (including protocol). For markdown links, extract the target URL only.
    - For numbers, prefer the numeric form if present in the answer; otherwise extract the phrase given.
    - If the answer lists multiple URLs, set "publication_url" to the most authoritative reference to the publication itself (prefer a Nature page), and put other URLs into "additional_urls".
    - If an item is not present, set it to null (or [] for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_valid_url(s: Optional[str]) -> bool:
    if not s:
        return False
    s = s.strip()
    return s.startswith("http://") or s.startswith("https://")


def collect_all_sources(ext: MarsDischargePaperExtraction) -> List[str]:
    urls: List[str] = []
    if ext.publication:
        if _is_valid_url(ext.publication.publication_url):
            urls.append(ext.publication.publication_url.strip())
        if ext.publication.additional_urls:
            for u in ext.publication.additional_urls:
                if _is_valid_url(u):
                    urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def build_publication_reference_and_access(
    evaluator: Evaluator,
    parent,
    ext: MarsDischargePaperExtraction
) -> None:
    # Parent node: PublicationReferenceAndAccess (critical)
    pub_root = evaluator.add_parallel(
        id="PublicationReferenceAndAccess",
        desc="Provide the complete publication reference and a publication reference URL.",
        parent=parent,
        critical=True
    )

    # Subnode: CompletePublicationReference (critical, split into presence + accuracy)
    ref_node = evaluator.add_parallel(
        id="CompletePublicationReference",
        desc="Complete publication reference provided, including journal, volume, pages, month/year, and full title.",
        parent=pub_root,
        critical=True
    )

    # Presence checks (each as a separate binary custom node)
    journal_present = evaluator.add_custom_node(
        result=bool(ext.publication and ext.publication.journal and ext.publication.journal.strip()),
        id="reference_journal_present",
        desc="Answer includes journal name for the publication reference.",
        parent=ref_node,
        critical=True
    )
    volume_present = evaluator.add_custom_node(
        result=bool(ext.publication and ext.publication.volume and ext.publication.volume.strip()),
        id="reference_volume_present",
        desc="Answer includes volume number for the publication reference.",
        parent=ref_node,
        critical=True
    )
    pages_present = evaluator.add_custom_node(
        result=bool(ext.publication and ext.publication.pages and ext.publication.pages.strip()),
        id="reference_pages_present",
        desc="Answer includes page range for the publication reference.",
        parent=ref_node,
        critical=True
    )
    month_present = evaluator.add_custom_node(
        result=bool(ext.publication and ext.publication.publication_month and ext.publication.publication_month.strip()),
        id="reference_month_present",
        desc="Answer includes publication month for the publication reference.",
        parent=ref_node,
        critical=True
    )
    year_present = evaluator.add_custom_node(
        result=bool(ext.publication and ext.publication.publication_year and ext.publication.publication_year.strip()),
        id="reference_year_present",
        desc="Answer includes publication year for the publication reference.",
        parent=ref_node,
        critical=True
    )
    title_present = evaluator.add_custom_node(
        result=bool(ext.publication and ext.publication.title and ext.publication.title.strip()),
        id="reference_title_present",
        desc="Answer includes full paper title for the publication reference.",
        parent=ref_node,
        critical=True
    )

    # Accuracy checks against sources
    ref_acc_node = evaluator.add_parallel(
        id="CompletePublicationReferenceAccuracy",
        desc="Publication reference components are correct per the cited source(s).",
        parent=ref_node,
        critical=True
    )

    all_sources = collect_all_sources(ext)

    journal_correct = evaluator.add_leaf(
        id="reference_journal_correct",
        desc="Journal is Nature (source-supported).",
        parent=ref_acc_node,
        critical=True
    )
    volume_correct = evaluator.add_leaf(
        id="reference_volume_correct",
        desc=f"Volume is {EXPECTED_VOLUME} (source-supported).",
        parent=ref_acc_node,
        critical=True
    )
    pages_correct = evaluator.add_leaf(
        id="reference_pages_correct",
        desc=f"Pages are {EXPECTED_PAGES} (source-supported).",
        parent=ref_acc_node,
        critical=True
    )
    pub_month_correct = evaluator.add_leaf(
        id="reference_pub_month_correct",
        desc=f"Publication month is {EXPECTED_PUB_MONTH} (source-supported).",
        parent=ref_acc_node,
        critical=True
    )
    pub_year_correct = evaluator.add_leaf(
        id="reference_pub_year_correct",
        desc=f"Publication year is {EXPECTED_PUB_YEAR} (source-supported).",
        parent=ref_acc_node,
        critical=True
    )
    title_correct = evaluator.add_leaf(
        id="reference_title_correct",
        desc=f"Paper title matches '{EXPECTED_TITLE}' (source-supported).",
        parent=ref_acc_node,
        critical=True
    )

    claims_and_sources = [
        (
            "This publication is in the journal Nature.",
            all_sources,
            journal_correct,
            "Verify the page explicitly indicates the journal is Nature."
        ),
        (
            f"The publication volume is {EXPECTED_VOLUME}.",
            all_sources,
            volume_correct,
            "Accept Arabic numerals; check the publication details/metadata."
        ),
        (
            f"The page range (pages) for the publication is {EXPECTED_PAGES}.",
            all_sources,
            pages_correct,
            "Allow minor punctuation variations: hyphen vs en dash, with or without spaces."
        ),
        (
            f"The publication date indicates the month is {EXPECTED_PUB_MONTH}.",
            all_sources,
            pub_month_correct,
            "If a specific day is shown (e.g., 05 November 2025), consider it consistent with the stated month."
        ),
        (
            f"The publication year is {EXPECTED_PUB_YEAR}.",
            all_sources,
            pub_year_correct,
            "If a specific day is shown, ensure the year matches 2025."
        ),
        (
            f"The paper's full title is '{EXPECTED_TITLE}'.",
            all_sources,
            title_correct,
            "Allow minor variations in punctuation/casing but ensure the semantic title matches exactly."
        )
    ]
    await evaluator.batch_verify(claims_and_sources)

    # Publication URL node (critical)
    url_node = evaluator.add_parallel(
        id="PublicationURL",
        desc="A reference URL to the publication is provided (valid) and points to the correct paper.",
        parent=pub_root,
        critical=True
    )

    url_provided = evaluator.add_custom_node(
        result=bool(ext.publication and _is_valid_url(ext.publication.publication_url)),
        id="publication_url_provided",
        desc="Publication reference URL is provided and validly formatted (starts with http/https).",
        parent=url_node,
        critical=True
    )

    url_points_to_paper = evaluator.add_leaf(
        id="publication_url_points_to_paper",
        desc="The provided publication URL corresponds to the specified Nature paper (title matches).",
        parent=url_node,
        critical=True
    )
    main_url = (ext.publication.publication_url if ext.publication else None)
    await evaluator.verify(
        claim=f"The page at this URL corresponds to the paper titled '{EXPECTED_TITLE}'.",
        node=url_points_to_paper,
        sources=main_url if _is_valid_url(main_url) else None,
        additional_instruction="If the URL is a Nature article page, the title must match. If not accessible or irrelevant, mark as not supported."
    )


async def build_authorship_and_institution(
    evaluator: Evaluator,
    parent,
    ext: MarsDischargePaperExtraction
) -> None:
    # Parent node: AuthorshipAndInstitution (critical)
    auth_root = evaluator.add_parallel(
        id="AuthorshipAndInstitution",
        desc="Provide lead author identity and primary affiliation, plus institution location.",
        parent=parent,
        critical=True
    )

    all_sources = collect_all_sources(ext)

    # Lead author name
    lead_author_node = evaluator.add_parallel(
        id="LeadAuthorName",
        desc="Lead (first) author is identified as Baptiste Chide.",
        parent=auth_root,
        critical=True
    )
    lead_author_present = evaluator.add_custom_node(
        result=bool(ext.author_institution and ext.author_institution.lead_author_name and ext.author_institution.lead_author_name.strip()),
        id="lead_author_present",
        desc="Answer includes the lead (first) author name.",
        parent=lead_author_node,
        critical=True
    )
    lead_author_correct = evaluator.add_leaf(
        id="lead_author_correct",
        desc=f"Lead (first) author is {EXPECTED_LEAD_AUTHOR} (source-supported).",
        parent=lead_author_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The first (lead) author of the paper is {EXPECTED_LEAD_AUTHOR}.",
        node=lead_author_correct,
        sources=all_sources,
        additional_instruction="Verify authors list; the first listed name should be recognized as the lead (first) author."
    )

    # Lead author's primary affiliation
    primary_aff_node = evaluator.add_parallel(
        id="LeadAuthorPrimaryAffiliation",
        desc=f"Lead author’s primary institutional affiliation is {EXPECTED_PRIMARY_AFFILIATION}.",
        parent=auth_root,
        critical=True
    )
    primary_aff_present = evaluator.add_custom_node(
        result=bool(ext.author_institution and ext.author_institution.lead_author_primary_affiliation and ext.author_institution.lead_author_primary_affiliation.strip()),
        id="primary_affiliation_present",
        desc="Answer includes the lead author's primary institutional affiliation.",
        parent=primary_aff_node,
        critical=True
    )
    primary_aff_correct = evaluator.add_leaf(
        id="primary_affiliation_correct",
        desc="Lead author's primary institutional affiliation is correctly stated (IRAP).",
        parent=primary_aff_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The lead author, {EXPECTED_LEAD_AUTHOR}, is affiliated with {EXPECTED_PRIMARY_AFFILIATION}.",
        node=primary_aff_correct,
        sources=all_sources,
        additional_instruction="Check the affiliations listed for the lead author; accept expanded forms including CNRS/UPS/CNES/Université de Toulouse as long as IRAP (Institut de Recherche en Astrophysique et Planétologie) is clearly indicated."
    )

    # Institution location (city & country)
    loc_node = evaluator.add_parallel(
        id="InstitutionCityCountry",
        desc=f"The institution location is provided as {EXPECTED_CITY}, {EXPECTED_COUNTRY} (city and country).",
        parent=auth_root,
        critical=True
    )

    city_present = evaluator.add_custom_node(
        result=bool(ext.author_institution and ext.author_institution.institution_city and ext.author_institution.institution_city.strip()),
        id="institution_city_present",
        desc="Answer includes the institution city.",
        parent=loc_node,
        critical=True
    )
    country_present = evaluator.add_custom_node(
        result=bool(ext.author_institution and ext.author_institution.institution_country and ext.author_institution.institution_country.strip()),
        id="institution_country_present",
        desc="Answer includes the institution country.",
        parent=loc_node,
        critical=True
    )

    city_correct = evaluator.add_leaf(
        id="institution_city_correct",
        desc=f"Institution city is {EXPECTED_CITY} (source-supported).",
        parent=loc_node,
        critical=True
    )
    country_correct = evaluator.add_leaf(
        id="institution_country_correct",
        desc=f"Institution country is {EXPECTED_COUNTRY} (source-supported).",
        parent=loc_node,
        critical=True
    )

    await evaluator.batch_verify([
        (
            f"The affiliation information indicates the city {EXPECTED_CITY}.",
            all_sources,
            city_correct,
            "Check the affiliation line(s) for the location. Allow presence anywhere in the affiliation details."
        ),
        (
            f"The affiliation information indicates the country {EXPECTED_COUNTRY}.",
            all_sources,
            country_correct,
            "Check the affiliation line(s) for the location. Accept 'France' mentioned within institutional address."
        )
    ])


async def build_study_quantitative_details(
    evaluator: Evaluator,
    parent,
    ext: MarsDischargePaperExtraction
) -> None:
    # Parent node: StudyQuantitativeDetails (critical)
    study_root = evaluator.add_parallel(
        id="StudyQuantitativeDetails",
        desc="Provide the requested quantitative details about recordings and detections.",
        parent=parent,
        critical=True
    )

    all_sources = collect_all_sources(ext)

    # Recording hours analyzed
    rec_hours_node = evaluator.add_parallel(
        id="RecordingHoursAnalyzed",
        desc=f"Total analyzed microphone recording duration is stated as {EXPECTED_RECORDING_HOURS} hours.",
        parent=study_root,
        critical=True
    )
    rec_hours_present = evaluator.add_custom_node(
        result=bool(ext.study and ext.study.recording_hours_analyzed and ext.study.recording_hours_analyzed.strip()),
        id="recording_hours_present",
        desc="Answer includes the total analyzed microphone recording hours.",
        parent=rec_hours_node,
        critical=True
    )
    rec_hours_correct = evaluator.add_leaf(
        id="recording_hours_correct",
        desc=f"Recording hours analyzed equal {EXPECTED_RECORDING_HOURS} (source-supported).",
        parent=rec_hours_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The study analyzed {EXPECTED_RECORDING_HOURS} hours of microphone recordings.",
        node=rec_hours_correct,
        sources=all_sources,
        additional_instruction="Accept forms like '28 h' or '28 hours' as equivalent."
    )

    # Recording timespan (Martian years and Earth days)
    timespan_node = evaluator.add_parallel(
        id="RecordingTimespan",
        desc=f"Recording timespan is stated as two Martian years and also given as {EXPECTED_TIMESPAN_EARTH_DAYS} Earth days.",
        parent=study_root,
        critical=True
    )
    my_present = evaluator.add_custom_node(
        result=bool(ext.study and ext.study.timespan_martian_years and ext.study.timespan_martian_years.strip()),
        id="timespan_martian_years_present",
        desc="Answer includes the recording timespan expressed in Martian years.",
        parent=timespan_node,
        critical=True
    )
    ed_present = evaluator.add_custom_node(
        result=bool(ext.study and ext.study.timespan_earth_days and ext.study.timespan_earth_days.strip()),
        id="timespan_earth_days_present",
        desc="Answer includes the recording timespan expressed in Earth days.",
        parent=timespan_node,
        critical=True
    )
    my_correct = evaluator.add_leaf(
        id="timespan_martian_years_correct",
        desc="Timespan equals two (2) Martian years (source-supported).",
        parent=timespan_node,
        critical=True
    )
    ed_correct = evaluator.add_leaf(
        id="timespan_earth_days_correct",
        desc=f"Timespan equals {EXPECTED_TIMESPAN_EARTH_DAYS} Earth days (source-supported).",
        parent=timespan_node,
        critical=True
    )
    await evaluator.batch_verify([
        (
            "The recordings span two Martian years (i.e., approximately 2 Mars years).",
            all_sources,
            my_correct,
            "Accept 'two Martian years'/'2 Martian years'/'two Mars years' as equivalent."
        ),
        (
            f"The recordings span {EXPECTED_TIMESPAN_EARTH_DAYS} Earth days.",
            all_sources,
            ed_correct,
            "Accept formatting variations such as '1,374' vs '1374'."
        )
    ])

    # Discharge event count
    events_node = evaluator.add_parallel(
        id="DischargeEventCount",
        desc=f"Total number of detected electrical discharge events is stated as {EXPECTED_EVENT_COUNT}.",
        parent=study_root,
        critical=True
    )
    events_present = evaluator.add_custom_node(
        result=bool(ext.study and ext.study.discharge_event_count and ext.study.discharge_event_count.strip()),
        id="discharge_event_count_present",
        desc="Answer includes the total number of detected electrical discharge events.",
        parent=events_node,
        critical=True
    )
    events_correct = evaluator.add_leaf(
        id="discharge_event_count_correct",
        desc=f"Detected electrical discharge events total {EXPECTED_EVENT_COUNT} (source-supported).",
        parent=events_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The total number of detected electrical discharge events is {EXPECTED_EVENT_COUNT}.",
        node=events_correct,
        sources=all_sources,
        additional_instruction="Accept phrasing like '55 events', '55 electrical discharges', or similar equivalent wording."
    )


async def build_paper_scope_and_method_constraints(
    evaluator: Evaluator,
    parent,
    ext: MarsDischargePaperExtraction
) -> None:
    # Parent node: PaperScopeAndMethodConstraints (critical)
    scope_root = evaluator.add_parallel(
        id="PaperScopeAndMethodConstraints",
        desc="Paper content/method constraints (as specified) are satisfied.",
        parent=parent,
        critical=True
    )

    all_sources = collect_all_sources(ext)

    # PhenomenonConstraint
    phenomenon_node = evaluator.add_leaf(
        id="PhenomenonConstraint",
        desc="The paper’s findings concern atmospheric electrical phenomena on Mars, specifically triboelectric discharges.",
        parent=scope_root,
        critical=True
    )
    await evaluator.verify(
        claim="The paper reports triboelectric electrical discharges occurring in the Martian atmosphere.",
        node=phenomenon_node,
        sources=all_sources,
        additional_instruction="This can be supported by the title and/or abstract explicitly mentioning 'triboelectric discharges' on Mars."
    )

    # MethodConstraint
    method_node = evaluator.add_leaf(
        id="MethodConstraint",
        desc="Detection uses acoustic measurements from NASA Perseverance rover’s SuperCam microphone (in situ detection).",
        parent=scope_root,
        critical=True
    )
    await evaluator.verify(
        claim="The detection uses acoustic measurements recorded by the Perseverance rover's SuperCam microphone, i.e., in situ on Mars.",
        node=method_node,
        sources=all_sources,
        additional_instruction="Look for mention of SuperCam microphone and acoustic detection methodology."
    )

    # EventAssociationConstraint
    assoc_node = evaluator.add_leaf(
        id="EventAssociationConstraint",
        desc="Detected events are associated with dust devils and dust storm convective fronts.",
        parent=scope_root,
        critical=True
    )
    await evaluator.verify(
        claim="The detected discharges are associated with dust devils and with convective fronts of dust storms.",
        node=assoc_node,
        sources=all_sources,
        additional_instruction="Support may appear in results/conclusions indicating associations with dust devils and dust storm convective fronts."
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
    # Initialize evaluator and root
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

    # Extract structured info from the answer
    extracted: MarsDischargePaperExtraction = await evaluator.extract(
        prompt=prompt_extract_paper_info(),
        template_class=MarsDischargePaperExtraction,
        extraction_name="mars_discharge_paper_extraction"
    )

    # Add ground truth (for reference in the final summary)
    evaluator.add_ground_truth({
        "expected": {
            "title": EXPECTED_TITLE,
            "journal": EXPECTED_JOURNAL,
            "volume": EXPECTED_VOLUME,
            "pages": EXPECTED_PAGES,
            "publication_month": EXPECTED_PUB_MONTH,
            "publication_year": EXPECTED_PUB_YEAR,
            "lead_author": EXPECTED_LEAD_AUTHOR,
            "primary_affiliation": EXPECTED_PRIMARY_AFFILIATION,
            "institution_city": EXPECTED_CITY,
            "institution_country": EXPECTED_COUNTRY,
            "recording_hours": EXPECTED_RECORDING_HOURS,
            "timespan_martian_years": EXPECTED_TIMESPAN_MARTIAN_YEARS,
            "timespan_earth_days": EXPECTED_TIMESPAN_EARTH_DAYS,
            "discharge_event_count": EXPECTED_EVENT_COUNT,
            "phenomenon": "triboelectric discharges",
            "method": "acoustic measurements from Perseverance SuperCam microphone",
            "event_association": "dust devils and dust storm convective fronts"
        }
    }, gt_type="ground_truth")

    # Build the top-level critical node for the task (all children under this must be critical)
    task_root = evaluator.add_parallel(
        id="ResearchIdentificationTask",
        desc="Identify the specified Nature (Nov 2025) Mars electrical-discharge paper and provide all requested bibliographic, author/institution, location, and quantitative study details, while satisfying all stated constraints.",
        parent=root,
        critical=True
    )

    # Build subtrees
    await build_publication_reference_and_access(evaluator, task_root, extracted)
    await build_authorship_and_institution(evaluator, task_root, extracted)
    await build_study_quantitative_details(evaluator, task_root, extracted)
    await build_paper_scope_and_method_constraints(evaluator, task_root, extracted)

    # Return standardized summary
    return evaluator.get_summary()