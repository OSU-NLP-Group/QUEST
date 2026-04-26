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
TASK_ID = "three_natsci_publications_2024_2025"
TASK_DESCRIPTION = """Identify three peer-reviewed research publications from 2024 or 2025 that report major discoveries in natural sciences:

1. Publication 1: A study reporting the discovery of large prehistoric shark fossils from Australia, including specific vertebrae measurements and estimated body size

2. Publication 2: A study reporting potential biosignatures or signs of ancient microbial life from a Mars rover sample collected in 2024 in Jezero Crater

3. Publication 3: Any additional significant peer-reviewed study from 2024 or 2025 in paleontology, astrobiology, or planetary science that reports a major discovery

For each publication, provide:
- Complete bibliographic information: Journal name, publication date (month/year), DOI or article link, full article title
- Lead author details: Full name, primary institutional affiliation, department or research role
- Specific research findings: Main discovery description, quantitative measurements (sizes, ages, dates), taxonomic or feature classifications
- Location/context information: Geographic location (for fossils) or planetary location details (for space missions), including specific formations or sites
- Additional metadata: Number of authors, journal publisher, open access status (where applicable)
- Reference URL: A working link to the published paper or official press release

All information must be verifiable through publicly accessible sources.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Publication(BaseModel):
    # Generic identifiers / categorization
    pub_id: Optional[str] = None  # "1", "2", "3" or equivalent label in the answer
    domain: Optional[str] = None  # (for Pub3) domain/scope (e.g., paleontology, astrobiology, planetary science)

    # Required bibliographic information
    title: Optional[str] = None
    journal: Optional[str] = None
    publication_date: Optional[str] = None  # month/year or full date string
    doi: Optional[str] = None
    article_link: Optional[str] = None

    # Reference URL (paper page or press release)
    reference_url: Optional[str] = None

    # Lead author details
    lead_author_name: Optional[str] = None
    lead_author_affiliation: Optional[str] = None
    lead_author_department_or_role: Optional[str] = None

    # Findings details
    findings_main: Optional[str] = None
    quantitative_measurements: List[str] = Field(default_factory=list)
    classifications: List[str] = Field(default_factory=list)

    # Location/context
    location_context: Optional[str] = None

    # Additional metadata
    discovery_collection_date: Optional[str] = None  # date of discovery/collection if applicable
    number_of_authors: Optional[str] = None
    journal_publisher: Optional[str] = None
    open_access_status: Optional[str] = None
    journal_metric: Optional[str] = None

    # Extra URLs
    additional_urls: List[str] = Field(default_factory=list)

    # Pub1-specific fields
    australian_shark_fossil: Optional[bool] = None
    vertebrae_measurements: List[str] = Field(default_factory=list)
    estimated_body_size: Optional[str] = None

    # Pub2-specific fields
    jezero_crater: Optional[bool] = None
    sample_collected_in_2024: Optional[bool] = None
    sample_collection_month_year: Optional[str] = None
    biosignature_claim: Optional[str] = None
    specific_site_or_formation: Optional[str] = None

    # Pub3-specific fields
    major_discovery: Optional[str] = None
    quantitative_finding: Optional[str] = None


class PublicationsExtraction(BaseModel):
    publications: List[Publication] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_publications() -> str:
    return """
    Extract exactly the first three relevant publications presented in the answer that match the three topic requirements
    (Pub1: Australian large prehistoric shark fossils; Pub2: Jezero Crater biosignature-related sample collected in 2024;
     Pub3: any significant natural-science discovery in paleontology/astrobiology/planetary science from 2024–2025).

    For each publication, return a JSON object with the following fields when available (set to null if not present in the answer):

    Generic fields:
    - pub_id: The publication label used by the answer (e.g., "1", "2", "3"), if any.
    - domain: Field/domain, especially for Publication 3 (e.g., "paleontology", "astrobiology", "planetary science").
    - title: Full article title.
    - journal: Journal name.
    - publication_date: Publication date as month/year or full date string.
    - doi: DOI string if provided.
    - article_link: Direct article page URL (if provided).
    - reference_url: Public link to the published paper page or official press release.
    - lead_author_name: Full name of the lead/corresponding author.
    - lead_author_affiliation: Primary institutional affiliation of the lead/corresponding author.
    - lead_author_department_or_role: Department or research role of the lead/corresponding author.
    - findings_main: Main discovery description.
    - quantitative_measurements: Array of specific quantitative measurements or dates (include units where applicable).
    - classifications: Array of taxonomic/feature classifications reported.
    - location_context: Geographic (for fossils) or planetary/site context (for mission samples).
    - discovery_collection_date: Date of discovery/collection, if applicable/available.
    - number_of_authors: Number of authors if stated or inferable in the answer (return as string).
    - journal_publisher: Journal publisher name if provided.
    - open_access_status: Open access status if provided (e.g., "Open Access", "Closed", "Hybrid").
    - journal_metric: Any metric/indexing/indicator (e.g., "indexed in Scopus", "Impact Factor 7.3") when applicable.
    - additional_urls: Array of any additional URLs mentioned that relate to this publication.

    Pub1-specific:
    - australian_shark_fossil: true/false if the article reports discovery of a large prehistoric shark fossil from Australia.
    - vertebrae_measurements: Array of vertebra(e) measurements with units (e.g., "vertebra diameter 230 mm").
    - estimated_body_size: Estimated body size (e.g., length/mass) with units, if provided.

    Pub2-specific:
    - jezero_crater: true/false if findings are explicitly tied to Jezero Crater.
    - sample_collected_in_2024: true/false if it involves a rover sample collected in 2024.
    - sample_collection_month_year: Month/year of the 2024 sample collection, if provided (e.g., "July 2024").
    - biosignature_claim: Text summarizing potential biosignatures or signs of ancient microbial life tied to the 2024 sample.
    - specific_site_or_formation: Specific site/formation/feature in Jezero associated with the sample (e.g., "deltaic deposits", "carbonate-bearing outcrops").

    Pub3-specific:
    - major_discovery: Short text describing the major discovery claimed.
    - quantitative_finding: A specific quantitative measurement/date/numerical finding with units where applicable.

    Return the final JSON object with a single field:
    - publications: an array of up to three publication objects as defined above (if more are present, include only the first three; if fewer, include what is available).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_doi_to_url(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    doi = doi.strip()
    if not doi:
        return None
    # If already a URL to doi.org, return as is
    if doi.lower().startswith("http://doi.org/") or doi.lower().startswith("https://doi.org/"):
        return doi
    # If already a URL to dx.doi.org or similar, return as is
    if "doi.org" in doi.lower():
        return doi
    # Otherwise, convert to canonical doi.org URL
    return f"https://doi.org/{doi}"


def build_sources(pub: Publication) -> List[str]:
    urls: List[str] = []
    if pub.article_link:
        urls.append(pub.article_link.strip())
    if pub.reference_url:
        urls.append(pub.reference_url.strip())
    doi_url = _normalize_doi_to_url(pub.doi)
    if doi_url:
        urls.append(doi_url)
    for u in pub.additional_urls:
        if isinstance(u, str) and u.strip():
            urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _safe_str(s: Optional[str]) -> str:
    return (s or "").strip()


def _has_nonempty(s: Optional[str]) -> bool:
    return bool((s or "").strip())


def _list_has_items(lst: Optional[List[str]]) -> bool:
    return bool(lst) and any((item or "").strip() for item in lst or [])


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_publication_eligibility(
    evaluator: Evaluator,
    parent_node,
    pub: Publication,
    prefix: str,
) -> None:
    """
    Add and verify PubX_Eligibility checks: peer-reviewed, date in range+provided, original research.
    """
    elig_node = evaluator.add_parallel(
        id=f"{prefix}_Eligibility",
        desc=f"{prefix.replace('_', ' ')} meets venue/time/original-research constraints.",
        parent=parent_node,
        critical=True
    )

    # Peer-reviewed journal
    leaf_peer = evaluator.add_leaf(
        id=f"{prefix}_Peer_Reviewed_Journal",
        desc="Published in a peer-reviewed scientific journal.",
        parent=elig_node,
        critical=True,
    )
    claim_peer = f"The article '{_safe_str(pub.title)}' in journal '{_safe_str(pub.journal)}' is a peer-reviewed scientific journal article."
    await evaluator.verify(
        claim=claim_peer,
        node=leaf_peer,
        sources=build_sources(pub),
        additional_instruction="Use the article page or journal page to check indicators like 'Research Article', editorial policies, or indexing that imply peer review."
    )

    # Publication date in 2024 or 2025 and provided (on/before Nov 30, 2025)
    leaf_date = evaluator.add_leaf(
        id=f"{prefix}_Date_In_Range_And_Provided",
        desc="Publication date (month/year at minimum) is provided and is in 2024 or 2025, and on/before Nov 30, 2025 (published/announced).",
        parent=elig_node,
        critical=True,
    )
    claim_date = (
        f"The publication date for '{_safe_str(pub.title)}' is in 2024 or 2025 "
        f"and not later than November 30, 2025. The answer lists the date as '{_safe_str(pub.publication_date)}'."
    )
    await evaluator.verify(
        claim=claim_date,
        node=leaf_date,
        sources=build_sources(pub),
        additional_instruction="Verify the publication or early-online date shown on the paper/DOI page; accept any 2024/2025 date that is not after 2025-11-30."
    )

    # Original research (not review/editorial/news)
    leaf_orig = evaluator.add_leaf(
        id=f"{prefix}_Original_Research",
        desc="Is original research (not solely review/editorial/news).",
        parent=elig_node,
        critical=True,
    )
    claim_orig = (
        f"The article '{_safe_str(pub.title)}' presents original research (e.g., with methods/results), and it is not solely a review, editorial, or news piece."
    )
    await evaluator.verify(
        claim=claim_orig,
        node=leaf_orig,
        sources=build_sources(pub),
        additional_instruction="Check for sections like Methods/Results/Discussion or categorization such as 'Article'/'Research Article' rather than 'Review'/'Editorial'/'News'."
    )


async def verify_pub1_topic(
    evaluator: Evaluator,
    parent_node,
    pub: Publication
) -> None:
    """
    Pub1 topic checks: Australian shark fossil, vertebrae measurements, estimated body size.
    """
    topic_node = evaluator.add_parallel(
        id="Pub1_Topic",
        desc="Publication 1 satisfies the specified fossil discovery/topic requirements.",
        parent=parent_node,
        critical=True
    )

    # Australian large prehistoric shark fossil
    leaf_shark = evaluator.add_leaf(
        id="Pub1_Australian_Shark_Fossil",
        desc="Reports discovery of a large prehistoric shark fossil from Australia.",
        parent=topic_node,
        critical=True,
    )
    claim_shark = "This study reports the discovery of a large prehistoric shark fossil from Australia."
    await evaluator.verify(
        claim=claim_shark,
        node=leaf_shark,
        sources=build_sources(pub),
        additional_instruction="Confirm the fossil is a shark (prehistoric) and the discovery location is in Australia."
    )

    # Vertebrae measurements with units
    leaf_vertebra = evaluator.add_leaf(
        id="Pub1_Vertebrae_Measurements",
        desc="Includes specific vertebrae measurement(s) with units.",
        parent=topic_node,
        critical=True,
    )
    example_meas = pub.vertebrae_measurements[0] if _list_has_items(pub.vertebrae_measurements) else "vertebra measurement(s) with units"
    claim_vertebra = f"The paper includes specific vertebra(e) measurements with units, for example: {example_meas}."
    await evaluator.verify(
        claim=claim_vertebra,
        node=leaf_vertebra,
        sources=build_sources(pub),
        additional_instruction="Look for measurements (e.g., diameter/length) with units (mm, cm, etc.) associated with shark vertebrae."
    )

    # Estimated body size with units
    leaf_body = evaluator.add_leaf(
        id="Pub1_Estimated_Body_Size",
        desc="Includes estimated body size (e.g., length and/or mass) with units.",
        parent=topic_node,
        critical=True,
    )
    body_example = _safe_str(pub.estimated_body_size) or "an estimated body size (length/mass) with units"
    claim_body = f"The paper includes {body_example}."
    await evaluator.verify(
        claim=claim_body,
        node=leaf_body,
        sources=build_sources(pub),
        additional_instruction="Confirm presence of a quantitative estimate of body size (length/mass) with units."
    )


async def add_pub1_required_fields(
    evaluator: Evaluator,
    parent_node,
    pub: Publication
) -> None:
    """
    Pub1 required fields: Split into essentials (critical) and optional metadata (non-critical).
    """
    req_node = evaluator.add_parallel(
        id="Pub1_Required_Fields",
        desc="Required bibliographic/author/context/metadata fields and URLs are provided for Publication 1.",
        parent=parent_node,
        critical=False
    )

    essentials = evaluator.add_parallel(
        id="Pub1_Required_Essentials",
        desc="Essential required fields provided for Publication 1.",
        parent=req_node,
        critical=True
    )
    # Essential existence checks (custom nodes)
    evaluator.add_custom_node(_has_nonempty(pub.title), "Pub1_Full_Title", "Full article title is provided.", parent=essentials, critical=True)
    evaluator.add_custom_node(_has_nonempty(pub.journal), "Pub1_Journal_Name", "Journal name is provided.", parent=essentials, critical=True)
    evaluator.add_custom_node(_has_nonempty(pub.doi) or _has_nonempty(pub.article_link), "Pub1_DOI_or_Article_Link", "DOI or direct article link is provided.", parent=essentials, critical=True)
    evaluator.add_custom_node(_has_nonempty(pub.reference_url), "Pub1_Reference_URL_Public", "A publicly accessible working URL to the published paper or official press release is provided.", parent=essentials, critical=True)
    evaluator.add_custom_node(_has_nonempty(pub.lead_author_name), "Pub1_Lead_Author_Name", "Lead/corresponding author full name is provided.", parent=essentials, critical=True)
    evaluator.add_custom_node(_has_nonempty(pub.lead_author_affiliation), "Pub1_Lead_Author_Affiliation", "Lead/corresponding author primary institutional affiliation is provided and identifiable.", parent=essentials, critical=True)
    evaluator.add_custom_node(_has_nonempty(pub.lead_author_department_or_role), "Pub1_Lead_Author_Department_or_Role", "Lead/corresponding author department or research role is provided.", parent=essentials, critical=True)
    evaluator.add_custom_node(_list_has_items(pub.quantitative_measurements), "Pub1_Quantitative_Findings", "Includes quantitative measurements/findings beyond vertebrae/body size where applicable (with units when applicable).", parent=essentials, critical=True)
    evaluator.add_custom_node(_list_has_items(pub.classifications), "Pub1_Taxonomic_or_Feature_Classification", "At least one taxonomic/feature classification reported in the paper is provided.", parent=essentials, critical=True)
    evaluator.add_custom_node(_has_nonempty(pub.location_context), "Pub1_Location_Context", "Specific geographic/geologic site context in Australia is provided.", parent=essentials, critical=True)
    evaluator.add_custom_node(_has_nonempty(pub.discovery_collection_date), "Pub1_Discovery_Date_If_Applicable", "Date of discovery/collection is provided if applicable/available from sources.", parent=essentials, critical=True)

    optional = evaluator.add_parallel(
        id="Pub1_Optional_Metadata",
        desc="Optional metadata provided for Publication 1.",
        parent=req_node,
        critical=False
    )
    evaluator.add_custom_node(_has_nonempty(pub.number_of_authors), "Pub1_Number_of_Authors", "Number of authors is provided.", parent=optional, critical=False)
    evaluator.add_custom_node(_has_nonempty(pub.journal_publisher), "Pub1_Journal_Publisher", "Journal publisher is provided.", parent=optional, critical=False)
    evaluator.add_custom_node(_has_nonempty(pub.open_access_status), "Pub1_Open_Access_Status", "Open access status is provided when determinable.", parent=optional, critical=False)


async def verify_pub1_verifiability(
    evaluator: Evaluator,
    parent_node,
    pub: Publication
) -> None:
    """
    Pub1 verifiability leaf: Verify key facts against provided public sources.
    """
    leaf_verif = evaluator.add_leaf(
        id="Pub1_Verifiability",
        desc="Key stated metadata and findings for Publication 1 are verifiable from the provided public sources.",
        parent=parent_node,
        critical=True,
    )
    vertebra_summary = ", ".join(pub.vertebrae_measurements) if _list_has_items(pub.vertebrae_measurements) else "vertebra(e) measurements with units"
    claim_verif = (
        f"Verify that the following facts are supported by the provided URLs: "
        f"title '{_safe_str(pub.title)}', journal '{_safe_str(pub.journal)}', Australian context '{_safe_str(pub.location_context)}', "
        f"vertebrae measurements '{vertebra_summary}', estimated body size '{_safe_str(pub.estimated_body_size)}', "
        f"lead author '{_safe_str(pub.lead_author_name)}' with affiliation '{_safe_str(pub.lead_author_affiliation)}' and role '{_safe_str(pub.lead_author_department_or_role)}'."
    )
    await evaluator.verify(
        claim=claim_verif,
        node=leaf_verif,
        sources=build_sources(pub),
        additional_instruction="The URLs should explicitly support these facts; closely check article pages or official press releases."
    )


async def verify_pub2_topic(
    evaluator: Evaluator,
    parent_node,
    pub: Publication
) -> None:
    """
    Pub2 topic checks: Jezero Crater link, 2024 sample with month/year, potential biosignatures tied to 2024 sample.
    """
    topic_node = evaluator.add_parallel(
        id="Pub2_Topic",
        desc="Publication 2 satisfies the specified mission/sample/topic requirements.",
        parent=parent_node,
        critical=True
    )

    leaf_jezero = evaluator.add_leaf(
        id="Pub2_Jezero_Crater",
        desc="Findings are explicitly tied to Jezero Crater.",
        parent=topic_node,
        critical=True,
    )
    claim_jezero = "The study's findings are explicitly tied to Jezero Crater."
    await evaluator.verify(
        claim=claim_jezero,
        node=leaf_jezero,
        sources=build_sources(pub),
        additional_instruction="Look for explicit mentions of Jezero Crater and related site context."
    )

    leaf_sample = evaluator.add_leaf(
        id="Pub2_Sample_Collected_In_2024_With_Date",
        desc="Involves a rover sample collected in 2024, and the collection month/year is provided.",
        parent=topic_node,
        critical=True,
    )
    claim_sample = (
        f"The study involves a rover sample collected in 2024, and the collection month/year is '{_safe_str(pub.sample_collection_month_year)}'."
    )
    await evaluator.verify(
        claim=claim_sample,
        node=leaf_sample,
        sources=build_sources(pub),
        additional_instruction="Confirm that the page(s) specify a 2024 collection date and include the month/year for the sample."
    )

    leaf_bio = evaluator.add_leaf(
        id="Pub2_Potential_Biosignatures",
        desc="Reports potential biosignatures or signs of ancient microbial life tied to the 2024 sample.",
        parent=topic_node,
        critical=True,
    )
    claim_bio = "The paper reports potential biosignatures or signs of ancient microbial life tied to the 2024 sample."
    await evaluator.verify(
        claim=claim_bio,
        node=leaf_bio,
        sources=build_sources(pub),
        additional_instruction="Check for language indicating possible biosignatures, microfossil-like textures, organics, isotopic patterns, etc., explicitly connected to the 2024 sample."
    )


async def add_pub2_required_fields(
    evaluator: Evaluator,
    parent_node,
    pub: Publication
) -> None:
    """
    Pub2 required fields: Essentials (critical) and optional metadata (non-critical).
    """
    req_node = evaluator.add_parallel(
        id="Pub2_Required_Fields",
        desc="Required bibliographic/author/context/metadata fields and URLs are provided for Publication 2.",
        parent=parent_node,
        critical=False
    )

    essentials = evaluator.add_parallel(
        id="Pub2_Required_Essentials",
        desc="Essential required fields provided for Publication 2.",
        parent=req_node,
        critical=True
    )
    evaluator.add_custom_node(_has_nonempty(pub.title), "Pub2_Full_Title", "Full article title is provided.", parent=essentials, critical=True)
    evaluator.add_custom_node(_has_nonempty(pub.journal), "Pub2_Journal_Name", "Journal name is provided.", parent=essentials, critical=True)
    evaluator.add_custom_node(_has_nonempty(pub.doi) or _has_nonempty(pub.article_link), "Pub2_DOI_or_Article_Link", "DOI or direct article link is provided.", parent=essentials, critical=True)
    evaluator.add_custom_node(_has_nonempty(pub.reference_url), "Pub2_Reference_URL_Public", "A publicly accessible working URL to the published paper or official press release is provided.", parent=essentials, critical=True)
    evaluator.add_custom_node(_has_nonempty(pub.lead_author_name), "Pub2_Lead_Author_Name", "Lead/corresponding author full name is provided.", parent=essentials, critical=True)
    evaluator.add_custom_node(_has_nonempty(pub.lead_author_affiliation), "Pub2_Lead_Author_Affiliation", "Lead/corresponding author primary institutional affiliation is provided and identifiable.", parent=essentials, critical=True)
    evaluator.add_custom_node(_has_nonempty(pub.lead_author_department_or_role), "Pub2_Lead_Author_Department_or_Role", "Lead/corresponding author department or research role is provided.", parent=essentials, critical=True)
    evaluator.add_custom_node(_has_nonempty(pub.biosignature_claim), "Pub2_Key_Indicators_Described", "Main discovery description includes specific features/indicators supporting the biosignature claim.", parent=essentials, critical=True)
    evaluator.add_custom_node(_has_nonempty(pub.specific_site_or_formation), "Pub2_Specific_Site_or_Formation", "Specific Martian site/formation/feature within Jezero Crater tied to the sample is provided.", parent=essentials, critical=True)

    optional = evaluator.add_parallel(
        id="Pub2_Optional_Metadata",
        desc="Optional metadata provided for Publication 2.",
        parent=req_node,
        critical=False
    )
    evaluator.add_custom_node(_has_nonempty(pub.number_of_authors), "Pub2_Number_of_Authors", "Number of authors is provided.", parent=optional, critical=False)
    evaluator.add_custom_node(_has_nonempty(pub.journal_publisher), "Pub2_Journal_Publisher", "Journal publisher is provided.", parent=optional, critical=False)
    evaluator.add_custom_node(_has_nonempty(pub.open_access_status), "Pub2_Open_Access_Status", "Open access status is provided when determinable.", parent=optional, critical=False)


async def verify_pub2_verifiability(
    evaluator: Evaluator,
    parent_node,
    pub: Publication
) -> None:
    """
    Pub2 verifiability leaf: Verify key facts against provided public sources.
    """
    leaf_verif = evaluator.add_leaf(
        id="Pub2_Verifiability",
        desc="Key stated metadata and findings for Publication 2 are verifiable from the provided public sources.",
        parent=parent_node,
        critical=True,
    )
    claim_verif = (
        f"Verify that the following facts are supported by the provided URLs: "
        f"title '{_safe_str(pub.title)}', journal '{_safe_str(pub.journal)}', Jezero Crater context, "
        f"sample collection month/year '{_safe_str(pub.sample_collection_month_year)}' in 2024, "
        f"and the described biosignature indicators ('{_safe_str(pub.biosignature_claim)}') tied to that sample."
    )
    await evaluator.verify(
        claim=claim_verif,
        node=leaf_verif,
        sources=build_sources(pub),
        additional_instruction="Confirm that the article/press-release pages explicitly support these details."
    )


async def verify_pub3_scope(
    evaluator: Evaluator,
    parent_node,
    pub: Publication
) -> None:
    """
    Pub3 scope and significance checks: domain in scope, major discovery described, quantitative finding included.
    """
    scope_node = evaluator.add_parallel(
        id="Pub3_Scope_And_Significance",
        desc="Publication 3 is in-scope and reports a major discovery with quantitative support.",
        parent=parent_node,
        critical=True
    )

    leaf_domain = evaluator.add_leaf(
        id="Pub3_Domain_In_Scope",
        desc="Field is paleontology, astrobiology, planetary science, or clearly related natural-science field.",
        parent=scope_node,
        critical=True,
    )
    claim_domain = (
        f"The article '{_safe_str(pub.title)}' is within paleontology, astrobiology, planetary science, or a closely related natural-science field. "
        f"The answer indicates domain '{_safe_str(pub.domain)}'."
    )
    await evaluator.verify(
        claim=claim_domain,
        node=leaf_domain,
        sources=build_sources(pub),
        additional_instruction="Check the article context and journal scope to confirm domain."
    )

    leaf_major = evaluator.add_leaf(
        id="Pub3_Major_Discovery_Described",
        desc="Reports a major/significant discovery, described in the answer.",
        parent=scope_node,
        critical=True,
    )
    claim_major = f"The article reports a major/significant discovery: '{_safe_str(pub.major_discovery)}'."
    await evaluator.verify(
        claim=claim_major,
        node=leaf_major,
        sources=build_sources(pub),
        additional_instruction="Verify that the paper clearly presents a non-trivial new finding or discovery."
    )

    leaf_quant = evaluator.add_leaf(
        id="Pub3_Quantitative_Finding",
        desc="Includes at least one specific quantitative measurement/date/numerical finding with units where applicable.",
        parent=scope_node,
        critical=True,
    )
    q_example = _safe_str(pub.quantitative_finding) or (
        pub.quantitative_measurements[0] if _list_has_items(pub.quantitative_measurements) else "a specific quantitative datum"
    )
    claim_quant = f"The article includes a specific quantitative measurement/date/numerical finding (e.g., '{q_example}')."
    await evaluator.verify(
        claim=claim_quant,
        node=leaf_quant,
        sources=build_sources(pub),
        additional_instruction="Look for numbers, dates or measurements with units where applicable."
    )


async def add_pub3_required_fields(
    evaluator: Evaluator,
    parent_node,
    pub: Publication
) -> None:
    """
    Pub3 required fields: Essentials (critical) and optional metadata (non-critical).
    """
    req_node = evaluator.add_parallel(
        id="Pub3_Required_Fields",
        desc="Required bibliographic/author/context/metadata fields and URLs are provided for Publication 3.",
        parent=parent_node,
        critical=False
    )

    essentials = evaluator.add_parallel(
        id="Pub3_Required_Essentials",
        desc="Essential required fields provided for Publication 3.",
        parent=req_node,
        critical=True
    )
    evaluator.add_custom_node(_has_nonempty(pub.title), "Pub3_Full_Title", "Full article title is provided.", parent=essentials, critical=True)
    evaluator.add_custom_node(_has_nonempty(pub.journal), "Pub3_Journal_Name", "Journal name is provided.", parent=essentials, critical=True)
    evaluator.add_custom_node(_has_nonempty(pub.doi) or _has_nonempty(pub.article_link), "Pub3_DOI_or_Article_Link", "DOI or direct article link is provided.", parent=essentials, critical=True)
    evaluator.add_custom_node(_has_nonempty(pub.reference_url), "Pub3_Reference_URL_Public", "A publicly accessible working URL to the published paper or official press release is provided.", parent=essentials, critical=True)
    evaluator.add_custom_node(_has_nonempty(pub.lead_author_name), "Pub3_Lead_Author_Name", "Lead/corresponding author full name is provided.", parent=essentials, critical=True)
    evaluator.add_custom_node(_has_nonempty(pub.lead_author_affiliation), "Pub3_Lead_Author_Affiliation", "Lead/corresponding author primary institutional affiliation is provided and identifiable.", parent=essentials, critical=True)
    evaluator.add_custom_node(_has_nonempty(pub.lead_author_department_or_role), "Pub3_Lead_Author_Department_or_Role", "Lead/corresponding author department or research role is provided.", parent=essentials, critical=True)
    evaluator.add_custom_node(_has_nonempty(pub.location_context), "Pub3_Location_or_Context", "Relevant geographic/planetary/sample-site context is provided.", parent=essentials, critical=True)
    evaluator.add_custom_node(_has_nonempty(pub.discovery_collection_date), "Pub3_Discovery_or_Collection_Date_If_Applicable", "Date of discovery/collection is provided where applicable/available from sources.", parent=essentials, critical=True)

    optional = evaluator.add_parallel(
        id="Pub3_Optional_Metadata",
        desc="Optional metadata provided for Publication 3.",
        parent=req_node,
        critical=False
    )
    evaluator.add_custom_node(_has_nonempty(pub.number_of_authors), "Pub3_Number_of_Authors", "Number of authors is provided.", parent=optional, critical=False)
    evaluator.add_custom_node(_has_nonempty(pub.journal_publisher), "Pub3_Journal_Publisher", "Journal publisher is provided.", parent=optional, critical=False)
    evaluator.add_custom_node(_has_nonempty(pub.open_access_status), "Pub3_Open_Access_Status", "Open access status is provided when determinable.", parent=optional, critical=False)


async def verify_pub3_verifiability(
    evaluator: Evaluator,
    parent_node,
    pub: Publication
) -> None:
    """
    Pub3 verifiability leaf: Verify key facts against provided public sources.
    """
    leaf_verif = evaluator.add_leaf(
        id="Pub3_Verifiability",
        desc="Key stated metadata and findings for Publication 3 are verifiable from the provided public sources.",
        parent=parent_node,
        critical=True,
    )
    claim_verif = (
        f"Verify that the following facts are supported by the provided URLs: "
        f"title '{_safe_str(pub.title)}', journal '{_safe_str(pub.journal)}', context '{_safe_str(pub.location_context)}', "
        f"major discovery '{_safe_str(pub.major_discovery)}', and at least one quantitative datum (e.g., '{_safe_str(pub.quantitative_finding)}')."
    )
    await evaluator.verify(
        claim=claim_verif,
        node=leaf_verif,
        sources=build_sources(pub),
        additional_instruction="Confirm these facts via the article/DOI page or official press releases."
    )


async def add_journal_metric_nodes(
    evaluator: Evaluator,
    root_node,
    pubs: List[Publication]
) -> None:
    """
    Optional journal metrics aggregator.
    """
    jm_node = evaluator.add_parallel(
        id="Journal_Metric_When_Applicable",
        desc="When applicable/available, provide a journal metric/indexing indicator for each publication venue.",
        parent=root_node,
        critical=False
    )
    # Pub1 metric
    evaluator.add_custom_node(
        _has_nonempty(pubs[0].journal_metric),
        "Pub1_Journal_Metric",
        "Metric/indexing indicator for Publication 1's journal is provided when applicable.",
        parent=jm_node,
        critical=False
    )
    # Pub2 metric
    evaluator.add_custom_node(
        _has_nonempty(pubs[1].journal_metric),
        "Pub2_Journal_Metric",
        "Metric/indexing indicator for Publication 2's journal is provided when applicable.",
        parent=jm_node,
        critical=False
    )
    # Pub3 metric
    evaluator.add_custom_node(
        _has_nonempty(pubs[2].journal_metric),
        "Pub3_Journal_Metric",
        "Metric/indexing indicator for Publication 3's journal is provided when applicable.",
        parent=jm_node,
        critical=False
    )


# --------------------------------------------------------------------------- #
# Main verification builders per publication                                  #
# --------------------------------------------------------------------------- #
async def verify_publication_1(evaluator: Evaluator, parent_node, pub: Publication) -> None:
    # Pub1 container
    pub_node = evaluator.add_parallel(
        id="Publication_1",
        desc="Publication 1 requirements (Australian large prehistoric shark fossil) and required metadata.",
        parent=parent_node,
        critical=False
    )
    await verify_publication_eligibility(evaluator, pub_node, pub, prefix="Pub1")
    await verify_pub1_topic(evaluator, pub_node, pub)
    await add_pub1_required_fields(evaluator, pub_node, pub)
    await verify_pub1_verifiability(evaluator, pub_node, pub)


async def verify_publication_2(evaluator: Evaluator, parent_node, pub: Publication) -> None:
    # Pub2 container
    pub_node = evaluator.add_parallel(
        id="Publication_2",
        desc="Publication 2 requirements (Jezero Crater rover sample collected in 2024; potential biosignatures) and required metadata.",
        parent=parent_node,
        critical=False
    )
    await verify_publication_eligibility(evaluator, pub_node, pub, prefix="Pub2")
    await verify_pub2_topic(evaluator, pub_node, pub)
    await add_pub2_required_fields(evaluator, pub_node, pub)
    await verify_pub2_verifiability(evaluator, pub_node, pub)


async def verify_publication_3(evaluator: Evaluator, parent_node, pub: Publication) -> None:
    # Pub3 container
    pub_node = evaluator.add_parallel(
        id="Publication_3",
        desc="Publication 3 requirements (additional significant 2024–2025 major discovery in scope) and required metadata.",
        parent=parent_node,
        critical=False
    )
    await verify_publication_eligibility(evaluator, pub_node, pub, prefix="Pub3")
    await verify_pub3_scope(evaluator, pub_node, pub)
    await add_pub3_required_fields(evaluator, pub_node, pub)
    await verify_pub3_verifiability(evaluator, pub_node, pub)


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
    Evaluate an answer for the three natural-science publications task.
    """
    # Initialize evaluator with root parallel strategy
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

    # Extract publications from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_publications(),
        template_class=PublicationsExtraction,
        extraction_name="publications_extraction"
    )

    # Normalize to exactly three publications (pad with empty if fewer)
    pubs: List[Publication] = list(extracted.publications[:3])
    while len(pubs) < 3:
        pubs.append(Publication())

    # Check exactly three distinct publications provided (critical gate)
    # Distinctness heuristic: use DOI URL or article_link or reference_url or title as identifier
    identifiers = []
    for p in pubs:
        key = _normalize_doi_to_url(p.doi) or _safe_str(p.article_link) or _safe_str(p.reference_url) or _safe_str(p.title)
        identifiers.append(key.lower() if key else "")
    valid_ids = [k for k in identifiers if k]
    distinct = len(set(valid_ids)) == 3 and len(valid_ids) == 3
    all_minimum_present = all(_has_nonempty(p.title) or _has_nonempty(p.doi) or _has_nonempty(p.article_link) or _has_nonempty(p.reference_url) for p in pubs)

    evaluator.add_custom_node(
        result=distinct and all_minimum_present,
        id="Three_Publications_Provided",
        desc="Exactly three distinct publications are provided (Publication 1, 2, and 3).",
        parent=root,
        critical=True
    )

    # Build verification trees for each publication
    await verify_publication_1(evaluator, root, pubs[0])
    await verify_publication_2(evaluator, root, pubs[1])
    await verify_publication_3(evaluator, root, pubs[2])

    # Optional Journal Metrics (non-critical)
    await add_journal_metric_nodes(evaluator, root, pubs)

    # Return structured result
    return evaluator.get_summary()