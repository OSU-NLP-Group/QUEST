import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "kz_interdisciplinary_papers_2020_2025"
TASK_DESCRIPTION = (
    "Identify three interdisciplinary research papers published between January 2020 and December 2025 (inclusive) "
    "where at least one author is a faculty member currently affiliated with either Nazarbayev University or "
    "Al-Farabi Kazakh National University in Kazakhstan. Each paper must meet ALL specified criteria."
)

VALID_START_YEAR = 2020
VALID_END_YEAR = 2025
AS_OF_DATE_STR = "March 2026"  # for citation cutoff


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AuthorEntry(BaseModel):
    name: Optional[str] = None
    affiliation: Optional[str] = None
    country: Optional[str] = None
    is_from_kazakhstan: Optional[bool] = None
    is_from_target_univ: Optional[bool] = None  # Nazarbayev University or Al-Farabi Kazakh National University (KazNU)
    is_faculty_member: Optional[bool] = None
    faculty_profile_url: Optional[str] = None
    orcid_id: Optional[str] = None
    orcid_url: Optional[str] = None
    scopus_author_id: Optional[str] = None
    scopus_author_url: Optional[str] = None


class VenueQualityInfo(BaseModel):
    venue_name: Optional[str] = None
    venue_type: Optional[str] = None  # "journal" or "conference"
    is_q1_journal: Optional[bool] = None
    is_indexed_scopus_or_wos_cpci: Optional[bool] = None
    quality_indicator_text: Optional[str] = None
    quality_source_urls: List[str] = Field(default_factory=list)


class StructureInfo(BaseModel):
    has_abstract: Optional[bool] = None
    abstract_word_count: Optional[str] = None  # keep as string for robustness
    has_standard_sections: Optional[bool] = None
    references_count: Optional[str] = None  # keep as string for robustness
    structure_urls: List[str] = Field(default_factory=list)


class ImpactInfo(BaseModel):
    citation_count: Optional[str] = None  # keep as string for robustness
    citation_source: Optional[str] = None  # e.g., "Google Scholar", "Scopus", "Web of Science"
    impact_urls: List[str] = Field(default_factory=list)


class AccessInfo(BaseModel):
    access_status: Optional[str] = None  # e.g., "open access gold", "green via repository", "subscription-based"
    access_urls: List[str] = Field(default_factory=list)


class EthicsInfo(BaseModel):
    ethics_applicable: Optional[bool] = None  # whether human/animal/sensitive data involved
    ethics_statement_present: Optional[bool] = None
    ethics_urls: List[str] = Field(default_factory=list)


class PaperItem(BaseModel):
    # Basic identification
    title: Optional[str] = None
    doi_or_url: Optional[str] = None
    paper_urls: List[str] = Field(default_factory=list)
    publication_date: Optional[str] = None  # any format; we'll extract year heuristically

    # Authors
    authors: List[AuthorEntry] = Field(default_factory=list)
    kazakhstan_affiliation_present: Optional[bool] = None
    representative_faculty_author_name: Optional[str] = None  # name of a NU or KazNU faculty author (if given)

    # Venue quality
    venue: VenueQualityInfo = VenueQualityInfo()

    # International collaboration
    collaboration_countries_outside_kz: List[str] = Field(default_factory=list)
    collaboration_urls: List[str] = Field(default_factory=list)

    # Interdisciplinary
    disciplines: List[str] = Field(default_factory=list)  # at least 2 distinct disciplines
    integration_evidence_text: Optional[str] = None
    interdisciplinary_urls: List[str] = Field(default_factory=list)

    # Structure
    structure: StructureInfo = StructureInfo()

    # Impact
    impact: ImpactInfo = ImpactInfo()

    # Access
    access: AccessInfo = AccessInfo()

    # Ethics
    ethics: EthicsInfo = EthicsInfo()


class PapersExtraction(BaseModel):
    papers: List[PaperItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_papers() -> str:
    return """
Extract up to three research papers from the answer that meet the following overall theme:
- Published between Jan 1, 2020 and Dec 31, 2025 (inclusive).
- Interdisciplinary research.
- At least one author affiliated with either Nazarbayev University (NU) or Al-Farabi Kazakh National University (KazNU) in Kazakhstan.

For each paper, extract the following fields exactly as they appear in the answer. If something is missing, return null or an empty list as appropriate.

papers: [
  {
    "title": string | null,
    "doi_or_url": string | null,  // DOI link or a persistent URL to the paper record
    "paper_urls": [string],       // any direct URLs that identify the paper (DOI page, publisher page, indexing entry)
    "publication_date": string | null, // any human-readable date string

    "authors": [
      {
        "name": string | null,
        "affiliation": string | null,     // institution name if present
        "country": string | null,         // country of the affiliation if present
        "is_from_kazakhstan": boolean | null,
        "is_from_target_univ": boolean | null,  // true if affiliation is explicitly Nazarbayev University or Al-Farabi Kazakh National University (KazNU)
        "is_faculty_member": boolean | null,
        "faculty_profile_url": string | null,   // official university faculty profile or research portal page URL
        "orcid_id": string | null,
        "orcid_url": string | null,             // full ORCID profile URL
        "scopus_author_id": string | null,
        "scopus_author_url": string | null      // author profile URL on Scopus
      }
    ],
    "kazakhstan_affiliation_present": boolean | null,
    "representative_faculty_author_name": string | null,  // one NU/KazNU faculty author to focus on for verification

    "venue": {
      "venue_name": string | null,                 // journal or conference name
      "venue_type": string | null,                 // "journal" or "conference"
      "is_q1_journal": boolean | null,             // true if journal is Q1 in its category
      "is_indexed_scopus_or_wos_cpci": boolean | null, // true if conference proceedings indexed by Scopus or WoS CPCI
      "quality_indicator_text": string | null,     // any text like "Q1 SJR", "CiteScore Q1", "indexed by Scopus", etc.
      "quality_source_urls": [string]              // URLs supporting Q1 or indexing claims
    },

    "collaboration_countries_outside_kz": [string], // distinct countries outside Kazakhstan represented in authors' affiliations
    "collaboration_urls": [string],                 // URLs showing author affiliations (e.g., publisher page)

    "disciplines": [string],                        // at least two disciplines integrated (e.g., "computer science", "biology")
    "integration_evidence_text": string | null,     // short phrase/sentence from the answer supporting the integration
    "interdisciplinary_urls": [string],             // URLs supporting the interdisciplinary claim

    "structure": {
      "has_abstract": boolean | null,
      "abstract_word_count": string | null,        // numeric string if provided; else any text mentioning count
      "has_standard_sections": boolean | null,     // sections like Introduction/Methods/Results/Discussion
      "references_count": string | null,           // numeric string if provided
      "structure_urls": [string]                   // URLs (publisher/PDF) where structure is visible
    },

    "impact": {
      "citation_count": string | null,             // numeric string if provided
      "citation_source": string | null,            // "Google Scholar", "Scopus", or "Web of Science" (preferred), else whatever is stated
      "impact_urls": [string]                      // URLs where citation counts can be checked
    },

    "access": {
      "access_status": string | null,              // "open access gold", "green via repository", or "subscription-based"
      "access_urls": [string]                      // URLs where access status is visible
    },

    "ethics": {
      "ethics_applicable": boolean | null,         // whether the research involves human/animal subjects or sensitive data
      "ethics_statement_present": boolean | null,  // if applicable, whether ethics approval or IRB acknowledgment is stated
      "ethics_urls": [string]                      // URLs where any ethics statement appears (paper or supplement)
    }
  }
]

Rules:
- Extract only what appears in the answer. Do not invent or infer.
- Include only valid, complete URLs (add http:// if missing protocol).
- For any missing info, use null or empty list.
- Keep numbers as strings when uncertain.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def _is_nonempty(s: Optional[str]) -> bool:
    return bool(_norm(s))


def _has_url(s: Optional[str]) -> bool:
    u = _norm(s)
    return u.startswith("http://") or u.startswith("https://")


def _any_urls(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    return any(_has_url(u) for u in urls)


def _collect_sources(*args: Any) -> List[str]:
    urls: List[str] = []
    for arg in args:
        if isinstance(arg, str):
            if _has_url(arg):
                urls.append(arg)
        elif isinstance(arg, list):
            for x in arg:
                if isinstance(x, str) and _has_url(x):
                    urls.append(x)
    # deduplicate preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _parse_year_from_date(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    # Find all 4-digit years
    years = re.findall(r"(19|20)\d{2}", date_str)
    if not years:
        return None
    try:
        # Choose the last year occurrence heuristically
        yr = int(years[-1])
        return yr
    except Exception:
        return None


def _to_int_maybe(num_str: Optional[str]) -> Optional[int]:
    if not num_str:
        return None
    # Extract first integer-like sequence
    m = re.search(r"\d+", num_str)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _is_year_in_range(year: Optional[int], start_year: int, end_year: int) -> bool:
    if year is None:
        return False
    return start_year <= year <= end_year


def _allowed_citation_source(src: Optional[str]) -> bool:
    if not src:
        return False
    s = src.lower()
    return ("google scholar" in s) or ("scopus" in s) or ("web of science" in s) or ("wos" in s)


def _pick_faculty_author(paper: PaperItem) -> Optional[AuthorEntry]:
    # Priority: explicit representative_faculty_author_name match
    target_name = _norm(paper.representative_faculty_author_name).lower()
    if target_name:
        for a in paper.authors:
            if _norm(a.name).lower() == target_name:
                return a

    # Next: first author from target university flags or affiliation contains NU/KazNU keywords
    for a in paper.authors:
        if a.is_from_target_univ:
            return a

    for a in paper.authors:
        aff = _norm(a.affiliation).lower()
        if any(
            kw in aff
            for kw in [
                "nazarbayev university",
                "al-farabi kazakh national university",
                "al farabi kazakh national university",
                "kaznu",
                "al-farabi kazakh national univ",
                "al-farabi kazakh national",
                "al farabi kaznu",
            ]
        ):
            return a

    return None


def _paper_main_sources(p: PaperItem) -> List[str]:
    return _collect_sources(p.doi_or_url, p.paper_urls)


def _venue_quality_sources(p: PaperItem) -> List[str]:
    return _collect_sources(p.venue.quality_source_urls)


def _structure_sources(p: PaperItem) -> List[str]:
    return _collect_sources(p.structure.structure_urls, _paper_main_sources(p))


def _collab_sources(p: PaperItem) -> List[str]:
    return _collect_sources(p.collaboration_urls, _paper_main_sources(p))


def _interdisc_sources(p: PaperItem) -> List[str]:
    return _collect_sources(p.interdisciplinary_urls, _paper_main_sources(p))


def _impact_sources(p: PaperItem) -> List[str]:
    return _collect_sources(p.impact.impact_urls)


def _access_sources(p: PaperItem) -> List[str]:
    return _collect_sources(p.access.access_urls, _paper_main_sources(p))


def _ethics_sources(p: PaperItem) -> List[str]:
    return _collect_sources(p.ethics.ethics_urls, _paper_main_sources(p))


# --------------------------------------------------------------------------- #
# Verification for a single paper                                             #
# --------------------------------------------------------------------------- #
async def verify_single_paper(evaluator: Evaluator, parent_node, paper: PaperItem, paper_idx: int) -> None:
    """
    Build and run verification sub-tree for one paper according to the rubric.
    paper_idx is 1-based.
    """
    # Container for this paper (allow partial credit between papers)
    paper_node = evaluator.add_parallel(
        id=f"paper_{paper_idx}",
        desc=("First" if paper_idx == 1 else "Second" if paper_idx == 2 else "Third") + " paper meets all specified criteria",
        parent=parent_node,
        critical=False
    )

    # -------------------- Basic info -------------------- #
    basic_node = evaluator.add_parallel(
        id=f"paper_{paper_idx}_basic_info",
        desc="Basic paper identification and metadata",
        parent=paper_node,
        critical=True
    )

    # title + DOI/URL provided
    title_doi_ok = _is_nonempty(paper.title) and _is_nonempty(paper.doi_or_url)
    evaluator.add_custom_node(
        result=title_doi_ok,
        id=f"paper_{paper_idx}_title_doi",
        desc="Paper title and DOI/persistent URL provided",
        parent=basic_node,
        critical=True
    )

    # publication in [2020, 2025]
    pub_year = _parse_year_from_date(paper.publication_date)
    in_range = _is_year_in_range(pub_year, VALID_START_YEAR, VALID_END_YEAR)
    evaluator.add_custom_node(
        result=in_range,
        id=f"paper_{paper_idx}_publication_date",
        desc=f"Publication date is between January 1, {VALID_START_YEAR} and December 31, {VALID_END_YEAR} inclusive",
        parent=basic_node,
        critical=True
    )

    # reference URL provided for identification verification
    basic_has_url = _has_url(paper.doi_or_url) or _any_urls(paper.paper_urls)
    evaluator.add_custom_node(
        result=basic_has_url,
        id=f"paper_{paper_idx}_basic_url",
        desc="Reference URL provided for paper identification verification",
        parent=basic_node,
        critical=True
    )

    # -------------------- Authors & Central Asian affiliation -------------------- #
    authors_node = evaluator.add_parallel(
        id=f"paper_{paper_idx}_authors",
        desc="Author information and Central Asian affiliation",
        parent=paper_node,
        critical=True
    )

    # "Complete author list with institutional affiliations provided" -> existence/format check on extracted data
    authors_provided = len(paper.authors) > 0 and all(_is_nonempty(a.name) and _is_nonempty(a.affiliation) for a in paper.authors)
    evaluator.add_custom_node(
        result=authors_provided,
        id=f"paper_{paper_idx}_author_list",
        desc="Complete author list with institutional affiliations provided",
        parent=authors_node,
        critical=True
    )

    # At least one author affiliated with NU or KazNU (verify with sources)
    kz_aff_leaf = evaluator.add_leaf(
        id=f"paper_{paper_idx}_kazakhstan_affiliation",
        desc="At least one author affiliated with Nazarbayev University or Al-Farabi Kazakh National University",
        parent=authors_node,
        critical=True
    )
    kz_claim = (
        "At least one of the paper's authors is affiliated with either Nazarbayev University or Al-Farabi Kazakh National University (KazNU). "
        "Look for author affiliation lines on the paper page or PDF."
    )
    await evaluator.verify(
        claim=kz_claim,
        node=kz_aff_leaf,
        sources=_paper_main_sources(paper),
        additional_instruction="Accept reasonable variants of the university names (e.g., 'KazNU', 'Al-Farabi Kazakh National Univ.')."
    )

    # Faculty verification (sequential)
    fac_node = evaluator.add_sequential(
        id=f"paper_{paper_idx}_faculty_verification",
        desc="Central Asian university-affiliated author has verifiable faculty profile",
        parent=authors_node,
        critical=True
    )

    # Create 'faculty_url' presence first (so it's a blocking critical sibling for verification)
    faculty_author = _pick_faculty_author(paper)
    faculty_url_present = _is_nonempty(faculty_author.faculty_profile_url if faculty_author else None) and _has_url(
        faculty_author.faculty_profile_url if faculty_author else None
    )
    fac_url_presence_node = evaluator.add_custom_node(
        result=faculty_url_present,
        id=f"paper_{paper_idx}_faculty_url",
        desc="Reference URL provided for faculty profile verification",
        parent=fac_node,
        critical=True
    )

    # Faculty profile exists and is official
    fac_exists_leaf = evaluator.add_leaf(
        id=f"paper_{paper_idx}_faculty_profile_exists",
        desc="Faculty profile found on official university website or research portal",
        parent=fac_node,
        critical=True
    )
    fac_name = _norm(faculty_author.name if faculty_author else None) or "the author"
    fac_claim = (
        f"This is an official faculty profile page for {fac_name} on the university's official website or research portal "
        f"(Nazarbayev University or Al-Farabi Kazakh National University)."
    )
    await evaluator.verify(
        claim=fac_claim,
        node=fac_exists_leaf,
        sources=(faculty_author.faculty_profile_url if faculty_author else None),
        additional_instruction="Confirm that the page is clearly part of the official university domain or portal (e.g., nu.edu.kz, kaznu.kz)."
    )

    # Author ID verification (ORCID or Scopus)
    id_node = evaluator.add_sequential(
        id=f"paper_{paper_idx}_author_id",
        desc="Central Asian university-affiliated author has ORCID iD or Scopus Author ID linked to publication",
        parent=authors_node,
        critical=True
    )

    # Presence of an identifier URL first
    id_url: Optional[str] = None
    if faculty_author:
        id_url = (
            faculty_author.orcid_url
            if _has_url(faculty_author.orcid_url)
            else (faculty_author.scopus_author_url if _has_url(faculty_author.scopus_author_url) else None)
        )
    id_present = _is_nonempty(id_url)
    id_url_presence_node = evaluator.add_custom_node(
        result=id_present,
        id=f"paper_{paper_idx}_id_url",
        desc="Reference URL provided for author identifier verification",
        parent=id_node,
        critical=True
    )

    id_verify_leaf = evaluator.add_leaf(
        id=f"paper_{paper_idx}_id_type_verified",
        desc="ORCID iD or Scopus Author ID identified and linkable to paper",
        parent=id_node,
        critical=True
    )
    id_claim = (
        f"The author profile at this URL corresponds to {fac_name} and shows a link or association to this publication "
        f"(by title '{_norm(paper.title)}' or DOI '{_norm(paper.doi_or_url)}')."
    )
    await evaluator.verify(
        claim=id_claim,
        node=id_verify_leaf,
        sources=id_url,
        additional_instruction="Confirm the author identity matches and that the publication is listed or clearly associated on the profile."
    )

    # -------------------- Venue quality -------------------- #
    venue_node = evaluator.add_sequential(
        id=f"paper_{paper_idx}_venue",
        desc="Publication venue quality verification",
        parent=paper_node,
        critical=True
    )

    # Venue identification (existence)
    venue_identified = _is_nonempty(paper.venue.venue_name)
    evaluator.add_custom_node(
        result=venue_identified,
        id=f"paper_{paper_idx}_venue_identification",
        desc="Publication venue (journal or conference) identified",
        parent=venue_node,
        critical=True
    )

    quality_node = evaluator.add_sequential(
        id=f"paper_{paper_idx}_venue_quality",
        desc="Venue is Q1 journal OR Scopus/WoS CPCI-indexed conference",
        parent=venue_node,
        critical=True
    )

    # Presence of quality URLs first
    quality_url_present = _any_urls(paper.venue.quality_source_urls)
    quality_url_node = evaluator.add_custom_node(
        result=quality_url_present,
        id=f"paper_{paper_idx}_quality_url",
        desc="Reference URL provided for venue quality verification",
        parent=quality_node,
        critical=True
    )

    # Verify Q1 or indexing
    quality_leaf = evaluator.add_leaf(
        id=f"paper_{paper_idx}_quality_verified",
        desc="Q1 quartile ranking or conference indexing status confirmed",
        parent=quality_node,
        critical=True
    )
    vname = _norm(paper.venue.venue_name)
    if paper.venue.is_q1_journal:
        quality_claim = (
            f"The journal '{vname}' is ranked Q1 (top quartile, top 25%) in its subject category (e.g., SJR/Scimago or CiteScore) "
            f"for the relevant period."
        )
    elif paper.venue.is_indexed_scopus_or_wos_cpci:
        quality_claim = (
            f"The conference proceedings for '{vname}' are indexed in Scopus or Web of Science Conference Proceedings Citation Index (CPCI)."
        )
    else:
        # Generic claim if the specific flags were not extracted but URLs were provided
        quality_claim = (
            f"The venue '{vname}' meets the quality criterion: either a Q1 journal or a conference indexed by Scopus/WoS CPCI, "
            f"as supported by the provided evidence."
        )
    await evaluator.verify(
        claim=quality_claim,
        node=quality_leaf,
        sources=_venue_quality_sources(paper),
        additional_instruction="Use the provided URL(s) (e.g., Scimago, Elsevier, Scopus, or WoS pages) to confirm the stated venue quality."
    )

    # -------------------- International collaboration -------------------- #
    collab_node = evaluator.add_parallel(
        id=f"paper_{paper_idx}_collaboration",
        desc="International collaboration verification",
        parent=paper_node,
        critical=True
    )

    # Countries identified: at least 2 outside KZ (verify on paper page)
    countries_list = paper.collaboration_countries_outside_kz or []
    countries_text = ", ".join(countries_list) if countries_list else "None listed"
    countries_leaf = evaluator.add_leaf(
        id=f"paper_{paper_idx}_countries_identified",
        desc="Co-author affiliations from at least 2 countries outside Kazakhstan identified",
        parent=collab_node,
        critical=True
    )
    countries_claim = (
        f"The publication lists author affiliations from at least two distinct countries outside Kazakhstan: {countries_text}."
    )
    await evaluator.verify(
        claim=countries_claim,
        node=countries_leaf,
        sources=_collab_sources(paper),
        additional_instruction="Verify from the author affiliation lines on the publisher or PDF page that at least two non-Kazakhstan countries are present."
    )

    # Affiliation clarity: institutions clearly stated
    clarity_leaf = evaluator.add_leaf(
        id=f"paper_{paper_idx}_affiliation_clarity",
        desc="Institutional affiliations clearly stated in publication",
        parent=collab_node,
        critical=True
    )
    clarity_claim = (
        "The publication clearly states institutional affiliations for each author (e.g., institution names next to authors or in metadata)."
    )
    await evaluator.verify(
        claim=clarity_claim,
        node=clarity_leaf,
        sources=_collab_sources(paper),
        additional_instruction="Look for explicit institution names associated with author names on the paper page or PDF."
    )

    # Collaboration reference URL provided
    collab_url_present = _any_urls(paper.collaboration_urls) or basic_has_url
    evaluator.add_custom_node(
        result=collab_url_present,
        id=f"paper_{paper_idx}_collaboration_url",
        desc="Reference URL provided for collaboration verification",
        parent=collab_node,
        critical=True
    )

    # -------------------- Interdisciplinary nature -------------------- #
    inter_node = evaluator.add_parallel(
        id=f"paper_{paper_idx}_interdisciplinary",
        desc="Interdisciplinary nature verification",
        parent=paper_node,
        critical=True
    )

    # Disciplines identified: at least 2 distinct
    disciplines_list = [d.strip() for d in (paper.disciplines or []) if _is_nonempty(d)]
    disciplines_text = ", ".join(disciplines_list) if disciplines_list else "None listed"
    disc_leaf = evaluator.add_leaf(
        id=f"paper_{paper_idx}_disciplines_identified",
        desc="At least 2 distinct academic disciplines explicitly integrated in research",
        parent=inter_node,
        critical=True
    )
    disc_claim = f"The research explicitly integrates at least two distinct disciplines: {disciplines_text}."
    await evaluator.verify(
        claim=disc_claim,
        node=disc_leaf,
        sources=_interdisc_sources(paper),
        additional_instruction="Confirm the integration from the abstract/introduction/methods or venue scope; allow reasonable synonyms for disciplines."
    )

    # Evidence of integration in methodology/framework/application
    integ_leaf = evaluator.add_leaf(
        id=f"paper_{paper_idx}_integration_evidence",
        desc="Evidence of disciplinary integration in methodology, framework, or application",
        parent=inter_node,
        critical=True
    )
    evidence_text = _norm(paper.integration_evidence_text) or "The paper describes combining methods or concepts from multiple fields."
    integ_claim = (
        f"Evidence shows that the paper integrates disciplines in its methodology/framework/application. Example/evidence: {evidence_text}"
    )
    await evaluator.verify(
        claim=integ_claim,
        node=integ_leaf,
        sources=_interdisc_sources(paper),
        additional_instruction="Look for explicit mentions of cross-disciplinary methods, frameworks, datasets, or application areas."
    )

    # Interdisciplinary reference URL provided
    inter_url_present = _any_urls(paper.interdisciplinary_urls) or basic_has_url
    evaluator.add_custom_node(
        result=inter_url_present,
        id=f"paper_{paper_idx}_interdisciplinary_url",
        desc="Reference URL provided for interdisciplinary nature verification",
        parent=inter_node,
        critical=True
    )

    # -------------------- Publication structure -------------------- #
    struct_node = evaluator.add_parallel(
        id=f"paper_{paper_idx}_structure",
        desc="Publication structure and formatting",
        parent=paper_node,
        critical=True
    )

    # Abstract present and within 100–400 words (verify via web)
    abs_leaf = evaluator.add_leaf(
        id=f"paper_{paper_idx}_abstract",
        desc="Abstract present and within 100-400 word range",
        parent=struct_node,
        critical=True
    )
    abs_claim = "The paper includes an abstract and its length is between 100 and 400 words."
    await evaluator.verify(
        claim=abs_claim,
        node=abs_leaf,
        sources=_structure_sources(paper),
        additional_instruction="Estimate the abstract word count from the page/PDF. If very close to bounds, allow small counting variability."
    )

    # Standard academic sections present
    sec_leaf = evaluator.add_leaf(
        id=f"paper_{paper_idx}_sections",
        desc="Standard academic structure with identifiable sections present",
        parent=struct_node,
        critical=True
    )
    sec_claim = (
        "The paper has a standard structure with identifiable sections such as Introduction, Methods (or Materials and Methods), Results, and Discussion."
    )
    await evaluator.verify(
        claim=sec_claim,
        node=sec_leaf,
        sources=_structure_sources(paper),
        additional_instruction="Minor variations in section naming (e.g., 'Methodology', 'Findings') are acceptable."
    )

    # Reference list contains at least 15 citations
    ref_leaf = evaluator.add_leaf(
        id=f"paper_{paper_idx}_references",
        desc="Reference list contains at least 15 scholarly citations",
        parent=struct_node,
        critical=True
    )
    ref_claim = "The reference list contains at least 15 scholarly citations."
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=_structure_sources(paper),
        additional_instruction="Count or estimate the references in the References/Bibliography section; ignore acknowledgments or footnotes."
    )

    # Structure URL provided
    struct_url_present = _any_urls(paper.structure.structure_urls) or basic_has_url
    evaluator.add_custom_node(
        result=struct_url_present,
        id=f"paper_{paper_idx}_structure_url",
        desc="Reference URL provided for structure verification",
        parent=struct_node,
        critical=True
    )

    # -------------------- Impact (citations) -------------------- #
    impact_node = evaluator.add_parallel(
        id=f"paper_{paper_idx}_impact",
        desc="Research impact metrics",
        parent=paper_node,
        critical=True
    )

    # At least 3 citations excluding self-citations (verify via citation source)
    cites_leaf = evaluator.add_leaf(
        id=f"paper_{paper_idx}_citations_count",
        desc="Paper has received at least 3 citations excluding self-citations",
        parent=impact_node,
        critical=True
    )
    cites_claim = (
        f"As of {AS_OF_DATE_STR}, this paper has at least 3 citations excluding self-citations."
    )
    await evaluator.verify(
        claim=cites_claim,
        node=cites_leaf,
        sources=_impact_sources(paper),
        additional_instruction="Use the provided citation source page(s). Exclude obvious self-citations (same authors). Consider close variants of the paper title/DOI."
    )

    # Citation source identified (custom check)
    evaluator.add_custom_node(
        result=_allowed_citation_source(paper.impact.citation_source),
        id=f"paper_{paper_idx}_citation_source",
        desc="Citation count source identified (Google Scholar, Scopus, or Web of Science)",
        parent=impact_node,
        critical=True
    )

    # Impact URL provided
    impact_url_present = _any_urls(paper.impact.impact_urls)
    evaluator.add_custom_node(
        result=impact_url_present,
        id=f"paper_{paper_idx}_impact_url",
        desc="Reference URL provided for citation verification",
        parent=impact_node,
        critical=True
    )

    # -------------------- Access status -------------------- #
    access_node = evaluator.add_sequential(
        id=f"paper_{paper_idx}_access",
        desc="Publication accessibility information",
        parent=paper_node,
        critical=True
    )

    # Access URL presence first
    access_url_present = _any_urls(paper.access.access_urls) or basic_has_url
    evaluator.add_custom_node(
        result=access_url_present,
        id=f"paper_{paper_idx}_access_url",
        desc="Reference URL provided for access status verification",
        parent=access_node,
        critical=True
    )

    # Access status verified by URL
    access_leaf = evaluator.add_leaf(
        id=f"paper_{paper_idx}_access_status",
        desc="Access status clearly identified (open access gold, green, or subscription)",
        parent=access_node,
        critical=True
    )
    access_status_text = _norm(paper.access.access_status) or "clearly identifiable (open access gold, green via repository, or subscription-based)"
    access_claim = f"The paper's access status is {access_status_text}."
    await evaluator.verify(
        claim=access_claim,
        node=access_leaf,
        sources=_access_sources(paper),
        additional_instruction="Check the publisher or repository page for labels like Open Access/Gold, Green (repository), or Subscription."
    )

    # -------------------- Ethics (non-critical) -------------------- #
    ethics_node = evaluator.add_sequential(
        id=f"paper_{paper_idx}_ethics",
        desc="Ethical compliance verification",
        parent=paper_node,
        critical=False
    )

    # Ethics URL presence first (non-critical)
    ethics_url_present = _any_urls(paper.ethics.ethics_urls) or basic_has_url
    evaluator.add_custom_node(
        result=ethics_url_present,
        id=f"paper_{paper_idx}_ethics_url",
        desc="Reference URL provided for ethics statement verification if applicable",
        parent=ethics_node,
        critical=False
    )

    # Applicability (verify from paper if human/animal/sensitive data)
    ethics_app_leaf = evaluator.add_leaf(
        id=f"paper_{paper_idx}_ethics_applicability",
        desc="Determination of whether research involves human subjects or sensitive data",
        parent=ethics_node,
        critical=False
    )
    ethics_app_claim = (
        "Determine whether the research involves human or animal subjects, or sensitive data collection. "
        "If not applicable (e.g., only simulations or public datasets without sensitive content), indicate not applicable."
    )
    await evaluator.verify(
        claim=ethics_app_claim,
        node=ethics_app_leaf,
        sources=_ethics_sources(paper),
        additional_instruction="Rely on the methods/ethics sections or footnotes. A 'not applicable' situation should be acceptable."
    )

    # Ethics statement (verify only conceptually; the LLM will reason based on the page)
    ethics_stmt_leaf = evaluator.add_leaf(
        id=f"paper_{paper_idx}_ethics_statement",
        desc="If applicable, ethics approval or IRB acknowledgment is stated",
        parent=ethics_node,
        critical=False
    )
    ethics_stmt_claim = (
        "If the research involves human/animal subjects or sensitive data, the paper (or supplement) includes an ethics approval or IRB acknowledgment."
    )
    await evaluator.verify(
        claim=ethics_stmt_claim,
        node=ethics_stmt_leaf,
        sources=_ethics_sources(paper),
        additional_instruction="If ethics is not applicable, consider this requirement not necessary. Otherwise, look for explicit IRB/ethics statements."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the Kazakhstan interdisciplinary papers task using the Mind2Web2 framework.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # allow partial credit across the three papers
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

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_papers(),
        template_class=PapersExtraction,
        extraction_name="papers_extraction",
    )

    # Normalize to exactly 3 papers (pad if fewer)
    papers = list(extracted.papers or [])
    while len(papers) < 3:
        papers.append(PaperItem())

    # Build per-paper verification subtrees
    for i in range(3):
        await verify_single_paper(evaluator, root, papers[i], paper_idx=i + 1)

    # Add some helpful custom info summary
    try:
        provided_titles = [p.title for p in papers if _is_nonempty(p.title)]
        evaluator.add_custom_info(
            {
                "num_papers_parsed": len(papers),
                "titles_provided": provided_titles,
                "evaluation_time_utc": datetime.utcnow().isoformat()
            },
            info_type="run_meta",
            info_name="run_metadata"
        )
    except Exception:
        pass

    return evaluator.get_summary()