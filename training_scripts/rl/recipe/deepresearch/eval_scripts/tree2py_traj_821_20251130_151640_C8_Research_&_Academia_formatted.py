import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "mars_lightning_nature_2025"
TASK_DESCRIPTION = (
    "Locate the recent Nature journal publication from November 2025 that reports the first detection of "
    "electrical discharges (lightning) on Mars by NASA's Perseverance rover. Extract and compile the following "
    "information for inclusion in a comprehensive literature review: (1) complete bibliographic details including "
    "journal name, publication date, volume, issue, page range, DOI, and article title; (2) lead author name and "
    "institutional affiliation; (3) names and affiliations of key co-authors including those from Johns Hopkins "
    "Applied Physics Laboratory, LATMOS France, and Purdue University; (4) methodology specifications including "
    "the instrument used, data collection time period, total hours of recordings analyzed, and number of discharge "
    "events detected; and (5) data access information including the primary data repository and the specific DOI for "
    "accessing the SuperCam acoustic data."
)

# -----------------------------------------------------------------------------
# Ground truth values for verification
# -----------------------------------------------------------------------------
GT = {
    "journal_name": "Nature",
    "publication_date_preferred": "26 November 2025",  # allow "November 26, 2025" variants
    "publication_date_alt": "November 26, 2025",
    "volume": "647",
    "issue": "8091",
    "page_range": "865–869",  # allow "865-869"
    "article_title": "Detection of triboelectric discharges during dust events on Mars",
    "doi_plain": "10.1038/s41586-025-09736-y",
    "doi_url": "https://doi.org/10.1038/s41586-025-09736-y",
    "lead_author_name": "Baptiste Chide",
    "lead_author_affiliation_core": "Institut de Recherche en Astrophysique et Planétologie (IRAP), Université de Toulouse, CNRS, Toulouse, France",
    "coauthor_apl_name": "Ralph D. Lorenz",
    "coauthor_apl_affil": "Johns Hopkins Applied Physics Laboratory",
    "coauthor_latmos_name": "Franck Montmessin",
    "coauthor_latmos_affil_core": "Laboratoire Atmosphères, Milieux, Observations Spatiales (LATMOS), CNRS",
    "coauthor_purdue_name": "Roger C. Wiens",
    "coauthor_purdue_affil": "Purdue University, Department of Earth, Atmospheric, and Planetary Sciences",
    "instrument_used_core": "SuperCam microphone aboard NASA's Perseverance rover",
    "data_collection_period_core": "two Martian years (~1,374 Earth days)",
    "total_recording_time_hours": "28",
    "discharge_event_count": "55",
    "data_repository_name": "NASA Planetary Data System (PDS)",
    "supercam_acoustic_doi_plain": "10.17189/1522646",
    "supercam_acoustic_doi_url": "https://doi.org/10.17189/1522646",
}

# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class BibliographicDetails(BaseModel):
    journal_name: Optional[str] = None
    publication_date: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    page_range: Optional[str] = None
    doi: Optional[str] = None
    article_title: Optional[str] = None
    nature_article_url: Optional[str] = None
    doi_url: Optional[str] = None
    supporting_urls: List[str] = Field(default_factory=list)


class LeadAuthorInfo(BaseModel):
    name: Optional[str] = None
    affiliation: Optional[str] = None


class CoAuthorUnit(BaseModel):
    name: Optional[str] = None
    affiliation: Optional[str] = None


class MethodologySpecifications(BaseModel):
    instrument_used: Optional[str] = None
    data_collection_period: Optional[str] = None
    total_recording_time_analyzed: Optional[str] = None
    discharge_event_count: Optional[str] = None


class DataAccessInformation(BaseModel):
    primary_data_repository: Optional[str] = None
    supercam_acoustic_data_doi: Optional[str] = None
    data_repository_url: Optional[str] = None


class MarsLightningExtraction(BaseModel):
    bibliography: Optional[BibliographicDetails] = None
    lead_author: Optional[LeadAuthorInfo] = None
    coauthor_jhu_apl: Optional[CoAuthorUnit] = None
    coauthor_latmos_france: Optional[CoAuthorUnit] = None
    coauthor_purdue_university: Optional[CoAuthorUnit] = None
    methodology: Optional[MethodologySpecifications] = None
    data_access: Optional[DataAccessInformation] = None


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_mars_lightning_info() -> str:
    return """
    Extract from the answer exactly the structured information listed below about the specified Nature article reporting lightning on Mars observed by Perseverance:

    1) bibliography:
       - journal_name
       - publication_date (verbatim as written; formats like "26 November 2025" or "November 26, 2025" are acceptable)
       - volume
       - issue
       - page_range (keep dash/en-dash as written)
       - doi (the DOI string as written; may be with or without the https://doi.org/ prefix)
       - article_title
       - nature_article_url (the direct URL to the article on nature.com if provided)
       - doi_url (the full https://doi.org/... URL if provided)
       - supporting_urls (list of any other URLs cited in the answer that are relevant to this article’s details)
    
    2) lead_author:
       - name
       - affiliation (verbatim as written; include full institute and organization string)

    3) coauthor_jhu_apl:
       - name (co-author from Johns Hopkins Applied Physics Laboratory)
       - affiliation (verbatim)

    4) coauthor_latmos_france:
       - name (co-author from LATMOS France)
       - affiliation (verbatim)

    5) coauthor_purdue_university:
       - name (co-author from Purdue University)
       - affiliation (verbatim)

    6) methodology:
       - instrument_used
       - data_collection_period (e.g., "two Martian years (~1,374 Earth days)")
       - total_recording_time_analyzed (e.g., "28 hours")
       - discharge_event_count (e.g., "55")

    7) data_access:
       - primary_data_repository (e.g., "NASA Planetary Data System (PDS)")
       - supercam_acoustic_data_doi (as written; may be the DOI or full https://doi.org/ link)
       - data_repository_url (if a specific landing page URL is provided)

    IMPORTANT:
    - Extract only what is explicitly present in the answer.
    - For any missing field, return null (for strings) or an empty list (for URL lists).
    - For URL fields, include the full URL; if protocol is missing, prepend http://.
    - Do not fabricate or infer values.
    """


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _combine_sources(*args: Any) -> Optional[List[str]]:
    """Combine optional URL(s) and deduplicate; return None if empty."""
    seen = set()
    out: List[str] = []
    for a in args:
        if not a:
            continue
        if isinstance(a, str):
            s = a.strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        elif isinstance(a, list):
            for s in a:
                if not s:
                    continue
                ss = s.strip()
                if ss and ss not in seen:
                    seen.add(ss)
                    out.append(ss)
    return out if out else None


def _safe(s: Optional[str]) -> str:
    return s if s is not None else ""


# -----------------------------------------------------------------------------
# Verification subtrees
# -----------------------------------------------------------------------------
async def verify_bibliographic_details(evaluator: Evaluator, parent_node, ex: MarsLightningExtraction) -> None:
    bib = ex.bibliography or BibliographicDetails()
    sources = _combine_sources(
        bib.nature_article_url,
        bib.doi_url,
        bib.supporting_urls,
        GT["doi_url"]
    )

    bnode = evaluator.add_parallel(
        id="bibliographic_details",
        desc="Provide complete bibliographic details for citation.",
        parent=parent_node,
        critical=True
    )

    # Journal name
    n = evaluator.add_leaf(
        id="bibliographic_journal_name",
        desc="Journal name is Nature.",
        parent=bnode,
        critical=True
    )
    claim = f"The journal name stated in the answer ('{_safe(bib.journal_name)}') equals 'Nature' (case-insensitive)."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=sources,
        additional_instruction="Accept 'Nature' regardless of letter casing or minor variants like 'Nature (London)'."
    )

    # Publication date
    n = evaluator.add_leaf(
        id="bibliographic_publication_date",
        desc="Publication date is November 26, 2025.",
        parent=bnode,
        critical=True
    )
    claim = (
        f"The publication date stated in the answer ('{_safe(bib.publication_date)}') corresponds to "
        f"{GT['publication_date_preferred']} (i.e., {GT['publication_date_alt']})."
    )
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=sources,
        additional_instruction="Allow both '26 November 2025' and 'November 26, 2025' formats and common abbreviations."
    )

    # Volume
    n = evaluator.add_leaf(
        id="bibliographic_volume",
        desc="Volume is 647.",
        parent=bnode,
        critical=True
    )
    claim = f"The volume stated in the answer ('{_safe(bib.volume)}') equals {GT['volume']}."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=sources,
        additional_instruction="Match the numeric volume; ignore surrounding text."
    )

    # Issue
    n = evaluator.add_leaf(
        id="bibliographic_issue",
        desc="Issue is 8091.",
        parent=bnode,
        critical=True
    )
    claim = f"The issue stated in the answer ('{_safe(bib.issue)}') equals {GT['issue']}."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=sources,
        additional_instruction="Match the numeric issue; ignore surrounding text."
    )

    # Page range
    n = evaluator.add_leaf(
        id="bibliographic_page_range",
        desc="Page range is 865–869.",
        parent=bnode,
        critical=True
    )
    claim = f"The page range stated in the answer ('{_safe(bib.page_range)}') corresponds to {GT['page_range']}."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=sources,
        additional_instruction="Allow hyphen vs en-dash and minor spacing: '865–869' == '865-869'."
    )

    # DOI
    n = evaluator.add_leaf(
        id="bibliographic_doi",
        desc=f"DOI is {GT['doi_url']}.",
        parent=bnode,
        critical=True
    )
    claim = (
        f"The DOI stated in the answer ('{_safe(bib.doi)}') equals '{GT['doi_plain']}' "
        f"(with or without the 'https://doi.org/' prefix)."
    )
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=sources,
        additional_instruction="Treat '10.1038/...' and 'https://doi.org/10.1038/...' as equivalent."
    )

    # Article title
    n = evaluator.add_leaf(
        id="bibliographic_article_title",
        desc=f"Article title is '{GT['article_title']}'.",
        parent=bnode,
        critical=True
    )
    claim = (
        f"The article title stated in the answer ('{_safe(bib.article_title)}') matches "
        f"'{GT['article_title']}' allowing minor punctuation/capitalization differences."
    )
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=sources,
        additional_instruction="Allow minor punctuation/case differences; focus on semantic equivalence."
    )


async def verify_lead_author(evaluator: Evaluator, parent_node, ex: MarsLightningExtraction) -> None:
    lead = ex.lead_author or LeadAuthorInfo()
    bib = ex.bibliography or BibliographicDetails()
    sources = _combine_sources(bib.nature_article_url, bib.doi_url, bib.supporting_urls)

    lnode = evaluator.add_parallel(
        id="lead_author_information",
        desc="Provide lead author name and institutional affiliation.",
        parent=parent_node,
        critical=True
    )

    # Lead author name
    n = evaluator.add_leaf(
        id="lead_author_name",
        desc=f"Lead author is {GT['lead_author_name']}.",
        parent=lnode,
        critical=True
    )
    claim = (
        f"The lead author name stated in the answer ('{_safe(lead.name)}') matches "
        f"'{GT['lead_author_name']}' (allow diacritics and minor variants)."
    )
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=sources,
        additional_instruction="Allow minor variations like missing middle initials or diacritics."
    )

    # Lead author affiliation
    n = evaluator.add_leaf(
        id="lead_author_affiliation",
        desc="Lead author affiliation includes Institut de Recherche en Astrophysique et Planétologie (IRAP), Université de Toulouse, CNRS, Toulouse, France.",
        parent=lnode,
        critical=True
    )
    claim = (
        f"The lead author affiliation stated in the answer ('{_safe(lead.affiliation)}') includes "
        f"IRAP, Université de Toulouse, and CNRS (Toulouse, France)."
    )
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=sources,
        additional_instruction="Check for presence of the key institutions IRAP, Université de Toulouse, and CNRS; allow minor French accent/wording variations."
    )


async def verify_key_coauthors(evaluator: Evaluator, parent_node, ex: MarsLightningExtraction) -> None:
    bib = ex.bibliography or BibliographicDetails()
    sources = _combine_sources(bib.nature_article_url, bib.doi_url, bib.supporting_urls)

    cnode = evaluator.add_parallel(
        id="key_coauthors_information",
        desc="Provide names and affiliations of key co-authors including those from the specified institutions.",
        parent=parent_node,
        critical=True
    )

    # Johns Hopkins APL
    apl = ex.coauthor_jhu_apl or CoAuthorUnit()
    apl_node = evaluator.add_parallel(
        id="coauthor_johns_hopkins_apl",
        desc="Co-author from Johns Hopkins Applied Physics Laboratory: name and affiliation.",
        parent=cnode,
        critical=True
    )
    n = evaluator.add_leaf(
        id="coauthor_jhu_apl_name",
        desc="Co-author name is Ralph D. Lorenz.",
        parent=apl_node,
        critical=True
    )
    claim = f"The co-author name stated for the Johns Hopkins APL collaborator ('{_safe(apl.name)}') matches '{GT['coauthor_apl_name']}' (allow missing middle initial)."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=sources,
        additional_instruction="Allow 'Ralph Lorenz' vs 'Ralph D. Lorenz' as equivalent."
    )
    n = evaluator.add_leaf(
        id="coauthor_jhu_apl_affiliation",
        desc="Affiliation is Johns Hopkins Applied Physics Laboratory.",
        parent=apl_node,
        critical=True
    )
    claim = f"The affiliation stated for this co-author ('{_safe(apl.affiliation)}') includes '{GT['coauthor_apl_affil']}' or 'Johns Hopkins University Applied Physics Laboratory'."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=sources,
        additional_instruction="Accept either 'Johns Hopkins Applied Physics Laboratory' or 'Johns Hopkins University Applied Physics Laboratory'."
    )

    # LATMOS France
    latmos = ex.coauthor_latmos_france or CoAuthorUnit()
    lat_node = evaluator.add_parallel(
        id="coauthor_latmos_france",
        desc="Co-author from LATMOS France: name and affiliation.",
        parent=cnode,
        critical=True
    )
    n = evaluator.add_leaf(
        id="coauthor_latmos_france_name",
        desc="Co-author name is Franck Montmessin.",
        parent=lat_node,
        critical=True
    )
    claim = f"The LATMOS co-author name stated in the answer ('{_safe(latmos.name)}') matches '{GT['coauthor_latmos_name']}'."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=sources,
        additional_instruction="Allow minor accent/case variations."
    )
    n = evaluator.add_leaf(
        id="coauthor_latmos_france_affiliation",
        desc="Affiliation includes Laboratoire Atmosphères, Milieux, Observations Spatiale (LATMOS), CNRS.",
        parent=lat_node,
        critical=True
    )
    claim = (
        f"The LATMOS co-author affiliation stated ('{_safe(latmos.affiliation)}') includes "
        f"'{GT['coauthor_latmos_affil_core']}' allowing pluralization ('Spatiale' vs 'Spatiales')."
    )
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=sources,
        additional_instruction="Allow 'Laboratoire Atmosphères, Milieux, Observations Spatiale(s) (LATMOS), CNRS' with or without diacritics."
    )

    # Purdue University
    purdue = ex.coauthor_purdue_university or CoAuthorUnit()
    pur_node = evaluator.add_parallel(
        id="coauthor_purdue_university",
        desc="Co-author from Purdue University: name and affiliation.",
        parent=cnode,
        critical=True
    )
    n = evaluator.add_leaf(
        id="coauthor_purdue_university_name",
        desc="Co-author name is Roger C. Wiens.",
        parent=pur_node,
        critical=True
    )
    claim = f"The Purdue co-author name stated in the answer ('{_safe(purdue.name)}') matches '{GT['coauthor_purdue_name']}' (allow missing middle initial)."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=sources,
        additional_instruction="Allow 'Roger Wiens' vs 'Roger C. Wiens' as equivalent."
    )
    n = evaluator.add_leaf(
        id="coauthor_purdue_university_affiliation",
        desc="Affiliation is Purdue University, Department of Earth, Atmospheric, and Planetary Sciences.",
        parent=pur_node,
        critical=True
    )
    claim = f"The affiliation stated ('{_safe(purdue.affiliation)}') includes '{GT['coauthor_purdue_affil']}' (Department commonly abbreviated as EAPS)."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=sources,
        additional_instruction="Accept 'Earth, Atmospheric, and Planetary Sciences' or 'EAPS' along with Purdue University."
    )


async def verify_methodology(evaluator: Evaluator, parent_node, ex: MarsLightningExtraction) -> None:
    meth = ex.methodology or MethodologySpecifications()
    bib = ex.bibliography or BibliographicDetails()
    sources = _combine_sources(bib.nature_article_url, bib.doi_url, bib.supporting_urls)

    mnode = evaluator.add_parallel(
        id="methodology_specifications",
        desc="Provide the required methodology specifications.",
        parent=parent_node,
        critical=True
    )

    # Instrument used
    n = evaluator.add_leaf(
        id="methodology_instrument_used",
        desc="Instrument used is the SuperCam microphone aboard NASA's Perseverance rover.",
        parent=mnode,
        critical=True
    )
    claim = (
        f"The instrument stated in the answer ('{_safe(meth.instrument_used)}') is the SuperCam microphone aboard NASA's Perseverance rover."
    )
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=sources,
        additional_instruction="Require both 'SuperCam microphone' and 'Perseverance' (NASA rover) be implied/explicit."
    )

    # Data collection period
    n = evaluator.add_leaf(
        id="methodology_data_collection_period",
        desc="Data collection period spans two Martian years (approximately 1,374 Earth days).",
        parent=mnode,
        critical=True
    )
    claim = (
        f"The data collection period stated in the answer ('{_safe(meth.data_collection_period)}') corresponds to two Martian years (~1,374 Earth days)."
    )
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=sources,
        additional_instruction="Allow 'two Mars years' wording and approximate Earth-days equivalence like ~1374 days."
    )

    # Total recording time analyzed
    n = evaluator.add_leaf(
        id="methodology_total_recording_time_analyzed",
        desc="Total analyzed recording time is 28 hours of microphone recordings.",
        parent=mnode,
        critical=True
    )
    claim = (
        f"The total analyzed recording time stated in the answer ('{_safe(meth.total_recording_time_analyzed)}') corresponds to 28 hours of microphone recordings."
    )
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=sources,
        additional_instruction="Accept '28 h' or similar synonymous representations."
    )

    # Discharge event count
    n = evaluator.add_leaf(
        id="methodology_discharge_event_count",
        desc="Number of detected discharge events is 55 triboelectric discharges.",
        parent=mnode,
        critical=True
    )
    claim = (
        f"The number of detected discharge events stated in the answer ('{_safe(meth.discharge_event_count)}') equals 55 triboelectric discharges."
    )
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=sources,
        additional_instruction="Accept '55 events' or '55 discharges' as equivalent."
    )


async def verify_data_access(evaluator: Evaluator, parent_node, ex: MarsLightningExtraction) -> None:
    data = ex.data_access or DataAccessInformation()
    bib = ex.bibliography or BibliographicDetails()
    sources_repo = _combine_sources(
        bib.nature_article_url, bib.doi_url, data.data_repository_url, data.supercam_acoustic_data_doi, GT["supercam_acoustic_doi_url"]
    )

    dnode = evaluator.add_parallel(
        id="data_access_information",
        desc="Provide primary repository and DOI for accessing the SuperCam acoustic data.",
        parent=parent_node,
        critical=True
    )

    # Primary repository
    n = evaluator.add_leaf(
        id="data_primary_repository",
        desc="Primary data repository is NASA Planetary Data System (PDS).",
        parent=dnode,
        critical=True
    )
    claim = (
        f"The primary data repository stated in the answer ('{_safe(data.primary_data_repository)}') is the NASA Planetary Data System (PDS)."
    )
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=sources_repo,
        additional_instruction="Accept 'NASA PDS' as equivalent to 'NASA Planetary Data System (PDS)'."
    )

    # SuperCam acoustic data DOI
    n = evaluator.add_leaf(
        id="data_supercam_acoustic_data_doi",
        desc=f"SuperCam acoustic data DOI is {GT['supercam_acoustic_doi_url']}.",
        parent=dnode,
        critical=True
    )
    claim = (
        f"The SuperCam acoustic data DOI stated in the answer ('{_safe(data.supercam_acoustic_data_doi)}') equals "
        f"'{GT['supercam_acoustic_doi_plain']}' (with or without the 'https://doi.org/' prefix)."
    )
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=sources_repo,
        additional_instruction="Treat '10.17189/1522646' and 'https://doi.org/10.17189/1522646' as equivalent."
    )


# -----------------------------------------------------------------------------
# Main evaluate function
# -----------------------------------------------------------------------------
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Mars lightning Nature 2025 compilation task.
    """
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_mars_lightning_info(),
        template_class=MarsLightningExtraction,
        extraction_name="extracted_mars_lightning_article_info"
    )

    # Ground truth info for auditing
    evaluator.add_ground_truth({
        "expected": {
            "journal_name": GT["journal_name"],
            "publication_date": [GT["publication_date_preferred"], GT["publication_date_alt"]],
            "volume": GT["volume"],
            "issue": GT["issue"],
            "page_range": GT["page_range"],
            "doi": [GT["doi_plain"], GT["doi_url"]],
            "article_title": GT["article_title"],
            "lead_author": {
                "name": GT["lead_author_name"],
                "affiliation_includes": GT["lead_author_affiliation_core"]
            },
            "key_coauthors": {
                "jhu_apl": {"name": GT["coauthor_apl_name"], "affiliation_includes": GT["coauthor_apl_affil"]},
                "latmos": {"name": GT["coauthor_latmos_name"], "affiliation_includes": GT["coauthor_latmos_affil_core"]},
                "purdue": {"name": GT["coauthor_purdue_name"], "affiliation_includes": GT["coauthor_purdue_affil"]}
            },
            "methodology": {
                "instrument": GT["instrument_used_core"],
                "data_collection_period": GT["data_collection_period_core"],
                "total_recording_hours": GT["total_recording_time_hours"],
                "discharge_events": GT["discharge_event_count"]
            },
            "data_access": {
                "primary_repository": GT["data_repository_name"],
                "supercam_acoustic_doi": [GT["supercam_acoustic_doi_plain"], GT["supercam_acoustic_doi_url"]]
            }
        }
    })

    # Build the rubric tree under a critical compilation node (the task's main requirement)
    compilation_node = evaluator.add_parallel(
        id="mars_lightning_research_compilation",
        desc="Compile all required bibliographic, author, methodology, and data access information for the specified November 2025 Nature article on Mars electrical discharges detected by Perseverance.",
        parent=root,
        critical=True
    )

    # Run verification subtrees
    await verify_bibliographic_details(evaluator, compilation_node, extracted)
    await verify_lead_author(evaluator, compilation_node, extracted)
    await verify_key_coauthors(evaluator, compilation_node, extracted)
    await verify_methodology(evaluator, compilation_node, extracted)
    await verify_data_access(evaluator, compilation_node, extracted)

    return evaluator.get_summary()