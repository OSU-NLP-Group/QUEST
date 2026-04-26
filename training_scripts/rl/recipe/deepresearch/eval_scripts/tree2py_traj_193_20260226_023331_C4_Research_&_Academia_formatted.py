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
TASK_ID = "doaj_top5_university_journal_2025"
TASK_DESCRIPTION = (
    "Identify one peer-reviewed, fully open access academic journal that is published by or affiliated with a university "
    "ranked in the QS World University Rankings 2025 top 5 (MIT, Imperial College London, University of Oxford, Harvard "
    "University, or University of Cambridge), is currently indexed in the Directory of Open Access Journals (DOAJ), publishes "
    "at least 5 research articles per year, has either a publishing history of more than one year as an open access journal "
    "or has published at least 10 open access research articles, has a valid ISSN, and targets researchers or practitioners "
    "as its primary audience."
)

ALLOWED_UNIVERSITIES = [
    "Massachusetts Institute of Technology",
    "MIT",
    "Imperial College London",
    "University of Oxford",
    "Oxford University",
    "Harvard University",
    "University of Cambridge",
    "Cambridge University",
]

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class JournalExtraction(BaseModel):
    # Identification
    journal_name: Optional[str] = None
    journal_homepage_url: Optional[str] = None

    # University affiliation / publisher
    affiliated_university: Optional[str] = None
    publisher_or_affiliation: Optional[str] = None
    affiliation_evidence_urls: List[str] = Field(default_factory=list)

    # DOAJ
    doaj_url: Optional[str] = None

    # ISSNs
    issn_print: Optional[str] = None
    issn_electronic: Optional[str] = None

    # OA model
    open_access_model: Optional[str] = None
    oa_evidence_urls: List[str] = Field(default_factory=list)

    # Peer review
    peer_review_evidence_urls: List[str] = Field(default_factory=list)

    # Audience / aims & scope
    audience_description: Optional[str] = None
    audience_evidence_urls: List[str] = Field(default_factory=list)

    # Publication volume (>=5 research articles per year)
    publication_volume_statement: Optional[str] = None
    publication_volume_evidence_urls: List[str] = Field(default_factory=list)

    # Publication history requirement (OA > 1 year OR >=10 OA research articles)
    publication_history_statement: Optional[str] = None
    publication_history_evidence_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_journal() -> str:
    return """
    You must extract exactly one journal described in the answer that is intended to be: a peer-reviewed, fully open access academic journal indexed in DOAJ and affiliated with a QS 2025 top-5 university (MIT, Imperial College London, University of Oxford, Harvard University, or University of Cambridge).
    If multiple journals are mentioned, extract only the first one that appears to match these criteria. If none clearly match, extract the first journal mentioned in the answer.

    Extract the following fields (use null if missing). For URL fields, extract only explicit URLs from the answer text:
    - journal_name: The journal's name.
    - journal_homepage_url: URL of the journal homepage (if provided).
    - affiliated_university: The stated university the journal is published by or affiliated with (verbatim from the answer).
    - publisher_or_affiliation: Any publisher/affiliation name as mentioned in the answer (verbatim).
    - affiliation_evidence_urls: All URLs cited that substantiate the university affiliation or publisher relationship (journal About page, university site, etc.).
    - doaj_url: The journal's DOAJ record URL if provided in the answer.
    - issn_print: Print ISSN if provided (format ####-####).
    - issn_electronic: Electronic ISSN if provided (format ####-####).
    - open_access_model: Verbatim description of OA model from the answer (e.g., "fully open access").
    - oa_evidence_urls: URLs cited that support OA status (policy page, DOAJ page, etc.).
    - peer_review_evidence_urls: URLs cited that describe the peer review process (e.g., "peer review", "refereed").
    - audience_description: Verbatim text about primary audience from the answer (e.g., "for researchers and practitioners").
    - audience_evidence_urls: URLs cited that show aims & scope, target readership, etc.
    - publication_volume_statement: Verbatim statement regarding the number of research articles per year or counts in a recent year.
    - publication_volume_evidence_urls: URLs cited that show issue archives or article lists supporting >= 5 research articles per year.
    - publication_history_statement: Verbatim statement about OA history length (> 1 year) or total OA research articles published (>= 10).
    - publication_history_evidence_urls: URLs cited that support the publication history requirement.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(*parts: Any) -> List[str]:
    """
    Merge multiple source inputs (string, list of strings, or None) into a deduplicated list of URLs.
    Keeps order of first appearance, filters out empty strings and falsy values.
    """
    seen = set()
    merged: List[str] = []
    for p in parts:
        if not p:
            continue
        if isinstance(p, str):
            candidates = [p]
        elif isinstance(p, list):
            candidates = p
        else:
            continue
        for url in candidates:
            if not url:
                continue
            u = url.strip()
            if not u:
                continue
            # Optionally filter to http/https only
            if not (u.startswith("http://") or u.startswith("https://")):
                # Allow non-protocol URLs from answers per toolkit rule (Extractor may prepend protocol)
                pass
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Tree construction and verification                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_journal_tree(
    evaluator: Evaluator,
    parent_node,
    j: JournalExtraction
) -> None:
    """
    Build the verification nodes according to the rubric and run LLM-backed verifications.
    All leaves here are critical, as each criterion is required.
    """
    # Create the rubric's main critical parallel node
    main_node = evaluator.add_parallel(
        id="Open_Access_Journal_Identification",
        desc="Verify that the identified journal meets all specified criteria for DOAJ indexing and university affiliation",
        parent=parent_node,
        critical=True
    )

    journal_label = j.journal_name or "the journal"
    aff_label = j.affiliated_university or j.publisher_or_affiliation or "the stated institution"

    # 1) University Affiliation
    node_aff = evaluator.add_leaf(
        id="University_Affiliation",
        desc="The journal must be published by or affiliated with a university ranked in the QS World University Rankings 2025 top 5 (MIT, Imperial College London, University of Oxford, Harvard University, or University of Cambridge)",
        parent=main_node,
        critical=True
    )
    claim_aff = f"{journal_label} is published by or affiliated with {aff_label}."
    sources_aff = _merge_sources(j.affiliation_evidence_urls, j.journal_homepage_url, j.doaj_url)
    add_ins_aff = (
        "Only mark Correct if the page(s) explicitly indicate that the journal is published by, hosted by, owned by, "
        "or otherwise affiliated with one of these institutions: MIT, Imperial College London, University of Oxford, "
        "Harvard University, or University of Cambridge (accept reasonable naming variants, e.g., 'Oxford University', 'Cambridge University'). "
        "Do not require verification of QS rankings; you only need to confirm the affiliation is with one of the five listed universities."
    )
    await evaluator.verify(claim=claim_aff, node=node_aff, sources=sources_aff, additional_instruction=add_ins_aff)

    # 2) DOAJ Indexing Status
    node_doaj = evaluator.add_leaf(
        id="DOAJ_Indexing_Status",
        desc="The journal must be currently indexed in the Directory of Open Access Journals (DOAJ)",
        parent=main_node,
        critical=True
    )
    claim_doaj = f"{journal_label} is currently indexed in the Directory of Open Access Journals (DOAJ)."
    sources_doaj = _merge_sources(j.doaj_url)
    add_ins_doaj = (
        "Check the provided URL(s). Only mark Correct if there is an accessible page at doaj.org that corresponds to the journal "
        "(match by journal name and/or ISSN) and represents a current journal record (not just an application page)."
    )
    await evaluator.verify(claim=claim_doaj, node=node_doaj, sources=sources_doaj, additional_instruction=add_ins_doaj)

    # 3) Minimum Annual Publication Volume (>= 5 research articles per year)
    node_volume = evaluator.add_leaf(
        id="Minimum_Annual_Publication_Volume",
        desc="The journal must publish at least 5 research articles per year, as required by DOAJ basic criteria",
        parent=main_node,
        critical=True
    )
    claim_volume = f"{journal_label} publishes at least 5 research articles per year."
    sources_volume = _merge_sources(j.publication_volume_evidence_urls, j.journal_homepage_url, j.doaj_url)
    add_ins_volume = (
        "Use the journal's archive, issues, or article listing pages (and DOAJ if helpful) to determine whether a recent year shows "
        "at least 5 research articles. Treat labels such as 'Article', 'Research Article', 'Original Research' as research articles and "
        "exclude editorials/letters unless clearly identified as research articles. If the evidence reasonably indicates >= 5 in a year, mark Correct."
    )
    await evaluator.verify(claim=claim_volume, node=node_volume, sources=sources_volume, additional_instruction=add_ins_volume)

    # 4) Publication History Requirement (OA > 1 year OR >= 10 OA research articles)
    node_history = evaluator.add_leaf(
        id="Publication_History_Requirement",
        desc="The journal must have either a publishing history of more than one year as an open access journal OR have published at least 10 open access research articles",
        parent=main_node,
        critical=True
    )
    claim_history = (
        f"{journal_label} has either been a fully open access journal for more than one year OR has published at least 10 open access research articles."
    )
    sources_history = _merge_sources(j.publication_history_evidence_urls, j.publication_volume_evidence_urls, j.journal_homepage_url, j.doaj_url)
    add_ins_history = (
        "OR logic applies: mark Correct if EITHER (a) evidence shows the journal has been fully open access for more than one year "
        "based on dated issues/policies; OR (b) the journal has published 10 or more open access research articles (a reasonable count from archives suffices). "
        "If neither is supported, mark Incorrect."
    )
    await evaluator.verify(claim=claim_history, node=node_history, sources=sources_history, additional_instruction=add_ins_history)

    # 5) Open Access Model (fully OA, not hybrid)
    node_oa = evaluator.add_leaf(
        id="Open_Access_Model",
        desc="The journal must be a fully open access publication (not a hybrid or subscription-based journal)",
        parent=main_node,
        critical=True
    )
    claim_oa = f"{journal_label} is fully open access (not a hybrid or subscription journal)."
    sources_oa = _merge_sources(j.oa_evidence_urls, j.doaj_url, j.journal_homepage_url)
    add_ins_oa = (
        "Accept explicit statements such as 'fully open access', 'open access journal', or DOAJ inclusion as strong evidence of being fully OA. "
        "If evidence indicates hybrid or subscription (only some articles OA), mark Incorrect."
    )
    await evaluator.verify(claim=claim_oa, node=node_oa, sources=sources_oa, additional_instruction=add_ins_oa)

    # 6) Peer Review Process
    node_peer = evaluator.add_leaf(
        id="Peer_Review_Process",
        desc="The journal must employ a peer review process for submitted manuscripts",
        parent=main_node,
        critical=True
    )
    claim_peer = f"{journal_label} employs a peer review process for submitted manuscripts."
    sources_peer = _merge_sources(j.peer_review_evidence_urls, j.doaj_url, j.journal_homepage_url)
    add_ins_peer = (
        "Accept language such as 'peer review', 'refereed', 'double-blind peer review', 'single-blind peer review'. "
        "If the evidence only mentions editorial checks without peer review, mark Incorrect."
    )
    await evaluator.verify(claim=claim_peer, node=node_peer, sources=sources_peer, additional_instruction=add_ins_peer)

    # 7) ISSN Verification (valid ISSN exists)
    node_issn = evaluator.add_leaf(
        id="ISSN_Verification",
        desc="The journal must have a valid ISSN (International Standard Serial Number)",
        parent=main_node,
        critical=True
    )
    claim_issn = f"{journal_label} has a valid ISSN."
    sources_issn = _merge_sources(j.doaj_url, j.journal_homepage_url)
    add_ins_issn = (
        "Mark Correct if a valid ISSN-like identifier (format ####-####) is present for the journal on the provided page(s); "
        "eISSN or print ISSN either is acceptable. If no valid ISSN is shown, mark Incorrect."
    )
    await evaluator.verify(claim=claim_issn, node=node_issn, sources=sources_issn, additional_instruction=add_ins_issn)

    # 8) Target Audience (researchers/practitioners)
    node_audience = evaluator.add_leaf(
        id="Target_Audience",
        desc="The journal's primary target audience must be researchers or practitioners in academia",
        parent=main_node,
        critical=True
    )
    claim_audience = f"The primary target audience of {journal_label} is researchers or practitioners."
    sources_audience = _merge_sources(j.audience_evidence_urls, j.journal_homepage_url, j.doaj_url)
    add_ins_audience = (
        "Use Aims & Scope, About, or submission information to judge audience. Accept terms like 'researchers', 'scholars', 'academics', "
        "'practitioners'. If the page suggests a general-population readership without focus on researchers/practitioners, mark Incorrect."
    )
    await evaluator.verify(claim=claim_audience, node=node_audience, sources=sources_audience, additional_instruction=add_ins_audience)


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
    Evaluate an answer for the 'DOAJ + QS Top-5 University Journal' identification task.
    """
    # Initialize evaluator/root (root is non-critical by default; we create a critical child node per rubric)
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

    # Extract structured journal info from the answer
    journal_info = await evaluator.extract(
        prompt=prompt_extract_journal(),
        template_class=JournalExtraction,
        extraction_name="journal_extraction",
    )

    # Add ground-truth/context info for transparency
    evaluator.add_ground_truth({
        "allowed_universities_QS2025_top5": [
            "MIT",
            "Imperial College London",
            "University of Oxford",
            "Harvard University",
            "University of Cambridge",
        ]
    })

    # Optional: record a concise snapshot of the extracted journal for debugging
    evaluator.add_custom_info(
        info={
            "journal_name": journal_info.journal_name,
            "journal_homepage_url": journal_info.journal_homepage_url,
            "affiliated_university": journal_info.affiliated_university or journal_info.publisher_or_affiliation,
            "doaj_url": journal_info.doaj_url,
            "issn_print": journal_info.issn_print,
            "issn_electronic": journal_info.issn_electronic,
        },
        info_type="extracted_overview",
        info_name="extracted_journal_overview"
    )

    # Build verification tree and run checks
    await build_and_verify_journal_tree(evaluator, root, journal_info)

    # Return the evaluation summary
    return evaluator.get_summary()