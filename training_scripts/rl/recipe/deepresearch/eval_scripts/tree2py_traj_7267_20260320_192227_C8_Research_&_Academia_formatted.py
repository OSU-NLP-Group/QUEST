import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "lunar_eclipse_multi_institution_papers"
TASK_DESCRIPTION = """
Find 4 peer-reviewed astronomy research papers published between 2020 and 2025 that document total lunar eclipse observations conducted through multi-institutional university collaborations (at least 2 universities per paper). For each paper, provide: paper title, journal name, publication year, names and locations of the collaborating universities listed in author affiliations, the observed eclipse's date and time, observation location, confirmation of methodology and results sections, abstract word count (must be 150-250 words), and an accessible URL reference.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Institution(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None


class PaperItem(BaseModel):
    title: Optional[str] = None
    journal: Optional[str] = None
    publication_year: Optional[str] = None
    institutions: List[Institution] = Field(default_factory=list)
    eclipse_datetime: Optional[str] = None  # as presented in the answer (free-form)
    observation_location: Optional[str] = None  # free-form (can be a list serialized or a single string)
    abstract_word_count: Optional[str] = None  # prefer string for compatibility (e.g., "~200")
    url: Optional[str] = None
    supporting_urls: List[str] = Field(default_factory=list)  # any extra links the answer cites (ADS, DOI, journal page, etc.)


class PapersExtraction(BaseModel):
    papers: List[PaperItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_papers() -> str:
    return """
    Extract up to 6 candidate papers mentioned in the answer (we will later evaluate only the first 4).
    For each paper, return the following fields (use null if missing):
    - title: the paper title exactly as stated in the answer
    - journal: the journal name
    - publication_year: 4-digit year string if present (e.g., "2021"); otherwise null
    - institutions: an array of objects, each with:
        * name: the collaborating university/institution name from author affiliations (if available in the answer)
        * location: the location of that institution (city and country or reasonable location string) if mentioned
      Include at least two entries if the answer provides them.
    - eclipse_datetime: the observed eclipse date and time as given in the answer (free-form is OK)
    - observation_location: the geographic observation location(s) as given in the answer (can be a single string summarizing locations)
    - abstract_word_count: the abstract length as a number or a string if approximate (e.g., "200" or "~200")
    - url: the main accessible reference URL to the paper (prefer the publisher or journal landing page; DOI URL is also acceptable)
    - supporting_urls: any additional URLs cited in the answer for this paper (e.g., ADS, DOI, journal overview)

    Rules:
    - Do NOT invent data; only extract what is explicitly present in the answer.
    - Accept URLs in plain or markdown formats; extract actual link targets.
    - If multiple URLs are given, put the most direct article page in 'url' and others in 'supporting_urls'.
    - If some institutions lack locations in the answer, include them with location set to null.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_text(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _normalize_title(t: Optional[str]) -> Optional[str]:
    if not _has_text(t):
        return None
    return " ".join(t.strip().lower().split())


def _normalize_url(u: Optional[str]) -> Optional[str]:
    if not _has_text(u):
        return None
    url = u.strip()
    # strip trailing slash and trivial fragments
    if url.endswith("/"):
        url = url[:-1]
    return url


def build_sources_list(p: PaperItem) -> List[str]:
    urls: List[str] = []
    if _has_text(p.url):
        urls.append(p.url.strip())
    for su in p.supporting_urls or []:
        if _has_text(su):
            urls.append(su.strip())
    # de-duplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        nu = _normalize_url(u) or u
        if nu in seen:
            continue
        seen.add(nu)
        deduped.append(u)
    return deduped


def exactly_4_distinct(papers_all: List[PaperItem]) -> Tuple[bool, Dict[str, Any]]:
    total_reported = len(papers_all)
    # Distinctness by normalized title and normalized main URL; if either collides, treat as duplicate
    titles = []
    urls = []
    for p in papers_all:
        titles.append(_normalize_title(p.title))
        urls.append(_normalize_url(p.url))

    distinct_title_count = len({t for t in titles if t})
    distinct_url_count = len({u for u in urls if u})

    # To be strict: exactly 4 reported AND at least 4 distinct titles AND at least 4 distinct URLs (when present)
    # If some URLs are missing but titles are distinct, still okay provided we have 4 items and 4 distinct titles.
    has_4 = (total_reported == 4)
    # For URLs, only consider non-null URLs. If some are null, rely primarily on titles.
    non_null_urls = [u for u in urls if u]
    distinct_enough = (distinct_title_count == 4) and (len(set(non_null_urls)) == len(non_null_urls))

    result = has_4 and distinct_enough
    debug = {
        "total_reported": total_reported,
        "distinct_title_count": distinct_title_count,
        "non_null_url_count": len(non_null_urls),
        "distinct_non_null_url_count": len(set(non_null_urls)),
        "has_4": has_4,
        "distinct_enough": distinct_enough
    }
    return result, debug


# --------------------------------------------------------------------------- #
# Verification per paper                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_paper(evaluator: Evaluator, parent_node, paper: PaperItem, idx: int) -> None:
    paper_no = idx + 1
    paper_node = evaluator.add_parallel(
        id=f"Paper_{paper_no}",
        desc=f"Paper {paper_no}: if provided, evaluate eligibility constraints and required reported fields.",
        parent=parent_node,
        critical=False
    )

    # Presence checks (custom, critical)
    evaluator.add_custom_node(
        result=_has_text(paper.title),
        id=f"P{paper_no}_Title_Provided",
        desc="Paper title is provided.",
        parent=paper_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_text(paper.journal),
        id=f"P{paper_no}_Journal_Name_Provided",
        desc="Journal name is provided.",
        parent=paper_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_text(paper.publication_year),
        id=f"P{paper_no}_Publication_Year_Provided",
        desc="Publication year is provided.",
        parent=paper_node,
        critical=True
    )

    # Institutions with locations provided (require >=2 with name and some location text)
    has_insts = False
    if paper.institutions and len(paper.institutions) >= 2:
        ok_list = [(_has_text(inst.name) and _has_text(inst.location)) for inst in paper.institutions]
        has_insts = all(ok_list)
    evaluator.add_custom_node(
        result=has_insts,
        id=f"P{paper_no}_List_Collaborating_Institutions_With_Locations",
        desc="Names and locations of the collaborating universities/institutions (from author affiliations) are provided.",
        parent=paper_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_text(paper.eclipse_datetime),
        id=f"P{paper_no}_Eclipse_Observation_Date_And_Time",
        desc="Specific eclipse observation date and time are provided as documented in the paper.",
        parent=paper_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_text(paper.observation_location),
        id=f"P{paper_no}_Observation_Geographic_Location",
        desc="Geographic observation location(s) are provided as documented in the paper.",
        parent=paper_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_text(paper.abstract_word_count),
        id=f"P{paper_no}_Abstract_Word_Count_Provided",
        desc="Abstract word count is provided.",
        parent=paper_node,
        critical=True
    )

    # Accessible URL (verify first so later URL-based checks can auto-skip if this fails)
    p_sources = build_sources_list(paper)

    url_leaf = evaluator.add_leaf(
        id=f"P{paper_no}_Accessible_URL",
        desc="An accessible URL reference to the paper is provided.",
        parent=paper_node,
        critical=True
    )
    url_claim = "This is an accessible article page or landing page for the paper (the page loads and contains article metadata such as title, authors, journal, or abstract)."
    await evaluator.verify(
        claim=url_claim,
        node=url_leaf,
        sources=p_sources,
        additional_instruction="Pass if the URL(s) load and show a recognizable article landing page or PDF (publisher page, DOI resolver, NASA/ADS record linking to the publisher page). Fail if dead link or completely irrelevant."
    )

    # URL-based leaves (evaluate after URL accessibility)
    # 1) Peer-reviewed astronomy/astrophysics journal
    peer_leaf = evaluator.add_leaf(
        id=f"P{paper_no}_Peer_Reviewed_Astro_Journal",
        desc="Paper is published in a peer-reviewed astronomy/astrophysics journal.",
        parent=paper_node,
        critical=True
    )
    jrnl_part = f" named '{paper.journal}'" if _has_text(paper.journal) else ""
    peer_claim = f"The article is published in a peer-reviewed astronomy or astrophysics journal{jrnl_part}."
    peer_ins = (
        "Use only the provided page(s). Accept if the page clearly indicates a reputable astronomy/astrophysics journal "
        "(e.g., ApJ, AJ, A&A, MNRAS, PASP, Icarus, Nature Astronomy, etc.) or if it reasonably implies peer review. "
        "Do not accept arXiv-only preprints without a publisher record."
    )

    # 2) Year between 2020 and 2025 inclusive
    year_range_leaf = evaluator.add_leaf(
        id=f"P{paper_no}_Publication_Year_2020_2025",
        desc="Publication year is between 2020 and 2025 inclusive.",
        parent=paper_node,
        critical=True
    )
    yr_claim = "The paper's publication year shown on this page is between 2020 and 2025 inclusive."
    yr_ins = "Use the publication or online publication year from the page. Ignore submission dates. If unclear, use the most authoritative date on the landing page."

    # 3) Total lunar eclipse observation (not partial/penumbral only)
    tle_leaf = evaluator.add_leaf(
        id=f"P{paper_no}_Total_Lunar_Eclipse_Observation",
        desc="Paper documents observations of a total lunar eclipse (not only partial/penumbral).",
        parent=paper_node,
        critical=True
    )
    tle_claim = "This paper reports observational data for a total lunar eclipse (i.e., totality is part of the observations), not only partial or penumbral phases."
    tle_ins = "Look for phrases like 'total lunar eclipse', 'totality', or equivalent. The focus must be on real observational data, not purely simulation."

    # 4) Affiliations clearly stated
    aff_clear_leaf = evaluator.add_leaf(
        id=f"P{paper_no}_Affiliations_Clearly_Stated",
        desc="Author institutional affiliations are clearly stated in the paper (sufficient to identify institutions).",
        parent=paper_node,
        critical=True
    )
    aff_clear_claim = "The article page displays authors' institutional affiliations clearly enough to identify the institutions."
    aff_clear_ins = "Check for an Affiliations section or footnotes linking authors to universities/research institutes with names visible."

    # 5) Collaboration at least 2 institutions
    collab_leaf = evaluator.add_leaf(
        id=f"P{paper_no}_Collaboration_At_Least_2_Institutions",
        desc="Affiliations indicate collaboration between at least 2 distinct universities/research institutions.",
        parent=paper_node,
        critical=True
    )
    collab_claim = "According to the affiliations on this page, at least two distinct universities or research institutions collaborated on the paper."
    collab_ins = "Count unique institution names. Universities, national observatories, or research labs all count as institutions."

    # 6) Methods/Methodology/Observations section present
    methods_leaf = evaluator.add_leaf(
        id=f"P{paper_no}_Methods_Section_Present",
        desc="Paper includes a distinct methodology/methods section describing observation procedures.",
        parent=paper_node,
        critical=True
    )
    methods_claim = "The paper includes a distinct section describing methodology or observations (e.g., 'Methods', 'Methodology', 'Observations', 'Materials and Methods')."
    methods_ins = "Accept common variants like 'Observations' or 'Materials and Methods' that clearly describe how data were collected."

    # 7) Results section present
    results_leaf = evaluator.add_leaf(
        id=f"P{paper_no}_Results_Section_Present",
        desc="Paper includes a distinct results section presenting observation findings.",
        parent=paper_node,
        critical=True
    )
    results_claim = "The paper includes a distinct 'Results' section (or 'Results and Discussion') presenting observational findings."
    results_ins = "Also accept combined headers like 'Results and Discussion' if they clearly present results."

    # 8) IMRaD or equivalent structure
    imrad_leaf = evaluator.add_leaf(
        id=f"P{paper_no}_IMRaD_Or_Equivalent_Structure",
        desc="Paper follows standard academic research paper structure (IMRaD or equivalent).",
        parent=paper_node,
        critical=True
    )
    imrad_claim = "The article follows a standard scholarly structure such as IMRaD or an equivalent (e.g., Abstract, Introduction, Methods/Observations, Results, Discussion/Conclusion)."
    imrad_ins = "Minor variations are acceptable. The core research sections should be identifiable on the page or PDF."

    # 9) Abstract word count 150–250
    abslen_leaf = evaluator.add_leaf(
        id=f"P{paper_no}_Abstract_Word_Count_150_250",
        desc="Abstract length is between 150 and 250 words inclusive.",
        parent=paper_node,
        critical=True
    )
    abslen_claim = "The abstract on this page has a length between 150 and 250 words inclusive."
    abslen_ins = "Estimate by counting words displayed in the abstract text on the page or PDF. Reasonable counting approximations are acceptable."

    # Batch verify all URL-based leaves (after URL check is done)
    claims_and_sources: List[Tuple[str, List[str], Any, Optional[str]]] = [
        (peer_claim, p_sources, peer_leaf, peer_ins),
        (yr_claim, p_sources, year_range_leaf, yr_ins),
        (tle_claim, p_sources, tle_leaf, tle_ins),
        (aff_clear_claim, p_sources, aff_clear_leaf, aff_clear_ins),
        (collab_claim, p_sources, collab_leaf, collab_ins),
        (methods_claim, p_sources, methods_leaf, methods_ins),
        (results_claim, p_sources, results_leaf, results_ins),
        (imrad_claim, p_sources, imrad_leaf, imrad_ins),
        (abslen_claim, p_sources, abslen_leaf, abslen_ins),
    ]
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

    # Extract papers from answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_papers(),
        template_class=PapersExtraction,
        extraction_name="papers_extraction"
    )

    # Record basic stats
    total_found = len(extracted.papers)
    evaluator.add_custom_info(
        {"extracted_total_papers": total_found},
        info_type="extraction_stats",
        info_name="extraction_stats"
    )

    # We will evaluate only the first 4 papers; pad if fewer
    selected = list(extracted.papers[:4])
    while len(selected) < 4:
        selected.append(PaperItem())

    # Global check: Exactly 4 distinct (non-duplicate) papers are provided (based on original extraction)
    global_node = evaluator.add_custom_node(
        result=exactly_4_distinct(extracted.papers)[0],
        id="Global_Exactly_4_Distinct_Papers",
        desc="Exactly 4 distinct (non-duplicate) papers are provided.",
        parent=root,
        critical=True
    )
    # Add debug info for the global check
    _, distinct_debug = exactly_4_distinct(extracted.papers)
    evaluator.add_custom_info(distinct_debug, info_type="global_check_debug", info_name="global_check_debug")

    # Per-paper verification
    tasks = []
    for i in range(4):
        tasks.append(verify_single_paper(evaluator, root, selected[i], i))
    for t in tasks:
        await t

    return evaluator.get_summary()