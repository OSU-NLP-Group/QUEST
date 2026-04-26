import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "lpsc_3I_atlas_2026"
TASK_DESCRIPTION = """
You are a planetary scientist preparing to attend the 57th Lunar and Planetary Science Conference (LPSC 2026). Provide the following information:

Part 1: Conference Details
Verify and provide the following details about LPSC 2026:
- The exact dates of the conference (start date to end date)
- The full name and address of the conference venue
- The abstract submission deadline date(s)
- A link to the official LPSC 2026 conference website

Part 2: Research Papers on 3I/ATLAS
Identify four distinct research papers or preprints that report observations or analysis of the interstellar comet 3I/ATLAS. The observations or data analysis in each paper must have been conducted between July 2025 and February 2026.

For each of the four papers, provide:
1. The complete paper title
2. At least one author's full name
3. The institutional affiliation of at least one author (must be a university or research institute)
4. Whether the paper is published in a journal or available as a preprint
5. A direct URL link to the paper or preprint
6. Confirmation that the observations or analysis were conducted between July 2025 and February 2026

Additional Requirements:
- All four papers must be distinct from each other (different titles, different primary research focus, or different author teams)
- At least one of the four papers must report observations conducted using space-based telescopes or spacecraft instruments
- At least one of the four papers must report the detection of specific chemical compounds or molecules in 3I/ATLAS
"""


# --------------------------- Data Models ---------------------------------- #
class LPSCDetails(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None
    abstract_deadlines: List[str] = Field(default_factory=list)
    conference_url: Optional[str] = None


class PaperInfo(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    institution: Optional[str] = None
    publication_type: Optional[str] = None  # e.g., "journal", "preprint"
    url: Optional[str] = None
    observation_dates_text: Optional[str] = None  # Free-text dates mentioned in answer


class PapersExtraction(BaseModel):
    papers: List[PaperInfo] = Field(default_factory=list)


# ------------------------ Extraction Prompts ------------------------------ #
def prompt_extract_conference_details() -> str:
    return """
    Extract the LPSC 2026 conference details provided in the answer.

    Return a JSON object with the following fields:
    - start_date: The start date of the conference (as written in the answer, e.g., "March 16, 2026")
    - end_date: The end date of the conference (e.g., "March 20, 2026")
    - venue_name: The full venue name (e.g., "The Woodlands Waterway Marriott Hotel and Convention Center")
    - venue_address: The venue address or city/state text (e.g., "The Woodlands, Texas" or a full street address)
    - abstract_deadlines: An array of deadline date(s) mentioned for abstract submission (include both original and extended if present)
    - conference_url: A single valid URL that the answer claims is the official LPSC 2026 conference website

    Rules:
    - Extract exactly what appears in the answer; do not invent or change formats.
    - If any field is not present in the answer, return null for that field (or an empty array for abstract_deadlines).
    - For the conference_url, include only a single URL, prioritize the one labeled as official.
    """


def prompt_extract_papers() -> str:
    return """
    Extract up to FOUR distinct research papers or preprints reported in the answer that are about interstellar comet 3I/ATLAS.

    For each paper, return an array 'papers' of objects with fields:
    - title: The complete paper or preprint title
    - author: At least one author's full name (as provided in the answer)
    - institution: The institutional affiliation of at least one author (university or research institute)
    - publication_type: Either "journal" or "preprint" (as stated in the answer; if unclear, return null)
    - url: A direct URL link to the paper or preprint
    - observation_dates_text: Any dates or date ranges in the answer indicating when observations or analysis were conducted (free text)

    Rules:
    - Extract only details explicitly present in the answer; do not infer missing fields.
    - If the answer lists more than 4 papers, include only the first 4.
    - If fewer than 4 papers are present, return only those available.
    - Ensure titles are distinct strings; do not merge items.
    """


# --------------------------- Helper Utilities ----------------------------- #
def normalize_text(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def is_title_distinct(current_title: Optional[str], previous_titles: List[Optional[str]]) -> bool:
    cur = normalize_text(current_title)
    prev_norms = [normalize_text(t) for t in previous_titles]
    return cur != "" and cur not in prev_norms


def join_deadlines(deadlines: List[str]) -> str:
    if not deadlines:
        return ""
    return "; ".join(d.strip() for d in deadlines if d and d.strip())


# ------------------------- Verification Functions ------------------------- #
async def verify_conference_section(evaluator: Evaluator, root_node, details: LPSCDetails) -> None:
    conf_node = evaluator.add_parallel(
        id="conference_verification",
        desc="Verify the 57th Lunar and Planetary Science Conference (LPSC 2026) details",
        parent=root_node,
        critical=True  # All child checks must pass
    )

    # Leaf: Official conference URL provided and valid
    url_node = evaluator.add_leaf(
        id="conference_url",
        desc="A valid URL to the official LPSC 2026 conference website is provided",
        parent=conf_node,
        critical=True
    )
    await evaluator.verify(
        claim="This URL is an official page for the 57th Lunar and Planetary Science Conference (LPSC 2026).",
        node=url_node,
        sources=details.conference_url,
        additional_instruction="The page should be under USRA/LPI domains (e.g., hou.usra.edu or lpi.usra.edu) and clearly reference LPSC 2026."
    )

    # Leaf: Conference dates
    dates_node = evaluator.add_leaf(
        id="conference_dates",
        desc="Conference is scheduled for March 16-20, 2026",
        parent=conf_node,
        critical=True
    )
    if details.start_date and details.end_date:
        dates_claim = f"The 57th Lunar and Planetary Science Conference (LPSC 2026) is scheduled from {details.start_date} to {details.end_date}."
    else:
        # Fallback to explicit ground truth phrasing to allow verification, but still tied to official URL
        dates_claim = "The 57th Lunar and Planetary Science Conference (LPSC 2026) is scheduled for March 16–20, 2026."
    await evaluator.verify(
        claim=dates_claim,
        node=dates_node,
        sources=details.conference_url,
        additional_instruction="Check the official LPSC 2026 website for the exact conference dates; allow minor formatting variations (en dash vs 'to', etc.)."
    )

    # Leaf: Conference location (venue full name and address)
    location_node = evaluator.add_leaf(
        id="conference_location",
        desc="Conference venue is The Woodlands Waterway Marriott Hotel and Convention Center in The Woodlands, Texas",
        parent=conf_node,
        critical=True
    )
    if details.venue_name and details.venue_address:
        location_claim = f"The LPSC 2026 venue is {details.venue_name} located in {details.venue_address}."
    else:
        location_claim = "The LPSC 2026 venue is The Woodlands Waterway Marriott Hotel and Convention Center in The Woodlands, Texas."
    await evaluator.verify(
        claim=location_claim,
        node=location_node,
        sources=details.conference_url,
        additional_instruction="Confirm that the venue name and location match the official conference site information."
    )

    # Leaf: Abstract deadline (allow original or extended)
    deadline_node = evaluator.add_leaf(
        id="abstract_deadline",
        desc="Abstract submission deadline was January 6, 2026 (or the extended deadline of January 8, 2026)",
        parent=conf_node,
        critical=True
    )
    deadlines_text = join_deadlines(details.abstract_deadlines)
    if deadlines_text:
        deadline_claim = f"The abstract submission deadline(s) for LPSC 2026, as provided in the answer ({deadlines_text}), are correct according to the official site (January 6, 2026, and/or the extended January 8, 2026)."
    else:
        deadline_claim = "The abstract submission deadline for LPSC 2026 was January 6, 2026, with an extended deadline on January 8, 2026."
    await evaluator.verify(
        claim=deadline_claim,
        node=deadline_node,
        sources=details.conference_url,
        additional_instruction="Verify the abstract submission deadline(s); accept either January 6, 2026 (original) or January 8, 2026 (extended)."
    )


async def verify_single_paper(
    evaluator: Evaluator,
    parent_node,
    paper: PaperInfo,
    idx: int,
    previous_titles: List[Optional[str]],
) -> Dict[str, Any]:
    """
    Verify a single paper with structured sub-checks.
    Returns dict with references to optional instrument/detection leaf nodes for aggregate checks.
    """
    paper_node = evaluator.add_parallel(
        id=f"paper_{idx+1}",
        desc=f"{['First','Second','Third','Fourth'][idx]} research paper about 3I/ATLAS observations",
        parent=parent_node,
        critical=False  # Allow partial credit within each paper
    )

    # Basic info group
    basic_node = evaluator.add_parallel(
        id=f"paper_{idx+1}_basic_info",
        desc=f"Basic identification information for Paper {idx+1}",
        parent=paper_node,
        critical=True
    )

    # Title exists (existence check)
    title_exists = bool(paper.title and paper.title.strip())
    evaluator.add_custom_node(
        result=title_exists,
        id=f"paper_{idx+1}_title",
        desc=f"A paper title related to 3I/ATLAS observations or analysis is provided",
        parent=basic_node,
        critical=True
    )

    # Publication type acceptable (journal or preprint) — source-backed
    publication_node = evaluator.add_leaf(
        id=f"paper_{idx+1}_publication",
        desc=f"The paper is published in a journal or available as a preprint",
        parent=basic_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page indicates the work is a peer‑reviewed journal article or a preprint (e.g., arXiv, ResearchSquare, OSF Preprints, etc.).",
        node=publication_node,
        sources=paper.url,
        additional_instruction="Accept 'Journal', 'Article', or preprint indications on recognized servers; the page must be the actual paper or preprint page."
    )

    # URL points to paper/preprint about 3I/ATLAS
    url_node = evaluator.add_leaf(
        id=f"paper_{idx+1}_url",
        desc=f"A valid URL to the paper or preprint is provided",
        parent=basic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This URL leads directly to the paper or preprint reporting observations or analysis of the interstellar comet 3I/ATLAS: '{paper.title or ''}'.",
        node=url_node,
        sources=paper.url,
        additional_instruction="The page should mention '3I/ATLAS', 'ATLAS', or designation of the interstellar comet; generic news or blog pages should fail."
    )

    # Authorship group
    auth_node = evaluator.add_parallel(
        id=f"paper_{idx+1}_authorship",
        desc=f"Author and institutional information for Paper {idx+1}",
        parent=paper_node,
        critical=True
    )

    # Author names check
    author_node = evaluator.add_leaf(
        id=f"paper_{idx+1}_author_names",
        desc=f"At least one author's full name is identified",
        parent=auth_node,
        critical=True
    )
    if paper.author and paper.author.strip():
        author_claim = f"The paper page lists at least one author named '{paper.author}'."
    else:
        author_claim = "The paper page lists at least one author by full name."
    await evaluator.verify(
        claim=author_claim,
        node=author_node,
        sources=paper.url,
        additional_instruction="Allow fuzzy matching and minor variations (middle initials, accents)."
    )

    # Institution check (must be university or research institute)
    inst_node = evaluator.add_leaf(
        id=f"paper_{idx+1}_institution",
        desc=f"At least one author's institutional affiliation (university or research institute) is identified",
        parent=auth_node,
        critical=True
    )
    if paper.institution and paper.institution.strip():
        inst_claim = f"At least one author is affiliated with '{paper.institution}', which is a university or research institute."
    else:
        inst_claim = "At least one author is affiliated with a recognized university or research institute."
    await evaluator.verify(
        claim=inst_claim,
        node=inst_node,
        sources=paper.url,
        additional_instruction="Check author affiliation blocks, footnotes, or PDF metadata; observatories or national labs count as research institutes."
    )

    # Observation period check
    period_node = evaluator.add_leaf(
        id=f"paper_{idx+1}_observation_period",
        desc=f"The paper's observations or data analysis were conducted between July 2025 and February 2026",
        parent=paper_node,
        critical=True
    )
    await evaluator.verify(
        claim="This paper reports that observations or data analysis were conducted between July 2025 and February 2026.",
        node=period_node,
        sources=paper.url,
        additional_instruction="Check observation sections, dates in text/figures, or methods; accept any date within 2025‑07‑01 to 2026‑02‑28."
    )

    # Distinctness from previous papers (for paper 2-4)
    distinct_result = True
    if idx >= 1:
        distinct_result = is_title_distinct(paper.title, previous_titles[:idx])
        evaluator.add_custom_node(
            result=distinct_result,
            id=f"paper_{idx+1}_distinctness",
            desc=f"This paper is distinct from previous papers (different title, primary focus, or author team)",
            parent=paper_node,
            critical=True
        )

    # Optional checks used for aggregate requirements (non‑critical)
    space_based_node = evaluator.add_leaf(
        id=f"paper_{idx+1}_space_based",
        desc=f"This paper reports observations using space‑based telescopes or spacecraft instruments",
        parent=paper_node,
        critical=False
    )
    await evaluator.verify(
        claim="This paper reports observations using space‑based telescopes or spacecraft instruments (e.g., HST, JWST, NEOWISE, TESS, Gaia, Chandra, Swift, etc.).",
        node=space_based_node,
        sources=paper.url,
        additional_instruction="Look for instrument names known to be space‑based; if only ground‑based observatories are present, this should fail."
    )

    chemical_node = evaluator.add_leaf(
        id=f"paper_{idx+1}_chemical_detection",
        desc=f"This paper reports detection of specific chemical compounds or molecules in 3I/ATLAS",
        parent=paper_node,
        critical=False
    )
    await evaluator.verify(
        claim="This paper reports detection or measurement of specific chemical compounds or molecules in 3I/ATLAS (e.g., CN, C2, NH, CO, H2O, OH, CO2, HCN, NH2).",
        node=chemical_node,
        sources=paper.url,
        additional_instruction="Check spectra, compositional analysis, or lines; general mentions without detection should fail."
    )

    return {
        "space_based_node": space_based_node,
        "chemical_node": chemical_node,
    }


# ----------------------- Main Evaluation Entry Point ---------------------- #
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
        strategy=AggregationStrategy.PARALLEL,  # Root allows parallel sub‑sections
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

    # Extract conference details and paper list from the answer
    conference_details = await evaluator.extract(
        prompt=prompt_extract_conference_details(),
        template_class=LPSCDetails,
        extraction_name="conference_details",
    )
    papers_extraction = await evaluator.extract(
        prompt=prompt_extract_papers(),
        template_class=PapersExtraction,
        extraction_name="papers_extraction",
    )

    # Record ground truth hints
    evaluator.add_ground_truth({
        "conference_dates_expected": "March 16–20, 2026",
        "venue_expected": "The Woodlands Waterway Marriott Hotel and Convention Center, The Woodlands, Texas",
        "abstract_deadline_expected": ["January 6, 2026", "January 8, 2026 (extended)"],
        "observation_window_expected": "Between July 2025 and February 2026",
    })

    # Build conference verification subtree
    await verify_conference_section(evaluator, root, conference_details)

    # Research papers verification subtree
    papers_root = evaluator.add_parallel(
        id="research_papers",
        desc="Identify four distinct research papers or preprints about 3I/ATLAS observations with required characteristics",
        parent=root,
        critical=False  # Allow partial credit across papers
    )

    # Prepare up to 4 papers
    papers: List[PaperInfo] = (papers_extraction.papers or [])[:4]
    while len(papers) < 4:
        papers.append(PaperInfo())

    # Verify each paper and track instrument/detection nodes for aggregate requirements
    space_nodes = []
    chem_nodes = []
    prev_titles = [p.title for p in papers]  # full list for distinctness comparisons

    for idx, paper in enumerate(papers):
        result_refs = await verify_single_paper(
            evaluator=evaluator,
            parent_node=papers_root,
            paper=paper,
            idx=idx,
            previous_titles=prev_titles
        )
        space_nodes.append(result_refs["space_based_node"])
        chem_nodes.append(result_refs["chemical_node"])

    # Aggregate requirements: at least one space‑based; at least one chemical detection
    space_pass = any(n.status == "passed" for n in space_nodes)
    chem_pass = any(n.status == "passed" for n in chem_nodes)

    evaluator.add_custom_node(
        result=space_pass,
        id="aggregate_space_based_requirement",
        desc="At least one of the four papers reports observations using space-based telescopes or spacecraft instruments",
        parent=papers_root,
        critical=True
    )
    evaluator.add_custom_node(
        result=chem_pass,
        id="aggregate_chemical_detection_requirement",
        desc="At least one of the four papers reports detection of specific chemical compounds or molecules in 3I/ATLAS",
        parent=papers_root,
        critical=True
    )

    # Final summary
    return evaluator.get_summary()