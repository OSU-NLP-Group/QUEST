import asyncio
import logging
import re
from datetime import datetime
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "atlas3i_first_paper"
TASK_DESCRIPTION = (
    "On July 1, 2025, astronomers discovered interstellar comet 3I/ATLAS, marking only the third known interstellar "
    "object to pass through our solar system. Identify the first peer-reviewed scientific paper published in a major "
    "astronomical journal that describes the discovery and physical characterization of 3I/ATLAS. Provide the following "
    "information: (1) the paper's title, (2) the journal name where it was published, (3) the name of the first listed "
    "author, (4) the institutional affiliation of the first listed author, (5) the date the paper was accepted by the "
    "journal, and (6) a reference URL (DOI or arXiv identifier) for verification. Additionally, explain how quickly the "
    "paper was published relative to the discovery date."
)

DISCOVERY_DATE = datetime(2025, 7, 1)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SelectedPaper(BaseModel):
    """Structured information for the selected peer-reviewed paper."""
    title: Optional[str] = None
    journal_name: Optional[str] = None
    volume: Optional[str] = None
    pages_or_article_id: Optional[str] = None
    first_author_name: Optional[str] = None
    first_author_affiliation: Optional[str] = None
    acceptance_date: Optional[str] = None
    reference_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)

    # Helpful context fields extracted from the answer
    scope_claim_text: Optional[str] = None
    firstness_justification_text: Optional[str] = None
    speed_explanation_text: Optional[str] = None
    publication_date: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_selected_paper() -> str:
    return """
    You must extract the SINGLE peer-reviewed journal paper that the answer claims is the FIRST peer-reviewed publication
    describing the discovery and physical characterization of interstellar comet 3I/ATLAS.

    Extract the following fields from the answer text. Use null when missing:

    1. title: exact paper title string.
    2. journal_name: exact journal or journal section name (e.g., "Monthly Notices of the Royal Astronomical Society: Letters" or "MNRAS Letters").
    3. volume: journal volume (string).
    4. pages_or_article_id: page range (e.g., "L1–L5") OR article identifier (e.g., "e.g., Article ID L10").
    5. first_author_name: exact name of the first listed author.
    6. first_author_affiliation: institutional affiliation of the first listed author as shown in the paper.
    7. acceptance_date: the date the paper was ACCEPTED by the journal (string as shown).
    8. reference_url: DOI or arXiv identifier URL for verification (e.g., https://doi.org/... or https://arxiv.org/abs/...).
    9. additional_urls: list of any other URLs the answer cites that support the selection or firstness (e.g., journal pages for other candidate papers).
    10. scope_claim_text: the portion of the answer that claims this paper describes the discovery AND physical characterization of 3I/ATLAS.
    11. firstness_justification_text: the portion of the answer explaining why this is the FIRST peer-reviewed paper (e.g., earliest acceptance/publication among peer-reviewed journal papers).
    12. speed_explanation_text: the portion of the answer explaining how quickly the paper was published relative to the July 1, 2025 discovery (e.g., “accepted 12 days later”).
    13. publication_date: the publication date (if provided in the answer), string.

    IMPORTANT:
    - Return exactly what appears in the answer; do not infer or add new information.
    - For URLs, include fully qualified URLs (with protocol).
    - If any field is absent in the answer, set it to null. For additional_urls, return an empty list if none were provided.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12
}


def _parse_date_fuzzy(date_str: Optional[str]) -> Optional[datetime]:
    """Attempt to parse a wide range of common date string formats without external libs."""
    if not date_str:
        return None
    s = date_str.strip()

    # ISO-like YYYY-MM-DD
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception:
            pass

    # YYYY/MM/DD
    m = re.search(r"(\d{4})/(\d{1,2})/(\d{1,2})", s)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception:
            pass

    # Month DD, YYYY
    m = re.search(r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", s)
    if m:
        month = _MONTHS.get(m.group(1).lower())
        if month:
            try:
                return datetime(int(m.group(3)), month, int(m.group(2)))
            except Exception:
                pass

    # DD Month YYYY
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", s)
    if m:
        month = _MONTHS.get(m.group(2).lower())
        if month:
            try:
                return datetime(int(m.group(3)), month, int(m.group(1)))
            except Exception:
                pass

    # Try extracting the first plausible date pattern (e.g., "Accepted: 1 July 2025")
    m = re.search(r"(Accepted|Accepted on|Accepted:)\s*(.+)", s, re.IGNORECASE)
    if m:
        return _parse_date_fuzzy(m.group(2))

    return None


def _compose_sources(paper: SelectedPaper) -> List[str]:
    """Compose a list of source URLs from reference + additional ones."""
    urls: List[str] = []
    if paper.reference_url and paper.reference_url.strip():
        urls.append(paper.reference_url.strip())
    for u in paper.additional_urls:
        if isinstance(u, str) and u.strip():
            urls.append(u.strip())
    return urls


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_selected_paper(evaluator: Evaluator, parent_node, paper: SelectedPaper) -> None:
    """
    Build the verification tree per rubric and perform verifications.
    """

    # Top-level sequential critical node
    top_node = evaluator.add_sequential(
        id="Research_Paper_Identification",
        desc="Identify the first peer-reviewed paper (in the constrained journal) describing discovery and physical characterization of 3I/ATLAS, and provide the required bibliographic, author, and timeline details.",
        parent=parent_node,
        critical=True,
    )

    # ------------------ 1) Eligibility & Firstness (Parallel, Critical) ------------------ #
    elig_node = evaluator.add_parallel(
        id="Paper_Eligibility_and_Firstness",
        desc="Selected paper satisfies the stated constraints and is justified as the first peer-reviewed paper describing the discovery and physical characterization of 3I/ATLAS.",
        parent=top_node,
        critical=True,
    )

    # 1.a Scope matches discovery & characterization
    scope_leaf = evaluator.add_leaf(
        id="Scope_Matches_Discovery_and_Characterization",
        desc="Paper explicitly reports the discovery and physical characterization of interstellar comet 3I/ATLAS.",
        parent=elig_node,
        critical=True,
    )
    scope_claim = (
        "This paper explicitly reports the discovery and physical characterization of interstellar comet 3I/ATLAS."
    )
    await evaluator.verify(
        claim=scope_claim,
        node=scope_leaf,
        sources=_compose_sources(paper),
        additional_instruction=(
            "Verify the paper's abstract, title, or main text states discovery AND physical characterization of 3I/ATLAS. "
            "Allow minor naming variations (e.g., '3I', 'ATLAS', '3I/ATLAS') and minor formatting differences."
        ),
    )

    # 1.b First peer-reviewed paper justification
    firstness_leaf = evaluator.add_leaf(
        id="First_Peer_Reviewed_Paper_Justification",
        desc="Provides verifiable justification that the selected paper is the first peer-reviewed scientific publication describing the discovery/characterization of 3I/ATLAS.",
        parent=elig_node,
        critical=True,
    )
    firstness_claim = (
        "Among peer-reviewed journal publications describing the discovery and/or physical characterization of 3I/ATLAS, "
        "this selected paper is the first (earliest accepted/published) peer-reviewed paper."
    )
    await evaluator.verify(
        claim=firstness_claim,
        node=firstness_leaf,
        sources=_compose_sources(paper),
        additional_instruction=(
            "Judge 'firstness' relative to other peer-reviewed journal papers (ignore non-peer-reviewed preprints unless they map to the same DOI). "
            "Use acceptance or publication dates shown on the provided sources to assess earliest peer-reviewed publication. "
            "If the answer provides comparison URLs, consider them; otherwise, if no explicit comparison exists, be conservative."
        ),
    )

    # ------------------ 2) Required citation & author details (Parallel, Critical) ------------------ #
    citation_node = evaluator.add_parallel(
        id="Required_Citation_and_Author_Details",
        desc="Provide the required title, venue/citation metadata, and first-author details for the selected paper.",
        parent=top_node,
        critical=True,
    )

    # 2.a Paper title
    title_leaf = evaluator.add_leaf(
        id="Paper_Title",
        desc="Provides the paper's exact title.",
        parent=citation_node,
        critical=True,
    )
    title_claim = f"The paper's title is '{paper.title or ''}'."
    await evaluator.verify(
        claim=title_claim,
        node=title_leaf,
        sources=paper.reference_url,
        additional_instruction=(
            "Match the displayed title on the reference page (DOI or journal page). Allow minor punctuation or casing differences."
        ),
    )

    # 2.b Journal name & venue constraint: MNRAS Letters
    journal_leaf = evaluator.add_leaf(
        id="Journal_Name_and_Venue_Constraint",
        desc="Provides the journal name and it matches the constrained venue: Monthly Notices of the Royal Astronomical Society Letters (as specified).",
        parent=citation_node,
        critical=True,
    )
    journal_claim = (
        "The paper was published in 'Monthly Notices of the Royal Astronomical Society: Letters' (MNRAS Letters)."
    )
    await evaluator.verify(
        claim=journal_claim,
        node=journal_leaf,
        sources=paper.reference_url,
        additional_instruction=(
            "Check the journal venue displayed on the identifier page. Accept reasonable variants such as "
            "'MNRAS Letters', 'Monthly Notices of the Royal Astronomical Society Letters', or 'Monthly Notices of the Royal Astronomical Society: Letters'."
        ),
    )

    # 2.c Complete citation metadata (volume + pages or article ID)
    citation_meta_leaf = evaluator.add_leaf(
        id="Complete_Citation_Metadata",
        desc="Provides volume and page range (or article identifier) as part of the complete citation.",
        parent=citation_node,
        critical=True,
    )
    vol_str = paper.volume or ""
    pp_str = paper.pages_or_article_id or ""
    citation_meta_claim = (
        f"The paper's citation includes volume '{vol_str}' and a page range or article identifier '{pp_str}'."
    )
    await evaluator.verify(
        claim=citation_meta_claim,
        node=citation_meta_leaf,
        sources=paper.reference_url,
        additional_instruction=(
            "Confirm that the citation shows a volume and either a page range (e.g., 'L1–L5') or an article identifier. "
            "Pass if the provided values match or reasonably correspond to what's on the journal/DOI page."
        ),
    )

    # 2.d Reference identifier URL (DOI or arXiv)
    ref_leaf = evaluator.add_leaf(
        id="Reference_Identifier_URL",
        desc="Provides a verification reference URL/identifier (DOI or arXiv identifier).",
        parent=citation_node,
        critical=True,
    )
    ref_claim = (
        f"The provided reference URL '{paper.reference_url or ''}' is a DOI or arXiv identifier page for this paper."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=paper.reference_url,
        additional_instruction=(
            "Verify the URL structure (e.g., doi.org/..., arxiv.org/abs/...) and that the page corresponds to the selected paper."
        ),
    )

    # 2.e First listed author name
    author_leaf = evaluator.add_leaf(
        id="First_Listed_Author_Name",
        desc="Correctly identifies the first listed author.",
        parent=citation_node,
        critical=True,
    )
    author_claim = f"The first listed author is '{paper.first_author_name or ''}'."
    await evaluator.verify(
        claim=author_claim,
        node=author_leaf,
        sources=paper.reference_url,
        additional_instruction=(
            "Verify the author list on the identifier page; allow minor formatting differences (e.g., middle initials)."
        ),
    )

    # 2.f First listed author affiliation
    affiliation_leaf = evaluator.add_leaf(
        id="First_Listed_Author_Affiliation",
        desc="Provides the institutional affiliation of the first listed author as shown in the paper.",
        parent=citation_node,
        critical=True,
    )
    affiliation_claim = (
        f"The institutional affiliation of the first listed author is '{paper.first_author_affiliation or ''}'."
    )
    await evaluator.verify(
        claim=affiliation_claim,
        node=affiliation_leaf,
        sources=paper.reference_url,
        additional_instruction=(
            "Use the affiliations shown on the journal/DOI page. If multiple affiliations are listed, accept the first or primary "
            "affiliation stated in the paper."
        ),
    )

    # ------------------ 3) Timeline & speed explanation (Parallel, Critical) ------------------ #
    time_node = evaluator.add_parallel(
        id="Timeline_and_Speed_Explanation",
        desc="Provide acceptance date and explain how quickly the paper was published relative to the discovery date (July 1, 2025).",
        parent=top_node,
        critical=True,
    )

    # 3.a Acceptance date
    acceptance_leaf = evaluator.add_leaf(
        id="Acceptance_Date",
        desc="Provides the date the paper was accepted by the journal.",
        parent=time_node,
        critical=True,
    )
    acceptance_claim = f"The paper was accepted on '{paper.acceptance_date or ''}'."
    await evaluator.verify(
        claim=acceptance_claim,
        node=acceptance_leaf,
        sources=paper.reference_url,
        additional_instruction=(
            "Locate an 'Accepted' date on the journal/DOI page. Confirm the provided acceptance date matches what is shown. "
            "If only 'Received' or 'Published' dates appear, this claim should not pass unless an explicit 'Accepted' date is present."
        ),
    )

    # 3.b Speed relative to discovery
    speed_leaf = evaluator.add_leaf(
        id="Speed_Relative_to_Discovery",
        desc="Explains how quickly relative to July 1, 2025 by quantifying elapsed time using the acceptance date.",
        parent=time_node,
        critical=True,
    )
    # Compute elapsed days if possible
    acc_dt = _parse_date_fuzzy(paper.acceptance_date)
    if acc_dt:
        elapsed_days = (acc_dt - DISCOVERY_DATE).days
        speed_claim = (
            f"The acceptance date is '{paper.acceptance_date}', which is approximately {elapsed_days} days after July 1, 2025. "
            f"The explanation provided ('{paper.speed_explanation_text or ''}') correctly quantifies this elapsed time."
        )
        add_ins = (
            "Verify the arithmetic from the discovery date (July 1, 2025) to the acceptance date. "
            "Allow rounding to the nearest day/week. If the explanation's number matches the computed elapsed time (within reasonable rounding), pass."
        )
    else:
        # If we cannot parse the acceptance date, the check should fail; but still attempt a logical verification
        speed_claim = (
            f"The explanation ('{paper.speed_explanation_text or ''}') correctly quantifies elapsed time relative to July 1, 2025."
        )
        add_ins = (
            "The acceptance date could not be parsed from the provided value; unless a clear, correct quantitative explanation is present, "
            "this check should fail."
        )
    await evaluator.verify(
        claim=speed_claim,
        node=speed_leaf,
        sources=None,  # Pure arithmetic/logical check based on dates extracted from answer; no web evidence needed here
        additional_instruction=add_ins,
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the 3I/ATLAS first paper identification task.
    """
    # Initialize evaluator with sequential root to mirror task flow
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract selected paper info from the answer
    paper_info = await evaluator.extract(
        prompt=prompt_extract_selected_paper(),
        template_class=SelectedPaper,
        extraction_name="selected_paper",
    )

    # Build verification tree and execute checks
    await verify_selected_paper(evaluator, root, paper_info)

    # Return structured summary
    return evaluator.get_summary()