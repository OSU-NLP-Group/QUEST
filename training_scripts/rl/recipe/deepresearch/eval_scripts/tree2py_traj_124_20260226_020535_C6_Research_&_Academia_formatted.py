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
TASK_ID = "3i_atlas_special_telescopes_2026"
TASK_DESCRIPTION = """
Identify research papers about interstellar comet 3I/ATLAS that involved special space telescope observations conducted during the week of January 15-22, 2026. These papers should meet the following criteria:

1. The observations were conducted by either the Hubble Space Telescope or the Transiting Exoplanet Survey Satellite (TESS)
2. For Hubble observations: they occurred specifically on January 22, 2026, during a rare near-opposition alignment where the comet was positioned within 0.69 degrees of the Sun-Earth axis, and the lead observer was affiliated with Shanghai Astronomical Observatory
3. For TESS observations: they occurred during January 15-22, 2026, as part of the specially designated Sector 1751 observation that temporarily interrupted the regular Sector 99 observations
4. The research is relevant to or was presented at the 247th American Astronomical Society meeting held in Phoenix, Arizona during January 4-8, 2026
5. The papers are published in or submitted to peer-reviewed astronomical journals

For each qualifying paper, provide:
- The paper title or description of the research
- The space telescope platform used (Hubble or TESS)
- The specific observation dates
- The publication venue or journal
- A reference URL to the paper or announcement
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PaperItem(BaseModel):
    """Single paper/research item extracted from the answer."""
    title_or_description: Optional[str] = None
    platform: Optional[str] = None  # Expected values (normalized by the judge): "Hubble" or "TESS"
    observation_dates: List[str] = Field(default_factory=list)  # Keep raw strings (ranges OK)
    publication_venue: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    aas_relevance: Optional[str] = None  # e.g., note of AAS 247 presentation or relevance
    peer_review_status: Optional[str] = None  # e.g., "submitted to ApJ", "published in AJ", etc.


class PapersExtraction(BaseModel):
    """Top-level extraction of up to 5 candidate papers."""
    papers: List[PaperItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_papers() -> str:
    return """
    Extract up to the first five candidate research papers/items mentioned in the answer that relate to interstellar comet 3I/ATLAS and special telescope observations during January 15-22, 2026.

    For each item, extract the following fields exactly as stated in the answer:
    - title_or_description: The paper title or a clear description of the research (string; may be a descriptive sentence).
    - platform: The telescope platform stated for the observations (string). Use the exact wording in the answer (e.g., "Hubble Space Telescope", "HST", "TESS"). If unspecified, return null.
    - observation_dates: All specific observation date(s) given (array of strings). Include ranges like "Jan 15–22, 2026" or specific dates like "January 22, 2026". If unspecified, return an empty array.
    - publication_venue: The publication or submission venue/journal (string). If unspecified, return null.
    - reference_urls: All reference URLs related to this paper or announcement (array of strings). Extract only valid URLs explicitly present in the answer. If none are provided, return an empty array.
    - aas_relevance: Any text indicating relevance to or presentation at the 247th AAS meeting in Phoenix, AZ (string). If not mentioned, return null.
    - peer_review_status: Any text indicating peer-review status (e.g., published in ApJ, submitted to AJ, under review in MNRAS) (string). If not mentioned, return null.

    IMPORTANT:
    - Do not invent any information; extract only what's present in the answer.
    - For URLs: include full URLs (if missing protocol, prepend "http://").
    - If more than five items are in the answer, only include the first five.
    - If fewer than five items are present, include all available.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_platform(platform: Optional[str]) -> str:
    """Normalize platform string to 'Hubble', 'TESS', or 'Unknown' for conditional checks."""
    if not platform:
        return "Unknown"
    p = platform.strip().lower()
    if "hubble" in p or "hst" in p:
        return "Hubble"
    if "tess" in p:
        return "TESS"
    return "Unknown"


def has_any_candidate(papers: List[PaperItem]) -> bool:
    """Check if there is at least one usable candidate item in the extracted list."""
    for p in papers:
        if (p.title_or_description and p.title_or_description.strip()) or (p.reference_urls and len(p.reference_urls) > 0):
            return True
    return False


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_required_output_fields(
    evaluator: Evaluator,
    parent_node,
    paper: PaperItem,
    idx: int
) -> None:
    """
    Build and evaluate 'Required_Output_Fields' group:
    - Title or description provided
    - Telescope platform stated
    - Specific observation dates provided
    - Publication venue provided
    - Reference URL provided
    """
    fields_node = evaluator.add_parallel(
        id=f"paper_{idx+1}_required_fields",
        desc=f"Provides all required metadata fields for paper #{idx+1}.",
        parent=parent_node,
        critical=True
    )

    # Title/Description presence (critical)
    evaluator.add_custom_node(
        result=bool(paper.title_or_description and paper.title_or_description.strip()),
        id=f"paper_{idx+1}_title_or_description",
        desc=f"Paper #{idx+1}: Title or research description is provided",
        parent=fields_node,
        critical=True
    )

    # Platform presence (critical)
    evaluator.add_custom_node(
        result=bool(paper.platform and paper.platform.strip()),
        id=f"paper_{idx+1}_platform_present",
        desc=f"Paper #{idx+1}: Telescope platform stated (Hubble or TESS)",
        parent=fields_node,
        critical=True
    )

    # Observation dates presence (critical)
    evaluator.add_custom_node(
        result=bool(paper.observation_dates and len(paper.observation_dates) > 0),
        id=f"paper_{idx+1}_observation_dates_present",
        desc=f"Paper #{idx+1}: Specific observation date(s) provided",
        parent=fields_node,
        critical=True
    )

    # Publication venue presence (critical)
    evaluator.add_custom_node(
        result=bool(paper.publication_venue and paper.publication_venue.strip()),
        id=f"paper_{idx+1}_publication_venue_present",
        desc=f"Paper #{idx+1}: Publication/submission venue provided",
        parent=fields_node,
        critical=True
    )

    # Reference URL presence (critical)
    evaluator.add_custom_node(
        result=bool(paper.reference_urls and len(paper.reference_urls) > 0),
        id=f"paper_{idx+1}_reference_url_present",
        desc=f"Paper #{idx+1}: Reference URL provided",
        parent=fields_node,
        critical=True
    )


async def build_qualifying_criteria(
    evaluator: Evaluator,
    parent_node,
    paper: PaperItem,
    idx: int
) -> None:
    """
    Build and evaluate 'Qualifying_Criteria' group:
    - Subject is 3I/ATLAS
    - Telescope is Hubble or TESS
    - AAS 247 relevance
    - Peer-reviewed status
    - Hubble constraints (if applicable)
    - TESS constraints (if applicable)
    """
    qualify_node = evaluator.add_parallel(
        id=f"paper_{idx+1}_qualify",
        desc=f"Paper #{idx+1} meets qualifying constraints (topic, telescope, dates, AAS relevance, peer-review, platform-specific constraints)",
        parent=parent_node,
        critical=True
    )

    # Subject is 3I/ATLAS (critical)
    subject_node = evaluator.add_leaf(
        id=f"paper_{idx+1}_subject_3i_atlas",
        desc=f"Paper #{idx+1}: Pertains to interstellar comet 3I/ATLAS",
        parent=qualify_node,
        critical=True
    )
    await evaluator.verify(
        claim="This research pertains to interstellar comet 3I/ATLAS (allow reasonable alternate designations if clearly referring to the same object).",
        node=subject_node,
        sources=paper.reference_urls,
        additional_instruction="Check the page(s) for mentions of '3I/ATLAS' or clear equivalent references to the same interstellar comet. Allow minor naming variations and context."
    )

    # Telescope is Hubble or TESS (critical)
    telescope_node = evaluator.add_leaf(
        id=f"paper_{idx+1}_telescope_hubble_or_tess",
        desc=f"Paper #{idx+1}: Involves observations conducted by Hubble or TESS",
        parent=qualify_node,
        critical=True
    )
    await evaluator.verify(
        claim="This research involved observations conducted by either the Hubble Space Telescope (HST) or the Transiting Exoplanet Survey Satellite (TESS).",
        node=telescope_node,
        sources=paper.reference_urls,
        additional_instruction="Look for evidence on the referenced page(s) indicating the use of HST/Hubble or TESS in the observations. Do not accept other instruments."
    )

    # AAS 247 relevance (critical)
    aas_node = evaluator.add_leaf(
        id=f"paper_{idx+1}_aas_247_relevance",
        desc=f"Paper #{idx+1}: Relevant to or presented at the 247th AAS meeting (Phoenix, Jan 4–8, 2026)",
        parent=qualify_node,
        critical=True
    )
    await evaluator.verify(
        claim="The research is relevant to or was presented at the 247th American Astronomical Society meeting in Phoenix, Arizona (January 4–8, 2026).",
        node=aas_node,
        sources=paper.reference_urls,
        additional_instruction="Accept clear mentions of being presented at AAS 247 (Phoenix, Jan 4–8, 2026), listed in the program, or explicitly tied to that meeting."
    )

    # Peer-reviewed status (critical)
    peer_node = evaluator.add_leaf(
        id=f"paper_{idx+1}_peer_review_status",
        desc=f"Paper #{idx+1}: Published in or submitted to a peer-reviewed astronomical journal",
        parent=qualify_node,
        critical=True
    )
    await evaluator.verify(
        claim="This paper is published in or submitted to a peer-reviewed astronomical journal.",
        node=peer_node,
        sources=paper.reference_urls,
        additional_instruction="Consider journals such as ApJ, AJ, A&A, MNRAS, Icarus, PASP, Nature Astronomy, etc. If an arXiv page explicitly states 'submitted to' or 'accepted by' a peer-reviewed journal, treat it as meeting the criterion."
    )

    # Hubble-specific constraints (critical; pass as N/A if platform is not Hubble)
    hubble_node = evaluator.add_parallel(
        id=f"paper_{idx+1}_hubble_constraints",
        desc=f"Paper #{idx+1}: Hubble-specific constraints (applicable only if platform is Hubble)",
        parent=qualify_node,
        critical=True
    )
    platform_norm = normalize_platform(paper.platform)

    # Hubble Date Jan 22, 2026
    hubble_date_node = evaluator.add_leaf(
        id=f"paper_{idx+1}_hubble_date_jan22",
        desc=f"Paper #{idx+1}: Hubble observations occurred on January 22, 2026",
        parent=hubble_node,
        critical=True
    )
    await evaluator.verify(
        claim="If the platform is Hubble, the observations occurred on January 22, 2026; otherwise, this check is not applicable.",
        node=hubble_date_node,
        sources=paper.reference_urls,
        additional_instruction=f"Platform for this item: {platform_norm}. If not Hubble, return Correct (N/A). If Hubble, verify evidence for the specific date Jan 22, 2026."
    )

    # Hubble near-opposition within 0.69 degrees
    hubble_opposition_node = evaluator.add_leaf(
        id=f"paper_{idx+1}_hubble_near_opposition_0p69",
        desc=f"Paper #{idx+1}: Captured near-opposition alignment within 0.69 degrees of Sun–Earth axis",
        parent=hubble_node,
        critical=True
    )
    await evaluator.verify(
        claim="If the platform is Hubble, the observation captured a near-opposition alignment within 0.69 degrees of the Sun–Earth axis; otherwise, this check is not applicable.",
        node=hubble_opposition_node,
        sources=paper.reference_urls,
        additional_instruction=f"Platform for this item: {platform_norm}. If not Hubble, return Correct (N/A). If Hubble, verify explicit mention of near-opposition and ~0.69° alignment."
    )

    # Hubble lead observer affiliated with Shanghai Astronomical Observatory
    hubble_lead_node = evaluator.add_leaf(
        id=f"paper_{idx+1}_hubble_lead_shanghai_ao",
        desc=f"Paper #{idx+1}: Lead observer/PI affiliated with Shanghai Astronomical Observatory",
        parent=hubble_node,
        critical=True
    )
    await evaluator.verify(
        claim="If the platform is Hubble, the lead observer or PI is affiliated with the Shanghai Astronomical Observatory; otherwise, this check is not applicable.",
        node=hubble_lead_node,
        sources=paper.reference_urls,
        additional_instruction=f"Platform for this item: {platform_norm}. If not Hubble, return Correct (N/A). If Hubble, look for 'Shanghai Astronomical Observatory' in PI/lead observer affiliation."
    )

    # TESS-specific constraints (critical; pass as N/A if platform is not TESS)
    tess_node = evaluator.add_parallel(
        id=f"paper_{idx+1}_tess_constraints",
        desc=f"Paper #{idx+1}: TESS-specific constraints (applicable only if platform is TESS)",
        parent=qualify_node,
        critical=True
    )

    # TESS window Jan 15–22, 2026
    tess_window_node = evaluator.add_leaf(
        id=f"paper_{idx+1}_tess_window_jan15_22_2026",
        desc=f"Paper #{idx+1}: TESS observation window Jan 15–22, 2026",
        parent=tess_node,
        critical=True
    )
    await evaluator.verify(
        claim="If the platform is TESS, the observations occurred within January 15–22, 2026; otherwise, this check is not applicable.",
        node=tess_window_node,
        sources=paper.reference_urls,
        additional_instruction=f"Platform for this item: {platform_norm}. If not TESS, return Correct (N/A). If TESS, verify the window overlaps Jan 15–22, 2026."
    )

    # TESS Sector 1751 interrupts Sector 99
    tess_sector_node = evaluator.add_leaf(
        id=f"paper_{idx+1}_tess_sector1751_interrupts_99",
        desc=f"Paper #{idx+1}: Designated Sector 1751 temporarily interrupted regular Sector 99",
        parent=tess_node,
        critical=True
    )
    await evaluator.verify(
        claim="If the platform is TESS, the observation was designated as Sector 1751 and temporarily interrupted regular Sector 99; otherwise, this check is not applicable.",
        node=tess_sector_node,
        sources=paper.reference_urls,
        additional_instruction=f"Platform for this item: {platform_norm}. If not TESS, return Correct (N/A). If TESS, look for explicit mention of 'Sector 1751' and interruption of 'Sector 99'."
    )


async def verify_single_paper(
    evaluator: Evaluator,
    root_node,
    paper: PaperItem,
    idx: int
) -> None:
    """
    Build the tree for a single paper (Paper_i) with:
    - Qualifying_Criteria (critical)
    - Required_Output_Fields (critical)
    """
    paper_node = evaluator.add_parallel(
        id=f"paper_{idx+1}",
        desc=f"Evaluation of the {idx+1}st listed paper/research item" if idx == 0 else (
            f"Evaluation of the {idx+1}nd listed paper/research item" if idx == 1 else (
                f"Evaluation of the {idx+1}rd listed paper/research item" if idx == 2 else
                f"Evaluation of the {idx+1}th listed paper/research item"
            )
        ),
        parent=root_node,
        critical=False  # Allow partial credit across papers
    )

    # Build Qualifying Criteria (critical group)
    await build_qualifying_criteria(evaluator, paper_node, paper, idx)

    # Build Required Output Fields (critical group)
    await build_required_output_fields(evaluator, paper_node, paper, idx)


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the 3I/ATLAS special space telescope observation papers task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates multiple paper checks independently
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

    # Extract candidate papers from the answer
    extraction: PapersExtraction = await evaluator.extract(
        prompt=prompt_extract_papers(),
        template_class=PapersExtraction,
        extraction_name="papers_extraction"
    )

    # Enforce evaluation up to first 5 papers
    papers: List[PaperItem] = list(extraction.papers[:5])
    while len(papers) < 5:
        papers.append(PaperItem())

    # Critical check: At least one candidate item provided
    evaluator.add_custom_node(
        result=has_any_candidate(papers),
        id="at_least_one_paper_provided",
        desc="Response includes at least one candidate paper/research item.",
        parent=root,
        critical=True
    )

    # Build verification subtrees for up to 5 papers
    for idx in range(5):
        await verify_single_paper(evaluator, root, papers[idx], idx)

    # Return structured evaluation summary
    return evaluator.get_summary()