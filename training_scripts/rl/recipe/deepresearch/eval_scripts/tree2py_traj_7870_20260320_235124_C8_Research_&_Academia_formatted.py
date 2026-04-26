import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ai_ml_faculty_diverse_us_universities"
TASK_DESCRIPTION = """
Identify four faculty members currently working in artificial intelligence or machine learning research at four different universities in the United States. Each researcher must meet all of the following criteria:

1. Hold a tenured or tenure-track position (Assistant Professor, Associate Professor, or Full Professor)
2. Have a Google Scholar profile with an h-index of at least 20
3. Work in AI, machine learning, computer vision, or natural language processing as evidenced by their research profile
4. Have published at least one paper at a top-tier AI/ML conference (NeurIPS, ICML, ICLR, CVPR, or AAAI) in 2024 or later
5. Have at least one co-authored publication with a researcher from a different institution published between 2023 and 2026
6. Have a publicly accessible institutional faculty profile page

Additionally, all four researchers must be from different universities—no two researchers can share the same university affiliation.

For each of the four researchers, provide:
- Full name
- Current university affiliation and department
- Faculty rank/title
- Link to their institutional faculty profile page
- Link to their Google Scholar profile
- Current h-index value
- Title and publication year of at least one qualifying conference paper from a top-tier venue (NeurIPS, ICML, ICLR, CVPR, or AAAI) published in 2024 or later
- Title and publication year of at least one qualifying collaborative publication from 2023-2026, including the co-author's name and their institution
"""

ALLOWED_VENUES = ["NeurIPS", "ICML", "ICLR", "CVPR", "AAAI"]
VENUE_SYNONYMS = {
    "NeurIPS": ["Neural Information Processing Systems", "Advances in Neural Information Processing Systems", "NIPS", "NeurIPS"],
    "ICML": ["International Conference on Machine Learning", "ICML"],
    "ICLR": ["International Conference on Learning Representations", "ICLR"],
    "CVPR": ["IEEE/CVF Conference on Computer Vision and Pattern Recognition", "CVPR"],
    "AAAI": ["AAAI Conference on Artificial Intelligence", "AAAI"],
}
CURRENT_YEAR = datetime.utcnow().year
COLLAB_START_YEAR = 2023
COLLAB_END_YEAR = 2026  # Inclusive, per task


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PublicationInfo(BaseModel):
    title: Optional[str] = None
    year: Optional[str] = None
    venue: Optional[str] = None
    url: Optional[str] = None


class CollaborationInfo(BaseModel):
    title: Optional[str] = None
    year: Optional[str] = None
    coauthor_name: Optional[str] = None
    coauthor_institution: Optional[str] = None
    url: Optional[str] = None


class ResearcherRecord(BaseModel):
    name: Optional[str] = None
    university: Optional[str] = None
    department: Optional[str] = None
    rank_title: Optional[str] = None
    institutional_url: Optional[str] = None
    scholar_url: Optional[str] = None
    h_index: Optional[str] = None  # Keep as string for flexible extraction

    conference_publication: Optional[PublicationInfo] = None
    collaboration_publication: Optional[CollaborationInfo] = None


class ResearchersExtraction(BaseModel):
    researchers: List[ResearcherRecord] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_researchers() -> str:
    allowed = ", ".join(ALLOWED_VENUES)
    return f"""
    Extract up to four researchers exactly as presented in the answer. For each, produce a JSON object with:
    - name: full name (string)
    - university: current university affiliation (string)
    - department: current department (string)
    - rank_title: faculty rank/title (string)
    - institutional_url: URL to their institutional faculty profile page (must be a URL if present)
    - scholar_url: URL to their Google Scholar profile (must be a URL if present)
    - h_index: the current h-index value as written in the answer (string; do not convert to number)

    - conference_publication:
        - title: title of one qualifying conference paper (string)
        - year: publication year (string)
        - venue: the venue name (string; should be one of: {allowed}, if available in the answer)
        - url: a URL to the paper page if provided (string or null)

    - collaboration_publication:
        - title: title of one qualifying collaborative publication (string)
        - year: publication year (string)
        - coauthor_name: name of at least one co-author from a different institution (string)
        - coauthor_institution: that co-author's institution (string)
        - url: a URL to the publication or page if provided (string or null)

    Rules:
    - Extract ONLY what appears in the answer. If any field is missing in the answer, set it to null.
    - Do not invent URLs; only extract URLs that actually appear in the answer text.
    - Preserve strings exactly as written (names, titles, venues), except you may trim surrounding whitespace.
    - Return an object with a `researchers` array containing between 1 and 4 researcher objects, in the same order as the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def to_int_safe(s: Optional[str]) -> Optional[int]:
    try:
        if s is None:
            return None
        return int(str(s).strip())
    except Exception:
        # Try to extract digits
        import re
        m = re.search(r"\d{1,4}", str(s))
        if m:
            try:
                return int(m.group(0))
            except Exception:
                return None
        return None


def normalize_university_name(name: Optional[str]) -> Optional[str]:
    if not nonempty(name):
        return None
    return " ".join(name.strip().lower().split())


def within_year_range(year_str: Optional[str], start: int, end: int) -> bool:
    y = to_int_safe(year_str)
    if y is None:
        return False
    return start <= y <= end


def is_venue_allowed(venue: Optional[str]) -> bool:
    if not nonempty(venue):
        return False
    v = venue.strip().lower()
    # Direct check
    for short in ALLOWED_VENUES:
        if v == short.lower():
            return True
    # Synonyms check
    for short, syns in VENUE_SYNONYMS.items():
        for s in syns:
            if v == s.strip().lower():
                return True
    # Partial contains (robustness)
    for short, syns in VENUE_SYNONYMS.items():
        if any(s.strip().lower() in v for s in syns):
            return True
    return False


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_one_researcher(
    evaluator: Evaluator,
    parent_node,
    r: ResearcherRecord,
    idx: int,
) -> None:
    num = idx + 1
    container = evaluator.add_parallel(
        id=f"researcher_{num}",
        desc=f"{['First', 'Second', 'Third', 'Fourth'][idx]} researcher meeting all requirements with complete information",
        parent=parent_node,
        critical=False,  # container non-critical to allow partial credit per researcher
    )

    # -------- Basic info (critical group) --------
    basic = evaluator.add_parallel(
        id=f"r{num}_basic_info",
        desc=f"Basic information for Researcher {num} is provided",
        parent=container,
        critical=True,
    )

    evaluator.add_custom_node(
        result=nonempty(r.name),
        id=f"r{num}_name",
        desc=f"Full name of Researcher {num} is provided",
        parent=basic,
        critical=True,
    )
    evaluator.add_custom_node(
        result=nonempty(r.university),
        id=f"r{num}_university",
        desc=f"Current university affiliation of Researcher {num} is provided",
        parent=basic,
        critical=True,
    )
    evaluator.add_custom_node(
        result=nonempty(r.department),
        id=f"r{num}_department",
        desc=f"Current department of Researcher {num} is provided",
        parent=basic,
        critical=True,
    )
    evaluator.add_custom_node(
        result=nonempty(r.rank_title),
        id=f"r{num}_rank_provided",
        desc=f"Faculty rank/title of Researcher {num} is provided",
        parent=basic,
        critical=True,
    )
    evaluator.add_custom_node(
        result=nonempty(r.institutional_url),
        id=f"r{num}_institutional_url",
        desc=f"Link to Researcher {num}'s institutional faculty profile page is provided",
        parent=basic,
        critical=True,
    )
    evaluator.add_custom_node(
        result=nonempty(r.scholar_url),
        id=f"r{num}_scholar_url",
        desc=f"Link to Researcher {num}'s Google Scholar profile is provided",
        parent=basic,
        critical=True,
    )
    evaluator.add_custom_node(
        result=nonempty(r.h_index),
        id=f"r{num}_hindex_value",
        desc=f"Current h-index value of Researcher {num} is provided",
        parent=basic,
        critical=True,
    )

    # -------- Constraints (critical group) --------
    constraints = evaluator.add_parallel(
        id=f"r{num}_constraints",
        desc=f"Researcher {num} meets all eligibility constraints",
        parent=container,
        critical=True,
    )

    # US university verification
    node_us = evaluator.add_leaf(
        id=f"r{num}_us_university",
        desc=f"Researcher {num} is a current faculty member at a U.S. university",
        parent=constraints,
        critical=True,
    )
    claim_us = f"The institutional page indicates that {r.university or 'the university'} is a university in the United States (U.S.-based)."
    await evaluator.verify(
        claim=claim_us,
        node=node_us,
        sources=r.institutional_url,
        additional_instruction="Use the institutional profile page to check location cues such as .edu domain, address, state/city in the U.S., or explicit mention of 'United States' or U.S. campus.",
    )

    # Field verification (AI/ML/CV/NLP)
    node_field = evaluator.add_leaf(
        id=f"r{num}_field",
        desc=f"Researcher {num} works in AI, ML, computer vision, or NLP as evidenced by their research profile",
        parent=constraints,
        critical=True,
    )
    claim_field = f"{r.name or 'The researcher'} works in AI, machine learning, computer vision, or natural language processing."
    field_sources: List[str] = []
    if nonempty(r.institutional_url):
        field_sources.append(r.institutional_url)  # type: ignore
    if nonempty(r.scholar_url):
        field_sources.append(r.scholar_url)  # type: ignore
    await evaluator.verify(
        claim=claim_field,
        node=node_field,
        sources=field_sources if field_sources else None,
        additional_instruction="Allow synonyms (e.g., deep learning, generative AI, LLMs, RL, perception, transformers). Prefer explicit research interests, group descriptions, or publication topics.",
    )

    # h-index >= 20
    node_hidx = evaluator.add_leaf(
        id=f"r{num}_hindex_threshold",
        desc=f"Researcher {num} has an h-index of at least 20",
        parent=constraints,
        critical=True,
    )
    h_val = to_int_safe(r.h_index)
    claim_h = f"The Google Scholar profile of {r.name or 'the researcher'} shows an h-index of at least 20" + (f" (the answer states h-index={r.h_index})." if nonempty(r.h_index) else ".")
    await evaluator.verify(
        claim=claim_h,
        node=node_hidx,
        sources=r.scholar_url,
        additional_instruction="Check the 'h-index' value displayed on the Google Scholar profile. Minor formatting differences are acceptable. Threshold is ≥ 20.",
    )

    # Tenure-track/tenured rank
    node_rank = evaluator.add_leaf(
        id=f"r{num}_tenure_track",
        desc=f"Researcher {num} holds a tenured or tenure-track position (Assistant, Associate, or Full Professor)",
        parent=constraints,
        critical=True,
    )
    claim_rank = f"On the institutional page, {r.name or 'the researcher'} holds a rank/title that is Assistant Professor, Associate Professor, or (Full) Professor. The listed title is '{r.rank_title or 'N/A'}'."
    await evaluator.verify(
        claim=claim_rank,
        node=node_rank,
        sources=r.institutional_url,
        additional_instruction="Accept variants like 'Assistant Professor', 'Associate Professor', 'Professor', including endowed chair versions. Exclude lecturers, adjuncts, research scientists, and visiting titles.",
    )

    # -------- Conference publication (critical group) --------
    conf_group = evaluator.add_parallel(
        id=f"r{num}_conference_pub",
        desc=f"Conference publication requirement for Researcher {num}",
        parent=container,
        critical=True,
    )

    # Details provided check
    conf_details_provided = evaluator.add_custom_node(
        result=(
            r.conference_publication is not None
            and nonempty(r.conference_publication.title if r.conference_publication else None)
            and nonempty(r.conference_publication.year if r.conference_publication else None)
            and nonempty(r.conference_publication.venue if r.conference_publication else None)
            and is_venue_allowed(r.conference_publication.venue if r.conference_publication else None)
            and (to_int_safe(r.conference_publication.year if r.conference_publication else None) or 0) >= 2024
        ),
        id=f"r{num}_conference_details",
        desc=f"Title and publication year of at least one qualifying conference paper are provided",
        parent=conf_group,
        critical=True,
    )

    # Has qualifying conference paper (verify against sources)
    node_has_conf = evaluator.add_leaf(
        id=f"r{num}_has_conference_paper",
        desc=f"Researcher {num} has published at least one paper in NeurIPS, ICML, ICLR, CVPR, or AAAI in 2024 or later",
        parent=conf_group,
        critical=True,
    )
    conf_title = r.conference_publication.title if r.conference_publication else None
    conf_year = r.conference_publication.year if r.conference_publication else None
    conf_venue = r.conference_publication.venue if r.conference_publication else None
    conf_urls: List[str] = []
    if nonempty(r.scholar_url):
        conf_urls.append(r.scholar_url)  # type: ignore
    if r.conference_publication and nonempty(r.conference_publication.url):
        conf_urls.append(r.conference_publication.url)  # type: ignore

    claim_conf = (
        f"{r.name or 'The researcher'} is an author of a paper titled '{conf_title or 'UNKNOWN'}' "
        f"published in {conf_year or 'UNKNOWN'} at {conf_venue or 'UNKNOWN'}, which is one of the venues: {', '.join(ALLOWED_VENUES)} "
        f"and the year is 2024 or later."
    )
    await evaluator.verify(
        claim=claim_conf,
        node=node_has_conf,
        sources=conf_urls if conf_urls else None,
        additional_instruction=(
            "Confirm authorship and venue on the provided page(s). Accept canonical/synonym forms of the venues: "
            "NeurIPS (NIPS/Neural Information Processing Systems), ICML, ICLR, CVPR, AAAI. "
            "The publication year must be ≥ 2024. Allow minor title variations (capitalization, punctuation). "
            "If the venue/year cannot be confirmed from provided URLs, mark as not supported."
        ),
    )

    # -------- Collaborative publication (critical group) --------
    collab_group = evaluator.add_parallel(
        id=f"r{num}_collab_pub",
        desc=f"Collaborative publication requirement for Researcher {num}",
        parent=container,
        critical=True,
    )

    # Details provided check
    collab_details_ok = evaluator.add_custom_node(
        result=(
            r.collaboration_publication is not None
            and nonempty(r.collaboration_publication.title if r.collaboration_publication else None)
            and nonempty(r.collaboration_publication.year if r.collaboration_publication else None)
            and within_year_range(
                r.collaboration_publication.year if r.collaboration_publication else None,
                COLLAB_START_YEAR,
                COLLAB_END_YEAR
            )
            and nonempty(r.collaboration_publication.coauthor_name if r.collaboration_publication else None)
            and nonempty(r.collaboration_publication.coauthor_institution if r.collaboration_publication else None)
            and (
                normalize_university_name(r.collaboration_publication.coauthor_institution if r.collaboration_publication else None)
                != normalize_university_name(r.university)
            )
        ),
        id=f"r{num}_collab_details",
        desc=f"Title, publication year, co-author's name, and co-author's institution for at least one qualifying collaborative publication are provided",
        parent=collab_group,
        critical=True,
    )

    # Has collaboration publication (verify against sources)
    node_has_collab = evaluator.add_leaf(
        id=f"r{num}_has_collab",
        desc=f"Researcher {num} has at least one co-authored publication with a researcher from a different institution in 2023-2026",
        parent=collab_group,
        critical=True,
    )
    collab_urls: List[str] = []
    if nonempty(r.scholar_url):
        collab_urls.append(r.scholar_url)  # type: ignore
    if r.collaboration_publication and nonempty(r.collaboration_publication.url):
        collab_urls.append(r.collaboration_publication.url)  # type: ignore

    claim_collab = (
        f"{r.name or 'The researcher'} co-authored a publication titled '{(r.collaboration_publication.title if r.collaboration_publication else 'UNKNOWN')}' "
        f"published in {(r.collaboration_publication.year if r.collaboration_publication else 'UNKNOWN')} "
        f"together with {(r.collaboration_publication.coauthor_name if r.collaboration_publication else 'UNKNOWN')}, "
        f"whose institution is {(r.collaboration_publication.coauthor_institution if r.collaboration_publication else 'UNKNOWN')}, "
        f"which is different from {(r.university or 'the researcher’s university')}. "
        f"The year must be between {COLLAB_START_YEAR} and {COLLAB_END_YEAR}, inclusive."
    )
    await evaluator.verify(
        claim=claim_collab,
        node=node_has_collab,
        sources=collab_urls if collab_urls else None,
        additional_instruction=(
            "Use the provided URLs to confirm co-authorship and year (2023–2026 inclusive). "
            "If co-author institutional affiliation is shown, confirm it's different from the researcher's institution. "
            "Allow minor name variants. If evidence is insufficient to confirm different institutions, mark as not supported."
        ),
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
    # Initialize evaluator (root as non-critical to allow partial scoring aggregation)
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

    # Extract structured researchers info
    extraction = await evaluator.extract(
        prompt=prompt_extract_researchers(),
        template_class=ResearchersExtraction,
        extraction_name="researchers_extraction",
    )

    # Keep only the first four researchers; pad with empty if fewer
    researchers: List[ResearcherRecord] = list(extraction.researchers[:4])
    while len(researchers) < 4:
        researchers.append(ResearcherRecord())

    # Ground truth/policy info for context in summary
    evaluator.add_ground_truth({
        "allowed_venues": ALLOWED_VENUES,
        "venue_synonyms": VENUE_SYNONYMS,
        "conference_year_min": 2024,
        "collaboration_year_range": [COLLAB_START_YEAR, COLLAB_END_YEAR],
        "require_us_university": True,
        "tenure_track_ranks": ["Assistant Professor", "Associate Professor", "Professor"],
    })

    # Build researcher subtrees
    for i in range(4):
        await verify_one_researcher(evaluator, root, researchers[i], i)

    # Global constraint: universities must be all different
    uni_names = [
        normalize_university_name(r.university) for r in researchers
        if nonempty(r.university)
    ]
    diversity_ok = (len(uni_names) == 4) and (len(set(uni_names)) == 4)

    evaluator.add_custom_node(
        result=diversity_ok,
        id="university_diversity",
        desc="All four researchers are from different U.S. universities (no two researchers share the same university affiliation)",
        parent=root,
        critical=True,
    )

    # Return structured summary
    return evaluator.get_summary()