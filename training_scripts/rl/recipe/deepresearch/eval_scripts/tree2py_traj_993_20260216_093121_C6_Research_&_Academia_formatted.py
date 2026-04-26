import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ai_research_comparative"
TASK_DESCRIPTION = """Conduct a comparative study of AI/ML researchers by identifying two prominent researchers who have made significant contributions to different areas of modern artificial intelligence. Specifically:

1. Identify one researcher who has made significant contributions to Neural Architecture Search (NAS). This researcher must have published at least one highly-cited paper (with 500 or more citations) in the Neural Architecture Search area, and at least one recent paper (published between 2023-2025) in the same area.

2. Identify one researcher who has made significant contributions to Large Language Models (LLMs). This researcher must have published at least one highly-cited paper (with 1,000 or more citations) in the Large Language Models area, and at least one recent paper (published between 2023-2025) in the same area.

For each researcher, provide:
- Full name
- Link to their Google Scholar profile
- Their h-index value as shown on Google Scholar
- Their current institutional affiliation
- For the highly-cited paper: complete paper title, all author names, publication venue (conference or journal), publication year, current citation count, and a reference URL (Google Scholar page or official publication page)
- For the recent paper: complete paper title, publication venue, publication year (must be 2023, 2024, or 2025), and a reference URL

Additionally, provide:
- A brief comparison highlighting at least one similarity in their research approaches or methodologies
- A brief note on at least one key difference between their contributions to their respective fields
- A short description of one current trend or development in either Neural Architecture Search or Large Language Models research

Ensure all information is verifiable through the provided URLs.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ResearcherProfile(BaseModel):
    name: Optional[str] = None
    scholar_url: Optional[str] = None
    h_index: Optional[str] = None
    affiliation: Optional[str] = None


class PaperRef(BaseModel):
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    venue: Optional[str] = None
    year: Optional[str] = None
    citation_count: Optional[str] = None
    reference_url: Optional[str] = None


class ResearcherBundle(BaseModel):
    profile: Optional[ResearcherProfile] = None
    highly_cited_paper: Optional[PaperRef] = None
    recent_paper: Optional[PaperRef] = None


class StudyExtraction(BaseModel):
    nas: Optional[ResearcherBundle] = None
    llm: Optional[ResearcherBundle] = None
    similarity: Optional[str] = None
    difference: Optional[str] = None
    trend_description: Optional[str] = None
    trend_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_study() -> str:
    return """
Extract the required structured information from the answer text for two researchers: one focused on Neural Architecture Search (NAS) and one focused on Large Language Models (LLMs).

Return a single JSON object with the following structure (use null where information is missing; extract only URLs explicitly present in the answer):

{
  "nas": {
    "profile": {
      "name": "...",
      "scholar_url": "https://scholar.google.com/...",
      "h_index": "...",
      "affiliation": "..."
    },
    "highly_cited_paper": {
      "title": "...",
      "authors": ["Author A", "Author B", "..."],  // list ALL authors named in the answer for this paper
      "venue": "...",                               // conference or journal name
      "year": "YYYY",
      "citation_count": "...",                      // as written in the answer; keep as a string
      "reference_url": "http(s)://..."              // Google Scholar page for the paper OR an official publication page
    },
    "recent_paper": {
      "title": "...",
      "authors": ["..."],                           // if the answer lists authors; otherwise return empty list
      "venue": "...",
      "year": "YYYY",
      "citation_count": null,                       // recent paper citation count is not required; set to null if not provided
      "reference_url": "http(s)://..."
    }
  },
  "llm": {
    "profile": {
      "name": "...",
      "scholar_url": "https://scholar.google.com/...",
      "h_index": "...",
      "affiliation": "..."
    },
    "highly_cited_paper": {
      "title": "...",
      "authors": ["Author A", "Author B", "..."],
      "venue": "...",
      "year": "YYYY",
      "citation_count": "...",
      "reference_url": "http(s)://..."
    },
    "recent_paper": {
      "title": "...",
      "authors": ["..."],
      "venue": "...",
      "year": "YYYY",
      "citation_count": null,
      "reference_url": "http(s)://..."
    }
  },
  "similarity": "One meaningful similarity in their research approaches/methodologies",
  "difference": "One key difference between their contributions",
  "trend_description": "A concise description of one current trend/development in NAS or LLM research",
  "trend_urls": ["http(s)://...", "..."]           // list all supporting URLs cited for the trend; leave empty if none
}

Rules:
- Do not invent any URLs or data; extract exactly what appears in the answer.
- For Google Scholar profiles, extract the full profile URL (citations?user=...).
- Keep numeric-looking fields (e.g., citation counts, years) as strings exactly as written.
- Authors should be listed individually in the 'authors' array when provided.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _digits_only_to_int(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    digits = re.sub(r"[^0-9]", "", value)
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None


def _extract_year(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    m = re.search(r"(20\d{2}|19\d{2})", value)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return _digits_only_to_int(value)


def _year_in_range_2023_2025(value: Optional[str]) -> bool:
    y = _extract_year(value)
    return y in {2023, 2024, 2025}


def _build_paper_metadata_claim(paper: PaperRef, include_authors: bool = True, include_citations: bool = False) -> str:
    parts: List[str] = []
    if paper.title:
        parts.append(f"Title is '{paper.title}'")
    if include_authors and paper.authors:
        parts.append(f"Authors include: {', '.join(paper.authors)}")
    if paper.venue:
        parts.append(f"Venue is '{paper.venue}'")
    if paper.year:
        parts.append(f"Year is {paper.year}")
    if include_citations and paper.citation_count:
        parts.append(f"Citation count is at least {paper.citation_count}")
    if not parts:
        return "The paper metadata matches the provided information."
    return " and ".join(parts) + "."


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_researcher_group(
    evaluator: Evaluator,
    parent_node,
    prefix: str,  # "NAS" or "LLM"
    area_name: str,  # "Neural Architecture Search (NAS)" or "Large Language Models (LLMs)"
    citations_threshold: int,  # 500 or 1000
    bundle: Optional[ResearcherBundle],
) -> None:
    """
    Build verification nodes for one researcher group (NAS or LLM) according to the rubric.
    All children under this group are critical as per the rubric.
    """
    group_node = evaluator.add_parallel(
        id=f"{prefix}_Researcher",
        desc=f"Provide a {prefix} researcher meeting all {prefix}-specific requirements",
        parent=parent_node,
        critical=True
    )

    profile = bundle.profile if bundle and bundle.profile else ResearcherProfile()
    hc = bundle.highly_cited_paper if bundle and bundle.highly_cited_paper else PaperRef()
    recent = bundle.recent_paper if bundle and bundle.recent_paper else PaperRef()

    # ---------------------- Profile ---------------------- #
    profile_node = evaluator.add_parallel(
        id=f"{prefix}_Profile",
        desc=f"Provide required {prefix} researcher profile fields",
        parent=group_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(profile.name and profile.name.strip()),
        id=f"{prefix}_Name_Provided",
        desc="Researcher full name is provided",
        parent=profile_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(profile.scholar_url and profile.scholar_url.strip()),
        id=f"{prefix}_Scholar_URL_Provided",
        desc="Google Scholar profile URL is provided",
        parent=profile_node,
        critical=True
    )

    scholar_profile_leaf = evaluator.add_leaf(
        id=f"{prefix}_Scholar_URL_Is_Profile",
        desc="Provided URL points to a Google Scholar profile page for the researcher",
        parent=profile_node,
        critical=True
    )
    scholar_name_for_claim = profile.name or "the researcher"
    await evaluator.verify(
        claim=f"This webpage is the Google Scholar profile page of '{scholar_name_for_claim}'.",
        node=scholar_profile_leaf,
        sources=profile.scholar_url,
        additional_instruction=(
            "Treat as supported if the page is clearly a Google Scholar profile and shows the researcher's name near the top. "
            "Allow minor name variations (casing, middle initials). If the URL is missing or not a Scholar profile, mark as not supported."
        )
    )

    evaluator.add_custom_node(
        result=bool(profile.h_index and profile.h_index.strip()),
        id=f"{prefix}_HIndex_Provided",
        desc="h-index value is provided",
        parent=profile_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(profile.affiliation and profile.affiliation.strip()),
        id=f"{prefix}_Affiliation_Provided",
        desc="Current institutional affiliation is provided",
        parent=profile_node,
        critical=True
    )

    # ---------------------- Highly Cited Paper ---------------------- #
    hc_node = evaluator.add_parallel(
        id=f"{prefix}_Highly_Cited_Paper",
        desc=f"Provide one highly-cited {prefix} paper with required metadata and thresholds",
        parent=group_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(hc.title and hc.title.strip()),
        id=f"{prefix}_HC_Title_Provided",
        desc="Highly-cited paper complete title is provided",
        parent=hc_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(hc.authors and len(hc.authors) > 0),
        id=f"{prefix}_HC_All_Authors_Provided",
        desc="Highly-cited paper includes all author names",
        parent=hc_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(hc.venue and hc.venue.strip()),
        id=f"{prefix}_HC_Venue_Provided",
        desc="Highly-cited paper publication venue (conference/journal) is provided",
        parent=hc_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(hc.year and hc.year.strip()),
        id=f"{prefix}_HC_Year_Provided",
        desc="Highly-cited paper publication year is provided",
        parent=hc_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(hc.citation_count and hc.citation_count.strip()),
        id=f"{prefix}_HC_Citation_Count_Provided",
        desc="Highly-cited paper current citation count is provided",
        parent=hc_node,
        critical=True
    )

    # Threshold check as custom: rely on metadata-verifiable leaf to ensure URL-grounding separately
    hc_citations_int = _digits_only_to_int(hc.citation_count)
    evaluator.add_custom_node(
        result=bool(hc_citations_int is not None and hc_citations_int >= citations_threshold),
        id=f"{prefix}_HC_Citations_Threshold",
        desc=f"Highly-cited paper citation count is ≥ {citations_threshold}",
        parent=hc_node,
        critical=True
    )

    # Area match for highly-cited paper via URL
    hc_area_leaf = evaluator.add_leaf(
        id=f"{prefix}_HC_Area_Match",
        desc=f"Highly-cited paper is in the {area_name} research area",
        parent=hc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The paper titled '{hc.title or ''}' is in the research area: {area_name}.",
        node=hc_area_leaf,
        sources=hc.reference_url,
        additional_instruction=(
            f"Confirm the paper is about {area_name}. Look for explicit keywords (e.g., "
            f"'neural architecture search', 'NAS' for NAS; 'large language model', 'LLM', 'language model', "
            f"or prominent LLM names for LLMs). Allow synonyms or clear contextual evidence. "
            f"If no reference URL is provided or the page is irrelevant, mark as not supported."
        )
    )

    evaluator.add_custom_node(
        result=bool(hc.reference_url and hc.reference_url.strip()),
        id=f"{prefix}_HC_Reference_URL_Provided",
        desc="Highly-cited paper reference URL is provided (Google Scholar page or official publication page)",
        parent=hc_node,
        critical=True
    )

    # ---------------------- Recent Paper ---------------------- #
    recent_node = evaluator.add_parallel(
        id=f"{prefix}_Recent_Paper",
        desc=f"Provide one recent {prefix} paper (2023–2025) with required metadata",
        parent=group_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(recent.title and recent.title.strip()),
        id=f"{prefix}_Recent_Title_Provided",
        desc="Recent paper complete title is provided",
        parent=recent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(recent.venue and recent.venue.strip()),
        id=f"{prefix}_Recent_Venue_Provided",
        desc="Recent paper publication venue is provided",
        parent=recent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_year_in_range_2023_2025(recent.year),
        id=f"{prefix}_Recent_Year_In_Range",
        desc="Recent paper publication year is 2023, 2024, or 2025",
        parent=recent_node,
        critical=True
    )

    recent_area_leaf = evaluator.add_leaf(
        id=f"{prefix}_Recent_Area_Match",
        desc=f"Recent paper is in the {area_name} research area",
        parent=recent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The paper titled '{recent.title or ''}' is in the research area: {area_name}.",
        node=recent_area_leaf,
        sources=recent.reference_url,
        additional_instruction=(
            f"Confirm the paper is about {area_name}. Look for explicit keywords or clear contextual evidence. "
            f"If no valid URL is provided, mark as not supported."
        )
    )

    evaluator.add_custom_node(
        result=bool(recent.reference_url and recent.reference_url.strip()),
        id=f"{prefix}_Recent_Reference_URL_Provided",
        desc="Recent paper reference URL is provided",
        parent=recent_node,
        critical=True
    )

    # ---------------------- Verifiability of all claims ---------------------- #
    verif_node = evaluator.add_parallel(
        id=f"{prefix}_Verifiability_All_Claims",
        desc=f"All required {prefix} researcher/paper facts are verifiable via the provided URLs",
        parent=group_node,
        critical=True
    )

    # h-index verifiable from Scholar
    hidx_leaf = evaluator.add_leaf(
        id=f"{prefix}_HIndex_Verifiable",
        desc="Provided h-index is verifiable from the provided Google Scholar (or stated reliable) URL",
        parent=verif_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The h-index of {profile.name or 'the researcher'} is {profile.h_index or ''}.",
        node=hidx_leaf,
        sources=profile.scholar_url,
        additional_instruction=(
            "Verify on the Google Scholar profile that the h-index matches. "
            "Allow minor drift (±2) due to updates over time. "
            "If the profile is missing, inaccessible, or shows a clearly different value, mark as not supported."
        )
    )

    # Affiliation verifiable from Scholar (or reliable page)
    aff_leaf = evaluator.add_leaf(
        id=f"{prefix}_Affiliation_Verifiable",
        desc="Provided current affiliation is supported by at least one provided URL (e.g., Google Scholar affiliation field or institutional page)",
        parent=verif_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The current affiliation of {profile.name or 'the researcher'} is '{profile.affiliation or ''}'.",
        node=aff_leaf,
        sources=profile.scholar_url,
        additional_instruction=(
            "Check the affiliation field on the Google Scholar profile (or obvious institutional page if given). "
            "Allow minor formatting differences. Mark as unsupported if the provided URL does not substantiate the affiliation."
        )
    )

    # HC metadata verifiable by provided reference URL
    hc_meta_leaf = evaluator.add_leaf(
        id=f"{prefix}_HC_Metadata_Verifiable",
        desc="Highly-cited paper title/authors/venue/year and citation count are supported by the provided reference URL",
        parent=verif_node,
        critical=True
    )
    await evaluator.verify(
        claim=_build_paper_metadata_claim(hc, include_authors=True, include_citations=True),
        node=hc_meta_leaf,
        sources=hc.reference_url,
        additional_instruction=(
            "Verify that the page supports the stated title, author list (names may be abbreviated but should match), venue, year, "
            "and the citation count (accept if the page shows at least the given count; small drift is acceptable). "
            "If the page does not provide citation information at all, mark this as not supported."
        )
    )

    # Recent metadata verifiable by provided reference URL
    recent_meta_leaf = evaluator.add_leaf(
        id=f"{prefix}_Recent_Metadata_Verifiable",
        desc="Recent paper title/venue/year are supported by the provided reference URL",
        parent=verif_node,
        critical=True
    )
    await evaluator.verify(
        claim=_build_paper_metadata_claim(recent, include_authors=False, include_citations=False),
        node=recent_meta_leaf,
        sources=recent.reference_url,
        additional_instruction=(
            "Verify that the page supports the stated title, venue, and year (allow minor formatting differences). "
            "If the URL is missing or irrelevant, mark as not supported."
        )
    )


async def verify_comparative_analysis(
    evaluator: Evaluator,
    parent_node,
    similarity: Optional[str],
    difference: Optional[str],
) -> None:
    comp_node = evaluator.add_parallel(
        id="Comparative_Analysis",
        desc="Provide required comparison between the two researchers",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(similarity and similarity.strip()),
        id="Similarity_Provided",
        desc="At least one meaningful similarity in research approaches/methodologies is stated",
        parent=comp_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(difference and difference.strip()),
        id="Difference_Provided",
        desc="At least one key difference between their contributions is stated",
        parent=comp_node,
        critical=True
    )


async def verify_research_trend(
    evaluator: Evaluator,
    parent_node,
    trend_description: Optional[str],
    trend_urls: List[str],
) -> None:
    trend_node = evaluator.add_parallel(
        id="Research_Trend",
        desc="Provide one current trend/development in NAS or LLM research",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(trend_description and trend_description.strip()),
        id="Trend_Description_Provided",
        desc="A short description of one current trend/development is provided",
        parent=trend_node,
        critical=True
    )

    trend_support_leaf = evaluator.add_leaf(
        id="Trend_Supported_By_URL",
        desc="At least one reference URL is provided to support the trend/development claim (per 'all information verifiable via provided URLs')",
        parent=trend_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"At least one of the provided URLs supports the trend: '{trend_description or ''}'.",
        node=trend_support_leaf,
        sources=trend_urls if trend_urls else None,
        additional_instruction=(
            "Judge as supported only if at least one provided URL discusses or substantiates the stated trend. "
            "If no URLs are provided, or all URLs are irrelevant/inaccessible, mark as not supported."
        )
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
    Evaluate an answer for the comparative study of NAS/LLM researchers.
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
    extraction = await evaluator.extract(
        prompt=prompt_extract_study(),
        template_class=StudyExtraction,
        extraction_name="study_extraction"
    )

    # Top-level critical aggregation node according to the rubric
    research_task_node = evaluator.add_parallel(
        id="Research_Task",
        desc="Complete comparative study meeting all stated constraints",
        parent=root,
        critical=True
    )

    # NAS researcher group (threshold 500)
    await verify_researcher_group(
        evaluator=evaluator,
        parent_node=research_task_node,
        prefix="NAS",
        area_name="Neural Architecture Search (NAS)",
        citations_threshold=500,
        bundle=extraction.nas
    )

    # LLM researcher group (threshold 1000)
    await verify_researcher_group(
        evaluator=evaluator,
        parent_node=research_task_node,
        prefix="LLM",
        area_name="Large Language Models (LLMs)",
        citations_threshold=1000,
        bundle=extraction.llm
    )

    # Comparative analysis (similarity & difference)
    await verify_comparative_analysis(
        evaluator=evaluator,
        parent_node=research_task_node,
        similarity=extraction.similarity,
        difference=extraction.difference
    )

    # Research trend
    await verify_research_trend(
        evaluator=evaluator,
        parent_node=research_task_node,
        trend_description=extraction.trend_description,
        trend_urls=extraction.trend_urls
    )

    return evaluator.get_summary()