import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cs_researcher_multi_criteria"
TASK_DESCRIPTION = """Identify a computer science researcher who meets all of the following criteria:

1. Institutional Affiliation: Currently affiliated with a university that holds Carnegie Classification R1 designation (Doctoral Universities – Very High Research Activity)
2. Program Ranking: The researcher's institution must have a Computer Science program ranked in the top 20 according to U.S. News & World Report or QS World University Rankings
3. Academic Position: Holds a tenured faculty position as Associate Professor or Full Professor
4. PhD Timeline: Received their PhD between 2005 and 2015 (inclusive)
5. Citation Metrics: According to their Google Scholar profile:
   - h-index of at least 40
   - i10-index of at least 100
   - Total citations of at least 10,000
6. Conference Publications: Has published at least 15 papers at top-tier AI/ML conferences (NeurIPS, ICML, or ICLR) since 2015
7. Journal Publications: Has published at least 5 papers in Q1-ranked journals in their field since 2015
8. Recent Activity: Has published at least 3 papers in 2024 or 2025
9. Research Area: Primary research focus is in artificial intelligence, machine learning, or computer vision
10. International Collaboration: Has co-authored papers with researchers from at least 3 different countries
11. Open Access: At least 30% of their publications are available as open access or preprints (on platforms like arXiv, bioRxiv, or institutional repositories)
12. Current Status: Is currently actively conducting research (not on extended sabbatical or administrative leave as of the most recent publicly available information)

Provide the researcher's name and supporting evidence (URLs) for each criterion.
"""


# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class ResearcherExtraction(BaseModel):
    # Identity and affiliation
    researcher_name: Optional[str] = None
    affiliation: Optional[str] = None
    affiliation_urls: List[str] = Field(default_factory=list)

    # Institutional profile
    r1_urls: List[str] = Field(default_factory=list)
    ranking_urls: List[str] = Field(default_factory=list)

    # Career background
    position_title: Optional[str] = None
    position_urls: List[str] = Field(default_factory=list)
    phd_year: Optional[str] = None
    phd_urls: List[str] = Field(default_factory=list)
    active_urls: List[str] = Field(default_factory=list)

    # Citation impact (Google Scholar)
    scholar_profile_url: Optional[str] = None

    # Publications
    conference_urls: List[str] = Field(default_factory=list)   # NeurIPS/ICML/ICLR since 2015
    journal_q1_urls: List[str] = Field(default_factory=list)   # Q1 journals since 2015
    recent_pub_urls: List[str] = Field(default_factory=list)   # 2024/2025

    # Research characteristics
    area_urls: List[str] = Field(default_factory=list)         # AI/ML/CV focus
    collaboration_urls: List[str] = Field(default_factory=list)  # ≥3 countries
    open_access_urls: List[str] = Field(default_factory=list)  # ≥30% OA/preprints


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_researcher() -> str:
    return """
Extract the researcher and criterion-specific sources mentioned in the answer. Return a single JSON object with the following fields:

- researcher_name: The full name of the identified researcher.
- affiliation: The university or institution the researcher is CURRENTLY affiliated with.
- affiliation_urls: URLs that explicitly confirm the current affiliation (e.g., official faculty page, lab page, CV).
- r1_urls: URLs that explicitly support that the institution has Carnegie Classification R1 (Doctoral Universities – Very High Research Activity). Prefer the official Carnegie Classification page or official institutional statement.
- ranking_urls: URLs that show the institution’s Computer Science program ranked in the top 20 by U.S. News & World Report or QS World University Rankings. Provide the direct ranking page that lists the program and rank.
- position_title: The researcher’s academic title (e.g., Associate Professor, Professor).
- position_urls: URLs that support they hold a tenured Associate or Full Professor role (official faculty profile, CV, or department announcement).
- phd_year: The year the researcher received their PhD (string; if multiple years are mentioned pick the degree year).
- phd_urls: URLs that confirm the PhD year (CV, university bio, Google Scholar bio, DBLP profile, etc.).
- active_urls: URLs indicating the researcher is actively conducting research now (e.g., recent group/news page, personal website updates, recent talks, active lab or open projects page). If none, leave empty.
- scholar_profile_url: The URL to the researcher’s Google Scholar profile (the canonical scholar.google.com citation profile).
- conference_urls: URLs that demonstrate the researcher has at least 15 papers at NeurIPS, ICML, or ICLR since 2015 (e.g., DBLP filtered pages, Google Scholar queries, personal CV sections listing those venues).
- journal_q1_urls: URLs that support at least 5 papers in Q1-ranked journals since 2015 (e.g., CV/publications list plus SJR/JCR pages evidencing Q1 status).
- recent_pub_urls: URLs showing at least 3 papers published in 2024 or 2025 (e.g., DBLP or Scholar filtered list, personal publication list with years).
- area_urls: URLs that show the primary research area is in AI, machine learning, or computer vision (e.g., faculty profile or research statement).
- collaboration_urls: URLs that support the claim that the researcher has co-authored with researchers from at least 3 different countries (e.g., curated CV/publications page that shows affiliations/countries, institutional news summarizing international collaborations).
- open_access_urls: URLs supporting that at least 30% of publications are open access or as preprints (e.g., arXiv profile, publications page with many arXiv links, institutional repository listing).

Rules:
- Only include URLs that are explicitly present in the answer text. Do not invent or infer new URLs.
- Use complete URLs; if a URL is missing protocol, prepend http://
- If any field is absent in the answer, set it to null (for strings) or an empty array (for lists).
- When multiple URLs are provided in the answer for the same criterion, include up to 5 that most directly support the criterion.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if isinstance(u, str):
            uu = u.strip()
            if uu and uu not in seen:
                seen.add(uu)
                out.append(uu)
    return out


def _merge_urls(*lists: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in lists:
        merged.extend(lst or [])
    return _dedup_urls(merged)


# --------------------------------------------------------------------------- #
# Subtree builders                                                            #
# --------------------------------------------------------------------------- #
async def build_institutional_profile(evaluator: Evaluator, parent_node, data: ResearcherExtraction) -> None:
    node = evaluator.add_parallel(
        id="Institutional_Profile",
        desc="Verify institutional affiliation and ranking",
        parent=parent_node,
        critical=True
    )

    # Leaf 1: R1 University Status
    r1_leaf = evaluator.add_leaf(
        id="R1_University_Status",
        desc="Confirm affiliation with Carnegie Classification R1 university with URL reference",
        parent=node,
        critical=True
    )
    r1_claim = f"The university {data.affiliation or 'the researcher’s institution'} is classified as 'Doctoral Universities – Very High Research Activity' (Carnegie R1)."
    r1_sources = _merge_urls(data.r1_urls, [])  # prefer explicit R1 proof
    await evaluator.verify(
        claim=r1_claim,
        node=r1_leaf,
        sources=r1_sources if r1_sources else None,
        additional_instruction="Check that the page explicitly indicates Carnegie R1 or 'Very High Research Activity'. Official Carnegie Classification site or official institutional statements are acceptable."
    )

    # Leaf 2: CS Program Top 20
    top20_leaf = evaluator.add_leaf(
        id="CS_Program_Top20",
        desc="Confirm institution's CS program is ranked in top 20 (U.S. News or QS) with URL reference",
        parent=node,
        critical=True
    )
    top20_claim = f"The Computer Science program at {data.affiliation or 'the institution'} is ranked in the top 20 by either U.S. News & World Report or QS World University Rankings."
    top20_sources = _merge_urls(data.ranking_urls, [])
    await evaluator.verify(
        claim=top20_claim,
        node=top20_leaf,
        sources=top20_sources if top20_sources else None,
        additional_instruction="Verify that the page lists the Computer Science program in the top 20. Accept either U.S. News or QS rankings; ensure it's for CS specifically (not overall institutional ranking)."
    )


async def build_career_background(evaluator: Evaluator, parent_node, data: ResearcherExtraction) -> None:
    node = evaluator.add_parallel(
        id="Career_Background",
        desc="Verify career stage and position",
        parent=parent_node,
        critical=True
    )

    # Leaf 1: Tenured Position (Associate or Full)
    tenured_leaf = evaluator.add_leaf(
        id="Tenured_Position",
        desc="Confirm tenured Associate or Full Professor status with URL reference",
        parent=node,
        critical=True
    )
    tenure_claim = (
        f"The researcher {data.researcher_name or ''} holds a tenured faculty position as an Associate Professor or a Full Professor."
    ).strip()
    tenure_sources = _merge_urls(data.position_urls, data.affiliation_urls)
    await evaluator.verify(
        claim=tenure_claim,
        node=tenured_leaf,
        sources=tenure_sources if tenure_sources else None,
        additional_instruction="Accept titles 'Associate Professor' or 'Professor' on official department or university pages/CV as tenured unless explicitly stated otherwise. Exclude 'Assistant', 'Research Professor', 'Adjunct', or 'Teaching Professor' unless it explicitly mentions tenure."
    )

    # Leaf 2: PhD Timeline 2005–2015 inclusive
    phd_leaf = evaluator.add_leaf(
        id="PhD_Timeline_2005_2015",
        desc="Confirm PhD obtained between 2005-2015 (inclusive) with URL reference",
        parent=node,
        critical=True
    )
    year_txt = data.phd_year or "an appropriate year"
    phd_claim = f"The researcher {data.researcher_name or ''} received their PhD in {year_txt}, which is between 2005 and 2015 inclusive."
    await evaluator.verify(
        claim=phd_claim,
        node=phd_leaf,
        sources=_dedup_urls(data.phd_urls) if data.phd_urls else None,
        additional_instruction="Check the PhD year on the page and confirm it lies within 2005–2015 inclusive. Accept CVs, official bios, or similarly authoritative profiles."
    )

    # Leaf 3: Active Research Status
    active_leaf = evaluator.add_leaf(
        id="Active_Research_Status",
        desc="Confirm currently active in research (not on extended leave) with URL reference",
        parent=node,
        critical=True
    )
    active_claim = (
        f"The researcher {data.researcher_name or ''} is currently actively conducting research and is not on extended sabbatical or administrative leave."
    ).strip()
    active_sources = _merge_urls(data.active_urls, data.recent_pub_urls, [data.scholar_profile_url or ""])
    await evaluator.verify(
        claim=active_claim,
        node=active_leaf,
        sources=active_sources if active_sources else None,
        additional_instruction="Use evidence such as recent publications (within the last 1–2 years), an active lab page with current projects, recent news updates, or similar indicators of ongoing research activity."
    )


async def build_citation_impact(evaluator: Evaluator, parent_node, data: ResearcherExtraction) -> None:
    node = evaluator.add_parallel(
        id="Citation_Impact",
        desc="Verify citation metrics from Google Scholar",
        parent=parent_node,
        critical=True
    )
    scholar_sources = [data.scholar_profile_url] if data.scholar_profile_url else []

    # h-index >= 40
    h_leaf = evaluator.add_leaf(
        id="H_Index_Minimum_40",
        desc="Confirm h-index ≥ 40 on Google Scholar with URL reference",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="According to the researcher's Google Scholar profile, the h-index is at least 40.",
        node=h_leaf,
        sources=scholar_sources if scholar_sources else None,
        additional_instruction="Open the Google Scholar profile and confirm the h-index value is ≥ 40. Use the summary metrics on the profile page."
    )

    # i10-index >= 100
    i10_leaf = evaluator.add_leaf(
        id="I10_Index_Minimum_100",
        desc="Confirm i10-index ≥ 100 on Google Scholar with URL reference",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="According to the researcher's Google Scholar profile, the i10-index is at least 100.",
        node=i10_leaf,
        sources=scholar_sources if scholar_sources else None,
        additional_instruction="Open the Google Scholar profile and confirm the i10-index value is ≥ 100. Use the summary metrics on the profile page."
    )

    # Total citations >= 10,000
    cites_leaf = evaluator.add_leaf(
        id="Total_Citations_10K",
        desc="Confirm total citations ≥ 10,000 on Google Scholar with URL reference",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="According to the researcher's Google Scholar profile, the total number of citations is at least 10,000.",
        node=cites_leaf,
        sources=scholar_sources if scholar_sources else None,
        additional_instruction="Open the Google Scholar profile and check the total citations count (usually the left-most metric). Confirm it is ≥ 10,000."
    )


async def build_publication_portfolio(evaluator: Evaluator, parent_node, data: ResearcherExtraction) -> None:
    node = evaluator.add_parallel(
        id="Publication_Portfolio",
        desc="Verify publication record and output",
        parent=parent_node,
        critical=True
    )

    # Conferences: ≥ 15 papers at NeurIPS/ICML/ICLR since 2015
    conf_leaf = evaluator.add_leaf(
        id="Conference_Papers_15Plus",
        desc="Confirm ≥ 15 papers at NeurIPS/ICML/ICLR since 2015 with URL references",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Since 2015, the researcher has published at least 15 papers at NeurIPS, ICML, or ICLR.",
        node=conf_leaf,
        sources=_dedup_urls(data.conference_urls) if data.conference_urls else None,
        additional_instruction="Check lists (e.g., DBLP or CV) filtered for NeurIPS/ICML/ICLR since 2015 and ensure the count is ≥ 15. Reasonable counting from a single page is sufficient."
    )

    # Journals: ≥ 5 papers in Q1-ranked journals since 2015
    journal_leaf = evaluator.add_leaf(
        id="Journal_Papers_5Plus_Q1",
        desc="Confirm ≥ 5 papers in Q1 journals since 2015 with URL references",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Since 2015, the researcher has published at least 5 papers in Q1-ranked journals in their field.",
        node=journal_leaf,
        sources=_dedup_urls(data.journal_q1_urls) if data.journal_q1_urls else None,
        additional_instruction="Use evidence showing the journals are Q1 (e.g., SJR or JCR pages) and that the researcher has ≥ 5 such papers in those journals since 2015. Accept a curated CV/publication list combined with a Q1 indicator on the page."
    )

    # Recent: ≥ 3 papers in 2024 or 2025
    recent_leaf = evaluator.add_leaf(
        id="Recent_Papers_3Plus",
        desc="Confirm ≥ 3 papers published in 2024 or 2025 with URL references",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="In 2024 or 2025, the researcher has published at least 3 papers.",
        node=recent_leaf,
        sources=_dedup_urls(data.recent_pub_urls) if data.recent_pub_urls else None,
        additional_instruction="Confirm that at least 3 listed publications have year 2024 or 2025 (conference, journal, or preprints). Single-page evidence that lists multiple such items is sufficient."
    )


async def build_research_characteristics(evaluator: Evaluator, parent_node, data: ResearcherExtraction) -> None:
    node = evaluator.add_parallel(
        id="Research_Characteristics",
        desc="Verify research focus, collaboration, and accessibility",
        parent=parent_node,
        critical=True
    )

    # AI/ML/Computer Vision focus
    area_leaf = evaluator.add_leaf(
        id="AI_ML_Vision_Focus",
        desc="Confirm primary research area is AI, machine learning, or computer vision with URL reference",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The researcher's primary research area is in artificial intelligence, machine learning, or computer vision.",
        node=area_leaf,
        sources=_dedup_urls(data.area_urls) if data.area_urls else None,
        additional_instruction="Use an official research statement, faculty profile, or similar authoritative page indicating AI/ML/CV as the primary focus."
    )

    # International collaborations: ≥ 3 different countries
    intl_leaf = evaluator.add_leaf(
        id="International_Collaboration_3Countries",
        desc="Confirm co-authorship with researchers from ≥ 3 different countries with URL references",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The researcher has co-authored papers with researchers from at least three different countries.",
        node=intl_leaf,
        sources=_dedup_urls(data.collaboration_urls) if data.collaboration_urls else None,
        additional_instruction="Accept a single page that reasonably demonstrates collaborations spanning ≥3 countries (e.g., curated CV or publications page that indicates affiliations/countries). If the page explicitly states international collaborations across multiple countries, that suffices."
    )

    # Open Access: ≥ 30% publications OA/preprints
    oa_leaf = evaluator.add_leaf(
        id="Open_Access_30Percent",
        desc="Confirm ≥ 30% of publications available as open access or preprints with URL references",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="At least 30% of the researcher's publications are available as open access or as preprints (e.g., arXiv, institutional repositories).",
        node=oa_leaf,
        sources=_dedup_urls(data.open_access_urls) if data.open_access_urls else None,
        additional_instruction="Accept credible evidence indicating a substantial share (≥30%) of works have OA/preprint links (e.g., arXiv profile with many items, publication list showing numerous arXiv links). A single page summarizing or listing sufficient OA items is acceptable."
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
    Evaluate an answer for the multi-criteria CS researcher identification task.
    """
    # Initialize evaluator
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

    # Extract structured information from the answer
    extracted: ResearcherExtraction = await evaluator.extract(
        prompt=prompt_extract_researcher(),
        template_class=ResearcherExtraction,
        extraction_name="researcher_extraction",
    )

    # Add a critical root node corresponding to the rubric's top-level node
    research_id_node = evaluator.add_parallel(
        id="Researcher_Identification",
        desc="Identify a computer science researcher who meets all specified criteria",
        parent=root,
        critical=True,
    )

    # Store a brief overview for convenience
    evaluator.add_custom_info(
        info={
            "researcher_name": extracted.researcher_name,
            "affiliation": extracted.affiliation,
            "scholar_profile_url": extracted.scholar_profile_url,
        },
        info_type="extracted_overview",
    )

    # Build and verify subtrees according to rubric
    await build_institutional_profile(evaluator, research_id_node, extracted)
    await build_career_background(evaluator, research_id_node, extracted)
    await build_citation_impact(evaluator, research_id_node, extracted)
    await build_publication_portfolio(evaluator, research_id_node, extracted)
    await build_research_characteristics(evaluator, research_id_node, extracted)

    # Return final summary
    return evaluator.get_summary()