import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task Constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "university_research_center_2026"
TASK_DESCRIPTION = (
    "Identify a university research center that meets ALL of the following criteria:\n"
    "1) The research center must be located at a university that ranks in the top 100 in either the QS World University Rankings 2026 or the Times Higher Education World University Rankings 2026.\n"
    "2) The research center must focus on interdisciplinary research (at least two distinct disciplines).\n"
    "3) The research center must have a director who: holds a doctoral degree, has >= 20 peer-reviewed articles, and h-index >= 15.\n"
    "4) The research center must have an active postdoctoral fellowship program (or currently host postdocs), with duration between 1–5 years or a 5-year maximum policy.\n"
    "5) The research center must have received at least one external research grant of $500,000 or more within 2020–2025.\n"
    "Provide: university name; research center name; director name and title; evidence of director’s qualifications; postdoctoral program info; details of at least one qualifying external grant; and URLs supporting all claims."
)

QUALIFYING_GRANT_YEARS_DESC = "2020–2025 (inclusive)"


# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class ResearchCenterExtraction(BaseModel):
    # University level
    university_name: Optional[str] = None
    ranking_system: Optional[str] = None  # e.g., "QS", "THE", "QS & THE"
    ranking_url: Optional[str] = None

    # Center identification
    center_name: Optional[str] = None
    center_official_url: Optional[str] = None

    # Interdisciplinary requirement
    interdisciplinary_disciplines: List[str] = Field(default_factory=list)
    interdisciplinary_desc: Optional[str] = None
    interdisciplinary_urls: List[str] = Field(default_factory=list)

    # Director and qualifications
    director_name: Optional[str] = None
    director_title: Optional[str] = None
    director_degree: Optional[str] = None
    director_publications: Optional[str] = None  # keep as text (e.g., "over 50 publications")
    director_h_index: Optional[str] = None       # keep as text (e.g., "h-index 22")
    director_urls: List[str] = Field(default_factory=list)

    # Postdoc program
    postdoc_exists: Optional[bool] = None  # True if answer explicitly states existence/hosting of postdocs
    postdoc_duration_text: Optional[str] = None
    postdoc_urls: List[str] = Field(default_factory=list)

    # External grant
    grant_amount: Optional[str] = None     # textual amount mentioned (e.g., "$1.2M", "£600,000")
    grant_year: Optional[str] = None       # textual date or year
    grant_funding_source: Optional[str] = None
    grant_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_center_info() -> str:
    return (
        "Extract exactly one university research center that the answer claims satisfies all constraints. "
        "If multiple are mentioned, select the first one that appears complete. "
        "Return the following fields (use null if absent):\n"
        "- university_name: string\n"
        "- ranking_system: which ranking source is claimed (e.g., 'QS', 'THE', 'QS & THE')\n"
        "- ranking_url: a URL that directly supports the university's top-100 status in QS 2026 or THE 2026\n"
        "- center_name: string (research center official name)\n"
        "- center_official_url: URL to the center's official page or authoritative page\n"
        "- interdisciplinary_disciplines: array of distinct disciplines (e.g., ['Computer Science','Biology']); must list 2+ if claimed\n"
        "- interdisciplinary_desc: short text explaining the interdisciplinary focus\n"
        "- interdisciplinary_urls: array of URLs supporting interdisciplinary focus; include center page if it supports this\n"
        "- director_name: string\n"
        "- director_title: string (role/title indicating director/lead)\n"
        "- director_degree: string (e.g., 'PhD in Physics', 'DPhil')\n"
        "- director_publications: string describing publication count (e.g., '35 peer-reviewed papers', 'over 100 publications')\n"
        "- director_h_index: string describing h-index (e.g., 'h-index 18')\n"
        "- director_urls: array of URLs supporting director role/degree/metrics (e.g., center people page, Google Scholar, Scopus)\n"
        "- postdoc_exists: boolean (true if the center has an active postdoc program or hosts postdocs; false if explicitly stated otherwise; null if unclear)\n"
        "- postdoc_duration_text: text describing duration/policy (e.g., '1–3 years', 'up to 5 years')\n"
        "- postdoc_urls: array of URLs supporting postdoc existence and duration (center page acceptable if information is there)\n"
        "- grant_amount: string (e.g., '$750,000', 'USD 0.6M')\n"
        "- grant_year: string with award date/year (e.g., '2022', 'June 2023')\n"
        "- grant_funding_source: string (e.g., 'NSF', 'NIH', 'UKRI', 'Horizon Europe')\n"
        "- grant_urls: array of URLs supporting the grant details\n"
        "Notes:\n"
        "• Only extract URLs explicitly present in the answer.\n"
        "• ranking_url should point to a page showing QS 2026 or THE 2026 ranking status for the university.\n"
        "• If the answer uses multiple URLs for one item, include them all in the corresponding array.\n"
    )


# --------------------------------------------------------------------------- #
# Helper Utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _merge_urls(*url_lists: List[str], also: Optional[List[str]] = None) -> List[str]:
    seen = set()
    combined: List[str] = []
    for lst in url_lists:
        for u in lst:
            if _nonempty(u) and u not in seen:
                seen.add(u)
                combined.append(u)
    if also:
        for u in also:
            if _nonempty(u) and u not in seen:
                seen.add(u)
                combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Build Verification Tree                                                     #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, ext: ResearchCenterExtraction) -> None:
    # Root (sequential): enforce order – University -> Center & constraints
    root = evaluator.add_sequential(
        id="root_task",
        desc="Identify one university research center and verify all constraints with supporting URLs",
        parent=evaluator.root,
        critical=True
    )

    # ---------------- University Verification (parallel, critical) ----------------
    univ_node = evaluator.add_parallel(
        id="university_verification",
        desc="Verify the chosen university satisfies the ranking constraint and is properly cited",
        parent=root,
        critical=True
    )

    # University Name (existence)
    evaluator.add_custom_node(
        result=_nonempty(ext.university_name),
        id="university_name",
        desc="Provide the name of the university",
        parent=univ_node,
        critical=True
    )

    # University Ranking URL (existence)
    evaluator.add_custom_node(
        result=_nonempty(ext.ranking_url),
        id="university_ranking_url",
        desc="Provide a URL reference confirming the university's top-100 status in QS 2026 or THE 2026",
        parent=univ_node,
        critical=True
    )

    # University Top-100 Verification
    univ_rank_leaf = evaluator.add_leaf(
        id="university_top100_2026",
        desc="University is ranked in the top 100 in either QS 2026 or THE 2026",
        parent=univ_node,
        critical=True
    )
    univ_name = ext.university_name or "the university"
    ranking_source = (ext.ranking_system or "QS or THE").strip()
    claim_univ_top100 = (
        f"This page shows that {univ_name} is ranked within the top 100 in the 2026 {ranking_source} World University Rankings."
    )
    await evaluator.verify(
        claim=claim_univ_top100,
        node=univ_rank_leaf,
        sources=ext.ranking_url,
        additional_instruction=(
            "Confirm the page is for QS 2026 or THE 2026 global rankings. "
            "A rank number of 1–100 (inclusive) qualifies. Allow reasonable naming variations for the university."
        )
    )

    # ---------------- Research Center Verification (sequential, critical) ----------------
    center_root = evaluator.add_sequential(
        id="research_center_verification",
        desc="Identify a research center at the university and verify all center/director/postdoc/grant constraints with URLs",
        parent=root,
        critical=True
    )

    # Center Identification (parallel)
    center_ident = evaluator.add_parallel(
        id="center_identification",
        desc="Provide the center identity and official reference",
        parent=center_root,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(ext.center_name),
        id="center_name",
        desc="Provide the official name of the research center",
        parent=center_ident,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(ext.center_official_url),
        id="center_official_url",
        desc="Provide a URL reference to the research center’s official page (or equivalent authoritative page)",
        parent=center_ident,
        critical=True
    )

    # Center Constraints Check (parallel)
    center_checks = evaluator.add_parallel(
        id="center_constraints_check",
        desc="Verify all required constraints for the center, director, postdoc program, and grant",
        parent=center_root,
        critical=True
    )

    # Interdisciplinary Focus
    interdisc_urls = _merge_urls(ext.interdisciplinary_urls, also=[ext.center_official_url] if _nonempty(ext.center_official_url) else [])
    evaluator.add_custom_node(
        result=len(interdisc_urls) > 0,
        id="interdisciplinary_focus_url_exists",
        desc="Provide URL reference supporting the interdisciplinary focus and the involvement of at least two disciplines",
        parent=center_checks,
        critical=True
    )
    interdisc_leaf = evaluator.add_leaf(
        id="interdisciplinary_focus",
        desc="Center focuses on interdisciplinary research involving at least two distinct academic disciplines",
        parent=center_checks,
        critical=True
    )
    disciplines_list = ", ".join([d for d in ext.interdisciplinary_disciplines if _nonempty(d)]) or "multiple distinct fields"
    claim_interdisc = (
        f"The research center '{ext.center_name or 'the center'}' conducts interdisciplinary research involving at least two distinct academic disciplines. "
        f"The disciplines mentioned include: {disciplines_list}."
    )
    await evaluator.verify(
        claim=claim_interdisc,
        node=interdisc_leaf,
        sources=interdisc_urls,
        additional_instruction=(
            "Pass if the page(s) clearly indicate an interdisciplinary mission or activities explicitly spanning 2 or more distinct fields. "
            "Allow synonyms and near-synonyms for disciplines."
        )
    )

    # Director Qualifications (parallel)
    director_checks = evaluator.add_parallel(
        id="director_qualifications",
        desc="Verify the center director identity and required qualifications with supporting evidence",
        parent=center_checks,
        critical=True
    )

    # Director URLs existence
    evaluator.add_custom_node(
        result=len(ext.director_urls) > 0,
        id="director_qualifications_urls",
        desc="Provide URL references supporting the director’s role, degree, publication count, and h-index",
        parent=director_checks,
        critical=True
    )
    director_sources = _merge_urls(ext.director_urls, also=[ext.center_official_url] if _nonempty(ext.center_official_url) else [])

    # Director Name and Title
    director_role_leaf = evaluator.add_leaf(
        id="director_name_and_title",
        desc="Provide the director’s name and title/role as director of the center",
        parent=director_checks,
        critical=True
    )
    claim_director_role = (
        f"According to the cited page(s), {ext.director_name or 'the individual'} serves as the director (or equivalent leadership title) of the research center '{ext.center_name or 'the center'}'."
    )
    await evaluator.verify(
        claim=claim_director_role,
        node=director_role_leaf,
        sources=director_sources,
        additional_instruction="Titles like Director, Executive Director, Founding Director, or Center Head qualify if they clearly indicate leadership of the center."
    )

    # Director Doctoral Degree
    director_degree_leaf = evaluator.add_leaf(
        id="director_phd_or_equivalent",
        desc="Director holds a doctoral degree (PhD or equivalent)",
        parent=director_checks,
        critical=True
    )
    claim_director_degree = (
        f"The director {ext.director_name or ''} holds a doctoral degree (e.g., PhD, DPhil, ScD, MD/PhD or equivalent)."
    )
    await evaluator.verify(
        claim=claim_director_degree,
        node=director_degree_leaf,
        sources=director_sources,
        additional_instruction="Accept standard doctoral equivalents (e.g., PhD, DPhil, ScD, EngD)."
    )

    # Director Publications >= 20
    director_pubs_leaf = evaluator.add_leaf(
        id="director_publications_ge_20",
        desc="Director has published at least 20 peer-reviewed articles",
        parent=director_checks,
        critical=True
    )
    claim_director_pubs = (
        f"The director {ext.director_name or ''} has published at least 20 peer-reviewed articles."
    )
    await evaluator.verify(
        claim=claim_director_pubs,
        node=director_pubs_leaf,
        sources=director_sources,
        additional_instruction=(
            "Prefer authoritative sources such as Google Scholar, Scopus, or institutional CVs. "
            "If a profile shows total publications ≥ 20, pass."
        )
    )

    # Director h-index >= 15
    director_hindex_leaf = evaluator.add_leaf(
        id="director_hindex_ge_15",
        desc="Director has an h-index of at least 15",
        parent=director_checks,
        critical=True
    )
    claim_director_hindex = (
        f"The director {ext.director_name or ''} has an h-index of at least 15."
    )
    await evaluator.verify(
        claim=claim_director_hindex,
        node=director_hindex_leaf,
        sources=director_sources,
        additional_instruction=(
            "Prefer Google Scholar or Scopus metrics. "
            "If the h-index is shown as ≥ 15 on any cited page, pass."
        )
    )

    # Postdoc Program (parallel)
    postdoc_checks = evaluator.add_parallel(
        id="postdoc_program",
        desc="Verify the center has an active postdoctoral program/hosts postdocs and that duration policy matches the constraint",
        parent=center_checks,
        critical=True
    )
    postdoc_sources = _merge_urls(ext.postdoc_urls, also=[ext.center_official_url] if _nonempty(ext.center_official_url) else [])

    evaluator.add_custom_node(
        result=len(postdoc_sources) > 0,
        id="postdoc_url",
        desc="Provide URL reference supporting postdoctoral program existence and duration policy",
        parent=postdoc_checks,
        critical=True
    )

    postdoc_exist_leaf = evaluator.add_leaf(
        id="postdoc_exists",
        desc="Center has an active postdoctoral fellowship program or currently hosts postdoctoral fellows",
        parent=postdoc_checks,
        critical=True
    )
    claim_postdoc_exists = (
        f"The research center '{ext.center_name or 'the center'}' has an active postdoctoral program or currently hosts postdoctoral fellows."
    )
    await evaluator.verify(
        claim=claim_postdoc_exists,
        node=postdoc_exist_leaf,
        sources=postdoc_sources,
        additional_instruction="Look for phrases like 'postdoctoral fellows', 'postdoctoral program', 'postdocs at the center', or similar."
    )

    postdoc_duration_leaf = evaluator.add_leaf(
        id="postdoc_duration_complies",
        desc="Postdoctoral fellowship duration is between 1 and 5 years OR the policy states a 5-year maximum",
        parent=postdoc_checks,
        critical=True
    )
    claim_postdoc_duration = (
        "The postdoctoral fellowship duration is between 1 and 5 years inclusive, OR the policy explicitly states a maximum duration of 5 years."
    )
    await evaluator.verify(
        claim=claim_postdoc_duration,
        node=postdoc_duration_leaf,
        sources=postdoc_sources,
        additional_instruction=(
            "Look for explicit duration statements (e.g., 1–2 years, 1–3 years, up to 5 years). "
            "If the page states a 5-year maximum policy, this also qualifies."
        )
    )

    # External Grant (parallel)
    grant_checks = evaluator.add_parallel(
        id="external_grant_qualification",
        desc="Verify at least one qualifying external grant and provide required details with citation",
        parent=center_checks,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(ext.grant_urls) > 0,
        id="grant_url",
        desc="Provide URL reference supporting the grant amount, funding source, and award date/recency",
        parent=grant_checks,
        critical=True
    )
    grant_leaf = evaluator.add_leaf(
        id="qualifying_external_grant_details",
        desc="Provide details of ≥$500,000 external grant awarded within 2020–2025, including funding source",
        parent=grant_checks,
        critical=True
    )
    grant_amount_text = ext.grant_amount or "at least $500,000"
    grant_year_text = ext.grant_year or QUALIFYING_GRANT_YEARS_DESC
    funding_src_text = ext.grant_funding_source or "an external funding source"
    claim_grant = (
        f"The research center '{ext.center_name or 'the center'}' received an external research grant of {grant_amount_text} "
        f"from {funding_src_text} within {QUALIFYING_GRANT_YEARS_DESC}. "
        f"This amount is at least $500,000 and the award date falls within {QUALIFYING_GRANT_YEARS_DESC}."
    )
    await evaluator.verify(
        claim=claim_grant,
        node=grant_leaf,
        sources=ext.grant_urls,
        additional_instruction=(
            "Verify that the grant amount on the page is ≥ $500,000 (or clearly equivalent in another currency) "
            f"AND the award date is within {QUALIFYING_GRANT_YEARS_DESC}. University/center press releases, sponsor award pages, and news articles are acceptable."
        )
    )


# --------------------------------------------------------------------------- #
# Main Evaluation Entry                                                       #
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
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_center_info(),
        template_class=ResearchCenterExtraction,
        extraction_name="research_center_extraction"
    )

    # Build and run verification tree
    await build_verification_tree(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()