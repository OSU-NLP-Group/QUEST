import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "astro_faculty_2026_top_ranked"
TASK_DESCRIPTION = (
    "Identify four astronomy or planetary science faculty members at top-ranked U.S. universities who were actively "
    "engaged in research and professional activities during 2026. For each researcher, provide: "
    "1) Full name and institutional affiliation (must be affiliated with a top-ranked U.S. astronomy or astrophysics graduate program "
    "as recognized in major rankings), 2) Primary research area (must include lunar, planetary, or solar system science), "
    "3) Conference participation in 2026 (participated in/presented at/organized sessions for/was prominently featured in at least one "
    "major astronomy or planetary science conference held in 2026), and 4) Professional visibility in 2026 (quoted in news media as an "
    "expert, authored/co-authored peer-reviewed publications, received research awards/honors, or held leadership positions in "
    "professional organizations). Include reference URLs that verify each of the four requirements for every researcher."
)
TARGET_YEAR = 2026


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------
class ConferenceInfo(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class VisibilityInfo(BaseModel):
    type: Optional[str] = None  # e.g., media quote, publication, award, leadership
    description: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ResearcherItem(BaseModel):
    full_name: Optional[str] = None
    affiliation: Optional[str] = None
    affiliation_urls: List[str] = Field(default_factory=list)  # faculty page, dept page, etc.
    ranking_urls: List[str] = Field(default_factory=list)      # US News, QS, or similar
    research_area: Optional[str] = None
    research_area_urls: List[str] = Field(default_factory=list)
    conference_2026: Optional[ConferenceInfo] = None
    visibility_2026: Optional[VisibilityInfo] = None


class ResearchersExtraction(BaseModel):
    researchers: List[ResearcherItem] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_researchers() -> str:
    return f"""
Extract up to four researchers (faculty members) exactly as presented in the answer. If more than four are provided, keep the first four; if fewer, include what is available.

For each researcher, extract the following fields:
- full_name: The person's full name as written (ideally first and last name).
- affiliation: The U.S. university or department affiliation (astronomy/astrophysics/planetary/Earth & planetary sciences, etc.).
- affiliation_urls: All URLs in the answer that directly support the person's affiliation (e.g., faculty profile, department directory, university bio).
- ranking_urls: All URLs in the answer that support that the institution's astronomy/astrophysics graduate program is top-ranked per a major ranking (e.g., U.S. News, QS, etc.). These links should show a clear ranking statement.
- research_area: The primary research area as described in the answer (should include lunar, planetary, or solar system science if the answer is correct).
- research_area_urls: URLs in the answer that support the research area claim (often the same faculty profile page or lab page).
- conference_2026:
  - name: The name of at least one major astronomy/planetary science conference in {TARGET_YEAR} that the researcher participated in (presented, organized, featured, etc.).
  - role: The described role if mentioned (e.g., presenter, organizer, session chair, invited speaker, panelist, etc.). If not mentioned, return null.
  - urls: URLs in the answer that show this {TARGET_YEAR} conference participation (e.g., conference program page, abstract page, institutional news about the 2026 event).
- visibility_2026:
  - type: One of: news_media_quote, publication, award_or_honor, leadership_role (choose the closest label; if unclear, describe briefly).
  - description: A short description of the {TARGET_YEAR} professional visibility (e.g., quoted in XYZ news about an astronomical event; co-authored a 2026 peer-reviewed paper; received ABC award in 2026; served as officer/committee chair in 2026).
  - urls: URLs in the answer that support this {TARGET_YEAR} visibility evidence.

Rules:
- Do not invent any person, institution, role, or URL that is not in the answer.
- Only include valid URLs that appear in the answer text. When the same URL is used for multiple fields, repeat it where appropriate.
- If any field is not present in the answer, set it to null (for strings/objects) or [] (for lists).
- The goal is to capture exactly what the answer provided so the verification step can check it.
- Return a JSON with a top-level field "researchers" which is an array of up to 4 ResearcherItem objects, in the order they appeared in the answer.
"""


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _dedup_urls(urls: List[str]) -> List[str]:
    cleaned = [u.strip() for u in urls if isinstance(u, str) and u.strip()]
    # Deduplicate preserving order
    return list(dict.fromkeys(cleaned))


def _has_any_url(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    return any(isinstance(u, str) and u.strip() for u in urls)


def _safe(obj: Optional[str]) -> str:
    return obj or ""


def _merge_sources(*url_lists: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in url_lists:
        merged.extend(lst or [])
    return _dedup_urls(merged)


# -----------------------------------------------------------------------------
# Verification per researcher
# -----------------------------------------------------------------------------
async def verify_one_researcher(
    evaluator: Evaluator,
    parent_node,
    item: ResearcherItem,
    index: int,
) -> None:
    rid = index + 1
    name = _safe(item.full_name)
    affiliation = _safe(item.affiliation)
    research_area = _safe(item.research_area)

    aff_urls = _dedup_urls(item.affiliation_urls)
    rank_urls = _dedup_urls(item.ranking_urls)
    area_urls = _merge_sources(item.research_area_urls, item.affiliation_urls)

    conf = item.conference_2026 or ConferenceInfo()
    conf_name = _safe(conf.name)
    conf_role = _safe(conf.role)
    conf_urls = _dedup_urls(conf.urls)

    vis = item.visibility_2026 or VisibilityInfo()
    vis_type = _safe(vis.type)
    vis_desc = _safe(vis.description)
    vis_urls = _dedup_urls(vis.urls)

    # Researcher group node (non-critical to allow partial credit across researchers)
    r_node = evaluator.add_parallel(
        id=f"researcher_{rid}",
        desc=f"Researcher #{rid} verification (must meet all specified criteria)",
        parent=parent_node,
        critical=False,
    )

    # 1) Full name provided (critical)
    name_ok = bool(name.strip()) and (" " in name.strip())
    evaluator.add_custom_node(
        result=name_ok,
        id=f"r{rid}_full_name",
        desc=f"Researcher {rid}'s full name (first and last name) is provided",
        parent=r_node,
        critical=True,
    )

    # 2) Affiliation verification (critical group)
    aff_group = evaluator.add_parallel(
        id=f"r{rid}_affiliation_group",
        desc=f"Researcher {rid} affiliation and ranking verification",
        parent=r_node,
        critical=True,
    )

    # 2.a Affiliation string provided
    evaluator.add_custom_node(
        result=bool(affiliation.strip()),
        id=f"r{rid}_affiliation_text_present",
        desc=f"Researcher {rid} affiliation text is provided",
        parent=aff_group,
        critical=True,
    )

    # 2.b Affiliation supported by URLs
    aff_verify_node = evaluator.add_leaf(
        id=f"r{rid}_affiliation",
        desc=f"Researcher {rid} is affiliated with the stated institution",
        parent=aff_group,
        critical=True,
    )
    aff_claim = (
        f"{name} is a faculty member (e.g., Professor/Research Professor/Associate/Assistant Professor or similar faculty role) "
        f"affiliated with the astronomy/astrophysics/planetary-related program/department at {affiliation}."
    )
    await evaluator.verify(
        claim=aff_claim,
        node=aff_verify_node,
        sources=aff_urls,
        additional_instruction=(
            "Verify that the page indicates the person is faculty (acceptable titles include Professor, Associate/Assistant Professor, "
            "Research Professor, Research Scientist with faculty appointment, etc.) at the named institution and relevant department/program."
        ),
    )

    # 2.c Top-ranked program supported by URLs
    rank_verify_node = evaluator.add_leaf(
        id=f"r{rid}_top_ranked",
        desc=f"The institution's astronomy/astrophysics graduate program is top-ranked per a major ranking",
        parent=aff_group,
        critical=True,
    )
    rank_claim = (
        f"The astronomy or astrophysics graduate program at {affiliation} is top-ranked in major rankings "
        f"(such as U.S. News, QS, Times Higher Education, or similar)."
    )
    await evaluator.verify(
        claim=rank_claim,
        node=rank_verify_node,
        sources=rank_urls,
        additional_instruction=(
            "Consider 'top-ranked' satisfied if the ranking source indicates a clearly high standing (e.g., top 20, top tier). "
            "Accept authoritative ranking sources; institutional pages summarizing such rankings are also acceptable if they cite the source."
        ),
    )

    # 3) Research area includes lunar/planetary/solar system (critical group)
    area_group = evaluator.add_parallel(
        id=f"r{rid}_research_area_group",
        desc=f"Researcher {rid} research area verification",
        parent=r_node,
        critical=True,
    )

    # 3.a Research area string provided
    evaluator.add_custom_node(
        result=bool(research_area.strip()),
        id=f"r{rid}_research_area_text_present",
        desc=f"Researcher {rid}'s primary research area text is provided",
        parent=area_group,
        critical=True,
    )

    # 3.b Research area supported by URLs (must include lunar/planetary/solar system science)
    area_verify_node = evaluator.add_leaf(
        id=f"r{rid}_research_area",
        desc=f"Researcher {rid}'s primary research area includes lunar, planetary, or solar system science",
        parent=area_group,
        critical=True,
    )
    area_claim = (
        f"{name}'s research interests include lunar, planetary, or solar system science. "
        f"The described area is: '{research_area}'."
    )
    await evaluator.verify(
        claim=area_claim,
        node=area_verify_node,
        sources=area_urls,
        additional_instruction=(
            "Look for explicit evidence (keywords or phrases) such as lunar, Moon, Mars, planets, planetary science, "
            "planetary geology/geophysics, planetary atmospheres, small bodies (asteroids, comets), Kuiper Belt, "
            "planet formation, or 'solar system science'. The source(s) should clearly support involvement in at least one of these."
        ),
    )

    # 4) Conference participation in 2026 (critical group)
    conf_group = evaluator.add_parallel(
        id=f"r{rid}_conference_group",
        desc=f"Researcher {rid} 2026 conference participation verification",
        parent=r_node,
        critical=True,
    )

    # 4.a Conference info provided (at least a name/role and URLs)
    conf_info_present = (_has_any_url(conf_urls)) and (bool(conf_name.strip()) or bool(conf_role.strip()))
    evaluator.add_custom_node(
        result=conf_info_present,
        id=f"r{rid}_conference_info_present",
        desc=f"Researcher {rid} has 2026 conference info (name/role and URLs) provided",
        parent=conf_group,
        critical=True,
    )

    # 4.b Conference participation verified by URLs
    conf_verify_node = evaluator.add_leaf(
        id=f"r{rid}_conference_2026",
        desc=f"Researcher {rid} participated in at least one major astronomy/planetary science conference held in {TARGET_YEAR}",
        parent=conf_group,
        critical=True,
    )
    conf_claim = (
        f"In {TARGET_YEAR}, {name} participated in the conference '{conf_name}'"
        + (f" with role '{conf_role}'." if conf_role else ".")
    )
    await evaluator.verify(
        claim=conf_claim,
        node=conf_verify_node,
        sources=conf_urls,
        additional_instruction=(
            f"Confirm the event year is {TARGET_YEAR}. Participation can include being a presenter, organizer, session chair, "
            f"panelist, or being prominently featured in the program or institutional news about the {TARGET_YEAR} event."
        ),
    )

    # 5) Professional visibility in 2026 (critical group)
    vis_group = evaluator.add_parallel(
        id=f"r{rid}_visibility_group",
        desc=f"Researcher {rid} 2026 professional visibility verification",
        parent=r_node,
        critical=True,
    )

    # 5.a Visibility info provided (type/description and URLs)
    vis_info_present = (_has_any_url(vis_urls)) and (bool(vis_type.strip()) or bool(vis_desc.strip()))
    evaluator.add_custom_node(
        result=vis_info_present,
        id=f"r{rid}_visibility_info_present",
        desc=f"Researcher {rid} has 2026 professional visibility info (type/description and URLs) provided",
        parent=vis_group,
        critical=True,
    )

    # 5.b Visibility verified by URLs
    vis_verify_node = evaluator.add_leaf(
        id=f"r{rid}_public_visibility",
        desc=f"Researcher {rid} had public professional visibility in {TARGET_YEAR}",
        parent=vis_group,
        critical=True,
    )
    vis_claim = (
        f"In {TARGET_YEAR}, {name} had professional visibility via '{vis_type}'"
        + (f" (details: {vis_desc})." if vis_desc else ".")
    )
    await evaluator.verify(
        claim=vis_claim,
        node=vis_verify_node,
        sources=vis_urls,
        additional_instruction=(
            f"Accept evidence such as: being quoted in news media as an expert, authoring/co-authoring peer-reviewed publications "
            f"in {TARGET_YEAR}, receiving research awards/honors in {TARGET_YEAR}, or holding leadership positions in professional "
            f"organizations in {TARGET_YEAR}. The page(s) must clearly indicate the year {TARGET_YEAR}."
        ),
    )

    # 6) Reference URLs provided for each requirement (custom, critical)
    #    Requirements with expected URLs: affiliation, research area, conference 2026, professional visibility 2026.
    refs_ok = (
        _has_any_url(aff_urls) and
        (_has_any_url(area_urls)) and
        _has_any_url(conf_urls) and
        _has_any_url(vis_urls)
    )
    evaluator.add_custom_node(
        result=refs_ok,
        id=f"r{rid}_reference_urls",
        desc=f"Reference URLs are provided that verify Researcher {rid}'s info (affiliation, research area, conference participation, and professional visibility)",
        parent=r_node,
        critical=True,
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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

    # Extract researchers from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_researchers(),
        template_class=ResearchersExtraction,
        extraction_name="researchers_extraction",
    )

    # Normalize to exactly 4 researchers
    researchers: List[ResearcherItem] = list(extracted.researchers or [])
    if len(researchers) > 4:
        researchers = researchers[:4]
    while len(researchers) < 4:
        researchers.append(ResearcherItem())

    # Add minimal ground-truth context info (for transparency)
    evaluator.add_ground_truth({
        "required_researchers": 4,
        "target_year": TARGET_YEAR,
        "requirements": [
            "Full name and affiliation at a top-ranked U.S. astronomy/astrophysics graduate program",
            "Primary research area includes lunar/planetary/solar system science",
            f"Participation in at least one major astronomy/planetary conference in {TARGET_YEAR}",
            f"Professional visibility in {TARGET_YEAR}: news media quote, publication, award, or leadership"
        ]
    })

    # Build and verify per researcher
    for idx, item in enumerate(researchers):
        await verify_one_researcher(evaluator, root, item, idx)

    return evaluator.get_summary()