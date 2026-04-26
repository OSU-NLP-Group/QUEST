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
TASK_ID = "ai_ml_us_top10_faculty_diversity"
TASK_DESCRIPTION = """
Identify four artificial intelligence or machine learning researchers who are current faculty members at universities in the United States. Each researcher must satisfy the following criteria:

1. University Ranking: They must be affiliated with a university that is ranked in the top 10 globally for Computer Science according to at least one of the following rankings: QS World University Rankings by Subject 2024 or 2025 (Computer Science & Information Systems), Times Higher Education World University Rankings 2024 (Computer Science), or U.S. News Best Graduate Schools (Computer Science).

2. Research Focus: Their primary research area must be in Artificial Intelligence, Machine Learning, or a closely related subfield such as Computer Vision, Natural Language Processing, or Robotics.

3. Recognition: They must have received at least one of the following forms of recognition:
   - NSF CAREER Award (any year)
   - ACM Fellow designation
   - IEEE Fellow designation
   - Best Paper Award or Distinguished Paper Award from a top-tier conference (examples include NeurIPS, ICML, ICLR, CVPR, ICSE, FSE, AAAI, or equivalent)

4. Publication Record: They must have authored or co-authored publications that appeared at major AI/ML conferences, specifically: NeurIPS (Conference on Neural Information Processing Systems), ICML (International Conference on Machine Learning), ICLR (International Conference on Learning Representations), CVPR (Conference on Computer Vision and Pattern Recognition), or AAAI (Conference on the Association for the Advancement of Artificial Intelligence).

Additionally, the collective set of four researchers must satisfy:

5. Geographic Diversity: The four researchers must be affiliated with at least three different universities.

6. Career Stage Diversity: At least two of the four researchers must be early-career faculty, defined as either holding the rank of Assistant Professor or having received an NSF CAREER Award between 2020 and 2024 (inclusive).

For each researcher, provide their full name, current university affiliation, research area, at least one specific award or recognition they have received (with year if available), and at least one conference where they have published their work. Include URL references to verify each piece of information.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RankingSource(BaseModel):
    name: Optional[str] = None  # e.g., "QS 2024 Computer Science & Information Systems", "THE 2024 Computer Science", "U.S. News Best Graduate Schools (Computer Science)"
    year: Optional[str] = None
    url: Optional[str] = None


class Recognition(BaseModel):
    type: Optional[str] = None  # e.g., "NSF CAREER", "ACM Fellow", "IEEE Fellow", "Best Paper Award", "Distinguished Paper Award"
    venue_or_org: Optional[str] = None  # e.g., "NeurIPS", "ICML", "CVPR", "AAAI", "ACM", "IEEE"
    year: Optional[str] = None
    url: Optional[str] = None


class Publication(BaseModel):
    conference: Optional[str] = None  # e.g., "NeurIPS", "ICML", "ICLR", "CVPR", "AAAI"
    paper_title: Optional[str] = None
    year: Optional[str] = None
    url: Optional[str] = None


class Researcher(BaseModel):
    full_name: Optional[str] = None
    university: Optional[str] = None
    department: Optional[str] = None
    country: Optional[str] = None  # expected "United States" or equivalent
    faculty_title: Optional[str] = None  # e.g., "Assistant Professor", "Associate Professor", "Professor"
    affiliation_urls: List[str] = Field(default_factory=list)  # faculty or dept profile page(s)
    university_location_urls: List[str] = Field(default_factory=list)  # wikipedia or official page showing US location
    ranking_sources: List[RankingSource] = Field(default_factory=list)
    research_area: Optional[str] = None
    research_area_urls: List[str] = Field(default_factory=list)
    recognitions: List[Recognition] = Field(default_factory=list)
    publications: List[Publication] = Field(default_factory=list)


class ResearchersExtraction(BaseModel):
    researchers: List[Researcher] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_researchers() -> str:
    return """
    Extract all researchers mentioned in the answer (in the order they appear). For each researcher, extract the following fields exactly as provided by the answer and include URLs to verify each claim:

    For each researcher object:
    - full_name: The researcher's full name as stated.
    - university: The current university affiliation.
    - department: The department, school, or lab (if available).
    - country: The country of the university (e.g., "United States" or "USA").
    - faculty_title: The current faculty rank (e.g., "Assistant Professor", "Associate Professor", "Professor").
    - affiliation_urls: Array of URLs that verify the affiliation and faculty status (official university pages preferred).
    - university_location_urls: Array of URLs that verify the university is in the United States (e.g., university "About" page or Wikipedia).
    - ranking_sources: Array of objects, each with:
        * name: Name of the ranking source, ideally one of:
            - "QS World University Rankings by Subject 2024 (Computer Science & Information Systems)"
            - "QS World University Rankings by Subject 2025 (Computer Science & Information Systems)"
            - "Times Higher Education World University Rankings 2024 (Computer Science)"
            - "U.S. News Best Graduate Schools (Computer Science)"
        * year: The year associated with the ranking (if available)
        * url: A URL to the ranking page that explicitly shows the ranking position or top-10 status
    - research_area: The stated primary research area (e.g., "Artificial Intelligence", "Machine Learning", "Computer Vision", "Natural Language Processing", "Robotics").
    - research_area_urls: Array of URLs that support the research area claim (official profile pages preferred).
    - recognitions: Array of objects, each with:
        * type: One of "NSF CAREER", "ACM Fellow", "IEEE Fellow", "Best Paper Award", "Distinguished Paper Award" (or close variant if provided by the answer)
        * venue_or_org: The awarding organization or conference (e.g., "NSF", "ACM", "IEEE", "NeurIPS", "ICML", "ICLR", "CVPR", "AAAI", "ICSE", "FSE")
        * year: Year of the award (if available)
        * url: A URL to evidence (official announcement, award list, conference proceedings, etc.)
    - publications: Array of objects, each with:
        * conference: The conference name (e.g., "NeurIPS", "ICML", "ICLR", "CVPR", "AAAI")
        * paper_title: The title of a paper by the researcher at that conference (if the answer provides it)
        * year: Year of the paper (if provided)
        * url: A URL to evidence (conference proceedings page, openreview, dblp, official page)

    Return a JSON object with a single key "researchers" that is an array of researcher objects. If any field is missing for a researcher, set it to null (or an empty array for URLs).
    Make sure to include every researcher mentioned in the answer, not just four (we will select at most the first four for verification).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def _lower(s: Optional[str]) -> str:
    return (s or "").strip().lower()


ALLOWED_CONF_PUB = {"neurips", "nips", "icml", "iclr", "cvpr", "aaai"}
ALLOWED_CONF_AWARD = {"neurips", "nips", "icml", "iclr", "cvpr", "aaai", "icse", "fse"}


def conference_is_allowed_for_pub(conf: Optional[str]) -> bool:
    s = _lower(conf)
    if not s:
        return False
    # Handle variants like "Conference on Neural Information Processing Systems (NeurIPS)"
    for key in ALLOWED_CONF_PUB:
        if key in s:
            return True
    # Also check common expansions
    expansions = {
        "neural information processing systems": "neurips",
        "international conference on machine learning": "icml",
        "international conference on learning representations": "iclr",
        "computer vision and pattern recognition": "cvpr",
        "association for the advancement of artificial intelligence": "aaai",
    }
    for phrase, abbr in expansions.items():
        if phrase in s:
            return True
    return False


def conference_is_allowed_for_award(conf_or_org: Optional[str]) -> bool:
    s = _lower(conf_or_org)
    if not s:
        return False
    for key in ALLOWED_CONF_AWARD:
        if key in s:
            return True
    # Similar expansions as above
    expansions = {
        "neural information processing systems": "neurips",
        "international conference on machine learning": "icml",
        "international conference on learning representations": "iclr",
        "computer vision and pattern recognition": "cvpr",
        "association for the advancement of artificial intelligence": "aaai",
        "international conference on software engineering": "icse",
        "foundations of software engineering": "fse",
    }
    for phrase, abbr in expansions.items():
        if phrase in s:
            return True
    return False


def is_allowed_ranking_name(name: Optional[str]) -> bool:
    s = _lower(name)
    if not s:
        return False
    # QS CS & IS 2024 or 2025
    if "qs" in s and "computer" in s and "science" in s and ("2024" in s or "2025" in s):
        return True
    # THE World University Rankings 2024 - Computer Science
    if (("times higher education" in s) or s.startswith("the ") or s == "the") and "computer science" in s and "2024" in s:
        return True
    # U.S. News Best Graduate Schools (Computer Science)
    if ("u.s. news" in s or "us news" in s or "u.s.news" in s) and "best graduate" in s and "computer science" in s:
        return True
    return False


def pick_allowed_ranking_source(r: Researcher) -> Optional[RankingSource]:
    for rs in r.ranking_sources or []:
        if is_allowed_ranking_name(rs.name) and _norm(rs.url):
            return rs
    return None


def pick_allowed_recognition(r: Researcher) -> Optional[Recognition]:
    for rec in r.recognitions or []:
        t = _lower(rec.type)
        if "nsf" in t and "career" in t and _norm(rec.url):
            return rec
        if "acm" in t and "fellow" in t and _norm(rec.url):
            return rec
        if "ieee" in t and "fellow" in t and _norm(rec.url):
            return rec
        if ("best paper" in t or "distinguished paper" in t) and conference_is_allowed_for_award(rec.venue_or_org) and _norm(rec.url):
            return rec
    return None


def pick_allowed_publication(r: Researcher) -> Optional[Publication]:
    for pub in r.publications or []:
        if conference_is_allowed_for_pub(pub.conference) and _norm(pub.url):
            return pub
    return None


def parse_year_int(y: Optional[str]) -> Optional[int]:
    try:
        return int(_norm(y))
    except Exception:
        return None


def is_early_career(r: Researcher) -> bool:
    title = _lower(r.faculty_title)
    if "assistant professor" in title:
        return True
    # NSF CAREER between 2020 and 2024 inclusive
    for rec in r.recognitions or []:
        if rec.type and "nsf" in _lower(rec.type) and "career" in _lower(rec.type):
            year = parse_year_int(rec.year)
            if year is not None and 2020 <= year <= 2024:
                return True
    return False


def non_empty_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if _norm(u)]


# --------------------------------------------------------------------------- #
# Verification subroutine per researcher                                      #
# --------------------------------------------------------------------------- #
async def verify_researcher(
    evaluator: Evaluator,
    parent_node,
    r: Researcher,
    idx: int
) -> None:
    # Create researcher node
    rnode = evaluator.add_parallel(
        id=f"Researcher_{idx+1}",
        desc=f"Researcher {idx+1} satisfies all per-researcher constraints.",
        parent=parent_node,
        critical=False
    )

    # R#_Full_Name_Provided
    evaluator.add_custom_node(
        result=bool(_norm(r.full_name)),
        id=f"R{idx+1}_Full_Name_Provided",
        desc=f"Researcher {idx+1} full name is provided.",
        parent=rnode,
        critical=True
    )

    # R#_US_University_Affiliation (we focus here on verifying the university is in the U.S.)
    us_aff_node = evaluator.add_leaf(
        id=f"R{idx+1}_US_University_Affiliation",
        desc=f"Researcher {idx+1} is at a U.S. university.",
        parent=rnode,
        critical=True
    )
    uni = _norm(r.university)
    claim_us_uni = f"The university '{uni}' is located in the United States."
    us_sources = non_empty_urls(r.university_location_urls)
    if not us_sources:
        # fall back to ranking source URLs which often show country info
        us_sources = [rs.url for rs in (r.ranking_sources or []) if _norm(rs.url)]
    await evaluator.verify(
        claim=claim_us_uni,
        node=us_aff_node,
        sources=us_sources,
        additional_instruction="Verify that the page explicitly or clearly indicates that the university is in the United States. Accept common variants such as 'United States', 'USA', 'U.S.'."
    )

    # R#_Faculty_Status_AsOf2024 (verify tenure-track/tenured status via title)
    fac_node = evaluator.add_leaf(
        id=f"R{idx+1}_Faculty_Status_AsOf2024",
        desc=f"Researcher {idx+1} holds a tenure-track or tenured faculty position as of 2024.",
        parent=rnode,
        critical=True
    )
    title_txt = _norm(r.faculty_title)
    name_txt = _norm(r.full_name)
    claim_fac = f"{name_txt} holds a faculty position at {uni} with a title consistent with a tenure-track or tenured role (e.g., Assistant, Associate, or Professor)."
    await evaluator.verify(
        claim=claim_fac,
        node=fac_node,
        sources=non_empty_urls(r.affiliation_urls),
        additional_instruction="Pass only if the page shows a faculty title such as Assistant Professor, Associate Professor, or Professor. Titles like 'Adjunct', 'Visiting', 'Lecturer', or 'Research Scientist' should not be considered tenure-track/tenured."
    )

    # R#_Top10_Ranking
    top10_node = evaluator.add_leaf(
        id=f"R{idx+1}_Top10_Ranking",
        desc=f"Researcher {idx+1} university is top-10 globally for CS in an allowed ranking.",
        parent=rnode,
        critical=True
    )
    rs = pick_allowed_ranking_source(r)
    rs_name = rs.name if rs else ""
    rs_url = rs.url if rs else None
    claim_rank = f"According to '{rs_name}', the university '{uni}' is ranked within the top 10 for Computer Science."
    await evaluator.verify(
        claim=claim_rank,
        node=top10_node,
        sources=rs_url,
        additional_instruction="Only pass if this page is one of the allowed ranking sources: QS World University Rankings by Subject 2024/2025 (Computer Science & Information Systems), Times Higher Education World University Rankings 2024 (Computer Science), or U.S. News Best Graduate Schools (Computer Science). The page must indicate a rank of 10 or better (i.e., top 10)."
    )

    # R#_AI_ML_Research_Focus
    focus_node = evaluator.add_leaf(
        id=f"R{idx+1}_AI_ML_Research_Focus",
        desc=f"Researcher {idx+1} primary research is AI/ML or a close subfield.",
        parent=rnode,
        critical=True
    )
    area = _norm(r.research_area)
    claim_focus = f"{name_txt}'s primary research area is '{area}', which falls under AI/ML or a closely related subfield (e.g., Computer Vision, NLP, Robotics)."
    await evaluator.verify(
        claim=claim_focus,
        node=focus_node,
        sources=non_empty_urls(r.research_area_urls) or non_empty_urls(r.affiliation_urls),
        additional_instruction="Confirm that the page indicates the person's main research focus in AI, Machine Learning, Computer Vision, Natural Language Processing, or Robotics."
    )

    # R#_Recognition_Allowed_List
    recog_node = evaluator.add_leaf(
        id=f"R{idx+1}_Recognition_Allowed_List",
        desc=f"Researcher {idx+1} has an allowed recognition.",
        parent=rnode,
        critical=True
    )
    rec = pick_allowed_recognition(r)
    rec_type = rec.type if rec else ""
    rec_venue = rec.venue_or_org if rec else ""
    rec_year = rec.year if rec else ""
    claim_rec = f"{name_txt} has an allowed recognition: '{rec_type}' {rec_year} {('at ' + rec_venue) if rec_venue else ''}."
    await evaluator.verify(
        claim=claim_rec,
        node=recog_node,
        sources=(rec.url if rec and _norm(rec.url) else None),
        additional_instruction="Allowed recognitions include NSF CAREER (any year), ACM Fellow, IEEE Fellow, or Best/Distinguished Paper Award from a top-tier venue (NeurIPS, ICML, ICLR, CVPR, ICSE, FSE, AAAI). Verify that the page clearly supports the claimed recognition."
    )

    # R#_Publication_At_Specified_Conference
    pub_node = evaluator.add_leaf(
        id=f"R{idx+1}_Publication_At_Specified_Conference",
        desc=f"Researcher {idx+1} has publication at NeurIPS/ICML/ICLR/CVPR/AAAI.",
        parent=rnode,
        critical=True
    )
    pub = pick_allowed_publication(r)
    pub_conf = pub.conference if pub else ""
    claim_pub = f"{name_txt} has authored or co-authored at least one publication at {pub_conf}."
    await evaluator.verify(
        claim=claim_pub,
        node=pub_node,
        sources=(pub.url if pub and _norm(pub.url) else None),
        additional_instruction="Verify that the page is a reliable source (e.g., conference proceedings, OpenReview, DBLP, conference website) showing the person as an author at one of the specified conferences: NeurIPS, ICML, ICLR, CVPR, or AAAI."
    )

    # R#_URL_Evidence_For_Each_Claim (existence of URLs for each major claim)
    has_aff_urls = len(non_empty_urls(r.affiliation_urls)) > 0
    has_loc_urls = len(non_empty_urls(r.university_location_urls)) > 0 or any(_norm(s.url) for s in (r.ranking_sources or []))
    has_rank_url = rs is not None and _norm(rs.url)
    has_focus_urls = len(non_empty_urls(r.research_area_urls)) > 0 or has_aff_urls
    has_rec_url = rec is not None and _norm(rec.url)
    has_pub_url = pub is not None and _norm(pub.url)

    evaluator.add_custom_node(
        result=bool(has_aff_urls and has_loc_urls and has_rank_url and has_focus_urls and has_rec_url and has_pub_url),
        id=f"R{idx+1}_URL_Evidence_For_Each_Claim",
        desc=f"URLs cover affiliation, ranking, research area, recognition, and publication evidence for Researcher {idx+1}.",
        parent=rnode,
        critical=True
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
        default_model=model,
    )

    # Extract structured researchers from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_researchers(),
        template_class=ResearchersExtraction,
        extraction_name="researchers_extraction"
    )

    all_researchers = extracted.researchers or []
    selected = all_researchers[:4]
    # Pad to 4 if fewer provided
    while len(selected) < 4:
        selected.append(Researcher())

    # ---------------- Response Format (Critical) ----------------
    resp_fmt = evaluator.add_parallel(
        id="Response_Format",
        desc="Solution provides exactly four distinct researchers (not fewer/more) and they are distinct individuals.",
        parent=root,
        critical=True
    )

    exactly_four = evaluator.add_custom_node(
        result=(len(all_researchers) == 4),
        id="Exactly_Four_Researchers",
        desc="Exactly four researchers are provided.",
        parent=resp_fmt,
        critical=True
    )

    names_first_four = [(_lower(r.full_name)) for r in selected]
    distinct_names = len({n for n in names_first_four if n}) == 4 and all(n for n in names_first_four)

    evaluator.add_custom_node(
        result=(len(all_researchers) == 4 and distinct_names),
        id="Researchers_Are_Distinct",
        desc="The four researchers are distinct individuals (no duplicates).",
        parent=resp_fmt,
        critical=True
    )

    # ---------------- Per-Researcher Verification (Non-Critical Nodes) -----
    for i in range(4):
        await verify_researcher(evaluator, root, selected[i], i)

    # ---------------- Set-Level Constraints (Critical) ---------------------
    set_level = evaluator.add_parallel(
        id="Set_Level_Constraints",
        desc="The set of four researchers satisfies diversity constraints.",
        parent=root,
        critical=True
    )

    # Geographic diversity: at least 3 different universities among the four selected
    universities = [(_lower(r.university)) for r in selected if _norm(r.university)]
    unique_unis = len(set(universities))
    evaluator.add_custom_node(
        result=(unique_unis >= 3),
        id="Geographic_Diversity",
        desc="The four researchers are affiliated with at least three different universities (and these are U.S. universities).",
        parent=set_level,
        critical=True
    )

    # Career stage diversity: at least 2 early-career (Assistant Professor OR NSF CAREER 2020–2024)
    early_career_count = sum(1 for r in selected if is_early_career(r))
    evaluator.add_custom_node(
        result=(early_career_count >= 2),
        id="Career_Stage_Diversity",
        desc="At least two of the four researchers are early-career faculty (Assistant Professor OR NSF CAREER Award between 2020–2024 inclusive).",
        parent=set_level,
        critical=True
    )

    # Add custom info for transparency
    evaluator.add_custom_info(
        info={
            "total_researchers_extracted": len(all_researchers),
            "selected_universities": [r.university for r in selected],
            "early_career_flags": [is_early_career(r) for r in selected],
        },
        info_type="analysis",
        info_name="extraction_summary"
    )

    return evaluator.get_summary()