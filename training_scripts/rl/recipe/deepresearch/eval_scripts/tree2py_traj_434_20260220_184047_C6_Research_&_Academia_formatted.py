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
TASK_ID = "odu_cs_faculty_scholar_eval"
TASK_DESCRIPTION = """
Identify a faculty member in the Computer Science Department at Old Dominion University who meets all of the following criteria:

1. Holds the academic rank of Associate Professor or Professor (not Assistant Professor or Lecturer)
2. Has a verified Google Scholar profile with an email address ending in @odu.edu or @cs.odu.edu
3. Has published research in at least one of the following areas: Artificial Intelligence, Machine Learning, Cybersecurity, Data Analytics, or Human-Computer Interaction
4. Has at least 15 total publications listed on their Google Scholar profile
5. Has an h-index of at least 20 on Google Scholar
6. Has published at least 3 papers between 2020 and 2024 (inclusive)
7. Has collaborated with at least 5 distinct co-authors in publications from 2020 to 2024

Provide the faculty member's full name, their Google Scholar profile URL, and URLs confirming their departmental affiliation and academic rank.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FacultyExtraction(BaseModel):
    # Core identity and profile URLs
    full_name: Optional[str] = None
    google_scholar_url: Optional[str] = None

    # Affiliation and rank URLs (often ODU CS people pages)
    affiliation_url: Optional[str] = None  # ODU or CS ODU domain page showing department affiliation
    rank_url: Optional[str] = None         # URL confirming academic rank (can be the same as affiliation_url)

    # Claimed rank string if present in the answer (e.g., "Associate Professor", "Professor")
    claimed_rank: Optional[str] = None

    # Research areas and references
    research_areas_claimed: List[str] = Field(default_factory=list)
    area_evidence_urls: List[str] = Field(default_factory=list)

    # Additional references for recent publications and collaboration
    recent_publications_urls: List[str] = Field(default_factory=list)
    collaboration_urls: List[str] = Field(default_factory=list)

    # Optional numeric claims if present in the answer (strings to be permissive)
    total_publications_claim: Optional[str] = None
    h_index_claim: Optional[str] = None
    pubs_2020_2024_claim: Optional[str] = None
    distinct_coauthors_2020_2024_claim: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_faculty() -> str:
    return """
Extract the faculty member details exactly as presented in the answer.

Return a JSON object with the following fields:
- full_name: The faculty member's full name (string)
- google_scholar_url: The Google Scholar profile URL (string, if provided)
- affiliation_url: A URL on odu.edu (or a subdomain like cs.odu.edu) confirming the faculty member’s departmental affiliation (string, if provided)
- rank_url: A URL confirming the academic rank (string, if provided; can be the same as affiliation_url if rank is shown there)
- claimed_rank: The academic rank mentioned in the answer (e.g., "Associate Professor", "Professor"). Use the exact phrasing from the answer if available; otherwise null.
- research_areas_claimed: An array of research areas claimed in the answer (strings). Only include areas explicitly mentioned in the answer.
- area_evidence_urls: An array of URLs that the answer cites to support research area(s) (e.g., Google Scholar, personal website, research page). Only include URLs that appear in the answer.
- recent_publications_urls: An array of URLs in the answer that show recent publications (e.g., Google Scholar, publication list). Only include URLs explicitly in the answer.
- collaboration_urls: An array of URLs in the answer that show co-authorship info for recent publications (e.g., Google Scholar, DBLP). Only include URLs explicitly in the answer.
- total_publications_claim: If the answer mentions a total publications count, extract it as a string; otherwise null.
- h_index_claim: If the answer mentions the h-index, extract it as a string; otherwise null.
- pubs_2020_2024_claim: If the answer mentions the number of publications between 2020 and 2024, extract it as a string; otherwise null.
- distinct_coauthors_2020_2024_claim: If the answer mentions the number of distinct co-authors between 2020 and 2024, extract it as a string; otherwise null.

Important URL rules:
- Extract only URLs that are explicitly present in the answer text (including markdown links). Do not infer or create URLs.
- Include full URLs, with http:// or https://.
- Do not include duplicate URLs. If both http and https versions appear, keep the https version.

If a field is missing from the answer, set it to null (for strings) or an empty array (for lists).
""".strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _is_odu_domain(url: Optional[str]) -> bool:
    if not _non_empty(url):
        return False
    u = url.strip().lower()
    return "odu.edu" in u  # Accept main and subdomains (e.g., cs.odu.edu)


def _is_scholar_url(url: Optional[str]) -> bool:
    if not _non_empty(url):
        return False
    u = url.strip().lower()
    return "scholar.google." in u and "/citations" in u


def _collect_sources(*urls: Optional[str], extra_lists: Optional[List[List[str]]] = None) -> List[str]:
    seen = set()
    results: List[str] = []
    for u in urls:
        if _non_empty(u):
            uu = u.strip()
            if uu not in seen:
                seen.add(uu)
                results.append(uu)
    if extra_lists:
        for lst in extra_lists:
            for u in lst:
                if _non_empty(u) and u not in seen:
                    seen.add(u)
                    results.append(u)
    return results


def _rank_source_url(data: FacultyExtraction) -> Optional[str]:
    # Prefer rank_url if present; otherwise fallback to affiliation_url
    return data.rank_url if _non_empty(data.rank_url) else (data.affiliation_url if _non_empty(data.affiliation_url) else None)


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, root_node, data: FacultyExtraction) -> None:
    # Top-level critical node
    top = evaluator.add_parallel(
        id="Faculty_Member_Verification",
        desc="Verify that a Computer Science faculty member at Old Dominion University meets all specified research criteria",
        parent=root_node,
        critical=True
    )

    # --------------------------- Basic Qualifications -------------------- #
    basic = evaluator.add_parallel(
        id="Basic_Qualifications",
        desc="Verify the faculty member's institutional affiliation and academic rank",
        parent=top,
        critical=True
    )

    # Institutional Affiliation
    inst = evaluator.add_parallel(
        id="Institutional_Affiliation",
        desc="Verify the faculty member's affiliation with Old Dominion University Computer Science Department",
        parent=basic,
        critical=True
    )

    # Affiliation_Reference: custom existence + domain check
    affiliation_ref_ok = evaluator.add_custom_node(
        result=_is_odu_domain(data.affiliation_url),
        id="Affiliation_Reference",
        desc="Provide a URL from odu.edu domain confirming the faculty member's departmental affiliation",
        parent=inst,
        critical=True
    )

    # ODU_CS_Department: verify listing on ODU CS (or ODU) page
    odu_cs_leaf = evaluator.add_leaf(
        id="ODU_CS_Department",
        desc="Confirm the faculty member is listed in the Old Dominion University Computer Science Department",
        parent=inst,
        critical=True
    )
    await evaluator.verify(
        claim=f"The webpage confirms that '{data.full_name or 'the faculty member'}' is a faculty member in the Computer Science Department at Old Dominion University.",
        node=odu_cs_leaf,
        sources=data.affiliation_url,
        additional_instruction="Accept wording variants such as 'Department of Computer Science' or 'Computer Science, ODU'. The page should clearly associate the person with ODU CS."
    )

    # Email_Domain: verify verified email domain on Scholar
    email_leaf = evaluator.add_leaf(
        id="Email_Domain",
        desc="Verify the faculty member's email ends with @odu.edu or @cs.odu.edu",
        parent=inst,
        critical=True
    )
    await evaluator.verify(
        claim="The Google Scholar profile displays a verified email at domain odu.edu or cs.odu.edu.",
        node=email_leaf,
        sources=data.google_scholar_url,
        additional_instruction="On Google Scholar, look for the 'Verified email' indicator near the name; it should read 'Verified email at odu.edu' or 'Verified email at cs.odu.edu'. Minor phrasing variations are acceptable."
    )

    # Academic Rank
    rank_parent = evaluator.add_parallel(
        id="Academic_Rank",
        desc="Verify the faculty member holds the rank of Associate Professor or Professor",
        parent=basic,
        critical=True
    )

    # Rank_Reference: ensure rank URL exists (ODU domain or using affiliation page)
    chosen_rank_url = _rank_source_url(data)
    rank_ref_ok = evaluator.add_custom_node(
        result=_is_odu_domain(chosen_rank_url),
        id="Rank_Reference",
        desc="Provide a URL confirming the faculty member's academic rank",
        parent=rank_parent,
        critical=True
    )

    # Rank_Status: verify allowed rank
    rank_status_leaf = evaluator.add_leaf(
        id="Rank_Status",
        desc="Confirm the faculty member's rank is Associate Professor or Professor (not Assistant Professor or Lecturer)",
        parent=rank_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The webpage shows that '{data.full_name or 'the faculty member'}' holds the academic rank of Associate Professor or Professor (and not Assistant Professor or Lecturer).",
        node=rank_status_leaf,
        sources=chosen_rank_url,
        additional_instruction="Accept variants like 'Associate Professor of Computer Science', 'Professor of Computer Science', or 'Tenured Professor'. If the page clearly says Assistant Professor, Lecturer, or similar, this should fail."
    )

    # --------------------------- Research Qualifications ----------------- #
    research = evaluator.add_parallel(
        id="Research_Qualifications",
        desc="Verify the faculty member's research profile and recent activity meet specified criteria",
        parent=top,
        critical=True
    )

    # Research Profile
    profile = evaluator.add_parallel(
        id="Research_Profile",
        desc="Verify the faculty member's research profile meets specified criteria",
        parent=research,
        critical=True
    )

    # Research Area
    area_parent = evaluator.add_parallel(
        id="Research_Area",
        desc="Verify the faculty member has published research in at least one specified area",
        parent=profile,
        critical=True
    )

    # Area_Reference: presence of at least one area evidence URL (scholar_url qualifies)
    area_ref_ok = evaluator.add_custom_node(
        result=(len(data.area_evidence_urls) > 0) or _non_empty(data.google_scholar_url),
        id="Area_Reference",
        desc="Provide a URL showing research work in the specified area(s)",
        parent=area_parent,
        critical=True
    )

    # Area_Verification
    area_leaf = evaluator.add_leaf(
        id="Area_Verification",
        desc="Confirm research publications in at least one specified area",
        parent=area_parent,
        critical=True
    )
    allowed_areas = [
        "Artificial Intelligence", "AI",
        "Machine Learning", "ML",
        "Cybersecurity", "Security",
        "Data Analytics", "Data Mining",
        "Human-Computer Interaction", "HCI"
    ]
    claimed_areas_str = ", ".join(data.research_areas_claimed) if data.research_areas_claimed else "N/A"
    await evaluator.verify(
        claim=f"The researcher has publications in at least one of these areas: Artificial Intelligence (AI), Machine Learning (ML), Cybersecurity, Data Analytics (including Data Mining), or Human-Computer Interaction (HCI).",
        node=area_leaf,
        sources=_collect_sources(data.google_scholar_url, extra_lists=[data.area_evidence_urls]),
        additional_instruction=f"Check publication titles, keywords, topics, or research statements on the provided pages. Accept reasonable synonyms. Areas explicitly claimed in the answer: {claimed_areas_str}. Passing requires evidence for at least one allowed area."
    )

    # Publication Metrics
    metrics_parent = evaluator.add_parallel(
        id="Publication_Metrics",
        desc="Verify the faculty member's publication metrics on Google Scholar",
        parent=profile,
        critical=True
    )

    # Metrics_Reference: scholar URL existence and format
    metrics_ref_ok = evaluator.add_custom_node(
        result=_is_scholar_url(data.google_scholar_url),
        id="Metrics_Reference",
        desc="Provide the Google Scholar profile URL showing the metrics",
        parent=metrics_parent,
        critical=True
    )

    # Total_Publications >= 15
    total_pubs_leaf = evaluator.add_leaf(
        id="Total_Publications",
        desc="Verify the faculty member has at least 15 publications listed on Google Scholar",
        parent=metrics_parent,
        critical=True
    )
    await evaluator.verify(
        claim="The Google Scholar profile lists at least 15 publications for this researcher.",
        node=total_pubs_leaf,
        sources=data.google_scholar_url,
        additional_instruction="Use the publications list on the profile. If the visible list clearly shows 15 or more items (possibly across multiple pages), pass. If unclear or fewer than 15 are evident, fail."
    )

    # H-Index >= 20
    hindex_leaf = evaluator.add_leaf(
        id="H_Index",
        desc="Verify the faculty member has an h-index of at least 20",
        parent=metrics_parent,
        critical=True
    )
    await evaluator.verify(
        claim="The h-index on the Google Scholar profile is at least 20.",
        node=hindex_leaf,
        sources=data.google_scholar_url,
        additional_instruction="On the Scholar profile, check the 'h-index' metric (All or Since-year are both acceptable, prefer 'All' if both are shown). Pass if h-index ≥ 20."
    )

    # Scholar profile verified email (duplicate check under metrics)
    scholar_verified_leaf = evaluator.add_leaf(
        id="Scholar_Profile_Verified",
        desc="Verify the Google Scholar profile is publicly accessible with @odu.edu or @cs.odu.edu email",
        parent=metrics_parent,
        critical=True
    )
    await evaluator.verify(
        claim="The Google Scholar profile shows a 'Verified email at odu.edu' or 'Verified email at cs.odu.edu'.",
        node=scholar_verified_leaf,
        sources=data.google_scholar_url,
        additional_instruction="Confirm the 'Verified email' indicator on the profile reflects an odu.edu or cs.odu.edu domain."
    )

    # Recent Activity
    recent_parent = evaluator.add_parallel(
        id="Recent_Activity",
        desc="Verify the faculty member's recent research activity and collaboration",
        parent=research,
        critical=True
    )

    # Recent Publications
    recent_pubs_parent = evaluator.add_parallel(
        id="Recent_Publications",
        desc="Verify the faculty member published at least 3 papers between 2020-2024",
        parent=recent_parent,
        critical=True
    )

    # Recent_Pubs_Reference: at least one URL to check recency (scholar_url acceptable)
    rec_ref_ok = evaluator.add_custom_node(
        result=(len(data.recent_publications_urls) > 0) or _non_empty(data.google_scholar_url),
        id="Recent_Pubs_Reference",
        desc="Provide a URL (Google Scholar or publication list) showing recent publications",
        parent=recent_pubs_parent,
        critical=True
    )

    # Publication_Count_2020_2024
    pubs_2020_2024_leaf = evaluator.add_leaf(
        id="Publication_Count_2020_2024",
        desc="Confirm at least 3 publications in the 2020-2024 period",
        parent=recent_pubs_parent,
        critical=True
    )
    await evaluator.verify(
        claim="Between 2020 and 2024 (inclusive), the researcher has at least 3 publications.",
        node=pubs_2020_2024_leaf,
        sources=_collect_sources(data.google_scholar_url, extra_lists=[data.recent_publications_urls]),
        additional_instruction="On Google Scholar (or equivalent list), count distinct publications with years 2020, 2021, 2022, 2023, or 2024. If at least 3 are visible or clearly indicated, pass."
    )

    # Collaboration Pattern
    collab_parent = evaluator.add_parallel(
        id="Collaboration_Pattern",
        desc="Verify the faculty member has collaborated with at least 5 distinct co-authors in publications from 2020-2024",
        parent=recent_parent,
        critical=True
    )

    # Collaboration_Reference: presence of URL to inspect coauthors (scholar_url acceptable)
    collab_ref_ok = evaluator.add_custom_node(
        result=(len(data.collaboration_urls) > 0) or _non_empty(data.google_scholar_url),
        id="Collaboration_Reference",
        desc="Provide a URL showing co-authorship information for recent publications",
        parent=collab_parent,
        critical=True
    )

    # Distinct_Coauthors
    coauthors_leaf = evaluator.add_leaf(
        id="Distinct_Coauthors",
        desc="Confirm at least 5 different co-authors in 2020-2024 publications",
        parent=collab_parent,
        critical=True
    )
    await evaluator.verify(
        claim="Between 2020 and 2024 (inclusive), the researcher's publications include at least 5 distinct co-authors (excluding the researcher).",
        node=coauthors_leaf,
        sources=_collect_sources(data.google_scholar_url, extra_lists=[data.collaboration_urls]),
        additional_instruction="Inspect publication entries for 2020-2024 and count unique co-author names (do not count the researcher). Pass if 5 or more distinct co-authors appear."
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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregator; actual top-level critical node added under root
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

    # Extract structured information from the answer
    extracted: FacultyExtraction = await evaluator.extract(
        prompt=prompt_extract_faculty(),
        template_class=FacultyExtraction,
        extraction_name="faculty_candidate_extraction"
    )

    # Build tree and run verifications
    await build_and_verify_tree(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()