import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "hci_researchers_chi2024_top_us_universities"
TASK_DESCRIPTION = """
Identify academic researchers who satisfy ALL of the following criteria:

1. Institutional Affiliation: The researcher must be currently affiliated (as of February 2026) with at least one of the following top U.S. universities recognized for Human-Computer Interaction (HCI) research: Carnegie Mellon University, University of Washington, University of Maryland, Georgia Institute of Technology, Massachusetts Institute of Technology (MIT), or Stanford University. The affiliation must be verifiable through an official university faculty page or profile.

2. CHI 2024 Publication: The researcher must have published at least one paper at the CHI 2024 conference (held May 11-16, 2024 in Honolulu, Hawaii). The paper must appear in the official CHI 2024 proceedings or program.

3. Google Scholar Profile: The researcher must maintain an active, publicly accessible Google Scholar profile that displays their h-index and citation metrics.

4. Research Area: The researcher's work must focus on Human-Computer Interaction or closely related areas (such as accessibility, social computing, human-centered AI, interaction design, or related HCI subfields).

For each researcher you identify, provide:
- Full name
- Primary institutional affiliation and location (city, state)
- Title of at least one paper published at CHI 2024
- Google Scholar profile URL
- University faculty page or profile URL
- Current h-index value from Google Scholar (as of February 2026)

Provide information for at least three (3) different researchers who meet all the above criteria.
"""

ALLOWED_HCI_INSTITUTIONS = [
    "Carnegie Mellon University",
    "CMU",
    "University of Washington",
    "UW",
    "University of Maryland",
    "UMD",
    "Georgia Institute of Technology",
    "Georgia Tech",
    "Massachusetts Institute of Technology",
    "MIT",
    "Stanford University",
    "Stanford",
]

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ResearcherItem(BaseModel):
    """Single researcher info extracted from the answer."""
    full_name: Optional[str] = None
    institution: Optional[str] = None
    location_city: Optional[str] = None
    location_state: Optional[str] = None
    faculty_page_url: Optional[str] = None
    chi_paper_title: Optional[str] = None
    chi_paper_url: Optional[str] = None
    scholar_profile_url: Optional[str] = None
    h_index_value: Optional[str] = None
    research_area_summary: Optional[str] = None


class ResearchersExtraction(BaseModel):
    """All researchers extracted."""
    researchers: List[ResearcherItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_researchers() -> str:
    return """
    Extract up to five researchers mentioned in the answer who match the requested information fields. For each researcher, return a JSON object with the following keys:

    - full_name: The complete name of the researcher.
    - institution: The primary institutional affiliation provided in the answer (e.g., Carnegie Mellon University). Do not invent text.
    - location_city: The city of the institution if explicitly present in the answer; otherwise null.
    - location_state: The state of the institution if explicitly present in the answer; otherwise null.
    - faculty_page_url: The official university faculty page or profile URL (department or lab profile is acceptable if it is official and within the university domain). Return null if not present.
    - chi_paper_title: The title of at least one CHI 2024 paper attributed to the researcher in the answer. If multiple are present, choose the first; otherwise null.
    - chi_paper_url: The official CHI 2024 proceedings or program URL for the paper (e.g., ACM Digital Library 'dl.acm.org' page for CHI '24, or the official CHI 2024 program site 'chi2024.acm.org'). Return null if not present.
    - scholar_profile_url: The Google Scholar profile URL for the researcher (scholar.google.com). Return null if not present.
    - h_index_value: The h-index value stated in the answer (as of February 2026); if absent, return null. Keep it as a string exactly as shown (e.g., "45", "h-index: 45").
    - research_area_summary: A brief summary (as quoted in the answer) of the researcher's area (e.g., "Human-Computer Interaction, accessibility, social computing"). If not present, return null.

    Important guidelines:
    - Extract only what is explicitly present in the answer text. Do NOT infer or invent missing information.
    - For URLs, only return valid ones explicitly shown in the answer (plain URLs or markdown links). If a URL is missing a protocol, prepend "http://".
    - Keep names, titles, and affiliations as written in the answer (preserve casing and punctuation).
    - If any field is missing for a researcher, set it to null.

    Return an object with key "researchers" that is an array of these researcher objects, in the same order they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def ensure_min_researchers(extracted: ResearchersExtraction, k: int = 3) -> List[ResearcherItem]:
    """
    Take the first k researchers; if fewer than k are present, pad with empty placeholders.
    """
    items = list(extracted.researchers[:k])
    while len(items) < k:
        items.append(ResearcherItem())
    return items


def safe_list_urls(*urls: Optional[str]) -> List[str]:
    """Return a list of non-empty URLs."""
    return [u for u in urls if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_researcher_subtree(
    evaluator: Evaluator,
    parent,
    r: ResearcherItem,
    idx: int
) -> None:
    """
    Build verification nodes for a single researcher (parallel aggregation) following the rubric.
    idx is 0-based; display as 1-based.
    """
    display_idx = idx + 1
    rnode = evaluator.add_parallel(
        id=f"Researcher_{display_idx}",
        desc=f"Researcher #{display_idx} meeting all criteria",
        parent=parent,
        critical=False
    )

    # ---------------- Basic Information (Critical) ----------------
    basic_node = evaluator.add_parallel(
        id=f"R{display_idx}_Basic_Information",
        desc="Basic identifying information is provided",
        parent=rnode,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(r.full_name) and bool(str(r.full_name).strip()),
        id=f"R{display_idx}_Full_Name",
        desc="Full name of the researcher is provided",
        parent=basic_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(r.institution) and bool(r.location_city) and bool(r.location_state),
        id=f"R{display_idx}_Affiliation_Location",
        desc="Primary institutional affiliation with city and state is provided",
        parent=basic_node,
        critical=True
    )

    # ---------------- Institutional Affiliation (Critical) ----------------
    inst_node = evaluator.add_parallel(
        id=f"R{display_idx}_Institutional_Affiliation",
        desc="Researcher is affiliated with a required top U.S. HCI university",
        parent=rnode,
        critical=True
    )

    # Leaf: Top HCI institution check (simple logical verification)
    top_inst_leaf = evaluator.add_leaf(
        id=f"R{display_idx}_Top_HCI_Institution",
        desc="Affiliation is with Carnegie Mellon University, University of Washington, University of Maryland, Georgia Institute of Technology, MIT, or Stanford University",
        parent=inst_node,
        critical=True
    )
    aff_str = r.institution or ""
    claim_top_inst = (
        f"The institution '{aff_str}' belongs to the allowed set of top U.S. HCI universities: "
        f"Carnegie Mellon University (CMU), University of Washington (UW), University of Maryland (UMD), "
        f"Georgia Institute of Technology (Georgia Tech), Massachusetts Institute of Technology (MIT), "
        f"and Stanford University."
    )
    await evaluator.verify(
        claim=claim_top_inst,
        node=top_inst_leaf,
        additional_instruction="Allow common abbreviations and synonyms (CMU, UW, UMD, Georgia Tech, MIT, Stanford). Consider sub-department names acceptable if they clearly belong to one of these universities."
    )

    # Leaf: Faculty page URL presence (existence is required by rubric)
    evaluator.add_custom_node(
        result=bool(r.faculty_page_url) and bool(str(r.faculty_page_url).strip()),
        id=f"R{display_idx}_Faculty_Page_URL",
        desc="University faculty page or profile URL is provided to verify affiliation",
        parent=inst_node,
        critical=True
    )

    # ---------------- CHI 2024 Publication (Critical) ----------------
    chi_node = evaluator.add_parallel(
        id=f"R{display_idx}_CHI_2024_Publication",
        desc="Researcher published at least one paper at CHI 2024",
        parent=rnode,
        critical=True
    )

    # Leaf: Paper title provided and matches the CHI page (ground with CHI URL)
    chi_title_leaf = evaluator.add_leaf(
        id=f"R{display_idx}_Paper_Title",
        desc="Title of at least one CHI 2024 paper is provided",
        parent=chi_node,
        critical=True
    )
    chi_title = r.chi_paper_title or ""
    chi_url = r.chi_paper_url or ""
    auth_name = r.full_name or ""
    claim_chi_title = (
        f"The CHI 2024 paper referenced has the title '{chi_title}', and the page shows '{auth_name}' as one of its authors."
    )
    await evaluator.verify(
        claim=claim_chi_title,
        node=chi_title_leaf,
        sources=chi_url if chi_url else None,
        additional_instruction="Verify case-insensitively. Minor punctuation or formatting differences are acceptable. Confirm the author's name appears among the listed authors on the page."
    )

    # Leaf: CHI page belongs to official CHI 2024 proceedings/program
    chi_ref_leaf = evaluator.add_leaf(
        id=f"R{display_idx}_CHI_Paper_Reference",
        desc="ACM Digital Library or CHI 2024 program URL for the paper is provided",
        parent=chi_node,
        critical=True
    )
    claim_chi_ref = (
        "This page is part of the official CHI 2024 proceedings or program (e.g., dl.acm.org showing CHI '24 proceedings, or chi2024.acm.org program page indicating the year 2024)."
    )
    await evaluator.verify(
        claim=claim_chi_ref,
        node=chi_ref_leaf,
        sources=chi_url if chi_url else None,
        additional_instruction="Confirm that the page explicitly indicates CHI 2024 (CHI '24), not a different CHI year or venue."
    )

    # ---------------- Google Scholar (Critical) ----------------
    scholar_node = evaluator.add_parallel(
        id=f"R{display_idx}_Google_Scholar",
        desc="Researcher has an active Google Scholar profile with required information",
        parent=rnode,
        critical=True
    )

    scholar_url = r.scholar_profile_url or ""
    # Leaf: Scholar profile URL - verify that it's a profile for the named researcher and publicly shows metrics
    scholar_profile_leaf = evaluator.add_leaf(
        id=f"R{display_idx}_Scholar_Profile_URL",
        desc="Google Scholar profile URL is provided",
        parent=scholar_node,
        critical=True
    )
    claim_scholar_profile = (
        f"This URL is a Google Scholar profile page for '{auth_name}', publicly accessible and showing citation metrics."
    )
    await evaluator.verify(
        claim=claim_scholar_profile,
        node=scholar_profile_leaf,
        sources=scholar_url if scholar_url else None,
        additional_instruction="The page should be on scholar.google.com and show sections like 'Citations', 'h-index'. Minor name variants are acceptable if clearly the same person."
    )

    # Leaf: h-index value matches what is displayed on Scholar
    hindex_leaf = evaluator.add_leaf(
        id=f"R{display_idx}_H_Index_Value",
        desc="Current h-index value from Google Scholar is provided",
        parent=scholar_node,
        critical=True
    )
    hindex_str = r.h_index_value or ""
    claim_hindex = f"The Google Scholar profile shows an overall h-index value equal to '{hindex_str}'."
    await evaluator.verify(
        claim=claim_hindex,
        node=hindex_leaf,
        sources=scholar_url if scholar_url else None,
        additional_instruction="Prefer the overall h-index, not 'h-index (last 5 years)'. Allow minor formatting differences (e.g., 'h-index: 45' vs '45'). If both are present, match the overall h-index."
    )

    # ---------------- Research Area (Critical) ----------------
    area_leaf = evaluator.add_leaf(
        id=f"R{display_idx}_Research_Area",
        desc="Researcher's work focuses on HCI or closely related areas (accessibility, social computing, human-centered AI, interaction design, etc.)",
        parent=rnode,
        critical=True
    )
    area_claim = (
        f"The research scope of '{auth_name}' is Human-Computer Interaction or closely related areas such as accessibility, social computing, human-centered AI, or interaction design."
    )
    area_sources = safe_list_urls(r.faculty_page_url, r.scholar_profile_url)
    await evaluator.verify(
        claim=area_claim,
        node=area_leaf,
        sources=area_sources if area_sources else None,
        additional_instruction="Use the faculty profile and/or Google Scholar to confirm HCI alignment. Consider departmental context (HCI institute, interactive computing, information school) and publication venues indicative of HCI."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the HCI researchers with CHI 2024 publication and top U.S. university affiliation task.
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
        default_model=model
    )

    # Record allowed institutions for transparency
    evaluator.add_custom_info(
        info={"allowed_institutions": ALLOWED_HCI_INSTITUTIONS},
        info_type="constraints",
        info_name="institution_constraints"
    )

    # Extract researchers from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_researchers(),
        template_class=ResearchersExtraction,
        extraction_name="researchers_extraction"
    )

    # Use first 3 researchers; pad if fewer
    top3 = ensure_min_researchers(extracted, k=3)

    # Build verification subtrees for three researchers (parallel under root)
    build_tasks = []
    for idx, r in enumerate(top3):
        build_tasks.append(build_researcher_subtree(evaluator, root, r, idx))
    # Execute sequentially to maintain deterministic logging order
    for task in build_tasks:
        await task

    # Return evaluation summary
    return evaluator.get_summary()