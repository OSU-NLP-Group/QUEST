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
TASK_ID = "stanford_cs_orcid_scholar_oa_faculty"
TASK_DESCRIPTION = (
    "Identify three faculty members from Stanford University's Computer Science department who meet all of the following criteria:\n"
    "1) Have an active ORCID identifier; 2) Have a Google Scholar profile showing an h-index of at least 40; "
    "3) Have at least one research paper available in an open access repository; 4) Are currently affiliated with Stanford CS. "
    "For each, provide their name/title, ORCID and link, Google Scholar link, h-index, and details of at least one open-access paper (title, repository name, direct link)."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ResearchProfiles(BaseModel):
    orcid_id: Optional[str] = None
    orcid_url: Optional[str] = None
    scholar_url: Optional[str] = None
    h_index: Optional[str] = None  # Keep as string to be flexible (e.g., "42", "H-index: 42")


class OAPaper(BaseModel):
    title: Optional[str] = None
    publication_year: Optional[str] = None
    repository_name: Optional[str] = None
    repository_url: Optional[str] = None


class BasicInfo(BaseModel):
    full_name: Optional[str] = None
    faculty_title: Optional[str] = None
    affiliation_source_url: Optional[str] = None  # A URL (ideally Stanford CS page) supporting affiliation


class FacultyItem(BaseModel):
    basic: Optional[BasicInfo] = None
    profiles: Optional[ResearchProfiles] = None
    paper: Optional[OAPaper] = None


class FacultyExtraction(BaseModel):
    faculty: List[FacultyItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_faculty() -> str:
    return (
        "Extract up to the first three faculty members described in the answer who meet the specified criteria. "
        "For each identified faculty member, extract the following structured fields:\n"
        "faculty: [\n"
        "  {\n"
        "    basic: {\n"
        "      full_name: string or null,\n"
        "      faculty_title: string or null,  # current faculty position/title at Stanford CS\n"
        "      affiliation_source_url: string or null  # a URL from the answer that directly supports Stanford CS affiliation (prefer cs.stanford.edu or other stanford.edu pages). If none is present in the answer, set null.\n"
        "    },\n"
        "    profiles: {\n"
        "      orcid_id: string or null,              # the ORCID identifier as written in the answer (e.g., 0000-0002-1825-0097)\n"
        "      orcid_url: string or null,             # link to the ORCID profile (e.g., https://orcid.org/0000-0002-1825-0097)\n"
        "      scholar_url: string or null,           # link to the Google Scholar profile\n"
        "      h_index: string or null                # the h-index value stated in the answer as shown on Google Scholar (extract as-is, do not coerce to number)\n"
        "    },\n"
        "    paper: {\n"
        "      title: string or null,                 # title of at least one open access paper\n"
        "      publication_year: string or null,      # publication year for that paper if provided\n"
        "      repository_name: string or null,       # name of the open access repository (e.g., arXiv, PubMed Central, Stanford Digital Repository)\n"
        "      repository_url: string or null         # direct link to access the paper in the repository\n"
        "    }\n"
        "  }\n"
        "]\n\n"
        "Rules:\n"
        "- Only extract URLs that are explicitly present in the answer text. Do not invent URLs.\n"
        "- If a field is missing in the answer, set it to null.\n"
        "- Preserve the order of appearance from the answer and include at most three faculty items.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _ordinal(idx: int) -> str:
    mapping = {0: "First", 1: "Second", 2: "Third"}
    return mapping.get(idx, f"#{idx+1}")


# --------------------------------------------------------------------------- #
# Verification logic for one faculty member                                   #
# --------------------------------------------------------------------------- #
async def verify_faculty_member(
    evaluator: Evaluator,
    parent_node,
    item: FacultyItem,
    idx: int,
) -> None:
    ordinal = _ordinal(idx)

    # Parent node for the faculty member (non-critical to allow partial credit if fewer than 3)
    fm_node = evaluator.add_parallel(
        id=f"faculty_member_{idx+1}",
        desc=f"{ordinal} qualifying faculty member with complete required information",
        parent=parent_node,
        critical=False,
    )

    # 1) Basic Information and Affiliation
    basic_node = evaluator.add_parallel(
        id=f"fm{idx+1}_basic_information",
        desc="Full name and current faculty position at Stanford CS department provided and affiliation supported",
        parent=fm_node,
        critical=True,  # Critical under this faculty member
    )

    name_exists = _nonempty(item.basic.full_name) if item.basic else False
    title_exists = _nonempty(item.basic.faculty_title) if item.basic else False
    affiliation_src_exists = _nonempty(item.basic.affiliation_source_url) if item.basic else False

    evaluator.add_custom_node(
        result=name_exists,
        id=f"fm{idx+1}_name_provided",
        desc="Full name is provided",
        parent=basic_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=title_exists,
        id=f"fm{idx+1}_title_provided",
        desc="Current faculty position/title is provided",
        parent=basic_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=affiliation_src_exists,
        id=f"fm{idx+1}_affiliation_source_provided",
        desc="Affiliation source URL is provided",
        parent=basic_node,
        critical=True,
    )

    # Verify affiliation with provided source
    aff_leaf = evaluator.add_leaf(
        id=f"fm{idx+1}_affiliation_verified",
        desc="Affiliation with Stanford Computer Science department is supported by the provided source",
        parent=basic_node,
        critical=True,
    )
    aff_claim = (
        f"This webpage indicates that {item.basic.full_name if item and item.basic else 'the person'} "
        "is currently a faculty member in the Stanford University Computer Science department."
    )
    await evaluator.verify(
        claim=aff_claim,
        node=aff_leaf,
        sources=item.basic.affiliation_source_url if item and item.basic else None,
        additional_instruction=(
            "Accept if the page clearly shows affiliation with Stanford University's Department of Computer Science "
            "(e.g., cs.stanford.edu domain, Stanford CS people page, or an official Stanford page stating 'Computer Science'). "
            "Treat the claim as not supported if the page is unrelated or does not show CS department affiliation."
        ),
    )

    # 2) Research Profiles (ORCID + Google Scholar h-index ≥ 40)
    profiles_node = evaluator.add_parallel(
        id=f"fm{idx+1}_research_profiles",
        desc="Valid ORCID identifier/profile and Google Scholar profile with h-index ≥ 40",
        parent=fm_node,
        critical=True,
    )

    orcid_id_exists = _nonempty(item.profiles.orcid_id) if item.profiles else False
    orcid_url_exists = _nonempty(item.profiles.orcid_url) if item.profiles else False
    scholar_url_exists = _nonempty(item.profiles.scholar_url) if item.profiles else False
    hindex_exists = _nonempty(item.profiles.h_index) if item.profiles else False

    evaluator.add_custom_node(
        result=orcid_id_exists,
        id=f"fm{idx+1}_orcid_id_provided",
        desc="ORCID identifier is provided",
        parent=profiles_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=orcid_url_exists,
        id=f"fm{idx+1}_orcid_url_provided",
        desc="ORCID profile URL is provided",
        parent=profiles_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=scholar_url_exists,
        id=f"fm{idx+1}_scholar_url_provided",
        desc="Google Scholar profile URL is provided",
        parent=profiles_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=hindex_exists,
        id=f"fm{idx+1}_hindex_value_provided",
        desc="h-index value is provided",
        parent=profiles_node,
        critical=True,
    )

    # ORCID profile is valid for the person and matches the ORCID ID
    orcid_leaf = evaluator.add_leaf(
        id=f"fm{idx+1}_orcid_profile_valid",
        desc="ORCID profile URL corresponds to the person's ORCID iD and identity",
        parent=profiles_node,
        critical=True,
    )
    orcid_claim = (
        f"This webpage is the ORCID profile for {item.basic.full_name if item and item.basic else 'the person'} "
        f"and shows the ORCID iD {item.profiles.orcid_id if item and item.profiles else ''}."
    )
    await evaluator.verify(
        claim=orcid_claim,
        node=orcid_leaf,
        sources=item.profiles.orcid_url if item and item.profiles else None,
        additional_instruction=(
            "Confirm the page is on orcid.org and shows an ORCID iD matching the provided identifier; "
            "allow minor name variants (e.g., middle initials, diacritics)."
        ),
    )

    # Scholar profile is valid for the person
    scholar_profile_leaf = evaluator.add_leaf(
        id=f"fm{idx+1}_scholar_profile_valid",
        desc="Google Scholar profile URL corresponds to the person",
        parent=profiles_node,
        critical=True,
    )
    scholar_profile_claim = (
        f"This webpage is a Google Scholar profile for {item.basic.full_name if item and item.basic else 'the person'}."
    )
    await evaluator.verify(
        claim=scholar_profile_claim,
        node=scholar_profile_leaf,
        sources=item.profiles.scholar_url if item and item.profiles else None,
        additional_instruction=(
            "The page should be a Google Scholar citations profile page; allow reasonable name variations."
        ),
    )

    # Scholar h-index ≥ 40 (threshold check)
    hindex_leaf = evaluator.add_leaf(
        id=f"fm{idx+1}_scholar_hindex_ge_40",
        desc="h-index on Google Scholar is at least 40",
        parent=profiles_node,
        critical=True,
    )
    hindex_claim = "The h-index displayed on this Google Scholar profile is at least 40."
    await evaluator.verify(
        claim=hindex_claim,
        node=hindex_leaf,
        sources=item.profiles.scholar_url if item and item.profiles else None,
        additional_instruction=(
            "Check the 'Citations' metrics panel on the profile; accept if h-index >= 40. "
            "Allow minor rendering differences; rely on page text or screenshot."
        ),
    )

    # 3) Open Access Paper
    paper_node = evaluator.add_parallel(
        id=f"fm{idx+1}_open_access_paper",
        desc="At least one open access paper details provided and supported",
        parent=fm_node,
        critical=True,
    )

    paper_title_exists = _nonempty(item.paper.title) if item and item.paper else False
    paper_year_exists = _nonempty(item.paper.publication_year) if item and item.paper else False
    repo_name_exists = _nonempty(item.paper.repository_name) if item and item.paper else False
    repo_url_exists = _nonempty(item.paper.repository_url) if item and item.paper else False

    evaluator.add_custom_node(
        result=paper_title_exists,
        id=f"fm{idx+1}_paper_title_provided",
        desc="Open access paper title is provided",
        parent=paper_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=paper_year_exists,
        id=f"fm{idx+1}_paper_year_provided",
        desc="Publication year is provided for the paper",
        parent=paper_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=repo_name_exists,
        id=f"fm{idx+1}_repo_name_provided",
        desc="Open access repository name is provided",
        parent=paper_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=repo_url_exists,
        id=f"fm{idx+1}_repo_url_provided",
        desc="Direct repository link is provided",
        parent=paper_node,
        critical=True,
    )

    # Verify repository page shows the paper title
    repo_title_leaf = evaluator.add_leaf(
        id=f"fm{idx+1}_repo_title_match",
        desc="Repository page shows the claimed paper title",
        parent=paper_node,
        critical=True,
    )
    repo_title_claim = (
        f"This page shows a paper titled '{item.paper.title if item and item.paper else ''}'."
    )
    await evaluator.verify(
        claim=repo_title_claim,
        node=repo_title_leaf,
        sources=item.paper.repository_url if item and item.paper else None,
        additional_instruction=(
            "Confirm the title on the repository page matches the claimed title, allowing minor punctuation or case differences."
        ),
    )

    # Verify repository/platform name matches (by content or domain)
    repo_name_leaf = evaluator.add_leaf(
        id=f"fm{idx+1}_repo_name_match",
        desc="Repository/platform matches the claimed repository name",
        parent=paper_node,
        critical=True,
    )
    repo_name_claim = (
        f"The repository/platform for this page is '{item.paper.repository_name if item and item.paper else ''}'."
    )
    await evaluator.verify(
        claim=repo_name_claim,
        node=repo_name_leaf,
        sources=item.paper.repository_url if item and item.paper else None,
        additional_instruction=(
            "Accept if the page branding or domain clearly corresponds to the claimed repository name "
            "(e.g., arXiv, PubMed Central, Stanford Digital Repository)."
        ),
    )

    # Verify that the page provides open access to the paper
    repo_oa_leaf = evaluator.add_leaf(
        id=f"fm{idx+1}_repo_is_open_access",
        desc="Repository page provides open access to the full text",
        parent=paper_node,
        critical=True,
    )
    repo_oa_claim = "This page provides open access (free public access) to the full text of the paper."
    await evaluator.verify(
        claim=repo_oa_claim,
        node=repo_oa_leaf,
        sources=item.paper.repository_url if item and item.paper else None,
        additional_instruction=(
            "Open access is satisfied if the page clearly offers a free PDF/full text (e.g., arXiv PDF, PMC free full text, institutional repository open item). "
            "If the link is paywalled or only an abstract without free full text, mark as not supported."
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
    # Initialize evaluator; root is non-critical parallel to allow partial score if fewer than 3 members are valid
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify three Stanford CS faculty who have active ORCID, Google Scholar h-index ≥ 40, and at least one open access paper; verify details and links.",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_faculty(),
        template_class=FacultyExtraction,
        extraction_name="faculty_extraction",
    )

    # Keep at most first 3 faculty; pad to 3 if fewer
    items = list(extracted.faculty)[:3] if extracted and extracted.faculty else []
    while len(items) < 3:
        items.append(FacultyItem())

    # Build verification nodes and run checks for each faculty member
    # Use gather for parallel verification of different members for speed
    tasks = []
    for i, item in enumerate(items[:3]):
        tasks.append(verify_faculty_member(evaluator, root, item, i))
    await asyncio.gather(*tasks)

    # Return structured result summary
    return evaluator.get_summary()