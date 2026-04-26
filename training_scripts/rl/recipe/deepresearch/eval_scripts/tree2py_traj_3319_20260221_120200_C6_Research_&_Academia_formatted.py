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
TASK_ID = "pnw_ai_4papers"
TASK_DESCRIPTION = """I am conducting a literature review on recent artificial intelligence and machine learning research from universities in the Pacific Northwest region. Please identify four distinct research papers that ALL satisfy the following criteria:

1. The paper was published at one of these top-tier conferences: NeurIPS (Conference on Neural Information Processing Systems), ICML (International Conference on Machine Learning), or CVPR (IEEE/CVF Conference on Computer Vision and Pattern Recognition)

2. The paper was published in 2024 or 2025

3. At least one author of the paper must be affiliated with a university that is physically located in Washington state or Oregon state

4. The qualifying author (the one with the Pacific Northwest university affiliation) must have a Google Scholar h-index of at least 30

For each of the four papers, provide:
- Paper title
- Conference name and year
- URL to the paper in the conference proceedings or official conference website
- Name of at least one qualifying author (affiliated with a WA or OR university AND has h-index ≥ 30)
- The author's university affiliation
- URL to the author's faculty profile page or institutional webpage
- The author's current h-index value
- URL to the author's Google Scholar profile
- (Optional but appreciated) The name of the author's PhD advisor and the institution where the advisor was/is faculty, along with a supporting reference URL
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AuthorInfo(BaseModel):
    name: Optional[str] = None
    university: Optional[str] = None
    faculty_profile_url: Optional[str] = None
    h_index: Optional[str] = None
    scholar_url: Optional[str] = None

    # Optional academic genealogy info
    advisor_name: Optional[str] = None
    advisor_institution: Optional[str] = None
    advisor_reference_url: Optional[str] = None


class PaperInfo(BaseModel):
    title: Optional[str] = None
    conference_name: Optional[str] = None
    conference_year: Optional[str] = None
    conference_url: Optional[str] = None
    qualifying_author: Optional[AuthorInfo] = None


class PapersExtraction(BaseModel):
    papers: List[PaperInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_papers() -> str:
    return """
Extract up to FOUR distinct research papers exactly as presented in the answer text. For each paper, extract the following fields if they are explicitly present in the answer:

For each paper, return an object with:
- title: The paper title (string)
- conference_name: The conference name (NeurIPS, ICML, or CVPR as written in the answer; allow minor variants like 'Neural Information Processing Systems', 'NIPS', 'IEEE/CVF CVPR', etc.) (string)
- conference_year: The year of the conference/publication (string, e.g., "2024" or "2025"; do not coerce to number)
- conference_url: URL to the paper page on the official conference website or proceedings (e.g., neurips.cc, proceedings.mlr.press, icml.cc, openaccess.thecvf.com, cvpr.thecvf.com, openreview.net for NeurIPS/ICLR-style pages if used by NeurIPS) (string URL)
- qualifying_author: An object describing one author who both (1) is affiliated with a WA or OR university and (2) has Google Scholar h-index ≥ 30. Choose one author that meets both if multiple are listed. This object should include:
  - name: Author's name (string)
  - university: The author's university affiliation (string)
  - faculty_profile_url: A URL to the author's faculty or institutional profile page (string URL)
  - h_index: The author's h-index value as stated in the answer (string, keep as-is; if not present, null)
  - scholar_url: A URL to the author's Google Scholar profile (string URL)
  - advisor_name: (Optional) The PhD advisor's name if provided in the answer (string or null)
  - advisor_institution: (Optional) Institution where the advisor was/is faculty or where the author earned the PhD if provided (string or null)
  - advisor_reference_url: (Optional) A URL that supports the advisor relationship (e.g., faculty page, dissertation acknowledgment, Mathematics Genealogy Project) (string URL or null)

GENERAL RULES:
- Extract only what is explicitly present in the answer. Do not invent or infer missing information.
- If the answer mentions more than four papers, only extract the first four mentioned.
- If fewer than four papers are present, extract as many as available.
- For any missing field, return null for that field.
- Preserve strings as-is (do not normalize or translate).
- For URLs, extract the actual URLs as written in the answer (plain or markdown links).
Return a JSON object with a single key 'papers' that is an array of up to four paper objects in the order they appear.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_nonempty(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


def allowed_conference_note() -> str:
    return ("Allowed top-tier conferences for this task are: NeurIPS, ICML, CVPR. "
            "Accept reasonable variants like 'Neural Information Processing Systems', 'NIPS' for NeurIPS; "
            "'International Conference on Machine Learning' or 'Proceedings of Machine Learning Research' for ICML; "
            "'IEEE/CVF Conference on Computer Vision and Pattern Recognition' for CVPR. "
            "However, the final identification must map to one of NeurIPS, ICML, or CVPR.")


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_paper(evaluator: Evaluator, parent_node, paper: PaperInfo, idx: int) -> None:
    """
    Build and verify the rubric sub-tree for a single paper.
    """
    # Ensure sub-objects are not None
    author = paper.qualifying_author or AuthorInfo()

    # Paper node (parallel, non-critical)
    paper_node = evaluator.add_parallel(
        id=f"paper_{idx+1}",
        desc=f"{['First','Second','Third','Fourth'][idx]} qualifying paper with all required information",
        parent=parent_node,
        critical=False
    )

    # 1) Paper Title (critical existence)
    evaluator.add_custom_node(
        result=is_nonempty(paper.title),
        id=f"paper_{idx+1}_title",
        desc="Paper title is provided",
        parent=paper_node,
        critical=True
    )

    # 2) Conference Venue (critical group)
    conf_node = evaluator.add_parallel(
        id=f"paper_{idx+1}_conference_venue",
        desc="Paper is published at NeurIPS, ICML, or CVPR in 2024 or 2025",
        parent=paper_node,
        critical=True
    )

    # 2.1 Conference Name (critical leaf)
    conf_name_leaf = evaluator.add_leaf(
        id=f"paper_{idx+1}_conference_name",
        desc="Conference is one of: NeurIPS, ICML, or CVPR",
        parent=conf_node,
        critical=True
    )
    conf_name_claim = f"The paper is published at '{paper.conference_name}', which maps to one of NeurIPS, ICML, or CVPR."
    await evaluator.verify(
        claim=conf_name_claim,
        node=conf_name_leaf,
        sources=paper.conference_url,
        additional_instruction=(
            f"{allowed_conference_note()} Use the provided page to confirm the venue; "
            "allow naming variants but ensure the venue is one of the three."
        )
    )

    # 2.2 Conference Year (critical leaf)
    conf_year_leaf = evaluator.add_leaf(
        id=f"paper_{idx+1}_conference_year",
        desc="Conference year is 2024 or 2025",
        parent=conf_node,
        critical=True
    )
    conf_year_claim = f"The paper's conference year is '{paper.conference_year}', and it is either 2024 or 2025."
    await evaluator.verify(
        claim=conf_year_claim,
        node=conf_year_leaf,
        sources=paper.conference_url,
        additional_instruction=(
            "Verify the publication year on the official conference/proceedings page. "
            "Accept if the page indicates 2024 or 2025 for this paper."
        )
    )

    # 2.3 Conference URL validity (critical leaf)
    conf_url_leaf = evaluator.add_leaf(
        id=f"paper_{idx+1}_conference_url",
        desc="URL to conference proceedings or official paper page provided",
        parent=conf_node,
        critical=True
    )
    conf_url_claim = (
        f"This URL is the official conference proceedings or paper page for the paper titled '{paper.title}' "
        f"at {paper.conference_name} {paper.conference_year}."
    )
    await evaluator.verify(
        claim=conf_url_claim,
        node=conf_url_leaf,
        sources=paper.conference_url,
        additional_instruction=(
            "Confirm the page corresponds to the paper (title match or clear equivalence). "
            "Prefer official domains such as neurips.cc, openreview.net (if used by NeurIPS), "
            "proceedings.mlr.press/icml (ICML), icml.cc, and openaccess.thecvf.com or cvpr.thecvf.com (CVPR). "
            "Minor title formatting differences are acceptable."
        )
    )

    # 3) Author Affiliation in WA/OR (critical group)
    aff_node = evaluator.add_parallel(
        id=f"paper_{idx+1}_author_affiliation",
        desc="At least one author is affiliated with a university in Washington or Oregon",
        parent=paper_node,
        critical=True
    )

    # 3.1 Author identified (critical existence)
    evaluator.add_custom_node(
        result=is_nonempty(author.name) and is_nonempty(author.university),
        id=f"paper_{idx+1}_author_identified",
        desc="Author name and institutional affiliation provided",
        parent=aff_node,
        critical=True
    )

    # 3.2 University Location (critical subgroup)
    uni_loc_node = evaluator.add_parallel(
        id=f"paper_{idx+1}_university_location",
        desc="The affiliated university is physically located in Washington or Oregon state",
        parent=aff_node,
        critical=True
    )

    # 3.2.1 University name provided (critical existence)
    evaluator.add_custom_node(
        result=is_nonempty(author.university),
        id=f"paper_{idx+1}_university_name",
        desc="University name provided",
        parent=uni_loc_node,
        critical=True
    )

    # 3.2.2 State verification (critical leaf)
    state_leaf = evaluator.add_leaf(
        id=f"paper_{idx+1}_state_verification",
        desc="University is in WA or OR",
        parent=uni_loc_node,
        critical=True
    )
    state_claim = f"The university '{author.university}' is physically located in the U.S. state of Washington or Oregon."
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=author.faculty_profile_url,
        additional_instruction=(
            "Use the faculty or institutional page to confirm or strongly support that the university is in Washington (WA) or Oregon (OR). "
            "Evidence may include address lines like 'Seattle, WA' or 'Corvallis, OR', or clearly naming 'University of Washington', "
            "'Washington State University', 'Oregon State University', 'University of Oregon', 'Portland State University', etc. "
            "Allow reasonable inference from campus location if explicitly shown on the page."
        )
    )

    # 3.3 Faculty profile URL provided (critical leaf with grounding)
    profile_leaf = evaluator.add_leaf(
        id=f"paper_{idx+1}_faculty_profile_url",
        desc="Link to author's faculty profile or institutional page provided",
        parent=aff_node,
        critical=True
    )
    profile_claim = f"The URL is a faculty/institutional profile page for '{author.name}' at '{author.university}'."
    await evaluator.verify(
        claim=profile_claim,
        node=profile_leaf,
        sources=author.faculty_profile_url,
        additional_instruction=(
            "Verify that the page corresponds to the named author and belongs to the stated university's domain or portal. "
            "People directories or lab pages within the university domain are acceptable."
        )
    )

    # 4) Author H-index (critical group)
    hidx_node = evaluator.add_parallel(
        id=f"paper_{idx+1}_author_hindex",
        desc="The identified author has a Google Scholar h-index of at least 30",
        parent=paper_node,
        critical=True
    )

    # 4.1 H-index value stated and ≥ 30 (critical leaf)
    hidx_value_leaf = evaluator.add_leaf(
        id=f"paper_{idx+1}_hindex_value",
        desc="H-index value stated and is ≥ 30",
        parent=hidx_node,
        critical=True
    )
    hidx_claim = (
        f"According to the author's Google Scholar profile, the h-index is at least 30. "
        f"The answer-stated h-index is '{author.h_index}'."
    )
    await evaluator.verify(
        claim=hidx_claim,
        node=hidx_value_leaf,
        sources=author.scholar_url,
        additional_instruction=(
            "Check the 'h-index' on the Google Scholar profile. Pass if the h-index shown is 30 or higher. "
            "Minor formatting differences in the stated value are acceptable as long as the profile shows ≥ 30."
        )
    )

    # 4.2 Scholar URL provided (critical leaf with grounding)
    scholar_url_leaf = evaluator.add_leaf(
        id=f"paper_{idx+1}_scholar_url",
        desc="Link to author's Google Scholar profile provided",
        parent=hidx_node,
        critical=True
    )
    scholar_url_claim = f"The URL is the Google Scholar profile page of '{author.name}'."
    await evaluator.verify(
        claim=scholar_url_claim,
        node=scholar_url_leaf,
        sources=author.scholar_url,
        additional_instruction=(
            "Confirm the page is a Google Scholar author profile corresponding to the named author "
            "(scholar.google.com/citations...). Minor name variations are acceptable if identity is clear."
        )
    )

    # 5) Academic Genealogy (optional, non-critical, sequential)
    genealogy_node = evaluator.add_sequential(
        id=f"paper_{idx+1}_academic_genealogy",
        desc="PhD advisor information for the identified author is provided",
        parent=paper_node,
        critical=False
    )

    # 5.1 Advisor name provided (non-critical existence)
    evaluator.add_custom_node(
        result=is_nonempty(author.advisor_name),
        id=f"paper_{idx+1}_advisor_name",
        desc="PhD advisor's name is provided",
        parent=genealogy_node,
        critical=False
    )

    # 5.2 Advisor institution provided (non-critical existence)
    evaluator.add_custom_node(
        result=is_nonempty(author.advisor_institution),
        id=f"paper_{idx+1}_advisor_institution",
        desc="Institution where advisor was faculty (or where author earned PhD) is provided",
        parent=genealogy_node,
        critical=False
    )

    # 5.3 Genealogy reference URL supports relationship (non-critical leaf)
    genealogy_leaf = evaluator.add_leaf(
        id=f"paper_{idx+1}_genealogy_url",
        desc="Reference URL supporting the advisor relationship (e.g., faculty page, dissertation acknowledgment, MGP, etc.)",
        parent=genealogy_node,
        critical=False
    )
    genealogy_claim = (
        f"The referenced page supports that '{author.advisor_name}' is an advisor (PhD advisor/supervisor) of '{author.name}'"
        f"{' at ' + author.advisor_institution if is_nonempty(author.advisor_institution) else ''}."
    )
    await evaluator.verify(
        claim=genealogy_claim,
        node=genealogy_leaf,
        sources=author.advisor_reference_url,
        additional_instruction=(
            "Pass if the page explicitly states or strongly implies the advisor-student relationship. "
            "Accept faculty pages, dissertation acknowledgments, institutional bios, or Mathematics Genealogy Project entries."
        )
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
    """
    Evaluate an answer for the 'four Pacific Northwest AI/ML papers' task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root as parallel per rubric
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_papers(),
        template_class=PapersExtraction,
        extraction_name="papers_extraction"
    )

    # Normalize to exactly 4 items (pad with empty placeholders if needed)
    papers: List[PaperInfo] = list(extracted.papers[:4])
    while len(papers) < 4:
        papers.append(PaperInfo())

    # Build verification subtrees for each paper
    for i in range(4):
        await verify_paper(evaluator, root, papers[i], i)

    # Optionally record a custom info about counts
    num_provided = sum(1 for p in papers if is_nonempty(p.title))
    evaluator.add_custom_info({"papers_extracted": len(extracted.papers), "papers_evaluated": 4, "nonempty_titles": num_provided},
                              info_type="extraction_stats")

    return evaluator.get_summary()