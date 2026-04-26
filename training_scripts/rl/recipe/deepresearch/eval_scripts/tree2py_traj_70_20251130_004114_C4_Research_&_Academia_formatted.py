import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ai_conf_paper_2024"
TASK_DESCRIPTION = (
    "Identify a research paper that was accepted at one of the major 2024 artificial intelligence or machine learning "
    "conferences (ICML 2024, NeurIPS 2024, CVPR 2024, or ACL 2024). The paper must have multiple authors, with at least "
    "one author affiliated with a university located in the United States. The paper must be in the computer science or "
    "artificial intelligence research domain and must have been published or accepted in 2024. Provide the paper title, "
    "complete author list with their affiliations, the conference venue, and a reference URL to the official conference "
    "proceedings or paper page."
)

ALLOWED_VENUE_TOKENS = {"icml", "neurips", "cvpr", "acl"}
ALLOWED_AREAS = {"computer vision", "natural language processing", "machine learning theory", "robotics"}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AuthorEntry(BaseModel):
    name: Optional[str] = None
    affiliations: List[str] = Field(default_factory=list)


class PaperExtraction(BaseModel):
    title: Optional[str] = None
    conference: Optional[str] = None  # Expected normalized tokens like "ICML", "NeurIPS", "CVPR", "ACL" if available
    year: Optional[str] = None        # Expected "2024" if mentioned; else null
    official_url: Optional[str] = None
    research_area: Optional[str] = None  # One of the allowed areas if the answer mentions it; else null
    authors: List[AuthorEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_paper() -> str:
    return """
    Extract details for a single research paper described in the answer. Return a JSON object with the following fields:
    - title: The paper title exactly as mentioned. If not provided, return null.
    - conference: The conference venue token string if explicitly provided, choosing one of {"ICML","NeurIPS","CVPR","ACL"}.
                  If the answer provides variants such as "International Conference on Machine Learning", "Neural Information Processing Systems",
                  "IEEE Conference on Computer Vision and Pattern Recognition", or "Association for Computational Linguistics",
                  normalize to {"ICML","NeurIPS","CVPR","ACL"} respectively. If unclear or not provided, return null.
    - year: The acceptance/publish year explicitly mentioned or implied (e.g., "ICML 2024" implies 2024). If not provided, return null.
    - official_url: The URL to the official conference proceedings or official paper page (e.g., neurips.cc, icml.cc / proceedings.mlr.press, openaccess.thecvf.com, aclanthology.org).
                    If the answer provides multiple URLs, select the one most likely to be the official page. If missing, return null.
    - research_area: If the answer mentions the primary research area clearly, map it to one of: "computer vision", "natural language processing",
                     "machine learning theory", or "robotics". Use lowercase. If unclear or not provided, return null.
    - authors: An array of objects for all authors listed in the answer. For each author, extract:
        * name: Author's full name exactly as in the answer (or null if missing).
        * affiliations: A list of affiliation strings as presented (e.g., "University of X", "Company Y"). If none are provided, return an empty list.
    Rules:
    - Extract only information explicitly present in the answer. Do not invent or infer beyond the answer.
    - If the answer includes multiple papers, choose the first one mentioned and extract details for that one.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_venue_token(venue: Optional[str]) -> Optional[str]:
    if not venue:
        return None
    v = venue.strip().lower()
    if "icml" in v or "international conference on machine learning" in v:
        return "ICML"
    if "neurips" in v or "nips" in v or "neural information processing systems" in v:
        return "NeurIPS"
    if "cvpr" in v or "computer vision and pattern recognition" in v:
        return "CVPR"
    if "acl" in v or "association for computational linguistics" in v:
        return "ACL"
    return None


def is_allowed_venue(venue: Optional[str]) -> bool:
    if not venue:
        return False
    token = normalize_venue_token(venue)
    return token is not None and token.lower() in ALLOWED_VENUE_TOKENS


def authors_have_affiliations(authors: List[AuthorEntry]) -> bool:
    if not authors:
        return False
    for a in authors:
        if (a.name is None or not a.name.strip()):
            return False
        # Must include at least one affiliation string for each author
        if not a.affiliations or all((not aff or not aff.strip()) for aff in a.affiliations):
            return False
    return True


def format_authors_affils(authors: List[AuthorEntry]) -> str:
    # Create a concise string summarizing authors and affiliations
    parts = []
    for a in authors:
        name = a.name or "Unknown"
        affs = "; ".join([aff for aff in a.affiliations if aff and aff.strip()]) or "No affiliation"
        parts.append(f"{name} — {affs}")
    return "; ".join(parts) if parts else "No authors listed."


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, parent_node, info: PaperExtraction) -> None:
    """
    Build the verification tree under the critical 'Paper_Identification' node (parallel aggregation),
    and perform verifications according to rubric requirements.
    """
    # Create the main critical node to aggregate all criteria
    paper_node = evaluator.add_parallel(
        id="Paper_Identification",
        desc="Identify one qualifying 2024 AI/ML conference paper and provide the required bibliographic details and official source link.",
        parent=parent_node,
        critical=True
    )

    # Leaf: Provide_Official_Proceedings_URL (existence)
    url_provided = bool(info.official_url and info.official_url.strip())
    official_url_leaf = evaluator.add_custom_node(
        result=url_provided,
        id="Provide_Official_Proceedings_URL",
        desc="The response provides a reference URL to the official conference proceedings or official paper page.",
        parent=paper_node,
        critical=True
    )

    # Leaf: Provide_Paper_Title (existence)
    title_provided = bool(info.title and info.title.strip())
    evaluator.add_custom_node(
        result=title_provided,
        id="Provide_Paper_Title",
        desc="The response provides the paper title.",
        parent=paper_node,
        critical=True
    )

    # Leaf: Provide_Conference_Venue (existence + allowed venue)
    venue_ok = is_allowed_venue(info.conference)
    evaluator.add_custom_node(
        result=venue_ok,
        id="Provide_Conference_Venue",
        desc="The response provides the conference venue (ICML/NeurIPS/CVPR/ACL) corresponding to the identified paper.",
        parent=paper_node,
        critical=True
    )

    # Leaf: Multi_Author_Collaboration (existence: more than one author)
    multi_authors = len(info.authors) > 1
    evaluator.add_custom_node(
        result=multi_authors,
        id="Multi_Author_Collaboration",
        desc="The paper has multiple authors (more than one author).",
        parent=paper_node,
        critical=True
    )

    # Leaf: Provide_Complete_Author_List_With_Affiliations (existence: each author has at least one affiliation)
    authors_complete = authors_have_affiliations(info.authors)
    evaluator.add_custom_node(
        result=authors_complete,
        id="Provide_Complete_Author_List_With_Affiliations",
        desc="The response provides the complete author list and includes an affiliation for each listed author.",
        parent=paper_node,
        critical=True
    )

    # Leaf: US_University_Affiliation_Present (verify with official URL)
    us_affil_leaf = evaluator.add_leaf(
        id="US_University_Affiliation_Present",
        desc="At least one author is affiliated with a university located in the United States.",
        parent=paper_node,
        critical=True
    )
    affil_summary = format_authors_affils(info.authors)
    us_affil_claim = (
        "At least one listed author is affiliated with a U.S. university. "
        f"Here are the affiliations from the answer: {affil_summary}"
    )
    await evaluator.verify(
        claim=us_affil_claim,
        node=us_affil_leaf,
        sources=info.official_url if url_provided else None,
        extra_prerequisites=[official_url_leaf] if url_provided else None,
        additional_instruction=(
            "Use the official paper page to confirm if any author's affiliation corresponds to a United States university. "
            "Allow reasonable identification of U.S. universities by name (e.g., 'Stanford University', 'MIT', 'University of California', etc.). "
            "If the page clearly lists affiliations that are U.S.-based universities, consider this supported."
        )
    )

    # Leaf: Research_Area_Allowed (verify with official URL)
    research_leaf = evaluator.add_leaf(
        id="Research_Area_Allowed",
        desc="The paper's primary research area is one of: computer vision, natural language processing, machine learning theory, or robotics.",
        parent=paper_node,
        critical=True
    )
    if info.research_area and info.research_area.strip():
        area_text = info.research_area.strip().lower()
        area_claim = (
            f"The paper's primary research area is '{area_text}', which belongs to one of the allowed categories: "
            "computer vision, natural language processing, machine learning theory, or robotics."
        )
    else:
        area_claim = (
            "The paper's primary research area is within the allowed categories: computer vision, natural language processing, "
            "machine learning theory, or robotics."
        )
    await evaluator.verify(
        claim=area_claim,
        node=research_leaf,
        sources=info.official_url if url_provided else None,
        extra_prerequisites=[official_url_leaf] if url_provided else None,
        additional_instruction=(
            "Judge based on the official page's title, abstract, keywords, and subject area tags. "
            "Accept close synonyms (e.g., 'vision', 'image recognition' → computer vision; 'NLP', 'language models' → natural language processing; "
            "'theoretical machine learning', 'generalization/optimization theory' → machine learning theory; "
            "'robot learning', 'robot control/manipulation/navigation' → robotics)."
        )
    )

    # Leaf: Conference_Acceptance_2024_List (verify with official URL)
    accept_leaf = evaluator.add_leaf(
        id="Conference_Acceptance_2024_List",
        desc="The paper is accepted at one of: ICML 2024, NeurIPS 2024, CVPR 2024, or ACL 2024.",
        parent=paper_node,
        critical=True
    )
    # Build claim using known venue and year if available
    norm_venue = normalize_venue_token(info.conference)
    year_text = (info.year or "").strip()
    if norm_venue and year_text == "2024":
        accept_claim = f"The paper titled '{info.title or 'Unknown Title'}' was accepted or published at {norm_venue} 2024."
    elif norm_venue:
        accept_claim = (
            f"The paper titled '{info.title or 'Unknown Title'}' was accepted or published at {norm_venue} in 2024."
        )
    else:
        accept_claim = (
            "The paper was accepted or published at one of the following conferences in 2024: ICML 2024, NeurIPS 2024, CVPR 2024, or ACL 2024."
        )
    await evaluator.verify(
        claim=accept_claim,
        node=accept_leaf,
        sources=info.official_url if url_provided else None,
        extra_prerequisites=[official_url_leaf] if url_provided else None,
        additional_instruction=(
            "Verify that the official page clearly indicates the conference (ICML/NeurIPS/CVPR/ACL) and the year 2024 for this paper. "
            "Accept standard variants: 'International Conference on Machine Learning' = ICML, 'Conference on Neural Information Processing Systems' = NeurIPS, "
            "'IEEE Conference on Computer Vision and Pattern Recognition' = CVPR, and 'Association for Computational Linguistics (ACL Annual Meeting)' = ACL."
        )
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
    Evaluate an answer for the 2024 AI/ML conference paper identification task.
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

    # Extract structured information from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_paper(),
        template_class=PaperExtraction,
        extraction_name="paper_extraction"
    )

    # Build verification tree and perform checks
    await build_verification_tree(evaluator, root, extracted_info)

    # Return standardized summary
    return evaluator.get_summary()