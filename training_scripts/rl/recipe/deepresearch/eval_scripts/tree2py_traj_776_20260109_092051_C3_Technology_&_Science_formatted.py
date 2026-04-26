import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "hopfield_1982_nobel"
TASK_DESCRIPTION = (
    "One of the two scientists who received the 2024 Nobel Prize in Physics for foundational work in machine learning "
    "with artificial neural networks published a seminal paper in 1982 that introduced a neural network model based on associative memory. "
    "Identify this scientist and provide: (1) the complete title of their 1982 paper; (2) the full publication details including the journal name, "
    "volume number, issue number, page range, and publication month/year, along with a reference URL to the paper; (3) the name of the neural network "
    "model introduced in this paper; (4) the type of physical system that this network model was analogized to; and (5) the scientist's institutional "
    "affiliation at the time they received the Nobel Prize in 2024."
)

# Expected Ground Truth (used for guidance and verification claims)
EXPECTED_SCIENTIST = "John J. Hopfield"
EXPECTED_PAPER_TITLE = "Neural networks and physical systems with emergent collective computational abilities"
EXPECTED_JOURNAL = "Proceedings of the National Academy of Sciences"
EXPECTED_VOLUME = "79"
EXPECTED_ISSUE = "8"
EXPECTED_PAGES = "2554–2558"
EXPECTED_MONTH_YEAR = "April 1982"
EXPECTED_MODEL_NAME = "Hopfield network"
EXPECTED_PHYSICAL_ANALOGY = "Atomic spin systems (spin glass/magnetic systems)"
EXPECTED_AFFILIATION_2024 = "Princeton University"


class ExtractionResult(BaseModel):
    scientist_name: Optional[str] = None

    paper_title: Optional[str] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    publication_month_year: Optional[str] = None

    paper_url: Optional[str] = None
    paper_extra_urls: List[str] = Field(default_factory=list)

    model_name: Optional[str] = None
    model_associative_memory_description: Optional[str] = None
    physical_system_analogy: Optional[str] = None
    model_extra_urls: List[str] = Field(default_factory=list)

    affiliation_2024: Optional[str] = None
    affiliation_source_urls: List[str] = Field(default_factory=list)

    nobel_laureate_source_urls: List[str] = Field(default_factory=list)


def prompt_extract_main() -> str:
    return (
        "Extract the following fields exactly as presented in the answer. Only extract information explicitly stated. "
        "If an item is missing in the answer, return null for single fields or an empty array for list fields.\n"
        "Required JSON fields:\n"
        "1) scientist_name: The scientist identified by the answer as the 2024 Nobel Prize in Physics laureate relevant to the 1982 paper.\n"
        "2) paper_title: The exact title of the 1982 paper.\n"
        "3) journal: The journal name stated (e.g., 'Proceedings of the National Academy of Sciences', 'PNAS').\n"
        "4) volume: The volume number as stated (e.g., '79').\n"
        "5) issue: The issue number as stated (e.g., '8').\n"
        "6) pages: The page range as stated (e.g., '2554–2558' or '2554-2558').\n"
        "7) publication_month_year: The publication month/year as stated (e.g., 'April 1982', 'Apr 1982').\n"
        "8) paper_url: A primary reference URL directly linking to the paper or its official page (if provided).\n"
        "9) paper_extra_urls: Any additional URLs in the answer that refer to the paper or its bibliographic entry.\n"
        "10) model_name: The name of the neural network model introduced in the 1982 paper (e.g., 'Hopfield network').\n"
        "11) model_associative_memory_description: A phrase or sentence from the answer describing the model as based on associative memory.\n"
        "12) physical_system_analogy: The physical system analogy described (e.g., 'spin glass', 'magnetic systems', 'atomic spins', 'Ising spins').\n"
        "13) model_extra_urls: Any URLs about the model (e.g., Wikipedia or official references) if provided.\n"
        "14) affiliation_2024: The scientist's institutional affiliation at the time of receiving the 2024 Nobel Prize.\n"
        "15) affiliation_source_urls: URLs in the answer that support the 2024 affiliation.\n"
        "16) nobel_laureate_source_urls: URLs in the answer that support the claim that the scientist is a 2024 Nobel Prize in Physics laureate.\n"
        "Return a single JSON object matching the ExtractionResult schema."
    )


def _collect_urls(primary: Optional[str], extras: List[str]) -> List[str]:
    urls: List[str] = []
    if primary and primary.strip():
        urls.append(primary.strip())
    urls.extend([u for u in extras if isinstance(u, str) and u.strip()])
    return urls


async def build_laureate_identification(evaluator: Evaluator, parent_node, ext: ExtractionResult) -> None:
    node = evaluator.add_sequential(
        id="laureate_identification",
        desc="Identify the correct 2024 Nobel Prize in Physics laureate per constraints",
        parent=parent_node,
        critical=True,
    )

    # Leaf: scientist_name_matches_constraint
    leaf_name = evaluator.add_leaf(
        id="scientist_name_matches_constraint",
        desc=f"Scientist is identified as {EXPECTED_SCIENTIST}",
        parent=node,
        critical=True,
    )
    name_claim = f"The identified scientist '{ext.scientist_name or ''}' refers to the same person as '{EXPECTED_SCIENTIST}'."
    await evaluator.verify(
        claim=name_claim,
        node=leaf_name,
        additional_instruction=(
            "Judge whether the two names refer to the same person. Allow reasonable variations such as missing middle initial, "
            "different punctuation, or minor formatting differences (e.g., 'John Hopfield' vs 'John J. Hopfield')."
        ),
    )

    # Leaf: nobel_laureate_2024_physics_verified
    leaf_nobel = evaluator.add_leaf(
        id="nobel_laureate_2024_physics_verified",
        desc="Scientist is verified to be one of the two 2024 Nobel Prize in Physics laureates",
        parent=node,
        critical=True,
    )
    nobel_claim = (
        f"{EXPECTED_SCIENTIST} is one of the two laureates who received the Nobel Prize in Physics in 2024."
    )
    await evaluator.verify(
        claim=nobel_claim,
        node=leaf_nobel,
        sources=ext.nobel_laureate_source_urls if ext.nobel_laureate_source_urls else None,
        additional_instruction=(
            "Verify the awarding year (2024), the category (Physics), and the presence of the named laureate on the provided source(s). "
            "If multiple laureates are listed, confirm that the person is among the recipients."
        ),
    )


async def build_paper_details_and_contributions(evaluator: Evaluator, parent_node, ext: ExtractionResult) -> None:
    node = evaluator.add_parallel(
        id="paper_details_and_contributions",
        desc="Provide and verify the required 1982 paper details and its contributions per constraints",
        parent=parent_node,
        critical=True,
    )

    paper_urls = _collect_urls(ext.paper_url, ext.paper_extra_urls)
    model_urls = _collect_urls(ext.paper_url, ext.model_extra_urls)

    # Leaf: paper_title_matches_constraint
    leaf_title = evaluator.add_leaf(
        id="paper_title_matches_constraint",
        desc=f"Paper title matches: '{EXPECTED_PAPER_TITLE}'",
        parent=node,
        critical=True,
    )
    title_claim = (
        f"The 1982 paper's title is '{EXPECTED_PAPER_TITLE}'."
    )
    await evaluator.verify(
        claim=title_claim,
        node=leaf_title,
        sources=paper_urls if paper_urls else None,
        additional_instruction=(
            "If a URL is provided, confirm the exact title displayed on the paper's page. Allow minor punctuation or hyphen/en-dash variations, "
            "but the wording must match the canonical title."
        ),
    )

    # Group: publication_details_match_constraints (parallel)
    pub_node = evaluator.add_parallel(
        id="publication_details_match_constraints",
        desc="Publication details match all constraint-specified fields and include a reference URL (month/year check also satisfies the 1982 publication-year constraint)",
        parent=node,
        critical=True,
    )

    # Journal
    leaf_journal = evaluator.add_leaf(
        id="journal_matches_constraint",
        desc=f"Journal name matches: {EXPECTED_JOURNAL} (PNAS)",
        parent=pub_node,
        critical=True,
    )
    journal_claim = f"The journal name is '{EXPECTED_JOURNAL}'."
    await evaluator.verify(
        claim=journal_claim,
        node=leaf_journal,
        sources=paper_urls if paper_urls else None,
        additional_instruction="Allow the abbreviation 'PNAS' to be equivalent to 'Proceedings of the National Academy of Sciences'.",
    )

    # Volume
    leaf_volume = evaluator.add_leaf(
        id="volume_matches_constraint",
        desc=f"Volume number matches: {EXPECTED_VOLUME}",
        parent=pub_node,
        critical=True,
    )
    volume_claim = f"The volume number is {EXPECTED_VOLUME}."
    await evaluator.verify(
        claim=volume_claim,
        node=leaf_volume,
        sources=paper_urls if paper_urls else None,
        additional_instruction="Accept formats such as 'Vol. 79' or '79'.",
    )

    # Issue
    leaf_issue = evaluator.add_leaf(
        id="issue_matches_constraint",
        desc=f"Issue number matches: {EXPECTED_ISSUE}",
        parent=pub_node,
        critical=True,
    )
    issue_claim = f"The issue number is {EXPECTED_ISSUE}."
    await evaluator.verify(
        claim=issue_claim,
        node=leaf_issue,
        sources=paper_urls if paper_urls else None,
        additional_instruction="Accept formats such as 'No. 8', 'Issue 8', or '8'.",
    )

    # Pages
    leaf_pages = evaluator.add_leaf(
        id="pages_match_constraint",
        desc=f"Page range matches: {EXPECTED_PAGES}",
        parent=pub_node,
        critical=True,
    )
    pages_claim = f"The page range is {EXPECTED_PAGES}."
    await evaluator.verify(
        claim=pages_claim,
        node=leaf_pages,
        sources=paper_urls if paper_urls else None,
        additional_instruction="Allow minor hyphen/en-dash variations (e.g., '2554-2558' vs '2554–2558').",
    )

    # Month/Year
    leaf_month_year = evaluator.add_leaf(
        id="publication_month_year_matches_constraint",
        desc=f"Publication month/year matches: {EXPECTED_MONTH_YEAR}",
        parent=pub_node,
        critical=True,
    )
    month_year_claim = f"The publication month and year are {EXPECTED_MONTH_YEAR}."
    await evaluator.verify(
        claim=month_year_claim,
        node=leaf_month_year,
        sources=paper_urls if paper_urls else None,
        additional_instruction="Accept abbreviated month forms like 'Apr 1982'.",
    )

    # Reference URL provided (existence check)
    evaluator.add_custom_node(
        result=bool(paper_urls),
        id="reference_url_provided",
        desc="A reference URL to the paper is provided",
        parent=pub_node,
        critical=True,
    )

    # Group: model_and_analogy_match_constraints (parallel)
    model_node = evaluator.add_parallel(
        id="model_and_analogy_match_constraints",
        desc="Model name, associative-memory basis, and physical analogy match constraints",
        parent=node,
        critical=True,
    )

    # Model name
    leaf_model_name = evaluator.add_leaf(
        id="model_name_matches_constraint",
        desc="Neural network model introduced is identified as the Hopfield network (associative memory / associative network acceptable)",
        parent=model_node,
        critical=True,
    )
    model_name_claim = (
        "The neural network model introduced in the 1982 paper is commonly known as the 'Hopfield network'."
    )
    await evaluator.verify(
        claim=model_name_claim,
        node=leaf_model_name,
        sources=model_urls if model_urls else None,
        additional_instruction=(
            "Confirm that the model is referred to as the 'Hopfield network' (synonyms such as 'Hopfield model' acceptable). "
            "If the paper itself uses descriptive naming rather than an eponym, external references may still identify it as the Hopfield network."
        ),
    )

    # Associative memory basis
    leaf_assoc = evaluator.add_leaf(
        id="associative_memory_basis_stated",
        desc="The model is explicitly stated/described as based on associative memory",
        parent=model_node,
        critical=True,
    )
    assoc_claim = "The model is explicitly based on associative memory (autoassociative/content-addressable memory)."
    await evaluator.verify(
        claim=assoc_claim,
        node=leaf_assoc,
        sources=paper_urls if paper_urls else None,
        additional_instruction="Look for explicit mentions like 'associative memory', 'content-addressable memory', or 'autoassociative'.",
    )

    # Physical system analogy
    leaf_analogy = evaluator.add_leaf(
        id="physical_system_analogy_matches_constraint",
        desc="Physical system analogy is identified as atomic spin systems (spin glass/magnetic systems acceptable)",
        parent=model_node,
        critical=True,
    )
    analogy_claim = (
        "The paper analogizes the network to atomic spin systems, such as an Ising spin system or spin glass in magnetic systems."
    )
    await evaluator.verify(
        claim=analogy_claim,
        node=leaf_analogy,
        sources=paper_urls if paper_urls else None,
        additional_instruction="Accept phrasing like 'spin glass', 'Ising spins', or 'magnetic spin systems' as equivalent.",
    )


async def build_institutional_affiliation(evaluator: Evaluator, parent_node, ext: ExtractionResult) -> None:
    node = evaluator.add_parallel(
        id="institutional_affiliation_2024",
        desc="Provide and verify the scientist's institutional affiliation at the time of receiving the 2024 Nobel Prize",
        parent=parent_node,
        critical=True,
    )

    leaf_affil = evaluator.add_leaf(
        id="affiliation_matches_constraint",
        desc=f"Affiliation matches: {EXPECTED_AFFILIATION_2024} (Princeton University, NJ, USA acceptable)",
        parent=node,
        critical=True,
    )

    affil_sources = ext.affiliation_source_urls if ext.affiliation_source_urls else ext.nobel_laureate_source_urls
    affil_claim = (
        f"At the time of receiving the 2024 Nobel Prize, the scientist's institutional affiliation was {EXPECTED_AFFILIATION_2024}."
    )
    await evaluator.verify(
        claim=affil_claim,
        node=leaf_affil,
        sources=affil_sources if affil_sources else None,
        additional_instruction="Accept 'Princeton University' with location qualifiers (e.g., 'NJ, USA').",
    )


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
    ext: ExtractionResult = await evaluator.extract(
        prompt=prompt_extract_main(),
        template_class=ExtractionResult,
        extraction_name="main_extraction",
    )

    # Add ground truth information to summary
    evaluator.add_ground_truth({
        "expected_scientist": EXPECTED_SCIENTIST,
        "expected_paper_title": EXPECTED_PAPER_TITLE,
        "expected_publication": {
            "journal": EXPECTED_JOURNAL,
            "volume": EXPECTED_VOLUME,
            "issue": EXPECTED_ISSUE,
            "pages": EXPECTED_PAGES,
            "month_year": EXPECTED_MONTH_YEAR,
        },
        "expected_model_name": EXPECTED_MODEL_NAME,
        "expected_physical_analogy": EXPECTED_PHYSICAL_ANALOGY,
        "expected_affiliation_2024": EXPECTED_AFFILIATION_2024,
    })

    # Build the rubric tree under a critical sequential node
    complete_node = evaluator.add_sequential(
        id="complete_investigation",
        desc="Complete investigation matching the question and all provided constraints",
        parent=root,
        critical=True,
    )

    await build_laureate_identification(evaluator, complete_node, ext)
    await build_paper_details_and_contributions(evaluator, complete_node, ext)
    await build_institutional_affiliation(evaluator, complete_node, ext)

    return evaluator.get_summary()