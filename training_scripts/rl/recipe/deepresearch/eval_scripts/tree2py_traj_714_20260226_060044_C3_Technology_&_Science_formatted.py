import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "aaai26_outstanding_clip_llm"
TASK_DESCRIPTION = (
    "Among the five outstanding paper award winners announced at the AAAI-26 conference (held January 20–27, 2026, "
    "at Singapore EXPO), identify the paper that uses large language models to unlock richer cross-modality "
    "representations in vision-language models, specifically enhancing CLIP. For this paper, provide: "
    "(1) the complete paper title, (2) all authors listed in the order they appear, "
    "(3) the direct arXiv.org preprint URL, (4) which major technology company's research division was recognized "
    "for contributing to this work based on public award announcements, and (5) the total number of authors on the paper."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PaperExtraction(BaseModel):
    paper_title: Optional[str] = None
    arxiv_url: Optional[str] = None
    authors_ordered: List[str] = Field(default_factory=list)
    author_count: Optional[str] = None
    recognized_company: Optional[str] = None

    # Source URL groups (explicitly provided in the answer)
    award_sources: List[str] = Field(default_factory=list)     # AAAI/award announcement pages
    arxiv_sources: List[str] = Field(default_factory=list)     # Pages referencing arXiv preprint
    authors_sources: List[str] = Field(default_factory=list)   # Pages listing authors/order (e.g., arXiv page)
    company_sources: List[str] = Field(default_factory=list)   # Pages recognizing the company division
    general_sources: List[str] = Field(default_factory=list)   # Any other URLs provided


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_paper_info() -> str:
    return """
    You must extract all information the answer provides about the specific AAAI-26 Outstanding Paper that uses large language models to unlock richer cross-modality representations in vision-language models, specifically enhancing CLIP.

    Extract the following fields exactly as they appear in the answer:
    1) paper_title: The complete title of the identified paper.
    2) arxiv_url: The direct arXiv.org preprint URL for this paper (must be a full arXiv URL).
    3) authors_ordered: The complete list of authors in the exact order they appear on the paper.
    4) author_count: The total number of authors as stated in the answer (string; do not convert to integer).
    5) recognized_company: The name of the major technology company's research division recognized for contributing to the work (e.g., 'Google Research', 'Microsoft Research', 'Meta AI', 'Apple', 'OpenAI', etc.), as stated in public award announcements and captured in the answer.

    Also extract any URLs explicitly mentioned in the answer and classify them into these groups:
    - award_sources: URLs that announce or list AAAI-26 Outstanding Paper winners or otherwise confirm this paper won the AAAI-26 Outstanding Paper award.
    - arxiv_sources: URLs that reference or confirm the arXiv preprint URL for the paper (can be the arXiv page itself).
    - authors_sources: URLs that list the authors of the paper (including the arXiv page).
    - company_sources: URLs that explicitly recognize the major company's research division related to this work in the award announcements.
    - general_sources: Any other URLs mentioned in the answer that are relevant but do not fit the above categories.

    IMPORTANT:
    - Extract only what is explicitly present in the answer; do not invent or infer.
    - If a specific field is missing in the answer, set it to null (for strings) or an empty list (for arrays).
    - For all URL fields: include only valid, full URLs. If a URL is missing a protocol, prepend http://
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(*lists: List[str]) -> List[str]:
    """Merge multiple URL lists and deduplicate while preserving order."""
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for url in lst:
            if url and url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


def _safe_sources(sources: List[str]) -> Optional[List[str]]:
    """Return None if sources list is empty to signal no sources; else return list."""
    return sources if sources else None


def _count_from_authors_or_string(authors: List[str], author_count_str: Optional[str]) -> Optional[int]:
    """Derive an integer author count. Prefer authors list length; else parse the provided string."""
    if authors:
        return len(authors)
    if author_count_str:
        try:
            # Extract digits from the string
            digits = "".join(ch for ch in author_count_str if ch.isdigit())
            return int(digits) if digits else None
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_paper_identification_and_preprint(
    evaluator: Evaluator,
    parent_node,
    extracted: PaperExtraction
) -> None:
    """Build and verify 'Paper-Identification-and-Preprint' subtree."""
    node = evaluator.add_parallel(
        id="Paper-Identification-and-Preprint",
        desc="Identify the correct paper and provide its arXiv preprint link",
        parent=parent_node,
        critical=False
    )

    # --- Paper-Selection (critical) ---
    paper_sel = evaluator.add_parallel(
        id="Paper-Selection",
        desc="Correctly identify the AAAI-26 outstanding paper that uses LLMs to unlock richer cross-modality (CLIP) representations, including its complete title",
        parent=node,
        critical=True
    )

    # Existence check: Title provided
    title_provided = evaluator.add_custom_node(
        result=bool(extracted.paper_title and extracted.paper_title.strip()),
        id="paper_title_provided",
        desc="Paper title is provided in the answer",
        parent=paper_sel,
        critical=True
    )

    # Verify paper focus/topic matches the described CLIP enhancement via LLMs
    paper_focus_leaf = evaluator.add_leaf(
        id="paper_focus_matches",
        desc="The identified paper explicitly focuses on using LLMs to unlock richer cross-modality representations in CLIP/vision-language models",
        parent=paper_sel,
        critical=True
    )
    focus_sources = _merge_sources(
        extracted.award_sources,
        extracted.general_sources,
        [extracted.arxiv_url] if extracted.arxiv_url else [],
        extracted.arxiv_sources
    )
    focus_claim = (
        f"The paper titled '{extracted.paper_title or ''}' uses large language models to unlock richer "
        f"cross-modality representations in vision-language models, specifically enhancing CLIP."
    )
    await evaluator.verify(
        claim=focus_claim,
        node=paper_focus_leaf,
        sources=_safe_sources(focus_sources),
        additional_instruction=(
            "Check the provided pages for explicit statements that the paper uses LLMs to enhance CLIP or unlock richer "
            "cross-modality representations in vision-language models. Allow reasonable paraphrases; focus on the core idea."
        )
    )

    # Verify award status via URL reference
    award_ref_leaf = evaluator.add_leaf(
        id="Paper-URL-Reference",
        desc="Provide URL reference confirming this paper won an AAAI-26 outstanding paper award",
        parent=paper_sel,
        critical=True
    )
    award_claim = (
        f"The paper titled '{extracted.paper_title or ''}' won an Outstanding Paper award at AAAI-26 (held January 20–27, 2026, at Singapore EXPO)."
    )
    await evaluator.verify(
        claim=award_claim,
        node=award_ref_leaf,
        sources=_safe_sources(_merge_sources(extracted.award_sources, extracted.general_sources)),
        additional_instruction=(
            "Confirm that the page(s) explicitly list this paper among the AAAI-26 Outstanding Paper award winners. "
            "If the page lists multiple winners, ensure this exact paper is included."
        )
    )

    # --- ArXiv-Availability (critical) ---
    arxiv_node = evaluator.add_parallel(
        id="ArXiv-Availability",
        desc="Provide the complete and direct arXiv.org preprint URL for the paper",
        parent=node,
        critical=True
    )

    # Existence check: arXiv URL provided and looks like arXiv
    arxiv_url_ok = evaluator.add_custom_node(
        result=bool(extracted.arxiv_url and ("arxiv.org" in extracted.arxiv_url)),
        id="arxiv_url_provided",
        desc="arXiv URL is provided and appears to be an arXiv.org link",
        parent=arxiv_node,
        critical=True
    )

    # Verify arXiv URL corresponds to the paper title
    arxiv_ref_leaf = evaluator.add_leaf(
        id="ArXiv-URL-Reference",
        desc="URL reference confirming the arXiv preprint link",
        parent=arxiv_node,
        critical=True
    )
    arxiv_claim = (
        f"The arXiv page at '{extracted.arxiv_url or ''}' corresponds to the paper titled '{extracted.paper_title or ''}'."
    )
    await evaluator.verify(
        claim=arxiv_claim,
        node=arxiv_ref_leaf,
        sources=extracted.arxiv_url if extracted.arxiv_url else None,
        additional_instruction=(
            "Check the title on the arXiv page and confirm it matches or is equivalent to the provided paper title."
        )
    )


async def build_author_information(
    evaluator: Evaluator,
    parent_node,
    extracted: PaperExtraction
) -> None:
    """Build and verify 'Author-Information' subtree."""
    node = evaluator.add_parallel(
        id="Author-Information",
        desc="Extract complete and accurate author information from the paper",
        parent=parent_node,
        critical=False
    )

    # --- Complete-Author-List (critical) ---
    cal_node = evaluator.add_parallel(
        id="Complete-Author-List",
        desc="Provide all authors in the exact order they appear on the paper",
        parent=node,
        critical=True
    )

    # Existence check: authors provided
    authors_provided = evaluator.add_custom_node(
        result=bool(extracted.authors_ordered),
        id="authors_provided",
        desc="Authors list is provided in the answer",
        parent=cal_node,
        critical=True
    )

    # Verify authors and order via sources
    authors_order_leaf = evaluator.add_leaf(
        id="authors_order_accurate",
        desc="Authors are listed completely and in the correct order",
        parent=cal_node,
        critical=True
    )
    author_list_str = ", ".join(extracted.authors_ordered) if extracted.authors_ordered else ""
    authors_claim = (
        f"For the paper titled '{extracted.paper_title or ''}', the complete author list in order is: {author_list_str}."
    )
    authors_sources_all = _merge_sources(extracted.authors_sources, [extracted.arxiv_url] if extracted.arxiv_url else [])
    await evaluator.verify(
        claim=authors_claim,
        node=authors_order_leaf,
        sources=_safe_sources(authors_sources_all),
        additional_instruction=(
            "Confirm that the provided author list fully matches the order shown on the authoritative page (e.g., arXiv). "
            "Allow minor variations in name formatting (middle initials, accents), but the order must match exactly."
        )
    )

    # Additional explicit URL reference leaf
    authors_url_ref_leaf = evaluator.add_leaf(
        id="Authors-URL-Reference",
        desc="URL reference confirming the complete author list and order",
        parent=cal_node,
        critical=True
    )
    authors_url_claim = (
        f"The provided URL(s) explicitly list the complete author list and their order for the paper titled '{extracted.paper_title or ''}'."
    )
    await evaluator.verify(
        claim=authors_url_claim,
        node=authors_url_ref_leaf,
        sources=_safe_sources(authors_sources_all),
        additional_instruction=(
            "At least one provided URL must display the full author list in order (e.g., arXiv page)."
        )
    )

    # --- Author-Count (critical) ---
    ac_node = evaluator.add_parallel(
        id="Author-Count",
        desc="Provide the total number of authors on the paper",
        parent=node,
        critical=True
    )

    # Compute count from authors or provided string
    computed_author_count = _count_from_authors_or_string(extracted.authors_ordered, extracted.author_count)

    # Existence/consistency check: Did we derive a count?
    count_available = evaluator.add_custom_node(
        result=bool(computed_author_count is not None),
        id="author_count_available",
        desc="Author count can be determined from the provided information",
        parent=ac_node,
        critical=True
    )

    # Verify count via sources (arXiv typically suffices)
    count_leaf = evaluator.add_leaf(
        id="author_count_correct",
        desc="Total number of authors is correctly stated",
        parent=ac_node,
        critical=True
    )
    count_claim = (
        f"The paper titled '{extracted.paper_title or ''}' has {computed_author_count if computed_author_count is not None else ''} authors."
    )
    await evaluator.verify(
        claim=count_claim,
        node=count_leaf,
        sources=_safe_sources(authors_sources_all),
        additional_instruction=(
            "Count the authors listed on the authoritative page and confirm it matches the stated total."
        )
    )

    # Reference leaf ensuring URLs allow verification of count
    count_url_ref_leaf = evaluator.add_leaf(
        id="Count-URL-Reference",
        desc="URL reference that allows verification of the author count",
        parent=ac_node,
        critical=True
    )
    count_url_claim = (
        f"The provided URL(s) include the author list for the paper titled '{extracted.paper_title or ''}', enabling verification of the total author count."
    )
    await evaluator.verify(
        claim=count_url_claim,
        node=count_url_ref_leaf,
        sources=_safe_sources(authors_sources_all),
        additional_instruction=(
            "Confirm that at least one provided URL displays the authors clearly enough to count them."
        )
    )

    # Record computed count in custom info
    evaluator.add_custom_info(
        info={"computed_author_count": computed_author_count},
        info_type="computed_metrics",
        info_name="author_count_computed"
    )


async def build_organizational_attribution(
    evaluator: Evaluator,
    parent_node,
    extracted: PaperExtraction
) -> None:
    """Build and verify 'Organizational-Attribution' subtree."""
    node = evaluator.add_parallel(
        id="Organizational-Attribution",
        desc="Identify the major technology company whose research division contributed to this award-winning work",
        parent=parent_node,
        critical=True
    )

    # Existence check: company provided
    company_provided = evaluator.add_custom_node(
        result=bool(extracted.recognized_company and extracted.recognized_company.strip()),
        id="recognized_company_provided",
        desc="Recognized company/research division is provided",
        parent=node,
        critical=True
    )

    # Verify via URL reference
    company_ref_leaf = evaluator.add_leaf(
        id="Company-URL-Reference",
        desc="URL reference confirming which company's researchers were recognized for this work",
        parent=node,
        critical=True
    )
    company_sources_all = _merge_sources(extracted.company_sources, extracted.award_sources, extracted.general_sources)
    company_claim = (
        f"Public award announcement(s) explicitly recognize {extracted.recognized_company or ''} (or its research division) "
        f"as contributing to the paper titled '{extracted.paper_title or ''}' that won an AAAI-26 Outstanding Paper award."
    )
    await evaluator.verify(
        claim=company_claim,
        node=company_ref_leaf,
        sources=_safe_sources(company_sources_all),
        additional_instruction=(
            "Allow variant naming (e.g., 'Google', 'Google Research', 'Google DeepMind'; 'Microsoft', 'Microsoft Research', "
            "'Microsoft Research Asia'; 'Meta', 'Meta AI'). Confirm that the page recognizes the company's research division "
            "as part of the winning work."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the AAAI-26 outstanding paper (CLIP + LLM cross-modality) task.
    """
    # Initialize evaluator and root node
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

    # Top-level investigation node
    investigation_node = evaluator.add_parallel(
        id="AAAI-26-Outstanding-Paper-Investigation",
        desc="Complete investigation of the AAAI-26 outstanding paper that focuses on enhancing vision-language models using large language models",
        parent=root,
        critical=False
    )

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_paper_info(),
        template_class=PaperExtraction,
        extraction_name="paper_extraction"
    )

    # Build subtrees per rubric
    await build_paper_identification_and_preprint(evaluator, investigation_node, extracted)
    await build_author_information(evaluator, investigation_node, extracted)
    await build_organizational_attribution(evaluator, investigation_node, extracted)

    # Return evaluation summary
    return evaluator.get_summary()