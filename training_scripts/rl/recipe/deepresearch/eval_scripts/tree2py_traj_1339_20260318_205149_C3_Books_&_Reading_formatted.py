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
TASK_ID = "novel_adaptation_publication_year"
TASK_DESCRIPTION = """
A contemporary film director has adapted (or is currently adapting) at least three different novels into feature films.
Among the authors of these adapted novels, one author is deceased and is not American.
What is the original publication year of the novel by this deceased non-American author that the director adapted into film?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AdaptedNovelItem(BaseModel):
    novel_title: Optional[str] = None
    author_name: Optional[str] = None
    adaptation_reference_urls: List[str] = Field(default_factory=list)  # URLs that confirm the adaptation of this novel


class DirectorBlock(BaseModel):
    director_name: Optional[str] = None
    adapted_novels: List[AdaptedNovelItem] = Field(default_factory=list)  # Only novels, no plays/screenplays/originals
    director_reference_urls: List[str] = Field(default_factory=list)  # URLs confirming the director's novel adaptations


class DeceasedAuthorBlock(BaseModel):
    author_name: Optional[str] = None
    nationality: Optional[str] = None  # e.g., "British", "French"; must not be "American"
    nationality_reference_urls: List[str] = Field(default_factory=list)  # URLs confirming nationality
    death_year: Optional[str] = None  # numeric year as string if possible
    death_reference_urls: List[str] = Field(default_factory=list)  # URLs confirming death year


class TargetNovelBlock(BaseModel):
    novel_title: Optional[str] = None  # The specific novel by the deceased non-American author, adapted by the director
    adaptation_reference_urls: List[str] = Field(default_factory=list)  # URLs confirming this specific adaptation


class PublicationBlock(BaseModel):
    publication_year: Optional[str] = None  # Original publication year of the target novel
    publication_reference_urls: List[str] = Field(default_factory=list)  # URLs confirming original publication year


class TaskExtraction(BaseModel):
    director: Optional[DirectorBlock] = None
    deceased_author: Optional[DeceasedAuthorBlock] = None
    target_novel: Optional[TargetNovelBlock] = None
    publication: Optional[PublicationBlock] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_task() -> str:
    return """
    You must extract structured information from the answer to support verifying:
    1) a contemporary film director who has adapted at least three different novels into feature films (completed or in active production),
    2) among those adapted-novel authors, one author who is deceased and is not American,
    3) the specific novel by that deceased non-American author that the director adapted (or is adapting),
    4) the original publication year of that specific novel.

    STRICT RULES:
    - Only count/admit "novel" adaptations (exclude plays, short stories, novellas if the answer differentiates; exclude original screenplays).
    - For each requested URL list, include only URLs explicitly present in the answer text (including markdown links).
    - Do not invent any URLs or facts not in the answer.
    - If multiple candidates exist, choose ONE coherent set (one director, one deceased non-American author among that director's adapted novel authors, and one specific adapted novel by that author) that is best supported by the answer's provided URLs.
    - Prefer full 4-digit years for publication/death if provided; otherwise return the string as written in the answer.

    OUTPUT FIELDS:
    director:
      - director_name: full name of the director mentioned in the answer.
      - adapted_novels: an array of unique novel-based film adaptations the answer attributes to this director; for each item include:
          * novel_title
          * author_name
          * adaptation_reference_urls: URLs in the answer that confirm this specific novel adaptation
      - director_reference_urls: URLs in the answer that broadly confirm the director's novel adaptations (e.g., filmography pages, interviews, announcements). Include all such URLs if present.

    deceased_author:
      - author_name: the name of the deceased, non-American author among the director's adapted-novel authors.
      - nationality: their nationality (must NOT be American).
      - nationality_reference_urls: URLs in the answer confirming this nationality.
      - death_year: the year of death (as presented in the answer; ideally 4-digit).
      - death_reference_urls: URLs confirming the death year.

    target_novel:
      - novel_title: the exact title of the novel by the deceased non-American author that the director adapted (or is adapting).
      - adaptation_reference_urls: URLs confirming THIS specific novel was adapted (or is being adapted) by the identified director.

    publication:
      - publication_year: the year the target novel was first published (as presented in the answer; ideally 4-digit).
      - publication_reference_urls: URLs confirming the novel's original publication year.

    Return null for any field the answer does not provide. Do not guess.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _unique_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if _non_empty_str(u):
            u2 = u.strip()
            if u2 not in seen:
                seen.add(u2)
                out.append(u2)
    return out


def _flatten_urls(list_of_lists: List[List[str]]) -> List[str]:
    combo = []
    for lst in list_of_lists:
        if lst:
            combo.extend(lst)
    return _unique_urls(combo)


def _first_k(items: List[Any], k: int) -> List[Any]:
    return items[:k] if items else []


def _get_adaptation_sources_for_title(extracted: TaskExtraction, title: Optional[str]) -> List[str]:
    if not extracted or not extracted.director or not extracted.director.adapted_novels or not _non_empty_str(title):
        return []
    title_lower = title.strip().lower()
    for it in extracted.director.adapted_novels:
        if it.novel_title and it.novel_title.strip().lower() == title_lower:
            return _unique_urls(it.adaptation_reference_urls)
    return []


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_director_identification(evaluator: Evaluator, parent, ex: TaskExtraction) -> None:
    node = evaluator.add_parallel(
        id="director_identification",
        desc="Identify a film director who has adapted at least three different novels (not plays, screenplays, or original stories) into films or is currently in production for such adaptations",
        parent=parent,
        critical=True
    )

    director_name = ex.director.director_name if ex and ex.director else None
    adapted_novels = ex.director.adapted_novels if ex and ex.director else []
    director_urls = ex.director.director_reference_urls if ex and ex.director else []

    # director_name existence (critical)
    evaluator.add_custom_node(
        result=_non_empty_str(director_name),
        id="director_name",
        desc="Provide the full name of the director",
        parent=node,
        critical=True
    )

    # reference_url existence (critical due to parent critical rule)
    evaluator.add_custom_node(
        result=(len(_unique_urls(director_urls)) > 0),
        id="reference_url",
        desc="Provide a URL reference that confirms the director's novel adaptations",
        parent=node,
        critical=True
    )

    # adaptation_count verification (critical)
    adapt_count_leaf = evaluator.add_leaf(
        id="adaptation_count",
        desc="Verify that the director has adapted at least three novels",
        parent=node,
        critical=True
    )

    # Build claim and sources
    titles = [it.novel_title for it in adapted_novels if _non_empty_str(it.novel_title)]
    unique_titles = []
    seen = set()
    for t in titles:
        t2 = t.strip()
        if t2.lower() not in seen:
            seen.add(t2.lower())
            unique_titles.append(t2)

    top3 = _first_k(unique_titles, 3)
    examples_str = "; ".join(top3) if top3 else "N/A"

    claim = (
        f"{director_name} has adapted at least three different novels into feature films "
        f"(either released or currently in production). "
        f"Examples include: {examples_str}."
    )

    # Combine sources: director-wide refs + any adaptation refs (useful if the director-wide page is weak)
    adaptation_urls = _flatten_urls([it.adaptation_reference_urls for it in adapted_novels])
    combined_sources = _unique_urls(director_urls + adaptation_urls)

    await evaluator.verify(
        claim=claim,
        node=adapt_count_leaf,
        sources=combined_sources if combined_sources else None,
        additional_instruction=(
            "Confirm 'novel' adaptations only (exclude plays, short stories, novellas if differentiated, and original screenplays). "
            "A single authoritative page (e.g., filmography, profile, or major interview/announcement) that explicitly or clearly implies "
            "three or more novel-based adaptations is sufficient. If the page indicates fewer than three or does not support the claim, mark as not supported."
        )
    )


async def verify_deceased_author_identification(evaluator: Evaluator, parent, ex: TaskExtraction) -> None:
    node = evaluator.add_parallel(
        id="deceased_author_identification",
        desc="Among the novels adapted by the identified director, identify which author is deceased and is not American",
        parent=parent,
        critical=True
    )

    # Subgroup: author_nationality_verification
    nat_node = evaluator.add_parallel(
        id="author_nationality_verification",
        desc="Verify the nationality of the identified deceased author to confirm they are not American",
        parent=node,
        critical=True
    )

    author_name = ex.deceased_author.author_name if ex and ex.deceased_author else None
    author_nat = ex.deceased_author.nationality if ex and ex.deceased_author else None
    nat_urls = ex.deceased_author.nationality_reference_urls if ex and ex.deceased_author else []

    # author_name existence
    evaluator.add_custom_node(
        result=_non_empty_str(author_name),
        id="author_name",
        desc="Provide the full name of the deceased non-American author",
        parent=nat_node,
        critical=True
    )

    # nationality_reference_url existence
    evaluator.add_custom_node(
        result=(len(_unique_urls(nat_urls)) > 0),
        id="nationality_reference_url",
        desc="Provide a URL reference confirming the author's nationality",
        parent=nat_node,
        critical=True
    )

    # author_nationality verification
    author_nat_leaf = evaluator.add_leaf(
        id="author_nationality",
        desc="State the nationality of the author",
        parent=nat_node,
        critical=True
    )
    nat_claim = (
        f"{author_name} is {author_nat} and is not American (not a United States national/citizen)."
    )
    await evaluator.verify(
        claim=nat_claim,
        node=author_nat_leaf,
        sources=_unique_urls(nat_urls) if nat_urls else None,
        additional_instruction=(
            "Verify the author's nationality as stated and ensure they are not American. "
            "If multiple nationalities are listed, ensure that 'American' is not one of them. "
            "Minor name/casing variations should be treated as the same person."
        )
    )

    # Subgroup: death_verification
    death_node = evaluator.add_parallel(
        id="death_verification",
        desc="Verify that the identified author is deceased",
        parent=node,
        critical=True
    )

    death_year = ex.deceased_author.death_year if ex and ex.deceased_author else None
    death_urls = ex.deceased_author.death_reference_urls if ex and ex.deceased_author else []

    # death_reference_url existence
    evaluator.add_custom_node(
        result=(len(_unique_urls(death_urls)) > 0),
        id="death_reference_url",
        desc="Provide a URL reference confirming the author's death year",
        parent=death_node,
        critical=True
    )

    # death_year verification
    death_year_leaf = evaluator.add_leaf(
        id="death_year",
        desc="Provide the year of the author's death",
        parent=death_node,
        critical=True
    )
    death_claim = f"{author_name} died in {death_year}."
    await evaluator.verify(
        claim=death_claim,
        node=death_year_leaf,
        sources=_unique_urls(death_urls) if death_urls else None,
        additional_instruction="Verify that the page explicitly states the author's year of death."
    )


async def verify_novel_identification(evaluator: Evaluator, parent, ex: TaskExtraction) -> None:
    node = evaluator.add_parallel(
        id="novel_identification",
        desc="Identify which novel by the deceased non-American author was adapted by the director",
        parent=parent,
        critical=True
    )

    director_name = ex.director.director_name if ex and ex.director else None
    author_name = ex.deceased_author.author_name if ex and ex.deceased_author else None
    target_title = ex.target_novel.novel_title if ex and ex.target_novel else None

    # novel_title existence
    evaluator.add_custom_node(
        result=_non_empty_str(target_title),
        id="novel_title",
        desc="Provide the exact title of the novel",
        parent=node,
        critical=True
    )

    # adaptation_reference_url existence
    target_adapt_urls = _unique_urls(ex.target_novel.adaptation_reference_urls) if ex and ex.target_novel else []
    evaluator.add_custom_node(
        result=(len(target_adapt_urls) > 0),
        id="adaptation_reference_url",
        desc="Provide a URL reference confirming the director's adaptation of this specific novel",
        parent=node,
        critical=True
    )

    # adaptation_confirmation verification
    adapt_leaf = evaluator.add_leaf(
        id="adaptation_confirmation",
        desc="Confirm that this novel was adapted (or is being adapted) by the identified director",
        parent=node,
        critical=True
    )

    # If target block lacks URLs, try to supplement from director.adapted_novels matching the title
    if not target_adapt_urls:
        supplemental = _get_adaptation_sources_for_title(ex, target_title)
        target_adapt_urls = _unique_urls(supplemental)

    adapt_claim = (
        f"{director_name} adapted (or is adapting) the novel '{target_title}' by {author_name} into a feature film."
    )
    await evaluator.verify(
        claim=adapt_claim,
        node=adapt_leaf,
        sources=target_adapt_urls if target_adapt_urls else None,
        additional_instruction=(
            "Treat in-production, announced, or completed feature films as valid adaptations. "
            "Accept phrasing such as 'based on the novel ...' as confirmation. "
            "The film title may differ from the novel title; focus on confirming that the director is adapting/adapted this specific novel."
        )
    )


async def verify_publication_year(evaluator: Evaluator, parent, ex: TaskExtraction) -> None:
    node = evaluator.add_parallel(
        id="publication_year_extraction",
        desc="Determine the original publication year of the identified novel",
        parent=parent,
        critical=True
    )

    author_name = ex.deceased_author.author_name if ex and ex.deceased_author else None
    target_title = ex.target_novel.novel_title if ex and ex.target_novel else None
    pub_year = ex.publication.publication_year if ex and ex.publication else None
    pub_urls = _unique_urls(ex.publication.publication_reference_urls) if ex and ex.publication else []

    # publication_reference_url existence
    evaluator.add_custom_node(
        result=(len(pub_urls) > 0),
        id="publication_reference_url",
        desc="Provide a URL reference confirming the novel's publication year",
        parent=node,
        critical=True
    )

    # publication_year verification
    pub_leaf = evaluator.add_leaf(
        id="publication_year",
        desc="Provide the year the novel was first published",
        parent=node,
        critical=True
    )
    pub_claim = f"The novel '{target_title}' by {author_name} was first published in {pub_year}."
    await evaluator.verify(
        claim=pub_claim,
        node=pub_leaf,
        sources=pub_urls if pub_urls else None,
        additional_instruction="Verify the original (first) publication year of the novel. If multiple editions exist, the first publication year should be used."
    )


async def verify_timeline(evaluator: Evaluator, parent, ex: TaskExtraction) -> None:
    # Final sequential check: publication year occurs before death year
    tl_leaf = evaluator.add_leaf(
        id="timeline_verification",
        desc="Verify that the novel was published before the author's death",
        parent=parent,
        critical=True
    )

    author_name = ex.deceased_author.author_name if ex and ex.deceased_author else None
    pub_year = ex.publication.publication_year if ex and ex.publication else None
    death_year = ex.deceased_author.death_year if ex and ex.deceased_author else None

    claim = (
        f"The novel was published in {pub_year}, which was before {author_name}'s death year {death_year}."
    )
    await evaluator.verify(
        claim=claim,
        node=tl_leaf,
        sources=None,
        additional_instruction="Pure logical check: compare the two years numerically. Treat equality (same year) as NOT before."
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
    Evaluate an answer for the 'novel_adaptation_publication_year' task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Enforce task order
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

    # 1) Extract structured info from the answer
    extracted: TaskExtraction = await evaluator.extract(
        prompt=prompt_extract_task(),
        template_class=TaskExtraction,
        extraction_name="structured_task_extraction"
    )

    # 2) Build verification tree according to rubric
    #    All nodes are set to critical to satisfy the framework requirement that critical parents must have all-critical children.
    await verify_director_identification(evaluator, root, extracted)
    await verify_deceased_author_identification(evaluator, root, extracted)
    await verify_novel_identification(evaluator, root, extracted)
    await verify_publication_year(evaluator, root, extracted)
    await verify_timeline(evaluator, root, extracted)

    # 3) Return evaluation summary
    return evaluator.get_summary()