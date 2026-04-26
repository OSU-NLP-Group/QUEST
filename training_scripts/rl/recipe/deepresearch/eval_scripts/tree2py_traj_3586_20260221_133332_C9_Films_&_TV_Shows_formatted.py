import asyncio
import logging
import re
from typing import Any, List, Dict, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "emmy_lead_2024_productions"
TASK_DESCRIPTION = (
    "Identify 3 distinct films or television productions that were released or premiered between October 1, 2024, and March 31, 2026, "
    "where each production is directed by a filmmaker who was born between 1960 and 1975 and who has directed at least 4 feature films "
    "or major television series prior to that production, and each production features in a lead or major role an actor or actress who "
    "won a Primetime Emmy Award in a Lead Acting category (Lead Actor/Actress in Drama, Comedy, or Limited Series) in 2024. For each "
    "production, provide the title, director's name, the Emmy-winning actor's name, and reference URLs documenting these facts."
)

DATE_RANGE_START_TEXT = "October 1, 2024"
DATE_RANGE_END_TEXT = "March 31, 2026"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProductionItem(BaseModel):
    # Basic identifying information
    title: Optional[str] = None
    format_type: Optional[str] = None  # e.g., "feature film", "television series", "TV series", "limited series"
    release_date: Optional[str] = None  # a date string as presented in the answer
    release_urls: List[str] = Field(default_factory=list)

    # Director information
    director_name: Optional[str] = None
    director_birth_date: Optional[str] = None  # e.g., "May 12, 1965"
    director_birth_year: Optional[str] = None  # e.g., "1965"
    director_prior_works_count: Optional[str] = None  # e.g., "5", "at least 4"
    director_prior_works_list: List[str] = Field(default_factory=list)  # if the answer lists prior works
    director_urls: List[str] = Field(default_factory=list)

    # Emmy-winning actor information
    emmy_actor_name: Optional[str] = None
    emmy_actor_category: Optional[str] = None  # e.g., "Lead Actor in a Drama Series"
    emmy_actor_year: Optional[str] = None      # should be "2024"
    actor_role_description: Optional[str] = None  # e.g., "lead", "starring", "major role", "main cast"
    actor_urls: List[str] = Field(default_factory=list)

    # Additional sources if provided
    extra_sources: List[str] = Field(default_factory=list)


class ProductionsExtraction(BaseModel):
    productions: List[ProductionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_productions() -> str:
    return """
    Extract all distinct productions mentioned in the answer that are candidates for the task. For each production, return a structured object with the following fields:

    BASIC IDENTIFICATION:
    - title: The production's title as stated in the answer.
    - format_type: The production format as stated, e.g., "feature film", "television series", "TV series", "limited series", etc.
    - release_date: The official release or premiere date as cited in the answer (string; keep the original format).
    - release_urls: An array of URLs explicitly provided in the answer that verify the release/premiere date and/or official production page.

    DIRECTOR:
    - director_name: Full name of the director.
    - director_birth_date: The director's birthdate if provided in the answer (e.g., "May 12, 1965"); else null.
    - director_birth_year: The director's birth year if provided in the answer (e.g., "1965"); else null.
    - director_prior_works_count: The stated count for how many feature films or major television series the director had directed prior to this production (string; keep exactly as stated).
    - director_prior_works_list: A list of titles of prior works if the answer provides them. Else return an empty list.
    - director_urls: An array of URLs explicitly provided in the answer that verify the director's birthdate/year and/or filmography.

    EMMY-WINNING ACTOR:
    - emmy_actor_name: Full name of the actor/actress who won a Primetime Emmy Award in 2024 and is featured in a lead or major role in this production.
    - emmy_actor_category: The exact Emmy category as stated (e.g., "Lead Actor in a Drama Series", "Lead Actress in a Comedy Series", "Lead Actor in a Limited Series or Movie", etc.).
    - emmy_actor_year: The year of the Emmy win; should be stated as "2024" if present.
    - actor_role_description: The role level in this production as described (e.g., "lead", "starring", "major role", "main cast").
    - actor_urls: An array of URLs explicitly provided in the answer that verify the actor's 2024 Primetime Emmy win and/or their role in the production.

    ADDITIONAL:
    - extra_sources: Any other URLs cited in the answer that are relevant for documentation.

    RULES:
    - Only extract information explicitly present in the answer. Do not invent or infer.
    - For any field that is missing, return null (for single values) or an empty list (for arrays).
    - Extract all candidate productions mentioned; we will later filter to the first 3 distinct titles.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_title(t: Optional[str]) -> str:
    if not t:
        return ""
    # Normalize by lowercasing and removing non-alphanumeric characters
    return re.sub(r"[^a-z0-9]+", "", t.lower())


def dedup_and_take_first_k(items: List[ProductionItem], k: int = 3) -> List[ProductionItem]:
    seen = set()
    unique_items: List[ProductionItem] = []
    for it in items:
        key = normalize_title(it.title)
        if not key:
            # still include placeholders with empty title if needed later to pad
            continue
        if key in seen:
            continue
        seen.add(key)
        unique_items.append(it)
        if len(unique_items) >= k:
            break
    # Pad to k with empty items if fewer found
    while len(unique_items) < k:
        unique_items.append(ProductionItem())
    return unique_items


def combine_sources(*lists: List[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for lst in lists:
        for u in lst:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification logic per production                                           #
# --------------------------------------------------------------------------- #
async def verify_one_production(
    evaluator: Evaluator,
    parent_node,
    prod: ProductionItem,
    index_one_based: int,
) -> None:
    """
    Build verification sub-tree for a single production and perform verifications.
    All nodes under this production are marked critical to satisfy the rubric requirement.
    """
    # Create the Production node (critical under Task_Completion)
    prod_node = evaluator.add_parallel(
        id=f"Production_{index_one_based}",
        desc=f"{['First','Second','Third'][index_one_based-1] if index_one_based<=3 else f'Production #{index_one_based}'} identified production meeting all criteria",
        parent=parent_node,
        critical=True,
    )

    # ---------------- Basic Information ----------------
    basic_node = evaluator.add_parallel(
        id=f"Production_{index_one_based}_Basic_Information",
        desc=f"Essential identifying information for Production {index_one_based}",
        parent=prod_node,
        critical=True,
    )

    title_exists = bool(prod.title and prod.title.strip())
    director_name_exists = bool(prod.director_name and prod.director_name.strip())
    actor_name_exists = bool(prod.emmy_actor_name and prod.emmy_actor_name.strip())

    evaluator.add_custom_node(
        result=title_exists,
        id=f"Production_{index_one_based}_Title",
        desc=f"The production's title is clearly stated",
        parent=basic_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=director_name_exists,
        id=f"Production_{index_one_based}_Director_Name",
        desc=f"The director's full name is provided",
        parent=basic_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=actor_name_exists,
        id=f"Production_{index_one_based}_Actor_Name",
        desc=f"The Emmy-winning actor's full name is provided",
        parent=basic_node,
        critical=True,
    )

    # ---------------- Director Criteria ----------------
    director_node = evaluator.add_parallel(
        id=f"Production_{index_one_based}_Director_Criteria",
        desc=f"Director requirements for Production {index_one_based}",
        parent=prod_node,
        critical=True,
    )

    # Existence of director verification URLs (critical prerequisite)
    dir_urls_exist_node = evaluator.add_custom_node(
        result=bool(prod.director_urls),
        id=f"Production_{index_one_based}_Director_Verification_URL",
        desc=f"Reference URL(s) provided to verify director's birth year and filmography",
        parent=director_node,
        critical=True,
    )

    # Director birth year within 1960–1975 and supported by URLs
    birth_leaf = evaluator.add_leaf(
        id=f"Production_{index_one_based}_Director_Birth_Year",
        desc=f"The director was born between January 1, 1960, and December 31, 1975 (inclusive)",
        parent=director_node,
        critical=True,
    )
    birth_claim: str
    if prod.director_name:
        if prod.director_birth_date:
            birth_claim = (
                f"{prod.director_name} was born on {prod.director_birth_date}. "
                f"This birthdate is between January 1, 1960 and December 31, 1975."
            )
        elif prod.director_birth_year:
            birth_claim = (
                f"{prod.director_name} was born in {prod.director_birth_year}. "
                f"This year is between 1960 and 1975."
            )
        else:
            birth_claim = (
                f"The provided sources confirm that {prod.director_name} was born between 1960 and 1975."
            )
    else:
        birth_claim = "The director was born between 1960 and 1975."
    await evaluator.verify(
        claim=birth_claim,
        node=birth_leaf,
        sources=prod.director_urls,
        additional_instruction=(
            "Use the provided source URLs to confirm the director's birthdate or year and ensure it falls within 1960–1975 inclusive. "
            "Minor formatting variations are acceptable; focus on the factual date/year reported by reliable sources."
        ),
        extra_prerequisites=[dir_urls_exist_node],
    )

    # Director has directed >=4 prior feature films or major TV series (source-supported)
    filmography_leaf = evaluator.add_leaf(
        id=f"Production_{index_one_based}_Director_Filmography",
        desc=f"The director has directed at least 4 feature films or major television series prior to this production",
        parent=director_node,
        critical=True,
    )
    filmography_claim = (
        f"{prod.director_name or 'The director'} has directed at least 4 feature films or major television series "
        f"prior to {f'\"{prod.title}\"' if prod.title else 'this production'}."
    )
    await evaluator.verify(
        claim=filmography_claim,
        node=filmography_leaf,
        sources=prod.director_urls,
        additional_instruction=(
            "Verify via the director filmography pages or reliable sources that there are at least four prior directorial credits "
            "(feature films or major TV series) before this production. Ignore producer/writer-only credits. "
            "The claim should be supported by the provided URLs."
        ),
        extra_prerequisites=[dir_urls_exist_node],
    )

    # ---------------- Lead Actor Criteria ----------------
    actor_node = evaluator.add_parallel(
        id=f"Production_{index_one_based}_Lead_Actor_Criteria",
        desc=f"Lead actor Emmy requirements for Production {index_one_based}",
        parent=prod_node,
        critical=True,
    )

    # Existence of actor verification URLs (critical prerequisite)
    actor_urls_exist_node = evaluator.add_custom_node(
        result=bool(prod.actor_urls),
        id=f"Production_{index_one_based}_Actor_Verification_URL",
        desc=f"Reference URL(s) provided to verify actor's 2024 Emmy win and role in production",
        parent=actor_node,
        critical=True,
    )

    # Actor won a Primetime Emmy in 2024
    emmy_win_leaf = evaluator.add_leaf(
        id=f"Production_{index_one_based}_Actor_Emmy_Win",
        desc=f"The production features at least one lead actor or actress who won a Primetime Emmy Award in 2024",
        parent=actor_node,
        critical=True,
    )
    emmy_win_claim = (
        f"In 2024, {prod.emmy_actor_name or 'the actor'} won a Primetime Emmy Award."
    )
    await evaluator.verify(
        claim=emmy_win_claim,
        node=emmy_win_leaf,
        sources=prod.actor_urls,
        additional_instruction=(
            "Confirm using the provided URLs that the named actor/actress is a Primetime Emmy winner in 2024. "
            "The win must be from the Primetime Emmys (not Daytime)."
        ),
        extra_prerequisites=[actor_urls_exist_node],
    )

    # Emmy category is a Lead Acting category
    emmy_category_leaf = evaluator.add_leaf(
        id=f"Production_{index_one_based}_Actor_Emmy_Category",
        desc=(
            "The Emmy won was in a Lead Acting category (Lead Actor/Actress in Drama, Comedy, or Limited Series)"
        ),
        parent=actor_node,
        critical=True,
    )
    if prod.emmy_actor_category and prod.emmy_actor_name:
        cat_claim = (
            f"In 2024, {prod.emmy_actor_name} won the Primetime Emmy for {prod.emmy_actor_category}, "
            f"which is a Lead Acting category."
        )
    else:
        cat_claim = (
            f"In 2024, {prod.emmy_actor_name or 'the actor'} won a Primetime Emmy in a Lead Acting category "
            f"(Lead Actor/Actress in Drama, Comedy, or Limited Series)."
        )
    await evaluator.verify(
        claim=cat_claim,
        node=emmy_category_leaf,
        sources=prod.actor_urls,
        additional_instruction=(
            "Confirm that the category is a Lead Acting category: Outstanding Lead Actor/Actress in a Drama Series, "
            "Outstanding Lead Actor/Actress in a Comedy Series, or Outstanding Lead Actor/Actress in a Limited Series or Movie."
        ),
        extra_prerequisites=[actor_urls_exist_node],
    )

    # Actor has a lead or major role in this production
    role_leaf = evaluator.add_leaf(
        id=f"Production_{index_one_based}_Actor_Role_Verification",
        desc=f"The Emmy-winning actor has a lead or major role in this production",
        parent=actor_node,
        critical=True,
    )
    role_claim = (
        f"{prod.emmy_actor_name or 'The actor'} has a lead or major role in "
        f"{f'\"{prod.title}\"' if prod.title else 'this production'}."
    )
    role_sources = combine_sources(prod.actor_urls, prod.release_urls)
    await evaluator.verify(
        claim=role_claim,
        node=role_leaf,
        sources=role_sources,
        additional_instruction=(
            "Confirm via official cast lists, credible news, or production pages that the Emmy-winning actor is credited "
            "as lead, main cast, starring, or a major role (co-lead acceptable)."
        ),
        extra_prerequisites=[actor_urls_exist_node],
    )

    # ---------------- Release Criteria ----------------
    release_node = evaluator.add_parallel(
        id=f"Production_{index_one_based}_Release_Criteria",
        desc=f"Release timing requirements for Production {index_one_based}",
        parent=prod_node,
        critical=True,
    )

    # Existence of release verification URLs (critical prerequisite)
    rel_urls_exist_node = evaluator.add_custom_node(
        result=bool(prod.release_urls),
        id=f"Production_{index_one_based}_Release_Verification_URL",
        desc=f"Reference URL(s) provided to verify official release/premiere date",
        parent=release_node,
        critical=True,
    )

    # Release date within range and supported by URLs
    release_leaf = evaluator.add_leaf(
        id=f"Production_{index_one_based}_Release_Date",
        desc=f"The production was released or premiered between October 1, 2024, and March 31, 2026 (inclusive)",
        parent=release_node,
        critical=True,
    )
    if prod.title and prod.release_date:
        release_claim = (
            f"The production \"{prod.title}\" was released/premiered on {prod.release_date}, "
            f"which is between {DATE_RANGE_START_TEXT} and {DATE_RANGE_END_TEXT}."
        )
    elif prod.title:
        release_claim = (
            f"The production \"{prod.title}\" was released or premiered between {DATE_RANGE_START_TEXT} and {DATE_RANGE_END_TEXT}."
        )
    else:
        release_claim = (
            f"The production was released or premiered between {DATE_RANGE_START_TEXT} and {DATE_RANGE_END_TEXT}."
        )
    await evaluator.verify(
        claim=release_claim,
        node=release_leaf,
        sources=prod.release_urls,
        additional_instruction=(
            "Confirm the official release/premiere date from the provided URLs and ensure it falls within the specified window. "
            "Accept regional releases or festival premieres if explicitly stated."
        ),
        extra_prerequisites=[rel_urls_exist_node],
    )

    # Format type is feature film or television series (source-supported)
    format_leaf = evaluator.add_leaf(
        id=f"Production_{index_one_based}_Format_Type",
        desc=f"The production is either a feature film or a television series",
        parent=release_node,
        critical=True,
    )
    if prod.title and prod.format_type:
        format_claim = (
            f"\"{prod.title}\" is a {prod.format_type}, which is either a feature film or a television series."
        )
    elif prod.title:
        format_claim = (
            f"\"{prod.title}\" is either a feature film or a television series."
        )
    else:
        format_claim = "The production is either a feature film or a television series."
    await evaluator.verify(
        claim=format_claim,
        node=format_leaf,
        sources=prod.release_urls,
        additional_instruction=(
            "Use the provided release/official pages to confirm the format. Accept synonyms such as 'film', 'movie', "
            "'TV series', 'television series', 'limited series/miniseries'."
        ),
        extra_prerequisites=[rel_urls_exist_node],
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
    """
    Evaluate an answer for the Emmy lead 2024 productions task.
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
        default_model=model,
    )

    # Extract productions from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_productions(),
        template_class=ProductionsExtraction,
        extraction_name="productions_extraction",
    )

    # Filter to first 3 distinct productions by title; pad if fewer
    first_three = dedup_and_take_first_k(extraction.productions, k=3)

    # Record informational context
    evaluator.add_custom_info(
        info={
            "date_window": {
                "start": DATE_RANGE_START_TEXT,
                "end": DATE_RANGE_END_TEXT,
            },
            "requested_items": 3,
            "extracted_total": len(extraction.productions),
            "distinct_taken": sum(1 for p in first_three if p.title),
        },
        info_type="task_context",
    )

    # Build Task_Completion node (critical)
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc=(
            "Find at least 3 distinct productions released between October 1, 2024, and March 31, 2026, "
            "each directed by a filmmaker born 1960–1975 with >=4 prior features/major series, and featuring "
            "a 2024 Primetime Emmy Lead Acting winner in a lead/major role"
        ),
        parent=root,
        critical=True,
    )

    # Minimum item count (critical leaf)
    # Check there are at least 3 distinct productions identified (by unique non-empty titles)
    distinct_titles = {normalize_title(p.title) for p in extraction.productions if p.title and p.title.strip()}
    min_count_ok = len(distinct_titles) >= 3
    evaluator.add_custom_node(
        result=min_count_ok,
        id="Minimum_Item_Count",
        desc="At least 3 distinct productions are identified that meet all criteria",
        parent=task_node,
        critical=True,
    )

    # Verify each of the first three productions
    for idx, prod in enumerate(first_three, start=1):
        await verify_one_production(evaluator, task_node, prod, idx)

    # Return standardized summary
    return evaluator.get_summary()