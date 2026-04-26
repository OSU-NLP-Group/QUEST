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
TASK_ID = "nbc_thanksgiving_2025_dog_show_bis_judge"
TASK_DESCRIPTION = """
At the nationally televised dog show that aired on NBC on Thanksgiving Day 2025, who was the Best in Show judge, what Florida city does this judge reside in, and how many years has this person been a professional AKC judge?
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ShowInfo(BaseModel):
    """Information about the identified dog show event."""
    show_name: Optional[str] = None
    prestige_notes: Optional[str] = None  # any text indicating prestige/major status (e.g., "nationally televised", "major AKC event")
    show_sources: List[str] = Field(default_factory=list)  # URLs supporting show identity and NBC/Thanksgiving broadcast


class JudgeInfo(BaseModel):
    """Information about the Best in Show judge."""
    judge_name: Optional[str] = None
    judge_role: Optional[str] = None  # e.g., "Best in Show judge"
    florida_city: Optional[str] = None  # City of residence in Florida
    akc_judging_years: Optional[str] = None  # number of years as professional AKC judge (as stated in the answer)
    judge_sources: List[str] = Field(default_factory=list)  # URLs supporting judge identity, city, and years


class DogShowExtraction(BaseModel):
    """Top-level extraction model for show and judge info."""
    show: Optional[ShowInfo] = None
    judge: Optional[JudgeInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_show_and_judge() -> str:
    return """
    Extract, from the provided answer text, the specific dog show (event) that matches the query about NBC's nationally televised broadcast on Thanksgiving Day 2025, and the requested Best in Show judge information.

    You must return a JSON object with two top-level fields: "show" and "judge".

    For "show", extract:
    - show_name: The exact name of the dog show event (e.g., "The National Dog Show Presented by Purina").
    - prestige_notes: Any phrasing that indicates the show is a major or prestigious event (e.g., "nationally televised", "prestigious", "major AKC event", "longstanding tradition").
    - show_sources: A list of URL strings that were explicitly provided in the answer and that support the identified show and specifically its NBC/Thanksgiving Day 2025 airing. Extract only valid URLs that appear in the answer; do not infer or invent URLs.

    For "judge", extract:
    - judge_name: The full name of the Best in Show judge.
    - judge_role: The role as stated (should be "Best in Show judge" or equivalent wording explicitly indicating Best in Show).
    - florida_city: The Florida city where the judge resides (e.g., "Orlando, FL" or "Orlando, Florida"). If a neighborhood or county is mentioned, prefer the city if present.
    - akc_judging_years: A specific number of years the person has been a professional AKC judge as claimed in the answer (return as a string). If the answer provides a computed or rounded value, extract that number as-is.
    - judge_sources: A list of URL strings explicitly provided in the answer that support the judge identity, Florida city of residence, and years as a professional AKC judge. Extract only valid URLs; do not infer or invent.

    IMPORTANT URL RULES:
    - Only extract URLs that are explicitly present in the answer text (including plain URLs or markdown links).
    - If an attribution like "according to AKC" is given without a URL, do not add a URL—return an empty list instead.
    - Include complete URLs. If a URL is missing a protocol, prepend "http://".
    - If no relevant URLs are provided for a section, return an empty list.

    If any required field is missing from the answer, set it to null (for strings) or an empty list (for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def has_valid_urls(urls: Optional[List[str]]) -> bool:
    """Return True if there is at least one plausible URL string."""
    if not urls:
        return False
    for u in urls:
        if isinstance(u, str) and u.strip():
            # Basic plausibility check
            if "http://" in u or "https://" in u or u.strip().startswith("www."):
                return True
    return False


def combine_sources(*lists: List[str]) -> List[str]:
    """Combine multiple URL lists and deduplicate while preserving order."""
    seen = set()
    combined: List[str] = []
    for lst in lists:
        for u in lst:
            if u not in seen:
                seen.add(u)
                combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def build_and_verify_show(
    evaluator: Evaluator,
    parent_node,
    extracted: DogShowExtraction,
) -> None:
    """Build the 'show_match' subtree and run verifications."""
    show = extracted.show or ShowInfo()

    show_node = evaluator.add_parallel(
        id="show_match",
        desc="Identify the dog show that fits the broadcast/date/prestige constraints.",
        parent=parent_node,
        critical=True
    )

    # Leaf 1: show_name (existence check)
    evaluator.add_custom_node(
        result=bool(show.show_name and show.show_name.strip()),
        id="show_name",
        desc="Provide the name of the dog show event.",
        parent=show_node,
        critical=True
    )

    # Leaf 2: NBC Thanksgiving 2025 broadcast verification (by sources)
    nbc_leaf = evaluator.add_leaf(
        id="show_nbc_thanksgiving_2025",
        desc="Show was nationally televised on NBC and aired on Thanksgiving Day 2025.",
        parent=show_node,
        critical=True
    )

    nbc_claim = (
        f"The event '{show.show_name or ''}' was nationally televised on NBC and aired on Thanksgiving Day 2025."
    )
    await evaluator.verify(
        claim=nbc_claim,
        node=nbc_leaf,
        sources=show.show_sources,
        additional_instruction=(
            "Verify that the provided source(s) explicitly indicate the show aired on NBC on Thanksgiving Day 2025. "
            "Thanksgiving Day 2025 is Thursday, November 27, 2025. Accept phrases like 'airs on NBC on Thanksgiving' "
            "or official schedules/press releases indicating NBC and Thanksgiving Day 2025."
        )
    )

    # Leaf 3: Prestige/major status verification (by sources)
    prestige_leaf = evaluator.add_leaf(
        id="show_major_prestigious",
        desc="Show is a major, prestigious dog show event.",
        parent=show_node,
        critical=True
    )
    prestige_claim = (
        f"The event '{show.show_name or ''}' is recognized as a major, prestigious dog show."
    )
    await evaluator.verify(
        claim=prestige_claim,
        node=prestige_leaf,
        sources=show.show_sources,
        additional_instruction=(
            "Use the source(s) to determine if the show is widely recognized as major/prestigious. "
            "Signals can include national TV coverage, long-standing tradition, 'major' wording, AKC prominence, "
            "and broad public attention."
        )
    )


async def build_and_verify_judge(
    evaluator: Evaluator,
    parent_node,
    extracted: DogShowExtraction,
) -> None:
    """Build the 'judge_answer' subtree and run verifications."""
    show = extracted.show or ShowInfo()
    judge = extracted.judge or JudgeInfo()

    judge_node = evaluator.add_parallel(
        id="judge_answer",
        desc="Provide the requested Best in Show judge information.",
        parent=parent_node,
        critical=True
    )

    # Leaf 1: Best in Show judge name (existence check)
    evaluator.add_custom_node(
        result=bool(judge.judge_name and judge.judge_name.strip()),
        id="best_in_show_judge_name",
        desc="Provide the Best in Show judge’s name.",
        parent=judge_node,
        critical=True
    )

    # Leaf 2: Role confirmed as Best in Show judge (verify via sources)
    bis_role_leaf = evaluator.add_leaf(
        id="best_in_show_role_confirmed",
        desc="Person is specifically the Best in Show judge (not a group judge or other role).",
        parent=judge_node,
        critical=True
    )

    role_claim = (
        f"{judge.judge_name or ''} served specifically as the Best in Show judge for the event '{show.show_name or ''}'."
    )
    await evaluator.verify(
        claim=role_claim,
        node=bis_role_leaf,
        sources=combine_sources(judge.judge_sources, show.show_sources),
        additional_instruction=(
            "Confirm that the person held the 'Best in Show judge' role (final round judge), not a group or breed judge. "
            "Look for explicit 'Best in Show judge' wording in official schedules, press releases, or the show website."
        )
    )

    # Leaf 3: Florida city residence verification
    city_leaf = evaluator.add_leaf(
        id="judge_florida_city",
        desc="Provide the Florida city where the judge resides.",
        parent=judge_node,
        critical=True
    )

    city_claim = (
        f"{judge.judge_name or ''} resides in {judge.florida_city or ''}, Florida."
    )
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        sources=judge.judge_sources,
        additional_instruction=(
            "Verify the city of residence in Florida from the provided source(s). "
            "Accept variants like 'City, FL' or 'City, Florida'."
        )
    )

    # Leaf 4: AKC professional judging years verification
    years_leaf = evaluator.add_leaf(
        id="judge_akc_judging_years",
        desc="Provide a specific number of years the person has been a professional AKC judge.",
        parent=judge_node,
        critical=True
    )

    years_claim = (
        f"{judge.judge_name or ''} has been a professional AKC judge for {judge.akc_judging_years or ''} years."
    )
    await evaluator.verify(
        claim=years_claim,
        node=years_leaf,
        sources=judge.judge_sources,
        additional_instruction=(
            "Use the provided source(s) to confirm the stated years. "
            "If the source lists an initial approval year (e.g., 'AKC judge since 1992'), "
            "allow reasonable calculations/rounding to the stated number of years if consistent."
        )
    )


async def build_and_verify_sources_minimum(
    evaluator: Evaluator,
    parent_node,
    extracted: DogShowExtraction,
) -> None:
    """Build the 'sources_minimum' subtree requiring at least minimal reliable sources."""
    show = extracted.show or ShowInfo()
    judge = extracted.judge or JudgeInfo()

    sources_node = evaluator.add_parallel(
        id="sources_minimum",
        desc="Provide a small set of reliable sources sufficient to verify the key required claims (can be the same source for multiple claims).",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_valid_urls(show.show_sources),
        id="source_for_show",
        desc="Provide at least one reliable source URL that supports the identified show and its NBC/Thanksgiving Day 2025 airing.",
        parent=sources_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_valid_urls(judge.judge_sources),
        id="source_for_judge_and_attributes",
        desc="Provide at least one reliable source URL that supports the Best in Show judge identity, Florida city of residence, and years as a professional AKC judge.",
        parent=sources_node,
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
    """
    Evaluate an answer for the NBC Thanksgiving Day 2025 dog show Best in Show judge query.
    Builds a sequential, fully critical tree:
      1) show_match
      2) judge_answer
      3) sources_minimum
    If any critical part fails, subsequent parts are skipped per sequential aggregation.
    """
    # Initialize evaluator
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
    # Set root critical to enforce consistency: children of critical must be critical per framework rules
    root.critical = True

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_show_and_judge(),
        template_class=DogShowExtraction,
        extraction_name="show_and_judge_extraction",
    )

    # Build verification subtrees according to rubric
    await build_and_verify_show(evaluator, root, extracted)
    await build_and_verify_judge(evaluator, root, extracted)
    await build_and_verify_sources_minimum(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()