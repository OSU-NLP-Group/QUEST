import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "entertainment_achievements_jan_feb_2026"
TASK_DESCRIPTION = (
    "Identify four major entertainment achievements from the film and television industry that occurred during January "
    "or February 2026. The achievements must come from at least three different categories among the following: "
    "major film awards ceremonies, large-scale live performance events, film festival awards, or television "
    "competition outcomes. For each achievement, provide: (1) the event name, (2) the exact date it occurred, "
    "(3) the primary winner, performer, or recipient, (4) the specific award category or role they received/performed, "
    "and (5) a reference URL supporting your answer. The achievements must be significant industry milestones that were "
    "widely covered by major entertainment news outlets."
)

ALLOWED_CATEGORY_TYPES = {
    "film_awards_ceremony",
    "live_performance_event",
    "film_festival_award",
    "television_competition_outcome",
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AchievementItem(BaseModel):
    event_name: Optional[str] = None
    date: Optional[str] = None  # Keep as free text for robustness (e.g., "February 8, 2026")
    primary: Optional[str] = None  # winner/performer/recipient
    specific_category_or_role: Optional[str] = None  # e.g., "Best Picture", "Halftime Headliner"
    category_type: Optional[str] = None  # one of ALLOWED_CATEGORY_TYPES
    reference_url: Optional[str] = None  # single principal supporting URL


class AchievementsExtraction(BaseModel):
    achievements: List[AchievementItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_achievements() -> str:
    return """
    From the provided answer, extract up to 6 distinct entertainment achievements that are stated to have occurred during January or February 2026.
    For EACH achievement mentioned in the answer, extract the following fields exactly as presented:
    - event_name: The name of the event (e.g., "Golden Globe Awards", "Sundance Film Festival", "Super Bowl Halftime Show").
    - date: The specific date the achievement occurred (e.g., "January 10, 2026"). If a date range is provided, extract the single most relevant date for the achievement (e.g., the ceremony date or the performance date).
    - primary: The primary winner, performer, or recipient (e.g., a person, film, show, ensemble).
    - specific_category_or_role: The specific award category or role (e.g., "Best Picture", "Best Actor in a Drama", "Halftime Headliner", "Competition Winner").
    - category_type: Choose EXACTLY one string from this set:
        ["film_awards_ceremony", "live_performance_event", "film_festival_award", "television_competition_outcome"].
      Pick the most appropriate category type for the achievement as described in the answer text.
    - reference_url: A single URL from the answer that best supports this achievement (must be explicitly present in the answer).
      If multiple URLs are present, choose the one that most directly corroborates the details (event, date, winner/performer, category/role).
    
    IMPORTANT:
    - Only extract data explicitly provided in the answer. Do NOT invent or infer missing details.
    - If any field is missing for an achievement, return null for that field.
    - If no URL is provided in the answer for a given achievement, set reference_url to null.
    - The category_type MUST be one of the four allowed values above. If ambiguous, choose the best fit based on the answer's own description.
    - Dates should remain as free text exactly as written in the answer (do not reformat).
    - Return a JSON object with a single key "achievements" containing an array of achievement objects.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    try:
        parsed = urlparse(url.strip())
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def canonicalize_category(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    s = label.strip().lower().replace("-", "_").replace(" ", "_")
    # If the answer already used a canonical label
    if s in ALLOWED_CATEGORY_TYPES:
        return s

    # Heuristic normalization for common synonyms/variants
    if "festival" in s:
        return "film_festival_award"
    if "television" in s and ("competition" in s or "contest" in s):
        return "television_competition_outcome"
    if any(k in s for k in ["live", "performance", "concert", "halftime", "tour", "opening_ceremony"]):
        return "live_performance_event"
    if any(k in s for k in ["award", "awards", "ceremony", "oscars", "academy", "bafta", "globe", "sag", "critics_choice", "emmys"]):
        return "film_awards_ceremony"
    return None


def ordinal(i: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"][i] if 0 <= i < 6 else f"#{i+1}"


# --------------------------------------------------------------------------- #
# Verification for one achievement                                            #
# --------------------------------------------------------------------------- #
async def verify_one_achievement(
    evaluator: Evaluator,
    parent_node,
    idx_zero_based: int,
    item: AchievementItem,
) -> None:
    ach_idx = idx_zero_based + 1
    ach_node = evaluator.add_parallel(
        id=f"achievement_{ach_idx}",
        desc=f"{ordinal(idx_zero_based)} major entertainment achievement from January-February 2026",
        parent=parent_node,
        critical=False
    )

    # Reference URL existence/validity (critical gate)
    ref_ok = is_valid_url(item.reference_url)
    ref_node = evaluator.add_custom_node(
        result=ref_ok,
        id=f"achievement_{ach_idx}_reference",
        desc="Provides a valid reference URL supporting the achievement",
        parent=ach_node,
        critical=True
    )

    # Event name verification (critical)
    event_leaf = evaluator.add_leaf(
        id=f"achievement_{ach_idx}_event",
        desc="Correctly identifies the event name",
        parent=ach_node,
        critical=True
    )
    event_claim = f"The referenced page is about the event '{item.event_name}'. Allow minor variations (year qualifiers, edition numbers) but the event identity must match."
    await evaluator.verify(
        claim=event_claim,
        node=event_leaf,
        sources=item.reference_url if ref_ok else None,
        additional_instruction="Confirm the page centers on or clearly identifies the specified event name. Accept minor naming variations (e.g., '78th Annual ... 2026')."
    )

    # Date verification (critical)
    date_leaf = evaluator.add_leaf(
        id=f"achievement_{ach_idx}_date",
        desc="Provides the correct event date within January-February 2026",
        parent=ach_node,
        critical=True
    )
    date_claim = (
        f"The referenced page indicates that the event took place on '{item.date}', and that this date is in January or February 2026."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=item.reference_url if ref_ok else None,
        additional_instruction="Verify that the page states the event occurred on the given date and that the date falls within Jan or Feb 2026. "
                              "If the page shows a date range, the claimed date should match the ceremony/performance date."
    )

    # Winner/Performer/Recipient verification (critical)
    winner_leaf = evaluator.add_leaf(
        id=f"achievement_{ach_idx}_winner",
        desc="Correctly identifies the primary winner, performer, or recipient",
        parent=ach_node,
        critical=True
    )
    winner_claim = (
        f"The referenced page clearly identifies '{item.primary}' as the primary winner/performer/recipient for this event."
    )
    await evaluator.verify(
        claim=winner_claim,
        node=winner_leaf,
        sources=item.reference_url if ref_ok else None,
        additional_instruction="Confirm that the page explicitly credits this person/film/show as the main winner, recipient, or key performer for the achievement."
    )

    # Specific category/role verification (critical)
    category_leaf = evaluator.add_leaf(
        id=f"achievement_{ach_idx}_category",
        desc="Specifies the correct award category, role type, or achievement type",
        parent=ach_node,
        critical=True
    )
    category_claim = (
        f"The referenced page states that '{item.primary}' received/performed in the specific category/role '{item.specific_category_or_role}' at the event '{item.event_name}'."
    )
    await evaluator.verify(
        claim=category_claim,
        node=category_leaf,
        sources=item.reference_url if ref_ok else None,
        additional_instruction="Match the specific award category or performance role text on the page. Accept reasonable abbreviations or near-synonyms."
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
) -> Dict[str, Any]:
    # Initialize evaluator (root non-critical due to framework constraint: critical parent cannot have non-critical children)
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

    # Extract structured achievements from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_achievements(),
        template_class=AchievementsExtraction,
        extraction_name="achievements_extraction"
    )

    # Keep only first 4 achievements; pad with empty placeholders if fewer
    items: List[AchievementItem] = list(extracted.achievements[:4])
    while len(items) < 4:
        items.append(AchievementItem())

    # Build verification subtrees for each of the four achievements
    for idx in range(4):
        await verify_one_achievement(evaluator, root, idx, items[idx])

    # Category diversity check (critical for overall task success)
    normalized_cats = []
    for it in items:
        norm = canonicalize_category(it.category_type)
        if norm in ALLOWED_CATEGORY_TYPES:
            normalized_cats.append(norm)
    distinct_cat_count = len(set(normalized_cats))

    evaluator.add_custom_info(
        info={"extracted_categories": normalized_cats, "distinct_count": distinct_cat_count},
        info_type="intermediate",
        info_name="category_diversity_debug"
    )

    evaluator.add_custom_node(
        result=(distinct_cat_count >= 3),
        id="category_diversity",
        desc="The four achievements represent at least three different entertainment categories (e.g., film awards ceremonies, live performance events, film festivals, television competitions)",
        parent=root,
        critical=True
    )

    # Return final structured evaluation result
    return evaluator.get_summary()