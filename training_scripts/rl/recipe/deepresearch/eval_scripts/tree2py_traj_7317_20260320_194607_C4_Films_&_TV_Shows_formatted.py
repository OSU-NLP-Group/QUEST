import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "awards_gg_emmys_2025_2026"
TASK_DESCRIPTION = """
I'm researching the 2025-2026 television awards season and want to identify TV shows and individual actor/actress performances that achieved remarkable recognition by winning at both major awards ceremonies. Please identify at least 4 TV shows or individual actor/actress performances that won awards at BOTH the 83rd Golden Globes (held January 11, 2026) and the 77th Primetime Emmy Awards (held September 14, 2025) in matching categories. For matching categories, I mean: (1) For TV series: Drama Series categories at both ceremonies, OR Comedy Series categories at both ceremonies, OR Limited/Anthology Series categories at both ceremonies. (2) For acting performances: Best Actor/Actress Drama Series at both ceremonies, OR Best Actor/Actress Comedy Series at both ceremonies. For each identified item, provide: (1) The name of the TV show (for series wins) OR the actor/actress name (for acting wins), (2) For acting wins: the TV show they performed in, (3) The specific award category won at the Golden Globes 2026, (4) The specific award category won at the Emmy Awards 2025, (5) A URL reference to verify the Golden Globes 2026 win, (6) A URL reference to verify the Emmy Awards 2025 win.
"""

GG_YEAR = 2026  # 83rd Golden Globes (Jan 11, 2026)
EMMY_YEAR = 2025  # 77th Primetime Emmy Awards (Sep 14, 2025)

GG_OFFICIAL_DOMAIN = "goldenglobes.com"
EMMY_OFFICIAL_DOMAIN = "televisionacademy.com"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AwardItem(BaseModel):
    # item_type: "series" or "acting"
    item_type: Optional[str] = None
    # If series -> series/show name; If acting -> performer name
    primary_name: Optional[str] = None
    # Only required if item_type == "acting": the TV show the performance is for
    show_name: Optional[str] = None
    # Category strings as written in the answer
    gg_category: Optional[str] = None
    emmy_category: Optional[str] = None
    # Official verification URLs
    gg_url: Optional[str] = None
    emmy_url: Optional[str] = None


class AwardsExtraction(BaseModel):
    items: List[AwardItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_award_items() -> str:
    return """
    Extract the list of items (TV series wins or acting performance wins) presented in the answer that won BOTH at the
    83rd Golden Globes (January 11, 2026) and the 77th Primetime Emmy Awards (September 14, 2025) in matching categories.

    For each item, extract the following fields:
    - item_type: one of ["series", "acting"] exactly.
    - primary_name:
        * If item_type == "series": the TV series name.
        * If item_type == "acting": the actor/actress name (the performer).
    - show_name:
        * Only if item_type == "acting": the TV series the performer won for. If the answer does not state it, set to null.
        * If item_type == "series": set to null.
    - gg_category: the Golden Globes category string as stated in the answer text.
    - emmy_category: the Emmy category string as stated in the answer text.
    - gg_url: a single URL provided in the answer that verifies the Golden Globes 2026 win (prefer goldenglobes.com).
    - emmy_url: a single URL provided in the answer that verifies the Emmy Awards 2025 win (prefer televisionacademy.com).

    Rules:
    - Extract only what is explicitly provided in the answer text.
    - If a required field is missing for an item, set it to null.
    - Return all items in order of appearance. Do not fabricate items.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> str:
    if not s:
        return ""
    # Normalize string for robust comparisons
    t = s.strip().lower()
    # unify dashes, punctuation that often vary
    t = t.replace("–", "-").replace("—", "-")
    # common synonyms
    t = t.replace("&", "and")
    # collapse whitespace
    t = " ".join(t.split())
    return t


def _is_series_type(cat: str) -> bool:
    t = _norm(cat)
    if not t:
        return False
    # Globes often: "Best Television Series – Drama/Best Television Series – Musical or Comedy/Best Limited or Anthology Series or Television Film"
    # Emmys: "Outstanding Drama Series/Outstanding Comedy Series/Outstanding Limited or Anthology Series"
    if "drama series" in t:
        return True
    if "comedy series" in t:
        return True
    if "limited" in t and "series" in t:
        return True
    if "anthology" in t and "series" in t:
        return True
    # For Globes limited category variants including TV film
    if "television film" in t or "tv movie" in t or "made for television" in t:
        return True
    # Generic fallback
    if "television series" in t and ("drama" in t or "comedy" in t or "musical" in t):
        return True
    return False


def _is_acting_type(cat: str) -> bool:
    t = _norm(cat)
    if not t:
        return False
    # Globes acting: "Best Performance by an Actor/Actress in a Television Series – Drama/Musical or Comedy"
    # Emmys acting: "Outstanding Lead Actor/Actress in a Drama/Comedy Series"
    has_actor = "actor" in t
    has_actress = "actress" in t
    has_drama = "drama" in t
    has_comedy = "comedy" in t
    # Must be clearly acting and in Drama/Comedy Series
    if (has_actor or has_actress) and (has_drama or has_comedy) and "series" in t:
        return True
    return False


def categorize_gg_category(cat: Optional[str]) -> Optional[str]:
    """
    Map Golden Globes category string to one of:
    series_drama, series_comedy, series_limited,
    actor_drama, actress_drama, actor_comedy, actress_comedy
    """
    t = _norm(cat)
    if not t:
        return None

    # Series
    if _is_series_type(t):
        if "drama" in t and "series" in t and "limited" not in t and "anthology" not in t and "television film" not in t and "tv movie" not in t:
            return "series_drama"
        if ("comedy" in t or "musical" in t) and "series" in t and "limited" not in t and "anthology" not in t:
            return "series_comedy"
        if "limited" in t or "anthology" in t or "television film" in t or "tv movie" in t or "made for television" in t:
            return "series_limited"

    # Acting
    if _is_acting_type(t):
        gender = "actor" if "actor" in t and "actress" not in t else ("actress" if "actress" in t else None)
        if not gender:
            return None
        if "drama" in t:
            return f"{gender}_drama"
        if "comedy" in t or "musical" in t:
            return f"{gender}_comedy"

    return None


def categorize_emmy_category(cat: Optional[str]) -> Optional[str]:
    """
    Map Primetime Emmy category string (77th, 2025) to:
    series_drama, series_comedy, series_limited,
    actor_drama, actress_drama, actor_comedy, actress_comedy
    """
    t = _norm(cat)
    if not t:
        return None

    # Series
    if "drama series" in t:
        return "series_drama"
    if "comedy series" in t:
        return "series_comedy"
    if "limited" in t and "series" in t:
        return "series_limited"
    if "anthology" in t and "series" in t:
        return "series_limited"

    # Acting (lead actor/actress)
    gender = "actor" if "actor" in t and "actress" not in t else ("actress" if "actress" in t else None)
    if gender:
        # require "lead"
        if "lead" not in t:
            return None
        if "drama series" in t:
            return f"{gender}_drama"
        if "comedy series" in t:
            return f"{gender}_comedy"

    return None


def gg_emmy_types_match(gg_type: Optional[str], emmy_type: Optional[str]) -> bool:
    """
    Ensure mapping consistency:
    - series_drama <-> series_drama
    - series_comedy <-> series_comedy
    - series_limited <-> series_limited
    - actor_drama <-> actor_drama
    - actress_drama <-> actress_drama
    - actor_comedy <-> actor_comedy
    - actress_comedy <-> actress_comedy
    """
    if not gg_type or not emmy_type:
        return False
    return gg_type == emmy_type


def infer_item_type(item: AwardItem) -> Optional[str]:
    t = _norm(item.item_type)
    if t in ("series", "acting"):
        return t
    # Fallback inference: if show_name present and primary_name seems like a person (contains space) -> acting
    if item.show_name and item.primary_name:
        return "acting"
    if item.primary_name and not item.show_name:
        # ambiguous; default to series if the categories look like series
        gg_is_series = _is_series_type(item.gg_category or "")
        emmy_is_series = _is_series_type(item.emmy_category or "")
        gg_is_acting = _is_acting_type(item.gg_category or "")
        emmy_is_acting = _is_acting_type(item.emmy_category or "")
        if gg_is_series or emmy_is_series:
            return "series"
        if gg_is_acting or emmy_is_acting:
            return "acting"
    return None


def domain_is(url: Optional[str], domain: str) -> bool:
    if not url:
        return False
    u = url.strip().lower()
    return domain in u


def build_identity_key(item: AwardItem) -> Optional[str]:
    t = infer_item_type(item)
    if not t:
        return None
    if t == "series":
        if not item.primary_name:
            return None
        return f"series::{_norm(item.primary_name)}"
    if t == "acting":
        if not item.primary_name or not item.show_name:
            return None
        return f"acting::{_norm(item.primary_name)}::{_norm(item.show_name)}"
    return None


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_award_item(evaluator: Evaluator, parent_node, item: AwardItem, idx: int) -> None:
    """
    Build and verify the subtree for a single item.
    """
    item_node = evaluator.add_parallel(
        id=f"item_{idx+1}",
        desc=f"Item {idx+1} (TV series win or acting performance win).",
        parent=parent_node,
        critical=False
    )

    itype = infer_item_type(item)
    primary_ok = bool(item.primary_name and item.primary_name.strip())
    # Primary identity provided
    evaluator.add_custom_node(
        result=primary_ok,
        id=f"item_{idx+1}_primary_identity_provided",
        desc="Provides the primary identity: TV series name (for series win) OR actor/actress name (for acting win).",
        parent=item_node,
        critical=True
    )

    # Acting includes show (only if acting)
    acting_show_ok = True
    if itype == "acting":
        acting_show_ok = bool(item.show_name and item.show_name.strip())
    evaluator.add_custom_node(
        result=acting_show_ok,
        id=f"item_{idx+1}_acting_includes_show_if_applicable",
        desc="If (and only if) the item is an acting performance, includes the TV show of the performance.",
        parent=item_node,
        critical=True
    )

    # GG category stated and eligible
    gg_cat_ok = False
    gg_kind = None
    if item.gg_category:
        gg_kind = categorize_gg_category(item.gg_category)
        if itype == "series":
            gg_cat_ok = gg_kind in {"series_drama", "series_comedy", "series_limited"}
        elif itype == "acting":
            gg_cat_ok = gg_kind in {"actor_drama", "actress_drama", "actor_comedy", "actress_comedy"}
        else:
            gg_cat_ok = False
    evaluator.add_custom_node(
        result=gg_cat_ok,
        id=f"item_{idx+1}_gg_category_stated_and_eligible_type",
        desc="States the Golden Globes category and it is one of the eligible types (Drama Series / Musical-or-Comedy Series / Limited-or-Anthology Series-or-TV Movie / Actor or Actress in Drama Series / Actor or Actress in Musical-or-Comedy Series).",
        parent=item_node,
        critical=True
    )

    # Emmy category stated and eligible
    emmy_cat_ok = False
    emmy_kind = None
    if item.emmy_category:
        emmy_kind = categorize_emmy_category(item.emmy_category)
        if itype == "series":
            emmy_cat_ok = emmy_kind in {"series_drama", "series_comedy", "series_limited"}
        elif itype == "acting":
            emmy_cat_ok = emmy_kind in {"actor_drama", "actress_drama", "actor_comedy", "actress_comedy"}
        else:
            emmy_cat_ok = False
    evaluator.add_custom_node(
        result=emmy_cat_ok,
        id=f"item_{idx+1}_emmy_category_stated_and_eligible_type",
        desc="States the Emmy category and it is one of the eligible types (Outstanding Drama Series / Outstanding Comedy Series / Outstanding Limited or Anthology Series / Outstanding Lead Actor or Actress in a Drama Series / Outstanding Lead Actor or Actress in a Comedy Series).",
        parent=item_node,
        critical=True
    )

    # Category pair matches mapping
    pair_match_ok = gg_emmy_types_match(gg_kind, emmy_kind)
    evaluator.add_custom_node(
        result=pair_match_ok,
        id=f"item_{idx+1}_category_pair_matches_mapping",
        desc="Golden Globes category and Emmy category match per the prompt’s mappings (Drama↔Drama, Comedy↔Comedy, Limited/Anthology↔Limited/Anthology; Lead Actor/Actress Drama↔Drama; Lead Actor/Actress Comedy↔Comedy).",
        parent=item_node,
        critical=True
    )

    # GG official URL verifies win (single leaf; will be skipped if any above critical checks fail)
    gg_verify_node = evaluator.add_leaf(
        id=f"item_{idx+1}_gg_official_url_verifies_win",
        desc="Provides a Golden Globes official-site URL (goldenglobes.com) for the 83rd ceremony that supports the claim the item won the stated Golden Globes category.",
        parent=item_node,
        critical=True
    )

    if itype == "series":
        subject_str = f"The TV series '{item.primary_name}'"
    elif itype == "acting":
        subject_str = f"The performer '{item.primary_name}' for the TV series '{item.show_name}'"
    else:
        subject_str = f"The item '{item.primary_name or ''}'"

    gg_claim = (
        f"{subject_str} won the 83rd Golden Globes ({GG_YEAR}) in the category '{item.gg_category}'. "
        f"This must be verified on an official Golden Globes ({GG_OFFICIAL_DOMAIN}) page for the 83rd ceremony."
    )

    gg_additional_instruction = (
        f"- Treat the claim as NOT SUPPORTED if: "
        f"(a) no URL is provided; "
        f"(b) the URL is not on {GG_OFFICIAL_DOMAIN}; "
        f"(c) the page is not about the 83rd Golden Globes ({GG_YEAR}); or "
        f"(d) it indicates nominee/nomination only rather than a WIN. "
        f"Allow minor variations in category naming (e.g., punctuation, hyphens, 'musical or comedy' vs 'comedy'). "
        f"Confirm the winner status explicitly."
    )

    await evaluator.verify(
        claim=gg_claim,
        node=gg_verify_node,
        sources=item.gg_url if item.gg_url else None,
        additional_instruction=gg_additional_instruction
    )

    # Emmy official URL verifies win
    emmy_verify_node = evaluator.add_leaf(
        id=f"item_{idx+1}_emmy_official_url_verifies_win",
        desc="Provides a Television Academy official-site URL (televisionacademy.com) for the 77th Primetime Emmy Awards that supports the claim the item won the stated Emmy category.",
        parent=item_node,
        critical=True
    )

    emmy_claim = (
        f"{subject_str} won the 77th Primetime Emmy Awards ({EMMY_YEAR}) in the category '{item.emmy_category}'. "
        f"This must be verified on an official Television Academy ({EMMY_OFFICIAL_DOMAIN}) page for the 77th ceremony."
    )

    emmy_additional_instruction = (
        f"- Treat the claim as NOT SUPPORTED if: "
        f"(a) no URL is provided; "
        f"(b) the URL is not on {EMMY_OFFICIAL_DOMAIN}; "
        f"(c) the page is not about the 77th Primetime Emmys ({EMMY_YEAR}); or "
        f"(d) it indicates nominee only rather than a WIN. "
        f"Allow minor category phrasing variations but ensure the mapping (Drama/Comedy/Limited and Lead Actor/Actress) is consistent. "
        f"Confirm the winner status explicitly."
    )

    await evaluator.verify(
        claim=emmy_claim,
        node=emmy_verify_node,
        sources=item.emmy_url if item.emmy_url else None,
        additional_instruction=emmy_additional_instruction
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
    Evaluate an answer for cross-award winners between the 83rd Golden Globes (2026) and 77th Primetime Emmys (2025).
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify ≥4 TV series wins or acting-performance wins that won at BOTH the 83rd Golden Globes (Jan 11, 2026) and the 77th Primetime Emmy Awards (Sept 14, 2025) in matching eligible categories, and provide the required fields and official-source URLs for each item.",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # 1) Extract items
    extracted = await evaluator.extract(
        prompt=prompt_extract_award_items(),
        template_class=AwardsExtraction,
        extraction_name="awards_dual_wins"
    )

    # Build a list of "presented" items (clearly delineated)
    presented_items: List[AwardItem] = []
    for it in extracted.items:
        if it.primary_name and infer_item_type(it):
            presented_items.append(it)

    # 2) Top-level critical checks
    min4_node = evaluator.add_custom_node(
        result=(len(presented_items) >= 4),
        id="min_4_items_provided",
        desc="Response presents at least 4 clearly delineated items (series or acting performances).",
        parent=root,
        critical=True
    )

    # Select the first 4 presented items for detailed verification (as per instruction: filter to first k)
    selected: List[AwardItem] = presented_items[:4]

    # Distinctness among the 4 selected
    distinct_ok = False
    if len(selected) == 4:
        keys = [build_identity_key(it) for it in selected]
        # A valid identity key must exist for each item; duplicates are not allowed
        distinct_ok = all(k is not None for k in keys) and (len(set(keys)) == 4)
    else:
        distinct_ok = False

    evaluator.add_custom_node(
        result=distinct_ok,
        id="items_are_distinct",
        desc="The 4 items are distinct entities (no repeated TV series; no repeated acting performance defined by actor/actress + show).",
        parent=root,
        critical=True
    )

    # Record which items we evaluated
    evaluator.add_custom_info(
        info={
            "evaluated_item_count": len(selected),
            "all_presented_count": len(presented_items),
            "selected_items_preview": [
                {
                    "item_type": infer_item_type(it),
                    "primary_name": it.primary_name,
                    "show_name": it.show_name,
                    "gg_category": it.gg_category,
                    "emmy_category": it.emmy_category,
                    "gg_url": it.gg_url,
                    "emmy_url": it.emmy_url,
                } for it in selected
            ]
        },
        info_type="selection_info"
    )

    # 3) Build per-item verification subtrees (item_1..item_4 in rubric)
    # If fewer than 4 items provided, still instantiate nodes using placeholders to match rubric structure
    while len(selected) < 4:
        selected.append(AwardItem())

    # Create and verify each item subtree
    for i in range(4):
        await verify_award_item(evaluator, root, selected[i], i)

    # 4) Return structured evaluation result
    return evaluator.get_summary()