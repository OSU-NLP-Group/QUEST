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
TASK_ID = "beauty_wnba_ambassador_2024"
TASK_DESCRIPTION = (
    "Identify a beauty or cosmetics brand partnership with a WNBA player that meets all of the following criteria: "
    "(1) the partnership must be a formal brand ambassadorship, not just a team sponsorship; "
    "(2) it must have been publicly announced in 2024; "
    "(3) it must be described as a multi-year or long-term partnership; "
    "(4) the announcement must specifically name at least two signature products that the player will promote; "
    "(5) the beauty brand must have had a prior sponsorship relationship with the player's WNBA team before establishing the ambassadorship with the individual player; "
    "(6) the partnership must be documented in an official press release or brand announcement; and "
    "(7) the player must be currently active in the WNBA (not retired) at the time of the announcement. "
    "Provide the player's name, the beauty brand name, the announcement date, the two or more specific products mentioned, and a reference URL to the official announcement."
)
TARGET_YEAR = 2024


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PartnershipInfo(BaseModel):
    # Core entities
    player_name: Optional[str] = None
    player_team: Optional[str] = None
    brand_name: Optional[str] = None

    # Announcement details
    announcement_date: Optional[str] = None
    official_announcement_url: Optional[str] = None

    # Product details mentioned in announcement (should be 2+)
    products: List[str] = Field(default_factory=list)

    # Evidence URLs for specific checks
    prior_team_sponsorship_urls: List[str] = Field(default_factory=list)  # Prior brand–team sponsorship sources (before announcement)
    brand_category_urls: List[str] = Field(default_factory=list)          # Brand homepage/about page confirming beauty/cosmetics
    player_status_urls: List[str] = Field(default_factory=list)           # WNBA.com player page or official team roster pages

    # Phrases/terms captured from answer (optional, helpful to craft claims)
    role_terms: List[str] = Field(default_factory=list)                   # e.g., ["brand ambassador", "spokesperson"]
    duration_terms: List[str] = Field(default_factory=list)               # e.g., ["multi-year", "long-term"]


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_partnership_info() -> str:
    return """
    From the answer, extract exactly one (the primary) beauty/cosmetics brand partnership with a WNBA player.
    If multiple candidates are listed, choose the first that best satisfies the task criteria.

    Extract the following fields:
    - player_name: The WNBA player's full name.
    - player_team: The WNBA team name associated with the player at the time of the announcement, if provided.
    - brand_name: The beauty or cosmetics brand's name.
    - announcement_date: The announcement/publication date as written (keep as a string; do not reformat).
    - official_announcement_url: The URL to the official press release or brand announcement cited in the answer. This should be an official brand newsroom page or a press release wire (e.g., Business Wire, PR Newswire). If the answer does not provide a URL, return null.
    - products: A list of product names explicitly mentioned in the answer as being promoted by the player in this partnership. Include the exact product names as written (e.g., "Lush Rose Lipstick", "HydraGlow Serum"). Do not include generic categories (e.g., "lipstick", "skincare"). If fewer than two are given, return whatever is present.
    - prior_team_sponsorship_urls: URLs provided in the answer that document the brand's prior sponsorship relationship with the player's WNBA team (before the ambassadorship). Prefer official team/brand press releases or announcements with explicit dates. If none are provided, return an empty list.
    - brand_category_urls: URLs provided in the answer that confirm the brand is a beauty/cosmetics brand (e.g., brand homepage/about page). If none, return an empty list.
    - player_status_urls: URLs provided in the answer that help verify the player’s active status at the time (e.g., WNBA.com player profile, official team roster page). If none, return an empty list.
    - role_terms: Any role terms present in the answer describing the partnership role (e.g., "brand ambassador", "spokesperson", "face of brand").
    - duration_terms: Any terms indicating partnership duration (e.g., "multi-year", "long-term", "multi season", "through 2026").

    IMPORTANT URL EXTRACTION RULES:
    - Only extract URLs that are explicitly present in the answer (including markdown links). Do not infer or create URLs.
    - Extract complete URLs. If missing a protocol, prepend "http://".
    - If a required URL field is not present in the answer, return null (for single URL) or [] (for list).

    Return a single JSON object matching the specified schema.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _collect_urls(*candidates: Any) -> List[str]:
    """Collect strings and lists of strings into a single de-duplicated list, preserving order."""
    seen = set()
    result: List[str] = []
    for item in candidates:
        if not item:
            continue
        if isinstance(item, str):
            u = item.strip()
            if u and u not in seen:
                seen.add(u)
                result.append(u)
        elif isinstance(item, list):
            for u in item:
                if not isinstance(u, str):
                    continue
                uu = u.strip()
                if uu and uu not in seen:
                    seen.add(uu)
                    result.append(uu)
    return result


def _safe_join(items: List[str]) -> str:
    clean = [x.strip() for x in items if isinstance(x, str) and x.strip()]
    return ", ".join(clean)


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_partnership(evaluator: Evaluator, parent_node, info: PartnershipInfo) -> None:
    """
    Build the verification tree for the beauty/cosmetics brand × WNBA player partnership and run checks.
    """
    # Create critical parallel node as main task container
    beauty_node = evaluator.add_parallel(
        id="beauty_brand_wnba_partnership",
        desc="Evaluate whether the identified beauty brand partnership with a WNBA player meets all specified criteria",
        parent=parent_node,
        critical=True
    )

    player_name = info.player_name or "the player"
    brand_name = info.brand_name or "the brand"
    player_team = info.player_team or "the player's WNBA team"
    announcement_date = info.announcement_date or "the announcement date"

    # Sources to use for different checks
    official_sources = _collect_urls(info.official_announcement_url)
    brand_category_sources = _collect_urls(info.brand_category_urls)
    prior_team_sources = _collect_urls(info.prior_team_sponsorship_urls)
    player_status_sources = _collect_urls(info.player_status_urls)
    products_list = [p for p in (info.products or []) if isinstance(p, str) and p.strip()]
    products_str = _safe_join(products_list)

    # 1) Partnership type: beauty/cosmetics brand with a WNBA player
    node_partnership_type = evaluator.add_leaf(
        id="partnership_type",
        desc="The partnership must be between a beauty or cosmetics brand and a WNBA player",
        parent=beauty_node,
        critical=True
    )
    claim_partnership_type = (
        f"This announcement documents a partnership between {brand_name}, which is a beauty or cosmetics brand, "
        f"and WNBA player {player_name}."
    )
    await evaluator.verify(
        claim=claim_partnership_type,
        node=node_partnership_type,
        sources=_collect_urls(official_sources, brand_category_sources),
        additional_instruction=(
            "Confirm both: (a) the brand is a beauty/cosmetics company, and (b) the partner is an individual WNBA player. "
            "To confirm (a), you may use the brand homepage/about page if provided. "
            "To confirm (b), the announcement should clearly name the WNBA player. "
            "If the provided sources don't explicitly support both points, mark as not supported."
        ),
    )

    # 2) Ambassadorship role (not a team sponsorship)
    node_ambassador = evaluator.add_leaf(
        id="ambassadorship_role",
        desc="The partnership must be a brand ambassadorship (not merely a team sponsorship)",
        parent=beauty_node,
        critical=True
    )
    claim_ambassador = (
        f"The announcement states that {player_name} is appointed as a brand ambassador (or equivalent, e.g., "
        f"spokesperson/face of brand) for {brand_name}, i.e., an individual ambassadorship rather than a team-only sponsorship."
    )
    await evaluator.verify(
        claim=claim_ambassador,
        node=node_ambassador,
        sources=official_sources,
        additional_instruction=(
            "Look for explicit terms like 'brand ambassador', 'ambassador', 'spokesperson', or 'face of the brand'. "
            "Ensure the arrangement is with the individual player, not a sponsorship of the team only."
        ),
    )

    # 3) Announcement in 2024
    node_2024 = evaluator.add_leaf(
        id="announcement_in_2024",
        desc="The partnership must have been publicly announced in 2024",
        parent=beauty_node,
        critical=True
    )
    claim_2024 = "The official announcement was published in the calendar year 2024."
    await evaluator.verify(
        claim=claim_2024,
        node=node_2024,
        sources=official_sources,
        additional_instruction=(
            "Use the page's publication date, dateline, or metadata to determine the year. "
            "It must be within 2024 (not 2023 or 2025). If the page is undated, treat as not supported."
        ),
    )

    # 4) Multi-year or long-term duration
    node_multi_year = evaluator.add_leaf(
        id="multi_year_duration",
        desc="The partnership must be described as multi-year or long-term",
        parent=beauty_node,
        critical=True
    )
    claim_multi_year = (
        "The announcement describes the partnership as multi-year or long-term (e.g., uses phrases like 'multi-year', "
        "'long-term', 'multi season', 'through <future year>')."
    )
    await evaluator.verify(
        claim=claim_multi_year,
        node=node_multi_year,
        sources=official_sources,
        additional_instruction=(
            "Accept synonyms and equivalent phrasing indicating duration beyond a single campaign or season."
        ),
    )

    # 5) At least two signature products named
    node_products = evaluator.add_leaf(
        id="featured_products",
        desc="The partnership must specifically name at least two signature products being promoted",
        parent=beauty_node,
        critical=True
    )
    if len(products_list) >= 2:
        claim_products = (
            f"The announcement explicitly names at least two specific products that {player_name} will promote, "
            f"including: {products_str}."
        )
    else:
        claim_products = (
            f"The announcement explicitly names at least two specific products that {player_name} will promote."
        )
    await evaluator.verify(
        claim=claim_products,
        node=node_products,
        sources=official_sources,
        additional_instruction=(
            "Verify that the page names two or more specific, proper-named products (not just generic categories). "
            "Minor naming variations are acceptable."
        ),
    )

    # 6) Prior brand–team sponsorship (before the ambassadorship)
    node_prior = evaluator.add_leaf(
        id="prior_team_sponsorship",
        desc="The brand must have had a prior sponsorship relationship with the player's WNBA team before the ambassadorship",
        parent=beauty_node,
        critical=True
    )
    if info.player_team and info.announcement_date:
        claim_prior = (
            f"Before {announcement_date}, {brand_name} had an official sponsorship relationship with the WNBA team "
            f"{player_team}."
        )
    elif info.player_team:
        claim_prior = (
            f"{brand_name} had an official sponsorship relationship with the WNBA team {player_team} before the "
            f"individual ambassadorship with {player_name}."
        )
    else:
        claim_prior = (
            f"Prior to the individual ambassadorship with {player_name}, {brand_name} had an official sponsorship "
            f"relationship with the player's WNBA team."
        )
    await evaluator.verify(
        claim=claim_prior,
        node=node_prior,
        sources=_collect_urls(prior_team_sources, official_sources),
        additional_instruction=(
            "Use prior press releases or official announcements (team or brand) that clearly indicate a sponsorship "
            "relationship with the WNBA team. Prefer sources dated earlier than the ambassadorship announcement. "
            "If the only evidence is dated the same day or later than the ambassadorship, do not consider it 'prior'."
        ),
    )

    # 7) Official press release or brand announcement exists
    node_official = evaluator.add_leaf(
        id="official_verification",
        desc="The partnership must be documented in an official press release or brand announcement with a verifiable URL",
        parent=beauty_node,
        critical=True
    )
    claim_official = (
        f"The provided URL is an official press release or brand announcement that documents the ambassadorship "
        f"between {brand_name} and {player_name}."
    )
    await evaluator.verify(
        claim=claim_official,
        node=node_official,
        sources=official_sources,
        additional_instruction=(
            "Accept brand-owned newsroom pages or reputable press release wires (e.g., Business Wire, PR Newswire). "
            "Do not accept third-party news articles or blog posts as 'official announcements'."
        ),
    )

    # 8) Player active (not retired) at time of announcement
    node_active = evaluator.add_leaf(
        id="active_player_status",
        desc="The WNBA player must be currently active (not retired) at the time of the partnership announcement",
        parent=beauty_node,
        critical=True
    )
    if info.announcement_date:
        claim_active = (
            f"On {announcement_date}, {player_name} was an active (non-retired) WNBA player."
        )
    else:
        claim_active = (
            f"At the time of the announcement, {player_name} was an active (non-retired) WNBA player."
        )
    await evaluator.verify(
        claim=claim_active,
        node=node_active,
        sources=_collect_urls(player_status_sources, official_sources),
        additional_instruction=(
            "Use WNBA.com player profile, official team roster pages, or equivalent official sources to confirm the "
            "player was active (not retired) at the announcement time. The official announcement may also imply current "
            "active status; however, explicit official roster/profile evidence is preferred when available."
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
    """
    Evaluate an answer for the beauty/cosmetics brand × WNBA player ambassadorship task.
    """
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

    # Extract structured info from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_partnership_info(),
        template_class=PartnershipInfo,
        extraction_name="partnership_info",
    )

    # Build verification nodes and run checks
    await verify_partnership(evaluator, root, extracted_info)

    # Return the evaluation summary
    return evaluator.get_summary()