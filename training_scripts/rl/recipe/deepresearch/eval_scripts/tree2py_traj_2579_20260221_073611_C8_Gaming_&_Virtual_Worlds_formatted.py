import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "gaming_platform_outages_2025_2026"
TASK_DESCRIPTION = (
    "I'm researching the reliability of major gaming platforms for a report on service availability. "
    "Identify four different major gaming platforms (from among Steam, Epic Games, Xbox Live, PlayStation Network, "
    "Nintendo Switch Online, Riot Games, Blizzard Battle.net, or EA) that experienced documented service outages or "
    "disruptions in 2025 or 2026. For each platform, provide: (1) The platform name, (2) The URL of the platform's "
    "official status page, (3) The specific date of a documented outage that occurred in 2025 or 2026, "
    "(4) A description of the outage duration or impact, and (5) A reference URL from a news article, status page, "
    "or monitoring service that documents this outage."
)

ALLOWED_PLATFORMS = [
    "Steam",
    "Epic Games",
    "Xbox Live",
    "PlayStation Network",
    "Nintendo Switch Online",
    "Riot Games",
    "Blizzard Battle.net",
    "EA",
]

ALLOWED_YEARS = {"2025", "2026"}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PlatformItem(BaseModel):
    name: Optional[str] = None
    status_url: Optional[str] = None
    outage_date: Optional[str] = None
    outage_description: Optional[str] = None
    reference_url: Optional[str] = None


class PlatformsExtraction(BaseModel):
    items: List[PlatformItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_platforms() -> str:
    return (
        "Extract the platform outage information explicitly mentioned in the answer. For each platform item, return:\n"
        "1. name: The platform name as stated (e.g., Steam, Epic Games, Xbox Live, PlayStation Network, Nintendo Switch Online, Riot Games, Blizzard Battle.net, EA). If a synonym is used (e.g., PSN), extract that exact text.\n"
        "2. status_url: The official status or service health page URL for that platform (e.g., status.playstation.com, status.riotgames.com, status.epicgames.com, status.blizzard.com/battle.net, status.xbox.com). "
        "If an official status page URL is not provided in the answer, set this to null.\n"
        "3. outage_date: The specific date of a documented outage that occurred in 2025 or 2026, as presented in the answer (keep the original string).\n"
        "4. outage_description: A brief description of the outage duration or impact (as presented in the answer).\n"
        "5. reference_url: A URL to a news article, monitoring service page (e.g., Downdetector, statuspage), or official status/incident post that documents the outage. "
        "If not provided, set this to null.\n"
        "Return a JSON object with an 'items' array containing all platforms the answer mentioned. "
        "If the answer mentions more than four platforms, include them all; the evaluator will select the first four. "
        "If any field is missing for a platform, set it to null."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_text(s: Optional[str]) -> str:
    return (s or "").strip()


def canonicalize_platform_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = name.strip().lower()
    if "steam" in n:
        return "Steam"
    if "epic" in n:
        return "Epic Games"
    if "xbox" in n or "xbox live" in n or "xbox network" in n:
        return "Xbox Live"
    if "playstation" in n or "psn" in n or "sony" in n:
        return "PlayStation Network"
    if "nintendo" in n or "switch online" in n:
        return "Nintendo Switch Online"
    if "riot" in n:
        return "Riot Games"
    if "battle.net" in n or "blizzard" in n or "bnet" in n:
        return "Blizzard Battle.net"
    if n == "ea" or "electronic arts" in n or "origin" in n or "ea app" in n:
        return "EA"
    # If none matched, return name capitalized as is
    return name.strip()


def extract_years_from_text(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return re.findall(r"\b(20(?:25|26))\b", text)


def is_year_in_allowed(text: Optional[str]) -> bool:
    years = set(extract_years_from_text(text))
    return bool(years & ALLOWED_YEARS)


def first_k(items: List[PlatformItem], k: int) -> List[PlatformItem]:
    arr = items[:k]
    while len(arr) < k:
        arr.append(PlatformItem())
    return arr


# --------------------------------------------------------------------------- #
# Verification logic per platform                                             #
# --------------------------------------------------------------------------- #
async def verify_one_platform(
    evaluator: Evaluator,
    parent_node,
    item: PlatformItem,
    index: int,
    prior_canonical_names: List[str],
) -> None:
    plat_idx = index + 1
    plat_node = evaluator.add_parallel(
        id=f"Platform_{plat_idx}",
        desc=f"{['First','Second','Third','Fourth'][index]} gaming platform information",
        parent=parent_node,
        critical=False,  # allow partial credit across platforms
    )

    # Prepare normalized values
    raw_name = normalize_text(item.name)
    canon_name = canonicalize_platform_name(raw_name)
    status_url = normalize_text(item.status_url)
    outage_date = normalize_text(item.outage_date)
    outage_desc = normalize_text(item.outage_description)
    reference_url = normalize_text(item.reference_url)

    # 1) Name validity
    name_valid_leaf = evaluator.add_leaf(
        id=f"Platform_{plat_idx}_Name_Valid",
        desc=(
            "Platform name is one of the major gaming platforms "
            "(Steam, Epic Games, Xbox Live, PlayStation Network, Nintendo Switch Online, Riot Games, Blizzard Battle.net, or EA)"
        ),
        parent=plat_node,
        critical=True,
    )
    name_claim = (
        f"The platform name '{raw_name}' refers to one of these major gaming platforms: "
        f"{', '.join(ALLOWED_PLATFORMS)}."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_valid_leaf,
        additional_instruction=(
            "Consider common synonyms and abbreviations (e.g., PSN -> PlayStation Network, Battle.net -> Blizzard Battle.net, "
            "Xbox Network -> Xbox Live). If the provided name clearly maps to one of the listed platforms, judge as correct."
        ),
    )

    # 1b) Name uniqueness (for platforms after the first)
    if index > 0:
        unique_result = canon_name is not None and canon_name not in prior_canonical_names
        evaluator.add_custom_node(
            result=unique_result,
            id=f"Platform_{plat_idx}_Name_Unique",
            desc=f"Platform name is different from previously listed platforms",
            parent=plat_node,
            critical=True,
        )
    # Update seen names for later platforms (only if non-empty)
    if canon_name:
        prior_canonical_names.append(canon_name)

    # 2) Status URL presence
    evaluator.add_custom_node(
        result=bool(status_url),
        id=f"Platform_{plat_idx}_Status_URL_Provided",
        desc="Official status page URL is provided",
        parent=plat_node,
        critical=True,
    )

    # 2b) Status URL is official and accessible
    status_official_leaf = evaluator.add_leaf(
        id=f"Platform_{plat_idx}_Status_URL_Official",
        desc="Official status page URL is accessible and is an official status/health page for the platform",
        parent=plat_node,
        critical=True,
    )
    status_claim = (
        f"This webpage is the official status or service health page for {canon_name or raw_name}."
    )
    await evaluator.verify(
        claim=status_claim,
        node=status_official_leaf,
        sources=status_url if status_url else None,
        additional_instruction=(
            "Confirm the page is an official status/health page (look for cues like 'Status', 'Service status', 'Incidents', "
            "'Uptime', or known official status domains such as status.riotgames.com, status.playstation.com, "
            "status.xbox.com, status.epicgames.com, Blizzard Battle.net status, Steam status, EA status/origin, etc.). "
            "If the page is inaccessible, or not an official status page, judge as not supported."
        ),
    )

    # 3) Outage Date presence
    evaluator.add_custom_node(
        result=bool(outage_date),
        id=f"Platform_{plat_idx}_Recent_Outage_Date_Provided",
        desc="Date of a documented outage is provided",
        parent=plat_node,
        critical=True,
    )

    # 3b) Outage Date is in 2025 or 2026
    evaluator.add_custom_node(
        result=is_year_in_allowed(outage_date),
        id=f"Platform_{plat_idx}_Recent_Outage_Date_In_Range",
        desc="Provided outage date is in 2025 or 2026",
        parent=plat_node,
        critical=True,
    )

    # 4) Outage duration/impact presence
    evaluator.add_custom_node(
        result=bool(outage_desc),
        id=f"Platform_{plat_idx}_Outage_Duration_Provided",
        desc="Outage duration or impact description is provided",
        parent=plat_node,
        critical=True,
    )

    # 5) Reference URL presence
    evaluator.add_custom_node(
        result=bool(reference_url),
        id=f"Platform_{plat_idx}_URL_Reference_Provided",
        desc="Reference URL supporting the outage information is provided",
        parent=plat_node,
        critical=True,
    )

    # 5b) Reference supports the outage on the stated date
    ref_support_leaf = evaluator.add_leaf(
        id=f"Platform_{plat_idx}_URL_Reference_Supports_Outage",
        desc="Reference URL documents the outage (platform and date) in 2025 or 2026",
        parent=plat_node,
        critical=True,
    )
    ref_claim = (
        f"A documented outage affecting {canon_name or raw_name} occurred on {outage_date} in 2025 or 2026."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_support_leaf,
        sources=[u for u in [reference_url, status_url] if u],
        additional_instruction=(
            "Verify that at least one provided URL (news article, monitoring service, or official status/incident page) "
            "explicitly documents an outage for the specified platform. Prefer exact date matches; allow minor timezone/date "
            "boundary variations (±1 day). If no outage is documented on or around the stated date, judge as not supported."
        ),
    )

    # 4b) Reference supports the stated duration/impact
    impact_support_leaf = evaluator.add_leaf(
        id=f"Platform_{plat_idx}_Outage_Duration_Supported",
        desc="Reference URL documents the outage duration or impact consistent with the provided description",
        parent=plat_node,
        critical=True,
    )
    impact_claim = (
        f"The outage described for {canon_name or raw_name} had the following duration or impact: '{outage_desc}'."
    )
    await evaluator.verify(
        claim=impact_claim,
        node=impact_support_leaf,
        sources=[u for u in [reference_url, status_url] if u],
        additional_instruction=(
            "Check whether the reference page corroborates the duration/impact (e.g., 'lasting 2 hours', "
            "'widespread login failures', 'matchmaking unavailable', 'degraded performance'). "
            "Allow reasonable paraphrases and minor wording variations."
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
    Evaluate an answer for the gaming platforms outages (2025/2026) task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Platforms evaluated independently
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

    # NOTE: Framework constraint—critical parent cannot have non-critical children.
    # We set root as non-critical to allow partial credit across platforms.
    root.critical = False

    # Extract structured platform items from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_platforms(),
        template_class=PlatformsExtraction,
        extraction_name="platforms_extraction",
    )

    # Add ground truth info for transparency
    evaluator.add_ground_truth({
        "allowed_platforms": ALLOWED_PLATFORMS,
        "required_years": sorted(list(ALLOWED_YEARS)),
        "requirement": "Four unique platforms; each must include official status URL, outage date (2025/2026), impact/duration, and a reference URL."
    })

    platforms = first_k(extracted.items or [], 4)

    # Track canonical names to enforce uniqueness across platforms
    seen_canonical_names: List[str] = []

    # Build verification nodes for each platform
    for idx, item in enumerate(platforms):
        await verify_one_platform(
            evaluator=evaluator,
            parent_node=root,
            item=item,
            index=idx,
            prior_canonical_names=seen_canonical_names,
        )

    # Return the evaluation summary
    return evaluator.get_summary()