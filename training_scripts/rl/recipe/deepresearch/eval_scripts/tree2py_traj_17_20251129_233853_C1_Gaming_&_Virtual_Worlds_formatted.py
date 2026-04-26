import asyncio
import logging
from typing import Any, Dict, Optional, List

from pydantic import BaseModel

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "gaming_platform_status_page"
TASK_DESCRIPTION = (
    "Identify a major gaming platform or publisher that maintains an official status page monitoring at least 20 "
    "distinct technical service components (such as Login, Matchmaking, Voice Chat, Item Shop, Authentication, "
    "Leaderboards, Parties, Friends, Stats, Player Data Storage, Anti-cheat, etc.) in addition to tracking individual "
    "games. Provide the name of the platform and the URL of their official status page."
)

# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class PlatformStatusPageExtraction(BaseModel):
    platform_name: Optional[str] = None
    status_page_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction Prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_platform_status_page() -> str:
    return (
        "Extract from the answer the following fields:\n"
        "1) platform_name: The name of the gaming platform or publisher.\n"
        "2) status_page_url: The URL to the official, platform-owned status page.\n"
        "Rules:\n"
        "- The platform_name must be explicitly mentioned in the answer text.\n"
        "- The status_page_url must be a valid URL explicitly present in the answer text. If a URL is missing a protocol, prepend http://.\n"
        "- Do not invent or infer values not present in the answer.\n"
        "- If either field is not present, return null for that field."
    )


# --------------------------------------------------------------------------- #
# Verification Logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_platform_status_page(
    evaluator: Evaluator,
    parent_node,
    extracted: PlatformStatusPageExtraction
) -> None:
    """
    Build verification nodes and run checks for the identified platform/publisher status page.
    """
    # Create the main critical parallel node to encapsulate all required checks
    main_node = evaluator.add_parallel(
        id="Gaming_Platform_Status_Page_Identification",
        desc="Identify a major gaming platform/publisher and provide its official status page URL that tracks individual games and monitors at least 20 technical service components with current operational status shown.",
        parent=parent_node,
        critical=True
    )

    platform_name = extracted.platform_name or ""
    status_url = extracted.status_page_url

    # Create leaf nodes for each required verification (all critical under the main critical node)
    # 1) Platform Is Major And MultiService
    node_major = evaluator.add_leaf(
        id="Platform_Is_Major_And_MultiService",
        desc="The provided platform/publisher qualifies as a major gaming platform or publisher with multiple games or services.",
        parent=main_node,
        critical=True
    )

    # 2) Official Public Status Page URL Provided (and official/platform-owned)
    node_official_url = evaluator.add_leaf(
        id="Official_Public_Status_Page_URL_Provided",
        desc="Provides a URL to an official (platform-owned), publicly accessible, verifiable status page.",
        parent=main_node,
        critical=True
    )

    # 3) Status Page Tracks Individual Games
    node_tracks_games = evaluator.add_leaf(
        id="Status_Page_Tracks_Individual_Games",
        desc="The status page tracks individual games (i.e., includes game-specific status entries) in addition to component/service monitoring.",
        parent=main_node,
        critical=True
    )

    # 4) Monitors At Least 20 Distinct Technical Service Components
    node_20_components = evaluator.add_leaf(
        id="Monitors_At_Least_20_Distinct_Technical_Service_Components",
        desc="The status page monitors at least 20 distinct technical service components/features (e.g., login, matchmaking, voice chat, authentication, leaderboards, etc.), not merely a list of game titles.",
        parent=main_node,
        critical=True
    )

    # 5) Displays Current Operational Status For Components
    node_current_status = evaluator.add_leaf(
        id="Displays_Current_Operational_Status_For_Components",
        desc="The status page displays current operational status (e.g., operational/degraded/partial outage) for the monitored components.",
        parent=main_node,
        critical=True
    )

    # Build claims and additional instructions
    claim_major = (
        f"The organization '{platform_name}' is a major gaming platform or publisher that operates multiple games or services."
        if platform_name else
        "The identified organization is a major gaming platform or publisher that operates multiple games or services."
    )
    add_ins_major = (
        "Use widely known industry knowledge to judge major gaming platforms/publishers (e.g., Microsoft Xbox, Sony PlayStation, Nintendo, Electronic Arts, Activision Blizzard, Epic Games, Valve, Ubisoft, Riot Games, etc.). "
        "If the platform_name is missing or the entity is minor/unclear, mark as Incorrect."
    )

    claim_official = (
        f"This URL is the official, platform-owned status page for '{platform_name}', and it is publicly accessible."
        if platform_name else
        "This URL is the official, platform-owned status page for the named organization, and it is publicly accessible."
    )
    add_ins_official = (
        "Verify page ownership and official branding via domain and on-page cues (header/footer/company name). "
        "Third-party aggregators (e.g., community or unofficial sites) should be marked as Incorrect. "
        "If the URL is missing or inaccessible, mark as Incorrect."
    )

    claim_tracks_games = (
        "This status page includes game-specific status entries (e.g., titles like Fortnite, Rocket League, Fall Guys) in addition to service/component monitoring."
    )
    add_ins_tracks_games = (
        "Look for explicit game titles or dedicated sections/pages per game within the status page. "
        "Filters, tabs, or listings that enumerate individual games qualify. "
        "If only generic platform services are listed without any game-specific entries, mark as Incorrect. "
        "If the URL is missing, mark as Incorrect."
    )

    claim_20_components = (
        "This status page monitors at least 20 distinct technical service components/features (not just a list of game titles)."
    )
    add_ins_20_components = (
        "Count unique components/features such as Login, Matchmaking, Voice Chat, Authentication, Leaderboards, Parties, Friends, Stats, Inventory, Purchases/Commerce, Store/Item Shop, Cloud Saves/Player Data Storage, Anti-cheat, Messaging/Presence, Server Browser, Group Management, Achievements, DLC/Entitlements, Networking, Regions, APIs, etc. "
        "Do not count duplicate region entries or repeated items per game; count distinct component types. "
        "If fewer than 20 distinct components are present or the URL is missing, mark as Incorrect."
    )

    claim_current_status = (
        "This status page displays current operational status (e.g., Operational, Degraded Performance, Partial Outage, Major Outage) for its monitored components."
    )
    add_ins_current_status = (
        "Look for explicit status indicators next to components or in component detail pages. "
        "If the page only lists components without any current status labels/indicators, mark as Incorrect. "
        "If the URL is missing, mark as Incorrect."
    )

    # Prepare batch verification tuples: (claim, sources, node, additional_instruction)
    claims_and_sources: List[tuple[str, Optional[str], Any, Optional[str]]] = [
        (claim_major, None, node_major, add_ins_major),
        (claim_official, status_url, node_official_url, add_ins_official),
        (claim_tracks_games, status_url, node_tracks_games, add_ins_tracks_games),
        (claim_20_components, status_url, node_20_components, add_ins_20_components),
        (claim_current_status, status_url, node_current_status, add_ins_current_status),
    ]

    # Execute verifications concurrently to avoid unintended sibling gating
    await evaluator.batch_verify(claims_and_sources)


# --------------------------------------------------------------------------- #
# Main Evaluation Entry Point                                                 #
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
    Evaluate an answer for the Gaming Platform Status Page identification task.
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
        default_model=model
    )

    # Extract platform name and status page URL
    extracted = await evaluator.extract(
        prompt=prompt_extract_platform_status_page(),
        template_class=PlatformStatusPageExtraction,
        extraction_name="platform_status_page_extraction"
    )

    # Build verification tree and run checks
    await verify_platform_status_page(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()