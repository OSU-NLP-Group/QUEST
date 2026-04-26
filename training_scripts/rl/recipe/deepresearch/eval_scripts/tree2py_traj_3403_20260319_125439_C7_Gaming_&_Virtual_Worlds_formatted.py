import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "gaming_status_pages"
TASK_DESCRIPTION = """
I'm researching how major gaming companies communicate service status to their users for a project on digital infrastructure transparency. Please identify four major gaming platforms or publishers that provide official public status pages for monitoring their online services. For each of the four platforms, provide: (1) The platform or publisher name, (2) The official status page URL (the direct link to where service status is displayed), and (3) Confirmation of one specific type of service monitoring: For the first platform, verify that the status page monitors authentication or login services; for the second platform, verify that it monitors matchmaking services; for the third platform, verify that it monitors social features (such as friends, parties, or messaging); and for the fourth platform, verify that it monitors voice chat services. Each platform must have a publicly accessible official status page (no login required), and the monitored services must be explicitly listed as separate components or categories on that status page.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PlatformEntry(BaseModel):
    name: Optional[str] = None
    status_url: Optional[str] = None


class PlatformsExtraction(BaseModel):
    platforms: List[PlatformEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_platforms() -> str:
    return """
    Extract up to four (4) gaming platforms or game publishers with their official public service status page URLs from the provided answer.

    For each item, extract:
    - name: The platform or publisher name exactly as written in the answer.
    - status_url: The direct URL to the official status page where live service status (components/indicators) is displayed. Do NOT invent URLs. If multiple URLs are given, choose the one that directly displays the service status (not a general support landing page). If none is present, set to null.

    Output JSON:
    {
      "platforms": [
        {"name": string|null, "status_url": string|null},
        ...
      ]
    }

    Rules:
    - Only extract items explicitly present in the answer.
    - The status_url must be a valid URL present in the answer. If a URL is missing a protocol, prepend http://.
    - If the answer lists more than 4, return only the first 4.
    - If fewer than 4 are present, return as many as available; the rest can be omitted (the evaluator will pad as needed).
    """.strip()


# --------------------------------------------------------------------------- #
# Helper functions for verification                                           #
# --------------------------------------------------------------------------- #
def _service_meta_by_index(idx: int) -> Dict[str, str]:
    if idx == 0:
        return {
            "key": "login",
            "desc": "The status page explicitly lists authentication or login services as a separate monitored component with operational status indicators",
            "node_suffix": "Login_Service",
            "add_ins": (
                "Confirm the page includes a clearly labeled component/category for authentication or account sign-in, "
                "e.g., 'Authentication', 'Login', 'Sign-In', 'Account Login', 'Identity/SSO'. "
                "It must appear as a separate monitored item (component/category) with status indicators "
                "such as 'Operational', 'Degraded', 'Partial Outage', etc. Do not accept mere mentions in text; "
                "it must be a listed component or category."
            ),
        }
    if idx == 1:
        return {
            "key": "matchmaking",
            "desc": "The status page explicitly lists matchmaking services as a separate monitored component with operational status indicators",
            "node_suffix": "Matchmaking_Service",
            "add_ins": (
                "Confirm the page includes a clearly labeled component/category for 'Matchmaking' or equivalents like "
                "'Game Sessions', 'Session Management', 'Lobby/Matchmaking', or 'Multiplayer Matchmaking'. "
                "It must be a separate monitored component/category with status indicators."
            ),
        }
    if idx == 2:
        return {
            "key": "social",
            "desc": "The status page explicitly lists social features (friends, parties, or messaging) as a separate monitored component with operational status indicators",
            "node_suffix": "Social_Service",
            "add_ins": (
                "Confirm the page includes a clearly labeled component/category for social features such as "
                "'Friends', 'Parties', 'Groups', 'Messaging/Chat', or 'Presence'. "
                "It must be a separate monitored component/category with status indicators."
            ),
        }
    return {
        "key": "voice",
        "desc": "The status page explicitly lists voice chat services as a separate monitored component with operational status indicators",
        "node_suffix": "Voice_Service",
        "add_ins": (
            "Confirm the page includes a clearly labeled component/category for voice communications such as "
            "'Voice Chat', 'Party Voice', 'In‑Game Voice', 'VOIP'. "
            "It must be a separate monitored component/category with status indicators."
        ),
    }


def _platform_node_desc_by_index(idx: int) -> str:
    if idx == 0:
        return "First gaming platform with authentication/login service monitoring"
    if idx == 1:
        return "Second gaming platform with matchmaking service monitoring"
    if idx == 2:
        return "Third gaming platform with social features monitoring"
    return "Fourth gaming platform with voice chat service monitoring"


# --------------------------------------------------------------------------- #
# Verification logic per platform                                             #
# --------------------------------------------------------------------------- #
async def verify_one_platform(
    evaluator: Evaluator,
    parent_node,
    idx: int,
    platform: PlatformEntry,
) -> None:
    """
    Build verification subtree for a single platform.
    """
    p_index = idx + 1
    p_name = platform.name or ""
    p_url = platform.status_url or ""

    # Platform group node (non-critical to allow partial credit across platforms)
    platform_node = evaluator.add_parallel(
        id=f"platform_{p_index}",
        desc=_platform_node_desc_by_index(idx),
        parent=parent_node,
        critical=False,
    )

    # -------------------- Name checks --------------------
    # 1) Name exists (critical existence gate)
    evaluator.add_custom_node(
        result=bool(p_name.strip()),
        id=f"platform_{p_index}_name_exists",
        desc="Platform/publisher name is provided",
        parent=platform_node,
        critical=True,
    )

    # 2) Name represents a major platform/publisher (critical)
    name_major_node = evaluator.add_leaf(
        id=f"platform_{p_index}_name_is_major",
        desc="The platform name represents a major gaming platform or major game publisher",
        parent=platform_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{p_name}' is a major gaming platform or a major game publisher.",
        node=name_major_node,
        additional_instruction=(
            "Accept examples like PlayStation, Xbox, Nintendo, Steam, Epic Games, Ubisoft, EA, Riot Games, Blizzard, "
            "Rockstar, Bethesda, etc. Use general industry knowledge; evaluate reasonableness with common sense."
        ),
    )

    # -------------------- URL checks --------------------
    # 3) URL exists (critical existence gate)
    evaluator.add_custom_node(
        result=(p_url.strip().startswith("http://") or p_url.strip().startswith("https://")),
        id=f"platform_{p_index}_url_exists",
        desc="Official status page URL is provided",
        parent=platform_node,
        critical=True,
    )

    # 4) URL is an official public status page (critical)
    official_status_node = evaluator.add_leaf(
        id=f"platform_{p_index}_url_official_status",
        desc="The URL is an official status page that displays live service status/components (not a third‑party aggregator or forum)",
        parent=platform_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The page at {p_url} is the official public service status page for {p_name}, showing components/services and incident/outage information.",
        node=official_status_node,
        sources=p_url,
        additional_instruction=(
            "Accept dedicated status portals (e.g., Atlassian Statuspage, Better Stack, Status.io) if branded for the company "
            "or under the company's domain/subdomain (e.g., status.company.com or company.statuspage.io labeled clearly). "
            "Reject third‑party aggregators (e.g., Downdetector, IsTheServiceDown), user forums, social media, or generic support articles. "
            "The page should show service health/components and incident history, not just static text."
        ),
    )

    # 5) Publicly accessible without login (critical)
    public_node = evaluator.add_leaf(
        id=f"platform_{p_index}_url_public",
        desc="The status page is publicly accessible without login to view status information",
        parent=platform_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The status information on this page can be viewed without logging in.",
        node=public_node,
        sources=p_url,
        additional_instruction=(
            "If the page shows a login wall, authentication prompt, or 401/403 notice before viewing status, it should be rejected. "
            "Publicly accessible means the components/status are visible to any visitor."
        ),
    )

    # -------------------- Service-specific monitoring check --------------------
    meta = _service_meta_by_index(idx)
    service_node = evaluator.add_leaf(
        id=f"platform_{p_index}_{meta['node_suffix']}",
        desc=meta["desc"],
        parent=platform_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"On this status page for {p_name}, there is a separate monitored component or category for {meta['key']} "
            "and it has operational status indicators (e.g., Operational/Degraded/Outage)."
        ),
        node=service_node,
        sources=p_url,
        additional_instruction=meta["add_ins"],
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
    Evaluate an answer for the 'gaming_status_pages' task.
    """
    # Initialize evaluator (root is always non-critical in framework)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent platforms; allow partial credit
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify four major gaming platforms or publishers with official public status pages, each monitoring specific service types",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract platform entries from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_platforms(),
        template_class=PlatformsExtraction,
        extraction_name="platforms_extraction",
    )

    platforms = list(extracted.platforms[:4])
    while len(platforms) < 4:
        platforms.append(PlatformEntry())

    # Build and run verifications for each of the four platforms
    for idx in range(4):
        await verify_one_platform(
            evaluator=evaluator,
            parent_node=root,
            idx=idx,
            platform=platforms[idx],
        )

    # Return evaluation summary
    return evaluator.get_summary()