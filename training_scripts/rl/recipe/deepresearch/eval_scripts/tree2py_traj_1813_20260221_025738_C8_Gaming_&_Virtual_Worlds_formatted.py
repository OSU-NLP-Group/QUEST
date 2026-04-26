import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "gaming_platform_transparency_comparison"
TASK_DESCRIPTION = (
    "Compare the server status transparency and communication practices of three major gaming platforms by researching and analyzing their publicly accessible status pages and official communication channels. "
    "For each of the three platforms you select, provide: 1. Status Page Information: The direct URL to their public server status page, confirmation that the page displays real-time service health status, "
    "whether the page shows component-level status (e.g., separate status for login services, matchmaking, game servers, store, etc.) rather than just an overall status indicator, whether the page includes "
    "historical incident and maintenance records, and whether the platform provides API access to their status data. 2. Communication Channels: Identification of any dedicated support or status social media "
    "accounts (such as Twitter/X accounts) and the specific social media handle(s) used for server status communications. 3. Maintenance Practices: How the platform communicates scheduled maintenance to users "
    "and a specific example of their maintenance timing, schedule pattern, or typical duration. All information must be supported by reference URLs from your research. Present findings in a clear, structured "
    "format allowing direct comparison across the three platforms."
)


# =========================
# Data Models
# =========================
class StatusFeatures(BaseModel):
    realtime_status: Optional[str] = None
    component_level_status: Optional[str] = None
    incident_history: Optional[str] = None
    api_access: Optional[str] = None


class SocialAccount(BaseModel):
    platform: Optional[str] = None  # e.g., "Twitter", "X", "Facebook"
    handle: Optional[str] = None    # e.g., "@PlayStation", "XboxSupport"
    url: Optional[str] = None       # direct account URL


class PlatformEntry(BaseModel):
    name: Optional[str] = None
    status_page_url: Optional[str] = None
    status_features: StatusFeatures = Field(default_factory=StatusFeatures)
    additional_feature_sources: List[str] = Field(default_factory=list)
    social_accounts: List[SocialAccount] = Field(default_factory=list)
    maintenance_method: Optional[str] = None
    maintenance_example: Optional[str] = None
    maintenance_sources: List[str] = Field(default_factory=list)


class PlatformsExtraction(BaseModel):
    platforms: List[PlatformEntry] = Field(default_factory=list)


# =========================
# Extraction Prompt
# =========================
def prompt_extract_platforms() -> str:
    return (
        "Extract up to the first three gaming platforms discussed in the answer, along with the required transparency and communication details. "
        "For each platform, return an object with the following fields:\n"
        "- name: The platform name (e.g., 'PlayStation Network', 'Xbox Live', 'Steam').\n"
        "- status_page_url: The direct URL to the public server status page. Must be a valid URL if provided.\n"
        "- status_features: An object containing:\n"
        "    * realtime_status: 'yes' if the status page displays real-time or near-real-time service health information, 'no' if not, 'unknown' if unclear.\n"
        "    * component_level_status: 'yes' if the page shows component-level status (e.g., login, matchmaking, store), 'no' otherwise, 'unknown' if unclear.\n"
        "    * incident_history: 'yes' if historical incident/maintenance records are included, 'no' otherwise, 'unknown' if unclear.\n"
        "    * api_access: 'yes' if the platform provides API access to status data, 'no' otherwise, 'unknown' if unclear.\n"
        "- additional_feature_sources: A list of URLs (if any) that support the feature claims (optional; include if mentioned in the answer).\n"
        "- social_accounts: A list of social media accounts used for support/status communications. Each item must include:\n"
        "    * platform: The social platform name (e.g., 'Twitter', 'X').\n"
        "    * handle: The account handle (e.g., '@XboxSupport').\n"
        "    * url: The account URL.\n"
        "- maintenance_method: A short description of how scheduled maintenance is communicated to users (e.g., 'status page notice', 'tweets from @XboxSupport').\n"
        "- maintenance_example: A specific example of maintenance timing, schedule pattern, or typical duration (as described in the answer).\n"
        "- maintenance_sources: A list of URL references that substantiate the maintenance method and example.\n\n"
        "Rules:\n"
        "1) Only extract what is explicitly present in the answer; do not invent information. Use null for missing fields.\n"
        "2) Return exactly up to 3 platforms (first three mentioned). If more are present, include only the first three; if fewer are present, include those.\n"
        "3) Extract only valid URLs; if a URL is missing protocol, prepend http://.\n"
    )


# =========================
# Helper Functions
# =========================
def normalize_bool_str(s: Optional[str]) -> Optional[bool]:
    if s is None:
        return None
    val = s.strip().lower()
    if val in {"yes", "true", "y", "t"}:
        return True
    if val in {"no", "false", "n", "f"}:
        return False
    return None


def combine_sources(*args: Optional[List[str] | str]) -> List[str]:
    urls: List[str] = []
    for item in args:
        if item is None:
            continue
        if isinstance(item, list):
            urls.extend([u for u in item if isinstance(u, str) and u.strip() != ""])
        elif isinstance(item, str):
            if item.strip() != "":
                urls.append(item)
    # Deduplicate while preserving order
    seen = set()
    result = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def first_social_account(platform: PlatformEntry) -> Optional[SocialAccount]:
    return platform.social_accounts[0] if platform.social_accounts else None


# =========================
# Verification Subtree for One Platform
# =========================
async def verify_platform(
    evaluator: Evaluator,
    parent_node,
    platform: PlatformEntry,
    idx: int,
) -> None:
    pid = idx + 1
    plat_name = platform.name or f"Platform {pid}"

    # Platform Analysis Node (parallel, non-critical)
    platform_node = evaluator.add_parallel(
        id=f"Platform_{pid}_Analysis",
        desc=f"Analysis of {plat_name}",
        parent=parent_node,
        critical=False,
    )

    # 1) Status Page Identification & Validation (sequential, critical)
    status_seq = evaluator.add_sequential(
        id=f"Platform_{pid}_Status_Page",
        desc=f"Status page identification and validation for {plat_name}",
        parent=platform_node,
        critical=True,
    )

    # 1.1) Status URL: format/existence check (custom, critical)
    status_url_present = isinstance(platform.status_page_url, str) and platform.status_page_url.strip() != "" and platform.status_page_url.strip().startswith(("http://", "https://"))
    evaluator.add_custom_node(
        result=status_url_present,
        id=f"Platform_{pid}_Status_URL",
        desc="Valid, publicly accessible status page URL provided with working HTTPS/HTTP link",
        parent=status_seq,
        critical=True,
    )

    # 1.2) URL Reference: verify the page is an official/public status page (leaf, critical)
    url_ref_leaf = evaluator.add_leaf(
        id=f"Platform_{pid}_URL_Reference",
        desc="Reference URL source confirming the status page exists",
        parent=status_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This webpage is the official public server/service status page for {plat_name}. It shows service status information.",
        node=url_ref_leaf,
        sources=platform.status_page_url,
        additional_instruction="Check for explicit phrases like 'Service Status', 'Server Status', 'Network Status', or similar, and any branding indicating it is the official status page for the platform.",
    )

    # 2) Status Page Features (parallel, non-critical)
    features_node = evaluator.add_parallel(
        id=f"Platform_{pid}_Status_Features",
        desc=f"Status page features and capabilities for {plat_name}",
        parent=platform_node,
        critical=False,
    )
    feature_sources = combine_sources(platform.status_page_url, platform.additional_feature_sources)

    # 2.1) Real-time/Near-real-time service health (leaf, critical)
    realtime_leaf = evaluator.add_leaf(
        id=f"Platform_{pid}_Realtime_Status",
        desc="Status page displays real-time or near-real-time service health information",
        parent=features_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This status page displays real-time or near-real-time service health information (e.g., automatic updates, 'last updated' timestamps, or live component statuses).",
        node=realtime_leaf,
        sources=feature_sources,
        additional_instruction="Look for signals of real-time updates such as 'Last updated', live component indicators, or statements about automatic/live updates.",
    )

    # 2.2) Component-level status tracking (leaf, critical)
    component_leaf = evaluator.add_leaf(
        id=f"Platform_{pid}_Component_Tracking",
        desc="Status page shows component-level status (e.g., login, matchmaking, store) rather than just overall status",
        parent=features_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This status page shows component-level statuses (e.g., separate entries for login, matchmaking, game servers, store, community, etc.), not just a single overall indicator.",
        node=component_leaf,
        sources=feature_sources,
        additional_instruction="Look for sections listing multiple services/components each with individual statuses.",
    )

    # 2.3) Incident history availability (leaf, non-critical)
    incident_leaf = evaluator.add_leaf(
        id=f"Platform_{pid}_Incident_History",
        desc="Status page provides access to historical incident and maintenance records",
        parent=features_node,
        critical=False,
    )
    await evaluator.verify(
        claim="This status page provides access to historical incident and/or maintenance records (e.g., 'Past Incidents', timeline, archive, or history pages).",
        node=incident_leaf,
        sources=feature_sources,
        additional_instruction="Check for 'Incident History', archives, timelines, or 'Past Incidents' sections that list resolved issues or maintenance events.",
    )

    # 2.4) API access to status data (leaf, non-critical)
    api_leaf = evaluator.add_leaf(
        id=f"Platform_{pid}_API_Access",
        desc="Platform provides API access to status data for programmatic queries",
        parent=features_node,
        critical=False,
    )
    await evaluator.verify(
        claim="This platform provides API access to its status data (e.g., documented endpoints, Statuspage.io API, RSS/JSON feeds, or developer docs).",
        node=api_leaf,
        sources=feature_sources,
        additional_instruction="Search for references to 'API', 'JSON', 'RSS', 'Statuspage API', or developer documentation indicating programmatic access to status information.",
    )

    # 3) Communication Channels (parallel, non-critical)
    comms_node = evaluator.add_parallel(
        id=f"Platform_{pid}_Communication_Channels",
        desc=f"Additional communication channels beyond status page for {plat_name}",
        parent=platform_node,
        critical=False,
    )
    social = first_social_account(platform)
    social_sources = social.url if social and social.url else None
    social_handle_text = social.handle if social and social.handle else ""

    # 3.1) Dedicated support/status social media account presence (leaf, non-critical)
    social_media_leaf = evaluator.add_leaf(
        id=f"Platform_{pid}_Social_Media",
        desc="Platform maintains dedicated support or status social media account (e.g., Twitter/X)",
        parent=comms_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"{plat_name} maintains a dedicated support/status social media account used to communicate server/status updates.",
        node=social_media_leaf,
        sources=social_sources,
        additional_instruction="Confirm the account page indicates it is for support/status updates (e.g., bio mentions support, status, outages, or maintenance).",
    )

    # 3.2) Specific social media handle provided with reference (leaf, non-critical)
    social_handle_leaf = evaluator.add_leaf(
        id=f"Platform_{pid}_Social_Handle",
        desc="Specific social media handle or account name provided with reference",
        parent=comms_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The official support/status account handle used for server status communications is '{social_handle_text}'.",
        node=social_handle_leaf,
        sources=social_sources,
        additional_instruction="Verify the account handle displayed on the social media page matches the provided handle. Allow minor formatting variants (e.g., with or without '@').",
    )

    # 4) Maintenance Practices (parallel, non-critical)
    maint_node = evaluator.add_parallel(
        id=f"Platform_{pid}_Maintenance_Practices",
        desc=f"Documented scheduled maintenance practices for {plat_name}",
        parent=platform_node,
        critical=False,
    )
    maint_sources = combine_sources(platform.maintenance_sources, platform.status_page_url)

    # 4.1) Maintenance schedule communication method (leaf, non-critical)
    maint_schedule_leaf = evaluator.add_leaf(
        id=f"Platform_{pid}_Maintenance_Schedule",
        desc="Platform communicates scheduled maintenance information to users",
        parent=maint_node,
        critical=False,
    )
    maint_method_text = platform.maintenance_method or "scheduled maintenance announcements via official channels"
    await evaluator.verify(
        claim=f"{plat_name} communicates scheduled maintenance to users, for example via: {maint_method_text}.",
        node=maint_schedule_leaf,
        sources=maint_sources,
        additional_instruction="Confirm the referenced page(s) show scheduled maintenance notices or guidelines about how maintenance is communicated (status page banners, support posts, or social announcements).",
    )

    # 4.2) Specific maintenance example (leaf, non-critical)
    maint_example_leaf = evaluator.add_leaf(
        id=f"Platform_{pid}_Maintenance_Example",
        desc="Specific example of maintenance timing, duration, or pattern provided with reference",
        parent=maint_node,
        critical=False,
    )
    maint_example_text = platform.maintenance_example or "an example maintenance timing or duration cited by the platform"
    await evaluator.verify(
        claim=f"A specific example of {plat_name}'s maintenance timing, schedule pattern, or typical duration is: {maint_example_text}.",
        node=maint_example_leaf,
        sources=maint_sources,
        additional_instruction="Verify that the reference includes a concrete example (date/time, typical window, or recurring schedule pattern). Allow paraphrase but ensure the example is truly supported.",
    )


# =========================
# Main Evaluation Function
# =========================
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
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

    # Extract platforms info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_platforms(),
        template_class=PlatformsExtraction,
        extraction_name="platforms_transparency",
    )

    platforms: List[PlatformEntry] = list(extracted.platforms or [])
    # Ensure exactly 3 entries (pad with empty if fewer; trim if more)
    platforms = platforms[:3]
    while len(platforms) < 3:
        platforms.append(PlatformEntry())

    # Build verification tree under a dedicated comparison node (optional for clarity)
    comparison_node = evaluator.add_parallel(
        id="Gaming_Platform_Transparency_Comparison",
        desc="Evaluate comparison of transparency and communication practices across three major gaming platforms",
        parent=root,
        critical=False,
    )

    # Verify each platform subtree
    for i in range(3):
        await verify_platform(evaluator, comparison_node, platforms[i], i)

    # Add custom info for summary
    evaluator.add_custom_info(
        info={"extracted_platform_count": len(extracted.platforms or []), "evaluated_platforms": 3},
        info_type="meta",
        info_name="extraction_stats",
    )

    return evaluator.get_summary()