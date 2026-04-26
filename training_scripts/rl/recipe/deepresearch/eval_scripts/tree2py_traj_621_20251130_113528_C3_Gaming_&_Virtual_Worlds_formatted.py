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
TASK_ID = "arc_raiders_support_infra"
TASK_DESCRIPTION = """
For players of Arc Raiders who are experiencing connectivity issues and want to understand the game's official support infrastructure, provide a comprehensive guide that includes: 
(1) the name of the developer/publisher and the game's release date, 
(2) the URL and main features of the official Help Center including at least three support categories available, 
(3) the official Discord server invite link and Twitter/X account handle for community communication, 
(4) whether Embark Studios provides a dedicated public status page for Arc Raiders server status (if not, explain what method players currently use to check server status), and 
(5) for comparison to industry standards, provide the URL of the Epic Games status page and describe at least two key features it offers for real-time status communication.
"""

# Ground truth/expectations based on rubric (used for verification)
EXPECTEDS = {
    "developer_publisher": "Embark Studios",
    "release_date": "October 30, 2025",
    "help_center_url": "id.embark.games/arc-raiders/support",
    "discord_invite": "discord.com/invite/arcraiders",
    "twitter_handle": "@ARCRaidersGame",
    "epic_status_url": "status.epicgames.com",
    "dedicated_status_page_exists": False
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HelpCenterInfo(BaseModel):
    url: Optional[str] = None
    features: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)


class CommunityInfo(BaseModel):
    discord_url: Optional[str] = None
    twitter_handle: Optional[str] = None
    twitter_url: Optional[str] = None


class StatusInfo(BaseModel):
    dedicated_status_page_exists: Optional[bool] = None
    method_description: Optional[str] = None
    method_urls: List[str] = Field(default_factory=list)


class EpicStatusInfo(BaseModel):
    url: Optional[str] = None
    features: List[str] = Field(default_factory=list)


class ArcRaidersSupportExtraction(BaseModel):
    developer_publisher: Optional[str] = None
    release_date: Optional[str] = None
    help_center: Optional[HelpCenterInfo] = None
    community: Optional[CommunityInfo] = None
    status: Optional[StatusInfo] = None
    epic_status: Optional[EpicStatusInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_arc_raiders_support() -> str:
    return """
    Extract the following fields from the answer text about Arc Raiders support information. Follow these rules:
    - Do not infer anything; only extract information explicitly provided in the answer.
    - If a requested field is not mentioned, return null or an empty list as appropriate.
    - For URLs, extract the exact URLs present in the answer (e.g., https://example.com/path or example.com/path). If missing protocol, keep as-is; do not invent.
    - If multiple options are provided for the same item, keep the most official-looking one or the first occurrence.

    Return a JSON with the following structure:
    {
      "developer_publisher": string | null,
      "release_date": string | null,
      "help_center": {
        "url": string | null,
        "features": string[],           // examples: "ticket creation", "submit a request", "browsable categories", "knowledge base search"
        "categories": string[]          // list of category names exactly as in the answer (at least three if available)
      },
      "community": {
        "discord_url": string | null,   // e.g., discord.com/invite/arcraiders or discord.gg/arcraiders
        "twitter_handle": string | null,// e.g., @ARCRaidersGame
        "twitter_url": string | null    // full URL if provided, e.g., https://x.com/ARCRaidersGame
      },
      "status": {
        "dedicated_status_page_exists": boolean | null, // true if the answer says there is a dedicated public status page for Arc Raiders; false if it says there isn't
        "method_description": string | null,            // how players currently check server status (e.g., rely on Downdetector)
        "method_urls": string[]                         // any URLs mentioned for status checking (e.g., Downdetector links)
      },
      "epic_status": {
        "url": string | null,           // Epic Games status page URL
        "features": string[]            // at least two features described in the answer, e.g., "real-time component status", "incident history"
      }
    }
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def normalize_label_list(items: List[str], k: int) -> List[str]:
    """Pick at most k non-empty, unique labels preserving order."""
    seen = set()
    out = []
    for s in items or []:
        s_norm = (s or "").strip()
        if not s_norm:
            continue
        if s_norm.lower() in seen:
            continue
        seen.add(s_norm.lower())
        out.append(s_norm)
        if len(out) >= k:
            break
    return out


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_developer_and_release(
    evaluator: Evaluator,
    parent_node,
    info: ArcRaidersSupportExtraction
) -> None:
    group = evaluator.add_parallel(
        id="Developer_Publisher_and_Release_Date",
        desc="Provide developer/publisher name and the game's release date.",
        parent=parent_node,
        critical=True
    )

    # Developer/Publisher Name
    dev_leaf = evaluator.add_leaf(
        id="Developer_Publisher_Name",
        desc="State the developer/publisher name as Embark Studios.",
        parent=group,
        critical=True
    )
    dev_claim = f"The answer states that the developer/publisher is 'Embark Studios'. Extracted value: '{(info.developer_publisher or '').strip()}'"
    await evaluator.verify(
        claim=dev_claim,
        node=dev_leaf,
        additional_instruction="Judge correct if the answer clearly identifies Embark Studios as the developer or publisher. Allow minor casing differences."
    )

    # Release Date
    rel_leaf = evaluator.add_leaf(
        id="Release_Date",
        desc="State the official release date as October 30, 2025.",
        parent=group,
        critical=True
    )
    release_text = (info.release_date or "").strip()
    rel_claim = f"The answer provides the game's release date as October 30, 2025. Extracted value: '{release_text}'. Treat 'Oct 30, 2025' or similar minor variations as equivalent."
    await evaluator.verify(
        claim=rel_claim,
        node=rel_leaf,
        additional_instruction="Pass if the extracted release_date equals October 30, 2025 allowing small format variations (e.g., Oct 30, 2025). Fail if any other date or missing."
    )


async def verify_help_center(
    evaluator: Evaluator,
    parent_node,
    info: ArcRaidersSupportExtraction
) -> None:
    group = evaluator.add_parallel(
        id="Official_Help_Center",
        desc="Provide the official Help Center URL plus its main features and at least three support categories.",
        parent=parent_node,
        critical=True
    )
    hc = info.help_center or HelpCenterInfo()

    # Help Center URL Equality check
    hc_url_leaf = evaluator.add_leaf(
        id="Help_Center_URL",
        desc="Provide the official Help Center URL: id.embark.games/arc-raiders/support.",
        parent=group,
        critical=True
    )
    provided_hc_url = (hc.url or "").strip()
    hc_url_claim = f"The provided Help Center URL equals 'id.embark.games/arc-raiders/support' when normalized (ignore http/https and trailing slash). Extracted URL: '{provided_hc_url}'."
    await evaluator.verify(
        claim=hc_url_claim,
        node=hc_url_leaf,
        additional_instruction="Consider 'http(s)://', 'www.', and trailing slashes as non-essential. If the path and host equal id.embark.games/arc-raiders/support, pass."
    )

    # Feature: Ticket creation / support request system
    ticket_leaf = evaluator.add_leaf(
        id="Help_Center_Feature_Ticket_Creation",
        desc="Describe that the Help Center provides a ticket creation/support request system.",
        parent=group,
        critical=True
    )
    ticket_claim = "The Help Center page provides a ticket creation or support request feature (e.g., 'Submit a request', 'Contact support', or 'Create ticket')."
    await evaluator.verify(
        claim=ticket_claim,
        node=ticket_leaf,
        sources=provided_hc_url if provided_hc_url else None,
        additional_instruction="Use the provided Help Center URL to check for a support request or ticket submission function."
    )

    # Feature: Browsable categories
    browse_leaf = evaluator.add_leaf(
        id="Help_Center_Feature_Browsable_Categories",
        desc="Describe that the Help Center provides browsable support categories.",
        parent=group,
        critical=True
    )
    browse_claim = "The Help Center organizes content into browsable support categories (e.g., tiles or lists of categories on the page)."
    await evaluator.verify(
        claim=browse_claim,
        node=browse_leaf,
        sources=provided_hc_url if provided_hc_url else None,
        additional_instruction="Verify that the Help Center landing or main support page shows clearly labeled categories to browse."
    )

    # Categories: At least three
    cat_leaf = evaluator.add_leaf(
        id="Support_Categories_At_Least_Three",
        desc="List at least three available support categories (must be consistent with the Help Center’s categories; examples include Technical, Release Notes, Getting Started, Social, Security and Anti-cheat, User Terms).",
        parent=group,
        critical=True
    )
    top3_cats = normalize_label_list(hc.categories or [], 3)
    cats_text = ", ".join(top3_cats) if top3_cats else "(none)"
    cat_claim = f"The Help Center includes the following categories: {cats_text}. These category names (allowing minor naming variations) appear on the Help Center page."
    await evaluator.verify(
        claim=cat_claim,
        node=cat_leaf,
        sources=provided_hc_url if provided_hc_url else None,
        additional_instruction="Pass only if at least three categories listed in the claim can be reasonably matched on the Help Center page; allow small naming differences (e.g., punctuation/casing)."
    )


async def verify_community_channels(
    evaluator: Evaluator,
    parent_node,
    info: ArcRaidersSupportExtraction
) -> None:
    group = evaluator.add_parallel(
        id="Community_Communication_Channels",
        desc="Provide the official Discord invite link and Twitter/X handle for community communication.",
        parent=parent_node,
        critical=True
    )
    comm = info.community or CommunityInfo()

    # Discord invite link
    discord_leaf = evaluator.add_leaf(
        id="Discord_Invite_Link",
        desc="Provide the official Discord invite link: discord.com/invite/arcraiders.",
        parent=group,
        critical=True
    )
    discord_url = (comm.discord_url or "").strip()
    discord_claim = f"The Discord invite link provided equals 'discord.com/invite/arcraiders' when normalized (ignore http/https, www, and allow 'discord.gg/arcraiders' as equivalent). Extracted: '{discord_url}'."
    await evaluator.verify(
        claim=discord_claim,
        node=discord_leaf,
        additional_instruction="Accept either discord.com/invite/arcraiders or discord.gg/arcraiders (with or without protocol). Fail otherwise."
    )

    # Twitter/X handle
    twitter_leaf = evaluator.add_leaf(
        id="Twitter_X_Handle",
        desc="Provide the official Twitter/X handle: @ARCRaidersGame.",
        parent=group,
        critical=True
    )
    tw_handle = (comm.twitter_handle or "").strip()
    twitter_claim = f"The official Twitter/X handle is @ARCRaidersGame (case-insensitive). Extracted handle: '{tw_handle}'."
    await evaluator.verify(
        claim=twitter_claim,
        node=twitter_leaf,
        additional_instruction="Pass if the extracted twitter handle matches @ARCRaidersGame ignoring case and whether it includes or omits the leading '@'."
    )


async def verify_status_assessment(
    evaluator: Evaluator,
    parent_node,
    info: ArcRaidersSupportExtraction
) -> None:
    group = evaluator.add_parallel(
        id="Arc_Raiders_Status_Page_Assessment",
        desc="State whether a dedicated public status page exists for Arc Raiders; if not, explain the current method players use to check server status.",
        parent=parent_node,
        critical=True
    )
    status = info.status or StatusInfo()

    # Dedicated status page exists (should be "no")
    exists_result = (status.dedicated_status_page_exists is False)
    evaluator.add_custom_node(
        result=exists_result,
        id="Dedicated_Status_Page_Exists",
        desc="Correctly state that Embark Studios does not currently operate a dedicated public Arc Raiders status page.",
        parent=group,
        critical=True
    )

    # Current player method for status checking
    method_leaf = evaluator.add_leaf(
        id="Current_Player_Method_For_Status_Checking",
        desc="Explain that players currently rely on third-party services (e.g., Downdetector) to check server status.",
        parent=group,
        critical=True
    )
    method_desc = (status.method_description or "").strip()
    method_claim = "Players currently rely on third-party services (for example, Downdetector) to check Arc Raiders server status."
    await evaluator.verify(
        claim=method_claim,
        node=method_leaf,
        additional_instruction=f"Pass if the answer's explanation conveys reliance on third-party status trackers (e.g., Downdetector). Extracted explanation: '{method_desc}'."
    )


async def verify_epic_status(
    evaluator: Evaluator,
    parent_node,
    info: ArcRaidersSupportExtraction
) -> None:
    group = evaluator.add_parallel(
        id="Epic_Games_Status_Page_Comparison",
        desc="Provide the Epic Games status page URL and describe at least two key features it offers for real-time status communication.",
        parent=parent_node,
        critical=True
    )
    epic = info.epic_status or EpicStatusInfo()

    # Epic status URL
    epic_url_leaf = evaluator.add_leaf(
        id="Epic_Status_Page_URL",
        desc="Provide the Epic Games status page URL: status.epicgames.com.",
        parent=group,
        critical=True
    )
    epic_url = (epic.url or "").strip()
    epic_url_claim = f"The Epic Games status page URL equals 'status.epicgames.com' when normalized (ignore http/https and trailing slash). Extracted URL: '{epic_url}'."
    await evaluator.verify(
        claim=epic_url_claim,
        node=epic_url_leaf,
        additional_instruction="Consider 'http(s)://' and trailing slashes non-essential for equality."
    )

    # Epic status features (at least two)
    epic_feat_leaf = evaluator.add_leaf(
        id="Epic_Status_Page_Features_At_Least_Two",
        desc="Describe at least two key features of the Epic Games status page (e.g., real-time component status display; incident history with timestamps/status updates).",
        parent=group,
        critical=True
    )
    epic_feats = normalize_label_list(epic.features or [], 2)
    feats_text = "; ".join(epic_feats) if epic_feats else "(none)"
    epic_feat_claim = f"The Epic Games status page offers these key features: {feats_text}. Examples include real-time component status, incident/maintenance history, and per-component updates."
    await evaluator.verify(
        claim=epic_feat_claim,
        node=epic_feat_leaf,
        sources=epic_url if epic_url else None,
        additional_instruction="Verify that at least two features listed are evident on the Epic Games status page (allow reasonable naming variations)."
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
    Evaluate an answer for the Arc Raiders support infrastructure task.
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
        default_model=model
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_arc_raiders_support(),
        template_class=ArcRaidersSupportExtraction,
        extraction_name="arc_raiders_support_extraction"
    )

    # Add ground-truth expectations for transparency
    evaluator.add_ground_truth({
        "expected_developer_publisher": EXPECTEDS["developer_publisher"],
        "expected_release_date": EXPECTEDS["release_date"],
        "expected_help_center_url": EXPECTEDS["help_center_url"],
        "expected_discord_invite": EXPECTEDS["discord_invite"],
        "expected_twitter_handle": EXPECTEDS["twitter_handle"],
        "expected_epic_status_url": EXPECTEDS["epic_status_url"],
        "expected_dedicated_status_page_exists": EXPECTEDS["dedicated_status_page_exists"],
    }, gt_type="rubric_expectations")

    # Build top-level node for this guide (critical root for rubric)
    guide_root = evaluator.add_parallel(
        id="Arc_Raiders_Support_Infrastructure_Guide",
        desc="Provide a guide covering Arc Raiders official support infrastructure, status communication, and comparison to Epic Games status standards.",
        parent=root,
        critical=True
    )

    # Sub-verifications
    await verify_developer_and_release(evaluator, guide_root, extracted)
    await verify_help_center(evaluator, guide_root, extracted)
    await verify_community_channels(evaluator, guide_root, extracted)
    await verify_status_assessment(evaluator, guide_root, extracted)
    await verify_epic_status(evaluator, guide_root, extracted)

    # Return summary
    return evaluator.get_summary()