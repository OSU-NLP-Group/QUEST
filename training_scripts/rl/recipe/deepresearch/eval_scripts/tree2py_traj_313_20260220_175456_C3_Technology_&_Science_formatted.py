import asyncio
import logging
from typing import List, Optional, Any, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tech_deploy_2026"
TASK_DESCRIPTION = (
    "A technology consultant is preparing a deployment plan for a client who wants to leverage cutting-edge AI "
    "capabilities, emergency communication features, and cloud gaming services. The consultant needs to verify the "
    "following specifications:\n\n"
    "1. Confirm that Apple and Google have officially announced a multi-year partnership where Google's Gemini models "
    "will power the next generation of Apple Intelligence and Siri. Identify the exact announcement date and provide "
    "the URL from an official Apple or Google source.\n\n"
    "2. Determine the planned iOS version and release timeframe (month and year) for when the Gemini-powered Siri "
    "features are scheduled to become publicly available. Provide a URL reference supporting this timeline.\n\n"
    "3. Identify which iPhone models (starting from which model number) support Emergency SOS via satellite "
    "functionality, and provide the URL from an official Apple source confirming this information.\n\n"
    "4. Verify whether FCC regulations require all U.S. wireless carriers to transmit all 911 emergency calls "
    "regardless of subscriber status, and provide the URL from an official FCC source (fcc.gov domain) or official "
    "legal code confirming this requirement.\n\n"
    "5. Determine the minimum internet connection speed (in Mbps) required to establish a PlayStation Plus Premium "
    "cloud streaming session, and the minimum speed required specifically for 1080p quality streaming. Provide a URL "
    "from an official PlayStation/Sony source confirming these requirements.\n\n"
    "Provide all required URLs and specifications in your answer."
)

ROOT_NODE_DESC = (
    "A technology consultant is verifying specifications for a client's planned deployment. The consultant must "
    "validate the official partnership announcement that will enable AI-powered features, confirm device requirements "
    "for emergency communication capabilities, verify regulatory compliance for emergency services, and validate "
    "internet infrastructure requirements for cloud-based services."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PartnershipInfo(BaseModel):
    announcement_urls: List[str] = Field(default_factory=list)
    announcement_date_text: Optional[str] = None
    ios_version: Optional[str] = None
    ios_release_timeframe: Optional[str] = None
    ios_release_urls: List[str] = Field(default_factory=list)


class EmergencyInfo(BaseModel):
    emergency_sos_urls: List[str] = Field(default_factory=list)
    emergency_starting_model: Optional[str] = None
    fcc_urls: List[str] = Field(default_factory=list)


class CloudInfo(BaseModel):
    playstation_urls: List[str] = Field(default_factory=list)
    min_speed_session: Optional[str] = None
    min_speed_1080p: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_partnership() -> str:
    return """
    Extract the information about the Apple–Google AI partnership and related iOS release details from the answer text.

    Return a JSON object with the following fields:
    - announcement_urls: array of all URLs the answer cites as the official Apple or Google announcement (press release, newsroom, official blog, investor relations). Extract exactly as shown; include full URLs.
    - announcement_date_text: the announcement date stated in the answer (verbatim string, e.g., "January 12, 2026"); set null if not explicitly stated.
    - ios_version: the iOS version where Gemini-powered Siri/features are scheduled to launch (verbatim string, e.g., "iOS 26.4"); set null if missing.
    - ios_release_timeframe: the release timeframe stated in the answer (verbatim string, e.g., "spring 2026 (March or early April 2026)"); set null if missing.
    - ios_release_urls: array of URLs the answer cites to support the release version/timeframe; set to [] if none.

    Only extract what is explicitly in the answer text. Do not invent any URLs or values.
    """


def prompt_extract_emergency() -> str:
    return """
    Extract the information about iPhone Emergency SOS via satellite compatibility and FCC 911 regulations.

    Return a JSON object with:
    - emergency_sos_urls: array of URLs the answer cites from Apple (e.g., Apple Support or other official apple.com pages) about Emergency SOS via satellite device compatibility; [] if none.
    - emergency_starting_model: the earliest iPhone model the answer claims supports Emergency SOS via satellite (verbatim, e.g., "iPhone 14"); null if not stated.
    - fcc_urls: array of URLs the answer cites from the FCC (fcc.gov) or official legal code (e.g., eCFR) that confirm carriers must transmit all 911 calls regardless of subscriber status; [] if none.

    Extract exactly as presented in the answer text; do not infer missing items.
    """


def prompt_extract_cloud() -> str:
    return """
    Extract the information about PlayStation Plus Premium cloud streaming speed requirements.

    Return a JSON object with:
    - playstation_urls: array of URLs the answer cites from official PlayStation/Sony sources; [] if none.
    - min_speed_session: the minimum connection speed (verbatim, e.g., "5 Mbps") required to establish any cloud streaming session; null if not stated.
    - min_speed_1080p: the minimum connection speed (verbatim, e.g., "13 Mbps") required specifically for 1080p quality streaming; null if not stated.

    Only extract values explicitly given in the answer text and the cited URLs. Do not invent values.
    """


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_partnership_subtree(evaluator: Evaluator, parent_node, info: PartnershipInfo) -> None:
    # Parent: Partnership_Announcement_Validation (sequential, critical)
    partnership_node = evaluator.add_sequential(
        id="Partnership_Announcement_Validation",
        desc="Verify that Apple and Google officially announced a multi-year AI partnership where Google's Gemini models will power the next generation of Apple Intelligence and Siri. The announcement must be from an official source and contain specific details about the partnership and timeline.",
        parent=parent_node,
        critical=True,
    )

    # 1) Official Source Verification (parallel, critical)
    official_src_node = evaluator.add_parallel(
        id="Official_Source_Verification",
        desc="The partnership announcement must come from an official company source (e.g., official blog, press release, or investor relations page from Apple or Google).",
        parent=partnership_node,
        critical=True,
    )

    # Existence check for announcement URL(s)
    evaluator.add_custom_node(
        result=bool(info.announcement_urls),
        id="Partnership_Announcement_URL_Provided",
        desc="At least one official announcement URL is provided by the answer.",
        parent=official_src_node,
        critical=True,
    )

    # Official announcement URL: verify official source
    url_official_leaf = evaluator.add_leaf(
        id="Partnership_Announcement_URL",
        desc="Provide the URL of the official announcement from Apple's or Google's official channels.",
        parent=official_src_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This page is hosted on an official Apple or Google domain (e.g., apple.com, blog.google, about.google, ai.google, abc.xyz) and is an official announcement (newsroom/press/blog) about the Apple–Google AI partnership.",
        node=url_official_leaf,
        sources=info.announcement_urls,
        additional_instruction="Focus on whether the URL is an official Apple or Google property and is an announcement/press blog. Do not use third-party articles.",
    )

    # 2) Announcement Date Verification
    ann_date_leaf = evaluator.add_leaf(
        id="Announcement_Date_Verification",
        desc="The official announcement must have been made on January 12, 2026.",
        parent=partnership_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The official Apple/Google announcement was published on January 12, 2026.",
        node=ann_date_leaf,
        sources=info.announcement_urls,
        additional_instruction="Verify the publication date shown on the page itself. Minor timezone format differences are acceptable, but the date should clearly be January 12, 2026.",
    )

    # 3) Partnership Content Verification
    content_leaf = evaluator.add_leaf(
        id="Partnership_Content_Verification",
        desc="The partnership announcement must specify that Google's Gemini models will power the next generation of Apple Intelligence and Siri.",
        parent=partnership_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The announcement explicitly states that Google's Gemini models will power the next generation of Apple Intelligence and Siri.",
        node=content_leaf,
        sources=info.announcement_urls,
        additional_instruction="Look for explicit mention that Gemini (Google's models) will power Apple Intelligence and Siri (next generation).",
    )

    # 4) iOS Release Details (parallel, critical)
    ios_release_node = evaluator.add_parallel(
        id="iOS_Release_Details",
        desc="Verify the iOS version and release timeframe for the Gemini-powered features.",
        parent=partnership_node,
        critical=True,
    )

    # Ensure URL(s) provided for release timeline
    evaluator.add_custom_node(
        result=bool(info.ios_release_urls),
        id="iOS_Release_URL",
        desc="Provide a URL reference supporting the iOS 26.4 spring 2026 release timeline for Gemini-powered features.",
        parent=ios_release_node,
        critical=True,
    )

    # iOS Version Verification
    ios_version_leaf = evaluator.add_leaf(
        id="iOS_Version_Verification",
        desc="The Gemini-powered features are scheduled to launch in iOS 26.4.",
        parent=ios_release_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The page states that Gemini-powered Siri/features are scheduled to launch in iOS 26.4.",
        node=ios_version_leaf,
        sources=info.ios_release_urls,
        additional_instruction="The page should explicitly mention 'iOS 26.4' as the launch version for these features.",
    )

    # Release Timeframe Verification
    timeframe_leaf = evaluator.add_leaf(
        id="Release_Timeframe_Verification",
        desc="The release is planned for spring 2026 (March or early April 2026).",
        parent=ios_release_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The page states that the release is planned for spring 2026 (March or early April 2026).",
        node=timeframe_leaf,
        sources=info.ios_release_urls,
        additional_instruction="Look for explicit timeframe mentions like 'spring 2026', 'March 2026', or 'early April 2026'.",
    )


async def build_emergency_subtree(evaluator: Evaluator, parent_node, info: EmergencyInfo) -> None:
    # Parent: Emergency_Communication_Requirements (parallel, critical)
    emergency_node = evaluator.add_parallel(
        id="Emergency_Communication_Requirements",
        desc="Verify device requirements for Emergency SOS via satellite and confirm FCC regulatory compliance for emergency services.",
        parent=parent_node,
        critical=True,
    )

    # Device Compatibility Verification (parallel, critical)
    device_node = evaluator.add_parallel(
        id="Device_Compatibility_Verification",
        desc="Emergency SOS via satellite must be available on iPhone 14 and all later iPhone models.",
        parent=emergency_node,
        critical=True,
    )

    # Emergency_SOS_Device_URL (leaf): verify official Apple URL and that it confirms the iPhone 14 or later support claim
    emergency_url_leaf = evaluator.add_leaf(
        id="Emergency_SOS_Device_URL",
        desc="Provide a URL reference from Apple Support or official Apple source confirming device compatibility for Emergency SOS via satellite.",
        parent=device_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This page is hosted on an official Apple domain (apple.com) and confirms that Emergency SOS via satellite is available on iPhone 14 and later iPhone models.",
        node=emergency_url_leaf,
        sources=info.emergency_sos_urls,
        additional_instruction="Accept Apple Support or other official Apple pages. The page must explicitly indicate iPhone 14 or later support.",
    )

    # FCC Compliance Verification (parallel, critical)
    fcc_node = evaluator.add_parallel(
        id="FCC_Compliance_Verification",
        desc="Confirm that FCC regulations require all wireless carriers to transmit all 911 calls to emergency services (PSAP) regardless of subscriber status.",
        parent=emergency_node,
        critical=True,
    )

    # FCC_Requirement_URL (leaf): verify official FCC/legal URL and that it confirms the requirement
    fcc_url_leaf = evaluator.add_leaf(
        id="FCC_Requirement_URL",
        desc="Provide a URL reference from an official FCC source (fcc.gov) or official legal code confirming the requirement for carriers to transmit all 911 calls.",
        parent=fcc_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This page is on an official FCC domain (fcc.gov) or an official legal code (e.g., eCFR.gov), and it states that wireless carriers must transmit all 911 calls to emergency services regardless of subscriber status.",
        node=fcc_url_leaf,
        sources=info.fcc_urls,
        additional_instruction="Look for rules requiring carriers to transmit all 911 calls, including from non-service initialized devices or non-subscribers.",
    )


async def build_cloud_subtree(evaluator: Evaluator, parent_node, info: CloudInfo) -> None:
    # Parent: Cloud_Service_Infrastructure_Requirements (parallel, critical)
    cloud_node = evaluator.add_parallel(
        id="Cloud_Service_Infrastructure_Requirements",
        desc="Verify minimum internet speed requirements for PlayStation Plus Premium cloud streaming services.",
        parent=parent_node,
        critical=True,
    )

    # Minimum connection speed (leaf)
    min_speed_leaf = evaluator.add_leaf(
        id="Minimum_Connection_Speed",
        desc="PlayStation Plus Premium cloud streaming requires a minimum internet connection of 5 Mbps to establish a streaming session.",
        parent=cloud_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This official PlayStation/Sony page states that the minimum connection required to establish PlayStation Plus Premium cloud streaming is 5 Mbps.",
        node=min_speed_leaf,
        sources=info.playstation_urls,
        additional_instruction="Look for minimum speed requirements specifically for starting/establishing a cloud streaming session.",
    )

    # 1080p streaming speed requirement (leaf)
    hd_speed_leaf = evaluator.add_leaf(
        id="HD_Streaming_Speed_Requirement",
        desc="For 1080p quality streaming on PlayStation Plus Premium, the minimum internet speed requirement must be at least 13 Mbps.",
        parent=cloud_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This official PlayStation/Sony page states that 1080p cloud streaming requires at least 13 Mbps.",
        node=hd_speed_leaf,
        sources=info.playstation_urls,
        additional_instruction="Verify that the page explicitly mentions the 1080p requirement as at least 13 Mbps (or directly equivalent wording).",
    )

    # PlayStation speed URL (leaf): official URL that confirms speed requirements
    ps_url_leaf = evaluator.add_leaf(
        id="PlayStation_Speed_URL",
        desc="Provide a URL reference from an official PlayStation/Sony source confirming the internet speed requirements for cloud streaming.",
        parent=cloud_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This page is on an official PlayStation or Sony domain (e.g., playstation.com, sony.com) and mentions the cloud streaming internet speed requirements.",
        node=ps_url_leaf,
        sources=info.playstation_urls,
        additional_instruction="Focus on official domains and explicit mention of cloud streaming speed requirements.",
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
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

    # Create main critical node to mirror rubric root
    main_node = evaluator.add_parallel(
        id="Technology_Deployment_Verification",
        desc=ROOT_NODE_DESC,
        parent=root,
        critical=True,
    )

    # Extract sections from the answer
    partnership_info = await evaluator.extract(
        prompt=prompt_extract_partnership(),
        template_class=PartnershipInfo,
        extraction_name="partnership_info",
    )

    emergency_info = await evaluator.extract(
        prompt=prompt_extract_emergency(),
        template_class=EmergencyInfo,
        extraction_name="emergency_info",
    )

    cloud_info = await evaluator.extract(
        prompt=prompt_extract_cloud(),
        template_class=CloudInfo,
        extraction_name="cloud_info",
    )

    # Build verification subtrees
    await build_partnership_subtree(evaluator, main_node, partnership_info)
    await build_emergency_subtree(evaluator, main_node, emergency_info)
    await build_cloud_subtree(evaluator, main_node, cloud_info)

    # Return summary
    return evaluator.get_summary()