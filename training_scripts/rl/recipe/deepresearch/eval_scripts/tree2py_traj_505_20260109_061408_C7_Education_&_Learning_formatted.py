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
TASK_ID = "online_learning_platforms_apps_certs_pricing"
TASK_DESCRIPTION = """
Identify four different online learning platforms or MOOC providers that are currently operational and offer courses in English. For each platform, provide: 
(1) confirmation that official mobile applications are available for both iOS and Android with a link to the platform's mobile app page or app store listings, 
(2) details about the type of certificates offered (professional certificates, certification prep courses, or course completion certificates) with a reference link, and 
(3) specific subscription pricing information (monthly and/or annual rates) or details about their free tier with a reference link to their official pricing page.
"""

CURRENT_MONTH_YEAR = "January 2026"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AppLinks(BaseModel):
    ios_app_url: Optional[str] = None
    android_app_url: Optional[str] = None
    mobile_apps_page_url: Optional[str] = None


class CertificateInfo(BaseModel):
    description: Optional[str] = None  # e.g., "professional certificates", "course completion certificates", "cert prep"
    reference_url: Optional[str] = None


class PricingInfo(BaseModel):
    description: Optional[str] = None  # e.g., "monthly $59, annual $399" or "free tier with limited access"
    pricing_page_url: Optional[str] = None


class PlatformItem(BaseModel):
    name: Optional[str] = None
    homepage_url: Optional[str] = None
    english_evidence_url: Optional[str] = None  # any official page showing English interface or English courses
    video_evidence_url: Optional[str] = None  # official page showing video-based content/lessons
    app_links: Optional[AppLinks] = None
    certificates: Optional[CertificateInfo] = None
    pricing: Optional[PricingInfo] = None
    additional_sources: List[str] = Field(default_factory=list)


class PlatformsExtraction(BaseModel):
    platforms: List[PlatformItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_platforms() -> str:
    return """
    Extract up to the first four distinct online learning platforms or MOOC providers mentioned in the answer. 
    For each platform, extract the following fields exactly as presented in the answer:

    - name: The platform/provider's name.
    - homepage_url: Official homepage URL of the platform/provider (if present).
    - english_evidence_url: An official page URL showing English-language interface or English course pages (if present).
    - video_evidence_url: An official page URL demonstrating video-based course content, features, or lessons (if present).
    - app_links:
        - ios_app_url: The official iOS App Store listing URL for the platform's app (if present).
        - android_app_url: The official Google Play listing URL for the platform's app (if present).
        - mobile_apps_page_url: The platform's official mobile/apps page URL (if present), separate from store listings.
    - certificates:
        - description: The text describing certificate type(s) offered (e.g., "professional certificates", "course completion certificates", "certification prep courses").
        - reference_url: The official page URL that describes the certificate(s).
    - pricing:
        - description: The text describing specific subscription pricing (monthly/annual) OR free-tier details with any limitations.
        - pricing_page_url: The official pricing page URL.
    - additional_sources: Any other official URLs cited in the answer specifically about this platform (if present).

    RULES:
    - Extract only URLs explicitly present in the answer. Do not invent or infer any URLs.
    - Always include complete URLs with protocol (http:// or https://). If missing, prepend http://.
    - If a field is not mentioned, set it to null (or an empty list for arrays).
    - If more than four platforms are mentioned, extract only the first four (in order).
    - Ensure platforms are distinct by name; if duplicates occur, include only the first occurrence and skip subsequent duplicates.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_urls(*urls: Optional[str]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]

def _name_or_placeholder(name: Optional[str], idx: int) -> str:
    return name.strip() if name else f"Platform #{idx + 1}"

# --------------------------------------------------------------------------- #
# Verification for a single platform                                          #
# --------------------------------------------------------------------------- #
async def verify_platform(
    evaluator: Evaluator,
    parent_node,
    platform: PlatformItem,
    index: int,
) -> None:
    plat_name = _name_or_placeholder(platform.name, index)
    plat_node = evaluator.add_parallel(
        id=f"platform_{index+1}",
        desc=f"Platform {index + 1} satisfies all platform, mobile app, certificate, and pricing/free-tier requirements with references",
        parent=parent_node,
        critical=False  # Non-critical so that each platform contributes partial credit
    )

    # 1) Publicly accessible online learning platform or MOOC provider
    node_is_mooc = evaluator.add_leaf(
        id=f"platform_{index+1}_is_public_online_learning_or_mooc",
        desc=f"Platform {index + 1} is a publicly accessible online learning platform or MOOC provider",
        parent=plat_node,
        critical=True
    )
    mooc_sources = _non_empty_urls(platform.homepage_url, platform.english_evidence_url)
    mooc_claim = f"'{plat_name}' operates as a publicly accessible online learning platform or MOOC provider that offers courses to the general public."
    await evaluator.verify(
        claim=mooc_claim,
        node=node_is_mooc,
        sources=mooc_sources or None,
        additional_instruction="Verify that the official page(s) indicate this is an online learning platform or MOOC provider (e.g., offers courses or programs to the public). Accept synonyms like 'online course platform', 'learning marketplace', 'MOOC'."
    )

    # 2) Operational and accessible in English as of January 2026
    node_operational_english = evaluator.add_leaf(
        id=f"platform_{index+1}_operational_and_english_jan_2026",
        desc=f"Platform {index + 1} is currently operational and accessible in English as of {CURRENT_MONTH_YEAR} (supported by a reference link)",
        parent=plat_node,
        critical=True
    )
    op_sources = _non_empty_urls(platform.english_evidence_url, platform.homepage_url)
    op_claim = f"As of {CURRENT_MONTH_YEAR}, '{plat_name}' is operational and provides English-language interface or English courses."
    await evaluator.verify(
        claim=op_claim,
        node=node_operational_english,
        sources=op_sources or None,
        additional_instruction=f"Check that the page(s) load and show English-language content. If the page shows English text or provides an English UI, consider it accessible in English as of {CURRENT_MONTH_YEAR}."
    )

    # 3) Video-based course content
    node_video = evaluator.add_leaf(
        id=f"platform_{index+1}_video_based_content",
        desc=f"Platform {index + 1} offers video-based course content (supported by a reference link)",
        parent=plat_node,
        critical=True
    )
    video_sources = _non_empty_urls(platform.video_evidence_url, platform.homepage_url)
    video_claim = f"'{plat_name}' offers video-based course content (e.g., video lessons, lectures, or streamed classes)."
    await evaluator.verify(
        claim=video_claim,
        node=node_video,
        sources=video_sources or None,
        additional_instruction="Look for terms like 'video lessons', 'watch lectures', 'video-based content', 'streaming classes'. Minor wording variation is acceptable."
    )

    # 4) Mobile apps for both iOS and Android with links
    mobile_main = evaluator.add_sequential(
        id=f"platform_{index+1}_mobile_apps_ios_and_android_with_links",
        desc=f"Platform {index + 1} has official mobile apps for both iOS and Android and provides links to official app pages or store listings",
        parent=plat_node,
        critical=True
    )

    ios_link_present = evaluator.add_custom_node(
        result=bool(platform.app_links and platform.app_links.ios_app_url),
        id=f"platform_{index+1}_ios_link_provided",
        desc=f"Platform {index + 1} provides an iOS App Store link or official iOS app page",
        parent=mobile_main,
        critical=True
    )
    ios_leaf = evaluator.add_leaf(
        id=f"platform_{index+1}_ios_app_supported",
        desc=f"Platform {index + 1} official iOS app existence is supported by the provided link",
        parent=mobile_main,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{plat_name}' has an official iOS app.",
        node=ios_leaf,
        sources=_non_empty_urls(platform.app_links.ios_app_url if platform.app_links else None) or None,
        additional_instruction="If the URL is an Apple App Store listing for this platform's app, pass. If it's a platform official mobile-app page that clearly states an iOS app exists, that also counts."
    )

    android_link_present = evaluator.add_custom_node(
        result=bool(platform.app_links and platform.app_links.android_app_url),
        id=f"platform_{index+1}_android_link_provided",
        desc=f"Platform {index + 1} provides a Google Play link or official Android app page",
        parent=mobile_main,
        critical=True
    )
    android_leaf = evaluator.add_leaf(
        id=f"platform_{index+1}_android_app_supported",
        desc=f"Platform {index + 1} official Android app existence is supported by the provided link",
        parent=mobile_main,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{plat_name}' has an official Android app.",
        node=android_leaf,
        sources=_non_empty_urls(platform.app_links.android_app_url if platform.app_links else None) or None,
        additional_instruction="If the URL is a Google Play listing for this platform's app, pass. If it's a platform official mobile-app page that clearly states an Android app exists, that also counts."
    )

    # 5) Certificates: type(s) offered + reference link
    cert_main = evaluator.add_sequential(
        id=f"platform_{index+1}_certificates_type_and_reference",
        desc=f"Platform {index + 1} describes certificate type(s) offered and provides a reference link",
        parent=plat_node,
        critical=True
    )

    cert_link_present = evaluator.add_custom_node(
        result=bool(platform.certificates and platform.certificates.reference_url),
        id=f"platform_{index+1}_cert_reference_link_provided",
        desc=f"Platform {index + 1} provides an official page link describing certificate(s)",
        parent=cert_main,
        critical=True
    )

    cert_leaf = evaluator.add_leaf(
        id=f"platform_{index+1}_certificate_type_supported",
        desc=f"Platform {index + 1} certificate type(s) are supported by the reference page",
        parent=cert_main,
        critical=True
    )
    cert_desc = platform.certificates.description if platform.certificates and platform.certificates.description else "certificate(s)"
    cert_claim = f"According to its official page, '{plat_name}' offers {cert_desc}."
    await evaluator.verify(
        claim=cert_claim,
        node=cert_leaf,
        sources=_non_empty_urls(platform.certificates.reference_url if platform.certificates else None) or None,
        additional_instruction="Accept types including 'professional certificates', 'course completion certificates', or 'certification prep courses'. The page should explicitly mention certificate offerings."
    )

    # 6) Pricing or free tier details + official pricing page link
    price_main = evaluator.add_sequential(
        id=f"platform_{index+1}_pricing_or_free_tier_with_official_pricing_link",
        desc=f"Platform {index + 1} provides specific subscription pricing OR free-tier details with an official pricing page link",
        parent=plat_node,
        critical=True
    )

    price_link_present = evaluator.add_custom_node(
        result=bool(platform.pricing and platform.pricing.pricing_page_url),
        id=f"platform_{index+1}_pricing_link_provided",
        desc=f"Platform {index + 1} provides an official pricing page link",
        parent=price_main,
        critical=True
    )

    price_leaf = evaluator.add_leaf(
        id=f"platform_{index+1}_pricing_info_supported",
        desc=f"Platform {index + 1} pricing/free-tier details are supported by the official pricing page",
        parent=price_main,
        critical=True
    )
    price_desc = platform.pricing.description if platform.pricing and platform.pricing.description else "pricing or free-tier details"
    price_claim = f"The official pricing page for '{plat_name}' shows {price_desc}."
    await evaluator.verify(
        claim=price_claim,
        node=price_leaf,
        sources=_non_empty_urls(platform.pricing.pricing_page_url if platform.pricing else None) or None,
        additional_instruction="Verify presence of specific monthly and/or annual subscription rates OR clear free-tier details (including limitations) on the official pricing page."
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
    Evaluate an answer for the online learning platforms / apps / certificates / pricing task.
    """
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

    # Extract platform info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_platforms(),
        template_class=PlatformsExtraction,
        extraction_name="platforms_extraction",
    )

    # Prepare first four distinct platforms (by name)
    unique_platforms: List[PlatformItem] = []
    seen_names_lower: set = set()
    for p in extracted.platforms:
        nm = (p.name or "").strip().lower()
        if not nm:
            unique_platforms.append(p)  # allow inclusion if fewer good entries; will likely fail downstream verifications
            if len(unique_platforms) >= 4:
                break
            continue
        if nm in seen_names_lower:
            continue
        seen_names_lower.add(nm)
        unique_platforms.append(p)
        if len(unique_platforms) >= 4:
            break

    # Pad if fewer than 4 to keep tree structure consistent
    while len(unique_platforms) < 4:
        unique_platforms.append(PlatformItem())

    # Critical check: "four platforms and distinctness" — aligned with evaluation guidelines:
    # We require at least four mentioned and first four are distinct.
    names_first4 = [(p.name or "").strip().lower() for p in unique_platforms]
    distinct_first4 = len([n for n in names_first4 if n]) == 4 and len(set([n for n in names_first4 if n])) == 4
    at_least_four_in_answer = len(extracted.platforms) >= 4

    evaluator.add_custom_info(
        info={
            "total_platforms_mentioned": len(extracted.platforms),
            "first_four_names": [p.name for p in unique_platforms]
        },
        info_type="extraction_summary",
        info_name="platforms_summary"
    )

    # Add critical distinctness/count node
    evaluator.add_custom_node(
        result=(at_least_four_in_answer and distinct_first4),
        id="platform_count_and_distinctness",
        desc="Response identifies exactly four platforms and they are all different (distinct providers)",
        parent=root,
        critical=True
    )

    # Build per-platform verification subtrees
    for i in range(4):
        await verify_platform(
            evaluator=evaluator,
            parent_node=root,
            platform=unique_platforms[i],
            index=i
        )

    return evaluator.get_summary()