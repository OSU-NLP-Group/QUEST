import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "vr_platforms_2026_accessibility"
TASK_DESCRIPTION = (
    "A community education nonprofit organization is planning to launch a year-round virtual engagement program for 2026 that will host quarterly large-group workshops, weekly small discussion sessions, and provide a persistent virtual space for member networking. Due to budget constraints and the need to ensure broad accessibility, they require technology that does not mandate specialized hardware purchases for participants.\n\n"
    "The organization needs to identify exactly four (4) distinct virtual world or social VR platforms that are currently operational as of February 2026 and meet ALL of the following technical specifications and accessibility requirements:\n\n"
    "System Requirements:\n"
    "- Must support desktop/PC mode without requiring a VR headset for full participation\n"
    "- Minimum system requirements must be accessible to users with: Windows 10 or Windows 11 operating systems, 8GB RAM or less, and mid-range graphics cards (NVIDIA GeForce GTX 970 / AMD Radeon R9 290 equivalent or lower specification GPUs)\n"
    "- Must support Intel i5-4590 / AMD FX 8350 equivalent or lower-specification processors\n\n"
    "Platform Accessibility:\n"
    "- Must offer free account creation and basic platform access (no mandatory purchase required to join and participate)\n"
    "- Must be accessible via standard desktop/PC without requiring VR equipment\n\n"
    "Capacity and Communication Features:\n"
    "- Must support at least 40 concurrent users per world/instance/session\n"
    "- Must include built-in voice communication functionality\n\n"
    "Platform Status:\n"
    "- Must be actively maintained and operational as of February 2026\n"
    "- Must support user-created content, customizable avatars, or user-generated worlds/experiences\n\n"
    "For each of the four platforms identified, provide:\n"
    "1. The platform name\n"
    "2. Specific minimum RAM requirement\n"
    "3. Minimum GPU specification\n"
    "4. Concurrent user capacity per instance\n"
    "5. Confirmation of free access availability\n"
    "6. Confirmation of desktop mode availability\n"
    "7. Confirmation of voice chat functionality\n"
    "8. At least one reference URL from official platform documentation or reliable technical specifications source that verifies these requirements"
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PlatformRecord(BaseModel):
    name: Optional[str] = None
    os_supported: Optional[str] = None  # e.g., "Windows 10/11", "Windows 10", etc.
    min_ram: Optional[str] = None       # e.g., "4 GB", "8GB", "6 GB"
    min_gpu: Optional[str] = None       # e.g., "NVIDIA GTX 960", "AMD R9 290", "Intel Iris"
    min_cpu: Optional[str] = None       # e.g., "Intel i5-4590", "AMD FX 8350", "Ryzen 3"
    concurrent_capacity: Optional[str] = None  # e.g., "50 users", "100+", "up to 200"
    free_access: Optional[str] = None          # textual confirmation or "Yes/No"
    desktop_mode: Optional[str] = None         # textual confirmation or "Yes/No"
    voice_chat: Optional[str] = None           # textual confirmation or "Yes/No"
    user_content: Optional[str] = None         # textual confirmation about UGC/avatars/worlds
    reference_urls: List[str] = Field(default_factory=list)


class PlatformsExtraction(BaseModel):
    platforms: List[PlatformRecord] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_platforms() -> str:
    return """
    Extract up to four (4) distinct virtual world or social VR platforms described in the answer, along with the required technical and accessibility details.

    For each platform, extract the following fields exactly as stated in the answer:
    - name: The platform name (string).
    - os_supported: The operating system compatibility text as stated (e.g., 'Windows 10/11', 'Windows PC').
    - min_ram: The minimum RAM requirement text (e.g., '4 GB', '8GB', etc.).
    - min_gpu: The minimum GPU requirement text (e.g., 'NVIDIA GTX 960', 'AMD R9 290', 'Intel HD 4000').
    - min_cpu: The minimum CPU requirement text (e.g., 'Intel i5-4590', 'AMD FX 8350').
    - concurrent_capacity: The stated concurrent user capacity per world/instance/session (e.g., '50 users', '100', 'up to 200').
    - free_access: The text confirming free account creation/basic access (e.g., 'free to join', 'free account').
    - desktop_mode: The text confirming desktop/PC mode availability without requiring a VR headset (e.g., 'Desktop mode supported').
    - voice_chat: The text confirming built-in voice communication (e.g., 'voice chat included').
    - user_content: The text confirming user-created content/customizable avatars/user-generated worlds support (e.g., 'user-created worlds', 'UGC', 'custom avatars supported').
    - reference_urls: An array of URLs the answer provides as evidence for this platform's requirements. Extract only actual URLs. If none are provided, return an empty array.

    Rules:
    - Only extract information explicitly present in the answer; do not infer or invent.
    - Extract at most four platforms. If the answer lists more than four, include only the first four in order of appearance.
    - If a required field is not mentioned for a platform, set it to null (or an empty array for reference_urls).
    - For URLs, accept plain URLs or markdown links; extract the URL string.
    - Do not include duplicate platforms; if duplicates appear, only keep the first occurrence.

    Return a JSON object with a 'platforms' array containing up to four objects with the fields above.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return name.strip().lower()


def is_unique_among(name: Optional[str], all_names: List[Optional[str]]) -> bool:
    if not name or not name.strip():
        return False
    n = normalize_name(name)
    matches = [normalize_name(x) for x in all_names if normalize_name(x) is not None]
    return matches.count(n) == 1


# --------------------------------------------------------------------------- #
# Verification for one platform                                               #
# --------------------------------------------------------------------------- #
async def verify_one_platform(
    evaluator: Evaluator,
    parent_node,
    plat: PlatformRecord,
    plat_index: int,
    all_names: List[Optional[str]],
) -> None:
    """
    Build verification subtree and perform checks for a single platform.
    """

    # Top-level node for this platform (non-critical; allows partial credit per platform)
    platform_node = evaluator.add_parallel(
        id=f"Platform_{plat_index + 1}",
        desc=f"{['First','Second','Third','Fourth'][plat_index]} qualifying virtual world platform identified with complete verification",
        parent=parent_node,
        critical=False
    )

    # -------------------- Identity -------------------- #
    identity_node = evaluator.add_parallel(
        id=f"Platform_{plat_index + 1}_Identity",
        desc="Platform name is provided and is distinct from other identified platforms",
        parent=platform_node,
        critical=True
    )

    # Name provided
    evaluator.add_custom_node(
        result=bool(plat.name and plat.name.strip()),
        id=f"Platform_{plat_index + 1}_Platform_Name_Provided",
        desc="A specific platform name is provided",
        parent=identity_node,
        critical=True
    )

    # Distinctness among four
    evaluator.add_custom_node(
        result=is_unique_among(plat.name, all_names),
        id=f"Platform_{plat_index + 1}_Platform_Distinctness",
        desc="Platform is different from the other three platforms identified",
        parent=identity_node,
        critical=True
    )

    # Operational status as of Feb 2026
    op_status_leaf = evaluator.add_leaf(
        id=f"Platform_{plat_index + 1}_Platform_Operational_Status",
        desc="Platform is actively maintained and operational as of February 2026",
        parent=identity_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"As of February 2026, the platform '{plat.name or 'UNKNOWN'}' is actively maintained and operational.",
        node=op_status_leaf,
        sources=plat.reference_urls,
        additional_instruction="Check for evidence such as current downloads, recent updates/release notes, active support pages, or operational service notices. If the reference pages clearly indicate the service is running and maintained around 2026, consider it supported."
    )

    # -------------------- System Requirements -------------------- #
    sysreq_node = evaluator.add_parallel(
        id=f"Platform_{plat_index + 1}_System_Requirements_Compliance",
        desc="Platform meets all minimum system requirement constraints",
        parent=platform_node,
        critical=True
    )

    # OS support group
    os_group = evaluator.add_parallel(
        id=f"Platform_{plat_index + 1}_Operating_System_Support",
        desc="Platform supports Windows 10 or Windows 11",
        parent=sysreq_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(plat.os_supported and plat.os_supported.strip()),
        id=f"Platform_{plat_index + 1}_OS_Specification_Provided",
        desc="Operating system compatibility information is provided",
        parent=os_group,
        critical=True
    )

    os_meets_leaf = evaluator.add_leaf(
        id=f"Platform_{plat_index + 1}_OS_Meets_Requirement",
        desc="Specified operating system includes Windows 10 or Windows 11",
        parent=os_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official documentation indicates that '{plat.name or 'UNKNOWN'}' supports Windows 10 or Windows 11.",
        node=os_meets_leaf,
        sources=plat.reference_urls,
        additional_instruction="Verify OS compatibility info; acceptable evidence includes system requirements pages or download pages stating Windows 10 or Windows 11 support."
    )

    # RAM requirement group
    ram_group = evaluator.add_parallel(
        id=f"Platform_{plat_index + 1}_RAM_Requirement",
        desc="Minimum RAM requirement is 8GB or less",
        parent=sysreq_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(plat.min_ram and plat.min_ram.strip()),
        id=f"Platform_{plat_index + 1}_RAM_Specification_Provided",
        desc="Specific minimum RAM requirement is stated",
        parent=ram_group,
        critical=True
    )

    ram_meets_leaf = evaluator.add_leaf(
        id=f"Platform_{plat_index + 1}_RAM_Meets_Threshold",
        desc="Stated RAM requirement is 8GB or less",
        parent=ram_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The minimum RAM requirement for '{plat.name or 'UNKNOWN'}' is 8GB or less (e.g., {plat.min_ram or 'unknown'}).",
        node=ram_meets_leaf,
        sources=plat.reference_urls,
        additional_instruction="Confirm the minimum RAM is no more than 8 GB. If the page lists 4 GB, 6 GB, or 8 GB minimum, it meets the threshold. If it lists higher than 8 GB minimum, it does not."
    )

    # GPU requirement group
    gpu_group = evaluator.add_parallel(
        id=f"Platform_{plat_index + 1}_GPU_Requirement",
        desc="Minimum GPU requirement is NVIDIA GTX 970 / AMD Radeon R9 290 equivalent or lower specification",
        parent=sysreq_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(plat.min_gpu and plat.min_gpu.strip()),
        id=f"Platform_{plat_index + 1}_GPU_Specification_Provided",
        desc="Specific minimum GPU requirement is stated",
        parent=gpu_group,
        critical=True
    )

    gpu_meets_leaf = evaluator.add_leaf(
        id=f"Platform_{plat_index + 1}_GPU_Meets_Threshold",
        desc="Stated GPU requirement is equivalent to or lower specification than GTX 970 / Radeon R9 290",
        parent=gpu_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The minimum GPU requirement for '{plat.name or 'UNKNOWN'}' is equivalent to or lower than NVIDIA GTX 970 or AMD Radeon R9 290 (e.g., {plat.min_gpu or 'unknown'}).",
        node=gpu_meets_leaf,
        sources=plat.reference_urls,
        additional_instruction="Treat GPUs equal to GTX 970 / R9 290 or less-powerful (e.g., GTX 960, GTX 750 Ti, Intel integrated that is sufficient) as meeting the threshold. If the minimum requires GPUs stronger than GTX 970 / R9 290, it does not meet the threshold."
    )

    # CPU requirement group
    cpu_group = evaluator.add_parallel(
        id=f"Platform_{plat_index + 1}_CPU_Requirement",
        desc="Minimum CPU requirement is Intel i5-4590 / AMD FX 8350 equivalent or lower specification",
        parent=sysreq_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(plat.min_cpu and plat.min_cpu.strip()),
        id=f"Platform_{plat_index + 1}_CPU_Specification_Provided",
        desc="Specific minimum CPU requirement is stated",
        parent=cpu_group,
        critical=True
    )

    cpu_meets_leaf = evaluator.add_leaf(
        id=f"Platform_{plat_index + 1}_CPU_Meets_Threshold",
        desc="Stated CPU requirement is equivalent to or lower specification than i5-4590 / FX 8350",
        parent=cpu_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The minimum CPU requirement for '{plat.name or 'UNKNOWN'}' is Intel i5-4590 / AMD FX 8350 equivalent or lower (e.g., {plat.min_cpu or 'unknown'}).",
        node=cpu_meets_leaf,
        sources=plat.reference_urls,
        additional_instruction="Accept i5-4590 or FX 8350 or any equal/older/lower-tier CPU as meeting the threshold. If the minimum requires a significantly newer or higher-tier CPU than these, it does not."
    )

    # -------------------- Accessibility -------------------- #
    access_node = evaluator.add_parallel(
        id=f"Platform_{plat_index + 1}_Accessibility_Features",
        desc="Platform meets accessibility and access requirements",
        parent=platform_node,
        critical=True
    )

    desktop_leaf = evaluator.add_leaf(
        id=f"Platform_{plat_index + 1}_Desktop_Mode_Available",
        desc="Platform supports desktop/PC mode without VR headset requirement",
        parent=access_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{plat.name or 'UNKNOWN'}' supports desktop/PC mode without requiring a VR headset.",
        node=desktop_leaf,
        sources=plat.reference_urls,
        additional_instruction="Look for mentions of 'Desktop mode', 'PC mode', 'No VR required', or similar phrasing in official docs or specs."
    )

    free_access_leaf = evaluator.add_leaf(
        id=f"Platform_{plat_index + 1}_Free_Access_Provision",
        desc="Platform offers free account creation and basic access without mandatory purchase",
        parent=access_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{plat.name or 'UNKNOWN'}' offers free account creation and basic platform access without mandatory purchase.",
        node=free_access_leaf,
        sources=plat.reference_urls,
        additional_instruction="Verify language such as 'free account', 'free to play', 'no purchase required to join', or similar in official pages."
    )

    # -------------------- Capacity & Communication -------------------- #
    capcomm_node = evaluator.add_parallel(
        id=f"Platform_{plat_index + 1}_Capacity_And_Communication",
        desc="Platform meets user capacity and communication feature requirements",
        parent=platform_node,
        critical=True
    )

    # Concurrent capacity group
    capacity_group = evaluator.add_parallel(
        id=f"Platform_{plat_index + 1}_Concurrent_User_Capacity",
        desc="Platform supports at least 40 concurrent users per instance/world/session",
        parent=capcomm_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(plat.concurrent_capacity and plat.concurrent_capacity.strip()),
        id=f"Platform_{plat_index + 1}_Capacity_Specification_Provided",
        desc="Specific concurrent user capacity information is provided",
        parent=capacity_group,
        critical=True
    )

    capacity_meets_leaf = evaluator.add_leaf(
        id=f"Platform_{plat_index + 1}_Capacity_Meets_Minimum",
        desc="Stated capacity is at least 40 concurrent users",
        parent=capacity_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{plat.name or 'UNKNOWN'}' supports at least 40 concurrent users per instance/world/session.",
        node=capacity_meets_leaf,
        sources=plat.reference_urls,
        additional_instruction="Check for capacity numbers; phrases like 'max users', 'concurrent users', or instance/world capacity. If ≥40, it meets the requirement."
    )

    voice_leaf = evaluator.add_leaf(
        id=f"Platform_{plat_index + 1}_Voice_Communication",
        desc="Platform includes built-in voice communication functionality",
        parent=capcomm_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{plat.name or 'UNKNOWN'}' includes built-in voice chat functionality.",
        node=voice_leaf,
        sources=plat.reference_urls,
        additional_instruction="Look for mentions of 'voice chat', 'voice communication', 'proximity voice', 'in-world voice', etc."
    )

    ugc_leaf = evaluator.add_leaf(
        id=f"Platform_{plat_index + 1}_User_Content_Support",
        desc="Platform supports user-created content, customizable avatars, or user-generated worlds",
        parent=capcomm_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{plat.name or 'UNKNOWN'}' supports user-created content, customizable avatars, or user-generated worlds/experiences.",
        node=ugc_leaf,
        sources=plat.reference_urls,
        additional_instruction="Accept evidence of UGC systems, avatar customization, world-building tools, or creator platforms documented officially."
    )

    # -------------------- Reference Documentation -------------------- #
    refs_node = evaluator.add_parallel(
        id=f"Platform_{plat_index + 1}_Reference_Documentation",
        desc="At least one reference URL from official platform documentation or reliable technical specifications source that verifies the requirements",
        parent=platform_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(plat.reference_urls),
        id=f"Platform_{plat_index + 1}_Reference_URL_Provided",
        desc="At least one valid reference URL is provided",
        parent=refs_node,
        critical=True
    )

    ref_quality_leaf = evaluator.add_leaf(
        id=f"Platform_{plat_index + 1}_Reference_URL_Quality",
        desc="Reference URL is from official platform documentation or reliable technical specifications source",
        parent=refs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"At least one provided reference for '{plat.name or 'UNKNOWN'}' is official documentation or a reliable technical specifications source.",
        node=ref_quality_leaf,
        sources=plat.reference_urls,
        additional_instruction="Consider official domains, docs/support/knowledge base pages, product manuals/spec pages, or reputable technical sources. Marketing pages alone are acceptable if they include explicit specs."
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
    Evaluate the answer for identifying four virtual world/social VR platforms meeting specific accessibility and system requirements.
    """

    # Initialize evaluator; root set to PARALLEL, non-critical to allow partial credit
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify four distinct virtual world or social VR platforms meeting all specified technical and accessibility requirements for community education use",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract platform data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_platforms(),
        template_class=PlatformsExtraction,
        extraction_name="platforms_extraction"
    )

    # Keep exactly four entries: first 4 if more, pad with empty records if fewer
    platforms: List[PlatformRecord] = list(extracted.platforms[:4])
    while len(platforms) < 4:
        platforms.append(PlatformRecord())

    # Build name list for distinctness checks
    all_names: List[Optional[str]] = [p.name for p in platforms]

    # Add a ground truth-style metadata entry for requirements (for context)
    evaluator.add_ground_truth({
        "requirements_summary": {
            "desktop_mode": "Required without VR headset",
            "os": "Windows 10 or Windows 11",
            "ram": "Minimum 8GB or less",
            "gpu": "Equal or lower than GTX 970 / Radeon R9 290",
            "cpu": "Equal or lower than Intel i5-4590 / AMD FX 8350",
            "capacity": "≥ 40 concurrent users per instance/world/session",
            "voice": "Built-in voice communication required",
            "status": "Operational and actively maintained as of Feb 2026",
            "ugc": "User-created content/avatars/worlds support required",
            "free_access": "Free account creation/basic access required"
        },
        "note": "These constraints are used to verify each platform against official or reliable technical documentation."
    })

    # Build and run verification for each platform (parallel node under root)
    verification_tasks = []
    for i in range(4):
        verification_tasks.append(
            verify_one_platform(evaluator, root, platforms[i], i, all_names)
        )
    await asyncio.gather(*verification_tasks)

    # Return the structured summary
    return evaluator.get_summary()